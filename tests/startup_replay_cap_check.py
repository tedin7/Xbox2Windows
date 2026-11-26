#!/usr/bin/env python3
"""X2W_REPLAY_MAX_MIB ends a replay recording at its size cap (docs/startup-runtime.md#d3d12-replay-record).

startup_replay_quad draws 300 presents with X2W_REPLAY_RECORD=all and a 1 MiB cap. Checked:
  - the scene still runs to the end (recording never stops the title);
  - the file is well formed and ends with its XEND, holding fewer presents than were drawn;
  - opt-in XEND counters match the capped file and the known call counts of a four-frame run;
  - the file stays near the cap (cap + one thread buffer + the shader cache files);
  - the log says the cap was reached;
  - "slow" opens its window at the first present after a pause, and that file ends with its XEND too.
Also printed, not checked (timing on a shared machine): the same 300 presents plain, with the window closed
(X2W_REPLAY_RECORD=f100000,1: every slot wrapped, nothing recorded) and recorded without a cap.

usage: startup_replay_cap_check.py <startup_replay_quad> <warp|hardware>
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter

from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex
from startup_replay_record_check import CALL, chunks

PRESENTS = 300
CAP_MIB = 1
SLACK = (1 << 20) + (2 << 20)  # one thread buffer flush + the shader cache files written before XEND


COUNTER_PREFIX = "x2w-replay record: counts"


def check_end(path, log, counts_enabled):
    """Check XEND against the chunks in the file and return its present count and slot totals."""
    slots, sizes = Counter(), Counter()
    last = None
    for tag, header, payload in chunks(path.read_bytes()):
        sizes[tag.rstrip()] += 16 + len(header) + len(payload)
        if tag == "CALL":
            _, _, kind, slot, *_ = CALL.unpack(header)
            slots[kind, slot] += 1
        last = tag, header, payload
    assert last and last[0] == "XEND", "the file does not end with XEND"
    calls, _, preceding_bytes, presents = struct.unpack("<QQQQ", last[1])
    lines = [line for line in log.splitlines() if line.startswith(COUNTER_PREFIX)]
    if counts_enabled:
        line = last[2].decode("ascii")
        assert line.startswith(COUNTER_PREFIX + " "), "missing counters in XEND"
        assert lines == [line], "expected one counter line matching XEND"
        tokens = line[len(COUNTER_PREFIX):].split()
        actual = {key: int(value) for key, value in (token.split("=") for token in tokens)}
        expected = {f"call.{kind}.{slot}": count for (kind, slot), count in slots.items()}
        expected.update({f"bytes.{tag}": size for tag, size in sizes.items()})
        assert len(actual) == len(tokens) and actual == expected, (actual, expected)
        assert calls == sum(slots.values())
        assert preceding_bytes == sum(sizes.values()) - sizes["XEND"]
    else:
        assert not last[2] and not lines, "counters written with the switch unset"
    return presents, slots


def timed(args, env):
    started = time.perf_counter()
    result = subprocess.run([str(a) for a in args], env=env, capture_output=True, text=True, timeout=600)
    return result.returncode, result.stdout + result.stderr, time.perf_counter() - started


def main(scene, backend):
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        cache = directory / "cache"
        cache.mkdir()
        vs, ps, root = (directory / name for name in ("vs", "ps", "root"))
        vs.write_bytes(vertex())
        ps.write_bytes(pixel(True))
        env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
                   X2W_SHADER_CACHE=str(cache))
        env.setdefault("X2W_D3D12_DEBUG", "0")
        for name in ("X2W_REPLAY_RECORD", "X2W_REPLAY_RECORD_FILE", "X2W_REPLAY_DIR", "X2W_REPLAY_MAX_MIB",
                     "X2W_REPLAY_COUNTS", "X2W_PRESENT_PICTURES"):
            env.pop(name, None)
        args = [scene, vs, ps, root, backend]
        code, text, _ = timed(args + [1], env)
        if code == 77 and backend == "hardware":
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        assert code == 0, text[-3000:]
        subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)

        capped = directory / "capped.x2wr"
        code, text, _ = timed(args + [PRESENTS], dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(capped),
                                                      X2W_REPLAY_MAX_MIB=str(CAP_MIB), X2W_REPLAY_COUNTS="1"))
        assert code == 0, "the scene did not finish under a recording cap:\n" + text[-3000:]
        assert "replay quad: %d presents drawn" % PRESENTS in text, text[-2000:]
        assert "X2W_REPLAY_MAX_MIB reached" in text, text[-3000:]
        presents, slots = check_end(capped, text, counts_enabled=True)
        assert presents < PRESENTS, "the capped file holds every present (%d): the cap did not stop it" % presents
        # Each present follows one draw. The cap can also land after the draw in the next frame.
        assert slots[1, 32] == presents, (slots[1, 32], presents)
        assert slots[4, 12] in (presents, presents + 1), (slots[4, 12], presents)
        size = capped.stat().st_size
        assert size <= (CAP_MIB << 20) + SLACK, "capped file is %d bytes" % size

        # "slow": the window opens at the first present after a gap of 300 ms or more (a 400 ms pause before frame 15:
        # present 16), lasts 0.15 s, and the file ends with its XEND.
        slow = directory / "slow.x2wr"
        code, text, _ = timed(args + [30], dict(env, X2W_REPLAY_RECORD="slow,0,300,0.15,60", X2W_REPLAY_RECORD_FILE=str(slow),
                                               X2W_TEST_REPLAY_PAUSE="15,400"))
        assert code == 0, text[-3000:]
        opened = [line for line in text.splitlines() if "x2w-replay record: window opens at" in line]
        assert len(opened) == 1 and "present 16 (first slowdown" in opened[0], opened or text[-3000:]
        check_end(slow, text, counts_enabled=False)

        timings = {}
        for name, extra in (("plain", {}), ("closed", {"X2W_REPLAY_RECORD": "f100000,1"}),
                            ("recorded", {"X2W_REPLAY_RECORD": "all"})):
            extra = dict(extra, X2W_REPLAY_RECORD_FILE=str(directory / ("%s.x2wr" % name)))
            code, text, seconds = timed(args + [PRESENTS], dict(env, **extra))
            assert code == 0, "%s run failed:\n%s" % (name, text[-2000:])
            timings[name] = seconds
        full = (directory / "recorded.x2wr").stat().st_size

        # Exercise process-exit finalization with an independently known scene size.
        counted = directory / "counted.x2wr"
        frames = 4
        code, text, _ = timed(args + [frames], dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(counted),
                                                    X2W_REPLAY_COUNTS="1"))
        assert code == 0, text[-3000:]
        end, slots = check_end(counted, text, counts_enabled=True)
        assert end == frames
        for kind_slot in ((1, 10), (1, 14), (1, 32), (4, 12), (4, 52), (4, 139)):
            assert slots[kind_slot] == frames, (kind_slot, slots[kind_slot], frames)
        print("replay cap: %d MiB cap -> %d bytes, %d of %d presents, XEND last; cost for %d presents: plain %.2f s, "
              "window closed %.2f s, recorded %.2f s (%d bytes, %.1f KiB per present)"
              % (CAP_MIB, size, presents, PRESENTS, PRESENTS, timings["plain"], timings["closed"], timings["recorded"],
                 full, full / 1024.0 / PRESENTS))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
