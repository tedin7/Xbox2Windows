#!/usr/bin/env python3
"""X2W_REPLAY_MAX_MIB ends a replay recording at its size cap (docs/startup-runtime.md#d3d12-replay-record).

startup_replay_quad hits a 1 MiB file cap in the prefix of a future recording window.
The scene must finish, with no recorder work after XEND. A second capture uses a
reserved address, a committed page in the same granule, and oversized blobs to
check first-chance access exceptions and the per-call capture budget.
The "slow" window check exercises the first present after a pause.
Also printed, not checked (timing on a shared machine): the same 300 presents plain, with the window closed
(X2W_REPLAY_RECORD=f100000,1: every slot wrapped, nothing recorded) and recorded without a cap.

usage: startup_replay_cap_check.py <startup_replay_quad> <warp|hardware>
"""
import os
from collections import defaultdict
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time

from startup_replay_record_check import CALL, chunks
from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex

PRESENTS = 300
CAP_MIB = 1
SLACK = (1 << 20) + (2 << 20)  # one thread buffer flush + the shader cache files written before XEND


def end_totals(path):
    """XEND's totals. The chunk reader rejects truncated recordings."""
    last = None
    for last in chunks(path.read_bytes()):
        pass
    assert last and last[0] == "XEND", "the file does not end with XEND"
    return struct.unpack("<QQQQ", last[1])


def check_argument_memory(path):
    budget = 1 << 20
    calls, stream_calls, memory = [], [], defaultdict(list)
    for tag, header, payload in chunks(path.read_bytes()):
        if tag == "CALL":
            seq, _, kind, slot, _, *args = CALL.unpack(header)
            if (kind, slot, args[0]) == (0, 16, 0x11223344):
                calls.append(seq)
            if (kind, slot, args[3]) == (0, 86, 0x11223344):
                stream_calls.append(seq)
        elif tag == "ARGM":
            seq, arg, depth, _, length, _ = struct.unpack("<QIIQQQ", header)
            memory[seq].append((arg, depth, length))
        elif tag == "DARR":
            seq, = struct.unpack("<Q", header)
            memory[seq].append((None, 0, len(payload)))
    assert len(calls) == 24, calls
    assert all(sum(length for _, _, length in ranges) <= budget for ranges in memory.values())
    expected_blob_sizes = [[], [64], [], [64]] + [[] for _ in range(20)]
    actual_blob_sizes = [[length for arg, depth, length in memory[seq] if (arg, depth) == (1, 0)] for seq in calls]
    assert actual_blob_sizes == expected_blob_sizes, actual_blob_sizes
    assert all(arg in (1, 3) and depth == 0 for seq in calls for arg, depth, _ in memory[seq])
    [stream_call] = stream_calls
    assert [(arg, length) for arg, depth, length in memory[stream_call] if depth == 0 and arg >= 256] == [(512, 600 << 10)]
    size = path.stat().st_size
    assert size < budget, "argument capture fixture wrote %d bytes" % size
    print("replay argument memory: 25 calls, zero access exceptions, %d bytes" % size)


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
                     "X2W_PRESENT_PICTURES", "X2W_TEST_REPLAY_RECORDER", "X2W_TEST_INPUT_LAYOUT", "X2W_REPLAY_PAGES",
                     "X2W_REPLAY_SHADERS", "X2W_REPLAY_PREFIX_MAX_MIB"):
            env.pop(name, None)
        args = [scene, vs, ps, root, backend]
        code, text, _ = timed(args + [1], env)
        if code == 77 and backend == "hardware":
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        assert code == 0, text[-3000:]
        subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)

        memory = directory / "memory.x2wr"
        code, text, _ = timed(args + [1], dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(memory),
                                              X2W_TEST_REPLAY_RECORDER="memory"))
        assert code == 0, text[-3000:]
        assert "replay quad: recorder access exceptions 0" in text, text[-3000:]
        end_totals(memory)
        check_argument_memory(memory)
        # Recordings are mailed: no folder of this machine in the file (the shader cache and the run's temp folder are).
        settings = [payload for tag, _, payload in chunks(memory.read_bytes()) if tag == "XENV"]
        assert settings and b"X2W_D3D12_RENDER=1" in settings[0], settings
        assert b":\\" not in settings[0] and b"X2W_SHADER_CACHE" not in settings[0], settings[0]
        assert str(directory).encode() not in memory.read_bytes(), "the recording holds the run's folder"

        capped = directory / "capped.x2wr"
        code, text, _ = timed(args + [PRESENTS], dict(env, X2W_REPLAY_RECORD="f100000,1", X2W_REPLAY_RECORD_FILE=str(capped),
                                                      X2W_REPLAY_MAX_MIB=str(CAP_MIB), X2W_TEST_REPLAY_RECORDER="cap"))
        assert code == 0, "the scene did not finish under a recording cap:\n" + text[-3000:]
        assert "replay quad: %d presents drawn" % PRESENTS in text, text[-2000:]
        assert "X2W_REPLAY_MAX_MIB reached" in text, text[-3000:]
        calls, _, _, presents = end_totals(capped)
        assert presents == 0, "cap should stop the recording in the prefix"
        # tools/startup/tester_pack.py reports such a file unusable by this line.
        assert "the cap stopped the recording BEFORE the window opened: prefix only" in text, text[-3000:]
        prefix = "replay quad: recorder calls after scene "
        [after] = [int(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]
        assert after == calls, "recorder processed %d calls after the file was closed" % (after - calls)
        print("replay prefix cap: %d calls at XEND, zero recorder calls afterward" % calls)
        size = capped.stat().st_size
        assert size <= (CAP_MIB << 20) + SLACK, "capped file is %d bytes" % size

        # X2W_REPLAY_SHADERS=0: no SHDR chunk and no shader cache line; without it the cache files are embedded.
        with_shaders = directory / "with_shaders.x2wr"
        code, text, _ = timed(args + [12], dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(with_shaders)))
        assert code == 0, text[-2000:]
        assert b"SHDR" in with_shaders.read_bytes(), "the control recording holds no shader cache"
        bare = directory / "bare.x2wr"
        code, text, _ = timed(args + [12], dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(bare),
                                               X2W_REPLAY_SHADERS="0"))
        assert code == 0, text[-2000:]
        assert b"SHDR" not in bare.read_bytes(), "X2W_REPLAY_SHADERS=0 still wrote shader chunks"
        end_totals(bare)

        # X2W_REPLAY_PREFIX_MAX_MIB: the prefix has its own cap and the window cap counts from the window's start, so the
        # same prefix that the 1 MiB cap above stops before the window now lets the window open and record its presents.
        windowed = directory / "windowed.x2wr"
        code, text, _ = timed(args + [PRESENTS], dict(env, X2W_REPLAY_RECORD="f5,3", X2W_REPLAY_RECORD_FILE=str(windowed),
                                                      X2W_REPLAY_MAX_MIB=str(CAP_MIB), X2W_REPLAY_PREFIX_MAX_MIB="64",
                                                      X2W_TEST_REPLAY_RECORDER="cap", X2W_REPLAY_SHADERS="0"))
        assert code == 0, text[-3000:]
        assert "BEFORE the window opened" not in text, "the prefix hit a cap that is not its own:\n" + text[-3000:]
        _, _, _, window_presents = end_totals(windowed)
        assert window_presents > 0, "the window recorded no present"

        # "slow": the window opens at the first present after a gap of 300 ms or more, lasts 0.15 s, and the file ends with
        # its XEND. The 400 ms pause before frame 15 makes present 16 such a present; a loaded machine can stall 300 ms
        # earlier on its own, so the check reads the gap the recorder measured instead of expecting present 16 exactly.
        slow = directory / "slow.x2wr"
        code, text, _ = timed(args + [30], dict(env, X2W_REPLAY_RECORD="slow,0,300,0.15,60", X2W_REPLAY_RECORD_FILE=str(slow),
                                               X2W_TEST_REPLAY_PAUSE="15,400"))
        assert code == 0, text[-3000:]
        opened = [line for line in text.splitlines() if "x2w-replay record: window opens at" in line]
        trigger = re.search(r"present (\d+) \(first slowdown, gap (\d+) ms\)", opened[0]) if len(opened) == 1 else None
        assert trigger and int(trigger.group(1)) <= 16 and int(trigger.group(2)) >= 300, opened or text[-3000:]
        end_totals(slow)

        # The prefix stays compact: the scene's 16000 device AddRef/Release calls take no argument, so their stack words
        # are not captured as memory.
        refcount, argument_seqs = set(), set()
        for tag, header, _ in chunks(windowed.read_bytes()):
            if tag == "CALL":
                seq, _, kind, slot, _, *_ = CALL.unpack(header)
                if kind == 0 and slot in (1, 2):
                    refcount.add(seq)
            elif tag == "ARGM":
                argument_seqs.add(struct.unpack_from("<Q", header)[0])
        assert len(refcount) >= 16000 and not refcount & argument_seqs, (len(refcount), len(refcount & argument_seqs))

        timings = {}
        for name, extra in (("plain", {}), ("closed", {"X2W_REPLAY_RECORD": "f100000,1"}),
                            ("recorded", {"X2W_REPLAY_RECORD": "all"})):
            extra = dict(extra, X2W_REPLAY_RECORD_FILE=str(directory / ("%s.x2wr" % name)))
            code, text, seconds = timed(args + [PRESENTS], dict(env, **extra))
            assert code == 0, "%s run failed:\n%s" % (name, text[-2000:])
            timings[name] = seconds
        full = (directory / "recorded.x2wr").stat().st_size
        print("replay cap: %d MiB cap -> %d bytes, %d of %d presents, XEND last; cost for %d presents: plain %.2f s, "
              "window closed %.2f s, recorded %.2f s (%d bytes, %.1f KiB per present)"
              % (CAP_MIB, size, presents, PRESENTS, PRESENTS, timings["plain"], timings["closed"], timings["recorded"],
                 full, full / 1024.0 / PRESENTS))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
