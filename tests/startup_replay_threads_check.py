#!/usr/bin/env python3
"""Two threads recording at once keep their order across threads, and x2w_replay plays them back in that order
(docs/startup-runtime.md#d3d12-replay-record).

startup_replay_threads has two threads draw and present in turns, each learning of the other only through a shared
fence's completed value. Present 2f + 1 is thread A's frame f, present 2f + 2 is thread B's. Checked:
  - recording: the calls of at least three threads (main and the two) have distinct sequence numbers; the Signal calls on
    the shared fence, in sequence order, carry the values 1, 2, 3 ...; every poll of the fence that returned v or more has
    a higher sequence number than the Signal of v (the recorded order across threads is the order they ran in);
  - replay (default, one OS thread per recorded thread): every picture equals the recorded run's (so present order is
    kept), no ordered step finished out of sequence order, none timed out, every call played;
  - replay with --threads 1 (one thread, sequence order): the same pictures;
  - control: replay with --sync off (the threads start last first and run unordered) shows ordered steps finished out of
    order and a changed picture sequence, so the checks above can fail.

usage: startup_replay_threads_check.py <startup_replay_threads> <x2w_replay> <warp|hardware>
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_replay_check import chunks, pictures
from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex

FRAMES = 6
PRESENTS = 2 * FRAMES
QUEUE_KIND, FENCE_KIND = 1, 2
QUEUE_SIGNAL, FENCE_COMPLETED = 14, 8
CALL = struct.Struct("<QIHHQ7QQdii")


def calls(path):
    """[(seq, thread, kind, slot, object, args, ret)] of every CALL chunk."""
    data = path.read_bytes()
    out = []
    for tag, header, _header_bytes, _payload, _size in chunks(data):
        if tag == b"CALL":
            seq, thread, kind, slot, obj, *rest = CALL.unpack_from(data, header)
            out.append((seq, thread, kind, slot, obj, rest[:7], rest[7]))
    return out


def check_recording(path):
    rows = sorted(calls(path))
    seqs = [row[0] for row in rows]
    assert len(set(seqs)) == len(seqs), "two calls share a sequence number"
    threads = {row[1] for row in rows}
    assert len(threads) >= 3, "the recording holds %d threads, expected main and two" % len(threads)
    signals = [(seq, args[1]) for seq, _t, kind, slot, _o, args, _r in rows if kind == QUEUE_KIND and slot == QUEUE_SIGNAL]
    values = [v for _s, v in signals]
    assert values == list(range(1, PRESENTS + 1)), "the signals on the shared fence are not in order: %s" % values
    signal_at = {v: seq for seq, v in signals}
    polls = 0
    for seq, _t, kind, slot, _o, _args, ret in rows:
        if kind == FENCE_KIND and slot == FENCE_COMPLETED and 0 < ret <= PRESENTS:
            assert seq > signal_at[ret], "a poll that saw %d is recorded before the signal of %d" % (ret, ret)
            polls += 1
    assert polls, "no poll of the shared fence saw a signal"
    return len(threads), len(rows), polls


def run(args, env):
    result = subprocess.run([str(a) for a in args], env=env, capture_output=True, text=True, timeout=300)
    return result.returncode, result.stdout + result.stderr


def replay(player, recording, out, backend, env, *flags):
    env = {k: v for k, v in env.items() if not k.startswith("X2W_")}
    code, text = run([player, recording, out, "--adapter", backend, *flags], env)
    log = (out / "stderr.txt").read_text(errors="replace")[-3000:] if (out / "stderr.txt").exists() else ""
    assert code == 0, "x2w_replay exit %d\n%s\n%s" % (code, text[-2000:], log)
    summary = json.loads((out / "replay_summary.json").read_text())
    assert summary["presents"] == PRESENTS and summary["missing_blob"] == 0, (summary, log)
    return summary, pictures(out)


def main(scene, player, backend):
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
        for name in ("X2W_REPLAY_RECORD", "X2W_REPLAY_RECORD_FILE", "X2W_PRESENT_PICTURES"):
            env.pop(name, None)
        code, text = run([scene, vs, ps, root, backend, 1], env)
        if code == 77 and backend == "hardware":
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        assert code == 0, text[-3000:]
        subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)

        recorded_dir, recording = directory / "recorded", directory / "run.x2wr"
        recorded_dir.mkdir()
        code, text = run([scene, vs, ps, root, backend, FRAMES],
                         dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                              X2W_PRESENT_PICTURES=str(recorded_dir)))
        assert code == 0, text[-3000:]
        recorded = pictures(recorded_dir)
        names = ["present_%04d.png" % n for n in range(1, PRESENTS + 1)]
        assert sorted(recorded) == names, sorted(recorded)
        assert recorded[names[0]] != recorded[names[1]], "the two threads draw the same picture"
        thread_count, call_count, polls = check_recording(recording)

        summary, replayed = replay(player, recording, directory / "replayed", backend, env)
        assert summary["threads"] == thread_count, summary
        assert summary["unknown_object"] == 0 and summary["fence_timeouts"] == 0 and summary["order_timeouts"] == 0, summary
        assert summary["sync_order_inversions"] == 0, summary
        assert summary["played"] == call_count, summary
        differ = [n for n in names if replayed[n] != recorded[n]]
        assert not differ, "threaded replay differs from the recorded run at %s" % differ

        _summary, single = replay(player, recording, directory / "single", backend, env, "--threads", "1")
        differ = [n for n in names if single[n] != recorded[n]]
        assert not differ, "single-thread replay differs from the recorded run at %s" % differ

        # Control: no order, threads started last first. The order the ordered steps finished in, and the pictures, change.
        free, unordered = replay(player, recording, directory / "unordered", backend, env, "--sync", "off")
        assert free["sync_order_inversions"] > 0, "the control kept the order: %s" % free
        assert any(unordered.get(n) != recorded[n] for n in names), "the control drew the same pictures in the same order"
        print("replay threads: %d threads, %d calls, %d polls recorded in order; %d presents equal on %d threads and on one; "
              "control (no order): %d steps out of order" % (thread_count, call_count, polls, PRESENTS, summary["threads"],
                                                             free["sync_order_inversions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
