#!/usr/bin/env python3
"""A scene of the texture / sampler / image family replays from its recording (docs/startup-runtime.md#d3d12-replay-record).

The scene is any in-repo check that drives the D3D12 stand-in and exits 0 (a texture copy, a view table, a UAV clear, ...).
Checked, for one scene command:
  - the plain run and the recorded run (X2W_REPLAY_RECORD=all) end with the same exit code, and recording adds no
    failure to a scene that passes;
  - the recording holds calls (a scene that never reaches the stand-in leaves no file: that is a gap, not a pass);
  - x2w_replay plays it: exit 0, no call on an unknown object, no missing blob, no fence timeout, every recorded call
    played;
  - when the scene presents (X2W_PRESENT_PICTURES), every replayed picture equals the recorded run's picture;

usage: startup_replay_family_check.py <x2w_replay> <warp|hardware> -- <scene exe> [scene args...]
Prints one line: `replay family <scene>: calls N played N pictures K equal`.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from startup_replay_check import CALL, chunks, pictures, run


def scrub(env):
    env = dict(env)
    for name in ("X2W_REPLAY_RECORD", "X2W_REPLAY_RECORD_FILE", "X2W_REPLAY_DIR", "X2W_PRESENT_PICTURES"):
        env.pop(name, None)
    return env


def call_list(recording, limit=60):
    """The first recorded calls as 'seq kind.slot arg0 arg1 arg2' lines, for a failure message (what the player met)."""
    out = []
    data = recording.read_bytes()
    for tag, header, _hb, _payload, _size in chunks(data):
        if tag != b"CALL":
            continue
        seq, _thread, kind, slot, obj, *rest = CALL.unpack_from(data, header)
        out.append("%d k%d.s%d obj %x a %s ret %x" % (seq, kind, slot, obj, " ".join("%x" % a for a in rest[:4]), rest[7]))
        if len(out) >= limit:
            break
    return "\n".join(out)


def phase(text):
    print('[%.1f s] %s' % (time.monotonic() - START, text), flush=True)


START = time.monotonic()


def main(argv):
    if "--" not in argv or argv.index("--") < 2:
        print(__doc__)
        return 2
    split = argv.index("--")
    player, backend = argv[0], argv[1]
    scene = argv[split + 1:]
    if not scene:
        return 2
    name = Path(scene[0]).stem + "".join("_" + a for a in scene[1:])
    env = scrub(os.environ)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        phase("plain run")
        code, text = run(scene, env)
        if code == 77:
            print("scene skipped (77):", name)
            return 77
        assert code == 0, "the scene fails on its own, not a replay question:\n" + text[-2000:]

        phase("recorded run")
        recorded_dir, recording = directory / "recorded", directory / "run.x2wr"
        recorded_dir.mkdir()
        recorded_code, recorded_text = run(scene, dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                                                       X2W_PRESENT_PICTURES=str(recorded_dir)))
        assert recorded_code == 0, "recording changed the scene's result (exit %d):\n%s" % (recorded_code, recorded_text[-2000:])
        assert recording.exists(), "GAP no recording: the scene never reaches the stand-in (%s)" % name
        calls = sum(1 for tag, *_ in chunks(recording.read_bytes()) if tag == b"CALL")
        assert calls, "GAP the recording of %s holds no call" % name
        recorded = pictures(recorded_dir)

        def replay(path, out):
            out.mkdir()
            code, text = run([player, path, out, "--adapter", backend], env)
            summary_path = out / "replay_summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            return code, text, summary, pictures(out)

        phase("replay")
        code, text, summary, replayed = replay(recording, directory / "replayed")
        log = directory / "replayed" / "stderr.txt"
        tail = log.read_text(errors="replace")[-2500:] if log.exists() else ""
        assert code == 0, "GAP x2w_replay exit %d for %s\n%s\n%s\nrecorded calls:\n%s" % (
            code, name, text[-800:], tail[-1200:], call_list(recording))
        assert summary.get("unknown_object") == 0 and summary.get("missing_blob") == 0 and summary.get("fence_timeouts") == 0, \
            "GAP %s: %s\n%s" % (name, summary, tail)
        assert summary.get("calls") == calls and summary.get("played") == calls, \
            "GAP %s: %d calls recorded, %s played (%s)\n%s" % (name, calls, summary.get("played"), summary, tail)
        assert sorted(replayed) == sorted(recorded), "GAP %s pictures %s recorded, %s replayed" % (
            name, sorted(recorded), sorted(replayed))
        differ = [n for n in sorted(recorded) if replayed[n] != recorded[n]]
        assert not differ, "GAP %s: replayed pictures differ from the recorded run: %s\n%s" % (name, differ, tail)

        print("replay family %s: calls %d played %d pictures %d equal" % (name, calls, summary["played"], len(recorded)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
