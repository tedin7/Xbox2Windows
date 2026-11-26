#!/usr/bin/env python3
"""A scene that creates resources, placed resources or heaps through the stand-in replays to the same creations
(docs/startup-runtime.md#d3d12-replay-record, family: resource creation, placed resources, heaps).

The scene is an existing in-repo fixture, run unchanged. Checked:
  - recorded run: X2W_REPLAY_RECORD=all writes a recording and the scene still exits with the code of a plain run;
  - replay: x2w_replay plays it with no X2W_ setting, exits 0, every call lands on an object it knows (unknown_object 0) and no
    blob is missing (missing_blob 0);
  - same creations: the stand-in's creation log lines (Device::Create*, heap and placement lines), with addresses and
    counters blanked, are the same list in the recorded run and in the replay (a creation the player dropped or changed shows);
  - control (--control): one byte of the first creation call's description memory is changed in the recording; the replay must
    then NOT log the same creations.

usage: startup_replay_creations_check.py <x2w_replay> <warp|hardware> [--control] -- <scene exe> [scene args]
"""
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

CREATION = re.compile(r"x2w-d3d12 (Device::(Create|Open|Place|slot(2[7-9]|3\d|7\d|8\d|9\d|1[0-4]\d)\b)|Heap|Resource::).*")
HEX = re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-fA-F]{8,16}\b")


def run(args, env, timeout=300):
    result = subprocess.run([str(a) for a in args], env=env, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def creations(text, raw=False):
    """The stand-in's creation lines, addresses and pointers blanked (kept with raw=True), in order."""
    lines = []
    for line in text.splitlines():
        if line.startswith("x2w-d3d12 ") and CREATION.match(line) and re.search(r"create|place|heap", line, re.I):
            lines.append(line if raw else HEX.sub("#", line))
    return lines


def chunks(data):
    at = 0
    while at < len(data):
        tag, header_bytes, data_bytes = struct.unpack_from("<4sIQ", data, at)
        yield tag, at + 16, header_bytes, at + 16 + header_bytes, data_bytes
        at += 16 + header_bytes + data_bytes


def damage_first_creation(source, target):
    """Copies the recording with one byte of the first Create* call's argument memory flipped (a blob's first byte)."""
    data = bytearray(source.read_bytes())
    blobs, first = {}, None
    for tag, header, _hb, payload, size in chunks(data):
        if tag == b"ARGM" and first is None:
            seq, arg, _depth, _address, nbytes, blob_hash = struct.unpack_from("<QIIQQQ", data, header)
            if nbytes >= 8:
                first = blob_hash
        elif tag == b"BLOB":
            blobs[struct.unpack_from("<Q", data, header)[0]] = (payload, size)
    if first is None or first not in blobs:
        return False
    at, size = blobs[first]
    data[at + min(size - 1, 12)] ^= 0xFF
    target.write_bytes(bytes(data))
    return True


def main(argv):
    control = "--control" in argv
    argv = [a for a in argv if a != "--control"]
    split = argv.index("--")
    player, backend = argv[0], argv[1]
    scene = argv[split + 1:]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("X2W_REPLAY", "X2W_PRESENT_PICTURES"))}
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        recording = directory / "scene.x2wr"
        plain_code, _out, _err = run(scene, env)
        if plain_code == 77:
            print("scene skipped (77)")
            return 77
        code, out, err = run(scene, dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording)))
        assert code == plain_code, "recorded run exits %d, a plain run %d\n%s" % (code, plain_code, (out + err)[-2000:])
        assert recording.exists() and recording.stat().st_size > 0, "no recording written"
        recorded = creations(out + err)
        assert recorded, "the scene logged no creation line (is X2W_D3D12 logging on?)\n" + (out + err)[-1500:]

        def play(path, name):
            target = directory / name
            target.mkdir()
            player_env = {k: v for k, v in env.items() if not k.startswith("X2W_")}
            player_env["X2W_REPLAY_TRACE"] = "1"
            code, out, err = run([player, path, target, "--adapter", backend], player_env)
            log = (target / "stderr.txt").read_text(errors="replace") if (target / "stderr.txt").exists() else ""
            return code, out + err, log, target

        code, text, log, target = play(recording, "replayed")
        assert code == 0, "x2w_replay exit %d\n%s\n%s" % (code, text[-1500:], log[-1500:])
        summary = json.loads((target / "replay_summary.json").read_text())
        assert summary["unknown_object"] == 0, "calls on unknown objects: %s" % summary
        assert summary["missing_blob"] == 0, "missing blobs: %s" % summary
        replayed = creations(log)
        if replayed != recorded:
            # First positions where the two ordered lists differ, with the raw lines (addresses kept).
            raw_r, raw_p = creations(out + err, raw=True), creations(log, raw=True)
            at = [i for i in range(min(len(recorded), len(replayed))) if recorded[i] != replayed[i]][:4]
            only_recorded = ["#%d %s" % (i, raw_r[i]) for i in at]
            only_replayed = ["#%d %s" % (i, raw_p[i]) for i in at]
            raise AssertionError("creation lines differ: recorded %d, replayed %d\nonly recorded:\n%s\nonly replayed:\n%s" % (
                len(recorded), len(replayed), "\n".join(only_recorded), "\n".join(only_replayed)))
        if control:
            damaged = directory / "damaged.x2wr"
            assert damage_first_creation(recording, damaged), "no argument memory to damage"
            code, text, log, _t = play(damaged, "damaged")
            assert code != 0 or creations(log) != recorded, "a damaged creation replayed to the same creations"
        print("replay creations: %d creation lines equal; calls %s" % (len(recorded), summary.get("calls")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
