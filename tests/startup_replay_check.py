#!/usr/bin/env python3
"""x2w_replay plays a recording made with X2W_REPLAY_RECORD=all and draws the same pictures
(docs/startup-runtime.md#d3d12-replay-record).

startup_replay_quad draws an animated scene only through the stand-in and presents it every frame. Checked:
  - recorded run: X2W_PRESENT_PICTURES writes present_<nnnn>.png per present, and the pictures change over time;
  - replay: with no X2W_ setting (the stand-in's settings and shader cache travel in the recording), x2w_replay
    writes the same picture names, every one equal to the recorded run's, every call played on an
    object it knows, and one "x2w-replay score" line;
  - a window in the middle (f6,6: frames 7-12) replays to the same pictures, every object known
    (the state-making calls before the window are recorded too). A prefix-only ranged descriptor copy
    supplies the render target. Removing the snapshot marker leaves the fallback clear visible;
  - a target made through device slot 127 (ten argument words) replays equal (format 3 records eleven);
  - control: the same recording with the last frame's clear colour changed in its memory replays to a different last
    picture, and the other pictures stay equal (the replay draws from the recording, and the comparison sees it).

usage: startup_replay_check.py <startup_replay_quad> <x2w_replay> <warp|hardware>
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex

PRESENTS = 12
LIST_KIND, CLEAR_SLOT = 4, 52
CALL = struct.Struct("<QIHHQ11QQdii")


def pixels(path):
    """(width, height, raw scanlines) of a PNG written by the stand-in."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    at, idat, size = 8, b"", None
    while at < len(data):
        length, tag = struct.unpack_from(">I4s", data, at)
        body = data[at + 8:at + 8 + length]
        if tag == b"IHDR":
            size = struct.unpack_from(">II", body)
        elif tag == b"IDAT":
            idat += body
        at += 12 + length
    return size, zlib.decompress(idat)


def pictures(folder):
    return {p.name: pixels(p) for p in sorted(Path(folder).glob("present_*.png"))}


def explain(expected, actual, log):
    """What differs between two pictures (size, colours on the first row), and the replay log's tail."""
    def row(picture):
        (width, _height), raw = picture
        return [tuple(raw[1 + x * 4:5 + x * 4]) for x in range(0, width, max(1, width // 8))]
    text = log.read_text(errors="replace") if log.exists() else "(no replay log)"
    shader = [line for line in text.splitlines() if any(w in line for w in ("translat", "manifest", "shader", "not drawn"))]
    return "recorded %s row %s\nreplayed %s row %s\nreplay log, shader lines:\n%s\nreplay log tail:\n%s" % (
        expected[0], row(expected), actual[0], row(actual), "\n".join(shader[:25]), text[-1500:])


def explain_log(name, log):
    text = log.read_text(errors="replace") if log.exists() else ""
    shader = [line for line in text.splitlines() if any(w in line for w in ("translat", "manifest", "shader", "not drawn"))]
    return "%s, shader lines:\n%s" % (name, "\n".join(shader[:25]))


def chunks(data):
    at = 0
    while at < len(data):
        tag, header_bytes, data_bytes = struct.unpack_from("<4sIQ", data, at)
        yield tag, at + 16, header_bytes, at + 16 + header_bytes, data_bytes
        at += 16 + header_bytes + data_bytes


def change_last_clear(source, target):
    """Copies the recording with the colour of the last ClearRenderTargetView changed (its blob's first float)."""
    data = bytearray(source.read_bytes())
    clears, argm, blobs = [], {}, {}
    for tag, header, _hb, payload, size in chunks(data):
        if tag == b"CALL":
            seq, _t, kind, slot = struct.unpack_from("<QIHH", data, header)
            if kind == LIST_KIND and slot == CLEAR_SLOT:
                clears.append(seq)
        elif tag == b"ARGM":
            seq, arg, _depth, _address, _bytes, blob_hash = struct.unpack_from("<QIIQQQ", data, header)
            argm[(seq, arg)] = blob_hash
        elif tag == b"BLOB":
            blobs[struct.unpack_from("<Q", data, header)[0]] = payload
    assert len(clears) == PRESENTS, clears
    at = blobs[argm[(clears[-1], 1)]]
    struct.pack_into("<f", data, at, 1.0 - struct.unpack_from("<f", data, at)[0])
    target.write_bytes(bytes(data))


def run(args, env):
    result = subprocess.run([str(a) for a in args], env=env, capture_output=True, text=True, timeout=300)
    return result.returncode, result.stdout + result.stderr


def replay(player, recording, out, backend, env, presents=PRESENTS):
    # No X2W_ setting: the stand-in's settings (XENV) and its shader cache (SHDR) come from the recording.
    env = {k: v for k, v in env.items() if not k.startswith("X2W_")}
    code, text = run([player, recording, out, "--adapter", backend], env)
    if code:
        log = (out / "stderr.txt").read_text(errors="replace")[-3000:] if (out / "stderr.txt").exists() else ""
        raise AssertionError("x2w_replay exit %d on %s (%d bytes)\n%s\n%s" % (code, recording.name, recording.stat().st_size,
                                                                        text[-2000:], log))
    score = [line for line in text.splitlines() if line.startswith("x2w-replay score ")]
    assert len(score) == 1, text
    summary = json.loads((out / "replay_summary.json").read_text())
    assert summary["unknown_object"] == 0 and summary["missing_blob"] == 0, summary
    assert summary["presents"] == presents and summary["fence_timeouts"] == 0, summary
    return score[0], pictures(out)


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
        for name in ("X2W_REPLAY_RECORD", "X2W_REPLAY_RECORD_FILE", "X2W_PRESENT_PICTURES",
                     "X2W_REPLAY_DESCRIPTOR_SNAPSHOT", "X2W_TEST_REPLAY_DESCRIPTOR_WINDOW"):
            env.pop(name, None)
        code, text = run([scene, vs, ps, root, backend, 1], env)
        if code == 77 and backend == "hardware":
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        assert code == 0, text[-3000:]
        subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)

        recorded_dir, recording = directory / "recorded", directory / "run.x2wr"
        recorded_dir.mkdir()
        code, text = run([scene, vs, ps, root, backend, PRESENTS],
                         dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                              X2W_PRESENT_PICTURES=str(recorded_dir)))
        assert code == 0, text[-3000:]
        (directory / "recorded.txt").write_text(text)
        recorded = pictures(recorded_dir)
        names = ["present_%04d.png" % n for n in range(1, PRESENTS + 1)]
        assert sorted(recorded) == names, sorted(recorded)
        assert recorded[names[0]] != recorded[names[-1]], "the scene does not change between presents"

        score, replayed = replay(player, recording, directory / "replayed", backend, env)
        assert sorted(replayed) == names, sorted(replayed)
        differ = [n for n in names if replayed[n] != recorded[n]]
        assert not differ, "replayed pictures differ from the recorded run: %s\n%s\n%s" % (
            differ, explain(recorded[differ[0]], replayed[differ[0]], directory / "replayed" / "stderr.txt"),
            explain_log("recorded run", directory / "recorded.txt"))

        changed = directory / "changed.x2wr"
        change_last_clear(recording, changed)
        _score, mutated = replay(player, changed, directory / "changed", backend, env)
        assert mutated[names[-1]] != recorded[names[-1]], "a changed clear colour replayed to the same last picture"
        assert all(mutated[n] == recorded[n] for n in names[:-1]), "the change reached other presents"

        # A window in the middle: present 6 opens it (that Present call itself is before it), so frames 7 to 12 are in
        # it, each with its own commands; the last present ends it. The calls before it that make the state
        # (creations, views, heap starts, signals) are recorded too, so every object the window names is known.
        window_dir, windowed = directory / "window_recorded", directory / "window.x2wr"
        window_dir.mkdir()
        code, text = run([scene, vs, ps, root, backend, PRESENTS],
                         dict(env, X2W_REPLAY_RECORD="f6,6", X2W_REPLAY_RECORD_FILE=str(windowed),
                              X2W_REPLAY_DESCRIPTOR_SNAPSHOT="1", X2W_TEST_REPLAY_DESCRIPTOR_WINDOW="1",
                              X2W_PRESENT_PICTURES=str(window_dir)))
        assert code == 0, text[-3000:]
        whole = pictures(window_dir)
        _score, played = replay(player, windowed, directory / "window_replayed", backend, env, presents=6)
        # Replayed present k is recorded present k + 6.
        pairs = [("present_%04d.png" % k, "present_%04d.png" % (k + 6)) for k in range(1, 7)]
        differ = [b for a, b in pairs if played[a] != whole[b]]
        assert not differ, "window replay differs from the recorded run at %s\n%s" % (
            differ, explain(whole[differ[0]], played["present_%04d.png" % (int(differ[0][8:12]) - 6)],
                            directory / "window_replayed" / "stderr.txt"))
        data = windowed.read_bytes()
        snapshot, arrays, source_calls = [], [], []
        control = bytearray()
        for tag, header, _hb, payload, size in chunks(data):
            if tag == b"DSNP":
                snapshot.append(struct.unpack_from("<Q", data, header)[0])
                continue
            if tag == b"DARR":
                arrays.append((struct.unpack_from("<Q", data, header)[0], size))
            if tag == b"CALL":
                seq, _tid, kind, slot = struct.unpack_from("<QIHH", data, header)
                if kind == 0 and slot in (20, 23):
                    source_calls.append(seq)
            control.extend(data[header - 16:payload + size])
        assert len(snapshot) == 1 and len(arrays) == 1 and arrays[0][1] == 40 * 8 + 8 + 4
        assert len(source_calls) == 41 and all(seq < snapshot[0] for seq in source_calls)
        assert arrays[0][0] == source_calls[-1]
        missing = directory / "window_without_snapshot.x2wr"
        missing.write_bytes(control)
        _score, without_snapshot = replay(player, missing, directory / "window_without_snapshot", backend, env, presents=6)
        for a, b in pairs:
            assert without_snapshot[a] != whole[b], "removing the descriptor snapshot did not change " + b
            (width, _height), pixels = without_snapshot[a]
            last_pixel = 1 + 4 * (width - 1)
            assert pixels[last_pixel:last_pixel + 4] == bytes((255, 255, 0, 255)), "fallback clear missing in " + a
        # The target made through device slot 127 (ten arguments: its output is the tenth): replays equal too.
        wide_dir, wide = directory / "wide_recorded", directory / "wide.x2wr"
        wide_dir.mkdir()
        code, text = run([scene, vs, ps, root, backend, PRESENTS],
                         dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(wide), X2W_PRESENT_PICTURES=str(wide_dir),
                              X2W_TEST_REPLAY_CREATE127="1"))
        assert code == 0, text[-3000:]
        wide_recorded = pictures(wide_dir)
        _score, wide_played = replay(player, wide, directory / "wide_replayed", backend, env)
        differ = [n for n in names if wide_played.get(n) != wide_recorded[n]]
        assert not differ, "slot-127 replay differs at %s" % differ
        print("replay: %d presents equal to the recorded run; changed recording differs at the last present only; "
              "window f6,6 replays presents 7-12 equal and all six differ without the descriptor snapshot; %s" %
              (PRESENTS, score))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
