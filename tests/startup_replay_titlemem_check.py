#!/usr/bin/env python3
"""x2w_replay restores the TITLE MEMORY a recorded draw reads (PAGE/MAP replay), end to end.

startup_replay_titlemem is a scene whose only changing input is a page of the title's own memory: every frame the "title" writes
a colour there (four floats), binds the address as root constant buffer 0 and draws with a translated pixel shader that
returns those floats. Checked, on the recorded run and on the replay:
  - the picture's dominant colour is the colour the title wrote for that frame (so the bytes really reached the draw),
    and the 12 pictures differ from each other (the page changes between presents);
  - replay: x2w_replay (no X2W_ setting) writes the same picture names, every one equal to the recorded run's;
  - control: the last frame's colour changed inside the recording's page data replays to a different last picture only
    (skipped with a printed note when the recording keeps pages compressed so the bytes cannot be found by value);
  - control: a recording of the same run in which every page chunk is blanked (zero bytes at the colour's place of every frame)
    does NOT replay to the recorded pictures (the pictures come from the restored pages, not from the calls).

usage: startup_replay_titlemem_check.py <startup_replay_titlemem> <x2w_replay> <warp|hardware>
"""
from collections import Counter
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_replay_check import PRESENTS, explain, pictures, pixels, replay, run
from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex

HLSL = (
    "ByteAddressBuffer constants : register(t0);\n"
    "float4 main(float4 p : SV_Position) : SV_Target {\n"
    "  return asfloat(constants.Load4(0));\n"
    "}\n"
)


def colour_of(frame):
    t = frame / PRESENTS
    return (t, 0.25, 1.0 - t, 1.0)


def dominant(picture):
    """The most common non-black pixel of a PNG's raw scanlines, as 8-bit RGBA."""
    (width, height), raw = picture
    counts = Counter()
    stride = width * 4 + 1
    for y in range(height):
        row = raw[y * stride + 1:(y + 1) * stride]
        for x in range(width):
            px = bytes(row[x * 4:x * 4 + 4])
            if px[:3] != b"\x00\x00\x00":
                counts[px] += 1
    return counts.most_common(1)[0][0] if counts else None


def expect_colour(picture, frame, name):
    got = dominant(picture)
    want = [round(c * 255) for c in colour_of(frame)]
    assert got is not None and all(abs(g - w) <= 2 for g, w in zip(got, want)), \
        "%s: dominant colour %s, the title wrote %s" % (name, list(got) if got else None, want)


def main(scene, player, backend):
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        cache = directory / "cache"
        cache.mkdir()
        vs, ps, root = (directory / name for name in ("vs", "ps", "root"))
        vs.write_bytes(vertex())
        ps.write_bytes(pixel(False))
        env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
                   X2W_D3D12_DEVICE_NOOP="1", X2W_SHADER_CACHE=str(cache), X2W_TEST_ROOT_CBV="bound")
        env.setdefault("X2W_D3D12_DEBUG", "0")
        for name in ("X2W_REPLAY_RECORD", "X2W_REPLAY_RECORD_FILE", "X2W_PRESENT_PICTURES"):
            env.pop(name, None)
        code, text = run([scene, vs, ps, root], env)
        if code == 77:
            print("capability skip")
            return 77
        assert code == 0, text[-3000:]
        subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
        manifest, = (p for p in cache.glob("*.txt") if p.name != "revision.txt")
        with manifest.open("a", encoding="ascii") as file:
            file.write("\nsrv 0 buf root 0\n")
        source = manifest.with_suffix(".ps.hlsl")
        source.write_text(HLSL)
        fxc = env.get("X2W_FXC")
        assert fxc, "X2W_FXC is not set"
        subprocess.run([fxc, "/nologo", "/T", "ps_5_0", "/E", "main", "/Fo", manifest.with_suffix(".ps.cso"), source], check=True)

        recorded_dir, recording = directory / "recorded", directory / "run.x2wr"
        recorded_dir.mkdir()
        code, text = run([scene, vs, ps, root, backend, PRESENTS],
                         dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                              X2W_PRESENT_PICTURES=str(recorded_dir)))
        if code == 77 and backend == "hardware":
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        assert code == 0, text[-3000:]
        pages_log = [l for l in text.splitlines() if "x2w-replay pages:" in l]
        recorded = pictures(recorded_dir)
        names = ["present_%04d.png" % n for n in range(1, PRESENTS + 1)]
        assert sorted(recorded) == names, sorted(recorded)
        assert len({recorded[n][1] for n in names}) == PRESENTS, "the pictures do not change between presents"
        for number, name in enumerate(names):
            expect_colour(recorded[name], number, "recorded " + name)

        data = recording.read_bytes()
        assert pages_log, "the recorder never ran its page scan (no 'x2w-replay pages:' line):\n" + text[-1500:]
        ranges = re.search(r"ranges (\d+)", pages_log[0]) if pages_log else None
        assert ranges and int(ranges.group(1)) >= 1, "the page scan found no title-memory range: " + (pages_log[0] if pages_log else "no log")
        assert data.count(b"PAGE") >= 1 and data.count(b"MAP ") >= 1, "the recording holds no PAGE/MAP chunk"
        score, replayed = replay(player, recording, directory / "replayed", backend, env)
        assert sorted(replayed) == names, sorted(replayed)
        differ = [n for n in names if replayed[n] != recorded[n]]
        assert not differ, "replayed pictures differ from the recorded run: %s\n%s" % (
            differ, explain(recorded[differ[0]], replayed[differ[0]], directory / "replayed" / "stderr.txt")) + "\nrecorder: " + pages_log[0] + "\nreplay page lines:\n" + "\n".join(
            l for l in (directory / "replayed" / "stderr.txt").read_text(errors="replace").splitlines() if "page" in l.lower() or "MAP" in l)[-1500:]

        note = "page control skipped (page bytes not stored by value)"
        last = struct.pack("<4f", *colour_of(PRESENTS - 1))
        if data.count(last) >= 1:
            changed = directory / "changed.x2wr"
            patched = bytearray(data)
            at = -1
            while True:       # every stored copy of the last frame's colour (the page and the argument copy, if any)
                at = data.find(last, at + 1)
                if at < 0:
                    break
                patched[at:at + 16] = struct.pack("<4f", 1.0 - colour_of(PRESENTS - 1)[0], 0.25, 1.0 - colour_of(PRESENTS - 1)[2], 1.0)
            changed.write_bytes(bytes(patched))
            _score, mutated = replay(player, changed, directory / "changed", backend, env)
            assert mutated[names[-1]] != recorded[names[-1]], "a changed page replayed to the same last picture"
            assert all(mutated[n] == recorded[n] for n in names[:-1]), "the change reached other presents"
            note = "changed last page differs at the last present only"

        blank = directory / "blank.x2wr"
        wiped = bytearray(data)
        wipes = 0
        for frame in range(PRESENTS):
            value = struct.pack("<4f", *colour_of(frame))
            at = -1
            while True:
                at = data.find(value, at + 1)
                if at < 0:
                    break
                wiped[at:at + 16] = bytes(16)
                wipes += 1
        if wipes:
            blank.write_bytes(bytes(wiped))
            _score, zeroed = replay(player, blank, directory / "blank", backend, env)
            assert any(zeroed[n] != recorded[n] for n in names), "blanked page data replayed to the recorded pictures"
        report = ("replay pages %s: %s; PAGE %d MAP %d; recording %d bytes, page bytes by value %d, blank wipes %d, %s" % (
            backend, pages_log[0].strip()[:200], data.count(b"PAGE"), data.count(b"MAP "), len(data), data.count(last), wipes, note))
        if os.environ.get("X2W_PAGES_REPORT"):      # the diagnostic line for the board (a passing test prints nothing under ctest)
            Path(os.environ["X2W_PAGES_REPORT"]).write_text(report + "\n")
        print("replay pages: %d pictures equal to the recorded run, every one the colour the title wrote; %s; "
              "blank control wipes=%d; %s" % (PRESENTS, note, wipes, score))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
