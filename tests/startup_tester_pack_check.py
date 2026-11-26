#!/usr/bin/env python3
"""Check tools/startup/tester_pack.py: the `menus` input preset and the zip a tester mails back.

Pure Python, no title, no GPU. The zip must hold every log of the run plus a valid manifest, never a dump or the shader cache, and split into
numbered parts instead of dropping a log when it is over the size limit.
"""
import json
import os
import random
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "startup"))
import tester_pack as T  # noqa: E402

MANIFEST = {"commit": "abcdef12", "build_commit": "abcdef12", "dirty": False, "seconds": 300,
            "input_script": "4@70", "env": {"X2W_D3D12_BUDGET": "1"}, "gpu": "GPU", "gpu_driver": "1.0",
            "windows_build": "26100", "wall_seconds": 305.5, "deep": False, "lazy_textures": True, "shader_cache": "shader_cache_abcdef12",
            "translations_at_start": 0}


def write(folder, name, data):
    (folder / name).write_bytes(data)


def check_preset():
    script = T.input_preset("menus", 300)
    items = script.split(",")
    assert items[:4] == ["80@5", "80@11", "4@17", "80@23"], items[:4]
    times = [int(i.split("@")[1]) for i in items]
    assert times == list(range(5, 300, 6)), times
    assert [i.split("@")[0] for i in items[:6]] == ["80", "80", "4", "80", "80", "4"]
    assert T.input_preset("menus", 5) == "", "no entry before the start second"
    assert T.input_preset("menus", 420).count(",") + 1 == 70
    long_menus = T.input_preset("menus", 600).split(",")
    assert [int(i.split("@")[1]) for i in long_menus] == list(range(5, 500, 6)), "the menus preset stops at 500 s"
    assert len(long_menus) <= T.MAX_ENTRIES
    long_run = T.input_preset("menus", 100000).split(",")
    assert len(long_run) == len(range(5, 500, 6)) and len(",".join(long_run)) <= T.MAX_SCRIPT_CHARS
    menus2 = T.input_preset("menus2", 600)
    items2 = menus2.split(",")
    assert len(items2) <= T.MAX_ENTRIES and len(menus2) <= T.MAX_SCRIPT_CHARS and items2[0] == "80@5"
    assert menus2 == T.input_preset("menus2", 30), "menus2 does not depend on the run length"
    assert items2[1:5] == ["80@+1", "80@+1", "4@+1", "wait=2/90/go"], "down, down, A, then wait for the new picture"
    for n, item in enumerate(items2[1:]):
        assert (item.startswith("wait")) == (n % 4 == 3), "a wait follows every A, and only an A"
        assert (item.startswith("4@")) == (n % 4 == 2), "A is the third press of each cycle"
    assert T.input_preset("4@60,1@70", 300) == "4@60,1@70", "a raw script is passed through"
    for bad in ("4@1," * 300, "x" * 5000):
        try:
            T.input_preset(bad, 300)
        except ValueError:
            continue
        raise AssertionError("an oversized raw script was accepted")


def make_run(folder, captures, same_picture=False, events_bytes=1000):
    folder = Path(folder)
    rng = random.Random(3)
    rng_same = random.Random(5)
    same = bytes(rng_same.randrange(256) for _ in range(2000))
    write(folder, "finished.txt", b"finished\n")
    write(folder, "events.txt", b"e" * events_bytes)
    write(folder, "stderr.txt", b"err\n")
    write(folder, "translate.txt", b"t\n")
    write(folder, "translate_err.txt", b"")
    write(folder, "window.png", bytes(rng.randrange(256) for _ in range(2000)))
    for n in range(captures):
        picture = same if same_picture else bytes(rng.randrange(256) for _ in range(2000))
        write(folder, "window_%03ds.png" % (15 * (n + 1)), picture)
    # The rest of the run folder travels too; a dump and a sub-folder do not.
    write(folder, "stdout.txt", b"title text")
    write(folder, "files.txt", b"paths")
    write(folder, "trap.log", b"x")
    write(folder, "threads.txt", b"x")
    write(folder, "threads_stop.txt", b"x")
    write(folder, "bink_luma.png", b"png")
    write(folder, "crash.dmp", b"memory")
    os.makedirs(folder / "shader_cache", exist_ok=True)
    write(folder / "shader_cache", "request.bin", b"x")


def check_zip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = tmp / "run"
        run.mkdir()
        make_run(run, 5)
        write(tmp, "run.out", b"debugger summary\n")  # <logdir>.out beside the folder
        parts = T.pack(str(run), MANIFEST, str(tmp / "r.zip"))
        assert [p for p, _ in parts] == [str(tmp / "r.zip")], parts
        with zipfile.ZipFile(tmp / "r.zip") as z:
            names = sorted(z.namelist())
            manifest = json.loads(z.read("manifest.json"))
            assert z.read("stdout.txt") == b"title text" and z.read("debugger_summary.out") == b"debugger summary\n"
        everything = sorted(["finished.txt", "events.txt", "stderr.txt", "translate.txt", "translate_err.txt", "window.png",
                             "stdout.txt", "files.txt", "trap.log", "threads.txt", "threads_stop.txt", "bink_luma.png",
                             "debugger_summary.out", "manifest.json"] + ["window_%03ds.png" % (15 * (n + 1)) for n in range(5)])
        assert names == everything, names
        for gone in ("request.bin", "crash.dmp", "shader_cache"):
            assert gone not in names
        assert manifest["commit"] == "abcdef12" and manifest["deep"] is False and manifest["parts"] == 1
        assert manifest["first_poll_seconds"] is None
        assert set(manifest["files"]) == set(names) - {"manifest.json"}

        # Past the limit nothing is dropped: the logs go into numbered parts, each under the limit.
        run2 = tmp / "run2"
        run2.mkdir()
        make_run(run2, 12)
        limit = T.MANIFEST_ROOM + 20 * 1024
        parts2 = T.pack(str(run2), MANIFEST, str(tmp / "r2.zip"), limit, chunk_bytes=64 * 1024)
        assert len(parts2) > 1 and all(size <= limit for _, size in parts2), parts2
        assert parts2[0][0].endswith("_part01of%02d.zip" % len(parts2)), parts2
        seen = {}
        for path, _ in parts2:
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    assert name not in seen, name
                    seen[name] = z.read(name)
        assert json.loads(seen.pop("manifest.json"))["parts"] == len(parts2)
        for name in ("events.txt", "window.png", "window_180s.png", "stdout.txt"):
            assert name in seen, name
        on_disk = sorted(n for n in os.listdir(run2) if (run2 / n).is_file() and not n.endswith(".dmp"))
        assert sorted(seen) == on_disk, "a log was dropped or an extra file travelled"

        # A file larger than a piece is cut into numbered pieces that join back to the original.
        run3 = tmp / "run3"
        run3.mkdir()
        make_run(run3, 0)
        rng = random.Random(9)
        big = bytes(rng.randrange(256) for _ in range(400 * 1024))
        write(run3, "events.txt", big)
        parts3 = T.pack(str(run3), MANIFEST, str(tmp / "r3.zip"), T.MANIFEST_ROOM + 300 * 1024, chunk_bytes=128 * 1024)
        pieces = {}
        for path, size in parts3:
            assert size <= T.MANIFEST_ROOM + 300 * 1024, size
            with zipfile.ZipFile(path) as z:
                pieces.update({n: z.read(n) for n in z.namelist() if n.startswith("events.txt.")})
        assert sorted(pieces) == ["events.txt.001", "events.txt.002", "events.txt.003", "events.txt.004"], sorted(pieces)
        assert b"".join(pieces[n] for n in sorted(pieces)) == big


def check_cache_stats():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert T.cache_stats(str(tmp / "missing")) == (0, 0, 0)
        for name in ("req_0123456789abcdef.a", "req_0123456789abcdef.txt", "req_0123456789abcdef.cs.cso",
                     "req_1123456789abcdef.a", "req_1123456789abcdef.fail", "req_2123456789abcdef.m57",
                     "req_2123456789abcdef.ms57.txt", "req_3123456789abcdef.c67", "revision.txt", "notes.txt"):
            write(tmp, name, b"x")
        assert T.cache_stats(str(tmp)) == (2, 4, 1), T.cache_stats(str(tmp))


def check_cache_tag():
    with tempfile.TemporaryDirectory() as tmp:
        translator = Path(tmp) / "gfx2hlsl.py"
        translator.write_text("a = b'x2w-startup-shader-28'\nb = b'x2w-startup-shader-28cs'\nc = b'x2w-startup-mesh-57-3'\n")
        assert T.cache_tag(str(translator)) == "28-28-3"
        translator.write_text("a = b'x2w-startup-shader-29'\nb = b'x2w-startup-shader-29cs'\n")
        assert T.cache_tag(str(translator)) == "29-29-x", "a missing literal must not crash"


def check_first_poll():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert T.first_poll(str(tmp)) is None
        write(tmp, "stderr.txt", b"noise\nx2w-input script time base first-poll: first poll at 151.9 s, first entry at 156.9 s since process start\n")
        assert T.first_poll(str(tmp)) == (151.9, 156.9)


def check_manifest():
    for key in T.MANIFEST_KEYS:
        broken = {k: v for k, v in MANIFEST.items() if k != key}
        try:
            T.validate_manifest(broken)
        except ValueError:
            continue
        raise AssertionError("manifest without %s accepted" % key)
    try:
        T.validate_manifest(dict(MANIFEST, env={"PATH": "x"}))
    except ValueError:
        return
    raise AssertionError("a non X2W_ environment variable was accepted")


if __name__ == "__main__":
    check_preset()
    check_manifest()
    check_first_poll()
    check_cache_stats()
    check_cache_tag()
    check_zip()
    print("startup_tester_pack_check: preset, manifest and zip checks pass")
