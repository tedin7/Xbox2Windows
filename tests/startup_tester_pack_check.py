#!/usr/bin/env python3
"""Check tools/startup/tester_pack.py: the `menus` input preset and the zip a tester mails back.

Pure Python, no title, no GPU. The zip must hold every log of the run plus a valid manifest, never a dump or the shader cache, and split into
numbered parts instead of dropping a log when it is over the size limit.
"""
import json
import os
import random
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "startup"))
import tester_pack as T  # noqa: E402

MANIFEST = {"commit": "abcdef12", "build_commit": "abcdef12", "dirty": False, "seconds": 300,
            "input_script": "4@70", "env": {"X2W_D3D12_BUDGET": "1"}, "gpu": "GPU", "gpu_driver": "1.0",
            "windows_build": "26100", "wall_seconds": 305.5, "deep": False, "lazy_memory": True, "memory_gib": 14, "shader_cache": "shader_cache_abcdef12",
            "translations_at_start": 0}


DXBC = b"DXBC" + bytes(20) + struct.pack("<II", 40, 1) + bytes(8)  # a container whose header length matches (4 + 20 + 8 + 8 = 40)


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
        # A shader cache as the translator writes it (gfx2hlsl.py): request req_<key>.<kind>; the finished result is the
        # manifest <key>.txt (<key>.ms57.txt for a mesh pipeline) plus the bytecode; a refusal is <key>.fail.
        files = {
            "req_000000000000000a.a": b"x", "000000000000000a.txt": b"x", "000000000000000a.vs.cso": b"x",
            "000000000000000a.ps.cso": b"x",                                        # 1 pixel pipeline, ready
            "req_000000000000000b.c67": b"x", "000000000000000b.txt": b"x", "000000000000000b.cs.cso": b"x",  # compute, ready
            "req_000000000000000c.m57": b"x", "000000000000000c.ms57.txt": b"x", "000000000000000c.ms57.cso": DXBC,
            "000000000000000c.ps57.cso": DXBC,                                      # mesh, ready
            "req_000000000000000d.m57": b"x", "000000000000000d.ms57.txt": b"ps none\n", "000000000000000d.ms57.cso": DXBC,
            "req_000000000000000e.rtv": b"x", "000000000000000e.txt": b"x", "000000000000000e.vs.cso": b"x",
            "000000000000000e.ps.cso": b"x",                                        # a variant request, ready
            "req_0000000000000010.a": b"x", "0000000000000010.fail": b"unsupported\n",   # refused
            "req_0000000000000011.a": b"x",                                         # pending: nothing yet
            "req_0000000000000012.a": b"x", "0000000000000012.txt": b"x", "0000000000000012.vs.cso": b"x",
            "0000000000000012.ps.cso": b"",                                         # pending: empty bytecode
            "req_0000000000000013.a": b"x", "0000000000000013.txt": b"x",
            "0000000000000013.vs.cso": b"x", "0000000000000013.ps.cso": b"x", "0000000000000013.fail": b"old\n",  # ready, stale .fail
            "req_0000000000000014.m57": b"x", "0000000000000014.ms57.txt": b"x",    # pending: mesh without bytecode
            "revision.txt": b"x", "notes.txt": b"x", "0000000000000099.txt": b"x",  # no request: not counted
        }
        for name, data in files.items():
            write(tmp, name, data)
        assert T.cache_stats(str(tmp)) == (6, 10, 1), T.cache_stats(str(tmp))
        # The translator itself calls the same requests finished: nothing it would translate again is counted ready.
        try:
            import gfx2hlsl
        except ImportError:
            return
        pending = {"req_%016x" % key for key in gfx2hlsl.pending_keys(str(tmp), False)}
        stem = {n.split(".")[0] for n in files if n.startswith("req_")}
        assert stem - pending == {"req_000000000000000a", "req_000000000000000b", "req_000000000000000c",
                                  "req_000000000000000d", "req_000000000000000e", "req_0000000000000013"}, pending


def check_cache_tag():
    with tempfile.TemporaryDirectory() as tmp:
        translator = Path(tmp) / "gfx2hlsl.py"
        translator.write_text("a = b'x2w-startup-shader-28'\nb = b'x2w-startup-shader-28cs'\nc = b'x2w-startup-mesh-57-3'\n")
        assert T.cache_tag(str(translator)) == "28-28-3"
        translator.write_text("a = b'x2w-startup-shader-29'\nb = b'x2w-startup-shader-29cs'\n")
        assert T.cache_tag(str(translator)) == "29-29-x", "a missing literal must not crash"


def check_memory_gib():
    gib = 1 << 30
    assert T.memory_gib(8 * gib) == 14 and T.memory_gib(16 * gib) == 14, "never below run.ps1's default"
    assert T.memory_gib(32 * gib) == 24 and T.memory_gib(64 * gib) == 48 and T.memory_gib(128 * gib) == 96
    assert T.memory_gib(int(31.9 * gib)) == 23, "whole GiB, rounded down"


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


HLSL_SOURCE = b"float4 main(float4 p : SV_Position) : SV_Target { return secret_shader_body(p); }"


def make_cache(folder):
    """A fake shader cache: every kind, every status, a compiler message that quotes source, binaries and a long reason."""
    ok = "0000000000000001"
    layout = {
        # graphics
        "req_%s.a" % ok: b"ISA", "req_%s.txt" % ok: b"ready", "req_%s.vs.cso" % ok: b"\x00DXBC secret binary",
        "req_1000000000000002.a": b"ISA", "req_1000000000000002.fail": b"v_cvt_pknorm_u16_f32 is not covered: v12, s[4:5] 0x1f\nsecond line\n",
        "req_1000000000000003.a": b"ISA", "req_1000000000000003.fail": b"v_cvt_pknorm_u16_f32 is not covered: v9, s[6:7] 0x2a\n",
        "req_1000000000000004.a": b"ISA",  # pending: a request nobody answered
        # compute
        "req_2000000000000001.c67": b"ISA", "req_2000000000000001.fail": b"image_store on SQ_RSRC_IMG_3D: only 2D images are lowered\n",
        "req_2000000000000002.c67": b"ISA",
        "req_2000000000000003.c67": b"ISA",
        "req_2000000000000003.fail": b"dxc cs: " + HLSL_SOURCE + b"\nerror X3004: undeclared identifier 'secret_shader_body'\nerror X3004: again\n",
        # mesh 57 and 86
        "req_3000000000000001.m57": b"ISA", "req_3000000000000001.ms57.txt": b"ready", "req_3000000000000001.ms57.cso": b"\x00DXBC mesh",
        "req_3000000000000002.m57": b"ISA", "req_3000000000000002.fail": b"slot 57 mesh: a loop outside the group-memory exchanges is not covered\n",
        "req_4000000000000001.m86": b"stream", "4000000000000001.mesh86.txt": b"x2w-mesh86 ready",
        "req_4000000000000002.m86": b"stream",
        # not a key
        "revision.txt": b"28", "notes.txt": b"x",
    }
    for name, data in layout.items():
        write(folder, name, data)
    write(folder, "req_5000000000000001.a", b"ISA")
    write(folder, "req_5000000000000001.fail", b"unsupported: " + b"x" * 500 + b"\n")


def check_cache_results():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert T.cache_results(str(tmp / "missing")) == []
        cache = tmp / "cache"
        cache.mkdir()
        make_cache(cache)
        rows = {key: (kind, status, reason) for key, kind, status, reason in T.cache_results(str(cache))}
        assert len(rows) == 12, sorted(rows)
        assert rows["0000000000000001"] == ("graphics", "ok", "")
        assert rows["1000000000000002"] == ("graphics", "refused", "v_cvt_pknorm_u16_f32 is not covered: v12, s[4:5] 0x1f"), rows["1000000000000002"]
        assert rows["1000000000000004"] == ("graphics", "pending", "")
        assert rows["2000000000000001"][:2] == ("compute", "refused") and rows["2000000000000002"][:2] == ("compute", "pending")
        assert rows["3000000000000001"][:2] == ("mesh57", "ok") and rows["3000000000000002"][:2] == ("mesh57", "refused")
        assert rows["4000000000000001"][:2] == ("mesh86", "ok") and rows["4000000000000002"][:2] == ("mesh86", "pending")
        assert len(rows["5000000000000001"][2]) == T.REASON_CHARS, "a long reason is cut"
        # a compiler message quotes the source: only the stage and the error codes stay
        assert rows["2000000000000003"] == ("compute", "refused", "dxc cs compile failed (error X3004)"), rows["2000000000000003"]

        text = T.format_results(T.cache_results(str(cache)))
        assert text.splitlines()[1] == "keys 12 (compute 3, graphics 5, mesh57 2, mesh86 2): ok 3, refused 6, pending 3", text.splitlines()[1]
        assert b"secret_shader_body" not in text.encode() and "DXBC" not in text and "ISA" not in text
        # the two cvt refusals differ only in registers and constants: one wall, counted twice, first in the ranking
        ranking = text.split("## keys")[0].splitlines()
        assert ranking[4].startswith("    2  v_cvt_pknorm_uN_fN is not covered: R, s[N:N] X   [graphics 2]"), ranking[4]
        assert "2000000000000001 compute refused image_store on SQ_RSRC_IMG_3D: only 2D images are lowered" in text
        assert "0000000000000001 graphics ok\n" in text and "4000000000000002 mesh86 pending\n" in text

        # The zip carries shader_results.txt only when a cache is given, and never a cache file.
        run = tmp / "run"
        run.mkdir()
        make_run(run, 1)
        T.pack(str(run), MANIFEST, str(tmp / "without.zip"))
        T.pack(str(run), MANIFEST, str(tmp / "with.zip"), cache_dir=str(cache))
        with zipfile.ZipFile(tmp / "without.zip") as z:
            assert T.RESULTS_NAME not in z.namelist()
        with zipfile.ZipFile(tmp / "with.zip") as z:
            names = z.namelist()
            assert T.RESULTS_NAME in names and not [n for n in names if n.startswith("req_")], names
            assert z.read(T.RESULTS_NAME).decode() == text
            manifest = json.loads(z.read("manifest.json"))
            assert manifest["cache_keys"] == 12 and manifest["files"][T.RESULTS_NAME]["bytes"] == len(text.encode())


if __name__ == "__main__":
    check_cache_results()
    check_preset()
    check_manifest()
    check_first_poll()
    check_cache_stats()
    check_cache_tag()
    check_memory_gib()
    check_zip()
    print("startup_tester_pack_check: preset, manifest and zip checks pass")
