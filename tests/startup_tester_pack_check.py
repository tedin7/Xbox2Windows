#!/usr/bin/env python3
"""Check tools/startup/tester_pack.py: the `menus` input preset and the zip a tester mails back.

Pure Python, no title, no GPU. The zip must hold every log of the run plus a valid manifest, never a dump or the shader cache, and split into
numbered parts instead of dropping a log when it is over the size limit.
"""
import json
import os
import random
import re
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "startup"))
import tester_pack as T  # noqa: E402

MANIFEST = {"commit": "abcdef12", "build_commit": "abcdef12", "dirty": False, "seconds": 300,
            "input_script": "4@70", "env": {"X2W_D3D12_BUDGET": "1"}, "gpu": "GPU", "gpu_driver": "1.0",
            "windows_build": "26100", "wall_seconds": 305.5, "deep": False, "lazy_textures": True, "lazy_memory": True, "memory_gib": 14, "shader_cache": "shader_cache_abcdef12",
            "translations_at_start": 0}


DXBC = b"DXBC" + bytes(20) + struct.pack("<II", 40, 1) + bytes(8)  # a container whose header length matches (4 + 20 + 8 + 8 = 40)


def write(folder, name, data):
    (folder / name).write_bytes(data)


def check_preset():
    script = T.input_preset("menus", 300, 0)
    items = script.split(",")
    assert items[:4] == ["80@5+0.8", "80@6+0.8", "4@7+0.8", "80@8+0.8"], items[:4]
    times = [int(i.split("@")[1].split("+")[0]) for i in items]
    assert times == list(range(5, 5 + T.MAX_ENTRIES)), "one press a second until the entry limit"
    assert [i.split("@")[0] for i in items[:6]] == ["80", "80", "4", "80", "80", "4"]
    assert T.input_preset("menus", 5, 0) == "", "no entry before the start second"
    assert T.input_preset("menus", 30, 0).count(",") + 1 == len(range(5, 30)), "stops at the run length"
    long_run = T.input_preset("menus", 100000, 0).split(",")
    assert len(long_run) == T.MAX_ENTRIES and len(",".join(long_run)) <= T.MAX_SCRIPT_CHARS
    passes = T.privacy_cycles(2)
    assert passes == [["40", "40", "4", "80", "4", "80", "4"],      # tick both boxes, Conferma
                      ["40", "40", "4", "80", "80", "4"],           # then one box at a time (Gray code), Conferma each time
                      ["40", "40", "80", "4", "80", "4"],
                      ["40", "40", "4", "80", "80", "4"]], "each pass anchors with up x2 and ends on Conferma"
    flat = [i.split("@")[0] for i in T.input_preset("menus", 300).split(",")]
    assert flat[:len(sum(passes, []))] == sum(passes, []), "the menus script is those passes in order"
    assert T.input_preset("menus", 60, 0).startswith("80@5+0.8,80@6+0.8,4@7+0.8"), "0 boxes = the plain pattern"
    labels = T.input_labels("menus", 300)
    assert len(labels) == len(T.input_preset("menus", 300).split(",")) and len(set(len(l) > 0 for l in labels)) == 1
    assert len(T.privacy_labels(2)) == len(sum(passes, [])), "one label per press of the passes"
    assert labels[:7] == ["privacy pass 1: up (anchor on box 1)", "privacy pass 1: up (anchor on box 1)", "privacy pass 1: tick box 1",
                          "privacy pass 1: down to box 2", "privacy pass 1: tick box 2", "privacy pass 1: down to Conferma",
                          "privacy pass 1: Conferma (Avanti on the legal notice)"], labels[:7]
    assert labels[7:13] == ["privacy pass 2: up (anchor on box 1)"] * 2 + ["privacy pass 2: toggle box 1", "privacy pass 2: down to box 2",
                                                                          "privacy pass 2: down to Conferma", "privacy pass 2: Conferma (Avanti on the legal notice)"]
    assert T.input_labels("menus", 20, 0)[:3] == ["menu: down", "menu: down", "menu: A"] and T.input_labels("menus2", 600) == []
    menus2 = T.input_preset("menus2", 600)
    items2 = menus2.split(",")
    assert len(items2) <= T.MAX_ENTRIES and len(menus2) <= T.MAX_SCRIPT_CHARS and items2[0] == "80@5+1.2"
    assert menus2 == T.input_preset("menus2", 30), "menus2 does not depend on the run length"
    assert items2[1:5] == ["80@+2+1.2", "80@+2+1.2", "4@+2+1.2", "wait=2/90/go"], "down, down, A, then wait for the new picture"
    for n, item in enumerate(items2[1:]):
        assert (item.startswith("wait")) == (n % 4 == 3), "a wait follows every A, and only an A"
        assert (item.startswith("4@")) == (n % 4 == 2), "A is the third press of each cycle"
    accept = T.input_preset("accept", 600)
    presses = accept.split(",")
    assert presses[:4] == ["4@35+0.3", "4@45+0.3", "4@55+1.5", "4@65+0.3"], presses[:4]
    assert presses[-1] == "4@415+1.5" and len(presses) == 39, (presses[-1], len(presses))
    assert {p.split("@")[0] for p in presses} == {"4"}, "only A (mask 4): never Start (1), D-pad or the other buttons"
    holds = [p for p in presses if p.endswith("+1.5")]
    assert holds and len(holds) == len(presses) // 3 and all(p.endswith("+0.3") for i, p in enumerate(presses) if (i + 1) % 3), "every 3rd press is the hold"
    assert max(int(p.split("@")[1].split("+")[0]) for p in presses) < T.ACCEPT_END_S == 420, "no press after the cutoff"
    assert max(int(p.split("@")[1].split("+")[0]) for p in T.input_preset("accept", 100000).split(",")) < 420, "a long run still stops pressing at 420 s"
    assert T.input_preset("accept", 35) == "" and T.input_preset("accept", 36) == "4@35+0.3", "presses stop with the run"
    assert len(T.input_preset("accept", 100000)) <= T.MAX_SCRIPT_CHARS
    labels = T.input_labels("accept", 600)
    assert len(labels) == 39 and labels[:3] == ["press A #1 (accept, tap)", "press A #2 (accept, tap)", "press A #3 (accept, hold 1.5 s)"], labels[:3]
    assert T.input_labels("accept", 20) == []
    for none in ("none", "off", "OFF", ""):
        assert T.input_preset(none, 300) == "" and T.input_labels(none, 300) == [], "%r presses nothing" % none
    assert T.input_preset("4@60,1@70", 300) == "4@60,1@70", "a raw script is passed through"
    for bad in ("4@1," * 300, "x" * 5000):
        try:
            T.input_preset(bad, 300)
        except ValueError:
            continue
        raise AssertionError("an oversized raw script was accepted")


def run_ui(script, arrival_delay, start, legal_focus=0, boxes_at_start=(False, False)):
    """A model of the screens the title shows before loading (Run L pictures): navigation and brightness (A continues), the
    legal notice (two list items and Avanti), the privacy screen (two boxes and Conferma) and loading. Lists clamp at both
    ends. A press only counts once its screen has arrived. Returns (screen at the end, boxes ticked, time loading began)."""
    presses = []
    for item in script.split(","):
        mask, rest = item.split("@")
        presses.append((int(rest.split("+")[0]), int(mask, 16)))
    order = ["nav", "bright", "legal", "privacy", "loading"]
    screen, active_at = 0, start
    focus = {"legal": legal_focus, "privacy": 0}
    privacy_opened = None
    boxes = list(boxes_at_start)
    for at, mask in presses:
        name = order[screen]
        if name == "loading":
            break
        if at < active_at:
            continue
        moved = False
        if name in ("nav", "bright"):
            moved = mask == 4
        else:
            if mask == 0x40:
                focus[name] = max(0, focus[name] - 1)
            elif mask == 0x80:
                focus[name] = min(2, focus[name] + 1)
            elif mask == 4:
                if name == "legal":
                    moved = focus[name] == 2
                elif focus[name] < 2:
                    boxes[focus[name]] = not boxes[focus[name]]
                else:
                    moved = all(boxes)
        if moved:
            screen += 1
            active_at = at + arrival_delay
            if order[screen] == "privacy":
                privacy_opened = active_at
    return order[screen], boxes, (active_at - arrival_delay if order[screen] == "loading" else None), privacy_opened


def check_privacy_screen():
    """The script against the screen model, over screen-arrival delays, start offsets, initial focus and box states."""
    script = T.input_preset("menus", 300)
    worst = worst_clean = 0
    for delay in (0, 3, 7, 15, 30):
        for start in (0, 12, 33, 70):
            for legal_focus in (0, 1):
                for ticked in ((False, False), (True, False), (False, True), (True, True)):
                    end, boxes, when, opened = run_ui(script, delay, start, legal_focus, ticked)
                    assert end == "loading" and boxes == [True, True] and when < 300, (delay, start, legal_focus, ticked, end, boxes, when)
                    took = when - opened   # seconds from the privacy screen being usable to the press that leaves it
                    worst = max(worst, took)
                    if ticked == (False, False) and delay == 0:
                        worst_clean = max(worst_clean, took)
    # fast: a press every second; nothing ticked yet (the normal case): at most the rest of the pass the screen opened in plus two passes
    assert worst_clean <= 34, worst_clean   # the screen opening in the middle of a pass is the worst case; aligned with a pass start it is one 7-press pass
    assert worst <= 34, worst   # any starting state, screen opening in the middle of a pass: the rest of that pass plus one round of four passes
    # control: the old pattern walks down to Conferma without ticking both boxes and is stuck on the privacy screen
    plain = T.input_preset("menus", 300, 0)
    for delay in (0, 15):
        end, boxes, _, _ = run_ui(plain, delay, 0)
        assert end == "privacy", (delay, end, boxes)   # never gets through: it cannot tick both boxes and confirm


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
        # every part says which part it is and what all parts hold
        notes = {n: seen.pop(n).decode() for n in sorted(seen) if re.match(r"^part_\d\d_of_\d\d\.txt$", n)}
        assert sorted(notes) == ["part_%02d_of_%02d.txt" % (i, len(parts2)) for i in range(1, len(parts2) + 1)], sorted(notes)
        first_note = notes["part_01_of_%02d.txt" % len(parts2)]
        assert first_note.startswith("part 1 of %d: mail ALL %d parts" % (len(parts2), len(parts2))), first_note[:80]
        assert all(("part %d of %d" % (i, len(parts2))) in first_note for i in range(1, len(parts2) + 1)), first_note
        assert "(this one)" in first_note and "events.txt" in first_note, first_note
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


def check_end_cause():
    """The end cause is recorded on every path, with the last present milestone."""
    beats = b"x2w-d3d12 heartbeat: 100 frames presented in 30.0 s (3.3 fps overall)\nx2w-d3d12 heartbeat: 450 frames presented in 120.0 s (3.7 fps overall)\n"

    def run(finished=None, files=()):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write(folder, "stderr.txt", beats)
            if finished is not None:
                write(folder, "finished.txt", finished.encode())
            for name, data in files:
                write(folder, name, data)
            return T.end_cause(str(folder))

    milestone = "last present #450 at 120 s"
    cases = [
        ("timeout after 600.02 s\n", (), "timeout after 600.02 s (the run was not stopped early)"),
        ("stopped by request after 90.5 s (stop file)\n", (), "stopped by request at 90.5 s (stop file, no watchdog stop)"),
        ("stopped by request after 90.5 s (commit headroom: system commit has 100 MiB free, under the headroom of 256 MiB)\n", (),
         "stopped by request at 90.5 s (commit headroom: system commit has 100 MiB free, under the headroom of 256 MiB, no watchdog stop)"),
        ("process exited with 0xffffffff after 256.1 s\n", (), "title killed externally (exit code 0xffffffff, no stop file) at 256.1 s"),
        ("process exited with 0x00000000 after 12.0 s\n", (), "title exited by itself (code 0x0) at 12.0 s"),
        ("process exited with 0xc0000005 after 40.0 s\n", (), "title crashed (exit code 0xc0000005) at 40.0 s"),
        ("stopped at access violation (0xc0000005) at d3d12_xs+0x10 after 33.3 s\n", (), "debugger stop: access violation (0xc0000005) at 33.3 s"),
        ("timeout after 300.0 s\n", (("stall_stop.json", b'{"at_seconds": 300, "reason": "deadlock: no frame"}'),),
         "watchdog at 300 s: deadlock: no frame"),
        (None, (("hard_end.json", b'{"at_seconds": 70, "note": "title 5 gone"}'),), "debugger did not finish: title 5 gone"),
        (None, (), "unknown: the debugger wrote no summary (killed, or still running)"),
    ]
    for finished, files, want in cases:
        got = run(finished, files)
        assert got == want + "; " + milestone, (finished, got)
    # No heartbeat at all: the milestone says so.
    with tempfile.TemporaryDirectory() as tmp:
        write(Path(tmp), "finished.txt", b"timeout after 5.0 s\n")
        assert T.end_cause(tmp).endswith("; no present reported"), T.end_cause(tmp)
    # A hard end after a finished summary is appended, not replacing it.
    got = run("timeout after 5.0 s\n", (("hard_end.json", b'{"note": "debugger ended"}'),))
    assert "timeout after 5.0 s" in got and "hard end: debugger ended" in got, got


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


def check_key_use():
    """The runtime's 'x2w-render key use' lines give each key its first and last use in shader_results.txt."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache = tmp / "cache"
        cache.mkdir()
        make_cache(cache)
        run = tmp / "run"
        run.mkdir()
        (run / "stderr.txt").write_text(
            "x2w-render key use compute 2000000000000001 first 12.5 s last 30.0 s uses 4\n"
            "x2w-render key use compute 2000000000000001 first 12.5 s last 250.5 s uses 900\n"  # printed again: the count grew
            "x2w-render key use graphics 1000000000000002 first 241.0 s last 242.0 s uses 3\n"
            "x2w-render key use graphics 0000000000000001 first 1.0 s last 2.0 s uses 7\n"  # ok key: only on its own line
            "x2w-render key use compute zzzz first 1 s last 2 s uses 1\n"  # malformed: ignored
            "x2w-render compute census: 3 keys dispatched\n", encoding="utf-8")
        use = T.key_use(str(run))
        assert use == {"2000000000000001": (12.5, 250.5, 900), "1000000000000002": (241.0, 242.0, 3), "0000000000000001": (1.0, 2.0, 7)}, use
        assert T.key_use(str(tmp / "missing")) == {} and T.key_use(None) == {}
        text = T.format_results(T.cache_results(str(cache)), use)
        assert "2000000000000001 compute refused image_store on SQ_RSRC_IMG_3D: only 2D images are lowered first=12.5s last=250.5s uses=900\n" in text, text
        assert "0000000000000001 graphics ok first=1.0s last=2.0s uses=7\n" in text
        assert not [l for l in text.splitlines() if l.startswith("1000000000000003 ") and "first=" in l], "a key never used has no use"
        # the refused keys used latest come first, latest last use on top; an ok key is not listed there
        section = text.split("## refused keys used latest")[1].split("## keys")[0].splitlines()[1:]
        keys = [line.split()[3] for line in section if line.strip()]
        assert keys == ["2000000000000001", "1000000000000002"], keys
        assert T.format_results(T.cache_results(str(cache))) == T.format_results(T.cache_results(str(cache)), {}), "no use lines: unchanged"
        assert "refused keys used latest" not in T.format_results(T.cache_results(str(cache))) and "first=1" not in T.format_results(T.cache_results(str(cache)))
        # packed with the cache, the zip's shader_results.txt carries the use
        T.pack(str(run), MANIFEST, str(tmp / "use.zip"), cache_dir=str(cache))
        with zipfile.ZipFile(tmp / "use.zip") as z:
            assert "uses=900" in z.read(T.RESULTS_NAME).decode()


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


def check_request_files():
    """The request files of the worst refused keys travel in their own zip folder: ranked by draws not drawn / dispatches
    without a translation, refused keys only, whole keys within the byte cap, listed in the manifest."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache = tmp / "cache"
        cache.mkdir()
        make_cache(cache)
        write(cache, "req_1000000000000003.root", b"R" * 1000)
        write(cache, "req_2000000000000001.root", b"S" * 2000)
        write(cache, "req_2000000000000001.sa", b"T" * 500)
        write(cache, "req_2000000000000001.txt", b"not a request file")
        stderr = (b"x2w-render not drawn key 1000000000000002: 500 (translation_not_ready)\n"
                  b"x2w-render not drawn key 1000000000000003: 900 (translation_not_ready)\n"
                  b"x2w-render not drawn key 0000000000000001: 99999 (an ok key is never picked)\n"
                  b"x2w-render compute census key 2000000000000001: 1000 dispatches, 100 translated, first refusal: x\n"
                  b"x2w-render compute census key 2000000000000001: 7000 dispatches, 100 translated, first refusal: x\n"
                  b"x2w-render compute census key 2000000000000003: 50 dispatches, 50 translated, first refusal: none\n"
                  b"x2w-render compute census key 3000000000000002: 0 dispatches, 0 translated, first refusal: none\n")
        weights = T.key_weights(stderr.decode())
        assert weights["2000000000000001"] == 6900 and weights["1000000000000003"] == 900, weights
        rows = T.cache_results(str(cache))
        chosen = T.select_requests(str(cache), rows, weights)
        assert [c[0] for c in chosen] == ["2000000000000001", "1000000000000003", "1000000000000002"], [c[0] for c in chosen]
        assert [n for n, _ in chosen[0][3]] == ["req_2000000000000001.c67", "req_2000000000000001.root", "req_2000000000000001.sa"]
        assert len(T.select_requests(str(cache), rows, weights, top=2)) == 2, "top N keys"
        assert [c[0] for c in T.select_requests(str(cache), rows, weights, cap=1500)] == ["1000000000000003", "1000000000000002"], \
            "a key that does not fit whole is skipped, the next one still goes in"
        run = tmp / "run"
        run.mkdir()
        make_run(run, 1)
        write(run, "stderr.txt", stderr)
        T.pack(str(run), MANIFEST, str(tmp / "with.zip"), cache_dir=str(cache))
        with zipfile.ZipFile(tmp / "with.zip") as z:
            names = z.namelist()
            folder = sorted(n for n in names if n.startswith(T.REQUESTS_DIR + "/"))
            assert folder == sorted(T.REQUESTS_DIR + "/" + n for c in chosen for n, _ in c[3]), folder
            assert not [n for n in names if n.startswith("req_")] and T.RESULTS_NAME in names
            assert z.read(T.REQUESTS_DIR + "/req_1000000000000003.root") == b"R" * 1000
            listed = json.loads(z.read("manifest.json"))["request_files"]
            assert listed["keys"] == 3 and listed["files"] == len(folder) and listed["folder"] == T.REQUESTS_DIR
            assert listed["bytes"] == sum(b for c in chosen for _, b in c[3]) <= T.REQUESTS_CAP_BYTES
            assert [r["key"] for r in listed["ranked"]] == [c[0] for c in chosen] and listed["ranked"][0]["weight"] == 6900
        # no stderr or no counters: no request files, still a valid zip
        write(run, "stderr.txt", b"err\n")
        T.pack(str(run), MANIFEST, str(tmp / "none.zip"), cache_dir=str(cache))
        with zipfile.ZipFile(tmp / "none.zip") as z:
            assert not [n for n in z.namelist() if n.startswith(T.REQUESTS_DIR)] and json.loads(z.read("manifest.json"))["request_files"]["keys"] == 0


def check_part_one_has_the_small_files():
    """Several parts: shader_results.txt, the debugger summary and every small log are in part 1 (Run Q: part 2 never arrived and
    took shader_results.txt and the summary with it), the big log chunks come after them."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache = tmp / "cache"
        cache.mkdir()
        make_cache(cache)
        run = tmp / "run"
        run.mkdir()
        make_run(run, 1)
        rng = random.Random(9)
        write(run, "stderr.txt", bytes(rng.randrange(256) for _ in range(700 * 1024)))  # incompressible, bigger than a part
        (tmp / "run.out").write_bytes(b"debugger summary\n")
        parts = T.pack(str(run), MANIFEST, str(tmp / "p.zip"), T.MANIFEST_ROOM + 300 * 1024, chunk_bytes=128 * 1024, cache_dir=str(cache))
        assert len(parts) >= 3, parts
        with zipfile.ZipFile(parts[0][0]) as z:
            names = z.namelist()
            assert T.RESULTS_NAME in names and T.SUMMARY_NAME in names, names
            assert {"events.txt", "finished.txt", "translate.txt"} <= set(names), names
            files = json.loads(z.read("manifest.json"))["files"]
        assert files[T.RESULTS_NAME]["part"] == 1 and files[T.SUMMARY_NAME]["part"] == 1
        assert max(v["part"] for v in files.values()) >= 3, "the big log spills over into later parts"


def check_request_code_files():
    """Run R: the compute keys came without .c67/.data and graphics keys were missing. Every req_<key>.* file of a chosen key ships
    (compute: .c67 code + .data + state; graphics: .a/.b + state), compute and graphics each have their own ranking and quota,
    and all of them sit in part 1 even when the logs fill several parts."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache = tmp / "cache"
        cache.mkdir()
        rows, weights, stderr = [], {}, []
        for n in range(30):   # 30 compute keys, huge dispatch counts
            key = "c%015x" % n
            for ext in (".c67", ".data", ".desc", ".root", ".sa", ".sb"):
                write(cache, "req_%s%s" % (key, ext), b"C" * 100)
            write(cache, "%s.fail" % key, b"v_readfirstlane_b32 requires a proven uniform source\n")
            stderr.append("x2w-render compute census key %s: %d dispatches, 0 translated, first refusal: x" % (key, 1000000 - n))
        for n in range(20):   # 20 graphics keys, small draw counts
            key = "a%015x" % n
            for ext in (".a", ".b", ".root", ".psin"):
                write(cache, "req_%s%s" % (key, ext), b"G" * 100)
            write(cache, "%s.fail" % key, b"MRTZ requires one uncompressed depth export\n")
            stderr.append("x2w-render not drawn key %s: %d (translation_not_ready)" % (key, 5000 - n))
        (cache / ("req_%s.data" % ("c%015x" % 29))).unlink()   # a compute key whose .data is not in the cache
        cache_rows = T.cache_results(str(cache))
        chosen = T.select_requests(str(cache), cache_rows, T.key_weights("\n".join(stderr)))
        kinds = [c[1] for c in chosen]
        assert len(chosen) == 40 and kinds.count("compute") == 25 and kinds.count("graphics") == 15, (len(chosen), kinds.count("compute"))
        for key, kind, weight, files, missing in chosen:
            names = {os.path.splitext(n)[1] for n, _ in files}
            if kind == "compute":
                assert {".c67", ".data", ".desc", ".root", ".sa", ".sb"} == names and not missing, (key, names)
            else:
                assert {".a", ".b", ".root", ".psin"} == names and not missing, (key, names)
        assert [c[0] for c in chosen[:2]] == ["c000000000000000", "c000000000000001"], "compute keys by dispatches first"
        top_graphics = [c[0] for c in chosen if c[1] == "graphics"]
        assert top_graphics[0] == "a000000000000000" and top_graphics == sorted(top_graphics), "graphics keys by draws"
        # the one that lost its code files is listed with what is missing (here: it is outside the top 25, so use a small family)
        few = T.select_requests(str(cache), cache_rows, T.key_weights("\n".join(stderr)), top=40, compute_share=40)
        broken = [c for c in few if c[0] == "c%015x" % 29]
        assert broken and broken[0][4] == [".data"], broken

        # RTV variants: draws counted under a variant key count on its base, whose code ships with the variants' request files
        base = "a%015x" % 19                       # the weakest graphics key by its own draws (5000 - 19 = 4981 < the others' 4999..)
        for n, (variant, drawn) in enumerate((("b000000000000001", 4000), ("b000000000000002", 3000))):
            write(cache, "req_%s.rtv" % variant, ("rtv1 %d 28" % int(base, 16)).encode())
            stderr.append("x2w-render not drawn key %s: %d (translation_not_ready)" % (variant, drawn))
        chosen = T.select_requests(str(cache), T.cache_results(str(cache)), T.key_weights("\n".join(stderr)))
        graphics = [c for c in chosen if c[1] == "graphics"]
        assert graphics[0][0] == base and graphics[0][2] == 4981 + 7000, graphics[0][:3]
        assert {n for n, _ in graphics[0][3]} == {"req_%s%s" % (base, e) for e in (".a", ".b", ".root", ".psin")} | {"req_b000000000000001.rtv", "req_b000000000000002.rtv"}
        assert all(not c[0].startswith("b") for c in chosen), "a variant is never a key of its own"
        run = tmp / "run"
        run.mkdir()
        make_run(run, 1)
        noise = random.Random(4)
        write(run, "stderr.txt", "\n".join(stderr).encode() + b"\n" + bytes(noise.randrange(256) for _ in range(700 * 1024)))
        parts = T.pack(str(run), MANIFEST, str(tmp / "p.zip"), T.MANIFEST_ROOM + 300 * 1024, chunk_bytes=128 * 1024, cache_dir=str(cache))
        assert len(parts) >= 3, len(parts)
        with zipfile.ZipFile(parts[0][0]) as z:
            first = set(z.namelist())
            files = json.loads(z.read("manifest.json"))["files"]
        want = {T.REQUESTS_DIR + "/" + n for c in chosen for n, _ in c[3]}
        assert want <= first, sorted(want - first)[:5]
        assert any(n.endswith(".c67") for n in first) and any(n.endswith(".data") for n in first)
        assert all(files[n]["part"] == 1 for n in want), "every request file is in part 1"


def check_mail_ready_parts():
    """With the default limit every part is a file the tester can mail as it is (at most 9 MB), whatever the size of the logs,
    and the shader request files stay together in part 1."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache = tmp / "cache"
        cache.mkdir()
        make_cache(cache)
        run = tmp / "run"
        run.mkdir()
        make_run(run, 3)
        write(run, "stderr.txt", b"x2w-render compute census key 2000000000000001: 900 dispatches, 0 translated, first refusal: x\n" + os.urandom(20 * 1024 * 1024))
        parts = T.pack(str(run), MANIFEST, str(tmp / "mail.zip"), cache_dir=str(cache))
        assert len(parts) >= 3, [size for _, size in parts]
        assert all(size <= 9 * 1024 * 1024 for _, size in parts), [size for _, size in parts]
        for index, (path, _) in enumerate(parts, 1):
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                assert ("part_%02d_of_%02d.txt" % (index, len(parts))) in names, names
                if index == 1:
                    assert T.RESULTS_NAME in names and any(n.startswith(T.REQUESTS_DIR + "/") for n in names), names
                    assert "manifest.json" in names


def check_phase_table():
    """phases.txt: presents, fps and picture mean per 30 s window from the stand-in heartbeats and the capture rows."""
    beats = ("x2w-d3d12 heartbeat: 150 frames presented in 5.0 s (30.0 fps overall); x\n"
             "x2w-d3d12 heartbeat: 900 frames presented in 30.0 s (30.0 fps overall); x\n"
             "x2w-d3d12 heartbeat: 1000 frames presented in 60.0 s (16.7 fps overall); x\n"
             "x2w-d3d12 heartbeat: 1010 frames presented in 90.0 s (11.2 fps overall); x\n")
    timeline = ("   10.0 s  picture window_010s.png mean 0.0  first\n   40.0 s  picture window_040s.png mean 85.5  changed\n"
                "   55.0 s  picture window_055s.png mean 36.0  changed\n")
    rows = T.phase_table(beats, timeline).splitlines()
    assert rows[2].split() == ["0", "30", "900", "30.0", "0.0"], rows[2]
    assert rows[3].split() == ["30", "60", "100", "3.3", "36.0"], rows[3]
    assert rows[4].split() == ["60", "90", "10", "0.3", "-"], rows[4]
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        run.mkdir()
        make_run(run, 1)
        write(run, "stderr.txt", beats.encode())
        write(run, "timeline.txt", timeline.encode())
        T.pack(str(run), MANIFEST, str(Path(tmp) / "p.zip"))
        with zipfile.ZipFile(Path(tmp) / "p.zip") as z:
            assert z.read(T.PHASES_NAME).decode().splitlines()[2].split()[:4] == ["0", "30", "900", "30.0"]
        write(run, "stderr.txt", b"no heartbeats\n")
        T.pack(str(run), MANIFEST, str(Path(tmp) / "q.zip"))
        with zipfile.ZipFile(Path(tmp) / "q.zip") as z:
            assert T.PHASES_NAME not in z.namelist(), "no heartbeat, no phases file"


def check_request_families():
    """The top refusal families each bring their 20 keys with the most uses, whatever their dispatch/draw weight (c20: the code of
    the top 20 'branch out of the block' keys), on top of the weight ranking; the byte cap still holds."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        cache.mkdir()
        uses = {}
        for n in range(30):   # 30 keys of one wall, no census weight at all
            key = "d%015x" % n
            for ext in (".c67", ".data", ".root"):
                write(cache, "req_%s%s" % (key, ext), b"B" * 50)
            write(cache, "%s.fail" % key, ("branch out of the block %d is not covered\n" % n).encode())
            uses[key] = (1.0, 2.0, 100 + n)
        for n in range(5):    # a small other wall
            key = "e%015x" % n
            write(cache, "req_%s.c67" % key, b"E" * 50)
            write(cache, "%s.fail" % key, b"v_readfirstlane_b32 requires a proven uniform source\n")
            uses[key] = (1.0, 2.0, 5)
        rows = T.cache_results(str(cache))
        assert T.select_requests(str(cache), rows, {}) == [], "without uses nothing is ranked (no weights)"
        chosen = T.select_requests(str(cache), rows, {}, uses=uses)
        wall = [c[0] for c in chosen if c[0].startswith("d")]
        assert sorted(wall) == sorted("d%015x" % n for n in range(10, 30)), "the 20 most used keys of the wall: %r" % wall
        assert all(any(n.endswith(".c67") for n, _ in c[3]) and any(n.endswith(".data") for n, _ in c[3]) for c in chosen if c[0].startswith("d"))
        assert len([c for c in chosen if c[0].startswith("e")]) == 5, "a second wall brings its keys too"
        assert len(T.select_requests(str(cache), rows, {}, uses=uses, cap=700)) < len(chosen), "the byte cap still holds"
        capped = T.select_requests(str(cache), rows, {}, uses=uses, cap=700)
        assert sum(b for c in capped for _, b in c[3]) <= 700


def check_combine():
    """A matrix report: three runs in one mail-ready report, run_<label>/ folders, a matrix manifest with each run's env."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        runs = []
        for label, extra_env in (("a", {}), ("b", {"X2W_TEST_ADDRESS_UPLOAD_RING": "1"}), ("c", {"X2W_BIG_COPY_REUSE": "1"})):
            run = tmp / ("run_" + label)
            run.mkdir()
            make_run(run, 2)
            write(run, "stderr.txt", ("stderr of " + label + "\n").encode() + os.urandom(3 * 1024 * 1024))
            manifest = dict(MANIFEST, seconds=240, env=dict(MANIFEST["env"], **extra_env), end_cause="timeout", exit_code=0)
            parts = T.pack(str(run), manifest, str(tmp / ("report_" + label + ".zip")))
            runs.append((label, [p for p, _ in parts]))
        out = T.combine(str(tmp / "matrix.zip"), runs, 9 * 1024 * 1024, "gray")
        assert len(out) >= 2 and all(size <= 9 * 1024 * 1024 for _, size in out), out
        seen = {}
        for index, (path, _) in enumerate(out, 1):
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    assert name not in seen, name
                    seen[name] = z.read(name)
                assert ("part_%02d_of_%02d.txt" % (index, len(out))) in z.namelist()
                if index == 1:
                    assert "matrix_manifest.json" in z.namelist() and "run_a/manifest.json" in z.namelist() and "run_c/events.txt" in z.namelist()
        top = json.loads(seen["matrix_manifest.json"])
        assert top["matrix"] == "gray" and sorted(top["runs"]) == ["a", "b", "c"] and top["parts"] == len(out), top
        assert "X2W_TEST_ADDRESS_UPLOAD_RING" not in top["runs"]["a"]["env"] and top["runs"]["b"]["env"]["X2W_TEST_ADDRESS_UPLOAD_RING"] == "1"
        assert top["runs"]["c"]["env"]["X2W_BIG_COPY_REUSE"] == "1" and top["runs"]["a"]["seconds"] == 240 and top["runs"]["a"]["end_cause"] == "timeout"
        for label in "abc":
            assert seen["run_%s/stderr.txt" % label].startswith(("stderr of " + label).encode()), label
            assert json.loads(seen["run_%s/manifest.json" % label])["env"] == top["runs"][label]["env"]
        assert all(top["files"][n]["part"] == 1 for n in top["files"] if n.endswith("manifest.json") and n.startswith("run_")), "manifests in part 1"


def make_old_cache(folder):
    """Five requests of an older revision (3 graphics, 1 compute, 1 mesh 57) with their results, which must not travel."""
    graphics = ("a", "b", "root")
    for n in range(3):
        key = "00000000000000a%d" % n
        for ext in graphics:
            write(folder, "req_%s.%s" % (key, ext), ("graphics %d %s" % (n, ext)).encode())
        write(folder, "req_%s.txt" % key, b"old ready marker")
        write(folder, "req_%s.vs.cso" % key, b"old binary")
        write(folder, "%s.fail" % key, b"old refusal")
    for ext in ("desc", "sa", "sb", "data", "root", "c67"):
        write(folder, "req_00000000000000c0.%s" % ext, ("compute " + ext).encode())
    write(folder, "req_00000000000000c0.txt", b"old ready marker")
    for ext in ("ps57", "root", "m57"):
        write(folder, "req_00000000000000e0.%s" % ext, ("mesh " + ext).encode())
    write(folder, "req_00000000000000e0.ms57.txt", b"old ready marker")


def check_seed_cache():
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old, new = root / "shader_cache_27-27-1", root / "shader_cache_28-28-1"
        old.mkdir()
        new.mkdir()
        make_old_cache(old)
        old_names = sorted(os.listdir(old))

        # Nothing to seed from, or the new cache is already warm: skipped, nothing written, never an error.
        assert T.seed_cache(str(root / "none"), str(root / "none" / "shader_cache_1"))["skipped"] == "no older shader_cache_* folder"
        lonely = root / "lonely"
        (lonely / "shader_cache_1").mkdir(parents=True)
        assert T.seed_cache(str(lonely), str(lonely / "shader_cache_1"))["skipped"] == "no older shader_cache_* folder"

        # The jobs the pre-translation may use reach the re-keying tool: a stub records its environment.
        stub = root / "rekey_stub.py"
        stub.write_text("import os, sys\nopen(os.path.join(sys.argv[2], 'jobs.txt'), 'w').write(os.environ['X2W_TRANSLATE_JOBS'])\n")
        T.seed_cache(str(root), str(new), jobs=3, rekey=str(stub))
        assert (new / "jobs.txt").read_text() == "3"
        os.remove(new / "jobs.txt")

        result = T.seed_cache(str(root), str(new), jobs=2)
        assert result["skipped"] is None and result["sources"] == 1 and result["seeded"] == 5 and not result["failed"], result
        assert T.seed_line(result) == "pre-translate: 5 requests from 1 older cache(s) in %g s" % result["seconds"]
        assert sorted(os.listdir(old)) == old_names, "the old cache is only read"
        names = os.listdir(new)
        # Requests only, under new names, and no result of the old revision.
        assert not [n for n in names if n.endswith((".txt", ".fail", ".cso", ".hlsl", ".tmp"))], names
        assert not [n for n in names if n.startswith(("req_00000000000000a", "req_00000000000000c0", "req_00000000000000e0"))], names
        assert T.request_count(str(new)) == 5, names
        assert sorted(n.rsplit(".", 1)[1] for n in names if n.startswith("req_") and n.endswith((".a", ".c67", ".m57"))) == [
            "a", "a", "a", "c67", "m57"], names

        # The requests are now new-revision requests: this checkout's translator answers every one of them (a refusal
        # is an answer), from its own process pool, and nothing else is written.
        env = dict(os.environ, X2W_TRANSLATE_JOBS="2")
        done = subprocess.run([sys.executable, "-B", str(Path(T.__file__).with_name("gfx2hlsl.py")), "--pending", str(new)],
                              capture_output=True, text=True, env=env, timeout=300)
        assert done.returncode == 0, done.stderr[-500:]
        keys = {n[4:20] for n in os.listdir(new) if n.startswith("req_")}
        answered = {k for k in keys if any((new / name).exists() for name in ("req_%s.txt" % k, "%s.fail" % k, "req_%s.fail" % k))}
        assert len(keys) == 5 and answered == keys, (sorted(keys), sorted(answered), done.stdout[-400:])

        # A seeded cache is not seeded again; a broken old request leaves the new cache untouched and is reported.
        assert T.seed_cache(str(root), str(new))["skipped"] == "the new cache already holds requests"
        broken_root = root / "broken"
        (broken_root / "shader_cache_1").mkdir(parents=True)
        (broken_root / "shader_cache_2").mkdir()
        write(broken_root / "shader_cache_1", "req_00000000000000b0.a", b"only the marker")
        broken = T.seed_cache(str(broken_root), str(broken_root / "shader_cache_2"))
        assert broken["skipped"] == "every older cache was refused" and broken["seeded"] == 0 and len(broken["failed"]) == 1, broken
        assert T.request_count(str(broken_root / "shader_cache_2")) == 0
        assert T.seed_line(broken).startswith("pre-translate: skipped (every older cache was refused); shader_cache_1: error:"), T.seed_line(broken)

        # The command line prints the line and writes the manifest part.
        cli_root = root / "cli"
        (cli_root / "shader_cache_old").mkdir(parents=True)
        make_old_cache(cli_root / "shader_cache_old")
        (cli_root / "shader_cache_new").mkdir()
        out = subprocess.run([sys.executable, "-B", T.__file__, "seed-cache", str(cli_root), str(cli_root / "shader_cache_new"),
                              "--jobs", "2", "--json", str(root / "seed.json")], capture_output=True, text=True)
        assert out.returncode == 0 and out.stdout.startswith("pre-translate: 5 requests from 1 older cache(s) in "), out
        assert json.loads((root / "seed.json").read_text())["seeded"] == 5


MSVC_MAP = """ d3d12_xs

 Timestamp is 6563a1f2 (Wed Nov 26 15:00:02 2026)

 Preferred load address is 0000000180000000

 Start         Length     Name                   Class
 0001:00000000 00a00000H .text$mn                CODE

  Address         Publics by Value              Rva+Base               Lib:Object

 0001:00001000       DllMain                    0000000180001000 f   dllmain.obj
 0001:00001400       ?Draw@Queue@@QEAAXXZ       0000000180001400 f   d3d12_draw.obj
 0001:00001800       g_table                    0000000180001800     d3d12_draw.obj
 0001:00002000       ?Present@Queue@@QEAAXXZ    0000000180002000 f i present.obj
"""


def check_symbols():
    import subprocess
    rows = T.parse_symbols(MSVC_MAP)  # functions only (flag f), RVA = address - preferred load address
    assert rows == [(0x1000, "DllMain"), (0x1400, "?Draw@Queue@@QEAAXXZ"), (0x2000, "?Present@Queue@@QEAAXXZ")], rows
    nm = "0000000000001000 T alpha\n0000000000001200 t beta\n0000000000001300 D data\n0000000000001000 T alias\n"
    assert T.parse_symbols(nm) == [(0x1000, "alpha"), (0x1200, "beta")]
    assert T.load_symbols(T.format_symbols(rows)) == rows
    try:
        T.parse_symbols("not a map\n")
        raise AssertionError("a text without function symbols must be refused")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run = tmp / "run"
        run.mkdir()
        make_run(run, 1)
        (tmp / "d3d12_xs.map").write_text(MSVC_MAP)
        (tmp / "m.json").write_text(json.dumps(MANIFEST))
        out = subprocess.run([sys.executable, "-B", T.__file__, "pack", str(run),
                              str(tmp / "m.json"), str(tmp / "r.zip"), "--map", str(tmp / "d3d12_xs.map")],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        with zipfile.ZipFile(tmp / "r.zip") as z:
            assert z.read(T.SYMBOLS_NAME).decode() == "1000 DllMain\n1400 ?Draw@Queue@@QEAAXXZ\n2000 ?Present@Queue@@QEAAXXZ\n"
            assert json.loads(z.read("manifest.json"))["symbols"] == 3
        T.pack(str(run), MANIFEST, str(tmp / "plain.zip"))
        with zipfile.ZipFile(tmp / "plain.zip") as z:
            assert T.SYMBOLS_NAME not in z.namelist()  # no map: no symbol member


def check_producer_report():
    config = "x2w-red auto configured floor=40 frames=2 rearm=3\n"
    waiting = config + "x2w-red auto waiting present=33\n"
    assert "floor not reached: last present 33, floor 40" in T.producer_report(waiting)
    assert "floor not reached: last present 33, floor 40s" in T.producer_report(waiting.replace("floor=40", "floor=40s"))
    reached = waiting + "x2w-red auto floor reached present=40\n"
    assert "no flat onset seen" in T.producer_report(reached)
    capture = "x2w-red auto armed sampled_present={} published_at={} target=123 capture={}..{} rgb={} ordinal={}\n"
    armed = reached + capture.format(45, 46, 47, 48, "255,255,255", 1) + capture.format(55, 56, 57, 58, "0,0,0", 2)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        log = tmp / "stderr.txt"
        log.write_text(waiting, encoding="utf-8")
        T.pack(str(tmp), MANIFEST, str(tmp / "waiting.zip"))
        with zipfile.ZipFile(tmp / "waiting.zip") as z:
            assert "floor not reached: last present 33, floor 40" in z.read(T.PRODUCER_NAME).decode()
        log.write_text(armed, encoding="utf-8")
        T.pack(str(tmp), MANIFEST, str(tmp / "report.zip"))
        with zipfile.ZipFile(tmp / "report.zip") as z:
            report = z.read(T.PRODUCER_NAME).decode()
    assert report.splitlines()[1:] == [
        "capture 1: RGB 255,255,255, onset present 45, frames 47..48",
        "capture 2: RGB 0,0,0, onset present 55, frames 57..58",
    ]
    assert not T.producer_report("unrelated log\n")


if __name__ == "__main__":
    check_symbols()
    check_seed_cache()
    check_cache_results()
    check_key_use()
    check_request_files()
    check_request_code_files()
    check_mail_ready_parts()
    check_phase_table()
    check_request_families()
    check_combine()
    check_part_one_has_the_small_files()
    check_preset()
    check_privacy_screen()
    check_manifest()
    check_first_poll()
    check_end_cause()
    check_producer_report()
    check_cache_stats()
    check_cache_tag()
    check_memory_gib()
    check_zip()
    print("startup_tester_pack_check: preset, manifest and zip checks pass")
