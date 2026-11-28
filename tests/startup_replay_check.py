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
    supplies the render target. Changing the last destination handle in the exact arrays (DARR) leaves the fallback clear visible;
  - a target made through device slot 127 (ten argument words) replays equal (format 3 records eleven);
  - opt-in player timing reports every recorded slot and accounts for at least 90% of playback wall time;
  - control: the same recording with the last frame's clear colour changed in its memory replays to a different last
    picture, and the other pictures stay equal (the replay draws from the recording, and the comparison sees it).

usage: startup_replay_check.py <startup_replay_quad> <x2w_replay> <warp|hardware> [slot-layouts|null-views]
"""
import json
import os
import re
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


def change_last_clear(source, target, clear_count=PRESENTS):
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
    assert len(clears) == clear_count, clears
    at = blobs[argm[(clears[-1], 1)]]
    struct.pack_into("<f", data, at, 1.0 - struct.unpack_from("<f", data, at)[0])
    target.write_bytes(bytes(data))


def run(args, env):
    result = subprocess.run([str(a) for a in args], env=env, capture_output=True, text=True, timeout=300)
    return result.returncode, result.stdout + result.stderr


def check_timing(summary, recording, progress):
    data = recording.read_bytes()
    expected = {struct.unpack_from("<QIHH", data, header)[2:]
                for tag, header, *_ in chunks(data) if tag == b"CALL"}
    slots = summary["slot_times"]
    keys = [(entry["kind"], entry["slot"]) for entry in slots]
    assert len(keys) == len(expected) and set(keys) == expected, keys
    assert all(entry["total_us"] >= 0 for entry in slots), slots
    # Per-call cost needs the call count: every recorded call is counted once, under its own slot.
    counts = {}
    for tag, header, *_ in chunks(data):
        if tag == b"CALL":
            key = struct.unpack_from("<QIHH", data, header)[2:]
            counts[key] = counts.get(key, 0) + 1
    assert {(e["kind"], e["slot"]): e["calls"] for e in slots} == counts, (slots, counts)
    slot_us = sum(entry["total_us"] for entry in slots)
    wall_us = summary["wall_s"] * 1_000_000
    assert wall_us > 0 and abs(slot_us - wall_us) <= 0.10 * wall_us, (slot_us, wall_us)

    if slot_us >= 1_000_000:
        assert progress, "no timing report after a second of playback"
    calls = presents = 0
    for line in progress:
        _, _, calls_label, call_count, presents_label, present_count, top_label, *top = line.split()
        assert (calls_label, presents_label, top_label) == ("calls", "presents", "top_us"), line
        calls += int(call_count)
        presents += int(present_count)
        assert 1 <= len(top) <= 5, line
        times = []
        for entry in top:
            key, duration = entry.split("=")
            assert tuple(map(int, key.split(":"))) in expected, line
            times.append(float(duration))
        assert times == sorted(times, reverse=True) and all(t >= 0 for t in times), line
    assert calls <= summary["played"] and presents <= summary["presents"], progress


def replay(player, recording, out, backend, env, presents=PRESENTS, timing=False, extra=()):
    # No X2W_ setting: the stand-in's settings (XENV) and its shader cache (SHDR) come from the recording.
    env = {k: v for k, v in env.items() if not k.startswith("X2W_")}
    if timing:
        env["X2W_REPLAY_TIMING"] = "1"
    code, text = run([player, recording, out, "--adapter", backend, *extra], env)
    if code:
        log = (out / "stderr.txt").read_text(errors="replace")[-3000:] if (out / "stderr.txt").exists() else ""
        raise AssertionError("x2w_replay exit %d on %s (%d bytes)\n%s\n%s" % (code, recording.name, recording.stat().st_size,
                                                                        text[-2000:], log))
    score = [line for line in text.splitlines() if line.startswith("x2w-replay score ")]
    assert len(score) == 1, text
    summary = json.loads((out / "replay_summary.json").read_text())
    assert summary["unknown_object"] == 0 and summary["missing_blob"] == 0, summary
    assert summary["presents"] == presents and summary["fence_timeouts"] == 0, summary
    # The draw scoreboard counts every draw, not the capped log lines: one entry per present, adding up to the totals.
    draws = summary["draws"]
    assert len(draws["per_present"]) == presents, draws
    assert sum(entry["drawn"] for entry in draws["per_present"]) <= draws["drawn"], draws
    assert sum(draws["not_drawn_by_reason"].values()) <= draws["not_drawn"], draws
    assert any(line.startswith("x2w-replay draws drawn %d not_drawn %d " % (draws["drawn"], draws["not_drawn"]))
               for line in text.splitlines()), (draws, text[-500:])
    assert draws["magenta_draws"] == 0 and " magenta draws 0" in text, draws  # the switch is off: no placeholder draw
    progress = [line for line in text.splitlines() if line.startswith("x2w-replay timing ")]
    if timing:
        check_timing(summary, recording, progress)
    else:
        assert "slot_times" not in summary and not progress, (summary, progress)
    return score[0], pictures(out)


def check_slot_layouts(scene, player, backend, env, directory, vs, ps, root):
    # The first wait is in the prefix. The list no-op enters the log only after the window opens.
    frames = 3
    recorded_dir = directory / "recorded"
    recorded_dir.mkdir()
    recording = directory / "slots.x2wr"
    code, text = run([scene, vs, ps, root, backend, frames],
                     dict(env, X2W_REPLAY_RECORD="f1,2", X2W_REPLAY_RECORD_FILE=str(recording),
                          X2W_REPLAY_SLOT_LAYOUTS="1", X2W_PRESENT_PICTURES=str(recorded_dir)))
    assert code == 0, text[-3000:]
    data = bytearray(recording.read_bytes())
    calls, call_headers, memory, blobs = {}, {}, {}, {}
    for tag, header, _hb, payload, size in chunks(data):
        if tag == b"CALL":
            call = CALL.unpack_from(data, header)
            calls[call[0]] = call
            call_headers[call[0]] = header
        elif tag == b"ARGM":
            seq, arg, depth, _address, length, blob_hash = struct.unpack_from("<QIIQQQ", data, header)
            memory.setdefault(seq, []).append((arg, depth, length, blob_hash))
        elif tag == b"BLOB":
            blobs[struct.unpack_from("<Q", data, header)[0]] = (payload, size)

    def sequences(kind, slot):
        return sorted(seq for seq, call in calls.items() if call[2:4] == (kind, slot))

    waits, barriers = sequences(0, 80), sequences(LIST_KIND, 146)
    assert len(waits) == 3 and len(barriers) == 2, (waits, barriers)
    assert all(not memory.get(seq) for seq in barriers), barriers
    for expected, seq in enumerate(waits, 1):
        [(arg, depth, length, blob_hash)] = memory[seq]
        assert (arg, depth, length) == (4, 0, 8), memory[seq]
        at, size = blobs[blob_hash]
        assert size == 8 and struct.unpack_from("<Q", data, at)[0] == expected, seq

    memory_views, uploads = sequences(0, 70), sequences(LIST_KIND, 148)
    assert len(memory_views) == 3 and len(uploads) == 2, (memory_views, uploads)
    assert all(not memory.get(seq) for seq in uploads), uploads
    for seq in memory_views:
        assert [(arg, depth, length) for arg, depth, length, _hash in memory[seq]] == [(0, 0, 56), (1, 0, 48)]

    [list_create] = sequences(0, 73)
    assert [(arg, length) for arg, _depth, length, _hash in memory[list_create]] == [(0, 16), (3, 16), (4, 8)]
    pipeline_creates = sequences(0, 86)
    assert len(pipeline_creates) == 2, pipeline_creates
    for seq in pipeline_creates:
        args = memory[seq]
        assert [(arg, length) for arg, _depth, length, _hash in args if arg < 256] == [(0, 16), (1, 16), (2, 8)]
        [stream_length] = [length for arg, depth, length, _hash in args if (arg, depth) == (512, 1)]
        desc_at, _ = blobs[args[0][3]]
        assert stream_length == struct.unpack_from("<Q", data, desc_at)[0], args

    out = directory / "replayed"
    _score, rendered = replay(player, recording, out, backend, env, presents=2)
    recorded = pictures(recorded_dir)
    assert sorted(rendered) == ["present_0001.png", "present_0002.png"]
    for frame in (1, 2):
        assert rendered["present_%04d.png" % frame] == recorded["present_%04d.png" % (frame + 1)], frame
    summary = json.loads((out / "replay_summary.json").read_text())
    assert summary["slot_layouts"] == {"frame_waits": 3, "barrier_noops": 2, "upload_ranges": 2, "result_mismatches": 0}, summary

    # Mutate both expected results. The calls and rendered pixels should be unaffected.
    struct.pack_into("<Q", data, at, 99)
    ret_offset = CALL.size - struct.calcsize("<Qdii")
    struct.pack_into("<Q", data, call_headers[barriers[-1]] + ret_offset, 1)
    mutant = directory / "bad-results.x2wr"
    mutant.write_bytes(data)
    out = directory / "mutant"
    clean_env = {k: v for k, v in env.items() if not k.startswith("X2W_")}
    code, text = run([player, mutant, out, "--adapter", backend], clean_env)
    assert code == 6, text[-3000:]
    log = (out / "stderr.txt").read_text(errors="replace")
    assert f"WaitFrameEvent result mismatch at seq {waits[-1]}" in log, log[-3000:]
    assert f"BarrierLikeNoop result mismatch at seq {barriers[-1]}" in log, log[-3000:]
    assert pictures(out) == rendered
    print("slot layouts: 3 matching waits including the prefix, 2 matching no-ops, 2 matching pictures, both mutations detected")
    return 0


def check_null_views(scene, player, backend, env, directory, vs, ps, root):
    frames = 3
    recorded_dir = directory / "recorded"
    recorded_dir.mkdir()
    recording = directory / "null_views.x2wr"
    code, text = run([scene, vs, ps, root, backend, frames],
                     dict(env, X2W_TEST_REPLAY_NULL_VIEWS="1", X2W_REPLAY_PAGES="0",
                          X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                          X2W_PRESENT_PICTURES=str(recorded_dir)))
    assert code == 0, text[-3000:]
    data = recording.read_bytes()
    backing, calls = [], []
    for tag, header, header_bytes, _payload, size in chunks(data):
        if tag == b"DHPM":
            assert header_bytes == 24 and size == 0
            backing.append(struct.unpack_from("<QQQ", data, header))
        elif tag == b"CALL":
            calls.append(CALL.unpack_from(data, header))
        assert tag not in (b"PAGE", b"MAP "), tag
    [(seq, address, length)] = backing
    [create] = [call for call in calls if call[0] == seq]
    assert create[2:4] == (0, 14), create
    null_rtv, render_rtv = sorted(call for call in calls if call[2:4] == (0, 20))
    assert seq < null_rtv[0] < render_rtv[0] and null_rtv[5] == 0 and render_rtv[5] != 0
    assert length > 0 and (null_rtv[7], render_rtv[7]) == (address, address + length // 2)
    out = directory / "replayed"
    _score, rendered = replay(player, recording, out, backend, env, presents=frames)
    summary = json.loads((out / "replay_summary.json").read_text())
    [reserved] = summary["ranges"]
    base = int(reserved["address"], 16)
    assert base <= address < address + length <= base + reserved["bytes"], reserved
    recorded = pictures(recorded_dir)
    assert len(recorded) == frames and len(set(recorded.values())) > 1
    assert recorded == rendered, "null_views replay differs from the recorded scene"
    print("null_views: stack-backed RTV heap replayed with all three pictures equal")
    return 0


def check_paced(scene, player, backend, env, directory, vs, ps, root):
    """A scene that waits for a 100 ms frame clock before every frame (device slots 78 and 80) replays to the same
    pictures without sleeping in those waits, and with --pace on it sleeps in them again."""
    recorded_dir, recording = directory / "paced_recorded", directory / "paced.x2wr"
    recorded_dir.mkdir()
    code, text = run([scene, vs, ps, root, backend, PRESENTS],
                     dict(env, X2W_TEST_REPLAY_PACED="1", X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                          X2W_PRESENT_PICTURES=str(recorded_dir)))
    assert code == 0, text[-3000:]
    recorded = pictures(recorded_dir)
    out = directory / "unpaced"
    _score, rendered = replay(player, recording, out, backend, env)
    assert rendered == recorded, "the unpaced replay differs from the recorded scene"
    waits = json.loads((out / "replay_summary.json").read_text())["frame_event_waits"]
    assert waits["calls"] == waits["not_paced"] == PRESENTS and waits["wall_s"] < 0.1 * PRESENTS * 0.1, waits
    paced = directory / "paced"
    code, text = run([player, recording, paced, "--adapter", backend, "--pace", "on"],
                     {k: v for k, v in env.items() if not k.startswith("X2W_")})
    assert code == 0, text[-3000:]
    assert pictures(paced) == recorded, "the paced replay differs from the recorded scene"
    slept = json.loads((paced / "replay_summary.json").read_text())["frame_event_waits"]
    # Each wait sleeps up to the 100 ms interval minus the frame's own work.
    assert slept["calls"] == PRESENTS and slept["not_paced"] == 0 and slept["wall_s"] > 0.3, slept
    log = (out / "stderr.txt").read_text(errors="replace")
    assert "x2w-replay: frame waits not paced: %d of %d" % (PRESENTS, PRESENTS) in log, log[-2000:]
    print("paced: %d frame waits replayed in %.3f s without pacing, %.3f s with --pace on, pictures equal"
          % (PRESENTS, waits["wall_s"], slept["wall_s"]))
    return 0


def check_placed_window(scene, player, backend, env, directory, vs, ps, root):
    """The target placed by device slot 133 at a title address (no desktop heap) in frame 0, a window opened mid-run
    (f6,6): the file carries the placement's description, IID and output, and presents 7-12 replay equal. The same
    prefix asks device slot 108 for two descriptions' allocation info (returned by value, arrays on the title's stack):
    the file carries both arrays, so the replay passes no recorded stack address."""
    recorded_dir, recording = directory / "placed_recorded", directory / "placed.x2wr"
    recorded_dir.mkdir()
    code, text = run([scene, vs, ps, root, backend, PRESENTS],
                     dict(env, X2W_REPLAY_RECORD="f6,6", X2W_REPLAY_RECORD_FILE=str(recording),
                          X2W_PRESENT_PICTURES=str(recorded_dir), X2W_TEST_REPLAY_CREATE133="1",
                          X2W_TEST_REPLAY_ALLOC108="1"))
    assert code == 0, text[-3000:]
    data = recording.read_bytes()
    placements, infos, memory, outputs, sizes, maps = set(), set(), set(), set(), {}, []
    for tag, header, _hb, _payload, _size in chunks(data):
        if tag == b"MAP ":
            maps.append(struct.unpack_from("<QQ", data, header))
        if tag == b"CALL":
            seq, _tid, kind, slot = struct.unpack_from("<QIHH", data, header)
            if (kind, slot) == (0, 133):
                placements.add(seq)
            if (kind, slot) == (0, 108):
                infos.add(seq)
        elif tag in (b"ARGM", b"OUTP"):
            seq, arg = struct.unpack_from("<QI", data, header)
            (memory if tag == b"ARGM" else outputs).add((seq, arg))
            if tag == b"ARGM":
                sizes[(seq, arg)] = struct.unpack_from("<QIIQQ", data, header)[4]
    assert len(infos) == 1, infos
    [info] = infos
    # Two D3D12_RESOURCE_DESC (56 bytes) and two D3D12_RESOURCE_ALLOCATION_INFO1 (24 bytes).
    assert (sizes.get((info, 3)), sizes.get((info, 4))) == (112, 48), (info, sorted(memory))
    assert len(placements) == 1, placements
    [seq] = placements
    assert {(seq, 1), (seq, 4)} <= memory and (seq, 5) in outputs, (seq, sorted(memory), sorted(outputs))
    # The texture's title memory is in the pages although the stand-in notes no size for a texture placed at an address:
    # the recorder sizes it with the XG footprint.
    target = 0x21000000000
    assert any(address <= target < address + size for address, size in maps), [hex(a) for a, _ in maps]
    whole = pictures(recorded_dir)
    _score, played = replay(player, recording, directory / "placed_replayed", backend, env, presents=6)
    pairs = [("present_%04d.png" % k, "present_%04d.png" % (k + 6)) for k in range(1, 7)]
    differ = [b for a, b in pairs if played.get(a) != whole[b]]
    assert not differ, "placed-target window replay differs at %s\n%s" % (
        differ, explain(whole[differ[0]], played.get("present_%04d.png" % (int(differ[0][8:12]) - 6), ((0, 0), b"")),
                        directory / "placed_replayed" / "stderr.txt"))
    print("replay: slot-133 target made before window f6,6 replays presents 7-12 equal")


def check_prefix_live(scene, player, backend, env, directory, vs, ps, root):
    """--prefix-live on: a window opened mid-run (f6,6) whose prefix wrote the target's view twice (a null view first),
    mapped an upload buffer and asked its GPU address every frame, and asked for an allocation size (device 108). The
    replay leaves the null view, the repeated Map, Unmap and GPU address calls and the query out, and presents 7-12 are
    the same pictures as the recorded run and as the replay of the whole prefix."""
    recorded_dir, recording = directory / "live_recorded", directory / "live.x2wr"
    recorded_dir.mkdir()
    code, text = run([scene, vs, ps, root, backend, PRESENTS],
                     dict(env, X2W_REPLAY_RECORD="f6,6", X2W_REPLAY_RECORD_FILE=str(recording),
                          X2W_PRESENT_PICTURES=str(recorded_dir), X2W_TEST_REPLAY_PREFIX_LIVE="1",
                          X2W_TEST_REPLAY_ALLOC108="1"))
    assert code == 0, text[-3000:]
    whole = pictures(recorded_dir)
    pairs = [("present_%04d.png" % k, "present_%04d.png" % (k + 6)) for k in range(1, 7)]
    results = {}
    for name, extra in (("live_full", ()), ("live_cut", ("--prefix-live", "on"))):
        _score, played = replay(player, recording, directory / name, backend, env, presents=6, extra=extra)
        differ = [b for a, b in pairs if played.get(a) != whole[b]]
        assert not differ, "%s window replay differs at %s\n%s" % (
            name, differ, explain(whole[differ[0]], played.get("present_%04d.png" % (int(differ[0][8:12]) - 6), ((0, 0), b"")),
                                  directory / name / "stderr.txt"))
        results[name] = json.loads((directory / name / "replay_summary.json").read_text())["prefix"]
    full, cut = results["live_full"], results["live_cut"]
    log = (directory / "live_cut" / "stderr.txt").read_text(errors="replace")
    counts = re.search(r"prefix live: of (\d+) prefix calls, (\d+) descriptor writes .*?, (\d+) repeated .*? and (\d+) "
                       r"allocation-size queries", log)
    assert counts, log[-3000:]
    calls, writes, repeats, queries = map(int, counts.groups())
    # One null view; frames 1-6 map, unmap and ask the GPU address: all but the first of each are repeats.
    assert writes >= 1 and repeats >= 3 * 5 - 1 and queries >= 1, counts.group(0)
    assert full["left_out"] == 0 and cut["left_out"] == writes + repeats + queries, (full, cut)
    assert cut["calls"] == full["calls"] == calls and cut["played"] < full["played"], (full, cut)
    print("replay: --prefix-live left out %d of %d prefix calls, window f6,6 presents 7-12 equal" % (cut["left_out"], calls))


def check_base_slices(scene, player, backend, env, directory, vs, ps, root):
    """The base image of title memory is written before the window, a slice per present (X2W_REPLAY_BASE_SCAN_MS),
    so the title never stops for all of it in one present. A 3 MiB title buffer and slices of one 1 MiB slab: the window
    f6,6 waits two presents or more, then opens and still records its six presents, the buffer's pages all come before
    them, and the window replays equal to the recorded presents it holds."""
    recorded_dir, recording = directory / "slices_recorded", directory / "slices.x2wr"
    recorded_dir.mkdir()
    code, text = run([scene, vs, ps, root, backend, PRESENTS + 8],
                     dict(env, X2W_REPLAY_RECORD="f6,6", X2W_REPLAY_RECORD_FILE=str(recording),
                          X2W_PRESENT_PICTURES=str(recorded_dir), X2W_TEST_REPLAY_TITLE_BUFFER="1",
                          X2W_REPLAY_BASE_SCAN_MS="0.001"))
    assert code == 0, text[-3000:]
    waited = re.search(r"the base image took ([0-9.]+) s and (\d+) presents before the window: the window opens at present (\d+)",
                       text)
    assert waited and int(waited.group(2)) >= 2, text[-3000:]
    shift = int(waited.group(2))
    assert shift <= 8, waited.group(0)
    data = recording.read_bytes()
    presents, maps = [], []
    for tag, header, _hb, _payload, _size in chunks(data):
        if tag == b"PRES":
            presents.append(struct.unpack_from("<Q", data, header)[0])
        if tag == b"MAP ":
            maps.append(struct.unpack_from("<QQQ", data, header))
    assert len(presents) == 6, (len(presents), waited.group(0))
    base, size = 0x21100000000, 3 << 20
    covered = sorted((max(a, base), min(a + n, base + size)) for a, n, stamp in maps
                     if stamp < presents[0] and a < base + size and a + n > base)
    reach = base
    for start, end in covered:
        if start <= reach:
            reach = max(reach, end)
    assert reach == base + size, "the title buffer's base image reaches %#x of %#x before the window" % (reach, base + size)
    whole = pictures(recorded_dir)
    _score, played = replay(player, recording, directory / "slices_replayed", backend, env, presents=6)
    pairs = [("present_%04d.png" % k, "present_%04d.png" % (k + 6 + shift)) for k in range(1, 7)]
    differ = [b for a, b in pairs if played.get(a) != whole[b]]
    assert not differ, "base-slices window replay differs at %s" % differ
    print("replay: base image before window f6,6 in slices over %d presents, 6 presents recorded and equal" % shift)


def main(scene, player, backend, mode=None):
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
        for name in ("X2W_REPLAY_RECORD", "X2W_REPLAY_RECORD_FILE", "X2W_PRESENT_PICTURES", "X2W_REPLAY_SLOT_LAYOUTS",
                     "X2W_TEST_REPLAY_DESCRIPTOR_WINDOW", "X2W_TEST_REPLAY_NULL_VIEWS"):
            env.pop(name, None)
        code, text = run([scene, vs, ps, root, backend, 1], env)
        if code == 77 and backend == "hardware":
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        assert code == 0, text[-3000:]
        subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)

        if mode == "slot-layouts":
            return check_slot_layouts(scene, player, backend, env, directory, vs, ps, root)
        if mode == "null-views":
            return check_null_views(scene, player, backend, env, directory, vs, ps, root)
        if mode == "paced":
            return check_paced(scene, player, backend, env, directory, vs, ps, root)

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

        score, replayed = replay(player, recording, directory / "replayed", backend, env, timing=True)
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
                              X2W_TEST_REPLAY_DESCRIPTOR_WINDOW="1",
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
        arrays, source_calls = [], []
        control = bytearray()
        for tag, header, _hb, payload, size in chunks(data):
            if tag == b"DARR":
                arrays.append((struct.unpack_from("<Q", data, header)[0], size))
                # The control points the last destination handle at the one before it in the exact arrays: the replay runs the
                # copy from them, so the descriptor the frames draw through is never written.
                swapped = bytearray(data[payload:payload + size])
                swapped[39 * 8:40 * 8] = swapped[38 * 8:39 * 8]
                control.extend(data[header - 16:payload])
                control.extend(swapped)
                continue
            if tag == b"CALL":
                seq, _tid, kind, slot = struct.unpack_from("<QIHH", data, header)
                if kind == 0 and slot in (20, 23):
                    source_calls.append(seq)
            control.extend(data[header - 16:payload + size])
        assert len(arrays) == 1 and arrays[0][1] == 40 * 8 + 8 + 4, arrays
        assert len(source_calls) == 41 and arrays[0][0] == source_calls[-1], (len(source_calls), arrays)
        missing = directory / "window_without_snapshot.x2wr"
        missing.write_bytes(control)
        _score, without_snapshot = replay(player, missing, directory / "window_without_snapshot", backend, env, presents=6)
        for a, b in pairs:
            assert without_snapshot[a] != whole[b], "changing the exact descriptor arrays did not change " + b
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
        # A failure names the pictures and shows the first one's rows and the replay log, so a rare difference is diagnosable.
        assert not differ, "slot-127 replay differs at %s\n%s" % (differ, explain(
            wide_recorded[differ[0]], wide_played.get(differ[0], ((0, 0), b"")), directory / "wide_replayed" / "stderr.txt")
            if differ else "")
        check_placed_window(scene, player, backend, env, directory, vs, ps, root)
        check_prefix_live(scene, player, backend, env, directory, vs, ps, root)
        check_base_slices(scene, player, backend, env, directory, vs, ps, root)
        print("replay: %d presents equal to the recorded run; changed recording differs at the last present only; "
              "window f6,6 replays presents 7-12 equal and all six differ when the exact descriptor arrays are changed; %s" %
              (PRESENTS, score))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
