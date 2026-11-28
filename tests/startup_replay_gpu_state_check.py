"""Check that the GPU snapshot restores prefix writes and preserves timed windows."""

import json
import mmap
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

from startup_replay_check import CLEAR_SLOT, LIST_KIND, chunks, pictures, replay, run
from startup_stream_draw_check import pixel, vertex


WINDOW_PRESENTS = 2
PREFIX_PRESENTS = 2
COLOUR = bytes((255, 0, 255, 255))
ZERO = bytes(4)
COPY_RESOURCE_SLOT = 17
GPUR = struct.Struct("<QIIQIIIIQ")


def check_timed_window(folder, scene, vs, ps, root, env):
    window_seconds = 0.5
    big_mib = 1024
    recording = folder / "gpu-timed.x2wr"
    code, log = run([scene, vs, ps, root, "warp", 50], dict(
        env, X2W_TEST_REPLAY_PACED="1", X2W_TEST_REPLAY_GPU_STATE_BIG=str(big_mib),
        X2W_REPLAY_GPU_STATE_MAX_MIB=str(big_mib + 16),
        X2W_REPLAY_RECORD=f"3,{window_seconds}", X2W_REPLAY_RECORD_FILE=str(recording)))
    assert code == 0, log[-3000:]
    opened = re.findall(r"x2w-replay record: window open at [\d.]+ s, present (\d+), ends at [\d.]+ s", log)
    assert len(opened) == 1 and int(opened[0]) > PREFIX_PRESENTS, (opened, log[-3000:])
    snapshot = re.search(r"x2w-replay gpu state: [^\r\n]* took (\d+) ms", log)
    assert snapshot, log[-3000:]
    snapshot_seconds = int(snapshot[1]) / 1000
    print(f"Timed GPU state: {big_mib} MiB buffer, snapshot took {snapshot_seconds:.3f} s")

    # Walk the chunk headers without copying the 1 GiB payload into Python.
    presents = []
    big_slices = 0
    with recording.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for tag, header, _, _, size in chunks(data):
            if tag == b"PRES":
                presents.append(struct.unpack_from("<i", data, header + 8)[0])
            elif tag == b"GPUR" and size == big_mib << 20:
                big_slices += 1
    assert big_slices == 1, (big_slices, log[-3000:])
    if snapshot_seconds <= window_seconds:
        print(f"SKIP timed GPU state: snapshot must exceed the {window_seconds:.3f} s window")
        return 77

    assert presents and presents[0] == int(opened[0]), (opened, presents)
    assert len(presents[1:]) >= 4, f"Only {len(presents)} window presents after a {snapshot_seconds:.3f} s snapshot: {presents}"
    shifts = re.findall(
        r"x2w-replay record: the GPU state took ([\d.]+) s: the window now ends at ([\d.]+) s \(was ([\d.]+) s\)", log)
    assert len(shifts) == 1, log[-3000:]
    took, new_end, old_end = map(float, shifts[0])
    assert abs(new_end - old_end - took) <= 0.001, shifts
    print(f"Timed GPU state: {len(presents) - 1} presents after the opening present")
    return 0


TINY_BUDGET_MIB = "0.006"  # 6144 bytes: 5529 usable, so a 4096-byte staging buffer
STARVED_BUDGET_MIB = "0.001"
STAGING_LIMIT_KIB = 4


def assert_bounded_staging(log, limit_kib, skipped):
    found = re.search(r"x2w-replay gpu state: (\d+) skipped \(vram budget\), \d+ skipped \(time limit\), "
                      r"peak staging (\d+) KiB", log)
    assert found, log[-3000:]
    assert int(found[1]) == skipped, (found[0], log[-3000:])
    assert int(found[2]) <= limit_kib, f"staging {found[2]} KiB is over the {limit_kib} KiB the budget allows: {found[0]}"


def check_starved(folder, scene, vs, ps, root, env):
    """With no video memory left to stage in, the snapshot is skipped and the title is not touched."""
    recording = folder / "gpu-starved.x2wr"
    code, log = run([scene, vs, ps, root, "warp", PREFIX_PRESENTS + WINDOW_PRESENTS], dict(
        env, X2W_REPLAY_RECORD="f2,2", X2W_REPLAY_RECORD_FILE=str(recording),
        X2W_TEST_VRAM_BUDGET_MIB=STARVED_BUDGET_MIB))
    assert code == 0, log[-3000:]
    assert_bounded_staging(log, 0, skipped=3)
    assert not [c for c in chunks(recording.read_bytes()) if c[0] == b"GPUR"], "a starved snapshot wrote resources"
    print("GPU state: with no budget the snapshot skips all 3 resources and logs them")


def assert_colour(picture, colour):
    (width, height), scanlines = picture
    assert scanlines == (b"\0" + colour * width) * height, (colour, scanlines[:17])


def check(folder, scene, player):
    vs, ps, root = (folder / name for name in ("vs", "ps", "root"))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(False))
    cache = folder / "cache"
    cache.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("X2W_")}
    env.update(X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_TEST_REPLAY_GPU_STATE="1", X2W_REPLAY_PAGES="0", X2W_SHADER_CACHE=str(cache))
    live_reference = None
    for label, extra in (("default", {}), ("off", {"X2W_REPLAY_GPU_STATE": "0"}),
                         ("tiny-budget", {"X2W_TEST_VRAM_BUDGET_MIB": TINY_BUDGET_MIB})):
        enabled = label != "off"
        recording = folder / f"gpu-{label}.x2wr"
        original = folder / f"original-{label}"
        original.mkdir()
        record_env = dict(env, X2W_REPLAY_RECORD="f2,2", X2W_REPLAY_RECORD_FILE=str(recording),
                          X2W_PRESENT_PICTURES=str(original))
        record_env.update(extra)
        code, log = run([scene, vs, ps, root, "warp", PREFIX_PRESENTS + WINDOW_PRESENTS],
                        record_env)
        if code == 77:
            return 77
        assert code == 0, log[-3000:]
        live = pictures(original)
        assert len(live) == PREFIX_PRESENTS + WINDOW_PRESENTS, live.keys()
        if label == "tiny-budget":
            # 90% of the simulated budget is smaller than a texture: the copies go through a 4 KiB staging buffer in pieces.
            assert_bounded_staging(log, STAGING_LIMIT_KIB, skipped=0)
        expected = {
            f"present_{index:04d}.png": live[f"present_{index + PREFIX_PRESENTS:04d}.png"]
            for index in range(1, WINDOW_PRESENTS + 1)
        }
        for picture in expected.values():
            assert_colour(picture, COLOUR)

        data = recording.read_bytes()
        slices, list_slots = [], []
        for tag, header, hb, payload, size in chunks(data):
            if tag == b"GPUR":
                slices.append((header, hb, payload, size))
            elif tag == b"CALL":
                kind, slot = struct.unpack_from("<HH", data, header + 12)
                if kind == LIST_KIND:
                    list_slots.append(slot)
        assert CLEAR_SLOT not in list_slots and list_slots.count(COPY_RESOURCE_SLOT) == 2 * WINDOW_PRESENTS, list_slots
        if enabled:
            # The two RTs and the prefix copy destination exist at the boundary. The window copy texture does not.
            assert len(slices) == 3, slices
            first_pixels = []
            for header, hb, payload, size in slices:
                assert hb == GPUR.size, hb
                obj, subresource, fmt, offset, width, height, depth, pitch, length = GPUR.unpack_from(data, header)
                assert obj and (subresource, fmt, offset, depth) == (0, 28, 0, 1)
                assert (width, height) == next(iter(expected.values()))[0]
                assert pitch >= width * 4 and pitch % 256 == 0 and length == size
                first_pixels.append(data[payload:payload + 4])
            assert first_pixels.count(COLOUR) == 2, first_pixels
            live_reference = live
        else:
            assert not slices, slices
            assert live == live_reference, "the switch changed the live pictures"

        blobs = [data[payload:payload + size] for _, _, payload, size in slices]
        out = folder / f"played-{label}"
        _, played = replay(player, recording, out, "warp", env, presents=WINDOW_PRESENTS)
        assert played.keys() == expected.keys(), played.keys()
        if enabled:
            assert played == expected, "the replay lost the copy destination written in the prefix"
        else:
            for picture in played.values():
                assert_colour(picture, ZERO)
        summary = json.loads((out / "replay_summary.json").read_text())
        assert summary["gpu_state"] == {"restored": len(slices), "failed": 0, "untranslated": 0}, summary
        if label == "tiny-budget":
            assert sorted(blobs) == sorted(reference_blobs), "piecewise staging changed the captured bytes"
        elif enabled:
            reference_blobs = blobs
    check_starved(folder, scene, vs, ps, root, env)
    print("GPU state: two replayed presents match live with capture, and are zero without capture")
    return check_timed_window(folder, scene, vs, ps, root, env)


def main(scene, player):
    with tempfile.TemporaryDirectory() as tmp:
        return check(Path(tmp), scene, player)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
