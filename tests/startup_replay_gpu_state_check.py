"""Check that the first replayed copy reads a DEFAULT texture written only in the prefix."""

import json
import os
from pathlib import Path
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
    for enabled in (True, False):
        label = "default" if enabled else "off"
        recording = folder / f"gpu-{label}.x2wr"
        original = folder / f"original-{label}"
        original.mkdir()
        record_env = dict(env, X2W_REPLAY_RECORD="f2,2", X2W_REPLAY_RECORD_FILE=str(recording),
                          X2W_PRESENT_PICTURES=str(original))
        if not enabled:
            record_env["X2W_REPLAY_GPU_STATE"] = "0"
        code, log = run([scene, vs, ps, root, "warp", PREFIX_PRESENTS + WINDOW_PRESENTS],
                        record_env)
        if code == 77:
            return 77
        assert code == 0, log[-3000:]
        live = pictures(original)
        assert len(live) == PREFIX_PRESENTS + WINDOW_PRESENTS, live.keys()
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
    print("GPU state: two replayed presents match live with capture, and are zero without capture")
    return 0


def main(scene, player):
    with tempfile.TemporaryDirectory() as tmp:
        return check(Path(tmp), scene, player)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
