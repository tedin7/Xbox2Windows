"""Replay changes to plain title memory at byte 4096 and descriptor copies made by the CPU."""

import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from startup_replay_check import chunks, pictures, replay, run
from startup_stream_draw_check import gfx2hlsl, pixel, vertex


PRESENTS = 4
HLSL = """ByteAddressBuffer constants : register(t0);
float4 main() : SV_Target { return asfloat(constants.Load4(4096)); }
"""


def prepare(folder, scene, mode):
    vs, ps, root, cache = (folder / name for name in ("vs", "ps", "root", "cache"))
    cache.mkdir()
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(False))
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_") or k == "X2W_FXC"}
    env.update(X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache), X2W_TEST_REPLAY_TITLE_RANGE=mode)
    if mode == "root":
        env["X2W_TEST_ROOT_CBV"] = "bound"
    code, text = run([scene, vs, ps, root, "warp", 1], env)
    if code == 77:
        return None
    assert code == 0, text[-3000:]
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
    manifest, = (p for p in cache.glob("*.txt") if p.name != "revision.txt")
    binding = "root 0" if mode == "root" else "table 0 0"
    with manifest.open("a", encoding="ascii") as file:
        file.write(f"\nsrv 0 buf {binding}\n")
    source = manifest.with_suffix(".ps.hlsl")
    source.write_text(HLSL, encoding="ascii")
    subprocess.run([env["X2W_FXC"], "/nologo", "/T", "ps_5_0", "/E", "main", "/Fo",
                    manifest.with_suffix(".ps.cso"), source], check=True)
    return [scene, vs, ps, root, "warp", PRESENTS], env


def mapped_pages(path):
    data = path.read_bytes()
    for tag, header, _hb, payload, _size in chunks(data):
        if tag == b"MAP ":
            address, _bytes, seq, runs = struct.unpack_from("<QQQI", data, header)
            for _ in range(runs):
                first, count, kind = struct.unpack_from("<III", data, payload)
                payload += 12 + (16 * count if kind == 0 else 0)
                for i in range(first, first + count):
                    yield address + 4096 * i, seq


def check(folder, player, args, env):
    reference = None
    for enabled in ("1", "0"):
        recording = folder / f"ranges-{enabled}.x2wr"
        original = folder / f"original-{enabled}"
        original.mkdir()
        # Draw pages capture descriptor tables and root spans on their own (startup_replay_draw_tables_warp and
        # startup_replay_draw_pages_warp cover them); they are off here so the control proves the title-range rule alone.
        code, text = run(args, dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                                    X2W_PRESENT_PICTURES=str(original), X2W_REPLAY_TITLE_RANGES=enabled,
                                    X2W_REPLAY_DRAW_PAGES="0"))
        assert code == 0, text[-3000:]
        constants, heap = (int(word, 16) for word in re.search(r"title ranges: ([0-9a-f]+) ([0-9a-f]+)", text).groups())
        captured = list(mapped_pages(recording))
        recorded = pictures(original)
        assert len(recorded) == PRESENTS, recorded.keys()
        colours = [bytes((255, 0, 0, 255)), bytes((0, 255, 0, 255))]
        for frame, (_size, scanlines) in enumerate(recorded.values()):
            assert colours[frame % 2] * 4 in scanlines, (frame, "constant colour missing")
            assert colours[1 - frame % 2] * 4 not in scanlines, (frame, "previous constant colour present")
        _score, played = replay(player, recording, folder / f"played-{enabled}", "warp", env, presents=PRESENTS)
        assert played.keys() == recorded.keys()
        if enabled == "1":
            assert played == recorded, [name for name in recorded if played[name] != recorded[name]]
            assert len({seq for address, seq in captured if address == constants + 4096}) >= PRESENTS
            if heap:
                assert any(address == heap for address, _seq in captured), "CPU heap page missing"
            reference = recorded
        else:
            assert recorded == reference, "the switch changed the live draw"
            assert all(address not in {constants, constants + 4096, heap} for address, _seq in captured)
            assert played != recorded, "the control replay matched without the title ranges"


def main(scene, player):
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("root", "generated", "supplied"):
            folder = Path(tmp) / mode
            folder.mkdir()
            prepared = prepare(folder, scene, mode)
            if prepared is None:
                return 77
            check(folder, player, *prepared)
            print(f"{mode}: {PRESENTS} presents match with title ranges enabled, control differs")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
