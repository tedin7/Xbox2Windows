"""Replay a placed ring whose title bytes are cleared before Present."""

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
HLSL = """ByteAddressBuffer root_data : register(t0);
ByteAddressBuffer vertex_data : register(t1);
float4 main() : SV_Target { return float4(asfloat(root_data.Load2(0)), asfloat(vertex_data.Load2(0))); }
"""


def prepare(folder, scene):
    vs, ps, root, cache = (folder / name for name in ("vs", "ps", "root", "cache"))
    cache.mkdir()
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(False))
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_") or k == "X2W_FXC"}
    env.update(X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache), X2W_TEST_REPLAY_TITLE_RANGE="ring")
    code, text = run([scene, vs, ps, root, "warp", 1], env)
    if code == 77:
        return None
    assert code == 0, text[-3000:]
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
    manifest, = (p for p in cache.glob("*.txt") if p.name not in ("revision.txt", "x2w_key_seed.txt"))
    with manifest.open("a", encoding="ascii") as file:
        file.write("\nsrv 0 buf root 0\nsrv 1 buf vb 0\n")
    source = manifest.with_suffix(".ps.hlsl")
    source.write_text(HLSL, encoding="ascii")
    subprocess.run([env["X2W_FXC"], "/nologo", "/T", "ps_5_0", "/E", "main", "/Fo",
                    manifest.with_suffix(".ps.cso"), source], check=True)
    return [scene, vs, ps, root, "warp", PRESENTS], env


def maps_ring_start(path, ring):
    data = path.read_bytes()
    for tag, header, _hb, payload, _size in chunks(data):
        if tag != b"MAP ":
            continue
        address, size, _seq, runs = struct.unpack_from("<QQQI", data, header)
        if not address <= ring < address + size:
            continue
        for _ in range(runs):
            first, count, kind = struct.unpack_from("<III", data, payload)
            if address + first * 4096 <= ring < address + (first + count) * 4096:
                return True
            payload += 12 + (16 * count if kind == 0 else 0)
    return False


def check(folder, player, args, env):
    colours = [bytes(rgba) for rgba in ((255, 0, 128, 255), (0, 255, 64, 255),
                                      (191, 64, 0, 255), (64, 191, 255, 255))]
    reference = None
    for enabled in ("1", "0"):
        recording = folder / f"ring-{enabled}.x2wr"
        original = folder / f"original-{enabled}"
        original.mkdir()
        code, text = run(args, dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                                    X2W_PRESENT_PICTURES=str(original), X2W_REPLAY_DRAW_PAGES=enabled))
        if code == 77:
            return 77
        assert code == 0, text[-3000:]
        match = re.search(r"title ranges: ([0-9a-f]+) 0\b", text)
        assert match, text[-3000:]
        ring = int(match[1], 16)
        recorded = pictures(original)
        assert len(recorded) == PRESENTS, recorded.keys()
        for frame, (_size, scanlines) in enumerate(recorded.values()):
            assert colours[frame] * 4 in scanlines, (frame, "ring colour missing")
            if frame:
                assert colours[frame - 1] * 4 not in scanlines, (frame, "previous ring colour present")
        _score, played = replay(player, recording, folder / f"played-{enabled}", "warp", env, presents=PRESENTS)
        assert played.keys() == recorded.keys()
        if enabled == "1":
            assert played == recorded, [name for name in recorded if played[name] != recorded[name]]
            reference = recorded
            print(f"draw pages enabled: {PRESENTS} replayed presents match live pictures")
        else:
            assert recorded == reference, "the switch changed the live draw"
            assert maps_ring_start(recording, ring), "the scan missed the ring"
            assert played != recorded, "the control replay matched without draw pages"
            print("draw pages disabled: live pictures match the enabled case and the replay differs")
    return 0


def main(scene, player):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        prepared = prepare(folder, scene)
        if prepared is None:
            return 77
        return check(folder, player, *prepared)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
