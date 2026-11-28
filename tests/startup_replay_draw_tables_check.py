"""Replay CPU descriptor copies and translate the view ids stored in the captured pages."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_replay_check import chunks, pictures, run
from startup_stream_draw_check import gfx2hlsl, pixel, vertex


PRESENTS = 4
DEVICE_KIND, CREATE_SRV_SLOT = 0, 18
COLOURS = (bytes((255, 0, 0, 255)), bytes((0, 255, 0, 255)))
HLSL = """ByteAddressBuffer colour : register(t0);
float4 main() : SV_Target { return asfloat(colour.Load4(0)); }
"""


def prepare(folder, scene):
    vs, ps, root, cache = (folder / name for name in ("vs", "ps", "root", "cache"))
    cache.mkdir()
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(False))
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_") or k == "X2W_FXC"}
    env.update(X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache), X2W_TEST_REPLAY_TITLE_RANGE="table")
    code, text = run([scene, vs, ps, root, "warp", 1], env)
    if code == 77:
        return None
    assert code == 0, text[-3000:]
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
    manifest, = (p for p in cache.glob("*.txt") if p.name not in ("revision.txt", "x2w_key_seed.txt"))
    with manifest.open("a", encoding="ascii") as file:
        file.write("\nsrv 0 buf table 0 0\n")
    source = manifest.with_suffix(".ps.hlsl")
    source.write_text(HLSL, encoding="ascii")
    subprocess.run([env["X2W_FXC"], "/nologo", "/T", "ps_5_0", "/E", "main", "/Fo",
                    manifest.with_suffix(".ps.cso"), source], check=True)
    return [scene, vs, ps, root, "warp", PRESENTS + 1], env


def record(folder, args, env, enabled):
    recording = folder / f"table-{enabled}.x2wr"
    original = folder / f"original-{enabled}"
    original.mkdir()
    code, text = run(args, dict(env, X2W_REPLAY_RECORD=f"f1,{PRESENTS}", X2W_REPLAY_RECORD_FILE=str(recording),
                                X2W_PRESENT_PICTURES=str(original), X2W_REPLAY_DRAW_PAGES=enabled))
    assert code == 0, text[-3000:]
    live = list(pictures(original).values())
    assert len(live) == PRESENTS + 1, len(live)
    for frame, (_size, scanlines) in enumerate(live):
        assert COLOURS[frame % 2] * 4 in scanlines, (frame, "table colour missing")
        assert COLOURS[1 - frame % 2] * 4 not in scanlines, (frame, "other table colour present")
    return recording, live[1:]


def check_prefix_views(recording):
    data = recording.read_bytes()
    view_ids = set()
    srv_seq = None
    for tag, header, _hb, _payload, _size in chunks(data):
        if tag == b"PRES":
            break
        if tag == b"CALL":
            seq, _thread, kind, slot = struct.unpack_from("<QIHH", data, header)
            srv_seq = seq if (kind, slot) == (DEVICE_KIND, CREATE_SRV_SLOT) else None
        if tag == b"VIDS":
            seq, view_id = struct.unpack_from("<QI", data, header)
            if seq == srv_seq:
                view_ids.add(view_id)
    assert len(view_ids) == 2, view_ids


def play(player, recording, out, env, **switches):
    env = {k: v for k, v in env.items() if not k.startswith("X2W_")}
    code, text = run([player, recording, out, "--adapter", "warp"], dict(env, **switches))
    log_path = out / "stderr.txt"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    assert code == 0, (code, text[-2000:], log[-3000:])
    summary = json.loads((out / "replay_summary.json").read_text())
    assert summary["unknown_object"] == summary["missing_blob"] == summary["fence_timeouts"] == 0, summary
    assert summary["presents"] == PRESENTS, summary
    played = list(pictures(out).values())
    assert len(played) == PRESENTS, len(played)
    return played, summary["view_tokens"], log


def check(folder, player, args, env):
    recording, live = record(folder, args, env, "1")
    check_prefix_views(recording)
    played, _, _ = play(player, recording, folder / "played", env)
    assert played == live, "draw pages replay differs from the live table colours"
    print(f"draw tables: all {PRESENTS} replayed pictures match the alternating live pictures")

    stale, stale_live = record(folder, args, env, "0")
    assert stale_live == live, "disabling draw pages changed the live pictures"
    played, _, _ = play(player, stale, folder / "stale", env)
    assert played != live, "the replay matched without table page capture"

    shifted = {"X2W_TEST_VIEW_ID_SHIFT": "7"}
    played, tokens, _ = play(player, recording, folder / "shifted", env, **shifted)
    assert played == live, "the replay differs with shifted view ids"
    assert tokens["rewritten"] > 0 and tokens["unmapped"] == 0, tokens
    played, tokens, log = play(player, recording, folder / "untranslated", env, X2W_REPLAY_VIEW_TOKENS="0", **shifted)
    assert played != live, "the shifted replay matched with token translation disabled"
    assert tokens == {"rewritten": 0, "unmapped": 0}, tokens
    assert "x2w-draw unknown view token id" in log, log[-3000:]
    print("draw tables controls: stale pages and untranslated shifted ids both change the replay")
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
