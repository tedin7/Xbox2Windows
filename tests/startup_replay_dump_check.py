"""Record and replay compute and indirect-draw submissions that never Present."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import startup_compute_dispatch_texture_check as compute
from startup_replay_check import change_last_clear, chunks, pictures, replay, run
from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex

DUMP = re.compile(r"x2w-replay dump submit (\d+) resource (\d+) (\d+)x(\d+) fmt (\d+) hash ([0-9a-f]{16})")


def dumps(text):
    rows = set()
    for line in text.splitlines():
        assert not line.startswith("x2w-replay dump:"), line
        if line.startswith("x2w-replay dump submit "):
            match = DUMP.fullmatch(line)
            assert match, line
            row = tuple(map(int, match.groups()[:5])) + (match[6],)
            assert row not in rows, line
            rows.add(row)
    return rows


def read_dump(player, recording, out, adapter, switch=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_")}
    if switch is not None:
        env["X2W_REPLAY_DUMP"] = switch
    code, text = run([player, recording, out, "--adapter", adapter], env)
    log = out / "stderr.txt"
    trace = log.read_text(errors="replace") if log.exists() else ""
    assert code == 0, (code, text, trace[-4000:])
    summary = json.loads((out / "replay_summary.json").read_text())
    assert all(summary[k] == 0 for k in ("presents", "unknown_object", "missing_blob", "fence_timeouts")), summary
    assert not list(out.glob("present_*.png")), out
    return dumps(trace)


def recorded_dump(recording, text, submits, width, height, fmt):
    # submits: the submit numbers that wrote a target (an int N means 1..N).
    wanted = set(range(1, submits + 1)) if isinstance(submits, int) else set(submits)
    rows = dumps(text)
    assert {row[:5] for row in rows} == {(n, 1, width, height, fmt) for n in wanted}, (rows, text[-4000:])
    assert len(rows) == len(wanted), rows
    assert len({row[-1] for row in rows}) == len(wanted), rows
    tags = {tag for tag, *_ in chunks(recording.read_bytes())}
    assert b"PRES" not in tags, tags
    return rows


def indirect(directory, quad, player, adapter, env):
    directory.mkdir()
    vs, ps, root = (directory / name for name in ("vs", "ps", "root"))
    cache = directory / "cache"
    cache.mkdir()
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(True))
    args = [quad, vs, ps, root, adapter, 3]
    env = dict(env, X2W_SHADER_CACHE=str(cache), X2W_TEST_REPLAY_INDIRECT="1",
               X2W_TRANSLATED_INDIRECT_DRAW="1", X2W_D3D12_LINEAR_BUFFERS="copy")
    code, text = run(args, env)
    if code == 77:
        return 77
    assert code == 0, text[-4000:]
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True, timeout=120)

    recording = directory / "draw.x2wr"
    code, text = run(args, dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                                X2W_REPLAY_DUMP="1", X2W_REPLAY_PAGES="1"))
    assert code == 0, text[-4000:]
    # Submit 2 is the ExecuteIndirect draw. The scene's stream pipeline carries no guest root signature, so the
    # translated indirect draw is refused (missing_guest_root), nothing is drawn and no row is due for it.
    expected = recorded_dump(recording, text, (1, 3), 64, 32, 28)
    for n in (1, 3):
        assert text.index(f"about to signal submit {n}\n") < text.index(f"x2w-replay dump submit {n} "), text
    assert {b"PAGE", b"MAP "} <= {tag for tag, *_ in chunks(recording.read_bytes())}
    assert read_dump(player, recording, directory / "replayed", adapter, "1") == expected
    assert not read_dump(player, recording, directory / "disabled", adapter, "0")

    changed = directory / "changed.x2wr"
    change_last_clear(recording, changed, clear_count=2)
    mutated = read_dump(player, changed, directory / "mutated", adapter)
    assert {r[:-1] for r in mutated} == {r[:-1] for r in expected}
    assert [r[0] for r in expected - mutated] == [3], (expected, mutated)
    assert [r[0] for r in mutated - expected] == [3], (expected, mutated)
    return 0


def late_indirect(directory, quad, player, env, mode="late"):
    directory.mkdir()
    source = directory / "triangle.hlsl"
    source.write_text("""
float4 vs(uint id : SV_VertexID) : SV_Position {
    return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0, 1);
}
float4 ps() : SV_Target { return float4(1, 0, 1, 1); }
""", encoding="ascii")
    vs, ps, root = (directory / name for name in ("vs", "ps", "root"))
    for entry, target, output in (("vs", "vs_5_0", vs), ("ps", "ps_5_0", ps)):
        subprocess.run([env["X2W_FXC"], "/nologo", "/E", entry, "/T", target, "/Fo", output, source],
                       env=env, check=True, timeout=30)
    frames = 4
    args = [quad, vs, ps, root, "warp", frames]
    env = dict(env, X2W_TEST_REPLAY_INDIRECT=mode, X2W_D3D12_LINEAR_BUFFERS="copy")
    reference = None
    for enabled in ("1", "0"):
        recording = directory / f"late-{enabled}.x2wr"
        original = directory / f"live-{enabled}"
        original.mkdir()
        code, text = run(args, dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                                    X2W_PRESENT_PICTURES=str(original), X2W_REPLAY_DRAW_PAGES=enabled))
        assert code == 0, text[-4000:]
        live = pictures(original)
        assert len(live) == frames, live.keys()
        magenta = bytes((255, 0, 255, 255)) * 4
        for frame, (_size, scanlines) in enumerate(live.values()):
            assert (magenta in scanlines) == (frame % 2 == 0), (frame, text[-4000:])

        _score, played = replay(player, recording, directory / f"played-{enabled}", "warp", env, presents=frames)
        assert played.keys() == live.keys()
        mismatches = [frame for frame, name in enumerate(live) if played[name] != live[name]]
        if enabled == "1":
            assert not mismatches, ("late indirect replay mismatches", mismatches, mode, [l for l in text.splitlines() if "x2w-replay" in l and "pages" in l][-6:])
            reference = live
        else:
            assert live == reference, "draw pages changed the live pictures"
            # CPU-written words need the capture. GPU-written words come back with the replayed copy.
            assert mismatches == ([0, 2] if mode == "late" else []), ("indirect control mismatches", mismatches)
    print("Four late indirect replay pictures match with draw pages enabled. Frames 0 and 2 differ with pages disabled.")


def dispatch(directory, consumer, player, dxc, adapter, env):
    cache, _ = compute.prepare(directory)
    env = dict(env, X2W_SHADER_CACHE=str(cache), X2W_TEST_TRUNCATED_ONLY="1")
    root = directory / "compute_root"
    root.write_bytes(compute.C.root_blob(3, table_range=True))
    (directory / "program").write_bytes(compute.S.fixture.WORDS)
    (directory / "data").write_bytes(compute.DATA)
    (directory / "state").write_bytes(compute.state_a(compute.S.fixture.GROUP, compute.S.RSRC2))
    env.update(X2W_TEST_COMPUTE_ROOT=str(root), X2W_TEST_DISPATCH_OUT=str(directory / "out.bin"))
    if adapter == "hardware":
        env["X2W_TEST_DISPATCH_HARDWARE"] = "1"
    args = [consumer, directory / "vertex", directory / "pixel", directory / "root", "slot67",
            directory / "program", directory / "data", directory / "state", "texture-store-replay"]
    code, text = run(args, env)
    if code == 77:
        return 77
    assert code == 0, text[-4000:]
    request = next(line for line in text.splitlines() if line.startswith("x2w-draw compute request "))
    compute.translate(cache, dxc, int(request.split()[3], 16))

    recording = directory / "compute.x2wr"
    code, text = run(args, dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording), X2W_REPLAY_DUMP="1"))
    assert code == 0 and "Dispatch(2, 1, 1) translated" in text, text[-4000:]
    expected = recorded_dump(recording, text, 2, 32, 8, 10)
    assert read_dump(player, recording, directory / "replayed", adapter) == expected
    return 0


def main(quad, consumer, player, dxc, adapter):
    env = {k: v for k, v in os.environ.items() if not k.startswith("X2W_") or k == "X2W_FXC"}
    env.update(X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1")
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        code = indirect(directory / "draw", quad, player, adapter, env)
        if code:
            return code
        code = dispatch(directory / "compute", consumer, player, dxc, adapter, env)
        if code:
            return code
        if adapter == "warp":
            late_indirect(directory / "late", quad, player, env)
            late_indirect(directory / "gpu", quad, player, env, "gpu")
    print("Replay dumps match for three indirect-scene and two compute-scene submissions. Only the mutated clear differs.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
