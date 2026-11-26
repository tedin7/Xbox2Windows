#!/usr/bin/env python3
"""Replay family "draws": the stream draw scenes, recorded with X2W_REPLAY_RECORD=all, replay to the same pictures.

startup_stream_consumer (tests/startup_stream_draw.cpp) draws a two-triangle scene through the stand-in and presents
its target twice. For each scene below (red or green pixel shader, and the three packed-target modes of the draw check):
  - the recorded run writes present_<nnnn>.png per present (X2W_PRESENT_PICTURES);
  - x2w_replay plays the recording with no X2W_ setting and writes the same picture names, every one equal;
  - control: the green scene's recording is no replay of the red scene: the pictures of two different scenes differ,
    so an equal comparison is a comparison of content.

usage: startup_replay_draw_scenes_check.py <startup_stream_consumer> <x2w_replay> <warp|hardware>
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_replay_check import pictures, replay, run
from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex

# name, green pixel shader, scene mode argument ("" = the plain scene)
SCENES = (
    ("red", False, ""),
    ("green", True, ""),
    ("parent-a8", True, "parent-a8"),
    ("child-a8", True, "child-a8"),
    ("child-r8", True, "child-r8"),
    ("child-r8-red", False, "child-r8"),
)


def record(consumer, directory, backend, name, green, mode, base_env):
    """Runs the scene recorded; returns (recording, pictures directory) or 77 when the backend is missing."""
    work = directory / name
    cache, shots = work / "cache", work / "recorded"
    cache.mkdir(parents=True)
    shots.mkdir()
    vs, ps, root, output, recording = (work / n for n in ("vs", "ps", "root", "pixels", "run.x2wr"))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(green))
    env = dict(base_env, X2W_SHADER_CACHE=str(cache))
    args = [consumer, vs, ps, root]
    code, text = run(args, env)  # first run publishes the shader request
    assert code == 0, text[-2000:]
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
    code, text = run(args + [backend, output] + ([mode] if mode else []),
                     dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording), X2W_PRESENT_PICTURES=str(shots)))
    if code == 77:
        return 77, None
    assert code == 0 and "-> drawn;" in text, "%s: recorded run failed (%d)\n%s" % (name, code, text[-2500:])
    (work / "recorded.txt").write_text(text)
    return recording, shots


def main(consumer, player, backend):
    base_env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1")
    base_env.setdefault("X2W_D3D12_DEBUG", "0")
    for name in ("X2W_REPLAY_RECORD", "X2W_REPLAY_RECORD_FILE", "X2W_PRESENT_PICTURES"):
        base_env.pop(name, None)
    played = {}
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        for name, green, mode in SCENES:
            recording, shots = record(consumer, directory, backend, name, green, mode, base_env)
            if recording == 77:
                print("No D3D12 feature-level 12 hardware adapter")
                return 77
            recorded = pictures(shots)
            log = [l for l in (directory / name / "recorded.txt").read_text().splitlines() if "present" in l]
            assert sorted(recorded) == ["present_0001.png", "present_0002.png"], (name, sorted(recorded), log[:8])
            _score, replayed = replay(player, recording, directory / name / "replayed", backend, base_env, presents=2)
            assert sorted(replayed) == sorted(recorded), (name, sorted(replayed))
            differ = [n for n in recorded if replayed[n] != recorded[n]]
            assert not differ, "%s: replayed pictures differ from the recorded run: %s" % (name, differ)
            played[name] = replayed
            print("replay draws %s: %d presents equal" % (name, len(recorded)))
        # The control: scenes that draw different things replay to different pictures.
        assert played["red"]["present_0002.png"] != played["green"]["present_0002.png"], "red and green replay to one picture"
        assert played["child-r8"]["present_0002.png"] != played["child-r8-red"]["present_0002.png"], "r8 scenes replay alike"
        print("replay draws: %d scenes replay equal; different scenes replay to different pictures" % len(SCENES))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
