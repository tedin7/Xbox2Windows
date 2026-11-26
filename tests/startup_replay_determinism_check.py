#!/usr/bin/env python3
"""Replaying one recording twice gives the same output (docs/startup-runtime.md#d3d12-replay-record).

startup_replay_quad presents an animated scene every frame; its recording (X2W_REPLAY_RECORD=all) is played by x2w_replay.
Checked, with every replay in its own output folder and its own shader cache:
  - two replays one after the other: present_<nnnn>.png for exactly presents 1..12 in order, every picture byte equal, and the
    replay_summary.json counters equal (calls played, presents, unknown objects, missing blobs, fence timeouts, argument bytes,
    pages applied/written/zero/missing, and the hash of every restored title range); only the timing fields may differ;
  - two replays running at the same time (the CPU is shared, the threads and fences interleave differently): the same output again;
  - control: a recording with the last frame's clear colour changed replays to a different last picture and the same
    other pictures, so the equality above is not a comparison of nothing.

usage: startup_replay_determinism_check.py <startup_replay_quad> <x2w_replay> <warp|hardware>
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_replay_check import PRESENTS, change_last_clear, gfx2hlsl, pictures, pixel, replay, run, vertex

TIMING = {"wall_s", "fps"}


def summary_of(folder):
    """The replay_summary.json of a replay without its timing fields."""
    data = json.loads((Path(folder) / "replay_summary.json").read_text())
    return {key: value for key, value in data.items() if key not in TIMING}


def play(player, recording, folder, backend, env):
    folder.mkdir()
    _score, shown = replay(player, recording, folder, backend, env)
    return shown, summary_of(folder)


def explain(first, second, label):
    names = sorted(set(first[0]) | set(second[0]))
    differ = [n for n in names if first[0].get(n) != second[0].get(n)]
    keys = [k for k in sorted(set(first[1]) | set(second[1])) if first[1].get(k) != second[1].get(k)]
    return "%s: pictures differ at %s, summary fields differ at %s" % (label, differ, keys)


def main(scene, player, backend):
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
        for name in ("X2W_REPLAY_RECORD", "X2W_REPLAY_RECORD_FILE", "X2W_PRESENT_PICTURES"):
            env.pop(name, None)
        code, text = run([scene, vs, ps, root, backend, 1], env)
        if code == 77 and backend == "hardware":
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        assert code == 0, text[-3000:]
        subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)

        recorded_dir, recording = directory / "recorded", directory / "run.x2wr"
        recorded_dir.mkdir()
        code, text = run([scene, vs, ps, root, backend, PRESENTS],
                         dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                              X2W_PRESENT_PICTURES=str(recorded_dir)))
        assert code == 0, text[-3000:]

        names = ["present_%04d.png" % n for n in range(1, PRESENTS + 1)]
        first = play(player, recording, directory / "first", backend, env)
        assert sorted(first[0]) == names, "the replay wrote %s, not presents 1..%d" % (sorted(first[0]), PRESENTS)
        assert first[1]["presents"] == PRESENTS and first[1]["unknown_object"] == 0 and first[1]["missing_blob"] == 0, first[1]
        sizes = {first[0][n][0] for n in names}
        assert len(sizes) == 1, "the presented surface changes size between presents: %s" % sizes

        second = play(player, recording, directory / "second", backend, env)
        assert second == first, explain(first, second, "second replay after the first")

        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(play, player, recording, directory / ("together_%d" % k), backend, env) for k in (1, 2)]
            together = [job.result() for job in jobs]
        for k, result in enumerate(together, 1):
            assert result == first, explain(first, result, "replay %d of two running at the same time" % k)

        changed = directory / "changed.x2wr"
        change_last_clear(recording, changed)
        mutated = play(player, changed, directory / "changed", backend, env)
        assert mutated[0][names[-1]] != first[0][names[-1]], "a changed clear colour replayed to the same last picture"
        assert all(mutated[0][n] == first[0][n] for n in names[:-1]), "the change reached other presents"
        print("replay determinism: %d presents, 2 sequential and 2 concurrent replays equal in pictures and counters; "
              "a changed recording differs at the last present only" % PRESENTS)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
