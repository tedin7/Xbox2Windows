#!/usr/bin/env python3
"""x2w_replay plays descriptor copies: a scene that draws through a copied render target view replays to the same pictures
(docs/startup-runtime.md#d3d12-replay-record).

startup_replay_quad with the sixth argument `heaps` copies its render target view into a second RTV heap every frame (even
frames CopyDescriptorsSimple into slot 2, odd frames CopyDescriptors with ranges into slot 1) and clears and draws through
the copy. Checked:
  - recorded run: one picture per present, and the pictures change over time (the copy resolved: the target was drawn);
  - replay: every picture equal to the recorded run's, every call played on a known object;
  - a window in the middle (f6,6) replays frames 7-12 equal: the heaps and the copies before the window are recorded;
  - control: the same recording with the descriptor count of every copy set to 0 replays to different pictures (the copy
    really ran in the replay, and the comparison sees it).

usage: startup_replay_heaps_check.py <startup_replay_quad> <x2w_replay> <warp|hardware>
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_replay_check import PRESENTS, chunks, explain, explain_log, pictures, replay, run
from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex

DEVICE_KIND, COPY_DESCRIPTORS, COPY_DESCRIPTORS_SIMPLE = 0, 23, 24
CALL_ARG0 = 24  # offset of args[0] (the descriptor count of both copy calls) in a CALL header


def zero_copy_counts(source, target):
    """Copies the recording with the count of every device CopyDescriptors / CopyDescriptorsSimple call set to 0."""
    data = bytearray(source.read_bytes())
    changed = 0
    for tag, header, _hb, _payload, _size in chunks(data):
        if tag != b"CALL":
            continue
        _seq, _thread, kind, slot = struct.unpack_from("<QIHH", data, header)
        if kind == DEVICE_KIND and slot in (COPY_DESCRIPTORS, COPY_DESCRIPTORS_SIMPLE):
            struct.pack_into("<Q", data, header + CALL_ARG0, 0)
            changed += 1
    assert changed == PRESENTS, "expected one copy call per frame, found %d" % changed
    target.write_bytes(bytes(data))


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
        code, text = run([scene, vs, ps, root, backend, 1, "heaps"], env)
        if code == 77 and backend == "hardware":
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        assert code == 0, text[-3000:]
        subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)

        recorded_dir, recording = directory / "recorded", directory / "run.x2wr"
        recorded_dir.mkdir()
        code, text = run([scene, vs, ps, root, backend, PRESENTS, "heaps"],
                         dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording),
                              X2W_PRESENT_PICTURES=str(recorded_dir)))
        assert code == 0, text[-3000:]
        (directory / "recorded.txt").write_text(text)
        recorded = pictures(recorded_dir)
        names = ["present_%04d.png" % n for n in range(1, PRESENTS + 1)]
        assert sorted(recorded) == names, sorted(recorded)
        assert recorded[names[0]] != recorded[names[-1]], "the scene does not change between presents"

        score, replayed = replay(player, recording, directory / "replayed", backend, env)
        assert sorted(replayed) == names, sorted(replayed)
        differ = [n for n in names if replayed[n] != recorded[n]]
        assert not differ, "replayed pictures differ from the recorded run: %s\n%s\n%s" % (
            differ, explain(recorded[differ[0]], replayed[differ[0]], directory / "replayed" / "stderr.txt"),
            explain_log("recorded run", directory / "recorded.txt"))

        dropped = directory / "dropped.x2wr"
        zero_copy_counts(recording, dropped)
        _score, mutated = replay(player, dropped, directory / "dropped", backend, env)
        same = [n for n in names if mutated[n] == recorded[n]]
        assert not same, "with every copy dropped the pictures %s are still equal to the recorded run" % same

        window_dir, windowed = directory / "window_recorded", directory / "window.x2wr"
        window_dir.mkdir()
        code, text = run([scene, vs, ps, root, backend, PRESENTS, "heaps"],
                         dict(env, X2W_REPLAY_RECORD="f6,6", X2W_REPLAY_RECORD_FILE=str(windowed),
                              X2W_PRESENT_PICTURES=str(window_dir)))
        assert code == 0, text[-3000:]
        whole = pictures(window_dir)
        _score, played = replay(player, windowed, directory / "window_replayed", backend, env, presents=6)
        pairs = [("present_%04d.png" % k, "present_%04d.png" % (k + 6)) for k in range(1, 7)]
        differ = [b for a, b in pairs if played[a] != whole[b]]
        assert not differ, "window replay differs from the recorded run at %s\n%s" % (
            differ, explain(whole[differ[0]], played["present_%04d.png" % (int(differ[0][8:12]) - 6)],
                            directory / "window_replayed" / "stderr.txt"))
        print("replay heaps: %d presents drawn through copied render target views equal to the recorded run; every copy "
              "dropped differs at all presents; window f6,6 replays presents 7-12 equal; %s" % (PRESENTS, score))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
