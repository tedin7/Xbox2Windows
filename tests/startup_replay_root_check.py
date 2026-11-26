#!/usr/bin/env python3
"""Root constants and root constant buffer views survive record and replay (docs/startup-runtime.md#d3d12-replay-record).

startup_replay_root fills a 64 x 32 target every frame with a colour that reaches the pixel shader only through
  const  SetGraphicsRoot32BitConstants (list slot 40), one word, or
  cbv    SetGraphicsRootConstantBufferView (list slot 42), the address of title memory the title rewrites each frame.
Both pixel shaders are written in HLSL and installed in the shader cache by hand (the translator's own output is not the
subject here). Checked per mode:
  - the recorded run presents pictures that change from frame to frame;
  - x2w_replay with no X2W_ setting plays every call on a known object and writes the same pictures, byte for byte;
  - control: the recording with the last frame's root argument changed (the constant word, or the recorded page behind the CBV)
    replays to a different last picture and the same earlier ones.

usage: startup_replay_root_check.py <startup_replay_root> <x2w_replay> <warp|hardware> <fxc>
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_replay_check import CALL, LIST_KIND, chunks, explain, pictures, replay, run
from startup_stream_check import container, gfx2hlsl

PRESENTS = 8
SET_CONSTANTS, SET_CBV = 40, 42

VERTEX = """float4 main(uint id : SV_VertexID) : SV_Position {
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
}
"""
PIXEL = {
    "const": ("cbuffer entry : register(b0) { uint ud0; };\n"
              "float4 main(float4 p : SV_Position) : SV_Target {\n"
              "  return float4(ud0 & 255, (ud0 >> 8) & 255, (ud0 >> 16) & 255, 255) / 255.0;\n}\n",
              "udsrc 0 rootconst 1 0\n"),
    "cbv": ("ByteAddressBuffer constants : register(t0);\n"
            "float4 main(float4 p : SV_Position) : SV_Target {\n  return asfloat(constants.Load4(0));\n}\n",
            "udsrc 0 rect\nsrv 0 buf root 0\n"),
}


def change_last(source, target, mode):
    """Copies the recording with the last frame's root argument changed."""
    data = bytearray(source.read_bytes())
    slot = SET_CONSTANTS if mode == "const" else SET_CBV
    calls, argm, blobs = [], {}, {}
    for tag, header, _hb, payload, size in chunks(data):
        if tag == b"CALL":
            seq, _t, kind, call_slot = struct.unpack_from("<QIHH", data, header)
            if kind == LIST_KIND and call_slot == slot:
                calls.append((seq, header))
        elif tag == b"ARGM":
            seq, arg, _depth, _address, _bytes, blob_hash = struct.unpack_from("<QIIQQQ", data, header)
            argm[(seq, arg)] = blob_hash
        elif tag == b"BLOB":
            blobs[struct.unpack_from("<Q", data, header)[0]] = payload
    assert len(calls) == PRESENTS, (mode, len(calls))
    seq, header = calls[-1]
    if mode == "const":
        at = blobs[argm[(seq, 2)]]
        data[at] ^= 0xFF                                     # the red byte of the last frame's word
    else:
        # The CBV's data is title memory. Damage the recorded copy of the last frame's colour: the PAGE chunk that holds it, or
        # (memory the recorder did not take as pages) the argument memory of the call itself.
        last = struct.pack("<4f", (PRESENTS - 1) / PRESENTS, 0.25, 1.0 - (PRESENTS - 1) / PRESENTS, 1.0)
        hit = 0
        for tag, header, _hb, payload, size in chunks(data):
            if tag == b"PAGE" and bytes(data[payload:payload + 16]) == last:
                data[payload + 3] ^= 0x40
                hit += 1
        if hit:
            print("replay root cbv control: damaged %d PAGE chunk(s)" % hit)
        else:
            at = blobs[argm[(seq, 1)]]
            assert bytes(data[at:at + 16]) == last, "the call's argument memory is not the last frame's constant buffer"
            data[at + 3] ^= 0x40                                # the red float 0.875 becomes huge (the low byte would not show in 8 bits)
            print("replay root cbv control: no PAGE holds the buffer (plain VirtualAlloc memory); damaged the call's argument memory")
    target.write_bytes(bytes(data))


def main(scene, player, backend, fxc):
    for mode in ("const", "cbv"):
        with tempfile.TemporaryDirectory() as temp:
            code = one(Path(temp), mode, scene, player, backend, fxc)
        if code:
            return code
    return 0


def one(directory, mode, scene, player, backend, fxc):
    cache = directory / "cache"
    cache.mkdir()
    vs, ps, root = (directory / name for name in ("vs", "ps", "root"))
    vs.write_bytes(container(1))
    ps.write_bytes(container(0))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1", X2W_SHADER_CACHE=str(cache))
    env.setdefault("X2W_D3D12_DEBUG", "0")
    env.update(X2W_TEST_ROOT_RESET="bound") if mode == "const" else env.update(X2W_TEST_ROOT_CBV="bound")
    for name in ("X2W_REPLAY_RECORD", "X2W_REPLAY_RECORD_FILE", "X2W_PRESENT_PICTURES"):
        env.pop(name, None)
    # A first one-frame run writes the root signature file, which names the pipeline key.
    bootstrap = subprocess.run([scene, vs, ps, root, backend, "1", mode], env=env,
                               capture_output=True, text=True)
    if bootstrap.returncode == 77 and backend == "hardware":
        print("No D3D12 feature-level 12 hardware adapter")
        return 77
    assert bootstrap.returncode == 0, bootstrap.stdout + bootstrap.stderr
    key = f"{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes()):016x}"
    hlsl_ps, manifest = PIXEL[mode]
    (cache / f"{key}.txt").write_text(manifest, encoding="ascii")
    for stage, source, profile in (("vs", VERTEX, "vs_5_0"), ("ps", hlsl_ps, "ps_5_0")):
        path = directory / f"{stage}.hlsl"
        path.write_text(source, encoding="ascii")
        subprocess.run([fxc, "/nologo", "/T", profile, "/E", "main", "/Fo", cache / f"{key}.{stage}.cso", path], check=True)

    recorded_dir, recording = directory / "recorded", directory / "run.x2wr"
    recorded_dir.mkdir()
    code, text = run([scene, vs, ps, root, backend, PRESENTS, mode],
                     dict(env, X2W_REPLAY_RECORD="all", X2W_REPLAY_RECORD_FILE=str(recording), X2W_PRESENT_PICTURES=str(recorded_dir)))
    assert code == 0, text[-3000:]
    recorded = pictures(recorded_dir)
    names = ["present_%04d.png" % n for n in range(1, PRESENTS + 1)]
    assert sorted(recorded) == names, sorted(recorded)
    assert len({recorded[n] for n in names}) == PRESENTS, "%s: the pictures do not change between presents" % mode

    _score, replayed = replay(player, recording, directory / "replayed", backend, env, presents=PRESENTS)
    differ = [n for n in names if replayed[n] != recorded[n]]
    assert not differ, "%s: replayed pictures differ at %s\n%s" % (
        mode, differ, explain(recorded[differ[0]], replayed[differ[0]], directory / "replayed" / "stderr.txt"))

    changed = directory / "changed.x2wr"
    change_last(recording, changed, mode)
    _score, mutated = replay(player, changed, directory / "changed", backend, env, presents=PRESENTS)
    assert mutated[names[-1]] != recorded[names[-1]], "%s: a changed root argument replayed to the same last picture" % mode
    assert all(mutated[n] == recorded[n] for n in names[:-1]), "%s: the change reached other presents" % mode
    print("replay root %s: %d presents equal to the recorded run; changed last root argument (word or CBV page) differs at the last present only"
          % (mode, PRESENTS))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
