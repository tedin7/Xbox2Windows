"""Explicit-LOD offset gather controls and D3D12 compute readback."""
import argparse
import hashlib
import math
from pathlib import Path
import struct
import tempfile

from startup_compute_gather_l_check import (COMPUTE_HEAD, COMPUTE_TAIL, HEIGHT, ORDER, ROOT,
    SENTINEL, WIDTH, WORDS, cs_pipe, gather_insn, run_case, s_mov_word, texel, v_mov_word)
from startup_gfx2hlsl_compute_check import refused
import startup_compute_gather_l_check as lod_control
import startup_compute_gather_lz_check as lz_control


UV = (0.328125, 0.6875)


def shader(dynamic=False, nsa=False, *, flags=0, mask=1, nan=None):
    # Add the lane ID to IEEE-754 bit patterns: runtime VGPR values, not literal moves.
    coords = (10, 11, 12) if nsa else (1, 2, 3)
    def add_tid(reg, value):
        return struct.pack('<II', 0x4a0000ff | (reg << 17), value)
    moves = add_tid(coords[0], 0x7fc00000 if nan == 'u' else 0x3ea80000)
    moves += add_tid(coords[1], 0x3f300000)
    moves += v_mov_word(coords[2], '0x7fc00000' if nan == 'lod' else '1.0')
    # Thread ID v0 is zero for the single dispatched lane, but unknown at translation time.
    moves += (bytes.fromhex('ff00004a') + struct.pack('<I', 0x3e01) if dynamic
              else v_mov_word(0, '0x3e01'))
    insn = gather_insn(mask=mask, opcode=0x54, flags=flags)
    if nsa:
        insn = bytearray(insn)
        insn[0] |= 2
        insn += bytes((*coords, 0))
    return (COMPUTE_HEAD + s_mov_word(25, 0xfff000) + s_mov_word(26, 0x04500000)
            + s_mov_word(27, 0) + moves + bytes.fromhex('7fc08cbf') + insn + COMPUTE_TAIL)


def reference(level, offset=(1, -2)):
    width, height = WIDTH >> level, HEIGHT >> level
    x = math.floor(UV[0] * width - 0.5) + offset[0]
    y = math.floor(UV[1] * height - 0.5) + offset[1]
    taps = [texel(level, min(max(x + dx, 0), width - 1), min(max(y + dy, 0), height - 1), 0)
            for dx, dy in ORDER]
    return taps + [SENTINEL] * (WORDS - 4)


def controls():
    # Golden text from revision a08c038e2d42, before this lowering existed.
    old_lod = lod_control.cs_pipe(lod_control.compute_shader()).hlsl_cs()
    old_lz_o = lz_control.pipeline(lz_control.shader_o(1, lz_control.pack_offset(1, -2))).hlsl_cs()
    assert hashlib.sha256(old_lod.encode()).hexdigest() == 'f8c030c649e46c08ae75832fcc6b04b1b89ff5b88179f99a9a2b845a193e2910'
    assert hashlib.sha256(old_lz_o.encode()).hexdigest() == '87643965fae5ee23e25686d0b1970c81de1e257ad317fe141e33207e5b7d5f09'
    want = reference(1)
    assert want != reference(1, (0, 0)) and want != reference(0)
    for dynamic in (False, True):
        for nsa in (False, True):
            hlsl = cs_pipe(shader(dynamic, nsa)).hlsl_cs()
            regs = (10, 11, 12) if nsa else (1, 2, 3)
            assert f'gather_2d_lod_offset(r0, float2(F(v{regs[0]}), F(v{regs[1]})), F(v{regs[2]}), v0, 0, 15.99609375f)' in hlsl
            assert f'v{regs[0]} = 0x3ea80000u + v0;' in hlsl
            assert f'v{regs[1]} = 0x3f300000u + v0;' in hlsl
            assert 'int2 offset = int2(asint(packed_offset << 26u)' in hlsl
            if dynamic and nsa:
                assert hashlib.sha256(hlsl.encode()).hexdigest() == '8e99b7223c1b404b71ae4cacbd7374e7eccdedb8e30c891f0429935c12122b41'
    for coord, size in zip(UV, (WIDTH >> 1, HEIGHT >> 1)):
        scaled = coord * size - 0.5
        assert abs(scaled - round(scaled)) >= 1 / 16
    for nan in ('u', 'lod'):
        assert '0x7f800000u' in cs_pipe(shader(nan=nan)).hlsl_cs()
    refused('one-hot dmask', lambda: cs_pipe(shader(mask=3)))
    refused('modifier tfe', lambda: cs_pipe(shader(flags=1 << 16)))
    refused('modifier lwe', lambda: cs_pipe(shader(flags=1 << 17)))
    print('image_gather4_l_o controls passed')


def dispatch(directory, compiler, renderer, adapter):
    for dynamic in (False, True):
        for nsa in (False, True):
            source = cs_pipe(shader(dynamic, nsa)).hlsl_cs()
            result = run_case(directory, compiler, renderer, adapter, f'gather_l_o_{dynamic}_{nsa}', source, reference(1), 0)
            if result.returncode == 77:
                return 77
            assert result.returncode == 0 and f'compared_words={WORDS} mismatches=0' in result.stdout, (result.stdout, result.stderr)
    for nan in ('u', 'lod'):
        source = cs_pipe(shader(nan=nan)).hlsl_cs()
        want = [0] * 4 + [SENTINEL] * (WORDS - 4)
        result = run_case(directory, compiler, renderer, adapter, f'gather_l_o_nan_{nan}', source, want, 0)
        if result.returncode == 77:
            return 77
        assert result.returncode == 0 and f'compared_words={WORDS} mismatches=0' in result.stdout, (result.stdout, result.stderr)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RENDERER', 'ADAPTER'))
    args = parser.parse_args()
    if args.dispatch:
        (ROOT / 'out').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'out') as tmp:
            raise SystemExit(dispatch(Path(tmp), *args.dispatch))
    controls()
