"""Synthetic uniform-phi and guest first-active-lane compute readbacks."""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import startup_compute_waterfall as waterfall
from compute_load_format_check import check_references
from startup_compute_global_load_check import USER9, root_blob_cs
from startup_compute_readfirstlane_exec_check import REFUSAL, bytecode
from startup_gfx2hlsl_compute_check import SENTINEL, WORDS, refused

PHI = (
    ('s_cmp_eq_u32 s4, 7', '048706bf'),
    ('s_cbranch_scc1 3', '030085bf'),
    ('s_mov_b32 s21, 1', '810395be'),
    ('v_mov_b32_e32 v8, 1', '8102107e'),
    ('s_branch 2', '020082bf'),
    ('s_mov_b32 s21, 4', '840395be'),
    ('v_mov_b32_e32 v8, 4', '8402107e'),
)
XOR = ('v_xnor_b32 v9, v9, s5', '09001ed5090b0000')
DROP_XOR = ('v_xnor_b32 v9, v9, 0', '09001ed509010100')
WAIT = ('s_waitcnt vmcnt(0)', '703f8cbf')
USE = ('v_mov_b32_e32 v7, s22', '16020e7e')
VARYING = ('v_mov_b32_e32 v8, v0', '0003107e')
NARROW = ('v_cmpx_lt_u32_e32 s4, v0', '0400a27d')


def uniform_body(phi=PHI, xor=XOR):
    return (*waterfall.PREFIX[:5], *phi,
            ('v_add_nc_u32_e32 v8, s21, v8', '1510104a'),
            ('v_add_nc_u32_e32 v8, s7, v8', '0710104a'),
            ('v_lshlrev_b32_e32 v8, 2, v8', '82101034'),
            ('buffer_load_dword v9, v8, s[12:15], 0 offen', '001030e008090380'), WAIT, xor,
            ('v_mov_b32_e32 v10, v9', '0903147e'),
            ('v_readfirstlane_b32 s22, v10', '0a052c7e'),
            ('s_lshl_b32 s23, s7, 5', '0785178f'),
            ('v_add_nc_u32_e32 v4, s23, v4', '1708084a'), USE, waterfall.STORE, waterfall.END)


def scalar_body(xor=XOR):
    return (('s_mov_b32 s10, s8', '08038abe'), waterfall.PREFIX[1],
            ('s_load_dwordx4 s[16:19], s[10:11], 0', '050408f4000000fa'),
            ('s_load_dword s24, s[6:7], 4', '030600f4040000fa'), waterfall.PREFIX[3],
            ('v_mov_b32_e32 v9, s24', '1802127e'), xor,
            ('v_mov_b32_e32 v10, v9', '0903147e'),
            ('v_readfirstlane_b32 s22, v10', '0a052c7e'),
            waterfall.PREFIX[4], USE, waterfall.STORE, waterfall.END)


def active_body(width, mutant=False):
    if width == 64:
        save = ('s_mov_b64 s[26:27], exec', '7e049abe')
        restore = ('s_mov_b64 exec, s[26:27]', '1a04febe')
    else:
        save, restore = waterfall.PREFIX[-2], waterfall.RESTORE
    mask = () if mutant else (NARROW,)
    return (*waterfall.PREFIX[:7], save, *mask, waterfall.HEAD, restore, USE, waterfall.STORE, waterfall.END)


def translate(rows, width, llvm_mc=None, *, scalar=False, group=None, refusals=None):
    size = group or (4 if width else 8)
    rsrc2 = USER9 if scalar else waterfall.USER7 | (1 << 7 if width is None else 0)
    root = root_blob_cs() if scalar else waterfall.root_blob()
    return waterfall.G.ComputePipeline(bytecode(rows, llvm_mc), root,
                                       rsrc2, (size, 1, 1), wave_size=width, refusals=refusals).hlsl_cs()


def programs():
    yield 'uniform-phi', None, uniform_body(), uniform_body(xor=DROP_XOR)
    yield 'uniform-smem', None, scalar_body(), scalar_body(xor=DROP_XOR)
    for width in (32, 64):
        yield f'active-wave{width}', width, active_body(width), active_body(width, mutant=True)


def controls(llvm_mc):
    for label, width, rows, mutant in programs():
        scalar = label == 'uniform-smem'
        for variant in (rows, mutant):
            refusals = []
            hlsl = translate(variant, width, llvm_mc, scalar=scalar, refusals=refusals)
            assert not refusals, refusals
            assert hlsl == translate(variant, width, scalar=scalar), label
            if width:
                assert 'WaveReadLaneFirst(v5)' in hlsl and 'WaveActiveMin((exec & 1u)' in hlsl, hlsl
            else:
                assert 's22 = v10;' in hlsl and 'WaveReadLaneFirst' not in hlsl, hlsl
    # The fixture's offsets are for the small group. Test the uniform-source and host-wave proofs in isolation.
    with patch.object(waterfall.G.ComputePipeline, 'retry_group_vote_for', return_value=False):
        refused(REFUSAL, lambda: translate(uniform_body(phi=(*PHI[:-1], VARYING)), None, llvm_mc))
        for width in (32, 64):
            refused(REFUSAL, lambda: translate(active_body(width), width, llvm_mc, group=8))
    print('uniform phi and scalar load, first-active wave32/wave64 and refusal controls passed')


def references(width, key, salt, source, scalar):
    if width:
        selected = next((lane for lane in range(4) if lane > key), 0)
        return [source[selected]] * 4, source[0]
    if scalar:
        return [(~(source[1] ^ salt)) & 0xffffffff] * 8, ~source[1] & 0xffffffff
    index = 8 if key == 7 else 2
    return [(~(source[index + gid] ^ salt)) & 0xffffffff for gid in range(2) for _ in range(8)], (
        ~source[index] & 0xffffffff)


def dispatch(directory, dxc, runner, adapter):
    source = [29 + 53 * i for i in range(WORDS)]
    for label, width, rows, mutant in programs():
        scalar = label == 'uniform-smem'
        inputs = [(key, 0) for key in range(4)] if width else [(6, 13), (7, 29)]
        cases, mutations = [], []
        for key, salt in inputs:
            expected, wrong = references(width, key, salt, source, scalar)
            case = directory / f'{label}-{key}.case'
            words = [1 if width or scalar else 2, *expected, *([SENTINEL] * (WORDS - len(expected))),
                     key, salt, *([0] * 14), *source]
            case.write_bytes(struct.pack('<%dI' % len(words), *words))
            cases.append((case, f'key {key}', None))
            if wrong != expected[0]:
                mutations.append((case, f'key {key}, mutant',
                                  f'word=0 expected={expected[0]:08x} actual={wrong:08x}'))
        assert mutations, label
        for name, variant, checks in ((label, rows, cases), (f'{label}-mutant', mutant, mutations)):
            src, shader = directory / f'{name}.hlsl', directory / f'{name}.cso'
            src.write_text(translate(variant, width, scalar=scalar), encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
            if check_references(name, runner, adapter, shader, checks, allow_skip=True) == 77:
                return 77
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args()
    controls(args.llvm_mc)
    if args.dispatch:
        output = Path(__file__).resolve().parents[1] / 'out'
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            return dispatch(Path(tmp), *args.dispatch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
