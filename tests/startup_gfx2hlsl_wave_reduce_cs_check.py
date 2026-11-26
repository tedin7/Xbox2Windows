#!/usr/bin/env python3
"""Synthetic wave32 compute reductions and D3D12 readback controls."""
import functools
import operator
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_gfx2hlsl_wave_reduce_check as wave
import startup_gfx2hlsl_dpp_quad_swap_cs_check as base
from startup_gfx2hlsl_permlane16_cs_check import first_difference

ROOT = Path(__file__).resolve().parents[1]
GROUP = (64, 1, 1)
MASK = 0xffffffff
RSRC2 = base.USER7 | base.TGID_X | base.tidig(1)
COMBINE = dict(zip(wave.OPS, (operator.or_, min, max, operator.and_, operator.xor, operator.add)))
BITWISE = {'v_or_b32', 'v_and_b32', 'v_xor_b32'}
FORMS = ((True, False), (False, False), (True, True))  # (in-place, literal identity)
INTRINSICS = dict(zip(wave.OPS, ('WaveActiveBitOr', 'WaveActiveMin', 'WaveActiveMax',
                                'WaveActiveBitAnd', 'WaveActiveBitXor', 'WaveActiveSum')))


def program(op, *, inplace=True, literal=False, mask_reg=None, identity=None, live_other_row=False, branch=False):
    """Two guest waves, with different active-lane ranges and an observable restored EXEC."""
    setup = [
        wave.sop1(wave.S_MOV_B32, 10, 6),
        wave.sop1(wave.S_MOV_B32, 11, 128),
        base.S_LOAD_X8_S12_S10,
        base.S_WAITCNT_LGKM0,
        wave.vop2(base.V_LSHLREV_B32, 4, 132, 0),  # 16-byte records, indexed by local thread
        wave.vop2(wave.VOP2_OPS['v_add_nc_u32'], 5, 129, 0),
        wave.vop1_mov(9, 128),
        wave.sop1(wave.S_MOV_B32, 26, wave.EXEC_LO),
    ]
    if op in BITWISE:
        setup += [
            wave.vop2(0x16, 7, 133, 0),  # second guest wave shifts by one extra bit
            wave.vop2(wave.VOP2_OPS['v_add_nc_u32'], 7, 256, 7),
            wave.vop2(base.V_LSHLREV_B32, 5, 129, 7),
        ]
    if op == 'v_and_b32':
        setup += [wave.vop2(wave.VOP2_OPS['v_xor_b32'], 5, wave.INLINE_M1, 5)]
    # Keep lanes 1..30 of each wave, so both unsigned extrema detect a dropped mask.
    setup += [
        wave.vop2(wave.VOP2_OPS['v_add_nc_u32'], 6, 129, 0),
        wave.vop2(wave.VOP2_OPS['v_and_b32'], 6, 159, 6),
        # Compare inline 1 < v6 and also update EXEC (VOPC gfx10 CMPX_LT_U32).
        struct.pack('<I', 0x7c000000 | (0xd1 << 17) | (6 << 9) | 129),
    ]
    vt = 5 if inplace else 7
    chain = wave.chain(op, vt=vt, src=5, vp=20, ss=24, sd=25, identity=identity, fill_mask=mask_reg)
    if literal:
        value = wave.IDENTITY[op]
        fill = wave.vop3(wave.V_CNDMASK_B32_E64, vt, 255, 256 + 5, 24)
        chain[1] = ('literal identity fill', fill + struct.pack('<I', value))
    tail = [
        wave.vop1_mov(9, 25),  # old EXEC must mask this write
        wave.sop1(wave.S_MOV_B32, wave.EXEC_LO, 26),
        wave.vop1_mov(8, 25),
        wave.vop1_mov(10, 256 + (20 if live_other_row else vt)),
        wave.vop1_mov(11, 256),
        base.PROGRAM_TAIL_STORE,
    ]
    if branch:
        setup.insert(0, wave.sopp(0x02, 0))  # unconditional branch to the next instruction
    return b''.join(setup + [word for _, word in chain] + tail)


def translate(op, *, group=GROUP, wave_size=None, **options):
    return wave.G.ComputePipeline(program(op, **options), base.root_blob(), RSRC2, group,
                                  wave_size=wave_size).hlsl_cs()


def reference(op, include_inactive=False):
    out = [base.SENTINEL] * base.WORDS
    for start in range(0, 64, 32):
        values = []
        for x in range(start, start + 32):
            if include_inactive or 0 < x % 32 < 31:
                word = 1 << ((x + start // 32) % 32) if op in BITWISE else x + 1
                values.append(MASK ^ word if op == 'v_and_b32' else word)
        result = functools.reduce(COMBINE[op], values, wave.IDENTITY[op]) & MASK
        for x in range(start, start + 32):
            out[4 * x:4 * x + 4] = [result, result if 0 < x % 32 < 31 else 0, result, x]
    return out


def controls():
    for op, intrinsic in INTRINSICS.items():
        for inplace, literal in FORMS:
            hlsl = translate(op, inplace=inplace, literal=literal)
            assert f'{intrinsic}(word)' in hlsl, hlsl
            assert 'uint word = (((exec & 1u) != 0u) ? v5 :' in hlsl, hlsl
            assert 'first = flat & ~31u;' in hlsl, hlsl
            assert hlsl.count('GroupMemoryBarrierWithGroupSync();') == 2, hlsl
        first_difference(reference(op), reference(op, include_inactive=True))
    op = 'v_add_nc_u32'
    base.refused('the identity fill after s_or_saveexec_b32', lambda: translate(op, mask_reg=23))
    base.refused('the inactive lanes get', lambda: translate(op, identity=1))
    base.refused('whole guest waves', lambda: translate(op, group=(48, 1, 1)))
    base.refused('whole guest waves', lambda: translate(op, wave_size=64))
    base.refused('is read after the chain', lambda: translate(op, live_other_row=True))
    base.refused('straight-line control flow', lambda: translate(op, branch=True))
    print('compute wave reductions: six operations, in-place, separate and literal fills, six refusals')


def dispatch(dxc, runner, mode):
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    runner = str(Path(runner).resolve())
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        src, cso, case = (Path(tmp) / name for name in ('reduce.hlsl', 'reduce.cso', 'reduce.case'))
        for op in wave.OPS:
            expected = reference(op)
            case.write_bytes(struct.pack('<I', 0x010101) + struct.pack(f'<{len(expected)}I', *expected)
                             + bytes(64))
            cases = [(inplace, literal, False) for inplace, literal in FORMS]
            for inplace, literal, force_shared in cases + [(True, True, True)]:
                hlsl = translate(op, inplace=inplace, literal=literal)
                if mode == 'mutant':
                    marker = next(line for line in hlsl.splitlines() if 'uint word = ' in line)
                    hlsl = hlsl.replace(marker, '        uint word = v5;')
                if force_shared:
                    # Exercise the cross-host-wave fallback even on WARP with 32-lane waves.
                    guard = 'WaveActiveCountBits(true) == 32u && WaveActiveAllEqual(first)'
                    assert guard in hlsl
                    hlsl = hlsl.replace(guard, 'false')
                src.write_text(hlsl, encoding='ascii')
                subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
                run = subprocess.run([runner, 'warp', '--compute-store-wave-case', str(cso), str(case)],
                                     capture_output=True, text=True)
                print(run.stdout, end='')
                if run.returncode == 77:
                    return 77
                if mode == 'mutant':
                    want = first_difference(expected, reference(op, include_inactive=True))
                    if run.returncode != 1 or run.stderr.splitlines()[:1] != [want]:
                        raise AssertionError(f'{op}: expected mutant mismatch {want}, got {run.stderr}')
                else:
                    if run.returncode:
                        raise AssertionError(f'{op}: readback failed ({run.returncode}): {run.stderr}')
                print(f'{op}, inplace={inplace}, literal={literal}, shared={force_shared}, {mode}: verified')
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 4 and sys.argv[3] in ('dispatch', 'mutant'):
        sys.exit(dispatch(*sys.argv[1:]))
    else:
        sys.exit('usage: script [dxc offscreen_check dispatch|mutant]')
