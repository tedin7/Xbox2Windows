"""Single-guest-wave compute collectives, checked against a scalar RDNA 2 interpreter."""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import startup_compute_waterfall as waterfall
from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, refused, root_blob

EXECZ = 'inside a block s_cbranch_execz skips per lane: '
READFIRST = 'v_readfirstlane_b32 requires a proven uniform source'
WATERFALL = 'compute readfirstlane waterfall:'
BRANCHES = {
    's_branch': 0xbf82, 's_cbranch_scc1': 0xbf85, 's_cbranch_vccz': 0xbf86,
    's_cbranch_vccnz': 0xbf87, 's_cbranch_execz': 0xbf88, 's_cbranch_execnz': 0xbf89,
}
SET_3 = ('s_mov_b32 s20, 3', '830394be')
SET_40 = ('s_mov_b32 s20, 40', 'a80394be')
ARM = ('v_add_nc_u32_e32 v7, s20, v5', '140a0e4a')
BIAS = ('v_add_nc_u32_e32 v7, s20, v7', '140e0e4a')
SAVE64 = ('s_mov_b64 s[26:27], exec', '7e049abe')
RESTORE64 = ('s_mov_b64 exec, s[26:27]', '1a04febe')
USE_FIRST = ('v_add_nc_u32_e32 v7, s22, v7', '160e0e4a')
USE_COUNT = ('v_mov_b32_e32 v7, s24', '18020e7e')
INIT_COUNT = ('s_mov_b32 s24, 0', '800398be')
INC_COUNT = ('s_add_u32 s24, s24, 1', '18811880')
CMP_VCC = ('v_cmp_eq_i32_e32 vcc_lo, 7, v5', '870a047d')
NARROW_ABOVE = ('v_cmpx_lt_u32_e32 s4, v0', '0400a27d')


def encode(rows):
    offsets, instructions = {}, []
    pc = 0
    for row in rows:
        if isinstance(row, str):
            offsets[row] = pc
        else:
            instructions.append((pc, *row))
            pc += 4 if row[1].startswith('@') else len(row[1]) // 2
    words = []
    for pc, asm, code in instructions:
        if code.startswith('@'):
            delta = (offsets[code[1:]] - pc) // 4 - 1
            code = struct.pack('<I', BRANCHES[asm] << 16 | (delta & 0xffff)).hex()
        words.append(code)
    return bytes.fromhex(''.join(words))


def programs():
    prefix, restore = waterfall.PREFIX, waterfall.RESTORE
    ending = (waterfall.STORE, waterfall.END)
    scalar = (SET_3, ('s_cbranch_execz', '@join'), SET_40, ARM, 'join', restore, BIAS, *ending)
    yield 'execz-scalar', 32, (*prefix, *scalar), EXECZ + 'definition of s20', True
    # With a threshold of 32 or more, only the high half of guest EXEC is nonzero.
    yield 'execz-scalar64', 64, (*prefix[:-2], SAVE64, NARROW_ABOVE, *scalar[:5], RESTORE64, *scalar[6:]), \
        EXECZ + 'definition of s20', True
    yield 'execz-exit', 32, (*prefix, SET_3, ('s_cbranch_execz', '@join'), ARM,
                            ('s_cmp_eq_u32 s4, s4', '040406bf'), ('s_cbranch_scc1', '@tail'),
                            'join', SET_40, 'tail', restore, BIAS, *ending), EXECZ + 'branch out of the block', True
    for latch, reason in (('s_cbranch_scc1', WATERFALL), ('s_cbranch_execnz', READFIRST)):
        yield latch, 32, (*prefix, INIT_COUNT, ('s_cbranch_execz', '@tail'), waterfall.SAVE, 'head',
                          waterfall.HEAD, waterfall.COMPARE, INC_COUNT, waterfall.CLOSE, (latch, '@head'),
                          'tail', restore, USE_COUNT, *ending), reason, False
    for branch in ('s_cbranch_vccz', 's_cbranch_vccnz'):
        yield branch, 32, (*prefix, waterfall.HEAD, SET_3, CMP_VCC, (branch, '@join'), SET_40,
                           'join', restore, USE_FIRST, BIAS, *ending), READFIRST, False
    # Read after a different narrowing, so the result must change between an empty mask and a nonzero first lane.
    yield 'first-active', 32, (*prefix[:-1], NARROW_ABOVE, waterfall.HEAD,
                              restore, USE_FIRST, *ending), READFIRST, False


def interpret(rows, active_count, keys, lanes):
    """Execute one wave. The interpreter uses the assembly rows and never reads generated HLSL."""
    labels = {}
    instructions = []
    for row in rows:
        if isinstance(row, str):
            labels[row] = len(instructions)
        else:
            instructions.append(row)
    lanes = set(lanes)
    vgpr = {lane: {'v0': lane} for lane in lanes}
    sgpr = {'s4': active_count, 's6': 0}
    output = {}
    exec_mask, vcc, scc = lanes.copy(), set(), False

    def value(token, lane=None):
        if token in ('exec', 'exec_lo'):
            return exec_mask.copy()
        if token.startswith('s'):
            return sgpr[token]
        if token.startswith('v'):
            return vgpr[lane][token]
        return int(token, 0)

    pc = 0
    for _ in range(1024):
        asm, encoding = instructions[pc]
        pc += 1
        op, _, operands = asm.partition(' ')
        op = op.removesuffix('_e32')
        dst, src0, src1, *_ = operands.split(', ') + [None] * 3
        if op in BRANCHES:
            take = {'s_branch': True, 's_cbranch_execz': not exec_mask, 's_cbranch_execnz': bool(exec_mask),
                    's_cbranch_vccz': not vcc, 's_cbranch_vccnz': bool(vcc), 's_cbranch_scc1': scc}[op]
            if take:
                pc = labels[encoding[1:]]
        elif op == 's_endpgm':
            return output
        elif op in ('s_load_dwordx8', 's_waitcnt'):
            continue  # the fixture binds its input and output directly below
        elif op in ('s_mov_b32', 's_mov_b64'):
            word = value(src0)
            if dst in ('exec', 'exec_lo'):
                exec_mask = word.copy()
            else:
                sgpr[dst] = word
        elif op == 's_add_u32':
            total = value(src0) + value(src1)
            sgpr[dst], scc = total & 0xffffffff, total > 0xffffffff
        elif op == 's_cmp_eq_u32':
            scc = value(dst) == value(src0)
        elif op == 's_andn2_wrexec_b32':
            exec_mask = value(src0) - exec_mask
            sgpr[dst], scc = exec_mask.copy(), bool(exec_mask)
        elif op == 'v_readfirstlane_b32':
            sgpr[dst] = value(src0, min(exec_mask, default=0))
        elif op in ('v_cmpx_gt_u32', 'v_cmpx_lt_u32', 'v_cmpx_eq_i32'):
            lhs, rhs = dst, src0
            exec_mask = {lane for lane in exec_mask if {
                'v_cmpx_gt_u32': value(lhs, lane) > value(rhs, lane),
                'v_cmpx_lt_u32': value(lhs, lane) < value(rhs, lane),
                'v_cmpx_eq_i32': value(lhs, lane) == value(rhs, lane),
            }[op]}
        elif op == 'v_cmp_eq_i32':
            vcc = {lane for lane in exec_mask if value(src0, lane) == value(src1, lane)}
        else:
            for lane in exec_mask:
                if op == 'buffer_store_dword':
                    output[value(src0, lane) // 4] = value(dst, lane)
                    continue
                if op == 'buffer_load_dword':
                    word = keys[value(src0, lane) // 4]
                elif op == 'v_mov_b32':
                    word = value(src0, lane)
                elif op == 'v_lshlrev_b32':
                    word = value(src1, lane) << value(src0, lane)
                elif op == 'v_add_nc_u32':
                    word = value(src0, lane) + value(src1, lane)
                else:
                    raise AssertionError(asm)
                vgpr[lane][dst] = word & 0xffffffff
    raise AssertionError('fixture did not terminate')


def translate(rows, width=None, group=(32, 1, 1), refusals=None):
    binary = encode(rows)
    return G.ComputePipeline(binary, root_blob(), USER7, group, wave_size=width, refusals=refusals)


def reference_cases(label, width, rows, mutate):
    keys = ([7, 12, 7, 5, 12, 7, 18, 5] * 8)[:width]
    for count in ((0, 5, 32) if width == 32 else (0, 5, 32, 37, 64)):
        result = interpret(rows, count, keys, range(width))
        expected = [result[i] for i in range(width)]
        if label in ('s_cbranch_scc1', 's_cbranch_execnz'):
            assert expected == [len(set(keys[:count]))] * width, (label, count, expected)
        if label == 'first-active':
            selected = count + 1 if count + 1 < width else 0
            assert expected == [13 + keys[selected]] * width, (count, expected)
        mismatch = None
        if mutate:
            for lane, correct in enumerate(expected):
                wrong = interpret(rows, count, keys, (lane,))[lane]
                if correct != wrong:
                    mismatch = f'word={lane} expected={correct:08x} actual={wrong:08x}'
                    break
        yield count, keys, expected, mismatch


def lane_mutant(hlsl):
    hlsl, count = re.subn(r'^bool x2w_wave_any\(uint lane, bool bit\) \{\n.*?^\}',
                          'bool x2w_wave_any(uint lane, bool bit) {\n    return bit;\n}', hlsl, flags=re.M | re.S)
    assert count == 1
    return hlsl


def controls():
    for label, width, rows, legacy_reason, mutate in programs():
        group = (width, 1, 1)
        with patch.object(G.ComputePipeline, 'retry_group_vote_for', return_value=False):
            refused(legacy_reason, lambda: translate(rows, width, group))
            legacy_census = []
            translate(rows, width, group, refusals=legacy_census)
            assert legacy_reason in legacy_census[0][0], (label, legacy_census)
        pipe = translate(rows, width, group)
        assert pipe.retry_group_vote and pipe.cs.uniform_cf, label
        assert not pipe.cs.execz_counted_loops and not pipe.cs.waterfall_reads, label
        hlsl = pipe.hlsl_cs()
        census = [('earlier shader', 'ps')]
        counted = translate(rows, width, group, refusals=census)
        assert census == [('earlier shader', 'ps')], (label, census)
        assert (counted.hlsl_cs(), counted.manifest()) == (hlsl, pipe.manifest()), label
        assert not re.search(r'\bWave\w*\(', hlsl), label
        main = hlsl.split('void main', 1)[1]
        assert f'uint lane = tid.x + {width}u * (tid.y + 1u * tid.z);' in main, label
        needs_first = waterfall.HEAD in rows
        needs_any = label != 'first-active'
        assert ('x2w_wave_first(lane,' in main) == needs_first, label
        assert ('x2w_wave_any(lane,' in main) == needs_any, label
        assert not re.search(r'if \(exec & 1\).*x2w_wave_first', main), label
        assert hlsl.count('GroupMemoryBarrierWithGroupSync();') == 3 * (needs_first + needs_any), label
        if label == 's_cbranch_scc1':
            assert 'scc = x2w_wave_any(lane, exec != 0u);' in main and 'pc = (scc)' in main
        if label == 's_cbranch_execnz':
            assert 'pc = (x2w_wave_any(lane, (exec & 1u) != 0u))' in main
        if label in ('s_cbranch_vccz', 's_cbranch_vccnz'):
            negate = '!' if label == 's_cbranch_vccz' else ''
            assert f'pc = ({negate}x2w_wave_any(lane, (vcc & 1u) != 0u))' in main
        if label == 'execz-scalar64':
            hlsl_2d = translate(rows, group=(8, 8, 1)).hlsl_cs()
            assert 'uint lane = tid.x + 8u * (tid.y + 8u * tid.z);' in hlsl_2d
            refused(legacy_reason, lambda: translate(rows, 32, group))
        cases = list(reference_cases(label, width, rows, mutate))
        if mutate:
            assert any(diagnostic for _, _, _, diagnostic in cases), label
            assert 'return bit;' in lane_mutant(hlsl)

    unproven = (*waterfall.PREFIX[:8], waterfall.HEAD, USE_FIRST, waterfall.STORE, waterfall.END)
    assert not G.wave64_exec_proof(G.decode(encode(unproven)))
    assert translate(unproven).retry_group_vote
    refused(READFIRST, lambda: translate(unproven, group=(64, 1, 1)))
    refused(READFIRST, lambda: translate(unproven, 64, (65, 1, 1)))
    unrelated = (*unproven[:8], ('s_sleep 1', '01008ebf'), *unproven[8:])
    refusals = []
    assert not translate(unrelated, 32, refusals=refusals).retry_group_vote
    assert refusals == [('s_sleep', 'cs'), (READFIRST, 'cs')], refusals

    varying_scc = (*waterfall.PREFIX, waterfall.HEAD, ('s_and_b32 s25, exec_lo, -1', '7ec11987'),
                   ('s_cbranch_scc1', '@tail'), ('v_mov_b32_e32 v7, 7', '87020e7e'), 'tail',
                   waterfall.RESTORE, USE_FIRST, waterfall.STORE, waterfall.END)
    branch_error = 'neither wave-uniform nor an EXEC / VCC vote'
    refused(branch_error, lambda: translate(varying_scc, 32))
    refusals = []
    translate(varying_scc, 32, refusals=refusals)
    assert branch_error in refusals[0][0], refusals
    scc_as_data = (*varying_scc[:12], ('s_cselect_b32 s20, 3, 40', '83a81485'),
                   waterfall.RESTORE, BIAS, waterfall.STORE, waterfall.END)
    refused('scc from an unproved wave mask', lambda: translate(scc_as_data, 32))

    legacy = (*waterfall.PREFIX[:8], waterfall.STORE, waterfall.END)
    pipe = translate(legacy, 32)
    with patch.object(G.ComputePipeline, 'retry_group_vote_for', return_value=False) as retry:
        original = translate(legacy, 32)
        retry.assert_not_called()
    assert not pipe.retry_group_vote and 'x2w_wave_' not in pipe.hlsl_cs()
    assert (pipe.hlsl_cs(), pipe.manifest()) == (original.hlsl_cs(), original.manifest())
    print('Eight group-vote fixtures, wave-size and divergent SCC refusal controls passed')


def dispatch(directory, dxc, runner, adapter):
    force_mutant = os.environ.get('X2W_TEST_GROUP_VOTE_MUTANT') == 'lane'
    for label, width, rows, _, mutate in programs():
        hlsl = translate(rows, width, (width, 1, 1)).hlsl_cs()
        cases, mutations = [], []
        for count, keys, expected, diagnostic in reference_cases(label, width, rows, mutate):
            case = directory / f'{label}-{count}.case'
            words = [1, *expected, *([SENTINEL] * (WORDS - width)), count, *([0] * 15),
                     *keys, *([0] * (WORDS - width))]
            case.write_bytes(struct.pack('<%dI' % len(words), *words))
            cases.append((case, f'active={count}', None))
            if diagnostic:
                mutations.append((case, f'active={count}, lane mutant', diagnostic))
        shaders = [(label, lane_mutant(hlsl) if mutate and force_mutant else hlsl, cases)]
        if mutate and not force_mutant:
            # These fixtures have no collective after the branch, so the mutant can finish after PCs diverge.
            assert mutations, label
            shaders.append((f'{label}-lane-mutant', lane_mutant(hlsl), mutations))
        for name, source, checks in shaders:
            src, shader = directory / f'{name}.hlsl', directory / f'{name}.cso'
            src.write_text(source, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
            if check_references(name, runner, adapter, shader, checks, allow_skip=True) == 77:
                return 77
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args()
    controls()
    if args.dispatch:
        output = Path(__file__).resolve().parents[1] / 'out'
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            return dispatch(Path(tmp), *args.dispatch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
