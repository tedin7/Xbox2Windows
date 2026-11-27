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

import startup_compute_readfirstlane_narrow_check as narrow
import startup_compute_waterfall as waterfall
from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, refused, root_blob

EXECZ = 'inside a block s_cbranch_execz skips per lane: '
READFIRST = 'v_readfirstlane_b32 requires a proven uniform source'
WATERFALL = 'compute readfirstlane waterfall:'
BRANCHES = {
    's_branch': 0xbf82, 's_cbranch_scc0': 0xbf84, 's_cbranch_scc1': 0xbf85, 's_cbranch_vccz': 0xbf86,
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
EXEC_ONE = ('s_mov_b32 exec_lo, 1', '8103febe')
REMAINING_ONE = ('s_mov_b32 s21, 1', '810395be')
EXEC_HIGH_PAIR = (('s_mov_b32 s30, 0', '80039ebe'), ('s_mov_b32 s31, 1', '81039fbe'),
                  ('s_mov_b64 exec, s[30:31]', '1e04febe'))


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
    rows = narrow.program(after_cmpx=(EXEC_ONE,))
    i = rows.index(narrow.BRANCH[1])
    rows = (*rows[:i], ('s_cbranch_scc0', '@load'), rows[i + 1], 'load', *rows[i + 2:])
    assert encode(rows) == bytes.fromhex(''.join(code for _, code in narrow.program(after_cmpx=(EXEC_ONE,))))
    yield 'exec-constant', 32, rows, READFIRST, False
    i = rows.index(EXEC_ONE)
    # Reconstruct the flat lane in v0 before using it for buffer offsets and compare inputs.
    mbcnt = (('v_mbcnt_lo_u32_b32 v0, -1, 0', '000065d7c1000100'),
             ('v_mbcnt_hi_u32_b32 v0, -1, v0', '000066d7c1000200'))
    entry = ('v_readfirstlane_b32 s22, v0', '00052c7e')  # Force the retry before MBCNT, whose legacy path is wave32.
    yield 'exec-constant-pair', 64, (entry, *mbcnt, *rows[:i - 2], SAVE64, rows[i - 1], *EXEC_HIGH_PAIR,
                                    *rows[i + 1:-3], RESTORE64, *ending), READFIRST, False
    rows = waterfall.program(after_save=(REMAINING_ONE,))
    i, j = rows.index(waterfall.HEAD), rows.index(waterfall.CLOSE) + 1
    loop = (*rows[:i], 'head', *rows[i:j], ('s_cbranch_scc1', '@head'), *rows[j + 1:])
    assert encode(loop) == bytes.fromhex(''.join(code for _, code in rows))
    yield 'waterfall-constant', 32, loop, WATERFALL + ' remaining mask is not an unchanged copy of EXEC', False
    # The VCC and SGPR phis both combine EXEC with a scalar word. The two selects make each bit observable.
    yield 'mask-word-phi', 32, (*prefix, SET_3, ('s_mov_b32 vcc_lo, 0', '8003eabe'),
                               ('s_and_b32 s24, s4, exec_lo', '047e1887'),
                               ('s_cbranch_execz', '@join'), SET_40,
                               ('s_and_b32 vcc_lo, s4, exec_lo', '047e6a87'),
                               ('s_and_b32 s24, s24, exec_lo', '187e1887'), 'join', restore,
                               ('v_cndmask_b32_e32 v7, s20, v7, vcc_lo', '140e0e02'),
                               ('v_cndmask_b32_e64 v8, 0, 64, s24', '080001d580806100'),
                               ('v_add_nc_u32_e32 v7, v8, v7', '080f0e4a'), *ending), EXECZ + 'definition of s20', False


def fixture_group(label, width):
    if label in ('exec-constant', 'waterfall-constant'):
        return (4, 1, 1)
    if label == 'exec-constant-pair':
        return (8, 4, 2)
    return (width, 1, 1)


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
    entry_mask = sum(1 << lane for lane in lanes)
    exec_mask, vcc, scc = entry_mask, 0, False

    def value(token, lane=None):
        if token == 'exec':
            return exec_mask
        if token == 'exec_lo':
            return exec_mask & 0xffffffff
        if token in ('vcc', 'vcc_lo'):
            return vcc
        if token.startswith('s['):
            lo, hi = token[2:-1].split(':')
            return sgpr['s' + lo] | (sgpr['s' + hi] << 32)
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
        op = op.removesuffix('_e32').removesuffix('_e64')
        dst, src0, src1, mask, *_ = operands.split(', ') + [None] * 4
        if op in BRANCHES:
            take = {'s_branch': True, 's_cbranch_execz': not exec_mask, 's_cbranch_execnz': bool(exec_mask),
                    's_cbranch_vccz': not vcc, 's_cbranch_vccnz': bool(vcc),
                    's_cbranch_scc0': not scc, 's_cbranch_scc1': scc}[op]
            if take:
                pc = labels[encoding[1:]]
        elif op == 's_endpgm':
            return output
        elif op in ('s_load_dwordx8', 's_waitcnt', 's_nop'):
            continue  # the fixture binds its input and output directly below
        elif op in ('s_mov_b32', 's_mov_b64'):
            word = value(src0)
            if dst == 'exec':
                exec_mask = word & entry_mask
            elif dst == 'exec_lo':
                exec_mask = ((exec_mask & ~0xffffffff) | (word & 0xffffffff)) & entry_mask
            elif dst == 'vcc_lo':
                vcc = word & 0xffffffff
            elif dst.startswith('s['):
                lo, hi = dst[2:-1].split(':')
                sgpr['s' + lo], sgpr['s' + hi] = word & 0xffffffff, (word >> 32) & 0xffffffff
            else:
                sgpr[dst] = word & 0xffffffff
        elif op == 's_add_u32':
            total = value(src0) + value(src1)
            sgpr[dst], scc = total & 0xffffffff, total > 0xffffffff
        elif op == 's_and_b32':
            word = value(src0) & value(src1)
            if dst == 'vcc_lo':
                vcc = word
            else:
                sgpr[dst] = word
            scc = bool(word)
        elif op == 's_cmp_eq_u32':
            scc = value(dst) == value(src0)
        elif op == 's_andn2_wrexec_b32':
            low = value(src0) & ~exec_mask & entry_mask & 0xffffffff
            sgpr[dst], scc = low, bool(low)
            exec_mask = (exec_mask & ~0xffffffff) | low
        elif op == 'v_readfirstlane_b32':
            sgpr[dst] = value(src0, (exec_mask & -exec_mask).bit_length() - 1 if exec_mask else min(lanes))
        elif op in ('v_cmpx_gt_u32', 'v_cmpx_lt_u32', 'v_cmpx_eq_i32'):
            lhs, rhs = dst, src0
            exec_mask = sum(1 << lane for lane in lanes if exec_mask & (1 << lane) and {
                'v_cmpx_gt_u32': value(lhs, lane) > value(rhs, lane),
                'v_cmpx_lt_u32': value(lhs, lane) < value(rhs, lane),
                'v_cmpx_eq_i32': value(lhs, lane) == value(rhs, lane),
            }[op])
        elif op == 'v_cmp_eq_i32':
            vcc = sum(1 << lane for lane in lanes if exec_mask & (1 << lane) and value(src0, lane) == value(src1, lane))
        else:
            for lane in sorted(lanes):
                if not exec_mask & (1 << lane):
                    continue
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
                elif op == 'v_xor_b32':
                    word = value(src0, lane) ^ value(src1, lane)
                elif op == 'v_cndmask_b32':
                    word = value(src1 if value(mask) & (1 << lane) else src0, lane)
                elif op in ('v_mbcnt_lo_u32_b32', 'v_mbcnt_hi_u32_b32'):
                    start = 32 if '_hi_' in op else 0
                    below = ((1 << lane) - 1) >> start
                    word = (value(src0, lane) & below & 0xffffffff).bit_count() + value(src1, lane)
                else:
                    raise AssertionError(asm)
                vgpr[lane][dst] = word & 0xffffffff
    raise AssertionError('fixture did not terminate')


def translate(rows, width=None, group=(32, 1, 1), refusals=None):
    binary = encode(rows)
    return G.ComputePipeline(binary, root_blob(), USER7, group, wave_size=width, refusals=refusals)


def reference_cases(label, width, rows, mutate):
    x, y, z = fixture_group(label, width)
    threads = x * y * z
    keys = ([7, 12, 7, 5, 12, 7, 18, 5] * 8)[:threads]
    counts = (0, 5, 32) if width == 32 else (0, 5, 32, 37, 64)
    if label == 'exec-constant':
        counts = (0, 4)
    if label == 'waterfall-constant':
        keys, counts = [64, 32, 128, 96], (4,)
    for count in counts:
        result = interpret(rows, count, keys, range(threads))
        expected = [result[i] for i in range(threads)]
        if label in ('s_cbranch_scc1', 's_cbranch_execnz'):
            assert expected == [len(set(keys[:count]))] * width, (label, count, expected)
        if label == 'first-active':
            selected = count + 1 if count + 1 < width else 0
            assert expected == [13 + keys[selected]] * width, (count, expected)
        if label == 'exec-constant':
            assert expected == [20, 13, 13, 13], (count, expected)
        if label == 'exec-constant-pair':
            assert expected == [20 if lane == 32 else 13 for lane in range(64)], (count, expected)
        if label == 'waterfall-constant':
            assert expected == [81, 13, 13, 13], expected
        if label == 'mask-word-phi':
            assert expected == [77 if lane < count and count & (1 << lane) else 40 if count else 3
                                for lane in range(width)], (count, expected)
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


def mask_operand_controls(rows):
    # Keep the read of EXEC=1 that selects the retry, then exercise the other mask consumers.
    i = rows.index(narrow.USE)
    select = ('v_cndmask_b32_e64 v5, 3, v0, s24', '050001d583006200')
    probes = (
        ('s_mov_b32 s24, s4', '040398be'), select,
        ('v_add_co_ci_u32_e64 v7, s0, v5, v7, s24', '070028d5050f6200'),
        ('s_mov_b32 vcc_lo, 2', '8203eabe'), ('s_cbranch_vccz', '@after_vcc'), narrow.BRANCH[-1], 'after_vcc',
        ('s_and_b32 s24, vcc_lo, 2', '6a821887'), select,
        ('s_mov_b32 s28, 2', '82039cbe'), ('s_and_saveexec_b32 s28, s28', '1c3c9cbe'),
        ('s_cselect_b32 exec_lo, 2, 4', '82847e85'), ('s_or_b32 exec_lo, exec_lo, 2', '7e827e88'),
    )
    group = (4, 1, 1)
    pipe = translate((*rows[:i], *probes, *rows[i:]), 32, group)
    assert pipe.retry_group_vote
    hlsl = pipe.hlsl_cs()
    for snippet in (
        '((s24 >> lane) & 1u) & 1) ? v0', 'a = a + (((s24 >> lane) & 1u) & 1u);',
        'vcc = (((0x00000002u >> lane) & 1u)) & 1;', '!x2w_wave_any(lane, (vcc & 1u) != 0u)',
        's24 = (vcc & ((0x00000002u >> lane) & 1u)) & 1u;', '(s24 & 1) ? v0',
        'w = ((0x00000002u >> lane) & 1u);\n        s28 = exec;\n        exec = (exec & w) & 1;',
        'exec = (scc ? ((0x00000002u >> lane) & 1u) : ((0x00000004u >> lane) & 1u)) & 1;',
        'exec = (exec | ((0x00000002u >> lane) & 1u)) & 1; scc = x2w_wave_any(lane, exec != 0u);',
    ):
        assert snippet in hlsl, snippet
    data_read = ('v_mov_b32_e32 v7, s24', '18020e7e')
    refused('data operand s24 is a lane mask', lambda: translate((*rows[:i], *probes, data_read, *rows[i:]), 32, group))


def controls():
    for label, width, rows, legacy_reason, mutate in programs():
        group = fixture_group(label, width)
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
        assert f'uint lane = tid.x + {group[0]}u * (tid.y + {group[1]}u * tid.z);' in main, label
        needs_first = any(row[0].startswith('v_readfirstlane_b32') for row in rows if isinstance(row, tuple))
        needs_any = label not in ('first-active', 'exec-constant', 'exec-constant-pair')
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
        if label == 'exec-constant':
            assert 'exec = (((0x00000001u >> lane) & 1u)) & 1;' in main
            mask_operand_controls(rows)
        if label == 'exec-constant-pair':
            assert 'lane < 32u ? ((0x00000000u >> (lane & 31u)) & 1u)' in main
            assert ': ((0x00000001u >> (lane & 31u)) & 1u)' in main
            assert 'countbits(0xffffffffu & (lane < 32u ?' in main
            assert 'countbits(0xffffffffu & ((1u << (max(lane, 32u) - 32u)) - 1u))' in main
        if label == 'waterfall-constant':
            assert '((s21 >> lane) & 1u) & ~exec' in main
            assert 's21 = (lane < 32u ? (exec << (lane & 31u)) : 0u);' in main
        if label == 'mask-word-phi':
            assert 'vcc = (((ud0 >> lane) & 1u) & exec) & 1;' in main
            assert 's24 = (((ud0 >> lane) & 1u) & exec) & 1u;' in main
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
    print('Twelve group-vote fixtures, wave-size and divergent SCC refusal controls passed')


def dispatch(directory, dxc, runner, adapter):
    force_mutant = os.environ.get('X2W_TEST_GROUP_VOTE_MUTANT') == 'lane'
    for label, width, rows, _, mutate in programs():
        hlsl = translate(rows, width, fixture_group(label, width)).hlsl_cs()
        cases, mutations = [], []
        for count, keys, expected, diagnostic in reference_cases(label, width, rows, mutate):
            case = directory / f'{label}-{count}.case'
            words = [1, *expected, *([SENTINEL] * (WORDS - len(expected))), count, *([0] * 15),
                     *keys, *([0] * (WORDS - len(keys)))]
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
