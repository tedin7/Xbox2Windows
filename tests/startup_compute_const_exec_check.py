"""Multi-wave constant EXEC prologues, with an independent guest-wave reference."""
import argparse
import math
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import startup_compute_waterfall as waterfall
from compute_load_format_check import check_references
from startup_compute_readfirstlane_exec_check import FORMAT, READ, REFUSAL, VM_WAIT, bytecode
from startup_gfx2hlsl_compute_check import MASK, SENTINEL, WORDS, refused, tidig


SAVE64 = ('s_mov_b64 s[26:27], exec', '7e049abe')
RESTORE64 = ('s_mov_b64 exec, s[26:27]', '1a04febe')
ALL64 = ('s_mov_b64 exec, -1', 'c104febe')
ALL32 = ('s_mov_b32 exec_lo, -1', 'c103febe')
ONE64 = ('s_mov_b64 exec, 1', '8104febe')
ONE32 = ('s_mov_b32 exec_lo, 1', '8103febe')
ZERO64 = ('s_mov_b64 exec, 0', '8004febe')
TWO64 = ('s_mov_b64 exec, 2', '8204febe')
INVERT64 = ('s_andn2_b64 exec, -1, exec', 'c17efe8a')
INVERT32 = ('s_andn2_b32 exec_lo, -1, exec_lo', 'c17e7e8a')
HIGH64 = (('s_mov_b32 s30, 0', '80039ebe'), ('s_mov_b32 s31, 1', '81039fbe'),
          ('s_mov_b64 exec, s[30:31]', '1e04febe'))
INDEX = ('v_mov_b32_e32 v9, s20', '1402127e')
VARYING_INDEX = ('v_mov_b32_e32 v9, v0', '0003127e')
SMEM = ('s_buffer_load_dword s20, s[12:15], 0', '060520f4000000fa')
OLD_MARKER, NEW_MARKER = 13, 57
PIECE_VALUES = (11, 29, 23)  # root s4, loaded s20, root s5 in readfirstlane order
PIECE_INPUTS = ('ud0', 's20', 'ud1')
PIECE_WRITES = (('v_mov_b32_e32 v10, s4', '0402147e'),
                ('v_mov_b32_e32 v11, s5', '0502167e'),
                ('v_mov_b32_e32 v12, s20', '1402187e'))
PIECE_ZEROS = (('v_mov_b32_e32 v10, 0', '8002147e'),
              ('v_mov_b32_e32 v11, 0', '8002167e'),
              ('v_mov_b32_e32 v12, 0', '8002187e'))
PIECE_READS = (('v_readfirstlane_b32 s23, v10', '0a052e7e'),
               ('v_readfirstlane_b32 s24, v12', '0c05307e'),
               ('v_readfirstlane_b32 s25, v11', '0b05327e'))
PIECE_PACK = (
    (('v_lshlrev_b32_e32 v7, 16, v10', '90140e34'), ('v_or_b32_e32 v7, s23, v7', '170e0e38')),
    (('v_lshlrev_b32_e32 v7, 16, v12', '90180e34'), ('v_or_b32_e32 v7, s24, v7', '180e0e38')),
    (('v_lshlrev_b32_e32 v7, 16, v11', '90160e34'), ('v_or_b32_e32 v7, s25, v7', '190e0e38')),
)
LANE_GROUP = (128, 1, 1)
LANE_COMPARE = ('v_cmp_gt_u32_e32 vcc, s4, v0', '0400887d')
LANE_SAVE = ('s_and_saveexec_b64 s[26:27], vcc', '6a249abe')
LANE_RESTORE = ('s_or_b64 exec, exec, s[26:27]', '7e1afe88')
LANE_STORE = ('buffer_store_dword v7, v4, s[16:19], 0 offen offset:512', '001270e004070480')
MIXED_MASK = (*HIGH64[:2], ('s_and_b64 exec, vcc, s[30:31]', '6a1efe87'))
LANE_THRESHOLDS = (0, 40, 97)
UNKNOWN_EXEC = 'compute constant EXEC: s_andn2_b64 writes EXEC from an unknown whole-word value'
WAVE32_EXEC_HI = 'compute constant EXEC: s_mov_b32 writes EXEC_HI in wave32'
WAVE32_SAVE = 'compute constant EXEC: s_and_saveexec_b64 writes 64-bit SAVEEXEC in wave32'
MISSING_OLD_EXEC = 'compute constant EXEC: s_mov_b32 needs a known old mask for a partial EXEC write'


def flat_id(group):
    if group[1] == 1:
        return (('v_mov_b32_e32 v3, v0', '0003067e'),)
    return (('v_lshlrev_b32_e32 v3, 4, v1', '84020634'),
            ('v_add_nc_u32_e32 v3, v0, v3', '0007064a'))


def program(group, width, *, saved=False, mask=None, index=INDEX, before_read=(), after_restore=()):
    width_mask = ('v_and_b32_e32 v7, 63, v3', 'bf060e36') if width == 64 else (
        'v_and_b32_e32 v7, 31, v3', '9f060e36')
    save = ((SAVE64 if width == 64 else waterfall.PREFIX[-2]),) if saved else ()
    if width == 64:
        restore = RESTORE64 if saved else ALL64
    else:
        restore = waterfall.RESTORE if saved else ALL32
    if mask is None:
        mask = (ONE64 if width == 64 else ONE32,)
    return (*waterfall.PREFIX[:4], *flat_id(group), width_mask, ('v_mov_b32_e32 v8, 13', '8d02107e'),
            *save, *mask, SMEM, waterfall.PREFIX[3], ('s_add_u32 s20, s20, s4', '14041480'),
            ('s_and_b32 s20, s20, 3', '14831487'), ('s_lshl_b32 s20, s20, 1', '1481148f'),
            index, FORMAT, VM_WAIT, *before_read, READ, ('v_mov_b32_e32 v8, 57', 'b902107e'),
            restore, *after_restore, ('v_add_nc_u32_e32 v7, s22, v7', '160e0e4a'),
            ('v_and_b32_e32 v8, s5, v8', '05101036'), ('v_xor_b32_e32 v7, v8, v7', '080f0e3a'),
            ('v_lshlrev_b32_e32 v4, 2, v3', '82060834'), waterfall.STORE, waterfall.END)


def piecewise_program(group, width, component, *, switch=None, writes=PIECE_WRITES,
                      zeros=PIECE_ZEROS, restore=None):
    one, all_lanes = (ONE64, ALL64) if width == 64 else (ONE32, ALL32)
    if switch is None:
        switch = (INVERT64 if width == 64 else INVERT32,)
    if restore is None:
        restore = (all_lanes,)
    return (*waterfall.PREFIX[:4], *flat_id(group), SMEM, waterfall.PREFIX[3],
            one, INDEX, READ, *writes, *switch, *zeros,
            ('s_buffer_load_dword s20, s[12:15], 0x4', '060520f4040000fa'), waterfall.PREFIX[3],
            *restore, *PIECE_READS, *PIECE_PACK[component],
            ('v_lshlrev_b32_e32 v4, 2, v3', '82060834'), waterfall.STORE, waterfall.END)


def translate(rows, group, width, refusals=None):
    rsrc2 = waterfall.USER7 | tidig(2 if group[1] > 1 else 1)
    return waterfall.G.ComputePipeline(bytecode(rows), waterfall.root_blob(), rsrc2, group,
                                       wave_size=width, refusals=refusals)


def programs():
    yield 'wave64-1d', (128, 1, 1), 64, 1, False, None
    yield 'wave64-2d', (16, 16, 1), 64, 1, True, None
    yield 'wave32-1d', (64, 1, 1), 32, 1, True, None
    yield 'wave64-high-bit', (128, 1, 1), 64, 1 << 32, False, HIGH64


def piecewise_cases():
    yield 'pieces-wave64-2d', (16, 16, 1), 64
    yield 'pieces-wave64-1d', (128, 1, 1), 64
    yield 'pieces-wave32-1d', (64, 1, 1), 32


def reference(group, width, mask, key, marker_bits, source, *, broadcast=False):
    """Interpret the fixture per wave. A VGPR assignment only updates the EXEC-selected lanes."""
    output = [SENTINEL] * WORDS
    for start in range(0, math.prod(group), width):
        lanes = range(min(width, math.prod(group) - start))
        entry_exec = (1 << len(lanes)) - 1
        vgpr = [0] * len(lanes)
        marker = [OLD_MARKER] * len(lanes)
        exec_mask = entry_exec if broadcast else mask & entry_exec
        active = [lane for lane in lanes if exec_mask & (1 << lane)]
        index = ((source[0] + key) & 3) << 1
        for lane in active:
            vgpr[lane] = index
        for lane in active:
            vgpr[lane] = source[vgpr[lane]]
        first = vgpr[active[0] if active else 0]
        for lane in active:
            marker[lane] = NEW_MARKER
        exec_mask = entry_exec
        for lane in lanes:
            if exec_mask & (1 << lane):
                output[start + lane] = ((first + lane) & MASK) ^ (marker[lane] & marker_bits)
    return output


def piecewise_reference(group, width, values, component):
    """Interpret the three writes and reads in each wave, using per-lane VGPR storage."""
    output = [SENTINEL] * WORDS
    for start in range(0, math.prod(group), width):
        lanes = range(min(width, math.prod(group) - start))
        wave = (1 << len(lanes)) - 1
        vgprs = [[None] * len(lanes) for _ in values]
        for exec_mask, writes in ((1, values), (wave & ~1, (0, 0, 0))):
            for reg, value in zip(vgprs, writes):
                for lane in lanes:
                    if exec_mask & (1 << lane):
                        reg[lane] = value
        exec_mask = wave
        first = next(lane for lane in lanes if exec_mask & (1 << lane))
        readback = [reg[first] for reg in vgprs]
        for lane in lanes:
            output[start + lane] = (vgprs[component][lane] << 16) | readback[component]
    return output


def lane_program(*, narrow=(), restore=LANE_RESTORE):
    prefix = piecewise_program(LANE_GROUP, 64, 0)[:-2]
    return (*prefix, LANE_COMPARE, LANE_SAVE, *narrow, ('s_cbranch_execz 2', '020088bf'),
            waterfall.STORE, restore, LANE_STORE, waterfall.END)


def lane_reference(threshold, mask):
    """Interpret the compare, SAVEEXEC, branch and restore for each 64-lane guest wave."""
    values = (threshold, *PIECE_VALUES[1:])
    packed = piecewise_reference(LANE_GROUP, 64, values, 0)
    output = [SENTINEL] * WORDS
    for start in range(0, math.prod(LANE_GROUP), 64):
        saved = (1 << 64) - 1
        vcc = sum(1 << lane for lane in range(64) if start + lane < threshold)
        exec_mask = saved & vcc & mask
        if exec_mask:  # s_cbranch_execz skips the first store on an empty wave.
            for lane in range(64):
                if exec_mask & (1 << lane):
                    output[start + lane] = packed[start + lane]
        exec_mask |= saved
        for lane in range(64):
            if exec_mask & (1 << lane):
                output[128 + start + lane] = packed[start + lane]
    return output


def broadcast_exec(hlsl):
    # Negative control: turn the first EXEC restriction into the legacy broadcast of bit 0 to all threads.
    mutant, count = re.subn(r'^(\s*exec = )[^\n]*tid\.x[^\n]*;$', r'\g<1>1u;', hlsl, count=1, flags=re.M)
    assert count == 1
    return mutant


def check_piecewise_reads(hlsl, inputs):
    for reg, (scalar, vector) in zip(inputs, ((23, 10), (24, 12), (25, 11))):
        assignment = re.search(r'\bs%d = (ul[0-9a-f]+_%d);' % (scalar, vector), hlsl)
        assert assignment, (scalar, vector, hlsl)
        assert re.search(r'^ +%s = %s;$' % (assignment[1], re.escape(reg)), hlsl, re.M), assignment[1]


def piecewise_controls():
    for label, group, width in piecewise_cases():
        for component, value in enumerate(PIECE_VALUES):
            rows = piecewise_program(group, width, component)
            pipe = translate(rows, group, width)
            assert pipe.retry_const_exec, label
            hlsl = pipe.hlsl_cs()
            check_piecewise_reads(hlsl, PIECE_INPUTS)
            assert 'scc = true;' in hlsl and '0xfffffffeu' in hlsl, label
            for reg in (10, 11, 12):
                assert f'if (exec & 1) v{reg} = 0x00000000u;' in hlsl, label
            result = piecewise_reference(group, width, PIECE_VALUES, component)
            assert result[:math.prod(group)] == [
                value | (value << 16 if i % width == 0 else 0) for i in range(math.prod(group))
            ], label

    group = (128, 1, 1)
    unknown = (('s_mov_b32 s21, 0', '800395be'),
               ('s_andn2_b64 exec, -1, s[20:21]', 'c114fe8a'))
    varying = (('v_mov_b32_e32 v10, v0', '0003147e'), *PIECE_ZEROS[1:])
    refused(UNKNOWN_EXEC, lambda: translate(piecewise_program(group, 64, 0, switch=unknown), group, 64))
    for change in (dict(writes=PIECE_WRITES[1:]), dict(restore=(ZERO64,)), dict(zeros=varying)):
        refused(REFUSAL, lambda: translate(piecewise_program(group, 64, 0, **change), group, 64))
    hlsl = translate(piecewise_program(group, 64, 0, restore=(TWO64,)), group, 64).hlsl_cs()
    check_piecewise_reads(hlsl, ('0x00000000u',) * 3)
    copy = ('v_mov_b32_e32 v10, v10', '0a03147e')
    hlsl = translate(piecewise_program(group, 64, 0, writes=(*PIECE_WRITES, copy)), group, 64).hlsl_cs()
    assert re.search(r'\bul[0-9a-f]+_10 = ul[0-9a-f]+_10;', hlsl)
    refused(REFUSAL, lambda: translate(piecewise_program(group, 64, 0, restore=(ALL64, copy)), group, 64))

    partial = (65, 1, 1)
    translate(piecewise_program(partial, 64, 0), partial, 64)
    refused(REFUSAL, lambda: translate(piecewise_program(partial, 64, 0, restore=(TWO64,)), partial, 64))

    dynamic = (LANE_COMPARE, LANE_SAVE)
    refused(REFUSAL, lambda: translate(piecewise_program(group, 64, 0, restore=dynamic), group, 64))
    rows = piecewise_program(group, 64, 0, restore=(*dynamic, ALL64))
    check_piecewise_reads(translate(rows, group, 64).hlsl_cs(), PIECE_INPUTS)
    rows = piecewise_program(group, 64, 0, restore=(*dynamic, varying[0], ALL64))
    refused(REFUSAL, lambda: translate(rows, group, 64))


def mask_logic_controls():
    for width, group, exec_reg, saved, restore, invert in (
        (64, (128, 1, 1), 'exec', 's[26:27]', RESTORE64, INVERT64),
        (32, (64, 1, 1), 'exec_lo', 's26', waterfall.RESTORE, INVERT32),
    ):
        # Identity operations on the complemented mask, followed by two complements.
        delta = int(width == 64)
        logic = tuple(
            (f's_{op}_b{width} {exec_reg}, {exec_reg}, {src}',
             bytes((0x7e, 0xc1 if src == -1 else 0x80, 0x7e | (delta << 7), opcode)).hex())
            for op, src, opcode in (('and', -1, 0x87), ('or', 0, 0x88), ('xor', 0, 0x89), ('orn2', -1, 0x8b))
        )
        negate = (f's_not_b{width} {exec_reg}, {exec_reg}', bytes((0x7e, 7 + delta, 0xfe, 0xbe)).hex())
        rows = piecewise_program(group, width, 0, switch=(invert, *logic, negate, negate))
        check_piecewise_reads(translate(rows, group, width).hlsl_cs(), PIECE_INPUTS)
        for i, (op, src) in enumerate((('and', 0), ('or', 0), ('xor', 1), ('andn2', -1), ('orn2', 0))):
            opcode = (0x24 if width == 64 else 0x3c) + i
            save = (f's_{op}_saveexec_b{width} {saved}, {src}',
                    bytes((0xc1 if src == -1 else 0x80 + src, opcode, 0x9a, 0xbe)).hex())
            rows = piecewise_program(group, width, 0, switch=(save, restore, invert))
            hlsl = translate(rows, group, width).hlsl_cs()
            check_piecewise_reads(hlsl, PIECE_INPUTS)
            assert 's26 = 0x00000001u;' in hlsl
            if width == 64:
                assert 's27 = 0x00000000u;' in hlsl
            assert re.findall(r'^    scc = (true|false);$', hlsl, re.M) == [
                'false' if op in ('and', 'xor') else 'true', 'true'
            ], op


def lane_cases():
    yield 'lane-execz', lane_program(), (1 << 64) - 1
    yield 'lane-execz-projected', lane_program(narrow=MIXED_MASK, restore=RESTORE64), 1 << 32


def lane_controls():
    for label, rows, mask in lane_cases():
        refusals = []
        pipe = translate(rows, LANE_GROUP, 64, refusals)
        assert pipe.retry_const_exec and not refusals, (label, refusals)
        hlsl = pipe.hlsl_cs()
        check_piecewise_reads(hlsl, PIECE_INPUTS)
        assert 's26 = exec;' in hlsl and 'exec = (vcc & exec) & 1; scc = exec != 0;' in hlsl, label
        assert hlsl.count('if (exec & 1) u0.Store(a, v7);') == 2, label
        assert '+ 512' in hlsl, label
        assert ('exec = (s26 & 1u);' if mask == 1 << 32 else 'exec = (exec | s26) & 1;') in hlsl, label
        if mask == 1 << 32:
            assert re.search(r'exec = \(vcc & .*< 32u \? 0x00000000u : 0x00000001u\)', hlsl), label
        for threshold in LANE_THRESHOLDS:
            expected = lane_reference(threshold, mask)
            assert [i for i, value in enumerate(expected[:128]) if value != SENTINEL] == [
                i for i in range(threshold) if mask & (1 << (i % 64))
            ], (label, threshold)
            values = (threshold, *PIECE_VALUES[1:])
            assert expected[128:256] == piecewise_reference(LANE_GROUP, 64, values, 0)[:128]

    high = ('s_mov_b32 exec_hi, 0', '8003ffbe')
    group32 = (64, 1, 1)
    for op, reason in ((high, WAVE32_EXEC_HI), (LANE_SAVE, WAVE32_SAVE)):
        rows = program(group32, 32, after_restore=(op,))
        refused(reason, lambda: translate(rows, group32, 32))
    rows = program(LANE_GROUP, 64, after_restore=(LANE_COMPARE, LANE_SAVE, ALL32))
    refused(MISSING_OLD_EXEC, lambda: translate(rows, LANE_GROUP, 64))


def controls():
    lane_controls()
    for label, group, width, mask, saved, rows_mask in programs():
        rows = program(group, width, saved=saved, mask=rows_mask)
        with patch.object(waterfall.G.ComputePipeline, 'retry_const_exec_for', return_value=False):
            refused(REFUSAL, lambda: translate(rows, group, width))
        refusals = []
        pipe = translate(rows, group, width, refusals)
        assert pipe.retry_const_exec and not refusals, (label, refusals)
        hlsl = pipe.hlsl_cs()
        assert hlsl == translate(rows, group, None).hlsl_cs(), label
        assert re.search(r'\bs22 = ul[0-9a-f]+_9;', hlsl), label
        assert 'if (exec & 1) v8 = 0x00000039u;' in hlsl, label
        assert hlsl != broadcast_exec(hlsl)
        source = [29 + 53 * i for i in range(WORDS)]
        expected = reference(group, width, mask, 3, MASK, source)
        mutant = reference(group, width, mask, 3, MASK, source, broadcast=True)
        assert next(i for i, (a, b) in enumerate(zip(expected, mutant)) if a != b) == (1 if mask == 1 else 0)
    group = (128, 1, 1)
    refused(REFUSAL, lambda: translate(program(group, 64, index=VARYING_INDEX), group, 64))
    refused(REFUSAL, lambda: translate(program(group, 64, mask=(ZERO64,)), group, 64))
    refused(REFUSAL, lambda: translate(program(group, 64, before_read=(VARYING_INDEX,)), group, 64))
    assert translate(program(group, 64, before_read=(ONE64,)), group, 64).retry_const_exec
    refused(REFUSAL, lambda: translate(program(group, 64, before_read=(TWO64,)), group, 64))
    high_clobber = (('s_mov_b32 s27, 0', '80039bbe'),)
    refused('compute constant EXEC: s_mov_b64 writes EXEC from an unknown whole-word value',
            lambda: translate(program(group, 64, saved=True, before_read=high_clobber), group, 64))
    join = (('s_branch 0', '000082bf'),)
    refused(REFUSAL, lambda: translate(program(group, 64, before_read=join), group, 64))
    partial = (65, 1, 1)
    refused(REFUSAL, lambda: translate(program(partial, 64, mask=HIGH64), partial, 64))
    mixed = program(group, 64, after_restore=(ALL32,))
    refused(waterfall.G.CONST_EXEC_WAVE_SIZE, lambda: translate(mixed, group, None))
    piecewise_controls()
    mask_logic_controls()
    print('constant EXEC: multi-wave projections, piecewise recomputation, lane flow and refusal controls passed')


def dispatch(directory, dxc, runner, adapter):
    source = [29 + 53 * i for i in range(WORDS)]
    for label, group, width, mask, saved, rows_mask in programs():
        pipe = translate(program(group, width, saved=saved, mask=rows_mask), group, width)
        hlsl = pipe.hlsl_cs()
        cases, mutations = [], []
        for key, marker_bits in ((3, 0), (0, MASK)):
            expected = reference(group, width, mask, key, marker_bits, source)
            case = directory / f'{label}-{key}.case'
            values = {('rootconst', 0, 0): key, ('rootconst', 0, 1): marker_bits,
                      ('stride', 0): 4, ('descword3', 0): (22 << 12) | 4}  # FLOAT32, select X
            constants = [values[ud] for ud in pipe.uds]
            words = [1, *expected, *constants, *([0] * (16 - len(constants))), *source]
            case.write_bytes(struct.pack('<%dI' % len(words), *words))
            cases.append((case, f'key {key}', None))
            if marker_bits:
                actual = reference(group, width, mask, key, marker_bits, source, broadcast=True)
                at = next(i for i, (a, b) in enumerate(zip(expected, actual)) if a != b)
                mutations.append((case, 'broadcast EXEC', f'word={at} expected={expected[at]:08x} actual={actual[at]:08x}'))
        for name, variant, checks in ((label, hlsl, cases), (f'{label}-mutant', broadcast_exec(hlsl), mutations)):
            src, shader = directory / f'{name}.hlsl', directory / f'{name}.cso'
            src.write_text(variant, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
            if check_references(name, runner, adapter, shader, checks, allow_skip=True) == 77:
                return 77
    result = dispatch_piecewise(directory, dxc, runner, adapter, source)
    return result or dispatch_lanes(directory, dxc, runner, adapter, source)


def dispatch_piecewise(directory, dxc, runner, adapter, source):
    values = {('rootconst', 0, 0): PIECE_VALUES[0], ('rootconst', 0, 1): PIECE_VALUES[2]}
    assert source[0] == PIECE_VALUES[1]
    for label, group, width in piecewise_cases():
        for component, (scalar, vector) in enumerate(((23, 10), (24, 12), (25, 11))):
            pipe = translate(piecewise_program(group, width, component), group, width)
            expected = piecewise_reference(group, width, PIECE_VALUES, component)
            name = f'{label}-s{scalar}'
            case, src, shader = (directory / (name + ext) for ext in ('.case', '.hlsl', '.cso'))
            constants = [values[ud] for ud in pipe.uds]
            words = [1, *expected, *constants, *([0] * (16 - len(constants))), *source]
            case.write_bytes(struct.pack('<%dI' % len(words), *words))
            src.write_text(pipe.hlsl_cs(), encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
            if check_references(name, runner, adapter, shader, [(case, f's{scalar} and v{vector}', None)],
                                allow_skip=True) == 77:
                return 77
    return 0


def dispatch_lanes(directory, dxc, runner, adapter, source):
    for name, rows, mask in lane_cases():
        pipe = translate(rows, LANE_GROUP, 64)
        src, shader = directory / f'{name}.hlsl', directory / f'{name}.cso'
        src.write_text(pipe.hlsl_cs(), encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
        cases = []
        for threshold in LANE_THRESHOLDS:
            expected = lane_reference(threshold, mask)
            values = {('rootconst', 0, 0): threshold, ('rootconst', 0, 1): PIECE_VALUES[2]}
            constants = [values[ud] for ud in pipe.uds]
            words = [1, *expected, *constants, *([0] * (16 - len(constants))), *source]
            case = directory / f'{name}-{threshold}.case'
            case.write_bytes(struct.pack('<%dI' % len(words), *words))
            cases.append((case, f'threshold {threshold}', None))
        if check_references(name, runner, adapter, shader, cases, allow_skip=True) == 77:
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
