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
HIGH64 = (('s_mov_b32 s30, 0', '80039ebe'), ('s_mov_b32 s31, 1', '81039fbe'),
          ('s_mov_b64 exec, s[30:31]', '1e04febe'))
INDEX = ('v_mov_b32_e32 v9, s20', '1402127e')
VARYING_INDEX = ('v_mov_b32_e32 v9, v0', '0003127e')
SMEM = ('s_buffer_load_dword s20, s[12:15], 0', '060520f4000000fa')
OLD_MARKER, NEW_MARKER = 13, 57


def program(group, width, *, saved=False, mask=None, index=INDEX, before_read=(), after_restore=()):
    if group[1] == 1:
        flat = (('v_mov_b32_e32 v3, v0', '0003067e'),)
    else:  # x-major group layout, with X = 16
        flat = (('v_lshlrev_b32_e32 v3, 4, v1', '84020634'),
                ('v_add_nc_u32_e32 v3, v0, v3', '0007064a'))
    width_mask = ('v_and_b32_e32 v7, 63, v3', 'bf060e36') if width == 64 else (
        'v_and_b32_e32 v7, 31, v3', '9f060e36')
    save = ((SAVE64 if width == 64 else waterfall.PREFIX[-2]),) if saved else ()
    if width == 64:
        restore = RESTORE64 if saved else ALL64
    else:
        restore = waterfall.RESTORE if saved else ALL32
    if mask is None:
        mask = (ONE64 if width == 64 else ONE32,)
    return (*waterfall.PREFIX[:4], *flat, width_mask, ('v_mov_b32_e32 v8, 13', '8d02107e'),
            *save, *mask, SMEM, waterfall.PREFIX[3], ('s_add_u32 s20, s20, s4', '14041480'),
            ('s_and_b32 s20, s20, 3', '14831487'), ('s_lshl_b32 s20, s20, 1', '1481148f'),
            index, FORMAT, VM_WAIT, *before_read, READ, ('v_mov_b32_e32 v8, 57', 'b902107e'),
            restore, *after_restore, ('v_add_nc_u32_e32 v7, s22, v7', '160e0e4a'),
            ('v_and_b32_e32 v8, s5, v8', '05101036'), ('v_xor_b32_e32 v7, v8, v7', '080f0e3a'),
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


def broadcast_exec(hlsl):
    # Negative control: turn the first EXEC restriction into the legacy broadcast of bit 0 to all threads.
    mutant, count = re.subn(r'^(\s*exec = )[^\n]*tid\.x[^\n]*;$', r'\g<1>1u;', hlsl, count=1, flags=re.M)
    assert count == 1
    return mutant


def controls():
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
    refused(REFUSAL, lambda: translate(program(group, 64, before_read=(ONE64,)), group, 64))
    high_clobber = (('s_mov_b32 s27, 0', '80039bbe'),)
    refused(REFUSAL, lambda: translate(program(group, 64, saved=True, before_read=high_clobber), group, 64))
    join = (('s_branch 0', '000082bf'),)
    refused(REFUSAL, lambda: translate(program(group, 64, before_read=join), group, 64))
    partial = (65, 1, 1)
    refused(REFUSAL, lambda: translate(program(partial, 64, mask=HIGH64), partial, 64))
    mixed = program(group, 64, after_restore=(ALL32,))
    refused(waterfall.G.CONST_EXEC_WAVE_SIZE, lambda: translate(mixed, group, None))
    print('constant EXEC: multi-wave projections, recomputation and refusal controls passed')


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
