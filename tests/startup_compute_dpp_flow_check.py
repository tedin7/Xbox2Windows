"""Synthetic gfx1013 compute DPP readback after a lane-divergent EXEC region."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_compute_tg_size_check import (
    BUFFER_STORE_X4_V8_V4_S16, EXEC_LO, INLINE_0, INLINE_4, S_ENDPGM, S_LOAD_X8_S12_S10,
    S_MOV_B32, S_WAITCNT_LGKM0, V_LSHLREV_B32, V_MOV_B32, le32, sop1, sop2, sopc, sopp, vop1, vop2,
)
from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, refused, root_blob

LANES = 32
MIN, MAX, ADD = 0x0f, 0x10, 0x03
QUAD = (2, 0, 3, 1)


def dpp(op, dst, src0, src1, control, *, bound=False):
    return vop2(op, dst, 250, src1) + le32(0xff000000 | (int(bound) << 19) | control << 8 | src0)


SETUP = b''.join([
    sop1(S_MOV_B32, 10, 6), sop1(S_MOV_B32, 11, INLINE_0),
    S_LOAD_X8_S12_S10, S_WAITCNT_LGKM0,
    vop2(V_LSHLREV_B32, 4, INLINE_4, 0),
    vop1(0x06, 5, 256),  # float(tid.x)
])
SAVE = sop1(0x3c, 20, 106)  # s_and_saveexec_b32 s20, vcc_lo
RESTORE = sop1(S_MOV_B32, EXEC_LO, 20)
RESTORE_OR = sop2(0x10, EXEC_LO, EXEC_LO, 20)
COMPARE = le32(0x7c000000 | 0xc1 << 17 | (128 + 4))  # vcc_lo = 4 < tid.x
REPLACE = vop2(ADD, 5, 255, 5, 0xc1080000)  # subtract 8.5
BRANCH = sopp(0x08, len(REPLACE) // 4)
REGION = COMPARE + SAVE + BRANCH + REPLACE + RESTORE
SCAN = dpp(MIN, 8, 5, 5, 0xb1)  # swap neighboring pairs before the prefix scan
SCAN += b''.join(dpp(MIN, 8, 8, 8, 0x110 + k) for k in (1, 2, 4, 8))
QUAD_CTRL = sum(lane << (2 * i) for i, lane in enumerate(QUAD))
EXCHANGES = SCAN + b''.join([
    dpp(MAX, 9, 5, 5, 0x153),  # row_share:3
    dpp(ADD, 10, 5, 5, QUAD_CTRL),
    dpp(ADD, 11, 5, 5, 0x113, bound=True),
])
TAIL = BUFFER_STORE_X4_V8_V4_S16 + S_ENDPGM
PROGRAM = SETUP + REGION + EXCHANGES + TAIL
PROGRAM_OR = SETUP + REGION[:-len(RESTORE)] + RESTORE_OR + EXCHANGES + TAIL
PROGRAM_BRANCH_ONLY = SETUP + BRANCH + REPLACE + EXCHANGES + TAIL


def translate(code=PROGRAM, lanes=LANES):
    return G.ComputePipeline(code, root_blob(), USER7, (lanes, 1, 1), wave_size=32).hlsl_cs()


def float_word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def expected(mutant=None, lanes=LANES, threshold=4):
    values = [float(lane) - (8.5 if lane > threshold else 0.0) for lane in range(lanes)]
    scan = values if mutant == 'swap' else [min(value, values[lane ^ 1]) for lane, value in enumerate(values)]
    words = []
    for lane, local in enumerate(values):
        row, quad = lane & ~15, lane & ~3
        scan_start = max(0, lane - 15) if mutant == 'shift' else row
        share_lane = 4 if mutant == 'share' else 3
        quad_lane = lane % 4 if mutant == 'quad' else QUAD[lane % 4]
        shift_lane = lane - 3 if lane - row >= 3 else None
        other = values[shift_lane] if shift_lane is not None else (local if mutant == 'bound' else 0.0)
        words.extend(map(float_word, (
            min(scan[scan_start:lane + 1]), max(values[row + share_lane], local),
            values[quad + quad_lane] + local, other + local,
        )))
    if mutant == 'restore':
        words[:20] = [SENTINEL] * 20  # lanes 0 through 4 stay disabled and never store
    return words + [SENTINEL] * (WORDS - len(words))


MUTANTS = {
    'swap': (' ^ 1u]', ' ^ 0u]'),
    'shift': ('& 15u) >= ', '& 31u) >= '),
    'share': ('& ~15u) | 3u]', '& ~15u) | 4u]'),
    'quad': (f'{QUAD_CTRL}u >> (2u *', '228u >> (2u *'),  # identity quad mapping
    'bound': (': 0u)) + F(v5)', ': v5)) + F(v5)'),
}


def mutate(source, name):
    old, new = MUTANTS[name]
    if name == 'shift':
        # Change the addressing and bounds of the four min steps only.
        lines = [
            line.replace(old, new).replace('& ~15u', '& ~31u').replace(') & 15u)]', ') & 31u)]')
            if 'v8 =' in line else line for line in source.splitlines()
        ]
        mutated = '\n'.join(lines) + '\n'
    else:
        assert source.count(old) == 1, (name, source)
        mutated = source.replace(old, new)
    assert mutated != source, name
    return mutated


def controls():
    source = translate()
    assert source.count('GroupMemoryBarrierWithGroupSync();') == 16
    for line in source.splitlines():
        if 'GroupMemoryBarrierWithGroupSync();' in line:
            assert line == '        GroupMemoryBarrierWithGroupSync();'
    assert 'if (pc == 0)' in source and 'if (pc == 1)' in source and 'if (pc == 2)' not in source
    for name in MUTANTS:
        mutate(source, name)
        assert expected(name) != expected(), name

    reason = 'v_min_f32_dpp requires a straight-line compute program with entry EXEC'
    # DPP reached with EXEC narrowed on both edges.
    refused(reason, lambda: translate(SETUP + COMPARE + SAVE + BRANCH + REPLACE + EXCHANGES + TAIL))
    # EXEC restored on the fall-through edge only.
    compare_scc = sopc(0x06, 128, 128)
    skips_restore = compare_scc + sopp(0x04, len(RESTORE) // 4)
    refused(reason, lambda: translate(SETUP + REGION[:-len(RESTORE)] + skips_restore + RESTORE + EXCHANGES + TAIL))
    # DPP is reached with entry EXEC, but another path terminates first.
    exits_early = compare_scc + sopp(0x04, 1) + S_ENDPGM
    refused(reason, lambda: translate(SETUP + REGION + exits_early + EXCHANGES + TAIL))
    clobber = sop1(S_MOV_B32, 20, INLINE_0)
    refused(reason, lambda: translate(SETUP + REGION[:-len(RESTORE)] + clobber + RESTORE + EXCHANGES + TAIL))
    print('DPP flow: 8 exchanges converge after EXEC restore; narrowed EXEC and bypassing paths refused')
    partial_wave = translate(PROGRAM_OR, lanes=16)
    branch_only = translate(PROGRAM_BRANCH_ONLY)
    for hlsl in (partial_wave, branch_only):
        assert hlsl.count('GroupMemoryBarrierWithGroupSync();') == 16
    return source, partial_wave, branch_only


def draw(sources, dxc, renderer, adapter):
    source, partial_wave, branch_only = sources
    with tempfile.TemporaryDirectory() as tmp:
        src, cso, case = (Path(tmp) / name for name in ('dpp.hlsl', 'dpp.cso', 'dpp.case'))
        cases = [(None, source, LANES, 4), (None, partial_wave, 16, 4), (None, branch_only, LANES, -1)]
        cases.extend((name, mutate(source, name), LANES, 4) for name in MUTANTS)
        for hlsl, restore, lanes in ((source, '(s20)', LANES), (partial_wave, '(exec | s20)', 16)):
            assignment = f'exec = {restore} & 1;'
            assert hlsl.count(assignment) == 1
            cases.append(('restore', hlsl.replace(assignment, 'exec &= 1;'), lanes, 4))
        cases.append(('quad', mutate(branch_only, 'quad'), LANES, -1))
        for index, (name, hlsl, lanes, threshold) in enumerate(cases):
            words = expected(lanes=lanes, threshold=threshold)
            case.write_bytes(struct.pack(f'<{1 + WORDS}I', 1, *words))
            src.write_text(hlsl, encoding='ascii')
            subprocess.run([dxc, '-Gis', '-WX', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
            run = subprocess.run([renderer, adapter, '--compute-store-case', str(cso), str(case)],
                                 capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77 and index == 0:
                return 77
            if name is None:
                assert run.returncode == 0, run.stderr
                continue
            bad_words = expected(name, lanes, threshold)
            i = next(i for i, pair in enumerate(zip(words, bad_words)) if pair[0] != pair[1])
            mismatch = f'word={i} expected={words[i]:08x} actual={bad_words[i]:08x}'
            assert run.returncode == 1 and mismatch in run.stderr, (name, mismatch, run.stderr)
            print(f'control {name}: rejected at {mismatch}')
    return 0


if __name__ == '__main__':
    hlsl = controls()
    if len(sys.argv) > 1:
        sys.exit(draw(hlsl, *sys.argv[1:]))
