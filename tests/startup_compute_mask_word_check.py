"""Exercise whole-word lane-mask SOP2 lowering through compute UAV readback."""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import startup_compute_waterfall as W
from compute_load_format_check import check_references
from startup_gfx2hlsl_sop2_complement_check import sop2

G = W.G
OPS = ('s_nor_b32', *G.SHIFT_ADD, 's_mul_i32')
PREFIX = bytes.fromhex(''.join(word for _, word in W.PREFIX[:7]))
TAIL = bytes.fromhex(W.STORE[1] + W.END[1])
COMPARE = struct.pack('<I', 0x7c000000 | (0xc4 << 17) | (5 << 9) | 4)
SCC_SEED = bytes.fromhex('048706bf')  # s_cmp_eq_u32 s4, 7
SCC_USE = sop2('s_cselect_b32', 22, 129, 128)
SGPR = 21
VCC_LO = 106
# v_cndmask_b32 v7, 0, 1, s21; v_mov_b32 v7, s21; v_xor_b32 v7, s22, v7.
MASK_USE = struct.pack('<II', 0xd5010007, 128 | (129 << 9) | (SGPR << 18))
WORD_USE = bytes.fromhex('15020e7e')
XOR_SCC = bytes.fromhex('160e0e3a')
CASES = (
    (7, 0, (1, 9, 3, 9)),
    (7, 0xfffffff0, (9, 1, 2, 3)),
    (4, 0xffffffff, (0, 1, 2, 3)),
    (0, 11, (0, 1, 2, 3)),
    (7, 0x80000001, (9, 1, 9, 2)),
)


def program(op, *, reverse=False):
    left, right = (5, VCC_LO) if reverse else (VCC_LO, 5)
    return (PREFIX + COMPARE + SCC_SEED + sop2(op, SGPR, left, right) + SCC_USE
            + (MASK_USE if op == 's_nor_b32' else WORD_USE) + XOR_SCC + TAIL)


def translate(code, group=(4, 1, 1)):
    return G.ComputePipeline(code, W.root_blob(), W.USER7, group).hlsl_cs()


def reference(op, inputs, *, reverse=False, mutant=False):
    threshold, rhs, keys = inputs
    ballot = sum(1 << i for i, key in enumerate(keys) if key < threshold)
    if mutant:
        ballot &= 1  # silently dropping every lane except logical lane zero
    left, right = (rhs, ballot) if reverse else (ballot, rhs)
    if op == 's_nor_b32':
        word = ~(left | right) & W.MASK
        return [((word >> lane) & 1) ^ int(word != 0) for lane in range(4)]
    if op in G.SHIFT_ADD:
        total = (left << G.SHIFT_ADD[op]) + right
        scc = int(total > W.MASK)
    else:
        total, scc = left * right, int(threshold == 7)
    return [(total & W.MASK) ^ scc] * 4


def variants():
    for op in OPS:
        yield op, False
    yield 's_lshl4_add_u32', True
    yield 's_mul_i32', True


def controls():
    for op, reverse in variants():
        code = program(op, reverse=reverse)
        text = translate(code)
        assert text.count(' = x2w_ballot(vcc, tid.x);') == 1, text
        assert 'WaveActiveBallot' in text, text
        assert any(reference(op, case, reverse=reverse) != reference(op, case, reverse=reverse, mutant=True)
                   for case in CASES), (op, reverse)
        reason = ('s_mul_i32 reads lane mask vcc_lo as numeric data' if op == 's_mul_i32'
                  else f'{op} on a lane mask')
        W.refused(reason, lambda: translate(code, (5, 1, 1)))
        with patch.object(G.Stage, 'lower_mask_word_sop2', return_value=False):
            W.refused(reason, lambda: translate(code))
        # An intervening mask copy with mixed numeric data lacks a whole-word proof.
        clobber = sop2('s_or_b32', VCC_LO, VCC_LO, 130)
        unsafe = code.replace(COMPARE, COMPARE + clobber, 1)
        W.refused(reason, lambda: translate(unsafe))
    print('8 synthetic variants passed translation and refusal controls')


def dispatch(directory, dxc, runner, adapter):
    for op, reverse in variants():
        label = op + ('-reverse' if reverse else '')
        text = translate(program(op, reverse=reverse))
        cases, mutations = [], []
        for index, inputs in enumerate(CASES):
            expected = reference(op, inputs, reverse=reverse)
            wrong = reference(op, inputs, reverse=reverse, mutant=True)
            threshold, rhs, keys = inputs
            case = directory / f'{label}-{index}.case'
            words = [1, *expected, *([W.SENTINEL] * (W.WORDS - 4)), threshold, rhs, *([0] * 14),
                     *keys, *([0] * (W.WORDS - 4))]
            case.write_bytes(struct.pack('<%dI' % len(words), *words))
            cases.append((case, str(inputs), None))
            mismatch = next((i for i, pair in enumerate(zip(expected, wrong)) if pair[0] != pair[1]), None)
            if mismatch is not None:
                mutations.append((case, str(inputs),
                                  f'word={mismatch} expected={expected[mismatch]:08x} actual={wrong[mismatch]:08x}'))
        mutant = text.replace('bit != 0u && lane == i', 'bit != 0u && lane == i && i == 0u')
        assert mutant != text and mutations
        for name, hlsl, checks in ((label, text, cases), (label + '-mutant', mutant, mutations)):
            src, shader = directory / f'{name}.hlsl', directory / f'{name}.cso'
            src.write_text(hlsl, encoding='ascii')
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
        with tempfile.TemporaryDirectory(dir=output) as directory:
            return dispatch(Path(directory), *args.dispatch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
