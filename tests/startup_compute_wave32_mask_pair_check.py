"""A wave32 compare into an even SGPR leaves its neighbour a plain data register.

The translator tags the odd neighbour of an even compare destination as the high word of a wave64 mask pair, so a later
numeric read of that neighbour is refused ("s_and_b32 reads lane mask s21 as numeric data"). In a program that is wave32
(the caller says so, or every exec write is a wave32 form) the compare writes one word only: s21 keeps the value the
program gave it, and the numeric read is exact. A wave64 or unproven program keeps the refusal.

program: s21 = s5 + 1 (data); v_cmp_gt_u32 s20, s4, v5 (the compare into an even SGPR); s21 &= 63 (numeric read);
v7 = lane's compare bit (v_cndmask on s20); v7 += s21; store v7 per lane.

--dispatch DXC RUNNER ADAPTER also reads the result back from D3D12 and compares it with the CPU reference; a mutant
that makes the compare bit depend on the first lane only is rejected by the same readback.
"""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_compute_waterfall as W
from compute_load_format_check import check_references
from startup_gfx2hlsl_sop2_complement_check import sop2

G = W.G
PREFIX = bytes.fromhex(''.join(word for _, word in W.PREFIX[:7]))
TAIL = bytes.fromhex(W.STORE[1] + W.END[1])
DATA = 21        # holds data the whole program
MASK_SGPR = 20   # the compare destination: even, so s21 would be its wave64 high word
# v_cmp_gt_u32_e64 s20, s4, v5
COMPARE = struct.pack('<II', 0xd4000000 | (0xc4 << 16) | MASK_SGPR, 4 | ((256 + 5) << 9))
# v_cndmask_b32 v7, 0, 1, s20
SELECT = struct.pack('<II', 0xd5010007, 128 | (129 << 9) | (MASK_SGPR << 18))
ADD_DATA = bytes.fromhex('150e0e4a')  # v_add_nc_u32 v7, s21, v7
CASES = (
    (3, 5, (1, 9, 2, 9)),
    (7, 62, (9, 1, 2, 3)),
    (4, 63, (0, 1, 2, 3)),
    (0, 200, (0, 1, 2, 3)),
    (9, 0xffffffff, (9, 1, 9, 2)),
)


def program():
    return (PREFIX + sop2('s_add_u32', DATA, 5, 129) + COMPARE + sop2('s_and_b32', DATA, DATA, 128 + 63)
            + SELECT + ADD_DATA + TAIL)


def translate(wave_size, group=(4, 1, 1)):
    return G.ComputePipeline(program(), W.root_blob(), W.USER7, group, wave_size=wave_size).hlsl_cs()


def reference(inputs, mutant=False):
    threshold, rhs, keys = inputs
    number = ((rhs + 1) & 0xffffffff) & 63
    bits = [int(threshold > key) for key in keys]
    if mutant:
        bits = [bits[0]] * len(bits)  # the first lane's compare bit for every lane
    return [(number + bit) & W.MASK for bit in bits]


def controls():
    text = translate(32)
    assert 's21 = s21 & 0x0000003fu;' in text and 's20 = ((ud0 > v5) ? 1u : 0u);' in text, text
    for wave_size in (None, 64):
        W.refused('s_and_b32 reads lane mask s21 as numeric data', lambda: translate(wave_size))
    # Other numeric reads of the mask register itself stay refused in a wave32 program.
    unsafe = PREFIX + COMPARE + sop2('s_add_u32', DATA, MASK_SGPR, 129) + TAIL
    W.refused('s_add_u32 reads lane mask s20 as numeric data',
              lambda: G.ComputePipeline(unsafe, W.root_blob(), W.USER7, (4, 1, 1), wave_size=32).hlsl_cs())
    assert any(reference(case) != reference(case, mutant=True) for case in CASES)
    print('wave32 mask pair: translated; wave64 and unproven programs and a read of the mask word itself stay refused')


def dispatch(directory, dxc, runner, adapter):
    text = translate(32)
    cases, mutations = [], []
    for index, inputs in enumerate(CASES):
        expected, wrong = reference(inputs), reference(inputs, mutant=True)
        threshold, rhs, keys = inputs
        case = directory / f'pair-{index}.case'
        words = [1, *expected, *([W.SENTINEL] * (W.WORDS - 4)), threshold, rhs, *([0] * 14), *keys,
                 *([0] * (W.WORDS - 4))]
        case.write_bytes(struct.pack('<%dI' % len(words), *words))
        cases.append((case, str(inputs), None))
        mismatch = next((i for i, pair in enumerate(zip(expected, wrong)) if pair[0] != pair[1]), None)
        if mismatch is not None:
            mutations.append((case, str(inputs), f'word={mismatch} expected={expected[mismatch]:08x} '
                                                 f'actual={wrong[mismatch]:08x}'))
    # The compare bit of every lane follows lane 0's key.
    mutant = text.replace('s20 = ((ud0 > v5) ? 1u : 0u);', 's20 = ((ud0 > r0.Load(0)) ? 1u : 0u);')
    assert mutant != text, text
    for name, hlsl, checks in (('pair', text, cases), ('pair-mutant', mutant, mutations)):
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
