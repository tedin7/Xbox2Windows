"""Synthetic off-store and live scalar join programs, with WARP readback and mutants."""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, assemble, refused, root_blob

# s4 is a bitmask of active guest lanes, s5 is the scalar byte offset or shift input.
PREFIX = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v4, 2, v0
v_mov_b32_e32 v5, s4
v_lshrrev_b32_e32 v5, v0, v5
v_and_b32_e32 v5, 1, v5
s_mov_b32 s26, exec_lo
"""
NARROW = 'v_cmpx_ne_u32_e32 0, v5\n'
RESTORE = 's_mov_b32 exec_lo, s26\n'
END = 's_endpgm\n'
JOIN_TAIL = RESTORE + """\
v_mov_b32_e32 v7, s20
buffer_store_dword v7, v4, s[16:19], 0 offen
""" + END
PROGRAMS = {
    'store': PREFIX + NARROW + """\
buffer_store_dword v0, off, s[16:19], s5
""" + END,
    'load': PREFIX + 's_mov_b32 s20, 13\n' + NARROW + """\
s_cbranch_execz 3
s_buffer_load_dword s20, s[12:15], s5
s_waitcnt lgkmcnt(0)
""" + JOIN_TAIL,
    'shift': PREFIX + 's_mov_b32 s20, 13\n' + NARROW + """\
s_cbranch_execz 1
s_lshl_b32 s20, s5, 2
""" + JOIN_TAIL,
}
MUTANTS = {
    'store': ('WaveActiveMax(x2w_store_active ? x2w_guest_lane : 0u)',
              'WaveActiveMin(x2w_store_active ? x2w_guest_lane : ~0u)'),
    'load': ('!WaveActiveAnyTrue((exec & 1u) != 0u)', '(exec & 1u) == 0u'),
    'shift': ('!WaveActiveAnyTrue((exec & 1u) != 0u)', '(exec & 1u) == 0u'),
}

PREFIX_WORDS = '06038abe 80038bbe 05030cf4 000000fa 7fc08cbf 82000834 04020a7e 000b0a2c 810a0a36 7e039abe '
JOIN_WORDS = '1a03febe 14020e7e 001070e0 04070480 000081bf'
CODES = {
    'store': bytes.fromhex(PREFIX_WORDS + '800aaa7d 000070e0 00000405 000081bf'),
    'load': bytes.fromhex(PREFIX_WORDS + '8d0394be 800aaa7d 030088bf 060520f4 0000000a 7fc08cbf ' + JOIN_WORDS),
    'shift': bytes.fromhex(PREFIX_WORDS + '8d0394be 800aaa7d 010088bf 0582148f ' + JOIN_WORDS),
}


def translate(code, group=(4, 1, 1), wave_size=None, refusals=None):
    return G.ComputePipeline(code, root_blob(), USER7, group, wave_size=wave_size, refusals=refusals).hlsl_cs()


def controls(llvm_mc):
    for label, code in CODES.items():
        if llvm_mc:
            assert assemble(llvm_mc, PROGRAMS[label]) == code, label
        hlsl = translate(code)
        assert hlsl.count(MUTANTS[label][0]) == 1, hlsl
        census = []
        assert translate(code, refusals=census) == hlsl and not census, census
        wide = translate(code, (8, 1, 1), wave_size=32)
        assert '[WaveSize(32)]\n[numthreads(8, 1, 1)]' in wide, wide
    refused('buffer store off with data not proven wave-uniform',
            lambda: translate(CODES['store'], (65, 1, 1), wave_size=64))
    # A numeric shift of a per-lane EXEC bit cannot supply one uniform scalar at the join.
    varying = CODES['shift'].replace(bytes.fromhex('0582148f'), bytes.fromhex('7e82148f'))
    refused('s_lshl_b32 inside a block s_cbranch_execz skips per lane: definition of s20',
            lambda: translate(varying))


def make_cases(directory, label):
    source = [0x4000 + i * 37 for i in range(WORDS)]
    normal, mutants = [], []
    # Sparse lanes 1 and 3, an empty wave, one lane, then all lanes.
    for mask, arg in ((0b1010, 12), (0, 8), (0b0100, 4), (0b1111, 16)):
        expected = [SENTINEL] * WORDS
        if label == 'store':
            word = arg // 4
            if mask:
                expected[word] = mask.bit_length() - 1
            wrong = (mask & -mask).bit_length() - 1
        else:
            expected[:4] = [(source[arg // 4] if label == 'load' else arg << 2) if mask else 13] * 4
            word, wrong = 0, 13
        words = [1, *expected, mask, arg, *([0] * 14), *source]
        case = directory / f'{label}-{mask}.case'
        case.write_bytes(struct.pack('<%dI' % len(words), *words))
        normal.append((case, f'EXEC {mask:04b}, input {arg}', None))
        if mask == 0b1010:
            assert expected[word] != wrong
            mutants.append((case, 'mutant, sparse EXEC',
                            f'word={word} expected={expected[word]:08x} actual={wrong:08x}'))
    return normal, mutants


def dispatch(directory, dxc, runner, adapter):
    for label, code in CODES.items():
        hlsl = translate(code)
        normal, mutants = make_cases(directory, label)
        old, new = MUTANTS[label]
        for name, shader, cases in ((label, hlsl, normal),
                                    (label + '-mutant', hlsl.replace(old, new), mutants)):
            src, cso = directory / (name + '.hlsl'), directory / (name + '.cso')
            src.write_text(shader, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', cso, src], check=True)
            if check_references(name, runner, adapter, cso, cases, allow_skip=True) == 77:
                return 77
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args()
    controls(args.llvm_mc)
    print('off store, live scalar load and live shift translation controls passed')
    if args.dispatch:
        output = Path(__file__).resolve().parent.parent / 'out'
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            return dispatch(Path(tmp), *args.dispatch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
