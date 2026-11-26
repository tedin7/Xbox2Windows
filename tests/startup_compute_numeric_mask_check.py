"""Translate copied lane masks used by numeric ALU and compare WARP readback with the CPU."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import G, MASK, SENTINEL, USER7, WORDS, assemble, refused, root_blob

PREFIX = (
    ('s_mov_b32 s10, s6', '06038abe'),
    ('s_mov_b32 s11, 0', '80038bbe'),
    ('s_load_dwordx8 s[12:19], s[10:11], 0x0', '05030cf4000000fa'),
    ('s_waitcnt lgkmcnt(0)', '7fc08cbf'),
)
COMPARE = ('v_cmp_le_u32_e32 vcc_lo, s4, v0', '0400867d')
COPY = ('s_mov_b32 s20, vcc_lo', '6a0394be')
STORE = (
    ('v_lshlrev_b32_e32 v4, 2, v0', '82000834'),
    ('buffer_store_dword v5, v4, s[16:19], 0 offen', '001070e004050480'),
    ('s_endpgm', '000081bf'),
)
CASES = {
    's_mul_i32': (
        ('s_mul_i32 s20, s20, 3', '14831493'),
        ('v_mov_b32_e32 v5, s20', '14020a7e'),
    ),
    'v_sub_f32': (
        ('v_mov_b32_e32 v6, 1.0', 'f2020c7e'),
        ('v_sub_f32_e32 v5, s20, v6', '140c0a08'),
    ),
    'v_mul_f32': (
        ('v_mov_b32_e32 v6, 2.0', 'f4020c7e'),
        ('v_mul_f32_e32 v5, s20, v6', '140c0a10'),
    ),
}
EXEC_COPY = ('s_mov_b32 s20, exec_lo', '7e0394be')


def program(body, mask=(COMPARE, COPY)):
    return (*PREFIX, *mask, *body, *STORE)


def lower(rows, width=32, llvm_mc=None):
    code = bytes.fromhex(''.join(word for _, word in rows))
    if llvm_mc:
        source = '\n'.join(asm for asm, _ in rows) + '\n'
        assert assemble(llvm_mc, source) == code, source
    return G.ComputePipeline(code, root_blob(), USER7, (width, 1, 1)).hlsl_cs()


def controls(llvm_mc):
    for opcode, body in CASES.items():
        hlsl = lower(program(body), llvm_mc=llvm_mc)
        assert 'groupshared uint x2w_mask_bits[32];' in hlsl
        assert hlsl.count('GroupMemoryBarrierWithGroupSync();') == 2
        assert '= (s20 & 1u) <<' in hlsl
        assert ' |= x2w_mask_bits[lane];' not in lane_bit_mutant(hlsl)
        one = lower(program(body), width=1, llvm_mc=llvm_mc)
        assert 'x2w_mask' not in one
        refused(f'{opcode} reads lane mask s20 as numeric data', lambda: lower(program(body), width=64))
    assert 'x2w_mask_bits[32]' in lower(program(CASES['s_mul_i32'], (EXEC_COPY,)), llvm_mc=llvm_mc)

    body = CASES['s_mul_i32']
    for mask in (
        (COMPARE, ('s_mov_b32 vcc_lo, 2', '8203eabe'), COPY),
        (COMPARE, COPY, ('s_and_b32 s20, s20, 2', '14821487')),
        (COMPARE, COPY, ('s_mov_b32 exec_lo, vcc_lo', '6a03febe')),
        (COMPARE, COPY, ('s_branch 0', '000082bf')),
    ):
        refused('s_mul_i32 reads lane mask s20 as numeric data', lambda: lower(program(body, mask)))


def reference(opcode, mask):
    if opcode == 's_mul_i32':
        return mask * 3 & MASK
    value, = struct.unpack('<f', struct.pack('<I', mask))
    result = value - 1.0 if opcode == 'v_sub_f32' else value * 2.0
    return struct.unpack('<I', struct.pack('<f', result))[0]


def lane_bit_mutant(hlsl):
    # Keep the ALU and store, but feed them only this invocation's bit.
    reduction, = [line for line in hlsl.splitlines() if ' |= x2w_mask_bits[lane];' in line]
    word = reduction.split(' |= ', 1)[0].rsplit(maxsplit=1)[1]
    return hlsl.replace(reduction, f'    {word} = s20 & 1u;')


def dispatch(directory, dxc, runner, backend):
    for label, opcode, mask_ops in (
        *((opcode, opcode, (COMPARE, COPY)) for opcode in CASES),
        ('exec-copy', 's_mul_i32', (EXEC_COPY,)),
    ):
        hlsl = lower(program(CASES[opcode], mask_ops))
        correct, wrong = [], []
        for threshold in (29, 30, 32):
            mask = MASK if label == 'exec-copy' else sum(1 << lane for lane in range(threshold, 32))
            expected = reference(opcode, mask)
            case = directory / f'{label}-{threshold}.case'
            contents = [1, *([expected] * 32), *([SENTINEL] * (WORDS - 32)), threshold, *([0] * 15)]
            case.write_bytes(struct.pack('<%dI' % len(contents), *contents))
            correct.append((case, f'threshold {threshold}', None))
            lane_zero = reference(opcode, mask & 1)
            if lane_zero != expected:
                wrong.append((case, 'lane-local bit', f'word=0 expected={expected:08x} actual={lane_zero:08x}'))
        assert wrong, label
        mutant = lane_bit_mutant(hlsl)
        for name, source, cases in ((label, hlsl, correct), (label + '-mutant', mutant, wrong)):
            src, cso = (directory / f'{name}.{ext}' for ext in ('hlsl', 'cso'))
            src.write_text(source, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', cso, src], check=True)
            if check_references(name, runner, backend, cso, cases, allow_skip=True) == 77:
                return 77
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'BACKEND'))
    args = parser.parse_args()
    controls(args.llvm_mc)
    print('numeric lane-mask translation controls passed')
    if args.dispatch:
        with tempfile.TemporaryDirectory() as tmp:
            return dispatch(Path(tmp), *args.dispatch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
