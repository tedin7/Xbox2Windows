"""Synthetic gfx1013 .c67 programs for entry EXEC recovery and scalar waterfall reads."""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_compute_waterfall as waterfall
from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import SENTINEL, WORDS, assemble, refused

REFUSAL = 'v_readfirstlane_b32 requires a proven uniform source'
CMP = ('s_cmp_eq_u32 s4, 7', '048706bf')
SKIP_TWO = ('s_cbranch_scc1 2', '020085bf')
SKIP_ONE = ('s_cbranch_scc1 1', '010085bf')
MOVE = ('v_mov_b32_e32 v9, s5', '0502127e')
ZERO = ('v_mov_b32_e32 v9, 0', '8002127e')
VARYING = ('v_mov_b32_e32 v9, v0', '0003127e')
FORMAT = ('buffer_load_format_x v9, v9, s[12:15], 0 idxen', '002000e009090380')
VM_WAIT = ('s_waitcnt vmcnt(0)', '703f8cbf')
READ = ('v_readfirstlane_b32 s22, v9', '09052c7e')
USE = ('v_add_nc_u32 v7, s22, v7', '160e0e4a')
WIDE_DESC = ('s_load_dwordx8 s[28:35], s[10:11], 0', '05070cf4000000fa')
LGKM_WAIT = ('s_waitcnt lgkmcnt(0)', '7fc08cbf')
BFE = ('s_bfe_u32 s23, s22, 0x40000', '16ff979300000400')
MAX = ('s_max_u32 s23, s23, 2', '17829784')
MUL = ('s_mul_i32 s23, s23, 9', '17891793')
SCALAR_LOAD = ('s_buffer_load_dwordx2 s[24:25], s[32:35], s23', '100624f40000002e')
XOR_LOW = ('v_xor_b32_e32 v7, s24, v0', '18000e3a')
XOR_HIGH = ('v_xor_b32_e32 v7, s25, v7', '190e0e3a')
DROP_HIGH = ('v_xor_b32_e32 v7, 0, v7', '800e0e3a')
BODY = (WIDE_DESC, LGKM_WAIT, BFE, MAX, MUL, SCALAR_LOAD, LGKM_WAIT, XOR_LOW, XOR_HIGH)
SHIFTED_TABLE = ('s_load_dwordx8 s[12:19], s[10:11], 16', '05030cf4100000fa')


def at_branch(*, join=False, back_edge=False, index=MOVE):
    # The taken edge still carries entry EXEC. In the control the narrowed arm can also reach the load.
    prefix = waterfall.PREFIX[:-1]
    clear = ('s_mov_b32 exec_lo, 0', '8003febe')
    arms = (CMP, SKIP_ONE, clear) if join else (CMP, SKIP_TWO, clear, waterfall.END)
    tail = (clear, ('s_branch -10', 'f6ff82bf')) if back_edge else ()
    return (*prefix, *arms, index, FORMAT, VM_WAIT, READ, USE, waterfall.STORE, *tail, waterfall.END)


def after_restore(*, clobber=(), value=MOVE, wide=False, move_before=False):
    prefix, restore = waterfall.PREFIX, waterfall.RESTORE
    if wide:
        save = ('s_mov_b64 s[26:27], exec', '7e049abe')
        restore = ('s_mov_b64 exec, s[26:27]', '1a04febe')
        prefix = (*prefix[:-2], save, prefix[-1])
    if move_before:
        prefix = (*prefix[:-1], value, prefix[-1])
    suffix = (restore,) if move_before else (restore, value)
    return (*prefix, *clobber, *suffix, READ, USE, waterfall.STORE, waterfall.END)


def in_waterfall(body=BODY, *, writable=False, **kwargs):
    rows = list(waterfall.program(body=body, **kwargs))
    rows[2] = SHIFTED_TABLE  # t0 comes from the second descriptor of WIDE_DESC, u0 from the next slot.
    if writable:
        rows[-2] = ('buffer_store_dword v7, v4, s[12:15], 0 offen', '001070e004070380')
    return rows


def translate(code, refusals=None):
    return waterfall.G.ComputePipeline(code, waterfall.root_blob(), waterfall.USER7, (4, 1, 1), refusals=refusals)


def bytecode(rows, llvm_mc=None):
    code = bytes.fromhex(''.join(word for _, word in rows))
    if llvm_mc:
        source = '\n'.join(asm for asm, _ in rows) + '\n'
        assert assemble(llvm_mc, source) == code, source
    return code


def controls(llvm_mc=None):
    for label, rows, mutant in programs():
        code = bytecode(rows, llvm_mc)
        bytecode(mutant, llvm_mc)
        pipe = translate(code)
        assert pipe.retry_entry_exec == (label != 'waterfall-smem'), label
        pipe.hlsl_cs()
        refusals = []
        translate(code, refusals)
        assert not refusals, refusals
    clobber = ('s_mov_b32 s26, 0', '80039abe')
    high_clobber = ('s_mov_b32 s27, 0', '80039bbe')
    key_table = ('s_load_dwordx8 s[28:35], s[22:23], 0', '0b070cf4000000fa')
    leak = ('v_mov_b32_e32 v7, s25', '19020e7e')
    for rows, reason in (
        (at_branch(join=True), REFUSAL),
        (at_branch(back_edge=True), REFUSAL),
        (at_branch(index=VARYING), REFUSAL),
        (after_restore(clobber=(clobber,)), REFUSAL),
        (after_restore(value=VARYING), REFUSAL),
        (after_restore(move_before=True, clobber=(VARYING,)), REFUSAL),
        (after_restore(wide=True, clobber=(high_clobber,)), REFUSAL),
        (in_waterfall(body=(BFE, key_table, *BODY[3:])), 'per-lane descriptor selection'),
        (in_waterfall(writable=True), 'requires a read-only descriptor'),
        (in_waterfall(after_restore=(leak,)), 'body scalar is live at the loop exit'),
    ):
        code = bytecode(rows, llvm_mc)
        refused(reason, lambda: translate(code).hlsl_cs())
    print('entry EXEC and scalar waterfall translation and refusal checks passed')


def programs():
    return (
        ('branch-format', at_branch(), at_branch(index=ZERO)),
        ('restore-valu32', after_restore(), after_restore(value=ZERO)),
        ('restore-valu64', after_restore(wide=True, move_before=True),
         after_restore(wide=True, move_before=True, value=ZERO)),
        ('waterfall-smem', in_waterfall(), in_waterfall(body=(*BODY[:-1], DROP_HIGH))),
    )


def dispatch(directory, dxc, runner, adapter):
    # The case runner supplies one t0, one u0 and sixteen entry constants.
    for label, rows, mutant in programs():
        branch = label == 'branch-format'
        loop = label == 'waterfall-smem'
        source = [17 * i + 9 for i in range(WORDS)]
        source[:4] = [0x11, 0x5, 0x21, 0x37]  # duplicate extracted key, clamped keys, and separate offsets
        inputs = ((7, 6), (6, 6), (7, 8)) if branch else [(n, 6) for n in range(5)]
        cases, mutations = [], []
        for active, index in inputs:
            roots = [active, index] + ([4, (22 << 12) | 4] if branch else [])
            expected = [13] * 4
            if loop:
                for lane, key in enumerate(source[:active]):
                    word = 9 * max(key & 15, 2) // 4
                    expected[lane] = source[word] ^ source[word + 1] ^ lane
                wrong = source[4] if active else expected[0]
            elif branch:
                expected = [13 + source[index]] * 4 if active == 7 else [SENTINEL] * 4
                wrong = 13 + source[0] if active == 7 else expected[0]
            else:
                expected = [13 + index] * 4
                wrong = 13
            case = directory / f'{label}-{active}-{index}.case'
            words = [1, *expected, *([SENTINEL] * (WORDS - 4)), *roots, *([0] * (16 - len(roots))), *source]
            case.write_bytes(struct.pack('<%dI' % len(words), *words))
            cases.append((case, f'root constants {roots}', None))
            if wrong != expected[0]:
                mutations.append((case, f'mutant with root constants {roots}',
                                  f'word=0 expected={expected[0]:08x} actual={wrong:08x}'))
        assert mutations, label
        for name, variant, checks in ((label, rows, cases), (f'{label}-mutant', mutant, mutations)):
            code = directory / f'{name}.c67'
            code.write_bytes(bytecode(variant))
            hlsl = translate(code.read_bytes()).hlsl_cs()
            src, shader = code.with_suffix('.hlsl'), code.with_suffix('.cso')
            src.write_text(hlsl, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
            if check_references(name, runner, adapter, shader, checks, allow_skip=True) == 77:
                return 77
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args()
    controls(args.llvm_mc)
    if args.dispatch:
        output = Path(__file__).resolve().parent.parent / 'out'
        with tempfile.TemporaryDirectory(dir=output if output.is_dir() else None) as tmp:
            return dispatch(Path(tmp), *args.dispatch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
