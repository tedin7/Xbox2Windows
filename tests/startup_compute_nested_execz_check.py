"""Synthetic nested gfx1013 EXECZ regions, checked against a four-lane CPU reference."""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import startup_compute_waterfall as W
from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, assemble, refused, root_blob


OUTER_MASK = W.PREFIX[-1]
INNER_MASK = ('v_cmpx_gt_u32_e32 s5, v0', '0500a87d')
NARROW = ('v_cmpx_lt_u32_e32 0, v0', '8000a27d')
CMP = ('s_cmp_eq_u32 s5, 1', '058106bf')
RESTORE_OUTER = ('s_mov_b32 exec_lo, s21', '1503febe')
INNER_MARK = ('v_mov_b32_e32 v7, 19', '93020e7e')
OUTER_MARK = ('v_add_nc_u32_e32 v7, 7, v7', '870e0e4a')
INIT_SCALAR = ('s_mov_b32 s23, 9', '890397be')
MOVE_SCALAR = ('s_mov_b32 s23, 17', '910397be')
SCALAR_RESULT = ('v_mov_b32_e32 v7, s23', '17020e7e')
NOP = ('s_nop 0', '000080bf')
BRANCHES = {'s_cbranch_execz': 0xbf880000, 's_cbranch_scc1': 0xbf850000, 's_cbranch_scc0': 0xbf840000}
INPUTS = ((0, 4), (4, 0), (4, 1), (1, 4), (4, 2), (4, 4))
CASES = ('execz-exit', 'scc0-exit', 'live-sgpr')
REFUSALS = {
    'execz-exit': 's_cbranch_execz inside a block s_cbranch_execz skips per lane: branch out of the block',
    'scc0-exit': 's_cbranch_scc0 inside a block s_cbranch_execz skips per lane: branch out of the block',
    'live-sgpr': 's_mov_b32 inside a block s_cbranch_execz skips per lane: definition of s23, live where the block joins',
}


def program(case):
    scalar = case == 'live-sgpr'
    rows = [*W.PREFIX[:5], W.PREFIX[7], W.PREFIX[8]]
    if scalar:
        rows.append(INIT_SCALAR)
    rows += [OUTER_MASK, ('s_cbranch_execz', 'outer'), W.SAVE, INNER_MASK, ('s_cbranch_execz', 'inner')]
    if scalar:
        rows.append(MOVE_SCALAR)
    else:
        # The SCC0 exits unless the inner threshold is 1. EXECZ exits if narrowing removes the sole active lane.
        rows += ([NARROW, ('s_cbranch_execz', 'outer')] if case == 'execz-exit'
                 else [CMP, ('s_cbranch_scc0', 'outer')])
        rows.append(INNER_MARK)
    rows += ['inner', RESTORE_OUTER, NOP if scalar else OUTER_MARK, 'outer', W.RESTORE]
    if scalar:
        rows.append(SCALAR_RESULT)
    return [*rows, W.STORE, W.END]


def encode(rows, llvm_mc=None):
    offset, labels = 0, {}
    for row in rows:
        if isinstance(row, str):
            labels[row] = offset
        else:
            asm, word = row
            offset += 1 if asm in BRANCHES else len(word) // 8
    code = bytearray()
    for row in rows:
        if isinstance(row, str):
            continue
        asm, word = row
        if asm in BRANCHES:
            delta = labels[word] - len(code) // 4 - 1
            word = struct.pack('<I', BRANCHES[asm] | (delta & 0xffff)).hex()
            asm += ' %d' % delta
        data = bytes.fromhex(word)
        if llvm_mc:
            assert assemble(llvm_mc, asm + '\n') == data, asm
        code += data
    return bytes(code)


def translate(rows, llvm_mc=None, group=(4, 1, 1)):
    return G.ComputePipeline(encode(rows, llvm_mc), root_blob(), USER7, group).hlsl_cs()


def reference(case, n, m, mutant=False):
    outer = set(range(min(n, 4)))
    inner = outer & set(range(m))
    if case == 'live-sgpr':
        return [17 if inner and (not mutant or lane in inner) else 9 for lane in range(4)]
    active = inner - {0} if case == 'execz-exit' else inner
    exits = inner and (not active if case == 'execz-exit' else m != 1)
    if not mutant and exits:
        return [13] * 4
    return [(19 if lane in active else 13) + (7 if lane in outer else 0) for lane in range(4)]


def mutate(case, hlsl):
    if case == 'live-sgpr':
        old, new = 's23 = 0x00000011u;', 'if (exec & 1u) s23 = 0x00000011u;'
        assert hlsl.count(old) == 1, hlsl
    else:
        old = ' = false;'
        # Match the running flag alone. Other bool declarations/assignments keep their original values.
        old = next(line for line in hlsl.splitlines() if 'x2w_running_' in line and line.endswith(old))
        new = old.replace(' = false;', ' = true;')
        assert hlsl.count(old) == 1, hlsl
    return hlsl.replace(old, new)


def controls(llvm_mc=None):
    for case in CASES:
        rows = program(case)
        hlsl = translate(rows, llvm_mc)
        assert hlsl.count('if (x2w_execz_any(') >= 1 and hlsl.count('x2w_execz_any(tid,') >= 2, hlsl
        assert 'uint pc =' not in hlsl and mutate(case, hlsl) != hlsl
        assert any(reference(case, *args) != reference(case, *args, mutant=True) for args in INPUTS)
        with patch.object(G.Stage, 'structured_execz_plan', return_value=None):
            refused(REFUSALS[case], lambda: translate(rows))
        refused(REFUSALS[case], lambda: translate(rows, group=(33, 1, 1)))
    rows = program('live-sgpr')
    rows[rows.index(MOVE_SCALAR)] = ('s_mov_b32 s23, exec_lo', '7e0397be')
    refused(REFUSALS['live-sgpr'], lambda: translate(rows))
    rows = program('scc0-exit')
    rows[rows.index(CMP)] = ('s_cmp_eq_u32 exec_lo, 1', '7e8106bf')
    refused(REFUSALS['scc0-exit'], lambda: translate(rows))
    rows = program('scc0-exit')
    rows[rows.index(('s_cbranch_scc0', 'outer'))] = ('s_cbranch_scc0', 'after')
    rows.insert(rows.index(W.RESTORE) + 1, 'after')
    refused(REFUSALS['scc0-exit'], lambda: translate(rows))
    print('nested EXECZ translations, refusal controls and CPU mutant references passed')


def dispatch(directory, dxc, runner, adapter):
    for case in CASES:
        hlsl = translate(program(case))
        refs, mutants = [], []
        for n, m in INPUTS:
            expected = reference(case, n, m)
            wrong = reference(case, n, m, mutant=True)
            path = directory / f'{case}-{n}-{m}.case'
            words = [1, *expected, *([SENTINEL] * (WORDS - 4)), n, m, *([0] * 14), *([0] * WORDS)]
            path.write_bytes(struct.pack('<%dI' % len(words), *words))
            label = f'outer={n} inner={m}'
            refs.append((path, label, None))
            if wrong != expected:
                lane = next(i for i, value in enumerate(expected) if value != wrong[i])
                mutants.append((path, label, f'word={lane} expected={expected[lane]:08x} actual={wrong[lane]:08x}'))
        assert mutants
        for name, text, checks in ((case, hlsl, refs), (case + '-mutant', mutate(case, hlsl), mutants)):
            src, shader = directory / f'{name}.hlsl', directory / f'{name}.cso'
            src.write_text(text, encoding='ascii')
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
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            return dispatch(Path(tmp), *args.dispatch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
