"""Synthetic gfx1013 scalar counted loop with a per-lane EXEC guard."""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_compute_waterfall as fixture
from compute_load_format_check import check_references
from startup_compute_readfirstlane_exec_check import bytecode
from startup_gfx2hlsl_compute_check import SENTINEL, WORDS, refused

INIT = ('s_mov_b32 s23, 0', '800397be')
GUARD = ('s_cbranch_execz 4', '040088bf')
CMP = ('s_cmp_lt_u32 s23, 5', '17850abf')
EXIT = ('s_cbranch_scc0 3', '030084bf')
ACCUMULATE = ('v_add_nc_u32_e32 v7, s23, v7', '170e0e4a')
MUTANT = ('s_cmp_lt_u32 s23, 4', '17840abf')
STEP = ('s_add_u32 s23, s23, 1', '17811780')
LATCH = ('s_branch -5', 'fbff82bf')
LEAK = ('v_mov_b32_e32 v7, s23', '17020e7e')
REFUSAL = ('s_cbranch_scc0 inside a block s_cbranch_execz skips per lane: '
           'loop exit crosses a backward s_branch')


def program(*, cmp=CMP, after=(), step=STEP):
    return (*fixture.PREFIX, INIT, GUARD, cmp, EXIT, ACCUMULATE, step, LATCH,
            fixture.RESTORE, *after, fixture.STORE, fixture.END)


def controls(llvm_mc):
    for cmp in (CMP, MUTANT):
        rows = program(cmp=cmp)
        bytecode(rows, llvm_mc)
        hlsl = fixture.translate(rows)
        guard = hlsl.index('if (!((exec & 1) == 0)) {')
        loop = hlsl.index('[loop] while (true) {', guard)
        stop = hlsl.index('if (!scc) break;', loop)
        add = hlsl.index('v7 = ', stop)
        restore = hlsl.index('exec = ', add)
        assert guard < loop < stop < add < restore
        assert 'uint guard = ' not in hlsl
    for rows in (program(after=(LEAK,)),
                 program(step=('s_add_u32 s23, s23, 2', '17821780'))):
        bytecode(rows, llvm_mc)
        refused(REFUSAL, lambda: fixture.translate(rows, (65, 1, 1)))
    print('EXEC-guarded scalar loop translation and refusal checks passed')


def dispatch(directory, dxc, renderer, backend):
    reference, negative = [], []
    for active in (0, 1, 2, 4):
        expected = [13 + sum(range(5)) if lane < active else 13 for lane in range(4)]
        words = [1, *expected, *([SENTINEL] * (WORDS - 4)), active, *([0] * 15), *([0] * WORDS)]
        path = directory / f'active-{active}.case'
        path.write_bytes(struct.pack(f'<{len(words)}I', *words))
        reference.append((path, f'{active} active lanes', None))
        if active:
            negative.append((path, f'{active} active lanes, wrong SCC limit',
                             f'word=0 expected={expected[0]:08x} actual={13 + sum(range(4)):08x}'))
    for label, cmp, cases in (('counted-loop', CMP, reference), ('mutant', MUTANT, negative)):
        src, shader = directory / f'{label}.hlsl', directory / f'{label}.cso'
        src.write_text(fixture.translate(program(cmp=cmp)), encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
        if check_references(label, renderer, backend, shader, cases, allow_skip=True) == 77:
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
