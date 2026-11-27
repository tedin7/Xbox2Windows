"""Source-built gfx1013 compute fixture: a uniform-address buffer load read with v_readfirstlane_b32 after a v_cmpx.

  (no arguments)                                decode and translate, check the HLSL and the refusals (pure Python)
  --llvm-mc <llvm-mc>                           the same, with the fixtures assembled by llvm-mc
  --dispatch <dxc> <offscreen_check> <adapter>  compile the HLSL with DXC, run GPU readback against a CPU reference

A v_cmpx keeps only the lanes the mask already had, so a load whose address is the same in every lane still gives every
active lane one value, and the first active lane's v_readfirstlane_b32 of it is that value. The translation used to
drop that proof for good at the first v_cmpx (EXEC was no longer the entry mask) and refused the read.

  v7 = 13; exec &= (lane < s4); v8 = 8; v9 = word 2 of the input buffer (load at the one address v8);
  s22 = readfirstlane(v9); v7 = v7 + s22 for the active lanes; exec restored; every lane stores v7.
"""
import argparse
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))

import startup_compute_waterfall as waterfall  # noqa: E402
from compute_load_format_check import check_references  # noqa: E402
from startup_gfx2hlsl_compute_check import SENTINEL, WORDS, assemble, refused  # noqa: E402

ADDRESS = ('v_mov_b32_e32 v8, 8', '8802107e')
VARYING_ADDRESS = ('v_mov_b32_e32 v8, v0', '0003107e')
LOAD = ('buffer_load_dword v9, v8, s[12:15], 0 offen', '001030e008090380')
WAIT = ('s_waitcnt vmcnt(0)', '703f8cbf')
FIRST_LANE = ('v_readfirstlane_b32 s22, v9', '09052c7e')
USE = ('v_add_nc_u32 v7, s22, v7', '160e0e4a')
REFUSAL = 'v_readfirstlane_b32 requires a proven uniform source'
# A forward branch on the uniform root constant (never equal to the lane count here): the load is past the entry block, as in
# the shaders this is for, where the proof recomputes the load instead of reading one lane's copy.
BRANCH = (('s_cmp_eq_u32 s4, 7', '048706bf'), ('s_cbranch_scc0 1', '010084bf'), ('s_nop 0', '000080bf'))


def program(*, address=ADDRESS, after_cmpx=()):
    return (*waterfall.PREFIX, *after_cmpx, *BRANCH, address, LOAD, WAIT, FIRST_LANE, USE, waterfall.RESTORE, waterfall.STORE, waterfall.END)


def translate(rows, group=(4, 1, 1)):
    return waterfall.translate(rows, group)


def controls(llvm_mc=None):
    cases = (
        (program(), None),
        # Replacing EXEC with a literal can enable lanes outside the known subset.
        (program(after_cmpx=(('s_mov_b32 exec_lo, 1', '8103febe'),)), REFUSAL),
        # An address that differs between lanes is not one value for the first active lane.
        (program(address=VARYING_ADDRESS), REFUSAL),
    )
    for rows, reason in cases:
        if llvm_mc:
            source = '\n'.join(asm for asm, _ in rows) + '\n'
            assert assemble(llvm_mc, source) == bytes.fromhex(''.join(word for _, word in rows)), source
        if reason:
            refused(reason, lambda: translate(rows))
        else:
            hlsl = translate(rows)
            assert re.search(r's22 = ul[0-9a-f_]+;', hlsl), hlsl
    print('readfirstlane narrowed-exec checks passed')


def expected(active, source):
    return [13 + source[2] if lane < active else 13 for lane in range(4)]


def dispatch(directory, dxc, runner, adapter):
    controls()
    hlsl = translate(program())
    # The control reads zero instead of the loaded word: the readback must then differ from the reference.
    mutant = re.sub(r's22 = ul[0-9a-f_]+;', 's22 = 0u;', hlsl, count=1)
    assert mutant != hlsl
    for label, text in (('baseline', hlsl), ('dropped-load', mutant)):
        src, shader = directory / ('narrow_%s.hlsl' % label), directory / ('narrow_%s.cso' % label)
        src.write_text(text, encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
        cases = []
        for active in range(5):
            source = [17 * i + 9 for i in range(WORDS)]
            output = expected(active, source)
            case = directory / ('narrow-%s-%d.case' % (label, active))
            words = [1, *output, *([SENTINEL] * (WORDS - 4)), active, *([0] * 15), *source]
            case.write_bytes(struct.pack('<%dI' % len(words), *words))
            if label == 'baseline':
                cases.append((case, '%d active lanes' % active, None))
            elif active:  # no active lane: the dropped load changes nothing
                lane = 0
                cases.append((case, '%d active lanes, load dropped' % active,
                              'word=%d expected=%08x actual=%08x' % (lane, output[lane], 13)))
        if check_references('narrowed exec %s' % label, runner, adapter, shader, cases) == 77:
            return 77
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args(argv)
    if args.dispatch:
        with tempfile.TemporaryDirectory(dir=HERE.parent / 'out' if (HERE.parent / 'out').is_dir() else None) as tmp:
            return dispatch(Path(tmp), *args.dispatch)
    controls(args.llvm_mc)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
