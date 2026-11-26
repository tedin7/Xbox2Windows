"""Compute tick mode: a branch that decides for the whole guest wave runs once for the wave, never per lane.

  (no arguments)                                    translate the programs and check that they take tick mode (Python)
  --dispatch <dxc> <offscreen_check> <adapter> [fxc] compile, run GPU readback against the CPU reference, run the
                                                    mutants that must be rejected, and (with fxc) check that FXC stops
                                                    at the DXC-only guard instead of compiling the tick loop

Every program is case A of startup_engine_compute_check.py with a block spliced in before its store. Case A stores
word x + 2y + 6z + 24 * (gx + 3gy + 6gz) for thread (x, y, z) of group (gx, gy, gz); s20 holds 24 * (gx + 3gy + 6gz)
there. The spliced block is skipped or run by a wave-wide branch and writes s20 = gx + 1, which the store then writes:

    v_cmp_gt_u32 vcc_lo, 1, v0 ; (exec = vcc by s_mov or s_and_saveexec) ; s_cbranch_execz skip
    s_add_u32 s20, s7, 1                      (s7 = gx)
    skip: (exec restored) ; v_mov_b32 v5, s20

The guest wave runs the block when ANY lane of the wave decides so, and then every lane of the wave stores gx + 1; a
per-lane skip leaves the old s20 in the lanes that did not decide. The programs cover a 24-thread wave32 group, a
2-thread group, a 72-thread wave64 group (a full wave and a short last wave of 8 lanes, each deciding through the low
or the high ballot word), and s_cbranch_vccz on a VCC written under an older EXEC: the guest tests every bit of VCC,
also the bits of lanes the newer EXEC turned off, and a compare writes 0 into the VCC bit of an inactive lane.
"""
import x2w_engine_shim  # noqa: F401 - every gfx2hlsl import below is the opt-in engine
import argparse
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))

import startup_engine_compute_check as CC  # noqa: E402

G = CC.G
ENCODING = dict(CC.SPLICE, **{
    'v_cmpx_gt_u32 1, v0': '8100a87d',
    'v_cmp_eq_u32 vcc_lo, 1, v0': '8100847d',
    'v_cmp_eq_u32 vcc, 40, v0': 'a800847d',
    'v_cmp_lt_u32 vcc, 64, v0': 'c000827d',
    's_and_saveexec_b64 s[0:1], vcc': '6a2480be',
    's_or_b64 exec, exec, s[0:1]': '7e00fe88',
})
SKIP = ['s_cbranch_execz 1', 's_add_u32 s20, s7, 1']
ENTER = ['s_mov_b32 s21, exec_lo', 'v_cmp_gt_u32 vcc_lo, 1, v0', 's_mov_b32 exec_lo, vcc_lo']
SAVEEXEC = ['v_cmp_gt_u32 vcc_lo, 1, v0', 's_and_saveexec_b32 s0, vcc_lo'] + SKIP + [
    's_or_b32 exec_lo, exec_lo, s0', 'v_mov_b32 v5, s20']


def wave64(compare):
    return [compare, 's_and_saveexec_b64 s[0:1], vcc'] + SKIP + ['s_or_b64 exec, exec, s[0:1]', 'v_mov_b32 v5, s20']


def vcc_branch(rewrite):
    """VCC from a compare under the full EXEC (lanes with x = 1), EXEC narrowed to x = 0 by v_cmpx, then (rewrite)
    the same compare again under the narrow EXEC, which clears the VCC bit of every lane."""
    return (['s_mov_b32 s21, exec_lo', 'v_cmp_eq_u32 vcc_lo, 1, v0', 'v_cmpx_gt_u32 1, v0']
            + ['v_cmp_eq_u32 vcc_lo, 1, v0'] * rewrite
            + ['s_cbranch_vccz 1', 's_add_u32 s20, s7, 1', 's_mov_b32 exec_lo, s21', 'v_mov_b32 v5, s20'])


def any_lane(test):
    return lambda lanes: any(test(x) for x in lanes)


ANY = '(x2w_b0 | x2w_b1) != 0u'


def site(replacement):
    return lambda hlsl: hlsl.replace(ANY, replacement)


MUTANTS = {
    # Each thread decides from its own lane bit: the per-lane skip the tick mode replaces.
    'own-lane': site('((x2w_lane < 32u ? x2w_b0 >> x2w_lane : x2w_b1 >> (x2w_lane - 32u)) & 1u) != 0u'),
    # The wave's decision read from lane 1 alone.
    'lane1': site('(x2w_b0 & 2u) != 0u'),
    # Lanes 32-63 of a wave64 left out of the decision.
    'low-word': site('x2w_b0 != 0u'),
    # Every wave reads the ballot of wave 0.
    'wave0': lambda hlsl: re.sub(r'(x2w_b[01] = x2w_ballot\[b\]\[)x2w_wave', r'\g<1>0u', hlsl),
    # vccz decided from the lanes active at the branch only.
    'active-vcc': lambda hlsl: hlsl.replace('x2w_bit = (vcc & 1u);', 'x2w_bit = (vcc & exec & 1u);'),
    # A compare keeps the old VCC bit of an inactive lane.
    'stale-vcc': lambda hlsl: re.sub(r'vcc = \(exec & (\(.*\))\);', r'vcc = (exec & 1u) != 0u ? \1 : vcc;', hlsl),
}

# name: (spliced lines, group size, group counts, guest wave, wave runs the block when, mutants that must fail)
PROGRAMS = {
    'move': (ENTER + SKIP + ['s_mov_b32 exec_lo, s21', 'v_mov_b32 v5, s20'], (2, 3, 4), (3, 2, 2), 32,
             any_lane(lambda x: x < 1), ('own-lane', 'lane1')),
    'saveexec': (SAVEEXEC, (2, 3, 4), (3, 2, 2), 32, any_lane(lambda x: x < 1), ('own-lane', 'lane1')),
    'two-threads': (SAVEEXEC, (2, 1, 1), (3, 2, 2), 32, any_lane(lambda x: x < 1), ('own-lane', 'lane1')),
    'wave64-high-word': (wave64('v_cmp_eq_u32 vcc, 40, v0'), (72, 1, 1), (1, 1, 1), 64, any_lane(lambda x: x == 40),
                         ('own-lane', 'lane1', 'low-word')),
    'wave64-short-wave': (wave64('v_cmp_lt_u32 vcc, 64, v0'), (72, 1, 1), (1, 1, 1), 64, any_lane(lambda x: x > 64),
                          ('own-lane', 'wave0')),
    'vccz-older-exec': (vcc_branch(False), (2, 3, 4), (3, 2, 2), 32, any_lane(lambda x: x == 1),
                        ('own-lane', 'active-vcc')),
    'vccz-rewritten': (vcc_branch(True), (2, 3, 4), (3, 2, 2), 32, lambda lanes: False, ('stale-vcc',)),
}


def translate(name):
    lines, group, _, wave, _, _ = PROGRAMS[name]
    code = CC.WORDS_A[:CC.STORE_A] + b''.join(bytes.fromhex(ENCODING[l]) for l in lines) + CC.WORDS_A[CC.STORE_A:]
    return CC.translate('A', code=code, group=group, wave=wave)


def expected(name):
    """The guest, lane by lane: a wave runs the block when its decision holds for its lanes (x of each)."""
    _, (sx, sy, sz), (nx, ny, nz), wave, runs, _ = PROGRAMS[name]
    out = [CC.SENTINEL] * CC.WORDS
    threads = [(x, y, z) for z in range(sz) for y in range(sy) for x in range(sx)]
    waves = [threads[i:i + wave] for i in range(0, len(threads), wave)]
    for gz in range(nz):
        for gy in range(ny):
            for gx in range(nx):
                base = 24 * (gx + 3 * gy + 6 * gz)
                for lanes in waves:
                    value = gx + 1 if runs([x for x, _, _ in lanes]) else base
                    for x, y, z in lanes:
                        out[base + x + 2 * y + 6 * z] = value
    return out


def shape(hlsl):
    assert 'x2w_ballot' in hlsl and hlsl.count(ANY) == 1, 'one tick site decides the branch'
    assert hlsl.startswith(G.TICK_HEADER) and '#ifndef __hlsl_dx_compiler' in hlsl, 'tick header and DXC-only guard'


def compile_and_run(dxc, runner, adapter, td, label, text, case):
    src, cso = td / (label + '.hlsl'), td / (label + '.cso')
    src.write_text(text, encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    run = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)
    print(run.stdout, end='')
    return run


def dispatch(dxc, runner, adapter, fxc=None):
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        td = Path(tmp)
        for name, (_, _, groups, _, _, kills) in PROGRAMS.items():
            case = td / (name + '.case')
            case.write_bytes(struct.pack('<I', groups[0] | groups[1] << 8 | groups[2] << 16)
                             + struct.pack('<%dI' % CC.WORDS, *expected(name))
                             + struct.pack('<16I', CC.KEY, CC.STRIDE, *([0] * 14)))
            hlsl = translate(name).hlsl_cs()
            shape(hlsl)
            run = compile_and_run(dxc, runner, adapter, td, name + '-baseline', hlsl, case)
            if run.returncode == 77:
                return 77
            assert not run.returncode and not [l for l in run.stderr.splitlines()
                                               if not l.startswith('XBOX2WINDOWS_ADAPTER: skipping ')], run.stderr
            print('%s baseline (%s): all words match' % (name, adapter))
            for label in kills:
                text = MUTANTS[label](hlsl)
                assert text != hlsl, 'mutant %s does not apply to %s' % (label, name)
                run = compile_and_run(dxc, runner, adapter, td, '%s-%s' % (name, label), text, case)
                assert run.returncode == 1, 'mutant %s/%s was not rejected: %r' % (name, label, run.stderr)
                print('%s mutant %s (%s): rejected' % (name, label, adapter))
            if fxc:
                run = subprocess.run([fxc, '/nologo', '/T', 'cs_5_0', '/E', 'main', '/Fo', str(td / 'fxc.cso'),
                                      str(td / (name + '-baseline.hlsl'))], capture_output=True, text=True)
                assert run.returncode and 'needs DXC' in run.stdout + run.stderr, (run.returncode, run.stdout, run.stderr)
                print('%s: FXC stops at the DXC-only guard' % name)
    return 0


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dispatch', nargs='+', metavar='ARG', help='DXC RUNNER ADAPTER [FXC]')
    args = parser.parse_args(argv)
    for name, (_, _, _, _, _, kills) in PROGRAMS.items():
        hlsl = translate(name).hlsl_cs()
        shape(hlsl)
        for label in kills:
            assert MUTANTS[label](hlsl) != hlsl, 'mutant %s does not apply to %s' % (label, name)
    if args.dispatch:
        if len(args.dispatch) not in (3, 4):
            parser.error('--dispatch takes DXC RUNNER ADAPTER [FXC]')
        return dispatch(*args.dispatch)
    print('compute tick mode: translation shape ok')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
