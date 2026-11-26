"""Compute wave-wide skip of an s_cbranch_execz block that holds v_readfirstlane_b32 (forward branches only).

  (no arguments)                                translate case A with the block and check the shape (pure Python)
  --dispatch <dxc> <offscreen_check> <adapter>  compile the HLSL, run GPU readback against the CPU reference, then run
                                                two mutants that must be rejected

The program is case A of startup_engine_compute_check.py (one 24-thread wave per group, 3 x 2 x 2 groups) with this
block before its store:

    v_cmp_lt_u32 vcc_lo, s7, v0 ; s_mov_b32 exec_lo, vcc_lo ; s_cbranch_execz skip
    v_readfirstlane_b32 s22, v5 ; v_add_nc_u32 v5, s22, v5
    skip: s_mov_b32 exec_lo, s21

Groups with gid.x = 0 run the block for the lanes with tid.x = 1 (the odd lanes): the first active lane is lane 1, so
those lanes add the value of lane 1, not lane 0. Groups with gid.x = 1 or 2 have no active lane and skip the block.
"""
import x2w_engine_shim  # noqa: F401 - every gfx2hlsl import below is the opt-in engine
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))

import startup_engine_compute_check as CC  # noqa: E402

G = CC.G
BLOCK = ['s_mov_b32 s21, exec_lo', 'v_cmp_lt_u32 vcc_lo, s7, v0', 's_mov_b32 exec_lo, vcc_lo', 's_cbranch_execz 2',
         'v_readfirstlane_b32 s22, v5', 'v_add_nc_u32 v5, s22, v5', 's_mov_b32 exec_lo, s21']
LANES, GROUPS = 24, (3, 2, 2)
MASK = 0xffffffff


# Instruction words recorded from llvm-mc 19 (-triple=amdgcn--amdpal -mcpu=gfx1013).
RECORDED = {
    's_mov_b32 s21, exec_lo': '7e0395be',
    'v_cmp_lt_u32 vcc_lo, s7, v0': '0700827d',
    's_mov_b32 exec_lo, vcc_lo': '6a03febe',
    's_cbranch_execz 2': '020088bf',
    's_barrier': '00008abf',
    'v_readfirstlane_b32 s22, v5': '05052c7e',
    'v_add_nc_u32 v5, s22, v5': '160a0a4a',
    's_mov_b32 exec_lo, s21': '1503febe',
}


def program():
    """The case A bytes with BLOCK spliced in before its store."""
    for line, words in RECORDED.items():
        CC.SPLICE.setdefault(line, words)
    return CC.splice(*BLOCK)


def translate():
    return CC.translate('A', code=program())


def expected():
    """Case A's words, then the block: lanes with tid.x = 1 in groups with gid.x = 0 add the value of lane 1."""
    out = list(CC.expected('A'))
    base = CC.expected('A')
    for gz in range(GROUPS[2]):
        for gy in range(GROUPS[1]):
            gx = 0
            first = (gx + 3 * gy + 6 * gz) * LANES
            for lane in range(LANES):
                if lane % 2 == 1:  # tid.x = lane & 1 for local (2, 3, 4)
                    out[first + lane] = (base[first + 1] + base[first + lane]) & MASK
    return out


MUTANTS = {
    # The lowest active lane replaced by lane 0: the odd lanes add lane 0's value.
    'lane0': ('x2w_first_1 == 0xffffffffu ? x2w_value_2 : x2w_value_1', 'x2w_value_2'),
    # The vector write loses its exec guard: the even lanes add too.
    'unguarded': ('if (exec & 1) v5 = s22 + v5;', 'v5 = s22 + v5;'),
}


def shape(hlsl):
    assert 'skip_to = 2' in hlsl and 'bool skip = false' in hlsl, 'wave-wide skip flag'
    assert hlsl.count('GroupMemoryBarrierWithGroupSync()') >= 4, 'prologue + execz + first lane + lane 0'
    guarded = False
    for line in hlsl.splitlines():
        if line.strip() == 'if (!skip) {':
            guarded = True
        elif line.strip() == '}':
            guarded = False
        assert not (guarded and 'GroupMemoryBarrierWithGroupSync' in line), 'a barrier inside a guard'


def dispatch(dxc, runner, adapter):
    hlsl = translate().hlsl_cs()
    shape(hlsl)
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    words = expected()
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        td = Path(tmp)
        case = td / 'execz_wave.case'
        constants = [CC.KEY, CC.STRIDE] + [0] * 14
        case.write_bytes(struct.pack('<I', GROUPS[0] | GROUPS[1] << 8 | GROUPS[2] << 16)
                         + struct.pack('<%dI' % CC.WORDS, *(words + [CC.SENTINEL] * (CC.WORDS - len(words))))
                         + struct.pack('<16I', *constants))
        for label, mutant in [('baseline', None)] + list(MUTANTS.items()):
            text = hlsl
            if mutant:
                assert text.count(mutant[0]) == 1, label
                text = text.replace(*mutant)
            src, cso = td / (label + '.hlsl'), td / (label + '.cso')
            src.write_text(text, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
            run = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)],
                                 capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            if mutant:
                assert run.returncode == 1, 'mutant %s was not rejected: %r' % (label, run.stderr)
                print('mutant %s (%s): rejected' % (label, adapter))
            else:
                assert not run.returncode and not [l for l in run.stderr.splitlines()
                                                   if not l.startswith('XBOX2WINDOWS_ADAPTER: skipping ')], run.stderr
                print('baseline (%s): all words match' % adapter)
    return 0


def controls():
    # A branch on exec alone names no wave size: the retry is refused, not run as wave 32.
    program()
    assert G.guest_wave_size(CC.splice('s_cbranch_execz 1', 'v_mov_b32 v5, s20')) is None
    # A group barrier inside a block the branch can skip is refused (the guest never skips it for one wave only, but the
    # host flow would not be uniform).
    inside = BLOCK[:5] + ['s_barrier'] + BLOCK[5:]
    try:
        text = CC.translate('A', code=CC.splice(*inside)).hlsl_cs()
    except G.Unsupported:
        pass
    else:
        # Tick mode (bench c4 S5) takes a guest barrier as a park site, so the program is translated by the tick machine,
        # never by the linear skip: the group barrier is then reached by every thread at the tick boundary.
        assert 'x2w_parked' in text, 'a barrier inside the skipped block was accepted outside tick mode'


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args(argv)
    shape(translate().hlsl_cs())
    controls()
    if args.dispatch:
        return dispatch(*args.dispatch)
    print('compute execz wave-wide skip: translation shape ok')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
