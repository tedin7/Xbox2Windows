"""A scalar register joined at a branch target and read as a number, in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_join_phi_check.py                              translation controls (Linux)
       startup_gfx2hlsl_join_phi_check.py draw <dxc> <offscreen_check> <adapter>
       startup_gfx2hlsl_join_phi_check.py binary <llvm-mc>              the checked-in words against the assembler

A pixel program loads four descriptor words from a different descriptor-table entry on each arm of a
uniform branch, joins, and converts two of the words to float. Each word of a fixed table entry reaches
the shader as an entry value, so every incoming value is a known scalar. The translator first refuses the
join; after that refusal it translates again and every incoming edge assigns the register, so the join
reads the register of the path that was taken.

Controls: an arm that loads through a register (not a known scalar) keeps its refusal; a program that
translated before is unchanged (the retry runs only after the refusal). The WARP draw runs both arms and
compares every pixel with the CPU values; a mutant that swaps the two edge assignments must differ at
every pixel.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

# s8 holds the descriptor-table pointer (s[8:9] in a load) and s9 the root constant (0 selects the first arm).
SOURCE = """s_cmp_eq_u32 s9, 0
s_cbranch_scc0 4
s_load_dwordx4 s[12:15], s[8:9], 0x10
s_waitcnt lgkmcnt(0)
s_branch 3
s_load_dwordx4 s[12:15], s[8:9], 0x30
s_waitcnt lgkmcnt(0)
v_cvt_f32_u32_e32 v2, s13
v_cvt_f32_u32_e32 v3, s14
v_cvt_f32_u32_e32 v4, v0
v_cvt_f32_u32_e32 v5, s9
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
"""
# The same program with the second arm loading through s20 (a run-time value): not a known scalar.
UNKNOWN_SOURCE = SOURCE.replace('s_load_dwordx4 s[12:15], s[8:9], 0x30', 's_load_dwordx4 s[12:15], s[20:21], 0x30')
FIRST, SECOND = 0x10, 0x30
VALUES = {FIRST: 3, FIRST + 4: 7, FIRST + 8: 11, FIRST + 12: 13, SECOND: 50, SECOND + 4: 100, SECOND + 8: 200, SECOND + 12: 300}


def words(llvm_mc, source):
    return assemble(llvm_mc, source)


def translate(ps_words):
    return G.Pipeline(VS_WORDS, ps_words, root_blob(), ps_hw_inputs=(0x8000, 0x8000))


def controls(ps_words, unknown_words):
    pipe = translate(ps_words)
    hlsl, manifest = pipe.hlsl_ps(), pipe.manifest()
    descwords = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r'^udsrc (\d+) descword 0 (\d+)$', manifest, re.M)}
    assert sorted(descwords.values()) == sorted(VALUES), manifest
    # Each arm assigns the four joined words (s12..s15) from its own entry values before the join.
    for offset in (FIRST + 4, SECOND + 4):
        index = next(i for i, off in descwords.items() if off == offset)
        assert 's13 = ud%d;' % index in hlsl, hlsl
    for name in ('s12', 's13', 's14', 's15'):
        assert len(re.findall(r'^\s+%s = ud\d+;$' % name, hlsl, re.M)) == 2, hlsl
    try:
        translate(unknown_words)
    except G.Unsupported as error:
        assert 'joined at a branch target' in str(error) or 'holding' in str(error), error
    else:
        raise AssertionError('a join with an arm that loads through a run-time register translated')
    print('join phi: both arms assign the joined words from entry values; an unknown arm keeps its refusal')
    return hlsl, descwords


def pixel_reference(selector, width=4, height=4):
    first, second = (VALUES[FIRST + 4], VALUES[FIRST + 8]) if selector == 0 else (VALUES[SECOND + 4], VALUES[SECOND + 8])
    out = bytearray()
    for y in range(height):
        for x in range(width):
            packed = x | (y << 16)
            out += struct.pack('<4f', float(first), float(second), float(packed), float(selector))
    return bytes(out)


def draw(dxc, runner, adapter, hlsl, descwords, ud_count):
    out = Path(here).parent / 'out'
    out.mkdir(exist_ok=True)
    count = ud_count   # the manifest's entry values: rect, root constant, base vertex and instance, the table words
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'join_phi_vs', 'vs_6_0', directory)

        def run_case(name, source, selector):
            ps = compile_hlsl(dxc, 'dxc', source, name, 'ps_6_0', directory)
            expected = directory / (name + '.bin')
            expected.write_bytes(pixel_reference(selector))
            entry = [0] * count
            for index, offset in descwords.items():
                entry[index] = VALUES[offset]
            entry[ROOT_INDEX[0]] = selector
            run = subprocess.run([runner, adapter, '--startup-pixel-buffer-words', str(vs), str(ps), '4', '4',
                                  str(expected), ','.join('%08x' % word for word in entry), '00000000'],
                                 capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                sys.exit(77)
            return run

        for selector in (0, 1):
            run = run_case('join_phi_%d' % selector, hlsl, selector)
            assert run.returncode == 0 and 'all 16 pixels matched' in run.stdout, run.stdout + run.stderr
        # Mutant: the two arms assign each other's words, so both selectors read the wrong arm.
        swapped = hlsl
        for name in ('s13', 's14'):
            parts = [m for m in re.finditer(r'%s = (ud\d+);' % name, swapped)]
            assert len(parts) == 2, swapped
            a, b = parts[0].group(1), parts[1].group(1)
            swapped = swapped.replace('%s = %s;' % (name, a), '%s = @@;' % name)
            swapped = swapped.replace('%s = %s;' % (name, b), '%s = %s;' % (name, a))
            swapped = swapped.replace('%s = @@;' % name, '%s = %s;' % (name, b))
        for selector in (0, 1):
            run = run_case('join_phi_mutant_%d' % selector, swapped, selector)
            assert run.returncode != 0 and 'pixel mismatch (' in run.stderr and '16 of 16 pixels differ' in run.stderr, \
                run.stdout + run.stderr
    print('join phi: WARP readback matches both arms; swapped edge assignments differ at every pixel')


ROOT_INDEX = [0]


def main(argv):
    if len(argv) == 3 and argv[1] == 'binary':
        built = words(argv[2], SOURCE)
        assert built == PS_WORDS, built.hex()
        assert words(argv[2], UNKNOWN_SOURCE) == UNKNOWN_WORDS
        print('join phi fixture words match llvm-mc')
        return 0
    hlsl, descwords = controls(PS_WORDS, UNKNOWN_WORDS)
    pipe = translate(PS_WORDS)
    ROOT_INDEX[0] = next(int(m.group(1)) for m in re.finditer(r'^udsrc (\d+) rootconst 1 0$', pipe.manifest(), re.M))
    if len(argv) == 5 and argv[1] == 'draw':
        draw(argv[2], argv[3], argv[4], hlsl, descwords, int(re.search(r'^ud (\d+)$', pipe.manifest(), re.M).group(1)))
    elif len(argv) != 1:
        raise SystemExit(__doc__)
    return 0


PS_WORDS = bytes.fromhex('098006bf040084bf040308f4100000fa7fc08cbf030082bf040308f4300000fa7fc08cbf0d0c047e0e0c067e000d087e090c0a7e0f1800f802030405000081bf')
UNKNOWN_WORDS = bytes.fromhex('098006bf040084bf040308f4100000fa7fc08cbf030082bf0a0308f4300000fa7fc08cbf0d0c047e0e0c067e000d087e090c0a7e0f1800f802030405000081bf')

if __name__ == '__main__':
    sys.exit(main(sys.argv))
