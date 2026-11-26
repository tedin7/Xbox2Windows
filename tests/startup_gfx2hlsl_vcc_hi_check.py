"""wave32 vcc_hi as an ordinary scalar register in tools/startup/gfx2hlsl.py, read back from a real D3D12 draw.

  (no arguments)                       translation controls and refusals, pure Python
  binary <llvm-mc>                     assemble the fixture sources with LLVM (gfx1013) and compare with the words here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pixel stage (DXC, and FXC when given), draw 16 x 16
                                       pixels and compare every word with the CPU reference below

In wave32 VCC is vcc_lo alone, and compilers allocate vcc_hi (SGPR 107) like any other scalar register: a compare or
carry-out destination, a v_cndmask mask, a scalar operand. The decoder prints wave32 syntax, so the text does not
give the wave size; the translator renames vcc_hi to an SGPR the program never names only when that is exact in both
wave sizes: no instruction names the 64-bit pair `vcc`, and either a VALU compare/carry writes vcc_hi (or an odd
SGPR) or a VOP3 mask source reads one (a misaligned pair in wave64, so the program is wave32), or no VALU instruction
and no vcc branch uses vcc_lo as a lane mask (in wave64 those also write or read vcc_hi for lanes 32..63). Otherwise vcc_hi is refused by name, as before, and a
scalar write into vcc_hi or exec_hi is now a refusal instead of a TypeError.

The draw fixture PS (packed screen position v0 = x | y << 16):
  v_cmp_gt_u32_e64 vcc_hi, v0, 0x50005         ; v_cndmask_b32_e64 v2, 1.0, 2.0, vcc_hi -> 2.0 where v0 > 0x50005
  v_add_co_u32 v3, vcc_hi, v0, 0xfff7fff8      ; v_add_co_ci_u32_e64 v4, s3, v0, 2.0, vcc_hi -> + the carry
  s_mov_b32 vcc_hi, 0x1234 ; s_add_u32 s0, vcc_hi, 16 ; v5 = s0 + v0
The reference below computes the same words with Python integers. The selected words are 1.0 and 2.0, not the
integers 1 and 2, and the carry rides on v0 + 0x40000000: with FXC a word that can only be 0 or 1 came back as
0 on WARP where it was 1 (DXC kept it), a denormal export artifact, not a translation error. Mutations (the compare bit not read, the carry
lost, vcc_hi read as zero) must fail the draw.
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

MASK = 0xffffffff
WIDTH = HEIGHT = 16
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
failures = []

# name -> (source, words assembled by llvm-mc 19 for gfx1013, expected refusal or None)
FIXTURES = {
    'PS': ('''v_cmp_gt_u32_e64 vcc_hi, v0, 0x50005
v_cndmask_b32_e64 v2, 1.0, 2.0, vcc_hi
v_add_co_u32 v3, vcc_hi, v0, 0xfff7fff8
v_add_co_ci_u32_e64 v4, s3, v0, 0x40000000, vcc_hi
s_mov_b32 vcc_hi, 0x1234
s_add_u32 s0, vcc_hi, 16
v_add_nc_u32_e32 v5, s0, v0
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
''', '6b00c4d400ff010005000500020001d5f2e8ad01036b0fd700ff0100f8fff7ff040328d500e9ad01ff03ebbe341200006b900080'
         '00000a4a0f1800f802030405000081bf', None),
    # Only scalar writes of vcc_hi and no VALU VCC write: exact in either wave size.
    'SCALAR_ONLY': ('''s_mov_b32 vcc_hi, 0x1234
s_lshl_b32 vcc_hi, vcc_hi, 4
s_add_u32 s0, vcc_hi, 16
v_add_nc_u32_e32 v5, s0, v0
exp mrt0 v5, v0, v0, v0 done vm
s_endpgm
''', 'ff03ebbe341200006b846b8f6b90008000000a4a0f1800f805000000000081bf', None),
    # A VOPC writes vcc_lo and nothing proves wave32: in wave64 the compare would also write vcc_hi.
    'SCALAR_AND_VOPC': ('''s_mov_b32 vcc_hi, 0x1234
s_lshl_b32 vcc_hi, vcc_hi, 4
s_add_u32 s0, vcc_hi, 16
v_cmp_gt_u32_e32 vcc_lo, 0x50005, v0
v_cndmask_b32_e64 v2, 1, 2, vcc_lo
v_add_nc_u32_e32 v5, s0, v0
exp mrt0 v2, v5, v0, v0 done vm
s_endpgm
''', 'ff03ebbe341200006b846b8f6b900080ff00887d05000500020001d58104a90100000a4a0f1800f802050000000081bf',
        's_mov_b32 destination vcc_hi'),
    # A VALU mask read of vcc_lo and nothing proves wave32: in wave64 lanes 32..63 would read vcc_hi as their mask
    # bit, which the renamed scalar register does not model (found by an independent review).
    'MASK_READ': ('''s_mov_b32 vcc_lo, 0
s_mov_b32 vcc_hi, -1
v_mov_b32_e32 v2, 2.0
v_cndmask_b32_e32 v1, 1.0, v2, vcc_lo
exp mrt0 v1, v1, v1, v1 done vm
s_endpgm
''', '8003eabec103ebbef402047ef20402020f1800f801010101000081bf', 's_mov_b32 destination vcc_hi'),
    # A whole scalar word is not a lane mask: the lowering would test bit 0 of 0x55555555 for every lane.
    'SCALAR_MASK': ('''s_mov_b32 vcc_hi, 0x55555555
v_cndmask_b32_e64 v2, 1.0, 2.0, vcc_hi
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''', 'ff03ebbe55555555020001d5f2e8ad010f1800f802020202000081bf', 's_mov_b32 destination vcc_hi'),
    # The 64-bit pair `vcc` names vcc_hi as its high word.
    'PAIR': ('''v_cmp_gt_u32_e64 vcc_hi, v0, 0x50005
s_and_b64 s[2:3], vcc, exec
v_cndmask_b32_e64 v2, 1, 2, vcc_hi
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''', '6b00c4d400ff0100050005006a7e8287020001d58104ad010f1800f802020202000081bf', 'v_cmp_gt_u32 into vcc_hi'),
    # exec_hi is a mask half in every wave size: a scalar write is refused, not a crash.
    'EXEC_HI': ('''s_mov_b32 exec_hi, 0
v_mov_b32_e32 v2, 1
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''', '8003ffbe8102047e0f1800f802020202000081bf', 's_mov_b32 destination exec_hi'),
}


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + str(detail)[:1500]))
    if not ok:
        failures.append(label)


def translate(name):
    try:
        return G.Pipeline(VS_WORDS, bytes.fromhex(FIXTURES[name][1]), root_blob(), ps_hw_inputs=PACKED).hlsl_ps(), None
    except G.Unsupported as reason:
        return None, str(reason)


def image():
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            v0 = x | y << 16
            total = v0 + 0xfff7fff8
            out += [0x40000000 if v0 > 0x50005 else 0x3f800000, total & MASK, (v0 + 0x40000000 + (total >> 32)) & MASK,
                    (0x1244 + v0) & MASK]
    return out


def sub(text, pattern, replacement):
    mutated, count = re.subn(pattern, replacement, text)
    assert count == 1, pattern
    return mutated


# The renamed register is whichever SGPR the program never names; the patterns accept any number.
MUTATIONS = {
    'the compare bit is not read': (r'v2 = \(s(\d+) & 1\)', 'v2 = (0u & 1)'),
    'the carry-out is lost': (r'a = a \+ \(s(\d+) & 1u\);', 'a = a;'),
    'vcc_hi reads as zero': (r's0 = 0x00001234u \+', 's0 = 0x00000000u +'),
}


def controls():
    for name, (_source, _words, refusal) in FIXTURES.items():
        try:
            text, reason = translate(name)
        except Exception as error:  # noqa: BLE001 - a crash is the failure this guards
            check('%s translates or refuses without a crash' % name, False, '%s: %s' % (type(error).__name__, error))
            continue
        if refusal is None:
            check('%s translates' % name, text is not None and 'vcc_hi' not in text, str(reason))
        else:
            check('%s refused: %s' % (name, refusal), text is None and reason == refusal, str(reason))
    # A wide operand covering SGPR 107 (s_load_dwordx4 s[104:107]) writes vcc_hi: no free alias exists, so it stays refused.
    wide = [(0, 0, 's_mov_b32', 's_mov_b32 vcc_hi, 0x1234'),
            (4, 1, 's_load_dwordx4', 's_load_dwordx4 s[104:107], s[0:1], 0x0'),
            (12, 2, 'v_mov_b32_e32', 'v_mov_b32_e32 v2, vcc_hi')]
    check('a load into s[104:107] keeps vcc_hi refused', G.wave32_vcc_hi(wide) is wide)
    text, _ = translate('PS')
    if text is not None:
        check('PS keeps the compare and carry in the renamed register',
              re.search(r's(\d+) = \(\(v0 > 0x00050005u\) \? 1u : 0u\);', text) is not None, text)
        for label, (pattern, replacement) in MUTATIONS.items():
            check('PS mutation "%s" changes the text' % label, sub(text, pattern, replacement) != text)
    reference = image()
    check('reference covers both compare outcomes and both carries',
          {reference[i] for i in range(0, len(reference), 4)} == {0x3f800000, 0x40000000}
          and {(reference[i] - 0x40000000 - (i // 4) % WIDTH - ((i // 4 // WIDTH) << 16)) & MASK
               for i in range(2, len(reference), 4)} == {0, 1})


def draw(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    text, reason = translate('PS')
    if text is None:
        check('PS translates for the draw', False, reason)
        return
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        reference = directory / 'reference.bin'
        reference.write_bytes(struct.pack('<%dI' % (WIDTH * HEIGHT * 4), *image()))
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_' + kind, vs_profile, directory)
            ps = compile_hlsl(compiler, kind, text, 'ps_' + kind, ps_profile, directory)
            run = draw(renderer, adapter, vs, ps, reference)
            check('%s %s PS: all %d pixels match the reference' % (adapter, kind, WIDTH * HEIGHT),
                  run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                  run.stdout + run.stderr)
            for n, (label, (pattern, replacement)) in enumerate(MUTATIONS.items()):
                bad = compile_hlsl(compiler, kind, sub(text, pattern, replacement), 'bad_%s_%d' % (kind, n), ps_profile,
                                   directory)
                run = draw(renderer, adapter, vs, bad, reference)
                check('%s %s PS mutation "%s" fails' % (adapter, kind, label),
                      run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)


def check_binary(llvm_mc):
    for name, (source, words, _refusal) in FIXTURES.items():
        check('llvm-mc assembles the %s fixture to the same words' % name,
              assemble(llvm_mc, source) == bytes.fromhex(words))


def main():
    if len(sys.argv) == 1:
        controls()
    else:
        mode, *args = sys.argv[1:]
        if mode == 'binary':
            check_binary(*args)
        elif mode == 'draw':
            controls()
            check_draw(*args)
        else:
            raise SystemExit(__doc__)
    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
