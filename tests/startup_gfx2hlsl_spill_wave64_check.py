"""SGPR spills to VGPR lanes 32..63 (wave64) and in vertex stages, tools/startup/gfx2hlsl.py, read back from a real
D3D12 draw.

  (no arguments)                       translation controls and refusals, pure Python
  binary <llvm-mc>                     assemble the fixture sources with LLVM (gfx1013) and compare with the words here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated stages with DXC (and FXC when given), draw and compare
                                       every exported word with the slot reference below

`v_writelane_b32 vN, sM, k` writes SGPR sM into lane k of vN and `v_readlane_b32 sD, vN, k` reads lane k back (AMD
RDNA 2 ISA reference, VOP3 lane operations). Both ignore EXEC and touch only lane k, so with a constant k and a VGPR
no other instruction names, vN is a table of scalar slots (tests/startup_gfx2hlsl_sgpr_spill_check.py has the
lanes 0..31 pixel case). A wave64 program spills into lanes 0..63, and a vertex stage spills the same way. The
translator does not know the wave size: a wave32 program takes the lane select modulo 32, so lanes k and k + 32 of one
VGPR are one slot there and two in wave64, and that pair is refused; lane 64 and above stay refused.

Fixtures:
  PS  pixel stage, slots v10[40], v10[63], v11[32], v12[33]; the SGPRs are clobbered, reloaded (v10[40] after a
      second write) and ORed into the packed screen position, 16 x 16 pixels compared word by word.
  VS  vertex stage, slots v40[0], v40[48], v41[0]; reloads (v40[48] after a second write) go to param0, which the
      flat pixel stage FLATPS exports; one pixel compared.
The reference is the last value written to (VGPR, lane), computed here by replaying the lane writes on a dictionary.
Mutations of the translated text (lane 63 read as lane 40, the overwrite ignored, another VGPR's slot) must fail.
"""
import os
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
from startup_flat_param_check import assemble, draw as draw_flat  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

WIDTH = HEIGHT = 16
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
failures = []

PS_TEXT = '''s_mov_b32 s0, 0x40200000
s_mov_b32 s1, 0x40400000
s_mov_b32 s2, 4.0
v_writelane_b32 v10, s0, 40
v_writelane_b32 v10, s1, 63
v_writelane_b32 v11, s2, 32
s_or_b32 s8, s0, s1
v_writelane_b32 v12, s8, 33
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s8, 0
v_readlane_b32 s3, v10, 40
v_readlane_b32 s4, v10, 63
v_readlane_b32 s5, v11, 32
v_writelane_b32 v10, s5, 40
v_readlane_b32 s6, v10, 40
v_readlane_b32 s9, v12, 33
v_or_b32_e32 v1, s3, v0
v_or_b32_e32 v2, s4, v0
v_or_b32_e32 v3, s6, v0
v_or_b32_e32 v4, s9, v0
exp mrt0 v1, v2, v3, v4 done vm
s_endpgm
'''
PS_WORDS = bytes.fromhex(
    'ff0380be00002040ff0381be00004040f60382be0a0061d7005001000a0061d7017e01000b0061d702400100000108880c0061d708420100'
    '800380be800381be800382be800388be030060d70a510100040060d70a7f0100050060d70b4101000a0061d705500100060060d70a510100'
    '090060d70c430100030002380400043806000638090008380f1800f801020304000081bf')


def _vop3(op, vdst, src0, lane):
    return struct.pack('<2I', 0xd7000000 | op << 16 | vdst, src0 | (128 + lane) << 9)


# The same program with the derived word spilled to v10 lane 8 (v_writelane_b32 v10, s8, 8 / v_readlane_b32 s9, v10, 8)
# instead of v12 lane 33: v10 then holds lanes 8 and 40, one slot in wave32 and two in wave64.
PAIR_PS_TEXT = (PS_TEXT.replace('v_writelane_b32 v12, s8, 33', 'v_writelane_b32 v10, s8, 8')
                .replace('v_readlane_b32 s9, v12, 33', 'v_readlane_b32 s9, v10, 8'))
PAIR_PS_WORDS = (PS_WORDS.replace(_vop3(0x361, 12, 8, 33), _vop3(0x361, 10, 8, 8))
                 .replace(_vop3(0x360, 9, 256 + 12, 33), _vop3(0x360, 9, 256 + 10, 8)))
OPT_IN = 'X2W_SPILL_WAVE64_PAIR'
VS_TEXT = '''v_mov_b32_e32 v3, 0x40400000
v_mov_b32_e32 v10, 0xc0400000
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e32 v1, -1.0, v3, vcc_lo
v_cmp_eq_u32_e32 vcc_lo, 2, v0
v_cndmask_b32_e32 v2, 1.0, v10, vcc_lo
v_mov_b32_e32 v4, 1.0
v_mov_b32_e32 v5, 0
s_mov_b32 s20, 0x12345678
s_mov_b32 s21, 0x9abcdef0
v_writelane_b32 v40, s20, 0
v_writelane_b32 v40, s21, 48
v_writelane_b32 v41, s21, 0
s_mov_b32 s20, 0
s_mov_b32 s21, 0
v_readlane_b32 s22, v40, 0
v_readlane_b32 s23, v40, 48
v_readlane_b32 s24, v41, 0
v_writelane_b32 v40, s22, 48
v_readlane_b32 s25, v40, 48
v_mov_b32_e32 v11, s22
v_mov_b32_e32 v12, s23
v_mov_b32_e32 v13, s24
v_mov_b32_e32 v14, s25
exp pos0 v1, v2, v5, v4 done
exp param0 v11, v12, v13, v14
s_endpgm
'''
VS_FIXTURE = bytes.fromhex(
    'ff02067e00004040ff02147e000040c08100847df30602028200847df2140402f202087e80020a7eff0394be78563412ff0395bef0debc9a'
    '280061d714000100280061d715600100290061d715000100800394be800395be160060d728010100170060d728610100180060d729010100'
    '280061d716600100190060d7286101001602167e1702187e18021a7e19021c7ecf0800f8010205040f0200f80b0c0d0e000081bf')
FLAT_PS_TEXT = '''s_mov_b32 m0, s2
v_interp_mov_f32_e32 v0, p0, attr0.x
v_interp_mov_f32_e32 v1, p0, attr0.y
v_interp_mov_f32_e32 v2, p0, attr0.z
v_interp_mov_f32_e32 v3, p0, attr0.w
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
'''
FLAT_PS_WORDS = bytes.fromhex('0203fcbe020002c8020106c802020ac802030ec80f1800f800010203000081bf')


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + str(detail)[:1500]))
    if not ok:
        failures.append(label)


# ---- reference: replay the lane writes and reads on a dictionary ------------------------------------------------------
def replay(text, outputs):
    lanes, scalars = {}, {}
    for line in text.splitlines():
        op, _, rest = line.partition(' ')
        args = [a.strip() for a in rest.split(',')]
        if op == 's_mov_b32':
            scalars[args[0]] = 0x40800000 if args[1] == '4.0' else int(args[1], 0)
        elif op == 's_or_b32':
            scalars[args[0]] = scalars[args[1]] | scalars[args[2]]
        elif op == 'v_writelane_b32':
            lanes[(args[0], int(args[2]))] = scalars[args[1]]
        elif op == 'v_readlane_b32':
            scalars[args[0]] = lanes[(args[1], int(args[2]))]
    return [scalars[r] for r in outputs]


PS_SLOTS = replay(PS_TEXT, ('s3', 's4', 's6', 's9'))
assert replay(PAIR_PS_TEXT, ('s3', 's4', 's6', 's9')) == PS_SLOTS
VS_SLOTS = replay(VS_TEXT, ('s22', 's23', 's24', 's25'))


def ps_image():
    return [(x | y << 16) | value for y in range(HEIGHT) for x in range(WIDTH) for value in PS_SLOTS]


def ps_text():
    return G.Pipeline(VS_WORDS, PS_WORDS, root_blob(), ps_hw_inputs=PACKED).hlsl_ps()


def pair_text(opt_in):
    """The translated pair program; opt_in sets X2W_SPILL_WAVE64_PAIR for this call only."""
    saved = os.environ.pop(OPT_IN, None)
    if opt_in:
        os.environ[OPT_IN] = '1'
    try:
        return G.Pipeline(VS_WORDS, PAIR_PS_WORDS, root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
    finally:
        os.environ.pop(OPT_IN, None)
        if saved is not None:
            os.environ[OPT_IN] = saved


def vs_pipeline():
    return G.Pipeline(VS_FIXTURE, FLAT_PS_WORDS, root_blob())


def mutate(text, old, new):
    mutated = re.sub(old, new, text)
    assert mutated != text, old
    return mutated


# Each wrong lowering, written on the folded text (a constant slot folds into the export).
PS_MUTATIONS = {
    'lane 63 reads lane 40 of the same VGPR': (r'v2 = 0x40400000u', 'v2 = 0x40200000u'),
    'the overwritten slot keeps its first value': (r'sp10_40 = s5;', 'sp10_40 = sp10_40;'),
    'v11 lane 32 reads v10 lane 40': (r'= sp11_32;', '= sp10_40;'),
}
# The pair program's derived word reads the slot of lane 8; reading lane 40 instead must fail the readback.
PAIR_MUTATION = (r's9 = sp10_8;', 's9 = sp10_40;')
VS_MUTATIONS = {
    'lane 48 reads lane 0 of the same VGPR': (r'v12 = 0x9abcdef0u', 'v12 = 0x12345678u'),
    'the overwritten slot keeps its first value': (r'v14 = 0x12345678u', 'v14 = 0x9abcdef0u'),
}


# ---- controls ---------------------------------------------------------------------------------------------------------
def refused(program, kind='ps'):
    stage = G.Stage.__new__(G.Stage)
    stage.kind, stage.lines, stage.sym = kind, [], {}
    stage.used_s, stage.used_v, stage.uniform_v, stage.varying_s = set(), set(), set(), set()
    stage.scc_uniform, stage.exec_known, stage.has_branches, stage.blocks, stage.masks = True, True, False, [], set()
    stage.ins = [(4 * i, 1, re.sub(r'_e(32|64)$', '', a.split(' ', 1)[0]), a) for i, a in enumerate(program)]
    for _, _, name, asm in stage.ins:
        try:
            stage.one(name, asm)
        except G.Unsupported as reason:
            return str(reason)
    return None


def controls():
    check('pixel reference: v10[40], v10[63], v10[40] after the overwrite with v11[32], and the derived word',
          PS_SLOTS == [0x40200000, 0x40400000, 0x40800000, 0x40600000], [hex(v) for v in PS_SLOTS])
    check('vertex reference: v40[0], v40[48], v41[0], v40[48] after the overwrite',
          VS_SLOTS == [0x12345678, 0x9abcdef0, 0x9abcdef0, 0x12345678], [hex(v) for v in VS_SLOTS])
    for label, text, words in (('PS', PS_TEXT, PS_WORDS), ('VS', VS_TEXT, VS_FIXTURE), ('FLATPS', FLAT_PS_TEXT,
                                                                                         FLAT_PS_WORDS)):
        got = [re.sub(r'_e(32|64)\b', '', asm) for _o, _n, _m, asm in G.decode(words)]
        want = [re.sub(r'_e(32|64)\b', '', line) for line in text.splitlines()]
        check('%s fixture decodes to its source' % label, got == want, '%s\n%s' % (got, want))
    try:
        text = ps_text()
        check('wave64 pixel slots translate to one scalar per (VGPR, lane)',
              all(re.search(r'\b%s = 0' % s, text) for s in ('sp10_40', 'sp10_63', 'sp11_32', 'sp12_33')), text)
        for label, (old, new) in PS_MUTATIONS.items():
            check('PS mutation "%s" changes the text' % label, mutate(text, old, new) != text)
    except G.Unsupported as reason:
        check('wave64 pixel slots translate', False, str(reason))
    try:
        text = vs_pipeline().hlsl_vs()
        check('vertex stage slots translate', all(re.search(r'\b%s = 0' % s, text) for s in ('sp40_0', 'sp40_48', 'sp41_0')),
              text)
        for label, (old, new) in VS_MUTATIONS.items():
            check('VS mutation "%s" changes the text' % label, mutate(text, old, new) != text)
    except G.Unsupported as reason:
        check('vertex stage slots translate', False, str(reason))
    got = [re.sub(r'_e(32|64)\b', '', asm) for _o, _n, _m, asm in G.decode(PAIR_PS_WORDS)]
    check('pair fixture decodes to its source', got == [re.sub(r'_e(32|64)\b', '', l) for l in PAIR_PS_TEXT.splitlines()], got)
    try:
        pair_text(False)
        check('lanes 8 and 40 of v10 are refused without the opt-in', False, 'translated')
    except G.Unsupported as reason:
        check('lanes 8 and 40 of v10 are refused without the opt-in (%s)' % reason,
              'one slot in wave32 and two in wave64' in str(reason) and OPT_IN in str(reason), reason)
    try:
        text = pair_text(True)
        check('with %s=1 lanes 8 and 40 of v10 are two slots' % OPT_IN,
              all(re.search(r'\b%s = 0' % v, text) for v in ('sp10_8', 'sp10_40')) and 's9 = sp10_8;' in text, text)
        check('the pair mutation changes the text', mutate(text, *PAIR_MUTATION) != text)
    except G.Unsupported as reason:
        check('with the opt-in the pair translates', False, str(reason))
    for label, program, kind in (
            ('lane 64', ['v_writelane_b32 v10, s0, 64'], 'ps'),
            ('lanes 8 and 40 of one VGPR', ['v_writelane_b32 v10, s0, 8', 'v_writelane_b32 v10, s1, 40'], 'ps'),
            ('lanes 63 and 31 of one VGPR in a vertex stage',
             ['v_writelane_b32 v10, s0, 63', 'v_readlane_b32 s1, v10, 31'], 'vs'),
            ('a lane of a stage outside pixel, vertex, mesh and compute', ['v_writelane_b32 v10, s0, 40'], 'gs'),
            ('a lane chosen by an SGPR', ['v_writelane_b32 v10, s0, s2'], 'vs')):
        reason = refused(program, kind)
        check('refused: %s (%s)' % (label, reason), reason is not None)
    apart = ['v_writelane_b32 v10, s0, 8', 'v_writelane_b32 v11, s1, 40', 'v_readlane_b32 s2, v11, 40']
    check('lanes 8 and 40 of two different VGPRs are accepted', refused(apart) is None, str(refused(apart)))


# ---- draw -------------------------------------------------------------------------------------------------------------
def draw_pixels(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    ps_source = ps_text()
    pipe = vs_pipeline()
    vs_source, flat_source = pipe.hlsl_vs(), pipe.hlsl_ps()
    expected_vs = ','.join('%08x' % v for v in VS_SLOTS)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        reference = directory / 'reference.bin'
        reference.write_bytes(struct.pack('<%dI' % (WIDTH * HEIGHT * 4), *ps_image()))
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_' + kind, vs_profile, directory)
            ps = compile_hlsl(compiler, kind, ps_source, 'ps_' + kind, ps_profile, directory)
            run = draw_pixels(renderer, adapter, vs, ps, reference)
            check('%s %s PS: all %d pixels equal the slot reference' % (adapter, kind, WIDTH * HEIGHT),
                  run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                  run.stdout + run.stderr)
            for label, (old, new) in PS_MUTATIONS.items():
                bad = compile_hlsl(compiler, kind, mutate(ps_source, old, new), 'bad_ps_%s_%d' % (kind, len(old)),
                                   ps_profile, directory)
                run = draw_pixels(renderer, adapter, vs, bad, reference)
                check('%s %s PS mutation "%s" fails' % (adapter, kind, label),
                      run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)
            pair_source = pair_text(True)
            pair = compile_hlsl(compiler, kind, pair_source, 'ps_pair_' + kind, ps_profile, directory)
            run = draw_pixels(renderer, adapter, vs, pair, reference)
            check('%s %s pair PS: all %d pixels equal the slot reference' % (adapter, kind, WIDTH * HEIGHT),
                  run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout, run.stdout + run.stderr)
            bad = compile_hlsl(compiler, kind, mutate(pair_source, *PAIR_MUTATION), 'bad_ps_pair_' + kind, ps_profile, directory)
            run = draw_pixels(renderer, adapter, vs, bad, reference)
            check('%s %s pair PS mutation (lane 8 reads lane 40) fails' % (adapter, kind),
                  run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)
            vs2 = compile_hlsl(compiler, kind, vs_source, 'vs_spill_' + kind, vs_profile, directory)
            flat = compile_hlsl(compiler, kind, flat_source, 'ps_flat_' + kind, ps_profile, directory)
            run = draw_flat(renderer, adapter, vs2, flat, expected_vs)
            check('%s %s VS: exported param0 equals the slot reference %s' % (adapter, kind, expected_vs),
                  run.returncode == 0, run.stdout + run.stderr)
            for label, (old, new) in VS_MUTATIONS.items():
                bad = compile_hlsl(compiler, kind, mutate(vs_source, old, new), 'bad_vs_%s_%d' % (kind, len(old)),
                                   vs_profile, directory)
                run = draw_flat(renderer, adapter, bad, flat, expected_vs)
                check('%s %s VS mutation "%s" fails' % (adapter, kind, label), run.returncode not in (0, 77),
                      run.stdout + run.stderr)


def check_binary(llvm_mc):
    for label, text, words in (('PS', PS_TEXT, PS_WORDS), ('VS', VS_TEXT, VS_FIXTURE),
                               ('FLATPS', FLAT_PS_TEXT, FLAT_PS_WORDS)):
        check('llvm-mc assembles the %s fixture to the same words' % label, assemble(llvm_mc, text) == words)


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
