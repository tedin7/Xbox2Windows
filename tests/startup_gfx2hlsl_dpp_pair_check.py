"""DPP16 quad pair broadcasts in tools/startup/gfx2hlsl.py (#2125 item C), read back from a real D3D12 draw.

  (no arguments)                       translation controls and refusals, pure Python
  binary <llvm-mc>                     assemble the fixture sources with LLVM (gfx1013) and compare with the words here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pixel stage with DXC ps_6_0 (the lowering uses the shader
                                       model 6.0 quad reads, so FXC is not used), draw 16 x 16 pixels and compare every
                                       pixel with a CPU reference. The optional fxc argument is accepted and ignored.

Semantics (RDNA 2 ISA, DPP16 and the quad_perm control): in a wave the lanes of a pixel quad are 0 1 / 2 3, that is
the screen pixels (x, y) = (0, 0) (1, 0) / (0, 1) (1, 1) relative to the even corner, and lane i reads SRC0 of lane
quad_perm[i] of its own quad. The four pair broadcasts of item C move one lane of a horizontal or vertical pair into
the other lane of that pair:
  [1,1,3,3]  lanes 0 and 2 read lanes 1 and 3      (the odd column fills the even one)
  [0,0,2,2]  lanes 1 and 3 read lanes 0 and 2      (the even column fills the odd one)
  [2,3,2,3]  lanes 0 and 1 read lanes 2 and 3      (the odd row fills the even one)
  [0,1,0,1]  lanes 2 and 3 read lanes 0 and 1      (the even row fills the odd one)
The other lane of every pair keeps its own word. The reference below is written from that table with plain integers
and does not use the lowering.

Helper lanes: the instruction reads lanes the pixel does not cover. In a pixel shader the four lanes of a quad run
together and the hardware keeps the uncovered ones as helpers, which is what QuadReadLaneAt relies on as well. The
translator accepts the form only in straight-line pixel code after s_wqm exec, exec on the entry mask.

Why not derivatives: a first version rebuilt the other lane's word from ddx_fine / ddy_fine of each 16-bit half. WARP
agreed with the reference, but an RTX adapter returned other words for all four patterns, because a hardware derivative
is not required to be the exact difference of the two pixels' values. The lowering now reads the four lanes with
QuadReadLaneAt, the path the quad swaps already use, and selects the lane quad_perm names.

The draw: a pixel program multiplies the packed screen position by a constant, so every pixel carries a different
32-bit word with both halves in use, applies the DPP form to it and exports (v_mov_b32_dpp result, v_add_nc_u32_dpp
result with the same register as both sources, the pixel's own word, the packed position). The second form shows that
the second source keeps the pixel's own word and does not see the permuted one. Mutations of the translated text
(opposite sign, opposite lanes, the other axis) must each fail the same draw.
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
MULTIPLIER = 0x9b5d
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
PERMS = ((1, 1, 3, 3), (0, 0, 2, 2), (2, 3, 2, 3), (0, 1, 0, 1))
CONTROLS = 'row_mask:0xf bank_mask:0xf'
failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + detail))
    if not ok:
        failures.append(label)


def spelling(perm):
    return 'quad_perm:[%d,%d,%d,%d]' % perm


def control(perm):
    return sum(lane << (2 * index) for index, lane in enumerate(perm))


def vop1_dpp(op, vdst, src, ctrl, bound_ctrl=False, bank=0xf, row=0xf):
    word0 = 0x7e000000 | (vdst << 17) | (op << 9) | 250
    return struct.pack('<II', word0, src | (ctrl << 8) | (int(bound_ctrl) << 19) | (bank << 24) | (row << 28))


def vop2_dpp(op, vdst, src0, vsrc1, ctrl, bank=0xf, row=0xf):
    word0 = (op << 25) | (vdst << 17) | (vsrc1 << 9) | 250
    return struct.pack('<II', word0, src0 | (ctrl << 8) | (bank << 24) | (row << 28))


V_MOV_B32, V_ADD_NC_U32 = 1, 0x25
MUL_WORDS = struct.pack('<II', (0x0b << 25) | (1 << 17) | 255, MULTIPLIER)  # v_mul_u32_u24 v1, 0x9b5d, v0
WQM_WORDS = struct.pack('<I', 0xbefe0a7e)                                   # s_wqm_b64 exec, exec
EXPORT_WORDS = struct.pack('<III', 0xf800180f, 0x00010302, 0xbf810000)    # exp mrt0 v2, v3, v1, v0 done vm; s_endpgm


def ps_source(perm):
    return ('s_wqm_b64 exec, exec\n'
            'v_mul_u32_u24_e32 v1, 0x9b5d, v0\n'
            'v_mov_b32_dpp v2, v1 %s %s\n'
            'v_add_nc_u32_dpp v3, v1, v1 %s %s\n'
            'exp mrt0 v2, v3, v1, v0 done vm\n'
            's_endpgm\n' % (spelling(perm), CONTROLS, spelling(perm), CONTROLS))


def ps_words(perm):
    return (WQM_WORDS + MUL_WORDS + vop1_dpp(V_MOV_B32, 2, 1, control(perm)) + vop2_dpp(V_ADD_NC_U32, 3, 1, 1, control(perm))
            + EXPORT_WORDS)


def pixel_hlsl(perm, how='reference'):
    text = G.Pipeline(VS_WORDS, ps_words(perm), root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
    return text if how == 'reference' else MUTATIONS[how](text, perm)


# ---- CPU reference (from the lane table above) -----------------------------------------------------------------------
def word(x, y):
    return ((x | (y << 16)) * MULTIPLIER) & MASK


def permuted(perm, x, y):
    lane = (y & 1) * 2 + (x & 1)
    source = perm[lane]
    return word((x & ~1) | (source & 1), (y & ~1) | (source >> 1))


def image(perm):
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            moved = permuted(perm, x, y)
            words += [moved, (moved + word(x, y)) & MASK, word(x, y), x | (y << 16)]
    return words


# ---- mutations of the translated text --------------------------------------------------------------------------------
CHAIN = re.compile(r'x2w_lane == 0u \? x2w_q(\d) : x2w_lane == 1u \? x2w_q(\d) : x2w_lane == 2u \? x2w_q(\d) : x2w_q(\d);')


def with_sources(text, sources):
    assert CHAIN.search(text)
    return CHAIN.sub('x2w_lane == 0u ? x2w_q%d : x2w_lane == 1u ? x2w_q%d : x2w_lane == 2u ? x2w_q%d : x2w_q%d;'
                     % tuple(sources), text)


def opposite_lanes(text, perm):
    return with_sources(text, [lane ^ 3 for lane in perm])


def own_lane(text, perm):
    return with_sources(text, (0, 1, 2, 3))


def transposed_lane(text, perm):
    old = '((uint(i.pos.y) & 1u) << 1u) | (uint(i.pos.x) & 1u)'
    assert old in text
    return text.replace(old, '((uint(i.pos.x) & 1u) << 1u) | (uint(i.pos.y) & 1u)')


def read_other_source(text, perm):
    """The second source of v_add_nc_u32_dpp reads the permuted word instead of the pixel's own."""
    old = 'v3 = v256 + v1;'
    assert old in text
    return text.replace(old, 'v3 = v256 + v256;')


MUTATIONS = {'opposite lanes': opposite_lanes, 'own lane': own_lane, 'transposed lane index': transposed_lane,
             'second source permuted': read_other_source}


def simulate(text):
    """Run the translated broadcast statement on every pixel of the image, the way a quad of pixels would: every lane
    of the aligned 2 x 2 block holds its own word, QuadReadLaneAt(word, n) returns the word of lane n of the block."""
    block = re.search(r'\{ uint x2w_q0 .*? \}', text).group(0)
    sources = [int(n) for n in CHAIN.search(block).groups()]
    a, b = re.search(r'x2w_lane = \(\(uint\(i\.pos\.([xy])\) & 1u\) << 1u\) \| \(uint\(i\.pos\.([xy])\) & 1u\)',
                     block).groups()
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            at = {'x': x, 'y': y}
            lane = ((at[a] & 1) << 1) | (at[b] & 1)
            source = sources[lane]
            words.append(word((x & ~1) | (source & 1), (y & ~1) | (source >> 1)))
    return words


# ---- translation controls --------------------------------------------------------------------------------------------
class Probe(G.Stage):
    """A Stage with no pipeline: enough state for decoded instructions to be translated."""

    def __init__(self, kind='ps', exec_known=True, has_branches=False, wqm_exec=True):
        self.lines, self.sym, self.blocks = [], {}, []
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.kind, self.exec_known, self.has_branches = kind, exec_known, has_branches
        self.wqm_exec = wqm_exec


def refused(asm, **kwargs):
    stage = Probe(**kwargs)
    name = asm.split(' ', 1)[0]
    try:
        stage.one(name, asm)
    except G.Unsupported as reason:
        return str(reason)
    return None


def translate(words):
    try:
        return G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=PACKED).hlsl_ps(), None
    except G.Unsupported as reason:
        return None, str(reason)


def pipeline_controls():
    """Whole-quad proof and used-register bookkeeping, seen through a whole pixel program."""
    ctrl = control((1, 1, 3, 3))
    dpp = vop1_dpp(V_MOV_B32, 2, 1, ctrl)
    text, reason = translate(MUL_WORDS + dpp + EXPORT_WORDS)
    check('refuses DPP without s_wqm exec, exec (entry mask may leave a neighbour inactive)',
          text is None and reason is not None and 'whole-quad' in reason, str(reason))
    save, restore = struct.pack('<I', 0xbe84047e), struct.pack('<I', 0xbefe0404)  # s_mov_b64 s[4:5], exec / exec, s[4:5]
    text, reason = translate(save + WQM_WORDS + restore + MUL_WORDS + dpp + EXPORT_WORDS)
    check('refuses DPP after an exact entry mask is restored over the WQM mask',
          text is None and reason is not None and 'whole-quad' in reason, str(reason))
    text, reason = translate(save + WQM_WORDS + MUL_WORDS + dpp + EXPORT_WORDS)
    check('accepts DPP after s_wqm with a saved copy still unrestored', text is not None, str(reason))
    # DPP is the only reader of v0: v0 must still be declared and initialized from the packed position input.
    only = struct.pack('<III', 0xf800180f, 0x02020202, 0xbf810000)  # exp mrt0 v2, v2, v2, v2 done vm; s_endpgm
    text, reason = translate(WQM_WORDS + vop1_dpp(V_MOV_B32, 2, 0, ctrl) + only)
    check('a DPP-only source VGPR is declared', text is not None and 'v0' in text.split('x2w_q0', 1)[0], str(reason))
    check('a DPP-only source VGPR is initialized before the lowering',
          text is not None and 'uint v0' in text and text.index('uint v0') < text.index('x2w_q0'), str(text)[:300])


def controls():
    for perm in PERMS:
        decoded = [text for _o, _s, _n, text in G.decode(ps_words(perm))]
        check('%s fixture decodes to its source' % spelling(perm), decoded == ps_source(perm).splitlines(), str(decoded))
        text = pixel_hlsl(perm)
        check('%s reads the four lanes of each of the two forms with QuadReadLaneAt, no derivative' % spelling(perm),
              text.count('QuadReadLaneAt(') == 8 and 'ddx' not in text and 'ddy' not in text, text)
        reference = image(perm)
        check('%s translated text moves the right lane (CPU quad model)' % spelling(perm),
              simulate(text) == reference[0::4])
        for name, mutate in MUTATIONS.items():
            if name != 'second source permuted':
                check('%s %s mutation differs on the CPU quad model' % (spelling(perm), name),
                      simulate(mutate(text, perm)) != reference[0::4])
        identity = [value for y in range(HEIGHT) for x in range(WIDTH) for value in
                    (word(x, y), (2 * word(x, y)) & MASK, word(x, y), x | (y << 16))]
        check('%s reference differs from a lane-local lowering' % spelling(perm), reference != identity)
        for name, mutate in MUTATIONS.items():
            check('%s %s mutation changes the text' % (spelling(perm), name), mutate(text, perm) != text)
    accepted = 'v_mov_b32_dpp v2, v1 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf'
    check('accepted form translates', refused(accepted) is None, str(refused(accepted)))
    check('bound_ctrl does not change the lowering', refused(accepted + ' bound_ctrl:1') is None)
    stages = [Probe(), Probe()]
    stages[0].one('v_mov_b32_dpp', accepted + ' bound_ctrl:1 fi:1')
    stages[1].one('v_mov_b32_dpp', accepted)
    check('fi:1 does not change the lowering (no source lane of a whole quad is disabled)',
          stages[0].lines == stages[1].lines and any('QuadReadLaneAt' in line for line in stages[0].lines))
    check('quad swap also translates', refused(accepted.replace('[1,1,3,3]', '[1,0,3,2]')) is None)
    check('quad broadcast also translates', refused(accepted.replace('[1,1,3,3]', '[0,0,0,0]')) is None)
    for label, asm, kwargs in (
            ('the identity permutation stays with the lane-local forms', accepted.replace('[1,1,3,3]', '[0,1,2,3]'), {}),
            ('another pair pattern', accepted.replace('[1,1,3,3]', '[3,3,0,0]'), {}),
            ('projected exchange with integer ALU',
             'v_add_nc_u32_dpp v2, v1, v1 quad_perm:[3,3,1,1] row_mask:0xf bank_mask:0xf', {}),
            ('a row control', 'v_mov_b32_dpp v2, v1 row_shr:1 row_mask:0xf bank_mask:0xf', {}),
            ('a wave control', 'v_mov_b32_dpp v2, v1 wave_shl:1 row_mask:0xf bank_mask:0xf', {}),
            ('a mirror control', 'v_mov_b32_dpp v2, v1 row_mirror row_mask:0xf bank_mask:0xf', {}),
            ('a partial row mask', accepted.replace('row_mask:0xf', 'row_mask:0x5'), {}),
            ('a partial bank mask', accepted.replace('bank_mask:0xf', 'bank_mask:0xe'), {}),
            ('fi:1 with an entry mask not proven whole-quad', accepted + ' fi:1', {'wqm_exec': False}),
            ('fi:1 with an unknown entry mask', accepted + ' fi:1', {'exec_known': False}),
            ('fi:1 in a program with branches', accepted + ' fi:1', {'has_branches': True}),
            ('no control', 'v_mov_b32_dpp v2, v1 row_mask:0xf bank_mask:0xf', {}),
            ('a vertex stage', accepted, {'kind': 'vs'}),
            ('a program with branches', accepted, {'has_branches': True}),
            ('an unknown entry mask', accepted, {'exec_known': False}),
            ('an entry mask not proven whole-quad', accepted, {'wqm_exec': False}),
            ('a scalar source', 'v_mov_b32_dpp v2, s1 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf', {})):
        reason = refused(asm, **kwargs)
        check('refuses %s' % label, reason is not None, 'translated')
    pipeline_controls()
    wave = refused('v_mov_b32_dpp v2, v1 wave_shl:1 row_mask:0xf bank_mask:0xf')
    check('the refusal names the control', wave is not None and 'wave_shl:1' in wave, str(wave))


# ---- draw -------------------------------------------------------------------------------------------------------------
def words_file(directory, name, words):
    path = directory / ('%s.bin' % name)
    path.write_bytes(struct.pack('<%dI' % len(words), *words))
    return path


def draw(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


def tag(perm):
    return ''.join(str(lane) for lane in perm)


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')]  # the quad reads need shader model 6.0, so no FXC leg
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            for perm in PERMS:
                reference = words_file(directory, 'reference_' + tag(perm), image(perm))
                ps = compile_hlsl(compiler, kind, pixel_hlsl(perm), 'ps_%s_%s' % (tag(perm), kind), ps_profile,
                                  directory)
                run = draw(renderer, adapter, vs, ps, reference)
                print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                                      if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
                check('%s %s %s: all %d pixels match the reference' % (kind, adapter, spelling(perm), WIDTH * HEIGHT),
                      run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                      run.stdout + run.stderr)
                for how in MUTATIONS:
                    bad = compile_hlsl(compiler, kind, pixel_hlsl(perm, how),
                                       'ps_%s_%s_%s' % (tag(perm), how.replace(' ', '_'), kind), ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad, reference)
                    lines = [ln for ln in run.stderr.splitlines() if 'pixel' in ln]
                    check('%s %s %s %s mutation fails against the reference: %s' % (
                        kind, adapter, spelling(perm), how, ' / '.join(lines)),
                        run.returncode != 0 and bool(lines) and lines[0].startswith('pixel mismatch'),
                        run.stdout + run.stderr)


def check_binary(llvm_mc):
    for perm in PERMS:
        check('llvm-mc assembles the %s fixture to the same words' % spelling(perm),
              assemble(llvm_mc, ps_source(perm)) == ps_words(perm))
    check('bound_ctrl:0 sets the bound_ctrl bit',
          assemble(llvm_mc, 'v_mov_b32_dpp v2, v1 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf bound_ctrl:0\n')
          == vop1_dpp(V_MOV_B32, 2, 1, control((1, 1, 3, 3)), bound_ctrl=True))


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
