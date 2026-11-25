"""DPP16 quad swaps in the pixel stage of tools/startup/gfx2hlsl.py (#2125, item B).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words built here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with DXC vs_6_0/ps_6_0, draw 16 x 16 pixels
                                       and compare every pixel with the quad reference below

In a DPP16 `quad_perm:[a,b,c,d]` source, lane i of each aligned group of four reads lane perm[i] of that group
(AMD RDNA 2 ISA reference, "Data-Parallel Primitives"; LLVM 15 AMDGPU `quad_perm`). The three exchanges
[1,0,3,2], [2,3,0,1] and [3,2,1,0] flip lane bit 0, bit 1 and both, which for the 2x2 pixel quad (lane = 2 * y + x)
is the pixel at x ^ 1, y ^ 1 or both. The lowering is QuadReadAcrossX, QuadReadAcrossY and QuadReadAcrossDiagonal
(SM 6.0), which read helper lanes too, so a quad that is only partly covered still gets a value for every lane.
Only a pixel stage at entry EXEC is lowered; FXC (ps_5_1) has no quad intrinsic, so a pair that needs one is
compiled with DXC as vs_6_0/ps_6_0 and refused when no DXC is found.

The fixtures read the packed screen position (x | y << 16) in v0 and OR 0x40000000 into every exported word so
the words are normal binary32 values. The reference is written here from the lane rule with integers: the exchanged
source is the position of the pixel at x ^ 1 and/or y ^ 1. Mutations of the translated text (an exchange that
reads the wrong neighbour, and no exchange) must read a different image.
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
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

WIDTH = HEIGHT = 16
MASK = 0xffffffff
FLAG = 0x40000000  # binary32 2.0, which the assembler encodes as the inline constant 244
WQM = struct.pack('<I', 0xbefe0a7e)  # s_wqm_b64 exec, exec
WQM_TEXT = 's_wqm_b64 exec, exec\n'
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
V_OR = 0x1c
V_MIN_U32 = 0x13
NAMES = ('or', 'mov', 'min')
SWAPS = {(1, 0, 3, 2): ('X', 1, 0), (2, 3, 0, 1): ('Y', 0, 1), (3, 2, 1, 0): ('Diagonal', 1, 1)}
failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def ctrl(perm):
    return perm[0] | perm[1] << 2 | perm[2] << 4 | perm[3] << 6


def dpp_text(mnemonic, ops, perm, extra=' row_mask:0xf bank_mask:0xf'):
    return '%s %s quad_perm:[%s]%s' % (mnemonic, ops, ','.join(map(str, perm)), extra)


def vop1_dpp(op, vdst, src, perm, bank=0xf, row=0xf, bound=0):
    return struct.pack('<2I', 0x7e000000 | vdst << 17 | op << 9 | 250,
                       src | ctrl(perm) << 8 | bound << 19 | bank << 24 | row << 28)


def vop2_dpp(op, vdst, src0, vsrc1, perm, bank=0xf, row=0xf, bound=0):
    return struct.pack('<2I', op << 25 | vdst << 17 | vsrc1 << 9 | 250,
                       src0 | ctrl(perm) << 8 | bound << 19 | bank << 24 | row << 28)


def vop1_mov_flag(vdst):
    return struct.pack('<I', 0x7e000000 | vdst << 17 | 1 << 9 | 244)


def vop2_plain(op, vdst, src0, vsrc1):
    return struct.pack('<I', op << 25 | vdst << 17 | vsrc1 << 9 | 256 + src0)


EXPORT = struct.pack('<3I', 0xf800180f, 0x04030201, 0xbf810000)  # exp mrt0 v1, v2, v3, v4 done vm; s_endpgm
EXPORT_TEXT = 'exp mrt0 v1, v2, v3, v4 done vm\ns_endpgm\n'
FLAG_TEXT = 'v_mov_b32_e32 v9, 2.0\n'


def fixture(name, wqm=True):
    """-> (assembly text, machine words). 'or': VOP2 DPP v_or_b32 with the flag in v9 as the own-lane source.
    'mov': VOP1 DPP v_mov_b32 into v5..v7, then a plain OR with v9. v4 is always the own-lane position | flag.
    'min': VOP2 DPP v_min_u32 (#2023 item H) of the exchanged position and the own position into v1..v3, then a
    plain OR with v9. Positions order by y first, so the minimum is the neighbour or the lane itself by pixel."""
    text, words = (WQM_TEXT, WQM) if wqm else ('', b'')
    text, words = text + FLAG_TEXT, words + vop1_mov_flag(9)
    perms = list(SWAPS)
    for i, perm in enumerate(perms):
        if name == 'or':
            text += dpp_text('v_or_b32_dpp', 'v%d, v0, v9' % (1 + i), perm) + '\n'
            words += vop2_dpp(V_OR, 1 + i, 0, 9, perm)
        elif name == 'min':
            text += dpp_text('v_min_u32_dpp', 'v%d, v0, v0' % (1 + i), perm) + '\n'
            words += vop2_dpp(V_MIN_U32, 1 + i, 0, 0, perm)
        else:
            text += dpp_text('v_mov_b32_dpp', 'v%d, v0' % (5 + i), perm) + '\n'
            words += vop1_dpp(1, 5 + i, 0, perm)
    if name == 'min':
        for i in range(3):
            text += 'v_or_b32_e32 v%d, v%d, v9\n' % (1 + i, 1 + i)
            words += vop2_plain(V_OR, 1 + i, 1 + i, 9)
    if name == 'mov':
        for i in range(3):
            text += 'v_or_b32_e32 v%d, v%d, v9\n' % (1 + i, 5 + i)
            words += vop2_plain(V_OR, 1 + i, 5 + i, 9)
    text += 'v_or_b32_e32 v4, v0, v9\n'
    words += vop2_plain(V_OR, 4, 0, 9)
    return text + EXPORT_TEXT, words + EXPORT


def pixel_words(x, y, name='or'):
    """The four exported words of pixel (x, y): the position of its neighbour in the quad for each swap, then its own.
    For 'min' each word is the unsigned minimum of the neighbour position and the own position."""
    out = []
    for _, dx, dy in SWAPS.values():
        word = (x ^ dx) | (y ^ dy) << 16
        out.append((min(word, x | y << 16) if name == 'min' else word) | FLAG)
    return out + [(x | y << 16) | FLAG]


def image(skip=None, name='or'):
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            row = pixel_words(x, y, name)
            if skip is not None:
                row[skip] = (x | y << 16) | FLAG  # the exchange missing: each lane keeps its own source
            words += row
    return words


def pipeline(words):
    return G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=PACKED)


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for DPP instructions to be translated one at a time."""

    def __init__(self, kind='ps', exec_known=True, wqm_exec=True):
        self.kind = kind
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = exec_known
        self.wqm_exec = wqm_exec
        self.has_branches, self.blocks = False, []


def refused(asm, **kwargs):
    stage = Probe(**kwargs)
    try:
        stage.one(asm.split(' ', 1)[0], asm)
    except G.Unsupported as reason:
        return str(reason)
    return None


def controls():
    # 1. The quad_perm selections that are swaps are exactly the lane rule: flipping lane bit 0, bit 1 or both.
    for perm, (name, dx, dy) in SWAPS.items():
        check('quad_perm:%s is the neighbour at x^%d, y^%d' % (list(perm), dx, dy),
              all(perm[lane] == lane ^ (dx | dy << 1) for lane in range(4)))
    check('the lowering table names X, Y and Diagonal for those three selections',
          {p: v for p, v in G.QUAD_SWAPS.items()} == {p: 'QuadReadAcross' + n for p, (n, _, _) in SWAPS.items()})

    # 2. The fixture words decode to the assembly text they were built from (the decoder is the reference for the bits).
    for name in NAMES:
        text, words = fixture(name)
        decoded = G.decode(words)
        got = [asm for _, _, _, asm in decoded]
        want = [line for line in text.splitlines()]
        normal = lambda s: re.sub(r'_e(32|64)\b', '', s)  # noqa: E731
        check('%s fixture decodes to its source (%d instructions)' % (name, len(want)),
              [normal(a) for a in got] == [normal(w) for w in want], '%s\n%s' % (got, want))

    # 3. Translation: each fixture lowers to the three intrinsics and asks for shader model 6.0.
    for name in NAMES:
        pipe = pipeline(fixture(name)[1])
        ps = pipe.hlsl_ps()
        check('%s fixture: three distinct quad intrinsics and the stage needs SM 6.0' % name,
              all(ps.count('QuadReadAcross%s(' % n) == 1 for n in ('X', 'Y', 'Diagonal')) and pipe.ps.needs_sm6
              and not pipe.vs.needs_sm6)

    # 4. Refusals: everything outside item B stays refused, with a reason.
    base = 'v_mov_b32_dpp v5, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf'
    check('the plain swap is accepted', refused(base) is None)
    check('bound_ctrl:1 is accepted (a quad_perm source is always in range)', refused(base + ' bound_ctrl:1') is None)
    check('quad broadcast also translates', refused(base.replace('[1,0,3,2]', '[0,0,0,0]')) is None)
    check('pair broadcast also translates', refused(base.replace('[1,0,3,2]', '[1,1,3,3]')) is None)
    cases = [
        ('identity [0,1,2,3]', base.replace('[1,0,3,2]', '[0,1,2,3]'), {}),
        ('partial row_mask', base.replace('row_mask:0xf', 'row_mask:0x3'), {}),
        ('partial bank_mask', base.replace('bank_mask:0xf', 'bank_mask:0x5'), {}),
        ('row_mirror', 'v_mov_b32_dpp v5, v0 row_mirror row_mask:0xf bank_mask:0xf', {}),
        ('DPP8', 'v_mov_b32_dpp v5, v0 dpp8:[1,0,3,2,5,4,7,6]', {}),
        ('fi:1', base + ' fi:1', {}),
        ('negated source', 'v_add_f32_dpp v5, -v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf', {}),
        ('SGPR source', 'v_mov_b32_dpp v5, s0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf', {}),
        ('operation without a table entry', 'v_cvt_f64_f32_dpp v[5:6], v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf', {}),
        ('vertex stage', base, {'kind': 'vs'}),
        ('EXEC no longer the entry mask', base, {'exec_known': False}),
        ('entry EXEC not proven whole-quad (partly covered quads)', base, {'wqm_exec': False}),
    ]
    for label, asm, kwargs in cases:
        reason = refused(asm, **kwargs)
        check('refused: %s (%s)' % (label, reason), reason is not None)
    branched = Probe()
    branched.has_branches = True
    try:
        branched.one('v_mov_b32_dpp', base)
        check('refused: a shader with branches', False)
    except G.Unsupported:
        check('refused: a shader with branches', True)

    # 4b. Whole-quad EXEC comes only from s_wqm of exec into itself, seen on the real translation path.
    for name in NAMES:
        try:
            pipeline(fixture(name, wqm=False)[1]).hlsl_ps()
            check('%s fixture without s_wqm exec is refused' % name, False)
        except G.Unsupported as reason:
            check('%s fixture without s_wqm exec is refused (%s)' % (name, reason), 'whole-quad' in str(reason))
    # A copy of EXEC taken before the s_wqm and restored after it narrows EXEC again: the proof must go.
    narrowing = struct.pack('<3I', 0xbe80047e, 0xbefe0a7e, 0xbefe0400)  # mov s[0:1], exec; wqm exec; mov exec, s[0:1]
    words = narrowing + fixture('mov', wqm=False)[1]
    try:
        pipeline(words).hlsl_ps()
        check('refused: whole-quad proof dropped by a restored exec copy', False)
    except G.Unsupported as reason:
        check('refused: whole-quad proof dropped by a restored exec copy (%s)' % reason, 'whole-quad' in str(reason))

    # 4c. The min fixture keeps src1 in its own lane and its reference is neither the plain exchange nor a maximum.
    line = pipeline(fixture('min')[1]).hlsl_ps()
    check('v_min_u32_dpp lowers to min(exchanged src0, own src1)',
          all(re.search(r'min\(QuadReadAcross%s\(v0\), v0\)' % n, line) for n in ('X', 'Y', 'Diagonal')))
    check('the min reference differs from the plain exchange at some pixels in every swap',
          all(any(pixel_words(x, y, 'min')[i] != pixel_words(x, y)[i] for x in range(WIDTH) for y in range(HEIGHT))
              for i in range(3)))
    check('the min reference keeps the own position where it is the smaller one',
          pixel_words(0, 2, 'min')[2] == (0 | 2 << 16 | FLAG) and pixel_words(0, 3, 'min')[2] == (1 | 2 << 16 | FLAG))

    # 5. A float VOP2 keeps its second source in its own lane.
    stage = Probe()
    stage.one('v_add_f32_dpp', dpp_text('v_add_f32_dpp', 'v5, v0, v1', (1, 0, 3, 2)))
    line = ' '.join(stage.lines)
    check('v_add_f32_dpp exchanges src0 only: %s' % line,
          'F(QuadReadAcrossX(v0))' in line and 'F(v1)' in line and 'QuadReadAcrossX(v1)' not in line)

    # 6. A pair that needs shader model 6.0 is refused when no DXC exists and published without bytecode otherwise.
    saved = os.environ.get('X2W_FXC')
    real_find = G.find_dxc
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ['X2W_FXC'] = sys.executable  # any existing file: only its presence is read before DXC is looked up
            G.find_dxc = lambda: None
            message = G.translate_to(tmp, 1, VS_WORDS, fixture('or')[1], root_blob())
            check('without dxc the quad swap pair is refused with a reason: %s' % message,
                  'dxc not found' in message and os.path.exists(os.path.join(tmp, '%016x.fail' % 1)))
            message = G.translate_to(tmp, 2, VS_WORDS, fixture('or')[1], root_blob(), check=False)
            check('with check=False the pair is published without bytecode: %s' % message,
                  message == 'ok' and os.path.exists(os.path.join(tmp, '%016x.txt' % 2)))
    finally:
        G.find_dxc = real_find
        if saved is None:
            os.environ.pop('X2W_FXC', None)
        else:
            os.environ['X2W_FXC'] = saved


def check_binary(llvm_mc):
    for name in NAMES:
        text, words = fixture(name)
        check('llvm-mc assembles the %s fixture to the same words' % name, assemble(llvm_mc, text) == words)


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


def mutate(text, old, new):
    mutated = re.sub(old, new, text)
    assert mutated != text, (old, text)
    return mutated


def check_draw(dxc, renderer, adapter, fxc=None):
    if fxc:
        print('FXC ps_5_1 skipped: it has no quad intrinsic', flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs', 'vs_6_0', directory)
        for name in NAMES:
            good = words_file(directory, 'good_' + name, image(name=name))
            text = pipeline(fixture(name)[1]).hlsl_ps()
            ps = compile_hlsl(dxc, 'dxc', text, 'ps_' + name, 'ps_6_0', directory)
            run = draw(renderer, adapter, vs, ps, good)
            check('%s %s: all %d pixels equal the quad reference' % (adapter, name, WIDTH * HEIGHT),
                  run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                  run.stdout + run.stderr)
            # Negative controls: each wrong lowering must read a different image than the reference.
            wrong = {
                'X reads the Y neighbour': mutate(text, r'QuadReadAcrossX\(', 'QuadReadAcrossY('),
                'Y reads the diagonal neighbour': mutate(text, r'QuadReadAcrossY\(', 'QuadReadAcrossDiagonal('),
                'Diagonal reads the X neighbour': mutate(text, r'QuadReadAcrossDiagonal\(', 'QuadReadAcrossX('),
                'no exchange (each lane keeps its own value)': re.sub(r'QuadReadAcross\w+\((v\d+)\)', r'\1', text),
            }
            if name == 'min':
                wrong['unsigned maximum instead of minimum'] = mutate(text, r'\bmin\(', 'max(')
            for label, source in wrong.items():
                bad = compile_hlsl(dxc, 'dxc', source, 'bad_%s_%d' % (name, abs(hash(label))), 'ps_6_0', directory)
                run = draw(renderer, adapter, vs, bad, good)
                check('%s %s mutation "%s" fails against the reference' % (adapter, name, label),
                      run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)
            # The unmutated lowering also differs from an image in which one exchange is missing.
            for index in range(3):
                partial = words_file(directory, 'partial%d' % index, image(skip=index, name=name))
                run = draw(renderer, adapter, vs, ps, partial)
                check('%s %s: reference with swap %d dropped is rejected' % (adapter, name, index),
                      run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)


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
