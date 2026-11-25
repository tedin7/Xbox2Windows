"""DPP16 quad broadcasts (quad_perm:[k,k,k,k]) in tools/startup/gfx2hlsl.py (#2125 item A).

  (no arguments)                       translation controls and refusals, pure Python
  binary <llvm-mc>                     assemble the pixel fixtures with llvm-mc (gfx1013) and compare the words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the pixel fixtures with DXC ps_6_0, draw 16 x 16 pixels and compare
                                       every pixel with a CPU reference (with fxc: also the FXC refusal). The vertex
                                       stage is DXIL too: D3D12 rejects a DXBC + DXIL pipeline state pair

Semantics (RDNA 2 ISA, "DPP": a lane reads SRC0 of lane (lane & ~3) | quad_perm[lane & 3] of its row; LLVM 15
AMDGPU DPP quad_perm): quad_perm:[k,k,k,k] gives every lane of a quad the word of lane k of that quad. The four
lanes of a pixel-stage quad are one 2x2 pixel block: lane = (x & 1) | (y & 1) << 1. The reference below is plain
Python over that layout, not the lowering's expression. The lowering is SM 6.0 QuadReadLaneAt, so a stage that holds
one compiles as ps_6_0 (Pipeline.ps_profile), which the pipeline compiler routes to DXC.

Fixtures (pixel stage, v0 = packed screen position x | y << 16, every lane live because the fullscreen triangle
covers whole quads):
  integer: v_mov_b32_dpp from lanes 0, 1 and 2 (with and without bound_ctrl), and v_add_nc_u32_dpp from lane 3 whose
           second source is a plain VGPR, so a permuted second source or a lane-0 read through the add would differ
  float:   the coarse derivative idiom, v_sub_f32_dpp (lane 1 minus own lane), and v_mul_f32_dpp with a negated
           permuted source (the modifier applies to the word the lane read)
Two mutated translations, one that reads lane k + 1 and one that reads the lane's own word, must fail the draw
against the reference and match their own plain-Python image.

Refused (the single-lane model cannot read a neighbour, or the quad may be partly off): swaps, pair broadcasts, row
shifts, mirrors, DPP8, a partial row_mask or bank_mask, fi, bound_ctrl:0, a scalar or constant source, a vertex
stage, a pixel stage after EXEC changed or with branches, and operations outside the VOP1/VOP2 tables.
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
PACKED = 0x8000  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print(detail)


def dpp16(vop2, op, vdst, src0, vsrc1, perm, bound_ctrl, neg0=False):
    """The two words of a VOP1 (vop2 False) or VOP2 DPP16 instruction with full row and bank masks."""
    control = perm[0] | perm[1] << 2 | perm[2] << 4 | perm[3] << 6
    if vop2:
        word0 = op << 25 | vdst << 17 | vsrc1 << 9 | 250
    else:
        word0 = 0x7e000000 | vdst << 17 | op << 9 | 250
    word1 = src0 | control << 8 | int(bound_ctrl) << 19 | int(neg0) << 20 | 0xf << 24 | 0xf << 28
    return struct.pack('<2I', word0, word1)


def export(regs):
    return struct.pack('<3I', 0xf800180f, regs[0] | regs[1] << 8 | regs[2] << 16 | regs[3] << 24, 0xbf810000)


def vop1(op, vdst, src0):
    return struct.pack('<I', 0x7e000000 | vdst << 17 | op << 9 | 0x100 | src0)


WQM = struct.pack('<I', 0xBE800000 | 126 << 16 | 0x0a << 8 | 126)  # s_wqm_b64 exec, exec
DPP = ' row_mask:0xf bank_mask:0xf'
INT_SOURCE = """\
s_wqm_b64 exec, exec
v_mov_b32_dpp v1, v0 quad_perm:[0,0,0,0]%(m)s bound_ctrl:1
v_mov_b32_dpp v2, v0 quad_perm:[1,1,1,1]%(m)s
v_mov_b32_dpp v3, v0 quad_perm:[2,2,2,2]%(m)s bound_ctrl:1
v_add_nc_u32_dpp v4, v0, v1 quad_perm:[3,3,3,3]%(m)s bound_ctrl:1
exp mrt0 v1, v2, v3, v4 done vm
s_endpgm
""" % {'m': DPP}
INT_WORDS = (WQM + dpp16(False, 1, 1, 0, 0, (0, 0, 0, 0), True) + dpp16(False, 1, 2, 0, 0, (1, 1, 1, 1), False)
             + dpp16(False, 1, 3, 0, 0, (2, 2, 2, 2), True) + dpp16(True, 0x25, 4, 0, 1, (3, 3, 3, 3), True)
             + export((1, 2, 3, 4)))
FLOAT_SOURCE = """\
s_wqm_b64 exec, exec
v_cvt_f32_u32_e32 v5, v0
v_sub_f32_dpp v6, v5, v5 quad_perm:[1,1,1,1]%(m)s bound_ctrl:1
v_mul_f32_dpp v7, -v5, v5 quad_perm:[0,0,0,0]%(m)s bound_ctrl:1
exp mrt0 v6, v7, v5, v5 done vm
s_endpgm
""" % {'m': DPP}
FLOAT_WORDS = (WQM + vop1(6, 5, 0) + dpp16(True, 0x4, 6, 5, 5, (1, 1, 1, 1), True)
               + dpp16(True, 0x8, 7, 5, 5, (0, 0, 0, 0), True, neg0=True) + export((6, 7, 5, 5)))
NO_WQM_WORDS = INT_WORDS[len(WQM):]  # the entry EXEC may leave out lanes of a partly covered quad
OTHER_WQM_WORDS = struct.pack('<I', 0xBE800000 | 2 << 16 | 0x0a << 8) + NO_WQM_WORDS  # s_wqm_b64 s[2:3], s[0:1]
FIXTURES = {'integer': (INT_SOURCE, INT_WORDS), 'float': (FLOAT_SOURCE, FLOAT_WORDS)}
MUTATIONS = ('wrong lane', 'own lane')


def f32(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def lane_word(x, y, lane):
    """The packed position of pixel lane `lane` of the quad that holds pixel (x, y)."""
    return (x & ~1) + (lane & 1) | ((y & ~1) + (lane >> 1)) << 16


def read(x, y, k, how):
    """The word the lane of pixel (x, y) reads from quad lane k, for the reference or a mutated lowering."""
    own = x | y << 16
    if how == 'own lane':
        return own
    return lane_word(x, y, (k + 1) & 3 if how == 'wrong lane' else k)


def as_float(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def image(kind, how='reference'):
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            own = x | y << 16
            if kind == 'integer':
                q0 = read(x, y, 0, how)
                words += [q0, read(x, y, 1, how), read(x, y, 2, how), (read(x, y, 3, how) + q0) & MASK]
            else:
                own_f = as_float(f32(float(own)))
                q1 = as_float(f32(float(read(x, y, 1, how))))
                q0 = as_float(f32(float(read(x, y, 0, how))))
                words += [f32(q1 - own_f), f32(-q0 * own_f), f32(float(own)), f32(float(own))]
    return words


def pixel_stage(words):
    return G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=(PACKED, PACKED))


def pixel_hlsl(kind, how='reference'):
    text = pixel_stage(FIXTURES[kind][1]).hlsl_ps()
    if how == 'wrong lane':
        text, count = re.subn(r'QuadReadLaneAt\((v\d+), ([0-3])u\)',
                              lambda m: 'QuadReadLaneAt(%s, %du)' % (m.group(1), (int(m.group(2)) + 1) & 3), text)
    elif how == 'own lane':
        text, count = re.subn(r'QuadReadLaneAt\((v\d+), [0-3]u\)', r'\1', text)
    else:
        return text
    assert count == (4 if kind == 'integer' else 2), (kind, how, count)
    return text


class Probe(G.Stage):
    """A pixel Stage with no pipeline: enough state for single instructions to be translated."""

    def __init__(self, kind='ps', exec_known=True, branches=False, quad_exec=True):
        self.kind = kind
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known, self.has_branches, self.blocks, self.needs_sm6 = exec_known, branches, [], False
        self.wqm_exec = quad_exec


def refused(assembly, **kwargs):
    try:
        Probe(**kwargs).one(assembly.split(' ', 1)[0], assembly)
    except G.Unsupported:
        return True
    return False


def refused_in_stage(words):
    try:
        pixel_stage(words).hlsl_ps()
    except G.Unsupported:
        return True
    return False


def controls():
    for kind, (source, words) in FIXTURES.items():
        decoded = [asm for _o, _s, _n, asm in G.decode(words)]
        check('%s fixture decodes to its source' % kind, decoded == source.splitlines(), str(decoded))
        stage = pixel_stage(words)
        text = stage.hlsl_ps()
        reads = re.findall(r'v256 = QuadReadLaneAt\((v\d+), ([0-3])u\);', text)
        want = [('v0', '0'), ('v0', '1'), ('v0', '2'), ('v0', '3')] if kind == 'integer' else [('v5', '1'), ('v5', '0')]
        check('%s fixture reads the quad lanes' % kind, reads == want, text)
        check('%s fixture needs ps_6_0' % kind, stage.ps_profile() == 'ps_6_0')
        reference = image(kind)
        for how in MUTATIONS:
            other = image(kind, how)
            differing = sum(other[i:i + 4] != reference[i:i + 4] for i in range(0, len(other), 4))
            check('%s: %s mutation changes %d pixels' % (kind, how, differing), differing > 0)
            pixel_hlsl(kind, how)
    plain = pixel_stage(vop1(6, 5, 0) + export((5, 5, 5, 5)))
    check('a stage without a quad read stays ps_5_1', plain.ps_profile() == 'ps_5_1')
    # The four lanes of the integer image are four distinct words in every quad, so a lane mix-up cannot hide.
    quad = {tuple(image('integer')[(y * WIDTH + x) * 4:(y * WIDTH + x) * 4 + 3]) for y in (0, 1) for x in (0, 1)}
    check('every lane of a quad reads the same three words', len(quad) == 1)
    accepted = ['v_mov_b32_dpp v1, v0 quad_perm:[0,0,0,0]' + DPP,
                'v_mov_b32_dpp v1, v0 quad_perm:[3,3,3,3]' + DPP + ' bound_ctrl:1',
                'v_add_f32_dpp v1, |v2|, v3 quad_perm:[2,2,2,2]' + DPP,
                'v_mul_f32_dpp v1, -v2, v3 quad_perm:[1,1,1,1]' + DPP + ' mul:2',
                'v_sub_nc_u32_dpp v1, v2, v3 quad_perm:[1,1,1,1]' + DPP]
    check('accepted forms translate', not any(refused(a) for a in accepted), str([a for a in accepted if refused(a)]))
    sane = 'v_mov_b32_dpp v1, v0 quad_perm:[1,1,1,1]' + DPP
    check('the merged quad swap also translates', not refused('v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2]' + DPP))
    check('the pair broadcast also translates', not refused('v_mov_b32_dpp v1, v0 quad_perm:[1,1,3,3]' + DPP))
    bad = {
        'identity': 'v_mov_b32_dpp v1, v0 quad_perm:[0,1,2,3]' + DPP,
        'row_shl': 'v_mov_b32_dpp v1, v0 row_shl:1' + DPP,
        'row_mirror': 'v_mov_b32_dpp v1, v0 row_mirror' + DPP,
        'dpp8': 'v_mov_b32_dpp v1, v0 dpp8:[1,1,1,1,1,1,1,1]',
        'partial row_mask': 'v_mov_b32_dpp v1, v0 quad_perm:[1,1,1,1] row_mask:0xe bank_mask:0xf',
        'partial bank_mask': 'v_mov_b32_dpp v1, v0 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x7',
        'no masks': 'v_mov_b32_dpp v1, v0 quad_perm:[1,1,1,1]',
        'fi': sane + ' fi:1',
        'bound_ctrl:0': sane + ' bound_ctrl:0',
        'scalar source': 'v_mov_b32_dpp v1, s2 quad_perm:[1,1,1,1]' + DPP,
        'constant source': 'v_mov_b32_dpp v1, 1.0 quad_perm:[1,1,1,1]' + DPP,
        'vector source': 'v_mov_b32_dpp v1, v[2:3] quad_perm:[1,1,1,1]' + DPP,
        'operation outside the tables': 'v_cndmask_b32_dpp v1, v0, v2 quad_perm:[1,1,1,1]' + DPP,
        'mov with a modifier': sane + ' clamp',
    }
    for label, line in bad.items():
        check('refused: %s' % label, refused(line), line)
    check('refused: vertex stage', refused(sane, kind='vs'))
    check('refused: EXEC no longer the entry mask', refused(sane, exec_known=False))
    check('refused: no s_wqm exec, exec before the broadcast', refused(sane, quad_exec=False))
    check('refused: the stage translator without s_wqm', refused_in_stage(NO_WQM_WORDS))
    check('refused: s_wqm into another register', refused_in_stage(OTHER_WQM_WORDS))
    check('refused: pixel stage with branches', refused(sane, branches=True))
    check('control: the sane line itself translates', not refused(sane))


def check_binary(llvm_mc):
    for kind, (source, words) in FIXTURES.items():
        check('llvm-mc assembles the %s fixture to the same words' % kind, assemble(llvm_mc, source) == words)


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


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        stages = [('dxc', dxc, 'vs_6_0')]
        for kind in FIXTURES:
            reference = words_file(directory, 'reference_' + kind, image(kind))
            for vs_kind, vs_compiler, vs_profile in stages:
                vs = compile_hlsl(vs_compiler, vs_kind, FULLSCREEN_VS, 'vs_%s' % vs_kind, vs_profile, directory)
                label = '%s vertex + dxc ps_6_0 %s %s' % (vs_kind, adapter, kind)
                ps = compile_hlsl(dxc, 'dxc', pixel_hlsl(kind), 'ps_%s' % kind, 'ps_6_0', directory)
                run = draw(renderer, adapter, vs, ps, reference)
                print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                                      if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
                check('%s: all %d pixels match the reference' % (label, WIDTH * HEIGHT),
                      run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                      run.stdout + run.stderr)
                for how in MUTATIONS:
                    name = 'ps_%s_%s' % (kind, how.replace(' ', '_'))
                    bad = compile_hlsl(dxc, 'dxc', pixel_hlsl(kind, how), name, 'ps_6_0', directory)
                    run = draw(renderer, adapter, vs, bad, reference)
                    lines = [ln for ln in run.stderr.splitlines() if 'pixel' in ln]
                    check('%s: %s mutation fails against the reference: %s' % (label, how, ' / '.join(lines)),
                          run.returncode != 0 and lines and lines[0].startswith('pixel mismatch'),
                          run.stdout + run.stderr)
                    own = words_file(directory, '%s_%s' % (kind, how.replace(' ', '_')), image(kind, how))
                    run = draw(renderer, adapter, vs, bad, own)
                    check('%s: %s mutation matches its own plain-Python image' % (label, how), run.returncode == 0,
                          run.stdout + run.stderr)
        if fxc:
            # The quad read has no shader model 5.1 spelling: FXC must refuse it, which is why the pipeline
            # compiler sends such a pixel stage to DXC.
            try:
                compile_hlsl(fxc, 'fxc', pixel_hlsl('integer'), 'ps_fxc_control', 'ps_5_1', directory)
            except subprocess.CalledProcessError:
                check('fxc ps_5_1 refuses the quad read (control)', True)
            else:
                check('fxc ps_5_1 refuses the quad read (control)', False)


def main():
    if len(sys.argv) == 1:
        controls()
    elif sys.argv[1] == 'binary':
        check_binary(*sys.argv[2:])
    elif sys.argv[1] == 'draw':
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
    if failures:
        raise SystemExit('FAILED: %s' % ', '.join(failures))


if __name__ == '__main__':
    main()
