"""Whole-wave reductions in tools/startup/gfx2hlsl.py (#2125 item D), read back from a real D3D12 draw.

  (no arguments)                       a 32-lane model of the source chains, translation controls and refusals
  binary <llvm-mc>                     assemble every fixture with LLVM (gfx1013) and compare with the words here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pixel stage with DXC ps_6_0 (the lowering uses shader
                                       model 6.0 wave intrinsics, so FXC is not used), draw a triangle over 32 x 32
                                       pixels and compare every covered pixel with a CPU reference for its desktop
                                       wave. The optional fxc argument is accepted and ignored.

The three wave32 shapes of the #2125 censuses (RDNA 2 ISA: DPP16 quad_perm, row_half_mirror, row_mirror,
V_PERMLANEX16_B32, V_READLANE_B32; every DPP step with row_mask:0xf bank_mask:0xf fi:1):
  OR    s_or_saveexec_b32 sS, -1 / v_cndmask_b32_e64 vT, 0, vX, sS / v_or_b32_dpp vT, vT, vT quad_perm:[1,0,3,2],
        quad_perm:[2,3,0,1], row_half_mirror, row_mirror / v_permlanex16_b32 vP, vT, 0, 0 / v_or_b32_e32 vT, vP, vT /
        s_mov_b32 exec_lo, sS / v_readlane_b32 sD, vT, 31
  MIN   the same with identity -1, but after the restore come one v_cndmask_b32_sdwa, the s_or_saveexec_b32 sS2, -1 of
        the max chain, and only then v_readlane_b32 sD, vT, 31
  MAX   right after a min: s_or_saveexec_b32 sS2, -1 / (the min's readlane) / v_cndmask_b32_e64 vT, 0, vX, sS2 / the
        same steps with v_max_u32 / restore / readlane at once
The fixtures here are those shapes, written in LLVM 15 spelling and assembled from source, plus the immediate readlane
for each operation alone and variants that must be accepted or refused.

The model below runs the source lines on 32 lanes with plain integers, from the ISA text and not from the lowering,
and shows that sD is OP over the lanes active before the chain for random EXEC masks, and that each refused mutation
either is not that reduction for some mask or is a conservative refusal (said so in its label). It also shows the
order that matters in the interleaved form: with vT shared by the two chains, the min's readlane moved after the max
fill reads the max.

The desktop contract: the console packs pixels into waves in its own way, so a correct program cannot depend on which
pixels share a wave; the lowering is one WaveActive* over the active lanes of the desktop wave. The draw cannot know
the desktop packing in advance, so the test appends one observation to the translated pixel stage: the fourth channel
is replaced by the lane index, the lane count (WaveGetLaneCount) and the pixel id of WaveReadLaneFirst, which names
the wave. The readback (offscreen_check --startup-pixel-dump) is grouped by that id and every covered pixel's results
are compared with OP over the inputs of the covered pixels of its group. The triangle leaves quads partly covered:
their helper lanes run the stage, are not in the console EXEC (no s_wqm), and must not change the result. Planted
errors in the translated text (another operation, signed for unsigned, the lane's own value, the first lane's value;
for the interleaved pair the min's readlane lowered after the max, min and max swapped, the v_cndmask_b32_sdwa
dropped) must fail.
"""
import collections
import random
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
from startup_image_3d_check import VS_WORDS, compile_hlsl, root_blob  # noqa: E402

MASK = 0xffffffff
SIZE = 32                  # the draw is SIZE x SIZE pixels
CLEAR = 0xbf800000         # -1.0: the target's clear value, kept by uncovered pixels
MULTIPLIER = 0x9b5d
BRANCH_AND_BIT = 0x00800000
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
failures = []

# A triangle with three slanted edges, so many quads are partly covered (helper lanes) and some waves are partial.
TRIANGLE_VS = """\
float4 main(uint id : SV_VertexID) : SV_Position
{
    return id == 0 ? float4(-1.0, 1.0, 0.0, 1.0) : id == 1 ? float4(1.0, 0.37, 0.0, 1.0) : float4(-0.61, -1.0, 0.0, 1.0);
}
"""


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + detail))
    if not ok:
        failures.append(label)


# ---- encoders (gfx10 field layouts; the binary mode checks every word against llvm-mc) --------------------------------
def sop1(op, sdst, ssrc):
    return struct.pack('<I', 0xbe800000 | (sdst << 16) | (op << 8) | ssrc)


def vop1_mov(vdst, src):
    return struct.pack('<I', 0x7e000000 | (vdst << 17) | (1 << 9) | src)


def vop2(op, vdst, src0, vsrc1):
    return struct.pack('<I', (op << 25) | (vdst << 17) | (vsrc1 << 9) | src0)


def vop2_dpp(op, vdst, src0, vsrc1, ctrl, bound_ctrl=False, fi=True, bank=0xf, row=0xf):
    return struct.pack('<II', (op << 25) | (vdst << 17) | (vsrc1 << 9) | 250,
                       src0 | (ctrl << 8) | (int(fi) << 18) | (int(bound_ctrl) << 19) | (bank << 24) | (row << 28))


def vop2_sdwa(op, vdst, src0, vsrc1, dst_sel, src0_sel, src1_sel):
    """VOP2 SDWA with VGPR sources and dst_unused:UNUSED_PAD (selects: 0..3 BYTE_n, 4 WORD_0, 5 WORD_1, 6 DWORD)."""
    return struct.pack('<II', (op << 25) | (vdst << 17) | (vsrc1 << 9) | 249,
                       src0 | (dst_sel << 8) | (src0_sel << 16) | (src1_sel << 24))


def vop3(op, vdst, src0, src1, src2=0, opsel=0):
    return struct.pack('<II', 0xd4000000 | (op << 16) | (opsel << 11) | vdst, src0 | (src1 << 9) | (src2 << 18))


def sopp(op, simm=0):
    return struct.pack('<I', 0xbf800000 | (op << 16) | (simm & 0xffff))


S_MOV_B32, S_WQM_B32, S_AND_SAVEEXEC_B32, S_OR_SAVEEXEC_B32, S_OR_SAVEEXEC_B64 = 0x03, 0x09, 0x3c, 0x3d, 0x25
EXEC_LO, INLINE_M1 = 126, 0xc1
V_PERMLANEX16_B32, V_READLANE_B32, V_CNDMASK_B32_E64, V_LSHLREV_B32_E64 = 0x378, 0x360, 0x101, 0x11a
V_CNDMASK_B32, V_CMP_GT_U32 = 0x01, 0xc4
VOP2_OPS = {'v_or_b32': 0x1c, 'v_min_u32': 0x13, 'v_max_u32': 0x14, 'v_max_i32': 0x12, 'v_add_nc_u32': 0x25,
            'v_and_b32': 0x1b, 'v_xor_b32': 0x1d, 'v_min_i32': 0x11,
            'v_max_f32': 0x10}
CTRL = {'quad_perm:[1,0,3,2]': 0xb1, 'quad_perm:[3,2,1,0]': 0x1b, 'quad_perm:[2,3,0,1]': 0x4e, 'row_half_mirror': 0x141, 'row_mirror': 0x140,
        'row_shr:1': 0x111}
IDENTITY = {'v_or_b32': 0, 'v_min_u32': 0xffffffff, 'v_max_u32': 0, 'v_max_i32': 0x80000000, 'v_add_nc_u32': 0,
            'v_and_b32': 0xffffffff, 'v_xor_b32': 0, 'v_min_i32': 0x7fffffff}
OPS = ('v_or_b32', 'v_min_u32', 'v_max_u32', 'v_and_b32', 'v_xor_b32', 'v_add_nc_u32')
STEPS = ('quad_perm:[1,0,3,2]', 'quad_perm:[2,3,0,1]', 'row_half_mirror', 'row_mirror')


def inline(value):
    """A fill constant as an inline constant: (LLVM spelling, encoded field)."""
    return {0: ('0', 128), 1: ('1', 129), 5: ('5', 133), MASK: ('-1', INLINE_M1)}[value]


def cndmask_fill(vt, src, ss, value, mask=None):
    """The census fill: v_cndmask_b32_e64 vT, <identity>, vX, sS."""
    text, field = inline(value)
    mask = ss if mask is None else mask
    return [('v_cndmask_b32_e64 v%d, %s, v%d, s%d' % (vt, text, src, mask),
             vop3(V_CNDMASK_B32_E64, vt, field, 256 + src, mask))]


def dpp_line(op, vt, ctrl, bound_ctrl=False, fi=True, row=0xf):
    text = '%s_dpp v%d, v%d, v%d %s row_mask:0x%x bank_mask:0xf%s%s' % (
        op, vt, vt, vt, ctrl, row, ' bound_ctrl:1' if bound_ctrl else '', ' fi:1' if fi else '')
    return text, vop2_dpp(VOP2_OPS[op], vt, vt, vt, CTRL[ctrl], bound_ctrl, fi, row=row)


def saveexec(ss):
    return ('s_or_saveexec_b32 s%d, -1' % ss, sop1(S_OR_SAVEEXEC_B32, ss, INLINE_M1))


def readlane(sd, vt, lane=31):
    return ('v_readlane_b32 s%d, v%d, %d' % (sd, vt, lane), vop3(V_READLANE_B32, sd, 256 + vt, 128 + lane))


def chain(op, vt=2, vp=3, ss=0, sd=1, src=1, fill='census', identity=None, fill_mask=None, steps=STEPS,
          step_ops=None, perm=(0, 0), perm_opsel=0, lane=31, combine='census', tail_order=('restore', 'readlane'),
          restore_reg=None, wave='b32', bound_ctrl=False, fi=True, row=0xf):
    """One chain as (source line, words) pairs; the keyword arguments build the mutations."""
    identity = IDENTITY[op] if identity is None else identity
    if wave == 'b64':
        lines = [('s_or_saveexec_b64 s[%d:%d], -1' % (ss, ss + 1), sop1(S_OR_SAVEEXEC_B64, ss, INLINE_M1))]
    else:
        lines = [saveexec(ss)]
    if fill == 'census':
        lines += cndmask_fill(vt, src, ss, identity, fill_mask)
    elif fill == 'copy':  # the input copied with no identity: the lanes outside sS keep their own value
        lines.append(('v_mov_b32_e32 v%d, v%d' % (vt, src), vop1_mov(vt, 256 + src)))
    for k, ctrl in enumerate(steps):
        lines.append(dpp_line((step_ops or {}).get(k, op), vt, ctrl, bound_ctrl, fi, row if k == 0 else 0xf))
    spelling = 'v_permlanex16_b32 v%d, v%d, %d, %d%s' % ((vp, vt) + perm + (' op_sel:[1,0]' if perm_opsel else '',))
    lines.append((spelling, vop3(V_PERMLANEX16_B32, vp, 256 + vt, 128 + perm[0], 128 + perm[1], perm_opsel)))
    if combine == 'census':   # v_or_b32_e32 vT, vP, vT
        lines.append(('%s_e32 v%d, v%d, v%d' % (op, vt, vp, vt), vop2(VOP2_OPS[op], vt, 256 + vp, vt)))
    elif combine == 'e32':    # the sources the other way round
        lines.append(('%s_e32 v%d, v%d, v%d' % (op, vt, vt, vp), vop2(VOP2_OPS[op], vt, 256 + vt, vp)))
    else:                     # VOP3
        lines.append(('%s_e64 v%d, v%d, v%d' % (op, vt, vp, vt), vop3(0x100 + VOP2_OPS[op], vt, 256 + vp, 256 + vt)))
    restore_reg = ss if restore_reg is None else restore_reg
    parts = {'readlane': readlane(sd, vt, lane),
             'restore': ('s_mov_b32 exec_lo, s%d' % restore_reg, sop1(S_MOV_B32, EXEC_LO, restore_reg))}
    return lines + [parts[t] for t in tail_order]


START_MUL = [('v_mul_u32_u24_e32 v1, 0x9b5d, v0', struct.pack('<II', (0x0b << 25) | (1 << 17) | 255, MULTIPLIER))]
START_BIT = [('v_lshlrev_b32_e64 v1, v0, 1', vop3(V_LSHLREV_B32_E64, 1, 256 + 0, 129))]  # 1 << (x & 31)
WQM = [('s_wqm_b32 exec_lo, exec_lo', sop1(S_WQM_B32, EXEC_LO, EXEC_LO))]
CMP = [('v_cmp_gt_u32_e32 vcc_lo, v1, v0', struct.pack('<I', 0x7c000000 | (V_CMP_GT_U32 << 17) | (0 << 9) | 257))]


def sdwa_line(vdst=6):
    """The census's v_cndmask_b32_sdwa between the min chain's restore and the max chain: vcc ? y : x * MULTIPLIER low
    word, here with v1 (the input) and v0 (the packed position, y in the high word)."""
    return [('v_cndmask_b32_sdwa v%d, v1, v0, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 '
             'src1_sel:WORD_1' % vdst, vop2_sdwa(V_CNDMASK_B32, vdst, 1, 0, 6, 4, 5))]


def export(first, second, third, last='v0'):
    regs = [int(r[1:]) for r in (first, second, third, last)]
    return [('exp mrt0 %s, %s, %s, %s done vm' % (first, second, third, last),
             struct.pack('<II', 0xf800180f, regs[0] | regs[1] << 8 | regs[2] << 16 | regs[3] << 24)),
            ('s_endpgm', sopp(0x01))]


def move(vd, sn):
    return ('v_mov_b32_e32 v%d, s%d' % (vd, sn), vop1_mov(vd, sn))


RESULT = [move(4, 1)]
END = export('v4', 'v1', 'v2')           # result, input, vT after the chain, position
END_READS_VP = export('v4', 'v1', 'v3')
KILL_VP = [('v_mov_b32_e32 v3, 0', vop1_mov(3, 128))]


def program(*parts):
    lines = [line for part in parts for line in part]
    return '\n'.join(text for text, _w in lines) + '\n', b''.join(w for _t, w in lines)


def start(op):
    return START_BIT if op == 'v_or_b32' else START_MUL


def word(op, x, y):
    return (1 << (x & 31)) if op == 'v_or_b32' else ((x | (y << 16)) * MULTIPLIER) & MASK


def fixture(op):
    return program(start(op), chain(op), RESULT, END)


def interleaved(min_kw=None, max_kw=None, between=None, carry=True, carrier='v_max_u32', min_op='v_min_u32'):
    """The census's min then max: the min chain up to its restore, the v_cndmask_b32_sdwa, the max chain's saveexec,
    the min's readlane, then the rest of the max chain. Defaults: both chains on v1, vT v2 shared, vP v3, sS s0 and s4,
    sD s1 and s5."""
    min_kw = dict(dict(vt=2, vp=3, ss=0, sd=1), **(min_kw or {}))
    max_kw = dict(dict(vt=2, vp=3, ss=4, sd=5), **(max_kw or {}))
    head = chain(min_op, tail_order=('restore',), **min_kw)
    tail = chain(carrier, **max_kw)
    later = [readlane(min_kw['sd'], min_kw['vt'])] if carry else []
    return head + (sdwa_line() if between is None else between) + tail[:1] + later + tail[1:]


# The census's min then max (vT shared), with the results and the v_cndmask_b32_sdwa exported.
PAIR_RESULTS = [move(4, 1), move(8, 5)]
PAIR = program(START_MUL, CMP, interleaved(), PAIR_RESULTS, export('v4', 'v6', 'v8'))
# The same with a vT of its own for the max chain.
PAIR_SPLIT = program(START_MUL, CMP, interleaved(max_kw=dict(vt=5)), PAIR_RESULTS, export('v4', 'v6', 'v8'))
# Min then max each read at once, back to back with their own saveexec (accepted: the OR shape for each).
PAIR_AT_ONCE = program(START_MUL, chain('v_min_u32'), chain('v_max_u32', vt=5, vp=3, ss=6, sd=7),
                       [move(4, 1), move(8, 7)], export('v4', 'v1', 'v8'))
# One set as the capture reports it per program: an or, then the min and max pair.
SET = program(START_MUL, CMP, chain('v_or_b32', vt=9, vp=3, ss=10, sd=11), interleaved(),
              [move(4, 1), move(8, 5), move(12, 11)], export('v4', 'v8', 'v12'))


def branch_program(op, into):
    """s_cbranch_execz at the top, landing on the chain's v_permlanex16_b32 (into) or on the export after it."""
    body = start(op) + chain(op) + RESULT
    upto = start(op) + chain(op)[:7] if into else body
    words_before = sum(len(w) for _t, w in upto) // 4
    label = [('s_cbranch_execz label_%04X' % (4 * (1 + words_before)), sopp(0x08, words_before))]
    return program(label, body, END)


NARROW_EXEC = [('s_and_saveexec_b32 s6, 0', sop1(S_AND_SAVEEXEC_B32, 6, 128))]
RESTORE_EXEC = [('s_mov_b32 exec_lo, s6', sop1(S_MOV_B32, EXEC_LO, 6))]


def branch_reduction(op, before_dispatch=(), result=END, pair=False, scc_vote=False):
    """Jump to the condition, then select a reduction or a zero result for the whole wave."""
    def jump(opcode, name, pc, target):
        return [('%s label_%04X' % (name, target), sopp(opcode, (target - pc - 4) // 4))]

    def size(part):
        return sum(len(data) for _, data in part)

    # 0xfffff < packed position iff y >= 16.
    compare = [('v_cmp_lt_u32_e32 vcc_lo, 0xfffff, v0',
                struct.pack('<II', 0x7c000000 | (0xc1 << 17) | 255, 0xfffff))]
    zero_regs = (4, 6, 8) if pair else (2, 4)
    zero = [('v_mov_b32_e32 v%d, 0' % reg, vop1_mov(reg, 128)) for reg in zero_regs]
    head = start(op)
    if op == 'v_and_b32':
        # Keep one bit set in every input so even a large wave's AND differs from the other arm's zero.
        head = head + [('v_or_b32_e32 v1, 0x%x, v1' % BRANCH_AND_BIT,
                        vop2(VOP2_OPS['v_or_b32'], 1, 255, 1) + struct.pack('<I', BRANCH_AND_BIT))]
    body = CMP + interleaved() + PAIR_RESULTS if pair else chain(op) + RESULT
    entry = size(head)
    reduce = entry + 4
    dispatch = reduce + size(body) + 4
    vote = dispatch + size(before_dispatch) + size(compare)
    join = vote + 4 + size(zero)
    opcode, name = (5, 's_cbranch_scc1') if scc_vote else (7, 's_cbranch_vccnz')
    return program(head, jump(2, 's_branch', entry, dispatch), body,
                   jump(2, 's_branch', dispatch - 4, join), before_dispatch, compare,
                   jump(opcode, name, vote, reduce), zero, result)


BRANCH_PAIR = branch_reduction('v_min_u32', result=export('v4', 'v6', 'v8'), pair=True)


def branch_into_later_readlane():
    """s_cbranch_execz at the top, landing on the min chain's later readlane inside the max chain."""
    body = START_MUL + CMP + interleaved() + PAIR_RESULTS
    upto = START_MUL + CMP + interleaved()[:11]
    assert upto[-1][0].startswith('s_or_saveexec_b32 s4')
    words_before = sum(len(w) for _t, w in upto) // 4
    label = [('s_cbranch_execz label_%04X' % (4 * (1 + words_before)), sopp(0x08, words_before))]
    return program(label, body, export('v4', 'v6', 'v8'))


def translate(words):
    try:
        return G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=PACKED).hlsl_ps(), None
    except G.Unsupported as reason:
        return None, str(reason)


# ---- 32-lane model of the source chain (the ISA, not the lowering) --------------------------------------------------
def apply(op, a, b):
    if op == 'v_or_b32':
        return a | b
    if op == 'v_min_u32':
        return min(a, b)
    if op == 'v_max_u32':
        return max(a, b)
    if op == 'v_max_i32':
        return max(a, b, key=lambda v: v - (1 << 32) if v & 0x80000000 else v)
    if op == 'v_min_i32':
        return min(a, b, key=lambda v: v - (1 << 32) if v & 0x80000000 else v)
    if op == 'v_and_b32':
        return a & b
    if op == 'v_xor_b32':
        return a ^ b
    if op == 'v_add_nc_u32':
        return (a + b) & MASK
    raise AssertionError(op)


def dpp_source(ctrl, lane):
    row, within = lane & ~15, lane & 15
    if ctrl.startswith('quad_perm:'):
        perm = [int(c) for c in re.findall(r'\d', ctrl)]
        return (lane & ~3) | perm[lane & 3]
    if ctrl == 'row_half_mirror':
        return row | (within & 8) | (7 - (within & 7))
    if ctrl == 'row_mirror':
        return row | (15 - within)
    if ctrl == 'row_shr:1':
        return lane - 1 if within >= 1 else None   # no source: bound_ctrl decides
    raise AssertionError(ctrl)


SELECT = {'BYTE_0': (0, 0xff), 'BYTE_1': (8, 0xff), 'BYTE_2': (16, 0xff), 'BYTE_3': (24, 0xff), 'WORD_0': (0, 0xffff),
          'WORD_1': (16, 0xffff), 'DWORD': (0, MASK)}


def run_model(source, inputs, exec_mask, rng):
    """Run the source lines on 32 lanes; v1 holds the inputs, every other VGPR starts with arbitrary words, as
    registers a lane never wrote do. Returns the SGPRs."""
    v = collections.defaultdict(lambda: [rng.getrandbits(32) for _ in range(32)])
    v[1] = list(inputs)
    s, exec_lo, vcc = {}, exec_mask, 0

    def reg(tok):
        return v[int(tok[1:])]

    for line in source.splitlines():
        name, ops, mods = G.parse(line)
        base = re.sub(r'_e(32|64)$', '', name)
        if base in ('v_mul_u32_u24', 'v_lshlrev_b32', 'exp', 's_endpgm') or (base == 'v_mov_b32' and ops[1][0] == 's'):
            continue
        if base == 's_or_saveexec_b32':
            s[int(ops[0][1:])], exec_lo = exec_lo, MASK
        elif base == 's_mov_b32' and ops[0] == 'exec_lo':
            exec_lo = s[int(ops[1][1:])]
        elif base == 'v_mov_b32':
            value = reg(ops[1]) if ops[1].startswith('v') else [int(ops[1], 0) & MASK] * 32
            dst = reg(ops[0])
            v[int(ops[0][1:])] = [value[i] if exec_lo >> i & 1 else dst[i] for i in range(32)]
        elif base == 'v_cmp_gt_u32':
            a, b = reg(ops[1]), reg(ops[2])
            vcc = sum(1 << i for i in range(32) if exec_lo >> i & 1 and a[i] > b[i])
        elif base == 'v_cndmask_b32_sdwa':
            sel = dict(m.split(':') for m in mods)
            assert sel['dst_sel'] == 'DWORD' and ops[3] == 'vcc_lo', line
            pick = [(lambda w, k=k: (w >> SELECT[sel[k]][0]) & SELECT[sel[k]][1]) for k in ('src0_sel', 'src1_sel')]
            a, b, dst = reg(ops[1]), reg(ops[2]), reg(ops[0])
            v[int(ops[0][1:])] = [(pick[1](b[i]) if vcc >> i & 1 else pick[0](a[i])) if exec_lo >> i & 1 else dst[i]
                                  for i in range(32)]
        elif base == 'v_cndmask_b32':
            mask, a, b, dst = s[int(ops[3][1:])], int(ops[1], 0) & MASK, reg(ops[2]), reg(ops[0])
            v[int(ops[0][1:])] = [(b[i] if mask >> i & 1 else a) if exec_lo >> i & 1 else dst[i] for i in range(32)]
        elif base.endswith('_dpp'):
            ctrl = [m for m in mods if not m.startswith(('row_mask:', 'bank_mask:', 'bound_ctrl:', 'fi:'))][0]
            row_mask = int([m for m in mods if m.startswith('row_mask:')][0][9:], 0)
            dst, src0, src1 = int(ops[0][1:]), reg(ops[1]), reg(ops[2])
            out = list(v[dst])
            for i in range(32):
                if not (exec_lo >> i & 1 and row_mask >> (i // 16) & 1):
                    continue
                k = dpp_source(ctrl, i)
                if k is not None and (exec_lo >> k & 1 or 'fi:1' in mods):
                    word_in = src0[k]          # fi:1 reads a disabled source lane too
                elif 'bound_ctrl:1' in mods:
                    word_in = 0
                else:
                    continue                   # the lane keeps its old value
                out[i] = apply(base[:-4], word_in, src1[i])
            v[dst] = out
        elif base == 'v_permlanex16_b32':
            sel, src = (int(ops[2], 0), int(ops[3], 0)), reg(ops[1])
            out = list(reg(ops[0]))
            for i in range(32):
                if exec_lo >> i & 1:
                    within = i & 15
                    nibble = ((sel[0] >> (4 * within)) if within < 8 else (sel[1] >> (4 * (within - 8)))) & 15
                    out[i] = src[(i ^ 16) & ~15 | nibble]
            v[int(ops[0][1:])] = out
        elif base in IDENTITY:
            a, b, dst = reg(ops[1]), reg(ops[2]), reg(ops[0])
            v[int(ops[0][1:])] = [apply(base, a[i], b[i]) if exec_lo >> i & 1 else dst[i] for i in range(32)]
        elif base == 'v_readlane_b32':
            s[int(ops[0][1:])] = reg(ops[1])[int(ops[2], 0)]
        else:
            raise AssertionError(line)
    return s


def reduction(op, inputs, exec_mask):
    value = IDENTITY[op]
    for i in range(32):
        if exec_mask >> i & 1:
            value = apply(op, value, inputs[i])
    return value


def model_says_reduction(source, expect, trials=300, seed=2125):
    """expect: [(SGPR, op)]: each SGPR must be op over the lanes active before the program."""
    rng = random.Random(seed)
    masks = [MASK, 1, 1 << 31, 0x0000ffff, 0xffff0000, 0x80000001] + [rng.getrandbits(32) | 1 for _ in range(trials)]
    for mask in masks:
        inputs = [rng.getrandbits(32) for _ in range(32)]
        s = run_model(source, inputs, mask, rng)
        if any(s.get(sd) != reduction(op, inputs, mask) for sd, op in expect):
            return False
    return True


# ---- controls ---------------------------------------------------------------------------------------------------------
INTRINSIC = {'v_or_b32': 'WaveActiveBitOr(v1)', 'v_min_u32': 'WaveActiveMin(v1)', 'v_max_u32': 'WaveActiveMax(v1)',
             'v_and_b32': 'WaveActiveBitAnd(v1)', 'v_xor_b32': 'WaveActiveBitXor(v1)', 'v_add_nc_u32': 'WaveActiveSum(v1)',
             'v_min_i32': 'asuint(WaveActiveMin(asint(v1)))', 'v_max_i32': 'asuint(WaveActiveMax(asint(v1)))'}
PAIR_EXPECT = [(1, 'v_min_u32'), (5, 'v_max_u32')]


def refusals():
    """(label, program, refusal fragment, model: True a conservative refusal, False not the reduction, None not run,
    or a list of (SGPR, op) to run the model with when the result is not s1)."""
    op = 'v_max_u32'
    full = lambda **kw: program(START_MUL, chain(op, **kw), RESULT, END)  # noqa: E731
    pair = lambda **kw: program(START_MUL, CMP, interleaved(**kw), PAIR_RESULTS, export('v4', 'v6', 'v8'))  # noqa: E731
    yield ('wave64 (s_or_saveexec_b64 s[0:1], -1)', full(wave='b64'), 'wave64', None)
    yield ('no identity fill', full(fill=None), 'identity fill', False)
    yield ('the input copied without the identity', full(fill='copy'), 'identity fill', False)
    yield ('a fill with another constant (1 for v_max_u32)', full(identity=1), 'identity fill', None)
    for other in OPS:   # a constant that is no operation's identity, for each operation
        yield ('a fill with another constant (5 for %s)' % other,
               program(start(other), chain(other, identity=5), RESULT, END), 'identity fill', None)
    yield ('a fill under another saved mask (s5)', full(fill_mask=5), 'identity fill', None)
    yield ('wrong identity (-1 for v_max_u32)', full(identity=MASK), 'identity', False)
    yield ('wrong identity (0 for v_min_u32)', program(START_MUL, chain('v_min_u32', identity=0), RESULT, END),
           'identity', [(1, 'v_min_u32')])
    yield ('wrong identity (-1 for v_or_b32)', program(START_BIT, chain('v_or_b32', identity=MASK), RESULT, END),
           'identity', [(1, 'v_or_b32')])
    yield ('wrong identity (0 for the interleaved v_min_u32)', pair(min_kw=dict(identity=0)), 'identity',
           [(1, 'v_min_u32')])
    yield ('row_mirror step removed', full(steps=STEPS[:3]), 'missing step 4 of 4', False)
    yield ('row_half_mirror step removed', full(steps=STEPS[:2] + STEPS[3:]), 'missing step 4 of 4', False)
    yield ('first quad step removed', full(steps=STEPS[1:]), 'missing step 4 of 4', False)
    yield ('a quad step repeated in place of the half mirror', full(steps=(STEPS[0], STEPS[1], STEPS[1], STEPS[3])),
           'do not bring every lane', False)
    yield ('the half mirror twice', full(steps=(STEPS[0], STEPS[1], STEPS[2], STEPS[2])), 'do not bring every lane', False)
    yield ('row_shr:1 in place of row_mirror', full(steps=STEPS[:3] + ('row_shr:1',)), 'row_shr:1 in place of', False)
    yield ('mixed operations (one v_min_u32 step)', full(step_ops={2: 'v_min_u32'}), 'mixes', False)
    yield ('v_max_i32 (not in the census)', program(START_MUL, chain('v_max_i32', identity=0), RESULT, END),
           'only v_or_b32', None)
    yield ('a float step (v_max_f32)', full(step_ops={0: 'v_max_f32'}), 'only v_or_b32', None)
    yield ('readlane of lane 30 (conservative)', full(lane=30), 'lane 30', True)
    yield ('permlanex16 lane selects 1, 0 (conservative)', full(perm=(1, 0)), 'lane selects', True)
    yield ('permlanex16 with op_sel (fi) (conservative)', full(perm_opsel=1), 'lane selects', True)
    yield ('readlane before the restore (conservative)', full(tail_order=('readlane', 'restore')),
           'EXEC restored before the readlane', True)
    yield ('EXEC restored from another SGPR', full(restore_reg=5), 'in place of s_mov_b32 exec_lo, s0', None)
    yield ('partial row mask on a step', full(row=0x1), 'partial row or bank mask', False)
    yield ('branch target inside the chain', branch_program(op, True), 'branch target', None)
    yield ('branch target on the later readlane of the v_min_u32 chain', branch_into_later_readlane(),
           'branch target', None)
    yield ('after s_wqm exec_lo, exec_lo', program(WQM, START_MUL, chain(op), RESULT, END), 'helper lanes', None)
    yield ('the permlane VGPR read after the chain', program(START_MUL, chain(op), RESULT, END_READS_VP),
           'read after the chain', None)
    # The interleaved min then max of the census, each piece missing or changed.
    yield ('the later readlane of the v_min_u32 chain missing', pair(carry=False),
           'in place of v_readlane_b32 sN, v2, 31', [(1, 'v_min_u32')])
    yield ('vT rewritten between the restore and the later readlane',
           pair(between=sdwa_line(vdst=2)), 'rewrites v2 before the readlane', [(1, 'v_min_u32')])
    # Only the v_cndmask_b32_sdwa reads v0 after the min chain (the export does not), so the scan must start at the
    # restore, not after the later readlane.
    yield ('the permlane VGPR read by the v_cndmask_b32_sdwa (vP = v0)',
           program(START_MUL, CMP, interleaved(min_kw=dict(vp=0)), PAIR_RESULTS, export('v4', 'v6', 'v8', 'v8')),
           'read after the chain by v_cndmask_b32_sdwa', None)
    yield ('another instruction in place of the v_cndmask_b32_sdwa (conservative)', pair(between=KILL_VP),
           'in place of v_readlane_b32 sN, v2, 31', (True, PAIR_EXPECT))
    yield ('a v_or_b32 chain carrying the later readlane (conservative)', pair(carrier='v_or_b32'),
           'only a v_max_u32 chain', (True, [(1, 'v_min_u32'), (5, 'v_or_b32')]))
    yield ('a v_max_u32 chain that reads its lane later (conservative)', pair(min_op='v_max_u32'),
           'only the v_min_u32 chain reads its lane later', (True, [(1, 'v_max_u32'), (5, 'v_max_u32')]))
    yield ('the later readlane overwrites the EXEC saved by the v_max_u32 chain', pair(min_kw=dict(sd=4)),
           'overwrites s4', [(4, 'v_min_u32'), (5, 'v_max_u32')])
    lone = START_MUL + CMP + chain('v_min_u32', tail_order=('restore',)) + sdwa_line() + \
        [saveexec(4), readlane(1, 2), ('s_mov_b32 exec_lo, s4', sop1(S_MOV_B32, EXEC_LO, 4))] + RESULT
    yield ('the later readlane after an s_or_saveexec_b32 that starts no chain', program(lone, END),
           'not carried by a v_max_u32 chain', None)


def body_of(text):
    return text.split('PSOut o = (PSOut)0;', 1)[1]


def model_run(label, source, model, default):
    """model: None (not run), True (a conservative refusal: still reduces), False (not the reduction), a list of
    (SGPR, op) (not the reduction, checked on those), or (flag, list)."""
    if model is None:
        return
    if isinstance(model, list):
        model = (False, model)
    want, expect = model if isinstance(model, tuple) else (model, default)
    got = model_says_reduction(source, expect)
    check('%s: the 32-lane model %s' % (label, 'still reduces' if want else 'is not the reduction'), got == want)


def controls():
    fills = G.Stage.WAVE_REDUCE_FILLS
    census = G.Stage.__dict__['wave_reduce_cndmask_fill']
    check('the identity fill table holds the census form only', fills == (census,), str(fills))
    for op in OPS:
        source, words = fixture(op)
        decoded = [text for _o, _s, _n, text in G.decode(words)]
        check('%s fixture decodes to its source' % op, decoded == source.splitlines(), str(decoded))
        check('%s chain is the reduction on the 32-lane model' % op, model_says_reduction(source, [(1, op)]))
        text, reason = translate(words)
        check('%s translates' % op, text is not None, str(reason))
        if text is None:
            continue
        body = body_of(text)
        check('%s lowers to one %s' % (op, INTRINSIC[op]),
              body.count('Wave') == 1 and 'v2 = %s;' % INTRINSIC[op] in body and 's1 = v2;' in body
              and 'QuadRead' not in body and 'exec &' not in body, body)
        check('%s keeps the saved EXEC and SCC of the chain' % op, 's0 = exec;' in body and 'scc = true;' in body, body)
    # The census's interleaved min then max, and the full set of one program.
    for label, (source, words), expect, order in (
            ('min then max, interleaved, vT shared', PAIR, PAIR_EXPECT,
             ('v2 = WaveActiveMin(v1);', 'v6 = ', 's1 = v2;', 'v2 = WaveActiveMax(v1);', 's5 = v2;')),
            ('min then max, interleaved, vT apart', PAIR_SPLIT, PAIR_EXPECT,
             ('v2 = WaveActiveMin(v1);', 'v6 = ', 's1 = v2;', 'v5 = WaveActiveMax(v1);', 's5 = v5;')),
            ('min then max, each read at once', PAIR_AT_ONCE, [(1, 'v_min_u32'), (7, 'v_max_u32')],
             ('v2 = WaveActiveMin(v1);', 's1 = v2;', 'v5 = WaveActiveMax(v1);', 's7 = v5;')),
            ('or, then min and max interleaved', SET, PAIR_EXPECT + [(11, 'v_or_b32')],
             ('v9 = WaveActiveBitOr(v1);', 's11 = v9;', 'v2 = WaveActiveMin(v1);', 'v6 = ', 's1 = v2;',
              'v2 = WaveActiveMax(v1);', 's5 = v2;'))):
        decoded = [text for _o, _s, _n, text in G.decode(words)]
        check('%s fixture decodes to its source' % label, decoded == source.splitlines(), str(decoded))
        check('%s chains are the reductions on the 32-lane model' % label, model_says_reduction(source, expect))
        text, reason = translate(words)
        check('%s translates' % label, text is not None, str(reason))
        if text is None:
            continue
        body = body_of(text)
        at = [body.find('\n    ' + piece) for piece in order]   # statements, not the declarations
        check('%s: one wave operation per chain, in program order' % label,
              -1 not in at and at == sorted(at) and body.count('WaveActive') == len(expect), body)
    # The order that matters: the min's readlane after the max fill reads the max when vT is shared.
    lines = PAIR[0].splitlines()
    late = lines[:]
    moved = late.pop(late.index('v_readlane_b32 s1, v2, 31'))
    late.insert(late.index('v_cndmask_b32_e64 v2, 0, v1, s4') + 1, moved)
    check('planted error: the min readlane after the max fill is not the min on the 32-lane model',
          not model_says_reduction('\n'.join(late) + '\n', [(1, 'v_min_u32')]))
    for label, variant in (('bound_ctrl on every step', chain('v_max_u32', bound_ctrl=True)),
                           ('steps without fi', chain('v_max_u32', fi=False)),
                           ('the combine with the sources the other way round', chain('v_max_u32', combine='e32')),
                           ('the combine in VOP3', chain('v_max_u32', combine='e64')),
                           ('the mirror steps in the other order', chain('v_max_u32', steps=STEPS[:2] + STEPS[3:1:-1])),
                           ('the quad steps in the other order', chain('v_max_u32', steps=STEPS[1::-1] + STEPS[2:])),
                           ('the half mirror first', chain('v_max_u32', steps=(STEPS[2], STEPS[0], STEPS[1], STEPS[3]))),
                           ('a reversing quad step after the swap', chain('v_max_u32', steps=('quad_perm:[1,0,3,2]', 'quad_perm:[3,2,1,0]', STEPS[2], STEPS[3])))):
        source, words = program(START_MUL, variant, RESULT, END)
        text, reason = translate(words)
        check('accepts %s' % label, text is not None and 'v2 = WaveActiveMax(v1);' in text, str(reason))
        check('%s is the reduction on the 32-lane model' % label, model_says_reduction(source, [(1, 'v_max_u32')]))
    _source, words = program(START_MUL, chain('v_max_u32'), RESULT, KILL_VP, END_READS_VP)
    text, reason = translate(words)
    check('accepts the permlane VGPR rewritten before it is read', text is not None and 'v3 = 0' in text, str(reason))
    # A program with an execz branch ahead of the chain: the branch is decided over the host wave (every thread reaches the
    # chain together), the chain keeps its lowering.
    text, reason = translate(branch_program('v_max_u32', False)[1])
    check('a forward execz branch is a wave vote and the chain still lowers', text is not None and
          'WaveActiveAnyTrue((exec & 1u) != 0u)' in text and 'WaveActiveMax(' in text, str(reason))
    for op in OPS:
        text, reason = translate(branch_reduction(op)[1])
        check('accepts wave-uniform if/else with a backward edge for ' + op,
              text is not None and INTRINSIC[op] in text and 'WaveActiveAnyTrue((vcc & 1u) != 0u)' in text
              and '!WaveActiveAnyTrue' not in text, str(reason))
    text, reason = translate(BRANCH_PAIR[1])
    check('accepts the interleaved min/max with shared scratch in the reduction arm', text is not None and
          'WaveActiveMin(v1)' in text and 'WaveActiveMax(v1)' in text, str(reason))
    collected = []
    G.Pipeline(VS_WORDS, BRANCH_PAIR[1], root_blob(), ps_hw_inputs=PACKED, refusals=collected)
    check('the branched reduction also succeeds in refusal collection mode', not collected, str(collected))
    # Narrow EXEC, then restore the saved entry mask before the branch back.
    text, reason = translate(branch_reduction('v_max_u32', NARROW_EXEC + RESTORE_EXEC)[1])
    check('accepts a proven EXEC restore on the incoming path', text is not None and
          'v2 = WaveActiveMax(v1);' in text, str(reason))
    for name, before, result, fragment in (
        ('changed EXEC at the reduction', NARROW_EXEC, END, 'EXEC is not proven'),
        ('saved EXEC overwritten before the restore',
         NARROW_EXEC + [('s_mov_b32 s6, 0', sop1(S_MOV_B32, 6, 128))] + RESTORE_EXEC,
         END, 'EXEC is not proven'),
        ('WQM can enable helpers', WQM, END, 'EXEC is not proven'),
        ('scratch read across the join', [], END_READS_VP, 'read after the chain'),
    ):
        text, reason = translate(branch_reduction('v_max_u32', before, result)[1])
        check('refuses ' + name, text is None and reason is not None and fragment in reason, str(reason))
    # Saveexec leaves an SCC that cannot be used as a wave-uniform condition in the one-lane model.
    text, reason = translate(branch_reduction('v_max_u32', NARROW_EXEC + RESTORE_EXEC, scc_vote=True)[1])
    check('refuses a backward SCC branch without a uniform condition', text is None and reason is not None and
          'neither wave-uniform nor an EXEC / VCC vote' in reason, str(reason))
    for label, (source, words), fragment, model in refusals():
        text, reason = translate(words)
        check('refuses %s, by name' % label, text is None and reason is not None and 'wave reduction' in reason
              and fragment in reason, str(reason))
        model_run(label, source, model, [(1, 'v_max_u32')])
    # Not a chain: s_or_saveexec_b32 with -1 and no mirror or permlane after it keeps its own handler.
    _source, words = program(START_MUL, [saveexec(0)], export('v1', 'v1', 'v1'))
    text, reason = translate(words)
    check('s_or_saveexec_b32 s0, -1 outside a chain keeps its handler', reason is None or 'wave reduction' not in reason,
          str(reason))
    # A vertex stage, through a Stage probe: the hook needs only the instruction list and the program counter.
    probe = G.Stage.__new__(G.Stage)
    probe.lines, probe.sym, probe.blocks, probe.masks = [], {}, [], set()
    probe.used_s, probe.used_v, probe.uniform_v = set(), set(), set()
    probe.kind, probe.exec_known, probe.has_branches, probe.wqm_exec = 'vs', True, False, False
    probe.ins = G.decode(fixture('v_max_u32')[1])
    probe.pc = probe.ins[1][0]
    try:
        probe.lower('s_or_saveexec_b32', 's_or_saveexec_b32 s0, -1', ['s0', '-1'], [])
        reason = None
    except G.Unsupported as error:
        reason = str(error)
    check('refuses a chain in a vertex stage, by name', reason is not None and 'outside a pixel stage' in reason,
          str(reason))


# ---- draw -------------------------------------------------------------------------------------------------------------
OBSERVE = ('    o.mrt0.w = asfloat(WaveGetLaneIndex() | (WaveGetLaneCount() << 8) |'
           ' (WaveReadLaneFirst(uint(i.pos.x) | (uint(i.pos.y) << 5)) << 16));\n    return o;')


def observed(text):
    assert text.count('    return o;') == 1
    return text.replace('    return o;', OBSERVE)


def late_readlane(text):
    """Planted error: the min chain's readlane lowered after the max chain's result (vT shared)."""
    lines = text.split('\n')
    moved = [ln for ln in lines if ln.strip() == 's1 = v2;']
    if len(moved) != 1:
        return text
    lines.remove(moved[0])
    at = [k for k, ln in enumerate(lines) if ln.strip() == 'v2 = WaveActiveMax(v1);']
    return '\n'.join(lines[:at[0] + 1] + moved + lines[at[0] + 1:]) if len(at) == 1 else text


def drop_sdwa(text):
    """Planted error: the v_cndmask_b32_sdwa between the chains lowered as nothing (v6 keeps its old value)."""
    lines = text.split('\n')
    kept = [ln for ln in lines if not ln.strip().startswith('v6 = ')]
    return '\n'.join(kept) if len(kept) == len(lines) - 1 else text


OTHER = {'v_or_b32': 'v_max_u32', 'v_min_u32': 'v_max_u32', 'v_max_u32': 'v_min_u32', 'v_and_b32': 'v_or_b32',
         'v_xor_b32': 'v_or_b32', 'v_add_nc_u32': 'v_xor_b32', 'v_min_i32': 'v_max_i32', 'v_max_i32': 'v_min_i32'}
SIGNED = {'v_min_i32': 'v_min_u32', 'v_max_i32': 'v_max_u32'}


def sign_swapped(text, op):
    """The other signedness of a min or max; text unchanged for an operation that has none."""
    if op in SIGNED:
        return text.replace(INTRINSIC[op], INTRINSIC[SIGNED[op]])
    if op in ('v_min_u32', 'v_max_u32'):
        return text.replace(INTRINSIC[op], INTRINSIC[op].replace('(v1)', '(int(v1))').replace('Wave', 'uint(Wave', 1) + ')')
    return text

MUTATIONS = {
    'another operation': lambda text, op: text.replace(INTRINSIC[op], INTRINSIC[OTHER[op]]),
    'signed for unsigned': lambda text, op: sign_swapped(text, op),
    "the lane's own value": lambda text, op: text.replace(INTRINSIC[op], 'v1'),
    "the first lane's value": lambda text, op: text.replace(INTRINSIC[op], 'WaveReadLaneFirst(v1)'),
}
PAIR_MUTATIONS = {
    'the min readlane after the max': late_readlane,
    'min and max swapped': lambda text: text.replace('WaveActiveMin', 'WaveActiveTmp').replace(
        'WaveActiveMax', 'WaveActiveMin').replace('WaveActiveTmp', 'WaveActiveMax'),
    'the v_cndmask_b32_sdwa dropped': drop_sdwa,
}


def single(op):
    """Channels of a one-chain draw: the readlane result, the input, vT after the chain."""
    return op, (('reduce', op), ('input', None), ('reduce', op))


def pair_sdwa(x, y):
    value = word('v_min_u32', x, y)
    return y if value > (x | (y << 16)) else value & 0xffff


DRAWS = [(op, fixture(op)[1]) + single(op) for op in OPS] + [
    ('min then max, interleaved', PAIR[1], 'v_min_u32',
     (('reduce', 'v_min_u32'), ('lane', pair_sdwa), ('reduce', 'v_max_u32')))]

# The recognition simulates the steps, so another order of the same four steps is drawn too (label -> operation).
REORDERED = {'v_max_u32, half mirror first': ('v_max_u32', (STEPS[2], STEPS[0], STEPS[1], STEPS[3])),
             'v_add_nc_u32, quad steps swapped then reversed': ('v_add_nc_u32', ('quad_perm:[1,0,3,2]', 'quad_perm:[3,2,1,0]', STEPS[2], STEPS[3]))}
DRAWS += [(label, program(start(op), chain(op, steps=steps), RESULT, END)[1]) + single(op) for label, (op, steps) in REORDERED.items()]


BRANCHED = {'branched ' + op: (op, []) for op in OPS}
BRANCHED['branched max with EXEC restore'] = ('v_max_u32', NARROW_EXEC + RESTORE_EXEC)
DRAWS += [(label, branch_reduction(op, before)[1], op, (('branch', op), ('input', None), ('branch', op)))
          for label, (op, before) in BRANCHED.items()]
BRANCH_PAIR_LABEL = 'branched min then max, interleaved'
DRAWS.append((BRANCH_PAIR_LABEL, BRANCH_PAIR[1], 'v_min_u32',
              (('branch', 'v_min_u32'), ('branch_lane', pair_sdwa), ('branch', 'v_max_u32'))))


def verify(input_op, channels, words):
    """Group the readback by desktop wave and compare with the CPU reference. Returns (errors, facts)."""
    pixels = struct.unpack('<%dI' % (len(words) // 4), words)
    groups, errors = collections.defaultdict(list), []
    for y in range(SIZE):
        for x in range(SIZE):
            p = pixels[(y * SIZE + x) * 4:(y * SIZE + x) * 4 + 4]
            if p == (CLEAR,) * 4:
                continue
            groups[p[3] >> 16].append((x, y, p))
    counts, arms = set(), set()
    branched = any(kind == 'branch' for kind, _ in channels)
    input_bits = BRANCH_AND_BIT if branched and input_op == 'v_and_b32' else 0
    for key, members in groups.items():
        lanes = [p[3] & 0xff for _x, _y, p in members]
        counts.update((p[3] >> 8) & 0xff for _x, _y, p in members)
        if len(set(lanes)) != len(lanes) or (key & 31, key >> 5) not in {(x, y) for x, y, _p in members}:
            errors.append('wave %d: lanes %s do not name one wave' % (key, lanes))
        taken = any(y >= 16 for _x, y, _p in members)
        arms.add(taken)
        reduced = {}
        for kind, op in channels:
            if kind in ('reduce', 'branch'):
                value = IDENTITY[op]
                for x, y, _p in members:
                    value = apply(op, value, word(input_op, x, y) | input_bits)
                reduced[kind, op] = 0 if kind == 'branch' and not taken else value
        for x, y, p in members:
            expected = []
            for kind, arg in channels:
                if kind in ('reduce', 'branch'):
                    expected.append(reduced[kind, arg])
                elif kind == 'input':
                    expected.append(word(input_op, x, y) | input_bits)
                else:
                    expected.append(0 if kind == 'branch_lane' and not taken else arg(x, y))
            if list(p[:3]) != expected:
                errors.append('pixel (%d,%d): %s, reference %s over %d pixels of its wave'
                              % (x, y, ' / '.join('%08x' % w for w in p[:3]),
                                 ' / '.join('%08x' % w for w in expected), len(members)))
    covered = {(x, y) for members in groups.values() for x, y, _p in members}
    quads = {(x // 2, y // 2) for x, y in covered}
    partial_quads = sum(1 for qx, qy in quads
                        if sum((2 * qx + i, 2 * qy + j) in covered for i in (0, 1) for j in (0, 1)) < 4)
    if branched and arms != {False, True}:
        errors.append('the draw did not exercise both branch arms')
    if len(counts) != 1:
        errors.append('lane counts %s' % sorted(counts))
    lane_count = min(counts) if counts else 0
    facts = ('covered=%d waves=%d lane_count=%s partial_waves=%d partial_quads=%d'
             % (len(covered), len(groups), '/'.join(map(str, sorted(counts))),
                sum(1 for members in groups.values() if len(members) < lane_count), partial_quads))
    if partial_quads == 0 or len(covered) < 100:
        errors.append('the draw does not exercise partly covered quads: ' + facts)
    return errors, facts


def dump(renderer, adapter, vs, ps, out):
    run = subprocess.run([renderer, adapter, '--startup-pixel-dump', str(vs), str(ps), str(SIZE), str(SIZE), str(out)],
                         capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(77)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', TRIANGLE_VS, 'vs_dxc', 'vs_6_0', directory)
        for label, words, input_op, channels in DRAWS:
            text, reason = translate(words)
            if text is None:
                check('%s translates for the draw' % label, False, str(reason))
                continue
            branched = label in BRANCHED or label == BRANCH_PAIR_LABEL
            if label == BRANCH_PAIR_LABEL:
                hows = [('min/max swapped', PAIR_MUTATIONS['min and max swapped'])]
            elif label in BRANCHED:
                op, _ = BRANCHED[label]
                hows = [('lane-local instead of a wave value', lambda t: t.replace(INTRINSIC[op], 'v1')),
                        ('reversed wave vote', lambda t: t.replace('WaveActiveAnyTrue((vcc & 1u) != 0u)',
                                                                  '!WaveActiveAnyTrue((vcc & 1u) != 0u)'))]
            elif label in OPS or label in REORDERED:
                label_op = REORDERED[label][0] if label in REORDERED else label
                hows = [(how, (lambda t, f=f: f(t, label_op))) for how, f in MUTATIONS.items()]
            else:
                hows = list(PAIR_MUTATIONS.items())
            for how, mutate in [(None, None)] + hows:
                source = observed(text if how is None else mutate(text))
                if how is not None and source == observed(text):
                    if label not in ('v_min_u32', 'v_max_u32', 'v_min_i32', 'v_max_i32') and how == 'signed for unsigned':
                        continue   # only min and max have a signedness
                    check('%s %s %s: the planted error applies' % (adapter, label, how), False)
                    continue
                tag = re.sub(r'\W+', '_', label + ('' if how is None else '_' + how))
                ps = compile_hlsl(dxc, 'dxc', source, 'ps_' + tag, 'ps_6_0', directory)
                out = directory / ('%s.bin' % tag)
                run = dump(renderer, adapter, vs, ps, out)
                if run.returncode != 0 or not out.exists():
                    check('%s %s %s: draw' % (adapter, label, how or 'lowering'), False, run.stdout + run.stderr)
                    continue
                errors, facts = verify(input_op, channels, out.read_bytes())
                clean = 'debug_layer=enabled' not in run.stdout or 'debug_errors=0 debug_warnings=0' in run.stdout
                if how is None:
                    print('  ' + ' '.join(ln for ln in run.stdout.splitlines() if ln.startswith(
                        ('backend=', 'adapter=', 'driver=', 'debug_errors='))))
                    print('  %s: %s' % (label, facts))
                    check('%s %s: every covered pixel matches its desktop wave' % (adapter, label),
                          not errors and clean, '; '.join(errors[:3]) + ' ' + run.stdout)
                else:
                    mismatches = [err for err in errors if err.startswith('pixel (')] if branched else errors
                    check('%s %s %s mutation fails against the reference: %s (%d mismatches)' % (
                        adapter, label, how, mismatches[0] if mismatches else '', len(mismatches)), bool(mismatches))


def assemble_object(llvm_mc, source):
    """The .text bytes of an object assembled by llvm-mc. -show-encoding prints a branch to a label with its offset
    still a fixup ([A,A,0x88,0xbf]); the object has it resolved."""
    with tempfile.TemporaryDirectory() as tmp:
        src, obj = Path(tmp) / 'a.s', Path(tmp) / 'a.o'
        src.write_text(source)
        subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-filetype=obj', str(src), '-o', str(obj)],
                       check=True)
        elf = obj.read_bytes()
    # ELF64 little endian: section headers at e_shoff, e_shentsize bytes each; e_shstrndx names them.
    shoff, = struct.unpack_from('<Q', elf, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', elf, 0x3a)
    sections = [struct.unpack_from('<IIQQQQ', elf, shoff + k * shentsize) for k in range(shnum)]
    names = sections[shstrndx][4]
    for name, _type, _flags, _addr, offset, size in sections:
        if elf[names + name:elf.index(b'\0', names + name)] == b'.text':
            return elf[offset:offset + size]
    raise AssertionError('no .text section in the object llvm-mc wrote')


def check_binary(llvm_mc):
    programs = [('%s fixture' % op, fixture(op)) for op in OPS]
    programs += [('min then max, interleaved, vT shared', PAIR), ('min then max, interleaved, vT apart', PAIR_SPLIT),
                 ('min then max, each read at once', PAIR_AT_ONCE), ('or, then min and max interleaved', SET)]
    programs += [(label, prog) for label, prog, _f, _m in refusals()]
    programs += [('bound_ctrl variant', program(START_MUL, chain('v_max_u32', bound_ctrl=True), RESULT, END)),
                 ('no fi variant', program(START_MUL, chain('v_max_u32', fi=False), RESULT, END)),
                 ('e32 combine variant', program(START_MUL, chain('v_max_u32', combine='e32'), RESULT, END)),
                 ('VOP3 combine variant', program(START_MUL, chain('v_max_u32', combine='e64'), RESULT, END)),
                 ('rewritten permlane VGPR', program(START_MUL, chain('v_max_u32'), RESULT, KILL_VP, END_READS_VP))]
    programs += [(label, branch_reduction(op, before)) for label, (op, before) in BRANCHED.items()]
    programs.append((BRANCH_PAIR_LABEL, BRANCH_PAIR))
    for label, (source, words) in programs:
        if 'label_' not in source:
            built = assemble(llvm_mc, source)
        else:  # a branch: llvm-mc needs the label defined, at the word the fixture names
            targets = {int(label, 16) for label in re.findall(r'label_([0-9A-F]{4})', source)}
            out, offset = [], 0
            for text, size in zip(source.splitlines(), [n for _o, n, _name, _t in G.decode(words)]):
                if offset in targets:
                    out.append('L%04X:' % offset)
                out.append(text.replace('label_', 'L'))
                offset += size
            built = assemble_object(llvm_mc, '\n'.join(out) + '\n')
        check('llvm-mc assembles the program "%s" to the same words' % label, built == words,
              built.hex() + ' / ' + words.hex())


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
