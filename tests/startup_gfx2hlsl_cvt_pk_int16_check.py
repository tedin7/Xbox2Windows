"""Saturating 32-to-16-bit integer packs (v_cvt_pk_u16_u32, v_cvt_pk_i16_i32) in tools/startup/gfx2hlsl.py (#1723).

  (no arguments)                       translation, CPU boundary-grid evaluation, low16_only / uniformity / compute
                                       provenance, encoding-guard and refusal controls, pure Python
  binary <llvm-mc>                     reassemble every fixture and instruction spelling and compare with recorded words
  decode <libLLVM>                     disassemble the VOP3A words with LLVM's C API and compare with disasm_rdna2.py
  draw-snorm <dxc> <offscreen_check> warp
                                       read raw R16_UNORM words from a compressed signed-pack export
  draw-unorm <dxc> <offscreen_check> warp
                                       draw normalized packs and require the truncation mutant to fail
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pairs with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw 16 input pairs per opcode and verify four mutated HLSL controls per opcode

AMD RDNA 2 ISA reference (VOP3A opcodes 0x36a / 874 and 0x36b / 875) places the narrowed src0 in bits 15:0 and src1
in bits 31:16, with saturation bounds [0, 65535] (unsigned) and [-32768, 32767] (signed) matching Mesa's NIR
pack_uvec2_to_uint / pack_ivec2_to_uint selected in ACO and the pixel_aot lowering (#779). Both instructions exist
only in the VOP3A encoding on gfx10.
"""
import math
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gpu_case_runner  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import VS_WORDS, assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    MASK,
    POST_SOURCE,
    POST_WORDS,
    PRE_SOURCE,
    PRE_WORDS,
    Probe,
    check,
    refused,
    signed,
    vs_source,
    vs_words,
    words_text,
)

from startup_gfx2hlsl_scalar_m0_check import root_with_one_constant  # noqa: E402


OPCODES = {
    'v_cvt_pk_u16_u32': 0x36a,
    'v_cvt_pk_i16_i32': 0x36b,
}


def vop3a(op, vdst, src0, src1, src2=0, *, abs_bits=0, op_sel=0, clamp=0, omod=0, neg=0, lit=None):
    """Encode one synthetic gfx1013 VOP3A 64-bit (or 96-bit with literal) instruction word."""
    w0 = 0xd4000000 | (op << 16) | (clamp << 15) | (op_sel << 11) | (abs_bits << 8) | vdst
    w1 = (neg << 29) | (omod << 27) | (src2 << 18) | (src1 << 9) | src0
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


BODY_SOURCE = {
    op: ''.join(f'{op} v{12 + lane}, v{lane}, v{4 + lane}\n' for lane in range(4))
    for op in OPCODES
}
BODY_WORDS = {
    op: b''.join(vop3a(code, 12 + lane, 256 + lane, 256 + 4 + lane) for lane in range(4))
    for op, code in OPCODES.items()
}

# Canonical LLVM 15 round-trip lines (disasm_rdna2 <-> llvm-mc <-> libLLVM).
CANONICAL_CASES = (
    ('v_cvt_pk_u16_u32 v12, v0, v4', vop3a(0x36a, 12, 256, 260)),
    ('v_cvt_pk_u16_u32 v1, s4, 64', vop3a(0x36a, 1, 4, 192)),
    ('v_cvt_pk_u16_u32 v2, 0x12345, v3', vop3a(0x36a, 2, 255, 259, lit=0x12345)),
    ('v_cvt_pk_u16_u32 v5, 0, -1', vop3a(0x36a, 5, 128, 193)),
    ('v_cvt_pk_i16_i32 v12, v0, v4', vop3a(0x36b, 12, 256, 260)),
    ('v_cvt_pk_i16_i32 v1, -16, s4', vop3a(0x36b, 1, 208, 4)),
    ('v_cvt_pk_i16_i32 v2, v3, 0x12345', vop3a(0x36b, 2, 259, 255, lit=0x12345)),
    ('v_cvt_pk_i16_i32 v5, -1, 0', vop3a(0x36b, 5, 193, 128)),
)

# Additional disasm_rdna2 + gfx2hlsl inline-float / signed-hex literal encodings checked in pure-Python controls().
DISASM_EXTRA_CASES = (
    ('v_cvt_pk_u16_u32 v5, 1.0, -1', vop3a(0x36a, 5, 242, 193)),
    ('v_cvt_pk_i16_i32 v2, v3, 0xffff7fff', vop3a(0x36b, 2, 259, 255, lit=0xffff7fff)),
    ('v_cvt_pk_i16_i32 v5, -2.0, 0.5', vop3a(0x36b, 5, 245, 240)),
)


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


def sat_u16(x):
    return min(x & MASK, 0xffff)


def sat_i16(x):
    s = signed(x & MASK)
    return max(-32768, min(32767, s)) & 0xffff


def ref_pk_u16(a, b):
    return sat_u16(a) | (sat_u16(b) << 16)


def ref_pk_i16(a, b):
    return sat_i16(a) | (sat_i16(b) << 16)


REF = {
    'v_cvt_pk_u16_u32': ref_pk_u16,
    'v_cvt_pk_i16_i32': ref_pk_i16,
}

# Four distinct mutations per opcode: low truncation, high truncation, swapped halves, and opposite signedness.
# Set 0 of DRAW is chosen so every mutation produces 4 words distinct from REF on the very first draw.
MUTATIONS = {
    'v_cvt_pk_u16_u32': (
        (
            'trunc_lo',
            r'min\(v(\d+), 0xffffu\) \| \(min\(v(\d+), 0xffffu\) << 16u\)',
            r'(v\1 & 0xffffu) | (min(v\2, 0xffffu) << 16u)',
            lambda a, b: (a & 0xffff) | (sat_u16(b) << 16),
        ),
        (
            'trunc_hi',
            r'min\(v(\d+), 0xffffu\) \| \(min\(v(\d+), 0xffffu\) << 16u\)',
            r'min(v\1, 0xffffu) | ((v\2 & 0xffffu) << 16u)',
            lambda a, b: sat_u16(a) | ((b & 0xffff) << 16),
        ),
        (
            'swap_halves',
            r'min\(v(\d+), 0xffffu\) \| \(min\(v(\d+), 0xffffu\) << 16u\)',
            r'min(v\2, 0xffffu) | (min(v\1, 0xffffu) << 16u)',
            lambda a, b: sat_u16(b) | (sat_u16(a) << 16),
        ),
        (
            'signed_clamp',
            r'min\(v(\d+), 0xffffu\) \| \(min\(v(\d+), 0xffffu\) << 16u\)',
            r'(asuint(clamp(asint(v\1), -32768, 32767)) & 0xffffu) | '
            r'(asuint(clamp(asint(v\2), -32768, 32767)) << 16u)',
            ref_pk_i16,
        ),
    ),
    'v_cvt_pk_i16_i32': (
        (
            'trunc_lo',
            r'\(asuint\(clamp\(asint\(v(\d+)\), -32768, 32767\)\) & 0xffffu\) \| '
            r'\(asuint\(clamp\(asint\(v(\d+)\), -32768, 32767\)\) << 16u\)',
            r'(v\1 & 0xffffu) | (asuint(clamp(asint(v\2), -32768, 32767)) << 16u)',
            lambda a, b: (a & 0xffff) | (sat_i16(b) << 16),
        ),
        (
            'trunc_hi',
            r'\(asuint\(clamp\(asint\(v(\d+)\), -32768, 32767\)\) & 0xffffu\) \| '
            r'\(asuint\(clamp\(asint\(v(\d+)\), -32768, 32767\)\) << 16u\)',
            r'(asuint(clamp(asint(v\1), -32768, 32767)) & 0xffffu) | ((v\2 & 0xffffu) << 16u)',
            lambda a, b: sat_i16(a) | ((b & 0xffff) << 16),
        ),
        (
            'swap_halves',
            r'\(asuint\(clamp\(asint\(v(\d+)\), -32768, 32767\)\) & 0xffffu\) \| '
            r'\(asuint\(clamp\(asint\(v(\d+)\), -32768, 32767\)\) << 16u\)',
            r'(asuint(clamp(asint(v\2), -32768, 32767)) & 0xffffu) | '
            r'(asuint(clamp(asint(v\1), -32768, 32767)) << 16u)',
            lambda a, b: sat_i16(b) | (sat_i16(a) << 16),
        ),
        (
            'unsigned_clamp',
            r'\(asuint\(clamp\(asint\(v(\d+)\), -32768, 32767\)\) & 0xffffu\) \| '
            r'\(asuint\(clamp\(asint\(v(\d+)\), -32768, 32767\)\) << 16u\)',
            r'min(v\1, 0xffffu) | (min(v\2, 0xffffu) << 16u)',
            ref_pk_u16,
        ),
    ),
}

BOUNDARY_WORDS = (
    0x00000000, 0x00000001, 0x00007ffe, 0x00007fff, 0x00008000, 0x00008001,
    0x0000fffe, 0x0000ffff, 0x00010000, 0x00010001, 0x00012345, 0x12345678,
    0x7ffffffe, 0x7fffffff, 0x80000000, 0x80000001, 0xffff0000, 0xffff7ffe,
    0xffff7fff, 0xffff8000, 0xffff8001, 0xfffffffe, 0xffffffff, 0xdeadbeef,
    0x3f000000, 0xbf800000,
)

# 16 (src0, src1, unused_src2) triples per opcode (4 draws x 4 lanes).
DRAW = {
    'v_cvt_pk_u16_u32': [
        (0x00010002, 0x00020003, 0),
        (0x00008000, 0x00010001, 0),
        (0xffffffff, 0x00001234, 0),
        (0x80000001, 0x7ffffffe, 0),
        (0x00000000, 0x00000000, 0),
        (0x00000001, 0x0000ffff, 0),
        (0x0000ffff, 0x00000001, 0),
        (0x0000fffe, 0x00010000, 0),
        (0x00010000, 0x0000fffe, 0),
        (0x12345678, 0x0000abcd, 0),
        (0x0000fedc, 0x87654321, 0),
        (0x7fffffff, 0x80000000, 0),
        (0x00000000, 0xffffffff, 0),
        (0xffffffff, 0x00000000, 0),
        (0x00007fff, 0x00008000, 0),
        (0xdeadbeef, 0x0000cafe, 0),
    ],
    'v_cvt_pk_i16_i32': [
        (0x00008000, 0xffff7fff, 0),
        (0x7fffffff, 0x80000001, 0),
        (0xffff0001, 0x00010002, 0),
        (0x80000000, 0x00007ffe, 0),
        (0x00000000, 0x00000000, 0),
        (0x00007fff, 0xffff8000, 0),
        (0xffff8000, 0x00007fff, 0),
        (0xffffffff, 0x00000001, 0),
        (0x00000001, 0xffffffff, 0),
        (0xffff8001, 0x00007ffe, 0),
        (0x00001234, 0xffffedcc, 0),
        (0xfffffffe, 0xffff7ffe, 0),
        (0x00008001, 0x00010000, 0),
        (0x00000000, 0xffffffff, 0),
        (0xffffffff, 0x00000000, 0),
        (0xcafebabe, 0x00001357, 0),
    ],
}


def eval_hlsl_uint_expr(expr, env):
    """Evaluate a translated HLSL uint expression for v_cvt_pk_{u16_u32,i16_i32} on 32-bit unsigned inputs."""
    py = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', expr)
    scope = {
        'min': min,
        'clamp': lambda x, lo, hi: max(lo, min(hi, x)),
        'asint': lambda u: signed(u & MASK),
        'asuint': lambda i: i & MASK,
        **{k: v & MASK for k, v in env.items()},
    }
    return eval(py, {'__builtins__': {}}, scope) & MASK


def translate(op, mut_entry=None):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob())
    text = pipe.hlsl_ps()
    if mut_entry is not None:
        name, pattern, replacement, _ = mut_entry
        mutated, count = re.subn(pattern, replacement, text)
        assert count == 4 and mutated != text, (op, name, count)
        text = mutated
    return pipe, text


def decode_single(word):
    ins = G.decode(word)
    assert len(ins) == 1, ins
    _, _, name, asm = ins[0]
    probe = Probe()
    probe.one(name, asm)
    return probe.lines


def controls():
    # 1. Verify recorded VOP3A encodings round-trip through disasm_rdna2.decode_instruction.
    for text, raw in CANONICAL_CASES + DISASM_EXTRA_CASES:
        n, _, asm, _ = D.decode_instruction(raw, 0)
        check(f'disasm_rdna2 round-trip: {text}', n == len(raw) and asm == text)
    for op, blob in BODY_WORDS.items():
        off, lines = 0, []
        while off < len(blob):
            n, _, asm, _ = D.decode_instruction(blob, off)
            lines.append(asm + '\n')
            off += n
        check(f'{op}: BODY_WORDS matches BODY_SOURCE', ''.join(lines) == BODY_SOURCE[op])

    # 2. Exhaustive boundary-pair evaluation of the live emitted HLSL expressions against independent ISA math.
    grid = [(a, b) for a in BOUNDARY_WORDS for b in BOUNDARY_WORDS]
    for op in OPCODES:
        _, ps_text = translate(op)
        lanes = dict(re.findall(r'^\s*v(1[2-5]) = (.*);$', ps_text, re.M))
        check(f'{op}: four lanes translated', sorted(lanes) == ['12', '13', '14', '15'])
        expr0 = lanes['12']
        ref_fn = REF[op]
        mismatches = 0
        for a, b in grid:
            if eval_hlsl_uint_expr(expr0, {'v0': a, 'v4': b}) != ref_fn(a, b):
                mismatches += 1
        check(f'{op}: {len(grid)} boundary pairs match ISA saturation reference', mismatches == 0)

        first = DRAW[op][:4]
        ref_first = [ref_fn(a, b) for a, b, _ in first]
        for mut in MUTATIONS[op]:
            mut_name, _, _, mut_fn = mut
            _, mut_text = translate(op, mut)
            mut_lanes = dict(re.findall(r'^\s*v(1[2-5]) = (.*);$', mut_text, re.M))
            mut_first = [mut_fn(a, b) for a, b, _ in first]
            eval_first = [
                eval_hlsl_uint_expr(mut_lanes[str(12 + i)], {f'v{i}': first[i][0], f'v{4 + i}': first[i][1]})
                for i in range(4)
            ]
            check(f'{op} ({mut_name}): mutates draw set 0 and matches mutated model',
                  eval_first == mut_first and mut_first != ref_first)

    # 3. ComputePipeline, _e64 alias, inline negative/float constants, literals, and EXEC-masked emission.
    cs_blob = (
        vop3a(0x36a, 1, 4, 255, lit=0x00012345)
        + vop3a(0x36b, 2, 208, 242)  # -16 and 1.0 (0x3f800000)
        + struct.pack('<I', 0xbf810000)
    )
    cs_hlsl = G.ComputePipeline(cs_blob, root_blob(), (4 << 1) | (1 << 7), (1, 1, 1)).hlsl_cs()
    check('ComputePipeline: SGPR, literal, inline -16 and inline 1.0 lower cleanly',
          'v1 = min(s4, 0xffffu) | (min(0x00012345u, 0xffffu) << 16u);' in cs_hlsl
          and 'v2 = (asuint(clamp(asint(0xfffffff0u), -32768, 32767)) & 0xffffu) | '
              '(asuint(clamp(asint(0x3f800000u), -32768, 32767)) << 16u);' in cs_hlsl)

    for op in OPCODES:
        p_e64 = Probe()
        p_e64.one(f'{op}_e64', f'{op}_e64 v1, v2, s4')
        check(f'{op}_e64: explicit _e64 suffix accepted', len(p_e64.lines) == 1)

        masked = Probe()
        masked.exec_known = False
        masked.one(op, f'{op} v1, v2, v3')
        check(f'{op}: exec-masked emission guards assignment',
              len(masked.lines) == 1 and masked.lines[0].startswith('if (exec & 1) v1 = '))

    # 4. UNIFORM_VALU (v_readfirstlane_b32) and odd-dmask D16 low16_only_v provenance.
    class DummyPipe:
        resources = {}
        uavs = {}
        rtv_formats = ()
        srv_formats = {}

        def __init__(self):
            self.uds = []

        def resource(self, key, kind):
            self.resources.setdefault(key, {'index': 0, 'kind': kind})
            return 'r0'

        def ud(self, tag):
            if tag not in self.uds:
                self.uds.append(tag)
            return self.uds.index(tag)

    def run_stage(*instructions):
        s = G.Stage.__new__(G.Stage)
        s.pipe, s.kind, s.lines = DummyPipe(), 'ps', []
        s.attrs, s.exports, s.write_masks = {}, {}, {}
        s.sym = {16 + i: ('desc', ('table', 0), 0, i) for i in range(8)}
        s.used_s, s.used_v, s.masks = set(), set(), set()
        s.exec_known = True
        s.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(instructions)]
        s.translate()
        return [line for _, lines, _ in s.blocks for line in lines]

    for op in OPCODES:
        u_lines = run_stage(f'{op} v7, s2, 0x1234', 'v_readfirstlane_b32 s9, v7')
        check(f'{op}: uniform SGPR/constant pack proves v_readfirstlane_b32', u_lines[-1] == 's9 = v7;')
        try:
            run_stage(f'{op} v7, s2, v0', 'v_readfirstlane_b32 s9, v7')
        except G.Unsupported as exc:
            check(f'{op}: varying VGPR source refuses v_readfirstlane_b32', 'proven uniform source' in str(exc))
        else:
            raise AssertionError(f'{op}: varying source unexpectedly accepted by v_readfirstlane_b32')

        odd_d16 = 'image_load v4, v[0:1], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D d16'
        for bad_seq in ((odd_d16, f'{op} v5, v4, v2'), (odd_d16, f'{op} v5, v2, v4')):
            try:
                run_stage(*bad_seq)
            except G.Unsupported as exc:
                assert 'whole-word read of v4' in str(exc), exc
            else:
                raise AssertionError(f'{op}: accepted odd-dmask D16 source without masking')
        cleared = run_stage(odd_d16, f'{op} v4, v2, v3', 'v_add_nc_u32_e32 v5, 1, v4')
        check(f'{op}: whole-VGPR destination overwrite clears low16_only_v', cleared[-1] == 'v5 = 0x00000001u + v4;')

    # 5. Binary VOP3A encoding-guard and Stage.one refusal controls.
    for op, code in OPCODES.items():
        modifiers = [
            ('clamp', vop3a(code, 12, 256, 260, clamp=1)),
            ('omod=mul:2', vop3a(code, 12, 256, 260, omod=1)),
            ('omod=mul:4', vop3a(code, 12, 256, 260, omod=2)),
            ('omod=div:2', vop3a(code, 12, 256, 260, omod=3)),
            ('op_sel=[1,0]', vop3a(code, 12, 256, 260, op_sel=1)),
            ('op_sel=[0,1]', vop3a(code, 12, 256, 260, op_sel=2)),
            ('op_sel_dst', vop3a(code, 12, 256, 260, op_sel=8)),
        ]
        abs_neg = [
            ('abs src0 VGPR', vop3a(code, 12, 256, 260, abs_bits=1)),
            ('abs src1 VGPR', vop3a(code, 12, 256, 260, abs_bits=2)),
            ('abs unused src2', vop3a(code, 12, 256, 260, abs_bits=4)),
            ('abs src0 inline 1', vop3a(code, 12, 129, 260, abs_bits=1)),
            ('neg src0 VGPR', vop3a(code, 12, 256, 260, neg=1)),
            ('neg src1 VGPR', vop3a(code, 12, 256, 260, neg=2)),
            ('neg unused src2', vop3a(code, 12, 256, 260, neg=4)),
            ('neg src0 inline 1', vop3a(code, 12, 129, 260, neg=1)),
            ('neg src1 inline 1', vop3a(code, 12, 256, 129, neg=2)),
        ]
        unused_src2 = [
            ('src2=1', vop3a(code, 12, 256, 260, src2=1)),
            ('src2=256', vop3a(code, 12, 256, 260, src2=256)),
        ]
        for label, bad_word in modifiers + abs_neg + unused_src2:
            try:
                G.Pipeline(vs_words([(0, 0, 0)] * 4), PRE_WORDS + bad_word + POST_WORDS, root_blob())
            except G.Unsupported:
                pass
            else:
                raise AssertionError(f'{op}: accepted binary VOP3A with {label}')
        for label, bad_word in abs_neg:
            reasons = []
            G.decode(bad_word, reasons)
            check(f'{op}: encoding guard refuses {label}',
                  [r for r, _ in reasons] == [f'{op} with neg/abs'])
        for label, bad_word in unused_src2:
            reasons = []
            G.decode(bad_word, reasons)
            check(f'{op}: encoding guard refuses {label}',
                  [r for r, _ in reasons] == [f'{op} with unused src2 field set'])

        check(f'{op}: Stage refuses non-VOP3 spellings, modifiers, lane masks and bad operands', all(refused(asm) for asm in (
            f'{op}_e32 v1, v2, v3',
            f'{op}_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD',
            f'{op}_dpp v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf',
            f'{op}_dpp v1, v2, v3 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1',
            f'{op} v1, v2, v3 clamp',
            f'{op} v1, v2, v3 mul:2',
            f'{op} v1, v2, v3 div:2',
            f'{op} v1, v2, v3 op_sel:[1,0]',
            f'{op} v1, |v2|, v3',
            f'{op} v1, v2, |v3|',
            f'{op} v1, -v2, v3',
            f'{op} v1, v2, -v3',
            f'{op} v1, vcc_lo, v3',
            f'{op} v1, v2, exec_lo',
            f'{op} s0, v2, v3',
            f'{op} v[0:1], v2, v3',
            f'{op} v1, v[2:3], v4',
            f'{op} v1, s[2:3], v4',
            f'{op} v1, s106, v4',
            f'{op} v1, 0x100000000, v4',
            f'{op}',
            f'{op} v1',
            f'{op} v1, v2',
            f'{op} v1, v2, v3, v4',
        )))

    # 6. Unlowered control: removing the entries from VEC_U must restore refusals.
    saved = {op: G.VEC_U.pop(op) for op in OPCODES}
    try:
        for op in OPCODES:
            try:
                translate(op)
            except G.Unsupported:
                pass
            else:
                raise AssertionError(f'{op}: translated after removal from VEC_U')
            assert refused(f'{op} v1, v2, v3'), op
    finally:
        G.VEC_U.update(saved)
    check('unlowered control: removing v_cvt_pk_{u16_u32,i16_i32} from VEC_U restores refusals', True)


def check_binary(llvm_mc):
    for op, src in BODY_SOURCE.items():
        built = assemble(llvm_mc, src)
        assert built == BODY_WORDS[op], f'{op}: assembled {built.hex()}, recorded {BODY_WORDS[op].hex()}'
    for text, expected in CANONICAL_CASES:
        built = assemble(llvm_mc, text + '\n')
        assert built == expected, f'{text}: assembled {built.hex()}, recorded {expected.hex()}'
    # Explicit _e64 spelling assembles to the identical VOP3A word.
    for op, code in OPCODES.items():
        assert assemble(llvm_mc, f'{op}_e64 v12, v0, v4\n') == vop3a(code, 12, 256, 260)
    for name, source, raw in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        assert assemble(llvm_mc, source) == raw, f'{name} fixture differs from llvm-mc'
    for op, sets in DRAW.items():
        for i in range(0, len(sets), 4):
            chunk = sets[i:i + 4]
            assert assemble(llvm_mc, vs_source(chunk)) == vs_words(chunk), (op, chunk)
    print('llvm-mc: all VOP3A cases, _e64 aliases and draw fixtures match recorded words')


def check_decode(library):
    from pixel_aot import disassemble
    blob = b''.join(raw for _, raw in CANONICAL_CASES)
    llvm_lines = disassemble(blob, library)
    ours = [text for text, _ in CANONICAL_CASES]
    assert llvm_lines == ours, f'LLVM vs disasm_rdna2 mismatch: {list(zip(llvm_lines, ours))}'
    print(f'libLLVM: {len(CANONICAL_CASES)} VOP3A encodings match disasm_rdna2.py verbatim')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    draws = mut_checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op, triples in DRAW.items():
                ref_fn = REF[op]
                good_ps = compile_hlsl(compiler, kind, translate(op)[1], f'ps_{op}_{kind}', ps_profile, directory)
                for index in range(0, len(triples), 4):
                    chunk = triples[index:index + 4]
                    vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), ps_words(op), root_blob()).hlsl_vs(),
                                      f'vs_{op}_{index}_{kind}', vs_profile, directory)
                    expected = words_text(ref_fn(a, b) for a, b, _ in chunk)
                    run = draw(renderer, adapter, vs, good_ps, expected)
                    if run.returncode:
                        raise AssertionError(f'{kind} {op} inputs {chunk}\n{run.stdout}\n{run.stderr}')
                    draws += 1
                    print(f'{kind} {op} {[tuple(f"{v:x}" for v in t[:2]) for t in chunk]}: {expected}', flush=True)
                    if index:
                        continue
                    for mut in MUTATIONS[op]:
                        mut_name, _, _, mut_fn = mut
                        wrong = words_text(mut_fn(a, b) for a, b, _ in chunk)
                        bad_ps = compile_hlsl(compiler, kind, translate(op, mut)[1],
                                              f'mut_{op}_{mut_name}_{kind}', ps_profile, directory)
                        mrun = draw(renderer, adapter, vs, bad_ps, expected)
                        if mrun.returncode == 0 or f'actual={wrong}' not in mrun.stderr:
                            raise AssertionError(
                                f'{kind} {op} mutation {mut_name} did not read {wrong}\n{mrun.stdout}\n{mrun.stderr}'
                            )
                        mut_checks += 1
                        print(f'{kind} {op} mutation ({mut_name}): fails as expected, reads {wrong}', flush=True)
    print(f'{adapter}: {draws} draws matched, {mut_checks} mutated draws read their mutated words')


def f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def unorm_cases():
    midpoint = f32_bits(0.5)
    ties = [f32_bits((n + 0.5) / 65535) for n in (0, 1234, 65534)]
    values = [f32_bits(x) for x in (-1.0, 0.0, 0.5, 1.0, 2.0, math.inf, -math.inf)]
    values.extend((0x7fc12345, midpoint - 1, midpoint + 1))
    values.extend(center + delta for center in ties for delta in (-1, 1))
    return [(left, right, 0) for left, right in zip(values, values[4:] + values[:4])]


def unorm_reference(bits, truncate=False):
    value = struct.unpack('<f', struct.pack('<I', bits))[0]
    clipped = min(1.0, max(0.0, value)) if not math.isnan(value) else 0.0
    # A binary32 multiplied by a 16-bit integer is exact in Python's binary64.
    scaled = clipped * 65535
    return int(scaled) if truncate else round(scaled)


def unorm_words():
    op = vop3a(0x369, 12, 256, 260) + vop3a(0x369, 13, 257, 261)
    # Define every exported component, though only the first two carry packed results.
    op += struct.pack('<2I', 0x7e1c0280, 0x7e1e0280)
    return PRE_WORDS + op + POST_WORDS


def unorm_pipeline(cases):
    return G.Pipeline(vs_words(cases + [(0, 0, 0)] * 2), unorm_words(), root_blob())


def unorm_controls():
    midpoint = f32_bits(0.5)
    assert [unorm_reference(x) for x in (midpoint - 1, midpoint, midpoint + 1)] == [32767, 32768, 32768]
    assert unorm_reference(midpoint, truncate=True) == 32767
    assert 'pknorm_u16(asuint(F(v0)))' in unorm_pipeline(unorm_cases()[:2]).hlsl_ps()
    for reserved in (dict(src2=1), dict(op_sel=1), dict(omod=1), dict(clamp=1), dict(neg=4), dict(abs_bits=4)):
        try:
            G.decode(vop3a(0x369, 12, 256, 260, **reserved))
        except G.Unsupported as exc:
            assert str(exc) == 'v_cvt_pknorm_u16_f32 with reserved operand or modifier bits', exc
        else:
            raise AssertionError(reserved)


def check_unorm_draw(dxc, renderer, adapter):
    unorm_controls()
    cases = unorm_cases()
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        text = unorm_pipeline(cases[:2]).hlsl_ps()
        # Carry each half as the low bits of a finite normal float. The RGBA32F target
        # can then read back every bit of two packed words, including NaN encodings.
        export = 'float4(F(v12), F(v13), F(v14), F(v15))'
        transport = 'float4(' + ', '.join(
            f'asfloat(0x3f800000u | ((v{reg} >> {shift}u) & 0xffffu))'
            for reg in (12, 13) for shift in (0, 16)) + ')'
        assert text.count(export) == 1
        text = text.replace(export, transport)
        nearest = 'return q + (guard & uint(sticky || (q & 1u) != 0u));'
        assert text.count(nearest) == 1
        good = compile_hlsl(dxc, 'dxc', text, 'unorm', 'ps_6_0', directory)
        mutant = compile_hlsl(dxc, 'dxc', text.replace(nearest, 'return q;'), 'truncate', 'ps_6_0', directory)

        def expected(chunk, truncate=False):
            return words_text(0x3f800000 | unorm_reference(x, truncate) for a, b, _ in chunk for x in (a, b))

        rejected = 0
        for index in range(0, len(cases), 2):
            chunk = cases[index:index + 2]
            vs = compile_hlsl(dxc, 'dxc', unorm_pipeline(chunk).hlsl_vs(), f'vs_{index}', 'vs_6_0', directory)
            want = expected(chunk)
            run = draw(renderer, adapter, vs, good, want)
            assert run.returncode == 0, (chunk, run.stdout, run.stderr)
            truncated = expected(chunk, truncate=True)
            if truncated != want:
                run = draw(renderer, adapter, vs, mutant, want)
                assert run.returncode == 1 and f'actual={truncated}' in run.stderr, (chunk, run.stdout, run.stderr)
                rejected += 1
        assert rejected
        print(f'{adapter}: {len(cases)} normalized packs matched, {rejected} truncation draws rejected')


# Convert packed pixel x to float, scale and clamp it, and use root word s8 at x=4.
# Pack v3 and -0.5 into v7, then export its first channel pair.
SNORM_PS = bytes.fromhex(
    '000d067eff0606102549123e030057d50301c9038400847d08020a7e030b0602'
    '070068d703e30100030c00f807000000000081bf'
)
SNORM_TRUNCATION = 'packed = q + (guard & uint(sticky || (q & 1u) != 0u));'


def snorm_source(fmt=56, code=SNORM_PS):
    formats = () if fmt is None else (fmt,)
    return G.Pipeline(VS_WORDS, code, root_with_one_constant(), rtv_formats=formats,
                      ps_hw_inputs=(0x8000, 0x8000)).hlsl_ps()


def snorm_reference(bits, truncate=False):
    value = struct.unpack('<f', struct.pack('<I', bits))[0]
    scaled = 0.0 if math.isnan(value) else min(1.0, max(-1.0, value)) * 32767
    return (int(scaled) if truncate else round(scaled)) & 0xffff


def snorm_controls():
    source = snorm_source()
    assert 'v7 = pknorm_i16(v3) | (pknorm_i16(0xbf000000u) << 16u);' in source
    assert 'float((v7) & 0xffffu) / 65535.0' in source and 'v5 = ud1;' in source
    assert 'asint((v7) << 16u) >> 16) / 32767.0' in snorm_source(58)
    assert 'f16tof32(v7)' in snorm_source(54)
    move = struct.pack('<I', 0x7e120307)  # v_mov_b32_e32 v9, v7
    moved = SNORM_PS[:40] + move + SNORM_PS[40:44] + bytes((9, 0, 0, 0)) + SNORM_PS[48:]
    assert 'float((v9) & 0xffffu) / 65535.0' in snorm_source(code=moved)
    overwrite = struct.pack('<I', 0x7e0e0280)  # v_mov_b32_e32 v7, 0
    assert 'f16tof32(v7)' in snorm_source(code=SNORM_PS[:40] + overwrite + SNORM_PS[40:])
    for fmt, code, message in (
        (None, SNORM_PS, 'with RTV format 0'),
        (28, SNORM_PS, 'with RTV format 28'),
        (56, SNORM_PS[:32] + vop3a(0x368, 7, 8, 241) + SNORM_PS[40:],
         'source s8 is not a lane register or constant'),
        (56, SNORM_PS[:32] + vop3a(0x368, 7, 259, 241, src2=1) + SNORM_PS[40:],
         'with reserved operand or modifier bits'),
    ):
        try:
            snorm_source(fmt, code)
        except G.Unsupported as error:
            assert message in str(error), error
        else:
            raise AssertionError(message)
    midpoint = f32_bits(0.5)
    assert [snorm_reference(x) for x in (midpoint - 1, midpoint, midpoint + 1)] == [16383, 16384, 16384]
    assert snorm_reference(midpoint, truncate=True) == 16383
    assert snorm_reference(f32_bits(-1.0)) == 0x8001
    assert snorm_reference(0x7fc12345) == 0


def check_snorm_draw(dxc, renderer, adapter):
    snorm_controls()
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        # The translated pixel stage takes PARAM0 and SV_Position, in that order.
        vertex = """struct Out { float4 param0 : PARAM0; float4 pos : SV_Position; };
Out main(uint id : SV_VertexID) {
    Out o;
    o.param0 = float4(0, 0, 0, 0);
    o.pos = float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, 0, 1);
    return o;
}
"""
        vs = compile_hlsl(dxc, 'dxc', vertex, 'vs', 'vs_6_0', directory)
        source = snorm_source()
        assert source.count(SNORM_TRUNCATION) == 1
        ps = compile_hlsl(dxc, 'dxc', source, 'ps', 'ps_6_0', directory)
        mutant = compile_hlsl(dxc, 'dxc', source.replace(SNORM_TRUNCATION, 'packed = q;'),
                              'truncate', 'ps_6_0', directory)
        expected = directory / 'expected.bin'

        def run(shader, input_word):
            result = gpu_case_runner.run(renderer, adapter, (
                '--startup-r16-unorm', vs, shader, 8, 1, expected, f'{input_word:08x}'))
            print(result.stdout, end='')
            if result.returncode == 77:
                raise SystemExit(77)
            return result

        for input_word in (f32_bits(4 / 7), f32_bits(0.5), 0x7fc12345, f32_bits(-1.0)):
            inputs = [f32_bits(x / 7) for x in range(8)]
            inputs[4] = input_word
            expected.write_bytes(b''.join(struct.pack('<4I', snorm_reference(bits), 0, 0, 0) for bits in inputs))
            result = run(ps, input_word)
            assert result.returncode == 0, (input_word, result.stdout, result.stderr)
            if input_word == f32_bits(0.5):
                # Match the mutant at the other sample points so the comparator isolates the tie.
                expected.write_bytes(b''.join(struct.pack('<4I', snorm_reference(bits, truncate=x != 4), 0, 0, 0)
                                              for x, bits in enumerate(inputs)))
                result = run(mutant, input_word)
                assert result.returncode == 1 and '1 of 8 pixels differ' in result.stderr, result.stderr
                assert 'pixel mismatch (4,0): expected 00004000,00000000,00000000,00000000 actual 00003fff,' in result.stderr

def main():
    if len(sys.argv) == 1:
        controls()
        unorm_controls()
        snorm_controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary' and len(args) == 1:
        check_binary(args[0])
    elif mode == 'decode' and len(args) == 1:
        check_decode(args[0])
    elif mode == 'draw-snorm' and len(args) == 3:
        check_snorm_draw(*args)
    elif mode == 'draw-unorm' and len(args) == 3:
        check_unorm_draw(*args)
    elif mode == 'draw' and len(args) in (3, 4):
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
