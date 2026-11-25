"""16-bit integer VOPC compares in tools/startup/gfx2hlsl.py: v_cmp_*_{i16,u16} and v_cmpx_*_{i16,u16}, e32 and e64.

  (no arguments)                       translation controls and fixture checks, pure Python
  binary <llvm-mc> <llvm-objcopy>      assemble every control form and the pixel fixtures with LLVM
                                       (-filetype=obj, .text extracted with llvm-objcopy) and compare with the words
                                       built or recorded here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the pixel fixtures with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw 16 x 16 pixels and compare every pixel with the plain-integer reference

Semantics, from the AMD RDNA 2 ISA guide (VOPC opcodes): V_CMP_<C>_I16 sets the lane bit to S0.i16 <C> S1.i16 and
V_CMP_<C>_U16 to S0.u16 <C> S1.u16, the 16-bit source being bits 15:0 of the operand; v_cmpx writes the result to
EXEC. LLVM 15 models the same: `icmp slt i16` of two truncated 32-bit values selects v_cmp_lt_i16_e32 on the whole
VGPRs, and the i16 operands take 16-bit literals only. The references below are written from that meaning with plain
integer arithmetic: the low half, sign-extended for i16 or zero-extended for u16, compared as the condition says.

Controls: all 60 decoded cases (12 v_cmp e32 into vcc, 12 v_cmp e64 into vcc_lo, 12 into s4, 24 v_cmpx into EXEC)
evaluated on the CPU with the integer evaluator of startup_gfx2hlsl_bit_alu_check.py over words whose high halves
carry junk; SGPR, inline and literal operands; the EXEC intersection when EXEC is not known; the refusals (SDWA,
op_sel, clamp, NEG/ABS bits, inline float constants, literals wider than 16 bits, register ranges, vcc/exec/m0
sources, vcc_hi/m0 destinations); the 32-bit forms unchanged; and the unused s0 that the decoder prints for an e64
v_cmpx: an SGPR value set in s0 before the compare still reads back after it, as for the 32-bit form.

The draws: a pixel program derives v1 and v2 from the packed screen position (v2 is v1 with junk in the high half and
sometimes bit 15 and low bits flipped), then
  K    v_cmpx_le_i16_e32 v1, v2           (a killed pixel keeps the clear value -1.0)
  K64  v_cmpx_gt_u16_e64 v2, v1
  M    for each of lt, eq, le, gt, ne, ge (bit k): v12 v_cmp_C_i16_e32 vcc_lo, v1, v2; v13 v_cmp_C_u16_e32 vcc_lo, v1,
       v2; v14 v_cmp_C_i16_e64 s4, v2, -16; v15 v_cmp_C_u16_e64 s4, 0x8000, v1; each bit folded in with
       v_cndmask_b32_e64 and v_or_b32, then every word ORed with 0x3f800000 (a normal binary32 word) and exported.
Expected image per pixel p = x | y << 16: a = p * 0x9b5d, b = a ^ (p * 0x6d2b & 0xffff800f) (32-bit products);
K keeps 0x3f800000 | p in all four words when ext_i16(a) <= ext_i16(b), else 0xbf800000; K64 when
ext_u16(b) > ext_u16(a);
M writes 0x3f800000 | sum(1 << k for each condition k that holds) per channel with the operands above.
Two mutated translations (register sources only; constants unchanged): `no sext` reads an i16 register as its
zero-extended low half (an int in 0..65535), `whole word` reads a u16 register as the whole 32-bit word. Each mutated
draw must fail against the reference and match the image of the same formula with the mutated register reading.

Not covered: gfx1013 hardware (the draws run the translated HLSL on desktop adapters), the SDWA forms (refused here;
tools/pixel_aot.py lowers them on its own), and inline float constants on 16-bit operands (refused).
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
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
import startup_image_3d_check as I3  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
CONDITIONS = ('lt', 'eq', 'le', 'gt', 'ne', 'ge')
RELATION = {'lt': lambda x, y: x < y, 'eq': lambda x, y: x == y, 'le': lambda x, y: x <= y,
            'gt': lambda x, y: x > y, 'ne': lambda x, y: x != y, 'ge': lambda x, y: x >= y}
# (cmpx, condition, type) -> VOPC opcode, from the decoder's table.
OPCODES = {}
for _op, (_name, _) in D.VOPC_OPS.items():
    _m = re.fullmatch(r'v_cmp(x?)_(\w+)_([iu]16)_e32', _name)
    if _m:
        OPCODES[_m.group(1) == 'x', _m.group(2), _m.group(3)] = _op

VCC_LO, VCC_HI, M0, LITERAL = 106, 107, 124, 255
SGPR_DST = 4


def V(n):
    return 256 + n


def inline(value):
    return 128 + value if value >= 0 else 192 - value


def e32(op, src0, vsrc1, literal=None):
    word = (0b0111110 << 25) | (op << 17) | (vsrc1 << 9) | src0
    return struct.pack('<I', word) + (b'' if literal is None else struct.pack('<I', literal))


def e64(op, sdst, src0, src1, literal=None, absolute=0, neg=0, opsel=0, clamp=0):
    w0 = (0b110101 << 26) | (op << 16) | (clamp << 15) | (opsel << 11) | (absolute << 8) | sdst
    w1 = src0 | (src1 << 9) | (neg << 29)
    return struct.pack('<II', w0, w1) + (b'' if literal is None else struct.pack('<I', literal))


# ---- reference -----------------------------------------------------------------------------------------------------
def ext(word, typ):
    low = word & 0xffff
    return low - 0x10000 if typ == 'i16' and low & 0x8000 else low


def reference(cond, typ, a, b):
    return int(RELATION[cond](ext(a, typ), ext(b, typ)))


def whole(cond, typ, a, b):
    """A compare of the whole words, as the 32-bit form of the same signedness reads them."""
    s = (lambda v: v - (1 << 32) if v >> 31 else v) if typ == 'i16' else (lambda v: v)
    return int(RELATION[cond](s(a), s(b)))


# High halves carry junk; low halves include 0, 1, 0x7fff, 0x8000, 0xffff and equal halves under different highs.
VALUES = [0x00000000, 0x12340000, 0xabcd8000, 0x00007fff, 0x5a5affff, 0xffff0001, 0x80000001, 0x7fff8000,
          0x0000ffff, 0xfedc7fff]
PAIRS = [(a, b) for a in VALUES for b in VALUES]

# ---- translation ---------------------------------------------------------------------------------------------------
failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for decoded instructions to be translated."""

    def __init__(self, exec_known=True):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = exec_known


def decode_one(code):
    decoded = G.decode(code)
    assert len(decoded) == 1 and decoded[0][1] == len(code), decoded
    return decoded[0][2], decoded[0][3]


def translate(code, exec_known=True):
    """(decoded text, destination, expression) of one compare."""
    name, asm = decode_one(code)
    stage = Probe(exec_known)
    stage.one(name, asm)
    assert len(stage.lines) == 1, stage.lines
    m = re.fullmatch(r'(\w+) = (.*);', stage.lines[0])
    assert m, stage.lines
    return asm, m.group(1), m.group(2)


def refusal(code=None, name=None, asm=None):
    try:
        if code is not None:
            name, asm = decode_one(code)
        Probe().one(name, asm)
    except G.Unsupported as error:
        return str(error)
    return None


def forms():
    """(label, code, destination, cmpx, condition, type) for the 60 decoded cases."""
    for (cmpx, cond, typ), op in sorted(OPCODES.items()):
        if cmpx:
            yield 'e32', e32(op, V(1), 2), 'exec', cmpx, cond, typ
            yield 'e64', e64(op, 0, V(1), V(2)), 'exec', cmpx, cond, typ
        else:
            yield 'e32', e32(op, V(1), 2), 'vcc', cmpx, cond, typ
            yield 'e64 vcc_lo', e64(op, VCC_LO, V(1), V(2)), 'vcc', cmpx, cond, typ
            yield 'e64 s4', e64(op, SGPR_DST, V(1), V(2)), 's4', cmpx, cond, typ


def llvm_text(asm):
    """The LLVM 15 source spelling of a decoded form: an e64 v_cmpx is written without the unused s0."""
    return re.sub(r'^(v_cmpx_\w+_e64) s0, ', r'\1 ', asm)


def controls():
    # 1. The encoder writes what LLVM 15 assembles (bytes recorded with the binary mode of this check).
    known = [
        (e32(OPCODES[False, 'lt', 'i16'], V(1), 2), '0105127d'),
        (e64(OPCODES[False, 'lt', 'i16'], SGPR_DST, V(1), V(2)), '040089d401050200'),
        (e32(OPCODES[True, 'le', 'i16'], V(1), 2), '0105367d'),
        (e64(OPCODES[True, 'le', 'i16'], 0, V(1), 2), '00009bd401050000'),
        (e32(OPCODES[False, 'lt', 'i16'], inline(-1), 2), 'c104127d'),
        (e64(OPCODES[False, 'lt', 'i16'], SGPR_DST, inline(-16), 5), '040089d4d00a0000'),
        (e32(OPCODES[False, 'lt', 'i16'], LITERAL, 2, 0x8000), 'ff04127d00800000'),
    ]
    check('encoder matches LLVM 15 encodings', all(code.hex() == want for code, want in known),
          str([(code.hex(), want) for code, want in known if code.hex() != want]))
    check('24 opcodes: 6 conditions x i16/u16 x v_cmp/v_cmpx',
          sorted(OPCODES) == sorted((x, c, t) for x in (False, True) for c in CONDITIONS for t in ('i16', 'u16')))

    # 2. The value set separates the reference from whole-word compares and, for the ordered relations, from the
    # opposite signedness.
    for typ in ('i16', 'u16'):
        other = 'u16' if typ == 'i16' else 'i16'
        for cond in CONDITIONS:
            check('values separate %s %s from a whole-word compare' % (cond, typ),
                  any(reference(cond, typ, a, b) != whole(cond, typ, a, b) for a, b in PAIRS))
            if cond not in ('eq', 'ne'):
                check('values separate %s %s from %s' % (cond, typ, other),
                      any(reference(cond, typ, a, b) != reference(cond, other, a, b) for a, b in PAIRS))

    # 3. The 60 decoded cases, evaluated.
    count = 0
    for label, code, destination, cmpx, cond, typ in forms():
        asm, dst, expr = translate(code)
        tree = Expr(expr)
        wrong = [(hex(a), hex(b)) for a, b in PAIRS
                 if tree.run({'v1': a, 'v2': b})[0] != reference(cond, typ, a, b)]
        check('%s -> %s' % (asm, dst), dst == destination and not wrong, '%s first wrong pairs %s' % (expr, wrong[:3]))
        count += 1
    check('60 decoded cases', count == 60)

    # 4. Other operand kinds: an SGPR, inline integers and 16-bit literals, in both orders.
    op = OPCODES[False, 'lt', 'i16'], OPCODES[False, 'lt', 'u16'], OPCODES[False, 'ge', 'u16']
    constants = [(inline(-16), None, -16), (inline(-1), None, -1), (inline(0), None, 0), (inline(64), None, 64),
                 (LITERAL, 0x8000, 0x8000), (LITERAL, 0x7fff, 0x7fff), (LITERAL, 0xffef, 0xffef)]
    for opcode in op:
        _, cond, typ = next(k for k, v in OPCODES.items() if v == opcode)
        for src, literal, value in constants:
            for order in (0, 1):
                code = (e64(opcode, SGPR_DST, src, V(2), literal) if order == 0
                        else e64(opcode, SGPR_DST, V(2), src, literal))
                asm, _, expr = translate(code)
                tree = Expr(expr)
                ok = all(tree.run({'v2': w})[0] == (reference(cond, typ, value & MASK, w) if order == 0
                                                     else reference(cond, typ, w, value & MASK)) for w in VALUES)
                check('%s' % asm, ok, expr)
        asm, _, expr = translate(e64(opcode, SGPR_DST, 5, V(2)))
        tree = Expr(expr)
        check('%s (SGPR source)' % asm, all(tree.run({'s5': a, 'v2': b})[0] == reference(cond, typ, a, b)
                                             for a, b in PAIRS), expr)

    # 5. EXEC not known: every destination is intersected with EXEC.
    for label, code, destination, cmpx, cond, typ in forms():
        asm, dst, expr = translate(code, exec_known=False)
        tree = Expr(expr)
        ok = expr.startswith('(exec & ') and all(
            tree.run({'v1': a, 'v2': b, 'exec': e})[0] == (e & reference(cond, typ, a, b))
            for a, b in PAIRS[:20] for e in (0, 1))
        if not ok:
            check('%s with EXEC unknown' % asm, False, expr)
            break
    else:
        check('all 60 cases intersect with an unknown EXEC', True)

    # 6. Refusals.
    lt_i16, cmpx_lt_u16 = OPCODES[False, 'lt', 'i16'], OPCODES[True, 'lt', 'u16']
    refused = []
    for (cmpx, cond, typ), opcode in sorted(OPCODES.items()):
        name = 'v_cmp%s_%s_%s_sdwa' % ('x' if cmpx else '', cond, typ)
        text = '%s %sv1, v2 src0_sel:DWORD src1_sel:DWORD' % (name, '' if cmpx else 's6, ')
        refused.append(('SDWA ' + name, refusal(name=name, asm=text) == name))
    refused += [
        ('op_sel', 'op_sel' in (refusal(e64(lt_i16, SGPR_DST, V(1), V(2), opsel=1)) or '')),
        ('clamp', 'clamp' in (refusal(e64(lt_i16, SGPR_DST, V(1), V(2), clamp=1)) or '')),
    ]
    for bit in range(3):
        for src0 in (V(1), inline(1)):
            refused.append(('NEG bit %d on %d' % (bit, src0),
                            'neg/abs' in (refusal(e64(lt_i16, SGPR_DST, src0, V(2), neg=1 << bit)) or '')))
            refused.append(('ABS bit %d on %d' % (bit, src0),
                            'neg/abs' in (refusal(e64(lt_i16, SGPR_DST, src0, V(2), absolute=1 << bit)) or '')))
    for field in ('neg', 'absolute'):  # inline 1 as src1, with the src1 bit
        refused.append(('%s on inline 1 in src1' % field,
                        'neg/abs' in (refusal(e64(lt_i16, SGPR_DST, V(1), inline(1), **{field: 2})) or '')))
    for code in range(240, 249):
        refused.append(('inline float %d e32' % code,
                        'binary16' in (refusal(e32(lt_i16, code, 2)) or '')))
        refused.append(('inline float %d e64 src1' % code,
                        'binary16' in (refusal(e64(cmpx_lt_u16, 0, V(1), code)) or '')))
    refused.append(('literal 0x18000', 'wider than 16 bits' in (refusal(e32(lt_i16, LITERAL, 2, 0x18000)) or '')))
    for token in ('-17', 's[2:3]', 'v[1:2]', 'vcc_lo', 'exec_lo', 'm0', '-v1', '|v1|', 'neg(v1)'):
        refused.append(('source %s' % token, refusal(name='v_cmp_lt_i16_e64',
                                                     asm='v_cmp_lt_i16_e64 s4, %s, v2' % token) is not None))
    refused.append(('destination vcc_hi', 'into vcc_hi' in (refusal(e64(lt_i16, VCC_HI, V(1), V(2))) or '')))
    refused.append(('destination m0', 'into m0' in (refusal(e64(lt_i16, M0, V(1), V(2))) or '')))
    refused.append(('no 16-bit f', refusal(name='v_cmp_f_i16_e64', asm='v_cmp_f_i16_e64 s4, v1, v2')
                    == 'v_cmp_f_i16'))
    bad = [label for label, ok in refused if not ok]
    check('%d refusals' % len(refused), not bad, str(bad))

    # 7. The 32-bit forms are unchanged.
    stage = Probe()
    stage.one('v_cmp_lt_i32_e32', 'v_cmp_lt_i32_e32 vcc_lo, v1, v2')
    stage.one('v_cmpx_gt_u32_e64', 'v_cmpx_gt_u32_e64 exec_lo, v3, s4')
    check('32-bit compares unchanged', stage.lines == ['vcc = ((int(v1) < int(v2)) ? 1u : 0u);',
                                                       'exec = ((v3 > s4) ? 1u : 0u);'], str(stage.lines))

    # 8. The unused s0 of an e64 v_cmpx: a value set in s0 before the compare reads back after it, as it does for
    # the 32-bit form (the decoder prints s0, and the translation must not treat it as written).
    for words, name in ((S0_I16_WORDS, 'v_cmpx_le_i16_e64'), (S0_I32_WORDS, 'v_cmpx_le_i32_e64')):
        decoded = [asm for _o, _s, _n, asm in G.decode(words)]
        text = G.Pipeline(I3.VS_WORDS, words, I3.root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
        body = text[text.index('s0 = 0x00001234u;'):]
        check('%s keeps s0 readable' % name,
              decoded[3] == '%s s0, v1, v2' % name and 'if (exec & 1) v3 = 0x00001234u;' in body
              and not re.search(r'^\s*s0 = (?!0x00001234u;)', body, re.M), body)


# ---- the pixel fixtures --------------------------------------------------------------------------------------------
WIDTH = HEIGHT = 16
M1, M2, FLIP = 0x9b5d, 0x6d2b, 0xffff800f
CLEAR = 0xbf800000  # -1.0, the target's clear value: a killed pixel keeps it
ONE = 0x3f800000
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
PROLOGUE = ('v_mul_u32_u24_e32 v1, 0x%x, v0\nv_mul_u32_u24_e32 v2, 0x%x, v0\nv_and_b32_e32 v2, 0x%x, v2\n'
            'v_xor_b32_e32 v2, v1, v2\n' % (M1, M2, FLIP))
# Each exported word is the packed position ORed with 0x3f800000, a normal binary32 value.
EXPORT_V0 = 'v_or_b32_e32 v3, 0x3f800000, v0\nexp mrt0 v3, v3, v3, v3 done vm\ns_endpgm\n'
K_SOURCE = PROLOGUE + 'v_cmpx_le_i16_e32 v1, v2\n' + EXPORT_V0
K64_SOURCE = PROLOGUE + 'v_cmpx_gt_u16_e64 v2, v1\n' + EXPORT_V0
M_CHANNELS = (('v_cmp_%s_i16_e32 vcc_lo, v1, v2', 'vcc_lo'), ('v_cmp_%s_u16_e32 vcc_lo, v1, v2', 'vcc_lo'),
              ('v_cmp_%s_i16_e64 s4, v2, -16', 's4'), ('v_cmp_%s_u16_e64 s4, 0x8000, v1', 's4'))
M_SOURCE = PROLOGUE + ''.join('v_mov_b32_e32 v%d, 0\n' % r for r in range(12, 16)) + ''.join(
    (cmp % cond) + '\nv_cndmask_b32_e64 v10, 0, %d, %s\nv_or_b32_e32 v%d, v%d, v10\n' % (1 << k, mask, 12 + c, 12 + c)
    for k, cond in enumerate(CONDITIONS) for c, (cmp, mask) in enumerate(M_CHANNELS)) + ''.join(
    'v_or_b32_e32 v%d, 0x3f800000, v%d\n' % (r, r) for r in range(12, 16)) + 'exp mrt0 v12, v13, v14, v15 done vm\ns_endpgm\n'
PROLOGUE_WORDS = 'ff0002165d9b0000ff0004162b6d0000ff0404360f80ffff0105043a'
K_WORDS = bytes.fromhex(PROLOGUE_WORDS + '0105367d' + 'f20006380f1800f803030303000081bf')
K64_WORDS = bytes.fromhex(PROLOGUE_WORDS + '0000bcd402030200' + 'f20006380f1800f803030303000081bf')
M_WORDS = bytes.fromhex(
    PROLOGUE_WORDS + '8002187e80021a7e80021c7e80021e7e'
    '0105127d0a0001d58002a9010c1518380105527d0a0001d58002a9010d151a38040089d402a101000a0001d5800211000e151c38'
    '0400a9d4ff020200008000000a0001d5800211000f151e380105147d0a0001d58004a9010c1518380105547d0a0001d58004a901'
    '0d151a3804008ad402a101000a0001d5800411000e151c380400aad4ff020200008000000a0001d5800411000f151e380105167d'
    '0a0001d58008a9010c1518380105567d0a0001d58008a9010d151a3804008bd402a101000a0001d5800811000e151c380400abd4'
    'ff020200008000000a0001d5800811000f151e380105187d0a0001d58010a9010c1518380105587d0a0001d58010a9010d151a38'
    '04008cd402a101000a0001d5801011000e151c380400acd4ff020200008000000a0001d5801011000f151e3801051a7d0a0001d5'
    '8020a9010c15183801055a7d0a0001d58020a9010d151a3804008dd402a101000a0001d5802011000e151c380400add4ff020200'
    '008000000a0001d5802011000f151e3801051c7d0a0001d58040a9010c15183801055c7d0a0001d58040a9010d151a3804008ed4'
    '02a101000a0001d5804011000e151c380400aed4ff020200008000000a0001d5804011000f151e38f2181838f21a1a38f21c1c38'
    'f21e1e380f1800f80c0d0e0f000081bf')
S0_SOURCE = ('v_mov_b32_e32 v1, v0\nv_lshrrev_b32_e32 v2, 16, v0\ns_mov_b32 s0, 0x1234\n%s v1, v2\n'
             'v_mov_b32_e32 v3, s0\nexp mrt0 v3, v3, v3, v3 done vm\ns_endpgm\n')
S0_I16_WORDS = bytes.fromhex('0003027e9000042cff0380be3412000000009bd4010502000002067e0f1800f803030303000081bf')
S0_I32_WORDS = bytes.fromhex('0003027e9000042cff0380be34120000000093d4010502000002067e0f1800f803030303000081bf')
FIXTURES = {'K': (K_SOURCE, K_WORDS), 'K64': (K64_SOURCE, K64_WORDS), 'M': (M_SOURCE, M_WORDS),
            'S0 i16': (S0_SOURCE % 'v_cmpx_le_i16_e64', S0_I16_WORDS),
            'S0 i32': (S0_SOURCE % 'v_cmpx_le_i32_e64', S0_I32_WORDS)}

# Mutations of the translated text: (pattern, replacement, expected count per program).
MUTATIONS = {
    'no sext': (r'\(asint\((v\d+) << 16u\) >> 16u\)', r'asint(\1 & 0xffffu)', {'K': 2, 'M': 18}),
    'whole word': (r'\((v\d+) & 0xffffu\)', r'\1', {'K64': 2, 'M': 18}),
}


def operands(position):
    a = (position * M1) & MASK  # both factors are below 2^24, so v_mul_u32_u24 is the plain product
    return a, a ^ ((position * M2) & MASK & FLIP)


def read(word, typ, how):
    """A register source as the (possibly mutated) translation reads it."""
    if typ == 'i16' and how == 'no sext':
        return word & 0xffff
    if typ == 'u16' and how == 'whole word':
        return word
    return ext(word, typ)


def holds(cond, typ, x, y, how):
    """x and y are ('r', word) for a register or ('c', value) for a constant, which no mutation changes."""
    vx, vy = ((read(v, typ, how) if kind == 'r' else v) for kind, v in (x, y))
    return RELATION[cond](vx, vy)


def image(program, how='reference'):
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            position = x | (y << 16)
            a, b = operands(position)
            if program == 'K':
                words += [ONE | position] * 4 if holds('le', 'i16', ('r', a), ('r', b), how) else [CLEAR] * 4
            elif program == 'K64':
                words += [ONE | position] * 4 if holds('gt', 'u16', ('r', b), ('r', a), how) else [CLEAR] * 4
            else:
                channels = [('i16', ('r', a), ('r', b)), ('u16', ('r', a), ('r', b)),
                            ('i16', ('r', b), ('c', -16)), ('u16', ('c', 0x8000), ('r', a))]
                words += [ONE | sum(1 << k for k, cond in enumerate(CONDITIONS) if holds(cond, typ, x0, y0, how))
                          for typ, x0, y0 in channels]
    return words


def pixel_hlsl(program, how='reference'):
    text = G.Pipeline(I3.VS_WORDS, FIXTURES[program][1], I3.root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
    if how != 'reference':
        pattern, replacement, counts = MUTATIONS[how]
        text, count = re.subn(pattern, replacement, text)
        assert count == counts[program], (program, how, count)
    return text


def expected_decode(source):
    lines = [llvm_line.replace('0x3f800000', '1.0') for llvm_line in source.splitlines()]
    return [re.sub(r'^(v_cmpx_\w+_e64) ', r'\1 s0, ', line) for line in lines]


def check_fixture():
    for name, (source, words) in FIXTURES.items():
        decoded = [asm for _o, _s, _n, asm in G.decode(words)]
        check('%s fixture decodes to its source' % name, decoded == expected_decode(source), str(decoded))
    pairs = [operands(x | (y << 16)) for y in range(HEIGHT) for x in range(WIDTH)]
    equal = sum((a ^ b) & 0xffff == 0 for a, b in pairs)
    crossing = sum((a ^ b) >> 15 & 1 for a, b in pairs)
    high = sum((a ^ b) >> 16 != 0 for a, b in pairs)
    check('position pairs: %d equal low halves, %d differ in bit 15, %d differ in the high half'
          % (equal, crossing, high), equal and crossing and high)
    for program in ('K', 'K64'):
        reference_image = image(program)
        kept = sum(reference_image[i] != CLEAR for i in range(0, len(reference_image), 4))
        check('%s keeps %d of %d pixels' % (program, kept, WIDTH * HEIGHT), 0 < kept < WIDTH * HEIGHT)
    for how, (_, _, counts) in MUTATIONS.items():
        for program in counts:
            pixel_hlsl(program, how)
            differing = sum(a != b for a, b in zip(image(program), image(program, how)))
            check('%s: %s mutation changes %d words' % (program, how, differing), differing > 0)
    # Every predicate bit of M takes both values somewhere, except eq/ne against a constant: no position gives a low
    # half equal to -16 or 0x8000, so those two bits are constant in the draw (control step 4 evaluates them).
    m = image('M')
    for channel in range(4):
        for k, cond in enumerate(CONDITIONS):
            if channel >= 2 and cond in ('eq', 'ne'):
                continue
            bits = {m[4 * i + channel] >> k & 1 for i in range(WIDTH * HEIGHT)}
            check('M channel %d %s takes both values' % (channel, cond), bits == {0, 1})
    # The translated compares, evaluated on the CPU, agree with the images.
    for program, how in (('K', 'reference'), ('K64', 'reference'), ('K', 'no sext'), ('K64', 'whole word')):
        text = pixel_hlsl(program, how)
        line = next(ln.strip() for ln in text.splitlines() if ln.strip().startswith('exec = '))
        tree = Expr(line[len('exec = '):-1])
        img = image(program, how)
        check('%s %s: evaluated compare equals the image' % (program, how), all(
            bool(tree.run(dict(zip(('v1', 'v2'), operands(x | y << 16))))[0])
            == (img[4 * (y * WIDTH + x)] != CLEAR) for y in range(HEIGHT) for x in range(WIDTH)))


# ---- LLVM ----------------------------------------------------------------------------------------------------------
def assemble(llvm_mc, objcopy, source):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        src, obj, raw = directory / 'fixture.s', directory / 'fixture.o', directory / 'fixture.bin'
        src.write_text(source, encoding='ascii')
        subprocess.run([llvm_mc, '-triple=amdgcn--amdpal', '-mcpu=gfx1013', '-filetype=obj', str(src), '-o', str(obj)],
                       check=True)
        subprocess.run([objcopy, '-O', 'binary', '--only-section=.text', str(obj), str(raw)], check=True)
        return raw.read_bytes()


def control_codes():
    codes = [code for _, code, *_ in forms()]
    for opcode in (OPCODES[False, 'lt', 'i16'], OPCODES[False, 'lt', 'u16'], OPCODES[False, 'ge', 'u16']):
        for src, literal in ((inline(-16), None), (inline(-1), None), (inline(0), None), (inline(64), None),
                             (LITERAL, 0x8000), (LITERAL, 0x7fff), (LITERAL, 0xffef)):
            codes += [e64(opcode, SGPR_DST, src, V(2), literal), e64(opcode, SGPR_DST, V(2), src, literal)]
        codes.append(e64(opcode, SGPR_DST, 5, V(2)))
    return codes


def check_binary(llvm_mc, objcopy):
    codes = control_codes()
    source = ''.join(llvm_text(decode_one(code)[1]) + '\n' for code in codes)
    check('LLVM assembles %d decoded control forms to the same words' % len(codes),
          assemble(llvm_mc, objcopy, source) == b''.join(codes))
    # LLVM reads a 16-bit literal as the low half: it encodes 0xffff on these operands as the inline constant -1.
    check('LLVM encodes the literal 0xffff on an i16 operand as inline -1',
          assemble(llvm_mc, objcopy, 'v_cmp_lt_i16_e64 s4, 0xffff, v2\n')
          == e64(OPCODES[False, 'lt', 'i16'], SGPR_DST, inline(-1), V(2)))
    for name, (fixture_source, words) in FIXTURES.items():
        check('LLVM assembles the %s fixture to the recorded words' % name,
              assemble(llvm_mc, objcopy, fixture_source) == words)


# ---- draws ---------------------------------------------------------------------------------------------------------
def words_file(directory, name, words):
    path = directory / ('%s.bin' % name)
    path.write_bytes(struct.pack('<%dI' % len(words), *words))
    return path


def draw(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        # A capability skip must not hide a failure found earlier in this run (fixture checks or earlier draws).
        if failures:
            finish()
        sys.exit(77)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = I3.compile_hlsl(compiler, kind, I3.FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            for program in ('K', 'K64', 'M'):
                reference = words_file(directory, '%s_reference' % program, image(program))
                ps = I3.compile_hlsl(compiler, kind, pixel_hlsl(program), 'ps_%s_%s' % (program, kind), ps_profile,
                                     directory)
                run = draw(renderer, adapter, vs, ps, reference)
                print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                                      if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
                check('%s %s %s: all %d pixels match the reference' % (kind, adapter, program, WIDTH * HEIGHT),
                      run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                      run.stdout + run.stderr)
                for how, (_, _, counts) in MUTATIONS.items():
                    if program not in counts:
                        continue
                    tag = '%s_%s_%s' % (program, how.replace(' ', '_'), kind)
                    bad = I3.compile_hlsl(compiler, kind, pixel_hlsl(program, how), 'ps_' + tag, ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad, reference)
                    lines = [ln for ln in run.stderr.splitlines() if 'pixel' in ln]
                    check('%s %s %s %s mutation fails against the reference: %s'
                          % (kind, adapter, program, how, ' / '.join(lines[:1])),
                          run.returncode != 0 and lines and lines[0].startswith('pixel mismatch'),
                          run.stdout + run.stderr)
                    own = words_file(directory, tag, image(program, how))
                    run = draw(renderer, adapter, vs, bad, own)
                    check('%s %s %s %s mutation matches its own image' % (kind, adapter, program, how),
                          run.returncode == 0, run.stdout + run.stderr)


def main():
    if len(sys.argv) == 1:
        controls()
        check_fixture()
    else:
        mode, *args = sys.argv[1:]
        if mode == 'binary' and len(args) == 2:
            check_binary(*args)
        elif mode == 'draw' and len(args) in (3, 4):
            check_fixture()
            check_draw(*args)
        else:
            raise SystemExit(__doc__)
    finish()


def finish():
    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
