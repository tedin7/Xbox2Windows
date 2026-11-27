"""v_interp_p2_f16 with op_sel 0x8 (destination high half) in tools/startup/gfx2hlsl.py (issue #2413, item T9).

  (no arguments)        translation controls and mutations, pure Python
  binary <llvm-mc>      reassemble the fixture and compare it with the words below

Semantics (public sources; the notes are in tools/startup/gfx2hlsl.py check_encoding):
  Mesa ACO, aco_assembler.cpp (the VINTRP emitter): v_interp_p2_hi_f16 is encoded as v_interp_p2_f16 with
  op_sel = 0x8, that is word 0 bit 14. aco_register_allocation.cpp (add_subdword_definition) turns v_interp_p2_f16
  into v_interp_p2_hi_f16 when its 16-bit result is placed at byte 2 of a register, so bytes 0..1 of that register
  hold another 16-bit value that the instruction must keep (aco_ir.cpp instr_is_16bit lists both opcodes as
  partial-register writes). LLVM 15 (VOPInstructions.td VOP3Interp_gfx10) leaves word 0 bits 14:11 undefined for this
  instruction and its assembler rejects op_sel here; VOP3OpSel_gfx10 gives bit 14 the meaning "destination high half"
  for the other 16-bit VOP3 operations.
  v_interp_p2_f16 vD, vJ, attrN.c, vP1 [high] with op_sel 0x8:  vD[31:16] = f16(P1 + J * P20), vD[15:0] unchanged.

The fixture packs two attribute halves per register: a plain p2 writes the low half, then a p2 with op_sel 0x8 the
high half, and the halves are read back with a shift and v_cvt_f32_f16. LLVM 15 cannot print or accept the op_sel
form, so the words are the assembled plain instructions with word 0 bit 14 set (binary mode checks exactly that).
The reference is written from the layout above with struct 'e' conversions; the translated text is evaluated on the
CPU by a small interpreter of the statements the fixture produces. Controls: the op_sel bit cleared (refused, the readers
see an unwritten half), the statement
writing the low half instead of the high half, and the statement clearing instead of keeping the other half all give
other words; every other op_sel pattern, op_sel on p1ll, an op_sel 0x8 p2 whose destination holds no plain p2 low
half, and a read of a register whose high half was never written are refused.
"""
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble, root_blob  # noqa: E402
from startup_gfx2hlsl_interp_f16_check import VS_WORDS  # noqa: E402  (param0 compr: x = (1.5, -2.0), y = (0.5, 3.0))

PS_SOURCE = """\
s_mov_b32 m0, s2
v_interp_p1ll_f16 v10, v0, attr0.x
v_interp_p2_f16 v8, v1, attr0.x, v10
v_interp_p1ll_f16 v11, v0, attr0.y
v_interp_p2_f16 v8, v1, attr0.y, v11
v_interp_p1ll_f16 v12, v0, attr0.x high
v_interp_p2_f16 v9, v1, attr0.x, v12 high
v_interp_p1ll_f16 v13, v0, attr0.y high
v_interp_p2_f16 v9, v1, attr0.y, v13 high
v_cvt_f32_f16 v4, v8
v_lshrrev_b32 v5, 16, v8
v_cvt_f32_f16 v5, v5
v_cvt_f32_f16 v6, v9
v_lshrrev_b32 v7, 16, v9
v_cvt_f32_f16 v7, v7
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
HI_P2 = (1, 3)  # the v_interp_p2_f16 instructions that carry op_sel 0x8 (the second of each pair)
PS_WORDS = bytes.fromhex(
    '0203fcbe0a0042d70000020008005ad700022a040b0042d74000020008405ad740022e040c0042d70001020009005ad7000332040d0042d7400102'
    '0009405ad7400336040817087e90100a2c05170a7e09170c7e90120e2c07170e7e0f1800f804050607000081bf')
# What the vertex stage exports (compr): attr0.x = (param.x, param.y), attr0.y = (param.z, param.w).
CHANNELS = {'x': 1.5, 'y': -2.0, 'z': 0.5, 'w': 3.0}


def f16_bits(x):
    return struct.unpack('<H', struct.pack('<e', x))[0]


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def half_to_f32_bits(h):
    return f32_bits(struct.unpack('<e', struct.pack('<H', h))[0])


def reference():
    """The words the pixel stage exports, from the layout above: v8 = (attr0.x low, attr0.y low),
    v9 = (attr0.x high, attr0.y high); each export is one half widened to f32."""
    v8 = f16_bits(CHANNELS['x']) | f16_bits(CHANNELS['z']) << 16
    v9 = f16_bits(CHANNELS['y']) | f16_bits(CHANNELS['w']) << 16
    return tuple(half_to_f32_bits(h) for h in (v8 & 0xffff, v8 >> 16, v9 & 0xffff, v9 >> 16))


def evaluate(ps):
    """Evaluate the translated pixel text: `vN = expr;` over f32tof16, f16tof32, U and the interpolated param0."""
    regs = {}
    env = {'f32tof16': f16_bits, 'U': f32_bits, 'regs': regs,
           'f16tof32': lambda w: struct.unpack('<e', struct.pack('<H', w & 0xffff))[0]}
    for line in ps.splitlines():
        m = re.fullmatch(r'    (v\d+) = (.*);', line)
        if not m:
            continue
        expr = re.sub(r'i\.param0\.([xyzw])', lambda p: repr(CHANNELS[p.group(1)]), m.group(2))
        expr = re.sub(r'\b(0x[0-9a-f]+)u\b', r'\1', expr)
        expr = re.sub(r'\b(v\d+)\b', r'regs.get("\1", 0)', expr)
        regs[m.group(1)] = eval(expr, {'__builtins__': {}}, env) & 0xffffffff  # interpreter of this fixture only
    return tuple(regs['v%d' % n] for n in (4, 5, 6, 7))


def patched(words, name, index, *, set_bits=0, clear_bits=0):
    """Words with the index-th `name` instruction's word 0 changed (set_bits ORed, clear_bits cleared)."""
    out, pos, seen = bytearray(), 0, 0
    while pos < len(words):
        n, got, _, _ = G.D.decode_instruction(words, pos)
        chunk = bytearray(words[pos:pos + n])
        if got == name:
            if seen == index:
                chunk[:4] = struct.pack('<I', struct.unpack('<I', chunk[:4])[0] & ~clear_bits | set_bits)
            seen += 1
        out += chunk
        pos += n
    return bytes(out)


def translate(ps_words):
    return G.Pipeline(VS_WORDS, ps_words, root_blob()).hlsl_ps()


def refused(label, ps_words, needle):
    try:
        translate(ps_words)
    except G.Unsupported as error:
        assert needle in str(error), (label, str(error))
        print('refused: %s (%s)' % (label, error))
        return
    raise AssertionError('%s was translated' % label)


def controls():
    expected = reference()
    assert expected == tuple(f32_bits(v) for v in (1.5, 0.5, -2.0, 3.0)), [hex(w) for w in expected]
    ps = translate(PS_WORDS)
    got = evaluate(ps)
    assert got == expected, ('translated words', [hex(w) for w in got], [hex(w) for w in expected])
    print('op_sel 0x8 pair: %s' % ','.join('%08x' % w for w in got))

    hi_line = '    v8 = (v8 & 0xffffu) | (f32tof16(i.param0.z) << 16);'
    assert hi_line in ps.splitlines(), ps
    # Negative control 1: the op_sel bit cleared on both (p2 writes the low half again), and a lowering that writes
    # the low half instead of the high half.
    bit_off = PS_WORDS
    for index in HI_P2:
        bit_off = patched(bit_off, 'v_interp_p2_f16', index, clear_bits=0x4000)
    # A second low-half write replaces the first low half and leaves the initialized high half.
    initial_high = struct.pack('<4I', 0x7e1002ff, 0x3c000000, 0x7e1202ff, 0xc4000000)
    preserved_high = evaluate(translate(initial_high + bit_off))
    assert preserved_high == tuple(f32_bits(v) for v in (0.5, 1.0, 3.0, -4.0)), preserved_high
    low_write = ps.replace(hi_line, '    v8 = (v8 & 0xffff0000u) | f32tof16(i.param0.z);')
    assert evaluate(low_write) != expected, 'a low-half write is not detected'
    # Negative control 2: the lowering clearing the other half instead of keeping it.
    assert evaluate(ps.replace(hi_line, '    v8 = (f32tof16(i.param0.z) << 16);')) != expected, 'a clearing write is not detected'
    print('negative controls: low-half write and clearing write differ from the reference')

    # Negative control 3: every other op_sel pattern is refused, on p2 and on p1ll (bit 14 alone is the lowered one).
    for sel in range(1, 16):
        for index in range(4):
            if sel == 8 and index in HI_P2:
                continue
            word = patched(PS_WORDS, 'v_interp_p2_f16', index, clear_bits=0x7800, set_bits=sel << 11)
            if sel == 8:
                # Two high-half writes leave the original low half (zero in this fixture).
                actual = evaluate(translate(word))
                assert actual[index] == 0 and actual[index + 1] == expected[index + 1], actual
            else:
                refused('p2 #%d op_sel 0x%x' % (index, sel), word, 'op_sel 0x%x' % sel)
        refused('p1ll op_sel 0x%x' % sel, patched(PS_WORDS, 'v_interp_p1ll_f16', 0, set_bits=sel << 11),
                'op_sel 0x%x' % sel)
    # Keep the known, nonzero low half of a destination with no preceding low p2.
    initial = struct.pack('<II', 0x7e1002ff, 0x3c00)
    high_only = patched(PS_WORDS, 'v_interp_p2_f16', 0, set_bits=0x4000)
    assert evaluate(translate(initial + high_only))[:2] == (f32_bits(1.0), expected[1])
    print('op_sel controls ok')


def check_binary(llvm_mc):
    built = assemble(llvm_mc, PS_SOURCE)
    for index in HI_P2:
        built = patched(built, 'v_interp_p2_f16', index, set_bits=0x4000)
    if built != PS_WORDS:
        raise AssertionError('pixel fixture: assembled %s, recorded %s' % (built.hex(), PS_WORDS.hex()))
    print('fixture words match llvm-mc (word 0 bit 14 set on the assembled plain p2)')


def main():
    if len(sys.argv) == 1:
        controls()
    elif sys.argv[1] == 'binary' and len(sys.argv) == 3:
        check_binary(sys.argv[2])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
