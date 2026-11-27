"""Direct compute binary16 <-> 16-bit integer conversions (v_cvt_f16_u16, v_cvt_f16_i16, v_cvt_u16_f16, v_cvt_i16_f16)."""
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
from native_direct_container_check import words as pack_words
import native_f16_alu_check as F
from native_scalar_lds_check import scalar, sop1


OPS = {
    'v_cvt_f16_u16': (0x50, 0x1D0, False, True),
    'v_cvt_f16_i16': (0x51, 0x1D1, True, True),
    'v_cvt_u16_f16': (0x52, 0x1D2, False, False),
    'v_cvt_i16_f16': (0x53, 0x1D3, True, False),
}
HI_DST = 0xABCD0000
HI_SRC = 0xBEEF0000
DST_SEED = HI_DST | 0x5555


def vop1(op, vdst, src0):
    return 0x7E000000 | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)


def vop3_unary(op, vdst, src0, *, abs0=0, neg0=0, clamp=0, omod=0, op_sel=0, src1=0):
    w0 = (0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) |
          ((op_sel & 0xF) << 11) | ((abs0 & 7) << 8) | (vdst & 0xFF))
    w1 = ((neg0 & 7) << 29) | ((omod & 3) << 27) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return [w0, w1]


def v_mov_lit(vdst, val):
    return [0x7E0002FF | ((vdst & 0xFF) << 17), val & 0xFFFFFFFF]


def half_bits_to_f32(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xFFFF))[0]


def ref_cvt_f16_int(signed, raw, *, clamp=False, rne=True):
    """Exact binary16 encoding of low 16-bit integer (round to nearest even)."""
    val = raw & 0xFFFF
    if signed and (val & 0x8000):
        val -= 0x10000
    f = float(val)
    if rne:
        try:
            out = struct.unpack('<H', struct.pack('<e', f))[0]
        except OverflowError:
            out = 0xFC00 if f < 0.0 else 0x7C00
    else:
        # Broken mutant: round toward zero via f16 mantissa truncation.
        fbits = struct.unpack('<I', struct.pack('<f', f))[0]
        sign = (fbits >> 16) & 0x8000
        mag = fbits & 0x7FFFFFFF
        exp = mag >> 23
        frac = mag & 0x7FFFFF
        if mag == 0:
            out = sign
        elif exp > 142:
            out = sign | 0x7C00
        elif exp <= 112:
            out = sign | ((0x800000 | frac) >> (126 - exp)) if exp >= 102 else sign
        else:
            out = sign | (((exp - 112) << 10) | (frac >> 13))
    if clamp:
        if (out & 0x7FFF) > 0x7C00 or (out & 0x8000):
            out = 0
        elif out > 0x3BFF:
            out = 0x3C00
    return out


def ref_cvt_int_f16(signed, raw, *, abs0=0, neg0=0, nan_check=True):
    """Truncate toward zero and saturate binary16 to 16-bit int; NaN gives 0."""
    bits = raw & 0xFFFF
    if abs0:
        bits &= 0x7FFF
    if neg0:
        bits ^= 0x8000
    f = half_bits_to_f32(bits)
    if math.isnan(f):
        return 0 if nan_check else (0x8000 if signed else 0)
    lo, hi = (-32768, 32767) if signed else (0, 65535)
    if f <= lo:
        return lo & 0xFFFF
    if f >= hi:
        return hi & 0xFFFF
    return int(math.trunc(f)) & 0xFFFF


def cases():
    items = []

    def add(op, kind, src_val, *, src_code=257, abs0=0, neg0=0, clamp=0, wide=False, want=None):
        vop1_op, vop3_op, signed, to_f16 = OPS[op]
        if want is None:
            raw = F.INLINE_F16[src_code] if kind == 'inline_float' else src_val
            want = (ref_cvt_f16_int(signed, raw, clamp=bool(clamp)) if to_f16
                    else ref_cvt_int_f16(signed, raw, abs0=abs0, neg0=neg0))
        items.append((op, vop1_op, vop3_op, signed, to_f16, kind, src_val, src_code, abs0, neg0, clamp, wide, want))

    # v_cvt_f16_u16 (RNE rounding + overflow at 65520 + clamp + source kinds)
    for v in (0, 1, 2048, 2049, 2051, 32768, 65504, 65519, 65520, 65535):
        add('v_cvt_f16_u16', 'vgpr', HI_SRC | v)
    add('v_cvt_f16_u16', 'sgpr', HI_SRC | 2051)
    add('v_cvt_f16_u16', 'lit', 0xBEEF0803)  # 2051 -> 2052 (0x6804)
    add('v_cvt_f16_u16', 'inline_int', 0, src_code=128)
    add('v_cvt_f16_u16', 'inline_int', 64, src_code=192)
    add('v_cvt_f16_u16', 'inline_int', 0xFFFF, src_code=193)  # -1 low 16 bits = 65535 -> +inf
    add('v_cvt_f16_u16', 'inline_float', 0x3C00, src_code=242)  # 1.0 bits (15360) -> 15360.0 (0x7380)
    add('v_cvt_f16_u16', 'vop3', HI_SRC | 2051, wide=True)
    for v in (0, 1, 65535):
        add('v_cvt_f16_u16', 'vop3', v, clamp=1, wide=True)

    # v_cvt_f16_i16 (signed boundaries, RNE ties, clamp, source kinds)
    for v in (0, 1, 0xFFFF, 0xFFFD, 2049, (-2049) & 0xFFFF, (-2051) & 0xFFFF, 0x7FFF, 0x8000):
        add('v_cvt_f16_i16', 'vgpr', HI_SRC | v)
    add('v_cvt_f16_i16', 'sgpr', HI_SRC | 0xFFFD)  # -3.0
    add('v_cvt_f16_i16', 'lit', 0xCAFE8000)
    add('v_cvt_f16_i16', 'inline_int', 0xFFFF, src_code=193)  # -1.0 (0xBC00)
    add('v_cvt_f16_i16', 'inline_int', (-16) & 0xFFFF, src_code=208)  # -16.0
    add('v_cvt_f16_i16', 'inline_float', 0x3C00, src_code=242)
    for v in (0xFFFD, 0, 1, 0x7FFF):
        add('v_cvt_f16_i16', 'vop3', v, clamp=1, wide=True)

    # v_cvt_u16_f16 (truncation, clamping at 0..65535, NaN->0, abs/neg, source kinds)
    for h in (0x0000, 0x8000, 0x0001, 0x8001, 0x3800, 0xB800, 0x4180, 0xC180, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE01):
        add('v_cvt_u16_f16', 'vgpr', HI_SRC | h)
    add('v_cvt_u16_f16', 'sgpr', HI_SRC | 0x4180)
    add('v_cvt_u16_f16', 'lit', 0x12344580)  # 5.5 -> 5
    add('v_cvt_u16_f16', 'inline_float', 0x4400, src_code=246)  # 4.0 -> 4
    add('v_cvt_u16_f16', 'inline_float', 0xBC00, src_code=243)  # -1.0 -> 0
    add('v_cvt_u16_f16', 'vop3', HI_SRC | 0xC180, abs0=1, wide=True)  # |-2.75| -> 2
    add('v_cvt_u16_f16', 'vop3', HI_SRC | 0xC180, neg0=1, wide=True)  # -(-2.75) -> 2
    add('v_cvt_u16_f16', 'vop3', HI_SRC | 0x4180, abs0=1, neg0=1, wide=True)  # -|2.75| -> 0

    # v_cvt_i16_f16 (truncation, clamping at -32768..32767, NaN->0, abs/neg, source kinds)
    for h in (0x0000, 0x8000, 0x0001, 0x8001, 0x3800, 0xB800, 0x4180, 0xC180,
              0x77FF, 0x7800, 0x7BFF, 0xF800, 0xF801, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE01):
        add('v_cvt_i16_f16', 'vgpr', HI_SRC | h)
    add('v_cvt_i16_f16', 'sgpr', HI_SRC | 0xC180)
    add('v_cvt_i16_f16', 'lit', 0xABCD7800)  # 32768.0 -> 32767
    add('v_cvt_i16_f16', 'inline_float', 0xC400, src_code=247)  # -4.0 -> -4 (0xFFFC)
    add('v_cvt_i16_f16', 'vop3', HI_SRC | 0xF800, abs0=1, wide=True)  # |-32768.0| -> 32767
    add('v_cvt_i16_f16', 'vop3', HI_SRC | 0xF800, neg0=1, wide=True)  # -(-32768.0) -> 32767
    add('v_cvt_i16_f16', 'vop3', HI_SRC | 0x7800, abs0=1, neg0=1, wide=True)  # -|32768.0| -> -32768

    assert len(items) < F.STRIDE
    return items


def emit_program(items):
    code = [sop1(3, 20, 4), sop1(3, 21, 128), 0xF408080A, 0xFA000020,
            scalar(0x1E, 7, 5, 128 + F.SHIFT), F.vmove(8, 7), sop1(3, 126, 129),
            0xBFA40000, 0xBFA5000F, 0xBA800881, 0]
    want = []
    for index, (op, vop1_op, vop3_op, _, _, kind, src_val, src_code, abs0, neg0, clamp, wide, expected) in enumerate(items):
        code += v_mov_lit(3, DST_SEED)
        if kind == 'vgpr':
            code += v_mov_lit(1, src_val) + [vop1(vop1_op, 3, 257)]
        elif kind == 'sgpr':
            code += [sop1(3, 10, 255), src_val, vop1(vop1_op, 3, 10)]
        elif kind == 'lit':
            code += [vop1(vop1_op, 3, 255), src_val]
        elif kind in ('inline_int', 'inline_float'):
            code += [vop1(vop1_op, 3, src_code)]
        elif kind == 'vop3':
            assert wide
            code += v_mov_lit(1, src_val) + vop3_unary(vop3_op, 3, 257, abs0=abs0, neg0=neg0, clamp=clamp)
        else:
            raise AssertionError((op, kind))
        code += F.store(4 * index, 3)
        want.append(HI_DST | expected)
    code.append(0xBF810000)
    return code, want


def check_mutants(items):
    def match(got, expected):
        return got == (HI_DST | expected)

    for op, _, _, signed, to_f16, _, _, _, _, _, clamp, _, expected in items:
        assert match(HI_DST | expected, expected)

    # Zeroed upper half must fail.
    assert any(not match(expected, expected) for _, _, _, _, _, _, _, _, _, _, _, _, expected in items)

    # Round-toward-zero instead of RNE on f16<-int must fail (2051 / 65519 / 32767).
    assert any(
        ref_cvt_f16_int(signed, F.INLINE_F16[sc] if kind == 'inline_float' else sv, clamp=bool(cl), rne=False) != exp
        for _, _, _, signed, to_f16, kind, sv, sc, _, _, cl, _, exp in items if to_f16
    )

    # Skipping NaN->0 on int<-f16 must fail.
    assert any(
        ref_cvt_int_f16(signed, F.INLINE_F16[sc] if kind == 'inline_float' else sv, abs0=a0, neg0=n0, nan_check=False) != exp
        for _, _, _, signed, to_f16, kind, sv, sc, a0, n0, _, _, exp in items if not to_f16
    )


def refusal_programs():
    programs = []
    for op, (vop1_op, vop3_op, _, to_f16) in OPS.items():
        programs.append(([vop1(vop1_op, 3, 250), 0xFF00B101, 0xBF810000], 'unsupported DPP arithmetic'))
        programs.append(([vop1(vop1_op, 3, 249), 0x06060601, 0xBF810000], 'unimplemented direct SDWA operand profile'))
        programs.append((vop3_unary(vop3_op, 3, 257, omod=1) + [0xBF810000], f'unsupported {op} clamp or omod'))
        programs.append((vop3_unary(vop3_op, 3, 257, op_sel=1) + [0xBF810000], f'unsupported {op} high-half op_sel'))
        programs.append((vop3_unary(vop3_op, 3, 257, op_sel=8) + [0xBF810000], f'unsupported {op} high-half op_sel'))
        programs.append((vop3_unary(vop3_op, 3, 257, src1=1) + [0xBF810000], f'unsupported {op} operand modifiers'))
        if to_f16:
            programs.append((vop3_unary(vop3_op, 3, 257, abs0=1) + [0xBF810000], f'unsupported {op} operand modifiers'))
            programs.append((vop3_unary(vop3_op, 3, 257, neg0=1) + [0xBF810000], f'unsupported {op} operand modifiers'))
        else:
            programs.append((vop3_unary(vop3_op, 3, 257, clamp=1) + [0xBF810000], f'unsupported {op} clamp or omod'))

    for op in ('v_cvt_f16_i16', 'v_cvt_i16_f16'):
        insn = vop1(OPS[op][0], 3, 257)
        programs.append(([0xBFA4000C, insn, 0xBF810000], 'unsupported binary16 rounding or denormal mode'))
        programs.append(([0xBFA50000, insn, 0xBF810000], 'unsupported binary16 rounding or denormal mode'))
        programs.append(([0xBA8005C1, 1, insn, 0xBF810000], 'unsupported binary16 rounding or denormal mode'))
        programs.append(([0xB98005C1, insn, 0xBF810000], 'unsupported binary16 rounding or denormal mode'))
    return programs


def _float_val(token, values):
    if token.startswith('%'):
        kind, bits = values[token]
        assert kind == 'float', (token, kind)
        return F._f32(bits)
    return float(token)


def _fcmp(pred, a, b):
    if pred == 'ord':
        return not (math.isnan(a) or math.isnan(b))
    if pred == 'oge':
        return a >= b
    if pred == 'olt':
        return a < b
    if pred == 'ogt':
        return a > b
    raise AssertionError((pred, a, b))


def _fptoi(signed, f):
    val = int(math.trunc(f))
    return (val & 0xFFFFFFFF)


def interpret_cvt(ir_text):
    """Evaluate lane 0 of the lowered DXIL IR, handling int<->float conversions and fcmp."""
    body = ir_text.split('define void @main()', 1)[1].split('\n}', 1)[0]
    blocks, label, buf = {}, None, []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line == '{':
            continue
        if line.endswith(':') and ' ' not in line:
            if label is not None:
                blocks[label] = buf
            label, buf = line[:-1], []
            continue
        if label is None:
            raise AssertionError(line)
        buf.append(line)
    if label is not None:
        blocks[label] = buf
    values, memory, stores = {}, {}, []
    label, steps = 'entry', 0
    while steps < 200000:
        steps += 1
        followed = False
        for line in blocks[label]:
            if line.startswith('br '):
                label = F._branch(line, values)
                followed = True
                break
            if line == 'ret void':
                return sorted(stores)
            m = re.fullmatch(r'(%\w+) = (.+)', line)
            if m:
                name, rhs = m.group(1), m.group(2)
                if rhs.startswith(('sitofp i32 ', 'uitofp i32 ')):
                    parts = re.fullmatch(r'(sitofp|uitofp) i32 (\S+) to float', rhs)
                    raw = F._i32(parts.group(2), values)
                    val = float(F._signed(raw) if parts.group(1) == 'sitofp' else raw)
                    values[name] = ('float', struct.unpack('<I', struct.pack('<f', val))[0])
                    continue
                if rhs.startswith(('fptosi float ', 'fptoui float ')):
                    parts = re.fullmatch(r'(fptosi|fptoui) float (\S+) to i32', rhs)
                    val = _float_val(parts.group(2), values)
                    values[name] = ('i32', _fptoi(parts.group(1) == 'fptosi', val))
                    continue
                if rhs.startswith('fcmp '):
                    parts = re.fullmatch(r'fcmp (\w+) float (\S+), (\S+)', rhs)
                    ok = _fcmp(parts.group(1), _float_val(parts.group(2), values), _float_val(parts.group(3), values))
                    values[name] = ('i1', int(ok))
                    continue
                if rhs.startswith('select i1 ') and ', float ' in rhs:
                    parts = re.fullmatch(r'select i1 (\S+), float (\S+), float (\S+)', rhs)
                    cond = F._i1(parts.group(1), values)
                    fval = _float_val(parts.group(2) if cond else parts.group(3), values)
                    values[name] = ('float', struct.unpack('<I', struct.pack('<f', fval))[0])
                    continue
            F._exec(line, values, memory, stores)
        if not followed:
            raise AssertionError('block fell through: ' + label)
    raise AssertionError('step limit')


def ir(dumper):
    items = cases()
    check_mutants(items)
    code, want = emit_program(items)
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'cvt')
        assert run.returncode == 0, run.stderr
        stores = interpret_cvt(run.stdout)
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i][0], hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if got != exp]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (why, r.returncode, r.stderr)
    print(f'native_cvt_f16_i16 IR: {len(want)} conversions match exact reference; {len(refusals)} refusals matched')
    return 0


def controls(cli, sdk, work, root):
    items = cases()
    check_mutants(items)
    code, want = emit_program(items)
    marker = request(work, root, code)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    assert 'legacyF16ToF32' in dump and 'legacyF32ToF16' not in dump, dump

    refusals = refusal_programs()
    for n, (prog_words, why) in enumerate(refusals):
        m = request(work, root, prog_words)
        o = m.with_suffix(f'.refused{n}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_cvt_f16_i16 controls: compiled {len(want)} conversions to DXIL; verified {len(refusals)} refusals')
    return 0


def hardware(replay, sdk, work, root):
    items = cases()
    check_mutants(items)
    code, want = emit_program(items)
    marker = request(work, root, code)
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    words_out = list(struct.unpack(f'<{F.WORDS}I', output.read_bytes()))
    for group in range(F.GROUPS):
        head = F.BASE + group * F.STRIDE
        got = words_out[head:head + len(want)]
        assert got == want, (group, [(i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(got, want)) if a != b][:8])
    words_out[F.BASE:F.BASE + len(want)] = [0] * len(want)
    words_out[F.BASE + F.STRIDE:F.BASE + F.STRIDE + len(want)] = [0] * len(want)
    assert all(w == F.SENTINEL for w in words_out if w != 0), 'guard overwritten'
    print(f'native_cvt_f16_i16 hardware: {len(want)} conversions matched across {F.GROUPS} groups')
    return 0


def main(mode, tool, sdk=None):
    if mode == 'ir':
        return ir(tool)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
