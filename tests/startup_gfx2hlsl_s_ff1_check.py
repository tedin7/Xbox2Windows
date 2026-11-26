"""s_ff1_i32_b32 of a general scalar and s_ff1_i32_b64 immediates + SGPR pairs (#2022 Item F).

usage: startup_gfx2hlsl_s_ff1_check.py
       startup_gfx2hlsl_s_ff1_check.py dispatch <dxc> <offscreen_check> <warp|hardware>

RDNA 2 ISA SOP1: S_FF1_I32_B32 finds the lowest set bit of a 32-bit word (0xffffffff if zero);
SCC is unchanged. The compute-entry special case on exec_lo still folds to lane 0. Lane masks
and descriptor sources are refused.

S_FF1_I32_B64 accepts a 32-bit SOP1 literal/inline (ISA zero-extends the high word to 0) and
folds it, and also lowers a live SGPR pair with the same 64-bit firstbitlow scan as the scalar
bit-scan forms.

With `dispatch`, the b32 lowering runs as a compute shader on the adapter (D3D12 UAV
readback via offscreen_check, same pattern as startup_gfx2hlsl_pack_absdiff_check). The
reference is plain Python; the translated text is evaluated on the CPU without `dispatch`.
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
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
SOP1 = {name: op for op, (name, _) in D.SOP1_OPS.items()}
WORDS = (0x00000000, 0xffffffff, 0x80000001, 0x00010020, 0x55555555, 0xaaaaaaaa, 0x12345678, 0x10)
# GPU cases from review: prove firstbitlow(0) == 0xffffffff and the named edge bits.
DRAW_B32 = (0x00000000, 0x00000001, 0x80000000, 0x00010000)
# b64 immediates only (hi forced to 0 by zero-extension). hi-nonzero is #2072.
IMMEDIATE_B64 = (0x00000000, 0x00000001, 0x00010000, 0x80000000)
# Live SGPR pair s4:s5 as (lo, hi): both zero, lo only, hi only, both set (lo wins), edge bits in each half.
PAIR_B64 = ((0, 0), (1, 0), (0, 1), (0, 0x80000000), (0, 0x10), (0x80000000, 0xffffffff), (0x00010000, 0x1),
            (0xffffffff, 0xffffffff))
SCC_UNSET = 2
VCC_LO, EXEC_LO = 106, 126

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def sop1(name, sdst, ssrc, lit=None):
    word = struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (SOP1[name] << 8) | ssrc)
    if lit is not None:
        word += struct.pack('<I', lit & MASK)
    return word


def ff1_32(word):
    word &= MASK
    return MASK if word == 0 else (word & -word).bit_length() - 1


class Probe(G.Stage):
    def __init__(self, kind='vs'):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True
        self.kind = kind
        self.past_entry = False


def translate(words, masks=(), sym=None, kind='vs'):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    stage = Probe(kind=kind)
    stage.masks = set(masks)
    if sym:
        stage.sym.update(sym)
    try:
        stage.one(decoded[0][2], decoded[0][3])
    except G.Unsupported as error:
        stage.lines = ['refused: %s' % error]
    return stage.lines, stage.sym


def run(lines, env):
    env = dict(env)
    for line in lines:
        m = re.fullmatch(r'(\w+) = (.*);', line)
        assert m, line
        env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


def controls():
    # --- s_ff1_i32_b32 of a general SGPR ---
    bad = None
    lines, _ = translate(sop1('s_ff1_i32_b32', 3, 1))
    assert lines and not lines[0].startswith('refused'), lines
    for word in WORDS:
        out = run(lines, {'s1': word, 'scc': SCC_UNSET})
        want = (ff1_32(word), SCC_UNSET)
        if (out['s3'], out['scc']) != want:
            bad = (hex(word), (out['s3'], out['scc']), want)
            break
    check('s_ff1_i32_b32 SGPR source', bad is None, str(bad))

    bad = None
    for word in WORDS:
        stage = Probe()
        try:
            stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s3, %d' % word)
        except G.Unsupported as error:
            bad = ('refused %s' % error,)
            break
        out = run(stage.lines, {'scc': SCC_UNSET})
        want = (ff1_32(word), SCC_UNSET)
        if (out['s3'], out['scc']) != want:
            bad = (hex(word), (out['s3'], out['scc']), want)
            break
    check('s_ff1_i32_b32 immediate source', bad is None, str(bad))

    # Constant source folds.
    lines, sym = translate(sop1('s_ff1_i32_b32', 3, 1), sym={1: ('const', 0x10)})
    check('s_ff1_i32_b32 folds a constant source',
          lines == ['s3 = 0x00000004u;'] and sym.get(3) == ('const', 4),
          str((lines, sym.get(3))))

    # firstbitlow of zero is 0xffffffff (HLSL / ISA).
    lines, _ = translate(sop1('s_ff1_i32_b32', 3, 1))
    out = run(lines, {'s1': 0, 'scc': SCC_UNSET})
    check('s_ff1_i32_b32 of 0 is 0xffffffff',
          out['s3'] == MASK and out['scc'] == SCC_UNSET,
          str((out['s3'], out['scc'])))

    # Compute-entry exec_lo still folds to lane 0.
    lines, sym = translate(sop1('s_ff1_i32_b32', 5, EXEC_LO), kind='cs')
    check('s_ff1_i32_b32 of compute-entry exec_lo folds to 0',
          lines == ['s5 = 0u;'] and sym.get(5) == ('const', 0),
          str((lines, sym.get(5))))

    # Outside compute, or past entry, exec_lo is refused.
    stage = Probe(kind='vs')
    try:
        stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s5, exec_lo')
        check('s_ff1_i32_b32 of exec_lo outside compute is refused', False, stage.lines)
    except G.Unsupported as error:
        check('s_ff1_i32_b32 of exec_lo outside compute is refused', 'exec_lo' in str(error), str(error))

    # --- s_ff1_i32_b64 of a 32-bit immediate (zero-extended; hi = 0) ---
    bad = None
    for word in IMMEDIATE_B64:
        lines, sym = translate(sop1('s_ff1_i32_b64', 3, 255, word))
        if lines[0].startswith('refused'):
            bad = ('refused %s for %s' % (lines[0], hex(word)),)
            break
        want = ff1_32(word)
        if lines != ['s3 = 0x%08xu;' % want] or sym.get(3) != ('const', want):
            bad = (hex(word), lines, sym.get(3), want)
            break
    check('s_ff1_i32_b64 folds a zero-extended immediate', bad is None, str(bad))

    lines, _ = translate(sop1('s_ff1_i32_b64', 3, 255, 0))
    check('s_ff1_i32_b64 of immediate 0 is 0xffffffff',
          lines == ['s3 = 0xffffffffu;'],
          str(lines))

    # Inline constant 0 (ssrc 128) is the same zero-extend rule.
    lines, sym = translate(sop1('s_ff1_i32_b64', 3, 128))
    check('s_ff1_i32_b64 folds inline 0',
          lines == ['s3 = 0xffffffffu;'] and sym.get(3) == ('const', MASK),
          str((lines, sym.get(3))))

    # SGPR pairs use the same 64-bit firstbitlow scan as #2072 / SCALAR_BITS.
    lines, _ = translate(sop1('s_ff1_i32_b64', 3, 4))
    check('s_ff1_i32_b64 of a live SGPR pair emits firstbitlow over the pair',
          (not lines[0].startswith('refused:')
           and 'firstbitlow' in lines[0]
           and 's4' in lines[0]
           and 's5' in lines[0]),
          str(lines))

    lines, _ = translate(sop1('s_ff1_i32_b64', 3, 4),
                         sym={4: ('const', 0), 5: ('const', 0x10)})
    # Known consts substitute into the scan text (hi half at +32); still not refused.
    check('s_ff1_i32_b64 of a constant SGPR pair still scans both halves',
          (not lines[0].startswith('refused:')
           and 'firstbitlow' in lines[0]
           and '32u' in lines[0]
           and '0x00000010u' in lines[0]),
          str(lines))

    # Refusals: lane masks and descriptors.
    refused = [translate(sop1('s_ff1_i32_b32', d, s))[0][0] for d, s in (
        (3, VCC_LO), (3, EXEC_LO), (VCC_LO, 1), (EXEC_LO, 1))]
    check('s_ff1_i32_b32 refuses lane-mask destinations/sources',
          all(r.startswith('refused: ') for r in refused), str(refused))

    stage = Probe()
    stage.masks.add(1)
    try:
        stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s3, s1')
        check('s_ff1_i32_b32 refuses a tracked lane-mask source', False, stage.lines)
    except G.Unsupported as error:
        check('s_ff1_i32_b32 refuses a tracked lane-mask source', 'lane mask' in str(error), str(error))

    stage = Probe()
    stage.sym[1] = ('desc', ('root', 0), 0, 0)
    try:
        stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s3, s1')
        check('s_ff1_i32_b32 refuses a descriptor source', False, stage.lines)
    except G.Unsupported as error:
        check('s_ff1_i32_b32 refuses a descriptor source', 'desc' in str(error), str(error))

    # Negative control: replace firstbitlow with a constant; evaluation must diverge.
    stage = Probe()
    stage.one('s_ff1_i32_b32', 's_ff1_i32_b32 s3, s1')
    mutated = [re.sub(r'firstbitlow\(s1\)', '0u', stage.lines[0])]
    assert mutated != stage.lines, stage.lines
    out_ok = run(stage.lines, {'s1': 0x10, 'scc': 0})
    out_bad = run(mutated, {'s1': 0x10, 'scc': 0})
    check('mutation that drops firstbitlow diverges',
          out_ok['s3'] == 4 and out_bad['s3'] == 0,
          str((out_ok['s3'], out_bad['s3'])))

    if failures:
        print('FAILED: %s' % ', '.join(failures))
        sys.exit(1)
    print('all ok')


def dispatch_b32(dxc, runner, adapter, cases, directory):
    """Translate s_ff1_i32_b32 once; feed each case through a UAV and read the result back."""
    lines, _ = translate(sop1('s_ff1_i32_b32', 3, 1))
    assert lines and not lines[0].startswith('refused'), lines
    fill = tuple(((i + 1) * 0x9e3779b1 & MASK) | 1 for i in range(len(cases)))
    hlsl = ['RWByteAddressBuffer data : register(u0);', '[numthreads(1, 1, 1)]', 'void main() {']
    for i, word in enumerate(cases):
        dst_byte = 4 * (len(fill) + i)
        hlsl.extend((
            '    {',
            '        uint s1 = data.Load(%d) ^ %s;' % (4 * i, '%#010xu' % (fill[i] ^ word)),
            '        uint s3;',
            '        bool scc = true;',
        ))
        hlsl.extend('        ' + line for line in lines)
        hlsl.append('        data.Store(%d, s3);' % dst_byte)
        hlsl.append('    }')
    hlsl.append('}')

    src, cso, output = (directory / ('s_ff1_b32' + suffix) for suffix in ('.hlsl', '.cso', '.bin'))
    src.write_text('\n'.join(hlsl) + '\n', encoding='ascii')
    compile_result = subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)],
                                    stderr=subprocess.PIPE, text=True)
    if compile_result.returncode or compile_result.stderr:
        raise AssertionError(compile_result.stderr or 'DXC exited with %d' % compile_result.returncode)
    word_count = len(fill) + len(cases)
    dump_words = (word_count + 3) // 4 * 4
    run_result = subprocess.run(
        [runner, adapter, '--raw-uav-dump', str(cso), '1', str(4 * dump_words), str(output)],
        capture_output=True, text=True)
    print(run_result.stdout, end='')
    if run_result.returncode == 77:
        sys.exit(77)
    if run_result.returncode or run_result.stderr:
        raise AssertionError(run_result.stderr or 'raw UAV dump exited with %d' % run_result.returncode)
    actual = struct.unpack('<%dI' % dump_words, output.read_bytes())
    assert actual[:len(fill)] == fill, 'operand region changed: %s' % (actual[:len(fill)],)
    return actual[len(fill):word_count]


def ff1_64(lo, hi):
    """CPU reference for s_ff1_i32_b64: lowest set bit of the pair, -1 when both words are zero."""
    lo, hi = lo & MASK, hi & MASK
    return ff1_32(lo) if lo else (32 + ff1_32(hi) if hi else MASK)


def dispatch_b64_pairs(dxc, runner, adapter, cases, directory):
    """Translate s_ff1_i32_b64 of the live pair s4:s5 once; feed each (lo, hi) through a UAV and read it back."""
    lines, _ = translate(sop1('s_ff1_i32_b64', 3, 4))
    assert lines and not lines[0].startswith('refused'), lines
    fill = tuple(((i + 1) * 0x9e3779b1 & MASK) | 1 for i in range(2 * len(cases)))
    hlsl = ['RWByteAddressBuffer data : register(u0);', '[numthreads(1, 1, 1)]', 'void main() {']
    for i, (lo, hi) in enumerate(cases):
        hlsl.extend((
            '    {',
            '        uint s4 = data.Load(%d) ^ %s;' % (8 * i, '%#010xu' % (fill[2 * i] ^ lo)),
            '        uint s5 = data.Load(%d) ^ %s;' % (8 * i + 4, '%#010xu' % (fill[2 * i + 1] ^ hi)),
            '        uint s3;',
            '        bool scc = true;',
        ))
        hlsl.extend('        ' + line for line in lines)
        hlsl.append('        data.Store(%d, s3);' % (4 * (len(fill) + i)))
        hlsl.append('    }')
    hlsl.append('}')

    src, cso, output = (directory / ('s_ff1_b64_pair' + suffix) for suffix in ('.hlsl', '.cso', '.bin'))
    src.write_text('\n'.join(hlsl) + '\n', encoding='ascii')
    compile_result = subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)],
                                    stderr=subprocess.PIPE, text=True)
    if compile_result.returncode or compile_result.stderr:
        raise AssertionError(compile_result.stderr or 'DXC exited with %d' % compile_result.returncode)
    word_count = len(fill) + len(cases)
    dump_words = (word_count + 3) // 4 * 4
    run_result = subprocess.run(
        [runner, adapter, '--raw-uav-dump', str(cso), '1', str(4 * dump_words), str(output)],
        capture_output=True, text=True)
    print(run_result.stdout, end='')
    if run_result.returncode == 77:
        sys.exit(77)
    if run_result.returncode or run_result.stderr:
        raise AssertionError(run_result.stderr or 'raw UAV dump exited with %d' % run_result.returncode)
    actual = struct.unpack('<%dI' % dump_words, output.read_bytes())
    assert actual[:len(fill)] == fill, 'operand region changed: %s' % (actual[:len(fill)],)
    return actual[len(fill):word_count]


def dispatch_b64_immediates(dxc, runner, adapter, cases, directory):
    """Store the folded b64-immediate results; proves the zero-extend fold on the adapter."""
    hlsl = ['RWByteAddressBuffer data : register(u0);', '[numthreads(1, 1, 1)]', 'void main() {']
    for i, word in enumerate(cases):
        stage = Probe()
        stage.one('s_ff1_i32_b64', 's_ff1_i32_b64 s3, %#x' % word)
        hlsl.extend((
            '    {',
            '        uint s3;',
            '        bool scc = true;',
        ))
        hlsl.extend('        ' + line for line in stage.lines)
        hlsl.append('        data.Store(%d, s3);' % (4 * i))
        hlsl.append('    }')
    hlsl.append('}')

    src, cso, output = (directory / ('s_ff1_b64_imm' + suffix) for suffix in ('.hlsl', '.cso', '.bin'))
    src.write_text('\n'.join(hlsl) + '\n', encoding='ascii')
    compile_result = subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)],
                                    stderr=subprocess.PIPE, text=True)
    if compile_result.returncode or compile_result.stderr:
        raise AssertionError(compile_result.stderr or 'DXC exited with %d' % compile_result.returncode)
    word_count = len(cases)
    dump_words = (word_count + 3) // 4 * 4
    run_result = subprocess.run(
        [runner, adapter, '--raw-uav-dump', str(cso), '1', str(4 * dump_words), str(output)],
        capture_output=True, text=True)
    print(run_result.stdout, end='')
    if run_result.returncode == 77:
        sys.exit(77)
    if run_result.returncode or run_result.stderr:
        raise AssertionError(run_result.stderr or 'raw UAV dump exited with %d' % run_result.returncode)
    actual = struct.unpack('<%dI' % dump_words, output.read_bytes())
    return actual[:word_count]


def check_dispatch(dxc, runner, adapter):
    controls()
    runner = str(Path(runner).resolve())
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        got = dispatch_b32(dxc, runner, adapter, DRAW_B32, directory)
        for i, word in enumerate(DRAW_B32):
            want = ff1_32(word)
            check('dispatch s_ff1_i32_b32 %#010x' % word, got[i] == want,
                  'got %#x want %#x' % (got[i], want))
        # Negative control: a wrong expected word for 0 must not match.
        check('dispatch proves firstbitlow(0) is 0xffffffff', got[0] == MASK, hex(got[0]))

        got64 = dispatch_b64_immediates(dxc, runner, adapter, IMMEDIATE_B64, directory)
        for i, word in enumerate(IMMEDIATE_B64):
            want = ff1_32(word)  # hi = 0 by zero-extension
            check('dispatch s_ff1_i32_b64 immediate %#010x (hi=0)' % word, got64[i] == want,
                  'got %#x want %#x' % (got64[i], want))
        gotpair = dispatch_b64_pairs(dxc, runner, adapter, PAIR_B64, directory)
        for i, (lo, hi) in enumerate(PAIR_B64):
            want = ff1_64(lo, hi)
            check('dispatch s_ff1_i32_b64 pair hi=%#010x lo=%#010x' % (hi, lo), gotpair[i] == want,
                  'got %#x want %#x' % (gotpair[i], want))
        # Negative control: a lo-first scan that ignored the high word would answer 0xffffffff here.
        check('dispatch scans the high word when lo is zero', gotpair[PAIR_B64.index((0, 0x10))] == 36,
              hex(gotpair[PAIR_B64.index((0, 0x10))]))
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        sys.exit(1)
    print('dispatch ok')


def mask_fixture():
    import startup_compute_cross_lane_fixture as fixture

    program = b''.join([
        fixture.load(), fixture.WAIT_LOAD,
        fixture.vopc(0xc2, fixture.inline(1), 41),
        sop1('s_ff1_i32_b32', 47, 106),
        fixture.move(45, 47), fixture.store(45),
    ])
    return fixture, program


def check_mask_translation():
    fixture, program = mask_fixture()
    hlsl = fixture.translate(program)
    assert 'vcc;' in hlsl and 's47 = bit;' in hlsl
    fixture.refused('requires a proven wave size', lambda: fixture.translate(program, wave_size=None))
    fixture.refused('requires a group of whole waves', lambda: fixture.translate(program, group=(31, 2, 1)))
    assert '& ~63u' in fixture.translate(program, wave_size=64)


def dispatch_mask(dxc, runner, adapter):
    fixture, program = mask_fixture()
    hlsl = fixture.translate(program)

    def top_bit(shader):
        return shader.replace('uint bit = 0u; bit < 32u; ++bit', 'uint bit = 31u; bit < 32u; --bit')

    for masks in ((0, 1 << 7), (1 << 31, MASK)):
        source = [(mask >> bit) & 1 for mask in masks for bit in range(32)]
        expected = [ff1_32(mask) for mask in masks for _ in range(32)]
        if masks[0] == 0:
            # The descending scan agrees for empty/singleton masks. Run the mutation
            # against a separate pair with a dense mask to distinguish the order.
            rc = fixture.dispatch(dxc, runner, adapter, hlsl, expected, source)
        else:
            mutant = [mask.bit_length() - 1 for mask in masks for _ in range(32)]
            rc = fixture.dispatch(dxc, runner, adapter, hlsl, expected, source, top_bit, mutant)
        if rc:
            return rc
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
        check_mask_translation()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        check_dispatch(*sys.argv[2:])
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch-mask':
        sys.exit(dispatch_mask(*sys.argv[2:]))
    else:
        sys.exit(__doc__)
