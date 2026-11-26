"""s_bitreplicate_b64_b32 in tools/startup/gfx2hlsl.py (Run R key 3c50f16a).

usage: startup_gfx2hlsl_s_bitreplicate_check.py
       startup_gfx2hlsl_s_bitreplicate_check.py dispatch <dxc> <offscreen_check> <warp|hardware>

RDNA 2 ISA SOP1: every bit i of the 32-bit source S0 is written twice, to bits 2i and 2i+1 of the 64-bit destination
pair (low dword = source bits 15:0, high dword = bits 31:16). SCC is not modified. An integer inline constant or a
32-bit literal is the source and folds. Lane masks, descriptor sources, odd pairs and a source that is one of the
destination registers are refused. The reference is plain Python; the translated text is evaluated on the CPU.

With `dispatch`, the lowering runs as a compute shader on the adapter (D3D12 UAV readback through offscreen_check,
same pattern as startup_gfx2hlsl_s_quadmask_check). Each case stores both destination dwords and an SCC that was set
before the instruction (it must stay set).
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
WORDS = (0x00000000, 0xffffffff, 0x00000001, 0x80000000, 0x0000ffff, 0xffff0000, 0x12345678, 0xa5c3f00f)
# GPU cases: zero, lowest bit, bit 15 / 16 (the word boundary), highest bit, an irregular word.
DRAW = (0x00000000, 0x00000001, 0x00008000, 0x00010000, 0x80000000, 0x9a5c3e71)
VCC_LO = 106

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def sop1(name, sdst, ssrc, literal=None):
    words = struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (SOP1[name] << 8) | ssrc)
    if literal is not None:
        words += struct.pack('<I', literal & MASK)
    return words


def replicate(word):
    """The 64-bit result as (low dword, high dword): bit i of word -> bits 2i and 2i+1."""
    out = 0
    for bit in range(32):
        if (word >> bit) & 1:
            out |= 3 << (2 * bit)
    return out & MASK, out >> 32


class Probe(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True


def translate(words, masks=(), sym=None):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    stage = Probe()
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
    lines, _ = translate(sop1('s_bitreplicate_b64_b32', 4, 2))
    assert lines and not lines[0].startswith('refused'), lines
    bad = None
    for word in WORDS:
        out = run(lines, {'s2': word, 'scc': 1})
        if (out['s4'], out['s5'], out['scc']) != (*replicate(word), 1):
            bad = (hex(word), (out['s4'], out['s5'], out['scc']), replicate(word))
            break
    check('s_bitreplicate_b64_b32 SGPR source, SCC unchanged', bad is None, str(bad))
    check('lowering names no scc write', not any('scc' in line for line in lines), str(lines))

    bad = None
    for word in WORDS:
        lines_l, _ = translate(sop1('s_bitreplicate_b64_b32', 4, 255, word))
        if lines_l[0].startswith('refused'):
            bad = ('refused %s for %s' % (lines_l[0], hex(word)),)
            break
        out = run(lines_l, {'scc': 1})
        if (out['s4'], out['s5']) != replicate(word):
            bad = (hex(word), (out['s4'], out['s5']), replicate(word))
            break
    check('s_bitreplicate_b64_b32 32-bit literal folds', bad is None, str(bad))

    # SSRC 193 and 208 decode as the inline integers -1 and -16.
    for ssrc, word in ((193, 0xffffffff), (208, 0xfffffff0)):
        lines_i, _ = translate(sop1('s_bitreplicate_b64_b32', 4, ssrc))
        out = run(lines_i, {'scc': 1})
        check('s_bitreplicate_b64_b32 inline SSRC %d' % ssrc, (out['s4'], out['s5']) == replicate(word),
              str((out['s4'], out['s5'], replicate(word))))

    lines_c, sym = translate(sop1('s_bitreplicate_b64_b32', 4, 2), sym={2: ('const', 0x80010001)})
    lo, hi = replicate(0x80010001)
    check('s_bitreplicate_b64_b32 folds a constant register',
          sym.get(4) == ('const', lo) and sym.get(5) == ('const', hi), str((lines_c, sym.get(4), sym.get(5))))

    refused = []
    for words, masks, sym_in in (
            (sop1('s_bitreplicate_b64_b32', VCC_LO, 2), (), None),
            (sop1('s_bitreplicate_b64_b32', 4, VCC_LO), (), None),
            (sop1('s_bitreplicate_b64_b32', 4, 2), (2,), None),
            (sop1('s_bitreplicate_b64_b32', 4, 2), (), {2: ('desc', 0, 0, 0)}),
            (sop1('s_bitreplicate_b64_b32', 3, 2), (), None),  # odd destination
            (sop1('s_bitreplicate_b64_b32', 4, 4), (), None),  # source is the low destination register
            (sop1('s_bitreplicate_b64_b32', 4, 5), (), None),  # source is the high destination register
    ):
        refused.append(translate(words, masks=masks, sym=sym_in)[0][0])
    check('s_bitreplicate_b64_b32 refuses lane masks, descriptors, odd pairs and a source inside the pair',
          all(r.startswith('refused:') for r in refused), str(refused))

    # Negative controls: each mutation of the lowering must change a result.
    mutations = (
        ('writes bit 0 once', [lines[0].replace('0x3u', '0x1u', 1), lines[1]]),
        ('writes bit 15 once', [lines[0].replace('0xc0000000u', '0x40000000u', 1), lines[1]]),
        ('writes bit 16 once', [lines[0], lines[1].replace('0x3u', '0x1u', 1)]),
        ('swaps the halves', [lines[1].replace('s5 =', 's4 =', 1), lines[0].replace('s4 =', 's5 =', 1)]),
    )
    for name, mutated in mutations:
        diverged = False
        for word in WORDS:
            out = run(mutated, {'s2': word, 'scc': 1})
            diverged = diverged or (out['s4'], out['s5']) != replicate(word)
        check('mutation that %s diverges' % name, diverged and mutated != lines)


def dispatch_bitreplicate(dxc, runner, adapter, cases, directory):
    """Translate once; feed each case through a UAV and read both destination dwords and SCC."""
    lines, _ = translate(sop1('s_bitreplicate_b64_b32', 4, 2))
    assert lines and not lines[0].startswith('refused'), lines
    fill = tuple(((i + 1) * 0x9e3779b1 & MASK) | 1 for i in range(len(cases)))
    result_width = 3
    hlsl = ['RWByteAddressBuffer data : register(u0);', '[numthreads(1, 1, 1)]', 'void main() {']
    for i, word in enumerate(cases):
        dst_byte = 4 * (len(fill) + result_width * i)
        hlsl.extend((
            '    {',
            '        uint s2 = data.Load(%d) ^ %s;' % (4 * i, '%#010xu' % (fill[i] ^ word)),
            '        uint s4, s5;',
            '        bool scc = true;',
        ))
        hlsl.extend('        ' + line for line in lines)
        hlsl.append('        data.Store(%d, s4);' % dst_byte)
        hlsl.append('        data.Store(%d, s5);' % (dst_byte + 4))
        hlsl.append('        data.Store(%d, uint(scc));' % (dst_byte + 8))
        hlsl.append('    }')
    hlsl.append('}')

    src, cso, output = (directory / ('s_bitreplicate' + suffix) for suffix in ('.hlsl', '.cso', '.bin'))
    src.write_text('\n'.join(hlsl) + '\n', encoding='ascii')
    compile_result = subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)],
                                    stderr=subprocess.PIPE, text=True)
    if compile_result.returncode or compile_result.stderr:
        raise AssertionError(compile_result.stderr or 'DXC exited with %d' % compile_result.returncode)
    word_count = len(fill) + result_width * len(cases)
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


def check_dispatch(dxc, runner, adapter):
    controls()
    runner = str(Path(runner).resolve())
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        got = dispatch_bitreplicate(dxc, runner, adapter, DRAW, Path(tmp))
        for i, word in enumerate(DRAW):
            lo, hi = replicate(word)
            base = 3 * i
            label = 'dispatch s_bitreplicate_b64_b32 %#010x' % word
            check('%s low dword' % label, got[base] == lo, 'got %#x want %#x' % (got[base], lo))
            check('%s high dword' % label, got[base + 1] == hi, 'got %#x want %#x' % (got[base + 1], hi))
            check('%s leaves SCC set' % label, got[base + 2] == 1, 'got %#x' % got[base + 2])
        check('dispatch bit 15 lands in the low dword top', got[3 * 2] == 0xc0000000 and got[3 * 2 + 1] == 0,
              str((hex(got[6]), hex(got[7]))))
        check('dispatch bit 16 lands in the high dword bottom', got[3 * 3] == 0 and got[3 * 3 + 1] == 3,
              str((hex(got[9]), hex(got[10]))))
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        sys.exit(1)
    print('dispatch ok')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
        if failures:
            print('FAILED: %s' % ', '.join(failures))
            sys.exit(1)
        print('all ok')
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        check_dispatch(*sys.argv[2:])
    else:
        sys.exit(__doc__)
