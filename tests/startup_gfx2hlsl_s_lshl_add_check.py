"""The scalar shift-and-add forms s_lshl1..4_add_u32 in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_s_lshl_add_check.py

D = (S0 << N) + S1 on 32 bits, and SCC = carry out of the wide sum (S0 << N) + S1 (bits shifted out of S0 count).
Words are built from the SOP2 layout tools/disasm_rdna2.py reads: [31:30] = 0b10, [29:23] opcode, [22:16] sdst,
[15:8] ssrc1, [7:0] ssrc0, plus one literal dword when a source is 255. The emitted lines are run in order on the CPU
by the evaluator of startup_gfx2hlsl_bit_alu_check.py; the reference is plain Python integer arithmetic.
SCC starts at a value neither 0 nor 1, so a form that does not write it fails. Pixel fixtures also check the numeric entry mask with WARP readback.
"""
import collections
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
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402
from startup_gfx2hlsl_wave_reduce_check import (  # noqa: E402
    SIZE, dump, export, sop1, sopp, translate as translate_pixel, vop1_mov)
from startup_compute_tg_size_check import vop1  # noqa: E402
from startup_image_3d_check import compile_hlsl  # noqa: E402

MASK = 0xffffffff
OPCODE = {name: op for op, (name, _) in D.SOP2_OPS.items()}
NAMES = {n: 's_lshl%d_add_u32' % n for n in (1, 2, 3, 4)}
SPECIAL = [0, MASK, 0xaaaaaaaa, 0x55555555, 0x80000000, 0x7fffffff, 1, 0xfffffffe, 0xf0000000, 0x0fffffff,
           0x12345678, 0xdeadbeef]
VALUES = SPECIAL + [1 << k for k in range(32)] + [~(1 << k) & MASK for k in range(32)]
PAIRS = [(a, b) for a in SPECIAL for b in VALUES] + [(a, b) for a in VALUES for b in SPECIAL]
SCC_UNSET = 2
VCC_LO, M0, NULL, EXEC_LO, LITERAL = 106, 124, 125, 126, 255
INLINE = {128: 0, 129: 1, 193: MASK}
# (sdst, ssrc0, ssrc1, literal)
FORMS = [(3, 1, 2, None), (1, 1, 2, None), (2, 1, 2, None), (3, 1, 1, None), (3, 128, 2, None),
         (3, 193, 2, None), (3, 1, 129, None), (3, LITERAL, 2, 0xaaaaaaaa), (3, 1, LITERAL, 0x55555555),
         (M0, 1, 2, None)]
MASK_FORMS = [(3, VCC_LO, 2, None), (3, 1, EXEC_LO, None), (VCC_LO, 1, 2, None), (EXEC_LO, 1, 2, None)]

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def sop2(name, sdst, ssrc0, ssrc1, literal=None):
    word = (0b10 << 30) | (OPCODE[name] << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0
    return struct.pack('<I', word) + (b'' if literal is None else struct.pack('<I', literal))


class Probe(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True


def translate(words):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    stage = Probe()
    try:
        stage.one(decoded[0][2], decoded[0][3])
    except G.Unsupported as error:
        stage.lines = ['refused: %s' % error]
    return stage.lines


def source_value(code, literal, env):
    if code == LITERAL:
        return literal
    return INLINE[code] if code in INLINE else env['s%d' % code]


def run(lines, env):
    env = dict(env)
    for line in lines:
        m = re.fullmatch(r'(\w+) = (.*);', line)
        assert m, line
        env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


def reference(n, a, b):
    wide = (a << n) + b
    return wide & MASK, int(wide > MASK)


def first_mismatch(n):
    name = NAMES[n]
    for sdst, ssrc0, ssrc1, literal in FORMS:
        lines = translate(sop2(name, sdst, ssrc0, ssrc1, literal))
        if lines[0].startswith('refused: '):
            return name, sdst, ssrc0, ssrc1, lines[0]
        dest = 'm0' if sdst == M0 else 's%d' % sdst
        for a, b in PAIRS:
            env = {'s1': a, 's2': b, 'scc': SCC_UNSET}
            want = reference(n, source_value(ssrc0, literal, env), source_value(ssrc1, literal, env))
            out = run(lines, env)
            if (out.get(dest), out['scc']) != want:
                return name, sdst, ssrc0, ssrc1, hex(a), hex(b), (out.get(dest), out['scc']), want
    return None


# Packed four-entry table. The branch on SCC changes the green channel.
TABLE = (7, 23, 11, 39)
LEFT = 0x80000001
TRIANGLE = """float4 main(uint id : SV_VertexID) : SV_Position {
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
}
"""


def pixel_program(n):
    return b''.join((
        sop1(3, 12, EXEC_LO),  # save entry EXEC in the same SGPR overwritten by the result
        sop2(NAMES[n], 12, LITERAL, 12, LEFT),
        vop1_mov(4, 128),
        sopp(0x04, 1),  # s_cbranch_scc0 skips the green-channel update
        vop1(0x06, 4, 129),
        sop2('s_and_b32', 13, 12, 131),
        sop2('s_lshl_b32', 13, 13, 131),
        sop2('s_lshr_b32', 14, LITERAL, 13, int.from_bytes(bytes(TABLE), 'little')),
        sop2('s_and_b32', 14, 14, LITERAL, 255),
        vop1(0x06, 5, 14),
        vop1_mov(6, 128),
        b''.join(word for _, word in export('v5', 'v4', 'v6', 'v6')),
    ))


def pixel_source(n):
    source, reason = translate_pixel(pixel_program(n))
    assert source is not None, reason
    return source


def observe_waves(source):
    # Record lane index and a unique wave ID independently of the ballot.
    assignment = re.search(r'(^ +)uint shift_mask_.* = WaveActiveBallot\(true\)\.x;$', source, re.M)
    assert assignment, source
    indent = assignment[1]
    observation = (indent + 'x2w_test_wave = WaveGetLaneIndex() | '
                   f'(WaveActiveMin(uint(i.pos.x) + {SIZE}u * uint(i.pos.y)) << 7);\n')
    source = source[:assignment.start()] + observation + source[assignment.start():]
    assert source.count('    return o;') == 1
    source = source.replace('    return o;', '    o.mrt0.w = float(x2w_test_wave);\n    return o;')
    declaration = '    PSOut o = (PSOut)0;'
    assert source.count(declaration) == 1
    return source.replace(declaration, declaration + '\n    uint x2w_test_wave = 0u;')


def pixel_mismatches(n, raw):
    pixels = list(struct.iter_unpack('<4f', raw))
    assert len(pixels) == SIZE * SIZE, len(pixels)
    waves = collections.defaultdict(list)
    for colour, green, blue, tag in pixels:
        assert tag >= 0 and tag.is_integer(), ('uncovered pixel or invalid tag', tag)
        wave, lane = divmod(int(tag), 128)
        waves[wave].append((lane, (colour, green, blue)))
    mismatches = []
    for members in waves.values():
        lanes = {lane for lane, _ in members}
        assert len(lanes) == len(members), members
        mask = sum(1 << lane for lane in lanes if lane < 32)
        result, carry = reference(n, LEFT, mask)
        want = (TABLE[result & 3], carry, 0)
        mismatches.extend((got, want) for _, got in members if got != want)
    return mismatches


def without_pixel_carry(source):
    text, replaced = re.subn(r'^( +scc = ).*shift_mask_.*;$', r'\g<1>false;', source, flags=re.M)
    assert replaced == 1, source
    return text


def draw_pixels(dxc, renderer):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', TRIANGLE, 'triangle', 'vs_6_0', directory)
        for n in NAMES:
            text = observe_waves(pixel_source(n))
            for mutant in (False, True):
                source = without_pixel_carry(text) if mutant else text
                ps = compile_hlsl(dxc, 'dxc', source, 'shift_add', 'ps_6_0', directory)
                out = directory / 'pixels.bin'
                out.unlink(missing_ok=True)
                proc = dump(renderer, 'warp', vs, ps, out)
                assert proc.returncode == 0, proc.stdout + proc.stderr
                print(proc.stdout, end='')
                assert 'debug_layer=enabled' not in proc.stdout or 'debug_errors=0 debug_warnings=0' in proc.stdout
                errors = pixel_mismatches(n, out.read_bytes())
                assert bool(errors) == mutant, (n, mutant, errors[:1])
                print(f'pixel shift {n}, mutant={mutant}: {len(errors)} mismatches')


def controls():
    for n in NAMES:
        mismatch = first_mismatch(n)
        check('%s result and carry across %d forms and %d pairs' % (NAMES[n], len(FORMS), len(PAIRS)),
              mismatch is None, str(mismatch))

    for n in NAMES:
        refused = [translate(sop2(NAMES[n], *form))[0] for form in MASK_FORMS]
        check('%s refuses lane-mask sources and destinations' % NAMES[n],
              all(r == f'refused: {NAMES[n]} on a lane mask: wave word not available in this stage'
                  for r in refused), str(refused))
        refused_null = translate(sop2(NAMES[n], NULL, 129, 129))
        check('%s refuses the discard destination without raising a translator error' % NAMES[n],
              refused_null == ['refused: %s destination null' % NAMES[n]], str(refused_null))
        stage = Probe()
        stage.varying_s = {1}
        stage.scc_uniform = True
        stage.one(NAMES[n], f'{NAMES[n]} s3, s1, s2')
        check('%s clears a stale SCC uniformity proof' % NAMES[n], not stage.scc_uniform)

    for n in NAMES:
        without_pixel_carry(observe_waves(pixel_source(n)))

    if failures:
        print('FAILED: %s' % ', '.join(failures))
        sys.exit(1)
    print('all ok')


if __name__ == '__main__':
    controls()
    if len(sys.argv) > 1:
        mode, *args = sys.argv[1:]
        if mode != 'draw' or len(args) != 2:
            raise SystemExit('usage: startup_gfx2hlsl_s_lshl_add_check.py [draw <dxc> <offscreen_check>]')
        draw_pixels(*args)
