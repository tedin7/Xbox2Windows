"""Check GETPC refusals and draw a program-embedded vertex table on WARP."""
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_calculated_target_check import POSITIVE, assert_result, translate  # noqa: E402

import gpu_case_runner  # noqa: E402
from startup_flat_param_check import compile_hlsl, root_blob  # noqa: E402

REASON = ('s_getpc_b64: the shader load address is only elided for a program that also calculates a branch target')
failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH  ' + detail))
    if not condition:
        failures.append(label)


def probe(kind):
    class Probe(G.Stage):
        """A Stage with no pipeline: enough state for one program to be translated."""

        def __init__(self):
            self.lines, self.sym, self.attrs = [], {}, {}
            self.used_s, self.used_v, self.uniform_v = set(), set(), set()
            self.masks = set()
            self.exec_known, self.kind = True, kind

    return Probe()


def refusal(lines, kind='ps'):
    stage = probe(kind)
    stage.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(lines)]
    try:
        stage.translate()
    except G.Unsupported as reason:
        return str(reason), stage.lines
    return None, stage.lines


# The forms a captured request can carry: the instruction alone, with the low/high word arithmetic that makes the
# pair a pointer, and with the pair used directly as a descriptor.
FORMS = (
    ['s_getpc_b64 s[4:5]'],
    ['s_getpc_b64 s[4:5]', 's_add_u32 s4, s4, 0x10', 's_addc_u32 s5, s5, 0'],
    ['s_getpc_b64 s[4:5]', 's_load_dword s0, s[4:5], 0x0'],
)
def refusal_controls():
    for kind in ('ps', 'cs'):
        for lines in FORMS:
            reason, _ = refusal(lines, kind)
            check('%s: %s names the reason' % (kind, ' ; '.join(lines)[:52]), reason == REASON, repr(reason))

    # The refusal is not the bare mnemonic: that is the regression this check exists for.
    reason, _ = refusal(['s_getpc_b64 s[4:5]'])
    check('the refusal is longer than the mnemonic', reason != 's_getpc_b64', repr(reason))
    check('the refusal mentions the load address', reason is not None and 'load address' in reason, repr(reason))

    # A proven control must translate and skip the decoy.
    assert_result(translate(POSITIVE), (0, 20, 24), 24)
    print('control: the proven calculated jump translates and skips the decoy: ok')

    print('getpc refusal checks: %d failed' % len(failures))
    if failures:
        raise SystemExit(1)


ROWS = 8
STRIDE = 16
TABLE_SIZE = ROWS * STRIDE
FLAGS = 0x31014FAC  # raw bounds, resource level 1, 32_UINT, identity selectors
DESC = 20
# Include a nonzero row after the declared table for the bounds and shifted-address controls.
DATA = b''.join(struct.pack('<I', 0x3e414243 + row * 0x00010101 + channel * 0x00020202)
                for row in range(ROWS + 1) for channel in range(4))


def words(*values):
    return struct.pack('<%dI' % len(values), *values)


def mubuf(opcode, dst, offset):
    return words(0xe0000000 | opcode << 18 | 1 << 12 | offset,
                 128 << 24 | (DESC // 4) << 16 | dst << 8 | 3)


def export(target, registers, done=False):
    sources = int.from_bytes(bytes(registers), 'little')
    return words(0xf800000f | int(done) << 11 | target << 4, sources)


def vertex(*, shift=0, size=TABLE_SIZE, flags=FLAGS, high=None, byte_skew=0, extra=b'',
           skip_setup=False, branch_to_load=False, short_size=False):
    # v0 is supplied by the test's vertex entry point as 4 * instance + corner.
    # Compute the row number and then its byte offset, leaving v4..v7 for its four words.
    getpc = words(0xbe801f00 | DESC << 16)
    size_word = words(0xb0000000 | (DESC + 2) << 16 | size) if short_size else words(
        0xbe8003ff | (DESC + 2) << 16, size)
    size_and_flags = size_word + words(0xbe8003ff | (DESC + 3) << 16, flags)
    carry = b'' if high is None else words(0x82008000 | (DESC + 1) << 16 | (DESC + 1) | high << 8)
    shifts = words(0x16 << 25 | 2 << 17 | 130, 0x1a << 25 | 3 << 17 | 2 << 9 | 132)
    loads = b''.join(mubuf(0x0c, 4 + channel, 4 * channel + byte_skew) for channel in range(4))
    before_loads = words(0xbf820001, 0xbf800000) if branch_to_load else b''
    tail = (carry + shifts + before_loads + loads + extra + words(0xbf8c0000, 0x7e1002f2, 0x7e120280)
            + export(12, (9, 9, 9, 8), done=True) + export(32, (4, 5, 6, 7)) + words(0xbf810000))
    # This padding deliberately looks like a second terminator and an invalid end-of-blob instruction.
    padding = words(0xbf810000)
    branch = words(0xbf820007) if skip_setup else b''
    disp = len(size_and_flags) + len(tail) + len(padding) + 8 + shift
    add = words(0x8000ff00 | DESC << 16 | DESC, disp)
    return G.PREFETCH + branch + getpc + size_and_flags + add + tail + padding + DATA + words(0xf8000000)


def pipeline(code, decoded=None, refusals=None):
    # A minimal pixel stage supplies the normal pipeline linkage. The GPU fixture's pixel
    # stage below simply forwards the vertex color without interpolation.
    pixel = export(0, (0, 1, 2, 3), done=True) + words(0xbf810000)
    return G.Pipeline(code, pixel, root_blob(), decoded=decoded, refusals=refusals)


def reference(size=TABLE_SIZE, shift=0, byte_skew=0):
    return b''.join(DATA[shift + addr:shift + addr + 4] if addr + 4 <= size else bytes(4)
                    for row in range(ROWS) for channel in range(4)
                    for addr in (row * STRIDE + channel * 4 + byte_skew,))


def table_controls():
    code = vertex()
    pipe = pipeline(code)
    stage = pipe.vs
    setup, loads, size, table = stage.embedded_buffer
    assert len(setup) == 4 and len(loads) == 4 and size == TABLE_SIZE
    assert words(*table) == DATA[:TABLE_SIZE]
    assert stage.raw_code[stage.ins[-1][0] + 4:][:4] == words(0xbf810000)
    assert G.classify(code) == 'vs'
    encoding_refusals = []
    decoded = G.decode(code, encoding_refusals, embedded_vertex=True)
    assert not encoding_refusals
    assert pipeline(code, {'vs': decoded}, encoding_refusals).hlsl_vs() == pipeline(code).hlsl_vs()
    assert not encoding_refusals
    assert pipeline(b'\0' * 32 + code).hlsl_vs() == pipeline(code).hlsl_vs()
    source = pipe.hlsl_vs()
    assert 'static const uint embedded_vs[32]' in source and not pipe.resources
    assert source.count('= embedded_vs_load(') == 4
    assert all('s%d' % reg not in source for reg in range(DESC, DESC + 4))
    assert 'embedded_vs' not in pipe.hlsl_ps()
    assert pipeline(vertex(high=0)).hlsl_vs() == source
    assert pipeline(vertex(short_size=True)).hlsl_vs() == source
    assert pipeline(vertex(branch_to_load=True)).vs.embedded_buffer
    assert G.lifted_family(REASON) == 'embedded-vertex-table'
    assert reference(shift=16) != reference()
    assert reference(size=7 * STRIDE)[-STRIDE:] == bytes(STRIDE)
    assert DATA[7 * STRIDE:8 * STRIDE] != bytes(STRIDE)
    for label, bad in {
        'range past the blob': vertex(size=len(DATA) + 8),
        'range overlaps instructions': vertex(shift=-8),
        'misaligned table start': vertex(shift=1),
        'nonzero high add': vertex(high=1),
        'formatted descriptor': vertex(flags=FLAGS ^ (1 << 12) ^ (1 << 14)),
        'non-raw bounds': vertex(flags=FLAGS ^ (1 << 28)),
        'raw store': vertex(extra=mubuf(0x1c, 4, 0)),
        'formatted load': vertex(extra=mubuf(0x00, 10, 0)),
        'numeric read': vertex(extra=words(0x7e140200 | DESC)),
        'branch bypasses setup': vertex(skip_setup=True),
        'calculated branch': vertex(extra=words(0xbe802000 | DESC)),
    }.items():
        try:
            pipeline(bad)
        except G.Unsupported as error:
            if label == 'calculated branch':
                assert str(error).startswith('s_setpc_b64: calculated PC word escapes into buffer_load_dword'), error
            else:
                assert str(error) == REASON, (label, error)
            print('%s refused: %s' % (label, error))
        else:
            raise AssertionError('%s was accepted' % label)


def draw_tables(compiler, renderer, kind='dxc'):
    # The production vertex ABI takes a single vertex index. The fixture packs the
    # instance ID into it and supplies a triangle covering just that instance's pixel.
    entry = """
struct DrawOut {
    float4 pos : SV_Position;
    nointerpolation float4 color : COLOR;
};
DrawOut main(uint corner : SV_VertexID, uint row : SV_InstanceID) {
    VSOut translated = body(4u * row + corner);
    DrawOut o;
    o.pos = float4(-1.0 + 0.25 * row + (corner == 1u ? 0.5 : 0.0),
                   corner == 2u ? 1.0 : -1.0, 0.0, 1.0);
    o.color = translated.param0;
    return o;
}
"""
    # SV_Position first: the vertex outputs it at o0, so COLOR must be register 1 in both stages.
    pixel = 'float4 main(float4 position : SV_Position, nointerpolation float4 color : COLOR) : SV_Target { return color; }'
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        model = '5_1' if kind == 'fxc' else '6_0'
        ps = compile_hlsl(compiler, kind, pixel, 'color', 'ps_' + model, directory)
        expected = directory / 'expected.bin'

        def draw(vs, cpu, differing=0):
            expected.write_bytes(cpu)
            run = gpu_case_runner.run(renderer, 'warp', ['--startup-instanced-pixel-words', vs, ps, '8', '1', expected])
            print(run.stdout + run.stderr, end='')
            if run.returncode == 77:
                raise SystemExit(77)
            if differing:
                assert run.returncode != 0 and f'{differing} of 8 pixels differ' in run.stdout + run.stderr
            else:
                assert run.returncode == 0

        for label, kwargs in (
                ('table', {}), ('shifted', {'shift': STRIDE}),
                ('bounded', {'size': 7 * STRIDE}), ('unaligned', {'byte_skew': 1}),
                ('carry', {'high': 0})):
            source = pipeline(vertex(**kwargs)).hlsl_vs().replace('VSOut main(', 'VSOut guest_main(', 1)
            vs = compile_hlsl(compiler, kind, source + entry, label, 'vs_' + model, directory)
            draw(vs, reference(**{k: v for k, v in kwargs.items() if k != 'high'}))
            if label == 'shifted':
                draw(vs, reference(), differing=8)
            elif label == 'bounded':
                draw(vs, reference(), differing=1)


if __name__ == '__main__':
    refusal_controls()
    table_controls()
    if len(sys.argv) > 1:
        dxc, renderer, *fxc = sys.argv[1:]
        draw_tables(dxc, renderer)
        if fxc and fxc[0]:
            draw_tables(fxc[0], renderer, 'fxc')
    print('PASS')
