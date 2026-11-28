"""buffer_load_format_xyzw of BUF_FMT_10_10_10_2_SSCALED (gfx10 FORMAT 47) from a hardware descriptor, on a device.

usage: startup_vertex_fetch_sscaled_check.py <dxc> <offscreen_check> <adapter>
       startup_vertex_fetch_sscaled_check.py controls

FORMAT 47 packs four signed fields in one dword, named most significant first: x is the 2-bit field in bits 1:0,
y, z and w are 10-bit fields in bits 11:2, 21:12 and 31:22. SSCALED converts each sign-extended field to a float
without dividing (RDNA 2 ISA guide, MTBUF/MUBUF data formats and number formats; the format list is in
tools/disasm_rdna2.py MTBUF_FORMATS). No typed DXGI view has this layout, so only a hardware descriptor reaches it:
the runtime used to refuse it ("unsupported FORMAT 47"). The translated vertex body is called from a compute
wrapper; sixteen elements at an unaligned stride are read back bit for bit against a reference decoded here.
Mutations (sign extension dropped, wrong field order, the switch case removed) must fail the same reference.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import startup_vertex_fetch_formats_check as vf  # noqa: E402

CODE, STRIDE, OFFSET = 47, 6, 2
IDENTITY = 4 | 5 << 3 | 6 << 6 | 7 << 9
FIELDS = ((0, 2), (2, 10), (12, 10), (22, 10))  # (low bit, width) of x, y, z, w


def signed(value, width):
    return value - (1 << width) if value >> (width - 1) else value


def f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def expected(order=(0, 1, 2, 3), extend=True):
    words = [vf.SENTINEL] * 320
    for vertex in range(16):
        address = vertex * STRIDE + OFFSET
        packed = int.from_bytes(vf.DATA[address:address + 4], 'little')
        values = []
        for low, width in FIELDS:
            raw = (packed >> low) & ((1 << width) - 1)
            values.append(f32_bits(signed(raw, width) if extend else raw))
        words[vertex * 4:vertex * 4 + 4] = [values[i] for i in order]
    return words


WRAPPER = '''
RWByteAddressBuffer outputWords : register(u0);
[numthreads(16, 1, 1)]
void readback(uint id : SV_GroupIndex) {
    outputWords.Store4(id * 16, asuint(body(id).pos));
}
'''


def translated():
    return vf.pipeline_mubuf(OFFSET, 3)


def run(directory, compiler, consumer, adapter, source, words, tr):
    hlsl, binary = directory / 'sscaled.hlsl', directory / 'sscaled.cso'
    hlsl.write_text(source + WRAPPER, encoding='ascii')
    subprocess.run([compiler, '-T', 'cs_6_0', '-E', 'readback', '-Fo', binary, hlsl], check=True)
    constants = [0] * 16
    constants[tr.uds.index(('stride', 0))] = STRIDE
    constants[tr.uds.index(('descword3', 0))] = CODE << 12 | IDENTITY
    case = directory / 'sscaled.case'
    case.write_bytes(struct.pack('<337I', 1, *words, *constants))
    return subprocess.run([consumer, adapter, '--compute-store-case', binary, case], capture_output=True, text=True)


def controls():
    source = translated().hlsl_vs()
    assert 'case %du:' % CODE in source, 'FORMAT 47 has no decoder case'
    good = expected()
    # The data must exercise negative fields in every channel, or sign extension would not be tested.
    for channel in range(4):
        assert any(good[v * 4 + channel] >> 31 for v in range(16)), channel
        assert any(not good[v * 4 + channel] >> 31 for v in range(16)), channel
    assert expected(extend=False) != good and expected(order=(1, 0, 2, 3)) != good
    header = (Path(__file__).resolve().parents[1] / 'src' / 'startup_stubs' / 'formatted_buffer.h').read_text()
    assert 'case 47:' in header.split('if (!extended)')[1], 'the runtime must accept FORMAT 47'


def check(directory, compiler, consumer, adapter):
    tr = translated()
    source = tr.hlsl_vs()
    result = run(directory, compiler, consumer, adapter, source, expected(), tr)
    if result.returncode == 77:
        return 77
    assert result.returncode == 0, result.stdout + result.stderr
    print('FORMAT 47 xyzw: 16 elements bit-exact')
    lines = [i for i, line in enumerate(source.split('\n')) if line.strip().startswith('case 47u:')]
    assert len(lines) == 1, lines
    rows = source.split('\n')
    mutants = (
        ('case removed', lambda r: r[:lines[0]] + [r[lines[0]].replace('case 47u:', 'case 255u:', 1)] + r[lines[0] + 1:]),
        ('sign extension dropped', lambda r: r[:lines[0]] + [r[lines[0]].replace('int s = asint', 'int s = 0 + asint', 1)
                                                            .replace('float(s)', 'float(x)')] + r[lines[0] + 1:]),
    )
    for label, mutate in mutants:
        mutated = '\n'.join(mutate(rows))
        assert mutated != source, label
        bad = run(directory, compiler, consumer, adapter, mutated, expected(), tr)
        assert bad.returncode == 1 and 'Compute readback differs' in bad.stderr, (label, bad.stdout, bad.stderr)
        print('mutation %r fails' % label)
    bad = run(directory, compiler, consumer, adapter, source, expected(order=(1, 0, 2, 3)), tr)
    assert bad.returncode == 1 and 'Compute readback differs' in bad.stderr, 'wrong field order passed'
    print('wrong field order fails')
    return 0


if __name__ == '__main__':
    controls()
    if sys.argv[1:] == ['controls']:
        print('sscaled fetch controls passed')
        sys.exit(0)
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
