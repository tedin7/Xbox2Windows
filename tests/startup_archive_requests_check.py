"""archive_requests.py on a synthetic archive (pure Python, no Windows, no GPU, no title files).

Programs follow the measured layout: a 36-byte header, s_inst_prefetch, code, s_code_end padding to
64 bytes, each on a 256-byte step of a zlib stream that starts on a 64-byte boundary. Each pixel
program directly followed by a vertex program must become one request per root, keyed exactly as
the runtime keys it; the expected keys are computed here with an independent FNV-1a, not with the
translator's function.

Controls that must write nothing: a pixel program followed by another pixel program, a vertex
program with no pixel program before it, a vertex program one empty 256-byte step after a pixel
program, a pixel program without s_endpgm, a program without the header tag, a pair at an offset
that is not a multiple of 4, a program that runs to the end of its stream, a valid pair in a stream
that is not on a 64-byte boundary, an empty or duplicate root file, a stream that inflates past the
limit, and a run with no root signature. A pair split between two consecutive zlib streams must be
carved whole. A request whose .root cannot be written must leave no .a and no temporary file and make
the run fail. An archive without any pair must make the run fail. 65 MiB of consecutive streams
must be split into two joined runs. A second run must write nothing new.
"""
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / 'tools' / 'startup' / 'archive_requests.py'

REVISION = re.search(rb'constexpr char revision\[\] = "([^"]+)";',
                     (ROOT / 'src' / 'startup_stubs' / 'd3d12_draw.inl').read_bytes()).group(1)
PREFETCH = bytes.fromhex('0300a0bf')  # s_inst_prefetch 0x3
CODE_END = bytes.fromhex('00009fbf')
ENDPGM = struct.pack('<I', 0xbf810000)
failures = []


def expect(label, got, want):
    if got != want:
        failures.append('%s: got %r, want %r' % (label, got, want))


def runtime_key(a, b, root):
    """The runtime's key (fnv_bytes and cached_translation in src/startup_stubs/d3d12_draw.inl): FNV-1a 64
    over the revision text, then the first recorded blob, the second and the root signature. The
    runtime's start value is 1469598103934665603, not the standard offset basis 14695981039346656037;
    the cache keys depend on it, so it is reproduced as written. The revision text is read from the
    runtime source, so a revision bump there is followed here and checked against the tool."""
    h = 1469598103934665603
    for part in (REVISION, a, b, root):
        for byte in part:
            h = ((h ^ byte) * 1099511628211) & 0xffffffffffffffff
    return h


def program(stage, serial, blocks=1, tag=b'\x80\xd5', extra=b''):
    target = {'vs': 12, 'ps': 0}[stage]  # exp pos0 / exp mrt0, then s_endpgm
    header = b'\x00\x00' + tag + struct.pack('<I', serial) + bytes(28)
    body = header + PREFETCH + extra + struct.pack('<2I', 0xf800080f | target << 4, 0) + ENDPGM
    size = 0x40 * blocks
    return body + CODE_END * ((size - len(body)) // 4)


def stream(layout, length=0x1000):
    """layout: [(offset, program)], offsets on 256-byte steps from 0x2c as in the measured streams."""
    data = bytearray(length)
    for at, blob in layout:
        data[at:at + len(blob)] = blob
    return bytes(data)


def summary(text):
    return dict(item.split('=', 1) for item in text.split() if '=' in item)


PS1, VS1 = program('ps', 1), program('vs', 2, blocks=2)
PS2, PS3, VS3, VS4 = program('ps', 3), program('ps', 4), program('vs', 5), program('vs', 6)
PS_TAIL, VS_TAIL = program('ps', 7), program('vs', 12, blocks=2)  # VS_TAIL ends where the stream ends
# VS5 holds v_cndmask_b32_e64 v1, -1, v0, vcc_lo with NEG on the inline -1 (VOP3 opcode 0x101,
# src0 193, src1 v0, src2 vcc_lo, neg bit 0): a form the translator refuses while decoding. Pairing
# must depend on the export targets only, not on whether the program can be translated.
CNDMASK_NEG_IMM = struct.pack('<2I', 0xd5010001, 193 | 256 << 9 | 106 << 18 | 1 << 29)
PS5, VS5 = program('ps', 8), program('vs', 9, extra=CNDMASK_NEG_IMM)
PS6, VS6 = program('ps', 10), program('vs', 11)

first = stream([(0x02c, PS1), (0x12c, VS1), (0x22c, PS2), (0x32c, PS3), (0x42c, VS3), (0x52c, VS4),
                (0x62c, PS_TAIL), (0x72c, VS_TAIL)], length=0x72c + len(VS_TAIL))
second = stream([(0x02c, PS5), (0x12c, VS5)])
misaligned = stream([(0x02c, PS6), (0x12c, VS6)])

# A pair split between two zlib streams, the second starting on the 64-byte boundary right after the
# first, must be carved whole.
PS8, VS8 = program('ps', 13), program('vs', 14, blocks=2)
split = stream([(0x02c, PS8), (0x12c, VS8)], length=0x300)

# Must not pair: a vertex program one empty 256-byte step after its pixel program, and a vertex
# program after a pixel program that has no s_endpgm (cut short by an all-zero block).
PS9, VS9 = program('ps', 15), program('vs', 16)
cut = program('ps', 17).replace(ENDPGM, CODE_END)
VS10 = program('vs', 18)
gaps = stream([(0x02c, PS9), (0x22c, VS9), (0x32c, cut), (0x42c, VS10)], length=0x600)

# Must not pair: a pixel program without the header tag, and a pair at offsets that are not a
# multiple of 4 (whose bytes would otherwise form a complete, adjacent pair).
untagged, VS11 = program('ps', 19, tag=b'\x00\x00'), program('vs', 20)
PS12, VS12 = program('ps', 21), program('vs', 22)
decoys = stream([(0x02c, untagged), (0x12c, VS11), (0x32e, PS12), (0x42e, VS12)], length=0x600)

# A stream that inflates to 2 MiB: refused as oversized, never inflated in full.
oversized = zlib.compress(bytes(0x200000), 9)


def aligned(archive, alignment=0):
    while len(archive) % 64 != alignment:
        archive.append(0)


archive = bytearray()
for blob, offset in ((first, 0), (misaligned, 32), (second, 0)):
    aligned(archive, offset)
    archive += zlib.compress(blob, 9)
for blob in (gaps, decoys):
    aligned(archive)
    archive += bytes(64)  # a gap keeps each stream apart from the previous one
    archive += zlib.compress(blob, 9)
aligned(archive)
archive += bytes(64) + oversized
aligned(archive)
archive += bytes(64)
for piece in (split[:0x150], split[0x150:]):
    aligned(archive)
    archive += zlib.compress(piece, 9)

roots = [b'DXBC-root-one', b'DXBC-root-two']
want = {runtime_key(a, b, r): (a, b, r) for a, b in ((PS1, VS1), (PS3, VS3), (PS5, VS5), (PS8, VS8)) for r in roots}

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / 'archive.bin').write_bytes(archive)
    (tmp / 'nopairs.bin').write_bytes(zlib.compress(stream([(0x02c, PS2), (0x12c, PS3)]), 9))
    rootdir = tmp / 'roots'
    rootdir.mkdir()
    for i, r in enumerate(roots + roots[:1] + [b'']):  # a duplicate and an empty root must be ignored
        (rootdir / ('req_%016x.root' % i)).write_bytes(r)
    cache = tmp / 'cache'

    def run(cache_dir, *extra, archive_name='archive.bin'):
        return subprocess.run([sys.executable, '-B', str(TOOL), str(tmp / archive_name), str(cache_dir), *extra],
                              capture_output=True, text=True, timeout=120)

    missing = run(cache, '--roots', str(tmp / 'absent'))
    expect('missing roots path: refused', missing.returncode != 0 and 'not found' in missing.stderr, True)
    cache.mkdir()  # an existing cache without any req_*.root
    empty = run(cache)
    expect('no roots: refused with a message', empty.returncode != 0 and 'no root signature' in empty.stderr, True)
    expect('no roots: requests written', sorted(cache.glob('req_*')), [])

    out = run(cache, '--roots', str(rootdir))
    expect('run exit status', out.returncode, 0)
    got = {int(p.name[4:20], 16) for p in cache.glob('req_*.a')}
    expect('request keys', sorted(got), sorted(want))
    for key, (a, b, r) in want.items():
        name = cache / ('req_%016x' % key)
        parts = tuple(name.with_suffix(ext).read_bytes() if name.with_suffix(ext).exists() else None
                      for ext in ('.a', '.b', '.root'))
        expect('request %016x parts' % key, parts, (a, b, r))
    expect('no temporary files left', sorted(p.name for p in cache.glob('*.tmp')), [])
    counts = summary(out.stdout)
    expect('summary', {k: counts.get(k) for k in ('streams', 'oversized', 'incomplete', 'pairs', 'written', 'roots')},
           {'streams': '5', 'oversized': '1', 'incomplete': '1', 'pairs': '4', 'written': '8', 'roots': '2'})

    again = run(cache, '--roots', str(rootdir))
    expect('second run', {k: summary(again.stdout).get(k) for k in ('written', 'existing')}, {'written': '0', 'existing': '8'})

    # A .root that cannot be written (a directory stands in the way): no .a, no temporary file, exit 1.
    blocked = tmp / 'blocked'
    blocked.mkdir()
    stuck = runtime_key(PS1, VS1, roots[0])
    (blocked / ('req_%016x.root' % stuck)).mkdir()
    jam = run(blocked, '--roots', str(rootdir))
    expect('blocked write: run fails', jam.returncode != 0 and 'could not be written' in jam.stderr, True)
    expect('blocked write: no ready marker', (blocked / ('req_%016x.a' % stuck)).exists(), False)
    expect('blocked write: no temporary files', sorted(p.name for p in blocked.glob('*.tmp')), [])
    expect('blocked write: others written', summary(jam.stdout).get('written'), '7')

    none = run(tmp / 'none', '--roots', str(rootdir), archive_name='nopairs.bin')
    expect('no pairs: run fails', none.returncode != 0 and 'no program pair' in none.stderr, True)

    # 130 consecutive 512 KiB streams (65 MiB inflated, a few KB on disk) would join into one run;
    # the 64 MiB cap must split them into two.
    piece = zlib.compress(bytes(0x80000), 9)
    joined = bytearray()
    for _ in range(130):
        aligned(joined)
        joined += piece
    (tmp / 'joined.bin').write_bytes(joined)
    capped = run(tmp / 'capped', '--roots', str(rootdir), archive_name='joined.bin')
    expect('joined run capped at 64 MiB', summary(capped.stdout).get('streams'), '2')
    if failures:
        print(out.stdout, out.stderr, again.stdout, jam.stdout, jam.stderr, sep='\n')

for item in failures:
    print('FAIL', item)
if failures:
    sys.exit(1)
print('startup_archive_requests_check passed')
