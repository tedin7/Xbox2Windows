"""Partial colour exports in tools/startup/gfx2hlsl.py carry their channel mask (#1674).

usage: startup_gfx2hlsl_export_mask_check.py [--mutate drop-mask]

A colour export whose channels are 'off' leaves those channels of the render target unwritten. The
translated pixel stage returns a float4 per target, so the manifest records which channels the export
writes ('wmask <target> 0x<mask>') and the runtime keeps the others out of the write mask
(src/startup_stubs/export_write_mask.h).

Every export below is a synthetic instruction encoded from the EXP layout (DW0: EN [3:0], TARGET
[9:4], COMPR [10], DONE [11], VM [12], encoding 0b111110 in [31:26]; DW1: VSRC0..3, one byte each).
It goes through decode() and the whole translation with a recorded vertex stage, and the check reads the
lowered HLSL and the manifest. --mutate drop-mask removes the 'wmask' lines from the manifest; the
check must then fail.
"""
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools', 'startup'))
sys.path.insert(0, HERE)
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import VS_WORDS, root_blob  # noqa: E402

failures = []
S_ENDPGM = struct.pack('<I', 0xbf810000)

if sys.argv[1:] == ['--mutate', 'drop-mask']:
    # Negative control: the manifest loses its 'wmask' lines, so every mask check below must fail.
    _manifest = G.Pipeline.manifest
    G.Pipeline.manifest = lambda self: ''.join(line for line in _manifest(self).splitlines(True) if not line.startswith('wmask'))
elif sys.argv[1:]:
    sys.exit(__doc__)


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH  ' + detail}")
    if not condition:
        failures.append(label)


def exp(target, en, vsrc, *, compr=False, done=True, vm=True):
    """One EXP instruction. target: 0..7 for mrt0..7, 12..15 for pos0..3, 32+ for paramN."""
    w0 = 0xf8000000 | (en & 0xf) | (target & 0x3f) << 4 | int(compr) << 10 | int(done) << 11 | int(vm) << 12
    return struct.pack('<2I', w0, sum((v & 0xff) << (8 * k) for k, v in enumerate(vsrc)))


def disasm(words):
    return [asm for _, _, _, asm in G.decode(words)]


def translate(ps_words, rtv_formats=()):
    """(pixel HLSL, manifest) or the refusal message."""
    try:
        pipe = G.Pipeline(VS_WORDS, ps_words + S_ENDPGM, root_blob(), rtv_formats=rtv_formats)
        pipe.hlsl_vs()
        return pipe.hlsl_ps(), pipe.manifest()
    except G.Unsupported as reason:
        return str(reason)


def masks(manifest):
    return re.findall(r'^wmask (\d+) (0x[0-9a-f]+)$', manifest, re.M)


def assignment(ps, target):
    m = re.search(r'o\.mrt%d = (.*);' % target, ps)
    return m.group(1) if m else ''


# 1. The disassembler forms the translator receives.
check('disasm: low pair', disasm(exp(0, 0x3, (0, 1, 0, 0), compr=True)) == ['exp mrt0 v0, v0, off, off done compr vm'],
      repr(disasm(exp(0, 0x3, (0, 1, 0, 0), compr=True))))
check('disasm: high pair', disasm(exp(0, 0xc, (0, 1, 0, 0), compr=True)) == ['exp mrt0 off, off, v1, v1 done compr vm'],
      repr(disasm(exp(0, 0xc, (0, 1, 0, 0), compr=True))))
check('disasm: non-pair mask is flagged', disasm(exp(0, 0x1, (0, 1, 0, 0), compr=True))
      == ['exp mrt0 v0, v0, off, off done compr vm en:0x1'], repr(disasm(exp(0, 0x1, (0, 1, 0, 0), compr=True))))

# 2. Compressed exports of whole channel pairs lower, and the manifest carries the pair mask.
low = translate(exp(0, 0x3, (0, 1, 0, 0), compr=True))
check('compr low pair translates', isinstance(low, tuple), repr(low))
if isinstance(low, tuple):
    check('compr low pair: channels 0 and 1 from v0',
          assignment(low[0], 0) == 'float4(f16tof32(v0), f16tof32(v0 >> 16), 0.0, 0.0)', assignment(low[0], 0))
    check('compr low pair: wmask 0 0x3', masks(low[1]) == [('0', '0x3')], repr(masks(low[1])))

high = translate(exp(0, 0xc, (0, 1, 0, 0), compr=True))
check('compr high pair translates', isinstance(high, tuple), repr(high))
if isinstance(high, tuple):
    check('compr high pair: channels 2 and 3 from v1',
          assignment(high[0], 0) == 'float4(0.0, 0.0, f16tof32(v1), f16tof32(v1 >> 16))', assignment(high[0], 0))
    check('compr high pair: wmask 0 0xc', masks(high[1]) == [('0', '0xc')], repr(masks(high[1])))

full = translate(exp(0, 0xf, (0, 1, 0, 0), compr=True))
check('compr full export translates', isinstance(full, tuple), repr(full))
if isinstance(full, tuple):
    check('compr full export: all four channels',
          assignment(full[0], 0) == 'float4(f16tof32(v0), f16tof32(v0 >> 16), f16tof32(v1), f16tof32(v1 >> 16))',
          assignment(full[0], 0))
    check('compr full export: no wmask line', masks(full[1]) == [], repr(masks(full[1])))

# 3. Uncompressed partial exports record their mask too (they used to write 0 to the 'off' channels).
part = translate(exp(0, 0x5, (0, 1, 2, 3)))
check('uncompressed red and blue translates', isinstance(part, tuple), repr(part))
if isinstance(part, tuple):
    check('uncompressed red and blue: wmask 0 0x5', masks(part[1]) == [('0', '0x5')], repr(masks(part[1])))


def revision_key(revision):
    root = root_blob()
    ps_code = exp(0, 0x5, (0, 1, 2, 3)) + S_ENDPGM
    return G.fnv(root, G.fnv(ps_code, G.fnv(VS_WORDS, G.fnv(revision))))


runtime = os.path.join(HERE, '..', 'src', 'startup_stubs', 'd3d12_draw.inl')
with open(runtime, 'rb') as runtime_file:
    revision = re.search(rb'constexpr char revision\[\] = "([^"]+)";', runtime_file.read())
key = G.pipeline_key(VS_WORDS, exp(0, 0x5, (0, 1, 2, 3)) + S_ENDPGM, root_blob())
check('offline and runtime use the same partial export cache key', bool(revision and key == revision_key(revision[1])))
check('older zero-filled export cache is not reused', key != revision_key(b'x2w-startup-shader-6'))
whole = translate(exp(0, 0xf, (0, 1, 2, 3)))
check('uncompressed full export: no wmask line', isinstance(whole, tuple) and masks(whole[1]) == [], repr(whole))

# 4. Two targets: each keeps its own mask; a full target writes no line.
two = translate(exp(0, 0xf, (0, 1, 0, 0), compr=True, done=False) + exp(1, 0x3, (2, 3, 0, 0), compr=True))
check('two targets translate', isinstance(two, tuple), repr(two))
if isinstance(two, tuple):
    check('two targets: only mrt1 has a mask', masks(two[1]) == [('1', '0x3')], repr(masks(two[1])))

# 5. Refusals kept.
refusals = [
    ('compr export of no channel', exp(0, 0x0, (0, 1, 0, 0), compr=True), (), 'export to mrt0 writes no channel'),
    ('uncompressed export of no channel', exp(0, 0x0, (0, 1, 2, 3)), (), 'export to mrt0 writes no channel'),
    ('compr mask that is not a pair', exp(0, 0x1, (0, 1, 0, 0), compr=True), (),
     'compressed export enable mask 0x1 does not select whole channel pairs'),
    ('compr mask across pairs', exp(0, 0x6, (0, 1, 0, 0), compr=True), (),
     'compressed export enable mask 0x6 does not select whole channel pairs'),
    ('packed integer compr export', exp(0, 0x3, (0, 1, 0, 0), compr=True), (42,), 'packed integer export mrt0'),
    ('two masks on one target', exp(0, 0x3, (0, 1, 0, 0), compr=True, done=False) + exp(0, 0xf, (0, 1, 0, 0), compr=True),
     (), 'exports to mrt0 write channel masks 0x3 and 0xf'),
]
for label, words, formats, want in refusals:
    got = translate(words, formats)
    check('refused: ' + label, got == want, repr(got))

# 6. A partial compressed export to a vertex output has no write mask to carry it.
stage = G.Stage.__new__(G.Stage)
stage.kind, stage.exec_known, stage.lines, stage.exports, stage.write_masks = 'vs', True, [], {}, {}
stage.compr_params = set()
stage.used_s, stage.used_v = set(), set()
stage.pipe = None
try:
    stage.export('exp param0 v0, v0, off, off compr')
    got = 'accepted'
except G.Unsupported as reason:
    got = str(reason)
check('refused: partial compr export to param0', got == 'partial compressed export to param0', got)

# 7. Typed colour targets (#1999 F): each verified format gets a matching shader output.
typed = (('R32G32B32A32_UINT', 3, 'uint4 mrt0', 'uint4(v0'), ('R32G32B32A32_SINT', 4, 'int4 mrt0', 'asint(uint4'),
         ('R32G32_UINT', 17, 'uint4 mrt0', 'uint4(v0'), ('R32G32_SINT', 18, 'int4 mrt0', 'asint(uint4'),
         ('R8G8B8A8_UINT', 30, 'uint4 mrt0', 'uint4(v0'), ('R8G8B8A8_SINT', 32, 'int4 mrt0', 'asint(uint4'),
         ('R32_UINT', 42, 'uint4 mrt0', 'uint4(v0'), ('R32_SINT', 43, 'int4 mrt0', 'asint(uint4'),
         ('R32G32_FLOAT', 16, 'float4 mrt0', 'float4(F(('), ('R32_FLOAT', 41, 'float4 mrt0', 'float4(F(('))
for name, fmt, signature, store in typed:
    got = translate(exp(0, 0xf, (0, 1, 2, 3)), (fmt,))
    ok = isinstance(got, tuple) and signature in got[0] and store in assignment(got[0], 0).replace('uint4(U(v', 'uint4(v')
    check('typed target ' + name, ok, repr(got)[:300])
for name, fmt in (('R16G16B16A16_UINT', 12), ('R16G16B16A16_SINT', 14), ('R8_UINT', 62),
                  ('R16_SINT', 59), ('R8_SINT', 64)):
    got = translate(exp(0, 0xf, (0, 1, 2, 3)), (fmt,))
    want = 'unverified %s RTV format %d' % ('UINT' if fmt in G.UINT_RTVS else 'SINT', fmt)
    check('typed target %s refused' % name, got == want, repr(got)[:200])
for fmt in (30, 32):
    got = translate(exp(0, 0xf, (0, 1, 2, 3), compr=True), (fmt,))
    check('packed integer export refused for format %d' % fmt, got == 'packed integer export mrt0', repr(got)[:200])

print('export mask checks: %d failed' % len(failures))
sys.exit(1 if failures else 0)
