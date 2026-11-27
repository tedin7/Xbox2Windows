"""`v_fract_f32` lowers through the clamped helper `fract32` in tools/startup/gfx2hlsl.py (#1741).

usage: startup_gfx2hlsl_fract_check.py

Two NVIDIA reports fail `startup_gfx2hlsl_float_round_hardware` on the `v_fract_f32` input `0xb0800000`
(-2^-30): expected `0x3f7fffff`, actual `0x3f800000` (RTX 3070 Ti, driver 32.0.16.2101; Quadro M1000M
Maxwell, driver 32.0.15.7371). The same draw passes on WARP, on Intel HD Graphics 530 and on the RDNA 4
card used here. The table used to emit a bare D3D `frac`, which has no upper cap and differs between
adapters, while `tools/pixel_aot.py` has always capped `x - floor(x)` at `0x3f7fffff` through `fract32`.

The cap on a float comparison, `f >= 1.0f ? ... : f`, did not survive FXC: it treats `frac` as strictly
below 1, folds that comparison to false and compiles the helper to a bare `frc`, so the cap disappeared on
exactly the adapters the fix was for. DXC keeps the compare and the select, which is why the pixel_aot
lowering, compiled by DXC only, never showed this. The cap is now a comparison on the BITS,
`asuint(f) == 0x3f800000u`, which FXC emits as `frc`, `ieq`, `movc`. `frac` can only reach the cap by
rounding to exactly +1.0, so the equality covers the case a `>=` comparison covered, and a NaN word misses
it and passes through.

This check reads the emitted text only. It cannot reproduce the NVIDIA result: that needs one of those
adapters, and no such GPU is available here. What it pins is the lowering the adapters disagreed on, and
the shader cache revision that keeps the pre-fix entries out of the cache.
"""
import os
import re
import struct
import sys
from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))
root = Path(here).resolve().parent
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
sys.path.insert(0, here)
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Probe  # noqa: E402

BITS_FORM = 'asuint(f) == 0x3f800000u ? asfloat(0x3f7fffffu) : f'
FLOAT_FORM = 'f >= 1.0f ? asfloat(0x3f7fffffu) : f'
ONE = 0x3f800000
CAP = 0x3f7fffff
failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


def lower(line):
    """The translated statement of one assembly line, as the caller sees it."""
    stage = Probe()
    stage.one(line.split(' ', 1)[0], line)
    return ' '.join(stage.lines)


HELPER = re.compile(r'float fract32\(float value\) \{.*?\}')
BARE_FRAC = re.compile(r'(?<![\w.])frac\(')


def common():
    pipe = object.__new__(G.Pipeline)
    pipe.uds, pipe.resources, pipe.samplers, pipe.uses_sampler_heap = [], {}, {}, False
    return ' '.join(pipe.common())


def problems(statement, helper):
    """Everything wrong with one lowered statement and the helper it calls; empty when both are right."""
    found = []
    if 'fract32(' not in statement:
        found.append('the instruction does not call fract32')
    if BARE_FRAC.search(statement):
        found.append('the instruction calls a bare frac(')
    if '0x3f7fffff' not in helper:
        found.append('the helper does not clamp at 0x3f7fffff')
    if FLOAT_FORM in helper:
        found.append('the helper clamps on the float comparison `%s`, which FXC folds to false and emits a bare '
                     'frc for: the cap is lost' % FLOAT_FORM)
    elif BITS_FORM not in helper:
        found.append('the clamp is not the bits equality `%s`' % BITS_FORM)
    if 'min(' in helper:
        found.append('the clamp goes through min, which loses a NaN')
    return found


def capped(word, test):
    """The word the helper returns for one input word, under one comparison of the bits."""
    fire = {'==': word == ONE, '>': word > ONE, '>=': word >= ONE}[test]
    return CAP if fire else word


preamble = common()
match = HELPER.search(preamble)
check('the fract32 helper is emitted in the common preamble', match is not None, preamble[:240])
helper = match.group(0) if match else ''
check('the helper is defined once', preamble.count('float fract32(') == 1, preamble[:240])
source = Path(G.__file__).read_text(encoding='utf-8')
check('the helper carries the pixel_aot comment',
      '# Match pixel_aot: x - floor(x) capped at 0x3f7fffff' in source)
check('the helper comment names FXC folding the float comparison',
      'FXC folds `frac(value) >= 1.0f` to false' in source)

# 1. The table entry itself: one source, through the helper.
check('the VEC_F entry goes through fract32', G.VEC_F.get('v_fract_f32') == (1, 'fract32({0})'),
      repr(G.VEC_F.get('v_fract_f32')))
users = sorted(op for op, (_arity, text) in G.VEC_F.items() if 'fract32(' in text)
check('fract32 is used by v_fract_f32 alone and no other entry', users == ['v_fract_f32'], repr(users))

# 2. The emitted statement of each spelling, with the source modifiers the form accepts.
for line, expected in (
        ('v_fract_f32 v12, v0', 'v12 = U(fract32(F(v0)));'),
        ('v_fract_f32 v0, v0', 'v0 = U(fract32(F(v0)));'),
        ('v_fract_f32_e64 v13, |v1|', 'v13 = U(fract32(abs(F(v1))));'),
        ('v_fract_f32_e64 v14, -|v2|', 'v14 = U(fract32((-abs(F(v2)))));'),
):
    statement = lower(line)
    found = problems(statement, helper)
    check(f'{line} lowers through the clamp', not found, '%r: %s' % (statement, '; '.join(found)))
    check(f'{line} emits the expected statement', statement == expected, statement)

# 3. Negative controls: the checks above must reject the lowerings that caused the report and the ones that
#    would let it come back. The same problem list is used, so the controls cannot drift from it.
uncapped = problems(lower('v_fract_f32 v12, v0').replace('fract32(', 'frac('), helper)
check('control: a bare frac is rejected', any('bare frac' in text for text in uncapped), repr(uncapped))
capped_with_min = problems(lower('v_fract_f32 v12, v0'), helper.replace(BITS_FORM,
                                                                        'min(frac(value), asfloat(0x3f7fffffu))'))
check('control: a min-form clamp is rejected', bool(capped_with_min), repr(capped_with_min))
raised_cap = problems(lower('v_fract_f32 v12, v0'), helper.replace('0x3f7fffffu', '0x3f800000u'))
check('control: a cap other than 0x3f7fffff is rejected', bool(raised_cap), repr(raised_cap))
float_compare = problems(lower('v_fract_f32 v12, v0'), helper.replace(BITS_FORM, FLOAT_FORM))
check('control: the float comparison FXC folds away is rejected', any('FXC folds' in text for text in float_compare),
      repr(float_compare))
strict = problems(lower('v_fract_f32 v12, v0'), helper.replace(BITS_FORM, BITS_FORM.replace('==', '>')))
check('control: a strict > on the bits is rejected', bool(strict), repr(strict))

# What a strict > would do, evaluated on the words the helper can see. It misses the one case that occurs,
# because 1.0 is the value the cap is for, and it caps any NaN whose sign-and-exponent bits sort above 1.0.
check('the strict > misses 0x3f800000 and leaves 1.0, the word the report measured',
      capped(ONE, '>') == ONE and capped(ONE, '==') == CAP,
      '> reads %08x, == reads %08x' % (capped(ONE, '>'), capped(ONE, '==')))
check('the strict > turns a NaN word above 1.0 into the cap, and the == form passes it',
      capped(0x7fc00000, '>') == CAP and capped(0x7fc00000, '==') == 0x7fc00000,
      '> reads %08x, == reads %08x' % (capped(0x7fc00000, '>'), capped(0x7fc00000, '==')))
check('both comparisons leave a word below 1.0 alone', all(capped(w, t) == w for w in (0, CAP, 0x3f000000)
                                                          for t in ('==', '>')))
def float_capped(word):
    """The word the float comparison `frac(value) >= 1.0f` caps, read as a float: the rule the bits form keeps."""
    value = struct.unpack('<f', struct.pack('<I', word))[0]
    return CAP if value >= 1.0 else word


check('the bits equality caps the same words the float comparison does over the range frac returns',
      all(capped(w, '==') == float_capped(w) for w in (0, 0x80000000, 0x3f000000, CAP, ONE, 0x7fc00000)),
      repr([(hex(w), hex(capped(w, '==')), hex(float_capped(w)))
            for w in (0, CAP, ONE, 0x7fc00000)]))
# Outside that range the bits form is narrower: it leaves 2.0 alone where the float form caps it. frac cannot
# return such a word, so this is the one place the two rules differ, and the narrower one hides nothing.
check('the bits form is narrower than the float form above 1.0, which frac cannot return',
      capped(0x40000000, '==') == 0x40000000 and float_capped(0x40000000) == CAP,
      'bits %08x, float %08x' % (capped(0x40000000, '=='), float_capped(0x40000000)))

# 4. The shader cache revision: both literals are the same word, and the old seed cannot produce the new key.
gfx_source = Path(G.__file__).read_text(encoding='utf-8')
seed = re.search(r"(?:(?:fnv|shader_cache_seed)\(b'|PIPELINE_KEY_REVISION = b')(x2w-startup-shader-\d+)'", gfx_source)
stub_source = (root / 'src' / 'startup_stubs' / 'd3d12_draw.inl').read_text(encoding='utf-8')
literal = re.search(r'constexpr char revision\[\] = "(x2w-startup-shader-\d+)";', stub_source)
check('both files spell a x2w-startup-shader revision', seed is not None and literal is not None,
      repr((seed and seed.group(1), literal and literal.group(1))))
check('the two literals are identical', seed and literal and seed.group(1) == literal.group(1),
      repr((seed and seed.group(1), literal and literal.group(1))))
check('the revision is x2w-startup-shader-28, after the precise v_fma_mix lowering',
      bool(seed and seed.group(1) == 'x2w-startup-shader-28'), repr(seed and seed.group(1)))


def key_from_seed(word):
    """pipeline_key's hash chain with one seed, over the same three blobs, without the layout."""
    return G.fnv(b'root', G.fnv(b'ps', G.fnv(b'vs', G.shader_cache_seed(word.encode()))))


if seed:
    check('pipeline_key hashes the seed the source spells',
          G.pipeline_key(b'vs', b'ps', b'root') == key_from_seed(seed.group(1)),
          '%016x vs %016x' % (G.pipeline_key(b'vs', b'ps', b'root'), key_from_seed(seed.group(1))))
    old, new = key_from_seed('x2w-startup-shader-16'), key_from_seed(seed.group(1))
    check('the -16 seed produces a different key, so the pre-fix entries are out of the cache', old != new,
          '%016x vs %016x' % (old, new))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
