"""image_sample_c_lz with sampler word 2 bits 28-30 set (MIP_POINT_PRECLAMP, ANISO_OVERRIDE, BLEND_ZERO_PRT).

The explicit level-zero form ignores them, so tools/startup/gfx2hlsl.py must lower it. Bit 31 and the implicit
image_sample_c form (which does select a mip) keep refusing them. The fixture is the 'lz' program of
startup_image_sample_c_check.py with the sampler word 2 literal patched in.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_image_sample_c_check as C  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, root_blob  # noqa: E402

S_MOV_S26_ZERO = bytes.fromhex('80039abe')  # s_mov_b32 s26, 0
S_MOV_S26_LITERAL = bytes.fromhex('ff039abe')  # s_mov_b32 s26, <literal>


def pipeline(base, w2):
    words = C.WORDS[base]
    assert words.count(S_MOV_S26_ZERO) == 1
    words = words.replace(S_MOV_S26_ZERO, S_MOV_S26_LITERAL + struct.pack('<I', w2))
    return G.Pipeline(VS_WORDS, words, root_blob(flags=0))


def main():
    for bit in (28, 29, 30, 28 | 29 | 30):
        w2 = (1 << bit) if bit < 31 else 7 << 28
        pipe = pipeline('lz', w2)
        C.check_call(pipe.hlsl_ps(), 'SampleCmpLevelZero', 'v1', 'v2', 'v0')
        assert C.manifest_samplers(pipe) == ['smpc 0 0x%x 0x0 0x%x 0x0' % (C.POINT | C.LESS, w2)], bit
    for w2 in (1 << 31, 1 << 31 | 1 << 28):
        try:
            pipeline('lz', w2).hlsl_ps()
        except G.Unsupported as error:
            assert 'word 2 bits 28-31' in str(error), error
        else:
            raise AssertionError('bit 31 accepted')
    try:
        pipeline('c_max0', 1 << 28).hlsl_ps()
    except G.Unsupported as error:
        assert 'word 2 bits 28-31' in str(error), error
    else:
        raise AssertionError('implicit image_sample_c accepted word 2 bit 28')
    print('PASS')


if __name__ == '__main__':
    main()
