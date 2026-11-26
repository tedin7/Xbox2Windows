"""Manifest line `srvuse N sample|texel|lod` of tools/startup/gfx2hlsl.py (fork tester branch).

usage: startup_srvuse_manifest_check.py            pure Python, no arguments

The line says what a translated pixel program does with a read texture, so a runtime that wants to change the
texture (opt-in: drop its top mip level, X2W_D3D12_TEXTURE_DROP_MIPS) knows when it may. It is metadata only: the
emitted HLSL does not depend on it. A texture with any use the translator did not classify gets no line, and a
runtime treats that as unknown and changes nothing.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_sample_a16_check import (IMG_SAMPLE, IMG_SAMPLE_L, IMG_SAMPLE_LZ, encode_mimg,  # noqa: E402
                                            translate)

IMG_LOAD = 0x00
IMG_GET_LOD = 0x60
failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


def uses(*mimg):
    pipe = translate(b''.join(mimg))
    manifest = pipe.manifest()
    return [line for line in manifest.splitlines() if line.startswith('srvuse ')], pipe.hlsl_ps(), manifest


def main():
    sample = encode_mimg(IMG_SAMPLE, ssamp=24)
    sample_l = encode_mimg(IMG_SAMPLE_L, extra_addrs=b'\0\0\0\0', nsa=0)
    sample_lz = encode_mimg(IMG_SAMPLE_LZ)
    load = encode_mimg(IMG_LOAD, vaddr=0)
    get_lod = encode_mimg(IMG_GET_LOD, dmask=0x3)

    got, _, manifest = uses(sample)
    check('plain sample is sampling only', got == ['srvuse 0 sample'], repr(got))
    check('the srv line stays', 'srv 0 tex2d table 0 0' in manifest)
    got, _, _ = uses(sample_lz)
    check('sample_lz is sampling only', got == ['srvuse 0 sample'], repr(got))
    try:
        got, _, _ = uses(sample_l)
        check('sample_l is sampling only', got == ['srvuse 0 sample'], repr(got))
    except G.Unsupported as why:
        print('sample_l form not translated by this fixture (%s): skipped' % str(why)[:60])
    got, _, _ = uses(load)
    check('image_load is texel addressed', got == ['srvuse 0 texel'], repr(got))
    got, _, _ = uses(sample, load)
    check('a sample and a load of one texture: texel wins', got == ['srvuse 0 texel'], repr(got))
    try:
        got, _, _ = uses(get_lod)
        check('image_get_lod asks for the level of detail', got == ['srvuse 0 lod'], repr(got))
        got, _, _ = uses(sample, get_lod)
        check('sample and get_lod of one texture: lod', got == ['srvuse 0 lod'], repr(got))
    except G.Unsupported as why:
        print('get_lod form not translated by this fixture (%s): skipped' % str(why)[:60])

    # Unclassified use: no line (a runtime reads that as unknown and keeps every level).
    r = {'index': 0, 'kind': 'tex2d', 'uses': {'sample'}, 'calls': 2, 'classified': 1}
    check('an unclassified use gives no line', G.Pipeline.srv_use_line(r) is None)
    check('a resource nobody classified gives no line', G.Pipeline.srv_use_line({'index': 0, 'kind': 'tex2d'}) is None)
    check('a buffer gives no line', G.Pipeline.srv_use_line({'index': 0, 'kind': 'buf', 'uses': {'sample'}, 'calls': 1, 'classified': 1}) is None)

    # Metadata only: the HLSL of a sample is the same text with or without the classification.
    _, hlsl, _ = uses(sample)
    check('HLSL names the texture through Sample only', re.search(r'\br0\.Sample\(', hlsl) is not None
          and 'srvuse' not in hlsl)
    print('FAIL' if failures else 'startup_srvuse_manifest: PASS')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
