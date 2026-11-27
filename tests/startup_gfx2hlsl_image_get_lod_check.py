"""image_get_lod on a 2D texture and a 2D array in tools/startup/gfx2hlsl.py (M200-01 L).

image_get_lod returns [0] the LOD a sample would use, clamped, and [1] the computed LOD before the clamp (Mesa
nir_texop_lod, emitted by ACO as image_get_lod; GLSL textureQueryLod x/y). It is lowered to HLSL
CalculateLevelOfDetail / CalculateLevelOfDetailUnclamped, which take implicit derivatives like Sample, so the stage
must be a pixel stage whose EXEC is still the entry mask (every lane active) with wave-uniform control flow. A
uniform (scc) join or later loop does not drop that proof; divergent EXEC or a vcc/exec-conditioned branch before
the read does. The clamped value depends on the sampler's LOD limits and bias, so the sampler must be a constant
descriptor under the validation of image_sample_b.

The 2D-array form (dim:SQ_RSRC_IMG_2D_ARRAY) takes a third address dword, the slice, which the decoder reads
back. DXIL CalculateLOD uses two active coordinates (c0, c1) for Texture2DArray -- the slice is not part of the
LOD -- and HLSL's Texture2DArray CalculateLevelOfDetail takes float2, so the slice register is consumed but
dropped from the call: Texture2DArray r0 queried at float2(u, v), exactly the two-component coordinate of
glslang's hlsl.calculatelod.dx10.frag ("LOD queries do not pass array level in coordinate").

Fixture (synthetic): the recorded c_lz stage of startup_image_sample_c_check with the MIMG opcode set to 0x60
(image_get_lod; the decoder reads it back with a 2-register, or 3-register for the array, address), sampler
word 2 loaded from a literal (mip filter point), and for the second form dmask 0x3. The array fixture keeps the
original v2 = 1.0 as its slice dword; a variant swaps the slice register pair so the consumed dword is proven
to come from the address position, not the register number.
Checks: dmask 0x1, 0x2 and 0x3 lower to the two calls on the sampler, which carries its LOD limits, on a 2D
texture and a 2D array (Texture2DArray r0, srv tex2da, slice dropped). A uniform join before the read is
accepted on both. Controls (refused): dmask 0x4, a 3D image, s_cbranch_vccz/execz before the read (2D and
array), a sampler LOD bias, a sampler with mipmapping disabled, an array decoded with two address registers,
and the stage without op_image_get_lod (negative control: refused as on main).
  draw <dxc> [fxc]: also compile the dmask 0x3 2D and 2D-array stages with dxc ps_6_0 (and fxc ps_5_1).
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_image_sample_c_check as C  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, root_blob  # noqa: E402

W2_AT = 0x14  # s_mov_b32 s26, 0 (sampler word 2) in the c_lz fixture
AT = 0x34 + 4  # the MIMG instruction once word 2 takes a literal
EXPORT_AT = 0x40 + 4
MIP_POINT = 1 << 26  # SQ_IMG_SAMP_WORD2 bits 27:26 MIP_FILTER


def sopp(op, simm):
    return struct.pack('<I', (simm & 0xffff) | op << 16 | 0x17f << 23)


def stage(dmask=1, dim=1, branch=False, divergent=False, join=False, cf_op=None, word2=MIP_POINT, back_op=None,
          nsa=None):
    w = bytearray(C.WORDS['lz'])
    assert w[W2_AT:W2_AT + 4] == bytes.fromhex('80039abe'), w[W2_AT:W2_AT + 4].hex()
    w[W2_AT:W2_AT + 4] = struct.pack('<II', 0xbe9a03ff, word2)  # s_mov_b32 s26, literal
    w0, w1 = struct.unpack_from('<II', w, AT)
    w0 = w0 & ~(0x7f << 18) & ~(0xf << 8) & ~(7 << 3) | 0x60 << 18 | dmask << 8 | dim << 3
    if dmask == 3:  # export v4, v5
        struct.pack_into('<II', w, EXPORT_AT, 0xf800180f, 4 | 5 << 8 | 4 << 16 | 5 << 24)
    if nsa is not None:  # nsize 1: one NSA dword whose first bytes give the address registers after vaddr0
        w0 = w0 & ~0x6 | 1 << 1
        struct.pack_into('<II', w, AT, w0, w1)
        return bytes(w[:AT + 8]) + bytes(nsa) + bytes(w[AT + 8:])
    struct.pack_into('<II', w, AT, w0, w1)
    code = bytes(w)
    if back_op is not None:  # s_cbranch_<op> right after the read, back to it (a divergent back edge spanning the read)
        code = code[:AT + 8] + sopp(back_op, -3) + code[AT + 8:]
    insert = b''
    if branch:  # s_cbranch_scc0 to the next instruction: uniform control flow before the read
        insert += sopp(4, 0)
    if join:  # uniform forward join; every lane still has the entry EXEC at the read
        insert += (bytes.fromhex('800380be') + bytes.fromhex('008006bf') + sopp(5, 1) + bytes.fromhex('810381be'))
    if divergent:  # s_mov_b64 exec, 0: the entry-mask proof is gone
        insert += bytes.fromhex('8004febe')
    if cf_op is not None:  # s_mov; s_cbranch_<op> +1; s_mov (vcc/exec: divergent CF)
        insert += bytes.fromhex('800380be') + sopp(cf_op, 1) + bytes.fromhex('810381be')
    if insert:
        code = code[:AT] + insert + code[AT:]
    return code


def ps(code):
    pipe = G.Pipeline(VS_WORDS, code, root_blob(flags=0))
    return pipe, pipe.hlsl_ps()


def refused(code, text):
    try:
        ps(code)
    except G.Unsupported as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('accepted, expected %r' % text)


def controls():
    listing = [a for _, _, _, a in G.decode(stage())]
    assert 'image_get_lod v4, v[0:1], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D' in listing, listing
    pipe, text = ps(stage())
    call = 't = float4(r0.CalculateLevelOfDetail(q0, float2(F(v0), F(v1))), ' \
           'r0.CalculateLevelOfDetailUnclamped(q0, float2(F(v0), F(v1))), 0.0, 0.0);'
    assert call in text and 'v4 = U(t.x);' in text, text
    _, text = ps(stage(dmask=3))
    assert call in text and 'v4 = U(t.x);' in text and 'v5 = U(t.y);' in text, text
    _, text = ps(stage(dmask=2))  # the unclamped LOD alone lands in the first destination register
    assert call in text and 'v4 = U(t.y);' in text and 'U(t.x)' not in text, text
    _, text = ps(stage(branch=True))
    assert call in text and 'v4 = U(t.x);' in text, text
    _, text = ps(stage(join=True))
    assert call in text and 'v4 = U(t.x);' in text, text
    refused(stage(dmask=4), 'image_get_lod form')
    refused(stage(dim=2), 'image_get_lod form')  # SQ_RSRC_IMG_3D
    # The 2D-array form: Texture2DArray, the slice consumed and dropped from the float2 coordinate.
    array_call = ('t = float4(r0.CalculateLevelOfDetail(q0, float2(F(v0), F(v1))), '
                  'r0.CalculateLevelOfDetailUnclamped(q0, float2(F(v0), F(v1))), 0.0, 0.0);')
    listing = [a for _, _, _, a in G.decode(stage(dim=5))]
    assert 'image_get_lod v4, v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY' in listing, listing
    array_pipe, text = ps(stage(dim=5))
    assert array_call in text and 'v4 = U(t.x);' in text, text
    assert 'Texture2DArray<float4> r0 : register(t0);' in text and 'Texture2D<' not in text, text
    assert 'srv 0 tex2da table 0 0' in array_pipe.manifest(), array_pipe.manifest()
    assert 'smpl 0 0x8001012 0x0 0x4000000 0x0' in array_pipe.manifest(), array_pipe.manifest()
    _, text = ps(stage(dmask=3, dim=5))
    assert array_call in text and 'v4 = U(t.x);' in text and 'v5 = U(t.y);' in text, text
    _, text = ps(stage(dmask=2, dim=5))
    assert array_call in text and 'v4 = U(t.y);' in text and 'U(t.x)' not in text, text
    code = stage(dmask=3, dim=5, nsa=[1, 3, 0, 0])
    listing = [a for _, _, _, a in G.decode(code)]
    assert 'image_get_lod v[4:5], [v0, v1, v3], s[16:23], s[24:27] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY' in listing, listing
    _, text = ps(code)
    assert array_call in text and 'F(v2)' not in text and 'F(v3)' not in text, text
    _, text = ps(stage(branch=True, dim=5))
    assert array_call in text and 'v4 = U(t.x);' in text, text
    _, text = ps(stage(join=True, dim=5))
    assert array_call in text and 'v4 = U(t.x);' in text, text
    for op in (7, 9):  # vccnz, execnz back edge after the array read
        refused(stage(dim=5, back_op=op), 'image_get_lod needs implicit derivatives')
    for op in (6, 7, 8, 9):  # vccz, vccnz, execz, execnz
        refused(stage(dim=5, cf_op=op), 'image_get_lod needs implicit derivatives')
    refused(stage(dim=5, word2=MIP_POINT | 1), 'image_get_lod with a sampler LOD bias')
    refused(stage(dim=5, word2=0), 'image_get_lod with mipmapping disabled')
    # An array decoded with two address registers is refused (the slice dword would be unread).
    original = G.D.MIMG_DIMS[5]
    G.D.MIMG_DIMS[5] = (original[0], 2, original[2])
    try:
        refused(stage(dim=5), 'image_get_lod form')
    finally:
        G.D.MIMG_DIMS[5] = original
    _, text = ps(stage(divergent=True))  # a cleared EXEC is an exact per-lane write: the quad program keeps the proof
    assert call in text, text
    for op in (7, 9):  # vccnz, execnz back edge after the read
        refused(stage(back_op=op), 'image_get_lod needs implicit derivatives')
    for op in (6, 7, 8, 9):  # vccz, vccnz, execz, execnz
        refused(stage(cf_op=op), 'image_get_lod needs implicit derivatives')
    refused(stage(word2=MIP_POINT | 1), 'image_get_lod with a sampler LOD bias')
    refused(stage(word2=0), 'image_get_lod with mipmapping disabled')
    saved = G.Stage.op_image_get_lod
    del G.Stage.op_image_get_lod
    try:
        refused(stage(), 'image_get_lod')
    finally:
        G.Stage.op_image_get_lod = saved
    print('image_get_lod: CalculateLevelOfDetail / Unclamped for dmask 0x1, 0x2 and 0x3 on a 2D texture and a 2D '
          'array (Texture2DArray, slice consumed and dropped from the float2 coordinate, manifest srv tex2da, NSA '
          'address); uniform join/branch before the read accepted on both, also after a cleared EXEC; dmask 0x4, a '
          '3D image, vcc/exec branches, a sampler bias and a sampler without mipmapping refused on both, an array '
          'with two address registers refused; refused without the handler')
    return pipe


def compile_check(dxc, fxc=None):
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in (('2D', ps(stage(dmask=3))[1]), ('2D array', ps(stage(dmask=3, dim=5))[1])):
            slug = name.replace(' ', '_')
            path = Path(tmp) / ('ps_%s.hlsl' % slug)
            path.write_text(text)
            runs = [('dxc', [dxc, '-T', 'ps_6_0', '-E', 'main', '-Fo', str(Path(tmp) / ('%s.cso' % slug)),
                             str(path)])]
            if fxc:
                runs.append(('fxc', [fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo',
                                     str(Path(tmp) / ('%s_fxc.cso' % slug)), str(path)]))
            for kind, args in runs:
                r = subprocess.run(args, capture_output=True, text=True, timeout=60)
                if r.returncode:
                    raise AssertionError('%s (%s): %s' % (kind, name, r.stderr[-1500:]))
                print('%s compiles the dmask 0x3 %s stage' % (kind, name))


def main():
    controls()
    if len(sys.argv) > 1:
        if sys.argv[1] != 'draw':
            raise SystemExit(__doc__)
        compile_check(*sys.argv[2:])


if __name__ == '__main__':
    main()
