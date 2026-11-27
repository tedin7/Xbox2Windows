"""Check explicit PS input masks with a fullscreen triangle and a per-pixel integer oracle."""
import hashlib
import io
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, G, compile_hlsl, root_blob


def pixel_shader(reg, masks):
    # exp mrt0 vN, vN, vN, vN done vm; s_endpgm
    words = struct.pack('<III', 0xf800180f, reg * 0x01010101, 0xbf810000)
    return G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=masks).hlsl_ps()


# Vertex output control fields from the register layout.
MISC = 1 << 21
LAYER = MISC | (1 << 18)
VIEWPORT = MISC | (1 << 19)
CLIP = MISC | (1 << 22) | 1
CLIP_BOTH = MISC | (3 << 22) | 0x1f
CULL = MISC | (3 << 22) | (1 << 12)


def vertex_position_words():
    """Two triangles, in order BL BR TL, TL BR TR. Feed misc and both distance vectors."""
    from startup_stream_draw_check import vop1, vop2

    words = []
    for reg, corners in ((1, 0b110010), (2, 0b101100)):
        words += [vop1(1, reg, 128 + corners), vop2(0x16, reg, 256, reg),
                  vop2(0x1b, reg, 129, reg), vop1(6, reg, 256 + reg),
                  vop2(8, reg, 244, reg), vop2(3, reg, 243, reg)]
    # z=0, w=1. The low layer bit comes from the vertex-ID entry value:
    # ((vid >> 3) XOR 1) = 1 for these six vertices. Viewport is also 1.
    words += [vop1(1, 3, 128), vop1(1, 4, 242), vop2(0x16, 5, 131, 0),
              vop2(0x1d, 5, 129, 5), vop2(0x1c, 5, 255, 5), 1 << 16,
              vop1(1, 6, 243)]
    for slot, mask, registers in (
        (0, 15, (1, 2, 3, 4)), (1, 5, (4, 0, 5, 0)),
        (2, 15, (1, 4, 4, 4)), (3, 1, (6, 0, 0, 0)),
    ):
        words += [0xf8000000 | ((12 + slot) << 4) | mask | (0x800 if slot == 3 else 0),
                  sum(reg << (8 * c) for c, reg in enumerate(registers))]
    return words + [0xbf810000]


def vertex_position_controls():
    from startup_stream_check import container
    from startup_stream_draw_check import pixel
    from pixel_aot import container_code
    from resource_inventory import parts
    from shader_cache_rekey import rekey

    words = vertex_position_words()
    vs = struct.pack('<%dI' % len(words), *words)
    ps = container_code(parts(pixel(False))[b'DXIL'], stage=0)
    try:
        G.Pipeline(vs, ps, root_blob())
    except G.Unsupported as err:
        assert str(err) == 'export target pos1: vertex output control not in the request', err
    else:
        raise AssertionError('position exports accepted without vertex output control')
    baseline = G.Pipeline(VS_WORDS, ps, root_blob())
    baseline_hlsl = (baseline.hlsl_vs(), baseline.hlsl_ps())
    assert hashlib.sha256(''.join(baseline_hlsl).encode()).hexdigest() == '76c3f85f2ffb9908a00b39414661c3c903bc102ea6cee231bc962081edacd25e'
    same = G.Pipeline(VS_WORDS, ps, root_blob(), vs_out_cntl=CLIP)
    assert (same.hlsl_vs(), same.hlsl_ps()) == baseline_hlsl
    for word in (0, CLIP, CLIP_BOTH, CULL, LAYER, VIEWPORT, MISC | (1 << 16)):
        pipe = G.Pipeline(vs, ps, root_blob(), vs_out_cntl=word)
        assert 'vsposx 1 2 3' in pipe.manifest().splitlines()
        assert pipe.hlsl_ps() == baseline_hlsl[1]
        assert ('Point size is ignored' in pipe.hlsl_vs()) == bool(word & (1 << 16))
    with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ', {'X2W_FXC': ''}):
        cache = Path(temp)
        shader, pixel_blob, root = container(1, words), pixel(False), root_blob()
        key = G.pipeline_key(shader, pixel_blob, root, vs_out_cntl=CLIP)
        assert key != G.pipeline_key(shader, pixel_blob, root, vs_out_cntl=0)
        for ext, blob in (('a', shader), ('b', pixel_blob), ('root', root), ('vsout', struct.pack('<I', CLIP))):
            (cache / f'req_{key:016x}.{ext}').write_bytes(blob)
        rtv = f'rtv1 {key} 28'
        variant = G.fnv(rtv.encode())
        (cache / f'req_{variant:016x}.rtv').write_text(rtv, encoding='ascii')
        for request in (key, variant):
            assert G.translate_request(cache, request) == 'ok'
            assert 'vsposx 1 2 3' in (cache / f'{request:016x}.txt').read_text().splitlines()
        assert G.pending_keys(cache, False) == []

        sidecar = cache / f'req_{key:016x}.vsout'
        rekeyed = cache / 'rekeyed'
        assert rekey(cache, rekeyed, jobs=1, out=io.StringIO()) == 0
        assert (rekeyed / sidecar.name).read_bytes() == sidecar.read_bytes()
        assert G.translate_request(rekeyed, variant) == 'ok'
        sidecar.write_bytes(struct.pack('<I', CULL))
        for request in (key, variant):
            assert G.translate_request(cache, request) == 'unsupported: vertex output control names another pipeline'
            assert not (cache / f'{request:016x}.txt').exists()
        sidecar.write_bytes(struct.pack('<I', CLIP))
        for request in (key, variant):
            assert G.translate_request(cache, request) == 'ok'
        sidecar.unlink()
        assert set(G.pending_keys(cache, True)) == {key, variant}
        for request in (key, variant):
            assert G.translate_request(cache, request) == (
                'unsupported: vertex output control description missing for a request translated with one')
            assert not (cache / f'{request:016x}.txt').exists()
    print('Vertex position exports: metadata refusal and unchanged VS/PS HLSL controls passed')


def main(dxc, renderer, adapter, fxc=None):
    vertex_position_controls()
    try:
        pixel_shader(1, (0x8100, 0x8000))
    except G.Unsupported as err:
        assert 'enable/address' in str(err)
    else:
        raise AssertionError('contradictory metadata accepted')
    compilers = [('dxc', dxc, '6_0')]
    if fxc:
        compilers.append(('fxc', fxc, '5_1'))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for kind, compiler, profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs', 'vs_' + profile, folder)
            for label, reg, masks, good in (
                ('one earlier word', 1, (0x8100, 0x8100), True),
                ('one earlier pair', 2, (0x8002, 0x8002), True),
                ('missing metadata', 1, None, False),
                ('wrong register', 1, (0x8002, 0x8002), False),
                ('packed position disabled', 1, (0x0100, 0x0100), False),
            ):
                ps = compile_hlsl(compiler, kind, pixel_shader(reg, masks), 'ps', 'ps_' + profile, folder)
                run = subprocess.run([renderer, adapter, '--startup-packed-position', vs, ps],
                                     capture_output=True, text=True)
                if run.returncode == 77:
                    print(run.stdout + run.stderr)
                    return 77
                if good:
                    assert run.returncode == 0 and 'all 91 packed positions matched' in run.stdout, (run.stdout, run.stderr)
                else:
                    assert run.returncode != 0 and 'position mismatch (1,0): expected 00000001 actual 00000000' in run.stderr, run.stderr
                print(f'{kind} {adapter} {label}: ' + ('91 pixels matched' if good else 'expected mismatch at (1,0)'))
    return 0


if __name__ == '__main__':
    if sys.argv[1:] == ["controls"]:
        vertex_position_controls()
    else:
        sys.exit(main(*sys.argv[1:]))
