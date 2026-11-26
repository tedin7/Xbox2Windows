"""Synthetic pixel descriptor waterfalls and a D3D12 quad readback with two negative controls."""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_image_3d_check import FULLSCREEN_VS, G, VS_WORDS, compile_hlsl


# Extract integer x/y from the packed position, with normalized UV = (x,y)/2.
PREFIX = bytes.fromhex(
    '800389be 800398be ff0399be00002000 ff039abe00000004 80039bbe '  # table high word and point sampler
    '9000022c ff000036ffff0000 000d047e 010d067e f0040410 f0060610'
)


def waterfall(result, key_input, *, wrexec=False):
    return bytes.fromhex(
        f'81{key_input:02x}1436 '  # v_and_b32 v10, 1, v[key_input]
        '7e049ebe '               # s_mov_b64 s[30:31], exec
        '0a05287e 1414047d '      # v_readfirstlane_b32 s20, v10; v_cmp_eq_i32 vcc, s20, v10
        '6a24a0be 1485148f '      # s_and_saveexec_b64 s[32:33], vcc; s_lshl_b32 s20, s20, 5
        '040a0cf400000028 '      # s_load_dwordx8 s[40:47], s[8:9], s20
        f'7fc08cbf 080180f002{result:02x}ca00 703f8cbf '  # wait; image_sample v[result], v[2:3]; wait
    ) + bytes.fromhex(
        '203aa0be f4ff85bf' if wrexec else '7e20fe89 f4ff89bf'
        # WREXEC + SCC1 or XOR + EXECNZ; either branch returns to readfirstlane.
    ) + bytes.fromhex('1e04febe')  # s_mov_b64 exec, s[30:31]


LOOP0 = waterfall(4, 0)
LOOP1 = waterfall(5, 10, wrexec=True)
INVERT_KEY = bytes.fromhex('8114143a')  # v_xor_b32 v10, 1, v10
END = bytes.fromhex('0f1800f804050405 000081bf')  # exp mrt0 v4, v5, v4, v5 done vm; s_endpgm
PROGRAM = PREFIX + LOOP0 + INVERT_KEY + LOOP1 + END
INDEX = 'NonUniformResourceIndex(v10)'
UV = 'float2(F(v2), F(v3))'
SAMPLE = f'.SampleGrad(q0, {UV}, ddx({UV}), ddy({UV}))'
OLD_REFUSAL = 'v_readfirstlane_b32 requires a proven uniform source'


def root_blob(count=2):
    # Root signature 1.0, one pixel-visible descriptor table, a single SRV range with append at slot 0.
    rts = struct.pack('<16I', 1, 1, 24, 0, 0, 0, 0, 5, 36, 1, 44, 0, count, 0, 0, 0xffffffff)
    part = b'RTS0' + struct.pack('<I', len(rts)) + rts
    return b'DXBC' + bytes(16) + struct.pack('<4I', 1, 36 + len(part), 1, 36) + part


def translate(code=PROGRAM, count=2):
    return G.Pipeline(VS_WORDS, code, root_blob(count), ps_hw_inputs=(0x8000, 0x8000))


def must_refuse(expected, translate_case):
    try:
        translate_case()
    except G.Unsupported as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError(f'Expected refusal {expected!r}')


def controls():
    pipeline = translate()
    text = pipeline.hlsl_ps()
    assert 'Texture2D<float4> wf0[2] : register(t0);' in text
    assert text.count(INDEX) == text.count(SAMPLE) == 2
    assert 'switch (' not in text and 'if (' not in text.split('PSOut main', 1)[1]
    assert [line for line in pipeline.manifest().splitlines() if line.startswith('srv ')] == [
        'srv 0 tex2d table 0 0', 'srv 1 tex2d table 0 32',
    ]
    must_refuse(OLD_REFUSAL, lambda: G.Stage(pipeline, 'ps', PROGRAM, lower_waterfalls=False))
    must_refuse('pixel waterfall requires a bounded SRV descriptor table', lambda: translate(count=1))
    # A loop that retains selected lanes instead of removing them must stay refused.
    wrong_close = LOOP0.replace(bytes.fromhex('7e20fe89'), bytes.fromhex('7e20fe87'))
    assert wrong_close != LOOP0
    must_refuse(OLD_REFUSAL, lambda: translate(PREFIX + wrong_close + INVERT_KEY + LOOP1 + END))
    # The selected scalar key depends on iteration order after the restore.
    key_escape = bytes.fromhex('14020c7e')  # v_mov_b32 v6, s20
    must_refuse(OLD_REFUSAL, lambda: translate(PREFIX + LOOP0 + INVERT_KEY + LOOP1 + key_escape + END))
    print('Pixel waterfall: two samples, bounded bindings, legacy refusal and malformed loop controls passed')
    return pipeline


def reference(*, fixed_index=False, mip=1):
    expected = []
    for y in range(2):
        for x in range(2):
            # UV = (x,y)/2 and point filtering. At mip 1 the texture is 2x2.
            tx, ty = x << (1 - mip), y << (1 - mip)
            red = (1 + tx + 4 * ty + 32 * mip) / 128
            alpha = float(2 * mip)  # red through the second SRV component mapping
            first, second = (red, red) if fixed_index else ((alpha, red) if x else (red, alpha))
            expected.append(struct.pack('>4f', first, second, first, second))
    return expected


def read_quad(renderer, adapter, vs, ps, sampler):
    run = subprocess.run([renderer, adapter, '--startup-image-waterfall', str(vs), str(ps), sampler],
                         capture_output=True, text=True, timeout=30)
    print(run.stdout, end='')
    if run.returncode == 77:
        raise SystemExit(77)
    assert run.returncode == 0, run.stdout + run.stderr
    if 'debug_layer=enabled' in run.stdout:
        assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
    pixels = re.findall(r'^waterfall_rgba=((?:[0-9a-f]{8},){3}[0-9a-f]{8})$', run.stdout, re.M)
    assert len(pixels) == 4, run.stdout
    return [bytes.fromhex(pixel.replace(',', '')) for pixel in pixels]


def draw(dxc, renderer, adapter):
    pipeline = controls()
    source = pipeline.hlsl_ps()
    sampler, = (line for line in pipeline.manifest().splitlines() if line.startswith('smpl '))
    work_root = Path(__file__).resolve().parents[1] / 'out'
    work_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as tmp:
        work = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'waterfall_vs', 'vs_6_0', work)
        cases = (
            ('translated', source, reference()),
            ('fixed_index', source.replace(INDEX, '0'), reference(fixed_index=True)),
            ('mip_zero', source.replace(SAMPLE, f'.SampleLevel(q0, {UV}, 0)'), reference(mip=0)),
        )
        for name, hlsl, expected in cases:
            ps = compile_hlsl(dxc, 'dxc', hlsl, name, 'ps_6_0', work)
            actual = read_quad(renderer, adapter, vs, ps, sampler)
            assert actual == expected, (name, [p.hex() for p in actual], [p.hex() for p in expected])
            if name != 'translated':
                assert all(pixel != correct for pixel, correct in zip(actual, reference())), name


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'draw':
        draw(*sys.argv[2:])
    else:
        raise SystemExit('usage: startup_gfx2hlsl_pixel_waterfall_check.py [draw <dxc> <offscreen_check> <warp>]')
