"""Check a one-word descriptor-backed wave32 pixel mask, with optional WARP readback."""

import struct
import sys
import tempfile
from pathlib import Path

import gpu_case_runner
from startup_gfx2hlsl_desc_vcc_check import G, TABLE_ROOT, compile_hlsl, smem, vs_words
from startup_gfx2hlsl_refusals_check import put_request


REASON = 's_buffer_load destination vcc_lo for 1 dwords'
SHIFT = 'WaveGetLaneIndex() & 3u'
VERTEX = """
struct Out {
    float4 param0 : PARAM0;
    float4 param1 : PARAM1;
    float4 param2 : PARAM2;
    float4 pos : SV_Position;
};
Out main(uint vertex : SV_VertexID) {
    float2 uv = float2((vertex << 1u) & 2u, vertex & 2u);
    Out o;
    o.param0 = o.param1 = o.param2 = float4(0, 0, 0, 0);
    o.pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return o;
}
"""


def pixel_words(wave32=True, scalar_consumer=False):
    words = (
        0xbefe097e if wave32 else 0xbf800000,  # s_wqm_b32 exec_lo, exec_lo or s_nop
        *smem(2, 36, 8, 0),  # s_load_dwordx4 s[36:39], s[8:9], 0x0
        0xbf8c0000,
        0x7e0a02f2,  # v_mov_b32_e32 v5, 1.0
        *smem(8, 106, 36, 4),  # s_buffer_load_dword vcc_lo, s[36:39], 0x4
        0xbf8c0000,
        *((0xbe9a036a,) if scalar_consumer else ()),  # s_mov_b32 s26, vcc_lo
        0x020c0a80,  # v_cndmask_b32_e32 v6, 0, v5, vcc_lo
        0xf800180f, 0x06060606,  # exp mrt0 v6, v6, v6, v6 done vm
        0xbf810000,
    )
    return struct.pack('<%dI' % len(words), *words)


def reference(word):
    # A 2x2 target is one quad. Lanes 0,2 are left and lanes 1,3 are right.
    return struct.pack('<16f', *(float((word >> lane) & 1) for lane in range(4) for _ in range(4)))


def controls(directory):
    vs = vs_words([(2, 3, 5), (7, 11, 13), (17, 19, 23), (29, 31, 37)])
    for code, expected in (
        (pixel_words(), []),
        (pixel_words(wave32=False), [(REASON, 'ps')]),
        (pixel_words(scalar_consumer=True), [(REASON, 'ps')]),
    ):
        key = put_request(directory, vs, code, TABLE_ROOT)
        before = set(directory.iterdir())
        found = G.request_refusals(directory, key)
        assert set(directory.iterdir()) == before
        assert found == expected, (found, expected)

    pipe = G.Pipeline(vs, pixel_words(), TABLE_ROOT)
    source = pipe.hlsl_ps()
    assert pipe.ps_profile() == 'ps_6_0'
    assert source.count(SHIFT) == 1 and 'ByteAddressBuffer r0 : register(t0);' in source, source
    return source


def draw(directory, source, dxc, renderer):
    vs = compile_hlsl(dxc, 'dxc', VERTEX, 'vs', 'vs_6_0', directory)
    for mutant in (False, True):
        name = 'bit_zero' if mutant else 'mask_word'
        text = source.replace(SHIFT, '0u') if mutant else source
        ps = compile_hlsl(dxc, 'dxc', text, name, 'ps_6_0', directory)
        for word in (0xaaaaaaaa, 0x55555555):
            expected = directory / 'reference.bin'
            expected.write_bytes(reference(word))
            result = gpu_case_runner.run(renderer, 'warp', [
                '--startup-pixel-buffer-words', vs, ps, '2', '2', expected, '0', f'deadbeef,{word:08x}',
            ])
            report = result.stdout + result.stderr
            print(report, end='')
            if result.returncode == 77:
                return 77
            if mutant:
                column = 1 - (word & 1)
                assert result.returncode != 0 and f'pixel mismatch ({column},0)' in report, report
                assert '2 of 4 pixels differ' in report, report
            else:
                assert result.returncode == 0 and 'all 4 pixels matched' in report, report
    return 0


def main():
    output = Path(__file__).resolve().parents[1] / 'out'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        directory = Path(tmp)
        source = controls(directory)
        print('wave32 translates, unknown wave size and scalar consumer still refuse')
        if len(sys.argv) > 1:
            return draw(directory, source, *sys.argv[1:])
    return 0


if __name__ == '__main__':
    sys.exit(main())
