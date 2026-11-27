"""Synthetic fixed descriptor-table join with WARP readback and controls."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_compute_global_load_check import (
    G, SENTINEL, WORDS, encode_mubuf, encode_smem, encode_sop1, encode_sopp,
    encode_vop1, run_case,
)
from startup_gfx2hlsl_join_symbols_check import VS_WORDS, root


def kernel(*, other_offset=16, read_offset=12):
    # Use the second descriptor of a wide load on one arm and load that
    # descriptor directly on the other. Both arms execute for some dispatch.
    return b''.join((
        encode_sop1(3, 12, 4),           # s_mov_b32 s12, s4 (table)
        encode_sop1(3, 13, 128),         # s_mov_b32 s13, 0 (fixed high word)
        struct.pack("<I", 0xbf068005),                 # s_cmp_eq_u32 s5, 0 (root constant)
        encode_sopp(5, 3),               # s_cbranch_scc1 to narrow load
        encode_smem(3, 20, 12),          # s_load_dwordx8 s[20:27], s[12:13], 0
        encode_sopp(2, 2),               # s_branch to join
        encode_smem(2, 24, 12, imm=other_offset),
        encode_sopp(12, 0),              # s_waitcnt
        encode_smem(8, 32, 24, imm=read_offset),
        encode_sopp(12, 0),
        encode_vop1(1, 1, 32),          # v_mov_b32 v1, s32
        encode_smem(2, 28, 12, imm=32),  # output descriptor
        encode_sopp(12, 0),
        encode_mubuf(0x1c, 1, 0, 28, offen=False),
        encode_sopp(1),
    ))


def translate(**changes):
    return G.ComputePipeline(kernel(**changes), root('t', 1), 12, (1, 1, 1))


def pixel(selector, mip=0):
    code = b''.join((
        encode_sop1(3, 14, 128 + selector),
        struct.pack("<I", 0xbf06800e),  # s_cmp_eq_u32 s14, 0
        encode_sopp(5, 3),
        encode_smem(4, 16, 8),          # wide load includes the image at byte 32
        encode_sopp(2, 2),
        encode_smem(3, 24, 8, imm=32),
        encode_sopp(12, 0),
        encode_vop1(1, 2, 128),
        encode_vop1(1, 3, 128),
        encode_vop1(1, 4, 128 + mip),
        # image_load_mip v[8:11], v[2:4], s[24:31], 2D, RGBA
        struct.pack("<II", 0xf0040f08, 2 | (8 << 8) | ((24 // 4) << 16)),
        encode_sopp(12, 0),
        # exp mrt0 v8, v9, v10, v11 done vm
        struct.pack("<II", 0xf800180f, 8 | (9 << 8) | (10 << 16) | (11 << 24)),
        encode_sopp(1),
    ))
    return G.Pipeline(VS_WORDS, code, root('t'))


def controls():
    assert "srv 0 tex2d table 0 32\n" in pixel(0).manifest()
    pipe = translate()
    assert 'srv 0 buf table 0 16\n' in pipe.manifest(), pipe.manifest()
    assert 'uav 0 table 0 32\n' in pipe.manifest(), pipe.manifest()
    assert 'r0.Load(0x0000000cu + 0)' in pipe.hlsl_cs(), pipe.hlsl_cs()
    try:
        translate(other_offset=0)
    except G.Unsupported as reason:
        assert str(reason) == 'descriptor operand s[24:27] holds (\'joined at a branch target\',)', reason
    else:
        raise AssertionError('different table addresses merged')


def dispatch(compiler, renderer, adapter):
    source = [i * 17 + 39 for i in range(WORDS)]
    reference = [source[3]] + [SENTINEL] * (WORDS - 1)
    out = Path(__file__).resolve().parents[1] / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as directory:
        for selector in (0, 1):
            for offset in (12, 8):
                mutated = offset != 12
                result = run_case(Path(directory), compiler, renderer, adapter, f'join_{selector}_{offset}',
                                  translate(read_offset=offset).hlsl_cs(), (1, 1, 1),
                                  reference, selector, 0, source)
                print(result.stdout, end='')
                if result.returncode == 77:
                    return 77
                if mutated:
                    assert result.returncode == 1 and 'Compute readback differs' in result.stderr, result
                else:
                    assert result.returncode == 0 and f'compared_words={WORDS} mismatches=0' in result.stdout, result

        for selector in (0, 1):
            for mip in (0, 2):
                source_path = Path(directory) / 'join_ps.hlsl'
                dxil = source_path.with_suffix('.dxil')
                hlsl = pixel(selector, mip).hlsl_ps()
                hlsl = hlsl.replace('struct PSIn {', 'struct PSIn {\n    float4 uv : TEXCOORD0;')
                source_path.write_text(hlsl, encoding='ascii')
                subprocess.run([compiler, '-T', 'ps_6_0', '-E', 'main', '-Fo', dxil, source_path], check=True)
                result = subprocess.run([renderer, adapter, '--check-pixel-mip-exact', dxil, '0'],
                                        capture_output=True, text=True)
                report = result.stdout + result.stderr
                print(report, end='')
                if result.returncode == 77:
                    return 77
                if mip == 0:
                    assert result.returncode == 0 and 'pixels=64 channels=256 max_absolute_error=0' in report, result
                else:
                    assert result.returncode == 1 and 'pixel_mismatch x=0 y=0' in report, result
    return 0


if __name__ == '__main__':
    controls()
    if len(sys.argv) > 1:
        sys.exit(dispatch(*sys.argv[1:]))
