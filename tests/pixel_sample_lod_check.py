"""Verify image_sample_lz, image_sample_l and image_sample_b lowering on D3D12."""
from pathlib import Path
import os
import struct
import subprocess
import sys
import tempfile

from pixel_aot_check import RAMP, must_reject, translate, write_record
from startup_image_sample_lod_check import LINEAR_MIP, bias_stage, sampler_line


def single(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def export16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xff
    mant = bits & 0x7fffff
    if exp == 0:
        return -0.0 if sign else 0.0
    h_exp = exp - 112
    if h_exp <= 0:
        return -0.0 if sign else 0.0
    h_mant = mant >> 13
    rebuilt = (sign << 31) | ((h_exp + 112) << 23) | (h_mant << 13)
    return struct.unpack("<f", struct.pack("<I", rebuilt))[0]


def ramp_expected():
    out = bytearray()
    for y in RAMP:
        for x in RAMP:
            out += struct.pack("<4f", export16(single(x / 255.0)),
                               export16(single(y / 255.0)),
                               export16(single(128.0 / 255.0)), 1.0)
    return bytes(out)


def base_prefix(*, quad_scale=False, swap_uv=False):
    ux, uy = ("attr0.y", "attr0.x") if swap_uv else ("attr0.x", "attr0.y")
    lines = [
        "s_mov_b64 s[50:51], exec",
        "s_wqm_b64 exec, exec",
        "s_mov_b32 m0, s24",
        f"v_interp_p1_f32_e32 v8, v0, {ux}",
        f"v_interp_p1_f32_e32 v9, v0, {uy}",
        f"v_interp_p2_f32_e32 v9, v1, {uy}",
        f"v_interp_p2_f32_e32 v8, v1, {ux}",
        "s_mov_b32 s21, 0",
    ]
    if quad_scale:
        lines += [
            "s_load_dwordx4 s[44:47], s[20:21], 0x40",
            "s_waitcnt lgkmcnt(0)",
            "s_buffer_load_dwordx4 s[60:63], s[44:47], null",
            "s_waitcnt lgkmcnt(0)",
            "v_fma_f32 v8, s62, v8, 0",
        ]
    lines += [
        "s_and_b64 exec, exec, s[50:51]",
        "s_mov_b32 s40, 1",
        "s_mov_b32 s41, 2",
        "s_mov_b32 s42, 3",
        "s_mov_b32 s43, 4",
        "s_load_dwordx8 s[28:35], s[20:21], 0x0",
        "s_waitcnt lgkmcnt(0)",
    ]
    return lines


def export_tail(swap_rg=False):
    r, g = ("v13", "v12") if swap_rg else ("v12", "v13")
    return [
        "s_waitcnt vmcnt(0)",
        f"v_cvt_pkrtz_f16_f32_e32 v16, {r}, {g}",
        "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
        "exp mrt0 v16, v16, v17, v17 done compr vm",
        "s_endpgm",
    ]


def lz_program(*, quad_scale=False, swap_rg=False):
    return base_prefix(quad_scale=quad_scale) + [
        "image_sample_lz v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
    ] + export_tail(swap_rg=swap_rg)


def lz_constant_uv_program():
    # Explicit-LOD sample using constant-buffer UVs without WQM or v_interp.
    return [
        "s_mov_b32 s21, 0",
        "s_load_dwordx4 s[44:47], s[20:21], 0x40",
        "s_load_dwordx8 s[28:35], s[20:21], 0x0",
        "s_mov_b32 s40, 1",
        "s_mov_b32 s41, 2",
        "s_mov_b32 s42, 3",
        "s_mov_b32 s43, 4",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dwordx4 s[60:63], s[44:47], null",
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v8, s60",
        "v_mov_b32_e32 v9, s61",
        "image_sample_lz v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
    ] + export_tail()


def lod_program(lod_literal):
    return base_prefix() + [
        f"v_mov_b32_e32 v10, {lod_literal}",
        "image_sample_l v[12:15], v[8:10], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
    ] + export_tail()


def bias_program(bias_literal, *, swap_uv=False):
    return base_prefix(swap_uv=swap_uv) + [
        f"v_mov_b32_e32 v7, {bias_literal}",
        "image_sample_b v[12:15], v[7:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
    ] + export_tail()


def check_rejections():
    # Wrong coordinate register widths.
    must_reject(translate, [s.replace("v[8:9]", "v[8:10]") for s in lz_program()])
    must_reject(translate, [s.replace("v[8:10]", "v[8:9]") for s in lod_program("0.0")])
    must_reject(translate, [s.replace("v[7:9]", "v[8:9]") for s in bias_program("0.0")])
    # Uninitialized lod register.
    must_reject(translate, [s for s in lod_program("0.0") if not s.startswith("v_mov_b32_e32 v10")])
    # image_sample_b requires helper-lane UV coordinates (implicit derivatives).
    bad_bias_uv = bias_program("0.0")
    idx = bad_bias_uv.index("v_mov_b32_e32 v7, 0.0")
    bad_bias_uv.insert(idx + 1, "v_mov_b32_e32 v8, 0.5")
    must_reject(translate, bad_bias_uv)
    # Using a sampled channel before vmcnt(0) wait.
    must_reject(translate, [s for s in lz_program() if s != "s_waitcnt vmcnt(0)"])
    must_reject(translate, [s for s in lod_program("0.0") if s != "s_waitcnt vmcnt(0)"])
    must_reject(translate, [s for s in bias_program("0.0") if s != "s_waitcnt vmcnt(0)"])
    # Using an outstanding sample VGPR as LOD for a second sample.
    bad_lod_dep = base_prefix() + [
        "image_sample_lz v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
        "v_mov_b32_e32 v10, v12",
        "image_sample_l v[16:19], v[8:10], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
        "s_waitcnt vmcnt(0)",
        "v_cvt_pkrtz_f16_f32_e32 v16, v16, v17",
        "v_cvt_pkrtz_f16_f32_e32 v18, v18, v19",
        "exp mrt0 v16, v16, v18, v18 done compr vm",
        "s_endpgm",
    ]
    must_reject(translate, bad_lod_dep)


def compile_ps(compiler, hlsl_text, directory, stem):
    src = directory / f"{stem}.hlsl"
    cso = directory / f"{stem}.cso"
    src.write_text(hlsl_text, encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", cso, src], check=True)
    return cso


def check_static_sampler_bias(directory, compiler, renderer, backend):
    def reference(lod):
        # The three 8x8 texture mips in startup_sampler_lod.cpp hold these solid red values.
        colors = (0.125, 0.375, 0.625)
        lod = min(max(lod, 0), len(colors) - 1)
        mip = int(lod)
        fraction = lod - mip
        return colors[mip] * (1 - fraction) + colors[min(mip + 1, len(colors) - 1)] * fraction

    def draw(shader, sampler, lod, *, wrong_lod=None):
        result = subprocess.run([renderer, backend, "--startup-sampler-bias", shader, sampler, str(reference(lod))],
                                capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode == 77:
            sys.exit(77)
        if "debug_layer=enabled" in result.stdout:
            assert "debug_errors=0 debug_warnings=0" in result.stdout, (result.stdout, result.stderr)
        if wrong_lod is None:
            assert result.returncode == 0 and "pixels=16 mismatches=0" in result.stdout, (sampler, result.stderr)
        else:
            assert result.returncode == 1 and "pixels=16 mismatches=16" in result.stdout, (sampler, result.stderr)
            actual = f"actual={reference(wrong_lod):g}"
            lines = [line for line in result.stderr.splitlines() if line.startswith("sampler_bias_mismatch ")]
            assert len(lines) == 16 and all(line.endswith(actual) for line in lines), result.stderr

    # A 4x4 draw over the full 8x8 base level yields log2(8/4) = 1.
    implicit = bias_stage("sample", LINEAR_MIP | 0x3f80)
    sample = compile_ps(compiler, implicit.hlsl_ps(), directory, "static_bias")
    biased = sampler_line(implicit)
    draw(sample, biased, 0.5)
    zero = sampler_line(bias_stage("sample", LINEAR_MIP))
    draw(sample, zero, 1.0)
    draw(sample, zero, 0.5, wrong_lod=1.0)
    doubled = biased.replace("0x8003f80", "0x8003f00")
    draw(sample, doubled, 0.5, wrong_lod=0.0)

    for field in (0x3f80, 0x80):
        level_zero = bias_stage("lz", LINEAR_MIP | field)
        shader = compile_ps(compiler, level_zero.hlsl_ps(), directory, f"static_bias_lz_{field:x}")
        draw(shader, sampler_line(level_zero), 0.0)
    # With a positive bias, failing to clear it changes the mip even at explicit LOD zero.
    leaked = sampler_line(bias_stage("sample", LINEAR_MIP | 0x80))
    draw(shader, leaked, 0.0, wrong_lod=0.5)


def run(directory, compiler, renderer, backend):
    check_rejections()

    # 1. Verify mip selection on the 2-mip quad-LOD fixture (mip 0 = red, mip 1 = green).
    # image_sample_lz ignores the 4x horizontal derivative and stays on mip 0 (red -> swapped to green).
    lz_quad = compile_ps(compiler, translate(lz_program(quad_scale=True, swap_rg=True)),
                         directory, "lz_quad")
    res = subprocess.run([renderer, backend, "--check-pixel-quad-lod", lz_quad])
    if res.returncode:
        return res.returncode

    # Mutation control: implicit image_sample on the same 4x UV selects mip 1 (green -> swapped to red).
    implicit_quad_prog = [s.replace("image_sample_lz", "image_sample")
                          for s in lz_program(quad_scale=True, swap_rg=True)]
    implicit_quad = compile_ps(compiler, translate(implicit_quad_prog), directory, "implicit_quad")
    wrong = subprocess.run([renderer, backend, "--check-pixel-quad-lod", implicit_quad],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing image_sample_lz with image_sample escaped detection")

    # image_sample_l with lod=1.0 selects mip 1 (green) on unscaled UVs.
    lod1 = compile_ps(compiler, translate(lod_program("1.0")), directory, "lod1")
    subprocess.run([renderer, backend, "--check-pixel-quad-lod", lod1], check=True)

    # Mutation control: lod=0.0 selects mip 0 (red) and must fail.
    lod0 = compile_ps(compiler, translate(lod_program("0.0")), directory, "lod0_quad")
    wrong = subprocess.run([renderer, backend, "--check-pixel-quad-lod", lod0],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Changing image_sample_l lod from 1.0 to 0.0 escaped detection")

    # image_sample_b with bias=1.0 shifts implicit mip 0 to mip 1 (green).
    bias1 = compile_ps(compiler, translate(bias_program("1.0")), directory, "bias1")
    subprocess.run([renderer, backend, "--check-pixel-quad-lod", bias1], check=True)

    # Mutation control: bias=0.0 stays on mip 0 (red) and must fail.
    bias0 = compile_ps(compiler, translate(bias_program("0.0")), directory, "bias0_quad")
    wrong = subprocess.run([renderer, backend, "--check-pixel-quad-lod", bias0],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Changing image_sample_b bias from 1.0 to 0.0 escaped detection")

    # 2. Verify spatially varying 8x8 ramp coordinates on --check-pixel-case.
    case = directory / "ramp.case"
    case.write_bytes(bytes(256) + ramp_expected())
    for stem, program in (
        ("lz_ramp", lz_program()),
        ("lod_ramp", lod_program("0.0")),
        ("bias_ramp", bias_program("0.0")),
    ):
        cso = compile_ps(compiler, translate(program), directory, stem)
        subprocess.run([renderer, backend, "--check-pixel-case", cso, case], check=True)

    # Mutation control: swapped (u, v) on image_sample_b must be detected.
    swapped_bias = compile_ps(compiler, translate(bias_program("0.0", swap_uv=True)), directory, "bias_swapped")
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", swapped_bias, case],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapping UV coordinates in image_sample_b escaped detection")

    # 3. Verify image_sample_lz with non-WQM constant-buffer UV coordinates.
    const_u, const_v = (3 + 0.5) / 8.0, (5 + 0.5) / 8.0
    const_cb = struct.pack("<4f", const_u, const_v, 0.0, 0.0) + bytes(240)
    const_px = struct.pack("<4f", export16(single(RAMP[3] / 255.0)),
                           export16(single(RAMP[5] / 255.0)),
                           export16(single(128.0 / 255.0)), 1.0)
    const_case = directory / "const_uv.case"
    const_case.write_bytes(const_cb + const_px * 64)
    lz_const = compile_ps(compiler, translate(lz_constant_uv_program()), directory, "lz_const")
    subprocess.run([renderer, backend, "--check-pixel-case", lz_const, const_case], check=True)

    check_static_sampler_bias(directory, compiler, renderer, backend)
    write_record(os.environ.get("FORM_COVERAGE_RECORD"))
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(run(Path(tmp), *sys.argv[1:]))
