"""Parameter exports up to param31 and pixel-input pairing in mesh/startup translators (#2413 T4).

  (no arguments)                          pure-Python linkage controls, refusals and guard mutations
  binary <llvm-mc>                        reassemble wave64 mesh and wave32 pixel fixtures with llvm-mc
  draw <dxc> <offscreen_check> <adapter>  compile ms_6_5 + ps_6_0 pairs, draw on D3D12, and check pixel readback
                                          plus negative controls
"""

from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "startup"))
sys.path.insert(0, str(ROOT / "tests"))

import gfx2hlsl as G  # noqa: E402
import mesh_dynamic_aot  # noqa: E402
import mesh_entry_aot  # noqa: E402
import mesh_param_link  # noqa: E402
from mesh_dynamic_aot_draw_check import compile_shader  # noqa: E402
from mesh_dynamic_geometry_check import CODE as BASE_CODE, MESH_LIMITS, SECOND, decode  # noqa: E402
from startup_flat_param_check import root_blob  # noqa: E402
from startup_gfx2hlsl_interp_f16_check import P1LL_SRC2_MASK  # noqa: E402

WAVE64 = "+wavefrontsize64,-wavefrontsize32"
SIDE = 32

# Insert extra parameter exports right after `exp param0 v5, v6, v7, v4` in tests/mesh_dynamic_geometry.s.
# All branches in BASE_CODE precede that block, so instruction PCs after the insertion stay branch-free.
PARAM0_ANCHOR = bytes.fromhex("0f0200f805060704")
ASSERT_ANCHOR = BASE_CODE.count(PARAM0_ANCHOR) == 1


def encode_exp(target, vs, *, done=False, vm=False):
    w0 = 0xF8000000 | (int(vm) << 12) | (int(done) << 11) | (target << 4) | 0xF
    w1 = vs[0] | (vs[1] << 8) | (vs[2] << 16) | (vs[3] << 24)
    return struct.pack("<II", w0, w1)


def encode_v_mov_lit(vdst, literal):
    return struct.pack("<II", 0x7E0002FF | ((vdst & 0xFF) << 17), literal & 0xFFFFFFFF)


def encode_vintrp(op, vdst, attr, chan, vsrc):
    return struct.pack("<I", 0xC8000000 | ((vdst & 0xFF) << 18) | (op << 16) | ((attr & 0x3F) << 10)
                       | ((chan & 3) << 8) | (vsrc & 0xFF))


def encode_interp_f16(op, vdst, src1, attr, chan, high=0, src2=0):
    w0 = (0b110101 << 26) | (op << 16) | (vdst & 0xFF)
    attribute = ((high & 1) << 8) | ((chan & 3) << 6) | (attr & 0x3F)
    w1 = attribute | ((src1 & 0x1FF) << 9) | ((src2 & 0x1FF) << 18)
    return struct.pack("<II", w0, w1)


def encode_cvt_f32_f16(vdst, vsrc):
    return struct.pack("<I", 0x7E000000 | ((vdst & 0xFF) << 17) | (0x0B << 9) | (0x100 + (vsrc & 0xFF)))


def with_mesh_exports(extra_bytes):
    assert ASSERT_ANCHOR
    return BASE_CODE.replace(PARAM0_ANCHOR, PARAM0_ANCHOR + extra_bytes)


# Mesh variant 1: exports param0 (u, v, 0.5, 1.0), param5 (0.25, 0, 1.0, 1.0), and param12 (v, u, 0, 1.0).
MESH_PARAM12_EXTRA = (
    encode_v_mov_lit(8, 0x3E800000)                 # v8 = 0.25f
    + encode_exp(32 + 5, (8, 3, 4, 4))              # exp param5 v8, v3, v4, v4
    + encode_exp(32 + 12, (6, 5, 3, 4))             # exp param12 v6, v5, v3, v4
)
MESH_PARAM12_CODE = with_mesh_exports(MESH_PARAM12_EXTRA)
MESH_PARAM12_ASM = (
    "v_mov_b32_e32 v8, 0x3e800000\n"
    "exp param5 v8, v3, v4, v4\n"
    "exp param12 v6, v5, v3, v4\n"
)

# Mesh variant 2: exports param0 and param12 holding packed binary16 pairs (0.25, 1.0) and (0.125, 0.75).
MESH_F16_EXTRA = (
    encode_v_mov_lit(8, 0x3C003400)                 # v8  = pack_f16(0.25, 1.0)
    + encode_v_mov_lit(12, 0x3A003000)              # v12 = pack_f16(0.125, 0.75)
    + encode_exp(32 + 12, (8, 12, 3, 4))            # exp param12 v8, v12, v3, v4
)
MESH_F16_CODE = with_mesh_exports(MESH_F16_EXTRA)
MESH_F16_ASM = (
    "v_mov_b32_e32 v8, 0x3c003400\n"
    "v_mov_b32_e32 v12, 0x3a003000\n"
    "exp param12 v8, v12, v3, v4\n"
)

# Mesh variant 3: sparse high indices, param0 (u, v, 0.5, 1.0), param18 (v, u, 0, 1.0) and param28 (0.25, 0, 1.0, 1.0).
MESH_PARAM28_EXTRA = (
    encode_v_mov_lit(8, 0x3E800000)                 # v8 = 0.25f
    + encode_exp(32 + 28, (8, 3, 4, 4))             # exp param28 v8, v3, v4, v4
    + encode_exp(32 + 18, (6, 5, 3, 4))             # exp param18 v6, v5, v3, v4
)
MESH_PARAM28_CODE = with_mesh_exports(MESH_PARAM28_EXTRA)
MESH_PARAM28_ASM = (
    "v_mov_b32_e32 v8, 0x3e800000\n"
    "exp param28 v8, v3, v4, v4\n"
    "exp param18 v6, v5, v3, v4\n"
)

# Negative-control mesh variant with an export outside param0..param31 (target 32 + 32 is not a parameter).
MESH_PARAM32_TEXT = "exp param32 v5, v6, v7, v4"

# Pixel shader 1: linear attr0.x (R), linear attr12.x (G), flat attr5.x (B), linear attr12.w (A).
PS_LINEAR_FLAT_ASM = (
    "s_mov_b32 m0, s2\n"
    "v_interp_p1_f32 v2, v0, attr0.x\n"
    "v_interp_p2_f32 v2, v1, attr0.x\n"
    "v_interp_p1_f32 v3, v0, attr12.x\n"
    "v_interp_p2_f32 v3, v1, attr12.x\n"
    "v_interp_mov_f32 v4, p0, attr5.x\n"
    "v_interp_p1_f32 v5, v0, attr12.w\n"
    "v_interp_p2_f32 v5, v1, attr12.w\n"
    "exp mrt0 v2, v3, v4, v5 done vm\n"
    "s_endpgm\n"
)
PS_LINEAR_FLAT_WORDS = b"".join((
    bytes.fromhex("0203fcbe"),
    encode_vintrp(0, 2, 0, 0, 0),
    encode_vintrp(1, 2, 0, 0, 1),
    encode_vintrp(0, 3, 12, 0, 0),
    encode_vintrp(1, 3, 12, 0, 1),
    encode_vintrp(2, 4, 5, 0, 2),
    encode_vintrp(0, 5, 12, 3, 0),
    encode_vintrp(1, 5, 12, 3, 1),
    encode_exp(0, (2, 3, 4, 5), done=True, vm=True),
    bytes.fromhex("000081bf"),
))

# Pixel shader 3: linear attr0.x (R), linear attr18.x (G), linear attr28.x (B, the constant 0.25), linear attr18.w (A).
PS_HIGH_ASM = (
    "s_mov_b32 m0, s2\n"
    "v_interp_p1_f32 v2, v0, attr0.x\n"
    "v_interp_p2_f32 v2, v1, attr0.x\n"
    "v_interp_p1_f32 v3, v0, attr18.x\n"
    "v_interp_p2_f32 v3, v1, attr18.x\n"
    "v_interp_p1_f32 v4, v0, attr28.x\n"
    "v_interp_p2_f32 v4, v1, attr28.x\n"
    "v_interp_p1_f32 v5, v0, attr18.w\n"
    "v_interp_p2_f32 v5, v1, attr18.w\n"
    "exp mrt0 v2, v3, v4, v5 done vm\n"
    "s_endpgm\n"
)
PS_HIGH_WORDS = b"".join((
    bytes.fromhex("0203fcbe"),
    encode_vintrp(0, 2, 0, 0, 0),
    encode_vintrp(1, 2, 0, 0, 1),
    encode_vintrp(0, 3, 18, 0, 0),
    encode_vintrp(1, 3, 18, 0, 1),
    encode_vintrp(0, 4, 28, 0, 0),
    encode_vintrp(1, 4, 28, 0, 1),
    encode_vintrp(0, 5, 18, 3, 0),
    encode_vintrp(1, 5, 18, 3, 1),
    encode_exp(0, (2, 3, 4, 5), done=True, vm=True),
    bytes.fromhex("000081bf"),
))

# Pixel shader 2: reads both halves of attr12.x and attr12.y via v_interp_p1ll_f16 / v_interp_p2_f16.
PS_F16_ASM = (
    "s_mov_b32 m0, s2\n"
    "v_interp_p1ll_f16 v4, v0, attr12.x\n"
    "v_interp_p2_f16 v4, v1, attr12.x, v4\n"
    "v_interp_p1ll_f16 v5, v0, attr12.x high\n"
    "v_interp_p2_f16 v5, v1, attr12.x, v5 high\n"
    "v_interp_p1ll_f16 v6, v0, attr12.y\n"
    "v_interp_p2_f16 v6, v1, attr12.y, v6\n"
    "v_interp_p1ll_f16 v7, v0, attr12.y high\n"
    "v_interp_p2_f16 v7, v1, attr12.y, v7 high\n"
    "v_cvt_f32_f16 v4, v4\n"
    "v_cvt_f32_f16 v5, v5\n"
    "v_cvt_f32_f16 v6, v6\n"
    "v_cvt_f32_f16 v7, v7\n"
    "exp mrt0 v4, v5, v6, v7 done vm\n"
    "s_endpgm\n"
)
PS_F16_WORDS = b"".join((
    bytes.fromhex("0203fcbe"),
    encode_interp_f16(0x342, 4, 0x100, 12, 0, high=0, src2=0),
    encode_interp_f16(0x35A, 4, 0x101, 12, 0, high=0, src2=0x104),
    encode_interp_f16(0x342, 5, 0x100, 12, 0, high=1, src2=0),
    encode_interp_f16(0x35A, 5, 0x101, 12, 0, high=1, src2=0x105),
    encode_interp_f16(0x342, 6, 0x100, 12, 1, high=0, src2=0),
    encode_interp_f16(0x35A, 6, 0x101, 12, 1, high=0, src2=0x106),
    encode_interp_f16(0x342, 7, 0x100, 12, 1, high=1, src2=0),
    encode_interp_f16(0x35A, 7, 0x101, 12, 1, high=1, src2=0x107),
    encode_cvt_f32_f16(4, 4),
    encode_cvt_f32_f16(5, 5),
    encode_cvt_f32_f16(6, 6),
    encode_cvt_f32_f16(7, 7),
    encode_exp(0, (4, 5, 6, 7), done=True, vm=True),
    bytes.fromhex("000081bf"),
))

# Depth-only pixel shader (no attribute reads, mrtz depth export).
PS_DEPTH_ONLY_WORDS = bytes.fromhex("8002007e" "810800f800000000" "000081bf")


def assemble_mc(llvm_mc, source, mattr=None):
    cmd = [llvm_mc, "-arch=amdgcn", "-mcpu=gfx1013"]
    if mattr:
        cmd.append(f"-mattr={mattr}")
    cmd.append("-show-encoding")
    run = subprocess.run(cmd, input=source, capture_output=True, text=True, check=True)
    words = bytearray()
    for line in run.stdout.splitlines():
        m = re.search(r"encoding: \[([^\]]*)\]", line)
        if m:
            words += bytes(int(b, 16) for b in m.group(1).split(","))
    return bytes(words)


def normalise_p1ll(words):
    out = bytearray(words)
    off = 0
    for _, size, name, _ in G.decode(words):
        if name.startswith("v_interp_p1ll_f16"):
            w1 = struct.unpack_from("<I", out, off + 4)[0] & ~P1LL_SRC2_MASK
            struct.pack_into("<I", out, off + 4, w1)
        off += size
    return bytes(out)


def expect_error(exc_type, needle, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as error:
        assert needle in str(error), (needle, str(error))
        return
    raise AssertionError(f"Expected {exc_type.__name__}({needle!r})")


def expected_pixels(counts, mode):
    """Independent CPU rasterizer for 32x32 target (triangle=0x1003, quad=0x2004)."""
    assert counts in (0x1003, 0x2004), hex(counts)
    quad = counts == 0x2004
    pixels = bytearray()
    for y in range(SIDE):
        for x in range(SIDE):
            inside = quad or y > x
            if not inside:
                pixels.extend((0, 0, 0, 255))
            elif mode == "linear_flat":
                r = round((x + 0.5) / SIDE * 255)
                g = round((SIDE - y - 0.5) / SIDE * 255)
                pixels.extend((r, g, 64, 255))
            elif mode == "f16":
                pixels.extend((64, 255, 32, 191))
            else:
                raise ValueError(mode)
    return bytes(pixels)


def fixture(counts, geometry, pixels):
    data = bytearray(1536)
    struct.pack_into("<II", data, 1024, counts, geometry)
    data.extend(b"\xcd" * 80)
    data.extend(pixels)
    return bytes(data)


def controls():
    # 0. Verify all newly assembled instruction bytes decode to the exact expected assembly.
    for raw, asm_text in (
        (MESH_PARAM12_EXTRA, MESH_PARAM12_ASM),
        (MESH_F16_EXTRA, MESH_F16_ASM),
        (MESH_PARAM28_EXTRA, MESH_PARAM28_ASM),
        (PS_LINEAR_FLAT_WORDS, PS_LINEAR_FLAT_ASM),
        (PS_F16_WORDS, PS_F16_ASM),
        (PS_HIGH_WORDS, PS_HIGH_ASM),
    ):
        decoded_lines = "\n".join(asm.replace("_e32", "") for _, _, _, asm in G.decode(raw)) + "\n"
        want_lines = "\n".join(l.replace("_e32", "") for l in asm_text.splitlines()) + "\n"
        assert decoded_lines == want_lines, (decoded_lines, want_lines)

    # 1. Static + dynamic mesh paired with MeshPixelStage (param0, flat param5, linear param12).
    ps_stage = G.MeshPixelStage((0, 5, 12), PS_LINEAR_FLAT_WORDS, root_blob())
    linkage = ps_stage.linkage()
    assert linkage.params == (0, 5, 12)
    assert linkage.flat == (5,)
    assert linkage.f16_plain == ()
    assert linkage.pixel_attrs == (0, 5, 12)
    ps_hlsl = ps_stage.hlsl_ps()
    assert "float4 param12 : PARAM12;" in ps_hlsl and "nointerpolation uint4 fparam5 : FPARAM5;" in ps_hlsl
    assert "cbuffer entry" not in ps_hlsl

    decoded_m12 = decode(MESH_PARAM12_CODE)
    static_hlsl = mesh_entry_aot.lower(decoded_m12, [0x2004, SECOND], MESH_LIMITS, param_linkage=linkage)
    dynamic_hlsl = mesh_dynamic_aot.lower(decoded_m12, 2, MESH_LIMITS, param_linkage=linkage)
    for text in (static_hlsl, dynamic_hlsl):
        for token in ("float4 param0 : PARAM0;", "float4 param12 : PARAM12;", "nointerpolation uint4 fparam5 : FPARAM5;",
                      ".param1 = float4(0.0f, 0.0f, 0.0f, 0.0f);", ".fparam5 = asuint(p5);"):
            assert token in text, (token, text)

    # 2. Static + dynamic mesh paired with MeshPixelStage reading binary16 halves from param12.
    ps_f16_stage = G.MeshPixelStage((0, 12), PS_F16_WORDS, root_blob())
    f16_link = ps_f16_stage.linkage()
    assert f16_link.f16_plain == (12,)
    assert "float4 hparam12_lo : HPARAML12;" in ps_f16_stage.hlsl_ps()
    decoded_f16 = decode(MESH_F16_CODE)
    static_f16 = mesh_entry_aot.lower(decoded_f16, [0x2004, SECOND], MESH_LIMITS, param_linkage=f16_link)
    dynamic_f16 = mesh_dynamic_aot.lower(decoded_f16, 2, MESH_LIMITS, param_linkage=f16_link)
    for text in (static_f16, dynamic_f16):
        assert "float4 hparam12_lo : HPARAML12;" in text and "float4 hparam12_hi : HPARAMH12;" in text, text
        assert ".hparam12_lo = f16tof32(h12);" in text and ".hparam12_hi = f16tof32(h12 >> 16);" in text, text

    # 2b. Sparse high indices up to the signature budget: param18 and param28 export and pair with the pixel reads.
    ps_high_stage = G.MeshPixelStage((0, 18, 28), PS_HIGH_WORDS, root_blob())
    high_link = ps_high_stage.linkage()
    assert high_link.params == (0, 18, 28) and high_link.pixel_attrs == (0, 18, 28)
    assert "float4 param28 : PARAM28;" in ps_high_stage.hlsl_ps()
    decoded_high = decode(MESH_PARAM28_CODE)
    static_high = mesh_entry_aot.lower(decoded_high, [0x2004, SECOND], MESH_LIMITS, param_linkage=high_link)
    dynamic_high = mesh_dynamic_aot.lower(decoded_high, 2, MESH_LIMITS, param_linkage=high_link)
    for text in (static_high, dynamic_high):
        for token in ("float4 param18 : PARAM18;", "float4 param28 : PARAM28;", ".param18 = ", ".param28 = "):
            assert token in text, (token, text)
    # param30 is the last index that fits beside SV_Position in 32 vectors; param31 is a valid index over the budget.
    assert mesh_param_link.validate_linkage((0, 30)).params == (0, 30)
    expect_error(ValueError, "needs 33 vectors", mesh_param_link.validate_linkage, (0, 31))

    # 3. Depth-only pairing (empty mesh_params and no pixel attribute reads) and legacy TEXCOORD0 default.
    depth_ps = G.MeshPixelStage((), PS_DEPTH_ONLY_WORDS, root_blob())
    assert depth_ps.linkage().params == ()
    assert "PARAM0" not in depth_ps.hlsl_ps()
    legacy_hlsl = mesh_entry_aot.lower(decode(BASE_CODE), [0x1003, SECOND], MESH_LIMITS)
    assert "float4 attr : TEXCOORD0;" in legacy_hlsl and "PARAM0" not in legacy_hlsl, legacy_hlsl

    # 4. Refusals: param13+, off component, compr, duplicate, incomplete vertex export, variant signature mismatch,
    # missing pixel attr, compute fallback with param12, and D3D12 32-vector signature limit.
    expect_error(ValueError, "Unsupported mesh export", mesh_entry_aot.lower,
                 [(pc, MESH_PARAM32_TEXT if text == "exp param0 v5, v6, v7, v4" else text)
                  for pc, text in decode(BASE_CODE)], [0x1003, SECOND], MESH_LIMITS)
    expect_error(ValueError, "Unsupported live export", mesh_entry_aot.lower,
                 [(0, "s_mov_b32 s15, 0x1003"), *decode(with_mesh_exports(encode_exp(32 + 12, (6, 5, 3, 4))))[:-1],
                  (900, "exp param12 v6, v5, off, v4"), (908, "s_endpgm")],
                 [0x1003, SECOND], MESH_LIMITS)
    expect_error(ValueError, "Duplicate vertex parameter param12", mesh_entry_aot.lower,
                 decode(with_mesh_exports(MESH_PARAM12_EXTRA + encode_exp(32 + 12, (6, 5, 3, 4)))),
                 [0x1003, SECOND], MESH_LIMITS)
    incomplete = [(pc, text.replace("exp param12 v6, v5, v3, v4", "exp param12 v6, v5, v3, v4") if pc else text)
                  for pc, text in decoded_m12]
    # Emit param12 only for vertex 0 (skip vertices 1 and 2 via v_cmpx_eq_u32).
    split_incomplete = []
    for pc, text in decoded_m12:
        if text == "exp param12 v6, v5, v3, v4":
            split_incomplete.extend([
                (pc, "v_cmpx_eq_u32_e32 0, v0"),
                (pc + 1, "exp param12 v6, v5, v3, v4"),
                (pc + 2, "s_mov_b64 exec, s[22:23]"),
                (pc + 3, "v_cmpx_gt_u32_e32 s16, v0"),
            ])
        else:
            split_incomplete.append((pc, text))
    expect_error(ValueError, "Incomplete vertex parameter param12",
                 mesh_entry_aot.lower, split_incomplete, [0x1003, SECOND], MESH_LIMITS)

    # Dynamic branch mismatch: only one runtime branch exports param12.
    mismatch_dyn = []
    shift = 0
    for pc, text in decode(BASE_CODE):
        if text == "exp param0 v5, v6, v7, v4":
            mismatch_dyn.extend([
                (pc, text),
                (pc + 8, "s_cmp_eq_u32 s15, 0x2004"),
                (pc + 12, "s_cbranch_scc0 2"),
                (pc + 16, "exp param12 v6, v5, v3, v4"),
            ])
            shift = 16
        else:
            mismatch_dyn.append((pc + shift, text))
    expect_error(ValueError, "disagree on the vertex signature",
                 mesh_dynamic_aot.lower, mismatch_dyn, 2, MESH_LIMITS)

    expect_error(G.Unsupported, "does not export", G.MeshPixelStage, (0, 5), PS_LINEAR_FLAT_WORDS, root_blob())
    expect_error(G.Unsupported, "does not export", G.MeshPixelStage, (0, 12), PS_LINEAR_FLAT_WORDS, root_blob())
    expect_error(ValueError, "compute fallback translates only param0",
                 mesh_entry_aot.lower, decoded_m12, [0x1003, SECOND], MESH_LIMITS, compute_fallback=True)
    expect_error(ValueError, "exceeding D3D12 limit", mesh_param_link.validate_linkage,
                 tuple(range(13)), flat=tuple(range(13)), f16_plain=(0, 1, 2, 3))
    expect_error(ValueError, "must be an integer in 0..31", mesh_param_link.validate_linkage, (0, 32))
    expect_error(G.Unsupported, "outside 0..31", G.MeshPixelStage, (0, 32), PS_DEPTH_ONLY_WORDS, root_blob())

    # 5. Guard mutation controls: weakening any check must be caught.
    with patch.object(mesh_param_link, "MAX_VERTEX_VECTORS", 64):
        expect_error(AssertionError, "Expected ValueError", expect_error, ValueError, "exceeding D3D12 limit",
                     mesh_param_link.validate_linkage, tuple(range(13)), flat=tuple(range(13)), f16_plain=(0, 1, 2, 3))
    # Past param31 the 32-vector budget refuses too, so the range check is not the only guard there; param31 itself is
    # a valid index that only the budget refuses (above), which a raised MAX_PARAM cannot hide.
    with patch.object(mesh_param_link, "MAX_PARAM", 30):
        expect_error(ValueError, "must be an integer in 0..30", mesh_param_link.validate_linkage, (0, 31))
    with patch.object(mesh_entry_aot, "hlsl_param_writes", lambda *a, **k: []):
        no_writes = mesh_entry_aot.lower(decoded_m12, [0x2004, SECOND], MESH_LIMITS, param_linkage=linkage)
        assert ".param12" not in no_writes

    print("mesh param0..param31 linkage controls passed")


def check_binary(llvm_mc):
    for label, src, raw in (
        ("mesh_param12_extra", MESH_PARAM12_ASM, MESH_PARAM12_EXTRA),
        ("mesh_f16_extra", MESH_F16_ASM, MESH_F16_EXTRA),
        ("mesh_param28_extra", MESH_PARAM28_ASM, MESH_PARAM28_EXTRA),
    ):
        built = assemble_mc(llvm_mc, src, mattr=WAVE64)
        if built != raw:
            raise AssertionError(f"{label}: assembled {built.hex()}, recorded {raw.hex()}")
    for label, src, raw, norm in (
        ("ps_linear_flat", PS_LINEAR_FLAT_ASM, PS_LINEAR_FLAT_WORDS, False),
        ("ps_f16", PS_F16_ASM, PS_F16_WORDS, True),
        ("ps_high", PS_HIGH_ASM, PS_HIGH_WORDS, False),
    ):
        built = assemble_mc(llvm_mc, src)
        if norm:
            built = normalise_p1ll(built)
        if built != raw:
            raise AssertionError(f"{label}: assembled {built.hex()}, recorded {raw.hex()}")
    print("mesh param0..param31 fixture words match llvm-mc")


def check_draw(dxc, renderer, adapter):
    controls()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        ps_lf_stage = G.MeshPixelStage((0, 5, 12), PS_LINEAR_FLAT_WORDS, root_blob())
        ps_lf_dxil = compile_shader(dxc, directory, "ps_lf", ps_lf_stage.hlsl_ps(), "ps_6_0")
        lf_link = ps_lf_stage.linkage()
        decoded_m12 = decode(MESH_PARAM12_CODE)
        dyn_lf_hlsl = mesh_dynamic_aot.lower(decoded_m12, 2, MESH_LIMITS, param_linkage=lf_link)
        dyn_lf_dxil = compile_shader(dxc, directory, "ms_dyn_lf", dyn_lf_hlsl, "ms_6_5")
        stat_lf_hlsl = mesh_entry_aot.lower(decoded_m12, [0x2004, SECOND], MESH_LIMITS, param_linkage=lf_link)
        stat_lf_dxil = compile_shader(dxc, directory, "ms_stat_lf", stat_lf_hlsl, "ms_6_5")

        ps_f16_stage = G.MeshPixelStage((0, 12), PS_F16_WORDS, root_blob())
        ps_f16_dxil = compile_shader(dxc, directory, "ps_f16", ps_f16_stage.hlsl_ps(), "ps_6_0")
        f16_link = ps_f16_stage.linkage()
        decoded_f16 = decode(MESH_F16_CODE)
        dyn_f16_dxil = compile_shader(
            dxc, directory, "ms_dyn_f16",
            mesh_dynamic_aot.lower(decoded_f16, 2, MESH_LIMITS, param_linkage=f16_link), "ms_6_5")
        stat_f16_dxil = compile_shader(
            dxc, directory, "ms_stat_f16",
            mesh_entry_aot.lower(decoded_f16, [0x2004, SECOND], MESH_LIMITS, param_linkage=f16_link), "ms_6_5")

        ps_high_stage = G.MeshPixelStage((0, 18, 28), PS_HIGH_WORDS, root_blob())
        ps_high_dxil = compile_shader(dxc, directory, "ps_high", ps_high_stage.hlsl_ps(), "ps_6_0")
        high_link = ps_high_stage.linkage()
        decoded_high = decode(MESH_PARAM28_CODE)
        dyn_high_hlsl = mesh_dynamic_aot.lower(decoded_high, 2, MESH_LIMITS, param_linkage=high_link)
        dyn_high_dxil = compile_shader(dxc, directory, "ms_dyn_high", dyn_high_hlsl, "ms_6_5")
        stat_high_dxil = compile_shader(
            dxc, directory, "ms_stat_high",
            mesh_entry_aot.lower(decoded_high, [0x2004, SECOND], MESH_LIMITS, param_linkage=high_link), "ms_6_5")

        draws = (
            ("dynamic_lf_triangle", dyn_lf_dxil, ps_lf_dxil, 0x1003, expected_pixels(0x1003, "linear_flat")),
            ("dynamic_lf_quad", dyn_lf_dxil, ps_lf_dxil, 0x2004, expected_pixels(0x2004, "linear_flat")),
            ("static_lf_quad", stat_lf_dxil, ps_lf_dxil, 0x2004, expected_pixels(0x2004, "linear_flat")),
            ("dynamic_f16_quad", dyn_f16_dxil, ps_f16_dxil, 0x2004, expected_pixels(0x2004, "f16")),
            ("static_f16_quad", stat_f16_dxil, ps_f16_dxil, 0x2004, expected_pixels(0x2004, "f16")),
            ("dynamic_high_triangle", dyn_high_dxil, ps_high_dxil, 0x1003, expected_pixels(0x1003, "linear_flat")),
            ("dynamic_high_quad", dyn_high_dxil, ps_high_dxil, 0x2004, expected_pixels(0x2004, "linear_flat")),
            ("static_high_quad", stat_high_dxil, ps_high_dxil, 0x2004, expected_pixels(0x2004, "linear_flat")),
        )
        for name, ms, ps, counts, pixels in draws:
            case = directory / f"{name}.bin"
            case.write_bytes(fixture(counts, SECOND, pixels))
            run = subprocess.run([renderer, adapter, "--mesh-indexed", ms, ps, case],
                                 capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout, run.stderr)
                return 77
            if run.returncode:
                raise AssertionError(f"{name}:\n{run.stdout}\n{run.stderr}")
            print(f"{adapter} {name}: pass ({[l for l in run.stdout.splitlines() if 'mismatches=' in l][0]})",
                  flush=True)

        # GPU negative controls: swapping param0<->param12 or corrupting fparam5 must fail readback.
        quad_case = directory / "dynamic_lf_quad.bin"
        swapped = dyn_lf_hlsl.replace(".param0 = ", ".paramX = ").replace(".param12 = ", ".param0 = ").replace(
            ".paramX = ", ".param12 = ")
        assert swapped != dyn_lf_hlsl
        bad_flat = dyn_lf_hlsl.replace(".fparam5 = asuint(p5);", ".fparam5 = uint4(0, 0, 0, 0);")
        assert bad_flat != dyn_lf_hlsl
        swapped_high = dyn_high_hlsl.replace(".param0 = ", ".paramX = ").replace(".param18 = ", ".param0 = ").replace(
            ".paramX = ", ".param18 = ")
        assert swapped_high != dyn_high_hlsl
        zeroed_28 = dyn_high_hlsl.replace(".param28 = ", ".param28 = float4(0, 0, 0, 0); //")
        assert zeroed_28 != dyn_high_hlsl
        for label, bad_hlsl, want_mismatches, ps_bad in (
            ("swapped_param0_param12", swapped, "pixels=1024 mismatches=992", ps_lf_dxil),
            ("zeroed_flat_fparam5", bad_flat, "pixels=1024 mismatches=1024", ps_lf_dxil),
            ("swapped_param0_param18", swapped_high, "pixels=1024 mismatches=992", ps_high_dxil),
            ("zeroed_param28", zeroed_28, "pixels=1024 mismatches=1024", ps_high_dxil),
        ):
            bad_ms = compile_shader(dxc, directory, label, bad_hlsl, "ms_6_5")
            wrong = subprocess.run([renderer, adapter, "--mesh-indexed", bad_ms, ps_bad, quad_case],
                                   capture_output=True, text=True)
            if wrong.returncode != 1 or want_mismatches not in wrong.stdout.splitlines() or "Draw reference mismatch" not in wrong.stderr:
                raise AssertionError(f"{label} did not fail as expected:\n{wrong.stdout}\n{wrong.stderr}")
            print(f"{adapter} control {label}: fails as expected ({want_mismatches})", flush=True)
    return 0


def main():
    if len(sys.argv) == 1:
        controls()
        return 0
    mode, *args = sys.argv[1:]
    if mode == "binary":
        check_binary(*args)
        return 0
    if mode == "draw":
        return check_draw(*args)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
