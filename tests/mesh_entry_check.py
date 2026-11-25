"""Translate a source-built mesh triangle and compare exports and rasterization."""

from pathlib import Path
import json
import math
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mesh_entry_aot import UNDEF, Wave, lower

# Two explicitly modeled wave64s. Only the first three group threads export vertices.
PROGRAM = """
s_mov_b32 s10, s17
s_mov_b32 s11, 0
s_load_dwordx4 s[24:27], s[10:11], 0x20
s_load_dwordx4 s[16:19], s[10:11], 0x60
s_load_dwordx4 s[20:23], s[10:11], 0x80
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s15, s[24:27], null
s_buffer_load_dword s30, s[16:19], 0x4
s_buffer_load_dword s31, s[20:23], 0xc
s_waitcnt lgkmcnt(0)
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_bfe_u32 s8, s3, 0x40018
v_lshl_add_u32 v0, s8, 6, v0
s_mov_b64 s[22:23], exec
v_cmpx_eq_u32_e32 0, v0
s_cbranch_execz 2
s_mov_b32 s7, s15
s_mov_b32 m0, s7
s_cbranch_execz 1
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
v_mov_b32_e32 v9, 0x200400
exp prim, v9, off, off, off done
s_mov_b64 exec, s[22:23]
v_cmpx_gt_u32_e32 3, v0
v_cmp_eq_u32_e64 s[10:11], 1, v0
v_cmp_eq_u32_e64 s[12:13], 2, v0
v_mov_b32_e32 v8, 0x40400000
v_mov_b32_e32 v9, 0xbf800000
v_cndmask_b32_e64 v1, v9, v8, s[10:11]
v_cndmask_b32_e64 v2, v9, v8, s[12:13]
v_mov_b32_e32 v3, s30
v_add_f32_e32 v3, s31, v3
v_mov_b32_e32 v4, 0x3f800000
exp pos0, v1, v2, v3, v4 done
s_endpgm
""".strip().splitlines()
TRIANGLE = [(-1, -1, 0.5, 1), (3, -1, 0.5, 1), (-1, 3, 0.5, 1)]
PIXEL = "float4 main() : SV_Target { return float4(0,1,0,1); }"


def fixture(constants, positions, visible, traced):
    trace = bytearray([0xCD] * 80)
    if traced:
        struct.pack_into("<2I", trace, 0, 3, 1)
        for i, vertex in enumerate(positions):
            struct.pack_into("<4f", trace, 16 + i * 16, *vertex)
        struct.pack_into("<3I", trace, 64, 0, 1, 2)
    pixels = (
        visible
        if isinstance(visible, bytes)
        else bytes([0, 255 if visible else 0, 0, 255]) * 1024
    )
    return constants + trace + pixels


def draw(dxc, renderer, adapter, tmp, source, cases, *, indexed=False, pixel_source=PIXEL):
    pixel = tmp / "pixel.hlsl"
    pixel.write_text(pixel_source)
    ps = tmp / "pixel.dxil"
    subprocess.run([dxc, "-T", "ps_6_0", "-E", "main", "-Fo", ps, pixel], check=True)
    shader = tmp / "mesh.hlsl"
    shader.write_text(source)
    for traced in (False, True):
        ms = tmp / "mesh.dxil"
        flags = ["-DTRACE_EXPORTS"] if traced else []
        subprocess.run(
            [dxc, "-T", "ms_6_5", "-E", "main", *flags, "-Fo", ms, shader], check=True
        )
        for constants, positions, visible in cases:
            path = tmp / "draw.bin"
            data = fixture(constants, positions, visible, traced)
            path.write_bytes(data)
            args = [
                renderer,
                adapter,
                "--mesh-indexed" if indexed else "--mesh-entry",
                ms,
                ps,
                path,
            ]
            result = subprocess.run(args)
            if result.returncode == 77:
                return 77
            result.check_returncode()
            if traced:
                bad = bytearray(data)
                bad[len(constants) + 64] ^= 1
                path.write_bytes(bad)
                result = subprocess.run(args, capture_output=True, text=True)
                if (
                    result.returncode != 1
                    or "Mesh export trace mismatch" not in result.stderr
                ):
                    raise AssertionError(
                        "Unexpected result for a wrong primitive index"
                    )
    return 0


def triangle_inputs():
    inputs = bytearray(1024)
    struct.pack_into("<f", inputs, 4, 0.125)
    struct.pack_into("<f", inputs, 512 + 12, 0.375)
    return inputs


def enumerate_pcs(program):
    # All fixture branch displacements span only scalar four-byte instructions.
    return [(i * 4, line) for i, line in enumerate(program)]


def source():
    # Branches only span scalar instructions of known 4-byte length.
    decoded = enumerate_pcs(PROGRAM)
    for bad in ((2, 1), (3, 0)):
        try:
            lower(decoded, [0x1003], bad)
        except ValueError:
            continue
        raise AssertionError("Output declaration exceeded")
    for change in (
        {"s_waitcnt lgkmcnt(0)": "s_nop 0"},
        {"s_mov_b32 s7, s15": "s_mov_b32 s7, s14"},
        {"s_cbranch_execz 1": "s_branch -1"},
        {
            "s_buffer_load_dword s15, s[24:27], null": "s_buffer_load_dword s15, s[24:27], 0x4"
        },
        {"v_mov_b32_e32 v9, 0x200400": "v_mov_b32_e32 v9, 0x200600"},
        {"exp pos0, v1, v2, v3, v4 done": "exp pos0, v1, v2, v3, v4"},
    ):
        try:
            lower([(pc, change.get(op, op)) for pc, op in decoded], [0x1003], (3, 1))
        except (KeyError, ValueError):
            continue
        raise AssertionError("Invalid mesh path accepted")
    return lower(decoded, [0x1003], (3, 1))


ALU_PROGRAM = PROGRAM.copy()
ALU_PROGRAM[-2:-2] = [
    # Carry out of an add and no borrow from a subtract; multiplies keep SCC.
    "s_add_u32 s5, 1, 0xffffffff",
    "s_mul_i32 s6, s5, 7",
    "s_cselect_b32 s14, 0x3e800000, 0",
    "s_sub_u32 s9, 11, 3",
    "s_mul_i32 s6, s9, 0x07800000",
    "s_cselect_b32 s4, 0, 0x3e000000",
    "v_mov_b32_e32 v11, 0x3f000000",
    "v_or_b32_e32 v11, 0x200000, v11",
    "v_xor_b32_e32 v12, 0x600000, v11",
    "v_min_u32_e32 v13, v11, v12",
    "v_max_u32_e32 v14, v11, v12",
    "v_mov_b32_e32 v16, 0x100000",
    "v_sub_nc_u32_e32 v15, v14, v16",
    # The sign bit makes these differ from signed comparisons.
    "v_max_u32_e32 v17, 0xbfc00000, v13",
    "v_min_u32_e32 v18, 0xbfc00000, v13",
    "v_add_f32_e32 v3, s14, v3",
    "v_add_f32_e32 v3, s4, v3",
    "v_add_f32_e32 v3, s6, v3",
    "v_add_f32_e32 v3, v15, v3",
    # Unequal use counts: a signed min/max would swap the two results and change the depth.
    "v_add_f32_e32 v3, v18, v3",
    "v_add_f32_e32 v3, v18, v3",
    "v_add_f32_e32 v3, v18, v3",
    "v_add_f32_e32 v3, v17, v3",
    "v_add_f32_e32 v3, v17, v3",
]
# Operands of the float adds above, as bit patterns.
ALU_OPERANDS = (0x3e800000, 0x3e000000, 0x3c000000, 0x3f300000, 0x3f200000, 0xbfc00000)


def alu_draw():
    # 0.5 + 0.25 + 0.125 + 2**-7 + 0.6875 + 3 * 0.625 - 2 * 1.5, all exact in binary32. Swapping the unsigned
    # min and max results (a signed comparison) would give 0.5 + ... + 3 * -1.5 + 2 * 0.625, which is -3.2578125.
    depth = 0.4453125
    positions = [(x, y, depth, w) for x, y, _, w in TRIANGLE]
    return (
        lower(enumerate_pcs(ALU_PROGRAM), [4099], (3, 1)),
        [(triangle_inputs(), positions, True)],
    )


FLOAT_PROGRAM = PROGRAM.copy()
FLOAT_PROGRAM[-2:-2] = [
    "v_mov_b32_e32 v10, 0x40800000",
    "v_mov_b32_e32 v11, 0x3fc00000",
    "v_mov_b32_e32 v12, 0x3ec00000",
    "v_mov_b32_e32 v13, 0x3e000000",
    "v_mov_b32_e32 v14, 0x3d800000",
    "v_mov_b32_e32 v15, 0x3e800000",
    "v_mov_b32_e32 v16, 0x3f000000",
    "v_sub_f32_e32 v3, v10, v3",
    "v_subrev_f32_e32 v3, v11, v3",
    "v_mul_f32_e32 v3, v12, v3",
    "v_add_f32_e64 v3, v3, v13",
    "v_sub_f32_e64 v3, v3, v14",
    "v_subrev_f32_e64 v3, v15, v3",
    "v_mul_f32_e64 v3, v3, v16",
]
# Each result below is exact in binary32; the sibling opcode gives a different depth.
SIBLINGS = {
    "v_sub_f32_e32": "v_subrev_f32_e32",
    "v_subrev_f32_e32": "v_sub_f32_e32",
    "v_mul_f32_e32": "v_add_f32_e32",
    "v_add_f32_e64": "v_sub_f32_e64",
    "v_sub_f32_e64": "v_subrev_f32_e64",
    "v_subrev_f32_e64": "v_sub_f32_e64",
    "v_mul_f32_e64": "v_add_f32_e64",
}


def float_reference():
    # Independent of the lowering: plain arithmetic in the order the program runs.
    z = 0.125 + 0.375  # v3 = pointData y, then plus the matrix value
    z = 4.0 - z  # v_sub_f32_e32 v3, v10, v3
    z = z - 1.5  # v_subrev_f32_e32 v3, v11, v3 is v3 - v11
    z = 0.375 * z  # v_mul_f32_e32
    z = z + 0.125  # v_add_f32_e64
    z = z - 0.0625  # v_sub_f32_e64
    z = z - 0.25  # v_subrev_f32_e64 v3, v15, v3 is v3 - v15
    return z * 0.5  # v_mul_f32_e64


class F32(float):
    def __new__(cls, x):
        return super().__new__(cls, struct.unpack("<f", struct.pack("<f", x))[0])

    def __add__(self, o):
        return F32(float(self) + float(o))

    def __sub__(self, o):
        return F32(float(self) - float(o))

    def __mul__(self, o):
        return F32(float(self) * float(o))


# Host models of the HLSL intrinsics the unary lowering emits, on exact inputs only. round is half-even.
HLSL_MATH = {
    "rcp": lambda x: F32(1 / x),
    "sqrt": lambda x: F32(math.sqrt(x)),
    "rsqrt": lambda x: F32(1 / math.sqrt(x)),
    "floor": lambda x: F32(math.floor(x)),
    "trunc": lambda x: F32(math.trunc(x)),
    "round": lambda x: F32(round(float(x))),
    "frac": lambda x: F32(x - math.floor(x)) if math.isfinite(x) else F32(math.nan),
    # D3D min returns the other operand when one is a NaN.
    "min": lambda a, b: b if math.isnan(a) else a if math.isnan(b) else F32(min(a, b)),
    # The emitted comparison cap: a NaN fails the comparison and passes through.
    "fract32": lambda x: F32(0.99999994) if HLSL_MATH["frac"](x) >= 1.0 else HLSL_MATH["frac"](x),
}


# The emitted cap on a fraction: group 1 is the frac call, group 2 its argument.
FRACT_CAP = r"\((frac\(([^;]*?)\)) >= 1\.0f \? asfloat\(0x3f7fffffu\) : \1\)"


def evaluate_depth(text, inputs, error=0.0):
    # Evaluate the emitted float chain in binary32 and return position component z of vertex 0. A nonzero relative
    # error scales the reciprocal, reciprocal square root and square root results, as an approximate one would.
    math_ops = {
        name: (lambda x, fn=fn: F32(float(fn(x)) * (1 + error)))
        for name, fn in HLSL_MATH.items()
        if name in ("rcp", "rsqrt", "sqrt")
    }
    math_ops = {**HLSL_MATH, **math_ops}

    def word(m):
        return repr(F32(struct.unpack("<f", struct.pack("<I", int(m[1], 16)))[0]))

    def buffer(m):
        base = 0 if m[1] == "point" else 512
        return repr(F32(struct.unpack_from("<f", inputs, base + int(m[2]) * 16 + 4 * "xyzw".index(m[3]))[0]))

    def value(expr):
        # Integer conversions of an exact word, as the lowering writes them.
        expr = re.sub(
            r"(?<!\w)float\(asint\(0x([0-9a-f]+)u\)\)",
            lambda m: repr(F32(int(m[1], 16) - (2**32 if int(m[1], 16) >> 31 else 0))),
            expr,
        )
        expr = re.sub(r"(?<!\w)float\(0x([0-9a-f]+)u\)", lambda m: repr(F32(int(m[1], 16))), expr)
        expr = re.sub(r"^\(1\.0f / (.*)\)$", r"rcp(\1)", expr)
        expr = re.sub(FRACT_CAP, r"fract32(\2)", expr)
        expr = re.sub(r"asfloat\(0x([0-9a-f]+)u\)", word, expr)
        expr = re.sub(r"(point|matrix)Data\[(\d+)\]\.([xyzw])", buffer, expr)
        expr = re.sub(r"(?<![\w.])(-?\d+\.\d+(?:e[+-]?\d+)?)f?", r"F32(\1)", expr)
        return eval(expr, {"F32": F32, "nan": math.nan, **math_ops}, names)

    names = {}
    for name, expr in re.findall(r"precise float (f\d+) = (.*);", text):
        names[name] = value(expr)
    q = re.search(r"if\(id==0\) \{ float4 q=float4\((.*?)\);", text)[1].split(", ")
    return value(q[2])


def float_draw(program=FLOAT_PROGRAM):
    depth = float_reference()
    positions = [(x, y, depth, w) for x, y, _, w in TRIANGLE]
    return lower(enumerate_pcs(program), [4099], (3, 1)), [(triangle_inputs(), positions, True)]


def mismatch(dxc, renderer, adapter, tmp, source, positions):
    # A wrong expected depth word must be reported by the export trace comparison.
    ps, ms = tmp / "pixel.dxil", tmp / "mesh.dxil"
    (tmp / "pixel.hlsl").write_text(PIXEL)
    (tmp / "mesh.hlsl").write_text(source)
    if not ps.exists():
        subprocess.run([dxc, "-T", "ps_6_0", "-E", "main", "-Fo", ps, tmp / "pixel.hlsl"], check=True)
    subprocess.run(
        [dxc, "-T", "ms_6_5", "-E", "main", "-DTRACE_EXPORTS", "-Fo", ms, tmp / "mesh.hlsl"],
        check=True,
    )
    path = tmp / "draw.bin"
    path.write_bytes(fixture(triangle_inputs(), positions, True, True))
    result = subprocess.run(
        [renderer, adapter, "--mesh-entry", ms, ps, path], capture_output=True, text=True
    )
    if result.returncode == 77:
        return 77
    if result.returncode != 1 or "Mesh export trace mismatch" not in result.stderr:
        raise AssertionError("A wrong depth word was not reported")
    return 0


def float_negative_draws(dxc, renderer, adapter, tmp):
    source, cases = float_draw()
    expected = cases[0][1]
    # One unit in the last place of the depth word, and every sibling opcode.
    word = struct.unpack("<I", struct.pack("<f", expected[0][2]))[0]
    off_by_one = struct.unpack("<f", struct.pack("<I", word + 1))[0]
    wrong = [[(x, y, off_by_one, w) for x, y, _, w in TRIANGLE]]
    for op, other in SIBLINGS.items():
        program = [line.replace(op, other, 1) if line.startswith(op) else line for line in FLOAT_PROGRAM]
        wrong.append((float_draw(program)[0], expected))
    for item in wrong:
        text, positions = (source, item) if isinstance(item, list) else item
        if mismatch(dxc, renderer, adapter, tmp, text, positions) == 77:
            return 77
    return 0


CVT_MINMAX_PROGRAM = PROGRAM.copy()
CVT_MINMAX_PROGRAM[-2:-2] = [
    # Each converted value is exact in binary32 and is then clamped by a min or max, so an unsigned or signed
    # reading of the same word, or a min where a max belongs, moves the depth by many orders of magnitude.
    "v_cvt_f32_u32_e32 v10, 0xc0000000",
    "v_cvt_f32_i32_e32 v11, 0xc0000000",
    "v_cvt_f32_u32_e64 v12, 0x80000000",
    "v_cvt_f32_i32_e64 v13, 0x80000000",
    "v_mov_b32_e32 v14, 0x3e000000",
    "v_mov_b32_e32 v15, 0xbd800000",
    "v_min_f32_e32 v16, 0.5, v10",
    "v_max_f32_e32 v17, 0xbe800000, v11",
    "v_min_f32_e64 v18, v12, v14",
    "v_max_f32_e64 v19, v13, v15",
    "v_add_f32_e32 v3, v16, v3",
    "v_add_f32_e32 v3, v17, v3",
    "v_add_f32_e32 v3, v18, v3",
    "v_add_f32_e32 v3, v19, v3",
]
# Swapping any one opcode for its sibling must change the depth.
CVT_MINMAX_SIBLINGS = {
    "v_cvt_f32_u32_e32": "v_cvt_f32_i32_e32",
    "v_cvt_f32_i32_e32": "v_cvt_f32_u32_e32",
    "v_cvt_f32_u32_e64": "v_cvt_f32_i32_e64",
    "v_cvt_f32_i32_e64": "v_cvt_f32_u32_e64",
    "v_min_f32_e32": "v_max_f32_e32",
    "v_max_f32_e32": "v_min_f32_e32",
    "v_min_f32_e64": "v_max_f32_e64",
    "v_max_f32_e64": "v_min_f32_e64",
}


def cvt_minmax_program(swap=None, line=None):
    # The program with one opcode swapped for its sibling, or with one unsupported line appended before the exports.
    program = CVT_MINMAX_PROGRAM.copy()
    if swap:
        program = [
            l.replace(swap, CVT_MINMAX_SIBLINGS[swap], 1) if l.startswith(swap) else l for l in program
        ]
    if line:
        program.insert(-2, line)
    return program


def cvt_minmax_reference():
    # Independent of the lowering: plain arithmetic on exact values, in the order the program runs.
    word = 3 * 2**30  # 0xc0000000 as unsigned; 0xc0000000 - 2**32 as signed
    z = 0.125 + 0.375  # pointData y plus the matrix value
    z += min(0.5, float(word))
    z += max(-0.25, float(word - 2**32))
    z += min(float(2**31), 0.125)
    z += max(float(2**31 - 2**32), -0.0625)
    return z


def cvt_minmax_draw(program=None):
    depth = cvt_minmax_reference()
    positions = [(x, y, depth, w) for x, y, _, w in TRIANGLE]
    program = CVT_MINMAX_PROGRAM if program is None else program
    return lower(enumerate_pcs(program), [4099], (3, 1)), [(triangle_inputs(), positions, True)]


def cvt_minmax_negative_draws(dxc, renderer, adapter, tmp):
    source, cases = cvt_minmax_draw()
    expected = cases[0][1]
    # One unit in the last place of the depth word of the last vertex only, then every sibling opcode.
    word = struct.unpack("<I", struct.pack("<f", expected[2][2]))[0]
    off_by_one = struct.unpack("<f", struct.pack("<I", word + 1))[0]
    wrong = [(source, expected[:2] + [(*expected[2][:2], off_by_one, expected[2][3])])]
    wrong += [(cvt_minmax_draw(cvt_minmax_program(op))[0], expected) for op in CVT_MINMAX_SIBLINGS]
    for text, positions in wrong:
        if mismatch(dxc, renderer, adapter, tmp, text, positions) == 77:
            return 77
    return 0


def cvt_minmax_binary(tmp, tools):
    inputs, reference = triangle_inputs(), cvt_minmax_reference()
    text = binary(tmp, *tools, program=CVT_MINMAX_PROGRAM, words=[4099])
    assert evaluate_depth(text, inputs) == reference, text
    for op in CVT_MINMAX_SIBLINGS:
        mutated = evaluate_depth(binary(tmp, *tools, program=cvt_minmax_program(op), words=[4099]), inputs)
        assert mutated != reference, (op, mutated)
    # Modifiers change the result, and a float expression is not an integer source: all are refused.
    for line in (
        "v_min_f32_e64 v18, -v12, v14",
        "v_max_f32_e64 v19, |v13|, v15",
        "v_min_f32_e64 v18, v12, v14 clamp",
        "v_max_f32_e64 v19, v13, v15 mul:2",
        "v_cvt_f32_i32_e64 v13, 0x80000000 clamp",
        "v_cvt_f32_u32_e32 v10, v3",
    ):
        try:
            lower(enumerate_pcs(cvt_minmax_program(line=line)), [4099], (3, 1))
        except ValueError:
            continue
        raise AssertionError(line)


def binary(tmp, assembler, objcopy, library, *, program=None, words=None, header=None):
    from pixel_aot_check import envelope
    from pixel_aot import _read_named_container

    src, obj, raw = (tmp / name for name in ("fixture.s", "fixture.o", "fixture.bin"))
    lines = PROGRAM if program is None else program
    src.write_text(
        "\n".join(
            s.replace("mrtz,", "mrtz")
            .replace("null,", "null")
            .replace("prim,", "prim")
            .replace("pos0,", "pos0")
            for s in lines
        )
        + "\n"
    )
    subprocess.run(
        [
            assembler,
            "-triple=amdgcn--amdpal",
            "-mcpu=gfx1013",
            "-mattr=+wavefrontsize64,-wavefrontsize32",
            "-filetype=obj",
            src,
            "-o",
            obj,
        ],
        check=True,
    )
    subprocess.run(
        [objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True
    )
    code = raw.read_bytes()
    padded = code + bytes.fromhex("00009fbf") * (4 - (len(code) // 4 + 10) % 4)
    record = envelope(
        padded,
        directory_end=(40 + len(padded), 288 + len(padded)),
        compute_width=128,
        mesh_limits=(3, 1),
    )
    container = tmp / "input.dxbc"
    container.write_bytes(_read_named_container(record, len(b"fixture") + 2))
    translated = tmp / "out.hlsl"
    geometry = tmp / "geometry.json"
    geometry.write_text(json.dumps(words if words is not None else [4099]))
    tool = Path(__file__).resolve().parents[1] / "tools" / "mesh_entry_aot.py"
    command = [
        sys.executable,
        "-B",
        tool,
        container,
        "--llvm",
        library,
        "--output",
        translated,
        *(["--vertex-header", str(header)] if header is not None else []),
        "--geometry-words",
        geometry,
    ]
    subprocess.run(command, check=True)
    expected = (
        source()
        if program is None
        else lower(enumerate_pcs(program), words, (3, 1), header)
    )
    assert translated.read_text() == expected
    if header is not None:
        subprocess.run([*command, "--compute-fallback"], check=True)
        expected = lower(enumerate_pcs(program), words, (3, 1), header, compute_fallback=True)
        assert translated.read_text() == expected
    return translated.read_text()


def float_binary(tmp, tools):
    inputs, reference = triangle_inputs(), float_reference()
    text = binary(tmp, *tools, program=FLOAT_PROGRAM, words=[4099])
    assert evaluate_depth(text, inputs) == reference, text
    for op, other in SIBLINGS.items():
        program = [l.replace(op, other, 1) if l.startswith(op) else l for l in FLOAT_PROGRAM]
        mutated = evaluate_depth(binary(tmp, *tools, program=program, words=[4099]), inputs)
        assert mutated != reference, (op, mutated)
    # Source modifiers and output modifiers change the result, so they are refused.
    for line in (
        "v_add_f32_e64 v3, -v3, v13",
        "v_add_f32_e64 v3, |v3|, v13",
        "v_mul_f32_e64 v3, v3, v16 mul:2",
        "v_sub_f32_e64 v3, v3, v14 clamp",
    ):
        program = [line if l == "v_mul_f32_e64 v3, v3, v16" else l for l in FLOAT_PROGRAM]
        try:
            lower(enumerate_pcs(program), [4099], (3, 1))
        except ValueError:
            continue
        raise AssertionError(line)


def cvt_minmax_binary(tmp, tools):
    inputs, reference = triangle_inputs(), cvt_minmax_reference()
    text = binary(tmp, *tools, program=CVT_MINMAX_PROGRAM, words=[4099])
    assert evaluate_depth(text, inputs) == reference, text
    for op in CVT_MINMAX_SIBLINGS:
        mutated = evaluate_depth(binary(tmp, *tools, program=cvt_minmax_program(op), words=[4099]), inputs)
        assert mutated != reference, (op, mutated)
    # Modifiers change the result, and a float expression is not an integer source: all are refused.
    for line in (
        "v_min_f32_e64 v18, -v12, v14",
        "v_max_f32_e64 v19, |v13|, v15",
        "v_min_f32_e64 v18, v12, v14 clamp",
        "v_max_f32_e64 v19, v13, v15 mul:2",
        "v_cvt_f32_i32_e64 v13, 0x80000000 clamp",
        "v_cvt_f32_u32_e32 v10, v3",
    ):
        try:
            lower(enumerate_pcs(cvt_minmax_program(line=line)), [4099], (3, 1))
        except ValueError:
            continue
        raise AssertionError(line)


# One-source binary32 math. The inputs reach each opcode through a register that holds an added zero the compiler
# cannot prove, so the compiler's own constant folding is not what the draw checks. Reciprocal, reciprocal square
# root and square root are approximate in D3D, so each result is rounded to the nearest integer (add 0.5, floor)
# before it counts: it is an integer for any result within 0.5 of the exact one, and the sibling opcode gives a
# different integer. The rounding opcodes are exact and are compared directly: floor and truncation of negative
# fractions, a tie rounded to even (-2.5 and -0.5; floor and round half away give -3 and -1), a fractional part.
UNARY_CASES = [
    ("v_rcp_f32_e32", 0.25, "v_rsq_f32_e32"),
    ("v_rsq_f32_e32", 0.0625, "v_rcp_f32_e32"),
    ("v_sqrt_f32_e32", 16.0, "v_rsq_f32_e32"),
    ("v_floor_f32_e32", -2.5, "v_trunc_f32_e32"),
    ("v_trunc_f32_e32", -2.5, "v_floor_f32_e32"),
    ("v_rndne_f32_e32", -2.5, "v_floor_f32_e32"),
    ("v_fract_f32_e32", -2.75, "v_floor_f32_e32"),
    ("v_rcp_f32_e64", 0.125, "v_rsq_f32_e64"),
    ("v_rsq_f32_e64", 0.25, "v_rcp_f32_e64"),
    ("v_sqrt_f32_e64", 9.0, "v_rsq_f32_e64"),
    ("v_floor_f32_e64", -7.75, "v_trunc_f32_e64"),
    ("v_trunc_f32_e64", -6.75, "v_floor_f32_e64"),
    ("v_rndne_f32_e64", -0.5, "v_floor_f32_e64"),
    ("v_fract_f32_e64", 5.125, "v_floor_f32_e64"),
]
APPROXIMATE = ("v_rcp", "v_rsq", "v_sqrt")
UNARY_BIAS = -4.25  # brings the sum of the rounded results to a depth inside the clip range


def fword(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def unary_program(swap=None, line=None):
    # v5 is zero, but not a zero the compiler can see. A swap replaces one case's opcode with its sibling.
    program = PROGRAM.copy()
    body = ["v_sub_f32_e32 v5, v3, v3"]
    for i, (op, x, sibling) in enumerate(UNARY_CASES):
        approximate = op.startswith(APPROXIMATE)
        op = sibling if op == swap else op
        out = f"v{20 + i}"
        body += [f"v_add_f32_e32 v{40 + i}, 0x{fword(x):08x}, v5", f"{op} {out}, v{40 + i}"]
        if approximate:
            body += [f"v_add_f32_e32 {out}, 0x3f000000, {out}", f"v_floor_f32_e32 {out}, {out}"]
        body.append(f"v_add_f32_e32 v3, {out}, v3")
    body.append(f"v_add_f32_e32 v3, 0x{fword(UNARY_BIAS):08x}, v3")
    program[-2:-2] = body
    if line:
        program.insert(-2, line)
    return program


def unary_reference():
    # Independent of the lowering: Python arithmetic on exact values, in the order the program runs. Python rounds
    # a tie to even, which is what v_rndne_f32 does. An approximate result counts as its nearest integer.
    exact = {
        "v_rcp": lambda x: math.floor(1 / x + 0.5),
        "v_rsq": lambda x: math.floor(1 / math.sqrt(x) + 0.5),
        "v_sqrt": lambda x: math.floor(math.sqrt(x) + 0.5),
        "v_floor": math.floor,
        "v_trunc": math.trunc,
        "v_rndne": round,
        "v_fract": lambda x: x - math.floor(x),
    }
    z = 0.125 + 0.375  # pointData y plus the matrix value
    for op, x, _ in UNARY_CASES:
        z += exact[op.rsplit("_f32", 1)[0]](x)
    z += UNARY_BIAS
    # The draw keeps depth clipping on, so a final depth outside [0, 1] would clip the whole triangle.
    assert 0.0 <= z <= 1.0, z
    return z


def unary_draw(program=None):
    depth = unary_reference()
    positions = [(x, y, depth, w) for x, y, _, w in TRIANGLE]
    program = unary_program() if program is None else program
    return lower(enumerate_pcs(program), [4099], (3, 1)), [(triangle_inputs(), positions, True)]


def unary_negative_draws(dxc, renderer, adapter, tmp):
    source, cases = unary_draw()
    expected = cases[0][1]
    # One unit in the last place of the depth word of the last vertex only, then every sibling opcode.
    word = fword(expected[2][2])
    off_by_one = struct.unpack("<f", struct.pack("<I", word + 1))[0]
    wrong = [(source, expected[:2] + [(*expected[2][:2], off_by_one, expected[2][3])])]
    wrong += [(unary_draw(unary_program(op))[0], expected) for op, _, _ in UNARY_CASES]
    for text, positions in wrong:
        if mismatch(dxc, renderer, adapter, tmp, text, positions) == 77:
            return 77
    return 0


def unary_binary(tmp, tools):
    inputs, reference = triangle_inputs(), unary_reference()
    text = binary(tmp, *tools, program=unary_program(), words=[4099])
    assert evaluate_depth(text, inputs) == reference, text
    # The three approximate operations may be off by a thousandth either way without moving the depth.
    for error in (-2**-10, 2**-10):
        assert evaluate_depth(text, inputs, error) == reference, (error, text)
    # Each lowering is present, and each sibling opcode changes the depth.
    for hlsl in ("(1.0f / ", "rsqrt(", "sqrt(", "floor(", "trunc(", "round(", "frac("):
        assert hlsl in text, hlsl
    for op, _, _ in UNARY_CASES:
        mutated = evaluate_depth(binary(tmp, *tools, program=unary_program(op), words=[4099]), inputs)
        assert mutated != reference, (op, mutated)
    # Source and output modifiers change the result, so they are refused, as is a clamp.
    for line in (
        "v_floor_f32_e64 v21, -v47",
        "v_sqrt_f32_e64 v21, |v47|",
        "v_rcp_f32_e64 v21, v47 clamp",
        "v_fract_f32_e64 v21, v47 mul:2",
        "v_rndne_f32_e64 v21, v47 div:2",
    ):
        try:
            lower(enumerate_pcs(unary_program(line=line)), [4099], (3, 1))
        except ValueError:
            continue
        raise AssertionError(line)
    # A second source is not a form of these opcodes.
    try:
        lower(enumerate_pcs(unary_program(line="v_floor_f32_e32 v21, v47, v48")), [4099], (3, 1))
    except ValueError:
        return
    raise AssertionError("two-source unary")


# Fract of a tiny negative input. x - floor(x) rounds to 1.0 in binary32, but v_fract_f32 returns at most 0x3f7fffff.
# Both encodings get their own input (-2^-30 and -2^-28), so the two emitted expressions stay distinct. Each result
# is moved by -1.0 first, which makes it exact (-2^-24, or zero when the clamp is missing).
FRACT_TINY = [("v_fract_f32_e32", 0xB0800000), ("v_fract_f32_e64", 0xB1800000)]
# A NaN input must come out as a NaN, in each encoding. The payloads differ so that the expressions stay distinct.
FRACT_NAN = [("v_fract_f32_e32", 0x7FC00000), ("v_fract_f32_e64", 0x7FC00001)]


def fract_program(cases):
    program = PROGRAM.copy()
    body = ["v_sub_f32_e32 v5, v3, v3"]
    for i, (op, word) in enumerate(cases):
        body += [
            f"v_add_f32_e32 v{40 + i}, 0x{word:08x}, v5",
            f"{op} v{20 + i}, v{40 + i}",
            f"v_add_f32_e32 v{20 + i}, 0xbf800000, v{20 + i}",
            f"v_add_f32_e32 v3, v{20 + i}, v3",
        ]
    program[-2:-2] = body
    return program


def fract_reference(cases):
    # The RDNA rule in binary32: fract(x) is x - floor(x) capped at 0x3f7fffff, and a NaN stays a NaN. The bare
    # difference is 1.0 for both tiny inputs.
    top = F32(struct.unpack("<f", struct.pack("<I", 0x3F7FFFFF))[0])
    z = F32(0.125) + F32(0.375)  # pointData y plus the matrix value
    for _, word in cases:
        x = F32(struct.unpack("<f", struct.pack("<I", word))[0])
        if math.isnan(x):
            return F32(math.nan)
        assert F32(x - math.floor(x)) == 1.0
        z = z + (F32(min(F32(x - math.floor(x)), top)) - F32(1.0))
    return z


def drop_cap(text, indexes, keep):
    # Replace the chosen emitted caps by keep(frac call), in place.
    matches = list(re.finditer(FRACT_CAP, text))
    for i in sorted(indexes, reverse=True):
        m = matches[i]
        text = text[: m.start()] + keep(m[1]) + text[m.end() :]
    return text


def fract_tiny_binary(tmp, tools):
    inputs, reference = triangle_inputs(), fract_reference(FRACT_TINY)
    assert reference == 0.5 - 2**-23, reference
    text = binary(tmp, *tools, program=fract_program(FRACT_TINY), words=[4099])
    assert evaluate_depth(text, inputs) == reference, text
    assert len(re.findall(FRACT_CAP, text)) == 2, text
    # Without the cap, either encoding alone moves the depth, and dropping both moves it further.
    for chosen in ([0], [1], [0, 1]):
        mutated = drop_cap(text, chosen, lambda frac: frac)
        assert evaluate_depth(mutated, inputs) != reference, (chosen, mutated)
    # The cap as a min gives the same depth on these finite inputs, so only the NaN check below tells it apart.
    old = drop_cap(text, [0, 1], lambda frac: f"min({frac}, asfloat(0x3f7fffffu))")
    assert evaluate_depth(old, inputs) == reference, old


def fract_nan_binary(tmp, tools):
    inputs = triangle_inputs()
    # Each encoding alone, so that a lost NaN in one cannot hide behind the other.
    for case in FRACT_NAN:
        text = binary(tmp, *tools, program=fract_program([case]), words=[4099])
        assert len(re.findall(FRACT_CAP, text)) == 1, text
        assert math.isnan(evaluate_depth(text, inputs)), (case, text)
        # The cap as a min turns the NaN into the finite 0x3f7fffff, and a bare frac keeps it a NaN.
        old = drop_cap(text, [0], lambda frac: f"min({frac}, asfloat(0x3f7fffffu))")
        assert not math.isnan(evaluate_depth(old, inputs)), (case, old)
        assert math.isnan(evaluate_depth(drop_cap(text, [0], lambda frac: frac), inputs)), (case, text)


def offset_draw():
    program = PROGRAM.copy()
    program[program.index("s_buffer_load_dword s30, s[16:19], 0x4")] = (
        "s_buffer_load_dword s30, s[16:19], 0x104"
    )
    program[-2:-2] = [
        "s_buffer_load_dword s9, s[24:27], 0x404",
        "v_mov_b32_e32 v10, 0x408",
        "buffer_load_dword v10, v10, s[24:27], 0 offen",
        "s_waitcnt vmcnt(0) lgkmcnt(0)",
        "s_cmp_lt_u32 3, 3",
        "s_cselect_b32 s14, 0x3fa00000, 0",
        "v_add_f32_e32 v3, s9, v3",
        "v_add_f32_e32 v3, v10, v3",
        "v_add_f32_e32 v3, s14, v3",
    ]

    point, matrix, scalar, vector = 0.125, 0.375, 0.0625, 0.125
    inputs = triangle_inputs()
    struct.pack_into("<f", inputs, 4, 0.625)
    struct.pack_into("<f", inputs, 260, point)
    # Geometry words 1 and 2 are distinct decoys for high words 257 and 258.
    words = [4099, *struct.unpack("<2I", struct.pack("<2f", 0.875, 1.125))]
    words += [0] * 254
    words.extend(struct.unpack("<2I", struct.pack("<2f", scalar, vector)))
    # The equal operands contribute zero to the reference depth.
    depth = point + matrix + scalar + vector
    positions = [(x, y, depth, w) for x, y, _, w in TRIANGLE]
    return lower(enumerate_pcs(program), words, (3, 1)), [(inputs, positions, True)]


SKIPPED_PROGRAM = PROGRAM.copy()
# s7 is written only inside the range that wave 1 skips (no lane has global thread 0 there), yet it is
# read after the range. Wave 0 sees 0x1003, wave 1 never has a defined s7, and none of its lanes are live.
SKIPPED_PROGRAM[-2:-2] = [
    "s_add_u32 s9, s7, 0x3e7feffd",  # 0x1003 + 0x3e7feffd = 0x3e800000 = 0.25
    "v_add_f32_e32 v3, s9, v3",
]
SKIPPED_DEPTH = 0.5 + 0.25
RESTORE_EXEC = SKIPPED_PROGRAM.index("s_mov_b64 exec, s[22:23]") + 1


def skipped_draw(program=None):
    program = SKIPPED_PROGRAM if program is None else program
    positions = [(x, y, SKIPPED_DEPTH, w) for x, y, _, w in TRIANGLE]
    return lower(enumerate_pcs(program), [0x1003], (3, 1)), [(triangle_inputs(), positions, True)]


def skipped_refusals():
    # Each control lets an undefined scalar reach something the draw can see. All must be refused.
    after_range = SKIPPED_PROGRAM.index("s_add_u32 s9, s7, 0x3e7feffd") + 1
    controls = {
        "live lane reads it": (RESTORE_EXEC, ["v_add_f32_e32 v3, s9, v3"]),
        "live lane selects with it": (SKIPPED_PROGRAM.index("v_mov_b32_e32 v9, 0xbf800000") + 1, ["v_cndmask_b32_e64 v1, v9, v8, s[8:9]"]),
        "branch on it": (after_range, ["s_cmp_eq_u32 s9, 0", "s_cbranch_scc0 0"]),
        "exec from it": (after_range, ["s_and_b64 exec, exec, s[8:9]"]),
        "m0 from it": (after_range, ["s_mov_b32 m0, s9"]),
        "never written anywhere": (after_range, ["s_add_u32 s14, s5, 1", "v_add_f32_e32 v3, s14, v3"]),
    }
    for name, (at, lines) in controls.items():
        program = SKIPPED_PROGRAM.copy()
        program[at:at] = lines
        try:
            lower(enumerate_pcs(program), [0x1003], (3, 1))
        except KeyError:
            # A register no instruction writes is an unknown operand, refused by its number.
            if name != "never written anywhere":
                raise
            continue
        except ValueError as error:
            if all(text not in str(error) for text in ("not written on this wave", "Undefined", "before it is written")):
                raise AssertionError(f"{name}: refused for another reason: {error}")
            continue
        raise AssertionError(f"{name}: accepted")


def skipped_unsupported_refusal():
    # An opcode with implicit EXEC/SCC effects is refused even when an operand is undefined.
    after_range = SKIPPED_PROGRAM.index("s_add_u32 s9, s7, 0x3e7feffd") + 1
    program = SKIPPED_PROGRAM.copy()
    program[after_range:after_range] = ["s_nand_saveexec_b64 s[34:35], s[8:9]"]
    # Wave 1 reaches the instruction with an undefined operand; it must still see the unsupported opcode.
    try:
        Wave(1, [0x1003]).run(enumerate_pcs(program), {})
    except ValueError as error:
        assert "Unsupported scalar instruction s_nand_saveexec_b64" in str(error), error
    else:
        raise AssertionError("Unsupported scalar opcode accepted with an undefined operand")


def skipped_cpu():
    text = skipped_draw()[0]
    # The first range in the program is skipped only by wave 1: the skipped write must not leak into wave 0.
    wave = Wave(1, [0x1003])
    wave.run(enumerate_pcs(SKIPPED_PROGRAM), {})
    assert 7 not in wave.s and wave.s[9] is UNDEF, wave.s
    wave = Wave(0, [0x1003])
    wave.run(enumerate_pcs(SKIPPED_PROGRAM), {})
    assert wave.s[7] == 0x1003 and wave.s[9] == 0x3E800000, wave.s
    # Dropping the contribution changes the exported depth, so the reference is not satisfied by accident.
    plain = lower(enumerate_pcs(PROGRAM), [0x1003], (3, 1))
    assert plain != text
    skipped_refusals()
    skipped_unsupported_refusal()


def skipped_negative_draws(dxc, renderer, adapter, tmp):
    source, cases = skipped_draw()
    expected = cases[0][1]
    plain = lower(enumerate_pcs(PROGRAM), [0x1003], (3, 1))
    word = struct.unpack("<I", struct.pack("<f", SKIPPED_DEPTH))[0]
    off_by_one = struct.unpack("<f", struct.pack("<I", word + 1))[0]
    wrong = [
        (source, [(x, y, off_by_one, w) for x, y, _, w in TRIANGLE]),
        (plain, expected),
    ]
    for text, positions in wrong:
        if mismatch(dxc, renderer, adapter, tmp, text, positions) == 77:
            return 77
    return 0


# Exec restored inside a per-lane range without the plain "copy, then move it back" form (#1998 item C): a copy
# moved between registers, 32-bit halves, a saved mask flipped and joined with the saveexec family, and a restore
# joined from two arms. Every form must give the same mesh as PROGRAM.
EXEC_COPY = "s_mov_b64 s[22:23], exec"
EXEC_BACK = "s_mov_b64 exec, s[22:23]"
EXEC_FIRST = ["s_mov_b64 s[42:43], s[22:23]"]
EXEC_HALVES = ["s_mov_b32 exec_lo, s42", "s_mov_b32 exec_hi, s43"]
EXEC_JOIN = [
    "v_cmp_eq_u32_e64 s[10:11], 1, v0",
    "s_and_saveexec_b64 s[46:47], s[10:11]",
    "s_xor_b64 exec, exec, s[46:47]",
    "s_or_saveexec_b64 s[48:49], s[46:47]",
    "s_cmp_eq_u32 s8, 0",
    "s_cbranch_scc0 2",
    "s_mov_b64 s[50:51], s[46:47]",
    "s_branch 1",
    "s_mov_b64 s[50:51], 0",
    "s_mov_b64 exec, s[50:51]",
]


def exec_restore_program(change=None):
    # The program, with one line of the join replaced (a dict of old line to new line) for the refusal controls.
    program = PROGRAM.copy()
    program[program.index(EXEC_COPY) + 1 : program.index(EXEC_COPY) + 1] = EXEC_FIRST
    at = program.index(EXEC_BACK)
    program[at : at + 1] = EXEC_HALVES + EXEC_JOIN
    return [(change or {}).get(line, line) for line in program]


def exec_restore_draw(program=None):
    program = exec_restore_program() if program is None else program
    return lower(enumerate_pcs(program), [0x1003], (3, 1)), [(triangle_inputs(), TRIANGLE, True)]


# Each change breaks the restore, or uses a form that is not modeled: all are refused.
EXEC_REFUSED = (
    # The mask saved before the flip is not the mask before the range: lane 1 is lost.
    {"s_mov_b64 s[50:51], s[46:47]": "s_mov_b64 s[50:51], s[48:49]"},
    # The restore reads a pair that was never written.
    {"s_mov_b64 s[50:51], s[46:47]": "s_mov_b64 s[50:51], s[52:53]"},
    # Only the high half is restored: lanes 1 to 31 stay off.
    {"s_mov_b32 exec_lo, s42": "s_nop 0"},
    # A 64-bit write to one half, and a saveexec form that is not modeled.
    {"s_mov_b32 exec_lo, s42": "s_mov_b64 exec_lo, s[42:43]"},
    {"s_or_saveexec_b64 s[48:49], s[46:47]": "s_nand_saveexec_b64 s[48:49], s[46:47]"},
    # The wave 0 arm is never taken, so the join gives wave 0 an empty mask.
    {"s_cbranch_scc0 2": "s_cbranch_scc1 2"},
    # The join takes a backward branch.
    {"s_branch 1": "s_branch -1"},
)


# The complemented saveexec forms negate the old exec, not the source: ANDN2 is SRC0 & ~EXEC, ORN2 is SRC0 | ~EXEC.
# With exec 7 (wave 0) and source 2, ANDN2 leaves no lane (the swapped form would leave lanes 0 and 2). A write under
# the new mask must match the same masks built from ordinary scalar instructions, and differ from PROGRAM.
COMPLEMENT_FORMS = {
    "s_andn2_saveexec_b64 s[46:47], s[44:45]": ["s_mov_b64 s[46:47], exec", "s_andn2_b64 exec, s[44:45], exec"],
    "s_orn2_saveexec_b64 s[46:47], s[44:45]": [
        "s_mov_b64 s[46:47], exec", "s_not_b64 s[48:49], exec", "s_or_b64 exec, s[44:45], s[48:49]",
    ],
}


def complement_program(mask_lines):
    program = PROGRAM.copy()
    at = program.index("v_mov_b32_e32 v4, 0x3f800000") + 1
    program[at:at] = ["s_mov_b64 s[44:45], 2", *mask_lines, "v_mov_b32_e32 v4, 0x40000000", "s_mov_b64 exec, s[46:47]"]
    return program


def complement_checks():
    reference = lower(enumerate_pcs(PROGRAM), [0x1003], (3, 1))
    for form, scalar in COMPLEMENT_FORMS.items():
        saved = lower(enumerate_pcs(complement_program([form])), [0x1003], (3, 1))
        plain = lower(enumerate_pcs(complement_program(scalar)), [0x1003], (3, 1))
        assert saved == plain, form
    # ANDN2 leaves no lane under mask 7, so its write never lands.
    andn2 = lower(enumerate_pcs(complement_program(list(COMPLEMENT_FORMS)[:1])), [0x1003], (3, 1))
    assert andn2 == reference
    orn2 = lower(enumerate_pcs(complement_program(list(COMPLEMENT_FORMS)[1:])), [0x1003], (3, 1))
    assert orn2 != reference


def exec_restore_checks():
    reference = lower(enumerate_pcs(PROGRAM), [0x1003], (3, 1))
    assert lower(enumerate_pcs(exec_restore_program()), [0x1003], (3, 1)) == reference
    complement_checks()
    for change in EXEC_REFUSED:
        try:
            lower(enumerate_pcs(exec_restore_program(change)), [0x1003], (3, 1))
        except (KeyError, ValueError):
            continue
        raise AssertionError(f"Invalid exec restore accepted: {change}")


def exec_restore_negative_draws(dxc, renderer, adapter, tmp):
    # A depth word one unit in the last place off in the last vertex must be reported by the export trace.
    source, cases = exec_restore_draw()
    expected = cases[0][1]
    word = struct.unpack("<I", struct.pack("<f", expected[2][2]))[0]
    off_by_one = struct.unpack("<f", struct.pack("<I", word + 1))[0]
    wrong = expected[:2] + [(*expected[2][:2], off_by_one, expected[2][3])]
    return mismatch(dxc, renderer, adapter, tmp, source, wrong)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        if sys.argv[1] == "binary":
            text0 = binary(Path(directory), *sys.argv[2:])
            text = binary(
                Path(directory), *sys.argv[2:], program=ALU_PROGRAM, words=[4099]
            )
            for operand in ALU_OPERANDS:
                assert f"asfloat(0x{operand:08x}u)" in text, hex(operand)
            # The unsigned min is 0x3f200000 and the unsigned max 0xbfc00000; a signed comparison swaps them, and the
            # unequal use counts (three and two) make the swap visible in the text and in the drawn depth.
            assert text.count("asfloat(0x3f200000u)") == 3, text
            assert text.count("asfloat(0xbfc00000u)") == 2, text
            float_binary(Path(directory), sys.argv[2:])
            cvt_minmax_binary(Path(directory), sys.argv[2:])
            # The assembled restore forms must lower to the same text as the plain restore.
            assert binary(Path(directory), *sys.argv[2:], program=exec_restore_program(), words=[4099]) == text0
            unary_binary(Path(directory), sys.argv[2:])
            fract_tiny_binary(Path(directory), sys.argv[2:])
            fract_nan_binary(Path(directory), sys.argv[2:])
            sys.exit(0)
        if sys.argv[1] == "skipped":
            skipped_cpu()
            sys.exit(0)
        if sys.argv[1] == "draw-skipped":
            args = sys.argv[2:]
            sys.exit(
                draw(*args, Path(directory), *skipped_draw())
                or skipped_negative_draws(*args, Path(directory))
            )
        if sys.argv[1] == "exec-restore":
            exec_restore_checks()
            print("Exec restore forms matched the plain restore and the refusals held")
            sys.exit(0)
        if sys.argv[1] == "draw-exec-restore":
            args = sys.argv[2:]
            sys.exit(
                draw(*args, Path(directory), *exec_restore_draw())
                or exec_restore_negative_draws(*args, Path(directory))
            )
        if sys.argv[1] == "draw-cvt-minmax":
            args = sys.argv[2:]
            sys.exit(
                draw(*args, Path(directory), *cvt_minmax_draw())
                or cvt_minmax_negative_draws(*args, Path(directory))
            )
        if sys.argv[1] == "draw-unary":
            args = sys.argv[2:]
            sys.exit(
                draw(*args, Path(directory), *unary_draw())
                or unary_negative_draws(*args, Path(directory))
            )
        if sys.argv[1] == "draw-float":
            args = sys.argv[2:]
            sys.exit(
                draw(*args, Path(directory), *float_draw())
                or float_negative_draws(*args, Path(directory))
            )
        clipped = triangle_inputs()
        struct.pack_into("<f", clipped, 4, -0.625)  # Plus 0.375 gives Z = -0.25.
        clipped_positions = [(x, y, -0.25, w) for x, y, _, w in TRIANGLE]
        # The triangle runs right then up in render-target coordinates, so it
        # is back-facing with the harness's clockwise front-face convention.
        winding_pixel = (
            "float4 main(bool front : SV_IsFrontFace) : SV_Target "
            "{ return float4(0, front ? 0 : 1, 0, 1); }"
        )
        sys.exit(
            draw(
                *sys.argv[1:],
                Path(directory),
                source(),
                [(triangle_inputs(), TRIANGLE, True), (clipped, clipped_positions, False)],
                pixel_source=winding_pixel,
            )
            or draw(*sys.argv[1:], Path(directory), *offset_draw())
            or draw(*sys.argv[1:], Path(directory), *alu_draw())
        )
