"""Build a differential probe for gfx1030 2D gathers with instruction bias."""
import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

from pixel_compare_gather_check import FEATURES, depth
import pixel_gather_check as gather


SIDE = 8
MIPS = 3
PHASE = Fraction(5, 128)
REFERENCE = 0.5
TAPS = ((0, 1), (1, 1), (1, 0), (0, 0))
OUT = Path(__file__).resolve().parents[1] / "out" / "gfx1030-gather-bias-probe"


@dataclass(frozen=True)
class Form:
    suffix: str
    comparison: bool
    offset: tuple[int, int]
    encoding: str

    @property
    def opcode(self):
        return "image_gather4" + self.suffix

    @property
    def component(self):
        return 0 if self.comparison else 2

    @property
    def sampler(self):
        word0, *rest = gather.sampler(MIPS)
        return (word0 | (1 << 12 if self.comparison else 0), *rest)  # LESS


FORMS = (
    Form("_b", False, (0, 0), "080414f1060c4701"),
    Form("_b_o", False, (-1, 1), "080454f1050c4701"),
    Form("_c_b", True, (0, 0), "080134f1060c4701"),
    Form("_c_b_o", True, (-1, 1), "080174f1050c4701"),
)


@dataclass(frozen=True)
class Case:
    name: str
    scale: tuple[float, float]
    bias: int
    mip: int


CASES = (
    Case("positive", (1, 2), 1, 2),
    Case("negative", (2, 1), -1, 0),
    Case("magnify", (0.5, 0.25), 2, 1),
)


def stem(form, case):
    return f"{form.opcode}-{case.name}"


def native_program(form, case, *, explicit=False):
    # Contiguous address roles follow ISA 8.2.5. The compare and body shift down
    # by one for _l, which puts its explicit LOD after the body.
    first = 5 if any(form.offset) else 6
    regs = iter(range(first, 10))
    offset = next(regs) if any(form.offset) else None
    level = None if explicit else next(regs)
    reference = next(regs) if form.comparison else None
    u, v = next(regs), next(regs)
    if explicit:
        level = next(regs)
    last = level if explicit else v
    opcode = form.opcode.replace("_b", "_l") if explicit else form.opcode
    instruction = (f"{opcode} v[12:15], v[{first}:{last}], s[28:35], s[40:43] "
                   f"dmask:0x{1 << form.component:x} dim:SQ_RSRC_IMG_2D")
    lines = ["s_load_dwordx8 s[28:35], s[20:21], null", "s_waitcnt lgkmcnt(0)",
             *(f"s_mov_b32 s{40+i}, {word:#x}" for i, word in enumerate(form.sampler)),
             "s_mov_b64 s[50:51], exec", "s_wqm_b64 exec, exec", "s_mov_b32 m0, s24",
             f"v_mov_b32_e32 v11, {float(PHASE)}"]
    for dest, axis, scale in zip((u, v), "xy", case.scale):
        lines += [f"v_interp_p1_f32_e32 v{dest}, v0, attr0.{axis}",
                  f"v_interp_p2_f32_e32 v{dest}, v1, attr0.{axis}",
                  f"v_fma_f32 v{dest}, {float(scale)}, v{dest}, v11"]
    if offset is not None:
        x, y = form.offset
        lines.append(f"v_mov_b32_e32 v{offset}, {((x & 63) | ((y & 63) << 8)):#x}")
    if reference is not None:
        lines.append(f"v_mov_b32_e32 v{reference}, {REFERENCE}")
    lines += [f"v_mov_b32_e32 v{level}, {float(case.mip if explicit else case.bias)}", instruction,
              "s_waitcnt vmcnt(0)", "s_and_b64 exec, exec, s[50:51]",
              "exp mrt0 v12, v13, v14, v15 done vm", "s_endpgm"]
    return lines, instruction


def coordinates(case, x, y):
    return tuple(Fraction(2 * p + 1, 2 * SIDE) * Fraction(scale) + PHASE
                 for p, scale in zip((x, y), case.scale))


def texel(form, mip, x, y, component):
    if form.comparison and component == 0:
        return depth(mip, x, y)
    return gather.texel(mip, x, y, component)


def pixels(form, case, mip, *, offset=None, reverse_compare=False):
    extent = SIDE >> mip
    offset = form.offset if offset is None else offset

    def clamp(i):
        return min(extent - 1, max(0, i))

    result = []
    for y in range(SIDE):
        for x in range(SIDE):
            u, v = coordinates(case, x, y)
            left = math.floor(u * extent - Fraction(1, 2)) + offset[0]
            top = math.floor(v * extent - Fraction(1, 2)) + offset[1]
            taps = tuple(float(texel(form, mip, clamp(left + dx), clamp(top + dy), form.component))
                         for dx, dy in TAPS)
            if form.comparison:
                taps = tuple(float(d < REFERENCE if reverse_compare else REFERENCE < d) for d in taps)
            result.extend(taps)
    return tuple(result)


def first_mismatch(expected, observed):
    for index, (wanted, actual) in enumerate(zip(expected, observed, strict=True)):
        if struct.pack("<f", wanted) != struct.pack("<f", actual):
            return index
    return None


def first_difference(expected, observed):
    index = first_mismatch(expected, observed)
    if index is None:
        return None
    y, x = divmod(index // 4, SIDE)
    return f"pixel=({x},{y}) tap={index % 4} expected={expected[index]:g} actual={observed[index]:g}"


def verify_readback(form, case, data):
    words = SIDE * SIDE * 4
    if len(data) != 4 * words:
        raise ValueError(f"Expected {4 * words} bytes of tightly packed row-major RGBA32F")
    actual = struct.unpack(f"<{words}f", data)
    differences = [first_difference(pixels(form, case, mip), actual) for mip in range(MIPS)]
    matches = [i for i, difference in enumerate(differences) if difference is None]
    return matches, differences[case.mip]


def cpu_check(form, case):
    # Check the proposed derivative rule from actual neighbors of every 2x2
    # quad. The expected mip is independently specified in CASES.
    for y in range(0, SIDE, 2):
        for x in range(0, SIDE, 2):
            tl, tr, bl, br = (coordinates(case, x + dx, y + dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)))
            dh = tuple((right - left) * SIDE for left, right in zip(tl, tr))
            dv = tuple((bottom - top) * SIDE for top, bottom in zip(tl, bl))
            assert dh == tuple((right - left) * SIDE for left, right in zip(bl, br))
            assert dv == tuple((bottom - top) * SIDE for top, bottom in zip(tr, br))
            assert dh[1] == dv[0] == 0 and min(dh[0], dv[1]) > 0
            lod = math.log2(max(math.hypot(*dh), math.hypot(*dv)))
            assert min(MIPS - 1, max(0, math.floor(lod + case.bias + 0.5))) == case.mip

    # Keep every sampled coordinate away from a half-texel footprint boundary.
    for p in range(SIDE):
        for value in coordinates(case, p, p):
            for mip in range(MIPS):
                footprint = value * (SIDE >> mip) - Fraction(1, 2)
                assert abs(footprint - round(footprint)) >= Fraction(1, 64)

    variants = [pixels(form, case, mip) for mip in range(MIPS)]
    assert len(set(variants)) == MIPS, stem(form, case)
    expected = variants[case.mip]
    raw = struct.pack(f"<{len(expected)}f", *expected)
    assert verify_readback(form, case, raw) == ([case.mip], None)
    # Ignoring bias gives mip 1 in both minification cases. Clamping the base
    # LOD before adding bias gives mip 2 for magnification instead of mip 1.
    wrong_mip = 2 if case.name == "magnify" else 1
    difference = first_difference(expected, variants[wrong_mip])
    assert difference
    print(f"{stem(form, case)} wrong mip {wrong_mip}: {difference}")
    if case.name == "positive":
        # The horizontal-only mutant selects the same wrong mip as dropping bias.
        print(f"{stem(form, case)} horizontal slope only: {difference}")
    if case.name == "negative":
        difference = first_difference(expected, variants[2])
        assert difference
        print(f"{stem(form, case)} lost bias sign: {difference}")
    if any(form.offset):
        difference = first_difference(expected, pixels(form, case, case.mip, offset=(63, form.offset[1])))
        assert difference
        print(f"{stem(form, case)} unsigned x offset: {difference}")
    if form.comparison:
        difference = first_difference(expected, pixels(form, case, case.mip, reverse_compare=True))
        assert difference
        print(f"{stem(form, case)} reverse comparison: {difference}")
    return expected


def desktop_source(form, case, *, mutant=False):
    # This standalone shader exercises the candidate only on the full-screen
    # probe's affine coordinates and fixed 8x8 view. It has no sampler LOD query.
    resources = gather.aot.gather_compare_resources(1) if form.comparison else gather.aot.GATHER_RESOURCES
    function = "gather_compare" if form.comparison else "gather_2d"
    operand = "0.5f" if form.comparison else f"{form.component}u"
    sx, sy = case.scale
    ox, oy = form.offset
    lod = "max(0.0f, log2(rho))" if mutant and case.name == "magnify" else "log2(rho)"
    bias = 0 if mutant and case.name != "magnify" else case.bias
    return resources + f"""
float4 main(float2 entry_uv : TEXCOORD0) : SV_Target0 {{
    precise float2 uv = entry_uv * float2({float(sx)}, {float(sy)}) + {float(PHASE)}f;
    float2 dx = ddx_fine(uv) * 8.0f;
    float2 dy = ddy_fine(uv) * 8.0f;
    float rho = max(length(dx), length(dy));
    float biased = {lod} + {float(bias)}f;
    int mip = int(floor(clamp(biased, 0.0f, 2.0f) + 0.5f));
    return {function}(uv, int2({ox}, {oy}), mip, {operand});
}}
"""


def assemble(source, instruction, assembler, objcopy, library, encoding=None):
    obj = source.with_suffix(".o")
    raw = source.with_suffix(".bin")
    subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1030", f"-mattr={FEATURES}",
                    "-filetype=obj", source, "-o", obj], check=True)
    subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
    code = raw.read_bytes()
    decoded = gather.aot.disassemble(code, library, cpu=b"gfx1030", features=FEATURES.encode(), offsets=True)
    assert decoded[-1][1] == "s_endpgm"
    images = [(offset, line) for offset, line in decoded if line.startswith("image_")]
    assert len(images) == 1 and images[0][1] == instruction, images
    if encoding is not None:
        start = images[0][0]
        assert code[start:start + 8].hex() == encoding, images


def emit(form, case, expected, llvm, dxc, draw):
    name = stem(form, case)
    for explicit in (False, True):
        source = OUT / f"{name}{'-explicit' if explicit else ''}.s"
        lines, instruction = native_program(form, case, explicit=explicit)
        source.write_text("\n".join(lines) + "\n", encoding="ascii")
        if llvm:
            assemble(source, instruction, *llvm, encoding=None if explicit else form.encoding)

    source = OUT / f"{name}.hlsl"
    source.write_text(desktop_source(form, case), encoding="ascii")
    if dxc:
        subprocess.run([dxc, "-T", "ps_6_0", "-E", "main", "-Gis", "-Fo", source.with_suffix(".dxil"), source],
                       check=True)

    # Existing desktop gather case format, binding the three supplied levels
    # at view mip zero. Also emit each level as a standalone native upload.
    data = bytearray(struct.pack("<II", 0, MIPS))
    for mip in range(MIPS):
        extent = SIDE >> mip
        level = b"".join(struct.pack("<4f", *(texel(form, mip, x, y, c) for c in range(4)))
                         for y in range(extent) for x in range(extent))
        (OUT / f"{name}.mip{mip}.rgba32f").write_bytes(level)
        data += level
    raw = struct.pack(f"<{len(expected)}f", *expected)
    (OUT / f"{name}.expected.rgba32f").write_bytes(raw)
    draw_case = OUT / f"{name}.case"
    draw_case.write_bytes(data + raw)
    if draw:
        renderer, backend = draw
        shader = source.with_suffix(".dxil")
        command = [renderer, backend, "--check-pixel-gather"]
        result = subprocess.run([*command, shader, draw_case])
        if result.returncode == 77:
            sys.exit(77)
        result.check_returncode()

        mutation = OUT / f"{name}-mutated.hlsl"
        mutation.write_text(desktop_source(form, case, mutant=True), encoding="ascii")
        mutant_shader = mutation.with_suffix(".dxil")
        subprocess.run([dxc, "-T", "ps_6_0", "-E", "main", "-Gis", "-Fo", mutant_shader, mutation], check=True)
        wrong_mip = 2 if case.name == "magnify" else 1
        wrong = pixels(form, case, wrong_mip)
        wrong_case = OUT / f"{name}-mutated.case"
        wrong_case.write_bytes(data + struct.pack(f"<{len(wrong)}f", *wrong))
        subprocess.run([*command, mutant_shader, wrong_case], check=True)
        result = subprocess.run([*command, mutant_shader, draw_case], capture_output=True, text=True)
        output = result.stdout + result.stderr
        index = first_mismatch(expected, wrong)
        assert index is not None
        y, x = divmod(index // 4, SIDE)
        assert result.returncode == 1 and f"pixel_mismatch x={x} y={y}" in output, output
        print(f"{name} {backend} negative control\n{output}", flush=True)
    return {"stem": name, "scale": case.scale, "phase": float(PHASE), "bias": case.bias, "mip": case.mip,
            "sampler": [f"0x{word:08x}" for word in form.sampler], "offset": form.offset,
            "dmask": 1 << form.component}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="write source and inputs below out/")
    parser.add_argument("--binary", nargs=3, metavar=("LLVM_MC", "LLVM_OBJCOPY", "LIBLLVM"))
    parser.add_argument("--dxc", metavar="DXC", help="compile standalone desktop probe shaders offline")
    parser.add_argument("--draw", nargs=2, metavar=("OFFSCREEN_CHECK", "BACKEND"),
                        help="render the desktop probe and its wrong-mip control")
    parser.add_argument("--readback", nargs=2, metavar=("CASE_STEM", "RGBA32F"), help="identify mip and first mismatch")
    args = parser.parse_args()
    fixtures = {stem(form, case): (form, case) for form in FORMS for case in CASES}
    if args.draw and (not args.dxc or args.draw[1] not in ("warp", "hardware")):
        parser.error("--draw requires --dxc and a warp or hardware backend")
    if args.readback:
        name, path = args.readback
        if name not in fixtures:
            parser.error(f"unknown case stem {name}")
        if args.emit or args.binary or args.dxc:
            parser.error("--readback cannot be combined with generation options")
        try:
            matches, difference = verify_readback(*fixtures[name], Path(path).read_bytes())
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(f"{name} matching mips={matches}")
        if difference:
            print(difference)
        return int(difference is not None)

    write_files = args.emit or args.binary or args.dxc
    if write_files:
        OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for form, case in fixtures.values():
        expected = cpu_check(form, case)
        if write_files:
            manifest.append(emit(form, case, expected, args.binary, args.dxc, args.draw))
    if write_files:
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(f"CPU footprint and negative controls passed for {len(fixtures)} biased gather cases")
    if args.binary:
        print(f"LLVM round-tripped {2 * len(fixtures)} native shaders and checked the four biased encodings")
    if args.dxc:
        print(f"DXC compiled {len(fixtures)} standalone desktop shaders")
    if args.draw:
        print(f"{args.draw[1]} matched {len(fixtures)} desktop draws and their wrong-mip controls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
