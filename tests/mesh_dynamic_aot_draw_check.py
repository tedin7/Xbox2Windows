"""Run the generated runtime mesh shader through the existing D3D12 oracle."""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from mesh_dynamic_geometry_check import CASES, CODE, MESH_LIMITS, RETIRED_PREFIX, SECOND, decode
from mesh_dynamic_geometry_check import WIDE_COUNTS, WIDE_SECOND, surviving_wave_code
from mesh_dynamic_aot import lower
from mesh_entry_aot import lower as lower_static


SIDE = 32
COLOR = b"\xff\x00\xff\xff"
CLEAR = b"\x00\x00\x00\xff"


def compile_shader(dxc, directory, stem, text, profile):
    source = directory / (stem + ".hlsl")
    output = directory / (stem + ".dxil")
    source.write_text(text, encoding="ascii")
    subprocess.run([dxc, "-WX", "-T", profile, "-E", "main", "-Fo", output, source], check=True)
    return output


def fixture(counts, geometry, indices):
    data = bytearray(1536)
    struct.pack_into("<II", data, 1024, counts, geometry)
    data.extend(b"\xcd" * 80)  # The unmodified shader has no trace UAV.
    full = (2, 1, 3) in indices or (2, 1, 67) in indices
    triangle = (0, 1, 2) in indices
    pixels = (COLOR if full or (triangle and y > x) else CLEAR
              for y in range(SIDE) for x in range(SIDE))
    data.extend(b"".join(pixels))
    return data


def check(dxc, runner, backend, target):
    pixel = compile_shader(dxc, target, "solid", "float4 main() : SV_Target { return float4(1,0,1,1); }", "ps_6_0")
    mesh = compile_shader(dxc, target, "dynamic", lower(decode(CODE), 2, MESH_LIMITS), "ms_6_5")
    old_key = struct.pack("<I", 0x300402)
    assert CODE.count(old_key) == 1
    alternate_code = CODE.replace(old_key, struct.pack("<I", 0x300401))
    alternate = compile_shader(dxc, target, "changed", lower(decode(alternate_code), 2, MESH_LIMITS), "ms_6_5")
    cases = [(mesh, *case) for case in CASES]
    retired_program = decode(RETIRED_PREFIX + CODE)
    retired = compile_shader(dxc, target, "retired", lower(retired_program, 2, MESH_LIMITS), "ms_6_5")
    # One triangle and two triangles take distinct runtime geometry paths after the early exit.
    cases.extend((retired, "retired wave: " + name, *values) for name, *values in CASES[:2])
    static = compile_shader(dxc, target, "retired-static",
                            lower_static(retired_program, [0x2004, SECOND], MESH_LIMITS), "ms_6_5")
    cases.append((static, "retired wave: specialized quad", *CASES[1][1:]))
    wide = decode(surviving_wave_code())
    wide_static = lower_static(wide, [WIDE_COUNTS, WIDE_SECOND], (68, 2))
    for name, source in (("static", wide_static), ("runtime", lower(wide, 2, (68, 2)))):
        shader = compile_shader(dxc, target, "wide-" + name, source, "ms_6_5")
        cases.append((shader, "surviving wave 1: " + name, WIDE_COUNTS, WIDE_SECOND, 68, ((0, 1, 2), (2, 1, 67))))
    cases.extend([
        (alternate, "changed source rejects old input", 0x2004, 0x300402, 0, ()),
        (alternate, "changed source accepts new input", 0x2004, 0x300401, 4, ((0, 1, 2), (1, 1, 3))),
    ])
    for index, (shader, name, counts, geometry, _, indices) in enumerate(cases):
        inputs = target / f"case-{index}.bin"
        inputs.write_bytes(fixture(counts, geometry, indices))
        print(name, flush=True)
        result = subprocess.run([runner, backend, "--mesh-indexed", shader, pixel, inputs])
        if result.returncode:
            return result.returncode
    wrong_case = target / "quad-with-triangle-reference.bin"
    wrong_case.write_bytes(fixture(0x2004, SECOND, ((0, 1, 2),)))
    # Give vertex 67 the position of vertex 2. The second triangle collapses
    # unless the surviving wave writes its distinct top-right corner.
    write = "v[67].position=q;"
    assert wide_static.count(write) == 1
    stale = compile_shader(dxc, target, "wide-stale-corner",
                           wide_static.replace(write, "v[67].position=float4(-1,1,0,1);"), "ms_6_5")
    wide_reference = target / "wide-quad.bin"
    wide_reference.write_bytes(fixture(WIDE_COUNTS, WIDE_SECOND, ((0, 1, 2), (2, 1, 67))))
    for shader, reference in ((retired, wrong_case), (stale, wide_reference)):
        wrong = subprocess.run([runner, backend, "--mesh-indexed", shader, pixel, reference], capture_output=True, text=True)
        if (wrong.returncode != 1 or "Draw reference mismatch" not in wrong.stderr
                or "pixels=1024 mismatches=528" not in wrong.stdout.splitlines()):
            raise AssertionError(wrong.stdout + wrong.stderr)
    print("Both retirement orders detected the incorrect corner or reference at 528 pixels")
    return 0


if __name__ == "__main__":
    output = Path("out").resolve()
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mesh-dynamic-native-", dir=output) as target:
        sys.exit(check(*sys.argv[1:], Path(target)))
