"""image_bvh_intersect_ray: the triangle-node (kinds 0..3) translation slice.

The instruction is RDNA 2 MIMG opcode 230. A triangle node is 64 bytes: five float3 vertices
and one ID dword. This check covers the translator, the watertight ray/triangle test and the
four-word return `{t, determinant, v, w}` for triangle return mode 1. The independent model is
tests/gfx1030_bvh_box/bvh_model.py; the dispatch runs the emitted helper on WARP or hardware.
"""
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "startup"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "gfx1030_bvh_box"))
import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_bvh_check as BOX  # noqa: E402
import check as F  # noqa: E402
from gfx1030_bvh_box import bvh_model as B  # noqa: E402

SORT_ENABLE = 1 << 31
RESOURCE_TYPE = 1 << 31
TRIANGLE_MODE = 1 << 24
TRIANGLES = tuple(case for case in F.WITNESSES if case.name.startswith("tri/") and not case.wide)

failures = []


def check(label, ok, detail=""):
    print("%s: %s" % (label, "ok" if ok else "MISMATCH"), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print("  " + detail)


def translate_triangle(node, **kwargs):
    kwargs.setdefault("word3", RESOURCE_TYPE | TRIANGLE_MODE)
    return BOX.translate_program(node=node, **kwargs)


def controls():
    bad = None
    for case in TRIANGLES:
        origin, direction, inverse, extent = case.ray
        actual = tuple(B.intersect(
            bytearray(case.image), B.descriptor(0, sort=True), case.pointer,
            origin=origin, direction=direction, inverse=inverse, extent=extent))
        if actual != case.expected:
            bad = (case.name, actual, case.expected)
            break
    check("independent model matches every triangle fixture", bad is None, str(bad))

    for kind in range(4):
        pipe, hlsl, refusals = translate_triangle(kind)
        check("triangle kind %d translates" % kind, pipe is not None and not refusals, str(refusals))
        if pipe is None:
            continue
        manifest = pipe.manifest()
        check("triangle kind %d manifest names the node allocation" % kind,
              "srv 0 bvh base 0x100000 size 0xb" in manifest, manifest)
        check("triangle kind %d HLSL contains the helper" % kind,
              "uint4 x2w_bvh_intersect_triangle" in hlsl and "x2w_bvh_triangle_vertex" in hlsl
              and "x2w_bvh_shear" in hlsl, "")
        call_line = next((line for line in hlsl.splitlines() if "x2w_bvh_intersect_triangle(r0" in line), "")
        check("triangle kind %d call carries all four descriptor DWORDs" % kind,
              "0x00001000u, 0x80000000u, 0x0000000bu, 0x81000000u" in call_line, call_line)
        check("triangle kind %d call carries the node and four results" % kind,
              "0x0000000%du" % kind in call_line
              and all("v%d = bt" % reg in hlsl for reg in range(32, 36)), call_line)
        check("triangle kind %d call passes origin, direction and inverse from v18..v26" % kind,
              "float3(F(v18), F(v19), F(v20)), float3(F(v21), F(v22), F(v23)), "
              "float3(F(v24), F(v25), F(v26))" in call_line, call_line)

    pipe, hlsl, refusals = translate_triangle(0x00, sort_boxes=False)
    check("sort-disabled triangle translates", pipe is not None and not refusals, str(refusals))
    if pipe is not None:
        call = next((line for line in hlsl.splitlines() if "x2w_bvh_intersect_triangle(r0" in line), "")
        check("sort-disabled triangle call keeps the zero sort word",
              "0x00001000u, 0x00000000u, 0x0000000bu, 0x81000000u" in call, call)

    # One T# can cover both node types, so a box node ignores the triangle return mode bit.
    pipe, hlsl, refusals = translate_triangle(node=0x05, word3=RESOURCE_TYPE | TRIANGLE_MODE)
    check("box node with triangle return mode bit translates", pipe is not None and not refusals, str(refusals))
    if pipe is not None:
        call = next((line for line in hlsl.splitlines() if "x2w_bvh_intersect_box32(r0" in line), "")
        check("box call preserves the ignored return mode bit",
              "0x00001000u, 0x80000000u, 0x0000000bu, 0x81000000u" in call, call)

    helpers = "\n".join(G.BVH_HELPERS)
    check("translated triangle shader embeds the dispatch-tested helpers",
          hlsl is not None and helpers in hlsl, "")

    # Size is a count minus one in 64-byte units. A triangle uses one unit.
    for node, word2, label in ((0x00, 0, "triangle at offset zero in a one-unit span"),
                               (0x08, 1, "triangle in the last unit of a two-unit span")):
        pipe, hlsl, refusals = translate_triangle(node=node, word2=word2)
        check("exact-fit %s translates" % label, pipe is not None and not refusals, str(refusals))
    pipe, hlsl, refusals = translate_triangle(0x08, word2=0)
    check("too-short triangle in the last unit of a one-unit span refuses",
          len(refusals) == 1 and "outside descriptor size" in refusals[0], str(refusals))
    pipe, hlsl, refusals = translate_triangle(0x00, word2=0x03ffffff)
    check("largest decoded extent translates", pipe is not None and not refusals, str(refusals))
    pipe, hlsl, refusals = translate_triangle(0x00, word2=0x04000000)
    check("extent above 4 GiB refuses", len(refusals) == 1 and "above 4 GiB" in refusals[0], str(refusals))

    cases = [
        ("triangle return mode 0", dict(node=0x00, word3=RESOURCE_TYPE), "does not match node kind 0"),
        ("box reserved word 3 bit", dict(node=0x05, word3=RESOURCE_TYPE | 0x100), "does not match node kind 5"),
        ("unknown node kind", dict(node=0x06), "node kind 6"),
        ("node outside size", dict(node=0x20, word2=3), "outside descriptor size"),
        ("absolute base", dict(node=0x00, word0=0, word1=0), "absolute node base"),
        ("loaded descriptor", dict(node=0x00, loaded_descriptor=True), "loaded at run time"),
        ("descriptor type", dict(node=0x00, word3=0x60000000), "descriptor type 6 is not 8"),
        ("box growing", dict(node=0x00, word1=SORT_ENABLE | (2 << 23)), "box growing amount 2"),
        ("reserved word 1", dict(node=0x00, word1=SORT_ENABLE | 0x100), "reserved word 1"),
        ("8-bit return mode extra bit", dict(node=0x00, word3=RESOURCE_TYPE | TRIANGLE_MODE | 0x100),
         "does not match node kind 0"),
        ("64-bit form", dict(node=0x00, opcode="image_bvh64_intersect_ray"), "image_bvh64_intersect_ray"),
        ("A16", dict(node=0x00, modifier="a16"), "modifier a16"),
        ("unaligned box32 node", dict(node=0x0d), "not an aligned 128-byte node"),
    ]
    for label, kwargs, reason in cases:
        if label == "NSA address":
            pipe, hlsl, refusals = None, None, []
            try:
                G.Stage.op_image_bvh_intersect_ray(
                    type("P", (), {"kind": "cs", "const_u32": lambda self, tok: 0})(),
                    ("v[32:35]", "[v0, v2, v4, v6, v8, v10, v12, v14, v16, v18, v20]", "s[12:15]"), [])
            except G.Unsupported as error:
                refusals = [str(error)]
        else:
            pipe, hlsl, refusals = translate_triangle(**kwargs)
        ok = len(refusals) == 1 and reason in refusals[0]
        check("refuses %s" % label, ok, str(refusals))
        if hlsl is not None:
            check("  no partial output for %s" % label,
                  "uint4 bt" not in hlsl and "x2w_bvh_intersect_triangle(r" not in hlsl, "")

    words = (0xffffffff, SORT_ENABLE | 0xff, 0x0b, RESOURCE_TYPE | TRIANGLE_MODE)
    pipe, boundary_hlsl, refusals = translate_triangle(0x10, word0=words[0], word1=words[1])
    check("triangle node ending exactly at the 48-bit boundary translates",
          pipe is not None and not refusals, str(refusals))
    pipe, wrap_hlsl, refusals = translate_triangle(0x20, word0=words[0], word1=words[1])
    check("refuses a triangle 48-bit wrap before resource binding",
          refusals == ["image_bvh_intersect_ray node address wraps the 48-bit address space"]
          and " bvh base " not in pipe.manifest()
          and "x2w_bvh_intersect_triangle" not in wrap_hlsl and "uint4 bt" not in wrap_hlsl, str(refusals))

    pipe, hlsl, refusals = translate_triangle(0x00, stage_kind="ps")
    check("refuses a triangle pixel stage", len(refusals) == 1 and "outside a compute stage" in refusals[0], str(refusals))


def find_dxc():
    named = os.environ.get("X2W_DXC")
    if named and Path(named).is_file():
        return str(Path(named))
    found = shutil.which("dxc")
    if found:
        return found
    kits = os.environ.get("ProgramFiles(x86)", "")
    if kits:
        candidates = sorted((Path(kits) / "Windows Kits" / "10" / "bin").glob("*/x64/dxc.exe"), reverse=True)
        if candidates:
            return str(candidates[0])
    return None


def compile_check(hlsl):
    dxc = find_dxc()
    if dxc is None:
        print("dxc compile: skip (no dxc on this host)")
        return None
    with tempfile.TemporaryDirectory(dir=ROOT / "out" if (ROOT / "out").is_dir() else None) as temporary:
        source = Path(temporary) / "bvh_triangle.hlsl"
        source.write_text(hlsl, encoding="ascii")
        result = subprocess.run([dxc, "-Gis", "-T", "cs_6_0", "-E", "main", "-Fo", str(source.with_suffix(".cso")), str(source)],
                                capture_output=True, text=True)
    if result.returncode or result.stderr:
        check("dxc compiles the translated triangle helper", False, result.stderr or "exit %d" % result.returncode)
        return False
    print("dxc compile: ok (%s)" % dxc)
    return True


def ffloat(value):
    return "asfloat(0x%08xu)" % struct.unpack("<I", struct.pack("<f", value))[0]


def fbits(value):
    return "0x%08xu" % struct.unpack("<I", struct.pack("<f", value))[0]


def triangle_layout():
    images = []
    offsets = {}
    for case in TRIANGLES:
        if case.image not in offsets:
            offsets[case.image] = 64 * len(images)
            images.append(case.image)
    result_offset = (64 * len(images) + 63) & ~63
    return images, offsets, result_offset, result_offset + 16 * len(TRIANGLES)


MUTATIONS = {
    "det_sign": (
        "bool ahead = (det > 0.0 && t >= 0.0) || (det < 0.0 && t <= 0.0);",
        "bool ahead = (det > 0.0 && t >= 0.0) || (det > 0.0 && t <= 0.0);"),
    "id_shift": (
        "uint id_byte = (id >> (8u * kind)) & 0xffu;",
        "uint id_byte = id & 0xffu;"),
    "u_sign": (
        "float u = sc.x * sb.y - sc.y * sb.x;",
        "float u = sc.x * sb.y + sc.y * sb.x;"),
    "miss_words": (
        "uint4 miss = uint4(0x7f800000u, 0x3f800000u, 0u, 0u);",
        "uint4 miss = uint4(0u, 0u, 0u, 0u);"),
}


def dispatch_hlsl(mutation=None):
    helpers = "\n".join(G.BVH_HELPERS).replace("ByteAddressBuffer", "RWByteAddressBuffer")
    if mutation:
        old, new = MUTATIONS[mutation]
        assert helpers.count(old) == 1, (mutation, old)
        helpers = helpers.replace(old, new)
    images, offsets, result_offset, _buffer_size = triangle_layout()
    lines = [helpers, "RWByteAddressBuffer data : register(u0);", "[numthreads(1, 1, 1)]", "void main() {"]
    for image in images:
        base = offsets[image]
        for i in range(16):
            lines.append("    data.Store(%du, 0x%08xu);" % (base + 4 * i, struct.unpack_from("<I", image, 4 * i)[0]))
    lines.append("    DeviceMemoryBarrierWithGroupSync();")
    for case, witness in enumerate(TRIANGLES):
        origin, direction, inverse, _extent = witness.ray
        pointer = (offsets[witness.image] >> 3) | witness.pointer
        lines.append(
            "    uint4 r%d = x2w_bvh_intersect_triangle(data, 0x%08xu, 0x1000u, 0x80000000u, "
            "0x400u, 0x81000000u, float3(%s, %s, %s), float3(%s, %s, %s), float3(%s, %s, %s));" % (
                case, pointer, ffloat(origin[0]), ffloat(origin[1]), ffloat(origin[2]),
                ffloat(direction[0]), ffloat(direction[1]), ffloat(direction[2]),
                ffloat(inverse[0]), ffloat(inverse[1]), ffloat(inverse[2])))
        for word in range(4):
            lines.append("    data.Store(%du, r%d.%s);" % (result_offset + 16 * case + 4 * word, case, "xyzw"[word]))
    lines.append("}")
    return "\n".join(lines) + "\n"


TRIANGLE_BOUNDARY_RESULT_OFFSET = 64
TRIANGLE_BOUNDARY_BUFFER_SIZE = 96


def boundary_hlsl():
    witness = TRIANGLES[0]
    helpers = "\n".join(G.BVH_HELPERS).replace("ByteAddressBuffer", "RWByteAddressBuffer")
    lines = [helpers, "RWByteAddressBuffer data : register(u0);", "[numthreads(1, 1, 1)]", "void main() {"]
    for i in range(16):
        lines.append("    data.Store(%du, 0x%08xu);" % (4 * i, struct.unpack_from("<I", witness.image, 4 * i)[0]))
    lines.append("    DeviceMemoryBarrierWithGroupSync();")
    origin, direction, inverse, _extent = witness.ray
    expected = []
    for case, pointer, word2, result in ((0, 0x00, 0, witness.expected),
                                         (1, 0x08, 0, (0x7f800000, 0x3f800000, 0, 0))):
        lines.append(
            "    uint4 b%d = x2w_bvh_intersect_triangle(data, 0x%08xu, 0x1000u, 0x80000000u, "
            "0x%08xu, 0x81000000u, float3(%s, %s, %s), float3(%s, %s, %s), float3(%s, %s, %s));" % (
                case, pointer, word2, ffloat(origin[0]), ffloat(origin[1]), ffloat(origin[2]),
                ffloat(direction[0]), ffloat(direction[1]), ffloat(direction[2]),
                ffloat(inverse[0]), ffloat(inverse[1]), ffloat(inverse[2])))
        for word in range(4):
            lines.append("    data.Store(%du, b%d.%s);" % (
                TRIANGLE_BOUNDARY_RESULT_OFFSET + 16 * case + 4 * word, case, "xyzw"[word]))
        expected.append(result)
    lines.append("}")
    return "\n".join(lines) + "\n", expected


def run_boundary_dispatch(dxc, runner, adapter, directory):
    source = directory / "bvh_triangle_boundary.hlsl"
    cso = directory / "bvh_triangle_boundary.cso"
    output = directory / "bvh_triangle_boundary.bin"
    hlsl, expected = boundary_hlsl()
    source.write_text(hlsl, encoding="ascii")
    compiled = subprocess.run([dxc, "-Gis", "-T", "cs_6_0", "-E", "main", "-Fo", str(cso), str(source)],
                              capture_output=True, text=True)
    if compiled.returncode or compiled.stderr:
        raise AssertionError(compiled.stderr or "dxc exited with %d" % compiled.returncode)
    result = subprocess.run([runner, adapter, "--raw-uav-dump", str(cso), "1", str(TRIANGLE_BOUNDARY_BUFFER_SIZE), str(output)],
                            capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 77:
        return 77, None, expected
    if result.returncode or result.stderr:
        raise AssertionError(result.stderr or "offscreen_check exited with %d" % result.returncode)
    data = output.read_bytes()
    assert len(data) == TRIANGLE_BOUNDARY_BUFFER_SIZE, (len(data), TRIANGLE_BOUNDARY_BUFFER_SIZE)
    return 0, struct.unpack("<%dI" % (len(data) // 4), data), expected


def run_dispatch(dxc, runner, adapter, directory, mutation=None):
    _images, _offsets, _result_offset, buffer_size = triangle_layout()
    name = "bvh_triangle_" + mutation if mutation else "bvh_triangle"
    source = directory / (name + ".hlsl")
    cso = directory / (name + ".cso")
    output = directory / (name + ".bin")
    source.write_text(dispatch_hlsl(mutation), encoding="ascii")
    compiled = subprocess.run([dxc, "-Gis", "-T", "cs_6_0", "-E", "main", "-Fo", str(cso), str(source)],
                              capture_output=True, text=True)
    if compiled.returncode or compiled.stderr:
        raise AssertionError(compiled.stderr or "dxc exited with %d" % compiled.returncode)
    result = subprocess.run([runner, adapter, "--raw-uav-dump", str(cso), "1", str(buffer_size), str(output)],
                            capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 77:
        return 77, None
    if result.returncode or result.stderr:
        raise AssertionError(result.stderr or "offscreen_check exited with %d" % result.returncode)
    data = output.read_bytes()
    assert len(data) == buffer_size, (len(data), buffer_size)
    return 0, struct.unpack("<%dI" % (len(data) // 4), data)


def check_dispatch(dxc, runner, adapter):
    controls()
    if failures:
        return 1
    if not dxc or not runner:
        print("dispatch: skip (no dxc or offscreen_check)")
        return 77
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    _images, _offsets, result_offset, _buffer_size = triangle_layout()
    result_word = result_offset // 4
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        directory = Path(temporary)
        status, words = run_dispatch(dxc, runner, adapter, directory)
        if status == 77:
            print("dispatch: skip (adapter unavailable)")
            return 77
        for case, witness in enumerate(TRIANGLES):
            start = result_word + 4 * case
            got = tuple(words[start:start + 4])
            check("dispatch triangle %d %s" % (case, witness.name), got == witness.expected,
                  "got %s expected %s" % (["%08x" % w for w in got], ["%08x" % w for w in witness.expected]))
        for mutation in MUTATIONS:
            status, mutant = run_dispatch(dxc, runner, adapter, directory, mutation=mutation)
            if status == 77:
                print("dispatch control: skip (adapter unavailable)")
                return 77
            changed = sum(tuple(mutant[result_word + 4 * case:result_word + 4 * case + 4]) != witness.expected
                          for case, witness in enumerate(TRIANGLES))
            check("dispatch %s control changes a triangle result" % mutation, changed > 0, str(changed))
        status, boundary_words, boundary_expected = run_boundary_dispatch(dxc, runner, adapter, directory)
        if status == 77:
            print("dispatch boundary: skip (adapter unavailable)")
            return 77
        for case, wanted in enumerate(boundary_expected):
            start = TRIANGLE_BOUNDARY_RESULT_OFFSET // 4 + 4 * case
            got = tuple(boundary_words[start:start + 4])
            check("triangle decoded extent case %d" % case, got == wanted,
                  "got %s expected %s" % (["%08x" % w for w in got], ["%08x" % w for w in wanted]))
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    print("dispatch: all ok on %s" % adapter)
    return 0


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "dispatch":
        return check_dispatch(sys.argv[2], sys.argv[3], sys.argv[4])
    if len(sys.argv) != 1:
        sys.exit(__doc__)
    controls()
    pipe, hlsl, refusals = translate_triangle(0x00)
    assert not refusals, refusals
    compile_check(hlsl)
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    print("startup gfx2hlsl bvh triangle: all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

