"""Clip and cull distance exports in mesh programs (#1998 item F).

tests/mesh_clip_cull.s exports pos0, then pos1 and pos2 as the CCDIST0 and CCDIST1 vectors that
PA_CL_VS_OUT_CNTL 0x00c01003 enables: clip distances 0 and 1, cull distance 4. Geometry word 1
selects one of two draws of two triangles on a 32x32 target, so clipping and culling are tested
separately:
  clip  A loses the pixels left of x = 9 to clip distance 0 and B the pixels above y = 7 to clip
        distance 1. Both triangles stay partly visible.
  cull  C is culled because cull distance 4 is negative on all three of its vertices. Its
        neighbour D, with a positive cull distance on every vertex, stays.
A triangle with mixed cull distance signs is left out on purpose: on an RTX 3070 such a triangle
lost the pixels where the interpolated cull distance is negative, while WARP kept it whole.
The expected images come from an exact model of those rules that does not use the translator.

  (no arguments)                                   translations, refusals and guard mutants
  binary <llvm-mc> <llvm-objcopy> <libLLVM>        the source reassembles to CODE; LLVM decode and CLIs agree
  draw <dxc> <offscreen_check> <adapter>           fenced readback on a D3D12 adapter, with negative controls
"""

from fractions import Fraction
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import mesh_clip_cull  # noqa: E402
import mesh_dynamic_aot  # noqa: E402
import mesh_entry_aot  # noqa: E402
from mesh_clip_cull import DistanceLayout  # noqa: E402
from mesh_dynamic_geometry_check import decode  # noqa: E402

SOURCE = Path(__file__).with_name("mesh_clip_cull.s")
WAVE64 = "+wavefrontsize64,-wavefrontsize32"
LIMITS = (6, 2)
ALLOCATION = 0x2006
# Geometry word 1 for each draw: the first model vertex of its two triangles.
DRAWS = {"clip": 0, "cull": 6}
# CLIP_DIST_ENA_0, CLIP_DIST_ENA_1, CULL_DIST_ENA_4, VS_OUT_CCDIST0_VEC_ENA, VS_OUT_CCDIST1_VEC_ENA.
WORD = 0x00C01003

# LLVM 15 .text for SOURCE. Keep these bytes in source so CPU checks need no LLVM.
CODE = bytes.fromhex(
    """
    11038abe 80038bbe 050608f4 200000fa
    7fc08cbf cc0320f4 000000fa 8c0320f4
    040000fa 7fc08cbf 0fff06bf 06200000
    060084bf 0e8606bf 020084bf 860389be
    040082bf 800389be 020082bf 80038fbe
    800389be 0fff9093 00000c00 0f8c1390
    ff03a8be 030c3000 ff03a9be 00042000
    000065d7 c1000100 000066d7 c1000200
    03ff8893 18000400 000046d7 080c0104
    0b0046d7 09000104 7e0496be 8000a47d
    020088bf 0f03fcbe 090090bf 1604febe
    00008abf f002067e f202087e f2020a7e
    f2020c7e f2020e7e f302107e f402147e
    8016a47d ff02027e 000060bf ff02047e
    0000603f ff020a7e 0000e040 1604febe
    8116a47d ff02027e 0000e0bd ff02047e
    0000603f ff020a7e 0000a8c0 1604febe
    8216a47d ff02027e 000060bf ff02047e
    0000e03d ff020a7e 0000e040 1604febe
    8316a47d ff02027e 0000003e ff02047e
    0000603f ff020c7e 0000a0c0 1604febe
    8416a47d ff02027e 0000643f ff02047e
    0000603f ff020c7e 0000a0c0 1604febe
    8516a47d ff02027e 0000003e ff02047e
    0000e03d ff020c7e 0000e840 1604febe
    8616a47d ff02027e 000060bf ff02047e
    000000be f3020e7e 1604febe 8716a47d
    ff02027e 0000e0bd ff02047e 000000be
    f3020e7e 1604febe 8816a47d ff02027e
    000060bf ff02047e 000064bf f3020e7e
    1604febe 8916a47d ff02027e 0000003e
    ff02047e 000000be 1604febe 8a16a47d
    ff02027e 0000643f ff02047e 000000be
    1604febe 8b16a47d ff02027e 0000003e
    ff02047e 000064bf 1604febe 1000a87d
    cf0000f8 01020304 df0000f8 05060808
    ef0800f8 070a0808 1604febe 1300a87d
    090043d5 0051a400 410900f8 09000000
    000081bf
    """
)
POS0 = "exp pos0 v1, v2, v3, v4"
POS1 = "exp pos1 v5, v6, v8, v8"
POS2 = "exp pos2 v7, v10, v8, v8 done"
ALLOCATION_CHECK = "s_cmp_eq_u32 s15, 0x2006"

# The independent model. Vertex 3t + c is corner c of triangle t, in screen pixels, followed by
# the four components of pos1 (CCDIST0) and of pos2 (CCDIST1) that the program exports.
F = Fraction
CORNERS = ((0, 0), (F(49, 4), 0), (0, F(49, 4)))
ORIGINS = ((2, 2), (18, 2), (2, 18), (18, 18))
OFF = (-1, -1)  # disabled components of pos1; pos2 holds 2.0 and -1.0, -1.0 after cull distance 4
DISTANCES = (
    ((7, 1), (F(-21, 4), 1), (7, 1)),  # A: clip 0 = 9 - x
    ((1, -5), (1, -5), (1, F(29, 4))),  # B: clip 1 = y - 7
    ((1, 1), (1, 1), (1, 1)),  # C
    ((1, 1), (1, 1), (1, 1)),  # D
)
CULL4 = ((1, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 1, 1))
SIDE = 32
GREEN, BLACK = bytes((0, 255, 0, 255)), bytes((0, 0, 0, 255))
PIXEL = "float4 main() : SV_Target { return float4(0,1,0,1); }\n"


# The calibration draw gives D's first two vertices a negative cull distance: mixed signs.
MIXED_CULL4 = CULL4[:3] + ((-1, -1, 1),)


def vertices(cull4=CULL4):
    """(x, y, components 0-7) per vertex: x and y in screen pixels, components as exported."""
    out = []
    for t, (ox, oy) in enumerate(ORIGINS):
        for c, (dx, dy) in enumerate(CORNERS):
            clip = DISTANCES[t][c]
            out.append((ox + dx, oy + dy, (*clip, *OFF, cull4[t][c], 2, -1, -1)))
    return out


def clip_space(value, flip):
    """Screen pixel to clip space, checked to be an exact binary32 value."""
    result = 1 - value / 16 if flip else value / 16 - 1
    assert F(struct.unpack("<f", struct.pack("<f", float(result)))[0]) == result
    return result


def expected(draw, clip, cull, per_pixel_cull=False, cull4=CULL4):
    """Covered pixel centers of one draw for the given clip and cull component indices.

    A triangle is culled when one cull component is negative on all three vertices. A covered
    center is kept when every clip component, interpolated linearly (w = 1), is not negative.
    With per_pixel_cull, the cull components of a triangle that is not culled also remove the
    pixels where they interpolate to a negative value, as observed on an RTX 3070.
    No center may lie on a triangle edge or within 1/8 pixel of a clip plane.
    """
    points = vertices(cull4)
    if per_pixel_cull:
        clip = tuple(clip) + tuple(cull)
    first = DRAWS[draw] // 3
    covered = set()
    for t in (first, first + 1):
        tri = points[3 * t : 3 * t + 3]
        if any(all(p[2][k] < 0 for p in tri) for k in cull):
            continue
        (ax, ay, _), (bx, by, _), (cx, cy, _) = tri
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        for py in range(SIDE):
            for px in range(SIDE):
                x, y = F(2 * px + 1, 2), F(2 * py + 1, 2)
                w = [((bx - x) * (cy - y) - (by - y) * (cx - x)) / area,
                     ((cx - x) * (ay - y) - (cy - y) * (ax - x)) / area,
                     ((ax - x) * (by - y) - (ay - y) * (bx - x)) / area]
                if 0 in w:
                    raise AssertionError(f"Pixel center {px},{py} lies on an edge of triangle {t}")
                if any(v < 0 for v in w):
                    continue
                for k in clip:
                    d = [p[2][k] for p in tri]
                    value = sum(wi * di for wi, di in zip(w, d))
                    # Gradient of the interpolated distance in pixels: its zero line must stay 1/8 pixel away.
                    gx = ((d[1] - d[0]) * (cy - ay) - (d[2] - d[0]) * (by - ay)) / area
                    gy = ((d[2] - d[0]) * (bx - ax) - (d[1] - d[0]) * (cx - ax)) / area
                    if (gx or gy) and value * value < (gx * gx + gy * gy) / 64:
                        raise AssertionError(f"Pixel center {px},{py} is within 1/8 pixel of clip plane {k}")
                    if not (gx or gy) and value == 0:
                        raise AssertionError(f"Clip distance {k} is zero on triangle {t}")
                if all(sum(wi * p[2][k] for wi, p in zip(w, tri)) > 0 for k in clip):
                    covered.add((px, py))
    return covered


def image(covered):
    return b"".join(GREEN if (x, y) in covered else BLACK for y in range(SIDE) for x in range(SIDE))


GOOD = {draw: expected(draw, clip=(0, 1), cull=(4,)) for draw in DRAWS}


def edit(decoded, old, new):
    """Replace one decoded line with zero or more lines; the CPU model needs only their order."""
    assert sum(text == old for _, text in decoded) == 1, old
    out = []
    for pc, text in decoded:
        if text != old:
            out.append((pc, text))
        elif new is not None:
            out.extend((pc, line) for line in new.split("\n"))
    return out


def static(decoded, word=WORD, draw="clip", **kwargs):
    return mesh_entry_aot.lower(decoded, [ALLOCATION, DRAWS[draw]], LIMITS, vs_out_cntl=word, **kwargs)


def dynamic(decoded, word=WORD):
    return mesh_dynamic_aot.lower(decoded, 2, LIMITS, vs_out_cntl=word)


def refuses(lower, decoded, word, text):
    try:
        lower(decoded, word)
    except ValueError as error:
        if str(error) != text:
            raise AssertionError(f"Expected refusal {text!r}, got {str(error)!r}") from None
        return
    raise AssertionError(f"Expected refusal {text!r}, translation succeeded")


def distance_write(prefix, vertex, model_vertex):
    """The HLSL that must set one vertex's clip distances 0-1 and cull distance 4."""
    components = vertices()[model_vertex][2]
    words = [struct.unpack("<I", struct.pack("<f", float(components[k])))[0] for k in (0, 1, 4)]
    return (f"{prefix}[{vertex}].clip0=float2(asfloat(0x{words[0]:08x}u), asfloat(0x{words[1]:08x}u));"
            f"{prefix}[{vertex}].cull1=asfloat(0x{words[2]:08x}u);")


def check_translation(decoded):
    fields = "float2 clip0:SV_ClipDistance0;float cull1:SV_CullDistance1;};"
    runtime = dynamic(decoded)
    assert runtime.count(fields) == 1, "runtime: struct fields"
    for draw, first in DRAWS.items():
        text = static(decoded, draw=draw)
        assert text.count(fields) == 1, f"static {draw}: struct fields"
        # Clip distances 0 and 1 from pos1.xy, cull distance 4 from pos2.x; the disabled -1.0 and 2.0 are dropped.
        for vertex in range(LIMITS[0]):
            assert distance_write("v", vertex, first + vertex) in text, f"static {draw}: vertex {vertex}"
            assert distance_write("vertices", vertex, first + vertex) in runtime, f"runtime {draw}: vertex {vertex}"
        # Nothing else is written: the disabled components have no field.
        assert text.count("clip0=") == text.count("cull1=") == LIMITS[0], f"static {draw}: distance writes"
        assert "clip1" not in text + runtime and "cull0" not in text + runtime, f"{draw}: a disabled vector"
    assert runtime.count("clip0=") == runtime.count("cull1=") == 2 * LIMITS[0], "runtime: distance writes"
    # Mesa writes disabled components as off; the translation is the same.
    mesa = edit(edit(decoded, POS1, "exp pos1 v5, v6, off, off"), POS2, "exp pos2 v7, off, off, off done")
    assert static(mesa) == static(decoded) and dynamic(mesa) == runtime
    # Only CCDIST1 enabled: it is the first distance vector, so pos1 carries it.
    only_second = edit(edit(decoded, POS1, None), POS2, POS2.replace("pos2", "pos1"))
    text = static(only_second, 0x00801000)
    assert "float cull1:SV_CullDistance1;};" in text and "clip" not in text
    # radeonsi sets CULL_DIST_ENA for every clip distance too; both enables declare both semantics.
    both = static(decoded, WORD | 0x300)
    assert "float2 clip0:SV_ClipDistance0;float2 cull0:SV_CullDistance0;float cull1:SV_CullDistance1;};" in both
    # A program without distances is unchanged by a word that enables none.
    plain = edit(edit(edit(decoded, POS1, None), POS2, None), POS0, POS0 + " done")
    assert static(plain, 0) == static(plain, None) and dynamic(plain, 0) == dynamic(plain, None)
    assert "Distance" not in static(plain, None) + dynamic(plain, None)
    print("static translations of both draws and the runtime translation declare clip 0-1 from pos1 and"
          " cull 4 from pos2; off components, CCDIST1 alone, both enables and the no-distance word checked")


def drop(decoded, *lines):
    for line in lines:
        decoded = edit(decoded, line, None)
    return decoded


REFUSALS = (
    # Unsupported state: no word, the misc vector, USE_VTX_* bits, unknown bits, contradictory enables.
    (lambda d: d, None, "Mesh export pos1 needs the PA_CL_VS_OUT_CNTL word"),
    (lambda d: d, WORD | 1 << 21,
     "PA_CL_VS_OUT_CNTL 0x00e01003 sets VS_OUT_MISC_VEC_ENA; only clip and cull distance vectors are translated"),
    (lambda d: d, WORD | 1 << 16 | 1 << 19,
     "PA_CL_VS_OUT_CNTL 0x00c91003 sets USE_VTX_POINT_SIZE, USE_VTX_VIEWPORT_INDX; "
     "only clip and cull distance vectors are translated"),
    (lambda d: d, WORD | 1 << 28, "PA_CL_VS_OUT_CNTL 0x10c01003 sets unknown bits 0x10000000"),
    (lambda d: d, 1 << 32, "PA_CL_VS_OUT_CNTL must be a 32-bit unsigned word"),
    (lambda d: d, 0x00800003,
     "PA_CL_VS_OUT_CNTL 0x00800003 enables distance components 0, 1 without VS_OUT_CCDIST0_VEC_ENA"),
    (lambda d: d, 0x00C00003,
     "PA_CL_VS_OUT_CNTL 0x00c00003 sets VS_OUT_CCDIST1_VEC_ENA with no clip or cull distance in components 4-7"),
    (lambda d: d, 0x00C0FFFF, "PA_CL_VS_OUT_CNTL 0x00c0ffff enables 16 clip and cull components; D3D12 allows 8"),
    # A distance export the word does not enable, and an enabled one that is missing.
    (lambda d: d, 0x00400003, "Mesh export pos2 is not enabled by PA_CL_VS_OUT_CNTL 0x00400003"),
    (lambda d: d, 0, "Mesh export pos1 is not enabled by PA_CL_VS_OUT_CNTL 0x00000000"),
    (lambda d: edit(drop(d, POS2), POS1, POS1 + " done"), WORD,
     "PA_CL_VS_OUT_CNTL 0x00c01003 enables pos2 (CCDIST1), which vertex 0 does not export"),
    (lambda d: edit(d, POS2, POS2.replace("pos2", "pos3")), WORD,
     "Mesh export pos3 is not enabled by PA_CL_VS_OUT_CNTL 0x00c01003"),
    (lambda d: edit(d, POS2, POS2.replace("pos2", "pos4")), WORD, "Unsupported mesh export"),
    # An enabled component written as off.
    (lambda d: edit(d, POS1, "exp pos1 v5, off, v8, v8"), WORD, "Enabled clip distance 1 is not exported in pos1 of vertex 0"),
    (lambda d: edit(d, POS2, "exp pos2 off, v10, v8, v8 done"), WORD,
     "Enabled cull distance 4 is not exported in pos2 of vertex 0"),
    # Duplicate, reordered and wrongly completed exports.
    (lambda d: edit(d, POS1, POS1 + "\n" + POS1), WORD, "Duplicate position export pos1 for vertex 0"),
    (lambda d: edit(edit(d, POS1, POS2.replace(" done", "")), POS2, POS1 + " done"), WORD,
     "Position exports of vertex 0 are out of order: pos0, pos2, pos1"),
    (lambda d: edit(d, POS1, POS1 + " done"), WORD, "Expected done on the last position export pos2 of vertex 0 only"),
    (lambda d: edit(d, POS2, POS2.replace(" done", "")), WORD,
     "Expected done on the last position export pos2 of vertex 0 only"),
    (lambda d: drop(d, POS1, POS2), None, "Expected completed primitive or position export"),
)


def check_refusals(decoded):
    count = 0
    for mutate, word, text in REFUSALS:
        program = mutate(decoded)
        for lower in (static, dynamic):
            refuses(lower, program, word, text)
            count += 1
    # The compute fallback has no room for distances; one triangle passes its own count check first.
    triangle = edit(decoded, ALLOCATION_CHECK, "s_cmp_eq_u32 s15, 0x1003")
    try:
        mesh_entry_aot.lower(triangle, [0x1003, 0], LIMITS, vs_out_cntl=WORD, compute_fallback=True)
    except ValueError as error:
        assert str(error) == "The compute fallback has no room for clip or cull distances", error
    else:
        raise AssertionError("Compute fallback accepted distances")
    print(f"{len(REFUSALS)} forms refused by name in both lowerers ({count} checks) and the compute fallback")


GUARDS = (
    ("refused bits", "    if named:\n", "    if False:\n"),
    ("unknown bits", "    if word & ~KNOWN_BITS:\n", "    if False:\n"),
    ("export not enabled", "            if target in expected:\n", "            if True:\n"),
    ("enabled export missing", "            if target not in targets:\n", "            if False:\n"),
    ("duplicate export", "        if len(set(targets)) != len(targets):\n", "        if False:\n"),
    ("component missing", "and value is None:\n", "and False:\n"),
    ("done on the last export", "        if [done for _, _, done in exports]", "        if False and [done for _, _, done in exports]"),
)


def mutated_module(old, new):
    source = Path(mesh_clip_cull.__file__).read_text(encoding="utf-8")
    assert source.count(old) == 1, old
    module = types.ModuleType("mesh_clip_cull_mutant")
    exec(compile(source.replace(old, new), "mesh_clip_cull_mutant", "exec"), module.__dict__)
    return module


def guard_mutants(decoded):
    """Each guard removed must turn this check red."""
    names = ("decode_vs_out_cntl", "vertex_distances", "hlsl_fields", "hlsl_writes")
    for name, old, new in GUARDS:
        module = mutated_module(old, new)
        with patch.multiple(mesh_entry_aot, **{n: getattr(module, n) for n in names}), \
                patch.multiple(mesh_dynamic_aot, **{n: getattr(module, n) for n in names}):
            try:
                check_refusals(decoded)
            except (AssertionError, TypeError) as error:
                print(f"guard removed ({name}): check failed as required: {type(error).__name__}: {error}")
                continue
        raise AssertionError(f"Removing the {name} guard left the check green")


def layout_with(**changes):
    """A translator mutant: the decoded word with one part of its meaning changed."""
    def decode_mutant(word):
        layout = mesh_clip_cull.decode_vs_out_cntl(word)
        return DistanceLayout(word, changes.get("vectors", layout.vectors), frozenset(changes.get("clip", layout.clip)),
                              frozenset(changes.get("cull", layout.cull)))
    return decode_mutant


# Negative controls: the exported components (0-3 pos1, 4-7 pos2) each translator mutant draws as
# clip and as cull distances.
CONTROLS = {
    "clip and cull swapped": (((4,), (0, 1)), layout_with(clip=(4,), cull=(0, 1))),
    # pos1 read as CCDIST1 and pos2 as CCDIST0: clip 0-1 take pos2.xy, cull 4 takes pos1.x.
    "CCDIST0 and CCDIST1 swapped": (((4, 5), (0,)), layout_with(vectors=(("pos1", 1), ("pos2", 0)))),
    "clip distance 1 dropped": (((0,), (4,)), layout_with(clip=(0,))),
    "cull distance 4 dropped": (((0, 1), ()), layout_with(cull=())),
    "word ignored, every exported component a clip distance": ((tuple(range(8)), ()),
                                                               layout_with(clip=range(8), cull=())),
}


def check_reference():
    for x, y, _ in vertices():
        clip_space(x, False), clip_space(y, True)
    for per_pixel in (False, True):
        kind = "per-pixel" if per_pixel else "whole-primitive"
        for draw, covered in GOOD.items():
            # No triangle of the good draws has mixed cull signs, so both behaviours draw the same image.
            assert expected(draw, (0, 1), (4,), per_pixel) == covered, draw
            changed = {name: len(expected(draw, *roles, per_pixel) ^ covered) for name, (roles, _) in CONTROLS.items()}
            print(f"reference {draw} ({kind} cull): {len(covered)} covered pixels; controls change {changed}")
    mixed = len(expected("cull", (0, 1), (4,), True, MIXED_CULL4) ^ GOOD["cull"])
    assert mixed and expected("cull", (0, 1), (4,), False, MIXED_CULL4) == GOOD["cull"]
    print(f"calibration: a mixed-sign cull distance changes {mixed} pixels if applied per pixel, 0 otherwise")
    # Each draw isolates one rule: clipping alone changes the clip draw, culling alone the cull draw.
    assert expected("clip", (), ()) != GOOD["clip"] and expected("clip", (), (4,)) == expected("clip", (), ())
    assert expected("cull", (0, 1), ()) != GOOD["cull"] and expected("cull", (), (4,)) == GOOD["cull"]
    for name, (roles, _) in CONTROLS.items():
        if all(expected(draw, *roles) == GOOD[draw] for draw in DRAWS):
            raise AssertionError(f"control {name} draws the same pixels")


def binary(assembler, objcopy, library):
    from pixel_aot import _read_named_container, disassemble
    from pixel_aot_check import envelope

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        obj, raw = tmp / "fixture.o", tmp / "fixture.bin"
        subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-mattr=" + WAVE64,
                        "-filetype=obj", SOURCE, "-o", obj], check=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
        code = raw.read_bytes()
        assert code == CODE, "Embedded bytes differ from the assembled mesh"
        native = disassemble(code, library, offsets=True, features=WAVE64.encode("ascii"))
        assert native == decode(code), "LLVM and the Python decoder disagree"
        padded = code + bytes.fromhex("00009fbf") * (4 - (len(code) // 4 + 10) % 4)
        record = envelope(padded, directory_end=(40 + len(padded), 288 + len(padded)),
                          compute_width=128, mesh_limits=LIMITS)
        container = tmp / "input.dxbc"
        container.write_bytes(_read_named_container(record, len(b"fixture") + 2))
        geometry = tmp / "geometry.json"
        geometry.write_text(f"[{ALLOCATION}, {DRAWS['cull']}]")
        out = tmp / "out.hlsl"
        tools = {
            "static": ([sys.executable, "-B", ROOT / "tools" / "mesh_entry_aot.py", container,
                        "--geometry-words", geometry], static(decode(CODE), draw="cull")),
            "runtime": ([sys.executable, "-B", ROOT / "tools" / "mesh_dynamic_aot.py", container,
                         "--geometry-word-count", "2"], dynamic(decode(CODE))),
        }
        for name, (tool, text) in tools.items():
            command = [*tool, "--llvm", library, "--output", out]
            subprocess.run([*command, "--pa-cl-vs-out-cntl", hex(WORD)], check=True)
            assert out.read_text() == text, f"{name}: CLI output differs from the source text"
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 1, (name, result)
            assert "Mesh export pos1 needs the PA_CL_VS_OUT_CNTL word" in result.stderr, (name, result)
            result = subprocess.run([*command, "--pa-cl-vs-out-cntl", "word"], capture_output=True, text=True)
            assert result.returncode == 2 and "--pa-cl-vs-out-cntl" in result.stderr, (name, result)
            print(f"{name} CLI: same HLSL as the source text; exit 1 without the word, 2 for an invalid word")
    print(f"LLVM 15 reassembled {len(CODE)} bytes; native decode matches the Python decoder")


def fixture(draw, covered):
    """Two 512-byte constant buffers, the 512-byte geometry SRV, the untouched 80-byte trace, the image."""
    data = bytearray(1536)
    struct.pack_into("<II", data, 1024, ALLOCATION, DRAWS[draw])
    return bytes(data) + b"\xcd" * 80 + image(covered)


def readback_mismatches(result):
    lines = [line for line in result.stdout.splitlines() if line.startswith("pixels=1024 mismatches=")]
    return int(lines[0].split("=")[-1]) if len(lines) == 1 else None


def expect(result, adapter, code, mismatches, label):
    lines = result.stdout.splitlines()
    debug = f"backend={adapter} debug_layer=enabled" in lines
    got = readback_mismatches(result)
    ok = (result.returncode == code and got == mismatches
          and (not debug or "debug_errors=0 debug_warnings=0" in lines))
    print(f"{label}: exit={result.returncode} mismatches={got} expected={mismatches} {'ok' if ok else 'UNEXPECTED'}",
          flush=True)
    if not ok:
        sys.stderr.write(result.stdout + result.stderr)
        raise AssertionError(f"{label}: unexpected result (exit {result.returncode})")


def draw(dxc, runner, adapter):
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    decoded = decode(CODE)
    with tempfile.TemporaryDirectory(prefix="mesh-clip-cull-", dir=out) as directory:
        tmp = Path(directory)

        def compile_shader(stem, text, profile):
            (tmp / f"{stem}.hlsl").write_text(text, encoding="ascii")
            subprocess.run([dxc, "-WX", "-T", profile, "-E", "main", "-Fo", tmp / f"{stem}.dxil",
                            tmp / f"{stem}.hlsl"], check=True)
            return tmp / f"{stem}.dxil"

        def run(mesh, draw_name, covered):
            path = tmp / "draw.bin"
            path.write_bytes(fixture(draw_name, covered))
            return subprocess.run([runner, adapter, "--mesh-indexed", mesh, pixel, path],
                                  capture_output=True, text=True, timeout=120)

        pixel = compile_shader("pixel", PIXEL, "ps_6_0")
        runtime = compile_shader("runtime", dynamic(decoded), "ms_6_5")
        for draw_name, covered in GOOD.items():
            for mesh, label in ((compile_shader(f"static_{draw_name}", static(decoded, draw=draw_name), "ms_6_5"),
                                 "static"), (runtime, "runtime")):
                result = run(mesh, draw_name, covered)
                if result.returncode == 77 or "mesh_shader_tier=unsupported" in result.stdout:
                    print(result.stdout, end="")
                    return 77
                expect(result, adapter, 0, 0, f"{draw_name} draw, {label} translation: {len(covered)} covered pixels")
        # Calibration: D with a mixed-sign cull distance. WARP keeps the whole triangle; an RTX 3070 also
        # removed the pixels where the cull distance interpolates to a negative value. The two good draws
        # have no mixed signs, so they do not depend on it; some controls do.
        mixed = static(decoded, draw="cull")
        for vertex in (3, 4):
            old = f"v[{vertex}].cull1=asfloat(0x3f800000u)"
            assert mixed.count(old) == 1, old
            mixed = mixed.replace(old, f"v[{vertex}].cull1=asfloat(0xbf800000u)")
        result = run(compile_shader("mixed", mixed, "ms_6_5"), "cull", GOOD["cull"])
        pixel_count = len(expected("cull", (0, 1), (4,), True, MIXED_CULL4) ^ GOOD["cull"])
        got = readback_mismatches(result)
        if got not in (0, pixel_count):
            sys.stderr.write(result.stdout + result.stderr)
            raise AssertionError(f"mixed-sign cull distance: {got} pixels differ, expected 0 or {pixel_count}")
        per_pixel = got == pixel_count
        expect(result, adapter, 1 if got else 0, got,
               "mixed-sign cull distance: " + ("pixels removed per pixel" if per_pixel else "whole triangle kept"))
        for name, (roles, decode_mutant) in CONTROLS.items():
            seen = 0
            for draw_name, covered in GOOD.items():
                with patch.object(mesh_entry_aot, "decode_vs_out_cntl", decode_mutant):
                    mesh = compile_shader("mutant", static(decoded, draw=draw_name), "ms_6_5")
                drawn = expected(draw_name, *roles, per_pixel)
                want = len(drawn ^ covered)
                seen += want
                # The good reference against the mutant's draw: exactly the pixels the model predicts differ.
                expect(run(mesh, draw_name, covered), adapter, 1 if want else 0, want,
                       f"{name}, {draw_name} draw: mutant draws {len(drawn)} pixels")
            if not seen:
                print(f"{name}: no pixel differs on this adapter, as its cull behaviour predicts")
    return 0


def main(argv):
    decoded = decode(CODE)
    if len(argv) == 4 and argv[0] == "binary":
        binary(*argv[1:])
    elif len(argv) == 4 and argv[0] == "draw":
        return draw(*argv[1:])
    elif argv:
        raise SystemExit(__doc__)
    check_reference()
    check_translation(decoded)
    check_refusals(decoded)
    guard_mutants(decoded)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
