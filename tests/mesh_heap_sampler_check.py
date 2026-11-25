"""Mesh sampling through a sampler loaded from the sampler heap (#1998 item E).

tests/mesh_heap_sampler.s draws a quad whose PARAM0 is one image_sample_lz of a texture at
(1.25, 0.5). The sampler comes from the sampler heap: geometry word 0 = 0x10 or 0x20 is used as
the register offset, any other word takes the constant offset 0x30. The expected colors come from
an independent model of that control flow and of the sampler fields (address mode, LOD clamp).

  (no arguments)                                   translation, refusals and guard mutants
  binary <llvm-mc> <llvm-objcopy> <libLLVM>        the source reassembles to CODE; LLVM decode and CLI agree
  draw <dxc> <offscreen_check> <adapter>           fenced readback on a D3D12 adapter, with negative controls
"""

from pathlib import Path
import math
import re
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "startup"))
import mesh_dynamic_aot  # noqa: E402
from mesh_dynamic_aot import enumerate_variants, lower  # noqa: E402
from mesh_dynamic_geometry_check import decode  # noqa: E402

SOURCE = Path(__file__).with_name("mesh_heap_sampler.s")
WAVE64 = "+wavefrontsize64,-wavefrontsize32"
LIMITS = (4, 2)
HEAP_SGPR = 16
SAMPLE_U = 1.25

# LLVM 15 .text for SOURCE. Keep these bytes in source so CPU checks need no LLVM.
CODE = bytes.fromhex(
    """
    11038abe 80038bbe 050608f4 200000fa
    05070cf4 400000fa 10038cbe 80038dbe
    7fc08cbf 4c0120f4 000000fa 7fc08cbf
    059006bf 010084bf 020082bf 05a006bf
    030084bf 060a08f4 0000000a 020082bf
    060a08f4 300000fa 000065d7 c1000100
    000066d7 c1000200 03ff8893 18000400
    000046d7 080c0104 7e0496be 8000a47d
    030088bf ff03fcbe 04200000 090090bf
    1604febe 00008abf 7fc08cbf 8400a87d
    81001436 82001636 1400c2d4 81140200
    0e00c2d4 82160200 010001d5 f3e45100
    020001d5 f3e43900 8002067e f202087e
    ff020e7e 0000a03f f002107e 080f9cf0
    070c4701 703f8cbf cf0800f8 01020304
    0f0200f8 0c0d0e0f 1604febe 8200a87d
    1400c2d4 81000200 ff02127e 00042000
    ff020c7e 02043000 090001d5 090d5200
    410900f8 09000000 000081bf
    """
)

# SQ_IMG_SAMP words (RDNA2 ISA, image sampler descriptor): WORD0 bits 2:0 CLAMP_X (0 wrap,
# 2 clamp to last texel); WORD1 bits 11:0 MIN_LOD and 23:12 MAX_LOD, unsigned 4.8; WORD2
# bits 27:26 MIP_FILTER (1 point). Entry 0 is all zero, the sampler gfx2hlsl's default shortcut uses.
MAX_LOD = 0xFFF << 12
POINT_MIP = 1 << 26
ENTRIES = (
    (0, 0, 0, 0),
    (2, MAX_LOD, POINT_MIP, 0),  # 0x10: clamp, LOD 0 and up
    (0, MAX_LOD | 0x100, POINT_MIP, 0),  # 0x20: wrap, LOD 1 and up
    (2, MAX_LOD | 0x100, POINT_MIP, 0),  # 0x30: clamp, LOD 1 and up
)
# RGBA8 texels: level 0 is 4x1, level 1 is 2x1.
LEVELS = (
    ((200, 30, 30, 255), (30, 200, 30, 255), (30, 30, 200, 255), (200, 200, 30, 255)),
    ((30, 200, 200, 255), (200, 30, 200, 255)),
)
WORDS = (0x10, 0x20, 0x30, 0x0, 0x18)


def program_offset(word):
    """The source's control flow: the sampler heap byte offset for a geometry word."""
    return word if word in (0x10, 0x20) else 0x30


def sample(words, u=SAMPLE_U):
    """Point sample at LOD 0 clamped by the sampler, for the two address modes the fixture uses."""
    lod = min(max(0.0, (words[1] & 0xFFF) / 256), ((words[1] >> 12) & 0xFFF) / 256)
    level = min(math.floor(lod + 0.5), len(LEVELS) - 1)
    width = len(LEVELS[level])
    mode = words[0] & 7
    assert mode in (0, 2), mode
    u = u - math.floor(u) if mode == 0 else min(max(u, 0.0), 1.0)
    return LEVELS[level][min(math.floor(u * width), width - 1)]


def expected(word, entries=ENTRIES):
    return sample(entries[program_offset(word) // 16])


def pack(color):
    return struct.unpack("<I", bytes(color))[0]


def edit(decoded, old, new):
    assert sum(text == old for _, text in decoded) == 1, old
    return [(pc, new if text == old else text) for pc, text in decoded]


def insert_after(decoded, anchor, *lines):
    out = []
    for pc, text in decoded:
        out.append((pc, text))
        if text == anchor:
            out.extend((pc + 1 + i, line) for i, line in enumerate(lines))
    assert len(out) == len(decoded) + len(lines), anchor
    return out


CONSTANT_LOAD = "s_load_dwordx4 s[40:43], s[12:13], 0x30"
REGISTER_LOAD = "s_load_dwordx4 s[40:43], s[12:13], s5"
SAMPLE = "image_sample_lz v[12:15], v[7:8], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D"
READY = "v_cmpx_gt_u32_e32 4, v0"  # first instruction after the sampler words are waited for


def unbounded(decoded):
    return edit(decoded, CONSTANT_LOAD, REGISTER_LOAD)


def refuses(decoded, text, sgpr=HEAP_SGPR):
    try:
        lower(decoded, 1, LIMITS, sgpr)
    except ValueError as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError(f"Expected refusal: {text}")


def entry_of(variants, word):
    matches = [v for v in variants if all((word == value) == equal for _, value, equal in v.conditions)]
    assert len(matches) == 1, (word, matches)
    entries = {int(m) for expr in matches[0].expressions for m in re.findall(r"qh\[(\d+)\]", expr)}
    assert len(entries) == 1, entries
    return entries.pop()


def check_translation(decoded):
    variants = enumerate_variants(decoded, 1, LIMITS, HEAP_SGPR)
    assert len(variants) == 3
    for word in WORDS:
        assert entry_of(variants, word) == program_offset(word) // 16, word
    for variant in variants:
        assert variant.counts == (4, 2) and variant.indices == {0: [0, 1, 2], 1: [2, 1, 3]}
        names = [variant.expressions[e] for e in variant.expressions]
        assert all(variant.attributes[i] == names for i in range(4)), variant.attributes
    hlsl = lower(decoded, 1, LIMITS, HEAP_SGPR)
    assert "Texture2D<float4> meshTexture : register(t1);" in hlsl
    assert "SamplerState qh[2048] : register(s0, space1);" in hlsl
    assert hlsl.count(".SampleLevel(") == 12 and "SampleLevel(qh[0]" not in hlsl
    # One table for both lowerers.
    import gfx2hlsl
    assert (mesh_dynamic_aot.SAMPLER_HEAP_SIZE, mesh_dynamic_aot.SAMPLER_HEAP_SPACE,
            mesh_dynamic_aot.SAMPLER_HEAP_FALLBACK) == (
        gfx2hlsl.SAMPLER_HEAP_SIZE, gfx2hlsl.SAMPLER_HEAP_SPACE, gfx2hlsl.SAMPLER_HEAP_FALLBACK)
    # The last title entry is accepted; the reserved fallback and anything unaligned are not.
    last = lower(edit(decoded, CONSTANT_LOAD, "s_load_dwordx4 s[40:43], s[12:13], 0x7fe0"), 1, LIMITS, HEAP_SGPR)
    assert "qh[2046]" in last
    # s_mov copies keep the four words of one load; the original registers may then be reused.
    copied = edit(insert_after(decoded, READY, "s_mov_b64 s[44:45], s[40:41]", "s_mov_b64 s[46:47], s[42:43]",
                               "s_mov_b32 s40, 0"), SAMPLE, SAMPLE.replace("s[40:43]", "s[44:47]"))
    assert lower(copied, 1, LIMITS, HEAP_SGPR) == hlsl
    # The independent colors differ for every path and from the default sampler.
    colors = {expected(word) for word in WORDS}
    assert len(colors) == 3 and sample(ENTRIES[0]) not in colors
    print("Three input cases read heap entries 1, 2 and 3; entry 2046 accepted")
    return hlsl


REFUSALS = (
    (lambda d: d, "Unsupported scalar instruction s_load_dwordx8", None),
    (lambda d: d, "Invalid sampler heap SGPR 17", 17),
    (unbounded, "Unbounded sampler heap offset: runtime geometry word 0 in s5", HEAP_SGPR),
    (lambda d: edit(d, CONSTANT_LOAD, "s_load_dwordx4 s[40:43], s[12:13], 0x38"),
     "Sampler heap offset 0x38 is not one of the 2047 title entries", HEAP_SGPR),
    (lambda d: edit(d, CONSTANT_LOAD, "s_load_dwordx4 s[40:43], s[12:13], 0x7ff0"),
     "Sampler heap offset 0x7ff0 is not one of the 2047 title entries", HEAP_SGPR),
    (lambda d: insert_after(d, READY, "s_mov_b32 s41, 0"),
     "Sampler operand s[40:43] no longer holds the four words of one heap load", HEAP_SGPR),
    (lambda d: insert_after(d, READY, "s_and_b32 s41, s41, 0xfff"),
     "Sampler heap word in s41 changed after the load by s_and_b32", HEAP_SGPR),
    (lambda d: insert_after(d, READY, "s_cmp_eq_u32 s3, 0", "s_cselect_b32 s40, s40, 0"),
     "Sampler heap word in s40 mixed with another value by s_cselect_b32", HEAP_SGPR),
    (lambda d: edit(insert_after(d, READY, "s_mov_b64 s[44:45], 0", "s_mov_b64 s[46:47], 0"),
                    SAMPLE, SAMPLE.replace("s[40:43]", "s[44:47]")),
     "Static sampler words in s[44:47] are not translated in mesh programs", HEAP_SGPR),
    (lambda d: insert_after(d, READY, "v_mov_b32_e32 v20, s40"),
     "Sampler heap word in s40 used by v_mov_b32_e32", HEAP_SGPR),
    (lambda d: insert_after(d, "s_mov_b32 s13, 0", "s_add_u32 s12, s12, 16"),
     "Sampler heap base in s12 used by s_add_u32", HEAP_SGPR),
    (lambda d: edit(d, "s_mov_b32 s13, 0", "s_mov_b32 s13, 1"),
     "Sampler heap address is not the entry SGPR with a zero high word", HEAP_SGPR),
    (lambda d: insert_after(d, "s_barrier", "s_mov_b32 s28, 0"),
     "Unknown mesh texture descriptor s[28:35]", HEAP_SGPR),
    (lambda d: edit(d, SAMPLE, SAMPLE.replace("image_sample_lz", "image_sample_l")),
     "Unsupported mesh image instruction image_sample_l", HEAP_SGPR),
    (lambda d: edit(d, SAMPLE, SAMPLE.replace("dmask:0xf", "dmask:0x7")),
     "image_sample_lz destination v[12:15] does not match dmask 0x7", HEAP_SGPR),
    (lambda d: edit(d, SAMPLE, SAMPLE + " a16"), "Unsupported image_sample_lz operands", HEAP_SGPR),
)


def check_refusals(decoded):
    for mutate, text, sgpr in REFUSALS:
        refuses(mutate(decoded), text, sgpr)
    print(f"{len(REFUSALS)} unproven forms refused by name")


def guard_mutants(decoded):
    """Each guard removed must turn this check red."""
    shortcut = lambda self, value, token: 0  # noqa: E731 - gfx2hlsl's default sampler shortcut
    mutants = (
        ("alignment and range", "heap_entry", lambda offset: offset // 16, check_refusals),
        ("bounded offset", "proven_offset", shortcut, check_refusals),
        ("four words of one load", "sampler_words", lambda self, values: values[0].entry, check_refusals),
    )
    for name, attribute, replacement, check in mutants:
        owner = mesh_dynamic_aot if attribute == "heap_entry" else mesh_dynamic_aot.RuntimeWave
        with patch.object(owner, attribute, replacement):
            try:
                check(decoded)
            except AssertionError as error:
                print(f"guard removed ({name}): check failed as required: {error}")
                continue
        raise AssertionError(f"Removing the {name} guard left the check green")


def reassemble(assembler, objcopy, library):
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
        out = tmp / "out.hlsl"
        tool = [sys.executable, "-B", ROOT / "tools" / "mesh_dynamic_aot.py", container,
                "--geometry-word-count", "1", "--llvm", library, "--output", out]
        subprocess.run([*tool, "--sampler-heap-sgpr", str(HEAP_SGPR)], check=True)
        assert out.read_text() == lower(decode(CODE), 1, LIMITS, HEAP_SGPR)
        for extra, code_expected, text in (
            ([], 1, "Unsupported scalar instruction s_load_dwordx8"),
            (["--sampler-heap-sgpr", "17"], 2, "--sampler-heap-sgpr must name a free SGPR"),
        ):
            result = subprocess.run([*tool, *extra], capture_output=True, text=True)
            assert result.returncode == code_expected and text in result.stderr, (extra, result)
    print(f"LLVM 15 reassembled {len(CODE)} bytes; native decode and CLI match the source text")


PIXEL = "float4 main(float4 p : SV_Position, float4 c : TEXCOORD0) : SV_Target { return c; }\n"


def fixture(word, entries, color):
    data = struct.pack("<II", word, len(entries))
    data += b"".join(struct.pack("<4I", *entry) for entry in entries)
    data += b"".join(bytes(texel) for level in LEVELS for texel in level)
    return data + bytes(color)


def run_draw(runner, adapter, mesh, pixel, path, word, entries, color):
    path.write_bytes(fixture(word, entries, color))
    result = subprocess.run([runner, adapter, "--mesh-heap-sampler", mesh, pixel, path],
                            capture_output=True, text=True, timeout=120)
    out = result.stdout
    if "mesh_shader_tier=unsupported" in out:
        return None
    lines = out.splitlines()
    if f"backend={adapter} debug_layer=enabled" in lines:
        # The stand-in's dynamic_sampler() keeps ComparisonFunc ALWAYS on point samplers, which the
        # debug layer reports once per sampler as warning 1361. Every other message fails the check.
        messages = [line for line in result.stderr.splitlines() if line.startswith("D3D12 ")]
        unexpected = [line for line in messages if not line.startswith("D3D12 1361: ")]
        if not re.search(r"^debug_errors=0 debug_warnings=\d+$", out, re.M) or unexpected:
            raise AssertionError(f"debug layer messages for word {word:#x}:\n{out}\n" + "\n".join(unexpected))
    match = re.search(r"first=([0-9a-f]{8}) pixels=1024 mismatches=(\d+)", out)
    if not match:
        raise AssertionError(f"no readback line (exit {result.returncode}):\n{out}\n{result.stderr}")
    return result.returncode, int(match[1], 16), int(match[2])


def draw(dxc, runner, adapter):
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    decoded = decode(CODE)
    with tempfile.TemporaryDirectory(prefix="mesh-heap-sampler-", dir=out) as directory:
        tmp = Path(directory)

        def compile_shader(stem, text, profile):
            (tmp / f"{stem}.hlsl").write_text(text, encoding="ascii")
            subprocess.run([dxc, "-WX", "-T", profile, "-E", "main", "-Fo", tmp / f"{stem}.dxil",
                            tmp / f"{stem}.hlsl"], check=True)
            return tmp / f"{stem}.dxil"

        hlsl = lower(decoded, 1, LIMITS, HEAP_SGPR)
        pixel = compile_shader("pixel", PIXEL, "ps_6_0")
        mesh = compile_shader("mesh", hlsl, "ms_6_5")
        default = compile_shader("default", re.sub(r"qh\[\d+\]", "qh[0]", hlsl), "ms_6_5")
        with patch.object(mesh_dynamic_aot.RuntimeWave, "proven_offset", lambda self, value, token: 0):
            unguarded = compile_shader("unguarded", lower(unbounded(decoded), 1, LIMITS, HEAP_SGPR), "ms_6_5")
        swapped = list(ENTRIES)
        swapped[1], swapped[2] = swapped[2], swapped[1]
        path = tmp / "draw.bin"
        cases = [(f"word {w:#x}", mesh, w, ENTRIES, expected(w), 0, pack(expected(w))) for w in WORDS]
        cases += [(f"swapped entries, word {w:#x}", mesh, w, swapped, expected(w), 1, pack(expected(w, swapped)))
                  for w in (0x10, 0x20)]
        cases += [(f"default sampler shortcut, word {w:#x}", default, w, ENTRIES, expected(w), 1,
                   pack(sample(ENTRIES[0]))) for w in WORDS]
        cases += [("bounded-offset guard removed, word 0x30", unguarded, 0x30, ENTRIES, expected(0x30), 1,
                   pack(sample(ENTRIES[0])))]
        for name, shader, word, entries, color, code, first in cases:
            result = run_draw(runner, adapter, shader, pixel, path, word, entries, color)
            if result is None:
                print("mesh_shader_tier=unsupported")
                return 77
            exit_code, actual, mismatches = result
            ok = (exit_code == 0) == (code == 0) and actual == first and mismatches == (1024 if code else 0)
            print(f"{name}: exit={exit_code} first={actual:08x} mismatches={mismatches}"
                  f" expected_first={first:08x} {'ok' if ok else 'UNEXPECTED'}", flush=True)
            if not ok:
                return 1
    return 0


def main(argv):
    decoded = decode(CODE)
    if len(argv) == 4 and argv[0] == "binary":
        reassemble(*argv[1:])
    elif len(argv) == 4 and argv[0] == "draw":
        return draw(*argv[1:])
    elif argv:
        raise SystemExit(__doc__)
    check_translation(decoded)
    check_refusals(decoded)
    guard_mutants(decoded)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
