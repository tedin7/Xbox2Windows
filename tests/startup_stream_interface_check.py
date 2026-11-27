"""A slot-86 pixel shader reads the vertex output its interface names, matched by semantic, not by number.

Source-built gfx1013 code only. The vertex shader computes a full-screen triangle from the vertex id and exports
param0 = COLOR, the constant 0.5, and param1 = TEXCOORD, u = (x + 1) / 2. Its container's interface table lists
COLOR (output register 1) as parameter 0 and TEXCOORD (register 2) as parameter 1. The pixel shader declares COLOR
and TEXCOORD but reads only TEXCOORD; its interface table lists TEXCOORD as attribute 0, which it interpolates and
writes to red.

Without arguments only the translator cases run (no consumer, device or compiler):
  linked:   attribute 0 reads vertex parameter 1; an interface whose order matches keeps parameter 0.
  refused:  a packed pixel record, a vertex record with packing bytes, components that differ between the stages,
            a semantic the vertex stage does not output, a pixel header of another size, and an attribute read that
            the pixel table does not list.
With a consumer and a backend they run first, then the draw on an 8x4 target (left half drawn):
  draw:     red follows u = (x + 0.5) / 8 per column (16, 48, 80, 112, each within 1).
  numbered: the same pipeline translated with pairing by number (a mutation) reads COLOR, a uniform 128, and fails.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import vop1, vop2

WIDTH, HEIGHT = 8, 4
VS_BASE, PS_BASE = 0x2f8, 0xef8
POSITION = ("SV_POSITION", 0, 0, 1, 0xf)          # (semantic, index, register, system value, mask)
COLOR = ("COLOR", 0, 1, 0, 0xf)
TEXCOORD = ("TEXCOORD", 0, 2, 0, 0x3)
VS_RECORDS = [(6, 1, 0, 0, 0, 0), (6, 2, 1, 0, 0, 0)]  # COLOR -> param0, TEXCOORD -> param1
PS_RECORDS = [(6, 2, 3, 0, 0, 0, 2, 2)]                # TEXCOORD -> attr0, mask 3, linear


def vertex_words():
    # x = 2 * ((2*i) & 2) - 1, y = 2 * (i & 2) - 1 (startup_stream_draw_check.vertex), v3 = 0, v4 = 1.0.
    words = [vop2(0x1a, 1, 129, 0), vop2(0x1b, 1, 130, 1), vop2(0x1b, 2, 130, 0)]
    for v in (1, 2):
        words.extend([vop1(6, v, 256 + v), vop2(8, v, 244, v), vop2(3, v, 243, v)])
    words.extend([vop1(1, 3, 128), vop1(1, 4, 242),
                  vop2(3, 5, 242, 1), vop2(8, 5, 240, 5),   # v5 = (x + 1.0) * 0.5
                  vop1(1, 6, 240),                           # v6 = 0.5
                  0xf80000cf, 0x04030201,                    # exp pos0 v1, v2, v3, v4
                  0xf800020f, 0x06060606,                    # exp param0 v6, v6, v6, v6 (COLOR)
                  0xf8000a1f, 0x06060505,                    # exp param1 v5, v5, v6, v6 done (TEXCOORD)
                  0xbf810000])
    return words


def pixel_words(attr=0):
    # v2 = interpolated attr<attr>.x; exp mrt0 v2, 0, 0, 1.0 done vm.
    return [0xc8080000 | attr << 10, 0xc8090001 | attr << 10, vop1(1, 3, 128), vop1(1, 4, 242),
            0xf800180f, 0x04030302, 0xbf810000]


def signature(tag, entries):
    names, body = b"", b""
    first = 8 + 32 * len(entries)
    for name, index, register, system, mask in entries:
        body += struct.pack("<6IBBHI", 0, first + len(names), index, system, 3, register, mask, mask, 0, 0)
        names += name.encode("ascii") + b"\0"
    data = struct.pack("<II", len(entries), 8) + body + names
    return tag + struct.pack("<I", len(data)) + data


def container(stage, words, base, sections, table):
    """A container with an FSSM header of `base` bytes; `table` maps header offsets to bytes."""
    code = struct.pack("<%dI" % len(words), *words)
    extent = (40 + len(code) + 4 + 15) & ~15
    image = bytearray(base + extent)
    image[:4] = b"FSSM"
    struct.pack_into("<I", image, 4, 0x101)
    struct.pack_into("<QQ", image, 192, base, extent)
    struct.pack_into("<Q", image, 232, len(image))
    for offset, data in table.items():
        image[offset:offset + len(data)] = data
    image[base + 40:base + 40 + len(code)] = code
    padding = (len(image) - base - 40 - len(code)) // 4
    image[base + 40 + len(code):] = struct.pack("<I", 0xbf9f0000) * padding
    native = b"XBCH" + bytes(12) + struct.pack("<II", len(image), 0) + zlib.compress(bytes(image))
    program = struct.pack("<HH", 0, stage) + bytes(20) + b"XDAT" + struct.pack("<5I", 1, 0, 13, 48, len(native)) + native
    parts = sections + [b"DXIL" + struct.pack("<I", len(program)) + program]
    offsets, at = [], 32 + 4 * len(parts)
    for part in parts:
        offsets.append(at)
        at += len(part)
    return b"DXBC" + bytes(16) + struct.pack("<3I", 1, at, len(parts)) + struct.pack("<%dI" % len(parts), *offsets) + b"".join(parts)


def vertex(records=VS_RECORDS, outputs=(POSITION, COLOR, TEXCOORD), base=VS_BASE):
    table = {0x1dc: b"".join(bytes(r) for r in records), 0x29c: struct.pack("<I", len(records))}
    return container(1, vertex_words(), base, [signature(b"OSG1", outputs)], table)


def pixel(records=PS_RECORDS, inputs=(POSITION, COLOR, TEXCOORD), base=PS_BASE, attr=0, count=None):
    table = {0x138: b"".join(bytes(r) for r in records), 0x238: struct.pack("<I", len(records) if count is None else count)}
    return container(0, pixel_words(attr), base, [signature(b"ISG1", inputs)] if inputs else [], table)


EMPTY_ROOT = (b"DXBC" + bytes(16) + struct.pack("<4I", 1, 68, 1, 36) + b"RTS0" + struct.pack("<I", 24)
              + struct.pack("<6I", 1, 0, 24, 0, 24, 0))


def translate(directory, vs, ps, root=EMPTY_ROOT):
    """Publish a request and translate it without compiling: (status, ps.hlsl)."""
    directory.mkdir(parents=True)
    key = gfx2hlsl.pipeline_key(vs, ps, root)
    for ext, data in ((".a", vs), (".b", ps), (".root", root)):
        (directory / ("req_%016x%s" % (key, ext))).write_bytes(data)
    status = gfx2hlsl.translate_request(str(directory), key)
    hlsl = directory / ("%016x.ps.hlsl" % key)
    return status, hlsl.read_text() if hlsl.exists() else ""


def translate_cases(directory):
    status, ps = translate(directory / "linked", vertex(), pixel())
    assert status == "ok" and "U(i.param1.x)" in ps and "i.param0" not in ps, (status, ps)
    print("translate: attribute 0 reads vertex parameter 1 (TEXCOORD)")
    status, ps = translate(directory / "ordered", vertex(records=[(6, 2, 0, 0, 0, 0), (6, 1, 1, 0, 0, 0)]), pixel())
    assert status == "ok" and "U(i.param0.x)" in ps, (status, ps)
    print("translate: with TEXCOORD as parameter 0, attribute 0 reads parameter 0")
    # The semantic index counts: the pixel shader reads TEXCOORD1, which the vertex shader exports as parameter 0.
    status, ps = translate(directory / "semantic_index",
                           vertex(outputs=(POSITION, ("TEXCOORD", 1, 1, 0, 0x3), ("TEXCOORD", 0, 2, 0, 0x3))),
                           pixel(inputs=(POSITION, ("TEXCOORD", 0, 1, 0, 0x3), ("TEXCOORD", 1, 2, 0, 0x3))))
    assert status == "ok" and "U(i.param0.x)" in ps, (status, ps)
    print("translate: TEXCOORD1 links by semantic index, not by name alone")
    for label, vs, ps, want in (
            ("packed pixel record", vertex(), pixel(records=[(6, 2, 3, 0, 0, 1, 2, 26)]),
             "unverified pixel input record 0602030000010" + "21a"),
            ("vertex record packing", vertex(records=[(6, 1, 0, 0, 0, 0), (6, 2, 1, 3, 0, 3)]), pixel(),
             "unverified vertex output record 060201030003"),
            ("different components", vertex(outputs=(POSITION, COLOR, ("TEXCOORD", 0, 2, 0, 0xf))), pixel(),
             "pixel input TEXCOORD0 has no vertex output with the same components"),
            ("semantic not output", vertex(records=[(6, 1, 0, 0, 0, 0)], outputs=(POSITION, COLOR)), pixel(),
             "pixel input TEXCOORD0 has no vertex output with the same components"),
            ("pixel header size", vertex(), pixel(base=0x2f8), "unverified FSSM header for the stage interface"),
            ("attribute beyond table", vertex(), pixel(attr=1),
             "pixel stage reads attribute 1, which its interface table does not list"),
            ("attribute component missing", vertex(outputs=(POSITION, COLOR, ("TEXCOORD", 0, 2, 0, 2))),
             pixel(inputs=(POSITION, COLOR, ("TEXCOORD", 0, 2, 0, 2)), records=[(6, 2, 2, 0, 0, 0, 2, 2)]),
             "pixel stage reads undeclared component x of attribute 0"),
            ("duplicate vertex semantic", vertex(outputs=(POSITION, ("TEXCOORD", 0, 1, 0, 3), TEXCOORD)),
             pixel(), "duplicate vertex output semantic TEXCOORD0"),
            ("pixel table too large", vertex(), pixel(count=33), "pixel input table exceeds its header range"),
            ("record mask differs", vertex(), pixel(records=[(6, 2, 0xf, 0, 0, 0, 2, 2)]),
             "pixel input record mask 0xf differs from the signature mask 0x3 of register 2"),
            ("two vertex sources", vertex(outputs=(POSITION, ("TEXCOORD", 0, 1, 0, 0x3), ("TEXCOORD", 1, 2, 0, 0xc))),
             pixel(inputs=(POSITION, ("TEXCOORD", 0, 2, 0, 0x3), ("TEXCOORD", 1, 2, 0, 0xc)),
                   records=[(6, 2, 0xf, 0, 0, 0, 2, 2)]),
             "pixel attribute 0 does not come from exactly one exported vertex register"),
            ("duplicate pixel register", vertex(), pixel(records=[PS_RECORDS[0], (6, 2, 3, 1, 0, 0, 2, 2)]),
             "unverified pixel input record 0602030100000202"),
            ("vertex header size", vertex(base=0x2f0), pixel(), "unverified FSSM header for the stage interface"),
            ("pixel count 0", vertex(), pixel(count=0),
             "pixel stage reads attribute 0, which its interface table does not list"),
            ("no pixel signature", vertex(), pixel(inputs=()),
             "pixel stage reads attribute 0, which its interface table does not list")):
        status, _ = translate(directory / label.replace(" ", "_"), vs, ps)
        assert status == "unsupported: " + want, (label, status)
        print(f"translate: {label} refused")
    return 0


def translate_checks(directory):
    saved = os.environ.get("X2W_FXC")
    os.environ["X2W_FXC"] = ""
    try:
        return translate_cases(directory)
    finally:
        if saved is None:
            os.environ.pop("X2W_FXC", None)
        else:
            os.environ["X2W_FXC"] = saved


def expected_red(column):
    return round((column + 0.5) / 8 * 255)


def check_pixels(pixels):
    """None when the left half follows the TEXCOORD gradient (each red within 1) and the right half keeps the clear."""
    for y in range(HEIGHT):
        for x in range(WIDTH):
            r, g, b, a = pixels[(y * WIDTH + x) * 4:(y * WIDTH + x) * 4 + 4]
            want = (expected_red(x), 0, 0, 255) if x < WIDTH // 2 else (0, 0, 255, 255)
            if abs(r - want[0]) > 1 or (g, b, a) != want[1:]:
                return "pixel (%d, %d) is %s, expected %s" % (x, y, (r, g, b, a), want)
    return None


def draw(directory, consumer, backend, numbered, *, vertex_blob=None, pixel_blob=None, full_target=False, entry_word=None):
    cache = directory / "cache"
    cache.mkdir(parents=True)
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex() if vertex_blob is None else vertex_blob)
    ps.write_bytes(pixel() if pixel_blob is None else pixel_blob)
    args = [consumer, vs, ps, root]
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache))
    env.setdefault("X2W_D3D12_DEBUG", "0")
    if entry_word is not None:
        env["X2W_TEST_ENTRY_WORD"] = str(entry_word)
    if full_target:
        env["X2W_TEST_FULL_TARGET"] = "1"
    subprocess.run(args, env=env, check=True)
    key = int(sorted(cache.glob("req_*.a"))[0].name[4:20], 16)
    original = gfx2hlsl.stream_interface
    if numbered:
        gfx2hlsl.stream_interface = lambda vs, ps: None
    try:
        status = gfx2hlsl.translate_request(str(cache), key)
    finally:
        gfx2hlsl.stream_interface = original
    assert status == "ok", status
    result = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if backend == "hardware" and result.returncode == 77:
        return None
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"draw failed on {backend}: {result.returncode}\n{result.stderr}")
    return output.read_bytes()


def check(directory, consumer, backend):
    pixels = draw(directory / "linked", consumer, backend, False)
    if pixels is None:
        print("No D3D12 feature-level 12 hardware adapter")
        return 77
    difference = check_pixels(pixels)
    if difference:
        raise AssertionError("linked: " + difference)
    print(f"{backend} draw: red follows TEXCOORD, {[expected_red(x) for x in range(WIDTH // 2)]} within 1")
    pixels = draw(directory / "numbered", consumer, backend, True)
    if not check_pixels(pixels):
        raise AssertionError("pairing by number passed the gradient check")
    print(f"{backend} pairing by number: the gradient check fails (red {pixels[0]} in column 0)")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        result = translate_checks(Path(temp) / "translate")
        if len(sys.argv) > 1:
            result = check(Path(temp), *sys.argv[1:])
        sys.exit(result)
