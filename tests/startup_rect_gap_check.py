"""The eight-vertex quad-and-edge-ring draw of #3131, with a 12-byte vertex record and XY at +4.

Source-built gfx1013 code only. The vertex stage calls the driver's fetch routine: two records, POSITION
(register 0, mask 0x7) read as R32G32_FLOAT at +4 of a 12-byte record and TEXCOORD (register 1, mask 0x1)
read as R32_FLOAT at +0. The stream's input layout therefore names slot 0 TEXCOORD R32_FLOAT at offset 0 and
slot 1 POSITION R32G32_FLOAT at offset 4, and both views name the same records with a 12-byte stride.

Records 0-2 are the edge ring that encloses the quad (mask 0) and 3-5 the inner quad (mask 1); 6 and 7 are
their fourth corners, so a rectangle list of six vertices and an indexed triangle list built from the same
bytes describe the same four triangles. The ring is drawn first, so the quad's diagonal stays visible. Two
pixel shaders read the same draw:
  mask:     interpolates TEXCOORD (the mask) and paints the quad red and the ring green, so every pixel
            says which region covered it and a missing triangle leaves the clear colour.
  constant: exports opaque red without reading an input, so a gap here would be a coverage gap and a gap
            only in the mask picture would be an attribute or interpolation gap.

Draws, each read back as 32 RGBA pixels:
  triangle       R16 indexed triangle list, ring then quad
  triangle-copy  the same draw from a stream and index view no placed buffer holds: the recording-time
                 vertex read and the index scan size what is copied
  rect           non-indexed rectangle list of six vertices
  rect8          the eight-vertex rectangle list (the last two vertices are not part of a rectangle)
  rect8-copy     the eight-vertex rectangle list from an unregistered stream
  none           no draw: the cleared target
The rect modes use the six-vertex expansion the vertex stage applies to a rectangle list. With
X2W_TEST_RASTERIZER=back the same picture must come back with back-face culling on, and =back-ccw
(counter-clockwise front faces) must cull the whole draw: that is the winding control.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib
import math

from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import vop1, vop2

WIDTH, HEIGHT = 8, 4
BLUE, RED, GREEN = bytes((0, 0, 255, 255)), bytes((255, 0, 0, 255)), bytes((0, 255, 0, 255))
VS_BASE, PS_BASE = 0x2f8, 0xef8
FETCH_CALL, PREFETCH, WAIT_VM, ENDPGM = 0xbe842104, 0xbfa00003, 0xbf8c3f70, 0xbf810000

# mask, x, y per 12-byte record. Each rectangle is three corners clockwise on screen; 6 and 7 hold the
# fourth corner v0 + v2 - v1, which the indexed list names and the rectangle list derives.
VERTICES = [(0.0, -1.0, -1.0), (0.0, -1.0, 1.0), (0.0, 1.0, 1.0),
            (1.0, -0.75, -0.625), (1.0, -0.75, 0.625), (1.0, 0.75, 0.625),
            (0.0, 1.0, -1.0), (1.0, 0.75, -0.625)]
INDICES = [0, 1, 2, 0, 2, 6, 3, 4, 5, 3, 5, 7]
TRIANGLES = [(0, 1, 2), (0, 2, 6), (3, 4, 5), (3, 5, 7)]
ROTATED = math.radians(15.0)
LAYOUT_TEXT = "ia1 2\nTEXCOORD 0 41 0 0 0 0\nPOSITION 0 16 1 4 0 0\n"
LAYOUT_LINE = "x2w-draw stream input layout: 2 elements (format/slot/offset/class) 41/0/0/0 16/1/4/0"


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
    return b"DXBC" + bytes(16) + struct.pack("<3I", 1, at, len(parts)) + struct.pack("<%dI" % len(parts), *offsets) \
        + b"".join(parts)


def vertex():
    # Fetch POSITION into v4-v6 (mask 0x7, count 3) and TEXCOORD into v8; export pos0 from v4-v7 and
    # param0 from v8.
    records = [(6, 0, 0x7, 4, 2), (6, 1, 0x1, 8, 0)]
    words = [FETCH_CALL, PREFETCH, vop1(1, 7, 242), WAIT_VM,
             0xf80008cf, 0x07060504,       # exp pos0 v4, v5, v6, v7
             0xf800020f, 0x08080808,       # exp param0 v8, v8, v8, v8
             ENDPGM]
    table = {0x138: b"".join(bytes(r) for r in records), 0x1d8: struct.pack("<I", len(records)),
             0x1dc: bytes((6, 1, 0, 0, 0, 0)), 0x29c: struct.pack("<I", 1)}
    outputs = [("SV_POSITION", 0, 0, 1, 0xf), ("TEXCOORD", 0, 1, 0, 0x1)]
    return container(1, words, VS_BASE, [signature(b"ISG1", [("POSITION", 0, 0, 0, 0x7), ("TEXCOORD", 0, 1, 0, 0x1)]),
                                        signature(b"OSG1", outputs)], table)


def pixel(mask):
    if not mask:
        words = [vop1(1, 0, 128), vop1(1, 1, 242), 0xf800180f, 0x01000001, ENDPGM]  # opaque red
        return container(0, words, PS_BASE, [signature(b"ISG1", [("SV_POSITION", 0, 0, 1, 0xf)])],
                         {0x238: struct.pack("<I", 0)})
    # v2 = interpolated TEXCOORD.x (the mask); vcc = 0.5 > mask; red where the mask is 1, green where it is 0.
    words = [0xc8080000, 0xc8090001, vop1(1, 0, 128), vop1(1, 1, 242),
             0x7c0804f0,                    # v_cmp_gt_f32 vcc_lo, 0.5, v2
             vop2(1, 3, 256 + 1, 0),        # v3 = vcc ? v0 : v1  (red)
             vop2(1, 4, 256 + 0, 1),        # v4 = vcc ? v1 : v0  (green)
             0xf800180f, 0x01000403,        # exp mrt0 v3, v4, v0, v1 done vm
             ENDPGM]
    inputs = [("SV_POSITION", 0, 0, 1, 0xf), ("TEXCOORD", 0, 2, 0, 0x1)]
    return container(0, words, PS_BASE, [signature(b"ISG1", inputs)],
                     {0x138: bytes((6, 2, 0x1, 0, 0, 0, 2, 2)), 0x238: struct.pack("<I", 1)})


EMPTY_ROOT = (b"DXBC" + bytes(16) + struct.pack("<4I", 1, 68, 1, 36) + b"RTS0" + struct.pack("<I", 24)
              + struct.pack("<6I", 1, 0, 24, 0, 24, 0))


def cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def placed(rotated):
    """The vertex records a mode writes: the rotation the fixture applies to a rect-* / triangle-* run."""
    if not rotated:
        return VERTICES
    c, s = math.cos(ROTATED), math.sin(ROTATED)
    return [(m, c * x - s * y, s * x + c * y) for m, x, y in VERTICES]


def region(x, y, rotated=False):
    """The last triangle in draw order that holds pixel centre (x, y), or None. Asserts no centre is on an edge."""
    records = placed(rotated)
    p = ((2.0 * x + 1.0) / WIDTH - 1.0, 1.0 - (2.0 * y + 1.0) / HEIGHT)
    hit = None
    for triangle in TRIANGLES:
        points = [records[i][1:] for i in triangle]
        edges = [cross(points[i], points[(i + 1) % 3], p) for i in range(3)]
        assert min(abs(e) for e in edges) > 1e-3, f"pixel ({x}, {y}) lies on an edge of {triangle}"
        if not (any(e < 0 for e in edges) and any(e > 0 for e in edges)):
            hit = "quad" if triangle[0] >= 3 else "ring"
    return hit


def expected(mask, rotated=False):
    out = bytearray()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            covered = region(x, y, rotated)
            out += (BLUE if covered is None else RED if (covered == "quad" or not mask) else GREEN)
    return bytes(out)


def describe(image):
    """Counts of each colour, so a failure says which region moved."""
    names = {BLUE: "clear", RED: "red", GREEN: "green"}
    counts = {}
    for i in range(0, len(image), 4):
        name = names.get(image[i:i + 4], image[i:i + 4].hex())
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{n} {c}" for n, c in sorted(counts.items()))


def first_difference(actual, want):
    for i in range(0, len(want), 4):
        if actual[i:i + 4] != want[i:i + 4]:
            return "pixel (%d, %d) is %s, expected %s" % (i // 4 % WIDTH, i // 4 // WIDTH,
                                                          actual[i:i + 4].hex(), want[i:i + 4].hex())
    return None


class Run:
    def __init__(self, directory, consumer, mode, mask, rasterizer=None):
        directory.mkdir(parents=True)
        self.cache = directory / "cache"
        self.cache.mkdir()
        self.output = directory / "pixels"
        self.bound = directory / "bound"
        vs, ps, root = directory / "vs", directory / "ps", directory / "root"
        vs.write_bytes(vertex())
        ps.write_bytes(pixel(mask))
        self.args = [consumer, str(vs), str(ps), str(root)]
        self.env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
                        X2W_SHADER_CACHE=str(self.cache), X2W_TEST_INPUT_LAYOUT="gradient",
                        X2W_TEST_RECT_GAP=mode, X2W_TEST_RECT_GAP_BYTES=str(self.bound))
        self.env.setdefault("X2W_D3D12_DEBUG", "0")
        if rasterizer:
            self.env["X2W_TEST_RASTERIZER"] = rasterizer
        else:
            self.env.pop("X2W_TEST_RASTERIZER", None)
        subprocess.run(self.args, env=self.env, check=True)
        self.key = None
        if mode != "none":
            requests = sorted(self.cache.glob("req_*.a"))
            assert len(requests) == 1, requests
            self.key = int(requests[0].name[4:20], 16)

    def translate(self):
        status = gfx2hlsl.translate_request(str(self.cache), self.key)
        assert status == "ok", status
        return self.cache / ("%016x.vs.hlsl" % self.key)

    def draw(self, backend):
        result = subprocess.run(self.args + [backend, str(self.output)], env=self.env, capture_output=True, text=True)
        for line in result.stderr.splitlines():
            if line.startswith("adapter=") or "debug layer:" in line:
                print("  " + line)
        return result


def check(directory, consumer, backend, rasterizer=None):
    # The clear colour, before any draw.
    plain = Run(directory / "none", consumer, "none", True)
    result = plain.draw(backend)
    if result.returncode == 77:
        print("No D3D12 feature-level 12 adapter")
        return 77
    if result.returncode:
        raise RuntimeError(f"none failed on {backend}: {result.returncode}\n{result.stderr}")
    if plain.output.read_bytes() != BLUE * (WIDTH * HEIGHT):
        raise AssertionError("the cleared target is not the expected blue: " + describe(plain.output.read_bytes()))
    print(f"{backend} none: 32 clear pixels, pre-draw state exact")

    for mode, mask in (("triangle", True), ("triangle", False), ("triangle-copy", True),
                       ("rect", True), ("rect8", True), ("rect8-copy", True), ("rect8", False),
                       ("triangle-rotated", True), ("rect-rotated", True)):
        label = f"{mode} {'mask' if mask else 'constant'}"
        rotated = "rotated" in mode
        run = Run(directory / (label.replace(" ", "_") + ("_" + rasterizer if rasterizer else "")), consumer,
                  mode, mask, rasterizer)
        hlsl = run.translate()
        text = hlsl.read_text()
        layout = (run.cache / ("req_%016x.ia" % run.key)).read_bytes()
        assert layout == LAYOUT_TEXT.encode(), layout
        # POSITION (slot 1, R32G32_FLOAT at +4) and the mask (slot 0, R32_FLOAT at +0), each scaled by its
        # own stream stride.
        for line in ("a = (vid + ud1) * ud3 + 4;", "a = (vid + ud1) * ud4 + 0;"):
            assert "    " + line + "\n" in text, line + "\n" + text
        if mode.startswith("rect"):
            for line in ("uint first = (vid / 6) * 3, corner = vid % 6;", "o.pos = p.pos + r.pos - q.pos;"):
                assert line in text, line + "\n" + text
        result = run.draw(backend)
        if result.returncode == 77:
            print("No D3D12 feature-level 12 adapter")
            return 77
        if result.returncode or "-> drawn;" not in result.stderr:
            raise RuntimeError(f"{label} failed on {backend}: {result.returncode}\n{result.stderr}")
        if LAYOUT_LINE not in result.stderr:
            raise AssertionError("missing layout diagnostic:\n" + result.stderr)
        bound = run.bound.read_bytes()
        records = placed(rotated)
        got = struct.unpack("<24f", bound[:96])
        for i, (mask_value, x, y) in enumerate(records):
            for name, actual, want in (("mask", got[3 * i], mask_value), ("x", got[3 * i + 1], x),
                                       ("y", got[3 * i + 2], y)):
                if abs(actual - want) > 1e-6:
                    raise AssertionError(f"{label}: record {i} {name} is {actual}, expected {want}")
        if "triangle" in mode and bound[96:] != struct.pack("<%dH" % len(INDICES), *INDICES):
            raise AssertionError(f"{label}: bound indices {bound[96:].hex()}")
        actual = run.output.read_bytes()
        want = expected(mask, rotated)
        difference = first_difference(actual, want)
        if difference:
            raise AssertionError(f"{label}: {difference} ({describe(actual)})")
        print(f"{backend} {label}: 32 pixels exact ({describe(actual)}); 8 records of 12 bytes bound")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        args = sys.argv[1:]
        rasterizer = args[2] if len(args) > 2 else None
        sys.exit(check(Path(temp), *args[:2], rasterizer=rasterizer))
