"""A slot-86 vertex shader that starts by calling the driver's fetch routine draws from the stream's input layout.

Source-built gfx1013 code only. The vertex shader begins with s_swappc_b64 s[4:5], s[4:5], then s_inst_prefetch, and
exports pos0 from v4-v7 and param0 from v8-v10. Its container lists two fetch records (input register 0 -> v4,
register 1 -> v8, three components each; the header counts 2) and its input signature names them POSITION and
TEXCOORD. The stream's input layout declares TEXCOORD in slot 0 and POSITION in slot 1
(D3D12_APPEND_ALIGNED_ELEMENT), both float3.

Two draws on an 8x4 target cleared to blue with a red pixel shader (tests/startup_stream_draw.cpp): one from start
vertex 3, one indexed (indices 4-9, base vertex 5), so the fetch index must include the base-vertex entry value.
  fetch:     POSITION covers pixels [1, 3) x [1, 3) and column 0; exactly those are red.
  swapped:   the generated fetch with the two records' registers exchanged (a mutation) draws the TEXCOORD
             rectangles, row 0 of [0, 4) and pixel (3, 3), so the exact check above fails.
  one slot:  the generated fetch reading both elements from slot 0 (a mutation) also draws the TEXCOORD rectangles.
  no layout: the same stream without the INPUT_LAYOUT subobject is refused by the translator and not drawn.

Without arguments only the translator cases run (no consumer, device or compiler); with a consumer and a backend
they run first, then the draws.
  fetch text: the generated loads and manifest entries for both slots.
  refused:    no layout; an export of v12 that nothing defines; v12 written only with exec masked off, or only in
              code a branch jumps over; a z-only record whose count stops below z (count 3 translates); a fetched
              element in slot 8 (the stand-in binds 0 to 7), per-instance or of format 10; a record of kind 7 or
              reaching past v255; a header record count that disagrees with the table.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vop1

WIDTH, HEIGHT = 8, 4
RED, BLUE = bytes((255, 0, 0, 255)), bytes((0, 0, 255, 255))
SWAPPC, PREFETCH, WAIT_VM, ENDPGM = 0xbe842104, 0xbfa00003, 0xbf8c3f70, 0xbf810000
RECORDS = [(6, 0, 0x7, 4, 2), (6, 1, 0x7, 8, 2)]
INPUTS = [("POSITION", 0, 0x7), ("TEXCOORD", 1, 0x7)]
LAYOUT_TEXT = "ia1 2\nTEXCOORD 0 6 0 0 0 0\nPOSITION 0 6 1 4294967295 0 0\n"
LAYOUT_LINE = "x2w-draw stream input layout: 2 elements (format/slot/offset/class) 6/0/0/0 6/1/4294967295/0"


def vertex(param_first=8, records=RECORDS):
    # exp pos0 v4, v5, v6, v7 done; exp param0 v<first>, v<first+1>, v<first+2>, v7.
    words = [SWAPPC, PREFETCH, vop1(1, 7, 242), WAIT_VM, 0xf80008cf, 0x07060504,
             0xf800020f, param_first | (param_first + 1) << 8 | (param_first + 2) << 16 | 7 << 24, ENDPGM]
    return fetch_container(words, records)


def fetch_container(words, records, header_count=None):
    """A vertex container with an FSSM header of 0x2f8 bytes (fetch records at +0x138) and an ISG1 part."""
    code = struct.pack("<%dI" % len(words), *words)
    base = 0x2f8
    extent = (40 + len(code) + 4 + 15) & ~15
    image = bytearray(base + extent)
    image[:4] = b"FSSM"
    struct.pack_into("<I", image, 4, 0x101)
    struct.pack_into("<QQ", image, 192, base, extent)
    struct.pack_into("<Q", image, 232, len(image))
    for i, record in enumerate(records):
        image[0x138 + 5 * i:0x13d + 5 * i] = bytes(record)
    struct.pack_into("<I", image, 0x1d8, len(records) if header_count is None else header_count)
    image[base + 40:base + 40 + len(code)] = code
    padding = (len(image) - base - 40 - len(code)) // 4
    image[base + 40 + len(code):] = struct.pack("<I", 0xbf9f0000) * padding
    native = b"XBCH" + bytes(12) + struct.pack("<II", len(image), 0) + zlib.compress(bytes(image))
    program = (struct.pack("<HH", 0, 1) + bytes(20) + b"XDAT" + struct.pack("<5I", 1, 0, 13, 48, len(native)) + native)
    names = b""
    entries = b""
    first = 8 + 32 * len(INPUTS)
    for name, register, mask in INPUTS:
        entries += struct.pack("<6IBBHI", 0, first + len(names), 0, 0, 3, register, mask, mask, 0, 0)
        names += name.encode("ascii") + b"\0"
    signature = struct.pack("<II", len(INPUTS), 8) + entries + names
    sections = [b"ISG1" + struct.pack("<I", len(signature)) + signature, b"DXIL" + struct.pack("<I", len(program)) + program]
    table = 32 + 4 * len(sections)
    offsets, at = [], table
    for section in sections:
        offsets.append(at)
        at += len(section)
    header = b"DXBC" + bytes(16) + struct.pack("<3I", 1, at, len(sections)) + struct.pack("<%dI" % len(offsets), *offsets)
    return header + b"".join(sections)


def expected(inside):
    return b"".join(RED if inside(x, y) else BLUE for y in range(HEIGHT) for x in range(WIDTH))


def first_difference(actual, want):
    for index in range(0, len(want), 4):
        if actual[index:index + 4] != want[index:index + 4]:
            return "pixel (%d, %d) is %s, expected %s" % (index // 4 % WIDTH, index // 4 // WIDTH,
                                                         actual[index:index + 4].hex(), want[index:index + 4].hex())
    return None


FETCHED = expected(lambda x, y: (1 <= x < 3 and 1 <= y < 3) or x == 0)
SWAPPED = expected(lambda x, y: (0 <= x < 4 and y < 1) or (x, y) == (3, 3))


class Run:
    def __init__(self, directory, consumer, backend, vs_bytes, layout, zero_fill=False):
        self.cache = directory / "cache"
        self.cache.mkdir(parents=True)
        self.vs, ps, self.root, self.output = (directory / name for name in ("vs", "ps", "root", "pixels"))
        self.vs.write_bytes(vs_bytes)
        ps.write_bytes(pixel(False))
        self.args = [consumer, self.vs, ps, self.root]
        self.backend = backend
        self.env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
                        X2W_SHADER_CACHE=str(self.cache), X2W_TEST_VERTEX_FETCH="1")
        self.env.setdefault("X2W_D3D12_DEBUG", "0")
        self.env.pop("X2W_STREAM_ZERO_FILL", None)
        if zero_fill:
            self.env["X2W_STREAM_ZERO_FILL"] = "1"
        if layout:
            self.env["X2W_TEST_INPUT_LAYOUT"] = layout if isinstance(layout, str) else "1"
        else:
            self.env.pop("X2W_TEST_INPUT_LAYOUT", None)
        subprocess.run(self.args, env=self.env, check=True)
        requests = sorted(self.cache.glob("req_*.a"))
        assert len(requests) == 1, requests
        self.key = int(requests[0].name[4:20], 16)

    def draw(self):
        result = subprocess.run(self.args + [self.backend, self.output], env=self.env, capture_output=True, text=True)
        print(result.stdout, end="")
        return result


def check(directory, consumer, backend):
    # fetch: the request carries the layout and the readback is exact.
    run = Run(directory / "fetch", consumer, backend, vertex(), True)
    name = run.cache / ("req_%016x" % run.key)
    layout = (name.parent / (name.name + ".ia")).read_bytes()
    assert layout == LAYOUT_TEXT.encode(), layout
    assert run.key == gfx2hlsl.pipeline_key(run.vs.read_bytes(), (run.vs.parent / "ps").read_bytes(),
                                            run.root.read_bytes(), layout), "request key does not include the layout"
    status = gfx2hlsl.translate_request(str(run.cache), run.key)
    assert status == "ok", status
    result = run.draw()
    if backend == "hardware" and result.returncode == 77:
        print("No D3D12 feature-level 12 hardware adapter")
        return 77
    if result.returncode or result.stderr.count("-> drawn;") < 2:
        raise RuntimeError(f"fetch draws failed on {backend}: {result.returncode}\n{result.stderr}")
    if LAYOUT_LINE not in result.stderr:
        raise AssertionError("missing layout diagnostic:\n" + result.stderr)
    difference = first_difference(run.output.read_bytes(), FETCHED)
    if difference:
        raise AssertionError(f"fetch: {difference}")
    print(f"{backend} fetch: all {WIDTH * HEIGHT} pixels exact")

    # Mutations of the generated fetch: each must fail the exact check and draw the TEXCOORD rectangles instead.
    def exchange(fetch):
        fetch[0]["vgpr"], fetch[1]["vgpr"] = fetch[1]["vgpr"], fetch[0]["vgpr"]

    def one_slot(fetch):
        for element in fetch:
            element["slot"] = 0
    for label, mutate in (("swapped registers", exchange), ("one slot", one_slot)):
        mutated = Run(directory / label.replace(" ", "_"), consumer, backend, vertex(), True)
        original = gfx2hlsl.stream_fetch

        def patched(*args):
            fetch = original(*args)
            mutate(fetch)
            return fetch
        gfx2hlsl.stream_fetch = patched
        try:
            status = gfx2hlsl.translate_request(str(mutated.cache), mutated.key)
        finally:
            gfx2hlsl.stream_fetch = original
        assert status == "ok", status
        result = mutated.draw()
        if result.returncode or "-> drawn;" not in result.stderr:
            raise RuntimeError(f"{label} draw failed on {backend}: {result.returncode}\n{result.stderr}")
        actual = mutated.output.read_bytes()
        if not first_difference(actual, FETCHED):
            raise AssertionError(f"{label}: the mutated fetch passed the exact check")
        difference = first_difference(actual, SWAPPED)
        if difference:
            raise AssertionError(f"{label}: {difference}")
        print(f"{backend} {label}: the exact check fails; the TEXCOORD rectangles are drawn instead")

    # no layout: refused by the translator, so the runtime has no translation and draws nothing.
    bare = Run(directory / "bare", consumer, backend, vertex(), False)
    assert not list(bare.cache.glob("*.ia"))
    status = gfx2hlsl.translate_request(str(bare.cache), bare.key)
    assert status == "unsupported: " + NO_LAYOUT, status
    result = bare.draw()
    if "-> drawn;" in result.stderr:
        raise AssertionError("a vertex shader with a fetch call and no layout was drawn")
    print(f"{backend} no layout: refused and not drawn")

    # empty layout, production default: declared with no elements, nothing is published and the fetch is refused.
    refused = Run(directory / "empty_default", consumer, backend, vertex(), "empty")
    assert not list(refused.cache.glob("req_*.ia")), "an empty layout was published without the opt-in"
    result = refused.draw()
    if "-> drawn;" in result.stderr:
        raise AssertionError("an empty declared layout was drawn without X2W_STREAM_ZERO_FILL")
    print(f"{backend} empty layout, default: refused and not drawn")

    # empty layout, experimental opt-in X2W_STREAM_ZERO_FILL=1: every fetched input reads zero (an assumption, not a
    # measurement); the vertices collapse and nothing but the blue clear reaches the readback.
    empty = Run(directory / "empty", consumer, backend, vertex(), "empty", zero_fill=True)
    layout = (empty.cache / ("req_%016x.ia" % empty.key)).read_bytes()
    assert layout == b"ia1 0\n", layout
    status = gfx2hlsl.translate_request(str(empty.cache), empty.key)
    assert status == "ok", status
    result = empty.draw()
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"empty layout draw failed on {backend}: {result.returncode}\n{result.stderr}")
    difference = first_difference(empty.output.read_bytes(), expected(lambda x, y: False))
    if difference:
        raise AssertionError(f"empty layout: {difference}")
    print(f"{backend} empty layout: zero-fed vertices draw nothing, all {WIDTH * HEIGHT} pixels exact")
    return 0


# Translator-only cases: no consumer, no device, no shader compiler.
NO_LAYOUT = "vertex stage calls a fetch routine and the pipeline declares no input layout"
UNDEFINED = "vertex stage reads v12, which neither the fetch nor an unconditional write before the first branch defines"
EXEC_LO_0, EXEC_LO_ALL, BRANCH_OVER_ONE = 0xbefe0380, 0xbefe03c1, 0xbf820001  # s_mov_b32 exec_lo, 0 / -1; s_branch +1
WQM_EXEC_LO_0, WQM_EXEC_EXEC, EXEC_HI_0 = 0xbefe0980, 0xbefe0a7e, 0xbeff0380  # s_wqm_b32 exec_lo, 0; s_wqm_b64 exec, exec; s_mov_b32 exec_hi, 0
V7_ONE, V12_ONE = vop1(1, 7, 242), vop1(1, 12, 242)                         # v_mov_b32 v7 / v12, 1.0
EXPORTS_V12 = [0xf80008cf, 0x07060504, 0xf800020f, 0x070c0c0c, ENDPGM]      # pos0 v4-v7; param0 v12, v12, v12, v7
EMPTY_ROOT = (b"DXBC" + bytes(16) + struct.pack("<4I", 1, 68, 1, 36) + b"RTS0" + struct.pack("<I", 24)
              + struct.pack("<6I", 1, 0, 24, 0, 24, 0))


def translate(directory, vs, layout):
    """Publish a request as the stand-in would and translate it without compiling: (status, vs.hlsl, manifest)."""
    directory.mkdir(parents=True)
    ps = pixel(False)
    ia = layout.encode() if layout else b""
    key = gfx2hlsl.pipeline_key(vs, ps, EMPTY_ROOT, ia)
    files = {".a": vs, ".b": ps, ".root": EMPTY_ROOT}
    if ia:
        files[".ia"] = ia
    for ext, data in files.items():
        (directory / ("req_%016x%s" % (key, ext))).write_bytes(data)
    status = gfx2hlsl.translate_request(str(directory), key)
    result = [directory / ("%016x%s" % (key, ext)) for ext in (".vs.hlsl", ".txt")]
    hlsl, manifest = (p.read_text() if p.exists() else "" for p in result)
    return status, gfx2hlsl.ORIGIN.sub("", hlsl), manifest


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


def translate_cases(directory):
    # The generated fetch: POSITION (record 0, slot 1) into v4-v6 and TEXCOORD (record 1, slot 0) into v8-v10,
    # each at (vertex id + base-vertex entry value) * the slot's stride.
    status, vs, manifest = translate(directory / "fetch", vertex(), LAYOUT_TEXT)
    assert status == "ok", status
    for line in ("a = (vid + ud1) * ud3 + 0;", "v4 = r0.Load(a);", "v6 = r0.Load(a + 8);",
                 "a = (vid + ud1) * ud4 + 0;", "v8 = r1.Load(a);", "v10 = r1.Load(a + 8);"):
        assert "    %s\n" % line in vs, line + "\n" + vs
    for line in ("udsrc 1 basevertex", "udsrc 3 stride 0", "udsrc 4 stride 1", "srv 0 buf vb 1", "srv 1 buf vb 0"):
        assert line + "\n" in manifest, line + "\n" + manifest
    print("translate: fetch loads and manifest as expected")

    # R8G8B8A8_UINT contributes the first three raw bytes to this XYZ fetch record.
    layout = LAYOUT_TEXT.replace("POSITION 0 6 ", "POSITION 0 30 ")
    status, vs, _ = translate(directory / "rgba8_uint", vertex(), layout)
    assert status == "ok", status
    for line in ("w = ldu(r0, a);", "v4 = w & 255u;", "v5 = (w >> 8) & 255u;", "v6 = (w >> 16) & 255u;"):
        assert "    %s\n" % line in vs, line + "\n" + vs

    # A layout declared with no elements translates: no vertex-buffer load, every fetched register reads zero.
    status, vs, manifest = translate(directory / "empty_layout", vertex(), "ia1 0\n")
    assert status == "ok", status
    for line in ("v4 = 0u;", "v6 = 0u;", "v8 = 0u;", "v10 = 0u;"):
        assert "    %s\n" % line in vs, line + "\n" + vs
    assert "r0.Load(" not in vs and "buf vb" not in manifest, vs + manifest
    print("translate: an empty declared layout reads every fetched input as zero")

    # Refused: no layout; a register nothing defines; the only write masked off by exec, or bypassed by a branch.
    for label, words, layout, want in (
            ("no layout", None, None, NO_LAYOUT),
            ("unwritten", [SWAPPC, PREFETCH, V7_ONE, WAIT_VM] + EXPORTS_V12, LAYOUT_TEXT, UNDEFINED),
            ("masked write", [SWAPPC, PREFETCH, V7_ONE, EXEC_LO_0, V12_ONE, EXEC_LO_ALL, WAIT_VM] + EXPORTS_V12,
             LAYOUT_TEXT, UNDEFINED),
            ("bypassed write", [SWAPPC, PREFETCH, V7_ONE, BRANCH_OVER_ONE, V12_ONE, WAIT_VM] + EXPORTS_V12,
             LAYOUT_TEXT, UNDEFINED),
            # Whole-quad mode into exec from another source, and a write to exec_hi: refused, not a no-op.
            ("s_wqm into exec", [SWAPPC, PREFETCH, V7_ONE, WQM_EXEC_LO_0, V12_ONE, EXEC_LO_ALL, WAIT_VM] + EXPORTS_V12,
             LAYOUT_TEXT, "s_wqm_b32 exec_lo, 0 after a fetch call"),
            ("exec_hi write", [SWAPPC, PREFETCH, V7_ONE, EXEC_HI_0, V12_ONE, WAIT_VM] + EXPORTS_V12,
             LAYOUT_TEXT, "s_mov_b32 into exec_hi after a fetch call")):
        vs = vertex() if words is None else fetch_container(words, RECORDS)
        status, _, _ = translate(directory / label.replace(" ", "_"), vs, layout)
        assert status == "unsupported: " + want, (label, status)
        print(f"translate: {label} refused")
    # s_wqm of exec into itself only adds lanes: it stays a no-op and the fetch still translates.
    words = [SWAPPC, PREFETCH, WQM_EXEC_EXEC, V7_ONE, WAIT_VM, 0xf80008cf, 0x07060504, 0xf800020f, 0x070a0908, ENDPGM]
    status, _, _ = translate(directory / "wqm_exec_exec", fetch_container(words, RECORDS), LAYOUT_TEXT)
    assert status == "ok", status
    print("translate: s_wqm_b64 exec, exec translates")

    # z only: the count reaches the highest component read; a mask above the count is refused.
    for count, want in ((3, "ok"), (2, "unsupported: unverified vertex fetch record 0600040401")):
        status, _, _ = translate(directory / ("z%d" % count), vertex(records=[(6, 0, 0x4, 4, count - 1), RECORDS[1]]),
                                 LAYOUT_TEXT)
        assert status == want, status
    print("translate: z-only record with count 3 translates, count 2 refused")

    # Record and layout refusals.
    for container, text, want in (
            (vertex(), LAYOUT_TEXT.replace("POSITION 0 6 1 ", "POSITION 0 6 8 "),
             "input layout slot 8: the stand-in binds vertex-buffer slots 0 to 7"),
            (vertex(), LAYOUT_TEXT.replace("POSITION 0 6 1 4294967295 0 0", "POSITION 0 6 1 4294967295 1 1"),
             "per-instance input layout element"),
            (vertex(), LAYOUT_TEXT.replace("POSITION 0 6 ", "POSITION 0 10 "), "input layout format 10"),
            (vertex(records=[(7,) + RECORDS[0][1:], RECORDS[1]]), LAYOUT_TEXT,
             "unverified vertex fetch record 0700070402"),
            (vertex(records=[(6, 0, 0x7, 254, 2), RECORDS[1]]), LAYOUT_TEXT,
             "unverified vertex fetch record 060007fe02"),
            (fetch_container([SWAPPC, ENDPGM], RECORDS, header_count=3), LAYOUT_TEXT,
             "vertex fetch header counts 3 records, the table holds 2")):
        try:
            gfx2hlsl.stream_fetch(container, text)
        except gfx2hlsl.Unsupported as reason:
            assert str(reason) == want, reason
        else:
            raise AssertionError("accepted: " + want)
    print("translate: slot 8, per-instance, format 10, record kind 7, a record past v255 and a mismatched count refused")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        result = translate_checks(Path(temp) / "translate")
        if len(sys.argv) > 1:
            result = check(Path(temp), *sys.argv[1:])
        sys.exit(result)
