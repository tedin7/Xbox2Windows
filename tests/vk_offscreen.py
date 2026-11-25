"""Vulkan stand-in for the offscreen_check operations that `pixel_aot_check.py draw-core` reaches.

pixel_aot_check.py runs `[renderer, backend, <operation>, shader, ...]`. In `draw-core-vulkan` mode the renderer
is the Python interpreter and the backend is this script, so every call keeps the command line, exit codes, stderr
words and batch status lines of tests/offscreen_check.cpp:

  --check-pixel SHADER rgba|bgra            run_check_pixel (the 8 x 8 ramp texture, no case file)
  --check-pixel-case SHADER CASE [RAW]      run_check_pixel_case
  --check-pixel-batch SHADER CASE...        run_check_pixel_batch (one `case=N status=pass|mismatch|error` per case)
  --check-pixel-quad-lod SHADER [RAW]       run_check_pixel_quad_lod (two mips, three indexed vertices)

The draw and the readback come from tests/vk_pixel_draw.c in its --draw form. The comparison of draw_verified
(tests/pixel_aot_render.cpp) is done here: every channel bit for bit, except the channels a case marks in its
NaN-class mask, which accept any NaN. The pixel shader and the vertex shader are SPIR-V.

Environment (set by `pixel_aot_check.py draw-core-vulkan`):
  X2W_VK_HOST   the built vk_pixel_draw binary
  X2W_VK_VS     the vertex shader (tests/pixel_aot_vs.hlsl as SPIR-V, compiled with -fvk-invert-y)
  X2W_VK_DEVICE substring of the Vulkan device name
  X2W_VK_ALLOW_NO_DENORM=1  draw a module that declares DenormPreserve 32 on a device that lacks
                shaderDenormPreserveFloat32 (diagnostic run; otherwise the host exits 77, a capability skip)
  X2W_VK_WORK   directory for the temporary files (default: the system temporary directory)

`--self-test` (or no arguments) needs no Vulkan device, DXC or vk_pixel_draw: it checks the case parser, the
comparison, the batch status lines and the refusals against synthetic readbacks and a fake host.

Exit codes follow offscreen_check: 0 pass, 1 mismatch or error, 2 invalid arguments, 77 capability skip. 78 is a
validation refusal by the Vulkan driver.
"""
import math
import os
import struct
import subprocess
import sys
import tempfile

RAMP = (0, 16, 37, 64, 101, 153, 219, 254)  # pixel_aot_render.cpp, ramp
TITLE_MEMORY = 4


class PixelMismatch(Exception):
    def __init__(self):
        super().__init__("AOT color export differs from CPU reference")


class Skip(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def expected_channel(byte):
    """pixel_aot_render.cpp expected_channel: the UNORM value truncated to the 11 bits of a binary16 export."""
    if not byte:
        return 0.0
    value = byte / 255.0
    _, exponent = math.frexp(value)
    scale = math.ldexp(1.0, 11 - exponent)
    return f32(math.floor(value * scale) / scale)


class Input:
    """PixelInput of pixel_aot_render.cpp, reduced to the fields these operations use. Expected rows are words."""

    def __init__(self, width, height, scale):
        self.width, self.height, self.scale = width, height, scale
        self.levels = []  # RGBA8 bytes per mip level
        self.expected = []  # (4 words) per texel of level 0
        self.constants = bytes(256)
        self.extra = None  # 512 bytes: b1 then b2
        self.write_mask = 0xF
        self.masked_case = False
        self.integer_rt = False
        self.nan_mask = b""
        self.viewport = None


def ramp_input(swap_rb):
    """check_pixel without a case file: the ramp texture and its reference colors."""
    texture = Input(8, 8, 8)
    rows = bytearray()
    for g in RAMP:
        for r in RAMP:
            rows += bytes((r, g, 128, 255))
            color = [expected_channel(r), expected_channel(g), expected_channel(128), 1.0]
            if swap_rb:
                color[0], color[2] = color[2], color[0]
            texture.expected.append(tuple(bits(c) for c in color))
    texture.levels = [bytes(rows)]
    return texture


def read_case(texture, path):
    """check_pixel's case reader: constants, 64 float4 rows, then the optional trailer of write_case."""
    try:
        data = open(path, "rb").read()
    except OSError as error:
        raise RuntimeError(f"Cannot read the case file: {error}")
    if len(data) < 256 + 64 * 16:
        raise RuntimeError("Expected a 256-byte constant buffer and 64 float4 reference colors")
    texture.constants = data[:256]
    texture.expected = [struct.unpack_from("<4I", data, 256 + 16 * i) for i in range(64)]
    pos = 256 + 64 * 16
    if pos == len(data):
        return
    flags = data[pos]
    pos += 1
    if flags & 4 and flags & 0xF0:
        raise RuntimeError("Unknown case trailer flag bits")
    texture.integer_rt = bool(flags & 1)
    mask = flags >> 4
    if mask:
        texture.masked_case = True
        texture.write_mask = mask
    if flags & 2:
        rect = struct.unpack_from("<4I", data, pos) if len(data) - pos >= 16 else None
        pos += 16
        if rect is None:
            raise RuntimeError("Invalid pixel case viewport")
        left, top, width, height = rect
        if (not width or not height or left >= texture.width or top >= texture.height or
                width > texture.width - left or height > texture.height - top):
            raise RuntimeError("Invalid pixel case viewport")
        texture.scale = 1
        texture.viewport = rect
    if flags & TITLE_MEMORY:
        raise RuntimeError("Title memory cases are not supported by the Vulkan adapter")
    if flags & 8:
        if len(data) - pos < 512:
            raise RuntimeError("Expected two 256-byte constant buffers for b1 and b2")
        texture.extra = data[pos:pos + 512]
        pos += 512
    if pos != len(data):
        if texture.integer_rt:
            raise RuntimeError("NaN-class masks require a floating-point render target")
        texture.nan_mask = data[pos:]
        if len(texture.nan_mask) != 64:
            raise RuntimeError("Expected exactly 64 NaN mask bytes after the case trailer fields")


def quad_lod_input():
    """check_pixel_quad_lod: an 8 x 8 red mip over a 4 x 4 green one, the partial triangle of vertices 3 to 5."""
    texture = Input(8, 8, 1)
    constants = bytearray(256)
    struct.pack_into("<f", constants, 8, 4.0)
    texture.constants = bytes(constants)
    texture.levels = [bytes((255, 0, 0, 255)) * 64, bytes((0, 255, 0, 255)) * 16]
    green, clear = (bits(0.0), bits(1.0), bits(0.0), bits(1.0)), (bits(-1.0),) * 4
    texture.expected = [green if x + y <= 6 else clear for y in range(8) for x in range(8)]
    return texture


def is_nan_bits(word):
    return (word & 0x7F800000) == 0x7F800000 and (word & 0x007FFFFF) != 0


def as_float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def compare(texture, readback, width, height):
    """draw_verified's loop for one draw without tolerance: the first differing pixel in row-major order raises."""
    words = struct.unpack(f"<{width * height * 4}I", readback)
    for y in range(height):
        sample_y = (2 * y + 1) * texture.height // (2 * height)
        for x in range(width):
            sample_x = (2 * x + 1) * texture.width // (2 * width)
            index = sample_y * texture.width + sample_x
            expected = texture.expected[index]
            actual = words[(y * width + x) * 4:(y * width + x) * 4 + 4]
            mask = texture.nan_mask[index] if index < len(texture.nan_mask) else 0
            matched = True
            for c in range(4):
                if mask >> c & 1:
                    if not is_nan_bits(expected[c]):
                        raise RuntimeError("NaN-class mask on a channel whose expected word is not NaN")
                    print(f"nan_class x={x} y={y} ch={c} expected={expected[c]:08x} actual={actual[c]:08x}")
                    if not is_nan_bits(actual[c]):
                        matched = False
                elif expected[c] != actual[c]:
                    matched = False
            if not matched:
                sys.stderr.write(f"pixel_mismatch x={x} y={y}\n")
                for c in range(4):
                    sys.stderr.write(f"channel={c} expected={as_float(expected[c]):g} actual={as_float(actual[c]):g} "
                                     f"expected_bits={expected[c]:08x} actual_bits={actual[c]:08x}\n")
                raise PixelMismatch()


def draw_verified(texture, shader, raw_readback=None, indices=None):
    host, vertex = os.environ.get("X2W_VK_HOST"), os.environ.get("X2W_VK_VS")
    if not host or not vertex:
        raise RuntimeError("X2W_VK_HOST and X2W_VK_VS name the Vulkan host and the vertex shader")
    width, height = texture.width * texture.scale, texture.height * texture.scale
    with tempfile.TemporaryDirectory(prefix="vk-offscreen-", dir=os.environ.get("X2W_VK_WORK") or None) as work:
        def put(name, data):
            path = os.path.join(work, name)
            with open(path, "wb") as file:
                file.write(data)
            return path

        out = os.path.join(work, "readback.bin")
        command = [host, "--draw", os.environ.get("X2W_VK_DEVICE", ""), vertex, shader, out,
                   "--target", str(width), str(height), "uint" if texture.integer_rt else "float",
                   "--constants", put("constants.bin", texture.constants),
                   "--texture", str(texture.width), str(texture.height),
                   *(put(f"level{i}.rgba", data) for i, data in enumerate(texture.levels)),
                   "--write-mask", str(texture.write_mask)]
        if texture.extra is not None:
            command += ["--extra", put("extra.bin", texture.extra)]
        if texture.viewport is not None:
            command += ["--viewport", *map(str, texture.viewport)]
        if indices is not None:
            command += ["--indices", ",".join(map(str, indices))]
        # RTV clear colors of draw_verified: 1 2 3 4 for an integer target, -1 (2 -3 4 for masked cases) otherwise.
        if texture.integer_rt:
            command += ["--clear", "1,2,3,4"]
        elif texture.masked_case:
            command += ["--clear", "-1,2,-3,4"]
        if os.environ.get("X2W_VK_ALLOW_NO_DENORM") == "1":
            command.append("--allow-denorm-mismatch")
        run = subprocess.run(command, capture_output=True, text=True)
        if run.stderr:
            sys.stderr.write(run.stderr)
        if run.returncode in (77, 78):
            raise Skip(run.returncode)
        if run.returncode:
            raise RuntimeError(f"vk_pixel_draw exited with {run.returncode}")
        readback = open(out, "rb").read()
        if raw_readback:
            with open(raw_readback, "wb") as file:
                file.write(readback)
        compare(texture, readback, width, height)
        print(f"pixels={width * height} channels={width * height * 4} max_absolute_error=0")



def check_translated_pixel(shader, swap_rb, reference=None, raw_readback=None):
    texture = ramp_input(swap_rb)
    if reference:
        read_case(texture, reference)
    draw_verified(texture, shader, raw_readback)


def run_check_pixel(args):
    check_translated_pixel(args[0], args[1] == "bgra")
    return 0


def run_check_pixel_case(args):
    check_translated_pixel(args[0], False, args[1], args[2] if len(args) == 3 else None)
    return 0


def run_check_pixel_quad_lod(args):
    draw_verified(quad_lod_input(), args[0], args[1] if len(args) == 2 else None, indices=(3, 4, 5))
    return 0


def run_check_pixel_batch(args):
    failed = False
    for i, case in enumerate(args[1:]):
        sys.stderr.write(f"case={i} path={case}\n")
        sys.stderr.flush()
        status = "pass"
        try:
            check_translated_pixel(args[0], False, case)
        except PixelMismatch as error:
            sys.stderr.write(f"{error}\n")
            status = "mismatch"
        except Skip:
            raise
        except Exception as error:
            sys.stderr.write(f"{error}\n")
            status = "error"
        print(f"case={i} status={status}", flush=True)
        failed |= status != "pass"
    return 1 if failed else 0


# name: (minimum argc, maximum argc, argument check, function); argc counts offscreen_check's argv, which has
# the executable and the backend before the operation, so it is len(sys.argv) + 1 here.
OPERATIONS = {
    "--check-pixel": (5, 5, lambda a: a[1] in ("rgba", "bgra"), run_check_pixel),
    "--check-pixel-case": (5, 6, None, run_check_pixel_case),
    "--check-pixel-batch": (5, 1 << 30, None, run_check_pixel_batch),
    "--check-pixel-quad-lod": (4, 5, None, run_check_pixel_quad_lod),
}


def self_test():
    """Device-free controls for the parser, the comparison, the batch lines and the refusals."""
    import contextlib
    import io
    import stat
    import unittest.mock

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pixel_aot_check import write_case

    def must_raise(exception, text, function, *args):
        try:
            function(*args)
        except exception as error:
            if text not in str(error):
                raise AssertionError(f"Expected {text!r}, got {error}")
            return
        raise AssertionError(f"Expected {exception.__name__} containing {text!r}")

    def load(path):
        texture = ramp_input(False)
        read_case(texture, path)
        return texture

    def readback_of(texture):
        words = []
        for y in range(texture.height * texture.scale):
            for x in range(texture.width * texture.scale):
                row = (2 * y + 1) * texture.height // (2 * texture.height * texture.scale)
                col = (2 * x + 1) * texture.width // (2 * texture.width * texture.scale)
                words += texture.expected[row * texture.width + col]
        return words

    def as_bytes(words):
        return struct.pack(f"<{len(words)}I", *words)

    one = (bits(0.25), bits(0.5), bits(0.75), bits(1.0))
    nan = 0x7FC00000
    with tempfile.TemporaryDirectory(prefix="vk-offscreen-self-") as work:
        def case(name, constants=bytes(range(256)), pixels=(0.25, 0.5, 0.75, 1.0), **options):
            return str(write_case(os.path.join(work, name), constants, pixels, **options))

        # Parser: the original format, the integer target, the mask nibble, the viewport and the NaN mask.
        plain = load(case("plain.case"))
        assert plain.constants == bytes(range(256)) and plain.expected == [one] * 64
        assert not plain.integer_rt and plain.write_mask == 0xF and not plain.masked_case and plain.viewport is None
        assert load(case("int.case", integer_rt=True)).integer_rt
        masked = load(case("mask.case", write_mask=5))
        assert masked.write_mask == 5 and masked.masked_case and not masked.integer_rt
        assert load(case("extra.case", extra_constants=bytes(512))).extra == bytes(512)
        masks = load(case("nan.case", pixels=(float("nan"),) * 4, nan_mask=0xF))
        assert masks.nan_mask == b"\x0f" * 64 and not masks.integer_rt
        view = write_case(os.path.join(work, "view.case"), bytes(256), (0.25, 0.5, 0.75, 1.0))
        with open(view, "ab") as file:  # flags byte with bit 1, then the viewport rectangle
            file.write(bytes((2,)) + struct.pack("<4I", 2, 1, 4, 3))
        viewport = load(view)
        assert viewport.viewport == (2, 1, 4, 3) and viewport.scale == 1
        bad = open(view, "rb").read()[:-16] + struct.pack("<4I", 6, 1, 4, 3)
        open(view, "wb").write(bad)
        must_raise(RuntimeError, "Invalid pixel case viewport", load, str(view))

        # Refusals.
        must_raise(RuntimeError, "Title memory", load, case("title.case", title_memory=b"x"))
        short = os.path.join(work, "short.case")
        open(short, "wb").write(bytes(100))
        must_raise(RuntimeError, "Expected a 256-byte", load, short)
        flags = os.path.join(work, "flags.case")
        open(flags, "wb").write(open(case("base.case"), "rb").read() + bytes((4 | 0x10,)))
        must_raise(RuntimeError, "Unknown case trailer flag bits", load, flags)
        bad_nan = os.path.join(work, "badnan.case")
        open(bad_nan, "wb").write(open(case("base2.case"), "rb").read() + bytes((0,)) + bytes(10))
        must_raise(RuntimeError, "exactly 64 NaN mask bytes", load, bad_nan)
        must_raise(RuntimeError, "Cannot read the case file", load, os.path.join(work, "missing.case"))

        # Comparison: exact pass, a one-ulp difference, the scan order and the NaN mask.
        width = height = 64
        exact = readback_of(plain)
        compare(plain, as_bytes(exact), width, height)
        for channel in range(4):
            drift = list(exact)
            drift[(5 * width + 9) * 4 + channel] += 1  # x=9 y=5, one ulp in this channel only
            drift[(20 * width + 1) * 4] += 1  # a later pixel must not be reported
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                must_raise(PixelMismatch, "differs from CPU reference", compare, plain, as_bytes(drift), width, height)
            report = stderr.getvalue().splitlines()
            assert report[0] == "pixel_mismatch x=9 y=5", report[0]
            for c in range(4):
                line = report[1 + c]
                word = (bits(0.25), bits(0.5), bits(0.75), bits(1.0))[c]
                assert line.startswith(f"channel={c} expected="), line
                assert line.endswith(f"expected_bits={word:08x} actual_bits={word + (c == channel):08x}"), line
        masks.expected = [(nan,) * 4] * 64
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            compare(masks, as_bytes([0x7FC12345] * (width * height * 4)), width, height)
        assert shown.getvalue().startswith("nan_class x=0 y=0 ch=0 expected=7fc00000 actual=7fc12345")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            must_raise(PixelMismatch, "differs", compare, masks, as_bytes([bits(1.0)] + [nan] * (width * height * 4 - 1)),
                       width, height)
        masks.expected = [(bits(1.0),) * 4] * 64
        must_raise(RuntimeError, "not NaN", compare, masks, as_bytes([nan] * (width * height * 4)), width, height)

        # Batch lines through a fake host that copies a prepared readback to the output path.
        good, wrong = os.path.join(work, "good.bin"), os.path.join(work, "wrong.bin")
        open(good, "wb").write(as_bytes(exact))
        open(wrong, "wb").write(as_bytes(drift))
        script = os.path.join(work, "fake_host.py")
        with open(script, "w") as file:
            file.write("import shutil, sys, os\n"
                       "if os.environ.get('FAKE_EXIT'): sys.exit(int(os.environ['FAKE_EXIT']))\n"
                       "shutil.copy(os.environ['FAKE_READBACK'], sys.argv[5])\n")
        # The adapter takes one host path, so a launcher runs the script through sys.executable (no shebang needed).
        if os.name == "nt":
            host = os.path.join(work, "fake_host.cmd")
            launcher = f'@"{sys.executable}" "{script}" %*\r\n'
        else:
            host = os.path.join(work, "fake_host")
            launcher = f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n'
        with open(host, "w", newline="") as file:
            file.write(launcher)
        os.chmod(host, os.stat(host).st_mode | stat.S_IXUSR)
        cases = [case("b0.case"), os.path.join(work, "missing.case"), case("b2.case")]

        def batch(readback, **extra):
            environment = {"X2W_VK_HOST": host, "X2W_VK_VS": "vs.spv", "FAKE_READBACK": readback, **extra}
            with unittest.mock.patch.dict(os.environ, environment):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    try:
                        code = run_check_pixel_batch(["shader.spv", *cases])
                    except Skip as skip:
                        code = skip.code
            return code, out.getvalue(), err.getvalue()

        code, out, err = batch(good)
        assert code == 1 and [l for l in out.splitlines() if l.startswith("case=")] == [
            "case=0 status=pass", "case=1 status=error", "case=2 status=pass"], out
        assert err.count("case=") == 3 and "Cannot read the case file" in err
        code, out, err = batch(wrong)
        lines = [line for line in out.splitlines() if line.startswith("case=")]
        assert code == 1 and lines == ["case=0 status=mismatch", "case=1 status=error", "case=2 status=mismatch"], out
        assert "pixel_mismatch x=9 y=5" in err
        cases.pop(1)
        code, out, err = batch(good)
        assert code == 0 and [l for l in out.splitlines() if l.startswith("case=")] == [
            "case=0 status=pass", "case=1 status=pass"], out
        assert "pixels=4096 channels=16384 max_absolute_error=0" in out
        assert batch(good, FAKE_EXIT="77")[0] == 77 and batch(good, FAKE_EXIT="78")[0] == 78
        assert batch(good, FAKE_EXIT="3")[0] == 1

        # Argument validation of main.
        with contextlib.redirect_stderr(io.StringIO()):
            assert main(["--check-pixel", "shader.spv", "argb"]) == 2
            assert main(["--check-pixel-case", "shader.spv"]) == 2
            assert main(["--nope"]) == 2
    print("vk_offscreen self-test passed")
    return 0


def main(argv):
    if not argv or argv == ["--self-test"]:
        return self_test()
    spec = OPERATIONS.get(argv[0]) if argv else None
    argc = len(argv) + 2
    if not spec or not spec[0] <= argc <= spec[1] or (spec[2] and not spec[2](argv[1:])):
        sys.stderr.write(f"Unknown GPU operation or invalid arguments: {argv[0] if argv else ''}\n")
        return 2
    print(f"backend=vulkan debug_layer=unavailable")
    try:
        return spec[3](argv[1:])
    except Skip as skip:
        return skip.code
    except Exception as error:
        sys.stderr.write(f"{error}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
