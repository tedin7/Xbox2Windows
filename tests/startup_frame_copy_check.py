"""Check title-memory snapshots for writes between draws and across frame boundaries."""
from pathlib import Path
import re
import sys
import tempfile

from startup_address_srv_check import draw_texture

HLSL = (
    "ByteAddressBuffer constants : register(t0);\n"
    "float4 main(float4 p : SV_Position) : SV_Target {\n"
    "  return asfloat(constants.Load4(0));\n"
    "}\n"
)
RED = bytes((255, 0, 0, 255))
BLUE = bytes((0, 0, 255, 255))


def run(directory, consumer, backend, mode):
    folder = directory / mode
    folder.mkdir()
    log = folder / "trap.log"
    # The cross-frame reuse is off by default: the big-* cases switch it on, the controls below leave it off or break it.
    reuse = {"X2W_BIG_COPY_REUSE": "1"} if mode.startswith("big-") else {}
    modes = {"X2W_TEST_FRAME_COPY": mode, "X2W_TRAP_LOG": str(log), "X2W_BIGBIND_ONCE": "0", **reuse}
    pixels = draw_texture(folder, consumer, backend, HLSL, modes, binding="srv 0 buf table 0 0")
    refreshed = [line for line in log.read_text(errors="replace").splitlines() if "frame copy of" in line] if log.exists() else []
    return pixels, refreshed


def check(directory, consumer, backend):
    # The left half of the 8x4 target is drawn first, the right half second.
    cases = (
        ("changed", (RED * 4 + BLUE * 4) * 4, 1),  # rewritten in place: the second draw must see blue
        ("same", RED * 8 * 4, 0),                  # untouched: both draws red, no new copy
        # A 64 KiB buffer, the second draw in the NEXT frame: a copy kept from the first frame must show the new bytes.
        ("big-changed", (RED * 4 + BLUE * 4) * 4, None),
        ("big-same", RED * 8 * 4, None),
    )
    for mode, expected, refreshed_lines in cases:
        pixels, refreshed = run(directory, consumer, backend, mode)
        if pixels is None:
            return 77
        if pixels != expected:
            raise AssertionError(f"{mode}: actual {pixels.hex()} expected {expected.hex()}")
        if refreshed_lines is not None and len(refreshed) != refreshed_lines:
            raise AssertionError(f"{mode}: {len(refreshed)} refresh log lines, expected {refreshed_lines}:\n" + "\n".join(refreshed))
        print(f"{backend} {mode}: 32 pixels exact, {refreshed_lines} refresh log lines")
    # The controls of the cross-frame reuse: reuse off still shows the new bytes; a reuse that never compares (the mutant) must show
    # the OLD bytes, which proves the big-changed case above would catch it.
    for name, extra, expected in (
        ("big-changed with reuse off (default)", {}, (RED * 4 + BLUE * 4) * 4),
        ("big-changed, never-refill mutant", {"X2W_BIG_COPY_REUSE": "1", "X2W_TEST_FRAME_COPY_STALE": "1"}, RED * 8 * 4),
    ):
        folder = directory / name.replace(" ", "_").replace(",", "")
        folder.mkdir()
        modes = {"X2W_TEST_FRAME_COPY": "big-changed", "X2W_BIGBIND_ONCE": "0", **extra}
        pixels = draw_texture(folder, consumer, backend, HLSL, modes, binding="srv 0 buf table 0 0")
        if pixels is None:
            return 77
        if pixels != expected:
            raise AssertionError(f"{name}: actual {pixels.hex()} expected {expected.hex()}")
        print(f"{backend} {name}: 32 pixels exact")
    return 0


def check_once(directory, consumer, backend):
    shader = HLSL.replace("asfloat(constants.Load4(0))",
                          "0.5 * (asfloat(constants.Load4(0)) + asfloat(constants.Load4(16777200)))")
    log = directory / "trap.log"
    pixels = draw_texture(directory, consumer, backend, shader, {
        "X2W_TEST_FRAME_COPY": "once", "X2W_TRAP_LOG": str(log),
        "X2W_BIGBIND_ONCE": "1", "X2W_BIG_COPY_REUSE": "0",
    }, binding="srv 0 buf table 0 0")
    if pixels is None:
        return 77
    expected = (RED * 4 + BLUE * 4) * 4
    if pixels != expected:
        raise AssertionError(f"once: actual {pixels.hex()} expected {expected.hex()}")
    frames = re.findall(r"large bindings frame (\d+): (\d+) trims, (\d+) copies, (\d+) hits, (\d+) bytes",
                        log.read_text(errors="replace"))
    expected_frames = [(str(frame), "1", "1", "3", str(16 << 20)) for frame in (1, 2)]
    if frames != expected_frames:
        raise AssertionError(f"once: frame counters {frames!r}, expected {expected_frames!r}")
    print(f"{backend} once: 32 pixels exact, 1 trim and 1 full 16 MiB copy per frame, 3 hits per frame")
    # Two views of ONE address with different lengths in one frame: the cache key must include the length. The 64 KiB view reads the
    # zero head and nothing past its end (pixels 0); the 16 MiB view also reads the red words at its end.
    keys_shader = HLSL.replace("asfloat(constants.Load4(0))",
                               "(asfloat(constants.Load4(0)) + asfloat(constants.Load4(16777200)))")
    ZERO = bytes(4)
    expected_keys = b"".join((ZERO * 2 if column % 2 == 0 else RED * 2) for column in range(4)) * 4
    for name, extra, expected in (("keys", {}, expected_keys),
                                  ("keys, address-only key (mutant)", {"X2W_TEST_BIGBIND_KEY_ADDRESS": "1"}, None)):
        folder = directory / name.split(",")[0].replace(" ", "_")
        if extra:
            folder = directory / "keys_mutant"
        folder.mkdir()
        pixels = draw_texture(folder, consumer, backend, keys_shader, {
            "X2W_TEST_FRAME_COPY": "once-keys", "X2W_BIGBIND_ONCE": "1", "X2W_BIG_COPY_REUSE": "0", **extra,
        }, binding="srv 0 buf table 0 0")
        if pixels is None:
            return 77
        if expected is not None and pixels != expected:
            raise AssertionError(f"{name}: actual {pixels.hex()} expected {expected.hex()}")
        if expected is None and pixels == expected_keys:
            raise AssertionError(f"{name}: the mutant still drew the right picture, the control proves nothing")
        print(f"{backend} {name}: " + ("32 pixels exact, small and large view of one address read their own bytes" if expected else "wrong picture, as the control expects"))
    # Control: with the reset at Present broken, the second frame reuses the first frame's snapshot and must show the old colour
    # (red in all eight columns), which proves the check above catches a cache that outlives its frame.
    folder = directory / "no_reset"
    folder.mkdir()
    pixels = draw_texture(folder, consumer, backend, shader, {
        "X2W_TEST_FRAME_COPY": "once", "X2W_BIGBIND_ONCE": "1", "X2W_BIG_COPY_REUSE": "0", "X2W_TEST_BIGBIND_NO_RESET": "1",
    }, binding="srv 0 buf table 0 0")
    if pixels != RED * 8 * 4:
        raise AssertionError(f"once, no reset at Present: actual {pixels.hex() if pixels else pixels} expected all red")
    print(f"{backend} once, no reset at Present (mutant): second frame shows the old bytes, as the control expects")
    return 0


def check_mirror_two_lists(directory, consumer, backend):
    shader = """ByteAddressBuffer data : register(t0);
float4 main(float4 p : SV_Position) : SV_Target {
    uint first = ((uint)p.y * 8 + (uint)p.x) * 8;
    uint sum = 0;
    for (uint i = 0; i < 8; ++i) {
        uint block = first + i;
        uint marker = data.Load(block * 65536 + 20);
        if (marker != 5 * block + 13)
            return float4(1, 0, 1, 1);
        sum += marker;
    }
    return float4((sum & 255) / 255.0, 1, 0, 1);
}
"""
    mib = 1 << 20
    reference = bytes(channel for pixel in range(32)
                      for channel in (sum(5 * block + 13 for block in range(pixel * 8, pixel * 8 + 8)) & 255,
                                      255, 0, 255))
    clear = BLUE * 32
    cases = (
        ("mirror_two_lists", "1", "0", "0", (reference, reference), (1, 1537, 3, 112 * mib), (4, 0, 0), []),
        ("two_lists_mirror_off", "0", "0", "0", (reference, reference), (1, 1, 3, 16 * mib), (4, 0, 0), []),
        ("two_lists_uncapped_mutant", "1", "1", "0", (reference, clear),
         (3, 1536, 1, 96 * mib), (2, 2, 2), []),
        ("two_lists_ring_full", "1", "0", str(64 << 10), (clear, clear),
         (1, 0, 3, 0), (0, 4, 0), [65536, 16 * mib]),
    )
    for name, mirror, uncapped, ring_kib, wanted_pixels, wanted_bindings, wanted_draws, wanted_capacity in cases:
        folder = directory / name
        folder.mkdir()
        log = folder / "trap.log"
        second = draw_texture(folder, consumer, backend, shader, {
            "X2W_TEST_FRAME_COPY": "once-mirror-two", "X2W_TRAP_LOG": str(log),
            "X2W_BIGBIND_ONCE": "1", "X2W_BIG_COPY_REUSE": "0", "X2W_GPU_MIRROR": mirror,
            "X2W_TEST_GPU_MIRROR_UNCAPPED": uncapped, "X2W_TEST_UPLOAD_RING_MAX_KIB": ring_kib,
            "X2W_TEST_GPU_MIRROR_STALE": "0", "X2W_TEST_GPU_MIRROR_FULL_COPY": "0",
        }, binding="srv 0 buf table 0 0")
        if second is None:
            return 77
        actual = ((folder / "pixels.0").read_bytes(), second)
        diagnostics = log.read_text(errors="replace")
        bindings = re.findall(r"large bindings frame 1: (\d+) trims, (\d+) copies, (\d+) hits, (\d+) bytes", diagnostics)
        draws = re.findall(r"mirror fixture frame 1: (\d+) drawn, (\d+) refused, (\d+) oversize", diagnostics)
        capacity = list(map(int, re.findall(r"no unused capacity \(asked (\d+) bytes", diagnostics)))
        oversize_logs = len(re.findall(r"upload ring reservation refused: oversize", diagnostics))
        if (actual != wanted_pixels or bindings != [tuple(map(str, wanted_bindings))]
                or draws != [tuple(map(str, wanted_draws))] or capacity != wanted_capacity
                or oversize_logs != wanted_draws[2]):
            raise AssertionError(f"{name}: pixels {[pixels.hex() for pixels in actual]}, expected "
                                 f"{[pixels.hex() for pixels in wanted_pixels]}; bindings {bindings}, "
                                 f"expected {wanted_bindings}; draws {draws}, expected {wanted_draws}; "
                                 f"capacity refusals {capacity}, expected {wanted_capacity}; oversize logs {oversize_logs}")
        print(f"{backend} {name}: both targets match reference, binding counters {wanted_bindings}, draws {wanted_draws}")
    return 0


def check_mirror(directory, consumer, backend):
    shader = """ByteAddressBuffer constants : register(t0);
float4 main(float4 p : SV_Position) : SV_Target {
    uint first = ((uint)p.y * 8 + (uint)p.x) * 16;
    for (uint block = first; block < first + 16; ++block) {
        if (asfloat(constants.Load(block * 65536 + 4)) != float(block + 1))
            return float4(1, 0, 1, 1);
    }
    return asfloat(uint4(constants.Load(0), constants.Load(11534336),
                         constants.Load(33554424), constants.Load(33554428)));
}
"""
    block = 64 << 10
    size = 32 << 20
    expected = [bytes(255 if channel <= frame or channel == 3 else 0
                      for channel in range(4)) * 32 for frame in range(3)]
    cases = (
        ("mirror", "1", "0", (512, 1, 1), (size, block, block), expected),
        ("mirror_off", "0", "0", (1, 1, 1), (size, size, size), expected),
        ("stale_mutant", "1", "1", (512, 0, 0), (size, 0, 0), [expected[0]] * 3),
    )
    for name, enabled, stale, copies, uploaded, reference in cases:
        folder = directory / name
        folder.mkdir()
        log = folder / "trap.log"
        last = draw_texture(folder, consumer, backend, shader, {
            "X2W_TEST_FRAME_COPY": "once-mirror", "X2W_TRAP_LOG": str(log),
            "X2W_BIGBIND_ONCE": "1", "X2W_BIG_COPY_REUSE": "0",
            "X2W_GPU_MIRROR": enabled, "X2W_TEST_GPU_MIRROR_STALE": stale,
            "X2W_TEST_GPU_MIRROR_FULL_COPY": "1" if enabled == "0" else "0",
        }, binding="srv 0 buf table 0 0")
        if last is None:
            return 77
        actual = [(folder / f"pixels.{frame}").read_bytes() for frame in range(2)] + [last]
        for frame, (pixels, wanted) in enumerate(zip(actual, reference), 1):
            if pixels != wanted:
                raise AssertionError(f"{name} frame {frame}: actual {pixels.hex()} expected {wanted.hex()}")
        frames = re.findall(r"large bindings frame (\d+): (\d+) trims, (\d+) copies, (\d+) hits, (\d+) bytes",
                            log.read_text(errors="replace"))
        counters = [tuple(map(int, record)) for record in frames]
        wanted = [(frame, 1, copies[frame - 1], 1, uploaded[frame - 1]) for frame in range(1, 4)]
        if counters != wanted:
            raise AssertionError(f"{name}: actual counters {counters}, expected {wanted}")
        print(f"{backend} {name}: upload bytes {uploaded}, "
              + ("stale pixels detected in frames 2 and 3" if stale == "1" else "all frames match CPU"))
    return check_mirror_two_lists(directory, consumer, backend)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        consumer, backend = sys.argv[1:3]
        check_mode = {"once": check_once, "mirror": check_mirror}[sys.argv[3]] if len(sys.argv) > 3 else check
        sys.exit(check_mode(Path(temporary), consumer, backend))
