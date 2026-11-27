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
        ("mirror", "1", "0", "0", (512, 1, 1), (size, block, block), expected),
        ("mirror_off", "0", "0", "0", (1, 1, 1), (size, size, size), expected),
        ("stale_mutant", "1", "0", "1", (512, 0, 0), (size, 0, 0), [expected[0]] * 3),
        ("mirror_shared", "1", "1", "0", (512, 1, 1), (size, block, block), expected),
        ("shared_stale_mutant", "1", "1", "1", (512, 0, 0), (size, 0, 0), [expected[0]] * 3),
    )
    for name, enabled, shared, stale, copies, uploaded, reference in cases:
        folder = directory / name
        folder.mkdir()
        log = folder / "trap.log"
        last = draw_texture(folder, consumer, backend, shader, {
            "X2W_TEST_FRAME_COPY": "once-mirror", "X2W_TRAP_LOG": str(log),
            "X2W_BIGBIND_ONCE": "1", "X2W_BIG_COPY_REUSE": "0",
            "X2W_GPU_MIRROR": enabled, "X2W_GPU_MIRROR_SHARED": shared, "X2W_TEST_GPU_MIRROR_STALE": stale,
            "X2W_TEST_GPU_MIRROR_SKIP_WAIT": "0", "X2W_TEST_GPU_MIRROR_UPLOAD_AFTER": "0",
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
    return 0


def check_shared_mirror(directory, consumer, backend):
    shader = """ByteAddressBuffer data : register(t0);
uint marker(uint block) { return block * 3 + 17; }
float4 main(float4 pixel : SV_Position) : SV_Target {
    uint left = data.Load(7 * 65536);
    uint right = data.Load(1403 * 65536);
    bool old = left == marker(7) && right == marker(1403);
    bool updated = left == marker(7) + 8192 && right == marker(1403) + 8192;
    if ((!old && !updated) || data.Load(100663292) != 73)
        return float4(1, 0, 1, 1);
    uint first = ((uint)pixel.x + (uint)pixel.y * 8) * 48;
    for (uint i = first; i < first + 48; ++i) {
        uint expected = marker(i) + ((updated && (i == 7 || i == 1403)) ? 8192 : 0);
        if (data.Load(i * 65536) != expected)
            return float4(1, 0, 1, 1);
    }
    return updated ? float4(0, 1, 0, 1) : float4(1, 0, 0, 1);
}
"""
    old = RED * 32
    updated = bytes((0, 255, 0, 255)) * 32
    cases = ("shared", "present_upload", "upload_after", "skip_wait")
    for case in cases:
        folder = directory / case
        folder.mkdir()
        log = folder / "trap.log"
        pixels = draw_texture(folder, consumer, backend, shader, {
            "X2W_TEST_FRAME_COPY": "once-shared-p" if case == "present_upload" else "once-shared",
            "X2W_TRAP_LOG": str(log),
            "X2W_BIGBIND_ONCE": "1", "X2W_BIG_COPY_REUSE": "0",
            "X2W_GPU_MIRROR": "1", "X2W_GPU_MIRROR_SHARED": "1", "X2W_TEST_GPU_MIRROR_STALE": "0",
            "X2W_TEST_GPU_MIRROR_FULL_COPY": "0",
            "X2W_TEST_GPU_MIRROR_UPLOAD_AFTER": str(int(case == "upload_after")),
            "X2W_TEST_GPU_MIRROR_SKIP_WAIT": str(int(case == "skip_wait")),
        }, binding="srv 0 buf table 0 0")
        if pixels is None:
            return 77
        actual = [(folder / f"pixels.{frame}.{i}").read_bytes()
                  for frame in range(2) for i in range(3) if (frame, i) != (1, 2)] + [pixels]
        reference = [old] * 3 + [updated] * 3
        if case == "upload_after":
            # List 2, submitted first, sees the missing upload in frame 1
            # and the previous snapshot in frame 2. All later lists see the copy.
            if len(actual[2]) != len(old) or actual[2] == old:
                raise AssertionError("upload_after: the first submitted list still matched the CPU")
            reference[2] = actual[2]
            reference[5] = old
        elif case == "skip_wait":
            # Two queued frame 1 readers see the new data after the overwrite.
            reference[0:2] = [updated] * 2
        for i, (actual_pixels, expected) in enumerate(zip(actual, reference)):
            if actual_pixels != expected:
                raise AssertionError(f"{case} frame {i // 3 + 1} list {i % 3}: "
                                     f"actual {actual_pixels.hex()} expected {expected.hex()}")

        counters = re.findall(
            r"shared mirror frame (\d+): (\d+) lists, (\d+) blocks, (\d+) bytes, (\d+) binds, (\d+) fallbacks",
            log.read_text(errors="replace"),
        )
        observed = [tuple(map(int, row)) for row in counters]
        expected = [(1, 1, 1536, 96 << 20, 4, 0), (2, 1, 2, 128 << 10, 3, 0)]
        if case == "present_upload":
            expected = [(1, 1, 1536, 96 << 20, 1, 0), (2, 0, 0, 0, 3, 0), (3, 1, 2, 128 << 10, 3, 0)]
        if observed != expected:
            raise AssertionError(f"{case}: mirror counters {observed}, expected {expected}")
        print(f"{backend} {case}: six readbacks and mirror counters match expected results")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        consumer, backend = sys.argv[1:3]
        check_mode = {"once": check_once, "mirror": check_mirror, "shared": check_shared_mirror}[sys.argv[3]] if len(sys.argv) > 3 else check
        sys.exit(check_mode(Path(temporary), consumer, backend))
