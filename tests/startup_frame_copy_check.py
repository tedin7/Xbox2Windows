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
COUNT_FIELDS = r"(\d+) trims, (\d+) copies, (\d+) hits, (\d+) bytes, (\d+) failed reservations"
FRAME_COUNTS = re.compile(r"large bindings frame (\d+): " + COUNT_FIELDS)
TOTAL_COUNTS = re.compile(r"large bindings totals \(frame (\d+)\): " + COUNT_FIELDS)


def read_counts(log, pattern=FRAME_COUNTS):
    return [tuple(map(int, match)) for match in pattern.findall(log.read_text(errors="replace"))]


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
        "X2W_BIGBIND_ONCE": "1", "X2W_BIGBIND_POOL": "0", "X2W_BIG_COPY_REUSE": "0",
    }, binding="srv 0 buf table 0 0")
    if pixels is None:
        return 77
    expected = (RED * 4 + BLUE * 4) * 4
    if pixels != expected:
        raise AssertionError(f"once: actual {pixels.hex()} expected {expected.hex()}")
    frames = read_counts(log)
    expected_frames = [(frame, 1, 1, 3, 16 << 20, 0) for frame in (1, 2)]
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
            "X2W_TEST_FRAME_COPY": "once-keys", "X2W_BIGBIND_ONCE": "1", "X2W_BIGBIND_POOL": "0", "X2W_BIG_COPY_REUSE": "0", **extra,
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
        "X2W_TEST_FRAME_COPY": "once", "X2W_BIGBIND_ONCE": "1", "X2W_BIGBIND_POOL": "0", "X2W_BIG_COPY_REUSE": "0", "X2W_TEST_BIGBIND_NO_RESET": "1",
    }, binding="srv 0 buf table 0 0")
    if pixels != RED * 8 * 4:
        raise AssertionError(f"once, no reset at Present: actual {pixels.hex() if pixels else pixels} expected all red")
    print(f"{backend} once, no reset at Present (mutant): second frame shows the old bytes, as the control expects")
    return check_ring_full(directory, consumer, backend)


def check_ring_full(directory, consumer, backend):
    trimmed = (16 << 20) - 4352
    shader = f"""ByteAddressBuffer payload : register(t0);
float4 main(float4 position : SV_Position) : SV_Target {{
    float head = asfloat(payload.Load(36));
    float tail = asfloat(payload.Load({trimmed - 4}));
    return float4(head, tail > 0 ? 1.0 : 0.0, tail < 0 ? 1.0 : 0.0, 1.0);
}}
"""
    # CPU reference for the two payloads and the untouched blue clear value.
    colors = tuple(bytes((int(head * 255), 255 * (tail > 0), 255 * (tail < 0), 255))
                   for head, tail in ((0, 1), (1, -1)))
    copied = (1, 1, 3, trimmed, 0)
    refused = (1, 0, 3, 0, 4)
    revalidated = (4, 0, 0, 0, 4)
    cases = (
        ("pool", "1", "32", "4096", "0", colors, (copied, copied)),
        ("pool_full", "1", "16", "4096", "0", (colors[0], BLUE), (copied, refused)),
        ("ring_fallback", "1", "16", "65536", "0", colors, (copied, copied)),
        ("pool_off", "0", "32", "4096", "0", (BLUE, BLUE), (refused, refused)),
        ("drop_failed_mutant", "0", "32", "4096", "1", (BLUE, BLUE), (revalidated, revalidated)),
    )
    for name, pool, cap, ring, mutant, frame_colors, frame_counts in cases:
        folder = directory / name
        folder.mkdir()
        log = folder / "trap.log"
        pixels = draw_texture(folder, consumer, backend, shader, {
            "X2W_TEST_FRAME_COPY": "once-ring-full", "X2W_TRAP_LOG": str(log),
            "X2W_BIGBIND_ONCE": "1", "X2W_BIGBIND_POOL": pool, "X2W_BIGBIND_POOL_MIB": cap,
            "X2W_TEST_UPLOAD_RING_MAX_KIB": ring, "X2W_TEST_BIGBIND_DROP_FAILED": mutant,
            "X2W_BIG_COPY_REUSE": "0",
        }, binding="srv 0 buf table 0 0")
        if pixels is None:
            return 77
        expected_pixels = b"".join(color * 4 for color in frame_colors) * 4
        if pixels != expected_pixels:
            raise AssertionError(f"{name}: actual {pixels.hex()} expected {expected_pixels.hex()}")
        counts = read_counts(log)
        expected_counts = [(frame, *values) for frame, values in enumerate(frame_counts, 1)]
        if counts != expected_counts:
            raise AssertionError(f"{name}: frame counters {counts!r}, expected {expected_counts!r}")
        totals = read_counts(log, TOTAL_COUNTS)
        expected_totals = [(300, *(sum(values) for values in zip(*frame_counts)))]
        if totals != expected_totals:
            raise AssertionError(f"{name}: totals {totals!r}, expected {expected_totals!r}")
        print(f"{backend} once-ring-full {name}: 32 pixels exact, frame counters {counts}")
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
    return 0


MAPPED_COUNTS = re.compile(r"^x2w-render mapped spans: (\d+) queries, (\d+) hits, (\d+) forgets$", re.M)


def mapped_reference(frame):
    return b"".join(
        bytes((0, 0, 0, 255)) if (frame, region) == (50, 0)
        else bytes((1 + (3 * region + frame) % 251, 1 + (region + 7 * frame) % 251, 0, 255))
        for region in range(64)
    )


def mapped_query_budget(queries):
    # One cold fill, one rewarm after the stale touch, and room for shim bookkeeping.
    if not 64 <= queries <= 192:
        raise AssertionError(f"{queries} mapped queries outside the 64–192 budget")


def check_mapped_spans(directory, consumer, backend):
    shader = """ByteAddressBuffer input_data : register(t0);
float4 main() : SV_Target {
    return float4(asfloat(input_data.Load2(0)), 0, 1);
}
"""
    for name, single in (("table", "0"), ("one_entry", "1")):
        folder = directory / name
        folder.mkdir()
        log = folder / "trap.log"
        last = draw_texture(folder, consumer, backend, shader, {
            "X2W_TEST_FRAME_COPY": "mapped-spans", "X2W_TRAP_LOG": str(log),
            "X2W_MAPPED_SPAN_TABLE": "1", "X2W_TEST_MAPPED_SPAN_ONE": single,
            "X2W_BIGBIND_ONCE": "0", "X2W_BIG_COPY_REUSE": "0",
            "X2W_DRAW_NULL_IDENTITY": "0",
        }, binding="srv 0 buf table 0 0")
        if last is None:
            return 77
        for frame in range(100):
            pixels = last if frame == 99 else (folder / f"pixels.{frame}").read_bytes()
            expected = mapped_reference(frame)
            if pixels != expected:
                raise AssertionError(f"{name} frame {frame}: actual {pixels.hex()} expected {expected.hex()}")
        [(queries, hits, forgets)] = read_counts(log, MAPPED_COUNTS)
        if single == "0":
            mapped_query_budget(queries)
            if hits < 6300 or forgets != 1:
                raise AssertionError(f"expected at least 6300 hits and one failed touch, got {hits}, {forgets}")
        else:
            try:
                mapped_query_budget(queries)
            except AssertionError:
                if queries < 6000:
                    raise AssertionError(f"one-entry table made only {queries} queries") from None
            else:
                raise AssertionError("one-entry table passed the query budget")
        print(f"{backend} {name}: 6400 pixels match CPU, {queries} queries, {hits} hits, {forgets} forgets")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        consumer, backend = sys.argv[1:3]
        modes = {"once": check_once, "mirror": check_mirror, "mapped-spans": check_mapped_spans}
        check_mode = modes[sys.argv[3]] if len(sys.argv) > 3 else check
        sys.exit(check_mode(Path(temporary), consumer, backend))
