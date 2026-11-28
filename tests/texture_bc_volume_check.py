"""Byte-exact synthetic checks for the BC table, block rows and a BC7 volume."""
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from texture_blocks import (FORMATS, format_info, linearize_blocks, linearize_volume_4k_s,
                            surface_bytes, tile_blocks, tile_volume_4k_s)

BC_NAMES = ("BC1_UNORM", "BC2_UNORM", "BC3_UNORM", "BC4_UNORM", "BC5_UNORM",
            "BC6H_UF16", "BC7_UNORM")


def block_index(x: int, y: int) -> int:
    """Oracle for y0 y1 x0 x1 y2 x2 y3 x3, low to high."""
    return ((y & 3) | ((x & 3) << 2)
            | ((y & 4) << 2) | ((x & 4) << 3)
            | ((y & 8) << 3) | ((x & 8) << 4))


def block_tag(index: int, x: int, y: int, block_bytes: int, depth: int = 0) -> bytes:
    head = struct.pack("<4sHH", f"BC{index}K".encode(), x, y)
    if block_bytes > len(head):
        head += struct.pack("<H", depth)
    return head + b"\xa5" * (block_bytes - len(head))


def padding(block_bytes: int) -> bytes:
    return b"\xff" * block_bytes


def columns_rows(width: int, height: int) -> tuple[int, int]:
    return (width + 3) // 4, (height + 3) // 4


def oracle_rows(index: int, block_bytes: int, width: int, height: int,
                pitch: int) -> tuple[bytes, bytes]:
    columns, rows = columns_rows(width, height)
    padded = bytearray(padding(block_bytes) * (rows * pitch))
    tight = bytearray()
    for y in range(rows):
        for x in range(columns):
            block = block_tag(index, x, y, block_bytes)
            start = (y * pitch + x) * block_bytes
            padded[start:start + block_bytes] = block
            tight += block
    return bytes(padded), bytes(tight)


def oracle_volume_slice(index: int, width: int, height: int, z: int) -> bytes:
    columns, rows = columns_rows(width, height)
    tile_pitch = (columns + 15) // 16
    payload = bytearray(padding(16) * (tile_pitch * ((rows + 15) // 16) * 256))
    for y in range(rows):
        for x in range(columns):
            tile = (y // 16) * tile_pitch + x // 16
            start = tile * 4096 + block_index(x % 16, y % 16) * 16
            payload[start:start + 16] = block_tag(index, x, y, 16, z)
    return bytes(payload)


def tight_volume_slice(index: int, width: int, height: int, z: int) -> bytes:
    columns, rows = columns_rows(width, height)
    return b"".join(block_tag(index, x, y, 16, z) for y in range(rows) for x in range(columns))


def first_mismatch(actual: bytes, expected: bytes, block_bytes: int,
                   columns: int) -> tuple[int, int] | None:
    for offset in range(0, max(len(actual), len(expected)), block_bytes):
        if actual[offset:offset + block_bytes] != expected[offset:offset + block_bytes]:
            return (offset // block_bytes) % columns, (offset // block_bytes) // columns
    return None


def first_volume_mismatch(actual: bytes, expected: bytes, columns: int,
                          rows: int) -> tuple[int, int, int] | None:
    slice_blocks = columns * rows
    for offset in range(0, max(len(actual), len(expected)), 16):
        if actual[offset:offset + 16] != expected[offset:offset + 16]:
            block = offset // 16
            z, remaining = divmod(block, slice_blocks)
            return remaining % columns, remaining // columns, z
    return None


class FormatChecks(unittest.TestCase):
    def test_table(self) -> None:
        expected = {
            "BC1_UNORM": (71, (4, 4, 1), 8, 2),
            "BC2_UNORM": (74, (4, 4, 1), 16, 2),
            "BC3_UNORM": (77, (4, 4, 1), 16, 2),
            "BC4_UNORM": (80, (4, 4, 1), 8, 2),
            "BC5_UNORM": (83, (4, 4, 1), 16, 2),
            "BC6H_UF16": (95, (4, 4, 1), 16, 2),
            "BC7_UNORM": (98, (4, 4, 1), 16, 2),
            "BC7_UNORM_VOLUME": (98, (4, 4, 1), 16, 3),
        }
        self.assertEqual(set(FORMATS), set(expected))
        for name, (dxgi, extent, block_bytes, dimension) in expected.items():
            with self.subTest(format=name):
                format = format_info(name)
                self.assertEqual(format.dxgi_format, dxgi)
                self.assertEqual(format.block_extent, extent)
                self.assertEqual(format.bytes_per_block, block_bytes)
                self.assertEqual(format.dimension, dimension)

    def test_unknown_format_is_named(self) -> None:
        with self.assertRaises(ValueError) as raised:
            format_info("BC9_UNORM")
        self.assertEqual(str(raised.exception), "Unknown texture format 'BC9_UNORM'")
        for call in (lambda: surface_bytes(8, 8, "BC9_UNORM"),
                     lambda: linearize_blocks(bytes(64), 8, 8, "BC9_UNORM"),
                     lambda: tile_blocks(bytes(64), 8, 8, "BC9_UNORM"),
                     lambda: tile_volume_4k_s([bytes(64)], 8, 8, 1, "BC9_UNORM"),
                     lambda: linearize_volume_4k_s(bytes(4096), 8, 8, 1, "BC9_UNORM")):
            with self.assertRaisesRegex(ValueError, "BC9_UNORM"):
                call()

    def test_dimension_mismatch_names_the_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "BC7_UNORM_VOLUME is a volume format"):
            linearize_blocks(bytes(64), 8, 8, "BC7_UNORM_VOLUME")
        with self.assertRaisesRegex(ValueError, "BC7_UNORM is a 2-D format"):
            tile_volume_4k_s([bytes(64)], 8, 8, 1, "BC7_UNORM")


class BlockRowChecks(unittest.TestCase):
    def test_oracle_covers_one_tile(self) -> None:
        self.assertEqual({block_index(x, y) for y in range(16) for x in range(16)},
                         set(range(256)))
        self.assertEqual([block_index(x, 0) for x in range(4)], [0, 4, 8, 12])
        self.assertEqual([block_index(0, y) for y in range(4)], [0, 1, 2, 3])
        self.assertEqual((block_index(4, 0), block_index(0, 4), block_index(8, 0), block_index(0, 8)),
                         (32, 16, 128, 64))

    def test_surface_sizes(self) -> None:
        for width, height, name, expected in (
                (8, 8, "BC1_UNORM", 32), (8, 8, "BC7_UNORM", 64),
                (9, 9, "BC1_UNORM", 72), (9, 9, "BC7_UNORM", 144),
                (81, 69, "BC1_UNORM", 3024), (81, 69, "BC7_UNORM", 6048)):
            with self.subTest(width=width, height=height, format=name):
                self.assertEqual(surface_bytes(width, height, name), expected)
        for width, height in ((0, 4), (4, 0), (-4, 4), (4, -1)):
            with self.assertRaises(ValueError):
                surface_bytes(width, height, "BC7_UNORM")

    def test_block_rows_round_trip(self) -> None:
        for index, name in enumerate(BC_NAMES):
            block_bytes = format_info(name).bytes_per_block
            for width, height, extra in ((1, 1, 0), (3, 5, 2), (4, 4, 1),
                                         (5, 3, 3), (17, 9, 1), (81, 69, 1)):
                columns, _ = columns_rows(width, height)
                pitch = columns + extra
                padded, tight = oracle_rows(index, block_bytes, width, height, pitch)
                with self.subTest(format=name, width=width, height=height, pitch=pitch):
                    self.assertEqual(linearize_blocks(padded, width, height, name,
                                                       pitch_blocks=pitch), tight)
                    self.assertEqual(linearize_blocks(tight, width, height, name), tight)
                    self.assertEqual(tile_blocks(tight, width, height, name,
                                                 pitch_blocks=pitch), padded)
                    self.assertEqual(tile_blocks(tight, width, height, name), tight)
                    self.assertEqual(tile_blocks(linearize_blocks(padded, width, height, name,
                                                                  pitch_blocks=pitch),
                                                 width, height, name, pitch_blocks=pitch), padded)

    def test_wrong_block_pitch_or_stride(self) -> None:
        width, height, pitch = 81, 69, 22
        columns, rows = columns_rows(width, height)
        for index, name in enumerate(("BC1_UNORM", "BC7_UNORM")):
            block_bytes = format_info(name).bytes_per_block
            padded, tight = oracle_rows(index, block_bytes, width, height, pitch)
            # A producer that packed rows at 21 blocks but declared 22.
            short_stride = tight + padding(block_bytes) * (rows * (pitch - columns))
            self.assertEqual(len(short_stride), rows * pitch * block_bytes)
            with self.subTest(format=name):
                self.assertEqual(first_mismatch(
                    linearize_blocks(short_stride, width, height, name, pitch_blocks=pitch),
                    tight, block_bytes, columns), (0, 1))
                with self.assertRaisesRegex(ValueError, "Expected .* bytes"):
                    linearize_blocks(padded, width, height, name, pitch_blocks=pitch - 1)
                with self.assertRaisesRegex(ValueError, "narrower"):
                    surface_bytes(width, height, name, pitch_blocks=columns - 1)
                with self.assertRaises(ValueError):
                    linearize_blocks(padded, width, height, name, pitch_blocks=pitch + 1)

    def test_tile_blocks_padding_and_tight_length(self) -> None:
        # 8x4 BC1 is two 8-byte blocks in one row; pitch 3 adds one padding block.
        tight = block_tag(1, 0, 0, 8) + block_tag(1, 1, 0, 8)
        self.assertEqual(tile_blocks(tight, 8, 4, "BC1_UNORM", pitch_blocks=3,
                                     padding=b"\x5a" * 8), tight + b"\x5a" * 8)
        with self.assertRaisesRegex(ValueError, "Expected one 8-byte padding block"):
            tile_blocks(tight, 8, 4, "BC1_UNORM", pitch_blocks=3, padding=b"\x5a" * 16)
        with self.assertRaisesRegex(ValueError, "Expected 16 tight bytes for BC1_UNORM, got 24"):
            tile_blocks(tight + padding(8), 8, 4, "BC1_UNORM", pitch_blocks=3)


class VolumeChecks(unittest.TestCase):
    def test_volume_round_trip(self) -> None:
        for index, name, width, height, depth in (
                (7, "BC7_UNORM_VOLUME", 8, 8, 4),
                (7, "BC7_UNORM_VOLUME", 68, 8, 2),
                # 16 columns and one block row: exactly one 4 KiB tile per slice.
                (7, "BC7_UNORM_VOLUME", 64, 4, 2)):
            columns, rows = columns_rows(width, height)
            slices = [tight_volume_slice(index, width, height, z) for z in range(depth)]
            payload = tile_volume_4k_s(slices, width, height, depth, name)
            expected_payload = b"".join(oracle_volume_slice(index, width, height, z)
                                         for z in range(depth))
            expected = b"".join(slices)
            with self.subTest(width=width, height=height, depth=depth):
                self.assertEqual(payload, expected_payload)
                self.assertEqual(linearize_volume_4k_s(payload, width, height, depth, name), expected)
                self.assertEqual(linearize_volume_4k_s(memoryview(payload), width, height, depth,
                                                       name), expected)
                slice_bytes = columns * rows * 16
                self.assertEqual(tile_volume_4k_s(
                    [expected[z * slice_bytes:(z + 1) * slice_bytes] for z in range(depth)],
                    width, height, depth, name), payload)

    def test_volume_slice_order_is_first_mismatch(self) -> None:
        name, width, height, depth = "BC7_UNORM_VOLUME", 8, 8, 4
        columns, rows = columns_rows(width, height)
        slices = [tight_volume_slice(7, width, height, z) for z in range(depth)]
        payload = tile_volume_4k_s(slices, width, height, depth, name)
        expected = b"".join(slices)

        swapped_first = payload[4096:8192] + payload[0:4096] + payload[8192:]
        self.assertEqual(first_volume_mismatch(
            linearize_volume_4k_s(swapped_first, width, height, depth, name),
            expected, columns, rows), (0, 0, 0))

        swapped_later = payload[0:4096] + payload[3 * 4096:] + payload[2 * 4096:3 * 4096] \
            + payload[4096:2 * 4096]
        self.assertEqual(first_volume_mismatch(
            linearize_volume_4k_s(swapped_later, width, height, depth, name),
            expected, columns, rows), (0, 0, 1))

        reversed_tiles = tile_volume_4k_s([slices[1], slices[0], slices[3], slices[2]],
                                          width, height, depth, name)
        self.assertEqual(first_volume_mismatch(reversed_tiles, payload, columns, rows), (0, 0, 0))

    def test_volume_size_and_depth_controls(self) -> None:
        name, width, height, depth = "BC7_UNORM_VOLUME", 8, 8, 4
        slices = [tight_volume_slice(7, width, height, z) for z in range(depth)]
        payload = tile_volume_4k_s(slices, width, height, depth, name)
        with self.assertRaisesRegex(ValueError, "slice 0 of BC7_UNORM_VOLUME is 48 bytes; expected 64"):
            tile_volume_4k_s([slices[0][:-16], *slices[1:]], width, height, depth, name)
        with self.assertRaisesRegex(ValueError, "Expected 4 volume slices, got 3"):
            tile_volume_4k_s(slices[:3], width, height, depth, name)
        with self.assertRaisesRegex(ValueError, "Expected 4 slices of 4096 bytes"):
            linearize_volume_4k_s(payload[:-4096], width, height, depth, name)
        with self.assertRaisesRegex(ValueError, "Expected 4 slices of 4096 bytes"):
            linearize_volume_4k_s(payload + bytes(4096), width, height, depth, name)
        with self.assertRaisesRegex(ValueError, "Expected 3 slices of 4096 bytes"):
            linearize_volume_4k_s(payload, width, height, depth - 1, name)
        for bad_depth in (0, -1):
            with self.assertRaises(ValueError):
                tile_volume_4k_s(slices, width, height, bad_depth, name)
            with self.assertRaises(ValueError):
                linearize_volume_4k_s(payload, width, height, bad_depth, name)
        # Zero depth is refused even when the slice list and payload agree with it.
        with self.assertRaisesRegex(ValueError, "Expected a positive volume depth"):
            tile_volume_4k_s([], width, height, 0, name)
        with self.assertRaisesRegex(ValueError, "Expected a positive volume depth"):
            linearize_volume_4k_s(b"", width, height, 0, name)


if __name__ == "__main__":
    unittest.main()
