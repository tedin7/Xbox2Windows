#!/usr/bin/env python3
"""Standalone unit, bounds-guard, and mutation suite for tools/texture_blocks.py."""

from __future__ import annotations

import array
from pathlib import Path
import struct
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "texture_blocks.py"
sys.path.insert(0, str(ROOT / "tools"))

import texture_blocks  # noqa: E402


BC_2D_FORMATS = (
    ("BC1_UNORM", 71, 8),
    ("BC2_UNORM", 74, 16),
    ("BC3_UNORM", 77, 16),
    ("BC4_UNORM", 79, 8),
    ("BC5_UNORM", 83, 16),
    ("BC6H_UF16", 95, 16),
    ("BC7_UNORM", 98, 16),
)


def _block_bytes_tag(fam: int, x: int, y: int, block_bytes: int, z: int = 0) -> bytes:
    head = struct.pack("<BBHI", fam, x & 0xFF, y & 0xFFFF, z)
    return head[:block_bytes] + bytes(((fam * 17 + x * 3 + y * 5 + z + i) & 0xFF) for i in range(block_bytes - len(head)))


def _columns_rows(width: int, height: int) -> tuple[int, int]:
    return (width + 3) // 4, (height + 3) // 4


class TextureBlocksPositiveTests(unittest.TestCase):
    def test_formats_table_preserved(self) -> None:
        for name, dxgi, bpb in BC_2D_FORMATS:
            with self.subTest(format=name):
                info = texture_blocks.format_info(name)
                self.assertEqual(info.dxgi_format, dxgi)
                self.assertEqual(info.block_extent, (4, 4, 1))
                self.assertEqual(info.bytes_per_block, bpb)
                self.assertEqual(info.dimension, 2)
        vol = texture_blocks.format_info("BC7_UNORM_VOLUME")
        self.assertEqual((vol.dxgi_format, vol.block_extent, vol.bytes_per_block, vol.dimension),
                         (98, (4, 4, 1), 16, 3))

    def test_2d_bc1_to_bc7_round_trips_non_mult4_and_pitched(self) -> None:
        dims = ((1, 1, 0), (3, 5, 1), (5, 3, 2), (17, 9, 3), (65, 19, 4))
        for fam, (name, _, bpb) in enumerate(BC_2D_FORMATS, start=1):
            custom_pad = bytes([(0xA0 + fam) & 0xFF]) * bpb
            for width, height, extra in dims:
                cols, rows = _columns_rows(width, height)
                pitch = cols + extra
                tight = b"".join(_block_bytes_tag(fam, x, y, bpb) for y in range(rows) for x in range(cols))
                with self.subTest(format=name, width=width, height=height, pitch=pitch):
                    self.assertEqual(texture_blocks.surface_bytes(width, height, name), cols * rows * bpb)
                    self.assertEqual(
                        texture_blocks.surface_bytes(width, height, name, pitch_blocks=pitch),
                        pitch * rows * bpb,
                    )

                    tiled_tight = texture_blocks.tile_blocks(tight, width, height, name)
                    self.assertEqual(tiled_tight, tight)
                    self.assertEqual(texture_blocks.linearize_blocks(tiled_tight, width, height, name), tight)

                    for pad in (custom_pad, bytearray(custom_pad), memoryview(custom_pad)):
                        tiled = texture_blocks.tile_blocks(
                            tight,
                            width,
                            height,
                            name,
                            pitch_blocks=pitch,
                            padding=pad,
                        )
                        self.assertEqual(len(tiled), pitch * rows * bpb)
                        if extra > 0:
                            for y in range(rows):
                                row_tail_start = (y * pitch + cols) * bpb
                                row_tail_end = ((y + 1) * pitch) * bpb
                                self.assertEqual(tiled[row_tail_start:row_tail_end], custom_pad * extra)
                        self.assertEqual(
                            texture_blocks.linearize_blocks(memoryview(tiled), width, height, name, pitch_blocks=pitch),
                            tight,
                        )

    def test_3d_volume_round_trip_with_non_mult4_dims(self) -> None:
        for width, height, depth in ((1, 1, 1), (3, 5, 3), (65, 19, 2)):
            cols, rows = _columns_rows(width, height)
            slices = [
                b"".join(_block_bytes_tag(7, x, y, 16, z) for y in range(rows) for x in range(cols))
                for z in range(depth)
            ]
            expected_tight = b"".join(slices)
            with self.subTest(width=width, height=height, depth=depth):
                payload = texture_blocks.tile_volume_4k_s(
                    (s for s in slices),
                    width,
                    height,
                    depth,
                    "BC7_UNORM_VOLUME",
                )
                tile_pitch = (cols + 15) // 16
                slice_stride = tile_pitch * ((rows + 15) // 16) * 4096
                self.assertEqual(len(payload), depth * slice_stride)
                self.assertEqual(
                    texture_blocks.linearize_volume_4k_s(payload, width, height, depth, "BC7_UNORM_VOLUME"),
                    expected_tight,
                )
                self.assertEqual(
                    texture_blocks.linearize_volume_4k_s(memoryview(payload), width, height, depth, "BC7_UNORM_VOLUME"),
                    expected_tight,
                )


class TextureBlocksNegativeAndMutationTests(unittest.TestCase):
    def test_bool_non_int_and_non_positive_dims_and_pitch_rejected(self) -> None:
        fmt = texture_blocks.format_info("BC7_UNORM")
        tight_4x4 = bytes(16)
        padded_4x4 = bytes(32)
        vol_slice = bytes(16)
        vol_payload = bytes(4096)

        for bad_dim in (True, False, 0, -1, -8, 1.5, "4", None):
            with self.subTest(bad_dim=bad_dim):
                with self.assertRaises(ValueError):
                    texture_blocks._grid(bad_dim, 4, fmt)
                with self.assertRaises(ValueError):
                    texture_blocks._grid(4, bad_dim, fmt)
                with self.assertRaises(ValueError):
                    texture_blocks.surface_bytes(bad_dim, 4, "BC7_UNORM")
                with self.assertRaises(ValueError):
                    texture_blocks.surface_bytes(4, bad_dim, "BC7_UNORM")
                with self.assertRaises(ValueError):
                    texture_blocks.linearize_blocks(tight_4x4, bad_dim, 4, "BC7_UNORM")
                with self.assertRaises(ValueError):
                    texture_blocks.linearize_blocks(tight_4x4, 4, bad_dim, "BC7_UNORM")
                with self.assertRaises(ValueError):
                    texture_blocks.tile_blocks(tight_4x4, bad_dim, 4, "BC7_UNORM")
                with self.assertRaises(ValueError):
                    texture_blocks.tile_blocks(tight_4x4, 4, bad_dim, "BC7_UNORM")
                with self.assertRaises(ValueError):
                    texture_blocks.tile_volume_4k_s([vol_slice], bad_dim, 4, 1, "BC7_UNORM_VOLUME")
                with self.assertRaises(ValueError):
                    texture_blocks.tile_volume_4k_s([vol_slice], 4, bad_dim, 1, "BC7_UNORM_VOLUME")
                with self.assertRaises(ValueError):
                    texture_blocks.tile_volume_4k_s([vol_slice], 4, 4, bad_dim, "BC7_UNORM_VOLUME")
                with self.assertRaises(ValueError):
                    texture_blocks.linearize_volume_4k_s(vol_payload, bad_dim, 4, 1, "BC7_UNORM_VOLUME")
                with self.assertRaises(ValueError):
                    texture_blocks.linearize_volume_4k_s(vol_payload, 4, bad_dim, 1, "BC7_UNORM_VOLUME")
                with self.assertRaises(ValueError):
                    texture_blocks.linearize_volume_4k_s(vol_payload, 4, 4, bad_dim, "BC7_UNORM_VOLUME")

        for bad_pitch in (True, False, 0, 1, -1, 2.0, "2"):
            with self.subTest(bad_pitch=bad_pitch):
                with self.assertRaises(ValueError):
                    texture_blocks.surface_bytes(8, 4, "BC7_UNORM", pitch_blocks=bad_pitch)
                with self.assertRaises(ValueError):
                    texture_blocks.linearize_blocks(padded_4x4, 8, 4, "BC7_UNORM", pitch_blocks=bad_pitch)
                with self.assertRaises(ValueError):
                    texture_blocks.tile_blocks(padded_4x4, 8, 4, "BC7_UNORM", pitch_blocks=bad_pitch)

    def test_wrong_payload_padding_and_format_mismatches_rejected(self) -> None:
        tight_bc1 = bytes(16)  # 8x4 BC1_UNORM = 2 blocks * 8 bytes
        for bad_len_payload in (b"", bytes(15), bytes(17), bytes(24)):
            with self.subTest(bad_len=len(bad_len_payload)):
                with self.assertRaises(ValueError):
                    texture_blocks.linearize_blocks(bad_len_payload, 8, 4, "BC1_UNORM")
                with self.assertRaises(ValueError):
                    texture_blocks.tile_blocks(bad_len_payload, 8, 4, "BC1_UNORM")

        int_view = memoryview(array.array("I", [1, 2]))  # nbytes=8 but format != 'B'
        for bad_pad in (b"", b"x" * 7, b"x" * 9, "x" * 8, [0] * 8, (0,) * 8, 123, int_view):
            with self.subTest(bad_pad=repr(bad_pad)):
                with self.assertRaises(ValueError):
                    texture_blocks.tile_blocks(tight_bc1, 8, 4, "BC1_UNORM", pitch_blocks=3, padding=bad_pad)
        with self.assertRaisesRegex(ValueError, "Expected one 8-byte padding block"):
            texture_blocks.tile_blocks(bytes(8), 4, 4, "BC1_UNORM", pitch_blocks=1, padding=b"")

        # Volume slice count and payload length mismatches.
        with self.assertRaises(ValueError):
            texture_blocks.tile_volume_4k_s([bytes(16)], 4, 4, 2, "BC7_UNORM_VOLUME")
        with self.assertRaises(ValueError):
            texture_blocks.tile_volume_4k_s([bytes(15)], 4, 4, 1, "BC7_UNORM_VOLUME")
        with self.assertRaises(ValueError):
            texture_blocks.linearize_volume_4k_s(bytes(4095), 4, 4, 1, "BC7_UNORM_VOLUME")

        # 2D vs 3D and unknown format mismatches.
        with self.assertRaisesRegex(ValueError, "volume format"):
            texture_blocks.surface_bytes(4, 4, "BC7_UNORM_VOLUME")
        with self.assertRaisesRegex(ValueError, "volume format"):
            texture_blocks.tile_blocks(bytes(16), 4, 4, "BC7_UNORM_VOLUME")
        with self.assertRaisesRegex(ValueError, "2-D format"):
            texture_blocks.tile_volume_4k_s([bytes(16)], 4, 4, 1, "BC7_UNORM")
        with self.assertRaisesRegex(ValueError, "2-D format"):
            texture_blocks.linearize_volume_4k_s(bytes(4096), 4, 4, 1, "BC1_UNORM")
        for unknown in ("UNKNOWN_BC", 123, None, ["BC7_UNORM"]):
            with self.subTest(unknown=unknown):
                with self.assertRaises(ValueError):
                    texture_blocks.format_info(unknown)

    def test_fast_temp_mutations(self) -> None:
        source = TOOL.read_text(encoding="utf-8")

        def _check_bool_mutation(mod) -> None:
            self.assertEqual(mod.surface_bytes(True, 4, "BC7_UNORM"), 16)

        def _check_narrow_pitch_mutation(mod) -> None:
            self.assertEqual(mod.surface_bytes(8, 4, "BC7_UNORM", pitch_blocks=1), 16)

        def _check_padding_type_mutation(mod) -> None:
            out = mod.tile_blocks(bytes(8), 4, 4, "BC1_UNORM", pitch_blocks=2, padding=[0x5A] * 8)
            self.assertEqual(len(out), 16)

        def _check_padding_len_mutation(mod) -> None:
            out = mod.tile_blocks(bytes(8), 4, 4, "BC1_UNORM", pitch_blocks=1, padding=b"")
            self.assertEqual(out, bytes(8))

        def _check_depth_positive_mutation(mod) -> None:
            self.assertEqual(mod.linearize_volume_4k_s(b"", 4, 4, 0, "BC7_UNORM_VOLUME"), b"")

        mutations = (
            (
                "drop_bool_guard",
                "    if isinstance(value, bool) or not isinstance(value, int):\n",
                "    if not isinstance(value, int):\n",
                _check_bool_mutation,
            ),
            (
                "drop_narrow_pitch_guard",
                '    if pitch < columns:\n        raise ValueError(f"Block pitch {pitch} is narrower than the {columns}-block surface")\n',
                "    pass\n",
                _check_narrow_pitch_mutation,
            ),
            (
                "drop_padding_type_guard",
                '    elif isinstance(padding, (bytes, bytearray)):\n'
                '        padding_bytes = bytes(padding)\n'
                '    elif isinstance(padding, memoryview) and padding.ndim == 1 and padding.format == "B":\n'
                '        padding_bytes = bytes(padding)\n'
                '    else:\n'
                '        raise ValueError(f"Expected bytes-like padding of {block_bytes} bytes, got {type(padding).__name__}")\n',
                "    else:\n        padding_bytes = padding\n",
                _check_padding_type_mutation,
            ),
            (
                "drop_padding_len_guard",
                '    if len(padding_bytes) != block_bytes:\n        raise ValueError(f"Expected one {block_bytes}-byte padding block")\n',
                "    pass\n",
                _check_padding_len_mutation,
            ),
            (
                "drop_depth_positive_guard",
                '    if depth <= 0:\n        raise ValueError("Expected a positive volume depth")\n',
                "    pass\n",
                _check_depth_positive_mutation,
            ),
        )

        for label, target, replacement, check_fn in mutations:
            with self.subTest(mutation=label):
                count = source.count(target)
                self.assertGreaterEqual(count, 1, target)
                # For depth guard, target appears twice (tile_volume_4k_s and linearize_volume_4k_s);
                # mutate the linearize_volume_4k_s guard or replace all for clean mutant check.
                mutated_src = source.replace(target, replacement)
                mod = types.ModuleType(f"texture_blocks_mut_{label}")
                mod.__file__ = str(TOOL)
                sys.modules[mod.__name__] = mod
                try:
                    exec(compile(mutated_src, str(TOOL), "exec"), mod.__dict__)
                    check_fn(mod)
                finally:
                    sys.modules.pop(mod.__name__, None)


if __name__ == "__main__":
    unittest.main()
