"""Verify primitive topology 0x11 (RECTLIST) geometry expansion, winding, and oracle.

Refs #870.
"""

from pathlib import Path
import struct
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from rectlist_topology import (
    compute_signed_area_2d,
    convert_indirect_draw_arguments,
    derive_fourth_vertex,
    expand_indices,
    expand_rectangles,
    rect_raster_oracle,
    validate_corner_order,
)


class RectlistTopologyTests(unittest.TestCase):
    def test_derive_fourth_vertex_fullscreen(self):
        # Full-screen NDC rectangle in standard CCW order:
        # v0: bottom-left (-1, -1)
        # v1: bottom-right (1, -1)
        # v2: top-right (1, 1)
        # Expected derived 4th corner: top-left (-1, 1) = v0 + v2 - v1
        v0 = [-1.0, -1.0, 0.0, 1.0]
        v1 = [1.0, -1.0, 0.0, 1.0]
        v2 = [1.0, 1.0, 0.0, 1.0]
        v3 = derive_fourth_vertex(v0, v1, v2)
        self.assertAlmostEqual(v3[0], -1.0)
        self.assertAlmostEqual(v3[1], 1.0)
        self.assertAlmostEqual(v3[2], 0.0)
        self.assertAlmostEqual(v3[3], 1.0)

    def test_derive_fourth_vertex_with_uv_attributes(self):
        # 6 attributes: x, y, z, w, u, v
        # v0: (-0.5, -0.5), uv = (0, 0)
        # v1: (0.5, -0.5), uv = (1, 0)
        # v2: (0.5, 0.5), uv = (1, 1)
        # Derived v3: (-0.5, 0.5), uv = (0, 1)
        v0 = [-0.5, -0.5, 0.0, 1.0, 0.0, 0.0]
        v1 = [0.5, -0.5, 0.0, 1.0, 1.0, 0.0]
        v2 = [0.5, 0.5, 0.0, 1.0, 1.0, 1.0]
        v3 = derive_fourth_vertex(v0, v1, v2)
        expected = [-0.5, 0.5, 0.0, 1.0, 0.0, 1.0]
        for a, b in zip(v3, expected):
            self.assertAlmostEqual(a, b)

    def test_derive_fourth_vertex_parallelogram(self):
        # General non-axis-aligned parallelogram
        # v1 = (0, 0), v0 = (2, 1), v2 = (1, 3)
        # v3 = (2, 1) + (1, 3) - (0, 0) = (3, 4)
        v0 = [2.0, 1.0]
        v1 = [0.0, 0.0]
        v2 = [1.0, 3.0]
        v3 = derive_fourth_vertex(v0, v1, v2)
        self.assertEqual(v3, [3.0, 4.0])

    def test_derive_attribute_length_mismatch_refused(self):
        with self.assertRaises(ValueError):
            derive_fourth_vertex([1.0, 2.0], [1.0], [1.0, 2.0])

    def test_algebraic_signed_area_equality(self):
        # Verify that for any valid rectangle, Triangle 0 (v0, v1, v2)
        # and Triangle 1 (v0, v2, v3) have algebraically identical signed area.
        test_rects = [
            ([-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]),
            ([-0.5, -0.2], [0.8, -0.2], [0.8, 0.6]),
            ([0.0, 1.0], [2.0, 3.0], [1.0, 4.0]),
        ]
        for v0, v1, v2 in test_rects:
            v3 = derive_fourth_vertex(v0, v1, v2)
            area_t0 = compute_signed_area_2d(v0, v1, v2)
            area_t1 = compute_signed_area_2d(v0, v2, v3)
            self.assertAlmostEqual(area_t0, area_t1)
            self.assertGreater(area_t0, 0.0)

    def test_corner_order_reversal_inverts_both_triangles(self):
        # Swapping v0 and v2 produces clockwise output for both triangles.
        v0 = [-1.0, -1.0]
        v1 = [1.0, -1.0]
        v2 = [1.0, 1.0]
        # Inverted / reversed order
        v0_rev, v2_rev = v2, v0
        v3_rev = derive_fourth_vertex(v0_rev, v1, v2_rev)
        area_rev_t0 = compute_signed_area_2d(v0_rev, v1, v2_rev)
        area_rev_t1 = compute_signed_area_2d(v0_rev, v2_rev, v3_rev)
        self.assertAlmostEqual(area_rev_t0, area_rev_t1)
        self.assertLess(area_rev_t0, 0.0)

    def test_validate_corner_order_precondition(self):
        # Valid CCW rectangle passes
        v0 = [-1.0, -1.0]
        v1 = [1.0, -1.0]
        v2 = [1.0, 1.0]
        area = validate_corner_order(v0, v1, v2, require_ccw=True, require_orthogonal=True)
        self.assertGreater(area, 0.0)

        # Inverted CW rectangle raises ValueError under require_ccw=True
        with self.assertRaises(ValueError) as ctx:
            validate_corner_order(v2, v1, v0, require_ccw=True)
        self.assertIn("counter-clockwise winding precondition", str(ctx.exception))

        # Collinear vertices raise ValueError
        with self.assertRaises(ValueError) as ctx:
            validate_corner_order([0.0, 0.0], [1.0, 1.0], [2.0, 2.0])
        self.assertIn("Degenerate rectangle", str(ctx.exception))

        # Non-orthogonal rectangle raises ValueError under require_orthogonal=True
        with self.assertRaises(ValueError) as ctx:
            validate_corner_order([0.0, 0.0], [2.0, 0.0], [1.0, 2.0], require_orthogonal=True)
        self.assertIn("not orthogonal", str(ctx.exception))

    def test_expand_rectangles_count_refusals(self):
        # Vertex counts not divisible by 3 must be refused
        for count in (1, 2, 4, 5, 7, 8):
            verts = [[0.0, 0.0]] * count
            with self.assertRaises(ValueError) as ctx:
                expand_rectangles(verts)
            self.assertIn("multiple of 3", str(ctx.exception))

    def test_expand_rectangles_single(self):
        v0 = [-1.0, -1.0]
        v1 = [1.0, -1.0]
        v2 = [1.0, 1.0]
        triangles = expand_rectangles([v0, v1, v2], validate_winding=True)
        self.assertEqual(len(triangles), 6)
        # Triangle 0: (v0, v1, v2)
        self.assertEqual(triangles[0], v0)
        self.assertEqual(triangles[1], v1)
        self.assertEqual(triangles[2], v2)
        # Triangle 1: (v0, v2, v3) where v3 = (-1, 1)
        self.assertEqual(triangles[3], v0)
        self.assertEqual(triangles[4], v2)
        self.assertEqual(triangles[5], [-1.0, 1.0])

        # validate_winding=True refuses the clockwise order (v2, v1, v0);
        # without the flag the same corners still expand.
        with self.assertRaises(ValueError) as ctx:
            expand_rectangles([v2, v1, v0], validate_winding=True)
        self.assertIn("counter-clockwise winding precondition", str(ctx.exception))
        self.assertEqual(len(expand_rectangles([v2, v1, v0])), 6)

    def test_expand_rectangles_multiple(self):
        rects = [
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],  # Rect 0
            [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],  # Rect 1
        ]
        expanded = expand_rectangles(rects)
        self.assertEqual(len(expanded), 12)
        # Rect 1 reads its own corners 3..5; v3 = (-0.5, -0.5) + (0.5, 0.5) - (0.5, -0.5).
        self.assertEqual(
            expanded[6:],
            [
                [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                [-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5],
            ],
        )

    def test_expand_indices(self):
        indices = expand_indices(2)
        self.assertEqual(indices, [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7])
        # base_vertex offsets every index: rect i starts at 8 + 4 * i.
        self.assertEqual(
            expand_indices(2, base_vertex=8),
            [8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15],
        )

    def test_raster_oracle_fullscreen(self):
        oracle = rect_raster_oracle([((-1.0, -1.0), (1.0, 1.0))], 32, 32)
        self.assertEqual(len(oracle), 32 * 32 * 4)
        # Every pixel must be red (foreground: 255, 0, 0, 255)
        expected_pixel = bytes([255, 0, 0, 255])
        for i in range(32 * 32):
            self.assertEqual(oracle[i * 4 : (i + 1) * 4], expected_pixel)

    def test_raster_oracle_quadrant(self):
        # Top-right quadrant [0, 1] x [0, 1]
        oracle = rect_raster_oracle([((0.0, 0.0), (1.0, 1.0))], 32, 32)
        fg_count = 0
        bg_count = 0
        fg = bytes([255, 0, 0, 255])
        bg = bytes([0, 0, 0, 255])
        for i in range(32 * 32):
            pix = oracle[i * 4 : (i + 1) * 4]
            if pix == fg:
                fg_count += 1
            elif pix == bg:
                bg_count += 1
        # Exactly 16 x 16 = 256 pixels in quadrant
        self.assertEqual(fg_count, 256)
        self.assertEqual(bg_count, 768)
        # Row 0 is the top (+y) and column 0 the left (-x) edge, so the quadrant
        # covers x >= 16, y < 16: x = 16 has x_ndc = 33 / 32 - 1 = 1 / 32 and
        # y = 15 has y_ndc = 1 - 31 / 32 = 1 / 32; x = 15 and y = 16 fall at -1 / 32.
        for y in range(32):
            for x in range(32):
                expected = fg if x >= 16 and y < 16 else bg
                offset = (y * 32 + x) * 4
                self.assertEqual(oracle[offset : offset + 4], expected, (x, y))

    def test_raster_oracle_mutant_detection(self):
        baseline = rect_raster_oracle([((0.0, 0.0), (1.0, 1.0))], 16, 16)
        mutant = rect_raster_oracle([((-0.5, -0.5), (0.5, 0.5))], 16, 16)
        self.assertNotEqual(baseline, mutant)


class IndirectArgumentTests(unittest.TestCase):
    def test_counts_become_six_vertices_per_rectangle(self):
        self.assertEqual(convert_indirect_draw_arguments((3, 1, 0, 0)), (6, 1, 0, 0))
        self.assertEqual(convert_indirect_draw_arguments((6, 5, 7, 9)), (12, 5, 7, 9))

    def test_partial_rectangle_is_dropped_like_the_direct_draw(self):
        self.assertEqual(convert_indirect_draw_arguments((4, 1, 0, 0))[0], 6)
        self.assertEqual(convert_indirect_draw_arguments((2, 1, 0, 0))[0], 0)

    def test_overflowing_count_draws_nothing(self):
        self.assertEqual(convert_indirect_draw_arguments((0xFFFFFFFF, 1, 0, 0))[0], 0)
        self.assertEqual(convert_indirect_draw_arguments((0x2AAAAAAA * 3, 1, 0, 0))[0], 0x2AAAAAAA * 6)
        self.assertEqual(convert_indirect_draw_arguments((0x2AAAAAAB * 3, 1, 0, 0))[0], 0)

    def test_malformed_record_is_refused(self):
        with self.assertRaises(ValueError):
            convert_indirect_draw_arguments((3, 1, 0))
        with self.assertRaises(ValueError):
            convert_indirect_draw_arguments((3, 1, 0, 1 << 32))


def run_draw(renderer: str, backend: str) -> int:
    corners = (
        (-0.5, -0.5, 0.0, 1.0, 0.2, 0.1, 0.5, 1.0),
        (0.5, -0.5, 0.0, 1.0, 0.8, 0.1, 0.5, 1.0),
        (0.5, 0.5, 0.0, 1.0, 0.8, 0.9, 0.5, 1.0),
    )
    triangles = expand_rectangles(corners, validate_winding=True)
    vertices = b"".join(struct.pack("<8f", *vertex) for vertex in triangles)
    return subprocess.run([renderer, backend], input=vertices, timeout=45).returncode


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--draw":
        sys.exit(run_draw(*sys.argv[2:]))
    unittest.main()
