"""Primitive topology 0x11 (RECTLIST) geometry expansion and oracle.

AMD GCN/RDNA hardware architecture specifies VGT_PRIMITIVE_TYPE_RECTLIST = 0x11
(17 decimal). Three vertices define a rectangle:
  v0: primary origin corner
  v1: adjacent corner sharing an edge with v0
  v2: corner opposite v0, sharing an edge with v1

The fourth corner is automatically synthesized by the hardware as:
  v3 = v0 + v2 - v1

Corner Order and Winding Specification:
  - Input Precondition: The input vertices must be supplied in defined sequence
    (v0, v1, v2) where v0 and v1 are adjacent corners, and v2 is diagonally
    opposite to v0 and adjacent to v1.
  - Winding Preservation: Lowering expands the rectangle into two triangles:
      Triangle 0: (v0, v1, v2)
      Triangle 1: (v0, v2, v3) = (v0, v2, v0 + v2 - v1)
    The winding order of both lowered triangles strictly preserves the winding
    of the input triangle (v0, v1, v2).
    In 2D coordinates (x, y), the signed cross product:
      cross_2d(v0, v1, v2) = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    For Triangle 1:
      cross_2d(v0, v2, v3) = (x2 - x0) * (y3 - y0) - (y2 - y0) * (x3 - x0)
    Substituting (v3 - v0) = (v2 - v1):
      cross_2d(v0, v2, v3) = (x2 - x0) * (y2 - y1) - (y2 - y0) * (x2 - x1)
                           = cross_2d(v0, v1, v2)
    Thus, both triangles possess algebraically identical signed area in magnitude
    and sign. A counter-clockwise (CCW, front-facing) input yields two CCW triangles;
    exchanging v0 and v2 yields clockwise (CW, back-facing) triangles, which are
    discarded under desktop D3D12 backface culling (D3D12_CULL_MODE_BACK).
"""

from typing import List, Sequence, Tuple


def compute_signed_area_2d(
    v0: Sequence[float], v1: Sequence[float], v2: Sequence[float]
) -> float:
    """Compute the 2D signed cross product z-component: (v1 - v0) x (v2 - v0)."""
    return (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])


def validate_corner_order(
    v0: Sequence[float],
    v1: Sequence[float],
    v2: Sequence[float],
    require_ccw: bool = True,
    require_orthogonal: bool = False,
    tolerance: float = 1e-5,
) -> float:
    """Validate rectangle corner order precondition and return signed 2D area.
    
    Raises ValueError if vertices are degenerate, violate CCW winding (when require_ccw=True),
    or edges (v1 - v0) and (v2 - v1) are not orthogonal (when require_orthogonal=True).
    """
    area = compute_signed_area_2d(v0, v1, v2)
    if abs(area) < tolerance:
        raise ValueError("Degenerate rectangle: collinear vertices (zero area)")
    if require_ccw and area < 0.0:
        raise ValueError(
            f"Input vertices violate counter-clockwise winding precondition (signed area = {area})"
        )
    if require_orthogonal:
        # Dot product of edge 0 (v1 - v0) and edge 1 (v2 - v1)
        dot = (v1[0] - v0[0]) * (v2[0] - v1[0]) + (v1[1] - v0[1]) * (v2[1] - v1[1])
        if abs(dot) > tolerance:
            raise ValueError(
                f"Edges are not orthogonal: dot product = {dot} (tolerance = {tolerance})"
            )
    return area


def derive_fourth_vertex(
    v0: Sequence[float], v1: Sequence[float], v2: Sequence[float]
) -> List[float]:
    """Derive the 4th corner of a rectangle: v3 = v0 + v2 - v1 for all attributes."""
    if not (len(v0) == len(v1) == len(v2)):
        raise ValueError("Vertex attribute lengths must match")
    return [a + c - b for a, b, c in zip(v0, v1, v2)]


def expand_rectangles(
    vertices: Sequence[Sequence[float]],
    validate_winding: bool = False,
) -> List[Sequence[float]]:
    """Expand N 3-vertex rectangles into 2*N 3-vertex triangles (6 vertices per rect).
    
    Input: [v0_0, v0_1, v0_2, v1_0, v1_1, v1_2, ...]
    Output: [v0_0, v0_1, v0_2, v0_0, v0_2, v0_3, ...]
    
    The winding order of both generated triangles strictly follows the input winding.
    """
    if len(vertices) % 3 != 0:
        raise ValueError(
            f"Rectangle list vertex count must be a multiple of 3, got {len(vertices)}"
        )
    
    expanded: List[Sequence[float]] = []
    num_rects = len(vertices) // 3
    for i in range(num_rects):
        v0 = vertices[3 * i + 0]
        v1 = vertices[3 * i + 1]
        v2 = vertices[3 * i + 2]
        if validate_winding:
            validate_corner_order(v0, v1, v2, require_ccw=True)
        v3 = derive_fourth_vertex(v0, v1, v2)
        # Triangle 0: (v0, v1, v2)
        expanded.append(v0)
        expanded.append(v1)
        expanded.append(v2)
        # Triangle 1: (v0, v2, v3)
        expanded.append(v0)
        expanded.append(v2)
        expanded.append(v3)
    return expanded


def expand_indices(rect_count: int, base_vertex: int = 0) -> List[int]:
    """Generate triangle list indices for rectangle vertices.
    
    If vertices are supplied as 4 per rectangle [v0, v1, v2, v3],
    emits 6 indices per rect: [0, 1, 2,  0, 2, 3].
    """
    indices: List[int] = []
    for i in range(rect_count):
        base = base_vertex + i * 4
        indices.extend([base + 0, base + 1, base + 2, base + 0, base + 2, base + 3])
    return indices


def rect_raster_oracle(
    rectangles: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    width: int,
    height: int,
    foreground: Tuple[int, int, int, int] = (255, 0, 0, 255),
    background: Tuple[int, int, int, int] = (0, 0, 0, 255),
) -> bytes:
    """Analytical 2D pixel oracle for axis-aligned NDC rectangles [-1, 1] x [-1, 1].
    
    Each rectangle is ((min_x, min_y), (max_x, max_y)) in NDC.
    Pixel centers: x_ndc = (2 * x + 1) / width - 1, y_ndc = 1 - (2 * y + 1) / height.
    """
    pixels = bytearray()
    for y in range(height):
        y_ndc = 1.0 - (2.0 * y + 1.0) / float(height)
        for x in range(width):
            x_ndc = (2.0 * x + 1.0) / float(width) - 1.0
            hit = False
            for (min_x, min_y), (max_x, max_y) in rectangles:
                if min_x <= x_ndc <= max_x and min_y <= y_ndc <= max_y:
                    hit = True
                    break
            color = foreground if hit else background
            pixels.extend(color)
    return bytes(pixels)


MAX_CONVERTIBLE_RECTANGLES = 0x2AAAAAAA


def convert_indirect_draw_arguments(arguments: Sequence[int]) -> Tuple[int, int, int, int]:
    """Model of the GPU pass that turns a rectangle-list indirect Draw record into a triangle-list one.

    `arguments` is (VertexCountPerInstance, InstanceCount, StartVertexLocation, StartInstanceLocation).
    Three console vertices name one rectangle; the desktop draw uses six vertices per rectangle. A trailing
    partial rectangle is dropped, like the direct draw (count // 3 * 6). A count that would not fit 32 bits
    becomes 0 vertices, so nothing is drawn.
    Mirrors src/startup_stubs/rect_indirect_args.h.
    """
    if len(arguments) != 4 or any(not 0 <= word <= 0xFFFFFFFF for word in arguments):
        raise ValueError("an indirect Draw record is four 32-bit words")
    count, instances, start_vertex, start_instance = arguments
    rectangles = count // 3
    vertices = 0 if rectangles > MAX_CONVERTIBLE_RECTANGLES else rectangles * 6
    return (vertices, instances, start_vertex, start_instance)
