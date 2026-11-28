"""Read back synthetic BC block rows and a slice-major BC7 volume.

The format table records the public DXGI block sizes. The volume helpers apply
the tested GFX10 4 KiB two-dimensional equation to each Z slice; the XG/GFX10
three-dimensional address equation is not public, so this module makes no
console volume-layout claim.
"""

from dataclasses import dataclass

import bc7_tiles


@dataclass(frozen=True)
class Format:
    name: str
    dxgi_format: int
    block_width: int
    block_height: int
    block_depth: int
    bytes_per_block: int
    dimension: int

    @property
    def block_extent(self) -> tuple[int, int, int]:
        return self.block_width, self.block_height, self.block_depth


# Canonical DXGI UNORM/UF16 entries, one per BC family. BC7_UNORM_VOLUME is the
# tool-level three-dimensional reading of BC7_UNORM: the same DXGI format with a
# one-texel block depth. It is not a DXGI format name.
FORMATS: dict[str, Format] = {
    "BC1_UNORM": Format("BC1_UNORM", 71, 4, 4, 1, 8, 2),
    "BC2_UNORM": Format("BC2_UNORM", 74, 4, 4, 1, 16, 2),
    "BC3_UNORM": Format("BC3_UNORM", 77, 4, 4, 1, 16, 2),
    "BC4_UNORM": Format("BC4_UNORM", 80, 4, 4, 1, 8, 2),
    "BC5_UNORM": Format("BC5_UNORM", 83, 4, 4, 1, 16, 2),
    "BC6H_UF16": Format("BC6H_UF16", 95, 4, 4, 1, 16, 2),
    "BC7_UNORM": Format("BC7_UNORM", 98, 4, 4, 1, 16, 2),
    "BC7_UNORM_VOLUME": Format("BC7_UNORM_VOLUME", 98, 4, 4, 1, 16, 3),
}


def format_info(name: str) -> Format:
    if not isinstance(name, str) or name not in FORMATS:
        raise ValueError(f"Unknown texture format {name!r}")
    return FORMATS[name]


def _require_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer {name}, got {type(value).__name__}")
    return value


def _grid(width: int, height: int, format: Format) -> tuple[int, int]:
    width = _require_int(width, "width")
    height = _require_int(height, "height")
    if width <= 0 or height <= 0:
        raise ValueError("Expected positive pixel dimensions")
    return ((width + format.block_width - 1) // format.block_width,
            (height + format.block_height - 1) // format.block_height)


def _two_dimensional(name: str) -> Format:
    format = format_info(name)
    if format.dimension != 2:
        raise ValueError(f"{format.name} is a volume format; a 2-D format is required")
    return format


def _volume(name: str) -> Format:
    format = format_info(name)
    if format.dimension != 3:
        raise ValueError(f"{format.name} is a 2-D format; a volume format is required")
    if (format.block_extent, format.bytes_per_block) != ((4, 4, 1), 16):
        raise ValueError(f"{format.name} cannot use the 4 KiB block equation")
    return format


def surface_bytes(width: int, height: int, format_name: str, *, pitch_blocks: int | None = None) -> int:
    """Bytes in one block-row surface, including any row pitch padding."""
    format = _two_dimensional(format_name)
    columns, rows = _grid(width, height, format)
    pitch = columns if pitch_blocks is None else _require_int(pitch_blocks, "pitch_blocks")
    if pitch < columns:
        raise ValueError(f"Block pitch {pitch} is narrower than the {columns}-block surface")
    return rows * pitch * format.bytes_per_block


def linearize_blocks(payload, width: int, height: int, format_name: str, *,
                     pitch_blocks: int | None = None) -> bytes:
    """Copy the visible blocks of a pitched block-row surface into tight rows."""
    format = _two_dimensional(format_name)
    columns, rows = _grid(width, height, format)
    pitch = columns if pitch_blocks is None else _require_int(pitch_blocks, "pitch_blocks")
    expected = surface_bytes(width, height, format_name, pitch_blocks=pitch)
    if len(payload) != expected:
        raise ValueError(f"Expected {expected} bytes for {format.name}, got {len(payload)}")
    block_bytes = format.bytes_per_block
    row_bytes = columns * block_bytes
    output = bytearray(rows * row_bytes)
    for y in range(rows):
        source = (y * pitch) * block_bytes
        output[y * row_bytes:(y + 1) * row_bytes] = payload[source:source + row_bytes]
    return bytes(output)


def tile_blocks(blocks, width: int, height: int, format_name: str, *,
                pitch_blocks: int | None = None, padding: bytes | bytearray | memoryview | None = None) -> bytes:
    """Write tight block rows into a pitched block-row surface."""
    format = _two_dimensional(format_name)
    columns, rows = _grid(width, height, format)
    pitch = columns if pitch_blocks is None else _require_int(pitch_blocks, "pitch_blocks")
    expected = surface_bytes(width, height, format_name, pitch_blocks=pitch)
    block_bytes = format.bytes_per_block
    tight = columns * rows * block_bytes
    if len(blocks) != tight:
        raise ValueError(f"Expected {tight} tight bytes for {format.name}, got {len(blocks)}")
    if padding is None:
        padding_bytes = b"\xff" * block_bytes
    elif isinstance(padding, (bytes, bytearray)):
        padding_bytes = bytes(padding)
    elif isinstance(padding, memoryview) and padding.ndim == 1 and padding.format == "B":
        padding_bytes = bytes(padding)
    else:
        raise ValueError(f"Expected bytes-like padding of {block_bytes} bytes, got {type(padding).__name__}")
    if len(padding_bytes) != block_bytes:
        raise ValueError(f"Expected one {block_bytes}-byte padding block")
    output = bytearray(padding_bytes * (rows * pitch))
    row_bytes = columns * block_bytes
    for y in range(rows):
        source = y * row_bytes
        destination = (y * pitch) * block_bytes
        output[destination:destination + row_bytes] = blocks[source:source + row_bytes]
    if len(output) != expected:
        raise ValueError(f"Expected {expected} bytes for {format.name}, got {len(output)}")
    return bytes(output)


def _volume_geometry(width: int, height: int, format: Format) -> tuple[int, int, int, int]:
    columns, rows = _grid(width, height, format)
    slice_bytes = columns * rows * format.bytes_per_block
    tile_pitch = (columns + 15) // 16
    slice_stride = tile_pitch * ((rows + 15) // 16) * 4096
    return columns, rows, slice_bytes, slice_stride


def tile_volume_4k_s(slices, width: int, height: int, depth: int, format_name: str) -> bytes:
    """Tile tight BC7 volume slices, one padded 4 KiB surface per Z slice."""
    format = _volume(format_name)
    depth = _require_int(depth, "depth")
    if depth <= 0:
        raise ValueError("Expected a positive volume depth")
    _, _, slice_bytes, _ = _volume_geometry(width, height, format)
    slices = list(slices)
    if len(slices) != depth:
        raise ValueError(f"Expected {depth} volume slices, got {len(slices)}")
    output = bytearray()
    for z, blocks in enumerate(slices):
        if len(blocks) != slice_bytes:
            raise ValueError(f"Volume slice {z} of {format.name} is {len(blocks)} bytes; "
                             f"expected {slice_bytes}")
        output.extend(bc7_tiles.tile_4k_s(blocks, width, height))
    return bytes(output)


def linearize_volume_4k_s(payload, width: int, height: int, depth: int, format_name: str) -> bytes:
    """Read tight BC7 volume slices from one 4 KiB surface per Z slice."""
    format = _volume(format_name)
    depth = _require_int(depth, "depth")
    if depth <= 0:
        raise ValueError("Expected a positive volume depth")
    _, _, _, slice_stride = _volume_geometry(width, height, format)
    expected = depth * slice_stride
    if len(payload) != expected:
        raise ValueError(f"Expected {depth} slices of {slice_stride} bytes "
                         f"({expected} total) for {format.name}, got {len(payload)}")
    return b"".join(
        bc7_tiles.linearize_4k_s(payload[z * slice_stride:(z + 1) * slice_stride], width, height)
        for z in range(depth)
    )
