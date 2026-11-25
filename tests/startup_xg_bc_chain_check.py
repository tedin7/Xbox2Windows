"""Exercise six BC5/BC7 mips through XG and an address SRV on a real backend."""
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import struct
import sys
import tempfile

from startup_address_srv_check import draw_texture
from startup_xg_bc7_layout_check import PTR, HRESULT, DESC, LAYOUT_BYTES, LAYOUT, COPY_INTO, COPY_FROM, RELEASE

OFFSETS = (0x60000, 0x20000, 0x10000, 0x8000, 0x4000, 0x2000)
TOTAL = 0x160000


def color(fmt, mip, x, y):
    r = (mip * 22 + x * 6 + y * 14) & 254
    g = (mip * 38 + x * 10 + y * 26) & 254
    return bytes((r, g, 0, 255 if fmt == 83 else 254))


@lru_cache(maxsize=8192)
def block(fmt, rgba):
    if fmt == 83:
        # Two BC4 endpoints selected with index zero for every texel.
        return struct.pack("<B7xB7x", rgba[0], rgba[1])
    # BC7 mode 6 with equal even endpoints, zero p-bits and indices.
    encoded = 1 << 6
    for i, channel in enumerate(rgba):
        endpoint = channel // 2
        encoded |= (endpoint | endpoint << 7) << (7 + i * 14)
    return encoded.to_bytes(16, "little")


def address(mip, x, y):
    # Reference permutation for the fixed 128bpp, non-XOR standard tile,
    # checked against Addr2ComputeSurfaceAddrFromCoord on all 87,360 blocks.
    widths = (1, 2, 4, 8, 16, 32)
    address_bits = (0x40, 0x80, 0x200, 0x800, 0x2000, 0x8000)
    y_bits = (0x10, 0x20, 0x100, 0x400, 0x1000, 0x4000)
    inside = sum(b for flag, b in zip(widths, address_bits) if x & flag)
    inside += sum(b for flag, b in zip(widths, y_bits) if y & flag)
    tiles_wide = max(1, 4 >> mip)
    return OFFSETS[mip] + ((y // 64) * tiles_wide + x // 64) * 65536 + inside


def prepare(create, fmt, path):
    desc = ctypes.create_string_buffer(DESC.pack(3, 0, 1024, 1024, 1, 6, fmt, 1, 0, 0x109, 0))
    computer = PTR()
    assert create(desc, ctypes.byref(computer), 0, 0) == 0 and computer
    slots = ctypes.cast(computer, ctypes.POINTER(ctypes.POINTER(PTR)))[0]
    try:
        layout = ctypes.create_string_buffer(LAYOUT_BYTES)
        assert LAYOUT(slots[2])(computer, layout) == 0
        assert struct.unpack_from("<QQII", layout) == (TOTAL, 65536, 6, 1)
        offsets = tuple(struct.unpack_from("<Q", layout, 0x58 + 0x70 * i)[0] for i in range(6))
        assert offsets == OFFSETS, offsets
        size = ctypes.WINFUNCTYPE(ctypes.c_uint64, PTR)
        level_offset = ctypes.WINFUNCTYPE(ctypes.c_uint64, PTR, ctypes.c_uint, ctypes.c_uint)
        element = ctypes.WINFUNCTYPE(ctypes.c_uint64, PTR, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint64,
                                     ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, PTR)
        tiling = ctypes.WINFUNCTYPE(HRESULT, PTR, PTR, PTR, PTR, PTR, ctypes.c_uint, PTR)
        assert size(slots[3])(computer) == TOTAL and size(slots[4])(computer) == 65536
        for level, start in enumerate(OFFSETS):
            assert level_offset(slots[5])(computer, 0, level) == start
            last = (256 >> level) - 1
            bit = ctypes.c_uint(99)
            actual = element(slots[6])(computer, 0, level, last, last, 0, 0, ctypes.byref(bit))
            assert actual == address(level, last, last) and bit.value == 0
        tiles, count = ctypes.c_uint(), ctypes.c_uint(6)
        packed = ctypes.create_string_buffer(12)
        shape = (ctypes.c_uint * 3)()
        entries = ctypes.create_string_buffer(b"\xfe" * 72, 72)
        assert tiling(slots[10])(computer, ctypes.byref(tiles), packed, shape, ctypes.byref(count), 0, entries) == 0
        assert tiles.value == 22 and count.value == 3
        assert struct.unpack("<BB2xII", packed) == (3, 3, 1, 0)
        assert list(shape) == [256, 256, 1]
        assert list(struct.iter_unpack("<IHHI", entries.raw[:36])) == [(4, 4, 1, 6), (2, 2, 1, 2), (1, 1, 1, 1)]
        assert entries.raw[36:] == b"\xfe" * 36
        guest = ctypes.create_string_buffer(b"\xa5" * TOTAL, TOTAL)
        expected_guest = bytearray(guest.raw)
        copy_from = COPY_FROM(slots[9])
        for level in range(6):
            side = 256 >> level
            row_pitch = side * 16 + 16
            linear = bytearray(b"\xe7" * (row_pitch * side))
            for y in range(side):
                for x in range(side):
                    encoded = block(fmt, color(fmt, level, x, y))
                    offset = address(level, x, y)
                    expected_guest[offset:offset + 16] = encoded
                    pos = y * row_pitch + x * 16
                    linear[pos:pos + 16] = encoded
            source = ctypes.create_string_buffer(bytes(linear), len(linear))
            assert COPY_INTO(slots[8])(computer, guest, 0, level, source, row_pitch, len(linear)) == 0
            result = ctypes.create_string_buffer(b"\xe7" * len(linear), len(linear))
            assert copy_from(computer, result, 0, level, 0, guest, row_pitch, 0) == 0
            assert result.raw == bytes(linear), f"mip {level} copy-out or row padding mismatch"
        refused_mip = 5
        blocks = 256 >> refused_mip
        pitch = blocks * 16
        sentinel = b"\xad" * (pitch * blocks)
        refused = ctypes.create_string_buffer(sentinel, len(sentinel))
        assert copy_from(computer, refused, 0, refused_mip, 1, guest, pitch, 0) & 0xffffffff == 0x80004001
        assert refused.raw == sentinel, "unsupported copy-out option changed destination"
        assert guest.raw == bytes(expected_guest), "tiled blocks or padding differ"
        error = COPY_INTO(slots[8])(computer, guest, 0, 6, source, row_pitch, len(linear))
        assert error & 0xffffffff == 0x80070057 and guest.raw == bytes(expected_guest)
        path.write_bytes(guest.raw)
    finally:
        RELEASE(slots[1])(computer)


def run(dll, consumer, backend, directory):
    os.environ["X2W_D3D12_PERMISSIVE"] = "1"
    create = ctypes.WinDLL(dll).XGCreateTextureComputer
    create.argtypes = (PTR, ctypes.POINTER(PTR), ctypes.c_uint64, ctypes.c_uint64)
    create.restype = HRESULT
    shader = """Texture2D<float4> t : register(t0);
float4 main(float4 p : SV_Position) : SV_Target {
    uint mip = (uint(p.x) / 32) % 6;
    uint2 xy = uint2(p.xy) % 32;
    return t.Load(int3(xy * (32 >> mip), p.x < 192 ? mip : 0));
}
"""
    for fmt in (83, 98):
        case = directory / str(fmt)
        case.mkdir()
        path = case / "chain.bin"
        prepare(create, fmt, path)
        options = {"X2W_TEST_BC_CHAIN": str(path)}
        if fmt == 83:
            options["X2W_TEST_BC5"] = "1"
        pixels = draw_texture(case, consumer, backend, shader, options)
        if pixels is None:
            return 77
        expected = bytearray()
        wrong_level = bytearray()
        for y in range(32):
            for x in range(192):
                mip = x // 32
                bx, by = ((x % 32) * (32 >> mip)) // 4, (y * (32 >> mip)) // 4
                expected.extend(color(fmt, mip, bx, by))
                wrong_level.extend(color(fmt, 0, bx, by))
        assert len(pixels) == 384 * 32 * 4
        for y in range(32):
            row = slice(y * 192 * 4, (y + 1) * 192 * 4)
            ref = expected[row] + wrong_level[row]
            actual = pixels[y * 384 * 4:(y + 1) * 384 * 4]
            if actual != ref:
                i = next(i for i in range(0, len(ref), 4) if actual[i:i+4] != ref[i:i+4])
                raise AssertionError(f"{backend} format {fmt}, pixel {i//4},{y}: "
                                     f"{actual[i:i+4].hex()} != {ref[i:i+4].hex()}")
        wrong = sum(expected[i:i+4] != wrong_level[i:i+4] for i in range(0, len(expected), 4))
        assert wrong > 0, "forced-mip-zero draw did not differ"
        print(f"{backend} format {fmt}: 87,360 block addresses and 12,288 GPU pixels exact; "
              f"forced mip zero changes {wrong}/6144 pixels")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        sys.exit(run(*sys.argv[1:], Path(directory)))
