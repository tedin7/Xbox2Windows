"""Match nine-mip XG copies with byte offsets and sampled pixels in both array slices."""
import ctypes
import os
from pathlib import Path
import struct
import sys
import tempfile

from startup_address_srv_check import draw_texture
from startup_xg_bc7_layout_check import DESC, PTR, HRESULT, LAYOUT_BYTES, LAYOUT, COPY_INTO, COPY_FROM, RELEASE
from startup_xg_bc_chain_check import block, color

GUARD = b"\xd7" * 64
TAIL = (0x8000, 0x4000, 0x2000, 0x1000, 0x800, 0x600)


def rgba(fmt, level, array_slice, x, y):
    return color(fmt, level + 11 * array_slice, x, y)


def offset_in_level(level, x, y):
    inside = ((y & 3) << 4) | ((x & 3) << 6)
    for bit, xb, yb in zip((4, 8, 16, 32), (0x200, 0x800, 0x2000, 0x8000), (0x100, 0x400, 0x1000, 0x4000)):
        if x & bit:
            inside += xb
        if y & bit:
            inside += yb
    return ((y // 64) * max(1, 4 >> level) + x // 64) * 65536 + inside


def prepare(create, fmt, height, path):
    stride = 0xc0000 if height == 512 else 0x160000
    starts = (0x40000 if height == 512 else 0x60000, 0x20000, 0x10000, *TAIL)
    desc = ctypes.create_string_buffer(DESC.pack(3, 0, 1024, height, 2, 9, fmt, 1, 0, 0x109, 0))
    instance = PTR()
    assert create(desc, ctypes.byref(instance), 0, 0) == 0 and instance
    vt = ctypes.cast(instance, ctypes.POINTER(ctypes.POINTER(PTR)))[0]
    try:
        layout = ctypes.create_string_buffer(LAYOUT_BYTES)
        assert LAYOUT(vt[2])(instance, layout) == 0
        assert struct.unpack_from("<QQII", layout) == (stride * 2, 65536, 9, 1)
        assert struct.unpack_from("<QQ", layout, 0x20) == (stride * 2, stride)
        assert struct.unpack_from("<II", layout, 0x44) == (3, 6)
        expected = bytearray(b"\xe7" * (2 * stride))
        memory = ctypes.create_string_buffer(GUARD + bytes(expected) + GUARD)
        base = ctypes.byref(memory, len(GUARD))
        subresources = []
        for array_slice in range(2):
            for level, start in enumerate(starts):
                cols = max(1, 256 >> level)
                rows = max(1, (height // 4) >> level)
                off = 0x50 + level * 0x70
                assert struct.unpack_from("<Q", layout, off + 8)[0] == start
                assert struct.unpack_from("<5I", layout, off + 0x2c) == (2, cols, rows, 1024 >> level, height >> level)
                pitch = cols * 16 + 16
                linear = bytearray(b"\xa1" * (pitch * rows))
                for y in range(rows):
                    for x in range(cols):
                        encoded = block(fmt, rgba(fmt, level, array_slice, x, y))
                        loc = pitch * y + 16 * x
                        linear[loc:loc + 16] = encoded
                        address = stride * array_slice + start + offset_in_level(level, x, y)
                        expected[address:address + 16] = encoded
                sub = 9 * array_slice + level
                assert COPY_INTO(vt[8])(instance, base, 0, sub, bytes(linear), pitch, len(linear)) == 0
                subresources.append((sub, pitch, rows, linear))
        assert memory.raw[:-1] == GUARD + expected + GUARD, "tile address, tail padding or slice guard changed"
        for sub, pitch, rows, linear in subresources:
            dst = ctypes.create_string_buffer(b"\xa1" * (pitch * rows))
            assert COPY_FROM(vt[9])(instance, dst, 0, sub, 0, base, pitch, len(linear)) == 0
            assert dst.raw[:-1] == linear, f"copy-out mismatch at subresource {sub}"
        assert COPY_INTO(vt[8])(instance, base, 0, 18, b"\x00" * 16, 16, 16) & 0xffffffff == 0x80070057
        assert memory.raw[:-1] == GUARD + expected + GUARD
        path.write_bytes(memory.raw[len(GUARD):len(GUARD) + 2 * stride])
    finally:
        RELEASE(vt[1])(instance)


def draw(dll, consumer, backend):
    os.environ["X2W_D3D12_PERMISSIVE"] = "1"
    create = ctypes.WinDLL(dll).XGCreateTextureComputer
    create.argtypes = (PTR, ctypes.POINTER(PTR), ctypes.c_uint64, ctypes.c_uint64)
    create.restype = HRESULT
    with tempfile.TemporaryDirectory() as temp:
        for height in (512, 1024):
            for fmt in (83, 98):
                folder = Path(temp) / f"{fmt}-{height}"
                folder.mkdir()
                path = folder / "chain.bin"
                prepare(create, fmt, height, path)
                shader = f"""Texture2DArray<float4> t : register(t0);
float4 main(float4 p : SV_Position) : SV_Target {{
    uint x = uint(p.x), y = uint(p.y), mip = min(8u, (x % 192u) / 21u);
    uint2 xy = uint2((x*13u) % (1024u >> mip), (y*7u) % ({height}u >> mip));
    return t.Load(int4(xy, x/192u, mip));
}}"""
                options = {"X2W_TEST_BC_CHAIN": str(path), "X2W_TEST_BC_ARRAY": "1"}
                if height == 512:
                    options["X2W_TEST_BC_RECT"] = "1"
                if fmt == 83:
                    options["X2W_TEST_BC5"] = "1"
                pixels = draw_texture(folder, consumer, backend, shader, options, "tex2da")
                if pixels is None:
                    return 77
                expected, swapped = bytearray(), bytearray()
                for y in range(32):
                    for x in range(384):
                        level = min(8, (x % 192) // 21)
                        bx = ((x * 13) % (1024 >> level)) // 4
                        by = ((y * 7) % (height >> level)) // 4
                        expected.extend(rgba(fmt, level, x // 192, bx, by))
                        swapped.extend(rgba(fmt, level, 1 - x // 192, bx, by))
                assert len(pixels) == len(expected)
                if pixels != expected:
                    i = next(i for i in range(0, len(pixels), 4) if pixels[i:i+4] != expected[i:i+4])
                    raise AssertionError(f"{backend} {fmt=} {height=} pixel {i//4%384},{i//4//384}: "
                                         f"{pixels[i:i+4].hex()} expected {expected[i:i+4].hex()}")
                negative_dir = folder / "swapped"
                negative_dir.mkdir()
                other = draw_texture(negative_dir, consumer, backend, shader.replace("x/192u", "1u-x/192u"), options, "tex2da")
                assert other == swapped and other != pixels, "slice-swap negative control"
                print(f"{backend} {fmt=} {height=}: 18 subresources, guarded tile bytes and 12288 pixels exact")
    return 0


if __name__ == "__main__":
    sys.exit(draw(*sys.argv[1:]))
