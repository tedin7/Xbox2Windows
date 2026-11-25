"""Check the experimental xg_xs layout and copy slots with source-built BC7 blocks."""

import ctypes
import os
import struct
import sys
from unittest.mock import patch

from gfx10_texture_roundtrip_check import PADDING, oracle_surface


DESC = struct.Struct("<I4xQQIHHIIIII4x")
LAYOUT_BYTES = 0x18 + 4 * 0x6C8 + 8
GUARD = b"\xa5" * 16
PTR = ctypes.c_void_p
UINT = ctypes.c_uint32
HRESULT = ctypes.c_int32
LAYOUT = ctypes.WINFUNCTYPE(HRESULT, PTR, PTR)
COPY_INTO = ctypes.WINFUNCTYPE(HRESULT, PTR, PTR, UINT, UINT, PTR, UINT, UINT)
COPY_FROM = ctypes.WINFUNCTYPE(HRESULT, PTR, PTR, UINT, UINT, UINT, PTR, UINT, UINT)
RELEASE = ctypes.WINFUNCTYPE(ctypes.c_uint64, PTR)


def expect(label, actual, expected):
    if actual != expected:
        raise RuntimeError(f"{label}: got {actual!r}, expected {expected!r}")


def expect_bytes(label, actual, expected):
    expect(f"{label} length", len(actual), len(expected))
    if actual != expected:
        offset = next(i for i, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b)
        raise RuntimeError(f"{label}: first difference at byte 0x{offset:x}, "
                           f"got {actual[offset]:02x}, expected {expected[offset]:02x}")


def check_copy(computer, copy, width, height, tiled, linear):
    rows = (height + 3) // 4
    row_bytes = len(linear) // rows
    row_pitch = row_bytes + 16
    padding = b"\x5a" * 16
    source_bytes = padding.join(linear[i:i + row_bytes] for i in range(0, len(linear), row_bytes))
    source = ctypes.create_string_buffer(source_bytes, len(source_bytes))
    target_bytes = GUARD + b"\xa5" * len(tiled) + GUARD
    target = ctypes.create_string_buffer(target_bytes, len(target_bytes))
    dest = ctypes.byref(target, len(GUARD))
    label = f"BC7 {width}x{height}"

    # Layout padding belongs to the caller. Seed it differently from the
    # oracle's filler so a copy that writes the padding cannot pass.
    expected = b"".join(b"\xa5" * 16 if tiled[i:i + 16] == PADDING else tiled[i:i + 16]
                        for i in range(0, len(tiled), 16))
    slice_pitch = len(source_bytes) if width == 81 else 0
    expect(f"{label} CopyIntoSubresource HRESULT",
           copy(computer, dest, 0, 0, source, row_pitch, slice_pitch), 0)
    expect_bytes(f"{label} tiled blocks, padding and guards", target.raw, GUARD + expected + GUARD)
    expect_bytes(f"{label} source preservation", source.raw, source_bytes)

    for case, plane, subresource, src, pitch, slice_bytes in (
        ("wrong plane", 1, 0, source, row_pitch, slice_pitch),
        ("wrong subresource", 0, 1, source, row_pitch, slice_pitch),
        ("short row", 0, 0, source, row_bytes - 1, slice_pitch),
        ("short slice", 0, 0, source, row_pitch, len(source_bytes) - 1),
        ("aliased source", 0, 0, dest, row_pitch, slice_pitch),
    ):
        result = copy(computer, dest, plane, subresource, src, pitch, slice_bytes)
        expect(f"{label} {case} HRESULT", result & 0xFFFFFFFF, 0x80070057)
        expect_bytes(f"{label} {case} preserves destination", target.raw, GUARD + expected + GUARD)

    # A source that cannot be read is refused before the copy instead of faulting inside it.
    result = copy(computer, dest, 0, 0, ctypes.cast(1, PTR), row_pitch, slice_pitch)
    expect(f"{label} unreadable source HRESULT", result & 0xFFFFFFFF, 0x80004005)
    expect_bytes(f"{label} unreadable source preserves destination", target.raw, GUARD + expected + GUARD)

    os.environ["X2W_D3D12_RENDER"] = "1"
    try:
        result = copy(computer, dest, 0, 0, source, row_pitch, slice_pitch)
        expect(f"{label} render mode HRESULT", result & 0xFFFFFFFF, 0x80004001)
        expect_bytes(f"{label} render mode preserves destination", target.raw, GUARD + expected + GUARD)
    finally:
        os.environ["X2W_D3D12_RENDER"] = "0"


def check_copy_from(computer, copy, width, height, tiled, linear):
    rows = (height + 3) // 4
    row_bytes = len(linear) // rows
    label = f"BC7 {width}x{height} copy-out"
    source = ctypes.create_string_buffer(tiled, len(tiled))
    for padding in (b"", b"\xcd" * 16):
        pitch = row_bytes + len(padding)
        expected = b"".join(linear[i:i + row_bytes] + padding for i in range(0, len(linear), row_bytes))
        blank = GUARD + b"\xcd" * len(expected) + GUARD
        target = ctypes.create_string_buffer(blank, len(blank))
        dest = ctypes.byref(target, len(GUARD))
        span = (rows - 1) * pitch + row_bytes
        expect(f"{label} HRESULT", copy(computer, dest, 0, 0, 0, source, pitch, span if padding else 0), 0)
        expect_bytes(f"{label} rows and guards", target.raw, GUARD + expected + GUARD)
        expect_bytes(f"{label} source preservation", source.raw, tiled)

        target.raw = blank
        for case, plane, subresource, option, row_pitch, slice_pitch, hr in (
            ("plane", 1, 0, 0, pitch, span, 0x80070057),
            ("subresource", 0, 1, 0, pitch, span, 0x80070057),
            ("Scarlett option", 0, 0, 1, pitch, span, 0x80004001),
            ("short row", 0, 0, 0, row_bytes - 1, span, 0x80070057),
            ("short slice", 0, 0, 0, pitch, span - 1, 0x80070057),
        ):
            result = copy(computer, dest, plane, subresource, option, source, row_pitch, slice_pitch)
            expect(f"{label} {case} HRESULT", result & 0xFFFFFFFF, hr)
            expect_bytes(f"{label} {case} preserves destination", target.raw, blank)

    # A partial surface has different tiled and linear lengths. Adjacent regions are valid in either order.
    empty = b"\xcd" * len(linear)
    for initial, expected, src_offset, dst_offset in (
        (tiled + empty, tiled + linear, 0, len(tiled)),
        (empty + tiled, linear + tiled, len(linear), 0),
    ):
        slab = ctypes.create_string_buffer(initial, len(initial))
        result = copy(computer, ctypes.byref(slab, dst_offset), 0, 0, 0, ctypes.byref(slab, src_offset), row_bytes, 0)
        expect(f"{label} adjacent buffers HRESULT", result, 0)
        expect_bytes(f"{label} adjacent buffers", slab.raw, expected)


def check_layout(create, width, height, padded_width, padded_height):
    # 2D, default alignment, one slice, one mip, BC7_UNORM, one sample,
    # UNKNOWN layout, and no resource flags.
    desc = ctypes.create_string_buffer(DESC.pack(3, 0, width, height, 1, 1, 98, 1, 0, 0, 0))
    computer = PTR()
    expect("XGCreateTextureComputer HRESULT", create(desc, ctypes.byref(computer), 0, 0), 0)
    if not computer:
        raise RuntimeError("XGCreateTextureComputer returned a null object")
    vtable = ctypes.cast(computer, ctypes.POINTER(ctypes.POINTER(PTR)))[0]
    try:
        # A null destination is refused with E_POINTER before anything is written.
        expect("GetResourceLayout(null) HRESULT", LAYOUT(vtable[2])(computer, None) & 0xFFFFFFFF, 0x80004003)
        storage = ctypes.create_string_buffer(GUARD + b"\xa5" * LAYOUT_BYTES + GUARD)
        expect("GetResourceLayout HRESULT", LAYOUT(vtable[2])(computer, ctypes.byref(storage, len(GUARD))), 0)
        expect("prefix guard", bytes(storage[:len(GUARD)]), GUARD)
        end = len(GUARD) + LAYOUT_BYTES
        expect("suffix guard", bytes(storage[end:end + len(GUARD)]), GUARD)
        layout = storage[len(GUARD):end]

        tiled, linear = oracle_surface(0, width, height)
        span = len(tiled)
        pitch = padded_width // 4 * 16
        label = f"BC7 {width}x{height}"
        expect(f"{label} mip size/offset/slice span", struct.unpack_from("<QQQ", layout, 0x18 + 0x38),
               (span, 0, span))
        expect(f"{label} size/alignment/mips/planes", struct.unpack_from("<QQII", layout),
               (span, 4096, 1, 1))
        expect(f"{label} plane size/slice stride/alignment", struct.unpack_from("<QQQ", layout, 0x18 + 8),
               (span, 0, 4096))
        expect(f"{label} pitch/alignment/padded and visible extent",
               struct.unpack_from("<9I", layout, 0x18 + 0x38 + 0x18),
               (padded_width, pitch, 4096, padded_width, padded_height, 1, width, height, 1))

        check_copy(computer, COPY_INTO(vtable[8]), width, height, tiled, linear)
        # Readback needs only the two CPU buffers, even with rendering enabled.
        with patch.dict(os.environ, X2W_D3D12_RENDER="1"):
            check_copy_from(computer, COPY_FROM(vtable[9]), width, height, tiled, linear)
        print(f"{label}: span={span} offset=0 padded_blocks={padded_width // 4}x{padded_height // 4} "
              f"pitch={pitch} copy-in and copy-out exact, refusal controls passed")
    finally:
        RELEASE(vtable[1])(computer)


def main():
    os.environ["X2W_D3D12_RENDER"] = "0"
    try:
        module = ctypes.WinDLL(sys.argv[1])
    except OSError as error:
        # A test directory configured before the shim was built has no xg_xs.dll, and ctypes then
        # reports only the loader error. Name the target that produces it.
        raise SystemExit("cannot load %s (%s); build the startup_stub_xg_xs target first"
                         % (sys.argv[1], error))
    create = module.XGCreateTextureComputer
    create.argtypes = (PTR, ctypes.POINTER(PTR), ctypes.c_uint64, ctypes.c_uint64)
    create.restype = HRESULT
    check_layout(create, 81, 69, 128, 128)
    check_layout(create, 64, 64, 64, 64)
    # A shape whose padded extent is not square: both padded fields are asserted against the
    # expected pair, so a swapped or duplicated width/height is caught only here (the two shapes
    # above pad to a square, where the two fields hold the same number).
    check_layout(create, 200, 72, 256, 128)


if __name__ == "__main__":
    main()
