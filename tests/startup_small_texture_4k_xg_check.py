"""XG slots 2 and 8 for a small texture given the 4 KiB allocation answer (src/startup_stubs/small_texture_4k.h).

A 32x32 R8G8B8A8 texture with two mips, layout UNKNOWN, asked at Alignment 0x1000 gets 0x2000 bytes at 4 KiB
alignment. Its desktop linear footprint (256-byte rows) would not fit; the stand-in uses one compact plan
instead: mip 0 at offset 0 in 128-byte rows, mip 1 at 0x1000 in 64-byte rows.
Cases:
  limited      Slot 2 reports 0x2000 at 0x1000 and exactly those mip records. Slot 8 writes both mips (mip 0 from
               a source with padded 160-byte rows) into a
               0x2000-byte allocation followed by a no-access guard page. Every row is distinct (row index and
               mip in each texel); mip 0 and mip 1 carry different sentinels. Memory must equal the plan
               byte for byte, the bytes after the plan keep their fill, and the guard is not touched.
  unsupported  A row pitch shorter than a row, and a subresource past the last mip: slot 8 fails and leaves
               memory unchanged.
  unchanged    Alignment 0 (the allocation-info entries keep the desktop answer): slot 2 reports 0x10000 at
               0x10000 and slot 8 writes all 32 rows at the desktop 256-byte pitch, as before this change.

usage: startup_small_texture_4k_xg_check.py <xg_xs.dll> [--mutate no-limit]
--mutate no-limit runs the limited case at an explicit 64 KiB alignment, which gets no compact plan: slot 8 then
needs the desktop footprint in front of the guard page and fails (negative control).
"""
import ctypes as ct
import struct
import sys

from startup_xg_bc7_layout_check import COPY_INTO, DESC, HRESULT, LAYOUT, LAYOUT_BYTES, PTR, RELEASE

MEM_COMMIT, MEM_RESERVE, MEM_RELEASE = 0x1000, 0x2000, 0x8000
PAGE_NOACCESS, PAGE_READWRITE = 0x01, 0x04
FILL = 0xA5
MIP_RECORD = 0x18 + 0x38  # first mip record: after the layout header and the plane header
MIP_BYTES = 0x70

kernel = ct.WinDLL("kernel32", use_last_error=True)
kernel.VirtualAlloc.restype = PTR
kernel.VirtualAlloc.argtypes = (PTR, ct.c_size_t, ct.c_uint32, ct.c_uint32)
kernel.VirtualProtect.argtypes = (PTR, ct.c_size_t, ct.c_uint32, ct.POINTER(ct.c_uint32))
kernel.VirtualFree.argtypes = (PTR, ct.c_size_t, ct.c_uint32)


def texels(mip, width, height):
    """Distinct rows: each texel is (x, y, mip sentinel, 0x5c)."""
    sentinel = (0x11, 0xEE)[mip]
    return bytes(b for y in range(height) for x in range(width) for b in (x, y, sentinel, 0x5C))


class Computer:
    def __init__(self, create, alignment, mips):
        desc = ct.create_string_buffer(DESC.pack(3, alignment, 32, 32, 1, mips, 28, 1, 0, 0, 0))
        self.handle = PTR()
        assert create(desc, ct.byref(self.handle), 0, 0) == 0 and self.handle
        self.slots = ct.cast(self.handle, ct.POINTER(ct.POINTER(PTR)))[0]

    def layout(self):
        buffer = ct.create_string_buffer(LAYOUT_BYTES)
        assert LAYOUT(self.slots[2])(self.handle, buffer) == 0
        return buffer.raw

    def copy(self, base, subresource, data, pitch):
        source = ct.create_string_buffer(data, len(data))
        return COPY_INTO(self.slots[8])(self.handle, base, 0, subresource, source, pitch, 0) & 0xFFFFFFFF

    def release(self):
        RELEASE(self.slots[1])(self.handle)


def guarded(size):
    block = kernel.VirtualAlloc(None, size + 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
    assert block
    old = ct.c_uint32()
    assert kernel.VirtualProtect(block + size, 0x1000, PAGE_NOACCESS, ct.byref(old))
    ct.memset(block, FILL, size)
    return block


def limited(create, alignment):
    computer = Computer(create, alignment, 2)
    block = None
    try:
        raw = computer.layout()
        total, align = struct.unpack_from("<QQ", raw, 0)
        records = [struct.unpack_from("<QQ", raw, MIP_RECORD + m * MIP_BYTES) + struct.unpack_from("<I", raw, MIP_RECORD + m * MIP_BYTES + 0x1C)
                   for m in range(2)]
        print(f"limited: slot 2 total 0x{total:x} alignment 0x{align:x} mips {[tuple(hex(v) for v in r) for r in records]}")
        if alignment == 0x10000:
            assert (total, align) == (0x10000, 0x10000), (hex(total), hex(align))
            size = 0x2000
        else:
            assert (total, align) == (0x2000, 0x1000), (hex(total), hex(align))
            assert records == [(0x1000, 0, 128), (0x400, 0x1000, 64)], records
            size = total
        block = guarded(size)
        mip0, mip1 = texels(0, 32, 32), texels(1, 16, 16)
        # Mip 0 comes from a source with 160-byte rows (32 bytes of padding each): the plan's rows stay 128 bytes.
        padded = b"".join(mip0[y * 128:(y + 1) * 128] + b"\x77" * 32 for y in range(32))
        hr0 = computer.copy(block, 0, padded, 160)
        hr1 = computer.copy(block, 1, mip1, 64)
        print(f"limited: slot 8 mip 0 -> 0x{hr0:08x}, mip 1 -> 0x{hr1:08x} (0x{size:x} bytes before a guard page)")
        assert (hr0, hr1) == (0, 0), (hex(hr0), hex(hr1))
        memory = ct.string_at(block, size)
        expected = mip0 + mip1 + bytes([FILL]) * (size - len(mip0) - len(mip1))
        if memory != expected:
            first = next(i for i in range(size) if memory[i] != expected[i])
            raise AssertionError(f"limited: first difference at byte 0x{first:x}")
        rows = {memory[y * 128:(y + 1) * 128] for y in range(32)} | {memory[0x1000 + y * 64:0x1000 + (y + 1) * 64] for y in range(16)}
        assert len(rows) == 48, len(rows)
        print("limited: 32 + 16 distinct rows in both halves of mip 0 and in mip 1, plan bytes exact, rest untouched")

        before = ct.string_at(block, size)
        short = computer.copy(block, 0, mip0, 64)
        past = computer.copy(block, 2, mip1, 64)
        print(f"unsupported: short pitch -> 0x{short:08x}, subresource 2 -> 0x{past:08x}")
        assert short != 0 and past != 0 and ct.string_at(block, size) == before
    finally:
        if block:
            kernel.VirtualFree(block, 0, MEM_RELEASE)
        computer.release()


def unchanged(create):
    computer = Computer(create, 0, 1)
    block = None
    try:
        total, align = struct.unpack_from("<QQ", computer.layout(), 0)
        print(f"unchanged: slot 2 total 0x{total:x} alignment 0x{align:x}")
        assert (total, align) == (0x10000, 0x10000), (hex(total), hex(align))
        block = guarded(0x10000)
        mip0 = texels(0, 32, 32)
        hr = computer.copy(block, 0, mip0, 128)
        assert hr == 0, hex(hr)
        memory = ct.string_at(block, 0x10000)
        written = sum(memory[y * 256:y * 256 + 128] == mip0[y * 128:(y + 1) * 128] for y in range(32))
        print(f"unchanged: {written} of 32 rows written at the desktop pitch")
        assert written == 32, written
    finally:
        if block:
            kernel.VirtualFree(block, 0, MEM_RELEASE)
        computer.release()


def main(path, mutate):
    create = ct.WinDLL(path).XGCreateTextureComputer
    create.argtypes = (PTR, ct.POINTER(PTR), ct.c_uint64, ct.c_uint64)
    create.restype = HRESULT
    if mutate:
        limited(create, 0x10000)
        return
    limited(create, 0x1000)
    unchanged(create)
    print("startup_small_texture_4k_xg: PASS")


if __name__ == "__main__":
    args = sys.argv[1:]
    mutate = args[1:] == ["--mutate", "no-limit"]
    if not args or (len(args) > 1 and not mutate):
        sys.exit("usage: startup_small_texture_4k_xg_check.py <xg_xs.dll> [--mutate no-limit]")
    main(args[0], mutate)
