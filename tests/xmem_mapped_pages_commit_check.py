"""Check that opening a physical-page slab does not charge its whole size to system commit (#1199).

Mapping a few pages must commit about those pages, not the 256 MiB slab that holds them.
usage: xmem_mapped_pages_commit_check.py <xmem_shim.dll> [failure]
"""
import ctypes as C
from ctypes import wintypes as W
import sys

from xmem_page_runs_check import PAGE as PAGE_BYTES, Shim, bind

WIN_RESERVE = 0x2000
RESERVE_PLACEHOLDER = 0x40000
WIN_RELEASE = 0x8000
NOACCESS = 1
ERROR_COMMITMENT_LIMIT = 1455
MAPPED_PAGES = 16
SLAB_BYTES = 256 << 20
# Other processes move system commit a little while this runs; a whole slab is 256 MiB.
LIMIT_BYTES = 32 << 20


class PerformanceInformation(C.Structure):
    _fields_ = [("cb", W.DWORD)] + [(name, C.c_size_t) for name in (
        "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache",
        "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
        (name, W.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


def main(path, fail_commit=False):
    shim = Shim(path)
    kernel = C.WinDLL("kernelbase.dll", use_last_error=True)
    psapi = C.WinDLL("kernel32.dll", use_last_error=True)
    pointer, number = C.c_void_p, C.c_size_t
    performance = bind(psapi, "K32GetPerformanceInfo", [C.POINTER(PerformanceInformation), W.DWORD], C.c_int)
    reserve = bind(kernel, "VirtualAlloc2", [pointer, pointer, number, C.c_ulong, C.c_ulong, pointer, C.c_ulong],
                   pointer)

    def commit_bytes():
        info = PerformanceInformation()
        info.cb = C.sizeof(info)
        assert performance(C.byref(info), info.cb), C.get_last_error()
        return info.CommitTotal * info.PageSize

    size = MAPPED_PAGES * PAGE_BYTES
    pages = shim.pages(MAPPED_PAGES)
    base = shim.whole
    try:
        if fail_commit:
            stats = bind(shim.library, "x2w_xmem_slab_stats", [C.POINTER(C.c_ulonglong)] * 3, C.c_int)
            counts = (C.c_ulonglong(), C.c_ulonglong(), C.c_ulonglong())
            before_shape = shim.query(base)
            assert stats(*(C.byref(count) for count in counts)) and all(count.value == 0 for count in counts)
            mapped = shim.contiguous(base, MAPPED_PAGES, pages)
            error = C.get_last_error()
            assert not mapped and error == ERROR_COMMITMENT_LIMIT, f"expected commitment limit, got {mapped!r} and {error}"
            assert shim.query(base) == before_shape, "a refused commit changed the reservation"
            assert stats(*(C.byref(count) for count in counts)) and all(count.value == 0 for count in counts)

        before = commit_bytes()
        mapped = shim.contiguous(base, MAPPED_PAGES, pages)
        assert mapped == base, C.get_last_error()
        grown = commit_bytes() - before
        print(f"commit grew by {grown >> 20} MiB for {MAPPED_PAGES} mapped pages (a slab is {SLAB_BYTES >> 20} MiB)")
        assert grown < LIMIT_BYTES, f"mapping {MAPPED_PAGES} pages committed {grown >> 20} MiB: the slab is charged whole"

        # The pages are usable and start zero-filled.
        view = (C.c_ubyte * size).from_address(mapped)
        assert not any(view[::4096]), "a fresh page is not zero-filled"
        view[0], view[size - 1] = 0x5A, 0xA5
        assert view[0] == 0x5A and view[size - 1] == 0xA5
        assert shim.contiguous(base, MAPPED_PAGES, None) == base

        # The caller's independently reserved two-page region cannot be split by a one-page mapping.
        external = reserve(pointer(-1), None, PAGE_BYTES * 2, WIN_RESERVE | RESERVE_PLACEHOLDER, NOACCESS, None, 0)
        assert external, C.get_last_error()
        try:
            before_shape = shim.query(external)
            mapped = shim.contiguous(external, 1, pages)
            if mapped:
                shim.contiguous(external, 1, None)
            assert not mapped and shim.query(external) == before_shape, "caller placeholder was split or mapped"
        finally:
            shim.free(external, 0, WIN_RELEASE)
    finally:
        shim.release(MAPPED_PAGES, pages)
        shim.close()
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] == ["failure"])
