"""Check X2W_XMEM_LAZY_COMMIT=1: mapped physical pages are committed on first touch (#1199).

- Mapping commits almost nothing: the mapped pages stay reserved until they are touched.
- A CPU write commits only the page it touches, and the bytes read back.
- A kernel-mode write (ReadFile) into an untouched page fails without a fault; after x2w_xmem_commit_range
  the same read succeeds. This is why writers that cannot fault commit first.
- Threads that touch the same untouched pages at once all succeed.
- Released pages come back zero-filled when their slots are reused.
With `off` (control, switch set to 0): mapping commits every page up front as before, and
x2w_xmem_commit_range returns TRUE without doing anything.
usage: xmem_lazy_commit_check.py <xmem_shim.dll> [off]
"""
import ctypes as C
from ctypes import wintypes as W
import os
import sys
import tempfile
import threading

from xmem_page_runs_check import MEM_COMMIT, MEM_RESERVE, PAGE, Shim, bind

PAGES = 1024  # 64 MiB of mapped pages
# Other processes move system commit a little while this runs; committing every mapped page would be 64 MiB.
LIMIT_BYTES = 24 << 20
ERROR_NOACCESS = 998
GENERIC_READ, OPEN_EXISTING = 0x80000000, 3
THREADS = 8


class PerformanceInformation(C.Structure):
    _fields_ = [("cb", W.DWORD)] + [(name, C.c_size_t) for name in (
        "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache",
        "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
        (name, W.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


def main(path, mode=""):
    kernel32 = C.WinDLL("kernel32.dll", use_last_error=True)
    lazy = mode != "off"
    # The shim reads the switch when it loads.
    assert bind(kernel32, "SetEnvironmentVariableW", [W.LPCWSTR, W.LPCWSTR], W.BOOL)(
        "X2W_XMEM_LAZY_COMMIT", "1" if lazy else "0")
    shim = Shim(path)
    commit_range = bind(shim.library, "x2w_xmem_commit_range", [C.c_void_p, C.c_size_t], W.BOOL)
    performance = bind(kernel32, "K32GetPerformanceInfo", [C.POINTER(PerformanceInformation), W.DWORD], W.BOOL)
    create_file = bind(kernel32, "CreateFileW", [W.LPCWSTR, W.DWORD, W.DWORD, C.c_void_p, W.DWORD, W.DWORD, W.HANDLE],
                       W.HANDLE)
    read_file = bind(kernel32, "ReadFile", [W.HANDLE, C.c_void_p, W.DWORD, C.POINTER(W.DWORD), C.c_void_p], W.BOOL)
    close_handle = bind(kernel32, "CloseHandle", [W.HANDLE], W.BOOL)

    def commit_bytes():
        info = PerformanceInformation()
        info.cb = C.sizeof(info)
        assert performance(C.byref(info), info.cb), C.get_last_error()
        return info.CommitTotal * info.PageSize

    def state(address):
        return shim.query(address)[3]

    base = shim.whole
    pages = shim.pages(PAGES)
    try:
        # Mapping: the pages are reserved, not committed.
        before = commit_bytes()
        assert shim.contiguous(base, PAGES, pages) == base, C.get_last_error()
        grown = commit_bytes() - before
        print(f"mapping {PAGES} pages ({PAGES * PAGE >> 20} MiB) grew commit by {grown >> 20} MiB")
        if not lazy:
            # Without the switch every mapped page is committed when it is mapped, and the export does nothing.
            assert state(base) == MEM_COMMIT and state(base + (PAGES - 1) * PAGE) == MEM_COMMIT, "off: a page was left lazy"
            assert grown >= PAGES * PAGE - LIMIT_BYTES, f"off: mapping committed only {grown >> 20} MiB"
            assert commit_range(C.c_void_p(base), PAGES * PAGE)
            assert shim.contiguous(base, PAGES, None) == base
            print("ok (off)")
            return
        assert grown < LIMIT_BYTES, f"mapping committed {grown >> 20} MiB: the pages were not left lazy"
        assert state(base) == MEM_RESERVE and state(base + (PAGES - 1) * PAGE) == MEM_RESERVE

        # First CPU write: only that page is committed; the bytes read back.
        C.c_uint32.from_address(base + 8).value = 0x12345678
        assert C.c_uint32.from_address(base + 8).value == 0x12345678
        assert state(base) == MEM_COMMIT and state(base + PAGE) == MEM_RESERVE, "a write committed more than its page"
        assert C.c_uint32.from_address(base + PAGE - 4).value == 0, "a committed page is not zero-filled"

        # Kernel-mode write: no fault, so it fails on an untouched page until the range is committed.
        data = bytes(range(256)) * 16
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(data)
        try:
            target = base + 2 * PAGE
            for committed in (False, True):
                if committed:
                    assert commit_range(C.c_void_p(target), len(data)), C.get_last_error()
                    assert state(target) == MEM_COMMIT
                handle = create_file(file.name, GENERIC_READ, 0, None, OPEN_EXISTING, 0, None)
                assert handle and handle != W.HANDLE(-1).value, C.get_last_error()
                read = W.DWORD()
                ok = read_file(handle, C.c_void_p(target), len(data), C.byref(read), None)
                error = C.get_last_error()
                close_handle(handle)
                if committed:
                    assert ok and read.value == len(data) and C.string_at(target, len(data)) == data, error
                else:
                    assert not ok and error == ERROR_NOACCESS, f"expected ERROR_NOACCESS, got ok={ok} error={error}"
        finally:
            os.unlink(file.name)
        assert commit_range(C.c_void_p(base), PAGES * PAGE * 2), "a range reaching past the mapping must not fail"

        # Threads touching the same untouched pages at once: each write lands, none faults through.
        first = 16
        barrier = threading.Barrier(THREADS)
        errors = []

        def touch(index):
            try:
                barrier.wait()
                for page in range(first, first + 32):
                    value = C.c_uint32(page << 8 | index)
                    C.memmove(base + page * PAGE + index * 4, C.byref(value), 4)
            except Exception as error:  # noqa: BLE001 - reported below
                errors.append(error)

        threads = [threading.Thread(target=touch, args=(i,)) for i in range(THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, errors
        for page in range(first, first + 32):
            for index in range(THREADS):
                assert C.c_uint32.from_address(base + page * PAGE + index * 4).value == page << 8 | index

        # Released pages: their slots are reused zero-filled.
        assert shim.contiguous(base, PAGES, None) == base
        assert shim.release(PAGES, pages)
        pages = shim.pages(PAGES)
        assert shim.contiguous(base, PAGES, pages) == base, C.get_last_error()
        for page in (0, 2, first, first + 31):
            assert not any(C.string_at(base + page * PAGE, 64)), f"reused page {page} is not zero-filled"
        assert shim.contiguous(base, PAGES, None) == base
    finally:
        shim.release(PAGES, pages)
        shim.close()
    print("ok")


if __name__ == "__main__":
    main(*sys.argv[1:3])
