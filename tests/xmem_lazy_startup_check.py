"""Check X2W_XMEM_LAZY_STARTUP=1 (with X2W_XMEM_LAZY_COMMIT=1): large startup blocks are committed on first
touch (#1199).

- A 64 MiB startup block (type 0x20403000) and a 32 MiB large-page block (type 0x20003000) are only reserved;
  the large-page one keeps its 2 MiB alignment. A 16 MiB block, below the threshold, is committed as before.
- A CPU write commits the 256 KiB chunk it touches, and the bytes read back; the rest of the block reads zero.
- A kernel-mode write (ReadFile) into an untouched chunk fails without a fault; after x2w_xmem_commit_range the
  same read succeeds.
- Threads that touch the same untouched chunks at once all succeed.
- x2w_xmem_commit_block commits the whole block; later writes anywhere in it succeed.
With `off` (control, X2W_XMEM_LAZY_STARTUP=0 and X2W_XMEM_LAZY_COMMIT=1): every block is committed when it is
allocated, as before, and x2w_xmem_commit_block returns TRUE.
usage: xmem_lazy_startup_check.py <xmem_shim.dll> [off]
"""
import ctypes as C
from ctypes import wintypes as W
import os
import sys
import tempfile
import threading

from xmem_page_runs_check import MEM_COMMIT, MEM_RESERVE, WIN_RELEASE, Shim, bind

CHUNK = 256 << 10
STARTUP_64K, LARGE = 0x20403000, 0x20003000
ERROR_NOACCESS = 998
GENERIC_READ, OPEN_EXISTING = 0x80000000, 3
THREADS = 8


def main(path, mode=""):
    kernel32 = C.WinDLL("kernel32.dll", use_last_error=True)
    set_env = bind(kernel32, "SetEnvironmentVariableW", [W.LPCWSTR, W.LPCWSTR], W.BOOL)
    lazy = mode != "off"
    # The shim reads the switches when it loads.
    assert set_env("X2W_XMEM_LAZY_COMMIT", "1") and set_env("X2W_XMEM_LAZY_STARTUP", "1" if lazy else "0")
    shim = Shim(path)
    commit_range = bind(shim.library, "x2w_xmem_commit_range", [C.c_void_p, C.c_size_t], W.BOOL)
    commit_block = bind(shim.library, "x2w_xmem_commit_block", [C.c_void_p], W.BOOL)
    create_file = bind(kernel32, "CreateFileW", [W.LPCWSTR, W.DWORD, W.DWORD, C.c_void_p, W.DWORD, W.DWORD, W.HANDLE],
                       W.HANDLE)
    read_file = bind(kernel32, "ReadFile", [W.HANDLE, C.c_void_p, W.DWORD, C.POINTER(W.DWORD), C.c_void_p], W.BOOL)
    close_handle = bind(kernel32, "CloseHandle", [W.HANDLE], W.BOOL)

    def state(address):
        return shim.query(address)[3]

    blocks = []
    try:
        # Reserved, not committed; below the threshold a block is committed as before.
        block = shim.reserve_startup(None, 64 << 20, STARTUP_64K, 0x181, 4)
        large = shim.reserve_startup(None, 32 << 20, LARGE, 0x183, 0x22004)
        small = shim.reserve_startup(None, 16 << 20, STARTUP_64K, 0x181, 4)
        blocks += [b for b in (block, large, small) if b]
        assert block and large and small, C.get_last_error()
        if not lazy:
            for base, size in ((block, 64 << 20), (large, 32 << 20)):
                assert state(base) == MEM_COMMIT and state(base + size - 1) == MEM_COMMIT, "off: a block was left lazy"
            assert large % (2 << 20) == 0 and commit_block(C.c_void_p(block))
            print("ok (off)")
            return
        assert state(block) == MEM_RESERVE and state(block + (64 << 20) - 1) == MEM_RESERVE, "a large block was committed"
        assert state(large) == MEM_RESERVE and large % (2 << 20) == 0, "the large-page block lost its 2 MiB alignment"
        assert state(small) == MEM_COMMIT and state(small + (16 << 20) - 1) == MEM_COMMIT, "a small block was left lazy"

        # First CPU write: only its chunk is committed; the bytes read back, the rest reads zero.
        target = block + CHUNK + 300
        C.c_uint32.from_address(target).value = 0x12345678
        assert C.c_uint32.from_address(target).value == 0x12345678
        assert state(block + CHUNK) == MEM_COMMIT and state(block + 2 * CHUNK - 1) == MEM_COMMIT
        assert state(block) == MEM_RESERVE and state(block + 2 * CHUNK) == MEM_RESERVE, "a write committed more than its chunk"
        assert C.c_uint32.from_address(block + 2 * CHUNK - 4).value == 0, "a committed chunk is not zero-filled"

        # Kernel-mode write: no fault, so it fails on an untouched chunk until the range is committed.
        data = bytes(range(256)) * 16
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(data)
        try:
            target = block + 4 * CHUNK - 1024  # crosses into the next chunk
            for committed in (False, True):
                if committed:
                    assert commit_range(C.c_void_p(target), len(data)), C.get_last_error()
                    assert state(target) == MEM_COMMIT and state(target + len(data) - 1) == MEM_COMMIT
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

        # Threads touching the same untouched chunks at once: each write lands, none faults through.
        first = 16
        barrier = threading.Barrier(THREADS)
        errors = []

        def touch(index):
            try:
                barrier.wait()
                for chunk in range(first, first + 32):
                    value = C.c_uint32(chunk << 8 | index)
                    C.memmove(block + chunk * CHUNK + index * 4, C.byref(value), 4)
            except Exception as error:  # noqa: BLE001 - reported below
                errors.append(error)

        threads = [threading.Thread(target=touch, args=(i,)) for i in range(THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, errors
        for chunk in range(first, first + 32):
            for index in range(THREADS):
                assert C.c_uint32.from_address(block + chunk * CHUNK + index * 4).value == chunk << 8 | index

        # The whole block, as before a GPU heap is opened on it; then writes anywhere succeed.
        assert commit_block(C.c_void_p(large + 12345)), C.get_last_error()
        assert state(large) == MEM_COMMIT and state(large + (32 << 20) - 1) == MEM_COMMIT, "commit_block left part lazy"
        C.c_uint32.from_address(large + (32 << 20) - 4).value = 7
        assert commit_block(C.c_void_p(small)), "an address outside the lazy blocks must not fail"
        print("ok")
    finally:
        for b in blocks:
            shim.free(b, 0, WIN_RELEASE)
        shim.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
