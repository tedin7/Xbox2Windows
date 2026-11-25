"""Exercise the production shim using Windows placeholders and shared page views."""
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys

PAGE_BYTES = 65536
PHYSICAL_64K = 0x20400000
WIN_RESERVE = 0x2000
RESERVE_PLACEHOLDER = 0x40000
PRESERVE_PLACEHOLDER = 0x2
WIN_RELEASE = 0x8000
NOACCESS, READWRITE = 1, 4


def bind(library, name, args, result):
    function = getattr(library, name)
    function.argtypes, function.restype = args, result
    return function


def permissive_count(path, count):
    """Child process: one permissive XMemMapPhysicalPages call with a huge count, with array and without."""
    library = C.WinDLL(str(Path(path).resolve()), use_last_error=True, winmode=0x900)
    function = library.XMemMapPhysicalPages
    function.argtypes, function.restype = [C.c_void_p, C.c_size_t, C.c_void_p], C.c_void_p
    scatter = library.XMemMapPhysicalPagesScatter
    scatter.argtypes, scatter.restype = [C.c_void_p, C.c_size_t, C.c_void_p], C.c_int
    tokens = (C.c_size_t * 1)(0)
    print("control", function(0x10000, 0, None), flush=True)  # a null array with no pages is a plain success
    for args in ((None,), (tokens,)):
        C.set_last_error(0)
        result = function(0x10000, count, *args)
        print("map", result, C.get_last_error(), flush=True)
    C.set_last_error(0)
    result = scatter((C.c_void_p * 1)(0x10000), count, tokens)
    print("scatter", result, C.get_last_error(), flush=True)


def check_permissive_count(path):
    """A huge count must fail with ERROR_INVALID_PARAMETER, not abort the process on an allocation."""
    environment = dict(os.environ, X2W_XMEM_PERMISSIVE="1")
    done = subprocess.run([sys.executable, "-B", __file__, path, "--permissive-count", str(1 << 40)],
                          env=environment, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, f"huge permissive count ended the process: {done.returncode:#x} {done.stderr}"
    assert done.stdout.split() == ["control", "65536", "map", "None", "87", "map", "None", "87", "scatter", "0", "87"], done.stdout


def scatter_failure(path):
    """Child process: an unsorted scatter batch whose second address in mapping order fails."""
    library = C.WinDLL(str(Path(path).resolve()), use_last_error=True, winmode=0x900)
    kernel = C.WinDLL("kernelbase.dll", use_last_error=True)
    pointer, number, dword = C.c_void_p, C.c_size_t, C.c_ulong
    allocate = bind(library, "XMemAllocatePhysicalPages", [number, C.POINTER(number), C.POINTER(number)], C.c_int)
    scatter = bind(library, "XMemMapPhysicalPagesScatter", [C.POINTER(pointer), number, C.POINTER(number)], C.c_int)
    reserve = bind(kernel, "VirtualAlloc2", [pointer, pointer, number, dword, dword, pointer, dword], pointer)
    free = bind(kernel, "VirtualFree", [pointer, number, dword], C.c_int)
    count, tokens = number(2), (number * 2)()
    assert allocate(PHYSICAL_64K, C.byref(count), tokens) and count.value == 2
    base = reserve(pointer(-1), None, 2 * PAGE_BYTES, WIN_RESERVE | RESERVE_PLACEHOLDER, NOACCESS, None, 0)
    assert base, C.get_last_error()
    bad = base + PAGE_BYTES
    assert free(base, PAGE_BYTES, WIN_RELEASE | PRESERVE_PLACEHOLDER), C.get_last_error()
    assert free(bad, 0, WIN_RELEASE), C.get_last_error()
    assert reserve(pointer(-1), bad, PAGE_BYTES, WIN_RESERVE, READWRITE, None, 0) == bad, C.get_last_error()
    destinations = (pointer * 2)(bad + 1, base)
    assert not scatter(destinations, 2, tokens) and C.get_last_error() == 87
    destinations[0] = bad
    assert not scatter(destinations, 2, tokens)
    error = C.get_last_error()
    assert error
    assert scatter((pointer * 1)(base), 1, tokens), "first view was not rolled back"
    print(f"{bad:016X} {tokens[0] >> 4:x} {error}")


def check_scatter_diagnostic(path):
    environment = dict(os.environ, X2W_XMEM_PERMISSIVE="1", X2W_XMEM_SHARED_VIEWS="1",
                       X2W_XMEM_TRACE_FAILED_BATCH="1")
    done = subprocess.run([sys.executable, "-B", __file__, path, "--scatter-failure"],
                          env=environment, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    address, page, error = done.stdout.split()
    unaligned = f"{int(address, 16) + 1:016X}"
    base = f"{int(address, 16) - PAGE_BYTES:016X}"
    assert f"failed stage=invalid-address index=0 address={unaligned} page=0x{page} error=87" in done.stderr, done.stderr
    assert f"failed stage=map-view index=0 address={address} page=0x{page} error={error}" in done.stderr, done.stderr
    assert f"failed batch map first={base} last={address} pages=2" in done.stderr, done.stderr
    print("Permissive scatter diagnostics identify the caller index and page after address sorting and rollback")


def main(path):
    check_permissive_count(path)
    check_scatter_diagnostic(path)
    library = C.WinDLL(str(Path(path).resolve()), use_last_error=True, winmode=0x900)
    kernel = C.WinDLL("kernelbase.dll", use_last_error=True)
    pointer, number = C.c_void_p, C.c_size_t
    allocate = bind(library, "XMemAllocatePhysicalPages", [number, C.POINTER(number), C.POINTER(number)], C.c_int)
    release = bind(library, "XMemFreePhysicalPages", [number, C.POINTER(number)], C.c_int)
    contiguous = bind(library, "XMemMapPhysicalPages", [pointer, number, C.POINTER(number)], pointer)
    scatter = bind(library, "XMemMapPhysicalPagesScatter", [C.POINTER(pointer), number, C.POINTER(number)], C.c_int)
    dword = C.c_ulong
    reserve_startup = bind(library, "XMemVirtualAlloc", [pointer, number, dword, number, dword], pointer)
    reserve = bind(kernel, "VirtualAlloc2", [pointer, pointer, number, dword, dword, pointer, dword], pointer)
    free = bind(kernel, "VirtualFree", [pointer, number, dword], C.c_int)
    reserved = []

    def placeholder(count):
        base = reserve(pointer(-1), None, count * PAGE_BYTES, WIN_RESERVE | RESERVE_PLACEHOLDER,
                       NOACCESS, None, 0)
        assert base, C.get_last_error()
        reserved.append(base)
        for i in range(count - 1):
            assert free(base + i * PAGE_BYTES, PAGE_BYTES, WIN_RELEASE | PRESERVE_PLACEHOLDER), C.get_last_error()
            reserved.append(base + (i + 1) * PAGE_BYTES)
        return base

    def value(address):
        return C.c_uint32.from_address(address)

    try:
        count, pages = number(2), (number * 2)()
        assert allocate(PHYSICAL_64K, C.byref(count), pages) and count.value == 2
        base, first, second = placeholder(2), placeholder(1), placeholder(1)
        destinations = (pointer * 2)(second, first)
        assert contiguous(base, 2, pages) == base
        value(base).value, value(base + PAGE_BYTES).value = 0x123, 0x456
        assert scatter(destinations, 2, pages)
        assert (value(second).value, value(first).value) == (0x123, 0x456)
        assert not release(2, pages), "released a mapped grant"
        invalid = (number * 2)(pages[0], 0)
        assert not scatter(destinations, 2, invalid)
        assert (value(second).value, value(first).value) == (0x123, 0x456)

        trial = placeholder(2)
        assert contiguous(trial, 2, invalid) is None
        unbackable = trial + PAGE_BYTES
        assert free(unbackable, 0, WIN_RELEASE)
        reserved.remove(unbackable)
        assert reserve(pointer(-1), unbackable, PAGE_BYTES, WIN_RESERVE, READWRITE, None, 0) == unbackable
        reserved.append(unbackable)
        assert contiguous(trial, 2, pages) is None
        assert contiguous(trial, 1, pages) == trial, "first view was not rolled back"
        assert value(trial).value == 0x123
        assert contiguous(trial, 1, None) == trial

        assert scatter(destinations, 2, None) and contiguous(base, 2, None) == base
        assert contiguous(base, 2, pages) == base
        assert (value(base).value, value(base + PAGE_BYTES).value) == (0x123, 0x456)
        assert contiguous(base, 2, None) == base
        assert not release(2, invalid)
        assert release(2, pages)
        assert contiguous(base, 2, pages) is None, "released IDs were still accepted"
        for size, attributes, protection in ((16 << 30, 0x21, 4), (24 << 30, 3, 0x801)):
            whole = reserve_startup(None, size, 0x20402000, attributes, protection)
            assert whole
            count, token = number(1), number()
            assert allocate(PHYSICAL_64K, C.byref(count), C.byref(token))
            interior = whole + PAGE_BYTES
            assert contiguous(interior, 1, C.byref(token)) == interior, C.get_last_error()
            value(interior).value = 0x8142
            assert contiguous(interior, 1, None) == interior
            assert contiguous(interior, 1, C.byref(token)) == interior
            assert value(interior).value == 0x8142
            assert contiguous(interior, 1, None) == interior
            assert release(1, C.byref(token))
            assert free(whole, 0, WIN_RELEASE), C.get_last_error()
            assert reserve(pointer(-1), whole, size, WIN_RESERVE, NOACCESS, None, 0) == whole
            assert free(whole, 0, WIN_RELEASE)
    finally:
        for address in reserved:
            free(address, 0, WIN_RELEASE)
    print("Aliased writes, scatter order, remap, rollback, and grant lifetime agree")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[2] == "--permissive-count":
        permissive_count(sys.argv[1], int(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[2] == "--scatter-failure":
        scatter_failure(sys.argv[1])
    else:
        main(sys.argv[1])
