"""Read the Windows system commit budget before starting a mapped PE package.

The system commit limit includes RAM and paging-file backing; it is NOT the
amount of free physical RAM. This is a snapshot, not a reservation.
"""

from collections.abc import Mapping
import ctypes as C
import sys

MIB = 1 << 20
_SNAPSHOT_KEYS = ("available_bytes", "limit_mib", "used_mib", "available_mib")


def _is_int(value):
    """Return True only for exact integers, excluding bool."""
    return isinstance(value, int) and not isinstance(value, bool)


class PerformanceInformation(C.Structure):
    """Windows psapi.h PERFORMANCE_INFORMATION; DWORD is always 32 bits."""

    _fields_ = [
        ("cb", C.c_uint32),
        ("CommitTotal", C.c_size_t),
        ("CommitLimit", C.c_size_t),
        ("CommitPeak", C.c_size_t),
        ("PhysicalTotal", C.c_size_t),
        ("PhysicalAvailable", C.c_size_t),
        ("SystemCache", C.c_size_t),
        ("KernelTotal", C.c_size_t),
        ("KernelPaged", C.c_size_t),
        ("KernelNonpaged", C.c_size_t),
        ("PageSize", C.c_size_t),
        ("HandleCount", C.c_uint32),
        ("ProcessCount", C.c_uint32),
        ("ThreadCount", C.c_uint32),
    ]


def snapshot_from_pages(limit_pages, used_pages, page_size):
    """Pure conversion of GetPerformanceInfo page counts; never wrap under pressure."""
    if not _is_int(page_size) or page_size <= 0 or not _is_int(limit_pages) or limit_pages <= 0:
        raise ValueError("Windows reported an invalid commit page size or limit")
    if not _is_int(used_pages) or used_pages < 0:
        raise ValueError("Windows reported an invalid commit used page count")
    limit_bytes = limit_pages * page_size
    used_bytes = used_pages * page_size
    free_bytes = max(limit_bytes - used_bytes, 0)
    return {"limit_mib": limit_bytes // MIB, "used_mib": used_bytes // MIB,
            "available_mib": free_bytes // MIB, "available_bytes": free_bytes}


def read_host_commit():
    """GetPerformanceInfo from System32 psapi.dll; never reads guest image bytes."""
    if sys.platform != "win32":
        raise OSError("Windows system commit information requires a Windows host")
    info = PerformanceInformation()
    info.cb = C.sizeof(PerformanceInformation)
    # Only the Windows system copy, never a DLL planted beside the package.
    psapi = C.WinDLL("psapi.dll", winmode=0x800, use_last_error=True)
    get_performance = psapi.GetPerformanceInfo
    get_performance.argtypes = [C.POINTER(PerformanceInformation), C.c_uint32]
    get_performance.restype = C.c_int
    if not get_performance(C.byref(info), info.cb):
        raise OSError(C.get_last_error(), "GetPerformanceInfo failed")
    return snapshot_from_pages(info.CommitLimit, info.CommitTotal, info.PageSize)


def require_commit_budget(minimum_mib, snapshot):
    """Reject a requested budget before any guest module is mapped or started."""
    if not _is_int(minimum_mib) or minimum_mib < 0 or minimum_mib > 1 << 20:
        raise ValueError("--min-commit-mib must be between 0 and 1048576")
    if not isinstance(snapshot, Mapping):
        raise ValueError("commit snapshot must be a Mapping")
    for key in _SNAPSHOT_KEYS:
        if key not in snapshot:
            raise ValueError(f"commit snapshot is missing required field: {key}")
        value = snapshot[key]
        if not _is_int(value) or value < 0:
            raise ValueError(f"commit snapshot field {key} must be a non-negative integer")
    if snapshot["available_bytes"] < minimum_mib * MIB:
        raise ValueError(
            f"Windows system commit headroom {snapshot['available_mib']} MiB is below the "
            f"requested {minimum_mib} MiB. This is commit budget, not free RAM: "
            "close other processes or review the Windows paging-file size")
    return {"limit_mib": snapshot["limit_mib"], "used_mib": snapshot["used_mib"],
            "available_mib": snapshot["available_mib"], "minimum_mib": minimum_mib,
            "meets_minimum": True}
