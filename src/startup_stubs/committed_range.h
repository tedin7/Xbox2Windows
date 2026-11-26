#pragma once

#include <windows.h>

#include <cstddef>
#include <cstdint>

namespace x2w_local {

// Number of leading bytes of [start, start + bytes) that are committed and readable, found with VirtualQuery alone: the
// memory is never touched, so a range that is not there yet costs no first-chance exception (with a debugger attached each
// one stops the whole process, and the title's lazily committed memory (MEM_RESERVE until the title writes it) turns every
// first probe of a new 64 KiB slot into one). A reserved or free page is "not there yet"; a no-access or guard page is
// refused too. Regions are walked one query at a time, so the cost follows the number of attribute changes in the range
// (a texture footprint is a few), not the size of the allocation around it.
template <class Query>
inline std::size_t committed_prefix_with(std::uintptr_t start, std::size_t bytes, Query&& query) noexcept
{
    if (bytes == 0 || start + bytes < start) return 0;
    const std::uintptr_t end = start + bytes;
    std::uintptr_t at = start;
    while (at < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (!query(reinterpret_cast<const void*>(at), &info)) return static_cast<std::size_t>(at - start);
        const DWORD protect = info.Protect & 0xFF;
        if (info.State != MEM_COMMIT || (info.Protect & PAGE_GUARD) || protect == PAGE_NOACCESS || protect == 0)
            return static_cast<std::size_t>(at - start);
        const auto region_end = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
        if (region_end <= at) return static_cast<std::size_t>(at - start);  // no progress: refuse rather than loop
        at = region_end < end ? region_end : end;
    }
    return bytes;
}

// Where the data of [start, start + bytes) can end, from VirtualQuery alone: the offset just past its last committed, readable
// byte. Reserved-only pages are memory the title has not written (lazy commit: its first write commits the slot), so they read as
// zero and need no copy; a write there turns them committed, which the next call sees. `ok` is false when a part of the range
// is neither (free, no-access or guard pages): the caller treats the range as unreadable, as before.
template <class Query>
inline std::size_t committed_extent_with(std::uintptr_t start, std::size_t bytes, Query&& query, bool* ok) noexcept
{
    *ok = false;
    if (bytes == 0 || start + bytes < start) return 0;
    const std::uintptr_t end = start + bytes;
    std::uintptr_t at = start;
    std::uintptr_t last_committed_end = start;
    while (at < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (!query(reinterpret_cast<const void*>(at), &info)) return 0;
        const auto region_end = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
        if (region_end <= at) return 0;
        const std::uintptr_t stop = region_end < end ? region_end : end;
        if (info.State == MEM_COMMIT) {
            const DWORD protect = info.Protect & 0xFF;
            if ((info.Protect & PAGE_GUARD) || protect == PAGE_NOACCESS || protect == 0) return 0;
            last_committed_end = stop;
        } else if (info.State != MEM_RESERVE) {
            return 0;
        }
        at = stop;
    }
    *ok = true;
    return static_cast<std::size_t>(last_committed_end - start);
}

inline std::size_t committed_extent(const void* start, std::size_t bytes, bool* ok) noexcept
{
    return committed_extent_with(reinterpret_cast<std::uintptr_t>(start), bytes, [](const void* address, MEMORY_BASIC_INFORMATION* info) {
        return VirtualQuery(address, info, sizeof(*info)) == sizeof(*info);
    }, ok);
}

inline std::size_t committed_prefix(const void* start, std::size_t bytes) noexcept
{
    return committed_prefix_with(reinterpret_cast<std::uintptr_t>(start), bytes, [](const void* address, MEMORY_BASIC_INFORMATION* info) {
        return VirtualQuery(address, info, sizeof(*info)) == sizeof(*info);
    });
}

inline bool committed_range(const void* start, std::size_t bytes) noexcept { return committed_prefix(start, bytes) == bytes && bytes != 0; }

}  // namespace x2w_local
