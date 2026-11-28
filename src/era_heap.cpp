#include "era_heap.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <malloc.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace era_heap {
namespace {
constexpr std::uint32_t heap = 0;
constexpr std::uint32_t graphics_first = 1;
constexpr std::uint32_t graphics_last = 6;

bool is_graphics(std::uint32_t memory_type) noexcept
{
    return memory_type >= graphics_first && memory_type <= graphics_last;
}

struct PageGeometry {
    std::size_t page;
    std::uintptr_t granularity;
};

const PageGeometry& geometry() noexcept
{
    static const PageGeometry value = [] {
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        return PageGeometry{info.dwPageSize, info.dwAllocationGranularity};
    }();
    return value;
}

std::size_t allocation_granularity() noexcept
{
    return geometry().granularity;
}

// Alignments above the page size come from their own reservation. Only the pages that
// hold the payload are committed, and VirtualAlloc already returns them zeroed.
void* allocate_pages(std::size_t bytes, std::size_t alignment) noexcept
{
    const auto page = geometry().page;
    const auto committed = (bytes + page - 1) & ~(page - 1);
    for (int attempt = 0; attempt < 8; ++attempt) {
        auto* probe = VirtualAlloc(nullptr, committed + alignment, MEM_RESERVE, PAGE_NOACCESS);
        if (!probe) {
            return nullptr;
        }
        const auto aligned = (reinterpret_cast<std::uintptr_t>(probe) + alignment - 1) & ~(alignment - 1);
        VirtualFree(probe, 0, MEM_RELEASE);
        // Another thread can take the range between the release and this call.
        if (auto* p = VirtualAlloc(reinterpret_cast<void*>(aligned), committed,
                                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) {
            return p;
        }
    }
    return nullptr;
}
}

Fields decode(std::uint64_t attributes) noexcept
{
    return {
        static_cast<std::uint32_t>(attributes & 0x3fff),
        static_cast<std::uint32_t>((attributes >> 14) & 0x3),
        static_cast<std::uint32_t>((attributes >> 16) & 0xff),
        static_cast<std::uint32_t>((attributes >> 24) & 0x1f),
        static_cast<std::uint32_t>((attributes >> 29) & 0xf),
        static_cast<std::uint32_t>((attributes >> 33) & 1),
        static_cast<std::uint32_t>((attributes >> 34) & 0x3fffffff),
    };
}

void* allocate(std::size_t bytes, std::uint64_t attributes) noexcept
{
    const auto fields = decode(attributes);
    // Object type and allocator ID are tags. SlimEra's default heap ignores them.
    if (bytes == 0 || fields.reserved != 0 || fields.page_size == 3) {
        return nullptr;
    }
    // The title asked for an address below 4 GiB. This allocator does not yet provide
    // one, so refuse instead of returning an address above that boundary.
    if (fields.address_below_4gib) {
        return nullptr;
    }
    const auto alignment = std::size_t{1} << std::max(4u, fields.alignment);
    // Leave room for CRT alignment bookkeeping even for deliberately oversized input.
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (bytes > limit - alignment - sizeof(void*)) {
        return nullptr;
    }

    if (fields.memory_type == heap) {
        // As in SlimEra, the page size does not affect heap requests. Unlike SlimEra, which
        // uses _aligned_malloc for every alignment, alignments above a page commit only the
        // payload pages instead of the CRT's alignment padding.
        if (alignment > geometry().page) {
            return allocate_pages(bytes, alignment);
        }
        void* allocation = _aligned_malloc(bytes, alignment);
        if (allocation) {
            std::memset(allocation, 0, bytes);
        }
        return allocation;
    }

    if (is_graphics(fields.memory_type)) {
        // SlimEra commits graphics memory as a separate virtual allocation. The desktop
        // places it on an allocation-granularity boundary (64 KiB on x64 Windows), which
        // covers 4 KB and 64 KB pages. Larger page sizes or alignments are not honored here.
        constexpr std::size_t page_bytes[]{std::size_t{4} << 10, std::size_t{64} << 10, std::size_t{2} << 20};
        if (std::max(page_bytes[fields.page_size], alignment) > allocation_granularity()) {
            return nullptr;
        }
        // Committed pages are zero-filled by the system.
        return VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    }

    // Physical memory types need title physical pages, which this experiment does not model.
    return nullptr;
}

void release(void* allocation) noexcept
{
    // _aligned_malloc keeps its base pointer below the returned address, so a CRT block
    // never starts its own allocation. Only page-backed blocks satisfy this test.
    const auto address = reinterpret_cast<std::uintptr_t>(allocation);
    MEMORY_BASIC_INFORMATION info{};
    if (allocation && (address & (geometry().granularity - 1)) == 0 &&
        VirtualQuery(allocation, &info, sizeof(info)) == sizeof(info) &&
        info.AllocationBase == allocation) {
        VirtualFree(allocation, 0, MEM_RELEASE);
        return;
    }
    _aligned_free(allocation);
}

void release_with_attributes(void* allocation, std::uint64_t attributes) noexcept
{
    if (!allocation) {
        return;
    }
    const auto memory_type = decode(attributes).memory_type;
    if (memory_type == heap) {
        release(allocation);
    } else if (is_graphics(memory_type)) {
        VirtualFree(allocation, 0, MEM_RELEASE);
    }
}
}
