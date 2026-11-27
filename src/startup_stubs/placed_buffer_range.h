#pragma once

// The memory question device slot 133 asks before it aliases a buffer at a title address: which
// allocation holds the address, of which memory type, and is every byte from the allocation base to
// the end of the buffer committed in that allocation. The answer must equal the one the full walk
// gives (measure the committed run from the base with one VirtualQuery per region, then compare the
// buffer with that length), and nothing is remembered between calls: memory can be decommitted,
// released or replaced directly through the Windows API without the stand-ins seeing it.
//
// VirtualQuery is not cheap on large private regions: to report RegionSize it scans the page-table
// entries from the queried address to the end of the run of identical attributes, so a query costs
// time proportional to the touched pages up to the next attribute change (milliseconds per GiB).
//
// Proof used first: QueryVirtualMemoryInformation(MemoryRegionInfo) reports, for the allocation
// holding the address, its base, its size and how many of its bytes are committed (CommitSize), from
// the allocation's own bookkeeping and without that scan. When the allocation is private and
// CommitSize equals its size, every page of it is committed, so the full walk's committed run is the
// whole allocation and the decision follows from the base and size alone. Any other answer (a partly
// committed or mapped allocation, a free address, the call missing on an older system) falls back to
// the walk from the base that stops at the end of the buffer.

#include <windows.h>
#include <cstdint>

#include "startup_stubs/local_log.h"

namespace x2w_startup {

struct PlacedBufferRange {
    void* base = nullptr;      // AllocationBase of the allocation holding the address
    DWORD type = 0;            // MEM_PRIVATE, MEM_MAPPED or MEM_IMAGE at the address
    bool covered = false;      // [base, address + width) is committed and belongs to that allocation
    bool region_info = false;  // proven by one region-information call, without VirtualQuery
    unsigned queries = 0;      // VirtualQuery calls made, for the stand-in's totals
    // Why a buffer is not covered, for the refusal cache (srv_refusal_cache.h): the first page, from the allocation
    // base to the end of the buffer, that is not committed in that allocation. 0 when none was found (covered, or the
    // answer came from the region information alone).
    std::uintptr_t witness = 0;
};

// exact is what slot 133 uses. The other two exist only for the negative controls in
// tests/startup_placed_buffer_range_check.cpp: walk_only never uses the region information, and
// region_only trusts the allocation size without checking that every page of it is committed.
enum class PlacedProof { exact, walk_only, region_only };

namespace placed_detail {
struct RegionInformation {  // WIN32_MEMORY_REGION_INFORMATION (Windows 10 1607)
    PVOID AllocationBase;
    ULONG AllocationProtect;
    ULONG Flags;            // bit 0 Private, bits 1-5 mapped data file, image, page file, physical, direct
    SIZE_T RegionSize;      // size of the whole allocation
    SIZE_T CommitSize;      // committed bytes in the allocation
};
using QueryRegion = BOOL(WINAPI*)(HANDLE, const VOID*, int, PVOID, SIZE_T, PSIZE_T);

inline QueryRegion query_region()
{
    // Resolved at run time so the stand-in still loads where the export does not exist.
    static const QueryRegion query = [] {
        HMODULE module = GetModuleHandleW(L"kernelbase.dll");
        return module ? reinterpret_cast<QueryRegion>(GetProcAddress(module, "QueryVirtualMemoryInformation")) : nullptr;
    }();
    return query;
}
} // namespace placed_detail

// Size of the whole allocation holding address, from one region-information call (no page-table scan);
// 0 when the call is missing or fails.
inline std::uint64_t allocation_bytes(const void* address)
{
    placed_detail::RegionInformation info{};
    SIZE_T returned = 0;
    const placed_detail::QueryRegion query = placed_detail::query_region();
    if (!address || !query || !query(GetCurrentProcess(), address, 0 /* MemoryRegionInfo */, &info, sizeof(info), &returned))
        return 0;
    return info.RegionSize;
}

// One region-information call for the allocation holding address: its base, size, committed bytes and flags, with no
// page-table scan. False when the call is missing or fails (the caller must then not cache anything on it).
inline bool region_stamp(const void* address, std::uint64_t& base, std::uint64_t& size, std::uint64_t& commit,
                         std::uint64_t& flags)
{
    placed_detail::RegionInformation info{};
    SIZE_T returned = 0;
    const placed_detail::QueryRegion query = placed_detail::query_region();
    if (!address || !query || !query(GetCurrentProcess(), address, 0 /* MemoryRegionInfo */, &info, sizeof(info), &returned))
        return false;
    base = reinterpret_cast<std::uintptr_t>(info.AllocationBase);
    size = info.RegionSize;
    commit = info.CommitSize;
    flags = info.Flags;
    return true;
}

namespace placed_detail {
// One pass of measure_placed_buffer; out is reset first, so a second pass replaces the first.
inline bool measure_once(const void* address, std::uint64_t width, PlacedBufferRange& out, PlacedProof proof)
{
    out = PlacedBufferRange{};
    if (!address) return false;
    const auto start = reinterpret_cast<std::uintptr_t>(address);

    if (proof != PlacedProof::walk_only) {
        placed_detail::RegionInformation info{};
        SIZE_T returned = 0;
        const placed_detail::QueryRegion query = placed_detail::query_region();
        if (query && query(GetCurrentProcess(), address, 0 /* MemoryRegionInfo */, &info, sizeof(info), &returned) &&
            (info.Flags & 0x3F) == 0x1 && info.RegionSize &&
            (info.CommitSize == info.RegionSize || proof == PlacedProof::region_only)) {
            const auto base = reinterpret_cast<std::uintptr_t>(info.AllocationBase);
            if (base <= start && start - base < info.RegionSize) {
                out.base = info.AllocationBase;
                out.type = MEM_PRIVATE;
                out.region_info = true;
                out.covered = width <= info.RegionSize - (start - base);
                return true;
            }
        }
    }

    // The walk from the base, stopped at the end of the buffer: regions after it cannot change the answer.
    MEMORY_BASIC_INFORMATION info{};
    ++out.queries;
    if (VirtualQuery(address, &info, sizeof(info)) != sizeof(info) || info.State != MEM_COMMIT) {
        out.witness = start;  // the address itself is not committed
        return false;
    }
    out.base = info.AllocationBase;
    out.type = info.Type;
    if (width > UINTPTR_MAX - start) return true;  // the end does not exist; the full walk refuses it too
    const std::uintptr_t end = start + static_cast<std::uintptr_t>(width);
    auto cursor = reinterpret_cast<std::uintptr_t>(out.base);
    MEMORY_BASIC_INFORMATION region{};
    while (cursor < end) {
        ++out.queries;
        if (!VirtualQuery(reinterpret_cast<const void*>(cursor), &region, sizeof(region)) ||
            region.AllocationBase != out.base || region.State != MEM_COMMIT) {
            out.witness = cursor;  // the walk stopped here: this page is not committed in the allocation
            return true;
        }
        cursor = reinterpret_cast<std::uintptr_t>(region.BaseAddress) + region.RegionSize;
    }
    out.covered = true;
    return true;
}
} // namespace placed_detail

// False when the address is not committed (slot 133 returns E_INVALIDARG). Otherwise covered is
// exactly "offset <= committed bytes from base && width <= committed bytes - offset" for the run the
// full walk measures. The queries run outside xmem's run breaks (x2w_local::outside_run_breaks): inside
// one, pages of a shared-view run read as reserved, and the buffer would be refused although it is mapped.
inline bool measure_placed_buffer(const void* address, std::uint64_t width, PlacedBufferRange& out,
                                  PlacedProof proof = PlacedProof::exact)
{
    bool committed = false;
    unsigned queries = 0;  // over every pass: a pass run again after a break replaces out
    x2w_local::outside_run_breaks([&] {
        committed = placed_detail::measure_once(address, width, out, proof);
        queries += out.queries;
    });
    out.queries = queries;
    return committed;
}

// True when an uncovered buffer at [address, address + width) lies inside a MEM_PRIVATE
// reservation at a 64 KiB-aligned offset, fits within its whole allocation size (nonzero and at
// most max_allocation_bytes when nonzero), and has nonzero width. Used before committing a lazy
// startup block (X2W_XMEM_LAZY_STARTUP) and re-measuring so an aligned nonzero offset with a
// MEM_RESERVE prefix can still alias an eligible small block without committing ineligible or
// out-of-bounds blocks.
inline bool can_commit_lazy_placed_buffer(const void* address, std::uint64_t width,
                                          std::uint64_t max_allocation_bytes)
{
    if (!address || !width) return false;
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(address, &info, sizeof(info)) != sizeof(info) || info.State != MEM_COMMIT ||
        info.Type != MEM_PRIVATE || !info.AllocationBase)
        return false;
    const auto start = reinterpret_cast<std::uintptr_t>(address);
    const auto base = reinterpret_cast<std::uintptr_t>(info.AllocationBase);
    if (start < base) return false;
    const std::uint64_t offset = start - base;
    if (offset & 0xFFFF) return false;
    const std::uint64_t alloc_bytes = allocation_bytes(info.AllocationBase);
    if (!alloc_bytes || (max_allocation_bytes && alloc_bytes > max_allocation_bytes)) return false;
    return offset <= alloc_bytes && width <= alloc_bytes - offset;
}

} // namespace x2w_startup
