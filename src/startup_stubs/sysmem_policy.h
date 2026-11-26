#pragma once

// System-memory fallback for a texture whose default-heap creation ran out of video memory.
// Device-free rules only (no Windows headers), so they can be checked on any platform.
//
// Source: Microsoft D3D12 docs: D3D12_HEAP_PROPERTIES (a custom heap is a pool plus a CPU page property;
// MEMORY_POOL_L0 is system memory; on a discrete adapter CPU_PAGE_PROPERTY_NOT_AVAILABLE belongs to
// MEMORY_POOL_L1), D3D12_MEMORY_POOL, D3D12_CPU_PAGE_PROPERTY and D3D12_FEATURE_DATA_ARCHITECTURE1 (UMA).

#include <cstdint>

namespace x2w_sysmem {

inline constexpr std::int32_t out_of_memory = static_cast<std::int32_t>(0x8007000e);

// The numeric values of D3D12_CPU_PAGE_PROPERTY.
enum CpuPage : int { not_available = 1, write_combine = 2, write_back = 3 };

struct Candidates {
    int count;
    int pages[3];
};

// The retry happens only for an out-of-memory failure of a default-heap creation, with the switch on.
// Any other failure, and every other heap type, keeps the original result.
constexpr bool wants_fallback(std::int32_t hr, bool switch_on, bool default_heap)
{
    return switch_on && default_heap && hr == out_of_memory;
}

// CPU page properties to try, in order, for a custom heap in the system pool (L0).
// A shared-memory adapter (UMA) accepts "not available" (the GPU-only page type of its default heap). A discrete adapter
// needs a CPU-visible page type for a system-memory heap.
constexpr Candidates candidate_pages(bool uma)
{
    return uma ? Candidates{3, {not_available, write_combine, write_back}}
               : Candidates{2, {write_combine, write_back}};
}

}  // namespace x2w_sysmem
