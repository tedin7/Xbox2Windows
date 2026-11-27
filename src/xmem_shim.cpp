// Desktop xmem.dll. The default allocation exports share the unhooked allocator.
// Call shapes come from the observed call sites, not from a published header:
//   XMemAlloc(size, attributes) and XMemFree(allocation, attributes): Bink and AMD callers.
//   XMemVirtualQuery(process, address, buffer): the four BVH callers pass no length, supply a
//     zeroed 64-byte buffer and test the 32-bit result.
//   XMemGetWorkingSetStatistics(selector, output): the seven AMD callers pass 0x20000001 and
//     read the 64-bit value at +8 (zero on failure).
// XMemAllocatePhysicalPages(flags, &count, pages) and XMemFreePhysicalPages(count, pages) follow the
// calls in the public Microsoft Xbox GDK samples; era_physical keeps the page bookkeeping.
// Graphics allocations go through era_title_gpu, which registers them with era_query and, once a
// D3D12 device is attached to the registry, gives them desktop GPU addresses. The exports are
// unchanged; nothing here loads D3D12. DXGI and gdi32 are delay-loaded.
#include "era_heap.h"
#include "era_query.h"
#include "era_title_gpu.h"
#include "xmem_gpu_usage.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/profile_zone.h"

#include <unordered_set>

#include <cstdint>
#include <cstring>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {
// The BVH callers' query buffer. era_query writes at most its first 0x31 bytes.
constexpr std::size_t query_buffer_bytes = 64;

bool is_graphics(std::uint64_t attributes) noexcept
{
    const auto type = era_heap::decode(attributes).memory_type;
    return type >= 1 && type <= 6;
}

// The selector the AMD callers pass, and the offset of the value they read.
constexpr UINT32 gpu_usage_selector = 0x20000001;
constexpr std::size_t gpu_usage_offset = 8;
}

// Experimental (docs/startup-runtime.md). With X2W_XMEM_PERMISSIVE=1, attribute
// words the allocator refuses (seen: 0xf0814000, memory type 7, a physical type, 0x60000 bytes for
// the audio stream queue) are served from committed process memory at 64 KiB granularity.
namespace {
SRWLOCK local_lock = SRWLOCK_INIT;
std::unordered_set<void*>& local_blocks()
{
    static std::unordered_set<void*> blocks;
    return blocks;
}

void* local_alloc(SIZE_T bytes, ULONGLONG attributes)
{
    void* block = bytes ? VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) : nullptr;
    if (block) {
        x2w_local::record_read_span(block, bytes);
        AcquireSRWLockExclusive(&local_lock);
        local_blocks().insert(block);
        ReleaseSRWLockExclusive(&local_lock);
    }
    static LONG calls;
    if (x2w_local::first_calls(calls, 16))
        x2w_local::logf("x2w-xmem XMemAlloc size=0x%llx attributes=0x%llx refused by the allocator -> %p (process memory)",
                        static_cast<unsigned long long>(bytes), attributes, block);
    return block;
}

bool local_free(void* block)
{
    AcquireSRWLockExclusive(&local_lock);
    const bool ours = local_blocks().erase(block) != 0;
    ReleaseSRWLockExclusive(&local_lock);
    if (ours) x2w_local::release_read_span(block);
    return ours;
}
}  // namespace

extern "C" void* XMemAlloc(SIZE_T bytes, ULONGLONG attributes)
{
    void* allocation = era_heap::allocate(bytes, attributes);
    // Desktop policy: graphics memory types count as GPU-intended for XMemVirtualQuery.
    if (!allocation && bytes && x2w_local::env_on("X2W_XMEM_PERMISSIVE")) allocation = local_alloc(bytes, attributes);
    else if (allocation && is_graphics(attributes) && !era_title_gpu::track(allocation, bytes)) {
        era_heap::release_with_attributes(allocation, attributes);
        return nullptr;
    }
    if (allocation) X2W_ALLOC(allocation, bytes, "XMemAlloc");
    return allocation;
}

extern "C" void XMemFree(void* allocation, ULONGLONG attributes)
{
    if (allocation) X2W_FREE(allocation, "XMemAlloc");
    if (allocation && local_free(allocation)) return;
    if (allocation && is_graphics(attributes)) {
        era_title_gpu::untrack(allocation);
    }
    era_heap::release_with_attributes(allocation, attributes);
}

extern "C" BOOL XMemVirtualQuery(HANDLE process, LPCVOID address, void* buffer)
{
    return era_query::query(process, address, buffer, query_buffer_bytes) != 0;
}

// Desktop policy (docs/experiments.md): writes only +8, with this process's GPU memory in bytes;
// on failure the output is untouched and the callers use zero.
extern "C" BOOL XMemGetWorkingSetStatistics(UINT32 selector, void* statistics)
{
    if (!statistics) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    const auto bytes = selector == gpu_usage_selector ? xmem_gpu_usage::process_bytes() : std::nullopt;
    if (!bytes) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return FALSE;
    }
    std::memcpy(static_cast<unsigned char*>(statistics) + gpu_usage_offset, &*bytes, sizeof(*bytes));
    return TRUE;
}

extern "C" BOOL WINAPI XMemIsToolsMemAvailable()
{
    // The desktop runtime has no devkit tools-memory partition.
    return FALSE;
}

// Desktop policy (docs/experiments.md): on a no-tools desktop runtime without devkit
// tools-memory backing (XMemIsToolsMemAvailable() == FALSE per #896), transfer memory
// returns FALSE with ERROR_NOT_SUPPORTED and leaves memory and accounting state unchanged.
// Without independent evidence of runtime parameter validation returning 87
// (ERROR_INVALID_PARAMETER), all calls take this narrow refusal.
extern "C" BOOL WINAPI XMemTransferMemory(UINT32 partition, UINT64 byte_count)
{
    (void)partition;
    (void)byte_count;
    SetLastError(ERROR_NOT_SUPPORTED);
    return FALSE;
}
