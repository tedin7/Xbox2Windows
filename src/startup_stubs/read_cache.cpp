#include "startup_stubs/read_cache.h"

#include <algorithm>
#include <atomic>
#include <cstring>

namespace {
using Address = std::uintptr_t;
using x2w_local::ReadCacheStats;

struct Span {
    Address begin, end;
};

constexpr SIZE_T capacity = 16384;
SRWLOCK lock = SRWLOCK_INIT;
Span spans[capacity];
SIZE_T count;
unsigned long long generation, erases;
std::atomic<unsigned long long> lookups{}, hits{}, queries{};

Span* first_ending_after(Address address) noexcept
{
    return std::upper_bound(spans, spans + count, address,
                            [](Address at, const Span& span) { return at < span.end; });
}

// Caller holds the exclusive lock. At capacity, a new disjoint span can be
// rediscovered on demand. An erase always takes effect, even if a split is full.
void insert(Address begin, Address end) noexcept
{
    auto* first = first_ending_after(begin - 1);
    auto* last = first;
    while (last != spans + count && last->begin <= end) {
        begin = (std::min)(begin, last->begin);
        end = (std::max)(end, last->end);
        ++last;
    }
    if (first == last && count == capacity) return;
    const SIZE_T tail = spans + count - last;
    std::memmove(first + 1, last, tail * sizeof(Span));
    *first = {begin, end};
    count = first - spans + 1 + tail;
}

bool readable_protection(DWORD protect) noexcept
{
    return !(protect & PAGE_GUARD) && (protect & 0xff) != PAGE_NOACCESS && (protect & 0xff) != 0;
}

bool query(const void* pointer, MEMORY_BASIC_INFORMATION* info) noexcept
{
    const auto address = reinterpret_cast<Address>(pointer);
    lookups.fetch_add(1, std::memory_order_relaxed);
    AcquireSRWLockShared(&lock);
    const auto* span = first_ending_after(address);
    const auto version = generation;
    if (span != spans + count && span->begin <= address) {
        *info = {};
        info->BaseAddress = reinterpret_cast<void*>(span->begin);
        info->RegionSize = span->end - span->begin;
        info->State = MEM_COMMIT;
        info->Protect = PAGE_READONLY;  // Readability fact only. Never use this for write permission.
        hits.fetch_add(1, std::memory_order_relaxed);
        ReleaseSRWLockShared(&lock);
        return true;
    }
    ReleaseSRWLockShared(&lock);

    // Keep the OS walk outside the table lock. A concurrent mutation invalidates the result for insertion.
    queries.fetch_add(1, std::memory_order_relaxed);
    if (VirtualQuery(pointer, info, sizeof(*info)) != sizeof(*info)) return false;
    if (info->State == MEM_COMMIT && readable_protection(info->Protect)) {
        const auto begin = reinterpret_cast<Address>(info->BaseAddress);
        AcquireSRWLockExclusive(&lock);
        if (version == generation) insert(begin, begin + info->RegionSize);
        ReleaseSRWLockExclusive(&lock);
    }
    return true;
}

void record(const void* pointer, SIZE_T bytes, DWORD protect) noexcept
{
    const auto start = reinterpret_cast<Address>(pointer);
    if (!start || !bytes || start > UINTPTR_MAX - 0xfff ||
        bytes > UINTPTR_MAX - start - 0xfff || !readable_protection(protect)) return;
    AcquireSRWLockExclusive(&lock);
    ++generation;
    insert(start & ~Address{0xfff}, (start + bytes + 0xfff) & ~Address{0xfff});
    ReleaseSRWLockExclusive(&lock);
}

void erase(const void* pointer, SIZE_T bytes) noexcept
{
    const auto start = reinterpret_cast<Address>(pointer);
    if (!bytes || start > UINTPTR_MAX - 0xfff || bytes > UINTPTR_MAX - start - 0xfff) return;
    const Address begin = start & ~Address{0xfff};
    const Address end = (start + bytes + 0xfff) & ~Address{0xfff};
    AcquireSRWLockExclusive(&lock);
    ++generation;
    auto* first = first_ending_after(begin);
    if (first != spans + count && first->begin < end) {
        auto* last = std::lower_bound(first, spans + count, end,
                                      [](const Span& span, Address at) { return span.begin < at; });
        const bool left = first->begin < begin;
        const Span right = {end, last[-1].end};
        auto* out = first;
        const auto tail = spans + count - last;
        const Address left_begin = first->begin;
        const bool keep_right = right.begin < right.end && (first - spans) + left + 1 + tail <= capacity;
        out += left + keep_right;
        std::memmove(out, last, tail * sizeof(Span));
        if (left) *first++ = {left_begin, begin};
        if (keep_right) *first = right;
        count = out - spans + tail;
        ++erases;
    }
    ReleaseSRWLockExclusive(&lock);
}

ReadCacheStats stats() noexcept
{
    AcquireSRWLockShared(&lock);
    const ReadCacheStats result{lookups.load(std::memory_order_relaxed), hits.load(std::memory_order_relaxed),
                                queries.load(std::memory_order_relaxed), count, erases};
    ReleaseSRWLockShared(&lock);
    return result;
}
}  // namespace

extern "C" const x2w_local::ReadCacheApi* WINAPI x2w_xmem_read_cache() noexcept
{
    static constexpr x2w_local::ReadCacheApi api{query, record, erase, stats};
    return &api;
}
