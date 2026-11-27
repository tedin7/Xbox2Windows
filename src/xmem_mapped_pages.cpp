#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "era_physical.h"
#include "xmem_slab_slots.h"
#include "xmem_section_lease.h"
#include "xmem_page_batch.h"
#include "xmem_import_patch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "startup_stubs/commit_state.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/resident_pages.h"
#include "startup_stubs/profile_zone.h"

#include <vector>

#include <iterator>
#include <limits>
#include <atomic>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <utility>

namespace {
struct Backing {
    HANDLE section;
    ULONG_PTR views = 0;
    ULONG64 offset = 0;  // the page is one 64 KiB slot of a shared section (see section_for)
    std::size_t slab = 0;  // index of that section among the slabs
    bool committed = true;  // false while a lazily mapped slot is untouched (X2W_XMEM_LAZY_COMMIT)
};

using Slot = std::pair<std::size_t, ULONG64>;
struct SlotPin { unsigned references{}; bool retired{}; };
struct SectionLease {
    UINT version;
    SIZE_T first_byte;
    SIZE_T length;
    std::vector<Slot> slots;
};

// Keep grant transitions and section views serialized. A grant can outlive all its
// views so its bytes survive unmapping. Its backing is released before reuse.
struct PhysicalMemory {
    std::mutex mutex;
    std::map<ULONG_PTR, Backing> backings;
    std::map<ULONG_PTR, ULONG_PTR> views;
    std::map<ULONG_PTR, SIZE_T> reservations;
    // A no-access graphics reservation (protection 0x801) made right after a
    // read-write one of the same size is that memory's GPU view on the console. Pages mapped into
    // the CPU view are mapped at the same offset into the GPU view too (the same section).
    ULONG_PTR last_cpu_base = 0;
    SIZE_T last_cpu_bytes = 0;
    ULONG_PTR mirror_cpu = 0, mirror_gpu = 0;
    SIZE_T mirror_bytes = 0;
    std::set<ULONG_PTR> explicit_mirror_pages;
    // OS views: base -> bytes, and the id of the run break inside which the view was mapped (0: mapped outside any
    // break). retry_remapped_page reads the id; it never expires while the view lasts.
    struct Run {
        SIZE_T bytes = 0;
        ULONG64 stamp = 0;
    };
    std::map<ULONG_PTR, Run> runs;
    // The slabs and their 64 KiB slots; released slots are reused lowest first, so that pages reused together
    // are adjacent slots again.
    x2w_xmem::SlabSlots<HANDLE> slots{era_physical::page_bytes};
    // A grant may be freed after unmapping while a submitted GPU operation
    // still uses the section. Keep those slots out of the reusable pool.
    std::map<Slot, SlotPin> pins;
    std::map<ULONG64, SectionLease> leases;
    ULONG64 next_lease = 1;

    ~PhysicalMemory()
    {
        for (const auto& run : runs)
            UnmapViewOfFile(reinterpret_cast<void*>(run.first));
        for (const HANDLE slab : slots.slabs())
            if (slab) CloseHandle(slab);
    }
};

PhysicalMemory memory;
constexpr SIZE_T page_bytes = era_physical::page_bytes;

// Caller holds memory.mutex. A lease may only refer to committed pages whose
// CPU and GPU views still name the same grant.
bool mirrored_range(ULONG_PTR address, SIZE_T length)
{
    if (!length || !memory.mirror_gpu || address < memory.mirror_gpu || address > UINT64_MAX - length)
        return false;
    const auto offset = address - memory.mirror_gpu;
    return offset < memory.mirror_bytes && length <= memory.mirror_bytes - offset;
}

const Backing* mirrored_backing(ULONG_PTR page)
{
    const auto mirror = memory.views.find(page);
    const auto cpu = memory.views.find(memory.mirror_cpu + (page - memory.mirror_gpu));
    if (mirror == memory.views.end() || (!memory.explicit_mirror_pages.count(page) &&
        (cpu == memory.views.end() || cpu->second != mirror->second)))
        return nullptr;
    const auto backing = memory.backings.find(mirror->second);
    return backing != memory.backings.end() && backing->second.committed ? &backing->second : nullptr;
}

// Caller holds memory.mutex.
void collect_empty_slabs(std::vector<HANDLE>& closed) noexcept
{
    memory.slots.reclaim_empty_slabs([&](HANDLE slab) noexcept {
        try {
            closed.push_back(slab);
        } catch (...) {
            CloseHandle(slab);
        }
    });
}

// Caller holds memory.mutex.
void unpin_slot(const Slot& slot, std::vector<HANDLE>* closed = nullptr)
{
    const auto found = memory.pins.find(slot);
    if (--found->second.references) return;
    const bool retired = found->second.retired;
    memory.pins.erase(found);
    if (retired && memory.slots.release(slot.first, slot.second) && closed)
        collect_empty_slabs(*closed);
}

BOOL physical_result(era_physical::Status status) noexcept
{
    switch (status) {
    case era_physical::Status::ok:
        SetLastError(ERROR_SUCCESS);
        return TRUE;
    case era_physical::Status::exhausted:
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return FALSE;
    default:
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
}

// Pages are slots of 256 MiB pagefile-backed sections rather than one section each: one handle and
// one view per page is what makes large batches slow. A slab is created with SEC_RESERVE, so it costs
// no commit until a slot is used; a slot is committed here, when its page is first mapped or written,
// and stays committed after it is freed so that reuse cannot fail for lack of commit. A freed slot is
// zeroed when it is reused, as a fresh section was. Fails with the system error (for instance
// ERROR_COMMITMENT_LIMIT) when the slot cannot be committed, leaving no state changed.
bool commit_slot(HANDLE slab, ULONG64 offset, bool zero)
{
#ifdef X2W_XMEM_TEST_CONTROLS
    // Refuse the first commit in a dedicated test process without consuming system commit.
    static bool refuse_commit = x2w_local::env_on("X2W_XMEM_TEST_COMMIT_FAILURE");
    if (refuse_commit) {
        refuse_commit = false;
        SetLastError(ERROR_COMMITMENT_LIMIT);
        return false;
    }
#endif
    void* view = MapViewOfFile(slab, FILE_MAP_WRITE, static_cast<DWORD>(offset >> 32), static_cast<DWORD>(offset), page_bytes);
    if (!view) return false;
    bool ok = VirtualAlloc(view, page_bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
    const DWORD error = ok ? ERROR_SUCCESS : GetLastError();
    if (ok && zero) std::memset(view, 0, page_bytes);  // a reused slot starts zero-filled
    UnmapViewOfFile(view);
    if (!ok) SetLastError(error);
    return ok;
}

// Experimental (X2W_XMEM_LAZY_COMMIT=1, #1199): a fresh slot is not committed when its page is mapped. The first
// CPU access faults and lazy_commit_fault commits that slot. Writers that do not fault -- kernel-mode I/O into
// mapped pages, GPU placement, stand-in writes behind a commit-state check -- call x2w_xmem_commit_range first.
// Reused slots and x2w_xmem_write_physical still commit before use.
bool lazy_commit_requested()
{
    static const bool on = x2w_local::env_on("X2W_XMEM_LAZY_COMMIT");
    return on;
}

// Set when xmem.dll loads, once the fault handler is installed. Without the handler a lazy page could never be
// committed, so the switch then acts as off: pages and blocks are committed up front.
bool lazy_handler_installed = false;

bool lazy_commit()
{
    return lazy_handler_installed && lazy_commit_requested();
}

// Experimental (X2W_XMEM_LAZY_STARTUP=1 with X2W_XMEM_LAZY_COMMIT=1, #1199): the large startup blocks
// XMemVirtualAlloc commits for the title (commit_startup_block, 32 MiB and up) are only reserved. The fault
// handler commits them 256 KiB at a time on first touch. x2w_xmem_commit_range commits the chunks behind a range
// for writers that cannot fault, and x2w_xmem_commit_block commits a whole block before the GPU uses it.
bool lazy_startup()
{
    static const bool on = x2w_local::env_on("X2W_XMEM_LAZY_STARTUP");
    return on && lazy_commit();
}

constexpr SIZE_T lazy_chunk = SIZE_T{256} << 10;
constexpr SIZE_T lazy_block_min = SIZE_T{32} << 20;

// The reserved startup blocks that are still committed lazily: base -> bytes.
struct LazyBlocks {
    SRWLOCK lock = SRWLOCK_INIT;
    std::map<ULONG_PTR, SIZE_T> blocks;
};
LazyBlocks lazy_blocks;
volatile LONG lazy_block_releases = 0;  // lazy blocks the title has released (keys the fault handler's retry)

// The lazy block holding address, as base and size; false when there is none. The caller holds lazy_blocks.lock
// until it has committed what it found, so that a release (virtual_free_lazy) cannot run in between and leave it
// committing addresses that no longer belong to the block.
bool lazy_block_at(ULONG_PTR address, ULONG_PTR& base, SIZE_T& bytes)
{
    auto it = lazy_blocks.blocks.upper_bound(address);
    const bool inside = it != lazy_blocks.blocks.begin() && (--it, address - it->first < it->second);
    if (inside) {
        base = it->first;
        bytes = it->second;
    }
    return inside;
}

// Commits the whole chunks of a lazy block that [from, to) touches; false with the system error if that fails.
bool commit_block_chunks(ULONG_PTR base, SIZE_T bytes, ULONG_PTR from, ULONG_PTR to)
{
    const SIZE_T first = (from - base) & ~(lazy_chunk - 1);
    SIZE_T last = to - base < bytes ? to - base : bytes;  // offsets in the block; to is exclusive
    last = (last + lazy_chunk - 1) & ~(lazy_chunk - 1);
    if (last > bytes) last = bytes;
    return VirtualAlloc(reinterpret_cast<void*>(base + first), last - first, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}

// Commits a lazily mapped slot under memory.mutex; false with the system error if the commit fails.
bool commit_backing(Backing& backing)
{
    if (backing.committed) return true;
    if (!commit_slot(backing.section, backing.offset, false)) return false;
    backing.committed = true;
    return true;
}

// The backing behind a mapped address (CPU view, or its GPU-view mirror), under memory.mutex.
Backing* backing_at(ULONG_PTR address)
{
    const auto view = memory.views.find(address & ~(ULONG_PTR{page_bytes} - 1));
    if (view == memory.views.end()) return nullptr;
    const auto backing = memory.backings.find(view->second);
    return backing == memory.backings.end() ? nullptr : &backing->second;
}

bool valid_lease_range(UINT version, ULONG_PTR address, SIZE_T length)
{
    if (version == x2w_xmem_section_lease_version) return mirrored_range(address, length);
    return version == x2w_xmem_single_page_lease_version && length &&
           length <= page_bytes - (address % page_bytes) && address <= UINTPTR_MAX - (length - 1);
}

const Backing* lease_backing(UINT version, ULONG_PTR page)
{
    if (version == x2w_xmem_section_lease_version || mirrored_range(page, page_bytes))
        return mirrored_backing(page);
    const auto* backing = backing_at(page);
    return backing && backing->committed ? backing : nullptr;
}

// Set while this thread holds memory.mutex to change mappings or to copy into mapped pages (MappingThread). The
// fault handlers pass such a thread's faults on: waiting for memory.mutex there would never return.
thread_local bool mapping_thread = false;

LONG CALLBACK lazy_commit_fault(EXCEPTION_POINTERS* e)
{
    const EXCEPTION_RECORD* record = e->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    const ULONG_PTR address = record->ExceptionInformation[1];
    if (lazy_startup()) {
        ULONG_PTR base = 0;
        SIZE_T bytes = 0;
        MEMORY_BASIC_INFORMATION info{};
        AcquireSRWLockShared(&lazy_blocks.lock);
        const bool lazy = lazy_block_at(address, base, bytes) &&
                          VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) && info.State == MEM_RESERVE;
        const bool committed = lazy && commit_block_chunks(base, bytes, address, address + 1);
        ReleaseSRWLockShared(&lazy_blocks.lock);
        if (lazy) return committed ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
        // Another thread committed this chunk (or the whole block, which then leaves the list) between the fault
        // and the query: committed read-write memory cannot fault, so retry once. The retry is keyed on the chunk
        // and on the release count, so a fault that stays is passed on, while a new block later placed at the same
        // addresses can still be raced once (separate from the slot path's retry below).
        thread_local ULONG_PTR retried_chunk = 0;
        thread_local LONG retried_releases = 0;
        const ULONG_PTR chunk = address & ~(ULONG_PTR{lazy_chunk} - 1);
        const LONG releases = lazy_block_releases;
        if (VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) && info.Type == MEM_PRIVATE &&
            info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE &&
            !(retried_chunk == chunk && retried_releases == releases)) {
            retried_chunk = chunk;
            retried_releases = releases;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    // A thread that raced another one to the same slot finds it committed: retry once, then pass the fault on.
    thread_local ULONG_PTR retried = 0;
    const ULONG_PTR page = address & ~(ULONG_PTR{page_bytes} - 1);
    if (mapping_thread) return EXCEPTION_CONTINUE_SEARCH;
    const std::lock_guard guard(memory.mutex);
    Backing* backing = backing_at(address);
    if (!backing) return EXCEPTION_CONTINUE_SEARCH;
    if (backing->committed) {
        if (retried == page) return EXCEPTION_CONTINUE_SEARCH;
        retried = page;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (!commit_backing(*backing)) return EXCEPTION_CONTINUE_SEARCH;
    retried = 0;
    return EXCEPTION_CONTINUE_EXECUTION;
}

HANDLE section_for(ULONG_PTR page, bool need_commit = true)
{
    const auto existing = memory.backings.find(page);
    if (existing != memory.backings.end()) {
        if (need_commit && !commit_backing(existing->second)) return nullptr;
        return existing->second.section;
    }
    // A fresh slot (never used) of a SEC_RESERVE slab reads as zero once committed, so it can wait for its first
    // touch. A reused slot is committed and zeroed now, as before.
    const bool defer = !need_commit && lazy_commit();
    x2w_xmem::SlabSlots<HANDLE>::Slot slot;
    if (memory.slots.lowest_free(slot)) {
        if (!commit_slot(slot.slab, slot.offset, true)) return nullptr;
        memory.backings.emplace(page, Backing{slot.slab, 0, slot.offset, slot.slab_index});
        memory.slots.take_lowest_free();
        return slot.slab;
    }
    const ULONG64 slab_bytes = memory.slots.slab_bytes();
    bool first_slot_committed = false;
    const auto make_slab = [slab_bytes, &first_slot_committed, defer](HANDLE& slab) {
        slab = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE,
                                  static_cast<DWORD>(slab_bytes >> 32), static_cast<DWORD>(slab_bytes), nullptr);
        if (!slab) return false;
        if (defer) return true;
        first_slot_committed = commit_slot(slab, 0, false);
        if (!first_slot_committed) {
            const DWORD error = GetLastError();
            CloseHandle(slab);
            SetLastError(error);
        }
        return first_slot_committed;
    };
    if (!memory.slots.take_fresh(make_slab, [](HANDLE slab) { CloseHandle(slab); }, slot)) return nullptr;
    if (!defer && !first_slot_committed && !commit_slot(slot.slab, slot.offset, false)) {
        const DWORD error = GetLastError();
        memory.slots.unwind_fresh(slot);
        SetLastError(error);
        return nullptr;
    }
    try {
        memory.backings.emplace(page, Backing{slot.slab, 0, slot.offset, slot.slab_index, !defer});
    } catch (...) {
        memory.slots.unwind_fresh(slot);
        throw;
    }
    return slot.slab;
}

auto owned_reservation(ULONG_PTR address)
{
    auto it = memory.reservations.upper_bound(address);
    if (it == memory.reservations.begin()) return memory.reservations.end();
    --it;
    return address - it->first < it->second ? it : memory.reservations.end();
}

// MapViewOfFile3 and placeholder-split attempts made by map_split (x2w_xmem_map_counts).
volatile LONG64 map_attempts, split_attempts;
#ifdef X2W_XMEM_TEST_CONTROLS
// Test build only (x2w_xmem_alias_counts): views install_alias_view could not map in place inside an owned
// reservation, split by whether alias_placeholders refused the range or a placeholder release was tried.
volatile LONG64 alias_refusals, alias_splits;
#endif

// True when [target, target + bytes) is a whole reserved region of its own: a placeholder that
// MEM_REPLACE_PLACEHOLDER can take as it is, so no split is needed first.
bool exact_placeholder(void* target, SIZE_T bytes)
{
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(target, &info, sizeof(info)) == sizeof(info) && info.State == MEM_RESERVE &&
           info.AllocationBase == target && info.BaseAddress == target && info.RegionSize == bytes;
}

// Only split a reservation owned by the shim. Caller-owned placeholders must already match the view.
// Inside an owned reservation the split comes first when the target is not already an exact placeholder:
// a map into the middle of a larger placeholder always fails, and under Wine every failed map is another
// server round trip whose cost grows with the number of views (one 64 KiB page per call at startup).
// The query and the split are not atomic, so if the split or the map after it fails, the map-first
// path below runs as before. Test build only (X2W_XMEM_TEST_CONTROLS): X2W_XMEM_TEST_MAP_FIRST=1 skips the
// split-first step; the deployed build ignores it.
// Once a split succeeds, report the retry's result even if it fails for a different reason.
template <class Map>
bool map_split(void* target, SIZE_T bytes, Map map)
{
#ifdef X2W_XMEM_TEST_CONTROLS
    static const bool map_first = x2w_local::env_on("X2W_XMEM_TEST_MAP_FIRST");
#else
    constexpr bool map_first = false;
#endif
    const auto address = reinterpret_cast<ULONG_PTR>(target);
    const auto reservation = owned_reservation(address);
    const bool owned = reservation != memory.reservations.end() &&
                       bytes <= reservation->second - (address - reservation->first);
    if (owned && !map_first && !exact_placeholder(target, bytes)) {
        InterlockedIncrement64(&split_attempts);
        if (VirtualFree(target, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            InterlockedIncrement64(&map_attempts);
            if (map()) return true;
        }
    }
    InterlockedIncrement64(&map_attempts);
    if (map()) return true;
    const DWORD error = GetLastError();
    if (owned) {
        InterlockedIncrement64(&split_attempts);
        if (VirtualFree(target, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            InterlockedIncrement64(&map_attempts);
            return map();
        }
    }
    SetLastError(error);
    return false;
}

bool map_view(ULONG_PTR address, const Backing& backing)
{
    const HANDLE section = backing.section;
    const ULONG64 section_offset = backing.offset;
    auto* target = reinterpret_cast<void*>(address);
    auto map = [&] {
        return MapViewOfFile3(section, GetCurrentProcess(), target, section_offset,
                              page_bytes, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0) != nullptr;
    };
    return map_split(target, page_bytes, map);
}

// Maps bytes of one section at address as a single view; falls back to one view per page when the
// range is not inside a single placeholder. stamp: the run break in progress, or 0 (PhysicalMemory::Run).
bool map_run(ULONG_PTR address, const Backing& backing, SIZE_T bytes, ULONG64 stamp,
             std::vector<std::pair<ULONG_PTR, SIZE_T>>& made, ULONG_PTR& failed_address)
{
    auto* target = reinterpret_cast<void*>(address);
    auto map = [&] {
        return MapViewOfFile3(backing.section, GetCurrentProcess(), target, backing.offset, bytes,
                              MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0) != nullptr;
    };
    if (map_split(target, bytes, map)) {
        memory.runs[address] = {bytes, stamp};
        made.emplace_back(address, bytes);
        return true;
    }
    failed_address = address;
    if (bytes == page_bytes) return false;
    for (SIZE_T done = 0; done < bytes; done += page_bytes) {
        Backing one = backing;
        one.offset += done;
        if (!map_view(address + done, one)) {
            failed_address = address + done;
            return false;
        }
        memory.runs[address + done] = {page_bytes, stamp};
        made.emplace_back(address + done, page_bytes);
    }
    return true;
}

bool unmap_view(ULONG_PTR address)
{
    return UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(address),
                           MEM_PRESERVE_PLACEHOLDER) != FALSE;
}

template <class Pages>
void coalesce_unused(const Pages& batch)
{
    auto previous = memory.reservations.end();
    for (const auto& entry : batch) {
        const auto reservation = owned_reservation(entry.first);
        if (reservation == memory.reservations.end() || reservation == previous) continue;
        previous = reservation;
        const auto [base, size] = *reservation;
        const auto view = memory.views.lower_bound(base);
        if (view == memory.views.end() || view->first - base >= size)
            VirtualFree(reinterpret_cast<void*>(base), size, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
    }
}

// Unmapping a page inside a run unmaps the whole run and maps its other pages again (map_pages), and so do a
// cross-call merge (merge_with_previous) for the run it joins and a page batch (remap_alias_pages) for the runs it
// changes. A thread that touches one of those pages in between faults. Such a run break has a generation that is odd
// while it is in progress; the break's id is that odd value. Every view mapped inside a break records the id
// (PhysicalMemory::Run::stamp, under memory.mutex), so a fault handler that runs late, after any number of later
// breaks on other runs (a merge of the CPU view and then of its GPU mirror in one call, or the next calls), still
// finds the break that mapped the page again.
struct RemapWindow {
    SRWLOCK lock = SRWLOCK_INIT;
    ULONG64 generation = 0;
};
RemapWindow remap_window;
// Set once shared views are on (FaultRetry::ready). Breaks also begin without them (every unmap counts in the
// generation), but then a page has a view of its own and no break takes a neighbour away, so x2w_xmem_stable_call
// does not wait for them.
std::atomic<bool> shared_views_on{false};
// Set once, with shared views, when X2W_XMEM_JOIN_CALLS=1 is set as well (FaultRetry::ready): only then does a run
// mapped by a call of its own join the recorded run that ends at its address (merge_with_previous).
std::atomic<bool> joins_on{false};

// Caller holds memory.mutex. Returns the break's id, the stamp of the views mapped until end_break.
ULONG64 begin_break()
{
    AcquireSRWLockExclusive(&remap_window.lock);
    const ULONG64 id = ++remap_window.generation;
    ReleaseSRWLockExclusive(&remap_window.lock);
    return id;
}
void end_break()
{
    AcquireSRWLockExclusive(&remap_window.lock);
    ++remap_window.generation;
    ReleaseSRWLockExclusive(&remap_window.lock);
}

struct EndBreak { ~EndBreak() { end_break(); } };

ULONG64 break_generation()
{
    AcquireSRWLockShared(&remap_window.lock);
    const ULONG64 value = remap_window.generation;
    ReleaseSRWLockShared(&remap_window.lock);
    return value;
}

// Test build only: X2W_XMEM_TEST_HOLD_RETRY=<name> holds the fault handler before it reads the break: it sets
// "<name>.faulted", then waits up to five seconds for "<name>.go", so that a test can end the break first.
#ifdef X2W_XMEM_TEST_CONTROLS
void hold_retry()
{
    char name[96];
    const DWORD length = GetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_RETRY", name, sizeof(name) - 16);
    if (!length || length >= sizeof(name) - 16) return;
    char event_name[128];
    std::snprintf(event_name, sizeof(event_name), "%s.faulted", name);
    HANDLE faulted = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.go", name);
    HANDLE go = CreateEventA(nullptr, TRUE, FALSE, event_name);
    if (faulted && go) {
        SetEvent(faulted);
        WaitForSingleObject(go, 5000);
    }
    if (faulted) CloseHandle(faulted);
    if (go) CloseHandle(go);
}
#else
void hold_retry() {}
#endif

// Retries an access violation only when all of this holds: after the break in progress, if any, has finished (the
// mutex), the page is recorded, covered by a view, and in a run mapped inside a break (a nonzero Run::stamp); and
// this thread has not already retried this page for that stamp. Everything else is passed on, so a fault that is not
// the transient one is reported after at most one retry per break that mapped the page again.
// Faults that cannot be the transient one do not wait for memory.mutex, which a mapping call holds for all of its
// work: when no break is in progress or begins or ends around a region query of the page (the generation is even
// before and unchanged after) and the query shows no mapped view there, the fault is passed on at once. Outside
// breaks only a break takes away a view mapped inside one, so the check below would pass it on as well. Faults on
// lazily committed memory and other faults outside the runs take that path.
LONG CALLBACK retry_remapped_page(EXCEPTION_POINTERS* e)
{
    const EXCEPTION_RECORD* record = e->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || mapping_thread)
        return EXCEPTION_CONTINUE_SEARCH;
    const ULONG_PTR address = record->ExceptionInformation[1];
    const ULONG_PTR page = address & ~(ULONG_PTR{page_bytes} - 1);
    hold_retry();
    const ULONG64 before = break_generation();
    MEMORY_BASIC_INFORMATION info{};
    const bool mapped_now = VirtualQuery(reinterpret_cast<void*>(page), &info, sizeof(info)) == sizeof(info) &&
                            info.State == MEM_COMMIT && info.Type == MEM_MAPPED;
    if (!(before & 1) && break_generation() == before && !mapped_now) return EXCEPTION_CONTINUE_SEARCH;
    // The break that last mapped the page's run, however many breaks on other runs came after it. A later break of
    // the same run stamps it again, so a fault in that break is retried again.
    ULONG64 stamp = 0;
    {
        const std::lock_guard guard(memory.mutex);  // held by the mapping thread until its break is over
        const auto run = memory.runs.upper_bound(page);
        if (memory.views.count(page) && run != memory.runs.begin() &&
            page - std::prev(run)->first < std::prev(run)->second.bytes)
            stamp = std::prev(run)->second.stamp;
    }
    if (!stamp) return EXCEPTION_CONTINUE_SEARCH;
    thread_local ULONG_PTR retried_page = 0;
    thread_local ULONG64 retried_stamp = 0;
    if (retried_page == page && retried_stamp == stamp) return EXCEPTION_CONTINUE_SEARCH;
    retried_page = page;
    retried_stamp = stamp;
    return EXCEPTION_CONTINUE_EXECUTION;
}
struct MappingThread {
    MappingThread() { mapping_thread = true; }
    ~MappingThread() { mapping_thread = false; }
};

// Shared views are opt-in: X2W_XMEM_SHARED_VIEWS=1, read once by the first mapping call. Without
// it every page has a view of its own, as before: unmapping a page never takes a neighbour away,
// no handler is registered, and the module unloads like any DLL. The switch is also read once when
// xmem.dll loads (virtual_query_ex_wrapped): with it set then, the main module's VirtualQueryEx imports
// are wrapped, and the module is pinned at load when one is.
//
// With it, the first mapping call registers the handler and pins the module. The handler is
// process-wide and Windows offers no way to remove one that also waits for every dispatch already
// under way, so it is never removed; the pin keeps xmem.dll, the handler's code and this state in
// place until the process ends, also across a host's FreeLibrary and LoadLibrary. Nothing is
// waited for or torn down at DLL detach. If the handler cannot be registered or the module cannot
// be pinned, the opt-in is refused and pages get a view each.
//
// X2W_XMEM_JOIN_CALLS=1 is read at the same time and only counts with shared views on: a mapping call may then join
// the run that ends at its address (merge_with_previous). Without it every call maps views of its own.
struct FaultRetry {
    SRWLOCK lock = SRWLOCK_INIT;
    bool tried = false;
    bool guaranteed = false;  // handler registered and module pinned: shared views may be created

    bool ready()
    {
        AcquireSRWLockExclusive(&lock);
        if (!tried) {
            tried = true;
#ifdef X2W_XMEM_TEST_CONTROLS
            // Test build only: shared views without the handler, to show that a reader then faults.
            if (x2w_local::env_on("X2W_XMEM_NO_FAULT_RETRY")) {
                guaranteed = true;
                joins_on.store(x2w_local::env_on("X2W_XMEM_JOIN_CALLS"), std::memory_order_release);
                ReleaseSRWLockExclusive(&lock);
                return true;
            }
#endif
            HMODULE pinned = nullptr;
            if (!x2w_local::env_on("X2W_XMEM_SHARED_VIEWS")) {
                // The default: one view per page, nothing process-wide, unloadable.
            } else if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                   reinterpret_cast<LPCWSTR>(&retry_remapped_page), &pinned) &&
                AddVectoredExceptionHandler(1, retry_remapped_page)) {
                guaranteed = true;
                x2w_local::logf("x2w-xmem shared views are on (X2W_XMEM_SHARED_VIEWS): xmem.dll stays loaded until the process ends");
                if (x2w_local::env_on("X2W_XMEM_JOIN_CALLS")) {
                    joins_on.store(true, std::memory_order_release);
                    x2w_local::logf("x2w-xmem joins across mapping calls are on (X2W_XMEM_JOIN_CALLS)");
                }
            } else
                x2w_local::logf("x2w-xmem shared views are off: the fault handler could not be set up (error %lu)", GetLastError());
        }
        const bool result = guaranteed;
        if (result) shared_views_on.store(true, std::memory_order_release);
        ReleaseSRWLockExclusive(&lock);
        return result;
    }
};
FaultRetry fault_retry;

// Test build only (X2W_XMEM_TEST_CONTROLS; tests/xmem_page_runs_check.py): X2W_XMEM_TEST_FAULT=<stage>:<skip> makes one step of
// the next unmap fail as if Windows had refused it, after skipping <skip> such steps. Stage 1 is
// unmapping a run, stage 2 mapping surviving pages again, stage 3 both the unmap and the restore.
// Stages 4 and 5 are steps of a cross-call merge (merge_with_previous): 4 the map of the joined view,
// 5 joining the two placeholders.
#ifdef X2W_XMEM_TEST_CONTROLS
struct TestFault {
    unsigned stage = 0, skip = 0;
    TestFault()
    {
        char text[32];
        if (GetEnvironmentVariableA("X2W_XMEM_TEST_FAULT", text, sizeof(text)) && text[0] >= '1' && text[0] <= '5' && text[1] == ':') {
            stage = static_cast<unsigned>(text[0] - '0');
            skip = static_cast<unsigned>(std::strtoul(text + 2, nullptr, 10));
        }
    }
    bool hit(unsigned at)
    {
        if (stage != at && !(stage == 3 && at == 1)) return false;
        if (skip) {
            --skip;
            return false;
        }
        if (stage != 3) stage = 0;      // one failure
        else if (at == 1) return true;  // stage 3: the unmap now, the restore next
        return true;
    }
};
#else
struct TestFault {
    unsigned stage = 0;
    bool hit(unsigned) { return false; }
};
#endif

// Test build only: X2W_XMEM_TEST_HOLD_WINDOW=<name> keeps an unmap, or a cross-call merge (merge_with_previous), in the
// window where the run is unmapped and not yet mapped again. It sets the event "<name>.unmapped", then waits up to
// five seconds for "<name>.release", so that a reader thread can reach the window every time.
#ifdef X2W_XMEM_TEST_CONTROLS
void hold_window()
{
    char name[96];
    const DWORD length = GetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", name, sizeof(name) - 16);
    if (!length || length >= sizeof(name) - 16) return;
    char event_name[128];
    std::snprintf(event_name, sizeof(event_name), "%s.unmapped", name);
    HANDLE unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.release", name);
    HANDLE release = CreateEventA(nullptr, TRUE, FALSE, event_name);
    if (unmapped && release) {
        SetEvent(unmapped);
        if (WaitForSingleObject(release, 5000) != WAIT_OBJECT_0)
            x2w_local::logf("x2w-xmem test hold: nobody released the unmapped window within 5 s");
    }
    if (unmapped) CloseHandle(unmapped);
    if (release) CloseHandle(release);
}
#else
void hold_window() {}
#endif

// Drops a page that has no view any more from the registry, so that the registry never names a
// page Windows does not show.
void forget_page(ULONG_PTR address)
{
    const auto live = memory.views.find(address);
    if (live == memory.views.end()) return;
    --memory.backings.at(live->second).views;
    memory.views.erase(live);
}

// Maps every recorded page of [base, base + size) that has no view: neighbours on adjacent slots
// together, one page at a time when a group is refused. A page that cannot be mapped is forgotten
// and counted. fail is true when the test control refuses this range. stamp: the break in progress.
struct MapFailure {
    const char* stage = nullptr;
    ULONG_PTR address = 0;
    ULONG_PTR page = 0;
    DWORD error = ERROR_SUCCESS;
    MEMORY_BASIC_INFORMATION region{};
    SIZE_T view_bytes = page_bytes;

    void record(const char* step, ULONG_PTR at, ULONG_PTR id, DWORD code, SIZE_T bytes = page_bytes) noexcept
    {
        if (stage) return;
        stage = step;
        address = at;
        page = id;
        error = code;
        view_bytes = bytes;
        VirtualQuery(reinterpret_cast<void*>(at), &region, sizeof(region));
        SetLastError(code);
    }
};

ULONG_PTR map_recorded(ULONG_PTR base, SIZE_T size, ULONG64 stamp, bool fail, MapFailure* failure)
{
    ULONG_PTR lost = 0;
    ULONG_PTR at = base;
    while (at < base + size) {
        const auto live = memory.views.find(at);
        if (live == memory.views.end()) {
            at += page_bytes;
            continue;
        }
        const Backing head = memory.backings.at(live->second);
        SIZE_T bytes = page_bytes;
        for (; at + bytes < base + size; bytes += page_bytes) {
            const auto next = memory.views.find(at + bytes);
            if (next == memory.views.end()) break;
            const Backing& b = memory.backings.at(next->second);
            if (b.section != head.section || b.offset != head.offset + bytes) break;
        }
        std::vector<std::pair<ULONG_PTR, SIZE_T>> made;
        ULONG_PTR failed_address;
        if (fail || !map_run(at, head, bytes, stamp, made, failed_address)) {
            // map_run may have mapped some pages one at a time before it stopped; keep those.
            for (SIZE_T done = 0; done < bytes; done += page_bytes) {
                if (memory.runs.count(at + done)) continue;
                Backing one = head;
                one.offset += done;
                if (!fail && map_view(at + done, one)) {
                    memory.runs[at + done] = {page_bytes, stamp};
                } else {
                    if (failure)
                        failure->record("restore-view", at + done, memory.views.at(at + done),
                                        fail ? ERROR_INVALID_ADDRESS : GetLastError());
                    forget_page(at + done);
                    ++lost;
                }
            }
        }
        at += bytes;
    }
    return lost;
}

// Shared views with X2W_XMEM_JOIN_CALLS=1 only (joins_on). Startup maps nearly every page with a call of its own,
// and under Wine each map costs more the more views the process has. With the switch, a run mapped by a call of its
// own therefore joins the recorded run R that ends at its address, when its first page is the next slot of the
// section behind R's last page and R and the new pages lie in one reservation of the shim: the new pages' range is
// split to a placeholder of its own, R is unmapped, the two placeholders are joined and one view is mapped over
// both. R is taken apart as in an unmap (begin_break), so a thread that touches R in between is retried and
// x2w_xmem_stable_call readers wait. On a refused step before R is unmapped nothing changed; after it, R is mapped
// again as it was (map_recorded). Both return separate, and the caller maps the new pages on their own. lost: R
// could not be mapped again; the pages that have no view are forgotten, as in an unmap. X2W_XMEM_TEST_NO_MERGE=1
// turns the merge off (tests only).
// A joined run stops at max_merged_run_pages: a page that would make R longer starts a run of its own, which the
// next pages join. The cap bounds the runs that joining builds: the work of each join, and of a later break of such a
// run (an unmap inside a run takes the whole run apart and maps its other pages again). A run mapped by one call
// (map_run in map_pages_locked) is not capped, and neither are the survivors map_recorded maps again after an unmap
// inside such a run; both are as long as they were before the cap.
enum class Merge { joined, separate, lost };

constexpr SIZE_T max_merged_run_pages = 64;

// Test build only: X2W_XMEM_TEST_MERGE_CAP=<pages> replaces the cap (xmem_cross_call_merge_check's timing mode).
SIZE_T merged_run_cap()
{
#ifdef X2W_XMEM_TEST_CONTROLS
    char text[24];
    if (GetEnvironmentVariableA("X2W_XMEM_TEST_MERGE_CAP", text, sizeof(text)))
        if (const auto pages = std::strtoull(text, nullptr, 10)) return static_cast<SIZE_T>(pages);
#endif
    return max_merged_run_pages;
}

Merge merge_with_previous(ULONG_PTR address, const Backing& head, SIZE_T bytes, MapFailure* failure)
{
#ifdef X2W_XMEM_TEST_CONTROLS
    if (x2w_local::env_on("X2W_XMEM_TEST_NO_MERGE")) return Merge::separate;
#endif
    const auto next = memory.runs.lower_bound(address);
    if (next == memory.runs.begin()) return Merge::separate;
    const ULONG_PTR base = std::prev(next)->first;
    const SIZE_T size = std::prev(next)->second.bytes;
    if (base + size != address) return Merge::separate;
    if ((size + bytes) / page_bytes > merged_run_cap()) return Merge::separate;
    const auto tail = memory.views.find(address - page_bytes);
    if (tail == memory.views.end()) return Merge::separate;
    const Backing& previous = memory.backings.at(tail->second);
    if (previous.section != head.section || previous.offset + page_bytes != head.offset) return Merge::separate;
    const auto reservation = owned_reservation(base);
    if (reservation == memory.reservations.end() || address + bytes > reservation->first + reservation->second)
        return Merge::separate;
    const Backing start = memory.backings.at(memory.views.at(base));
    auto* const target = reinterpret_cast<void*>(address);
    if (!exact_placeholder(target, bytes)) {
        InterlockedIncrement64(&split_attempts);
        if (!VirtualFree(target, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) return Merge::separate;
    }
    TestFault test;
    const ULONG64 stamp = begin_break();
    const EndBreak end;
    if (!unmap_view(base)) return Merge::separate;
    memory.runs.erase(base);
    hold_window();
    auto* const whole = reinterpret_cast<void*>(base);
    if (!test.hit(5) && VirtualFree(whole, size + bytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
        InterlockedIncrement64(&map_attempts);
        if (!test.hit(4) && MapViewOfFile3(start.section, GetCurrentProcess(), whole, start.offset, size + bytes,
                                           MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0)) {
            memory.runs[base] = {size + bytes, stamp};
            return Merge::joined;
        }
    }
    // map_recorded splits the joined placeholder back to R's size (map_split: the reservation is the shim's).
    const ULONG_PTR lost = map_recorded(base, size, stamp, false, failure);
    if (!lost) return Merge::separate;
    x2w_local::logf("x2w-xmem merge of a run with the next mapping call failed: %llu page(s) of the run could not be "
                    "mapped again and were forgotten", static_cast<unsigned long long>(lost));
    return Merge::lost;
}

// Destinations must be caller-owned, page-sized Windows placeholders. Live views
// are owned by this shim until it unmaps them. The caller must not replace them.
template <typename AddressAt>
BOOL map_pages_locked(ULONG_PTR count, const ULONG_PTR* tokens, const AddressAt& address_at,
                      bool share_views, MapFailure* failure = nullptr) noexcept
{
    if (!count || count > era_physical::capacity_pages)
        return physical_result(era_physical::Status::invalid);
    try {
        std::map<ULONG_PTR, ULONG_PTR> batch;
        for (ULONG_PTR i = 0; i < count; ++i) {
            const auto address = address_at(i);
            const auto live = memory.views.find(address);
            const ULONG_PTR page = tokens ? tokens[i] : (live != memory.views.end() ? live->second : 0);
            const char* invalid = nullptr;
            if (!address || address % page_bytes) invalid = "invalid-address";
            else if (tokens && live != memory.views.end()) invalid = "already-mapped";
            else if (tokens && !era_physical::owns(page)) invalid = "unowned-page";
            else if (!tokens && live == memory.views.end()) invalid = "unmapped-address";
            else if (!batch.emplace(address, page).second) invalid = "duplicate-address";
            if (invalid) {
                if (failure) failure->record(invalid, address, page, ERROR_INVALID_PARAMETER);
                return physical_result(era_physical::Status::invalid);
            }
        }
        // Adjacent pages that are adjacent slots of one section share a single view (a "run"): a
        // multi-gigabyte batch then takes a few views instead of one per 64 KiB page.
        if (tokens) {
            for (const auto& [address, page] : batch) {
                if (!section_for(page, false)) {  // committed on first touch with X2W_XMEM_LAZY_COMMIT
                    if (failure) failure->record("section-backing", address, page, GetLastError());
                    return FALSE;
                }
            }
            std::vector<std::pair<ULONG_PTR, SIZE_T>> made;
            auto first = batch.begin();
            while (first != batch.end()) {
                const Backing& head = memory.backings.at(first->second);
                SIZE_T bytes = page_bytes;
                auto last = std::next(first);
                for (; share_views && last != batch.end(); ++last, bytes += page_bytes) {
                    const Backing& b = memory.backings.at(last->second);
                    if (last->first != first->first + bytes || b.section != head.section || b.offset != head.offset + bytes) break;
                }
                // Only with X2W_XMEM_JOIN_CALLS, and only a run that is the whole batch joins an earlier one: a
                // later run of the batch that fails unmaps everything in made, which a joined run is not.
                const bool whole_batch = first == batch.begin() && last == batch.end();
                const Merge merge = share_views && joins_on.load(std::memory_order_acquire) && whole_batch
                                        ? merge_with_previous(first->first, head, bytes, failure)
                                        : Merge::separate;
                if (merge == Merge::lost) {
                    coalesce_unused(batch);
                    SetLastError(ERROR_NOT_ENOUGH_MEMORY);
                    return FALSE;
                }
                ULONG_PTR failed_address;
                if (merge == Merge::separate && !map_run(first->first, head, bytes, 0, made, failed_address)) {
                    const DWORD error = GetLastError();
                    if (failure) failure->record("map-view", failed_address, batch.at(failed_address), error);
                    for (const auto& [base, size] : made) {
                        UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(base), MEM_PRESERVE_PLACEHOLDER);
                        memory.runs.erase(base);
                    }
                    coalesce_unused(batch);
                    SetLastError(error);
                    return FALSE;
                }
                first = last;
            }
            for (const auto& [address, page] : batch)
                ++memory.backings.at(page).views;
            memory.views.merge(batch);
        } else {
            // Unmap every run that holds a page of the batch, then map the surviving pages of those
            // runs again. The registry (views, runs) matches what Windows shows on every way out:
            // - an unmap is refused: the runs already unmapped are mapped again whole, nothing is
            //   removed, and the call fails with that error;
            // - a surviving page cannot be mapped again: it is forgotten, the batch is unmapped, and
            //   the call fails.
            std::map<ULONG_PTR, SIZE_T> broken;
            for (const auto& [address, page] : batch) {
                auto run = memory.runs.upper_bound(address);
                if (run == memory.runs.begin()) continue;
                --run;
                if (address - run->first < run->second.bytes) broken.emplace(run->first, run->second.bytes);
            }
            TestFault test;
            const ULONG64 stamp = begin_break();
            const EndBreak end;
            DWORD error = ERROR_SUCCESS;
            std::vector<std::pair<ULONG_PTR, SIZE_T>> unmapped;
            for (const auto& [base, size] : broken) {
                const bool inject_failure = test.hit(1);
                if (inject_failure || !unmap_view(base)) {
                    error = inject_failure ? ERROR_INVALID_ADDRESS : GetLastError();
                    if (failure) failure->record("unmap-view", base, memory.views.at(base), error, size);
                    break;
                }
                memory.runs.erase(base);
                unmapped.emplace_back(base, size);
            }
            const bool refused = error != ERROR_SUCCESS;
            if (!refused) {
                for (const auto& [address, page] : batch) {
                    --memory.backings.at(page).views;
                    memory.views.erase(address);
                }
            }
            if (!refused && !unmapped.empty()) hold_window();
            ULONG_PTR lost = 0;
            for (const auto& [base, size] : unmapped)
                lost += map_recorded(base, size, stamp, refused ? test.stage == 3 && test.hit(3) : test.hit(2), failure);
            if (!refused) coalesce_unused(batch);
            if (refused || lost) {
                x2w_local::logf("x2w-xmem unmap of %llu page(s) failed (error %lu): %s; %llu surviving page(s) could not be mapped "
                                "again and were forgotten", static_cast<unsigned long long>(batch.size()),
                                static_cast<unsigned long>(error), refused ? "an unmap was refused and no page of the batch was removed" : "the pages were unmapped",
                                static_cast<unsigned long long>(lost));
                SetLastError(refused ? error : ERROR_NOT_ENOUGH_MEMORY);
                return FALSE;
            }
        }
    } catch (const std::bad_alloc&) {
        return physical_result(era_physical::Status::exhausted);
    }
    return physical_result(era_physical::Status::ok);
}

template <typename AddressAt>
BOOL map_pages(ULONG_PTR count, const ULONG_PTR* tokens, const AddressAt& address_at,
               MapFailure* failure = nullptr) noexcept
{
    X2W_ZONE("xmem map_pages");
    const bool share_views = fault_retry.ready();
    const MappingThread marker;
    const std::lock_guard guard(memory.mutex);
    return map_pages_locked(count, tokens, address_at, share_views, failure);
}
}

// ---------------------------------------------------------------------------------------------
// Experimental (docs/startup-runtime.md). With X2W_XMEM_PERMISSIVE=1 the exports
// below accept the call shapes of a startup run on top of the section-backed pages above.
// Observations behind it:
// - Page numbers: the caller rebuilds them as (21-bit entry << 4), so it expects 4 KiB frame numbers
//   with the low four bits clear. Callers get page << 4 and pass it back.
// - XMemAllocatePhysicalPages flags other than MEM_64K_PAGES occur (0x280) and are served the same.
// - XMemVirtualAlloc: commits (type 0x20003000/0x20403000) become plain commits; reservations become
//   placeholders that pages can be mapped into; a "replace placeholder" call (0x20406000) on a piece
//   the caller split from its own placeholder keeps the placeholder and records it for mapping.
// - Scatter mapping passes page 0 to unmap and may name pages that are already mapped elsewhere
//   (a second view; the sections give real sharing).
namespace {
bool local_permissive() noexcept { return x2w_local::env_on("X2W_XMEM_PERMISSIVE"); }

void record_reservation(void* base, SIZE_T bytes)
{
    const std::lock_guard guard(memory.mutex);
    memory.reservations.insert_or_assign(reinterpret_cast<ULONG_PTR>(base), bytes);
}

void* local_virtual_alloc(void* address, SIZE_T bytes, DWORD type, ULONGLONG attributes, DWORD protection)
{
    const DWORD low = protection & 0xFF;
    const bool accessible = low == PAGE_READONLY || low == PAGE_READWRITE || low == PAGE_EXECUTE_READ ||
                            low == PAGE_EXECUTE_READWRITE;
    const DWORD protect = accessible ? low : PAGE_READWRITE;
    void* result = nullptr;
    const char* how = "";
    if ((type & MEM_REPLACE_PLACEHOLDER) && address) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(address, &info, sizeof(info)) && info.State == MEM_RESERVE) {
            record_reservation(address, bytes);
            result = address;
        }
        how = "placeholder kept for page mapping";
    } else if (type & MEM_COMMIT) {
        if (!address) {
            result = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, protect);
            how = "commit";
        } else {
            result = VirtualAlloc(address, bytes, MEM_COMMIT, protect);
            if (!result) {
                VirtualFree(address, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                result = VirtualAlloc2(GetCurrentProcess(), address, bytes,
                                       MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER, protect, nullptr, 0);
            }
            how = "commit at address";
        }
    } else {
        result = VirtualAlloc2(GetCurrentProcess(), address, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
                               nullptr, 0);
        // A caller that asks for a placeholder splits it itself; other reservations are split here on mapping.
        if (result && !(type & MEM_RESERVE_PLACEHOLDER)) record_reservation(result, bytes);
        how = "placeholder reserve";
    }
#if X2W_TRACY
    if (result && (type & MEM_COMMIT)) {
        // A plot, not a pool: the title releases these ranges with Win32 calls that are not seen here.
        static std::atomic<long long> committed;
        X2W_PLOT("XMemVirtualAlloc commits (bytes, cumulative)", static_cast<std::int64_t>(committed += bytes));
    }
#endif
    const DWORD error = result ? 0 : GetLastError();
    x2w_local::logf("x2w-xmem XMemVirtualAlloc addr=%p size=0x%llx type=0x%lx attr=0x%llx prot=0x%lx -> %p (%s, err %lu)",
                    address, static_cast<unsigned long long>(bytes), static_cast<unsigned long>(type),
                    static_cast<unsigned long long>(attributes), static_cast<unsigned long>(protection), result, how,
                    static_cast<unsigned long>(error));
    SetLastError(error);
    return result;
}

void log_apply_failure(const char* what, const MapFailure& failure, ULONG_PTR count, const ULONG_PTR* addresses)
{
    // The apply batch is sorted and the last occurrence of each address wins. Report the caller's
    // index, including when the failing OS unmap covers a whole run containing a requested page.
    ULONG_PTR index = count;
    for (ULONG_PTR i = 0; i < count; ++i) {
        if (addresses[i] == failure.address) index = i;
    }
    if (index == count && failure.view_bytes > page_bytes) {
        for (ULONG_PTR i = 0; i < count; ++i) {
            if (addresses[i] - failure.address < failure.view_bytes) {
                index = i;
                break;
            }
        }
    }
    char request_index[24] = "none";
    if (index != count)
        std::snprintf(request_index, sizeof(request_index), "%llu", static_cast<unsigned long long>(index));
    x2w_local::logf("x2w-xmem %s failed stage=%s index=%s address=%p page=0x%llx error=%lu "
                    "region_base=%p allocation_base=%p region_size=0x%llx state=0x%lx protect=0x%lx type=0x%lx",
                    what, failure.stage, request_index, reinterpret_cast<void*>(failure.address),
                    static_cast<unsigned long long>(failure.page), static_cast<unsigned long>(failure.error),
                    failure.region.BaseAddress, failure.region.AllocationBase,
                    static_cast<unsigned long long>(failure.region.RegionSize),
                    static_cast<unsigned long>(failure.region.State), static_cast<unsigned long>(failure.region.Protect),
                    static_cast<unsigned long>(failure.region.Type));
    // A failed commit (1455 = system commit limit, 8 = not enough memory): say how full the system commit and the pool
    // are. The caller holds memory.mutex. The first three failures are reported.
    if (x2w_local::is_commit_failure(failure.error)) {
        static volatile LONG reported = 0;
        if (InterlockedIncrement(&reported) <= 3) {
            char line[384];
            x2w_local::format_exhausted(line, sizeof(line), what, static_cast<unsigned long>(failure.error),
                                        x2w_local::system_commit(), x2w_local::job_memory(), memory.backings.size(),
                                        memory.slots.slab_count(), memory.slots.slots_in_use(), memory.slots.slots_free());
            x2w_local::logf("%s", line);
        }
    }
}

void log_failed_ranges(const char* what, const char* operation, const std::vector<ULONG_PTR>& addresses)
{
    // Addresses are sorted by local_apply. A failed map rolls back its completed views, so all
    // of its destinations matter when correlating a later fault, even those preceding the error.
    constexpr std::size_t max_ranges = 2048;
    std::size_t first = 0;
    for (std::size_t range = 0; first < addresses.size() && range < max_ranges; ++range) {
        std::size_t end = first + 1;
        while (end < addresses.size() && addresses[end] - addresses[end - 1] == page_bytes) ++end;
        x2w_local::logf("x2w-xmem %s failed batch %s first=%p last=%p pages=%llu", what, operation,
                        reinterpret_cast<void*>(addresses[first]), reinterpret_cast<void*>(addresses[end - 1]),
                        static_cast<unsigned long long>(end - first));
        first = end;
    }
    if (first < addresses.size())
        x2w_local::logf("x2w-xmem %s failed batch %s truncated omitted_pages=%llu", what, operation,
                        static_cast<unsigned long long>(addresses.size() - first));
}

// Applies (address, page) pairs; page 0 unmaps. An address that already shows the requested page is
// left alone; one that shows another page is unmapped first.
BOOL local_apply_locked(ULONG_PTR count, const ULONG_PTR* addresses, const ULONG_PTR* pages,
                        const char* what, bool share_views)
{
    std::map<ULONG_PTR, ULONG_PTR> want;
    for (ULONG_PTR i = 0; i < count; ++i) want[addresses[i]] = pages[i];
    std::vector<ULONG_PTR> unmap, map_at, map_page;
    for (const auto& [address, page] : want) {
        const auto live = memory.views.find(address);
        if (!page) {
            if (live != memory.views.end()) unmap.push_back(address);
            continue;
        }
        if (live != memory.views.end()) {
            if (live->second == page) continue;
            unmap.push_back(address);
        }
        map_at.push_back(address);
        map_page.push_back(page);
    }
    BOOL ok = TRUE;
    DWORD error = 0;
    MapFailure failure;
    if (!unmap.empty() && !map_pages_locked(unmap.size(), nullptr, [&](ULONG_PTR i) { return unmap[i]; },
                                            share_views, &failure)) {
        ok = FALSE;
        error = GetLastError();
    }
    if (ok && !map_at.empty() &&
        !map_pages_locked(map_at.size(), map_page.data(), [&](ULONG_PTR i) { return map_at[i]; },
                          share_views, &failure)) {
        ok = FALSE;
        error = GetLastError();
    }
    if (failure.stage) log_apply_failure(what, failure, count, addresses);
    if (!ok && x2w_local::env_on("X2W_XMEM_TRACE_FAILED_BATCH")) {
        log_failed_ranges(what, "map", map_at);
        log_failed_ranges(what, "unmap", unmap);
    }
    static LONG calls;
    if (!ok || x2w_local::first_calls(calls, 16))
        x2w_local::logf("x2w-xmem %s count=0x%llx first=%p map=%llu unmap=%llu -> %d (err %lu)", what,
                        static_cast<unsigned long long>(count), count ? reinterpret_cast<void*>(addresses[0]) : nullptr,
                        static_cast<unsigned long long>(map_at.size()), static_cast<unsigned long long>(unmap.size()), ok,
                        static_cast<unsigned long>(error));
    SetLastError(error);
    return ok;
}

// Applies the same (address, page) pairs again at the GPU view for those addresses in the CPU view.
void local_mirror_locked(ULONG_PTR count, const ULONG_PTR* addresses, const ULONG_PTR* pages,
                         bool share_views)
{
    const auto cpu = memory.mirror_cpu, gpu = memory.mirror_gpu;
    const auto bytes = memory.mirror_bytes;
    if (!cpu) return;
    std::vector<ULONG_PTR> at, page;
    for (ULONG_PTR i = 0; i < count; ++i)
        if (addresses[i] >= cpu && addresses[i] - cpu < bytes) {
            const auto address = addresses[i] - cpu + gpu;
            if (memory.explicit_mirror_pages.count(address)) continue;
            at.push_back(address);
            page.push_back(pages[i]);
        }
    if (!at.empty()) local_apply_locked(at.size(), at.data(), page.data(), "GPU view mirror", share_views);
}

BOOL local_apply(ULONG_PTR count, const ULONG_PTR* addresses, const ULONG_PTR* pages, const char* what)
{
    const bool share_views = fault_retry.ready();
    const MappingThread marker;
    const std::lock_guard guard(memory.mutex);
    if (!local_apply_locked(count, addresses, pages, what, share_views)) return FALSE;
    local_mirror_locked(count, addresses, pages, share_views);
    return TRUE;
}

bool to_pages(ULONG_PTR count, const ULONG_PTR* tokens, std::vector<ULONG_PTR>& pages)
{
    if (count > era_physical::capacity_pages) return false;  // as the strict path: no grant is larger
    pages.resize(count);
    for (ULONG_PTR i = 0; i < count; ++i) {
        if (tokens[i] & 0xF) return false;
        pages[i] = tokens[i] >> 4;
    }
    return true;
}
}  // namespace

namespace {
BOOL free_pages(ULONG_PTR count, PULONG_PTR pages);
}

extern "C" BOOL WINAPI XMemAllocatePhysicalPages(ULONG_PTR flags, PULONG_PTR count, PULONG_PTR pages)
{
    static_assert(sizeof(ULONG_PTR) == sizeof(std::uint64_t));
    if (local_permissive()) {
        const ULONG_PTR requested = count ? *count : 0;
        era_physical::Status status;
        {
            const std::lock_guard guard(memory.mutex);
            status = era_physical::allocate(era_physical::mem_64k_pages, reinterpret_cast<std::uint64_t*>(count),
                                            reinterpret_cast<std::uint64_t*>(pages));
        }
        if (status == era_physical::Status::ok)
            for (ULONG_PTR i = 0; i < *count; ++i) pages[i] <<= 4;
        x2w_local::logf("x2w-xmem XMemAllocatePhysicalPages flags=0x%llx pages=0x%llx -> %d",
                        static_cast<unsigned long long>(flags), static_cast<unsigned long long>(requested),
                        status == era_physical::Status::ok);
        return physical_result(status);
    }
    const std::lock_guard guard(memory.mutex);
    return physical_result(era_physical::allocate(flags, reinterpret_cast<std::uint64_t*>(count),
                                                  reinterpret_cast<std::uint64_t*>(pages)));
}

extern "C" BOOL WINAPI XMemFreePhysicalPages(ULONG_PTR count, PULONG_PTR pages)
{
    if (local_permissive() && pages && count && count <= era_physical::capacity_pages) {
        std::vector<ULONG_PTR> converted;
        if (!to_pages(count, pages, converted)) return physical_result(era_physical::Status::invalid);
        return free_pages(count, converted.data());
    }
    return free_pages(count, pages);
}

namespace {
BOOL free_pages(ULONG_PTR count, PULONG_PTR pages)
{
    if (!pages || !count || count > era_physical::capacity_pages)
        return physical_result(era_physical::Status::invalid);
    std::vector<HANDLE> closed;
    era_physical::Status status;
    {
        const std::lock_guard guard(memory.mutex);
        for (ULONG_PTR i = 0; i < count; ++i) {
            const auto backing = memory.backings.find(pages[i]);
            if (backing != memory.backings.end() && backing->second.views)
                return physical_result(era_physical::Status::invalid);
        }
        status = era_physical::release(count, reinterpret_cast<const std::uint64_t*>(pages));
        if (status == era_physical::Status::ok) {
            for (ULONG_PTR i = 0; i < count; ++i) {
                const auto backing = memory.backings.find(pages[i]);
                if (backing != memory.backings.end()) {
                    const Slot slot{backing->second.slab, backing->second.offset};
                    if (const auto pin = memory.pins.find(slot); pin != memory.pins.end())
                        pin->second.retired = true;
                    else
                        memory.slots.release(slot.first, slot.second);
                    memory.backings.erase(backing);
                }
            }
            collect_empty_slabs(closed);
        }
    }
    for (const HANDLE slab : closed) CloseHandle(slab);
    return physical_result(status);
}
}  // namespace

namespace {
struct AliasRange {
    ULONG_PTR address;
    X2WPageRange source;
};

// Keep the guarded reads outside the transaction and out of C++ functions that need unwinding.
template <typename T>
bool read_alias_record(const T* array, UINT index, T& record)
{
    __try {
        record = array[index];
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ||
                GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

struct AliasEdit {
    ULONG_PTR old;
    ULONG_PTR desired;
    ULONG_PTR actual;
};

using AliasPlan = std::map<ULONG_PTR, AliasEdit>;
using ViewRuns = std::map<ULONG_PTR, PhysicalMemory::Run>;

// The region query exposes the placeholder reservation flag in Flags even though the SDK does not name that bit.
constexpr ULONG placeholder_reservation = 0x100;

// Measured once: whether the region query reports placeholder_reservation for a fresh placeholder, and, when it does
// not, whether VirtualFree(MEM_PRESERVE_PLACEHOLDER) refuses a plain reservation and leaves it whole. Windows reports
// the bit. Wine (Proton Experimental, 2026-11) leaves Flags 0 for placeholders but refuses the plain case with
// ERROR_INVALID_ADDRESS. Test build only (X2W_XMEM_TEST_CONTROLS): X2W_XMEM_TEST_NO_PLACEHOLDER_FALLBACK=1 skips the
// second probe; the deployed build ignores it.
bool placeholder_fallback()
{
    static const bool fallback = [] {
#ifdef X2W_XMEM_TEST_CONTROLS
        const bool skip_probe = x2w_local::env_on("X2W_XMEM_TEST_NO_PLACEHOLDER_FALLBACK");
#else
        constexpr bool skip_probe = false;
#endif
        bool bit_reported = false, refuses_plain = false;
        if (void* probe = VirtualAlloc2(GetCurrentProcess(), nullptr, 2 * page_bytes,
                                        MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0)) {
            WIN32_MEMORY_REGION_INFORMATION info{};
            bit_reported = QueryVirtualMemoryInformation(GetCurrentProcess(), probe, MemoryRegionInfo, &info,
                                                         sizeof(info), nullptr) &&
                           (info.Flags & placeholder_reservation);
            VirtualFree(probe, 0, MEM_RELEASE);
        }
        if (!bit_reported && !skip_probe) {
            if (auto* plain = static_cast<BYTE*>(VirtualAlloc(nullptr, 2 * page_bytes, MEM_RESERVE, PAGE_NOACCESS))) {
                MEMORY_BASIC_INFORMATION after{};
                refuses_plain = !VirtualFree(plain + page_bytes, page_bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) &&
                                VirtualQuery(plain, &after, sizeof(after)) == sizeof(after) &&
                                after.State == MEM_RESERVE && after.AllocationBase == plain &&
                                after.RegionSize == 2 * page_bytes;
                VirtualFree(plain, 0, MEM_RELEASE);
            }
        }
        x2w_local::logf("x2w-xmem placeholder bit reported=%d, plain reservation kept by a placeholder release=%d",
                        bit_reported, refuses_plain);
        return !bit_reported && refuses_plain;
    }();
    return fallback;
}

bool alias_placeholders(ULONG_PTR address, SIZE_T bytes)
{
    // VirtualQuery alone cannot distinguish a reserved private allocation from a placeholder, and on Windows
    // PRESERVE_PLACEHOLDER can silently release that allocation. Where the region query does not report the bit
    // and the OS refuses that release for a reservation made by VirtualAlloc (placeholder_fallback), reserved
    // regions may be offered. Every region the view covers must be reserved, not only the first one of each
    // allocation: Wine (Proton Experimental, 2026-11) released a private allocation made in place of a placeholder
    // (VirtualAlloc2 MEM_REPLACE_PLACEHOLDER) together with its committed second page, and the view then replaced
    // that page (xmem_placeholder_split_partial). Limit: Wine does not refuse the release for that kind of
    // allocation either, so its reserved pages are still offered, and releasing one cuts it out of the caller's
    // allocation.
    const bool fallback = placeholder_fallback();
    const auto end = address + bytes;
    while (address < end) {
        WIN32_MEMORY_REGION_INFORMATION info{};
        if (!QueryVirtualMemoryInformation(GetCurrentProcess(), reinterpret_cast<void*>(address),
                                           MemoryRegionInfo, &info, sizeof(info), nullptr))
            return false;
        const auto base = reinterpret_cast<ULONG_PTR>(info.AllocationBase);
        if (address - base >= info.RegionSize) return false;
        const auto allocation_end = base + info.RegionSize;
        if (!(info.Flags & placeholder_reservation)) {
            if (!fallback) return false;
            const auto stop = end < allocation_end ? end : allocation_end;
            for (auto at = address; at < stop;) {
                MEMORY_BASIC_INFORMATION region{};
                if (VirtualQuery(reinterpret_cast<void*>(at), &region, sizeof(region)) != sizeof(region) ||
                    region.State != MEM_RESERVE || !region.RegionSize)
                    return false;
                at = reinterpret_cast<ULONG_PTR>(region.BaseAddress) + region.RegionSize;
            }
        }
        address = allocation_end;
    }
    return true;
}

// Both maps have the same node type. The node for each possible run base was reserved before
// unmapping anything, so recording a successful OS call cannot fail for lack of C++ heap memory.
// stamp: the page batch's break (PhysicalMemory::Run).
bool install_alias_view(ULONG_PTR address, const Backing& backing, SIZE_T bytes, ULONG64 stamp,
                        ViewRuns& nodes, X2WPageBatchStats& stats)
{
    auto* target = reinterpret_cast<void*>(address);
    auto map = [&] {
        return MapViewOfFile3(backing.section, GetCurrentProcess(), target, backing.offset, bytes,
                              MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0) != nullptr;
    };
    if (!map()) {
        const DWORD original_error = GetLastError();
        const auto reservation = owned_reservation(address);
        const bool inside = reservation != memory.reservations.end() &&
                            bytes <= reservation->second - (address - reservation->first);
        const bool accepted = inside && alias_placeholders(address, bytes);
#ifdef X2W_XMEM_TEST_CONTROLS
        if (inside) InterlockedIncrement64(accepted ? &alias_splits : &alias_refusals);
#endif
        if (!accepted) {
            SetLastError(original_error);
            return false;
        }
        // A subrange of one placeholder must be split. Several exact placeholders left by old
        // views may be coalesced instead. Both operations stay within the recorded reservation.
        if (!VirtualFree(target, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) &&
            (bytes == page_bytes || !VirtualFree(target, bytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS))) {
            SetLastError(original_error);
            return false;
        }
        if (!map()) return false;
    }
    auto node = nodes.extract(address);
    node.mapped() = {bytes, stamp};
    memory.runs.insert(std::move(node));
    ++stats.mapCalls;
    return true;
}

DWORD install_alias_plan(AliasPlan& plan, ViewRuns& nodes, bool share_views, bool restoring, ULONG64 stamp,
                         std::vector<ULONG_PTR>& made, X2WPageBatchStats& stats)
{
    DWORD error = ERROR_SUCCESS;
    auto token = [&](const AliasEdit& edit) { return restoring ? edit.old : edit.desired; };
    auto it = plan.begin();
    while (it != plan.end()) {
        if (it->second.actual || !token(it->second)) {
            ++it;
            continue;
        }
        const auto address = it->first;
        const auto& head = memory.backings.at(token(it->second));
        auto end = std::next(it);
        SIZE_T bytes = page_bytes;
        for (; share_views && end != plan.end(); ++end, bytes += page_bytes) {
            if (end->first != address + bytes || end->second.actual || !token(end->second)) break;
            const auto& next = memory.backings.at(token(end->second));
            if (next.section != head.section || next.offset != head.offset + bytes) break;
        }
        if (install_alias_view(address, head, bytes, stamp, nodes, stats)) {
            made.push_back(address);
            for (auto page = it; page != end; ++page)
                page->second.actual = token(page->second);
        } else if (bytes == page_bytes) {
            error = GetLastError();
            if (!restoring) break;
        } else {
            // A grouped view may cross placeholder boundaries. Retry each page once.
            for (auto page = it; page != end; ++page) {
                if (install_alias_view(page->first, memory.backings.at(token(page->second)),
                                       page_bytes, stamp, nodes, stats)) {
                    made.push_back(page->first);
                    page->second.actual = token(page->second);
                } else {
                    error = GetLastError();
                    if (!restoring) return error;
                }
            }
        }
        it = end;
    }
    return error;
}

HRESULT remap_alias_pages(AliasPlan& plan, bool share_views, X2WPageBatchStats& stats)
{
    std::map<ULONG_PTR, SIZE_T> broken;
    for (const auto& [address, edit] : plan) {
        if (!edit.old) continue;
        const auto run = memory.runs.upper_bound(address);
        if (run == memory.runs.begin() || address - std::prev(run)->first >= std::prev(run)->second.bytes)
            return E_UNEXPECTED;
        broken.emplace(std::prev(run)->first, std::prev(run)->second.bytes);
    }
    for (const auto& [base, size] : broken) {
        // Surviving neighbours of a shared view are part of the transaction as well.
        for (auto view = memory.views.lower_bound(base);
             view != memory.views.end() && view->first - base < size; ++view) {
            const auto page = view->second;
            plan.try_emplace(view->first, AliasEdit{page, page, page});
        }
    }

    ViewRuns run_nodes;
    std::map<ULONG_PTR, ULONG_PTR> view_nodes;
    std::vector<ULONG_PTR> made;
    made.reserve(plan.size());
    for (const auto& [address, edit] : plan) {
        run_nodes.emplace(address, PhysicalMemory::Run{page_bytes, 0});
        view_nodes.emplace(address, 0);
    }
    auto clear_run = [&](ULONG_PTR address, SIZE_T size) {
        ++stats.unmapCalls;
        for (auto it = plan.lower_bound(address); it != plan.end() && it->first - address < size; ++it)
            it->second.actual = 0;
    };

    const ULONG64 stamp = begin_break();
    const EndBreak end;
    DWORD error = ERROR_SUCCESS;
    for (const auto& [base, size] : broken) {
        if (!unmap_view(base)) {
            error = GetLastError();
            break;
        }
        memory.runs.erase(base);
        clear_run(base, size);
    }
    if (!error)
        error = install_alias_plan(plan, run_nodes, share_views, false, stamp, made, stats);
    bool restore_failed = false;
    if (error) {
        for (const auto address : made) {
            if (!unmap_view(address)) {
                restore_failed = true;
                continue;
            }
            auto run = memory.runs.extract(address);
            clear_run(address, run.mapped().bytes);
            run_nodes.insert(std::move(run));
        }
        made.clear();
        if (install_alias_plan(plan, run_nodes, share_views, true, stamp, made, stats))
            restore_failed = true;
    }
    // Publish the OS state, including any views left over if Windows refused rollback.
    for (const auto& [address, edit] : plan) {
        if (edit.old == edit.actual) continue;
        forget_page(address);
        if (edit.actual) {
            ++memory.backings.at(edit.actual).views;
            auto node = view_nodes.extract(address);
            node.mapped() = edit.actual;
            memory.views.insert(std::move(node));
        }
    }
    if (restore_failed) {
        x2w_local::logf("x2w-xmem page batch rollback incomplete (original error %lu)", error);
        return E_UNEXPECTED;
    }
    if (!error) coalesce_unused(plan);
    return error ? HRESULT_FROM_WIN32(error) : S_OK;
}
}  // namespace

extern "C" HRESULT WINAPI X2WMapPageBatch(UINT count, const X2WPageBatch* batches, UINT64 pool,
                                           UINT poolPages, UINT flags, X2WPageBatchStats* stats)
{
    // Buffers are valid for this call. The descriptor arrays may be supplied by the caller's
    // address space, so take a bounded snapshot before taking the memory lock.
    if (stats) *stats = {};
    constexpr UINT limit = static_cast<UINT>(era_physical::capacity_pages);
    if (flags || count > limit || (count && !batches) ||
        (poolPages && !pool) || pool % page_bytes || pool > UINT64_MAX - UINT64{poolPages} * page_bytes)
        return E_INVALIDARG;
    if (!count) return S_OK;
    try {
        std::vector<AliasRange> ranges;
        UINT total = 0;
        bool ordered_disjoint = true;
        UINT64 previous_end = 0;
        for (UINT i = 0; i < count; ++i) {
            X2WPageBatch batch{};
            if (!read_alias_record(batches, i, batch)) return E_POINTER;
            if (!batch.destination || batch.destination % page_bytes || !batch.ranges ||
                !batch.count || batch.count > limit - total)
                return E_INVALIDARG;
            auto address = batch.destination;
            for (UINT j = 0; j < batch.count; ++j) {
                X2WPageRange range{};
                if (!read_alias_record(batch.ranges, j, range)) return E_POINTER;
                const auto bytes = UINT64{range.count} * page_bytes;
                if (!range.count || range.count > limit - total || address > UINT64_MAX - bytes ||
                    range.kind > 3 || (range.kind <= 1 && range.start >= poolPages) ||
                    (range.kind == 0 && range.count > poolPages - range.start))
                    return E_INVALIDARG;
                ordered_disjoint &= address >= previous_end;
                ranges.push_back({address, range});
                total += range.count;
                address += bytes;
                previous_end = address;
            }
        }

        const bool share_views = fault_retry.ready();
        const MappingThread marker;
        const std::lock_guard guard(memory.mutex);
        constexpr ULONG_PTR keep = UINT64_MAX;
        AliasPlan plan;
        std::set<ULONG_PTR> explicit_pages;
        X2WPageBatchStats result{};
        const auto add_edit = [&](ULONG_PTR address, ULONG_PTR desired) {
            if (desired == keep) {
                ++result.pagesSkipped;
                return;
            }
            const auto live = memory.views.find(address);
            const auto old = live != memory.views.end() ? live->second : 0;
            if (memory.mirror_bytes && address - memory.mirror_gpu < memory.mirror_bytes &&
                !memory.explicit_mirror_pages.count(address))
                explicit_pages.insert(address);
            if (old == desired) {
                ++result.pagesSkipped;
            } else {
                plan.emplace(address, AliasEdit{old, desired, old});
                ++result.pagesChanged;
            }
        };
        std::map<ULONG_PTR, ULONG_PTR> requested;
        for (const auto& [base, range] : ranges) {
            for (UINT i = 0; i < range.count; ++i) {
                const auto address = base + SIZE_T{i} * page_bytes;
                if (owned_reservation(address) == memory.reservations.end()) return E_INVALIDARG;
                ULONG_PTR desired = range.kind == 3 ? keep : 0;
                if (range.kind <= 1) {
                    const auto source = pool + SIZE_T{range.start + (range.kind == 0 ? i : 0)} * page_bytes;
                    const auto view = memory.views.find(source);
                    if (view == memory.views.end()) return E_INVALIDARG;
                    desired = view->second;
                }
                // Ordered, disjoint ranges need no per-page nodes for already satisfied edits.
                if (ordered_disjoint)
                    add_edit(address, desired);
                else if (desired == keep)
                    requested.try_emplace(address, keep);
                else
                    requested.insert_or_assign(address, desired);
            }
        }
        for (const auto& [address, desired] : requested) add_edit(address, desired);
        const HRESULT hr = plan.empty() ? S_OK : remap_alias_pages(plan, share_views, result);
        if (SUCCEEDED(hr)) {
            memory.explicit_mirror_pages.merge(explicit_pages);
            if (stats) *stats = result;
        }
        return hr;
    } catch (const std::bad_alloc&) {
        return E_OUTOFMEMORY;
    }
}

// The view is a dense range of committed slots in one section. A caller can
// map it as a separate view at section_offset and open that address as a heap.
extern "C" BOOL WINAPI x2w_xmem_lease_section(UINT version, ULONG_PTR address, SIZE_T length,
                                              X2WXMemSectionLease* out, SIZE_T bytes)
{
    if (!out || bytes != sizeof(*out)) return FALSE;
    *out = {};
    constexpr SIZE_T max_pages = 64;
    if (length > max_pages * page_bytes) return FALSE;
    const std::lock_guard guard(memory.mutex);
    if (!memory.next_lease || !valid_lease_range(version, address, length)) return FALSE;
    const auto first = address & ~(page_bytes - 1);
    const auto first_byte = address - first;
    const auto count = (first_byte + length + page_bytes - 1) / page_bytes;
    if (count > max_pages) return FALSE;
    std::vector<Slot> selected;
    std::size_t pinned = 0;
    HANDLE section = nullptr, duplicate = nullptr;
    try {
        selected.reserve(count);
        for (SIZE_T i = 0; i < count; ++i) {
            const auto page = first + i * page_bytes;
            const auto* slot = lease_backing(version, page);
            if (!slot) return FALSE;
            if (selected.empty())
                section = slot->section;
            else if (slot->section != section || slot->slab != selected.front().first ||
                     slot->offset != selected.front().second + i * page_bytes)
                return FALSE;
            selected.emplace_back(slot->slab, slot->offset);
        }
        if (!DuplicateHandle(GetCurrentProcess(), section, GetCurrentProcess(), &duplicate, 0, FALSE,
                             DUPLICATE_SAME_ACCESS))
            return FALSE;
        for (const auto& slot : selected) {
            ++memory.pins[slot].references;
            ++pinned;
        }
        const auto lease = memory.next_lease;
        memory.leases.emplace(lease, SectionLease{version, first_byte, length, selected});
        ++memory.next_lease;
        *out = {duplicate, lease, selected.front().second, count * page_bytes, first_byte};
        return TRUE;
    } catch (const std::bad_alloc&) {
        if (duplicate) CloseHandle(duplicate);
        for (std::size_t i = 0; i < pinned; ++i) unpin_slot(selected[i]);
        return FALSE;
    }
}

extern "C" BOOL WINAPI x2w_xmem_match_section(UINT version, ULONG64 lease, SIZE_T byte_offset,
                                              ULONG_PTR address, SIZE_T length)
{
    if (byte_offset % page_bytes != address % page_bytes) return FALSE;
    const std::lock_guard guard(memory.mutex);
    const auto found = memory.leases.find(lease);
    if (found == memory.leases.end() || found->second.version != version ||
        !valid_lease_range(version, address, length)) return FALSE;
    const auto& pin = found->second;
    if (byte_offset < pin.first_byte || byte_offset - pin.first_byte >= pin.length ||
        length > pin.length - (byte_offset - pin.first_byte)) return FALSE;
    const SIZE_T first_slot = byte_offset / page_bytes;
    const SIZE_T last_slot = (byte_offset + length - 1) / page_bytes;
    const auto first_page = address & ~(page_bytes - 1);
    for (SIZE_T slot = first_slot; slot <= last_slot; ++slot) {
        const auto* backing = lease_backing(version, first_page + (slot - first_slot) * page_bytes);
        if (!backing || pin.slots[slot] != Slot{backing->slab, backing->offset}) return FALSE;
    }
    return TRUE;
}

extern "C" BOOL WINAPI x2w_xmem_release_section(UINT version, ULONG64 lease)
{
    std::vector<HANDLE> closed;
    {
        const std::lock_guard guard(memory.mutex);
        const auto found = memory.leases.find(lease);
        if (found == memory.leases.end() || found->second.version != version) return FALSE;
        for (const auto& slot : found->second.slots) unpin_slot(slot, &closed);
        memory.leases.erase(found);
    }
    for (const HANDLE slab : closed) CloseHandle(slab);
    return TRUE;
}

// Experimental export: one snapshot of the physical-page accounting for the startup heartbeat
// (#1199). A page gets a 64 KiB slot of a shared slab on first use, by a mapping or by a page-token write, and
// loses it when the grant is released; the number of pages holding a slot is reported so runs can be compared.
extern "C" BOOL WINAPI x2w_xmem_page_stats(ULONGLONG* live_sections)
{
    if (!live_sections) return FALSE;
    const std::lock_guard guard(memory.mutex);
    *live_sections = static_cast<ULONGLONG>(memory.backings.size());
    return TRUE;
}

// Runs fn(context) so that its answer does not come from inside a run break: the moment, under memory.mutex,
// when a run of shared views is unmapped and its surviving pages are mapped again (map_pages_locked,
// remap_alias_pages). A thread that touches such a page faults and retry_remapped_page retries it, but a region
// query (VirtualQuery) or a kernel copy (ReadProcessMemory) does not fault: inside the break it reports the pages
// as reserved, or fails. fn must be a query, or a copy into the caller's memory, that can simply run again, and
// must not take memory.mutex or call the mapping exports.
// Without shared views (X2W_XMEM_SHARED_VIEWS unset, or no mapping call yet) fn runs once, directly: every page has
// a view of its own, so no break takes a page away from a reader. With them fn runs once when no break begins or
// ends around it (the break generation is even before and unchanged after), which is nearly every call; otherwise
// it runs again under memory.mutex, so after the break. A thread that holds memory.mutex runs it directly.
extern "C" BOOL WINAPI x2w_xmem_stable_call(void (*fn)(void*), void* context)
{
    if (!fn) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    if (mapping_thread || !shared_views_on.load(std::memory_order_acquire)) {
        fn(context);
        return TRUE;
    }
    const auto generation = [] {
        AcquireSRWLockShared(&remap_window.lock);
        const ULONG64 value = remap_window.generation;
        ReleaseSRWLockShared(&remap_window.lock);
        return value;
    };
    const ULONG64 before = generation();
    if (!(before & 1)) {
        fn(context);
        if (generation() == before) return TRUE;
    }
    const MappingThread marker;
    const std::lock_guard guard(memory.mutex);
    fn(context);
    return TRUE;
}

// Test export: MapViewOfFile3 and placeholder-split attempts made by the physical-page map path so far.
extern "C" BOOL WINAPI x2w_xmem_map_counts(ULONGLONG* maps, ULONGLONG* splits)
{
    if (!maps || !splits) return FALSE;
    *maps = static_cast<ULONGLONG>(map_attempts);
    *splits = static_cast<ULONGLONG>(split_attempts);
    return TRUE;
}

// Test export, counted in the test-controls build only: views X2WMapPageBatch could not map in place inside an
// owned reservation whose range alias_placeholders refused, and those for which it tried a placeholder release.
// The deployed build returns FALSE with ERROR_NOT_SUPPORTED.
extern "C" BOOL WINAPI x2w_xmem_alias_counts(ULONGLONG* refusals, ULONGLONG* splits)
{
#ifdef X2W_XMEM_TEST_CONTROLS
    if (!refusals || !splits) return FALSE;
    *refusals = static_cast<ULONGLONG>(alias_refusals);
    *splits = static_cast<ULONGLONG>(alias_splits);
    return TRUE;
#else
    (void)refusals;
    (void)splits;
    SetLastError(ERROR_NOT_SUPPORTED);
    return FALSE;
#endif
}

// Experimental export: the slab accounting behind those pages (#1199): slabs open, slots in use and slots free
// in the open slabs. A slab is reserved when it is created, its slots are committed on first use, and empty slabs
// are closed when all of their slots are freed and unpinned.
extern "C" BOOL WINAPI x2w_xmem_slab_stats(ULONGLONG* slabs, ULONGLONG* slots_in_use, ULONGLONG* slots_free)
{
    if (!slabs || !slots_in_use || !slots_free) return FALSE;
    const std::lock_guard guard(memory.mutex);
    *slabs = memory.slots.slab_count();
    *slots_in_use = memory.slots.slots_in_use();
    *slots_free = memory.slots.slots_free();
    return TRUE;
}

// Experimental export (X2W_XMEM_LAZY_COMMIT): commits the lazily mapped slots behind [address, address + bytes)
// before a writer that cannot fault one in uses the range: kernel-mode I/O into mapped pages fails with
// STATUS_ACCESS_VIOLATION instead of faulting, and the GPU reads and writes placed resources without the CPU.
// GPU-view addresses resolve to the same slots. TRUE without the switch or when nothing in the range is lazy;
// FALSE with the system error when a commit fails.
extern "C" BOOL WINAPI x2w_xmem_commit_range(const void* address, SIZE_T bytes)
{
    if (!lazy_commit() || !address || !bytes) return TRUE;
    const auto start = reinterpret_cast<ULONG_PTR>(address);
    const ULONG_PTR end = bytes > ~ULONG_PTR{0} - start ? ~ULONG_PTR{0} : start + bytes;
    ULONG_PTR from = start;  // the part of the range past any lazy startup blocks it starts in
    if (lazy_startup()) {
        ULONG_PTR base = 0;
        SIZE_T block_bytes = 0;
        bool committed = true;
        AcquireSRWLockShared(&lazy_blocks.lock);
        for (; from < end && lazy_block_at(from, base, block_bytes); from = base + block_bytes) {
            committed = commit_block_chunks(base, block_bytes, from, end);
            if (!committed) break;
        }
        ReleaseSRWLockShared(&lazy_blocks.lock);  // keeps the commit's error
        if (!committed) return FALSE;
        if (from >= end) return TRUE;  // the range lies in lazy startup blocks
    }
    const std::lock_guard guard(memory.mutex);
    for (ULONG_PTR page = from & ~(ULONG_PTR{page_bytes} - 1); page < end; page += page_bytes) {
        Backing* backing = backing_at(page);
        if (backing && !commit_backing(*backing)) return FALSE;
        if (page > ~ULONG_PTR{0} - page_bytes) break;
    }
    return TRUE;
}

namespace {
// Keep the guarded copy out of functions that need unwinding (as read_alias_record).
bool copy_guarded(void* destination, const void* data, SIZE_T bytes) noexcept
{
    __try {
        std::memcpy(destination, data, bytes);
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ||
                GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

// The longest hold of memory.mutex by a destination copy, and the largest copy, so far (under memory.mutex).
double copy_hold_max_ms = 0;
SIZE_T copy_bytes_max = 0;
#ifdef X2W_XMEM_TEST_CONTROLS
// Test build only (x2w_xmem_copy_counts): destination copies made under memory.mutex and without it.
volatile LONG64 locked_copies, unlocked_copies;
#endif

// Whether [address, address + bytes) overlaps a recorded run. Caller holds memory.mutex.
bool overlaps_run(ULONG_PTR address, SIZE_T bytes)
{
    if (!bytes) return false;
    const auto run = memory.runs.upper_bound(address);
    if (run != memory.runs.begin() && address - std::prev(run)->first < std::prev(run)->second.bytes) return true;
    return run != memory.runs.end() && run->first - address < bytes;
}

// [destination, destination + capacity) committed and writable: resident read-write pages from their working-set
// attributes, any other page through the region walk (as XCRDAPI's destination check without xmem.dll).
bool destination_writable(void* destination, SIZE_T capacity)
{
    const auto start = reinterpret_cast<ULONG_PTR>(destination);
    if (capacity && destination && capacity <= ~ULONG_PTR{0} - start &&
        x2w_local::resident_read_write(start, start + capacity))
        return true;
    return x2w_local::committed_writable(destination, capacity);
}
}  // namespace

// The DirectStorage stand-in's copy into a plain-memory destination (XCRDAPI): the check that all of
// [destination, destination + capacity) is committed and writable, then the copy of bytes from data. Unmapping a page
// of a shared run takes the whole run away until its other pages are mapped again (map_pages_locked,
// remap_alias_pages, merge_with_previous), all under memory.mutex. With shared views on (shared_views_on), a short
// hold of that mutex first looks up whether either range overlaps a recorded run. If one does, the check and the
// copy run under the mutex, so neither can fall into that window and no such window can open while the copy runs.
// Otherwise, and always with shared views off (every page then has a view of its own, and no unmap takes a
// neighbour away), both run without the mutex, as in XCRDAPI without xmem.dll. A run can then appear over either
// range only through a mapping call on those addresses during the copy, which is the caller's race. Lazily mapped
// pages of both ranges are committed first, before any mutex: a fault under it is passed on (mapping_thread), not
// served.
// TRUE when the bytes were copied. FALSE with ERROR_NOACCESS when the destination is not writable (nothing is
// written), ERROR_INVALID_ADDRESS when the copy faulted (a source page not readable, or a destination page that
// stopped being writable during the copy; part of the destination may be written), or ERROR_INVALID_PARAMETER for
// more bytes than capacity or null data.
// Test build only: X2W_XMEM_TEST_COPY_ERROR=<code> fails every copy with that error before anything is written
// (startup_dstorage_break_window_check outcomes: XCRDAPI's logged outcome per error).
extern "C" BOOL WINAPI x2w_xmem_copy_to_destination(void* destination, SIZE_T capacity, const void* data, SIZE_T bytes)
{
    if (bytes > capacity || (bytes && !data)) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
#ifdef X2W_XMEM_TEST_CONTROLS
    char forced[16];
    if (GetEnvironmentVariableA("X2W_XMEM_TEST_COPY_ERROR", forced, sizeof(forced)) && forced[0]) {
        SetLastError(static_cast<DWORD>(std::strtoul(forced, nullptr, 10)));
        return FALSE;
    }
#endif
    const auto start = reinterpret_cast<ULONG_PTR>(destination);
    if (capacity && destination && capacity <= ~ULONG_PTR{0} - start) x2w_xmem_commit_range(destination, capacity);
    if (bytes) x2w_xmem_commit_range(data, bytes);
    DWORD error = ERROR_SUCCESS;
    bool in_runs = false;
    if (shared_views_on.load(std::memory_order_acquire)) {
        const std::lock_guard guard(memory.mutex);
        in_runs = overlaps_run(start, capacity) || overlaps_run(reinterpret_cast<ULONG_PTR>(data), bytes);
    }
    if (!in_runs) {
#ifdef X2W_XMEM_TEST_CONTROLS
        InterlockedIncrement64(&unlocked_copies);
#endif
        if (!destination_writable(destination, capacity))
            error = ERROR_NOACCESS;
        else if (bytes && !copy_guarded(destination, data, bytes))
            error = ERROR_INVALID_ADDRESS;
        SetLastError(error);
        return error == ERROR_SUCCESS;
    }
    double held_ms = 0;
    bool longest = false, largest = false;
    {
        const MappingThread marker;
        const std::lock_guard guard(memory.mutex);
#ifdef X2W_XMEM_TEST_CONTROLS
        InterlockedIncrement64(&locked_copies);
#endif
        LARGE_INTEGER frequency{}, begin{}, end{};
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&begin);
        if (!destination_writable(destination, capacity))
            error = ERROR_NOACCESS;
        else if (bytes && !copy_guarded(destination, data, bytes))
            error = ERROR_INVALID_ADDRESS;
        QueryPerformanceCounter(&end);
        held_ms = frequency.QuadPart ? 1000.0 * static_cast<double>(end.QuadPart - begin.QuadPart) / frequency.QuadPart : 0;
        if (held_ms > copy_hold_max_ms) {
            copy_hold_max_ms = held_ms;
            longest = true;
        }
        if (!error && bytes > copy_bytes_max) {
            copy_bytes_max = bytes;
            largest = true;
        }
    }
    // A line per new maximum (about the logarithm of the number of copies), logged after the mutex is released.
    static volatile LONG lines;
    if ((longest || largest) && x2w_local::first_calls(lines, 64))
        x2w_local::logf("x2w-xmem destination copy: %s%s%.3f ms under the mapping mutex for 0x%llx bytes (capacity 0x%llx)",
                        longest ? "longest hold so far, " : "", largest ? "largest copy so far, " : "", held_ms,
                        static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(capacity));
    SetLastError(error);
    return error == ERROR_SUCCESS;
}

// Test export, counted in the test-controls build only: destination copies (x2w_xmem_copy_to_destination) made under
// memory.mutex because a range overlapped a run, and those made without it. The deployed build returns FALSE with
// ERROR_NOT_SUPPORTED.
extern "C" BOOL WINAPI x2w_xmem_copy_counts(ULONGLONG* locked, ULONGLONG* unlocked)
{
#ifdef X2W_XMEM_TEST_CONTROLS
    if (!locked || !unlocked) return FALSE;
    *locked = static_cast<ULONGLONG>(locked_copies);
    *unlocked = static_cast<ULONGLONG>(unlocked_copies);
    return TRUE;
#else
    (void)locked;
    (void)unlocked;
    SetLastError(ERROR_NOT_SUPPORTED);
    return FALSE;
#endif
}

// Experimental export (X2W_XMEM_LAZY_STARTUP): commits the whole lazy startup block holding address and stops
// committing it lazily. A GPU heap opened on the block covers the committed run from its base and is reused for
// later resources in the block, so the block is committed whole before the first one is placed. TRUE without the
// switch or when the address is in no lazy block; FALSE with the system error when the commit fails.
extern "C" BOOL WINAPI x2w_xmem_commit_block(const void* address)
{
    if (!lazy_startup()) return TRUE;
    ULONG_PTR base = 0;
    SIZE_T bytes = 0;
    AcquireSRWLockExclusive(&lazy_blocks.lock);
    const bool lazy = lazy_block_at(reinterpret_cast<ULONG_PTR>(address), base, bytes);
    const bool committed = !lazy || VirtualAlloc(reinterpret_cast<void*>(base), bytes, MEM_COMMIT, PAGE_READWRITE);
    if (lazy && committed) lazy_blocks.blocks.erase(base);
    ReleaseSRWLockExclusive(&lazy_blocks.lock);  // keeps the commit's error
    return committed;
}

namespace {
// The title reads files straight into mapped physical pages; a kernel-mode write into an uncommitted slot fails
// with STATUS_ACCESS_VIOLATION (no fault), so with X2W_XMEM_LAZY_COMMIT the main module's ReadFile import commits
// the buffer first.
using ReadFileFn = BOOL(WINAPI*)(HANDLE, LPVOID, DWORD, LPDWORD, LPOVERLAPPED);
ReadFileFn real_read_file;

BOOL WINAPI read_file_committed(HANDLE file, LPVOID buffer, DWORD bytes, LPDWORD read, LPOVERLAPPED overlapped)
{
    if (buffer && bytes && !x2w_xmem_commit_range(buffer, bytes)) return FALSE;
    return real_read_file(file, buffer, bytes, read, overlapped);
}

// The title releases its startup blocks through its VirtualFree import: with X2W_XMEM_LAZY_STARTUP a released
// block leaves the lazy list, so a later allocation at the same addresses is never committed as part of it. The
// release runs under the list lock, so no commit of the block is in flight, and the block leaves the list only if
// the release succeeded: a refused release (a nonzero size, for instance) leaves it reserved and still lazy.
using VirtualFreeFn = BOOL(WINAPI*)(LPVOID, SIZE_T, DWORD);
VirtualFreeFn real_virtual_free;

BOOL WINAPI virtual_free_lazy(LPVOID address, SIZE_T bytes, DWORD type)
{
    if (!(type & MEM_RELEASE)) return real_virtual_free(address, bytes, type);
    AcquireSRWLockExclusive(&lazy_blocks.lock);
    const BOOL released = real_virtual_free(address, bytes, type);
    const DWORD error = GetLastError();
    if (released && lazy_blocks.blocks.erase(reinterpret_cast<ULONG_PTR>(address)))
        InterlockedIncrement(&lazy_block_releases);
    ReleaseSRWLockExclusive(&lazy_blocks.lock);
    SetLastError(error);
    return released;
}

using x2w_xmem::patch_main_import;  // src/xmem_import_patch.h

// Test build only (tests/xmem_lazy_imports_check.cpp): X2W_XMEM_TEST_LAZY_HANDLER_FAILURE=1 refuses the handler.
bool add_lazy_handler()
{
#ifdef X2W_XMEM_TEST_CONTROLS
    if (x2w_local::env_on("X2W_XMEM_TEST_LAZY_HANDLER_FAILURE")) return false;
#endif
    return AddVectoredExceptionHandler(1, lazy_commit_fault) != nullptr;
}

// With the switch, install the fault handler and the ReadFile wrapper when xmem.dll loads, before the title runs.
// The module is pinned: the handler must outlive any unload. If the handler cannot be installed, lazy commit stays
// off and nothing is wrapped.
const bool lazy_commit_installed = [] {
    if (!lazy_commit_requested()) return false;
    HMODULE self = nullptr;
    lazy_handler_installed = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                                 reinterpret_cast<LPCWSTR>(&lazy_commit_fault), &self) &&
                             add_lazy_handler();
    if (!lazy_handler_installed) {
        x2w_local::logf("x2w-xmem lazy commit (X2W_XMEM_LAZY_COMMIT): fault handler failed, pages committed when mapped");
        return false;
    }
    real_read_file = reinterpret_cast<ReadFileFn>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "ReadFile"));
    const int reads = real_read_file ? patch_main_import("ReadFile", reinterpret_cast<const void*>(&read_file_committed)) : 0;
    x2w_local::logf("x2w-xmem lazy commit (X2W_XMEM_LAZY_COMMIT): fault handler on, %d ReadFile import(s) wrapped", reads);
    if (lazy_startup()) {
        real_virtual_free = reinterpret_cast<VirtualFreeFn>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "VirtualFree"));
        const int frees = real_virtual_free ? patch_main_import("VirtualFree", reinterpret_cast<const void*>(&virtual_free_lazy)) : 0;
        x2w_local::logf("x2w-xmem lazy startup blocks (X2W_XMEM_LAZY_STARTUP): %d VirtualFree import(s) wrapped", frees);
    }
    return true;
}();

// With X2W_XMEM_SHARED_VIEWS=1 the main module's VirtualQueryEx imports answer outside run breaks
// (x2w_xmem_stable_call): inside one, a page of the run being taken apart reads as reserved. A query of another
// process runs the same way; it only costs the retry. The module is pinned once an import points into it (a main
// module without the import leaves it unloadable, as before); if the pin is refused, the imports are restored.
using VirtualQueryExFn = SIZE_T(WINAPI*)(HANDLE, LPCVOID, PMEMORY_BASIC_INFORMATION, SIZE_T);
VirtualQueryExFn real_virtual_query_ex;

// The answer goes to a local buffer and is copied to the caller's after x2w_xmem_stable_call: on its mutex path the
// lazy-commit handler passes faults on (MappingThread), so a write into an uncommitted mapped slot would not be
// committed there.
SIZE_T WINAPI virtual_query_ex_stable(HANDLE process, LPCVOID address, PMEMORY_BASIC_INFORMATION info, SIZE_T bytes)
{
    if (!info || bytes < sizeof(MEMORY_BASIC_INFORMATION)) return real_virtual_query_ex(process, address, info, bytes);
    struct Query {
        HANDLE process;
        LPCVOID address;
        MEMORY_BASIC_INFORMATION info;
        SIZE_T result;
        DWORD error;
    } query{process, address, {}, 0, ERROR_SUCCESS};
    x2w_xmem_stable_call([](void* context) {
        auto& q = *static_cast<Query*>(context);
        q.result = real_virtual_query_ex(q.process, q.address, &q.info, sizeof(q.info));
        q.error = GetLastError();
    }, &query);
    if (query.result) std::memcpy(info, &query.info, query.result);
    SetLastError(query.error);
    return query.result;
}

const int virtual_query_ex_wrapped = [] {
    if (!x2w_local::env_on("X2W_XMEM_SHARED_VIEWS")) return 0;
    real_virtual_query_ex = reinterpret_cast<VirtualQueryExFn>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "VirtualQueryEx"));
    if (!real_virtual_query_ex) return 0;
    int queries = patch_main_import("VirtualQueryEx", reinterpret_cast<const void*>(&virtual_query_ex_stable));
    HMODULE self = nullptr;
    if (queries && !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                       reinterpret_cast<LPCWSTR>(&virtual_query_ex_stable), &self)) {
        const DWORD error = GetLastError();
        const int restored = patch_main_import("VirtualQueryEx", reinterpret_cast<const void*>(real_virtual_query_ex));
        x2w_local::logf("x2w-xmem shared views (X2W_XMEM_SHARED_VIEWS): xmem.dll could not be pinned (error %lu); "
                        "%d of %d VirtualQueryEx import(s) restored", static_cast<unsigned long>(error), restored, queries);
        queries = 0;
    }
    x2w_local::logf("x2w-xmem shared views (X2W_XMEM_SHARED_VIEWS): %d VirtualQueryEx import(s) answer outside run breaks",
                    queries);
    return queries;
}();

// With X2W_XMEM_SHARED_VIEWS=1 the main module's ReadFile and WriteFile imports also go around run breaks. Inside one,
// a kernel copy into or out of a page of the run being taken apart fails at once, with no fault to retry. Measured
// under Wine (tests/xmem_run_break_readers_check.cpp, ReadFile_import and WriteFile_import): ReadFile fails with
// ERROR_NOACCESS and leaves the file position where it was, WriteFile with ERROR_INVALID_USER_BUFFER. A synchronous
// call (no OVERLAPPED) on a disk file that fails with one of those while a break was in progress, or began or ended
// around it, and whose buffer lies in a run, waits for memory.mutex (the end of the break) and runs again, at most
// io_attempts times in all. A call that fails that way with no break around it, outside the runs, or on a handle that
// is not a disk file, fails at once as before: a pipe, socket or console may have consumed or sent data in the failed
// call, so running it again could lose or repeat that data. Under Wine a pipe read that fails this way leaves its data
// queued; whether Windows consumes it is unmeasured. GetFileType is asked only on that failure path.
// Not covered: overlapped I/O (the copy happens at completion, after the call has returned), a call that a break
// cuts short after part of the copy (a short count), and I/O from other modules or through GetProcAddress.
using WriteFileFn = BOOL(WINAPI*)(HANDLE, LPCVOID, DWORD, LPDWORD, LPOVERLAPPED);
ReadFileFn next_read_file;  // read_file_committed with lazy commit, kernel32's ReadFile without
WriteFileFn kernel_write_file;
constexpr int io_attempts = 4;

// Whether [buffer, buffer + bytes) holds a page of a run. Takes memory.mutex, so it returns after the break in progress.
bool in_a_run(const void* buffer, DWORD bytes)
{
    const std::lock_guard guard(memory.mutex);
    return overlaps_run(reinterpret_cast<ULONG_PTR>(buffer), bytes);
}

template <class Call>
BOOL io_outside_breaks(HANDLE file, const void* buffer, DWORD bytes, LPOVERLAPPED overlapped, const Call& call)
{
    if (overlapped || !buffer || !bytes || mapping_thread || !shared_views_on.load(std::memory_order_acquire)) return call();
    for (int attempt = 1;; ++attempt) {
        const ULONG64 before = break_generation();
        if (call()) return TRUE;
        const DWORD error = GetLastError();
        const bool broken = (before & 1) || break_generation() != before;
        if ((error != ERROR_NOACCESS && error != ERROR_INVALID_USER_BUFFER) || !broken || attempt == io_attempts ||
            GetFileType(file) != FILE_TYPE_DISK || !in_a_run(buffer, bytes)) {
            SetLastError(error);
            return FALSE;
        }
    }
}

BOOL WINAPI read_file_stable(HANDLE file, LPVOID buffer, DWORD bytes, LPDWORD read, LPOVERLAPPED overlapped)
{
    return io_outside_breaks(file, buffer, bytes, overlapped, [&] { return next_read_file(file, buffer, bytes, read, overlapped); });
}

BOOL WINAPI write_file_stable(HANDLE file, LPCVOID buffer, DWORD bytes, LPDWORD written, LPOVERLAPPED overlapped)
{
    return io_outside_breaks(file, buffer, bytes, overlapped, [&] { return kernel_write_file(file, buffer, bytes, written, overlapped); });
}

const int file_io_wrapped = [] {
    if (!x2w_local::env_on("X2W_XMEM_SHARED_VIEWS")) return 0;
    const HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
    const auto kernel_read_file = reinterpret_cast<ReadFileFn>(GetProcAddress(kernel32, "ReadFile"));
    kernel_write_file = reinterpret_cast<WriteFileFn>(GetProcAddress(kernel32, "WriteFile"));
    if (!kernel_read_file || !kernel_write_file) return 0;
    // With lazy commit the ReadFile imports already point at read_file_committed, which stays the inner call.
    next_read_file = lazy_commit_installed && real_read_file ? &read_file_committed : kernel_read_file;
    int wrapped = patch_main_import("ReadFile", reinterpret_cast<const void*>(&read_file_stable));
    wrapped += patch_main_import("WriteFile", reinterpret_cast<const void*>(&write_file_stable));
    HMODULE self = nullptr;
    if (wrapped && !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                       reinterpret_cast<LPCWSTR>(&read_file_stable), &self)) {
        const DWORD error = GetLastError();
        const int restored = patch_main_import("ReadFile", reinterpret_cast<const void*>(next_read_file)) +
                             patch_main_import("WriteFile", reinterpret_cast<const void*>(kernel_write_file));
        x2w_local::logf("x2w-xmem shared views (X2W_XMEM_SHARED_VIEWS): xmem.dll could not be pinned (error %lu); "
                        "%d of %d ReadFile/WriteFile import(s) restored", static_cast<unsigned long>(error), restored, wrapped);
        wrapped = 0;
    }
    x2w_local::logf("x2w-xmem shared views (X2W_XMEM_SHARED_VIEWS): %d ReadFile/WriteFile import(s) run again after a run "
                    "break that refused their buffer", wrapped);
    return wrapped;
}();
}  // namespace

// Experimental export: writes bytes into physical pages named by the title's
// page tokens (page number << 4, as XMemMapPhysicalPages takes them), starting offset bytes into the
// first page. DirectStorage requests can name such pages as their destination. A token that names
// no granted page is refused like a mapping of it (ERROR_INVALID_PARAMETER): a section for a page
// nobody owns would stay until that page number is granted again, and that grant would map its
// bytes instead of zeros.
extern "C" BOOL WINAPI x2w_xmem_write_physical(const ULONG_PTR* tokens, ULONG_PTR count, SIZE_T offset, const void* data, SIZE_T bytes)
{
    X2W_ZONE("x2w_xmem_write_physical");
    if (!tokens || !data || offset >= page_bytes) return FALSE;
    const auto* source = static_cast<const unsigned char*>(data);
    ULONG_PTR index = 0;
    while (bytes) {
        if (index >= count || (tokens[index] & 0xF)) return FALSE;
        HANDLE section = nullptr;
        ULONG64 section_offset = 0;
        {
            const std::lock_guard guard(memory.mutex);
            if (!era_physical::owns(tokens[index] >> 4)) return physical_result(era_physical::Status::invalid);
            section = section_for(tokens[index] >> 4);
            if (section) section_offset = memory.backings.at(tokens[index] >> 4).offset;
        }
        if (!section) return FALSE;
        void* view = MapViewOfFile(section, FILE_MAP_WRITE, static_cast<DWORD>(section_offset >> 32),
                                   static_cast<DWORD>(section_offset), page_bytes);
        if (!view) return FALSE;
        const SIZE_T chunk = bytes < page_bytes - offset ? bytes : page_bytes - offset;
        std::memcpy(static_cast<unsigned char*>(view) + offset, source, chunk);
        UnmapViewOfFile(view);
        source += chunk;
        bytes -= chunk;
        offset = 0;
        ++index;
    }
    return TRUE;
}

extern "C" void* WINAPI XMemMapPhysicalPages(void* base, ULONG_PTR count, const ULONG_PTR* tokens)
{
    const auto address = reinterpret_cast<ULONG_PTR>(base);
    if (!address || count > (std::numeric_limits<ULONG_PTR>::max() - address) / page_bytes) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (local_permissive()) {
        if (count > era_physical::capacity_pages) {  // as the strict path; sizes the vectors below
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        // A null array unmaps (seen on a cleanup path, also with a zero count).
        std::vector<ULONG_PTR> addresses(count), pages(count, 0);
        for (ULONG_PTR i = 0; i < count; ++i) addresses[i] = address + i * page_bytes;
        if (tokens && !to_pages(count, tokens, pages)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        if (!count) return base;
        if (!local_apply(count, addresses.data(), pages.data(), "XMemMapPhysicalPages")) return nullptr;
        return base;
    }
    return map_pages(count, tokens, [=](ULONG_PTR i) { return address + i * page_bytes; }) ? base : nullptr;
}

extern "C" BOOL WINAPI XMemMapPhysicalPagesScatter(void* const* addresses, ULONG_PTR count,
                                                   const ULONG_PTR* tokens)
{
    if (!addresses) return physical_result(era_physical::Status::invalid);
    if (local_permissive() && tokens) {
        std::vector<ULONG_PTR> pages;
        if (!to_pages(count, tokens, pages)) return physical_result(era_physical::Status::invalid);
        if (!local_apply(count, reinterpret_cast<const ULONG_PTR*>(addresses), pages.data(), "XMemMapPhysicalPagesScatter"))
            return FALSE;
        return TRUE;
    }
    return map_pages(count, tokens, [=](ULONG_PTR i) { return reinterpret_cast<ULONG_PTR>(addresses[i]); });
}

// Startup compatibility configuration (#1758, under #874): X2W_XMEM_STARTUP_BELOW_40=1 places the
// large-page pool blocks with attribute 0x283 (protection 0x2004 or 0x2404) so that the whole block,
// last byte included, lies below 2^40. The title passes the CPU address of such a block straight
// through as a GPU address, and a package acceleration-structure builder refuses destinations with
// more than 40 address bits. Off by default; every other shape keeps its placement. The low range is
// a requirement, never a preference: an allocation that cannot be placed there fails with the host
// error, and nothing is truncated, masked or retried higher.
constexpr ULONG_PTR startup_low_limit = ULONG_PTR{1} << 40;  // exclusive
constexpr SIZE_T startup_large_page = SIZE_T{2} << 20;

#ifdef X2W_XMEM_TEST_CONTROLS
// Test build only (tests/xmem_startup_below_40_check.cpp): X2W_XMEM_TEST_BELOW_40 names mutants
// that drop the address limit ("no_limit"), the check after the allocation ("no_check") or the
// release of a refused block ("no_release").
static bool startup_below_40_mutant(const char* name) noexcept
{
    char text[64]{};
    return GetEnvironmentVariableA("X2W_XMEM_TEST_BELOW_40", text, sizeof(text)) && std::strstr(text, name);
}
#else
static bool startup_below_40_mutant(const char*) noexcept { return false; }
#endif

// Checks a block allocated with the 2^40 requirement: 2 MiB aligned and base + bytes - 1 < 2^40,
// computed without overflow. A block outside that range is released and refused.
static void* keep_startup_block_below_40(void* block, SIZE_T bytes, ULONGLONG attributes, DWORD protection) noexcept
{
    if (!block) {
        const DWORD error = GetLastError();
        x2w_local::logf("x2w-xmem startup block below 2^40 refused: size=0x%llx attr=0x%llx prot=0x%lx, no 2 MiB-aligned "
                        "range ending below 2^40 (host error %lu)",
                        static_cast<unsigned long long>(bytes), attributes, static_cast<unsigned long>(protection),
                        static_cast<unsigned long>(error));
        SetLastError(error);
        return nullptr;
    }
    const auto base = reinterpret_cast<ULONG_PTR>(block);
    const bool inside = base % startup_large_page == 0 && bytes <= startup_low_limit && base <= startup_low_limit - bytes;
    if (!inside && !startup_below_40_mutant("no_check")) {
        const BOOL released = startup_below_40_mutant("no_release") ? FALSE : VirtualFree(block, 0, MEM_RELEASE);
        x2w_local::logf("x2w-xmem startup block below 2^40 refused: size=0x%llx attr=0x%llx prot=0x%lx, placed at %p "
                        "outside the 2 MiB-aligned range below 2^40 (%s)",
                        static_cast<unsigned long long>(bytes), attributes, static_cast<unsigned long>(protection), block,
                        released ? "released" : "not released");
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    x2w_local::logf("x2w-xmem startup block below 2^40: %p size=0x%llx last=%p attr=0x%llx prot=0x%lx", block,
                    static_cast<unsigned long long>(bytes), reinterpret_cast<void*>(base + (bytes - 1)), attributes,
                    static_cast<unsigned long>(protection));
    return block;
}

// Reserve+commit desktop profile for the startup blocks observed on #870: the pool loop asks
// for type 0x20003000 with 0x183/0x22004, 0x283/0x2404 or 0x283/0x2004, and later startup code for type
// 0x20403000 (64 KiB pages) with 0x181/PAGE_READWRITE, each with a null base. Proposed
// desktop policy: commit a fresh block of ordinary zero-filled read/write pages, whose size
// must be a nonzero multiple of 64 KiB and is committed exactly, without rounding. The
// MEM_LARGE_PAGES type gets a 2 MiB-aligned base (the x64 large-page size); the
// MEM_64K_PAGES type (MEM_LARGE_PAGES | MEM_PHYSICAL) keeps the 64 KiB allocation
// granularity. Large pages themselves and the console protection bits above PAGE_READWRITE
// are not reproduced. A host failure returns NULL with the host error. X2W_XMEM_STARTUP_BELOW_40
// adds an address limit for the 0x283 blocks (keep_startup_block_below_40).
static void* commit_startup_block(SIZE_T bytes, DWORD allocation_type, ULONGLONG attributes,
                                  DWORD protection) noexcept
{
    constexpr SIZE_T granularity = SIZE_T{64} << 10;
    const bool large_pages = allocation_type == 0x20003000;
    const bool supported = large_pages
                               ? (attributes == 0x183 && protection == 0x22004)
                                     || (attributes == 0x283 && (protection == 0x2404 || protection == 0x2004))
                               : attributes == 0x181 && protection == PAGE_READWRITE;
    if (!bytes || bytes % granularity || !supported) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    const bool below_40 = large_pages && attributes == 0x283 && x2w_local::env_on("X2W_XMEM_STARTUP_BELOW_40");
    // X2W_XMEM_LAZY_STARTUP: a large block is reserved and committed on first touch (lazy_startup).
    const bool lazy = lazy_startup() && bytes >= lazy_block_min;
    const DWORD type = lazy ? MEM_RESERVE : MEM_RESERVE | MEM_COMMIT;
    void* block = nullptr;
    if (!large_pages) {
        block = VirtualAlloc(nullptr, bytes, type, PAGE_READWRITE);
    } else {
        MEM_ADDRESS_REQUIREMENTS requirements{};
        requirements.Alignment = SIZE_T{2} << 20;
        // supported already limits attribute 0x283 to protection 0x2004 or 0x2404.
        if (below_40 && !startup_below_40_mutant("no_limit"))
            requirements.HighestEndingAddress = reinterpret_cast<void*>(startup_low_limit - 1);
        MEM_EXTENDED_PARAMETER parameter{};
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        block = VirtualAlloc2(GetCurrentProcess(), nullptr, bytes, type, PAGE_READWRITE, &parameter, 1);
        if (below_40) block = keep_startup_block_below_40(block, bytes, attributes, protection);
    }
    if (block && lazy) {
        bool listed = false;
        AcquireSRWLockExclusive(&lazy_blocks.lock);
        try {
            lazy_blocks.blocks[reinterpret_cast<ULONG_PTR>(block)] = bytes;
            listed = true;
        } catch (const std::bad_alloc&) {
        }
        ReleaseSRWLockExclusive(&lazy_blocks.lock);
        if (!listed && !VirtualAlloc(block, bytes, MEM_COMMIT, PAGE_READWRITE)) {  // not listed: commit it now
            const DWORD error = GetLastError();
            VirtualFree(block, 0, MEM_RELEASE);
            SetLastError(error);
            return nullptr;
        }
    }
    return block;
}

// The startup commit profile above, and the reserve-only desktop profile from the source-built
// #888 comparison.
extern "C" void* WINAPI XMemVirtualAlloc(void* address, SIZE_T bytes, DWORD allocation_type,
                                        ULONGLONG attributes, DWORD protection)
{
    constexpr DWORD startup_reserve = 0x20402000;
    constexpr SIZE_T gib = SIZE_T{1} << 30;
    if (!address && (allocation_type == 0x20003000 || allocation_type == (startup_reserve | MEM_COMMIT)))
        return commit_startup_block(bytes, allocation_type, attributes, protection);
    const bool supported = (attributes == 0x21 && protection == PAGE_READWRITE)
                           || (attributes == 3 && protection == (0x800 | PAGE_NOACCESS));
    if (address || (bytes != 16 * gib && bytes != 24 * gib) || allocation_type != startup_reserve || !supported) {
        if (local_permissive()) return local_virtual_alloc(address, bytes, allocation_type, attributes, protection);
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    const std::lock_guard guard(memory.mutex);
    // Callers release unused reservations with Win32. Drop entries for released
    // ranges before another allocation can reuse those virtual addresses.
    for (auto it = memory.reservations.begin(); it != memory.reservations.end();) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<void*>(it->first), &info, sizeof(info)) && info.State == MEM_FREE)
            it = memory.reservations.erase(it);
        else
            ++it;
    }
    auto* reservation = VirtualAlloc2(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                      PAGE_NOACCESS, nullptr, 0);
    if (!reservation) return nullptr;
    try {
        memory.reservations.insert_or_assign(reinterpret_cast<ULONG_PTR>(reservation), bytes);
        const auto base = reinterpret_cast<ULONG_PTR>(reservation);
        if (protection == PAGE_READWRITE) {
            memory.last_cpu_base = base;
            memory.last_cpu_bytes = bytes;
        } else if (memory.last_cpu_base && memory.last_cpu_bytes == bytes && x2w_local::env_on("X2W_XMEM_GPU_MIRROR")) {
            memory.mirror_cpu = memory.last_cpu_base;
            memory.mirror_gpu = base;
            memory.mirror_bytes = bytes;
            memory.explicit_mirror_pages.clear();
            x2w_local::logf("x2w-xmem GPU view %p (+0x%llx) mirrors CPU view %p, 0x%llx bytes", reservation,
                            static_cast<unsigned long long>(base - memory.last_cpu_base), reinterpret_cast<void*>(memory.last_cpu_base),
                            static_cast<unsigned long long>(bytes));
        }
    } catch (const std::bad_alloc&) {
        VirtualFree(reservation, 0, MEM_RELEASE);
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    return reservation;
}
