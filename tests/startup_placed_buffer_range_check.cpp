// measure_placed_buffer (src/startup_stubs/placed_buffer_range.h) without a device. Device slot 133
// aliases a buffer only if every byte from the allocation base to the end of the buffer is committed
// in that allocation. Every decision must equal the one the full walk (measure the whole committed
// run from the base, as slot 133 did before) gives on the same memory at the same moment. Each case
// places a buffer before changing the memory, so state kept from an earlier placement would be
// exposed by the next one.
//
// 1. Striped: a 64 MiB private block whose odd 64 KiB pages are read-only (1024 regions), 256
//    placements plus a misaligned one, one past the end and one at the base.
// 2. Middle-page decommit: three committed pages, the middle one decommitted. A buffer on the third
//    page (gap before its start), one across all three, one on the first page and one on the gap.
// 3. Recommit: the middle page committed again; the buffers after and across it are accepted again.
// 4. Gap before the start, far from the buffer: page 1 of eight is decommitted, the buffer is on page 6.
// 4b. The same shape with a reviewer's numbers: 1 MiB committed, a 64 KiB buffer at +0x80000, then
//    0x10000 bytes at +0x10000 decommitted.
// 5. Same-base replacement: the 64 MiB block is released and a 1 MiB block committed at its base. A
//    buffer inside it, one ending past its end and one past its end.
// 6. A mapped view is refused.
// 7. A reservation committed in two steps.
// 8. A placeholder split in two, its first piece replaced by committed private memory: a buffer inside
//    the piece and one ending in the placeholder after it.
// 8b. Other private allocations: MEM_WRITE_WATCH, MEM_TOP_DOWN, PAGE_EXECUTE_READWRITE, a heap block
//    (compare the observed layout with the full walk), a reservation with only its second 64 KiB
//    committed, an AWE reservation (MEM_PHYSICAL, nothing committed) and, when the process may lock
//    pages, MEM_LARGE_PAGES (otherwise reported, not run).
// 9. Cost: a 512 MiB committed private block with one page touched per 64 KiB, 64 buffers spread over
//    it. Decisions as above; in addition every placement must be proven without VirtualQuery (a call
//    count, not a time). The time per placement of the full walk, the bounded walk and the exact
//    proof is printed.
//
// --mutate region-only trusts the allocation size without checking that every page is committed: it
// passes (exits 0) only if decisions differ in the partly committed cases and nowhere else.
// --mutate walk-only never uses the region information: it passes only if the call count of case 9
// fails and nothing else does. Either exits 1 otherwise. The controls are ordinary tests, not
// WILL_FAIL, so a crash or another failure cannot pass them. A refused allocation or mapping is a
// capability skip (77) in the controls and a failure in the main check.
#include "startup_stubs/placed_buffer_range.h"

#include <cstdio>
#include <cstdint>
#include <cstring>

namespace {
using x2w_startup::PlacedProof;

constexpr SIZE_T page = 0x10000;
constexpr SIZE_T block_bytes = 64u << 20;
enum Category { other, partial, cost, category_count };
PlacedProof proof = PlacedProof::exact;
Category current = other;  // the case being run, for the controls
int failures[category_count];
int setup_failures;
unsigned long long full_queries, exact_queries, placements;

void fail(const char* what, const void* address)
{
    std::fprintf(stderr, "placed buffer range: %s (address %p, error %lu)\n", what, address, GetLastError());
    ++failures[current];
}

void setup_fail(const char* what, const void* address)
{
    std::fprintf(stderr, "placed buffer range: setup: %s (address %p, error %lu)\n", what, address, GetLastError());
    ++setup_failures;
}

double now_ms()
{
    static LARGE_INTEGER frequency{};
    if (!frequency.QuadPart) QueryPerformanceFrequency(&frequency);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart * 1000.0 / frequency.QuadPart;
}

// The decision slot 133 made before the change: measure the whole committed run from the base.
struct Full {
    bool committed = false;
    void* base = nullptr;
    DWORD type = 0;
    SIZE_T bytes = 0;
    unsigned queries = 0;
};
Full full_walk(const void* address)
{
    Full r;
    MEMORY_BASIC_INFORMATION info{};
    ++r.queries;
    if (!address || VirtualQuery(address, &info, sizeof(info)) != sizeof(info) || info.State != MEM_COMMIT) return r;
    r.committed = true;
    r.base = info.AllocationBase;
    r.type = info.Type;
    MEMORY_BASIC_INFORMATION region{};
    for (auto* cursor = static_cast<char*>(r.base);
         ++r.queries, VirtualQuery(cursor, &region, sizeof(region)) && region.AllocationBase == r.base &&
                      region.State == MEM_COMMIT;
         cursor += region.RegionSize)
        r.bytes += region.RegionSize;
    return r;
}

UINT64 offset_of(const void* address, const void* base)
{
    return static_cast<const char*>(address) - static_cast<const char*>(base);
}

// One placement: the full walk and measure_placed_buffer on the same memory, the slot-133 decision
// from each, compared. expect also checks the reference when the layout is controlled (1 alias,
// 0 refuse). Pass -1 to compare the decisions on an allocator's observed layout.
bool place(void* address, UINT64 width, int expect, const char* what, x2w_startup::PlacedBufferRange* range_out = nullptr)
{
    const Full full = full_walk(address);
    x2w_startup::PlacedBufferRange range;
    const bool committed = x2w_startup::measure_placed_buffer(address, width, range, proof);
    const UINT64 offset = full.committed ? offset_of(address, full.base) : 0;
    const bool full_alias = full.committed && !(offset & 0xFFFF) && offset <= full.bytes && width <= full.bytes - offset &&
                            full.type == MEM_PRIVATE;
    const bool alias = committed && !(offset_of(address, range.base) & 0xFFFF) && range.covered &&
                       range.type == MEM_PRIVATE;
    ++placements;
    full_queries += full.queries;
    exact_queries += range.queries;
    if (range_out) *range_out = range;
    if (committed != full.committed || alias != full_alias || (committed && (range.base != full.base || range.type != full.type))) {
        std::fprintf(stderr, "  %s: full walk committed=%d base=%p type=0x%lx bytes=0x%zx alias=%d; "
                     "measured committed=%d base=%p type=0x%lx covered=%d region_info=%d alias=%d\n", what, full.committed,
                     full.base, full.type, full.bytes, full_alias, committed, range.base, range.type, range.covered,
                     range.region_info, alias);
        fail(what, address);
    } else if (expect >= 0 && static_cast<int>(alias) != expect) {
        std::fprintf(stderr, "  %s: both give alias=%d, the case expects %d\n", what, alias, expect);
        fail(what, address);
    }
    return alias;
}

char* striped_block()
{
    auto* block = static_cast<char*>(VirtualAlloc(nullptr, block_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!block) return nullptr;
    DWORD old = 0;
    for (SIZE_T offset = page; offset < block_bytes; offset += 2 * page)
        if (!VirtualProtect(block + offset, page, PAGE_READONLY, &old)) return nullptr;
    return block;
}

char* committed_pages(SIZE_T count)
{
    return static_cast<char*>(VirtualAlloc(nullptr, count * page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
}
} // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "region-only") == 0)
        proof = PlacedProof::region_only;
    else if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "walk-only") == 0)
        proof = PlacedProof::walk_only;
    else if (argc != 1) {
        std::fprintf(stderr, "usage: startup_placed_buffer_range_check [--mutate region-only|walk-only]\n");
        return 64;
    }

    // Case 1: striped block.
    if (char* block = striped_block()) {
        const unsigned long long full_before = full_queries, exact_before = exact_queries;
        for (SIZE_T i = 0; i < 256; ++i)
            place(block + i * (block_bytes / 256), (i % 4 + 1) * page, 1, "striped placement");
        std::printf("striped block (1024 regions), 256 placements: VirtualQuery calls per placement: full walk %.1f, "
                    "measured %.1f\n", (full_queries - full_before) / 256.0, (exact_queries - exact_before) / 256.0);
        place(block + 0x1000, page, 0, "striped misaligned placement");
        place(block + block_bytes - page, 2 * page, 0, "striped placement past the end");
        place(block, page, 1, "striped placement at the base");
        VirtualFree(block, 0, MEM_RELEASE);
    } else {
        setup_fail("cannot allocate the striped block", nullptr);
    }

    // Cases 2 and 3: middle-page decommit, then recommit.
    if (char* pages = committed_pages(3)) {
        place(pages, 3 * page, 1, "three pages before the decommit");
        place(pages + 2 * page, page, 1, "third page before the decommit");
        if (!VirtualFree(pages + page, page, MEM_DECOMMIT)) {
            setup_fail("cannot decommit the middle page", pages + page);
        } else {
            current = partial;
            place(pages + 2 * page, page, 0, "decommit: buffer after the gap");
            place(pages, 3 * page, 0, "decommit: buffer across the gap");
            place(pages, page, 1, "decommit: buffer before the gap");
            place(pages + page, page, 0, "decommit: buffer on the gap");
            current = other;
            if (!VirtualAlloc(pages + page, page, MEM_COMMIT, PAGE_READWRITE)) {
                setup_fail("cannot commit the middle page again", pages + page);
            } else {
                place(pages + 2 * page, page, 1, "recommit: buffer after the former gap");
                place(pages, 3 * page, 1, "recommit: buffer across the former gap");
            }
        }
        VirtualFree(pages, 0, MEM_RELEASE);
    } else {
        setup_fail("cannot commit three pages", nullptr);
    }

    // Case 4: a gap before the start, five pages before the buffer.
    if (char* pages = committed_pages(8)) {
        place(pages + 6 * page, page, 1, "eight pages before the decommit");
        if (!VirtualFree(pages + page, page, MEM_DECOMMIT)) {
            setup_fail("cannot decommit page 1", pages + page);
        } else {
            current = partial;
            place(pages + 6 * page, page, 0, "gap before the start: buffer on page 6");
            place(pages + 6 * page, 2 * page, 0, "gap before the start: buffer on pages 6 and 7");
            current = other;
        }
        VirtualFree(pages, 0, MEM_RELEASE);
    } else {
        setup_fail("cannot commit eight pages", nullptr);
    }

    // Case 4b: 1 MiB committed, buffer at +0x80000, then +0x10000 decommitted for 0x10000 bytes.
    if (char* mib = committed_pages(16)) {
        place(mib + 0x80000, 0x10000, 1, "1 MiB: buffer at +0x80000 before the decommit");
        if (!VirtualFree(mib + 0x10000, 0x10000, MEM_DECOMMIT)) {
            setup_fail("cannot decommit +0x10000", mib + 0x10000);
        } else {
            current = partial;
            place(mib + 0x80000, 0x10000, 0, "1 MiB: buffer at +0x80000 after decommitting +0x10000");
            current = other;
        }
        VirtualFree(mib, 0, MEM_RELEASE);
    } else {
        setup_fail("cannot commit 1 MiB", nullptr);
    }

    // Case 5: same-base replacement.
    if (char* block = striped_block()) {
        place(block, page, 1, "replaced: before the release");
        place(block + (512u << 10), 1u << 20, 1, "replaced: buffer ending past 1 MiB before the release");
        VirtualFree(block, 0, MEM_RELEASE);
        char* smaller = static_cast<char*>(VirtualAlloc(block, 1u << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (smaller != block) {
            setup_fail("cannot allocate the smaller block at the same base", block);
        } else {
            place(smaller + 2 * page, page, 1, "replaced: buffer inside the new block");
            place(smaller + (512u << 10), 1u << 20, 0, "replaced: buffer ending past the new end");
            place(smaller + (2u << 20), page, 0, "replaced: buffer past the new end");
        }
        if (smaller) VirtualFree(smaller, 0, MEM_RELEASE);
    } else {
        setup_fail("cannot allocate the striped block", nullptr);
    }

    // Case 6: a mapped view.
    {
        HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, 1u << 20, nullptr);
        void* view = mapping ? MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0) : nullptr;
        if (!view) {
            setup_fail("cannot map a view", nullptr);
        } else {
            place(view, page, 0, "mapped view");
            place(view, page, 0, "mapped view again");
            UnmapViewOfFile(view);
        }
        if (mapping) CloseHandle(mapping);
    }

    // Case 7: a reservation committed in two steps (the allocation stays partly committed).
    if (auto* reserved = static_cast<char*>(VirtualAlloc(nullptr, 4u << 20, MEM_RESERVE, PAGE_NOACCESS))) {
        current = partial;
        if (!VirtualAlloc(reserved, 1u << 20, MEM_COMMIT, PAGE_READWRITE)) {
            setup_fail("cannot commit the first megabyte", reserved);
        } else {
            place(reserved, page, 1, "partly committed: first megabyte");
            place(reserved + (1u << 20), page, 0, "partly committed: reserved page");
            place(reserved + (512u << 10), 1u << 20, 0, "partly committed: buffer ending in reserved pages");
            if (!VirtualAlloc(reserved + (1u << 20), 1u << 20, MEM_COMMIT, PAGE_READWRITE)) {
                setup_fail("cannot commit the second megabyte", reserved);
            } else {
                place(reserved + (1u << 20), page, 1, "partly committed: after the second commit");
                place(reserved + (512u << 10), 1u << 20, 1, "partly committed: buffer across both commits");
            }
        }
        current = other;
        VirtualFree(reserved, 0, MEM_RELEASE);
    } else {
        setup_fail("cannot reserve 4 MiB", nullptr);
    }

    // Case 8: a placeholder piece replaced by committed private memory.
    {
        using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void*, ULONG);
        HMODULE kernelbase = GetModuleHandleW(L"kernelbase.dll");
        auto alloc2 = kernelbase ? reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(kernelbase, "VirtualAlloc2")) : nullptr;
        char* holder = alloc2 ? static_cast<char*>(alloc2(nullptr, nullptr, 8 * page, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                                          PAGE_NOACCESS, nullptr, 0)) : nullptr;
        char* piece = nullptr;
        if (holder && VirtualFree(holder, 4 * page, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
            piece = static_cast<char*>(alloc2(nullptr, holder, 4 * page, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                                              PAGE_READWRITE, nullptr, 0));
        if (!piece) {
            setup_fail("cannot replace a placeholder piece", holder);
        } else {
            place(piece + page, 2 * page, 1, "placeholder piece: buffer inside");
            place(piece + 2 * page, 4 * page, 0, "placeholder piece: buffer ending in the placeholder after it");
            VirtualFree(piece, 0, MEM_RELEASE);
        }
        if (holder) VirtualFree(holder + 4 * page, 0, MEM_RELEASE);
    }

    // Case 8b: other private allocations.
    {
        if (auto* p = static_cast<char*>(VirtualAlloc(nullptr, 16 * page, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE))) {
            place(p + page, 2 * page, 1, "write-watch allocation: buffer inside");
            place(p + 15 * page, 2 * page, 0, "write-watch allocation: buffer past the end");
            VirtualFree(p, 0, MEM_RELEASE);
        } else {
            setup_fail("cannot allocate with MEM_WRITE_WATCH", nullptr);
        }
        if (auto* p = static_cast<char*>(VirtualAlloc(nullptr, 16 * page, MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN, PAGE_READWRITE))) {
            place(p + page, 2 * page, 1, "top-down allocation: buffer inside");
            VirtualFree(p, 0, MEM_RELEASE);
        } else {
            setup_fail("cannot allocate with MEM_TOP_DOWN", nullptr);
        }
        if (auto* p = static_cast<char*>(VirtualAlloc(nullptr, 16 * page, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE))) {
            place(p + page, 2 * page, 1, "executable allocation: buffer inside");
            VirtualFree(p, 0, MEM_RELEASE);
        } else {
            setup_fail("cannot allocate executable memory", nullptr);
        }
        // The heap allocator can commit the allocation's prefix. Check its actual layout against the
        // full walk and construct the reserved-prefix case separately.
        HANDLE heap = HeapCreate(0, 0, 0);
        auto* block = heap ? static_cast<char*>(HeapAlloc(heap, 0, 4u << 20)) : nullptr;
        MEMORY_BASIC_INFORMATION info{};
        if (!block || VirtualQuery(block, &info, sizeof(info)) != sizeof(info) || info.State != MEM_COMMIT) {
            setup_fail("cannot allocate a 4 MiB heap block", block);
        } else {
            auto* base = static_cast<char*>(info.AllocationBase);
            current = partial;
            place(base + page, page, -1, "heap block near its allocation base");
            place(base + page, 5u << 20, -1, "heap block span wider than the heap request");
            current = other;
        }
        if (heap) HeapDestroy(heap);
        auto* reservation = static_cast<char*>(VirtualAlloc(nullptr, 2 * page, MEM_RESERVE, PAGE_READWRITE));
        if (reservation && VirtualAlloc(reservation + page, page, MEM_COMMIT, PAGE_READWRITE)) {
            current = partial;
            place(reservation + page, page, 0, "committed suffix after a reserved prefix");
            place(reservation + page, 2 * page, 0, "committed suffix with buffer past the allocation");
            current = other;
        } else {
            setup_fail("cannot construct an allocation with a reserved prefix", reservation);
        }
        if (reservation) VirtualFree(reservation, 0, MEM_RELEASE);
        if (auto* p = static_cast<char*>(VirtualAlloc(nullptr, 16 * page, MEM_RESERVE | MEM_PHYSICAL, PAGE_READWRITE))) {
            current = partial;
            place(p, page, 0, "AWE reservation: nothing committed");
            current = other;
            VirtualFree(p, 0, MEM_RELEASE);
        } else {
            setup_fail("cannot reserve with MEM_PHYSICAL", nullptr);
        }
        const SIZE_T large = GetLargePageMinimum();
        auto* p = large ? static_cast<char*>(VirtualAlloc(nullptr, large, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                                                          PAGE_READWRITE)) : nullptr;
        if (p) {
            place(p, page, 1, "large-page allocation: buffer at the base");
            place(p + large - page, 2 * page, 0, "large-page allocation: buffer past the end");
            VirtualFree(p, 0, MEM_RELEASE);
        } else {
            std::printf("large-page allocation not run (error %lu: the process may not lock pages)\n", GetLastError());
        }
    }

    // Case 9: cost on a large committed block with touched pages.
    {
        const SIZE_T large = SIZE_T{512} << 20;
        auto* block = static_cast<char*>(VirtualAlloc(nullptr, large, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!block) {
            setup_fail("cannot commit 512 MiB", nullptr);
        } else {
            for (SIZE_T offset = 0; offset < large; offset += page) block[offset] = 1;
            int proven = 0;
            unsigned queries = 0;
            const int count = 64;
            double t = now_ms();
            for (int i = 0; i < count; ++i) full_walk(block + i * (large / count));
            const double full_ms = (now_ms() - t) / count;
            t = now_ms();
            x2w_startup::PlacedBufferRange bounded;
            for (int i = 0; i < count; ++i)
                x2w_startup::measure_placed_buffer(block + i * (large / count), page, bounded, PlacedProof::walk_only);
            const double bounded_ms = (now_ms() - t) / count;
            t = now_ms();
            for (int i = 0; i < count; ++i) {
                x2w_startup::PlacedBufferRange range;
                x2w_startup::measure_placed_buffer(block + i * (large / count), page, range, proof);
                proven += range.region_info;
                queries += range.queries;
            }
            const double measured_ms = (now_ms() - t) / count;
            std::printf("512 MiB touched block, %d placements: ms per placement: full walk %.3f, bounded walk %.3f, "
                        "measured %.4f; region-information proofs %d, VirtualQuery calls %u\n", count, full_ms, bounded_ms,
                        measured_ms, proven, queries);
            for (int i = 0; i < count; ++i) place(block + i * (large / count), page, 1, "512 MiB block placement");
            current = cost;
            if (proven != count || queries != 0) fail("512 MiB block: placements not all proven without VirtualQuery", block);
            current = other;
            VirtualFree(block, 0, MEM_RELEASE);
        }
    }

    std::printf("all %llu placements: VirtualQuery calls full walk %llu, measured %llu\n", placements, full_queries,
                exact_queries);
    const int total = failures[other] + failures[partial] + failures[cost];
    if (proof != PlacedProof::exact) {
        if (setup_failures) return 77;
        const bool stated = proof == PlacedProof::region_only
                                ? failures[partial] > 0 && failures[other] == 0 && failures[cost] == 0
                                : failures[cost] > 0 && failures[other] == 0 && failures[partial] == 0;
        std::fprintf(stderr, "placed buffer range control: failures other %d, partly committed %d, cost %d -> %s\n",
                     failures[other], failures[partial], failures[cost],
                     stated ? "failed for the stated reason" : "did NOT fail for the stated reason");
        return stated ? 0 : 1;
    }
    if (total || setup_failures) {
        std::fprintf(stderr, "placed buffer range: %d failure(s), %d setup failure(s)\n", total, setup_failures);
        return 1;
    }
    std::printf("placed buffer range: every decision matches the full walk\n");
    return 0;
}
