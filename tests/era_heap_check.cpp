#include "era_heap.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdio>
#include <limits>
#include <memory>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>
#ifdef _DEBUG
#include <crtdbg.h>
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

using namespace std::literals;

namespace {
// Attribute words recorded at the Bink fallback and AMD upscaler XMemAlloc call sites.
constexpr std::uint64_t bink_word = 0x05e50000;
constexpr std::uint64_t amd_word = 0xa8004000;
constexpr std::uint64_t memory_type_mask = std::uint64_t{15} << 29;

std::size_t allocation_granularity()
{
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return info.dwAllocationGranularity;
}

bool rejects(std::size_t bytes, std::uint64_t attributes)
{
    void* p = era_heap::allocate(bytes, attributes);
    if (p) {
        era_heap::release_with_attributes(p, attributes);
        return false;
    }
    return true;
}
}

bool allocation_checks()
{
    constexpr std::size_t bytes = 257;
#ifdef _DEBUG
    _CrtMemState before{}, after{}, change{};
    _CrtMemCheckpoint(&before);
#endif
    for (unsigned exponent : {0u, 3u, 4u, 6u, 12u, 16u}) {
        auto* p = static_cast<unsigned char*>(
            era_heap::allocate(bytes, static_cast<std::uint64_t>(exponent) << 24));
        if (!p) {
            return false;
        }
        const auto alignment = std::size_t{1} << std::max(4u, exponent);
        const bool aligned = reinterpret_cast<std::uintptr_t>(p) % alignment == 0;
        const bool zeroed = std::all_of(p, p + bytes, [](auto b) { return b == 0; });
        std::fill_n(p, bytes, 0xa5);
        era_heap::release(p);
        if (!aligned || !zeroed) {
            return false;
        }
    }
    era_heap::release(nullptr);
#ifdef _DEBUG
    _CrtMemCheckpoint(&after);
    return !_CrtMemDifference(&change, &before, &after);
#else
    return true;
#endif
}

bool failure_checks()
{
    // Mix each of the 30 reserved bits with a valid heap alignment and with the AMD graphics word.
    for (unsigned bit = 34; bit < 64; ++bit) {
        const auto reserved = std::uint64_t{1} << bit;
        if (!rejects(257, reserved | (std::uint64_t{6} << 24)) || !rejects(512, reserved | amd_word)) {
            return false;
        }
    }
    // The address-below-4-GiB bit is valid but this allocator does not provide such an address.
    if (!rejects(257, (std::uint64_t{1} << 33) | (std::uint64_t{6} << 24)) ||
        !rejects(512, (std::uint64_t{1} << 33) | amd_word)) {
        return false;
    }
    // Physical memory types 7-9 are not modeled. Types 10-15 are undefined.
    for (std::uint64_t type = 7; type < 16; ++type) {
        if (!rejects(512, (amd_word & ~memory_type_mask) | (type << 29))) return false;
    }
    // Graphics memory with 2 MB pages or a 128 KiB alignment exceeds the desktop
    // allocation granularity. The undefined page size value 3 fails for every type.
    const auto two_mb_pages = (amd_word & ~(std::uint64_t{3} << 14)) | (std::uint64_t{2} << 14);
    const auto wide_alignment = (amd_word & ~(std::uint64_t{31} << 24)) | (std::uint64_t{17} << 24);
    if ((allocation_granularity() < (std::size_t{2} << 20) && !rejects(512, two_mb_pages)) ||
        (allocation_granularity() < (std::size_t{1} << 17) && !rejects(512, wide_alignment)) ||
        !rejects(512, amd_word | (std::uint64_t{3} << 14)) ||
        !rejects(64, bink_word | (std::uint64_t{3} << 14))) {
        return false;
    }
    return rejects(0, 0) && rejects(0, amd_word) &&
           rejects(std::numeric_limits<std::size_t>::max(), 0) &&
           rejects(std::numeric_limits<std::size_t>::max(), amd_word);
}

bool observed_word_checks()
{
    using Byte = unsigned char;
    const auto bink = era_heap::decode(bink_word);
    const auto amd = era_heap::decode(amd_word);
    if (bink.object_type != 0 || bink.page_size != 0 || bink.allocator_id != 0xe5 ||
        bink.alignment != 5 || bink.memory_type != 0 || bink.reserved != 0 || bink.address_below_4gib != 0 ||
        amd.object_type != 0 || amd.page_size != 1 || amd.allocator_id != 0 ||
        amd.alignment != 8 || amd.memory_type != 5 || amd.reserved != 0 || amd.address_below_4gib != 0) {
        return false;
    }

    // Bink requests the payload length plus 64 and builds its header in front of the payload.
    constexpr Byte marker = 0x51, tag = 0x34;
    for (std::size_t length : {33u, 129u}) {
        const auto total = length + 64;
        auto* base = static_cast<Byte*>(era_heap::allocate(total, bink_word));
        if (!base) return false;
        const auto address = reinterpret_cast<std::uintptr_t>(base);
        bool ok = address % 32 == 0 && std::all_of(base, base + total, [](Byte b) { return b == 0; });
        const auto offset = static_cast<Byte>(64 - (address & 31));
        auto* user = base + offset;
        ok = ok && reinterpret_cast<std::uintptr_t>(user) % 32 == 0 && user + length <= base + total;
        if (ok) {
            user[-2] = tag;
            user[-1] = offset;
            std::fill_n(user, length, marker);
            ok = user - user[-1] == base && user[-2] == tag &&
                 std::all_of(user, user + length, [](Byte b) { return b == marker; });
        }
        era_heap::release_with_attributes(base, bink_word);
        if (!ok) return false;
    }

    // AMD rounds each nonzero request up to a multiple of 256 before calling XMemAlloc.
    const auto granularity = allocation_granularity();
    for (std::size_t requested : {1u, 257u, 4097u, 65537u}) {
        const auto rounded = (requested + 255) & ~std::size_t{255};
        auto* p = static_cast<Byte*>(era_heap::allocate(rounded, amd_word));
        if (!p) return false;
        MEMORY_BASIC_INFORMATION info{};
        bool ok = VirtualQuery(p, &info, sizeof(info)) == sizeof(info) &&
                  info.AllocationBase == p && info.State == MEM_COMMIT &&
                  info.Protect == PAGE_READWRITE && info.RegionSize >= rounded &&
                  reinterpret_cast<std::uintptr_t>(p) % (std::size_t{64} << 10) == 0 &&
                  reinterpret_cast<std::uintptr_t>(p) % granularity == 0 &&
                  std::all_of(p, p + rounded, [](Byte b) { return b == 0; });
        if (ok) {
            std::fill_n(p, rounded, marker);
            ok = std::all_of(p, p + rounded, [](Byte b) { return b == marker; });
        }
        era_heap::release_with_attributes(p, amd_word);
        MEMORY_BASIC_INFORMATION after{};
        if (!ok || VirtualQuery(p, &after, sizeof(after)) != sizeof(after) || after.State != MEM_FREE) {
            return false;
        }
    }

    // Object type and allocator ID are tags and do not change heap placement.
    const auto tagged = bink_word | 0x1234 | (std::uint64_t{0x7f} << 16);
    auto* p = era_heap::allocate(97, tagged);
    const bool tagged_ok = p && reinterpret_cast<std::uintptr_t>(p) % 32 == 0;
    era_heap::release_with_attributes(p, tagged);
    if (!tagged_ok) return false;

    std::printf("bink_blocks=2 amd_blocks=4 granularity=%zu\n", granularity);
    return true;
}

bool caller_layout_checks()
{
    using Byte = unsigned char;
    using Block = std::unique_ptr<void, decltype(&era_heap::release)>;
    constexpr Byte guard = 0xec, marker = 0x51, tag = 0x34;
    // Supply owned storage with a known origin for each possible base address residue.
    for (std::size_t length : {1u, 31u, 32u, 33u, 255u, 256u, 257u, 4097u}) {
        const auto total = length + 64 + 96;
        Block block(era_heap::allocate(total, bink_word), era_heap::release);
        if (!block) return false;
        auto* storage = static_cast<Byte*>(block.get());
        for (unsigned residue = 0; residue < 32; ++residue) {
            std::fill_n(storage, total, guard);
            auto* base = storage + 32 + residue;
            const auto address = reinterpret_cast<std::uintptr_t>(base);
            const auto offset = static_cast<Byte>(64 - (address & 31));
            auto* user = base + offset;
            if ((address & 31) != residue || reinterpret_cast<std::uintptr_t>(user) % 32 ||
                offset < 33 || user + length > base + length + 64) return false;
            user[-2] = tag;
            user[-1] = offset;
            std::fill_n(user, length, marker);
            if (user - user[-1] != base || user[-2] != tag ||
                !std::all_of(storage, user - 2, [](Byte b) { return b == guard; }) ||
                !std::all_of(user + length, storage + total, [](Byte b) { return b == guard; }))
                return false;
        }
    }
    // AMD rounds the size. These ordinary heap blocks impose no 256-byte pointer alignment.
    struct SizeCase { std::size_t requested, allocated; };
    constexpr SizeCase sizes[]{{1,256}, {255,256}, {256,256}, {257,512}, {511,512},
                               {512,512}, {513,768}, {4095,4096}, {4096,4096}, {4097,4352}};
    for (const auto& size : sizes) {
        const auto rounded = (size.requested + 255) & ~std::size_t{255};
        if (rounded != size.allocated) return false;
        Block block(era_heap::allocate(rounded, 0), era_heap::release);
        if (!block) return false;
        auto* data = static_cast<Byte*>(block.get());
        std::fill_n(data, size.requested, marker);
        if (!std::all_of(data, data + size.requested, [](Byte b) { return b == marker; }) ||
            !std::all_of(data + size.requested, data + rounded, [](Byte b) { return b == 0; })) return false;
    }
    std::puts("bink_base_layouts=256 amd_size_boundaries=10");
    return true;
}

bool reuse_checks()
{
    using Byte = unsigned char;
    using Block = std::unique_ptr<void, decltype(&era_heap::release)>;
    constexpr Byte dirty = 0xa5;
    // A released block keeps its contents in the CRT heap. Dirty one, then hold up to 32
    // fresh blocks of the same shape until that address comes back. Each must read as zero.
    unsigned cases = 0, recycled = 0;
    for (unsigned exponent : {4u, 6u, 12u}) {
        for (std::size_t bytes : {64u, 257u, 4096u}) {
            const auto flags = std::uint64_t{exponent} << 24;
            auto* p = static_cast<Byte*>(era_heap::allocate(bytes, flags));
            if (!p) return false;
            std::fill_n(p, bytes, dirty);
            const auto released = reinterpret_cast<std::uintptr_t>(p);
            era_heap::release(p);
            std::vector<Block> held;
            for (int attempt = 0; attempt < 32; ++attempt) {
                held.emplace_back(era_heap::allocate(bytes, flags), era_heap::release);
                auto* q = static_cast<Byte*>(held.back().get());
                if (!q || !std::all_of(q, q + bytes, [](Byte b) { return b == 0; })) return false;
                if (reinterpret_cast<std::uintptr_t>(q) == released) {
                    ++recycled;
                    break;
                }
            }
            ++cases;
        }
    }
    std::printf("reuse_cases=%u recycled_address=%u\n", cases, recycled);
    return true;
}

bool large_alignment_checks()
{
    using Byte = unsigned char;
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    const std::size_t page = system.dwPageSize;
    // Alignments above one page. Each block must own its allocation and commit only the
    // pages that hold the payload, not the alignment padding.
    unsigned cases = 0;
    std::size_t committed_total = 0, alignment_total = 0;
    for (unsigned exponent : {13u, 16u, 17u, 20u, 31u}) {
        const auto alignment = std::size_t{1} << exponent;
        for (std::size_t bytes : {std::size_t{1}, std::size_t{257}, page, page + 1, 3 * page - 1}) {
            auto* p = static_cast<Byte*>(era_heap::allocate(bytes, std::uint64_t{exponent} << 24));
            if (!p) return false;
            const auto committed = (bytes + page - 1) & ~(page - 1);
            MEMORY_BASIC_INFORMATION info{};
            const bool shaped =
                reinterpret_cast<std::uintptr_t>(p) % alignment == 0 &&
                std::all_of(p, p + bytes, [](Byte b) { return b == 0; }) &&
                VirtualQuery(p, &info, sizeof(info)) == sizeof(info) &&
                info.AllocationBase == p && info.BaseAddress == p && info.State == MEM_COMMIT &&
                info.Protect == PAGE_READWRITE && info.Type == MEM_PRIVATE &&
                info.RegionSize == committed;
            if (!shaped) {
                std::printf("exponent=%u bytes=%zu base_matches=%d state=%lx region=%zu\n", exponent,
                            bytes, info.AllocationBase == p, static_cast<unsigned long>(info.State),
                            static_cast<std::size_t>(info.RegionSize));
                era_heap::release(p);
                return false;
            }
            std::fill_n(p, bytes, 0xa5);
            era_heap::release(p);
            if (VirtualQuery(p, &info, sizeof(info)) != sizeof(info) || info.State != MEM_FREE) {
                std::printf("exponent=%u bytes=%zu not released (state=%lx)\n", exponent, bytes,
                            static_cast<unsigned long>(info.State));
                return false;
            }
            committed_total += committed;
            alignment_total += alignment;
            ++cases;
        }
    }
    std::printf("large_alignment_cases=%u committed_bytes=%zu alignment_bytes=%zu\n", cases,
                committed_total, alignment_total);
    return true;
}

bool graphics_type_checks()
{
    using Byte = unsigned char;
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    const std::size_t page = system.dwPageSize;
    const std::size_t granularity = system.dwAllocationGranularity;
    const auto word = [](std::uint64_t type, std::uint64_t page_size, std::uint64_t exponent) {
        return (type << 29) | (exponent << 24) | (page_size << 14);
    };
    // Every graphics memory type is served from its own virtual allocation when the page size
    // and alignment fit the allocation granularity. Exponent 16 is the 64 KiB boundary itself.
    unsigned served = 0, rejected = 0;
    for (std::uint64_t type = 1; type <= 6; ++type) {
        for (std::uint64_t page_size : {0u, 1u}) {
            for (std::uint64_t exponent : {0u, 4u, 8u, 16u}) {
                if ((std::size_t{1} << exponent) > granularity) continue;
                const auto attributes = word(type, page_size, exponent);
                for (std::size_t bytes : {std::size_t{1}, granularity + 1}) {
                    auto* p = static_cast<Byte*>(era_heap::allocate(bytes, attributes));
                    if (!p) {
                        std::printf("type=%llu page_size=%llu exponent=%llu bytes=%zu refused\n",
                                    type, page_size, exponent, bytes);
                        return false;
                    }
                    const auto committed = (bytes + page - 1) & ~(page - 1);
                    MEMORY_BASIC_INFORMATION info{};
                    bool ok = reinterpret_cast<std::uintptr_t>(p) % granularity == 0 &&
                              VirtualQuery(p, &info, sizeof(info)) == sizeof(info) &&
                              info.AllocationBase == p && info.State == MEM_COMMIT &&
                              info.Protect == PAGE_READWRITE && info.Type == MEM_PRIVATE &&
                              info.RegionSize >= committed &&
                              std::all_of(p, p + bytes, [](Byte b) { return b == 0; });
                    if (ok) {
                        std::fill_n(p, bytes, Byte{0x5a});
                        ok = std::all_of(p, p + bytes, [](Byte b) { return b == 0x5a; });
                    }
                    era_heap::release_with_attributes(p, attributes);
                    MEMORY_BASIC_INFORMATION after{};
                    if (!ok || VirtualQuery(p, &after, sizeof(after)) != sizeof(after) ||
                        after.State != MEM_FREE) {
                        std::printf("type=%llu page_size=%llu exponent=%llu bytes=%zu ok=%d state=%lx\n",
                                    type, page_size, exponent, bytes, ok,
                                    static_cast<unsigned long>(after.State));
                        return false;
                    }
                    ++served;
                }
            }
        }
        // Negative controls for the same type: 4 MB pages and alignments one step above the
        // granularity exceed what the desktop honors, and page size 3 is undefined.
        for (const auto& [page_size, exponent, applies] :
             {std::tuple{std::uint64_t{2}, std::uint64_t{0}, granularity < (std::size_t{4} << 20)},
              std::tuple{std::uint64_t{0}, std::uint64_t{17}, granularity < (std::size_t{1} << 17)},
              std::tuple{std::uint64_t{3}, std::uint64_t{0}, true}}) {
            if (!applies) continue;
            if (!rejects(512, word(type, page_size, exponent))) {
                std::printf("type=%llu page_size=%llu exponent=%llu accepted\n", type, page_size, exponent);
                return false;
            }
            ++rejected;
        }
    }
    std::printf("graphics_served=%u graphics_rejected=%u granularity=%zu\n", served, rejected, granularity);
    return true;
}

bool concurrency_checks()
{
    constexpr std::size_t bytes = 511;
    std::array<std::thread, 4> workers;
    std::barrier rendezvous(static_cast<std::ptrdiff_t>(workers.size()));
    std::atomic<bool> ok{true};
    for (unsigned i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, i] {
            const auto marker = static_cast<unsigned char>(i + 1);
            for (int round = 0; round < 100; ++round) {
                auto* p = static_cast<unsigned char*>(era_heap::allocate(bytes, 6ull << 24));
                if (p) {
                    std::fill_n(p, bytes, marker);
                }
                rendezvous.arrive_and_wait();
                if (!p || !std::all_of(p, p + bytes, [marker](auto b) { return b == marker; })) {
                    ok = false;
                }
                rendezvous.arrive_and_wait();
                era_heap::release(p);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return ok.load();
}

// Every field of the XALLOC_ATTRIBUTES layout (objectType:14, pageSize:2, allocatorId:8, alignment:5, memoryType:4,
// address-below-4-GiB:1, reserved:30) at its widest value alone, all at once, and one word with a distinct value
// in each field. The
// observed words leave the object type and the reserved bits at 0, so a narrowed mask or a shifted field boundary
// would decode them unchanged. narrow_object_type expects a 13-bit object type and is the negative control.
bool decode_field_checks(bool narrow_object_type)
{
    struct Case {
        const char* name;
        std::uint64_t word;
        era_heap::Fields fields;
    };
    constexpr std::uint64_t mixed = 0x2a5cull | 2ull << 14 | 0xb7ull << 16 | 0x13ull << 24 | 9ull << 29 |
                                    1ull << 33 | 0x2a5a5a5ull << 34;
    const Case cases[] = {
        {"object type", 0x3fff, {0x3fff, 0, 0, 0, 0, 0, 0}},
        {"page size", 3ull << 14, {0, 3, 0, 0, 0, 0, 0}},
        {"allocator", 0xffull << 16, {0, 0, 0xff, 0, 0, 0, 0}},
        {"alignment", 0x1full << 24, {0, 0, 0, 0x1f, 0, 0, 0}},
        {"memory type", 0xfull << 29, {0, 0, 0, 0, 0xf, 0, 0}},
        {"address 32-bit mapping", 1ull << 33, {0, 0, 0, 0, 0, 1, 0}},
        {"reserved", 0x3fffffffull << 34, {0, 0, 0, 0, 0, 0, 0x3fffffff}},
        {"all ones", ~0ull, {0x3fff, 3, 0xff, 0x1f, 0xf, 1, 0x3fffffff}},
        {"mixed", mixed, {0x2a5c, 2, 0xb7, 0x13, 9, 1, 0x2a5a5a5}},
    };
    bool ok = true;
    for (const auto& c : cases) {
        auto want = c.fields;
        if (narrow_object_type) {
            want.object_type &= 0x1fff;
        }
        const auto got = era_heap::decode(c.word);
        if (got.object_type != want.object_type || got.page_size != want.page_size ||
            got.allocator_id != want.allocator_id || got.alignment != want.alignment ||
            got.memory_type != want.memory_type || got.reserved != want.reserved ||
            got.address_below_4gib != want.address_below_4gib) {
            std::printf("FAIL: decode %s: %x %u %x %u %u %x %u\n", c.name, got.object_type, got.page_size,
                        got.allocator_id, got.alignment, got.memory_type,
                        got.reserved, got.address_below_4gib);
            ok = false;
        }
    }
    return ok;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        return 2;
    }
    const std::string_view scenario = argv[1];
    bool passed;
    if (scenario == "allocation"sv) {
        passed = allocation_checks();
    } else if (scenario == "failure"sv) {
        passed = failure_checks();
    } else if (scenario == "caller_layout"sv) {
        passed = caller_layout_checks();
    } else if (scenario == "reuse"sv) {
        passed = reuse_checks();
    } else if (scenario == "observed_words"sv) {
        passed = observed_word_checks();
    } else if (scenario == "large_alignment"sv) {
        passed = large_alignment_checks();
    } else if (scenario == "graphics_types"sv) {
        passed = graphics_type_checks();
    } else if (scenario == "concurrency"sv) {
        passed = concurrency_checks();
    } else if (scenario == "decode_fields"sv) {
        passed = decode_field_checks(false);
    } else if (scenario == "decode_fields_control"sv) {
        passed = decode_field_checks(true);
    } else {
        return 2;
    }
    std::printf("Desktop ERA heap experiment %s: %s\n", argv[1], passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
