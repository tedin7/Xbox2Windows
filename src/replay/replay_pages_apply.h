// Memory pages for replay (docs/startup-runtime.md#d3d12-replay-record): the title memory the recording captured as
// PAGE/MAP chunks (src/startup_stubs/replay_pages.h) is reserved at its recorded addresses before anything else maps
// there, and every MAP is written into it before the first call with a larger sequence number. Argument memory inside
// those ranges is then passed as is: placed resources and mapped pointers name the replayed memory.
#pragma once

#include <windows.h>

#include <algorithm>
#include <string>
#include <vector>

#include "replay_file.h"

namespace x2w_replay {

struct ReservedSpan {
    u64 base, end;  // 64 KiB aligned, as VirtualAlloc reserves
};

struct PageReplay {
    std::vector<ReservedSpan> spans;  // sorted, disjoint
    std::size_t next_map = 0;
    u64 maps_applied = 0, pages_written = 0, zero_pages = 0, missing_pages = 0, bad_maps = 0;

    bool reserved(u64 address) const
    {
        const auto at = std::upper_bound(spans.begin(), spans.end(), address,
                                         [](u64 a, const ReservedSpan& s) { return a < s.base; });
        return at != spans.begin() && address < std::prev(at)->end;
    }
};

inline PageReplay& page_replay()
{
    static PageReplay state;
    return state;
}

// Reserves and commits every MAP range at its recorded address. False with the address that could not be had.
inline bool reserve_page_ranges(const Recording& r, std::string& problem)
{
    constexpr u64 granule = 0x10000;
    std::vector<ReservedSpan> wanted;
    for (const PageMap& m : r.maps)
        if (m.bytes) wanted.push_back({m.address & ~(granule - 1), (m.address + m.bytes + granule - 1) & ~(granule - 1)});
    std::sort(wanted.begin(), wanted.end(), [](const ReservedSpan& a, const ReservedSpan& b) { return a.base < b.base; });
    std::vector<ReservedSpan>& spans = page_replay().spans;
    for (const ReservedSpan& s : wanted) {
        if (!spans.empty() && s.base <= spans.back().end) spans.back().end = (std::max)(spans.back().end, s.end);
        else spans.push_back(s);
    }
    for (const ReservedSpan& s : spans) {
        void* const at = VirtualAlloc(reinterpret_cast<void*>(s.base), static_cast<SIZE_T>(s.end - s.base), MEM_RESERVE | MEM_COMMIT,
                                      PAGE_READWRITE);
        if (at != reinterpret_cast<void*>(s.base)) {
            char text[160];
            std::snprintf(text, sizeof(text), "title range 0x%llx-0x%llx could not be reserved at its address (error %lu)",
                          s.base, s.end, GetLastError());
            problem = text;
            if (at) VirtualFree(at, 0, MEM_RELEASE);
            return false;
        }
    }
    return true;
}

// Writes one MAP's runs into the reserved memory.
inline void apply_map(const Recording& r, const PageMap& m)
{
    PageReplay& state = page_replay();
    const unsigned char* p = m.data;
    u64 left = m.data_bytes;
    for (std::uint32_t n = 0; n < m.runs; ++n) {
        RunHeader run;
        if (left < sizeof(run)) {
            ++state.bad_maps;
            return;
        }
        std::memcpy(&run, p, sizeof(run));
        p += sizeof(run), left -= sizeof(run);
        if (u64{run.first} + run.pages > m.bytes / page_bytes) {
            ++state.bad_maps;
            return;
        }
        for (std::uint32_t i = 0; i < run.pages; ++i) {
            auto* page = reinterpret_cast<unsigned char*>(m.address + (u64{run.first} + i) * page_bytes);
            if (run.kind == 1) {
                std::memset(page, 0, page_bytes);
                ++state.zero_pages;
                continue;
            }
            if (left < sizeof(PageKey)) {
                ++state.bad_maps;
                return;
            }
            PageKey key;
            std::memcpy(&key, p, sizeof(key));
            p += sizeof(key), left -= sizeof(key);
            const auto found = r.pages.find(key);
            if (found == r.pages.end()) {
                ++state.missing_pages;
                continue;
            }
            std::memcpy(page, found->second, page_bytes);
            ++state.pages_written;
        }
    }
    ++state.maps_applied;
}

// Before the call with sequence number `seq`: every MAP recorded below it that is not written yet.
inline void apply_pages_before(const Recording& r, u64 seq)
{
    PageReplay& state = page_replay();
    while (state.next_map < r.maps.size() && r.maps[state.next_map].seq < seq) apply_map(r, r.maps[state.next_map++]);
}

// FNV-1a of a reserved span's bytes, for the summary (a replay's image can be compared with the recorded one).
inline u64 span_hash(const ReservedSpan& s)
{
    u64 hash = 1469598103934665603ull;
    for (const auto* p = reinterpret_cast<const unsigned char*>(s.base); p < reinterpret_cast<const unsigned char*>(s.end); ++p)
        hash = (hash ^ *p) * 1099511628211ull;
    return hash;
}

}  // namespace x2w_replay
