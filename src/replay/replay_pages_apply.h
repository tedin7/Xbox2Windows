// Memory pages for replay (docs/startup-runtime.md#d3d12-replay-record): the title memory the recording captured as
// PAGE/MAP chunks (src/startup_stubs/replay_pages.h) is reserved at its recorded addresses before anything else maps
// there, and every MAP is written into it before the first call with a larger sequence number. Argument memory inside
// those ranges is then passed as is: placed resources and mapped pointers name the replayed memory.
#pragma once

#include <windows.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "replay_file.h"

namespace x2w_replay {

struct ReservedSpan {
    u64 base, end;
};

inline bool span_contains(const std::vector<ReservedSpan>& spans, u64 address)
{
    const auto at = std::upper_bound(spans.begin(), spans.end(), address,
                                     [](u64 a, const ReservedSpan& s) { return a < s.base; });
    return at != spans.begin() && address < std::prev(at)->end;
}

struct PageReplay {
    std::vector<ReservedSpan> spans;  // sorted, disjoint, 64 KiB aligned
    std::vector<ReservedSpan> map_spans;  // only ranges whose contents are restored by MAP
    std::size_t next_map = 0;
    u64 maps_applied = 0, pages_written = 0, zero_pages = 0, missing_pages = 0, bad_maps = 0;
};

inline PageReplay& page_replay()
{
    static PageReplay state;
    return state;
}

inline void merge_spans(std::vector<ReservedSpan>& spans)
{
    std::sort(spans.begin(), spans.end(), [](const ReservedSpan& a, const ReservedSpan& b) { return a.base < b.base; });
    std::size_t count = 0;
    for (const ReservedSpan s : spans) {
        if (count && s.base <= spans[count - 1].end) spans[count - 1].end = (std::max)(spans[count - 1].end, s.end);
        else spans[count++] = s;
    }
    spans.resize(count);
}

// Reserves and commits MAP and descriptor backing ranges at their recorded addresses.
inline bool reserve_page_ranges(const Recording& r, std::string& problem)
{
    static constexpr u64 granule = 0x10000;
    const auto aligned = [](u64 address, u64 bytes) -> ReservedSpan {
        return {address & ~(granule - 1), (address + bytes + granule - 1) & ~(granule - 1)};
    };
    auto& state = page_replay();
    for (const PageMap& m : r.maps)
        if (m.bytes) state.map_spans.push_back(aligned(m.address, m.bytes));
    merge_spans(state.map_spans);
    std::vector<ReservedSpan> wanted = state.map_spans;
    for (const Step& step : r.steps)
        if (step.descriptor_bytes) wanted.push_back(aligned(step.descriptor_address, step.descriptor_bytes));
    // Reserve whole recorded allocations, even when only interior pages were captured.
    std::vector<u64> boundaries;
    for (const auto& entry : r.placement_heaps) {
        const HeapPlacement& heap = entry.second;
        const u64 end = (heap.base + heap.bytes + granule - 1) & ~(granule - 1);
        wanted.push_back({heap.base, end});
        boundaries.push_back(heap.base);
        boundaries.push_back(end);
    }
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    merge_spans(wanted);
    // Do not let adjacent page ranges merge two different recorded heaps into one allocation.
    state.spans.clear();
    for (const ReservedSpan& s : wanted) {
        u64 base = s.base;
        for (auto cut = std::upper_bound(boundaries.begin(), boundaries.end(), base);
             cut != boundaries.end() && *cut < s.end; ++cut) {
            state.spans.push_back({base, *cut});
            base = *cut;
        }
        state.spans.push_back({base, s.end});
    }
    for (const ReservedSpan& s : state.spans) {
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
// Where a recorded page is on the replay: its address there, with `bytes` clipped to the translated range. The
// recorded address itself when the page is not in a translated range.
using PageTarget = std::function<u64(u64 address, u64& bytes)>;

inline void write_page(const PageTarget& target, u64 at, const unsigned char* data)
{
    u64 bytes = page_bytes;
    auto* to = reinterpret_cast<unsigned char*>(target ? target(at, bytes) : at);
    if (data) std::memcpy(to, data, static_cast<std::size_t>(bytes));
    else std::memset(to, 0, static_cast<std::size_t>(bytes));
}

inline void apply_map(const Recording& r, const PageMap& m, const PageTarget& target)
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
            const u64 page = m.address + (u64{run.first} + i) * page_bytes;
            if (run.kind == 1) {
                write_page(target, page, nullptr);
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
            write_page(target, page, found->second);
            ++state.pages_written;
        }
    }
    ++state.maps_applied;
}

// Before the call with sequence number `seq`: every MAP recorded below it that is not written yet.
inline void apply_pages_before(const Recording& r, u64 seq, const PageTarget& target = {})
{
    PageReplay& state = page_replay();
    while (state.next_map < r.maps.size() && r.maps[state.next_map].seq < seq) apply_map(r, r.maps[state.next_map++], target);
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
