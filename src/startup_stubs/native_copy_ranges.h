#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace x2w_startup {
enum class CopyCoverage { none, partial, full };

// A union of disjoint half-open offsets in one placement. Callers validate
// the offsets against the buffer size before adding or looking up a range.
struct NativeCopyRanges {
    struct Span { std::uint64_t begin, end; };
    std::vector<Span> spans;

    void add(std::uint64_t begin, std::uint64_t end) {
        if (begin >= end) return;
        auto first = std::lower_bound(spans.begin(), spans.end(), begin,
            [](const Span& row, std::uint64_t start) { return row.end < start; });
        auto last = first;
        while (last != spans.end() && last->begin <= end) {
            begin = (std::min)(begin, last->begin);
            end = (std::max)(end, last->end);
            ++last;
        }
        spans.insert(spans.erase(first, last), {begin, end});
    }

    void cpu_write(std::uint64_t begin, std::uint64_t end) {
        if (begin >= end) return;
        for (auto it = spans.begin(); it != spans.end() && it->begin < end;) {
            if (it->end <= begin) { ++it; continue; }
            const Span old = *it;
            if (old.begin < begin) {
                it->end = begin;
                ++it;
                if (old.end > end) { spans.insert(it, {end, old.end}); return; }
            } else if (old.end > end) {
                it->begin = end;
                return;
            } else {
                it = spans.erase(it);
            }
        }
    }

    CopyCoverage coverage(std::uint64_t begin, std::uint64_t end) const {
        if (begin >= end) return CopyCoverage::none;
        auto row = std::lower_bound(spans.begin(), spans.end(), begin,
            [](const Span& s, std::uint64_t start) { return s.end <= start; });
        if (row == spans.end() || row->begin >= end) return CopyCoverage::none;
        return row->begin <= begin && end <= row->end ? CopyCoverage::full : CopyCoverage::partial;
    }
};
}
