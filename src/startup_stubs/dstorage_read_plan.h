// Read planning for the DirectStorage stand-in's HDD mode (X2W_STORAGE_HDD=1): from the file reads the next queued requests
// will make, work out the order and the merged reads that move the disk head forward through one file at a time.
// Pure functions (standard library only), so the plan is tested without a disk or a queue.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace x2w_startup {

struct PlannedRead {
    std::uintptr_t file = 0;       // the file's identity: a number unique to the open (a handle value can be reused)
    std::uintptr_t handle = 0;     // the handle to read it through (not part of the identity)
    std::uint64_t offset = 0;
    std::uint32_t size = 0;
    std::size_t index = 0;         // the caller's number for this read (its place in the look-ahead window)
};

// One disk read that serves every member: the bytes [offset, offset + span) of `file`.
struct ReadGroup {
    std::uintptr_t file = 0;
    std::uint64_t offset = 0;
    std::uint64_t span = 0;
    std::vector<PlannedRead> members;  // in offset order
};

// Sort by (file, offset, size); a read joins the open group of its file when it starts no further than `gap` bytes past the
// group's end and the group's span stays within `max_span`. A read larger than max_span is its own group. Reads of size 0
// are dropped (nothing to fetch). Groups come out in (file, offset) order: one forward sweep per file.
inline std::vector<ReadGroup> plan_reads(std::vector<PlannedRead> reads, std::uint64_t gap, std::uint64_t max_span)
{
    reads.erase(std::remove_if(reads.begin(), reads.end(), [](const PlannedRead& r) { return r.size == 0; }), reads.end());
    std::stable_sort(reads.begin(), reads.end(), [](const PlannedRead& a, const PlannedRead& b) {
        if (a.file != b.file) return a.file < b.file;
        if (a.offset != b.offset) return a.offset < b.offset;
        return a.size < b.size;
    });
    std::vector<ReadGroup> groups;
    for (const PlannedRead& r : reads) {
        const std::uint64_t end = r.offset + r.size;
        if (!groups.empty()) {
            ReadGroup& g = groups.back();
            const std::uint64_t group_end = g.offset + g.span;
            const std::uint64_t new_end = end > group_end ? end : group_end;
            const bool close_enough = r.offset <= group_end || r.offset - group_end <= gap;  // no overflow near 2^64
            if (g.file == r.file && close_enough && new_end - g.offset <= max_span) {
                g.span = new_end - g.offset;
                g.members.push_back(r);
                continue;
            }
        }
        ReadGroup g;
        g.file = r.file;
        g.offset = r.offset;
        g.span = r.size;
        g.members.push_back(r);
        groups.push_back(std::move(g));
    }
    return groups;
}

// How many disk reads the plan saves against reading every request on its own.
inline std::size_t reads_saved(const std::vector<ReadGroup>& groups)
{
    std::size_t members = 0;
    for (const ReadGroup& g : groups) members += g.members.size();
    return members - groups.size();
}

}  // namespace x2w_startup
