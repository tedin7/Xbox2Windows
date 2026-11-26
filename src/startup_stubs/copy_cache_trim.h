#pragma once

#include <cstddef>

namespace x2w_local {

// Copies of title memory made for draws, found again by (address, bytes). Small entries live for one frame. Large ones (64 KiB
// and up) outlive it: every draw still compares the title's memory with the entry's snapshot byte for byte, and the ring slice
// is reused only while its chunk keeps the generation it was made in (recycling a chunk bumps it), so a 16 MiB binding that did
// not change is not copied again every frame. The snapshots of the kept entries are bounded; past the bound all are dropped.
inline constexpr std::size_t big_copy_bytes = 64u << 10;
inline constexpr std::size_t big_copies_budget = 96u << 20;

// `copies` maps a key to an entry with a `content` vector (the snapshot). Called when a new frame starts.
template <class Map>
void trim_frame_copies(Map& copies)
{
    std::size_t kept = 0;
    for (auto at = copies.begin(); at != copies.end();) {
        if (at->second.content.size() < big_copy_bytes) {
            at = copies.erase(at);
        } else {
            kept += at->second.content.size();
            ++at;
        }
    }
    if (kept > big_copies_budget) copies.clear();
}

}  // namespace x2w_local
