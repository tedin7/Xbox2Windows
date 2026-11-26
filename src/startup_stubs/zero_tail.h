#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace x2w_local {

// A raw shader buffer view returns zero for every read past its end, and memory the title never wrote is zero. So the
// trailing all-zero part of a bound range need not be copied: the view over the shorter copy reads the same values. Returns the
// length of [data, data + bytes) without its trailing zero bytes, rounded up to `granule` (a power of two) and never below
// `granule` or above `bytes`. The scan walks backwards from the end and stops at the first non-zero word, so a buffer that is
// used to its end costs one word.
inline std::size_t used_length(const void* data, std::size_t bytes, std::size_t granule = 256) noexcept
{
    if (bytes <= granule) return bytes;
    const auto* base = static_cast<const unsigned char*>(data);
    std::size_t end = bytes;
    while (end % 8 && end > 0) {
        if (base[end - 1]) return (end + granule - 1) & ~(granule - 1) < bytes ? (end + granule - 1) & ~(granule - 1) : bytes;
        --end;
    }
    while (end >= 8) {
        std::uint64_t word;
        std::memcpy(&word, base + end - 8, 8);
        if (word) break;
        end -= 8;
    }
    if (end < granule) end = granule;
    const std::size_t rounded = (end + granule - 1) & ~(granule - 1);
    return rounded < bytes ? rounded : bytes;
}

}  // namespace x2w_local
