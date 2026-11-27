#pragma once

#include <cstdint>

namespace x2w_startup {

constexpr bool buffer_records_are_bytes(std::uint32_t word1, std::uint32_t word3)
{
    // RDNA 2 Table 35 selects byte bounds for mode 3 without address swizzling.
    // Word 1 bit 30 is a cache-bank hint. Only bit 31 enables address swizzling.
    return ((word3 >> 28) & 3) == 3 && !(word1 & (1u << 31));
}

constexpr std::uint64_t buffer_descriptor_bytes(
    std::uint32_t word1, std::uint32_t num_records, std::uint32_t word3)
{
    const auto stride = (word1 >> 16) & 0x3fff;
    return std::uint64_t{buffer_records_are_bytes(word1, word3) || !stride ? 1u : stride} * num_records;
}

}  // namespace x2w_startup
