#pragma once

#include <cstdint>

namespace x2w_startup {

inline constexpr char swizzled_input_buffer[] =
    "an input buffer descriptor is swizzled (V# swizzle_enable); swizzled addressing is not implemented";
inline constexpr char add_tid_input_buffer[] =
    "an input buffer descriptor adds the thread ID to the index (V# add_tid_enable); not implemented";

constexpr const char* buffer_input_addressing_refusal(std::uint32_t word1, std::uint32_t word3)
{
    // RDNA 2 Buffer Resource Descriptor. Word 1 bit 30 is a cache hint.
    if (word1 & (1u << 31)) return swizzled_input_buffer;
    if (word3 & (1u << 23)) return add_tid_input_buffer;
    return nullptr;
}

}  // namespace x2w_startup
