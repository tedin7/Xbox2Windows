#pragma once

#include <cstdint>

// GPUR header. Texture footprints are rebased to offset zero. Buffers use
// UNKNOWN format, height and depth one, and row_pitch zero. Buffer widths that
// exceed 32 bits use width zero and carry the full size in bytes.
namespace x2w_gpu_state {
struct Header {
    std::uint64_t object;
    std::uint32_t subresource, format;
    std::uint64_t offset;
    std::uint32_t width, height, depth, row_pitch;
    std::uint64_t bytes;
};
static_assert(sizeof(Header) == 48);
}
