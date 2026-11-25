// gfx10 V# word 3: formats for which gfx2hlsl emits a formatted-load case.
#pragma once

#include <cstdint>

namespace x2w_startup {

constexpr std::uint32_t formatted_buffer_code(std::uint32_t word3)
{
    return (word3 >> 12) & 127u;
}

constexpr bool formatted_buffer_supported(std::uint32_t code)
{
    switch (code) {
    case 1: case 5: case 14: case 22: case 23: case 24: case 27:
    case 29: case 56: case 60: case 64: case 71: case 74: case 77:
        return true;
    default:
        return false;
    }
}

}  // namespace x2w_startup
