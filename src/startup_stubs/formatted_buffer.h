// gfx10 V# word 3: formats for which gfx2hlsl emits a formatted-load case.
#pragma once

#include <cstdint>

namespace x2w_startup {

constexpr std::uint32_t formatted_buffer_code(std::uint32_t word3)
{
    return (word3 >> 12) & 127u;
}

constexpr bool formatted_buffer_supported(std::uint32_t code, bool extended = false)
{
    switch (code) {
    case 1: case 5: case 14: case 22: case 23: case 24: case 27:
    case 29: case 50: case 56: case 60: case 64: case 71: case 74: case 77:
        return true;
    default:
        break;
    }
    if (!extended) return false;
    switch (code) {
    case 2: case 6: case 7: case 8: case 11: case 12: case 13:
    case 15: case 18: case 19: case 20: case 21: case 28: case 36: case 47:
    case 54: case 57: case 61: case 62: case 63: case 65: case 66:
    case 69: case 70: case 75: case 76:
        return true;
    default:
        return false;
    }
}

}  // namespace x2w_startup
