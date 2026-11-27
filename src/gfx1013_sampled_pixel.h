#pragma once
#include <string>
#include <string_view>

namespace x2w {
struct SampledPixel {
    std::string hlsl;
    std::string bindings;
};

// Straight-line pixel programs with two entry descriptor tables (s8/s9),
// primitive mask in s10, perspective-center IJ in v0/v1, attr0 -> PARAM0,
// and R11G11B10_FLOAT/R16_FLOAT targets. The caller verifies this interface.
// code spans the instructions from entry through s_endpgm, without padding.
SampledPixel translate_gfx1013_sampled_pixel(std::string_view code);
}
