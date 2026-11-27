#pragma once
#include <string>
#include <string_view>

namespace x2w {
// Accepts a headerless, little-endian gfx1013 pixel program with one full,
// uncompressed float MRT0 export. Unknown instructions and entry VGPRs fail.
std::string direct_gfx1013_constant_pixel(std::string_view code);
std::string translate_gfx1013_constant_pixel(std::string_view code);
} // namespace x2w
