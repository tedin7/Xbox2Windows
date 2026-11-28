#pragma once
// Comparison rule of the mesh replay oracle (X2W_REPLAY_ORACLE=mesh, d3d12_mesh_oracle.inl): two read-back copies of
// one render target or depth subresource, the image the proof translation drew and the image the serial wave-loop
// translation drew from the same inputs.
//   - 32-bit float channels (R32*_FLOAT views, D32 depth): equal within mesh_oracle_f32_ulp units in the last place;
//   - 16-bit float channels (R16*_FLOAT views): equal within mesh_oracle_f16_ulp;
//   - every other format (UNORM, SNORM, UINT, SINT, packed floats, stencil): bit-exact per texel.
// Two NaNs are equal; NaN against a number differs. +0 and -0 are equal.
#include <dxgiformat.h>

#include <cstdint>
#include <cstring>

namespace x2w_startup {

inline constexpr std::uint32_t mesh_oracle_f32_ulp = 4;
inline constexpr std::uint32_t mesh_oracle_f16_ulp = 2;

enum class OracleElement { exact, f32, f16 };

// The element rule of plane `plane` of a view of format `view` (an RTV or DSV format).
inline OracleElement oracle_element(DXGI_FORMAT view, unsigned plane)
{
    switch (view) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_D32_FLOAT:
        return OracleElement::f32;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return plane == 0 ? OracleElement::f32 : OracleElement::exact;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16_FLOAT:
        return OracleElement::f16;
    default:
        return OracleElement::exact;
    }
}

// Distance in units in the last place between two IEEE values of `bits` width (32 or 16); ~0u when exactly one is NaN.
inline std::uint32_t oracle_ulp_distance(std::uint32_t a, std::uint32_t b, unsigned bits)
{
    const std::uint32_t sign = bits == 32 ? 0x80000000u : 0x8000u;
    const std::uint32_t exponent = bits == 32 ? 0x7f800000u : 0x7c00u;
    const std::uint32_t magnitude = sign - 1u;
    const bool nan_a = (a & exponent) == exponent && (a & magnitude & ~exponent) != 0;
    const bool nan_b = (b & exponent) == exponent && (b & magnitude & ~exponent) != 0;
    if (nan_a || nan_b) return nan_a == nan_b ? 0u : ~0u;
    // Map to one monotonic unsigned line: negative values below sign, positive above; -0 and +0 meet at sign.
    const auto ordered = [&](std::uint32_t v) -> std::uint64_t {
        return (v & sign) ? std::uint64_t{sign} - (v & magnitude) : std::uint64_t{sign} + (v & magnitude);
    };
    const std::uint64_t x = ordered(a), y = ordered(b);
    const std::uint64_t d = x > y ? x - y : y - x;
    return d > 0xffffffffull ? ~0u : static_cast<std::uint32_t>(d);
}

struct OracleDiff {
    std::uint64_t texels = 0;     // texels compared
    std::uint64_t differing = 0;  // texels with at least one element outside the rule
    bool any = false;
    std::uint64_t first_offset = 0;  // byte offset of the first differing texel inside the subresource footprint
    unsigned first_x = 0, first_y = 0, first_z = 0;
    std::uint32_t max_ulp = 0;  // largest float distance seen (also inside the limit)
};

// Compares one subresource laid out as a placed footprint: `rows` rows of `row_bytes` meaningful bytes per depth
// slice, `row_pitch` bytes apart, `depth` slices `rows * row_pitch` apart; `texel_bytes` bytes per texel.
inline void oracle_compare(const unsigned char* a, const unsigned char* b, std::uint64_t row_bytes, unsigned rows,
                           std::uint64_t row_pitch, unsigned depth, unsigned texel_bytes, OracleElement element,
                           OracleDiff& diff)
{
    if (!texel_bytes) return;
    const unsigned width = static_cast<unsigned>(row_bytes / texel_bytes);
    const unsigned element_bytes = element == OracleElement::f32 ? 4u : element == OracleElement::f16 ? 2u : 0u;
    const std::uint32_t limit = element == OracleElement::f32 ? mesh_oracle_f32_ulp : mesh_oracle_f16_ulp;
    for (unsigned z = 0; z < depth; ++z) {
        for (unsigned y = 0; y < rows; ++y) {
            const std::uint64_t row = (std::uint64_t{z} * rows + y) * row_pitch;
            for (unsigned x = 0; x < width; ++x) {
                const std::uint64_t at = row + std::uint64_t{x} * texel_bytes;
                ++diff.texels;
                bool differs = false;
                if (!element_bytes || texel_bytes % element_bytes) {
                    differs = std::memcmp(a + at, b + at, texel_bytes) != 0;
                } else {
                    for (unsigned e = 0; e < texel_bytes; e += element_bytes) {
                        std::uint32_t u = 0, v = 0;
                        std::memcpy(&u, a + at + e, element_bytes);
                        std::memcpy(&v, b + at + e, element_bytes);
                        const std::uint32_t d = u == v ? 0u : oracle_ulp_distance(u, v, element_bytes * 8);
                        if (d > diff.max_ulp) diff.max_ulp = d;
                        differs |= d > limit;
                    }
                }
                if (!differs) continue;
                if (!diff.any) {
                    diff.any = true;
                    diff.first_offset = at;
                    diff.first_x = x;
                    diff.first_y = y;
                    diff.first_z = z;
                }
                ++diff.differing;
            }
        }
    }
}

}  // namespace x2w_startup
