#pragma once
// X2W_DEBUG_PRESENT_TARGET (docs/startup-runtime.md#debug-present): the pure part. Parsing the setting, choosing the
// target to show and turning the copied bytes of a depth buffer or a colour target into 8-bit RGBA. No D3D12 here, so
// the same code is checked by the WARP fixture (tests/startup_debug_present.cpp) against a CPU reference.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace x2w_debug_present {

enum class Mode { off, depth, largest_rt, exact_rt };

struct Spec {
    Mode mode = Mode::off;
    unsigned width = 0, height = 0;  // exact_rt
    int format = 0;                  // exact_rt, a DXGI_FORMAT number
};

// "depth", "largest-rt", "rt:<W>x<H>:<fmt>". Anything else is off.
inline Spec parse_spec(const char* text)
{
    Spec spec;
    if (!text || !*text) return spec;
    const std::string value(text);
    if (value == "depth") spec.mode = Mode::depth;
    else if (value == "largest-rt") spec.mode = Mode::largest_rt;
    else if (value.compare(0, 3, "rt:") == 0) {
        unsigned w = 0, h = 0;
        int format = 0, used = 0;
        if (std::sscanf(value.c_str() + 3, "%ux%u:%d%n", &w, &h, &format, &used) == 3 && w && h && format > 0 &&
            value[3 + used] == 0)
            spec = {Mode::exact_rt, w, h, format};
    }
    return spec;
}

// DXGI_FORMAT numbers this code reads.
enum Dxgi : int {
    r32g32b32a32_float = 2, r16g16b16a16_float = 10, r16g16b16a16_unorm = 11, r32g32_float = 16,
    d32_float_s8x24 = 20, r10g10b10a2_unorm = 24, r11g11b10_float = 26, r8g8b8a8_unorm = 28, r8g8b8a8_srgb = 29,
    r16g16_float = 34, d32_float = 40, r32_float = 41, d24_unorm_s8 = 45, r8g8_unorm = 49, r16_float = 54, d16_unorm = 55,
    r8_unorm = 61, b8g8r8a8_unorm = 87, b8g8r8x8_unorm = 88, b8g8r8a8_srgb = 91,
};

inline bool is_depth_format(int format)
{
    return format == d32_float_s8x24 || format == d32_float || format == d24_unorm_s8 || format == d16_unorm;
}

// Bytes of one pixel of the copied plane (the depth plane for a depth format); 0 for a format not read.
inline unsigned plane_bytes(int format)
{
    switch (format) {
    case r32g32b32a32_float: return 16;
    case r16g16b16a16_float: case r16g16b16a16_unorm: case r32g32_float: return 8;
    case d32_float_s8x24: case d32_float: case d24_unorm_s8: case r10g10b10a2_unorm: case r11g11b10_float:
    case r8g8b8a8_unorm: case r8g8b8a8_srgb: case r16g16_float: case r32_float: case b8g8r8a8_unorm: case b8g8r8x8_unorm:
    case b8g8r8a8_srgb: return 4;
    case r8g8_unorm: case r16_float: case d16_unorm: return 2;
    case r8_unorm: return 1;
    default: return 0;
    }
}

struct Candidate {
    unsigned long long width = 0;
    unsigned height = 0;
    int format = 0;
    bool depth = false;
    bool valid() const { return width && height; }
    unsigned long long area() const { return width * height; }
};

// Whether `next` (bound later than `best`) replaces `best` under `spec`. The biggest area wins and the later one keeps a
// tie, so the last biggest target of the frame is shown.
inline bool wants(const Spec& spec, const Candidate& best, const Candidate& next)
{
    if (!next.valid() || !plane_bytes(next.format)) return false;
    bool eligible = false;
    switch (spec.mode) {
    case Mode::depth: eligible = next.depth; break;
    case Mode::largest_rt: eligible = !next.depth; break;
    case Mode::exact_rt: eligible = next.width == spec.width && next.height == spec.height && next.format == spec.format; break;
    case Mode::off: break;
    }
    return eligible && (!best.valid() || next.area() >= best.area());
}

inline float half_to_float(std::uint16_t h)
{
    const std::uint32_t sign = (h >> 15) & 1u, exponent = (h >> 10) & 31u, fraction = h & 1023u;
    float value;
    if (!exponent) value = std::ldexp(static_cast<float>(fraction), -24);
    else if (exponent == 31) value = fraction ? NAN : INFINITY;
    else value = std::ldexp(static_cast<float>(fraction | 1024u), static_cast<int>(exponent) - 25);
    return sign ? -value : value;
}

inline float small_float(std::uint32_t bits, unsigned mantissa_bits)
{
    // R11G11B10: 5 exponent bits, 6 (R, G) or 5 (B) mantissa bits, no sign.
    const std::uint32_t exponent = bits >> mantissa_bits, fraction = bits & ((1u << mantissa_bits) - 1u);
    if (!exponent) return std::ldexp(static_cast<float>(fraction), -14 - static_cast<int>(mantissa_bits));
    if (exponent == 31) return fraction ? NAN : INFINITY;
    return std::ldexp(1.0f + static_cast<float>(fraction) / static_cast<float>(1u << mantissa_bits), static_cast<int>(exponent) - 15);
}

inline unsigned char byte_of_unit(float v)
{
    if (!(v > 0.0f)) return 0;  // negative and NaN
    if (v >= 1.0f) return 255;
    return static_cast<unsigned char>(v * 255.0f + 0.5f);
}

// HDR value to 8 bits: Reinhard x / (1 + x), then a 2.2 gamma.
inline unsigned char tone_map(float v)
{
    if (!(v > 0.0f)) return 0;
    if (std::isinf(v)) return 255;
    return byte_of_unit(std::pow(v / (1.0f + v), 1.0f / 2.2f));
}

// Geometry never reaches black, so it stays distinguishable from the background (value 0). The darkest geometry is
// geometry_floor, the brightest 255.
constexpr int geometry_floor = 40;

inline float depth_at(const unsigned char* pixel, int format)
{
    if (format == d16_unorm) {
        std::uint16_t v;
        std::memcpy(&v, pixel, 2);
        return static_cast<float>(v) / 65535.0f;
    }
    if (format == d24_unorm_s8) {
        std::uint32_t v;
        std::memcpy(&v, pixel, 4);
        return static_cast<float>(v & 0xffffffu) / 16777215.0f;
    }
    float v;
    std::memcpy(&v, pixel, 4);
    return v;
}

// A depth plane to grey. The clear value is the more common of exactly 0 and exactly 1: 0 means reversed Z (near is
// large). The background (the clear value) is black; the rest is normalised between its minimum and maximum so that
// the nearest geometry is 255 and the farthest geometry is geometry_floor. Non-finite values are background.
inline void depth_to_rgba(const unsigned char* data, std::size_t pitch, unsigned ow, unsigned oh, unsigned step, int format,
                          std::vector<unsigned char>& rgba)
{
    const unsigned bytes = plane_bytes(format);
    const auto at = [&](unsigned x, unsigned y) { return depth_at(data + static_cast<std::size_t>(y) * step * pitch + static_cast<std::size_t>(x) * step * bytes, format); };
    std::size_t zeros = 0, ones = 0;
    for (unsigned y = 0; y < oh; ++y)
        for (unsigned x = 0; x < ow; ++x) {
            const float v = at(x, y);
            zeros += v == 0.0f;
            ones += v == 1.0f;
        }
    const bool reversed = zeros > ones;
    const float background = reversed ? 0.0f : 1.0f;
    float low = INFINITY, high = -INFINITY;
    for (unsigned y = 0; y < oh; ++y)
        for (unsigned x = 0; x < ow; ++x) {
            const float v = at(x, y);
            if (!std::isfinite(v) || v == background) continue;
            low = v < low ? v : low;
            high = v > high ? v : high;
        }
    rgba.assign(static_cast<std::size_t>(ow) * oh * 4, 255);
    for (unsigned y = 0; y < oh; ++y)
        for (unsigned x = 0; x < ow; ++x) {
            const float v = at(x, y);
            int grey = 0;
            if (std::isfinite(v) && v != background) {
                const float n = high > low ? (v - low) / (high - low) : 1.0f;  // 0 = smallest depth value
                const float near_ness = reversed ? n : 1.0f - n;
                grey = geometry_floor + static_cast<int>(near_ness * float(255 - geometry_floor) + 0.5f);
            }
            unsigned char* out = &rgba[(static_cast<std::size_t>(y) * ow + x) * 4];
            out[0] = out[1] = out[2] = static_cast<unsigned char>(grey);
        }
}

inline void color_pixel(const unsigned char* in, int format, unsigned char* out)
{
    const auto u16 = [&](unsigned at) { std::uint16_t v; std::memcpy(&v, in + at, 2); return v; };
    const auto f32 = [&](unsigned at) { float v; std::memcpy(&v, in + at, 4); return v; };
    out[3] = 255;
    switch (format) {
    case r8g8b8a8_unorm: case r8g8b8a8_srgb: out[0] = in[0]; out[1] = in[1]; out[2] = in[2]; break;
    case b8g8r8a8_unorm: case b8g8r8x8_unorm: case b8g8r8a8_srgb: out[0] = in[2]; out[1] = in[1]; out[2] = in[0]; break;
    case r8g8_unorm: out[0] = in[0]; out[1] = in[1]; out[2] = 0; break;
    case r8_unorm: out[0] = out[1] = out[2] = in[0]; break;
    case r10g10b10a2_unorm: {
        std::uint32_t v;
        std::memcpy(&v, in, 4);
        out[0] = static_cast<unsigned char>((v & 1023u) * 255u / 1023u);
        out[1] = static_cast<unsigned char>(((v >> 10) & 1023u) * 255u / 1023u);
        out[2] = static_cast<unsigned char>(((v >> 20) & 1023u) * 255u / 1023u);
        break;
    }
    case r16g16b16a16_unorm:
        for (unsigned c = 0; c < 3; ++c) out[c] = static_cast<unsigned char>(u16(c * 2) >> 8);
        break;
    case r16g16b16a16_float:
        for (unsigned c = 0; c < 3; ++c) out[c] = tone_map(half_to_float(u16(c * 2)));
        break;
    case r16g16_float: out[0] = tone_map(half_to_float(u16(0))); out[1] = tone_map(half_to_float(u16(2))); out[2] = 0; break;
    case r16_float: out[0] = out[1] = out[2] = tone_map(half_to_float(u16(0))); break;
    case r32g32b32a32_float: for (unsigned c = 0; c < 3; ++c) out[c] = tone_map(f32(c * 4)); break;
    case r32g32_float: out[0] = tone_map(f32(0)); out[1] = tone_map(f32(4)); out[2] = 0; break;
    case r32_float: out[0] = out[1] = out[2] = tone_map(f32(0)); break;
    case r11g11b10_float: {
        std::uint32_t v;
        std::memcpy(&v, in, 4);
        out[0] = tone_map(small_float(v & 2047u, 6));
        out[1] = tone_map(small_float((v >> 11) & 2047u, 6));
        out[2] = tone_map(small_float(v >> 22, 5));
        break;
    }
    default: out[0] = out[1] = out[2] = 0; break;
    }
}

// Every `step`-th pixel in both directions: a 4K target keeps its picture at 1/16 of the work.
inline unsigned step_for(unsigned width, unsigned limit = 1024) { return width > limit ? (width + limit - 1) / limit : 1; }

// The copied plane (rows `pitch` bytes apart) to RGBA8, sampling every `step`-th pixel: the picture is ow x oh with
// ow = ceil(width / step). False for a format that is not read.
inline bool to_rgba(const unsigned char* data, std::size_t pitch, unsigned width, unsigned height, unsigned step, int format,
                    std::vector<unsigned char>& rgba, unsigned& ow, unsigned& oh)
{
    const unsigned bytes = plane_bytes(format);
    if (!bytes || !width || !height || !step) return false;
    ow = (width + step - 1) / step;
    oh = (height + step - 1) / step;
    if (is_depth_format(format)) {
        depth_to_rgba(data, pitch, ow, oh, step, format, rgba);
        return true;
    }
    rgba.assign(static_cast<std::size_t>(ow) * oh * 4, 0);
    for (unsigned y = 0; y < oh; ++y)
        for (unsigned x = 0; x < ow; ++x)
            color_pixel(data + static_cast<std::size_t>(y) * step * pitch + static_cast<std::size_t>(x) * step * bytes, format,
                        &rgba[(static_cast<std::size_t>(y) * ow + x) * 4]);
    return true;
}

}  // namespace x2w_debug_present
