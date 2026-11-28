// Pass thermometer (X2W_PASS_STATS=1, docs/startup-runtime.md#pass-thermometer): the parts that need no device.
// The format table picks the reduction shader and the channels a written format has, combine() folds the
// per-group partial results the GPU wrote, and walk() finds where the presented picture lost its signal.
#pragma once

#include <dxgiformat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace x2w_pass_stats {

// Reduction shader variants, numbered as the shader's KIND macro.
enum class Sample : unsigned { float4 = 0, uint4 = 1, sint4 = 2, words = 3, none = 4 };

struct FormatInfo {
    Sample sample = Sample::none;
    unsigned channels = 0;
    DXGI_FORMAT view = DXGI_FORMAT_UNKNOWN;      // the shader resource view format the reduction reads
    DXGI_FORMAT snapshot = DXGI_FORMAT_UNKNOWN;  // the snapshot texture's format (the resource format, or its typeless group)
    bool depth = false;                          // the snapshot needs ALLOW_DEPTH_STENCIL (depth plane 0 is read)
};

namespace detail {
inline FormatInfo plain(Sample s, unsigned channels, DXGI_FORMAT view, DXGI_FORMAT resource)
{
    return {s, channels, view, resource, false};
}
inline FormatInfo depth(DXGI_FORMAT view, DXGI_FORMAT typeless)
{
    return {Sample::float4, 1, view, typeless, true};
}
}  // namespace detail

inline FormatInfo format_info(DXGI_FORMAT format, DXGI_FORMAT written, bool depth_flag);

namespace detail {
// A typeless resource is read through the view format it was written with when that is known (a UINT or SINT
// view picks the integer reduction), else through `fallback`; the snapshot keeps the typeless format.
inline FormatInfo typeless(DXGI_FORMAT format, DXGI_FORMAT written, DXGI_FORMAT fallback)
{
    FormatInfo info = format_info(written, DXGI_FORMAT_UNKNOWN, false);
    if (info.sample == Sample::none || info.depth) info = format_info(fallback, DXGI_FORMAT_UNKNOWN, false);
    info.snapshot = format;
    return info;
}
}  // namespace detail

// What the reduction reads for a texture of `format` written through a view of `written` (UNKNOWN when not known).
// A format the reduction cannot read keeps Sample::none.
inline FormatInfo format_info(DXGI_FORMAT format, DXGI_FORMAT written, bool depth_flag)
{
    using detail::plain;
    using detail::typeless;
    const Sample f = Sample::float4, u = Sample::uint4, s = Sample::sint4;
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return typeless(format, written, DXGI_FORMAT_R32G32B32A32_FLOAT);
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return plain(f, 4, format, format);
    case DXGI_FORMAT_R32G32B32A32_UINT: return plain(u, 4, format, format);
    case DXGI_FORMAT_R32G32B32A32_SINT: return plain(s, 4, format, format);
    case DXGI_FORMAT_R32G32B32_FLOAT: return plain(f, 3, format, format);
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return typeless(format, written, DXGI_FORMAT_R16G16B16A16_FLOAT);
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_SNORM: return plain(f, 4, format, format);
    case DXGI_FORMAT_R16G16B16A16_UINT: return plain(u, 4, format, format);
    case DXGI_FORMAT_R16G16B16A16_SINT: return plain(s, 4, format, format);
    case DXGI_FORMAT_R32G32_TYPELESS: return typeless(format, written, DXGI_FORMAT_R32G32_FLOAT);
    case DXGI_FORMAT_R32G32_FLOAT: return plain(f, 2, format, format);
    case DXGI_FORMAT_R32G32_UINT: return plain(u, 2, format, format);
    case DXGI_FORMAT_R32G32_SINT: return plain(s, 2, format, format);
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return typeless(format, written, DXGI_FORMAT_R10G10B10A2_UNORM);
    case DXGI_FORMAT_R10G10B10A2_UNORM: return plain(f, 4, format, format);
    case DXGI_FORMAT_R10G10B10A2_UINT: return plain(u, 4, format, format);
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return plain(f, 3, format, format);
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return typeless(format, written, DXGI_FORMAT_R8G8B8A8_UNORM);
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_SNORM: return plain(f, 4, format, format);
    case DXGI_FORMAT_R8G8B8A8_UINT: return plain(u, 4, format, format);
    case DXGI_FORMAT_R8G8B8A8_SINT: return plain(s, 4, format, format);
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return typeless(format, written, DXGI_FORMAT_B8G8R8A8_UNORM);
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B5G5R5A1_UNORM:
    case DXGI_FORMAT_B4G4R4A4_UNORM: return plain(f, 4, format, format);
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
    case DXGI_FORMAT_B5G6R5_UNORM: return plain(f, 3, format, format);
    case DXGI_FORMAT_R16G16_TYPELESS: return typeless(format, written, DXGI_FORMAT_R16G16_FLOAT);
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_SNORM: return plain(f, 2, format, format);
    case DXGI_FORMAT_R16G16_UINT: return plain(u, 2, format, format);
    case DXGI_FORMAT_R16G16_SINT: return plain(s, 2, format, format);
    case DXGI_FORMAT_R8G8_TYPELESS: return typeless(format, written, DXGI_FORMAT_R8G8_UNORM);
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_SNORM: return plain(f, 2, format, format);
    case DXGI_FORMAT_R8G8_UINT: return plain(u, 2, format, format);
    case DXGI_FORMAT_R8G8_SINT: return plain(s, 2, format, format);
    case DXGI_FORMAT_R32_TYPELESS:
        return depth_flag ? detail::depth(DXGI_FORMAT_R32_FLOAT, format) : typeless(format, written, DXGI_FORMAT_R32_FLOAT);
    case DXGI_FORMAT_R32_FLOAT: return plain(f, 1, format, format);
    case DXGI_FORMAT_R32_UINT: return plain(u, 1, format, format);
    case DXGI_FORMAT_R32_SINT: return plain(s, 1, format, format);
    case DXGI_FORMAT_D32_FLOAT: return detail::depth(DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS);
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return detail::depth(DXGI_FORMAT_R24_UNORM_X8_TYPELESS, DXGI_FORMAT_R24G8_TYPELESS);
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return detail::depth(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS, DXGI_FORMAT_R32G8X24_TYPELESS);
    case DXGI_FORMAT_R16_TYPELESS:
        return depth_flag ? detail::depth(DXGI_FORMAT_R16_UNORM, format) : typeless(format, written, DXGI_FORMAT_R16_FLOAT);
    case DXGI_FORMAT_D16_UNORM: return detail::depth(DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_TYPELESS);
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_SNORM: return plain(f, 1, format, format);
    case DXGI_FORMAT_R16_UINT: return plain(u, 1, format, format);
    case DXGI_FORMAT_R16_SINT: return plain(s, 1, format, format);
    case DXGI_FORMAT_R8_TYPELESS: return typeless(format, written, DXGI_FORMAT_R8_UNORM);
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_SNORM: return plain(f, 1, format, format);
    case DXGI_FORMAT_R8_UINT: return plain(u, 1, format, format);
    case DXGI_FORMAT_R8_SINT: return plain(s, 1, format, format);
    default: return {};
    }
}

// The partial record of one reduction group per channel, in dwords: min (complement of its ordered word), max
// (ordered word), sum (float bits), zeros, NaN, Inf. Ordered words keep the float order; 0 means no finite value.
constexpr unsigned partial_words = 6;
constexpr unsigned group_words = 4 * partial_words;
constexpr unsigned max_groups = 64;
constexpr unsigned group_threads = 256;

inline unsigned groups_for(std::uint64_t texels)
{
    const std::uint64_t needed = (texels + group_threads - 1) / group_threads;
    return static_cast<unsigned>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(max_groups, needed)));
}

struct Stats {
    std::uint64_t texels = 0;
    unsigned channels = 0;
    float min[4]{}, max[4]{};
    double mean[4]{};
    bool finite[4]{};  // the channel has at least one value that is neither NaN nor Inf
    std::uint64_t zero[4]{}, nan[4]{}, inf[4]{};
};

inline float ordered_float(std::uint32_t word);

inline float word_float(std::uint32_t word)
{
    float value;
    static_assert(sizeof(value) == sizeof(word));
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

// Folds `groups` partial records (group_words dwords each) into the stats of `texels` texels.
inline Stats combine(const std::uint32_t* words, unsigned groups, std::uint64_t texels, unsigned channels)
{
    Stats stats;
    stats.texels = texels;
    stats.channels = std::min(channels, 4u);
    for (unsigned c = 0; c < stats.channels; ++c) {
        double sum = 0;
        for (unsigned g = 0; g < groups; ++g) {
            const std::uint32_t* p = words + g * group_words + c * partial_words;
            const bool seen = p[0] != 0 && p[1] != 0;  // a group that saw no finite value left both words 0
            const float lo = seen ? ordered_float(~p[0]) : 0.0f, hi = seen ? ordered_float(p[1]) : 0.0f;
            stats.zero[c] += p[3];
            stats.nan[c] += p[4];
            stats.inf[c] += p[5];
            if (!seen) continue;
            stats.min[c] = stats.finite[c] ? std::min(stats.min[c], lo) : lo;
            stats.max[c] = stats.finite[c] ? std::max(stats.max[c], hi) : hi;
            stats.finite[c] = true;
            sum += word_float(p[2]);
        }
        const std::uint64_t counted = stats.nan[c] + stats.inf[c];
        if (stats.finite[c] && texels > counted) stats.mean[c] = sum / static_cast<double>(texels - counted);
    }
    return stats;
}

// The float of an order-preserving word (pass_stats_shader.h ordered).
inline float ordered_float(std::uint32_t word)
{
    return word_float((word & 0x80000000u) != 0 ? word & 0x7fffffffu : ~word);
}

// Signal: a colour channel (the first three; alpha alone is not a picture) holds a finite nonzero value somewhere.
inline bool has_signal(const Stats& stats)
{
    const unsigned colour = std::min(stats.channels, 3u);
    for (unsigned c = 0; c < colour; ++c)
        if (stats.zero[c] + stats.nan[c] + stats.inf[c] < stats.texels) return true;
    return false;
}

// JSON text pieces. Floats use %.9g; a channel without a finite value prints null.

inline void append_number(std::string& out, double value)
{
    char text[40];
    std::snprintf(text, sizeof(text), "%.9g", value);
    out += text;
}

template <class Value>
void append_array(std::string& out, const char* name, const Stats& stats, Value value)
{
    out += '"';
    out += name;
    out += "\":[";
    for (unsigned c = 0; c < stats.channels; ++c) {
        if (c) out += ',';
        value(c);
    }
    out += ']';
}

inline std::string stats_json(const Stats& s)
{
    std::string out = "{\"texels\":" + std::to_string(s.texels) + ",\"channels\":" + std::to_string(s.channels) + ',';
    const auto finite_or_null = [&](unsigned c, double v) { if (s.finite[c]) append_number(out, v); else out += "null"; };
    append_array(out, "min", s, [&](unsigned c) { finite_or_null(c, s.min[c]); });
    out += ',';
    append_array(out, "max", s, [&](unsigned c) { finite_or_null(c, s.max[c]); });
    out += ',';
    append_array(out, "mean", s, [&](unsigned c) { finite_or_null(c, s.mean[c]); });
    out += ',';
    append_array(out, "zero", s, [&](unsigned c) { out += std::to_string(s.zero[c]); });
    out += ',';
    append_array(out, "nan", s, [&](unsigned c) { out += std::to_string(s.nan[c]); });
    out += ',';
    append_array(out, "inf", s, [&](unsigned c) { out += std::to_string(s.inf[c]); });
    out += '}';
    return out;
}

// What the walk needs of one pass of a present, in submission order.
enum class Signal { unknown, dead, signal };
struct PassView {
    bool clear_only = false;
    bool inputs_known = false;
    std::vector<std::uintptr_t> writes, inputs;  // resource identities
    Signal signal = Signal::unknown;
};

struct Collapse {
    int final_pass = -1;   // the last pass that wrote the presented surface
    int last_signal = -1;  // the last pass on its chain with signal
    int first_dead = -1;   // the first all-zero/NaN pass on the chain after last_signal (clear-only passes skipped)
};

namespace detail {
inline int last_writer(const std::vector<PassView>& passes, int before, std::uintptr_t resource)
{
    for (int j = before - 1; j >= 0; --j)
        if (std::find(passes[j].writes.begin(), passes[j].writes.end(), resource) != passes[j].writes.end()) return j;
    return -1;
}
}  // namespace detail

// Walks back from the presented surface along the writes: a pass depends on the earlier writers of what it
// writes (its earlier content) and of its known inputs; a pass whose inputs are not known depends on every earlier pass.
inline Collapse walk(const std::vector<PassView>& passes, std::uintptr_t surface)
{
    Collapse result;
    const int n = static_cast<int>(passes.size());
    result.final_pass = detail::last_writer(passes, n, surface);
    if (result.final_pass < 0) return result;
    std::vector<bool> chain(passes.size(), false);
    std::vector<int> stack{result.final_pass};
    while (!stack.empty()) {
        const int i = stack.back();
        stack.pop_back();
        if (i < 0 || chain[i]) continue;
        chain[i] = true;
        const PassView& p = passes[i];
        for (const auto r : p.writes) stack.push_back(detail::last_writer(passes, i, r));
        if (!p.inputs_known) {
            for (int j = 0; j < i; ++j) stack.push_back(j);
            continue;
        }
        for (const auto r : p.inputs) stack.push_back(detail::last_writer(passes, i, r));
    }
    for (int i = 0; i <= result.final_pass; ++i)
        if (chain[i] && passes[i].signal == Signal::signal) result.last_signal = i;
    for (int i = result.last_signal + 1; i <= result.final_pass; ++i) {
        if (chain[i] && passes[i].signal == Signal::dead && !passes[i].clear_only) {
            result.first_dead = i;
            break;
        }
    }
    return result;
}

}  // namespace x2w_pass_stats
