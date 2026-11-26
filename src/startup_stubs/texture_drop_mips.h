#pragma once

// Opt-in "drop the top mip" for big read-only textures (X2W_D3D12_TEXTURE_DROP_MIPS=<levels>, default off).
//
// A texture above X2W_D3D12_TEXTURE_DROP_MIPS_MIN_BYTES (default 4 MiB) is created without its first <levels> mip
// levels: each dropped level removes about 75% of the bytes that are left. Only a texture that a translated draw
// binds for plain sampling (Sample, SampleBias, SampleLevel, SampleGrad, Gather) may lose a level: the sampler then
// reads a coarser picture, nothing is read at a wrong place. Everything addressed by texel or by size (Load,
// GetDimensions, CalculateLevelOfDetail), every copy, every view that starts below the top mip, and everything
// the draw did not say anything about keeps all its levels, and the reason is counted by name.
//
// Pure functions, no device: tests/startup_texture_drop_mips_check.cpp.

#include <d3d12.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace x2w_startup::drop_mips {

// What the translated draw that binds the texture does with it (manifest 'srvuse'). unknown is a draw that said
// nothing (a copy, a compute view, a pipeline translated before the hint existed).
enum class Use : std::uint8_t { unknown, sample, texel, size, lod };

enum class Reason : std::uint8_t {
    none,           // dropped
    off,            // the opt-in is not set
    below_min,        // below the byte threshold
    few_mips,       // not enough levels to drop and keep one
    not_2d,         // volume, 1D, array or cube
    flags,          // render target, depth, or UAV
    format,         // NV12 and other planar or typeless-only formats
    block_tail,     // a block-compressed level would fall under one 4x4 block
    multisampled,
    copy_or_other,  // materialized by a copy, a compute view, a map: no draw named the use
    unknown_use,    // the draw has no hint for this view
    texel,          // the shader loads texels
    size,           // the shader asks for the dimensions
    lod,            // the shader asks for the level of detail
    view_clamp,     // a view starts below the top mip or has a minimum LOD clamp
    count
};

inline const char* reason_name(Reason reason)
{
    switch (reason) {
    case Reason::none: return "dropped";
    case Reason::off: return "off";
    case Reason::below_min: return "small";
    case Reason::few_mips: return "few_mips";
    case Reason::not_2d: return "not_2d";
    case Reason::flags: return "flags";
    case Reason::format: return "format";
    case Reason::block_tail: return "block_tail";
    case Reason::multisampled: return "multisampled";
    case Reason::copy_or_other: return "copy_or_other";
    case Reason::unknown_use: return "unknown_use";
    case Reason::texel: return "texel_addressed";
    case Reason::size: return "get_dimensions";
    case Reason::lod: return "get_lod";
    case Reason::view_clamp: return "view_clamp";
    default: return "?";
    }
}

struct Config {
    unsigned levels = 0;                    // 0 = off
    std::uint64_t min_bytes = 4u << 20;     // textures at or above this size may lose levels
};

inline Config parse_config(const char* levels, const char* min_bytes)
{
    Config config;
    if (levels && *levels) {
        const unsigned long value = std::strtoul(levels, nullptr, 10);
        config.levels = value > 4 ? 4u : static_cast<unsigned>(value);
    }
    if (min_bytes && *min_bytes) {
        const unsigned long long value = std::strtoull(min_bytes, nullptr, 10);
        if (value) config.min_bytes = value;
    }
    return config;
}

inline Use parse_use(const char* word)
{
    if (!word) return Use::unknown;
    const auto is = [&](const char* name) {
        const char* a = word;
        while (*a && *name && *a == *name) ++a, ++name;
        return !*a && !*name;
    };
    return is("sample") ? Use::sample : is("texel") ? Use::texel : is("size") ? Use::size : is("lod") ? Use::lod
                                                                                                     : Use::unknown;
}

inline Reason use_reason(Use use)
{
    switch (use) {
    case Use::sample: return Reason::none;
    case Use::texel: return Reason::texel;
    case Use::size: return Reason::size;
    case Use::lod: return Reason::lod;
    default: return Reason::unknown_use;
    }
}

inline bool block_compressed(DXGI_FORMAT format)
{
    return (format >= DXGI_FORMAT_BC1_TYPELESS && format <= DXGI_FORMAT_BC5_SNORM) ||
           (format >= DXGI_FORMAT_BC6H_TYPELESS && format <= DXGI_FORMAT_BC7_UNORM_SRGB);
}

inline bool planar(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_NV12 || format == DXGI_FORMAT_P010 || format == DXGI_FORMAT_P016 ||
           format == DXGI_FORMAT_420_OPAQUE || format == DXGI_FORMAT_NV11 || format == DXGI_FORMAT_R32G8X24_TYPELESS ||
           format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT || format == DXGI_FORMAT_R24G8_TYPELESS ||
           format == DXGI_FORMAT_D24_UNORM_S8_UINT;
}

// The view the title made: it keeps the top mip when it starts at 0 and has no clamp.
struct ViewFacts {
    UINT most_detailed_mip = 0;
    float min_lod_clamp = 0.0f;
};

// Bits a texel (or a pixel of a block) takes, for the size estimate. Formats not listed count 32.
inline unsigned bits_per_texel(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_BC1_TYPELESS: case DXGI_FORMAT_BC1_UNORM: case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS: case DXGI_FORMAT_BC4_UNORM: case DXGI_FORMAT_BC4_SNORM: return 4;
    case DXGI_FORMAT_BC2_TYPELESS: case DXGI_FORMAT_BC2_UNORM: case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS: case DXGI_FORMAT_BC3_UNORM: case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_TYPELESS: case DXGI_FORMAT_BC5_UNORM: case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_TYPELESS: case DXGI_FORMAT_BC6H_UF16: case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS: case DXGI_FORMAT_BC7_UNORM: case DXGI_FORMAT_BC7_UNORM_SRGB:
    case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_A8_UNORM: return 8;
    case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_TYPELESS: case DXGI_FORMAT_B5G6R5_UNORM: return 16;
    case DXGI_FORMAT_R16G16B16A16_UNORM: case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT: return 64;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32A32_TYPELESS: return 128;
    default: return 32;
    }
}

// Bytes of the whole chain, an estimate for the byte threshold and the counter only (the threshold is a knob, not a
// proof): the top level at the format's bits per texel, the rest a third more.
inline std::uint64_t chain_estimate(const D3D12_RESOURCE_DESC& desc)
{
    const std::uint64_t base = std::uint64_t{desc.Width} * desc.Height * bits_per_texel(desc.Format) / 8;
    return desc.MipLevels != 1 ? base + base / 3 : base;
}

struct Decision {
    unsigned levels = 0;
    Reason reason = Reason::off;
};

inline Decision decide(const D3D12_RESOURCE_DESC& desc, const Config& config, Use use, const ViewFacts& view)
{
    if (!config.levels) return {0, Reason::off};
    if (use == Use::unknown) return {0, Reason::unknown_use};
    if (use != Use::sample) return {0, use_reason(use)};
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.DepthOrArraySize != 1) return {0, Reason::not_2d};
    if (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return {0, Reason::flags};
    if (desc.SampleDesc.Count != 1) return {0, Reason::multisampled};
    if (planar(desc.Format)) return {0, Reason::format};
    if (chain_estimate(desc) < config.min_bytes) return {0, Reason::below_min};
    if (view.most_detailed_mip != 0 || view.min_lod_clamp != 0.0f) return {0, Reason::view_clamp};
    const unsigned mips = desc.MipLevels ? desc.MipLevels : 1;
    if (mips <= config.levels) return {0, Reason::few_mips};
    if (block_compressed(desc.Format) &&
        ((std::max<std::uint64_t>(1, desc.Width >> config.levels) < 4) ||
         (std::max<UINT>(1, desc.Height >> config.levels) < 4)))
        return {0, Reason::block_tail};
    return {config.levels, Reason::none};
}

// The description of the texture that is created: the top `levels` levels are gone, the next one is the top.
inline D3D12_RESOURCE_DESC reduced_desc(D3D12_RESOURCE_DESC desc, unsigned levels)
{
    if (!levels) return desc;
    desc.Width = std::max<UINT64>(1, desc.Width >> levels);
    desc.Height = std::max<UINT>(1, desc.Height >> levels);
    const unsigned mips = desc.MipLevels ? desc.MipLevels : 1;
    desc.MipLevels = static_cast<UINT16>(mips > levels ? mips - levels : 1);
    return desc;
}

// Mip levels of a view on the reduced texture. The view the title made starts at the top (decide refuses another
// start): -1 (all levels) stays all levels; a count loses the dropped levels and keeps one.
inline UINT view_mip_levels(UINT title_levels, unsigned levels)
{
    if (title_levels == ~0u) return title_levels;
    return title_levels > levels ? title_levels - levels : 1;
}

// Bytes saved by dropping `levels` of a chain whose top level is `top_bytes` (a geometric estimate, for the counter).
inline std::uint64_t saved_estimate(std::uint64_t chain_bytes, unsigned levels)
{
    std::uint64_t left = chain_bytes;
    for (unsigned i = 0; i < levels; ++i) left /= 4;
    return chain_bytes - left;
}


// What the bind in progress says about the texture it is about to materialize or hand out. A translated draw sets it
// around its texture lookup; anything else (a copy, a compute view, a map) leaves it unknown, so it never drops a level
// and never receives a texture that lost one.
struct Hint {
    Use use = Use::unknown;
    ViewFacts view;
    bool draw = false;  // set by a translated draw's bind; false for everything else
};
inline thread_local Hint current_hint;

class BindHint {
public:
    BindHint(Use use, const ViewFacts& view) : saved_(current_hint) { current_hint = {use, view, true}; }
    ~BindHint() { current_hint = saved_; }
    BindHint(const BindHint&) = delete;
    BindHint& operator=(const BindHint&) = delete;

private:
    Hint saved_;
};

// A texture that lost levels may only be given to a bind that said it samples (every other reader addresses mips or
// texels of the title's size).
inline bool may_receive_dropped(const Hint& hint) { return hint.use == Use::sample; }

// Counters (one line with the heartbeat, see report_drop_mips in d3d12_xs_device.cpp).
struct Counters {
    std::atomic<std::uint64_t> dropped{0};
    std::atomic<std::uint64_t> saved_bytes{0};
    std::atomic<std::uint64_t> kept[static_cast<unsigned>(Reason::count)]{};
    std::atomic<std::uint64_t> conflicts{0};  // a dropped texture asked for by a reader that may not have it
    std::atomic<std::uint64_t> upload_unsupported{0};
    void keep(Reason reason) { kept[static_cast<unsigned>(reason)].fetch_add(1, std::memory_order_relaxed); }
};
inline Counters counters;

}  // namespace x2w_startup::drop_mips
