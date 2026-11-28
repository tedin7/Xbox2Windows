// Experimental: a versioned on-disk recipe for a translated desktop graphics pipeline.
//
// The shader cache stores the vertex and pixel bytecode, the root signature and the sampler
// descriptions. It does not store the draw state that produced a desktop pipeline state object
// (PSO). With X2W_PSO_RECIPE_DIR set, the runtime appends that state here after a successful
// CreateGraphicsPipelineState and replays it on a later run before creating another PSO. This is
// state reconstruction, not a D3D12 pipeline library: the driver may still compile or cache the
// PSO internally.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>

// windows.h may already have been included by the translation unit, so NOMINMAX above is too late
// to prevent the min/max function-like macros. std::numeric_limits<T>::max is used below.
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace x2w_startup {

constexpr unsigned pso_recipe_version = 1;
constexpr std::size_t pso_recipe_file_limit = 64u * 1024u;
constexpr std::size_t pso_recipe_record_limit = 64;
constexpr unsigned pso_recipe_target_limit = 8;

struct PsoRecipe {
    std::uint64_t translation_key = 0;  // the shader-cache key this record belongs to
    std::uint64_t state_key = 0;        // pipeline_for's target/depth/raster/indexed key
    UINT target_count = 0;
    DXGI_FORMAT target_formats[pso_recipe_target_limit]{};
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
    bool indexed_rect = false;
    bool use_depth_stencil = false;  // whether DepthStencilState participated in the PSO
    bool has_rasterizer = false;     // whether a decoded rasterizer state participated in the key
    D3D12_BLEND_DESC blend{};        // effective state, after apply_export_write_masks
    D3D12_DEPTH_STENCIL_DESC depth_stencil{};
    D3D12_RASTERIZER_DESC rasterizer{};
    UINT sample_mask = ~0u;
    UINT sample_count = 1;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
};

struct PsoRecipeParseResult {
    std::vector<PsoRecipe> recipes;
    unsigned rejected = 0;
    std::string error;
};

inline std::uint64_t pso_recipe_hash(const void* data, std::size_t bytes,
                                     std::uint64_t hash = 1469598103934665603ull)
{
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
    return hash;
}

inline std::uint64_t pso_targets_key(UINT target_count, const DXGI_FORMAT* target_formats, DXGI_FORMAT dsv_format)
{
    std::uint64_t key = target_count;
    for (UINT i = 0; i < target_count; ++i) key = key * 131 + static_cast<std::uint64_t>(target_formats[i]);
    return pso_recipe_hash(&dsv_format, sizeof(dsv_format), key);
}

// The same key pipeline_for uses. Keeping it here lets a parsed record prove that its own fields
// reproduce the state key it names.
inline std::uint64_t pso_state_key(UINT target_count, const DXGI_FORMAT* target_formats, DXGI_FORMAT dsv_format,
                                   bool use_depth_stencil, const D3D12_DEPTH_STENCIL_DESC& depth_stencil,
                                   bool has_rasterizer, const D3D12_RASTERIZER_DESC& rasterizer, bool indexed_rect)
{
    std::uint64_t key = pso_targets_key(target_count, target_formats, dsv_format);
    if (use_depth_stencil) key = pso_recipe_hash(&depth_stencil, sizeof(depth_stencil), key);
    if (has_rasterizer) key = pso_recipe_hash(&rasterizer, sizeof(rasterizer), key);
    key = pso_recipe_hash(&indexed_rect, sizeof(indexed_rect), key);
    return key;
}

inline std::uint64_t pso_recipe_state_key(const PsoRecipe& recipe)
{
    return pso_state_key(recipe.target_count, recipe.target_formats, recipe.dsv_format, recipe.use_depth_stencil,
                         recipe.depth_stencil, recipe.has_rasterizer, recipe.rasterizer, recipe.indexed_rect);
}

inline D3D12_RENDER_TARGET_BLEND_DESC default_render_target_blend()
{
    return {FALSE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE,
            D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
}

inline D3D12_BLEND_DESC default_pso_blend()
{
    D3D12_BLEND_DESC blend{};
    for (auto& target : blend.RenderTarget) target = default_render_target_blend();
    return blend;
}

inline D3D12_RASTERIZER_DESC default_pso_rasterizer()
{
    D3D12_RASTERIZER_DESC rasterizer{};
    rasterizer.FillMode = D3D12_FILL_MODE_SOLID;
    rasterizer.CullMode = D3D12_CULL_MODE_NONE;
    rasterizer.DepthClipEnable = TRUE;
    return rasterizer;
}

// A valid depth-stencil state for a recipe that does not use one. The serializer writes every
// field, and the parser rejects an invalid stencil op; the zero-filled D3D12 struct is not a
// valid serialized state even when the PSO ignores it.
inline D3D12_DEPTH_STENCIL_DESC default_pso_depth()
{
    D3D12_DEPTH_STENCIL_DESC depth{};
    depth.DepthEnable = FALSE;
    depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    depth.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    depth.StencilEnable = FALSE;
    depth.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
    depth.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
    depth.FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                       D3D12_COMPARISON_FUNC_ALWAYS};
    depth.BackFace = depth.FrontFace;
    return depth;
}

inline bool same_float_bits(float a, float b)
{
    std::uint32_t left = 0, right = 0;
    std::memcpy(&left, &a, sizeof(left));
    std::memcpy(&right, &b, sizeof(right));
    return left == right;
}

inline bool same_render_target_blend(const D3D12_RENDER_TARGET_BLEND_DESC& a, const D3D12_RENDER_TARGET_BLEND_DESC& b)
{
    return a.BlendEnable == b.BlendEnable && a.LogicOpEnable == b.LogicOpEnable && a.SrcBlend == b.SrcBlend &&
           a.DestBlend == b.DestBlend && a.BlendOp == b.BlendOp && a.SrcBlendAlpha == b.SrcBlendAlpha &&
           a.DestBlendAlpha == b.DestBlendAlpha && a.BlendOpAlpha == b.BlendOpAlpha && a.LogicOp == b.LogicOp &&
           a.RenderTargetWriteMask == b.RenderTargetWriteMask;
}

inline bool same_pso_blend(const D3D12_BLEND_DESC& a, const D3D12_BLEND_DESC& b)
{
    if (a.AlphaToCoverageEnable != b.AlphaToCoverageEnable || a.IndependentBlendEnable != b.IndependentBlendEnable)
        return false;
    for (unsigned i = 0; i < 8; ++i)
        if (!same_render_target_blend(a.RenderTarget[i], b.RenderTarget[i])) return false;
    return true;
}

inline bool same_pso_depth(const D3D12_DEPTH_STENCIL_DESC& a, const D3D12_DEPTH_STENCIL_DESC& b)
{
    return a.DepthEnable == b.DepthEnable && a.DepthWriteMask == b.DepthWriteMask && a.DepthFunc == b.DepthFunc &&
           a.StencilEnable == b.StencilEnable && a.StencilReadMask == b.StencilReadMask &&
           a.StencilWriteMask == b.StencilWriteMask && a.FrontFace.StencilFailOp == b.FrontFace.StencilFailOp &&
           a.FrontFace.StencilDepthFailOp == b.FrontFace.StencilDepthFailOp &&
           a.FrontFace.StencilPassOp == b.FrontFace.StencilPassOp && a.FrontFace.StencilFunc == b.FrontFace.StencilFunc &&
           a.BackFace.StencilFailOp == b.BackFace.StencilFailOp &&
           a.BackFace.StencilDepthFailOp == b.BackFace.StencilDepthFailOp &&
           a.BackFace.StencilPassOp == b.BackFace.StencilPassOp && a.BackFace.StencilFunc == b.BackFace.StencilFunc;
}

inline bool same_pso_rasterizer(const D3D12_RASTERIZER_DESC& a, const D3D12_RASTERIZER_DESC& b)
{
    return a.FillMode == b.FillMode && a.CullMode == b.CullMode && a.FrontCounterClockwise == b.FrontCounterClockwise &&
           a.DepthBias == b.DepthBias && same_float_bits(a.DepthBiasClamp, b.DepthBiasClamp) &&
           same_float_bits(a.SlopeScaledDepthBias, b.SlopeScaledDepthBias) && a.DepthClipEnable == b.DepthClipEnable &&
           a.MultisampleEnable == b.MultisampleEnable && a.AntialiasedLineEnable == b.AntialiasedLineEnable &&
           a.ForcedSampleCount == b.ForcedSampleCount && a.ConservativeRaster == b.ConservativeRaster;
}

inline bool pso_recipe_targets_match(const PsoRecipe& recipe, UINT target_count, const DXGI_FORMAT* target_formats,
                                     DXGI_FORMAT dsv_format)
{
    if (target_count != recipe.target_count || dsv_format != recipe.dsv_format) return false;
    for (UINT i = 0; i < target_count; ++i)
        if (target_formats[i] != recipe.target_formats[i]) return false;
    return true;
}

inline PsoRecipe make_pso_recipe(std::uint64_t translation_key, std::uint64_t state_key, bool indexed_rect,
                                 bool use_depth_stencil, bool has_rasterizer,
                                 const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc)
{
    PsoRecipe recipe;
    recipe.translation_key = translation_key;
    recipe.state_key = state_key;
    recipe.target_count = desc.NumRenderTargets;
    for (UINT i = 0; i < pso_recipe_target_limit; ++i) recipe.target_formats[i] = desc.RTVFormats[i];
    recipe.dsv_format = desc.DSVFormat;
    recipe.indexed_rect = indexed_rect;
    recipe.use_depth_stencil = use_depth_stencil;
    recipe.has_rasterizer = has_rasterizer;
    recipe.blend = desc.BlendState;
    recipe.depth_stencil = use_depth_stencil ? desc.DepthStencilState : default_pso_depth();
    recipe.rasterizer = has_rasterizer ? desc.RasterizerState : default_pso_rasterizer();
    recipe.sample_mask = desc.SampleMask;
    recipe.sample_count = desc.SampleDesc.Count;
    recipe.topology = desc.PrimitiveTopologyType;
    return recipe;
}

inline void apply_pso_recipe(const PsoRecipe& recipe, D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc)
{
    desc.BlendState = recipe.blend;
    desc.DepthStencilState = recipe.use_depth_stencil ? recipe.depth_stencil : D3D12_DEPTH_STENCIL_DESC{};
    desc.RasterizerState = recipe.rasterizer;
    desc.SampleMask = recipe.sample_mask;
    desc.PrimitiveTopologyType = recipe.topology;
    desc.NumRenderTargets = recipe.target_count;
    for (UINT i = 0; i < pso_recipe_target_limit; ++i)
        desc.RTVFormats[i] = i < recipe.target_count ? recipe.target_formats[i] : DXGI_FORMAT_UNKNOWN;
    desc.DSVFormat = recipe.dsv_format;
    desc.SampleDesc.Count = recipe.sample_count;
    desc.SampleDesc.Quality = 0;
}

inline void append_hex(std::string& out, std::uint64_t value)
{
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    out += text;
}

inline std::string serialize_pso_recipe(const PsoRecipe& recipe)
{
    std::string out;
    out.reserve(1536);
    out += "x2w-pso-recipe 1\nstate 0x";
    append_hex(out, recipe.state_key);
    out += "\ntranslation 0x";
    append_hex(out, recipe.translation_key);
    out += "\ntargets ";
    out += std::to_string(recipe.target_count);
    for (DXGI_FORMAT format : recipe.target_formats) {
        out += ' ';
        out += std::to_string(static_cast<unsigned>(format));
    }
    out += "\ndsv ";
    out += std::to_string(static_cast<unsigned>(recipe.dsv_format));
    out += "\nindexed ";
    out += recipe.indexed_rect ? '1' : '0';
    out += "\nblend ";
    out += recipe.blend.AlphaToCoverageEnable ? '1' : '0';
    out += ' ';
    out += recipe.blend.IndependentBlendEnable ? '1' : '0';
    out += '\n';
    for (unsigned i = 0; i < 8; ++i) {
        const auto& t = recipe.blend.RenderTarget[i];
        out += "blend";
        out += std::to_string(i);
        out += ' ';
        out += t.BlendEnable ? '1' : '0';
        out += ' ';
        out += t.LogicOpEnable ? '1' : '0';
        const unsigned values[] = {static_cast<unsigned>(t.SrcBlend), static_cast<unsigned>(t.DestBlend),
                                   static_cast<unsigned>(t.BlendOp), static_cast<unsigned>(t.SrcBlendAlpha),
                                   static_cast<unsigned>(t.DestBlendAlpha), static_cast<unsigned>(t.BlendOpAlpha),
                                   static_cast<unsigned>(t.LogicOp), static_cast<unsigned>(t.RenderTargetWriteMask)};
        for (unsigned value : values) {
            out += ' ';
            out += std::to_string(value);
        }
        out += '\n';
    }
    out += "depth ";
    out += recipe.use_depth_stencil ? '1' : '0';
    const unsigned depth_values[] = {
        recipe.depth_stencil.DepthEnable ? 1u : 0u, static_cast<unsigned>(recipe.depth_stencil.DepthWriteMask),
        static_cast<unsigned>(recipe.depth_stencil.DepthFunc), recipe.depth_stencil.StencilEnable ? 1u : 0u,
        recipe.depth_stencil.StencilReadMask, recipe.depth_stencil.StencilWriteMask,
        static_cast<unsigned>(recipe.depth_stencil.FrontFace.StencilFailOp),
        static_cast<unsigned>(recipe.depth_stencil.FrontFace.StencilDepthFailOp),
        static_cast<unsigned>(recipe.depth_stencil.FrontFace.StencilPassOp),
        static_cast<unsigned>(recipe.depth_stencil.FrontFace.StencilFunc),
        static_cast<unsigned>(recipe.depth_stencil.BackFace.StencilFailOp),
        static_cast<unsigned>(recipe.depth_stencil.BackFace.StencilDepthFailOp),
        static_cast<unsigned>(recipe.depth_stencil.BackFace.StencilPassOp),
        static_cast<unsigned>(recipe.depth_stencil.BackFace.StencilFunc)};
    for (unsigned value : depth_values) {
        out += ' ';
        out += std::to_string(value);
    }
    out += "\nraster ";
    out += recipe.has_rasterizer ? '1' : '0';
    std::uint32_t clamp_bits = 0, slope_bits = 0;
    std::memcpy(&clamp_bits, &recipe.rasterizer.DepthBiasClamp, sizeof(clamp_bits));
    std::memcpy(&slope_bits, &recipe.rasterizer.SlopeScaledDepthBias, sizeof(slope_bits));
    const unsigned raster_values[] = {static_cast<unsigned>(recipe.rasterizer.FillMode),
                                      static_cast<unsigned>(recipe.rasterizer.CullMode),
                                      recipe.rasterizer.FrontCounterClockwise ? 1u : 0u};
    for (unsigned value : raster_values) {
        out += ' ';
        out += std::to_string(value);
    }
    out += ' ';
    out += std::to_string(recipe.rasterizer.DepthBias);
    char hex[32];
    std::snprintf(hex, sizeof(hex), " 0x%08x 0x%08x", clamp_bits, slope_bits);
    out += hex;
    const unsigned raster_tail[] = {recipe.rasterizer.DepthClipEnable ? 1u : 0u,
                                    recipe.rasterizer.MultisampleEnable ? 1u : 0u,
                                    recipe.rasterizer.AntialiasedLineEnable ? 1u : 0u,
                                    recipe.rasterizer.ForcedSampleCount,
                                    static_cast<unsigned>(recipe.rasterizer.ConservativeRaster)};
    for (unsigned value : raster_tail) {
        out += ' ';
        out += std::to_string(value);
    }
    out += "\nsample ";
    out += std::to_string(recipe.sample_count);
    out += ' ';
    out += std::to_string(recipe.sample_mask);
    out += "\ntopology ";
    out += std::to_string(static_cast<unsigned>(recipe.topology));
    out += "\nend\n";
    return out;
}

inline bool parse_unsigned(std::string_view token, std::uint64_t& value, int base = 10)
{
    if (token.empty()) return false;
    if (base == 16 && token.size() >= 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X')) token.remove_prefix(2);
    if (token.empty() || token[0] == '+' || token[0] == '-') return false;
    std::uint64_t result = 0;
    for (char c : token) {
        unsigned digit = 0;
        if (c >= '0' && c <= '9') digit = static_cast<unsigned>(c - '0');
        else if (base == 16 && c >= 'a' && c <= 'f') digit = static_cast<unsigned>(c - 'a' + 10);
        else if (base == 16 && c >= 'A' && c <= 'F') digit = static_cast<unsigned>(c - 'A' + 10);
        else return false;
        if (digit >= static_cast<unsigned>(base)) return false;
        if (result > (std::numeric_limits<std::uint64_t>::max() - digit) / static_cast<unsigned>(base)) return false;
        result = result * static_cast<unsigned>(base) + digit;
    }
    value = result;
    return true;
}

inline bool parse_integer(std::string_view token, std::int64_t& value)
{
    if (token.empty()) return false;
    const bool negative = token[0] == '-';
    if (negative) token.remove_prefix(1);
    std::uint64_t magnitude = 0;
    if (!parse_unsigned(token, magnitude)) return false;
    if (negative) {
        if (magnitude > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1) return false;
        value = magnitude == static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1
                    ? std::numeric_limits<std::int64_t>::min()
                    : -static_cast<std::int64_t>(magnitude);
    } else {
        if (magnitude > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) return false;
        value = static_cast<std::int64_t>(magnitude);
    }
    return true;
}

inline std::vector<std::string_view> pso_recipe_tokens(std::string_view line)
{
    std::vector<std::string_view> tokens;
    std::size_t at = 0;
    while (at < line.size()) {
        while (at < line.size() && line[at] == ' ') ++at;
        const std::size_t begin = at;
        while (at < line.size() && line[at] != ' ') ++at;
        if (begin != at) tokens.push_back(line.substr(begin, at - begin));
    }
    return tokens;
}

inline bool read_bool(const std::vector<std::string_view>& tokens, std::size_t at, bool& out)
{
    std::uint64_t value = 0;
    if (at >= tokens.size() || !parse_unsigned(tokens[at], value) || value > 1) return false;
    out = value != 0;
    return true;
}

inline bool read_bool(const std::vector<std::string_view>& tokens, std::size_t at, BOOL& out)
{
    bool value = false;
    if (!read_bool(tokens, at, value)) return false;
    out = value ? TRUE : FALSE;
    return true;
}

inline bool read_uint(const std::vector<std::string_view>& tokens, std::size_t at, unsigned& out)
{
    std::uint64_t value = 0;
    if (at >= tokens.size() || !parse_unsigned(tokens[at], value) || value > std::numeric_limits<unsigned>::max()) return false;
    out = static_cast<unsigned>(value);
    return true;
}

inline bool read_uint_range(const std::vector<std::string_view>& tokens, std::size_t at, unsigned low, unsigned high,
                            unsigned& out)
{
    return read_uint(tokens, at, out) && out >= low && out <= high;
}

inline bool read_hex32(const std::vector<std::string_view>& tokens, std::size_t at, std::uint32_t& out)
{
    std::uint64_t value = 0;
    if (at >= tokens.size() || !parse_unsigned(tokens[at], value, 16) || value > 0xffffffffull) return false;
    out = static_cast<std::uint32_t>(value);
    return true;
}

inline bool parse_pso_recipe_lines(const std::vector<std::string_view>& lines, std::size_t at, PsoRecipe& recipe,
                                   std::string& why)
{
    if (at + 20 > lines.size()) {
        why = "truncated record";
        return false;
    }
    if (lines[at] != "x2w-pso-recipe 1") {
        why = "unknown recipe version";
        return false;
    }
    recipe = {};
    auto line = [&](std::size_t index) { return pso_recipe_tokens(lines[at + index]); };

    auto tokens = line(1);
    if (tokens.size() != 2 || tokens[0] != "state" || !parse_unsigned(tokens[1], recipe.state_key, 16)) {
        why = "malformed state line";
        return false;
    }
    tokens = line(2);
    if (tokens.size() != 2 || tokens[0] != "translation" || !parse_unsigned(tokens[1], recipe.translation_key, 16)) {
        why = "malformed translation line";
        return false;
    }
    tokens = line(3);
    if (tokens.size() != 10 || tokens[0] != "targets" || !read_uint(tokens, 1, recipe.target_count) ||
        recipe.target_count > pso_recipe_target_limit) {
        why = "malformed targets line";
        return false;
    }
    for (unsigned i = 0; i < pso_recipe_target_limit; ++i) {
        unsigned format = 0;
        if (!read_uint(tokens, 2 + i, format) || format > 0xffff) {
            why = "malformed target format";
            return false;
        }
        recipe.target_formats[i] = static_cast<DXGI_FORMAT>(format);
    }
    tokens = line(4);
    unsigned dsv = 0;
    if (tokens.size() != 2 || tokens[0] != "dsv" || !read_uint(tokens, 1, dsv) || dsv > 0xffff) {
        why = "malformed dsv line";
        return false;
    }
    recipe.dsv_format = static_cast<DXGI_FORMAT>(dsv);
    tokens = line(5);
    if (tokens.size() != 2 || tokens[0] != "indexed" || !read_bool(tokens, 1, recipe.indexed_rect)) {
        why = "malformed indexed line";
        return false;
    }
    tokens = line(6);
    if (tokens.size() != 3 || tokens[0] != "blend" || !read_bool(tokens, 1, recipe.blend.AlphaToCoverageEnable) ||
        !read_bool(tokens, 2, recipe.blend.IndependentBlendEnable)) {
        why = "malformed blend line";
        return false;
    }
    for (unsigned i = 0; i < 8; ++i) {
        tokens = line(7 + i);
        const std::string expected = "blend" + std::to_string(i);
        auto& target = recipe.blend.RenderTarget[i];
        unsigned src = 0, dest = 0, op = 0, src_alpha = 0, dest_alpha = 0, op_alpha = 0, logic = 0, write_mask = 0;
        if (tokens.size() != 11 || tokens[0] != expected || !read_bool(tokens, 1, target.BlendEnable) ||
            !read_bool(tokens, 2, target.LogicOpEnable) || !read_uint_range(tokens, 3, 1, 19, src) ||
            !read_uint_range(tokens, 4, 1, 19, dest) || !read_uint_range(tokens, 5, 1, 5, op) ||
            !read_uint_range(tokens, 6, 1, 19, src_alpha) || !read_uint_range(tokens, 7, 1, 19, dest_alpha) ||
            !read_uint_range(tokens, 8, 1, 5, op_alpha) || !read_uint_range(tokens, 9, 0, 15, logic) ||
            !read_uint_range(tokens, 10, 0, 15, write_mask)) {
            why = "malformed " + expected + " line";
            return false;
        }
        target.SrcBlend = static_cast<D3D12_BLEND>(src);
        target.DestBlend = static_cast<D3D12_BLEND>(dest);
        target.BlendOp = static_cast<D3D12_BLEND_OP>(op);
        target.SrcBlendAlpha = static_cast<D3D12_BLEND>(src_alpha);
        target.DestBlendAlpha = static_cast<D3D12_BLEND>(dest_alpha);
        target.BlendOpAlpha = static_cast<D3D12_BLEND_OP>(op_alpha);
        target.LogicOp = static_cast<D3D12_LOGIC_OP>(logic);
        target.RenderTargetWriteMask = static_cast<UINT8>(write_mask);
    }
    tokens = line(15);
    unsigned depth_enable = 0, write_mask = 0, depth_func = 0, stencil_enable = 0, read_mask = 0, stencil_write_mask = 0;
    unsigned front_fail = 0, front_depth_fail = 0, front_pass = 0, front_func = 0;
    unsigned back_fail = 0, back_depth_fail = 0, back_pass = 0, back_func = 0;
    if (tokens.size() != 16 || tokens[0] != "depth" || !read_bool(tokens, 1, recipe.use_depth_stencil) ||
        !read_bool(tokens, 2, recipe.depth_stencil.DepthEnable) ||
        !read_uint_range(tokens, 3, 0, 1, write_mask) || !read_uint_range(tokens, 4, 1, 8, depth_func) ||
        !read_bool(tokens, 5, recipe.depth_stencil.StencilEnable) || !read_uint_range(tokens, 6, 0, 255, read_mask) ||
        !read_uint_range(tokens, 7, 0, 255, stencil_write_mask) || !read_uint_range(tokens, 8, 1, 8, front_fail) ||
        !read_uint_range(tokens, 9, 1, 8, front_depth_fail) || !read_uint_range(tokens, 10, 1, 8, front_pass) ||
        !read_uint_range(tokens, 11, 1, 8, front_func) || !read_uint_range(tokens, 12, 1, 8, back_fail) ||
        !read_uint_range(tokens, 13, 1, 8, back_depth_fail) || !read_uint_range(tokens, 14, 1, 8, back_pass) ||
        !read_uint_range(tokens, 15, 1, 8, back_func)) {
        why = "malformed depth line";
        return false;
    }
    recipe.depth_stencil.DepthWriteMask = static_cast<D3D12_DEPTH_WRITE_MASK>(write_mask);
    recipe.depth_stencil.DepthFunc = static_cast<D3D12_COMPARISON_FUNC>(depth_func);
    recipe.depth_stencil.StencilReadMask = static_cast<UINT8>(read_mask);
    recipe.depth_stencil.StencilWriteMask = static_cast<UINT8>(stencil_write_mask);
    recipe.depth_stencil.FrontFace.StencilFailOp = static_cast<D3D12_STENCIL_OP>(front_fail);
    recipe.depth_stencil.FrontFace.StencilDepthFailOp = static_cast<D3D12_STENCIL_OP>(front_depth_fail);
    recipe.depth_stencil.FrontFace.StencilPassOp = static_cast<D3D12_STENCIL_OP>(front_pass);
    recipe.depth_stencil.FrontFace.StencilFunc = static_cast<D3D12_COMPARISON_FUNC>(front_func);
    recipe.depth_stencil.BackFace.StencilFailOp = static_cast<D3D12_STENCIL_OP>(back_fail);
    recipe.depth_stencil.BackFace.StencilDepthFailOp = static_cast<D3D12_STENCIL_OP>(back_depth_fail);
    recipe.depth_stencil.BackFace.StencilPassOp = static_cast<D3D12_STENCIL_OP>(back_pass);
    recipe.depth_stencil.BackFace.StencilFunc = static_cast<D3D12_COMPARISON_FUNC>(back_func);

    tokens = line(16);
    unsigned fill = 0, cull = 0, conservative = 0;
    std::int64_t depth_bias = 0;
    std::uint32_t clamp_bits = 0, slope_bits = 0;
    if (tokens.size() != 13 || tokens[0] != "raster" || !read_bool(tokens, 1, recipe.has_rasterizer) ||
        !read_uint_range(tokens, 2, 2, 3, fill) || !read_uint_range(tokens, 3, 1, 3, cull) ||
        !read_bool(tokens, 4, recipe.rasterizer.FrontCounterClockwise) || !parse_integer(tokens[5], depth_bias) ||
        depth_bias < std::numeric_limits<INT>::min() || depth_bias > std::numeric_limits<INT>::max() ||
        !read_hex32(tokens, 6, clamp_bits) || !read_hex32(tokens, 7, slope_bits) ||
        !read_bool(tokens, 8, recipe.rasterizer.DepthClipEnable) ||
        !read_bool(tokens, 9, recipe.rasterizer.MultisampleEnable) ||
        !read_bool(tokens, 10, recipe.rasterizer.AntialiasedLineEnable) ||
        !read_uint(tokens, 11, recipe.rasterizer.ForcedSampleCount) || !read_uint_range(tokens, 12, 0, 2, conservative)) {
        why = "malformed raster line";
        return false;
    }
    recipe.rasterizer.FillMode = static_cast<D3D12_FILL_MODE>(fill);
    recipe.rasterizer.CullMode = static_cast<D3D12_CULL_MODE>(cull);
    recipe.rasterizer.DepthBias = static_cast<INT>(depth_bias);
    std::memcpy(&recipe.rasterizer.DepthBiasClamp, &clamp_bits, sizeof(clamp_bits));
    std::memcpy(&recipe.rasterizer.SlopeScaledDepthBias, &slope_bits, sizeof(slope_bits));
    recipe.rasterizer.ConservativeRaster = static_cast<D3D12_CONSERVATIVE_RASTERIZATION_MODE>(conservative);

    tokens = line(17);
    if (tokens.size() != 3 || tokens[0] != "sample" || !read_uint_range(tokens, 1, 1, 16, recipe.sample_count) ||
        !read_uint(tokens, 2, recipe.sample_mask)) {
        why = "malformed sample line";
        return false;
    }
    unsigned topology = 0;
    tokens = line(18);
    if (tokens.size() != 2 || tokens[0] != "topology" || !read_uint_range(tokens, 1, 0, 4, topology)) {
        why = "malformed topology line";
        return false;
    }
    recipe.topology = static_cast<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(topology);
    if (line(19).size() != 1 || line(19)[0] != "end") {
        why = "missing end line";
        return false;
    }

    // The serializer records a valid default for an unused state so that every field it writes is
    // a legal enum. The state key ignores both blocks when their flags are clear; normalize them
    // here so a replay compares the same zero/default values the creation path uses.
    if (!recipe.use_depth_stencil) recipe.depth_stencil = {};
    if (!recipe.has_rasterizer) recipe.rasterizer = default_pso_rasterizer();
    if (pso_recipe_state_key(recipe) != recipe.state_key) {
        why = "state key does not match the recorded state";
        return false;
    }
    return true;
}

inline PsoRecipeParseResult parse_pso_recipe_file(std::string_view text)
{
    PsoRecipeParseResult result;
    if (text.size() > pso_recipe_file_limit) {
        result.rejected = 1;
        result.error = "recipe file exceeds the size limit";
        return result;
    }
    std::vector<std::string_view> lines;
    std::size_t at = 0;
    while (at <= text.size()) {
        const std::size_t end = text.find('\n', at);
        std::string_view line = text.substr(at, end == std::string_view::npos ? std::string_view::npos : end - at);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (!line.empty()) lines.push_back(line);
        if (end == std::string_view::npos) break;
        at = end + 1;
    }
    std::size_t i = 0;
    while (i < lines.size()) {
        if (lines[i].substr(0, 15) != "x2w-pso-recipe ") {
            result.rejected++;
            if (result.error.empty()) result.error = "unexpected line outside a recipe";
            ++i;
            continue;
        }
        if (lines[i] != "x2w-pso-recipe 1") {
            result.rejected++;
            if (result.error.empty()) result.error = "unknown recipe version";
            i += (i + 20 <= lines.size()) ? 20 : 1;
            continue;
        }
        if (result.recipes.size() >= pso_recipe_record_limit) {
            result.rejected++;
            if (result.error.empty()) result.error = "recipe record limit reached";
            i += (i + 20 <= lines.size()) ? 20 : 1;
            continue;
        }
        PsoRecipe recipe;
        std::string why;
        if (!parse_pso_recipe_lines(lines, i, recipe, why)) {
            result.rejected++;
            if (result.error.empty()) result.error = why;
            i += (i + 20 <= lines.size()) ? 20 : 1;
            continue;
        }
        result.recipes.push_back(recipe);
        i += 20;
    }
    return result;
}

// The parser accepts at most pso_recipe_record_limit complete records. The writer must stop at
// the same boundary instead of appending a record the next process will reject. A file with any
// rejected line is left alone: one more append cannot make it parse cleanly.
inline bool pso_recipe_append_allowed(std::string_view existing, std::string_view record)
{
    if (record.empty() || record.size() > pso_recipe_file_limit) return false;
    // A parseable last record may still lack its final LF. Appending then joins its `end` to the
    // next header on one line, so the next read loses both records. Refuse without writing.
    if (!existing.empty() && existing.back() != '\n') return false;
    if (existing.size() > pso_recipe_file_limit || record.size() > pso_recipe_file_limit - existing.size())
        return false;
    const PsoRecipeParseResult parsed = parse_pso_recipe_file(existing);
    if (parsed.rejected != 0) return false;
    return parsed.recipes.size() < pso_recipe_record_limit;
}

}  // namespace x2w_startup
