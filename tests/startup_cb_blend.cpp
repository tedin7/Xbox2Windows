// Colour-blend register decoding (src/startup_stubs/cb_blend.h, #1216).
//
// Part 1 checks the decoder's table: every BlendOp factor and CombFunc function, the alpha forms of
// colour factors, shared and separate alpha, ENABLE, the write mask, and the refusals.
//
// Part 2 draws: for each case a constant source colour over a cleared R32G32B32A32_FLOAT target with
// the decoded blend state, compared with a CPU model of the register fields (result = combine(src *
// source factor, dst * destination factor) per channel, written where the mask allows). The swapped
// control exchanges SrcBlend and DestBlend after decoding; it must disagree with the model.
#include "desktop_gpu.h"
#include "startup_cb_blend_ps.h"
#include "startup_cb_blend_vs.h"
#include "startup_stubs/cb_blend.h"
#include "startup_stubs/export_write_mask.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using x2w_cb_blend::control_word;

namespace {
using Color = std::array<float, 4>;

// BlendOp and CombFunc values (Mesa src/amd/registers/gfx10.json).
enum : std::uint32_t {
    ZERO = 0, ONE = 1, SRC_COLOR = 2, ONE_MINUS_SRC_COLOR = 3, SRC_ALPHA = 4, ONE_MINUS_SRC_ALPHA = 5,
    DST_ALPHA = 6, ONE_MINUS_DST_ALPHA = 7, DST_COLOR = 8, ONE_MINUS_DST_COLOR = 9, SRC_ALPHA_SATURATE = 10,
    BOTH_SRC_ALPHA = 11, BOTH_INV_SRC_ALPHA = 12, CONSTANT_COLOR = 13, ONE_MINUS_CONSTANT_COLOR = 14,
    SRC1_COLOR = 15, INV_SRC1_COLOR = 16, SRC1_ALPHA = 17, INV_SRC1_ALPHA = 18, CONSTANT_ALPHA = 19,
    ONE_MINUS_CONSTANT_ALPHA = 20,
};
enum : std::uint32_t { ADD = 0, SUB = 1, MIN = 2, MAX = 3, REV_SUB = 4 };

void expect(bool condition, const std::string& what)
{
    if (!condition) throw std::runtime_error("cb_blend decode: " + what);
}

void check_table()
{
    const D3D12_BLEND mapped[21] = {
        D3D12_BLEND_ZERO, D3D12_BLEND_ONE, D3D12_BLEND_SRC_COLOR, D3D12_BLEND_INV_SRC_COLOR, D3D12_BLEND_SRC_ALPHA,
        D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_DEST_ALPHA, D3D12_BLEND_INV_DEST_ALPHA, D3D12_BLEND_DEST_COLOR,
        D3D12_BLEND_INV_DEST_COLOR, D3D12_BLEND_SRC_ALPHA_SAT, static_cast<D3D12_BLEND>(0), static_cast<D3D12_BLEND>(0),
        D3D12_BLEND_BLEND_FACTOR, D3D12_BLEND_INV_BLEND_FACTOR, D3D12_BLEND_SRC1_COLOR, D3D12_BLEND_INV_SRC1_COLOR,
        D3D12_BLEND_SRC1_ALPHA, D3D12_BLEND_INV_SRC1_ALPHA, D3D12_BLEND_ALPHA_FACTOR, D3D12_BLEND_INV_ALPHA_FACTOR};
    int accepted = 0, refused = 0;
    for (std::uint32_t f = 0; f < 21; ++f) {
        for (bool alpha_factor : {false, true}) {
            const auto d = x2w_cb_blend::decode(control_word(f, ADD, ZERO, ONE, ADD, ZERO, true, true), 0xF, alpha_factor);
            const bool expect_refused = f == BOTH_SRC_ALPHA || f == BOTH_INV_SRC_ALPHA ||
                                        (!alpha_factor && (f == CONSTANT_ALPHA || f == ONE_MINUS_CONSTANT_ALPHA));
            expect(d.ok != expect_refused, "factor " + std::to_string(f) + (d.ok ? " accepted" : " refused"));
            if (d.ok) {
                expect(d.desc.SrcBlend == mapped[f], "factor " + std::to_string(f) + " maps to " + std::to_string(d.desc.SrcBlend));
                ++accepted;
            } else {
                ++refused;
            }
        }
    }
    // A colour factor in an alpha slot becomes its alpha form, explicitly and through shared alpha.
    const std::pair<std::uint32_t, D3D12_BLEND> alpha_forms[] = {
        {SRC_COLOR, D3D12_BLEND_SRC_ALPHA}, {ONE_MINUS_SRC_COLOR, D3D12_BLEND_INV_SRC_ALPHA},
        {DST_COLOR, D3D12_BLEND_DEST_ALPHA}, {ONE_MINUS_DST_COLOR, D3D12_BLEND_INV_DEST_ALPHA},
        {SRC1_COLOR, D3D12_BLEND_SRC1_ALPHA}, {INV_SRC1_COLOR, D3D12_BLEND_INV_SRC1_ALPHA},
        {CONSTANT_COLOR, D3D12_BLEND_BLEND_FACTOR}};
    for (const auto& [f, alpha] : alpha_forms) {
        const auto separate = x2w_cb_blend::decode(control_word(ONE, ADD, ZERO, f, ADD, ZERO, true, true), 0xF, false);
        const auto shared = x2w_cb_blend::decode(control_word(f, ADD, ZERO, ZERO, MAX, ZERO, false, true), 0xF, false);
        expect(separate.ok && separate.desc.SrcBlendAlpha == alpha, "separate alpha form of " + std::to_string(f));
        expect(shared.ok && shared.desc.SrcBlendAlpha == alpha && shared.desc.SrcBlend == mapped[f],
               "shared alpha form of " + std::to_string(f));
    }
    // Shared alpha ignores the alpha fields (MAX with ZERO factors there would be refused).
    const auto shared = x2w_cb_blend::decode(control_word(SRC_ALPHA, SUB, ONE_MINUS_SRC_ALPHA, ZERO, MAX, ZERO, false, true), 0xF, false);
    expect(shared.ok && shared.desc.BlendOpAlpha == D3D12_BLEND_OP_SUBTRACT && shared.desc.DestBlendAlpha == D3D12_BLEND_INV_SRC_ALPHA,
           "shared alpha takes the colour function and factors");
    const D3D12_BLEND_OP ops[5] = {D3D12_BLEND_OP_ADD, D3D12_BLEND_OP_SUBTRACT, D3D12_BLEND_OP_MIN, D3D12_BLEND_OP_MAX,
                                   D3D12_BLEND_OP_REV_SUBTRACT};
    for (std::uint32_t fcn = 0; fcn < 8; ++fcn) {
        const auto d = x2w_cb_blend::decode(control_word(ONE, fcn, ONE, ONE, fcn, ONE, true, true), 0xF, false);
        expect(d.ok == (fcn < 5), "combine " + std::to_string(fcn));
        if (d.ok) expect(d.desc.BlendOp == ops[fcn] && d.desc.BlendOpAlpha == ops[fcn], "combine " + std::to_string(fcn) + " maps");
    }
    // MIN / MAX keep their meaning only with ONE factors.
    expect(!x2w_cb_blend::decode(control_word(SRC_ALPHA, MIN, ONE, ONE, ADD, ZERO, true, true), 0xF, false).ok, "MIN with SRC_ALPHA");
    expect(!x2w_cb_blend::decode(control_word(ONE, ADD, ZERO, ONE, MAX, ZERO, true, true), 0xF, false).ok, "alpha MAX with ZERO");
    // ENABLE clear: unblended, mask kept, factors left at their defaults even if the fields hold others.
    const auto off = x2w_cb_blend::decode(control_word(BOTH_SRC_ALPHA, 7, SRC_ALPHA, ZERO, ADD, ZERO, true, false), 0x5, false);
    expect(off.ok && !off.desc.BlendEnable && off.desc.RenderTargetWriteMask == 0x5 && off.desc.SrcBlend == D3D12_BLEND_ONE,
           "disabled blend stays disabled");
    expect(!x2w_cb_blend::decode(0, 0x10, false).ok, "five-bit mask");
    // Words reported in #1216 from captured state blocks (ENABLE, SEPARATE_ALPHA_BLEND and bit 31,
    // DISABLE_ROP3, set). Bit 31 does not change the blend state.
    struct Observed { std::uint32_t word; D3D12_BLEND s, d, sa, da; };
    const Observed observed[] = {
        {0xE5040504, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA},
        {0xE5010504, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA},
        {0xE5010501, D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_ONE, D3D12_BLEND_INV_SRC_ALPHA},
        {0xE1000104, D3D12_BLEND_SRC_ALPHA, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_ONE}};
    for (const auto& o : observed) {
        const auto d = x2w_cb_blend::decode(o.word, 0xF, false);
        char name[32];
        std::snprintf(name, sizeof(name), "observed %08x", o.word);
        expect(d.ok && d.desc.BlendEnable && d.desc.SrcBlend == o.s && d.desc.DestBlend == o.d && d.desc.SrcBlendAlpha == o.sa &&
                   d.desc.DestBlendAlpha == o.da && d.desc.BlendOp == D3D12_BLEND_OP_ADD && d.desc.BlendOpAlpha == D3D12_BLEND_OP_ADD,
               name);
        expect(x2w_cb_blend::decode(o.word & 0x7FFFFFFFu, 0xF, false).desc.SrcBlend == d.desc.SrcBlend, std::string(name) + " without bit 31");
    }
    std::printf("startup_cb_blend table: %d factor forms accepted, %d refused; alpha forms, combine functions, "
                "min/max, enable and mask checked\n", accepted, refused);
}

float factor_value(std::uint32_t f, const Color& s, const Color& d, const Color& k, int channel)
{
    const bool alpha = channel == 3;
    switch (f) {
    case ZERO: return 0;
    case ONE: return 1;
    case SRC_COLOR: return s[channel];
    case ONE_MINUS_SRC_COLOR: return 1 - s[channel];
    case SRC_ALPHA: return s[3];
    case ONE_MINUS_SRC_ALPHA: return 1 - s[3];
    case DST_ALPHA: return d[3];
    case ONE_MINUS_DST_ALPHA: return 1 - d[3];
    case DST_COLOR: return d[channel];
    case ONE_MINUS_DST_COLOR: return 1 - d[channel];
    case SRC_ALPHA_SATURATE: return alpha ? 1.0f : (std::min)(s[3], 1 - d[3]);
    case CONSTANT_COLOR: return k[channel];
    case ONE_MINUS_CONSTANT_COLOR: return 1 - k[channel];
    case CONSTANT_ALPHA: return k[3];
    case ONE_MINUS_CONSTANT_ALPHA: return 1 - k[3];
    default: throw std::runtime_error("no model for factor " + std::to_string(f));
    }
}

// The register fields' meaning, independent of the decoder.
Color model(std::uint32_t control, std::uint32_t mask, const Color& s, const Color& d, const Color& k)
{
    Color out = d;
    const bool enable = (control >> 30) & 1, separate = (control >> 29) & 1;
    for (int c = 0; c < 4; ++c) {
        if (!((mask >> c) & 1)) continue;
        if (!enable) {
            out[c] = s[c];
            continue;
        }
        const bool use_alpha = c == 3 && separate;
        const std::uint32_t sf = use_alpha ? (control >> 16) & 0x1F : control & 0x1F;
        const std::uint32_t fcn = use_alpha ? (control >> 21) & 7 : (control >> 5) & 7;
        const std::uint32_t df = use_alpha ? (control >> 24) & 0x1F : (control >> 8) & 0x1F;
        const float a = s[c] * factor_value(sf, s, d, k, c), b = d[c] * factor_value(df, s, d, k, c);
        out[c] = fcn == ADD ? a + b : fcn == SUB ? a - b : fcn == MIN ? (std::min)(a, b) : fcn == MAX ? (std::max)(a, b) : b - a;
    }
    return out;
}

struct DrawCase {
    const char* name;
    std::uint32_t control;
    std::uint32_t mask;
    bool needs_alpha_factor;
};

Color draw(ID3D12Device* device, ID3D12RootSignature* root, const D3D12_RENDER_TARGET_BLEND_DESC& blend,
           const Color& source, const Color& destination, const Color& constant)
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root;
    pd.VS = {startup_cb_blend_vs, sizeof(startup_cb_blend_vs)};
    pd.PS = {startup_cb_blend_ps, sizeof(startup_cb_blend_ps)};
    pd.BlendState.RenderTarget[0] = blend;
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    pd.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create blend PSO");

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = rd.Height = 1;
    rd.DepthOrArraySize = rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R32G32B32A32_FLOAT};
    std::memcpy(clear.Color, destination.data(), sizeof(clear.Color));
    const auto target = resource(device, rd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&rd, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    const auto readback = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    ComPtr<ID3D12DescriptorHeap> rtvs;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs)), "Create blend RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    list->SetGraphicsRootSignature(root);
    list->SetGraphicsRoot32BitConstants(0, 4, source.data(), 0);
    list->OMSetBlendFactor(constant.data());
    const D3D12_VIEWPORT viewport{0, 0, 1, 1, 0, 1};
    const D3D12_RECT scissor{0, 0, 1, 1};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION to{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    to.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION from{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    commands.submit_and_wait(device);

    Color out{};
    void* mapped = nullptr;
    const D3D12_RANGE read{0, sizeof(out)}, none{0, 0};
    check(readback->Map(0, &read, &mapped), "Map blend readback");
    std::memcpy(out.data(), static_cast<std::uint8_t*>(mapped) + footprint.Offset, sizeof(out));
    readback->Unmap(0, &none);
    return out;
}

ComPtr<ID3D12RootSignature> create_root(ID3D12Device* device)
{
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants = {0, 0, 4};
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC desc{1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "Serialize blend root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "Create blend root");
    return root;
}
}  // namespace

// Returns the number of draw cases that differ from the model.
int check_startup_cb_blend(ID3D12Device* device, bool swapped)
{
    check_table();
    D3D12_FEATURE_DATA_D3D12_OPTIONS13 options13{};
    const bool alpha_factor = SUCCEEDED(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS13, &options13, sizeof(options13))) &&
                              options13.AlphaBlendFactorSupported;
    const DrawCase cases[] = {
        {"straight alpha", control_word(SRC_ALPHA, ADD, ONE_MINUS_SRC_ALPHA, 0, 0, 0, false, true), 0xF, false},
        {"premultiplied alpha", control_word(ONE, ADD, ONE_MINUS_SRC_ALPHA, 0, 0, 0, false, true), 0xF, false},
        {"straight colour, premultiplied alpha", control_word(SRC_ALPHA, ADD, ONE_MINUS_SRC_ALPHA, ONE, ADD, ONE_MINUS_SRC_ALPHA, true, true), 0xF, false},
        {"additive", control_word(ONE, ADD, ONE, 0, 0, 0, false, true), 0xF, false},
        {"modulate", control_word(DST_COLOR, ADD, ZERO, 0, 0, 0, false, true), 0xF, false},
        {"subtract", control_word(ONE, SUB, ONE, 0, 0, 0, false, true), 0xF, false},
        {"reverse subtract", control_word(SRC_ALPHA, REV_SUB, ONE, ONE, ADD, ZERO, true, true), 0xF, false},
        {"min / max", control_word(ONE, MIN, ONE, ONE, MAX, ONE, true, true), 0xF, false},
        {"constant colour", control_word(CONSTANT_COLOR, ADD, ONE_MINUS_CONSTANT_COLOR, 0, 0, 0, false, true), 0xF, false},
        {"saturate", control_word(SRC_ALPHA_SATURATE, ADD, ONE, ONE, ADD, ZERO, true, true), 0xF, false},
        {"inverse destination", control_word(ONE_MINUS_DST_COLOR, ADD, ONE_MINUS_DST_ALPHA, DST_ALPHA, ADD, ZERO, true, true), 0xF, false},
        {"blend, colour only", control_word(SRC_ALPHA, ADD, ONE_MINUS_SRC_ALPHA, 0, 0, 0, false, true), 0x7, false},
        {"disabled, red and alpha", control_word(SRC_ALPHA, ADD, ONE_MINUS_SRC_ALPHA, 0, 0, 0, false, false), 0x9, false},
        {"constant alpha", control_word(CONSTANT_ALPHA, ADD, ONE_MINUS_CONSTANT_ALPHA, 0, 0, 0, false, true), 0xF, true},
        // The four words reported in #1216, as captured (bit 31 set).
        {"observed e5040504", 0xE5040504u, 0xF, false},
        {"observed e5010504", 0xE5010504u, 0xF, false},
        {"observed e5010501", 0xE5010501u, 0xF, false},
        {"observed e1000104", 0xE1000104u, 0xF, false},
    };
    // Exact in binary32 for every sum and product above. Source alpha 0.75 keeps SRC_ALPHA and
    // INV_SRC_ALPHA apart, so the swapped control can see the straight-alpha cases.
    const Color source{0.75f, 0.5f, 0.25f, 0.75f}, destination{0.25f, 0.5f, 1.0f, 0.5f}, constant{0.5f, 0.25f, 0.75f, 0.25f};
    const auto root = create_root(device);
    int mismatches = 0, drawn = 0, skipped = 0;
    for (const auto& c : cases) {
        if (c.needs_alpha_factor && !alpha_factor) {
            std::printf("startup_cb_blend %s: skipped (no alpha blend factor support)\n", c.name);
            ++skipped;
            continue;
        }
        auto decoded = x2w_cb_blend::decode(c.control, c.mask, alpha_factor);
        if (!decoded.ok) throw std::runtime_error(std::string("cb_blend refused draw case ") + c.name + ": " + decoded.reason);
        if (swapped) {
            std::swap(decoded.desc.SrcBlend, decoded.desc.DestBlend);
            std::swap(decoded.desc.SrcBlendAlpha, decoded.desc.DestBlendAlpha);
        }
        const Color got = draw(device, root.Get(), decoded.desc, source, destination, constant);
        const Color want = model(c.control, c.mask, source, destination, constant);
        const bool same = std::equal(got.begin(), got.end(), want.begin(), [](float a, float b) { return std::fabs(a - b) <= 1e-6f; });
        mismatches += !same;
        ++drawn;
        std::printf("startup_cb_blend %s control=%08x mask=%x: got %g %g %g %g, model %g %g %g %g%s\n", c.name, c.control, c.mask,
                    got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3], same ? "" : "  MISMATCH");
    }
    std::printf("startup_cb_blend swapped=%d drawn=%d skipped=%d mismatches=%d\n", swapped ? 1 : 0, drawn, skipped, mismatches);
    return mismatches;
}

// Export write masks (src/startup_stubs/export_write_mask.h, #1674): a manifest 'wmask' line applied to an
// opaque blend state. The draw writes a constant source over a cleared target; the channels outside the mask
// must keep the cleared value and the others must take the source. With unmasked the mask is not applied, so
// every channel takes the source and every case must disagree. Returns the number of cases that differ from
// that expectation.
int check_startup_export_write_mask(ID3D12Device* device, bool unmasked)
{
    struct MaskCase {
        const char* line;
        unsigned mask;
    };
    const MaskCase cases[] = {{"wmask 0 0x3", 0x3}, {"wmask 0 0xc", 0xC}, {"wmask 0 0x5", 0x5}, {"wmask 0 0x8", 0x8}};
    // Every channel of the source differs from the cleared value, so each one shows whether it was written.
    const Color source{0.75f, 0.5f, 0.25f, 0.75f}, destination{0.125f, 0.875f, 1.0f, 0.0625f}, constant{0, 0, 0, 0};
    const auto root = create_root(device);
    int mismatches = 0;
    for (const auto& c : cases) {
        x2w_startup::ExportWriteMasks masks;
        if (!masks.read_manifest_line(c.line)) throw std::runtime_error(std::string("manifest line refused: ") + c.line);
        D3D12_BLEND_DESC blend{};
        blend.RenderTarget[0] = {FALSE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE,
                                 D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
        if (!unmasked && !x2w_startup::apply_export_write_masks(blend, masks, 1))
            throw std::runtime_error("single-target export mask refused");
        const Color got = draw(device, root.Get(), blend.RenderTarget[0], source, destination, constant);
        Color want{};
        for (int ch = 0; ch < 4; ++ch) want[ch] = (c.mask >> ch) & 1 ? source[ch] : destination[ch];
        const bool same = got == want;
        mismatches += !same;
        std::printf("startup_export_write_mask %s: got %g %g %g %g, written channels -> %g %g %g %g%s\n", c.line, got[0], got[1],
                    got[2], got[3], want[0], want[1], want[2], want[3], same ? "" : "  MISMATCH");
    }
    std::printf("startup_export_write_mask unmasked=%d cases=%zu mismatches=%d\n", unmasked ? 1 : 0, std::size(cases), mismatches);
    return mismatches;
}
