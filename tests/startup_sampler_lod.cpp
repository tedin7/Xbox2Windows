// Sampler word 1 LOD limits through the startup stand-in's production decoder (#2221).
//
// Two sampler heap entries differ only in word 1: entry 0 has MIN_LOD 2 (MAX_LOD 4095/256) and entry 1 has
// MAX_LOD 1 (MIN_LOD 0). Both keep MinLOD <= MaxLOD, as D3D12_SAMPLER_DESC requires, and the test checks it. They
// are snapshotted from title memory and decoded with dynamic_sampler() into the stand-in's sampler table
// (create_sampler_snapshot_heap), behind the stand-in's translated root signature. A source-built pixel
// shader reads an 8x8 R32_FLOAT texture whose four mips each hold one value, with SampleLevel at an
// explicit LOD, one pixel per draw. A static comparison sampler from an smpc line (LESS) is read with
// SampleCmpLevelZero at two references and must keep its comparison function.
//
// The dropped-clamp control decodes the same words and then resets MinLOD and MaxLOD to 0 and
// D3D12_FLOAT32_MAX, the values table entries had before #2221. It must read mip 0 for entry 0 at LOD 0 and
// mip 3 for entry 1 at LOD 3, the wrong mips.
#include "desktop_gpu.h"
#include "startup_sampler_lod_ps.h"
#include "startup_sampler_lod_vs.h"
#include "startup_stubs/translated_sampler_bindings.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT texture_size = 8;
constexpr UINT mip_levels = 4;

float mip_value(UINT mip)
{
    return 0.125f + 0.25f * static_cast<float>(mip);
}

// Word 0: CLAMP_LAST_TEXEL in X and Y. Word 2: point filters. Word 1: MIN_LOD bits 11:0 and MAX_LOD bits
// 23:12, unsigned 4.8 fixed point.
constexpr std::uint32_t clamp_xy = 2u | 2u << 3;
constexpr x2w_startup::SamplerWords heap_words[2]{
    {clamp_xy, 0x200u | 0xfffu << 12, 0, 0},  // MIN_LOD 2, MAX_LOD 4095/256
    {clamp_xy, 0x100u << 12, 0, 0},           // MIN_LOD 0, MAX_LOD 1
};
// LESS (word 0 bits 14:12 = 1), MAX_LOD 1.
constexpr const char* comparison_line = "smpc 0 0x1012 0x100000 0x0 0x0";

struct Probe {
    UINT mode;  // 0: heap entry, 1: static comparison sampler
    UINT entry;
    float lod;
    float reference;
    float expected;
    float dropped;  // what the same draw reads with the clamp dropped
};
// Mip m holds 0.125 + 0.25 m. The comparison texel (mip 0) is 0.125.
const std::array<Probe, 6> probes{{
    {0, 0, 0.0f, 0.0f, mip_value(2), mip_value(0)},  // MIN_LOD 2 raises LOD 0 to mip 2
    {0, 0, 3.0f, 0.0f, mip_value(3), mip_value(3)},
    {0, 1, 3.0f, 0.0f, mip_value(1), mip_value(3)},  // MAX_LOD 1 lowers LOD 3 to mip 1
    {0, 1, 0.0f, 0.0f, mip_value(0), mip_value(0)},
    {1, 0, 0.0f, 0.0625f, 1.0f, 1.0f},               // 0.0625 < 0.125
    {1, 0, 0.0f, 0.25f, 0.0f, 0.0f},                 // 0.25 >= 0.125
}};

int mip_of(float value)
{
    for (UINT mip = 0; mip < mip_levels; ++mip)
        if (value == mip_value(mip)) return static_cast<int>(mip);
    return -1;
}
}  // namespace

int check_startup_sampler_lod(ID3D12Device* device, bool drop_clamp)
{
    using namespace x2w_startup;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{DXGI_FORMAT_R32_FLOAT};
    check(check_feature(device, D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)), "Query R32_FLOAT support");
    const auto needed = D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE | D3D12_FORMAT_SUPPORT1_MIP |
                        D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE_COMPARISON | D3D12_FORMAT_SUPPORT1_RENDER_TARGET;
    if ((support.Support1 & needed) != needed) return 77;

    TranslatedSamplers bindings;
    if (!bindings.read_manifest_line("smpheap 2048 1 fallback 2047") || !bindings.read_manifest_line(comparison_line))
        throw std::runtime_error("Sampler LOD manifest refused");
    const auto statics = decode_statics(bindings);
    if (statics.size() != 1 || statics[0].ComparisonFunc != D3D12_COMPARISON_FUNC_LESS)
        throw std::runtime_error("Comparison sampler lost its function");
    std::printf("static comparison sampler: Filter=0x%x ComparisonFunc=%u MinLOD=%g MaxLOD=%g\n", statics[0].Filter,
                statics[0].ComparisonFunc, statics[0].MinLOD, statics[0].MaxLOD);

    ComPtr<ID3D12RootSignature> root;
    check(create_translated_root(device, 4, 1, bindings, root.GetAddressOf()), "Create translated sampler LOD root");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.VS = {startup_sampler_lod_vs, sizeof(startup_sampler_lod_vs)};
    pd.PS = {startup_sampler_lod_ps, sizeof(startup_sampler_lod_ps)};
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R32_FLOAT;
    pd.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create sampler LOD PSO");

    // The two entries as title memory, decoded by the production path.
    const auto snapshot = snapshot_samplers({reinterpret_cast<std::uintptr_t>(heap_words), 2}, true);
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    NullSamplerCounts counts;
    check(create_sampler_snapshot_heap(device, snapshot, [&](const SamplerWords& words, D3D12_SAMPLER_DESC& out) {
        out = dynamic_sampler(words);
        if (out.MinLOD > out.MaxLOD) throw std::runtime_error("Fixture sampler has MinLOD above MaxLOD");
        std::printf("heap entry words %08x %08x: MinLOD=%g MaxLOD=%g", words[0], words[1], out.MinLOD, out.MaxLOD);
        if (drop_clamp) {
            out.MinLOD = 0.0f;
            out.MaxLOD = D3D12_FLOAT32_MAX;
            std::printf(", dropped to MinLOD=0 MaxLOD=max");
        }
        std::printf("\n");
        return true;
    }, sampler_heap.GetAddressOf(), counts), "Create sampler LOD heap");
    if (counts.unbound || counts.unreadable || counts.invalid_token || counts.out_of_range != translated_sampler_fallback - 2)
        throw std::runtime_error("Unexpected null sampler counts");

    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = td.Height = texture_size;
    td.DepthOrArraySize = 1;
    td.MipLevels = mip_levels;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    td.SampleDesc.Count = 1;
    const auto texture = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, mip_levels> footprints{};
    UINT64 upload_size = 0;
    device->GetCopyableFootprints(&td, 0, mip_levels, 0, footprints.data(), nullptr, nullptr, &upload_size);
    const auto upload = make_buffer(device, upload_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    const D3D12_RANGE none{0, 0};
    void* mapped = nullptr;
    check(upload->Map(0, &none, &mapped), "Map sampler LOD texels");
    for (UINT mip = 0; mip < mip_levels; ++mip) {
        const auto& f = footprints[mip];
        for (UINT y = 0; y < f.Footprint.Height; ++y) {
            auto* row = reinterpret_cast<float*>(static_cast<std::uint8_t*>(mapped) + f.Offset + y * f.Footprint.RowPitch);
            for (UINT x = 0; x < f.Footprint.Width; ++x) row[x] = mip_value(mip);
        }
    }
    const D3D12_RANGE written{0, static_cast<SIZE_T>(upload_size)};
    upload->Unmap(0, &written);

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = static_cast<UINT64>(probes.size());
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R32_FLOAT;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R32_FLOAT, {-1.0f, -1.0f, -1.0f, -1.0f}};
    const auto target = resource(device, rd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT target_footprint{};
    UINT64 readback_size = 0;
    device->GetCopyableFootprints(&rd, 0, 1, 0, &target_footprint, nullptr, nullptr, &readback_size);
    const auto readback = make_buffer(device, readback_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views, rtvs;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "Create sampler LOD view heap");
    device->CreateShaderResourceView(texture.Get(), nullptr, views->GetCPUDescriptorHandleForHeapStart());
    hd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs)), "Create sampler LOD RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    for (UINT mip = 0; mip < mip_levels; ++mip) {
        D3D12_TEXTURE_COPY_LOCATION dst{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        dst.SubresourceIndex = mip;
        D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        src.PlacedFootprint = footprints[mip];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    transition(list, texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    list->SetGraphicsRootSignature(root.Get());
    bind_translated_heaps(list, views.Get(), sampler_heap.Get());
    list->SetGraphicsRootDescriptorTable(1, views->GetGPUDescriptorHandleForHeapStart());
    list->SetGraphicsRootDescriptorTable(sampler_root_parameter(1), sampler_heap->GetGPUDescriptorHandleForHeapStart());
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    for (UINT i = 0; i < probes.size(); ++i) {
        std::uint32_t constants[4]{probes[i].mode, probes[i].entry};
        std::memcpy(&constants[2], &probes[i].lod, sizeof(float));
        std::memcpy(&constants[3], &probes[i].reference, sizeof(float));
        list->SetGraphicsRoot32BitConstants(0, 4, constants, 0);
        const D3D12_VIEWPORT viewport{static_cast<float>(i), 0, 1, 1, 0, 1};
        const D3D12_RECT scissor{static_cast<LONG>(i), 0, static_cast<LONG>(i + 1), 1};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->DrawInstanced(3, 1, 0, 0);
    }
    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION out{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    out.PlacedFootprint = target_footprint;
    D3D12_TEXTURE_COPY_LOCATION in{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&out, 0, 0, 0, &in, nullptr);
    commands.submit_and_wait(device);

    std::array<float, probes.size()> actual{};
    const D3D12_RANGE read{0, sizeof(actual)};
    check(readback->Map(0, &read, &mapped), "Map sampler LOD readback");
    std::memcpy(actual.data(), static_cast<std::uint8_t*>(mapped) + target_footprint.Offset, sizeof(actual));
    readback->Unmap(0, &none);

    UINT mismatches = 0, wrong_mip = 0, unexpected = 0;
    for (UINT i = 0; i < probes.size(); ++i) {
        const Probe& p = probes[i];
        const float expected = drop_clamp ? p.dropped : p.expected;
        if (p.mode == 0) {
            std::printf("startup_sampler_lod probe=%u entry=%u lod=%g read=%g mip=%d expected_mip=%d\n", i, p.entry,
                        p.lod, actual[i], mip_of(actual[i]), mip_of(p.expected));
        } else {
            std::printf("startup_sampler_lod probe=%u comparison=LESS reference=%g read=%g expected=%g\n", i,
                        p.reference, actual[i], p.expected);
        }
        mismatches += actual[i] != p.expected;
        wrong_mip += p.mode == 0 && actual[i] != p.expected && actual[i] == p.dropped;
        unexpected += actual[i] != expected;
    }
    if (drop_clamp) {
        std::printf("startup_sampler_lod dropped_clamp wrong_mip=%u unexpected=%u\n", wrong_mip, unexpected);
        return wrong_mip == 2 && !unexpected ? 0 : 1;
    }
    std::printf("startup_sampler_lod mismatches=%u\n", mismatches);
    return mismatches ? 1 : 0;
}
