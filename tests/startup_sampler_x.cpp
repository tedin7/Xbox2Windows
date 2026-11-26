// Desktop sampler contract for the startup CreateSamplerX row (#874). The console descriptor layout
// is not established here; this fixture checks only the desktop D3D12_SAMPLER_DESC fields that a
// translated sampler must reproduce: filter, per-axis address mode, border colour and LOD limits.
#include "desktop_gpu.h"
#include "startup_sampler_x_cs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT texture_size = 8;
constexpr UINT mip_levels = 4;
constexpr float border_value = 1.0f;  // OPAQUE_WHITE red channel; every texel value is at least 2.
// Linear probes use 0.25-step texel fractions and a 0.5 LOD fraction, which fixed-point filter
// weights represent exactly. D3D12 only requires the bilinear weights and the mip weight to carry
// 8 fractional bits (D3D11.3 functional spec, texture filtering precision), so a sampler may move a weight
// by one 1/256 step. The tolerance is that step times the texel range the weight blends, plus float rounding.
constexpr float linear_tolerance = 0.01f;
constexpr float weight_step = 1.0f / 256.0f;

struct Probe {
    float u, v, lod;
};
// Coordinates are odd multiples of 1/16, so no point probe falls on a texel edge at any mip.
constexpr std::array<Probe, 8> point_probes{{
    {0.0625f, 0.0625f, 0.0f},
    {1.5625f, 0.0625f, 0.0f},
    {-0.3125f, 0.0625f, 0.0f},
    {0.0625f, 1.5625f, 0.0f},
    {0.0625f, -0.3125f, 0.0f},
    {1.5625f, -0.3125f, 0.0f},
    {0.5625f, 0.5625f, 2.0f},
    {0.5625f, 0.5625f, 8.0f},
}};
// Midpoints between texel centres on each axis and on both, then two fractional LODs.
constexpr std::array<Probe, 5> linear_probes{{
    {0.125f, 0.0625f, 0.0f},
    {0.0625f, 0.125f, 0.0f},
    {0.125f, 0.125f, 0.0f},
    {0.0625f, 0.0625f, 0.5f},
    {0.125f, 0.0625f, 1.5f},
}};

struct SamplerCase {
    D3D12_FILTER filter;
    D3D12_TEXTURE_ADDRESS_MODE u, v;
    float min_lod, max_lod;
};
constexpr SamplerCase defaults{D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                               D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f, D3D12_FLOAT32_MAX};
// Tables 0 and 1 share the point probes and trade descriptor slots in the swapped control.
constexpr std::array<SamplerCase, 3> samplers{{
    {D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_MIRROR, 1.0f, 2.0f},
    {D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_BORDER, 0.0f, 0.0f},
    {D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f,
     D3D12_FLOAT32_MAX},
}};
constexpr UINT word_count = static_cast<UINT>(2 * point_probes.size() + linear_probes.size());

bool is_linear(const SamplerCase& s)
{
    return s.filter == D3D12_FILTER_MIN_MAG_MIP_LINEAR;
}

const Probe* table_probes(UINT table, UINT& count)
{
    if (table < 2) {
        count = static_cast<UINT>(point_probes.size());
        return point_probes.data();
    }
    count = static_cast<UINT>(linear_probes.size());
    return linear_probes.data();
}

float texel_value(UINT mip, int x, int y)
{
    return static_cast<float>(2 + mip * 64 + y * 8 + x);
}

// Returns the texel index on one axis, or -1 for a border read.
int address(D3D12_TEXTURE_ADDRESS_MODE mode, int i, int n)
{
    switch (mode) {
    case D3D12_TEXTURE_ADDRESS_MODE_WRAP: return ((i % n) + n) % n;
    case D3D12_TEXTURE_ADDRESS_MODE_MIRROR: {
        const int m = ((i % (2 * n)) + 2 * n) % (2 * n);
        return m < n ? m : 2 * n - 1 - m;
    }
    case D3D12_TEXTURE_ADDRESS_MODE_CLAMP: return i < 0 ? 0 : (i >= n ? n - 1 : i);
    case D3D12_TEXTURE_ADDRESS_MODE_BORDER: return i < 0 || i >= n ? -1 : i;
    default: throw std::runtime_error("Sampler reference has no address mode");
    }
}

float fetch(const SamplerCase& s, UINT mip, int i, int j)
{
    const int n = static_cast<int>(texture_size >> mip);
    const int x = address(s.u, i, n);
    const int y = address(s.v, j, n);
    return x < 0 || y < 0 ? border_value : texel_value(mip, x, y);
}

float bilinear(const SamplerCase& s, UINT mip, float u, float v)
{
    const float n = static_cast<float>(texture_size >> mip);
    const float tx = u * n - 0.5f, ty = v * n - 0.5f;
    const int x = static_cast<int>(std::floor(tx)), y = static_cast<int>(std::floor(ty));
    const float fx = tx - x, fy = ty - y;
    const float top = fetch(s, mip, x, y) * (1 - fx) + fetch(s, mip, x + 1, y) * fx;
    const float bottom = fetch(s, mip, x, y + 1) * (1 - fx) + fetch(s, mip, x + 1, y + 1) * fx;
    return top * (1 - fy) + bottom * fy;
}

// Point probes use integer LODs, so nearest-mip rounding is not exercised.
float reference(const SamplerCase& s, const Probe& p)
{
    const float lod = (std::min)((std::max)(p.lod, s.min_lod), (std::min)(s.max_lod, float(mip_levels - 1)));
    const UINT mip = static_cast<UINT>(lod);
    if (!is_linear(s)) {
        const int n = static_cast<int>(texture_size >> mip);
        return fetch(s, mip, static_cast<int>(std::floor(p.u * n)), static_cast<int>(std::floor(p.v * n)));
    }
    const UINT next = (std::min)(mip + 1, mip_levels - 1);
    const float f = lod - mip;
    return bilinear(s, mip, p.u, p.v) * (1 - f) + bilinear(s, next, p.u, p.v) * f;
}

// Largest value difference inside the 2x2 footprint of one mip: what a one-step weight error is multiplied by.
float footprint_span(const SamplerCase& s, UINT mip, float u, float v)
{
    const float n = static_cast<float>(texture_size >> mip);
    const int x = static_cast<int>(std::floor(u * n - 0.5f)), y = static_cast<int>(std::floor(v * n - 0.5f));
    const float a = fetch(s, mip, x, y), b = fetch(s, mip, x + 1, y);
    const float c = fetch(s, mip, x, y + 1), d = fetch(s, mip, x + 1, y + 1);
    return (std::max)((std::max)(a, b), (std::max)(c, d)) - (std::min)((std::min)(a, b), (std::min)(c, d));
}

// Point filtering is exact. Linear filtering may move each 8-bit weight by one step: the two bilinear weights
// act within a mip, and the mip weight acts across the difference between the two bilinear results.
float tolerance_for(const SamplerCase& s, const Probe& p)
{
    if (!is_linear(s)) return 0.0f;
    const float lod = (std::min)((std::max)(p.lod, s.min_lod), (std::min)(s.max_lod, float(mip_levels - 1)));
    const UINT mip = static_cast<UINT>(lod);
    const UINT next = (std::min)(mip + 1, mip_levels - 1);
    const float mip_gap = std::fabs(bilinear(s, next, p.u, p.v) - bilinear(s, mip, p.u, p.v));
    const float span = footprint_span(s, mip, p.u, p.v) + footprint_span(s, next, p.u, p.v);
    return linear_tolerance + weight_step * (2.0f * span + mip_gap);
}

bool probes_differ(const SamplerCase& a, const SamplerCase& b)
{
    for (const auto& p : point_probes) {
        if (reference(a, p) != reference(b, p)) return true;
    }
    return false;
}

// For the two point samplers, each field that differs from its desktop default must change at least
// one probe when replaced by any other address mode or by that default. Otherwise the readback could
// not tell a dropped field from a translated one.
void check_probe_coverage()
{
    constexpr D3D12_TEXTURE_ADDRESS_MODE modes[]{D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
                                                 D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_BORDER};
    for (UINT i = 0; i < 2; ++i) {
        const auto& s = samplers[i];
        for (const auto mode : modes) {
            auto changed = s;
            changed.u = mode;
            if (mode != s.u && !probes_differ(s, changed)) throw std::runtime_error("Probes do not separate AddressU");
            changed = s;
            changed.v = mode;
            if (mode != s.v && !probes_differ(s, changed)) throw std::runtime_error("Probes do not separate AddressV");
        }
        auto changed = s;
        changed.min_lod = defaults.min_lod;
        if (s.min_lod != defaults.min_lod && !probes_differ(s, changed))
            throw std::runtime_error("Probes do not separate MinLOD");
        changed = s;
        changed.max_lod = defaults.max_lod;
        if (s.max_lod != defaults.max_lod && !probes_differ(s, changed))
            throw std::runtime_error("Probes do not separate MaxLOD");
    }
    if (!probes_differ(samplers[0], samplers[1])) throw std::runtime_error("Probes do not separate the two samplers");
    // A linear result must not be reachable by point sampling at either neighbouring mip.
    for (const auto& p : linear_probes) {
        auto point = samplers[2];
        point.filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        const float linear = reference(samplers[2], p);
        for (const float lod : {std::floor(p.lod), std::ceil(p.lod)}) {
            for (const float du : {-0.0625f, 0.0625f}) {
                for (const float dv : {-0.0625f, 0.0625f}) {
                    if (reference(point, {p.u + du, p.v + dv, lod}) == linear)
                        throw std::runtime_error("Linear probe matches a point sample");
                }
            }
        }
    }
}

D3D12_SAMPLER_DESC desktop_sampler(const SamplerCase& s)
{
    D3D12_SAMPLER_DESC desc{};
    desc.Filter = s.filter;
    desc.AddressU = s.u;
    desc.AddressV = s.v;
    desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    desc.MaxAnisotropy = 1;
    desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    desc.BorderColor[0] = desc.BorderColor[1] = desc.BorderColor[2] = desc.BorderColor[3] = border_value;
    desc.MinLOD = s.min_lod;
    desc.MaxLOD = s.max_lod;
    return desc;
}

ComPtr<ID3D12RootSignature> create_root(ID3D12Device* device)
{
    const D3D12_DESCRIPTOR_RANGE srv{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    const D3D12_DESCRIPTOR_RANGE sampler{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[4]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1, &srv};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &sampler};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[3].Constants = {0, 0, 4};
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = _countof(params);
    desc.pParameters = params;
    ComPtr<ID3DBlob> signature;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, nullptr),
          "Serialize sampler contract root signature");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&root)),
          "Create sampler contract root signature");
    return root;
}
}

bool check_startup_sampler_x(ID3D12Device* device, bool swapped)
{
    check_probe_coverage();
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{DXGI_FORMAT_R32_FLOAT};
    check(check_feature(device, D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)), "Query R32_FLOAT support");
    if (!(support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE) || !(support.Support1 & D3D12_FORMAT_SUPPORT1_MIP))
        return false;
    const auto root = create_root(device);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc{};
    pipeline_desc.pRootSignature = root.Get();
    pipeline_desc.CS = {startup_sampler_x_cs, sizeof(startup_sampler_x_cs)};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateComputePipelineState(&pipeline_desc, IID_PPV_ARGS(&pipeline)), "Create sampler contract PSO");

    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = td.Height = texture_size;
    td.DepthOrArraySize = 1;
    td.MipLevels = mip_levels;
    td.SampleDesc.Count = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    const auto texture = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, mip_levels> footprints{};
    UINT64 upload_size = 0;
    device->GetCopyableFootprints(&td, 0, mip_levels, 0, footprints.data(), nullptr, nullptr, &upload_size);
    const auto upload = make_buffer(device, upload_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    const D3D12_RANGE no_access{0, 0};
    void* mapped = nullptr;
    check(upload->Map(0, &no_access, &mapped), "Map sampler contract texels");
    for (UINT mip = 0; mip < mip_levels; ++mip) {
        const auto& f = footprints[mip];
        for (UINT y = 0; y < f.Footprint.Height; ++y) {
            auto* row = reinterpret_cast<float*>(static_cast<std::uint8_t*>(mapped) + f.Offset + y * f.Footprint.RowPitch);
            for (UINT x = 0; x < f.Footprint.Width; ++x) row[x] = texel_value(mip, static_cast<int>(x), static_cast<int>(y));
        }
    }
    const D3D12_RANGE written{0, static_cast<SIZE_T>(upload_size)};
    upload->Unmap(0, &written);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views, sampler_heap;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "Create sampler contract view heap");
    device->CreateShaderResourceView(texture.Get(), nullptr, views->GetCPUDescriptorHandleForHeapStart());
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    hd.NumDescriptors = static_cast<UINT>(samplers.size());
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&sampler_heap)), "Create sampler contract sampler heap");
    const UINT sampler_stride = device->GetDescriptorHandleIncrementSize(hd.Type);
    // The negative control writes the two point samplers into each other's slots; the dispatches stay
    // unchanged.
    auto bound_case = [swapped](UINT table) { return table < 2 ? table ^ static_cast<UINT>(swapped) : table; };
    for (UINT i = 0; i < samplers.size(); ++i) {
        auto handle = sampler_heap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += SIZE_T{bound_case(i)} * sampler_stride;
        const auto desc = desktop_sampler(samplers[i]);
        device->CreateSampler(&desc, handle);
    }

    using Words = std::array<float, word_count>;
    const auto output = make_buffer(device, sizeof(Words), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto readback = make_buffer(device, sizeof(Words), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands commands(device, pipeline.Get());
    auto* list = commands.list.Get();
    for (UINT mip = 0; mip < mip_levels; ++mip) {
        D3D12_TEXTURE_COPY_LOCATION dst{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        dst.SubresourceIndex = mip;
        D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        src.PlacedFootprint = footprints[mip];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    transition(list, texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(list, output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetComputeRootSignature(root.Get());
    ID3D12DescriptorHeap* heaps[]{views.Get(), sampler_heap.Get()};
    list->SetDescriptorHeaps(2, heaps);
    list->SetComputeRootDescriptorTable(0, views->GetGPUDescriptorHandleForHeapStart());
    list->SetComputeRootUnorderedAccessView(2, output->GetGPUVirtualAddress());
    UINT word = 0;
    for (UINT table = 0; table < samplers.size(); ++table) {
        auto gpu = sampler_heap->GetGPUDescriptorHandleForHeapStart();
        gpu.ptr += UINT64{table} * sampler_stride;
        list->SetComputeRootDescriptorTable(1, gpu);
        UINT count = 0;
        const Probe* probes = table_probes(table, count);
        for (UINT p = 0; p < count; ++p, ++word) {
            std::uint32_t constants[4];
            std::memcpy(&constants[0], &probes[p].u, sizeof(float));
            std::memcpy(&constants[1], &probes[p].v, sizeof(float));
            std::memcpy(&constants[2], &probes[p].lod, sizeof(float));
            constants[3] = word * sizeof(float);
            list->SetComputeRoot32BitConstants(3, 4, constants, 0);
            list->Dispatch(1, 1, 1);
            // Each dispatch writes a distinct word, so no UAV barrier is needed between probes.
        }
    }
    transition(list, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyResource(readback.Get(), output.Get());
    commands.submit_and_wait(device);

    Words actual{};
    const D3D12_RANGE read{0, sizeof(actual)};
    check(readback->Map(0, &read, &mapped), "Map sampler contract readback");
    std::memcpy(actual.data(), mapped, sizeof(actual));
    readback->Unmap(0, &no_access);

    UINT mismatches = 0;
    word = 0;
    for (UINT table = 0; table < samplers.size(); ++table) {
        // In the swapped control, table 0 holds sampler 1 and table 1 holds sampler 0.
        const auto& bound = samplers[bound_case(table)];
        UINT count = 0;
        const Probe* probes = table_probes(table, count);
        std::printf("startup_sampler_x swapped=%u table=%u values=", static_cast<unsigned>(swapped), table);
        for (UINT p = 0; p < count; ++p, ++word) {
            const float got = actual[word];
            const float want = reference(bound, probes[p]);
            std::printf("%s%g", p ? "," : "", got);
            if (std::fabs(got - want) <= tolerance_for(bound, probes[p])) continue;
            if (!mismatches++) {
                std::fprintf(stderr, "First sampler mismatch: table %u probe %u (u=%g v=%g lod=%g) expected %g, got %g\n",
                             table, p, probes[p].u, probes[p].v, probes[p].lod, want, got);
            }
        }
        std::printf("\n");
    }
    std::printf("startup_sampler_x swapped=%u mismatches=%u of %u\n", static_cast<unsigned>(swapped), mismatches,
                word_count);
    if (mismatches) throw std::runtime_error("Sampler contract readback differs from reference");
    return true;
}
