#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include "heap_indexed_root.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace x2w_startup {

using Microsoft::WRL::ComPtr;
using SamplerWords = std::array<std::uint32_t, 4>;
constexpr UINT translated_sampler_count = D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE;
constexpr UINT translated_sampler_space = 1;
constexpr UINT translated_sampler_fallback = translated_sampler_count - 1;
constexpr UINT title_sampler_stride = sizeof(SamplerWords);

// comparison: the sampler is read through a SamplerComparisonState (manifest line smpc). SQ_IMG_SAMP_WORD0 bits
// 14:12 hold SQ_TEX_DEPTH_COMPARE, which lists NEVER..ALWAYS in the order of D3D12_COMPARISON_FUNC minus one.
// lod_limits carries the unsigned 4.8 MIN_LOD (word 1 bits 11:0) and MAX_LOD (bits 23:12)
// for heap entries and smpl static samplers.
inline D3D12_STATIC_SAMPLER_DESC decode_sampler(const SamplerWords& words, UINT shader_register, bool comparison = false,
                                                bool lod_limits = false)
{
    static const D3D12_TEXTURE_ADDRESS_MODE modes[8] = {
        D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_MIRROR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE,
        D3D12_TEXTURE_ADDRESS_MODE_BORDER, D3D12_TEXTURE_ADDRESS_MODE_BORDER};
    const UINT mag = (words[2] >> 20) & 3, min = (words[2] >> 22) & 3, mip = (words[2] >> 26) & 3;
    D3D12_STATIC_SAMPLER_DESC s{};
    s.Filter = (mag >= 2 || min >= 2) ? D3D12_FILTER_ANISOTROPIC
        : D3D12_ENCODE_BASIC_FILTER(min & 1, mag & 1, mip == 2 ? 1 : 0, D3D12_FILTER_REDUCTION_TYPE_STANDARD);
    s.AddressU = modes[words[0] & 7];
    s.AddressV = modes[(words[0] >> 3) & 7];
    s.AddressW = modes[(words[0] >> 6) & 7];
    s.MaxAnisotropy = 1u << ((words[0] >> 9) & 7);
    if (s.MaxAnisotropy > 16) s.MaxAnisotropy = 16;
    s.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    if (comparison) {
        s.Filter = D3D12_ENCODE_BASIC_FILTER(min & 1, mag & 1, mip == 2 ? 1 : 0, D3D12_FILTER_REDUCTION_TYPE_COMPARISON);
        s.ComparisonFunc = static_cast<D3D12_COMPARISON_FUNC>(((words[0] >> 12) & 7) + 1);
    }
    // SQ_IMG_SAMP_WORD3 bits 31:30 (BORDER_COLOR_TYPE, Mesa gfx10-rsrc.json): 0 = TRANS_BLACK, 1 = OPAQUE_BLACK,
    // 2 = OPAQUE_WHITE. Value 3 (REGISTER, indexed by BORDER_COLOR_PTR) has no palette here and falls back to 0.
    const UINT border_type = (words[3] >> 30) & 3u;
    s.BorderColor = border_type == 2u ? D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE
                  : border_type == 1u ? D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK
                                      : D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    s.MaxLOD = D3D12_FLOAT32_MAX;
    if (lod_limits) {
        s.MinLOD = static_cast<float>(words[1] & 0xfffu) / 256.0f;
        s.MaxLOD = static_cast<float>((words[1] >> 12) & 0xfffu) / 256.0f;
    }
    s.ShaderRegister = shader_register;
    s.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    return s;
}

struct TranslatedSamplers {
    std::vector<SamplerWords> statics;
    std::vector<bool> comparison;  // parallel to statics: bound as SamplerComparisonState
    std::vector<bool> lod_limits;  // parallel to statics: LOD limits and primary bias carried over (line smpl)
    bool heap = false;
    bool reserved_fallback = false;

    bool read_manifest_line(const char* line)
    {
        if (std::strncmp(line, "smpheap", 7) == 0) {
            unsigned count = 0, space = 0, fallback = 0;
            char policy[16]{};
            char trailing = 0;
            const int fields = std::sscanf(line + 7, "%u %u %15s %u %c", &count, &space, policy, &fallback, &trailing);
            if ((line[7] != ' ' && line[7] != '\t') || heap ||
                (fields != 2 && (fields != 4 || std::strcmp(policy, "fallback") != 0 ||
                                 fallback != translated_sampler_fallback)) ||
                count != translated_sampler_count || space != translated_sampler_space)
                return false;
            heap = true;
            reserved_fallback = fields == 4;
        } else {
            int index = 0;
            unsigned words[4] = {};
            const bool compare = std::strncmp(line, "smpc", 4) == 0;
            const bool limits = std::strncmp(line, "smpl", 4) == 0;
            if (std::sscanf(line, compare ? "smpc %d %x %x %x %x" : limits ? "smpl %d %x %x %x %x" : "smp %d %x %x %x %x",
                            &index, &words[0], &words[1], &words[2], &words[3]) == 5) {
                statics.push_back({words[0], words[1], words[2], words[3]});
                comparison.push_back(compare);
                lod_limits.push_back(limits);
            }
        }
        return true;
    }
};

// RDNA 2 ISA Table 46: word 2 bits 13:0 are the signed primary LOD bias with eight fractional bits.
inline float decode_sampler_lod_bias(const SamplerWords& words)
{
    const int bias = static_cast<int>(words[2] & 0x3fffu);
    return static_cast<float>((bias ^ 0x2000) - 0x2000) / 256.0f;
}

// The static sampler bound at register i, in manifest order.
inline std::vector<D3D12_STATIC_SAMPLER_DESC> decode_statics(const TranslatedSamplers& samplers)
{
    std::vector<D3D12_STATIC_SAMPLER_DESC> statics;
    for (UINT i = 0; i < samplers.statics.size(); ++i) {
        const bool limits = i < samplers.lod_limits.size() && samplers.lod_limits[i];
        const auto& words = samplers.statics[i];
        auto sampler = decode_sampler(words, i, i < samplers.comparison.size() && samplers.comparison[i], limits);
        if (limits) {
            // The translator validates the D3D12 range and requires zero secondary bias.
            sampler.MipLODBias = decode_sampler_lod_bias(words);
        }
        statics.push_back(sampler);
    }
    return statics;
}

// views: the SRVs and UAVs of the view table together.
inline UINT sampler_root_parameter(UINT views) { return views ? 2u : 1u; }

// Parameter 1, when there are views, is one table: SRVs t0.. first, then UAVs u0.. right after them.
inline HRESULT create_translated_root(ID3D12Device* device, UINT constants, UINT srvs,
                                      const TranslatedSamplers& samplers, ID3D12RootSignature** out, UINT uavs = 0,
                                      bool dynamic_heap = false)
{
    D3D12_ROOT_PARAMETER parameters[3]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants.Num32BitValues = constants;
    // A compute program's outputs (u0..) follow its resources in the same table.
    const D3D12_DESCRIPTOR_RANGE view_ranges[2]{{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, srvs, 0, 0, 0},
                                                {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, uavs, 0, 0, srvs}};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    // A range of no descriptors does not serialize: a program without resources starts the table at its outputs.
    parameters[1].DescriptorTable = {(srvs ? 1u : 0u) + (uavs ? 1u : 0u), srvs ? view_ranges : view_ranges + 1};
    const D3D12_DESCRIPTOR_RANGE sampler_range{
        D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, translated_sampler_count, 0, translated_sampler_space, 0};
    UINT count = sampler_root_parameter(srvs + uavs);
    if (samplers.heap) {
        parameters[count].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[count].DescriptorTable = {1, &sampler_range};
        ++count;
    }
    const auto statics = decode_statics(samplers);
    const D3D12_ROOT_SIGNATURE_DESC desc{count, parameters, static_cast<UINT>(statics.size()), statics.data(),
                                         dynamic_heap ? static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(root_flag_view_heap_indexed)
                                                      : D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, nullptr);
    if (SUCCEEDED(hr)) hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(out));
    return hr;
}

struct SamplerSource {
    std::uintptr_t address = 0;
    UINT count = 0;
};

struct SamplerSnapshot {
    std::array<SamplerWords, translated_sampler_count> words{};
    std::bitset<translated_sampler_count> readable;
    UINT count = 0;
    bool reserved_fallback = false;

    bool operator==(const SamplerSnapshot& other) const
    {
        return reserved_fallback == other.reserved_fallback && count == other.count &&
               readable == other.readable && words == other.words;
    }
};

inline SamplerSnapshot snapshot_samplers(SamplerSource source, bool reserved_fallback)
{
    SamplerSnapshot snapshot;
    snapshot.reserved_fallback = reserved_fallback;
    if (!source.address || source.address > UINTPTR_MAX - translated_sampler_count * title_sampler_stride) return snapshot;
    const UINT capacity = reserved_fallback ? translated_sampler_fallback : translated_sampler_count;
    snapshot.count = source.count < capacity ? source.count : capacity;
    // Read page-sized groups without touching inaccessible memory in-process. A
    // single unaligned record can cross a page, so give it its own read.
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    const auto page_size = std::uintptr_t{system.dwPageSize};
    for (UINT i = 0; i < snapshot.count;) {
        const auto address = source.address + std::uintptr_t{i} * title_sampler_stride;
        const UINT room = static_cast<UINT>((page_size - address % page_size) / title_sampler_stride);
        const UINT available = snapshot.count - i;
        const UINT run = room ? (room < available ? room : available) : 1;
        SIZE_T copied = 0;
        const SIZE_T bytes = SIZE_T{run} * title_sampler_stride;
        if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), &snapshot.words[i], bytes, &copied) &&
            copied == bytes) {
            for (UINT j = i; j < i + run; ++j) snapshot.readable.set(j);
        } else {
            // ReadProcessMemory can copy part of a record on failure.
            for (UINT j = i; j < i + run; ++j) snapshot.words[j] = {};
        }
        i += run;
    }
    return snapshot;
}

inline D3D12_SAMPLER_DESC dynamic_sampler(const SamplerWords& words)
{
    // A heap entry is the title's own descriptor, so its MIN_LOD, MAX_LOD (#2221) and fixed border color apply.
    auto s = decode_sampler(words, 0, false, true);
    // MIP_FILTER 0 selects the view's base level for title heap descriptors.
    if (((words[2] >> 26) & 3u) == 0) {
        s.MinLOD = 0.0f;
        s.MaxLOD = 0.0f;
    }
    const float rgb = s.BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE ? 1.0f : 0.0f;
    const float alpha = (s.BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE ||
                         s.BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK) ? 1.0f : 0.0f;
    const float bias = std::clamp(decode_sampler_lod_bias(words), D3D12_MIP_LOD_BIAS_MIN, D3D12_MIP_LOD_BIAS_MAX);
    return {s.Filter, s.AddressU, s.AddressV, s.AddressW, bias, s.MaxAnisotropy, s.ComparisonFunc,
            {rgb, rgb, rgb, alpha}, s.MinLOD, s.MaxLOD};
}

inline D3D12_SAMPLER_DESC border_sampler()
{
    // The synthetic fallback keeps the full mip range despite its zero MIP_FILTER field.
    auto s = dynamic_sampler({6u | (6u << 3) | (6u << 6), 0, 0, 0});
    s.MaxLOD = D3D12_FLOAT32_MAX;
    return s;
}

struct NullSamplerCounts {
    UINT unbound = 0;
    UINT out_of_range = 0;
    UINT unreadable = 0;
    UINT invalid_token = 0;
};

// A resolver returns false for a readable entry that cannot be represented.
template<class Resolve>
HRESULT create_sampler_snapshot_heap(ID3D12Device* device, const SamplerSnapshot& snapshot, Resolve resolve,
                                     ID3D12DescriptorHeap** out, NullSamplerCounts& counts,
                                     const D3D12_SAMPLER_DESC* reserved = nullptr)
{
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, translated_sampler_count,
                                          D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    const HRESULT hr = device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(out));
    if (FAILED(hr)) return hr;
    counts = {};
    const auto fallback = border_sampler();
    auto dest = (*out)->GetCPUDescriptorHandleForHeapStart();
    const UINT stride = device->GetDescriptorHandleIncrementSize(desc.Type);
    const UINT capacity = snapshot.reserved_fallback ? translated_sampler_fallback : translated_sampler_count;
    for (UINT i = 0; i < capacity; ++i) {
        D3D12_SAMPLER_DESC sampler = fallback;
        if (!snapshot.count) ++counts.unbound;
        else if (i >= snapshot.count) ++counts.out_of_range;
        else if (!snapshot.readable.test(i)) ++counts.unreadable;
        else if (!resolve(snapshot.words[i], sampler)) {
            ++counts.invalid_token;
            sampler = fallback;
        }
        device->CreateSampler(&sampler, dest);
        dest.ptr += stride;
    }
    if (snapshot.reserved_fallback) device->CreateSampler(reserved ? reserved : &fallback, dest);
    return S_OK;
}

inline void bind_translated_heaps(ID3D12GraphicsCommandList* list, ID3D12DescriptorHeap* views,
                                  ID3D12DescriptorHeap* samplers)
{
    ID3D12DescriptorHeap* heaps[]{views, samplers};
    list->SetDescriptorHeaps(samplers ? 2 : 1, heaps);
}

}  // namespace x2w_startup
