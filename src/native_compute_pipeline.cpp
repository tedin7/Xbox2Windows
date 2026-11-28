#include "native_compute_pipeline.h"
#include "direct_compute_cache.h"
#include "startup_stubs/translated_sampler_bindings.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace x2w {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
bool is_uav(const ComputeViewBinding &binding) { return compute_uav(binding.kind); }
} // namespace

NativeComputePipeline create_native_compute_pipeline(ID3D12Device *device, ShaderDxilCompiler &compiler,
                                                     const std::filesystem::path &marker, unsigned entry,
                                                     const std::filesystem::path &cache, bool direct_dxil,
                                                     HeapViewContract heap_view) {
    if (!direct_dxil && heap_view != HeapViewContract::none)
        throw std::runtime_error("a heap view contract needs the direct DXIL path");
    NativeComputePipeline pipeline;
    auto start = Clock::now();
    auto program = direct_dxil
        ? prepare_direct_compute(marker,entry,marker.extension()==L".cdx",cache,pipeline.frontend_cache_hit,heap_view)
        : translate_compute(read_compute_request(marker), entry, direct_dxil);
    pipeline.instructions = program.instruction_count;
    pipeline.bindings = std::move(program.bindings);
    std::sort(pipeline.bindings.begin(), pipeline.bindings.end(), [](auto &a, auto &b) {
        if (is_uav(a) != is_uav(b))
            return !is_uav(a);
        return a.shader_register < b.shader_register;
    });

    UINT counts[2]{};
    for (const auto &binding : pipeline.bindings) {
        if (binding.kind == ComputeViewKind::heap_sampler) {
            pipeline.sampler_heap_indexed = true; // SamplerDescriptorHeap reads: no table register
            continue;
        }
        if (compute_heap_binding(binding.kind)) {
            pipeline.heap_indexed = true; // ResourceDescriptorHeap reads: no table register
            continue;
        }
        auto &count = counts[is_uav(binding)];
        if (binding.shader_register != count++)
            throw std::runtime_error("noncontiguous compute register range");
    }
    pipeline.uav_count = counts[1];
    pipeline.constants = std::move(program.constants);
    pipeline.stride_srvs = std::move(program.stride_srvs);
    for (const auto &block : pipeline.constants)
        pipeline.constant_words += block.count;
    pipeline.srv_count = counts[0];
    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
    for (const auto &sampler : program.samplers) {
        samplers.push_back(x2w_startup::decode_sampler(sampler.words, static_cast<UINT>(samplers.size()), sampler.comparison,
                                                       sampler.explicit_lod));
    }
    pipeline.frontend_ms = elapsed(start);
    start = Clock::now();
    auto shader = compiler.compile(direct_dxil ? program.dxil_ir : program.hlsl, L"cs_6_6", cache,
                                   direct_dxil ? ShaderSourceFormat::dxil_ir : ShaderSourceFormat::hlsl);
    pipeline.backend_ms = elapsed(start);
    pipeline.cache_hit = shader.cache_hit;

    start = Clock::now();
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    for (const auto &b : pipeline.bindings)
        pipeline.atomic_views += b.kind == ComputeViewKind::indexed_atomic_uav;
    const auto b0_words = pipeline.constant_words + static_cast<unsigned>(pipeline.stride_srvs.size());
    if (unsigned(bool(counts[0])) + unsigned(bool(counts[1])) + b0_words +
        2 * pipeline.atomic_views > D3D12_MAX_ROOT_COST)
        throw UnsupportedComputeShader("indexed atomic view metadata exceeds root parameter budget");
    std::vector<D3D12_ROOT_PARAMETER> params(3 + pipeline.atomic_views);
    unsigned param_count = 0;
    for (UINT i = 0; i < 2; ++i) {
        if (!counts[i])
            continue;
        auto &param = params[param_count++];
        ranges[i].RangeType = i ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[i].NumDescriptors = counts[i];
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        param.DescriptorTable = {1, &ranges[i]};
    }
    if (b0_words) {
        auto &param = params[param_count++];
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        param.Constants.Num32BitValues = b0_words;
    }
    for (const auto &b : pipeline.bindings) {
        if (b.kind != ComputeViewKind::indexed_atomic_uav) continue;
        auto &param = params[param_count++];
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        param.Constants = {b.shader_register + 1, 0, 2};
    }
    const UINT root_flags = (pipeline.heap_indexed ? 0x400u : 0u)           // CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                            | (pipeline.sampler_heap_indexed ? 0x800u : 0u); // SAMPLER_HEAP_DIRECTLY_INDEXED
    D3D12_ROOT_SIGNATURE_DESC desc{param_count, params.data(), static_cast<UINT>(samplers.size()), samplers.data(),
                                   static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(root_flags)};
    Microsoft::WRL::ComPtr<ID3DBlob> blob;
    if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr)) ||
        FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                           IID_PPV_ARGS(&pipeline.root))))
        throw std::runtime_error("native compute root signature creation failed");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = pipeline.root.Get();
    pso.CS = {shader.dxil.data(), shader.dxil.size()};
    if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline.state))))
        throw std::runtime_error("native compute PSO creation failed");
    pipeline.pipeline_ms = elapsed(start);
    return pipeline;
}

void NativeComputePipeline::bind(ID3D12GraphicsCommandList *list, ID3D12DescriptorHeap *heap,
                                 D3D12_GPU_DESCRIPTOR_HANDLE first, UINT increment, const unsigned *words,
                                 const std::vector<IndexedAtomicView> &atomic, ID3D12DescriptorHeap *sampler_heap,
                                 const std::vector<unsigned> &strides) const {
    if (constant_words && !words)
        throw std::runtime_error("compute root constants missing");
    if (atomic.size() != atomic_views)
        throw std::runtime_error("indexed atomic view metadata missing");
    if (strides.size() != stride_srvs.size())
        throw std::runtime_error("compute descriptor stride metadata missing");
    const bool views = srv_count || uav_count || heap_indexed;
    if (sampler_heap_indexed) {
        if (!sampler_heap)
            throw std::runtime_error("a sampler heap is needed");
        if (views) {
            if (!heap)
                throw std::runtime_error("a view heap is needed");
            // Views first, then samplers, as bind_translated_heaps does.
            ID3D12DescriptorHeap *heaps[2]{heap, sampler_heap};
            list->SetDescriptorHeaps(2, heaps);
        } else
            list->SetDescriptorHeaps(1, &sampler_heap);
    } else if (views)
        list->SetDescriptorHeaps(1, &heap);
    list->SetComputeRootSignature(root.Get());
    list->SetPipelineState(state.Get());
    unsigned parameter = 0;
    if (srv_count)
        list->SetComputeRootDescriptorTable(parameter++, first);
    first.ptr += UINT64(srv_count) * increment;
    if (uav_count)
        list->SetComputeRootDescriptorTable(parameter++, first);
    if (constant_words)
        list->SetComputeRoot32BitConstants(parameter, constant_words, words, 0);
    if (!strides.empty())
        list->SetComputeRoot32BitConstants(parameter, static_cast<UINT>(strides.size()), strides.data(), constant_words);
    if (constant_words || !strides.empty()) ++parameter;
    for (const auto &shape : atomic) {
        const unsigned pair[]{shape.stride, shape.count};
        list->SetComputeRoot32BitConstants(parameter++, 2, pair, 0);
    }
}
} // namespace x2w
