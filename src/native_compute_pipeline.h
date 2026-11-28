#pragma once

#include "gfx1013_compute_native.h"
#include "shader_dxil_compiler.h"
#include <d3d12.h>

namespace x2w {

// Bindings are in desktop heap order: t0..tN followed by u0..uM.
// Each entry retains the guest table and byte offset determined by translation.
struct NativeComputePipeline {
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> state;
    std::vector<ComputeViewBinding> bindings;
    std::vector<ComputeConstants> constants;
    std::vector<unsigned> stride_srvs;
    unsigned srv_count = 0;
    unsigned uav_count = 0;
    unsigned constant_words = 0, atomic_views = 0;
    unsigned instructions = 0;
    bool cache_hit = false, frontend_cache_hit = false;
    bool heap_indexed = false; // reads ResourceDescriptorHeap: the root signature indexes the heap directly, which must be set
                               // on the list (bind() does it) and holds the selectable slots from index 0
    bool sampler_heap_indexed = false; // reads SamplerDescriptorHeap: the root has the directly indexed sampler heap flag (0x800),
                                       // and bind() needs the shader-visible sampler heap (entry n = title byte offset 16 * n)
    double frontend_ms = 0;
    double backend_ms = 0;
    double pipeline_ms = 0;

    void bind(ID3D12GraphicsCommandList *list, ID3D12DescriptorHeap *heap, D3D12_GPU_DESCRIPTOR_HANDLE first,
              UINT increment, const unsigned *words = nullptr,
              const std::vector<IndexedAtomicView> &atomic = {}, ID3D12DescriptorHeap *sampler_heap = nullptr,
              const std::vector<unsigned> &strides = {}) const;
};

NativeComputePipeline create_native_compute_pipeline(ID3D12Device *device, ShaderDxilCompiler &compiler,
                                                     const std::filesystem::path &marker, unsigned entry,
                                                     const std::filesystem::path &cache, bool direct_dxil = false,
                                                     HeapViewContract heap_view = HeapViewContract::none);

} // namespace x2w
