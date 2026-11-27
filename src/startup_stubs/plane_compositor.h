// Experimental (docs/startup-runtime.md#presented-planes): the draw that composites the planes of a
// console present into one render target. The output window (d3d12_present.inl) records it into its
// back buffer; tests/startup_present_composite_warp.cpp records it into a texture it reads back.
//
// Contract: the steps come from composite_steps (present_planes.h), bottom to top. The first step
// drawn replaces the target with its plane's colour and ignores that plane's alpha (nothing is below
// it); each later step with `blended` set is drawn over the result with
//   colour = src.rgb * src.a + dst.rgb * (1 - src.a),  alpha = src.a + dst.a * (1 - src.a)
// (SRC_ALPHA / INV_SRC_ALPHA, ONE / INV_SRC_ALPHA), so after an opaque base the target alpha stays 1.
// A step without `blended` (or whose plane has no surface) after the first is drawn opaque or left
// out. Each plane is one full-target triangle sampled with a linear, clamped sampler: a plane the
// size of the target is copied texel for texel; a larger or smaller plane is scaled to fit.
// Plane surfaces must rest in COMMON and pass present_surface_supported (present_view_format.h); they
// are left in COMMON. The target must be in RENDER_TARGET.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <d3d12.h>
#include <d3dcompiler.h>

#include "startup_stubs/create_srv.h"
#include "startup_stubs/present_planes.h"
#include "startup_stubs/present_view_format.h"

namespace x2w_startup {

class PlaneCompositor {
public:
    // Create the opaque pipeline and a heap for up to `max_planes` views. More than one plane also needs a blend pipeline.
    HRESULT create(ID3D12Device* device, DXGI_FORMAT target_format, UINT max_planes)
    {
        release();
        max_planes_ = max_planes ? max_planes : 1;
        static const char source[] =
            "Texture2D<float4> surface : register(t0); SamplerState smooth : register(s0);\n"
            "struct V { float4 position : SV_Position; float2 uv : TEXCOORD0; };\n"
            "V vs(uint id : SV_VertexID) { V o; o.uv = float2((id << 1) & 2, id & 2); o.position = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1); return o; }\n"
            "float4 ps(V i) : SV_Target0 { return float4(surface.Sample(smooth, i.uv).rgb, 1); }\n"
            "float4 ps_over(V i) : SV_Target0 { return surface.Sample(smooth, i.uv); }\n";
        ID3DBlob* vs = nullptr;
        ID3DBlob* ps = nullptr;
        ID3DBlob* ps_over = nullptr;
        ID3DBlob* blob = nullptr;
        const D3D12_DESCRIPTOR_HEAP_DESC heap{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, max_planes_, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        HRESULT hr = device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&textures_));
        if (SUCCEEDED(hr)) hr = D3DCompile(source, sizeof(source) - 1, "present", nullptr, nullptr, "vs", "vs_5_0", 0, 0, &vs, nullptr);
        if (SUCCEEDED(hr)) hr = D3DCompile(source, sizeof(source) - 1, "present", nullptr, nullptr, "ps", "ps_5_0", 0, 0, &ps, nullptr);
        if (SUCCEEDED(hr) && max_planes_ > 1)
            hr = D3DCompile(source, sizeof(source) - 1, "present", nullptr, nullptr, "ps_over", "ps_5_0", 0, 0, &ps_over, nullptr);
        if (SUCCEEDED(hr)) {
            const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
            D3D12_ROOT_PARAMETER parameter{};
            parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameter.DescriptorTable = {1, &range};
            parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            D3D12_STATIC_SAMPLER_DESC sampler{};
            sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            sampler.MaxLOD = D3D12_FLOAT32_MAX;
            sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            const D3D12_ROOT_SIGNATURE_DESC desc{1, &parameter, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
            hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, nullptr);
            if (SUCCEEDED(hr)) hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_));
        }
        if (SUCCEEDED(hr)) {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = root_;
            desc.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
            desc.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            desc.SampleMask = ~0u;
            desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            desc.NumRenderTargets = 1;
            desc.RTVFormats[0] = target_format;
            desc.SampleDesc.Count = 1;
            hr = device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&opaque_));
            if (SUCCEEDED(hr) && ps_over) {
                desc.PS = {ps_over->GetBufferPointer(), ps_over->GetBufferSize()};
                D3D12_RENDER_TARGET_BLEND_DESC& blend = desc.BlendState.RenderTarget[0];
                blend.BlendEnable = TRUE;
                blend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
                blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                blend.BlendOp = D3D12_BLEND_OP_ADD;
                blend.SrcBlendAlpha = D3D12_BLEND_ONE;
                blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                hr = device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&blend_));
            }
        }
        if (vs) vs->Release();
        if (ps) ps->Release();
        if (ps_over) ps_over->Release();
        if (blob) blob->Release();
        if (FAILED(hr)) release();
        return hr;
    }

    // The pipeline a list can be reset with before record().
    ID3D12PipelineState* initial_pipeline() const { return opaque_; }
    UINT max_planes() const { return max_planes_; }

    // Records the composite into `list` (open) over `target` (width x height, in RENDER_TARGET).
    // steps[0..count) as from composite_steps; plane_surfaces[step.plane] is that plane's surface,
    // or null to leave the step out. Returns the number of planes drawn (0: nothing recorded).
    UINT record(ID3D12Device* device, ID3D12GraphicsCommandList* list, D3D12_CPU_DESCRIPTOR_HANDLE target, UINT width,
                UINT height, const PlaneStep* steps, unsigned count, ID3D12Resource* const* plane_surfaces,
                SrvLog srv_log = nullptr)
    {
        if (!opaque_ || !list || !steps || !plane_surfaces) return 0;
        ID3D12Resource* drawn[16] = {};
        bool blended[16] = {};
        UINT planes = 0;
        const UINT limit = max_planes_ < 16 ? max_planes_ : 16;
        for (unsigned i = 0; i < count && planes < limit; ++i) {
            ID3D12Resource* const surface = plane_surfaces[steps[i].plane];
            if (!surface) continue;
            blended[planes] = planes != 0 && steps[i].blended;  // the first plane drawn is the opaque base
            drawn[planes++] = surface;
        }
        if (!planes) return 0;
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        for (UINT i = 0; i < planes; ++i) {
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = present_view_format(drawn[i]->GetDesc().Format);
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            view.Texture2D.MipLevels = 1;
            x2w_create_srv(device, drawn[i], &view,
                           D3D12_CPU_DESCRIPTOR_HANDLE{textures_->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T{i} * step},
                           "plane-compositor", srv_log);
        }
        for (UINT i = 0; i < planes; ++i) transition(list, drawn[i], D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
        const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
        list->SetGraphicsRootSignature(root_);
        list->SetDescriptorHeaps(1, &textures_);
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &target, FALSE, nullptr);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        for (UINT i = 0; i < planes; ++i) {
            list->SetPipelineState(blended[i] ? blend_ : opaque_);
            list->SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE{textures_->GetGPUDescriptorHandleForHeapStart().ptr + UINT64{i} * step});
            list->DrawInstanced(3, 1, 0, 0);
        }
        for (UINT i = 0; i < planes; ++i) transition(list, drawn[i], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        return planes;
    }

    void release()
    {
        if (blend_) blend_->Release();
        if (opaque_) opaque_->Release();
        if (root_) root_->Release();
        if (textures_) textures_->Release();
        blend_ = opaque_ = nullptr;
        root_ = nullptr;
        textures_ = nullptr;
    }

private:
    static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = from;
        barrier.Transition.StateAfter = to;
        list->ResourceBarrier(1, &barrier);
    }

    ID3D12DescriptorHeap* textures_ = nullptr;
    ID3D12RootSignature* root_ = nullptr;
    ID3D12PipelineState* opaque_ = nullptr;  // the bottom plane, alpha ignored
    ID3D12PipelineState* blend_ = nullptr;   // later planes, over it by their alpha
    UINT max_planes_ = 0;
};

}  // namespace x2w_startup
