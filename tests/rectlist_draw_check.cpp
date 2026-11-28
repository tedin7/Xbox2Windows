#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <fcntl.h>
#include <io.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <string_view>
#include <stdexcept>

#include "rectlist_vs.h"
#include "rectlist_ps.h"

using Microsoft::WRL::ComPtr;

namespace {

struct Vertex {
    float pos[4];
    float col[4];
};

void check(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) {
        std::fprintf(stderr, "D3D12 check failed at stage '%s': 0x%08lx\n", stage, hr);
        std::exit(1);
    }
}

constexpr UINT kSide = 64;

// Separate analytical reference comparator for the rectangle [-0.5, 0.5] x [-0.5, 0.5]
// in normalized device coordinates (NDC) mapped to a 64x64 viewport.
unsigned count_rectangle_mismatches(const uint32_t* pixels)
{
    unsigned mismatches = 0;
    for (UINT y = 0; y < kSide; ++y) {
        for (UINT x = 0; x < kSide; ++x) {
            const float px = (static_cast<float>(x) + 0.5f) / 32.0f - 1.0f;
            const float py = 1.0f - (static_cast<float>(y) + 0.5f) / 32.0f;

            const uint32_t rgba = pixels[y * kSide + x];
            const float r = static_cast<float>(rgba & 0xFF) / 255.0f;
            const float g = static_cast<float>((rgba >> 8) & 0xFF) / 255.0f;
            const float b = static_cast<float>((rgba >> 16) & 0xFF) / 255.0f;
            const float a = static_cast<float>((rgba >> 24) & 0xFF) / 255.0f;

            // Interior of the rectangle with 2-pixel margin away from edge quantization
            if (px > -0.46f && px < 0.46f && py > -0.46f && py < 0.46f) {
                const float u = (px - (-0.5f)) / (0.5f - (-0.5f));
                const float v = (py - (-0.5f)) / (0.5f - (-0.5f));
                const float exp_r = 0.2f + 0.6f * u;
                const float exp_g = 0.1f + 0.8f * v;
                const float exp_b = 0.5f;
                const float exp_a = 1.0f;

                if (std::abs(r - exp_r) > 0.05f ||
                    std::abs(g - exp_g) > 0.05f ||
                    std::abs(b - exp_b) > 0.05f ||
                    std::abs(a - exp_a) > 0.05f) {
                    mismatches++;
                }
            } else if (px < -0.54f || px > 0.54f || py < -0.54f || py > 0.54f) {
                // Strictly outside the rectangle: must remain clear color (all 0)
                if (rgba != 0) {
                    mismatches++;
                }
            }
        }
    }
    return mismatches;
}

void execute_draw(
    ID3D12Device* device,
    ID3D12CommandQueue* queue,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list,
    ID3D12PipelineState* pipeline,
    ID3D12RootSignature* root,
    ID3D12Resource* target,
    ID3D12Resource* readback,
    ID3D12Resource* upload_vb,
    const D3D12_CPU_DESCRIPTOR_HANDLE& rtv,
    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint,
    ID3D12Fence* fence,
    UINT64& fence_value,
    const Vertex* vertices,
    UINT vertex_count,
    std::vector<uint32_t>& out_pixels)
{
    // Upload vertices
    void* mapped = nullptr;
    const D3D12_RANGE empty_range{};
    check(upload_vb->Map(0, &empty_range, &mapped), "Map upload VB");
    std::memcpy(mapped, vertices, sizeof(Vertex) * vertex_count);
    upload_vb->Unmap(0, nullptr);

    check(allocator->Reset(), "Reset allocator");
    check(list->Reset(allocator, pipeline), "Reset list");

    D3D12_RESOURCE_BARRIER to_rt{};
    to_rt.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_rt.Transition = {target, 0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET};
    list->ResourceBarrier(1, &to_rt);

    const float clear_color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    list->ClearRenderTargetView(rtv, clear_color, 0, nullptr);

    const D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(kSide), static_cast<float>(kSide), 0.0f, 1.0f};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(kSide), static_cast<LONG>(kSide)};

    list->SetGraphicsRootSignature(root);
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    const D3D12_VERTEX_BUFFER_VIEW vbv{
        upload_vb->GetGPUVirtualAddress(),
        static_cast<UINT>(sizeof(Vertex) * vertex_count),
        sizeof(Vertex)
    };
    list->IASetVertexBuffers(0, 1, &vbv);
    list->DrawInstanced(vertex_count, 1, 0, 0);

    D3D12_RESOURCE_BARRIER to_copy{};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition = {target, 0, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &to_copy);

    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource = target;
    src_loc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION dst_loc{};
    dst_loc.pResource = readback;
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_loc.PlacedFootprint = footprint;

    list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);

    D3D12_RESOURCE_BARRIER to_common{};
    to_common.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_common.Transition = {target, 0, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON};
    list->ResourceBarrier(1, &to_common);

    check(list->Close(), "Close list");
    ID3D12CommandList* submitted[]{list};
    queue->ExecuteCommandLists(1, submitted);

    fence_value++;
    check(queue->Signal(fence, fence_value), "Signal fence");
    check(fence->SetEventOnCompletion(fence_value, nullptr), "Wait fence");

    D3D12_RANGE read_range{0, footprint.Offset + footprint.Footprint.RowPitch * kSide};
    check(readback->Map(0, &read_range, &mapped), "Map readback");

    out_pixels.resize(kSide * kSide);
    const auto* byte_ptr = static_cast<const uint8_t*>(mapped) + footprint.Offset;
    for (UINT y = 0; y < kSide; ++y) {
        const auto* row = reinterpret_cast<const uint32_t*>(byte_ptr + y * footprint.Footprint.RowPitch);
        for (UINT x = 0; x < kSide; ++x) {
            out_pixels[y * kSide + x] = row[x];
        }
    }
    readback->Unmap(0, nullptr);
}

} // namespace

int main(int argc, char* argv[])
{
    const bool use_warp = (argc < 2 || std::string_view(argv[1]) != "hardware");

    ComPtr<IDXGIFactory4> factory;
    check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "Create DXGI factory");

    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;

    if (use_warp) {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
            FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
            std::fputs("Could not initialize D3D12 WARP device\n", stderr);
            return 1;
        }
    } else {
        ComPtr<IDXGIAdapter1> hw_adapter;
        for (UINT i = 0; factory->EnumAdapters1(i, &hw_adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            hw_adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(hw_adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
                adapter = hw_adapter;
                break;
            }
        }
        if (!device) {
            std::fputs("No compatible hardware GPU available (capability skip)\n", stdout);
            return 77;
        }
    }

    // 1. Root signature
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ComPtr<ID3DBlob> root_blob;
    check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, nullptr), "Serialize root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(), IID_PPV_ARGS(&root)), "Create root");

    // 2. Graphics Pipeline State with CULLING ENABLED
    const D3D12_INPUT_ELEMENT_DESC input_elements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = root.Get();
    pso.VS = {rectlist_vs, sizeof(rectlist_vs)};
    pso.PS = {rectlist_ps, sizeof(rectlist_ps)};
    pso.InputLayout = {input_elements, 2};
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;         // CULLING ENABLED
    pso.RasterizerState.FrontCounterClockwise = TRUE;            // CCW is front-facing
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.SampleMask = UINT_MAX;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso.SampleDesc.Count = 1;

    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline)), "Create pipeline");

    // 3. Render Target and Readback Buffers
    D3D12_RESOURCE_DESC tex_desc{};
    tex_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex_desc.Width = tex_desc.Height = kSide;
    tex_desc.DepthOrArraySize = tex_desc.MipLevels = tex_desc.SampleDesc.Count = 1;
    tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_CLEAR_VALUE clear_val{};
    clear_val.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    ComPtr<ID3D12Resource> target;
    check(device->CreateCommittedResource(
        &default_heap, D3D12_HEAP_FLAG_NONE, &tex_desc, D3D12_RESOURCE_STATE_COMMON, &clear_val, IID_PPV_ARGS(&target)),
        "Create target texture");

    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{};
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_heap_desc.NumDescriptors = 1;
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    check(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap)), "Create RTV heap");
    const auto rtv_handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv_handle);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    device->GetCopyableFootprints(&tex_desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
    footprint.Footprint.RowPitch = (footprint.Footprint.RowPitch + 255) & ~255;
    const UINT64 readback_bytes = footprint.Offset + static_cast<UINT64>(footprint.Footprint.RowPitch) * kSide;

    D3D12_RESOURCE_DESC buf_desc{};
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Width = readback_bytes;
    buf_desc.Height = buf_desc.DepthOrArraySize = buf_desc.MipLevels = buf_desc.SampleDesc.Count = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    D3D12_HEAP_PROPERTIES readback_heap{};
    readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
    ComPtr<ID3D12Resource> readback;
    check(device->CreateCommittedResource(
        &readback_heap, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)),
        "Create readback buffer");

    D3D12_HEAP_PROPERTIES upload_heap{};
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    buf_desc.Width = sizeof(Vertex) * 12;
    ComPtr<ID3D12Resource> upload_vb;
    check(device->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &buf_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload_vb)),
        "Create upload VB");

    // 4. Command structures
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    ComPtr<ID3D12CommandQueue> queue;
    check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "Create queue");

    ComPtr<ID3D12CommandAllocator> allocator;
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "Create allocator");

    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(), IID_PPV_ARGS(&list)), "Create list");
    check(list->Close(), "Initial close list");

    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create fence");
    UINT64 fence_val = 0;

    // The Python runner feeds the output of expand_rectangles through stdin.
    Vertex positive_triangles[6]{};
    static_assert(sizeof(Vertex) == 8 * sizeof(float));
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
        std::fread(positive_triangles, sizeof(Vertex), 6, stdin) != 6 ||
        std::fgetc(stdin) != EOF || std::ferror(stdin)) {
        std::fputs("Expected six packed output vertices from the rectlist lowerer\n", stderr);
        return 1;
    }
    const auto& v0 = positive_triangles[0];
    const auto& v1 = positive_triangles[1];
    const auto& v2 = positive_triangles[2];
    const auto& v3 = positive_triangles[5];
    std::vector<uint32_t> pixels;

    execute_draw(device.Get(), queue.Get(), allocator.Get(), list.Get(), pipeline.Get(), root.Get(),
                 target.Get(), readback.Get(), upload_vb.Get(), rtv_handle, footprint, fence.Get(), fence_val,
                 positive_triangles, 6, pixels);

    const unsigned positive_mismatches = count_rectangle_mismatches(pixels.data());
    if (positive_mismatches > 0) {
        std::fprintf(stderr, "Positive draw failed: %u mismatches against expected rectangle\n", positive_mismatches);
        return 1;
    }
    std::printf("positive_draw=pass mismatches=0\n");

    // --- NEGATIVE CONTROL 1: Reversed-Winding Mutant (CW winding) ---
    // Swapping v0 and v2 flips winding to clockwise. Under CullMode = BACK and FrontCounterClockwise = TRUE,
    // the hardware rasterizer must cull all triangles, leaving the buffer completely clear.
    const Vertex reversed_triangles[6] = {v2, v1, v0, v2, v0, v3};
    execute_draw(device.Get(), queue.Get(), allocator.Get(), list.Get(), pipeline.Get(), root.Get(),
                 target.Get(), readback.Get(), upload_vb.Get(), rtv_handle, footprint, fence.Get(), fence_val,
                 reversed_triangles, 6, pixels);

    const unsigned reversed_mismatches = count_rectangle_mismatches(pixels.data());
    // Since all triangles are culled, interior pixels are clear color (0), producing > 700 mismatches.
    if (reversed_mismatches < 500) {
        std::fprintf(stderr, "Reversed-winding mutant was NOT culled by GPU (mismatches=%u)\n", reversed_mismatches);
        return 1;
    }
    std::printf("reversed_winding_cull_control=pass culled_mismatches=%u\n", reversed_mismatches);

    // --- NEGATIVE CONTROL 2: Wrong-Corner Mutant ---
    // Synthesize v3 = v0 + v2 without subtracting v1 => v3 = (0, 0).
    // The top-left quadrant is left unrendered, producing interior mismatches.
    Vertex wrong_v3{};
    for (int i = 0; i < 4; ++i) {
        wrong_v3.pos[i] = v0.pos[i] + v2.pos[i];
        wrong_v3.col[i] = v0.col[i] + v2.col[i];
    }
    const Vertex wrong_corner_triangles[6] = {v0, v1, v2, v0, v2, wrong_v3};
    execute_draw(device.Get(), queue.Get(), allocator.Get(), list.Get(), pipeline.Get(), root.Get(),
                 target.Get(), readback.Get(), upload_vb.Get(), rtv_handle, footprint, fence.Get(), fence_val,
                 wrong_corner_triangles, 6, pixels);

    const unsigned wrong_corner_mismatches = count_rectangle_mismatches(pixels.data());
    if (wrong_corner_mismatches == 0) {
        std::fputs("Wrong-corner mutant failed to trigger mismatch against expected rectangle\n", stderr);
        return 1;
    }
    std::printf("wrong_corner_mutant_control=pass detected_mismatches=%u\n", wrong_corner_mismatches);

    // --- NEGATIVE CONTROL 3: Raw Desktop Topology 0x11 Rejection ---
    // Passing raw 0x11 to desktop D3D12 pipeline creation must fail.
    D3D12_GRAPHICS_PIPELINE_STATE_DESC invalid_pso = pso;
    invalid_pso.PrimitiveTopologyType = static_cast<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(0x11);
    ComPtr<ID3D12PipelineState> bad_pipeline;
    const HRESULT bad_hr = device->CreateGraphicsPipelineState(&invalid_pso, IID_PPV_ARGS(&bad_pipeline));
    if (SUCCEEDED(bad_hr) && bad_pipeline) {
        std::fputs("Desktop runtime unexpectedly accepted raw topology 0x11\n", stderr);
        return 1;
    }
    std::printf("desktop_raw_topology_rejection=pass hr=0x%08lx\n", bad_hr);

    std::puts("ALL RECTLIST WARP DRAW TESTS AND MUTANT CONTROLS PASSED");
    return 0;
}
