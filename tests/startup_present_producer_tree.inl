// Included by startup_present_producer.cpp. These desktop shaders are built from
// source by the fixture and use the same tree recorder and sampler as the shim.
namespace producer_tree_fixture {

using Microsoft::WRL::ComPtr;

constexpr UINT width = 8, height = 8;
constexpr auto colour_format = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr auto float_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
constexpr auto depth_format = DXGI_FORMAT_D32_FLOAT;

constexpr char shader[] = R"(
Texture2D<float4> x : register(t0);
Texture2D<float4> y : register(t1);
RWTexture2D<unorm float4> z : register(u0);
float4 vertex(uint id : SV_VertexID) : SV_Position {
    return float4(id == 2 ? 3 : -1, id == 1 ? -3 : 1, 0.25, 1);
}
float4 first(float4 position : SV_Position) : SV_Target {
    return float4(float(uint(position.x) + 1) / 256, 0.25, 257.0 / 512, 1);
}
float4 middle(uint2 pixel) {
    return x.Load(int3(pixel, 0)) + y.Load(int3(pixel, 0)) + float4(0, 1.0 / 255, 0, 0);
}
float4 second(float4 position : SV_Position) : SV_Target {
    return middle(uint2(position.xy));
}
float4 third(float4 position : SV_Position) : SV_Target {
    return x.Load(int3(uint2(position.xy), 0));
}
[numthreads(8, 8, 1)] void compute(uint3 id : SV_DispatchThreadID) {
    z[id.xy] = middle(id.xy);
}
)";

void trace(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
}

ComPtr<ID3DBlob> compile(const char* entry, const char* profile)
{
    ComPtr<ID3DBlob> code, errors;
    if (FAILED(D3DCompile(shader, sizeof(shader) - 1, nullptr, nullptr, nullptr, entry, profile,
                          D3DCOMPILE_ENABLE_STRICTNESS, 0, &code, &errors))) {
        if (errors) std::fputs(static_cast<const char*>(errors->GetBufferPointer()), stderr);
        return {};
    }
    return code;
}

D3D12_SHADER_BYTECODE bytecode(ID3DBlob* blob)
{
    return {blob->GetBufferPointer(), blob->GetBufferSize()};
}

ComPtr<ID3D12PipelineState> graphics(ID3D12Device* device, ID3D12RootSignature* root, ID3DBlob* vertex,
                                    const char* entry, bool depth_test)
{
    const auto pixel = compile(entry, "ps_5_0");
    if (!pixel) return {};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.VS = bytecode(vertex);
    desc.PS = bytecode(pixel.Get());
    auto& blend = desc.BlendState.RenderTarget[0];
    blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
    blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = UINT_MAX;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.DepthStencilState.DepthEnable = depth_test;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    const D3D12_DEPTH_STENCILOP_DESC stencil{D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                             D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS};
    desc.DepthStencilState.FrontFace = desc.DepthStencilState.BackFace = stencil;
    desc.DSVFormat = depth_test ? depth_format : DXGI_FORMAT_UNKNOWN;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = depth_test ? float_format : colour_format;
    desc.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> state;
    if (FAILED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&state)))) return {};
    return state;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &barrier);
}

template<bool IncludeCapture>
int run(IDXGIAdapter* adapter, const char* mode)
{
    static constexpr UINT x = 0, y = 1, z = 2, back = 3, depth = 4;
    const bool mutate = std::strcmp(mode, "tree-mutant") == 0;
    struct NoCapture {};
    std::conditional_t<IncludeCapture, std::unique_ptr<x2w_red::Tree>, NoCapture> tree;
    if constexpr (IncludeCapture) {
        char frames[16]{};
        if (GetEnvironmentVariableA("X2W_RED_PRODUCER_FRAMES", frames, sizeof(frames)))
            tree = std::make_unique<x2w_red::Tree>(trace);
    }
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))) ||
        FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return 20;

    const auto format_of = [](UINT texture) {
        return texture == depth ? depth_format : texture == x ? float_format : colour_format;
    };
    const D3D12_HEAP_PROPERTIES gpu_heap{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> textures[5];
    for (UINT i = 0; i < std::size(textures); ++i) {
        D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
            format_of(i), {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
            i == depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
        if (i == z) desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(device->CreateCommittedResource(&gpu_heap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&textures[i])))) return 21;
    }

    const D3D12_DESCRIPTOR_HEAP_DESC rt_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4};
    const D3D12_DESCRIPTOR_HEAP_DESC ds_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
    const D3D12_DESCRIPTOR_HEAP_DESC shader_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3,
                                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> rt_heap, ds_heap, descriptors;
    if (FAILED(device->CreateDescriptorHeap(&rt_desc, IID_PPV_ARGS(&rt_heap))) ||
        FAILED(device->CreateDescriptorHeap(&ds_desc, IID_PPV_ARGS(&ds_heap))) ||
        FAILED(device->CreateDescriptorHeap(&shader_desc, IID_PPV_ARGS(&descriptors)))) return 22;
    const UINT rt_step = device->GetDescriptorHandleIncrementSize(rt_desc.Type);
    const UINT view_step = device->GetDescriptorHandleIncrementSize(shader_desc.Type);
    D3D12_CPU_DESCRIPTOR_HANDLE rt[4];
    const auto rtv = rt_heap->GetCPUDescriptorHandleForHeapStart();
    const auto dsv = ds_heap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < std::size(rt); ++i) {
        rt[i] = {rtv.ptr + SIZE_T{i} * rt_step};
        device->CreateRenderTargetView(textures[i].Get(), nullptr, rt[i]);
    }
    device->CreateDepthStencilView(textures[depth].Get(), nullptr, dsv);
    const auto cpu = descriptors->GetCPUDescriptorHandleForHeapStart();
    const auto gpu = descriptors->GetGPUDescriptorHandleForHeapStart();
    device->CreateShaderResourceView(textures[x].Get(), nullptr, cpu);
    device->CreateShaderResourceView(textures[y].Get(), nullptr, {cpu.ptr + view_step});
    device->CreateUnorderedAccessView(textures[z].Get(), nullptr, nullptr, {cpu.ptr + 2 * view_step});

    const D3D12_DESCRIPTOR_RANGE ranges[]{
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0},
    };
    D3D12_ROOT_PARAMETER parameters[2]{};
    for (UINT i = 0; i < std::size(parameters); ++i) {
        parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[i].DescriptorTable = {1, ranges + i};
    }
    const D3D12_ROOT_SIGNATURE_DESC root_desc{2, parameters, 0, nullptr,
                                               D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    // The final draw's table has two SRVs even though the shader reads only slot 0.
    const D3D12_DESCRIPTOR_HEAP_DESC last_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                                D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> last;
    if (FAILED(device->CreateDescriptorHeap(&last_desc, IID_PPV_ARGS(&last)))) return 23;
    const auto last_cpu = last->GetCPUDescriptorHandleForHeapStart();
    device->CreateShaderResourceView(textures[z].Get(), nullptr, last_cpu);
    device->CreateShaderResourceView(textures[y].Get(), nullptr, {last_cpu.ptr + view_step});
    ComPtr<ID3DBlob> serialized;
    ComPtr<ID3D12RootSignature> root;
    if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr)) ||
        FAILED(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                           IID_PPV_ARGS(&root)))) return 23;
    const auto vertex = compile("vertex", "vs_5_0");
    const auto compute = compile("compute", "cs_5_0");
    if (!vertex || !compute) return 24;
    const auto p1 = graphics(device.Get(), root.Get(), vertex.Get(), "first", true);
    const auto p2 = graphics(device.Get(), root.Get(), vertex.Get(), "second", false);
    const auto p3 = graphics(device.Get(), root.Get(), vertex.Get(), "third", false);
    ComPtr<ID3D12PipelineState> cs;
    D3D12_COMPUTE_PIPELINE_STATE_DESC cs_desc{};
    cs_desc.pRootSignature = root.Get();
    cs_desc.CS = bytecode(compute.Get());
    if (!p1 || !p2 || !p3 || FAILED(device->CreateComputePipelineState(&cs_desc, IID_PPV_ARGS(&cs)))) return 24;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    auto desc = textures[back]->GetDesc();
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    const D3D12_HEAP_PROPERTIES cpu_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                                {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    ComPtr<ID3D12Resource> readback;
    if (FAILED(device->CreateCommittedResource(&cpu_heap, D3D12_HEAP_FLAG_NONE, &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)))) return 25;

    const auto move = [&](UINT texture, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
        transition(list.Get(), textures[texture].Get(), from, to);
    };
    const auto record = [&](LONG frame, UINT target, const char* op, std::uint64_t key,
                            const x2w_red::Input* inputs = nullptr, UINT count = 0) {
        if constexpr (IncludeCapture) {
            if (tree) tree->record(frame, list.Get(), {textures[target].Get(), 0, format_of(target)},
                                    op, key, inputs, count);
        }
    };
    const auto execute = [&](LONG frame, bool capture) {
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[]{list.Get()};
        queue->ExecuteCommandLists(1, lists);
        if constexpr (IncludeCapture) {
            if (tree) {
                tree->submit(frame, queue.Get(), list.Get(), capture);
                if (capture) tree->present(device.Get(), queue.Get(), textures[back].Get(), frame);
            }
        }
        if (FAILED(queue->Signal(fence.Get(), frame))) return false;
        const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) return false;
        const bool complete = SUCCEEDED(fence->SetEventOnCompletion(frame, event)) &&
            WaitForSingleObject(event, 15000) == WAIT_OBJECT_0;
        CloseHandle(event);
        if constexpr (IncludeCapture) {
            if (tree && complete) {
                tree->flush();
                tree->discard(list.Get());
            }
        }
        return complete;
    };
    const auto reset = [&] {
        return SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.Get(), nullptr));
    };
    const float black[4]{};
    move(y, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
    list->ClearRenderTargetView(rt[y], black, 0, nullptr);
    record(1, y, "clear-rtv", 0);
    move(y, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
    if (!execute(1, false)) return 26;

    for (LONG frame = 2; frame <= 4; ++frame) {
        if (!reset()) return 27;
        const bool dispatch = frame >= 3;
        const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
        const D3D12_RECT scissor{0, 0, width, height};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->SetGraphicsRootSignature(root.Get());
        move(depth, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.375f, 0, 0, nullptr);
        record(frame, depth, "clear-dsv", 0);
        // The DSV itself is writable, so bind it in DEPTH_WRITE even with a zero PSO write mask.
        move(x, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->SetPipelineState(p1.Get());
        list->OMSetRenderTargets(1, rt + x, FALSE, &dsv);
        list->DrawInstanced(3, 1, 0, 0);
        const x2w_red::Input depth_input{textures[depth].Get(), 32, depth_format};
        record(frame, x, "translated-draw", 0x11, &depth_input, 1);
        move(depth, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COMMON);

        const auto shader_state = dispatch ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        const auto write_state = dispatch ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_RENDER_TARGET;
        move(x, D3D12_RESOURCE_STATE_RENDER_TARGET, shader_state);
        move(y, D3D12_RESOURCE_STATE_COMMON, shader_state);
        move(z, D3D12_RESOURCE_STATE_COMMON, write_state);
        ID3D12DescriptorHeap* bound[]{descriptors.Get()};
        list->SetDescriptorHeaps(1, bound);
        const x2w_red::Input inputs[]{{textures[x].Get(), 0, float_format}, {textures[y].Get(), 1, colour_format}};
        if (dispatch) {
            list->SetPipelineState(cs.Get());
            list->SetComputeRootSignature(root.Get());
            list->SetComputeRootDescriptorTable(0, gpu);
            list->SetComputeRootDescriptorTable(1, {gpu.ptr + 2 * view_step});
            list->Dispatch(1, 1, 1);
            if (!mutate) record(frame, z, "translated-dispatch", 0x22, inputs, 2);
        } else {
            list->SetPipelineState(p2.Get());
            list->SetGraphicsRootDescriptorTable(0, gpu);
            list->OMSetRenderTargets(1, rt + z, FALSE, nullptr);
            list->DrawInstanced(3, 1, 0, 0);
            record(frame, z, "translated-draw", 0x22, inputs, 2);
        }
        move(x, shader_state, D3D12_RESOURCE_STATE_COMMON);
        move(y, shader_state, D3D12_RESOURCE_STATE_COMMON);
        move(z, write_state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

        bound[0] = last.Get();
        list->SetDescriptorHeaps(1, bound);
        list->SetPipelineState(p3.Get());
        list->SetGraphicsRootDescriptorTable(0, last->GetGPUDescriptorHandleForHeapStart());
        move(back, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->OMSetRenderTargets(1, rt + back, FALSE, nullptr);
        list->DrawInstanced(3, 1, 0, 0);
        const x2w_red::Input z_input{textures[z].Get(), 0, colour_format};
        record(frame, back, "translated-draw", 0x33, &z_input, 1);
        move(z, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        move(back, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        x2w_present_signature::copy_sample(list.Get(), textures[back].Get(), 0, readback.Get(), footprint, nullptr);
        move(back, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        if (!execute(frame, true)) return 28;

        void* data = nullptr;
        const D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)}, none{};
        if (FAILED(readback->Map(0, &range, &data))) return 29;
        std::printf("frame=%ld rgba=", frame);
        bool matches = true;
        for (UINT row = 0; row < height; ++row) {
            const auto* pixels = static_cast<const unsigned char*>(data) + row * footprint.Footprint.RowPitch;
            for (UINT col = 0; col < width; ++col) {
                const unsigned char expected[]{static_cast<unsigned char>(col + 1), 65, 128, 255};
                const auto* pixel = pixels + col * 4;
                if (matches && std::memcmp(expected, pixel, sizeof(expected)) != 0) {
                    std::fprintf(stderr, "frame %ld pixel (%u,%u) got %u,%u,%u,%u expected %u,65,128,255\n",
                        frame, col, row, pixel[0], pixel[1], pixel[2], pixel[3], col + 1);
                    matches = false;
                }
                for (unsigned c = 0; c < 4; ++c) std::printf("%02x", unsigned(pixel[c]));
            }
        }
        std::putchar('\n');
        readback->Unmap(0, &none);
        if (!matches) return 30;
    }
    return FAILED(device->GetDeviceRemovedReason()) || producer_errors(device.Get()) ? 31 : 0;
}

}  // namespace producer_tree_fixture
