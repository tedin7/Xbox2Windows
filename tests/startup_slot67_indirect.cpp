#include "startup_slot67_indirect.h"
#include "startup_indirect_writer.h"
#include "startup_stream_setup.h"
#include <d3dcompiler.h>
#include <d3d12sdklayers.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
using Microsoft::WRL::ComPtr;

namespace {
using Record = std::array<UINT, 7>;
constexpr Record initial{0xa5a5a5a5, 0, 0, 0, 0x5a5a5a5a, 0x12345678, 0x9abcdef0};
constexpr UINT offset = sizeof(UINT);  // leading guard and a neighboring three-DWORD guard record
constexpr char producer[] = R"(
RWByteAddressBuffer args : register(u0);
cbuffer Size : register(b0) { uint groups; };
[numthreads(1, 1, 1)]
void main() { args.Store3(4, uint3(groups, 1, 1)); }
)";
StreamWord word(const void* value) { return reinterpret_cast<StreamWord>(value); }
}

void Slot67Indirect::prepare(IDXGIAdapter* adapter, void* guest_device, void* guest_list, void* backing, UINT groups, bool write_zero,
                             void* guest_queue, void* guest_output)
{
    auto* guest = static_cast<ID3D12Device*>(guest_device);
    ComPtr<ID3D12Device> device;
    check(D3D12XboxCreateDevice(adapter, nullptr, IID_PPV_ARGS(&device)), "open producer's desktop device");
    auto list = indirect_native_object<ID3D12GraphicsCommandList>(device.Get(), guest, static_cast<IUnknown*>(guest_list));
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, sizeof(initial), 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    check(static_cast<HRESULT>(invoke(guest, 133, word(backing), word(&desc), D3D12_RESOURCE_STATE_COMMON, 0,
          word(&__uuidof(ID3D12Resource)), word(placed.GetAddressOf()))), "place indirect arguments");
    native_args = indirect_native_object<ID3D12Resource>(device.Get(), guest, placed.Get());
    const auto& args = native_args;
    if (guest_queue) {
        native_queue = indirect_native_object<ID3D12CommandQueue>(device.Get(), guest, static_cast<IUnknown*>(guest_queue));
        native_output = indirect_native_object<ID3D12Resource>(device.Get(), guest, static_cast<IUnknown*>(guest_output));
        std::printf("chain queue=%p output=%p gpu=0x%llx\n", static_cast<void*>(native_queue.Get()),
                    static_cast<void*>(native_output.Get()), static_cast<unsigned long long>(native_output->GetGPUVirtualAddress()));
    }
    std::printf("writer resource=%p gpu=0x%llx interval=[4,16) writer=%s\n",
                static_cast<void*>(args.Get()), static_cast<unsigned long long>(args->GetGPUVirtualAddress()),
                groups || write_zero ? "dispatch" : "omitted");
    upload = make_buffer(device.Get(), sizeof(initial), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    readback = make_buffer(device.Get(), sizeof(initial), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const D3D12_RANGE empty{0, 0};
    void* mapped = nullptr;
    check(upload->Map(0, &empty, &mapped), "map zero dispatch record");
    std::memcpy(mapped, initial.data(), sizeof(initial));
    upload->Unmap(0, nullptr);
    transition(list.Get(), args.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(args.Get(), 0, upload.Get(), 0, sizeof(initial));
    auto state = D3D12_RESOURCE_STATE_COPY_DEST;
    if (groups || write_zero) {
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants = {0, 0, 1};
        const D3D12_ROOT_SIGNATURE_DESC root_desc{2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob, code;
        check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "serialize producer root");
        check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "create producer root");
        check(D3DCompile(producer, sizeof(producer) - 1, nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0,
                         &code, nullptr), "compile indirect producer");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get();
        pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        check(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&writer)), "create indirect producer");
        transition(list.Get(), args.Get(), state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        list->SetComputeRootSignature(root.Get());
        list->SetComputeRoot32BitConstant(1, groups, 0);
        record_indirect_writer(list.Get(), writer.Get(), args.Get());
    }
    transition(list.Get(), args.Get(), state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, args.Get(), 0, sizeof(initial));
    transition(list.Get(), args.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    D3D12_INDIRECT_ARGUMENT_DESC command{D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH};
    const D3D12_COMMAND_SIGNATURE_DESC sig{sizeof(D3D12_DISPATCH_ARGUMENTS), 1, &command, 0};
    check(guest->CreateCommandSignature(&sig, nullptr, IID_PPV_ARGS(&signature)), "create guest dispatch signature");
}

void Slot67Indirect::execute(void* list, UINT commands, bool count_buffer, bool outside) const
{
    invoke(list, 61, word(signature.Get()), commands, word(placed.Get()), outside ? sizeof(initial) - sizeof(UINT) : offset,
           count_buffer ? word(placed.Get()) : 0, count_buffer ? offset : 0);
}

void Slot67Indirect::capture_before_consumer(IDXGIAdapter* adapter, void* guest_device, void* guest_list)
{
    auto* guest = static_cast<ID3D12Device*>(guest_device);
    ComPtr<ID3D12Device> device;
    check(D3D12XboxCreateDevice(adapter, nullptr, IID_PPV_ARGS(&device)), "open consumer's desktop device");
    auto list = indirect_native_object<ID3D12GraphicsCommandList>(device.Get(), guest, static_cast<IUnknown*>(guest_list));
    auto args = indirect_native_object<ID3D12Resource>(device.Get(), guest, placed.Get());
    if (args.Get() != native_args.Get() || args->GetGPUVirtualAddress() != native_args->GetGPUVirtualAddress())
        throw std::runtime_error("indirect argument resource changed");
    std::printf("consumer resource=%p gpu=0x%llx interval=[4,16)\n",
                static_cast<void*>(args.Get()), static_cast<unsigned long long>(args->GetGPUVirtualAddress()));
    before_consumer = make_buffer(device.Get(), sizeof(initial), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    transition(list.Get(), args.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(before_consumer.Get(), 0, args.Get(), 0, sizeof(initial));
    transition(list.Get(), args.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

bool Slot67Indirect::check_record(UINT groups, bool write_zero, bool consumer_boundary) const
{
    const D3D12_RANGE range{0, sizeof(initial)}, empty{0, 0};
    void* mapped = nullptr;
    auto* snapshot = consumer_boundary ? before_consumer.Get() : readback.Get();
    check(snapshot->Map(0, &range, &mapped), "map GPU argument readback after queue fence");
    Record actual{};
    std::memcpy(actual.data(), mapped, sizeof(actual));
    snapshot->Unmap(0, &empty);
    const UINT yz = groups || write_zero ? 1u : 0u;
    Record expected = initial;
    expected[1] = groups;
    expected[2] = expected[3] = yz;
    std::printf("GPU argument record: %08x [%u,%u,%u] %08x %08x %08x (%s)\n",
                actual[0], actual[1], actual[2], actual[3], actual[4], actual[5], actual[6],
                consumer_boundary ? "before consumer" : "after writer");
    std::fflush(stdout);
    return actual == expected;
}

// A source-built desktop graphics observer. The slot-67 shader's actual aliased UAV is the SRV;
// no CPU copy or CPU-derived colour is used as a draw input. Both captures are fenced on the same
// underlying DIRECT queue as the writer and translated ExecuteIndirect.
void Slot67Indirect::draw_pixels(IDXGIAdapter* adapter, void* guest_device, void* guest_queue, void* guest_output, bool stale) const
{
    auto* guest = static_cast<ID3D12Device*>(guest_device);
    ComPtr<ID3D12Device> device;
    check(D3D12XboxCreateDevice(adapter, nullptr, IID_PPV_ARGS(&device)), "open graphics observer's desktop device");
    auto queue = indirect_native_object<ID3D12CommandQueue>(device.Get(), guest, static_cast<IUnknown*>(guest_queue));
    auto output = indirect_native_object<ID3D12Resource>(device.Get(), guest, static_cast<IUnknown*>(guest_output));
    if (queue.Get() != native_queue.Get() || (output.Get() == native_output.Get()) == stale)
        throw std::runtime_error("graphics queue/output identity mismatch");
    std::printf("graphics queue=%p output=%p gpu=0x%llx\n", static_cast<void*>(queue.Get()),
                static_cast<void*>(output.Get()), static_cast<unsigned long long>(output->GetGPUVirtualAddress()));
    constexpr char shader[] = R"(
ByteAddressBuffer result : register(t0);
float4 vs(uint id : SV_VertexID) : SV_Position {
    return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0, 1);
}
float4 ps(float4 position : SV_Position) : SV_Target {
    uint word = result.Load(uint(position.x) * 4);
    if (word == 0xc5c5c5c5) discard;
    return float4(word & 255, (word >> 8) & 255, (word >> 16) & 255, 255) / 255.0;
}
)";
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    const D3D12_ROOT_SIGNATURE_DESC root_desc{1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob, vs, ps;
    ComPtr<ID3D12RootSignature> graphics_root;
    check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "serialize pixel root");
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&graphics_root)), "create pixel root");
    check(D3DCompile(shader, sizeof(shader)-1, nullptr, nullptr, nullptr, "vs", "vs_5_0", 0, 0, &vs, nullptr), "compile observer VS");
    check(D3DCompile(shader, sizeof(shader)-1, nullptr, nullptr, nullptr, "ps", "ps_5_0", 0, 0, &ps, nullptr), "compile observer PS");
    constexpr UINT width = 8, height = 4;
    constexpr DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = graphics_root.Get();
    pso.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    pso.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.SampleMask = UINT_MAX;
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = format;
    pso.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline)), "create observer graphics pipeline");
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        format, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE clear{format, {0, 0, 0, 1}};
    auto target = resource(device.Get(), desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    ComPtr<ID3D12Resource> captures[2];
    for (auto& capture : captures)
        capture = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    ComPtr<ID3D12DescriptorHeap> rtvs;
    check(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtvs)), "create observer RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);
    ComputeCommands commands(device.Get(), queue.Get(), pipeline.Get());
    auto* list = commands.list.Get();
    const auto capture = [&](ID3D12Resource* readback) {
        transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION to{readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint = footprint;
        const D3D12_TEXTURE_COPY_LOCATION from{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        transition(list, target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    };
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    capture(captures[0].Get());
    transition(list, output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list->SetGraphicsRootSignature(graphics_root.Get());
    list->SetGraphicsRootShaderResourceView(0, output->GetGPUVirtualAddress());
    const D3D12_VIEWPORT viewport{0, 0, width, height, 0, 1};
    const D3D12_RECT scissor{0, 0, width, height};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    capture(captures[1].Get());
    transition(list, output.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    commands.submit_and_wait(device.Get());
    std::puts("graphics fence completed: clear and draw captures");
    ComPtr<ID3D12InfoQueue> info;
    if (SUCCEEDED(device.As(&info))) {
        for (UINT64 i = 0, n = info->GetNumStoredMessagesAllowedByRetrievalFilter(); i < n; ++i) {
            SIZE_T size = 0;
            check(info->GetMessage(i, nullptr, &size), "size graphics debug message");
            std::vector<unsigned char> storage(size);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(info->GetMessage(i, message, &size), "read graphics debug message");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::fprintf(stderr, "graphics debug error: %s\n", message->pDescription);
                throw std::runtime_error("graphics debug layer error");
            }
        }
        std::puts("graphics debug layer: no errors");
    } else {
        std::puts("graphics debug layer: unavailable");
    }
    char path[MAX_PATH];
    const DWORD n = GetEnvironmentVariableA("X2W_TEST_INDIRECT_PIXEL_OUT", path, MAX_PATH);
    if (!n || n >= MAX_PATH) throw std::runtime_error("missing pixel output path");
    std::ofstream file(path, std::ios::binary);
    const D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)}, empty{0, 0};
    for (const auto& readback : captures) {
        void* mapped = nullptr;
        check(readback->Map(0, &range, &mapped), "map fenced pixel capture");
        for (UINT y = 0; y < height; ++y)
            file.write(static_cast<const char*>(mapped) + footprint.Offset + y * footprint.Footprint.RowPitch, width * sizeof(UINT));
        readback->Unmap(0, &empty);
    }
    file.close();
    if (!file) throw std::runtime_error("write pixel capture");
}
