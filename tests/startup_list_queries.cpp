// Occlusion queries recorded on a console command list (slots 28, 29 and 30).
//
//   startup_list_queries <warp|hardware> [default]
//
// The probe sets X2W_D3D12_QUERY_SLOTS=1, the opt-in that binds the three slots, unless "default" is given.
// Through D3D12XboxCreateDevice it creates an occlusion query heap of 8 (device slot 39) and a readback buffer
// in COPY_DEST (device slot 27), fills the buffer with a sentinel, and records on a DIRECT console list:
//
//   query 0: BeginQuery / EndQuery (slots 28, 29) around a draw that covers an 8x8 target   -> 64 samples
//   query 1: BeginQuery / EndQuery around nothing                                           -> 0 samples
//   query 2: BeginQuery / EndQuery with the pipeline-statistics type on the occlusion heap   -> refused
//   query 3: EndQuery without a BeginQuery                                                  -> refused
//   query 4: BeginQuery / EndQuery with the binary-occlusion type                           -> forwarded
//   queries 5 and 7: BeginQuery / EndQuery; query 6 never begun                             -> forwarded
//   query 2: BeginQuery (occlusion) without an EndQuery                                     -> ended at Close
//   ResolveQueryData (slot 30) of queries 0 and 1 (occlusion) to word 0                     -> forwarded
//   Resolves that must leave their words untouched (refused): query 3 (never begun), queries 5 to 7 (a
//   hole at 6), query 4 as occlusion (ended with the other type), queries 1 and 2 (2 still open), a range
//   past the heap's count, and one past the buffer's end.
//
// Which console methods sit at slots 28 to 30 is a hypothesis: the probe calls those slots directly, so it
// checks the forwarding, the refusals and the end at Close, not the numbering.
//
// The list is submitted and signalled through the console queue. The buffer, read through the console
// resource's Map, must hold 64 and 0 in its first two words and the sentinel everywhere else, and the D3D12
// debug layer must store no errors. A refused call that reached the desktop list would add one or make the
// desktop Close fail (an EndQuery without its BeginQuery does: exit code 3).
//
// With "default" the opt-in is not set and the three slots stay no-ops, the behaviour before this change:
// the buffer keeps its sentinel and the probe fails with exit code 1.
//
// Exit codes: 0 pass, 1 wrong query results, 2 usage, 3 setup failure, 4 debug-layer error, 77 no physical adapter.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "desktop_gpu.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Microsoft::WRL::ComPtr;
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f);
}

constexpr UINT side = 8;
constexpr std::uint64_t sentinel = 0xa5a5a5a5a5a5a5a5ull;
constexpr UINT heap_count = 8, buffer_words = 12;

constexpr char shader[] = R"(
float4 vs(uint id : SV_VertexID) : SV_Position
{
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
}
float4 ps() : SV_Target { return float4(1, 1, 1, 1); }
)";

ComPtr<ID3DBlob> compile(const char* entry, const char* target)
{
    ComPtr<ID3DBlob> code, errors;
    const HRESULT hr = D3DCompile(shader, sizeof(shader) - 1, nullptr, nullptr, nullptr, entry, target, 0, 0, &code, &errors);
    if (FAILED(hr) && errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
    check(hr, entry);
    return code;
}

template <typename T>
ComPtr<T> native_object(ID3D12Device* device, ID3D12Device* guest, IUnknown* object)
{
    // SetPrivateDataInterface unwraps a guest argument before the call reaches the shared desktop device.
    constexpr GUID key{0x7c1d52a0, 0x3b4e, 0x4f61, {0x9a, 0x0d, 0x51, 0x2e, 0x86, 0x14, 0xc7, 0x3b}};
    check(guest->SetPrivateDataInterface(key, object), "publish underlying object");
    ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT read = device->GetPrivateData(key, &bytes, stored.GetAddressOf());
    check(guest->SetPrivateDataInterface(key, nullptr), "clear underlying object");
    check(read, "retrieve underlying object");
    ComPtr<T> result;
    check(stored.As(&result), "query underlying object");
    return result;
}

UINT64 debug_errors(ID3D12Device* device)
{
    ComPtr<ID3D12InfoQueue> info;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&info)))) return 0;
    UINT64 errors = 0;
    for (UINT64 i = 0, n = info->GetNumStoredMessages(); i < n; ++i) {
        SIZE_T bytes = 0;
        if (FAILED(info->GetMessage(i, nullptr, &bytes))) continue;
        std::vector<char> storage(bytes);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(info->GetMessage(i, message, &bytes))) continue;
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
            ++errors;
            std::fprintf(stderr, "D3D12 debug message %u: %s\n", static_cast<unsigned>(message->ID), message->pDescription);
        }
    }
    return errors;
}
}  // namespace

int main(int argc, char** argv) try
{
    const bool default_slots = argc == 3 && std::strcmp(argv[2], "default") == 0;
    if (argc != 2 && !default_slots) return 2;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    if (!warp && std::strcmp(argv[1], "hardware") != 0) return 2;
    ComPtr<ID3D12Debug> debug;
    const HRESULT debug_hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
    if (debug_hr != DXGI_ERROR_SDK_COMPONENT_MISSING) check(debug_hr, "get debug layer");
    if (debug) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "find WARP");
    } else {
        for (UINT index = 0; factory->EnumAdapters1(index, adapter.ReleaseAndGetAddressOf()) == S_OK; ++index) {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) break;
        }
        if (!adapter) return 77;
    }
    ComPtr<ID3D12Device> device;
    check(D3D12XboxCreateDevice(adapter.Get(), nullptr, IID_PPV_ARGS(&device)), "create desktop device through d3d12_xs");
    SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", "1");
    SetEnvironmentVariableA("X2W_D3D12_RENDER", "1");
    SetEnvironmentVariableA("X2W_D3D12_QUERY_SLOTS", default_slots ? nullptr : "1");
    const std::array<unsigned char, 64> parameters{};
    ComPtr<ID3D12Device> guest;
    check(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), IID_PPV_ARGS(&guest)), "create guest device");
    DXGI_ADAPTER_DESC1 adapter_desc{};
    check(adapter->GetDesc1(&adapter_desc), "get adapter description");
    std::printf("adapter: %ls, debug layer %s\n", adapter_desc.Description, debug ? "enabled" : "unavailable");
    LARGE_INTEGER driver{};
    if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver)))
        std::printf("driver: %u.%u.%u.%u\n", HIWORD(driver.HighPart), LOWORD(driver.HighPart), HIWORD(driver.LowPart),
                    LOWORD(driver.LowPart));

    // Desktop draw state: an empty root signature, a full-target triangle, and an 8x8 target with its view.
    const D3D12_ROOT_SIGNATURE_DESC root_desc{0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> serialized;
    check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr), "serialize root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&root)),
          "create root");
    const auto vs = compile("vs", "vs_5_0");
    const auto ps = compile("ps", "ps_5_0");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.Get();
    pso_desc.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    pso_desc.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso_desc.SampleMask = ~0u;
    pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso_desc.RasterizerState.DepthClipEnable = TRUE;
    pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso_desc.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pso;
    check(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso)), "create pipeline");
    const D3D12_RESOURCE_DESC target_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, side, side, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                          {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const auto target = resource(device.Get(), target_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    check(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap)), "create view heap");
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    // Console objects: the query heap, the readback buffer, a queue, a fence, an allocator and a list.
    const D3D12_QUERY_HEAP_DESC heap_desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, heap_count, 0};
    ComPtr<ID3D12QueryHeap> heap;
    check(guest->CreateQueryHeap(&heap_desc, IID_PPV_ARGS(&heap)), "create guest query heap");
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_RESOURCE_DESC buffer_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, buffer_words * sizeof(std::uint64_t), 1, 1, 1,
                                          DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
    ComPtr<ID3D12Resource> results;
    check(guest->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                         nullptr, IID_PPV_ARGS(&results)), "create guest readback buffer");
    void* mapped = nullptr;
    check(results->Map(0, nullptr, &mapped), "map guest readback buffer");
    for (UINT i = 0; i < buffer_words; ++i) static_cast<std::uint64_t*>(mapped)[i] = sentinel;
    results->Unmap(0, nullptr);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    void* queue = nullptr;
    void* fence = nullptr;
    void* allocator = nullptr;
    void* list = nullptr;
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    check(static_cast<HRESULT>(invoke(guest.Get(), 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&queue))),
          "create guest queue");
    check(static_cast<HRESULT>(invoke(guest.Get(), 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&fence))), "create guest fence");
    check(static_cast<HRESULT>(invoke(guest.Get(), 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)),
                                      arg(&allocator))), "create guest allocator");
    check(static_cast<HRESULT>(invoke(guest.Get(), 73, arg(list_desc), arg(allocator), 0,
                                      arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))), "create guest list");
    check(static_cast<HRESULT>(invoke(list, 139, arg(allocator))), "reset guest list");
    const auto native_list = native_object<ID3D12GraphicsCommandList>(device.Get(), guest.Get(),
                                                                      static_cast<IUnknown*>(list));

    // The draw is recorded on the desktop list behind the console one; the queries go through the console slots.
    auto* desktop = native_list.Get();
    transition(desktop, target.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
    desktop->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
    const D3D12_RECT scissor{0, 0, side, side};
    desktop->RSSetViewports(1, &viewport);
    desktop->RSSetScissorRects(1, &scissor);
    desktop->SetGraphicsRootSignature(root.Get());
    desktop->SetPipelineState(pso.Get());
    desktop->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const Word occlusion = D3D12_QUERY_TYPE_OCCLUSION, statistics = D3D12_QUERY_TYPE_PIPELINE_STATISTICS;
    invoke(list, 28, arg(heap.Get()), occlusion, 0);
    desktop->DrawInstanced(3, 1, 0, 0);
    invoke(list, 29, arg(heap.Get()), occlusion, 0);
    invoke(list, 28, arg(heap.Get()), occlusion, 1);
    invoke(list, 29, arg(heap.Get()), occlusion, 1);
    invoke(list, 28, arg(heap.Get()), statistics, 2);  // refused: the heap holds occlusion queries
    invoke(list, 29, arg(heap.Get()), statistics, 2);  // refused
    invoke(list, 29, arg(heap.Get()), occlusion, 3);   // refused: query 3 was never begun
    const Word binary = D3D12_QUERY_TYPE_BINARY_OCCLUSION;
    invoke(list, 28, arg(heap.Get()), binary, 4);
    invoke(list, 29, arg(heap.Get()), binary, 4);
    invoke(list, 28, arg(heap.Get()), occlusion, 5);
    invoke(list, 29, arg(heap.Get()), occlusion, 5);
    invoke(list, 28, arg(heap.Get()), occlusion, 7);
    invoke(list, 29, arg(heap.Get()), occlusion, 7);
    invoke(list, 28, arg(heap.Get()), occlusion, 2);   // never ended: the stand-in ends it at Close
    transition(desktop, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
    invoke(list, 30, arg(heap.Get()), occlusion, 0, 2, arg(results.Get()), 0);
    // Each refused resolve below targets its own words, which must keep the sentinel.
    invoke(list, 30, arg(heap.Get()), occlusion, 3, 1, arg(results.Get()), 16);  // word 2: query 3 never begun
    invoke(list, 30, arg(heap.Get()), occlusion, 5, 3, arg(results.Get()), 24);  // words 3-5: hole at query 6
    invoke(list, 30, arg(heap.Get()), occlusion, 4, 1, arg(results.Get()), 48);  // word 6: query 4 is binary
    invoke(list, 30, arg(heap.Get()), occlusion, 1, 2, arg(results.Get()), 56);  // words 7-8: query 2 still open
    invoke(list, 30, arg(heap.Get()), occlusion, 7, 2, arg(results.Get()), 72);  // words 9-10: past the heap's count
    invoke(list, 30, arg(heap.Get()), occlusion, 0, 2, arg(results.Get()), 88);  // past the buffer's end
    check(static_cast<HRESULT>(invoke(list, 9)), "close guest list");

    void* lists[] = {list};
    invoke(queue, 10, 1, arg(lists));
    check(static_cast<HRESULT>(invoke(queue, 14, arg(fence), 1)), "signal guest fence");
    const auto deadline = GetTickCount64() + 10000;
    while (invoke(fence, 8) < 1) {
        if (GetTickCount64() >= deadline) {
            std::fprintf(stderr, "the guest fence did not reach 1\n");
            return 3;
        }
        Sleep(1);
    }
    check(device->GetDeviceRemovedReason(), "GPU completion");

    std::uint64_t observed[buffer_words]{};
    const D3D12_RANGE read{0, sizeof(observed)};
    check(results->Map(0, &read, &mapped), "map guest readback buffer");
    std::memcpy(observed, mapped, sizeof(observed));
    const D3D12_RANGE untouched{0, 0};
    results->Unmap(0, &untouched);
    std::uint64_t expected[buffer_words];
    for (auto& word : expected) word = sentinel;
    expected[0] = side * side;
    expected[1] = 0;
    std::printf("samples: query 0 = %llu (expected %llu), query 1 = %llu (expected 0)\n",
                static_cast<unsigned long long>(observed[0]), static_cast<unsigned long long>(expected[0]),
                static_cast<unsigned long long>(observed[1]));
    for (UINT i = 0; i < buffer_words; ++i) {
        if (observed[i] == expected[i]) continue;
        std::fprintf(stderr, "word %u: got 0x%016llx, expected 0x%016llx\n", i, static_cast<unsigned long long>(observed[i]),
                     static_cast<unsigned long long>(expected[i]));
        return 1;
    }
    if (debug_errors(device.Get())) return 4;
    return 0;
}
catch (const std::exception&) {
    return 3;
}
