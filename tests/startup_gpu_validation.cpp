// startup_gpu_validation <warp|hardware> <hazard|clean>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
struct DebugMessages {
    std::atomic<unsigned> errors{0};
    std::atomic<unsigned> state_hazards{0};
};

void WINAPI on_message(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id,
    LPCSTR text, void* context)
{
    if (severity > D3D12_MESSAGE_SEVERITY_ERROR) return;
    auto& messages = *static_cast<DebugMessages*>(context);
    if (++messages.errors <= 12) std::printf("debug layer id=%u: %.500s\n", static_cast<unsigned>(id), text);
    if (id == D3D12_MESSAGE_ID_GPU_BASED_VALIDATION_INCOMPATIBLE_RESOURCE_STATE) ++messages.state_hazards;
}

bool ok(HRESULT hr, const char* step)
{
    if (SUCCEEDED(hr)) return true;
    std::printf("FAIL %s (0x%08lx)\n", step, static_cast<unsigned long>(hr));
    return false;
}

bool create_pipeline(ID3D12Device* device, ComPtr<ID3D12RootSignature>& root, ComPtr<ID3D12PipelineState>& pso)
{
    constexpr char shader[] = R"(
        Texture2D<float> source : register(t0);
        RWByteAddressBuffer output : register(u0);
        [numthreads(1, 1, 1)]
        void main() {
            output.Store(0, asuint(source.Load(int3(0, 0, 0))));
        }
    )";
    ComPtr<ID3DBlob> code;
    ComPtr<ID3DBlob> errors;
    if (!ok(D3DCompile(shader, sizeof(shader) - 1, nullptr, nullptr, nullptr, "main", "cs_5_1",
            0, 0, &code, &errors), "compile compute shader")) {
        if (errors) std::printf("%s\n", static_cast<const char*>(errors->GetBufferPointer()));
        return false;
    }

    const D3D12_DESCRIPTOR_RANGE srv_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1, &srv_range};
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor = {0, 0};
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    const D3D12_ROOT_SIGNATURE_DESC root_desc{2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> root_code;
    if (!ok(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_code, nullptr),
            "serialize root signature") ||
        !ok(device->CreateRootSignature(0, root_code->GetBufferPointer(), root_code->GetBufferSize(),
            IID_PPV_ARGS(&root)), "create root signature")) return false;

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.Get();
    pso_desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    return ok(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso)), "create compute PSO");
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &barrier);
}

bool submit_and_wait(ID3D12GraphicsCommandList* list, ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64 value)
{
    if (!ok(list->Close(), "close command list")) return false;
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    // A null event makes the fence call wait synchronously.
    return ok(queue->Signal(fence, value), "signal fence") &&
           ok(fence->SetEventOnCompletion(value, nullptr), "wait for fence");
}

bool run_dispatch(ID3D12Device* device, bool hazard)
{
    constexpr float expected = 0.5f;
    const D3D12_HEAP_PROPERTIES gpu_heap{D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC texture_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, 1, 1,
        DXGI_FORMAT_R32_FLOAT, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    ComPtr<ID3D12Resource> texture;
    if (!ok(device->CreateCommittedResource(&gpu_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&texture)), "create texture")) return false;

    D3D12_RESOURCE_DESC buffer_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, sizeof(expected), 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    ComPtr<ID3D12Resource> output;
    if (!ok(device->CreateCommittedResource(&gpu_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&output)), "create output buffer")) return false;
    const D3D12_HEAP_PROPERTIES cpu_heap{D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    buffer_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> readback;
    if (!hazard && !ok(device->CreateCommittedResource(&cpu_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)), "create readback buffer")) return false;

    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    const D3D12_DESCRIPTOR_HEAP_DESC srv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    ComPtr<ID3D12DescriptorHeap> srv_heap;
    if (!ok(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap)), "create RTV heap") ||
        !ok(device->CreateDescriptorHeap(&srv_desc, IID_PPV_ARGS(&srv_heap)), "create SRV heap")) return false;
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(texture.Get(), nullptr, rtv);
    device->CreateShaderResourceView(texture.Get(), nullptr, srv_heap->GetCPUDescriptorHandleForHeapStart());

    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso;
    if (!create_pipeline(device, root, pso)) return false;

    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (!ok(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "create direct queue") ||
        !ok(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "create fence") ||
        !ok(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create allocator") ||
        !ok(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pso.Get(),
            IID_PPV_ARGS(&list)), "create command list")) return false;

    const float color[] = {expected, 0, 0, 0};
    list->ClearRenderTargetView(rtv, color, 0, nullptr);
    if (!hazard)
        transition(list.Get(), texture.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ID3D12DescriptorHeap* heaps[] = {srv_heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(root.Get());
    list->SetComputeRootDescriptorTable(0, srv_heap->GetGPUDescriptorHandleForHeapStart());
    list->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
    list->Dispatch(1, 1, 1);
    if (!hazard) {
        transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, sizeof(expected));
    }

    // Follow the completed dispatch with an empty submission to drain GBV messages.
    if (!submit_and_wait(list.Get(), queue.Get(), fence.Get(), 1) ||
        !ok(allocator->Reset(), "reset allocator") ||
        !ok(list->Reset(allocator.Get(), nullptr), "reset command list") ||
        !submit_and_wait(list.Get(), queue.Get(), fence.Get(), 2)) return false;

    if (!hazard) {
        const D3D12_RANGE read{0, sizeof(expected)};
        void* mapped = nullptr;
        if (!ok(readback->Map(0, &read, &mapped), "map readback")) return false;
        float actual = 0;
        std::memcpy(&actual, mapped, sizeof(actual));
        const D3D12_RANGE written{0, 0};
        readback->Unmap(0, &written);
        if (actual != expected) {
            std::printf("FAIL shader readback: expected %g, got %g\n", expected, actual);
            return false;
        }
    }
    return true;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    const bool hazard = std::strcmp(argv[2], "hazard") == 0;
    if (!hazard && std::strcmp(argv[2], "clean") != 0) return 2;

    ComPtr<IDXGIFactory6> factory;
    if (!ok(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory")) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(argv[1], "hardware") == 0) {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (!ok(hr, "enumerate hardware adapter")) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (!ok(candidate->GetDesc1(&desc), "get adapter description")) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    } else if (std::strcmp(argv[1], "warp") != 0 ||
        !ok(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "get WARP adapter")) return 3;
    DXGI_ADAPTER_DESC adapter_desc{};
    if (!ok(adapter->GetDesc(&adapter_desc), "get selected adapter description")) return 3;
    std::printf("adapter=%ls case=%s\n", adapter_desc.Description, argv[2]);

    std::array<unsigned char, 64> parameters{};
    void* wrapped = nullptr;
    DebugMessages messages;
    ComPtr<ID3D12Device> device;
    if (!ok(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &wrapped),
            "D3D12XboxCreateDevice") ||
        !ok(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)),
            "get desktop device")) return 4;

    ComPtr<ID3D12InfoQueue1> info;
    DWORD cookie = 0;
    if (FAILED(device.As(&info)) ||
        FAILED(info->RegisterMessageCallback(on_message, D3D12_MESSAGE_CALLBACK_FLAG_NONE, &messages, &cookie))) {
        std::puts("FAIL no debug layer");
        return 5;
    }

    const bool dispatched = run_dispatch(device.Get(), hazard);
    const HRESULT removal_reason = device->GetDeviceRemovedReason();
    const bool unregistered = ok(info->UnregisterMessageCallback(cookie), "unregister debug callback");
    if (removal_reason != S_OK)
        std::printf("FAIL device removal status 0x%08lx\n", static_cast<unsigned long>(removal_reason));
    if (!dispatched || removal_reason != S_OK || !unregistered) return 6;

    const unsigned errors = messages.errors.load();
    const unsigned state_hazards = messages.state_hazards.load();
    std::printf("debug errors=%u state hazards=%u\n", errors, state_hazards);
    if (hazard && state_hazards == 0) {
        std::puts("FAIL GBV did not report the state hazard");
        return 7;
    }
    if (!hazard && errors != 0) {
        std::puts("FAIL unexpected debug errors in clean case");
        return 8;
    }
    return 0;
}
