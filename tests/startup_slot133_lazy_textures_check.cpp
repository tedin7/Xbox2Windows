// X2W_D3D12_LAZY_TEXTURES=1 (Refs #2764, #2379, #1199): slot 133 read-only textures defer desktop
// CreateCommittedResource until first real GPU/copy/CPU-real use.
//   startup_slot133_lazy_textures_check <warp|hardware> [off|no-retire]
//
// Checks:
//   1 Creating 16 MiB of read-only 2D textures via slot 133 + slot 18 SRVs (and CopyDescriptorsSimple) leaves
//     x2w-budget texture at 0/0 and avoids PagefileUsage growth; Resource::GetDesc() still returns the texture desc.
//   2 An untouched texture with no view retires its side record immediately at guest Release() (even without
//     RELEASE_RETIRED=1), and sampling a pre-materialization slot 18 SRV reads null (0) without device removal.
//   3 Materializing ONE texture via CopyTextureRegion commits ONLY that texture (1 object, matching texels), while
//     untouched textures remain unmaterialized (0 bytes).
//   4 Materializing a second texture via CopyTextureRegion AFTER its slot 18 SRV was created & copied refreshes the
//     existing CPU-only and shader-visible descriptors in place so compute sampling reads the expected texels.
//   5 Releasing all textures and overwriting/dropping views retires cleanly (x2w-budget texture and
//     x2w_d3d12_lazy_texture_count() return to 0 with X2W_D3D12_RELEASE_RETIRED=1).
//   6 Control (X2W_D3D12_LAZY_TEXTURES=0): all 16 MiB textures are committed immediately.
#include "startup_stubs/resource_budget.h"

#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <psapi.h>
#include <wrl/client.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_budget_snapshot(x2w_budget::Snapshot*, SIZE_T);
extern "C" __declspec(dllimport) unsigned x2w_d3d12_lazy_texture_count();
extern "C" __declspec(dllimport) void x2w_d3d12_set_lazy_refresh_hook(void (*fn)(void*), void* ctx);

namespace {
using namespace x2w_budget;

constexpr UINT tex_count = 16;
constexpr UINT tex_side = 512;
constexpr UINT tex_pitch = tex_side * 4;
constexpr UINT tex_bytes = tex_side * tex_pitch;  // 1 MiB each -> 16 MiB total
constexpr DWORD wait_ms = 10000;
constexpr int settle_limit = 200;

template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

template<class T>
ComPtr<T> desktop_object(IUnknown* guest, ID3D12Device* device, IUnknown* object)
{
    constexpr GUID key{0x0e34d52f, 0xd5a5, 0x4790, {0x91, 0xd4, 0x11, 0xa6, 0x7c, 0xbc, 0x2b, 0xc1}};
    ComPtr<T> output;
    if (FAILED(invoke(guest, 5, &key, object))) return output;
    ComPtr<IUnknown> reference;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT hr = device->GetPrivateData(key, &bytes, reference.GetAddressOf());
    invoke(guest, 5, &key, static_cast<IUnknown*>(nullptr));
    if (SUCCEEDED(hr) && reference) reference.As(&output);
    return output;
}

bool require(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) std::printf("FAIL: %s hr=0x%08lx\n", stage, static_cast<unsigned long>(hr));
    return SUCCEEDED(hr);
}

ComPtr<IDXGIAdapter1> pick_adapter(IDXGIFactory4* factory, bool warp)
{
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        return adapter;
    }
    wchar_t name[128]{};
    GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", name, static_cast<DWORD>(std::size(name)));
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            (!name[0] || std::wcsstr(desc.Description, name)) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)))
            return adapter;
    }
    return nullptr;
}

SIZE_T query_pagefile_usage()
{
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
    return pmc.PagefileUsage;
}

void fill_pattern(std::uint8_t* dst, UINT seed)
{
    for (UINT y = 0; y < tex_side; ++y) {
        for (UINT x = 0; x < tex_side; ++x) {
            const UINT off = y * tex_pitch + x * 4;
            dst[off + 0] = static_cast<std::uint8_t>(((seed * 17 + 23) + x * 3 + y * 5) & 0xff);
            dst[off + 1] = static_cast<std::uint8_t>(((seed * 41 + 59) + x * 7 + y * 11) & 0xff);
            dst[off + 2] = static_cast<std::uint8_t>(((seed * 73 + 97) + x * 13 + y * 19) & 0xff);
            dst[off + 3] = 255;
        }
    }
}

constexpr char probe_shader[] = R"(
Texture2D<float4> source : register(t0);
RWByteAddressBuffer result : register(u0);
[numthreads(1, 1, 1)] void main()
{
    result.Store4(0, asuint(source.Load(int3(0, 0, 0))));
}
)";

constexpr char translated_copy_shader[] = R"(
Texture2D<float4> source : register(t0);
RWByteAddressBuffer result : register(u0);
[numthreads(1, 1, 1)] void main()
{
    result.Store4(0, asuint(source.Load(int3(0, 0, 0))));
}
)";

struct GuestHeap {
    ComPtr<IUnknown> guest;
    ComPtr<ID3D12DescriptorHeap> real;
    std::vector<std::uint8_t> backing;
    UINT64 cpu = 0;
    UINT64 gpu = 0;
    bool visible = false;
};

struct Context {
    IUnknown* guest = nullptr;
    ID3D12Device* device = nullptr;
    ComPtr<ID3D12CommandQueue> own_queue;
    ComPtr<ID3D12CommandAllocator> own_allocator;
    ComPtr<ID3D12GraphicsCommandList> own_list;
    ComPtr<ID3D12Fence> own_fence;
    UINT64 own_value = 0;
    HANDLE event = nullptr;
    ComPtr<ID3D12Resource> probe_out, probe_readback;
    ComPtr<ID3D12RootSignature> probe_root;
    ComPtr<ID3D12PipelineState> probe_pso;
    ComPtr<ID3D12DescriptorHeap> probe_heap;
    ComPtr<IUnknown> guest_queue, guest_fence, guest_alloc, guest_list, drain_alloc, drain_list;
    ComPtr<ID3D12Fence> guest_fence_real;
    UINT64 guest_fence_value = 0;
    UINT increment = 0;

    bool wait(ID3D12Fence* fence, UINT64 value)
    {
        if (fence->GetCompletedValue() >= value) return true;
        if (FAILED(fence->SetEventOnCompletion(value, event))) return false;
        return WaitForSingleObject(event, wait_ms) == WAIT_OBJECT_0;
    }

    bool own_run()
    {
        if (!require(own_list->Close(), "own list close")) return false;
        ID3D12CommandList* lists[] = {own_list.Get()};
        own_queue->ExecuteCommandLists(1, lists);
        if (!require(own_queue->Signal(own_fence.Get(), ++own_value), "own signal")) return false;
        return wait(own_fence.Get(), own_value);
    }

    bool guest_sync()
    {
        if (!require(invoke(guest_queue.Get(), 14, guest_fence.Get(), ++guest_fence_value), "guest signal")) return false;
        return wait(guest_fence_real.Get(), guest_fence_value);
    }

    bool make_heap(GuestHeap& h, UINT count, bool visible)
    {
        h.backing.assign(std::size_t{count} * increment, 0);
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            UINT64 backing, bytes;
        } request{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count,
                   visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE},
                  reinterpret_cast<UINT64>(h.backing.data()), h.backing.size()};
        if (!require(invoke(guest, 14, &request, &__uuidof(ID3D12DescriptorHeap), h.guest.GetAddressOf()), "guest heap"))
            return false;
        h.real = desktop_object<ID3D12DescriptorHeap>(guest, device, h.guest.Get());
        if (!h.real) {
            std::puts("FAIL: guest heap has no desktop object");
            return false;
        }
        h.cpu = invoke<UINT64>(h.guest.Get(), 9);
        h.gpu = visible ? invoke<UINT64>(h.guest.Get(), 10) : 0;
        h.visible = visible;
        return true;
    }

    bool read_slot(const GuestHeap& h, UINT index, float (&texel)[4])
    {
        if (!require(own_allocator->Reset(), "probe allocator") ||
            !require(own_list->Reset(own_allocator.Get(), nullptr), "probe list"))
            return false;
        ID3D12DescriptorHeap* bound = probe_heap.Get();
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = probe_heap->GetGPUDescriptorHandleForHeapStart();
        if (h.visible) {
            bound = h.real.Get();
            gpu = h.real->GetGPUDescriptorHandleForHeapStart();
            gpu.ptr += std::uint64_t{index} * increment;
        } else {
            D3D12_CPU_DESCRIPTOR_HANDLE src = h.real->GetCPUDescriptorHandleForHeapStart();
            src.ptr += std::uint64_t{index} * increment;
            device->CopyDescriptorsSimple(1, probe_heap->GetCPUDescriptorHandleForHeapStart(), src,
                                          D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
        own_list->SetDescriptorHeaps(1, &bound);
        own_list->SetComputeRootSignature(probe_root.Get());
        own_list->SetPipelineState(probe_pso.Get());
        own_list->SetComputeRootDescriptorTable(0, gpu);
        own_list->SetComputeRootUnorderedAccessView(1, probe_out->GetGPUVirtualAddress());
        own_list->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE, {}};
        b.Transition = {probe_out.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE};
        own_list->ResourceBarrier(1, &b);
        own_list->CopyBufferRegion(probe_readback.Get(), 0, probe_out.Get(), 0, 16);
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        own_list->ResourceBarrier(1, &b);
        if (!own_run()) return false;
        void* mapped = nullptr;
        const D3D12_RANGE all{0, 16};
        if (!require(probe_readback->Map(0, &all, &mapped), "probe readback map")) return false;
        std::memcpy(texel, mapped, sizeof(texel));
        probe_readback->Unmap(0, nullptr);
        return true;
    }

    bool copy_texture_to_readback(IUnknown* guest_tex, IUnknown* guest_readback_buf)
    {
        if (!require(invoke(guest_alloc.Get(), 8), "guest alloc reset") ||
            !require(invoke(guest_list.Get(), 139, guest_alloc.Get(), nullptr), "guest list reset"))
            return false;
        D3D12_TEXTURE_COPY_LOCATION dst{static_cast<ID3D12Resource*>(guest_readback_buf), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, tex_side, tex_side, 1, tex_pitch};
        D3D12_TEXTURE_COPY_LOCATION src{static_cast<ID3D12Resource*>(guest_tex), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        invoke<void>(guest_list.Get(), 16, &dst, 0u, 0u, 0u, &src, static_cast<const D3D12_BOX*>(nullptr));
        if (!require(invoke(guest_list.Get(), 9), "guest list close")) return false;
        IUnknown* lists[] = {guest_list.Get()};
        invoke<void>(guest_queue.Get(), 10, 1u, lists);
        return guest_sync();
    }

    bool reset_copy_list()
    {
        if (!require(invoke(guest_alloc.Get(), 8), "guest alloc reset") ||
            !require(invoke(guest_list.Get(), 139, guest_alloc.Get(), nullptr), "guest list reset"))
            return false;
        return require(invoke(guest_list.Get(), 9), "guest list close");
    }

    void drain_once()
    {
        IUnknown* lists[] = {drain_list.Get()};
        invoke<void>(guest_queue.Get(), 10, 1u, lists);
    }
};

D3D12_SHADER_RESOURCE_VIEW_DESC texture_srv_desc()
{
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    return srv;
}

bool verify_readback_bytes(ID3D12Resource* readback, UINT seed, const char* label)
{
    void* mapped = nullptr;
    const D3D12_RANGE all{0, tex_bytes};
    if (!require(readback->Map(0, &all, &mapped), "readback map")) return false;
    const auto* actual = static_cast<const std::uint8_t*>(mapped);
    std::vector<std::uint8_t> expected(tex_bytes);
    fill_pattern(expected.data(), seed);
    UINT mismatches = 0;
    for (UINT i = 0; i < tex_bytes; ++i) {
        if (actual[i] != expected[i]) ++mismatches;
    }
    readback->Unmap(0, nullptr);
    if (mismatches) {
        std::printf("FAIL: %s readback had %u mismatched bytes (out of %u)\n", label, mismatches, tex_bytes);
        return false;
    }
    return true;
}

bool verify_texel00(const float (&got)[4], UINT seed, const char* label)
{
    std::uint8_t expected_bytes[4]{};
    expected_bytes[0] = static_cast<std::uint8_t>((seed * 17 + 23) & 0xff);
    expected_bytes[1] = static_cast<std::uint8_t>((seed * 41 + 59) & 0xff);
    expected_bytes[2] = static_cast<std::uint8_t>((seed * 73 + 97) & 0xff);
    expected_bytes[3] = 255;
    for (int c = 0; c < 4; ++c) {
        const float exp = static_cast<float>(expected_bytes[c]) / 255.0f;
        if (std::fabs(got[c] - exp) > 1e-3f) {
            std::printf("FAIL: %s texel(0,0)[%d]=%.5f expected=%.5f\n", label, c, got[c], exp);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (argc == 3 && std::strcmp(argv[2], "off") && std::strcmp(argv[2], "no-retire")))
        return 2;
    const bool is_warp = std::strcmp(argv[1], "warp") == 0;
    const bool expect_off = argc == 3 && std::strcmp(argv[2], "off") == 0;
    const bool expect_no_retire = argc == 3 && std::strcmp(argv[2], "no-retire") == 0;

    char env_buf[2]{};
    const bool lazy_on = GetEnvironmentVariableA("X2W_D3D12_LAZY_TEXTURES", env_buf, sizeof(env_buf)) == 1 && env_buf[0] == '1';
    env_buf[0] = 0;
    const bool retire_on = GetEnvironmentVariableA("X2W_D3D12_RELEASE_RETIRED", env_buf, sizeof(env_buf)) == 1 && env_buf[0] == '1';
    if (lazy_on == expect_off || retire_on == expect_no_retire) return 2;

    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), is_warp);
    if (!adapter) return 77;

    env_buf[0] = 0;
    const bool debug_on = GetEnvironmentVariableA("X2W_D3D12_DEBUG", env_buf, sizeof(env_buf)) == 1 && env_buf[0] == '1';
    if (debug_on) {
        ComPtr<ID3D12Debug> dbg;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) return 77;
        dbg->EnableDebugLayer();
    }

    ComPtr<IUnknown> guest;
    ComPtr<ID3D12Device> device;
    std::array<unsigned char, 64> parameters{};
    if (!require(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                      reinterpret_cast<void**>(guest.GetAddressOf())), "Xbox device") ||
        !require(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "desktop device"))
        return 3;

    ComPtr<ID3D12InfoQueue> debug;
    if (debug_on && FAILED(device.As(&debug))) return 3;
    if (debug) {
        D3D12_MESSAGE_SEVERITY severities[] = {
            D3D12_MESSAGE_SEVERITY_CORRUPTION, D3D12_MESSAGE_SEVERITY_ERROR, D3D12_MESSAGE_SEVERITY_WARNING};
        D3D12_INFO_QUEUE_FILTER filter{};
        filter.AllowList.NumSeverities = static_cast<UINT>(std::size(severities));
        filter.AllowList.pSeverityList = severities;
        if (FAILED(debug->AddStorageFilterEntries(&filter))) return 3;
        debug->ClearStoredMessages();
    }

    DXGI_ADAPTER_DESC1 adapter_desc{};
    adapter->GetDesc1(&adapter_desc);
    std::printf("adapter=%ls debug_layer=%s lazy=%s retire=%s\n", adapter_desc.Description,
                debug ? "enabled" : "off", lazy_on ? "on" : "off", retire_on ? "on" : "off");

    Context c{guest.Get(), device.Get()};
    c.increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    c.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, readback_heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_RESOURCE_DESC out_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                       D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    const D3D12_RESOURCE_DESC small_rb_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                            D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    const D3D12_DESCRIPTOR_HEAP_DESC probe_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                                     D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (!c.event ||
        !require(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&c.own_queue)), "own queue") ||
        !require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&c.own_allocator)), "own alloc") ||
        !require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, c.own_allocator.Get(), nullptr, IID_PPV_ARGS(&c.own_list)), "own list") ||
        !require(c.own_list->Close(), "own list close") ||
        !require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c.own_fence)), "own fence") ||
        !require(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &out_desc,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&c.probe_out)), "probe out") ||
        !require(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &small_rb_desc,
                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&c.probe_readback)), "probe rb") ||
        !require(device->CreateDescriptorHeap(&probe_heap_desc, IID_PPV_ARGS(&c.probe_heap)), "probe heap"))
        return 4;

    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER root_params[2]{};
    root_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_params[0].DescriptorTable = {1, &range};
    root_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    root_params[1].Descriptor = {0, 0};
    const D3D12_ROOT_SIGNATURE_DESC root_desc{2, root_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> root_blob, errors, code;
    if (!require(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, root_blob.GetAddressOf(),
                                             errors.GetAddressOf()), "probe root serialize") ||
        !require(device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
                                             IID_PPV_ARGS(&c.probe_root)), "probe root") ||
        !require(D3DCompile(probe_shader, std::strlen(probe_shader), nullptr, nullptr, nullptr, "main", "cs_5_1", 0, 0,
                            code.GetAddressOf(), errors.ReleaseAndGetAddressOf()), "probe compile"))
        return 4;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = c.probe_root.Get();
    pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    if (!require(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&c.probe_pso)), "probe pso")) return 4;

    const unsigned list_desc[] = {static_cast<unsigned>(D3D12_COMMAND_LIST_TYPE_DIRECT), 2, 1, 0};
    if (!require(invoke(guest.Get(), 8, &qd, &__uuidof(ID3D12CommandQueue), c.guest_queue.GetAddressOf()), "guest queue") ||
        !require(invoke(guest.Get(), 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), c.guest_fence.GetAddressOf()), "guest fence") ||
        !require(invoke(guest.Get(), 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator), c.guest_alloc.GetAddressOf()), "guest alloc") ||
        !require(invoke(guest.Get(), 73, list_desc, c.guest_alloc.Get(), nullptr, &__uuidof(ID3D12GraphicsCommandList), c.guest_list.GetAddressOf()), "guest list") ||
        !require(invoke(c.guest_list.Get(), 139, c.guest_alloc.Get(), nullptr), "guest list reset") ||
        !require(invoke(c.guest_list.Get(), 9), "guest list close") ||
        !require(invoke(guest.Get(), 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator), c.drain_alloc.GetAddressOf()), "drain alloc") ||
        !require(invoke(guest.Get(), 73, list_desc, c.drain_alloc.Get(), nullptr, &__uuidof(ID3D12GraphicsCommandList), c.drain_list.GetAddressOf()), "drain list") ||
        !require(invoke(c.drain_list.Get(), 139, c.drain_alloc.Get(), nullptr), "drain list reset") ||
        !require(invoke(c.drain_list.Get(), 9), "drain list close"))
        return 4;
    c.guest_fence_real = desktop_object<ID3D12Fence>(guest.Get(), device.Get(), c.guest_fence.Get());
    if (!c.guest_fence_real) return 4;

    // Guest readback buffer for CopyTextureRegion materialization + texel verification.
    const D3D12_RESOURCE_DESC rb_buf_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, tex_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                                          {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    ComPtr<IUnknown> guest_rb;
    if (!require(invoke(guest.Get(), 27, &readback_heap, D3D12_HEAP_FLAG_NONE, &rb_buf_desc,
                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, &__uuidof(ID3D12Resource), guest_rb.GetAddressOf()),
                 "guest readback buffer"))
        return 4;
    ComPtr<ID3D12Resource> guest_rb_real = desktop_object<ID3D12Resource>(guest.Get(), device.Get(), guest_rb.Get());
    if (!guest_rb_real) return 4;

    GuestHeap cpu_heap, vis_heap;
    if (!c.make_heap(cpu_heap, tex_count + 4, false) || !c.make_heap(vis_heap, 8, true)) return 4;

    // Pre-commit guest VirtualAlloc backing for 16 x 1 MiB = 16 MiB textures before measuring slot-133 commit delta.
    std::vector<void*> title_mem(tex_count, nullptr);
    for (UINT i = 0; i < tex_count; ++i) {
        title_mem[i] = VirtualAlloc(nullptr, tex_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!title_mem[i]) return 4;
        fill_pattern(static_cast<std::uint8_t*>(title_mem[i]), i + 1);
    }

    Snapshot base{};
    if (!x2w_d3d12_budget_snapshot(&base, sizeof(base))) return 4;
    const SIZE_T commit_before = query_pagefile_usage();

    const D3D12_RESOURCE_DESC tex_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, tex_side, tex_side, 1, 1,
                                       DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                       D3D12_RESOURCE_FLAG_NONE};
    const D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = texture_srv_desc();
    std::vector<ComPtr<IUnknown>> tex(tex_count);
    for (UINT i = 0; i < tex_count; ++i) {
        if (!require(invoke(guest.Get(), 133, title_mem[i], &tex_desc, D3D12_RESOURCE_STATE_COMMON,
                            nullptr, &__uuidof(ID3D12Resource), tex[i].GetAddressOf()), "slot 133 texture"))
            return 5;
        invoke<void>(guest.Get(), 18, tex[i].Get(), &srv_desc, cpu_heap.cpu + std::uint64_t{i} * c.increment);
    }
    invoke<void>(guest.Get(), 24, 1u, vis_heap.cpu + std::uint64_t{0} * c.increment,
                 cpu_heap.cpu + std::uint64_t{3} * c.increment, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    invoke<void>(guest.Get(), 24, 1u, vis_heap.cpu + std::uint64_t{1} * c.increment,
                 cpu_heap.cpu + std::uint64_t{5} * c.increment, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    const SIZE_T commit_after = query_pagefile_usage();
    Snapshot snap{};
    if (!x2w_d3d12_budget_snapshot(&snap, sizeof(snap))) return 4;

    const auto free_title_mem = [&] {
        for (void* p : title_mem) if (p) VirtualFree(p, 0, MEM_RELEASE);
    };

    if (!lazy_on) {
        if (snap.live[texture].objects != base.live[texture].objects + tex_count ||
            snap.live[texture].bytes < base.live[texture].bytes + std::uint64_t{tex_count} * tex_bytes) {
            std::printf("FAIL: control expected %u textures (>= %llu bytes), got live=%llu/%llu\n",
                        tex_count, static_cast<unsigned long long>(tex_count) * tex_bytes,
                        snap.live[texture].bytes, snap.live[texture].objects);
            free_title_mem();
            return 5;
        }
        if (x2w_d3d12_lazy_texture_count() != 0) {
            std::printf("FAIL: control expected lazy_texture_count=0, got %u\n", x2w_d3d12_lazy_texture_count());
            free_title_mem();
            return 5;
        }
        for (UINT i = 0; i < tex_count; ++i) {
            if (!desktop_object<ID3D12Resource>(guest.Get(), device.Get(), tex[i].Get())) {
                std::printf("FAIL: control texture %u has no desktop resource\n", i);
                free_title_mem();
                return 5;
            }
        }
        if (is_warp && (commit_after <= commit_before || commit_after - commit_before < 12ull * 1024 * 1024)) {
            std::printf("FAIL: control WARP PagefileUsage delta=%zu expected >= 12 MiB\n",
                        commit_after > commit_before ? commit_after - commit_before : 0);
            free_title_mem();
            return 5;
        }
        std::printf("control: 16 textures committed immediately (%llu bytes, PagefileUsage delta=%zu)\n",
                    snap.live[texture].bytes - base.live[texture].bytes,
                    commit_after > commit_before ? commit_after - commit_before : 0);
        free_title_mem();
        CloseHandle(c.event);
        return 0;
    }

    // 1. Lazy creation leaves x2w-budget texture at 0/0, avoids PagefileUsage growth, preserves GetDesc().
    if (snap.live[texture].objects != base.live[texture].objects ||
        snap.live[texture].bytes != base.live[texture].bytes ||
        snap.retained[texture].objects != base.retained[texture].objects ||
        snap.retained[texture].bytes != base.retained[texture].bytes) {
        std::printf("FAIL: lazy creation committed textures: live=%llu/%llu retained=%llu/%llu\n",
                    snap.live[texture].bytes, snap.live[texture].objects,
                    snap.retained[texture].bytes, snap.retained[texture].objects);
        free_title_mem();
        return 5;
    }
    if (x2w_d3d12_lazy_texture_count() != tex_count) {
        std::printf("FAIL: expected lazy_texture_count=%u, got %u\n", tex_count, x2w_d3d12_lazy_texture_count());
        free_title_mem();
        return 5;
    }
    for (UINT i = 0; i < tex_count; ++i) {
        if (desktop_object<ID3D12Resource>(guest.Get(), device.Get(), tex[i].Get())) {
            std::printf("FAIL: lazy texture %u unexpectedly has a desktop resource\n", i);
            free_title_mem();
            return 5;
        }
    }
    const D3D12_RESOURCE_DESC got_desc = static_cast<ID3D12Resource*>(tex[0].Get())->GetDesc();
    if (got_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || got_desc.Width != tex_side ||
        got_desc.Height != tex_side || got_desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM) {
        std::printf("FAIL: GetDesc on unmaterialized lazy texture returned dim=%d %llux%u fmt=%d\n",
                    got_desc.Dimension, got_desc.Width, got_desc.Height, got_desc.Format);
        free_title_mem();
        return 5;
    }
    if (commit_after > commit_before && commit_after - commit_before >= 4ull * 1024 * 1024) {
        std::printf("FAIL: lazy PagefileUsage grew by %zu bytes (>= 4 MiB)\n", commit_after - commit_before);
        free_title_mem();
        return 5;
    }

    // 2. Pre-materialization SRV probe reads null (0,0,0,0) without materializing or removing device.
    float pre_texel[4]{1.0f, 1.0f, 1.0f, 1.0f};
    if (!c.read_slot(vis_heap, 1, pre_texel)) {
        free_title_mem();
        return 5;
    }
    if (pre_texel[0] != 0.0f || pre_texel[1] != 0.0f || pre_texel[2] != 0.0f || pre_texel[3] != 0.0f) {
        std::printf("FAIL: pre-materialization SRV read %.3f %.3f %.3f %.3f instead of 0\n",
                    pre_texel[0], pre_texel[1], pre_texel[2], pre_texel[3]);
        free_title_mem();
        return 5;
    }

    // Untouched texture with no view retires its side record immediately at guest Release().
    ComPtr<IUnknown> tex_noview;
    if (!require(invoke(guest.Get(), 133, title_mem[0], &tex_desc, D3D12_RESOURCE_STATE_COMMON,
                        nullptr, &__uuidof(ID3D12Resource), tex_noview.GetAddressOf()), "slot 133 no-view")) {
        free_title_mem();
        return 5;
    }
    if (x2w_d3d12_lazy_texture_count() != tex_count + 1) {
        std::printf("FAIL: expected lazy_texture_count=%u before Release, got %u\n",
                    tex_count + 1, x2w_d3d12_lazy_texture_count());
        free_title_mem();
        return 5;
    }
    tex_noview.Reset();
    if (x2w_d3d12_lazy_texture_count() != tex_count) {
        std::printf("FAIL: untouched no-view texture leaked side record at Release() (count=%u)\n",
                    x2w_d3d12_lazy_texture_count());
        free_title_mem();
        return 5;
    }

    // 3. Materialize ONLY tex[3] via CopyTextureRegion and verify texel readback + budget = 1 object.
    if (!c.copy_texture_to_readback(tex[3].Get(), guest_rb.Get()) ||
        !verify_readback_bytes(guest_rb_real.Get(), 4, "tex[3] CopyTextureRegion")) {
        free_title_mem();
        return 5;
    }
    if (!x2w_d3d12_budget_snapshot(&snap, sizeof(snap))) return 4;
    if (snap.live[texture].objects != base.live[texture].objects + 1 ||
        snap.live[texture].bytes < base.live[texture].bytes + tex_bytes) {
        std::printf("FAIL: after materializing tex[3], expected 1 texture (>= %u bytes), got live=%llu/%llu\n",
                    tex_bytes, snap.live[texture].bytes, snap.live[texture].objects);
        free_title_mem();
        return 5;
    }
    for (UINT i = 0; i < tex_count; ++i) {
        const bool has_real = desktop_object<ID3D12Resource>(guest.Get(), device.Get(), tex[i].Get()) != nullptr;
        if (has_real != (i == 3)) {
            std::printf("FAIL: after tex[3] materialization, tex[%u] desktop_object=%d expected=%d\n",
                        i, has_real ? 1 : 0, i == 3 ? 1 : 0);
            free_title_mem();
            return 5;
        }
    }

    // 4. Materialize tex[5] via CopyTextureRegion AFTER its slot 18 SRV was created and copied before materialization,
    //    then sample the pre-created SRVs in both the shader-visible and CPU-only heaps without recreating them.
    if (!c.copy_texture_to_readback(tex[5].Get(), guest_rb.Get()) ||
        !verify_readback_bytes(guest_rb_real.Get(), 6, "tex[5] CopyTextureRegion")) {
        free_title_mem();
        return 5;
    }
    if (!x2w_d3d12_budget_snapshot(&snap, sizeof(snap))) return 4;
    if (snap.live[texture].objects != base.live[texture].objects + 2) {
        std::printf("FAIL: after materializing tex[5], expected 2 textures, got %llu\n", snap.live[texture].objects);
        free_title_mem();
        return 5;
    }
    float post_vis[4]{}, post_cpu[4]{};
    if (!c.read_slot(vis_heap, 1, post_vis) || !verify_texel00(post_vis, 6, "vis_heap[1] refreshed SRV") ||
        !c.read_slot(cpu_heap, 5, post_cpu) || !verify_texel00(post_cpu, 6, "cpu_heap[5] refreshed SRV")) {
        free_title_mem();
        return 5;
    }

    // 4b. Prepare / overwrite / resume race control: overwrite vis_heap[3] with tex_rep (seed 19) and destroy
    //     doomed_heap right between materialize_lazy_texture's snapshot and descriptor publication.
    {
        void* race_mem = VirtualAlloc(nullptr, tex_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        void* rep_mem = VirtualAlloc(nullptr, tex_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!race_mem || !rep_mem) {
            if (race_mem) VirtualFree(race_mem, 0, MEM_RELEASE);
            if (rep_mem) VirtualFree(rep_mem, 0, MEM_RELEASE);
            free_title_mem();
            return 4;
        }
        fill_pattern(static_cast<std::uint8_t*>(race_mem), 12);
        fill_pattern(static_cast<std::uint8_t*>(rep_mem), 19);
        ComPtr<IUnknown> tex_race, tex_rep;
        GuestHeap doomed_heap;
        if (!require(invoke(guest.Get(), 133, race_mem, &tex_desc, D3D12_RESOURCE_STATE_COMMON,
                            nullptr, &__uuidof(ID3D12Resource), tex_race.GetAddressOf()), "slot 133 race") ||
            !require(invoke(guest.Get(), 133, rep_mem, &tex_desc, D3D12_RESOURCE_STATE_COMMON,
                            nullptr, &__uuidof(ID3D12Resource), tex_rep.GetAddressOf()), "slot 133 rep") ||
            !c.copy_texture_to_readback(tex_rep.Get(), guest_rb.Get()) ||
            !c.make_heap(doomed_heap, 2, false)) {
            VirtualFree(race_mem, 0, MEM_RELEASE);
            VirtualFree(rep_mem, 0, MEM_RELEASE);
            free_title_mem();
            return 5;
        }
        invoke<void>(guest.Get(), 18, tex_race.Get(), &srv_desc, cpu_heap.cpu + std::uint64_t{tex_count} * c.increment);
        invoke<void>(guest.Get(), 24, 1u, vis_heap.cpu + std::uint64_t{3} * c.increment,
                     cpu_heap.cpu + std::uint64_t{tex_count} * c.increment, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        invoke<void>(guest.Get(), 24, 1u, doomed_heap.cpu,
                     cpu_heap.cpu + std::uint64_t{tex_count} * c.increment, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        struct RaceState {
            IUnknown* guest = nullptr;
            IUnknown* tex_rep = nullptr;
            const D3D12_SHADER_RESOURCE_VIEW_DESC* srv = nullptr;
            UINT64 overwrite_dest = 0;
            GuestHeap* doomed = nullptr;
            bool fired = false;
        } race{guest.Get(), tex_rep.Get(), &srv_desc, vis_heap.cpu + std::uint64_t{3} * c.increment, &doomed_heap};

        x2w_d3d12_set_lazy_refresh_hook([](void* ctx) {
            auto* st = static_cast<RaceState*>(ctx);
            if (st->fired) return;
            st->fired = true;
            invoke<void>(st->guest, 18, st->tex_rep, st->srv, st->overwrite_dest);
            st->doomed->guest.Reset();
            st->doomed->real.Reset();
        }, &race);
        const bool copied = c.copy_texture_to_readback(tex_race.Get(), guest_rb.Get());
        x2w_d3d12_set_lazy_refresh_hook(nullptr, nullptr);
        if (!copied || !race.fired) {
            std::puts("FAIL: prepare/overwrite/resume race hook did not complete");
            VirtualFree(race_mem, 0, MEM_RELEASE);
            VirtualFree(rep_mem, 0, MEM_RELEASE);
            free_title_mem();
            return 5;
        }
        float got_race[4]{};
        if (!c.read_slot(vis_heap, 3, got_race) || !verify_texel00(got_race, 19, "vis_heap[3] overwritten mid-refresh")) {
            VirtualFree(race_mem, 0, MEM_RELEASE);
            VirtualFree(rep_mem, 0, MEM_RELEASE);
            free_title_mem();
            return 5;
        }
        invoke<void>(guest.Get(), 18, static_cast<IUnknown*>(nullptr), &srv_desc,
                     cpu_heap.cpu + std::uint64_t{tex_count} * c.increment);
        invoke<void>(guest.Get(), 18, static_cast<IUnknown*>(nullptr), &srv_desc,
                     vis_heap.cpu + std::uint64_t{3} * c.increment);
        tex_race.Reset();
        tex_rep.Reset();
        if (!c.reset_copy_list() || !c.guest_sync()) {
            VirtualFree(race_mem, 0, MEM_RELEASE);
            VirtualFree(rep_mem, 0, MEM_RELEASE);
            free_title_mem();
            return 5;
        }
        if (retire_on) {
            for (int attempt = 0; attempt < settle_limit; ++attempt) c.drain_once();
        }
        VirtualFree(race_mem, 0, MEM_RELEASE);
        VirtualFree(rep_mem, 0, MEM_RELEASE);
    }

    // 4c. #2764 order: create lazy tex_rel (seed 11) -> slot 18 SRV -> CopyDescriptorsSimple to vis_heap[4] ->
    //     overwrite cpu_heap -> last guest Release() while still unmaterialized -> first use on a fresh guest
    //     command list through the still-live descriptor (dispatch_translated) behind a gated queue -> overwrite
    //     while in flight -> open gate -> verify distinct nonzero texel and clean retirement.
    {
        const std::filesystem::path cache_dir =
            std::filesystem::temp_directory_path() / ("x2w-lazytex-cache-" + std::to_string(GetCurrentProcessId()));
        std::error_code ec;
        std::filesystem::remove_all(cache_dir, ec);
        std::filesystem::create_directories(cache_dir, ec);
        SetEnvironmentVariableW(L"X2W_SHADER_CACHE", cache_dir.c_str());

        const D3D12_DESCRIPTOR_RANGE srv_rng{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        const D3D12_DESCRIPTOR_RANGE uav_rng{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER comp_params[2]{};
        comp_params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        comp_params[0].DescriptorTable = {1, &srv_rng};
        comp_params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        comp_params[1].DescriptorTable = {1, &uav_rng};
        const D3D12_ROOT_SIGNATURE_DESC comp_root_desc{2, comp_params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> comp_blob, comp_err, comp_cso;
        ComPtr<IUnknown> guest_comp_root, guest_comp_pso;
        if (!require(D3D12SerializeRootSignature(&comp_root_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                                 comp_blob.GetAddressOf(), comp_err.GetAddressOf()), "comp root ser") ||
            !require(invoke(guest.Get(), 16, 0u, comp_blob->GetBufferPointer(), comp_blob->GetBufferSize(),
                            &__uuidof(ID3D12RootSignature), guest_comp_root.GetAddressOf()), "guest comp root") ||
            !require(D3DCompile(translated_copy_shader, std::strlen(translated_copy_shader), nullptr, nullptr, nullptr,
                                "main", "cs_5_1", 0, 0, comp_cso.GetAddressOf(), comp_err.ReleaseAndGetAddressOf()),
                     "translated copy cs compile")) {
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 4;
        }
        const std::array<UINT, 2> dummy_code{0xbfa00003u, 0xbf810000u};
        const std::array<UINT, 2> dummy_data{0x11223344u, 0x55667788u};
        std::array<UINT64, 12> pso_fields{};
        pso_fields[0] = reinterpret_cast<UINT64>(guest_comp_root.Get());
        pso_fields[5] = reinterpret_cast<UINT64>(dummy_code.data());
        pso_fields[6] = dummy_code.size() * sizeof(UINT);
        pso_fields[7] = reinterpret_cast<UINT64>(dummy_data.data());
        pso_fields[8] = dummy_data.size() * sizeof(UINT);
        if (!require(invoke(guest.Get(), 67, pso_fields.data(),
                            &__uuidof(ID3D12PipelineState), guest_comp_pso.GetAddressOf()), "guest slot 67 pso")) {
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 4;
        }
        std::string key_hex;
        for (const auto& ent : std::filesystem::directory_iterator(cache_dir, ec)) {
            const std::string fname = ent.path().filename().string();
            if (fname.size() == 24 && fname.rfind("req_", 0) == 0 && fname.substr(20) == ".c67") {
                key_hex = fname.substr(4, 16);
                break;
            }
        }
        if (key_hex.empty()) {
            std::puts("FAIL: slot 67 did not publish req_<key>.c67");
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 4;
        }
        {
            std::ofstream mf(cache_dir / (key_hex + ".txt"), std::ios::binary);
            mf << "x2w-manifest 2\nstage cs\ngroup 1 1 1\nud 0\nsrv 0 tex2d table 0 0\nuav 0 table 1 0\n";
            std::ofstream cf(cache_dir / (key_hex + ".cs.cso"), std::ios::binary);
            cf.write(static_cast<const char*>(comp_cso->GetBufferPointer()),
                     static_cast<std::streamsize>(comp_cso->GetBufferSize()));
        }

        void* rel_mem = VirtualAlloc(nullptr, tex_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        void* uav_mem = VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!rel_mem || !uav_mem) {
            if (rel_mem) VirtualFree(rel_mem, 0, MEM_RELEASE);
            if (uav_mem) VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 4;
        }
        fill_pattern(static_cast<std::uint8_t*>(rel_mem), 11);
        std::memset(uav_mem, 0, 65536);

        const D3D12_RESOURCE_DESC uav_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                                           {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
        ComPtr<IUnknown> tex_rel, uav_buf;
        if (!require(invoke(guest.Get(), 133, rel_mem, &tex_desc, D3D12_RESOURCE_STATE_COMMON,
                            nullptr, &__uuidof(ID3D12Resource), tex_rel.GetAddressOf()), "slot 133 tex_rel") ||
            !require(invoke(guest.Get(), 133, uav_mem, &uav_desc, D3D12_RESOURCE_STATE_COMMON,
                            nullptr, &__uuidof(ID3D12Resource), uav_buf.GetAddressOf()), "slot 133 uav_buf")) {
            VirtualFree(rel_mem, 0, MEM_RELEASE);
            VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 5;
        }
        if (!desktop_object<ID3D12Resource>(guest.Get(), device.Get(), uav_buf.Get())) {
            std::puts("FAIL: uav_buf was not placed");
            VirtualFree(rel_mem, 0, MEM_RELEASE);
            VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 5;
        }
        // Raw buffer descriptor words in vis_heap[5] for uav 0 table 1 0.
        const UINT64 uav_addr = reinterpret_cast<UINT64>(uav_mem);
        const std::uint32_t uav_words[4]{
            static_cast<std::uint32_t>(uav_addr & 0xffffffffu),
            static_cast<std::uint32_t>((uav_addr >> 32) & 0xffffu),
            64u,
            0u,
        };
        std::memcpy(vis_heap.backing.data() + std::size_t{5} * c.increment, uav_words, sizeof(uav_words));

        invoke<void>(guest.Get(), 18, tex_rel.Get(), &srv_desc, cpu_heap.cpu + std::uint64_t{tex_count + 1} * c.increment);
        invoke<void>(guest.Get(), 24, 1u, vis_heap.cpu + std::uint64_t{4} * c.increment,
                     cpu_heap.cpu + std::uint64_t{tex_count + 1} * c.increment, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        invoke<void>(guest.Get(), 18, static_cast<IUnknown*>(nullptr), &srv_desc,
                     cpu_heap.cpu + std::uint64_t{tex_count + 1} * c.increment);

        Snapshot pre_rel{};
        if (!x2w_d3d12_budget_snapshot(&pre_rel, sizeof(pre_rel))) return 4;

        // Last guest Release() BEFORE first use!
        tex_rel.Reset();

        Snapshot after_rel{};
        if (!x2w_d3d12_budget_snapshot(&after_rel, sizeof(after_rel))) return 4;
        if (after_rel.live[texture].objects != pre_rel.live[texture].objects ||
            after_rel.retained[texture].objects != pre_rel.retained[texture].objects) {
            std::puts("FAIL: tex_rel materialized on guest Release() before first use");
            VirtualFree(rel_mem, 0, MEM_RELEASE);
            VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 5;
        }

        // Fresh guest command list: first use through still-live copied descriptor vis_heap[4] after Release().
        ComPtr<IUnknown> gate_fence;
        if (!require(invoke(guest.Get(), 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence),
                            gate_fence.GetAddressOf()), "gate fence") ||
            !require(invoke(c.guest_alloc.Get(), 8), "alloc reset 4c") ||
            !require(invoke(c.guest_list.Get(), 139, c.guest_alloc.Get(), nullptr), "list reset 4c")) {
            VirtualFree(rel_mem, 0, MEM_RELEASE);
            VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 5;
        }
        IUnknown* heaps_arg[] = {vis_heap.guest.Get()};
        invoke<void>(c.guest_list.Get(), 32, 1u, heaps_arg);
        invoke<void>(c.guest_list.Get(), 33, guest_comp_root.Get());
        invoke<void>(c.guest_list.Get(), 25, guest_comp_pso.Get());
        invoke<void>(c.guest_list.Get(), 35, 0u, vis_heap.gpu + std::uint64_t{4} * c.increment);
        invoke<void>(c.guest_list.Get(), 35, 1u, vis_heap.gpu + std::uint64_t{5} * c.increment);
        invoke<void>(c.guest_list.Get(), 14, 1u, 1u, 1u);
        if (!require(invoke(c.guest_list.Get(), 9), "list close 4c") ||
            !require(invoke(c.guest_queue.Get(), 17, gate_fence.Get(), UINT64{1}), "queue wait gate")) {
            VirtualFree(rel_mem, 0, MEM_RELEASE);
            VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 5;
        }
        IUnknown* exec_lists[] = {c.guest_list.Get()};
        invoke<void>(c.guest_queue.Get(), 10, 1u, exec_lists);
        // Reset the recording while the submission stays in flight behind gate_fence (mirrors real frame reuse).
        if (!c.reset_copy_list()) {
            VirtualFree(rel_mem, 0, MEM_RELEASE);
            VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 5;
        }

        Snapshot mid_exec{};
        if (!x2w_d3d12_budget_snapshot(&mid_exec, sizeof(mid_exec))) return 4;
        if (mid_exec.retained[texture].objects != pre_rel.retained[texture].objects + 1) {
            std::printf("FAIL: dispatch_translated did not materialize released tex_rel (retained=%llu, expected=%llu)\n",
                        mid_exec.retained[texture].objects, pre_rel.retained[texture].objects + 1);
            VirtualFree(rel_mem, 0, MEM_RELEASE);
            VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 5;
        }
        // Overwrite vis_heap[4] while the dispatch is still in flight behind gate_fence.
        invoke<void>(guest.Get(), 18, static_cast<IUnknown*>(nullptr), &srv_desc,
                     vis_heap.cpu + std::uint64_t{4} * c.increment);
        for (int i = 0; i < 4; ++i) c.drain_once();
        Snapshot held_snap{};
        if (!x2w_d3d12_budget_snapshot(&held_snap, sizeof(held_snap))) return 4;
        if (retire_on && held_snap.retained[texture].objects < pre_rel.retained[texture].objects + 1) {
            std::puts("FAIL: tex_rel retired while dispatch_translated submission was still in flight");
            VirtualFree(rel_mem, 0, MEM_RELEASE);
            VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 5;
        }

        if (!require(invoke(gate_fence.Get(), 10, UINT64{1}), "signal gate fence") || !c.guest_sync()) {
            VirtualFree(rel_mem, 0, MEM_RELEASE);
            VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 5;
        }
        float out_texel[4]{};
        std::memcpy(out_texel, uav_mem, sizeof(out_texel));
        if (!verify_texel00(out_texel, 11, "dispatch_translated after Release()")) {
            VirtualFree(rel_mem, 0, MEM_RELEASE);
            VirtualFree(uav_mem, 0, MEM_RELEASE);
            std::filesystem::remove_all(cache_dir, ec);
            free_title_mem();
            return 5;
        }
        if (retire_on) {
            bool settled_4c = false;
            for (int attempt = 0; attempt < settle_limit; ++attempt) {
                c.drain_once();
                Snapshot after_gate{};
                if (x2w_d3d12_budget_snapshot(&after_gate, sizeof(after_gate)) &&
                    after_gate.retained[texture].objects == pre_rel.retained[texture].objects) {
                    settled_4c = true;
                    break;
                }
            }
            if (!settled_4c) {
                std::puts("FAIL: tex_rel did not retire after gate fence opened");
                VirtualFree(rel_mem, 0, MEM_RELEASE);
                VirtualFree(uav_mem, 0, MEM_RELEASE);
                std::filesystem::remove_all(cache_dir, ec);
                free_title_mem();
                return 5;
            }
        }
        uav_buf.Reset();
        VirtualFree(rel_mem, 0, MEM_RELEASE);
        VirtualFree(uav_mem, 0, MEM_RELEASE);
        SetEnvironmentVariableW(L"X2W_SHADER_CACHE", nullptr);
        std::filesystem::remove_all(cache_dir, ec);
    }

    // 5. Release all textures, drop copy-list holds & overwrite/drop SRVs, and verify clean retirement.
    for (UINT i = 0; i < tex_count; ++i) tex[i].Reset();
    if (!c.reset_copy_list()) {
        free_title_mem();
        return 5;
    }
    for (UINT i = 0; i < tex_count; ++i)
        invoke<void>(guest.Get(), 18, static_cast<IUnknown*>(nullptr), &srv_desc, cpu_heap.cpu + std::uint64_t{i} * c.increment);
    for (UINT i = 0; i < 2; ++i)
        invoke<void>(guest.Get(), 18, static_cast<IUnknown*>(nullptr), &srv_desc, vis_heap.cpu + std::uint64_t{i} * c.increment);
    if (!c.guest_sync()) {
        free_title_mem();
        return 5;
    }
    if (retire_on) {
        bool settled = false;
        for (int attempt = 0; attempt < settle_limit; ++attempt) {
            c.drain_once();
            if (!x2w_d3d12_budget_snapshot(&snap, sizeof(snap))) break;
            if (snap.live[texture].objects == base.live[texture].objects &&
                snap.retained[texture].objects == base.retained[texture].objects &&
                x2w_d3d12_lazy_texture_count() == 0) {
                settled = true;
                break;
            }
        }
        if (!settled) {
            std::printf("FAIL: retirement did not settle: live=%llu/%llu retained=%llu/%llu lazy_count=%u\n",
                        snap.live[texture].bytes, snap.live[texture].objects,
                        snap.retained[texture].bytes, snap.retained[texture].objects,
                        x2w_d3d12_lazy_texture_count());
            free_title_mem();
            return 5;
        }
    } else {
        if (!x2w_d3d12_budget_snapshot(&snap, sizeof(snap))) return 4;
        if (snap.retained[texture].objects != base.retained[texture].objects + 5) {
            std::printf("FAIL: no-retire expected 5 retained textures, got %llu\n", snap.retained[texture].objects);
            free_title_mem();
            return 5;
        }
    }

    free_title_mem();
    CloseHandle(c.event);
    const UINT64 debug_errors = debug ? debug->GetNumStoredMessages() : 0;
    std::printf("debug_errors_or_warnings=%llu\n", debug_errors);
    if (debug_errors || FAILED(device->GetDeviceRemovedReason())) return 4;
    std::printf("lazy textures check passed (PagefileUsage delta=%zu bytes, materialized 2/16, retired cleanly)\n",
                commit_after > commit_before ? commit_after - commit_before : 0);
    return 0;
}
