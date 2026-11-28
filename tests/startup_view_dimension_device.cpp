// Device slot 18 (CreateShaderResourceView) through d3d12_xs.dll with view descriptions whose ViewDimension is
// unknown or does not suit the resource (src/startup_stubs/view_dimension.h, docs/startup-runtime.md#unknown-and-
// mismatched-view-dimensions).
//
//   startup_view_dimension_device <warp|hardware> [off]
//
// The fixture reads the desktop descriptors the shim wrote: a compute shader on the fixture's own desktop queue
// samples texel (0,0) through a slot of a shader-visible guest heap, by the desktop heap behind it. A null
// descriptor reads zero.
//   (a) an SRV with ViewDimension 0 and a requested format that differs from the texture's (R32G32B32A32_FLOAT on an
//       R8G8B8A8_UNORM texture) gets the texture's default view: the slot samples the texture's distinct colour.
//   (b) an explicit TEXTURE3D SRV of the 2D texture is refused: the device is not removed.
//   (c) a slot that holds a valid view of a second texture (its colour is read first) receives a refused TEXTURE2D
//       SRV of a buffer. The device is not removed and the slot reads zero: the refusal leaves a null SRV, not the
//       old descriptor. A CopyDescriptorsSimple copy of the slot over another valid view reads zero too.
//   (d) a refused UAV replacement leaves the prior descriptor and its title token intact.
//
// `off` is the negative control and needs X2W_D3D12_VIEW_DEFAULTS=off: (a) must then fail, printing
// "FAIL: (a) default view reads 0.000 0.000 0.000 0.000 ... device not removed", and nothing else may fail.
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <string>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

extern "C" __declspec(dllimport) std::size_t x2w_d3d12_census_report(char* out, std::size_t capacity);

namespace {

constexpr UINT side = 4, pitch = 256, image_bytes = side * pitch;
constexpr DWORD wait_ms = 10000;
constexpr std::uint8_t colour_a[4] = {51, 153, 204, 255};  // 0.2 0.6 0.8 1.0
constexpr std::uint8_t colour_b[4] = {255, 64, 0, 128};

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
    if (SUCCEEDED(hr)) reference.As(&output);
    return output;
}

bool require(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) std::printf("FAIL: setup: %s hr=0x%08lx\n", stage, static_cast<unsigned long>(hr));
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

constexpr char probe_shader[] = R"(
Texture2D<float4> source : register(t0);
SamplerState point_sampler : register(s0);
RWByteAddressBuffer result : register(u0);
[numthreads(1, 1, 1)] void main()
{
    result.Store4(0, asuint(source.SampleLevel(point_sampler, float2(0.125, 0.125), 0)));
}
)";

constexpr char memory_buffer_probe[] = R"(
struct Record { uint4 first; uint4 rest[15]; };
StructuredBuffer<Record> source : register(t0);
float4 vs(uint id : SV_VertexID) : SV_Position
{
    return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0, 1);
}
float4 main() : SV_Target
{
    return float4(source[0].first.xy, source[0].rest[14].zw);
}
)";

// The same payload read through a typed buffer view of 16 elements of four words: element 0 and element 15 hold the words
// the structured probe reads.
constexpr char memory_buffer_typed_probe[] = R"(
Buffer<uint4> source : register(t0);
float4 vs(uint id : SV_VertexID) : SV_Position
{
    return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0, 1);
}
float4 main() : SV_Target
{
    const uint4 first = source[0];
    const uint4 last = source[15];
    return float4(first.x, first.y, last.z, last.w);
}
)";

struct Context {
    IUnknown* guest;
    ID3D12Device* device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    UINT64 value = 0;
    HANDLE event = nullptr;
    ComPtr<ID3D12Resource> upload, readback, probe_out, buffer_target;
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<IUnknown> heap;
    ComPtr<ID3D12DescriptorHeap> heap_real;
    UINT64 heap_cpu = 0;
    UINT increment = 0;

    bool run()
    {
        if (!require(list->Close(), "list close")) return false;
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (!require(queue->Signal(fence.Get(), ++value), "signal")) return false;
        if (fence->GetCompletedValue() >= value) return true;
        if (!require(fence->SetEventOnCompletion(value, event), "fence event")) return false;
        if (WaitForSingleObject(event, wait_ms) == WAIT_OBJECT_0) return true;
        std::puts("FAIL: setup: queue timed out");
        return false;
    }
    bool begin() { return require(allocator->Reset(), "allocator reset") && require(list->Reset(allocator.Get(), nullptr), "list reset"); }
    ComPtr<IUnknown> committed(const D3D12_RESOURCE_DESC& desc, const char* name)
    {
        const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC d = desc;
        const GUID iid = __uuidof(ID3D12Resource);
        ComPtr<IUnknown> out;
        if (!require(invoke(guest, 127, &props, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0u,
                            nullptr, &iid, out.GetAddressOf()), name))
            out.Reset();
        return out;
    }
    // Fills every texel of a guest 2D texture with one colour through its desktop object.
    bool fill(IUnknown* texture, const std::uint8_t (&colour)[4])
    {
        const auto real = desktop_object<ID3D12Resource>(guest, device, texture);
        if (!real) {
            std::puts("FAIL: setup: texture has no desktop object");
            return false;
        }
        void* mapped = nullptr;
        const D3D12_RANGE none{};
        if (!require(upload->Map(0, &none, &mapped), "upload map")) return false;
        auto* dst = static_cast<std::uint8_t*>(mapped);
        for (UINT y = 0; y < side; ++y)
            for (UINT x = 0; x < side; ++x) std::memcpy(dst + y * pitch + x * 4, colour, 4);
        upload->Unmap(0, nullptr);
        if (!begin()) return false;
        D3D12_TEXTURE_COPY_LOCATION to{real.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        to.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION from{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        from.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, side, side, 1, pitch};
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE, {}};
        b.Transition = {real.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COMMON};
        list->ResourceBarrier(1, &b);
        return run();
    }
    // Reads the desktop descriptor behind a guest slot through a one-pixel draw or a compute dispatch.
    bool read_slot(UINT index, float (&texel)[4])
    {
        if (!begin()) return false;
        ID3D12DescriptorHeap* bound = heap_real.Get();
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap_real->GetGPUDescriptorHandleForHeapStart();
        gpu.ptr += std::uint64_t{index} * increment;
        list->SetDescriptorHeaps(1, &bound);
        list->SetPipelineState(pso.Get());
        if (buffer_target) {
            const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
            const D3D12_VIEWPORT viewport{0, 0, 1, 1, 0, 1};
            const D3D12_RECT scissor{0, 0, 1, 1};
            list->SetGraphicsRootSignature(root.Get());
            list->SetGraphicsRootDescriptorTable(0, gpu);
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            const float clear[4]{-1, -1, -1, -1};
            list->ClearRenderTargetView(rtv, clear, 0, nullptr);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list->DrawInstanced(3, 1, 0, 0);
            D3D12_RESOURCE_BARRIER barrier{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE, {}};
            barrier.Transition = {buffer_target.Get(), 0, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE};
            list->ResourceBarrier(1, &barrier);
            D3D12_TEXTURE_COPY_LOCATION from{buffer_target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
            D3D12_TEXTURE_COPY_LOCATION to{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            to.PlacedFootprint.Footprint = {DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 1, 1, pitch};
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            list->ResourceBarrier(1, &barrier);
        } else {
            list->SetComputeRootSignature(root.Get());
            list->SetComputeRootDescriptorTable(0, gpu);
            list->SetComputeRootUnorderedAccessView(1, probe_out->GetGPUVirtualAddress());
            list->Dispatch(1, 1, 1);
            D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE, {}};
            b.Transition = {probe_out.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_SOURCE};
            list->ResourceBarrier(1, &b);
            list->CopyBufferRegion(readback.Get(), 0, probe_out.Get(), 0, 16);
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            list->ResourceBarrier(1, &b);
        }
        if (!run()) return false;
        void* mapped = nullptr;
        const D3D12_RANGE all{0, 16};
        if (!require(readback->Map(0, &all, &mapped), "readback map")) return false;
        std::memcpy(texel, mapped, sizeof(texel));
        readback->Unmap(0, nullptr);
        return true;
    }
    const char* device_state()
    {
        return SUCCEEDED(device->GetDeviceRemovedReason()) && SUCCEEDED(invoke(guest, 37)) ? "device not removed"
                                                                                          : "DEVICE REMOVED";
    }
    bool device_ok() { return SUCCEEDED(device->GetDeviceRemovedReason()) && SUCCEEDED(invoke(guest, 37)); }
    // want: the colour the slot must sample, or nullptr for null (all zero). *failed is set on a mismatch; false is
    // returned only when the probe itself could not run.
    bool expect(UINT index, const std::uint8_t* want, const char* what, bool* failed)
    {
        float texel[4]{};
        if (!read_slot(index, texel)) return false;
        bool match = true;
        for (int i = 0; i < 4; ++i) {
            const float expected = want ? want[i] / 255.0f : 0.0f;
            if (std::fabs(texel[i] - expected) > 0.5f / 255.0f) match = false;
        }
        if (match) {
            std::printf("%s: reads %.3f %.3f %.3f %.3f as expected\n", what, texel[0], texel[1], texel[2], texel[3]);
            return true;
        }
        std::printf("FAIL: %s reads %.3f %.3f %.3f %.3f, expected ", what, texel[0], texel[1], texel[2], texel[3]);
        if (want) std::printf("%.3f %.3f %.3f %.3f", want[0] / 255.0f, want[1] / 255.0f, want[2] / 255.0f, want[3] / 255.0f);
        else std::printf("null (all zero)");
        std::printf("; %s\n", device_state());
        *failed = true;
        return true;
    }
    UINT64 slot(UINT index) const { return heap_cpu + std::uint64_t{index} * increment; }
};

D3D12_SHADER_RESOURCE_VIEW_DESC texture2d_view()
{
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    return srv;
}

bool check_snapshot_counts(const char* expected)
{
    std::string report(x2w_d3d12_census_report(nullptr, 0) + 1, '\0');
    x2w_d3d12_census_report(report.data(), report.size());
    if (report.find(expected) != std::string::npos) return true;
    std::printf("FAIL: expected %s\n%s", expected, report.c_str());
    return false;
}

int check_memory_buffer(Context& c, bool off, bool mutant, bool recover, bool cache_first, bool typed)
{
    using Payload = std::array<std::uint32_t, 64>;
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    // The requested bytes are readable, but the surrounding 64 KiB are not committed.
    struct Reservation {
        void* base = VirtualAlloc(nullptr, 65536, MEM_RESERVE, PAGE_NOACCESS);
        ~Reservation() { if (base) VirtualFree(base, 0, MEM_RELEASE); }
    } reservation;
    if (!reservation.base || system.dwPageSize < sizeof(Payload) || system.dwPageSize >= 65536) return 4;
    auto* data = VirtualAlloc(static_cast<unsigned char*>(reservation.base) + system.dwPageSize,
                               system.dwPageSize, MEM_COMMIT, recover ? PAGE_NOACCESS : PAGE_READWRITE);
    if (!data) return 4;

    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, sizeof(Payload), 1, 1, 1,
                             DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    struct {
        D3D12_SHADER_RESOURCE_VIEW_DESC view;
        UINT64 address;
        UINT64 tail[2];
    } at{};
    static_assert(offsetof(decltype(at), address) == 0x28);
    at.view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    at.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    at.view.Buffer.NumElements = 1;
    at.view.Buffer.StructureByteStride = sizeof(Payload);
    if (typed) {
        at.view.Format = DXGI_FORMAT_R32G32B32A32_UINT;
        at.view.Buffer.NumElements = sizeof(Payload) / 16;
        at.view.Buffer.StructureByteStride = 0;
    }
    at.address = reinterpret_cast<UINT64>(data);

    D3D12_CPU_DESCRIPTOR_HANDLE cpu = c.heap_real->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < 3; ++i, cpu.ptr += c.increment)
        c.device->CreateShaderResourceView(nullptr, &at.view, cpu);

    Payload first{};
    first[0] = 17;
    first[1] = 63;
    first[62] = 141;
    first[63] = 201;
    Payload second = first;
    second[63] = 77;
    const auto reference = [](const Payload& bytes) {
        return std::array<float, 4>{float(bytes[0]), float(bytes[1]), float(bytes[62]), float(bytes[63])};
    };
    if (recover) {
        invoke<void>(c.guest, 70, &desc, &at, c.slot(0));
        // The first page stays reserved, so the cached witness remains valid after the protection change.
        DWORD old_protection = 0;
        if (!VirtualProtect(data, system.dwPageSize, PAGE_READWRITE, &old_protection)) return 4;
    }
    std::memcpy(data, first.data(), sizeof(first));
    invoke<void>(c.guest, 70, &desc, &at, c.slot(0));
    invoke<void>(c.guest, 70, &desc, &at, c.slot(1)); // unchanged generation
    if (!off) {
        const char* expected = cache_first
            ? "snapshot attempts 1 bound 0 bytes 0 not_readable 1 over_limit 0 allocation 0 view_does_not_fit 0 cached_skip 2 cached_rechecks 0;"
            : recover ? "snapshot attempts 2 bound 2 bytes 256 not_readable 1 over_limit 0 allocation 0 "
                        "view_does_not_fit 0 cached_skip 0 cached_rechecks 2;"
                      : "snapshot attempts 1 bound 2 bytes 256 not_readable 0";
        if (!check_snapshot_counts(expected)) return 4;
    }
    if (!recover) std::memcpy(data, second.data(), sizeof(second));
    if (!recover) invoke<void>(c.guest, 70, &desc, &at, c.slot(2));

    std::array<std::array<float, 4>, 3> readback{};
    for (UINT i = 0; i < (recover ? 2u : 3u); ++i) {
        float result[4]{};
        if (!c.read_slot(i, result) || !c.device_ok()) return 4;
        std::memcpy(readback[i].data(), result, sizeof(result));
        std::printf("slot70 slot %u readback %.0f %.0f %.0f %.0f\n", i, result[0], result[1], result[2], result[3]);
    }
    if (off) {
        if (readback != decltype(readback){}) return 1;
        std::puts("slot70 switch-off control: all three publications left the null SRVs intact");
        return 0;
    }
    if (readback[0] != reference(first) || readback[1] != reference(first)) {
        if (cache_first && readback == decltype(readback){})
            std::puts("slot70 cache-first mutant: readable view still samples null");
        return 1;
    }
    if (recover) return cache_first ? 2 : 0;
    if (readback[2] != reference(second)) {
        if (mutant && readback[2] == reference(first))
            std::puts("slot70 stale mutant: refreshed readback failed with the first generation");
        return 1;
    }
    return mutant ? 2 : 0;
}

// The regular lookup cannot serve these spans. Only the last span may be snapshotted.
int check_snapshot_diagnostics(Context& c, IUnknown* other_device, bool diagnostics_off, bool snapshot_off)
{
    static constexpr UINT stride = 256;  // Matches the structured-buffer probe.
    struct Reservation {
        void* base = VirtualAlloc(nullptr, 4 * 65536, MEM_RESERVE, PAGE_NOACCESS);
        ~Reservation() { if (base) VirtualFree(base, 0, MEM_RELEASE); }
    } reservation;
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    if (!reservation.base || system.dwPageSize < stride || system.dwPageSize >= 65536) return 4;
    auto* base = static_cast<unsigned char*>(reservation.base);
    const D3D12_RESOURCE_DESC request{D3D12_RESOURCE_DIMENSION_BUFFER, 0, stride, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    std::array<ComPtr<IUnknown>, 3> owners;
    struct {
        D3D12_SHADER_RESOURCE_VIEW_DESC view;
        UINT64 address;
        UINT64 tail[2];
    } at{};
    static_assert(offsetof(decltype(at), address) == 0x28);
    at.view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    at.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    at.view.Buffer.NumElements = 1;
    at.view.Buffer.StructureByteStride = stride;
    const std::array<float, 4> reference{23, 74, 119, 211};
    const char* const reasons[]{"snapshot_already_owned", "snapshot_placement_rejected", "snapshot_resource_overlap"};
    for (UINT i = 0; i < 4; ++i) {
        // Use a nonaligned placement address, so slot 133 uses a real committed default buffer.
        auto* data = base + i * 65536 + system.dwPageSize;
        if (!VirtualAlloc(data, system.dwPageSize, MEM_COMMIT, PAGE_READWRITE)) return 4;
        std::array<UINT, stride / sizeof(UINT)> words{};
        words[0] = static_cast<UINT>(reference[0]);
        words[1] = static_cast<UINT>(reference[1]);
        words[62] = static_cast<UINT>(reference[2]);
        words[63] = static_cast<UINT>(reference[3]);
        std::memcpy(data, words.data(), stride);
        if (i < owners.size()) {
            auto placed = request;
            // The guest DENY flag stays in the placement record. Slot 133 strips it from native buffers.
            if (i == 0) placed.Flags = D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
            if (i == 1) placed.Width = stride / 2;
            if (!require(invoke(i == 2 ? other_device : c.guest, 133, data, &placed,
                                D3D12_RESOURCE_STATE_COMMON, nullptr, &__uuidof(ID3D12Resource),
                                owners[i].GetAddressOf()), "snapshot diagnostic placement"))
                return 4;
        }
        at.address = reinterpret_cast<UINT64>(data);
        auto cpu = c.heap_real->GetCPUDescriptorHandleForHeapStart();
        cpu.ptr += i * c.increment;
        c.device->CreateShaderResourceView(nullptr, &at.view, cpu);
        invoke<void>(c.guest, 70, &request, &at, c.slot(i));
        if (!diagnostics_off && !snapshot_off && i < owners.size()) {
            const std::string total = "memory SRV diagnostics (slot 70): total " + std::to_string(i + 1) + ";";
            const std::string shape = std::string("reason=") + reasons[i] +
                " dim=1 fmt=0 size=256x1x1 mips=1 layout=0x1 flags=0x0 calls=1";
            const std::string count = std::string(reasons[i]) + " 1";
            if (!check_snapshot_counts(total.c_str()) || !check_snapshot_counts(count.c_str()) ||
                !check_snapshot_counts(shape.c_str())) return 4;
        }
    }
    if (diagnostics_off || snapshot_off) {
        std::string report(x2w_d3d12_census_report(nullptr, 0) + 1, '\0');
        x2w_d3d12_census_report(report.data(), report.size());
        if (report.find("memory SRV diagnostics") != std::string::npos) return 4;
    }
    if (!diagnostics_off && !snapshot_off && !check_snapshot_counts("memory SRV diagnostics (slot 70): total 3;"))
        return 4;
    if (!snapshot_off && !check_snapshot_counts("snapshot attempts 4 bound 1 bytes 256")) return 4;
    for (UINT i = 0; i < 4; ++i) {
        float texel[4]{};
        if (!c.read_slot(i, texel) || !c.device_ok()) return 4;
        const std::array<float, 4> actual{texel[0], texel[1], texel[2], texel[3]};
        const auto expected = i < 3 || snapshot_off ? std::array<float, 4>{} : reference;
        if (actual != expected) {
            std::printf("FAIL: diagnostic slot %u readback %.0f %.0f %.0f %.0f\n",
                        i, actual[0], actual[1], actual[2], actual[3]);
            return 1;
        }
        if (snapshot_off && i == 3 && actual != reference)
            std::puts("slot70 snapshot off control: fitting span failed the enabled readback oracle");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    const bool diagnostics_off = argc == 3 && std::strcmp(argv[2], "memory-buffer-diagnostics-off") == 0;
    const bool snapshot_off = argc == 3 && std::strcmp(argv[2], "memory-buffer-diagnostics-no-snapshot") == 0;
    const bool diagnostics = diagnostics_off || snapshot_off ||
        (argc == 3 && std::strcmp(argv[2], "memory-buffer-diagnostics") == 0);
    const bool memory_off = argc == 3 && std::strcmp(argv[2], "memory-buffer-off") == 0;
    const bool memory_mutant = argc == 3 && std::strcmp(argv[2], "memory-buffer-stale") == 0;
    const bool cache_first = argc == 3 && std::strcmp(argv[2], "memory-buffer-cache-first") == 0;
    const bool typed = argc == 3 && std::strcmp(argv[2], "memory-buffer-typed") == 0;
    const bool recover = cache_first || (argc == 3 && std::strcmp(argv[2], "memory-buffer-recover") == 0);
    const bool memory_buffer = diagnostics || recover || memory_off || memory_mutant || typed ||
        (argc == 3 && std::strcmp(argv[2], "memory-buffer") == 0);
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (argc == 3 && !memory_buffer && std::strcmp(argv[2], "off")))
        return 2;
    const bool off = argc == 3 && !memory_buffer;
    char mode[8]{};
    const DWORD mode_n = GetEnvironmentVariableA("X2W_D3D12_VIEW_DEFAULTS", mode, sizeof(mode));
    const bool env_off = mode_n == 3 && std::strcmp(mode, "off") == 0;
    if (env_off != off || (!off && mode_n)) {
        std::puts("FAIL: setup: X2W_D3D12_VIEW_DEFAULTS must be unset, or off with the `off` argument");
        return 2;
    }
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), std::strcmp(argv[1], "warp") == 0);
    if (!adapter) return 77;
    ComPtr<IUnknown> guest;
    ComPtr<ID3D12Device> device;
    std::array<unsigned char, 64> parameters{};
    if (!require(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                      reinterpret_cast<void**>(guest.GetAddressOf())), "Xbox device") ||
        !require(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "desktop device"))
        return 3;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    adapter->GetDesc1(&adapter_desc);
    std::printf("adapter=%ls debug_layer=off view_defaults=%s\n", adapter_desc.Description, off ? "off" : "unset (srv)");

    Context c{guest.Get(), device.Get()};
    c.increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    c.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const D3D12_COMMAND_QUEUE_DESC direct{};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD}, readback_heap{D3D12_HEAP_TYPE_READBACK},
        default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, image_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                     D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    D3D12_RESOURCE_DESC out_desc = buffer;
    out_desc.Width = 256;
    out_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (!c.event || !require(device->CreateCommandQueue(&direct, IID_PPV_ARGS(&c.queue)), "queue") ||
        !require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&c.allocator)), "allocator") ||
        !require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, c.allocator.Get(), nullptr, IID_PPV_ARGS(&c.list)), "list") ||
        !require(c.list->Close(), "list close") ||
        !require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c.fence)), "fence") ||
        !require(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                 nullptr, IID_PPV_ARGS(&c.upload)), "upload") ||
        !require(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                                 nullptr, IID_PPV_ARGS(&c.readback)), "readback") ||
        !require(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &out_desc,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&c.probe_out)), "probe output"))
        return 4;

    // The probe: samples (0,0) through descriptor table slot 0 with a point sampler.
    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER root_parameters[2]{};
    root_parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    root_parameters[0].DescriptorTable = {1, &range};
    root_parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    root_parameters[1].Descriptor = {0, 0};
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    const D3D12_ROOT_SIGNATURE_DESC root_desc{memory_buffer ? 1u : 2u, root_parameters, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const char* shader = typed ? memory_buffer_typed_probe : memory_buffer ? memory_buffer_probe : probe_shader;
    ComPtr<ID3DBlob> root_blob, errors, code;
    if (!require(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, root_blob.GetAddressOf(),
                                            errors.GetAddressOf()), "root serialize") ||
        !require(device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(), IID_PPV_ARGS(&c.root)),
                 "root signature") ||
        !require(D3DCompile(shader, std::strlen(shader), nullptr, nullptr, nullptr, "main", memory_buffer ? "ps_5_1" : "cs_5_1", 0, 0,
                            code.GetAddressOf(), errors.ReleaseAndGetAddressOf()), "probe shader"))
        return 4;
    if (memory_buffer) {
        ComPtr<ID3DBlob> vertex;
        if (!require(D3DCompile(shader, std::strlen(shader), nullptr, nullptr, nullptr, "vs", "vs_5_1", 0, 0,
                                vertex.GetAddressOf(), errors.ReleaseAndGetAddressOf()), "snapshot vertex shader"))
            return 4;
        constexpr auto format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = c.root.Get();
        pso.VS = {vertex->GetBufferPointer(), vertex->GetBufferSize()};
        pso.PS = {code->GetBufferPointer(), code->GetBufferSize()};
        pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso.SampleMask = UINT_MAX;
        pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso.RasterizerState.DepthClipEnable = TRUE;
        pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso.NumRenderTargets = 1;
        pso.RTVFormats[0] = format;
        pso.SampleDesc.Count = 1;
        const D3D12_RESOURCE_DESC target{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, 1, 1, format, {1, 0},
                                         D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
        const D3D12_DESCRIPTOR_HEAP_DESC rtv{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
        if (!require(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&c.pso)), "snapshot pipeline") ||
            !require(device->CreateDescriptorHeap(&rtv, IID_PPV_ARGS(&c.rtv_heap)), "snapshot RTV heap") ||
            !require(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &target,
                                                     D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                     IID_PPV_ARGS(&c.buffer_target)), "snapshot render target"))
            return 4;
        device->CreateRenderTargetView(c.buffer_target.Get(), nullptr, c.rtv_heap->GetCPUDescriptorHandleForHeapStart());
    } else {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = c.root.Get();
        pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        if (!require(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&c.pso)), "probe pipeline")) return 4;
    }

    // A shader-visible guest heap of four SRV slots.
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } request{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    if (!require(invoke(c.guest, 14, &request, &__uuidof(ID3D12DescriptorHeap), c.heap.GetAddressOf()), "guest heap")) return 4;
    c.heap_real = desktop_object<ID3D12DescriptorHeap>(c.guest, device.Get(), c.heap.Get());
    if (!c.heap_real) {
        std::puts("FAIL: setup: guest heap has no desktop object");
        return 4;
    }
    c.heap_cpu = invoke<UINT64>(c.heap.Get(), 9);
    if (diagnostics) {
        ComPtr<IUnknown> other_device;
        if (!require(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                          reinterpret_cast<void**>(other_device.GetAddressOf())), "second Xbox device") ||
            other_device.Get() == guest.Get()) return 4;
        return check_snapshot_diagnostics(c, other_device.Get(), diagnostics_off, snapshot_off);
    }
    if (memory_buffer) return check_memory_buffer(c, memory_off, memory_mutant, recover, cache_first, typed);

    const D3D12_RESOURCE_DESC texture_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, side, side, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                           {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    D3D12_RESOURCE_DESC small_buffer = buffer;
    small_buffer.Width = 256;
    const auto texture_a = c.committed(texture_desc, "texture A");
    const auto texture_b = c.committed(texture_desc, "texture B");
    const auto buffer_c = c.committed(small_buffer, "buffer");
    if (!texture_a || !texture_b || !buffer_c || !c.fill(texture_a.Get(), colour_a) || !c.fill(texture_b.Get(), colour_b))
        return 4;
    if (!c.device_ok()) {
        std::puts("FAIL: setup: device removed before the cases");
        return 4;
    }

    bool failed = false;
    // (a) ViewDimension 0 with another format: the texture's default view.
    D3D12_SHADER_RESOURCE_VIEW_DESC unknown{};
    unknown.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    unknown.ViewDimension = D3D12_SRV_DIMENSION_UNKNOWN;
    unknown.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    invoke<void>(c.guest, 18, texture_a.Get(), &unknown, c.slot(0));
    if (!c.device_ok()) {
        std::puts("FAIL: (a) the unknown-dimension SRV removed the device");
        return 5;
    }
    if (!c.expect(0, colour_a, "(a) default view", &failed)) return 4;

    // (b) an explicit TEXTURE3D SRV of the 2D texture: refused, the device stays.
    D3D12_SHADER_RESOURCE_VIEW_DESC volume = texture2d_view();
    volume.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    volume.Texture3D = {0, 1, 0.0f};
    invoke<void>(c.guest, 18, texture_a.Get(), &volume, c.slot(1));
    if (!c.device_ok()) {
        std::puts("FAIL: (b) the incompatible explicit SRV was forwarded and removed the device");
        return 5;
    }
    std::puts("(b) TEXTURE3D SRV of a 2D texture: refused, device not removed");

    // (c) a refused replacement over a valid view, and a copy of the slot over another valid view.
    const auto valid = texture2d_view();
    invoke<void>(c.guest, 18, texture_b.Get(), &valid, c.slot(2));
    invoke<void>(c.guest, 18, texture_b.Get(), &valid, c.slot(3));
    bool setup_failed = false;
    if (!c.expect(2, colour_b, "(c) valid view before the replacement", &setup_failed) ||
        !c.expect(3, colour_b, "(c) valid copy target before the copy", &setup_failed))
        return 4;
    if (setup_failed) {
        std::puts("FAIL: setup: valid views do not read their texture");
        return 4;
    }
    invoke<void>(c.guest, 18, buffer_c.Get(), &valid, c.slot(2));  // TEXTURE2D of a buffer
    if (!c.device_ok()) {
        std::puts("FAIL: (c) the refused replacement removed the device");
        return 5;
    }
    if (!c.expect(2, nullptr, "(c) slot after the refused replacement", &failed)) return 4;
    invoke<void>(c.guest, 24, 1u, c.slot(3), c.slot(2), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (!c.expect(3, nullptr, "(c) copy of the refused slot", &failed)) return 4;

    auto uav_desc = texture_desc;
    uav_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto uav_texture = c.committed(uav_desc, "UAV texture");
    if (!uav_texture) return 4;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    const auto uav_slot = c.slot(1);
    invoke<void>(c.guest, 19, uav_texture.Get(), static_cast<IUnknown*>(nullptr), &uav, uav_slot);
    if (!c.device_ok()) return 5;
    std::array<std::uint32_t, 8> before{}, after{};
    std::memcpy(before.data(), reinterpret_cast<const void*>(uav_slot), sizeof(before));
    if (!before[0] || before[2] != 19) {
        std::puts("FAIL: (d) no valid UAV token before the replacement");
        return 4;
    }
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
    uav.Texture3D = {0, 0, 1};
    invoke<void>(c.guest, 19, uav_texture.Get(), static_cast<IUnknown*>(nullptr), &uav, uav_slot);
    std::memcpy(after.data(), reinterpret_cast<const void*>(uav_slot), sizeof(after));
    if (after != before) {
        std::puts("FAIL: (d) refused UAV replacement changed the live descriptor token");
        failed = true;
    } else {
        std::puts("(d) refused UAV replacement kept the live descriptor token");
    }

    if (!c.device_ok()) {
        std::puts("FAIL: device removed after the cases");
        return 5;
    }
    std::printf("cases complete, %s\n", failed ? "with failures" : "all passed");
    if (failed) return 1;
    std::puts("startup_view_dimension_device: default SRV samples its texture; refused SRVs leave null descriptors; "
              "device not removed");
    return 0;
}
