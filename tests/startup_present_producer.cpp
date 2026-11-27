#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdarg>
#include <iterator>
#include <memory>
#include <type_traits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "startup_stubs/red_producer_tree.h"

bool producer_errors(ID3D12Device* device)
{
    Microsoft::WRL::ComPtr<ID3D12InfoQueue> messages;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&messages)))) return false;
    for (UINT64 i = 0; i < messages->GetNumStoredMessages(); ++i) {
        SIZE_T size = 0;
        if (FAILED(messages->GetMessage(i, nullptr, &size))) return true;
        std::vector<char> storage(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(messages->GetMessage(i, message, &size))) return true;
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
            std::fprintf(stderr, "D3D12 error %u: %s\n", message->ID, message->pDescription);
            return true;
        }
    }
    return false;
}

#include "startup_present_producer_tree.inl"

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Microsoft::WRL::ComPtr;
using Word = std::uint64_t;
template<class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
template<class Fn> Fn slot(void* object, unsigned index) { return reinterpret_cast<Fn>((*static_cast<void***>(object))[index]); }
Word call(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word);
    return slot<Slot>(object, index)(object, a, b, c, d, e);
}
bool ok(Word result) { return SUCCEEDED(static_cast<HRESULT>(result)); }
}

int main(int argc, char** argv)
{
    const bool reverse = argc >= 2 && std::strcmp(argv[1], "reverse") == 0;
    const bool use_magenta = argc >= 2 && std::strcmp(argv[1], "auto-magenta") == 0;
    const bool default_cadence = argc >= 2 && std::strcmp(argv[1], "auto-default") == 0;
    const bool flat_onset = argc >= 2 && std::strcmp(argv[1], "auto-onset") == 0;
    const bool automatic = use_magenta || default_cadence || flat_onset || (argc >= 2 && std::strcmp(argv[1], "auto") == 0);
    const bool barriers = argc >= 2 && std::strcmp(argv[1], "barriers") == 0;
    const bool hardware = argc == 3 && std::strcmp(argv[2], "hardware") == 0;
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    if (hardware) {
        for (UINT i = 0; ; ++i) {
            const HRESULT result = factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
            if (result == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(result)) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(adapter->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) break;
            adapter.Reset();
        }
    } else if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 77;
    }
    DXGI_ADAPTER_DESC1 adapter_desc{};
    if (FAILED(adapter->GetDesc1(&adapter_desc))) return 3;
    std::printf("adapter=%ls vendor=0x%x device=0x%x debug=%d\n", adapter_desc.Description, adapter_desc.VendorId,
                adapter_desc.DeviceId, debug ? 1 : 0);
    if (argc >= 2 && std::strncmp(argv[1], "tree-", 5) == 0) {
        if (std::strcmp(argv[1], "tree-baseline") == 0)
            return producer_tree_fixture::run<false>(adapter.Get(), argv[1]);
        return producer_tree_fixture::run<true>(adapter.Get(), argv[1]);
    }
    void *device = nullptr, *queue = nullptr, *heap = nullptr, *target = nullptr, *allocator = nullptr, *list = nullptr, *fence = nullptr;
    const std::array<unsigned char, 64> init{};
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), init.data(), __uuidof(ID3D12Device), &device))) return 4;
    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 4;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (!ok(call(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&queue))) ||
        !ok(call(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(call(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))) ||
        !ok(call(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&fence)))) return 5;
    const D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
    constexpr auto format = DXGI_FORMAT_R8G8B8A8_UNORM;
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 16, 16, 1, 1, format, {1, 0},
                                   D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    using Create = HRESULT (WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS, const D3D12_RESOURCE_DESC*,
                                     D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID, void**);
    if (FAILED(slot<Create>(device, 27)(device, &properties, D3D12_HEAP_FLAG_NONE, &desc,
                                       D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource), &target))) return 6;
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } rtv_heap{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    if (!ok(call(device, 14, arg(&rtv_heap), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&heap)))) return 7;
    const Word rtv = call(heap, 9);
    const D3D12_RENDER_TARGET_VIEW_DESC view{format, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    call(device, 20, arg(target), arg(&view), rtv);

    Word fence_value = 0;
    const auto finish = [&] {
        if (!ok(call(queue, 14, arg(fence), ++fence_value))) return false;
        const auto deadline = GetTickCount64() + 10000;
        while (call(fence, 8) < fence_value) {
            if (GetTickCount64() >= deadline) return false;
            Sleep(1);
        }
        return true;
    };
    const auto present = [&] {
        const Word plane[10]{0, 1, arg(&target)};
        const Word parameters[8]{};
        return ok(call(queue, 32, 1, arg(plane), arg(parameters))) && finish();
    };
    constexpr float red[]{1, 0, 0, 1}, near_red[]{254.0f / 255.0f, 0, 0, 1}, green[]{0, 1, 0, 1}, blue[]{0, 0, 1, 1};
    constexpr float magenta[]{1, 0, 1, 1}, near_magenta[]{254.0f / 255.0f, 0, 1, 1};
    const float* chosen = use_magenta ? magenta : red;
    const auto emit = [&](const float* first, const float* last = nullptr, const D3D12_RECT* rect = nullptr) {
        if (!ok(call(list, 139, arg(allocator)))) return false;
        call(list, 52, rtv, arg(first));
        if (last) call(list, 52, rtv, arg(last), rect ? 1 : 0, arg(rect));
        if (!ok(call(list, 9))) return false;
        void* lists[]{list};
        call(queue, 10, 1, arg(lists));
        return finish();
    };
    if (barriers) {
        D3D12_RESOURCE_DESC uav_desc = desc;
        uav_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        void* uav = nullptr;
        if (FAILED(slot<Create>(device, 27)(device, &properties, D3D12_HEAP_FLAG_NONE, &uav_desc,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, __uuidof(ID3D12Resource), &uav)) ||
            !uav || !ok(call(list, 139, arg(allocator)))) return 15;
        D3D12_RESOURCE_BARRIER records[7]{};
        records[0].Type = records[2].Type = records[3].Type = records[4].Type = records[5].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        records[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        records[2].UAV.pResource = static_cast<ID3D12Resource*>(uav);
        records[3].UAV.pResource = static_cast<ID3D12Resource*>(target); // lacks UAV access
        records[4].UAV.pResource = reinterpret_cast<ID3D12Resource*>(1); // unrecognized wrapper
        records[5].Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
        records[6].Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        if (call(list, 26, std::size(records), arg(records)) != 0 || call(list, 26, 1, 1) != 0 ||
            !ok(call(list, 9)) || call(list, 26, 1, arg(records)) != 0) return 16;
        void* lists[]{list};
        call(queue, 10, 1, arg(lists));
        if (!finish()) return 17;
    } else if (!emit(automatic && !default_cadence && !flat_onset ? chosen : blue) || !present()) {
        return 8; // discover the title's present surface
    } else if (flat_onset) {
        // #4 has a near-red center but a blue corner. #5 is uniformly near red.
        const D3D12_RECT corner{0, 0, 4, 4};
        if (!present() || !emit(green) || !present() || !emit(near_red, blue, &corner) || !present() ||
            !emit(near_red) || !present() || !present() || !emit(green, near_red) || !present() || !present()) return 18;
    } else if (automatic) {
        // #1 matches too early; #3 changes color, #4 almost matches and #5 matches again.
        // The delayed sample at #6 arms #7 and #8, then a blue write at #9 is out of range.
        if (!present() || !emit(use_magenta ? red : green) || !present() || !emit(use_magenta ? near_magenta : near_red) || !present() ||
            !emit(chosen) || !present() || !present() || !emit(green, chosen) || !present() || !present() ||
            !emit(blue) || !present() || !present()) return 14;
    } else {
        if (!ok(call(list, 139, arg(allocator)))) return 9;
        call(list, 52, rtv, arg(reverse ? red : green));
        call(list, 52, rtv, arg(reverse ? green : red));
        if (!ok(call(list, 9))) return 10;
        void* lists[]{list};
        for (unsigned i = 0; i < 2; ++i) {
            call(queue, 10, 1, arg(lists));
            if (!finish() || !present() || !present()) return 11;
        }
    }
    if (FAILED(desktop->GetDeviceRemovedReason())) return 12;
    return debug && producer_errors(desktop.Get()) ? 13 : 0;
}
