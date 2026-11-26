// Per-plane states for a depth stencil view with one read-only plane (src/startup_stubs/dsv_planes.h), on a real D3D12
// device with the debug layer: depth plane in DEPTH_READ, stencil plane in DEPTH_WRITE, the stencil plane cleared through a
// READ_ONLY_DEPTH view, copied out in COPY_SOURCE while the depth plane stays, and both planes returned to COMMON. A
// resource of two mips and three slices checks the subresource arithmetic. Any debug-layer error fails the run.
//   dsv_planes_check <warp|hardware>            the check
//   dsv_planes_check <warp|hardware> --mutate whole   moves the whole resource to DEPTH_READ instead: the stencil clear is
//                                               then invalid, the debug layer reports it and the run must fail
#include "startup_stubs/dsv_planes.h"

#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

void require(bool ok, const char* what)
{
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    std::exit(1);
}
void check(HRESULT hr, const char* what)
{
    if (SUCCEEDED(hr)) return;
    std::printf("FAIL: %s (0x%08lx)\n", what, static_cast<unsigned long>(hr));
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv)
{
    const bool warp = argc < 2 || std::strcmp(argv[1], "hardware") != 0;
    const bool mutate = argc >= 4 && std::strcmp(argv[2], "--mutate") == 0 && std::strcmp(argv[3], "whole") == 0;
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory4> factory;
    check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "DXGI factory");
    ComPtr<IDXGIAdapter> adapter;
    if (warp) check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "WARP adapter");
    ComPtr<ID3D12Device> device;
    const HRESULT created = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(created) && !warp) {
        std::puts("No hardware D3D12 device");
        return 77;
    }
    check(created, "D3D12 device");
    ComPtr<ID3D12InfoQueue> info;
    require(SUCCEEDED(device.As(&info)), "info queue (debug layer)");
    info->ClearStoredMessages();

    constexpr UINT width = 64, height = 64, mips = 2, slices = 3;
    D3D12_HEAP_PROPERTIES defaults{D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, slices, mips, DXGI_FORMAT_R24G8_TYPELESS, {1, 0},
                             D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
    D3D12_CLEAR_VALUE clear{};
    clear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    ComPtr<ID3D12Resource> depth;
    check(device->CreateCommittedResource(&defaults, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, &clear, IID_PPV_ARGS(&depth)),
          "depth stencil resource");

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "DSV heap");
    D3D12_DEPTH_STENCIL_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
    view.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
    view.Texture2DArray = {0, 0, slices};
    device->CreateDepthStencilView(depth.Get(), &view, heap->GetCPUDescriptorHandleForHeapStart());

    // Footprint of the stencil plane, mip 0, slice 0.
    x2w_dsv::Planes planes = x2w_dsv::planes_of(depth.Get());
    require(planes.mips == mips && planes.slices == slices, "the planes know the resource's mips and slices");
    const UINT stencil_sub = x2w_dsv::subresource(planes, 1, 0, 0);
    require(stencil_sub == mips * slices, "stencil plane, mip 0, slice 0 comes after every depth subresource");
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device->GetCopyableFootprints(&desc, stencil_sub, 1, 0, &footprint, &rows, &row_bytes, &total);
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    D3D12_RESOURCE_DESC buffer_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, total, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                    D3D12_RESOURCE_FLAG_NONE};
    ComPtr<ID3D12Resource> readback;
    check(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)),
          "readback buffer");

    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "queue");
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "allocator");
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "list");

    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    if (mutate) {
        // The whole resource in DEPTH_READ: the stencil plane is not writable, so the clear below is an error.
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {depth.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_READ};
        list->ResourceBarrier(1, &barrier);
        planes.state[0] = planes.state[1] = D3D12_RESOURCE_STATE_DEPTH_READ;
    } else {
        x2w_dsv::bind(list.Get(), planes, D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    }
    list->OMSetRenderTargets(0, nullptr, FALSE, &handle);
    list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_STENCIL, 0.0f, 0x5A, 0, nullptr);
    // Copy the cleared stencil plane out: only that plane changes state, the depth plane stays in DEPTH_READ.
    if (!mutate) x2w_dsv::move_plane(list.Get(), planes, 1, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    src.pResource = depth.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = stencil_sub;
    if (!mutate) list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (!mutate) require(planes.state[0] == D3D12_RESOURCE_STATE_DEPTH_READ && planes.state[1] == D3D12_RESOURCE_STATE_COPY_SOURCE, "the planes track their own states");
    x2w_dsv::release(list.Get(), planes);
    check(list->Close(), "close");
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "fence");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    check(queue->Signal(fence.Get(), 1), "signal");
    check(fence->SetEventOnCompletion(1, event), "event");
    require(WaitForSingleObject(event, 60000) == WAIT_OBJECT_0, "the queue finished");
    CloseHandle(event);

    const UINT64 errors = info->GetNumStoredMessagesAllowedByRetrievalFilter();
    UINT64 error_count = 0;
    for (UINT64 i = 0; i < errors; ++i) {
        SIZE_T length = 0;
        info->GetMessage(i, nullptr, &length);
        std::vector<char> bytes(length);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
        info->GetMessage(i, message, &length);
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
            ++error_count;
            if (error_count <= 3) std::printf("debug layer: %s\n", message->pDescription);
        }
    }
    if (error_count) {
        std::printf("FAIL: %llu debug-layer error(s)\n", static_cast<unsigned long long>(error_count));
        return 1;
    }
    if (mutate) {
        std::puts("FAIL: the mutated run reported no error");
        return 1;
    }
    D3D12_RANGE range{0, static_cast<SIZE_T>(total)};
    unsigned char* data = nullptr;
    check(readback->Map(0, &range, reinterpret_cast<void**>(&data)), "map readback");
    UINT bad = 0;
    for (UINT y = 0; y < rows; ++y)
        for (UINT x = 0; x < row_bytes; ++x)
            if (data[footprint.Footprint.RowPitch * y + x] != 0x5A) ++bad;
    readback->Unmap(0, nullptr);
    std::printf("stencil plane %llux%u: %u bytes differ from 0x5A\n", static_cast<unsigned long long>(row_bytes), rows, bad);
    require(bad == 0, "the stencil plane cleared through the READ_ONLY_DEPTH view reads back 0x5A");
    std::puts("dsv_planes: ok");
    return 0;
}
