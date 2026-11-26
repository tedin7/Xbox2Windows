// D3D12 readback after Evict / MakeResident (public ID3D12Device methods), the calls src/startup_stubs/d3d12_residency.inl makes.
// A committed texture with a known pattern is filled, evicted once its copy has completed, made resident again, copied to a
// readback buffer and compared with the CPU reference. `wrong-reference` compares against a changed reference and must fail.
// usage: startup_texture_residency_warp [warp|hardware] [wrong-reference]    exit 0 = match, 1 = mismatch, 77 = no adapter
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT W = 256, H = 256;  // 256 KiB of R8G8B8A8: large enough to be a real allocation

bool wait(ID3D12CommandQueue* q, ID3D12Fence* f, UINT64& value)
{
    if (FAILED(q->Signal(f, ++value))) return false;
    HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    f->SetEventOnCompletion(value, e);
    const DWORD r = WaitForSingleObject(e, 20000);
    CloseHandle(e);
    return r == WAIT_OBJECT_0;
}
}  // namespace

int main(int argc, char** argv)
{
    bool hardware = false, wrong = false;
    for (int i = 1; i < argc; ++i) {
        hardware |= !std::strcmp(argv[i], "hardware");
        wrong |= !std::strcmp(argv[i], "wrong-reference");
    }
    ComPtr<IDXGIFactory4> factory;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 77;
    ComPtr<IDXGIAdapter> adapter;
    if (!hardware) {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 77;
    } else if (FAILED(factory->EnumAdapters(0, &adapter))) {
        return 77;
    }
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) return 77;

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
        std::printf("FAIL: setup\n");
        return 1;
    }
    UINT64 value = 0;

    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = W; td.Height = H; td.DepthOrArraySize = 1; td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = W * H * 4; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
    bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES def{D3D12_HEAP_TYPE_DEFAULT}, up{D3D12_HEAP_TYPE_UPLOAD}, rb{D3D12_HEAP_TYPE_READBACK};
    ComPtr<ID3D12Resource> texture, upload, readback;
    if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture))) ||
        FAILED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload))) ||
        FAILED(device->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)))) {
        std::printf("FAIL: resources\n");
        return 1;
    }

    std::vector<unsigned> reference(W * H);
    for (UINT i = 0; i < W * H; ++i) reference[i] = 0xff000000u | ((i * 2654435761u) & 0x00ffffffu);
    void* mapped = nullptr;
    upload->Map(0, nullptr, &mapped);
    std::memcpy(mapped, reference.data(), W * H * 4);
    upload->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = texture.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = upload.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1, W * 4};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Transition = {texture.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    list->ResourceBarrier(1, &barrier);
    list->Close();
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    if (!wait(queue.Get(), fence.Get(), value)) { std::printf("FAIL: fill did not complete\n"); return 1; }

    // The fence rule: the copy above has completed, so nothing in flight names the texture. Evict, then bring it back.
    ID3D12Pageable* pageable = texture.Get();
    if (FAILED(device->Evict(1, &pageable))) { std::printf("FAIL: Evict\n"); return 1; }
    if (FAILED(device->MakeResident(1, &pageable))) { std::printf("FAIL: MakeResident\n"); return 1; }

    alloc->Reset();
    list->Reset(alloc.Get(), nullptr);
    dst.pResource = readback.Get(); dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1, W * 4};
    src.pResource = texture.Get(); src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex = 0;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    list->Close();
    queue->ExecuteCommandLists(1, lists);
    if (!wait(queue.Get(), fence.Get(), value)) { std::printf("FAIL: readback did not complete\n"); return 1; }

    if (wrong) reference[W * 17 + 5] ^= 0x00010101u;  // the control: one texel of the reference differs
    readback->Map(0, nullptr, &mapped);
    const unsigned mismatches = [&] {
        unsigned n = 0;
        const unsigned* got = static_cast<const unsigned*>(mapped);
        for (UINT i = 0; i < W * H; ++i) n += got[i] != reference[i];
        return n;
    }();
    readback->Unmap(0, nullptr);
    if (mismatches) {
        std::printf("FAIL: %u texels differ after Evict / MakeResident\n", mismatches);
        return 1;
    }
    std::printf("startup_texture_residency_warp: ok (%ux%u texels identical after Evict / MakeResident)\n", W, H);
    return 0;
}
