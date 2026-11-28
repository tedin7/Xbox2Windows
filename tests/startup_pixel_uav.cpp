// X2W_TEST_PIXEL_UAV=<case> (tests/startup_pixel_uav_check.py): a translated pixel program that stores to a
// 2D image (#1996 item F) drawn through d3d12_xs.dll, with the image bound from the title's descriptor table.
//
// Four committed 4x4 textures with two mips each, every word of every texel a distinct sentinel:
//   A R32G32B32A32_FLOAT with ALLOW_UNORDERED_ACCESS, B the same, C without the flag, D R32G32B32A32_TYPELESS.
// One shader-visible heap of nine entries:
//   0 UAV of A (explicit 2D view, mip 0)   1 UAV of B (no description: the default view)
//   2 UAV of C (the stand-in skips the desktop call and keeps the title's view)
//   3 UAV of D as R32G32B32A32_UINT        4 left unwritten
//   5 SRV of A                             6 UAV of A, mip 1
//   7 UAV of A as a one-slice 2D array     8 UAV of E, a 4x4 R8G8B8A8_UNORM render target with the flag
// The case names the table bound at root parameter 0 (and, for srv-and-uav, parameter 1); one draw follows.
// The target case binds E as the draw's render target first.
// After the fence every subresource of the four textures is written to <pixels>.uav, and the debug layer's
// errors are counted when it is on. The rw cases use 8x8 textures and two draws that add a colour to each texel.
// The stale cases read an untouched copy from entry 1.
// The atomic cases use one word per texel in R32_UINT, R32_FLOAT and R32_SINT views.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using Word = std::uint64_t;
Word invoke(void*, unsigned, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0, Word = 0);

namespace {
using Microsoft::WRL::ComPtr;
template<class T> Word ptr(T* value) { return reinterpret_cast<Word>(value); }

constexpr UINT side_default = 4, mips = 2, textures = 4, array_slices = 6;

// The sentinel of word c of texel (x, y) of mip m of texture t; tests/startup_pixel_uav_check.py makes the same.
std::uint32_t sentinel(UINT t, UINT m, UINT texel, UINT c) { return 0xc0000000u | (t << 12) | (m << 8) | (texel << 2) | c; }
std::uint32_t array_sentinel(UINT s, UINT m, UINT texel, UINT c) { return 0xd0000000u | (s << 12) | (m << 8) | (texel << 2) | c; }

struct Fixture {
    ID3D12Device* desktop = nullptr;
    bool array_mode = false, raw_mode = false;
    bool atomic_mode = false;
    bool read_write = false;
    UINT side = side_default;
    void* resources[textures]{};
    ComPtr<ID3D12Resource> natives[textures];
    void* array_resources[3]{};
    ComPtr<ID3D12Resource> array_natives[3];
} fixture;

ComPtr<ID3D12Resource> native_of(void* guest, void* wrapped)
{
    // The stand-in forwards SetPrivateDataInterface with the desktop object behind its argument.
    constexpr GUID key{0x5e1f0b42, 0x2d8c, 0x4f61, {0x9a, 0x37, 0x0c, 0x71, 0x52, 0x88, 0xe4, 0x1d}};
    if (FAILED(static_cast<HRESULT>(invoke(guest, 5, ptr(&key), ptr(wrapped))))) return nullptr;
    ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT read = fixture.desktop->GetPrivateData(key, &bytes, stored.GetAddressOf());
    invoke(guest, 5, ptr(&key), 0);
    ComPtr<ID3D12Resource> result;
    if (FAILED(read) || FAILED(stored.As(&result))) return nullptr;
    return result;
}

// The destination and the stale copy start with the same exact binary32 values.
std::uint32_t pattern(UINT layer, UINT texel, UINT channel)
{
    const float value = float(1 + layer * 256 + texel * 4 + channel) / 128.0f;
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void draw_twice(void* list)
{
    static constexpr float colour[] = {0.25f, 0.5f, -0.125f, 0.0625f};
    invoke(list, 40, 1, 4, ptr(colour), 0);
    invoke(list, 12, 3, 1);
    invoke(list, 12, 3, 1);
}

D3D12_RESOURCE_DESC texture_desc(UINT t)
{
    if (fixture.atomic_mode) {
        constexpr DXGI_FORMAT formats[]{DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_SINT,
                                        DXGI_FORMAT_R32_TYPELESS};
        return {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, fixture.side, fixture.side, 1, mips,
                formats[t], {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    }
    return {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, fixture.side, fixture.side, 1, mips,
            t == 3 ? DXGI_FORMAT_R32G32B32A32_TYPELESS : DXGI_FORMAT_R32G32B32A32_FLOAT, {1, 0},
            D3D12_TEXTURE_LAYOUT_UNKNOWN, t == 2 ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
}

D3D12_RESOURCE_DESC array_texture_desc(UINT t)
{
    return {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, fixture.side, fixture.side, array_slices, mips,
            t == 2 ? DXGI_FORMAT_R32G32B32A32_TYPELESS : DXGI_FORMAT_R32G32B32A32_FLOAT, {1, 0},
            D3D12_TEXTURE_LAYOUT_UNKNOWN, t == 1 ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
}

struct Copier {
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    bool open(ID3D12Device* device)
    {
        const D3D12_COMMAND_QUEUE_DESC desc{};
        return SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))) &&
               SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) &&
               SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list))) &&
               SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    }
    bool run()
    {
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence.Get(), 1))) return false;
        const ULONGLONG deadline = GetTickCount64() + 10000;
        while (fence->GetCompletedValue() < 1 && GetTickCount64() < deadline) Sleep(1);
        return fence->GetCompletedValue() == 1;
    }
};

void barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &b);
}

ComPtr<ID3D12Resource> buffer(ID3D12Device* device, D3D12_HEAP_TYPE type, UINT64 bytes, D3D12_RESOURCE_STATES state)
{
    const D3D12_HEAP_PROPERTIES heap{type};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                   D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    ComPtr<ID3D12Resource> result;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&result))))
        return nullptr;
    return result;
}

// Fills every subresource of the four textures with its sentinels (to_textures) or copies them all back.
bool copy_all(ID3D12Device* device, bool to_textures, std::vector<std::uint32_t>* words)
{
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[textures][mips]{};
    UINT64 total = 0;
    for (UINT t = 0; t < textures; ++t) {
        const auto desc = texture_desc(t);
        UINT64 bytes = 0;
        device->GetCopyableFootprints(&desc, 0, mips, 0, footprints[t], nullptr, nullptr, &bytes);
        if (!bytes || bytes == ~UINT64{0}) return false;
        if (bytes > total) total = bytes;
    }
    const UINT channels = fixture.atomic_mode ? 1 : 4;
    constexpr UINT64 alignment = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    const UINT64 stride = (total + alignment - 1) & ~(alignment - 1);
    auto staging = buffer(device, to_textures ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK, stride * textures,
                          to_textures ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST);
    Copier copier;
    if (!staging || !copier.open(device)) return false;
    unsigned char* memory = nullptr;
    if (to_textures) {
        if (FAILED(staging->Map(0, nullptr, reinterpret_cast<void**>(&memory)))) return false;
        for (UINT t = 0; t < textures; ++t)
            for (UINT m = 0; m < mips; ++m) {
                const UINT n = fixture.side >> m;
                for (UINT y = 0; y < n; ++y)
                    for (UINT x = 0; x < n; ++x)
                        for (UINT c = 0; c < channels; ++c) {
                            const std::uint32_t value = fixture.read_write && t < 2 && m == 0
                                ? pattern(0, y * n + x, c) : sentinel(t, m, y * n + x, c);
                            const auto& f = footprints[t][m];
                            std::memcpy(memory + t * stride + f.Offset + y * f.Footprint.RowPitch + (x * channels + c) * 4,
                                        &value, 4);
                        }
            }
        staging->Unmap(0, nullptr);
    }
    for (UINT t = 0; t < textures; ++t) {
        auto* texture = fixture.natives[t].Get();
        const auto state = to_textures ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier(copier.list.Get(), texture, D3D12_RESOURCE_STATE_COMMON, state);
        for (UINT m = 0; m < mips; ++m) {
            D3D12_TEXTURE_COPY_LOCATION placed{staging.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            placed.PlacedFootprint = footprints[t][m];
            placed.PlacedFootprint.Offset += t * stride;
            D3D12_TEXTURE_COPY_LOCATION subresource{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            subresource.SubresourceIndex = m;
            if (to_textures) copier.list->CopyTextureRegion(&subresource, 0, 0, 0, &placed, nullptr);
            else copier.list->CopyTextureRegion(&placed, 0, 0, 0, &subresource, nullptr);
        }
        barrier(copier.list.Get(), texture, state, D3D12_RESOURCE_STATE_COMMON);
    }
    if (!copier.run()) return false;
    if (to_textures) return true;
    if (FAILED(staging->Map(0, nullptr, reinterpret_cast<void**>(&memory)))) return false;
    for (UINT t = 0; t < textures; ++t)
        for (UINT m = 0; m < mips; ++m) {
            const UINT n = fixture.side >> m;
            for (UINT y = 0; y < n; ++y)
                for (UINT x = 0; x < n * channels; ++x) {
                    std::uint32_t value = 0;
                    const auto& f = footprints[t][m];
                    std::memcpy(&value, memory + t * stride + f.Offset + y * f.Footprint.RowPitch + x * 4, 4);
                    words->push_back(value);
                }
        }
    staging->Unmap(0, nullptr);
    return true;
}

// Fills every subresource of the 6-slice array texture with its sentinels (to_texture) or copies them all back.
bool copy_array(ID3D12Device* device, bool to_texture, std::vector<std::uint32_t>* words, UINT texture_index = 0)
{
    constexpr UINT subresources = array_slices * mips;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[subresources]{};
    UINT64 total = 0;
    const auto desc = array_texture_desc(0);
    device->GetCopyableFootprints(&desc, 0, subresources, 0, footprints, nullptr, nullptr, &total);
    if (!total || total == ~UINT64{0}) return false;
    auto staging = buffer(device, to_texture ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK, total,
                          to_texture ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST);
    Copier copier;
    if (!staging || !copier.open(device)) return false;
    unsigned char* memory = nullptr;
    if (to_texture) {
        if (FAILED(staging->Map(0, nullptr, reinterpret_cast<void**>(&memory)))) return false;
        for (UINT s = 0; s < array_slices; ++s)
            for (UINT m = 0; m < mips; ++m) {
                const UINT sub = s * mips + m, n = fixture.side >> m;
                for (UINT y = 0; y < n; ++y)
                    for (UINT x = 0; x < n; ++x)
                        for (UINT c = 0; c < 4; ++c) {
                            const std::uint32_t value = fixture.read_write && m == 0
                                ? pattern(s, y * n + x, c) : array_sentinel(s, m, y * n + x, c);
                            std::memcpy(memory + footprints[sub].Offset + y * footprints[sub].Footprint.RowPitch + (x * 4 + c) * 4,
                                        &value, 4);
                        }
            }
        staging->Unmap(0, nullptr);
    }
    auto* texture = fixture.array_natives[texture_index].Get();
    const auto state = to_texture ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier(copier.list.Get(), texture, D3D12_RESOURCE_STATE_COMMON, state);
    for (UINT sub = 0; sub < subresources; ++sub) {
        D3D12_TEXTURE_COPY_LOCATION placed{staging.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        placed.PlacedFootprint = footprints[sub];
        D3D12_TEXTURE_COPY_LOCATION subresource{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        subresource.SubresourceIndex = sub;
        if (to_texture) copier.list->CopyTextureRegion(&subresource, 0, 0, 0, &placed, nullptr);
        else copier.list->CopyTextureRegion(&placed, 0, 0, 0, &subresource, nullptr);
    }
    barrier(copier.list.Get(), texture, state, D3D12_RESOURCE_STATE_COMMON);
    if (!copier.run()) return false;
    if (to_texture) return true;
    if (FAILED(staging->Map(0, nullptr, reinterpret_cast<void**>(&memory)))) return false;
    for (UINT s = 0; s < array_slices; ++s)
        for (UINT m = 0; m < mips; ++m) {
            const UINT sub = s * mips + m, n = fixture.side >> m;
            for (UINT y = 0; y < n; ++y)
                for (UINT x = 0; x < n * 4; ++x) {
                    std::uint32_t value = 0;
                    std::memcpy(&value, memory + footprints[sub].Offset + y * footprints[sub].Footprint.RowPitch + x * 4, 4);
                    words->push_back(value);
                }
        }
    staging->Unmap(0, nullptr);
    return true;
}

// Two 32-DWORD buffers, distinct sentinels, and an explicit view of words 4..19.
bool copy_raw(ID3D12Device* device, bool upload, std::vector<std::uint32_t>* words)
{
    constexpr UINT bytes = 32 * 4;
    auto staging = buffer(device, upload ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK, 2 * bytes,
                          upload ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST);
    Copier copier;
    if (!staging || !copier.open(device)) return false;
    std::uint32_t* data = nullptr;
    if (upload) {
        if (FAILED(staging->Map(0, nullptr, reinterpret_cast<void**>(&data)))) return false;
        for (UINT i = 0; i < 64; ++i) data[i] = 0xabcd0000u + i;
        staging->Unmap(0, nullptr);
    }
    for (UINT t = 0; t < 2; ++t) {
        auto* resource = fixture.natives[t].Get();
        const auto state = upload ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier(copier.list.Get(), resource, D3D12_RESOURCE_STATE_COMMON, state);
        if (upload) copier.list->CopyBufferRegion(resource, 0, staging.Get(), t * bytes, bytes);
        else copier.list->CopyBufferRegion(staging.Get(), t * bytes, resource, 0, bytes);
        barrier(copier.list.Get(), resource, state, D3D12_RESOURCE_STATE_COMMON);
    }
    if (!copier.run()) return false;
    if (upload) return true;
    if (FAILED(staging->Map(0, nullptr, reinterpret_cast<void**>(&data)))) return false;
    words->assign(data, data + 64);
    staging->Unmap(0, nullptr);
    return true;
}

int draw_raw_uavs(ID3D12Device* desktop, void* guest, void* list, const std::string& mode)
{
    D3D12_FEATURE_DATA_SHADER_MODEL shader_model{D3D_SHADER_MODEL_6_6};
    if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shader_model, sizeof(shader_model))) ||
        shader_model.HighestShaderModel < D3D_SHADER_MODEL_6_6) {
        std::puts("raw pixel UAV: shader model 6.6 unavailable (skip)");
        return 77;
    }
    using Create = HRESULT (WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*, UINT,
                                     const void*, void*, UINT, const void*, REFIID, void**);
    const auto create = reinterpret_cast<Create>((*static_cast<void***>(guest))[127]);
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    for (UINT t = 0; t < 2; ++t) {
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 128, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
            {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, t == 1 && mode == "raw-no-ua" ? D3D12_RESOURCE_FLAG_NONE :
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
        if (FAILED(create(guest, &heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0, nullptr,
                          __uuidof(ID3D12Resource), &fixture.resources[t])) || !fixture.resources[t]) return 70;
        fixture.natives[t] = native_of(guest, fixture.resources[t]);
        if (!fixture.natives[t]) return 71;
    }
    if (!copy_raw(desktop, true, nullptr)) return 72;
    struct Extended { D3D12_DESCRIPTOR_HEAP_DESC desc; Word address, bytes; };
    Extended desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    void* views = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr(&desc), ptr(&__uuidof(ID3D12DescriptorHeap)), ptr(&views))))) return 73;
    const Word cpu = invoke(views, 9), gpu = invoke(views, 10);
    const UINT stride = static_cast<UINT>(invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
    if (!cpu || !gpu || stride < 32) return 73;
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{DXGI_FORMAT_R32_TYPELESS, D3D12_UAV_DIMENSION_BUFFER};
    view.Buffer.FirstElement = 4;
    const bool atomic = mode.rfind("raw-atomic", 0) == 0;
    view.Buffer.NumElements = atomic ? 20 : 16;
    view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    invoke(guest, 19, ptr(fixture.resources[0]), 0, ptr(&view), cpu);
    if (mode == "raw-short") view.Buffer.NumElements = 8;
    if (mode == "raw-last") view.Buffer.FirstElement = 16;
    if (mode == "raw-typed") { view.Format = DXGI_FORMAT_R32_UINT; view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE; }
    if (mode == "raw-structured") {
        view.Format = DXGI_FORMAT_UNKNOWN; view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE; view.Buffer.StructureByteStride = 4;
    }
    if (mode == "raw-srv") {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{DXGI_FORMAT_R32_TYPELESS, D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        srv.Buffer.NumElements = 32; srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        invoke(guest, 18, ptr(fixture.resources[1]), ptr(&srv), cpu + stride);
    } else if (mode != "raw-missing") {
        invoke(guest, 19, ptr(fixture.resources[1]), 0, ptr(&view), cpu + stride);
    }
    std::printf("raw pixel UAV case %s, title descriptor stride %u\n", mode.c_str(), stride);
    invoke(list, 32, 1, ptr(&views));
    if (mode != "raw-no-table") invoke(list, 36, 0, gpu);
    if (mode == "raw-atomic-edge") {
        const D3D12_RECT scissor{0, 0, 3, 3};
        invoke(list, 22, 1, ptr(&scissor));
    }
    invoke(list, 12, 3, 1);
    return 0;
}

ID3D12InfoQueue* debug_messages()
{
    static ComPtr<ID3D12InfoQueue> queue;
    if (!queue && fixture.desktop) fixture.desktop->QueryInterface(IID_PPV_ARGS(&queue));
    return queue.Get();
}
}  // namespace

int draw_pixel_uavs(ID3D12Device* desktop, void* guest, void* list)
{
    char mode[32]{};
    GetEnvironmentVariableA("X2W_TEST_PIXEL_UAV", mode, sizeof(mode));
    const std::string which(mode);
    fixture.desktop = desktop;
    fixture.raw_mode = which.rfind("raw-", 0) == 0;
    if (fixture.raw_mode) return draw_raw_uavs(desktop, guest, list, which);
    fixture.array_mode = which.rfind("arr-", 0) == 0;
    fixture.atomic_mode = which.rfind("atomic-", 0) == 0;
    fixture.read_write = GetEnvironmentVariableW(L"X2W_TEST_PIXEL_UAV_RW", nullptr, 0) != 0;
    fixture.side = fixture.read_write ? 8 : side_default;
    const bool stale = which == "rw-stale" || which == "arr-rw-stale";
    if (fixture.array_mode) {
        using Create = HRESULT (WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*, UINT, const void*,
                                         void*, UINT, const void*, REFIID, void**);
        const auto create = reinterpret_cast<Create>((*static_cast<void***>(guest))[127]);
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        for (UINT t = 0; t < 3; ++t) {
            const auto desc = array_texture_desc(t);
            if (FAILED(create(guest, &heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0, nullptr,
                              __uuidof(ID3D12Resource), &fixture.array_resources[t])) || !fixture.array_resources[t])
                return 70;
            fixture.array_natives[t] = native_of(guest, fixture.array_resources[t]);
            if (!fixture.array_natives[t]) return 71;
        }
        const auto plain_desc = texture_desc(0);
        if (FAILED(create(guest, &heap, 0, &plain_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0, nullptr,
                          __uuidof(ID3D12Resource), &fixture.resources[0])) || !fixture.resources[0])
            return 70;
        if (!copy_array(desktop, true, nullptr) || (stale && !copy_array(desktop, true, nullptr, 1))) return 72;

        struct Extended {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            Word address, bytes;
        } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
        void* desc_heap = nullptr;
        if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr(&heap_desc), ptr(&__uuidof(ID3D12DescriptorHeap)), ptr(&desc_heap)))))
            return 73;
        const Word cpu = invoke(desc_heap, 9), gpu = invoke(desc_heap, 10);
        const UINT stride = static_cast<UINT>(invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
        if (!cpu || !gpu || stride < 32) return 73;

        D3D12_UNORDERED_ACCESS_VIEW_DESC arr_view{DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_UAV_DIMENSION_TEXTURE2DARRAY};
        arr_view.Texture2DArray = {0, 1, 4, 0};  // FirstArraySlice=1, ArraySize=4 -> physical slices 1..4
        invoke(guest, 19, ptr(fixture.array_resources[0]), 0, ptr(&arr_view), cpu + 0 * stride);
        D3D12_UNORDERED_ACCESS_VIEW_DESC plain_view{DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_UAV_DIMENSION_TEXTURE2D};
        invoke(guest, 19, ptr(fixture.resources[0]), 0, ptr(&plain_view), cpu + 1 * stride);
        invoke(guest, 19, ptr(fixture.array_resources[0]), 0, 0, cpu + 2 * stride);
        D3D12_UNORDERED_ACCESS_VIEW_DESC mip1 = arr_view;
        mip1.Texture2DArray.MipSlice = 1;
        invoke(guest, 19, ptr(fixture.array_resources[0]), 0, ptr(&mip1), cpu + 3 * stride);
        D3D12_SHADER_RESOURCE_VIEW_DESC arr_srv{DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                                                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        arr_srv.Texture2DArray = {0, 1, 1, 4, 0, 0.0f};
        invoke(guest, 18, ptr(fixture.array_resources[0]), ptr(&arr_srv), cpu + 4 * stride);
        invoke(guest, 19, ptr(fixture.array_resources[1]), 0, ptr(&arr_view), cpu + 5 * stride);
        D3D12_UNORDERED_ACCESS_VIEW_DESC integer = arr_view;
        integer.Format = DXGI_FORMAT_R32G32B32A32_UINT;
        invoke(guest, 19, ptr(fixture.array_resources[2]), 0, ptr(&integer), cpu + 6 * stride);
        if (stale) invoke(guest, 18, ptr(fixture.array_resources[1]), ptr(&arr_srv), cpu + stride);

        struct Case { const char* name; int entry; int srv_entry; };
        static const Case cases[] = {
            {"arr-ok", 0, -1}, {"arr-2d-view", 1, -1}, {"arr-default", 2, -1}, {"arr-mip1", 3, -1},
            {"arr-srv-and-uav", 0, 4}, {"arr-no-ua", 5, -1}, {"arr-uint", 6, -1},
            {"arr-rw", 0, -1}, {"arr-rw-stale", 0, -1},
        };
        const Case* chosen = nullptr;
        for (const Case& c : cases)
            if (which == c.name) chosen = &c;
        if (!chosen) return 74;
        std::printf("pixel uav case %s: table entry %d, title descriptor stride %u\n", chosen->name, chosen->entry, stride);
        invoke(list, 32, 1, ptr(&desc_heap));
        if (chosen->entry >= 0) invoke(list, 36, 0, gpu + Word(chosen->entry) * stride);
        if (chosen->srv_entry >= 0) invoke(list, 36, 1, gpu + Word(chosen->srv_entry) * stride);
        if (fixture.read_write) draw_twice(list);
        else invoke(list, 12, 3, 1);
        return 0;
    }
    for (UINT t = 0; t < textures; ++t) {
        using Create = HRESULT (WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*, UINT, const void*,
                                         void*, UINT, const void*, REFIID, void**);
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        const auto desc = texture_desc(t);
        const auto create = reinterpret_cast<Create>((*static_cast<void***>(guest))[127]);
        if (FAILED(create(guest, &heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0, nullptr,
                          __uuidof(ID3D12Resource), &fixture.resources[t])) || !fixture.resources[t])
            return 70;
        fixture.natives[t] = native_of(guest, fixture.resources[t]);
        if (!fixture.natives[t]) return 71;
    }
    if (!copy_all(desktop, true, nullptr)) return 72;

    struct Extended {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word address, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 9, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    void* heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr(&heap_desc), ptr(&__uuidof(ID3D12DescriptorHeap)), ptr(&heap)))))
        return 73;
    const Word cpu = invoke(heap, 9), gpu = invoke(heap, 10);
    const UINT stride = static_cast<UINT>(invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
    if (!cpu || !gpu || stride < 32) return 73;

    if (fixture.atomic_mode) {
        UINT entry = 0;
        if (which == "atomic-float") entry = 1;
        else if (which == "atomic-sint") entry = 2;
        else if (which == "atomic-cast") entry = 3;
        else if (which != "atomic-uint") return 74;
        const auto format = entry == 3 ? DXGI_FORMAT_R32_UINT : texture_desc(entry).Format;
        const D3D12_UNORDERED_ACCESS_VIEW_DESC view{format, D3D12_UAV_DIMENSION_TEXTURE2D};
        invoke(guest, 19, ptr(fixture.resources[entry]), 0, ptr(&view), cpu);
        invoke(list, 32, 1, ptr(&heap));
        invoke(list, 36, 0, gpu);
        std::printf("pixel uav case %s: table entry 0, title descriptor stride %u\n", which.c_str(), stride);
        invoke(list, 12, 3, 1);
        return 0;
    }

    D3D12_UNORDERED_ACCESS_VIEW_DESC view{DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_UAV_DIMENSION_TEXTURE2D};
    invoke(guest, 19, ptr(fixture.resources[0]), 0, ptr(&view), cpu + 0 * stride);
    invoke(guest, 19, ptr(fixture.resources[1]), 0, 0, cpu + 1 * stride);
    invoke(guest, 19, ptr(fixture.resources[2]), 0, ptr(&view), cpu + 2 * stride);
    D3D12_UNORDERED_ACCESS_VIEW_DESC integer = view;
    integer.Format = DXGI_FORMAT_R32G32B32A32_UINT;
    invoke(guest, 19, ptr(fixture.resources[3]), 0, ptr(&integer), cpu + 3 * stride);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_SRV_DIMENSION_TEXTURE2D,
                                        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
    srv.Texture2D.MipLevels = 1;
    invoke(guest, 18, ptr(fixture.resources[0]), ptr(&srv), cpu + 5 * stride);
    if (stale) invoke(guest, 18, ptr(fixture.resources[1]), ptr(&srv), cpu + stride);
    D3D12_UNORDERED_ACCESS_VIEW_DESC mip1 = view;
    mip1.Texture2D.MipSlice = 1;
    invoke(guest, 19, ptr(fixture.resources[0]), 0, ptr(&mip1), cpu + 6 * stride);
    D3D12_UNORDERED_ACCESS_VIEW_DESC array{DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_UAV_DIMENSION_TEXTURE2DARRAY};
    array.Texture2DArray.ArraySize = 1;
    invoke(guest, 19, ptr(fixture.resources[0]), 0, ptr(&array), cpu + 7 * stride);

    // E: a 4x4 R8G8B8A8_UNORM texture that is both a render target and writable; entry 8 is its UAV. The target
    // case binds it as the draw's render target and names that entry.
    using Create = HRESULT (WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*, UINT, const void*,
                                     void*, UINT, const void*, REFIID, void**);
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC target_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, fixture.side, fixture.side, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    static void* target = nullptr;
    if (FAILED(reinterpret_cast<Create>((*static_cast<void***>(guest))[127])(guest, &default_heap, 0, &target_desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0, nullptr, __uuidof(ID3D12Resource), &target)) || !target)
        return 70;
    D3D12_UNORDERED_ACCESS_VIEW_DESC target_view{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_UAV_DIMENSION_TEXTURE2D};
    invoke(guest, 19, ptr(target), 0, ptr(&target_view), cpu + 8 * stride);
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word address, bytes;
    } rtv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    void* rtv_heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr(&rtv_heap_desc), ptr(&__uuidof(ID3D12DescriptorHeap)), ptr(&rtv_heap)))))
        return 73;
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(rtv_heap, 9))};
    const D3D12_RENDER_TARGET_VIEW_DESC target_rtv{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    invoke(guest, 20, ptr(target), ptr(&target_rtv), rtv.ptr);

    struct Case { const char* name; int entry; int srv_entry; };
    static const Case cases[] = {
        {"a", 0, -1}, {"b", 1, -1}, {"no-ua", 2, -1}, {"uint", 3, -1}, {"missing", 4, -1}, {"srv", 5, -1},
        {"mip1", 6, -1}, {"array", 7, -1}, {"out-of-range", 7, -1}, {"no-table", -1, -1}, {"srv-and-uav", 0, 5},
        {"target", 8, -1}, {"rw", 0, -1}, {"rw-stale", 0, -1},
    };
    const Case* chosen = nullptr;
    for (const Case& c : cases)
        if (which == c.name) chosen = &c;
    if (!chosen) return 74;
    std::printf("pixel uav case %s: table entry %d, title descriptor stride %u\n", chosen->name, chosen->entry, stride);
    if (which == "target") invoke(list, 50, 1, ptr(&rtv));
    invoke(list, 32, 1, ptr(&heap));
    if (chosen->entry >= 0) invoke(list, 36, 0, gpu + Word(chosen->entry) * stride);
    if (chosen->srv_entry >= 0) invoke(list, 36, 1, gpu + Word(chosen->srv_entry) * stride);
    if (fixture.read_write) draw_twice(list);
    else invoke(list, 12, 3, 1);
    return 0;
}

// After the fence: every word of every subresource of the four textures, texture by texture, mip by mip, row by
// row, to <pixels>.uav; then the debug layer's error count.
int report_pixel_uavs(const char* pixels)
{
    std::vector<std::uint32_t> words;
    const bool copied = fixture.raw_mode ? copy_raw(fixture.desktop, false, &words) :
        fixture.array_mode ? copy_array(fixture.desktop, false, &words) : copy_all(fixture.desktop, false, &words);
    if (copied) {
        std::ofstream out(std::string(pixels) + ".uav", std::ios::binary);
        out.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
        if (!out) return 76;
    }
    const HRESULT removed = fixture.desktop->GetDeviceRemovedReason();
    std::printf("device removed reason: 0x%08lx\n", static_cast<unsigned long>(removed));
    auto* messages = debug_messages();
    if (!messages) {
        std::printf("debug layer: absent\n");
        return copied ? 0 : 75;
    }
    UINT errors = 0;
    for (UINT64 i = 0; i < messages->GetNumStoredMessages(); ++i) {
        SIZE_T size = 0;
        if (FAILED(messages->GetMessage(i, nullptr, &size)) || !size) continue;
        std::vector<char> storage(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(messages->GetMessage(i, message, &size)) || message->Severity > D3D12_MESSAGE_SEVERITY_ERROR) continue;
        if (++errors <= 8) std::printf("debug layer error id %d: %.300s\n", static_cast<int>(message->ID), message->pDescription);
    }
    std::printf("debug layer: on, errors: %u\n", errors);
    return copied ? 0 : 75;
}
