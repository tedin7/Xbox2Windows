// CopyTextureRegion (command list slot 16) through the wrapped list, read back (#870).
//
//   startup_list_copy_texture_warp <warp|hardware>     (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1 X2W_D3D12_DEBUG=1)
//
// Through D3D12XboxCreateDevice the probe makes a 4x4 RGBA8 texture with 2 mips and 3 array slices (six
// subresources), an upload buffer holding two distinct 4x4 images and a readback buffer, all as console
// resources. Then it records, submits and reads back:
//
//   1. upload image A -> subresource 2 (mip 0, slice 1), image B -> subresource 4 (mip 0, slice 2), and both
//      back to the readback buffer: the readback holds A then B, and both copies were forwarded.
//   2. a list whose three calls must each be refused before any desktop copy: subresource 6 (one past the last
//      array slice, which a mip-only modulo would read as mip 0), subresource 0xffffffff, and a footprint
//      format that differs from the texture's. The readback buffer, pre-filled with a sentinel, is unchanged,
//      and the refusal counter rises by three with the forward counter unchanged.
//
// Exit codes: 0 pass, 77 no physical adapter, other values name the failing step.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_list_copy_texture_stats(LONG* forwarded, LONG* refused);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0, Word g = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f, g);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

constexpr UINT side = 4, mips = 2, slices = 3, pitch = 256;
constexpr UINT image_bytes = 1024;  // 4 rows at the 256-byte pitch; image B starts at this offset (a multiple of 512)
constexpr std::uint8_t sentinel = 0xcd;

// A console resource through device slot 127 (CreateCommittedResource3).
void* make_resource(void* device, D3D12_HEAP_TYPE type, const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state)
{
    using Create = HRESULT(WINAPI*)(void*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*, UINT, const void*,
                                    void*, UINT, const void*, const GUID*, void**);
    const D3D12_HEAP_PROPERTIES heap{type};
    void* resource = nullptr;
    const auto create = reinterpret_cast<Create>((*static_cast<void***>(device))[127]);
    return SUCCEEDED(create(device, &heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, nullptr, 0, nullptr,
                            &__uuidof(ID3D12Resource), &resource))
               ? resource
               : nullptr;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes)
{
    return {D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
}

D3D12_TEXTURE_COPY_LOCATION footprint(void* buffer, UINT64 offset, DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM)
{
    D3D12_TEXTURE_COPY_LOCATION location{static_cast<ID3D12Resource*>(buffer), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    location.PlacedFootprint = {offset, {format, side, side, 1, pitch}};
    return location;
}

D3D12_TEXTURE_COPY_LOCATION subresource(void* texture, UINT index)
{
    D3D12_TEXTURE_COPY_LOCATION location{static_cast<ID3D12Resource*>(texture), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    location.SubresourceIndex = index;
    return location;
}

// Image A holds byte i*3+1 at texel byte i, image B byte 255-i*5 (both wrap at 256).
std::uint8_t image_byte(int image, UINT row, UINT column)
{
    const UINT i = row * side * 4 + column;
    return image == 0 ? static_cast<std::uint8_t>(i * 3 + 1) : static_cast<std::uint8_t>(255 - i * 5);
}

bool wait_for(void* fence, Word value)
{
    const auto deadline = GetTickCount64() + 10000;
    while (invoke(fence, 8) < value) {
        if (GetTickCount64() >= deadline) return false;
        Sleep(1);
    }
    return true;
}

// Compares the 4x4 image the readback buffer holds at offset against image number, row by row.
bool image_matches(const std::uint8_t* memory, UINT offset, int image)
{
    for (UINT row = 0; row < side; ++row)
        for (UINT column = 0; column < side * 4; ++column)
            if (memory[offset + row * pitch + column] != image_byte(image, row, column)) {
                std::printf("image %d row %u byte %u: got %02x want %02x\n", image, row, column, memory[offset + row * pitch + column],
                            image_byte(image, row, column));
                return false;
            }
    return true;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(argv[1], "hardware") == 0) {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    } else if (std::strcmp(argv[1], "warp") != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 3;
    }
    DXGI_ADAPTER_DESC adapter_desc{};
    if (FAILED(adapter->GetDesc(&adapter_desc))) return 3;
    std::printf("adapter=%ls\n", adapter_desc.Description);

    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 4;

    const D3D12_RESOURCE_DESC texture_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, side, side, slices, mips,
                                           DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    void* texture = make_resource(device, D3D12_HEAP_TYPE_DEFAULT, texture_desc, D3D12_RESOURCE_STATE_COMMON);
    void* upload = make_resource(device, D3D12_HEAP_TYPE_UPLOAD, buffer_desc(2 * image_bytes), D3D12_RESOURCE_STATE_GENERIC_READ);
    void* readback = make_resource(device, D3D12_HEAP_TYPE_READBACK, buffer_desc(2 * image_bytes), D3D12_RESOURCE_STATE_COPY_DEST);
    if (!texture || !upload || !readback) return 5;

    void* memory = nullptr;
    if (!ok(invoke(upload, 8, 0, 0, arg(&memory))) || !memory) return 6;
    for (int image = 0; image < 2; ++image)
        for (UINT row = 0; row < side; ++row)
            for (UINT column = 0; column < side * 4; ++column)
                static_cast<std::uint8_t*>(memory)[image * image_bytes + row * pitch + column] = image_byte(image, row, column);
    invoke(upload, 9, 0, 0);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void *queue = nullptr, *fence = nullptr, *allocator = nullptr, *list = nullptr;
    if (!ok(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&queue))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&fence))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))))
        return 7;
    Word fence_value = 0;
    const auto submit = [&] {
        void* lists[] = {list};
        invoke(queue, 10, 1, arg(lists));
        return ok(invoke(queue, 14, arg(fence), ++fence_value)) && wait_for(fence, fence_value);
    };
    const auto copy = [&](const D3D12_TEXTURE_COPY_LOCATION& to, const D3D12_TEXTURE_COPY_LOCATION& from) {
        invoke(list, 16, arg(&to), 0, 0, 0, arg(&from), 0);
    };
    LONG forwarded = 0, refused = 0;
    x2w_d3d12_list_copy_texture_stats(&forwarded, &refused);

    // 1. Two distinct images into two array slices and back.
    const auto a_up = footprint(upload, 0), b_up = footprint(upload, image_bytes);
    const auto a_out = footprint(readback, 0), b_out = footprint(readback, image_bytes);
    const auto slice1 = subresource(texture, 2), slice2 = subresource(texture, 4);
    if (!ok(invoke(list, 139, arg(allocator), 0))) return 8;
    copy(slice1, a_up);
    copy(slice2, b_up);
    copy(a_out, slice1);
    copy(b_out, slice2);
    if (!ok(invoke(list, 9)) || !submit()) return 9;
    LONG forwarded1 = 0, refused1 = 0;
    x2w_d3d12_list_copy_texture_stats(&forwarded1, &refused1);
    std::printf("1 forwarded +%ld refused +%ld (expected +4, +0)\n", forwarded1 - forwarded, refused1 - refused);
    if (forwarded1 - forwarded != 4 || refused1 != refused) return 10;
    if (!ok(invoke(readback, 8, 0, 0, arg(&memory))) || !memory) return 11;
    const bool images = image_matches(static_cast<const std::uint8_t*>(memory), 0, 0) &&
                        image_matches(static_cast<const std::uint8_t*>(memory), image_bytes, 1);
    std::memset(memory, sentinel, 2 * image_bytes);  // mapped readback memory is CPU-writable for the next step
    invoke(readback, 9, 0, 0);
    if (!images) return 12;

    // 2. Out-of-range subresources and a format mismatch are refused before any desktop copy.
    if (!ok(invoke(list, 139, arg(allocator), 0))) return 13;
    copy(a_out, subresource(texture, mips * slices));           // one past the last array slice
    copy(subresource(texture, 0xffffffffu), a_up);              // an index far outside
    copy(subresource(texture, 0), footprint(upload, 0, DXGI_FORMAT_B8G8R8A8_UNORM));  // formats differ
    if (!ok(invoke(list, 9)) || !submit()) return 14;
    LONG forwarded2 = 0, refused2 = 0;
    x2w_d3d12_list_copy_texture_stats(&forwarded2, &refused2);
    std::printf("2 forwarded +%ld refused +%ld (expected +0, +3)\n", forwarded2 - forwarded1, refused2 - refused1);
    if (forwarded2 != forwarded1 || refused2 - refused1 != 3) return 15;
    if (!ok(invoke(readback, 8, 0, 0, arg(&memory))) || !memory) return 16;
    bool unchanged = true;
    for (UINT i = 0; i < 2 * image_bytes; ++i) unchanged = unchanged && static_cast<const std::uint8_t*>(memory)[i] == sentinel;
    invoke(readback, 9, 0, 0);
    if (!unchanged) {
        std::puts("2 the readback buffer changed");
        return 17;
    }
    std::puts("ok");
    return 0;
}
