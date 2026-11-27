// X2W_DEBUG_PRESENT_TARGET on WARP: the production selection, copy and conversion (x2w_d3d12_fixture_debug_present) run on
// desktop objects with a known scene. A depth buffer and a colour target hold different shapes and the presented surface
// starts magenta. Checked against hand-computed 8-bit pixels (not against the code's own conversion):
//   depth        near geometry 255, far geometry 40, background black, stretched 2x to the 32x32 surface (BGRA8)
//   reversed Z   the same picture from a buffer cleared to 0 with larger values nearer
//   largest-rt   the colour target (half-float 1.0 -> 186, 0.5 -> 155 after tone mapping), not the depth buffer
//   rt:WxH:fmt   the same colour picture; a size or format that matches nothing leaves the surface magenta
// A mutant that copies the other target fails the depth and colour comparisons, which differ in every shape pixel.
// usage: startup_debug_present <warp|hardware> [wrong-target]
// wrong-target is the mutant control: the depth setting is checked against the COLOUR picture, as if the wrong target were
// copied; it must fail (CTest WILL_FAIL).
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllimport) int x2w_d3d12_fixture_debug_present(ID3D12Device*, ID3D12CommandQueue*, const char*, ID3D12Resource*, int,
                                                                      ID3D12Resource*, int, ID3D12Resource*, char*, std::size_t);

namespace {

constexpr UINT size = 16, surface_size = 32;
constexpr int dxgi_rgba16f = 10, dxgi_d32 = 40;

struct Scene {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    UINT64 fence_value = 0;
};

bool fail(const char* what)
{
    std::fprintf(stderr, "startup_debug_present: %s\n", what);
    return false;
}

ComPtr<ID3D12Resource> texture(ID3D12Device* device, DXGI_FORMAT format, UINT width, UINT height, D3D12_RESOURCE_FLAGS flags)
{
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, format, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, flags};
    ComPtr<ID3D12Resource> resource;
    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource));
    return resource;
}

void barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &b);
}

bool run_list(Scene& s)
{
    if (FAILED(s.list->Close())) return false;
    ID3D12CommandList* lists[] = {s.list.Get()};
    s.queue->ExecuteCommandLists(1, lists);
    s.queue->Signal(s.fence.Get(), ++s.fence_value);
    if (s.fence->GetCompletedValue() < s.fence_value) {
        const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        s.fence->SetEventOnCompletion(s.fence_value, event);
        const DWORD waited = WaitForSingleObject(event, 20000);
        CloseHandle(event);
        if (waited != WAIT_OBJECT_0) return false;
    }
    return SUCCEEDED(s.allocator->Reset()) && SUCCEEDED(s.list->Reset(s.allocator.Get(), nullptr));
}

// A depth buffer: background `clear`, a left half at `left`, a block (x 8..15, y 4..11) at `block`.
bool fill_depth(Scene& s, ID3D12Resource* depth, float clear, float left, float block)
{
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
    ComPtr<ID3D12DescriptorHeap> heap;
    if (FAILED(s.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)))) return false;
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_DEPTH_STENCIL_VIEW_DESC view{DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_FLAG_NONE, {}};
    s.device->CreateDepthStencilView(depth, &view, handle);
    barrier(s.list.Get(), depth, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    s.list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, clear, 0, 0, nullptr);
    const D3D12_RECT half{0, 0, 8, 16}, box{8, 4, 16, 12};
    s.list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, left, 0, 1, &half);
    s.list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, block, 0, 1, &box);
    barrier(s.list.Get(), depth, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COMMON);
    return run_list(s);
}

bool clear_rects(Scene& s, ID3D12Resource* target, const float (&base)[4], const float (&a)[4], const D3D12_RECT& area_a,
                 const float (&b)[4], const D3D12_RECT& area_b)
{
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    ComPtr<ID3D12DescriptorHeap> heap;
    if (FAILED(s.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)))) return false;
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    s.device->CreateRenderTargetView(target, nullptr, handle);
    barrier(s.list.Get(), target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
    s.list->ClearRenderTargetView(handle, base, 0, nullptr);
    s.list->ClearRenderTargetView(handle, a, 1, &area_a);
    s.list->ClearRenderTargetView(handle, b, 1, &area_b);
    barrier(s.list.Get(), target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
    return run_list(s);
}

// The surface's pixels as B, G, R, A bytes, row by row.
bool read_surface(Scene& s, ID3D12Resource* surface, std::vector<unsigned char>& bgra)
{
    const D3D12_RESOURCE_DESC desc = surface->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    s.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_RESOURCE_DESC buffer_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, total, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
    ComPtr<ID3D12Resource> readback;
    if (FAILED(s.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                 IID_PPV_ARGS(&readback))))
        return false;
    barrier(s.list.Get(), surface, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{surface, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, to{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    to.PlacedFootprint = footprint;
    s.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    barrier(s.list.Get(), surface, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    if (!run_list(s)) return false;
    void* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped)) || !mapped) return false;
    bgra.assign(static_cast<std::size_t>(desc.Width) * desc.Height * 4, 0);
    for (UINT y = 0; y < desc.Height; ++y)
        std::memcpy(&bgra[static_cast<std::size_t>(y) * desc.Width * 4],
                    static_cast<const unsigned char*>(mapped) + footprint.Offset + static_cast<UINT64>(y) * footprint.Footprint.RowPitch,
                    desc.Width * 4);
    const D3D12_RANGE none{0, 0};
    readback->Unmap(0, &none);
    return true;
}

bool paint_magenta(Scene& s, ID3D12Resource* surface)
{
    const float magenta[4]{1, 0, 1, 1};
    const D3D12_RECT corner{0, 0, 1, 1};
    return clear_rects(s, surface, magenta, magenta, corner, magenta, corner);
}

struct Pixel { unsigned char b, g, r; };

// Expected picture: every surface pixel (x, y) comes from source pixel (x / 2, y / 2).
template<class Func>
bool expect(Scene& s, ID3D12Resource* surface, const char* label, Func source_pixel)
{
    std::vector<unsigned char> got;
    if (!read_surface(s, surface, got)) return fail("surface readback failed");
    for (UINT y = 0; y < surface_size; ++y)
        for (UINT x = 0; x < surface_size; ++x) {
            const Pixel want = source_pixel(x / 2, y / 2);
            const unsigned char* p = &got[(static_cast<std::size_t>(y) * surface_size + x) * 4];
            if (p[0] != want.b || p[1] != want.g || p[2] != want.r || p[3] != 255) {
                std::fprintf(stderr, "startup_debug_present: %s: surface (%u,%u) is B%u G%u R%u A%u, expected B%u G%u R%u A255\n", label, x, y,
                             p[0], p[1], p[2], p[3], want.b, want.g, want.r);
                return false;
            }
        }
    return true;
}

Pixel depth_pixel(UINT x, UINT y)  // standard Z: left half nearest (255), block farthest (40), background black
{
    if (x < 8) return {255, 255, 255};
    if (y >= 4 && y < 12) return {40, 40, 40};
    return {0, 0, 0};
}

Pixel reversed_pixel(UINT x, UINT y)  // reversed Z has the same shapes: left 0.9 is nearest, block 0.5 farthest
{
    return depth_pixel(x, y);
}

Pixel colour_pixel(UINT x, UINT y)  // red 1.0 -> 186 in the top-left 8x8, green 0.5 -> 155 in the bottom-right 8x8
{
    if (x < 8 && y < 8) return {0, 0, 186};
    if (x >= 8 && y >= 8) return {0, 155, 0};
    return {0, 0, 0};
}

Pixel magenta_pixel(UINT, UINT) { return {255, 0, 255}; }

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2 && !(argc == 3 && !std::strcmp(argv[2], "wrong-target"))) return 2;
    const bool wrong_target = argc == 3;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 20;
    if (!std::strcmp(argv[1], "warp")) {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 20;
    } else if (FAILED(factory->EnumAdapters(0, &adapter))) {
        return 20;
    }
    Scene s;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&s.device)))) return 77;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(s.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&s.queue))) ||
        FAILED(s.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s.allocator))) ||
        FAILED(s.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.allocator.Get(), nullptr, IID_PPV_ARGS(&s.list))) ||
        FAILED(s.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.fence))))
        return 21;

    auto depth = texture(s.device.Get(), DXGI_FORMAT_R32_TYPELESS, size, size, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    auto reversed = texture(s.device.Get(), DXGI_FORMAT_R32_TYPELESS, size, size, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    auto colour = texture(s.device.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT, size, size, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    auto surface = texture(s.device.Get(), DXGI_FORMAT_B8G8R8A8_UNORM, surface_size, surface_size, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    if (!depth || !reversed || !colour || !surface) return 22;

    const float black[4]{0, 0, 0, 1}, red[4]{1, 0, 0, 1}, green[4]{0, 0.5f, 0, 1};
    if (!fill_depth(s, depth.Get(), 1.0f, 0.25f, 0.75f) || !fill_depth(s, reversed.Get(), 0.0f, 0.9f, 0.5f) ||
        !clear_rects(s, colour.Get(), black, red, D3D12_RECT{0, 0, 8, 8}, green, D3D12_RECT{8, 8, 16, 16}))
        return fail("the scene could not be drawn"), 23;

    char what[128];
    const auto show = [&](const char* spec, ID3D12Resource* with_depth) {
        return x2w_d3d12_fixture_debug_present(s.device.Get(), s.queue.Get(), spec, colour.Get(), dxgi_rgba16f, with_depth, dxgi_d32,
                                               surface.Get(), what, sizeof(what));
    };

    // depth
    if (!paint_magenta(s, surface.Get())) return 24;
    if (show("depth", depth.Get()) != 0 || std::strcmp(what, "depth 16x16 fmt 40")) return fail(what), 30;
    if (!expect(s, surface.Get(), "depth", wrong_target ? colour_pixel : depth_pixel)) return 31;
    // reversed Z
    if (!paint_magenta(s, surface.Get())) return 24;
    if (show("depth", reversed.Get()) != 0 || !expect(s, surface.Get(), "reversed Z", reversed_pixel)) return 32;
    // the colour target, by "largest-rt" and by its exact size and format; the depth buffer is also bound and must not win
    if (!paint_magenta(s, surface.Get())) return 24;
    if (show("largest-rt", depth.Get()) != 0 || std::strcmp(what, "colour 16x16 fmt 10")) return fail(what), 33;
    if (!expect(s, surface.Get(), "largest-rt", colour_pixel)) return 34;
    if (!paint_magenta(s, surface.Get())) return 24;
    if (show("rt:16x16:10", depth.Get()) != 0 || !expect(s, surface.Get(), "rt:16x16:10", colour_pixel)) return 35;
    // a target that matches nothing leaves the picture alone
    if (!paint_magenta(s, surface.Get())) return 24;
    if (show("rt:16x16:28", depth.Get()) != 1 || show("rt:32x32:10", depth.Get()) != 1 || show("", depth.Get()) != 1) return fail("a setting that matches nothing drew"), 36;
    if (!expect(s, surface.Get(), "no match", magenta_pixel)) return 37;
    std::puts("startup_debug_present: PASS (depth, reversed Z, largest-rt and rt:WxH:fmt pictures match the hand-computed pixels; no match leaves the surface)");
    return 0;
}
