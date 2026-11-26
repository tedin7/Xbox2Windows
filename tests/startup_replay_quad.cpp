// A small animated scene drawn only through the D3D12 stand-in, for the replay test (tests/startup_replay_check.py):
// every object (render target, descriptor heap, queue, allocator, list, fence) is a stand-in object, and every frame
// is presented on the stand-in's queue, so a recording of this process holds everything a replay needs.
//
// Frame f: clear to a colour that moves with f, draw the stream pipeline's triangle in a scissor whose width moves
// with f, execute, present the target (queue slot 32), signal and wait. No mapped memory is written.
//
// usage: startup_replay_quad <vs> <ps> <root> <warp|hardware> <presents>   [X2W_TEST_REPLAY_PAUSE=<frame>,<ms>]
#include <d3d12.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "startup_stream_setup.h"

using Word = StreamWord;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }

namespace {

constexpr UINT width = 64, height = 32;

struct Scene {
    void* target = nullptr;
    void* rtv_heap = nullptr;
    void* queue = nullptr;
    void* allocator = nullptr;
    void* list = nullptr;
    void* fence = nullptr;
    SIZE_T rtv = 0, fallback_rtv = 0;
    alignas(64) unsigned char title_rtvs[512]{};
};

int create_scene(void* device, Scene& s)
{
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                   {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 1}};
    // X2W_TEST_REPLAY_CREATE127=1: the target comes from device slot 127 (the console CreateCommittedResource: ten
    // arguments after the device, the output last), so a recording must carry more than seven argument words. The
    // word after them (the caller's stack, which the slot ignores) points at the output variable too, as a title's
    // stack can: the replay must take the new object from the output the call wrote, not from that copy.
    if (GetEnvironmentVariableA("X2W_TEST_REPLAY_CREATE127", nullptr, 0)) {
        using Create = HRESULT (*)(void*, const void*, UINT, const void*, UINT, const void*, void*, UINT, const void*, const GUID*, void**,
                                   void**);
        const auto create = reinterpret_cast<Create>((*static_cast<void***>(device))[127]);
        if (FAILED(create(device, &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, &clear, nullptr, 0, nullptr,
                          &__uuidof(ID3D12Resource), &s.target, &s.target)) || !s.target)
            return 25;
    } else if (FAILED(static_cast<HRESULT>(invoke(device, 27, arg(&heap), D3D12_HEAP_FLAG_NONE, arg(&desc), D3D12_RESOURCE_STATE_COMMON,
                                                  arg(&clear), arg(&__uuidof(ID3D12Resource)), arg(&s.target)))) || !s.target) {
        return 21;
    }
    const bool descriptor_window = GetEnvironmentVariableA("X2W_TEST_REPLAY_DESCRIPTOR_WINDOW", nullptr, 0) != 0;
    const bool null_views = GetEnvironmentVariableA("X2W_TEST_REPLAY_NULL_VIEWS", nullptr, 0) != 0;
    constexpr UINT ranges = 40;  // handle array is larger than the recorder's generic 256-byte argument capture
    // The title's heap descriptor: the desktop one, then a title CPU address and its size (none here).
    struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } rtv_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, descriptor_window ? 2 * ranges : 1u}, 0, 0};
    const D3D12_RENDER_TARGET_VIEW_DESC view{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    Word title_stride = 0;
    if (null_views) {
        rtv_desc.desc.NumDescriptors = 2;
        title_stride = invoke(device, 15, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        if (!title_stride || title_stride > sizeof(s.title_rtvs) / 2) return 26;
        rtv_desc.backing = arg(s.title_rtvs);
        rtv_desc.bytes = 2 * title_stride;
    }
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, arg(&rtv_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&s.rtv_heap)))))
        return 22;
    if (null_views) {
        // The title uses the backing directly, before it queries the heap start.
        auto null_view = view;
        null_view.Format = DXGI_FORMAT_B4G4R4A4_UNORM;
        invoke(device, 20, 0, arg(&null_view), rtv_desc.backing);
        invoke(device, 20, arg(s.target), arg(&view), rtv_desc.backing + title_stride);
    }
    // The heap start read as a title compiled for struct returns reads it: (return buffer, object).
    using Start = SIZE_T* (*)(SIZE_T*, void*);
    const auto start = reinterpret_cast<Start>((*static_cast<void***>(s.rtv_heap))[9]);
    if (start(&s.rtv, s.rtv_heap) != &s.rtv || !s.rtv) return 24;
    if (null_views) {
        if (s.rtv != rtv_desc.backing) return 26;
        s.rtv += title_stride;
    } else if (descriptor_window) {
        const auto type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        const Word stride = invoke(device, 15, type);
        if (!stride) return 25;
        Word destinations[ranges];
        for (UINT i = 0; i < ranges; ++i) {
            invoke(device, 20, arg(s.target), arg(&view), s.rtv + i * stride);
            destinations[i] = s.rtv + (ranges + i) * stride;
        }
        const UINT source_size = ranges;
        invoke(device, 23, ranges, arg(destinations), 0, 1, arg(&s.rtv), arg(&source_size), type);
        s.fallback_rtv = s.rtv;
        s.rtv = destinations[ranges - 1];
    } else {
        invoke(device, 20, arg(s.target), arg(&view), s.rtv);
    }
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&s.queue)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)),
                                           arg(&s.allocator)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 73, arg(list_desc), arg(s.allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)),
                                           arg(&s.list)))) ||
        FAILED(static_cast<HRESULT>(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&s.fence)))))
        return 23;
    return 0;
}

int draw_frame(Scene& s, void* device, void* pso, UINT frame, UINT frames, bool slot_layouts)
{
    const float t = static_cast<float>(frame) / static_cast<float>(frames);
    const float colour[] = {t, 0.25f, 1.0f - t, 1.0f};
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(4 + (frame * 4) % (width - 4)), static_cast<LONG>(height)};
    if (FAILED(static_cast<HRESULT>(invoke(s.allocator, 8))) || FAILED(static_cast<HRESULT>(invoke(s.list, 139, arg(s.allocator)))))
        return 30;
    if (slot_layouts) {
        Word token = ~Word{0};
        if (invoke(device, 80, 0, 0, 0, 0, arg(&token), arg(&token)) != S_OK || token != frame + 1)
            return 34;
        // The capture probe suggests count and array. The stand-in must return zero
        // with a readable array and leave the rendered scene unchanged.
        Word barrier_like[8]{0x42};
        if (invoke(s.list, 146, 1, arg(barrier_like)) != 0) return 35;
        const Word upload = invoke(s.list, 148, 32, 16);
        if (!upload || (upload & 15)) return 36;
        const D3D12_RESOURCE_DESC resource{};
        const struct {
            D3D12_SHADER_RESOURCE_VIEW_DESC view;
            Word address;
        } memory_view{{}, upload};
        // Exercise the argument layout using a null destination, which refuses the descriptor write.
        invoke(device, 70, arg(&resource), arg(&memory_view), 0);
    }
    if (s.fallback_rtv) {
        const float fallback[] = {1.0f, 1.0f, 0.0f, 1.0f};
        invoke(s.list, 52, s.fallback_rtv, arg(fallback));
    }
    invoke(s.list, 52, s.rtv, arg(colour));
    invoke(s.list, 50, 1, arg(&s.rtv));
    invoke(s.list, 21, 1, arg(&viewport));
    invoke(s.list, 22, 1, arg(&scissor));
    invoke(s.list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    invoke(s.list, 25, arg(pso));
    invoke(s.list, 12, 3, 1);
    if (FAILED(static_cast<HRESULT>(invoke(s.list, 9)))) return 31;
    invoke(s.queue, 10, 1, arg(&s.list));
    // Plane record as queue_present reads it: +0x08 resource count, +0x10 pointer to the resource array.
    alignas(8) unsigned char plane[0x48]{};
    const Word resources[1]{arg(s.target)};
    const Word token = 7, one = 1, resources_at = arg(resources);
    std::memcpy(plane + 0x00, &token, 8);
    std::memcpy(plane + 0x08, &one, 8);
    std::memcpy(plane + 0x10, &resources_at, 8);
    invoke(s.queue, 32, 1, arg(plane), 0);
    const Word value = frame + 1;
    if (FAILED(static_cast<HRESULT>(invoke(s.queue, 14, arg(s.fence), value)))) return 32;
    for (const ULONGLONG deadline = GetTickCount64() + 10000; invoke(s.fence, 8) < value;) {
        if (GetTickCount64() > deadline) return 33;
        Sleep(1);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 6) return 2;
    const UINT frames = static_cast<UINT>(std::strtoul(argv[5], nullptr, 10));
    if (!frames || frames > 1000) return 2;
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    void* device = nullptr;
    void* pso = nullptr;
    if (const int rc = prepare_stream_pipeline(argv[1], argv[2], argv[3], std::strcmp(argv[4], "hardware") == 0, false, adapter,
                                               device, pso))
        return rc;
    Scene scene;
    if (const int rc = create_scene(device, scene)) return rc;
    // X2W_TEST_REPLAY_PAUSE=<frame>,<ms>: a pause before that frame (a slowdown for the recorder's "slow" trigger).
    char pause[32]{};
    unsigned pause_frame = ~0u, pause_ms = 0;
    if (GetEnvironmentVariableA("X2W_TEST_REPLAY_PAUSE", pause, sizeof(pause)) && std::sscanf(pause, "%u,%u", &pause_frame, &pause_ms) != 2)
        return 2;
    char layouts[2]{};
    const bool slot_layouts = GetEnvironmentVariableA("X2W_REPLAY_SLOT_LAYOUTS", layouts, sizeof(layouts)) == 1 && layouts[0] == '1';
    for (UINT frame = 0; frame < frames; ++frame) {
        if (frame == pause_frame) Sleep(pause_ms);
        if (const int rc = draw_frame(scene, device, pso, frame, frames, slot_layouts)) return rc;
    }
    std::printf("replay quad: %u presents drawn\n", frames);
    return 0;
}
