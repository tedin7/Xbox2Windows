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
    SIZE_T rtv = 0;
};

int create_scene(void* device, Scene& s)
{
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                   {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 1}};
    // X2W_TEST_REPLAY_CREATE127=1: the target comes from device slot 127 (the console CreateCommittedResource: ten
    // arguments after the device, the output last), so a recording must carry more than seven argument words.
    if (GetEnvironmentVariableA("X2W_TEST_REPLAY_CREATE127", nullptr, 0)) {
        using Create = HRESULT (*)(void*, const void*, UINT, const void*, UINT, const void*, void*, UINT, const void*, const GUID*, void**);
        const auto create = reinterpret_cast<Create>((*static_cast<void***>(device))[127]);
        if (FAILED(create(device, &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, &clear, nullptr, 0, nullptr,
                          &__uuidof(ID3D12Resource), &s.target)) || !s.target)
            return 25;
    } else if (FAILED(static_cast<HRESULT>(invoke(device, 27, arg(&heap), D3D12_HEAP_FLAG_NONE, arg(&desc), D3D12_RESOURCE_STATE_COMMON,
                                                  arg(&clear), arg(&__uuidof(ID3D12Resource)), arg(&s.target)))) || !s.target) {
        return 21;
    }
    // The title's heap descriptor: the desktop one, then a title CPU address and its size (none here).
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } rtv_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, arg(&rtv_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&s.rtv_heap)))))
        return 22;
    // The heap start read as a title compiled for struct returns reads it: (return buffer, object).
    using Start = SIZE_T* (*)(SIZE_T*, void*);
    const auto start = reinterpret_cast<Start>((*static_cast<void***>(s.rtv_heap))[9]);
    if (start(&s.rtv, s.rtv_heap) != &s.rtv || !s.rtv) return 24;
    const D3D12_RENDER_TARGET_VIEW_DESC view{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    invoke(device, 20, arg(s.target), arg(&view), s.rtv);
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

int draw_frame(Scene& s, void* pso, UINT frame, UINT frames)
{
    const float t = static_cast<float>(frame) / static_cast<float>(frames);
    const float colour[] = {t, 0.25f, 1.0f - t, 1.0f};
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(4 + (frame * 4) % (width - 4)), static_cast<LONG>(height)};
    if (FAILED(static_cast<HRESULT>(invoke(s.allocator, 8))) || FAILED(static_cast<HRESULT>(invoke(s.list, 139, arg(s.allocator)))))
        return 30;
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
    for (UINT frame = 0; frame < frames; ++frame) {
        if (frame == pause_frame) Sleep(pause_ms);
        if (const int rc = draw_frame(scene, pso, frame, frames)) return rc;
    }
    std::printf("replay quad: %u presents drawn\n", frames);
    return 0;
}
