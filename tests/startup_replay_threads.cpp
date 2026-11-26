// Two threads drawing through the D3D12 stand-in at once, for the replay test (tests/startup_replay_threads_check.py).
//
// Each thread has its own target, queue, allocator and list. They take turns through one shared fence S: thread A draws
// frame f after S reaches 2f, presents, then signals S = 2f + 1; thread B draws frame f after S reaches 2f + 1, presents,
// then signals S = 2f + 2. A thread learns of the other only through the fence's completed value (a call the recording
// holds), so a recording of this process shows the order across threads, and the pictures show it too: present 2f + 1
// is A's frame f (reddish) and present 2f + 2 is B's (bluish).
//
// usage: startup_replay_threads <vs> <ps> <root> <warp|hardware> <frames per thread>
#include <d3d12.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

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
    SIZE_T rtv = 0;
};

int create_scene(void* device, Scene& s)
{
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
                                   {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 1}};
    if (FAILED(static_cast<HRESULT>(invoke(device, 27, arg(&heap), D3D12_HEAP_FLAG_NONE, arg(&desc), D3D12_RESOURCE_STATE_COMMON,
                                           arg(&clear), arg(&__uuidof(ID3D12Resource)), arg(&s.target)))) || !s.target)
        return 21;
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } rtv_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, arg(&rtv_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&s.rtv_heap)))))
        return 22;
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
                                           arg(&s.list)))))
        return 23;
    return 0;
}

// Waits until the shared fence has reached `value`.
bool wait_for(void* fence, std::uint64_t value)
{
    for (const ULONGLONG deadline = GetTickCount64() + 20000; invoke(fence, 8) < value;) {
        if (GetTickCount64() > deadline) return false;
        Sleep(1);
    }
    return true;
}

// One frame: clear to a colour of this thread, draw the triangle, execute, present the target.
int draw_frame(Scene& s, void* pso, UINT frame, UINT frames, bool first)
{
    const float t = static_cast<float>(frame) / static_cast<float>(frames);
    const float colour[] = {first ? 1.0f - 0.5f * t : 0.0f, 0.25f, first ? 0.0f : 0.5f + 0.5f * t, 1.0f};
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
    alignas(8) unsigned char plane[0x48]{};
    const Word resources[1]{arg(s.target)};
    const Word token = 7, one = 1, resources_at = arg(resources);
    std::memcpy(plane + 0x00, &token, 8);
    std::memcpy(plane + 0x08, &one, 8);
    std::memcpy(plane + 0x10, &resources_at, 8);
    invoke(s.queue, 32, 1, arg(plane), 0);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 6) return 2;
    const UINT frames = static_cast<UINT>(std::strtoul(argv[5], nullptr, 10));
    if (!frames || frames > 500) return 2;
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    void* device = nullptr;
    void* pso = nullptr;
    if (const int rc = prepare_stream_pipeline(argv[1], argv[2], argv[3], std::strcmp(argv[4], "hardware") == 0, false, adapter,
                                               device, pso))
        return rc;
    Scene scenes[2];
    for (Scene& scene : scenes)
        if (const int rc = create_scene(device, scene)) return rc;
    void* shared = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&shared)))) || !shared) return 25;
    std::atomic<int> result[2]{0, 0};
    std::thread threads[2];
    for (int who = 0; who < 2; ++who)
        threads[who] = std::thread([&, who] {
            for (UINT frame = 0; frame < frames; ++frame) {
                if (!wait_for(shared, 2ull * frame + who)) {
                    result[who] = 40 + who;
                    return;
                }
                if (const int rc = draw_frame(scenes[who], pso, frame, frames, who == 0)) {
                    result[who] = rc;
                    return;
                }
                if (FAILED(static_cast<HRESULT>(invoke(scenes[who].queue, 14, arg(shared), 2ull * frame + who + 1)))) {
                    result[who] = 32;
                    return;
                }
            }
        });
    for (std::thread& t : threads) t.join();
    if (result[0] || result[1]) return result[0] ? result[0].load() : result[1].load();
    if (!wait_for(shared, 2ull * frames)) return 33;
    std::printf("replay threads: %u presents drawn on 2 threads\n", 2 * frames);
    return 0;
}
