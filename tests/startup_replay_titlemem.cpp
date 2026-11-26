// A replay scene whose picture depends on TITLE MEMORY that the title rewrites between presents (PAGE/MAP replay test,
// tests/startup_replay_titlemem_check.py).
//
// Frame f: the "title" writes four floats (a colour that moves with f) into a page of its own memory, binds that address as
// root constant buffer 0 (SetGraphicsRootConstantBufferView, list slot 42), draws the stream pipeline's triangle with the
// translated pixel shader that returns those four floats, presents, signals and waits. The render target is cleared to the
// same colour every frame, so the only thing that changes the picture is the bytes at the title's address: a replay that
// does not restore the page at that address (and the change between presents) draws another colour.
//
// usage: startup_replay_titlemem <vs> <ps> <root> [<warp|hardware> <presents>]   (three arguments: one frame on WARP, to publish
// the shader request); run with X2W_TEST_ROOT_CBV=bound so the stream root signature has the constant buffer parameter.
#include <windows.h>

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
    void* title_buffer = nullptr;   // a placed buffer over the title's page (device slot 133): the recorder knows the range
    void* queue = nullptr;
    void* allocator = nullptr;
    void* list = nullptr;
    void* fence = nullptr;
    SIZE_T rtv = 0;
    float* memory = nullptr;   // the title's page: the colour lives at its start
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
    s.rtv = static_cast<SIZE_T>(invoke(s.rtv_heap, 9));
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
    // One 64 KiB page of the title's own memory, 64 KiB aligned, so the descriptor's address is a page the recorder tracks.
    s.memory = static_cast<float*>(VirtualAlloc(nullptr, 1u << 16, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!s.memory) return 24;
    // The buffer is placed over that page once, before the window; the title never names the page again except as the
    // constant buffer address, so what a replay restores comes from the recorder's page scan of this resource.
    const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 1u << 16, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                     D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    if (FAILED(static_cast<HRESULT>(invoke(device, 133, arg(s.memory), arg(&buffer), D3D12_RESOURCE_STATE_COMMON, 0,
                                           arg(&__uuidof(ID3D12Resource)), arg(&s.title_buffer)))) || !s.title_buffer)
        return 25;
    return 0;
}

int draw_frame(Scene& s, void* pso, UINT frame, UINT frames)
{
    const float t = static_cast<float>(frame) / static_cast<float>(frames);
    // The title writes the colour into its memory between presents; nothing else tells the stand-in about it.
    const float colour[4] = {t, 0.25f, 1.0f - t, 1.0f};
    std::memcpy(s.memory, colour, sizeof(colour));
    const float black[] = {0.0f, 0.0f, 0.0f, 1.0f};
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    if (FAILED(static_cast<HRESULT>(invoke(s.allocator, 8))) || FAILED(static_cast<HRESULT>(invoke(s.list, 139, arg(s.allocator)))))
        return 30;
    invoke(s.list, 52, s.rtv, arg(black));
    invoke(s.list, 50, 1, arg(&s.rtv));
    invoke(s.list, 21, 1, arg(&viewport));
    invoke(s.list, 22, 1, arg(&scissor));
    invoke(s.list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    invoke(s.list, 34, reinterpret_cast<Word>(stream_root_signature()));   // SetGraphicsRootSignature
    invoke(s.list, 25, arg(pso));
    invoke(s.list, 42, 0, reinterpret_cast<Word>(s.memory));               // SetGraphicsRootConstantBufferView(0, title address)
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
    if (argc != 4 && argc != 6) return 2;
    const bool hardware = argc == 6 && std::strcmp(argv[4], "hardware") == 0;
    const UINT frames = argc == 6 ? static_cast<UINT>(std::strtoul(argv[5], nullptr, 10)) : 1;
    if (!frames || frames > 1000) return 2;
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    void* device = nullptr;
    void* pso = nullptr;
    if (const int rc = prepare_stream_pipeline(argv[1], argv[2], argv[3], hardware, false, adapter, device, pso)) return rc;
    Scene scene;
    if (const int rc = create_scene(device, scene)) return rc;
    for (UINT frame = 0; frame < frames; ++frame)
        if (const int rc = draw_frame(scene, pso, frame, frames)) return rc;
    std::printf("replay pages: %u presents drawn\n", frames);
    return 0;
}
