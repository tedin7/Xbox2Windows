#include <windows.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "startup_stream_setup.h"

// A present on a second queue advances the frame without executing the recorded draws.
static void* create_present_queue(void* guest)
{
    void* queue = nullptr;
    const D3D12_COMMAND_QUEUE_DESC desc{};
    if (FAILED(static_cast<HRESULT>(invoke(guest, 8, reinterpret_cast<std::uintptr_t>(&desc),
                                           reinterpret_cast<std::uintptr_t>(&__uuidof(ID3D12CommandQueue)),
                                           reinterpret_cast<std::uintptr_t>(&queue))))) return nullptr;
    return queue;
}

// Each list draws two halves of its own target. Both lists are closed before
// submission, so the second has no ordering dependency it can use to bind the mirror.
static int draw_mirror_two_lists(ID3D12Device* desktop, void* guest, StreamTarget& first, void* pso,
                                 UINT width, UINT height, const char* pixels, void* heap, StreamWord table)
{
    StreamTarget second{};
    if (const int result = open_stream_target(desktop, guest, pso, width, height, DXGI_FORMAT_R8G8B8A8_UNORM,
                                              width, second)) return result;
    void* lists[]{first.list, second.list};
    for (auto* list : lists) {
        invoke(list, 32, 1, reinterpret_cast<StreamWord>(&heap));
        invoke(list, 36, 0, table);
        for (UINT half = 0; half < 2; ++half) {
            const D3D12_RECT rect{static_cast<LONG>(half * width / 2), 0,
                                  static_cast<LONG>((half + 1) * width / 2), static_cast<LONG>(height)};
            invoke(list, 22, 1, reinterpret_cast<StreamWord>(&rect));
            invoke(list, 12, 3, 1);
        }
        if (FAILED(static_cast<HRESULT>(invoke(list, 9)))) return 47;
    }
    invoke(first.queue, 10, 2, reinterpret_cast<StreamWord>(lists));
    if (FAILED(static_cast<HRESULT>(invoke(first.queue, 14, reinterpret_cast<StreamWord>(first.fence), 1)))) return 48;
    const auto deadline = GetTickCount64() + 10000;
    while (invoke(first.fence, 8) < 1 && GetTickCount64() < deadline) Sleep(1);
    if (invoke(first.fence, 8) != 1) return 49;
    const std::string first_path = std::string(pixels) + ".0";
    if (const int result = read_stream_target(desktop, first, first_path.c_str())) return result;
    if (const int result = read_stream_target(desktop, second, pixels)) return result;
    invoke(first.queue, 32, 0, 0, 0);
    return 0;
}

// Two draws in one command list read the same constant buffer address through a descriptor table; the left half
// of the target is drawn first, the right half second. X2W_TEST_FRAME_COPY selects what the title does to the
// memory between the draws:
//   changed  the colour is rewritten in place (red, then blue)
//   same     the memory is left alone (red both times)
//   big-changed / big-same  as above with a 64 KiB buffer (a "large" copy, kept across frames), and a present between the draws, so
//            the second draw is a later frame: a copy kept from the first frame must still show the title's new bytes
//   once     four draws of a 16 MiB binding per frame, with a rewrite between frames
//   once-mirror  two draws and a readback per frame for three separately submitted 32 MiB snapshots
//   once-keys  one frame, two views of the SAME address: 64 KiB and 16 MiB. The words sit at the start (zero) and at the last 16
//            bytes of the 16 MiB (red), so a draw through the 64 KiB view reads zero (the tail is past its end) and a draw through
//            the 16 MiB view reads red. Draws alternate small, large, small, large: a cache keyed by address alone would hand
//            the second view the first view's bytes.
int draw_frame_copies(ID3D12Device* desktop, void* guest, StreamTarget& target, void* pso,
                      UINT width, UINT height, const char* pixels, const char* mode)
{
    void* list = target.list;
    const bool big = !std::strncmp(mode, "big-", 4);
    const bool keys = !std::strcmp(mode, "once-keys");
    const bool mirror = !std::strcmp(mode, "once-mirror");
    const bool two_lists = !std::strcmp(mode, "once-mirror-two");
    const bool once = two_lists || mirror || keys || !std::strcmp(mode, "once");
    const bool changed = !std::strcmp(mode + (big ? 4 : 0), "changed");
    if (!once && !changed && std::strcmp(mode + (big ? 4 : 0), "same")) return 2;

    // 64 KiB: every word after the colour is non-zero, so no zero tail is trimmed and the copy stays a large one.
    alignas(256) static float constants[16384] = {1.0f, 0.0f, 0.0f, 1.0f};
    if (big)
        for (int i = 4; i < 16384; ++i) constants[i] = 0.5f;
    const UINT increment = desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT entries = keys ? 2 : (32 + increment - 1) / increment;
    static std::vector<unsigned char> views;
    views.assign(static_cast<std::size_t>(entries) * increment, 0);
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        StreamWord backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, entries, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE},
                reinterpret_cast<std::uintptr_t>(views.data()), views.size()};
    void* heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, reinterpret_cast<std::uintptr_t>(&heap_desc),
                                           reinterpret_cast<std::uintptr_t>(&__uuidof(ID3D12DescriptorHeap)),
                                           reinterpret_cast<std::uintptr_t>(&heap))))) return 41;
    const StreamWord cpu = invoke(heap, 9);
    if (cpu != reinterpret_cast<std::uintptr_t>(views.data())) return 42;
    const UINT large_bytes = (two_lists ? 96u : mirror ? 32u : 16u) << 20;
    float* large = nullptr;
    const UINT large_words = large_bytes / sizeof(float);
    if (once) {
        large = static_cast<float*>(VirtualAlloc(nullptr, large_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!large) return 45;
        if (two_lists) {
            auto* bytes = reinterpret_cast<unsigned char*>(large);
            for (UINT block = 0; block < 256; ++block) {
                const UINT marker = 5u * block + 13u;
                std::memcpy(bytes + block * (64u << 10) + 20, &marker, sizeof(marker));
            }
            bytes[(16u << 20) - 1] = 1;
            bytes[large_bytes - 1] = 1;
        } else if (mirror) {
            constexpr UINT block_bytes = 64u << 10;
            for (UINT block = 0; block < large_bytes / block_bytes; ++block)
                large[block * (block_bytes / sizeof(float)) + 1] = static_cast<float>(block + 1);
            large[large_words - 1] = 1.0f;  // Alpha and the nonzero tail in every frame.
        } else {
            if (!keys) std::memcpy(large, constants, 4 * sizeof(float));
            std::memcpy(large + large_words - 4, constants, 4 * sizeof(float));
        }
    }
    const auto* memory = once ? large : constants;
    const D3D12_CONSTANT_BUFFER_VIEW_DESC view{reinterpret_cast<std::uintptr_t>(memory),
        once ? large_bytes : big ? static_cast<UINT>(sizeof(constants)) : 256u};
    invoke(guest, 17, reinterpret_cast<std::uintptr_t>(&view), cpu);  // CreateConstantBufferView
    if (keys) {
        // The second view: the same address, 64 KiB long.
        const D3D12_CONSTANT_BUFFER_VIEW_DESC small_view{reinterpret_cast<std::uintptr_t>(memory), 64u << 10};
        invoke(guest, 17, reinterpret_cast<std::uintptr_t>(&small_view), cpu + increment);
    }
    const StreamWord table = invoke(heap, 10);
    if (!table) return 43;

    if (two_lists) {
        const int result = draw_mirror_two_lists(desktop, guest, target, pso, width, height, pixels, heap, table);
        VirtualFree(large, 0, MEM_RELEASE);
        return result;
    }
    invoke(list, 32, 1, reinterpret_cast<std::uintptr_t>(&heap));  // SetDescriptorHeaps
    invoke(list, 36, 0, table);                                    // SetGraphicsRootDescriptorTable
    if (mirror) {
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(target.rtv_heap, 9))};
        const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
        const UINT edits[]{0, (11u << 20) / sizeof(float), large_words - 2};
        for (UINT frame = 0; frame < 3; ++frame) {
            if (frame) {
                if (FAILED(static_cast<HRESULT>(invoke(target.allocator, 8))) ||
                    FAILED(static_cast<HRESULT>(invoke(list, 139, reinterpret_cast<StreamWord>(target.allocator))))) return 46;
                const float clear[]{0.0f, 0.0f, 1.0f, 1.0f};
                invoke(list, 52, rtv.ptr, reinterpret_cast<StreamWord>(clear));
                invoke(list, 50, 1, reinterpret_cast<StreamWord>(&rtv));
                invoke(list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                invoke(list, 25, reinterpret_cast<StreamWord>(pso));
                invoke(list, 32, 1, reinterpret_cast<StreamWord>(&heap));
                invoke(list, 36, 0, table);
            }
            large[edits[frame]] = 1.0f;
            invoke(list, 21, 1, reinterpret_cast<StreamWord>(&viewport));
            for (UINT half = 0; half < 2; ++half) {
                const D3D12_RECT rect{static_cast<LONG>(half * width / 2), 0,
                                      static_cast<LONG>((half + 1) * width / 2), static_cast<LONG>(height)};
                invoke(list, 22, 1, reinterpret_cast<StreamWord>(&rect));
                invoke(list, 12, 3, 1);
            }
            if (FAILED(static_cast<HRESULT>(invoke(list, 9)))) return 47;
            invoke(target.queue, 10, 1, reinterpret_cast<StreamWord>(&list));
            const UINT64 fence_value = frame + 1;
            if (FAILED(static_cast<HRESULT>(invoke(target.queue, 14, reinterpret_cast<StreamWord>(target.fence), fence_value))))
                return 48;
            const auto deadline = GetTickCount64() + 10000;
            while (invoke(target.fence, 8) < fence_value && GetTickCount64() < deadline) Sleep(1);
            if (invoke(target.fence, 8) != fence_value) return 49;
            const std::string path = frame == 2 ? pixels : std::string(pixels) + "." + std::to_string(frame);
            if (const int result = read_stream_target(desktop, target, path.c_str())) return result;
            invoke(target.queue, 32, 0, 0, 0);
        }
        VirtualFree(large, 0, MEM_RELEASE);
        return 0;
    }
    if (keys) {
        const StreamWord large_table = table, small_table = table + increment;
        for (UINT draw = 0; draw < 4; ++draw) {
            invoke(list, 36, 0, draw % 2 ? large_table : small_table);  // SetGraphicsRootDescriptorTable
            const LONG x = static_cast<LONG>(draw * width / 4), end = static_cast<LONG>((draw + 1) * width / 4);
            const D3D12_RECT column{x, 0, end, static_cast<LONG>(height)};
            invoke(list, 22, 1, reinterpret_cast<std::uintptr_t>(&column));
            invoke(list, 12, 3, 1);
        }
        void* queue = create_present_queue(guest);
        if (!queue) return 44;
        invoke(queue, 32, 0, 0, 0);
        VirtualFree(large, 0, MEM_RELEASE);
        return 0;
    }
    if (once) {
        void* queue = create_present_queue(guest);
        if (!queue) return 44;
        for (UINT frame = 0; frame < 2; ++frame) {
            // Draw one column at a time so readback exercises all eight bindings.
            for (UINT draw = 0; draw < 4; ++draw) {
                const LONG x = static_cast<LONG>((frame * 4 + draw) * width / 8);
                const LONG end = static_cast<LONG>((frame * 4 + draw + 1) * width / 8);
                const D3D12_RECT column{x, 0, end, static_cast<LONG>(height)};
                invoke(list, 22, 1, reinterpret_cast<std::uintptr_t>(&column));
                invoke(list, 12, 3, 1);
            }
            invoke(queue, 32, 0, 0, 0);
            if (frame == 0) {
                large[0] = large[large_words - 4] = 0.0f;
                large[2] = large[large_words - 2] = 1.0f;
            }
        }
        VirtualFree(large, 0, MEM_RELEASE);
        return 0;
    }
    invoke(list, 12, 3, 1);                                        // draw into the left half
    if (changed) {
        constants[0] = 0.0f;  // the same address now holds blue
        constants[2] = 1.0f;
    }
    if (big) {
        void* queue = create_present_queue(guest);
        if (!queue) return 44;
        invoke(queue, 32, 0, 0, 0);
    }
    const D3D12_RECT right{static_cast<LONG>(width / 2), 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    invoke(list, 22, 1, reinterpret_cast<std::uintptr_t>(&right));  // RSSetScissorRects
    invoke(list, 12, 3, 1);                                         // draw into the right half
    return 0;
}
