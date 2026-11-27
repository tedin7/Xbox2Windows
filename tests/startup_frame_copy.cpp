#include <windows.h>

#include <array>
#include <cstdint>
#include <cstring>
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

// Two draws in one command list read the same constant buffer address through a descriptor table; the left half
// of the target is drawn first, the right half second. X2W_TEST_FRAME_COPY selects what the title does to the
// memory between the draws:
//   changed  the colour is rewritten in place (red, then blue)
//   same     the memory is left alone (red both times)
//   big-changed / big-same  as above with a 64 KiB buffer (a "large" copy, kept across frames), and a present between the draws, so
//            the second draw is a later frame: a copy kept from the first frame must still show the title's new bytes
//   once     four draws of a 16 MiB binding per frame, with a rewrite between frames
//   once-keys  one frame, two views of the SAME address: 64 KiB and 16 MiB. The words sit at the start (zero) and at the last 16
//            bytes of the 16 MiB (red), so a draw through the 64 KiB view reads zero (the tail is past its end) and a draw through
//            the 16 MiB view reads red. Draws alternate small, large, small, large: a cache keyed by address alone would hand
//            the second view the first view's bytes.
//   ring-control / ring-full  fill one chunk with one 64 KiB and six 32 KiB copies. The full case asks for
//            another 32 KiB with the same recording still open and checks that the refusal leaves its column red.
int draw_frame_copies(ID3D12Device* desktop, void* guest, void* list, UINT width, UINT height)
{
    char mode[16]{};
    GetEnvironmentVariableA("X2W_TEST_FRAME_COPY", mode, sizeof(mode));
    const bool ring_full = !std::strcmp(mode, "ring-full");
    const bool ring_case = ring_full || !std::strcmp(mode, "ring-control");
    const bool big = !std::strncmp(mode, "big-", 4);
    const bool keys = !std::strcmp(mode, "once-keys");
    const bool once = keys || !std::strcmp(mode, "once");
    const bool changed = !std::strcmp(mode + (big ? 4 : 0), "changed");
    if (!ring_case && !once && !changed && std::strcmp(mode + (big ? 4 : 0), "same")) return 2;

    // 64 KiB: every word after the colour is non-zero, so no zero tail is trimmed and the copy stays a large one.
    alignas(256) static float constants[16384] = {1.0f, 0.0f, 0.0f, 1.0f};
    if (big || ring_case)
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
    constexpr UINT large_bytes = 16u << 20;
    float* large = nullptr;
    constexpr UINT large_words = large_bytes / sizeof(float);
    if (once) {
        large = static_cast<float*>(VirtualAlloc(nullptr, large_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!large) return 45;
        if (!keys) std::memcpy(large, constants, 4 * sizeof(float));
        std::memcpy(large + large_words - 4, constants, 4 * sizeof(float));
    }
    const auto* memory = once ? large : constants;
    const D3D12_CONSTANT_BUFFER_VIEW_DESC view{reinterpret_cast<std::uintptr_t>(memory),
        once ? large_bytes : (big || ring_case) ? static_cast<UINT>(sizeof(constants)) : 256u};
    invoke(guest, 17, reinterpret_cast<std::uintptr_t>(&view), cpu);  // CreateConstantBufferView
    if (keys) {
        // The second view: the same address, 64 KiB long.
        const D3D12_CONSTANT_BUFFER_VIEW_DESC small_view{reinterpret_cast<std::uintptr_t>(memory), 64u << 10};
        invoke(guest, 17, reinterpret_cast<std::uintptr_t>(&small_view), cpu + increment);
    }
    const StreamWord table = invoke(heap, 10);
    if (!table) return 43;

    invoke(list, 32, 1, reinterpret_cast<std::uintptr_t>(&heap));  // SetDescriptorHeaps
    invoke(list, 36, 0, table);                                    // SetGraphicsRootDescriptorTable
    if (ring_case) {
        using ReportUpload = void (*)(BOOL);
        const auto report = reinterpret_cast<ReportUpload>(
            GetProcAddress(GetModuleHandleW(L"d3d12_xs.dll"), "x2w_d3d12_upload_diagnostics"));
        if (!report) return 46;
        if (GetEnvironmentVariableW(L"X2W_TEST_UPLOAD_CONSUMER_OTHER", nullptr, 0)) report(TRUE);

        const D3D12_RECT all{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
        invoke(list, 22, 1, reinterpret_cast<std::uintptr_t>(&all));
        invoke(list, 12, 3, 1);  // 64 KiB of red, including a nonzero tail.

        alignas(256) static float small[8192]{};
        const D3D12_CONSTANT_BUFFER_VIEW_DESC small_view{reinterpret_cast<std::uintptr_t>(small), sizeof(small)};
        invoke(guest, 17, reinterpret_cast<std::uintptr_t>(&small_view), cpu);
        const UINT small_draws = ring_full ? 7 : 6;
        for (UINT draw = 0; draw < small_draws; ++draw) {
            small[0] = draw % 2 ? 1.0f : 0.0f;
            small[2] = draw % 2 ? 0.0f : 1.0f;
            small[3] = 1.0f;
            // Six 32 KiB replacements fill the rest. The seventh would overwrite column 0.
            const LONG x = draw == 6 ? 0 : static_cast<LONG>(draw + 1);
            const D3D12_RECT column{x, 0, x + 1, static_cast<LONG>(height)};
            invoke(list, 22, 1, reinterpret_cast<std::uintptr_t>(&column));
            invoke(list, 12, 3, 1);
        }
        report(FALSE);  // The fixture caller closes, submits and reads back after this sample.
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
