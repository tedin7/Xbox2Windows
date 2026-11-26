#include <windows.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
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
int draw_frame_copies(ID3D12Device* desktop, void* guest, void* list, UINT width, UINT height)
{
    char mode[16]{};
    GetEnvironmentVariableA("X2W_TEST_FRAME_COPY", mode, sizeof(mode));
    const bool big = !std::strncmp(mode, "big-", 4);
    const bool keys = !std::strcmp(mode, "once-keys");
    const bool once = keys || !std::strcmp(mode, "once");
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
        once ? large_bytes : big ? static_cast<UINT>(sizeof(constants)) : 256u};
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

int draw_frame_copy_threads(ID3D12Device* desktop, void* guest, void* pso, const char* pixels)
{
    static constexpr UINT workers = 4, draws = 1000, frames = 2;
    static constexpr UINT small_bytes = 4096, large_bytes = 16u << 20;
    const auto arg = [](const void* pointer) { return reinterpret_cast<StreamWord>(pointer); };
    const auto failed = [](StreamWord result) { return FAILED(static_cast<HRESULT>(result)); };
    struct Ranges {
        float* small = static_cast<float*>(VirtualAlloc(nullptr, small_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        float* large = static_cast<float*>(VirtualAlloc(nullptr, large_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        ~Ranges()
        {
            if (small) VirtualFree(small, 0, MEM_RELEASE);
            if (large) VirtualFree(large, 0, MEM_RELEASE);
        }
        bool write(unsigned color) const
        {
            if (!small || !large) return false;
            small[0] = (color & 1) ? 1.0f : 0.0f;
            small[1] = (color & 2) ? 1.0f : 0.0f;
            large[large_bytes / sizeof(float) - 2] = (color & 4) ? 1.0f : 0.0f;
            large[large_bytes / sizeof(float) - 1] = 1.0f;
            return true;
        }
    };
    std::array<Ranges, workers + 1> ranges;
    const UINT increment = desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    // The manifest's two records are at byte offsets 0 and 32 from a table base.
    if (increment != 32) return 81;
    alignas(32) std::array<unsigned char, (workers + 1) * 64> descriptors{};
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        StreamWord backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2 * (workers + 1), D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE},
                arg(descriptors.data()), descriptors.size()};
    void* heap = nullptr;
    if (failed(invoke(guest, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&heap)))) return 82;
    const auto cpu = invoke(heap, 9);
    const auto gpu = invoke(heap, 10);
    if (!gpu || cpu != arg(descriptors.data())) return 83;
    for (UINT pair = 0; pair <= workers; ++pair) {
        if (!ranges[pair].write(pair + 1)) return 84;
        const D3D12_CONSTANT_BUFFER_VIEW_DESC small{arg(ranges[pair].small), small_bytes};
        const D3D12_CONSTANT_BUFFER_VIEW_DESC large{arg(ranges[pair].large), large_bytes};
        invoke(guest, 17, arg(&small), cpu + pair * 64);
        invoke(guest, 17, arg(&large), cpu + pair * 64 + 32);
    }

    std::array<StreamTarget, workers> targets;
    std::array<void*, workers> lists{};
    for (UINT worker = 0; worker < workers; ++worker) {
        if (const int rc = open_stream_target(desktop, guest, pso, draws, frames, DXGI_FORMAT_R8G8B8A8_UNORM,
                                              draws, targets[worker])) return rc;
        lists[worker] = targets[worker].list;
    }
    auto& main = targets.front();
    const auto execute = [&](UINT count, UINT64 epoch) {
        invoke(main.queue, 10, count, arg(lists.data()));
        if (failed(invoke(main.queue, 14, arg(main.fence), epoch))) return false;
        const auto deadline = GetTickCount64() + 60000;
        while (invoke(main.fence, 8) < epoch && GetTickCount64() < deadline) Sleep(1);
        return invoke(main.fence, 8) == epoch;
    };
    const auto reset = [&](StreamTarget& target) {
        if (failed(invoke(target.allocator, 8)) || failed(invoke(target.list, 139, arg(target.allocator)))) return false;
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(target.rtv_heap, 9))};
        const D3D12_VIEWPORT viewport{0, 0, float(draws), float(frames), 0, 1};
        invoke(target.list, 50, 1, arg(&rtv));
        invoke(target.list, 21, 1, arg(&viewport));
        invoke(target.list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        invoke(target.list, 25, arg(pso));
        return true;
    };
    const auto bind_heap = [&](void* list) { invoke(list, 32, 1, arg(&heap)); };
    // Warm the translation and the PSO before the synchronized first cache
    // miss. An empty scissor leaves every measured pixel for the real draws.
    bind_heap(main.list);
    const D3D12_RECT empty{};
    invoke(main.list, 22, 1, arg(&empty));
    invoke(main.list, 36, 0, gpu + 64);
    invoke(main.list, 12, 3, 1);
    if (failed(invoke(main.list, 9)) || !execute(1, 1)) return 85;
    invoke(main.queue, 32, 0, 0, 0);
    if (!reset(main)) return 86;

    char serial_mode[2]{};
    const bool serial = GetEnvironmentVariableA("X2W_TEST_DRAW_WALK_SERIAL", serial_mode, sizeof(serial_mode)) == 1 &&
                        serial_mode[0] == '1';
    for (UINT frame = 0; frame < frames; ++frame) {
        const auto record = [&](UINT worker) {
            void* list = lists[worker];
            bind_heap(list);
            for (UINT draw = 0; draw < draws; ++draw) {
                const LONG x = static_cast<LONG>(draw), y = static_cast<LONG>(frame);
                const D3D12_RECT pixel{x, y, x + 1, y + 1};
                invoke(list, 22, 1, arg(&pixel));
                invoke(list, 36, 0, gpu + (draw % 2 ? (worker + 1) * 64 : 0));
                invoke(list, 12, 3, 1);
            }
        };
        if (serial) {
            for (UINT worker = 0; worker < workers; ++worker) record(worker);
        } else {
            std::array<std::thread, workers> threads;
            for (UINT worker = 0; worker < workers; ++worker) threads[worker] = std::thread(record, worker);
            for (auto& thread : threads) thread.join();
        }
        for (auto* list : lists)
            if (failed(invoke(list, 9))) return 87;
        if (!execute(workers, frame + 2)) return 88;
        invoke(main.queue, 32, 0, 0, 0);
        if (frame + 1 < frames) {
            ranges[0].write(6);
            for (auto& target : targets)
                if (!reset(target)) return 89;
        }
    }

    std::ofstream combined(pixels, std::ios::binary);
    for (UINT worker = 0; worker < workers; ++worker) {
        const std::string part = std::string(pixels) + "." + std::to_string(worker);
        if (const int rc = read_stream_target(desktop, targets[worker], part.c_str())) return rc;
        std::ifstream input(part, std::ios::binary);
        combined << input.rdbuf();
    }
    return combined ? 0 : 90;
}
