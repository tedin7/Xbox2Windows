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
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "startup_stream_setup.h"

extern "C" __declspec(dllimport) std::uint64_t x2w_d3d12_replay_record_calls();

using Word = StreamWord;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }

namespace {

constexpr UINT width = 64, height = 32;
constexpr UINT colour_offset = 4096, cbv_bytes = 8192;
enum class TitleRange { none, root, generated, supplied };

struct Scene {
    void* target = nullptr;
    void* rtv_heap = nullptr;
    void* queue = nullptr;
    void* allocator = nullptr;
    void* list = nullptr;
    void* fence = nullptr;
    SIZE_T rtv = 0, fallback_rtv = 0;
    TitleRange title_range = TitleRange::none;
    unsigned char* constants = nullptr;
    void* view_heap = nullptr;
    Word view_cpu = 0, view_gpu = 0;
    UINT view_stride = 0;
    alignas(64) unsigned char title_rtvs[512]{};
    void* signature = nullptr;
    void* arguments = nullptr;
};

LONG volatile access_exceptions;

LONG CALLBACK count_access_exceptions(EXCEPTION_POINTERS* exception)
{
    const DWORD code = exception->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_GUARD_PAGE)
        InterlockedIncrement(&access_exceptions);
    return EXCEPTION_CONTINUE_SEARCH;
}

bool record_memory_cases(void* device)
{
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    auto* reserved = static_cast<unsigned char*>(VirtualAlloc(nullptr, system.dwAllocationGranularity, MEM_RESERVE, PAGE_NOACCESS));
    constexpr std::size_t large_size = 2u << 20;
    auto* large = static_cast<unsigned char*>(VirtualAlloc(nullptr, large_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    void* committed = reserved ? VirtualAlloc(reserved + system.dwPageSize, system.dwPageSize, MEM_COMMIT, PAGE_READWRITE) : nullptr;
    void* handler = AddVectoredExceptionHandler(1, count_access_exceptions);
    bool ok = reserved && large && committed && handler;
    if (ok) {
        // The unsupported IID keeps the real slot from consuming the blob.
        // The recorder sees a declared pointer and size in every call.
        constexpr UINT marker = 0x11223344;
        const GUID unsupported{};
        void* output = nullptr;
        auto root = [&](const void* blob, std::size_t bytes, Word tail) {
            return static_cast<HRESULT>(invoke(device, 16, marker, arg(blob), bytes, arg(&unsupported), arg(&output), tail,
                                               large_size)) == E_NOINTERFACE;
        };
        ok = root(reserved, 64, arg(large)) && root(committed, 64, arg(reserved));
        // Reserved when first named, committed and written before it is named again: the second call holds it.
        auto* late = reserved + 2 * system.dwPageSize;
        ok = ok && root(late, 64, 0) && VirtualAlloc(late, system.dwPageSize, MEM_COMMIT, PAGE_READWRITE);
        if (ok) std::memset(late, 0x5a, 64);
        ok = ok && root(late, 64, 0);
        for (unsigned i = 0; i < 20 && ok; ++i) {
            large[0] = static_cast<unsigned char>(i);
            ok = root(large, large_size, arg(reserved));
        }
        // Individually small ranges can still exceed the same call's remaining budget.
        constexpr std::size_t shader_bytes = 600u << 10;
        struct alignas(8) Shader {
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type;
            D3D12_SHADER_BYTECODE bytecode;
        } shaders[] = {
            {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, {large, shader_bytes}},
            {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, {large + shader_bytes, shader_bytes}},
        };
        const D3D12_PIPELINE_STATE_STREAM_DESC stream{sizeof(shaders), shaders};
        ok = ok && static_cast<HRESULT>(invoke(device, 86, arg(&stream), arg(&unsupported), arg(&output),
                                              marker)) == E_NOINTERFACE;
        invoke(device, 1, arg(reserved));
        invoke(device, 2, arg(reserved));
    }
    if (handler) RemoveVectoredExceptionHandler(handler);
    if (large) VirtualFree(large, 0, MEM_RELEASE);
    if (reserved) VirtualFree(reserved, 0, MEM_RELEASE);
    std::printf("replay quad: recorder access exceptions %ld\n", access_exceptions);
    return ok;
}

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
    // X2W_TEST_REPLAY_CREATE133=1: the target is placed at a title address by device slot 133 (CreatePlacedResourceX:
    // address, desc, state, clear, IID, out) with no desktop heap placement, the way a title makes most of its
    // resources. A window that opens after it must still carry its description, IID and output.
    if (GetEnvironmentVariableA("X2W_TEST_REPLAY_CREATE133", nullptr, 0)) {
        constexpr SIZE_T bytes = 64 * 1024;
        static_assert(width * height * 4 <= bytes, "the title memory holds the target");
        void* memory = VirtualAlloc(reinterpret_cast<void*>(0x21000000000ull), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!memory || FAILED(static_cast<HRESULT>(invoke(device, 133, arg(memory), arg(&desc), D3D12_RESOURCE_STATE_COMMON,
                                                          arg(&clear), arg(&__uuidof(ID3D12Resource)), arg(&s.target)))) ||
            !s.target)
            return 29;
    } else if (GetEnvironmentVariableA("X2W_TEST_REPLAY_CREATE127", nullptr, 0)) {
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

int create_indirect_arguments(void* device, Scene& s)
{
    const D3D12_INDIRECT_ARGUMENT_DESC argument{D3D12_INDIRECT_ARGUMENT_TYPE_DRAW};
    const D3D12_COMMAND_SIGNATURE_DESC signature{sizeof(D3D12_DRAW_ARGUMENTS), 1, &argument, 0};
    if (FAILED(static_cast<HRESULT>(invoke(device, 41, arg(&signature), 0,
                                           arg(&__uuidof(ID3D12CommandSignature)), arg(&s.signature)))))
        return 26;
    constexpr UINT bytes = 64 * 1024;
    // Keep the address away from the image and usual process allocations so the player can reserve it.
    auto* words = static_cast<D3D12_DRAW_ARGUMENTS*>(
        VirtualAlloc(reinterpret_cast<void*>(0x20000000000ull), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!words) return 27;
    const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    if (FAILED(static_cast<HRESULT>(invoke(device, 133, arg(words), arg(&buffer), D3D12_RESOURCE_STATE_COMMON, 0,
                                           arg(&__uuidof(ID3D12Resource)), arg(&s.arguments)))))
        return 28;
    // Written after creation, so replay must obtain the arguments from PAGE/MAP, not the creation call's ARGM.
    *words = {3, 1, 0, 0};
    return 0;
}

int create_title_range(void* device, Scene& s)
{
    if (s.title_range == TitleRange::none) return 0;
    s.constants = static_cast<unsigned char*>(VirtualAlloc(nullptr, cbv_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!s.constants) return 40;
    if (s.title_range == TitleRange::root) {
        std::printf("title ranges: %llx 0\n", arg(s.constants));
        return 0;
    }
    s.view_stride = static_cast<UINT>(invoke(device, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
    if (s.view_stride < 32) return 41;
    struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word cpu, bytes;
    } info{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    if (s.title_range == TitleRange::supplied) {
        info.bytes = 2 * s.view_stride;
        info.cpu = arg(VirtualAlloc(nullptr, info.bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!info.cpu) return 42;
    }
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, arg(&info), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&s.view_heap)))))
        return 43;
    s.view_cpu = invoke(s.view_heap, 9);
    s.view_gpu = invoke(s.view_heap, 10);
    if (!s.view_cpu || !s.view_gpu || (info.cpu && s.view_cpu != info.cpu)) return 44;
    const D3D12_CONSTANT_BUFFER_VIEW_DESC view{arg(s.constants), cbv_bytes};
    invoke(device, 17, arg(&view), s.view_cpu);
    // Copy the descriptor through CPU memory. No recorded call populates the slot used by the draw.
    std::memcpy(reinterpret_cast<void*>(s.view_cpu + s.view_stride),
                reinterpret_cast<const void*>(s.view_cpu), s.view_stride);
    std::printf("title ranges: %llx %llx\n", arg(s.constants), s.view_cpu);
    return 0;
}

int draw_frame(Scene& s, void* device, void* pso, UINT frame, UINT frames, bool slot_layouts)
{
    const float t = s.constants ? 0.0f : static_cast<float>(frame) / static_cast<float>(frames);
    if (s.constants) {
        const float colour[] = {float(frame % 2 == 0), float(frame % 2 != 0), 0.0f, 1.0f};
        std::memcpy(s.constants + colour_offset, colour, sizeof(colour));
    }
    const float colour[] = {t, 0.25f, 1.0f - t, 1.0f};
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(s.constants ? width : 4 + (frame * 4) % (width - 4)), static_cast<LONG>(height)};
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
    if (!s.signature || frame != 1) invoke(s.list, 52, s.rtv, arg(colour));
    if (!s.signature || frame == 1) {
        invoke(s.list, 50, 1, arg(&s.rtv));
        invoke(s.list, 21, 1, arg(&viewport));
        invoke(s.list, 22, 1, arg(&scissor));
        invoke(s.list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        invoke(s.list, 25, arg(pso));
        if (s.signature) {
            invoke(s.list, 34, arg(stream_root_signature()));
            invoke(s.list, 61, arg(s.signature), 1, arg(s.arguments), 0, 0, 0);
        } else {
            if (s.constants) {
                invoke(s.list, 34, arg(stream_root_signature()));
                if (s.title_range == TitleRange::root) {
                    invoke(s.list, 42, 0, arg(s.constants));
                } else {
                    invoke(s.list, 32, 1, arg(&s.view_heap));
                    invoke(s.list, 36, 0, s.view_gpu + s.view_stride);
                }
            }
            invoke(s.list, 12, 3, 1);
        }
    }
    if (FAILED(static_cast<HRESULT>(invoke(s.list, 9)))) return 31;
    invoke(s.queue, 10, 1, arg(&s.list));
    if (!s.signature) {
        // Plane record as queue_present reads it: +0x08 resource count, +0x10 pointer to the resource array.
        alignas(8) unsigned char plane[0x48]{};
        const Word resources[1]{arg(s.target)};
        const Word token = 7, one = 1, resources_at = arg(resources);
        std::memcpy(plane + 0x00, &token, 8);
        std::memcpy(plane + 0x08, &one, 8);
        std::memcpy(plane + 0x10, &resources_at, 8);
        invoke(s.queue, 32, 1, arg(plane), 0);
    }
    if (s.signature) {
        std::fprintf(stderr, "replay quad: about to signal submit %u\n", frame + 1);
        std::fflush(stderr);
    }
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
    Scene scene;
    char mode[16]{};
    if (GetEnvironmentVariableA("X2W_TEST_REPLAY_TITLE_RANGE", mode, sizeof(mode))) {
        if (!std::strcmp(mode, "root")) scene.title_range = TitleRange::root;
        else if (!std::strcmp(mode, "generated")) scene.title_range = TitleRange::generated;
        else if (!std::strcmp(mode, "supplied")) scene.title_range = TitleRange::supplied;
        else return 2;
    }
    const bool table = scene.title_range == TitleRange::generated || scene.title_range == TitleRange::supplied;
    if (const int rc = prepare_stream_pipeline(argv[1], argv[2], argv[3], std::strcmp(argv[4], "hardware") == 0, table, adapter,
                                               device, pso))
        return rc;
    if (const int rc = create_scene(device, scene)) return rc;
    if (const int rc = create_title_range(device, scene)) return rc;
    if (GetEnvironmentVariableA("X2W_TEST_REPLAY_INDIRECT", nullptr, 0)) {
        if (frames != 3) return 2;
        if (const int rc = create_indirect_arguments(device, scene)) return rc;
    }
    char recorder_case[16]{};
    GetEnvironmentVariableA("X2W_TEST_REPLAY_RECORDER", recorder_case, sizeof(recorder_case));
    if (std::strcmp(recorder_case, "memory") == 0 && !record_memory_cases(device)) return 37;
    if (std::strcmp(recorder_case, "cap") == 0) {
        // CALL chunks alone drive the file over 1 MiB while the timed window is still far away.
        for (unsigned i = 0; i < 8000; ++i) {
            invoke(device, 1);
            invoke(device, 2);
        }
    }
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
    if (std::strcmp(recorder_case, "cap") == 0)
        std::printf("replay quad: recorder calls after scene %llu\n",
                    static_cast<unsigned long long>(x2w_d3d12_replay_record_calls()));
    std::printf("replay quad: %u presents drawn\n", scene.signature ? 0 : frames);
    return 0;
}
