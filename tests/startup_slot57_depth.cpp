// Draw fixture for X2W_DEPTH_STATE (#1999, item A): translated draws of device slot 57 pipelines whose state
// block A carries DB_DEPTH_CONTROL / DB_STENCIL_CONTROL / DB_STENCILREFMASK words. The same renderer runs on
// WARP and on hardware; tests/startup_slot57_depth_check.py runs it with the switch on and off and compares.
//
// usage: startup_slot57_depth <vs_far> <vs_near> <ps_green> <ps_red> <root_out> <warp|hardware> [<out_dir>]
//
// vs_far / vs_near draw the same triangle at z = 0.25 / 0.75 (the check names them by their depth). Without
// out_dir the run only records the translation requests (the draws are skipped until the shaders are
// translated). With it, every case writes <out_dir>\<case>.bin, the 8x4 RGBA8 target after its draws.
//
// Register layouts: Mesa src/amd/registers/gfx10.json (MIT), as pipeline_state_decode.h documents them.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "startup_stream_setup.h"

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Word = std::uint64_t;
template<class T> Word ptr_arg(T* p) { return reinterpret_cast<Word>(p); }

// ---- register words ----
constexpr std::uint32_t depth_control(bool stencil, bool z, bool z_write, unsigned zfunc, unsigned sfunc)
{
    return std::uint32_t{stencil} | std::uint32_t{z} << 1 | std::uint32_t{z_write} << 2 | (zfunc & 7) << 4 | (sfunc & 7) << 8 |
           (sfunc & 7) << 20;
}
constexpr std::uint32_t stencil_ops(unsigned fail, unsigned zpass, unsigned zfail)
{
    return fail | zpass << 4 | zfail << 8 | fail << 12 | zpass << 16 | zfail << 20;
}
constexpr std::uint32_t ref_mask(unsigned test, unsigned mask, unsigned write) { return test | mask << 8 | write << 16 | 1u << 24; }
// CompareFrag
constexpr unsigned f_less = 1, f_equal = 2, f_greater = 4, f_always = 7;
// StencilOp
constexpr unsigned op_keep = 0, op_replace = 3, op_ones = 2;

struct Words { std::uint32_t depth = 0, stencil = 0, refmask = 0, refmask_bf = 0; };

bool read_bytes(const char* path, std::vector<char>& out)
{
    std::ifstream file(path, std::ios::binary);
    out.assign(std::istreambuf_iterator<char>(file), {});
    return !out.empty();
}

int pick_adapter(bool hardware, ComPtr<IDXGIAdapter>& adapter)
{
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 4;
    if (!hardware) return SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ? 0 : 4;
    ComPtr<IDXGIFactory6> factory6;
    if (FAILED(factory.As(&factory6))) return 77;
    for (UINT n = 0;; ++n) {
        ComPtr<IDXGIAdapter1> candidate;
        const HRESULT hr = factory6->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
        if (hr == DXGI_ERROR_NOT_FOUND) return 77;
        if (FAILED(hr)) return 4;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(candidate->GetDesc1(&desc))) return 4;
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            adapter = candidate;
            return 0;
        }
    }
}

// A slot 57 description as the stand-in reads it (see tests/startup_stream_consumer.cpp): 64-bit words 0 (root
// signature), 1-2 (state block A and its size), 5-6 and 9-10 (the two shaders).
void* make_slot57(void* device, void* root, const std::vector<char>& vs, const std::vector<char>& ps, const Words& w)
{
    struct Block { alignas(8) unsigned char a[0x200]; alignas(8) unsigned char desc[0x60]; };
    static std::vector<Block*> keep;  // the stand-in copies both, but keep them for the process lifetime anyway
    auto* b = new Block{};
    keep.push_back(b);
    auto put = [&](std::size_t off, std::uint32_t v) { std::memcpy(b->a + off, &v, 4); };
    put(0x0b8, w.depth);
    put(0x0bc, w.stencil);
    put(0x0c0, w.refmask);
    put(0x0c4, w.refmask_bf);
    const Word words[12] = {ptr_arg(root), ptr_arg(b->a), sizeof(b->a), 0, 0, ptr_arg(vs.data()), vs.size(), 0, 0,
                            ptr_arg(ps.data()), ps.size(), 0};
    std::memcpy(b->desc, words, sizeof(words));
    using Create = HRESULT (WINAPI*)(void*, const void*, REFIID, void**);
    void* pso = nullptr;
    if (FAILED(reinterpret_cast<Create>((*static_cast<void***>(device))[57])(device, b->desc, __uuidof(ID3D12PipelineState), &pso))) return nullptr;
    return pso;
}

struct alignas(8) StageSubobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag;
    D3D12_SHADER_BYTECODE shader;
};
struct alignas(8) RootSubobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag;
    ID3D12RootSignature* signature;
};
struct alignas(8) DepthStencilSubobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag;
    D3D12_DEPTH_STENCIL_DESC desc;
};
struct alignas(8) DsvFormatSubobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag;
    DXGI_FORMAT format;
};
struct DepthStream {
    StageSubobject ps;
    RootSubobject root;
    StageSubobject vs;
    DepthStencilSubobject ds;
    DsvFormatSubobject dsv_format;
};

void* create_stream_pso(void* device, ID3D12RootSignature* root, const std::vector<char>& vs, const std::vector<char>& ps)
{
    D3D12_DEPTH_STENCIL_DESC stencil{};
    stencil.DepthEnable = FALSE;
    stencil.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    stencil.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    stencil.StencilEnable = TRUE;
    stencil.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
    stencil.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
    stencil.FrontFace = stencil.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_EQUAL};
    DepthStream stream{
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, {ps.data(), ps.size()}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, {vs.data(), vs.size()}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, stencil},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT_D24_UNORM_S8_UINT},
    };
    using CreateStream = HRESULT (WINAPI*)(void*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    auto create = reinterpret_cast<CreateStream>((*static_cast<void***>(device))[86]);
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    void* pso = nullptr;
    return SUCCEEDED(create(device, &desc, __uuidof(ID3D12PipelineState), &pso)) ? pso : nullptr;
}

int clear_depth_stencil(ID3D12Device* desktop, ID3D12Resource* tex, float depth, UINT8 stencil)
{
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
    ComPtr<ID3D12DescriptorHeap> heap;
    if (FAILED(desktop->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)))) return 40;
    const D3D12_DEPTH_STENCIL_VIEW_DESC view{DXGI_FORMAT_D24_UNORM_S8_UINT, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_FLAG_NONE, {}};
    const D3D12_CPU_DESCRIPTOR_HANDLE handle = heap->GetCPUDescriptorHandleForHeapStart();
    desktop->CreateDepthStencilView(tex, &view, handle);

    D3D12_COMMAND_QUEUE_DESC qd{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(desktop->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(desktop->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(desktop->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return 41;

    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {tex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE};
    list->ResourceBarrier(1, &b);
    list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, depth, stencil, 0, nullptr);
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    list->ResourceBarrier(1, &b);
    if (FAILED(list->Close())) return 42;
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.Get(), 1))) return 43;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return 44;
    const bool ok = SUCCEEDED(fence->SetEventOnCompletion(1, ev)) && WaitForSingleObject(ev, 10000) == WAIT_OBJECT_0;
    CloseHandle(ev);
    return ok ? 0 : 45;
}

int execute_guest_list(void* queue, void* list, void* fence, UINT64 fence_value)
{
    if (FAILED(static_cast<HRESULT>(invoke(list, 9)))) return 50;
    void* lists[] = {list};
    invoke(queue, 10, 1, ptr_arg(lists));
    if (FAILED(static_cast<HRESULT>(invoke(queue, 14, ptr_arg(fence), fence_value)))) return 51;
    const auto deadline = GetTickCount64() + 10000;
    while (invoke(fence, 8) < fence_value && GetTickCount64() < deadline) Sleep(1);
    return invoke(fence, 8) >= fence_value ? 0 : 53;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 7 && argc != 8) return 2;
    std::vector<char> vs_far, vs_near, ps_green, ps_red;
    if (!read_bytes(argv[1], vs_far) || !read_bytes(argv[2], vs_near) || !read_bytes(argv[3], ps_green) || !read_bytes(argv[4], ps_red)) return 3;
    const bool hardware = std::strcmp(argv[6], "hardware") == 0;
    const std::string out_dir = argc == 8 ? argv[7] : "";

    ComPtr<IDXGIAdapter> adapter;
    if (const int rc = pick_adapter(hardware, adapter)) return rc;
    {
        DXGI_ADAPTER_DESC desc{};
        LARGE_INTEGER driver{};
        if (FAILED(adapter->GetDesc(&desc))) return 4;
        adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver);
        std::printf("adapter=%ls vendor=%04x device=%04x driver=%u.%u.%u.%u\n", desc.Description, desc.VendorId, desc.DeviceId,
                    HIWORD(driver.HighPart), LOWORD(driver.HighPart), HIWORD(driver.LowPart), LOWORD(driver.LowPart));
        std::fflush(stdout);
    }
    std::array<unsigned char, 64> parameters{};
    void* guest = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &guest))) return 5;

    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    ComPtr<ID3DBlob> blob;
    if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr))) return 6;
    std::ofstream(argv[5], std::ios::binary).write(static_cast<const char*>(blob->GetBufferPointer()), blob->GetBufferSize());
    ID3D12RootSignature* root = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 16, 0, ptr_arg(blob->GetBufferPointer()), blob->GetBufferSize(),
                                           ptr_arg(&__uuidof(ID3D12RootSignature)), ptr_arg(&root))))) return 7;

    // ---- pipelines: (shader pair, register words) ----
    const Words none{};                                                                       // every word zero: depth and stencil off
    const Words greater{depth_control(false, true, true, f_greater, f_always), 0, 0, 0};      // reversed-Z, written
    const Words less{depth_control(false, true, true, f_less, f_always), 0, 0, 0};            // the other direction
    Words tied = greater;
    tied.depth |= 1u << 30;                                                                   // ENABLE_COLOR_WRITES_ON_DEPTH_FAIL: refused
    const Words st_equal_5a{depth_control(true, false, false, f_always, f_equal), stencil_ops(op_keep, op_keep, op_keep), ref_mask(0x5a, 0xff, 0), ref_mask(0x5a, 0xff, 0)};
    const Words st_equal_5b{depth_control(true, false, false, f_always, f_equal), stencil_ops(op_keep, op_keep, op_keep), ref_mask(0x5b, 0xff, 0), ref_mask(0x5b, 0xff, 0)};
    const Words st_replace_7{depth_control(true, false, false, f_always, f_always), stencil_ops(op_keep, op_replace, op_keep), ref_mask(0x07, 0xff, 0xff), ref_mask(0x07, 0xff, 0xff)};
    const Words st_equal_7{depth_control(true, false, false, f_always, f_equal), stencil_ops(op_keep, op_keep, op_keep), ref_mask(0x07, 0xff, 0), ref_mask(0x07, 0xff, 0)};
    const Words st_equal_8{depth_control(true, false, false, f_always, f_equal), stencil_ops(op_keep, op_keep, op_keep), ref_mask(0x08, 0xff, 0), ref_mask(0x08, 0xff, 0)};
    const Words st_ones{depth_control(true, false, false, f_always, f_equal), stencil_ops(op_ones, op_keep, op_keep), ref_mask(0x5b, 0xff, 0), ref_mask(0x5b, 0xff, 0)};

    auto far_green = [&](const Words& w) { return make_slot57(guest, root, vs_far, ps_green, w); };
    auto near_red = [&](const Words& w) { return make_slot57(guest, root, vs_near, ps_red, w); };

    struct Case {
        const char* name;
        float depth_clear;
        UINT8 stencil_clear;
        bool bind_depth;
        std::vector<void*> draws;   // in draw order
        UINT guest_stencil_ref = 0;
    };
    const std::vector<Case> cases = {
        // Depth. The near triangle (z 0.75, red) is drawn first except in fz_less.
        {"rz_greater", 0.0f, 0, true, {near_red(greater), far_green(greater)}},    // reversed-Z, cleared to 0: red wins
        {"rz_less_wrong", 0.0f, 0, true, {near_red(less), far_green(less)}},       // wrong direction: nothing passes
        {"fz_less", 1.0f, 0, true, {far_green(less), near_red(less)}},             // forward-Z, cleared to 1: green wins
        {"rz_greater_no_dsv", 0.0f, 0, false, {near_red(greater), far_green(greater)}},  // no depth view: desktop default
        {"rz_refused", 0.0f, 0, true, {near_red(tied), far_green(tied)}},          // refused words: desktop default
        {"rz_none", 0.0f, 0, true, {near_red(none), far_green(none)}},             // zero words (depth off): same as the default
        // Stencil. Depth off in all of them; the target is drawn in green or red over blue.
        {"st_equal_hit", 1.0f, 0x5a, true, {far_green(st_equal_5a)}},              // stored 0x5A, reference 0x5A: drawn
        {"st_equal_miss", 1.0f, 0x5a, true, {far_green(st_equal_5b)}},             // stored 0x5A, reference 0x5B: not drawn
        {"st_replace_hit", 1.0f, 0, true, {near_red(st_replace_7), far_green(st_equal_7)}},   // REPLACE 7, then EQUAL 7: green
        {"st_replace_miss", 1.0f, 0, true, {near_red(st_replace_7), far_green(st_equal_8)}},  // REPLACE 7, then EQUAL 8: red stays
        {"st_ref_restore", 1.0f, 0x5a, true,
            {near_red(st_equal_5b), create_stream_pso(guest, root, vs_far, ps_green)}, 0x5a},
        {"st_ref_reset", 1.0f, 0, true,
            {near_red(st_equal_5b), create_stream_pso(guest, root, vs_far, ps_green)}},
        {"st_refused", 1.0f, 0x5a, true, {far_green(st_ones)}},                    // ONES has no D3D12 form: desktop default, drawn
    };
    for (const auto& c : cases)
        for (void* pso : c.draws) if (!pso) return 9;

    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 20;
    constexpr UINT width = 8, height = 4;
    StreamTarget state{};
    state.count = 1;
    state.attachments[0].row_bytes = width * 4;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const float clear_color[] = {0.0f, 0.0f, 1.0f, 1.0f};
    const D3D12_RESOURCE_DESC rtv_tex_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE rtv_clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0.0f, 0.0f, 1.0f, 1.0f}};
    if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rtv_tex_desc,
            D3D12_RESOURCE_STATE_COMMON, &rtv_clear, IID_PPV_ARGS(&state.attachments[0].texture)))) return 21;
    const D3D12_RESOURCE_DESC dsv_tex_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        DXGI_FORMAT_D24_UNORM_S8_UINT, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};

    const struct { D3D12_DESCRIPTOR_HEAP_DESC desc; Word backing, bytes; }
        rtv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0},
        dsv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1}, 0, 0};
    void* dsv_heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr_arg(&rtv_heap_desc), ptr_arg(&__uuidof(ID3D12DescriptorHeap)), ptr_arg(&state.rtv_heap)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr_arg(&dsv_heap_desc), ptr_arg(&__uuidof(ID3D12DescriptorHeap)), ptr_arg(&dsv_heap))))) return 22;
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(state.rtv_heap, 9))};
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv{static_cast<SIZE_T>(invoke(dsv_heap, 9))};
    const D3D12_RENDER_TARGET_VIEW_DESC rtv_view{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    const D3D12_DEPTH_STENCIL_VIEW_DESC dsv_view{DXGI_FORMAT_D24_UNORM_S8_UINT, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_FLAG_NONE, {}};
    invoke(guest, 20, ptr_arg(state.attachments[0].texture.Get()), ptr_arg(&rtv_view), rtv.ptr);

    D3D12_COMMAND_QUEUE_DESC qd{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(guest, 8, ptr_arg(&qd), ptr_arg(&__uuidof(ID3D12CommandQueue)), ptr_arg(&state.queue)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, ptr_arg(&__uuidof(ID3D12CommandAllocator)), ptr_arg(&state.allocator)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 73, ptr_arg(list_desc), ptr_arg(state.allocator), 0, ptr_arg(&__uuidof(ID3D12GraphicsCommandList)), ptr_arg(&state.list)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 36, 0, 0, ptr_arg(&__uuidof(ID3D12Fence)), ptr_arg(&state.fence))))) return 23;
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT rect{0, 0, 4, static_cast<LONG>(height)};

    UINT64 fence_value = 0;
    for (const auto& c : cases) {
        // A fresh depth-stencil texture per case, cleared on the desktop device with the case's values.
        ComPtr<ID3D12Resource> depth_tex;
        D3D12_CLEAR_VALUE clear{};
        clear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        clear.DepthStencil = {c.depth_clear, c.stencil_clear};
        if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &dsv_tex_desc,
                D3D12_RESOURCE_STATE_COMMON, &clear, IID_PPV_ARGS(&depth_tex)))) return 21;
        if (const int rc = clear_depth_stencil(desktop.Get(), depth_tex.Get(), c.depth_clear, c.stencil_clear)) return rc;
        invoke(guest, 21, ptr_arg(depth_tex.Get()), ptr_arg(&dsv_view), dsv.ptr);
        if (FAILED(static_cast<HRESULT>(invoke(state.allocator, 8)))) return 24;
        if (FAILED(static_cast<HRESULT>(invoke(state.list, 139, ptr_arg(state.allocator))))) return 25;
        invoke(state.list, 52, rtv.ptr, ptr_arg(clear_color));
        invoke(state.list, 21, 1, ptr_arg(&viewport));
        invoke(state.list, 22, 1, ptr_arg(&rect));
        invoke(state.list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        invoke(state.list, 50, 1, ptr_arg(&rtv), FALSE, c.bind_depth ? ptr_arg(&dsv) : 0);
        if (c.guest_stencil_ref) invoke(state.list, 24, c.guest_stencil_ref);
        for (void* pso : c.draws) {
            invoke(state.list, 25, ptr_arg(pso));
            invoke(state.list, 12, 3, 1);
        }
        if (const int rc = execute_guest_list(state.queue, state.list, state.fence, ++fence_value)) return rc;
        const std::string path = out_dir.empty() ? std::string(argv[5]) + ".request" : out_dir + "\\" + c.name + ".bin";
        if (const int rc = read_stream_target(desktop.Get(), state, path.c_str())) return rc;
        std::printf("case %s done\n", c.name);
    }
    return 0;
}
