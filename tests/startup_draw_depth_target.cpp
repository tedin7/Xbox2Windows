// Depth target binding for translated draws (#1999 item D). A depth-tested stream pipeline (LESS, writes on)
// draws a near green triangle (z 0.25) and then a far red one (z 0.75) through the stand-in, with the colour
// target and a D32 depth target bound by list slot 50 (OMSetRenderTargets). The colour target and the depth
// values are read back after the fence. Source-built: the shaders come from tests/startup_draw_depth_target_check.py.
//
// Cases, one per process so each run's debug-layer messages belong to it:
//   request  create the pipelines only (publishes the translation requests)
//   bound    the depth handle resolves: green in front, depth 0.25 where drawn, 1.0 elsewhere
//   null     the depth handle holds 0, which maps to no descriptor: the stand-in binds the colour target
//            without depth (logged, counted once), so red overwrites green and the depth target keeps 1.0
//   unknown  the depth handle is in no descriptor heap: the stand-in refuses the whole bind (logged,
//            counted once), so nothing is drawn
//   drop     the bound case's draw with the null handle, for the negative control
// After the read-back an empty list is executed, so the stand-in drains the debug-layer messages of the case.
// Prints "binds +<n>, unresolved +<n>" (x2w_d3d12_render_target_stats before and after the draw).
//
// usage: startup_draw_depth_target <vs_front> <vs_back> <ps_green> <ps_red> <root out> <warp|hardware> <case> <out prefix>
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
extern "C" __declspec(dllimport) void x2w_d3d12_render_target_stats(unsigned* binds, unsigned* unresolved);

namespace {

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

using Word = std::uint64_t;
template<class T> Word ptr_arg(T* p) { return reinterpret_cast<Word>(p); }
constexpr UINT width = 8, height = 4;

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

void* create_stream_pso(void* device, ID3D12RootSignature* root, const std::vector<char>& vs, const std::vector<char>& ps)
{
    D3D12_DEPTH_STENCIL_DESC less{};
    less.DepthEnable = TRUE;
    less.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    less.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    less.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
    less.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
    less.FrontFace = less.BackFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS};
    DepthStream stream{
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, {ps.data(), ps.size()}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, {vs.data(), vs.size()}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, less},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT_D32_FLOAT},
    };
    using CreateStream = HRESULT (WINAPI*)(void*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    auto create = reinterpret_cast<CreateStream>((*static_cast<void***>(device))[86]);
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    void* pso = nullptr;
    return SUCCEEDED(create(device, &desc, __uuidof(ID3D12PipelineState), &pso)) ? pso : nullptr;
}

// Runs `record` on a desktop direct list and waits for it.
template<class Record> int submit_desktop(ID3D12Device* desktop, Record record)
{
    D3D12_COMMAND_QUEUE_DESC qd{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(desktop->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(desktop->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(desktop->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return 41;
    record(list.Get());
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

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* tex, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {tex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &b);
}

// Clears the depth target to 1 on the desktop device, leaving it in COMMON.
int clear_depth(ID3D12Device* desktop, ID3D12Resource* tex)
{
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
    ComPtr<ID3D12DescriptorHeap> heap;
    if (FAILED(desktop->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)))) return 40;
    const D3D12_DEPTH_STENCIL_VIEW_DESC view{DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_FLAG_NONE, {}};
    const D3D12_CPU_DESCRIPTOR_HANDLE handle = heap->GetCPUDescriptorHandleForHeapStart();
    desktop->CreateDepthStencilView(tex, &view, handle);
    return submit_desktop(desktop, [&](ID3D12GraphicsCommandList* list) {
        transition(list, tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        transition(list, tex, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COMMON);
    });
}

// Writes the D32 depth values, row by row, to `path`.
int read_depth(ID3D12Device* desktop, ID3D12Resource* tex, const std::string& path)
{
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    const auto desc = tex->GetDesc();
    desktop->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, total, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                     D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
    ComPtr<ID3D12Resource> readback;
    if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&readback)))) return 46;
    if (const int rc = submit_desktop(desktop, [&](ID3D12GraphicsCommandList* list) {
            transition(list, tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION destination{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            destination.PlacedFootprint = footprint;
            D3D12_TEXTURE_COPY_LOCATION source{tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
            transition(list, tex, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        })) return rc;
    void* memory = nullptr;
    if (FAILED(readback->Map(0, nullptr, &memory))) return 47;
    std::ofstream out(path, std::ios::binary);
    for (UINT y = 0; y < height; ++y)
        out.write(static_cast<const char*>(memory) + footprint.Footprint.RowPitch * y, width * sizeof(float));
    readback->Unmap(0, nullptr);
    return out ? 0 : 48;
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
    if (argc != 9) {
        std::puts("usage: startup_draw_depth_target <vs_front> <vs_back> <ps_green> <ps_red> <root out> <warp|hardware> "
                  "<request|bound|null|unknown|drop> <out prefix>");
        return 2;
    }
    const std::string which = argv[7];
    if (which != "request" && which != "bound" && which != "null" && which != "unknown" && which != "drop") return 2;
    std::vector<char> vs_front, vs_back, ps_green, ps_red;
    if (!read_bytes(argv[1], vs_front) || !read_bytes(argv[2], vs_back) || !read_bytes(argv[3], ps_green) || !read_bytes(argv[4], ps_red))
        return 3;
    const bool hardware = std::strcmp(argv[6], "hardware") == 0;
    const std::string prefix = argv[8];

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
    void* green = create_stream_pso(guest, root, vs_front, ps_green);
    void* red = create_stream_pso(guest, root, vs_back, ps_red);
    if (!green || !red) return 9;
    if (which == "request") return 0;

    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 20;
    StreamTarget state{};
    state.count = 1;
    state.attachments[0].row_bytes = width * 4;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const float clear_color[] = {0.0f, 0.0f, 1.0f, 1.0f};
    const D3D12_RESOURCE_DESC rtv_tex_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0},
                                           D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE rtv_clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0.0f, 0.0f, 1.0f, 1.0f}};
    if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rtv_tex_desc, D3D12_RESOURCE_STATE_COMMON, &rtv_clear,
                                                IID_PPV_ARGS(&state.attachments[0].texture)))) return 21;
    ComPtr<ID3D12Resource> depth_tex;
    const D3D12_RESOURCE_DESC depth_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, DXGI_FORMAT_D32_FLOAT, {1, 0},
                                         D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
    D3D12_CLEAR_VALUE depth_clear{};
    depth_clear.Format = DXGI_FORMAT_D32_FLOAT;
    depth_clear.DepthStencil = {1.0f, 0};
    if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &depth_desc, D3D12_RESOURCE_STATE_COMMON, &depth_clear,
                                                IID_PPV_ARGS(&depth_tex)))) return 21;

    const struct { D3D12_DESCRIPTOR_HEAP_DESC desc; Word backing, bytes; }
        rtv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0}, dsv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1}, 0, 0};
    void* dsv_heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr_arg(&rtv_heap_desc), ptr_arg(&__uuidof(ID3D12DescriptorHeap)), ptr_arg(&state.rtv_heap)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr_arg(&dsv_heap_desc), ptr_arg(&__uuidof(ID3D12DescriptorHeap)), ptr_arg(&dsv_heap)))))
        return 22;
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(state.rtv_heap, 9))};
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv{static_cast<SIZE_T>(invoke(dsv_heap, 9))};
    // A zero handle maps to no descriptor; an address in no descriptor heap is not a known view.
    static const unsigned char not_a_heap[64] = {};
    const D3D12_CPU_DESCRIPTOR_HANDLE null_handle{0}, unknown_handle{reinterpret_cast<SIZE_T>(not_a_heap)};
    const D3D12_RENDER_TARGET_VIEW_DESC rtv_view{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    const D3D12_DEPTH_STENCIL_VIEW_DESC dsv_view{DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_FLAG_NONE, {}};
    invoke(guest, 20, ptr_arg(state.attachments[0].texture.Get()), ptr_arg(&rtv_view), rtv.ptr);
    invoke(guest, 21, ptr_arg(depth_tex.Get()), ptr_arg(&dsv_view), dsv.ptr);

    D3D12_COMMAND_QUEUE_DESC qd{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(guest, 8, ptr_arg(&qd), ptr_arg(&__uuidof(ID3D12CommandQueue)), ptr_arg(&state.queue)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, ptr_arg(&__uuidof(ID3D12CommandAllocator)), ptr_arg(&state.allocator)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 73, ptr_arg(list_desc), ptr_arg(state.allocator), 0, ptr_arg(&__uuidof(ID3D12GraphicsCommandList)), ptr_arg(&state.list)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 36, 0, 0, ptr_arg(&__uuidof(ID3D12Fence)), ptr_arg(&state.fence)))))
        return 23;
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT rect{0, 0, 4, static_cast<LONG>(height)};

    if (const int rc = clear_depth(desktop.Get(), depth_tex.Get())) return rc;
    unsigned binds_before = 0, unresolved_before = 0, binds_after = 0, unresolved_after = 0;
    x2w_d3d12_render_target_stats(&binds_before, &unresolved_before);
    if (FAILED(static_cast<HRESULT>(invoke(state.allocator, 8)))) return 24;
    if (FAILED(static_cast<HRESULT>(invoke(state.list, 139, ptr_arg(state.allocator))))) return 25;
    invoke(state.list, 52, rtv.ptr, ptr_arg(clear_color));
    invoke(state.list, 21, 1, ptr_arg(&viewport));
    invoke(state.list, 22, 1, ptr_arg(&rect));
    invoke(state.list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_CPU_DESCRIPTOR_HANDLE* depth_handle = which == "bound" ? &dsv : which == "unknown" ? &unknown_handle : &null_handle;
    invoke(state.list, 50, 1, ptr_arg(&rtv), FALSE, ptr_arg(depth_handle));
    invoke(state.list, 25, ptr_arg(green));
    invoke(state.list, 12, 3, 1);
    invoke(state.list, 25, ptr_arg(red));
    invoke(state.list, 12, 3, 1);
    if (const int rc = execute_guest_list(state.queue, state.list, state.fence, 1)) return rc;
    x2w_d3d12_render_target_stats(&binds_after, &unresolved_after);
    if (const int rc = read_stream_target(desktop.Get(), state, (prefix + ".rgba").c_str())) return rc;
    if (const int rc = read_depth(desktop.Get(), depth_tex.Get(), prefix + ".depth")) return rc;
    // An empty list after the fence: executing it drains the debug-layer messages stored so far.
    if (FAILED(static_cast<HRESULT>(invoke(state.allocator, 8))) || FAILED(static_cast<HRESULT>(invoke(state.list, 139, ptr_arg(state.allocator)))))
        return 26;
    if (const int rc = execute_guest_list(state.queue, state.list, state.fence, 2)) return rc;
    std::printf("binds +%u, unresolved +%u\n", binds_after - binds_before, unresolved_after - unresolved_before);
    return 0;
}
