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
#include <memory>
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

// Private data forwarding exposes the desktop resource for fixture initialization and readback.
ComPtr<ID3D12Resource> desktop_resource(ID3D12Device* desktop, void* guest, IUnknown* resource)
{
    constexpr GUID key{0x3c732b59, 0xf92a, 0x4e82, {0xbf, 0xab, 0x66, 0x62, 0xab, 0x84, 0x3e, 0xd9}};
    auto* device = static_cast<ID3D12Device*>(guest);
    if (FAILED(device->SetPrivateDataInterface(key, resource))) return {};
    ComPtr<IUnknown> object;
    UINT size = sizeof(IUnknown*);
    const auto hr = desktop->GetPrivateData(key, &size, object.GetAddressOf());
    device->SetPrivateDataInterface(key, nullptr);
    ComPtr<ID3D12Resource> result;
    if (SUCCEEDED(hr) && object) object.As(&result);
    return result;
}

int readonly_scene(ID3D12Device* desktop, void* guest, StreamTarget& state, void* root, void* pipeline,
                   D3D12_CPU_DESCRIPTOR_HANDLE rtv, const std::string& prefix, bool indirect, bool off)
{
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC depth_desc{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, DXGI_FORMAT_R32_TYPELESS, {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
    std::array<ComPtr<ID3D12Resource>, 2> depths, native_depths;
    for (unsigned i = 0; i < depths.size(); ++i) {
        if (FAILED(static_cast<HRESULT>(invoke(guest, 76, ptr_arg(&default_heap), 0, ptr_arg(&depth_desc), 0, 0,
                                              ptr_arg(&__uuidof(ID3D12Resource)), ptr_arg(depths[i].GetAddressOf())))))
            return 60;
        native_depths[i] = desktop_resource(desktop, guest, depths[i].Get());
        if (!native_depths[i]) return 61;
    }

    struct HeapDesc {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    };
    const HeapDesc dsv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1}, 0, 0};
    std::array<ComPtr<ID3D12DescriptorHeap>, 3> dsv_heaps;
    std::array<Word, 3> dsv{};
    for (unsigned i = 0; i < dsv.size(); ++i) {
        if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr_arg(&dsv_heap_desc), ptr_arg(&__uuidof(ID3D12DescriptorHeap)),
                                              ptr_arg(dsv_heaps[i].GetAddressOf()))))) return 62;
        dsv[i] = invoke(dsv_heaps[i].Get(), 9);
        const D3D12_DEPTH_STENCIL_VIEW_DESC view{
            DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D,
            i == 1 ? D3D12_DSV_FLAG_READ_ONLY_DEPTH : D3D12_DSV_FLAG_NONE, {}};
        invoke(guest, 21, ptr_arg(depths[i == 2 ? 1 : 0].Get()), ptr_arg(&view), dsv[i]);
    }
    const HeapDesc srv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    ComPtr<ID3D12DescriptorHeap> srv_heap;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr_arg(&srv_heap_desc), ptr_arg(&__uuidof(ID3D12DescriptorHeap)),
                                          ptr_arg(srv_heap.GetAddressOf()))))) return 63;
    struct {
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        Word tail[3]{};
    } srv;
    srv.view.Format = DXGI_FORMAT_R32_FLOAT;
    srv.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.view.Texture2D.MipLevels = 1;
    invoke(guest, 18, ptr_arg(depths[0].Get()), ptr_arg(&srv.view), invoke(srv_heap.Get(), 9));
    const Word table = invoke(srv_heap.Get(), 10);

    const auto free_backing = [](void* ptr) { VirtualFree(ptr, 0, MEM_RELEASE); };
    std::unique_ptr<void, decltype(free_backing)> backing(nullptr, free_backing);
    ComPtr<ID3D12CommandSignature> signature;
    ComPtr<ID3D12Resource> arguments;
    // The guest argument buffer is registered in COMMON. An upload resource seeds the GPU record.
    if (indirect) {
        D3D12_INDIRECT_ARGUMENT_DESC argument{};
        argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
        const D3D12_COMMAND_SIGNATURE_DESC signature_desc{sizeof(D3D12_DRAW_ARGUMENTS), 1, &argument, 0};
        if (FAILED(static_cast<HRESULT>(invoke(guest, 41, ptr_arg(&signature_desc), 0,
                                              ptr_arg(&__uuidof(ID3D12CommandSignature)), ptr_arg(signature.GetAddressOf())))))
            return 64;
        const D3D12_RESOURCE_DESC buffer{
            D3D12_RESOURCE_DIMENSION_BUFFER, 0, sizeof(D3D12_DRAW_ARGUMENTS), 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        backing.reset(VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!backing) return 65;
        auto* address = static_cast<char*>(backing.get()) + 4;
        if (FAILED(static_cast<HRESULT>(invoke(guest, 133, ptr_arg(address), ptr_arg(&buffer), D3D12_RESOURCE_STATE_COMMON, 0,
                                              ptr_arg(&__uuidof(ID3D12Resource)), ptr_arg(arguments.GetAddressOf())))))
            return 65;
        const auto native_args = desktop_resource(desktop, guest, arguments.Get());
        ComPtr<ID3D12Resource> upload;
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        if (!native_args || FAILED(desktop->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                           D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload))))
            return 66;
        void* data = nullptr;
        if (FAILED(upload->Map(0, nullptr, &data))) return 67;
        const D3D12_DRAW_ARGUMENTS draw{6, 1, 0, 0};
        std::memcpy(data, &draw, sizeof(draw));
        upload->Unmap(0, nullptr);
        if (const int rc = submit_desktop(desktop, [&](ID3D12GraphicsCommandList* list) {
                transition(list, native_args.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                list->CopyBufferRegion(native_args.Get(), 0, upload.Get(), 0, sizeof(draw));
                transition(list, native_args.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            })) return rc;
    }

    const auto reset = [&] {
        return SUCCEEDED(static_cast<HRESULT>(invoke(state.allocator, 8))) &&
               SUCCEEDED(static_cast<HRESULT>(invoke(state.list, 139, ptr_arg(state.allocator))));
    };
    if (!reset()) return 68;
    using ClearDepth = void (WINAPI*)(void*, Word, D3D12_CLEAR_FLAGS, float, UINT8, UINT, const D3D12_RECT*);
    const auto clear_depth_rect = reinterpret_cast<ClearDepth>((*static_cast<void***>(state.list))[51]);
    for (LONG x = 0; x < static_cast<LONG>(width); ++x) {
        const D3D12_RECT column{x, 0, x + 1, static_cast<LONG>(height)};
        const float z = static_cast<float>(x + 1) / 16.0f;
        clear_depth_rect(state.list, dsv[0], D3D12_CLEAR_FLAG_DEPTH, z, 0, 1, &column);
        clear_depth_rect(state.list, dsv[2], D3D12_CLEAR_FLAG_DEPTH, z, 0, 1, &column);
    }
    if (const int rc = execute_guest_list(state.queue, state.list, state.fence, 1)) return rc;
    if (const int rc = read_depth(desktop, native_depths[0].Get(), prefix + ".before.depth")) return rc;

    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT rect{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    const float clear_color[]{0, 0, 1, 1};
    for (UINT i = 0; i < (off ? 1u : 2u); ++i) {
        if (!reset()) return 69;
        const D3D12_CPU_DESCRIPTOR_HANDLE depth_handle{static_cast<SIZE_T>(dsv[i + 1])};
        invoke(state.list, 52, rtv.ptr, ptr_arg(clear_color));
        invoke(state.list, 50, 1, ptr_arg(&rtv), FALSE, ptr_arg(&depth_handle));
        invoke(state.list, 21, 1, ptr_arg(&viewport));
        invoke(state.list, 22, 1, ptr_arg(&rect));
        invoke(state.list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        invoke(state.list, 25, ptr_arg(pipeline));
        invoke(state.list, 34, ptr_arg(root));
        auto* heap = srv_heap.Get();
        invoke(state.list, 32, 1, ptr_arg(&heap));
        invoke(state.list, 36, 0, table);
        if (indirect)
            invoke(state.list, 61, ptr_arg(signature.Get()), 1, ptr_arg(arguments.Get()), 0, 0, 0);
        else
            invoke(state.list, 12, 6, 1);
        if (const int rc = execute_guest_list(state.queue, state.list, state.fence, i + 2)) return rc;
        const std::string output = prefix + (i ? ".writable" : ".readonly");
        if (const int rc = read_stream_target(desktop, state, (output + ".rgba").c_str())) return rc;
        if (const int rc = read_depth(desktop, native_depths[i].Get(), output + ".depth")) return rc;
    }
    if (!reset()) return 70;
    return execute_guest_list(state.queue, state.list, state.fence, off ? 3 : 4);
}

}  // namespace

int run_startup_draw_depth_target(int argc, char** argv, bool indirect)
{
    if (argc != 9) {
        std::puts("usage: startup_draw_depth_target <vs_front> <vs_back> <ps_green> <ps_red> <root out> <warp|hardware> "
                  "<request|bound|null|unknown|drop|readonly-request|readonly|readonly-off> <out prefix>");
        return 2;
    }
    const std::string which = argv[7];
    const bool read_only_case = which == "readonly" || which == "readonly-off" || which == "readonly-request";
    if (!read_only_case && which != "request" && which != "bound" && which != "null" && which != "unknown" && which != "drop") return 2;
    std::vector<char> vs_front, vs_back, ps_green, ps_red;
    if (!read_bytes(argv[1], vs_front) || !read_bytes(argv[2], vs_back) || !read_bytes(argv[3], ps_green) || !read_bytes(argv[4], ps_red))
        return 3;
    const bool hardware = std::strcmp(argv[6], "hardware") == 0;
    const std::string prefix = argv[8];

    ComPtr<IDXGIAdapter> adapter;
    if (const int rc = pick_stream_adapter(hardware, adapter)) return rc;
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

    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable = {1, &range};
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    const D3D12_ROOT_SIGNATURE_DESC root_desc{read_only_case ? 1u : 0u, read_only_case ? &parameter : nullptr};
    ComPtr<ID3DBlob> blob;
    if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr))) return 6;
    std::ofstream(argv[5], std::ios::binary).write(static_cast<const char*>(blob->GetBufferPointer()), blob->GetBufferSize());
    ID3D12RootSignature* root = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 16, 0, ptr_arg(blob->GetBufferPointer()), blob->GetBufferSize(),
                                           ptr_arg(&__uuidof(ID3D12RootSignature)), ptr_arg(&root))))) return 7;
    void* green = create_stream_pso(guest, root, vs_front, ps_green);
    void* red = read_only_case ? green : create_stream_pso(guest, root, vs_back, ps_red);
    if (!green || !red) return 9;
    if (which == "request" || which == "readonly-request") return 0;

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
    const struct { D3D12_DESCRIPTOR_HEAP_DESC desc; Word backing, bytes; }
        rtv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr_arg(&rtv_heap_desc), ptr_arg(&__uuidof(ID3D12DescriptorHeap)), ptr_arg(&state.rtv_heap)))))
        return 22;
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(state.rtv_heap, 9))};
    const D3D12_RENDER_TARGET_VIEW_DESC rtv_view{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    invoke(guest, 20, ptr_arg(state.attachments[0].texture.Get()), ptr_arg(&rtv_view), rtv.ptr);

    D3D12_COMMAND_QUEUE_DESC qd{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(guest, 8, ptr_arg(&qd), ptr_arg(&__uuidof(ID3D12CommandQueue)), ptr_arg(&state.queue)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, ptr_arg(&__uuidof(ID3D12CommandAllocator)), ptr_arg(&state.allocator)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 73, ptr_arg(list_desc), ptr_arg(state.allocator), 0, ptr_arg(&__uuidof(ID3D12GraphicsCommandList)), ptr_arg(&state.list)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 36, 0, 0, ptr_arg(&__uuidof(ID3D12Fence)), ptr_arg(&state.fence)))))
        return 23;
    if (read_only_case) return readonly_scene(desktop.Get(), guest, state, root, green, rtv, prefix, indirect, which == "readonly-off");

    ComPtr<ID3D12Resource> depth_tex;
    const D3D12_RESOURCE_DESC depth_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, DXGI_FORMAT_D32_FLOAT, {1, 0},
                                         D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
    D3D12_CLEAR_VALUE depth_clear{};
    depth_clear.Format = DXGI_FORMAT_D32_FLOAT;
    depth_clear.DepthStencil = {1.0f, 0};
    if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &depth_desc, D3D12_RESOURCE_STATE_COMMON, &depth_clear,
                                                IID_PPV_ARGS(&depth_tex)))) return 21;

    const struct { D3D12_DESCRIPTOR_HEAP_DESC desc; Word backing, bytes; }
        dsv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1}, 0, 0};
    void* dsv_heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, ptr_arg(&dsv_heap_desc), ptr_arg(&__uuidof(ID3D12DescriptorHeap)), ptr_arg(&dsv_heap)))))
        return 22;
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv{static_cast<SIZE_T>(invoke(dsv_heap, 9))};
    // A zero handle maps to no descriptor; an address in no descriptor heap is not a known view.
    static const unsigned char not_a_heap[64] = {};
    const D3D12_CPU_DESCRIPTOR_HANDLE null_handle{0}, unknown_handle{reinterpret_cast<SIZE_T>(not_a_heap)};
    const D3D12_DEPTH_STENCIL_VIEW_DESC dsv_view{DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_FLAG_NONE, {}};
    invoke(guest, 21, ptr_arg(depth_tex.Get()), ptr_arg(&dsv_view), dsv.ptr);

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

#ifndef X2W_DEPTH_TARGET_SHARED_FIXTURE
int main(int argc, char** argv)
{
    return run_startup_draw_depth_target(argc, argv, false);
}
#endif
