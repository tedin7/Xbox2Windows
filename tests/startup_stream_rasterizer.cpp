#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#include "startup_stream_setup.h"

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {

struct alignas(8) StageSubobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag;
    D3D12_SHADER_BYTECODE shader;
};
struct alignas(8) RootSubobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag;
    ID3D12RootSignature* signature;
};
struct alignas(8) RasterizerSubobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag;
    D3D12_RASTERIZER_DESC desc;
};
static_assert(sizeof(StageSubobject) == 24);
static_assert(sizeof(RootSubobject) == 16);
static_assert(sizeof(RasterizerSubobject) == 48);
static_assert(offsetof(RasterizerSubobject, desc) == 4);

struct BaseStream {
    StageSubobject ps;
    RootSubobject root;
    StageSubobject vs;
};
struct RasterStream {
    BaseStream base;
    RasterizerSubobject raster;
};
struct MixedRasterStream {
    BaseStream base;
    RasterizerSubobject first;
    RasterizerSubobject second;
};
static_assert(sizeof(BaseStream) % 8 == 0);
static_assert(sizeof(RasterStream) % 8 == 0);

using Word = std::uint64_t;
template<class T> Word ptr_arg(T* p) { return reinterpret_cast<Word>(p); }

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

D3D12_RASTERIZER_DESC raster_desc(D3D12_CULL_MODE cull, BOOL depth_clip)
{
    D3D12_RASTERIZER_DESC desc{};
    desc.FillMode = D3D12_FILL_MODE_SOLID;
    desc.CullMode = cull;
    desc.DepthClipEnable = depth_clip;
    return desc;
}

// kind 0 drops the rasterizer subobject. kind 1 is RASTERIZER. kind 2 is RASTERIZER1 (type 27),
// and bias replaces the int depth bias with the float bits.
void* create_stream_pso(void* device, ID3D12RootSignature* root, const std::vector<char>& vs,
                        const std::vector<char>& ps, int kind, const D3D12_RASTERIZER_DESC& raster, float bias,
                        bool mixed = false)
{
    using CreateStream = HRESULT (WINAPI*)(void*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    auto create = reinterpret_cast<CreateStream>((*static_cast<void***>(device))[86]);
    BaseStream base{
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, {ps.data(), ps.size()}},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, {vs.data(), vs.size()}},
    };
    void* pso = nullptr;
    if (kind == 0) {
        const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(base), &base};
        if (FAILED(create(device, &desc, __uuidof(ID3D12PipelineState), &pso))) return nullptr;
        return pso;
    }
    RasterizerSubobject sub{kind == 1 ? D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER
                                       : static_cast<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE>(27),
                            raster};
    if (kind == 2) std::memcpy(&sub.desc.DepthBias, &bias, sizeof(bias));
    if (mixed) {
        const auto other_type = kind == 2 ? D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER
                                          : static_cast<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE>(27);
        MixedRasterStream stream{base, sub, {other_type, raster_desc(D3D12_CULL_MODE_NONE, TRUE)}};
        const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
        if (FAILED(create(device, &desc, __uuidof(ID3D12PipelineState), &pso))) return nullptr;
        return pso;
    }
    RasterStream stream{base, sub};
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    if (FAILED(create(device, &desc, __uuidof(ID3D12PipelineState), &pso))) return nullptr;
    return pso;
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
    if (argc != 5 && argc != 13 && argc != 17) return 2;
    std::vector<char> vs, ps;
    if (!read_bytes(argv[1], vs) || !read_bytes(argv[2], ps)) return 3;
    const bool hardware = std::strcmp(argv[4], "hardware") == 0;

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
    std::ofstream(argv[3], std::ios::binary).write(static_cast<const char*>(blob->GetBufferPointer()), blob->GetBufferSize());
    ID3D12RootSignature* root = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 16, 0, ptr_arg(blob->GetBufferPointer()), blob->GetBufferSize(),
                                           ptr_arg(&__uuidof(ID3D12RootSignature)), ptr_arg(&root))))) return 7;

    const D3D12_RASTERIZER_DESC back = raster_desc(D3D12_CULL_MODE_BACK, TRUE);
    const D3D12_RASTERIZER_DESC front = raster_desc(D3D12_CULL_MODE_FRONT, TRUE);
    D3D12_RASTERIZER_DESC fractional = raster_desc(D3D12_CULL_MODE_NONE, FALSE);
    void* pso_back = create_stream_pso(guest, root, vs, ps, 1, back, 0);
    void* pso_front = create_stream_pso(guest, root, vs, ps, 1, front, 0);
    void* pso_dropped = create_stream_pso(guest, root, vs, ps, 0, back, 0);
    void* pso_r1_back = create_stream_pso(guest, root, vs, ps, 2, back, 0.0f);
    void* pso_r1_front = create_stream_pso(guest, root, vs, ps, 2, front, 0.0f);
    // 0.5 is not an integer depth bias. The pipeline must be refused, not drawn with cull none.
    void* pso_bias = create_stream_pso(guest, root, vs, ps, 2, fractional, 0.5f);
    if (!pso_back || !pso_front || !pso_dropped || !pso_r1_back || !pso_r1_front || !pso_bias) return 9;
    if (argc == 5) return 0;
    // Both mixed orders must be refused. Without the check, the final no-cull state draws pixels.
    void* mixed_10_27 = create_stream_pso(guest, root, vs, ps, 1, back, 0, true);
    void* mixed_27_10 = create_stream_pso(guest, root, vs, ps, 2, back, 0, true);
    if (!mixed_10_27 || !mixed_27_10) return 10;

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
        FAILED(static_cast<HRESULT>(invoke(guest, 36, 0, 0, ptr_arg(&__uuidof(ID3D12Fence)), ptr_arg(&state.fence))))) return 23;

    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT rect{0, 0, 4, static_cast<LONG>(height)};

    auto run_case = [&](UINT64 fence_val, const char* out_path, void* pso) -> int {
        if (FAILED(static_cast<HRESULT>(invoke(state.allocator, 8)))) return 24;
        if (FAILED(static_cast<HRESULT>(invoke(state.list, 139, ptr_arg(state.allocator))))) return 25;
        invoke(state.list, 52, rtv.ptr, ptr_arg(clear_color));
        invoke(state.list, 21, 1, ptr_arg(&viewport));
        invoke(state.list, 22, 1, ptr_arg(&rect));
        invoke(state.list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        invoke(state.list, 50, 1, ptr_arg(&rtv));
        invoke(state.list, 25, ptr_arg(pso));
        invoke(state.list, 12, 3, 1);
        if (const int rc = execute_guest_list(state.queue, state.list, state.fence, fence_val)) return rc;
        return read_stream_target(desktop.Get(), state, out_path);
    };

    if (const int rc = run_case(1, argv[5], pso_back)) return rc;
    if (const int rc = run_case(2, argv[6], pso_front)) return rc;
    if (const int rc = run_case(3, argv[7], pso_dropped)) return rc;
    if (const int rc = run_case(4, argv[8], pso_r1_back)) return rc;
    if (const int rc = run_case(5, argv[9], pso_r1_front)) return rc;
    if (const int rc = run_case(6, argv[10], pso_bias)) return rc;
    if (const int rc = run_case(7, argv[11], mixed_10_27)) return rc;
    if (const int rc = run_case(8, argv[12], mixed_27_10)) return rc;
    if (argc == 17) {
        int output = 13;
        for (int kind : {1, 2}) {
            for (auto cull : {D3D12_CULL_MODE_BACK, D3D12_CULL_MODE_FRONT}) {
                auto raster = raster_desc(cull, TRUE);
                raster.FrontCounterClockwise = TRUE;
                void* pso = create_stream_pso(guest, root, vs, ps, kind, raster, 0);
                if (!pso) return 9;
                if (const int rc = run_case(output - 4, argv[output], pso)) return rc;
                ++output;
            }
        }
    }
    return 0;
}
