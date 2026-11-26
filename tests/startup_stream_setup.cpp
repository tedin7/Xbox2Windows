#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <wrl/client.h>

#include <array>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "adapter_name_filter.h"
#include "startup_stream_setup.h"

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

struct alignas(8) Stage {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag;
    D3D12_SHADER_BYTECODE shader;
};
struct alignas(8) Root {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag;
    ID3D12RootSignature* signature;
};
struct Stream {
    Stage ps;
    Root root;
    Stage vs;
};
struct alignas(8) Layout {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE tag;
    D3D12_INPUT_LAYOUT_DESC desc;
};
// X2W_TEST_INPUT_LAYOUT: the stream also declares two per-vertex float3 elements, TEXCOORD in slot 0 and POSITION in
// slot 1 at D3D12_APPEND_ALIGNED_ELEMENT, in the opposite order of the vertex shader's input signature
// (startup_stream_fetch_check.py).
struct LayoutStream {
    Layout layout;
    Stream stream;
};
static const D3D12_INPUT_ELEMENT_DESC layout_elements[] = {
    {"TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
};

namespace {
void* stream_root = nullptr;
void* stream_alternate_root = nullptr;
std::vector<char> stream_root_bytes;

// X2W_TEST_ROOT_FLAGS: root signature flags (startup_root_heap.cpp). A root signature with flags is serialized as
// version 1.1, which carries the descriptor-heap indexing flags; without flags the version 1.0 blob is unchanged.
HRESULT serialize_root(const D3D12_ROOT_SIGNATURE_DESC& desc, ComPtr<ID3DBlob>& blob)
{
    if (!desc.Flags) return D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr);
    std::vector<D3D12_ROOT_PARAMETER1> parameters(desc.NumParameters);
    for (UINT i = 0; i < desc.NumParameters; ++i) {
        const D3D12_ROOT_PARAMETER& p = desc.pParameters[i];
        if (p.ParameterType != D3D12_ROOT_PARAMETER_TYPE_CBV && p.ParameterType != D3D12_ROOT_PARAMETER_TYPE_SRV) return E_INVALIDARG;
        parameters[i].ParameterType = p.ParameterType;
        parameters[i].Descriptor = {p.Descriptor.ShaderRegister, p.Descriptor.RegisterSpace, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};
        parameters[i].ShaderVisibility = p.ShaderVisibility;
    }
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC versioned{D3D_ROOT_SIGNATURE_VERSION_1_1};
    versioned.Desc_1_1 = {desc.NumParameters, parameters.data(), 0, nullptr, desc.Flags};
    return D3D12SerializeVersionedRootSignature(&versioned, &blob, nullptr);
}
}
void* stream_root_signature() { return stream_root; }
void* stream_alternate_root_signature() { return stream_alternate_root; }
const std::vector<char>& stream_root_blob() { return stream_root_bytes; }

int pick_stream_adapter(bool hardware, ComPtr<IDXGIAdapter>& adapter)
{
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 4;
    if (!hardware) return SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ? 0 : 4;

    AdapterNameFilter filter;
    if (!filter.read()) return 4;
    ComPtr<IDXGIFactory6> factory6;
    if (FAILED(factory.As(&factory6))) return 77;
    for (UINT n = 0;; ++n) {
        ComPtr<IDXGIAdapter1> candidate;
        const HRESULT hr = factory6->EnumAdapterByGpuPreference(
            n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
        if (hr == DXGI_ERROR_NOT_FOUND) return filter.no_match();
        if (FAILED(hr)) return 4;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(candidate->GetDesc1(&desc))) return 4;
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (filter.matches(desc.Description) && SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0,
                                                    __uuidof(ID3D12Device), nullptr))) {
            adapter = candidate;
            return 0;
        }
    }
}

int prepare_stream_pipeline(const char* vs_path, const char* ps_path, const char* root_path,
                            bool hardware, bool textured, Microsoft::WRL::ComPtr<IDXGIAdapter>& adapter,
                            void*& device, void*& pso)
{
    std::ifstream vs_file(vs_path, std::ios::binary), ps_file(ps_path, std::ios::binary);
    std::vector<char> vs{std::istreambuf_iterator<char>(vs_file), {}};
    std::vector<char> ps{std::istreambuf_iterator<char>(ps_file), {}};
    if (vs.empty() || ps.empty()) return 3;
    // X2W_TEST_DEBUG_LAYER=1 (startup_root_heap.cpp): the debug layer must be on before the first device exists.
    if (GetEnvironmentVariableW(L"X2W_TEST_DEBUG_LAYER", nullptr, 0)) {
        ComPtr<ID3D12Debug> debug;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            std::printf("no debug layer: skipped\n");
            return 77;
        }
        debug->EnableDebugLayer();
    }

    if (const int rc = pick_stream_adapter(hardware, adapter)) return rc;
    DXGI_ADAPTER_DESC desc{};
    LARGE_INTEGER driver{};
    if (FAILED(adapter->GetDesc(&desc))) return 4;
    adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver);
    std::printf("adapter=%ls driver=%u.%u.%u.%u\n", desc.Description,
        HIWORD(driver.HighPart), LOWORD(driver.HighPart), HIWORD(driver.LowPart), LOWORD(driver.LowPart));
    std::array<unsigned char, 64> parameters{};
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 5;

    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    const bool joined_table = GetEnvironmentVariableW(L"X2W_TEST_JOINED_TABLE", nullptr, 0) != 0;
    const bool vertex_srv_pair = GetEnvironmentVariableW(L"X2W_TEST_COPIED_VERTEX_SRV_PAIR", nullptr, 0) != 0;
    const D3D12_DESCRIPTOR_RANGE srv{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, joined_table || vertex_srv_pair ? 2u : 1u, 0, 0, 0};
    D3D12_ROOT_PARAMETER textured_parameters[2]{};
    if (textured) {
        D3D12_ROOT_PARAMETER& table = textured_parameters[0];
        table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        table.DescriptorTable = {1, &srv};
        table.ShaderVisibility = vertex_srv_pair ? D3D12_SHADER_VISIBILITY_VERTEX : D3D12_SHADER_VISIBILITY_PIXEL;
        root_desc.NumParameters = 1;
        root_desc.pParameters = textured_parameters;
        if (joined_table || GetEnvironmentVariableW(L"X2W_TEST_ROOT_RESET", nullptr, 0)) {
            D3D12_ROOT_PARAMETER& constant = textured_parameters[1];
            constant.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            constant.Constants = {0, 0, 1};
            constant.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            root_desc.NumParameters = 2;
        }
    }
    // X2W_TEST_ROOT_CBV=<mode> declares root parameter 0 as a constant buffer view (startup_root_cbv.cpp);
    // "wrong_type" declares a shader resource view there instead.
    D3D12_ROOT_PARAMETER root_cbv{};
    char cbv_mode[16]{};
    if (!textured && GetEnvironmentVariableA("X2W_TEST_ROOT_CBV", cbv_mode, sizeof(cbv_mode))) {
        root_cbv.ParameterType = std::strcmp(cbv_mode, "wrong_type") == 0 ? D3D12_ROOT_PARAMETER_TYPE_SRV
                                                                          : D3D12_ROOT_PARAMETER_TYPE_CBV;
        root_cbv.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        root_desc.NumParameters = 1;
        root_desc.pParameters = &root_cbv;
        char flags[16]{};
        if (GetEnvironmentVariableA("X2W_TEST_ROOT_FLAGS", flags, sizeof(flags)))
            root_desc.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(std::strtoul(flags, nullptr, 0));
    }
    // A runtime or device that rejects the requested root signature flags is a capability skip.
    const int rejected = root_desc.Flags ? 77 : 0;
    ComPtr<ID3DBlob> blob;
    if (FAILED(serialize_root(root_desc, blob))) return rejected ? rejected : 6;
    stream_root_bytes.assign(static_cast<const char*>(blob->GetBufferPointer()),
                             static_cast<const char*>(blob->GetBufferPointer()) + blob->GetBufferSize());
    std::ofstream(root_path, std::ios::binary).write(static_cast<const char*>(blob->GetBufferPointer()), blob->GetBufferSize());
    using CreateRoot = HRESULT (WINAPI*)(void*, UINT, const void*, SIZE_T, REFIID, void**);
    using CreateStream = HRESULT (WINAPI*)(void*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    auto** slots = *static_cast<void***>(device);
    ID3D12RootSignature* root = nullptr;
    if (FAILED(reinterpret_cast<CreateRoot>(slots[16])(device, 0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                        __uuidof(ID3D12RootSignature), reinterpret_cast<void**>(&root)))) {
        if (rejected) std::printf("CreateRootSignature rejects flags 0x%x: skipped\n", static_cast<unsigned>(root_desc.Flags));
        return rejected ? rejected : 7;
    }
    stream_root = root;
    if (std::strcmp(cbv_mode, "switch_root") == 0) {
        root_cbv.Descriptor.ShaderRegister = 1;
        ComPtr<ID3DBlob> alternate;
        if (FAILED(serialize_root(root_desc, alternate)) ||
            FAILED(reinterpret_cast<CreateRoot>(slots[16])(device, 0, alternate->GetBufferPointer(),
                                                            alternate->GetBufferSize(), __uuidof(ID3D12RootSignature),
                                                            &stream_alternate_root))) return 7;
    }

    wchar_t layout_mode[8] = L"";
    GetEnvironmentVariableW(L"X2W_TEST_INPUT_LAYOUT", layout_mode, 8);
    const bool empty_layout = std::wcscmp(layout_mode, L"empty") == 0;  // declared layout with zero elements
    LayoutStream with_layout{{D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, {layout_elements, empty_layout ? 0u : 2u}},
                             {{D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, {ps.data(), ps.size()}},
                              {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root},
                              {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, {vs.data(), vs.size()}}}};
    const bool layout = GetEnvironmentVariableW(L"X2W_TEST_INPUT_LAYOUT", nullptr, 0) != 0;
    void* first = layout ? static_cast<void*>(&with_layout) : static_cast<void*>(&with_layout.stream);
    const SIZE_T whole = layout ? sizeof(with_layout) : sizeof(with_layout.stream);
    auto submit = [&](SIZE_T bytes) {
        D3D12_PIPELINE_STATE_STREAM_DESC desc{bytes, first};
        const auto hr = reinterpret_cast<CreateStream>(slots[86])(device, &desc, __uuidof(ID3D12PipelineState), &pso);
        return SUCCEEDED(hr) && pso;
    };
    // A truncated last shader must not publish a partial request, even with readable backing memory.
    if (!submit(whole - 8)) return 8;
    if (GetEnvironmentVariableW(L"X2W_TEST_TRUNCATED_ONLY", nullptr, 0)) return 0;
    if (!submit(whole)) return 9;
    return 0;
}

int read_stream_target(ID3D12Device* desktop, StreamTarget& state, const char* pixels)
{
    auto& targets = state.attachments;
    const UINT count = state.count;
    const UINT height = targets[0].texture->GetDesc().Height;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    ComPtr<ID3D12CommandQueue> observer;
    ComPtr<ID3D12CommandAllocator> copy_allocator;
    ComPtr<ID3D12GraphicsCommandList> copy;
    ComPtr<ID3D12Fence> complete;
    if (FAILED(desktop->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&observer))) ||
        FAILED(desktop->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&copy_allocator))) ||
        FAILED(desktop->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, copy_allocator.Get(), nullptr, IID_PPV_ARGS(&copy))) ||
        FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&complete)))) return 26;

    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    for (UINT i = 0; i < count; ++i) {
        auto& target = targets[i];
        UINT64 total = 0;
        const auto desc = target.texture->GetDesc();
        desktop->GetCopyableFootprints(&desc, 0, 1, 0, &target.footprint, nullptr, nullptr, &total);
        const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, total, 1, 1, 1,
                                   DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&target.readback)))) return 27;
        D3D12_TEXTURE_COPY_LOCATION destination{target.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        destination.PlacedFootprint = target.footprint;
        D3D12_TEXTURE_COPY_LOCATION source{target.texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        copy->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    }
    if (FAILED(copy->Close())) return 28;
    ID3D12CommandList* submit[] = {copy.Get()};
    observer->ExecuteCommandLists(1, submit);
    if (FAILED(observer->Signal(complete.Get(), 1))) return 29;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return 30;
    const bool completed = SUCCEEDED(complete->SetEventOnCompletion(1, event)) &&
                           WaitForSingleObject(event, 10000) == WAIT_OBJECT_0 && complete->GetCompletedValue() == 1;
    CloseHandle(event);
    if (!completed) return 31;
    std::ofstream out(pixels, std::ios::binary);
    for (UINT i = 0; i < count; ++i) {
        auto& target = targets[i];
        void* memory = nullptr;
        if (FAILED(target.readback->Map(0, nullptr, &memory))) return 32;
        for (UINT y = 0; y < height; ++y)
            out.write(static_cast<const char*>(memory) + y * target.footprint.Footprint.RowPitch, target.row_bytes);
        target.readback->Unmap(0, nullptr);
    }
    return out ? 0 : 33;
}

using Word = std::uint64_t;
template<class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a, Word b, Word c, Word d,
            Word e, Word f, Word g, Word h)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f, g, h);
}

int open_stream_target(ID3D12Device* desktop, void* guest, void* pso, UINT width, UINT height,
                       DXGI_FORMAT format, UINT scissor_width, StreamTarget& state)
{
    StreamMrtFormat mrt{};
    const bool typed = stream_mrt_format(format, mrt);
    const bool integer = typed && mrt.integer;
    const bool narrow = format == DXGI_FORMAT_A8_UNORM || format == DXGI_FORMAT_R8_UNORM;
    // A caller that wants a second RGBA8 target sets state.count to 2 before the call.
    const UINT count = state.count = integer || state.count == 2 ? 2 : 1;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    // The wrapped heap gives the stand-in a descriptor shadow for the typed RTV.
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } rtv_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, count}, 0, 0};
    auto& rtv_heap = state.rtv_heap;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, arg(&rtv_desc), arg(&__uuidof(ID3D12DescriptorHeap)),
                                           arg(&rtv_heap))))) return 22;
    const SIZE_T first_rtv = invoke(rtv_heap, 9);
    const auto stride = desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    auto& targets = state.attachments;
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 2> rtvs{};
    const float clear[] = {0, 0, 1, narrow ? 0.0f : 1.0f};
    for (UINT i = 0; i < count; ++i) {
        auto& target = targets[i];
        const auto view_format = i ? DXGI_FORMAT_R8G8B8A8_UNORM : format;
        const auto resource_format = !i && integer ? mrt.typeless : view_format;
        D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
            resource_format, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
        const D3D12_CLEAR_VALUE optimized_clear{view_format, {clear[0], clear[1], clear[2], clear[3]}};
        if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COMMON, &optimized_clear, IID_PPV_ARGS(&target.texture)))) return 21;
        D3D12_RENDER_TARGET_VIEW_DESC view{view_format, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
        rtvs[i] = {first_rtv + SIZE_T{i} * stride};
        invoke(guest, 20, arg(target.texture.Get()), arg(&view), rtvs[i].ptr);
        target.row_bytes = width * (!i && typed ? mrt.bytes : view_format == DXGI_FORMAT_A8_UNORM ||
                                    view_format == DXGI_FORMAT_R8_UNORM ? 1 : 4);
    }

    auto& list = state.list;
    auto& queue = state.queue;
    auto& allocator = state.allocator;
    auto& fence = state.fence;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(guest, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&queue)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&fence))))) return 23;

    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT rect{0, 0, static_cast<LONG>(scissor_width), static_cast<LONG>(height)};
    invoke(list, 139, arg(allocator));
    for (UINT i = 0; i < count; ++i) invoke(list, 52, rtvs[i].ptr, arg(clear));
    invoke(list, 50, count, arg(rtvs.data()));
    // X2W_TEST_VIEWPORT: "none" and "grow" never set a viewport or scissor; "rect" and "reset" set a rectangle smaller than the target.
    wchar_t mode[8] = L"";
    GetEnvironmentVariableW(L"X2W_TEST_VIEWPORT", mode, 8);
    const bool none = std::wcscmp(mode, L"none") == 0 || std::wcscmp(mode, L"grow") == 0;
    const bool inset = std::wcscmp(mode, L"rect") == 0 || std::wcscmp(mode, L"reset") == 0;
    const D3D12_VIEWPORT inner{2, 1, 4, 2, 0, 1};
    const D3D12_RECT inner_rect{2, 1, 6, 3};
    if (!none) {
        invoke(list, 21, 1, inset ? arg(&inner) : arg(&viewport));
        invoke(list, 22, 1, inset ? arg(&inner_rect) : arg(&rect));
    }
    invoke(list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    invoke(list, 25, arg(pso));
    return 0;
}
