#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <iterator>
#include <cstring>
#include "triangle_vs.h"
#include "triangle_ps.h"
#include "ffx_provider_loop.h"
#include "dispatch_index_cs.h"
#include "group_index_cs.h"
#include "desktop_gpu.h"
#include "dxr_opacity_reference.h"

void check_spirv_window_atomic(ID3D12Device* device, const char* shader_path);
void check_spirv_window_signed(ID3D12Device* device, const char* shader_path);
void check_spirv_window_float(ID3D12Device* device, const char* shader_path);
void check_spirv_window_vec4(ID3D12Device* device, const char* shader_path);
void check_spirv_window_vec4_narrow(ID3D12Device* device, const char* shader_path);
void check_spirv_window_qword(ID3D12Device* device, const char* shader_path);
void check_spirv_window_qword_narrow(ID3D12Device* device, const char* shader_path);
void check_sampler_index_control(ID3D12Device* device,
                                const unsigned char* dispatch_bytecode, unsigned int dispatch_size,
                                const unsigned char* group_bytecode, unsigned int group_size);

#ifdef OFFSCREEN_AGILITY_620
extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion = 620;
__declspec(dllexport) const char* D3D12SDKPath = ".\\D3D12\\";
}
#endif

using Microsoft::WRL::ComPtr;
constexpr UINT side = 64;

void check_descriptor_heap_paging(ID3D12Device* device, bool swapped);

bool contains_ignoring_case(const wchar_t* text, const wchar_t* part)
{
    const size_t length = std::wcslen(part);
    for (; *text; ++text) {
        if (_wcsnicmp(text, part, length) == 0) return true;
    }
    return false;
}

// Without XBOX2WINDOWS_ADAPTER, the first D3D12 hardware adapter in high-performance order is used.
// With it, the first whose description contains the value, ignoring case, so a PC with several GPUs
// can test each one. A value that matches no adapter is an error, not a skipped test.
ComPtr<IDXGIAdapter1> select_adapter(IDXGIFactory6* factory, bool warp, ComPtr<ID3D12Device>& device
#ifdef OFFSCREEN_AGILITY_620
    , ID3D12DeviceFactory* device_factory
#endif
)
{
    auto create_device = [&](IDXGIAdapter1* candidate) {
#ifdef OFFSCREEN_AGILITY_620
        return device_factory->CreateDevice(candidate, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.ReleaseAndGetAddressOf()));
#else
        return D3D12CreateDevice(candidate, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.ReleaseAndGetAddressOf()));
#endif
    };
    auto supports_d3d12 = [&](IDXGIAdapter1* candidate) {
#ifdef OFFSCREEN_AGILITY_620
        return SUCCEEDED(device_factory->CreateDevice(candidate, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr));
#else
        return SUCCEEDED(D3D12CreateDevice(candidate, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr));
#endif
    };
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
        check(create_device(adapter.Get()), "Create device");
        return adapter;
    }
    wchar_t wanted[128]{};
    const DWORD filter = GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", wanted, 128);
    if (filter >= 128) throw std::runtime_error("XBOX2WINDOWS_ADAPTER is longer than 127 characters");
    for (UINT index = 0;; ++index) {
        HRESULT hr = factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                        IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf()));
        if (hr == DXGI_ERROR_NOT_FOUND) {
            if (!filter) return {};
            std::fprintf(stderr, "No D3D12 hardware adapter matches XBOX2WINDOWS_ADAPTER=%ls\n", wanted);
            throw std::runtime_error("Select adapter");
        }
        check(hr, "Enumerate hardware");
        DXGI_ADAPTER_DESC1 desc{};
        check(adapter->GetDesc1(&desc), "Adapter description");
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (!filter || contains_ignoring_case(desc.Description, wanted)) {
            if (SUCCEEDED(create_device(adapter.Get()))) return adapter;
        } else if (supports_d3d12(adapter.Get())) {
            std::fprintf(stderr, "XBOX2WINDOWS_ADAPTER: skipping %ls\n", desc.Description);
        }
    }
}

void save_bitmap(const char* path, const std::vector<BYTE>& bgra)
{
    BITMAPINFOHEADER info{};
    info.biSize = sizeof(info);
    info.biWidth = side;
    info.biHeight = -static_cast<LONG>(side); // top-down image
    info.biPlanes = 1;
    info.biBitCount = 32;
    BITMAPFILEHEADER header{};
    header.bfType = 0x4d42;
    header.bfOffBits = sizeof(header) + sizeof(info);
    header.bfSize = header.bfOffBits + static_cast<DWORD>(bgra.size());
    std::ofstream file(path, std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(&info), sizeof(info));
    file.write(reinterpret_cast<const char*>(bgra.data()), bgra.size());
    file.close();
}

std::vector<char> read_shader(const char* path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read reference shader");
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void draw(ID3D12Device* device, const char* output, char** reference_shaders)
{
    const bool reference = reference_shaders != nullptr;
    const auto format = reference ? DXGI_FORMAT_B8G8R8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    if (reference) root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
          "Serialize root signature");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
          "Create root signature");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = root.Get();
    pso.VS = {triangle_vs, sizeof(triangle_vs)};
    pso.PS = {triangle_ps, sizeof(triangle_ps)};
    std::vector<char> vs, ps;
    const D3D12_INPUT_ELEMENT_DESC elements[] = {
        {"SV_Position", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };
    if (reference) {
        vs = read_shader(reference_shaders[0]);
        ps = read_shader(reference_shaders[1]);
        ComPtr<ID3D12RootSignatureDeserializer> embedded, absent;
        HRESULT found = D3D12CreateRootSignatureDeserializer(vs.data(), vs.size(), IID_PPV_ARGS(&embedded));
        HRESULT missing = D3D12CreateRootSignatureDeserializer(triangle_vs, sizeof(triangle_vs), IID_PPV_ARGS(&absent));
        std::printf("desktop_blob_embedded=0x%08lx desktop_blob_no_signature=0x%08lx\n",
                    static_cast<unsigned long>(found), static_cast<unsigned long>(missing));
        check(found, "Read signature from complete reference shader");
        if (!embedded.Get() || SUCCEEDED(missing) || absent.Get())
            throw std::runtime_error("Unexpected embedded-signature result");
        pso.VS = {vs.data(), vs.size()};
        pso.PS = {ps.data(), ps.size()};
        pso.InputLayout = {elements, 2};
    }
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    auto& blend = pso.BlendState.RenderTarget[0];
    blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
    blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.SampleMask = UINT_MAX;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = format;
    pso.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline)), "Create graphics pipeline");

    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = tex.Height = side;
    tex.DepthOrArraySize = tex.MipLevels = 1;
    tex.SampleDesc.Count = 1;
    tex.Format = format;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear{};
    clear.Format = format;
    clear.Color[3] = 1;
    if (reference) clear.Color[0] = clear.Color[1] = clear.Color[2] = 0.052860655f;
    auto target = resource(device, tex, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    device->GetCopyableFootprints(&tex, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
    // Exercise a padded staging layout even when the texture width is already aligned.
    footprint.Footprint.RowPitch += D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    const UINT64 byte_count = footprint.Offset + UINT64{footprint.Footprint.RowPitch} * side;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = byte_count;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    auto readback = resource(device, buffer, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heap_desc.NumDescriptors = 1;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptors)), "Create RTV heap");
    const auto rtv = descriptors->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    ComPtr<ID3D12CommandQueue> queue;
    check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "Create queue");
    ComPtr<ID3D12CommandAllocator> allocator;
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "Create allocator");
    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(),
                                    IID_PPV_ARGS(&list)), "Create command list");
    const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
    const D3D12_RECT scissor{0, 0, side, side};
    list->SetGraphicsRootSignature(root.Get());
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ComPtr<ID3D12Resource> vertex_buffer;
    if (reference) {
        // Positions and linear RGBA from Microsoft's pinned SimpleTriangle sample.
        constexpr float vertices[][8] = {
            {0, .5f, .5f, 1, 1, 0, 0, 1}, {.5f, -.5f, .5f, 1, 0, 1, 0, 1},
            {-.5f, -.5f, .5f, 1, 0, 0, 1, 1}
        };
        auto vertex_desc = buffer;
        vertex_desc.Width = sizeof(vertices);
        vertex_buffer = resource(device, vertex_desc, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped_vertices = nullptr;
        const D3D12_RANGE empty{};
        check(vertex_buffer->Map(0, &empty, &mapped_vertices), "Map reference vertices");
        std::memcpy(mapped_vertices, vertices, sizeof(vertices));
        vertex_buffer->Unmap(0, nullptr);
        const D3D12_VERTEX_BUFFER_VIEW view{
            vertex_buffer->GetGPUVirtualAddress(), sizeof(vertices), sizeof(vertices[0])};
        list->IASetVertexBuffers(0, 1, &view);
    }
    list->DrawInstanced(3, 1, 0, 0);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {target.Get(), 0, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = target.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    check(list->Close(), "Close command list");
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create fence");
    ID3D12CommandList* submitted[]{list.Get()};
    queue->ExecuteCommandLists(1, submitted);
    check(queue->Signal(fence.Get(), 1), "Signal GPU fence");
    check(fence->SetEventOnCompletion(1, nullptr), "Wait for GPU");
    check(device->GetDeviceRemovedReason(), "GPU completion");

    std::vector<BYTE> bgra(side * side * 4);
    void* mapped = nullptr;
    D3D12_RANGE range{0, static_cast<SIZE_T>(byte_count)};
    check(readback->Map(0, &range, &mapped), "Map pixel readback");
    unsigned mismatches = 0;
    for (UINT y = 0; y < side; ++y) {
        const BYTE* row = static_cast<const BYTE*>(mapped) + footprint.Offset + y * footprint.Footprint.RowPitch;
        for (UINT x = 0; x < side; ++x) {
            const BYTE* rgba = row + x * 4;
            auto* pixel = &bgra[(y * side + x) * 4];
            if (reference) {
                std::memcpy(pixel, rgba, 4); // Reference target already uses BGRA.
                continue;
            }
            const bool inside = x >= 8 && y >= 8 && x + y <= 63;
            if (rgba[0] != 0 || rgba[1] != (inside ? 255 : 0) || rgba[2] != 0 || rgba[3] != 255) {
                ++mismatches;
            }
            pixel[0] = rgba[2];
            pixel[1] = rgba[1];
            pixel[2] = rgba[0];
            pixel[3] = rgba[3];
        }
    }
    const D3D12_RANGE no_writes{0, 0};
    readback->Unmap(0, &no_writes);
    std::printf("pixels=%u row_pitch=%u\n", side * side, footprint.Footprint.RowPitch);
    if (!reference) std::printf("mismatches=%u\n", mismatches);
    save_bitmap(output, bgra);
    if (mismatches) throw std::runtime_error("GPU pixels differ from CPU reference");
}

bool report_debug_errors(ID3D12InfoQueue* messages)
{
    if (!messages) return false;
    unsigned errors = 0;
    unsigned warnings = 0;
    const auto count = messages->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T bytes = 0;
        check(messages->GetMessage(i, nullptr, &bytes), "Read debug message size");
        std::vector<BYTE> storage(bytes);
        auto* msg = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        check(messages->GetMessage(i, msg, &bytes), "Read debug message");
        if (msg->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
            std::fprintf(stderr, "D3D12 %d: %s\n", msg->ID, msg->pDescription);
            if (msg->Severity == D3D12_MESSAGE_SEVERITY_WARNING) ++warnings;
            else ++errors;
        }
    }
    std::printf("debug_errors=%u debug_warnings=%u\n", errors, warnings);
    return errors != 0;
}

struct AdapterCaps {
    D3D_SHADER_MODEL highest_shader_model{};
    bool int64_shader_ops = false;
    bool wave_ops = false;
    UINT wave_lane_count_min = 0;
    UINT wave_lane_count_max = 0;
};

AdapterCaps query_adapter_caps(ID3D12Device* device)
{
    AdapterCaps caps{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    check(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)),
          "Query adapter D3D12 options1");
    caps.int64_shader_ops = options1.Int64ShaderOps != FALSE;
    caps.wave_ops = options1.WaveOps != FALSE;
    caps.wave_lane_count_min = options1.WaveLaneCountMin;
    caps.wave_lane_count_max = options1.WaveLaneCountMax;

    // Older runtimes reject shader-model enums above what they recognize with E_INVALIDARG.
    for (D3D_SHADER_MODEL requested : {
             D3D_SHADER_MODEL_6_6,
             D3D_SHADER_MODEL_6_5,
             D3D_SHADER_MODEL_6_4,
             D3D_SHADER_MODEL_6_3,
             D3D_SHADER_MODEL_6_2,
             D3D_SHADER_MODEL_6_1,
             D3D_SHADER_MODEL_6_0,
             D3D_SHADER_MODEL_5_1,
         }) {
        D3D12_FEATURE_DATA_SHADER_MODEL model{requested};
        const HRESULT hr = check_feature(device, D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model));
        if (hr == E_INVALIDARG) continue;
        check(hr, "Query adapter shader model");
        caps.highest_shader_model = model.HighestShaderModel;
        break;
    }
    return caps;
}

enum class OutcomeKind { pass, skip, fail };

struct OperationResult {
    OutcomeKind kind = OutcomeKind::pass;
    const char* detail = nullptr;
};

enum class CapsDiagnostic {
    none,
    shader_model_6_0,
    shader_model_6_5,
    wave_6_6,
};

struct CapabilityGate {
    D3D_SHADER_MODEL min_shader_model{};
    bool require_int64_ops = false;
    bool require_wave_ops = false;
    UINT required_wave_lane_count = 0;
    const char* skip_reason = nullptr;
    CapsDiagnostic diagnostic = CapsDiagnostic::none;
};

struct OperationContext {
    ID3D12Device* device = nullptr;
    ID3D12InfoQueue* messages = nullptr;
    const AdapterCaps& caps;
    std::string_view mode;
    int argc = 0;
    char** argv = nullptr;
    void* user_data = nullptr;
};

using ValidateArgsFn = bool (*)(int argc, char** argv);
using RunOperationFn = OperationResult (*)(const OperationContext& ctx);

struct OperationSpec {
    std::string_view name;
    int min_argc;
    int max_argc;
    ValidateArgsFn validate_args;
    CapabilityGate gate;
    RunOperationFn run;
};

constexpr int kUnboundedArgc = (std::numeric_limits<int>::max)();

OperationResult skip_if_false(bool ok, const char* skip_detail = nullptr)
{
    return ok ? OperationResult{OutcomeKind::pass, nullptr}
              : OperationResult{OutcomeKind::skip, skip_detail};
}

bool validate_check_pixel_args(int, char** argv)
{
    const std::string_view format = argv[4];
    return format == "rgba" || format == "bgra";
}

bool validate_pixel_pairs_args(int argc, char**)
{
    return (argc - 3) % 2 == 0;
}

bool validate_mesh_entry_args(int argc, char** argv)
{
    if (argc == 6) return true;
    const std::string_view order = argv[6];
    return order == "ab" || order == "ba";
}

bool validate_mesh_entry_table_args(int argc, char** argv)
{
    return argc == 6 || std::string_view(argv[6]) == "--swap";
}

bool validate_compute_mesh_texture_args(int argc, char** argv)
{
    return argc != 9 || std::string_view(argv[8]) == "--stale";
}

bool validate_compute_mesh_texture_fallback_args(int argc, char** argv)
{
    return std::string_view(argv[7]) != "--stale" &&
           (argc < 9 || std::string_view(argv[8]) != "--stale") &&
           (argc != 10 || std::string_view(argv[9]) == "--stale");
}

template<class AfterCase>
OperationResult run_readback_batch(const OperationContext& ctx, AfterCase after_case)
{
    const std::string_view operation(ctx.argv[2]);
    const bool pairs = operation == "--check-pixel-pairs";
    const bool compute = operation == "--compute-store-batch";
    const int stride = pairs ? 2 : 1;
    ComputeReadbackProgram program;
    bool failed = false;
    for (int i = 4, index = 0; i < ctx.argc; i += stride, ++index) {
        const char* shader = ctx.argv[pairs ? i - 1 : 3];
        const char* status = "pass";
        std::fprintf(stderr, "case=%d path=%s\n", index, ctx.argv[i]);
        if (pairs) std::fprintf(stderr, "shader=%s\n", shader);
        try {
            check(ctx.device->GetDeviceRemovedReason(), compute ? "Compute batch device" : "Pixel batch device");
            if (compute) {
                if (!program.pipeline) program = load_compute_store_program(ctx.device, shader);
                check_compute_store_reference(ctx.device, program, ctx.argv[i]);
            } else {
                check_translated_pixel(ctx.device, shader, false, ctx.argv[i]);
            }
            after_case(index);
        } catch (const ReadbackMismatch& e) {
            std::fprintf(stderr, "%s\n", e.what());
            status = "mismatch";
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            status = "error";
        }
        if (report_debug_errors(ctx.messages) || FAILED(ctx.device->GetDeviceRemovedReason()))
            status = "error";
        if (ctx.messages) ctx.messages->ClearStoredMessages();
        std::printf("case=%d status=%s\n", index, status);
        failed |= std::string_view(status) != "pass";
    }
    return failed ? OperationResult{OutcomeKind::fail, nullptr} : OperationResult{OutcomeKind::pass, nullptr};
}

OperationResult run_check_batch(const OperationContext& ctx)
{
    return run_readback_batch(ctx, [](int) {});
}

OperationResult run_pixel_batch_debug_control(const OperationContext& ctx)
{
    if (!ctx.messages) return {OutcomeKind::skip, "skip: pixel batch control requires the D3D12 debug layer"};
    return run_readback_batch(ctx, [&](int index) {
        if (index == 1)
            check(ctx.messages->AddApplicationMessage(D3D12_MESSAGE_SEVERITY_ERROR, "pixel batch control error"),
                  "Add pixel batch control message");
    });
}

OperationResult run_pixel_batch_removal_control(const OperationContext& ctx)
{
    ComPtr<ID3D12Device5> device;
    const HRESULT hr = ctx.device->QueryInterface(IID_PPV_ARGS(&device));
    if (hr == E_NOINTERFACE) return {OutcomeKind::skip, "skip: pixel batch control requires ID3D12Device5"};
    check(hr, "Query pixel batch control device");
    return run_readback_batch(ctx, [&](int index) {
        if (index == 1) device->RemoveDevice();
    });
}

OperationResult run_compute_root(const OperationContext& ctx)
{
    check_compute(ctx.device, false);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_compute_tables(const OperationContext& ctx)
{
    check_compute(ctx.device, true);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_compute_root_file(const OperationContext& ctx)
{
    check_compute_file(ctx.device, ctx.argv[3], false);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_ffx_immediate_conversion(const OperationContext& ctx)
{
    check_compute_file(ctx.device, ctx.argv[3], true);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_compute_store_case(const OperationContext& ctx)
{
    check_compute_file(ctx.device, ctx.argv[3], false, ctx.argv[4]);
    return {OutcomeKind::pass, nullptr};
}

// The store case with both raw buffers bound as descriptor-table views (defaulting to the full 1280
// bytes, or an explicit uav_words count in [1, 320]), so a shader reading a descriptor's size sees
// the view size (root descriptors carry none).
OperationResult run_compute_store_sized_case(const OperationContext& ctx)
{
    UINT uav_words = 0;
    if (ctx.argc == 6) {
        const std::string_view text = ctx.argv[5];
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), uav_words);
        if (error != std::errc{} || end != text.data() + text.size() || uav_words == 0)
            throw std::runtime_error("Invalid compute UAV word count");
    }
    check_compute_file(ctx.device, ctx.argv[3], false, ctx.argv[4], false, true, uav_words);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_compute_store_dpp_case(const OperationContext& ctx)
{
    check_compute_file(ctx.device, ctx.argv[3], false, ctx.argv[4], true);
    return {OutcomeKind::pass, nullptr};
}

// The image-load readback extends the store case with the source texture words and uses the
// descriptor-table signature, because a texture SRV cannot be a root descriptor.
OperationResult run_compute_texture_case(const OperationContext& ctx)
{
    check_compute_texture_file(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argc == 6 ? ctx.argv[5] : nullptr);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_compute_texture_msaa_case(const OperationContext& ctx)
{
    return skip_if_false(check_compute_texture_msaa_file(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], ctx.argv[6]),
                         "R32_UINT 4x MSAA render target and load support unavailable");
}

OperationResult run_compute_texture_store_case(const OperationContext& ctx)
{
    return skip_if_false(check_compute_texture_store_file(ctx.device, ctx.argv[3], ctx.argv[4]),
                         "Required typed UAV operations unavailable for the image case format");
}

OperationResult run_spirv_sampler_views(const OperationContext& ctx)
{
    check_sampler_views(ctx.device, ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_dynamic_sampler_index(const OperationContext& ctx)
{
    check_dynamic_sampler_index(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

// Two builds of one shader that differ only in the sampler descriptor index source (#1115). The
// SV_DispatchThreadID build must match the shared expected words; the SV_GroupID build is printed
// for the record, because an adapter that folds the index still matches on some drivers.
OperationResult run_sampler_index_control(const OperationContext& ctx)
{
    (void)ctx;
    check_sampler_index_control(ctx.device, sampler_index_dispatch_cs,
                                static_cast<unsigned int>(sizeof(sampler_index_dispatch_cs)),
                                sampler_index_group_cs,
                                static_cast<unsigned int>(sizeof(sampler_index_group_cs)));
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_raw_uav_dump(const OperationContext& ctx)
{
    dump_raw_uav(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], ctx.argv[6]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_spirv_address_window(const OperationContext& ctx)
{
    check_spirv_window(ctx.device, ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_spirv_address_window_atomic(const OperationContext& ctx)
{
    check_spirv_window_atomic(ctx.device, ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_spirv_address_window_signed(const OperationContext& ctx)
{
    check_spirv_window_signed(ctx.device, ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_spirv_address_window_float(const OperationContext& ctx)
{
    check_spirv_window_float(ctx.device, ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_spirv_address_window_vec4(const OperationContext& ctx)
{
    check_spirv_window_vec4(ctx.device, ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_spirv_address_window_vec4_narrow(const OperationContext& ctx)
{
    check_spirv_window_vec4_narrow(ctx.device, ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_spirv_address_window_qword(const OperationContext& ctx)
{
    check_spirv_window_qword(ctx.device, ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_spirv_address_window_qword_narrow(const OperationContext& ctx)
{
    check_spirv_window_qword_narrow(ctx.device, ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_spirv_compare_sampler(const OperationContext& ctx)
{
    check_spirv_compare_sampler(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5],
                                ctx.argc == 7 ? ctx.argv[6] : nullptr);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_spirv_compaction(const OperationContext& ctx)
{
    check_spirv_compaction(ctx.device, ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_ffx_device_init(const OperationContext& ctx)
{
    check_ffx_device_init(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_ffx_provider_loop(const OperationContext& ctx)
{
    check_ffx_provider_loop(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_ffx_recurrent(const OperationContext& ctx)
{
    check_ffx_recurrent(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_dual_uav(const OperationContext& ctx)
{
    check_dual_uav(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_stream_root_srv(const OperationContext& ctx)
{
    check_stream_root_srv(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_descriptor_window_submissions(const OperationContext& ctx)
{
    check_descriptor_window_submissions(ctx.device, DescriptorWindowCase::direct);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_cpu_srv_refresh(const OperationContext& ctx)
{
    check_cpu_srv_refresh(ctx.device, std::string_view(ctx.argv[2]) == "--cpu-srv-stale");
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_descriptor_window_stale(const OperationContext& ctx)
{
    check_descriptor_window_submissions(ctx.device, DescriptorWindowCase::stale_window);
    return {OutcomeKind::pass, nullptr};
}

bool validate_descriptor_record_args(int, char** argv)
{
    const std::string_view scenario = argv[3];
    return scenario == "live" || scenario == "stale" || scenario == "unknown";
}

OperationResult run_descriptor_record_copy(const OperationContext& ctx)
{
    const std::string_view scenario = ctx.argv[3];
    auto test = DescriptorWindowCase::copied_record;
    if (scenario == "stale") test = DescriptorWindowCase::stale_record;
    if (scenario == "unknown") test = DescriptorWindowCase::unknown_record;
    check_descriptor_window_submissions(ctx.device, test);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_texture_mips(const OperationContext& ctx)
{
    check_texture_mips(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_bc7_tail(const OperationContext& ctx)
{
    check_bc7_tail(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argc - 5, ctx.argv + 5);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_bc7_mip1(const OperationContext& ctx)
{
    check_bc7_pixel(ctx.device, ctx.argv[3], ctx.argv[4], nullptr, ctx.argv[5]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_bc7(const OperationContext& ctx)
{
    check_bc7_pixel(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argc == 6 ? ctx.argv[5] : nullptr);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_dual_pixel(const OperationContext& ctx)
{
    check_dual_pixel(ctx.device, ctx.argv[3], ctx.argv[4], false);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_dual_pixel_swapped(const OperationContext& ctx)
{
    check_dual_pixel(ctx.device, ctx.argv[3], ctx.argv[4], true);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_sampled_coverage(const OperationContext& ctx)
{
    check_sampled_coverage(ctx.device, ctx.argv[3], ctx.argv[4]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel_compare_mips(const OperationContext& ctx)
{
// Windows SDK 10.0.22621 does not declare the SM 6.8 capability query.
#if defined(__ID3D12Device14_INTERFACE_DEFINED__)
    const bool level = std::strcmp(ctx.argv[2], "--check-pixel-compare-level") == 0;
    const D3D_SHADER_MODEL required = level ? D3D_SHADER_MODEL_6_7 : D3D_SHADER_MODEL_6_8;
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{required};
    if (FAILED(check_feature(ctx.device, D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))) ||
        shaderModel.HighestShaderModel < required)
        return {OutcomeKind::skip, level ? "Comparison LOD requires SM 6.7" : "Comparison bias/gradient requires SM 6.8"};
    if (!level) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS21 options{};
        if (FAILED(check_feature(ctx.device, D3D12_FEATURE_D3D12_OPTIONS21, &options, sizeof(options))) ||
            !options.SampleCmpGradientAndBiasSupported)
            return {OutcomeKind::skip, "Comparison bias/gradient requires SampleCmpGradientAndBiasSupported"};
    } else {
        D3D12_FEATURE_DATA_D3D12_OPTIONS14 options{};
        if (FAILED(check_feature(ctx.device, D3D12_FEATURE_D3D12_OPTIONS14, &options, sizeof(options))) ||
            !options.AdvancedTextureOpsSupported)
            return {OutcomeKind::skip, "Comparison LOD requires AdvancedTextureOpsSupported"};
    }
    check_pixel_compare(ctx.device, ctx.argv[3], ctx.argv[4], true);
    return {OutcomeKind::pass, nullptr};
#else
    (void)ctx;
    return {OutcomeKind::skip, "Comparison mip fixture requires Windows SDK 10.0.26100 headers"};
#endif
}

OperationResult run_check_pixel_compare(const OperationContext& ctx)
{
    check_pixel_compare(ctx.device, ctx.argv[3], ctx.argv[4]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel_gather(const OperationContext& ctx)
{
    if (ctx.argc == 6)
        throw std::runtime_error("Vertex gather requires a sampler manifest line and vertex shader");
    check_pixel_gather(ctx.device, ctx.argv[3], ctx.argv[4],
                       ctx.argc == 7 ? ctx.argv[5] : nullptr, ctx.argc == 7 ? ctx.argv[6] : nullptr);
    return {OutcomeKind::pass, nullptr};
}

// A comparison gather needs a view format that advertises SHADER_GATHER_COMPARISON. R32_FLOAT does
// on both WARP and hardware; R32G32B32A32_FLOAT does not, so the ordinary gather case cannot be reused.
OperationResult run_check_pixel_compare_gather(const OperationContext& ctx)
{
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{
        DXGI_FORMAT_R32_FLOAT, D3D12_FORMAT_SUPPORT1_NONE, D3D12_FORMAT_SUPPORT2_NONE};
    if (FAILED(ctx.device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
        !(support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_GATHER_COMPARISON))
        return {OutcomeKind::skip, "R32_FLOAT comparison gather support unavailable"};
    check_pixel_compare_gather(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5],
                               ctx.argc == 7 ? ctx.argv[6] : nullptr);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel_case(const OperationContext& ctx)
{
    check_translated_pixel(ctx.device, ctx.argv[3], false, ctx.argv[4], D3D12_COLOR_WRITE_ENABLE_ALL,
                           ctx.argc == 6 ? ctx.argv[5] : nullptr);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel_case_rg(const OperationContext& ctx)
{
    check_translated_pixel(ctx.device, ctx.argv[3], false, ctx.argv[4],
                           D3D12_COLOR_WRITE_ENABLE_RED | D3D12_COLOR_WRITE_ENABLE_GREEN);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_compute_to_pixel(const OperationContext& ctx)
{
    check_compute_to_pixel(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], ctx.argc == 7 ? ctx.argv[6] : nullptr);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel_attribute(const OperationContext& ctx)
{
    check_attribute_pixel(ctx.device, ctx.argv[3], ctx.argv[4], false);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel_attribute_rg(const OperationContext& ctx)
{
    check_attribute_pixel(ctx.device, ctx.argv[3], ctx.argv[4], true);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel_quad_lod(const OperationContext& ctx)
{
    check_pixel_quad_lod(ctx.device, ctx.argv[3], ctx.argc == 5 ? ctx.argv[4] : nullptr);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel_get_lod_upper(const OperationContext& ctx)
{
    check_pixel_quad_lod(ctx.device, ctx.argv[3], ctx.argv[4], true);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel_mip_clamp(const OperationContext& ctx)
{
    const std::string_view level = ctx.argv[4];
    if (level.size() != 1 || level[0] < '0' || level[0] > '3')
        throw std::runtime_error("Expected a mip level from 0 to 3");
    check_pixel_mip_clamp(ctx.device, ctx.argv[3], unsigned(level[0] - '0'));
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel_mip_blend(const OperationContext& ctx)
{
    const auto number = [](std::string_view text) {
        float value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc{} || end != text.data() + text.size())
            throw std::runtime_error("Expected a decimal clamp and tolerance");
        return value;
    };
    check_pixel_mip_blend(ctx.device, ctx.argv[3], number(ctx.argv[4]), number(ctx.argv[5]));
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_check_pixel(const OperationContext& ctx)
{
    check_translated_pixel(ctx.device, ctx.argv[3], std::string_view(ctx.argv[4]) == "bgra");
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_depth_mask(const OperationContext& ctx)
{
    check_depth_mask_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], false);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_depth_mask_swapped(const OperationContext& ctx)
{
    check_depth_mask_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], true);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_depth_mask_color(const OperationContext& ctx)
{
    check_depth_mask_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], false, true);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_mesh_entry(const OperationContext& ctx)
{
    auto binding = MeshConstantBinding::root;
    if (ctx.argc == 7)
        binding = ctx.argv[6][0] == 'b' ? MeshConstantBinding::swapped_root_pair : MeshConstantBinding::root_pair;
    return skip_if_false(check_mesh_entry_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], false, binding));
}

OperationResult run_mesh_amplified(const OperationContext& ctx)
{
    return skip_if_false(check_mesh_amplified_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5],
                                                   ctx.argv[6]));
}

OperationResult run_mesh_shader_pipeline(const OperationContext& ctx)
{
    return skip_if_false(check_mesh_shader_pipeline(ctx.device, false));
}

OperationResult run_mesh_shader_pipeline_fallback(const OperationContext& ctx)
{
    return skip_if_false(check_mesh_shader_pipeline(ctx.device, true));
}

OperationResult run_mesh_heap_sampler(const OperationContext& ctx)
{
    return skip_if_false(check_mesh_heap_sampler_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5]));
}

OperationResult run_mesh_multi(const OperationContext& ctx)
{
    return skip_if_false(check_mesh_entry_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], false,
                                               MeshConstantBinding::root, MeshDrawMode::multi_primitive));
}

OperationResult run_mesh_indexed(const OperationContext& ctx)
{
    return skip_if_false(check_mesh_entry_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], true));
}

OperationResult run_mesh_indexed_fallback(const OperationContext& ctx)
{
    check_mesh_fallback_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], ctx.argv[6]);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_mesh_textured_fallback(const OperationContext& ctx)
{
    check_mesh_fallback_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], ctx.argv[6], MeshDrawMode::textured);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_mesh_entry_table(const OperationContext& ctx)
{
    const auto binding = ctx.argc == 7 ? MeshConstantBinding::swapped_table : MeshConstantBinding::table;
    return skip_if_false(check_mesh_entry_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], false, binding));
}

OperationResult run_mesh_textured(const OperationContext& ctx)
{
    return skip_if_false(check_mesh_entry_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5],
                                               true, MeshConstantBinding::root, MeshDrawMode::textured));
}

OperationResult run_mesh_texture_load(const OperationContext& ctx)
{
    return skip_if_false(check_mesh_entry_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5],
                                               true, MeshConstantBinding::root, MeshDrawMode::mesh_texture_load));
}

OperationResult run_compute_mesh_texture(const OperationContext& ctx)
{
    const MeshTextureProducers producers{ctx.argv[6], ctx.argc >= 8 ? ctx.argv[7] : nullptr, ctx.argc == 9};
    return skip_if_false(check_mesh_entry_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5],
                                               true, MeshConstantBinding::root, MeshDrawMode::textured, &producers));
}

OperationResult run_compute_mesh_texture_fallback(const OperationContext& ctx)
{
    const MeshTextureProducers producers{ctx.argv[7], ctx.argc >= 9 ? ctx.argv[8] : nullptr, ctx.argc == 10};
    check_mesh_fallback_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], ctx.argv[6],
                             MeshDrawMode::textured, &producers);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_mesh_depth_pair(const OperationContext& ctx)
{
    return skip_if_false(check_mesh_entry_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5],
                                               true, MeshConstantBinding::root, MeshDrawMode::depth_pair));
}

OperationResult run_mesh_depth_pair_fallback(const OperationContext& ctx)
{
    check_mesh_fallback_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], ctx.argv[6], MeshDrawMode::depth_pair);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_mesh_shadow(const OperationContext& ctx)
{
    return skip_if_false(check_mesh_shadow_draw(ctx.device, ctx.argv[3], ctx.argv[4], ctx.argv[5], ctx.argv[6],
                                                ctx.argc == 8));
}

bool validate_mesh_shadow_args(int argc, char** argv)
{
    return argc == 7 || std::string_view(argv[7]) == "stale";
}

OperationResult run_bvh_root_roundtrip(const OperationContext& ctx)
{
    check_bvh_root_roundtrip(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_bvh_differential(const OperationContext& ctx)
{
    return skip_if_false(check_bvh_differential(ctx.device, false));
}

OperationResult run_bvh_differential_pipeline(const OperationContext& ctx)
{
    return skip_if_false(check_bvh_differential(ctx.device, true));
}

OperationResult run_bvh_hits(const OperationContext& ctx)
{
    check_bvh_hits(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_bvh_packed(const OperationContext& ctx)
{
    check_bvh_packed(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_bvh_numeric(const OperationContext& ctx)
{
    check_bvh_numeric(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_bvh_atomics(const OperationContext& ctx)
{
    check_bvh_atomics(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_bvh_reads(const OperationContext& ctx)
{
    check_bvh_reads(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_bvh_waves(const OperationContext& ctx)
{
    return skip_if_false(check_bvh_waves(ctx.device, false));
}

OperationResult run_bvh_wave64(const OperationContext& ctx)
{
    return skip_if_false(check_bvh_waves(ctx.device, true));
}

OperationResult run_dxr_update(const OperationContext& ctx)
{
    return skip_if_false(check_dxr_update(ctx.device, false));
}

OperationResult run_dxr_update_pipeline(const OperationContext& ctx)
{
    return skip_if_false(check_dxr_update(ctx.device, true));
}

OperationResult run_dxr_facing(const OperationContext& ctx)
{
    return skip_if_false(check_dxr_facing(ctx.device, false));
}

OperationResult run_dxr_facing_pipeline(const OperationContext& ctx)
{
    return skip_if_false(check_dxr_facing(ctx.device, true));
}

std::optional<float> parse_dxr_opacity_tmin(int argc, char** argv)
{
    if (argc == 3) return dxr_opacity::defaultTMin;
    if (argc != 5 || std::string_view(argv[3]) != "--tmin") return std::nullopt;

    const std::string_view text = argv[4];
    const char* const last = text.data() + text.size();
    // Check the interval before rounding to the float stored in the ray buffer.
    double value = 0;
    const auto [end, error] = std::from_chars(text.data(), last, value);
    if (error != std::errc{} || end != last || !std::isfinite(value) || value < 0 || value > dxr_opacity::maxTMin)
        return std::nullopt;

    const float nearT = static_cast<float>(value);
    if (nearT == 0) return value == 0 ? std::optional<float>{0.0f} : std::nullopt;
    return nearT;
}

bool validate_dxr_opacity_args(int argc, char** argv)
{
    return parse_dxr_opacity_tmin(argc, argv).has_value();
}

OperationResult run_dxr_opacity_fixture(const OperationContext& ctx, bool use_pipeline)
{
    const float nearT = parse_dxr_opacity_tmin(ctx.argc, ctx.argv).value();
    return skip_if_false(check_dxr_opacity(ctx.device, ctx.mode == "warp", use_pipeline, nearT));
}

OperationResult run_dxr_opacity(const OperationContext& ctx)
{
    return run_dxr_opacity_fixture(ctx, false);
}

OperationResult run_dxr_opacity_pipeline(const OperationContext& ctx)
{
    return run_dxr_opacity_fixture(ctx, true);
}

OperationResult run_ray_query(const OperationContext& ctx)
{
    return skip_if_false(check_dxr_rays(ctx.device, false));
}

OperationResult run_dxr_address_ranges(const OperationContext& ctx)
{
    return skip_if_false(check_dxr_address_ranges(ctx.device));
}
OperationResult run_dxr_placed_build(const OperationContext& ctx)
{
    return skip_if_false(check_dxr_placed_build(ctx.device));
}
OperationResult run_dxr_slot70_as(const OperationContext& ctx)
{
    return skip_if_false(check_dxr_slot70_as(ctx.device));
}

OperationResult run_dxr_pipeline(const OperationContext& ctx)
{
    return skip_if_false(check_dxr_rays(ctx.device, true));
}

OperationResult run_gpu_alias(const OperationContext& ctx)
{
    return skip_if_false(check_gpu_alias(ctx.device));
}

OperationResult run_gpu_alias_window(const OperationContext& ctx)
{
    return skip_if_false(check_gpu_alias_window(ctx.device));
}

OperationResult run_gpu_alias_lookup(const OperationContext& ctx)
{
    return skip_if_false(check_gpu_alias_lookup(ctx.device));
}

OperationResult run_gpu_alias_lookup_writes(const OperationContext& ctx)
{
    return skip_if_false(check_gpu_alias_lookup_writes(ctx.device));
}

OperationResult run_gpu_alias_atomic_waterfall(const OperationContext& ctx)
{
    if (!check_gpu_alias_atomic_waterfall(ctx.device))
        return {OutcomeKind::fail, "Waterfall atomic counts differ from the reference"};
    return {OutcomeKind::pass, nullptr};
}
OperationResult run_gpu_alias_registry(const OperationContext& ctx)
{
    return skip_if_false(check_gpu_alias_registry(ctx.device));
}

OperationResult run_buffer_ops(const OperationContext& ctx)
{
    check_buffer_ops(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_wave_shuffle(const OperationContext& ctx)
{
    return skip_if_false(
        check_wave_shuffle(ctx.device),
        "skip: wave shuffle fixture requires SM 6.6 and a supported forced wave width");
}

OperationResult run_shader_wave(const OperationContext& ctx)
{
    return skip_if_false(check_shader_wave(ctx.device, 0),
                         "skip: shader wave fixture requires SM 6.0 and WaveOps");
}

OperationResult run_shader_wave32(const OperationContext& ctx)
{
    return skip_if_false(check_shader_wave(ctx.device, 32),
                         "skip: shader wave32 fixture requires SM 6.6, WaveOps, and a 32-lane wave");
}

OperationResult run_shader_wave64(const OperationContext& ctx)
{
    return skip_if_false(check_shader_wave(ctx.device, 64),
                         "skip: shader wave64 fixture requires SM 6.6, WaveOps, and a 64-lane wave");
}

template<bool Swapped>
OperationResult run_descriptor_heap_paging(const OperationContext& ctx)
{
    check_descriptor_heap_paging(ctx.device, Swapped);
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_wave64_emulate(const OperationContext& ctx)
{
    check_wave64_emulate(ctx.device);
    return {OutcomeKind::pass, nullptr};
}

// --startup-cb-blend: decoded colour-blend registers match the model in every draw case.
// --startup-cb-blend-swapped: with source and destination factors exchanged, at least one differs.
template<bool Swapped>
OperationResult run_startup_cb_blend(const OperationContext& ctx)
{
    const int mismatches = check_startup_cb_blend(ctx.device, Swapped);
    if (Swapped ? mismatches == 0 : mismatches != 0)
        return {OutcomeKind::fail, Swapped ? "swapped factors still matched the model" : "blend differs from the register model"};
    return {OutcomeKind::pass, nullptr};
}

// --startup-export-write-mask: a manifest 'wmask' line keeps the unwritten channels of the target (#1674).
// --startup-export-write-mask-unmasked: without the mask every channel is written, so every case differs.
template<bool Unmasked>
OperationResult run_startup_export_write_mask(const OperationContext& ctx)
{
    const int mismatches = check_startup_export_write_mask(ctx.device, Unmasked);
    if (Unmasked ? mismatches == 0 : mismatches != 0)
        return {OutcomeKind::fail, Unmasked ? "unmasked draws still kept the unwritten channels" : "a channel outside the export mask was written"};
    return {OutcomeKind::pass, nullptr};
}

// --startup-composite-provenance: source, destination before and destination after a composite draw on one
// list distinguish a drawn centre, a clipped centre and a UV that selects the source corner (#1628).
OperationResult run_startup_composite_provenance(const OperationContext& ctx)
{
    if (check_startup_composite_provenance(ctx.device) != 0)
        return {OutcomeKind::fail, "composite provenance differs from the model or a mutant matched"};
    return {OutcomeKind::pass, nullptr};
}

// --startup-range-provenance: a draw over a bound buffer window reads the colour its producer wrote before the
// fence that separated them, and a neighbouring, stale or unwritten window reads a different or zero colour
// (#870).
OperationResult run_startup_range_provenance(const OperationContext& ctx)
{
    if (check_startup_range_provenance(ctx.device) != 0)
        return {OutcomeKind::fail, "range provenance differs from the model or a mutant matched"};
    return {OutcomeKind::pass, nullptr};
}

template<bool Swapped>
OperationResult run_startup_sampler_x(const OperationContext& ctx)
{
    return skip_if_false(check_startup_sampler_x(ctx.device, Swapped),
                         "skip: R32_FLOAT does not report filtered sampling with mips");
}

// --startup-sampler-lod[-dropped]: sampler heap entries that differ only in word 1 LOD limits, decoded by
// the startup stand-in (tests/startup_sampler_lod.cpp). The dropped control must read the wrong mips.
template<bool DropClamp>
OperationResult run_startup_sampler_lod(const OperationContext& ctx)
{
    const int result = check_startup_sampler_lod(ctx.device, DropClamp);
    if (result == 77) return {OutcomeKind::skip, "skip: R32_FLOAT does not report sampling, comparison, mips and render target"};
    if (result) return {OutcomeKind::fail, DropClamp ? "dropped LOD clamp did not read the expected wrong mips"
                                                     : "sampler LOD readback differs"};
    return {OutcomeKind::pass, nullptr};
}

// --startup-sampler-bias <ps.cso> <sampler manifest line> <expected red>
OperationResult run_startup_sampler_bias(const OperationContext& ctx)
{
    const auto ps = read_shader(ctx.argv[3]);
    const int result = check_startup_sampler_lod(ctx.device, false, {ps.data(), ps.size()}, ctx.argv[4],
                                                std::stof(ctx.argv[5]));
    if (result == 77) return {OutcomeKind::skip, "skip: R32_FLOAT does not report sampling, mips and render target"};
    if (result) return {OutcomeKind::fail, "sampler bias readback differs"};
    return {OutcomeKind::pass, nullptr};
}

// --startup-sampler-heap <vs.cso> <ps.cso> <byte offset> <expected red bits>: draws one pixel with a
// gfx2hlsl pixel shader that indexes its sampler table at run time (tests/startup_sampler_heap.cpp).
OperationResult run_startup_sampler_heap(const OperationContext& ctx)
{
    const auto offset = static_cast<std::uint32_t>(std::stoul(ctx.argv[5], nullptr, 0));
    const auto expected = static_cast<std::uint32_t>(std::stoul(ctx.argv[6], nullptr, 16));
    const float red = check_startup_sampler_heap(ctx.device, ctx.argv[3], ctx.argv[4], offset,
        ctx.argc > 7 ? ctx.argv[7] : nullptr, ctx.argc > 8 ? ctx.argv[8] : nullptr);
    std::uint32_t bits;
    std::memcpy(&bits, &red, sizeof(bits));
    if (bits != expected) {
        std::fprintf(stderr, "startup_sampler_heap mismatch offset=%u expected=%08x actual=%08x\n", offset, expected, bits);
        return {OutcomeKind::fail, "sampler heap readback differs"};
    }
    return {OutcomeKind::pass, nullptr};
}

// --startup-image-3d <vs.cso> <ps.cso> <z> <expected rgba bits> [sampler manifest line]
// The expected value is four comma-separated hex words. Omit the sampler line for point wrap.
// --startup-image-2d-array takes the same arguments: the texels are two slices of a 2D array and the
// third coordinate is the slice. --startup-image-2d-array-mips uses a 4x4 array of two slices with three
// mip levels whose cells all differ (tests/startup_image_3d.cpp).
OperationResult run_startup_image_3d(const OperationContext& ctx)
{
    const bool array_mips = std::strcmp(ctx.argv[2], "--startup-image-2d-array-mips") == 0;
    const bool one_dimensional_mips = std::strcmp(ctx.argv[2], "--startup-image-1d-mips") == 0;
    const bool array = array_mips || std::strcmp(ctx.argv[2], "--startup-image-2d-array") == 0;
    const bool one_dimensional = one_dimensional_mips || std::strcmp(ctx.argv[2], "--startup-image-1d") == 0;
    const bool two_dimensional = std::strcmp(ctx.argv[2], "--startup-image-2d") == 0;
    float z = 0.0f;
    if (two_dimensional && ctx.argv[5][0] == '0' && (ctx.argv[5][1] == 'x' || ctx.argv[5][1] == 'X')) {
        char* end = nullptr;
        const unsigned long long raw = std::strtoull(ctx.argv[5], &end, 16);
        if (!end || *end != '\0' || raw > 0xffffffffull) return {OutcomeKind::fail, "packed 2D coordinate is not a 32-bit hex word"};
        const auto raw32 = static_cast<std::uint32_t>(raw);
        std::memcpy(&z, &raw32, sizeof(z));
    } else {
        z = std::stof(ctx.argv[5]);
    }
    std::uint32_t expected[4]{};
    if (std::sscanf(ctx.argv[6], "%8x,%8x,%8x,%8x", &expected[0], &expected[1], &expected[2], &expected[3]) != 4)
        return {OutcomeKind::fail, "expected value is not four hex words"};
    float color[4]{};
    const char* sampler = ctx.argc == 8 ? ctx.argv[7] : "smp 0 0x0 0x0 0x0 0x0";
    check_startup_image_3d(ctx.device, ctx.argv[3], ctx.argv[4], z, color, sampler, array,
                           array_mips || one_dimensional_mips ? 3 : 1, one_dimensional, two_dimensional);
    std::uint32_t bits[4];
    std::memcpy(bits, color, sizeof(bits));
    if (std::memcmp(bits, expected, sizeof(bits)) != 0) {
        std::fprintf(stderr, "startup_image mismatch z=%s expected=%08x,%08x,%08x,%08x actual=%08x,%08x,%08x,%08x\n",
                     ctx.argv[5], expected[0], expected[1], expected[2], expected[3], bits[0], bits[1], bits[2], bits[3]);
        return {OutcomeKind::fail, "image 3D readback differs"};
    }
    return {OutcomeKind::pass, nullptr};
}

// --startup-image-2d-view <vs.cso> <ps.cso> <sampler manifest line> <first/count> prints one RGBA32F pixel.
OperationResult run_startup_image_2d_view(const OperationContext& ctx)
{
    const std::string_view window = ctx.argv[6];
    if (window.size() != 3 || window[1] != '/' || window[0] < '0' || window[0] > '2' ||
        window[2] < '1' || window[2] > '3')
        return {OutcomeKind::fail, "expected a first/count mip window on a three-mip texture"};
    float color[4]{};
    check_startup_image_3d(ctx.device, ctx.argv[3], ctx.argv[4], 0, color, ctx.argv[5], false, 3, false, true,
                           UINT(window[0] - '0'), UINT(window[2] - '0'));
    std::uint32_t bits[4];
    std::memcpy(bits, color, sizeof(bits));
    std::printf("view_rgba=%08x,%08x,%08x,%08x\n", bits[0], bits[1], bits[2], bits[3]);
    return {OutcomeKind::pass, nullptr};
}

// --startup-image-load-d16 <r16_unorm|rgba32f|r16_unorm_array|rgba32f_array|rgba32f_sampled|rgba16f|rgba16f_r32f> <vs.cso> <ps.cso> <expected.bin>:
// draws an 8x2 full-screen pair to an R32G32B32A32_UINT target against a two-mip (8x2 + 4x1) 2D or two-slice
// 2D-array source texture. The RGBA16F cases use either a UINT target or a single-channel FLOAT target.
// Compares 64 little-endian words with expected.bin, with absent FLOAT channels represented as zeros.
OperationResult run_startup_image_load_d16(const OperationContext& ctx)
{
    const bool depth16 = std::strcmp(ctx.argv[3], "r16_unorm") == 0 || std::strcmp(ctx.argv[3], "r16_unorm_array") == 0;
    const bool rgba32f = std::strcmp(ctx.argv[3], "rgba32f") == 0 || std::strcmp(ctx.argv[3], "rgba32f_array") == 0;
    const bool array = std::strcmp(ctx.argv[3], "r16_unorm_array") == 0 || std::strcmp(ctx.argv[3], "rgba32f_array") == 0;
    const bool sampled = std::strcmp(ctx.argv[3], "rgba32f_sampled") == 0;
    const bool red_only = std::strcmp(ctx.argv[3], "rgba16f_r32f") == 0;
    const bool half_float = red_only || std::strcmp(ctx.argv[3], "rgba16f") == 0;
    if (!depth16 && !rgba32f && !sampled && !half_float)
        return {OutcomeKind::fail, "unknown image_load d16 source mode"};
    std::uint32_t expected[64]{};
    std::ifstream file(ctx.argv[6], std::ios::binary);
    file.read(reinterpret_cast<char*>(expected), sizeof(expected));
    if (!file || file.peek() != std::char_traits<char>::eof())
        return {OutcomeKind::fail, "expected file does not hold 64 words"};
    std::uint32_t actual[64]{};
    check_startup_image_load_d16(ctx.device, ctx.argv[4], ctx.argv[5], depth16, actual, array, sampled, half_float, red_only);
    unsigned differing = 0;
    for (unsigned i = 0; i < 16; ++i) {
        const std::uint32_t* want = &expected[i * 4];
        const std::uint32_t* got = &actual[i * 4];
        if (std::memcmp(want, got, 4 * sizeof(std::uint32_t)) == 0) continue;
        if (differing++ == 0) {
            std::fprintf(stderr,
                         "startup_image_load_d16 (%s) pixel (%u,%u) mismatch: expected %08x,%08x,%08x,%08x actual %08x,%08x,%08x,%08x\n",
                         ctx.argv[3], i % 8, i / 8, want[0], want[1], want[2], want[3], got[0], got[1], got[2], got[3]);
        }
    }
    if (differing) {
        std::fprintf(stderr, "%u of 16 pixels differ\n", differing);
        return {OutcomeKind::fail, "image_load d16 readback differs"};
    }
    std::printf("startup_image_load_d16 %s: all 16 pixels (64 words) matched\n", ctx.argv[3]);
    return {OutcomeKind::pass, nullptr};
}

// --startup-image-store <vs.cso> <ps.cso> <x,y> <expected rgba bits>: the translated pixel stage stores to
// u0 (tests/startup_image_store.cpp); texel (x, y) must hold the four hex words and every other texel -1.0.
OperationResult run_startup_image_store(const OperationContext& ctx)
{
    unsigned x = 0, y = 0;
    std::uint32_t expected[4]{};
    if (std::sscanf(ctx.argv[5], "%u,%u", &x, &y) != 2) return {OutcomeKind::fail, "texel is not x,y"};
    if (std::sscanf(ctx.argv[6], "%8x,%8x,%8x,%8x", &expected[0], &expected[1], &expected[2], &expected[3]) != 4)
        return {OutcomeKind::fail, "expected value is not four hex words"};
    if (check_startup_image_store(ctx.device, ctx.argv[3], ctx.argv[4], x, y, expected))
        return {OutcomeKind::fail, "image store readback differs"};
    return {OutcomeKind::pass, nullptr};
}

const char* check_startup_sharedexp_target(ID3D12Device* device, ID3D12InfoQueue* messages);

OperationResult run_startup_sharedexp_target(const OperationContext& ctx)
{
    if (const char* reason = check_startup_sharedexp_target(ctx.device, ctx.messages))
        return {OutcomeKind::skip, reason};
    return {OutcomeKind::pass, nullptr};
}

OperationResult run_startup_packed_position(const OperationContext& ctx)
{
    constexpr UINT width = 13, height = 7;
    std::uint32_t pixels[height * width * 4]{};
    check_startup_flat_param(ctx.device, ctx.argv[3], ctx.argv[4], pixels, width, height);
    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            const UINT expected = x | (y << 16);
            for (UINT component = 0; component < 4; ++component) {
                const UINT value = pixels[(y * width + x) * 4 + component];
                if (value != expected) {
                    std::fprintf(stderr, "position mismatch (%u,%u): expected %08x actual %08x\n", x, y, expected, value);
                    return {OutcomeKind::fail, "position readback differed"};
                }
            }
        }
    }
    std::puts("all 91 packed positions matched");
    return {OutcomeKind::pass, nullptr};
}

unsigned long pixel_extent(const char* text)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    return *end == '\0' && value >= 1 && value <= 64 ? static_cast<unsigned long>(value) : 0;
}

bool validate_pixel_words_args(int, char** argv)
{
    return pixel_extent(argv[5]) != 0 && pixel_extent(argv[6]) != 0;
}

bool validate_raster_args(int argc, char** argv)
{
    const std::string_view cull = argv[8], winding = argv[9];
    return validate_pixel_words_args(argc, argv) && (cull == "none" || cull == "front" || cull == "back") &&
           (winding == "cw" || winding == "ccw");
}

static std::vector<std::uint32_t> parse_hex_words(std::string_view text);

// --startup-descriptor-limits <vs> <ps> <expected.bin> <entry words> <bound 0|1>
OperationResult run_startup_descriptor_limits(const OperationContext& ctx)
{
    const auto entry = parse_hex_words(ctx.argv[6]);
    const std::string_view bound = ctx.argv[7];
    if (entry.empty() || entry.size() > 32 || (bound != "0" && bound != "1"))
        return {OutcomeKind::fail, "invalid descriptor limits entry or binding"};
    std::uint32_t expected[256]{}, actual[256]{};
    std::ifstream file(ctx.argv[5], std::ios::binary);
    file.read(reinterpret_cast<char*>(expected), sizeof(expected));
    if (!file || file.peek() != std::char_traits<char>::eof())
        return {OutcomeKind::fail, "expected file does not hold 64 RGBA pixels"};
    check_startup_image_load_d16(ctx.device, ctx.argv[3], ctx.argv[4], false, actual,
                                false, false, false, false, entry.data(), static_cast<UINT>(entry.size()), bound == "1");
    for (unsigned i = 0; i < 64; ++i) {
        if (std::memcmp(expected + i * 4, actual + i * 4, 4 * sizeof(std::uint32_t)) == 0)
            continue;
        std::fprintf(stderr, "descriptor limits pixel (%u,%u) mismatch: expected R=%08x actual R=%08x\n",
                     i % 8, i / 8, expected[i * 4], actual[i * 4]);
        return {OutcomeKind::fail, "descriptor limits readback differs"};
    }
    std::printf("descriptor limits: all 64 pixels matched\n");
    return {OutcomeKind::pass, nullptr};
}

// --startup-pixel-words <vs.cso> <ps.cso> <width> <height> <expected.bin>: draws a full-screen pair over a
// width x height R32G32B32A32_FLOAT target (tests/startup_flat_param.cpp; a discarded pixel keeps the clear
// value -1.0) and compares every pixel's four words with the file: width * height * 4 little-endian words,
// row by row. Reports the first differing pixel and the number of differing pixels.
// --startup-pixel-buffer-words appends comma-separated hex words for b0 and the raw t0 buffer.
OperationResult run_startup_pixel_words(const OperationContext& ctx)
{
    const unsigned long width = pixel_extent(ctx.argv[5]), height = pixel_extent(ctx.argv[6]);
    std::vector<std::uint32_t> expected(size_t(width) * height * 4);
    std::ifstream file(ctx.argv[7], std::ios::binary);
    file.read(reinterpret_cast<char*>(expected.data()), std::streamsize(expected.size() * sizeof(std::uint32_t)));
    if (!file || file.peek() != std::char_traits<char>::eof())
        return {OutcomeKind::fail, "expected file does not hold width * height * 4 words"};
    std::vector<std::uint32_t> pixels(expected.size());
    const std::string_view operation = ctx.argv[2];
    if (operation == "--startup-r16-unorm") {
        char* end = nullptr;
        const auto input = std::strtoul(ctx.argv[8], &end, 16);
        if (end == ctx.argv[8] || *end != '\0') return {OutcomeKind::fail, "invalid constant word"};
        const std::uint32_t constants[4] = {0, static_cast<std::uint32_t>(input), 0, 0};
        check_startup_flat_param(ctx.device, ctx.argv[3], ctx.argv[4], pixels.data(), UINT(width), UINT(height),
                                D3D12_CULL_MODE_NONE, FALSE, 3, constants, 4, {}, 1, DXGI_FORMAT_R16_UNORM);
    } else if (operation == "--raster-reference") {
        const std::string_view mode = ctx.argv[8], winding = ctx.argv[9];
        const auto cull = mode == "none" ? D3D12_CULL_MODE_NONE :
                          mode == "front" ? D3D12_CULL_MODE_FRONT : D3D12_CULL_MODE_BACK;
        check_startup_flat_param(ctx.device, ctx.argv[3], ctx.argv[4], pixels.data(), UINT(width), UINT(height),
                                cull, winding == "ccw", 6);
    } else if (std::string_view(ctx.argv[2]) == "--startup-pixel-buffer-words") {
        const auto entry = parse_hex_words(ctx.argv[8]);
        const auto buffer = parse_hex_words(ctx.argv[9]);
        if (entry.empty() || entry.size() > 32 || buffer.empty())
            return {OutcomeKind::fail, "expected 1..32 entry constants and a nonempty buffer"};
        check_startup_flat_param(ctx.device, ctx.argv[3], ctx.argv[4], pixels.data(), UINT(width), UINT(height),
                                D3D12_CULL_MODE_NONE, FALSE, 3, entry.data(), UINT(entry.size()), buffer);
    } else {
        const UINT instances = operation == "--startup-instanced-pixel-words" ? UINT(width) : 1;
        check_startup_flat_param(ctx.device, ctx.argv[3], ctx.argv[4], pixels.data(), UINT(width), UINT(height),
                                D3D12_CULL_MODE_NONE, FALSE, 3, nullptr, 4, {}, instances);
    }
    unsigned long differing = 0;
    for (unsigned long y = 0; y < height; ++y) {
        for (unsigned long x = 0; x < width; ++x) {
            const std::uint32_t* want = &expected[(y * width + x) * 4];
            const std::uint32_t* got = &pixels[(y * width + x) * 4];
            if (std::memcmp(want, got, 4 * sizeof(std::uint32_t)) == 0) continue;
            if (differing++ == 0) {
                std::fprintf(stderr, "pixel mismatch (%lu,%lu): expected %08x,%08x,%08x,%08x actual %08x,%08x,%08x,%08x\n",
                             x, y, want[0], want[1], want[2], want[3], got[0], got[1], got[2], got[3]);
            }
        }
    }
    if (differing) {
        std::fprintf(stderr, "%lu of %lu pixels differ\n", differing, width * height);
        return {OutcomeKind::fail, "pixel readback differs"};
    }
    std::printf("all %lu pixels matched\n", width * height);
    return {OutcomeKind::pass, nullptr};
}

// Fixed 8x8 RGBA8 fixture. A "-" pixel path records the clear without issuing a draw.
OperationResult run_startup_placeholder_pixels(const OperationContext& ctx)
{
    static constexpr UINT extent = 8, pixel_count = extent * extent;
    std::uint32_t colors[pixel_count * 4]{}, integers[pixel_count]{};
    const bool mixed = std::string_view(ctx.argv[5]) == "mixed";
    if (!mixed && std::string_view(ctx.argv[5]) != "color")
        return {OutcomeKind::fail, "expected color or mixed render targets"};
    const char* ps = std::string_view(ctx.argv[4]) == "-" ? nullptr : ctx.argv[4];
    check_startup_flat_param(ctx.device, ctx.argv[3], ps, colors, extent, extent,
                            D3D12_CULL_MODE_NONE, FALSE, 3, nullptr, 4, {}, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, false,
                             mixed ? integers : nullptr);
    std::ofstream out(ctx.argv[6], std::ios::binary);
    out.write(reinterpret_cast<const char*>(colors), sizeof(colors));
    if (mixed) out.write(reinterpret_cast<const char*>(integers), sizeof(integers));
    out.close();
    return out ? OperationResult{OutcomeKind::pass, nullptr} : OperationResult{OutcomeKind::fail, "write placeholder target readback"};
}

// --startup-pixel-dump <vs.cso> <ps.cso> <width> <height> <out.bin>: the draw of --startup-pixel-words, but the
// readback is written to the file (width * height * 4 little-endian words, row by row) for the caller to check.
// Gated on wave operations: the caller's pixel stage reads its own lane and wave (tests that need a reference
// per desktop wave, which only the readback can tell).
OperationResult run_startup_pixel_dump(const OperationContext& ctx)
{
    const unsigned long width = pixel_extent(ctx.argv[5]), height = pixel_extent(ctx.argv[6]);
    std::vector<std::uint32_t> pixels(size_t(width) * height * 4);
    check_startup_flat_param(ctx.device, ctx.argv[3], ctx.argv[4], pixels.data(), UINT(width), UINT(height));
    std::ofstream file(ctx.argv[7], std::ios::binary);
    file.write(reinterpret_cast<const char*>(pixels.data()), std::streamsize(pixels.size() * sizeof(std::uint32_t)));
    if (!file)
        return {OutcomeKind::fail, "cannot write the readback file"};
    std::printf("wrote %lu pixels\n", width * height);
    return {OutcomeKind::pass, nullptr};
}

// --startup-flat-param <vs.cso> <ps.cso> <expected words>: draws one pixel with a gfx2hlsl vertex and
// pixel stage pair (tests/startup_flat_param.cpp). The expected value is four 8-digit hex words,
// comma separated. An optional fourth argument sets the four entry constants (ud0..ud3) as four hex words;
// without it they are all 0.
OperationResult run_startup_flat_param(const OperationContext& ctx)
{
    std::uint32_t expected[4]{};
    if (std::sscanf(ctx.argv[5], "%8x,%8x,%8x,%8x", &expected[0], &expected[1], &expected[2], &expected[3]) != 4)
        return {OutcomeKind::fail, "expected value is not four hex words"};
    std::uint32_t words[4]{};
    std::uint32_t constants[4]{};
    if (ctx.argc == 7 && std::sscanf(ctx.argv[6], "%8x,%8x,%8x,%8x", &constants[0], &constants[1], &constants[2],
                                     &constants[3]) != 4)
        return {OutcomeKind::fail, "entry constants are not four hex words"};
    check_startup_flat_param(ctx.device, ctx.argv[3], ctx.argv[4], words, 1, 1,
                             D3D12_CULL_MODE_NONE, FALSE, 3, ctx.argc == 7 ? constants : nullptr);
    if (std::memcmp(words, expected, sizeof(words)) != 0) {
        std::fprintf(stderr, "startup_flat_param mismatch expected=%08x,%08x,%08x,%08x actual=%08x,%08x,%08x,%08x\n",
                     expected[0], expected[1], expected[2], expected[3], words[0], words[1], words[2], words[3]);
        return {OutcomeKind::fail, "flat parameter readback differs"};
    }
    return {OutcomeKind::pass, nullptr};
}

// Comma-separated 8-digit hex words, in order. An empty list or any other shape returns nothing.
static std::vector<std::uint32_t> parse_hex_words(std::string_view text)
{
    std::vector<std::uint32_t> words;
    while (!text.empty()) {
        std::uint32_t value = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value, 16);
        if (parsed.ec != std::errc{} || parsed.ptr == text.data()) return {};
        words.push_back(value);
        text.remove_prefix(std::size_t(parsed.ptr - text.data()));
        if (text.empty()) break;
        if (text.front() != ',') return {};
        text.remove_prefix(1);
    }
    return words;
}

// --startup-entry-words <vs.cso> <ps.cso> <entry words> <expected words>: draws one pixel with a gfx2hlsl
// vertex and pixel pair that reads its operands from the entry constant buffer (tests/startup_entry_words.cpp).
// <entry words> is 1..32 comma-separated 8-digit hex words, ud0 first; the translated vertex stage reads ud0
// as its rectangle-list switch, so a plain triangle fixture keeps it 0. <expected words> is the four words the
// target must hold, compared bit for bit.
OperationResult run_startup_entry_words(const OperationContext& ctx)
{
    const bool dynamic_heap = std::string_view(ctx.argv[2]) == "--startup-dynamic-heap";
    if (dynamic_heap) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        if (FAILED(check_feature(ctx.device, D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
            options.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3)
            return {OutcomeKind::skip, "direct heap indexing requires resource binding tier 3"};
    }
    const auto entry = parse_hex_words(ctx.argv[5]);
    const auto expected = parse_hex_words(ctx.argv[6]);
    if (entry.empty() || entry.size() > 32)
        return {OutcomeKind::fail, "entry constants are not 1..32 hex words"};
    if (expected.size() != 4)
        return {OutcomeKind::fail, "expected value is not four hex words"};
    std::uint32_t words[4]{};
    check_startup_entry_words(ctx.device, ctx.argv[3], ctx.argv[4], entry.data(), UINT(entry.size()), words, 1, 1, dynamic_heap);
    if (std::memcmp(words, expected.data(), sizeof(words)) != 0) {
        std::fprintf(stderr, "startup_entry_words mismatch expected=%08x,%08x,%08x,%08x "
                             "actual=%08x,%08x,%08x,%08x\n",
                     expected[0], expected[1], expected[2], expected[3], words[0], words[1], words[2], words[3]);
        return {OutcomeKind::fail, "entry constant readback differs"};
    }
    return {OutcomeKind::pass, nullptr};
}

constexpr CapabilityGate kNoGate{};
constexpr CapabilityGate kGateSm60BufferOps{
    D3D_SHADER_MODEL_6_0, false, false, 0,
    "Buffer operations require Shader Model 6.0",
    CapsDiagnostic::shader_model_6_0};
constexpr CapabilityGate kGateSm65{
    D3D_SHADER_MODEL_6_5, false, false, 0,
    "skip: fixture requires Shader Model 6.5",
    CapsDiagnostic::shader_model_6_5};
constexpr CapabilityGate kGateWave64{
    D3D_SHADER_MODEL_6_6, false, true, 64,
    "skip: 64-lane wave fixture requires SM 6.6, WaveOps, and 64-lane waves",
    CapsDiagnostic::wave_6_6};
constexpr CapabilityGate kGateWaveShuffle{
    D3D_SHADER_MODEL_6_5, false, true, 0,
    "skip: wave shuffle fixture requires SM 6.5 and WaveOps",
    CapsDiagnostic::wave_6_6};
constexpr CapabilityGate kGateShaderWave{
    D3D_SHADER_MODEL_6_0, false, true, 0,
    "skip: shader wave fixture requires SM 6.0 and WaveOps",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateShaderWave32{
    D3D_SHADER_MODEL_6_6, false, true, 32,
    "skip: shader wave32 fixture requires SM 6.6, WaveOps, and a 32-lane wave",
    CapsDiagnostic::wave_6_6};
constexpr CapabilityGate kGateShaderWave64{
    D3D_SHADER_MODEL_6_6, false, true, 64,
    "skip: shader wave64 fixture requires SM 6.6, WaveOps, and a 64-lane wave",
    CapsDiagnostic::wave_6_6};
constexpr CapabilityGate kGateSm60MeshFallback{
    D3D_SHADER_MODEL_6_0, false, false, 0,
    "skip: mesh fallback requires SM 6.0",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm60SamplerViews{
    D3D_SHADER_MODEL_6_0, false, false, 0,
    "skip: sampler fixture requires SM 6.0",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm60CompareSampler{
    D3D_SHADER_MODEL_6_0, false, false, 0,
    "skip: comparison-sampler fixture requires SM 6.0",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm60SamplerIndexControl{
    D3D_SHADER_MODEL_6_0, false, false, 0,
    "skip: sampler index control requires SM 6.0",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm60RawUavDump{
    D3D_SHADER_MODEL_6_0, false, false, 0,
    "skip: raw UAV dump requires SM 6.0",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm62RawUavDump{
    D3D_SHADER_MODEL_6_2, false, false, 0,
    "skip: denormal-preserving raw UAV dump requires SM 6.2",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateAddressWindow{
    D3D_SHADER_MODEL_6_0, true, false, 0,
    "skip: address window fixture requires SM 6.0 and 64-bit shader integer operations",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateComputeAtomic64{
    D3D_SHADER_MODEL_6_6, true, false, 0, "skip: 64-bit raw UAV atomics require SM 6.6 and Int64ShaderOps",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm66Ballot{
    D3D_SHADER_MODEL_6_6, false, false, 0,
    "skip: ballot fixture requires SM 6.6",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm60AtomicWaterfall{
    D3D_SHADER_MODEL_6_0, false, true, 0,
    "skip: atomic waterfall diagnostic requires SM 6.0 and WaveOps",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm60Compaction{
    D3D_SHADER_MODEL_6_0, false, true, 0,
    "skip: subgroup compaction fixture requires SM 6.0 and WaveOps",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm60MinLodSmoke{
    D3D_SHADER_MODEL_6_0, false, false, 0,
    "skip: MinLod smoke fixture requires SM 6.0",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm60MinLodLevels{
    D3D_SHADER_MODEL_6_0, false, false, 0,
    "skip: MinLod exact mip and blend fixture requires SM 6.0",
    CapsDiagnostic::none};
constexpr CapabilityGate kGateSm66DynamicHeap{
    D3D_SHADER_MODEL_6_6, false, false, 0,
    "direct heap indexing requires Shader Model 6.6", CapsDiagnostic::none};
constexpr CapabilityGate kGateSm66MinLod{
    D3D_SHADER_MODEL_6_6, false, false, 0,
    "skip: MinLod fixture requires SM 6.6",
    CapsDiagnostic::none};

// Static argument shapes and unconditional adapter prerequisites for every named operation.
// Scenario-dependent capability checks (such as DXR tier, mesh tier, or existing-heap support)
// remain inside the handlers and report OutcomeKind::skip.
OperationResult run_serve(const OperationContext& ctx);

const OperationSpec kOperations[] = {
    {"--serve", 3, 3, nullptr, kNoGate, run_serve},
    {"--check-pixel-batch", 5, kUnboundedArgc, nullptr, kNoGate, run_check_batch},
    {"--check-pixel-pairs", 5, kUnboundedArgc, validate_pixel_pairs_args, kNoGate, run_check_batch},
    {"--pixel-batch-debug-control", 7, 7, nullptr, kNoGate, run_pixel_batch_debug_control},
    {"--pixel-batch-removal-control", 7, 7, nullptr, kNoGate, run_pixel_batch_removal_control},
    {"--compute-root", 3, 3, nullptr, kNoGate, run_compute_root},
    {"--compute-tables", 3, 3, nullptr, kNoGate, run_compute_tables},
    {"--compute-root-file", 4, 4, nullptr, kNoGate, run_compute_root_file},
    {"--ffx-immediate-conversion", 4, 4, nullptr, kNoGate, run_ffx_immediate_conversion},
    {"--compute-store-case", 5, 5, nullptr, kNoGate, run_compute_store_case},
    {"--compute-store-batch", 5, kUnboundedArgc, nullptr, kNoGate, run_check_batch},
    {"--compute-store-sized-case", 5, 6, nullptr, kNoGate, run_compute_store_sized_case},
    {"--compute-store-dpp-case", 5, 5, nullptr, kGateShaderWave, run_compute_store_dpp_case},
    {"--compute-store-case-atomic64", 5, 5, nullptr, kGateComputeAtomic64, run_compute_store_case},
    {"--compute-texture-case", 5, 6, nullptr, kNoGate, run_compute_texture_case},
    {"--compute-texture-msaa-case", 7, 7, nullptr, kNoGate, run_compute_texture_msaa_case},
    {"--compute-texture-store-case", 5, 5, nullptr, kNoGate, run_compute_texture_store_case},
    {"--spirv-sampler-views", 4, 4, nullptr, kGateSm60SamplerViews, run_spirv_sampler_views},
    {"--dynamic-sampler-index", 3, 3, nullptr, kGateSm60SamplerViews, run_dynamic_sampler_index},
    // Two builds that differ only in the sampler descriptor index source (#1115): the
    // SV_DispatchThreadID build is asserted, the SV_GroupID build is recorded per swap.
    {"--sampler-index-control", 3, 3, nullptr, kGateSm60SamplerIndexControl, run_sampler_index_control},
    {"--startup-sampler-heap", 7, 9, nullptr, kNoGate, run_startup_sampler_heap},
    {"--startup-image-1d", 7, 8, nullptr, kNoGate, run_startup_image_3d},
    {"--startup-image-1d-mips", 7, 8, nullptr, kNoGate, run_startup_image_3d},
    {"--startup-image-2d", 7, 8, nullptr, kNoGate, run_startup_image_3d},
    {"--startup-image-2d-view", 7, 7, nullptr, kNoGate, run_startup_image_2d_view},
    {"--startup-image-3d", 7, 8, nullptr, kNoGate, run_startup_image_3d},
    {"--startup-image-store", 7, 7, nullptr, kNoGate, run_startup_image_store},
    {"--startup-image-2d-array", 7, 8, nullptr, kNoGate, run_startup_image_3d},
    {"--startup-image-2d-array-mips", 7, 8, nullptr, kNoGate, run_startup_image_3d},
    {"--startup-descriptor-limits", 8, 8, nullptr, kNoGate, run_startup_descriptor_limits},
    {"--startup-image-load-d16", 7, 7, nullptr, kNoGate, run_startup_image_load_d16},
    {"--startup-packed-position", 5, 5, nullptr, kNoGate, run_startup_packed_position},
    {"--startup-sharedexp-target", 3, 3, nullptr, kNoGate, run_startup_sharedexp_target},
    {"--startup-flat-param", 6, 7, nullptr, kNoGate, run_startup_flat_param},
    {"--startup-entry-words", 7, 7, nullptr, kNoGate, run_startup_entry_words},
    {"--startup-dynamic-heap", 7, 7, nullptr, kGateSm66DynamicHeap, run_startup_entry_words},
    {"--startup-placeholder-pixels", 7, 7, nullptr, kNoGate, run_startup_placeholder_pixels},
    {"--startup-pixel-words", 8, 8, validate_pixel_words_args, kNoGate, run_startup_pixel_words},
    {"--startup-pixel-buffer-words", 10, 10, validate_pixel_words_args, kNoGate, run_startup_pixel_words},
    {"--startup-instanced-pixel-words", 8, 8, validate_pixel_words_args, kNoGate, run_startup_pixel_words},
    {"--startup-r16-unorm", 9, 9, validate_pixel_words_args, kNoGate, run_startup_pixel_words},
    {"--raster-reference", 10, 10, validate_raster_args, kNoGate, run_startup_pixel_words},
    {"--startup-pixel-dump", 8, 8, validate_pixel_words_args, kGateShaderWave, run_startup_pixel_dump},
    // <cso> <groups 1|2> <UAV bytes> <out.bin>: whole raw UAV after one dispatch (#775 GVN parity).
    {"--raw-uav-dump", 7, 7, nullptr, kGateSm60RawUavDump, run_raw_uav_dump},
    {"--raw-uav-dump-sm62", 7, 7, nullptr, kGateSm62RawUavDump, run_raw_uav_dump},
    {"--spirv-address-window", 4, 4, nullptr, kGateAddressWindow, run_spirv_address_window},
    {"--spirv-address-window-atomic", 4, 4, nullptr, kGateAddressWindow, run_spirv_address_window_atomic},
    {"--spirv-address-window-signed", 4, 4, nullptr, kGateAddressWindow, run_spirv_address_window_signed},
    {"--spirv-address-window-float", 4, 4, nullptr, kGateAddressWindow, run_spirv_address_window_float},
    {"--spirv-address-window-vec4", 4, 4, nullptr, kGateAddressWindow, run_spirv_address_window_vec4},
    {"--spirv-address-window-vec4-narrow", 4, 4, nullptr, kGateAddressWindow, run_spirv_address_window_vec4_narrow},
    {"--spirv-address-window-qword", 4, 4, nullptr, kGateAddressWindow, run_spirv_address_window_qword},
    {"--spirv-address-window-qword-narrow", 4, 4, nullptr, kGateAddressWindow, run_spirv_address_window_qword_narrow},
    {"--spirv-compare-sampler", 6, 7, nullptr, kGateSm60CompareSampler, run_spirv_compare_sampler},
    // The SPIR-V ballot fixture compiles cs_6_6; it uses the compute-store readback behind an SM 6.6 gate.
    {"--spirv-ballot", 5, 5, nullptr, kGateSm66Ballot, run_compute_store_case},
    {"--spirv-compaction", 4, 4, nullptr, kGateSm60Compaction, run_spirv_compaction},
    {"--ffx-device-init", 3, 3, nullptr, kNoGate, run_ffx_device_init},
    {"--ffx-provider-loop", 3, 3, nullptr, kNoGate, run_ffx_provider_loop},
    {"--ffx-recurrent", 3, 3, nullptr, kNoGate, run_ffx_recurrent},
    {"--dual-uav", 3, 3, nullptr, kNoGate, run_dual_uav},
    {"--stream-root-srv", 3, 3, nullptr, kGateSm60BufferOps, run_stream_root_srv},
    {"--descriptor-window-submissions", 3, 3, nullptr, kGateSm60BufferOps, run_descriptor_window_submissions},
    {"--cpu-srv-refresh", 3, 3, nullptr, kGateSm60BufferOps, run_cpu_srv_refresh},
    {"--cpu-srv-stale", 3, 3, nullptr, kGateSm60BufferOps, run_cpu_srv_refresh},
    {"--descriptor-window-stale", 3, 3, nullptr, kGateSm60BufferOps, run_descriptor_window_stale},
    {"--descriptor-record-copy", 4, 4, validate_descriptor_record_args, kGateSm60BufferOps, run_descriptor_record_copy},
    {"--texture-mips", 3, 3, nullptr, kNoGate, run_texture_mips},
    {"--check-bc7-tail", 6, kUnboundedArgc, nullptr, kNoGate, run_check_bc7_tail},
    {"--check-bc7-mip1", 6, 6, nullptr, kNoGate, run_check_bc7_mip1},
    {"--check-bc7", 5, 6, nullptr, kNoGate, run_check_bc7},
    {"--dual-pixel", 5, 5, nullptr, kNoGate, run_dual_pixel},
    {"--dual-pixel-swapped", 5, 5, nullptr, kNoGate, run_dual_pixel_swapped},
    {"--sampled-coverage", 5, 5, nullptr, kNoGate, run_sampled_coverage},
    {"--check-pixel-compare", 5, 5, nullptr, kNoGate, run_check_pixel_compare},
    {"--check-pixel-compare-level", 5, 5, nullptr, kNoGate, run_check_pixel_compare_mips},
    {"--check-pixel-compare-bias", 5, 5, nullptr, kNoGate, run_check_pixel_compare_mips},
    {"--check-pixel-compare-grad", 5, 5, nullptr, kNoGate, run_check_pixel_compare_mips},
    {"--check-pixel-gather", 5, 7, nullptr, kNoGate, run_check_pixel_gather},
    {"--check-pixel-compare-gather", 6, 7, nullptr, kNoGate, run_check_pixel_compare_gather},
    {"--check-pixel-case", 5, 6, nullptr, kNoGate, run_check_pixel_case},
    {"--check-pixel-case-rg", 5, 5, nullptr, kNoGate, run_check_pixel_case_rg},
    {"--compute-to-pixel", 6, 7, nullptr, kNoGate, run_compute_to_pixel},
    {"--check-pixel-attribute", 5, 5, nullptr, kNoGate, run_check_pixel_attribute},
    {"--check-pixel-attribute-rg", 5, 5, nullptr, kNoGate, run_check_pixel_attribute_rg},
    {"--check-pixel-quad-lod", 4, 5, nullptr, kNoGate, run_check_pixel_quad_lod},
    {"--check-pixel-get-lod-upper", 5, 5, nullptr, kNoGate, run_check_pixel_get_lod_upper},
    {"--spirv-minlod-quad-lod", 4, 4, nullptr, kGateSm60MinLodSmoke, run_check_pixel_quad_lod},
    // Keep the fractional point-filter fixture on its existing gate (#809).
    {"--check-pixel-mip-clamp", 5, 5, nullptr, kGateSm66MinLod, run_check_pixel_mip_clamp},
    {"--check-pixel-mip-exact", 5, 5, nullptr, kGateSm60MinLodLevels, run_check_pixel_mip_clamp},
    {"--check-pixel-mip-blend", 6, 6, nullptr, kGateSm60MinLodLevels, run_check_pixel_mip_blend},
    {"--check-pixel", 5, 5, validate_check_pixel_args, kNoGate, run_check_pixel},
    {"--depth-mask", 6, 6, nullptr, kNoGate, run_depth_mask},
    {"--depth-mask-swapped", 6, 6, nullptr, kNoGate, run_depth_mask_swapped},
    {"--depth-mask-color", 6, 6, nullptr, kNoGate, run_depth_mask_color},
    {"--mesh-entry", 6, 7, validate_mesh_entry_args, kNoGate, run_mesh_entry},
    {"--mesh-amplified", 7, 7, nullptr, kNoGate, run_mesh_amplified},
    {"--mesh-shader-pipeline", 3, 3, nullptr, kNoGate, run_mesh_shader_pipeline},
    {"--mesh-shader-pipeline-fallback", 3, 3, nullptr, kNoGate, run_mesh_shader_pipeline_fallback},
    {"--mesh-indexed", 6, 6, nullptr, kNoGate, run_mesh_indexed},
    {"--mesh-multi", 6, 6, nullptr, kNoGate, run_mesh_multi},
    {"--mesh-heap-sampler", 6, 6, nullptr, kNoGate, run_mesh_heap_sampler},
    {"--mesh-indexed-fallback", 7, 7, nullptr, kGateSm60MeshFallback, run_mesh_indexed_fallback},
    {"--mesh-textured-fallback", 7, 7, nullptr, kGateSm60MeshFallback, run_mesh_textured_fallback},
    {"--mesh-entry-table", 6, 7, validate_mesh_entry_table_args, kNoGate, run_mesh_entry_table},
    {"--mesh-textured", 6, 6, nullptr, kNoGate, run_mesh_textured},
    {"--mesh-texture-load", 6, 6, nullptr, kNoGate, run_mesh_texture_load},
    {"--compute-mesh-texture", 7, 9, validate_compute_mesh_texture_args, kNoGate, run_compute_mesh_texture},
    {"--compute-mesh-texture-fallback", 8, 10, validate_compute_mesh_texture_fallback_args,
     kGateSm60MeshFallback, run_compute_mesh_texture_fallback},
    {"--mesh-depth-pair", 6, 6, nullptr, kNoGate, run_mesh_depth_pair},
    {"--mesh-depth-pair-fallback", 7, 7, nullptr, kGateSm60MeshFallback, run_mesh_depth_pair_fallback},
    {"--mesh-shadow", 7, 8, validate_mesh_shadow_args, kNoGate, run_mesh_shadow},
    {"--bvh-root-roundtrip", 3, 3, nullptr, kNoGate, run_bvh_root_roundtrip},
    {"--bvh-differential", 3, 3, nullptr, kNoGate, run_bvh_differential},
    {"--bvh-differential-pipeline", 3, 3, nullptr, kNoGate, run_bvh_differential_pipeline},
    {"--bvh-hits", 3, 3, nullptr, kGateSm65, run_bvh_hits},
    {"--bvh-packed", 3, 3, nullptr, kGateSm65, run_bvh_packed},
    {"--bvh-numeric", 3, 3, nullptr, kGateSm65, run_bvh_numeric},
    {"--bvh-atomics", 3, 3, nullptr, kGateSm65, run_bvh_atomics},
    {"--bvh-reads", 3, 3, nullptr, kGateSm65, run_bvh_reads},
    {"--bvh-waves", 3, 3, nullptr, kGateSm65, run_bvh_waves},
    {"--bvh-wave64", 3, 3, nullptr, kGateWave64, run_bvh_wave64},
    {"--dxr-update", 3, 3, nullptr, kNoGate, run_dxr_update},
    {"--dxr-update-pipeline", 3, 3, nullptr, kNoGate, run_dxr_update_pipeline},
    {"--dxr-facing", 3, 3, nullptr, kNoGate, run_dxr_facing},
    {"--dxr-facing-pipeline", 3, 3, nullptr, kNoGate, run_dxr_facing_pipeline},
    {"--dxr-opacity", 3, 5, validate_dxr_opacity_args, kNoGate, run_dxr_opacity},
    {"--dxr-opacity-pipeline", 3, 5, validate_dxr_opacity_args, kNoGate, run_dxr_opacity_pipeline},
    {"--ray-query", 3, 3, nullptr, kNoGate, run_ray_query},
    {"--dxr-address-ranges", 3, 3, nullptr, kNoGate, run_dxr_address_ranges},
    {"--dxr-placed-build", 3, 3, nullptr, kNoGate, run_dxr_placed_build},
    {"--dxr-slot70-as", 3, 3, nullptr, kNoGate, run_dxr_slot70_as},
    {"--dxr-pipeline", 3, 3, nullptr, kNoGate, run_dxr_pipeline},
    {"--gpu-alias", 3, 3, nullptr, kNoGate, run_gpu_alias},
    {"--gpu-alias-window", 3, 3, nullptr, kNoGate, run_gpu_alias_window},
    {"--gpu-alias-lookup", 3, 3, nullptr, kNoGate, run_gpu_alias_lookup},
    {"--gpu-alias-lookup-writes", 3, 3, nullptr, kNoGate, run_gpu_alias_lookup_writes},
    {"--gpu-alias-atomic-waterfall", 3, 3, nullptr, kGateSm60AtomicWaterfall, run_gpu_alias_atomic_waterfall},
    {"--gpu-alias-registry", 3, 3, nullptr, kNoGate, run_gpu_alias_registry},
    {"--buffer-ops", 3, 3, nullptr, kGateSm60BufferOps, run_buffer_ops},
    {"--descriptor-heap-paging", 3, 3, nullptr, kGateSm60BufferOps, run_descriptor_heap_paging<false>},
    {"--descriptor-heap-paging-swapped", 3, 3, nullptr, kGateSm60BufferOps, run_descriptor_heap_paging<true>},
    {"--startup-sampler-x", 3, 3, nullptr, kGateSm60BufferOps, run_startup_sampler_x<false>},
    {"--startup-sampler-x-swapped", 3, 3, nullptr, kGateSm60BufferOps, run_startup_sampler_x<true>},
    {"--startup-sampler-lod", 3, 3, nullptr, kNoGate, run_startup_sampler_lod<false>},
    {"--startup-sampler-lod-dropped", 3, 3, nullptr, kNoGate, run_startup_sampler_lod<true>},
    {"--startup-sampler-bias", 6, 6, nullptr, kNoGate, run_startup_sampler_bias},
    {"--startup-cb-blend", 3, 3, nullptr, kNoGate, run_startup_cb_blend<false>},
    {"--startup-composite-provenance", 3, 3, nullptr, kNoGate, run_startup_composite_provenance},
    {"--startup-range-provenance", 3, 3, nullptr, kNoGate, run_startup_range_provenance},
    {"--startup-cb-blend-swapped", 3, 3, nullptr, kNoGate, run_startup_cb_blend<true>},
    {"--startup-export-write-mask", 3, 3, nullptr, kNoGate, run_startup_export_write_mask<false>},
    {"--startup-export-write-mask-unmasked", 3, 3, nullptr, kNoGate, run_startup_export_write_mask<true>},
    {"--wave-shuffle", 3, 3, nullptr, kGateWaveShuffle, run_wave_shuffle},
    {"--wave64-emulate", 3, 3, nullptr, kNoGate, run_wave64_emulate},
    {"--shader-wave", 3, 3, nullptr, kGateShaderWave, run_shader_wave},
    {"--shader-wave32", 3, 3, nullptr, kGateShaderWave32, run_shader_wave32},
    {"--shader-wave64", 3, 3, nullptr, kGateShaderWave64, run_shader_wave64},
};

struct DiagnosticTarget {
    std::string* capture_stdout = nullptr;
    std::string* capture_stderr = nullptr;
};

void emit_line(std::string* capture, FILE* stream, const char* text)
{
    if (!text || !text[0]) return;
    if (capture) {
        capture->append(text);
        capture->push_back('\n');
        return;
    }
    std::fprintf(stream, "%s\n", text);
}

template <typename... Args>
void emit_format(std::string* capture, FILE* stream, const char* fmt, Args... args)
{
    char buffer[256];
    const int n = std::snprintf(buffer, sizeof(buffer), fmt, args...);
    if (n <= 0) return;
    if (capture) {
        capture->append(buffer);
        return;
    }
    std::fputs(buffer, stream);
}

enum class CliAction {
    usage_error,
    self_test,
    draw,
    operation,
};

struct CliResolution {
    CliAction action = CliAction::usage_error;
    std::string_view mode;
    const OperationSpec* spec = nullptr;
};

const OperationSpec* find_operation(std::string_view name)
{
    for (const auto& spec : kOperations) {
        if (spec.name == name) return &spec;
    }
    return nullptr;
}

CliResolution resolve_cli(int argc, char** argv, DiagnosticTarget diag = {})
{
    if (argc == 2 && std::string_view(argv[1]) == "--self-test")
        return {CliAction::self_test, {}, nullptr};
    if (argc < 3) return {CliAction::usage_error, {}, nullptr};
    const std::string_view mode = argv[1];
    if (mode != "warp" && mode != "hardware") return {CliAction::usage_error, {}, nullptr};

    const std::string_view operation = argv[2];
    if (operation.size() >= 2 && operation.substr(0, 2) == "--") {
        const OperationSpec* spec = find_operation(operation);
        if (!spec || argc < spec->min_argc || argc > spec->max_argc ||
            (spec->validate_args && !spec->validate_args(argc, argv))) {
            emit_format(diag.capture_stderr, stderr, "Unknown GPU operation or invalid arguments: %s\n", argv[2]);
            return {CliAction::usage_error, {}, nullptr};
        }
        return {CliAction::operation, mode, spec};
    }

    if (argc != 3 && argc != 5) return {CliAction::usage_error, {}, nullptr};
    return {CliAction::draw, mode, nullptr};
}

bool evaluate_gate(const CapabilityGate& gate, const AdapterCaps& caps, DiagnosticTarget diag = {})
{
    if (gate.diagnostic == CapsDiagnostic::shader_model_6_5) {
        emit_format(diag.capture_stdout, stdout, "shader_model_capped_at_6_5=0x%x\n",
                    static_cast<unsigned>((std::min)(caps.highest_shader_model, D3D_SHADER_MODEL_6_5)));
    }
    const bool ok =
        caps.highest_shader_model >= gate.min_shader_model &&
        (!gate.require_int64_ops || caps.int64_shader_ops) &&
        (!gate.require_wave_ops || caps.wave_ops) &&
        (gate.required_wave_lane_count == 0 ||
         (caps.wave_lane_count_min <= gate.required_wave_lane_count &&
          caps.wave_lane_count_max >= gate.required_wave_lane_count));
    if (!ok) {
        if (gate.diagnostic == CapsDiagnostic::wave_6_6) {
            emit_format(diag.capture_stdout, stdout,
                        "native_lane_range=%u..%u shader_model_capped_at_6_6=0x%x\n",
                        caps.wave_lane_count_min, caps.wave_lane_count_max,
                        static_cast<unsigned>((std::min)(caps.highest_shader_model, D3D_SHADER_MODEL_6_6)));
        }
        emit_line(diag.capture_stdout, stdout, gate.skip_reason);
        return false;
    }
    if (gate.diagnostic == CapsDiagnostic::shader_model_6_0) {
        emit_format(diag.capture_stdout, stdout, "shader_model_capped_at_6_0=0x%x\n",
                    static_cast<unsigned>((std::min)(caps.highest_shader_model, D3D_SHADER_MODEL_6_0)));
    }
    return true;
}

int run_operation_spec(const OperationSpec& spec, const OperationContext& ctx, DiagnosticTarget diag = {})
{
    if (!evaluate_gate(spec.gate, ctx.caps, diag)) return 77;
    try {
        const OperationResult result = spec.run(ctx);
        switch (result.kind) {
        case OutcomeKind::pass:
            return report_debug_errors(ctx.messages) ? 1 : 0;
        case OutcomeKind::skip:
            emit_line(diag.capture_stdout, stdout, result.detail);
            return 77;
        case OutcomeKind::fail:
            emit_line(diag.capture_stderr, stderr, result.detail);
            report_debug_errors(ctx.messages);
            return 1;
        }
    } catch (const std::exception& e) {
        emit_line(diag.capture_stderr, stderr, e.what());
        report_debug_errors(ctx.messages);
        return 1;
    }
    return 1;
}

// --serve: any other operation, case after case, on this one device. A test that checks hundreds of shader pairs starts one
// process and creates the D3D12 device once instead of once per case. After the banner it prints the marker pair for case -1 (the banner text), then
// reads one case per stdin line: the operation and its operands separated by tabs, exactly the arguments that follow the
// adapter name in a one-shot call. Each case runs through the same argument validation, capability gate, debug-layer check
// and exit-code mapping as a one-shot call (0 pass, 1 failure, 77 skip, 2 bad arguments); its stdout and stderr text is
// whatever the one-shot call prints. Both streams end the case with a marker line ("X2WEND <index>" on stderr, then
// "X2WCASE <index> <exit code>" on stdout) so the client can split the text per case. The line "--end" or closing stdin ends the loop; a removed
// device ends it with exit code 1 after answering, and the client starts a new process.
OperationResult run_serve(const OperationContext& ctx)
{
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::fprintf(stderr, "X2WEND -1\n");           // the banner printed so far; the client repeats it in front of every case, as a one-shot call prints it
    std::printf("X2WCASE -1 0\n");
    std::fflush(stdout);
    std::vector<char> buffer(65536);
    for (int index = 0; std::fgets(buffer.data(), int(buffer.size()), stdin); ++index) {
        std::string line(buffer.data());
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line == "--end") break;                 // an explicit end: not every launcher (a Wine wrapper) passes the end of stdin on
        std::vector<std::string> field{ctx.argv[0], std::string(ctx.mode)};
        for (std::size_t start = 0;;) {
            const auto tab = line.find('\t', start);
            field.push_back(line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
            if (tab == std::string::npos) break;
            start = tab + 1;
        }
        std::vector<char*> argv;
        for (auto& f : field) argv.push_back(f.data());
        int code = 2;
        const bool nested = field.size() >= 3 && field[2] != "--serve";
        if (nested) {
            const CliResolution cli = resolve_cli(int(argv.size()), argv.data());
            if (cli.action == CliAction::operation) {
                const OperationContext inner{ctx.device, ctx.messages, ctx.caps, ctx.mode, int(argv.size()), argv.data(), nullptr};
                code = run_operation_spec(*cli.spec, inner);
            }
        } else {
            std::fprintf(stderr, "--serve takes an operation and its operands, one case per line\n");
        }
        if (ctx.messages) ctx.messages->ClearStoredMessages();
        const bool removed = FAILED(ctx.device->GetDeviceRemovedReason());
        if (removed) {
            std::fprintf(stderr, "device removed during case %d\n", index);
            code = 1;
        }
        std::fflush(stdout);
        std::fprintf(stderr, "X2WEND %d\n", index);
        std::printf("X2WCASE %d %d\n", index, code);
        std::fflush(stdout);
        if (removed) return {OutcomeKind::fail, "device removed"};
    }
    return {OutcomeKind::pass, nullptr};
}

int run_self_test()
{
    auto require = [](bool condition, const char* what) {
        if (!condition) {
            std::fprintf(stderr, "offscreen_check self-test failed: %s\n", what);
            return false;
        }
        return true;
    };

    // Table sanity: unique --* names, valid arity bounds, non-null runners.
    for (std::size_t i = 0; i < std::size(kOperations); ++i) {
        const auto& op = kOperations[i];
        if (!require(op.name.size() > 2 && op.name.substr(0, 2) == "--", "operation name prefix")) return 1;
        if (!require(op.min_argc >= 3 && op.max_argc >= op.min_argc, "operation arity bounds")) return 1;
        if (!require(op.run != nullptr, "operation runner")) return 1;
        for (std::size_t j = i + 1; j < std::size(kOperations); ++j) {
            if (!require(op.name != kOperations[j].name, "duplicate operation name")) return 1;
        }
    }

    // (a) CLI validation rejects unknown --*, bad argc, bad flags, and bad positional draw argc with usage_error (exit 2).
    auto check_cli = [&](std::vector<const char*> args, CliAction expected_action,
                         std::string_view expected_err_substr = {}, std::optional<float> expected_tmin = {}) {
        std::vector<char*> mut_argv;
        std::vector<std::string> storage;
        storage.reserve(args.size());
        for (const char* s : args) storage.emplace_back(s);
        for (auto& s : storage) mut_argv.push_back(&s[0]);
        std::string out, err;
        const CliResolution res = resolve_cli(static_cast<int>(mut_argv.size()), mut_argv.data(), {&out, &err});
        if (!require(res.action == expected_action, args.size() > 2 ? args[2] : "cli_action")) return false;
        if (!expected_err_substr.empty() &&
            !require(err.find(expected_err_substr) != std::string::npos, "expected stderr substring")) {
            return false;
        }
        return !expected_tmin || require(parse_dxr_opacity_tmin(static_cast<int>(mut_argv.size()), mut_argv.data()) ==
                                         expected_tmin, "effective opacity TMin");
    };

    const auto check_extent = [&](const char* width, const char* height, CliAction action) {
        return check_cli({"offscreen_check", "warp", "--startup-pixel-words", "vs", "ps", width, height, "ref"}, action);
    };
    if (!check_extent("1", "64", CliAction::operation) || !check_extent("08", "+4", CliAction::operation)) return 1;
    for (const char* bad : {"", "8x4", "4junk", "4.5", "0", "65", "-1", "999999999999999999999"}) {
        if (!check_extent(bad, "4", CliAction::usage_error) || !check_extent("8", bad, CliAction::usage_error)) return 1;
    }
    const auto check_raster = [&](const char* width, const char* cull, const char* winding, CliAction action) {
        return check_cli({"offscreen_check", "warp", "--raster-reference", "vs", "ps", width, "4", "ref",
                          cull, winding}, action);
    };
    if (!check_raster("8", "back", "ccw", CliAction::operation) ||
        !check_raster("8x4", "none", "cw", CliAction::usage_error) ||
        !check_raster("8", "both", "cw", CliAction::usage_error) ||
        !check_raster("8", "front", "left", CliAction::usage_error)) return 1;

    for (const char* op : {"--dxr-opacity", "--dxr-opacity-pipeline"}) {
        if (!check_cli({"offscreen_check", "warp", op}, CliAction::operation, {}, 0.25f) ||
            !check_cli({"offscreen_check", "warp", op, "--tmin", "0"}, CliAction::operation, {}, 0.0f) ||
            !check_cli({"offscreen_check", "warp", op, "--tmin", "0.25"}, CliAction::operation, {}, 0.25f) ||
            !check_cli({"offscreen_check", "warp", op, "--tmin", "0.5"}, CliAction::operation, {}, 0.5f)) return 1;
        if (!check_cli({"offscreen_check", "warp", op, "--tmin"}, CliAction::usage_error) ||
            !check_cli({"offscreen_check", "warp", op, "--other", "0.25"}, CliAction::usage_error) ||
            !check_cli({"offscreen_check", "warp", op, "--tmin", "0", "--tmin", "0.5"}, CliAction::usage_error))
            return 1;
        for (const char* value : {"", "-0.1", "0.51", "0.50000001", "1", "nan", "inf", "1e999", "1e-100", "0,25", "0.25x"}) {
            if (!check_cli({"offscreen_check", "warp", op, "--tmin", value}, CliAction::usage_error)) return 1;
        }
    }
    if (!check_cli({"offscreen_check", "warp", "--dual-uav", "--tmin", "0.25"}, CliAction::usage_error)) return 1;

    if (!check_cli({"offscreen_check", "warp", "--bvh-wave"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --bvh-wave")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--dual-uav", "extra"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --dual-uav")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--descriptor-record-copy", "invalid"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --descriptor-record-copy")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--descriptor-record-copy", "unknown"}, CliAction::operation)) return 1;
    if (!check_cli({"offscreen_check", "warp", "--compute-root-file"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --compute-root-file")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--spirv-sampler-views"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --spirv-sampler-views")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--sampler-index-control", "extra"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --sampler-index-control")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--spirv-compare-sampler", "a.dxil", "5"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --spirv-compare-sampler")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--spirv-compare-sampler", "a.dxil", "5", "ref.bin", "baseline", "extra"},
                   CliAction::usage_error, "Unknown GPU operation or invalid arguments: --spirv-compare-sampler"))
        return 1;
    if (!check_cli({"offscreen_check", "warp", "--spirv-ballot", "a.dxil"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --spirv-ballot")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--check-pixel-pairs", "a.cso", "a.case", "b.cso"},
                   CliAction::usage_error, "Unknown GPU operation or invalid arguments: --check-pixel-pairs"))
        return 1;
    if (!check_cli({"offscreen_check", "warp", "--check-pixel", "a.cso", "invalid"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --check-pixel")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--mesh-entry", "a", "b", "c", "invalid"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --mesh-entry")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--mesh-amplified", "as", "ms", "ps"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --mesh-amplified")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--mesh-amplified", "as", "ms", "ps", "fixture"},
                   CliAction::operation)) return 1;
    if (!check_cli({"offscreen_check", "warp", "--mesh-indexed-fallback", "cs", "vs", "ps"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --mesh-indexed-fallback")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--mesh-depth-pair-fallback", "cs", "vs", "ps"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --mesh-depth-pair-fallback")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--mesh-entry-table", "a", "b", "c", "--bad"}, CliAction::usage_error,
                   "Unknown GPU operation or invalid arguments: --mesh-entry-table")) return 1;
    if (!check_cli({"offscreen_check", "warp", "--compute-mesh-texture", "a", "b", "c", "d", "e", "--bad"},
                   CliAction::usage_error, "Unknown GPU operation or invalid arguments: --compute-mesh-texture"))
        return 1;
    if (!check_cli({"offscreen_check", "warp", "out.bmp", "vs.cso"}, CliAction::usage_error)) return 1;
    if (!check_cli({"offscreen_check", "invalid_backend", "--dual-uav"}, CliAction::usage_error)) return 1;
    if (!check_cli({"offscreen_check", "warp", "out.bmp"}, CliAction::draw)) return 1;
    if (!check_cli({"offscreen_check", "hardware", "out.bmp", "vs.cso", "ps.cso"}, CliAction::draw)) return 1;
    if (!check_cli({"offscreen_check", "warp", "--check-pixel", "a.cso", "bgra"}, CliAction::operation)) return 1;
    if (!check_cli({"offscreen_check", "warp", "--mesh-indexed-fallback", "cs", "vs", "ps", "fixture"},
                   CliAction::operation)) return 1;
    if (!check_cli({"offscreen_check", "warp", "--spirv-compare-sampler", "a.dxil", "5", "ref.bin"},
                   CliAction::operation)) return 1;
    if (!check_cli({"offscreen_check", "warp", "--spirv-ballot", "a.dxil", "case.bin"}, CliAction::operation)) return 1;
    if (!check_cli({"offscreen_check", "warp", "--spirv-compare-sampler", "a.dxil", "5", "ref.bin", "baseline"},
                   CliAction::operation)) return 1;

    // (b) Unmet CapabilityGate exits 77 with named skip reason without invoking the handler.
    {
        bool called = false;
        const OperationSpec gated{
            "--self-test-gated",
            3,
            3,
            nullptr,
            {D3D_SHADER_MODEL_6_6, true, false, 0, "skip: self-test requires SM 6.6 and Int64ShaderOps",
             CapsDiagnostic::none},
            [](const OperationContext& ctx) -> OperationResult {
                *static_cast<bool*>(ctx.user_data) = true;
                return {OutcomeKind::fail, "handler must not execute when gate fails"};
            },
        };
        const AdapterCaps low_caps{D3D_SHADER_MODEL_6_5, false, true, 32, 32};
        char prog[] = "offscreen_check";
        char mode[] = "warp";
        char op[] = "--self-test-gated";
        char* argv[]{prog, mode, op};
        OperationContext ctx{nullptr, nullptr, low_caps, "warp", 3, argv, &called};
        std::string out, err;
        const int code = run_operation_spec(gated, ctx, {&out, &err});
        if (!require(code == 77, "unmet gate must exit 77")) return 1;
        if (!require(!called, "unmet gate must not invoke handler")) return 1;
        if (!require(out.find("skip: self-test requires SM 6.6 and Int64ShaderOps") != std::string::npos,
                     "unmet gate must print named skip reason")) return 1;

        // Also verify wave-64 gate prints native_lane_range before skipping.
        const AdapterCaps wave32_caps{D3D_SHADER_MODEL_6_6, true, true, 32, 32};
        const OperationSpec wave64_gated{
            "--self-test-wave64", 3, 3, nullptr, kGateWave64,
            [](const OperationContext& c) -> OperationResult {
                *static_cast<bool*>(c.user_data) = true;
                return {OutcomeKind::pass, nullptr};
            },
        };
        OperationContext wctx{nullptr, nullptr, wave32_caps, "warp", 3, argv, &called};
        out.clear();
        err.clear();
        const int wcode = run_operation_spec(wave64_gated, wctx, {&out, &err});
        if (!require(wcode == 77, "unmet wave64 gate must exit 77")) return 1;
        if (!require(!called, "unmet wave64 gate must not invoke handler")) return 1;
        if (!require(out.find("native_lane_range=32..32 shader_model_capped_at_6_6=0x66") != std::string::npos,
                     "unmet wave64 gate must preserve native_lane_range diagnostic")) return 1;

        const OperationSpec* depth_pair = find_operation("--mesh-depth-pair-fallback");
        if (!require(depth_pair != nullptr, "--mesh-depth-pair-fallback operation")) return 1;
        const AdapterCaps sm51_caps{D3D_SHADER_MODEL_5_1};
        char depth_pair_op[] = "--mesh-depth-pair-fallback";
        char cs[] = "cs.dxil";
        char vs[] = "vs.dxil";
        char ps[] = "ps.dxil";
        char fixture[] = "depth.case";
        char* depth_pair_argv[]{prog, mode, depth_pair_op, cs, vs, ps, fixture};
        OperationContext dctx{nullptr, nullptr, sm51_caps, "warp", 7, depth_pair_argv, nullptr};
        out.clear();
        err.clear();
        if (!require(run_operation_spec(*depth_pair, dctx, {&out, &err}) == 77,
                     "--mesh-depth-pair-fallback below SM 6.0 must exit 77")) return 1;
        if (!require(out.find("skip: mesh fallback requires SM 6.0") != std::string::npos,
                     "--mesh-depth-pair-fallback must print its skip reason")) return 1;

        const OperationSpec* compare_sampler = find_operation("--spirv-compare-sampler");
        if (!require(compare_sampler != nullptr, "--spirv-compare-sampler operation")) return 1;
        const AdapterCaps sm60_caps{D3D_SHADER_MODEL_6_0};
        if (!require(evaluate_gate(compare_sampler->gate, sm60_caps), "comparison sampler accepts SM 6.0"))
            return 1;
        out.clear();
        if (!require(!evaluate_gate(compare_sampler->gate, sm51_caps, {&out, &err}) &&
                     out.find("skip: comparison-sampler fixture requires SM 6.0") != std::string::npos,
                     "comparison sampler requires SM 6.0 and prints its skip reason")) return 1;

        // The ballot fixture skips on a runtime capped at SM 6.5, as the Windows 10 in-box runtime is (#751).
        const OperationSpec* ballot = find_operation("--spirv-ballot");
        if (!require(ballot != nullptr, "--spirv-ballot operation")) return 1;
        char ballot_op[] = "--spirv-ballot";
        char dxil[] = "a.dxil";
        char case_file[] = "case.bin";
        char* ballot_argv[]{prog, mode, ballot_op, dxil, case_file};
        OperationContext bctx{nullptr, nullptr, low_caps, "warp", 5, ballot_argv, nullptr};
        out.clear();
        err.clear();
        if (!require(run_operation_spec(*ballot, bctx, {&out, &err}) == 77, "--spirv-ballot below SM 6.6 must exit 77"))
            return 1;
        if (!require(out.find("skip: ballot fixture requires SM 6.6") != std::string::npos,
                     "--spirv-ballot must print its skip reason")) return 1;

        const OperationSpec* compaction = find_operation("--spirv-compaction");
        if (!require(compaction != nullptr, "--spirv-compaction operation")) return 1;
        const AdapterCaps sm60_waves{D3D_SHADER_MODEL_6_0, false, true, 4, 4};
        const AdapterCaps sm60_no_waves{D3D_SHADER_MODEL_6_0, false, false, 0, 0};
        const AdapterCaps sm51_waves{D3D_SHADER_MODEL_5_1, false, true, 32, 32};
        if (!require(evaluate_gate(compaction->gate, sm60_waves), "compaction accepts SM 6.0 with WaveOps"))
            return 1;
        for (const AdapterCaps& unmet : {sm60_no_waves, sm51_waves}) {
            out.clear();
            if (!require(!evaluate_gate(compaction->gate, unmet, {&out, &err}) &&
                         out.find("skip: subgroup compaction fixture requires SM 6.0 and WaveOps") != std::string::npos,
                         "compaction requires both SM 6.0 and WaveOps"))
                return 1;
        }

        const OperationSpec* address_window = find_operation("--spirv-address-window");
        if (!require(address_window != nullptr, "--spirv-address-window operation")) return 1;
        const AdapterCaps sm60_int64{D3D_SHADER_MODEL_6_0, true};
        const AdapterCaps sm51_int64{D3D_SHADER_MODEL_5_1, true};
        if (!require(evaluate_gate(address_window->gate, sm60_int64), "address window accepts SM 6.0 with Int64ShaderOps"))
            return 1;
        for (const AdapterCaps& unmet : {sm60_no_waves, sm51_int64}) {
            out.clear();
            if (!require(!evaluate_gate(address_window->gate, unmet, {&out, &err}) &&
                         out.find("skip: address window fixture requires SM 6.0 and 64-bit shader integer operations") !=
                             std::string::npos,
                         "address window requires both SM 6.0 and Int64ShaderOps"))
                return 1;
        }

        // The fractional point draw keeps its SM 6.6 gate. Exact mips and blends only need SM 6.0.
        const AdapterCaps sm66_caps{D3D_SHADER_MODEL_6_6};
        char clamp_op[] = "--check-pixel-mip-clamp";
        char exact_op[] = "--check-pixel-mip-exact";
        char blend_op[] = "--check-pixel-mip-blend";
        char level[] = "1";
        char tolerance[] = "0.0078125";
        char* clamp_argv[]{prog, mode, clamp_op, dxil, level};
        char* exact_argv[]{prog, mode, exact_op, dxil, level};
        char* blend_argv[]{prog, mode, blend_op, dxil, level, tolerance};
        const struct {
            char** argv;
            int argc;
            AdapterCaps supported;
            AdapterCaps unsupported;
            const char* skip_reason;
        } minlod_cases[]{
            {clamp_argv, 5, sm66_caps, low_caps, "skip: MinLod fixture requires SM 6.6"},
            {exact_argv, 5, sm60_caps, sm51_caps, "skip: MinLod exact mip and blend fixture requires SM 6.0"},
            {blend_argv, 6, sm60_caps, sm51_caps, "skip: MinLod exact mip and blend fixture requires SM 6.0"},
        };
        for (const auto& c : minlod_cases) {
            const OperationSpec* spec = find_operation(c.argv[2]);
            if (!require(spec != nullptr, "MinLod operation")) return 1;
            if (!require(evaluate_gate(spec->gate, c.supported), "MinLod draw accepts its minimum shader model"))
                return 1;
            OperationContext mctx{nullptr, nullptr, c.unsupported, "warp", c.argc, c.argv, nullptr};
            out.clear();
            err.clear();
            if (!require(run_operation_spec(*spec, mctx, {&out, &err}) == 77, "MinLod draw below its minimum must exit 77"))
                return 1;
            if (!require(out.find(c.skip_reason) != std::string::npos, "MinLod draw must print its skip reason")) return 1;
        }

        // The MinLod smoke draw accepts SM 6.0 without wave or int64 operations.
        char quad_op[] = "--spirv-minlod-quad-lod";
        char* quad_argv[]{prog, mode, quad_op, dxil};
        const OperationSpec* quad = find_operation(quad_op);
        if (!require(quad != nullptr, "MinLod smoke operation")) return 1;
        if (!require(evaluate_gate(quad->gate, sm60_caps), "MinLod smoke draw must accept SM 6.0")) return 1;
        OperationContext qctx{nullptr, nullptr, sm51_caps, "warp", 4, quad_argv, nullptr};
        out.clear();
        err.clear();
        if (!require(run_operation_spec(*quad, qctx, {&out, &err}) == 77,
                     "MinLod smoke draw below SM 6.0 must exit 77")) return 1;
        if (!require(out.find("skip: MinLod smoke fixture requires SM 6.0") != std::string::npos,
                     "MinLod smoke draw must print its skip reason")) return 1;
    }

    // (c) Handler OutcomeKind::fail and thrown exception exit 1 (never 77); OutcomeKind::skip exits 77; pass exits 0.
    {
        const AdapterCaps caps{D3D_SHADER_MODEL_6_6, true, true, 32, 64};
        char prog[] = "offscreen_check";
        char mode[] = "warp";
        char op[] = "--self-test-outcome";
        char* argv[]{prog, mode, op};
        OperationContext ctx{nullptr, nullptr, caps, "warp", 3, argv, nullptr};

        const OperationSpec fail_spec{
            "--self-test-fail", 3, 3, nullptr, kNoGate,
            [](const OperationContext&) -> OperationResult {
                return {OutcomeKind::fail, "explicit failure control"};
            },
        };
        std::string out, err;
        if (!require(run_operation_spec(fail_spec, ctx, {&out, &err}) == 1, "OutcomeKind::fail must exit 1"))
            return 1;
        if (!require(err.find("explicit failure control") != std::string::npos, "OutcomeKind::fail detail"))
            return 1;

        const OperationSpec throw_spec{
            "--self-test-throw", 3, 3, nullptr, kNoGate,
            [](const OperationContext&) -> OperationResult {
                throw std::runtime_error("thrown exception control");
            },
        };
        out.clear();
        err.clear();
        if (!require(run_operation_spec(throw_spec, ctx, {&out, &err}) == 1, "thrown exception must exit 1"))
            return 1;
        if (!require(err.find("thrown exception control") != std::string::npos, "thrown exception detail"))
            return 1;

        const OperationSpec skip_spec{
            "--self-test-skip", 3, 3, nullptr, kNoGate,
            [](const OperationContext&) -> OperationResult {
                return {OutcomeKind::skip, "scenario skip control"};
            },
        };
        out.clear();
        err.clear();
        if (!require(run_operation_spec(skip_spec, ctx, {&out, &err}) == 77, "OutcomeKind::skip must exit 77"))
            return 1;
        if (!require(out.find("scenario skip control") != std::string::npos, "OutcomeKind::skip detail"))
            return 1;

        const OperationSpec pass_spec{
            "--self-test-pass", 3, 3, nullptr, kNoGate,
            [](const OperationContext&) -> OperationResult {
                return {OutcomeKind::pass, nullptr};
            },
        };
        out.clear();
        err.clear();
        if (!require(run_operation_spec(pass_spec, ctx, {&out, &err}) == 0, "OutcomeKind::pass must exit 0"))
            return 1;
    }

    std::puts("offscreen_self_test: cli_validation=ok gate_skip=ok outcome_fail=ok outcome_throw=ok outcome_skip=ok outcome_pass=ok");
    return 0;
}

int main(int argc, char** argv)
{
    const CliResolution cli = resolve_cli(argc, argv);
    if (cli.action == CliAction::usage_error) return 2;
    if (cli.action == CliAction::self_test) return run_self_test();

    ComPtr<ID3D12InfoQueue> messages;
    try {
        ComPtr<ID3D12Debug> debug;
#ifdef OFFSCREEN_AGILITY_620
        ComPtr<ID3D12DeviceFactory> device_factory;
        check(D3D12GetInterface(CLSID_D3D12DeviceFactory, IID_PPV_ARGS(&device_factory)), "Load device factory");
        // Without Graphics Tools (for example on a default Windows 10 install) the debug layer reports
        // DXGI_ERROR_SDK_COMPONENT_MISSING even with the SDK layers app-local. The runtime itself still
        // loads, so continue without the layer and report it as unavailable. Other failures stay fatal.
        const HRESULT debug_hr = device_factory->GetConfigurationInterface(CLSID_D3D12Debug, IID_PPV_ARGS(&debug));
        if (debug_hr != DXGI_ERROR_SDK_COMPONENT_MISSING) check(debug_hr, "Load app-local debug layer");
#else
        D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
#endif
        if (debug) debug->EnableDebugLayer();
        std::printf("backend=%s debug_layer=%s\n", argv[1], debug ? "enabled" : "unavailable");
        ComPtr<IDXGIFactory6> factory;
        check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "Create DXGI factory");
        ComPtr<ID3D12Device> device;
#ifdef OFFSCREEN_AGILITY_620
        auto adapter = select_adapter(factory.Get(), cli.mode == "warp", device, device_factory.Get());
#else
        auto adapter = select_adapter(factory.Get(), cli.mode == "warp", device);
#endif
        if (!adapter) {
            std::puts("No D3D12-capable hardware adapter visible");
            return 77;
        }
        DXGI_ADAPTER_DESC1 desc{};
        check(adapter->GetDesc1(&desc), "Read DXGI adapter");
        LARGE_INTEGER driver{};
        std::printf("adapter=%ls vendor=%04x device=%04x\n", desc.Description, desc.VendorId, desc.DeviceId);
        if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver))) {
            std::printf("driver=%u.%u.%u.%u\n", HIWORD(driver.HighPart), LOWORD(driver.HighPart),
                        HIWORD(driver.LowPart), LOWORD(driver.LowPart));
        } else {
            std::puts("driver=unavailable via DXGI");
        }
#if defined(__ID3D12Device12_INTERFACE_DEFINED__)
        D3D12_FEATURE_DATA_D3D12_OPTIONS19 options19{};
        if (SUCCEEDED(check_feature(device.Get(), D3D12_FEATURE_D3D12_OPTIONS19,
                                    &options19, sizeof(options19))) &&
            options19.MaxViewDescriptorHeapSize) {
            std::printf("max_view_descriptor_heap_size=%u\n", options19.MaxViewDescriptorHeapSize);
        }
#endif
        if (debug) check(device.As(&messages), "Read debug queue");
        const AdapterCaps caps = query_adapter_caps(device.Get());
        if (cli.action == CliAction::draw) {
            draw(device.Get(), argv[2], argc == 5 ? argv + 3 : nullptr);
            return report_debug_errors(messages.Get()) ? 1 : 0;
        }
        const OperationContext ctx{device.Get(), messages.Get(), caps, cli.mode, argc, argv, nullptr};
        return run_operation_spec(*cli.spec, ctx);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        report_debug_errors(messages.Get());
        return 1;
    }
}
