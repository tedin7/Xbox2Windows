#include "startup_stubs/native_recording.h"
#include "native_compute_pipeline.h"
#include "direct_compute_cache.h"
#include "native_graphics_bindings.h"
#include "native_texture_store_view.h"
#include "startup_stubs/native_dispatch_states.h"
#include <dxgi1_4.h>
#include <cstdio>
#include <memory>
#include <filesystem>
#include <fstream>
#include <vector>

static int graphics_pso(ID3D12Device *device, ID3D12RootSignature *root, D3D12_SHADER_BYTECODE vs,
                        D3D12_SHADER_BYTECODE gs, D3D12_SHADER_BYTECODE ps) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC p{};
    p.VS = vs;
    p.GS = gs;
    p.PS = ps;
    p.pRootSignature = root;
    p.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    p.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    p.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    p.RasterizerState.DepthClipEnable = TRUE;
    p.BlendState.RenderTarget[0] = {FALSE,
                                    FALSE,
                                    D3D12_BLEND_ONE,
                                    D3D12_BLEND_ZERO,
                                    D3D12_BLEND_OP_ADD,
                                    D3D12_BLEND_ONE,
                                    D3D12_BLEND_ZERO,
                                    D3D12_BLEND_OP_ADD,
                                    D3D12_LOGIC_OP_NOOP,
                                    D3D12_COLOR_WRITE_ENABLE_ALL};
    p.NumRenderTargets = 1;
    p.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    p.SampleDesc.Count = 1;
    p.SampleMask = UINT_MAX;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline;
    const HRESULT hr = device->CreateGraphicsPipelineState(&p, IID_PPV_ARGS(&pipeline));
    if (FAILED(hr)) {
        std::fprintf(stderr, "graphics PSO failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return 10;
    }
    return 0;
}

static std::string file_bytes(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("missing shader fixture");
    return {std::istreambuf_iterator<char>(file), {}};
}

static int dispatch_states(ID3D12Device *device) {
    using x2w_startup::NativeDispatchStates;
    const auto rg = x2w::native_half_store_format(x2w::ComputeViewKind::half2_texture_uav);
    if (!rg.accepts(DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_TYPELESS) ||
        !rg.accepts(DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_FLOAT) ||
        rg.accepts(DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_R16G16_TYPELESS) ||
        rg.accepts(DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16B16A16_TYPELESS)) return 22;
    struct View {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        bool write{};
        x2w_startup::NativeTextureWindow texture{};
        struct { bool upload = false; } buffer;
    };
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = 8;
    desc.DepthOrArraySize = 6;
    desc.MipLevels = 3;
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    Microsoft::WRL::ComPtr<ID3D12Resource> texture;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, IID_PPV_ARGS(&texture)))) return 15;
    // Read mips 0 and 1 on all six faces; write mip 2. Two duplicate SRVs
    // exercise the merge. The read-only plan does not expand the face range.
    View read{texture, false, {0,2,0,6}}, write{texture, true, {2,1,0,6}};
    NativeDispatchStates reads(std::vector<View>{read, read});
    unsigned whole = 0;
    reads.whole_states([&](auto *r, auto state) {
        if (r == texture.Get() && state == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) ++whole;
    });
    if (whole != 1 || !reads.split().empty() || !reads.writes().empty()) return 16;
    NativeDispatchStates mixed(std::vector<View>{read, write, read, write});
    if (!mixed.mixed(texture.Get()) || mixed.split().size() != 18 ||
        mixed.writes().size() != 1 || mixed.writes()[0].UAV.pResource != texture.Get()) return 17;
    for (unsigned sub = 0; sub < 18; ++sub) {
        const auto &t = mixed.split()[sub].Transition;
        if (t.Subresource != sub || t.StateBefore != D3D12_RESOURCE_STATE_COMMON ||
            t.StateAfter != (sub % 3 == 2 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)) return 18;
    }
    mixed.reverse();
    if (mixed.split()[17].Transition.StateAfter != D3D12_RESOURCE_STATE_COMMON) return 19;
    // All mips of faces 0-3 vs all mips of faces 4-5 are adjacent.
    read.texture = {0,3,0,4};
    write.texture = {0,3,4,2};
    if (NativeDispatchStates(std::vector<View>{write,read}).split().size() != 18) return 20;
    write.texture = {1,1,3,1};
    try { NativeDispatchStates invalid(std::vector<View>{write, read}); }
    catch (const std::runtime_error &e) {
        if (std::string(e.what()) == "overlapping compute SRV and UAV subresource 10") return 0;
    }
    return 21;
}

// Allocation and lifetime smoke. No command list or GPU workload is required.
int wmain(int argc, wchar_t **argv) {
    using Microsoft::WRL::ComPtr;
    using x2w_startup::NativeRecording;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> warp;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) ||
        FAILED(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        std::puts("WARP device unavailable");
        return 77;
    }
    try {
        if (argc == 2 && std::wstring_view(argv[1]) == L"--subresources")
            return dispatch_states(device.Get());
        if (argc > 1) {
            D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
            if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof model)) ||
                model.HighestShaderModel < D3D_SHADER_MODEL_6_6)
                return 77;
        }
        if (argc == 7 && std::wstring_view(argv[1]) == L"--graphics-sampler") {
            const auto a = file_bytes(argv[3]), b = file_bytes(argv[4]), guest_root = file_bytes(argv[5]);
            x2w::ShaderDxilCompiler compiler(argv[2]);
            for (bool warm : {false, true}) {
                bool hit = false;
                auto pair = x2w::prepare_direct_graphics(a, b, argv[6], hit, guest_root);
                x2w::NativeGraphicsBindings bindings(pair);
                auto statics = x2w_startup::decode_statics(bindings.samplers);
                if (hit != warm || bindings.srv_count != 2 || bindings.views.size() != 4 || statics.size() != 2 ||
                    bindings.views[2].kind != x2w::ComputeViewKind::half_scalar_uav || bindings.views[2].shader_register != 0 ||
                    bindings.views[2].byte_offset != 64 || bindings.views[3].byte_offset != 32 ||
                    bindings.views[3].shader_register != 1 ||
                    bindings.views[0].kind != x2w::ComputeViewKind::comparison_texture_srv ||
                    statics[0].Filter != D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT ||
                    statics[0].ComparisonFunc != D3D12_COMPARISON_FUNC_LESS ||
                    statics[1].Filter != D3D12_FILTER_MIN_MAG_MIP_POINT)
                    return 13;
                ComPtr<ID3D12RootSignature> root;
                if (FAILED(x2w_startup::create_translated_root(device.Get(), 1, bindings.srv_count, bindings.samplers, &root, 2)))
                    return 8;
                auto vs = compiler.compile(pair.vertex.dxil_ir, L"vs_6_6", argv[6], x2w::ShaderSourceFormat::dxil_ir);
                auto ps = compiler.compile(pair.pixel.dxil_ir, L"ps_6_6", argv[6], x2w::ShaderSourceFormat::dxil_ir);
                if (vs.cache_hit != warm || ps.cache_hit != warm) return 14;
                int result = graphics_pso(device.Get(), root.Get(), {vs.dxil.data(), vs.dxil.size()}, {},
                                          {ps.dxil.data(), ps.dxil.size()});
                if (result) return result;
            }
            std::puts("WARP created cold and cached graphics pipelines with regular/comparison samples and vertex/pixel UAVs");
            return 0;
        }
        if (argc == 5 && std::wstring_view(argv[1]) == L"--sampler") {
            D3D12_FEATURE_DATA_D3D12_OPTIONS1 options{};
            if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options, sizeof options)) ||
                !options.WaveOps || options.WaveLaneCountMin > 32 || options.WaveLaneCountMax < 32) {
                std::printf("compute requires wave32: WaveOps=%u range=%u..%u\n", options.WaveOps,
                            options.WaveLaneCountMin, options.WaveLaneCountMax);
                return 77;
            }
            x2w::ShaderDxilCompiler compiler(argv[2]);
            for (bool warm : {false, true}) {
                auto pipeline = x2w::create_native_compute_pipeline(device.Get(), compiler, argv[3], 0, argv[4], true);
                if (!pipeline.state || pipeline.frontend_cache_hit != warm || pipeline.cache_hit != warm ||
                    pipeline.srv_count != 2 || pipeline.bindings[0].kind != x2w::ComputeViewKind::comparison_texture_srv)
                    return 11;
                bool hit = false;
                const auto shader = x2w::prepare_direct_compute(argv[3], 0, true, argv[4], hit);
                if (!hit || shader.samplers.size() != 2 || !shader.samplers[0].comparison ||
                    shader.samplers[1].comparison || shader.samplers[0].words != shader.samplers[1].words)
                    return 12;
            }
            std::puts("WARP created cold and cached comparison-sampling compute pipelines");
            return 0;
        }
        if (argc == 4) {
            D3D12_ROOT_SIGNATURE_DESC empty{};
            ComPtr<ID3DBlob> blob;
            ComPtr<ID3D12RootSignature> root;
            if (FAILED(D3D12SerializeRootSignature(&empty, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr)) ||
                FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root))))
                return 8;
            const auto vs = file_bytes(argv[1]), gs = file_bytes(argv[2]), ps = file_bytes(argv[3]);
            return graphics_pso(device.Get(), root.Get(), {vs.data(), vs.size()}, {gs.data(), gs.size()}, {ps.data(), ps.size()});
        }
        if (argc != 1)
            return 1;
        auto recording = std::make_shared<NativeRecording>();
        if (recording->allocate(device.Get(), 0).heap)
            return 1;
        const auto first = recording->allocate(device.Get(), 1);
        const UINT page_size = first.heap->GetDesc().NumDescriptors;
        const auto rest = recording->allocate(device.Get(), page_size - 1);
        if (rest.heap != first.heap || !first.increment || rest.gpu.ptr != first.gpu.ptr + first.increment ||
            rest.cpu.ptr != first.cpu.ptr + first.increment)
            return 2;
        const auto next = recording->allocate(device.Get(), 1);
        if (next.heap == first.heap)
            return 3;
        recording->retain(first.heap);
        if (recording->objects.size() != 2)
            return 4;

        auto submit1 = recording, submit2 = recording;
        std::weak_ptr<NativeRecording> previous = recording;
        recording = std::make_shared<NativeRecording>();
        auto fresh = recording->allocate(device.Get(), 1);
        submit1.reset();
        if (fresh.heap == first.heap || first.heap->GetDesc().NumDescriptors != page_size || previous.expired())
            return 5;
        submit2.reset();
        if (!previous.expired())
            return 6;
        std::puts("WARP descriptor ranges, rollover, reset and repeated submission ownership passed");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 7;
    }
}
