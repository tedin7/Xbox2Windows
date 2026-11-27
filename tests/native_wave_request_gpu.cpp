#include "desktop_gpu.h"
#include "native_compute_pipeline.h"
#include <cstdio>
#include <cstring>
#include <dxgi1_4.h>

using Microsoft::WRL::ComPtr;
int check_native_compute_grid(ID3D12Device*, x2w::ShaderDxilCompiler&, const wchar_t*, const wchar_t*, const std::wstring&);
int check_half_grid(ID3D12Device*, x2w::ShaderDxilCompiler&, const wchar_t*, const wchar_t*, const std::wstring&);
int wmain(int argc, wchar_t **argv) {
    if (argc != 4 && argc != 5)
        return 2;
    try {
        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "factory");
        ComPtr<ID3D12Device> device;
        for (UINT i = 0; !device; ++i) {
            ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(i, &adapter) != S_OK)
                return 77;
            DXGI_ADAPTER_DESC1 description{};
            check(adapter->GetDesc1(&description), "adapter description");
            if (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                continue;
            ComPtr<ID3D12Device> candidate;
            if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&candidate))))
                continue;
            D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
            D3D12_FEATURE_DATA_D3D12_OPTIONS1 waves{};
            if (FAILED(check_feature(candidate.Get(), D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) ||
                model.HighestShaderModel < D3D_SHADER_MODEL_6_6 ||
                FAILED(check_feature(candidate.Get(), D3D12_FEATURE_D3D12_OPTIONS1, &waves, sizeof(waves))) ||
                !waves.WaveOps || waves.WaveLaneCountMin > 32 || waves.WaveLaneCountMax < 32)
                continue;
            device = candidate;
        }
        x2w::ShaderDxilCompiler compiler(argv[1]);
        if (argc == 5 && std::wstring(argv[4]).find(L"half-") == 0)
            return check_half_grid(device.Get(), compiler, argv[2], argv[3], argv[4]);
        if(argc==5)return check_native_compute_grid(device.Get(),compiler,argv[2],argv[3],argv[4]);
        auto pipeline = x2w::create_native_compute_pipeline(device.Get(), compiler, argv[2], 0, argv[3]);
        if (!pipeline.cache_hit || pipeline.srv_count != 1 || pipeline.bindings.size() != 2)
            return 1;
        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        ComPtr<ID3D12DescriptorHeap> heap;
        check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "descriptor heap");
        UINT step = device->GetDescriptorHandleIncrementSize(hd.Type);
        auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
        auto input = make_buffer(device.Get(), 12, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        const unsigned words[]{0, 0x10000, 0x55555555};
        void *mapped;
        check(input->Map(0, nullptr, &mapped), "map input");
        std::memcpy(mapped, words, sizeof(words));
        input->Unmap(0, nullptr);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Format = DXGI_FORMAT_R32_TYPELESS;
        srv.Buffer.NumElements = 3;
        srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        device->CreateShaderResourceView(input.Get(), &srv, cpu);
        cpu.ptr += step;
        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = td.Height = 8;
        td.DepthOrArraySize = td.MipLevels = td.SampleDesc.Count = 1;
        td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        auto texture = resource(device.Get(), td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        device->CreateUnorderedAccessView(texture.Get(), nullptr, nullptr, cpu);
        ComputeCommands commands(device.Get(), pipeline.state.Get());
        pipeline.bind(commands.list.Get(), heap.Get(), heap->GetGPUDescriptorHandleForHeapStart(), step);
        commands.list->Dispatch(1, 1, 1);
        transition(commands.list.Get(), texture.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 bytes;
        device->GetCopyableFootprints(&td, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
        auto readback = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION src{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION dst{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = footprint;
        commands.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        commands.submit_and_wait(device.Get());
        check(readback->Map(0, nullptr, &mapped), "read pixels");
        for (unsigned i = 0; i < 64; ++i) {
            auto pixel = reinterpret_cast<const float *>(static_cast<const char *>(mapped) +
                                                         (i / 8) * footprint.Footprint.RowPitch) +
                         (i % 8) * 4;
            unsigned lane = i % 32;
            if (pixel[0] != lane || pixel[1] != (lane + 1) / 2 || pixel[2] != i / 32 || pixel[3] != 2) {
                std::fprintf(stderr, "lane %u: %g %g %g %g\n", i, pixel[0], pixel[1], pixel[2], pixel[3]);
                return 1;
            }
        }
        readback->Unmap(0, nullptr);
        std::puts("64 lanes, two wave32 groups, masked prefix counts and offline-to-runtime cache match");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
