// Runs one translated compute shader that samples a TextureCube and writes the samples to a buffer.
//
// usage: startup_cube_sample_check <warp|hardware> <shader.cso> <faces.bin> <samples.bin> <out.bin>
//   faces.bin    six 4x4 RGBA32_FLOAT faces in the order +X -X +Y -Y +Z -Z (6 * 16 texels * 16 bytes)
//   samples.bin  N records of 16 bytes (s, t, face, unused), N = the shader's thread count
//   out.bin      written: N records of 16 bytes, the sampled float4 of each record
// Bindings (the translator's manifest `srv 0 buf`, `srv 1 texcube`, `uav 0`): t0 raw buffer = samples, t1 = the cube view,
// u0 raw buffer = out, b0 = two zero root constants, s0 = a static point sampler with clamped addresses. Exit 77 when
// the adapter or the runtime is missing.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr UINT face_size = 4;
constexpr UINT face_bytes = face_size * face_size * 16;

void check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        char text[96];
        std::snprintf(text, sizeof text, "%s failed: 0x%08lx", what, static_cast<unsigned long>(hr));
        throw std::runtime_error(text);
    }
}

std::vector<char> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error(std::string("cannot read ") + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

ComPtr<ID3D12Resource> make_buffer(ID3D12Device* device, D3D12_HEAP_TYPE type, UINT64 bytes, D3D12_RESOURCE_STATES state,
                                   D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    const auto desc = buffer_desc(bytes, flags);
    ComPtr<ID3D12Resource> buffer;
    check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&buffer)),
          "CreateCommittedResource(buffer)");
    return buffer;
}

void fill(ID3D12Resource* upload, const void* data, size_t bytes) {
    void* mapped = nullptr;
    check(upload->Map(0, nullptr, &mapped), "Map");
    std::memcpy(mapped, data, bytes);
    upload->Unmap(0, nullptr);
}

void barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from,
             D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &b);
}

int run(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <warp|hardware> <shader.cso> <faces.bin> <samples.bin> <out.bin>\n", argv[0]);
        return 2;
    }
    const std::string adapter_name = argv[1];
    const auto shader = read_file(argv[2]);
    const auto faces = read_file(argv[3]);
    const auto samples = read_file(argv[4]);
    if (faces.size() != 6 * face_bytes || samples.empty() || samples.size() % 16 != 0) {
        std::fprintf(stderr, "bad input sizes\n");
        return 2;
    }

    ComPtr<IDXGIFactory4> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
    ComPtr<IDXGIAdapter> adapter;
    if (adapter_name == "warp") check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
    else if (factory->EnumAdapters(0, &adapter) != S_OK) return 77;
    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) return 77;

    // Root signature: b0 two constants, t0-t1 table, u0 table, static point sampler s0.
    D3D12_DESCRIPTOR_RANGE srv_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0};
    D3D12_DESCRIPTOR_RANGE uav_range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 2};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &srv_range};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = {1, &uav_range};
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC root{3, params, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob, errors;
    check(D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, &errors), "SerializeRootSignature");
    ComPtr<ID3D12RootSignature> signature;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&signature)),
          "CreateRootSignature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = signature.Get();
    pso.CS = {shader.data(), shader.size()};
    ComPtr<ID3D12PipelineState> state;
    check(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&state)), "CreateComputePipelineState");

    // Resources. The cube is one 4x4 RGBA32F 2D array of six slices; the view makes it a TextureCube.
    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC cube_desc{};
    cube_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    cube_desc.Width = face_size;
    cube_desc.Height = face_size;
    cube_desc.DepthOrArraySize = 6;
    cube_desc.MipLevels = 1;
    cube_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    cube_desc.SampleDesc.Count = 1;
    ComPtr<ID3D12Resource> cube;
    check(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &cube_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                          nullptr, IID_PPV_ARGS(&cube)), "CreateCommittedResource(cube)");
    constexpr UINT pitch = 256;  // D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
    const UINT64 slice_bytes = UINT64(pitch) * face_size;
    std::vector<char> staged(slice_bytes * 6, 0);
    for (UINT face = 0; face < 6; ++face)
        for (UINT row = 0; row < face_size; ++row)
            std::memcpy(&staged[face * slice_bytes + row * pitch], &faces[face * face_bytes + row * face_size * 16],
                        face_size * 16);
    auto cube_upload = make_buffer(device.Get(), D3D12_HEAP_TYPE_UPLOAD, staged.size(), D3D12_RESOURCE_STATE_GENERIC_READ);
    fill(cube_upload.Get(), staged.data(), staged.size());
    const UINT64 in_bytes = samples.size();
    auto input_upload = make_buffer(device.Get(), D3D12_HEAP_TYPE_UPLOAD, in_bytes, D3D12_RESOURCE_STATE_GENERIC_READ);
    fill(input_upload.Get(), samples.data(), samples.size());
    auto input = make_buffer(device.Get(), D3D12_HEAP_TYPE_DEFAULT, in_bytes, D3D12_RESOURCE_STATE_COPY_DEST);
    auto output = make_buffer(device.Get(), D3D12_HEAP_TYPE_DEFAULT, in_bytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = make_buffer(device.Get(), D3D12_HEAP_TYPE_READBACK, in_bytes, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap");
    const UINT stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = UINT(in_bytes / 4);
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    device->CreateShaderResourceView(input.Get(), &srv, cpu);
    cpu.ptr += stride;
    D3D12_SHADER_RESOURCE_VIEW_DESC cube_view{};
    cube_view.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    cube_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    cube_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    cube_view.TextureCube.MostDetailedMip = 0;
    cube_view.TextureCube.MipLevels = 1;
    device->CreateShaderResourceView(cube.Get(), &cube_view, cpu);
    cpu.ptr += stride;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = UINT(in_bytes / 4);
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    device->CreateUnorderedAccessView(output.Get(), nullptr, &uav, cpu);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
    ComPtr<ID3D12CommandAllocator> allocator;
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
          "CreateCommandList");

    for (UINT face = 0; face < 6; ++face) {
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = cube.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = face;
        src.pResource = cube_upload.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Offset = face * slice_bytes;
        src.PlacedFootprint.Footprint = {DXGI_FORMAT_R32G32B32A32_FLOAT, face_size, face_size, 1, pitch};
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    barrier(list.Get(), cube.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->CopyBufferRegion(input.Get(), 0, input_upload.Get(), 0, in_bytes);
    barrier(list.Get(), input.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(signature.Get());
    list->SetPipelineState(state.Get());
    const UINT zero[2] = {0, 0};
    list->SetComputeRoot32BitConstants(0, 2, zero, 0);
    auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
    list->SetComputeRootDescriptorTable(1, gpu);
    gpu.ptr += 2 * UINT64(stride);
    list->SetComputeRootDescriptorTable(2, gpu);
    list->Dispatch(1, 1, 1);
    barrier(list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, in_bytes);
    check(list->Close(), "Close");
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    check(queue->Signal(fence.Get(), 1), "Signal");
    check(fence->SetEventOnCompletion(1, event), "SetEventOnCompletion");
    if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0) throw std::runtime_error("GPU timeout");
    CloseHandle(event);

    void* mapped = nullptr;
    D3D12_RANGE range{0, size_t(in_bytes)};
    check(readback->Map(0, &range, &mapped), "Map(readback)");
    std::ofstream out(argv[5], std::ios::binary);
    out.write(static_cast<const char*>(mapped), std::streamsize(in_bytes));
    readback->Unmap(0, nullptr);
    return out ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "startup_cube_sample_check: %s\n", error.what());
        return 1;
    }
}
