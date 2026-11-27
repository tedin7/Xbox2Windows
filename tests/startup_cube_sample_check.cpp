// Read back a translated TextureCube sample from compute or through the D3D12 stand-in draw path.
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
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) std::size_t x2w_d3d12_census_report(char*, std::size_t);

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

constexpr UINT pitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
constexpr UINT64 slice_bytes = UINT64(pitch) * face_size;

ComPtr<ID3D12Resource> stage_faces(ID3D12Device* device, const std::vector<char>& faces, UINT slices) {
    std::vector<char> staged(slice_bytes * slices, 0);
    for (UINT face = 0; face < slices; ++face)
        for (UINT row = 0; row < face_size; ++row)
            std::memcpy(&staged[face * slice_bytes + row * pitch], &faces[face * face_bytes + row * face_size * 16],
                        face_size * 16);
    auto cube_upload = make_buffer(device, D3D12_HEAP_TYPE_UPLOAD, staged.size(), D3D12_RESOURCE_STATE_GENERIC_READ);
    fill(cube_upload.Get(), staged.data(), staged.size());
    return cube_upload;
}

void copy_faces(ID3D12GraphicsCommandList* list, ID3D12Resource* cube, ID3D12Resource* upload, UINT slices) {
    for (UINT face = 0; face < slices; ++face) {
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = cube;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = face;
        src.pResource = upload;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Offset = face * slice_bytes;
        src.PlacedFootprint.Footprint = {DXGI_FORMAT_R32G32B32A32_FLOAT, face_size, face_size, 1, pitch};
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
}

void wait_for_gpu(ID3D12Device* device, ID3D12CommandQueue* queue) {
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    check(queue->Signal(fence.Get(), 1), "Signal");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("cannot create completion event");
    const HRESULT hr = fence->SetEventOnCompletion(1, event);
    const DWORD completed = SUCCEEDED(hr) ? WaitForSingleObject(event, 30000) : WAIT_FAILED;
    CloseHandle(event);
    check(hr, "SetEventOnCompletion");
    if (completed != WAIT_OBJECT_0) throw std::runtime_error("GPU timeout");
    check(device->GetDeviceRemovedReason(), "GPU completion");
}

template <typename Method>
Method slot(IUnknown* object, unsigned index) {
    return reinterpret_cast<Method>((*reinterpret_cast<void***>(object))[index]);
}

template <typename T>
ComPtr<T> native_object(ID3D12Device* device, ID3D12Device* guest, IUnknown* wrapped) {
    // The guest forwards the underlying object to the desktop device.
    constexpr GUID key{0x9936b2de, 0xf642, 0x4d0d, {0x91, 0x74, 0x92, 0x95, 0x92, 0x6f, 0x30, 0x7a}};
    check(guest->SetPrivateDataInterface(key, wrapped), "publish desktop object");
    ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT hr = device->GetPrivateData(key, &bytes, stored.GetAddressOf());
    check(guest->SetPrivateDataInterface(key, nullptr), "clear desktop object");
    check(hr, "retrieve desktop object");
    ComPtr<T> native;
    check(stored.As(&native), "query desktop object");
    return native;
}

ComPtr<IUnknown> cube_pipeline(ID3D12Device* guest, ID3D12RootSignature* root, ID3DBlob* root_blob,
                               const std::filesystem::path& prefix, const std::vector<char>& vs,
                               const std::vector<char>& ps) {
    std::uint64_t key = 1469598103934665603ull;
    const auto add = [&key](const void* data, std::size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i) key = (key ^ p[i]) * 1099511628211ull;
    };
    constexpr char revision[] = "x2w-startup-shader-28";
    add(revision, sizeof(revision) - 1);
    add(vs.data(), vs.size());
    add(ps.data(), ps.size());
    add(root_blob->GetBufferPointer(), root_blob->GetBufferSize());
    char stem[17];
    std::snprintf(stem, sizeof(stem), "%016llx", static_cast<unsigned long long>(key));
    for (const char* suffix : {".vs.cso", ".ps.cso", ".txt"})
        std::filesystem::copy_file(prefix.string() + suffix, prefix.parent_path() / (std::string(stem) + suffix),
                                   std::filesystem::copy_options::overwrite_existing);
    const std::array<UINT64, 0x300 / sizeof(UINT64)> desc{
        reinterpret_cast<UINT64>(root), 0, 0, 0, 0,
        reinterpret_cast<UINT64>(vs.data()), vs.size(), 0, 0,
        reinterpret_cast<UINT64>(ps.data()), ps.size(), 0};
    using Create = HRESULT (WINAPI*)(IUnknown*, const void*, REFIID, void**);
    ComPtr<IUnknown> pipeline;
    check(slot<Create>(guest, 57)(guest, desc.data(), __uuidof(ID3D12PipelineState),
        reinterpret_cast<void**>(pipeline.GetAddressOf())), "create guest cube pipeline");
    return pipeline;
}

int draw_cube(int argc, char** argv) {
    if (argc != 7) throw std::runtime_error("usage: draw <prefix> <faces.bin> <samples.bin> <out.bin> <binding>");
    const std::filesystem::path prefix = argv[2];
    const auto faces = read_file(argv[3]);
    const auto samples = read_file(argv[4]);
    const std::string binding = argv[6];
    const bool is_null = binding != "cube";
    const bool short_resource = binding == "2d" || binding == "short-cube";
    if (is_null && !short_resource && binding != "cube-array") throw std::runtime_error("unknown cube binding case");
    constexpr UINT width = 8;
    if (faces.size() != 6 * face_bytes || samples.size() != width * 3 * sizeof(UINT))
        throw std::runtime_error("invalid cube fixture input sizes");

    ComPtr<ID3D12Debug> debug;
    const bool debug_enabled = SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
    if (debug_enabled) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory4> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 77;
    DXGI_ADAPTER_DESC adapter_desc{};
    check(adapter->GetDesc(&adapter_desc), "describe WARP adapter");
    LARGE_INTEGER driver{};
    std::printf("adapter=%ls debug_layer=%s\n", adapter_desc.Description, debug_enabled ? "enabled" : "unavailable");
    if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver)))
        std::printf("driver=%u.%u.%u.%u\n", HIWORD(driver.HighPart), LOWORD(driver.HighPart),
                    HIWORD(driver.LowPart), LOWORD(driver.LowPart));

    check(SetEnvironmentVariableW(L"X2W_SHADER_CACHE", prefix.parent_path().c_str()) ? S_OK : E_FAIL,
          "set cube cache directory");
    for (const char* name : {"X2W_D3D12_PERMISSIVE", "X2W_D3D12_LIST_NOOP", "X2W_D3D12_RENDER"})
        check(SetEnvironmentVariableA(name, "1") ? S_OK : E_FAIL, name);
    ComPtr<ID3D12Device> device, guest;
    check(D3D12XboxCreateDevice(adapter.Get(), nullptr, IID_PPV_ARGS(&device)), "create shim desktop device");
    const std::array<unsigned char, 64> parameters{};
    check(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), IID_PPV_ARGS(&guest)), "create guest device");
    ComPtr<ID3D12InfoQueue> messages;
    if (debug_enabled) check(device.As(&messages), "query debug message queue");

    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable = {1, &range};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants = {0, 0, 3};
    const D3D12_ROOT_SIGNATURE_DESC root_desc{2, params, 0, nullptr,
                                              D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    ComPtr<ID3DBlob> root_blob;
    check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, nullptr), "serialize guest root");
    ComPtr<ID3D12RootSignature> root;
    check(guest->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(), IID_PPV_ARGS(&root)),
          "create guest root");
    // The placeholder pipeline keeps the program pointers for its first bind.
    const auto vs = read_file((prefix.string() + ".a").c_str());
    const auto ps = read_file((prefix.string() + ".b").c_str());
    const auto pipeline = cube_pipeline(guest.Get(), root.Get(), root_blob.Get(), prefix, vs, ps);

    const UINT16 slices = short_resource ? 1 : 6;
    const D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC cube_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, face_size, face_size,
        slices, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN};
    using CreateTexture = HRESULT (WINAPI*)(IUnknown*, const D3D12_HEAP_PROPERTIES*, UINT,
        const D3D12_RESOURCE_DESC*, UINT, const D3D12_CLEAR_VALUE*, REFIID, void**);
    ComPtr<ID3D12Resource> wrapped_cube;
    check(slot<CreateTexture>(guest.Get(), 76)(guest.Get(), &properties, 0, &cube_desc, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&wrapped_cube)), "create guest cube resource");
    const auto cube = native_object<ID3D12Resource>(device.Get(), guest.Get(), wrapped_cube.Get());
    const auto upload = stage_faces(device.Get(), faces, slices);

    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } table_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    using CreateHeap = HRESULT (WINAPI*)(IUnknown*, const void*, REFIID, void**);
    ComPtr<ID3D12DescriptorHeap> table;
    check(slot<CreateHeap>(guest.Get(), 14)(guest.Get(), &table_desc, IID_PPV_ARGS(&table)), "create guest SRV table");
    using Handle = UINT64 (WINAPI*)(IUnknown*, void*);
    const UINT64 cpu = slot<Handle>(table.Get(), 9)(table.Get(), nullptr);
    const UINT64 gpu = slot<Handle>(table.Get(), 10)(table.Get(), nullptr);
    check(cpu && gpu ? S_OK : E_FAIL, "get guest SRV table handles");
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = cube_desc.Format;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (binding == "2d") {
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Texture2D.MipLevels = 1;
    } else if (binding == "cube-array") {
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
        view.TextureCubeArray.MipLevels = 1;
        view.TextureCubeArray.NumCubes = 1;
    } else {
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        view.TextureCube.MipLevels = 1;
    }
    using CreateView = void (WINAPI*)(IUnknown*, IUnknown*, const D3D12_SHADER_RESOURCE_VIEW_DESC*, UINT64, UINT64);
    slot<CreateView>(guest.Get(), 18)(guest.Get(), wrapped_cube.Get(), &view, cpu, 0);

    const D3D12_RESOURCE_DESC target_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, 1, 1, 1,
        DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE clear{target_desc.Format, {0.25f, 0.25f, 0.25f, 1.0f}};
    ComPtr<ID3D12Resource> target;
    check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &target_desc,
        D3D12_RESOURCE_STATE_COMMON, &clear, IID_PPV_ARGS(&target)), "create draw target");
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    check(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap)), "create target heap");
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    guest->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create guest allocator");
    using CreateList = HRESULT (WINAPI*)(IUnknown*, const void*, IUnknown*, IUnknown*, REFIID, void**);
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    ComPtr<IUnknown> wrapped;
    check(slot<CreateList>(guest.Get(), 73)(guest.Get(), list_desc, allocator.Get(), nullptr,
        __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(wrapped.GetAddressOf())), "create guest list");
    auto* commands = wrapped.Get();
    using Reset = HRESULT (WINAPI*)(IUnknown*, IUnknown*, void*, void*);
    check(slot<Reset>(commands, 139)(commands, allocator.Get(), nullptr, nullptr), "reset guest list");
    const auto list = native_object<ID3D12GraphicsCommandList>(device.Get(), guest.Get(), commands);
    barrier(list.Get(), cube.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    copy_faces(list.Get(), cube.Get(), upload.Get(), slices);
    barrier(list.Get(), cube.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);

    // A skipped draw leaves opaque grey. A successful draw with a null cube writes transparent black.
    using Clear = void (WINAPI*)(IUnknown*, SIZE_T, const float*, UINT, const D3D12_RECT*);
    slot<Clear>(commands, 52)(commands, rtv.ptr, clear.Color, 0, nullptr);
    using Targets = void (WINAPI*)(IUnknown*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, BOOL, const D3D12_CPU_DESCRIPTOR_HANDLE*);
    slot<Targets>(commands, 50)(commands, 1, &rtv, FALSE, nullptr);
    using Bind = void (WINAPI*)(IUnknown*, IUnknown*);
    slot<Bind>(commands, 25)(commands, pipeline.Get());
    slot<Bind>(commands, 34)(commands, root.Get());
    ID3D12DescriptorHeap* heaps[]{table.Get()};
    using Heaps = void (WINAPI*)(IUnknown*, UINT, ID3D12DescriptorHeap* const*);
    slot<Heaps>(commands, 32)(commands, 1, heaps);
    using Table = void (WINAPI*)(IUnknown*, UINT, UINT64);
    slot<Table>(commands, 36)(commands, 0, gpu);
    using Topology = void (WINAPI*)(IUnknown*, D3D12_PRIMITIVE_TOPOLOGY);
    slot<Topology>(commands, 20)(commands, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    using Viewport = void (WINAPI*)(IUnknown*, UINT, const D3D12_VIEWPORT*);
    using Scissor = void (WINAPI*)(IUnknown*, UINT, const D3D12_RECT*);
    using Constants = void (WINAPI*)(IUnknown*, UINT, UINT, const UINT*, UINT);
    using Draw = void (WINAPI*)(IUnknown*, UINT, UINT, UINT, UINT);
    for (UINT x = 0; x < width; ++x) {
        const D3D12_VIEWPORT viewport{static_cast<float>(x), 0, 1, 1, 0, 1};
        const D3D12_RECT scissor{static_cast<LONG>(x), 0, static_cast<LONG>(x + 1), 1};
        UINT address[3];
        std::memcpy(address, samples.data() + x * sizeof(address), sizeof(address));
        slot<Viewport>(commands, 21)(commands, 1, &viewport);
        slot<Scissor>(commands, 22)(commands, 1, &scissor);
        slot<Constants>(commands, 40)(commands, 1, 3, address, 0);
        slot<Draw>(commands, 12)(commands, 3, 1, 0, 0);
    }
    check(slot<HRESULT (WINAPI*)(IUnknown*)>(commands, 9)(commands), "close guest draw list");

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    const auto readback = make_buffer(device.Get(), D3D12_HEAP_TYPE_READBACK, bytes, D3D12_RESOURCE_STATE_COPY_DEST);
    ComPtr<ID3D12CommandAllocator> read_allocator;
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&read_allocator)), "create readback allocator");
    ComPtr<ID3D12GraphicsCommandList> read_list;
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, read_allocator.Get(), nullptr,
        IID_PPV_ARGS(&read_list)), "create readback list");
    barrier(read_list.Get(), target.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = target.Get();
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.pResource = readback.Get();
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = footprint;
    read_list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    check(read_list->Close(), "close readback list");
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    ComPtr<ID3D12CommandQueue> queue;
    check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "create draw queue");
    ID3D12CommandList* lists[]{list.Get(), read_list.Get()};
    queue->ExecuteCommandLists(2, lists);
    wait_for_gpu(device.Get(), queue.Get());

    char report[4096]{};
    x2w_d3d12_census_report(report, sizeof(report));
    const std::string census = report;
    const std::string counter = "draw null bindings: dimension ";
    const auto dimension_count = counter + std::to_string(width) + "\n";
    const bool count_ok = is_null ? census.find(dimension_count) != std::string::npos : census.find(counter) == std::string::npos;
    if (!count_ok)
        throw std::runtime_error("unexpected dimension census: " + census);
    if (is_null) std::printf("%s: %s", binding.c_str(), dimension_count.c_str());

    if (messages) {
        bool clean = true;
        for (UINT64 i = 0; i < messages->GetNumStoredMessages(); ++i) {
            SIZE_T size = 0;
            check(messages->GetMessage(i, nullptr, &size), "size debug message");
            std::vector<char> storage(size);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(messages->GetMessage(i, message, &size), "read debug message");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
                std::fprintf(stderr, "%s\n", message->pDescription);
                clean = false;
            }
        }
        if (!clean) throw std::runtime_error("debug layer reported a warning or error");
    }
    constexpr UINT pixel_bytes = width * 4;
    const D3D12_RANGE read{0, pixel_bytes};
    void* mapped = nullptr;
    check(readback->Map(0, &read, &mapped), "map cube draw result");
    std::ofstream output(argv[5], std::ios::binary);
    output.write(static_cast<const char*>(mapped), pixel_bytes);
    const D3D12_RANGE untouched{0, 0};
    readback->Unmap(0, &untouched);
    return output ? 0 : 1;
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
    const auto cube_upload = stage_faces(device.Get(), faces, 6);
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

    copy_faces(list.Get(), cube.Get(), cube_upload.Get(), 6);
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
    wait_for_gpu(device.Get(), queue.Get());

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
        if (argc > 1 && std::strcmp(argv[1], "draw") == 0) return draw_cube(argc, argv);
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "startup_cube_sample_check: %s\n", error.what());
        return 1;
    }
}
