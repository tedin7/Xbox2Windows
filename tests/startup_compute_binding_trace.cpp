#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Microsoft::WRL::ComPtr;

void check(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) {
        std::fprintf(stderr, "%s failed: 0x%08lx\n", stage, static_cast<unsigned long>(hr));
        std::exit(1);
    }
}

template<typename Method>
Method slot(IUnknown* object, unsigned index)
{
    return reinterpret_cast<Method>((*reinterpret_cast<void***>(object))[index]);
}
}

int main(int argc, char** argv)
{
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> warp;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
    check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "WARP adapter");

    const char source[] = R"(
        ByteAddressBuffer input : register(t0);
        RWByteAddressBuffer output : register(u0);
        cbuffer Params : register(b0) { uint4 args[4]; };
        [numthreads(1,1,1)] void main() { output.Store(0, input.Load(0) + args[0].y); }
    )";
    ComPtr<ID3DBlob> shader;
    check(D3DCompile(source, sizeof(source)-1, nullptr, nullptr, nullptr, "main", "cs_5_0",
                     0, 0, &shader, nullptr), "compile source-built shader");
    std::uint64_t hash = 1469598103934665603ull;
    const auto* bytes = static_cast<const unsigned char*>(shader->GetBufferPointer());
    for (std::size_t i = 0; i < shader->GetBufferSize(); ++i)
        hash = (hash ^ bytes[i]) * 1099511628211ull;
    char key[17];
    std::snprintf(key, sizeof(key), "%016llx", hash);
    std::printf("key=%s\n", key);
    char selected[38];
    std::snprintf(selected, sizeof(selected), "0123456789abcdef,%s", key);
    SetEnvironmentVariableA("X2W_D3D12_COMPUTE_BINDINGS", argc > 1 && std::strcmp(argv[1], "off") == 0 ? nullptr :
        argc > 1 && std::strcmp(argv[1], "unselected") == 0 ? "0123456789abcdef" : selected);
    SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", "1");
    SetEnvironmentVariableA("X2W_D3D12_RENDER", "1");

    ComPtr<ID3D12Device> guest;
    const std::uint64_t create_params[8]{};
    check(D3D12XboxCreateDevice(warp.Get(), create_params, IID_PPV_ARGS(&guest)), "guest device");
    ComPtr<ID3D12Device> host;
    check(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&host)), "desktop device");

    const auto root_blob = [](UINT space) {
        D3D12_ROOT_PARAMETER params[3]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[2].Constants = {0, space, 16};
        D3D12_ROOT_SIGNATURE_DESC desc{3, params};
        ComPtr<ID3DBlob> result;
        check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &result, nullptr), "serialize root");
        return result;
    };
    const auto blob = root_blob(0);
    ComPtr<ID3D12RootSignature> desktop_root, root, different;
    check(host->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&desktop_root)), "desktop root");
    check(guest->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "guest root");
    const auto other = root_blob(1);
    check(guest->CreateRootSignature(0, other->GetBufferPointer(), other->GetBufferSize(), IID_PPV_ARGS(&different)), "second guest root");
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = desktop_root.Get();
    desc.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
    ComPtr<ID3D12PipelineState> desktop_pso;
    check(host->CreateComputePipelineState(&desc, IID_PPV_ARGS(&desktop_pso)), "source-built desktop PSO");
    desc.pRootSignature = root.Get();
    using SetUp = HRESULT(WINAPI*)(IUnknown*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, UINT, const void*, REFIID, void**);
    ComPtr<ID3D12PipelineState> pso;
    check(slot<SetUp>(guest.Get(), 62)(guest.Get(), &desc, 0, nullptr, IID_PPV_ARGS(&pso)), "slot 62 PSO");

    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "guest allocator");
    const UINT list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    using Create = HRESULT(WINAPI*)(IUnknown*, const void*, IUnknown*, IUnknown*, REFIID, void**);
    ComPtr<ID3D12GraphicsCommandList> list;
    check(slot<Create>(guest.Get(), 73)(guest.Get(), list_desc, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "guest list");
    using Reset = HRESULT(WINAPI*)(IUnknown*, IUnknown*, void*, void*);
    check(slot<Reset>(list.Get(), 139)(list.Get(), allocator.Get(), nullptr, nullptr), "initial reset");

    using Upload = void*(WINAPI*)(IUnknown*, UINT64, UINT64);
    const auto upload = reinterpret_cast<UINT64>(slot<Upload>(list.Get(), 148)(list.Get(), 64, 16));
    if (!upload) return 2;
    using Place = HRESULT(WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
    std::unique_ptr<void, decltype([](void* p) { VirtualFree(p, 0, MEM_RELEASE); })> backing(
        VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!backing) return 3;
    const auto address = static_cast<char*>(backing.get()) + 4;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = 4096;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> placed;
    check(slot<Place>(guest.Get(), 133)(guest.Get(), address, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                       IID_PPV_ARGS(&placed)), "slot 133 buffer");

    using Pipeline = void(WINAPI*)(IUnknown*, IUnknown*);
    using Root = void(WINAPI*)(IUnknown*, IUnknown*);
    using Address = void(WINAPI*)(IUnknown*, UINT, UINT64);
    using Constant = void(WINAPI*)(IUnknown*, UINT, UINT, UINT);
    using Constants = void(WINAPI*)(IUnknown*, UINT, UINT, const void*, UINT);
    using Dispatch = void(WINAPI*)(IUnknown*, UINT, UINT, UINT);
    slot<Pipeline>(list.Get(), 25)(list.Get(), pso.Get());
    slot<Root>(list.Get(), 33)(list.Get(), root.Get());
    slot<Address>(list.Get(), 43)(list.Get(), 0, upload);
    slot<Address>(list.Get(), 45)(list.Get(), 1, reinterpret_cast<UINT64>(address));
    slot<Constant>(list.Get(), 37)(list.Get(), 2, 91, 1);
    const UINT words[] = {7, 11};
    slot<Constants>(list.Get(), 39)(list.Get(), 2, 2, words, 4);
    slot<Root>(list.Get(), 33)(list.Get(), root.Get());
    slot<Dispatch>(list.Get(), 14)(list.Get(), 2, 1, 1);
    slot<Address>(list.Get(), 43)(list.Get(), 0, upload + 4);
    slot<Dispatch>(list.Get(), 14)(list.Get(), 3, 1, 1);
    slot<Root>(list.Get(), 33)(list.Get(), different.Get());
    slot<Dispatch>(list.Get(), 14)(list.Get(), 4, 1, 1);

    check(list->Close(), "close guest list");
    check(slot<Reset>(list.Get(), 139)(list.Get(), allocator.Get(), nullptr, nullptr), "reset guest list");
    slot<Pipeline>(list.Get(), 25)(list.Get(), pso.Get());
    slot<Dispatch>(list.Get(), 14)(list.Get(), 5, 1, 1);
    check(list->Close(), "final close");
}
