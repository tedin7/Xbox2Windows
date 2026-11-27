// Synthetic SRV regression. Mutant: make check() return Refusal::none without changing
// the descriptor. The pure test must fail on the reported mip shape even if WARP accepts it.
#include "startup_stubs/view_desc_check.h"

#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
namespace srv = x2w_startup::view_desc;

static D3D12_RESOURCE_DESC texture_desc()
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = 128;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 8;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    return desc;
}

static D3D12_SHADER_RESOURCE_VIEW_DESC view_desc(UINT mip, UINT count, UINT mapping = 0x1688)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping = mapping;
    view.Texture2D = {mip, count, 0, 0.0f};
    return view;
}

static srv::Refusal check(D3D12_SHADER_RESOURCE_VIEW_DESC& view,
                          const D3D12_RESOURCE_DESC& resource, unsigned dropped = 0)
{
    return srv::check(view, resource, dropped, [](srv::Field, UINT, UINT) {});
}

static bool unchanged(D3D12_SHADER_RESOURCE_VIEW_DESC& view, const D3D12_RESOURCE_DESC& resource)
{
    unsigned char before[sizeof(view)];
    std::memcpy(before, &view, sizeof(view));
    return check(view, resource) == srv::Refusal::none && std::memcmp(before, &view, sizeof(view)) == 0;
}

static bool pure_checks()
{
    auto resource = texture_desc();
    auto reported = view_desc(13, 0);
    if (check(reported, resource) != srv::Refusal::mip_range) {
        std::puts("FAIL: reported mip shape must be refused");
        return false;
    }
    auto zero_mips = view_desc(0, 0);
    auto bad_start = view_desc(8, 1);
    if (check(zero_mips, resource) != srv::Refusal::mip_range ||
        check(bad_start, resource) != srv::Refusal::mip_range) return false;

    auto valid = view_desc(2, 3);
    auto all_mips = view_desc(7, ~0u, D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(2, 1, 0, 5));
    if (!unchanged(valid, resource) || !unchanged(all_mips, resource)) return false;

    auto repairable = view_desc(6, 8, 0x168f);  // component selector 7 is invalid
    repairable.Format = DXGI_FORMAT_R16_FLOAT;
    if (check(repairable, resource) != srv::Refusal::none || repairable.Texture2D.MipLevels != 2 ||
        repairable.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING ||
        repairable.Format != resource.Format || srv::valid_mapping(0x688)) return false;

    auto dropped = view_desc(0, 6);
    resource.MipLevels = 5;
    if (check(dropped, resource, 3) != srv::Refusal::none || dropped.Texture2D.MipLevels != 3 ||
        check(valid, resource, 3) != srv::Refusal::dropped_mips) return false;

    resource.DepthOrArraySize = 4;
    auto array = view_desc(0, 1);
    array.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    array.Texture2DArray = {0, 1, 2, 9, 0, 0.0f};
    if (check(array, resource) != srv::Refusal::none || array.Texture2DArray.ArraySize != 2 ||
        !unchanged(array, resource)) return false;
    array.Texture2DArray.FirstArraySlice = 4;
    if (check(array, resource) != srv::Refusal::array_range ||
        check(valid, resource) != srv::Refusal::dimension) return false;

    resource.DepthOrArraySize = 18;
    auto cubes = view_desc(0, 1);
    cubes.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
    cubes.TextureCubeArray = {0, 1, 6, 5, 0.0f};
    if (check(cubes, resource) != srv::Refusal::none || cubes.TextureCubeArray.NumCubes != 2 ||
        !unchanged(cubes, resource)) return false;

    resource = texture_desc();
    resource.SampleDesc.Count = 4;
    resource.MipLevels = 1;
    if (check(valid, resource) != srv::Refusal::dimension) return false;
    auto ms = view_desc(0, 1);
    ms.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
    ms.Texture2DMS = {};
    if (!unchanged(ms, resource)) return false;

    resource = texture_desc();
    resource.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
    return unchanged(valid, resource) && check(reported, resource) == srv::Refusal::mip_range &&
        srv::check_default(resource) == srv::Refusal::format;
}

static int warp_check()
{
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
    std::printf("WARP, debug layer %s\n", debug ? "enabled" : "unavailable");

    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        std::puts("SKIP: WARP D3D12 device unavailable");
        return 77;
    }
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    const auto desc = texture_desc();
    ComPtr<ID3D12Resource> resource;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&resource)))) return 1;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 2;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptors)))) return 1;

    ComPtr<ID3D12InfoQueue> messages;
    if (debug && FAILED(device.As(&messages))) return 1;
    if (messages) messages->ClearStoredMessages();

    auto repairable = view_desc(6, 8, 0);
    if (check(repairable, resource->GetDesc()) != srv::Refusal::none) return 1;
    auto dest = descriptors->GetCPUDescriptorHandleForHeapStart();
    device->CreateShaderResourceView(resource.Get(), &repairable, dest);

    auto reported = view_desc(13, 0);
    if (check(reported, resource->GetDesc()) != srv::Refusal::mip_range) return 1;
    auto null_view = view_desc(0, 1);
    dest.ptr += device->GetDescriptorHandleIncrementSize(heap_desc.Type);
    device->CreateShaderResourceView(nullptr, &null_view, dest);
    if (FAILED(device->GetDeviceRemovedReason())) return 1;
    if (messages && messages->GetNumStoredMessages() != 0) {
        std::puts("FAIL: debug layer reported messages while creating SRVs");
        return 1;
    }
    std::puts("PASS: WARP created the repaired and null SRVs");
    return 0;
}

int main(int argc, char** argv)
{
    if (!pure_checks()) {
        std::puts("FAIL: texture SRV descriptor checks");
        return 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "warp") == 0) return warp_check();
    std::puts("PASS: texture SRV descriptor checks");
    return 0;
}
