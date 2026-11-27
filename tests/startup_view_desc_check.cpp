// Synthetic SRV regression. Mutant: make check() return Refusal::none without changing
// the descriptor. The pure test must fail on the reported mip shape even if WARP accepts it.
#include "startup_stubs/create_srv.h"
#include "startup_stubs/address_texture.h"

#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdarg>
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

static D3D12_SHADER_RESOURCE_VIEW_DESC raw_buffer_view(UINT64 first, UINT count)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    view.Format = DXGI_FORMAT_R32_TYPELESS;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Buffer = {first, count, 0, D3D12_BUFFER_SRV_FLAG_RAW};
    return view;
}

static bool buffer_checks()
{
    D3D12_RESOURCE_DESC resource{};
    resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resource.Width = 64;
    auto raw = raw_buffer_view(4, 12);
    auto typed = raw;
    typed.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    typed.Format = DXGI_FORMAT_R32_UINT;
    auto structured = typed;
    structured.Format = DXGI_FORMAT_UNKNOWN;
    structured.Buffer = {1, 3, 16, D3D12_BUFFER_SRV_FLAG_NONE};
    if (!unchanged(raw, resource) || !unchanged(typed, resource) || !unchanged(structured, resource))
        return false;

    auto bad_raw = raw;
    bad_raw.Format = DXGI_FORMAT_R32_UINT;
    auto out_of_range = raw_buffer_view(16, 1);
    auto overflowing = raw_buffer_view(~UINT64{0}, 1);
    auto zero = raw_buffer_view(0, 0);
    auto wrong_stride = structured;
    wrong_stride.Format = DXGI_FORMAT_R32_UINT;
    auto wrong_typed = typed;
    wrong_typed.Format = DXGI_FORMAT_R32_TYPELESS;
    return check(bad_raw, resource) == srv::Refusal::format &&
        check(out_of_range, resource) == srv::Refusal::buffer_range &&
        check(overflowing, resource) == srv::Refusal::buffer_range &&
        check(zero, resource) == srv::Refusal::buffer_range &&
        check(wrong_stride, resource) == srv::Refusal::format &&
        check(wrong_typed, resource) == srv::Refusal::format &&
        check(raw, texture_desc()) == srv::Refusal::dimension;
}

static bool alias_checks()
{
    // A three-mip alias starts at mip 4. Its mip 1 maps to desktop mip 5,
    // with the all-mips sentinel limited to the two mips left in the alias.
    D3D12_TEX2D_SRV view{1, ~0u, 0, 0};
    if (!x2w_startup::remap_alias_mips(view, 3, 4, 8) ||
        view.MostDetailedMip != 5 || view.MipLevels != 2) return false;
    D3D12_TEX2D_ARRAY_SRV missing_first{1, ~0u, 0, 1, 0, 0};
    D3D12_TEX2D_SRV missing_tail{1, ~0u, 0, 0};
    return !x2w_startup::remap_alias_mips(missing_first, 3, 4, 5) &&
        missing_first.MostDetailedMip == 1 && missing_first.MipLevels == ~0u &&
        !x2w_startup::remap_alias_mips(missing_tail, 3, 4, 6) &&
        missing_tail.MostDetailedMip == 1 && missing_tail.MipLevels == ~0u;
}

static bool pure_checks()
{
    auto resource = texture_desc();
    auto reported = view_desc(13, 0);
    if (check(reported, resource) != srv::Refusal::mip_range) {
        std::puts("FAIL: reported mip shape must be refused");
        return false;
    }
    const D3D12_BUFFER_SRV buffer_fields{13, 1, 0, D3D12_BUFFER_SRV_FLAG_RAW};
    auto union_mismatch = view_desc(0, 1, 0);
    std::memcpy(&union_mismatch.Texture2D, &buffer_fields, sizeof(union_mismatch.Texture2D));
    if (union_mismatch.Texture2D.MostDetailedMip != 13 || union_mismatch.Texture2D.MipLevels != 0) return false;
    if (check(union_mismatch, resource) != srv::Refusal::mip_range) return false;

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

static D3D12_RESOURCE_DESC depth_stencil_desc()
{
    auto desc = texture_desc();
    desc.Width = desc.Height = 1024;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    return desc;
}

static bool depth_stencil_checks()
{
    auto resource = depth_stencil_desc();
    auto stencil = view_desc(0, 1);
    stencil.Format = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    // Mutant: omit the PlaneSlice update. This exact shape must fail even if
    // the native runtime tolerates plane 0.
    if (check(stencil, resource) != srv::Refusal::none || stencil.Texture2D.PlaneSlice != 1 ||
        stencil.Format != DXGI_FORMAT_X24_TYPELESS_G8_UINT) {
        std::puts("FAIL: D24S8 stencil SRV must select plane 1");
        return false;
    }

    auto depth = view_desc(0, 1);
    depth.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    if (!unchanged(depth, resource)) return false;
    depth.Texture2D.PlaneSlice = 1;
    if (check(depth, resource) != srv::Refusal::none || depth.Texture2D.PlaneSlice != 0) return false;

    auto wrong_family = view_desc(0, 1);
    wrong_family.Format = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    if (check(wrong_family, resource) != srv::Refusal::none ||
        wrong_family.Format != DXGI_FORMAT_X24_TYPELESS_G8_UINT ||
        wrong_family.Texture2D.PlaneSlice != 1) return false;
    auto unrelated = view_desc(0, 1);
    if (check(unrelated, resource) != srv::Refusal::format) return false;

    resource.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
    resource.DepthOrArraySize = 2;
    depth.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    depth.Texture2DArray = {0, 1, 1, 1, 3, 0.0f};
    if (check(depth, resource) != srv::Refusal::none ||
        depth.Format != DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS || depth.Texture2DArray.PlaneSlice != 0)
        return false;
    resource.SampleDesc.Count = 4;
    stencil.Format = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
    stencil.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
    stencil.Texture2DMSArray = {1, 1};
    if (!unchanged(stencil, resource)) return false;

    auto single_plane = view_desc(0, 1);
    single_plane.Texture2D.PlaneSlice = 1;
    if (check(single_plane, texture_desc()) != srv::Refusal::none || single_plane.Texture2D.PlaneSlice != 0)
        return false;
    resource = texture_desc();
    resource.Format = DXGI_FORMAT_R32_TYPELESS;
    single_plane.Format = DXGI_FORMAT_R32_FLOAT;
    single_plane.Texture2D.PlaneSlice = 1;
    return check(single_plane, resource) == srv::Refusal::none && single_plane.Texture2D.PlaneSlice == 0;
}

static unsigned log_lines;
static char last_log[256];

static void capture_format_log(const char* format, ...)
{
    if (std::strncmp(format, "x2w-d3d12 console format ", sizeof("x2w-d3d12 console format ") - 1) &&
        std::strncmp(format, "x2w-d3d12 SRV census ", sizeof("x2w-d3d12 SRV census ") - 1)) return;
    ++log_lines;
    va_list args;
    va_start(args, format);
    std::vsnprintf(last_log, sizeof(last_log), format, args);
    va_end(args);
}

static D3D12_RESOURCE_DESC console_depth_desc()
{
    auto desc = texture_desc();
    desc.Width = 2560;
    desc.Height = 1440;
    desc.MipLevels = 1;
    desc.Format = static_cast<DXGI_FORMAT>(118);
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    return desc;
}

static bool console_format_log_check()
{
    auto desc = console_depth_desc();
    const D3D12_CLEAR_VALUE clear{desc.Format};
    D3D12_CLEAR_VALUE copy{};
    const auto* mapped = x2w_startup::desktop_clear_value(&clear, desc.Format,
        x2w_startup::desktop_format(desc.Format), copy);
    log_lines = 0;
    x2w_startup::log_console_format("committed", 76, desc, &clear, mapped, capture_format_log);
    // Removing the size field (or any digit of it) must fail this comparison.
    if (log_lines != 1 || std::strcmp(last_log,
            "x2w-d3d12 console format 118 -> desktop format 45 path=committed slot=76 dim=3 size=2560x1440x1 "
            "mips=1 samples=1 flags=0x2 clear_present=1 clear_mapped=1")) return false;

    for (unsigned i = 1; i < 16; ++i) {
        ++desc.Width;
        x2w_startup::log_console_format("committed", 76, desc, nullptr, nullptr, capture_format_log);
    }
    if (log_lines != 16 || !std::strstr(last_log, "clear_present=0 clear_mapped=0")) return false;
    for (unsigned i = 17; i <= 32; ++i) {
        ++desc.Width;
        x2w_startup::log_console_format("committed", 76, desc, nullptr, nullptr, capture_format_log);
    }
    return log_lines == 17 && !std::strcmp(last_log, "x2w-d3d12 console format 118 -> desktop format 45 count=32");
}

static bool stencil_census_check(ID3D12Device* device, const D3D12_HEAP_PROPERTIES& heap,
                                 D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
    auto desc = console_depth_desc();
    const auto original = desc.Format;
    desc.Format = x2w_startup::desktop_format(original);
    D3D12_CLEAR_VALUE clear{desc.Format};
    clear.DepthStencil.Depth = 1.0f;
    ComPtr<ID3D12Resource> resource;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               &clear, IID_PPV_ARGS(&resource)))) return false;
    x2w_startup::mark_console_format(resource.Get(), original);
    if (!x2w_startup::has_console_format_remap(resource.Get())) return false;

    auto stencil = view_desc(0, 1);
    stencil.Format = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    stencil.Texture2D.PlaneSlice = 1;
    log_lines = 0;
    // The typed remap retains the existing refusal and writes a null descriptor.
    if (x2w_startup::x2w_create_srv(device, resource.Get(), &stencil, dest, "fixture-stencil", capture_format_log) ||
        log_lines != 1 || std::strcmp(last_log,
            "x2w-d3d12 SRV census size=2560x1440x1 resource_format=45 view_format=47 "
            "plane_requested=1 plane_used=-1 count=1")) return false;

    // A typeless depth-stencil backing can carry a stencil SRV using plane 1.
    desc.Format = DXGI_FORMAT_R24G8_TYPELESS;
    resource.Reset();
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               &clear, IID_PPV_ARGS(&resource)))) return false;
    log_lines = 0;
    for (unsigned i = 0; i < 4; ++i) {
        if (!x2w_startup::x2w_create_srv(device, resource.Get(), &stencil, dest, "fixture-stencil", capture_format_log))
            return false;
    }
    return log_lines == 3 && !std::strcmp(last_log,
        "x2w-d3d12 SRV census size=2560x1440x1 resource_format=44 view_format=47 "
        "plane_requested=1 plane_used=1 count=4");
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
    heap_desc.NumDescriptors = 3;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptors)))) return 1;

    ComPtr<ID3D12InfoQueue> messages;
    if (debug && FAILED(device.As(&messages))) return 1;
    if (messages) messages->ClearStoredMessages();

    auto repairable = view_desc(6, 8, 0);
    auto dest = descriptors->GetCPUDescriptorHandleForHeapStart();
    if (!x2w_startup::x2w_create_srv(device.Get(), resource.Get(), &repairable, dest, "fixture-repair"))
        return 1;

    auto reported = view_desc(13, 0);
    dest.ptr += device->GetDescriptorHandleIncrementSize(heap_desc.Type);
    if (x2w_startup::x2w_create_srv(device.Get(), resource.Get(), &reported, dest, "fixture-refusal"))
        return 1;
    const auto history = x2w_startup::recent_srvs();
    if (!x2w_startup::last_srv.site || std::strcmp(x2w_startup::last_srv.site, "fixture-refusal") ||
        !history[14].site || std::strcmp(history[14].site, "fixture-repair") ||
        !history[15].site || std::strcmp(history[15].site, "fixture-refusal")) return 1;
    const auto depth_desc = depth_stencil_desc();
    ComPtr<ID3D12Resource> depth;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &depth_desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&depth)))) {
        std::puts("FAIL: WARP could not create the D24S8 resource");
        return 1;
    }
    auto stencil = view_desc(0, 1);
    stencil.Format = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    dest.ptr += device->GetDescriptorHandleIncrementSize(heap_desc.Type);
    if (!x2w_startup::x2w_create_srv(device.Get(), depth.Get(), &stencil, dest, "fixture-stencil") ||
        FAILED(device->GetDeviceRemovedReason())) {
        std::puts("FAIL: WARP rejected the repaired D24S8 stencil SRV");
        return 1;
    }
    if (messages && messages->GetNumStoredMessages() != 0) {
        std::puts("FAIL: debug layer reported messages while creating SRVs");
        return 1;
    }
    if (!stencil_census_check(device.Get(), heap, dest)) {
        std::printf("FAIL: stencil SRV census, last line: %s\n", last_log);
        return 1;
    }
    if (FAILED(device->GetDeviceRemovedReason())) return 1;
    std::puts("PASS: WARP created the repaired color and stencil SRVs and the null SRV");
    return 0;
}

int main(int argc, char** argv)
{
    if (!buffer_checks() || !alias_checks() || !pure_checks()) {
        std::puts("FAIL: texture, alias mip and buffer SRV descriptor checks");
        return 1;
    }
    if (!depth_stencil_checks()) {
        std::puts("FAIL: depth-stencil SRV descriptor checks");
        return 1;
    }
    if (!console_format_log_check()) {
        std::printf("FAIL: console resource format census, last line: %s\n", last_log);
        return 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "warp") == 0) return warp_check();
    std::puts("PASS: texture, alias mip and buffer SRV descriptor checks");
    return 0;
}
