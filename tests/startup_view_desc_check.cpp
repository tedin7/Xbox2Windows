// Synthetic SRV regression. Mutants: make check() return Refusal::none without changing the descriptor (the pure test
// fails on the reported mip shape even if WARP accepts it); refuse MipLevels 0 again, or read a UAV token as SRV bytes
// (pure checks and the WARP readback of every mip go red).
#include "startup_stubs/create_srv.h"
#include "startup_stubs/address_texture.h"

#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cmath>
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

    // MipLevels 0 on a mipped texture (the draw-texture shape): repaired to -1, every level from MostDetailedMip.
    auto zero_mips = view_desc(3, 0);
    UINT zero_from = 1, zero_to = 0;
    const auto zero_refusal = srv::check(zero_mips, resource, 0, [&](srv::Field field, UINT from, UINT to) {
        if (field == srv::Field::mip_levels) zero_from = from, zero_to = to;
    });
    auto bad_start = view_desc(8, 1);
    if (zero_refusal != srv::Refusal::none || zero_mips.Texture2D.MipLevels != ~0u ||
        zero_mips.Texture2D.MostDetailedMip != 3 || zero_from != 0 || zero_to != ~0u ||
        check(bad_start, resource) != srv::Refusal::mip_range) {
        std::puts("FAIL: MipLevels 0 must be repaired to -1 and an out-of-range MostDetailedMip refused");
        return false;
    }

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

static D3D12_SHADER_RESOURCE_VIEW_DESC from_record(unsigned slot, const D3D12_UNORDERED_ACCESS_VIEW_DESC& uav)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC out{};
    if (!srv::recorded_srv(slot, &uav, sizeof(uav), out)) out.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    return out;
}

// A sampled texture token that names an unordered access view: the draw reads the UAV's mip as an SRV.
static bool uav_record_checks()
{
    const auto resource = texture_desc();
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uav.Texture2D = {2, 0};
    auto plain = from_record(19, uav);
    if (plain.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D || plain.Texture2D.MostDetailedMip != 2 ||
        plain.Texture2D.MipLevels != 1 || !unchanged(plain, resource)) return false;

    auto layered = texture_desc();
    layered.DepthOrArraySize = 4;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
    uav.Texture2DArray = {1, 2, 2, 0};
    auto array = from_record(19, uav);
    if (array.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2DARRAY || array.Texture2DArray.MostDetailedMip != 1 ||
        array.Texture2DArray.MipLevels != 1 || array.Texture2DArray.FirstArraySlice != 2 ||
        array.Texture2DArray.ArraySize != 2 || !unchanged(array, layered)) return false;

    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    if (from_record(19, uav).ViewDimension != D3D12_SRV_DIMENSION_UNKNOWN) return false;
    D3D12_SHADER_RESOURCE_VIEW_DESC shaped{};
    const auto single = view_desc(1, 2);
    if (!srv::as_dimension(single, D3D12_SRV_DIMENSION_TEXTURE2DARRAY, shaped) ||
        shaped.Texture2DArray.MostDetailedMip != 1 || shaped.Texture2DArray.MipLevels != 2 ||
        shaped.Texture2DArray.ArraySize != 1 || !unchanged(shaped, layered)) return false;
    auto back = shaped;
    if (!srv::as_dimension(shaped, D3D12_SRV_DIMENSION_TEXTURE2D, back) || std::memcmp(&back, &single, sizeof(back)))
        return false;
    shaped.Texture2DArray.FirstArraySlice = 1;
    return !srv::as_dimension(shaped, D3D12_SRV_DIMENSION_TEXTURE2D, back) &&
        !srv::as_dimension(single, D3D12_SRV_DIMENSION_TEXTURECUBE, back);
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
    // A planar depth-stencil backing now creates the stencil SRV on plane 1 and logs the census line.
    if (!x2w_startup::x2w_create_srv(device, resource.Get(), &stencil, dest, "fixture-stencil", capture_format_log) ||
        log_lines != 1 || std::strcmp(last_log,
            "x2w-d3d12 SRV census size=2560x1440x1 resource_format=45 view_format=47 "
            "plane_requested=1 plane_used=1 count=1")) return false;

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

static UINT mip_texel(UINT mip, UINT x, UINT y)
{
    return 0xff000000u | (mip * 0x50u + 0x20u) << 16 | (y * 0x30u + 0x10u) << 8 | (x * 0x30u + 0x08u);
}

static bool run_and_wait(ID3D12Device* device, ID3D12GraphicsCommandList* list)
{
    ComPtr<ID3D12CommandQueue> queue;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    ComPtr<ID3D12Fence> fence;
    if (FAILED(list->Close()) || FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return false;
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const bool waited = done && SUCCEEDED(queue->Signal(fence.Get(), 1)) &&
        SUCCEEDED(fence->SetEventOnCompletion(1, done)) && WaitForSingleObject(done, 10000) == WAIT_OBJECT_0;
    if (done) CloseHandle(done);
    return waited && fence->GetCompletedValue() == 1;
}

// The draw-texture shape on the GPU: a 4x4 texture with 3 mips, sampled through the repaired MipLevels-0 view (t0, every
// level) and through the SRV made from a UAV of mip 1 (t1). Each texel read back must equal the uploaded one.
static bool mip_zero_readback(ID3D12Device* device)
{
    constexpr UINT side = 4, mips = 3, pitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, out_bytes = 4 * 4 * 4 * 16;  // 4x4x4 threads, 16 bytes each
    const auto heap = [](D3D12_HEAP_TYPE type) { D3D12_HEAP_PROPERTIES p{}; p.Type = type; return p; };
    const auto buffer = [&](D3D12_HEAP_TYPE type, UINT64 bytes, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                       D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags};
        const auto properties = heap(type);
        ComPtr<ID3D12Resource> r;
        device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    };
    auto upload = buffer(D3D12_HEAP_TYPE_UPLOAD, pitch * side * mips, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto out = buffer(D3D12_HEAP_TYPE_DEFAULT, out_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto back = buffer(D3D12_HEAP_TYPE_READBACK, out_bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    const D3D12_RESOURCE_DESC tex{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, side, side, 1, mips, DXGI_FORMAT_R8G8B8A8_UNORM,
                                  {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    const auto defaults = heap(D3D12_HEAP_TYPE_DEFAULT);
    ComPtr<ID3D12Resource> texture;
    if (!upload || !out || !back || FAILED(device->CreateCommittedResource(&defaults, D3D12_HEAP_FLAG_NONE, &tex,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture)))) return false;
    unsigned char* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return false;
    for (UINT mip = 0; mip < mips; ++mip)
        for (UINT y = 0; y < (side >> mip); ++y)
            for (UINT x = 0; x < (side >> mip); ++x) {
                const UINT texel = mip_texel(mip, x, y);
                std::memcpy(mapped + pitch * (side * mip + y) + 4 * x, &texel, 4);
            }
    upload->Unmap(0, nullptr);

    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heaps;
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heaps)))) return false;
    const UINT step = device->GetDescriptorHandleIncrementSize(heap_desc.Type);
    auto cpu = heaps->GetCPUDescriptorHandleForHeapStart();
    auto all_levels = view_desc(0, 0);
    if (!x2w_startup::x2w_create_srv(device, texture.Get(), &all_levels, cpu, "fixture-draw-texture")) {
        std::puts("FAIL: the MipLevels 0 view of a mipped texture was refused");
        return false;
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uav.Texture2D.MipSlice = 1;
    auto from_uav = from_record(19, uav);
    cpu.ptr += step;
    if (!x2w_startup::x2w_create_srv(device, texture.Get(), &from_uav, cpu, "fixture-uav-token")) return false;
    D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
    raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    raw.Buffer.NumElements = out_bytes / 4;
    raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    cpu.ptr += step;
    device->CreateUnorderedAccessView(out.Get(), nullptr, &raw, cpu);

    static const char source[] =
        "Texture2D<float4> all_levels : register(t0); Texture2D<float4> uav_mip : register(t1);\n"
        "RWByteAddressBuffer dst : register(u0);\n"
        "[numthreads(4, 4, 4)] void main(uint3 id : SV_DispatchThreadID)\n"
        "{ float4 v = id.z < 3 ? all_levels.Load(int3(int2(id.xy >> id.z), int(id.z))) : uav_mip.Load(int3(int2(id.xy >> 1), 0));\n"
        "  dst.Store4(((id.z * 4 + id.y) * 4 + id.x) * 16, asuint(v)); }\n";
    ComPtr<ID3DBlob> code, signature, errors;
    if (FAILED(D3DCompile(source, sizeof(source) - 1, "mip_zero", nullptr, nullptr, "main", "cs_5_0", 0, 0, &code, &errors)))
        return false;
    const D3D12_DESCRIPTOR_RANGE ranges[2] = {{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0},
                                              {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 2}};
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable = {2, ranges};
    const D3D12_ROOT_SIGNATURE_DESC root_desc{1, &parameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3D12RootSignature> root;
    if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &errors)) ||
        FAILED(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&root))))
        return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipe{};
    pipe.pRootSignature = root.Get();
    pipe.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(device->CreateComputePipelineState(&pipe, IID_PPV_ARGS(&pipeline))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(),
                                         IID_PPV_ARGS(&list)))) return false;
    for (UINT mip = 0; mip < mips; ++mip) {
        D3D12_TEXTURE_COPY_LOCATION to{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        to.SubresourceIndex = mip;
        D3D12_TEXTURE_COPY_LOCATION from{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        from.PlacedFootprint.Offset = UINT64{pitch} * side * mip;
        from.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, side >> mip, side >> mip, 1, pitch};
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    const auto transition = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{};
        b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        list->ResourceBarrier(1, &b);
    };
    transition(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ID3D12DescriptorHeap* bound[] = {heaps.Get()};
    list->SetDescriptorHeaps(1, bound);
    list->SetComputeRootSignature(root.Get());
    list->SetComputeRootDescriptorTable(0, heaps->GetGPUDescriptorHandleForHeapStart());
    list->Dispatch(1, 1, 1);
    transition(out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(back.Get(), 0, out.Get(), 0, out_bytes);
    if (!run_and_wait(device, list.Get())) return false;

    const float* read = nullptr;
    if (FAILED(back->Map(0, nullptr, reinterpret_cast<void**>(const_cast<float**>(&read))))) return false;
    int bad = 0;
    for (UINT z = 0; z < 4; ++z)
        for (UINT y = 0; y < side; ++y)
            for (UINT x = 0; x < side; ++x) {
                const UINT mip = z < 3 ? z : 1;
                const UINT shift = z < 3 ? z : 1;
                const UINT texel = mip_texel(mip, x >> shift, y >> shift);
                const float* got = read + ((z * 4 + y) * 4 + x) * 4;
                for (int c = 0; c < 4; ++c) {
                    const float expected = static_cast<float>((texel >> (8 * c)) & 255u) / 255.0f;
                    if (std::fabs(got[c] - expected) > 1e-4f && bad++ < 3)
                        std::printf("FAIL: view %u (%u,%u) channel %d read %f, expected %f\n", z, x, y, c, got[c], expected);
                }
            }
    back->Unmap(0, nullptr);
    return bad == 0;
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
    if (!mip_zero_readback(device.Get()) || FAILED(device->GetDeviceRemovedReason())) {
        std::puts("FAIL: WARP readback through the repaired MipLevels 0 view and the UAV-token view");
        return 1;
    }
    std::puts("PASS: WARP created the repaired color and stencil SRVs and the null SRV, and sampled every mip");
    return 0;
}

int main(int argc, char** argv)
{
    if (!buffer_checks() || !alias_checks() || !pure_checks() || !uav_record_checks()) {
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
