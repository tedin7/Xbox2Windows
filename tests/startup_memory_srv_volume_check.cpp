// Checks validation, view fitting, and byte sizing for volume (Texture3D) memory SRVs (device slot 70) without a device.
// --mutate layout: expect layout 0x105 to be accepted.
// --mutate 2d-view: expect a Texture2D view on a 3D resource to be accepted.
// --mutate 3d-view: expect a Texture3D view on a 2D resource to be accepted.
// --mutate mips-overflow: expect a view specifying more mips than available to be accepted.
// Each mutated run is a negative control and must fail.
#include "startup_stubs/address_texture.h"

#include <cstdio>
#include <cstring>

namespace {

int failures;
const char* mutation = "";

bool mutated(const char* name)
{
    return std::strcmp(mutation, name) == 0;
}

void check(const char* name, bool condition)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL %s\n", name);
}

D3D12_RESOURCE_DESC tex3d(UINT64 width, UINT height, UINT16 depth, UINT16 mips, DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
                          UINT layout = 0)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = depth;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(layout);
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    return desc;
}

D3D12_SHADER_RESOURCE_VIEW_DESC view3d(DXGI_FORMAT format, UINT mostmip, UINT mips, float clamp = 0.0f)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = format;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Texture3D.MostDetailedMip = mostmip;
    v.Texture3D.MipLevels = mips;
    v.Texture3D.ResourceMinLODClamp = clamp;
    return v;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0 &&
        (std::strcmp(argv[2], "layout") == 0 || std::strcmp(argv[2], "2d-view") == 0 ||
         std::strcmp(argv[2], "3d-view") == 0 || std::strcmp(argv[2], "mips-overflow") == 0)) {
        mutation = argv[2];
    } else if (argc != 1) {
        std::printf("usage: startup_memory_srv_volume_check [--mutate layout|2d-view|3d-view|mips-overflow]\n");
        return 2;
    }

    // Base valid 3D resource and view.
    const auto res = tex3d(32, 32, 16, 4, DXGI_FORMAT_R8G8B8A8_UNORM, 0);
    const auto v_all = view3d(DXGI_FORMAT_R8G8B8A8_UNORM, 0, ~0u);
    const auto v_exact = view3d(DXGI_FORMAT_R8G8B8A8_UNORM, 0, 4);
    const auto v_sub = view3d(DXGI_FORMAT_R8G8B8A8_UNORM, 1, 2);
    const auto v_last = view3d(DXGI_FORMAT_R8G8B8A8_UNORM, 3, 1);

    check("all mips accepted", x2w_startup::supports_volume_memory_srv(res, v_all));
    check("exact mips accepted", x2w_startup::supports_volume_memory_srv(res, v_exact));
    check("sub-mip range accepted", x2w_startup::supports_volume_memory_srv(res, v_sub));
    check("last mip accepted", x2w_startup::supports_volume_memory_srv(res, v_last));

    // View dimension validation.
    auto v_2d = v_exact;
    v_2d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v_2d.Texture2D.MostDetailedMip = 0;
    v_2d.Texture2D.MipLevels = 4;
    v_2d.Texture2D.PlaneSlice = 0;
    v_2d.Texture2D.ResourceMinLODClamp = 0.0f;
    check("2D view on 3D resource refused", x2w_startup::supports_volume_memory_srv(res, v_2d) == mutated("2d-view"));

    auto v_2darr = v_exact;
    v_2darr.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    v_2darr.Texture2DArray.MostDetailedMip = 0;
    v_2darr.Texture2DArray.MipLevels = 4;
    v_2darr.Texture2DArray.FirstArraySlice = 0;
    v_2darr.Texture2DArray.ArraySize = 16;
    check("2D array view on 3D resource refused", !x2w_startup::supports_volume_memory_srv(res, v_2darr));

    // Resource dimension validation.
    auto res_2d = res;
    res_2d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    check("3D view on 2D resource refused", x2w_startup::supports_volume_memory_srv(res_2d, v_exact) == mutated("3d-view"));

    // Layout pinning: layout must be D3D12_TEXTURE_LAYOUT_UNKNOWN.
    auto res_swizzle = res;
    res_swizzle.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x105);
    check("console layout 0x105 refused", x2w_startup::supports_volume_memory_srv(res_swizzle, v_exact) == mutated("layout"));

    auto res_linear = res;
    res_linear.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x100);
    check("console linear layout 0x100 refused", !x2w_startup::supports_volume_memory_srv(res_linear, v_exact));

    // Mip bounds.
    auto v_overflow = view3d(DXGI_FORMAT_R8G8B8A8_UNORM, 1, 4);
    check("mips exceeding chain refused", x2w_startup::supports_volume_memory_srv(res, v_overflow) == mutated("mips-overflow"));

    auto v_first_past = view3d(DXGI_FORMAT_R8G8B8A8_UNORM, 4, 1);
    check("most detailed mip at or past count refused", !x2w_startup::supports_volume_memory_srv(res, v_first_past));

    auto v_zero_mips = view3d(DXGI_FORMAT_R8G8B8A8_UNORM, 0, 0);
    check("zero mips refused", !x2w_startup::supports_volume_memory_srv(res, v_zero_mips));

    // Explicit LOD clamp.
    auto v_clamp = view3d(DXGI_FORMAT_R8G8B8A8_UNORM, 0, 4, 1.5f);
    check("explicit LOD clamp refused", !x2w_startup::supports_volume_memory_srv(res, v_clamp));

    // Format compatibility and mapping.
    auto v_bad_fmt = view3d(DXGI_FORMAT_R8_UNORM, 0, 4);
    check("format mismatch refused", !x2w_startup::supports_volume_memory_srv(res, v_bad_fmt));

    auto v_bad_map = v_exact;
    v_bad_map.Shader4ComponentMapping = 0;
    check("invalid component mapping refused", !x2w_startup::supports_volume_memory_srv(res, v_bad_map));

    // Block-compressed 3D textures (BC3, BC7).
    const auto bc7_res = tex3d(16, 16, 4, 1, DXGI_FORMAT_BC7_UNORM, 0);
    const auto bc7_view = view3d(DXGI_FORMAT_BC7_UNORM, 0, 1);
    check("BC7 volume accepted", x2w_startup::supports_volume_memory_srv(bc7_res, bc7_view));
    // 4x4 blocks * 4 slices * 16 bytes = 1024 bytes.
    check("BC7 volume byte count", x2w_startup::volume_texture_bytes(bc7_res) == 1024);

    const auto bc3_res = tex3d(8, 8, 2, 2, DXGI_FORMAT_BC3_UNORM, 0);
    const auto bc3_view = view3d(DXGI_FORMAT_BC3_UNORM, 0, 2);
    check("BC3 multi-mip volume accepted", x2w_startup::supports_volume_memory_srv(bc3_res, bc3_view));
    // mip 0: 2x2 blocks * 2 slices * 16 = 128 bytes; mip 1: 1x1 blocks * 1 slice * 16 = 16 bytes. Total = 144 bytes.
    check("BC3 multi-mip volume byte count", x2w_startup::volume_texture_bytes(bc3_res) == 144);

    const auto bc7_sub4 = tex3d(16, 16, 4, 4, DXGI_FORMAT_BC7_UNORM, 0);
    const auto bc7_sub4_view = view3d(DXGI_FORMAT_BC7_UNORM, 0, 4);
    check("BC7 4-mip volume with sub-4x4 tail accepted", x2w_startup::supports_volume_memory_srv(bc7_sub4, bc7_sub4_view));
    // mip 0 (16x16x4): 4x4 blocks * 4 slices * 16 = 1024 bytes
    // mip 1 (8x8x2):   2x2 blocks * 2 slices * 16 = 128 bytes
    // mip 2 (4x4x1):   1x1 blocks * 1 slice * 16 = 16 bytes
    // mip 3 (2x2x1, sub-block): 1x1 blocks * 1 slice * 16 = 16 bytes. Total = 1184 bytes.
    check("BC7 4-mip sub-block volume byte count", x2w_startup::volume_texture_bytes(bc7_sub4) == 1184);

    auto bc7_bad_w = bc7_res;
    bc7_bad_w.Width = 14;
    check("BC7 width not multiple of 4 refused", !x2w_startup::supports_volume_memory_srv(bc7_bad_w, bc7_view));

    auto bc7_bad_h = bc7_res;
    bc7_bad_h.Height = 14;
    check("BC7 height not multiple of 4 refused", !x2w_startup::supports_volume_memory_srv(bc7_bad_h, bc7_view));

    check("depth format refused", !x2w_startup::supports_volume_memory_srv(tex3d(4, 4, 4, 1, DXGI_FORMAT_D24_UNORM_S8_UINT), view3d(DXGI_FORMAT_D24_UNORM_S8_UINT, 0, 1)));

    // Resource shape fields.
    auto res_w0 = res; res_w0.Width = 0;
    check("zero width refused", !x2w_startup::supports_volume_memory_srv(res_w0, v_exact));
    auto res_h0 = res; res_h0.Height = 0;
    check("zero height refused", !x2w_startup::supports_volume_memory_srv(res_h0, v_exact));
    auto res_d0 = res; res_d0.DepthOrArraySize = 0;
    check("zero depth refused", !x2w_startup::supports_volume_memory_srv(res_d0, v_exact));
    auto res_m0 = res; res_m0.MipLevels = 0;
    check("zero mips refused", !x2w_startup::supports_volume_memory_srv(res_m0, v_exact));

    // Deny SRV flag.
    auto res_deny = res;
    res_deny.Flags = D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    check("deny shader resource flag refused", !x2w_startup::supports_volume_memory_srv(res_deny, v_exact));

    // Multisample.
    auto res_ms = res;
    res_ms.SampleDesc.Count = 2;
    check("multisampled volume refused", !x2w_startup::supports_volume_memory_srv(res_ms, v_exact));

    // Byte count calculations.
    check("single-mip byte count", x2w_startup::volume_texture_bytes(tex3d(4, 4, 4, 1, DXGI_FORMAT_R8G8B8A8_UNORM)) == 256);
    // 4x4x2 (32 texels) + 2x2x1 (4 texels) = 36 texels * 4 = 144 bytes.
    check("multi-mip byte count", x2w_startup::volume_texture_bytes(tex3d(4, 4, 2, 2, DXGI_FORMAT_R8G8B8A8_UNORM)) == 144);
    // 32x32x32 R8_UNORM = 32768 bytes.
    check("R8 volume byte count", x2w_startup::volume_texture_bytes(tex3d(32, 32, 32, 1, DXGI_FORMAT_R8_UNORM)) == 32768);
    // Non-3D resource.
    check("2D resource byte count is 0", x2w_startup::volume_texture_bytes(res_2d) == 0);

    if (failures) {
        std::printf("startup_memory_srv_volume_check: %d FAILED\n", failures);
        return 1;
    }
    std::printf("startup_memory_srv_volume_check: all checks passed\n");
    return 0;
}
