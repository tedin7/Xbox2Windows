// Checks the diagnostic line of src/startup_stubs/memory_srv_report.h without a device. The expected
// lines are written out by hand: they are what a tester pastes into an issue, field by field.
#include "startup_stubs/memory_srv_report.h"

#include <cstdio>
#include <cstring>

namespace {
int failures;

void expect(const char* name, const char* actual, const char* expected)
{
    if (std::strcmp(actual, expected) == 0) return;
    ++failures;
    std::printf("FAIL %s\n  actual   %s\n  expected %s\n", name, actual, expected);
}

D3D12_SHADER_RESOURCE_VIEW_DESC view(D3D12_SRV_DIMENSION dimension, DXGI_FORMAT format)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = format;
    v.ViewDimension = dimension;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    return v;
}

int line(char (&out)[640], const char* reason, const D3D12_RESOURCE_DESC* r, const D3D12_SHADER_RESOURCE_VIEW_DESC* v,
         std::uint64_t address, bool readable, std::uint64_t destination, bool in_heap)
{
    return x2w_startup::report_refused_memory_srv(out, sizeof(out), "unsupported memory SRV arguments", reason, r, v,
                                                  address, readable, destination, in_heap);
}
}

int main()
{
    char out[640];
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = 4096;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    // A view dimension that has no named fields here is printed as its six raw words.
    auto as = view(static_cast<D3D12_SRV_DIMENSION>(11), DXGI_FORMAT_UNKNOWN);
    as.Buffer.FirstElement = 0x1122334455667788ull;
    line(out, "unsupported_description", &buffer, &as, 0x2000, false, 0x1000, true);
    expect("acceleration structure view", out,
           "x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description "
           "res{dim=1 fmt=0 size=4096x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0} "
           "view{dim=11 fmt=0 map=0x1688 raw=55667788,11223344,00000000,00000000,00000000,00000000 "
           "(acceleration structure view)} dest=0x1000 dest_in_view_heap=1 address=0x2000 address_readable=0");

    auto b = view(D3D12_SRV_DIMENSION_BUFFER, DXGI_FORMAT_UNKNOWN);
    b.Buffer.FirstElement = 7;
    b.Buffer.NumElements = 64;
    b.Buffer.StructureByteStride = 16;
    b.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    line(out, "unsupported_description", &buffer, &b, 0x2000, true, 0x1000, true);
    expect("buffer view", out,
           "x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description "
           "res{dim=1 fmt=0 size=4096x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0} "
           "view{dim=1 fmt=0 map=0x1688 first=7 count=64 stride=16 flags=0x1 (buffer views are not supported)} "
           "dest=0x1000 dest_in_view_heap=1 address=0x2000 address_readable=1");

    D3D12_RESOURCE_DESC bc7{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 65536, 128, 128, 2, 6, DXGI_FORMAT_BC7_UNORM, {1, 0},
                            static_cast<D3D12_TEXTURE_LAYOUT>(0x105), static_cast<D3D12_RESOURCE_FLAGS>(0x300000)};
    auto array = view(D3D12_SRV_DIMENSION_TEXTURE2DARRAY, DXGI_FORMAT_BC7_UNORM);
    array.Texture2DArray.MostDetailedMip = 1;
    array.Texture2DArray.MipLevels = 5;
    array.Texture2DArray.FirstArraySlice = 1;
    array.Texture2DArray.ArraySize = 1;
    array.Texture2DArray.ResourceMinLODClamp = 0.5f;
    line(out, "destination_outside_view_heaps", &bc7, &array, 0x7000, true, 0x1001, false);
    expect("texture array view", out,
           "x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=destination_outside_view_heaps "
           "res{dim=3 fmt=98 size=128x128x2 mips=6 layout=0x105 flags=0x300000 align=65536 samples=1/0} "
           "view{dim=5 fmt=98 map=0x1688 mostmip=1 mips=5 first=1 count=1 plane=0 clamp=0.5} "
           "dest=0x1001 dest_in_view_heap=0 address=0x7000 address_readable=1");

    // The same fields with a distinct value each, so that two fields swapped or a first slice
    // printed as the count shows up.
    auto slices = view(D3D12_SRV_DIMENSION_TEXTURE2DARRAY, DXGI_FORMAT_BC7_UNORM);
    slices.Texture2DArray.MostDetailedMip = 4;
    slices.Texture2DArray.MipLevels = 2;
    slices.Texture2DArray.FirstArraySlice = 0;
    slices.Texture2DArray.ArraySize = 1;
    slices.Texture2DArray.PlaneSlice = 3;
    slices.Texture2DArray.ResourceMinLODClamp = 2.25f;
    line(out, "unsupported_description", &bc7, &slices, 0x7000, true, 0x1040, true);
    expect("texture array view, distinct fields", out,
           "x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description "
           "res{dim=3 fmt=98 size=128x128x2 mips=6 layout=0x105 flags=0x300000 align=65536 samples=1/0} "
           "view{dim=5 fmt=98 map=0x1688 mostmip=4 mips=2 first=0 count=1 plane=3 clamp=2.25} "
           "dest=0x1040 dest_in_view_heap=1 address=0x7000 address_readable=1");

    // A 2D view with distinct mip and plane fields and a fractional clamp, at an address and a
    // destination above 4 GiB.
    D3D12_RESOURCE_DESC r8{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 256, 64, 1, 4, DXGI_FORMAT_R8_UNORM, {1, 0},
                           D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    auto plain = view(D3D12_SRV_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8_UNORM);
    plain.Texture2D.MostDetailedMip = 1;
    plain.Texture2D.MipLevels = 2;
    plain.Texture2D.PlaneSlice = 3;
    plain.Texture2D.ResourceMinLODClamp = 1.5f;
    line(out, "unsupported_description", &r8, &plain, 0x200001000ull, true, 0x123456780ull, true);
    expect("texture 2D view", out,
           "x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description "
           "res{dim=3 fmt=61 size=256x64x1 mips=4 layout=0x0 flags=0x0 align=0 samples=1/0} "
           "view{dim=4 fmt=61 map=0x1688 mostmip=1 mips=2 plane=3 clamp=1.5} "
           "dest=0x123456780 dest_in_view_heap=1 address=0x200001000 address_readable=1");

    // Another unnamed dimension: the five fields of a cube-array view as raw words, without the
    // acceleration-structure note.
    auto cubes = view(D3D12_SRV_DIMENSION_TEXTURECUBEARRAY, DXGI_FORMAT_R8_UNORM);
    cubes.TextureCubeArray.MostDetailedMip = 1;
    cubes.TextureCubeArray.MipLevels = 2;
    cubes.TextureCubeArray.First2DArrayFace = 6;
    cubes.TextureCubeArray.NumCubes = 2;
    cubes.TextureCubeArray.ResourceMinLODClamp = 0.5f;
    line(out, "unsupported_description", &r8, &cubes, 0x2000, true, 0x1000, true);
    expect("cube array view", out,
           "x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description "
           "res{dim=3 fmt=61 size=256x64x1 mips=4 layout=0x0 flags=0x0 align=0 samples=1/0} "
           "view{dim=10 fmt=61 map=0x1688 raw=00000001,00000002,00000006,00000002,3f000000,00000000} "
           "dest=0x1000 dest_in_view_heap=1 address=0x2000 address_readable=1");

    // A buffer wider than 4 GiB and a first element past 2^32 are printed in full.
    D3D12_RESOURCE_DESC wide = buffer;
    wide.Width = 0x100001000ull;
    auto raw = view(D3D12_SRV_DIMENSION_BUFFER, DXGI_FORMAT_R32_TYPELESS);
    raw.Buffer.FirstElement = 0x100000008ull;
    raw.Buffer.NumElements = 256;
    raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    line(out, "unsupported_description", &wide, &raw, 0x2000, true, 0x1000, true);
    expect("wide buffer view", out,
           "x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description "
           "res{dim=1 fmt=0 size=4294971392x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0} "
           "view{dim=1 fmt=39 map=0x1688 first=4294967304 count=256 stride=0 flags=0x1 (buffer views are not supported)} "
           "dest=0x1000 dest_in_view_heap=1 address=0x2000 address_readable=1");

    // Memory the guest handed over but the runtime could not read.
    line(out, "null_destination", nullptr, nullptr, 0, false, 0, false);
    expect("unreadable arguments", out,
           "x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=null_destination "
           "res{unreadable} view{unreadable} dest=0x0 dest_in_view_heap=0 address=0x0 address_readable=0");

    // A 3D texture view.
    D3D12_RESOURCE_DESC tex3d{D3D12_RESOURCE_DIMENSION_TEXTURE3D, 0, 32, 32, 32, 1, DXGI_FORMAT_R8_UNORM, {1, 0},
                             D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    auto v3d = view(D3D12_SRV_DIMENSION_TEXTURE3D, DXGI_FORMAT_R8_UNORM);
    v3d.Texture3D.MostDetailedMip = 0;
    v3d.Texture3D.MipLevels = 1;
    v3d.Texture3D.ResourceMinLODClamp = 0.0f;
    line(out, "destination_outside_view_heaps", &tex3d, &v3d, 0x3000, true, 0x1000, false);
    expect("texture 3D view", out,
           "x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=destination_outside_view_heaps "
           "res{dim=4 fmt=61 size=32x32x32 mips=1 layout=0x0 flags=0x0 align=0 samples=1/0} "
           "view{dim=8 fmt=61 map=0x1688 mostmip=0 mips=1 clamp=0} "
           "dest=0x1000 dest_in_view_heap=0 address=0x3000 address_readable=1");

    // A small buffer is cut, never overrun, and stays terminated.
    char tiny_buf[40];
    std::memset(tiny_buf, 'x', sizeof(tiny_buf));
    x2w_startup::report_refused_memory_srv(tiny_buf, sizeof(tiny_buf), "unsupported memory SRV arguments", "null_address", &bc7,
                                           &array, 0, false, 0x1000, true);
    if (std::memchr(tiny_buf, 0, sizeof(tiny_buf)) == nullptr) {
        ++failures;
        std::printf("FAIL truncated line is not terminated\n");
    }
    if (failures) return 1;
    std::printf("slot 70 refusal lines: 9 lines and a truncation exact\n");
    return 0;
}
