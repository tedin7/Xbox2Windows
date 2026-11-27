#include "startup_stubs/target_view_check.h"
#include <cstdio>
#include <cstring>
using namespace x2w_startup::target_view;
int main() {
    int failures = 0;
    auto expect = [&](bool condition, const char* name) { if (!condition) { std::printf("FAIL: %s\n", name); ++failures; } };
    auto repair = [](const char*, UINT64, UINT64) {};
    D3D12_RESOURCE_DESC r{}; r.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; r.Width=32; r.Height=32; r.DepthOrArraySize=2; r.MipLevels=4; r.Format=DXGI_FORMAT_R8G8B8A8_UNORM; r.SampleDesc.Count=1;
    D3D12_RENDER_TARGET_VIEW_DESC v{}; v.Format=r.Format; v.ViewDimension=D3D12_RTV_DIMENSION_TEXTURE2DARRAY; v.Texture2DArray.ArraySize=2;
    auto original=v;
    expect(check(&v, &r, repair)==Refusal::none && !std::memcmp(&v,&original,sizeof(v)), "valid unchanged");
    v.Texture2DArray.ArraySize=5; expect(check(&v,&r,repair)==Refusal::none && v.Texture2DArray.ArraySize==2, "array clamp");
    v.Texture2DArray.FirstArraySlice=2; expect(check(&v,&r,repair)==Refusal::array, "first slice");
    v=original; v.Texture2DArray.MipSlice=9; expect(check(&v,&r,repair)==Refusal::mip, "mip");
    v=original; v.Texture2DArray.PlaneSlice=7; expect(check(&v,&r,repair)==Refusal::none && v.Texture2DArray.PlaneSlice==0, "plane");
    v=original; v.Format=DXGI_FORMAT_R32_FLOAT; expect(check(&v,&r,repair)==Refusal::none && v.Format==r.Format, "format repair");
    v=original; r.SampleDesc.Count=4; expect(check(&v,&r,repair)==Refusal::dimension, "MS mismatch"); r.SampleDesc.Count=1;
    expect(check<D3D12_RENDER_TARGET_VIEW_DESC>(nullptr,nullptr,repair)==Refusal::null_desc, "null description");
    v={}; v.Format=r.Format; expect(check(&v,nullptr,repair)==Refusal::dimension, "null dimension");
    D3D12_UNORDERED_ACCESS_VIEW_DESC u{}; u.Format=DXGI_FORMAT_UNKNOWN; u.ViewDimension=D3D12_UAV_DIMENSION_BUFFER; u.Buffer.StructureByteStride=4; u.Buffer.CounterOffsetInBytes=100;
    D3D12_RESOURCE_DESC b{}; b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; b.Width=8192; b.SampleDesc.Count=1;
    expect(check_uav(&u,&b,nullptr,repair)==Refusal::none && u.Buffer.CounterOffsetInBytes==0, "counter reset");
    u.Buffer.CounterOffsetInBytes=100; expect(check_uav(&u,&b,&b,repair)==Refusal::counter, "counter alignment");
    u.Buffer.CounterOffsetInBytes=4096; auto saved=u; expect(check_uav(&u,&b,&b,repair)==Refusal::none && !std::memcmp(&u,&saved,sizeof(u)), "counter valid");
    D3D12_DEPTH_STENCIL_VIEW_DESC d{}; d.Format=DXGI_FORMAT_D32_FLOAT; d.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2D; d.Flags=D3D12_DSV_FLAG_READ_ONLY_STENCIL; r.Format=d.Format;
    expect(check(&d,&r,repair)==Refusal::none && d.Flags==D3D12_DSV_FLAG_NONE, "stencil flag");
    d.Format=DXGI_FORMAT_R32_FLOAT; expect(check(&d,&r,repair)==Refusal::format, "depth format");
    d={}; d.Format=DXGI_FORMAT_D24_UNORM_S8_UINT; d.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2D; d.Flags=D3D12_DSV_FLAG_READ_ONLY_STENCIL; r.Format=d.Format;
    auto saved_d=d; expect(check(&d,&r,repair)==Refusal::none && !std::memcmp(&d,&saved_d,sizeof(d)), "stencil valid unchanged");
    d.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2DMS; r.SampleDesc.Count=4; saved_d=d;
    expect(check(&d,&r,repair)==Refusal::none && !std::memcmp(&d,&saved_d,sizeof(d)), "MS depth unchanged");
    r.SampleDesc.Count=1; expect(check(&d,&r,repair)==Refusal::dimension, "MS on single sample");
    r.Format=DXGI_FORMAT_R8G8B8A8_UNORM; r.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE3D; r.DepthOrArraySize=8;
    v={}; v.Format=r.Format; v.ViewDimension=D3D12_RTV_DIMENSION_TEXTURE3D; v.Texture3D={1,1,~0u}; original=v;
    expect(check(&v,&r,repair)==Refusal::none && !std::memcmp(&v,&original,sizeof(v)), "volume all remaining unchanged");
    v.Texture3D.WSize=8; expect(check(&v,&r,repair)==Refusal::none && v.Texture3D.WSize==3, "volume clamp");
    v.Texture3D.FirstWSlice=4; expect(check(&v,&r,repair)==Refusal::array, "volume first slice");
    r.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; r.SampleDesc.Count=4;
    u={}; u.Format=r.Format; u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
    expect(check_uav(&u,&r,nullptr,repair)==Refusal::dimension, "multisample UAV");
    r.SampleDesc.Count=1; auto saved_u=u; expect(check_uav(&u,&r,nullptr,repair)==Refusal::none && !std::memcmp(&u,&saved_u,sizeof(u)), "UAV unchanged");
    u.Format=DXGI_FORMAT_R8G8B8A8_TYPELESS; r.Format=DXGI_FORMAT_R8G8B8A8_TYPELESS;
    expect(check_uav(&u,&r,nullptr,repair)==Refusal::format, "typeless refusal");
    r.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
    expect(check_uav(&u,&r,nullptr,repair)==Refusal::none && u.Format==r.Format, "typed repair");
    u={}; u.ViewDimension=D3D12_UAV_DIMENSION_BUFFER; u.Buffer.StructureByteStride=4; u.Buffer.CounterOffsetInBytes=8192;
    expect(check_uav(&u,&b,&b,repair)==Refusal::counter, "counter outside");
    u.Buffer.CounterOffsetInBytes=4096; u.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_RAW;
    expect(check_uav(&u,&b,&b,repair)==Refusal::counter, "raw counter");
    u.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_NONE; u.Buffer.StructureByteStride=0;
    expect(check_uav(&u,&b,&b,repair)==Refusal::counter, "unstructured counter");
    expect(check_uav(&u,nullptr,&b,repair)==Refusal::counter, "counter without resource");
    expect(check<D3D12_DEPTH_STENCIL_VIEW_DESC>(nullptr,&r,repair)==Refusal::format, "default depth format");
    return failures ? 1 : 0;
}
