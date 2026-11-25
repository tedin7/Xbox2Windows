// Device check for src/startup_stubs/small_texture_4k.h: both allocation-info entries of the startup device
// wrapper (slots 25 and 108) give the 4 KiB answer for a small 2D texture asked with a 4 KiB alignment and
// the desktop answer otherwise. usage: startup_small_texture_4k_device_check warp|hardware
#include "xmem_test_device.h"
#include "../src/startup_stubs/small_texture_4k.h"

#include <cstring>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
HRESULT create_wrapper(IUnknown* adapter, REFIID iid, void** out)
{
    unsigned char parameters[64]{};
    return D3D12XboxCreateDevice(adapter, parameters, iid, out);
}

using Info25 = void* (*)(void*, D3D12_RESOURCE_ALLOCATION_INFO*, UINT, UINT, const D3D12_RESOURCE_DESC*);
using Info108 = void* (*)(void*, D3D12_RESOURCE_ALLOCATION_INFO*, UINT, UINT, const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_ALLOCATION_INFO1*);

template <class F>
F slot(ID3D12Device* device, int index)
{
    return reinterpret_cast<F>((*reinterpret_cast<void***>(device))[index]);
}

D3D12_RESOURCE_DESC texture(DXGI_FORMAT format, UINT width, UINT height, UINT16 mips, UINT64 alignment, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = alignment;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

// expect_4k: the wrapper must give the helper's 4 KiB answer; otherwise the desktop answer for the same
// description at the default alignment, which is what the wrapper forwards.
int check(const char* name, ID3D12Device* desktop, ID3D12Device* wrapper, const D3D12_RESOURCE_DESC& desc, bool expect_4k)
{
    D3D12_RESOURCE_DESC as_default = desc;
    as_default.Alignment = 0;
    D3D12_RESOURCE_ALLOCATION_INFO expected = desktop->GetResourceAllocationInfo(0, 1, &as_default);
    if (x2w_startup::small_texture_4k(desc, expected) != expect_4k) {
        std::printf("FAIL %s: the helper %s the description\n", name, expect_4k ? "refused" : "accepted");
        return 1;
    }
    D3D12_RESOURCE_ALLOCATION_INFO by25{1, 1}, by108{1, 1};
    D3D12_RESOURCE_ALLOCATION_INFO1 each{1, 1, 1};
    slot<Info25>(wrapper, 25)(wrapper, &by25, 0, 1, &desc);
    slot<Info108>(wrapper, 108)(wrapper, &by108, 0, 1, &desc, &each);
    std::printf("%s: slot 25 size 0x%llx alignment 0x%llx, slot 108 size 0x%llx alignment 0x%llx, expected 0x%llx at 0x%llx\n", name,
                static_cast<unsigned long long>(by25.SizeInBytes), static_cast<unsigned long long>(by25.Alignment),
                static_cast<unsigned long long>(by108.SizeInBytes), static_cast<unsigned long long>(by108.Alignment),
                static_cast<unsigned long long>(expected.SizeInBytes), static_cast<unsigned long long>(expected.Alignment));
    const auto same = [&](const D3D12_RESOURCE_ALLOCATION_INFO& a) {
        return a.SizeInBytes == expected.SizeInBytes && a.Alignment == expected.Alignment;
    };
    if (!same(by25) || !same(by108)) {
        std::printf("FAIL %s: slot 25 or slot 108 differs from the expected answer\n", name);
        return 1;
    }
    if (expect_4k && (each.Offset != 0 || each.SizeInBytes != expected.SizeInBytes || each.Alignment != 0x1000)) {
        std::printf("FAIL %s: slot 108 per-resource entry does not describe the 4 KiB answer\n", name);
        return 1;
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 1;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    bool unmatched = false;
    auto desktop = create_device(warp, unmatched);
    if (!desktop) return warp || unmatched ? 1 : 77;
    SetEnvironmentVariableW(L"X2W_D3D12_PERMISSIVE", L"1");
    auto wrapper = create_device(warp, unmatched, create_wrapper);
    if (!wrapper) return 1;

    const auto none = D3D12_RESOURCE_FLAG_NONE;
    int failures = 0;
    failures += check("32x16 R8G8B8A8", desktop.Get(), wrapper.Get(), texture(DXGI_FORMAT_R8G8B8A8_UNORM, 32, 16, 1, 0x1000, none), true);
    failures += check("128x4 R16G16B16A16", desktop.Get(), wrapper.Get(),
                      texture(DXGI_FORMAT_R16G16B16A16_UNORM, 128, 4, 1, 0x1000, none), true);
    failures += check("128x128 BC1 6 mips", desktop.Get(), wrapper.Get(), texture(DXGI_FORMAT_BC1_UNORM, 128, 128, 6, 0x1000, none), true);
    // Controls: the desktop answer stays.
    failures += check("control alignment 0", desktop.Get(), wrapper.Get(), texture(DXGI_FORMAT_R8G8B8A8_UNORM, 32, 16, 1, 0, none), false);
    failures += check("control render target", desktop.Get(), wrapper.Get(),
                      texture(DXGI_FORMAT_R8G8B8A8_UNORM, 32, 16, 1, 0x1000, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), false);
    failures += check("control 256x128 R8G8B8A8", desktop.Get(), wrapper.Get(),
                      texture(DXGI_FORMAT_R8G8B8A8_UNORM, 256, 128, 1, 0x1000, none), false);
    if (failures) return 1;
    std::printf("startup_small_texture_4k_device: PASS (3 descriptions answered with 4 KiB, 3 controls unchanged)\n");
    return 0;
}
