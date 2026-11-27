#ifdef X2W_BUILD_NOOP_CENSUS_FIXTURE
#include "startup_noop_census_fixture.h"

#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>

extern "C" __declspec(dllimport) void x2w_d3d12_noop_census_mutant(BOOL on);
extern "C" __declspec(dllimport) UINT64 x2w_d3d12_noop_census_list_calls(UINT slot);

namespace {
using Word = std::uint64_t;
template<class T> Word arg(T* p) { return reinterpret_cast<Word>(p); }

Word invoke(void* object, int slot, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0, Word g = 0)
{
    using Slot = Word(__stdcall*)(void*, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[slot])(object, a, b, c, d, e, f, g);
}
}  // namespace

int record_noop_census_fixture(void* device, void* list, void* pipeline)
{
    using Microsoft::WRL::ComPtr;
    ComPtr<IUnknown> buffer, heap;
    D3D12_HEAP_PROPERTIES props{};
    props.Type = D3D12_HEAP_TYPE_DEFAULT;
    props.CreationNodeMask = props.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 256;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(static_cast<HRESULT>(invoke(device, 76, arg(&props), 0, arg(&desc), D3D12_RESOURCE_STATE_COMMON, 0,
                                           arg(&__uuidof(ID3D12Resource)), arg(buffer.GetAddressOf()))))) return 81;
    struct ExtendedHeapDesc {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word title_cpu, title_bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1}, 0, 0};
    if (FAILED(static_cast<HRESULT>(invoke(device, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)),
                                           arg(heap.GetAddressOf()))))) return 82;
    const Word token = invoke(heap.Get(), 9);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_UINT;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = 16;
    invoke(device, 18, arg(buffer.Get()), arg(&srv), token);
    const Word upload = invoke(list, 148, 128, 16);
    if (!token || !upload) return 83;

    unsigned char sample[32];
    for (unsigned i = 0; i < sizeof(sample); ++i) sample[i] = static_cast<unsigned char>(71 + 3 * i);
    void* inaccessible = VirtualAlloc(nullptr, 4096, MEM_RESERVE, PAGE_NOACCESS);
    if (!inaccessible) return 84;
    x2w_d3d12_noop_census_mutant(GetEnvironmentVariableW(L"X2W_TEST_NOOP_CENSUS_MUTANT", nullptr, 0) != 0);
    for (unsigned i = 1; i <= 13; ++i) invoke(list, census_overflow_test_slot, i);
    invoke(list, 25, arg(pipeline));
    for (unsigned i = 0; i < 5; ++i) {
        invoke(list, census_test_slot, arg(buffer.Get()), 11, 0, arg(sample), 0);
        ++sample[0];
        invoke(list, 12, 3, 1);
    }
    for (unsigned i = 0; i < 3; ++i) {
        invoke(list, census_test_slot, token, upload + 16, 0, 0, 29);
        invoke(list, 14, 1, 1, 1);
    }
    invoke(list, census_test_slot, 0, 0, 0, 0, arg(inaccessible));
    VirtualFree(inaccessible, 0, MEM_RELEASE);
    invoke(device, 61, 23);
    x2w_d3d12_noop_census_mutant(FALSE);
    return x2w_d3d12_noop_census_list_calls(census_test_slot) == 9 &&
           x2w_d3d12_noop_census_list_calls(census_overflow_test_slot) == 13 ? 0 : 85;
}
#else
// Input for the startup debugger's call-hook check (tests/startup_debug_check.py). The check copies
// this DLL next to a copy of tests/startup_debug_probe.cpp under the movie-player module name that
// tools/startup/debug_launch.py hooks with --trace-calls 1, so the export below is one of its call
// hooks. The export only counts its calls; no other input is involved.
#include <windows.h>

namespace {
volatile LONG calls;
}  // namespace

extern "C" __declspec(dllexport) LONG BinkNextFrame(void*)
{
    return InterlockedIncrement(&calls);
}
#endif
