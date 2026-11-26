// SOSetTargets through the production command list vtable (slot 49), recording only: draws do not write
// stream output yet. The slot is inferred from the desktop order (45 + 4), so it is opt-in.
//
//   startup_stream_output_slot <warp|hardware> <on|off>
//   (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1; X2W_D3D12_SO_TARGETS=1 for "on")
//
// "on": a call through list slot 49 binds slots 1 and 2, a partial unbind (null array) clears only slot 2,
// and rejected ranges (start past the end, count past the end, start + count overflowing) leave the retained
// state exactly as it was. "off": the same calls record nothing.
//
// Exit codes: 0 pass, 77 no physical adapter, other values name the failing step.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) int x2w_d3d12_stream_output_slot(void* list, unsigned slot, unsigned long long* out);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0, Word g = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f, g);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

// What a list holds in the four slots: 0 empty, else the BufferLocation.
struct Slots {
    unsigned long long location[4] = {};
    unsigned long long size[4] = {};
    unsigned long long filled[4] = {};
    bool read(void* list)
    {
        for (unsigned i = 0; i < 4; ++i) {
            unsigned long long v[3] = {};
            if (x2w_d3d12_stream_output_slot(list, i, v) < 0) return false;
            location[i] = v[0], size[i] = v[1], filled[i] = v[2];
        }
        return true;
    }
    bool same(const Slots& o) const { return !std::memcmp(this, &o, sizeof(*this)); }
};
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    const bool on = std::strcmp(argv[2], "on") == 0;
    if (!on && std::strcmp(argv[2], "off") != 0) return 2;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(argv[1], "hardware") == 0) {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    } else if (std::strcmp(argv[1], "warp") != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 3;
    }

    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 4;
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void *allocator = nullptr, *list = nullptr;
    if (!ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))))
        return 5;

    const D3D12_STREAM_OUTPUT_BUFFER_VIEW two[2] = {{0x1000, 256, 0x2000}, {0x3000, 512, 0x4000}};
    Slots now, expected;
    if (!now.read(list) || !now.same(expected)) return 6;  // a new list holds nothing

    // 1. bind slots 1 and 2.
    invoke(list, 49, 1, 2, arg(two));
    if (!now.read(list)) return 7;
    if (on) {
        expected.location[1] = 0x1000, expected.size[1] = 256, expected.filled[1] = 0x2000;
        expected.location[2] = 0x3000, expected.size[2] = 512, expected.filled[2] = 0x4000;
    }
    if (!now.same(expected)) return 8;

    // 2. rejected ranges leave the retained state as it was.
    invoke(list, 49, 3, 2, arg(two));
    invoke(list, 49, 4, 1, arg(two));
    invoke(list, 49, 0, 5, arg(two));
    invoke(list, 49, 0xffffffffu, 2, arg(two));
    invoke(list, 49, 2, 0xffffffffu, arg(two));
    if (!now.read(list) || !now.same(expected)) return 9;

    // 3. partial unbind: no array clears only the named slot.
    invoke(list, 49, 2, 1, 0);
    if (on) expected.location[2] = expected.size[2] = expected.filled[2] = 0;
    if (!now.read(list)) return 10;
    if (!now.same(expected)) return 11;

    // 4. a rejected unbind keeps what is left.
    invoke(list, 49, 1, 4, 0);
    if (!now.read(list) || !now.same(expected)) return 12;

    // 5. the last slot binds alone.
    invoke(list, 49, 3, 1, arg(&two[1]));
    if (on) expected.location[3] = 0x3000, expected.size[3] = 512, expected.filled[3] = 0x4000;
    if (!now.read(list) || !now.same(expected)) return 13;

    if (FAILED(static_cast<HRESULT>(invoke(device, 37)))) return 14;  // GetDeviceRemovedReason
    std::printf("stream output slot 49 (%s): bound %u of 4, device OK\n", on ? "on" : "off",
                (now.location[0] != 0) + (now.location[1] != 0) + (now.location[2] != 0) + (now.location[3] != 0));
    return 0;
}
