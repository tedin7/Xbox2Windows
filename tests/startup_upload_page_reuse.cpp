// List upload pages are reused once the GPU has passed the recording that used them, so a title that records one
// list again and again on an allocator it rarely resets keeps a flat number of pages.
//
//   startup_upload_page_reuse <warp|hardware> [untracked]
//       (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1)
//
// Through D3D12XboxCreateDevice one console list on one allocator, never reset, runs 10,000 cycles: Reset the list on
// the allocator, take 0x280 bytes of upload memory through list slot 148 (the builder's request), write them, Close,
// submit, and wait for the GPU every 16 cycles. Each recording takes its own arena, so without reuse every cycle
// adds a 64 KB page (10,000 pages, 640 MB).
//
//   1. steady: at most 64 pages alive at the end, at most 64 ever created, and the rest handed out again.
// X2W_TEST_UPLOAD_PAGES_MUTANT=never-reuse (the stand-in without the reuse) must fail step 1 (exit 12).
//
//   2. held: the queue waits on a fence the CPU has not signalled and 100 more cycles are submitted behind it. Nothing
//      the GPU has not passed may be reused: no arena is recycled early while the wait holds, and each cycle takes a
//      page (a new one or one from the free pool that an earlier, finished recording gave back). After the CPU
//      signals the fence, the next cycles recycle them again.
//
// With `untracked` (X2W_TEST_SUBMIT_FENCE_FAIL=1) no submit can be tracked: 300 cycles must create 300 pages and
// recycle none.
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
extern "C" __declspec(dllimport) void x2w_d3d12_upload_page_stats(unsigned long long* live, unsigned long long* free,
                                                                  unsigned long long* created, unsigned long long* reused,
                                                                  unsigned long long* recycled_early);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

constexpr unsigned kCycles = 10000, kWaitEvery = 16, kHeldCycles = 100, kUntrackedCycles = 300;
constexpr unsigned long long kMaxPages = 64;

struct Stats {
    unsigned long long live, free, created, reused, recycled;
};
Stats stats()
{
    Stats s{};
    x2w_d3d12_upload_page_stats(&s.live, &s.free, &s.created, &s.reused, &s.recycled);
    return s;
}
void print(const char* label, const Stats& s)
{
    std::printf("%s: pages alive %llu, free %llu, created %llu, reused %llu, arenas recycled early %llu\n", label,
                s.live, s.free, s.created, s.reused, s.recycled);
}

struct Loop {
    void* queue = nullptr;
    void* fence = nullptr;
    void* allocator = nullptr;
    void* list = nullptr;
    Word value = 0;

    bool reached()
    {
        const auto deadline = GetTickCount64() + 20000;
        while (invoke(fence, 8) < value) {
            if (GetTickCount64() >= deadline) return false;
            Sleep(0);
        }
        return true;
    }
    // One record/execute cycle; false if slot 148 refused or the list could not be submitted.
    bool cycle(unsigned index, bool wait)
    {
        if (!ok(invoke(list, 139, arg(allocator)))) return false;
        void* upload = reinterpret_cast<void*>(invoke(list, 148, 0x280, 0x10));
        if (!upload) return false;
        std::memset(upload, static_cast<int>(index & 0xff), 0x280);
        if (!ok(invoke(list, 9))) return false;
        void* lists[] = {list};
        invoke(queue, 10, 1, arg(lists));
        if (!ok(invoke(queue, 14, arg(fence), ++value))) return false;
        return !wait || reached();
    }
};

int select_adapter(const char* backend, ComPtr<IDXGIAdapter>& adapter)
{
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    if (std::strcmp(backend, "warp") == 0) return SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ? 0 : 3;
    if (std::strcmp(backend, "hardware") != 0) return 2;
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
            return 0;
        }
    }
}

int untracked_cycles(Loop& loop)
{
    const Stats before = stats();
    for (unsigned i = 0; i < kUntrackedCycles; ++i)
        if (!loop.cycle(i, false)) return 90;
    const Stats after = stats();
    print("untracked", after);
    // The queue fence of the fixture itself still advances: let the GPU finish before the process ends.
    if (!loop.reached()) return 91;
    return after.created - before.created == kUntrackedCycles && after.recycled == before.recycled ? 0 : 92;
}

int held_cycles(Loop& loop, void* device)
{
    void* hold = nullptr;
    if (!ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&hold))) || !hold) return 40;
    if (!ok(invoke(loop.queue, 17, arg(hold), 1))) return 41;  // queue Wait(hold, 1): nothing after it runs yet
    // The first cycle behind the wait still recycles the last steady recording, which the GPU passed: count after it.
    if (!loop.cycle(0, false)) return 42;
    const Stats before = stats();
    for (unsigned i = 1; i < kHeldCycles; ++i)
        if (!loop.cycle(i, false)) return 42;
    const Stats held = stats();
    print("2 held", held);
    if (invoke(loop.fence, 8) >= loop.value) return 43;  // the submits are still behind the wait
    const unsigned long long taken = (held.created - before.created) + (held.reused - before.reused);
    if (held.recycled != before.recycled || taken != kHeldCycles - 1) return 44;
    if (!ok(invoke(hold, 10, 1)) || !loop.reached()) return 45;  // hold Signal(1) from the CPU
    for (unsigned i = 0; i < 2 * kWaitEvery; ++i)
        if (!loop.cycle(i, true)) return 46;
    const Stats released = stats();
    print("2 released", released);
    return released.recycled > held.recycled && released.live <= held.live ? 0 : 47;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2 && !(argc == 3 && std::strcmp(argv[2], "untracked") == 0)) return 2;
    ComPtr<IDXGIAdapter> adapter;
    if (const int rc = select_adapter(argv[1], adapter)) return rc;
    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 4;

    Loop loop;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (!ok(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&loop.queue))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&loop.fence))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&loop.allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(loop.allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)),
                   arg(&loop.list))))
        return 7;
    if (!ok(invoke(loop.list, 9))) return 8;  // created open: close it, every cycle starts with a Reset

    if (argc == 3) return untracked_cycles(loop);

    for (unsigned i = 0; i < kCycles; ++i) {
        if (!loop.cycle(i, (i + 1) % kWaitEvery == 0)) return 10;
        if (i % 1000 == 999 && stats().live > 4 * kMaxPages) {  // growing: stop before it takes the machine
            loop.reached();
            print("1 growing", stats());
            return 12;
        }
    }
    if (!loop.reached()) return 11;
    const Stats steady = stats();
    print("1 steady", steady);
    if (steady.live > kMaxPages || steady.created > kMaxPages || steady.reused + steady.created < kCycles) return 12;
    return held_cycles(loop, device);
}
