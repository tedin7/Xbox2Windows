// The rule that gives a list upload arena's pages back (src/startup_stubs/list_upload_pages.h, Arena::finished): a retired
// recording whose own submits the GPU passed is not enough, because a list of another allocator that recorded a read of the
// pages may be submitted after the retire. The arena also waits for two later submits of every queue, or for a queue that
// has nothing left to run.
//
//   startup_upload_arena_guard [--own-only]    (--own-only is the rule of Run P: the checks that need the guard must FAIL)
//
// Fences stand for queues: the CPU signals them, so every state is exact. Exit 0 pass, 77 no adapter, 1 a check failed.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>

#include "startup_stubs/list_upload_pages.h"

using Microsoft::WRL::ComPtr;
namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    if (ok) return;
    ++failures;
    std::printf("FAIL: %s\n", what);
}
}  // namespace

int main(int argc, char** argv)
{
    const bool own_only = argc > 1 && !std::strcmp(argv[1], "--own-only");
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        std::puts("no adapter");
        return 77;
    }
    ComPtr<ID3D12Fence> own, other;  // the recording's queue and another queue
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&own));
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&other));
    UINT64 own_last = 0, other_last = 0;  // the last value each queue has signalled (what the runtime keeps per queue)

    const auto fresh = [&] {
        x2w_upload::Arena arena;
        arena.submitted(own.Get(), own_last);
        arena.retired = true;
        arena.guards = {{own.Get(), own_last, &own_last}, {other.Get(), other_last, &other_last}};
        return arena;
    };
    const auto finished = [&](const x2w_upload::Arena& arena) { return arena.finished(!own_only); };

    // Both queues have work signalled: own 5, other 7. Nothing has completed.
    own_last = 5;
    other_last = 7;
    x2w_upload::Arena a = fresh();
    check("own submit not passed: not finished", !finished(a));
    own->Signal(5);
    check("own passed, the other queue still has its work: not finished", !finished(a));
    other->Signal(7);
    check("both queues idle: finished", finished(a));

    // The other queue gets more work after the retire; its passed submits count only when two later ones passed.
    x2w_upload::Arena b = fresh();  // retire snapshot: own 5, other 7
    other_last = 9;                 // two submits made after the retire
    own_last = 5;
    check("later work on the other queue is pending: not finished", !finished(b));
    other->Signal(8);
    check("one later submit passed, one pending: not finished", !finished(b));
    other->Signal(9);
    check("two later submits passed: finished", finished(b));

    // A queue that stays busy: it has passed the snapshot plus two even though newer work is pending.
    x2w_upload::Arena c = fresh();  // snapshot: own 5, other 9
    other_last = 12;
    check("busy queue, nothing after the retire has passed: not finished", !finished(c));
    other->Signal(11);
    check("busy queue passed two later submits: finished", finished(c));

    // Never submitted (a bundle another list ran) and untracked recordings are never finished.
    x2w_upload::Arena d;
    d.retired = true;
    check("never submitted: not finished", !finished(d));
    x2w_upload::Arena e = fresh();
    e.untracked = true;
    check("untracked: not finished", !finished(e));
    x2w_upload::Arena f = fresh();
    f.retired = false;
    check("not retired: not finished", !finished(f));

    std::printf("startup_upload_arena_guard: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
