// GPU time probe (src/startup_stubs/gpu_time_probe.h): the per-frame arithmetic without a device, and the
// device side on WARP: every bracketed submission is read back as one span, and a busy GPU reads busy.
#include "startup_stubs/gpu_time_probe.h"

#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using Microsoft::WRL::ComPtr;
using x2w_gpu_time::FrameStats;

namespace {

void require(bool ok, const char* what)
{
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    std::exit(1);
}

bool close_to(double a, double b) { return std::fabs(a - b) < 1e-9; }

void arithmetic()
{
    FrameStats s;
    s.add_span(1, 2.0);
    s.add_span(1, 1.0);  // two submissions of frame 1: one 3 ms frame
    s.add_span(2, 5.0);
    s.add_span(3, 1.0);
    s.close_before(10);
    const auto r = s.take(100.0);
    require(r.frames == 3, "three frames closed");
    require(close_to(r.avg_ms, 3.0) && close_to(r.max_ms, 5.0), "average 3 ms, worst 5 ms");
    require(close_to(r.busy_ms, 9.0) && close_to(r.idle_share, 0.91), "9 ms busy of 100 ms is 91% idle");
    const auto again = s.take(50.0);
    require(again.frames == 0 && close_to(again.busy_ms, 0) && close_to(again.idle_share, 1.0), "a report starts a new window");
    s.add_span(4, 1.0);
    s.add_span(4 + 8, 1.0);  // same bucket, later frame: frame 4 closes on its own
    s.close_before(100);
    require(s.take(10.0).frames == 2, "a reused bucket closes the old frame");
    s.add_span(7, 200.0);
    s.close_before(100);
    require(close_to(s.take(100.0).idle_share, 0.0), "busy beyond the wall time clamps to zero idle");
}

void on_warp()
{
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    require(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))) && SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))),
            "WARP device");
    D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    ComPtr<ID3D12CommandQueue> queue;
    require(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))), "direct queue");
    desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ComPtr<ID3D12CommandQueue> other;
    require(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&other))), "compute queue");

    // A 256 MiB buffer copy: real GPU work the span has to cover.
    D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256u << 20, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    ComPtr<ID3D12Resource> from, to;
    require(SUCCEEDED(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&from))) &&
                SUCCEEDED(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&to))),
            "copy buffers");
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    require(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) &&
                SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list))),
            "copy list");
    list->CopyResource(to.Get(), from.Get());
    require(SUCCEEDED(list->Close()), "close copy list");
    ComPtr<ID3D12Fence> done;
    require(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))), "fence");

    x2w_gpu_time::Probe probe;
    require(probe.before(other.Get(), 1) == -1, "a compute queue is not covered");
    constexpr int submissions = 6;  // more than one frame, fewer than the ring
    for (int i = 0; i < submissions; ++i) {
        const int slot = probe.before(queue.Get(), 1 + i / 3);
        require(slot >= 0, "a slot for each submission");
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        probe.after(queue.Get(), slot);
    }
    require(probe.before(other.Get(), 1) == -1, "the probe stays on its first queue");
    require(SUCCEEDED(queue->Signal(done.Get(), 1)), "signal");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    require(SUCCEEDED(done->SetEventOnCompletion(1, event)) && WaitForSingleObject(event, 60000) == WAIT_OBJECT_0, "queue finished");
    CloseHandle(event);
    probe.harvest(100);
    const auto r = probe.stats.take(1000.0);
    std::printf("frames %llu avg %.3f ms max %.3f ms busy %.3f ms, skipped %u failed %u\n", static_cast<unsigned long long>(r.frames), r.avg_ms,
                r.max_ms, r.busy_ms, probe.skipped, probe.failed);
    require(r.frames == 2, "two frames of three submissions each");
    require(r.busy_ms > 0 && r.max_ms >= r.avg_ms && r.avg_ms > 0, "the copies took measurable GPU time");
    require(!probe.skipped && !probe.failed, "no stamp skipped or failed");
    // The ring is free again after the read back: a full ring of submissions still gets its slots.
    for (UINT i = 0; i < x2w_gpu_time::Probe::slots; ++i) {
        const int slot = probe.before(queue.Get(), 200);
        require(slot >= 0, "slots are reusable after harvest");
        probe.after(queue.Get(), slot);
    }
    require(probe.before(queue.Get(), 200) == -1 && probe.skipped == 1, "a full ring skips the stamp and counts it");
}

}  // namespace

int main()
{
    arithmetic();
    on_warp();
    std::puts("gpu_time_probe: ok");
    return 0;
}
