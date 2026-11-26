// Experimental (docs/startup-runtime.md): GPU time per frame of the title's direct queue, so a slow run can be
// told GPU-bound from CPU-bound. X2W_GPU_TIME=1 only; nothing is recorded or submitted otherwise.
//
// Every submission on the first direct queue seen is bracketed by two tiny lists that write a timestamp:
// one before the title's lists, one after them (the second also resolves both into a readback buffer). The
// span between the two is the time the GPU spent on that submission; the gap to the next one is idle (or
// the title's own queue waits). Spans are read back later, from slots whose fence has passed, so the
// submitting thread never waits. A busy ring (all slots in flight) skips that submission's stamps.
#pragma once

#include <d3d12.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace x2w_gpu_time {

// Per-frame GPU time and the idle share of the wall time, over the frames since the last report. Pure
// arithmetic so a test can drive it without a device.
class FrameStats {
public:
    // A span of gpu_ms that belongs to submission-time frame `frame`.
    void add_span(std::uint64_t frame, double gpu_ms)
    {
        Bucket& b = buckets_[frame % bucket_count];
        if (b.used && b.frame != frame) finish(b);
        b.used = true;
        b.frame = frame;
        b.ms += gpu_ms;
        busy_ms_ += gpu_ms;
    }
    // Closes every frame older than `frame` (their spans have all been read back or never will be).
    void close_before(std::uint64_t frame)
    {
        for (Bucket& b : buckets_)
            if (b.used && b.frame + 2 < frame) finish(b);
    }
    struct Report {
        std::uint64_t frames = 0;
        double avg_ms = 0, max_ms = 0, busy_ms = 0, wall_ms = 0, idle_share = 0;
    };
    // Takes the window's numbers and starts a new one. wall_ms is the time since the previous report.
    Report take(double wall_ms)
    {
        Report r;
        r.frames = frames_;
        r.avg_ms = frames_ ? sum_ms_ / static_cast<double>(frames_) : 0;
        r.max_ms = max_ms_;
        r.busy_ms = busy_ms_;
        r.wall_ms = wall_ms;
        r.idle_share = wall_ms > 0 ? std::clamp(1.0 - busy_ms_ / wall_ms, 0.0, 1.0) : 0;
        frames_ = 0;
        sum_ms_ = max_ms_ = busy_ms_ = 0;
        return r;
    }

private:
    static constexpr std::size_t bucket_count = 8;
    struct Bucket {
        bool used = false;
        std::uint64_t frame = 0;
        double ms = 0;
    };
    void finish(Bucket& b)
    {
        ++frames_;
        sum_ms_ += b.ms;
        max_ms_ = std::max(max_ms_, b.ms);
        b = {};
    }
    Bucket buckets_[bucket_count]{};
    std::uint64_t frames_ = 0;
    double sum_ms_ = 0, max_ms_ = 0, busy_ms_ = 0;
};

// Device side. All methods are called with the caller's submission lock held (or from one thread).
class Probe {
public:
    static constexpr UINT slots = 64;

    ~Probe()
    {
        for (Slot& s : slot_) {
            release(s.begin_list), release(s.end_list), release(s.begin_alloc), release(s.end_alloc);
        }
        release(heap_), release(readback_), release(fence_);
    }

    // Reserves a slot and submits its "begin" stamp on `queue` when the probe covers that queue. Returns the
    // slot, or -1 (nothing was submitted; after() with -1 does nothing).
    int before(ID3D12CommandQueue* queue, std::uint64_t frame)
    {
        if (!ready(queue)) return -1;
        harvest();
        for (UINT n = 0; n < slots; ++n) {
            const UINT i = (next_ + n) % slots;
            if (slot_[i].in_flight) continue;
            next_ = (i + 1) % slots;
            slot_[i].in_flight = true;
            slot_[i].frame = frame;
            ID3D12CommandList* lists[] = {slot_[i].begin_list};
            queue->ExecuteCommandLists(1, lists);
            return static_cast<int>(i);
        }
        ++skipped;
        return -1;
    }
    // Submits the "end" stamp (and the readback resolve) after the title's lists, then a fence for the slot.
    void after(ID3D12CommandQueue* queue, int slot)
    {
        if (slot < 0) return;
        ID3D12CommandList* lists[] = {slot_[slot].end_list};
        queue->ExecuteCommandLists(1, lists);
        slot_[slot].fence_value = ++fence_value_;
        if (FAILED(queue->Signal(fence_, slot_[slot].fence_value))) {
            slot_[slot].in_flight = false;  // never read back
            ++failed;
        }
    }
    // Reads every finished slot into the stats. `frame` closes frames that can no longer receive spans.
    void harvest(std::uint64_t frame = 0)
    {
        if (!fence_) return;
        const UINT64 done = fence_->GetCompletedValue();
        const UINT64* stamps = nullptr;
        for (UINT i = 0; i < slots; ++i) {
            Slot& s = slot_[i];
            if (!s.in_flight || s.fence_value == 0 || s.fence_value > done) continue;
            if (!stamps) {
                D3D12_RANGE range{0, slots * 2 * sizeof(UINT64)};
                if (FAILED(readback_->Map(0, &range, reinterpret_cast<void**>(const_cast<UINT64**>(&stamps))))) return;
            }
            const UINT64 t0 = stamps[2 * i], t1 = stamps[2 * i + 1];
            if (t1 >= t0 && frequency_) stats.add_span(s.frame, static_cast<double>(t1 - t0) * 1000.0 / static_cast<double>(frequency_));
            s.in_flight = false;
            s.fence_value = 0;
        }
        if (stamps) {
            D3D12_RANGE none{0, 0};
            readback_->Unmap(0, &none);
        }
        if (frame) stats.close_before(frame);
    }

    FrameStats stats;
    unsigned skipped = 0, failed = 0;

private:
    struct Slot {
        ID3D12CommandAllocator *begin_alloc = nullptr, *end_alloc = nullptr;
        ID3D12GraphicsCommandList *begin_list = nullptr, *end_list = nullptr;
        UINT64 fence_value = 0;
        std::uint64_t frame = 0;
        bool in_flight = false;
    };
    template <class T> static void release(T*& p)
    {
        if (p) p->Release();
        p = nullptr;
    }
    // True when the probe is built and `queue` is its queue. The first direct queue seen is the one; a failed
    // build is final.
    bool ready(ID3D12CommandQueue* queue)
    {
        if (queue_ == queue) return true;
        if (queue_ || attempted_) return false;
        const D3D12_COMMAND_QUEUE_DESC desc = queue->GetDesc();
        if (desc.Type != D3D12_COMMAND_LIST_TYPE_DIRECT) return false;
        attempted_ = true;
        ID3D12Device* device = nullptr;
        if (FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) || !build(device, queue)) {
            if (device) device->Release();
            return false;
        }
        device->Release();
        queue_ = queue;
        return true;
    }
    bool build(ID3D12Device* device, ID3D12CommandQueue* queue)
    {
        if (FAILED(queue->GetTimestampFrequency(&frequency_)) || !frequency_) return false;
        D3D12_QUERY_HEAP_DESC heap{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, slots * 2, 0};
        if (FAILED(device->CreateQueryHeap(&heap, IID_PPV_ARGS(&heap_)))) return false;
        D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
        D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, slots * 2 * sizeof(UINT64), 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                   D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        if (FAILED(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&readback_))))
            return false;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) return false;
        for (UINT i = 0; i < slots; ++i) {
            Slot& s = slot_[i];
            if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s.begin_alloc))) ||
                FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s.end_alloc))) ||
                FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.begin_alloc, nullptr, IID_PPV_ARGS(&s.begin_list))) ||
                FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.end_alloc, nullptr, IID_PPV_ARGS(&s.end_list))))
                return false;
            s.begin_list->EndQuery(heap_, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i);
            s.end_list->EndQuery(heap_, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i + 1);
            s.end_list->ResolveQueryData(heap_, D3D12_QUERY_TYPE_TIMESTAMP, 2 * i, 2, readback_, 2 * i * sizeof(UINT64));
            if (FAILED(s.begin_list->Close()) || FAILED(s.end_list->Close())) return false;
        }
        return true;
    }

    ID3D12CommandQueue* queue_ = nullptr;
    ID3D12QueryHeap* heap_ = nullptr;
    ID3D12Resource* readback_ = nullptr;
    ID3D12Fence* fence_ = nullptr;
    UINT64 frequency_ = 0, fence_value_ = 0;
    UINT next_ = 0;
    bool attempted_ = false;
    Slot slot_[slots];
};

}  // namespace x2w_gpu_time
