#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include "startup_stubs/d3d12_teardown.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace x2w_startup {

struct DrawDump {
    Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows{};
    std::uint64_t key{};
    UINT count{};
    DXGI_FORMAT format{};
    UINT index{};
    LONG frame{};
    // Set for a one-texel history sample (X2W_DRAW_DUMP_PIXEL): the copy covers this pixel only and
    // the report reads one 32-bit word instead of writing a BMP.
    bool pixel{};
    UINT pixel_x{};
    UINT pixel_y{};
};

// A recorded copy belongs to a command-list recording until it is either reset
// or submitted. After submission the queue fence owns the completion decision.
class DrawReadbacks {
    struct Entry {
        ID3D12CommandList* recording;
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        DrawDump dump;
    };
    static constexpr unsigned limit = 48;
    std::mutex mutex_;
    std::vector<Entry> entries_;
    unsigned consumed_ = 0;
    unsigned sequence_ = 0;

public:
    bool full()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size() + consumed_ >= limit;
    }

    bool record(ID3D12CommandList* list, DrawDump dump)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.size() + consumed_ >= limit) return false;
        dump.index = ++sequence_;
        entries_.push_back({list, {}, std::move(dump)});
        return true;
    }

    // Test hook (startup_teardown_unload, #2572; the review of the ownership change asked for one):
    // enters one recorded copy exactly as execute() leaves it after a submit -- the readback buffer and
    // the queue fence that decides its completion -- without a command list, a queue or a draw. No real
    // path calls it. False when the capture budget is full.
    bool record_for_test(Microsoft::WRL::ComPtr<ID3D12Resource> buffer, Microsoft::WRL::ComPtr<ID3D12Fence> fence)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.size() + consumed_ >= limit) return false;
        DrawDump dump;
        dump.buffer = std::move(buffer);
        dump.index = ++sequence_;
        entries_.push_back({nullptr, std::move(fence), std::move(dump)});
        return true;
    }

    // Test hook: the object the first recorded entry owns, without adding a reference -- slot 0 the
    // readback buffer, 1 its fence. The fixture takes its own reference and watches it stay alone once
    // the module is unloaded. Null for an unknown slot or an empty cache.
    IUnknown* owned_for_test(unsigned slot)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& entry : entries_) {
            if (slot == 0 && entry.dump.buffer) return entry.dump.buffer.Get();
            if (slot == 1 && entry.fence) return entry.fence.Get();
        }
        return nullptr;
    }

    void discard(ID3D12CommandList* list)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
            [list](const Entry& e) { return e.recording == list; }), entries_.end());
    }

    HRESULT execute(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto in_batch = [&](const Entry& entry) {
            return entry.recording && std::find(lists, lists + count, entry.recording) != lists + count;
        };
        queue->ExecuteCommandLists(count, lists);
        if (std::none_of(entries_.begin(), entries_.end(), in_batch)) return S_OK;

        Microsoft::WRL::ComPtr<ID3D12Device> device;
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        HRESULT hr = queue->GetDevice(IID_PPV_ARGS(&device));
        if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (SUCCEEDED(hr)) hr = queue->Signal(fence.Get(), 1);
        for (auto& entry : entries_) {
            if (!in_batch(entry)) continue;
            entry.recording = nullptr;
            // A signalling failure provides no safe point to read or release
            // submitted storage. Retain it within the same capture budget.
            if (SUCCEEDED(hr)) entry.fence = fence;
        }
        return hr;
    }

    std::vector<DrawDump> take_completed()
    {
        std::vector<DrawDump> complete;
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = entries_.begin(); it != entries_.end();) {
            const UINT64 value = it->fence ? it->fence->GetCompletedValue() : 0;
            if (!value) {
                ++it;
                continue;
            }
            // D3D12 reports UINT64_MAX for device removal, including during a wait.
            if (value != UINT64_MAX) {
                complete.push_back(std::move(it->dump));
                ++consumed_;
            }
            it = entries_.erase(it);
        }
        return complete;
    }

    // Lifetime boundary, same contract as the caches in startup_stubs/d3d12_teardown.h: a recorded copy and
    // its fence live for as long as this module is loaded, because a list that was resubmitted still holds
    // them. This object is a function-local static in draw_dumps(), so it is destroyed during the module's
    // CRT teardown. On process exit the display driver is already detached and a release there faults
    // inside it, so the references are detached; FreeLibrary keeps the process and the driver alive and
    // the entries release their objects normally. Each reference is accounted for on the path it took and,
    // with X2W_TEARDOWN_REPORT, reported as one x2w-teardown readbacks line.
    ~DrawReadbacks()
    {
        const bool detach = d3d12_process_terminating && !teardown_guard_disabled("readbacks");
        unsigned detached = 0, released = 0;
        for (auto& entry : entries_) {
            if (entry.dump.buffer) {
                if (detach) {
                    entry.dump.buffer.Detach();
                    ++detached;
                } else {
                    ++released;
                }
            }
            if (entry.fence) {
                if (detach) {
                    entry.fence.Detach();
                    ++detached;
                } else {
                    ++released;
                }
            }
        }
        teardown_report("readbacks", detached, released);
    }
};
}  // namespace x2w_startup
