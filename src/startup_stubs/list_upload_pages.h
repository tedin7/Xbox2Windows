#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "resource_budget.h"

namespace x2w_upload {
using Microsoft::WRL::ComPtr;

// Upload pages of the list arenas, for the x2w-budget line: live pages (in an arena or in the free pool) and their
// bytes, pages waiting in the free pool, pages ever created and pages handed out again from the pool.
struct PageCounters {
    LONG64 live, live_bytes, free, created, reused;
};
inline PageCounters page_counters{};

struct Page {
    ComPtr<ID3D12Resource> resource;
    unsigned char* cpu = nullptr;
    UINT64 gpu = 0, size = 0, cursor = 0;
    bool counted = false;
    Page() = default;
    Page(const Page&) = delete;
    Page& operator=(const Page&) = delete;
    ~Page() {
        if (!counted) return;
        InterlockedDecrement64(&page_counters.live);
        InterlockedAdd64(&page_counters.live_bytes, -static_cast<LONG64>(size));
    }
};
inline HRESULT make_page(ID3D12Device* device, UINT64 bytes, Page& page) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&page.resource));
    if (FAILED(hr)) return hr;
    x2w_budget::committed(device, page.resource.Get(), D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RANGE no_reads{};
    void* mapped = nullptr;
    hr = page.resource->Map(0, &no_reads, &mapped);
    if (FAILED(hr)) return hr;
    page.cpu = static_cast<unsigned char*>(mapped);
    page.gpu = page.resource->GetGPUVirtualAddress();
    page.size = bytes;
    page.counted = true;
    InterlockedIncrement64(&page_counters.live);
    InterlockedAdd64(&page_counters.live_bytes, static_cast<LONG64>(bytes));
    InterlockedIncrement64(&page_counters.created);
    return page.cpu && page.gpu ? S_OK : E_FAIL;
}

// Pages of finished recordings, kept mapped for the next recordings. Bounded: a page past either limit is released.
inline constexpr std::size_t max_free_pages = 256;
inline constexpr UINT64 max_free_bytes = UINT64{32} << 20;
struct FreePages {
    std::vector<std::unique_ptr<Page>> pages;
    UINT64 bytes = 0;

    void give(std::unique_ptr<Page> page) {
        if (!page || pages.size() >= max_free_pages || page->size > max_free_bytes - bytes) return;  // released here
        page->cursor = 0;
        bytes += page->size;
        pages.push_back(std::move(page));
        InterlockedIncrement64(&page_counters.free);
    }
    // The smallest free page of at least `bytes`, or null.
    std::unique_ptr<Page> take(UINT64 bytes) {
        auto best = pages.end();
        for (auto it = pages.begin(); it != pages.end(); ++it)
            if ((*it)->size >= bytes && (best == pages.end() || (*it)->size < (*best)->size)) best = it;
        if (best == pages.end()) return nullptr;
        std::unique_ptr<Page> page = std::move(*best);
        pages.erase(best);
        this->bytes -= page->size;
        InterlockedDecrement64(&page_counters.free);
        InterlockedIncrement64(&page_counters.reused);
        return page;
    }
};

struct Allocation { uintptr_t cpu; UINT64 gpu, bytes; ID3D12Resource* resource; UINT64 offset; };
struct DescriptorSnapshot {
    ComPtr<ID3D12DescriptorHeap> heap;
    std::vector<ComPtr<ID3D12Resource>> resources;
};
struct Arena {
    std::vector<DescriptorSnapshot> descriptor_snapshots;
    std::vector<std::unique_ptr<Page>> pages;
    std::vector<Allocation> allocations;
    // The queue fences the recording was submitted behind (per fence, the last value), whether one of its submits
    // could not be tracked, and whether its list has moved on (Reset or destroyed), so it can never be submitted again.
    std::vector<std::pair<ID3D12Fence*, UINT64>> submits;
    bool untracked = false, retired = false;
    // Taken when the recording is retired: for every queue, its fence, the last value it had signalled then and where its
    // current last value lives. A list of another allocator that recorded a read of these pages before the retire may be
    // submitted after it, and its submits are not this recording's own: the pages wait for later submits of every queue.
    struct QueueGuard { ID3D12Fence* fence; UINT64 at_retire; const UINT64* last; };
    std::vector<QueueGuard> guards;

    // The caller must retire all recordings and GPU submissions before releasing their pages.
    void reset() { descriptor_snapshots.clear(); allocations.clear(); pages.clear(); }

    // The recording's pages go to `free` (or are released past its limits). The same precondition as reset().
    void recycle(FreePages& free) {
        descriptor_snapshots.clear();
        allocations.clear();
        for (auto& page : pages) free.give(std::move(page));
        pages.clear();
    }

    void submitted(ID3D12Fence* fence, UINT64 value) {
        const auto same = std::find_if(submits.begin(), submits.end(), [&](const auto& s) { return s.first == fence; });
        if (same == submits.end()) submits.emplace_back(fence, value);
        else same->second = value;
    }

    // A retired recording whose every submit the GPU has passed. A recording never submitted is not finished: it may
    // be a bundle another list ran, and that list's submits are not its own.
    // `global`: also every queue has passed two submits made after the retire (or has nothing left to run).
    bool finished(bool global = true) const {
        if (!retired || untracked || submits.empty()) return false;
        for (const auto& submit : submits)
            if (submit.first->GetCompletedValue() < submit.second) return false;
        if (!global) return true;
        for (const auto& guard : guards) {
            const UINT64 done = guard.fence->GetCompletedValue();
            if (done == UINT64_MAX) return false;
            const UINT64 newest = *guard.last;
            if (done < newest && done < guard.at_retire + 2) return false;
        }
        return true;
    }

    Allocation range(UINT64 cpu, UINT64 bytes, bool remaining = false) const {
        if (!bytes) return {};
        for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
            if (cpu < it->cpu) continue;
            UINT64 offset = cpu - it->cpu;
            if (offset < it->bytes && bytes <= it->bytes - offset)
                return {static_cast<uintptr_t>(cpu), it->gpu + offset, remaining ? it->bytes - offset : bytes, it->resource, it->offset + offset};
        }
        return {};
    }
    UINT64 translate(UINT64 cpu, UINT64 bytes) const {
        return range(cpu, bytes).gpu;
    }
    Allocation allocate(ID3D12Device* device, UINT64 bytes, UINT64 alignment, FreePages* free = nullptr) {
        if (!device || !bytes || !alignment || (alignment & (alignment - 1))) return {};
        alignment = (std::max)(alignment, UINT64{16});
        if (bytes > UINT64_MAX - alignment) return {};
        auto take = [&](Page& page) -> Allocation {
            const auto base = reinterpret_cast<uintptr_t>(page.cpu);
            const UINT64 mask = alignment - 1;
            if ((base ^ page.gpu) & mask) return {};
            const UINT64 gap = (0 - (base + page.cursor)) & mask;
            if (gap > page.size - page.cursor) return {};
            const UINT64 offset = page.cursor + gap;
            if (bytes > page.size - offset) return {};
            Allocation result{base + offset, page.gpu + offset, bytes, page.resource.Get(), offset};
            allocations.push_back(result);
            page.cursor = offset + bytes;
            return result;
        };
        if (!pages.empty()) {
            Allocation a = take(*pages.back());
            if (a.cpu) return a;
        }
        const UINT64 page_bytes = (std::max)(UINT64{65536}, bytes + alignment - 1);
        std::unique_ptr<Page> page = free ? free->take(page_bytes) : nullptr;
        if (!page) {
            page = std::make_unique<Page>();
            if (FAILED(make_page(device, page_bytes, *page))) return {};
        }
        pages.push_back(std::move(page));
        Allocation a = take(*pages.back());
        if (!a.cpu) pages.pop_back();
        return a;
    }
};
}
