#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
#include "resource_budget.h"

namespace x2w_upload {
using Microsoft::WRL::ComPtr;
struct Page {
    ComPtr<ID3D12Resource> resource;
    unsigned char* cpu = nullptr;
    UINT64 gpu = 0, size = 0, cursor = 0;
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
    return page.cpu && page.gpu ? S_OK : E_FAIL;
}
struct Allocation { uintptr_t cpu; UINT64 gpu, bytes; ID3D12Resource* resource; UINT64 offset; };
struct Arena {
    std::vector<std::unique_ptr<Page>> pages;
    std::vector<Allocation> allocations;

    // The caller must retire all recordings and GPU submissions before releasing their pages.
    void reset() { allocations.clear(); pages.clear(); }

    Allocation range(UINT64 cpu, UINT64 bytes) const {
        if (!bytes) return {};
        for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
            if (cpu < it->cpu) continue;
            UINT64 offset = cpu - it->cpu;
            if (offset < it->bytes && bytes <= it->bytes - offset)
                return {static_cast<uintptr_t>(cpu), it->gpu + offset, bytes, it->resource, it->offset + offset};
        }
        return {};
    }
    UINT64 translate(UINT64 cpu, UINT64 bytes) const {
        return range(cpu, bytes).gpu;
    }
    Allocation allocate(ID3D12Device* device, UINT64 bytes, UINT64 alignment) {
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
        auto page = std::make_unique<Page>();
        if (FAILED(make_page(device, (std::max)(UINT64{65536}, bytes + alignment - 1), *page))) return {};
        pages.push_back(std::move(page));
        Allocation a = take(*pages.back());
        if (!a.cpu) pages.pop_back();
        return a;
    }
};
}
