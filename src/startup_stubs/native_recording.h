#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include "native_copy_ranges.h"

namespace x2w_startup {
struct NativeDescriptorTable {
    ID3D12DescriptorHeap *heap{};
    D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
    UINT increment{};
};
// draw_lock protects recording and submission. Each reset starts a new
// allocation sequence. A closed recording and each of its submits own the
// same immutable tables, even when the command-list pointer is reused.
struct NativeRecording {
    struct BufferCopy {
        const void* owner;
        std::uint64_t base, begin, end;
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    };
    std::vector<BufferCopy> copies;
    std::unordered_map<IUnknown *, Microsoft::WRL::ComPtr<IUnknown>> objects;
    NativeDescriptorTable next;
    UINT remaining = 0;
    void retain(IUnknown *object) { objects.try_emplace(object, object); }
    NativeDescriptorTable allocate(ID3D12Device *device, UINT count) {
        if (!count)
            return {};
        if (remaining < count) {
            const UINT capacity = (std::max)(256u, count);
            D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, capacity,
                                            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
            Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> page;
            if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&page))))
                throw std::runtime_error("native descriptor page allocation failed");
            retain(page.Get());
            next = {page.Get(), page->GetCPUDescriptorHandleForHeapStart(), page->GetGPUDescriptorHandleForHeapStart(),
                    device->GetDescriptorHandleIncrementSize(desc.Type)};
            remaining = capacity;
        }
        auto table = next;
        next.cpu.ptr += SIZE_T(count) * next.increment;
        next.gpu.ptr += UINT64(count) * next.increment;
        remaining -= count;
        return table;
    }
};
} // namespace x2w_startup
