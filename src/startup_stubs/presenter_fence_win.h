#pragma once
#include "presenter_fence.h"
#include <d3d12.h>

namespace x2w_startup {
template <class Slow>
FenceWait wait_for_fence(ID3D12Fence* fence, UINT64 value, HANDLE event, ID3D12Device* device, Slow slow)
{
    struct Operations {
        ID3D12Fence* fence;
        HANDLE event;
        ID3D12Device* device;
        Slow& report;
        std::uint64_t now() { return GetTickCount64(); }
        bool removed() { return device->GetDeviceRemovedReason() != S_OK; }
        std::uint64_t completed() { return fence->GetCompletedValue(); }
        bool arm(std::uint64_t value) { return event && SUCCEEDED(fence->SetEventOnCompletion(value, event)); }
        FenceSlice wait() {
            const DWORD result = WaitForSingleObject(event, 1000);
            return result == WAIT_OBJECT_0 ? FenceSlice::signaled : result == WAIT_TIMEOUT ? FenceSlice::timeout : FenceSlice::failed;
        }
        void slow() { report(); }
    } ops{fence, event, device, slow};
    return wait_for_submission(ops, value);
}
}  // namespace x2w_startup
