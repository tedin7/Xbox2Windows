#pragma once

// Opt-in backing for one ordinary linear buffer kind placed at a title address (#1914, #874):
// device slot 133 (CreatePlacedResourceX) with a buffer description, row-major or unknown layout,
// and initial state COMMON. The title writes it with the CPU and the GPU reads it (vertex, index,
// constant, shader-resource or indirect-argument data). X2W_D3D12_LINEAR_BUFFERS selects it:
//
//   unset or 0  the default: slot 133 keeps its backing (an existing heap cached by allocation
//               base for a private range, otherwise a committed default buffer that does not alias
//               the title memory and holds zeros). Nothing in this file runs.
//   1           share when the adapter, the extent and the use allow it, else copy, else keep the
//               default backing and record why.
//   copy        never share; copy, else keep the default backing and record why.
//
// XMem owns the title memory (identity, mapping, offsets, lifetime); this file owns the desktop
// backing and the fences that say when the GPU is done with it.
//
// Share. Only when D3D12_FEATURE_EXISTING_HEAPS reports support (queried once per device, before
// any open), the whole buffer lies in one committed private allocation (MEM_PRIVATE) at a 64 KiB-
// aligned offset, and the opened heap covers it. The heap is cached by allocation base, as the
// default backing caches it: while it is open the range cannot be released or decommitted
// (VirtualFree fails with error 87, measured in #1942), so a later allocation cannot appear at that
// base under it. Each opened heap gets a generation. The GPU reads and writes the title memory
// itself; nothing is copied.
//
// An XMem view (MEM_MAPPED) is not shared: a heap opened from a view keeps it mapped, and XMem's
// unmap of those pages then fails with error 87 (tests/linear_buffer_backing_check.cpp measures it).
// Desktop resources live for the process, so the title could never unmap or free those pages again.
// Sharing reports HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED) for a view and the buffer is copied. A
// buffer across several allocations, or at an unaligned offset, reports E_INVALIDARG. Both are
// decided without a D3D12 call.
//
// Copy. A committed default-heap buffer whose bytes are refreshed from the title memory at each
// submission through a command-queue wrapper, before the title's lists execute, never at bind or
// record. Only the bytes that differ from the last upload are copied, through an upload staging
// buffer.
//
// Every refresh runs on one internal COPY queue per device, which never waits on a title fence. A
// refresh's copy (staging -> backing) is executed there and signals an internal fence value; the
// buffer remembers the value of its last refresh. Every submission through a queue wrapper, on any
// queue type, first makes that queue Wait for the highest such value among the live copied buffers
// (when it has not completed yet). A consumer on any queue therefore runs after every refresh that
// was enqueued before it, wherever that refresh came from, so the bytes recorded as uploaded at
// enqueue are the bytes the consumer reads. A submission whose recording writes a copied buffer
// (gpu_written, below) is ordered behind that buffer's earlier refreshes by the same Wait.
//
// The refresh is ordered behind earlier GPU access too (the producer wait). Simultaneous access
// allows concurrent operations on one buffer from several queues only on disjoint ranges, so a copy
// into a backing must not overlap an earlier submission that still reads or writes it. After the
// title's lists, each queue wrapper signals a fence of its own (one per desktop queue); before a
// refresh's copy executes, the internal queue Waits for every such queue's latest value that has not
// completed (producer_waits in the totals). The wrappers cannot tell which buffers a submission
// reads (the title binds them by GPU virtual address in root arguments, views and descriptors), so
// every earlier submission counts as a user of every copied backing: the conservative choice.
//
// A submission whose refresh has no safe completion order is refused (unsafe_refresh_refusals,
// logged). That is the case when the refresh would have to wait for an earlier submission, on any
// queue including its own, that sits behind a title Wait (queue slot 17) on a fence that has not
// reached its value, or for a submission without a user fence value (its Signal failed). Waiting could
// deadlock: the fence may be signalled only by work that itself waits for this refresh (queue 1 waits on
// a fence that queue 2 signals after its lists, case xq.write; every later consumer waits for the
// refresh). Not waiting would let the copy overlap that read: a blocked queue is not known to stay
// blocked until the copy finishes, since another producer (the CPU, or another queue) may reach the
// fence while it runs. The decision is made under internal_mutex_, right before the refresh would be
// enqueued: nothing of the submission reaches the GPU, neither the refresh, nor the consumer Wait, nor
// the title's lists, nor the queue's user fence Signal after them (no value stands for lists that did
// not run). Bookkeeping stays as it was: the staging buffers and refresh list are free again, no
// buffer's upload record, refresh value, staleness or failure count changes, and a recorded copy's GPU
// write stays pending on its recording (list_copy.inl commits it only after an executed submission). The
// next submission compares the title memory with the same upload record and copies what still differs.
// The title's own Signal and Wait on its queues (slot 16 and any shape-dispatched Signal, slot 17) do not
// go through this file, so a later Signal still releases the earlier queue. A title Wait issued on the
// desktop queue past the wrapper is not seen, and that queue is waited for. A submission that has no
// changed bytes to refresh is never refused; it waits only for refreshes already enqueued, none of which
// waits for a blocked queue.
//
// The refusal is only as good as the record of which submissions sit behind which title Wait, and a
// title can call slot 17 and ExecuteCommandLists on one queue from two threads at once. Each queue's
// record therefore has an order lock (queue_order), so its title waits and user fence values are
// recorded in the order the desktop queue received them. Slot 17 holds it from the desktop Wait through
// note_title_wait; a submission holds it from the title's ExecuteCommandLists through after_submit's
// user fence Signal. Without it a submission could reach the desktop queue behind a Wait yet take its
// user value before that Wait was recorded, which would then count only later submissions as behind it:
// another queue's refresh would wait for the blocked submission instead of being refused. The lock is
// not held across the refresh (its staging waits and fixture pauses run without it), and nothing that
// waits on the CPU for the GPU runs under it, since a desktop Wait, ExecuteCommandLists and Signal only
// enqueue. The title's Signal (slot 16, or a shape-dispatched Signal) does not take it, so a queue whose
// submission was refused can still signal the fence its blocked peer waits on. The order is: a queue's
// order lock, then draw_lock and allocator_lock (render mode), then mutex_; mutex_ is never held while it
// is taken (queue_order looks the lock up under mutex_ and takes it after releasing mutex_). A refresh
// decision that runs while a Wait on another queue has been enqueued but not yet recorded is still safe:
// that Wait precedes only submissions whose user value is not yet assigned, and a refresh waits only for
// values already assigned.
//
// A staging buffer is reused only after the internal fence value of the copy that read it has
// completed (title fences do not matter: only that copy reads it). If the internal queue or its fence
// cannot be created, the copy cannot order its consumers: placements keep the default backing and
// record the refusal (below).
//
// Where neither path can keep its guarantees, slot 133 keeps the default backing and returns what it
// returns without this switch. A title on the console does not see CreatePlacedResourceX fail, so a
// failed call would be a new failure (a title run that took the null object faulted at 1.76 s). The
// refusal is recorded instead: in the slot 133 log line ("kept default backing: linear refusal
// 0x..."), in x2w_d3d12_linear_buffer_info (path 3 and refusal_hr) and in the totals (kept_default,
// last_refusal_hr). The refusals:
//   - a GPU-written use (ALLOW_UNORDERED_ACCESS) that cannot be shared: the copy has no point at which
//     GPU writes reach the title memory. refusal_hr is the sharing HRESULT, or DXGI_ERROR_UNSUPPORTED
//     when sharing was not attempted (X2W_D3D12_LINEAR_BUFFERS=copy).
//   - no internal refresh queue or fence (X2W_TEST_SUBMIT_FENCE_FAIL=1 gives E_NOT_VALID_STATE, else
//     the HRESULT of creating them): consumers could not be ordered behind refreshes.
//   - the copy's default buffer cannot be created: that HRESULT.
//   - the XMem lease path is on (X2W_MESH86_XMEM_SRV or X2W_MESH_INDIRECT_XMEM) and the placement is one
//     it takes: XMem pages (MEM_MAPPED) with console flags 0, kept as the default copied placement that
//     the lease path mirrors and publishes. HRESULT_FROM_WIN32(ERROR_BUSY): deferred to that owner, so
//     the two never back one placement.
//
// Title memory is read twice per refresh: compared with the last upload under mutex_ (after a probe),
// then staged without it. Every live copied buffer is snapshotted, not only those the submitted lists
// read (the wrappers cannot tell), so the title may release one on another thread in between and free
// or unmap its pages. Its memory is probed again right before it is staged, and staged under a fault
// guard: a buffer that is gone is skipped and counted (unreadable), a read that faults anyway is
// counted (copy_faults) and fails that buffer's refresh. A buffer released after the snapshot whose
// pages are still there is staged for that submission, which may read it.
//
// Apart from the unsafe refresh above, nothing is refused at submit: holding back a submission can
// leave a title waiting on its fences. When a copied buffer's refresh cannot be staged (a staging
// buffer or refresh list cannot be created, or the pool stays full of buffers in flight past a 5 s
// wait: DXGI_ERROR_WAS_STILL_DRAWING), the title's lists run with the backing's last upload. The
// buffer is marked stale until a later refresh succeeds; the HRESULT is logged (the first few times)
// and kept in the buffer's info and the totals (refresh_failures, last_refresh_hr).
//
// A copied buffer that a recorded copy writes (list CopyBufferRegion or CopyResource, list_copy.inl)
// keeps the GPU's bytes once that recording is submitted: it is no longer refreshed. The title memory
// never receives those bytes, and a later refresh that spans them would copy the title's older bytes
// over them. This is what the default backing does for such a buffer; info and totals report it
// (gpu_written). Until the submission the write is pending on the recording, with the texture writes
// of list_copy.inl: a Reset or a failed Close drops it, and the buffer stays refreshed. The recording's
// own submission still refreshes it first (its queue waits for that refresh, then the copy runs).
//
// Fixtures only:
//   X2W_TEST_EXISTING_HEAPS_UNSUPPORTED=1  the feature query reports no support (an adapter such as
//                                          Intel Gen9 reports EXISTING_HEAPS = 0).
//   X2W_TEST_LINEAR_BUFFER_MUTANT=<name>   removes one guarantee, for negative controls:
//     skip-refresh     the copy is taken at placement (bind) and never refreshed at submit
//     reuse-early      a staging buffer is reused without waiting for its fence
//     copy-gpu-writes  the copy path accepts a GPU-written use
//     ignore-feature   sharing is attempted although the feature query reports no support
//     refusal-silent   a kept default backing is not reported (no log text, info or totals)
//     hold-lock        the wait for a full staging pool keeps mutex_ held (other callers block)
//     skip-consumer-wait  submissions do not Wait for the internal fence (consumers race refreshes)
//     refresh-gpu-written  a copied buffer that a recorded copy wrote is still refreshed
//     gpuw-at-record   the buffer stops being refreshed when the copy is recorded, not when it is submitted
//     skip-producer-wait  a refresh does not Wait for earlier submissions (it may overwrite a backing in use)
//     publish-out-of-order  an older refresh's span is applied to `uploaded` after a newer one's
//     copy-after-retire  the title memory is not probed again right before it is staged (the fault guard
//                      then catches the read of a released buffer's freed pages)
//     accept-unsafe-refresh  a refresh with no safe completion order is enqueued without waiting for the
//                      blocked queue instead of refusing the submission (its copy may overlap that read)
//     unlocked-title-wait  queue slot 17 does not take the queue's order lock (a submission on another
//                      thread can take its user value between the desktop Wait and its record)
//   x2w_d3d12_linear_buffer_test_pause      parks the next refresh at one point (before staging, after
//                                          staging, after its copy executed), so a fixture can order two
//                                          threads' refreshes or release a buffer while one is parked;
//                                          or the next title Wait between the desktop Wait and its record
//   x2w_d3d12_linear_buffer_internal_queue  returns the internal queue, so a fixture can hold refreshes
//   X2W_TEST_LINEAR_STAGING_LIMIT=<n>      the staging pool holds n buffers instead of 256
//   X2W_TEST_LINEAR_REFRESH_FAIL=<n>       staging requests 2, 2 + n, 2 + 2n, ... fail with E_OUTOFMEMORY

#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

#include "startup_stubs/linear_buffer_report.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/memory_srv_buffer.h"
#include "startup_stubs/placed_buffer_range.h"
#include "startup_stubs/resource_budget.h"

namespace x2w_startup {

enum class LinearBufferMode { off, automatic, copy };

inline LinearBufferMode linear_buffer_mode()
{
    static const LinearBufferMode mode = [] {
        const DWORD saved = GetLastError();
        char value[16]{};
        const DWORD n = GetEnvironmentVariableA("X2W_D3D12_LINEAR_BUFFERS", value, sizeof(value));
        SetLastError(saved);
        if (!n || n >= sizeof(value)) return LinearBufferMode::off;
        if (std::strcmp(value, "1") == 0) return LinearBufferMode::automatic;
        if (_stricmp(value, "copy") == 0) return LinearBufferMode::copy;
        return LinearBufferMode::off;
    }();
    return mode;
}

namespace linear_detail {
inline bool mutant(const char* name)
{
    static char value[32] = {};
    static const bool read = [] {
        const DWORD saved = GetLastError();
        const DWORD n = GetEnvironmentVariableA("X2W_TEST_LINEAR_BUFFER_MUTANT", value, sizeof(value));
        if (n >= sizeof(value)) value[0] = 0;
        SetLastError(saved);
        return true;
    }();
    (void)read;
    return value[0] && std::strcmp(value, name) == 0;
}
inline bool existing_heaps_forced_off()
{
    static const bool on = x2w_local::env_on("X2W_TEST_EXISTING_HEAPS_UNSUPPORTED");
    return on;
}
inline unsigned refresh_fault_period()
{
    static const unsigned period = [] {
        const DWORD saved = GetLastError();
        char value[16]{};
        const DWORD n = GetEnvironmentVariableA("X2W_TEST_LINEAR_REFRESH_FAIL", value, sizeof(value));
        SetLastError(saved);
        return n && n < sizeof(value) ? static_cast<unsigned>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    return period;
}
inline bool submit_fence_forced_off()
{
    static const bool on = GetEnvironmentVariableW(L"X2W_TEST_SUBMIT_FENCE_FAIL", nullptr, 0) != 0;
    return on;
}
// Copies title memory that may have been freed or unmapped since it was probed: false on an access
// violation or an in-page error instead of a fault in the DLL.
inline bool guarded_copy(void* to, const void* from, std::size_t bytes) noexcept
{
    return x2w_local::guarded_memcpy(to, from, bytes, false);
}
inline D3D12_RESOURCE_DESC buffer(UINT64 width, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = width;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}
}  // namespace linear_detail

class LinearBuffers {
    struct Staging;
    struct RefreshList;

public:
    enum Path : std::uint32_t { shared_path = 1, copied_path = 2, kept_default_path = 3 };
    enum Use : std::uint32_t { gpu_read = 1, gpu_written = 2 };

    // The kind this file handles. Anything else keeps the default backing.
    static bool eligible(const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES initial_state)
    {
        return linear_buffer_mode() != LinearBufferMode::off && desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
               (desc.Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR || desc.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN) &&
               desc.Width && initial_state == D3D12_RESOURCE_STATE_COMMON;
    }

    // The refusal recorded for a placement that another owner takes: the XMem lease path, which takes
    // the default (copied) placements of XMem pages whose console flags are 0 while it is on.
    static constexpr HRESULT lease_deferral = HRESULT_FROM_WIN32(ERROR_BUSY);
    static HRESULT defer(bool uav, X2wLinearBufferInfo* report)
    {
        *report = X2wLinearBufferInfo{};
        report->use = uav ? gpu_written : gpu_read;
        report->feature_hr = S_FALSE;
        report->share_hr = S_FALSE;
        report->path = kept_default_path;
        report->refusal_hr = lease_deferral;
        return lease_deferral;
    }

    // False under the refusal-silent mutant: a kept default backing is then not reported anywhere.
    static bool report_refusals() { return !linear_detail::mutant("refusal-silent"); }

    // Slot 133 for an eligible buffer. range is measure_placed_buffer's answer for (address, width) and
    // uav says whether the desktop flags keep ALLOW_UNORDERED_ACCESS. On success *out holds the
    // desktop resource and *shared whether it aliases the title memory; a shared one's heap budget
    // (X2W_D3D12_BUDGET, or null) is in *backing_budget. A failure is a refusal: the caller keeps the
    // default backing and passes *report (path kept_default_path) to kept_default().
    HRESULT place(ID3D12Device* device, void* address, UINT64 width, bool uav, const PlacedBufferRange& range,
                  ID3D12Resource** out, bool* shared, x2w_budget::Ticket** backing_budget, X2wLinearBufferInfo* report)
    {
        *out = nullptr;
        *shared = false;
        *backing_budget = nullptr;
        std::lock_guard lock(mutex_);
        ++totals_.placements;
        Record record{};
        record.info.use = uav ? gpu_written : gpu_read;
        record.info.feature_hr = S_FALSE;
        record.info.share_hr = S_FALSE;
        PlacedBufferRange local_range = range;
        UINT64 offset = static_cast<UINT64>(static_cast<char*>(address) - static_cast<char*>(local_range.base));
        ID3D12Resource* resource = nullptr;
        if (linear_buffer_mode() == LinearBufferMode::automatic) {
            static const bool cap_existing_heap = x2w_local::env_on("X2W_D3D12_CAP_EXISTING_HEAP");
            record.info.feature_hr = feature(device);
            const bool allowed = SUCCEEDED(record.info.feature_hr) || linear_detail::mutant("ignore-feature");
            if (allowed && !local_range.covered && !local_range.region_info && local_range.type == MEM_PRIVATE &&
                can_commit_lazy_placed_buffer(address, width, cap_existing_heap ? unplaced_max_span : 0)) {
                x2w_local::commit_lazy_block(address);
                if (measure_placed_buffer(address, width, local_range))
                    offset = static_cast<UINT64>(static_cast<char*>(address) - static_cast<char*>(local_range.base));
                else
                    local_range.covered = false;
            }
            if (!allowed)
                record.info.share_hr = record.info.feature_hr;
            else if (!local_range.covered || (offset & 0xFFFF))
                record.info.share_hr = E_INVALIDARG;  // decided without D3D12: each failed open throws in D3D12Core
            else if (local_range.type == MEM_MAPPED)
                record.info.share_hr = HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);  // an open heap would pin the view
            else if (local_range.type != MEM_PRIVATE)
                record.info.share_hr = E_INVALIDARG;
            else if (cap_existing_heap && !unplaced_existing_heap_allowed(allocation_bytes(local_range.base)))
                record.info.share_hr = HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);
            else
                record.info.share_hr = share(device, local_range, offset, width, uav, &resource,
                                             &record.info.heap_generation, backing_budget);
        }
        HRESULT hr = S_OK;
        if (SUCCEEDED(record.info.share_hr) && record.info.share_hr != S_FALSE) {
            record.info.path = shared_path;
            ++totals_.shared;
        } else if (uav && !linear_detail::mutant("copy-gpu-writes")) {
            hr = record.info.share_hr == S_FALSE ? DXGI_ERROR_UNSUPPORTED : record.info.share_hr;
        } else if (const HRESULT internal = linear_detail::submit_fence_forced_off() ? E_NOT_VALID_STATE : internal_ready(device);
                   FAILED(internal)) {
            hr = internal;
        } else {
            const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
            const auto desc = linear_detail::buffer(width, uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                                               : D3D12_RESOURCE_FLAG_NONE);
            hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                 nullptr, IID_PPV_ARGS(&resource));
            if (SUCCEEDED(hr)) {
                try {
                    record.copy = std::make_shared<Copied>();
                    record.copy->title = static_cast<const unsigned char*>(address);
                    record.copy->bytes = width;
                    if (linear_detail::mutant("skip-refresh")) {
                        // Mutant: the bytes at placement are what every submission gets.
                        const std::size_t readable = x2w_local::probe_readable(reinterpret_cast<std::uintptr_t>(address),
                                                                               static_cast<std::size_t>(width));
                        record.copy->bound.assign(record.copy->title, record.copy->title + readable);
                        record.copy->bound.resize(static_cast<std::size_t>(width));
                    }
                } catch (const std::bad_alloc&) {
                    resource->Release();
                    resource = nullptr;
                    hr = E_OUTOFMEMORY;
                }
            }
            if (SUCCEEDED(hr)) {
                record.info.path = copied_path;
                ++totals_.copied;
            }
        }
        if (SUCCEEDED(hr)) {
            record.info.backing = resource;
            try {
                records_.emplace(resource, std::move(record));
            } catch (const std::bad_alloc&) {
                hr = E_OUTOFMEMORY;
                --(record.info.path == shared_path ? totals_.shared : totals_.copied);
            }
        }
        if (FAILED(hr)) {
            if (resource) resource->Release();
            *backing_budget = nullptr;
            record.info.path = kept_default_path;
            record.info.refusal_hr = hr;
            record.info.backing = nullptr;
            if (report) *report = record.info;
            return hr;
        }
        const auto& stored = records_.at(resource).info;
        if (report) *report = stored;
        const bool is_shared = stored.path == shared_path;
        *out = resource;
        *shared = is_shared;
        return S_OK;
    }

    // Slot 133 kept the default backing for a refused placement: record why, for this resource.
    void kept_default(ID3D12Resource* resource, const X2wLinearBufferInfo& report)
    {
        if (!report_refusals()) return;
        std::lock_guard lock(mutex_);
        ++totals_.kept_default;
        totals_.last_refusal_hr = report.refusal_hr;
        try {
            Record record{};
            record.info = report;
            record.info.path = kept_default_path;
            record.info.backing = resource;
            records_.insert_or_assign(resource, std::move(record));
        } catch (const std::bad_alloc&) {
        }
    }

    // A submitted recording writes the backing of `resource` (list_copy.inl, after its ExecuteCommandLists):
    // if it is a copied buffer, it keeps the GPU's bytes from now on and is no longer refreshed from the
    // title memory.
    void note_gpu_write(ID3D12Resource* resource)
    {
        if (linear_detail::mutant("refresh-gpu-written")) return;
        std::lock_guard lock(mutex_);
        const auto found = records_.find(resource);
        if (found == records_.end() || !found->second.copy || found->second.copy->gpu_written) return;
        found->second.copy->gpu_written = true;
        ++totals_.gpu_written;
        if (x2w_local::first_calls(gpu_write_logged_, 8))
            x2w_local::logf("x2w-d3d12 linear buffer %p: a recorded copy writes it; it keeps the GPU's bytes and is no "
                            "longer refreshed from title memory", static_cast<void*>(resource));
    }

    // The resource never reached the title and is about to be released.
    void discard(ID3D12Resource* resource)
    {
        std::lock_guard lock(mutex_);
        records_.erase(resource);
    }

    // The title released the resource: its memory is not read again. The desktop backing stays (as
    // every wrapper's does); a shared one keeps its heap.
    void retire(ID3D12Resource* resource)
    {
        std::lock_guard lock(mutex_);
        const auto found = records_.find(resource);
        if (found == records_.end() || !found->second.copy) return;
        found->second.copy->live = false;
        std::vector<unsigned char>().swap(found->second.copy->uploaded);
    }

    // Before the title's lists are executed on queue (the contract's copy point): refreshes every live
    // copied buffer whose title bytes changed, on the internal queue, then makes queue wait for every
    // refresh of a live copied buffer that has not landed yet. A buffer whose refresh fails keeps its last
    // upload and is marked stale; the submission still runs. False: the refresh has no safe completion
    // order (see the producer wait above) and the submission is refused: nothing was enqueued, and the
    // caller must neither execute the title's lists nor call after_submit or commit their GPU writes.
    //
    // mutex_ is held only to read and publish state: the work is snapshotted under it, then staging
    // is taken (and, with the pool full, its fence waited for), filled, recorded and executed without
    // it, and the results are published under it again. A release() or another queue's submission
    // therefore never waits behind a fence wait or ExecuteCommandLists here. internal_mutex_ only
    // keeps the internal queue's ExecuteCommandLists and Signal in fence-value order (and the refusal
    // decision next to them).
    [[nodiscard]] bool before_submit(ID3D12CommandQueue* queue)
    {
        struct Work {
            ID3D12Resource* dest;
            std::shared_ptr<Copied> copy;
            const unsigned char* source;
            UINT64 offset, bytes;
            Staging* staging = nullptr;
            HRESULT hr = S_OK;
            bool unreadable = false;  // the title memory was gone when it was to be staged
            bool faulted = false;     // reading it faulted (caught)
        };
        std::vector<Work> work;
        {
            std::lock_guard lock(mutex_);
            ++totals_.submissions;
            bool any = false;
            for (auto& [resource, record] : records_) any |= record.copy && record.copy->live;
            if (!any || !internal_queue_) return true;
            for (auto& [resource, record] : records_) {
                const std::shared_ptr<Copied>& copy = record.copy;
                if (!copy || !copy->live || copy->gpu_written) continue;
                const std::size_t bytes = static_cast<std::size_t>(copy->bytes);
                const unsigned char* source = copy->bound.empty() ? copy->title : copy->bound.data();
                if (copy->bound.empty() && x2w_local::probe_readable(reinterpret_cast<std::uintptr_t>(source), bytes) < bytes) {
                    if (x2w_local::first_calls(unreadable_logged_, 4))
                        x2w_local::logf("x2w-d3d12 linear buffer %p: title memory %p is no longer readable; its backing "
                                        "keeps the last upload", static_cast<void*>(resource), static_cast<const void*>(source));
                    ++totals_.unreadable;
                    continue;
                }
                std::size_t first = 0, last = bytes;
                if (copy->uploaded.size() == bytes) {
                    if (std::memcmp(copy->uploaded.data(), source, bytes) == 0) {
                        copy->stale = false;
                        continue;
                    }
                    while (first < bytes && copy->uploaded[first] == source[first]) ++first;
                    while (last > first && copy->uploaded[last - 1] == source[last - 1]) --last;
                    first &= ~std::size_t{3};
                    last = (std::min)(bytes, (last + 3) & ~std::size_t{3});
                }
                try {
                    work.push_back({resource, copy, source, first, last - first});
                } catch (const std::bad_alloc&) {
                    refresh_failed(resource, *copy, E_OUTOFMEMORY);
                }
            }
        }
        // This submission's own refreshes, including those of a buffer the title released meanwhile.
        bool refused = false;
        const UINT64 own = work.empty() ? 0 : refresh(work, queue, &refused);
        if (refused) return false;

        // Order this submission behind every refresh of a live copied buffer that may not have landed.
        UINT64 wait_for = own;
        {
            std::lock_guard lock(mutex_);
            for (auto& [resource, record] : records_)
                if (record.copy && record.copy->live && record.copy->refresh_value > wait_for) wait_for = record.copy->refresh_value;
            if (wait_for && internal_fence_->GetCompletedValue() < wait_for && !linear_detail::mutant("skip-consumer-wait"))
                ++totals_.consumer_waits;
            else
                wait_for = 0;
        }
        if (wait_for) {
            const HRESULT hr = queue->Wait(internal_fence_, wait_for);
            if (FAILED(hr) && x2w_local::first_calls(wait_logged_, 4))
                x2w_local::logf("x2w-d3d12 linear buffers: queue %p could not wait for refresh 0x%llx (0x%08lx)",
                                static_cast<void*>(queue), static_cast<unsigned long long>(wait_for), static_cast<unsigned long>(hr));
        }
        return true;
    }

    // After the title's lists were executed on `queue` (a queue wrapper): signals that queue's user fence,
    // so a later refresh can be ordered behind this submission (the producer wait). Nothing before the
    // first copied placement: earlier submissions cannot name a copied backing that did not exist yet.
    // Not called for a refused submission: its lists did not run, so no user value stands for them. The
    // caller holds the queue's order lock from its ExecuteCommandLists through this call.
    void after_submit(ID3D12CommandQueue* queue)
    {
        std::lock_guard lock(mutex_);
        if (!internal_queue_) return;
        UserQueue* user = user_queue(queue);
        if (!user) return;
        if (!user->fence && !user->untracked) {
            ID3D12Device* device = nullptr;
            HRESULT hr = queue->GetDevice(IID_PPV_ARGS(&device));
            if (SUCCEEDED(hr)) {
                hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&user->fence));
                device->Release();
            }
            if (FAILED(hr)) {
                user->fence = nullptr;
                x2w_local::logf("x2w-d3d12 linear buffers: queue %p has no user fence (0x%08lx); refreshes cannot wait for "
                                "its submissions", static_cast<void*>(queue), static_cast<unsigned long>(hr));
            }
        }
        if (!user->fence || FAILED(queue->Signal(user->fence, user->value + 1))) {
            // This submission cannot be waited for: a refresh that would have to wait for it is refused until
            // the queue's next submission has a value (which then covers this one).
            user->untracked = true;
            return;
        }
        user->untracked = false;
        ++user->value;
    }

    // The title enqueued Wait(fence, value) on `queue` (queue slot 17, already forwarded). Recorded until the
    // fence reaches the value: a refresh that would have to wait for a submission behind it is refused (see
    // the producer wait). The caller holds the queue's order lock from the desktop Wait through this call.
    void note_title_wait(ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64 value)
    {
        pause_at(pause_title_wait);
        if (!fence || fence->GetCompletedValue() >= value) return;
        std::lock_guard lock(mutex_);
        UserQueue* user = user_queue(queue);
        if (!user) return;
        prune_waits(*user);
        try {
            user->waits.push_back({fence, value, user->value + 1});
            fence->AddRef();
        } catch (const std::bad_alloc&) {
            // Not remembered with its fence: the queue's submissions from the next one on count as blocked until
            // that next one completes (its user fence value passes the unknown wait).
            if (!user->lost_wait_before) user->lost_wait_before = user->value + 1;
        }
    }

    // A queue's order lock, held until destroyed or release() (an empty one holds nothing).
    class QueueOrder {
    public:
        QueueOrder() = default;
        explicit QueueOrder(SRWLOCK* lock) : lock_(lock)
        {
            if (lock_) AcquireSRWLockExclusive(lock_);
        }
        QueueOrder(QueueOrder&& other) noexcept : lock_(other.lock_) { other.lock_ = nullptr; }
        QueueOrder(const QueueOrder&) = delete;
        QueueOrder& operator=(const QueueOrder&) = delete;
        QueueOrder& operator=(QueueOrder&&) = delete;
        ~QueueOrder() { release(); }
        void release()
        {
            if (lock_) ReleaseSRWLockExclusive(lock_);
            lock_ = nullptr;
        }

    private:
        SRWLOCK* lock_ = nullptr;
    };

    // Takes the order lock of `queue` (see the order lock above): queue slot 17 (title_wait) across the
    // desktop Wait and note_title_wait, a queue wrapper's submission across the title's ExecuteCommandLists
    // and after_submit. The caller holds none of this file's locks, nor draw_lock or allocator_lock. A queue
    // whose record cannot be stored shares one fallback lock.
    [[nodiscard]] QueueOrder queue_order(ID3D12CommandQueue* queue, bool title_wait)
    {
        // Mutant: slot 17 takes no lock, so a submission on another thread can take its user value between
        // the desktop Wait and its record.
        if (title_wait && linear_detail::mutant("unlocked-title-wait")) return QueueOrder{};
        SRWLOCK* lock = &order_fallback_;
        {
            std::lock_guard guard(mutex_);
            if (UserQueue* user = user_queue(queue)) lock = &user->order;
        }
        return QueueOrder(lock);
    }

    // Fixtures only (x2w_d3d12_linear_buffer_test_pause): the next refresh that reaches `point` sets
    // `reached`, then waits up to 20 s for `resume`, holding no lock. One shot per arming. The points:
    // before its staging (after the snapshot), after the title bytes are staged (before its copy is
    // executed), and after the copy is executed and signalled (before it is published). The last point
    // parks the next title Wait (queue slot 17) after the desktop Wait and before note_title_wait records
    // it, holding that queue's order lock (and nothing else). Null events disarm the point, so a fixture can
    // close its events when nothing came.
    enum PausePoint : unsigned { pause_before_stage, pause_after_stage, pause_after_execute, pause_title_wait, pause_points };
    bool arm_pause(unsigned point, HANDLE reached, HANDLE resume)
    {
        if (point >= pause_points) return false;
        std::lock_guard lock(pause_mutex_);
        pauses_[point] = reached && resume ? Pause{reached, resume} : Pause{};
        if (pauses_[point].reached) pauses_armed_.store(true);
        return true;
    }

    // The internal queue, AddRef'd, for fixtures that hold refreshes in flight; null before the first
    // copied placement.
    ID3D12CommandQueue* internal_queue()
    {
        std::lock_guard lock(mutex_);
        if (internal_queue_) internal_queue_->AddRef();
        return internal_queue_;
    }

    bool info(ID3D12Resource* resource, X2wLinearBufferInfo* out)
    {
        std::lock_guard lock(mutex_);
        const auto found = records_.find(resource);
        if (found == records_.end()) return false;
        *out = found->second.info;
        if (const Copied* copy = found->second.copy.get()) {
            out->refreshes = copy->refreshes;
            out->bytes_copied = copy->bytes_copied;
            out->last_staging = copy->last_staging;
            out->last_fence = copy->refresh_value;
            out->stale = copy->stale ? 1 : 0;
            out->gpu_written = copy->gpu_written ? 1 : 0;
            out->refresh_failures = copy->refresh_failures;
            out->last_refresh_hr = copy->last_refresh_hr;
        }
        return true;
    }

    void totals(X2wLinearBufferTotals* out)
    {
        std::lock_guard lock(mutex_);
        *out = totals_;
    }

private:
    struct Copied {
        const unsigned char* title = nullptr;
        UINT64 bytes = 0;
        bool live = true;
        std::vector<unsigned char> uploaded;  // what the backing holds (empty before the first refresh)
        std::vector<unsigned char> bound;     // skip-refresh mutant: the bytes at placement
        UINT64 refreshes = 0, bytes_copied = 0, last_staging = 0, refresh_failures = 0;
        UINT64 refresh_value = 0;  // internal fence value of the last enqueued refresh
        UINT64 published_value = 0;  // internal fence value of the newest refresh published into `uploaded`
        bool stale = false;
        bool gpu_written = false;  // a recorded copy wrote it: not refreshed any more
        HRESULT last_refresh_hr = S_OK;
    };
    struct Record {
        X2wLinearBufferInfo info{};
        std::shared_ptr<Copied> copy;  // shared: a submission's snapshot outlives a discard
    };
    struct Staging {
        ID3D12Resource* buffer = nullptr;
        unsigned char* mapped = nullptr;
        UINT64 bytes = 0, id = 0;
        ID3D12Fence* fence = nullptr;  // the internal fence, whose value completes the copy that read it
        UINT64 value = 0;              // 0: free; ~0: in use with no signalled value
    };
    struct RefreshList {
        ID3D12CommandAllocator* allocator = nullptr;
        ID3D12GraphicsCommandList* list = nullptr;
        UINT64 value = 0;
    };
    // A title Wait (queue slot 17) that had not completed when it was enqueued; `before` is the first user
    // value it precedes.
    struct TitleWait {
        ID3D12Fence* fence;
        UINT64 value, before;
    };
    // A queue wrapper's desktop queue, its user fence (signalled after each submission of the title's
    // lists), the title waits it may still sit behind and its order lock (queue_order). Kept for the
    // process, as the queues are, at a stable address (users_ is a deque that only grows).
    struct UserQueue {
        SRWLOCK order = SRWLOCK_INIT;
        ID3D12CommandQueue* queue = nullptr;
        ID3D12Fence* fence = nullptr;
        UINT64 value = 0;        // the latest submission's user fence value (0: none yet)
        bool untracked = false;  // the latest submission has no fence value: it cannot be waited for
        std::vector<TitleWait> waits;
        UINT64 lost_wait_before = 0;  // a title wait that could not be stored precedes this user value (0: none)
    };
    struct Heap {
        void* base;
        ID3D12Heap* heap;
        UINT64 generation, bytes;
        x2w_budget::Ticket* budget;  // the heap's budget ticket (X2W_D3D12_BUDGET), counted while a buffer uses it
    };
    // 256 staging buffers; X2W_TEST_LINEAR_STAGING_LIMIT=<n> (fixtures) lowers it, to make the pool fill.
    static std::size_t staging_limit()
    {
        static const std::size_t limit = [] {
            const DWORD saved = GetLastError();
            char value[16]{};
            const DWORD n = GetEnvironmentVariableA("X2W_TEST_LINEAR_STAGING_LIMIT", value, sizeof(value));
            SetLastError(saved);
            const unsigned long parsed = n && n < sizeof(value) ? std::strtoul(value, nullptr, 10) : 0;
            return parsed ? static_cast<std::size_t>(parsed) : std::size_t{256};
        }();
        return limit;
    }

    // A fixture's pause (arm_pause); nothing unless one was armed.
    void pause_at(unsigned point)
    {
        if (!pauses_armed_.load()) return;
        Pause pause{};
        {
            std::lock_guard lock(pause_mutex_);
            pause = pauses_[point];
            pauses_[point] = {};
        }
        if (!pause.reached) return;
        SetEvent(pause.reached);
        WaitForSingleObject(pause.resume, 20000);
    }

    // Under mutex_. The record of `queue`, created on first use; null if it cannot be stored.
    UserQueue* user_queue(ID3D12CommandQueue* queue)
    {
        for (UserQueue& user : users_)
            if (user.queue == queue) return &user;
        try {
            users_.emplace_back();
        } catch (const std::bad_alloc&) {
            return nullptr;
        }
        users_.back().queue = queue;
        return &users_.back();
    }
    // Under mutex_: forgets the title waits whose fences reached their values, and a lost one once the first
    // submission behind it completed.
    static void prune_waits(UserQueue& user)
    {
        if (user.lost_wait_before && user.fence && user.value >= user.lost_wait_before &&
            user.fence->GetCompletedValue() >= user.lost_wait_before)
            user.lost_wait_before = 0;
        auto& waits = user.waits;
        for (std::size_t i = 0; i < waits.size();) {
            if (waits[i].fence->GetCompletedValue() >= waits[i].value) {
                waits[i].fence->Release();
                waits[i] = waits.back();
                waits.pop_back();
            } else {
                ++i;
            }
        }
    }
    // Under mutex_: the (user fence, value) of each queue's latest submission that has not completed, which
    // a refresh's copy waits for, in *out. False when one of them has no safe completion order (behind an
    // unreached title wait, or without a fence value): the refresh must not be enqueued and the submission is
    // refused (see the producer wait above); *blocker and *why name the first such queue. bad_alloc
    // propagates: a refresh that cannot list what it waits for is not enqueued either.
    bool earlier_users(std::vector<std::pair<ID3D12Fence*, UINT64>>* out, ID3D12CommandQueue** blocker, const char** why)
    {
        bool safe = true;
        for (UserQueue& user : users_) {
            prune_waits(user);
            const bool pending = user.fence && user.value && user.fence->GetCompletedValue() < user.value;
            if (!pending && !user.untracked) continue;
            const char* reason = user.untracked ? "its latest submission has no user fence value" : nullptr;
            if (!reason && user.lost_wait_before && user.lost_wait_before <= user.value)
                reason = "its latest submission is behind a title Wait that could not be recorded";
            for (const TitleWait& wait : user.waits)
                if (!reason && wait.before <= user.value) reason = "its latest submission is behind a title Wait that has not completed";
            if (reason) {
                // Mutant: the blocked queue is not waited for and the refresh goes ahead (it may overlap that read).
                if (linear_detail::mutant("accept-unsafe-refresh")) continue;
                if (safe) {
                    *blocker = user.queue;
                    *why = reason;
                }
                safe = false;
                continue;
            }
            if (linear_detail::mutant("skip-producer-wait")) continue;
            out->emplace_back(user.fence, user.value);
        }
        return safe;
    }

    HRESULT feature(ID3D12Device* device)
    {
        if (feature_device_ != device) {
            feature_device_ = device;
            ++totals_.feature_queries;
            D3D12_FEATURE_DATA_EXISTING_HEAPS heaps{};
            const HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &heaps, sizeof(heaps));
            feature_hr_ = FAILED(hr) ? hr
                        : heaps.Supported && !linear_detail::existing_heaps_forced_off() ? S_OK : DXGI_ERROR_UNSUPPORTED;
            x2w_local::logf("x2w-d3d12 linear buffers (X2W_D3D12_LINEAR_BUFFERS=%s): D3D12_FEATURE_EXISTING_HEAPS -> 0x%08lx "
                            "supported=%d%s", linear_buffer_mode() == LinearBufferMode::copy ? "copy" : "1",
                            static_cast<unsigned long>(hr), heaps.Supported ? 1 : 0,
                            linear_detail::existing_heaps_forced_off() ? " (reported as unsupported: X2W_TEST_EXISTING_HEAPS_UNSUPPORTED)" : "");
        }
        return feature_hr_;
    }

    // A private range only (place() decides the rest). Its heap is opened once per allocation base and
    // cached: the open heap keeps the range from being released, so the base cannot name other memory.
    HRESULT share(ID3D12Device* device, const PlacedBufferRange& range, UINT64 offset, UINT64 width, bool uav,
                  ID3D12Resource** out, UINT64* generation, x2w_budget::Ticket** budget)
    {
        const Heap* entry = nullptr;
        for (const Heap& known : heaps_)
            if (known.base == range.base) entry = &known;
        if (entry) {
            ++totals_.heap_cache_hits;
        } else {
            ID3D12Device3* device3 = nullptr;
            ID3D12Heap* heap = nullptr;
            HRESULT hr = device->QueryInterface(IID_PPV_ARGS(&device3));
            if (SUCCEEDED(hr)) {
                x2w_local::commit_lazy_block(range.base);
                ++totals_.heap_opens;
                hr = device3->OpenExistingHeapFromAddress(range.base, IID_PPV_ARGS(&heap));
                device3->Release();
                if (FAILED(hr)) ++totals_.heap_open_failures;
            }
            if (FAILED(hr)) return hr;
            try {
                heaps_.push_back({range.base, heap, ++generation_, heap->GetDesc().SizeInBytes, x2w_budget::backing_heap(heap, 0)});
            } catch (const std::bad_alloc&) {
                heap->Release();
                return E_OUTOFMEMORY;
            }
            entry = &heaps_.back();
        }
        *generation = entry->generation;
        *budget = entry->budget;
        if (entry->bytes < offset || width > entry->bytes - offset) return E_INVALIDARG;
        const auto desc = linear_detail::buffer(width, static_cast<D3D12_RESOURCE_FLAGS>(
            (uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE) | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER));
        const HRESULT hr = device->CreatePlacedResource(entry->heap, offset, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(out));
        if (FAILED(hr)) ++totals_.place_failures;
        return hr;
    }

    // The internal refresh queue and its fence, created once (under mutex_, from place()).
    HRESULT internal_ready(ID3D12Device* device)
    {
        if (internal_queue_) return S_OK;
        if (internal_tried_) return internal_hr_;
        internal_tried_ = true;
        const D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_COPY};
        ID3D12CommandQueue* queue = nullptr;
        ID3D12Fence* fence = nullptr;
        internal_hr_ = device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue));
        if (SUCCEEDED(internal_hr_)) internal_hr_ = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (FAILED(internal_hr_)) {
            if (queue) queue->Release();
            x2w_local::logf("x2w-d3d12 linear buffers: the internal refresh queue could not be created (0x%08lx); copied "
                            "buffers keep the default backing", static_cast<unsigned long>(internal_hr_));
            return internal_hr_;
        }
        queue->SetName(L"x2w linear-buffer refresh");
        internal_device_ = device;
        internal_queue_ = queue;
        internal_fence_ = fence;
        return S_OK;
    }

    // Staging, the refresh list and the internal queue, without mutex_; then publish under it.
    // Returns the internal fence value of the refresh it enqueued (0: none). *refused: the refresh had no
    // safe completion order and nothing was enqueued or published (the submission on `queue` is refused).
    template <class Work>
    UINT64 refresh(std::vector<Work>& work, ID3D12CommandQueue* queue, bool* refused)
    {
        pause_at(pause_before_stage);
        std::vector<Staging*> taken;
        bool any_staged = false;
        for (Work& w : work) {
            w.hr = acquire_staging(w.bytes, taken, &w.staging);
            if (FAILED(w.hr)) continue;
            taken.push_back(w.staging);
            // The snapshot probed the title memory under mutex_, but the title may have released this buffer
            // since (any live buffer is snapshotted, not only those the lists read) and freed or unmapped its
            // pages: probe again right before reading, and read under a fault guard.
            const unsigned char* from = w.source + w.offset;
            const auto bytes = static_cast<std::size_t>(w.bytes);
            if (!linear_detail::mutant("copy-after-retire") &&
                x2w_local::probe_readable(reinterpret_cast<std::uintptr_t>(from), bytes) < bytes) {
                w.unreadable = true;
                w.hr = HRESULT_FROM_WIN32(ERROR_NOACCESS);
                continue;
            }
            if (!linear_detail::guarded_copy(w.staging->mapped, from, bytes)) {
                w.faulted = true;
                w.hr = HRESULT_FROM_WIN32(ERROR_NOACCESS);
                continue;
            }
            any_staged = true;
        }
        if (any_staged) pause_at(pause_after_stage);
        RefreshList* list = nullptr;
        HRESULT list_hr = S_OK;
        bool executed = false;
        UINT64 value = 0;
        if (any_staged) {
            list_hr = acquire_list(&list);
            if (SUCCEEDED(list_hr)) {
                // Buffers rest in COMMON: each copy promotes its destination to COPY_DEST on the internal
                // queue, and it decays to COMMON again when this ExecuteCommandLists completes.
                for (const Work& w : work)
                    if (SUCCEEDED(w.hr)) list->list->CopyBufferRegion(w.dest, w.offset, w.staging->buffer, 0, w.bytes);
                list_hr = list->list->Close();
            }
            if (SUCCEEDED(list_hr)) {
                std::lock_guard submit(internal_mutex_);
                // The producer wait: the copy runs after every earlier submission that may use a backing, or
                // the submission is refused when one of them has no safe completion order. Decided here, under
                // internal_mutex_, so no refresh can be enqueued between the decision and this copy.
                std::vector<std::pair<ID3D12Fence*, UINT64>> users;
                ID3D12CommandQueue* blocker = nullptr;
                const char* why = "";
                bool safe = true;
                {
                    std::lock_guard lock(mutex_);
                    try {
                        safe = earlier_users(&users, &blocker, &why);
                    } catch (const std::bad_alloc&) {
                        list_hr = E_OUTOFMEMORY;  // the refresh fails (stale); nothing is enqueued without its waits
                    }
                    if (SUCCEEDED(list_hr) && !safe) {
                        // Nothing was executed: the list and every staging buffer are free again, and no buffer's
                        // state changes (the next submission compares against the same upload record).
                        list->value = 0;
                        for (Work& w : work)
                            if (w.staging) w.staging->value = 0;
                        ++totals_.unsafe_refresh_refusals;
                        if (x2w_local::first_calls(unsafe_logged_, 8))
                            x2w_local::logf("x2w-d3d12 linear buffers: submission on queue %p refused: its refresh would have to "
                                            "wait for queue %p (%s), so it has no safe completion order; none of its lists ran",
                                            static_cast<void*>(queue), static_cast<void*>(blocker), why);
                        *refused = true;
                        return 0;
                    }
                    if (SUCCEEDED(list_hr)) totals_.producer_waits += users.size();
                }
                if (SUCCEEDED(list_hr)) {
                    for (const auto& [fence, user_value] : users) {
                        const HRESULT hr = internal_queue_->Wait(fence, user_value);
                        if (FAILED(hr) && x2w_local::first_calls(wait_logged_, 4))
                            x2w_local::logf("x2w-d3d12 linear buffers: the refresh queue could not wait for submission 0x%llx (0x%08lx)",
                                            static_cast<unsigned long long>(user_value), static_cast<unsigned long>(hr));
                    }
                    ID3D12CommandList* lists[] = {list->list};
                    internal_queue_->ExecuteCommandLists(1, lists);
                    executed = true;
                    list_hr = internal_queue_->Signal(internal_fence_, internal_value_ + 1);
                    if (SUCCEEDED(list_hr)) value = ++internal_value_;
                }
            }
        }
        if (executed) pause_at(pause_after_execute);

        // Publish under the lock. Two submissions on different threads can snapshot one buffer against the
        // same `uploaded`, stage the title's bytes at different times (X, then the CPU writes Y, then Y),
        // execute in one order (internal_mutex_: Y gets the lower value, X the higher, so the backing ends
        // with X) and publish in the other. Applying both publishes in lock order would leave `uploaded`
        // saying Y while the backing holds X, and no later submission would refresh it. So a buffer
        // remembers the internal value of its last applied publish: a newer one applies its span; an older
        // one arriving after it drops `uploaded` (the backing holds the newer copy's bytes where the spans
        // overlap and the older one's elsewhere), and the next submission copies the whole buffer. Every
        // consumer waits for the higher of the two fence values.
        std::lock_guard lock(mutex_);
        // Executed without a fence value: what it read can never be proven free, so it stays in use.
        const UINT64 held = executed ? ~UINT64{0} : 0;
        if (list) list->value = value ? value : (FAILED(list_hr) ? held : value);
        for (Work& w : work) {
            if (w.unreadable || w.faulted) {
                w.staging->value = 0;  // never read by a copy: free again
                if (w.faulted) {
                    ++totals_.copy_faults;
                    refresh_failed(w.dest, *w.copy, w.hr);
                } else {
                    ++totals_.unreadable;
                    if (x2w_local::first_calls(unreadable_logged_, 4))
                        x2w_local::logf("x2w-d3d12 linear buffer %p: title memory %p is no longer readable; its backing "
                                        "keeps the last upload", static_cast<void*>(w.dest), static_cast<const void*>(w.source));
                }
                continue;
            }
            const HRESULT hr = FAILED(w.hr) ? w.hr : list_hr;
            if (FAILED(hr)) {
                if (w.staging) w.staging->value = held;
                refresh_failed(w.dest, *w.copy, hr);
                continue;
            }
            w.staging->fence = internal_fence_;
            w.staging->value = value;
            Copied& copy = *w.copy;
            if (value > copy.refresh_value) copy.refresh_value = value;
            if (copy.live && value < copy.published_value && !linear_detail::mutant("publish-out-of-order")) {
                copy.uploaded.clear();  // an older copy after a newer one: unknown, the next submission copies it all
                ++totals_.reordered_publishes;
            } else if (copy.live) {
                // The backing holds these bytes once `value` completes; every consumer waits for it. An unknown
                // `uploaded` (empty) is set only by a copy of the whole buffer.
                const bool whole = w.offset == 0 && w.bytes == copy.bytes;
                try {
                    if (copy.uploaded.size() != copy.bytes && (whole || linear_detail::mutant("publish-out-of-order")))
                        copy.uploaded.assign(static_cast<std::size_t>(copy.bytes), 0);
                    if (copy.uploaded.size() == copy.bytes)
                        std::memcpy(copy.uploaded.data() + w.offset, w.staging->mapped, static_cast<std::size_t>(w.bytes));
                } catch (const std::bad_alloc&) {
                    copy.uploaded.clear();  // the next submission copies the whole buffer again
                }
            }
            if (value > copy.published_value) copy.published_value = value;
            copy.stale = false;
            ++copy.refreshes;
            copy.bytes_copied += w.bytes;
            copy.last_staging = w.staging->id;
            ++totals_.refreshes;
            totals_.bytes_copied += w.bytes;
        }
        return value;
    }

    static bool complete(const Staging& staging)
    {
        return staging.value == 0 ||
               (staging.value != ~UINT64{0} && staging.fence && staging.fence->GetCompletedValue() >= staging.value);
    }

    // Takes a staging buffer of at least `bytes` that no submission still reads, for this submission
    // alone (`taken` are the ones it already holds). Called without mutex_: the lock is taken only to
    // choose or publish a buffer; creating one and waiting for a fence happen outside it.
    HRESULT acquire_staging(UINT64 bytes, const std::vector<Staging*>& taken, Staging** out)
    {
        const bool early = linear_detail::mutant("reuse-early");
        {
            std::lock_guard lock(mutex_);
            if (const unsigned period = linear_detail::refresh_fault_period();
                period && ++staging_requests_ % period == 2 % period)
                return E_OUTOFMEMORY;  // fixture fault (X2W_TEST_LINEAR_REFRESH_FAIL)
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool create = false;
            ID3D12Fence* fence = nullptr;
            UINT64 value = 0;
            {
                std::lock_guard lock(mutex_);
                Staging* best = nullptr;
                for (const auto& staging : staging_) {
                    if (staging->bytes < bytes || (!early && !complete(*staging))) continue;
                    if (std::find(taken.begin(), taken.end(), staging.get()) != taken.end()) continue;
                    // The reuse-early mutant takes the most recently used buffer, in flight or not.
                    if (!best || (early ? staging->id > best->id : staging->bytes < best->bytes)) best = staging.get();
                }
                if (best) {
                    ++totals_.staging_reused;
                    best->value = ~UINT64{0};
                    *out = best;
                    return S_OK;
                }
                if (staging_.size() + creating_ < staging_limit()) {
                    ++creating_;
                    create = true;
                } else {
                    // The pool is full of buffers in flight: the oldest with a signalled value is waited for.
                    const Staging* oldest = nullptr;
                    for (const auto& staging : staging_)
                        if (staging->value != ~UINT64{0} && staging->fence && (!oldest || staging->value < oldest->value))
                            oldest = staging.get();
                    if (oldest) {
                        fence = oldest->fence;
                        value = oldest->value;
                    }
                }
            }
            if (create) return create_staging(bytes, out);
            if (!fence || attempt) break;
            // The hold-lock mutant waits with mutex_ held, as before the review: other callers block.
            std::unique_lock<std::mutex> held;
            if (linear_detail::mutant("hold-lock")) held = std::unique_lock<std::mutex>(mutex_);
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            const bool reached = event && SUCCEEDED(fence->SetEventOnCompletion(value, event)) &&
                                 WaitForSingleObject(event, 5000) == WAIT_OBJECT_0;
            if (event) CloseHandle(event);
            if (!reached) return DXGI_ERROR_WAS_STILL_DRAWING;
        }
        return DXGI_ERROR_WAS_STILL_DRAWING;
    }

    // Called without mutex_ after acquire_staging reserved a place in the pool (creating_).
    HRESULT create_staging(UINT64 bytes, Staging** out)
    {
        auto staging = std::make_unique<Staging>();
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
        const UINT64 rounded = (bytes + 0xFFFF) & ~UINT64{0xFFFF};
        const UINT64 size = rounded ? rounded : 0x10000;
        const auto desc = linear_detail::buffer(size, D3D12_RESOURCE_FLAG_NONE);
        HRESULT hr = internal_device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                               nullptr, IID_PPV_ARGS(&staging->buffer));
        staging->bytes = size;
        // The pool's staging buffers live for the process: counted as upload memory (X2W_D3D12_BUDGET).
        if (SUCCEEDED(hr)) x2w_budget::committed(internal_device_, staging->buffer, D3D12_HEAP_TYPE_UPLOAD);
        void* mapped = nullptr;
        const D3D12_RANGE none{0, 0};
        if (SUCCEEDED(hr)) hr = staging->buffer->Map(0, &none, &mapped);
        std::lock_guard lock(mutex_);
        --creating_;
        if (FAILED(hr)) {
            if (staging->buffer) staging->buffer->Release();
            return hr;
        }
        staging->mapped = static_cast<unsigned char*>(mapped);
        staging->id = ++staging_ids_;
        staging->value = ~UINT64{0};
        try {
            staging_.push_back(std::move(staging));
        } catch (const std::bad_alloc&) {
            return E_OUTOFMEMORY;  // the unique_ptr releases nothing: the buffer is leaked, not reused
        }
        ++totals_.staging_created;
        *out = staging_.back().get();
        return S_OK;
    }

    // A refresh list for the internal queue whose last use has completed, or a new one, reset and ours
    // alone. Called without mutex_; the lock covers only the choice and the pool.
    HRESULT acquire_list(RefreshList** out)
    {
        RefreshList* found = nullptr;
        {
            std::lock_guard lock(mutex_);
            const UINT64 completed = internal_fence_->GetCompletedValue();
            for (const auto& list : internal_lists_) {
                if (list->value == ~UINT64{0} || list->value > completed) continue;
                list->value = ~UINT64{0};
                found = list.get();
                break;
            }
        }
        if (found) {
            HRESULT hr = found->allocator->Reset();
            if (SUCCEEDED(hr)) hr = found->list->Reset(found->allocator, nullptr);
            *out = found;  // on failure the caller frees it again under the lock
            return hr;
        }
        auto list = std::make_unique<RefreshList>();
        HRESULT hr = internal_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&list->allocator));
        if (SUCCEEDED(hr))
            hr = internal_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, list->allocator, nullptr,
                                                     IID_PPV_ARGS(&list->list));
        if (FAILED(hr)) {
            if (list->list) list->list->Release();
            if (list->allocator) list->allocator->Release();
            return hr;
        }
        list->value = ~UINT64{0};
        std::lock_guard lock(mutex_);
        try {
            internal_lists_.push_back(std::move(list));
        } catch (const std::bad_alloc&) {
            return E_OUTOFMEMORY;
        }
        *out = internal_lists_.back().get();
        return S_OK;
    }

    // The submission runs without this buffer's refresh: its backing keeps the last upload.
    void refresh_failed(ID3D12Resource* resource, Copied& copy, HRESULT hr)
    {
        copy.stale = true;
        ++copy.refresh_failures;
        copy.last_refresh_hr = hr;
        ++totals_.refresh_failures;
        totals_.last_refresh_hr = hr;
        if (x2w_local::first_calls(refresh_logged_, 8))
            x2w_local::logf("x2w-d3d12 linear buffers: refresh of %p failed 0x%08lx; the submission runs with its last "
                            "upload (stale)", static_cast<void*>(resource), static_cast<unsigned long>(hr));
    }

    std::mutex mutex_;
    std::unordered_map<ID3D12Resource*, Record> records_;
    std::mutex internal_mutex_;  // the internal queue's ExecuteCommandLists and Signal, in value order
    ID3D12Device* internal_device_ = nullptr;
    ID3D12CommandQueue* internal_queue_ = nullptr;
    ID3D12Fence* internal_fence_ = nullptr;
    UINT64 internal_value_ = 0;  // under internal_mutex_
    bool internal_tried_ = false;
    HRESULT internal_hr_ = S_OK;
    std::vector<std::unique_ptr<RefreshList>> internal_lists_;
    std::vector<std::unique_ptr<Staging>> staging_;
    std::vector<Heap> heaps_;
    std::deque<UserQueue> users_;  // under mutex_; elements never move (their order locks are held outside it)
    SRWLOCK order_fallback_ = SRWLOCK_INIT;  // the order lock of a queue whose record could not be stored
    struct Pause {
        HANDLE reached = nullptr, resume = nullptr;
    };
    std::mutex pause_mutex_;  // fixtures' pauses (a leaf: nothing else is taken under it)
    Pause pauses_[pause_points];
    std::atomic<bool> pauses_armed_{false};
    ID3D12Device* feature_device_ = nullptr;
    HRESULT feature_hr_ = S_FALSE;
    UINT64 generation_ = 0, staging_ids_ = 0, staging_requests_ = 0;
    std::size_t creating_ = 0;  // staging buffers being created outside the lock
    X2wLinearBufferTotals totals_{};
    volatile LONG unreadable_logged_ = 0, wait_logged_ = 0, refresh_logged_ = 0, gpu_write_logged_ = 0, unsafe_logged_ = 0;
};

}  // namespace x2w_startup
