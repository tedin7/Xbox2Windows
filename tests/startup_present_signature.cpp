// Verifies the presented-picture signature (src/startup_stubs/present_signature_sampler.h) on WARP:
// the sampler copies only the sampled rows of a presented surface, publishes a grid equal to the CPU
// reduction of those rows, and the published signal changes once for two different pictures and
// never for repeated identical pictures (negative control). Also checks the sampling interval, that a
// bar of about 1 % of the frame between two base rows is missed by the 32 x 18 grid and caught by the
// fine 64 x 72 grid, prints the CPU-side cost per sampled present at 3840 x 2160, and checks that a
// sampler made with always = false records nothing until the block asks for samples.
//
// With "held_source" it runs the lifetime check under the D3D12 debug layer: the queue is stalled on
// a fence, the sampler records its copy, the caller releases its only reference to the surface,
// then the queue is released. The signature must still equal the picture, the sampler must drop its
// reference once the copy's fence has passed, and the debug layer must report no error. A queue whose
// Signal fails after submission must leave the sampler stranded, still holding the surface.
// With "unheld_control" it runs the same stall with a plain copy of its own that does not hold the
// surface (the defect the sampler avoids); it passes only if the debug layer reports that.
//
// usage: startup_present_signature [held_source | unheld_control]
//        (exit 77 when no WARP adapter, or for the two modes no debug layer, is available)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "startup_stubs/input_script.h"
#include "startup_stubs/present_signature_sampler.h"

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what)
{
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

constexpr UINT width = 200, height = 120;

// Picture contents, 4 bytes a texel in the surface's byte order.
using Picture = std::vector<unsigned char>;
Picture gradient(bool inverted)
{
    Picture p(std::size_t{width} * height * 4);
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x) {
            unsigned char* t = &p[(std::size_t{y} * width + x) * 4];
            const auto v = static_cast<unsigned char>((x * 255) / (width - 1));
            t[0] = inverted ? 255 - v : v;
            t[1] = static_cast<unsigned char>((y * 255) / (height - 1));
            t[2] = inverted ? v : static_cast<unsigned char>(255 - v);
            t[3] = 255;
        }
    return p;
}

struct Gpu {
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12Fence* fence = nullptr;
    UINT64 value = 0;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    void idle()
    {
        queue->Signal(fence, ++value);
        if (fence->GetCompletedValue() < value) {
            fence->SetEventOnCompletion(value, event);
            WaitForSingleObject(event, 10000);
        }
    }
};

// A texture in COMMON holding `picture` (w x h texels).
ID3D12Resource* surface(Gpu& gpu, DXGI_FORMAT format, const Picture& picture, UINT w = width, UINT h = height)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    ID3D12Resource* texture = nullptr;
    if (FAILED(gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   IID_PPV_ARGS(&texture))))
        return nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* upload = nullptr;
    gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
    void* mapped = nullptr;
    upload->Map(0, nullptr, &mapped);
    for (UINT y = 0; y < h; ++y)
        std::memcpy(static_cast<unsigned char*>(mapped) + footprint.Offset + std::size_t{y} * footprint.Footprint.RowPitch,
                    &picture[std::size_t{y} * w * 4], std::size_t{w} * 4);
    upload->Unmap(0, nullptr);
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list));
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION to{}, from{};
    to.pResource = texture;
    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    from.pResource = upload;
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = footprint;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    list->ResourceBarrier(1, &barrier);
    list->Close();
    ID3D12CommandList* lists[] = {list};
    gpu.queue->ExecuteCommandLists(1, lists);
    gpu.idle();
    list->Release();
    allocator->Release();
    upload->Release();
    return texture;
}

// The grid the sampler must publish for `picture`, reduced on the CPU from the same rows.
void expected_grid(const Picture& picture, x2w_present_signature::Layout layout, std::uint8_t (&grid)[x2w_present_signature::cells])
{
    for (int band = 0; band < x2w_present_signature::cells_y; ++band)
        x2w_present_signature::reduce_row(&picture[std::size_t{x2w_present_signature::sampled_row(band, height)} * width * 4], width, layout,
                                          grid + band * x2w_present_signature::cells_x);
}

// Presents `surfaces` in order through a sampler with interval 1, letting the GPU finish between
// presents, then one empty present to publish the last copy. Returns the published grids.
std::vector<x2w_present_signature::Snapshot> present_all(Gpu& gpu, x2w_present_signature::Block* block, const std::vector<ID3D12Resource*>& surfaces)
{
    std::vector<x2w_present_signature::Snapshot> published;
    x2w_present_signature::Sampler sampler(1);
    LONG64 present = 0;
    const std::uint64_t start = static_cast<std::uint64_t>(block->sequence / 2);
    const auto record = [&] {
        x2w_present_signature::Snapshot s{};
        if (x2w_present_signature::read(block, s) && s.sequence > start && (published.empty() || s.sequence != published.back().sequence))
            published.push_back(s);
    };
    for (ID3D12Resource* s : surfaces) {
        sampler.present(gpu.device, gpu.queue, s, ++present, block);
        record();
        gpu.idle();
    }
    sampler.present(gpu.device, gpu.queue, nullptr, ++present, block);
    record();
    check(!sampler.failed(), "sampler setup");
    return published;
}

// Forwards every call to a real fence, except that GetCompletedValue reports UINT64_MAX once
// `removed` is set: the device-removal case the sampler has to survive without treating it as
// completion. Signal and SetEventOnCompletion still succeed, as they do while a device is being
// removed. For the destructor controls, `pending_observations`/`next` make each successive
// GetCompletedValue report a chosen value a fixed number of times and then the real fence's own
// value, so a transition across the destructor's wait is deterministic: the pre-wait
// observation sees pending, the wait runs on the real fence, the post-wait observation sees the
// completed (or, with `removed`, the removal sentinel) state. The real fence is the only object
// the driver ever sees.
class RemovedFence : public ID3D12Fence {
public:
    explicit RemovedFence(ID3D12Fence* real) : real_(real) {}
    bool removed = false;
    UINT64 pending_observations = 0;  // report `next` this many times, then the real value
    UINT64 next = 0;
    unsigned observations = 0;        // GetCompletedValue calls since construction
    unsigned event_registrations = 0;
    ID3D12Fence* real() const { return real_; }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override { return real_ ? real_->QueryInterface(riid, out) : E_FAIL; }
    ULONG STDMETHODCALLTYPE AddRef() override { return real_ ? real_->AddRef() : 0; }
    ULONG STDMETHODCALLTYPE Release() override { return real_ ? real_->Release() : 0; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID g, UINT* n, void* d) override { return real_ ? real_->GetPrivateData(g, n, d) : E_FAIL; }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID g, UINT n, const void* d) override { return real_ ? real_->SetPrivateData(g, n, d) : E_FAIL; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID g, const IUnknown* d) override { return real_ ? real_->SetPrivateDataInterface(g, d) : E_FAIL; }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR name) override { return real_ ? real_->SetName(name) : E_FAIL; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** out) override { return real_ ? real_->GetDevice(riid, out) : E_FAIL; }
    UINT64 STDMETHODCALLTYPE GetCompletedValue() override
    {
        ++observations;
        if (pending_observations) {
            --pending_observations;
            return next;
        }
        if (removed) return UINT64_MAX;
        return real_ ? real_->GetCompletedValue() : 0;
    }
    HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64 value, HANDLE event) override
    {
        ++event_registrations;
        return real_ ? real_->SetEventOnCompletion(value, event) : E_FAIL;
    }
    HRESULT STDMETHODCALLTYPE Signal(UINT64 value) override { return real_ ? real_->Signal(value) : E_FAIL; }

private:
    ID3D12Fence* real_;
};
// Forwards every call to a real queue, except that Signal fails once `fail_signal` is set: the
// "submitted, but no fence" case the sampler has to survive. With `observed`, Signal and Wait
// unwrap exactly our RemovedFence to its real fence, so a test fake never enters the driver.

class SignalFailingQueue : public ID3D12CommandQueue {
public:
    explicit SignalFailingQueue(ID3D12CommandQueue* real, RemovedFence* observed = nullptr)
        : real_(real), observed_(observed) {}
    bool fail_signal = false;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override { return real_->QueryInterface(riid, out); }
    ULONG STDMETHODCALLTYPE AddRef() override { return 2; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID g, UINT* n, void* d) override { return real_->GetPrivateData(g, n, d); }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID g, UINT n, const void* d) override { return real_->SetPrivateData(g, n, d); }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID g, const IUnknown* d) override { return real_->SetPrivateDataInterface(g, d); }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR name) override { return real_->SetName(name); }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** out) override { return real_->GetDevice(riid, out); }
    void STDMETHODCALLTYPE UpdateTileMappings(ID3D12Resource* r, UINT n, const D3D12_TILED_RESOURCE_COORDINATE* c,
                                              const D3D12_TILE_REGION_SIZE* s, ID3D12Heap* h, UINT rn, const D3D12_TILE_RANGE_FLAGS* f,
                                              const UINT* o, const UINT* cnt, D3D12_TILE_MAPPING_FLAGS fl) override
    {
        real_->UpdateTileMappings(r, n, c, s, h, rn, f, o, cnt, fl);
    }
    void STDMETHODCALLTYPE CopyTileMappings(ID3D12Resource* d, const D3D12_TILED_RESOURCE_COORDINATE* dc, ID3D12Resource* s,
                                            const D3D12_TILED_RESOURCE_COORDINATE* sc, const D3D12_TILE_REGION_SIZE* rs,
                                            D3D12_TILE_MAPPING_FLAGS f) override
    {
        real_->CopyTileMappings(d, dc, s, sc, rs, f);
    }
    void STDMETHODCALLTYPE ExecuteCommandLists(UINT n, ID3D12CommandList* const* lists) override { real_->ExecuteCommandLists(n, lists); }
    void STDMETHODCALLTYPE SetMarker(UINT m, const void* d, UINT n) override { real_->SetMarker(m, d, n); }
    void STDMETHODCALLTYPE BeginEvent(UINT m, const void* d, UINT n) override { real_->BeginEvent(m, d, n); }
    void STDMETHODCALLTYPE EndEvent() override { real_->EndEvent(); }
    HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence* fence, UINT64 value) override
    {
        if (fail_signal) return E_FAIL;
        // A test fake never enters the driver: unwrap exactly our RemovedFence to its real fence.
        if (observed_ && fence == observed_) fence = observed_->real();
        return real_->Signal(fence, value);
    }
    HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence* fence, UINT64 value) override
    {
        return real_->Wait(observed_ && fence == observed_ ? observed_->real() : fence, value);
    }
    HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64* f) override { return real_->GetTimestampFrequency(f); }
    HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64* g, UINT64* c) override { return real_->GetClockCalibration(g, c); }
#if defined(_MSC_VER) || !defined(_WIN32)
    D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override { return real_->GetDesc(); }
#else
    D3D12_COMMAND_QUEUE_DESC* STDMETHODCALLTYPE GetDesc(D3D12_COMMAND_QUEUE_DESC* out) override { return real_->GetDesc(out); }
#endif

private:
    ID3D12CommandQueue* real_;
    RemovedFence* observed_;
};

// Forwards every ID3D12Device method to a real device, except that CreateFence returns the
// RemovedFence above, so the fixture controls what GetCompletedValue reports.
class RemovalDevice : public ID3D12Device {
public:
    explicit RemovalDevice(ID3D12Device* real) : real_(real) {}
    RemovedFence& fence() { return fence_; }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override { return real_->QueryInterface(riid, out); }
    ULONG STDMETHODCALLTYPE AddRef() override { return 2; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID g, UINT* n, void* d) override { return real_->GetPrivateData(g, n, d); }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID g, UINT n, const void* d) override { return real_->SetPrivateData(g, n, d); }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID g, const IUnknown* d) override { return real_->SetPrivateDataInterface(g, d); }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR name) override { return real_->SetName(name); }
    // clang-format off
    UINT STDMETHODCALLTYPE GetNodeCount() override { return real_->GetNodeCount(); }
    HRESULT STDMETHODCALLTYPE CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC *desc, REFIID riid, void **command_queue) override { return real_->CreateCommandQueue(desc, riid, command_queue); }
    HRESULT STDMETHODCALLTYPE CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void **command_allocator) override { return real_->CreateCommandAllocator(type, riid, command_allocator); }
    HRESULT STDMETHODCALLTYPE CreateGraphicsPipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC *desc, REFIID riid, void **pipeline_state) override { return real_->CreateGraphicsPipelineState(desc, riid, pipeline_state); }
    HRESULT STDMETHODCALLTYPE CreateComputePipelineState(const D3D12_COMPUTE_PIPELINE_STATE_DESC *desc, REFIID riid, void **pipeline_state) override { return real_->CreateComputePipelineState(desc, riid, pipeline_state); }
    HRESULT STDMETHODCALLTYPE CreateCommandList(UINT node_mask, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator *command_allocator, ID3D12PipelineState *initial_pipeline_state, REFIID riid, void **command_list) override { return real_->CreateCommandList(node_mask, type, command_allocator, initial_pipeline_state, riid, command_list); }
    HRESULT STDMETHODCALLTYPE CheckFeatureSupport(D3D12_FEATURE feature, void *feature_data, UINT feature_data_size) override { return real_->CheckFeatureSupport(feature, feature_data, feature_data_size); }
    HRESULT STDMETHODCALLTYPE CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC *desc, REFIID riid, void **descriptor_heap) override { return real_->CreateDescriptorHeap(desc, riid, descriptor_heap); }
    UINT STDMETHODCALLTYPE GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE descriptor_heap_type) override { return real_->GetDescriptorHandleIncrementSize(descriptor_heap_type); }
    HRESULT STDMETHODCALLTYPE CreateRootSignature(UINT node_mask, const void *bytecode, SIZE_T bytecode_length, REFIID riid, void **root_signature) override { return real_->CreateRootSignature(node_mask, bytecode, bytecode_length, riid, root_signature); }
    void STDMETHODCALLTYPE CreateConstantBufferView(const D3D12_CONSTANT_BUFFER_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { real_->CreateConstantBufferView(desc, descriptor); }
    void STDMETHODCALLTYPE CreateShaderResourceView(ID3D12Resource *resource, const D3D12_SHADER_RESOURCE_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { real_->CreateShaderResourceView(resource, desc, descriptor); }
    void STDMETHODCALLTYPE CreateUnorderedAccessView(ID3D12Resource *resource, ID3D12Resource *counter_resource, const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { real_->CreateUnorderedAccessView(resource, counter_resource, desc, descriptor); }
    void STDMETHODCALLTYPE CreateRenderTargetView(ID3D12Resource *resource, const D3D12_RENDER_TARGET_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { real_->CreateRenderTargetView(resource, desc, descriptor); }
    void STDMETHODCALLTYPE CreateDepthStencilView(ID3D12Resource *resource, const D3D12_DEPTH_STENCIL_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { real_->CreateDepthStencilView(resource, desc, descriptor); }
    void STDMETHODCALLTYPE CreateSampler(const D3D12_SAMPLER_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE descriptor) override { real_->CreateSampler(desc, descriptor); }
    void STDMETHODCALLTYPE CopyDescriptors(UINT dst_descriptor_range_count, const D3D12_CPU_DESCRIPTOR_HANDLE *dst_descriptor_range_offsets, const UINT *dst_descriptor_range_sizes, UINT src_descriptor_range_count, const D3D12_CPU_DESCRIPTOR_HANDLE *src_descriptor_range_offsets, const UINT *src_descriptor_range_sizes, D3D12_DESCRIPTOR_HEAP_TYPE descriptor_heap_type) override { real_->CopyDescriptors(dst_descriptor_range_count, dst_descriptor_range_offsets, dst_descriptor_range_sizes, src_descriptor_range_count, src_descriptor_range_offsets, src_descriptor_range_sizes, descriptor_heap_type); }
    void STDMETHODCALLTYPE CopyDescriptorsSimple(UINT descriptor_count, const D3D12_CPU_DESCRIPTOR_HANDLE dst_descriptor_range_offset, const D3D12_CPU_DESCRIPTOR_HANDLE src_descriptor_range_offset, D3D12_DESCRIPTOR_HEAP_TYPE descriptor_heap_type) override { real_->CopyDescriptorsSimple(descriptor_count, dst_descriptor_range_offset, src_descriptor_range_offset, descriptor_heap_type); }
#ifdef WIDL_EXPLICIT_AGGREGATE_RETURNS
    D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE GetResourceAllocationInfo(D3D12_RESOURCE_ALLOCATION_INFO* out, UINT visible_mask, UINT reource_desc_count, const D3D12_RESOURCE_DESC *resource_descs) override { return real_->GetResourceAllocationInfo(out, visible_mask, reource_desc_count, resource_descs); }
    D3D12_HEAP_PROPERTIES* STDMETHODCALLTYPE GetCustomHeapProperties(D3D12_HEAP_PROPERTIES* out, UINT node_mask, D3D12_HEAP_TYPE heap_type) override { return real_->GetCustomHeapProperties(out, node_mask, heap_type); }
#else
    D3D12_RESOURCE_ALLOCATION_INFO STDMETHODCALLTYPE GetResourceAllocationInfo(UINT visible_mask, UINT reource_desc_count, const D3D12_RESOURCE_DESC *resource_descs) override { return real_->GetResourceAllocationInfo(visible_mask, reource_desc_count, resource_descs); }
    D3D12_HEAP_PROPERTIES STDMETHODCALLTYPE GetCustomHeapProperties(UINT node_mask, D3D12_HEAP_TYPE heap_type) override { return real_->GetCustomHeapProperties(node_mask, heap_type); }
#endif
    HRESULT STDMETHODCALLTYPE CreateCommittedResource(const D3D12_HEAP_PROPERTIES *heap_properties, D3D12_HEAP_FLAGS heap_flags, const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE *optimized_clear_value, REFIID riid, void **resource) override { return real_->CreateCommittedResource(heap_properties, heap_flags, desc, initial_state, optimized_clear_value, riid, resource); }
    HRESULT STDMETHODCALLTYPE CreateHeap(const D3D12_HEAP_DESC *desc, REFIID riid, void **heap) override { return real_->CreateHeap(desc, riid, heap); }
    HRESULT STDMETHODCALLTYPE CreatePlacedResource(ID3D12Heap *heap, UINT64 heap_offset, const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE *optimized_clear_value, REFIID riid, void **resource) override { return real_->CreatePlacedResource(heap, heap_offset, desc, initial_state, optimized_clear_value, riid, resource); }
    HRESULT STDMETHODCALLTYPE CreateReservedResource(const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE *optimized_clear_value, REFIID riid, void **resource) override { return real_->CreateReservedResource(desc, initial_state, optimized_clear_value, riid, resource); }
    HRESULT STDMETHODCALLTYPE CreateSharedHandle(ID3D12DeviceChild *object, const SECURITY_ATTRIBUTES *attributes, DWORD access, const WCHAR *name, HANDLE *handle) override { return real_->CreateSharedHandle(object, attributes, access, name, handle); }
    HRESULT STDMETHODCALLTYPE OpenSharedHandle(HANDLE handle, REFIID riid, void **object) override { return real_->OpenSharedHandle(handle, riid, object); }
    HRESULT STDMETHODCALLTYPE OpenSharedHandleByName(const WCHAR *name, DWORD access, HANDLE *handle) override { return real_->OpenSharedHandleByName(name, access, handle); }
    HRESULT STDMETHODCALLTYPE MakeResident(UINT object_count, ID3D12Pageable *const *objects) override { return real_->MakeResident(object_count, objects); }
    HRESULT STDMETHODCALLTYPE Evict(UINT object_count, ID3D12Pageable *const *objects) override { return real_->Evict(object_count, objects); }
    HRESULT STDMETHODCALLTYPE CreateFence(UINT64 initial_value, D3D12_FENCE_FLAGS flags, REFIID riid, void **fence) override
    {
        if (riid != __uuidof(ID3D12Fence)) return E_NOINTERFACE;
        ID3D12Fence* real_fence = nullptr;
        const HRESULT hr = real_->CreateFence(initial_value, flags, IID_PPV_ARGS(&real_fence));
        if (SUCCEEDED(hr)) {
            fence_ = RemovedFence(real_fence);
            *fence = static_cast<ID3D12Fence*>(&fence_);
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason() override { return real_->GetDeviceRemovedReason(); }
    void STDMETHODCALLTYPE GetCopyableFootprints(const D3D12_RESOURCE_DESC *desc, UINT first_sub_resource, UINT sub_resource_count, UINT64 base_offset, D3D12_PLACED_SUBRESOURCE_FOOTPRINT *layouts, UINT *row_count, UINT64 *row_size, UINT64 *total_bytes) override { real_->GetCopyableFootprints(desc, first_sub_resource, sub_resource_count, base_offset, layouts, row_count, row_size, total_bytes); }
    HRESULT STDMETHODCALLTYPE CreateQueryHeap(const D3D12_QUERY_HEAP_DESC *desc, REFIID riid, void **heap) override { return real_->CreateQueryHeap(desc, riid, heap); }
    HRESULT STDMETHODCALLTYPE SetStablePowerState(BOOL enable) override { return real_->SetStablePowerState(enable); }
    HRESULT STDMETHODCALLTYPE CreateCommandSignature(const D3D12_COMMAND_SIGNATURE_DESC *desc, ID3D12RootSignature *root_signature, REFIID riid, void **command_signature) override { return real_->CreateCommandSignature(desc, root_signature, riid, command_signature); }
    void STDMETHODCALLTYPE GetResourceTiling(ID3D12Resource *resource, UINT *total_tile_count, D3D12_PACKED_MIP_INFO *packed_mip_info, D3D12_TILE_SHAPE *standard_tile_shape, UINT *sub_resource_tiling_count, UINT first_sub_resource_tiling, D3D12_SUBRESOURCE_TILING *sub_resource_tilings) override { real_->GetResourceTiling(resource, total_tile_count, packed_mip_info, standard_tile_shape, sub_resource_tiling_count, first_sub_resource_tiling, sub_resource_tilings); }
#ifdef WIDL_EXPLICIT_AGGREGATE_RETURNS
    LUID* STDMETHODCALLTYPE GetAdapterLuid(LUID* out) override { return real_->GetAdapterLuid(out); }
#else
    LUID STDMETHODCALLTYPE GetAdapterLuid() override { return real_->GetAdapterLuid(); }
#endif
    // clang-format on

private:
    ID3D12Device* real_;
    RemovedFence fence_{nullptr};
};


// Error and corruption messages the debug layer has stored since the last call; prints each one.
// `deleted_in_use` is set when one of them reports an object released while the GPU still used it.
int debug_errors(ID3D12InfoQueue* info, bool* deleted_in_use = nullptr)
{
    int errors = 0;
    const UINT64 stored = info->GetNumStoredMessages();
    for (UINT64 i = 0; i < stored; ++i) {
        SIZE_T bytes = 0;
        info->GetMessage(i, nullptr, &bytes);
        std::vector<unsigned char> storage(bytes);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (FAILED(info->GetMessage(i, message, &bytes))) continue;
        if (message->Severity != D3D12_MESSAGE_SEVERITY_ERROR && message->Severity != D3D12_MESSAGE_SEVERITY_CORRUPTION) continue;
        ++errors;
        if (deleted_in_use && message->ID == D3D12_MESSAGE_ID_OBJECT_DELETED_WHILE_STILL_IN_USE) *deleted_in_use = true;
        std::printf("debug layer: %.*s\n", static_cast<int>(message->DescriptionByteLength), message->pDescription);
    }
    info->ClearStoredMessages();
    return errors;
}

// Builds a stall: the queue waits on `gate` (not yet signalled) before anything submitted after it.
ID3D12Fence* stall(Gpu& gpu)
{
    ID3D12Fence* gate = nullptr;
    gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate));
    gpu.queue->Wait(gate, 1);
    return gate;
}

int lifetime(Gpu& gpu, ID3D12InfoQueue* info, x2w_present_signature::Block* block, bool control)
{
    const Picture picture = gradient(false);
    std::uint8_t want[x2w_present_signature::cells];
    expected_grid(picture, x2w_present_signature::Layout::rgba8, want);
    debug_errors(info);  // setup noise, if any, is not part of the check
    if (control) {
        // The defect: a copy whose source nobody holds. The test records it itself, releases its only
        // reference while the queue is stalled, then lets the queue run.
        ID3D12Resource* source = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, picture);
        ID3D12Fence* gate = stall(gpu);
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = 1024;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* readback = nullptr;
        ID3D12CommandAllocator* allocator = nullptr;
        ID3D12GraphicsCommandList* list = nullptr;
        gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));
        gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
        gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list));
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        from.pResource = source;
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to.pResource = readback;
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, width, 1, 1, 1024};
        const D3D12_BOX row{0, 0, 0, width, 1, 1};
        list->CopyTextureRegion(&to, 0, 0, 0, &from, &row);
        list->Close();
        ID3D12CommandList* lists[] = {list};
        gpu.queue->ExecuteCommandLists(1, lists);
        source->Release();  // nobody holds the source of the stalled copy now
        gate->Signal(1);
        gpu.idle();
        bool deleted_in_use = false;
        const int errors = debug_errors(info, &deleted_in_use);
        std::printf("control: %d debug-layer errors, released while in use %s\n", errors, deleted_in_use ? "reported" : "not reported");
        check(deleted_in_use, "the debug layer reports a stalled copy whose source was released (the check can fail)");
        list->Release();
        allocator->Release();
        readback->Release();
        gate->Release();
        return 0;
    }

    // The sampler: its copy is stalled, the caller drops its reference, then the queue runs.
    {
        ID3D12Resource* presented = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, picture);
        x2w_present_signature::Sampler sampler(1);
        ID3D12Fence* gate = stall(gpu);
        const LONG64 before = block->sequence / 2;
        sampler.present(gpu.device, gpu.queue, presented, 1, block);
        check(sampler.in_flight() && sampler.holds_source(), "the stalled copy is in flight and the sampler holds its surface");
        const ULONG left = presented->Release();  // the caller's only reference, right after the present
        check(left == 1, "after the caller's release, only the sampler's reference is left");
        sampler.present(gpu.device, gpu.queue, nullptr, 2, block);
        check(block->sequence / 2 == before && sampler.holds_source(), "nothing is published and the surface is kept while the queue is stalled");
        gate->Signal(1);
        gpu.idle();
        sampler.present(gpu.device, gpu.queue, nullptr, 3, block);
        x2w_present_signature::Snapshot snapshot{};
        check(block->sequence / 2 == before + 1 && x2w_present_signature::read(block, snapshot) && snapshot.sample_present == 1 &&
                  std::memcmp(snapshot.luma, want, sizeof(want)) == 0,
              "after the stall the published signature equals the released surface's picture");
        check(!sampler.holds_source() && !sampler.in_flight(), "the sampler drops the surface once the copy's fence has passed");
        gate->Release();
    }
    // Signal fails after the list was submitted: the sampler stops and keeps everything the copy uses.
    {
        ID3D12Resource* presented = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, picture);
        SignalFailingQueue failing(gpu.queue);
        failing.fail_signal = true;
        ID3D12Fence* gate = stall(gpu);
        auto* sampler = new x2w_present_signature::Sampler(1);
        sampler->present(gpu.device, &failing, presented, 1, block);
        check(sampler->failed() && sampler->stranded() && sampler->holds_source(), "a failed Signal strands the sampler with the surface held");
        const ULONG left = presented->Release();
        check(left == 1, "the stranded sampler still holds the surface after the caller's release");
        gate->Signal(1);
        gpu.idle();
        sampler->present(gpu.device, &failing, presented, 2, block);
        check(sampler->stranded() && !sampler->in_flight() && sampler->holds_source(), "a stranded sampler records nothing more and keeps the surface");
        delete sampler;  // a stranded sampler releases nothing it may still be using
        gate->Release();
    }
    // Device removal while a copy is in flight: the fence reports UINT64_MAX, which is not a
    // completion. Nothing is mapped or published, the sampler stops sampling, and the source stays
    // held; the destructor keeps every object the copy uses. A genuine prior sample (published
    // before the removal) stays as it was: no stale publish, no clearing.
    {
        ID3D12Resource* presented = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, picture);
        RemovalDevice removed_device(gpu.device);
        // The queue proxy unwraps the sampler's RemovedFence to its real fence by exact fake
        // identity, so no fake COM object ever reaches the driver.
        SignalFailingQueue unwrapping(gpu.queue, &removed_device.fence());
        x2w_present_signature::Sampler sampler(1, true, true);
        ID3D12Fence* gate = stall(gpu);
        const LONG64 before = block->sequence / 2;
        // First a genuine completed copy: it publishes and fills the historical sample state.
        sampler.present(&removed_device, &unwrapping, presented, 1, block);
        gate->Signal(1);
        gpu.idle();
        sampler.present(&removed_device, &unwrapping, nullptr, 2, block);
        ID3D12Resource* const sampled_surface = sampler.last_surface();
        const bool had_sample = block->sequence / 2 == before + 1 && sampler.last_present() == 1
                                && sampled_surface && !sampler.holds_source();
        check(had_sample, "a genuine copy completes and publishes before the removal");
        // Then a second copy whose fence will report device removal.
        sampler.present(&removed_device, &unwrapping, presented, 3, block);
        check(sampler.in_flight() && sampler.holds_source(), "the second copy is in flight before the removal");
        const ULONG left = presented->Release();  // the caller's only reference
        check(left == 1, "before the removal only the sampler's reference is left");
        gate->Signal(1);
        gpu.idle();  // the copy has really finished; the fence still reports device removal
        removed_device.fence().removed = true;
        const LONG64 published_before = block->sequence / 2;
        sampler.present(&removed_device, &unwrapping, nullptr, 4, block);
        check(!sampler.in_flight() && sampler.failed() && sampler.stranded(),
              "a removed fence is not a completion: the sampler stops, stranded like a failed Signal");
        check(sampler.holds_source(), "the source stays held after a removed fence");
        check(block->sequence / 2 == published_before, "a removed fence publishes no sample");
        // The prior historical sample survives exactly as it was: nothing cleared, nothing stale.
        check(sampler.last_present() == 1 && sampler.last_surface() == sampled_surface,
              "the prior published sample (number and surface) survives the removal unchanged");
        sampler.present(&removed_device, &unwrapping, nullptr, 5, block);
        check(sampler.failed() && sampler.stranded() && !sampler.in_flight() && block->sequence / 2 == published_before,
              "a stranded sampler records nothing more and keeps the surface");
        gate->Release();
        // The destructor must keep every object the copy uses (stranded): nothing is released
        // under a removed device. (The destructor runs at the block's end.)
    }
    // The destructor's own re-observation: a copy still pending at destruction transitions across
    // the destructor's 10 s wait deterministically, because the fence reports a chosen value for a
    // fixed number of observations and then the real (or removed) state. Pending that completes
    // while the wait runs releases normally; a fence that reports removal afterwards keeps
    // everything. The real fence is the only object the driver sees.
    {
        // Pending completes during the destructor's wait: normal release, no strand.
        {
            ID3D12Resource* presented = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, picture);
            RemovalDevice removal(gpu.device);
            SignalFailingQueue unwrapping(gpu.queue, &removal.fence());
            ID3D12Fence* gate = stall(gpu);
            x2w_present_signature::Sampler* sampler = new x2w_present_signature::Sampler(1);
            sampler->present(&removal, &unwrapping, presented, 1, block);
            check(sampler->in_flight() && sampler->holds_source(), "a stalled copy is in flight at destruction");
            gate->Signal(1);  // the copy really completes; the fence is scripted to report pending once
            removal.fence().pending_observations = 1;
            removal.fence().next = 0;
            delete sampler;  // pre-wait observation: pending; wait: on the real fence; post-wait: completed
            check(removal.fence().observations >= 2 && removal.fence().event_registrations == 1,
                  "the destructor observes before and after its wait, and registers its event once");
            check(presented->Release() == 0, "a copy that completes while the destructor waits releases the source normally");
            gate->Release();
        }
        // The fence reports the removal sentinel to the post-wait observation: nothing releases.
        {
            ID3D12Resource* presented = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, picture);
            RemovalDevice removal(gpu.device);
            SignalFailingQueue unwrapping(gpu.queue, &removal.fence());
            ID3D12Fence* gate = stall(gpu);
            x2w_present_signature::Sampler* sampler = new x2w_present_signature::Sampler(1);
            sampler->present(&removal, &unwrapping, presented, 2, block);
            check(sampler->in_flight() && sampler->holds_source(), "the second stalled copy is in flight at destruction");
            gate->Signal(1);  // the copy really completes
            removal.fence().pending_observations = 1;
            removal.fence().next = 0;
            removal.fence().removed = true;  // from the second observation on: the removal sentinel
            delete sampler;
            check(removal.fence().observations == 2 && removal.fence().event_registrations == 1,
                  "the destructor re-observed removal after exactly one wait registration");
            check(presented->Release() == 1, "a fence that reports removal after the wait keeps the sampler's source reference");
            gate->Release();
        }
    }
    const int errors = debug_errors(info);
    std::printf("held source: %d debug-layer errors\n", errors);
    check(errors == 0, "the debug layer reports no error for the sampler's stalled copy");
    return 0;
}

int changes(const std::vector<x2w_present_signature::Snapshot>& grids)
{
    int n = 0;
    for (std::size_t i = 1; i < grids.size(); ++i)
        n += x2w_input_script::signal_distance(grids[i - 1].luma, grids[i].luma) >= x2w_input_script::changed_cells;
    return n;
}

}  // namespace

int main(int argc, char** argv)
{
    const bool held = argc > 1 && std::strcmp(argv[1], "held_source") == 0;
    const bool control = argc > 1 && std::strcmp(argv[1], "unheld_control") == 0;
    if (argc > 1 && !held && !control) {
        std::printf("unknown mode %s\n", argv[1]);
        return 2;
    }
    if (held || control) {
        ID3D12Debug* debug = nullptr;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            std::printf("SKIP: no D3D12 debug layer\n");
            return 77;
        }
        debug->EnableDebugLayer();
        debug->Release();
    }
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter* warp = nullptr;
    Gpu gpu;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) ||
        FAILED(D3D12CreateDevice(warp, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&gpu.device)))) {
        std::printf("SKIP: no WARP device\n");
        return 77;
    }
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    gpu.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&gpu.queue));
    gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu.fence));

    // A block of its own: this test's process id plus a high bit, so it never meets a run's block.
    x2w_present_signature::Block* block = x2w_present_signature::create_block(GetCurrentProcessId() | 0x80000000u, 1);
    check(block != nullptr, "create the signature block");
    if (!block) return 1;

    if (held || control) {
        ID3D12InfoQueue* info = nullptr;
        if (FAILED(gpu.device->QueryInterface(IID_PPV_ARGS(&info)))) {
            std::printf("SKIP: the debug layer gave no info queue\n");
            return 77;
        }
        lifetime(gpu, info, block, control);
        info->Release();
        std::printf("%s\n", failures ? "FAIL" : "pass");
        return failures ? 1 : 0;
    }

    const Picture a = gradient(false), b = gradient(true);
    for (DXGI_FORMAT format : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_TYPELESS}) {
        const auto layout = format == DXGI_FORMAT_R8G8B8A8_UNORM ? x2w_present_signature::Layout::rgba8 : x2w_present_signature::Layout::bgra8;
        ID3D12Resource* a1 = surface(gpu, format, a);
        ID3D12Resource* a2 = surface(gpu, format, a);
        ID3D12Resource* b1 = surface(gpu, format, b);
        ID3D12Resource* b2 = surface(gpu, format, b);
        check(a1 && a2 && b1 && b2, "create the surfaces");
        if (!(a1 && a2 && b1 && b2)) return 1;

        // Two different pictures (A, A, B, B): the signal changes exactly once.
        const auto changing = present_all(gpu, block, {a1, a2, b1, b2});
        std::uint8_t grid_a[x2w_present_signature::cells], grid_b[x2w_present_signature::cells];
        expected_grid(a, layout, grid_a);
        expected_grid(b, layout, grid_b);
        check(changing.size() == 4, "four samples published for four presents");
        if (changing.size() == 4) {
            check(std::memcmp(changing[0].luma, grid_a, sizeof(grid_a)) == 0 && std::memcmp(changing[3].luma, grid_b, sizeof(grid_b)) == 0,
                  "published grids equal the CPU reduction of the sampled rows");
            check(changing[0].sample_present == 1 && changing[3].sample_present == 4, "samples carry their present numbers");
        }
        check(changes(changing) == 1, "two different pictures: the signal changes once");
        // Negative control: the same picture presented four times never changes the signal.
        const auto steady = present_all(gpu, block, {a1, a2, a1, a2});
        check(steady.size() == 4 && changes(steady) == 0, "identical pictures: the signal never changes");
        for (ID3D12Resource* r : {a1, a2, b1, b2}) r->Release();
    }

    // Opt-in: a sampler made with always = false creates no GPU objects and records no copy until the
    // block asks for samples; then it samples as usual.
    {
        ID3D12Resource* s = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, a);
        x2w_present_signature::Sampler sampler(1, false);
        InterlockedExchange(&block->sampling_wanted, 0);
        const LONG64 before = block->sequence / 2;
        for (LONG64 present = 1; present <= 4; ++present) {
            sampler.present(gpu.device, gpu.queue, s, present, block);
            gpu.idle();
        }
        check(!sampler.set_up() && !sampler.in_flight() && block->sequence / 2 == before && block->presents == 4,
              "without a request: no GPU objects, no copies, presents still counted");
        InterlockedExchange(&block->sampling_wanted, 1);
        sampler.present(gpu.device, gpu.queue, s, 5, block);
        gpu.idle();
        sampler.present(gpu.device, gpu.queue, nullptr, 6, block);
        x2w_present_signature::Snapshot last{};
        check(sampler.set_up() && block->sequence / 2 == before + 1 && x2w_present_signature::read(block, last) && last.sample_present == 5,
              "after the request: the next present is sampled");
        InterlockedExchange(&block->sampling_wanted, 0);
        s->Release();
    }
    // Interval: with every = 4, eight presents publish two samples (presents 1 and 5).
    {
        ID3D12Resource* s = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, a);
        x2w_present_signature::Sampler sampler(4);
        const LONG64 before = block->sequence / 2;
        for (LONG64 present = 1; present <= 8; ++present) {
            sampler.present(gpu.device, gpu.queue, s, present, block);
            gpu.idle();
        }
        sampler.present(gpu.device, gpu.queue, nullptr, 9, block);
        x2w_present_signature::Snapshot last{};
        check(block->sequence / 2 - before == 2 && x2w_present_signature::read(block, last) && last.sample_present == 5 && last.presents == 9,
              "every 4 presents: two samples from presents 1 and 5, latest present 9");
        s->Release();
    }
    // A format the signature does not read is refused, not guessed.
    {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = 16;
        desc.Height = 16;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.SampleDesc.Count = 1;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        ID3D12Resource* half = nullptr;
        gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&half));
        x2w_present_signature::Sampler sampler(1);
        sampler.present(gpu.device, gpu.queue, half, 1, block);
        check(half && sampler.unsupported_format() == DXGI_FORMAT_R16G16B16A16_FLOAT && !sampler.in_flight(),
              "16-bit float surface refused without a copy");
        if (half) half->Release();
    }
    // Resolution: a bar 60 x 4 texels (30 % x 3.3 %, 1 % of the frame) at rows 11..14, between the base
    // rows 10 and 16, switches from black to white. The base grid must not see it; the fine grid must,
    // and its published cells must equal the CPU reduction of the 72 fine rows.
    {
        Picture off = gradient(false), on = off;
        for (UINT y = 11; y < 15; ++y)
            for (UINT x = 20; x < 80; ++x)
                for (int c = 0; c < 3; ++c) {
                    off[(std::size_t{y} * width + x) * 4 + c] = 0;
                    on[(std::size_t{y} * width + x) * 4 + c] = 255;
                }
        ID3D12Resource* s_off = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, off);
        ID3D12Resource* s_on = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, on);
        InterlockedExchange(&block->fine_wanted, 1);
        const auto grids = present_all(gpu, block, {s_off, s_on});
        InterlockedExchange(&block->fine_wanted, 0);
        check(grids.size() == 2 && grids[0].fine_valid && grids[1].fine_valid, "fine grid published while asked for");
        if (grids.size() == 2) {
            x2w_input_script::Wait base_wait{}, fine_wait{};
            base_wait.level = fine_wait.level = x2w_input_script::cell_tolerance;
            fine_wait.fine = true;
            const int base_cells = x2w_input_script::wait_distance(base_wait, grids[0].luma, grids[1].luma);
            const int fine_cells = x2w_input_script::wait_distance(fine_wait, grids[0].fine, grids[1].fine);
            std::printf("1 %% bar: base grid %d changed cells, fine grid %d changed cells\n", base_cells, fine_cells);
            check(base_cells == 0, "the 32 x 18 grid misses a 1 % bar between its rows");
            check(fine_cells >= x2w_input_script::changed_cells, "the 64 x 72 grid sees the 1 % bar in at least 16 cells");
            std::uint8_t want[x2w_present_signature::fine_cells];
            for (int band = 0; band < x2w_present_signature::fine_y; ++band)
                x2w_present_signature::reduce_row(&on[std::size_t{x2w_present_signature::sampled_row(band, height, x2w_present_signature::fine_y)} * width * 4],
                                                  width, x2w_present_signature::Layout::rgba8, want + band * x2w_present_signature::fine_x,
                                                  x2w_present_signature::fine_x);
            check(std::memcmp(grids[1].fine, want, sizeof(want)) == 0, "published fine cells equal the CPU reduction of the fine rows");
        }
        const auto plain = present_all(gpu, block, {s_off, s_on});
        check(plain.size() == 2 && !plain[0].fine_valid && !plain[1].fine_valid, "no fine grid while nobody asks");
        s_off->Release();
        s_on->Release();
    }
    // Cost at 3840 x 2160: CPU time of the present() calls (record the row copies; map, reduce and
    // publish the previous copy) with and without the fine grid. The GPU copies 18 or 90 rows.
    {
        constexpr UINT big_w = 3840, big_h = 2160;
        Picture big(std::size_t{big_w} * big_h * 4, 128);
        ID3D12Resource* s = surface(gpu, DXGI_FORMAT_R8G8B8A8_UNORM, big, big_w, big_h);
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        double ms[2] = {};
        for (int fine = 0; fine < 2; ++fine) {
            InterlockedExchange(&block->fine_wanted, fine);
            x2w_present_signature::Sampler sampler(1);
            constexpr int rounds = 24;
            LONGLONG ticks = 0;
            for (int i = 0; i < rounds + 2; ++i) {
                LARGE_INTEGER a, b;
                QueryPerformanceCounter(&a);
                sampler.present(gpu.device, gpu.queue, s, i + 1, block);
                QueryPerformanceCounter(&b);
                if (i >= 2) ticks += b.QuadPart - a.QuadPart;  // the first two set up the buffers
                gpu.idle();
            }
            ms[fine] = ticks * 1000.0 / frequency.QuadPart / rounds;
        }
        InterlockedExchange(&block->fine_wanted, 0);
        const double row_kib = big_w * 4 / 1024.0;
        std::printf("cost per sampled present at %ux%u (CPU side, WARP): base %.3f ms (18 rows, %.0f KiB), fine %.3f ms (90 rows, %.0f KiB); "
                    "a full frame is %.0f KiB\n",
                    big_w, big_h, ms[0], 18 * row_kib, ms[1], 90 * row_kib, big_h * row_kib);
        s->Release();
    }
    std::printf("%s\n", failures ? "FAIL" : "pass");
    return failures ? 1 : 0;
}
