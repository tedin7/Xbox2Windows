// Experimental (docs/startup-runtime.md#input-script): fills the presented-picture signature
// (present_signature.h) from the console present of the D3D12 stand-in.
//
// Every `every` presents it copies the 18 sampled rows of the presented surface (one texel high
// each, never the whole surface), and the 72 rows of the fine grid while a reader asks for it, into
// a small readback buffer on the presenting queue and signals a fence. It never waits: a later call publishes the rows once the fence has passed, and no new
// copy starts while one is in flight. The surface rests in COMMON before and after.
//
// Lifetime: command lists do not hold the resources they reference, and the caller may release the
// presented surface right after the present. The sampler therefore holds its own reference to the
// surface from submission until the copy's fence has passed. If Signal fails after the list was
// submitted, nothing can tell when the copy ends: the sampler stops sampling and keeps the surface,
// allocator, list and readback buffer referenced for the rest of the process ("stranded").
//
// Sampling is opt-in: a sampler made with always = false records copies only while the block's
// sampling_wanted is set (a script with a picture wait asks for it); otherwise it creates no GPU
// objects and submits nothing, and only counts presents.
#pragma once
#include "resource_budget.h"

#include "startup_stubs/present_signature.h"
#include "startup_stubs/present_view_format.h"

#include <d3d12.h>

#include <cstdlib>
#include <initializer_list>

namespace x2w_present_signature {

// Layout of the 4-byte colour formats the signature reads; false for any other format.
inline bool layout_of(DXGI_FORMAT format, Layout& layout)
{
    switch (x2w_startup::present_view_format(format)) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: layout = Layout::rgba8; return true;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: layout = Layout::bgra8; return true;
    case DXGI_FORMAT_R10G10B10A2_UNORM: layout = Layout::rgb10a2; return true;
    default: return false;
    }
}

inline constexpr int onset_rgb_tolerance = 4;

class Sampler {
public:
    explicit Sampler(int every, bool always = true, bool capture_center = false, bool check_flat = false)
        : capture_center_(capture_center || check_flat), check_flat_(check_flat), every_(every), always_(always) {}

    // Called once per present with the presented surface. Returns true when this call published a sample.
    bool present(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Resource* surface, LONG64 present_number, Block* block)
    {
        bool published = false;
        if (in_flight_ && fence_->GetCompletedValue() >= fence_value_) {
            in_flight_ = false;
            if (capture_center_) {
                last_present_ = copied_present_;
                last_format_ = format_;
                last_surface_ = source_;
            }
            retire_source();
            void* mapped = nullptr;
            const D3D12_RANGE range{0, static_cast<SIZE_T>(row_stride_ * rows_)};
            if (SUCCEEDED(readback_->Map(0, &range, &mapped))) {
                const auto* base = static_cast<const unsigned char*>(mapped);
                if (capture_center_)
                    std::memcpy(last_rgba_, base + row_stride_ * (cells_y / 2) + (width_ / 2) * 4, 4);
                if (check_flat_ && last_format_ == DXGI_FORMAT_R8G8B8A8_UNORM) {
                    last_flat_ = true;
                    for (int band = 0; band < cells_y && last_flat_; ++band) {
                        const auto* row = base + row_stride_ * band;
                        for (UINT x = 0; x < width_ && last_flat_; ++x) {
                            for (unsigned c = 0; c < 3; ++c)
                                if (std::abs(int(row[x * 4 + c]) - int(last_rgba_[c])) > onset_rgb_tolerance)
                                    last_flat_ = false;
                        }
                    }
                }
                std::uint8_t luma[cells];
                for (int band = 0; band < cells_y; ++band) reduce_row(base + row_stride_ * band, width_, layout_, luma + band * cells_x);
                if (rows_ > cells_y)
                    for (int band = 0; band < fine_y; ++band)
                        reduce_row(base + row_stride_ * (cells_y + band), width_, layout_, fine_ + band * fine_x, fine_x);
                const D3D12_RANGE none{0, 0};
                readback_->Unmap(0, &none);
                if (block) publish(block, luma, copied_present_, rows_ > cells_y ? fine_ : nullptr);
                published = true;
            }
        }
        if (block) note_present(block, present_number);
        if (every_ <= 0 || failed_ || !device || !queue || !surface) return published;
        if (!always_ && !(block && block->sampling_wanted)) return published;
        if ((counter_++ % every_) != 0 || in_flight_) return published;
        const D3D12_RESOURCE_DESC desc = surface->GetDesc();
        Layout layout;
        if (!x2w_startup::present_surface_supported(desc) || !layout_of(desc.Format, layout) || desc.Width > 16384) {
            unsupported_ = desc.Format;
            return published;
        }
        const auto width = static_cast<UINT>(desc.Width);
        const UINT64 pitch = (UINT64{width} * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~UINT64{D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1};
        const UINT64 stride = (pitch + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) & ~UINT64{D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1};
        const int rows = cells_y + (block && block->fine_wanted ? fine_y : 0);
        if (!setup(device, stride * rows)) return published;
        if (FAILED(allocator_->Reset()) || FAILED(list_->Reset(allocator_, nullptr))) {
            failed_ = true;
            return published;
        }
        barrier(surface, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from{};
        from.pResource = surface;
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        for (int row = 0; row < rows; ++row) {
            const int band = row < cells_y ? row : row - cells_y;
            const UINT y = sampled_row(band, desc.Height, row < cells_y ? cells_y : fine_y);
            D3D12_TEXTURE_COPY_LOCATION to{};
            to.pResource = readback_;
            to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            to.PlacedFootprint.Offset = stride * row;
            to.PlacedFootprint.Footprint = {desc.Format, width, 1, 1, static_cast<UINT>(pitch)};
            const D3D12_BOX box{0, y, 0, width, y + 1, 1};
            list_->CopyTextureRegion(&to, 0, 0, 0, &from, &box);
        }
        barrier(surface, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        if (FAILED(list_->Close())) {
            failed_ = true;
            return published;
        }
        ID3D12CommandList* lists[] = {list_};
        surface->AddRef();  // held until the copy's fence has passed (retire_source)
        source_ = surface;
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence_, ++fence_value_))) {
            // Submitted, but no fence will tell when the copy ends: keep everything it uses.
            stranded_ = failed_ = true;
            return published;
        }
        in_flight_ = true;
        layout_ = layout;
        if (capture_center_) format_ = desc.Format;
        width_ = width;
        row_stride_ = stride;
        rows_ = rows;
        copied_present_ = present_number;
        return published;
    }

    bool last_flat() const { return last_flat_; }
    DXGI_FORMAT last_format() const { return last_format_; }
    LONG64 last_present() const { return last_present_; }
    const unsigned char* last_rgba() const { return last_rgba_; }
    ID3D12Resource* last_surface() const { return last_surface_; }

    bool failed() const { return failed_; }
    bool stranded() const { return stranded_; }
    bool holds_source() const { return source_ != nullptr; }
    bool set_up() const { return allocator_ != nullptr; }
    DXGI_FORMAT unsupported_format() const { return unsupported_; }
    bool in_flight() const { return in_flight_; }
    int rows_in_flight() const { return in_flight_ ? rows_ : 0; }

    // Waits (at most 10 s) for a copy in flight; if it cannot be seen to end, or the sampler is
    // stranded, the objects the copy uses stay referenced instead of being released under the GPU.
    ~Sampler()
    {
        if (in_flight_ && !stranded_ && fence_->GetCompletedValue() < fence_value_) {
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (event && SUCCEEDED(fence_->SetEventOnCompletion(fence_value_, event))) WaitForSingleObject(event, 10000);
            if (event) CloseHandle(event);
        }
        if (stranded_ || (in_flight_ && fence_->GetCompletedValue() < fence_value_)) return;
        retire_source();
        for (IUnknown* object : {static_cast<IUnknown*>(readback_), static_cast<IUnknown*>(fence_),
                                 static_cast<IUnknown*>(list_), static_cast<IUnknown*>(allocator_)})
            if (object) object->Release();
    }

private:
    void retire_source()
    {
        if (source_) source_->Release();
        source_ = nullptr;
    }
    bool setup(ID3D12Device* device, UINT64 bytes)
    {
        HRESULT hr = S_OK;
        if (!allocator_) {
            hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_));
            if (SUCCEEDED(hr)) hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_, nullptr, IID_PPV_ARGS(&list_));
            if (SUCCEEDED(hr)) hr = list_->Close();
            if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
        }
        if (SUCCEEDED(hr) && readback_bytes_ < bytes) {
            if (readback_) readback_->Release();
            readback_ = nullptr;
            readback_bytes_ = 0;
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = bytes;
            buffer.Height = 1;
            buffer.DepthOrArraySize = 1;
            buffer.MipLevels = 1;
            buffer.SampleDesc.Count = 1;
            buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                 IID_PPV_ARGS(&readback_));
            if (SUCCEEDED(hr)) {
                x2w_budget::committed(device, readback_, D3D12_HEAP_TYPE_READBACK);
                readback_bytes_ = bytes;
            }
        }
        if (FAILED(hr)) failed_ = true;
        return SUCCEEDED(hr);
    }
    void barrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
    {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = resource;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = from;
        b.Transition.StateAfter = to;
        list_->ResourceBarrier(1, &b);
    }

    const bool capture_center_, check_flat_;
    bool last_flat_ = false;
    DXGI_FORMAT format_ = DXGI_FORMAT_UNKNOWN, last_format_ = DXGI_FORMAT_UNKNOWN;
    LONG64 last_present_ = 0;
    unsigned char last_rgba_[4]{};
    ID3D12Resource* last_surface_ = nullptr;
    int every_;
    bool always_;
    LONG64 counter_ = 0, copied_present_ = 0;
    ID3D12CommandAllocator* allocator_ = nullptr;
    ID3D12GraphicsCommandList* list_ = nullptr;
    ID3D12Fence* fence_ = nullptr;
    ID3D12Resource* readback_ = nullptr;
    ID3D12Resource* source_ = nullptr;  // the surface of the copy in flight
    UINT64 readback_bytes_ = 0, fence_value_ = 0, row_stride_ = 0;
    UINT width_ = 0;
    int rows_ = cells_y;
    std::uint8_t fine_[fine_cells] = {};
    Layout layout_ = Layout::rgba8;
    DXGI_FORMAT unsupported_ = DXGI_FORMAT_UNKNOWN;
    bool in_flight_ = false, failed_ = false, stranded_ = false;
};

}  // namespace x2w_present_signature
