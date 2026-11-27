#pragma once

#include "startup_stubs/present_signature_sampler.h"

#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <new>
#include <utility>

namespace x2w_red {

using Resource = Microsoft::WRL::ComPtr<ID3D12Resource>;
using Log = void (*)(const char*, ...);

struct Input {
    ID3D12Resource* resource;
    UINT slot;
    DXGI_FORMAT format;
    UINT mip, slice, plane;
};

inline Input shader_input(ID3D12Resource* resource, UINT slot, const D3D12_SHADER_RESOURCE_VIEW_DESC* view)
{
    Input input{resource, slot};
    if (!view) return input;
    input.format = view->Format;
    switch (view->ViewDimension) {
    case D3D12_SRV_DIMENSION_TEXTURE2D:
        input.mip = view->Texture2D.MostDetailedMip;
        input.plane = view->Texture2D.PlaneSlice;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
        input.mip = view->Texture2DArray.MostDetailedMip;
        input.slice = view->Texture2DArray.FirstArraySlice;
        input.plane = view->Texture2DArray.PlaneSlice;
        break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE:
        input.mip = view->TextureCube.MostDetailedMip;
        break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
        input.mip = view->TextureCubeArray.MostDetailedMip;
        input.slice = view->TextureCubeArray.First2DArrayFace;
        break;
    default: break;
    }
    return input;
}

enum class Encoding { unsupported, unorm8, unorm16, snorm8, snorm16, half, f32, rgb10, rgb11f, rgb9e5, depth24 };
struct Format {
    Encoding encoding = Encoding::unsupported;
    unsigned channels = 0, bytes = 0;
    bool bgra = false;
};

inline Format sample_format(DXGI_FORMAT format)
{
    using E = Encoding;
    switch (format) {
    case DXGI_FORMAT_R8_UNORM: return {E::unorm8, 1, 1};
    case DXGI_FORMAT_R8G8_UNORM: return {E::unorm8, 2, 2};
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return {E::unorm8, 4, 4};
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return {E::unorm8, 4, 4, true};
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return {E::unorm8, 3, 4, true};
    case DXGI_FORMAT_R8_SNORM: return {E::snorm8, 1, 1};
    case DXGI_FORMAT_R8G8_SNORM: return {E::snorm8, 2, 2};
    case DXGI_FORMAT_R8G8B8A8_SNORM: return {E::snorm8, 4, 4};
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_D16_UNORM: return {E::unorm16, 1, 2};
    case DXGI_FORMAT_R16G16_UNORM: return {E::unorm16, 2, 4};
    case DXGI_FORMAT_R16G16B16A16_UNORM: return {E::unorm16, 4, 8};
    case DXGI_FORMAT_R16_SNORM: return {E::snorm16, 1, 2};
    case DXGI_FORMAT_R16G16_SNORM: return {E::snorm16, 2, 4};
    case DXGI_FORMAT_R16G16B16A16_SNORM: return {E::snorm16, 4, 8};
    case DXGI_FORMAT_R16_FLOAT: return {E::half, 1, 2};
    case DXGI_FORMAT_R16G16_FLOAT: return {E::half, 2, 4};
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return {E::half, 4, 8};
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_D32_FLOAT: return {E::f32, 1, 4};
    case DXGI_FORMAT_R32G32_FLOAT: return {E::f32, 2, 8};
    case DXGI_FORMAT_R32G32B32_FLOAT: return {E::f32, 3, 12};
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return {E::f32, 4, 16};
    case DXGI_FORMAT_R10G10B10A2_UNORM: return {E::rgb10, 4, 4};
    case DXGI_FORMAT_R11G11B10_FLOAT: return {E::rgb11f, 3, 4};
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return {E::rgb9e5, 3, 4};
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return {E::depth24, 1, 4};
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return {E::f32, 1, 8};
    default: return {};
    }
}

template <class T>
T read(const unsigned char* bytes)
{
    T result;
    std::memcpy(&result, bytes, sizeof(result));
    return result;
}

inline double small_float(unsigned bits, unsigned mantissa_bits)
{
    const unsigned exponent = (bits >> mantissa_bits) & 31;
    const unsigned mantissa = bits & ((1u << mantissa_bits) - 1);
    return exponent == 31 ? (mantissa ? NAN : INFINITY)
        : std::ldexp(double(mantissa + (exponent ? 1u << mantissa_bits : 0)),
                     (exponent ? int(exponent) - 15 : -14) - int(mantissa_bits));
}

inline double component(const unsigned char* bytes, Format format, unsigned channel)
{
    if (format.bgra && channel < 3) channel = 2 - channel;
    switch (format.encoding) {
    case Encoding::unorm8: return bytes[channel] / 255.0;
    case Encoding::unorm16: return read<std::uint16_t>(bytes + 2 * channel) / 65535.0;
    case Encoding::snorm8: return (std::max)(-1.0, read<std::int8_t>(bytes + channel) / 127.0);
    case Encoding::snorm16: return (std::max)(-1.0, read<std::int16_t>(bytes + 2 * channel) / 32767.0);
    case Encoding::f32: return read<float>(bytes + 4 * channel);
    case Encoding::half: {
        const unsigned bits = read<std::uint16_t>(bytes + 2 * channel);
        const double magnitude = small_float(bits, 10);
        return bits & 0x8000 ? -magnitude : magnitude;
    }
    case Encoding::rgb10: {
        const unsigned mask = channel == 3 ? 3 : 1023;
        return ((read<std::uint32_t>(bytes) >> (10 * channel)) & mask) / double(mask);
    }
    case Encoding::rgb11f: return small_float(read<std::uint32_t>(bytes) >> (11 * channel), channel == 2 ? 5 : 6);
    case Encoding::rgb9e5: {
        const auto bits = read<std::uint32_t>(bytes);
        return std::ldexp(double((bits >> (9 * channel)) & 511), int(bits >> 27) - 24);
    }
    case Encoding::depth24: return (read<std::uint32_t>(bytes) & 0xffffff) / double(0xffffff);
    default: return 0;
    }
}

class Tree {
    static constexpr unsigned writer_limit = 4096, input_limit = 33, row_limit = 1024, note_limit = 256;
    static constexpr UINT64 alignment = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    static constexpr UINT pitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    static constexpr UINT64 readback_limit = 128ull * 1024 * 1024;
    static constexpr unsigned samples = 17;
    struct Writer {
        ID3D12CommandList* list = nullptr;
        Resource target;
        const char* op = nullptr;
        std::uint64_t key = 0;
        unsigned slot = 0, count = 0;
        Input view;
        Input inputs[input_limit];
        Resource holds[input_limit];
    };
    struct Note {
        ID3D12Resource* resource;
        LONG frame = 0;
        const char* op = nullptr;
        std::uint64_t key = 0;
    };
    struct Row {
        Resource resource;
        Input view;
        LONG previous = 0;
        const char* op = nullptr;
        std::uint64_t key = 0;
        unsigned level = 0, order = 0, output = 0, parent = 0;
        UINT width = 0, height = 0, subresource = 0;
        bool overwritten = false, depth = false;
        Format format;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 offset = 0;
        int value = -1;
    };
    struct Frame {
        LONG number = 0;
        unsigned count = 0;
        std::array<Row, row_limit> rows{};
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        Resource readback;
        UINT64 bytes = 0;
        bool stranded = false;
    };

    Log log_;
    std::array<Writer, writer_limit> staged_{}, submitted_{};
    unsigned staged_count_ = 0, submitted_count_ = 0, note_count_ = 0;
    std::array<Note, note_limit> notes_{};
    std::deque<Frame> pending_;
    LONG frame_ = 0, full_ = 0, stopped_ = 0, previous_record_ = 0;
    std::uint64_t previous_shape_ = 0;
    ID3D12CommandQueue* queue_ = nullptr;
    bool multiple_queues_ = false, note_full_ = false;

    void full(LONG frame)
    {
        if (full_ == frame) return;
        full_ = frame;
        log_("x2w-red table full record=%ld", frame);
    }

    void note(const Writer& writer, LONG frame)
    {
        auto* resource = writer.target.Get();
        if (resource->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return;
        for (unsigned i = 0; i < note_count_; ++i) {
            if (notes_[i].resource != resource) continue;
            notes_[i] = {resource, frame, writer.op, writer.key};
            return;
        }
        if (note_count_ == note_limit) {
            if (!note_full_) full(frame);
            note_full_ = true;
            return;
        }
        notes_[note_count_++] = {resource, frame, writer.op, writer.key};
    }

    void reset_frame(LONG frame)
    {
        for (unsigned i = 0; i < submitted_count_; ++i) {
            note(submitted_[i], frame_);
            submitted_[i] = {};
        }
        submitted_count_ = 0;
        frame_ = frame;
        queue_ = nullptr;
        multiple_queues_ = false;
    }

    unsigned find(ID3D12Resource* resource, unsigned before) const
    {
        while (before) {
            if (submitted_[before - 1].target.Get() == resource) return before;
            --before;
        }
        return 0;
    }

    void walk(Frame& frame, Input input, unsigned level, unsigned before, unsigned parent)
    {
        if (frame.count == row_limit) {
            full(frame.number);
            return;
        }
        const unsigned index = frame.count++;
        Row& row = frame.rows[index];
        row.resource = input.resource;
        row.view = input;
        row.level = level;
        row.parent = parent;
        if (!input.resource) return;
        row.order = find(input.resource, before);
        row.overwritten = find(input.resource, submitted_count_) != row.order;
        if (row.order) {
            const Writer& writer = submitted_[row.order - 1];
            row.op = writer.op;
            row.key = writer.key;
            row.output = writer.slot;
            if (!level && row.view.format == DXGI_FORMAT_UNKNOWN) row.view = writer.view;
            if (level < 3) {
                for (unsigned i = 0; i < writer.count && frame.count < row_limit; ++i)
                    walk(frame, writer.inputs[i], level + 1, row.order - 1, index);
                if (frame.count == row_limit) full(frame.number);
            }
        } else {
            for (unsigned i = 0; i < note_count_; ++i) {
                if (notes_[i].resource != input.resource) continue;
                row.previous = notes_[i].frame;
                row.op = notes_[i].op;
                row.key = notes_[i].key;
                break;
            }
        }
    }

    void layout(ID3D12Device* device, Frame& frame, unsigned index, bool can_copy)
    {
        Row& row = frame.rows[index];
        if (!row.resource) return;
        const auto desc = row.resource->GetDesc();
        if (row.view.format == DXGI_FORMAT_UNKNOWN) {
            row.view.format = x2w_startup::present_view_format(desc.Format);
            if (row.view.format == DXGI_FORMAT_UNKNOWN) row.view.format = desc.Format;
            if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) {
                switch (desc.Format) {
                case DXGI_FORMAT_R16_TYPELESS: row.view.format = DXGI_FORMAT_D16_UNORM; break;
                case DXGI_FORMAT_R32_TYPELESS: row.view.format = DXGI_FORMAT_D32_FLOAT; break;
                case DXGI_FORMAT_R24G8_TYPELESS: row.view.format = DXGI_FORMAT_D24_UNORM_S8_UINT; break;
                case DXGI_FORMAT_R32G8X24_TYPELESS: row.view.format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT; break;
                default: break;
                }
            }
        }
        row.width = static_cast<UINT>((std::max)(UINT64{1}, desc.Width >> (std::min)(row.view.mip, 63u)));
        row.height = (std::max)(1u, desc.Height >> (std::min)(row.view.mip, 31u));
        row.format = sample_format(row.view.format);
        if (!row.format.channels || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
            row.view.mip >= desc.MipLevels || row.view.slice >= desc.DepthOrArraySize || row.view.plane != 0) return;
        row.value = -2;
        if (!can_copy) return;
        row.subresource = row.view.mip + row.view.slice * desc.MipLevels;
        for (unsigned i = 0; i < index; ++i) {
            const Row& other = frame.rows[i];
            if (other.value >= 0 && other.resource.Get() == row.resource.Get() &&
                other.subresource == row.subresource && other.view.format == row.view.format) {
                row.value = other.value;
                return;
            }
        }
        row.depth = (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
        row.offset = frame.bytes;
        UINT64 bytes = samples * alignment;
        if (row.depth) {
            // D3D12 requires a whole subresource copy for a depth/stencil plane.
            device->GetCopyableFootprints(&desc, row.subresource, 1, 0, &row.footprint, nullptr, nullptr, &bytes);
            if (bytes == UINT64_MAX) return;
            row.footprint.Offset = row.offset;
            bytes = (bytes + alignment - 1) & ~(alignment - 1);
        }
        if (bytes > readback_limit - frame.bytes) {
            full(frame.number);
            return;
        }
        row.value = static_cast<int>(index);
        frame.bytes += bytes;
    }

    static UINT coordinate(unsigned sample, UINT size, bool vertical)
    {
        return sample ? x2w_present_signature::sampled_row((sample - 1) / (vertical ? 4 : 1) % 4, size, 4) : size / 2;
    }

    static void copy(Frame& frame, const Row& row)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {row.resource.Get(), row.subresource,
                              D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE};
        frame.list->ResourceBarrier(1, &barrier);
        if (row.depth) {
            x2w_present_signature::copy_sample(frame.list.Get(), row.resource.Get(), row.subresource,
                                               frame.readback.Get(), row.footprint, nullptr);
        } else {
            const auto resource_format = row.resource->GetDesc().Format;
            for (unsigned i = 0; i < samples; ++i) {
                const UINT x = coordinate(i, row.width, false), y = coordinate(i, row.height, true);
                const D3D12_BOX box{x, y, 0, x + 1, y + 1, 1};
                const D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{
                    row.offset + i * alignment, {resource_format, 1, 1, 1, pitch}};
                x2w_present_signature::copy_sample(frame.list.Get(), row.resource.Get(), row.subresource,
                                                   frame.readback.Get(), footprint, &box);
            }
        }
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        frame.list->ResourceBarrier(1, &barrier);
    }

    bool start(ID3D12Device* device, ID3D12CommandQueue* queue, Frame& frame)
    {
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = frame.bytes;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&frame.readback))) ||
            FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&frame.allocator))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, frame.allocator.Get(), nullptr,
                IID_PPV_ARGS(&frame.list))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&frame.fence)))) return false;
        x2w_budget::committed(device, frame.readback.Get(), D3D12_HEAP_TYPE_READBACK);
        for (unsigned i = 0; i < frame.count; ++i)
            if (frame.rows[i].value == static_cast<int>(i)) copy(frame, frame.rows[i]);
        if (FAILED(frame.list->Close())) return false;
        ID3D12CommandList* lists[]{frame.list.Get()};
        queue->ExecuteCommandLists(1, lists);
        frame.stranded = FAILED(queue->Signal(frame.fence.Get(), 1));
        return true;
    }

    static bool describe(const Row& row, const unsigned char* memory, char (&text)[384])
    {
        bool zero = true;
        char* cursor = text;
        for (unsigned c = 0; c < row.format.channels; ++c) {
            double center = 0, lo = 0, hi = 0, sum = 0;
            for (unsigned i = 0; i < samples; ++i) {
                const UINT64 offset = row.depth
                    ? row.footprint.Offset + UINT64{row.footprint.Footprint.RowPitch} * coordinate(i, row.height, true)
                        + row.format.bytes * coordinate(i, row.width, false)
                    : row.offset + i * alignment;
                const double value = component(memory + offset, row.format, c);
                zero &= value == 0;
                if (!i) {
                    center = value;
                    continue;
                }
                lo = i == 1 || std::isnan(value) ? value : (std::min)(lo, value);
                hi = i == 1 || std::isnan(value) ? value : (std::max)(hi, value);
                sum += value;
            }
            const int written = std::snprintf(cursor, sizeof(text) - (cursor - text),
                "%sc%u(center=%.7g,min=%.7g,max=%.7g,mean=%.7g)", c ? " " : "", c, center, lo, hi, sum / 16);
            if (written < 0 || std::size_t(written) >= sizeof(text) - (cursor - text)) break;
            cursor += written;
        }
        return zero;
    }

    void print(Frame& frame, const unsigned char* memory)
    {
        // Record numbers, resource addresses and nonzero magnitudes may change without changing the tree.
        std::uint64_t shape = 14695981039346656037ull;
        const auto mix = [&](std::uint64_t v) { shape = (shape ^ v) * 1099511628211ull; };
        std::array<bool, row_limit> zero{};
        for (unsigned i = 0; i < frame.count; ++i) {
            const auto& row = frame.rows[i];
            char value[384]{};
            zero[i] = row.value >= 0 && memory && describe(frame.rows[row.value], memory, value);
            mix(row.parent); mix(row.level); mix(row.view.slot); mix(row.output); mix(row.key);
            mix(row.order != 0); mix(row.previous != 0); mix(row.overwritten);
            mix(row.view.format); mix(row.width); mix(row.height);
            mix(zero[i] ? 0 : row.value >= 0 && memory ? 1 : row.value == -1 ? 2 : 3);
            if (row.op) for (const char* c = row.op; *c; ++c) mix(static_cast<unsigned char>(*c));
            mix(0);
        }
        if (previous_record_ && shape == previous_shape_) {
            log_("x2w-red tree record=%ld same as record=%ld", frame.number, previous_record_);
            previous_record_ = frame.number;
            return;
        }
        previous_shape_ = shape;
        previous_record_ = frame.number;
        for (unsigned i = 0; i < frame.count; ++i) {
            const Row& row = frame.rows[i];
            char value[384]{}, writer[112]{};
            if (row.value >= 0 && memory) describe(frame.rows[row.value], memory, value);
            else if (!row.resource) std::snprintf(value, sizeof(value), "unbound");
            else if (row.value == -1) std::snprintf(value, sizeof(value), "unreadable format %d", row.view.format);
            else std::snprintf(value, sizeof(value), "readback unavailable");
            if (row.order) std::snprintf(writer, sizeof(writer), "%s %016llx output=%u order=%u",
                row.op, static_cast<unsigned long long>(row.key), row.output, row.order);
            else if (row.previous) std::snprintf(writer, sizeof(writer), "last written in record %ld (%s %016llx)",
                row.previous, row.op, static_cast<unsigned long long>(row.key));
            else std::snprintf(writer, sizeof(writer), row.overwritten ? "none before this reader" : "none in this frame");
            if (!row.level)
                log_("x2w-red tree record=%ld level=0 target=%p writer=%s value=%s%s",
                    frame.number, row.resource.Get(), writer, value, zero[i] ? " ZERO" : "");
            else
                log_("x2w-red tree record=%ld level=%u input slot=%u res=%p fmt=%d size=%ux%u value=%s writer=%s%s%s%s node=%u parent=%u",
                    frame.number, row.level, row.view.slot, row.resource.Get(), row.view.format, row.width, row.height,
                    value, writer, zero[i] ? " ZERO" : "", row.order || row.overwritten ? "" : " UNWRITTEN",
                    row.overwritten ? " WRITTEN LATER IN FRAME" : "", i, row.parent);
        }
    }

public:
    explicit Tree(Log log) : log_(log) {}

    void record(LONG frame, ID3D12GraphicsCommandList* list, Input target, const char* op, std::uint64_t key,
                const Input* inputs = nullptr, unsigned count = 0)
    {
        if (!target.resource || stopped_ == frame) return;
        if (staged_count_ == writer_limit || count > input_limit) {
            stopped_ = frame;
            full(frame);
            return;
        }
        Writer& writer = staged_[staged_count_++];
        writer.list = list;
        writer.target = target.resource;
        writer.view = target;
        writer.op = op;
        writer.key = key;
        writer.slot = target.slot;
        writer.count = count;
        for (unsigned i = 0; i < count; ++i) {
            writer.inputs[i] = inputs[i];
            writer.holds[i] = inputs[i].resource;
        }
    }

    void forget(ID3D12Resource* resource)
    {
        for (unsigned i = 0; i < note_count_; ++i) {
            if (notes_[i].resource != resource) continue;
            notes_[i] = notes_[--note_count_];
            note_full_ = false;
            return;
        }
    }

    void discard(ID3D12GraphicsCommandList* list)
    {
        unsigned to = 0;
        for (unsigned from = 0; from < staged_count_; ++from) {
            if (staged_[from].list == list) continue;
            if (to != from) staged_[to] = std::move(staged_[from]);
            ++to;
        }
        while (staged_count_ > to) staged_[--staged_count_] = {};
    }

    void submit(LONG frame, ID3D12CommandQueue* queue, ID3D12CommandList* list, bool capture)
    {
        if (frame_ != frame) reset_frame(frame);
        if (capture) {
            if (!queue_) queue_ = queue;
            else multiple_queues_ |= queue_ != queue;
        }
        for (unsigned i = 0; i < staged_count_; ++i) {
            const Writer& writer = staged_[i];
            if (writer.list != list) continue;
            if (!capture) note(writer, frame);
            else if (submitted_count_ < writer_limit) submitted_[submitted_count_++] = writer;
            else { full(frame); break; }
        }
    }

    void present(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Resource* surface, LONG record)
    {
        flush();
        if (pending_.size() >= 8) {
            full(record);
            reset_frame(record + 1);
            return;
        }
        try {
            pending_.emplace_back();
        } catch (const std::bad_alloc&) {
            full(record);
            reset_frame(record + 1);
            return;
        }
        Frame& frame = pending_.back();
        frame.number = record;
        if (frame_ != record) reset_frame(record);
        walk(frame, {surface}, 0, submitted_count_, 0);
        const bool can_copy = !multiple_queues_ && (!queue_ || queue_ == queue);
        if (!can_copy) log_("x2w-red tree record=%ld readback unavailable across queues", record);
        for (unsigned i = 0; i < frame.count; ++i) layout(device, frame, i, can_copy);
        if (frame.bytes && !start(device, queue, frame)) frame.readback.Reset();
        reset_frame(record + 1);
        flush();
    }

    void flush()
    {
        while (!pending_.empty()) {
            Frame& frame = pending_.front();
            if (frame.stranded) {
                // Signal failed after execution. Preserve the GPU objects and let later frames publish.
                for (unsigned i = 0; i < frame.count; ++i) frame.rows[i].resource.Detach();
                frame.list.Detach();
                frame.allocator.Detach();
                frame.readback.Detach();
                log_("x2w-red tree record=%ld fence signal failed", frame.number);
            }
            if (frame.readback && frame.fence->GetCompletedValue() == 0) break;
            void* memory = nullptr;
            const D3D12_RANGE read_range{0, static_cast<SIZE_T>(frame.bytes)}, none{};
            if (frame.readback && frame.fence->GetCompletedValue() == 1)
                frame.readback->Map(0, &read_range, &memory);
            print(frame, static_cast<const unsigned char*>(memory));
            if (memory) frame.readback->Unmap(0, &none);
            pending_.pop_front();
        }
    }

    void finish()
    {
        while (submitted_count_) submitted_[--submitted_count_] = {};
        while (staged_count_) staged_[--staged_count_] = {};
        while (note_count_) notes_[--note_count_] = {};
    }
};

}  // namespace x2w_red
