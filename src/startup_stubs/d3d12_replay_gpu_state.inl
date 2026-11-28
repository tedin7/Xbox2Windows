// Included after the resource retirement and command-list state trackers.
namespace replay_gpu_state {
using x2w_startup::ComPtr;
using x2w_gpu_state::Header;

constexpr UINT writable = UINT(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) |
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

bool copyable(const D3D12_RESOURCE_DESC& desc)
{
    return !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) &&
        desc.SampleDesc.Count == 1 && desc.Dimension >= D3D12_RESOURCE_DIMENSION_BUFFER &&
        desc.Dimension <= D3D12_RESOURCE_DIMENSION_TEXTURE3D;
}

bool snapshot_candidate(const WrapperOwn& own, ID3D12Resource* resource, const D3D12_RESOURCE_DESC& desc)
{
    if (desc.Flags & writable) return true;
    D3D12_HEAP_PROPERTIES heap{};
    D3D12_HEAP_FLAGS flags{};
    return !own.title_memory && SUCCEEDED(resource->GetHeapProperties(&heap, &flags)) &&
        heap.Type == D3D12_HEAP_TYPE_DEFAULT;
}

bool common_resource(u64 object, ID3D12Resource* resource)
{
    D3D12_HEAP_TYPE heap;
    return render_mode() && !fixed_heap(resource, heap) && tracked_from_common(object);
}

bool wait(ID3D12Fence* fence, ULONGLONG deadline, UINT64 value = 1)
{
    do {
        const UINT64 completed = fence->GetCompletedValue();
        if (completed >= value) return completed != UINT64_MAX;
        Sleep(1);
    } while (GetTickCount64() < deadline);
    return false;
}

bool signal(ID3D12CommandQueue* queue, ComPtr<ID3D12Fence>& fence)
{
    ComPtr<ID3D12Device> device;
    if (FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return false;
    if (SUCCEEDED(queue->Signal(fence.Get(), 1))) return true;
    fence.Reset();
    return false;
}

bool idle(const std::vector<ComPtr<ID3D12CommandQueue>>& queues)
{
    std::vector<ComPtr<ID3D12Fence>> fences(queues.size());
    bool ok = true;
    for (std::size_t i = 0; i < queues.size(); ++i)
        ok = signal(queues[i].Get(), fences[i]) && ok;
    const auto deadline = GetTickCount64() + 10000;
    for (auto& fence : fences) {
        if (fence && !wait(fence.Get(), deadline)) {
            fence.Detach();  // A queue may still have the signal pending.
            ok = false;
        }
    }
    return ok;
}

struct Copy {
    ComPtr<ID3D12Resource> resource, staging;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    UINT64 value = 0;
    bool fresh = true;  // the list from open() is still recording its first batch

    bool begin()
    {
        if (fresh) {
            fresh = false;
            return true;
        }
        return SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.Get(), nullptr));
    }

    bool open(ID3D12Device* device)
    {
        return SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) &&
            SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                IID_PPV_ARGS(&list))) &&
            SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    }
};

bool submit(std::unique_ptr<Copy>& copy)
{
    if (FAILED(copy->list->Close())) return false;
    ID3D12CommandList* lists[]{copy->list.Get()};
    copy->queue->ExecuteCommandLists(1, lists);
    const UINT64 value = ++copy->value;
    if (SUCCEEDED(copy->queue->Signal(copy->fence.Get(), value)) &&
        wait(copy->fence.Get(), GetTickCount64() + 10000, value))
        return true;
    // Without completion the GPU can still refer to these objects.
    copy.release();
    return false;
}

Header header_for(u64 object, UINT subresource, const replay_dump::Slice& slice)
{
    const auto& f = slice.footprint.Footprint;
    const UINT64 rows = UINT64(slice.rows) * f.Depth;
    return {object, subresource, static_cast<UINT>(f.Format), 0, f.Width, f.Height, f.Depth,
        f.RowPitch, (rows - 1) * f.RowPitch + slice.row_bytes};
}

Header buffer_header(u64 object, UINT64 bytes)
{
    return {object, 0, DXGI_FORMAT_UNKNOWN, 0, bytes <= UINT32_MAX ? static_cast<UINT>(bytes) : 0, 1, 1, 0, bytes};
}

// Snapshot streaming. The title's resources are read through one reusable READBACK (system memory) staging buffer of at
// most `staging` bytes, so the snapshot never asks the GPU's local memory for a copy of a resource.
constexpr u64 stage_unit = 4096;
constexpr u64 default_staging = 64ull << 20;

struct Stage {
    ComPtr<ID3D12Resource> buffer;
    u64 bytes = 0;
};

struct Piece {
    UINT z0, z1, r0, r1;  // depth slices and block rows [r0, r1) of a texture subresource
    bool whole;
};

bool planar(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_NV12 || format == DXGI_FORMAT_P010 || format == DXGI_FORMAT_P016 ||
        format == DXGI_FORMAT_420_OPAQUE || format == DXGI_FORMAT_NV11;
}

UINT texel_rows_per_block_row(DXGI_FORMAT format)
{
    const UINT f = static_cast<UINT>(format);
    return (f >= 70 && f <= 84) || (f >= 94 && f <= 99) ? 4 : 1;  // BC1..BC5, BC6H, BC7
}

// Calls fn(Piece) in file order for a subresource that must fit `staging`. False when it cannot be cut small enough.
template <class F>
bool each_piece(const replay_dump::Slice& slice, u64 staging, F&& fn)
{
    const auto& f = slice.footprint.Footprint;
    const u64 slice_bytes = u64(slice.rows) * f.RowPitch;
    if (slice_bytes * f.Depth <= staging) return fn(Piece{0, f.Depth, 0, slice.rows, true});
    if (planar(f.Format) || f.RowPitch > staging) return false;
    if (slice_bytes <= staging) {
        const UINT per = static_cast<UINT>(staging / slice_bytes);
        for (UINT z = 0; z < f.Depth; z += per)
            if (!fn(Piece{z, (std::min)(z + per, f.Depth), 0, slice.rows, false})) return false;
        return true;
    }
    const UINT per_rows = static_cast<UINT>(staging / f.RowPitch);
    for (UINT z = 0; z < f.Depth; ++z)
        for (UINT r = 0; r < slice.rows; r += per_rows)
            if (!fn(Piece{z, z + 1, r, (std::min)(r + per_rows, slice.rows), false})) return false;
    return true;
}

bool fits(const replay_dump::Readback& readback, u64 staging)
{
    if (readback.desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return staging >= stage_unit;
    for (const auto& slice : readback.slices)
        if (!each_piece(slice, staging, [](const Piece&) { return true; })) return false;
    return true;
}

// Records one piece's copy (the staging buffer is COPY_DEST for good); the resource goes COMMON -> COPY_SOURCE -> COMMON.
void record_piece(ID3D12GraphicsCommandList* list, ID3D12Resource* source, ID3D12Resource* stage, UINT subresource,
                  const replay_dump::Slice& slice, const Piece& piece)
{
    transition(list, source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    from.SubresourceIndex = subresource;
    D3D12_TEXTURE_COPY_LOCATION to{stage, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    to.PlacedFootprint = slice.footprint;
    to.PlacedFootprint.Offset = 0;
    if (piece.whole) {
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    } else {
        const auto& f = slice.footprint.Footprint;
        const UINT block = texel_rows_per_block_row(f.Format);
        const UINT top = piece.r0 * block, bottom = (std::min)(f.Height, piece.r1 * block);
        const D3D12_BOX box{0, top, piece.z0, f.Width, bottom, piece.z1};
        to.PlacedFootprint.Footprint.Height = (piece.r1 - piece.r0) * block;
        to.PlacedFootprint.Footprint.Depth = piece.z1 - piece.z0;
        list->CopyTextureRegion(&to, 0, 0, 0, &from, &box);
    }
    transition(list, source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

void record_buffer_piece(ID3D12GraphicsCommandList* list, ID3D12Resource* source, ID3D12Resource* stage, u64 offset,
                         u64 bytes)
{
    transition(list, source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(stage, 0, source, offset, bytes);
    transition(list, source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
}

// The local video memory group's headroom for a staging buffer: 90% of the budget minus the usage. False when the
// adapter cannot be asked (no limit then). X2W_TEST_VRAM_BUDGET_MIB (a decimal) replaces the budget and counts no usage.
bool vram_headroom(ID3D12Device* device, u64& headroom, u64& used, u64& budget)
{
    char text[32]{};
    const DWORD length = GetEnvironmentVariableA("X2W_TEST_VRAM_BUDGET_MIB", text, sizeof(text));
    if (length && length < sizeof(text)) {
        char* end = nullptr;
        const double mib = std::strtod(text, &end);
        if (end != text && !*end && mib >= 0 && mib < 1e9) {
            used = 0;
            budget = static_cast<u64>(mib * double(1 << 20));
            headroom = budget / 10 * 9;
            return true;
        }
    }
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter3> adapter;
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter))) ||
        FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) return false;
    used = info.CurrentUsage;
    budget = info.Budget;
    headroom = budget / 10 * 9 > used ? budget / 10 * 9 - used : 0;
    return true;
}

u64 decimal_mib_env(const char* name, u64 fallback)
{
    char text[32]{};
    const DWORD length = GetEnvironmentVariableA(name, text, sizeof(text));
    if (!length || length >= sizeof(text)) return fallback;
    char* end = nullptr;
    const double mib = std::strtod(text, &end);
    if (end == text || *end || mib < 0 || mib > 1e9) return fallback;
    return static_cast<u64>(mib * double(1 << 20));
}

bool make_stage(ID3D12Device* device, u64 bytes, Stage& stage)
{
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    for (bytes &= ~(stage_unit - 1); bytes >= stage_unit; bytes = (bytes / 2) & ~(stage_unit - 1)) {
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1,
            DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        if (SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&stage.buffer)))) {
            stage.bytes = bytes;
            return true;
        }
    }
    return false;
}

bool restore(void* object, UINT subresource, const void* footprint_header, const void* data, UINT64 bytes)
{
    auto* wrapper = as_wrapper(object);
    if (!wrapper || wrapper->own().kind != resource_kind || !wrapper->own().real ||
        wrapper->own().gpu_state_unsupported ||
        !footprint_header || !data || !bytes || bytes > SIZE_MAX) return false;
    auto* resource = static_cast<ID3D12Resource*>(wrapper->own().real);
    const auto desc = resource->GetDesc();
    const bool buffer = desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
    if (!copyable(desc) || !snapshot_candidate(wrapper->own(), resource, desc) ||
        !common_resource(reinterpret_cast<u64>(object), resource)) return false;
    ComPtr<ID3D12Device> device;
    if (FAILED(resource->GetDevice(IID_PPV_ARGS(&device)))) return false;

    Header supplied{};
    std::memcpy(&supplied, footprint_header, sizeof(supplied));
    replay_dump::Slice slice;
    UINT64 required = desc.Width;
    if (buffer) {
        if (subresource) return false;
    } else {
        if (subresource >= replay_dump::subresource_count(device.Get(), desc)) return false;
        device->GetCopyableFootprints(&desc, subresource, 1, 0, &slice.footprint, &slice.rows, &slice.row_bytes, &required);
        if (!required || required == UINT64_MAX || !slice.rows || !slice.footprint.Footprint.Depth) return false;
    }
    const Header expected = buffer ? buffer_header(supplied.object, required) : header_for(supplied.object, subresource, slice);
    if (bytes != required || std::memcmp(&supplied, &expected, sizeof(expected))) return false;

    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    auto copy = std::make_unique<Copy>();
    copy->resource = ComPtr<ID3D12Resource>(resource);
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_RESOURCE_DESC staging{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&copy->queue))) || !copy->open(device.Get()) ||
        FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &staging,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&copy->staging)))) return false;
    void* mapped = nullptr;
    const D3D12_RANGE none{0, 0};
    if (FAILED(copy->staging->Map(0, &none, &mapped))) return false;
    std::memcpy(mapped, data, static_cast<std::size_t>(bytes));
    copy->staging->Unmap(0, nullptr);
    // As in replay_dump::record, the shim starts and ends each submitted list in COMMON.
    transition(copy->list.Get(), resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    if (buffer) {
        copy->list->CopyBufferRegion(resource, 0, copy->staging.Get(), 0, bytes);
    } else {
        D3D12_TEXTURE_COPY_LOCATION to{resource, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        to.SubresourceIndex = subresource;
        D3D12_TEXTURE_COPY_LOCATION from{copy->staging.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        from.PlacedFootprint = slice.footprint;
        copy->list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    transition(copy->list.Get(), resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    return submit(copy);
}
}  // namespace replay_gpu_state

namespace replay_record {

// file_lock held. One GPUR chunk's head and header.
bool write_gpu_head(const x2w_gpu_state::Header& header)
{
    const struct { char tag[4]; std::uint32_t header_bytes; u64 data_bytes; } chunk{
        {'G', 'P', 'U', 'R'}, sizeof(header), header.bytes};
    const u64 start = bytes_written;
    write_raw(&chunk, sizeof(chunk));
    write_raw(&header, sizeof(header));
    return bytes_written - start == sizeof(chunk) + sizeof(header);
}

// Copies one piece into the staging buffer, waits for it and maps the first `bytes`. Unmap with `none` after reading.
bool copy_piece(std::unique_ptr<replay_gpu_state::Copy>& copy, replay_gpu_state::Stage& stage, u64 bytes,
                const void*& mapped)
{
    if (!replay_gpu_state::submit(copy)) return false;
    void* data = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)};
    if (FAILED(stage.buffer->Map(0, &range, &data))) return false;
    mapped = data;
    return true;
}

enum class Streamed { ok, failed };

// Writes one resource's GPUR chunks piece by piece through the staging buffer, holding file_lock so no other record lands
// inside a chunk, and cuts the file back to the resource's start when a piece fails.
Streamed write_gpu_resource(ThreadBuffer& b, replay_dump::Readback& readback, u64 bytes,
                            std::unique_ptr<replay_gpu_state::Copy>& copy, replay_gpu_state::Stage& stage, u64 staging,
                            std::vector<unsigned char>& payload, u64& payload_bytes)
{
    using namespace replay_gpu_state;
    AcquireSRWLockExclusive(&file_lock);
    const u64 before_drain = bytes_written, buffered = b.bytes.size();
    drain(b);
    const bool drained = bytes_written == before_drain + buffered;
    const u64 start = drained ? bytes_written : before_drain;
    u64 written = 0;
    bool ok = drained;
    auto* resource = readback.target.resource.Get();
    const D3D12_RANGE none{0, 0};
    const auto piece_copy = [&](u64 piece_bytes, auto&& record, auto&& consume) {
        if (!copy->begin()) return false;
        record(copy->list.Get());
        const void* mapped = nullptr;
        if (!copy_piece(copy, stage, piece_bytes, mapped)) return false;
        consume(static_cast<const unsigned char*>(mapped));
        stage.buffer->Unmap(0, &none);
        return true;
    };
    if (ok && readback.desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        const auto header = buffer_header(readback.target.id, bytes);
        ok = write_gpu_head(header);
        for (u64 at = 0; ok && at < bytes; at += staging) {
            const u64 part = (std::min)(staging, bytes - at);
            ok = piece_copy(part,
                [&](ID3D12GraphicsCommandList* list) { record_buffer_piece(list, resource, stage.buffer.Get(), at, part); },
                [&](const unsigned char* mapped) { write_raw(mapped, static_cast<std::size_t>(part)); });
        }
        written = header.bytes;
        ok = ok && bytes_written - start == sizeof(char[4]) + sizeof(std::uint32_t) + sizeof(u64) + sizeof(header) + bytes;
    }
    for (UINT i = 0; i < readback.slices.size() && ok; ++i) {
        const auto& slice = readback.slices[i];
        const auto header = header_for(readback.target.id, i, slice);
        const auto& f = slice.footprint.Footprint;
        const u64 last_row = u64(slice.rows) * f.Depth - 1;
        u64 row = 0;
        const u64 slice_start = bytes_written;
        ok = write_gpu_head(header) && each_piece(slice, staging, [&](const Piece& piece) {
            const u64 rows = u64(piece.z1 - piece.z0) * (piece.r1 - piece.r0);
            const u64 piece_bytes = rows * f.RowPitch;
            return piece_copy(piece_bytes,
                [&](ID3D12GraphicsCommandList* list) {
                    record_piece(list, resource, stage.buffer.Get(), i, slice, piece);
                },
                [&](const unsigned char* mapped) {
                    // Serialize footprint padding as zeros, independent of the staging allocation.
                    payload.assign(static_cast<std::size_t>(piece_bytes), 0);
                    for (u64 k = 0; k < rows; ++k)
                        std::memcpy(payload.data() + k * f.RowPitch, mapped + k * f.RowPitch,
                                    static_cast<std::size_t>(slice.row_bytes));
                    const u64 end = row + rows == last_row + 1 ? piece_bytes - (f.RowPitch - slice.row_bytes) : piece_bytes;
                    write_raw(payload.data(), static_cast<std::size_t>(end));
                    row += rows;
                });
        });
        written += header.bytes;
        ok = ok && bytes_written - slice_start == sizeof(char[4]) + sizeof(std::uint32_t) + sizeof(u64) + sizeof(header) + header.bytes;
    }
    if (ok) {
        whole_bytes = bytes_written;
        payload_bytes += written;
    } else {
        LARGE_INTEGER at{};
        at.QuadPart = static_cast<LONGLONG>(start);
        whole_bytes = start;
        if (file != INVALID_HANDLE_VALUE && SetFilePointerEx(file, at, nullptr, FILE_BEGIN) && SetEndOfFile(file))
            bytes_written = start;
        InterlockedExchange(&stop_requested, 1);
    }
    ReleaseSRWLockExclusive(&file_lock);
    return ok ? Streamed::ok : Streamed::failed;
}

u64 gpu_state_at_window_open(ThreadBuffer& b)
{
    using namespace replay_gpu_state;
    const auto start = GetTickCount64();
    u64 budget = 1024ull << 20;
    char value[32]{};
    const DWORD length = GetEnvironmentVariableA("X2W_REPLAY_GPU_STATE_MAX_MIB", value, sizeof(value));
    if (length && length < sizeof(value) && value[0] >= '0' && value[0] <= '9') {
        char* end = nullptr;
        const auto mib = std::strtoull(value, &end, 10);
        if (!*end && mib <= (UINT64_MAX >> 20)) budget = mib << 20;
    }
    u64 written = 0, payload_bytes = 0, used = 0, over_budget = 0, msaa = 0, depth = 0, unsupported = 0, failed = 0;
    u64 over_vram = 0, over_time = 0, peak_staging = 0;
    bool idle_ok = false;
    u64 seq = 0;
    try {
        // Present holds draw_lock. New render submissions wait until the snapshot is complete.
        std::vector<ComPtr<ID3D12CommandQueue>> queues;
        std::vector<replay_dump::Readback> candidates;
        AcquireSRWLockShared(&registry_lock);
        // A resource registered during the window cannot enter this candidate list.
        seq = static_cast<u64>(InterlockedIncrement64(&next_seq));
        update_window(true);
        AcquireSRWLockShared(&retire_lock);
        try {
            for (const void* entry : registry()) {
                auto& own = static_cast<const Wrapper*>(entry)->own();
                if (!own.real) continue;
                if (own.kind == queue_kind) {
                    queues.emplace_back(static_cast<ID3D12CommandQueue*>(own.real));
                } else if (own.kind == resource_kind && InterlockedCompareExchange(&own.refs, 0, 0) > 0) {
                    auto* real = static_cast<ID3D12Resource*>(own.real);
                    const auto desc = real->GetDesc();
                    if (!snapshot_candidate(own, real, desc)) continue;
                    if (desc.SampleDesc.Count != 1) ++msaa;
                    else if (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) ++depth;
                    else if (own.gpu_state_unsupported) ++unsupported;
                    else candidates.push_back({{reinterpret_cast<u64>(entry), ComPtr<ID3D12Resource>(real),
                        DXGI_FORMAT_UNKNOWN}, desc, {}, {}, {}});
                }
            }
        } catch (...) {
            ReleaseSRWLockShared(&retire_lock);
            ReleaseSRWLockShared(&registry_lock);
            throw;
        }
        ReleaseSRWLockShared(&retire_lock);
        ReleaseSRWLockShared(&registry_lock);

        struct Item {
            replay_dump::Readback* readback;
            ComPtr<ID3D12Device> device;
            u64 bytes;
        };
        std::vector<Item> items;
        for (auto& readback : candidates) {
            auto* resource = readback.target.resource.Get();
            Item item{&readback, {}, 0};
            if (!copyable(readback.desc) ||
                !common_resource(readback.target.id, resource) ||
                FAILED(resource->GetDevice(IID_PPV_ARGS(&item.device))) ||
                !replay_dump::layout(item.device.Get(), readback.desc, readback.slices, item.bytes)) {
                ++unsupported;
            } else {
                items.push_back(std::move(item));
            }
        }
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            return a.bytes != b.bytes ? a.bytes < b.bytes : a.readback->target.id < b.readback->target.id;
        });
        idle_ok = items.empty() || idle(queues);
        std::unordered_map<ID3D12Device*, ComPtr<ID3D12CommandQueue>> copy_queues;
        std::unordered_map<ID3D12Device*, Stage> stages;
        u64 largest = 0;
        for (const auto& item : items) largest = (std::max)(largest, item.bytes);
        const u64 configured_staging = (std::max)(stage_unit, decimal_mib_env("X2W_REPLAY_GPU_STATE_STAGING_MIB", default_staging));
        // Optional bound on the stall the title sees: resources not started by then are left out of the snapshot.
        u64 max_ms = 0;
        char ms_text[16]{};
        if (GetEnvironmentVariableA("X2W_REPLAY_GPU_STATE_MAX_MS", ms_text, sizeof(ms_text)) &&
            ms_text[0] >= '0' && ms_text[0] <= '9') max_ms = std::strtoull(ms_text, nullptr, 10);
        std::vector<unsigned char> payload;
        for (auto& item : items) {
            const u64 headers = 64 * (std::max)(std::size_t{1}, item.readback->slices.size());
            AcquireSRWLockExclusive(&file_lock);
            const bool file_budget = cap_exceeded(b.bytes.size() + item.bytes + headers);
            ReleaseSRWLockExclusive(&file_lock);
            if (item.bytes > budget - used || file_budget) {
                ++over_budget;
                continue;
            }
            if (!idle_ok) {
                ++failed;
                continue;
            }
            if (max_ms && GetTickCount64() - start >= max_ms) {
                ++over_time;
                continue;
            }
            // The staging buffer lives in system memory; it is also kept inside what the local video memory has left.
            u64 headroom = 0, vram_used = 0, vram_budget = 0, staging = configured_staging;
            if (vram_headroom(item.device.Get(), headroom, vram_used, vram_budget))
                staging = (std::min)(staging, headroom & ~(stage_unit - 1));
            auto& stage = stages[item.device.Get()];
            if (!stage.buffer) {
                // Made once, as large as the biggest resource needs and the limits allow, and kept for all of them.
                const u64 wanted = (std::min)(staging, (largest + stage_unit - 1) & ~(stage_unit - 1));
                if (wanted < stage_unit || !make_stage(item.device.Get(), wanted, stage)) {
                    ++over_vram;
                    continue;
                }
            }
            staging = (std::min)(staging, stage.bytes);
            if (staging < stage_unit || !fits(*item.readback, staging)) {
                ++over_vram;
                continue;
            }
            peak_staging = (std::max)(peak_staging, stage.bytes);
            auto& readback = *item.readback;
            auto& queue = copy_queues[item.device.Get()];
            const D3D12_COMMAND_QUEUE_DESC queue_desc{};
            auto copy = std::make_unique<Copy>();
            copy->resource = readback.target.resource;
            if ((!queue && FAILED(item.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)))) ||
                !copy->open(item.device.Get())) {
                ++failed;
                continue;
            }
            copy->queue = queue;
            const bool ok = write_gpu_resource(b, readback, item.bytes, copy, stage, staging, payload,
                                               payload_bytes) == Streamed::ok;
            if (!ok) {
                // A timed-out queue may still be accessing the resource. Stop issuing copies.
                ++failed;
                if (!copy) idle_ok = false;
                break;
            }
            ++written;
            used += item.bytes;
        }
    } catch (const std::bad_alloc&) {
        ++failed;
    }
    log("x2w-replay gpu state: %llu resources (%.2f MiB) written, %llu skipped (budget), %llu skipped (multisampled), "
        "took %llu ms; %llu skipped (depth-stencil), %llu skipped (unsupported), %llu failed, queues %s",
        written, static_cast<double>(payload_bytes) / (1ull << 20), over_budget, msaa, GetTickCount64() - start,
        depth, unsupported, failed, idle_ok ? "idle" : "idle/copy wait incomplete");
    log("x2w-replay gpu state: %llu skipped (vram budget), %llu skipped (time limit), peak staging %llu KiB (system memory)",
        over_vram, over_time, peak_staging >> 10);
    return seq;
}
}  // namespace replay_record
