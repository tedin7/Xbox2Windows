// Included after the resource retirement and command-list state trackers.
namespace replay_gpu_state {
using x2w_startup::ComPtr;
using x2w_gpu_state::Header;

constexpr UINT writable = UINT(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) |
    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

bool copyable(const D3D12_RESOURCE_DESC& desc)
{
    return (desc.Flags & writable) && !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) &&
        desc.SampleDesc.Count == 1 && desc.Dimension >= D3D12_RESOURCE_DIMENSION_BUFFER &&
        desc.Dimension <= D3D12_RESOURCE_DIMENSION_TEXTURE3D;
}

bool common_resource(u64 object, ID3D12Resource* resource)
{
    D3D12_HEAP_TYPE heap;
    return render_mode() && !fixed_heap(resource, heap) && tracked_from_common(object);
}

bool wait(ID3D12Fence* fence, ULONGLONG deadline)
{
    do {
        const UINT64 completed = fence->GetCompletedValue();
        if (completed) return completed != UINT64_MAX;
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
    if (SUCCEEDED(copy->queue->Signal(copy->fence.Get(), 1)) && wait(copy->fence.Get(), GetTickCount64() + 10000))
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

bool restore(void* object, UINT subresource, const void* footprint_header, const void* data, UINT64 bytes)
{
    auto* wrapper = as_wrapper(object);
    if (!wrapper || wrapper->own().kind != resource_kind || !wrapper->own().real ||
        wrapper->own().gpu_state_unsupported ||
        !footprint_header || !data || !bytes || bytes > SIZE_MAX) return false;
    auto* resource = static_cast<ID3D12Resource*>(wrapper->own().real);
    const auto desc = resource->GetDesc();
    const bool buffer = desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
    if (!copyable(desc) || !common_resource(reinterpret_cast<u64>(object), resource)) return false;
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

// file_lock held. Commit the complete resource or cut the file back to its first chunk.
bool write_gpu_slice(const x2w_gpu_state::Header& header, const void* data)
{
    const struct { char tag[4]; std::uint32_t header_bytes; u64 data_bytes; } chunk{
        {'G', 'P', 'U', 'R'}, sizeof(header), header.bytes};
    const u64 start = bytes_written;
    write_raw(&chunk, sizeof(chunk));
    write_raw(&header, sizeof(header));
    write_raw(data, static_cast<std::size_t>(header.bytes));
    return bytes_written - start == sizeof(chunk) + sizeof(header) + header.bytes;
}

bool write_gpu_resource(ThreadBuffer& b, const replay_dump::Readback& readback, u64 bytes,
                        const unsigned char* data, std::vector<unsigned char>& payload, u64& payload_bytes)
{
    using namespace replay_gpu_state;
    AcquireSRWLockExclusive(&file_lock);
    const u64 before_drain = bytes_written, buffered = b.bytes.size();
    drain(b);
    const bool drained = bytes_written == before_drain + buffered;
    const u64 start = drained ? bytes_written : before_drain;
    u64 written = 0;
    bool ok = drained;
    if (ok && readback.desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        const auto header = buffer_header(readback.target.id, bytes);
        ok = write_gpu_slice(header, data);
        written = header.bytes;
    }
    for (UINT i = 0; i < readback.slices.size() && ok; ++i) {
        const auto& slice = readback.slices[i];
        const auto header = header_for(readback.target.id, i, slice);
        // Serialize footprint padding as zeros, independent of the staging allocation.
        std::memset(payload.data(), 0, static_cast<std::size_t>(header.bytes));
        for (u64 row = 0; row < u64(slice.rows) * header.depth; ++row) {
            const u64 at = row * header.row_pitch;
            std::memcpy(payload.data() + at, data + slice.footprint.Offset + at,
                        static_cast<std::size_t>(slice.row_bytes));
        }
        ok = write_gpu_slice(header, payload.data());
        written += header.bytes;
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
    return ok;
}

void gpu_state_at_window_open(ThreadBuffer& b)
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
    bool idle_ok = false;
    try {
        // Present holds draw_lock. New render submissions wait until the snapshot is complete.
        std::vector<ComPtr<ID3D12CommandQueue>> queues;
        std::vector<replay_dump::Readback> candidates;
        AcquireSRWLockShared(&registry_lock);
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
                    if (!(desc.Flags & writable)) continue;
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
            auto& readback = *item.readback;
            auto& queue = copy_queues[item.device.Get()];
            const D3D12_COMMAND_QUEUE_DESC queue_desc{};
            auto copy = std::make_unique<Copy>();
            copy->resource = readback.target.resource;
            if ((!queue && FAILED(item.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)))) ||
                !copy->open(item.device.Get()) ||
                !replay_dump::record(item.device.Get(), copy->list.Get(), readback, item.bytes)) {
                ++failed;
                continue;
            }
            copy->staging = std::move(readback.buffer);
            copy->queue = queue;
            if (!submit(copy)) {
                // A timed-out queue may still be accessing the resource. Stop issuing copies.
                idle_ok = false;
                ++failed;
                continue;
            }
            for (const auto& slice : readback.slices) {
                const auto bytes = header_for(0, 0, slice).bytes;
                if (bytes > payload.size()) payload.resize(static_cast<std::size_t>(bytes));
            }
            void* mapped = nullptr;
            const D3D12_RANGE range{0, static_cast<SIZE_T>(item.bytes)};
            if (FAILED(copy->staging->Map(0, &range, &mapped))) {
                ++failed;
                continue;
            }
            const bool ok = write_gpu_resource(b, readback, item.bytes,
                static_cast<const unsigned char*>(mapped), payload, payload_bytes);
            const D3D12_RANGE none{0, 0};
            copy->staging->Unmap(0, &none);
            if (!ok) {
                ++failed;
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
}
}  // namespace replay_record
