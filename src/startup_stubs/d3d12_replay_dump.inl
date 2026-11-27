// Included by the shim after transition(). The player uses the same submission path as recording.
namespace replay_dump {
using x2w_startup::ComPtr;

struct Target {
    u64 id;
    ComPtr<ID3D12Resource> resource;
    DXGI_FORMAT view_format;
};
using Targets = std::vector<Target>;

constexpr GUID resource_id{0xbde8a785, 0xae70, 0x4609, {0x99, 0x82, 0x31, 0x51, 0x41, 0xe8, 0x09, 0x73}};

void note(Wrapper* list, ID3D12Resource* resource, DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN)
{
    if (!replay_dump_enabled() || !resource) return;
    u64 id = 0;
    UINT size = sizeof(id);
    if (FAILED(resource->GetPrivateData(resource_id, &size, &id)) || size != sizeof(id)) {
        log("x2w-replay dump: written resource has no creation id");
        return;
    }
    auto*& writes = list->own().replay_writes;
    if (!writes) writes = new Targets;
    const auto found = std::find_if(writes->begin(), writes->end(), [id](const Target& t) { return t.id == id; });
    if (found == writes->end()) writes->push_back({id, ComPtr<ID3D12Resource>(resource), format});
    else if (format != DXGI_FORMAT_UNKNOWN) found->view_format = format;
}

void note_copy(Wrapper* list, ID3D12Resource* destination)
{
    if (replay_dump_enabled() && (destination->GetDesc().Flags &
        (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)))
        note(list, destination);
}

struct Slice {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0;
};

struct Readback {
    Target target;
    D3D12_RESOURCE_DESC desc;
    std::vector<Slice> slices;
    ComPtr<ID3D12Resource> buffer;
    ComPtr<ID3D12Resource> resolved;
};

bool record(ID3D12Device* device, ID3D12GraphicsCommandList* list, Readback& readback)
{
    auto* source = readback.target.resource.Get();
    const auto& desc = readback.desc;
    auto copy_desc = desc;
    const bool multisampled = desc.SampleDesc.Count > 1;
    if (multisampled) {
        if (list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) return false;
        if (readback.target.view_format != DXGI_FORMAT_UNKNOWN) copy_desc.Format = readback.target.view_format;
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{copy_desc.Format};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
            !(support.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE)) return false;
        copy_desc.SampleDesc = {1, 0};
        copy_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        copy_desc.Alignment = 0;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &copy_desc,
                D3D12_RESOURCE_STATE_RESOLVE_DEST, nullptr, IID_PPV_ARGS(&readback.resolved)))) return false;
    }

    const bool buffer = desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
    UINT64 bytes = desc.Width;
    if (!buffer) {
        D3D12_FEATURE_DATA_FORMAT_INFO format{copy_desc.Format, 0};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &format, sizeof(format))) ||
            !format.PlaneCount) return false;
        const UINT count = UINT(desc.MipLevels) * format.PlaneCount *
            (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1 : desc.DepthOrArraySize);
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(count);
        std::vector<UINT> rows(count);
        std::vector<UINT64> row_bytes(count);
        device->GetCopyableFootprints(&copy_desc, 0, count, 0, footprints.data(), rows.data(), row_bytes.data(), &bytes);
        for (UINT i = 0; i < count; ++i) readback.slices.push_back({footprints[i], rows[i], row_bytes[i]});
    }
    if (!bytes || bytes == UINT64_MAX) return false;
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_RESOURCE_DESC staging{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &staging,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.buffer)))) return false;

    // The shim returns tracked resources to COMMON at Close. Execute this as a separate batch so
    // buffers promoted in the submitted lists also decay before the copy.
    if (multisampled) {
        transition(list, source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
        for (UINT i = 0; i < readback.slices.size(); ++i)
            list->ResolveSubresource(readback.resolved.Get(), i, source, i, copy_desc.Format);
        transition(list, source, D3D12_RESOURCE_STATE_RESOLVE_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        source = readback.resolved.Get();
        transition(list, source, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    } else {
        transition(list, source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    if (buffer) {
        list->CopyBufferRegion(readback.buffer.Get(), 0, source, 0, bytes);
    } else {
        for (UINT i = 0; i < readback.slices.size(); ++i) {
            D3D12_TEXTURE_COPY_LOCATION from{source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            from.SubresourceIndex = i;
            D3D12_TEXTURE_COPY_LOCATION to{readback.buffer.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            to.PlacedFootprint = readback.slices[i].footprint;
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        }
    }
    if (!multisampled)
        transition(list, source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    return true;
}

void emit(u64 submit, const Readback& readback)
{
    void* mapped = nullptr;
    if (FAILED(readback.buffer->Map(0, nullptr, &mapped))) {
        log("x2w-replay dump: cannot map submit %llu resource %llu", submit, readback.target.id);
        return;
    }
    u64 hash = 14695981039346656037ull;
    const auto* bytes = static_cast<const unsigned char*>(mapped);
    const auto add = [&](UINT64 offset, UINT64 length) {
        for (UINT64 i = 0; i < length; ++i) {
            hash ^= bytes[offset + i];
            hash *= 1099511628211ull;
        }
    };
    if (readback.slices.empty()) add(0, readback.desc.Width);
    for (const auto& slice : readback.slices) {
        const auto& fp = slice.footprint;
        for (UINT z = 0; z < fp.Footprint.Depth; ++z)
            for (UINT y = 0; y < slice.rows; ++y)
                add(fp.Offset + (UINT64(z) * slice.rows + y) * fp.Footprint.RowPitch, slice.row_bytes);
    }
    const D3D12_RANGE written{0, 0};
    readback.buffer->Unmap(0, &written);
    log("x2w-replay dump submit %llu resource %llu %llux%u fmt %u hash %016llx",
        submit, readback.target.id, readback.desc.Width, readback.desc.Height,
        static_cast<unsigned>(readback.desc.Format), hash);
}

struct Batch {
    u64 number;
    std::vector<Readback> readbacks;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
};
using Batches = std::vector<std::unique_ptr<Batch>>;

void keep_in_flight(Batches& batches)
{
    // There is no safe completion point to release any of these GPU references.
    for (auto& batch : batches) batch.release();
}

std::mutex queue_mutex;

std::unordered_map<ID3D12CommandQueue*, Batches>& pending()
{
    struct Pending {
        std::unordered_map<ID3D12CommandQueue*, Batches> queues;
        ~Pending()
        {
            for (auto& entry : queues) keep_in_flight(entry.second);
        }
    };
    static Pending pending_batches;
    return pending_batches.queues;
}

// queue_mutex is held from the guest Execute through the readback Execute, so a guest
// Signal on another thread cannot land between them.
void submit(ID3D12CommandQueue* queue, const std::vector<Wrapper*>& lists)
{
    if (!replay_dump_enabled() || (replay_record::armed && !replay_record::window_open)) return;
    static u64 ordinal = 0;
    const u64 number = ++ordinal;
    std::map<u64, Target> targets;
    for (auto* list : lists)
        if (const auto* writes = list->own().replay_writes)
            for (const auto& target : *writes) {
                auto [it, inserted] = targets.emplace(target.id, target);
                if (!inserted && target.view_format != DXGI_FORMAT_UNKNOWN) it->second.view_format = target.view_format;
            }
    if (targets.empty()) return;

    auto batch = std::make_unique<Batch>();
    batch->number = number;
    ComPtr<ID3D12Device> device;
    const auto type = queue->GetDesc().Type;
    if (FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) ||
        FAILED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&batch->allocator))) ||
        FAILED(device->CreateCommandList(0, type, batch->allocator.Get(), nullptr, IID_PPV_ARGS(&batch->list)))) {
        log("x2w-replay dump: cannot create readback context for submit %llu", number);
        return;
    }
    for (const auto& entry : targets) {
        Readback readback{entry.second, entry.second.resource->GetDesc(), {}, {}, {}};
        if (record(device.Get(), batch->list.Get(), readback)) batch->readbacks.push_back(std::move(readback));
        else log("x2w-replay dump: cannot copy submit %llu resource %llu", number, entry.first);
    }
    if (batch->readbacks.empty()) return;
    if (FAILED(batch->list->Close())) {
        log("x2w-replay dump: cannot close readback list for submit %llu", number);
        return;
    }
    ID3D12CommandList* const copies[]{batch->list.Get()};
    pending()[queue].push_back(std::move(batch));
    queue->ExecuteCommandLists(1, copies);
}

HRESULT signal(ID3D12CommandQueue* queue, ID3D12Fence* guest_fence, UINT64 value)
{
    if (!replay_dump_enabled()) return queue->Signal(guest_fence, value);

    const bool emit_rows = !replay_record::armed || replay_record::window_open;
    Batches batches;
    ComPtr<ID3D12Fence> completed;
    HRESULT hr;
    {
        const std::lock_guard lock(queue_mutex);
        hr = queue->Signal(guest_fence, value);
        const auto next = pending().find(queue);
        if (FAILED(hr) || next == pending().end()) return hr;

        // A guest fence can be shared or rewound. A private value behind this Signal proves
        // that this queue reached it, even when the guest value was already complete.
        ComPtr<ID3D12Device> device;
        if (FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completed))) ||
            FAILED(queue->Signal(completed.Get(), 1))) {
            log("x2w-replay dump: cannot signal readbacks through submit %llu", next->second.back()->number);
            return hr;
        }
        batches = std::move(next->second);
        pending().erase(next);
    }

    const auto deadline = GetTickCount64() + 10000;
    UINT64 progress = 0;
    while (!(progress = completed->GetCompletedValue()) && GetTickCount64() < deadline) Sleep(1);
    if (progress == 1) {
        if (!emit_rows) return hr;
        for (const auto& batch : batches)
            for (const auto& readback : batch->readbacks) emit(batch->number, readback);
    } else {
        log("x2w-replay dump: readback signal did not complete through submit %llu", batches.back()->number);
        if (progress != UINT64_MAX) {
            keep_in_flight(batches);
            completed.Detach();
        }
    }
    return hr;
}
} // namespace replay_dump
