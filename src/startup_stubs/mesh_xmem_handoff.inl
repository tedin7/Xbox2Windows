// Included after the command-list resource state tracker. Only slot 15 can
// publish this experimental lease. Other consumers of the placed resource
// keep using its normal desktop backing.
bool mesh_xmem_handoff()
{
    static const bool enabled = x2w_local::env_on("X2W_MESH86_XMEM_SRV");
    return enabled;
}

bool mesh_xmem_indirect()
{
    static const bool enabled = x2w_local::env_on("X2W_MESH_INDIRECT_XMEM");
    return enabled;
}

bool mesh_xmem_tracking()
{
    return mesh_xmem_handoff() || mesh_xmem_indirect();
}

struct MeshXMemLease {
    using Release = BOOL (WINAPI*)(UINT, ULONG64);
    X2WXMemSectionLease lease{};
    Release release{};
    X2WXMemMatchSection match{};
    Wrapper* owner{};
    Wrapper* device{};
    u64 offset{}, address{}, bytes{};
    void* view{};
    x2w_startup::ComPtr<ID3D12Heap> heap;
    x2w_startup::ComPtr<ID3D12Resource> buffer;
    x2w_startup::ComPtr<ID3D12Fence> last_access;
    ID3D12CommandQueue* last_queue{};
    bool untracked{};

    ~MeshXMemLease()
    {
        buffer.Reset();
        heap.Reset();
        if (lease.section) CloseHandle(lease.section);
        // A view that could not be removed still has access to the slots.
        if ((!view || UnmapViewOfFile(view)) && lease.id)
            release(x2w_xmem_section_lease_version, lease.id);
    }

    bool matches(u64 candidate) const
    {
        return match(x2w_xmem_section_lease_version, lease.id, lease.first_byte, candidate, bytes);
    }
};

struct MeshXMemRecording {
    struct Write { Wrapper* owner; u64 offset, bytes; bool leased; };
    struct Read {
        std::shared_ptr<MeshXMemLease> lease;
        u64 address{};
    };
    std::vector<Write> writes;
    std::array<Read, 2> readers;
    std::array<std::shared_ptr<MeshXMemLease>, 2> writers;
};
struct MeshXMemFlight {
    std::shared_ptr<MeshXMemLease> lease;
    x2w_startup::ComPtr<ID3D12Fence> completion;
};

struct MeshXMemPublication {
    std::weak_ptr<MeshXMemLease> active;
    std::shared_ptr<MeshXMemLease> ready;
};

std::mutex mesh_xmem_mutex;
// The root word and the dispatch record each have one active lease.
constexpr size_t mesh_xmem_srv_index = 0, mesh_xmem_argument_index = 1;
std::array<MeshXMemPublication, 2> mesh_xmem_publications;
std::unordered_map<ID3D12CommandList*, MeshXMemRecording> mesh_xmem_recordings;
std::vector<MeshXMemFlight> mesh_xmem_flights;
bool mesh_xmem_tracking_failed;

// All functions that mutate publication/flight state hold mesh_xmem_mutex.
void reap_mesh_xmem_flights()
{
    mesh_xmem_flights.erase(std::remove_if(mesh_xmem_flights.begin(), mesh_xmem_flights.end(),
        [](const MeshXMemFlight& flight) {
            return flight.completion && flight.completion->GetCompletedValue() == 1;
        }), mesh_xmem_flights.end());
}

void CALLBACK mesh_xmem_completed(PTP_CALLBACK_INSTANCE, void* context, PTP_WAIT wait, TP_WAIT_RESULT)
{
    {
        const std::lock_guard lock(mesh_xmem_mutex);
        reap_mesh_xmem_flights();
    }
    CloseThreadpoolWait(wait);
    CloseHandle(static_cast<HANDLE>(context));
}

void watch_mesh_xmem_completion(ID3D12Fence* fence)
{
    // The callback may outlive the last application reference to the shim.
    static const bool module_pinned = [] {
        HMODULE module{};
        return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                  reinterpret_cast<LPCWSTR>(&mesh_xmem_completed), &module) != FALSE;
    }();
    if (!module_pinned) return;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return;
    const auto wait = CreateThreadpoolWait(mesh_xmem_completed, event, nullptr);
    if (!wait || FAILED(fence->SetEventOnCompletion(1, event))) {
        if (wait) CloseThreadpoolWait(wait);
        CloseHandle(event);
        // Later wrapper activity can still reap the recorded fence.
        return;
    }
    SetThreadpoolWait(wait, event, nullptr);
}

void drop_mesh_lease_owner(Wrapper* w)
{
    if (!mesh_xmem_tracking()) return;
    const std::lock_guard lock(mesh_xmem_mutex);
    for (auto& publication : mesh_xmem_publications)
        if (publication.ready && publication.ready->owner == w) publication.ready.reset();
    reap_mesh_xmem_flights();
}

void discard_mesh_lease_recording(ID3D12CommandList* list)
{
    if (!mesh_xmem_tracking()) return;
    const std::lock_guard lock(mesh_xmem_mutex);
    mesh_xmem_recordings.erase(list);
    reap_mesh_xmem_flights();
}

bool mesh_xmem_windows_disjoint(const X2WXMemSectionLease& one, const X2WXMemSectionLease& other)
{
    if (one.section_offset + one.window_bytes <= other.section_offset ||
        other.section_offset + other.window_bytes <= one.section_offset) return true;
    using Compare = BOOL (WINAPI*)(HANDLE, HANDLE);
    static const auto compare = reinterpret_cast<Compare>(
        GetProcAddress(GetModuleHandleW(L"kernelbase.dll"), "CompareObjectHandles"));
    return compare && !compare(one.section, other.section);
}

std::shared_ptr<MeshXMemLease> open_mesh_xmem_lease(Wrapper* device, Wrapper* owner,
                                                   u64 offset, u64 address, u64 bytes, const MeshXMemLease* other)
{
    using Pin = BOOL (WINAPI*)(UINT, ULONG_PTR, SIZE_T, X2WXMemSectionLease*, SIZE_T);
    HMODULE xmem = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, L"xmem.dll", &xmem)) return {};
    const auto pin = reinterpret_cast<Pin>(GetProcAddress(xmem, "x2w_xmem_lease_section"));
    const auto release = reinterpret_cast<MeshXMemLease::Release>(GetProcAddress(xmem, "x2w_xmem_release_section"));
    const auto match = reinterpret_cast<X2WXMemMatchSection>(GetProcAddress(xmem, "x2w_xmem_match_section"));
    if (!pin || !release || !match) return {};
    D3D12_FEATURE_DATA_EXISTING_HEAPS feature{};
    x2w_startup::ComPtr<ID3D12Device3> desktop;
    if (FAILED(device_of(device)->QueryInterface(IID_PPV_ARGS(&desktop)))) return {};
    const auto check_heaps = [&] {
#if defined(X2W_MESH_XMEM_AGILITY_620) && defined(__ID3D12Device16_INTERFACE_DEFINED__)
        x2w_startup::ComPtr<ID3D12Device16> versioned;
        if (SUCCEEDED(desktop.As(&versioned)))
            return versioned->CheckFeatureSupport1(D3D12_FEATURE_EXISTING_HEAPS, &feature, sizeof(feature), 620);
#endif
        return desktop->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &feature, sizeof(feature));
    };
    if (FAILED(check_heaps()) || !feature.Supported) return {};
    auto result = std::make_shared<MeshXMemLease>();
    result->owner = owner;
    result->device = device;
    result->address = address;
    result->offset = offset;
    result->bytes = bytes;
    result->release = release;
    result->match = match;
    auto& lease = result->lease;
    if (!pin(x2w_xmem_section_lease_version, address, bytes, &lease, sizeof(lease))) return {};
    // The desktop buffers transition independently, so their state changes
    // must never cover the same physical section bytes. Compare the pinned
    // sections directly, since the guest addresses can be remapped here.
    if (other && !mesh_xmem_windows_disjoint(lease, other->lease)) return {};
    result->view = MapViewOfFile(lease.section, FILE_MAP_WRITE, static_cast<DWORD>(lease.section_offset >> 32),
                                 static_cast<DWORD>(lease.section_offset), static_cast<SIZE_T>(lease.window_bytes));
    if (!result->view || FAILED(desktop->OpenExistingHeapFromAddress(result->view, IID_PPV_ARGS(&result->heap))))
        return {};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = lease.window_bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    if (FAILED(desktop->CreatePlacedResource(result->heap.Get(), 0, &desc, D3D12_RESOURCE_STATE_COMMON,
                                             nullptr, IID_PPV_ARGS(&result->buffer)))) return {};
    return result->matches(address) ? result : nullptr;
}

size_t mesh_xmem_index(u64 bytes)
{
    return bytes == sizeof(UINT) ? mesh_xmem_srv_index : mesh_xmem_argument_index;
}

bool mesh_xmem_write_overlaps(const MeshXMemRecording::Write& write, const MeshXMemLease& lease)
{
    return write.owner == lease.owner && write.offset < lease.offset + lease.bytes &&
           lease.offset < write.offset + write.bytes;
}

// Track other writes to the copied placement too. An overlapping write that
// did not update the XMem view invalidates its publication at submission.
void note_mesh_xmem_write(Wrapper* list, Wrapper* owner, u64 offset, u64 bytes,
                          ID3D12Resource* source = nullptr, u64 source_offset = 0)
{
    if (!mesh_xmem_tracking() || !owner || !owner->own().title_memory) return;
    const auto* info = resource_info(owner);
    if (!info || info->placement != Placement::copied_buffer ||
        info->placement_device != list->own().list_device || info->console.Flags != D3D12_RESOURCE_FLAG_NONE)
        return;
    auto* desktop_list = static_cast<ID3D12GraphicsCommandList*>(list->own().real);
    const std::lock_guard lock(mesh_xmem_mutex);
    if (mesh_xmem_tracking_failed) return;
    try {
        auto& recorded = mesh_xmem_recordings[desktop_list];
        recorded.writes.push_back({owner, offset, bytes, false});
        const u64 base = reinterpret_cast<u64>(owner->own().title_memory);
        const u64 alignment = bytes == sizeof(UINT) ? D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT : sizeof(UINT);
        if (!source || offset > UINT64_MAX - base || recorded.readers[mesh_xmem_srv_index].lease ||
            recorded.readers[mesh_xmem_argument_index].lease || (base + offset) % alignment) return;
        x2w_startup::ComPtr<ID3D12Device> source_device;
        if (FAILED(source->GetDevice(IID_PPV_ARGS(&source_device))) ||
            source_device.Get() != device_of(list->own().list_device)) return;
        const u64 address = base + offset;
        const auto placed = placed_buffer_range(list->own().list_device, address, bytes);
        if (placed.reject || placed.owner != owner) return;
        const auto index = mesh_xmem_index(bytes);
        auto& publication = mesh_xmem_publications[index];
        auto lease = publication.active.lock();
        if (!lease) {
            const auto other_index = index == mesh_xmem_srv_index ? mesh_xmem_argument_index : mesh_xmem_srv_index;
            const auto other = mesh_xmem_publications[other_index].active.lock();
            lease = open_mesh_xmem_lease(list->own().list_device, owner, offset, address, bytes, other.get());
            if (!lease) return;
            publication.active = lease;
        }
        if (lease->untracked || lease->owner != owner || lease->offset != offset || lease->bytes != bytes ||
            !lease->matches(address)) return;
        auto& states = pass_of(list).states;
        // Reserve before recording a barrier so a failed allocation cannot
        // leave an untracked transition on the desktop list.
        states.reserve(states.size() + 1);
        require_state(list, lease->buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        desktop_list->CopyBufferRegion(lease->buffer.Get(), lease->lease.first_byte, source, source_offset, bytes);
        recorded.writers[index] = lease;
        recorded.writes.back().leased = true;
    } catch (const std::bad_alloc&) {
        mesh_xmem_tracking_failed = true;
        for (auto& publication : mesh_xmem_publications) publication.ready.reset();
        log("x2w-render mesh XMem handoff refused: recording allocation failed");
    }
}

struct MeshXMemRead {
    u64 address;
    u64 bytes;
    D3D12_RESOURCE_STATES state;
    ID3D12Resource* resource{};
    u64 offset{};
};

// Check all inputs to this draw before adding any recording pins or barriers.
const char* record_mesh_xmem_reads(Wrapper* list, MeshXMemRead* reads, size_t count)
{
    if (!count) return nullptr;
    const std::lock_guard lock(mesh_xmem_mutex);
    auto* desktop_list = static_cast<ID3D12GraphicsCommandList*>(list->own().real);
    const auto found = mesh_xmem_recordings.find(desktop_list);
    const auto* existing = found == mesh_xmem_recordings.end() ? nullptr : &found->second;
    std::array<MeshXMemRecording::Read, 2> verified{};
    for (size_t i = 0; i < count; ++i) {
        const auto& read = reads[i];
        if (read.bytes != sizeof(UINT) && read.bytes != sizeof(D3D12_DISPATCH_MESH_ARGUMENTS))
            return "mesh XMem read has an unsupported size";
        const auto index = mesh_xmem_index(read.bytes);
        const auto& lease = mesh_xmem_publications[index].ready;
        if (mesh_xmem_tracking_failed || !lease || lease->device != list->own().list_device || lease->bytes != read.bytes)
            return "mesh XMem read has no submitted writer for the required size";
        const u64 alignment = index == mesh_xmem_srv_index ? D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT : sizeof(UINT);
        if (!read.address || read.address % alignment)
            return "mesh XMem read address is missing or misaligned";
        const auto placed = placed_buffer_range(list->own().list_device, read.address, read.bytes);
        if (!lease->matches(read.address) || placed.reject || placed.owner != lease->owner)
            return "mesh XMem read does not match the writer's leased page and byte range";
        if (verified[index].lease)
            return "mesh XMem draw repeats the same input kind";
        if (existing) {
            const auto& previous = existing->readers[index];
            if (existing->writers[mesh_xmem_srv_index] || existing->writers[mesh_xmem_argument_index] ||
                (previous.lease && (previous.lease != lease || previous.address != read.address)))
                return "mesh XMem read requires a separate recording from its writer";
            for (const auto& write : existing->writes)
                if (mesh_xmem_write_overlaps(write, *lease))
                    return "mesh XMem read overlaps a write in the same recording";
        }
        verified[index] = {lease, read.address};
    }
    try {
        auto& states = pass_of(list).states;
        states.reserve(states.size() + count);
        auto& recorded = mesh_xmem_recordings[desktop_list];
        for (size_t i = 0; i < count; ++i) {
            auto& read = reads[i];
            const auto index = mesh_xmem_index(read.bytes);
            const auto& lease = verified[index].lease;
            require_state(list, lease->buffer.Get(), read.state);
            recorded.readers[index] = verified[index];
            read.resource = lease->buffer.Get();
            read.offset = lease->lease.first_byte;
            static LONG reports;
            if (first_calls(reports, 16))
                log("x2w-render mesh XMem %s: lease %llu section page +0x%llx byte +0x%llx, %llu bytes",
                    index == mesh_xmem_srv_index ? "root bind" : "indirect bind",
                    lease->lease.id, lease->lease.section_offset, lease->lease.first_byte, read.bytes);
        }
    } catch (const std::bad_alloc&) {
        return "mesh XMem read recording allocation failed";
    }
    return nullptr;
}

// Caller holds mesh_xmem_mutex. Other copied placements can keep their
// original batching when they cannot change either publication.
bool mesh_xmem_needs_submission(ID3D12CommandList* list)
{
    const auto found = mesh_xmem_recordings.find(list);
    if (found == mesh_xmem_recordings.end()) return false;
    const auto& work = found->second;
    for (size_t i = 0; i < mesh_xmem_publications.size(); ++i) {
        if (work.readers[i].lease || work.writers[i]) return true;
        const auto active = mesh_xmem_publications[i].active.lock();
        if (active && std::any_of(work.writes.begin(), work.writes.end(),
            [&](const MeshXMemRecording::Write& write) { return write.owner == active->owner; }))
            return true;
    }
    return false;
}

bool mesh_xmem_skip_read_wait(size_t index)
{
    return x2w_local::env_on("X2W_TEST_MESH_XMEM_SKIP_WAIT") ||
        x2w_local::env_on(index == mesh_xmem_srv_index ? "X2W_TEST_MESH_XMEM_SKIP_SRV_WAIT" :
                                                         "X2W_TEST_MESH_XMEM_SKIP_ARGUMENT_WAIT");
}

// Draw and allocator locks serialize queue submissions. Validate every lease
// before queuing a wait. Keep their references through each submission fence.
template<class Execute>
void execute_mesh_xmem_recording(ID3D12CommandQueue* queue, ID3D12CommandList* list, Execute execute)
{
    const std::lock_guard lock(mesh_xmem_mutex);
    reap_mesh_xmem_flights();
    const auto record = mesh_xmem_recordings.find(list);
    if (record == mesh_xmem_recordings.end()) {
        execute();
        return;
    }
    const auto& work = record->second;
    std::array<std::shared_ptr<MeshXMemLease>, 2> accesses{};
    size_t access_count = 0;
    for (size_t i = 0; i < accesses.size(); ++i) {
        const auto& read = work.readers[i];
        const auto& lease = read.lease ? read.lease : work.writers[i];
        if (!lease) continue;
        const u64 address = read.lease ? read.address : lease->address;
        const auto placed = placed_buffer_range(lease->device, address, lease->bytes);
        if (mesh_xmem_tracking_failed || lease->untracked || placed.reject || placed.owner != lease->owner ||
            !lease->matches(address) || (read.lease && mesh_xmem_publications[i].ready != lease)) {
            log("x2w-render mesh XMem submit refused: writer or live page identity changed");
            return;
        }
        accesses[i] = lease;
        ++access_count;
    }

    x2w_startup::ComPtr<ID3D12Fence> completion;
    if (access_count) {
        x2w_startup::ComPtr<ID3D12Device> device;
        if (submit_fence_fails() || FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) ||
            std::any_of(accesses.begin(), accesses.end(), [&](const auto& lease) {
                return lease && device.Get() != device_of(lease->device);
            }) || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion)))) {
            log("x2w-render mesh XMem submit refused: no completion fence on this device");
            return;
        }
        try {
            mesh_xmem_flights.reserve(mesh_xmem_flights.size() + access_count);
        } catch (const std::bad_alloc&) {
            log("x2w-render mesh XMem submit refused: retirement allocation failed");
            return;
        }
        ID3D12Fence* waited = nullptr;
        for (size_t i = 0; i < accesses.size(); ++i) {
            const auto& lease = accesses[i];
            if (!lease || !lease->last_access || lease->last_queue == queue ||
                lease->last_access.Get() == waited || (work.readers[i].lease && mesh_xmem_skip_read_wait(i)))
                continue;
            if (FAILED(queue->Wait(lease->last_access.Get(), 1))) {
                log("x2w-render mesh XMem submit refused: cross-queue wait failed");
                return;
            }
            waited = lease->last_access.Get();
            static LONG reports;
            if (first_calls(reports, 16)) log("x2w-render mesh XMem GPU wait for lease %llu", lease->lease.id);
        }
    }

    execute();
    const bool signalled = !completion || SUCCEEDED(queue->Signal(completion.Get(), 1));
    for (size_t i = 0; i < accesses.size(); ++i) {
        const auto& lease = accesses[i];
        if (!lease) continue;
        if (signalled) {
            mesh_xmem_flights.push_back({lease, completion});
            lease->last_access = completion;
            lease->last_queue = queue;
        } else {
            lease->untracked = true;
            mesh_xmem_publications[i].ready.reset();
            mesh_xmem_flights.push_back({lease, {}});
        }
    }
    if (!signalled)
        log("x2w-render mesh XMem completion signal failed; lease remains pinned");
    else if (completion)
        watch_mesh_xmem_completion(completion.Get());

    for (const auto& write : work.writes) {
        // A normal copy can invalidate either publication. Preserve the copy
        // order so a later lease write on this list can publish again.
        for (auto& publication : mesh_xmem_publications)
            if (publication.ready && mesh_xmem_write_overlaps(write, *publication.ready))
                publication.ready.reset();
        if (write.leased && signalled && InterlockedCompareExchange(&write.owner->own().refs, 0, 0)) {
            const auto index = mesh_xmem_index(write.bytes);
            const auto& lease = work.writers[index];
            mesh_xmem_publications[index].ready = lease;
            static LONG reports;
            if (first_calls(reports, 16))
                log("x2w-render mesh XMem writer: lease %llu section page +0x%llx byte +0x%llx, %llu bytes",
                    lease->lease.id, lease->lease.section_offset, lease->lease.first_byte, lease->bytes);
        }
    }
}
