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

bool mesh_xmem_loose_srv()
{
    static const bool enabled = x2w_local::env_on("X2W_MESH86_LOOSE_XMEM_SRV");
    return enabled;
}

bool mesh_xmem_tracking()
{
    return mesh_xmem_handoff() || mesh_xmem_indirect() || mesh_xmem_loose_srv();
}

// An ordinal selects one slot 86 call per process, keeping replay logs bounded.
UINT mesh_xmem_trace_ordinal()
{
    static const UINT selected = [] {
        char value[11]{};
        const DWORD size = GetEnvironmentVariableA("X2W_MESH86_WRITER_TRACE", value, sizeof(value));
        if (!size || size >= sizeof(value)) return 0u;
        u64 ordinal = 0;
        for (DWORD i = 0; i < size; ++i) {
            if (value[i] < '0' || value[i] > '9') return 0u;
            ordinal = ordinal * 10 + (value[i] - '0');
        }
        return ordinal <= UINT_MAX ? static_cast<UINT>(ordinal) : 0u;
    }();
    return selected;
}

u64 select_mesh_xmem_trace_call()
{
    static LONG64 calls;
    const UINT selected = mesh_xmem_trace_ordinal();
    if (!selected) return 0;
    const auto call = static_cast<u64>(InterlockedIncrement64(&calls));
    return call == selected ? call : 0;
}

const char* mesh_xmem_queue_type(ID3D12CommandQueue* queue)
{
    switch (queue->GetDesc().Type) {
    case D3D12_COMMAND_LIST_TYPE_DIRECT: return "direct";
    case D3D12_COMMAND_LIST_TYPE_COMPUTE: return "compute";
    case D3D12_COMMAND_LIST_TYPE_COPY: return "copy";
    default: return "other";
    }
}

struct MeshXMemLease : x2w_startup::XMemSectionBuffer {
    Wrapper* owner{};
    Wrapper* device{};
    u64 offset{}, address{}, bytes{};
    x2w_startup::ComPtr<ID3D12Fence> last_access;
    ID3D12CommandQueue* last_queue{};
    u64 trace_writer{}, trace_last_access{};
    bool untracked{};
};

struct MeshXMemRecording {
    struct Write {
        Wrapper* owner;
        u64 offset, bytes;
        bool leased;
        UINT slot;
        const char* source;
    };
    struct Read {
        std::shared_ptr<MeshXMemLease> lease;
        u64 address{};
    };
    std::vector<Write> writes;
    std::array<Read, 2> readers;
    std::array<std::shared_ptr<MeshXMemLease>, 2> writers;
    u64 trace_call{};
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
std::vector<std::weak_ptr<MeshXMemLease>> mesh_xmem_loose_views;
std::unordered_map<ID3D12CommandList*, MeshXMemRecording> mesh_xmem_recordings;
std::vector<MeshXMemFlight> mesh_xmem_flights;
bool mesh_xmem_tracking_failed;

struct MeshXMemTraceCopy {
    MeshXMemRecording::Write write;
    u64 submission;
    const char* queue;
};
std::array<MeshXMemTraceCopy, 1024> mesh_xmem_trace_copies;
u64 mesh_xmem_trace_copy_count, mesh_xmem_trace_submissions;

// Split submissions already hold mesh_xmem_mutex. A trace never changes which
// recordings are split or adds a completion fence to a normal copy.
u64 trace_mesh_xmem_submission(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists, bool locked)
{
    if (!mesh_xmem_trace_ordinal()) return 0;
    std::unique_lock lock(mesh_xmem_mutex, std::defer_lock);
    if (!locked) lock.lock();
    const u64 submission = ++mesh_xmem_trace_submissions;
    const char* queue_type = mesh_xmem_queue_type(queue);
    for (auto* list : lists) {
        const auto found = mesh_xmem_recordings.find(list);
        if (found == mesh_xmem_recordings.end()) continue;
        for (const auto& write : found->second.writes) {
            mesh_xmem_trace_copies[mesh_xmem_trace_copy_count++ % mesh_xmem_trace_copies.size()] =
                {write, submission, queue_type};
        }
    }
    return submission;
}

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
                                                   u64 offset, u64 address, u64 bytes)
{
    auto result = std::make_shared<MeshXMemLease>();
    result->owner = owner;
    result->device = device;
    result->address = address;
    result->offset = offset;
    result->bytes = bytes;
    const UINT version = owner ? x2w_xmem_section_lease_version : x2w_xmem_single_page_lease_version;
    const auto unused_window = [&](const X2WXMemSectionLease& lease) {
        // Do not transition two different desktop resources over the same
        // section bytes. The virtual addresses can be different XMem aliases.
        for (const auto& publication : mesh_xmem_publications)
            if (const auto other = publication.active.lock(); other && !mesh_xmem_windows_disjoint(lease, other->lease))
                return false;
        for (const auto& loose : mesh_xmem_loose_views)
            if (const auto other = loose.lock(); other && !mesh_xmem_windows_disjoint(lease, other->lease))
                return false;
        return true;
    };
    return result->open(device_of(device), version, address, bytes, unused_window) ? result : nullptr;
}

// Called with mesh_xmem_mutex held after the caller established there is no
// registered placement covering or overlapping the requested word.
std::shared_ptr<MeshXMemLease> loose_mesh_xmem_srv(Wrapper* device, u64 address, u64 bytes)
{
    mesh_xmem_loose_views.erase(std::remove_if(mesh_xmem_loose_views.begin(), mesh_xmem_loose_views.end(),
        [](const auto& view) { return view.expired(); }), mesh_xmem_loose_views.end());
    for (const auto& known : mesh_xmem_loose_views) {
        if (const auto view = known.lock(); view && view->device == device && view->address == address &&
            view->bytes == bytes && !view->untracked && view->matches(address))
            return view;
    }
    auto view = open_mesh_xmem_lease(device, nullptr, 0, address, bytes);
    if (view) mesh_xmem_loose_views.push_back(view);
    return view;
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
                          UINT slot, const char* source_kind, ID3D12Resource* source = nullptr, u64 source_offset = 0)
{
    if ((!mesh_xmem_handoff() && !mesh_xmem_indirect()) || !owner || !owner->own().title_memory) return;
    const auto* info = resource_info(owner);
    if (!info || info->placement != Placement::copied_buffer ||
        info->placement_device != list->own().list_device || info->console.Flags != D3D12_RESOURCE_FLAG_NONE)
        return;
    auto* desktop_list = static_cast<ID3D12GraphicsCommandList*>(list->own().real);
    const auto wait_start = x2w_phase::LockWaitScope::active ? x2w_phase::now_ticks() : 0;
    const std::lock_guard lock(mesh_xmem_mutex);
    if (x2w_phase::LockWaitScope::active) x2w_phase::LockWaitScope::note(x2w_phase::now_ticks() - wait_start);
    if (mesh_xmem_tracking_failed) return;
    try {
        auto& recorded = mesh_xmem_recordings[desktop_list];
        recorded.writes.push_back({owner, offset, bytes, false, slot, source_kind});
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
            lease = open_mesh_xmem_lease(list->own().list_device, owner, offset, address, bytes);
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

void trace_mesh_xmem_input(Wrapper* device, u64 call, size_t index, u64 address, u64 bytes, bool needs_lease,
                           bool anchor_only = false)
{
    const char* input = index == mesh_xmem_srv_index ? "root-srv" : "arguments";
    if (!address || address > UINT64_MAX - bytes) {
        log("x2w-mesh86-trace call=%llu input=%s range=unavailable", call, input);
        return;
    }
    const auto placed = placed_buffer_range(device, address, bytes);
    const auto& lease = mesh_xmem_publications[index].ready;
    const bool match = lease && !placed.reject && placed.owner == lease->owner &&
        lease->device == device && lease->bytes == bytes && lease->matches(address);
    log("x2w-mesh86-trace call=%llu input=%s range=%s bytes=%llu placement=%s lease=%s writer-submit=%llu",
        call, input, anchor_only ? "anchor-only" : "fixed-span", bytes,
        placed.reject ? "ambiguous" : !placed.owner ? "unresolved" : placed.alias ? "alias" : "copy",
        !needs_lease ? "not-checked" : match ? "matched" : lease ? "mismatch" : "absent",
        needs_lease && match ? lease->trace_writer : 0ull);
    if (placed.reject || !placed.owner) return;
    const u64 offset = address - reinterpret_cast<u64>(placed.owner->own().title_memory);
    const u64 retained = (std::min)(mesh_xmem_trace_copy_count, static_cast<u64>(mesh_xmem_trace_copies.size()));
    for (u64 i = 0; i < retained; ++i) {
        const auto& candidate = mesh_xmem_trace_copies[(mesh_xmem_trace_copy_count - i - 1) %
                                                      mesh_xmem_trace_copies.size()];
        const auto& write = candidate.write;
        if (write.owner != placed.owner || write.offset >= offset + bytes || offset >= write.offset + write.bytes)
            continue;
        const bool covers = write.offset <= offset && write.offset + write.bytes >= offset + bytes;
        log("x2w-mesh86-trace call=%llu input=%s copy-slot=%u source=%s queue=%s submit=%llu range=%s",
            call, input, write.slot, write.source, candidate.queue, candidate.submission,
            anchor_only ? "contains-anchor" : covers ? "covers-placement" : "overlaps-placement");
        return;
    }
    log("x2w-mesh86-trace call=%llu input=%s recent-copy=absent", call, input);
}

void trace_mesh_xmem_inputs(Wrapper* list, const Pass& pass, u64 call, void* argument, u64 argument_offset)
{
    const UINT proved_srv = pass.indirect_graphics.cached_srv_root;
    UINT srv = proved_srv;
    if (srv == UINT_MAX && pass.root && pass.root->size() <= std::size(pass.root_address)) {
        for (UINT i = 0; i < pass.root->size(); ++i) {
            if ((*pass.root)[i].type != D3D12_ROOT_PARAMETER_TYPE_SRV) continue;
            if (srv != UINT_MAX) {
                srv = UINT_MAX;
                break;
            }
            srv = i;
        }
    }
    const u64 srv_address = srv != UINT_MAX ? pass.root_address[srv] : 0;
    const Wrapper* buffer = as_wrapper(argument);
    const bool resource = buffer && buffer->own().kind == resource_kind;
    const u64 base = resource ? reinterpret_cast<u64>(buffer->own().title_memory) : 0;
    const u64 arg_address = base && argument_offset <= UINT64_MAX - base && !(argument_offset & 3)
        ? base + argument_offset : 0;
    const auto* info = resource ? resource_info(buffer) : nullptr;
    const bool leased_arguments = mesh_xmem_indirect() && info && info->placement == Placement::copied_buffer &&
        (!pass.indirect_graphics.cached_mesh || proved_srv != UINT_MAX);
    const bool anchor_only = proved_srv == UINT_MAX;
    auto* device = list->own().list_device;
    const std::lock_guard lock(mesh_xmem_mutex);
    trace_mesh_xmem_input(device, call, mesh_xmem_srv_index, srv_address, anchor_only ? 1 : sizeof(UINT),
                          !anchor_only, anchor_only);
    trace_mesh_xmem_input(device, call, mesh_xmem_argument_index, arg_address, sizeof(D3D12_DISPATCH_MESH_ARGUMENTS),
                          leased_arguments);
}

// Check all inputs to this draw before adding any recording pins or barriers.
const char* record_mesh_xmem_reads(Wrapper* list, MeshXMemRead* reads, size_t count, u64 trace_call)
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
        if (mesh_xmem_tracking_failed)
            return "mesh XMem read tracking is unavailable";
        const u64 alignment = index == mesh_xmem_srv_index ? D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT : sizeof(UINT);
        if (!read.address || read.address % alignment)
            return "mesh XMem read address is missing or misaligned";
        const auto placed = placed_buffer_range(list->own().list_device, read.address, read.bytes);
        if (placed.reject)
            return "mesh XMem read does not match the writer's leased page and byte range";
        auto lease = mesh_xmem_publications[index].ready;
        if (!placed.owner && index == mesh_xmem_srv_index && mesh_xmem_loose_srv()) {
            // The cached mesh PSO has already proved this raw root SRV reads
            // one DWORD at byte zero. The section may hold unrelated data.
            try {
                lease = loose_mesh_xmem_srv(list->own().list_device, read.address, read.bytes);
            } catch (const std::bad_alloc&) {
                return "mesh XMem loose root SRV allocation failed";
            }
            if (!lease) return "mesh XMem loose root SRV has no usable single-page lease";
        }
        if (!lease || lease->device != list->own().list_device || lease->bytes != read.bytes)
            return "mesh XMem read has no submitted writer for the required size";
        if (placed.owner != lease->owner || !lease->matches(read.address))
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
        if (trace_call) recorded.trace_call = trace_call;
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
                    !lease->owner ? "loose root bind" : index == mesh_xmem_srv_index ? "root bind" : "indirect bind",
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
    const auto trace_refuse = [&](const char* reason) {
        if (work.trace_call)
            log("x2w-mesh86-trace call=%llu stage=submit outcome=refused reason=%s", work.trace_call, reason);
    };
    std::array<std::shared_ptr<MeshXMemLease>, 2> accesses{};
    size_t access_count = 0;
    for (size_t i = 0; i < accesses.size(); ++i) {
        const auto& read = work.readers[i];
        const auto& lease = read.lease ? read.lease : work.writers[i];
        if (!lease) continue;
        const u64 address = read.lease ? read.address : lease->address;
        const auto placed = placed_buffer_range(lease->device, address, lease->bytes);
        if (mesh_xmem_tracking_failed || lease->untracked || placed.reject || placed.owner != lease->owner ||
            !lease->matches(address) || (read.lease && lease->owner && mesh_xmem_publications[i].ready != lease)) {
            log("x2w-render mesh XMem submit refused: writer or live page identity changed");
            trace_refuse(i == mesh_xmem_srv_index ? "root-srv-identity-changed" : "arguments-identity-changed");
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
            trace_refuse("no-completion-fence");
            return;
        }
        try {
            mesh_xmem_flights.reserve(mesh_xmem_flights.size() + access_count);
        } catch (const std::bad_alloc&) {
            log("x2w-render mesh XMem submit refused: retirement allocation failed");
            trace_refuse("retirement-allocation");
            return;
        }
        ID3D12Fence* waited = nullptr;
        for (size_t i = 0; i < accesses.size(); ++i) {
            const auto& lease = accesses[i];
            if (!lease) continue;
            const char* order;
            if (!lease->last_access) order = "no-fence";
            else if (lease->last_queue == queue) order = "same-queue";
            else if (lease->last_access.Get() == waited) order = "shared-wait";
            else if (work.readers[i].lease && mesh_xmem_skip_read_wait(i)) order = "test-omitted";
            else {
                if (FAILED(queue->Wait(lease->last_access.Get(), 1))) {
                    log("x2w-render mesh XMem submit refused: cross-queue wait failed");
                    trace_refuse("cross-queue-wait-failed");
                    return;
                }
                order = "wait";
                waited = lease->last_access.Get();
                static LONG reports;
                if (first_calls(reports, 16)) log("x2w-render mesh XMem GPU wait for lease %llu", lease->lease.id);
            }
            if (work.trace_call && work.readers[i].lease)
                log("x2w-mesh86-trace call=%llu input=%s order=%s writer-submit=%llu fence-submit=%llu",
                    work.trace_call, i == mesh_xmem_srv_index ? "root-srv" : "arguments", order,
                    lease->trace_writer, lease->trace_last_access);
        }
    }

    const u64 submission = execute();
    const bool signalled = !completion || SUCCEEDED(queue->Signal(completion.Get(), 1));
    if (work.trace_call)
        log("x2w-mesh86-trace call=%llu stage=submit outcome=forwarded submit=%llu queue=%s signal=%s",
            work.trace_call, submission, mesh_xmem_queue_type(queue), signalled ? "enqueued" : "failed");
    for (size_t i = 0; i < accesses.size(); ++i) {
        const auto& lease = accesses[i];
        if (!lease) continue;
        if (signalled) {
            mesh_xmem_flights.push_back({lease, completion});
            lease->last_access = completion;
            lease->last_queue = queue;
            lease->trace_last_access = submission;
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
            lease->trace_writer = submission;
            static LONG reports;
            if (first_calls(reports, 16))
                log("x2w-render mesh XMem writer: lease %llu section page +0x%llx byte +0x%llx, %llu bytes",
                    lease->lease.id, lease->lease.section_offset, lease->lease.first_byte, lease->bytes);
        }
    }
}
