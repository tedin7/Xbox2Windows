// Included from d3d12_xs_device.cpp after the command-list state helpers.
//
// Whole-resource copies on a recording list (render mode): CopyBufferRegion (list slot 15), CopyResource
// (17) and ResolveSubresource (19). The title's resources are mapped to desktop ones, the pair is checked
// by src/startup_stubs/list_copy.h, and a pair the desktop would reject is counted and skipped instead of
// forwarded. Desktop resources rest in COMMON between lists, so each is moved to the copy or resolve state
// around the call and back at Close; upload and readback buffers keep the state their heap requires.
// CopyTextureRegion (16) and CopyTiles (18) carry footprints and addresses and are not handled here.

LONG list_buffer_copies, list_resource_copies, list_resolves, list_copies_refused;

// One copy argument: the desktop resource, and its wrapper when the title passed one.
struct CopyEndpoint {
    ID3D12Resource* resource = nullptr;
    Wrapper* wrapper = nullptr;
};

// The desktop resource behind a copy argument; empty for a null value, or a wrapper that is not a resource.
// A value that is no wrapper is taken as a desktop resource, as the view slots do.
CopyEndpoint copy_endpoint(u64 value, bool materialize = true)
{
    if (!value) return {};
    if (Wrapper* w = as_wrapper(reinterpret_cast<const void*>(value))) {
        if (w->own().kind != resource_kind) return {};
        if (materialize && w->own().lazy_slot133_texture && !w->own().real)
            materialize_lazy_texture(w);
        return {static_cast<ID3D12Resource*>(w->own().real), w};
    }
    return {x2w_local::readable(reinterpret_cast<const void*>(value), sizeof(void*)) ? reinterpret_cast<ID3D12Resource*>(value) : nullptr,
            nullptr};
}

void refuse_copy(const char* what, x2w_startup::CopyRefusal reason)
{
    InterlockedIncrement(&list_copies_refused);
    static LONG logged;
    if (first_calls(logged, 6)) log("x2w-d3d12 %s skipped: %s", what, x2w_startup::copy_refusal_name(reason));
}

// Upload and readback buffers keep the one state their heap allows, so a copy may use them in one direction only:
// an upload buffer as a source, a readback buffer as a destination. True for a heap that rests in COMMON.
bool fixed_heap(ID3D12Resource* resource, D3D12_HEAP_TYPE& type)
{
    D3D12_HEAP_PROPERTIES heap{};
    D3D12_HEAP_FLAGS flags{};
    type = D3D12_HEAP_TYPE_DEFAULT;
    if (FAILED(resource->GetHeapProperties(&heap, &flags))) return false;
    type = heap.Type;
    return heap.Type == D3D12_HEAP_TYPE_UPLOAD || heap.Type == D3D12_HEAP_TYPE_READBACK;
}

// The first refusal for a pair of endpoints that the descriptions did not catch: a heap used in the wrong
// direction, or a placed buffer that does not start in COMMON (the state tracker starts every resource there).
x2w_startup::CopyRefusal check_endpoint_roles(const CopyEndpoint& to, const CopyEndpoint& from)
{
    D3D12_HEAP_TYPE type;
    if (fixed_heap(to.resource, type) && type == D3D12_HEAP_TYPE_UPLOAD) return x2w_startup::CopyRefusal::heap_direction;
    if (fixed_heap(from.resource, type) && type == D3D12_HEAP_TYPE_READBACK) return x2w_startup::CopyRefusal::heap_direction;
    for (const CopyEndpoint& endpoint : {to, from}) {
        if (!endpoint.wrapper) continue;
        const auto* info = resource_info(endpoint.wrapper);
        if (info && info->initial_state != D3D12_RESOURCE_STATE_COMMON) return x2w_startup::CopyRefusal::initial_state;
    }
    return x2w_startup::CopyRefusal::none;
}

// The state a copy needs from `resource`: an upload buffer stays readable and a readback buffer stays a
// copy destination, because desktop allows no other state for them.
void require_copy_state(Wrapper* self, ID3D12Resource* resource, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_TYPE type;
    if (fixed_heap(resource, type)) return;
    require_state(self, resource, state);
}

// A resource the title maps at its own memory has a desktop resource that map and unmap never touch. Before
// that resource is copied from, it must be current: a texture takes the draw path's upload, and a placed buffer
// aliasing the memory already is. False when the desktop copy cannot be trusted: a texture the upload could not
// stage, or a title-backed buffer in a default or upload heap. An upload buffer is refused because refreshing it
// would rewrite one backing that an earlier submission may still be reading.
bool sync_title_source(Wrapper* self, const CopyEndpoint& source)
{
    if (!source.wrapper || !source.wrapper->own().title_memory) return true;
    const D3D12_RESOURCE_DESC desc = source.resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
        // A target or UAV holds what the GPU wrote; its title memory is never uploaded.
        if (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return true;
        if (list_wrote(self, source.resource)) return true;
        AcquireSRWLockExclusive(&draw_lock);
        const auto written = texture_uploads().find(source.resource);
        bool ok = false;
        if (written != texture_uploads().end() && written->second.gpu_written) {
            ok = true;
        } else if (upload_texture(self, source.wrapper, source.resource)) {
            const auto found = texture_uploads().find(source.resource);
            ok = found != texture_uploads().end() && found->second.staged;
        }
        ReleaseSRWLockExclusive(&draw_lock);
        return ok;
    }
    const auto* info = resource_info(source.wrapper);
    return info && info->placement == Placement::aliased_buffer;
}

// Copy and resolve destinations a recording wrote but no ExecuteCommandLists has submitted. A discarded
// recording (Reset or a failed Close) must leave the destination's title memory uploadable, so the state a
// draw consults changes only on submission. A closed list stays submittable until its next Reset, so
// commit leaves the records in place and only Reset or a failed Close drops them.
SRWLOCK pending_writes_lock = SRWLOCK_INIT;
std::unordered_map<Wrapper*, std::vector<ID3D12Resource*>>& pending_gpu_writes()
{
    static std::unordered_map<Wrapper*, std::vector<ID3D12Resource*>> m;
    return m;
}
// X2W_D3D12_LINEAR_BUFFERS: the copied linear buffers a recording wrote (list slots 15 and 17), under the same
// rule. A copied buffer stops being refreshed from its title memory only once a recording that writes it is
// submitted; a recording that is discarded leaves it refreshed (linear_buffer_backing.h).
std::unordered_map<Wrapper*, std::vector<ID3D12Resource*>>& pending_linear_writes()
{
    static std::unordered_map<Wrapper*, std::vector<ID3D12Resource*>> m;
    return m;
}

bool list_wrote(Wrapper* list, ID3D12Resource* resource)
{
    AcquireSRWLockShared(&pending_writes_lock);
    bool found = false;
    if (const auto it = pending_gpu_writes().find(list); it != pending_gpu_writes().end())
        found = std::find(it->second.begin(), it->second.end(), resource) != it->second.end();
    ReleaseSRWLockShared(&pending_writes_lock);
    return found;
}

void discard_list_gpu_writes(Wrapper* self)
{
    AcquireSRWLockExclusive(&pending_writes_lock);
    pending_gpu_writes().erase(self);
    pending_linear_writes().erase(self);
    ReleaseSRWLockExclusive(&pending_writes_lock);
}

void commit_list_gpu_writes(Wrapper* self)
{
    std::vector<ID3D12Resource*> linear;
    AcquireSRWLockShared(&pending_writes_lock);
    if (const auto it = pending_gpu_writes().find(self); it != pending_gpu_writes().end()) {
        for (ID3D12Resource* resource : it->second) {
            TextureUpload& state = texture_uploads()[resource];
            state.gpu_written = true;
            state.done = true;
        }
    }
    if (const auto it = pending_linear_writes().find(self); it != pending_linear_writes().end()) linear = it->second;
    ReleaseSRWLockShared(&pending_writes_lock);
    // After the recording's ExecuteCommandLists, outside the lock (the backing takes its own).
    for (ID3D12Resource* resource : linear) linear_buffers().note_gpu_write(resource);
}

// A copy or resolve destination will hold what the GPU wrote once `self` is submitted. A title-backed texture
// would otherwise be uploaded over it from its stale memory the first time a draw samples it.
void mark_gpu_written(Wrapper* self, const CopyEndpoint& destination)
{
    if (!destination.wrapper || !destination.wrapper->own().title_memory) return;
    AcquireSRWLockExclusive(&pending_writes_lock);
    auto& writes = pending_gpu_writes()[self];
    if (std::find(writes.begin(), writes.end(), destination.resource) == writes.end()) writes.push_back(destination.resource);
    ReleaseSRWLockExclusive(&pending_writes_lock);
}

// X2W_D3D12_LINEAR_BUFFERS: a copied linear buffer that a recorded copy writes keeps the GPU's bytes once
// `self` is submitted, and is no longer refreshed from its title memory from then on (linear_buffer_backing.h).
// Until then the write is pending on the recording: Reset or a failed Close drops it with the texture writes.
void note_linear_gpu_write(Wrapper* self, const CopyEndpoint& destination)
{
    if (!destination.wrapper || !destination.wrapper->own().title_memory ||
        x2w_startup::linear_buffer_mode() == x2w_startup::LinearBufferMode::off)
        return;
    if (x2w_startup::linear_detail::mutant("gpuw-at-record")) {
        // Mutant: the backing changes at recording, so a discarded recording still stops its refreshes.
        linear_buffers().note_gpu_write(destination.resource);
        return;
    }
    AcquireSRWLockExclusive(&pending_writes_lock);
    auto& writes = pending_linear_writes()[self];
    if (std::find(writes.begin(), writes.end(), destination.resource) == writes.end()) writes.push_back(destination.resource);
    ReleaseSRWLockExclusive(&pending_writes_lock);
}

// Refusal shared by the three handlers once the descriptions passed, or none.
x2w_startup::CopyRefusal check_roles_and_sync(Wrapper* self, const CopyEndpoint& to, const CopyEndpoint& from, bool sync)
{
    const auto roles = check_endpoint_roles(to, from);
    if (roles != x2w_startup::CopyRefusal::none) return roles;
    if (sync && !sync_title_source(self, from)) return x2w_startup::CopyRefusal::title_backed_source;
    return x2w_startup::CopyRefusal::none;
}

// A covered range in a copied placement is useful only if the recorded copy read an authoritative source.
bool copy_source_authoritative(Wrapper* self, const CopyEndpoint& from, u64 offset, u64 bytes)
{
    if (!from.wrapper || !InterlockedCompareExchange(&from.wrapper->own().refs, 0, 0)) return false;
    const auto* info = resource_info(from.wrapper);
    if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON) return false;
    x2w_startup::ComPtr<ID3D12Device> device;
    if (FAILED(from.resource->GetDevice(IID_PPV_ARGS(device.GetAddressOf()))) || device.Get() != device_of(self->own().list_device))
        return false;
    if (!from.wrapper->own().title_memory) return true;
    if (!info->placement_device || info->placement_device != self->own().list_device) return false;
    const auto address = reinterpret_cast<u64>(from.wrapper->own().title_memory);
    if (offset > UINT64_MAX - address) return false;
    const auto placed = placed_buffer_range(self->own().list_device, address + offset, bytes);
    if (placed.reject || placed.owner != from.wrapper) return false;
    return placed.alias || list_copied_gpu_range(self, from.resource, offset, bytes);
}

// List slot 15, CopyBufferRegion(destination, destination offset, source, source offset, bytes).
void __stdcall list_copy_buffer_region(Wrapper* self, u64 destination, u64 destination_offset, u64 source, u64 source_offset,
                                       u64 bytes)
{
    if (!self->own().list_open) return;
    const CopyEndpoint to = copy_endpoint(destination, false);
    const CopyEndpoint from = copy_endpoint(source, false);
    D3D12_RESOURCE_DESC to_desc{}, from_desc{};
    if (to.resource) to_desc = to.resource->GetDesc();
    if (from.resource) from_desc = from.resource->GetDesc();
    auto refusal = x2w_startup::check_copy_buffer_region(to.resource ? &to_desc : nullptr, destination_offset,
                                                         from.resource ? &from_desc : nullptr, source_offset, bytes,
                                                         to.resource == from.resource);
    if (refusal == x2w_startup::CopyRefusal::none) refusal = check_endpoint_roles(to, from);
    const bool authoritative = refusal == x2w_startup::CopyRefusal::none && indirect_copy_ranges() &&
        copy_source_authoritative(self, from, source_offset, bytes);
    if (refusal == x2w_startup::CopyRefusal::none && !authoritative && !sync_title_source(self, from))
        refusal = x2w_startup::CopyRefusal::title_backed_source;
    if (refusal != x2w_startup::CopyRefusal::none) return refuse_copy("CopyBufferRegion", refusal);
    require_copy_state(self, from.resource, D3D12_RESOURCE_STATE_COPY_SOURCE);
    require_copy_state(self, to.resource, D3D12_RESOURCE_STATE_COPY_DEST);
    static_cast<ID3D12GraphicsCommandList*>(self->own().real)
        ->CopyBufferRegion(to.resource, destination_offset, from.resource, source_offset, bytes);
    if (mesh_xmem_tracking()) {
        D3D12_HEAP_TYPE source_heap;
        const bool upload = fixed_heap(from.resource, source_heap) && source_heap == D3D12_HEAP_TYPE_UPLOAD;
        const bool srv_word = mesh_xmem_handoff() && bytes == sizeof(UINT) && upload;
        auto* list = static_cast<ID3D12GraphicsCommandList*>(self->own().real);
        D3D12_HEAP_PROPERTIES source_properties{};
        const bool mesh_record = mesh_xmem_indirect() && bytes == sizeof(D3D12_DISPATCH_MESH_ARGUMENTS) &&
            list->GetType() == D3D12_COMMAND_LIST_TYPE_COMPUTE &&
            (from_desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) &&
            SUCCEEDED(from.resource->GetHeapProperties(&source_properties, nullptr)) &&
            source_properties.Type == D3D12_HEAP_TYPE_DEFAULT;
        const char* source_kind = upload ? "upload" : mesh_record ? "default-uav" :
            (from_desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ? "uav-capable-buffer" : "buffer";
        note_mesh_xmem_write(self, to.wrapper, destination_offset, bytes, 15, source_kind,
                             srv_word || mesh_record ? from.resource : nullptr, source_offset);
    }
    note_linear_gpu_write(self, to);
    if (indirect_copy_ranges()) {
        const auto* info = to.wrapper ? resource_info(to.wrapper) : nullptr;
        note_copy_gpu_range(self, to.resource, destination_offset, bytes, authoritative && info &&
            info->placement == Placement::copied_buffer && info->placement_device == self->own().list_device);
    }
    InterlockedIncrement(&list_buffer_copies);
}

// List slot 17, CopyResource(destination, source).
void __stdcall list_copy_resource(Wrapper* self, u64 destination, u64 source)
{
    if (!self->own().list_open) return;
    const CopyEndpoint to = copy_endpoint(destination);
    const CopyEndpoint from = copy_endpoint(source);
    D3D12_RESOURCE_DESC to_desc{}, from_desc{};
    if (to.resource) to_desc = to.resource->GetDesc();
    if (from.resource) from_desc = from.resource->GetDesc();
    auto refusal = x2w_startup::check_copy_resource(to.resource ? &to_desc : nullptr, from.resource ? &from_desc : nullptr,
                                                    to.resource == from.resource);
    if (refusal == x2w_startup::CopyRefusal::none) refusal = check_roles_and_sync(self, to, from, true);
    if (refusal != x2w_startup::CopyRefusal::none) return refuse_copy("CopyResource", refusal);
    require_copy_state(self, from.resource, D3D12_RESOURCE_STATE_COPY_SOURCE);
    require_copy_state(self, to.resource, D3D12_RESOURCE_STATE_COPY_DEST);
    static_cast<ID3D12GraphicsCommandList*>(self->own().real)->CopyResource(to.resource, from.resource);
    retire_hold(self, to.wrapper);
    retire_hold(self, from.wrapper);
    if (to_desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        note_mesh_xmem_write(self, to.wrapper, 0, to_desc.Width, 17, "buffer");
        note_copy_gpu_range(self, to.resource, 0, to_desc.Width, false);
        note_linear_gpu_write(self, to);
    } else mark_gpu_written(self, to);
    InterlockedIncrement(&list_resource_copies);
}

// List slot 19, ResolveSubresource(destination, destination subresource, source, source subresource, format).
// The format is a console value, mapped like a resource format; a typeless one resolves as its typed choice.
// The source is multisampled, which no title memory can back, so it needs no synchronisation.
void __stdcall list_resolve_subresource(Wrapper* self, u64 destination, UINT destination_subresource, u64 source,
                                        UINT source_subresource, UINT format)
{
    if (!self->own().list_open) return;
    const CopyEndpoint to = copy_endpoint(destination);
    const CopyEndpoint from = copy_endpoint(source);
    D3D12_RESOURCE_DESC to_desc{}, from_desc{};
    if (to.resource) to_desc = to.resource->GetDesc();
    if (from.resource) from_desc = from.resource->GetDesc();
    const auto plan = x2w_startup::plan_resolve(to.resource ? &to_desc : nullptr, destination_subresource,
                                                from.resource ? &from_desc : nullptr, source_subresource,
                                                desktop_format(static_cast<DXGI_FORMAT>(format)), to.resource == from.resource);
    auto refusal = plan.refusal;
    if (refusal == x2w_startup::CopyRefusal::none) refusal = check_roles_and_sync(self, to, from, false);
    if (refusal != x2w_startup::CopyRefusal::none) return refuse_copy("ResolveSubresource", refusal);
    require_copy_state(self, from.resource, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    require_copy_state(self, to.resource, D3D12_RESOURCE_STATE_RESOLVE_DEST);
    static_cast<ID3D12GraphicsCommandList*>(self->own().real)
        ->ResolveSubresource(to.resource, destination_subresource, from.resource, source_subresource, plan.format);
    retire_hold(self, to.wrapper);  // the source is multisampled, so never a member of the retire class
    mark_gpu_written(self, to);
    InterlockedIncrement(&list_resolves);
}

// Fixtures: how many buffer copies, resource copies and resolves were forwarded, and how many were refused.
extern "C" __declspec(dllexport) void x2w_d3d12_list_copy_stats(unsigned* buffer_copies, unsigned* resource_copies,
                                                                unsigned* resolves, unsigned* refused)
{
    if (buffer_copies) *buffer_copies = static_cast<unsigned>(InterlockedCompareExchange(&list_buffer_copies, 0, 0));
    if (resource_copies) *resource_copies = static_cast<unsigned>(InterlockedCompareExchange(&list_resource_copies, 0, 0));
    if (resolves) *resolves = static_cast<unsigned>(InterlockedCompareExchange(&list_resolves, 0, 0));
    if (refused) *refused = static_cast<unsigned>(InterlockedCompareExchange(&list_copies_refused, 0, 0));
}
