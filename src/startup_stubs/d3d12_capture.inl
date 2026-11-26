// Experimental (docs/startup-runtime.md): frame capture for offline translation work.
// Included inside d3d12_xs_device.cpp's anonymous namespace.
//
//   X2W_D3D12_CAPTURE=<first frame>[,<count>]   X2W_D3D12_CAPTURE_DIR=<directory>
//
// Writes <dir>\capture.bin: a sequence of chunks, each {char tag[4]; u32 header bytes; u64 data bytes;
// header; data}. Chunks: FRAM, CALL (every command-list call with its arguments), MEM (title memory
// an argument points at), PSO / ROOT / RES / HEAP (objects, once each), VIEW (descriptor contents),
// TABL (descriptor table binds), PRES (present). tools read it with out/gamerun/capture.py.

struct Translation;
struct PipelineInfo {
    Translation* stream_translation = nullptr;
    bool has_stream_blend = false;
    D3D12_BLEND_DESC stream_blend{};
    bool has_stream_depth_stencil = false;
    D3D12_DEPTH_STENCIL_DESC stream_depth_stencil{};
    bool has_stream_rasterizer = false;
    D3D12_RASTERIZER_DESC stream_rasterizer{};
    DXGI_FORMAT stream_dsv_format = DXGI_FORMAT_UNKNOWN;
    int path = 0;                    // creating device slot: 57, 67, 55, 86, 62
    Wrapper* base = nullptr;         // slot 55: the pipeline it derives from
    std::vector<unsigned char> desc;
    std::vector<unsigned char> state_a, state_b;  // slot 57: the two state blocks, copied at creation (the title frees them)
};
struct ViewRecord {
    std::uint32_t id, slot;          // slot: 17 CBV, 18 SRV, 19 UAV, 20 RTV, 21 DSV, 22 sampler
    u64 resource, counter;           // title-side wrapper pointers (or 0)
    u64 dest_title, dest_desktop;
    std::uint32_t desc_bytes, pad;
    unsigned char desc[64];
};
struct HeapInfo {
    u64 wrapper;
    std::uint32_t type, count, increment, logical_count;
    u64 title_cpu, desktop_cpu, desktop_gpu;
    u64 title_sampler_cpu;
    std::uint32_t* shadow;           // per descriptor: view id + 1 (0: nothing known)
};
enum class Placement { none, aliased_buffer, copied_buffer };
struct PlacedVertexView {
    const Wrapper* owner{};
    ID3D12Resource* alias{};
    bool reject{};
};
struct ResourceInfo {
    D3D12_RESOURCE_DESC console;     // the description as the title passed it
    u64 bytes;                       // allocation size when known, else 0
    Placement placement{};
    const Wrapper* placement_device{};
    D3D12_RESOURCE_STATES initial_state{};
};

SRWLOCK note_lock = SRWLOCK_INIT;

// Time spent waiting for note_lock, by kind of acquisition, printed with the heartbeat (print_note_lock_waits). A wait is the
// time between asking for the lock and holding it; a long one means another thread held it (or was descheduled while it did).
struct NoteLockWaits {
    volatile LONG64 calls[2], ticks[2], max_ticks[2], over_5ms[2];  // [0] shared, [1] exclusive
};
NoteLockWaits note_lock_waits;
inline void note_lock_acquire(bool exclusive)
{
    LARGE_INTEGER start, end, frequency;
    QueryPerformanceCounter(&start);
    if (exclusive) AcquireSRWLockExclusive(&note_lock);
    else AcquireSRWLockShared(&note_lock);
    QueryPerformanceCounter(&end);
    QueryPerformanceFrequency(&frequency);
    const LONG64 waited = end.QuadPart - start.QuadPart;
    const int k = exclusive ? 1 : 0;
    InterlockedIncrement64(&note_lock_waits.calls[k]);
    InterlockedAdd64(&note_lock_waits.ticks[k], waited);
    for (LONG64 seen = note_lock_waits.max_ticks[k]; waited > seen;) {
        const LONG64 before = InterlockedCompareExchange64(&note_lock_waits.max_ticks[k], waited, seen);
        if (before == seen) break;
        seen = before;
    }
    if (waited * 200 > frequency.QuadPart) InterlockedIncrement64(&note_lock_waits.over_5ms[k]);
}
// One line for the heartbeat; resets the counters. Empty when the lock was never taken.
inline bool note_lock_wait_line(char* out, std::size_t size)
{
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    const double per_ms = frequency.QuadPart / 1000.0;
    LONG64 calls[2], ticks[2], max_ticks[2], over[2];
    for (int k = 0; k < 2; ++k) {
        calls[k] = InterlockedExchange64(&note_lock_waits.calls[k], 0);
        ticks[k] = InterlockedExchange64(&note_lock_waits.ticks[k], 0);
        max_ticks[k] = InterlockedExchange64(&note_lock_waits.max_ticks[k], 0);
        over[k] = InterlockedExchange64(&note_lock_waits.over_5ms[k], 0);
    }
    if (!calls[0] && !calls[1]) return false;
    std::snprintf(out, size, "x2w-d3d12 note_lock waits since the last report: shared %lld calls, %.1f ms total, max %.1f ms, %lld over 5 ms; "
                  "exclusive %lld calls, %.1f ms total, max %.1f ms, %lld over 5 ms",
                  static_cast<long long>(calls[0]), ticks[0] / per_ms, max_ticks[0] / per_ms, static_cast<long long>(over[0]),
                  static_cast<long long>(calls[1]), ticks[1] / per_ms, max_ticks[1] / per_ms, static_cast<long long>(over[1]));
    return true;
}
std::unordered_map<const Wrapper*, PipelineInfo>& pipeline_infos() { static std::unordered_map<const Wrapper*, PipelineInfo> m; return m; }
std::unordered_map<const Wrapper*, std::vector<unsigned char>>& root_blobs() { static std::unordered_map<const Wrapper*, std::vector<unsigned char>> m; return m; }
std::unordered_map<const Wrapper*, ResourceInfo>& resource_infos() { static std::unordered_map<const Wrapper*, ResourceInfo> m; return m; }
std::vector<HeapInfo>& heap_infos() { static std::vector<HeapInfo> v; return v; }
std::vector<ViewRecord>& view_records() { static std::vector<ViewRecord> v; return v; }
std::vector<std::uint8_t>& view_materialized() { static std::vector<std::uint8_t> v; return v; }  // 1 if desktop view was created
std::unordered_map<u64, std::uint32_t>& view_ids() { static std::unordered_map<u64, std::uint32_t> m; return m; }  // content hash -> view id
LONG64 views_noted, views_distinct, descriptors_copied;

// X2W_D3D12_VIEW_CENSUS=1 (diagnostic only, docs/startup-runtime.md): new distinct view records by slot, and one line
// before an insertion that is expected to grow view_records or the view_ids buckets. All fields: note_lock held.
bool view_census_enabled()
{
    static const bool on = x2w_local::env_on("X2W_D3D12_VIEW_CENSUS");
    return on;
}
struct ViewCensus {
    u64 distinct[6];  // slots 17..22
    u64 grows_records, grows_ids;  // growths announced
    u64 missed;                    // capacity or bucket count changed without an announcement
};
ViewCensus view_census{};

std::size_t view_census_format(char* out, std::size_t capacity, const char* what)
{
    const ViewCensus& c = view_census;
    const int n = std::snprintf(out, capacity,
        "x2w-view-census %s: cbv=%llu srv=%llu uav=%llu rtv=%llu dsv=%llu sampler=%llu records=%llu capacity=%llu "
        "ids=%llu buckets=%llu grows-records=%llu grows-ids=%llu missed=%llu",
        what, c.distinct[0], c.distinct[1], c.distinct[2], c.distinct[3], c.distinct[4], c.distinct[5],
        static_cast<u64>(view_records().size()), static_cast<u64>(view_records().capacity()),
        static_cast<u64>(view_ids().size()), static_cast<u64>(view_ids().bucket_count()),
        c.grows_records, c.grows_ids, c.missed);
    return n < 0 ? 0 : (std::min)(static_cast<std::size_t>(n), capacity - 1);
}

// Fixture only (x2w_d3d12_fail_view_allocations, startup_view_alloc): the next `count` allocations at a site fail as an
// exhausted heap would. The vector sites fail only when the vector has to grow; view_ids fails before any insertion,
// because each new entry allocates.
enum ViewAllocation { view_records_site, view_materialized_site, view_ids_site, view_allocation_sites };
volatile LONG view_allocations_to_fail[view_allocation_sites];

void view_allocation_gate(ViewAllocation site)
{
    for (LONG n = view_allocations_to_fail[site]; n > 0; n = view_allocations_to_fail[site])
        if (InterlockedCompareExchange(&view_allocations_to_fail[site], n - 1, n) == n) throw std::bad_alloc();
}

// The shadow entry for a desktop CPU descriptor handle (note_lock held), or null.
std::uint32_t* shadow_slot(u64 desktop_handle, std::uint32_t* room = nullptr)
{
    for (const HeapInfo& h : heap_infos()) {
        if (desktop_handle < h.desktop_cpu || desktop_handle >= h.desktop_cpu + u64{h.count} * h.increment || !h.shadow) continue;
        const u64 index = (desktop_handle - h.desktop_cpu) / h.increment;
        if (room) *room = static_cast<std::uint32_t>(h.count - index);
        return h.shadow + index;
    }
    return nullptr;
}
std::uint32_t shadow_id(u64 desktop_handle)
{
    note_lock_acquire(false);
    const std::uint32_t* slot = shadow_slot(desktop_handle);
    const std::uint32_t id = slot ? *slot : 0;
    ReleaseSRWLockShared(&note_lock);
    return id;
}

// A descriptor slot that names a desktop resource: 18 SRV, 19 UAV, 20 RTV, 21 DSV. Slot 19 names two resources, the
// view resource and the counter resource it may be paired with; slots 17 (CBV) and 22 (sampler) name none.
bool view_slot_names_resource(std::uint32_t slot)
{
    return slot >= 18 && slot <= 21;
}
// The refs a materialized view record holds in a tracked descriptor slot: one for the slot's resource and, for a
// slot-19 UAV, one more for its counter resource. One ref per slot, so the draw drain (retire_view_refs == 0) frees the
// resource only once no tracked slot names it; repeated writes of the same view add nothing (note_view compares ids).
void view_refs_add(std::uint32_t slot, u64 resource, u64 counter)
{
    if (!view_slot_names_resource(slot)) return;
    if (resource) retire_add_view_ref(reinterpret_cast<const void*>(resource));
    if (slot == 19 && counter) retire_add_view_ref(reinterpret_cast<const void*>(counter));
}
// Drops exactly the refs view_refs_add added for a materialized record: the overwrite in note_view, a copy that moves
// another view over the slot (note_copy), and the destruction of the slot's heap (retire_drop_heap_views).
void view_refs_release(std::uint32_t slot, u64 resource, u64 counter)
{
    if (!view_slot_names_resource(slot)) return;
    if (resource) retire_release_view_ref(reinterpret_cast<const void*>(resource));
    if (slot == 19 && counter) retire_release_view_ref(reinterpret_cast<const void*>(counter));
}
// The untracked-descriptor fallback: no slot can hold the resource for a view in a heap the shim does not shadow, so
// the use takes the resource (and a slot-19 counter) out of the retire class for good, as before.
void view_refs_leave_class(std::uint32_t slot, u64 resource, u64 counter, RetireLeft why)
{
    if (!view_slot_names_resource(slot)) return;
    if (resource) retire_leave_class(reinterpret_cast<const void*>(resource), why);
    if (slot == 19 && counter) retire_leave_class(reinterpret_cast<const void*>(counter), why);
}

constexpr std::uint32_t view_token_magic = 0x56573258;  // "X2WV"

bool title_backed(u64 title_handle, u64 bytes)
{
    bool inside = false;
    AcquireSRWLockShared(&descriptor_lock);
    for (const auto& r : descriptor_ranges)
        if (title_handle >= r.title && bytes <= r.bytes && title_handle - r.title <= r.bytes - bytes) inside = true;
    ReleaseSRWLockShared(&descriptor_lock);
    return inside;
}
// True when the descriptor already holds a view created on the desktop: by its desktop shadow entry, or by a
// materialized token the title copied into title memory with its own memory copy.
bool descriptor_has_live_view(u64 desktop_handle, u64 title_handle)
{
    note_lock_acquire(false);
    const std::uint32_t* slot = shadow_slot(desktop_handle);
    std::uint32_t id = slot ? *slot : 0;
    bool live = id && id - 1 < view_records().size() && view_materialized()[id - 1];
    ReleaseSRWLockShared(&note_lock);
    if (live || title_handle == desktop_handle || !title_backed(title_handle, 32) ||
        !x2w_local::readable(reinterpret_cast<const void*>(title_handle), 32))
        return live;
    std::uint32_t words[8];
    std::memcpy(words, reinterpret_cast<const void*>(title_handle), sizeof(words));
    if (words[0] != view_token_magic) return false;
    note_lock_acquire(false);
    live = words[1] < view_records().size() && view_materialized()[words[1]];
    ReleaseSRWLockShared(&note_lock);
    return live;
}

void note_pipeline(Wrapper* w, int path, const void* base, const void* desc, std::size_t bytes,
                   Translation* translation = nullptr, const D3D12_BLEND_DESC* blend = nullptr,
                   const D3D12_DEPTH_STENCIL_DESC* depth_stencil = nullptr,
                   DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN,
                   const D3D12_RASTERIZER_DESC* rasterizer = nullptr)
{
    PipelineInfo info;
    info.stream_translation = translation;
    if (blend) {
        info.stream_blend = *blend;
        info.has_stream_blend = true;
    }
    if (depth_stencil) {
        info.stream_depth_stencil = *depth_stencil;
        info.has_stream_depth_stencil = true;
    }
    if (rasterizer) {
        info.stream_rasterizer = *rasterizer;
        info.has_stream_rasterizer = true;
    }
    info.stream_dsv_format = dsv_format;
    info.path = path;
    info.base = as_wrapper(base);
    {
        Timed probe_timed(time_pipe_probe);
        const std::size_t have = desc ? readable_window(reinterpret_cast<u64>(desc), bytes) : 0;
        if (have) info.desc.assign(static_cast<const unsigned char*>(desc), static_cast<const unsigned char*>(desc) + have);
        if (path == 57 && have >= 0x28) {
            u64 words[5];
            std::memcpy(words, desc, sizeof(words));
            if (words[2] && words[2] <= 0x1000 && x2w_local::readable(reinterpret_cast<const void*>(words[1]), words[2]))
                info.state_a.assign(reinterpret_cast<const unsigned char*>(words[1]), reinterpret_cast<const unsigned char*>(words[1]) + words[2]);
            if (words[4] && words[4] <= 0x1000 && x2w_local::readable(reinterpret_cast<const void*>(words[3]), words[4]))
                info.state_b.assign(reinterpret_cast<const unsigned char*>(words[3]), reinterpret_cast<const unsigned char*>(words[3]) + words[4]);
        }
    }
    {
        Timed lock_timed(time_pipe_lock);  // the wait for note_lock: the guard ends once the lock is held
        note_lock_acquire(true);
    }
    pipeline_infos()[w] = std::move(info);
    ReleaseSRWLockExclusive(&note_lock);
}
const PipelineInfo* pipeline_info(const Wrapper* w)
{
    note_lock_acquire(false);
    const auto found = pipeline_infos().find(w);
    const PipelineInfo* info = found == pipeline_infos().end() ? nullptr : &found->second;
    ReleaseSRWLockShared(&note_lock);
    return info;
}
void note_root_blob(Wrapper* w, const void* blob, std::size_t bytes)
{
    if (!blob || !bytes || bytes > (1u << 20)) return;
    std::vector<unsigned char> copy(static_cast<const unsigned char*>(blob), static_cast<const unsigned char*>(blob) + bytes);
    // The blob the d3d12_xs serializer returned after removing the title's heap-indexing flags (see
    // serialize_for_runtime in src/d3d12_xs_shim.cpp): the copy kept for shader translation gets that
    // call's bits back, because the title's shaders take the heap pointers in the registers they add.
    if (const std::uint32_t stripped = x2w_stripped_root_flags(blob, bytes); stripped && bytes >= 32 && !std::memcmp(copy.data(), "DXBC", 4)) {
        std::uint32_t parts = 0;
        std::memcpy(&parts, copy.data() + 28, 4);
        for (std::uint32_t i = 0; i < parts && 32 + 4 * (i + 1) <= bytes; ++i) {
            std::uint32_t at = 0;
            std::memcpy(&at, copy.data() + 32 + 4 * i, 4);
            if (std::size_t{at} + 8 + 24 <= bytes && !std::memcmp(copy.data() + at, "RTS0", 4)) {
                std::uint32_t flags = 0;
                std::memcpy(&flags, copy.data() + at + 8 + 20, 4);
                flags |= stripped;
                std::memcpy(copy.data() + at + 8 + 20, &flags, 4);
            }
        }
    }
    note_lock_acquire(true);
    root_blobs()[w] = std::move(copy);
    ReleaseSRWLockExclusive(&note_lock);
}
// This index only lists active slot-133 wrappers. The existing resource record
// owns the extent and backing. Large resources must not widen the search
// window for thousands of small vertex buffers.
struct PlacementIndex {
    std::multimap<u64, const Wrapper*> by_start;
    u64 max_width{};
};
std::array<PlacementIndex, 2> placement_index;
PlacementIndex& placement_group(u64 width)
{
    return placement_index[width > D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT];
}

// Resources by the title memory they start at (note_lock held), for the memory SRV lookups. An entry stays
// as long as the resource's record in resource_infos does (never pruned); every lookup applies its own
// live-reference filter, so a released resource is skipped there and found again if it is referenced again.
x2w_startup::AddressIndex<const Wrapper*>& resource_address_index() { static x2w_startup::AddressIndex<const Wrapper*> m; return m; }
// X2W_RESOURCE_INDEX_CHECK=1 checks the index on every memory SRV call (check_resource_address_index);
// =2 is the negative control: the same check with index inserts skipped, which it must report.
int resource_index_check_mode()
{
    // Only "1" and "2" turn it on; anything else, including a value too long for the buffer, is off.
    static const int mode = [] {
        char v[4]{};
        const DWORD n = GetEnvironmentVariableA("X2W_RESOURCE_INDEX_CHECK", v, sizeof(v));
        return n == 1 && (v[0] == '1' || v[0] == '2') ? v[0] - '0' : 0;
    }();
    return mode;
}
u64 resource_index_extent(const ResourceInfo& info)
{
    return (std::max)(info.bytes, info.console.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? info.console.Width : 0);
}
// The index entry of a resource whose record is already published (publish_placement's hook); may throw.
void index_resource(const Wrapper* w, const ResourceInfo& info)
{
    const u64 base = reinterpret_cast<u64>(w->own().title_memory);
    if (!base || resource_index_check_mode() == 2) return;
    resource_address_index().insert(base, resource_index_extent(info), w);
    x2w_startup::note_placement_change();
}
void note_resource(Wrapper* w, const D3D12_RESOURCE_DESC* console, u64 bytes)
{
    if (!w || !console) return;
    x2w_startup::ExclusiveNoteLock guard(note_lock);
    x2w_startup::note_placement_change();
    const ResourceInfo info{*console, bytes};
    // Record and index change together or not at all (store_indexed); mode 2 leaves the index out.
    x2w_startup::store_indexed(resource_address_index(), resource_infos(), static_cast<const Wrapper*>(w),
        resource_index_check_mode() == 2 ? 0 : reinterpret_cast<u64>(w->own().title_memory), resource_index_extent(info), info);
}
void retire_placement(Wrapper* w)
{
    // Keep immutable capture data available to resource_info() readers.
    note_lock_acquire(true);
    const auto found = resource_infos().find(w);
    if (found != resource_infos().end() && found->second.placement != Placement::none) {
        auto& group = placement_group(found->second.bytes);
        const auto [first, last] = group.by_start.equal_range(reinterpret_cast<u64>(w->own().title_memory));
        for (auto it = first; it != last; ++it) {
            if (it->second == w) {
                group.by_start.erase(it);
                break;
            }
        }
        found->second.placement = Placement::none;
        x2w_startup::note_placement_change();
    }
    ReleaseSRWLockExclusive(&note_lock);
}

PlacedVertexView placed_buffer_range(const Wrapper* device, u64 start, u64 bytes, bool aliases_only = false)
{
    PlacedVertexView result{};
    if (!start || !bytes) return result;
    if (bytes > ~u64{} - start) return {nullptr, nullptr, true};
    note_lock_acquire(false);
    const u64 end = start + bytes;
    for (const auto& group : placement_index) {
        if (group.by_start.empty()) continue;
        const u64 earliest = start >= group.max_width ? start - group.max_width + 1 : 0;
        for (auto it = group.by_start.lower_bound(earliest); it != group.by_start.end() && it->first < end; ++it) {
            const auto* owner = it->second;
            const auto& info = resource_infos().at(owner);
            if (info.placement_device != device || (aliases_only && info.placement != Placement::aliased_buffer))
                continue;
            const u64 base = it->first;
            if (base <= start && start - base >= info.bytes) continue;
            if (result.owner || start < base || bytes > info.bytes - (start - base)) {
                result = {nullptr, nullptr, true}; // Ambiguous or crossing an allocation.
                break;
            }
            result.owner = owner;
            if (info.placement == Placement::aliased_buffer)
                result.alias = static_cast<ID3D12Resource*>(owner->own().real);
        }
        if (result.reject) break;
    }
    ReleaseSRWLockShared(&note_lock);
    return result;
}
const ResourceInfo* resource_info(const Wrapper* w)
{
    note_lock_acquire(false);
    const auto found = resource_infos().find(w);
    const ResourceInfo* info = found == resource_infos().end() ? nullptr : &found->second;
    ReleaseSRWLockShared(&note_lock);
    return info;
}
void note_heap(Wrapper* w, UINT type, UINT count, UINT logical_count, UINT increment, u64 title_cpu, u64 desktop_cpu, u64 desktop_gpu,
               u64 title_sampler_cpu)
{
    note_lock_acquire(true);
    heap_infos().push_back(HeapInfo{reinterpret_cast<u64>(w), type, count, increment, logical_count, title_cpu, desktop_cpu, desktop_gpu, title_sampler_cpu,
        static_cast<std::uint32_t*>(VirtualAlloc(nullptr, (u64{count} + 1) * 4, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))});
    ReleaseSRWLockExclusive(&note_lock);
}

// Room for one more element, with the 1.5x growth push_back would use: reserve(size + 1) would copy on every insert.
template<class T>
void reserve_one_view_entry(std::vector<T>& v, ViewAllocation site)
{
    if (v.size() < v.capacity()) return;
    view_allocation_gate(site);
    v.reserve(v.capacity() + v.capacity() / 2 + 1);
}

// Adds a new record to the three view containers, or to none of them (note_lock held). Everything that can allocate
// comes first: the two reservations change no size, and a failed try_emplace leaves the map unchanged. The two
// push_back calls then have room and cannot fail. Returns the name of the container that could not allocate, or null.
const char* store_view(const ViewRecord& record, std::uint8_t created, u64 hash)
{
    const char* growing = "view_records";
    try {
        reserve_one_view_entry(view_records(), view_records_site);
        growing = "view_materialized";
        reserve_one_view_entry(view_materialized(), view_materialized_site);
        growing = "view_ids";
        view_allocation_gate(view_ids_site);
        view_ids().try_emplace(hash, record.id);
    } catch (const std::bad_alloc&) {
        return growing;
    }
    view_records().push_back(record);
    view_materialized().push_back(created);
    return nullptr;
}

// A new view could not be recorded. The desktop descriptor already holds it, so the destination must stop naming the
// view it held before: the slot and its title token are cleared, and a draw refuses the descriptor instead of binding
// the old view. Nothing records the new view's use of its resource, so the resource leaves the retire class and is not
// released early. Allocates nothing. Enters with note_lock held and releases it.
void drop_unrecorded_view(const ViewRecord& record, bool materialized, const char* container)
{
    if (std::uint32_t* slot_ptr = shadow_slot(record.dest_desktop)) {
        const std::uint32_t prev = *slot_ptr;
        *slot_ptr = 0;
        if (prev && prev - 1 < view_records().size() && view_materialized()[prev - 1])
            view_refs_release(view_records()[prev - 1].slot, view_records()[prev - 1].resource,
                              view_records()[prev - 1].counter);
    }
    if (materialized) view_refs_leave_class(record.slot, record.resource, record.counter, RetireLeft::view);
    ReleaseSRWLockExclusive(&note_lock);
    const std::size_t token_bytes = record.slot == 22 ? x2w_startup::title_sampler_stride : 32u;
    if (record.dest_title != record.dest_desktop && title_backed(record.dest_title, token_bytes) &&
        x2w_local::writable(reinterpret_cast<void*>(record.dest_title), token_bytes))
        std::memset(reinterpret_cast<void*>(record.dest_title), 0, token_bytes);
    static LONG failed;
    if (first_calls(failed, 8))
        log("x2w-d3d12 note_view: out of memory in %s -> view of slot %u not recorded, destination descriptor untracked",
            container, record.slot);
}

// A view is created: remembered by id; the id is also written into title-owned descriptor memory so
// that copies the title makes there itself can be followed.
void note_view(int slot, u64 resource, u64 counter, const void* desc, std::size_t desc_bytes, u64 dest_title, u64 dest_desktop,
               bool materialized = true)
{
    ViewRecord record{};
    record.slot = static_cast<std::uint32_t>(slot);
    record.resource = resource;
    record.counter = counter;
    record.dest_title = dest_title;
    record.dest_desktop = dest_desktop;
    if (desc && x2w_local::readable(desc, desc_bytes)) {
        record.desc_bytes = static_cast<std::uint32_t>(desc_bytes);
        std::memcpy(record.desc, desc, desc_bytes);
    }
    const std::uint8_t created = materialized ? 1 : 0;
    // Views with the same content and materialization state share one id, so per-frame re-creation does not grow the list.
    u64 hash = 1469598103934665603ull;
    const auto mix = [&hash](const void* data, std::size_t bytes) {
        for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ static_cast<const unsigned char*>(data)[i]) * 1099511628211ull;
    };
    mix(&record.slot, 4); mix(&record.resource, 8); mix(&record.counter, 8); mix(record.desc, sizeof(record.desc)); mix(&created, 1);
    note_lock_acquire(true);
    ++views_noted;
    const auto found = view_ids().find(hash);
    if (found != view_ids().end()) {
        record.id = found->second;
    } else {
        record.id = static_cast<std::uint32_t>(view_records().size());
        const bool census = view_census_enabled();
        std::size_t capacity_before = 0, buckets_before = 0;
        bool records_full = false, ids_full = false;
        if (census) {
            capacity_before = view_records().capacity();
            buckets_before = view_ids().bucket_count();
            records_full = view_records().size() == capacity_before;
            // MSVC's rule: unordered_map rehashes when the new size would exceed the maximum load factor.
            ids_full = static_cast<float>(view_ids().size() + 1) / static_cast<float>(buckets_before) >
                       view_ids().max_load_factor();
            // Logged before the allocation, so the last line survives an allocation that fails.
            if (records_full || ids_full) {
                view_census.grows_records += records_full;
                view_census.grows_ids += ids_full;
                char line[640];
                view_census_format(line, sizeof(line),
                                   records_full && ids_full ? "before growth of records and ids"
                                   : records_full           ? "before growth of records"
                                                            : "before growth of ids");
                log("%s (new slot %u)", line, record.slot);
            }
        }
        if (const char* container = store_view(record, created, hash)) {
            drop_unrecorded_view(record, materialized, container);
            return;
        }
        ++views_distinct;
        if (census) {
            if (record.slot >= 17 && record.slot <= 22) ++view_census.distinct[record.slot - 17];
            view_census.missed += !records_full && view_records().capacity() != capacity_before;
            view_census.missed += !ids_full && view_ids().bucket_count() != buckets_before;
        }
    }
    if (slot == 18 && resource && materialized) {
        if (Wrapper* res_w = as_wrapper(reinterpret_cast<const void*>(resource)); res_w && res_w->own().kind == resource_kind)
            res_w->own().lazy_view_created = true;
    }
    if (std::uint32_t* slot_ptr = shadow_slot(dest_desktop)) {
        const std::uint32_t prev = *slot_ptr;
        *slot_ptr = record.id + 1;
        // The same view written to the same slot again keeps its id and changes no reference count. A different view
        // takes over the slot: the new record adds its refs, the previous materialized record drops its own.
        if (prev != record.id + 1) {
            if (materialized) view_refs_add(static_cast<std::uint32_t>(slot), resource, counter);
            if (prev && prev - 1 < view_records().size() && view_materialized()[prev - 1])
                view_refs_release(view_records()[prev - 1].slot, view_records()[prev - 1].resource,
                                  view_records()[prev - 1].counter);
        }
    } else if (view_slot_names_resource(static_cast<std::uint32_t>(slot)) && resource) {
        // A descriptor no shadow slot tracks cannot hold the view: the use takes the resource out of the class.
        view_refs_leave_class(static_cast<std::uint32_t>(slot), resource, counter, RetireLeft::view);
    }
    ReleaseSRWLockExclusive(&note_lock);
    // The title wrote this view: a cached descriptor built for the id before the write is dropped (X2W_VIEW_CACHE).
    if (x2w_view_cache::Cache::enabled()) x2w_view_cache::view_cache().note_write(record.id);
    const std::size_t token_bytes = slot == 22 ? x2w_startup::title_sampler_stride : 32u;
    if (dest_title != dest_desktop && title_backed(dest_title, token_bytes) &&
        x2w_local::writable(reinterpret_cast<void*>(dest_title), token_bytes)) {
        std::uint32_t token[8] = {view_token_magic, record.id, static_cast<std::uint32_t>(slot), 0,
                                  static_cast<std::uint32_t>(resource), static_cast<std::uint32_t>(resource >> 32), 0, 0};
        std::memcpy(reinterpret_cast<void*>(dest_title), token, token_bytes);
    }
}
void note_copy(u64 dest_title, u64 src_title, u64 dest_desktop, u64 src_desktop, UINT count, UINT increment)
{
    note_lock_acquire(true);
    descriptors_copied += count;
    std::uint32_t dest_room = 0, src_room = 0;
    std::uint32_t* to = shadow_slot(dest_desktop, &dest_room);
    const std::uint32_t* from = shadow_slot(src_desktop, &src_room);
    if (to && count <= dest_room) {
        if (from && count <= src_room) {
            // Every copied-in view is now named by one more tracked slot; every view the destination named before is
            // replaced. Slots 18-21 count one ref (slot 19 one for the resource and one for its counter).
            for (UINT i = 0; i < count; ++i) {
                const std::uint32_t src_id = from[i];
                if (src_id && src_id - 1 < view_records().size() && view_materialized()[src_id - 1])
                    view_refs_add(view_records()[src_id - 1].slot, view_records()[src_id - 1].resource,
                                  view_records()[src_id - 1].counter);
            }
            for (UINT i = 0; i < count; ++i) {
                const std::uint32_t old_id = to[i];
                if (old_id && old_id - 1 < view_records().size() && view_materialized()[old_id - 1])
                    view_refs_release(view_records()[old_id - 1].slot, view_records()[old_id - 1].resource,
                                      view_records()[old_id - 1].counter);
            }
            std::memmove(to, from, u64{count} * 4);
        } else {
            for (UINT i = 0; i < count; ++i) {
                const std::uint32_t old_id = to[i];
                if (old_id && old_id - 1 < view_records().size() && view_materialized()[old_id - 1])
                    view_refs_release(view_records()[old_id - 1].slot, view_records()[old_id - 1].resource,
                                      view_records()[old_id - 1].counter);
            }
            std::memset(to, 0, u64{count} * 4);
        }
    } else if (from && count <= src_room) {
        // The destination is not tracked: the copy puts the views where no shadow slot can hold them.
        for (UINT i = 0; i < count; ++i) {
            const std::uint32_t src_id = from[i];
            if (src_id && src_id - 1 < view_records().size() && view_materialized()[src_id - 1])
                view_refs_leave_class(view_records()[src_id - 1].slot, view_records()[src_id - 1].resource,
                                      view_records()[src_id - 1].counter, RetireLeft::view_copy);
        }
    }
    ReleaseSRWLockExclusive(&note_lock);
    const UINT stride = title_descriptor_stride(dest_title, increment);
    const u64 bytes = u64{count} * stride;
    if (dest_title != dest_desktop && src_title != src_desktop &&
        stride == title_descriptor_stride(src_title, increment) && title_backed(dest_title, bytes) && title_backed(src_title, bytes))
        std::memmove(reinterpret_cast<void*>(dest_title), reinterpret_cast<const void*>(src_title), bytes);
}

LONG capture_active;
namespace capture {
HANDLE file = INVALID_HANDLE_VALUE;
SRWLOCK lock = SRWLOCK_INIT;
std::unordered_set<u64> objects, views, followed;
std::unordered_map<u64, u64> memory_seen;  // address -> hash of the content last written
LONG sequence, frame_now, first_frame = -1, frame_count = 1;
u64 total_bytes;

void put(const void* data, std::size_t bytes)
{
    DWORD written = 0;
    while (bytes) {
        const DWORD part = bytes > (16u << 20) ? (16u << 20) : static_cast<DWORD>(bytes);
        if (!WriteFile(file, data, part, &written, nullptr) || !written) break;  // unreadable source: stops, tool sees a short chunk
        data = static_cast<const char*>(data) + written;
        bytes -= written;
        total_bytes += written;
    }
}
void chunk(const char* tag, const void* header, std::size_t header_bytes, const void* data, std::size_t data_bytes)
{
    if (file == INVALID_HANDLE_VALUE) return;
    struct { char tag[4]; std::uint32_t header_bytes; u64 data_bytes; } head{};
    std::memcpy(head.tag, tag, 4);
    head.header_bytes = static_cast<std::uint32_t>(header_bytes);
    head.data_bytes = data_bytes;
    put(&head, sizeof(head));
    if (header_bytes) put(header, header_bytes);
    if (data_bytes) {
        // Title memory can go away under us: copy through a checked buffer.
        std::vector<unsigned char> copy(data_bytes);
        SIZE_T got = 0;
        ReadProcessMemory(GetCurrentProcess(), data, copy.data(), data_bytes, &got);
        put(copy.data(), data_bytes);
    }
}

void object(Wrapper* w);

// Dumps memory at address. depth > 0 also follows pointers found in its first 0x400 bytes: wrapped
// objects are described, other readable targets dumped (sized by the following word when plausible).
void memory(u64 address, std::size_t wanted, int depth = 0, std::uint32_t why = 0)
{
    if (address < 0x10000 || !wanted) return;
    const std::size_t bytes = readable_window(address, wanted);
    if (!bytes) return;
    // Written again whenever the content differs from what was last written for this address
    // (stack and transient memory is reused between calls); readers take the latest block.
    u64 hash = 1469598103934665603ull ^ bytes;
    {
        std::vector<unsigned char> probe(bytes);
        SIZE_T got = 0;
        ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), probe.data(), bytes, &got);
        for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ probe[i]) * 1099511628211ull;
    }
    const auto seen = memory_seen.find(address);
    if (seen == memory_seen.end() || seen->second != hash) {
        memory_seen[address] = hash;
        struct { u64 address, bytes, wanted; std::uint32_t why, sequence; } header{address, bytes, wanted, why, static_cast<std::uint32_t>(sequence)};
        chunk("MEM ", &header, sizeof(header), reinterpret_cast<const void*>(address), bytes);
    }
    if (depth <= 0) return;
    // Each address is expanded once per depth, and a capture follows a bounded number of pointers.
    if (!followed.insert(address * 4 + static_cast<u64>(depth)).second || followed.size() > 20000) return;
    const std::size_t scan = bytes < 0x400 ? bytes : 0x400;
    std::vector<u64> words(scan / 8);
    SIZE_T got = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), words.data(), words.size() * 8, &got)) return;
    for (std::size_t i = 0; i < words.size(); ++i) {
        const u64 value = words[i];
        if (value < 0x10000 || value >= 0x7fffffffffffull || (value & 3)) continue;
        if (Wrapper* w = as_wrapper(reinterpret_cast<const void*>(value))) { object(w); continue; }
        if (readable_window(value, 16) < 16) continue;
        const u64 next = i + 1 < words.size() ? words[i + 1] : 0;
        memory(value, next > 0 && next <= (4u << 20) ? static_cast<std::size_t>(next) : 0x100, depth - 1, why);
    }
}

void object(Wrapper* w)
{
    if (!w || !objects.insert(reinterpret_cast<u64>(w)).second) return;
    struct { u64 wrapper, real, title_memory, title_cpu; std::uint32_t kind, pad; } common{
        reinterpret_cast<u64>(w), reinterpret_cast<u64>(w->own().real), reinterpret_cast<u64>(w->own().title_memory), w->own().title_cpu,
        static_cast<std::uint32_t>(w->own().kind), 0};
    if (w->own().kind == pipeline_kind) {
        const PipelineInfo* info = pipeline_info(w);
        struct { u64 wrapper, base; std::uint32_t path, real; } header{common.wrapper, info ? reinterpret_cast<u64>(info->base) : 0,
            info ? static_cast<std::uint32_t>(info->path) : 0, w->own().real ? 1u : 0u};
        chunk("PSO ", &header, sizeof(header), info && !info->desc.empty() ? info->desc.data() : nullptr, info ? info->desc.size() : 0);
        if (info) {
            const std::size_t count = info->desc.size() / 8;
            for (std::size_t i = 0; i < count; ++i) {
                u64 value, next = 0;
                std::memcpy(&value, info->desc.data() + i * 8, 8);
                if (i + 1 < count) std::memcpy(&next, info->desc.data() + (i + 1) * 8, 8);
                if (value < 0x10000 || value >= 0x7fffffffffffull) continue;
                if (Wrapper* other = as_wrapper(reinterpret_cast<const void*>(value))) { object(other); continue; }
                if (readable_window(value, 16) < 16) continue;
                memory(value, next > 0 && next <= (32u << 20) ? static_cast<std::size_t>(next) : 0x100, 2, 0x50534f);
            }
            if (info->base) object(info->base);
        }
    } else if (w->own().kind == root_signature_kind) {
        note_lock_acquire(false);
        const auto found = root_blobs().find(w);
        std::vector<unsigned char> blob = found == root_blobs().end() ? std::vector<unsigned char>{} : found->second;
        ReleaseSRWLockShared(&note_lock);
        chunk("ROOT", &common, sizeof(common), blob.data(), blob.size());
    } else if (w->own().kind == resource_kind) {
        struct { u64 wrapper, title_memory, bytes, gpu; D3D12_RESOURCE_DESC console, desktop; } header{};
        header.wrapper = common.wrapper;
        header.title_memory = common.title_memory;
        if (const ResourceInfo* info = resource_info(w)) { header.console = info->console; header.bytes = info->bytes; }
        if (w->own().real) {
            header.desktop = static_cast<ID3D12Resource*>(w->own().real)->GetDesc();
            if (header.desktop.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) header.gpu = static_cast<ID3D12Resource*>(w->own().real)->GetGPUVirtualAddress();
        } else {
            lazy_texture_desc(w, header.desktop);
        }
        chunk("RES ", &header, sizeof(header), nullptr, 0);
    } else if (w->own().kind == descriptor_heap_kind) {
        HeapInfo found{};
        note_lock_acquire(false);
        for (const HeapInfo& heap : heap_infos())
            if (heap.wrapper == common.wrapper) found = heap;
        ReleaseSRWLockShared(&note_lock);
        found.wrapper = common.wrapper;
        chunk("HEAP", &found, sizeof(found), nullptr, 0);
    } else {
        chunk("OBJ ", &common, sizeof(common), nullptr, 0);
    }
}

// Texel/buffer bytes of a resource as the title wrote them (console layout), for sampled resources.
void resource_data(Wrapper* w)
{
    if (!w || w->own().kind != resource_kind || !w->own().title_memory) return;
    object(w);
    u64 bytes = 0;
    D3D12_RESOURCE_DESC desc{};
    if (const ResourceInfo* info = resource_info(w)) { bytes = info->bytes; desc = info->console; }
    if (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) return;
    if (!bytes && w->own().real) {
        const D3D12_RESOURCE_DESC desktop = static_cast<ID3D12Resource*>(w->own().real)->GetDesc();
        bytes = desktop.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? desktop.Width
            : device_of(the_device)->GetResourceAllocationInfo(0, 1, &desktop).SizeInBytes;
    }
    if (!bytes || bytes == ~0ull) bytes = 1u << 20;
    if (bytes > (96u << 20)) bytes = 96u << 20;
    memory(reinterpret_cast<u64>(w->own().title_memory), static_cast<std::size_t>(bytes), 0, 0x534552);
}

void view(std::uint32_t id_plus_one, bool with_data)
{
    if (!id_plus_one) return;
    ViewRecord record{};
    note_lock_acquire(false);
    const bool known = id_plus_one - 1 < view_records().size();
    if (known) record = view_records()[id_plus_one - 1];
    ReleaseSRWLockShared(&note_lock);
    if (!known) return;
    if (views.insert(record.id).second) chunk("VIEW", &record, sizeof(record), nullptr, 0);
    if (Wrapper* resource = as_wrapper(reinterpret_cast<const void*>(record.resource))) {
        object(resource);
        if (with_data && record.slot == 18) resource_data(resource);
    }
    if (record.slot == 17 && record.desc_bytes >= 12) {  // constant buffer view: {title GPU address, size}
        u64 address;
        std::uint32_t size;
        std::memcpy(&address, record.desc, 8);
        std::memcpy(&size, record.desc + 8, 4);
        memory(address, size, 0, 0x564243);
    }
}

// A descriptor table bind: the views behind the next `entries` descriptors, both as the desktop
// heap has them (shadow) and as the ids found in title descriptor memory (tokens).
void table(int slot, UINT parameter, u64 gpu_handle, std::uint32_t entries)
{
    HeapInfo heap{};
    note_lock_acquire(false);
    for (const HeapInfo& h : heap_infos())
        if (h.desktop_gpu && gpu_handle >= h.desktop_gpu && gpu_handle < h.desktop_gpu + u64{h.count} * h.increment) heap = h;
    ReleaseSRWLockShared(&note_lock);
    if (!heap.wrapper) {
        struct { std::uint32_t sequence, slot, parameter, entries; u64 gpu_handle, heap, index; } header{
            static_cast<std::uint32_t>(sequence), static_cast<std::uint32_t>(slot), parameter, 0, gpu_handle, 0, 0};
        chunk("TABL", &header, sizeof(header), nullptr, 0);
        return;
    }
    object(reinterpret_cast<Wrapper*>(heap.wrapper));
    const u64 offset = gpu_handle - heap.desktop_gpu;
    const u64 index = offset / heap.increment;
    if (index + entries > heap.count) entries = static_cast<std::uint32_t>(heap.count - index);
    std::vector<std::uint32_t> ids(entries * 2);  // [shadow ids + 1][token ids + 1]
    for (std::uint32_t i = 0; i < entries; ++i) ids[i] = heap.shadow ? heap.shadow[index + i] : 0;
    if (heap.title_cpu) {
        const UINT stride = title_descriptor_stride(heap.title_cpu, heap.increment);
        const u64 title_start = heap.title_cpu + index * stride;
        for (std::uint32_t i = 0; i < entries; ++i) {
            std::uint32_t token[2]{};
            SIZE_T got = 0;
            ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(title_start + u64{i} * stride), token, sizeof(token), &got);
            ids[entries + i] = token[0] == view_token_magic ? token[1] + 1 : 0;
        }
        memory(title_start, u64{entries} * stride, 0, 0x4c4254);
    }
    struct { std::uint32_t sequence, slot, parameter, entries; u64 gpu_handle, heap, index; } header{
        static_cast<std::uint32_t>(sequence), static_cast<std::uint32_t>(slot), parameter, entries, gpu_handle, heap.wrapper, index};
    chunk("TABL", &header, sizeof(header), ids.data(), ids.size() * 4);
    for (std::uint32_t id : ids) view(id, true);
}

void list_call(int slot, u64 list, u64 rdx, u64 r8, u64 r9, const u64* stack, u64 ret)
{
    AcquireSRWLockExclusive(&lock);
    if (file == INVALID_HANDLE_VALUE) { ReleaseSRWLockExclusive(&lock); return; }
    ++sequence;
    struct { std::uint32_t frame, sequence, thread, slot; u64 list, args[7], ret; } call{
        static_cast<std::uint32_t>(frame_now), static_cast<std::uint32_t>(sequence), GetCurrentThreadId(),
        static_cast<std::uint32_t>(slot), list, {rdx, r8, r9, stack[0], stack[1], stack[2], stack[3]}, ret};
    chunk("CALL", &call, sizeof(call), nullptr, 0);
    const u64 regs[3] = {rdx, r8, r9};
    for (u64 value : regs) {
        if (Wrapper* w = as_wrapper(reinterpret_cast<const void*>(value))) object(w);
        else memory(value, 0x100);
    }
    switch (slot) {
    case 32:  // SetDescriptorHeaps(count, heaps): also every sampler the sampler heap holds
        if (rdx <= 4 && x2w_local::readable(reinterpret_cast<const void*>(r8), rdx * 8)) {
            for (u64 i = 0; i < rdx; ++i) {
                Wrapper* heap = as_wrapper(reinterpret_cast<void* const*>(r8)[i]);
                object(heap);
                HeapInfo info{};
                note_lock_acquire(false);
                for (const HeapInfo& h : heap_infos())
                    if (h.wrapper == reinterpret_cast<u64>(heap)) info = h;
                ReleaseSRWLockShared(&note_lock);
                if (info.type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER && info.count <= 4096 && info.desktop_gpu) table(32, 0xffffffffu, info.desktop_gpu, info.count);
            }
        }
        break;
    case 35: case 36:  // Set{Compute,Graphics}RootDescriptorTable(parameter, GPU handle)
        table(slot, static_cast<UINT>(rdx), r8, 96);
        break;
    case 41: case 42: case 43: case 44: case 45: case 46:  // root CBV / SRV / UAV by title GPU address
        memory(r8, 0x10000);
        break;
    case 40: case 39:  // Set{Graphics,Compute}Root32BitConstants(parameter, count, data, offset)
        memory(r9, static_cast<std::size_t>(r8 & 0xffff) * 4);
        break;
    case 47:  // IASetIndexBuffer(view {address, size, format})
        if (x2w_local::readable(reinterpret_cast<const void*>(rdx), 16)) {
            const auto* v = reinterpret_cast<const D3D12_INDEX_BUFFER_VIEW*>(rdx);
            memory(v->BufferLocation, v->SizeInBytes > (32u << 20) ? (32u << 20) : v->SizeInBytes);
        }
        break;
    case 48:  // IASetVertexBuffers(start, count, views {address, size, stride})
        if (r8 <= 32 && x2w_local::readable(reinterpret_cast<const void*>(r9), r8 * 16)) {
            memory(r9, r8 * 16);
            for (u64 i = 0; i < r8; ++i) {
                const auto& v = reinterpret_cast<const D3D12_VERTEX_BUFFER_VIEW*>(r9)[i];
                memory(v.BufferLocation, v.SizeInBytes > (32u << 20) ? (32u << 20) : v.SizeInBytes);
            }
        }
        break;
    case 50:  // OMSetRenderTargets(count, handles, single range, depth handle)
        if (rdx <= 8 && x2w_local::readable(reinterpret_cast<const void*>(r8), rdx * 8)) {
            memory(r8, rdx * 8);
            std::vector<std::uint32_t> ids;
            for (u64 i = 0; i < rdx; ++i) ids.push_back(shadow_id(desktop_cpu_handle(reinterpret_cast<const u64*>(r8)[i])));
            if (stack[0] && x2w_local::readable(reinterpret_cast<const void*>(stack[0]), 8))
                ids.push_back(shadow_id(desktop_cpu_handle(*reinterpret_cast<const u64*>(stack[0]))));
            struct { std::uint32_t sequence, count; } header{static_cast<std::uint32_t>(sequence), static_cast<std::uint32_t>(ids.size())};
            chunk("TGTS", &header, sizeof(header), ids.data(), ids.size() * 4);
            for (std::uint32_t id : ids) view(id, false);
            if (stack[0]) memory(stack[0], 8);
        }
        break;
    case 52: case 51: {  // Clear{RenderTarget,DepthStencil}View(handle, ...)
        const std::uint32_t id = shadow_id(desktop_cpu_handle(rdx));
        struct { std::uint32_t sequence, count; } header{static_cast<std::uint32_t>(sequence), 1};
        chunk("TGTS", &header, sizeof(header), &id, 4);
        view(id, false);
        break;
    }
    case 146: case 30:  // barrier-like (count, array)
        memory(r8, (rdx && rdx <= 64 ? rdx : 1) * 0x40, 1);
        break;
    default:
        break;
    }
    ReleaseSRWLockExclusive(&lock);
}

void configure()
{
    char value[32]{};
    if (!GetEnvironmentVariableA("X2W_D3D12_CAPTURE", value, sizeof(value))) return;
    first_frame = std::atol(value);
    if (const char* comma = std::strchr(value, ',')) frame_count = std::atol(comma + 1);
    if (frame_count < 1) frame_count = 1;
}

// Called by present with the number of the frame it completes and the presented surface.
void present(LONG frame, Wrapper* surface)
{
    static const bool configured = (configure(), true);
    (void)configured;
    if (first_frame < 0) return;
    AcquireSRWLockExclusive(&lock);
    if (file != INVALID_HANDLE_VALUE) {
        struct { std::uint32_t frame, pad; u64 surface; } header{static_cast<std::uint32_t>(frame), 0, reinterpret_cast<u64>(surface)};
        if (surface) object(surface);
        chunk("PRES", &header, sizeof(header), nullptr, 0);
    }
    if (frame == first_frame - 1 && file == INVALID_HANDLE_VALUE) {
        wchar_t dir[MAX_PATH], path[MAX_PATH];
        if (GetEnvironmentVariableW(L"X2W_D3D12_CAPTURE_DIR", dir, MAX_PATH)) {
            swprintf(path, MAX_PATH, L"%s\\capture.bin", dir);
            file = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        }
        log("x2w-capture begin at frame %ld for %ld frames: %s", first_frame, frame_count, file == INVALID_HANDLE_VALUE ? "FAILED to open file" : "recording");
        InterlockedExchange(&capture_active, file != INVALID_HANDLE_VALUE);
    }
    if (file != INVALID_HANDLE_VALUE) {
        frame_now = frame + 1;
        if (frame >= first_frame + frame_count - 1) {
            InterlockedExchange(&capture_active, 0);
            CloseHandle(file);
            file = INVALID_HANDLE_VALUE;
            first_frame = -1;
            log("x2w-capture end: %ld calls, %llu bytes, %llu objects, %llu views", sequence, total_bytes,
                static_cast<u64>(objects.size()), static_cast<u64>(views.size()));
        } else {
            struct { std::uint32_t frame, pad; } header{static_cast<std::uint32_t>(frame_now), 0};
            chunk("FRAM", &header, sizeof(header), nullptr, 0);
        }
    }
    ReleaseSRWLockExclusive(&lock);
}
}  // namespace capture
