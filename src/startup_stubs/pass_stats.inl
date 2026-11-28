// Pass thermometer (X2W_PASS_STATS=1, docs/startup-runtime.md#pass-thermometer). Included by d3d12_xs_device.cpp
// after red_producer_diagnostic.inl: Pass, pass_of, find_state, find_split, device_of, the_device, frame_number
// and log are defined above it.
//
// A pass is a run of writes to one set of resources on one recording: draws (refused draws included) to the bound
// render targets and depth, consecutive dispatches with the same outputs, consecutive copies to render-target or
// unordered-access resources, and clears. When a pass closes, the title's list copies the written subresource of
// each resource into a snapshot (within a per-present pass cap and memory budget). When the list is executed, a
// reduction list on the same queue reads every snapshot (pass_stats_shader.h) into a small result buffer; at the next
// present the results are folded into one passes.jsonl line per pass and walked back from the presented surface.
#include "startup_stubs/pass_stats_math.h"
#include "startup_stubs/pass_stats_shader.h"

volatile LONG64 pass_stats_call = -1;  // the replay's current call (x2w_d3d12_pass_stats_call); -1 outside a replay

bool pass_stats_enabled()
{
    static const bool on = x2w_local::env_on("X2W_PASS_STATS");
    return on;
}

namespace pass_stats {
using x2w_startup::ComPtr;
namespace m = x2w_pass_stats;

enum class Kind { graphics, compute, copy, clear };
const char* kind_name(Kind kind)
{
    switch (kind) {
    case Kind::graphics: return "graphics";
    case Kind::compute: return "compute";
    case Kind::copy: return "copy";
    default: return "clear";
    }
}

struct Config {
    unsigned cap = 256;          // passes with stats per present (X2W_PASS_STATS_CAP)
    u64 budget = 2048ull << 20;  // snapshot bytes per present (X2W_PASS_STATS_MB)
    u64 buffer_limit = 64ull << 20;
};

unsigned env_number(const char* name, unsigned fallback, unsigned low, unsigned high)
{
    char value[16]{};
    const DWORD n = GetEnvironmentVariableA(name, value, sizeof(value));
    char* end = nullptr;
    const unsigned long parsed = n && n < sizeof(value) ? std::strtoul(value, &end, 10) : 0;
    return end && end != value && !*end && parsed >= low && parsed <= high ? static_cast<unsigned>(parsed) : fallback;
}

const Config& config()
{
    static const Config c = [] {
        Config made;
        made.cap = env_number("X2W_PASS_STATS_CAP", made.cap, 1, 1u << 20);
        made.budget = u64{env_number("X2W_PASS_STATS_MB", 2048, 16, 1u << 20)} << 20;
        return made;
    }();
    return c;
}

u64 creation_id(ID3D12Resource* resource)
{
    u64 id = 0;
    UINT size = sizeof(id);
    if (!resource || FAILED(resource->GetPrivateData(replay_dump::resource_id, &size, &id)) || size != sizeof(id)) return 0;
    return id;
}

struct Write {
    ID3D12Resource* resource = nullptr;  // identity only: no reference is held on the title's resource
    u64 id = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN, view = DXGI_FORMAT_UNKNOWN;
    UINT64 width = 0;
    UINT height = 0, mip = 0, slice = 0;
    bool buffer = false;
    u64 offset = 0, bytes = 0;  // a buffer: the written range (bytes 0: to the end)
    m::FormatInfo info;
    ComPtr<ID3D12Resource> snapshot;
    u64 texels = 0;                // texels (texture) or 32-bit words (buffer) in the snapshot
    const char* reason = nullptr;  // why there are no stats
};

struct Input {
    ID3D12Resource* resource;
    u64 id;
};

Write make_write(ID3D12Resource* resource, DXGI_FORMAT view, UINT mip = 0, UINT slice = 0, u64 offset = 0, u64 bytes = 0)
{
    Write w;
    w.resource = resource;
    w.id = creation_id(resource);
    w.view = view;
    if (!resource) return w;
    const auto desc = resource->GetDesc();
    w.format = desc.Format;
    w.buffer = desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
    w.mip = w.buffer ? 0 : std::min<UINT>(mip, desc.MipLevels ? desc.MipLevels - 1u : 0u);
    w.slice = w.buffer ? 0 : slice;
    w.width = w.buffer ? desc.Width : std::max<UINT64>(1, desc.Width >> w.mip);
    w.height = w.buffer ? 1 : std::max<UINT>(1, desc.Height >> w.mip);
    w.offset = offset;
    w.bytes = bytes;
    return w;
}

struct Record {
    Kind kind = Kind::graphics;
    std::vector<Write> writes;
    std::vector<Input> inputs;
    bool inputs_known = true;  // draws read what their tables name, which is not followed: false for graphics
    bool clear_only = false;
    bool snapped = false;
    std::vector<u64> keys;
    LONG64 first_call = -1, last_call = -1;
    unsigned calls = 0, refused = 0;
};

struct Recording {
    std::vector<Record> passes;
    bool open = false;           // passes.back() still takes writes
    unsigned carry_refused = 0;  // refused dispatches seen outside a compute pass, given to the next pass
};

bool same_writes(const std::vector<Write>& a, const std::vector<Write>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].resource != b[i].resource || a[i].mip != b[i].mip || a[i].slice != b[i].slice) return false;
    return true;
}

bool writes_resource(const std::vector<Write>& writes, ID3D12Resource* resource)
{
    return std::any_of(writes.begin(), writes.end(), [=](const Write& w) { return w.resource == resource; });
}

// Every resource of `part` is in `whole`: a clear followed by draws or a dispatch on those resources is one pass.
bool covers(const std::vector<Write>& whole, const std::vector<Write>& part)
{
    return !part.empty() && std::all_of(part.begin(), part.end(), [&](const Write& w) { return writes_resource(whole, w.resource); });
}

std::vector<Write> targets_of(const Pass& pass)
{
    std::vector<Write> writes;
    if (!pass.targets_bound) return writes;
    for (UINT i = 0; i < pass.target_count && i < 8; ++i)
        if (pass.bound_targets[i])
            writes.push_back(make_write(pass.bound_targets[i], pass.target_formats[i], pass.bound_target_ranges[i].first_mip,
                                        pass.bound_target_ranges[i].first_slice));
    if (pass.bound_depth)
        writes.push_back(make_write(pass.bound_depth, pass.dsv_format, pass.bound_depth_view.range.first_mip,
                                    pass.bound_depth_view.range.first_slice));
    return writes;
}

void barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, UINT subresource, D3D12_RESOURCE_STATES from,
             D3D12_RESOURCE_STATES to)
{
    if (from == to) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, subresource, from, to};
    list->ResourceBarrier(1, &b);
}

// One reduction per snapshot: its result words start at `word` of the submission's result buffer.
struct Job {
    std::size_t report, write;
    unsigned groups;
    u64 word;
};

struct Report {
    Record record;  // a copy: its snapshots stay referenced until the present reads them
};

struct Submission {
    LONG present = 0;
    ComPtr<ID3D12Fence> fence;
    u64 value = 0;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12Resource> result, readback, zeros;  // zeros: the upload copy that clears result
    std::vector<Report> reports;
    std::vector<Job> jobs;
    u64 words = 0;
    const char* failure = nullptr;  // the reduction was not submitted, or its result is not readable
};

struct Pipelines {
    bool tried = false;
    const char* failure = nullptr;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> kinds[4];
};

class Tracker {
public:
    void draw(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key);
    void not_drawn(const Pass& pass, ID3D12GraphicsCommandList* list, u64 key);
    void refused(ID3D12GraphicsCommandList* list);
    void dispatch(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key, std::vector<Write> outputs, std::vector<Input> inputs);
    void copy(Wrapper* self, ID3D12GraphicsCommandList* list, Write destination, ID3D12Resource* source);
    void clear(Wrapper* self, ID3D12GraphicsCommandList* list, ID3D12Resource* target, DXGI_FORMAT view);
    void close(Wrapper* self, ID3D12GraphicsCommandList* list);
    void discard(ID3D12GraphicsCommandList* list);
    void execute(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists);
    void present(LONG frame, ID3D12Resource* surface);

private:
    Record& begin(Recording& r, Kind kind, std::vector<Write> writes);
    void touch(Record& record, u64 key);
    void snap_closed(Pass& pass, ID3D12GraphicsCommandList* list, Recording& r);
    void snapshot(Pass& pass, ID3D12GraphicsCommandList* list, Write& w);
    void build(ID3D12CommandQueue* queue, Submission& s);
    bool record_reductions(Submission& s, ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type);
    const char* ensure_pipelines(ID3D12Device* device);
    std::vector<std::uint32_t> read(Submission& s);
    std::string line(LONG frame, int index, const Report& report, const Submission& s, const std::vector<std::uint32_t>& words,
                     m::PassView& view);
    void report_collapse(LONG frame, ID3D12Resource* surface, const std::vector<m::PassView>& views,
                         const std::vector<const Record*>& records);
    void write_lines(const std::string& text);

    std::mutex lock_;
    std::unordered_map<ID3D12GraphicsCommandList*, Recording> recordings_;
    std::vector<Submission> submissions_;
    std::unordered_map<ID3D12CommandQueue*, std::pair<ComPtr<ID3D12Fence>, u64>> fences_;
    Pipelines pipelines_;
    unsigned passes_this_present_ = 0;
    // Hook calls since the last present: draws, not drawn, dispatches, copies, clears, closes, executed lists, of them tracked.
    enum Count { draws, not_drawn_calls, dispatches, copies, clears, closes, executed, tracked, count_kinds };
    unsigned counts_[count_kinds]{};
    u64 bytes_this_present_ = 0;
    std::FILE* file_ = nullptr;
    bool file_tried_ = false;
};

Record& Tracker::begin(Recording& r, Kind kind, std::vector<Write> writes)
{
    r.open = true;
    r.passes.emplace_back();
    Record& record = r.passes.back();
    record.kind = kind;
    record.writes = std::move(writes);
    record.inputs_known = kind != Kind::graphics;
    record.clear_only = kind == Kind::clear;
    record.refused = r.carry_refused;
    r.carry_refused = 0;
    return record;
}

void Tracker::touch(Record& record, u64 key)
{
    const LONG64 call = InterlockedCompareExchange64(&pass_stats_call, 0, 0);
    if (record.first_call < 0) record.first_call = call;
    record.last_call = call;
    ++record.calls;
    if (key && record.keys.size() < 16 && std::find(record.keys.begin(), record.keys.end(), key) == record.keys.end())
        record.keys.push_back(key);
}

// Records the snapshots of the passes that closed since the last write on this list. Called before a write is
// recorded (a refused draw writes nothing, so its pass may close without a list to copy on).
void Tracker::snap_closed(Pass& pass, ID3D12GraphicsCommandList* list, Recording& r)
{
    for (std::size_t i = 0; i < r.passes.size(); ++i) {
        Record& record = r.passes[i];
        if (record.snapped || (r.open && i + 1 == r.passes.size())) continue;
        record.snapped = true;
        if (passes_this_present_ >= config().cap) {
            for (Write& w : record.writes) w.reason = "pass cap";
            continue;
        }
        ++passes_this_present_;
        for (Write& w : record.writes) snapshot(pass, list, w);
    }
}

void Tracker::snapshot(Pass& pass, ID3D12GraphicsCommandList* list, Write& w)
{
    const auto type = list->GetType();
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE) {
        w.reason = "list type";
        return;
    }
    if (!w.resource) {
        w.reason = "no resource";
        return;
    }
    const auto desc = w.resource->GetDesc();
    D3D12_RESOURCE_DESC snap{};
    u64 copy_bytes = 0;
    UINT subresource = 0;
    if (w.buffer) {
        const u64 end = w.bytes ? std::min<u64>(desc.Width, w.offset + w.bytes) : desc.Width;
        copy_bytes = (end > w.offset ? std::min(end - w.offset, config().buffer_limit) : 0) & ~u64{3};
        if (!copy_bytes) {
            w.reason = "empty range";
            return;
        }
        snap = {D3D12_RESOURCE_DIMENSION_BUFFER, 0, copy_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        w.info = {m::Sample::words, 1, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_UNKNOWN, false};
        w.texels = copy_bytes / 4;
    } else {
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) w.reason = "texture dimension";
        else if (desc.SampleDesc.Count != 1) w.reason = "multisampled";
        else if (w.slice >= desc.DepthOrArraySize) w.reason = "array slice";
        w.info = m::format_info(desc.Format, w.view, (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0);
        if (!w.reason && w.info.sample == m::Sample::none) w.reason = "format";
        if (w.reason) return;
        snap = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, w.width, w.height, 1, 1, w.info.snapshot, {1, 0},
                D3D12_TEXTURE_LAYOUT_UNKNOWN, w.info.depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_NONE};
        w.texels = w.width * w.height;
        subresource = w.mip + w.slice * desc.MipLevels;  // plane 0
    }
    ID3D12Device* device = device_of(the_device);
    const u64 size = device->GetResourceAllocationInfo(0, 1, &snap).SizeInBytes;
    if (size == UINT64_MAX || bytes_this_present_ + size > config().budget) {
        w.reason = "memory budget";
        return;
    }
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &snap, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&w.snapshot)))) {
        w.reason = "snapshot allocation";
        return;
    }
    bytes_this_present_ += size;
    // The state this list left the subresource in, from the shim's own tracking; it is restored after the copy.
    D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_COMMON;
    const auto split = find_split(pass, w.resource);
    if (split != pass.split.end()) {
        if (subresource < split->state.size()) before = split->state[subresource];
    } else if (const auto known = find_state(pass, w.resource); known != pass.states.end()) {
        before = known->second;
    }
    const UINT source_sub = w.buffer ? D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES : subresource;
    barrier(list, w.resource, source_sub, before, D3D12_RESOURCE_STATE_COPY_SOURCE);
    barrier(list, w.snapshot.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST);
    if (w.buffer) {
        list->CopyBufferRegion(w.snapshot.Get(), 0, w.resource, w.offset, copy_bytes);
    } else {
        D3D12_TEXTURE_COPY_LOCATION to{w.snapshot.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION from{w.resource, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        to.SubresourceIndex = 0;
        from.SubresourceIndex = subresource;
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    barrier(list, w.snapshot.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COMMON);
    barrier(list, w.resource, source_sub, D3D12_RESOURCE_STATE_COPY_SOURCE, before);
}

void Tracker::draw(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key)
{
    Pass& pass = pass_of(self);
    std::vector<Write> writes = targets_of(pass);
    std::lock_guard<std::mutex> guard(lock_);
    ++counts_[draws];
    Recording& r = recordings_[list];
    Record* open = r.open ? &r.passes.back() : nullptr;
    if (open && open->kind == Kind::clear && covers(writes, open->writes)) {
        open->kind = Kind::graphics;
        open->writes = std::move(writes);
        open->clear_only = false;
        open->inputs_known = false;
    } else if (!open || open->kind != Kind::graphics || !same_writes(open->writes, writes)) {
        r.open = false;
        snap_closed(pass, list, r);
        begin(r, Kind::graphics, std::move(writes));
    }
    snap_closed(pass, list, r);
    touch(r.passes.back(), key);
}

void Tracker::not_drawn(const Pass& pass, ID3D12GraphicsCommandList* list, u64 key)
{
    std::vector<Write> writes = targets_of(pass);
    std::lock_guard<std::mutex> guard(lock_);
    ++counts_[not_drawn_calls];
    Recording& r = recordings_[list];
    Record* open = r.open ? &r.passes.back() : nullptr;
    if (open && open->kind == Kind::clear && covers(writes, open->writes)) {
        open->kind = Kind::graphics;
        open->writes = std::move(writes);
        open->clear_only = false;
        open->inputs_known = false;
    } else if (!open || open->kind != Kind::graphics || !same_writes(open->writes, writes)) {
        r.open = false;  // its snapshots are recorded before the next write on this list
        begin(r, Kind::graphics, std::move(writes));
    }
    Record& record = r.passes.back();
    touch(record, key);
    ++record.refused;
}

void Tracker::refused(ID3D12GraphicsCommandList* list)
{
    std::lock_guard<std::mutex> guard(lock_);
    Recording& r = recordings_[list];
    if (r.open && r.passes.back().kind == Kind::compute) ++r.passes.back().refused;
    else ++r.carry_refused;
}

void Tracker::dispatch(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key, std::vector<Write> outputs,
                       std::vector<Input> inputs)
{
    if (outputs.empty()) return;  // a dispatch that writes nothing ends no pass
    Pass& pass = pass_of(self);
    std::lock_guard<std::mutex> guard(lock_);
    ++counts_[dispatches];
    Recording& r = recordings_[list];
    Record* open = r.open ? &r.passes.back() : nullptr;
    if (open && open->kind == Kind::clear && covers(outputs, open->writes)) {
        open->kind = Kind::compute;
        open->writes = std::move(outputs);
        open->clear_only = false;
    } else if (!open || open->kind != Kind::compute || !same_writes(open->writes, outputs)) {
        r.open = false;
        snap_closed(pass, list, r);
        begin(r, Kind::compute, std::move(outputs));
    }
    snap_closed(pass, list, r);
    Record& record = r.passes.back();
    for (const Input& in : inputs)
        if (std::none_of(record.inputs.begin(), record.inputs.end(), [&](const Input& known) { return known.resource == in.resource; }))
            record.inputs.push_back(in);
    touch(record, key);
}

void Tracker::copy(Wrapper* self, ID3D12GraphicsCommandList* list, Write destination, ID3D12Resource* source)
{
    Pass& pass = pass_of(self);
    std::lock_guard<std::mutex> guard(lock_);
    ++counts_[copies];
    Recording& r = recordings_[list];
    Record* open = r.open ? &r.passes.back() : nullptr;
    if (!open || open->kind != Kind::copy) {
        r.open = false;
        snap_closed(pass, list, r);
        begin(r, Kind::copy, {});
    }
    snap_closed(pass, list, r);
    Record& record = r.passes.back();
    const auto known = std::find_if(record.writes.begin(), record.writes.end(),
                                    [&](const Write& w) { return w.resource == destination.resource && w.mip == destination.mip; });
    if (known == record.writes.end()) {
        record.writes.push_back(std::move(destination));
    } else if (known->buffer && known->bytes && destination.bytes) {  // one range over both buffer copies
        const u64 end = std::max(known->offset + known->bytes, destination.offset + destination.bytes);
        known->offset = std::min(known->offset, destination.offset);
        known->bytes = end - known->offset;
    } else {
        known->offset = 0;
        known->bytes = 0;
    }
    if (source && std::none_of(record.inputs.begin(), record.inputs.end(), [&](const Input& in) { return in.resource == source; }))
        record.inputs.push_back({source, creation_id(source)});
    touch(record, 0);
}

void Tracker::clear(Wrapper* self, ID3D12GraphicsCommandList* list, ID3D12Resource* target, DXGI_FORMAT view)
{
    Pass& pass = pass_of(self);
    std::lock_guard<std::mutex> guard(lock_);
    ++counts_[clears];
    Recording& r = recordings_[list];
    Record* open = r.open ? &r.passes.back() : nullptr;
    if (open && open->kind != Kind::copy && writes_resource(open->writes, target)) {
        // A clear of what the open pass writes is part of that pass.
    } else if (open && open->kind == Kind::clear) {
        open->writes.push_back(make_write(target, view));
    } else {
        r.open = false;
        snap_closed(pass, list, r);
        begin(r, Kind::clear, {make_write(target, view)});
    }
    snap_closed(pass, list, r);
    touch(r.passes.back(), 0);
}

void Tracker::close(Wrapper* self, ID3D12GraphicsCommandList* list)
{
    std::lock_guard<std::mutex> guard(lock_);
    ++counts_[closes];
    const auto found = recordings_.find(list);
    if (found == recordings_.end()) return;
    found->second.open = false;
    snap_closed(pass_of(self), list, found->second);
}

void Tracker::discard(ID3D12GraphicsCommandList* list)
{
    std::lock_guard<std::mutex> guard(lock_);
    recordings_.erase(list);
}

#include "startup_stubs/pass_stats_submit.inl"

}  // namespace pass_stats

pass_stats::Tracker& pass_stats_tracker()
{
    static auto* const tracker = new pass_stats::Tracker;  // never destroyed: queued reductions hold its objects
    return *tracker;
}

// Hooks. Each is a no-op without X2W_PASS_STATS=1; a write hook runs before its write is recorded.
void pass_stats_draw(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key)
{
    if (pass_stats_enabled() && list) pass_stats_tracker().draw(self, list, key);
}
void pass_stats_not_drawn(const Pass& pass, ID3D12GraphicsCommandList* list, u64 key)
{
    if (pass_stats_enabled() && list) pass_stats_tracker().not_drawn(pass, list, key);
}
void pass_stats_refused(Wrapper* self)
{
    if (pass_stats_enabled() && recording(self)) pass_stats_tracker().refused(recording(self));
}
void pass_stats_clear(Wrapper* self, ID3D12Resource* target, DXGI_FORMAT view)
{
    if (pass_stats_enabled() && recording(self) && target) pass_stats_tracker().clear(self, recording(self), target, view);
}
// A copy to a resource that a render target or unordered-access view can write (replay_dump::note_copy's rule).
void pass_stats_copy(Wrapper* self, ID3D12Resource* destination, ID3D12Resource* source, UINT subresource = 0,
                     u64 offset = 0, u64 bytes = 0)
{
    if (!pass_stats_enabled() || !recording(self) || !destination) return;
    const auto desc = destination->GetDesc();
    if (!(desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))) return;
    const UINT mips = desc.MipLevels ? desc.MipLevels : 1;
    pass_stats_tracker().copy(self, recording(self),
                              pass_stats::make_write(destination, DXGI_FORMAT_UNKNOWN, subresource % mips, subresource / mips,
                                                     offset, bytes),
                              source);
}
void pass_stats_close(Wrapper* self)
{
    if (pass_stats_enabled() && recording(self)) pass_stats_tracker().close(self, recording(self));
}
void pass_stats_discard(ID3D12GraphicsCommandList* list)
{
    if (pass_stats_enabled() && list) pass_stats_tracker().discard(list);
}
void pass_stats_execute(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists)
{
    if (pass_stats_enabled()) pass_stats_tracker().execute(queue, lists);
}
void pass_stats_present(LONG frame, Wrapper* surface)
{
    if (!pass_stats_enabled()) return;
    auto* resource = surface && surface->own().kind == resource_kind ? static_cast<ID3D12Resource*>(surface->own().real) : nullptr;
    pass_stats_tracker().present(frame, resource);
}

// A translated Dispatch about to be recorded (record_translated_dispatch): views [0, inputs) are read, the rest written.
template <class View>
void pass_stats_translated_dispatch(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key, const View* views,
                                    std::size_t inputs, std::size_t count)
{
    if (!pass_stats_enabled() || !list) return;
    std::vector<pass_stats::Input> read;
    std::vector<pass_stats::Write> written;
    for (std::size_t i = 0; i < count; ++i) {
        ID3D12Resource* resource = views[i].resource;
        if (!resource) continue;
        if (i < inputs) {
            read.push_back({resource, pass_stats::creation_id(resource)});
            continue;
        }
        if (!views[i].texture) {
            written.push_back(pass_stats::make_write(resource, DXGI_FORMAT_UNKNOWN, 0, 0, views[i].offset, views[i].bytes));
            continue;
        }
        const D3D12_UNORDERED_ACCESS_VIEW_DESC* uav = views[i].described ? &views[i].desc.uav : nullptr;
        UINT mip = 0, slice = 0;
        if (uav && uav->ViewDimension == D3D12_UAV_DIMENSION_TEXTURE2D) mip = uav->Texture2D.MipSlice;
        if (uav && uav->ViewDimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY)
            mip = uav->Texture2DArray.MipSlice, slice = uav->Texture2DArray.FirstArraySlice;
        written.push_back(pass_stats::make_write(resource, uav ? uav->Format : DXGI_FORMAT_UNKNOWN, mip, slice));
    }
    pass_stats_tracker().dispatch(self, list, key, std::move(written), std::move(read));
}

extern "C" __declspec(dllexport) void x2w_d3d12_pass_stats_call(unsigned long long call)
{
    InterlockedExchange64(&pass_stats_call, static_cast<LONG64>(call));
}
