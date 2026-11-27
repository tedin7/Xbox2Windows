// X2W_DEBUG_PRESENT_TARGET = depth | largest-rt | rt:<W>x<H>:<fmt> (docs/startup-runtime.md#debug-present).
// Opt-in. At every title Present the picture the title presents is replaced, before anything reads it (the picture
// signature, X2W_PRESENT_PICTURES, the capture, the output window), by a picture of the chosen target of the last frame:
//   depth       the biggest depth buffer a pass bound, as grey (debug_present_convert.h: reversed-Z aware, near is bright)
//   largest-rt  the biggest colour target a pass bound, tone-mapped to 8 bits
//   rt:WxH:fmt  the colour target (or depth buffer) of exactly that size and DXGI format
// The target is stretched to the presented surface's size. Targets are found where a pass binds them (OMSetRenderTargets),
// so a clear alone shows too. The copy runs on the presenting queue and is waited for: a diagnostic, a GPU round trip.
// A resource is AddRef'd while it is a candidate, so the title releasing it cannot free it under the copy.
// Cost control: X2W_DEBUG_PRESENT_FROM=<seconds> does nothing before that process time; the picture is made only on every
// X2W_DEBUG_PRESENT_EVERY-th present (default 6, 1 = every present) and on presents that a timed picture is due on; a 4K
// target is sampled at every 4th pixel (about 1/16 of the work). `depth` and `largest-rt` prefer the candidate whose size
// equals the presented surface's, then the biggest, so a shadow atlas does not hide the scene depth.
#include "startup_stubs/debug_present_convert.h"

namespace debug_present {

struct Tracker {
    static constexpr std::size_t max_entries = 8;
    struct Entry {
        ID3D12Resource* resource = nullptr;  // holds a reference
        x2w_debug_present::Candidate info;
        unsigned sequence = 0;
        unsigned long long draws = 0;  // translated draws that wrote it since the last take
    };
    SRWLOCK lock = SRWLOCK_INIT;
    x2w_debug_present::Spec spec;
    std::vector<Entry> entries;  // distinct targets bound since the last take
    Entry kept;                  // the last choice, shown again when no pass bound a target since
    unsigned counter = 0;
    unsigned long long total_draws = 0;  // every translated draw seen while enabled (X2W_DEBUG_PRESENT_FROM_DRAWS)

    explicit Tracker(const x2w_debug_present::Spec& s = {}) : spec(s) {}
    Tracker(const Tracker&) = delete;
    Tracker& operator=(const Tracker&) = delete;
    ~Tracker()
    {
        for (Entry& e : entries) if (e.resource) e.resource->Release();
        if (kept.resource) kept.resource->Release();
    }

    bool enabled() const { return spec.mode != x2w_debug_present::Mode::off; }

    // A pass bound `resource` (a colour target, or the depth buffer when `depth`).
    void note(ID3D12Resource* resource, unsigned long long width, unsigned height, int format, bool depth)
    {
        if (!enabled() || !resource) return;
        const x2w_debug_present::Candidate next{width, height, format, depth};
        if (!x2w_debug_present::wants(spec, {}, next)) return;  // not eligible under the setting
        AcquireSRWLockExclusive(&lock);
        Entry* slot = nullptr;
        for (Entry& e : entries)
            if (e.resource == resource) slot = &e;
        if (!slot && entries.size() < max_entries) {
            entries.emplace_back();
            slot = &entries.back();
        } else if (!slot) {
            // Full: replace the smallest entry if this one is bigger.
            Entry* smallest = &entries[0];
            for (Entry& e : entries)
                if (e.draws < smallest->draws || (e.draws == smallest->draws && e.info.area() < smallest->info.area())) smallest = &e;
            if (smallest->draws == 0 && next.area() > smallest->info.area()) {
                smallest->resource->Release();
                *smallest = Entry{};
                slot = smallest;
            }
        }
        if (slot) {
            if (slot->resource != resource) {
                resource->AddRef();
                slot->resource = resource;
            }
            slot->info = next;
            slot->sequence = ++counter;
        }
        ReleaseSRWLockExclusive(&lock);
    }

    // A translated draw wrote these targets: count it for each one that is a candidate.
    void note_draw(ID3D12Resource* const* targets, UINT count, ID3D12Resource* depth)
    {
        if (!enabled()) return;
        AcquireSRWLockExclusive(&lock);
        ++total_draws;
        for (Entry& e : entries) {
            if (!e.resource) continue;
            bool hit = e.resource == depth;
            for (UINT i = 0; i < count && !hit; ++i) hit = e.resource == targets[i];
            if (hit) ++e.draws;
        }
        ReleaseSRWLockExclusive(&lock);
    }

    unsigned long long draws_seen()
    {
        AcquireSRWLockShared(&lock);
        const unsigned long long total = total_draws;
        ReleaseSRWLockShared(&lock);
        return total;
    }

    // The choice with its own reference, or null. The most draws in the interval wins; then the entry whose size equals
    // (want_w x want_h), then the biggest, then the latest. The entries are cleared; the choice is kept for an interval
    // with no new binds.
    ID3D12Resource* take(unsigned long long want_w, unsigned want_h, x2w_debug_present::Candidate& out,
                         unsigned long long* draws_out = nullptr)
    {
        AcquireSRWLockExclusive(&lock);
        Entry* best = nullptr;
        for (Entry& e : entries) {
            const bool match = e.info.width == want_w && e.info.height == want_h;
            const bool best_match = best && best->info.width == want_w && best->info.height == want_h;
            bool better = !best || e.draws > best->draws;
            if (best && e.draws == best->draws)
                better = (match && !best_match) ||
                         (match == best_match && (e.info.area() > best->info.area() ||
                                                  (e.info.area() == best->info.area() && e.sequence > best->sequence)));
            if (better) best = &e;
        }
        if (best) {
            if (kept.resource) kept.resource->Release();
            kept = *best;
            best->resource = nullptr;  // the reference moved to `kept`
        }
        for (Entry& e : entries) if (e.resource) e.resource->Release();
        entries.clear();
        ID3D12Resource* resource = kept.resource;
        out = kept.info;
        if (draws_out) *draws_out = kept.draws;
        if (resource) resource->AddRef();
        ReleaseSRWLockExclusive(&lock);
        return resource;
    }
};

// Copy objects for the readback of the chosen target (the readback buffer grows to the biggest target).
struct Copier {
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE event = nullptr;
    ID3D12Resource* readback = nullptr;
    UINT64 readback_bytes = 0, fence_value = 0;
    bool failed = false;

    bool setup(ID3D12Device* device, UINT64 bytes)
    {
        if (!allocator && (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
                           FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list))) ||
                           FAILED(list->Close()) || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) ||
                           !(event = CreateEventW(nullptr, FALSE, FALSE, nullptr))))
            return false;
        if (readback_bytes >= bytes) return true;
        if (readback) readback->Release();
        readback = nullptr;
        readback_bytes = 0;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                       D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&readback))))
            return false;
        x2w_budget::committed(device, readback, D3D12_HEAP_TYPE_READBACK);
        readback_bytes = bytes;
        return true;
    }
};

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &barrier);
}

struct Painter {
    Copier copier;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Resource* upload = nullptr;
    UINT64 upload_bytes = 0;
    std::vector<unsigned char> rgba;
    x2w_debug_present::DepthStats stats;  // of the last depth picture

    bool setup(ID3D12Device* device, UINT64 bytes)
    {
        if (!allocator && (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
                           FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list))) ||
                           FAILED(list->Close())))
            return false;
        if (upload_bytes >= bytes) return true;
        if (upload) upload->Release();
        upload = nullptr;
        upload_bytes = 0;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                       D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                   IID_PPV_ARGS(&upload))))
            return false;
        x2w_budget::committed(device, upload, D3D12_HEAP_TYPE_UPLOAD);
        upload_bytes = bytes;
        return true;
    }
};

// Submits the list the painter holds on `queue` and waits for it.
bool submit_and_wait(Copier& c, ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* list)
{
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(c.fence, ++c.fence_value)) || FAILED(c.fence->SetEventOnCompletion(c.fence_value, c.event)) ||
        WaitForSingleObject(c.event, 10000) != WAIT_OBJECT_0) {
        c.failed = true;  // may still be running: never reuse these objects
        return false;
    }
    return true;
}

// The row of `rgba` as the presented surface stores it.
// One output row from a source row of `source_width` RGBA8 pixels, stretched to `width` pixels by nearest neighbour, in the
// surface's layout.
void store_row(const unsigned char* in, UINT source_width, unsigned char* out, UINT width, x2w_present_signature::Layout layout)
{
    for (UINT x = 0; x < width; ++x, out += 4) {
        const unsigned char* px = in + static_cast<std::size_t>(static_cast<unsigned long long>(x) * source_width / width) * 4;
        if (layout == x2w_present_signature::Layout::bgra8) {
            out[0] = px[2], out[1] = px[1], out[2] = px[0], out[3] = 255;
        } else if (layout == x2w_present_signature::Layout::rgb10a2) {
            const std::uint32_t v = (std::uint32_t{px[0]} << 2) | (std::uint32_t{px[1]} << 12) | (std::uint32_t{px[2]} << 22) | (3u << 30);
            std::memcpy(out, &v, 4);
        } else {
            out[0] = px[0], out[1] = px[1], out[2] = px[2], out[3] = 255;
        }
    }
}

// Copies `source` (a target in COMMON state) to the readback, converts it and writes the picture into `surface` (COMMON).
bool show(Painter& p, ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Resource* source,
          const x2w_debug_present::Candidate& info, ID3D12Resource* surface, const char*& reason)
{
    const D3D12_RESOURCE_DESC sdesc = source->GetDesc(), tdesc = surface->GetDesc();
    x2w_present_signature::Layout layout;
    if (sdesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || sdesc.SampleDesc.Count != 1 ||
        !x2w_debug_present::plane_bytes(info.format) || sdesc.Width > 16384 || sdesc.Height > 16384) {
        reason = "the chosen target's format, size or sample count is not read";
        return false;
    }
    if (tdesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !x2w_present_signature::layout_of(tdesc.Format, layout) ||
        tdesc.Width > 16384 || tdesc.Height > 16384) {
        reason = "the presented surface's format or size is not written";
        return false;
    }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT from_fp{}, to_fp{};
    UINT64 from_total = 0, to_total = 0;
    device->GetCopyableFootprints(&sdesc, 0, 1, 0, &from_fp, nullptr, nullptr, &from_total);
    device->GetCopyableFootprints(&tdesc, 0, 1, 0, &to_fp, nullptr, nullptr, &to_total);
    auto& c = p.copier;
    if (c.failed || !c.setup(device, from_total) || !p.setup(device, to_total) || FAILED(c.allocator->Reset()) ||
        FAILED(c.list->Reset(c.allocator, nullptr))) {
        reason = "copy objects could not be made, or an earlier copy did not finish";
        return false;
    }
    transition(c.list, source, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, to{c.readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    to.PlacedFootprint = from_fp;
    c.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    transition(c.list, source, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    if (!submit_and_wait(c, queue, c.list)) {
        reason = "the copy of the chosen target did not finish in 10 s";
        return false;
    }
    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(from_total)};
    if (FAILED(c.readback->Map(0, &range, &mapped)) || !mapped) {
        reason = "the readback did not map";
        return false;
    }
    unsigned ow = 0, oh = 0;
    p.stats = {};
    const bool read = x2w_debug_present::to_rgba(static_cast<const unsigned char*>(mapped) + from_fp.Offset, from_fp.Footprint.RowPitch,
                                                 static_cast<UINT>(sdesc.Width), sdesc.Height,
                                                 x2w_debug_present::step_for(static_cast<unsigned>(sdesc.Width)), info.format, p.rgba, ow, oh, &p.stats);
    const D3D12_RANGE none{0, 0};
    c.readback->Unmap(0, &none);
    if (!read) {
        reason = "the chosen target's format is not read";
        return false;
    }
    const auto tw = static_cast<UINT>(tdesc.Width), th = tdesc.Height;
    void* up = nullptr;
    const D3D12_RANGE write_range{0, static_cast<SIZE_T>(to_total)};
    if (FAILED(p.upload->Map(0, &none, &up)) || !up) {
        reason = "the upload buffer did not map";
        return false;
    }
    // Nearest-neighbour stretch of the ow x oh picture, row by row straight into the upload buffer.
    for (UINT y = 0; y < th; ++y)
        store_row(&p.rgba[static_cast<std::size_t>(static_cast<unsigned long long>(y) * oh / th) * ow * 4], ow,
                  static_cast<unsigned char*>(up) + to_fp.Offset + static_cast<UINT64>(y) * to_fp.Footprint.RowPitch, tw, layout);
    p.upload->Unmap(0, &write_range);
    if (FAILED(p.allocator->Reset()) || FAILED(p.list->Reset(p.allocator, nullptr))) {
        reason = "the upload list did not reset";
        return false;
    }
    transition(p.list, surface, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst{surface, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, src{p.upload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    src.PlacedFootprint = to_fp;
    p.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    transition(p.list, surface, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    if (!submit_and_wait(c, queue, p.list)) {
        reason = "the picture copy into the presented surface did not finish in 10 s";
        return false;
    }
    return true;
}

Tracker& global_tracker()
{
    static Tracker tracker([] {
        char value[64]{};
        const DWORD n = GetEnvironmentVariableA("X2W_DEBUG_PRESENT_TARGET", value, sizeof(value));
        const x2w_debug_present::Spec spec = n && n < sizeof(value) ? x2w_debug_present::parse_spec(value) : x2w_debug_present::Spec{};
        if (n && spec.mode == x2w_debug_present::Mode::off)
            log("x2w-render debug present: X2W_DEBUG_PRESENT_TARGET=%s is not depth, largest-rt or rt:<W>x<H>:<fmt>; off", value);
        else if (spec.mode != x2w_debug_present::Mode::off)
            log("x2w-render debug present: on, target %s", value);
        return spec;
    }());
    return tracker;
}

// What the picture shows, for the log line: "depth 2560x1440 fmt 20".
std::string describe(const x2w_debug_present::Candidate& info)
{
    return std::string(info.depth ? "depth " : "colour ") + std::to_string(info.width) + "x" + std::to_string(info.height) +
           " fmt " + std::to_string(info.format);
}

// A pass bound these targets.
void note_bound(ID3D12Resource* const* targets, const x2w_startup::TargetDrawCensus::Surface* surfaces, UINT count,
                ID3D12Resource* depth, const x2w_startup::TargetDrawCensus::Surface& depth_surface)
{
    Tracker& tracker = global_tracker();
    if (!tracker.enabled()) return;
    for (UINT i = 0; i < count; ++i)
        tracker.note(targets[i], surfaces[i].width, surfaces[i].height, surfaces[i].format, false);
    if (depth) tracker.note(depth, depth_surface.width, depth_surface.height, depth_surface.format, true);
}

// A translated draw wrote the targets of the current pass.
void note_draw(ID3D12Resource* const* targets, UINT count, ID3D12Resource* depth)
{
    Tracker& tracker = global_tracker();
    if (tracker.enabled()) tracker.note_draw(targets, count, depth);
}

// The depth statistics line text: "spread 0.25..0.75 covered 75.0%".
std::string stats_text(const x2w_debug_present::DepthStats& stats)
{
    char text[96];
    std::snprintf(text, sizeof(text), "spread %.4g..%.4g covered %.1f%%", stats.low, stats.high, stats.covered_percent());
    return text;
}

// At a title Present, before the picture is read: replace plane 0's surface content by the chosen target. `picture_due`:
// a timed picture (the report's window pictures) is due on this present.
void present(Wrapper* queue, Wrapper* surface, LONG frame, bool picture_due)
{
    Tracker& tracker = global_tracker();
    if (!tracker.enabled()) return;
    static const double from_seconds = [] {
        char value[32]{};
        return GetEnvironmentVariableA("X2W_DEBUG_PRESENT_FROM", value, sizeof(value)) ? std::atof(value) : 0.0;
    }();
    static const LONG every = [] {
        char value[16]{};
        const long n = GetEnvironmentVariableA("X2W_DEBUG_PRESENT_EVERY", value, sizeof(value)) ? std::atol(value) : 6;
        return static_cast<LONG>(n > 0 ? n : 1);
    }();
    static const unsigned long long from_draws = [] {
        char value[32]{};
        return GetEnvironmentVariableA("X2W_DEBUG_PRESENT_FROM_DRAWS", value, sizeof(value)) ? std::strtoull(value, nullptr, 10) : 0ull;
    }();
    // Before FROM (seconds) or FROM_DRAWS (translated draws seen) nothing happens: no copy.
    if (static_cast<double>(GetTickCount64() - gray_t0) < from_seconds * 1000.0) return;
    static bool draws_reached;
    if (from_draws && !draws_reached) {
        if (tracker.draws_seen() < from_draws) return;
        draws_reached = true;
        log("x2w-render debug present: started at frame %ld, %llu translated draws seen", frame, tracker.draws_seen());
    }
    if (!picture_due && frame % every != 0) return;
    static SRWLOCK lock = SRWLOCK_INIT;
    static Painter painter;
    static LONG refused, shown;
    static ID3D12Resource* last_shown;
    static x2w_debug_present::Candidate last_info;
    x2w_debug_present::Candidate info;
    ID3D12Resource* real_surface = surface && surface->own().kind == resource_kind
                                       ? static_cast<ID3D12Resource*>(surface->own().real) : nullptr;
    const D3D12_RESOURCE_DESC surface_desc = real_surface ? real_surface->GetDesc() : D3D12_RESOURCE_DESC{};
    unsigned long long interval_draws = 0;
    ID3D12Resource* source = tracker.take(surface_desc.Width, surface_desc.Height, info, &interval_draws);
    const char* reason = !source ? "no pass has bound a matching target yet" : "no presented surface";
    bool ok = false;
    if (source && real_surface) {
        AcquireSRWLockExclusive(&lock);
        ok = show(painter, device_of(the_device), static_cast<ID3D12CommandQueue*>(queue->own().real), source, info, real_surface, reason);
        if (ok && (source != last_shown || info.width != last_info.width || info.format != last_info.format)) {
            log("x2w-render debug present: showing %s (presented surface %llux%u, %llu draws in the interval)", describe(info).c_str(),
                static_cast<unsigned long long>(surface_desc.Width), surface_desc.Height, interval_draws);
            last_shown = source;
            last_info = info;
        }
        // The automatic first silhouette: what the chosen depth holds, about every 10 s, so geometry shows without a picture.
        static ULONGLONG last_stats_tick;
        const ULONGLONG now_tick = GetTickCount64();
        if (ok && painter.stats.valid && now_tick - last_stats_tick >= 10000) {
            last_stats_tick = now_tick;
            log("x2w-render debug present: depth %s (%s, %llu draws in the interval, %zu of %zu sampled pixels are geometry%s)",
                stats_text(painter.stats).c_str(), describe(info).c_str(), interval_draws, painter.stats.geometry,
                painter.stats.sampled, painter.stats.reversed ? ", reversed Z" : "");
        }
        ReleaseSRWLockExclusive(&lock);
    }
    if (source) source->Release();
    if (ok) InterlockedIncrement(&shown);
    else if (first_calls(refused, 6)) log("x2w-render debug present: frame picture left as the title made it: %s", reason);
}

}  // namespace debug_present

// WARP fixture (tests/startup_debug_present.cpp): the production selection, copy and conversion on desktop objects. `spec`
// is the setting's text; the colour target and the two depth buffers are "bound" in that order, and `draws` / `draws2` translated
// draws are counted for the depth buffers. Returns 0 and the log text in `what` (and the depth statistics text in `stats`) when a
// picture was written, 1 when nothing matched, 2 when the picture could not be made (`what` has the reason).
extern "C" __declspec(dllexport) int x2w_d3d12_fixture_debug_present(ID3D12Device* device, ID3D12CommandQueue* queue, const char* spec,
        ID3D12Resource* color, int color_format, ID3D12Resource* depth, int depth_format, ID3D12Resource* depth2,
        unsigned draws, unsigned draws2, ID3D12Resource* surface, char* what, std::size_t capacity, char* stats, std::size_t stats_capacity)
{
    debug_present::Tracker tracker(x2w_debug_present::parse_spec(spec));
    const auto note = [&](ID3D12Resource* resource, int format, bool is_depth) {
        if (!resource) return;
        const D3D12_RESOURCE_DESC desc = resource->GetDesc();
        tracker.note(resource, desc.Width, desc.Height, format, is_depth);
    };
    note(color, color_format, false);
    note(depth, depth_format, true);
    note(depth2, depth_format, true);
    for (unsigned i = 0; i < draws; ++i) tracker.note_draw(nullptr, 0, depth);
    for (unsigned i = 0; i < draws2; ++i) tracker.note_draw(nullptr, 0, depth2);
    const D3D12_RESOURCE_DESC surface_desc = surface->GetDesc();
    x2w_debug_present::Candidate info;
    ID3D12Resource* source = tracker.take(surface_desc.Width, surface_desc.Height, info);
    if (stats && stats_capacity) stats[0] = 0;
    if (!source) {
        std::snprintf(what, capacity, "no match");
        return 1;
    }
    debug_present::Painter painter;
    const char* reason = "";
    const bool ok = debug_present::show(painter, device, queue, source, info, surface, reason);
    source->Release();
    std::snprintf(what, capacity, "%s", ok ? debug_present::describe(info).c_str() : reason);
    if (ok && painter.stats.valid && stats && stats_capacity)
        std::snprintf(stats, stats_capacity, "%s", debug_present::stats_text(painter.stats).c_str());
    return ok ? 0 : 2;
}
