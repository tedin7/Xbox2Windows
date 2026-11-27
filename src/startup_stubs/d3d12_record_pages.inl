// Memory pages for replay recordings (docs/startup-runtime.md#d3d12-replay-record), the page store of replay_pages.h:
// PAGE chunks (one per distinct non-zero 4 KiB page) and MAP chunks (which pages a range holds, only runs that changed).
// Included inside namespace replay_record by d3d12_record.inl; each hook appends to the caller's thread buffer `b` (its lock
// held by the caller). Replay applies a MAP before the first call with a larger sequence number.
//
// Memory = the resources that have title_memory and a known size (placed resources, hook allocations), merged and cut to
// whole pages. When the window opens one pass writes the base image of all of them; after each present a pass limited to
// X2W_REPLAY_SCAN_MS (default 250) continues round-robin and writes only pages that changed since this recording last saw
// them. By default a MAP read at present N is stamped with the mark of present N-1.
// X2W_REPLAY_PAGES_AT_SUBMIT=1 also scans before each submission inside the window.
// Draw pages capture root addresses at binding and again with vertex and index spans before draws and dispatches.
// X2W_REPLAY_DRAW_PAGES=0 disables these captures. X2W_REPLAY_DRAW_PAGES_MS limits their time per Present.
// X2W_REPLAY_TITLE_RANGES=1 adds named title memory and scans in call order before the stand-in reads it.
// The size cap is X2W_REPLAY_MAX_MIB as for calls. X2W_REPLAY_PAGES=0 leaves the pages out.
// One line of counters after the base image and about every 20th scan:
//   x2w-replay pages: ranges N (M MiB), pages seen S, distinct D, zero Z, unchanged U, written W MiB, skipped (no size) K,
//   unreadable R, scan T ms
constexpr std::size_t page_slab_bytes = 1u << 20;

struct PageRange { u64 address, bytes; };
// Draw capture state (pages_lock).
constexpr u64 draw_root_window = 0x10000;   // a root address has no size: the stand-in reads up to 64 KiB from it
constexpr u64 draw_span_limit = 4ull << 20;  // larger vertex/index spans are left to the page scans
constexpr std::size_t draw_pending_limit = 64;
std::unordered_map<u64, std::vector<PageRange>> draw_pending;  // list -> spans bound since its last draw or dispatch
std::vector<PageRange> draw_ranges;  // the title ranges by address
bool draw_ranges_missed;             // gathered again this Present for a span outside them
LONG draw_ranges_present = -1, draw_time_present = -1;
u64 draw_captures, draw_outside, draw_over_time, draw_pending_dropped;
double draw_time_used;
bool draw_time_logged;
std::vector<unsigned char> draw_read_buffer;
x2w_replay::PageWriter* page_writer;  // created lazily by the first page hook; never freed
bool page_writer_tried;
ThreadBuffer* page_target;            // the buffer the running scan appends to (its lock is held by the caller)
bool page_base_done;                  // the base image of all ranges was written (pages_lock)
u64 page_last_submit;                 // latest scanned ExecuteCommandLists call (pages_lock)
u64 page_prev_mark;                   // the previous present's mark, the stamp of the next scan
SRWLOCK pages_lock = SRWLOCK_INIT;   // one scan at a time; held for the whole scan
double page_scan_ms = 250.0;
// The pages have a budget of their own (X2W_REPLAY_PAGES_MAX_MIB, default half of X2W_REPLAY_MAX_MIB): when it is spent no more
// pages are written but the calls go on. Resources over X2W_REPLAY_PAGES_MAX_RANGE_MIB (default 8) are left out (counted), and
// the smaller ranges come first: constants, vertex and index data, small textures before the big ones.
u64 page_budget_bytes, page_range_limit = 8ull << 20, page_bytes_out, page_skipped_big;
bool page_budget_reached;
// The base image may use 3/4 of the budget: the rest keeps the window's changed pages (a per-frame descriptor ring,
// constants), without which the window replays from stale memory. When the base image stops there, the range it stopped
// in and the ones after it in scan order (by size, then address) stay out of the later scans too.
bool page_base_cut;
u64 page_cut_bytes, page_cut_address;  // the first range the base image did not finish
bool page_after_cut(const PageRange& r)
{
    return page_base_cut && (r.bytes > page_cut_bytes || (r.bytes == page_cut_bytes && r.address >= page_cut_address));
}
std::vector<PageRange> page_ranges;   // pages_lock
std::size_t page_cursor_range;        // pages_lock
u64 page_cursor_offset;               // pages_lock
u64 page_unreadable, page_skipped_no_size, page_scans;
// The base image written before the window (base_image_holds_window): at most X2W_REPLAY_BASE_SCAN_MS (default 500) per
// present, and where it has got to: a range by its size and address (the scan order) and the offset in it, so ranges that
// come or go between two slices do not move it.
double base_scan_ms = 500.0;
u64 base_cursor_bytes, base_cursor_address, base_cursor_offset;  // pages_lock
double base_started_s;      // present thread: when the window's time came
LONG base_started_present;  // and at which present
u64 base_slices;            // pages_lock: slices written
bool page_scan_order(const PageRange& a, const PageRange& b)
{
    return a.bytes != b.bytes ? a.bytes < b.bytes : a.address < b.address;
}
double page_ms_since(const LARGE_INTEGER& t0)
{
    LARGE_INTEGER t1;
    QueryPerformanceCounter(&t1);
    return static_cast<double>(t1.QuadPart - t0.QuadPart) * 1000.0 / static_cast<double>(frequency.QuadPart);
}

struct NamedPageRange {
    u64 bytes, allocation;
};
std::unordered_map<u64, NamedPageRange> named_page_ranges;  // pages_lock

void remember_page_range(u64 address, u64 bytes, bool heap)
{
    if (!address || !bytes) return;
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) ||
        info.State != MEM_COMMIT || (!heap && info.Type != MEM_PRIVATE)) return;
    if (!heap) {
        ULONG_PTR low, high;
        GetCurrentThreadStackLimits(&low, &high);
        if (address >= low && address < high) return;
    }
    const u64 allocation = reinterpret_cast<u64>(info.AllocationBase);
    auto& range = named_page_ranges[address];
    if (range.allocation != allocation) range = {};
    range = {(std::max)(range.bytes, bytes), allocation};
}

void remember_heap_pages(u64 address, u64 bytes)
{
    if (!extra_title_pages) return;
    AcquireSRWLockExclusive(&pages_lock);
    remember_page_range(address, bytes, true);
    ReleaseSRWLockExclusive(&pages_lock);
}

// A stand-in allocation can move on replay. Wait until the CPU-start call is recorded so the
// player learns the translation before a MAP can address this heap.
void remember_heap_start(u64 object, u64 second)
{
    void* result = nullptr;
    const Wrapper* heap = receiver(reinterpret_cast<void*>(object), reinterpret_cast<void*>(second), &result,
                                   "replay descriptor heap CPU start");
    const u64 address = heap->own().title_cpu;
    if (!address) return;
    u64 bytes = 0;
    AcquireSRWLockShared(&descriptor_lock);
    for (const auto& range : descriptor_ranges)
        if (range.title == address) bytes = range.bytes;
    ReleaseSRWLockShared(&descriptor_lock);
    remember_heap_pages(address, bytes);
}

void remember_cbv_pages(u64 desc)
{
    D3D12_CONSTANT_BUFFER_VIEW_DESC view{};
    SIZE_T got = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(desc), &view, sizeof(view), &got) ||
        got != sizeof(view)) return;
    AcquireSRWLockExclusive(&pages_lock);
    remember_page_range(view.BufferLocation, view.SizeInBytes, false);
    ReleaseSRWLockExclusive(&pages_lock);
}

// A root descriptor has no size. Its translation reads at most 64 KiB. Each named range is
// clipped to its allocation and split at protection boundaries, without committing title pages.
void gather_named_pages(std::vector<PageRange>& raw, u64& big)
{
    constexpr u64 mask = x2w_replay::page_bytes - 1;
    for (auto it = named_page_ranges.begin(); it != named_page_ranges.end();) {
        const u64 address = it->first;
        const NamedPageRange range = it->second;
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) ||
            reinterpret_cast<u64>(info.AllocationBase) != range.allocation) {
            it = named_page_ranges.erase(it);
            continue;
        }
        ++it;
        if (range.bytes > page_range_limit) { ++big; continue; }
        const u64 end = (address + range.bytes + mask) & ~mask;
        u64 at = address & ~mask;
        while (at < end && reinterpret_cast<u64>(info.AllocationBase) == range.allocation) {
            const u64 region_end = reinterpret_cast<u64>(info.BaseAddress) + info.RegionSize;
            const u64 next = (std::min)(end, region_end);
            if (info.State == MEM_COMMIT && !(info.Protect & (PAGE_NOACCESS | PAGE_GUARD)))
                raw.push_back({at, next - at});
            at = next;
            if (at == end || !VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info))) break;
        }
    }
}

// The writer's sink: the chunk is appended to the calling thread's buffer, written out when that is large.
void page_flush_target()
{
    if (!page_target || page_target->bytes.empty()) return;
    AcquireSRWLockExclusive(&file_lock);
    drain(*page_target);
    // The calls' cap check (record_call) only runs when a thread buffer fills; a scan that drains every present would skip it.
    if (cap_exceeded() && !InterlockedExchange(&cap_reached, 1)) InterlockedExchange(&stop_requested, 1);
    ReleaseSRWLockExclusive(&file_lock);
}
void page_sink(const char* tag, const void* header, std::size_t header_bytes, const void* data, std::size_t data_bytes)
{
    const std::size_t before = page_target->bytes.size();
    append_chunk(page_target->bytes, tag, header, header_bytes, data, data_bytes);
    page_bytes_out += page_target->bytes.size() - before;
    if (page_target->bytes.size() >= (1u << 20)) page_flush_target();
}

// A texture placed at a title address has no size in its ResourceInfo (the stand-in notes 0). Its title memory is what the
// title placed: the size the XG stand-in reported for its description (desktop copyable footprints rounded to 64 KiB).
// Computed once per resource (pages_lock), again when the wrapper now stands for another resource; 0 for a buffer or
// when the device cannot size it.
struct TitleTextureSize {
    const void* memory;
    D3D12_RESOURCE_DESC desc;
    u64 bytes;
};
std::unordered_map<const Wrapper*, TitleTextureSize> title_texture_sizes;  // pages_lock
u64 page_sized_textures;  // pages_lock: textures sized this way, for the counters line
u64 title_texture_bytes(const Wrapper* w, const void* memory, const D3D12_RESOURCE_DESC& desc)
{
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || desc.Dimension == D3D12_RESOURCE_DIMENSION_UNKNOWN || !the_device)
        return 0;
    auto& cached = title_texture_sizes[w];
    if (cached.memory != memory || std::memcmp(&cached.desc, &desc, sizeof(desc)) != 0) {
        cached = {memory, desc, xg_reported_bytes(the_device, desc)};
        if (cached.bytes) ++page_sized_textures;
    }
    return cached.bytes;
}

// Merge page-aligned resource ranges and any enabled named title ranges.
std::vector<PageRange> page_merge_ranges(bool placed, u64& unsized, u64& big)
{
    std::vector<PageRange> raw;
    if (placed) {
        note_lock_acquire(false);
        for (const auto& item : resource_infos()) {
            const void* memory = item.first->own().title_memory;
            if (!memory) continue;
            const u64 bytes = item.second.bytes ? item.second.bytes : title_texture_bytes(item.first, memory, item.second.console);
            if (!bytes) { ++unsized; continue; }
            if (bytes > page_range_limit) { ++big; continue; }
            const u64 start = reinterpret_cast<u64>(memory) & ~u64{x2w_replay::page_bytes - 1};
            const u64 end = (reinterpret_cast<u64>(memory) + bytes + x2w_replay::page_bytes - 1) & ~u64{x2w_replay::page_bytes - 1};
            raw.push_back({start, end - start});
        }
        ReleaseSRWLockShared(&note_lock);
    }
    if (extra_title_pages) gather_named_pages(raw, big);
    std::sort(raw.begin(), raw.end(), [](const PageRange& a, const PageRange& b) { return a.address < b.address; });
    std::vector<PageRange> merged;
    for (const PageRange& r : raw) {
        if (!merged.empty() && r.address <= merged.back().address + merged.back().bytes) {
            const u64 end = (std::max)(merged.back().address + merged.back().bytes, r.address + r.bytes);
            merged.back().bytes = end - merged.back().address;
        } else {
            merged.push_back(r);
        }
    }
    return merged;
}

void page_gather_ranges(bool placed)
{
    u64 unsized = 0, big = 0;
    page_ranges = page_merge_ranges(placed, unsized, big);
    std::sort(page_ranges.begin(), page_ranges.end(), page_scan_order);
    page_skipped_no_size = unsized;
    page_skipped_big = big;
    if (page_cursor_range >= page_ranges.size()) { page_cursor_range = 0; page_cursor_offset = 0; }
}

bool page_budget_or_cap_reached()
{
    if (page_bytes_out >= page_budget_bytes) {
        if (!page_budget_reached) {
            page_budget_reached = true;
            log("x2w-replay pages: budget of %llu MiB reached, no more pages are written (calls go on)", page_budget_bytes >> 20);
        }
        return true;
    }
    if (cap_exceeded(page_target->bytes.size())) {  // the same cap as the calls: the recording ends with its XEND
        if (!InterlockedExchange(&cap_reached, 1)) InterlockedExchange(&stop_requested, 1);
        return true;
    }
    return false;
}

// Whether a page pass ends before the next slab of `range` at `offset`: a base image at 3/4 of the page budget (one slab
// is up to 1 MiB of pages), the budget, or the recording's cap. pages_lock held.
bool page_pass_ends(const PageRange& range, u64 offset, bool base)
{
    const u64 slab = (std::min<u64>)(page_slab_bytes, range.bytes - offset);
    if (base && page_bytes_out + slab > page_budget_bytes - page_budget_bytes / 4) {
        page_base_cut = true;
        page_cut_bytes = range.bytes;
        page_cut_address = range.address;
        log("x2w-replay pages: the base image stopped at %llu MiB, 3/4 of the %llu MiB page budget: ranges of %llu "
            "bytes and more are left out, the rest of the budget keeps the window's changed pages",
            page_bytes_out >> 20, page_budget_bytes >> 20, range.bytes);
        return true;
    }
    return page_budget_or_cap_reached();
}

// One slab: the committed, readable pages at the start of `want` bytes at `at`, into the page writer.
void page_capture_slab(u64 at, std::size_t want, std::vector<unsigned char>& buffer, u64 stamp)
{
    const std::size_t readable = committed_readable_prefix(at, want) & ~std::size_t{x2w_replay::page_bytes - 1};
    SIZE_T got = 0;
    if (!readable || !ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(at), buffer.data(), readable, &got) ||
        got != readable) {
        page_unreadable += want / x2w_replay::page_bytes;
        return;
    }
    if (readable < want) page_unreadable += (want - readable) / x2w_replay::page_bytes;
    page_writer->capture(at, buffer.data(), readable, stamp);
}

void page_log_counters(double ms, bool base)
{
    const x2w_replay::PageStats& s = page_writer->stats();
    u64 total = 0;
    for (const PageRange& r : page_ranges) total += r.bytes;
    log("x2w-replay pages: ranges %zu (%llu MiB), pages seen %llu, distinct %llu, zero %llu, unchanged %llu, written %llu MiB, "
        "textures sized %llu, skipped (no size) %llu, over %llu MiB %llu, unreadable %llu, written+maps %llu MiB, scan %.0f ms%s",
        page_ranges.size(), total >> 20, s.seen, s.distinct, s.zero, s.unchanged, s.bytes_written >> 20, page_sized_textures,
        page_skipped_no_size, page_range_limit >> 20, page_skipped_big, page_unreadable, page_bytes_out >> 20, ms,
        base ? " (base image)" : "");
}

// One pass: all ranges (the base image), or from the cursor until the time budget is used. pages_lock held.
void page_scan(bool everything, u64 stamp, bool placed = true)
{
    if (!page_writer) return;
    LARGE_INTEGER t0;
    QueryPerformanceCounter(&t0);
    page_gather_ranges(placed);
    if (everything) { page_cursor_range = 0; page_cursor_offset = 0; if (placed) page_base_done = true; }
    std::vector<unsigned char> buffer(page_slab_bytes);
    std::size_t visited = 0;
    bool stopped_early = false;
    while (visited < page_ranges.size() && !stopped_early) {
        const PageRange range = page_ranges[page_cursor_range];
        if (page_after_cut(range)) page_cursor_offset = range.bytes;  // not in the base image: left out
        while (page_cursor_offset < range.bytes) {
            if (page_pass_ends(range, page_cursor_offset, everything)) {
                if (everything && page_base_cut) {
                    page_cursor_range = 0;
                    page_cursor_offset = 0;
                }
                stopped_early = true;
                break;
            }
            const std::size_t want = static_cast<std::size_t>((std::min<u64>)(page_slab_bytes, range.bytes - page_cursor_offset));
            page_capture_slab(range.address + page_cursor_offset, want, buffer, stamp);
            page_cursor_offset += want;
            if (!everything && page_ms_since(t0) > page_scan_ms) {
                stopped_early = true;
                break;
            }
        }
        if (stopped_early) break;
        page_cursor_offset = 0;
        page_cursor_range = (page_cursor_range + 1) % page_ranges.size();
        ++visited;
    }
    page_flush_target();
    ++page_scans;
    const double ms = page_ms_since(t0);
    // A window open from the recording's start writes its base image inside it (no present came before it): a window
    // counted in seconds starts counting after it, or it closes before a second frame is recorded.
    if (everything && placed && end_s < 1e300) {
        const double was = end_s;
        end_s = was + ms / 1000.0;
        log("x2w-replay record: the base image took %.6f s: the window now ends at %.6f s (was %.6f s)", ms / 1000.0, end_s, was);
    }
    if (everything || page_scans % 20 == 0) page_log_counters(ms, everything);
}

// One slice of the base image written before the window: from its cursor, slab after slab, until base_scan_ms is used
// (one slab at least). True when the base image is complete: every range written, or its share of the budget used.
// pages_lock held.
bool page_base_slice(u64 stamp)
{
    LARGE_INTEGER t0;
    QueryPerformanceCounter(&t0);
    page_gather_ranges(true);
    auto it = std::lower_bound(page_ranges.begin(), page_ranges.end(), PageRange{base_cursor_address, base_cursor_bytes},
                               page_scan_order);
    if (it == page_ranges.end() || it->address != base_cursor_address || it->bytes != base_cursor_bytes) base_cursor_offset = 0;
    std::vector<unsigned char> buffer(page_slab_bytes);
    bool first = true, ended = false;
    for (; it != page_ranges.end() && !ended; ++it, base_cursor_offset = 0) {
        base_cursor_address = it->address;
        base_cursor_bytes = it->bytes;
        while (base_cursor_offset < it->bytes) {
            if (!first && page_ms_since(t0) > base_scan_ms) {
                page_flush_target();
                return false;
            }
            first = false;
            if (page_pass_ends(*it, base_cursor_offset, true)) {
                ended = true;
                break;
            }
            const std::size_t want = static_cast<std::size_t>((std::min<u64>)(page_slab_bytes, it->bytes - base_cursor_offset));
            page_capture_slab(it->address + base_cursor_offset, want, buffer, stamp);
            base_cursor_offset += want;
        }
    }
    page_flush_target();
    page_base_done = true;
    ++page_scans;
    page_log_counters(page_ms_since(t0), true);
    return true;
}

// Either a Present or a submit can be the first scan (pages_lock held).
void init_page_writer()
{
    if (!page_writer_tried) {
        page_writer_tried = true;
        char value[32]{};
        if (GetEnvironmentVariableA("X2W_REPLAY_PAGES", value, sizeof(value)) && !std::strcmp(value, "0")) return;
        if (GetEnvironmentVariableA("X2W_REPLAY_SCAN_MS", value, sizeof(value))) {
            const double ms = std::atof(value);
            if (ms > 0.0) page_scan_ms = ms;
        }
        if (GetEnvironmentVariableA("X2W_REPLAY_BASE_SCAN_MS", value, sizeof(value))) {
            const double ms = std::atof(value);
            if (ms > 0.0) base_scan_ms = ms;
        }
        page_budget_bytes = max_bytes / 2;
        if (GetEnvironmentVariableA("X2W_REPLAY_PAGES_MAX_MIB", value, sizeof(value)) && std::atof(value) > 0.0)
            page_budget_bytes = static_cast<u64>(std::atof(value) * 1048576.0);
        if (GetEnvironmentVariableA("X2W_REPLAY_PAGES_MAX_RANGE_MIB", value, sizeof(value)) && std::atof(value) > 0.0)
            page_range_limit = static_cast<u64>(std::atof(value) * 1048576.0);
        page_writer = new x2w_replay::PageWriter(page_sink);
    }
}

void scan_pages_into(ThreadBuffer& b, bool everything, u64 stamp, bool placed = true)
{
    if (!page_writer) return;
    page_target = &b;
    page_scan(everything, stamp, placed);
    page_target = nullptr;
}

// The window's time has come at a Present (b's lock held). Its base image of title memory is written first, a slice of at
// most base_scan_ms per Present, while the calls go on into the prefix: the title keeps presenting (one 30 s scan
// inside a Present ended in a fault in a tester's title). When it is complete the window opens at that Present, and
// its seconds and Presents count from there. True while the window waits for it.
bool base_image_holds_window(ThreadBuffer& b, LONG present)
{
    if (window_open || page_base_done || extra_title_pages || replay_dump_enabled()) return false;
    if (!base_warming) {
        if (!in_window()) return false;
        base_warming = true;
        base_started_s = seconds_now();
        base_started_present = present;
    }
    if (!TryAcquireSRWLockExclusive(&pages_lock)) return true;  // a submit scan is running: the next Present goes on
    init_page_writer();
    bool complete = true;
    if (page_writer) {
        if (!base_slices++)
            log("x2w-replay record: the window waits for the base image of title memory (at most %.0f ms per present)", base_scan_ms);
        const u64 before = page_bytes_out;
        page_target = &b;
        complete = page_base_slice(static_cast<u64>(InterlockedIncrement64(&next_seq)));
        page_target = nullptr;
        base_prefix_bytes += page_bytes_out - before;
    }
    ReleaseSRWLockExclusive(&pages_lock);
    if (!complete) return true;
    base_warming = false;
    if (!page_writer) return false;
    const double took = seconds_now() - base_started_s;
    const LONG waited = present - base_started_present;
    if (last_present != 0x7fffffff) {
        first_present += waited;
        last_present += waited;
    }
    if (end_s < 1e300) {
        const double was = end_s;
        start_s += took;
        end_s = was + took;
        log("x2w-replay record: the base image took %.6f s and %ld presents before the window: the window now ends at %.6f s "
            "(was %.6f s)", took, waited, end_s, was);
    } else {
        log("x2w-replay record: the base image took %.6f s and %ld presents before the window: the window opens at present %ld",
            took, waited, present);
    }
    return false;
}

// The window has just opened at a Present. Its base image was written before (base_image_holds_window): the ranges
// changed since then are rescanned from the smallest (descriptor rings, constants), the same time budget as a Present
// rescan. With no base image yet it is taken now.
inline void pages_at_window_open(ThreadBuffer& b, u64 seq, LONG present)
{
    (void)present;
    if (replay_dump_enabled()) return;  // Dump mode stamps each scan at its submission.
    page_prev_mark = seq;
    if (!TryAcquireSRWLockExclusive(&pages_lock)) return;
    init_page_writer();
    if (page_base_done) {
        page_cursor_range = 0;
        page_cursor_offset = 0;
    }
    scan_pages_into(b, !page_base_done, seq);
    ReleaseSRWLockExclusive(&pages_lock);
}

const PageRange* draw_find_range(u64 address)
{
    const auto it = std::upper_bound(draw_ranges.begin(), draw_ranges.end(), address,
        [](u64 at, const PageRange& range) { return at < range.address; });
    if (it == draw_ranges.begin()) return nullptr;
    const auto& range = *(it - 1);
    return address - range.address < range.bytes ? &range : nullptr;
}

void draw_log_counters(LONG present, double used)
{
    log("x2w-replay draw pages: captures %llu, outside title memory %llu, over the per-present time %llu, "
        "pending dropped %llu, used %.1f ms of %.0f ms at present %ld",
        draw_captures, draw_outside, draw_over_time, draw_pending_dropped, used, draw_pages_ms, present);
}

bool draw_time_exhausted(const LARGE_INTEGER& start, u64 skipped)
{
    const double used = draw_time_used + page_ms_since(start);
    if (used < draw_pages_ms) return false;
    draw_over_time += skipped;
    return true;
}

void draw_capture_span(const PageRange& span, LONG present, u64 stamp, const LARGE_INTEGER& start)
{
    if (draw_time_exhausted(start, 1)) return;
    // The title ranges are gathered again at the first capture of each Present (resources come and go), and once more
    // in that Present when a span falls outside them (a resource made since).
    const auto gather = [&] {
        u64 unsized = 0, big = 0;
        draw_ranges = page_merge_ranges(true, unsized, big);
    };
    if (draw_ranges_present != present) {
        gather();
        draw_ranges_present = present;
        draw_ranges_missed = false;
    }
    const PageRange* range = draw_find_range(span.address);
    if (!range && !draw_ranges_missed) {
        draw_ranges_missed = true;
        gather();
        range = draw_find_range(span.address);
    }
    if (!range) { ++draw_outside; return; }
    if (page_after_cut(*range)) return;
    constexpr u64 mask = x2w_replay::page_bytes - 1;
    const u64 end = (std::min)(range->address + range->bytes,
        (span.address + (std::min<u64>)(span.bytes, draw_span_limit) + mask) & ~mask);
    bool captured = false;
    for (u64 at = span.address & ~mask; at < end; at += page_slab_bytes) {
        if (page_budget_or_cap_reached() || draw_time_exhausted(start, 1)) break;
        if (draw_read_buffer.empty()) draw_read_buffer.resize(page_slab_bytes);
        page_capture_slab(at, static_cast<std::size_t>((std::min<u64>)(end - at, page_slab_bytes)), draw_read_buffer, stamp);
        captured = true;
    }
    if (captured) ++draw_captures;
}

void draw_add_pending(u64 list, u64 address, u64 bytes)
{
    if (!address || !bytes) return;
    auto& spans = draw_pending[list];
    for (auto& span : spans) {
        if (span.address != address) continue;
        span.bytes = (std::max)(span.bytes, bytes);
        return;
    }
    if (spans.size() == draw_pending_limit) { ++draw_pending_dropped; return; }
    spans.push_back({address, bytes});
}

void draw_consume_pending(u64 list, LONG present, u64 stamp, const LARGE_INTEGER& start)
{
    const auto it = draw_pending.find(list);
    if (it == draw_pending.end()) return;
    for (const auto& span : it->second) draw_capture_span(span, present, stamp, start);
    draw_pending.erase(it);
}

void draw_pages_at_call(int slot, u64 list, u64 rdx, u64 r8, u64 r9, u64 stamp)
{
    const LONG present = InterlockedCompareExchange(&presents, 0, 0);
    if (draw_time_present != present) {
        draw_time_present = present;
        draw_time_used = 0;
    }
    LARGE_INTEGER start{};
    QueryPerformanceCounter(&start);
    if (slot <= 14 || slot == 61) {
        draw_consume_pending(list, present, stamp, start);
    } else if (!draw_time_exhausted(start, 1)) {
        if (slot <= 46) {
            if (r8) draw_capture_span({r8, draw_root_window}, present, stamp, start);  // slot 44 copies it now
            draw_add_pending(list, r8, draw_root_window);  // and a draw reads the root address again
        } else if (slot == 47) {
            D3D12_INDEX_BUFFER_VIEW view{};
            if (rdx && copy_readable(rdx, &view, sizeof(view)))
                draw_add_pending(list, view.BufferLocation, view.SizeInBytes);
        } else {
            D3D12_VERTEX_BUFFER_VIEW views[16];
            const UINT count = (std::min)(static_cast<UINT>(r8), 16u);
            if (r9 && count && copy_readable(r9, views, count * sizeof(views[0])))
                for (UINT i = 0; i < count; ++i)
                    draw_add_pending(list, views[i].BufferLocation, views[i].SizeInBytes);
        }
    }
    draw_time_used += page_ms_since(start);
    if (draw_time_used >= draw_pages_ms && !draw_time_logged) {
        draw_time_logged = true;
        draw_log_counters(present, draw_time_used);
    }
}

// With submit scans enabled, changes observed at Present must follow the latest scanned submission.
inline void pages_at_present(ThreadBuffer& b, u64 seq, LONG present, bool scan)
{
    scan = scan && !replay_dump_enabled();
    u64 stamp = 0;
    if (scan) {
        stamp = page_prev_mark ? page_prev_mark : seq;
        page_prev_mark = seq;
    }
    const bool log_draw = draw_pages && present % 20 == 0;
    if ((!scan && !log_draw) || !TryAcquireSRWLockExclusive(&pages_lock)) return;
    if (log_draw) draw_log_counters(present, draw_time_present == present - 1 ? draw_time_used : 0.0);
    if (scan) {
        init_page_writer();  // a window that was open from the start never calls pages_at_window_open
        scan_pages_into(b, !page_base_done, (std::max)(stamp, page_last_submit));
    }
    ReleaseSRWLockExclusive(&pages_lock);
}

// Allocate the call sequence with pages_lock held. Present scans may still use older stamps.
u64 pages_before_call(int kind, int slot, u64 rcx, u64 rdx, u64 r8, u64 r9, bool root, bool submit, bool scan)
{
    ThreadBuffer& b = my_buffer();
    AcquireSRWLockExclusive(&b.lock);
    AcquireSRWLockExclusive(&pages_lock);
    const u64 seq = static_cast<u64>(InterlockedIncrement64(&next_seq));
    if (root && extra_title_pages) remember_page_range(r8, 0x10000, false);
    init_page_writer();
    if (scan && page_writer && !page_budget_reached && !stop_requested && active) {
        scan_pages_into(b, window_open && !page_base_done, seq - 1, window_open);
        if (submit) page_last_submit = (std::max)(page_last_submit, seq);
    }
    const bool draw_slot = kind == list_kind &&
        (slot == 12 || slot == 13 || slot == 14 || slot == 61 || (slot >= 41 && slot <= 48));
    // No flush here: the chunks stay in the thread buffer (page_sink writes it out past 1 MiB, record_call when it is
    // full), so a draw costs no file write.
    if (draw_slot && draw_pages && window_open && page_writer && !page_budget_reached && !stop_requested && active) {
        page_target = &b;
        draw_pages_at_call(slot, rcx, rdx, r8, r9, seq - 1);
        page_target = nullptr;
    }
    ReleaseSRWLockExclusive(&pages_lock);
    ReleaseSRWLockExclusive(&b.lock);
    return seq;
}
