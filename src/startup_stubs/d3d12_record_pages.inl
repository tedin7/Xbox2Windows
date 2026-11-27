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
// X2W_REPLAY_TITLE_RANGES=1 adds named title memory and scans in call order before the stand-in reads it.
// The size cap is X2W_REPLAY_MAX_MIB as for calls. X2W_REPLAY_PAGES=0 leaves the pages out.
// One line of counters after the base image and about every 20th scan:
//   x2w-replay pages: ranges N (M MiB), pages seen S, distinct D, zero Z, unchanged U, written W MiB, skipped (no size) K,
//   unreadable R, scan T ms
constexpr std::size_t page_slab_bytes = 1u << 20;

struct PageRange { u64 address, bytes; };
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
void page_gather_ranges(bool placed)
{
    std::vector<PageRange> raw;
    u64 unsized = 0, big = 0;
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
    page_ranges.clear();
    for (const PageRange& r : raw) {
        if (!page_ranges.empty() && r.address <= page_ranges.back().address + page_ranges.back().bytes) {
            const u64 end = (std::max)(page_ranges.back().address + page_ranges.back().bytes, r.address + r.bytes);
            page_ranges.back().bytes = end - page_ranges.back().address;
        } else {
            page_ranges.push_back(r);
        }
    }
    std::sort(page_ranges.begin(), page_ranges.end(), [](const PageRange& a, const PageRange& b) {
        return a.bytes != b.bytes ? a.bytes < b.bytes : a.address < b.address;
    });
    page_skipped_no_size = unsized;
    page_skipped_big = big;
    if (page_cursor_range >= page_ranges.size()) { page_cursor_range = 0; page_cursor_offset = 0; }
}

// One pass: all ranges (the base image), or from the cursor until the time budget is used. pages_lock held.
void page_scan(bool everything, u64 stamp, bool placed = true)
{
    if (!page_writer) return;
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    page_gather_ranges(placed);
    if (everything) { page_cursor_range = 0; page_cursor_offset = 0; if (placed) page_base_done = true; }
    std::vector<unsigned char> buffer(page_slab_bytes);
    std::size_t visited = 0;
    bool stopped_early = false;
    const u64 base_limit = page_budget_bytes - page_budget_bytes / 4;
    while (visited < page_ranges.size() && !stopped_early) {
        const PageRange range = page_ranges[page_cursor_range];
        if (page_after_cut(range)) page_cursor_offset = range.bytes;  // not in the base image: left out
        while (page_cursor_offset < range.bytes) {
            // The next slab could take the base image past its share (one slab is up to 1 MiB of pages).
            const u64 slab = (std::min<u64>)(page_slab_bytes, range.bytes - page_cursor_offset);
            if (everything && page_bytes_out + slab > base_limit) {
                page_base_cut = true;
                page_cut_bytes = range.bytes;
                page_cut_address = range.address;
                log("x2w-replay pages: the base image stopped at %llu MiB, 3/4 of the %llu MiB page budget: ranges of %llu "
                    "bytes and more are left out, the rest of the budget keeps the window's changed pages",
                    page_bytes_out >> 20, page_budget_bytes >> 20, range.bytes);
                page_cursor_range = 0;
                page_cursor_offset = 0;
                stopped_early = true;
                break;
            }
            if (page_bytes_out >= page_budget_bytes) {
                if (!page_budget_reached) {
                    page_budget_reached = true;
                    log("x2w-replay pages: budget of %llu MiB reached, no more pages are written (calls go on)", page_budget_bytes >> 20);
                }
                stopped_early = true;
                break;
            }
            if (cap_exceeded(page_target->bytes.size())) {  // the same cap as the calls: the recording ends with its XEND
                if (!InterlockedExchange(&cap_reached, 1)) InterlockedExchange(&stop_requested, 1);
                stopped_early = true;
                break;
            }
            const u64 at = range.address + page_cursor_offset;
            const std::size_t want = static_cast<std::size_t>((std::min<u64>)(page_slab_bytes, range.bytes - page_cursor_offset));
            page_cursor_offset += want;
            const std::size_t readable = committed_readable_prefix(at, want) & ~std::size_t{x2w_replay::page_bytes - 1};
            SIZE_T got = 0;
            if (!readable || !ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(at), buffer.data(), readable, &got) ||
                got != readable) {
                page_unreadable += want / x2w_replay::page_bytes;
                continue;
            }
            if (readable < want) page_unreadable += (want - readable) / x2w_replay::page_bytes;
            page_writer->capture(at, buffer.data(), readable, stamp);
            if (!everything) {
                QueryPerformanceCounter(&t1);
                if (static_cast<double>(t1.QuadPart - t0.QuadPart) * 1000.0 / static_cast<double>(frequency.QuadPart) > page_scan_ms) {
                    stopped_early = true;
                    break;
                }
            }
        }
        if (stopped_early) break;
        page_cursor_offset = 0;
        page_cursor_range = (page_cursor_range + 1) % page_ranges.size();
        ++visited;
    }
    page_flush_target();
    ++page_scans;
    QueryPerformanceCounter(&t1);
    if (everything || page_scans % 20 == 0) {
        const x2w_replay::PageStats& s = page_writer->stats();
        u64 total = 0;
        for (const PageRange& r : page_ranges) total += r.bytes;
        log("x2w-replay pages: ranges %zu (%llu MiB), pages seen %llu, distinct %llu, zero %llu, unchanged %llu, written %llu MiB, "
            "textures sized %llu, skipped (no size) %llu, over %llu MiB %llu, unreadable %llu, written+maps %llu MiB, scan %.0f ms%s",
            page_ranges.size(), total >> 20, s.seen, s.distinct, s.zero, s.unchanged, s.bytes_written >> 20, page_sized_textures,
            page_skipped_no_size,
            page_range_limit >> 20, page_skipped_big, page_unreadable, page_bytes_out >> 20, static_cast<double>(t1.QuadPart - t0.QuadPart) * 1000.0 / static_cast<double>(frequency.QuadPart),
            everything ? " (base image)" : "");
    }
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

// The window has just opened at a Present: take the base image.
inline void pages_at_window_open(ThreadBuffer& b, u64 seq, LONG present)
{
    (void)present;
    if (replay_dump_enabled()) return;  // Dump mode stamps each scan at its submission.
    page_prev_mark = seq;
    if (!TryAcquireSRWLockExclusive(&pages_lock)) return;
    init_page_writer();
    scan_pages_into(b, true, seq);
    ReleaseSRWLockExclusive(&pages_lock);
}

// With submit scans enabled, changes observed at Present must follow the latest scanned submission.
inline void pages_at_present(ThreadBuffer& b, u64 seq, LONG present)
{
    (void)present;
    if (replay_dump_enabled()) return;
    const u64 stamp = page_prev_mark ? page_prev_mark : seq;
    page_prev_mark = seq;
    if (!TryAcquireSRWLockExclusive(&pages_lock)) return;
    init_page_writer();  // a window that was open from the start never calls pages_at_window_open
    scan_pages_into(b, !page_base_done, (std::max)(stamp, page_last_submit));
    ReleaseSRWLockExclusive(&pages_lock);
}

// Take the snapshot in sequence order, so PageWriter's deltas are applied in the order it saw them. A submit scan
// (X2W_REPLAY_PAGES_AT_SUBMIT) keeps the stamp of later Present scans after the submission.
u64 pages_before_call(bool root, u64 address, bool submit)
{
    ThreadBuffer& b = my_buffer();
    AcquireSRWLockExclusive(&b.lock);
    AcquireSRWLockExclusive(&pages_lock);
    const u64 seq = static_cast<u64>(InterlockedIncrement64(&next_seq));
    if (root) remember_page_range(address, 0x10000, false);
    init_page_writer();
    if (page_writer && !page_budget_reached && !stop_requested && active) {
        scan_pages_into(b, window_open && !page_base_done, seq - 1, window_open);
        if (submit) page_last_submit = (std::max)(page_last_submit, seq);
    }
    ReleaseSRWLockExclusive(&pages_lock);
    ReleaseSRWLockExclusive(&b.lock);
    return seq;
}
