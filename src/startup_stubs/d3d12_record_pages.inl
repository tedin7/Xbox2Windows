// Memory pages for replay recordings (docs/startup-runtime.md#d3d12-replay-record), the page store of replay_pages.h:
// PAGE chunks (one per distinct non-zero 4 KiB page) and MAP chunks (which pages a range holds, only runs that changed).
// Included inside namespace replay_record by d3d12_record.inl; each hook appends to the caller's thread buffer `b` (its lock
// held by the caller) and a replay applies a MAP before the call or mark with the MAP's seq.
//
// Memory = the resources that have title_memory and a known size (placed resources, hook allocations), merged and cut to
// whole pages. When the window opens one pass writes the base image of all of them; after each present a pass limited to
// X2W_REPLAY_SCAN_MS (default 250) continues round-robin and writes only pages that changed since this recording last saw
// them. A MAP read at present N is stamped with the mark of present N-1: the frame that used the memory starts after it.
// The size cap is X2W_REPLAY_MAX_MIB as for calls. X2W_REPLAY_PAGES=0 leaves the pages out.
// One line of counters after the base image and about every 20th present:
//   x2w-replay pages: ranges N (M MiB), pages seen S, distinct D, zero Z, unchanged U, written W MiB, skipped (no size) K,
//   unreadable R, scan T ms
constexpr std::size_t page_slab_bytes = 1u << 20;

struct PageRange { u64 address, bytes; };
x2w_replay::PageWriter* page_writer;  // made at the first window open; never freed
bool page_writer_tried;
ThreadBuffer* page_target;            // the buffer the running scan appends to (its lock is held by the caller)
u64 page_prev_mark;                   // the previous present's mark, the stamp of the next scan
SRWLOCK pages_lock = SRWLOCK_INIT;   // one scan at a time; held for the whole scan
double page_scan_ms = 250.0;
// The pages have a budget of their own (X2W_REPLAY_PAGES_MAX_MIB, default half of X2W_REPLAY_MAX_MIB): when it is spent no more
// pages are written but the calls go on. Resources over X2W_REPLAY_PAGES_MAX_RANGE_MIB (default 8) are left out (counted), and
// the smaller ranges come first: constants, vertex and index data, small textures before the big ones.
u64 page_budget_bytes, page_range_limit = 8ull << 20, page_bytes_out, page_skipped_big;
bool page_budget_reached;
std::vector<PageRange> page_ranges;   // pages_lock
std::size_t page_cursor_range;        // pages_lock
u64 page_cursor_offset;               // pages_lock
u64 page_unreadable, page_skipped_no_size, page_scans;

// The writer's sink: the chunk is appended to the calling thread's buffer, written out when that is large.
void page_flush_target()
{
    if (!page_target || page_target->bytes.empty()) return;
    AcquireSRWLockExclusive(&file_lock);
    drain(*page_target);
    ReleaseSRWLockExclusive(&file_lock);
}
void page_sink(const char* tag, const void* header, std::size_t header_bytes, const void* data, std::size_t data_bytes)
{
    const std::size_t before = page_target->bytes.size();
    append_chunk(page_target->bytes, tag, header, header_bytes, data, data_bytes);
    page_bytes_out += page_target->bytes.size() - before;
    if (page_target->bytes.size() >= (1u << 20)) page_flush_target();
}

// The merged, page-aligned ranges of every resource with title memory and a known size. note_lock is taken shared.
void page_gather_ranges()
{
    std::vector<PageRange> raw;
    u64 unsized = 0, big = 0;
    note_lock_acquire(false);
    for (const auto& item : resource_infos()) {
        const void* memory = item.first->own().title_memory;
        if (!memory) continue;
        if (!item.second.bytes) { ++unsized; continue; }
        if (item.second.bytes > page_range_limit) { ++big; continue; }
        const u64 start = reinterpret_cast<u64>(memory) & ~u64{x2w_replay::page_bytes - 1};
        const u64 end = (reinterpret_cast<u64>(memory) + item.second.bytes + x2w_replay::page_bytes - 1) & ~u64{x2w_replay::page_bytes - 1};
        raw.push_back({start, end - start});
    }
    ReleaseSRWLockShared(&note_lock);
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
void page_scan(bool everything, u64 stamp)
{
    if (!page_writer) return;
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);
    page_gather_ranges();
    if (everything) { page_cursor_range = 0; page_cursor_offset = 0; }
    std::vector<unsigned char> buffer(page_slab_bytes);
    std::size_t visited = 0;
    bool stopped_early = false;
    while (visited < page_ranges.size() && !stopped_early) {
        const PageRange range = page_ranges[page_cursor_range];
        while (page_cursor_offset < range.bytes) {
            if (page_bytes_out >= page_budget_bytes) {
                if (!page_budget_reached) {
                    page_budget_reached = true;
                    log("x2w-replay pages: budget of %llu MiB reached, no more pages are written (calls go on)", page_budget_bytes >> 20);
                }
                stopped_early = true;
                break;
            }
            if (bytes_written + page_target->bytes.size() >= max_bytes) {  // the same cap as the calls: the recording ends with its XEND
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
            "skipped (no size) %llu, over %llu MiB %llu, unreadable %llu, written+maps %llu MiB, scan %.0f ms%s",
            page_ranges.size(), total >> 20, s.seen, s.distinct, s.zero, s.unchanged, s.bytes_written >> 20, page_skipped_no_size,
            page_range_limit >> 20, page_skipped_big, page_unreadable, page_bytes_out >> 20, static_cast<double>(t1.QuadPart - t0.QuadPart) * 1000.0 / static_cast<double>(frequency.QuadPart),
            everything ? " (base image)" : "");
    }
}

// The window has just opened at present `present` (mark `seq`): the base image of the memory a replay needs.
inline void pages_at_window_open(ThreadBuffer& b, u64 seq, LONG present)
{
    (void)present;
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
    page_prev_mark = seq;
    if (!page_writer || !TryAcquireSRWLockExclusive(&pages_lock)) return;
    page_target = &b;
    page_scan(true, seq);
    page_target = nullptr;
    ReleaseSRWLockExclusive(&pages_lock);
}

// A present inside the window: the pages changed since the last capture, within the time budget.
inline void pages_at_present(ThreadBuffer& b, u64 seq, LONG present)
{
    (void)present;
    const u64 stamp = page_prev_mark ? page_prev_mark : seq;
    page_prev_mark = seq;
    if (!page_writer || !TryAcquireSRWLockExclusive(&pages_lock)) return;
    page_target = &b;
    page_scan(false, stamp);
    page_target = nullptr;
    ReleaseSRWLockExclusive(&pages_lock);
}

// Before an ExecuteCommandLists inside the window (queue slot 10; a compute-only stretch has no present): the pages
// changed since the last capture. `seq` is the Execute call's sequence number minus one, so a replay writes them before
// that call. Empty until the page store fills it.
inline void pages_at_submit(ThreadBuffer& b, u64 seq)
{
    (void)b, (void)seq;
}
