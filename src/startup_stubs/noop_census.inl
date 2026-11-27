// Included by d3d12_xs_device.cpp after the wrapper, upload and placement helpers.
namespace noop_census {
constexpr unsigned capacity = 12;
constexpr unsigned small_limit = 64;
enum Arg : unsigned {
    zero = 0, small = 1, integer = 65, wrapper = 70, resource = 100, token = 128,
    upload = 640, placed = 650, readable = 720, other, busy
};
static_assert(busy < 1024 && token + slot_count <= upload);
enum Credit : unsigned { draw, indexed, dispatch, indirect, after_pipeline, credit_count };

struct Shape {
    std::atomic<u64> key{};
    std::atomic<u64> calls{};
    std::atomic<u64> credits[credit_count]{};
    unsigned char sample[6][32]{};
    unsigned char sample_bytes[6]{};
    std::atomic<bool> sampled{};
};
struct Slot {
    Shape shapes[capacity];
    std::atomic<u64> overflow{};
};
static_assert(std::atomic<u64>::is_always_lock_free && std::atomic<bool>::is_always_lock_free);
Slot lists[slot_count], devices[slot_count];
thread_local bool view_as_pointer_for_test;

// Query protection without touching guard pages or committing lazy memory. Copy only the
// readable prefix in this region. ReadProcessMemory handles a concurrent unmap or protection change.
unsigned read_region(u64 address, void* output, unsigned bytes)
{
    MEMORY_BASIC_INFORMATION info{};
    const auto* pointer = reinterpret_cast<const void*>(address);
    if (!VirtualQuery(pointer, &info, sizeof(info)) || info.State != MEM_COMMIT ||
        (info.Protect & (PAGE_GUARD | PAGE_NOACCESS))) return 0;
    switch (info.Protect & 0xff) {
    case PAGE_READONLY: case PAGE_READWRITE: case PAGE_WRITECOPY:
    case PAGE_EXECUTE_READ: case PAGE_EXECUTE_READWRITE: case PAGE_EXECUTE_WRITECOPY:
        break;
    default: return 0;
    }
    const auto available = info.RegionSize - (address - reinterpret_cast<u64>(info.BaseAddress));
    const auto count = static_cast<unsigned>((std::min)(u64{bytes}, static_cast<u64>(available)));
    return !output || ReadProcessMemory(GetCurrentProcess(), pointer, output, count, nullptr) ? count : 0;
}

// Finish a sample that straddles two readable regions with different protections.
unsigned read(u64 address, void* output, unsigned bytes)
{
    unsigned copied = 0;
    auto* dest = static_cast<unsigned char*>(output);
    while (copied < bytes) {
        const unsigned count = read_region(address + copied, dest ? dest + copied : nullptr, bytes - copied);
        if (!count) break;
        copied += count;
    }
    return copied;
}

unsigned classify(Wrapper* self, Kind kind, u64 value)
{
    if (!value) return zero;
    if (value < small_limit) return small + static_cast<unsigned>(value);
    if (value < 0x10000) return integer;
    const auto* ptr = reinterpret_cast<const Wrapper*>(value);
    if (!(value & 7)) {
        // Try-only access to existing indexes. Record contention explicitly and never wait
        // for a writer or retain a wrapper pointer after releasing the registry lock.
        if (!TryAcquireSRWLockShared(&registry_lock)) return busy;
        const auto found = registry().count(ptr) != 0;
        const auto object_kind = found ? ptr->own().kind : other_kind;
        const unsigned dimension = found ? ptr->own().census_dimension : D3D12_RESOURCE_DIMENSION_UNKNOWN;
        ReleaseSRWLockShared(&registry_lock);
        if (found) return object_kind == resource_kind
            ? resource + (dimension <= D3D12_RESOURCE_DIMENSION_TEXTURE3D ? dimension : 0)
            : wrapper + object_kind;
    }
    std::uint32_t words[3]{};
    const unsigned bytes = read(value, words, sizeof(words));
    if (!view_as_pointer_for_test && bytes == sizeof(words) && words[0] == view_token_magic && words[2] >= 17 && words[2] <= 22)
        return token + words[2];
    if (kind == list_kind) {
        if (!TryAcquireSRWLockShared(&allocator_lock)) return busy;
        const auto& state = self->own();
        const bool in_upload = state.list_open && state.list_uploads && state.list_uploads->range(value, 1).resource;
        ReleaseSRWLockShared(&allocator_lock);
        if (in_upload) return upload;
    }
    if (!TryAcquireSRWLockShared(&note_lock)) return busy;
    u64 width = 0;
    const auto placement = find_placed_buffer_range(kind == list_kind ? self->own().list_device : self,
                                                    value, 1, false, &width);
    ReleaseSRWLockShared(&note_lock);
    if (placement.owner) {
        unsigned long size_class = 0;
        _BitScanReverse64(&size_class, width);
        return placed + size_class;
    }
    return bytes || read(value, nullptr, 1) ? readable : other;
}

void format_arg(char* text, std::size_t size, unsigned arg)
{
    static constexpr const char* types[] = {"CBV", "SRV", "UAV", "RTV", "DSV", "Sampler"};
    static constexpr const char* dimensions[] = {"unknown", "buffer", "texture1d", "texture2d", "texture3d"};
    if (arg == zero) std::snprintf(text, size, "zero");
    else if (arg < integer) std::snprintf(text, size, "small:%u", arg - small);
    else if (arg == integer) std::snprintf(text, size, "small");
    else if (arg >= wrapper && arg < wrapper + kind_count)
        std::snprintf(text, size, "wrapper:%s", kind_names[arg - wrapper]);
    else if (arg >= resource && arg < resource + 5)
        std::snprintf(text, size, "wrapper:Resource/%s", dimensions[arg - resource]);
    else if (arg >= token + 17 && arg <= token + 22)
        std::snprintf(text, size, "view:slot%u/%s", arg - token, types[arg - token - 17]);
    else if (arg == upload) std::snprintf(text, size, "upload");
    else if (arg >= placed && arg < placed + 64) std::snprintf(text, size, "placed:2^%u", arg - placed);
    else std::snprintf(text, size, "%s", arg == readable ? "readable" : arg == busy ? "index-busy" : "other");
}
}  // namespace noop_census

void noop_census_record(Kind kind, int slot, const u64 (&args)[6])
{
    using namespace noop_census;
    auto* self = reinterpret_cast<Wrapper*>(args[0]);
    if (kind == list_kind && (slot == 12 || slot == 13 || slot == 14 || slot == x2w_list_slots::execute_indirect_slot))
        noop_census_list_command(self, slot);
    const bool after = kind == list_kind && InterlockedExchange(&self->own().noop_previous, 0) == 1;
    unsigned types[6];
    u64 key = u64{1} << 60;  // Reserve zero for an unclaimed row.
    for (unsigned i = 0; i < 6; ++i) {
        types[i] = i == 0 ? wrapper + kind : classify(self, kind, args[i]);
        key |= u64{types[i]} << (10 * i);
    }
    auto& tally = (kind == list_kind ? lists : devices)[slot];
    for (unsigned i = 0; i < capacity; ++i) {
        auto& shape = tally.shapes[i];
        u64 seen = shape.key.load(std::memory_order_acquire);
        const bool first = !seen && shape.key.compare_exchange_strong(seen, key, std::memory_order_acq_rel);
        if (!first && seen != key) continue;
        shape.calls.fetch_add(1, std::memory_order_release);
        if (after) shape.credits[after_pipeline].fetch_add(1, std::memory_order_relaxed);
        if (kind == list_kind) InterlockedExchange(&self->own().noop_previous, static_cast<LONG>(2 + slot * capacity + i));
        // The thread that claims the row is the only writer. Counters can advance even if
        // this thread is preempted during sampling. Reports read bytes after publication.
        if (first) {
            for (unsigned a = 0; a < 6; ++a)
                if (types[a] == readable)
                    shape.sample_bytes[a] = static_cast<unsigned char>(read(args[a], shape.sample[a], 32));
            shape.sampled.store(true, std::memory_order_release);
        }
        return;
    }
    tally.overflow.fetch_add(1, std::memory_order_relaxed);
}

void noop_census_list_command(Wrapper* list, int slot)
{
    using namespace noop_census;
    const auto previous = InterlockedExchange(&list->own().noop_previous, slot == 25 ? 1 : 0);
    if (previous < 2) return;
    const unsigned credit = slot == 12 ? draw : slot == 13 ? indexed : slot == 14 ? dispatch :
                            slot == x2w_list_slots::execute_indirect_slot ? indirect : credit_count;
    if (credit != credit_count) {
        const auto row = previous - 2;
        lists[row / capacity].shapes[row % capacity].credits[credit].fetch_add(1, std::memory_order_relaxed);
    }
}

void report_noop_census()
{
    using namespace noop_census;
    for (const auto* slots : {lists, devices}) {
        const char* label = slots == lists ? "" : " device";
        for (int n = 0; n < slot_count; ++n) {
            const auto& slot = slots[n];
            const auto overflow = slot.overflow.load(std::memory_order_relaxed);
            u64 counts[capacity], total = overflow;
            unsigned rows = 0;
            for (unsigned i = 0; i < capacity; ++i) {
                counts[i] = slot.shapes[i].calls.load(std::memory_order_acquire);
                total += counts[i];
                rows += counts[i] != 0;
            }
            if (!total) continue;
            log("x2w-noop census%s slot %d: calls %llu, shapes %u, overflow %llu", label, n, total, rows, overflow);
            for (unsigned i = 0; i < capacity; ++i) {
                if (!counts[i]) continue;
                const auto& shape = slot.shapes[i];
                const auto key = shape.key.load(std::memory_order_relaxed);
                const bool sampled = shape.sampled.load(std::memory_order_acquire);
                char args[6][104]{};
                for (unsigned a = 0; a < 6; ++a) {
                    format_arg(args[a], sizeof(args[a]), static_cast<unsigned>((key >> (10 * a)) & 1023));
                    if (!sampled || !shape.sample_bytes[a]) continue;
                    auto size = std::strlen(args[a]);
                    args[a][size++] = ':';
                    for (unsigned b = 0; b < shape.sample_bytes[a]; ++b, size += 2)
                        std::snprintf(args[a] + size, sizeof(args[a]) - size, "%02x", unsigned{shape.sample[a][b]});
                }
                log("x2w-noop census%s slot %d shape %u: calls %llu, before draw %llu, before indexed %llu, "
                    "before dispatch %llu, before indirect %llu, after pipeline %llu, args %s %s %s %s %s %s",
                    label, n, i, counts[i], shape.credits[draw].load(), shape.credits[indexed].load(),
                    shape.credits[dispatch].load(), shape.credits[indirect].load(), shape.credits[after_pipeline].load(),
                    args[0], args[1], args[2], args[3], args[4], args[5]);
            }
        }
    }
}
struct NoopCensusExit {
    ~NoopCensusExit() { report_noop_census(); }
} noop_census_exit;
