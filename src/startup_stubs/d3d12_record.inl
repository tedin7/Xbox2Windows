// Record for replay (docs/startup-runtime.md#d3d12-replay-record). X2W_REPLAY_RECORD=all | <start s>,<seconds> |
// f<first present>,<presents> writes every call the title makes into the stand-in's objects (every kind, every wrapped
// slot) to X2W_REPLAY_RECORD_FILE (default replay.x2wr): the call with its register and stack arguments and return value,
// the memory its pointer arguments name (deduplicated blobs from declared argument ranges), and the wrapper an output
// pointer received. A replay feeds the same calls to this stand-in, so the runtime and the translator run for real.
//
// Each thread appends to its own buffer (no lock shared between title threads per call: a shared lock would serialise
// them and hide the cost being recorded); full buffers are written under the file lock. Calls are ordered by a global
// sequence number taken before the call. The opt-in page scan only runs before slots without float arguments.
// Float arguments in xmm registers reach the real slot unchanged (their values are not recorded yet).
//
// File: the capture chunk framing (tag, header bytes, data bytes). XREC (version, timer frequency), CALL, ARGM (argument
// memory: seq, arg, address, bytes, blob hash), BLOB (hash, bytes; once per content), OUTP (seq, arg, the pointer an
// output argument holds after the call), PRES (seq, present), XEND (calls, blobs, bytes, dropped).
#include "startup_stubs/replay_pages.h"

namespace replay_record {

thread_local u64 call_seq;

constexpr int wide_slots = 256;   // device and command-list slots wrapped
constexpr int narrow_slots = 64;  // slots wrapped for the other kinds
constexpr std::size_t flush_bytes = 1u << 20;
constexpr std::size_t argument_memory_limit = 1u << 20;

void* original[kind_count][wide_slots];
std::atomic<bool> active{false};
// A recording was installed in this process (set once by install(), never cleared, so it stays set after the recording
// stops). The submission dump reads it: the recorder writes dump rows only while its window is open; a replay
// (nothing installed) writes them all.
bool armed = false;
static_assert(device_kind == x2w_replay_slots::device && list_kind == x2w_replay_slots::list);
bool scan_submit_pages;
LONG64 volatile next_seq;
LONG volatile presents;
LONG first_present = 0, last_present = 0x7fffffff;
double start_s = 0.0, end_s = 1e300;
LARGE_INTEGER started{}, frequency{};
std::atomic<HANDLE> file{INVALID_HANDLE_VALUE};
std::atomic<bool> exiting{false};
// Lock order: a thread buffer's lock, then file_lock. blob_lock and buffers_lock are taken alone.
SRWLOCK file_lock = SRWLOCK_INIT, blob_lock = SRWLOCK_INIT, buffers_lock = SRWLOCK_INIT;
LONG64 volatile calls_written, blobs_stored;
u64 bytes_written;
LONG volatile stop_requested, cap_reached;
u64 max_bytes = 2048ull << 20;  // X2W_REPLAY_MAX_MIB: the recording stops (with its end chunk) once the file is this large
// X2W_REPLAY_PREFIX_MAX_MIB (0 = off, the legacy single cap): the prefix before the window has its own cap and the window
// cap above counts from the size the file had when the window opened, so a large prefix cannot eat the window's room.
u64 prefix_max_bytes = 0, window_base_bytes = 0;
bool window_base_set = false;
// X2W_REPLAY_PREFIX_GENERIC_KIB (default 16, 0 = no limit): before the window, a call without a declared layout copies at
// most this much memory behind each pointer argument. A size-looking next argument used to ask for up to 4 MiB per pointer,
// and the unique blobs of ~6000 prefix calls filled a 512 MiB prefix cap by 250 s.
u64 prefix_generic_bytes = 16u << 10;
// Unique blob bytes recorded before the window, per call kind and slot: the log at stop names the biggest.
LONG64 volatile prefix_slot_bytes[kind_count][wide_slots];
// The thunk checks active before doing any recorder work. Window and prefix
// selection is published by present(); the prefix ends when the window opens.
std::atomic<bool> window_open{false}, prefix_open{false};
// The window's base image of title memory is written before the window opens, a slice per present (see
// base_image_holds_window): its bytes are the window's share of the caps, as when it was written at the window's start.
bool base_warming;     // the window's time has come and its base image is being written
u64 base_prefix_bytes; // the base image's bytes written so far before the window
inline bool cap_exceeded(u64 extra = 0)
{
    const u64 total = bytes_written + extra;
    if (!prefix_max_bytes) return total >= max_bytes;
    if (!window_base_set) return prefix_open.load() ? total - base_prefix_bytes >= prefix_max_bytes : total >= max_bytes;
    return total - window_base_bytes >= max_bytes;
}
std::atomic<unsigned> memory_epoch{0};
constexpr LONG memory_cache_presents = 16;
// "slow": the window opens at the first present after slow_after_s whose gap to the previous present is at least
// slow_gap_s, or at slow_fallback_s; it lasts slow_seconds.
bool slow_mode;
double slow_after_s = 120, slow_gap_s = 0.5, slow_seconds = 10, slow_fallback_s = 180, last_present_s = -1;
// "d<draws>,<seconds>": the window opens at the first present after draw_trigger translated draws in total since the
// start, the rule of X2W_DEBUG_PRESENT_FROM_DRAWS (a title's play comes after its loading screens' draws; a count per
// frame does not tell them apart: a tester's title drew at most 24 a frame on its loading screen and about 22 in play).
// It lasts draw_seconds. Translated draws need X2W_D3D12_RENDER. While it waits, every 600 presents the total so far and
// the largest frame are logged, so the count can be checked.
bool draw_mode;
LONG draw_trigger;
double draw_seconds;
LONG draws_at_present, draws_most;
constexpr LONG draw_log_presents = 600;

struct ThreadBuffer {
    SRWLOCK lock = SRWLOCK_INIT;
    std::vector<unsigned char> bytes;
};
std::vector<ThreadBuffer*>& thread_buffers() { static std::vector<ThreadBuffer*> v; return v; }
std::unordered_set<u64>& stored_blobs() { static std::unordered_set<u64> s; return s; }  // blob_lock

u64 volatile whole_bytes;  // the file size after the last whole chunk: where a recording cut off at exit is closed
// file_lock keeps the handle open during writes. Detach can cut the file after a writer dies holding the lock.
void write_raw(const void* data, std::size_t bytes)
{
    DWORD written = 0;
    while (bytes && file != INVALID_HANDLE_VALUE) {
        const DWORD part = bytes > (16u << 20) ? (16u << 20) : static_cast<DWORD>(bytes);
        if (!WriteFile(file, data, part, &written, nullptr) || !written) break;
        data = static_cast<const char*>(data) + written;
        bytes -= written;
        bytes_written += written;
    }
}
void append_chunk(std::vector<unsigned char>& out, const char* tag, const void* header, std::size_t header_bytes,
                  const void* data, u64 data_bytes)
{
    struct { char tag[4]; std::uint32_t header_bytes; u64 data_bytes; } head{};
    std::memcpy(head.tag, tag, 4);
    head.header_bytes = static_cast<std::uint32_t>(header_bytes);
    head.data_bytes = data_bytes;
    const auto* h = reinterpret_cast<const unsigned char*>(&head);
    out.insert(out.end(), h, h + sizeof(head));
    if (header_bytes) out.insert(out.end(), static_cast<const unsigned char*>(header), static_cast<const unsigned char*>(header) + header_bytes);
    if (data_bytes) out.insert(out.end(), static_cast<const unsigned char*>(data), static_cast<const unsigned char*>(data) + data_bytes);
}
// Writes a buffer's chunks; file_lock held.
void drain(ThreadBuffer& b)
{
    write_raw(b.bytes.data(), b.bytes.size());
    if (b.bytes.size()) whole_bytes = bytes_written;  // a buffer holds whole chunks
    b.bytes.clear();
}

ThreadBuffer& my_buffer()
{
    thread_local ThreadBuffer* mine = nullptr;
    if (!mine) {
        mine = new ThreadBuffer;
        AcquireSRWLockExclusive(&buffers_lock);
        thread_buffers().push_back(mine);
        ReleaseSRWLockExclusive(&buffers_lock);
    }
    return *mine;
}

double seconds_now()
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return static_cast<double>(now.QuadPart - started.QuadPart) / static_cast<double>(frequency.QuadPart);
}
bool in_window()
{
    const LONG present = presents;
    if (present < first_present || present >= last_present) return false;
    const double t = seconds_now();
    return t >= start_s && t < end_s;
}
void update_window()
{
    window_open = in_window();
    if (window_open) {
        if (!window_base_set) {
            window_base_bytes = prefix_open ? bytes_written - base_prefix_bytes : 0;
            window_base_set = true;
        }
        prefix_open = false;
    }
}

// The calls recorded before the window: every device call (creations, views, descriptor copies), heap starts, Map and
// GPU addresses, fence and queue signals, and every object's QueryInterface/AddRef/Release.
constexpr bool prefix_slot(int kind, int slot)
{
    return kind == device_kind || slot <= 2 || (kind == descriptor_heap_kind && (slot == 9 || slot == 10)) ||
           (kind == resource_kind && (slot == 8 || slot == 9 || slot == 11)) || (kind == fence_kind && slot == 10) ||
           (kind == queue_kind && slot == 14);
}

u64 content_hash(const unsigned char* data, std::size_t bytes)
{
    u64 hash = 1469598103934665603ull ^ bytes;
    for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ data[i]) * 1099511628211ull;
    return hash;
}

bool committed_region_readable(const MEMORY_BASIC_INFORMATION& info)
{
    const DWORD protection = info.Protect & 0xff;
    return info.State == MEM_COMMIT && !(info.Protect & PAGE_GUARD) &&
        (protection == PAGE_READONLY || protection == PAGE_READWRITE || protection == PAGE_WRITECOPY ||
         protection == PAGE_EXECUTE_READ || protection == PAGE_EXECUTE_READWRITE || protection == PAGE_EXECUTE_WRITECOPY);
}

// Cache only the queried region within its 64 KiB granule. Page protection
// and commitment can change on smaller boundaries inside a granule.
std::size_t committed_readable_prefix(u64 start, std::size_t bytes)
{
    constexpr u64 granule = 0x10000;
    constexpr u64 user_end = 0x800000000000ull;
    if (start < granule || start >= user_end || bytes > user_end - start) return 0;
    struct Region {
        u64 first, end;
        bool readable;
    };
    struct Cache {
        unsigned epoch;
        Region entries[256];
    };
    static thread_local Cache cache{};
    const unsigned epoch = memory_epoch.load(std::memory_order_relaxed);
    if (cache.epoch != epoch) cache = Cache{epoch, {}};
    const u64 end = start + bytes;
    u64 at = start;
    while (at < end) {
        const u64 base = at & ~(granule - 1);
        Region& region = cache.entries[(base / granule) % std::size(cache.entries)];
        if (at < region.first || at >= region.end) {
            MEMORY_BASIC_INFORMATION info{};
            if (!VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info))) break;
            const u64 first = reinterpret_cast<u64>(info.BaseAddress);
            const bool readable = committed_region_readable(info);
            const Region found{(std::max)(first, base), (std::min)(first + info.RegionSize, base + granule), readable};
            // Only readable spans are kept: memory reserved now is often committed later (a title heap grows inside a
            // granule), and the prefix can run for minutes without a present to expire the cache.
            if (!found.readable) break;
            region = found;
        }
        at = (std::min)(end, region.end);
    }
    return static_cast<std::size_t>(at - start);
}

bool readable_range(u64 address, std::size_t bytes)
{
    return bytes && committed_readable_prefix(address, bytes) == bytes;
}

bool copy_readable(u64 address, void* destination, std::size_t bytes)
{
    // A mapping can change after VirtualQuery. Let the kernel copy and report
    // failure instead of dereferencing the address on a title thread.
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), destination, bytes, &got) &&
        got == bytes;
}

constexpr int call_args = 11;      // raw argument words after rcx, all recorded (format 3)
constexpr std::uint32_t rcx_arg = 15;  // ARGM argument index of a struct-return buffer

struct Call {
    u64 seq;
    std::uint32_t thread;
    std::uint16_t kind, slot;
    u64 object;
    u64 args[call_args];
    u64 ret;
    double seconds;
    std::int32_t present, pad;
};

struct ArgumentCapture {
    ThreadBuffer& buffer;
    const Call& call;
    std::size_t remaining = argument_memory_limit;

    std::vector<unsigned char> memory(std::uint32_t arg, u64 address, u64 bytes, unsigned depth = 0)
    {
        if (bytes > remaining || !readable_range(address, static_cast<std::size_t>(bytes))) return {};
        remaining -= static_cast<std::size_t>(bytes);
        std::vector<unsigned char> copy(static_cast<std::size_t>(bytes));
        if (!copy_readable(address, copy.data(), copy.size())) return {};

        const u64 hash = content_hash(copy.data(), copy.size());
        struct { u64 seq; std::uint32_t arg, depth; u64 address, bytes, hash; } header{call.seq, arg, depth, address, bytes, hash};
        append_chunk(buffer.bytes, "ARGM", &header, sizeof(header), nullptr, 0);
        AcquireSRWLockExclusive(&blob_lock);
        const bool fresh = stored_blobs().insert(hash).second;
        ReleaseSRWLockExclusive(&blob_lock);
        if (fresh) {
            struct { u64 hash, bytes; } blob{hash, bytes};
            append_chunk(buffer.bytes, "BLOB", &blob, sizeof(blob), copy.data(), copy.size());
            InterlockedIncrement64(&blobs_stored);
            if (prefix_open.load(std::memory_order_relaxed) && !window_open.load(std::memory_order_relaxed) &&
                static_cast<int>(call.kind) < kind_count && static_cast<int>(call.slot) < wide_slots)
                InterlockedAdd64(&prefix_slot_bytes[call.kind][call.slot], static_cast<LONG64>(bytes));
        }
        return copy;
    }

    void arg(unsigned i, u64 bytes) { memory(i, call.args[i], bytes); }

    // The committed readable start of [address, address + wanted), within the call's remaining budget.
    std::vector<unsigned char> prefix(std::uint32_t arg, u64 address, std::size_t wanted, unsigned depth)
    {
        if (address < 0x10000 || address >= 0x7fffffffffffull || as_wrapper(reinterpret_cast<const void*>(address)))
            return {};
        const std::size_t bytes = committed_readable_prefix(address, (std::min)(wanted, remaining));
        return bytes ? memory(arg, address, bytes, depth) : std::vector<unsigned char>{};
    }

    // A failed call records no output: the player does not play a creation recorded as a failure.
    void output(unsigned i)
    {
        if (FAILED(static_cast<HRESULT>(call.ret))) return;
        const auto copy = memory(i, call.args[i], sizeof(u64));
        if (copy.empty()) return;
        u64 value;
        std::memcpy(&value, copy.data(), sizeof(value));
        if (!as_wrapper(reinterpret_cast<const void*>(value))) return;
        struct { u64 seq; std::uint32_t arg, pad; u64 value; } out{call.seq, i, 0, value};
        append_chunk(buffer.bytes, "OUTP", &out, sizeof(out), nullptr, 0);
    }

    void create(unsigned iid)
    {
        arg(iid, sizeof(GUID));
        output(iid + 1);
    }

    bool descriptor_arrays()
    {
        const u64* args = call.args;
        u64 sizes[6]{};
        u64 total = 0;
        for (const unsigned i : {1u, 2u, 4u, 5u}) {
            if (!args[i]) continue;
            const u64 count = static_cast<UINT>(args[i < 3 ? 0 : 3]);
            sizes[i] = count * ((i == 1 || i == 4) ? sizeof(u64) : sizeof(UINT));
            total += sizes[i];
        }
        if (total > remaining) return false;
        remaining -= static_cast<std::size_t>(total);
        std::vector<unsigned char> arrays(static_cast<std::size_t>(total));
        std::size_t offset = 0;
        for (const unsigned i : {1u, 2u, 4u, 5u}) {
            const auto bytes = static_cast<std::size_t>(sizes[i]);
            if (!bytes) continue;
            if (!readable_range(args[i], bytes) || !copy_readable(args[i], arrays.data() + offset, bytes)) return false;
            offset += bytes;
        }
        append_chunk(buffer.bytes, "DARR", &call.seq, sizeof(call.seq), arrays.data(), arrays.size());
        return true;
    }
};


void record_stream(ArgumentCapture& capture, unsigned arg)
{
    const auto copy = capture.memory(arg, capture.call.args[arg], sizeof(D3D12_PIPELINE_STATE_STREAM_DESC), 2);
    if (copy.empty()) return;
    D3D12_PIPELINE_STATE_STREAM_DESC desc;
    std::memcpy(&desc, copy.data(), sizeof(desc));
    const auto stream = capture.memory(arg | 0x200u, reinterpret_cast<u64>(desc.pPipelineStateSubobjectStream),
                                       desc.SizeInBytes, 1);
    for (std::size_t at = 0; stream.size() - at >= sizeof(UINT);) {
        UINT type;
        std::memcpy(&type, stream.data() + at, sizeof(type));
        const auto bytes = desktop_stream_size(type);
        if (!bytes || bytes > stream.size() - at) break;
        if ((type >= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS && type <= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS) ||
            type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS || type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS ||
            type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO) {
            D3D12_SHADER_BYTECODE blob;
            std::memcpy(&blob, stream.data() + at + 8, sizeof(blob));
            capture.memory(arg | (static_cast<std::uint32_t>(at / 8 + 2) << 8),
                           reinterpret_cast<u64>(blob.pShaderBytecode), blob.BytecodeLength);
        }
        at += bytes;
    }
}

// Device slots without a declared layout (pipeline descriptions of slots 57 and 67, command signatures, derived
// pipelines, footprints, residency arrays): every pointer argument as a readable window, sized by the word after it
// when that is a plausible size, the pointers in its first words one level further, and the wrapper it holds (OUTP).
// The player needs that memory and the created objects. Device calls are rare, and the call budget bounds them.
constexpr std::size_t generic_window_bytes = 0x100;
constexpr std::size_t generic_scan_bytes = 0x400;
constexpr u64 generic_hint_limit = 4u << 20;
constexpr std::uint32_t generic_followed_args = 7;  // later words are mostly the caller's stack: a short look only

void generic_memory(ArgumentCapture& capture, std::uint32_t arg, u64 address, u64 hint, unsigned depth)
{
    std::size_t wanted = hint > generic_window_bytes && hint <= generic_hint_limit
        ? static_cast<std::size_t>(hint) : generic_window_bytes;
    if (prefix_generic_bytes && wanted > prefix_generic_bytes && prefix_open.load(std::memory_order_relaxed) &&
        !window_open.load(std::memory_order_relaxed))
        wanted = static_cast<std::size_t>(prefix_generic_bytes);
    const auto copy = capture.prefix(arg, address, wanted, depth);
    if (!depth || copy.empty()) return;
    const std::size_t scan = (std::min)(copy.size(), generic_scan_bytes) / 8;
    for (std::size_t i = 0; i < scan; ++i) {
        u64 value = 0, next = 0;
        std::memcpy(&value, copy.data() + 8 * i, 8);
        if (value < 0x10000 || value >= 0x7fffffffffffull || (value & 3)) continue;
        if (i + 1 < scan) std::memcpy(&next, copy.data() + 8 * (i + 1), 8);
        generic_memory(capture, (arg & 0xffu) | 0x100u * static_cast<std::uint32_t>(i + 1), value, next, depth - 1);
    }
}

// Device slots 57 and 67 (description, IID, out pipeline): the stand-in keeps 0x300 bytes of the description
// (note_pipeline) and reads, inside it, (pointer, size) word pairs: state blocks, shader code, data blobs. Exactly those
// are recorded, one level deep, without the generic capture's look into the blobs (which spent ~0.8 MB of unrelated
// memory per creation on a title and then ran into the per-slot bound).
constexpr std::size_t pipeline_desc_bytes = 0x300;
constexpr u64 pipeline_part_limit = 1u << 20;

void record_pipeline_desc(ArgumentCapture& capture)
{
    const auto desc = capture.prefix(0, capture.call.args[0], pipeline_desc_bytes, 1);
    const std::size_t words = desc.size() / 8;
    for (std::size_t i = 0; i + 1 < words; ++i) {
        u64 address = 0, bytes = 0;
        std::memcpy(&address, desc.data() + 8 * i, 8);
        std::memcpy(&bytes, desc.data() + 8 * (i + 1), 8);
        if (address < 0x10000 || address >= 0x7fffffffffffull || !bytes || bytes > pipeline_part_limit) continue;
        if (as_wrapper(reinterpret_cast<const void*>(address))) continue;
        if (!capture.memory(0x100u * static_cast<std::uint32_t>(i + 1), address, bytes).empty()) ++i;  // a size is no pointer
    }
    capture.arg(1, sizeof(GUID));
    capture.output(2);
}

// Generic capture is bounded per device slot: past generic_slot_budget bytes a slot's calls are recorded without that
// memory (logged once per slot), so one hot slot whose stack words look like pointers cannot fill the file.
constexpr LONG64 generic_slot_budget = 32ll << 20;
LONG64 volatile generic_spent[wide_slots];
LONG volatile generic_bounded[wide_slots];

void record_generic(ArgumentCapture& capture)
{
    const Call& call = capture.call;
    const int slot = call.slot < wide_slots ? call.slot : 0;
    if (InterlockedCompareExchange64(&generic_spent[slot], 0, 0) >= generic_slot_budget) {
        if (!InterlockedExchange(&generic_bounded[slot], 1))
            log("x2w-replay record: device slot %d spent its %lld MiB of generic argument memory; its later calls are "
                "recorded without it", slot, static_cast<long long>(generic_slot_budget >> 20));
        return;
    }
    const std::size_t before = capture.remaining;
    for (std::uint32_t i = 0; i < call_args; ++i) {
        const u64 value = call.args[i];
        if (value < 0x10000 || value >= 0x7fffffffffffull || as_wrapper(reinterpret_cast<const void*>(value))) continue;
        u64 held = 0;
        if (readable_range(value, sizeof(held)) && copy_readable(value, &held, sizeof(held)) &&
            as_wrapper(reinterpret_cast<const void*>(held))) {
            struct { u64 seq; std::uint32_t arg, pad; u64 value; } out{call.seq, i, 0, held};
            append_chunk(capture.buffer.bytes, "OUTP", &out, sizeof(out), nullptr, 0);
        }
        const bool followed = i < generic_followed_args;
        generic_memory(capture, i, value, followed && i + 1 < generic_followed_args ? call.args[i + 1] : 0, followed ? 2 : 0);
    }
    InterlockedAdd64(&generic_spent[slot], static_cast<LONG64>(before - capture.remaining));
}

void record_arguments(ArgumentCapture& capture)
{
    const Call& call = capture.call;
    const auto& a = call.args;
    if (const auto* layout = x2w_replay_slots::find(call.kind, call.slot); layout && layout->known_memory) {
        for (unsigned i = 0; i < std::size(layout->memory); ++i) {
            if (!layout->memory[i]) continue;
            if (static_cast<int>(i) == layout->follow) record_stream(capture, i);
            else if (static_cast<int>(i) == layout->wrapper_output) capture.output(i);
            else capture.arg(i, layout->memory[i]);
        }
        return;
    }
    if (call.slot == 0) {
        capture.create(0);  // QueryInterface
        return;
    }
    switch (call.kind) {
    case device_kind:
        switch (call.slot) {
        case 8: capture.arg(0, sizeof(D3D12_COMMAND_QUEUE_DESC)); capture.create(1); break;
        case 9: capture.create(1); break;
        case 12: capture.create(4); break;
        case 14: capture.arg(0, sizeof(D3D12_DESCRIPTOR_HEAP_DESC) + 2 * sizeof(u64)); capture.create(1); break;
        case 16: capture.arg(1, a[2]); capture.create(3); break;
        case 17: capture.arg(0, sizeof(D3D12_CONSTANT_BUFFER_VIEW_DESC)); break;
        case 18: capture.arg(1, sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)); break;
        case 19: capture.arg(2, sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC)); break;
        case 20: capture.arg(1, sizeof(D3D12_RENDER_TARGET_VIEW_DESC)); break;
        case 21: capture.arg(1, sizeof(D3D12_DEPTH_STENCIL_VIEW_DESC)); break;
        case 22: capture.arg(0, sizeof(D3D12_SAMPLER_DESC)); break;
        case 27:
        case 76:
        case 127:
            capture.arg(0, sizeof(D3D12_HEAP_PROPERTIES));
            [[fallthrough]];
        case 29:
            capture.arg(2, sizeof(D3D12_RESOURCE_DESC));
            capture.arg(4, sizeof(D3D12_CLEAR_VALUE));
            capture.create(call.slot == 127 ? 8 : 5);
            break;
        case 28: capture.arg(0, sizeof(D3D12_HEAP_DESC)); capture.create(1); break;
        case 30:
            capture.arg(0, sizeof(D3D12_RESOURCE_DESC));
            capture.arg(2, sizeof(D3D12_CLEAR_VALUE));
            capture.create(3);
            break;
        case 36: capture.create(2); break;
        case 39: capture.arg(0, sizeof(D3D12_QUERY_HEAP_DESC)); capture.create(1); break;
        case 25:  // GetResourceAllocationInfo(mask, count, descs), returned by value: the descriptions
            if (static_cast<UINT>(a[2]) <= 64) capture.arg(3, static_cast<u64>(static_cast<UINT>(a[2])) * sizeof(D3D12_RESOURCE_DESC));
            break;
        case 108:  // GetResourceAllocationInfo1(mask, count, descs, info1), returned by value: the descriptions and the
                   // per-resource array the stand-in writes (the title's stack otherwise)
            if (static_cast<UINT>(a[2]) <= 64) {
                const u64 count = static_cast<UINT>(a[2]);
                capture.arg(3, count * sizeof(D3D12_RESOURCE_DESC));
                if (a[4]) capture.arg(4, count * sizeof(D3D12_RESOURCE_ALLOCATION_INFO1));
            }
            break;
        case 55:  // CreateDerivedPipelineStateX(base, patch count, patches, IID, out)
            if (static_cast<UINT>(a[1]) <= 64) capture.arg(2, static_cast<u64>(static_cast<UINT>(a[1])) * x2w_derived_blend::record_size);
            capture.arg(3, sizeof(GUID));
            capture.output(4);
            break;
        case 133:  // CreatePlacedResourceX(title address, desc, state, clear, IID, out) without a desktop heap placement
            capture.arg(1, sizeof(D3D12_RESOURCE_DESC));
            capture.arg(3, sizeof(D3D12_CLEAR_VALUE));
            capture.arg(4, sizeof(GUID));
            capture.output(5);
            break;
        case 57:
        case 67: record_pipeline_desc(capture); break;
        case 1:   // AddRef and Release take no argument: the words are the caller's stack
        case 2:
        case 15:  // descriptor increment size: no memory
        case 23:  // CopyDescriptors: the exact DARR arrays only
        case 24:  // CopyDescriptorsSimple: CPU handles, which the player translates
            break;
        default: record_generic(capture); break;
        }
        break;
    case list_kind:
        switch (call.slot) {
        case 16:  // CopyTextureRegion(destination location, x, y, z, source location, box)
            capture.arg(0, sizeof(D3D12_TEXTURE_COPY_LOCATION));
            capture.arg(4, sizeof(D3D12_TEXTURE_COPY_LOCATION));
            capture.arg(5, sizeof(D3D12_BOX));
            break;
        case 32:  // SetDescriptorHeaps(count, heaps): the wrappers, which the player translates
            capture.arg(1, static_cast<u64>(static_cast<UINT>(a[0])) * sizeof(void*));
            break;
        case 39:  // Set{Compute,Graphics}Root32BitConstants(parameter, count, values, offset)
        case 40: capture.arg(2, static_cast<u64>(static_cast<UINT>(a[1])) * sizeof(UINT)); break;
        case 47: capture.arg(0, sizeof(D3D12_INDEX_BUFFER_VIEW)); break;  // IASetIndexBuffer(view)
        case 48:  // IASetVertexBuffers(start, count, views)
            capture.arg(2, static_cast<u64>(static_cast<UINT>(a[1])) * sizeof(D3D12_VERTEX_BUFFER_VIEW));
            break;
        case 21: capture.arg(1, static_cast<u64>(static_cast<UINT>(a[0])) * sizeof(D3D12_VIEWPORT)); break;
        case 22: capture.arg(1, static_cast<u64>(static_cast<UINT>(a[0])) * sizeof(D3D12_RECT)); break;
        case 50:
            capture.arg(1, (static_cast<BOOL>(a[2]) && static_cast<UINT>(a[0]) ? 1ull : static_cast<UINT>(a[0])) * sizeof(u64));
            capture.arg(3, sizeof(u64));
            break;
        case 51:  // ClearDepthStencilView(handle, flags, depth, stencil, rect count, rects)
            capture.arg(5, static_cast<u64>(static_cast<UINT>(a[4])) * sizeof(D3D12_RECT));
            break;
        case 52:
            capture.arg(1, 4 * sizeof(float));
            capture.arg(3, static_cast<u64>(static_cast<UINT>(a[2])) * sizeof(D3D12_RECT));
            break;
        }
        break;
    case queue_kind:
        if (call.slot == 10) capture.arg(1, static_cast<u64>(static_cast<UINT>(a[0])) * sizeof(u64));
        if (call.slot == 32) {
            const auto stride = x2w_startup::present_plane_stride;
            const auto planes = capture.memory(1, a[1], static_cast<u64>(static_cast<UINT>(a[0])) * stride, 1);
            for (std::size_t at = 0; at < planes.size(); at += stride) {
                u64 count, resources;
                std::memcpy(&count, planes.data() + at + 8, 8);
                std::memcpy(&resources, planes.data() + at + 16, 8);
                if (count) capture.memory(1u | (static_cast<std::uint32_t>(at / 8 + 3) << 8), resources, sizeof(u64));
            }
        }
        break;
    case descriptor_heap_kind:
        if ((call.slot == 9 || call.slot == 10) && as_wrapper(reinterpret_cast<const void*>(a[0])) &&
            !as_wrapper(reinterpret_cast<const void*>(call.object))) capture.memory(rcx_arg, call.object, sizeof(u64));
        break;
    case resource_kind:
        // Map(subresource, read range, out CPU address) and Unmap(subresource, written range): null ranges stay null.
        if ((call.slot == 8 || call.slot == 9) && call.args[1]) capture.arg(1, sizeof(D3D12_RANGE));
        if (call.slot == 8) capture.arg(2, sizeof(void*));  // Map's output CPU address
        break;
    }
}

// After the call: the call, then for each pointer argument the wrapper its memory holds (OUTP: a creation's result
// when the replayer has not seen that wrapper yet, else an input such as a list array) and its memory (ARGM).
// HPLC describes the desktop heap used by a successful slot 133 placement.
struct HeapPlacement {
    u64 id, base, bytes, offset;
};
thread_local HeapPlacement* active_placement = nullptr;

void record_call(int kind, int slot, u64 seq, u64 object, const u64 (&args)[call_args], u64 ret, bool descriptor_copy, const HeapPlacement& placement)
{
    const u64 view_id = take_view_id();
    ThreadBuffer& b = my_buffer();
    AcquireSRWLockExclusive(&b.lock);
    if (!active.load(std::memory_order_relaxed)) {
        ReleaseSRWLockExclusive(&b.lock);
        return;
    }
    Call call{seq, GetCurrentThreadId(), static_cast<std::uint16_t>(kind), static_cast<std::uint16_t>(slot), object, {}, ret,
              seconds_now(), presents, 0};
    std::memcpy(call.args, args, sizeof(call.args));
    append_chunk(b.bytes, "CALL", &call, sizeof(call), nullptr, 0);
    // Prefix creates can supply the view ids read through title memory in the window.
    if (view_id) {
        const struct { u64 seq; std::uint32_t id, pad; } header{seq, static_cast<std::uint32_t>(view_id), 0};
        append_chunk(b.bytes, "VIDS", &header, sizeof(header), nullptr, 0);
    }
    if (kind == device_kind && slot == 14 && SUCCEEDED(static_cast<HRESULT>(ret))) {
        // Copied, never dereferenced: the recorder must not fault on title memory.
        u64 created = 0, extra[2] = {};
        const auto* heap = copy_readable(args[2], &created, sizeof(created)) ? as_wrapper(reinterpret_cast<const void*>(created)) : nullptr;
        // The wrapper has already validated the extension against the heap's descriptor count and stride.
        if (heap && heap->own().kind == descriptor_heap_kind && heap->own().title_cpu &&
            copy_readable(args[0] + sizeof(D3D12_DESCRIPTOR_HEAP_DESC), extra, sizeof(extra)) &&
            extra[0] == heap->own().title_cpu && extra[1]) {
            const u64 backing[] = {seq, extra[0], extra[1]};
            append_chunk(b.bytes, "DHPM", backing, sizeof(backing), nullptr, 0);
        }
    }
    ArgumentCapture capture{b, call};
    const bool typed_arrays = descriptor_copy && capture.descriptor_arrays();
    if (descriptor_copy && !typed_arrays) {
        static LONG unrecorded;
        if (InterlockedIncrement(&unrecorded) <= 8)
            log("x2w-replay record: CopyDescriptors arrays of call %llu not recorded exactly (unreadable or over the size cap); "
                "the replay will skip that copy", static_cast<unsigned long long>(seq));
    }
    if (placement.id && SUCCEEDED(static_cast<HRESULT>(ret))) {
        const struct { u64 seq; HeapPlacement placement; } header{seq, placement};
        append_chunk(b.bytes, "HPLC", &header, sizeof(header), nullptr, 0);
        capture.arg(1, sizeof(D3D12_RESOURCE_DESC));
        capture.arg(4, sizeof(GUID));
        capture.output(5);
    } else {
        record_arguments(capture);
    }
    InterlockedIncrement64(&calls_written);
    if (b.bytes.size() >= flush_bytes) {
        AcquireSRWLockExclusive(&file_lock);
        drain(b);
        if (cap_exceeded() && !InterlockedExchange(&cap_reached, 1)) InterlockedExchange(&stop_requested, 1);
        ReleaseSRWLockExclusive(&file_lock);
    }
    ReleaseSRWLockExclusive(&b.lock);
}

void stop(bool at_exit, bool from_present = false);
bool draw_pages;
double draw_pages_ms = 100.0;
bool extra_title_pages;  // set before installing the recording thunks
void remember_heap_start(u64 object, u64 second);
u64 pages_before_call(int kind, int slot, u64 rcx, u64 rdx, u64 r8, u64 r9, bool root, bool submit, bool scan);
void remember_cbv_pages(u64 desc);

template <int K, int N>
__declspec(noinline) u64 thunk(u64 rcx, u64 rdx, u64 r8, u64 r9, u64 s0, u64 s1, u64 s2, u64 s3, u64 s4, u64 s5, u64 s6,
                               u64 s7)
{
    // Eleven arguments after rcx reach the real slot (D3D12's longest, UpdateTileMappings, has ten): stack words past
    // the caller's own arguments are read and passed on unused, which the x64 convention allows (the caller reserves
    // the home area; a slot reads only what it declares). All eleven are recorded.
    using Slot = u64 (*)(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);
    const auto call = reinterpret_cast<Slot>(original[K][N]);
    if (!active.load(std::memory_order_relaxed) ||
        (!window_open.load(std::memory_order_relaxed) && !(prefix_slot(K, N) && prefix_open.load(std::memory_order_relaxed))))
        return call(rcx, rdx, r8, r9, s0, s1, s2, s3, s4, s5, s6, s7);
    // CopyDescriptors takes no float arguments, so its counted arrays are read from the call's own registers.
    constexpr bool descriptor_copy = K == device_kind && N == 23;
    // These slots have no floating-point arguments, so a page scan may run before the real call (opt-in: named title
    // ranges X2W_REPLAY_TITLE_RANGES, or a scan before each submit X2W_REPLAY_PAGES_AT_SUBMIT).
    // Draw-page captures use the same sequence lock as scans.
    constexpr bool root = K == list_kind && N >= 41 && N <= 46;
    constexpr bool submit = K == queue_kind && N == 10;
    constexpr bool consumes_pages = root || (K == list_kind && ((N >= 12 && N <= 19) || N == 61)) ||
                                    (K == device_kind && (N == 23 || N == 24)) || submit;
    constexpr bool draw_page_slot = K == list_kind &&
        (N == 12 || N == 13 || N == 14 || N == 35 || N == 36 || N == 61 || (N >= 41 && N <= 48));
    u64 seq;
    if constexpr (consumes_pages || draw_page_slot) {
        const bool scan = consumes_pages && (extra_title_pages ||
            (submit && (scan_submit_pages || replay_dump_enabled()) && window_open));
        const bool capture = draw_page_slot && draw_pages && window_open.load(std::memory_order_relaxed);
        seq = scan || capture ? pages_before_call(K, N, rcx, rdx, r8, r9, root, submit, scan)
                              : static_cast<u64>(InterlockedIncrement64(&next_seq));
    } else {
        seq = static_cast<u64>(InterlockedIncrement64(&next_seq));
    }
    if constexpr (K == device_kind && N == 17)
        if (extra_title_pages) remember_cbv_pages(rdx);
    HeapPlacement placement{};
    HeapPlacement* previous = nullptr;
    if constexpr (K == device_kind && N == 133) {
        previous = active_placement;
        active_placement = &placement;
    }
    take_view_id();
    const u64 previous_seq = call_seq;
    call_seq = K == list_kind && window_open.load(std::memory_order_relaxed) ? seq : 0;
    const u64 ret = call(rcx, rdx, r8, r9, s0, s1, s2, s3, s4, s5, s6, s7);
    call_seq = previous_seq;
    if constexpr (K == device_kind && N == 133) active_placement = previous;
    const u64 args[call_args] = {rdx, r8, r9, s0, s1, s2, s3, s4, s5, s6, s7};
    record_call(K, N, seq, rcx, args, ret, descriptor_copy, placement);
    if constexpr (K == descriptor_heap_kind && N == 9)
        if (extra_title_pages) remember_heap_start(rcx, rdx);
    if (stop_requested) stop(false, true);  // after the window's last present: that Present call is in the file
    return ret;
}

template <int K, std::size_t... I>
void wrap_kind(std::index_sequence<I...>)
{
    ((tables[K][I] ? (original[K][I] = tables[K][I], tables[K][I] = reinterpret_cast<void*>(&thunk<K, static_cast<int>(I)>))
                   : nullptr), ...);
}
template <int K>
void wrap_one()
{
    if constexpr (K == device_kind || K == list_kind)
        wrap_kind<K>(std::make_index_sequence<wide_slots>{});
    else
        wrap_kind<K>(std::make_index_sequence<narrow_slots>{});
}
template <std::size_t... K>
void wrap_all(std::index_sequence<K...>)
{
    (wrap_one<static_cast<int>(K)>(), ...);
}

// Writes every thread's buffer. At process exit the other threads are gone, possibly holding their buffer's lock: those
// buffers are skipped (`at_exit`) instead of waited for.
void flush_all(bool at_exit)
{
    AcquireSRWLockShared(&buffers_lock);
    const std::vector<ThreadBuffer*> buffers = thread_buffers();
    ReleaseSRWLockShared(&buffers_lock);
    for (ThreadBuffer* b : buffers) {
        if (at_exit) {
            if (!TryAcquireSRWLockExclusive(&b->lock)) continue;
        } else {
            AcquireSRWLockExclusive(&b->lock);
        }
        AcquireSRWLockExclusive(&file_lock);
        drain(*b);
        ReleaseSRWLockExclusive(&file_lock);
        ReleaseSRWLockExclusive(&b->lock);
    }
}

// SHDR chunks: every file directly in X2W_SHADER_CACHE (requests and translations), header = its name (UTF-8), data =
// its bytes, so a replay on another machine has the translations this run had. Files over 16 MiB, and the files after
// 512 MiB in total, are left out (counted in the log). At process exit (DllMain, loader lock held) it stops after 5 s
// so a large cache cannot hold the exit up. file_lock held. whole_bytes is the file size after the last whole chunk.
void write_shader_cache(bool at_exit)
{
    // X2W_REPLAY_SHADERS=0: no SHDR chunks (the cache travels another way, e.g. with a tester run's mail parts).
    char shaders[8]{};
    if (GetEnvironmentVariableA("X2W_REPLAY_SHADERS", shaders, sizeof(shaders)) && shaders[0] == '0') {
        if (!at_exit) log("x2w-replay record: shader cache not written (X2W_REPLAY_SHADERS=0)");
        return;
    }
    // X2W_TEST_REPLAY_FINISHER_STALL_MS=<ms>: pause after each SHDR chunk to exercise process exit during finalization.
    char stall[16]{};
    const DWORD stall_length = at_exit ? 0 : GetEnvironmentVariableA("X2W_TEST_REPLAY_FINISHER_STALL_MS", stall, sizeof(stall));
    const DWORD stall_ms = stall_length && stall_length < sizeof(stall) ? std::strtoul(stall, nullptr, 10) : 0;
    wchar_t dir[MAX_PATH];
    const DWORD n = GetEnvironmentVariableW(L"X2W_SHADER_CACHE", dir, MAX_PATH);
    if (!n || n >= MAX_PATH - 4) return;
    wchar_t pattern[MAX_PATH];
    swprintf(pattern, MAX_PATH, L"%s\\*", dir);
    WIN32_FIND_DATAW found;
    const HANDLE find = FindFirstFileW(pattern, &found);
    if (find == INVALID_HANDLE_VALUE) return;
    constexpr u64 file_max = 16ull << 20, total_max = 512ull << 20;
    u64 files = 0, total = 0, left_out = 0;
    const ULONGLONG deadline = GetTickCount64() + (at_exit ? 5000 : 120000);
    std::vector<unsigned char> chunk;
    bool cut_short = false;
    do {
        if (found.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        if (!at_exit && exiting.load()) {
            cut_short = true;
            ++left_out;
            continue;
        }
        const u64 size = (static_cast<u64>(found.nFileSizeHigh) << 32) | found.nFileSizeLow;
        if (size > file_max || total + size > total_max || GetTickCount64() > deadline) {
            ++left_out;
            continue;
        }
        wchar_t path[MAX_PATH];
        swprintf(path, MAX_PATH, L"%s\\%s", dir, found.cFileName);
        const HANDLE f = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                     FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) continue;
        std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
        DWORD got = 0;
        const bool read = !size || (ReadFile(f, bytes.data(), static_cast<DWORD>(size), &got, nullptr) && got == size);
        CloseHandle(f);
        char name[MAX_PATH * 3];
        const int name_bytes = WideCharToMultiByte(CP_UTF8, 0, found.cFileName, -1, name, sizeof(name), nullptr, nullptr) - 1;
        if (!read || name_bytes <= 0) continue;
        chunk.clear();
        append_chunk(chunk, "SHDR", name, static_cast<std::size_t>(name_bytes), bytes.data(), size);
        write_raw(chunk.data(), chunk.size());
        whole_bytes = bytes_written;
        ++files;
        total += size;
        if (stall_ms) Sleep(stall_ms);
    } while (FindNextFileW(find, &found));
    FindClose(find);
    if (cut_short)
        log("x2w-replay record: shader cache cut short by process exit after %llu files", files);
    if (!at_exit)
        log("x2w-replay record: shader cache %llu files, %llu bytes, %llu left out", files, total, left_out);
}

// Ends the recording: every buffer is written, then the end chunk with the totals. Called once.
// Puts the title's slots back: after the recording ends no call goes through a thunk. A thread already inside one finds
// active cleared and calls straight through.
void unwrap_all()
{
    for (int k = 0; k < kind_count; ++k)
        for (int i = 0; i < wide_slots; ++i)
            if (original[k][i]) tables[k][i] = original[k][i];
}

void write_end_and_close(bool cut = false)
{
    std::vector<unsigned char> tail;
    struct { u64 calls, blobs, bytes, presents; } end{calls_written, blobs_stored, bytes_written, static_cast<u64>(presents)};
    append_chunk(tail, "XEND", &end, sizeof(end), nullptr, 0);
    write_raw(tail.data(), tail.size());
    FlushFileBuffers(file);
    if (cut) {
        LARGE_INTEGER end_offset;
        end_offset.QuadPart = static_cast<LONGLONG>(bytes_written);
        LARGE_INTEGER size;
        // Pending I/O from the terminated writer can extend the file past XEND.
        for (int retry = 0; retry < 3 && GetFileSizeEx(file, &size) && size.QuadPart > end_offset.QuadPart; ++retry) {
            if (SetFilePointerEx(file, end_offset, nullptr, FILE_BEGIN) && SetEndOfFile(file))
                FlushFileBuffers(file);
            Sleep(10);
        }
        if (!GetFileSizeEx(file, &size) || size.QuadPart != end_offset.QuadPart)
            log("x2w-replay record: could not verify the file size after the exit cut");
    }
    CloseHandle(file);
    file = INVALID_HANDLE_VALUE;
}

// When a present ends the recording, the file's tail (shader cache, end chunk) is written by a finisher thread: a cache
// of thousands of files takes minutes on a tester's disk and must not stall the title's presents. finished is set once
// the file is closed; a later stop (a fixture, FreeLibrary) waits for it. The exit path asks a running finisher
// to skip the rest of the shader cache and waits on its thread handle, which also signals if it was terminated.
HANDLE finished = nullptr;
static std::atomic<HANDLE> finisher_thread{nullptr};
bool stopped_in_prefix;

void close_cut_file();

void finish_file(bool at_exit)
{
    if (at_exit && !TryAcquireSRWLockExclusive(&file_lock)) {
        close_cut_file();
        return;
    }
    if (!at_exit) AcquireSRWLockExclusive(&file_lock);
    if (file == INVALID_HANDLE_VALUE) {
        ReleaseSRWLockExclusive(&file_lock);
        return;
    }
    const bool in_prefix = stopped_in_prefix;
    whole_bytes = bytes_written;
    write_shader_cache(at_exit);
    struct { u64 calls, bytes; } end{static_cast<u64>(calls_written), bytes_written};
    write_end_and_close();
    ReleaseSRWLockExclusive(&file_lock);
    if (!at_exit && in_prefix && cap_reached)
        log("x2w-replay record: the cap stopped the recording BEFORE the window opened: prefix only, %llu calls, %llu bytes, "
            "%ld presents (X2W_REPLAY_MAX_MIB=%llu)", end.calls, end.bytes, presents, max_bytes >> 20);
    if (!at_exit) {
        // The five slots that took the most unique blob bytes before the window (kind:slot=KiB), for sizing the prefix.
        struct { LONG64 bytes; int kind, slot; } top[5]{};
        for (int k = 0; k < kind_count; ++k)
            for (int s = 0; s < wide_slots; ++s) {
                const LONG64 bytes = prefix_slot_bytes[k][s];
                for (int t = 0; bytes > 0 && t < 5; ++t) {
                    if (bytes <= top[t].bytes) continue;
                    for (int u = 4; u > t; --u) top[u] = top[u - 1];
                    top[t] = {bytes, k, s};
                    break;
                }
            }
        if (top[0].bytes > 0)
            log("x2w-replay record: prefix blob KiB by kind:slot: %d:%d=%lld %d:%d=%lld %d:%d=%lld %d:%d=%lld %d:%d=%lld",
                top[0].kind, top[0].slot, top[0].bytes >> 10, top[1].kind, top[1].slot, top[1].bytes >> 10,
                top[2].kind, top[2].slot, top[2].bytes >> 10, top[3].kind, top[3].slot, top[3].bytes >> 10,
                top[4].kind, top[4].slot, top[4].bytes >> 10);
    }
    if (!at_exit)
        log("x2w-replay record: stopped after %llu calls, %llu blobs, %llu bytes, %ld presents%s", end.calls,
            static_cast<unsigned long long>(blobs_stored), end.bytes, presents, cap_reached ? " (X2W_REPLAY_MAX_MIB reached)" : "");
}

DWORD WINAPI finisher(void*)
{
    finish_file(false);
    SetEvent(finished);
    return 0;
}

void close_cut_file()
{
    exiting.store(true);
    const HANDLE thread = finisher_thread.load();
    DWORD waited = WAIT_OBJECT_0;
    if (thread && GetThreadId(thread) != GetCurrentThreadId()) {
        waited = WaitForSingleObject(thread, 30000);
        if (waited == WAIT_TIMEOUT)
            log("x2w-replay record: the finisher thread did not end within 30 s");
    }
    const bool locked = TryAcquireSRWLockExclusive(&file_lock);
    // A terminated writer may leave the SRW lock held. A live writer still needs exclusive access.
    if (!locked && waited != WAIT_OBJECT_0) return;
    if (file != INVALID_HANDLE_VALUE) {
        LARGE_INTEGER at;
        at.QuadPart = static_cast<LONGLONG>(whole_bytes);
        if (SetFilePointerEx(file, at, nullptr, FILE_BEGIN) && SetEndOfFile(file)) {
            bytes_written = whole_bytes;
            write_end_and_close(true);
        }
    }
    if (locked) ReleaseSRWLockExclusive(&file_lock);
}

void wait_finisher(bool at_exit)
{
    if (at_exit) {
        if (file != INVALID_HANDLE_VALUE) close_cut_file();
        return;
    }
    if (finished && WaitForSingleObject(finished, 0) != WAIT_OBJECT_0)
        WaitForSingleObject(finished, INFINITE);
}

void stop(bool at_exit, bool from_present)
{
    if (!active.exchange(false)) {
        wait_finisher(at_exit);
        return;
    }
    stopped_in_prefix = prefix_open;  // for the log: tester_pack.py flags a prefix-only file by this line
    window_open = false;
    prefix_open = false;
    unwrap_all();
    flush_all(at_exit);
    if (from_present && !at_exit) {
        finished = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (finished) {
            const HANDLE thread = CreateThread(nullptr, 0, finisher, nullptr, 0, nullptr);
            finisher_thread.store(thread);
            if (thread) return;
            CloseHandle(finished);
            finished = nullptr;
        }
    }
    finish_file(at_exit);
}

// A present ends a frame: noted in the recording, and the recording stops after the window's last present.
#include "startup_stubs/d3d12_record_pages.inl"

void present()
{
    if (!active.load(std::memory_order_relaxed)) return;
    ThreadBuffer& b = my_buffer();
    AcquireSRWLockExclusive(&b.lock);
    if (!active.load(std::memory_order_relaxed)) {
        ReleaseSRWLockExclusive(&b.lock);
        return;
    }
    const LONG present = InterlockedIncrement(&presents);
    if (present % memory_cache_presents == 0) memory_epoch.fetch_add(1, std::memory_order_relaxed);
    const bool was_open = window_open;
    if (slow_mode && end_s >= 1e300) {
        const double t = seconds_now(), gap = last_present_s < 0 ? 0 : t - last_present_s;
        last_present_s = t;
        if (t >= slow_after_s && (gap >= slow_gap_s || t >= slow_fallback_s)) {
            start_s = t;
            end_s = t + slow_seconds;
            log("x2w-replay record: window opens at %.1f s, present %ld (%s, gap %.0f ms) for %.0f s", t, present,
                gap >= slow_gap_s ? "first slowdown" : "fallback time", gap * 1000, slow_seconds);
        }
    }
    if (draw_mode && end_s >= 1e300) {
        const LONG draws = translated_draws, in_frame = draws - draws_at_present;
        draws_at_present = draws;
        draws_most = (std::max)(draws_most, in_frame);
        if (draws >= draw_trigger) {
            const double t = seconds_now();
            start_s = t;
            end_s = t + draw_seconds;
            log("x2w-replay record: window opens at %.1f s, present %ld (%ld translated draws so far, at least %ld) for %.1f s",
                t, present, draws, draw_trigger, draw_seconds);
        } else if (present % draw_log_presents == 0) {
            log("x2w-replay record: waiting for %ld translated draws: %ld so far at present %ld (the most in one frame of "
                "presents %ld-%ld was %ld)", draw_trigger, draws, present, present - draw_log_presents + 1, present, draws_most);
            draws_most = 0;
        }
    }
    if (!base_image_holds_window(b, present)) update_window();
    if (window_open || (present == first_present && !base_warming)) {
        const u64 seq = static_cast<u64>(InterlockedIncrement64(&next_seq));
        struct { u64 seq; std::int32_t present, pad; double seconds; } mark{seq, present, 0, seconds_now()};
        append_chunk(b.bytes, "PRES", &mark, sizeof(mark), nullptr, 0);
        if (!extra_title_pages && window_open && !was_open) pages_at_window_open(b, seq, present);
        if (window_open && (draw_pages || (!extra_title_pages && was_open)))
            pages_at_present(b, seq, present, !extra_title_pages && was_open);
    }
    if (!base_warming && (present >= last_present || (end_s < 1e300 && seconds_now() >= end_s)))
        InterlockedExchange(&stop_requested, 1);
    ReleaseSRWLockExclusive(&b.lock);
}

// A setting whose value names a file or folder on this machine (a drive path or a share) stays out of XENV: recordings
// are mailed, and such paths carry the user's name. The player sets its own paths (and skips path settings anyway).
bool local_path_setting(const wchar_t* entry)
{
    const wchar_t* value = std::wcschr(entry, L'=');
    if (!value) return false;
    ++value;
    return std::wcsstr(value, L":\\") || std::wcsstr(value, L":/") || std::wcsncmp(value, L"\\\\", 2) == 0;
}

// X2W_REPLAY_RECORD: all | <start s>,<seconds> | f<first present>,<presents> | slow[,<after s>,<gap ms>,<seconds>,
// <fallback s>] (default slow,120,500,10,180). Reads it, opens the file and wraps the slots; build_tables calls it
// last. Unset: nothing changes.
void install()
{
    char value[64]{};
    const DWORD n = GetEnvironmentVariableA("X2W_REPLAY_RECORD", value, sizeof(value));
    if (!n || n >= sizeof(value)) return;
    if (std::strncmp(value, "slow", 4) == 0) {
        double after = slow_after_s, gap_ms = slow_gap_s * 1000, seconds = slow_seconds, fallback = slow_fallback_s;
        if (value[4] && (value[4] != ',' || std::sscanf(value + 5, "%lf,%lf,%lf,%lf", &after, &gap_ms, &seconds, &fallback) != 4 ||
                         after < 0 || gap_ms <= 0 || seconds <= 0 || fallback < after)) {
            log("x2w-replay record: X2W_REPLAY_RECORD=%s not understood (slow | slow,<after s>,<gap ms>,<seconds>,<fallback s>)", value);
            return;
        }
        slow_mode = true;
        slow_after_s = after, slow_gap_s = gap_ms / 1000, slow_seconds = seconds, slow_fallback_s = fallback;
        start_s = 1e300;  // closed until the trigger sets it
    } else if (value[0] == 'd') {
        long draws = 0;
        double seconds = 0;
        if (std::sscanf(value + 1, "%ld,%lf", &draws, &seconds) != 2 || draws <= 0 || seconds <= 0) {
            log("x2w-replay record: X2W_REPLAY_RECORD=%s not understood (d<translated draws in a frame>,<seconds>)", value);
            return;
        }
        draw_mode = true;
        draw_trigger = draws, draw_seconds = seconds;
        start_s = 1e300;  // closed until the trigger sets it
    } else if (value[0] == 'f') {
        long first = 0, count = 0;
        if (std::sscanf(value + 1, "%ld,%ld", &first, &count) != 2 || first < 0 || count <= 0) {
            log("x2w-replay record: X2W_REPLAY_RECORD=%s not understood (all | <start s>,<seconds> | f<present>,<presents> | d<draws>,<seconds> | slow)", value);
            return;
        }
        first_present = first;
        last_present = first + count;
    } else if (std::strcmp(value, "all") != 0) {
        double start = 0, seconds = 0;
        if (std::sscanf(value, "%lf,%lf", &start, &seconds) != 2 || start < 0 || seconds <= 0) {
            log("x2w-replay record: X2W_REPLAY_RECORD=%s not understood (all | <start s>,<seconds> | f<present>,<presents> | d<draws>,<seconds> | slow)", value);
            return;
        }
        start_s = start;
        end_s = start + seconds;
    }
    char path[MAX_PATH]{};
    const DWORD p = GetEnvironmentVariableA("X2W_REPLAY_RECORD_FILE", path, sizeof(path));
    if (!p || p >= sizeof(path)) {
        // X2W_REPLAY_DIR (tester_run -Record): the file goes into that folder.
        char dir[MAX_PATH - 16]{};
        const DWORD d = GetEnvironmentVariableA("X2W_REPLAY_DIR", dir, sizeof(dir));
        if (d && d < sizeof(dir)) {
            CreateDirectoryA(dir, nullptr);
            std::snprintf(path, sizeof(path), "%s\\replay.x2wr", dir);
        } else {
            std::snprintf(path, sizeof(path), "%s", "replay.x2wr");
        }
    }
    char cap[24]{};
    const DWORD c = GetEnvironmentVariableA("X2W_REPLAY_MAX_MIB", cap, sizeof(cap));
    const unsigned long long cap_mib = c && c < sizeof(cap) ? std::strtoull(cap, nullptr, 10) : 0;
    if (cap_mib) max_bytes = cap_mib << 20;
    char prefix_cap[24]{};
    const DWORD pc = GetEnvironmentVariableA("X2W_REPLAY_PREFIX_MAX_MIB", prefix_cap, sizeof(prefix_cap));
    if (pc && pc < sizeof(prefix_cap)) prefix_max_bytes = std::strtoull(prefix_cap, nullptr, 10) << 20;
    char generic_kib[24]{};
    const DWORD gc = GetEnvironmentVariableA("X2W_REPLAY_PREFIX_GENERIC_KIB", generic_kib, sizeof(generic_kib));
    if (gc && gc < sizeof(generic_kib)) prefix_generic_bytes = std::strtoull(generic_kib, nullptr, 10) << 10;
    file = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        log("x2w-replay record: cannot create %s (error %lu)", path, GetLastError());
        return;
    }
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&started);
    // Window times count from process start (as the tester's times do), not from when the stand-in made its tables.
    FILETIME created{}, unused{}, now{};
    if (GetProcessTimes(GetCurrentProcess(), &created, &unused, &unused, &unused)) {
        GetSystemTimePreciseAsFileTime(&now);
        const auto ticks = [](const FILETIME& f) { return (static_cast<u64>(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
        const u64 alive = ticks(now) > ticks(created) ? ticks(now) - ticks(created) : 0;  // 100 ns units
        started.QuadPart -= static_cast<LONGLONG>(alive / 10000 * static_cast<u64>(frequency.QuadPart) / 1000);
    }
    std::vector<unsigned char> head;
    struct { std::uint32_t version, call_bytes; u64 frequency; } header{3, sizeof(Call), static_cast<u64>(frequency.QuadPart)};
    append_chunk(head, "XREC", &header, sizeof(header), nullptr, 0);
    // XENV: the process's X2W_ settings (name=value, NUL-separated), so a replay runs the stand-in as it ran here.
    if (wchar_t* block = GetEnvironmentStringsW()) {
        std::string settings;
        for (const wchar_t* entry = block; *entry; entry += std::wcslen(entry) + 1) {
            if (std::wcsncmp(entry, L"X2W_", 4) != 0 || std::wcsncmp(entry, L"X2W_REPLAY_RECORD", 17) == 0 ||
                local_path_setting(entry))
                continue;
            char narrow[2048];
            const int n = WideCharToMultiByte(CP_UTF8, 0, entry, -1, narrow, sizeof(narrow), nullptr, nullptr);
            if (n > 0) settings.append(narrow, static_cast<std::size_t>(n));  // n counts the NUL
        }
        FreeEnvironmentStringsW(block);
        append_chunk(head, "XENV", nullptr, 0, settings.data(), settings.size());
    }
    AcquireSRWLockExclusive(&file_lock);
    write_raw(head.data(), head.size());
    whole_bytes = bytes_written;
    ReleaseSRWLockExclusive(&file_lock);
    scan_submit_pages = x2w_local::env_on("X2W_REPLAY_PAGES_AT_SUBMIT");
    char ranges[8]{}, pages[8]{}, draw[8]{}, draw_ms[32]{};
    const bool pages_enabled = !(GetEnvironmentVariableA("X2W_REPLAY_PAGES", pages, sizeof(pages)) == 1 && pages[0] == '0');
    extra_title_pages = pages_enabled &&
        GetEnvironmentVariableA("X2W_REPLAY_TITLE_RANGES", ranges, sizeof(ranges)) == 1 && ranges[0] == '1';
    draw_pages = pages_enabled && !(GetEnvironmentVariableA("X2W_REPLAY_DRAW_PAGES", draw, sizeof(draw)) == 1 && draw[0] == '0');
    const DWORD draw_ms_length = GetEnvironmentVariableA("X2W_REPLAY_DRAW_PAGES_MS", draw_ms, sizeof(draw_ms));
    if (draw_ms_length && draw_ms_length < sizeof(draw_ms)) {
        const double ms = std::atof(draw_ms);
        if (ms > 0) draw_pages_ms = ms;
    }
    prefix_open = std::strcmp(value, "all") != 0;
    update_window();
    armed = true;
    active.store(true);
    wrap_all(std::make_index_sequence<kind_count>{});
    log("x2w-replay record: on (%s) -> %s", value, path);
}

}  // namespace replay_record

// A device made outside the recorded slots (D3D12XboxCreateDevice wraps it after install): an XOBJ chunk (kind,
// wrapper) names it, so a player binds it even when the first device call is a struct-return form.
extern "C" void x2w_d3d12_replay_record_object(int kind, const void* wrapper)
{
    using namespace replay_record;
    if (!active.load(std::memory_order_relaxed) || !wrapper) return;
    std::vector<unsigned char> chunk;
    const struct { std::uint32_t kind, pad; u64 object; } header{static_cast<std::uint32_t>(kind), 0, reinterpret_cast<u64>(wrapper)};
    append_chunk(chunk, "XOBJ", &header, sizeof(header), nullptr, 0);
    AcquireSRWLockExclusive(&file_lock);
    write_raw(chunk.data(), chunk.size());
    whole_bytes = bytes_written;
    ReleaseSRWLockExclusive(&file_lock);
}

// Ends the recording now (a fixture at its end, or a recording window that is still open when the process exits:
// DllMain, `at_exit` 1).
extern "C" __declspec(dllexport) void x2w_d3d12_replay_record_stop(int at_exit)
{
    replay_record::stop(at_exit != 0);
}

// Recorder work completed, including calls whose chunks have been flushed.
extern "C" __declspec(dllexport) u64 x2w_d3d12_replay_record_calls()
{
    return static_cast<u64>(InterlockedCompareExchange64(&replay_record::calls_written, 0, 0));
}
