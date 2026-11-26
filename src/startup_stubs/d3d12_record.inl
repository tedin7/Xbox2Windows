// Record for replay (docs/startup-runtime.md#d3d12-replay-record). X2W_REPLAY_RECORD=all | <start s>,<seconds> |
// f<first present>,<presents> writes every call the title makes into the stand-in's objects (every kind, every wrapped
// slot) to X2W_REPLAY_RECORD_FILE (default replay.x2wr): the call with its register and stack arguments and return value,
// the memory its pointer arguments name (deduplicated blobs, one level of pointers followed), and the wrapper an output
// pointer received. A replay feeds the same calls to this stand-in, so the runtime and the translator run for real.
//
// Each thread appends to its own buffer (no lock shared between title threads per call: a shared lock would serialise
// them and hide the cost being recorded); full buffers are written under the file lock. Calls are ordered by a global
// sequence number taken before the call; nothing else runs before the real slot, so float arguments in xmm registers
// reach it unchanged (their values are not recorded yet).
//
// File: the capture chunk framing (tag, header bytes, data bytes). XREC (version, timer frequency), CALL, ARGM (argument
// memory: seq, arg, address, bytes, blob hash), BLOB (hash, bytes; once per content), OUTP (seq, arg, the pointer an
// output argument holds after the call), PRES (seq, present), XEND (calls, blobs, bytes, dropped).
#include "startup_stubs/replay_pages.h"

namespace replay_record {

constexpr int wide_slots = 256;   // device and command-list slots wrapped
constexpr int narrow_slots = 64;  // slots wrapped for the other kinds
constexpr std::size_t flush_bytes = 1u << 20;
constexpr std::size_t blob_follow_bytes = 0x400;  // pointer words looked at inside an argument's memory
constexpr std::size_t blob_default_bytes = 0x100;
constexpr std::size_t blob_max_bytes = 4u << 20;
constexpr int record_depth = 2;  // argument memory, the memory it points at (a pipeline stream), and one more (its shaders)

void* original[kind_count][wide_slots];
bool armed;
bool use_slot_layouts;
static_assert(device_kind == x2w_replay_slots::device && list_kind == x2w_replay_slots::list);
LONG64 volatile next_seq;
LONG volatile presents;
LONG first_present = 0, last_present = 0x7fffffff;
double start_s = 0.0, end_s = 1e300;
LARGE_INTEGER started{}, frequency{};
HANDLE file = INVALID_HANDLE_VALUE;
// Lock order: a thread buffer's lock, then file_lock. blob_lock and buffers_lock are taken alone.
SRWLOCK file_lock = SRWLOCK_INIT, blob_lock = SRWLOCK_INIT, buffers_lock = SRWLOCK_INIT;
LONG64 volatile calls_written, blobs_stored;
u64 bytes_written;
LONG volatile stopped, stop_requested, cap_reached;
u64 max_bytes = 2048ull << 20;  // X2W_REPLAY_MAX_MIB: the recording stops (with its end chunk) once the file is this large
// Read by every wrapped slot before it calls the real one: a plain load, so nothing touches the xmm registers that carry
// float arguments. present() keeps it current (the window is checked per present).
bool volatile window_open;
// Before the window (a timed window, not "all"): the calls that make the state the window's calls name are recorded
// too, so a replay of a mid-game window knows every object: see prefix_slot. Cleared when the window opens.
bool volatile prefix_open;
// "slow": the window opens at the first present after slow_after_s whose gap to the previous present is at least
// slow_gap_s, or at slow_fallback_s; it lasts slow_seconds.
bool slow_mode;
double slow_after_s = 120, slow_gap_s = 0.5, slow_seconds = 10, slow_fallback_s = 180, last_present_s = -1;

struct ThreadBuffer {
    SRWLOCK lock = SRWLOCK_INIT;
    std::vector<unsigned char> bytes;
    std::unordered_map<u64, u64> blobs;  // address -> content hash last recorded from this thread
};
std::vector<ThreadBuffer*>& thread_buffers() { static std::vector<ThreadBuffer*> v; return v; }
std::unordered_set<u64>& stored_blobs() { static std::unordered_set<u64> s; return s; }  // blob_lock

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
    if (!armed || stopped) return false;
    const LONG present = presents;
    if (present < first_present || present >= last_present) return false;
    const double t = seconds_now();
    return t >= start_s && t < end_s;
}
void update_window()
{
    window_open = in_window();
    if (window_open) prefix_open = false;
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

// Records the memory at `address` (an argument, or a pointer found in an argument's memory) as an ARGM chunk, and its
// content once as a BLOB. The window is the larger of a default and `size_hint` (the word after the pointer, when it is a
// plausible size: a shader is a pointer, then its byte count). depth > 0 follows pointer words in its first bytes. A
// followed pointer's ARGM names its direct parent: `arg` = the call's argument | 0x100 * (word index + 1), and the
// parent is the last ARGM of the same call with depth one higher (they are written parent first).
void record_memory(ThreadBuffer& b, u64 seq, std::uint32_t arg, u64 address, std::size_t size_hint, int depth, bool exact = false)
{
    if (address < 0x10000 || address >= 0x7fffffffffffull || as_wrapper(reinterpret_cast<const void*>(address))) return;
    const std::size_t wanted = exact || size_hint > blob_default_bytes ? size_hint : blob_default_bytes;
    const std::size_t bytes = readable_window(address, wanted);
    if (!bytes || (exact && bytes != wanted)) return;
    std::vector<unsigned char> copy(bytes);
    SIZE_T got = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), copy.data(), bytes, &got) || got != bytes)
        return;
    const u64 hash = content_hash(copy.data(), bytes);
    struct { u64 seq; std::uint32_t arg, depth; u64 address, bytes, hash; } header{seq, arg, static_cast<std::uint32_t>(depth),
                                                                                    address, bytes, hash};
    append_chunk(b.bytes, "ARGM", &header, sizeof(header), nullptr, 0);
    AcquireSRWLockExclusive(&blob_lock);
    const bool fresh = stored_blobs().insert(hash).second;
    ReleaseSRWLockExclusive(&blob_lock);
    if (fresh) {
        struct { u64 hash, bytes; } blob{hash, bytes};
        append_chunk(b.bytes, "BLOB", &blob, sizeof(blob), copy.data(), bytes);
        InterlockedIncrement64(&blobs_stored);
    }
    if (depth <= 0) return;
    // The typed stream descriptor is (size, pointer). Follow just the stream,
    // and limit its copy to the declared size. Shader pointers inside still use the generic scan.
    if (exact && depth == record_depth) {
        u64 stream_bytes, stream;
        std::memcpy(&stream_bytes, copy.data(), 8);
        std::memcpy(&stream, copy.data() + 8, 8);
        if (stream_bytes && stream_bytes <= blob_max_bytes)
            record_memory(b, seq, arg | 0x200u, stream, static_cast<std::size_t>(stream_bytes), depth - 1, true);
        return;
    }
    const std::size_t scan = (bytes < blob_follow_bytes ? bytes : blob_follow_bytes) / 8;
    const auto* words = reinterpret_cast<const u64*>(copy.data());
    for (std::size_t i = 0; i < scan; ++i) {
        const u64 value = words[i];
        if (value < 0x10000 || value >= 0x7fffffffffffull || (value & 3)) continue;
        const u64 next = i + 1 < scan ? words[i + 1] : 0;
        record_memory(b, seq, (arg & 0xffu) | 0x100u * static_cast<std::uint32_t>(i + 1), value,
                      next > 0 && next <= blob_max_bytes ? static_cast<std::size_t>(next) : 0, depth - 1);
    }
}

constexpr int call_args = 11;      // argument words after rcx, all recorded (format 3)
constexpr std::uint32_t rcx_arg = 15;  // ARGM argument index of rcx when it is not an object
// Argument words followed in full: memory sized by the next word, and the pointers inside it. The words after them are
// mostly the caller's stack (few slots take more than seven arguments): a short look each, so a stale pointer and size
// pair there cannot pull megabytes into the file. A late real argument (an output, an interface id) fits in it.
constexpr std::uint32_t followed_args = 7;

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

void record_wrapper_output(ThreadBuffer& b, u64 seq, std::uint32_t arg, u64 address)
{
    if (!x2w_local::readable(reinterpret_cast<const void*>(address), 8)) return;
    const u64 now = *reinterpret_cast<const u64*>(address);
    if (!as_wrapper(reinterpret_cast<const void*>(now))) return;
    struct { u64 seq; std::uint32_t arg, pad; u64 value; } out{seq, arg, 0, now};
    append_chunk(b.bytes, "OUTP", &out, sizeof(out), nullptr, 0);
}

void record_arguments(ThreadBuffer& b, const Call& call)
{
    const auto& args = call.args;
    const u64 seq = call.seq;
    const auto* layout = use_slot_layouts ? x2w_replay_slots::find(call.kind, call.slot) : nullptr;
    if (layout && layout->known_memory) {
        for (std::uint32_t i = 0; i < 5; ++i) {
            if (!layout->memory[i]) continue;
            const int depth = static_cast<int>(i) == layout->follow ? record_depth : 0;
            record_memory(b, seq, i, args[i], layout->memory[i], depth, true);
            if (static_cast<int>(i) == layout->wrapper_output && SUCCEEDED(static_cast<HRESULT>(call.ret)))
                record_wrapper_output(b, seq, i, args[i]);
        }
        return;
    }
    if (!as_wrapper(reinterpret_cast<const void*>(call.object))) record_memory(b, seq, rcx_arg, call.object, 0, 0);
    // Words past a slot's own arguments are the caller's stack: recorded too (a replay passes them on, the slot ignores
    // them), the ones after followed_args with a short look only.
    for (std::uint32_t i = 0; i < call_args; ++i) {
        const u64 value = args[i];
        if (value < 0x10000 || value >= 0x7fffffffffffull || as_wrapper(reinterpret_cast<const void*>(value))) continue;
        record_wrapper_output(b, seq, i, value);
        if (i >= followed_args) {
            record_memory(b, seq, i, value, 0, 0);
            continue;
        }
        const u64 next = i + 1 < followed_args ? args[i + 1] : 0;
        record_memory(b, seq, i, value, next <= blob_max_bytes ? static_cast<std::size_t>(next) : 0, record_depth);
    }
}

// Exact CopyDescriptors inputs. The prefix already contains heap and view creation, simple
// copies, and object lifetimes. This supplies the counted arrays that generic ARGM can truncate.
bool record_descriptor_arrays(ThreadBuffer& b, u64 seq, const u64* args)
{
    std::vector<unsigned char> arrays;
    for (const unsigned arg : {1u, 2u, 4u, 5u}) {
        const u64 count = static_cast<std::uint32_t>(args[arg < 3 ? 0 : 3]);
        const u64 bytes = count * ((arg == 1 || arg == 4) ? sizeof(u64) : sizeof(UINT));
        if (!args[arg] || !bytes) continue;
        const auto start = arrays.size();
        if (bytes > max_bytes || start > max_bytes - bytes) return false;
        arrays.resize(start + static_cast<std::size_t>(bytes));
        SIZE_T got = 0;
        if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(args[arg]),
                               arrays.data() + start, static_cast<SIZE_T>(bytes), &got) || got != bytes)
            return false;
    }
    append_chunk(b.bytes, "DARR", &seq, sizeof(seq), arrays.data(), arrays.size());
    return true;
}

// After the call: the call, then for each pointer argument the wrapper its memory holds (OUTP: a creation's result
// when the replayer has not seen that wrapper yet, else an input such as a list array) and its memory (ARGM).
void record_call(int kind, int slot, u64 seq, u64 object, const u64 (&args)[call_args], u64 ret, bool descriptor_copy)
{
    ThreadBuffer& b = my_buffer();
    AcquireSRWLockExclusive(&b.lock);
    Call call{seq, GetCurrentThreadId(), static_cast<std::uint16_t>(kind), static_cast<std::uint16_t>(slot), object, {}, ret,
              seconds_now(), presents, 0};
    std::memcpy(call.args, args, sizeof(call.args));
    append_chunk(b.bytes, "CALL", &call, sizeof(call), nullptr, 0);
    if (kind == device_kind && slot == 14 && SUCCEEDED(static_cast<HRESULT>(ret)) &&
        x2w_local::readable(reinterpret_cast<const void*>(args[2]), sizeof(void*))) {
        const auto* heap = as_wrapper(*reinterpret_cast<void* const*>(args[2]));
        const u64 extension = args[0] + sizeof(D3D12_DESCRIPTOR_HEAP_DESC);
        if (heap && heap->own().kind == descriptor_heap_kind && heap->own().title_cpu &&
            x2w_local::readable(reinterpret_cast<const void*>(extension), 2 * sizeof(u64))) {
            const auto* extra = reinterpret_cast<const u64*>(extension);
            // The wrapper has already validated the extension against the heap's descriptor count and stride.
            if (extra[0] == heap->own().title_cpu && extra[1]) {
                const u64 backing[] = {seq, extra[0], extra[1]};
                append_chunk(b.bytes, "DHPM", backing, sizeof(backing), nullptr, 0);
            }
        }
    }
    // The generic argument memory is recorded as well: a file without the DARR chunk still replays the copy, cut short.
    const bool typed_arrays = descriptor_copy && record_descriptor_arrays(b, seq, args);
    if (descriptor_copy && !typed_arrays) {
        static LONG unrecorded;
        if (InterlockedIncrement(&unrecorded) <= 8)
            log("x2w-replay record: CopyDescriptors arrays of call %llu not recorded exactly (unreadable or over the size cap); "
                "the replay of that copy may be cut short", static_cast<unsigned long long>(seq));
    }
    record_arguments(b, call);
    InterlockedIncrement64(&calls_written);
    if (b.bytes.size() >= flush_bytes) {
        AcquireSRWLockExclusive(&file_lock);
        drain(b);
        if (bytes_written >= max_bytes && !InterlockedExchange(&cap_reached, 1)) InterlockedExchange(&stop_requested, 1);
        ReleaseSRWLockExclusive(&file_lock);
    }
    ReleaseSRWLockExclusive(&b.lock);
}

void stop(bool at_exit);
inline void pages_at_submit(ThreadBuffer& b, u64 seq);  // d3d12_record_pages.inl

template <int K, int N>
__declspec(noinline) u64 thunk(u64 rcx, u64 rdx, u64 r8, u64 r9, u64 s0, u64 s1, u64 s2, u64 s3, u64 s4, u64 s5, u64 s6,
                               u64 s7)
{
    // Eleven arguments after rcx reach the real slot (D3D12's longest, UpdateTileMappings, has ten): stack words past
    // the caller's own arguments are read and passed on unused, which the x64 convention allows (the caller reserves
    // the home area; a slot reads only what it declares). All eleven are recorded.
    using Slot = u64 (*)(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);
    const auto call = reinterpret_cast<Slot>(original[K][N]);
    if (!window_open && !(prefix_slot(K, N) && prefix_open)) return call(rcx, rdx, r8, r9, s0, s1, s2, s3, s4, s5, s6, s7);
    // CopyDescriptors takes no float arguments, so its counted arrays are read from the call's own registers.
    constexpr bool descriptor_copy = K == device_kind && N == 23;
    const u64 seq = static_cast<u64>(InterlockedIncrement64(&next_seq));
    // ExecuteCommandLists takes no float arguments, so the page scan may run before it.
    if constexpr (K == queue_kind && N == 10) {
        if (window_open) {
            ThreadBuffer& b = my_buffer();
            AcquireSRWLockExclusive(&b.lock);
            pages_at_submit(b, seq - 1);
            ReleaseSRWLockExclusive(&b.lock);
        }
    }
    const u64 ret = call(rcx, rdx, r8, r9, s0, s1, s2, s3, s4, s5, s6, s7);
    const u64 args[call_args] = {rdx, r8, r9, s0, s1, s2, s3, s4, s5, s6, s7};
    record_call(K, N, seq, rcx, args, ret, descriptor_copy);
    if (stop_requested) stop(false);  // after the window's last present: that Present call is in the file
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
// so a large cache cannot hold the exit up. file_lock held.
void write_shader_cache(bool at_exit)
{
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
    do {
        if (found.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
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
        ++files;
        total += size;
    } while (FindNextFileW(find, &found));
    FindClose(find);
    if (!at_exit)
        log("x2w-replay record: shader cache %llu files, %llu bytes, %llu left out", files, total, left_out);
}

// Ends the recording: every buffer is written, then the end chunk with the totals. Called once.
void stop(bool at_exit)
{
    if (!armed || InterlockedExchange(&stopped, 1)) return;
    window_open = false;
    flush_all(at_exit);
    if (at_exit && !TryAcquireSRWLockExclusive(&file_lock)) return;
    if (!at_exit) AcquireSRWLockExclusive(&file_lock);
    write_shader_cache(at_exit);
    std::vector<unsigned char> tail;
    struct { u64 calls, blobs, bytes, presents; } end{calls_written, blobs_stored, bytes_written, static_cast<u64>(presents)};
    append_chunk(tail, "XEND", &end, sizeof(end), nullptr, 0);
    write_raw(tail.data(), tail.size());
    FlushFileBuffers(file);
    CloseHandle(file);
    file = INVALID_HANDLE_VALUE;
    ReleaseSRWLockExclusive(&file_lock);
    if (!at_exit)
        log("x2w-replay record: stopped after %llu calls, %llu blobs, %llu bytes, %ld presents%s", end.calls, end.blobs,
            end.bytes, presents, cap_reached ? " (X2W_REPLAY_MAX_MIB reached)" : "");
}

// A present ends a frame: noted in the recording, and the recording stops after the window's last present.
#include "startup_stubs/d3d12_record_pages.inl"

void present()
{
    if (!armed || stopped) return;
    const LONG present = InterlockedIncrement(&presents);
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
    update_window();
    if (window_open || present == first_present) {
        ThreadBuffer& b = my_buffer();
        AcquireSRWLockExclusive(&b.lock);
        const u64 seq = static_cast<u64>(InterlockedIncrement64(&next_seq));
        struct { u64 seq; std::int32_t present, pad; double seconds; } mark{seq, present, 0, seconds_now()};
        append_chunk(b.bytes, "PRES", &mark, sizeof(mark), nullptr, 0);
        if (window_open && !was_open) pages_at_window_open(b, seq, present);
        else if (window_open) pages_at_present(b, seq, present);
        ReleaseSRWLockExclusive(&b.lock);
    }
    if (present >= last_present || (end_s < 1e300 && seconds_now() >= end_s)) InterlockedExchange(&stop_requested, 1);
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
    } else if (value[0] == 'f') {
        long first = 0, count = 0;
        if (std::sscanf(value + 1, "%ld,%ld", &first, &count) != 2 || first < 0 || count <= 0) {
            log("x2w-replay record: X2W_REPLAY_RECORD=%s not understood (all | <start s>,<seconds> | f<present>,<presents>)", value);
            return;
        }
        first_present = first;
        last_present = first + count;
    } else if (std::strcmp(value, "all") != 0) {
        double start = 0, seconds = 0;
        if (std::sscanf(value, "%lf,%lf", &start, &seconds) != 2 || start < 0 || seconds <= 0) {
            log("x2w-replay record: X2W_REPLAY_RECORD=%s not understood (all | <start s>,<seconds> | f<present>,<presents>)", value);
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
    char layouts[2]{};
    use_slot_layouts = GetEnvironmentVariableA(x2w_replay_slots::setting, layouts, sizeof(layouts)) == 1 && layouts[0] == '1';
    std::vector<unsigned char> head;
    struct { std::uint32_t version, call_bytes; u64 frequency; } header{3, sizeof(Call), static_cast<u64>(frequency.QuadPart)};
    append_chunk(head, "XREC", &header, sizeof(header), nullptr, 0);
    // XENV: the process's X2W_ settings (name=value, NUL-separated), so a replay runs the stand-in as it ran here.
    if (wchar_t* block = GetEnvironmentStringsW()) {
        std::string settings;
        for (const wchar_t* entry = block; *entry; entry += std::wcslen(entry) + 1) {
            if (std::wcsncmp(entry, L"X2W_", 4) != 0 || std::wcsncmp(entry, L"X2W_REPLAY_RECORD", 17) == 0) continue;
            char narrow[2048];
            const int n = WideCharToMultiByte(CP_UTF8, 0, entry, -1, narrow, sizeof(narrow), nullptr, nullptr);
            if (n > 0) settings.append(narrow, static_cast<std::size_t>(n));  // n counts the NUL
        }
        FreeEnvironmentStringsW(block);
        append_chunk(head, "XENV", nullptr, 0, settings.data(), settings.size());
    }
    write_raw(head.data(), head.size());
    armed = true;
    prefix_open = std::strcmp(value, "all") != 0;
    update_window();
    wrap_all(std::make_index_sequence<kind_count>{});
    log("x2w-replay record: on (%s) -> %s", value, path);
}

}  // namespace replay_record

// Ends the recording now (a fixture at its end, or a recording window that is still open when the process exits:
// DllMain, `at_exit` 1).
extern "C" __declspec(dllexport) void x2w_d3d12_replay_record_stop(int at_exit)
{
    replay_record::stop(at_exit != 0);
}
