// Desktop xapu.dll trap stubs.
// Every export traps; see trap.h. Scaffolding, not an ABI claim.
#define X2W_TRAP_MODULE "xapu.dll"
#define X2W_TRAP_MODULE_ID 3
#include "startup_stubs/trap.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/audio_telemetry.h"
#include <timeapi.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#ifdef X2W_HAVE_OPUS
#include <opus.h>
#endif

// Experimental (docs/startup-runtime.md). With X2W_AUDIO_PERMISSIVE=1 the exports
// stand in for the audio command queue; otherwise they trap. Call shapes observed:
//   XApuConnect(input*, output*, handle*) -> HRESULT (the handle is later passed to the others). In one
//     title's calls, the caller reads one pointer from output +0 and its later decode commands point into that
//     buffer; the size is taken from the u32 at input +0x0C (0x104000 there). Not traced to every consumer.
//   XApuGetResultCount(handle, 0) -> count
//   XApuDequeueResult(handle, result*) -> HRESULT; zeroes 0x58 bytes. With the experimental
//     X2W_XAPU_TYPED_COMPLETIONS=1, also echoes the command type at +0 and slot at +4.
//   XApuEnqueueCommand(handle, command*, ...) -> HRESULT; reads 32-bit type at +0 and slot at +4.
//   XApuDisconnect(handle)
// Optional Opus decoding with typed completions: a process-1 connection uses type 1 to select the
// stream's channels. Type 3 decodes a packet, skips the leading frames at +0x20, and returns the
// window that fits the caller's count and capacity. Connection option 4 requests planar stereo.
// Older untyped-format fixtures describe their decode rate and buffer layout on the type 3 command.
// Type 2 frees the slot's decoder and type 5 resets it. A command with no packet completes with 0 bytes.
// A refused decode returns an error, queues nothing and leaves the output untouched.
// The same build keeps bounded counters for decoded packets, dequeued decoded completions and
// refusals by reason, and writes one stable summary at the last disconnect and at process exit.
namespace {
using u64 = std::uint64_t;
constexpr long hr_ok = 0;
constexpr long hr_pointer = static_cast<long>(0x80004003);  // E_POINTER
constexpr long hr_handle = static_cast<long>(0x80070006);   // E_HANDLE
constexpr long hr_out_of_memory = static_cast<long>(0x8007000E); // E_OUTOFMEMORY
constexpr long hr_invalid_arg = static_cast<long>(0x80070057);  // E_INVALIDARG
constexpr long hr_fail = static_cast<long>(0x80004005);         // E_FAIL
constexpr long hr_pending_results = static_cast<long>(0x8AC81001); // XAPU_E_PENDING_RESULTS
constexpr std::size_t command_header_size = 8;
constexpr std::size_t result_size = 0x58;
constexpr std::size_t max_queue_entries = 16384;
constexpr std::uint32_t max_buffer_bytes = 64u << 20;  // sanity cap against a bad size, not a measured limit

bool permissive() noexcept { return x2w_local::env_on("X2W_AUDIO_PERMISSIVE"); }
bool typed_completions() noexcept { return x2w_local::env_on("X2W_XAPU_TYPED_COMPLETIONS"); }

struct QueueEntry {
    std::uint32_t type = 0;
    std::uint32_t slot = 0;
    std::uint32_t value = 0;  // typed result +0x34: decoded bytes of a type 3 command
#ifdef X2W_HAVE_OPUS
    bool decoded = false;     // this result came from a successful libopus call
    bool stream_command = false;
    unsigned char command_tag[16]{}; // sequence, options, caller context
    std::uint32_t frames = 0, input_bytes = 0;
    u64 input = 0, output = 0;
#endif
};

#ifdef X2W_HAVE_OPUS
constexpr std::uint32_t max_slots = 256;
constexpr std::size_t decode_command_size = 0x40;
constexpr std::uint32_t max_frames = 5760;         // 120 ms at 48 kHz, the longest Opus packet
constexpr std::uint32_t max_packet_bytes = 1u << 16;  // sanity cap, not a measured limit
struct SlotDecoder {
    OpusDecoder* decoder = nullptr;
    int channels = 0;
    int rate = 0;
};
#endif

struct Connection {
    Connection* next = nullptr;
    u64 handle = 0;
    QueueEntry* entries = nullptr;
    std::size_t head = 0;
    std::size_t count = 0;
    std::size_t capacity = 0;
    void* buffer = nullptr;  // returned in output +0
#ifdef X2W_HAVE_OPUS
    SlotDecoder decoders[max_slots];
    bool opus_streams = false;
    std::uint32_t options = 0;
#endif
};

SRWLOCK connection_lock = SRWLOCK_INIT;
Connection* connections = nullptr;
u64 next_handle = 0x5841500000000001ULL;
bool timer_active = false;

#ifdef X2W_HAVE_OPUS
// Telemetry for the optional decode path. It starts with the first readable type-3 command, so a
// default run with typed completions off reads and writes nothing here.
x2w_audio::OpusCounters opus_counts;
std::atomic<bool> opus_counters_on{false};
SRWLOCK opus_summary_lock = SRWLOCK_INIT;
x2w_audio::OpusCounterSnapshot reported_opus_counts{};

bool same_opus_counts(const x2w_audio::OpusCounterSnapshot& a, const x2w_audio::OpusCounterSnapshot& b) noexcept
{
    return a.type3_commands == b.type3_commands && a.packets_decoded == b.packets_decoded &&
           a.decoded_completions == b.decoded_completions && a.refused_header == b.refused_header &&
           a.refused_short_command == b.refused_short_command &&
           a.refused_unknown_handle == b.refused_unknown_handle &&
           a.refused_queue_full == b.refused_queue_full &&
           a.refused_invalid_argument == b.refused_invalid_argument &&
           a.refused_opus_error == b.refused_opus_error && a.refused_capacity == b.refused_capacity &&
           a.refused_out_of_memory == b.refused_out_of_memory;
}

// Writes the totals since the previous summary, when they moved. Called at the last disconnect and
// from atexit, so a run that never disconnects still ends with one line on stderr and X2W_TRAP_LOG.
void log_opus_summary() noexcept
{
    if (!opus_counters_on.load(std::memory_order_relaxed)) return;
    const x2w_audio::OpusCounterSnapshot now = x2w_audio::snapshot(opus_counts);
    AcquireSRWLockExclusive(&opus_summary_lock);
    const bool changed = !same_opus_counts(now, reported_opus_counts);
    if (changed) reported_opus_counts = now;
    ReleaseSRWLockExclusive(&opus_summary_lock);
    if (!changed || (!now.type3_commands && !x2w_audio::refused_total(now))) return;
    x2w_local::logf("x2w-audio XApu opus summary: type3=%llu packets=%llu completions=%llu refusals=%llu "
                    "invalid_argument=%llu opus_error=%llu capacity=%llu queue_full=%llu short_command=%llu "
                    "unknown_handle=%llu out_of_memory=%llu header=%llu",
                    static_cast<unsigned long long>(now.type3_commands),
                    static_cast<unsigned long long>(now.packets_decoded),
                    static_cast<unsigned long long>(now.decoded_completions),
                    static_cast<unsigned long long>(x2w_audio::refused_total(now)),
                    static_cast<unsigned long long>(now.refused_invalid_argument),
                    static_cast<unsigned long long>(now.refused_opus_error),
                    static_cast<unsigned long long>(now.refused_capacity),
                    static_cast<unsigned long long>(now.refused_queue_full),
                    static_cast<unsigned long long>(now.refused_short_command),
                    static_cast<unsigned long long>(now.refused_unknown_handle),
                    static_cast<unsigned long long>(now.refused_out_of_memory),
                    static_cast<unsigned long long>(now.refused_header));
}

void start_opus_counters() noexcept
{
    bool expected = false;
    if (!opus_counters_on.compare_exchange_strong(expected, true, std::memory_order_relaxed)) return;
    std::atexit(log_opus_summary);
}

enum class DecodeRefusal { none, invalid_argument, opus_error, capacity, out_of_memory };
#endif

// Makes room for one more entry, so a following place_result cannot fail.
bool reserve_result(Connection& conn) noexcept
{
    if (conn.count >= max_queue_entries) return false;
    if (conn.count == conn.capacity) {
        const std::size_t doubled = conn.capacity ? conn.capacity * 2 : 64;
        const std::size_t next_cap = doubled > max_queue_entries ? max_queue_entries : doubled;
        auto* next = new (std::nothrow) QueueEntry[next_cap];
        if (!next) return false;
        for (std::size_t i = 0; i < conn.count; ++i) {
            next[i] = conn.entries[(conn.head + i) % conn.capacity];
        }
        delete[] conn.entries;
        conn.entries = next;
        conn.head = 0;
        conn.capacity = next_cap;
    }
    return true;
}

void place_result(Connection& conn, QueueEntry entry) noexcept
{
    conn.entries[(conn.head + conn.count) % conn.capacity] = entry;
    ++conn.count;
}

bool push_result(Connection& conn, QueueEntry entry) noexcept
{
    if (!reserve_result(conn)) return false;
    place_result(conn, entry);
    return true;
}

QueueEntry pop_result(Connection& conn) noexcept
{
    const QueueEntry entry = conn.entries[conn.head];
    --conn.count;
    if (conn.count == 0) {
        conn.head = 0;
    } else {
        conn.head = (conn.head + 1) % conn.capacity;
    }
    return entry;
}

Connection* find_connection(u64 handle) noexcept
{
    for (auto* conn = connections; conn; conn = conn->next) {
        if (conn->handle == handle) return conn;
    }
    return nullptr;
}

void hex(const void* data, std::size_t bytes, char* out, std::size_t size)
{
    out[0] = 0;
    if (!data || !x2w_local::readable(data, bytes)) return;
    const auto* p = static_cast<const unsigned char*>(data);
    std::size_t used = 0;
    for (std::size_t i = 0; i < bytes && used + 4 < size; ++i) used += std::snprintf(out + used, size - used, "%02x ", p[i]);
}

#ifdef X2W_HAVE_OPUS
// Frees (close) or resets (reset) a slot's decoder. Called with connection_lock held.
void drop_decoder(Connection& conn, std::uint32_t slot, bool destroy) noexcept
{
    if (slot >= max_slots) return;
    SlotDecoder& d = conn.decoders[slot];
    if (destroy) {
        if (d.decoder) opus_decoder_destroy(d.decoder);
        d = SlotDecoder{};
    } else if (d.decoder) {
        opus_decoder_ctl(d.decoder, OPUS_RESET_STATE);
    }
}

// Decodes the packet of a type 3 command into a scratch buffer, so a refusal never touches the
// caller's output. On S_OK, *pcm holds entry.value bytes of float PCM for *dest. `decoded_packet`
// distinguishes a libopus decode from the no-packet completion, and `refusal` names the first check
// that refused the command. Called with connection_lock held, which also guards the scratch buffer.
long decode_packet(Connection& conn, const unsigned char* c, QueueEntry& entry, const float** pcm, void** dest,
                   bool& decoded_packet, DecodeRefusal& refusal) noexcept
{
    static float scratch[max_frames * 2];
    static float planar[max_frames * 2];
    decoded_packet = false;
    refusal = DecodeRefusal::none;
    std::uint32_t slot = 0, frames = 0, packet_bytes = 0, out_bytes = 0;
    std::uint32_t first_frame = 0;
    float rate_f = 0;
    u64 packet = 0, out = 0;
    std::memcpy(&slot, c + 4, 4);
    std::memcpy(&rate_f, c + 0x1C, 4);
    std::memcpy(&first_frame, c + 0x20, 4);
    std::memcpy(&frames, c + 0x24, 4);
    std::memcpy(&packet_bytes, c + 0x28, 4);
    std::memcpy(&packet, c + 0x2C, 8);
    std::memcpy(&out_bytes, c + 0x34, 4);
    std::memcpy(&out, c + 0x38, 8);
    if (conn.opus_streams) {
        entry.input = packet;
        entry.input_bytes = packet_bytes;
        entry.output = out;
    }
    if (!packet && !packet_bytes) {  // no packet: complete with 0 bytes, as without libopus
        return hr_ok;
    }
    if (slot >= max_slots) {
        refusal = DecodeRefusal::invalid_argument;
        return hr_invalid_arg;
    }
    SlotDecoder& d = conn.decoders[slot];
    const int rate = conn.opus_streams ? 48000 : rate_f == 8000.0f ? 8000 : rate_f == 12000.0f ? 12000 : rate_f == 16000.0f ? 16000
                   : rate_f == 24000.0f ? 24000 : rate_f == 48000.0f ? 48000 : 0;
    const std::uint32_t channels = conn.opus_streams ? d.channels :
        frames && frames <= max_frames && out_bytes % (frames * 4) == 0 ? out_bytes / (frames * 4) : 0;
    if (!rate || (channels != 1 && channels != 2) || packet_bytes == 0 || packet_bytes > max_packet_bytes ||
        !x2w_local::readable(reinterpret_cast<const void*>(packet), packet_bytes) ||
        !x2w_local::writable(reinterpret_cast<const void*>(out), out_bytes)) {
        refusal = DecodeRefusal::invalid_argument;
        return hr_invalid_arg;
    }
    // A new rate or channel count gets a new decoder, which replaces the slot's only after a good decode.
    OpusDecoder* decoder = d.decoder;
    const bool fresh = !decoder || d.channels != static_cast<int>(channels) || d.rate != rate;
    if (fresh) {
        int error = OPUS_OK;
        decoder = opus_decoder_create(rate, static_cast<int>(channels), &error);
        if (!decoder || error != OPUS_OK) {
            if (decoder) opus_decoder_destroy(decoder);
            refusal = DecodeRefusal::out_of_memory;
            return hr_out_of_memory;
        }
    }
    const int decoded = opus_decode_float(decoder, reinterpret_cast<const unsigned char*>(packet),
                                          static_cast<opus_int32>(packet_bytes), scratch, conn.opus_streams ? max_frames : static_cast<int>(frames), 0);
    if (decoded < 0) {
        if (fresh) opus_decoder_destroy(decoder);
        refusal = decoded == OPUS_BUFFER_TOO_SMALL ? DecodeRefusal::capacity : DecodeRefusal::opus_error;
        static LONG errors;
        if (x2w_local::first_calls(errors, 8))
            x2w_local::logf("x2w-audio XApu decode slot=%u bytes=%u -> opus error %d", slot, packet_bytes, decoded);
        return hr_fail;
    }
    if (fresh) {
        if (d.decoder) opus_decoder_destroy(d.decoder);
        d = SlotDecoder{decoder, static_cast<int>(channels), rate};
    }
    const auto skipped = conn.opus_streams ? (std::min)(first_frame, static_cast<std::uint32_t>(decoded)) : 0u;
    entry.frames = (std::min)({static_cast<std::uint32_t>(decoded) - skipped, frames, out_bytes / (4u * channels)});
    entry.value = entry.frames * channels * 4u;
    const float* kept = scratch + skipped * channels;
    if (conn.opus_streams && (conn.options & 4u) && channels == 2) {
        for (std::uint32_t frame = 0; frame < entry.frames; ++frame) {
            planar[frame] = kept[2 * frame];
            planar[entry.frames + frame] = kept[2 * frame + 1];
        }
        kept = planar;
    }
    decoded_packet = true;
    *pcm = kept;
    *dest = reinterpret_cast<void*>(out);
    return hr_ok;
}
#endif

// Activity counter (#1145), off unless X2W_AUDIO_ACTIVITY=1: every XApuEnqueueCommand call, counted by command class,
// taken as the first UINT32 of the command (a stand-in convention, like the result type above), with no cap. Classes
// 0..15 are counted one by one and larger ones together; a command that cannot be read and a call with an unknown
// handle are counted apart. commands= in the line is the classes plus the unreadable ones; rejected calls are not in
// it. There is no thread: a counted call (or x2w_xapu_command_counts) at least a second after the previous line (or the
// first such call) writes the next line, with the counts since that line, if a count moved; until then the counts wait
// for a later call. Nothing here runs outside a call into this DLL. Off, nothing here runs: no read of the command, no
// line.
constexpr unsigned command_classes = 16;
struct CommandCounts {
    u64 by_class[command_classes + 1];  // the last entry: every larger class
    u64 unreadable, rejected;
};
SRWLOCK count_lock = SRWLOCK_INIT;  // guards the four below
CommandCounts command_counts, reported;  // the counts now and at the last line
ULONGLONG last_line;  // GetTickCount64 of the last line written, or of the first counted call or counter-on export call
bool counting_started;

bool activity_on() noexcept { return x2w_local::env_on("X2W_AUDIO_ACTIVITY"); }

// If the last line is at least a second old and a count moved, writes the next line into text and returns true.
// Hold count_lock exclusively; log the text after releasing it.
bool poll_locked(char (&text)[512]) noexcept
{
    const ULONGLONG tick = GetTickCount64();
    if (!counting_started) {
        counting_started = true;
        last_line = tick;
    }
    const CommandCounts& now = command_counts;
    if (tick - last_line < 1000 || !std::memcmp(&now, &reported, sizeof(now))) return false;
    last_line = tick;
    char classes[320] = "";
    std::size_t used = 0;
    u64 commands = now.unreadable - reported.unreadable;
    for (unsigned c = 0; c <= command_classes; ++c) {
        const u64 n = now.by_class[c] - reported.by_class[c];
        commands += n;
        if (!n || used >= sizeof(classes)) continue;
        used += static_cast<std::size_t>(c < command_classes ? std::snprintf(classes + used, sizeof(classes) - used, " %u=%llu", c, n)
                                                             : std::snprintf(classes + used, sizeof(classes) - used, " other=%llu", n));
    }
    std::snprintf(text, sizeof(text),
                  "x2w-audio xapu-activity t=%.1f s: commands=%llu by class%s unreadable=%llu; rejected=%llu (not in commands)",
                  x2w_local::process_seconds(), commands, classes, now.unreadable - reported.unreadable,
                  now.rejected - reported.rejected);
    reported = now;
    return true;
}

// Counts one XApuEnqueueCommand call; the caller's last error is kept.
void count_command(long hr, const void* command) noexcept
{
    if (!activity_on()) return;
    const DWORD error = GetLastError();
    const bool readable = hr != hr_handle && x2w_local::readable(command, sizeof(std::uint32_t));
    const std::uint32_t command_class = readable ? *static_cast<const std::uint32_t*>(command) : 0;
    char text[512];
    AcquireSRWLockExclusive(&count_lock);
    if (hr == hr_handle)
        ++command_counts.rejected;
    else if (!readable)
        ++command_counts.unreadable;
    else
        ++command_counts.by_class[command_class < command_classes ? command_class : command_classes];
    const bool line = poll_locked(text);
    ReleaseSRWLockExclusive(&count_lock);
    if (line) x2w_local::logf("%s", text);
    SetLastError(error);
}
}  // namespace

extern "C" __declspec(noinline) long XApuConnect(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    if (!permissive())
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, "XApuConnect", 0, rcx, rdx, r8, r9, _ReturnAddress());
    char input[3 * 0x18 + 1], output[3 * 0x10 + 1];
    hex(reinterpret_cast<const void*>(rcx), 0x18, input, sizeof(input));
    hex(reinterpret_cast<const void*>(rdx), 0x10, output, sizeof(output));
    if (!r8 || !x2w_local::writable(reinterpret_cast<const void*>(r8), sizeof(void*)) || !rcx ||
        !x2w_local::readable(reinterpret_cast<const void*>(rcx), 0x10) || !rdx ||
        !x2w_local::writable(reinterpret_cast<const void*>(rdx), sizeof(void*))) {
        x2w_local::logf("x2w-audio XApuConnect input=[%s] output(before)=[%s] -> E_POINTER", input, output);
        return hr_pointer;
    }
    std::uint32_t buffer_bytes = 0;
    std::memcpy(&buffer_bytes, reinterpret_cast<const unsigned char*>(rcx) + 0x0C, sizeof(buffer_bytes));
    if (buffer_bytes == 0 || buffer_bytes > max_buffer_bytes) {
        x2w_local::logf("x2w-audio XApuConnect input=[%s] buffer bytes 0x%x -> E_INVALIDARG", input, buffer_bytes);
        return hr_invalid_arg;
    }
    void* buffer = VirtualAlloc(nullptr, buffer_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    auto* conn = buffer ? new (std::nothrow) Connection : nullptr;
    if (!conn) {
        if (buffer) VirtualFree(buffer, 0, MEM_RELEASE);
        return hr_out_of_memory;
    }
    conn->buffer = buffer;
#ifdef X2W_HAVE_OPUS
    std::uint32_t process = 0;
    std::memcpy(&process, reinterpret_cast<const void*>(rcx), 4);
    if (process == 1 && typed_completions()) {
        if (!x2w_local::readable(reinterpret_cast<const void*>(rcx), 24)) {
            VirtualFree(buffer, 0, MEM_RELEASE);
            delete conn;
            return hr_pointer;
        }
        conn->opus_streams = true;
        std::memcpy(&conn->options, reinterpret_cast<const unsigned char*>(rcx) + 20, 4);
    }
#endif
    AcquireSRWLockExclusive(&connection_lock);
    const u64 handle = next_handle++;
    if (next_handle == 0) next_handle = 0x5841500000000001ULL;
    conn->handle = handle;
    conn->next = connections;
    connections = conn;
    if (!conn->next) {
        // The mix worker sleeps for 2 ms between refills. A coarse Windows tick can
        // stretch that wait beyond the time represented by the queued samples.
        timer_active = timeBeginPeriod(1) == TIMERR_NOERROR;
        if (!timer_active) x2w_local::logf("x2w-audio 1 ms mixer timer request unavailable");
    }
    ReleaseSRWLockExclusive(&connection_lock);
    *reinterpret_cast<void**>(rdx) = buffer;
    *reinterpret_cast<void**>(r8) = reinterpret_cast<void*>(handle);
    x2w_local::logf("x2w-audio XApuConnect input=[%s] output(before)=[%s] -> handle %p, buffer %p (0x%x bytes), S_OK",
                    input, output, reinterpret_cast<void*>(handle), buffer, buffer_bytes);
    return hr_ok;
}

extern "C" __declspec(noinline) std::uint32_t XApuGetResultCount(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    if (!permissive())
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, "XApuGetResultCount", 4, rcx, rdx, r8, r9, _ReturnAddress());
    AcquireSRWLockShared(&connection_lock);
    const Connection* conn = find_connection(rcx);
    const std::uint32_t count = conn
        ? static_cast<std::uint32_t>(conn->count > 0xFFFFFFFFULL ? 0xFFFFFFFFULL : conn->count)
        : 0u;
    ReleaseSRWLockShared(&connection_lock);
    static LONG calls;
    if (x2w_local::first_calls(calls, 2))
        x2w_local::logf("x2w-audio XApuGetResultCount handle=0x%llx rdx=0x%llx -> %u", rcx, rdx, count);
    return count;
}

extern "C" __declspec(noinline) long XApuDequeueResult(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    if (!permissive())
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, "XApuDequeueResult", 1, rcx, rdx, r8, r9, _ReturnAddress());
    QueueEntry entry{};
    AcquireSRWLockExclusive(&connection_lock);
    Connection* conn = find_connection(rcx);
    auto size = result_size;
#ifdef X2W_HAVE_OPUS
    if (conn && conn->opus_streams) size = 0x50; // the packed stream result ends after its input pointer
#endif
    if (!rdx || !x2w_local::writable(reinterpret_cast<const void*>(rdx), size)) {
        ReleaseSRWLockExclusive(&connection_lock);
        static LONG bad_out_calls;
        if (x2w_local::first_calls(bad_out_calls, 2))
            x2w_local::logf("x2w-audio XApuDequeueResult handle=0x%llx -> E_POINTER", rcx);
        return hr_pointer;
    }
    long hr = hr_handle;
    if (conn) {
        if (conn->count > 0) {
            entry = pop_result(*conn);
            hr = hr_ok;
#ifdef X2W_HAVE_OPUS
            if (entry.decoded && typed_completions())
                x2w_audio::saturating_add(opus_counts.decoded_completions);
#endif
        } else {
            hr = hr_pending_results;
        }
    }
    ReleaseSRWLockExclusive(&connection_lock);
    if (hr == hr_ok) {
        auto* out = reinterpret_cast<unsigned char*>(rdx);
        std::memset(out, 0, size);
        if (typed_completions()) {
            std::memcpy(out, &entry.type, sizeof(entry.type));
            std::memcpy(out + 4, &entry.slot, sizeof(entry.slot));
            std::memcpy(out + 0x34, &entry.value, sizeof(entry.value));
#ifdef X2W_HAVE_OPUS
            if (entry.stream_command) {
                const float pitch = 1.0f;
                std::memcpy(out + 0x08, entry.command_tag, sizeof(entry.command_tag));
                std::memcpy(out + 0x30, &entry.frames, 4);
                std::memcpy(out + 0x38, &entry.output, 8);
                std::memcpy(out + 0x40, &pitch, 4);
                std::memcpy(out + 0x44, &entry.input_bytes, 4);
                std::memcpy(out + 0x48, &entry.input, 8);
            }
#endif
        }
    }
    static LONG calls;
    if (x2w_local::first_calls(calls, 2))
        x2w_local::logf("x2w-audio XApuDequeueResult handle=0x%llx -> 0x%08lx", rcx, static_cast<unsigned long>(hr));
    return hr;
}

extern "C" __declspec(noinline) long XApuEnqueueCommand(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    if (!permissive())
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, "XApuEnqueueCommand", 3, rcx, rdx, r8, r9, _ReturnAddress());
    if (!rdx || !x2w_local::readable(reinterpret_cast<const void*>(rdx), command_header_size)) {
#ifdef X2W_HAVE_OPUS
        if (typed_completions()) {
            start_opus_counters();
            x2w_audio::saturating_add(opus_counts.refused_header);
        }
#endif
        static LONG bad_cmd_calls;
        if (x2w_local::first_calls(bad_cmd_calls, 2))
            x2w_local::logf("x2w-audio XApuEnqueueCommand handle=0x%llx -> E_POINTER", rcx);
        count_command(hr_pointer, nullptr);  // a header this call refuses as unreadable counts as unreadable
        return hr_pointer;
    }
    QueueEntry entry{};
    std::memcpy(&entry.type, reinterpret_cast<const unsigned char*>(rdx), sizeof(entry.type));
    std::memcpy(&entry.slot, reinterpret_cast<const unsigned char*>(rdx) + 4, sizeof(entry.slot));
#ifdef X2W_HAVE_OPUS
    const bool decode = entry.type == 3 && typed_completions();
    DecodeRefusal refusal = DecodeRefusal::none;
    bool decoded_packet = false;
    bool queue_refused = false;
    if (decode) {
        start_opus_counters();
        x2w_audio::saturating_add(opus_counts.type3_commands);
    }
    if (decode && !x2w_local::readable(reinterpret_cast<const void*>(rdx), decode_command_size)) {
        x2w_audio::saturating_add(opus_counts.refused_short_command);
        x2w_local::logf("x2w-audio XApuEnqueueCommand handle=0x%llx short decode command -> E_POINTER", rcx);
        count_command(hr_pointer, reinterpret_cast<const void*>(rdx));  // its header was read: counted as class 3
        return hr_pointer;
    }
#endif
    AcquireSRWLockExclusive(&connection_lock);
    Connection* conn = find_connection(rcx);
    long hr = hr_handle;
    if (conn) {
#ifdef X2W_HAVE_OPUS
        const float* pcm = nullptr;
        void* dest = nullptr;
        // Refuse a full queue before touching any decoder state.
        hr = reserve_result(*conn) ? hr_ok : hr_out_of_memory;
        queue_refused = hr != hr_ok;
        if (hr == hr_ok && conn->opus_streams && typed_completions()) {
            const auto* command = reinterpret_cast<const unsigned char*>(rdx);
            if (!x2w_local::readable(command, entry.type == 1 ? 28 : 24)) {
                hr = hr_pointer;
            } else if (entry.slot >= max_slots) {
                refusal = DecodeRefusal::invalid_argument;
                hr = hr_invalid_arg;
            } else {
                entry.stream_command = true;
                std::memcpy(entry.command_tag, command + 8, sizeof(entry.command_tag));
                if (entry.type == 1) {
                    std::uint32_t channels = 0;
                    std::memcpy(&channels, command + 24, 4);
                    if (channels != 1 && channels != 2) {
                        refusal = DecodeRefusal::invalid_argument;
                        hr = hr_invalid_arg;
                    } else if (conn->decoders[entry.slot].channels) {
                        refusal = DecodeRefusal::invalid_argument;
                        hr = hr_invalid_arg;
                    } else {
                        conn->decoders[entry.slot].channels = static_cast<int>(channels);
                    }
                }
            }
        }
        if (hr == hr_ok && decode) {
            hr = decode_packet(*conn, reinterpret_cast<const unsigned char*>(rdx), entry, &pcm, &dest,
                               decoded_packet, refusal);
        } else if (hr == hr_ok && (entry.type == 2 || entry.type == 5)) {
            drop_decoder(*conn, entry.slot, entry.type == 2);
        }
        if (hr == hr_ok) {
            if (decode) {
                entry.decoded = decoded_packet;
                if (decoded_packet) x2w_audio::saturating_add(opus_counts.packets_decoded);
            }
            place_result(*conn, entry);
            if (pcm) std::memcpy(dest, pcm, entry.value);
        }
#else
        hr = push_result(*conn, entry) ? hr_ok : hr_out_of_memory;
#endif
    }
    ReleaseSRWLockExclusive(&connection_lock);
#ifdef X2W_HAVE_OPUS
    if (decode && hr != hr_ok) {
        if (!conn)
            x2w_audio::saturating_add(opus_counts.refused_unknown_handle);
        else if (queue_refused)
            x2w_audio::saturating_add(opus_counts.refused_queue_full);
        else if (refusal == DecodeRefusal::invalid_argument)
            x2w_audio::saturating_add(opus_counts.refused_invalid_argument);
        else if (refusal == DecodeRefusal::opus_error)
            x2w_audio::saturating_add(opus_counts.refused_opus_error);
        else if (refusal == DecodeRefusal::capacity)
            x2w_audio::saturating_add(opus_counts.refused_capacity);
        else if (refusal == DecodeRefusal::out_of_memory)
            x2w_audio::saturating_add(opus_counts.refused_out_of_memory);
    }
#endif
    count_command(hr, reinterpret_cast<const void*>(rdx));
    static LONG calls;
    if (x2w_local::first_calls(calls, 24)) {
        char command[3 * 0x60 + 1];
        hex(reinterpret_cast<const void*>(rdx), 0x60, command, sizeof(command));
        const auto offset = reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr));
        x2w_local::logf("x2w-audio XApuEnqueueCommand handle=0x%llx r8=0x%llx r9=0x%llx ret=+0x%llx command=[%s] -> 0x%08lx",
                        rcx, r8, r9, offset, command, static_cast<unsigned long>(hr));
    }
    return hr;
}

extern "C" __declspec(noinline) long XApuDisconnect(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    if (!permissive())
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, "XApuDisconnect", 2, rcx, rdx, r8, r9, _ReturnAddress());
    AcquireSRWLockExclusive(&connection_lock);
    Connection** slot = &connections;
    while (*slot && (*slot)->handle != rcx) slot = &(*slot)->next;
    Connection* conn = *slot;
    if (conn) *slot = conn->next;
#ifdef X2W_HAVE_OPUS
    const bool last_connection = conn && !connections;
#endif
    if (conn && !connections && timer_active) {
        timeEndPeriod(1);
        timer_active = false;
    }
    ReleaseSRWLockExclusive(&connection_lock);
    if (!conn) {
        x2w_local::logf("x2w-audio XApuDisconnect handle=0x%llx -> E_HANDLE", rcx);
        return hr_handle;
    }
    VirtualFree(conn->buffer, 0, MEM_RELEASE);
#ifdef X2W_HAVE_OPUS
    for (std::uint32_t slot = 0; slot < max_slots; ++slot) drop_decoder(*conn, slot, true);
#endif
    delete[] conn->entries;
    delete conn;
    x2w_local::logf("x2w-audio XApuDisconnect handle=0x%llx -> S_OK", rcx);
#ifdef X2W_HAVE_OPUS
    if (last_connection) log_opus_summary();
#endif
    return hr_ok;
}

// Experimental export for the source-built check (tests/startup_acphal_xma_activity.cpp): writes the line if one is
// due, as a counted call does, and the totals since start to out[0..18]: commands of classes 0..15, of larger classes,
// unreadable commands, calls with an unknown handle. With X2W_AUDIO_ACTIVITY off it writes zeros and returns FALSE.
extern "C" BOOL WINAPI x2w_xapu_command_counts(ULONGLONG* out)
{
    if (!out) return FALSE;
    const bool on = activity_on();
    CommandCounts now{};
    char text[512];
    bool line = false;
    if (on) {
        AcquireSRWLockExclusive(&count_lock);
        line = poll_locked(text);
        now = command_counts;
        ReleaseSRWLockExclusive(&count_lock);
    }
    if (line) x2w_local::logf("%s", text);
    for (unsigned c = 0; c <= command_classes; ++c) out[c] = now.by_class[c];
    out[command_classes + 1] = now.unreadable;
    out[command_classes + 2] = now.rejected;
    return on;
}
