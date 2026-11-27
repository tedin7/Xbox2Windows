// x2w_replay: plays a .x2wr recording into the D3D12 stand-in next to
// this program. No title code runs: the recorded calls are made again, in sequence order, on the stand-in's objects, so
// the stand-in, its translator and any fix run for real.
//
// usage: x2w_replay <recording.x2wr> <out dir> [--adapter warp|hardware] [--mode tester|fixed] [--threads N|recorded]
//                   [--from-present K] [--max-seconds S] [--dump HEX_ADDRESS:BYTES] [--prefix-live on] [--pace on]
// --prefix-live on: the prefix (the calls before the window's first present) plays without the calls that leave nothing
// for the window (prefix_live.h): descriptor writes a later write to the same destination replaces, repeated Map, Unmap,
// GPU address and heap start calls, and allocation-size queries.
// The title's frame-event waits (device slot 80) are played without their timeout, so they do not sleep to the title's
// frame interval (a loading prefix spends most of its time there); --pace on keeps the recorded timeout.
// --dump writes replayed title memory to dump.bin before applying trailing recorded pages.
// Writes <out dir>\stderr.txt (the stand-in's log), present_<nnnn>.png (X2W_PRESENT_PICTURES), replay_summary.json, and
// one "x2w-replay score ..." line on stdout. Exit: 0 played to the end, 2 bad arguments or file, 3 a title range of the
// recording could not be reserved at its address, 4 the stand-in made no device, 5 stopped at --max-seconds, 6 a slot result differs,
// 77 no D3D12 adapter.
//
// Version 1 plays on one thread. Per call:
//   - the object and every argument naming a recorded wrapper get the replay's wrapper (wrappers are learned from the
//     output arguments: OUTP);
//   - argument memory is copied out of the recording into memory the replay owns, pointers between those copies are
//     rewritten (ARGM parent/word), and words inside them naming a wrapper or a descriptor handle are rewritten too;
//   - descriptor handles map heap starts over the recorded descriptor count and increment;
//   - a fence's completed value is waited for until it reaches the recorded one (10 s at most).
//   - a buffer's mapped address and GPU address map to the replay's over the buffer's width;
//   - a struct-returning slot called as (return buffer, object) is played the same way, with a buffer of its own.
//   - title memory recorded as pages (PAGE/MAP) is reserved at its own addresses first and written before each call
//     (replay_pages_apply.h); argument memory inside it is passed as is.
// Not yet: float arguments passed by value, one thread per recorded thread.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <psapi.h>
#include <wrl/client.h>

#include <cerrno>
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "replay_file.h"
#include "replay_pages_apply.h"
#include "prefix_live.h"
#include "../startup_stubs/d3d12_replay_slots.h"

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

extern "C" std::uint64_t x2w_replay_invoke(void* target, std::uint64_t rcx, const std::uint64_t* args);

namespace x2w_replay {
namespace {

// Kinds and slots as the stand-in numbers them (d3d12_xs_device.cpp enum Kind and its tables).
enum Kind : std::uint16_t { device_kind = 0, queue_kind = 1, fence_kind = 2, descriptor_heap_kind = 5, resource_kind = 7 };
constexpr std::uint16_t device_create_descriptor_heap = 14, device_increment_size = 15;
constexpr std::uint16_t heap_cpu_start = 9, heap_gpu_start = 10, fence_completed = 8, queue_present = 32;
constexpr std::uint16_t resource_map = 8, resource_get_desc = 10, resource_gpu_address = 11;
constexpr std::uint16_t frame_event_wait = 80;  // device slot 80: (type, timeout, reserved, flags, out token)
constexpr std::uint16_t queue_timestamp_frequency = 18;
constexpr std::uint32_t rcx_arg = 15;  // ARGM argument of rcx when rcx is not an object (a struct-return buffer)
constexpr int top_slots = 16;

struct Options {
    std::wstring recording, out;
    bool hardware = false;
    double max_seconds = 0;
    u64 dump_address = 0, dump_bytes = 0;
    bool prefix_live = false;  // --prefix-live on
    bool pace = false;  // --pace on
};

struct HeapInfo {
    std::uint32_t type, count;
};

struct Range {
    u64 to, span;
};

using ReplayPlace = HRESULT (*)(void*, void*, UINT64, const D3D12_RESOURCE_DESC*, UINT, ID3D12Heap**, void**);
using ReplayTitleIncrement = void (*)(UINT, UINT);

struct Player {
    ReplayPlace replay_place = nullptr;
    UINT recorded_increments[4] = {};
    std::unordered_map<u64, u64> objects;      // recorded wrapper -> replay wrapper
    std::unordered_map<u64, HeapInfo> heaps;   // recorded heap wrapper -> its creation descriptor
    std::vector<ReservedSpan> descriptor_backings;  // exact ranges of heaps created so far
    std::unordered_map<u64, Microsoft::WRL::ComPtr<ID3D12Heap>> placement_heaps;
    std::map<u64, Range> handles;              // recorded handle / mapped / GPU address start -> replay start, span
    std::vector<std::unique_ptr<unsigned char[]>> arena;  // argument memory, kept: the stand-in may hold pointers
    u64 arena_bytes = 0, played = 0, unknown_object = 0, missing_blob = 0, presents = 0, fence_timeouts = 0;
    u64 adopted = 0;  // device-kind objects first seen as a call's object, bound to the player's device
    u64 scratch_outputs = 0;  // out pointers the file did not hold, given a buffer of the player's own
    std::map<std::string, u64> skips;  // "kind:slot reason" -> calls skipped (the census printed at the end)
    u64 skip_lines = 0, faults = 0;
    // Nested leaf copies of share_min_bytes or more that translation left unchanged, by blob hash: one copy serves
    // every call that names the same bytes (pipeline code and data repeat across thousands of creations).
    std::unordered_map<u64, const unsigned char*> shared;
    u64 shared_bytes = 0, failed_creates = 0;
    std::unordered_set<u64> failed_objects;  // recorded outputs of creations that failed here
    u64 mapped = 0, gpu_addresses = 0, title_memory_args = 0;
    mutable u64 page_translated = 0;  // pages written at a translated address (a heap's CPU memory, a Map pointer)
    bool use_slot_layouts = false;
    u64 frame_waits = 0, barrier_noops = 0, upload_ranges = 0, slot_result_mismatches = 0;
    // Frame-event waits (device slot 80): played without their timeout unless pace; the seconds spent inside them.
    bool pace = false;
    u64 unpaced_waits = 0, event_waits = 0;
    double event_wait_seconds = 0;
    // The prefix: the calls before the window's first present (Recording::window_seq), how many were played, how long.
    u64 window_seq = ~0ull, prefix_calls = 0, prefix_played = 0, prefix_left_out = 0;
    double prefix_seconds = 0;
    void* device = nullptr;
    std::wstring map_dump_dir;

    bool descriptor_backing(u64 address) const
    {
        return span_contains(descriptor_backings, address);
    }

    // A recorded byte range inside a translated range: the replay's address, `bytes` clipped to that range.
    u64 page_target(u64 address, u64& bytes) const
    {
        auto above = handles.upper_bound(address);
        if (above == handles.begin()) return address;
        --above;
        const u64 offset = address - above->first;
        if (offset >= above->second.span) return address;
        if (bytes > above->second.span - offset) bytes = above->second.span - offset;
        ++page_translated;
        return above->second.to + offset;
    }

    u64 translate(u64 value) const
    {
        if (const auto found = objects.find(value); found != objects.end()) return found->second;
        if (descriptor_backing(value)) return value;
        auto above = handles.upper_bound(value);
        if (above == handles.begin()) return value;
        --above;
        return value - above->first < above->second.span ? above->second.to + (value - above->first) : value;
    }
};

// One call being played: its step, the replay object, the copies of its argument memory, and the return value.
struct Played {
    const Step* step;
    u64 self = 0;
    bool buffer_first = false;
    u64 top[top_slots]{};
    u64 before[call_args]{};  // the first word of each argument copy before the call
    bool sampled[call_args]{};
    u64 ret = 0;

    bool read_output(std::uint32_t arg, u64& word) const
    {
        return arg < call_args && top[arg] &&
               ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(top[arg]), &word, sizeof(word), nullptr);
    }

    // The call wrote a new value into argument `arg`'s copy: what an output is. A stack word past the slot's own
    // arguments can point at the same variable as the real output; its copy is never written.
    bool wrote(std::uint32_t arg, u64& now) const
    {
        return arg < call_args && sampled[arg] && read_output(arg, now) && now != before[arg];
    }
};

static_assert(call_args == 11, "replay_invoke.asm argument count");

// Calls `slot` of `object`'s table with `rcx` as the first register (the object itself, or a struct-return buffer
// when the title passed the buffer first and the object second).
u64 call_slot(u64 object, u64 rcx, std::uint16_t slot, const u64 (&a)[call_args])
{
    const auto table = *reinterpret_cast<void* const* const*>(object);
    return x2w_replay_invoke(table[slot], rcx, a);
}
u64 call_slot(u64 object, std::uint16_t slot, const u64 (&a)[call_args])
{
    return call_slot(object, object, slot, a);
}

void say(const char* format, ...)
{
    char line[1024];
    va_list args;
    va_start(args, format);
    int n = std::vsnprintf(line, sizeof(line) - 2, format, args);
    va_end(args);
    if (n < 0) return;
    if (n > static_cast<int>(sizeof(line)) - 3) n = sizeof(line) - 3;
    line[n++] = '\r';
    line[n++] = '\n';
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), line, static_cast<DWORD>(n), &written, nullptr);
}

// The first word of the recorded memory of top-level argument `arg` (as recorded, before any rewrite); 0 if none.
u64 recorded_word(const Recording& r, const Step& step, std::uint32_t arg)
{
    for (const ArgMemory& m : step.memory) {
        if (m.arg != arg || m.bytes < 8) continue;
        const auto blob = r.blobs.find(m.hash);
        if (blob == r.blobs.end()) return 0;
        u64 word;
        std::memcpy(&word, blob->second.data, 8);
        return word;
    }
    return 0;
}

constexpr u64 share_min_bytes = 4096;

// The argument memory of one call, copied into the replay's arena. Fills the copy of each top-level argument
// (top[0..10]: the register and stack arguments after rcx; top[15]: rcx when it is not an object).
void relocate(Player& p, const Recording& r, const Step& step, u64 (&top)[top_slots], std::vector<unsigned char*>& copies)
{
    copies.assign(step.memory.size(), nullptr);
    for (std::size_t i = 0; i < step.memory.size(); ++i) {
        const ArgMemory& m = step.memory[i];
        // Descriptor writes rebuild the backing in place. Other arguments on the same allocation granule still need their blobs.
        if (p.descriptor_backing(m.address) || span_contains(page_replay().map_spans, m.address)) {
            copies[i] = reinterpret_cast<unsigned char*>(m.address);
            ++p.title_memory_args;
            continue;
        }
        const auto blob = r.blobs.find(m.hash);
        if (blob == r.blobs.end() || blob->second.bytes != m.bytes) {
            ++p.missing_blob;
            continue;
        }
        // A nested leaf (depth 0: no copy points below it, nothing patches it) is input the stand-in only reads.
        const bool shareable = m.arg >= 0x100 && m.depth == 0 && m.bytes >= share_min_bytes;
        if (shareable) {
            if (const auto shared = p.shared.find(m.hash); shared != p.shared.end()) {
                copies[i] = const_cast<unsigned char*>(shared->second);
                p.shared_bytes += m.bytes;
                continue;
            }
        }
        auto copy = std::make_unique<unsigned char[]>(static_cast<std::size_t>(m.bytes) + 8);
        std::memcpy(copy.get(), blob->second.data, static_cast<std::size_t>(m.bytes));
        // Words naming a recorded wrapper, descriptor handle, mapped or GPU address name the replay's.
        bool translated = false;
        for (u64 at = 0; at + 8 <= m.bytes; at += 8) {
            u64 word;
            std::memcpy(&word, copy.get() + at, 8);
            const u64 now = p.translate(word);
            if (now != word) {
                std::memcpy(copy.get() + at, &now, 8);
                translated = true;
            }
        }
        if (shareable && !translated) p.shared.emplace(m.hash, copy.get());
        copies[i] = copy.get();
        p.arena_bytes += m.bytes;
        p.arena.push_back(std::move(copy));
    }
    // Pointers between the copies: a nested ARGM's parent is the last one before it with depth one higher. An address
    // that names a wrapper or a translated range keeps the replay's value: a CPU descriptor handle is readable memory
    // on the recorded run, so its memory was recorded too, but the call wants the handle, not a copy.
    for (std::size_t i = 0; i < step.memory.size(); ++i) {
        const ArgMemory& m = step.memory[i];
        if (!copies[i] || p.translate(m.address) != m.address) continue;
        if (m.arg < 0x100) {
            if (m.arg < top_slots) top[m.arg] = reinterpret_cast<u64>(copies[i]);
            continue;
        }
        const u64 word = (m.arg >> 8) - 1;
        for (std::size_t j = i; j-- > 0;) {
            if (step.memory[j].depth != m.depth + 1) continue;
            if (copies[j] && (word + 1) * 8 <= step.memory[j].bytes) {
                const u64 address = reinterpret_cast<u64>(copies[i]);
                std::memcpy(copies[j] + word * 8, &address, 8);
            }
            break;
        }
    }
}

// A buffer's byte width on the replay (0 for anything else): the span of its mapped and GPU addresses.
u64 buffer_bytes(u64 resource)
{
    D3D12_RESOURCE_DESC desc{};
    const u64 a[call_args] = {reinterpret_cast<u64>(&desc)};
    call_slot(resource, resource_get_desc, a);
    return desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? desc.Width : 0;
}

// Dump only bytes explicitly requested by a successful buffer Map.
void dump_mapped_range(const Player& p, const Played& call, u64 mapped, u64 span)
{
    if (p.map_dump_dir.empty() || !call.top[1] || !mapped || FAILED(static_cast<HRESULT>(call.ret))) return;
    u64 written;
    if (!call.wrote(2, written)) return;
    D3D12_RANGE range{};
    std::memcpy(&range, reinterpret_cast<const void*>(call.top[1]), sizeof(range));
    if (range.Begin >= range.End || range.End > span) return;
    const std::wstring path = p.map_dump_dir + L"\\map_" + std::to_wstring(call.step->call.seq) + L".bin";
    if (std::FILE* out = _wfopen(path.c_str(), L"wb")) {
        const auto* data = reinterpret_cast<const unsigned char*>(written) + range.Begin;
        const auto bytes = range.End - range.Begin;
        const bool wrote = std::fwrite(data, 1, bytes, out) == bytes;
        if (std::fclose(out) == 0 && wrote) return;
    }
    say("x2w-replay: could not write %ls", path.c_str());
}

// What one played call teaches: new objects, and the recorded -> replay value of returned addresses.
void learn_after(Player& p, const Recording& r, const Played& call)
{
    const Step& step = *call.step;
    const CallRecord& c = step.call;
    // Outputs: a wrapper the call put into an argument's memory, different from what was there: a new object.
    std::vector<u64> made;  // recorded wrappers this call made
    for (const Output& o : step.outputs) {
        u64 now;
        if (o.arg >= static_cast<std::uint32_t>(call_args) || !call.wrote(o.arg, now) || !now) continue;
        p.objects[o.value] = now;
        made.push_back(o.value);
    }
    if (c.kind == device_kind && c.slot == device_create_descriptor_heap && call.top[0]) {
        HeapInfo info{};
        std::memcpy(&info, reinterpret_cast<const void*>(call.top[0]), sizeof(info));
        for (const u64 heap : made) p.heaps[heap] = info;
    }
    // Struct returns through a buffer: the value is in the buffer, on both runs.
    const u64 old_value = call.buffer_first ? recorded_word(r, step, rcx_arg) : c.ret;
    u64 new_value = call.ret;
    if (call.buffer_first && call.top[rcx_arg]) std::memcpy(&new_value, reinterpret_cast<const void*>(call.top[rcx_arg]), 8);
    const u64 recorded_object = call.buffer_first ? c.args[0] : c.object;
    if (c.kind == descriptor_heap_kind && (c.slot == heap_cpu_start || c.slot == heap_gpu_start) && old_value && new_value) {
        const auto heap = p.heaps.find(recorded_object);
        u64 span = 1ull << 20;
        if (heap != p.heaps.end()) {
            const u64 a[call_args] = {heap->second.type};
            const u64 increment = heap->second.type < 4 && p.recorded_increments[heap->second.type] ? p.recorded_increments[heap->second.type] :
                call_slot(reinterpret_cast<u64>(p.device), device_increment_size, a) & 0xffffffffu;
            span = static_cast<u64>(heap->second.count) * (increment ? increment : 64);
        }
        p.handles[old_value] = Range{new_value, span ? span : 1};
    }
    // A buffer's mapped address (Map's third argument) and its GPU address map over the buffer's width.
    const bool map = c.kind == resource_kind && c.slot == resource_map && call.top[2];
    const bool gpu = c.kind == resource_kind && c.slot == resource_gpu_address;
    if (map || gpu) {
        const u64 old_address = map ? recorded_word(r, step, 2) : c.ret;
        u64 new_address = call.ret;
        if (map) std::memcpy(&new_address, reinterpret_cast<const void*>(call.top[2]), 8);
        const u64 span = buffer_bytes(call.self);
        if (map) dump_mapped_range(p, call, new_address, span);
        if (old_address && new_address && span) {
            p.handles[old_address] = Range{new_address, span};
            ++(map ? p.mapped : p.gpu_addresses);
        }
    }
}

// What the player is doing, for the crash line: a phase name, and the call being played.
const char* volatile phase = "start";
const Step* volatile current_step = nullptr;

// The crash line: the exception, the module and offset it happened in, the address an access violation touched, and
// what the player was doing. Written from a vectored handler (it runs first, before any frame or filter): an
// exception some code handles itself is reported too, so the line says "first chance" and at most four are written.
LONG WINAPI note_crash(EXCEPTION_POINTERS* e)
{
    const EXCEPTION_RECORD& x = *e->ExceptionRecord;
    if (x.ExceptionCode < 0xC0000000ul || x.ExceptionCode == 0xE06D7363ul) return EXCEPTION_CONTINUE_SEARCH;  // C++ throw
    static volatile LONG reported = 0;
    if (InterlockedIncrement(&reported) > 4) return EXCEPTION_CONTINUE_SEARCH;
    char module[MAX_PATH] = "?";
    HMODULE owner = nullptr;
    u64 offset = reinterpret_cast<u64>(x.ExceptionAddress);
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<const char*>(x.ExceptionAddress), &owner) &&
        GetModuleFileNameA(owner, module, MAX_PATH)) {
        offset -= reinterpret_cast<u64>(owner);
    }
    const char* base = std::strrchr(module, '\\');
    base = base ? base + 1 : module;
    const u64 touched = x.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && x.NumberParameters >= 2 ? x.ExceptionInformation[1] : 0;
    const Step* step = current_step;
    if (step)
        say("x2w-replay: CRASH (first chance) 0x%08lx at %s+0x%llx touching 0x%llx while playing seq %llu (kind %u slot %u, "
            "object 0x%llx)", x.ExceptionCode, base, offset, touched, step->call.seq, step->call.kind, step->call.slot,
            step->call.object);
    else
        say("x2w-replay: CRASH (first chance) 0x%08lx at %s+0x%llx touching 0x%llx during %s", x.ExceptionCode, base, offset,
            touched, phase);
    return EXCEPTION_CONTINUE_SEARCH;
}

// Only slots wired to the stand-in's struct-return receiver accept an object in RDX.
// Other calls can leave a live wrapper in that register even when they take no arguments.
bool accepts_buffer_first(const CallRecord& call)
{
    if (call.kind == device_kind) return call.slot == 25 || call.slot == 26 || call.slot == 108;
    if (call.kind == descriptor_heap_kind) return call.slot == heap_cpu_start || call.slot == heap_gpu_start;
    return call.kind == resource_kind && call.slot == resource_get_desc;
}

// A call the player cannot play safely is skipped: counted in the census by kind, slot and reason, the first few logged.
bool skip(Player& p, const CallRecord& c, const char* reason)
{
    char key[96];
    std::snprintf(key, sizeof(key), "%u:%u %s", c.kind, c.slot, reason);
    if (++p.skips[key] <= 2 && ++p.skip_lines <= 64)
        say("replay: skipped %s (seq %llu)", key, static_cast<unsigned long long>(c.seq));
    return false;
}

// A creation or QueryInterface recorded as a success that fails here: played, and counted in the census, because every
// later call on its result will be an unknown object.
// The process's commit charge against its limit, the argument copies and the title ranges: a creation that fails
// with E_OUTOFMEMORY on a long file says whether the host ran out.
void say_commit(const Player& p, const char* when)
{
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    if (!GlobalMemoryStatusEx(&m)) return;
    u64 title = 0;
    for (const ReservedSpan& s : page_replay().spans) title += s.end - s.base;
    say("x2w-replay: commit %s: %llu of %llu MiB used, argument copies %llu MiB (%llu MiB shared), title ranges %llu MiB",
        when, static_cast<unsigned long long>((m.ullTotalPageFile - m.ullAvailPageFile) >> 20),
        static_cast<unsigned long long>(m.ullTotalPageFile >> 20), static_cast<unsigned long long>(p.arena_bytes >> 20),
        static_cast<unsigned long long>(p.shared_bytes >> 20), static_cast<unsigned long long>(title >> 20));
}

void note_failed_here(Player& p, const Step& step, u64 ret)
{
    const CallRecord& c = step.call;
    if (x2w_replay_slots::output_arg(c.kind, c.slot) < 0 || FAILED(static_cast<HRESULT>(c.ret)) ||
        SUCCEEDED(static_cast<HRESULT>(ret)))
        return;
    for (const Output& output : step.outputs) p.failed_objects.insert(output.value);
    if (++p.failed_creates == 1 || p.failed_creates % 500 == 0) {
        char when[64];
        std::snprintf(when, sizeof(when), "at failed create %llu (hr 0x%08lx)", static_cast<unsigned long long>(p.failed_creates),
                      static_cast<unsigned long>(ret));
        say_commit(p, when);
    }
    char key[96];
    std::snprintf(key, sizeof(key), "%u:%u failed here (recorded ok)", c.kind, c.slot);
    if (++p.skips[key] <= 2 && ++p.skip_lines <= 64)
        say("replay: %s (seq %llu, hr 0x%08lx)", key, static_cast<unsigned long long>(c.seq), static_cast<unsigned long>(ret));
}

// The slot call with an access violation in the stand-in caught (MSVC structured exceptions): the recorded process's
// state the file did not hold, an address the player could not translate, ends one call instead of the replay.
bool guarded_slot(u64 object, u64 rcx, std::uint16_t slot, const u64 (&a)[call_args], u64& ret)
{
#ifdef _MSC_VER
    __try {
        ret = call_slot(object, rcx, slot, a);
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
#else
    ret = call_slot(object, rcx, slot, a);
#endif
    return true;
}

// Plays one call. false: skipped (counted in the census).
// The argument a slot writes its result through: a wrapper output (x2w_replay_slots::output_arg), or a value the stand-in
// writes for the caller (command queue slot 18, GetTimestampFrequency).
int written_arg(const CallRecord& c)
{
    if (c.kind == queue_kind && c.slot == queue_timestamp_frequency) return 0;
    return x2w_replay_slots::output_arg(c.kind, c.slot);
}

bool play(Player& p, const Recording& r, const Step& step)
{
    current_step = &step;
    const CallRecord& c = step.call;
    Played call{&step};
    if (const auto object = p.objects.find(c.object); object != p.objects.end()) {
        call.self = object->second;
    } else if (const auto second = p.objects.find(c.args[0]); accepts_buffer_first(c) && second != p.objects.end()) {
        call.self = second->second;  // (return buffer, object): the struct-return form the stand-in accepts too
        call.buffer_first = true;
    } else if (c.kind == device_kind && !accepts_buffer_first(c)) {
        // A device-kind object created before the recorder installed (no OUTP names it): the player's device.
        p.objects[c.object] = reinterpret_cast<u64>(p.device);
        call.self = reinterpret_cast<u64>(p.device);
        if (++p.adopted <= 8) say("replay: adopted pre-install wrapper %u 0x%llx", c.kind, static_cast<unsigned long long>(c.object));
    } else {
        ++p.unknown_object;
        return skip(p, c, p.failed_objects.count(c.object) ? "dependent of failed create" : "unknown object");
    }
    // A creation or QueryInterface that failed when recorded made nothing: the out pointer may be the title's stack and
    // the inputs may not be in the file. Nothing later can name its result, so it is not played.
    if (x2w_replay_slots::output_arg(c.kind, c.slot) >= 0 && FAILED(static_cast<HRESULT>(c.ret)))
        return skip(p, c, "recorded failure");
    apply_pages_before(r, c.seq, [&p](u64 address, u64& bytes) { return p.page_target(address, bytes); });
    std::vector<unsigned char*> copies;
    relocate(p, r, step, call.top, copies);
    for (const Output& output : step.outputs)
        if (output.arg < call_args)
            call.sampled[output.arg] = call.read_output(output.arg, call.before[output.arg]);
    // Map's CPU address is an output too, but not a wrapper, so it has no OUTP: sample it for the mapped dump.
    if (c.kind == resource_kind && c.slot == resource_map)
        call.sampled[2] = call.read_output(2, call.before[2]);
    u64 a[call_args];
    for (int i = 0; i < call_args; ++i) a[i] = call.top[i] ? call.top[i] : p.translate(c.args[i]);
    // An argument the stand-in reads as memory (the recorder's declared captures) that the file does not hold: passing
    // the recorded process's address would fault in the stand-in. The call is skipped and counted as a call without its
    // memory. Other words are left alone: the unused register and stack words of a call often hold stale image or
    // stack addresses the slot never reads.
    for (int i = 0; i < call_args; ++i) {
        const u64 raw = c.args[i];
        if (call.top[i] || !raw || a[i] != raw || page_replay().reserved(raw) || p.descriptor_backing(raw)) continue;
        // An object whose creation failed here: the stand-in would take the recorded address for a wrapper.
        if (p.failed_objects.count(raw)) return skip(p, c, "argument from a failed create");
        if (x2w_replay_slots::declared_input(c.kind, c.slot, i)) {
            ++p.missing_blob;
            static const char* const reasons[call_args] = {
                "argument 0 not in the file", "argument 1 not in the file", "argument 2 not in the file",
                "argument 3 not in the file", "argument 4 not in the file", "argument 5 not in the file",
                "argument 6 not in the file", "argument 7 not in the file", "argument 8 not in the file",
                "argument 9 not in the file", "argument 10 not in the file"};
            return skip(p, c, reasons[i]);
        }
    }
    // An out pointer the file does not hold (older files dropped it when the call failed) is the recorded process's
    // stack address: the stand-in would write through it. It gets a zeroed buffer of the player's own.
    if (const int out = written_arg(c);
        out >= 0 && out < call_args && !call.top[out] && c.args[out] && a[out] == c.args[out] && !page_replay().reserved(a[out])) {
        auto scratch = std::make_unique<unsigned char[]>(16);
        std::memset(scratch.get(), 0, 16);
        a[out] = reinterpret_cast<u64>(scratch.get());
        p.arena.push_back(std::move(scratch));
        ++p.scratch_outputs;
    }
    u64 rcx = call.self;
    if (call.buffer_first) {
        if (!call.top[rcx_arg]) {
            auto scratch = std::make_unique<unsigned char[]>(64);
            std::memset(scratch.get(), 0, 64);
            call.top[rcx_arg] = reinterpret_cast<u64>(scratch.get());
            p.arena.push_back(std::move(scratch));
        }
        rcx = call.top[rcx_arg];
    }
    // A CopyDescriptors call whose arrays were recorded exactly (DARR) runs from them, in the prefix and in the window.
    if (const auto* arrays = step.descriptor_arrays) {
        std::vector<u64> handles[2];
        std::vector<std::uint32_t> sizes[2];
        for (unsigned side = 0; side < 2; ++side) {
            const unsigned base = side * 3;
            const auto count = static_cast<std::uint32_t>(c.args[base]);
            a[base + 1] = a[base + 2] = 0;
            if (count && c.args[base + 1]) {
                auto& translated = handles[side];
                translated.resize(count);
                for (u64& handle : translated) {
                    std::memcpy(&handle, arrays, sizeof(handle));
                    arrays += sizeof(handle);
                    handle = p.translate(handle);
                }
                a[base + 1] = reinterpret_cast<u64>(translated.data());
            }
            if (count && c.args[base + 2]) {
                auto& lengths = sizes[side];
                lengths.resize(count);
                const auto bytes = lengths.size() * sizeof(lengths[0]);
                std::memcpy(lengths.data(), arrays, bytes);
                arrays += bytes;
                a[base + 2] = reinterpret_cast<u64>(lengths.data());
            }
        }
        u64 ignored = 0;
        if (!guarded_slot(call.self, rcx, c.slot, a, ignored)) {
            ++p.faults;
            return skip(p, c, "fault in the stand-in");
        }
        ++p.played;
        return true;
    }
    // Older recordings can supply a generic ARGM for a descriptor array.
    // Skip a copy with missing arrays instead of passing captured process addresses to the stand-in.
    if (c.kind == device_kind && c.slot == 23) {
        for (const unsigned i : {1u, 2u, 4u, 5u}) {
            if (static_cast<UINT>(c.args[i < 3 ? 0 : 3]) && c.args[i] && a[i] == c.args[i] &&
                !page_replay().reserved(a[i])) {
                ++p.missing_blob;
                return skip(p, c, "descriptor arrays not in the file");
            }
        }
    }
    const auto* layout = p.use_slot_layouts ? x2w_replay_slots::find(c.kind, c.slot) : nullptr;
    // The frame-event wait sleeps until the title's next frame time when it has a timeout: the replay does not wait for
    // a frame clock (the token it returns still counts up).
    const bool event_wait = c.kind == device_kind && c.slot == frame_event_wait;
    if (event_wait && !p.pace && a[1]) {
        a[1] = 0;
        ++p.unpaced_waits;
    }
    const auto wait_started = event_wait ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool wait = layout && c.kind == device_kind && c.slot == 80;
    const bool barrier_noop = layout && c.kind == x2w_replay_slots::list && c.slot == 146;
    constexpr int token_arg = 4;
    const u64 expected_token = wait ? recorded_word(r, step, token_arg) : 0;
    u64 token = ~expected_token;
    // The recording holds the post-call token. Start with a different value to prove that replay writes it.
    if (wait && c.args[token_arg]) a[token_arg] = reinterpret_cast<u64>(&token);
    struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        u64 address, bytes;
    } heap_desc{};
    if (step.descriptor_bytes && call.top[0]) {
        std::memcpy(&heap_desc.desc, reinterpret_cast<const void*>(call.top[0]), sizeof(heap_desc.desc));
        heap_desc.address = step.descriptor_address;
        heap_desc.bytes = step.descriptor_bytes;
        a[0] = reinterpret_cast<u64>(&heap_desc);
    }
    if (const auto& placement = step.placement; placement.id) {
        auto& heap = p.placement_heaps[placement.id];
        const HRESULT hr = p.replay_place(
            reinterpret_cast<void*>(call.self), reinterpret_cast<void*>(placement.base), placement.offset,
            reinterpret_cast<const D3D12_RESOURCE_DESC*>(call.top[1]), static_cast<UINT>(c.args[2]), heap.GetAddressOf(),
            reinterpret_cast<void**>(call.top[5]));
        call.ret = static_cast<u64>(hr);
        say("x2w-replay: slot 133 seq %llu heap id %llu offset 0x%llx hr=0x%08lx",
            c.seq, placement.id, placement.offset, static_cast<unsigned long>(hr));
        if (FAILED(hr)) ++p.slot_result_mismatches;
    } else if (!guarded_slot(call.self, rcx, c.slot, a, call.ret)) {
        ++p.faults;
        return skip(p, c, "fault in the stand-in");
    }
    if (event_wait) {
        ++p.event_waits;
        p.event_wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_started).count();
    }
    note_failed_here(p, step, call.ret);
    if (step.descriptor_bytes && SUCCEEDED(static_cast<HRESULT>(call.ret))) {
        p.descriptor_backings.push_back({step.descriptor_address, step.descriptor_address + step.descriptor_bytes});
        merge_spans(p.descriptor_backings);
    }
    if (wait || barrier_noop) {
        p.frame_waits += wait;
        p.barrier_noops += barrier_noop;
        const bool output_differs = wait && c.args[token_arg] && token != expected_token;
        const bool return_differs = wait ? static_cast<UINT>(call.ret) != static_cast<UINT>(c.ret) : call.ret != c.ret;
        if (output_differs || return_differs) {
            ++p.slot_result_mismatches;
            say("x2w-replay: %s result mismatch at seq %llu", layout->name, c.seq);
        }
    }
    if (layout && c.kind == x2w_replay_slots::list && c.slot == 148 && c.ret && call.ret && c.args[0]) {
        p.handles[c.ret] = Range{call.ret, c.args[0]};
        ++p.upload_ranges;
    }
    if (c.kind == fence_kind && c.slot == fence_completed && call.ret < c.ret) {
        for (const ULONGLONG deadline = GetTickCount64() + 10000; call.ret < c.ret && GetTickCount64() < deadline; Sleep(1))
            call.ret = call_slot(call.self, rcx, c.slot, a);
        if (call.ret < c.ret) ++p.fence_timeouts;
    }
    if (c.kind == queue_kind && c.slot == queue_present) ++p.presents;
    learn_after(p, r, call);
    ++p.played;
    return true;
}

// The process's peak private bytes (commit charge) so far, or 0.
u64 peak_private_bytes()
{
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    return GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters))
               ? static_cast<u64>(counters.PeakPagefileUsage) : 0;
}

// The skip census: every (kind:slot, reason) the player skipped, most frequent first.
void print_skip_census(const Player& p)
{
    u64 total = 0;
    std::vector<std::pair<u64, std::string>> rows;
    for (const auto& [key, count] : p.skips) {
        total += count;
        rows.emplace_back(count, key);
    }
    std::sort(rows.begin(), rows.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    say("x2w-replay: skip census: %llu calls skipped or failed here in %zu kinds (%llu faults in the stand-in)",
        static_cast<unsigned long long>(total), rows.size(), static_cast<unsigned long long>(p.faults));
    for (std::size_t i = 0; i < rows.size() && i < 32; ++i)
        say("x2w-replay: skip census %s: %llu", rows[i].second.c_str(), static_cast<unsigned long long>(rows[i].first));
}

int pick_adapter(bool hardware, Microsoft::WRL::ComPtr<IDXGIAdapter>& adapter)
{
    Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 77;
    if (!hardware) return SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ? 0 : 77;
    for (UINT n = 0;; ++n) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> candidate;
        if (FAILED(factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate))))
            return 77;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(candidate->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            adapter = candidate;
            return 0;
        }
    }
}

bool parse_dump(const std::wstring& value, Options& o)
{
    const auto colon = value.find(L':');
    if (colon == std::wstring::npos || colon == 0 || colon + 1 == value.size()) return false;
    const auto address = value.substr(0, colon);
    const auto digits = address.compare(0, 2, L"0x") == 0 || address.compare(0, 2, L"0X") == 0 ? 2u : 0u;
    if (digits == address.size() || address.find_first_not_of(L"0123456789abcdefABCDEF", digits) != std::wstring::npos ||
        value.find_first_not_of(L"0123456789", colon + 1) != std::wstring::npos) return false;
    errno = 0;
    o.dump_address = _wcstoui64(address.c_str(), nullptr, 16);
    if (errno == ERANGE || !o.dump_address) return false;
    o.dump_bytes = _wcstoui64(value.c_str() + colon + 1, nullptr, 10);
    return o.dump_bytes && o.dump_bytes <= (u64{64} << 20);
}

bool dump_title_range(const Options& o)
{
    const std::wstring path = o.out + L"\\dump.bin";
    if (std::FILE* out = _wfopen(path.c_str(), L"wb")) {
        const auto bytes = static_cast<std::size_t>(o.dump_bytes);
        const bool wrote = std::fwrite(reinterpret_cast<const void*>(o.dump_address), 1, bytes, out) == bytes;
        if (std::fclose(out) == 0 && wrote) return true;
        DeleteFileW(path.c_str());
    }
    say("x2w-replay: could not write %ls", path.c_str());
    return false;
}

bool parse(int argc, wchar_t** argv, Options& o)
{
    if (argc < 3) return false;
    o.recording = argv[1];
    o.out = argv[2];
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) return false;
        const std::wstring name = argv[i], value = argv[i + 1];
        if (name == L"--adapter" && (value == L"warp" || value == L"hardware")) o.hardware = value == L"hardware";
        else if (name == L"--max-seconds") o.max_seconds = _wtof(value.c_str());
        else if (name == L"--prefix-live" && value == L"on") o.prefix_live = true;
        else if (name == L"--pace" && value == L"on") o.pace = true;
        else if (name == L"--dump") {
            if (!parse_dump(value, o)) return false;
        }
        else if (name == L"--mode" || name == L"--threads" || name == L"--from-present") say("x2w-replay: %ls %ls is not in version 1 (ignored)", name.c_str(), value.c_str());
        else return false;
    }
    return true;
}

using ReplayClock = std::chrono::steady_clock;
using SlotKey = std::pair<std::uint16_t, std::uint16_t>;
using Microseconds = std::chrono::duration<double, std::micro>;

struct SlotTiming {
    ReplayClock::duration total{}, interval{};
    u64 calls = 0;
};

struct ReplayTiming {
    std::map<SlotKey, SlotTiming> slots;
    ReplayClock::time_point next_report;
    u64 previous_played = 0, previous_presents = 0;

    void add(const CallRecord& call, ReplayClock::duration duration)
    {
        auto& slot = slots[{call.kind, call.slot}];
        slot.total += duration;
        ++slot.calls;
        slot.interval += duration;
    }

    void report(ReplayClock::time_point now, const Player& p)
    {
        if (now < next_report) return;
        // A blocking call can cross multiple deadlines. Report its cost when it returns.
        next_report += std::chrono::duration_cast<std::chrono::seconds>(now - next_report) + std::chrono::seconds(1);
        std::vector<std::pair<SlotKey, ReplayClock::duration>> ranked;
        for (auto& [key, timing] : slots) {
            if (timing.interval.count() > 0) ranked.emplace_back(key, timing.interval);
            timing.interval = {};
        }
        const std::size_t count = (std::min)(ranked.size(), std::size_t{5});
        std::partial_sort(ranked.begin(), ranked.begin() + count, ranked.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });
        std::printf("x2w-replay timing calls %llu presents %llu top_us", p.played - previous_played,
                    p.presents - previous_presents);
        for (std::size_t i = 0; i < count; ++i) {
            const auto& [key, duration] = ranked[i];
            std::printf(" %u:%u=%.3f", static_cast<unsigned>(key.first), static_cast<unsigned>(key.second),
                        Microseconds(duration).count());
        }
        std::putchar('\n');
        std::fflush(stdout);
        previous_played = p.played;
        previous_presents = p.presents;
    }

    void write_json(std::FILE* f) const
    {
        std::fprintf(f, ", \"slot_times\": [");
        const char* separator = "";
        for (const auto& [key, timing] : slots) {
            std::fprintf(f, "%s{\"kind\": %u, \"slot\": %u, \"calls\": %llu, \"total_us\": %.3f}", separator,
                         static_cast<unsigned>(key.first), static_cast<unsigned>(key.second), timing.calls,
                         Microseconds(timing.total).count());
            separator = ", ";
        }
        std::fprintf(f, "]");
    }
};

void write_summary(const Options& o, const Player& p, const Recording& r, double wall, bool stopped,
                   const ReplayTiming* timing)
{
    const double fps = wall > 0 ? static_cast<double>(p.presents) / wall : 0;
    const std::wstring path = o.out + L"\\replay_summary.json";
    if (std::FILE* f = _wfopen(path.c_str(), L"w")) {
        std::fprintf(f, "{\"version\": %u, \"calls\": %zu, \"played\": %llu, \"unknown_object\": %llu, \"missing_blob\": %llu, "
                        "\"fence_timeouts\": %llu, \"presents\": %llu, \"wall_s\": %.*f, \"fps\": %.2f, \"argument_bytes\": %llu, "
                        "\"shared_argument_bytes\": %llu, \"peak_private_bytes\": %llu, \"stopped_at_max_seconds\": %s, "
                        "\"frame_event_waits\": {\"calls\": %llu, \"not_paced\": %llu, \"wall_s\": %.3f}, ",
                     r.version, r.steps.size(), p.played, p.unknown_object, p.missing_blob, p.fence_timeouts, p.presents, timing ? 9 : 3, wall, fps,
                     p.arena_bytes, p.shared_bytes, peak_private_bytes(), stopped ? "true" : "false", p.event_waits,
                     p.unpaced_waits, p.event_wait_seconds);
        const PageReplay& pages = page_replay();
        std::fprintf(f, "\"pages\": {\"maps\": %zu, \"applied\": %llu, \"written\": %llu, \"zero\": %llu, \"missing\": %llu, "
                        "\"bad_maps\": %llu, \"title_memory_args\": %llu}, \"ranges\": [",
                     r.maps.size(), pages.maps_applied, pages.pages_written, pages.zero_pages, pages.missing_pages, pages.bad_maps,
                     p.title_memory_args);
        for (std::size_t i = 0; i < pages.spans.size(); ++i)
            std::fprintf(f, "%s{\"address\": \"0x%llx\", \"bytes\": %llu, \"hash\": \"%016llx\"}", i ? ", " : "", pages.spans[i].base,
                         pages.spans[i].end - pages.spans[i].base, span_hash(pages.spans[i]));
        std::fprintf(f, "]");
        if (p.use_slot_layouts)
            std::fprintf(f, ", \"slot_layouts\": {\"frame_waits\": %llu, \"barrier_noops\": %llu, "
                            "\"upload_ranges\": %llu, \"result_mismatches\": %llu}",
                         p.frame_waits, p.barrier_noops, p.upload_ranges, p.slot_result_mismatches);
        std::fprintf(f, ", \"prefix\": {\"calls\": %llu, \"played\": %llu, \"left_out\": %llu, \"wall_s\": %.3f}",
                     p.prefix_calls, p.prefix_played, p.prefix_left_out, p.prefix_seconds);
        if (timing) timing->write_json(f);
        std::fprintf(f, "}\n");
        std::fclose(f);
    }
    std::printf("x2w-replay score presents %llu calls %llu of %zu unknown_object %llu missing_blob %llu fence_timeouts %llu "
                "wall_s %.3f fps %.2f%s\n",
                p.presents, p.played, r.steps.size(), p.unknown_object, p.missing_blob, p.fence_timeouts, wall, fps,
                stopped ? " stopped_at_max_seconds" : "");
    std::fflush(stdout);
}

// The recorded process's X2W_ settings (XENV), for every one this process does not set itself. Paths of the recording
// machine (the shader cache, files and folders) are left out: the replay has its own.
void apply_settings(const std::vector<std::string>& settings)
{
    int applied = 0, kept = 0;
    for (const std::string& entry : settings) {
        const std::size_t equals = entry.find('=');
        if (equals == std::string::npos || equals == 0) continue;
        const std::string name = entry.substr(0, equals), value = entry.substr(equals + 1);
        const auto ends = [&](const char* tail) {
            const std::size_t n = std::strlen(tail);
            return name.size() >= n && name.compare(name.size() - n, n, tail) == 0;
        };
        if (name == "X2W_SHADER_CACHE" || name == "X2W_PRESENT_PICTURES" || name == "X2W_GFX_APPROX" ||
            ends("_DIR") || ends("_FILE") || ends("_PATH") || ends("_LOG"))
            continue;
        if (GetEnvironmentVariableA(name.c_str(), nullptr, 0)) {
            ++kept;
            continue;
        }
        SetEnvironmentVariableA(name.c_str(), value.c_str());
        ++applied;
    }
    say("x2w-replay: %d recorded settings applied, %d already set here", applied, kept);
}

// Writes the recording's shader cache files (SHDR) that the cache in use does not hold yet. A name with a path
// separator is refused.
void seed_shader_cache(const Recording& r)
{
    wchar_t dir[MAX_PATH];
    const DWORD n = GetEnvironmentVariableW(L"X2W_SHADER_CACHE", dir, MAX_PATH);
    if (!n || n >= MAX_PATH) return;
    CreateDirectoryW(dir, nullptr);
    int written = 0, present = 0, refused = 0;
    for (const ShaderFile& file : r.shader_files) {
        wchar_t name[MAX_PATH];
        const int length = MultiByteToWideChar(CP_UTF8, 0, file.name.data(), static_cast<int>(file.name.size()), name, MAX_PATH - 1);
        if (length <= 0 || file.name.find_first_of("\\/:") != std::string::npos || file.name == "..") {
            ++refused;
            continue;
        }
        name[length] = 0;
        const std::wstring path = std::wstring(dir) + L"\\" + name;
        const HANDLE f = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) {
            ++present;
            continue;
        }
        DWORD done = 0;
        const bool ok = WriteFile(f, file.content.data, static_cast<DWORD>(file.content.bytes), &done, nullptr) && done == file.content.bytes;
        CloseHandle(f);
        if (ok) ++written;
        else DeleteFileW(path.c_str());
    }
    say("x2w-replay: shader cache %ls: %d files from the recording, %d already there, %d refused", dir, written, present, refused);
}

// --prefix-live: the prefix calls in sequence order, as prefix_live.h reads them. Logged with its counts.
PrefixLive prefix_live_of(const Recording& r, u64 window_seq)
{
    PrefixLive live;
    Step step;
    std::vector<u64> outputs;
    for (const StepRef& ref : r.steps) {
        if (ref.seq >= window_seq) break;
        r.load(ref, step);
        outputs.clear();
        for (const Output& o : step.outputs) outputs.push_back(o.value);
        LiveCall c{step.call.seq, step.call.kind, step.call.slot, step.call.object, {}, outputs.data(), outputs.size()};
        std::memcpy(c.args, step.call.args, sizeof(c.args));
        live.add(c);
    }
    const PrefixLiveCounts& n = live.counts();
    say("x2w-replay: prefix live: of %llu prefix calls, %llu descriptor writes a later write replaces, %llu repeated Map, Unmap, "
        "GPU address or heap start calls and %llu allocation-size queries are left out",
        static_cast<unsigned long long>(n.calls), static_cast<unsigned long long>(n.superseded_writes),
        static_cast<unsigned long long>(n.repeated_teaching), static_cast<unsigned long long>(n.queries));
    return live;
}

int run(const Options& o)
{
    // Read the player switch before applying settings from the recording.
    wchar_t timing_switch[2]{};
    const bool time_slots = GetEnvironmentVariableW(L"X2W_REPLAY_TIMING", timing_switch, 2) == 1
                            && timing_switch[0] == L'1';
    CreateDirectoryW(o.out.c_str(), nullptr);
    const std::wstring log_path = o.out + L"\\stderr.txt";
    const HANDLE log = CreateFileW(log_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "x2w-replay: cannot write %ls\n", log_path.c_str());
        return 2;
    }
    SetStdHandle(STD_ERROR_HANDLE, log);
    AddVectoredExceptionHandler(1, note_crash);
    phase = "reading the recording";
    Recording r;
    if (const std::string problem = read_recording(o.recording.c_str(), r); !problem.empty()) {
        say("x2w-replay: %s", problem.c_str());
        std::printf("x2w-replay: %s\n", problem.c_str());
        return 2;
    }
    // Reserve title ranges before the stand-in can allocate at those addresses.
    if (std::string problem; !reserve_page_ranges(r, problem)) {
        say("x2w-replay: %s", problem.c_str());
        std::printf("x2w-replay: %s\n", problem.c_str());
        return 3;
    }
    for (const HeapExtent& heap : page_replay().heap_extents)
        say("x2w-replay: heap %llu opened %llu of %llu MiB", static_cast<unsigned long long>(heap.id),
            static_cast<unsigned long long>(heap.used >> 20), static_cast<unsigned long long>(heap.bytes >> 20));
    if (o.dump_bytes && !page_replay().reserved(o.dump_address, o.dump_bytes)) {
        say("x2w-replay: dump range 0x%llx + %llu is outside the recording's title pages", o.dump_address, o.dump_bytes);
        return 3;
    }
    {
        u64 shader_bytes = 0, reserved = 0;
        for (const ShaderFile& f : r.shader_files) shader_bytes += f.content.bytes;
        for (const ReservedSpan& s : page_replay().spans) reserved += s.end - s.base;
        say("x2w-replay: recording format %u, %llu bytes: %zu calls, %zu shader files (%llu bytes), %zu pages, %zu page maps; "
            "%zu title ranges reserved (%llu bytes)", r.version, r.size, r.steps.size(), r.shader_files.size(), shader_bytes,
            r.pages.size(), r.maps.size(), page_replay().spans.size(), reserved);
    }
    wchar_t dump_maps[4]{};
    const bool dump_map_ranges = GetEnvironmentVariableW(L"X2W_REPLAY_DUMP_MAP_RANGES", dump_maps, 4) == 1 && dump_maps[0] == L'1';
    phase = "applying settings";
    // X2W_REPLAY_DUMP reaches the delay-loaded shim from the environment or the recording's XENV.
    apply_settings(r.settings);
    if (!GetEnvironmentVariableW(L"X2W_PRESENT_PICTURES", nullptr, 0)) SetEnvironmentVariableW(L"X2W_PRESENT_PICTURES", o.out.c_str());
    if (!GetEnvironmentVariableW(L"X2W_SHADER_CACHE", nullptr, 0))
        SetEnvironmentVariableW(L"X2W_SHADER_CACHE", (o.out + L"\\shader_cache").c_str());
    phase = "seeding the shader cache";
    seed_shader_cache(r);
    phase = "making the device";
    say("x2w-replay: %zu calls, %zu blobs, %llu presents marked, %s", r.steps.size(), r.blobs.size(), r.presents_marked,
        r.ended ? "ended" : "no end chunk (the recorded process stopped early)");
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    if (pick_adapter(o.hardware, adapter)) {
        std::printf("x2w-replay: no D3D12 %s adapter\n", o.hardware ? "hardware" : "WARP");
        return 77;
    }
    Player p;
    const std::string slot_layout_setting = std::string(x2w_replay_slots::setting) + "=1";
    for (const std::string& entry : r.settings)
        if (entry == slot_layout_setting) p.use_slot_layouts = true;
    if (dump_map_ranges) p.map_dump_dir = o.out;
    unsigned char parameters[64]{};
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters, __uuidof(ID3D12Device), &p.device)) || !p.device) {
        say("x2w-replay: the stand-in made no device");
        return 4;
    }
    const HMODULE shim = GetModuleHandleW(L"d3d12_xs.dll");
    p.replay_place = reinterpret_cast<ReplayPlace>(GetProcAddress(shim, "x2w_d3d12_replay_place"));
    const auto replay_title_increment = reinterpret_cast<ReplayTitleIncrement>(
        GetProcAddress(shim, "x2w_d3d12_replay_title_increment"));
    if (!r.placed_steps.empty() && !p.replay_place) {
        say("x2w-replay: recorded heap placements require x2w_d3d12_replay_place");
        return 4;
    }
    // The recorded device: the object most device-kind calls are made on. A struct-return call (slots 25, 26, 108) names
    // the title's return buffer as its object and the device second, so those calls do not vote; a recording that starts
    // with one must not bind a stack address as the device.
    {
        std::unordered_map<u64, u64> votes;
        u64 device = 0, best = 0;
        for (const StepRef& ref : r.steps)
            if (const CallRecord c = r.call_of(ref); c.kind == device_kind && !accepts_buffer_first(c) && ++votes[c.object] > best) {
                best = votes[c.object];
                device = c.object;
            }
        if (device) p.objects[device] = reinterpret_cast<u64>(p.device);
        // XOBJ: devices the recorder saw made outside the recorded slots (D3D12XboxCreateDevice) are the player's device.
        for (const auto& [kind, object] : r.live_objects)
            if (kind == device_kind) p.objects[object] = reinterpret_cast<u64>(p.device);
    }
    bool increment_conflict[4] = {};
    for (const StepRef& ref : r.steps) {
        const CallRecord c = r.call_of(ref);
        if (c.kind != device_kind || c.slot != device_increment_size || c.args[0] >= 4) continue;
        const UINT type = static_cast<UINT>(c.args[0]), value = static_cast<UINT>(c.ret);
        if (!value) continue;
        if (!p.recorded_increments[type]) p.recorded_increments[type] = value;
        else if (p.recorded_increments[type] != value && !increment_conflict[type]) {
            increment_conflict[type] = true;
            say("x2w-replay: heap type %u recorded increments differ (%u, %u)", type, p.recorded_increments[type], value);
        }
    }
    for (UINT type = 0; type < 4; ++type) if (p.recorded_increments[type]) {
        say("x2w-replay: heap type %u recorded increment %u", type, p.recorded_increments[type]);
        if (replay_title_increment) {
            replay_title_increment(type, p.recorded_increments[type]);
        } else {
            const u64 args[11] = {type};
            const UINT current = static_cast<UINT>(call_slot(reinterpret_cast<u64>(p.device), device_increment_size, args));
            if (increment_conflict[type] || p.recorded_increments[type] != current) {
                say("x2w-replay: heap type %u increment differs from this device (%u); requires x2w_d3d12_replay_title_increment",
                    type, current);
                return 4;
            }
        }
    }
    phase = "playing";
    const auto started = ReplayClock::now();
    std::optional<ReplayTiming> timing;
    if (time_slots) timing.emplace().next_report = started + std::chrono::seconds(1);
    const auto elapsed = [&] { return std::chrono::duration<double>(ReplayClock::now() - started).count(); };
    bool stopped = false;
    p.window_seq = r.window_seq;
    std::optional<PrefixLive> live;
    if (o.prefix_live) live = prefix_live_of(r, p.window_seq);
    p.pace = o.pace;
    bool in_prefix = true;
    const auto end_prefix = [&](u64 calls) {
        in_prefix = false;
        p.prefix_calls = calls;
        p.prefix_played = p.played;
        p.prefix_seconds = elapsed();
        say("x2w-replay: prefix: %llu of %llu calls played (%llu left out by --prefix-live), wall %.3f s",
            static_cast<unsigned long long>(p.prefix_played),
            static_cast<unsigned long long>(calls), static_cast<unsigned long long>(p.prefix_left_out), p.prefix_seconds);
    };
    u64 index = 0;
    Step s;  // the call being played, loaded from the mapping
    for (const StepRef& ref : r.steps) {
        r.load(ref, s);
        if (o.max_seconds > 0 && elapsed() >= o.max_seconds) {
            stopped = true;
            break;
        }
        if (in_prefix && s.call.seq >= p.window_seq) end_prefix(index);
        if (in_prefix && live && live->left_out(static_cast<std::size_t>(index))) {
            ++p.prefix_left_out;
            ++index;
            continue;
        }
        ++index;
        if (timing) {
            const auto before = ReplayClock::now();
            play(p, r, s);
            const auto after = ReplayClock::now();
            timing->add(s.call, after - before);
            timing->report(after, p);
        } else {
            play(p, r, s);
        }
    }
    current_step = nullptr;
    if (in_prefix && !stopped) end_prefix(index);  // a file whose window has no call after its first present
    bool dump_ok = true;
    if (!stopped && o.dump_bytes) {
        phase = "dumping title memory";
        if (p.fence_timeouts) {
            say("x2w-replay: cannot dump title memory after a fence timeout");
            dump_ok = false;
        } else {
            dump_ok = dump_title_range(o);
        }
    }
    phase = "writing the last pages";
    if (!stopped)  // the pages recorded after the last call
        apply_pages_before(r, ~0ull, [&p](u64 address, u64& bytes) { return p.page_target(address, bytes); });
    phase = "finishing";
    const double wall = elapsed();
    say("x2w-replay: played %llu of %zu calls (%llu on objects never created, %llu without their memory), %llu presents, "
        "%llu fence waits timed out, %llu mapped and %llu GPU address ranges translated, %.3f s",
        p.played, r.steps.size(), p.unknown_object, p.missing_blob, p.presents, p.fence_timeouts, p.mapped, p.gpu_addresses, wall);
    say_commit(p, "at the end");
    say("x2w-replay: frame waits not paced: %llu of %llu (%.3f s inside them)%s", static_cast<unsigned long long>(p.unpaced_waits),
        static_cast<unsigned long long>(p.event_waits), p.event_wait_seconds, p.pace ? ", --pace on" : "");
    say("x2w-replay: memory: peak private %llu MiB; %zu calls kept as %zu-byte references into the mapped file",
        static_cast<unsigned long long>(peak_private_bytes() >> 20), r.steps.size(), sizeof(StepRef));
    print_skip_census(p);
    write_summary(o, p, r, wall, stopped, timing ? &*timing : nullptr);
    if (stopped) return 5;
    if (!dump_ok) return 2;
    return p.slot_result_mismatches ? 6 : 0;
}

}  // namespace
}  // namespace x2w_replay

int wmain(int argc, wchar_t** argv)
{
    x2w_replay::Options options;
    if (!x2w_replay::parse(argc, argv, options)) {
        std::fprintf(stderr, "usage: x2w_replay <recording.x2wr> <out dir> [--adapter warp|hardware] [--mode tester|fixed] "
                             "[--threads N|recorded] [--from-present K] [--max-seconds S] [--dump HEX_ADDRESS:BYTES] [--prefix-live on] [--pace on]\n");
        return 2;
    }
    return x2w_replay::run(options);
}
