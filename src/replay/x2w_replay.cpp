// x2w_replay: plays a replay recording (docs/startup-runtime.md#d3d12-replay-record) into the D3D12 stand-in next to
// this program. No title code runs: the recorded calls are made again, in sequence order, on the stand-in's objects, so
// the stand-in, its translator and any fix run for real.
//
// usage: x2w_replay <recording.x2wr> <out dir> [--adapter warp|hardware] [--mode tester|fixed] [--threads N|recorded]
//                   [--from-present K] [--max-seconds S]
// Writes <out dir>\stderr.txt (the stand-in's log), present_<nnnn>.png (X2W_PRESENT_PICTURES), replay_summary.json, and
// one "x2w-replay score ..." line on stdout. Exit: 0 played to the end, 2 bad arguments or file, 3 a title range of the
// pages could not be reserved at its address, 4 the stand-in made no device, 5 stopped at --max-seconds, 6 a slot result differs,
// 77 no D3D12 adapter.
//
// Version 1 plays on one thread. Per call:
//   - the object and every argument naming a recorded wrapper get the replay's wrapper (wrappers are learned from the
//     output arguments: OUTP);
//   - argument memory is copied out of the recording into memory the replay owns, pointers between those copies are
//     rewritten (ARGM parent/word), and words inside them naming a wrapper or a descriptor handle are rewritten too;
//   - descriptor handles: a heap's CPU/GPU start returned on the recorded run maps to the one returned now, over the
//     heap's size (its descriptor count from the creation, times the increment the replay device reports);
//   - a fence's completed value is waited for until it reaches the recorded one (10 s at most).
//   - a buffer's mapped address and GPU address map to the replay's over the buffer's width;
//   - a struct-returning slot called as (return buffer, object) is played the same way, with a buffer of its own.
//   - title memory recorded as pages (PAGE/MAP) is reserved at its own addresses first and written before each call
//     (replay_pages_apply.h); argument memory inside it is passed as is.
// Not yet: float arguments passed by value, one thread per recorded thread.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "replay_file.h"
#include "replay_pages_apply.h"
#include "../startup_stubs/d3d12_replay_slots.h"

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace x2w_replay {
namespace {

// Kinds and slots as the stand-in numbers them (d3d12_xs_device.cpp enum Kind and its tables).
enum Kind : std::uint16_t { device_kind = 0, queue_kind = 1, fence_kind = 2, descriptor_heap_kind = 5, resource_kind = 7 };
constexpr std::uint16_t device_create_descriptor_heap = 14, device_increment_size = 15;
constexpr std::uint16_t heap_cpu_start = 9, heap_gpu_start = 10, fence_completed = 8, queue_present = 32;
constexpr std::uint16_t resource_map = 8, resource_get_desc = 10, resource_gpu_address = 11;
constexpr std::uint32_t rcx_arg = 15;  // ARGM argument of rcx when rcx is not an object (a struct-return buffer)
constexpr int top_slots = 16;

struct Options {
    std::wstring recording, out;
    bool hardware = false;
    double max_seconds = 0;
};

struct HeapInfo {
    std::uint32_t type, count;
};

struct Range {
    u64 to, span;
};

struct Player {
    std::unordered_map<u64, u64> objects;      // recorded wrapper -> replay wrapper
    std::unordered_map<u64, HeapInfo> heaps;   // recorded heap wrapper -> its creation descriptor
    std::map<u64, Range> handles;              // recorded handle / mapped / GPU address start -> replay start, span
    std::vector<std::unique_ptr<unsigned char[]>> arena;  // argument memory, kept: the stand-in may hold pointers
    u64 arena_bytes = 0, played = 0, unknown_object = 0, missing_blob = 0, presents = 0, fence_timeouts = 0;
    u64 mapped = 0, gpu_addresses = 0, title_memory_args = 0;
    mutable u64 page_translated = 0;  // pages written at a translated address (a heap's CPU memory, a Map pointer)
    bool use_slot_layouts = false;
    u64 frame_waits = 0, barrier_noops = 0, upload_ranges = 0, slot_result_mismatches = 0;
    void* device = nullptr;
    std::wstring map_dump_dir;

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
    u64 before[top_slots]{};  // the first word of each argument copy before the call
    u64 ret = 0;

    // The call wrote a new value into argument `arg`'s copy: what an output is. A stack word past the slot's own
    // arguments can point at the same variable as the real output; its copy is never written.
    bool wrote(std::uint32_t arg, u64& now) const
    {
        if (arg >= static_cast<std::uint32_t>(top_slots) || !top[arg]) return false;
        std::memcpy(&now, reinterpret_cast<const void*>(top[arg]), 8);
        return now != before[arg];
    }
};

using Slot = u64 (*)(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

// Calls `slot` of `object`'s table with `rcx` as the first register (the object itself, or a struct-return buffer
// when the title passed the buffer first and the object second).
u64 call_slot(u64 object, u64 rcx, std::uint16_t slot, const u64 (&a)[call_args])
{
    const auto table = *reinterpret_cast<void* const* const*>(object);
    return reinterpret_cast<Slot>(table[slot])(rcx, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]);
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

// The argument memory of one call, copied into the replay's arena. Fills the copy of each top-level argument
// (top[0..10]: the register and stack arguments after rcx; top[15]: rcx when it is not an object).
void relocate(Player& p, const Recording& r, const Step& step, u64 (&top)[top_slots], std::vector<unsigned char*>& copies)
{
    copies.assign(step.memory.size(), nullptr);
    for (std::size_t i = 0; i < step.memory.size(); ++i) {
        const ArgMemory& m = step.memory[i];
        // Title memory the pages replay (reserved at its own address): passed as is, never copied.
        if (page_replay().reserved(m.address)) {
            copies[i] = reinterpret_cast<unsigned char*>(m.address);
            ++p.title_memory_args;
            continue;
        }
        const auto blob = r.blobs.find(m.hash);
        if (blob == r.blobs.end() || blob->second.bytes != m.bytes) {
            ++p.missing_blob;
            continue;
        }
        auto copy = std::make_unique<unsigned char[]>(static_cast<std::size_t>(m.bytes) + 8);
        std::memcpy(copy.get(), blob->second.data, static_cast<std::size_t>(m.bytes));
        // Words naming a recorded wrapper, descriptor handle, mapped or GPU address name the replay's.
        for (u64 at = 0; at + 8 <= m.bytes; at += 8) {
            u64 word;
            std::memcpy(&word, copy.get() + at, 8);
            const u64 now = p.translate(word);
            if (now != word) std::memcpy(copy.get() + at, &now, 8);
        }
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
            const u64 increment = call_slot(reinterpret_cast<u64>(p.device), device_increment_size, a) & 0xffffffffu;
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

// Plays one call. false: its object was never seen (skipped).
bool play(Player& p, const Recording& r, const Step& step)
{
    current_step = &step;
    const CallRecord& c = step.call;
    Played call{&step};
    if (const auto object = p.objects.find(c.object); object != p.objects.end()) {
        call.self = object->second;
    } else if (const auto second = p.objects.find(c.args[0]); second != p.objects.end()) {
        call.self = second->second;  // (return buffer, object): the struct-return form the stand-in accepts too
        call.buffer_first = true;
    } else {
        ++p.unknown_object;
        return false;
    }
    apply_pages_before(r, c.seq, [&p](u64 address, u64& bytes) { return p.page_target(address, bytes); });
    std::vector<unsigned char*> copies;
    relocate(p, r, step, call.top, copies);
    for (int i = 0; i < top_slots; ++i)
        if (call.top[i] && i != static_cast<int>(rcx_arg)) std::memcpy(&call.before[i], reinterpret_cast<const void*>(call.top[i]), 8);
    u64 a[call_args];
    for (int i = 0; i < call_args; ++i) a[i] = call.top[i] ? call.top[i] : p.translate(c.args[i]);
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
        call_slot(call.self, rcx, c.slot, a);
        ++p.played;
        return true;
    }
    const auto* layout = p.use_slot_layouts ? x2w_replay_slots::find(c.kind, c.slot) : nullptr;
    const bool wait = layout && c.kind == device_kind && c.slot == 80;
    const bool barrier_noop = layout && c.kind == x2w_replay_slots::list && c.slot == 146;
    constexpr int token_arg = 4;
    const u64 expected_token = wait ? recorded_word(r, step, token_arg) : 0;
    u64 token = ~expected_token;
    // The recording holds the post-call token. Start with a different value to prove that replay writes it.
    if (wait && c.args[token_arg]) a[token_arg] = reinterpret_cast<u64>(&token);
    call.ret = call_slot(call.self, rcx, c.slot, a);
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
        else if (name == L"--mode" || name == L"--threads" || name == L"--from-present") say("x2w-replay: %ls %ls is not in version 1 (ignored)", name.c_str(), value.c_str());
        else return false;
    }
    return true;
}

void write_summary(const Options& o, const Player& p, const Recording& r, double wall, bool stopped)
{
    const double fps = wall > 0 ? static_cast<double>(p.presents) / wall : 0;
    const std::wstring path = o.out + L"\\replay_summary.json";
    if (std::FILE* f = _wfopen(path.c_str(), L"w")) {
        std::fprintf(f, "{\"version\": %u, \"calls\": %zu, \"played\": %llu, \"unknown_object\": %llu, \"missing_blob\": %llu, "
                        "\"fence_timeouts\": %llu, \"presents\": %llu, \"wall_s\": %.3f, \"fps\": %.2f, \"argument_bytes\": %llu, "
                        "\"stopped_at_max_seconds\": %s, ",
                     r.version, r.steps.size(), p.played, p.unknown_object, p.missing_blob, p.fence_timeouts, p.presents, wall, fps,
                     p.arena_bytes, stopped ? "true" : "false");
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
        if (name == "X2W_SHADER_CACHE" || name == "X2W_PRESENT_PICTURES" || ends("_DIR") || ends("_FILE") || ends("_PATH") ||
            ends("_LOG"))
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

int run(const Options& o)
{
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
    // First, before the stand-in or anything else can map there: the title ranges the pages replay.
    if (std::string problem; !reserve_page_ranges(r, problem)) {
        say("x2w-replay: %s", problem.c_str());
        std::printf("x2w-replay: %s\n", problem.c_str());
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
    for (const Step& s : r.steps)
        if (s.call.kind == device_kind) {
            p.objects[s.call.object] = reinterpret_cast<u64>(p.device);
            break;
        }
    phase = "playing";
    const auto started = std::chrono::steady_clock::now();
    const auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };
    bool stopped = false;
    for (const Step& s : r.steps) {
        if (o.max_seconds > 0 && elapsed() >= o.max_seconds) {
            stopped = true;
            break;
        }
        play(p, r, s);
    }
    current_step = nullptr;
    phase = "writing the last pages";
    if (!stopped)  // the pages recorded after the last call
        apply_pages_before(r, ~0ull, [&p](u64 address, u64& bytes) { return p.page_target(address, bytes); });
    phase = "finishing";
    const double wall = elapsed();
    say("x2w-replay: played %llu of %zu calls (%llu on objects never created, %llu without their memory), %llu presents, "
        "%llu fence waits timed out, %llu mapped and %llu GPU address ranges translated, %.3f s",
        p.played, r.steps.size(), p.unknown_object, p.missing_blob, p.presents, p.fence_timeouts, p.mapped, p.gpu_addresses, wall);
    write_summary(o, p, r, wall, stopped);
    if (stopped) return 5;
    return p.slot_result_mismatches ? 6 : 0;
}

}  // namespace
}  // namespace x2w_replay

int wmain(int argc, wchar_t** argv)
{
    x2w_replay::Options options;
    if (!x2w_replay::parse(argc, argv, options)) {
        std::fprintf(stderr, "usage: x2w_replay <recording.x2wr> <out dir> [--adapter warp|hardware] [--mode tester|fixed] "
                             "[--threads N|recorded] [--from-present K] [--max-seconds S]\n");
        return 2;
    }
    return x2w_replay::run(options);
}
