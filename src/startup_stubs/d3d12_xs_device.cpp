// Experimental (docs/startup-runtime.md).
//
// With X2W_D3D12_PERMISSIVE=1, D3D12XboxCreateDevice accepts a non-null Xbox parameter block and
// returns a recording wrapper around a desktop ID3D12Device. Every object the wrappers hand out is
// wrapped too. Every vtable slot traps ("x2w-trap d3d12_xs.dll!<Object>::slot<N>", then terminates)
// unless it is enabled in build_tables(). Enabled slots log "x2w-d3d12" lines (the first calls only).
//
// Evidence so far: the title requests desktop IIDs (ID3D12Device11, ID3D12CommandQueue, ID3D12Fence,
// ID3D12CommandAllocator), and device slots 8, 9, 14, 26, 28 and 36 have desktop call shapes. Slot 48
// does not (Xbox queue creation), so slots from 48 up are only enabled one by one from call sites.
// Methods that return a structure by value are called with the hidden result pointer in rcx and the
// object in rdx (desktop: object first); those slots accept either order.
#define X2W_TRAP_MODULE "d3d12_xs.dll"
#define X2W_TRAP_MODULE_ID 7
#include "startup_stubs/trap.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/logical_view_table.h"
#include "startup_stubs/overlay_window.h"
#include "startup_stubs/draw_bmp.h"
#include "startup_stubs/draw_dump_tracking.h"
#include "startup_stubs/translated_sampler_bindings.h"
#include "startup_stubs/heap_indexed_root.h"
#include "startup_stubs/bc7_4k_s.h"
#include "startup_stubs/bc64_chain.h"
#include "startup_stubs/derived_blend.h"
#include "startup_stubs/placement_publish.h"
#include "startup_stubs/placed_build.h"
#include "startup_stubs/memory_srv_report.h"
#include "startup_stubs/memory_srv_buffer.h"

#include "startup_stubs/slice_footprints.h"

#include <d3d12.h>
#include "command_list_slot_names.h"
#include "console_plane_size.h"
#include "list_clear_depth.h"
#include "null_binding_cause.h"
#include "null_texture_view.h"
#include "packed_target_format.h"
#include "present_view_format.h"
#include "state_block_blend.h"
#include "title_heap_memory.h"
#include <d3d12sdklayers.h>
#include <d3d12shader.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using u64 = std::uint64_t;
constexpr int slot_count = 512;

enum Kind : int {
    device_kind, queue_kind, fence_kind, allocator_kind, list_kind, descriptor_heap_kind, root_signature_kind,
    resource_kind, heap_kind, query_heap_kind, pipeline_kind, command_signature_kind, other_kind, kind_count
};
const char* const kind_names[kind_count] = {
    "Device", "CommandQueue", "Fence", "CommandAllocator", "CommandList", "DescriptorHeap", "RootSignature",
    "Resource", "Heap", "QueryHeap", "PipelineState", "CommandSignature", "Object"};

// The title reads console object fields directly. Seen on a command list: +0x18 is the command
// buffer cursor and +0x30 its limit; inlined calls write GPU packets at the cursor while it is below
// the limit and otherwise call the virtual method (IASetPrimitiveTopology: slot 20, +0xa0). Both
// stay 0 here, so every inlined call takes its virtual path. The first 0x400 bytes after the vtable
// pointer are a zero-filled stand-in for such fields; the wrapper's own state follows.
//
// The title also writes into those fields (seen: a command list at +0x544), so the wrapper's own
// state lives in front of the object, out of the title's reach, and the zero-filled area behind the
// vtable pointer is sized per kind (64 KiB for command lists).
struct Wrapper;
enum class IndirectCommand { none, dispatch, draw, draw_indexed, dispatch_mesh };
struct IndirectGraphicsPipeline {
    ID3D12RootSignature* root{};
    DXGI_FORMAT target_format{};
    bool mesh{};
};

struct alignas(16) WrapperOwn {
    IUnknown* real;
    LONG refs;
    Kind kind;
    void* title_memory;       // Resource: the title-side memory (placement address or hook allocation), or null
    std::uint64_t title_cpu;  // DescriptorHeap: title-owned descriptor memory, or 0
    // Fence: the completed value the title polls through the address from slot 12 (monitorx/mwaitx).
    // Updated when the fence is signaled; GPU work counts as complete at once while lists are no-ops.
    volatile LONG64 completed;
    bool list_open;   // CommandList (render mode): recording between reset and close
    bool list_close_failed; // Desktop Reset repeats a failed Close result indefinitely.
    bool tiled_address_srv; // Resource backed by 4 KiB tiled guest memory for translated draws.
    bool list_ready;  // CommandList (render mode): closed with recorded work, not yet submitted
    bool compute_pipeline;   // PipelineState: a desktop compute pipeline
    bool root_without_bindings; // RootSignature: decoded and has no parameters or static samplers
    volatile LONG queue_wait_seen; // CommandQueue: publish later signals only after GPU completion.
    void* pass;       // CommandList (render mode): recording state (Pass)
    Wrapper* list_device;    // CommandList: wrapper that created this list.
    Wrapper* signature_device; // CommandSignature: owner of the supported single-command shape
    IndirectCommand indirect_command;
    IndirectGraphicsPipeline indirect_graphics;
    UINT list_node_mask;
};
static_assert(sizeof(WrapperOwn) % 16 == 0, "the title-visible object must stay 16-byte aligned");
struct Wrapper {
    void** vtable;
    std::uint64_t console_fields[0x7f];  // +0x08..+0x400 (more for command lists), zero until the title writes
    WrapperOwn& own() const { return const_cast<WrapperOwn*>(reinterpret_cast<const WrapperOwn*>(this))[-1]; }
};

void fence_reached(Wrapper* fence, std::uint64_t value) noexcept
{
    LONG64 seen = fence->own().completed;
    while (static_cast<std::uint64_t>(seen) < value || seen == -1) {
        const LONG64 previous = InterlockedCompareExchange64(&fence->own().completed, static_cast<LONG64>(value), seen);
        if (previous == seen) break;
        seen = previous;
    }
}

// Descriptor heaps created over title memory: title CPU handles in [title, title + bytes) map to
// desktop CPU handles at the same offset (the title's increments come from the forwarded desktop
// GetDescriptorHandleIncrementSize, so offsets agree). Descriptors the title writes or copies into
// that memory itself are not seen by the desktop heap.
struct DescriptorRange {
    std::uint64_t title;
    std::uint64_t bytes;
    std::uint64_t desktop;
    UINT title_stride;
    UINT desktop_stride;
    // How many title descriptors the desktop heap covers. Normally the whole heap; shorter only when
    // the console heap was larger than the driver accepts and the stand-in emulated the remainder.
    std::uint64_t desktop_descriptors;
    std::uint64_t tail_cpu; // CPU-only heap for view records outside the shader-visible prefix.
};
SRWLOCK descriptor_lock = SRWLOCK_INIT;
std::vector<DescriptorRange> descriptor_ranges;

// Desktop rejects a shader-visible CBV/SRV/UAV heap above a driver-specific descriptor count: the
// API documents 1,048,576, but a driver may accept fewer (1,000,000 on the GTX 1660 SUPER here), so
// the accepted maximum is measured once and cached rather than assumed.
constexpr std::uint64_t view_heap_probe_floor = 65536;

void log(const char* format, ...);
bool first_calls(LONG& counter, LONG limit);
std::uint64_t accepted_view_heap_descriptors(ID3D12Device* device, UINT node_mask);

std::uint64_t desktop_cpu_handle(std::uint64_t handle, UINT* contiguous = nullptr, UINT* available = nullptr)
{
    std::uint64_t result = handle;
    UINT room = UINT_MAX, remaining = UINT_MAX;
    bool past_end = false;
    AcquireSRWLockShared(&descriptor_lock);
    for (const auto& r : descriptor_ranges) {
        if (handle < r.title) continue;
        const u64 offset = handle - r.title;
        if (offset == r.bytes && r.tail_cpu) past_end = true;
        if (offset >= r.bytes) continue;
        const u64 index = offset / r.title_stride;
        const u64 limit = r.bytes / r.title_stride;
        remaining = static_cast<UINT>(limit - index);
        if (r.tail_cpu && offset % r.title_stride) {
            result = 0;
            room = 0;
        } else if (index >= r.desktop_descriptors) {
            result = r.tail_cpu + (index - r.desktop_descriptors) * r.desktop_stride;
            room = static_cast<UINT>(limit - index);
        } else {
            result = r.desktop + index * r.desktop_stride + offset % r.title_stride;
            room = static_cast<UINT>(r.desktop_descriptors - index);
        }
        past_end = false;
        break;
    }
    ReleaseSRWLockShared(&descriptor_lock);
    if (past_end) { result = 0; room = 0; }
    if (contiguous) *contiguous = room;
    if (available) *available = result ? remaining : 0;
    return result;
}

// Measure, once per process, the largest shader-visible CBV/SRV/UAV heap this driver creates. The
// documented ceiling is 1,048,576 descriptors, but drivers may accept fewer, so halve from the
// ceiling until a create succeeds and then bisect upward. Each probe heap is released before the next
// one, so at most one probe holds memory at a time.
std::uint64_t accepted_view_heap_descriptors(ID3D12Device* device, UINT node_mask)
{
    static std::uint64_t cached;
    static LONG state;  // 0 none, 1 measuring, 2 measured
    if (InterlockedCompareExchange(&state, 0, 0) == 2) return cached;
    if (InterlockedCompareExchange(&state, 1, 0) != 0) {
        // Another thread is measuring; wait for it rather than create a second set of probe heaps.
        for (int spin = 0; spin < 2000 && InterlockedCompareExchange(&state, 0, 0) != 2; ++spin) Sleep(1);
        return InterlockedCompareExchange(&state, 0, 0) == 2 ? cached : 0;
    }

    const auto probe = [device, node_mask](std::uint64_t count) {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = static_cast<UINT>(count);
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        desc.NodeMask = node_mask;
        ID3D12DescriptorHeap* heap = nullptr;
        const bool ok = SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap))) && heap;
        if (heap) heap->Release();
        return ok;
    };

    // Halve from the documented ceiling, testing the floor too: a driver whose maximum is below the
    // floor must not be recorded as accepting it, or the retry would ask for a size it rejects again.
    std::uint64_t accepted = 0;
    for (std::uint64_t size = 1048576; size >= view_heap_probe_floor; size /= 2) {
        if (probe(size)) {
            accepted = size;
            break;
        }
    }
    if (!accepted) {
        InterlockedExchange(&state, 2);
        log("x2w-d3d12 no probe size at or above %llu was accepted; the view heap shortcut is declined",
            view_heap_probe_floor);
        return 0;
    }
    std::uint64_t low = accepted, high = accepted * 2;
    while (low + 1 < high && high <= 1048576) {
        const std::uint64_t mid = low + (high - low) / 2;
        if (probe(mid)) low = mid; else high = mid;
    }
    cached = low;
    InterlockedExchange(&state, 2);
    log("x2w-d3d12 measured the shader-visible view heap limit: %llu descriptors", cached);
    return cached;
}

UINT title_descriptor_stride(u64 handle, UINT desktop_stride)
{
    AcquireSRWLockShared(&descriptor_lock);
    UINT result = desktop_stride;
    for (const auto& range : descriptor_ranges)
        if (handle >= range.title && handle - range.title < range.bytes) result = range.title_stride;
    ReleaseSRWLockShared(&descriptor_lock);
    return result;
}

void* tables[kind_count][slot_count];
Wrapper* the_device;

// Time spent inside the stand-in per entry-point family, reported with the heartbeat.
enum { time_list, time_device_noop, time_forward, time_create, time_view, time_queue, time_present, time_wait,
       time_pipeline, time_heap, time_copy, time_count };
const char* const time_names[time_count] = {"list", "device-noop", "forward", "create", "view", "queue", "present", "wait",
                                            "pipeline", "heap", "copy"};
volatile LONG64 time_ticks[time_count], time_calls[time_count];
volatile LONG64 list_slot_ticks[256], list_slot_calls[256];  // per command-list slot
volatile LONG64 list_slot_census[256];  // per command-list slot, reported with the render totals
struct Timed {
    int category;
    LARGE_INTEGER start;
    explicit Timed(int c) : category(c) { QueryPerformanceCounter(&start); }
    ~Timed()
    {
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        InterlockedAdd64(&time_ticks[category], end.QuadPart - start.QuadPart);
        InterlockedIncrement64(&time_calls[category]);
    }
};

void log(const char* format, ...)
{
    char line[1536];
    va_list args;
    va_start(args, format);
    int n = std::vsnprintf(line, sizeof(line) - 2, format, args);
    va_end(args);
    if (n < 0) return;
    if (n > static_cast<int>(sizeof(line)) - 3) n = sizeof(line) - 3;
    line[n++] = '\r';
    line[n++] = '\n';
    x2w_trap::write_all(GetStdHandle(STD_ERROR_HANDLE), line, static_cast<DWORD>(n));
    wchar_t log_path[MAX_PATH];
    const DWORD length = GetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path, MAX_PATH);
    if (length && length < MAX_PATH) {
        HANDLE out = CreateFileW(log_path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                 OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (out != INVALID_HANDLE_VALUE) {
            x2w_trap::write_all(out, line, static_cast<DWORD>(n));
            CloseHandle(out);
        }
    }
}

void guid_text(const GUID* g, char* out, std::size_t size)
{
    if (!g || !x2w_local::readable(g, sizeof(GUID))) {
        std::snprintf(out, size, "%p", static_cast<const void*>(g));
        return;
    }
    std::snprintf(out, size, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g->Data1, g->Data2, g->Data3,
                  g->Data4[0], g->Data4[1], g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6],
                  g->Data4[7]);
}

void hex_text(const void* data, std::size_t bytes, char* out, std::size_t size)
{
    out[0] = 0;
    if (!data || !x2w_local::readable(data, bytes)) return;
    const auto* p = static_cast<const unsigned char*>(data);
    std::size_t used = 0;
    for (std::size_t i = 0; i < bytes && used + 4 < size; ++i) used += std::snprintf(out + used, size - used, "%02x ", p[i]);
}

bool first_calls(LONG& counter, LONG limit = 3) { return InterlockedIncrement(&counter) <= limit; }

// X2W_D3D12_DEBUG=1: the desktop debug layer's messages are drained into the log after forwarded
// calls, tagged with the call that preceded them, and a device removal is reported once with its reason.
ID3D12InfoQueue* info_queue;
ID3D12Device* removal_device;
LONG removal_reported, debug_lines;
SRWLOCK debug_lock = SRWLOCK_INIT;

void check_device(const char* context, int slot)
{
    if (removal_device && !removal_reported) {
        const HRESULT reason = removal_device->GetDeviceRemovedReason();
        if (reason != S_OK && InterlockedExchange(&removal_reported, 1) == 0)
            log("x2w-d3d12 DEVICE REMOVED, reason 0x%08lx, first noticed after %s %d", static_cast<unsigned long>(reason),
                context, slot);
    }
    if (!info_queue) return;
    AcquireSRWLockExclusive(&debug_lock);
    const UINT64 count = info_queue->GetNumStoredMessages();
    for (UINT64 i = 0; i < count && debug_lines < 400; ++i) {
        SIZE_T size = 0;
        info_queue->GetMessage(i, nullptr, &size);
        if (!size || size > 8192) continue;
        alignas(16) char storage[8192];
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage);
        if (SUCCEEDED(info_queue->GetMessage(i, message, &size))) {
            ++debug_lines;
            log("x2w-d3d12-debug [%s %d] sev=%d id=%d %.900s", context, slot, message->Severity, message->ID,
                message->pDescription);
        }
    }
    info_queue->ClearStoredMessages();
    ReleaseSRWLockExclusive(&debug_lock);
}

// Wrapper detection is a registry lookup: no argument is ever dereferenced to find out what it is.
SRWLOCK registry_lock = SRWLOCK_INIT;
std::unordered_set<const void*>& registry()
{
    static std::unordered_set<const void*> wrappers;
    return wrappers;
}

Wrapper* as_wrapper(const void* p)
{
    const auto value = reinterpret_cast<std::uintptr_t>(p);
    if (value < 0x10000 || (value & 7)) return nullptr;
    AcquireSRWLockShared(&registry_lock);
    const bool known = registry().count(p) != 0;
    ReleaseSRWLockShared(&registry_lock);
    return known ? const_cast<Wrapper*>(static_cast<const Wrapper*>(p)) : nullptr;
}

u64 unwrap(u64 value)
{
    Wrapper* w = as_wrapper(reinterpret_cast<const void*>(value));
    return w ? reinterpret_cast<u64>(w->own().real) : value;
}

Wrapper* wrap(IUnknown* real, Kind kind)
{
    const std::size_t area = kind == list_kind ? 0x10000 : sizeof(Wrapper);
    auto* block = static_cast<unsigned char*>(_aligned_malloc(sizeof(WrapperOwn) + area, 16));
    std::memset(block, 0, sizeof(WrapperOwn) + area);
    auto* w = reinterpret_cast<Wrapper*>(block + sizeof(WrapperOwn));
    w->vtable = tables[kind];
    w->own().real = real;
    w->own().refs = 1;
    w->own().kind = kind;
    if (kind == fence_kind && real) w->own().completed = static_cast<LONG64>(static_cast<ID3D12Fence*>(real)->GetCompletedValue());
    AcquireSRWLockExclusive(&registry_lock);
    registry().insert(w);
    ReleaseSRWLockExclusive(&registry_lock);
    return w;
}

Kind kind_for(REFIID iid)
{
    if (iid == __uuidof(ID3D12Device) || iid == __uuidof(ID3D12Device1) || iid == __uuidof(ID3D12Device2) ||
        iid == __uuidof(ID3D12Device3) || iid == __uuidof(ID3D12Device4) || iid == __uuidof(ID3D12Device5) ||
        iid == __uuidof(ID3D12Device6) || iid == __uuidof(ID3D12Device7) || iid == __uuidof(ID3D12Device8) ||
        iid == __uuidof(ID3D12Device9) || iid == __uuidof(ID3D12Device10))
        return device_kind;
    // ID3D12Device11 arrived after the 10.0.22621 SDK headers; match its IID literally.
    static const GUID device11 = {0x5405c344, 0xd457, 0x444e, {0xb4, 0xdd, 0x23, 0x66, 0xe4, 0x5a, 0xee, 0x39}};
    if (iid == device11) return device_kind;
    if (iid == __uuidof(ID3D12CommandQueue)) return queue_kind;
    if (iid == __uuidof(ID3D12Fence) || iid == __uuidof(ID3D12Fence1)) return fence_kind;
    if (iid == __uuidof(ID3D12CommandAllocator)) return allocator_kind;
    if (iid == __uuidof(ID3D12CommandList) || iid == __uuidof(ID3D12GraphicsCommandList) ||
        iid == __uuidof(ID3D12GraphicsCommandList1) || iid == __uuidof(ID3D12GraphicsCommandList2) ||
        iid == __uuidof(ID3D12GraphicsCommandList3) || iid == __uuidof(ID3D12GraphicsCommandList4) ||
        iid == __uuidof(ID3D12GraphicsCommandList5) || iid == __uuidof(ID3D12GraphicsCommandList6) ||
        iid == __uuidof(ID3D12GraphicsCommandList7))
        return list_kind;
    // Xbox-only command list interface requested from device slot 73 (not in the desktop SDK header).
    static const GUID xbox_list = {0xc3827890, 0x4024, 0x474c, {0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74}};
    if (iid == xbox_list) return list_kind;
    if (iid == __uuidof(ID3D12DescriptorHeap)) return descriptor_heap_kind;
    if (iid == __uuidof(ID3D12RootSignature)) return root_signature_kind;
    if (iid == __uuidof(ID3D12Resource) || iid == __uuidof(ID3D12Resource1) || iid == __uuidof(ID3D12Resource2))
        return resource_kind;
    if (iid == __uuidof(ID3D12Heap) || iid == __uuidof(ID3D12Heap1)) return heap_kind;
    if (iid == __uuidof(ID3D12QueryHeap)) return query_heap_kind;
    if (iid == __uuidof(ID3D12PipelineState)) return pipeline_kind;
    if (iid == __uuidof(ID3D12CommandSignature)) return command_signature_kind;
    return other_kind;
}

bool is_base_iid(REFIID iid)
{
    return iid == __uuidof(IUnknown) || iid == __uuidof(ID3D12Object) || iid == __uuidof(ID3D12DeviceChild) ||
           iid == __uuidof(ID3D12Pageable);
}

// Traps. The kind is a template argument so a trap never dereferences its arguments.
template <int K, int N>
__declspec(noinline) u64 trap_slot(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    char name[64];
    std::snprintf(name, sizeof(name), "%s::slot%d", kind_names[K], N);
    x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, (0x80 | N) & 0xFF, rcx, rdx, r8, r9, _ReturnAddress());
}

template <int K, std::size_t... I>
void fill_traps(void** table, std::index_sequence<I...>)
{
    ((table[I] = reinterpret_cast<void*>(&trap_slot<K, static_cast<int>(I)>)), ...);
}

template <std::size_t... K>
void fill_all_traps(std::index_sequence<K...>)
{
    (fill_traps<static_cast<int>(K)>(tables[K], std::make_index_sequence<slot_count>{}), ...);
}

// A call that passes a known interface IID is a creation or query call: a no-op that returned 0
// would hand the title "success" with no object. Such calls trap instead of being ignored.
bool names_known_iid(u64 value)
{
    if (value < 0x10000 || !x2w_local::readable(reinterpret_cast<const void*>(value), sizeof(GUID))) return false;
    return kind_for(*reinterpret_cast<const GUID*>(value)) != other_kind;
}
bool create_shaped(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g)
{
    return names_known_iid(a) || names_known_iid(b) || names_known_iid(c) || names_known_iid(d) ||
           names_known_iid(e) || names_known_iid(f) || names_known_iid(g);
}

bool list_noop()
{
    char value[4]{};
    return GetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", value, sizeof(value)) && value[0] == '1';
}

// With X2W_D3D12_LIST_NOOP=1: an unknown command-list method records nothing and returns 0.
extern LONG draw_calls_seen;
// X2W_D3D12_TRACE_FRAME=<n>: every command-list call of presented frame n is logged ("x2w-frame")
// with its slot, the kinds of any wrapped objects among its arguments, and stack arguments.
LONG trace_frame_active;
extern LONG capture_active;
// Command-list methods with a real implementation (render mode). Every list call enters through
// noop_slot, which times, captures and traces it, then hands over to the entry here if there is one.
using ListTyped = u64(__stdcall*)(u64, u64, u64, u64, u64, u64, u64, u64);
ListTyped list_typed[slot_count];
namespace capture { void list_call(int slot, u64 list, u64 rdx, u64 r8, u64 r9, const u64* stack, u64 ret); }
const char* arg_kind(u64 value)
{
    Wrapper* w = as_wrapper(reinterpret_cast<const void*>(value));
    return w ? kind_names[w->own().kind] : "";
}
template <int K, int N>
__declspec(noinline) u64 noop_slot(u64 rcx, u64 rdx, u64 r8, u64 r9, u64 s0, u64 s1, u64 s2, u64 s3)
{
    Timed timed(K == list_kind ? time_list : K == queue_kind ? time_queue : time_device_noop);
    struct SlotTimed {
        LARGE_INTEGER start;
        SlotTimed() { QueryPerformanceCounter(&start); }
        ~SlotTimed()
        {
            LARGE_INTEGER end;
            QueryPerformanceCounter(&end);
            if (K == list_kind) {
                InterlockedAdd64(&list_slot_ticks[N & 255], end.QuadPart - start.QuadPart);
                InterlockedIncrement64(&list_slot_calls[N & 255]);
                InterlockedIncrement64(&list_slot_census[N & 255]);
            }
        }
    } slot_timed;
    if (K == list_kind) InterlockedIncrement(&draw_calls_seen);
    if (K == list_kind && capture_active) {
        const u64 stack[4] = {s0, s1, s2, s3};
        capture::list_call(N, rcx, rdx, r8, r9, stack,
                           reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr)));
    }
    if (K == list_kind && trace_frame_active) {
        const auto ret = reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr));
        log("x2w-frame tid=%lu list=%llx slot%d(0x%llx%s%s, 0x%llx%s%s, 0x%llx%s%s | 0x%llx, 0x%llx, 0x%llx, 0x%llx) ret=+0x%llx",
            GetCurrentThreadId(), rcx, N, rdx, *arg_kind(rdx) ? ":" : "", arg_kind(rdx), r8, *arg_kind(r8) ? ":" : "",
            arg_kind(r8), r9, *arg_kind(r9) ? ":" : "", arg_kind(r9), s0, s1, s2, s3, ret);
    }
    if (K == list_kind && list_typed[N]) return list_typed[N](rcx, rdx, r8, r9, s0, s1, s2, s3);
    // Register arguments only: stale caller stack words often still point at an IID. The probe
    // costs over a millisecond per call in this process (memory queries), so only the first calls
    // of each slot are checked.
    static LONG shape_checks;
    if (shape_checks < 32 && InterlockedIncrement(&shape_checks) <= 32 && create_shaped(rdx, r8, r9, 0, 0, 0, 0)) {
        char name[64];
        std::snprintf(name, sizeof(name), "%s::slot%d(passes an IID)", kind_names[K], N);
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, rcx, rdx, r8, r9, _ReturnAddress());
    }
    static LONG calls;
    if (first_calls(calls, 1)) {
        char name[64];
        const char* known = K == list_kind ? x2w_list_slots::name(N) : nullptr;
        if (known) std::snprintf(name, sizeof(name), "%s::slot%d(%s)", kind_names[K], N, known);
        else std::snprintf(name, sizeof(name), "%s::slot%d", kind_names[K], N);
        x2w_trap::record("x2w-noop", X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, rcx, rdx, r8, r9,
                         _ReturnAddress());
    }
    return 0;
}

bool device_noop()
{
    char value[4]{};
    return GetEnvironmentVariableA("X2W_D3D12_DEVICE_NOOP", value, sizeof(value)) && value[0] == '1';
}

// With X2W_D3D12_DEVICE_NOOP=1: an unimplemented Xbox-order device method (slot 44 and up) logs its
// first call with the first 48 bytes behind rdx and r8 and returns 0.
template <int N>
__declspec(noinline) u64 device_noop_slot(u64 rcx, u64 rdx, u64 r8, u64 r9, u64 s0, u64 s1, u64 s2, u64 s3)
{
    Timed timed(time_device_noop);
    static LONG shape_checks;
    if (shape_checks < 32 && InterlockedIncrement(&shape_checks) <= 32 && create_shaped(rdx, r8, r9, s0, s1, s2, s3)) {
        char name[64];
        std::snprintf(name, sizeof(name), "Device::slot%d(passes an IID)", N);
        char first[3 * 64 + 1];
        hex_text(reinterpret_cast<const void*>(rdx), 64, first, sizeof(first));
        log("x2w-d3d12 Device::slot%d passes an IID: rdx=0x%llx r8=0x%llx r9=0x%llx stack=0x%llx,0x%llx,0x%llx,0x%llx [rdx]=[%s]",
            N, rdx, r8, r9, s0, s1, s2, s3, first);
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, rcx, rdx, r8, r9, _ReturnAddress());
    }
    static LONG calls;
    if (first_calls(calls, 1)) {
        char name[64];
        std::snprintf(name, sizeof(name), "Device::slot%d", N);
        x2w_trap::record("x2w-noop", X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, rcx, rdx, r8, r9,
                         _ReturnAddress());
        char first[3 * 48 + 1], second[3 * 48 + 1];
        hex_text(reinterpret_cast<const void*>(rdx), 48, first, sizeof(first));
        hex_text(reinterpret_cast<const void*>(r8), 48, second, sizeof(second));
        log("x2w-noop   [rdx]=[%s] [r8]=[%s]", first, second);
    }
    return 0;
}

template <std::size_t... I>
void fill_device_noops(void** table, std::index_sequence<I...>)
{
    ((I >= 44 && table[I] == reinterpret_cast<void*>(&trap_slot<device_kind, static_cast<int>(I)>)
          ? (table[I] = reinterpret_cast<void*>(&device_noop_slot<static_cast<int>(I)>)) : nullptr), ...);
}

template <int K, std::size_t... I>
void fill_noops(void** table, std::index_sequence<I...>)
{
    ((I >= 8 ? (table[I] = reinterpret_cast<void*>(&noop_slot<K, static_cast<int>(I)>)) : nullptr), ...);
}

// With X2W_D3D12_DEVICE_NOOP=1, the same for the slots of the other object kinds that are still traps.
template <int K, std::size_t... I>
void fill_remaining_noops(void** table, std::index_sequence<I...>)
{
    ((I >= 8 && table[I] == reinterpret_cast<void*>(&trap_slot<K, static_cast<int>(I)>)
          ? (table[I] = reinterpret_cast<void*>(&noop_slot<K, static_cast<int>(I)>)) : nullptr), ...);
}
template <std::size_t... K>
void fill_all_remaining_noops(std::index_sequence<K...>)
{
    ((static_cast<int>(K) != device_kind && static_cast<int>(K) != list_kind
          ? fill_remaining_noops<static_cast<int>(K)>(tables[K], std::make_index_sequence<slot_count>{}) : void()), ...);
}

[[noreturn]] void fail(const char* what, const void* a, const void* b)
{
    x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, what, 0xFE, reinterpret_cast<u64>(a), reinterpret_cast<u64>(b),
                   0, 0, _ReturnAddress());
}

using Method = u64(__stdcall*)(IUnknown*, u64, u64, u64, u64, u64, u64, u64, u64);
Method method_of(IUnknown* real, int slot) { return reinterpret_cast<Method>((*reinterpret_cast<void***>(real))[slot]); }

// Desktop-order forwarder for a slot whose arguments are integers or pointers (no floating-point
// argument in the first four positions, at most eight arguments). Wrapped objects among the arguments
// are replaced by their desktop objects; unused argument words are caller stack the callee ignores.
template <int K, int N>
__declspec(noinline) u64 forward_slot(Wrapper* self, u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g, u64 h)
{
    Timed timed(time_forward);
    const u64 result = method_of(self->own().real, N)(self->own().real, unwrap(a), unwrap(b), unwrap(c), unwrap(d), unwrap(e),
                                                unwrap(f), unwrap(g), unwrap(h));
    static LONG calls;
    if (first_calls(calls))
        log("x2w-d3d12 %s::slot%d(0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx) -> 0x%llx", kind_names[K], N, a, b, c, d, e,
            result);
    check_device(kind_names[K], N);
    return result;
}

// Desktop-order create method whose last two arguments are (riid, ppv), riid at argument index R
// (0-based, after the object). The created object is wrapped by its interface.
// Resource descriptions: the title sets flag bits desktop D3D12 rejects (0x20000 on buffers,
// 0x100000 and 0x200000 on render-target textures). Desktop receives only the bits it defines
// up to ALLOW_SIMULTANEOUS_ACCESS (0x3f); dropped bits are logged once per value.
// Console-only DXGI format values (116-120 in the console's format extensions: two 10-bit float
// formats, D16_UNORM_S8_UINT and its two typed views) mapped to the nearest desktop formats. The
// bit layouts differ, so contents are not interchangeable; this only lets the resources exist.
DXGI_FORMAT desktop_format(DXGI_FORMAT format)
{
    DXGI_FORMAT mapped = format;
    switch (static_cast<UINT>(format)) {
    case 116: case 117: mapped = DXGI_FORMAT_R10G10B10A2_UNORM; break;
    case 118: mapped = DXGI_FORMAT_D24_UNORM_S8_UINT; break;
    case 119: mapped = DXGI_FORMAT_R24_UNORM_X8_TYPELESS; break;
    case 120: mapped = DXGI_FORMAT_X24_TYPELESS_G8_UINT; break;
    default: break;
    }
    if (mapped != format) {
        static LONG calls;
        if (first_calls(calls, 6)) log("x2w-d3d12 console format %u -> desktop format %u", format, mapped);
    }
    return mapped;
}

constexpr UINT desktop_resource_flags = 0x3f;
D3D12_RESOURCE_DESC sanitized(const D3D12_RESOURCE_DESC& desc)
{
    D3D12_RESOURCE_DESC copy = desc;
    const UINT dropped = static_cast<UINT>(desc.Flags) & ~desktop_resource_flags;
    if (dropped) {
        static UINT seen[16];
        static LONG count;
        bool known = false;
        for (LONG i = 0; i < count && i < 16; ++i) known |= seen[i] == dropped;
        if (!known && count < 16) {
            seen[count++] = dropped;
            log("x2w-d3d12 resource flags 0x%x: dropped Xbox bits 0x%x (dimension %d)", desc.Flags, dropped, desc.Dimension);
        }
    }
    copy.Flags = static_cast<D3D12_RESOURCE_FLAGS>(static_cast<UINT>(desc.Flags) & desktop_resource_flags);
    // Depth-stencil with unordered access or render target (seen: 0x6 on a depth texture) is refused by desktop.
    if (copy.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        copy.Flags &= ~(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    // The Layout field carries a console swizzle mode (0x100 and up, the value XGSuggestSwizzleMode
    // works on); desktop gets UNKNOWN. A 4 KiB alignment request becomes the desktop default.
    if (static_cast<UINT>(desc.Layout) > D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE) {
        static LONG layouts;
        if (first_calls(layouts, 4)) log("x2w-d3d12 resource layout 0x%x (console swizzle) -> UNKNOWN", desc.Layout);
        copy.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    }
    if (desc.Alignment != 0 && desc.Alignment != 0x10000 && desc.Alignment != 0x400000) copy.Alignment = 0;
    copy.Format = desktop_format(desc.Format);
    // A 16-bit packed target the adapter cannot render to or write as a UAV: packed_target_format.h.
    if (the_device && x2w_startup::packed_target_format(copy.Format)) {
        ID3D12Device* const device = static_cast<ID3D12Device*>(the_device->own().real);
        const DXGI_FORMAT stand_in = x2w_startup::packed_target_stand_in(copy, [device](DXGI_FORMAT format) -> UINT {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
            return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)))
                ? static_cast<UINT>(support.Support1) : 0u;
        });
        if (stand_in != copy.Format) {
            static LONG stand_ins;
            if (first_calls(stand_ins, 4))
                log("x2w-d3d12 texture format %d with flags 0x%x: shortcut - not supported for that use here, created as format %d",
                    copy.Format, copy.Flags, stand_in);
            copy.Format = stand_in;
        }
    }
    return copy;
}

template <int K, int N, int R, int Desc = -1>
__declspec(noinline) u64 create_slot(Wrapper* self, u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g, u64 h)
{
    Timed timed(time_create);
    u64 args[8] = {a, b, c, d, e, f, g, h};
    for (int i = 0; i < R; ++i) args[i] = unwrap(args[i]);
    D3D12_RESOURCE_DESC desc_copy{};
    DXGI_FORMAT requested_format = DXGI_FORMAT_UNKNOWN;
    if (Desc >= 0 && args[Desc]) {
        requested_format = desktop_format(reinterpret_cast<const D3D12_RESOURCE_DESC*>(args[Desc])->Format);
        desc_copy = sanitized(*reinterpret_cast<const D3D12_RESOURCE_DESC*>(args[Desc]));
        args[Desc] = reinterpret_cast<u64>(&desc_copy);
    }
    const u64 result = method_of(self->own().real, N)(self->own().real, args[0], args[1], args[2], args[3], args[4], args[5],
                                                args[6], args[7]);
    check_device(kind_names[K], N);
    const auto* iid = reinterpret_cast<const GUID*>(args[R]);
    auto** out = reinterpret_cast<void**>(args[R + 1]);
    Kind kind = other_kind;
    if (SUCCEEDED(static_cast<HRESULT>(result)) && iid && out && *out) {
        kind = kind_for(*iid);
        if (Desc >= 0 && kind == resource_kind && requested_format != DXGI_FORMAT_UNKNOWN)
            x2w_startup::mark_packed_stand_in(static_cast<ID3D12Resource*>(*out), requested_format);
        Wrapper* child = wrap(static_cast<IUnknown*>(*out), kind);
        if (K == device_kind && kind == list_kind) {
            child->own().list_device = self;
            if (N == 12) child->own().list_node_mask = static_cast<UINT>(a);
        }
        *out = child;
    }
    static LONG calls;
    if (first_calls(calls, 4)) {
        char text[48];
        guid_text(iid, text, sizeof(text));
        log("x2w-d3d12 %s::slot%d create(0x%llx, 0x%llx, 0x%llx, 0x%llx) riid=%s -> 0x%08llx %s", kind_names[K], N, a, b,
            c, d, text, result, kind_names[kind]);
    }
    return result;
}

// Methods returning a structure by value: accept the object in either of the first two registers.
Wrapper* receiver(void* a, void* b, void** result, const char* method)
{
    // Struct-returning methods arrive as (return buffer, object) or (object, return buffer). A return
    // buffer on the caller's stack can still hold a stale copy of an object's vtable pointer and then
    // looks like a wrapper too: when both do, the one on this thread's stack is the buffer.
    if (as_wrapper(a) && as_wrapper(b)) {
        ULONG_PTR low = 0, high = 0;
        GetCurrentThreadStackLimits(&low, &high);
        const auto on_stack = [&](void* p) { return reinterpret_cast<ULONG_PTR>(p) >= low && reinterpret_cast<ULONG_PTR>(p) < high; };
        static LONG noted;
        if (first_calls(noted, 4)) log("x2w-d3d12 %s: both arguments look like objects (%p, %p); the stack one is the result", method, a, b);
        if (on_stack(a) && !on_stack(b)) {
            *result = a;
            return as_wrapper(b);
        }
    }
    if (Wrapper* w = as_wrapper(a)) {
        *result = b;
        return w;
    }
    *result = a;
    if (Wrapper* w = as_wrapper(b)) return w;
    fail(method, a, b);
}

// Slot 0, all kinds.
HRESULT __stdcall query_interface(Wrapper* self, const GUID* iid, void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    HRESULT hr = E_NOINTERFACE;
    if (is_base_iid(*iid) || kind_for(*iid) == self->own().kind) {
        InterlockedIncrement(&self->own().refs);
        *out = self;
        hr = S_OK;
    }
    log("x2w-d3d12 %s::QueryInterface riid=%s -> 0x%08lx", kind_names[self->own().kind], text, static_cast<unsigned long>(hr));
    return hr;
}
ULONG __stdcall add_ref(Wrapper* self) { return static_cast<ULONG>(InterlockedIncrement(&self->own().refs)); }
// The wrappers and their desktop objects stay alive for the process (experiment scope).
void retire_placement(Wrapper* w);
ULONG __stdcall release(Wrapper* self)
{
    const auto refs = InterlockedDecrement(&self->own().refs);
    if (!refs && self->own().kind == resource_kind && self->own().title_memory) retire_placement(self);
    return static_cast<ULONG>(refs);
}

// Slot 7 of every device child, desktop ID3D12DeviceChild::GetDevice(riid, ppv).
HRESULT __stdcall child_get_device(Wrapper* self, const GUID* iid, void** out)
{
    if (!out || !iid) return E_POINTER;
    return query_interface(the_device, iid, out);
}

ID3D12Device* device_of(Wrapper* w) { return static_cast<ID3D12Device*>(w->own().real); }

// Device slot 8, desktop CreateCommandQueue(desc, riid, ppv).
HRESULT __stdcall device_create_command_queue(Wrapper* self, const D3D12_COMMAND_QUEUE_DESC* desc, const GUID* iid,
                                              void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    const HRESULT hr = kind_for(*iid) == queue_kind
        ? device_of(self)->CreateCommandQueue(desc, IID_PPV_ARGS(&queue)) : E_NOINTERFACE;
    log("x2w-d3d12 Device::CreateCommandQueue type=%d priority=%d flags=0x%x node=0x%x riid=%s -> 0x%08lx", desc->Type,
        desc->Priority, desc->Flags, desc->NodeMask, text, static_cast<unsigned long>(hr));
    if (FAILED(hr)) return hr;
    *out = wrap(queue, queue_kind);
    return S_OK;
}

// Device slot 48 (Xbox order; desktop slot 48 is OpenExistingHeapFromAddress): called four times at
// startup with a 16-byte queue descriptor (type 2, priority 0, flags 1/2/3/0, node 0/0/0/1), the
// desktop ID3D12CommandQueue IID and an output; the title asserts success. The Xbox flag meaning is
// unknown; the desktop queue is created with flags 0.
HRESULT __stdcall device_create_command_queue_x(Wrapper* self, const D3D12_COMMAND_QUEUE_DESC* desc, const GUID* iid,
                                                void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    D3D12_COMMAND_QUEUE_DESC copy = *desc;
    copy.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ID3D12CommandQueue* queue = nullptr;
    const HRESULT hr = kind_for(*iid) == queue_kind
        ? device_of(self)->CreateCommandQueue(&copy, IID_PPV_ARGS(&queue)) : E_NOINTERFACE;
    log("x2w-d3d12 Device::slot48(CreateCommandQueue?) type=%d priority=%d flags=0x%x(dropped) node=0x%x riid=%s -> 0x%08lx",
        desc->Type, desc->Priority, desc->Flags, desc->NodeMask, text, static_cast<unsigned long>(hr));
    if (FAILED(hr)) return hr;
    *out = wrap(queue, queue_kind);
    return S_OK;
}

// Device slot 10/11, desktop CreateGraphics/ComputePipelineState(desc, riid, ppv): logged and refused
// until the Xbox shader bytecode is translated (the title's shaders are not DXIL).
template <int N>
HRESULT __stdcall device_create_pipeline(Wrapper*, const void* desc, const GUID* iid, void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    char bytes[3 * 64 + 1];
    hex_text(desc, 64, bytes, sizeof(bytes));
    log("x2w-d3d12 Device::slot%d(Create%sPipelineState) desc=[%s] riid=%s -> trap", N, N == 10 ? "Graphics" : "Compute",
        bytes, text);
    if (out) *out = nullptr;
    trap_slot<device_kind, N>(0, reinterpret_cast<u64>(desc), reinterpret_cast<u64>(iid), reinterpret_cast<u64>(out));
    return E_NOTIMPL;
}

void note_console_size(const D3D12_RESOURCE_DESC& desc, const D3D12_RESOURCE_ALLOCATION_INFO& info)
{
    static LONG calls;
    if (first_calls(calls, 8))
        log("x2w-d3d12 allocation info: console size 0x%llx for %llux%u format %d samples %u (shortcut: desktop is larger)",
            info.SizeInBytes, desc.Width, desc.Height, desc.Format, desc.SampleDesc.Count);
}

// Device slot 25, desktop GetResourceAllocationInfo(visibleMask, count, descs), returned by value.
void* __stdcall device_resource_allocation_info(void* a, void* b, UINT mask, UINT count, const D3D12_RESOURCE_DESC* descs)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "Device::GetResourceAllocationInfo(no receiver)");
    std::vector<D3D12_RESOURCE_DESC> copies(count);
    for (UINT i = 0; i < count; ++i) copies[i] = sanitized(descs[i]);
    D3D12_RESOURCE_ALLOCATION_INFO info = device_of(self)->GetResourceAllocationInfo(mask, count, copies.data());
    // A single description of a family that is larger on desktop gets its console size (console_plane_size.h).
    if (count == 1 && x2w_startup::console_plane_size(descs[0], info)) note_console_size(descs[0], info);
    *static_cast<D3D12_RESOURCE_ALLOCATION_INFO*>(result) = info;
    static LONG calls, invalid;
    if (first_calls(calls))
        log("x2w-d3d12 Device::GetResourceAllocationInfo count=%u dim=%d -> size=0x%llx align=0x%llx", count,
            count && descs ? descs[0].Dimension : -1, info.SizeInBytes, info.Alignment);
    if (info.SizeInBytes == ~0ull && first_calls(invalid, 8)) {
        for (UINT i = 0; i < count && i < 4; ++i) {
            char bytes[3 * 0x38 + 1];
            hex_text(&descs[i], 0x38, bytes, sizeof(bytes));
            log("x2w-d3d12   invalid desc[%u]=[%s]", i, bytes);
        }
    }
    return result;
}

// Device slot 38, desktop GetCopyableFootprints(desc, first, count, base, layouts, rows, row bytes, total).
void __stdcall device_copyable_footprints(Wrapper* self, const D3D12_RESOURCE_DESC* desc, UINT first, UINT count,
                                          UINT64 base, D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layouts, UINT* rows,
                                          UINT64* row_bytes, UINT64* total)
{
    const D3D12_RESOURCE_DESC copy = sanitized(*desc);
    device_of(self)->GetCopyableFootprints(&copy, first, count, base, layouts, rows, row_bytes, total);
}

// Device slot 41, desktop CreateCommandSignature(desc, root, riid, ppv); on failure the descriptor and
// its argument descriptors are logged.
u64 __stdcall device_create_command_signature(Wrapper* self, const D3D12_COMMAND_SIGNATURE_DESC* desc, u64 root,
                                              u64 iid, u64 out)
{
    const u64 result = create_slot<device_kind, 41, 2>(self, reinterpret_cast<u64>(desc), root, iid, out, 0, 0, 0, 0);
    if (SUCCEEDED(static_cast<HRESULT>(result)) && out && desc && !root && desc->NumArgumentDescs == 1) {
        auto* signature = *reinterpret_cast<Wrapper**>(out);
        if (signature && signature->own().kind == command_signature_kind) {
            auto& state = signature->own();
            const auto type = desc->pArgumentDescs[0].Type;
            if (type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH && desc->ByteStride == sizeof(D3D12_DISPATCH_ARGUMENTS))
                state.indirect_command = IndirectCommand::dispatch;
            if (type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW && desc->ByteStride == sizeof(D3D12_DRAW_ARGUMENTS))
                state.indirect_command = IndirectCommand::draw;
            if (type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED && desc->ByteStride == sizeof(D3D12_DRAW_INDEXED_ARGUMENTS))
                state.indirect_command = IndirectCommand::draw_indexed;
            if (type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH && desc->ByteStride == sizeof(D3D12_DISPATCH_MESH_ARGUMENTS))
                state.indirect_command = IndirectCommand::dispatch_mesh;
            if (state.indirect_command != IndirectCommand::none) state.signature_device = self;
        }
    }
    static LONG failures;
    if (FAILED(static_cast<HRESULT>(result)) && desc && first_calls(failures, 8)) {
        log("x2w-d3d12   CreateCommandSignature failed: stride=%u args=%u node=0x%x root=0x%llx", desc->ByteStride,
            desc->NumArgumentDescs, desc->NodeMask, root);
        for (UINT i = 0; i < desc->NumArgumentDescs && i < 8; ++i) {
            char bytes[3 * 16 + 1];
            hex_text(&desc->pArgumentDescs[i], sizeof(D3D12_INDIRECT_ARGUMENT_DESC), bytes, sizeof(bytes));
            log("x2w-d3d12     arg[%u]=[%s]", i, bytes);
        }
    }
    // Argument types outside the desktop enumeration (0x69 seen, stride 12) cannot be created on
    // desktop; the title gets a placeholder CommandSignature.
    bool console_type = false;
    for (UINT i = 0; desc && i < desc->NumArgumentDescs; ++i)
        console_type |= static_cast<UINT>(desc->pArgumentDescs[i].Type) > D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
    if (FAILED(static_cast<HRESULT>(result)) && console_type && out) {
        log("x2w-d3d12   CreateCommandSignature: console argument type -> placeholder");
        *reinterpret_cast<void**>(out) = wrap(nullptr, command_signature_kind);
        return S_OK;
    }
    return result;
}

// Device slot 108 (Xbox order): (mask, count, descs, info1) returned by value, the desktop
// ID3D12Device4::GetResourceAllocationInfo1 shape. The caller asks with a 4 KiB alignment first and
// repeats with 0 when the result's alignment is not 4 KiB.
void* __stdcall device_resource_allocation_info1(void* a, void* b, UINT mask, UINT count, const D3D12_RESOURCE_DESC* descs,
                                                 D3D12_RESOURCE_ALLOCATION_INFO1* info1)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "Device::slot108(no receiver)");
    std::vector<D3D12_RESOURCE_DESC> copies(count);
    for (UINT i = 0; i < count; ++i) copies[i] = sanitized(descs[i]);
    D3D12_RESOURCE_ALLOCATION_INFO info{~0ull, 0};
    ID3D12Device4* device4 = nullptr;
    if (SUCCEEDED(device_of(self)->QueryInterface(IID_PPV_ARGS(&device4)))) {
        info = device4->GetResourceAllocationInfo1(mask, count, copies.data(), info1);
        device4->Release();
    }
    if (count == 1 && x2w_startup::console_plane_size(descs[0], info)) {
        if (info1) *info1 = {0, info.Alignment, info.SizeInBytes};
        note_console_size(descs[0], info);
    }
    *static_cast<D3D12_RESOURCE_ALLOCATION_INFO*>(result) = info;
    static LONG calls, invalid;
    if (first_calls(calls))
        log("x2w-d3d12 Device::slot108(GetResourceAllocationInfo1?) count=%u dim=%d align=0x%llx info1=%p -> size=0x%llx "
            "align=0x%llx", count, count ? descs[0].Dimension : -1, count ? descs[0].Alignment : 0ull,
            static_cast<void*>(info1), info.SizeInBytes, info.Alignment);
    if (info.SizeInBytes == ~0ull && first_calls(invalid, 8)) {
        char bytes[3 * 0x38 + 1];
        hex_text(descs, 0x38, bytes, sizeof(bytes));
        log("x2w-d3d12   slot108 invalid desc[0]=[%s]", bytes);
    }
    return result;
}

// Device slot 26, desktop GetCustomHeapProperties(nodeMask, heapType), returned by value.
void* __stdcall device_custom_heap_properties(void* a, void* b, UINT node_mask, D3D12_HEAP_TYPE type)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "Device::GetCustomHeapProperties(no receiver)");
    const D3D12_HEAP_PROPERTIES properties = device_of(self)->GetCustomHeapProperties(node_mask, type);
    *static_cast<D3D12_HEAP_PROPERTIES*>(result) = properties;
    static LONG calls;
    if (first_calls(calls))
        log("x2w-d3d12 Device::GetCustomHeapProperties order=%s node=0x%x type=%d -> type=%d page=%d pool=%d",
            result == a ? "ret,this" : "this,ret", node_mask, type, properties.Type, properties.CPUPageProperty,
            properties.MemoryPoolPreference);
    return result;
}

// Device slot 43, desktop GetAdapterLuid(), returned by value.

// Device slots 34/35, desktop MakeResident/Evict(count, objects).
template <int N>
HRESULT __stdcall device_residency(Wrapper* self, UINT count, void* const* objects)
{
    std::vector<ID3D12Pageable*> real(count);
    for (UINT i = 0; i < count; ++i) real[i] = reinterpret_cast<ID3D12Pageable*>(unwrap(reinterpret_cast<u64>(objects[i])));
    return N == 34 ? device_of(self)->MakeResident(count, real.data()) : device_of(self)->Evict(count, real.data());
}

// ---------------------------------------------------------------------------------------------
// X2W_D3D12_RENDER=1: first stage of real rendering. Command lists are really recorded for the
// calls implemented below (reset, close, render-target clears), really submitted, fences complete
// when the desktop GPU finishes, and the presented surface is read back and shown in the window.
// Everything else on a list is still dropped.
void release_targets(Wrapper* self);
void reset_viewport_state(Wrapper* self);
void discard_draw_sampler_heaps(ID3D12CommandList* list);
void use_as_target(Wrapper* self, ID3D12Resource* resource);
bool is_container(const void* p);
bool shader_dump_dir(wchar_t* out, DWORD size);
void write_file(const wchar_t* path, const void* data, std::size_t bytes);
bool render_mode()
{
    static const bool on = [] {
        char value[4]{};
        return GetEnvironmentVariableA("X2W_D3D12_RENDER", value, sizeof(value)) && value[0] == '1';
    }();
    return on;
}

// Render-target view handle (desktop) -> the resource it views, recorded when the view is created.
SRWLOCK target_lock = SRWLOCK_INIT;
struct ViewTarget {
    ID3D12Resource* resource;
    int slot;  // 20 RTV, 21 DSV
};
std::unordered_map<u64, ViewTarget>& view_targets()
{
    static std::unordered_map<u64, ViewTarget> map;
    return map;
}
void note_view_target(u64 handle, ID3D12Resource* resource, int slot)
{
    AcquireSRWLockExclusive(&target_lock);
    view_targets()[handle] = {resource, slot};
    ReleaseSRWLockExclusive(&target_lock);
}
// slot 0: either kind. A handle created as the other kind yields nothing.
ID3D12Resource* view_target(u64 handle, int slot = 0)
{
    AcquireSRWLockShared(&target_lock);
    const auto found = view_targets().find(handle);
    ID3D12Resource* resource = found == view_targets().end() || (slot && found->second.slot != slot) ? nullptr : found->second.resource;
    ReleaseSRWLockShared(&target_lock);
    return resource;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = from;
    barrier.Transition.StateAfter = to;
    list->ResourceBarrier(1, &barrier);
}

LONG draw_dump_frame();
x2w_startup::DrawReadbacks& draw_dumps()
{
    static x2w_startup::DrawReadbacks tracker;
    return tracker;
}

LONG render_resets, render_closes, render_clears, render_executes, render_reset_failures;
// OMSetRenderTargets calls that bound targets, and render target / depth handles that resolved to no view.
LONG render_target_binds, render_target_unresolved;
// ClearDepthStencilView calls forwarded to the desktop list, and calls refused by rule (not counting an unresolved view).
LONG render_depth_clears, render_depth_clears_refused;

// A console recording survives its allocator's Reset; a desktop one is discarded with it, and
// resetting an allocator whose lists the GPU still runs is invalid. Each title allocator therefore
// stands for a pool of desktop allocators. A list records on the pool's current one. In render
// mode the title's Reset resets the current one only when no list holds a recording on it and
// every queue has passed its last submit; otherwise the title continues on an idle allocator from
// the pool, or a new one, and the busy one keeps its recordings. Wrappers are never freed, so
// their addresses are stable keys.
struct DesktopAllocator {
    ID3D12CommandAllocator* real;
    LONG lists;                                             // lists whose recording is on this allocator
    std::vector<std::pair<ID3D12Fence*, UINT64>> submits;  // per queue fence: the last submit's value
    // A submit whose completion fence could not be created or signalled: the GPU may still run it
    // and nothing would say when it ends, so the allocator is never reset or reused again.
    bool untracked;
    // A command list recorded on this allocator failed Close(). Do not open another list on it
    // until it is idle and Reset(); hardware drivers can fault if a second recording starts on
    // an allocator whose failed recording is still attached.
    bool close_failed;
};
struct TitleAllocator {
    std::vector<DesktopAllocator*> pool;
    DesktopAllocator* current = nullptr;
    D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT;  // from the lists recorded on it
};
SRWLOCK allocator_lock = SRWLOCK_INIT;
std::unordered_map<Wrapper*, TitleAllocator> title_allocators;
std::unordered_map<Wrapper*, DesktopAllocator*> list_recordings;  // list -> allocator holding its recording
std::unordered_map<ID3D12CommandQueue*, std::pair<ID3D12Fence*, UINT64>> queue_submits;
LONG allocator_swaps, allocator_creates;

// allocator_lock held.
TitleAllocator& title_allocator(Wrapper* allocator)
{
    TitleAllocator& title = title_allocators[allocator];
    if (!title.current) {
        title.current = new DesktopAllocator{static_cast<ID3D12CommandAllocator*>(allocator->own().real), 0, {}, false, false};
        title.pool.push_back(title.current);
    }
    return title;
}

// allocator_lock held.
bool allocator_idle(const DesktopAllocator* allocator)
{
    if (allocator->lists || allocator->untracked) return false;
    for (const auto& submit : allocator->submits)
        if (submit.first->GetCompletedValue() < submit.second) return false;
    return true;
}

// Allocator slot 8, desktop Reset().
HRESULT __stdcall allocator_reset(Wrapper* self)
{
    if (!render_mode()) return static_cast<ID3D12CommandAllocator*>(self->own().real)->Reset();
    AcquireSRWLockExclusive(&allocator_lock);
    TitleAllocator& title = title_allocator(self);
    DesktopAllocator* next = nullptr;
    HRESULT hr = S_OK;
    if (allocator_idle(title.current)) {
        next = title.current;
    } else {
        for (DesktopAllocator* spare : title.pool)
            if (spare != title.current && allocator_idle(spare)) { next = spare; break; }
        if (!next) {
            ID3D12CommandAllocator* created = nullptr;
            hr = device_of(the_device)->CreateCommandAllocator(title.type, IID_PPV_ARGS(&created));
            if (SUCCEEDED(hr)) {
                next = new DesktopAllocator{created, 0, {}, false, false};
                title.pool.push_back(next);
                InterlockedIncrement(&allocator_creates);
            }
        }
    }
    if (next) {
        hr = next->real->Reset();
        if (SUCCEEDED(hr)) {
            next->submits.clear();
            next->close_failed = false;
            if (next != title.current) InterlockedIncrement(&allocator_swaps);
            title.current = next;
        }
    }
    const auto pool = title.pool.size();
    ReleaseSRWLockExclusive(&allocator_lock);
    static LONG calls;
    if ((FAILED(hr) || pool > 1) && first_calls(calls, 4))
        log("x2w-render allocator %p Reset -> 0x%08lx, %zu desktop allocators", static_cast<void*>(self),
            static_cast<unsigned long>(hr), pool);
    return hr;
}

// X2W_TEST_SUBMIT_FENCE_FAIL=1 (fixtures only): treat every submit fence as failed.
bool submit_fence_fails()
{
    static const bool on = GetEnvironmentVariableW(L"X2W_TEST_SUBMIT_FENCE_FAIL", nullptr, 0) != 0;
    return on;
}

// draw_lock and allocator_lock held, after the lists were executed on queue. Each list's allocator holds a fence value
// the queue must pass before the allocator can be reset. If that fence cannot be created or
// signalled, the work is on the GPU all the same: the allocators are marked untracked instead.
void note_submits(ID3D12CommandQueue* queue, const std::vector<Wrapper*>& lists)
{
    auto& [fence, value] = queue_submits[queue];
    if (!fence && !submit_fence_fails()) {
        ID3D12Device* device = nullptr;
        if (SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&device)))) {
            if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) fence = nullptr;
            device->Release();
        }
    }
    if (fence && SUCCEEDED(queue->Signal(fence, value + 1))) {
        ++value;
        for (Wrapper* list : lists) {
            const auto found = list_recordings.find(list);
            if (found == list_recordings.end()) continue;
            auto& submits = found->second->submits;
            const auto same = std::find_if(submits.begin(), submits.end(), [&](const auto& s) { return s.first == fence; });
            if (same == submits.end()) submits.emplace_back(fence, value);
            else same->second = value;
        }
    } else {
        for (Wrapper* list : lists)
            if (const auto found = list_recordings.find(list); found != list_recordings.end()) found->second->untracked = true;
        static LONG failures;
        if (first_calls(failures, 2)) log("x2w-render submit fence for allocator reuse failed; the allocators are not reused");
    }
}

// Fixtures: how many OMSetRenderTargets calls bound targets, and how many render target or depth handles
// (also in ClearRenderTargetView) resolved to no view.
extern "C" __declspec(dllexport) void x2w_d3d12_render_target_stats(unsigned* binds, unsigned* unresolved)
{
    if (binds) *binds = static_cast<unsigned>(InterlockedCompareExchange(&render_target_binds, 0, 0));
    if (unresolved) *unresolved = static_cast<unsigned>(InterlockedCompareExchange(&render_target_unresolved, 0, 0));
}

// Fixtures: ClearDepthStencilView calls forwarded, and calls refused for a read-only plane, a depth outside [0, 1] or
// unreadable rectangles. A view that holds nothing is counted by x2w_d3d12_render_target_stats instead.
extern "C" __declspec(dllexport) void x2w_d3d12_depth_clear_stats(unsigned* forwarded, unsigned* refused)
{
    if (forwarded) *forwarded = static_cast<unsigned>(InterlockedCompareExchange(&render_depth_clears, 0, 0));
    if (refused) *refused = static_cast<unsigned>(InterlockedCompareExchange(&render_depth_clears_refused, 0, 0));
}

// Fixtures: the number of desktop allocators a title allocator stands for (0 if it has none yet).
extern "C" __declspec(dllexport) unsigned x2w_d3d12_allocator_pool_size(void* allocator)
{
    Wrapper* a = as_wrapper(allocator);
    if (!a || a->own().kind != allocator_kind) return 0;
    AcquireSRWLockShared(&allocator_lock);
    const auto found = title_allocators.find(a);
    const unsigned size = found == title_allocators.end() ? 0 : static_cast<unsigned>(found->second.pool.size());
    ReleaseSRWLockShared(&allocator_lock);
    return size;
}

// List slot 139 (Xbox order): (allocator, 0, 0), the per-frame reset.
HRESULT __stdcall list_reset(Wrapper* self, void* allocator)
{
    Wrapper* a = as_wrapper(allocator);
    HRESULT hr = E_INVALIDARG;
    x2w_startup::ComPtr<ID3D12GraphicsCommandList> replaced;
    AcquireSRWLockExclusive(&allocator_lock);
    auto& state = self->own();
    auto* previous = static_cast<ID3D12GraphicsCommandList*>(state.real);
    if (a && a->own().kind == allocator_kind && previous) {
        TitleAllocator& title = title_allocator(a);
        const auto type = previous->GetType();
        DesktopAllocator* target = title.current;
        if (target->close_failed && title.type == type) {
            target = nullptr;
            for (DesktopAllocator* spare : title.pool) {
                if (spare != title.current && allocator_idle(spare) && SUCCEEDED(spare->real->Reset())) {
                    spare->submits.clear();
                    spare->close_failed = false;
                    target = spare;
                    break;
                }
            }
            if (!target) {
                ID3D12CommandAllocator* created = nullptr;
                hr = device_of(state.list_device)->CreateCommandAllocator(type, IID_PPV_ARGS(&created));
                if (SUCCEEDED(hr)) {
                    target = new DesktopAllocator{created, 0, {}, false, false};
                    title.pool.push_back(target);
                    InterlockedIncrement(&allocator_creates);
                }
            }
        }
        if (target) {
            if (state.list_close_failed) {
                ID3D12GraphicsCommandList* replacement = nullptr;
                hr = device_of(state.list_device)->CreateCommandList(state.list_node_mask, type, target->real,
                                                                   nullptr, IID_PPV_ARGS(&replacement));
                if (SUCCEEDED(hr)) {
                    // Keep the old object through recording cleanup. In-flight GPU work keeps
                    // its allocator and submitted descriptors through the existing fence tracking.
                    replaced.Attach(previous);
                    state.real = replacement;
                    state.list_close_failed = false;
                }
            } else {
                hr = previous->Reset(target->real, nullptr);
            }
        }
        if (SUCCEEDED(hr)) {
            if (target != title.current) {
                InterlockedIncrement(&allocator_swaps);
                title.current = target;
            }
            title.type = type;
            DesktopAllocator*& on = list_recordings[self];
            if (on) --on->lists;
            on = title.current;
            ++on->lists;
            state.list_ready = false;
            state.list_open = true;
        }
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    if (SUCCEEDED(hr)) {
        reset_viewport_state(self);
        discard_draw_sampler_heaps(previous);
        if (draw_dump_frame()) draw_dumps().discard(previous);
    }
    InterlockedIncrement(SUCCEEDED(hr) ? &render_resets : &render_reset_failures);
    static LONG failures;
    if (FAILED(hr) && first_calls(failures, 6))
        log("x2w-render list Reset failed 0x%08lx (allocator %p)", static_cast<unsigned long>(hr), allocator);
    return hr;
}

// List slot 9, desktop Close().
HRESULT __stdcall list_close(Wrapper* self)
{
    AcquireSRWLockExclusive(&allocator_lock);
    auto& state = self->own();
    if (!state.list_open) {
        ReleaseSRWLockExclusive(&allocator_lock);
        return S_OK;
    }
    auto* list = static_cast<ID3D12GraphicsCommandList*>(state.real);
    release_targets(self);
    const HRESULT hr = list->Close();
    state.list_open = false;
    state.list_ready = SUCCEEDED(hr);
    state.list_close_failed = FAILED(hr);
    if (FAILED(hr)) {
        if (const auto found = list_recordings.find(self); found != list_recordings.end() && found->second)
            found->second->close_failed = true;
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    InterlockedIncrement(&render_closes);
    if (FAILED(hr)) {
        discard_draw_sampler_heaps(list);
        if (draw_dump_frame()) draw_dumps().discard(list);
        static LONG failures;
        if (first_calls(failures, 6)) log("x2w-render list Close failed 0x%08lx", static_cast<unsigned long>(hr));
    }
    return hr;
}

// A fence signalled on a queue completes when the desktop GPU reaches it.
struct PendingSignal {
    Wrapper* fence;
    UINT64 value;
    HANDLE event;
};
void CALLBACK signal_reached(PTP_CALLBACK_INSTANCE, void* context, PTP_WAIT wait, TP_WAIT_RESULT)
{
    auto* pending = static_cast<PendingSignal*>(context);
    fence_reached(pending->fence, pending->value);
    CloseHandle(pending->event);
    delete pending;
    CloseThreadpoolWait(wait);
}
void complete_when_reached(Wrapper* fence, UINT64 value)
{
    auto* pending = new PendingSignal{fence, value, CreateEventW(nullptr, FALSE, FALSE, nullptr)};
    if (FAILED(static_cast<ID3D12Fence*>(fence->own().real)->SetEventOnCompletion(value, pending->event))) {
        fence_reached(fence, value);
        CloseHandle(pending->event);
        delete pending;
        return;
    }
    PTP_WAIT wait = CreateThreadpoolWait(&signal_reached, pending, nullptr);
    SetThreadpoolWait(wait, pending->event, nullptr);
}

// The presented surface, read back and reduced for the window (BGRA, top-down).
constexpr int shown_width = 960, shown_height = 540;
std::vector<std::uint32_t> shown_pixels;
LONG shown_frames;
SRWLOCK shown_lock = SRWLOCK_INIT;
char shown_note[160] = "no frame read back yet";

void write_file(const wchar_t* path, const void* data, std::size_t bytes);
void read_back_presented(ID3D12CommandQueue* queue, ID3D12Resource* surface)
{
    static ID3D12CommandAllocator* allocator;
    static ID3D12GraphicsCommandList* list;
    static ID3D12Fence* fence;
    static ID3D12Resource* readback;
    static UINT64 readback_bytes, fence_value;
    static HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ID3D12Device* device = device_of(the_device);
    const D3D12_RESOURCE_DESC desc = surface->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);
    const bool rgba = desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM || desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    const bool bgra = desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM || desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    if ((!rgba && !bgra) || total == 0 || total == ~0ull) {
        std::snprintf(shown_note, sizeof(shown_note), "presented surface format %d not shown yet", desc.Format);
        return;
    }
    HRESULT hr = S_OK;
    if (!allocator) {
        hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
        if (SUCCEEDED(hr)) hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list));
        if (SUCCEEDED(hr)) hr = list->Close();
        if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    }
    if (SUCCEEDED(hr) && (!readback || readback_bytes < total)) {
        if (readback) readback->Release();
        readback = nullptr;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = total;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&readback));
        readback_bytes = SUCCEEDED(hr) ? total : 0;
    }
    if (SUCCEEDED(hr)) hr = allocator->Reset();
    if (SUCCEEDED(hr)) hr = list->Reset(allocator, nullptr);
    if (FAILED(hr)) {
        std::snprintf(shown_note, sizeof(shown_note), "readback setup failed 0x%08lx", static_cast<unsigned long>(hr));
        return;
    }
    transition(list, surface, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = surface;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    to.pResource = readback;
    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    to.PlacedFootprint = footprint;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    transition(list, surface, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    list->Close();
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence, ++fence_value);
    fence->SetEventOnCompletion(fence_value, event);
    if (WaitForSingleObject(event, 3000) != WAIT_OBJECT_0) {
        std::snprintf(shown_note, sizeof(shown_note), "readback timed out");
        return;
    }
    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(total)};
    if (FAILED(readback->Map(0, &range, &mapped))) return;
    AcquireSRWLockExclusive(&shown_lock);
    shown_pixels.resize(shown_width * shown_height);
    std::uint64_t sum = 0;
    for (int y = 0; y < shown_height; ++y) {
        const auto* row = static_cast<const unsigned char*>(mapped) + footprint.Offset +
                          static_cast<UINT64>(y) * desc.Height / shown_height * footprint.Footprint.RowPitch;
        for (int x = 0; x < shown_width; ++x) {
            const unsigned char* texel = row + static_cast<UINT64>(x) * desc.Width / shown_width * 4;
            const unsigned r = rgba ? texel[0] : texel[2], g = texel[1], b = rgba ? texel[2] : texel[0];
            shown_pixels[y * shown_width + x] = (r << 16) | (g << 8) | b;
            sum += r + g + b;
        }
    }
    ReleaseSRWLockExclusive(&shown_lock);
    // X2W_D3D12_DUMP_SURFACE=<file>: every 40th read-back is also written at full resolution as a
    // 32-bit BMP (overwritten each time), for looking at detail the reduced overlay copy loses.
    {
        static wchar_t dump_path[MAX_PATH];
        static const bool dump = GetEnvironmentVariableW(L"X2W_D3D12_DUMP_SURFACE", dump_path, MAX_PATH) != 0;
        if (dump && (shown_frames % 4) == 3 && desc.Width <= 8192 && desc.Height <= 8192) {
            const UINT width = static_cast<UINT>(desc.Width), height = desc.Height;
            std::vector<unsigned char> file(54 + std::size_t{width} * height * 4);
            const std::uint32_t header[] = {static_cast<std::uint32_t>(file.size()), 0, 54, 40, width, static_cast<std::uint32_t>(-static_cast<int>(height)),
                                            1 | (32u << 16), 0, width * height * 4, 2835, 2835, 0, 0};
            file[0] = 'B';
            file[1] = 'M';
            std::memcpy(file.data() + 2, header, sizeof(header));
            for (UINT y = 0; y < height; ++y) {
                const auto* row = static_cast<const unsigned char*>(mapped) + footprint.Offset + UINT64{y} * footprint.Footprint.RowPitch;
                unsigned char* out = file.data() + 54 + std::size_t{y} * width * 4;
                for (UINT x = 0; x < width; ++x) {
                    out[x * 4 + 0] = rgba ? row[x * 4 + 2] : row[x * 4 + 0];
                    out[x * 4 + 1] = row[x * 4 + 1];
                    out[x * 4 + 2] = rgba ? row[x * 4 + 0] : row[x * 4 + 2];
                    out[x * 4 + 3] = 255;
                }
            }
            write_file(dump_path, file.data(), file.size());
        }
    }
    const D3D12_RANGE none{0, 0};
    readback->Unmap(0, &none);
    const LONG count = InterlockedIncrement(&shown_frames);
    std::snprintf(shown_note, sizeof(shown_note), "surface %llux%u format %d, readback %ld, mean level %.2f/255",
                  desc.Width, desc.Height, desc.Format, count, sum / (3.0 * shown_width * shown_height));
    if (count <= 3 || count % 40 == 0) log("x2w-render presented %s", shown_note);
}

// ---------------------------------------------------------------------------------------------
// Render stage 2: real pipelines and draws for the shaders that have a translation.
//
// Stream and descriptor graphics pipelines share the VS/PS translation cache (d3d12_draw.inl).
std::size_t readable_window(u64 address, std::size_t wanted);
#include "startup_stubs/d3d12_capture.inl"

struct RootParameter {
    D3D12_ROOT_PARAMETER_TYPE type;
    UINT shader_register, space;
};
SRWLOCK root_lock = SRWLOCK_INIT;
std::unordered_map<const Wrapper*, std::vector<RootParameter>>& root_infos()
{
    static std::unordered_map<const Wrapper*, std::vector<RootParameter>> map;
    return map;
}
// The heap-indexing flags (x2w_startup::root_flags_heap_indexed) each title root signature was created with.
std::unordered_map<const Wrapper*, UINT>& root_heap_flag_map()
{
    static std::unordered_map<const Wrapper*, UINT> map;
    return map;
}
UINT root_heap_flags(const Wrapper* root)
{
    AcquireSRWLockShared(&root_lock);
    const auto found = root_heap_flag_map().find(root);
    const UINT flags = found == root_heap_flag_map().end() ? 0 : found->second;
    ReleaseSRWLockShared(&root_lock);
    return flags;
}
const std::vector<RootParameter>* root_info(const Wrapper* root)
{
    AcquireSRWLockShared(&root_lock);
    const auto found = root_infos().find(root);
    const std::vector<RootParameter>* info = found == root_infos().end() ? nullptr : &found->second;
    ReleaseSRWLockShared(&root_lock);
    return info;
}

// Device slot 16, desktop CreateRootSignature(node, blob, size, riid, ppv); the parameter list is
// kept for the translated shaders (register and space of each root descriptor), and the heap-indexing
// flags for command-list slot 34.
HRESULT __stdcall device_create_root_signature(Wrapper* self, UINT node, const void* blob, SIZE_T size, const GUID* iid, void** out)
{
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    ID3D12RootSignature* root = nullptr;
    const HRESULT hr = kind_for(*iid) == root_signature_kind
        ? device_of(self)->CreateRootSignature(node, blob, size, IID_PPV_ARGS(&root)) : E_NOINTERFACE;
    static LONG calls;
    const bool log_it = first_calls(calls, 6);
    if (FAILED(hr)) {
        log("x2w-d3d12 Device::CreateRootSignature size=0x%llx -> 0x%08lx", static_cast<u64>(size), static_cast<unsigned long>(hr));
        return hr;
    }
    Wrapper* w = wrap(root, root_signature_kind);
    note_root_blob(w, blob, size);
    std::vector<RootParameter> parameters;
    UINT heap_flags = 0;
    ID3D12VersionedRootSignatureDeserializer* deserializer = nullptr;
    if (SUCCEEDED(D3D12CreateVersionedRootSignatureDeserializer(blob, size, IID_PPV_ARGS(&deserializer)))) {
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc = nullptr;
        if (SUCCEEDED(deserializer->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &desc)) && desc) {
            heap_flags = static_cast<UINT>(desc->Desc_1_1.Flags) & x2w_startup::root_flags_heap_indexed;
            w->own().root_without_bindings = !heap_flags && !desc->Desc_1_1.NumParameters && !desc->Desc_1_1.NumStaticSamplers;
            for (UINT i = 0; i < desc->Desc_1_1.NumParameters; ++i) {
                const auto& parameter = desc->Desc_1_1.pParameters[i];
                RootParameter info{parameter.ParameterType, 0, 0};
                if (parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_CBV || parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_SRV ||
                    parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_UAV) {
                    info.shader_register = parameter.Descriptor.ShaderRegister;
                    info.space = parameter.Descriptor.RegisterSpace;
                }
                parameters.push_back(info);
            }
        }
        deserializer->Release();
    }
    if (log_it) {
        char text[256] = "";
        int used = 0;
        for (std::size_t i = 0; i < parameters.size() && used < 220; ++i)
            used += std::snprintf(text + used, sizeof(text) - used, " %d:r%u,s%u", parameters[i].type, parameters[i].shader_register,
                                  parameters[i].space);
        log("x2w-d3d12 Device::CreateRootSignature size=0x%llx -> %p heap flags 0x%x parameters (type:register,space)%s",
            static_cast<u64>(size), static_cast<void*>(w), heap_flags, text);
    }
    AcquireSRWLockExclusive(&root_lock);
    root_infos()[w] = std::move(parameters);
    root_heap_flag_map()[w] = heap_flags;
    ReleaseSRWLockExclusive(&root_lock);
    *out = w;
    return S_OK;
}

bool readable_shader_container(const D3D12_SHADER_BYTECODE& code)
{
    return code.BytecodeLength >= 32 && code.BytecodeLength <= (1u << 20) &&
        x2w_local::readable(code.pShaderBytecode, code.BytecodeLength) &&
        !std::memcmp(code.pShaderBytecode, "DXBC", 4);
}

bool desktop_indirect_graphics_shader(const D3D12_SHADER_BYTECODE& code, D3D12_SHADER_VERSION_TYPE stage)
{
    if (!readable_shader_container(code)) return false;
    x2w_startup::ComPtr<ID3D12ShaderReflection> shader;
    if (FAILED(D3DReflect(code.pShaderBytecode, code.BytecodeLength, IID_PPV_ARGS(shader.GetAddressOf())))) return false;
    D3D12_SHADER_DESC desc{};
    if (FAILED(shader->GetDesc(&desc)) || desc.Version >> 16 != stage ||
        ((desc.Version >> 4) & 0xf) != 5 || desc.BoundResources) return false;
    if (stage == D3D12_SHVER_VERTEX_SHADER) {
        for (UINT i = 0; i < desc.InputParameters; ++i) {
            D3D12_SIGNATURE_PARAMETER_DESC input{};
            if (FAILED(shader->GetInputParameterDesc(i, &input)) ||
                (input.SystemValueType != D3D_NAME_VERTEX_ID && input.SystemValueType != D3D_NAME_INSTANCE_ID))
                return false;
        }
    }
    return true;
}

constexpr std::size_t stream_sizes[23] = {16, 24, 24, 24, 24, 24, 24, 40, 336, 8, 48, 56, 24, 8, 8, 40, 8, 16, 8, 24, 8, 64, 24};

std::size_t desktop_stream_size(UINT type)
{
    // Only native pipeline creation accepts a mesh shader subobject.
    if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS) return 24;
    return type < std::size(stream_sizes) ? stream_sizes[type] : 0;
}

// Converts a console pipeline stream to desktop form and creates the pipeline. The subobject type
// numbers and sizes are the desktop ones; only the pointers and formats inside need converting.
ID3D12PipelineState* build_stream_pipeline(const D3D12_PIPELINE_STATE_STREAM_DESC* desc, char* why, std::size_t why_size,
                                          bool& compute, IndirectGraphicsPipeline& graphics)
{
    const std::size_t bytes = desc->SizeInBytes;
    if (bytes < 16 || bytes > 0x2000 || (bytes & 7) || !x2w_local::readable(desc->pPipelineStateSubobjectStream, bytes)) {
        std::snprintf(why, why_size, "stream not readable");
        return nullptr;
    }
    std::vector<u64> copy(bytes / 8);
    std::memcpy(copy.data(), desc->pPipelineStateSubobjectStream, bytes);
    auto* stream = reinterpret_cast<unsigned char*>(copy.data());
    const Wrapper* root = nullptr;
    bool vertex = false, pixel = false, mesh = false, input_layout = false;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
    D3D12_RT_FORMAT_ARRAY targets{};
    DXGI_FORMAT depth = DXGI_FORMAT_UNKNOWN;
    DXGI_SAMPLE_DESC samples{1, 0};
    bool ok = true;
    // First pass: validate the layout and find the wrapped root signature.
    for (std::size_t offset = 0; offset < bytes;) {
        const UINT type = *reinterpret_cast<UINT*>(stream + offset);
        const auto size = desktop_stream_size(type);
        if (!size || size > bytes - offset) {
            ok = false;
            std::snprintf(why, why_size, "unknown or truncated subobject %u at +0x%zx", type, offset);
            break;
        }
        if (type == 0) root = as_wrapper(*reinterpret_cast<void**>(stream + offset + 8));
        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS) mesh = true;
        offset += size;
    }
    const bool empty_root = root && root->own().kind == root_signature_kind && root->own().root_without_bindings;
    for (std::size_t offset = 0; ok && offset < bytes;) {
        const UINT type = *reinterpret_cast<UINT*>(stream + offset);
        if (type == 0) {
            u64& pointer = *reinterpret_cast<u64*>(stream + offset + 8);
            pointer = unwrap(pointer);
        } else if (type >= 1 && type <= 5) {
            const auto* code = reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + offset + 8);
            if (code->pShaderBytecode) {
                const bool is_vertex = type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS;
                bool supported = false;
                if (empty_root && type <= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS) {
                    // The native PSO validates the DXIL stages against the empty root.
                    supported = mesh ? !is_vertex && readable_shader_container(*code)
                        : desktop_indirect_graphics_shader(*code, is_vertex ? D3D12_SHVER_VERTEX_SHADER : D3D12_SHVER_PIXEL_SHADER);
                }
                if (!supported) {
                    std::snprintf(why, why_size, "stage %u has no translation or supported desktop binding", type);
                    ok = false;
                    break;
                }
                if (is_vertex) vertex = true;
                else pixel = true;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS) {
            const auto* code = reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + offset + 8);
            if (!empty_root || !readable_shader_container(*code)) {
                std::snprintf(why, why_size, "mesh stage needs desktop bytecode and an empty root signature");
                ok = false;
                break;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS) {
            const auto* code = reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + offset + 8);
            compute = code->pShaderBytecode && code->BytecodeLength;
        } else if (mesh && type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT) {
            const auto* output = reinterpret_cast<const D3D12_STREAM_OUTPUT_DESC*>(stream + offset + 8);
            if (output->NumEntries || output->NumStrides) {
                std::snprintf(why, why_size, "mesh streams cannot use stream output");
                ok = false;
                break;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT) {
            input_layout = reinterpret_cast<const D3D12_INPUT_LAYOUT_DESC*>(stream + offset + 8)->NumElements != 0;
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY) {
            topology = *reinterpret_cast<const D3D12_PRIMITIVE_TOPOLOGY_TYPE*>(stream + offset + 4);
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS) {
            auto* formats = reinterpret_cast<D3D12_RT_FORMAT_ARRAY*>(stream + offset + 4);
            for (auto& format : formats->RTFormats) format = desktop_format(format);
            targets = *formats;
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT) {
            auto* format = reinterpret_cast<DXGI_FORMAT*>(stream + offset + 4);
            *format = desktop_format(*format);
            depth = *format;
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC) {
            samples = *reinterpret_cast<const DXGI_SAMPLE_DESC*>(stream + offset + 4);
        }
        offset += desktop_stream_size(type);
    }
    if (ok && (vertex || pixel || mesh) &&
        ((!vertex && !mesh) || !pixel || compute || input_layout || topology != D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE ||
         targets.NumRenderTargets != 1 || targets.RTFormats[0] == DXGI_FORMAT_UNKNOWN || depth != DXGI_FORMAT_UNKNOWN ||
         samples.Count != 1 || samples.Quality)) {
        std::snprintf(why, why_size, "desktop indirect graphics stream needs VS/PS or MS/PS, triangles, one single-sample target and no vertex layout or depth");
        ok = false;
    }
    ID3D12PipelineState* pipeline = nullptr;
    if (ok) {
        ID3D12Device2* device2 = nullptr;
        HRESULT hr = device_of(the_device)->QueryInterface(IID_PPV_ARGS(&device2));
        if (SUCCEEDED(hr)) {
            const D3D12_PIPELINE_STATE_STREAM_DESC desktop{bytes, stream};
            hr = device2->CreatePipelineState(&desktop, IID_PPV_ARGS(&pipeline));
            device2->Release();
        }
        if (FAILED(hr)) std::snprintf(why, why_size, "CreatePipelineState failed 0x%08lx", static_cast<unsigned long>(hr));
        if (FAILED(hr)) {
            wchar_t dir[MAX_PATH], path[MAX_PATH];
            if (shader_dump_dir(dir, MAX_PATH)) {
                swprintf(path, MAX_PATH, L"%s\\converted_fail.bin", dir);
                write_file(path, stream, bytes);
            }
        }
        check_device("Device CreatePipelineState", 86);
    }
    if (pipeline && (vertex || mesh) && pixel)
        graphics = {static_cast<ID3D12RootSignature*>(root->own().real), targets.RTFormats[0], mesh};
    return pipeline;
}

// Transient data the title addresses by GPU address (fetch tables, vertices, constants) is copied
// from its memory into a desktop upload buffer; the copy's GPU address is bound instead.
ID3D12Resource* upload_ring;
bool ring_copy(u64 title_address, std::size_t bytes, u64* offset);
D3D12_GPU_VIRTUAL_ADDRESS upload_from_title(u64 title_address, std::size_t bytes)
{
    u64 offset = 0;
    return ring_copy(title_address, bytes, &offset) ? upload_ring->GetGPUVirtualAddress() + offset : 0;
}
unsigned char* reserve_upload_ring(std::size_t bytes, u64* offset)
{
    constexpr u64 ring_bytes = u64{256} << 20;
    ID3D12Resource*& ring = upload_ring;
    static unsigned char* mapped;
    static volatile LONG64 cursor;
    static SRWLOCK lock = SRWLOCK_INIT;
    if (!ring) {
        AcquireSRWLockExclusive(&lock);
        if (!ring) {
            const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = ring_bytes;
            buffer.Height = 1;
            buffer.DepthOrArraySize = 1;
            buffer.MipLevels = 1;
            buffer.SampleDesc.Count = 1;
            buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            ID3D12Resource* created = nullptr;
            void* base = nullptr;
            if (SUCCEEDED(device_of(the_device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&created))) && SUCCEEDED(created->Map(0, nullptr, &base))) {
                mapped = static_cast<unsigned char*>(base);
                ring = created;
            }
        }
        ReleaseSRWLockExclusive(&lock);
    }
    if (!ring || bytes == 0 || bytes > ring_bytes / 4) return nullptr;
    const u64 rounded = (bytes + 511) & ~u64{511};  // 512: texture footprints are placed in the ring too
    u64 start = static_cast<u64>(InterlockedExchangeAdd64(&cursor, static_cast<LONG64>(rounded))) % ring_bytes;
    if (start + rounded > ring_bytes) start = static_cast<u64>(InterlockedExchangeAdd64(&cursor, static_cast<LONG64>(rounded) + (ring_bytes - start))) % ring_bytes, start = 0;
    *offset = start;
    return mapped + start;
}

bool ring_copy(u64 title_address, std::size_t bytes, u64* offset)
{
    auto* dest = reserve_upload_ring(bytes, offset);
    if (!dest) return false;
    std::memcpy(dest, reinterpret_cast<const void*>(title_address), bytes);
    return true;
}

// Number of contiguous readable bytes at address, up to wanted.
std::size_t readable_window(u64 address, std::size_t wanted)
{
    return x2w_local::probe_readable(static_cast<std::uintptr_t>(address), wanted);
}

// Per-list state while recording.
struct Pass {
    bool pipeline_real = false;      // the bound pipeline has a desktop pipeline
    bool compute_pipeline = false;
    bool compute_root = false;
    IndirectGraphicsPipeline indirect_graphics;
    ID3D12RootSignature* graphics_root = nullptr;
    bool targets_bound = false;
    bool topology_ok = false;
    unsigned srv_bound = 0;          // bit per root parameter with a desktop address bound
    const std::vector<RootParameter>* root = nullptr;
    // Heap-indexing flags of the title root signature last set by slot 34. While nonzero, that root signature is
    // deferred: it is not set on the desktop list, root SRVs are not forwarded, and real draws go through decide_real_root.
    UINT root_heap_flags = 0;
    ID3D12RootSignature* deferred_root = nullptr;
    std::vector<std::pair<ID3D12Resource*, D3D12_RESOURCE_STATES>> states;  // resources this list moved out of COMMON
    // What the title bound, kept for draws through translated pipelines:
    Wrapper* pipeline = nullptr;
    UINT topology = 0;
    x2w_startup::ViewTable view_tables[16]{};
    u64 source_view_heap = 0;
    u64 root_address[16] = {};       // title address of each root descriptor
    std::uint32_t constants[16][8] = {};
    D3D12_VERTEX_BUFFER_VIEW vb[8] = {};
    PlacedVertexView placed_vb[8] = {};
    D3D12_INDEX_BUFFER_VIEW ib = {};
    DXGI_FORMAT target_formats[8] = {};
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
    UINT target_count = 0;
    ID3D12Resource* target0 = nullptr;  // desktop resource of render target 0 (for per-draw dumps)
    bool heap_set = false;
    x2w_startup::SamplerSource title_samplers;
    std::unique_ptr<x2w_startup::SamplerSnapshot> sampler_snapshot;
    x2w_startup::ComPtr<ID3D12DescriptorHeap> sampler_heap;
    ID3D12DescriptorHeap* bound_sampler_heap = nullptr;
    // Whether the title set a viewport / scissor on this list; without them a desktop list draws no pixel.
    bool viewport_set = false;
    bool scissor_set = false;
    // Size of the automatic full-target default already forwarded to the desktop list (0 = none yet). Kept apart from
    // the title flags above so a later render target 0 of another size gets a fresh default.
    UINT64 default_viewport_w = 0, default_scissor_w = 0;
    UINT default_viewport_h = 0, default_scissor_h = 0;
};
LONG render_draws, render_draws_skipped, render_pipelines_real, render_pipelines_missing;
// Heap-indexed title root signatures not set on the desktop list, and real draws refused under one, by decision.
LONG heap_roots_deferred, heap_root_refusals[x2w_startup::real_root_decision_count];

ID3D12GraphicsCommandList* recording(Wrapper* self) { return self->own().list_open ? static_cast<ID3D12GraphicsCommandList*>(self->own().real) : nullptr; }
Pass& pass_of(Wrapper* self)
{
    if (!self->own().pass) self->own().pass = new Pass;
    return *static_cast<Pass*>(self->own().pass);
}
// Desktop resources rest in COMMON between lists; a list moves them as it needs and back at Close.
void require_state(Wrapper* self, ID3D12Resource* resource, D3D12_RESOURCE_STATES state)
{
    Pass& pass = pass_of(self);
    auto* list = static_cast<ID3D12GraphicsCommandList*>(self->own().real);
    for (auto& known : pass.states) {
        if (known.first != resource) continue;
        if (known.second != state) {
            transition(list, resource, known.second, state);
            known.second = state;
        }
        return;
    }
    transition(list, resource, D3D12_RESOURCE_STATE_COMMON, state);
    pass.states.emplace_back(resource, state);
}
// State for a draw that samples `resource`: a depth texture also bound as a read-only DSV keeps DEPTH_READ.
D3D12_RESOURCE_STATES sampled_state(Wrapper* self, ID3D12Resource* resource)
{
    for (const auto& known : pass_of(self).states)
        if (known.first == resource && (known.second & D3D12_RESOURCE_STATE_DEPTH_READ))
            return D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
}
void use_as_target(Wrapper* self, ID3D12Resource* resource) { require_state(self, resource, D3D12_RESOURCE_STATE_RENDER_TARGET); }
void release_targets(Wrapper* self)
{
    Pass& pass = pass_of(self);
    for (const auto& known : pass.states)
        transition(static_cast<ID3D12GraphicsCommandList*>(self->own().real), known.first, known.second, D3D12_RESOURCE_STATE_COMMON);
    pass = Pass{};
}

// A reset list starts without a viewport or scissor, whatever the previous recording set.
void reset_viewport_state(Wrapper* self)
{
    if (!self->own().pass) return;
    pass_of(self).viewport_set = false;
    pass_of(self).scissor_set = false;
    pass_of(self).default_viewport_w = pass_of(self).default_scissor_w = 0;
    pass_of(self).default_viewport_h = pass_of(self).default_scissor_h = 0;
}

// List slot 20, IASetPrimitiveTopology. 0x11 is a console-only rectangle list (no desktop draw yet).
void __stdcall list_topology(Wrapper* self, UINT topology)
{
    pass_of(self).topology_ok = topology >= 1 && topology <= 5;
    pass_of(self).topology = topology;
    if (auto* list = recording(self); list && pass_of(self).topology_ok) list->IASetPrimitiveTopology(static_cast<D3D12_PRIMITIVE_TOPOLOGY>(topology));
}
// List slot 25, SetPipelineState.
void __stdcall list_pipeline(Wrapper* self, void* pipeline)
{
    Wrapper* w = as_wrapper(pipeline);
    Pass& pass = pass_of(self);
    const bool available = w && w->own().kind == pipeline_kind && w->own().real;
    pass.indirect_graphics = available ? w->own().indirect_graphics : IndirectGraphicsPipeline{};
    // Keep direct draws on their existing translation path. Slot 61 binds
    // a recognized desktop graphics pipeline just before it records the draw.
    pass.pipeline_real = available && !pass.indirect_graphics.root;
    pass.compute_pipeline = pass.pipeline_real && w->own().compute_pipeline;
    pass.pipeline = w && w->own().kind == pipeline_kind ? w : nullptr;
    InterlockedIncrement(pass.pipeline_real ? &render_pipelines_real : &render_pipelines_missing);
    if (auto* list = recording(self); list && pass.pipeline_real) list->SetPipelineState(static_cast<ID3D12PipelineState*>(w->own().real));
}
// List slot 33 (desktop 29 + 4), SetComputeRootSignature.
void __stdcall list_compute_root(Wrapper* self, void* root)
{
    auto* list = recording(self);
    if (!list) return;
    Wrapper* w = as_wrapper(root);
    Pass& pass = pass_of(self);
    pass.compute_root = w && w->own().kind == root_signature_kind && w->own().real && !root_heap_flags(w);
    if (pass.compute_root) list->SetComputeRootSignature(static_cast<ID3D12RootSignature*>(w->own().real));
}

// The guest method has independent resource and byte-offset arguments.
void __stdcall list_execute_indirect(Wrapper* self, void* signature, UINT count, void* argument, u64 offset,
                                     void* count_buffer, u64 count_offset)
{
    const auto refuse = [](const char* reason) {
        static LONG refusals;
        if (first_calls(refusals, 16)) log("x2w-render ExecuteIndirect refused: %s", reason);
    };
    auto* list = recording(self);
    if (!list) return refuse("list is closed");
    const auto type = list->GetType();
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE)
        return refuse("requires a direct or compute command list");
    if (count != 1 || count_buffer || count_offset) return refuse("requires one command and no count buffer");
    Wrapper* sig = as_wrapper(signature);
    if (!sig || sig->own().kind != command_signature_kind || !sig->own().real)
        return refuse("missing or placeholder signature");
    const Wrapper* device = self->own().list_device;
    if (!device || sig->own().signature_device != device)
        return refuse("requires a supported single rootless signature from this device");
    const auto command = sig->own().indirect_command;
    const bool indexed = command == IndirectCommand::draw_indexed;
    const bool mesh = command == IndirectCommand::dispatch_mesh;
    const bool graphics = indexed || mesh || command == IndirectCommand::draw;
    Pass& pass = pass_of(self);
    u64 bytes = sizeof(D3D12_DISPATCH_ARGUMENTS);
    if (graphics) {
        if (indexed) bytes = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
        else if (mesh) bytes = sizeof(D3D12_DISPATCH_MESH_ARGUMENTS);
        else bytes = sizeof(D3D12_DRAW_ARGUMENTS);
        if (type != D3D12_COMMAND_LIST_TYPE_DIRECT || !pass.indirect_graphics.root ||
            pass.indirect_graphics.root != pass.graphics_root || pass.indirect_graphics.mesh != mesh)
            return refuse("indirect graphics requires a direct list with a matching desktop pipeline and root");
        if (!pass.targets_bound || pass.target_count != 1 || !pass.target0 ||
            pass.target_formats[0] != pass.indirect_graphics.target_format ||
            pass.dsv_format != DXGI_FORMAT_UNKNOWN || !pass.viewport_set || !pass.scissor_set)
            return refuse("indirect graphics requires one matching target, viewport and scissor");
        if (!mesh && pass.topology != D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
            return refuse("indirect draw requires triangle-list topology");
        const auto samples = pass.target0->GetDesc().SampleDesc;
        if (samples.Count != 1 || samples.Quality)
            return refuse("draw requires a single-sample render target");
    } else if (!pass.compute_pipeline || !pass.compute_root) {
        return refuse("requires a desktop compute pipeline and root");
    }
    Wrapper* buffer = as_wrapper(argument);
    if (!buffer || buffer->own().kind != resource_kind || !buffer->own().real)
        return refuse("argument has no desktop resource");
    const auto base = reinterpret_cast<u64>(buffer->own().title_memory);
    if (!base || (offset & 3) || offset > ~u64{} - base)
        return refuse("invalid argument offset or unmapped buffer");
    const auto found = placed_buffer_range(device, base + offset, bytes);
    if (found.reject || found.owner != buffer) return refuse("argument record is outside a unique placed buffer");
    const auto* info = resource_info(buffer);
    if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON)
        return refuse("argument buffer does not use COMMON state tracking");
    auto* resource = static_cast<ID3D12Resource*>(buffer->own().real);
    auto argument_state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    if (indexed) {
        auto view = pass.ib;
        const UINT alignment = view.Format == DXGI_FORMAT_R16_UINT ? 2 : view.Format == DXGI_FORMAT_R32_UINT ? 4 : 0;
        if (!alignment || !view.BufferLocation || !view.SizeInBytes ||
            view.BufferLocation % alignment || view.SizeInBytes % alignment)
            return refuse("invalid index view");
        const auto indices = placed_buffer_range(device, view.BufferLocation, view.SizeInBytes);
        if (indices.reject || !indices.owner || !indices.owner->own().real)
            return refuse("index view is outside a unique placed buffer");
        const auto* index_info = resource_info(indices.owner);
        if (!index_info || index_info->initial_state != D3D12_RESOURCE_STATE_COMMON)
            return refuse("index buffer does not use COMMON state tracking");
        auto* index = static_cast<ID3D12Resource*>(indices.owner->own().real);
        const auto index_base = reinterpret_cast<u64>(indices.owner->own().title_memory);
        const auto index_offset = view.BufferLocation - index_base;
        const auto native_base = index->GetGPUVirtualAddress();
        if (!native_base || index_offset > ~u64{} - native_base || (native_base + index_offset) % alignment)
            return refuse("index view has no aligned desktop address");
        view.BufferLocation = native_base + index_offset;
        if (index == resource) argument_state |= D3D12_RESOURCE_STATE_INDEX_BUFFER;
        else require_state(self, index, D3D12_RESOURCE_STATE_INDEX_BUFFER);
        list->IASetIndexBuffer(&view);
    }
    if (graphics) {
        list->SetPipelineState(static_cast<ID3D12PipelineState*>(pass.pipeline->own().real));
        list->SetGraphicsRootSignature(pass.graphics_root);
    }
    require_state(self, resource, argument_state);
    list->ExecuteIndirect(static_cast<ID3D12CommandSignature*>(sig->own().real), count, resource, offset, nullptr, 0);
}
// List slot 34 (desktop 30 + 4), SetGraphicsRootSignature. The title's parameters are tracked for translated
// draws either way. A root signature that indexes a descriptor heap directly is not set on the desktop list: the
// title's heaps are never forwarded there (slot 32), so the debug layer would reject it (message ID 1419).
void __stdcall list_graphics_root(Wrapper* self, void* root)
{
    Wrapper* w = as_wrapper(root);
    Pass& pass = pass_of(self);
    const auto* params = w ? root_info(w) : nullptr;
    // A different root signature invalidates the title's cached root arguments.
    if (pass.root != params) {
        for (auto& table : pass.view_tables) table = {};
        std::fill_n(pass.root_address, 16, 0);
        std::memset(pass.constants, 0, sizeof(pass.constants));
    }
    pass.root = params;
    pass.srv_bound = 0;
    const bool real = w && w->own().kind == root_signature_kind && w->own().real;
    pass.root_heap_flags = real ? root_heap_flags(w) : 0;
    pass.deferred_root = pass.root_heap_flags ? static_cast<ID3D12RootSignature*>(w->own().real) : nullptr;
    pass.graphics_root = nullptr;
    auto* list = recording(self);
    if (!list || !real) return;
    if (!pass.root_heap_flags) {
        pass.graphics_root = static_cast<ID3D12RootSignature*>(w->own().real);
        list->SetGraphicsRootSignature(pass.graphics_root);
        return;
    }
    const LONG deferred = InterlockedIncrement(&heap_roots_deferred);
    if (deferred <= 3)
        log("x2w-render root signature %p with heap flags 0x%x not set on the desktop list (title heaps are not forwarded)",
            static_cast<void*>(w), pass.root_heap_flags);
}
DXGI_FORMAT view_format_of(u64 desktop_handle, ID3D12Resource* target)
{
    DXGI_FORMAT format = target->GetDesc().Format;
    if (const std::uint32_t id = shadow_id(desktop_handle)) {
        AcquireSRWLockShared(&note_lock);
        DXGI_FORMAT view_format = DXGI_FORMAT_UNKNOWN;
        if (id - 1 < view_records().size() && view_records()[id - 1].desc_bytes >= 4)
            std::memcpy(&view_format, view_records()[id - 1].desc, 4);
        ReleaseSRWLockShared(&note_lock);
        if (view_format != DXGI_FORMAT_UNKNOWN) format = desktop_format(view_format);
    }
    return format;
}
// The view a descriptor holds now, read from its shadow record so copies are followed. A record of the
// other kind is wrong_kind; a handle with no record falls back to the creation-time map (render targets only).
struct ResolvedView {
    ID3D12Resource* resource = nullptr;
    UINT dsv_flags = 0;
    bool wrong_kind = false;
    bool skipped = false;
};
ResolvedView resolve_view(u64 handle, int slot)
{
    ResolvedView out;
    ViewRecord record{};
    bool known = false;
    bool materialized = false;
    if (const std::uint32_t id = shadow_id(handle)) {
        AcquireSRWLockShared(&note_lock);
        known = id - 1 < view_records().size();
        if (known) {
            record = view_records()[id - 1];
            materialized = id - 1 < view_materialized().size() && view_materialized()[id - 1] != 0;
        }
        ReleaseSRWLockShared(&note_lock);
    }
    if (!known) {
        if (slot == 20) out.resource = view_target(handle, slot);
        return out;
    }
    if (static_cast<int>(record.slot) != slot) {
        out.wrong_kind = true;
        return out;
    }
    if (!materialized) {
        out.skipped = true;
        return out;
    }
    out.resource = record.resource ? reinterpret_cast<ID3D12Resource*>(unwrap(record.resource)) : nullptr;
    if (slot == 21 && record.desc_bytes >= 12) std::memcpy(&out.dsv_flags, record.desc + 8, 4);
    return out;
}
bool format_has_stencil(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_D24_UNORM_S8_UINT || format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ||
           format == DXGI_FORMAT_R24G8_TYPELESS || format == DXGI_FORMAT_R32G8X24_TYPELESS;
}
// List slot 52 (desktop 48 + 4), ClearRenderTargetView(view, colour, rect count, rects). Desktop
// resources rest in COMMON here, so the target is moved to RENDER_TARGET around the clear. The view is
// read from its shadow record, as OMSetRenderTargets does, so a descriptor the title copied clears the
// resource it now holds; a handle with no record falls back to the creation-time map.
void __stdcall list_clear_target(Wrapper* self, u64 handle, const float* colour, UINT rect_count, const D3D12_RECT* rects)
{
    if (!self->own().list_open || !colour) return;
    const u64 desktop = desktop_cpu_handle(handle);
    const ResolvedView view = resolve_view(desktop, 20);
    ID3D12Resource* target = view.resource;
    static LONG logged;
    if (first_calls(logged, 4))
        log("x2w-render ClearRenderTargetView view=0x%llx target=%p colour=%.3f %.3f %.3f %.3f rects=%u", handle,
            static_cast<void*>(target), colour[0], colour[1], colour[2], colour[3], rect_count);
    if (!target) {
        InterlockedIncrement(&render_target_unresolved);
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearRenderTargetView skipped: view 0x%llx is %s", handle,
                view.wrong_kind ? "not a render target view"
                                : view.skipped ? "a skipped render target view" : "not a known render target view");
        return;
    }
    auto* list = static_cast<ID3D12GraphicsCommandList*>(self->own().real);
    use_as_target(self, target);
    list->ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(desktop)}, colour, rect_count,
                                rect_count ? rects : nullptr);
    InterlockedIncrement(&render_clears);
}
static_assert(x2w_clear_depth::clear_depth == D3D12_CLEAR_FLAG_DEPTH && x2w_clear_depth::clear_stencil == D3D12_CLEAR_FLAG_STENCIL,
              "the clear flag bits of list_clear_depth.h are the SDK values");
// List slot 51 (desktop 47 + 4), ClearDepthStencilView(view, flags, depth, stencil, rect count, rects). DERIVED, not
// seen in a title log: slots 50 and 52 around it are observed at desktop + 4. The depth value is the fourth argument,
// so it arrives in xmm3 and the generic noop_slot entry would lose it; the table points at this function directly
// (see where the list table is filled) and it keeps the per-slot counts noop_slot would have kept.
// The view is read from its shadow record, as OMSetRenderTargets does. Desktop resources rest in COMMON, so the
// resource is moved to DEPTH_WRITE first. A view that holds nothing is counted with the unresolved handles of
// OMSetRenderTargets; a clear the desktop list would reject (read-only plane, depth outside [0, 1], unreadable
// rectangles) is refused and counted in x2w_d3d12_depth_clear_stats. See startup_stubs/list_clear_depth.h.
void __stdcall list_clear_depth(Wrapper* self, u64 handle, UINT flags, float depth, UINT stencil, UINT rect_count,
                                const D3D12_RECT* rects)
{
    InterlockedIncrement64(&list_slot_calls[51]);
    InterlockedIncrement64(&list_slot_census[51]);
    InterlockedIncrement(&draw_calls_seen);
    auto* list = recording(self);
    if (!list) return;
    const u64 desktop = desktop_cpu_handle(handle);
    const ResolvedView view = desktop ? resolve_view(desktop, 21) : ResolvedView{};
    ID3D12Resource* target = view.resource;
    static LONG logged;
    if (first_calls(logged, 4))
        log("x2w-render ClearDepthStencilView view=0x%llx target=%p flags=0x%x depth=%.3f stencil=0x%x rects=%u", handle,
            static_cast<void*>(target), flags, depth, stencil & 0xffu, rect_count);
    if (!target) {
        InterlockedIncrement(&render_target_unresolved);
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: view 0x%llx is %s", handle,
                view.wrong_kind ? "not a depth stencil view"
                                : view.skipped ? "a skipped depth stencil view" : "not a known depth stencil view");
        return;
    }
    x2w_clear_depth::Request request;
    request.flags = flags;
    request.depth = depth;
    request.format_has_stencil = format_has_stencil(view_format_of(desktop, target));
    request.depth_read_only = (view.dsv_flags & D3D12_DSV_FLAG_READ_ONLY_DEPTH) != 0;
    request.stencil_read_only = (view.dsv_flags & D3D12_DSV_FLAG_READ_ONLY_STENCIL) != 0;
    request.rect_count = rect_count;
    request.rects_readable = !rect_count || (rects && x2w_local::readable(rects, std::size_t{rect_count} * sizeof(D3D12_RECT)));
    const x2w_clear_depth::Plan plan = x2w_clear_depth::plan(request);
    if (plan.verdict == x2w_clear_depth::Verdict::nothing_to_clear) return;
    if (plan.verdict != x2w_clear_depth::Verdict::forward) {
        InterlockedIncrement(&render_depth_clears_refused);
        static LONG refused_by_rule;
        if (first_calls(refused_by_rule, 6))
            log("x2w-d3d12 ClearDepthStencilView refused: %s (flags 0x%x, depth %.3f)", x2w_clear_depth::reason(plan.verdict), flags, depth);
        return;
    }
    require_state(self, target, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    list->ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(desktop)},
                                static_cast<D3D12_CLEAR_FLAGS>(plan.flags), depth, static_cast<UINT8>(stencil), plan.rect_count,
                                plan.rect_count ? rects : nullptr);
    InterlockedIncrement(&render_depth_clears);
    InterlockedIncrement(&render_clears);
}
// List slot 50 (desktop 46 + 4), OMSetRenderTargets(count, handles, single range, depth).
// With single_range the title passes the first handle only and the rest follow it at the title's own
// descriptor stride (also for one target). A descriptor of the wrong kind, a handle that holds no view, a
// DSV with only one plane read-only (needs per-plane states), or a depth view that resolves to nothing
// refuses the whole bind: nothing is set and draws are skipped. Each refusal is logged and counted.
void __stdcall list_targets(Wrapper* self, UINT count, const D3D12_CPU_DESCRIPTOR_HANDLE* handles, BOOL single_range,
                            const D3D12_CPU_DESCRIPTOR_HANDLE* depth)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    pass.targets_bound = false;
    pass.target_count = 0;
    pass.target0 = nullptr;
    pass.dsv_format = DXGI_FORMAT_UNKNOWN;
    if (!list) return;
    static LONG refused, unresolved_logged;
    if (count > 8) {
        InterlockedIncrement(&render_target_unresolved);
        if (first_calls(refused, 6)) log("x2w-d3d12 OMSetRenderTargets refused: %u render targets (at most 8)", count);
        return;
    }
    const UINT listed = single_range ? (count ? 1 : 0) : count;
    if (listed && !x2w_local::readable(handles, listed * sizeof(*handles))) return;
    UINT stride = 0;
    if (single_range && count) {
        const UINT increment = the_device ? device_of(the_device)->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) : 0;
        stride = increment ? title_descriptor_stride(handles[0].ptr, increment) : 0;
        if (!stride) {
            InterlockedIncrement(&render_target_unresolved);
            return;
        }
    }
    D3D12_CPU_DESCRIPTOR_HANDLE desktop[8];
    ID3D12Resource* targets[8]{};
    DXGI_FORMAT formats[8]{};
    for (UINT i = 0; i < count; ++i) {
        const u64 step = u64{i} * stride;
        if (single_range && handles[0].ptr > (~u64{0}) - step) {
            InterlockedIncrement(&render_target_unresolved);
            if (first_calls(unresolved_logged, 6))
                log("x2w-d3d12 OMSetRenderTargets refused: render target %u handle overflow (0x%llx + %llu)", i,
                    static_cast<unsigned long long>(handles[0].ptr), static_cast<unsigned long long>(step));
            return;
        }
        const u64 title = single_range ? handles[0].ptr + step : handles[i].ptr;
        desktop[i].ptr = static_cast<SIZE_T>(desktop_cpu_handle(title));
        const ResolvedView view = resolve_view(desktop[i].ptr, 20);
        if (view.wrong_kind && first_calls(refused, 6)) log("x2w-d3d12 OMSetRenderTargets refused: render target %u is not a render target view", i);
        targets[i] = view.resource;
        if (!targets[i]) {
            InterlockedIncrement(&render_target_unresolved);
            if (!view.wrong_kind && first_calls(unresolved_logged, 6))
                log("x2w-d3d12 OMSetRenderTargets refused: render target %u (0x%llx) %s", i, static_cast<unsigned long long>(title),
                    view.skipped ? "holds a skipped view" : "holds no view");
            return;
        }
        formats[i] = view_format_of(desktop[i].ptr, targets[i]);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE desktop_dsv{};
    const D3D12_CPU_DESCRIPTOR_HANDLE* dsv_ptr = nullptr;
    ID3D12Resource* dsv_target = nullptr;
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
    D3D12_RESOURCE_STATES dsv_state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    if (depth && x2w_local::readable(depth, sizeof(*depth))) {
        const u64 resolved = desktop_cpu_handle(depth->ptr);
        if (!resolved) {
            // The handle maps to no desktop descriptor: no depth is bound although the pipeline may expect one.
            InterlockedIncrement(&render_target_unresolved);
            if (first_calls(unresolved_logged, 6))
                log("x2w-d3d12 OMSetRenderTargets: depth handle 0x%llx maps to no descriptor; binding no depth", static_cast<unsigned long long>(depth->ptr));
        } else {
            const ResolvedView view = resolve_view(resolved, 21);
            dsv_target = view.resource;
            if (!dsv_target) {
                InterlockedIncrement(&render_target_unresolved);
                if (first_calls(refused, 6))
                    log("x2w-d3d12 OMSetRenderTargets refused: depth handle is %s",
                        view.wrong_kind ? "not a depth stencil view"
                                        : view.skipped ? "a skipped depth stencil view" : "not a known depth stencil view");
                return;
            }
            dsv_format = view_format_of(resolved, dsv_target);
            const bool read_depth = (view.dsv_flags & D3D12_DSV_FLAG_READ_ONLY_DEPTH) != 0;
            // A format without a stencil plane has no second plane to disagree with.
            const bool read_stencil = !format_has_stencil(dsv_format) ? read_depth : (view.dsv_flags & D3D12_DSV_FLAG_READ_ONLY_STENCIL) != 0;
            if (read_depth != read_stencil) {
                if (first_calls(refused, 6)) log("x2w-d3d12 OMSetRenderTargets refused: depth stencil view with one read-only plane (flags 0x%x)", view.dsv_flags);
                return;
            }
            if (read_depth) dsv_state = D3D12_RESOURCE_STATE_DEPTH_READ;
            desktop_dsv.ptr = static_cast<SIZE_T>(resolved);
            dsv_ptr = &desktop_dsv;
        }
    }
    if (!count && !dsv_ptr) return;
    for (UINT i = 0; i < count; ++i) {
        use_as_target(self, targets[i]);
        pass.target_formats[i] = formats[i];
        if (i == 0) pass.target0 = targets[i];
    }
    if (dsv_target) require_state(self, dsv_target, dsv_state);
    pass.target_count = count;
    pass.dsv_format = dsv_format;
    list->OMSetRenderTargets(count, count ? desktop : nullptr, FALSE, dsv_ptr);
    pass.targets_bound = true;
    InterlockedIncrement(&render_target_binds);
}
// List slots 21 / 22 / 23, RSSetViewports / RSSetScissorRects / OMSetBlendFactor.
void __stdcall list_viewports(Wrapper* self, UINT count, const D3D12_VIEWPORT* viewports)
{
    // Only a viewport that is forwarded counts as set; a call with no viewports leaves the state as it was.
    if (auto* list = recording(self); list && count && count <= 16 && x2w_local::readable(viewports, count * sizeof(*viewports))) {
        list->RSSetViewports(count, viewports);
        pass_of(self).viewport_set = true;
    }
}
void __stdcall list_scissors(Wrapper* self, UINT count, const D3D12_RECT* rects)
{
    if (auto* list = recording(self); list && count && count <= 16 && x2w_local::readable(rects, count * sizeof(*rects))) {
        list->RSSetScissorRects(count, rects);
        pass_of(self).scissor_set = true;
    }
}
void __stdcall list_blend_factor(Wrapper* self, const float* factor)
{
    if (auto* list = recording(self); list && x2w_local::readable(factor, 16)) list->OMSetBlendFactor(factor);
}
// List slot 24, OMSetStencilRef(reference): list state that the stencil test of the next draws compares against.
void __stdcall list_stencil_ref(Wrapper* self, UINT reference)
{
    if (auto* list = recording(self)) list->OMSetStencilRef(reference);
}
// List slot 44 (desktop 40 + 4), SetGraphicsRootShaderResourceView(parameter, title GPU address).
void __stdcall list_root_srv(Wrapper* self, UINT parameter, u64 address)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    if (parameter < 32) pass.srv_bound &= ~(1u << parameter);
    if (parameter < 16) pass.root_address[parameter] = address;
    // Under a deferred heap-indexed root the desktop list holds another root signature; only the title address is kept.
    if (!list || !pass.root || pass.root_heap_flags || parameter >= pass.root->size() || parameter >= 32 ||
        (*pass.root)[parameter].type != D3D12_ROOT_PARAMETER_TYPE_SRV)
        return;
    const std::size_t bytes = readable_window(address, 0x10000);
    const D3D12_GPU_VIRTUAL_ADDRESS desktop = bytes ? upload_from_title(address, bytes) : 0;
    static LONG logged;
    if (first_calls(logged, 4)) {
        char head[3 * 32 + 1];
        hex_text(reinterpret_cast<const void*>(address), 32, head, sizeof(head));
        log("x2w-render root SRV %u at title 0x%llx: %zu bytes -> desktop 0x%llx [%s]", parameter, address, bytes, desktop, head);
    }
    if (!desktop) return;
    list->SetGraphicsRootShaderResourceView(parameter, desktop);
    pass.srv_bound |= 1u << parameter;
}
// List slot 42 (desktop 38 + 4), SetGraphicsRootConstantBufferView(parameter, title GPU address).
// A translated shader reads a root CBV through the same pointer source as a root SRV (root_address). Nothing
// is forwarded to the desktop list and srv_bound is left alone: bit 0 means a vertex-fetch SRV is bound.
void __stdcall list_root_cbv(Wrapper* self, UINT parameter, u64 address)
{
    auto* list = recording(self);
    if (!list) return;
    Pass& pass = pass_of(self);
    if (!pass.root || parameter >= pass.root->size() || parameter >= 16) return;
    const D3D12_ROOT_PARAMETER_TYPE type = (*pass.root)[parameter].type;
    if (type != D3D12_ROOT_PARAMETER_TYPE_CBV) {
        static LONG refused;
        if (first_calls(refused, 1))
            log("x2w-render root CBV %u ignored: the root signature declares parameter type %d", parameter, static_cast<int>(type));
        return;
    }
    pass.root_address[parameter] = (address && readable_window(address, 1)) ? address : 0;
}
extern LONG translated_draws, untranslated_draws, frame_number;
bool draw_translated(Wrapper* self, bool indexed, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance);
// List slot 12, DrawInstanced.
void __stdcall list_draw(Wrapper* self, UINT vertices, UINT instances, UINT first_vertex, UINT first_instance)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    if (list && !pass.pipeline_real) {
        InterlockedIncrement(draw_translated(self, false, vertices, instances, first_vertex, 0, first_instance) ? &translated_draws
                                                                                                                : &untranslated_draws);
        return;
    }
    if (list && pass.root_heap_flags) {
        // The desktop list holds no copy of the title's heaps, and the root constants and descriptor tables the
        // title set are kept only as title values, so the deferred root signature cannot be set with its state.
        x2w_startup::RealRootDraw request;
        request.root_flags = pass.root_heap_flags;
        request.state_restorable = false;
        const x2w_startup::RealRootDecision decision = x2w_startup::decide_real_root(request);
        if (decision != x2w_startup::RealRootDecision::bind) {
            const LONG n = InterlockedIncrement(&heap_root_refusals[static_cast<int>(decision)]);
            InterlockedIncrement(&render_draws_skipped);
            if (n <= 3 || n % 1000 == 0)
                log("x2w-render real draw refused: root signature heap flags 0x%x, %s (%ld so far)", pass.root_heap_flags,
                    x2w_startup::real_root_reason(decision), n);
            return;
        }
        // Not reached with the facts above: a change that mirrors the title heaps and restores the root state on the
        // list, with an end-to-end fixture, makes the deferred bind here.
        list->SetGraphicsRootSignature(pass.deferred_root);
    }
    // The translated vertex shaders read their fetch table through root parameter 0.
    if (list && pass.pipeline_real && pass.targets_bound && pass.topology_ok && pass.topology != 0x11 && (pass.srv_bound & 1)) {
        list->DrawInstanced(vertices, instances, first_vertex, first_instance);
        const LONG n = InterlockedIncrement(&render_draws);
        if (n <= 3) log("x2w-render DrawInstanced(%u, %u, %u, %u) recorded", vertices, instances, first_vertex, first_instance);
    } else {
        InterlockedIncrement(&render_draws_skipped);
    }
}

extern SRWLOCK written_lock;
struct TextureWrite {
    u64 generation = 0;
    UINT bc7_width = 0, bc7_height = 0;
};
std::unordered_map<u64, TextureWrite>& written_textures();
ID3D12Resource* sampled_address_texture(Wrapper* resource);
#include "startup_stubs/d3d12_draw.inl"

// Queue slot 10, desktop ExecuteCommandLists(count, lists).
void __stdcall queue_execute(Wrapper* self, UINT count, void* const* lists)
{
    static LONG calls;
    if (render_mode()) {
        // A closed list stays submittable until its next Reset, as on the desktop.
        std::vector<ID3D12CommandList*> ready;
        std::vector<Wrapper*> submitted;
        // Pin the closed recording and its allocator through submission bookkeeping.
        AcquireSRWLockExclusive(&draw_lock);
        AcquireSRWLockExclusive(&allocator_lock);
        for (UINT i = 0; i < count; ++i) {
            Wrapper* w = as_wrapper(lists[i]);
            if (w && w->own().kind == list_kind && w->own().list_ready) {
                ready.push_back(static_cast<ID3D12CommandList*>(w->own().real));
                submitted.push_back(w);
            }
        }
        if (!ready.empty()) {
            auto* queue = static_cast<ID3D12CommandQueue*>(self->own().real);
            const UINT ready_count = static_cast<UINT>(ready.size());
            if (draw_dump_frame()) {
                const HRESULT hr = draw_dumps().execute(queue, ready_count, ready.data());
                if (FAILED(hr)) log("x2w-draw readback fence failed 0x%08lx", static_cast<unsigned long>(hr));
            } else {
                queue->ExecuteCommandLists(ready_count, ready.data());
            }
            retire_draw_sampler_heaps(queue, ready);
            note_submits(queue, submitted);
            InterlockedIncrement(&render_executes);
        }
        ReleaseSRWLockExclusive(&allocator_lock);
        ReleaseSRWLockExclusive(&draw_lock);
        check_device("CommandQueue execute", 35);
        return;
    }
    if (list_noop()) {
        if (first_calls(calls)) log("x2w-d3d12 CommandQueue::ExecuteCommandLists count=%u -> not submitted (noop lists)", count);
        return;
    }
    std::vector<ID3D12CommandList*> real(count);
    for (UINT i = 0; i < count; ++i) real[i] = reinterpret_cast<ID3D12CommandList*>(unwrap(reinterpret_cast<u64>(lists[i])));
    static_cast<ID3D12CommandQueue*>(self->own().real)->ExecuteCommandLists(count, real.data());
    if (first_calls(calls)) log("x2w-d3d12 CommandQueue::ExecuteCommandLists count=%u", count);
}

// Fence slot 10, desktop Signal(value) (also used by the DirectStorage stand-in's EnqueueSignal).
HRESULT __stdcall fence_signal(Wrapper* self, UINT64 value)
{
    const HRESULT hr = static_cast<ID3D12Fence*>(self->own().real)->Signal(value);
    fence_reached(self, value);
    return hr;
}
// Fence slot 8, desktop GetCompletedValue().
UINT64 __stdcall fence_completed_value(Wrapper* self) { return static_cast<UINT64>(self->own().completed); }
// Fence slot 12 (Xbox order): (value, ptr, flag) -> address of the completed value, which the title
// reads and waits on directly; all ones there means device removed.
volatile LONG64* __stdcall fence_completed_address(Wrapper* self, u64 value, u64 r8, u64 r9)
{
    static LONG calls;
    if (first_calls(calls, 2))
        log("x2w-d3d12 Fence::slot12(completed value address) value=0x%llx r8=0x%llx r9=0x%llx -> %p (now 0x%llx)", value, r8,
            r9, static_cast<void*>(const_cast<LONG64*>(&self->own().completed)), static_cast<unsigned long long>(self->own().completed));
    return &self->own().completed;
}

// Queue slot 18 (Xbox order): (UINT64* frequency) -> HRESULT; the title divides by the value.
HRESULT __stdcall queue_timestamp_frequency(Wrapper* self, UINT64* frequency)
{
    const HRESULT hr = static_cast<ID3D12CommandQueue*>(self->own().real)->GetTimestampFrequency(frequency);
    static LONG calls;
    if (first_calls(calls, 2))
        log("x2w-d3d12 CommandQueue::slot18(GetTimestampFrequency) -> 0x%08lx %llu", static_cast<unsigned long>(hr),
            frequency ? *frequency : 0ull);
    return hr;
}

// Queue slots whose Xbox meaning is unknown, dispatched by argument shape:
// - (fence wrapper, value): Signal on the desktop queue. Slot 17 (Wait) has the same shape and is
//   handled by queue_wait below; on other slots a wait would complete the fence early.
// - (count, array of command-list wrappers): ExecuteCommandLists (not submitted while lists are no-ops).
// Anything else logs its first call and returns 0. Each shape logs the slot it arrived on.
template <int N>
__declspec(noinline) u64 queue_dynamic(Wrapper* self, u64 rdx, u64 r8, u64 r9)
{
    Timed timed(time_queue);
    static LONG calls;
    const bool log_it = first_calls(calls, 2);
    if (Wrapper* fence = as_wrapper(reinterpret_cast<void*>(rdx)); fence && fence->own().kind == fence_kind) {
        const HRESULT hr = static_cast<ID3D12CommandQueue*>(self->own().real)->Signal(static_cast<ID3D12Fence*>(fence->own().real), r8);
        if (SUCCEEDED(hr)) {
            if (render_mode() || InterlockedCompareExchange(&self->own().queue_wait_seen, 0, 0))
                complete_when_reached(fence, r8);
            else
                fence_reached(fence, r8);
        }
        if (log_it)
            log("x2w-d3d12 CommandQueue::slot%d(fence=%p, value=0x%llx) -> Signal 0x%08lx", N, static_cast<void*>(fence), r8,
                static_cast<unsigned long>(hr));
        return static_cast<u64>(hr);
    }
    if (rdx >= 1 && rdx <= 64 && r8 >= 0x10000 && x2w_local::readable(reinterpret_cast<void*>(r8), rdx * sizeof(void*))) {
        auto* const* lists = reinterpret_cast<void* const*>(r8);
        bool all_lists = true;
        for (u64 i = 0; i < rdx; ++i) {
            Wrapper* w = as_wrapper(lists[i]);
            all_lists &= w && w->own().kind == list_kind;
        }
        if (all_lists) {
            if (log_it) log("x2w-d3d12 CommandQueue::slot%d(count=%llu, lists) -> ExecuteCommandLists", N, rdx);
            queue_execute(self, static_cast<UINT>(rdx), lists);
            return 0;
        }
    }
    if (create_shaped(rdx, r8, r9, 0, 0, 0, 0)) {
        char name[64];
        std::snprintf(name, sizeof(name), "CommandQueue::slot%d(passes an IID)", N);
        x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, reinterpret_cast<u64>(self), rdx, r8, r9,
                       _ReturnAddress());
    }
    if (log_it) {
        char name[64];
        std::snprintf(name, sizeof(name), "CommandQueue::slot%d", N);
        x2w_trap::record("x2w-noop", X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, N & 0xFF, reinterpret_cast<u64>(self), rdx, r8,
                         r9, _ReturnAddress());
    }
    return 0;
}

// Queue slot 17 (Xbox order): (fence, value) -> HRESULT, Wait. The console queue has two methods
// before the desktop's Signal and Wait (desktop slots 14 and 15), so Signal is slot 16 and Wait is
// slot 17; slot 18 (GetTimestampFrequency) and slot 32 (present) follow the same layout. The desktop
// queue waits on the GPU for the fence value. Any other argument shape keeps the shape dispatch.
u64 __stdcall queue_wait(Wrapper* self, u64 rdx, u64 r8, u64 r9)
{
    Wrapper* fence = as_wrapper(reinterpret_cast<void*>(rdx));
    if (!fence || fence->own().kind != fence_kind) return queue_dynamic<17>(self, rdx, r8, r9);
    InterlockedExchange(&self->own().queue_wait_seen, 1);
    const HRESULT hr = static_cast<ID3D12CommandQueue*>(self->own().real)->Wait(static_cast<ID3D12Fence*>(fence->own().real), r8);
    static LONG calls;
    if (first_calls(calls, 2))
        log("x2w-d3d12 CommandQueue::slot17(fence=%p, value=0x%llx) -> Wait 0x%08lx", static_cast<void*>(fence), r8,
            static_cast<unsigned long>(hr));
    return static_cast<u64>(hr);
}

template <std::size_t... I>
void fill_queue_dynamic(void** table, std::index_sequence<I...>)
{
    ((I >= 8 ? (table[I] = reinterpret_cast<void*>(&queue_dynamic<static_cast<int>(I)>)) : nullptr), ...);
}

// Queue GetDesc(), returned by value (desktop slot 18; Xbox slot unknown, not installed).
void* __stdcall queue_get_desc(void* a, void* b)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "CommandQueue::GetDesc(no receiver)");
    *static_cast<D3D12_COMMAND_QUEUE_DESC*>(result) = static_cast<ID3D12CommandQueue*>(self->own().real)->GetDesc();
    return result;
}

// Resource slot 8, desktop Map(subresource, readRange, data). On the console a resource's memory is
// title-visible: for a resource placed at a title address or backed by a hook allocation, the mapped
// pointer is that memory (subresource offsets inside it are not modeled). Others map on desktop.
HRESULT __stdcall resource_map(Wrapper* self, UINT subresource, const D3D12_RANGE* range, void** data)
{
    HRESULT hr = S_OK;
    void* mapped = self->own().title_memory;
    if (!mapped) hr = static_cast<ID3D12Resource*>(self->own().real)->Map(subresource, range, &mapped);
    if (data) *data = SUCCEEDED(hr) ? mapped : nullptr;
    if (FAILED(hr)) check_device("Resource::Map", 8);
    static LONG calls, failures;
    if (first_calls(calls, 4) || (FAILED(hr) && first_calls(failures, 8)))
        log("x2w-d3d12 Resource::Map subresource=%u title_memory=%p -> 0x%08lx %p", subresource, self->own().title_memory,
            static_cast<unsigned long>(hr), mapped);
    return hr;
}
void __stdcall resource_unmap(Wrapper* self, UINT subresource, const D3D12_RANGE* range)
{
    if (!self->own().title_memory) static_cast<ID3D12Resource*>(self->own().real)->Unmap(subresource, range);
}

// Resource slot 11, desktop GetGPUVirtualAddress() (called with no arguments). On the console a
// resource placed at an address has that address as its GPU address, so the title gets its own
// address back; resources without title memory report the desktop address (0 for textures).
UINT64 __stdcall resource_gpu_address(Wrapper* self)
{
    const UINT64 desktop = static_cast<ID3D12Resource*>(self->own().real)->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
        ? static_cast<ID3D12Resource*>(self->own().real)->GetGPUVirtualAddress() : 0;
    const UINT64 result = self->own().title_memory ? reinterpret_cast<UINT64>(self->own().title_memory) : desktop;
    static LONG calls;
    if (first_calls(calls, 4))
        log("x2w-d3d12 Resource::GetGPUVirtualAddress title_memory=%p desktop=0x%llx -> 0x%llx", self->own().title_memory, desktop,
            result);
    return result;
}

// Resource slot 10, desktop GetDesc(), returned by value.
void* __stdcall resource_get_desc(void* a, void* b)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "Resource::GetDesc(no receiver)");
    *static_cast<D3D12_RESOURCE_DESC*>(result) = static_cast<ID3D12Resource*>(self->own().real)->GetDesc();
    return result;
}

// Heap slot 8, desktop GetDesc(), returned by value.
void* __stdcall heap_get_desc(void* a, void* b)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "Heap::GetDesc(no receiver)");
    *static_cast<D3D12_HEAP_DESC*>(result) = static_cast<ID3D12Heap*>(self->own().real)->GetDesc();
    return result;
}

// Descriptor heap slots 8-10, desktop GetDesc / GetCPU/GPUDescriptorHandleForHeapStart, by value.
void* __stdcall descriptor_heap_get_desc(void* a, void* b)
{
    void* result = nullptr;
    Wrapper* self = receiver(a, b, &result, "DescriptorHeap::GetDesc(no receiver)");
    *static_cast<D3D12_DESCRIPTOR_HEAP_DESC*>(result) = static_cast<ID3D12DescriptorHeap*>(self->own().real)->GetDesc();
    return result;
}
// The title returns structures of at most 8 bytes in rax (this in rcx), larger ones through a hidden
// pointer passed first. If the second argument is a wrapper, the hidden-pointer form is used instead.
template <class T, class Get>
std::uint64_t small_result(void* a, void* b, const char* method, Get get)
{
    if (Wrapper* w = as_wrapper(b)) {
        const T value = get(w);
        *static_cast<T*>(a) = value;
        return reinterpret_cast<std::uint64_t>(a);
    }
    Wrapper* w = as_wrapper(a);
    if (!w) fail(method, a, b);
    const T value = get(w);
    std::uint64_t bits = 0;
    static_assert(sizeof(T) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}
std::uint64_t __stdcall descriptor_heap_cpu_start(void* a, void* b)
{
    return small_result<D3D12_CPU_DESCRIPTOR_HANDLE>(a, b, "DescriptorHeap::GetCPUDescriptorHandleForHeapStart(no receiver)",
        [](Wrapper* w) {
            if (w->own().title_cpu) return D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(w->own().title_cpu)};
            return static_cast<ID3D12DescriptorHeap*>(w->own().real)->GetCPUDescriptorHandleForHeapStart();
        });
}

// Device slot 14, desktop CreateDescriptorHeap(desc, riid, ppv). The title's descriptor is longer:
// +0x10 a title CPU address for the descriptors (or null) and +0x18 their size in bytes. With an
// address, GetCPUDescriptorHandleForHeapStart must return it (the title asserts this).
HRESULT __stdcall device_create_descriptor_heap(Wrapper* self, const D3D12_DESCRIPTOR_HEAP_DESC* desc, const GUID* iid,
                                                void** out)
{
    Timed timed(time_heap);
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    const auto* extra = reinterpret_cast<const std::uint64_t*>(desc + 1);
    ID3D12DescriptorHeap* heap = nullptr;
    HRESULT hr = kind_for(*iid) == descriptor_heap_kind
        ? device_of(self)->CreateDescriptorHeap(desc, IID_PPV_ARGS(&heap)) : E_NOINTERFACE;
    // A console view heap can be larger than the driver accepts. Measure the accepted maximum, create
    // the desktop heap at it, and keep the title's full-size descriptor memory. Desktop view writes
    // and copies past the physical prefix use a companion CPU-only heap.
    // The refusal code differs between drivers (E_INVALIDARG on a GTX 1660 SUPER, E_OUTOFMEMORY on an
    // RTX 3060). Preserve errors unrelated to capacity, including unsupported interface IDs.
    D3D12_DESCRIPTOR_HEAP_DESC clamped = *desc;
    bool did_clamp = false;
    if ((hr == E_INVALIDARG || hr == E_OUTOFMEMORY) && (desc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
        && desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        && desc->NumDescriptors > view_heap_probe_floor) {
        const std::uint64_t limit = accepted_view_heap_descriptors(device_of(self), desc->NodeMask);
        if (limit && limit < desc->NumDescriptors) {
            clamped.NumDescriptors = static_cast<UINT>(limit);
            did_clamp = true;
            hr = device_of(self)->CreateDescriptorHeap(&clamped, IID_PPV_ARGS(&heap));
            static LONG shortcuts;
            if (SUCCEEDED(hr) && first_calls(shortcuts, 1))
                log("x2w-d3d12 Device::CreateDescriptorHeap: shortcut - title asked for %u shader-visible "
                    "view descriptors, this driver accepts %u; the rest stay in title memory",
                    desc->NumDescriptors, clamped.NumDescriptors);
        }
    }
    const D3D12_DESCRIPTOR_HEAP_DESC* request = did_clamp ? &clamped : desc;
    const std::uint64_t desktop = heap ? heap->GetCPUDescriptorHandleForHeapStart().ptr : 0;
    static LONG calls;
    if (first_calls(calls, 16))
        log("x2w-d3d12 Device::CreateDescriptorHeap type=%d count=%u flags=0x%x node=0x%x title=0x%llx bytes=0x%llx riid=%s "
            "-> 0x%08lx desktop=0x%llx", desc->Type, desc->NumDescriptors, desc->Flags, desc->NodeMask, extra[0], extra[1],
            text, static_cast<unsigned long>(hr), desktop);
    if (FAILED(hr)) return hr;
    x2w_startup::ComPtr<ID3D12DescriptorHeap> tail;
    if (did_clamp) {
        D3D12_DESCRIPTOR_HEAP_DESC tail_desc = *desc;
        tail_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        tail_desc.NumDescriptors -= clamped.NumDescriptors;
        hr = device_of(self)->CreateDescriptorHeap(&tail_desc, IID_PPV_ARGS(&tail));
        // Retain the physical CPU tail for the lifetime of the original desktop heap.
        static constexpr GUID tail_lifetime{0x1aa68383,0xf1e9,0x4264,{0xa7,0x05,0x1f,0xf9,0x69,0xa8,0x03,0xac}};
        if (SUCCEEDED(hr)) hr = heap->SetPrivateDataInterface(tail_lifetime, tail.Get());
        if (FAILED(hr)) { heap->Release(); return hr; }
    }
    const UINT desktop_stride = device_of(self)->GetDescriptorHandleIncrementSize(desc->Type);
    const bool raw_samplers = desc->Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER &&
        extra[1] == u64{desc->NumDescriptors} * x2w_startup::title_sampler_stride;
    const UINT title_stride = raw_samplers ? x2w_startup::title_sampler_stride : desktop_stride;
    const u64 bytes = u64{desc->NumDescriptors} * title_stride;
    // Some callers pass the 16-byte desktop descriptor; the words after it are then unrelated stack
    // data. The extension is taken only when it holds a pointer and exactly the heap's size.
    std::uint64_t title_memory = extra[0] >= 0x10000 && extra[1] == bytes ? extra[0] : 0;
    // The title copies descriptors between heaps with its own memory copies. A view heap without
    // title memory therefore gets plain memory of ours as its CPU side, so that the view ids written
    // there (note_view) travel with those copies; the desktop heap behind it is still filled. A heap
    // of another type gets the same when its desktop start is not an address (title_heap_memory.h).
    if (!x2w_startup::ensure_title_heap_memory(desc->Type, desktop, bytes, title_memory, [](u64 size) {
        return reinterpret_cast<u64>(VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    })) {
        heap->Release();
        return E_OUTOFMEMORY;
    }
    Wrapper* w = wrap(heap, descriptor_heap_kind);
    if (title_memory) {
        w->own().title_cpu = title_memory;
        AcquireSRWLockExclusive(&descriptor_lock);
        descriptor_ranges.push_back({title_memory, bytes, desktop, title_stride, desktop_stride,
                                     static_cast<std::uint64_t>(request->NumDescriptors),
                                     tail ? tail->GetCPUDescriptorHandleForHeapStart().ptr : 0});
        ReleaseSRWLockExclusive(&descriptor_lock);
    }
    const u64 sampler_cpu = raw_samplers ? title_memory : 0;
    note_heap(w, desc->Type, request->NumDescriptors, desc->NumDescriptors, desktop_stride,
              title_memory, desktop,
              (desc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) ? heap->GetGPUDescriptorHandleForHeapStart().ptr : 0,
              sampler_cpu);
    *out = w;
    return S_OK;
}

// Device view slots: forwarded with the destination CPU handle (argument D, 0-based after the
// object) translated from title to desktop.
// Desktop removes the device (DXGI_ERROR_INVALID_CALL) on view calls it rejects, so they are checked
// first and skipped with a log: a view type the desktop resource's flags do not allow (the console
// makes unordered-access views of depth textures), and constant-buffer views, whose BufferLocation is
// a title GPU address. Console format values in the view description are mapped like resource formats.
template <int N, int D>
__declspec(noinline) u64 view_slot(Wrapper* self, u64 a, u64 b, u64 c, u64 d)
{
    Timed timed(time_view);
    u64 args[4] = {unwrap(a), unwrap(b), unwrap(c), unwrap(d)};
    const u64 title_dest = args[D];
    args[D] = desktop_cpu_handle(args[D]);
    if (!args[D]) return 0;
    const auto record_view = [&](bool materialized) {
        const u64 raw[4] = {a, b, c, d};
        const int described = N == 19 ? 2 : (N == 17 || N == 22) ? 0 : 1;
        const std::size_t described_bytes = N == 17 ? 16 : N == 22 ? 52 : N == 18 ? 40 : N == 19 ? 40 : N == 20 ? 20 : 16;
        note_view(N, (N >= 18 && N <= 21) ? raw[0] : 0, N == 19 ? raw[1] : 0, reinterpret_cast<const void*>(raw[described]),
                  described_bytes, title_dest, args[D], materialized);
    };
    const char* skip = nullptr;
    if (N == 17) skip = "constant buffer view (title GPU address)";
    if (N >= 18 && N <= 21 && args[0]) {
        const UINT flags = reinterpret_cast<ID3D12Resource*>(args[0])->GetDesc().Flags;
        if (N == 18 && (flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) skip = "resource denies shader resource views";
        if (N == 19 && !(flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) skip = "resource lacks unordered access";
        if (N == 20 && !(flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) skip = "resource lacks render target";
        if (N == 21 && !(flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) skip = "resource lacks depth stencil";
    }
    if (skip) {
        record_view(false);
        static LONG skipped;
        if (first_calls(skipped, 6)) log("x2w-d3d12 Device::slot%d view skipped: %s", N, skip);
        return 0;
    }
    // The view descriptions of slots 18-21 start with a DXGI_FORMAT.
    unsigned char desc_copy[64];
    const int desc_index = N == 19 ? 2 : 1;
    if (N >= 18 && N <= 21 && args[desc_index] && x2w_local::readable(reinterpret_cast<const void*>(args[desc_index]), sizeof(desc_copy))) {
        std::memcpy(desc_copy, reinterpret_cast<const void*>(args[desc_index]), sizeof(desc_copy));
        DXGI_FORMAT format;
        std::memcpy(&format, desc_copy, sizeof(format));
        DXGI_FORMAT mapped = desktop_format(format);
        if (args[0] && x2w_startup::packed_target_format(mapped)) {
            // A packed view follows a stand-in texture and is refused on any other texture of another format.
            auto* const resource = reinterpret_cast<ID3D12Resource*>(args[0]);
            const auto verdict = x2w_startup::packed_view(mapped, resource->GetDesc().Format, x2w_startup::packed_stand_in_of(resource));
            if (verdict == x2w_startup::PackedView::refused) {
                record_view(false);
                static LONG refused;
                if (first_calls(refused, 6))
                    log("x2w-d3d12 Device::slot%d view skipped: packed format %d on a texture of format %d that does not stand in for it",
                        N, mapped, resource->GetDesc().Format);
                return 0;
            }
            if (verdict == x2w_startup::PackedView::stand_in) mapped = DXGI_FORMAT_R8G8B8A8_UNORM;
        }
        if (mapped != format) {
            std::memcpy(desc_copy, &mapped, sizeof(mapped));
            args[desc_index] = reinterpret_cast<u64>(desc_copy);
        }
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC null_uav{};
    D3D12_RENDER_TARGET_VIEW_DESC null_rtv{};
    if constexpr (N == 19) {
        if (c && x2w_local::readable(reinterpret_cast<const void*>(c), sizeof(null_uav)) &&
            x2w_local::adapt_null_view(device_of(self), a != 0, b != 0,
                *reinterpret_cast<const D3D12_UNORDERED_ACCESS_VIEW_DESC*>(c), null_uav))
            args[2] = reinterpret_cast<u64>(&null_uav);
    }
    if constexpr (N == 20) {
        if (b && x2w_local::readable(reinterpret_cast<const void*>(b), sizeof(null_rtv)) &&
            x2w_local::adapt_null_view(device_of(self), a != 0,
                *reinterpret_cast<const D3D12_RENDER_TARGET_VIEW_DESC*>(b), null_rtv))
            args[1] = reinterpret_cast<u64>(&null_rtv);
    }
    const u64 result = method_of(self->own().real, N)(self->own().real, args[0], args[1], args[2], args[3], 0, 0, 0, 0);
    record_view(true);
    if ((N == 20 || N == 21) && args[0]) note_view_target(args[D], reinterpret_cast<ID3D12Resource*>(args[0]), N);
    check_device("Device view", N);
    static LONG calls;
    if (first_calls(calls))
        log("x2w-d3d12 Device::slot%d view(0x%llx, 0x%llx, 0x%llx, 0x%llx) dest 0x%llx -> 0x%llx", N, a, b, c, d,
            D == 0 ? a : D == 1 ? b : D == 2 ? c : d, args[D]);
    return result;
}

// Copy through actual desktop heaps. A logical range may straddle the last
// shader-visible descriptor and the first CPU-only descriptor.
void copy_descriptor_run(Wrapper* self, UINT count, u64 dest, u64 src, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    const UINT increment = device_of(self)->GetDescriptorHandleIncrementSize(type);
    const UINT dest_stride = title_descriptor_stride(dest, increment);
    const UINT src_stride = title_descriptor_stride(src, increment);
    while (count) {
        UINT dest_room, src_room, dest_total, src_total;
        const auto d = desktop_cpu_handle(dest, &dest_room, &dest_total);
        const auto s = desktop_cpu_handle(src, &src_room, &src_total);
        if (count > dest_total || count > src_total || !d || !s) return;
        const UINT chunk = (std::min)(count, (std::min)(dest_room, src_room));
        device_of(self)->CopyDescriptorsSimple(chunk, {static_cast<SIZE_T>(d)}, {static_cast<SIZE_T>(s)}, type);
        note_copy(dest, src, d, s, chunk, increment);
        dest += u64{chunk} * dest_stride;
        src += u64{chunk} * src_stride;
        count -= chunk;
    }
}

// Device slot 24, desktop CopyDescriptorsSimple(count, dest, src, type).
void __stdcall device_copy_descriptors_simple(Wrapper* self, UINT count, u64 dest, u64 src, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    Timed timed(time_copy);
    copy_descriptor_run(self, count, dest, src, type);
    check_device("Device CopyDescriptorsSimple", 24);
    static LONG calls;
    if (first_calls(calls)) log("x2w-d3d12 Device::CopyDescriptorsSimple count=%u dest=0x%llx src=0x%llx type=%d", count, dest, src, type);
}

// Device slot 23, desktop CopyDescriptors(destCount, destStarts, destSizes, srcCount, srcStarts, srcSizes, type).
void __stdcall device_copy_descriptors(Wrapper* self, UINT dest_count, const D3D12_CPU_DESCRIPTOR_HANDLE* dest_starts,
                                       const UINT* dest_sizes, UINT src_count, const D3D12_CPU_DESCRIPTOR_HANDLE* src_starts,
                                       const UINT* src_sizes, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    // Validate each complete logical range before the first desktop copy.
    const auto valid = [](UINT count, const D3D12_CPU_DESCRIPTOR_HANDLE* starts, const UINT* sizes, u64& total) {
        for (UINT i = 0; i < count; ++i) {
            const UINT size = sizes ? sizes[i] : 1;
            UINT available;
            desktop_cpu_handle(starts[i].ptr, nullptr, &available);
            if (size > available) return false;
            total += size;
        }
        return true;
    };
    u64 dest_total = 0, src_total = 0;
    if (!valid(dest_count, dest_starts, dest_sizes, dest_total) ||
        !valid(src_count, src_starts, src_sizes, src_total) || dest_total != src_total) return;
    const UINT increment = device_of(self)->GetDescriptorHandleIncrementSize(type);
    UINT di = 0, doff = 0, si = 0, soff = 0;
    while (di < dest_count && si < src_count) {
        const UINT dsize = dest_sizes ? dest_sizes[di] : 1, ssize = src_sizes ? src_sizes[si] : 1;
        const UINT run = (std::min)(dsize - doff, ssize - soff);
        if (run) copy_descriptor_run(self, run,
            dest_starts[di].ptr + u64{doff} * title_descriptor_stride(dest_starts[di].ptr, increment),
            src_starts[si].ptr + u64{soff} * title_descriptor_stride(src_starts[si].ptr, increment), type);
        doff += run; soff += run;
        if (doff == dsize) { ++di; doff = 0; }
        if (soff == ssize) { ++si; soff = 0; }
    }
    check_device("Device CopyDescriptors", 23);
    static LONG calls;
    if (first_calls(calls)) log("x2w-d3d12 Device::CopyDescriptors dest=%u src=%u type=%d", dest_count, src_count, type);
}
std::uint64_t __stdcall descriptor_heap_gpu_start(void* a, void* b)
{
    return small_result<D3D12_GPU_DESCRIPTOR_HANDLE>(a, b, "DescriptorHeap::GetGPUDescriptorHandleForHeapStart(no receiver)",
        [](Wrapper* w) { return static_cast<ID3D12DescriptorHeap*>(w->own().real)->GetGPUDescriptorHandleForHeapStart(); });
}
std::uint64_t __stdcall device_adapter_luid(void* a, void* b)
{
    return small_result<LUID>(a, b, "Device::GetAdapterLuid(no receiver)",
                              [](Wrapper* w) { return device_of(w)->GetAdapterLuid(); });
}

// Device slot 73 (Xbox order): (desc*, allocator, pso, riid, ppv) with a 16-byte descriptor
// (observed type 0/2 at +0, then 2, 1, 0) and an Xbox-only list IID. Creates a closed desktop
// list of the descriptor's type on the allocator so object methods have a real target; its Xbox methods are traps,
// or no-ops with X2W_D3D12_LIST_NOOP=1.
HRESULT __stdcall device_create_command_list_x(Wrapper* self, const void* desc, void* allocator, void* pso,
                                               const GUID* iid, void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    char bytes[3 * 16 + 1];
    hex_text(desc, 16, bytes, sizeof(bytes));
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    // Create on the allocator the title's allocator currently stands for: after a Reset that moved
    // it to another desktop allocator, the first one may still be in use.
    auto* desktop_allocator = reinterpret_cast<ID3D12CommandAllocator*>(unwrap(reinterpret_cast<u64>(allocator)));
    if (Wrapper* a = as_wrapper(allocator); a && a->own().kind == allocator_kind) {
        AcquireSRWLockExclusive(&allocator_lock);
        desktop_allocator = title_allocator(a).current->real;
        ReleaseSRWLockExclusive(&allocator_lock);
    }
    HRESULT hr = kind_for(*iid) == list_kind
        ? device_of(self)->CreateCommandList(0, desc ? *static_cast<const D3D12_COMMAND_LIST_TYPE*>(desc)
                                                     : D3D12_COMMAND_LIST_TYPE_DIRECT,
              desktop_allocator,
              reinterpret_cast<ID3D12PipelineState*>(unwrap(reinterpret_cast<u64>(pso))), IID_PPV_ARGS(&list))
        : E_NOINTERFACE;
    if (SUCCEEDED(hr)) {
        hr = list->Close();
        if (FAILED(hr)) list->Release();
    }
    static LONG calls;
    if (first_calls(calls, 4))
        log("x2w-d3d12 Device::slot73(CreateCommandList?) desc=[%s] allocator=%p pso=%p riid=%s -> 0x%08lx", bytes,
            allocator, pso, text, static_cast<unsigned long>(hr));
    if (FAILED(hr)) return hr;
    Wrapper* child = wrap(list, list_kind);
    child->own().list_device = self;
    *out = child;
    return S_OK;
}

// Device slots 78 and 79 (Xbox order): frame pacing setup. 78 receives (0, 33333, 3, 0), a 30 Hz
// frame length in microseconds and a period; 79 receives (0, offset, 0, 0). The title tolerates
// failure. Recorded and accepted; nothing paces frames yet.
UINT frame_interval_us, frame_period;
HRESULT __stdcall device_frame_interval(Wrapper*, void* output, UINT length_us, UINT period, UINT flags)
{
    frame_interval_us = length_us;
    frame_period = period;
    log("x2w-d3d12 Device::slot78(frame interval?) output=%p length=%u us period=%u flags=0x%x -> S_OK", output, length_us,
        period, flags);
    return S_OK;
}
UINT frame_event_offset_us[4];
HRESULT __stdcall device_frame_event(Wrapper*, UINT type, UINT offset_us, void* reserved, UINT flags)
{
    if (type < 4) frame_event_offset_us[type] = offset_us;
    log("x2w-d3d12 Device::slot79(frame event?) type=%u offset=%u us reserved=%p flags=0x%x -> S_OK", type, offset_us,
        reserved, flags);
    return S_OK;
}

// Device slot 133 (Xbox order): (address, D3D12_RESOURCE_DESC*, state, clear, riid, ppv), a resource
// placed at a title address. First seen with a row-major buffer desc (flags 0x20004) over a
// placeholder-replaced XMem region, state 0xffffffff, a null clear value and the ID3D12Resource IID.
// Desktop: the allocation containing the address is opened as an existing heap (cached) and a
// cross-adapter buffer is placed at the address's offset. Its desktop GPU address differs from the
// title address; nothing translates title GPU addresses yet. Textures trap.
std::mutex placed_mutex;
std::vector<std::pair<void*, ID3D12Heap*>> existing_heaps;

HRESULT __stdcall device_place_resource_x(Wrapper* self, void* address, const D3D12_RESOURCE_DESC* desc, UINT state,
                                          const void* clear, const GUID* iid, void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    check_device("Device before slot", 133);
    if (kind_for(*iid) != resource_kind) return E_NOINTERFACE;
    if (desc->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
        // Textures cannot be placed on title memory on desktop: a committed default-heap texture
        // stands in, not aliased with the title address. The clear value is dropped if refused.
        D3D12_RESOURCE_DESC texture = sanitized(*desc);
        if (x2w_startup::supports_bc7_4k_s(*desc) && x2w_local::env_on("X2W_XG_BC7_4K_S_LAYOUT")) {
            const x2w_startup::Bc16Surface surface(*desc);
            // Desktop BC7 allocations require whole blocks at the top mip.
            texture.Width = surface.columns * x2w_startup::Bc16Surface::block_texels;
            texture.Height = surface.rows * x2w_startup::Bc16Surface::block_texels;
        }
        const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
        const auto* clear_value = static_cast<const D3D12_CLEAR_VALUE*>(clear);
        ID3D12Resource* created = nullptr;
        HRESULT texture_hr = device_of(self)->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &texture,
                                                                      D3D12_RESOURCE_STATE_COMMON, clear_value,
                                                                      IID_PPV_ARGS(&created));
        if (FAILED(texture_hr) && clear_value)
            texture_hr = device_of(self)->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &texture,
                                                                  D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&created));
        static LONG textures, failures;
        if (first_calls(textures, 6) || (FAILED(texture_hr) && first_calls(failures, 8))) {
            char bytes[3 * 0x38 + 1];
            hex_text(desc, 0x38, bytes, sizeof(bytes));
            log("x2w-d3d12 Device::slot133(CreatePlacedResourceX?) texture address=%p dim=%d %llux%ux%u mips=%u fmt=%d "
                "samples=%u flags=0x%x layout=0x%x state=0x%x clear=%p -> 0x%08lx not aliased [%s]", address, desc->Dimension,
                desc->Width, desc->Height, desc->DepthOrArraySize, desc->MipLevels, desc->Format, desc->SampleDesc.Count,
                desc->Flags, desc->Layout, state, clear, static_cast<unsigned long>(texture_hr),
                FAILED(texture_hr) ? bytes : "");
        }
        if (FAILED(texture_hr)) return texture_hr;
        x2w_startup::mark_packed_stand_in(created, desktop_format(desc->Format));
        Wrapper* w = wrap(created, resource_kind);
        w->own().title_memory = address;
        note_resource(w, desc, 0);
        *out = w;
        return S_OK;
    }
    MEMORY_BASIC_INFORMATION info{};
    if (!address || VirtualQuery(address, &info, sizeof(info)) != sizeof(info) || info.State != MEM_COMMIT) {
        log("x2w-d3d12 Device::slot133(CreatePlacedResourceX?) address=%p not committed -> E_INVALIDARG", address);
        return E_INVALIDARG;
    }
    void* base = info.AllocationBase;
    const auto initial_state = state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE ||
        state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS ? static_cast<D3D12_RESOURCE_STATES>(state) : D3D12_RESOURCE_STATE_COMMON;
    if (initial_state != D3D12_RESOURCE_STATE_COMMON && !(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
        return E_INVALIDARG;
    const UINT64 offset = static_cast<char*>(address) - static_cast<char*>(base);
    ID3D12Heap* heap = nullptr;
    HRESULT hr = S_OK;
    // Desktop only accepts 64 KiB-aligned offsets, a private (VirtualAlloc) allocation, and a buffer
    // that lies inside that one allocation. Attempts that cannot succeed are skipped: each failure
    // throws inside D3D12Core.
    MEMORY_BASIC_INFORMATION region{};
    SIZE_T allocation_bytes = 0;
    for (auto* cursor = static_cast<char*>(base);
         VirtualQuery(cursor, &region, sizeof(region)) && region.AllocationBase == base && region.State == MEM_COMMIT;
         cursor += region.RegionSize)
        allocation_bytes += region.RegionSize;
    if ((offset & 0xFFFF) || offset > allocation_bytes || desc->Width > allocation_bytes - offset || info.Type != MEM_PRIVATE ||
        initial_state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE) hr = E_INVALIDARG;
    if (SUCCEEDED(hr)) {
        std::lock_guard lock(placed_mutex);
        for (auto& [known, h] : existing_heaps)
            if (known == base) heap = h;
        if (!heap) {
            ID3D12Device3* device3 = nullptr;
            hr = device_of(self)->QueryInterface(IID_PPV_ARGS(&device3));
            if (SUCCEEDED(hr)) {
                hr = device3->OpenExistingHeapFromAddress(base, IID_PPV_ARGS(&heap));
                device3->Release();
            }
            if (SUCCEEDED(hr)) existing_heaps.emplace_back(base, heap);
        }
    }
    D3D12_RESOURCE_DESC copy = *desc;
    copy.Flags = static_cast<D3D12_RESOURCE_FLAGS>((sanitized(*desc).Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) |
                                                   D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER);
    ID3D12Resource* resource = nullptr;
    if (SUCCEEDED(hr))
        hr = device_of(self)->CreatePlacedResource(heap, offset, &copy, initial_state, nullptr,
                                                   IID_PPV_ARGS(&resource));
    // A buffer that spans several allocations (adjacent placeholder pieces) cannot be placed on one
    // existing heap. Fallback: a committed default-heap buffer, not aliased with the title memory.
    const bool aliased = SUCCEEDED(hr);
    const char* placement = "aliased";
    if (FAILED(hr)) {
        const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
        copy.Flags = static_cast<D3D12_RESOURCE_FLAGS>(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        hr = device_of(self)->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &copy, initial_state,
                                                      nullptr, IID_PPV_ARGS(&resource));
        placement = "not aliased";
    }
    log("x2w-d3d12 Device::slot133(CreatePlacedResourceX?) address=%p base=%p offset=0x%llx width=0x%llx flags=0x%x "
        "state=0x%x clear=%p riid=%s -> 0x%08lx %s gpuva=0x%llx", address, base, offset, desc->Width, desc->Flags, state,
        clear, text, static_cast<unsigned long>(hr), placement, resource ? resource->GetGPUVirtualAddress() : 0ull);
    if (FAILED(hr)) return hr;
    Wrapper* w = wrap(resource, resource_kind);
    w->own().title_memory = address;
    const ResourceInfo record{*desc, desc->Width, aliased ? Placement::aliased_buffer : Placement::copied_buffer, self, initial_state};
    auto& group = placement_group(desc->Width);
    hr = x2w_startup::publish_placement(note_lock, resource_infos, group.by_start, group.max_width,
        static_cast<const Wrapper*>(w), reinterpret_cast<u64>(address), record);
    if (FAILED(hr)) {
        // The caller has never received this wrapper; release its desktop resource and storage.
        AcquireSRWLockExclusive(&registry_lock);
        registry().erase(w);
        ReleaseSRWLockExclusive(&registry_lock);
        resource->Release();
        _aligned_free(&w->own());
        return hr;
    }
    *out = w;
    return S_OK;
}

#include "d3d12_memory_srv.inl"

// The title exports __XMemAllocationHooks__ = {alloc(size, attributes), free(ptr, attributes), ...}.
// Its alloc routine selects a hook chain by allocatorId (attribute bits 16-23). Around the creation
// of its two display render targets the title pushes a bump allocator for allocatorId 0x80 that
// serves pageSize 2 (bits 14-15) from a 64 MiB block, then asserts that the block was used up:
// the console runtime evidently takes committed-resource memory from XMemAlloc with allocatorId
// 0x80. Hypothesis used here: attributes = pageSize (2 from 4 MiB up, else 1) << 14 | 0x80 << 16 |
// log2(alignment) << 24, memory type 0. The block is bookkeeping only; the desktop resource has its
// own memory and does not alias it.
using XMemAllocHook = void*(__stdcall*)(SIZE_T, ULONGLONG);
XMemAllocHook title_alloc_hook()
{
    static const XMemAllocHook hook = [] {
        auto* const* table = reinterpret_cast<void* const*>(GetProcAddress(GetModuleHandleW(nullptr), "__XMemAllocationHooks__"));
        return table ? reinterpret_cast<XMemAllocHook>(table[0]) : nullptr;
    }();
    return hook;
}

void* title_resource_memory(UINT64 size, UINT64 alignment)
{
    const XMemAllocHook hook = title_alloc_hook();
    if (!hook || !size) return nullptr;
    unsigned log2_alignment = 0;
    while ((UINT64{1} << (log2_alignment + 1)) <= alignment && log2_alignment < 31) ++log2_alignment;
    const ULONGLONG page = size >= (UINT64{4} << 20) ? 2 : 1;
    const ULONGLONG attributes = (page << 14) | (ULONGLONG{0x80} << 16) | (ULONGLONG{log2_alignment} << 24);
    void* memory = hook(static_cast<SIZE_T>(size), attributes);
    static LONG calls;
    if (first_calls(calls, 12))
        log("x2w-d3d12 title XMem hook alloc size=0x%llx attributes=0x%llx -> %p", size, attributes, memory);
    return memory;
}

// X2W_D3D12_DUMP_PSO=1: the first pipeline descriptions of each creation slot are written to the
// log: the description itself and, for every qword in it that points at readable memory, the first
// bytes there (shader containers start with "DXBC"). Diagnostics for the translation work.
void dump_pipeline(const char* what, int slot, const void* desc, std::size_t bytes)
{
    static const bool enabled = [] {
        char value[4]{};
        return GetEnvironmentVariableA("X2W_D3D12_DUMP_PSO", value, sizeof(value)) && value[0] == '1';
    }();
    static LONG dumped[512];
    if (!enabled || !desc || InterlockedIncrement(&dumped[slot & 511]) > 4) return;
    if (!x2w_local::readable(desc, bytes)) return;
    const auto* words = static_cast<const u64*>(desc);
    log("x2w-pso %s slot %d desc %p, %llu bytes", what, slot, desc, static_cast<unsigned long long>(bytes));
    for (std::size_t i = 0; i < bytes / 8; i += 4)
        log("x2w-pso   +0x%03llx: %016llx %016llx %016llx %016llx", static_cast<unsigned long long>(i * 8), words[i],
            i + 1 < bytes / 8 ? words[i + 1] : 0, i + 2 < bytes / 8 ? words[i + 2] : 0, i + 3 < bytes / 8 ? words[i + 3] : 0);
    for (std::size_t i = 0; i < bytes / 8; ++i) {
        const auto* target = reinterpret_cast<const unsigned char*>(words[i]);
        if (words[i] < 0x10000 || as_wrapper(target) || !x2w_local::readable(target, 64)) continue;
        char hex[3 * 48 + 1], text[49];
        hex_text(target, 48, hex, sizeof(hex));
        for (int c = 0; c < 48; ++c) text[c] = target[c] >= 0x20 && target[c] < 0x7f ? static_cast<char>(target[c]) : '.';
        text[48] = 0;
        const u64 next = i + 1 < bytes / 8 ? words[i + 1] : 0;
        log("x2w-pso   [+0x%03llx] -> %p (next word 0x%llx) %s |%s|", static_cast<unsigned long long>(i * 8),
            static_cast<const void*>(target), next, hex, text);
    }
}

// X2W_SHADER_DUMP_DIR=<dir>: every DXBC container a pipeline description points at is written to
// <dir>/<digest>.dxbc (digest = the 16 bytes after the "DXBC" tag), and each stream description to
// <dir>/stream_<n>.bin with a .txt listing which stream offsets point at which container. These feed
// the offline retarget/sign step (tools/dxil_retarget.py); translated results are loaded back by digest.
bool shader_dump_dir(wchar_t* out, DWORD size)
{
    const DWORD n = GetEnvironmentVariableW(L"X2W_SHADER_DUMP_DIR", out, size);
    return n && n < size;
}

void write_file(const wchar_t* path, const void* data, std::size_t bytes)
{
    HANDLE file = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    WriteFile(file, data, static_cast<DWORD>(bytes), &written, nullptr);
    CloseHandle(file);
}

bool is_container(const void* p)
{
    return p && x2w_local::readable(p, 32) && std::memcmp(p, "DXBC", 4) == 0;
}

void digest_text(const void* container, wchar_t* out)
{
    const auto* b = static_cast<const unsigned char*>(container) + 4;
    for (int i = 0; i < 16; ++i) swprintf(out + 2 * i, 3, L"%02x", b[i]);
}

void dump_containers(const char* what, const void* desc, std::size_t bytes)
{
    wchar_t dir[MAX_PATH];
    if (!desc || !shader_dump_dir(dir, MAX_PATH) || !x2w_local::readable(desc, bytes)) return;
    static LONG streams;
    const LONG index = InterlockedIncrement(&streams);
    char listing[4096];
    int used = std::snprintf(listing, sizeof(listing), "%s description, %llu bytes\n", what,
                             static_cast<unsigned long long>(bytes));
    const auto* words = static_cast<const u64*>(desc);
    for (std::size_t i = 0; i < bytes / 8; ++i) {
        const void* target = reinterpret_cast<const void*>(words[i]);
        if (words[i] < 0x10000 || !is_container(target)) continue;
        const std::uint32_t total = *reinterpret_cast<const std::uint32_t*>(static_cast<const char*>(target) + 24);
        if (total < 32 || total > (64u << 20) || !x2w_local::readable(target, total)) continue;
        wchar_t digest[33], path[MAX_PATH];
        digest_text(target, digest);
        swprintf(path, MAX_PATH, L"%s\\%s.dxbc", dir, digest);
        if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) write_file(path, target, total);
        if (used < static_cast<int>(sizeof(listing)) - 96)
            used += std::snprintf(listing + used, sizeof(listing) - used, "+0x%03llx %ls %u\n",
                                  static_cast<unsigned long long>(i * 8), digest, total);
    }
    wchar_t path[MAX_PATH];
    swprintf(path, MAX_PATH, L"%s\\%s_%04ld.bin", dir, what[0] == 's' ? L"stream" : L"compute", index);
    write_file(path, desc, bytes);
    swprintf(path, MAX_PATH, L"%s\\%s_%04ld.txt", dir, what[0] == 's' ? L"stream" : L"compute", index);
    write_file(path, listing, static_cast<std::size_t>(used));
}

// Placeholder objects (no desktop object behind the wrapper): ID3D12Object methods succeed and do nothing.
HRESULT __stdcall placeholder_ok(Wrapper*, u64, u64, u64) { return S_OK; }

// Device slot 62 (Xbox order): (D3D12_COMPUTE_PIPELINE_STATE_DESC*, 0, 0, riid, ppv); first seen from
// XboxBvhBuilder.dll, followed by SetName on the result. The bytecode is console GPU code, not DXIL,
// so no desktop pipeline is built: the result is a placeholder PipelineState until shader translation
// exists. Logs the root signature, bytecode address, length and first 32 bytes.
HRESULT __stdcall device_create_compute_pipeline_x(Wrapper*, const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, u64 extra0,
                                                   u64 extra1, const GUID* iid, void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    dump_pipeline("compute", 62, desc, sizeof(*desc));
    dump_containers("compute", desc, sizeof(*desc));
    char bytes[3 * 32 + 1];
    hex_text(desc->CS.pShaderBytecode, desc->CS.BytecodeLength < 32 ? desc->CS.BytecodeLength : 32, bytes, sizeof(bytes));
    static LONG calls;
    if (first_calls(calls, 24))
        log("x2w-d3d12 Device::slot62(CreateComputePipelineState?) root=%p cs=%p len=0x%llx node=0x%x flags=0x%x extra=0x%llx,0x%llx "
            "riid=%s -> placeholder [%s]", static_cast<void*>(desc->pRootSignature), desc->CS.pShaderBytecode,
            static_cast<unsigned long long>(desc->CS.BytecodeLength), desc->NodeMask, desc->Flags, extra0, extra1, text, bytes);
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    *out = wrap(nullptr, pipeline_kind);
    note_pipeline(static_cast<Wrapper*>(*out), 62, nullptr, desc, sizeof(*desc));
    return S_OK;
}

// Device slot 86 (Xbox order): (D3D12_PIPELINE_STATE_STREAM_DESC*, riid, ppv), the desktop
// CreatePipelineState shape (stream of 0x2b0 bytes first seen, ID3D12PipelineState IID). The stream
// carries console shaders; the result is a placeholder PipelineState.
HRESULT __stdcall device_create_pipeline_stream_x(Wrapper*, const D3D12_PIPELINE_STATE_STREAM_DESC* desc, const GUID* iid,
                                                  void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !desc || !iid) return E_POINTER;
    *out = nullptr;
    dump_pipeline("stream", 86, desc->pPipelineStateSubobjectStream, desc->SizeInBytes < 0x400 ? desc->SizeInBytes : 0x400);
    dump_containers("stream", desc->pPipelineStateSubobjectStream, desc->SizeInBytes < 0x1000 ? desc->SizeInBytes : 0x1000);
    static LONG calls;
    if (first_calls(calls, 8))
        log("x2w-d3d12 Device::slot86(CreatePipelineState?) stream=%p bytes=0x%llx riid=%s",
            desc->pPipelineStateSubobjectStream, static_cast<unsigned long long>(desc->SizeInBytes), text);
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    ID3D12PipelineState* real_pipeline = nullptr;
    Translation* translation = nullptr;
    D3D12_BLEND_DESC blend{};
    bool has_blend = false;
    D3D12_DEPTH_STENCIL_DESC depth_stencil{};
    bool has_depth_stencil = false;
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
    bool compute = false;
    IndirectGraphicsPipeline graphics;
    if (render_mode()) {
        char why[256] = "";
        // Recognized desktop stages must reach the native path before the
        // guest cache creates a pending translation for any DXBC container.
        real_pipeline = build_stream_pipeline(desc, why, sizeof(why), compute, graphics);
        if (!real_pipeline || graphics.root)
            translation = stream_translation(*desc, blend, has_blend, depth_stencil, has_depth_stencil, dsv_format);
        static LONG built;
        if ((real_pipeline || !translation) && first_calls(built, 24))
            log("x2w-render stream pipeline -> %s%s", real_pipeline ? "REAL desktop pipeline" : "placeholder: ", real_pipeline ? "" : why);
    }
    *out = wrap(real_pipeline, pipeline_kind);
    static_cast<Wrapper*>(*out)->own().compute_pipeline = real_pipeline && compute;
    static_cast<Wrapper*>(*out)->own().indirect_graphics = graphics;
    note_pipeline(static_cast<Wrapper*>(*out), 86, nullptr, desc->pPipelineStateSubobjectStream, desc->SizeInBytes < 0x2000 ? desc->SizeInBytes : 0x2000,
                  translation, translation && has_blend ? &blend : nullptr,
                  translation && has_depth_stencil ? &depth_stencil : nullptr,
                  translation ? dsv_format : DXGI_FORMAT_UNKNOWN);
    return S_OK;
}

// Device slots 57 and 67 (Xbox order): (desc*, riid, ppv) with the ID3D12PipelineState IID; pipeline
// descriptions whose layout is not decoded yet. The result is a placeholder PipelineState.
template <int N>
HRESULT __stdcall device_create_pipeline_placeholder(Wrapper*, const void* desc, const GUID* iid, void** out)
{
    Timed timed(time_pipeline);
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    dump_pipeline("pipeline", N, desc, 0x300);
    static LONG calls;
    if (first_calls(calls, 2)) {
        char bytes[3 * 96 + 1];
        hex_text(desc, 96, bytes, sizeof(bytes));
        log("x2w-d3d12 Device::slot%d(pipeline?) desc=[%s] riid=%s -> placeholder", N, bytes, text);
    }
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    *out = wrap(nullptr, pipeline_kind);
    note_pipeline(static_cast<Wrapper*>(*out), N, nullptr, desc, 0x300);
    return S_OK;
}

// CreateDerivedPipelineStateX: base, number of fixed-size state patches, patches, IID, result.
HRESULT __stdcall device_create_derived_pipeline(Wrapper*, void* base, UINT count, const void* desc, const GUID* iid,
                                                 void** out)
{
    Timed timed(time_pipeline);
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    if (count > 64) return E_INVALIDARG;
    const std::size_t bytes_in_patches = count * x2w_derived_blend::record_size;
    if (bytes_in_patches && (!desc || !x2w_local::readable(desc, bytes_in_patches))) return E_INVALIDARG;
    dump_pipeline("derived", 55, desc, bytes_in_patches);
    static LONG calls;
    if (first_calls(calls, 3)) {
        Wrapper* w = as_wrapper(base);
        char bytes[3 * 64 + 1];
        hex_text(desc, count ? 64 : 0, bytes, sizeof(bytes));
        log("x2w-d3d12 Device::slot55(derived pipeline) base=%p (%s) count=%u desc=[%s] riid=%s -> placeholder", base,
            w ? kind_names[w->own().kind] : "not a wrapper", count, bytes, text);
    }
    *out = wrap(nullptr, pipeline_kind);
    note_pipeline(static_cast<Wrapper*>(*out), 55, base, desc, bytes_in_patches);
    return S_OK;
}

// Device slot 127 (Xbox order): ten arguments in the CreateCommittedResource3 order: heap properties,
// heap flags (8, ALLOW_DISPLAY, for the 4K render targets), resource desc, an initial state or
// layout, clear value, then three zero words (protected session, castable count and formats in the
// desktop method), riid and ppv. Desktop: CreateCommittedResource with the sanitized desc in
// COMMON, GENERIC_READ on an upload heap and COPY_DEST on a readback heap (the initial states D3D12
// requires there; the Windows 10 runtime refuses COMMON with E_INVALIDARG). When desktop rejects the
// clear value it is retried without one.
HRESULT __stdcall device_create_committed_x(Wrapper* self, const D3D12_HEAP_PROPERTIES* props, UINT heap_flags,
                                            const D3D12_RESOURCE_DESC* desc, UINT state, const D3D12_CLEAR_VALUE* clear,
                                            void* session, UINT castable_count, const void* castable, const GUID* iid,
                                            void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !props || !desc || !iid) return E_POINTER;
    *out = nullptr;
    const D3D12_RESOURCE_DESC copy = sanitized(*desc);
    const auto flags = static_cast<D3D12_HEAP_FLAGS>(heap_flags);
    const D3D12_RESOURCE_ALLOCATION_INFO info = device_of(self)->GetResourceAllocationInfo(0, 1, &copy);
    check_device("Device before slot", 127);
    void* const title_memory = info.SizeInBytes != ~0ull ? title_resource_memory(info.SizeInBytes, info.Alignment) : nullptr;
    ID3D12Resource* resource = nullptr;
    const D3D12_RESOURCE_STATES initial = props->Type == D3D12_HEAP_TYPE_UPLOAD     ? D3D12_RESOURCE_STATE_GENERIC_READ
                                          : props->Type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                                    : D3D12_RESOURCE_STATE_COMMON;
    HRESULT hr = kind_for(*iid) == resource_kind
        ? device_of(self)->CreateCommittedResource(props, flags, &copy, initial, clear, IID_PPV_ARGS(&resource))
        : E_NOINTERFACE;
    bool cleared = clear != nullptr;
    if (FAILED(hr) && clear && kind_for(*iid) == resource_kind) {
        hr = device_of(self)->CreateCommittedResource(props, flags, &copy, initial, nullptr, IID_PPV_ARGS(&resource));
        cleared = false;
    }
    static LONG calls;
    if (first_calls(calls, 12))
        log("x2w-d3d12 Device::slot127(CreateCommittedResource3?) heap=%d flags=0x%x dim=%d %llux%u fmt=%d rflags=0x%x "
            "state=0x%x clear=%p%s session=%p castable=%u/%p riid=%s -> 0x%08lx", props->Type, heap_flags, desc->Dimension,
            desc->Width, desc->Height, desc->Format, desc->Flags, state, static_cast<const void*>(clear),
            clear && !cleared ? "(dropped)" : "", session, castable_count, castable, text, static_cast<unsigned long>(hr));
    if (FAILED(hr)) return hr;
    x2w_startup::mark_packed_stand_in(resource, desktop_format(desc->Format));
    Wrapper* w = wrap(resource, resource_kind);
    w->own().title_memory = title_memory;
    note_resource(w, desc, info.SizeInBytes != ~0ull ? info.SizeInBytes : 0);
    *out = w;
    return S_OK;
}

// Device slot 80 (Xbox order): (type, timeout, 0, flags, &token) -> HRESULT, the frame-event wait
// that hands out a frame token (called with type 2 / timeout -1 and type 0 / timeout 0, flags 0/2).
// Paces to the interval from slot 78 (33.3 ms by default) and returns increasing tokens.
LONG64 frame_token;
LONG frame_waits[4];
LONG64 frame_wait_ms;
HRESULT __stdcall device_wait_frame_event(Wrapper*, UINT type, UINT timeout, void* reserved, UINT flags, UINT64* token)
{
    Timed timed(time_wait);
    // One timeline: the origin event (type 0) fires once per frame interval; any other event type
    // fires at the current origin plus the offset it was scheduled with, so waiting for it after the
    // origin does not cost another whole interval.
    static LARGE_INTEGER frequency, origin;
    static SRWLOCK lock = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lock);
    if (!frequency.QuadPart) {
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&origin);
    }
    const LONGLONG interval = frequency.QuadPart * (frame_interval_us ? frame_interval_us : 33333) / 1000000;
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    LONGLONG target;
    if (type == 0) {
        target = origin.QuadPart + interval;
        if (target < now.QuadPart) target = now.QuadPart;  // running late: start a new timeline here
        origin.QuadPart = target;
    } else {
        target = origin.QuadPart + frequency.QuadPart * frame_event_offset_us[type < 4 ? type : 3] / 1000000;
    }
    const LONGLONG wait_ticks = target > now.QuadPart ? target - now.QuadPart : 0;
    ReleaseSRWLockExclusive(&lock);
    if (timeout != 0 && wait_ticks > 0) {
        // High-resolution timer where available (Sleep is too coarse for a 33 ms frame).
        static thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* high resolution */, TIMER_ALL_ACCESS);
        LARGE_INTEGER due;
        due.QuadPart = -(wait_ticks * 10000000 / frequency.QuadPart);
        if (timer && due.QuadPart < 0 && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, 1000);
        else Sleep(static_cast<DWORD>(wait_ticks * 1000 / frequency.QuadPart));
    }
    InterlockedIncrement(&frame_waits[type < 4 ? type : 3]);
    InterlockedExchangeAdd64(&frame_wait_ms, timeout != 0 ? wait_ticks * 1000 / frequency.QuadPart : 0);
    const UINT64 value = static_cast<UINT64>(InterlockedIncrement64(&frame_token));
    if (token) *token = value;
    static LONG calls;
    if (first_calls(calls, 4))
        log("x2w-d3d12 Device::slot80(frame event wait) type=%u timeout=0x%x reserved=%p flags=0x%x -> token %llu", type, timeout,
            reserved, flags, value);
    return S_OK;
}

// Diagnostic overlay: while nothing is rendered, the title window would stay blank. Every few
// presents a status text is drawn into it with GDI so the state is visible. This is not game output.
LONG draw_calls_seen;
void draw_overlay(LONG frame, double seconds)
{
    HWND window = x2w_startup::overlay_window();
    if (!window) return;
    HDC dc = GetDC(window);
    if (!dc) return;
    RECT client{};
    GetClientRect(window, &client);
    bool pictured = false;
    if (render_mode()) {
        AcquireSRWLockShared(&shown_lock);
        if (!shown_pixels.empty()) {
            BITMAPINFO info{};
            info.bmiHeader.biSize = sizeof(info.bmiHeader);
            info.bmiHeader.biWidth = shown_width;
            info.bmiHeader.biHeight = -shown_height;
            info.bmiHeader.biPlanes = 1;
            info.bmiHeader.biBitCount = 32;
            SetStretchBltMode(dc, COLORONCOLOR);
            StretchDIBits(dc, 0, 0, client.right, client.bottom, 0, 0, shown_width, shown_height, shown_pixels.data(), &info,
                          DIB_RGB_COLORS, SRCCOPY);
            pictured = true;
        }
        ReleaseSRWLockShared(&shown_lock);
    }
    HBRUSH background = CreateSolidBrush(RGB(16, 18, 24));
    if (pictured) {
        RECT box{client.left + 12, client.top + 12, client.left + 760, client.top + 190};
        FillRect(dc, &box, background);
    } else {
        FillRect(dc, &client, background);
    }
    DeleteObject(background);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(230, 232, 236));
    char text[640];
    std::snprintf(text, sizeof(text),
                  "Xbox2Windows startup experiment (diagnostic overlay)\n\n"
                  "Frame %ld presented, %.1f s since the first, %.1f fps; %ld draw and state calls recorded.\n"
                  "%s%s",
                  frame, seconds, seconds > 0 ? frame / seconds : 0.0, draw_calls_seen,
                  render_mode() ? "Rendering through translated pipelines; the picture behind is the presented surface: " : "",
                  render_mode() ? shown_note : "");
    RECT area{client.left + 24, client.top + 24, client.right - 24, client.bottom - 24};
    DrawTextA(dc, text, -1, &area, DT_LEFT | DT_TOP | DT_WORDBREAK);
    ReleaseDC(window, dc);
}

#include "startup_stubs/d3d12_present.inl"

// Queue slot 32 (Xbox order): (plane count, plane parameters*, present parameters*) -> HRESULT, the
// console present. Nothing is shown yet; frames are counted and a heartbeat is logged.
HRESULT __stdcall queue_present(Wrapper* self, UINT planes, const void* plane_parameters, const void* present_parameters)
{
    Timed timed(time_present);
    static LONG frames;
    static ULONGLONG started, last_report;
    const LONG frame = InterlockedIncrement(&frames);
    frame_number = frame;
    draw_dump_flush();
    {
        static const LONG wanted = [] {
            char value[16]{};
            return GetEnvironmentVariableA("X2W_D3D12_TRACE_FRAME", value, sizeof(value)) ? std::atol(value) : 0L;
        }();
        // The call that presents frame n-1 starts the trace; the one that presents frame n ends it.
        if (wanted && frame == wanted - 1) {
            log("x2w-frame ---- begin frame %ld", wanted);
            InterlockedExchange(&trace_frame_active, 1);
        } else if (wanted && frame == wanted) {
            InterlockedExchange(&trace_frame_active, 0);
            log("x2w-frame ---- end frame %ld", wanted);
        }
    }
    bool shown_in_window = false;
    {
        Wrapper* surface = nullptr;
        const auto* words = static_cast<const u64*>(plane_parameters);
        if (planes && words && x2w_local::readable(words, 0x18) && words[1] >= 1 && x2w_local::readable(reinterpret_cast<const void*>(words[2]), 8))
            surface = as_wrapper(*reinterpret_cast<void* const*>(words[2]));
        capture::present(frame, surface);
        // Render mode: the presented surface goes to the title's window through a swap chain.
        if (render_mode() && surface && surface->own().kind == resource_kind && surface->own().real)
            shown_in_window = present_to_window(static_cast<ID3D12CommandQueue*>(self->own().real), static_cast<ID3D12Resource*>(surface->own().real));
    }
    const ULONGLONG now = GetTickCount64();
    if (frame == 1) started = last_report = now;
    if (frame <= 2) {
        char a[3 * 0x50 + 1], b[3 * 0x40 + 1];
        hex_text(plane_parameters, 0x50, a, sizeof(a));
        hex_text(present_parameters, 0x40, b, sizeof(b));
        log("x2w-d3d12 CommandQueue::slot32(present) planes=%u plane=[%s] present=[%s]", planes, a, b);
    }
    if (render_mode() && frame % (shown_in_window ? 60 : 6) == 0 && planes && plane_parameters) {
        // Plane parameters: +0x00 frame token, +0x08 resource count, +0x10 pointer to the resources.
        const auto* words = static_cast<const u64*>(plane_parameters);
        if (x2w_local::readable(words, 0x18) && words[1] >= 1 && x2w_local::readable(reinterpret_cast<const void*>(words[2]), 8)) {
            Wrapper* surface = as_wrapper(*reinterpret_cast<void* const*>(words[2]));
            if (surface && surface->own().kind == resource_kind && surface->own().real)
                read_back_presented(static_cast<ID3D12CommandQueue*>(self->own().real), static_cast<ID3D12Resource*>(surface->own().real));
            else
                std::snprintf(shown_note, sizeof(shown_note), "presented object %p is not a known resource",
                              *reinterpret_cast<void* const*>(words[2]));
        }
    }
    if (shown_in_window) {
        if (frame % 30 == 0) {
            wchar_t text[200];
            swprintf(text, 200, L"Xbox2Windows - frame %ld, %.1f fps, %ld translated draws (title bar is diagnostics, picture is game output)",
                     frame, frame * 1000.0 / (now - started ? now - started : 1), translated_draws);
            SetWindowTextW(output_window(), text);
        }
    } else if (frame % 10 == 1 || (render_mode() && frame % 6 == 0)) {
        draw_overlay(frame, (now - started) / 1000.0);
    }
    // The null-binding total and its causes change together under draw_lock (draw_translated holds it), so they are copied there.
    LONG null_total = 0, null_causes[x2w_local::null_cause_count] = {};
    if (render_mode() && frame % 300 == 0) {
        AcquireSRWLockExclusive(&draw_lock);
        null_total = null_bindings;
        for (int c = 0; c < x2w_local::null_cause_count; ++c) null_causes[c] = null_by_cause[c];
        ReleaseSRWLockExclusive(&draw_lock);
    }
    if (render_mode() && frame % 300 == 0)
        log("x2w-render totals: resets %ld (failed %ld), closes %ld, clears %ld, target binds %ld (unresolved %ld), executes %ld, draws %ld (skipped %ld), "
            "translated draws %ld (not drawn %ld), pipelines translated %ld / missing %ld, null bindings %ld (entry %ld, buffer %ld, token %ld, wrapper %ld, dimension %ld), "
            "textures uploaded %ld (%ld KiB, unreadable %ld), heap-indexed roots deferred %ld (real draws refused: view heap %ld, "
            "sampler heap %ld, both heaps %ld, state %ld)", render_resets,
            render_reset_failures, render_closes, render_clears, render_target_binds, render_target_unresolved, render_executes, render_draws, render_draws_skipped,
            translated_draws, untranslated_draws, translation_ready, translation_missing, null_total,
            null_causes[0], null_causes[1], null_causes[2], null_causes[3], null_causes[4], textures_uploaded,
            texture_bytes_uploaded, textures_unreadable, heap_roots_deferred,
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::view_heap_missing)],
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::sampler_heap_missing)],
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::both_heaps_missing)],
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::state_not_restorable)]);
    if (render_mode() && frame % 300 == 0) {
        // Which command-list slots the title called in the last 300 frames, per frame.
        char line[2048];
        int n = std::snprintf(line, sizeof(line), "x2w-render list slots per frame:");
        for (int i = 0; i < 256 && n < static_cast<int>(sizeof(line)) - 24; ++i) {
            const LONG64 calls = InterlockedExchange64(&list_slot_census[i], 0);
            if (calls) n += std::snprintf(line + n, sizeof(line) - n, " %d:%.1f", i, calls / 300.0);
        }
        log("%s", line);
    }
    if (now - last_report >= 5000) {
        last_report = now;
        log("x2w-d3d12 views noted %lld (distinct %lld), descriptors copied %lld, heaps %llu", views_noted, views_distinct,
            descriptors_copied, static_cast<u64>(heap_infos().size()));
        {
            LARGE_INTEGER frequency;
            QueryPerformanceFrequency(&frequency);
            char text[512];
            int used = 0;
            for (int i = 0; i < time_count; ++i) {
                const LONG64 ticks = InterlockedExchange64(&time_ticks[i], 0), calls = InterlockedExchange64(&time_calls[i], 0);
                if (calls)
                    used += std::snprintf(text + used, sizeof(text) - used, " %s %lld ms/%lld", time_names[i],
                                          static_cast<long long>(ticks * 1000 / frequency.QuadPart), static_cast<long long>(calls));
            }
            log("x2w-d3d12 time inside the stand-in since the last report (ms/calls):%s", text);
            used = 0;
            text[0] = 0;
            for (int i = 0; i < 256; ++i) {
                const LONG64 ticks = InterlockedExchange64(&list_slot_ticks[i], 0), calls = InterlockedExchange64(&list_slot_calls[i], 0);
                if (calls && ticks * 1000 / frequency.QuadPart >= 20 && used < 440)
                    used += std::snprintf(text + used, sizeof(text) - used, " slot%d %lld ms/%lld", i,
                                          static_cast<long long>(ticks * 1000 / frequency.QuadPart), static_cast<long long>(calls));
            }
            if (used) log("x2w-d3d12 command-list slots over 20 ms:%s", text);
        }
        log("x2w-d3d12 heartbeat: %ld frames presented in %.1f s (%.1f fps overall); frame waits by type %ld/%ld/%ld/%ld, "
            "%lld ms slept", frame, (now - started) / 1000.0, frame * 1000.0 / (now - started ? now - started : 1),
            frame_waits[0], frame_waits[1], frame_waits[2], frame_waits[3], static_cast<long long>(frame_wait_ms));
    }
    return S_OK;
}

template <int K, int N>
void enable_forward() { tables[K][N] = reinterpret_cast<void*>(&forward_slot<K, N>); }
template <int K, int N, int R, int Desc = -1>
void enable_create() { tables[K][N] = reinterpret_cast<void*>(&create_slot<K, N, R, Desc>); }
template <int K>
void enable_object()
{
    // ID3D12Object: GetPrivateData, SetPrivateData, SetPrivateDataInterface, SetName.
    enable_forward<K, 3>();
    enable_forward<K, 4>();
    enable_forward<K, 5>();
    enable_forward<K, 6>();
    if (K != device_kind) tables[K][7] = reinterpret_cast<void*>(&child_get_device);
}
template <std::size_t... K>
void enable_objects(std::index_sequence<K...>) { (enable_object<static_cast<int>(K)>(), ...); }

void build_tables()
{
    fill_all_traps(std::make_index_sequence<kind_count>{});
    for (auto& table : tables) {
        table[0] = reinterpret_cast<void*>(&query_interface);
        table[1] = reinterpret_cast<void*>(&add_ref);
        table[2] = reinterpret_cast<void*>(&release);
    }
    enable_objects(std::make_index_sequence<kind_count>{});

    // Device, desktop ID3D12Device order (slots 7-47).
    enable_forward<device_kind, 7>();                    // GetNodeCount
    tables[device_kind][8] = reinterpret_cast<void*>(&device_create_command_queue);
    enable_create<device_kind, 9, 1>();                  // CreateCommandAllocator(type, riid, ppv)
    tables[device_kind][10] = reinterpret_cast<void*>(&device_create_pipeline<10>);
    tables[device_kind][11] = reinterpret_cast<void*>(&device_create_pipeline<11>);
    enable_create<device_kind, 12, 4>();                 // CreateCommandList(node, type, allocator, pso, riid, ppv)
    enable_forward<device_kind, 13>();                   // CheckFeatureSupport
    tables[device_kind][14] = reinterpret_cast<void*>(&device_create_descriptor_heap);
    enable_forward<device_kind, 15>();                   // GetDescriptorHandleIncrementSize
    enable_create<device_kind, 16, 3>();                 // CreateRootSignature(node, blob, size, riid, ppv)
    tables[device_kind][17] = reinterpret_cast<void*>(&view_slot<17, 1>);  // CreateConstantBufferView(desc, dest)
    tables[device_kind][18] = reinterpret_cast<void*>(&view_slot<18, 2>);  // CreateShaderResourceView(res, desc, dest)
    tables[device_kind][19] = reinterpret_cast<void*>(&view_slot<19, 3>);  // CreateUnorderedAccessView(res, counter, desc, dest)
    tables[device_kind][20] = reinterpret_cast<void*>(&view_slot<20, 2>);  // CreateRenderTargetView(res, desc, dest)
    tables[device_kind][21] = reinterpret_cast<void*>(&view_slot<21, 2>);  // CreateDepthStencilView(res, desc, dest)
    tables[device_kind][22] = reinterpret_cast<void*>(&view_slot<22, 1>);  // CreateSampler(desc, dest)
    tables[device_kind][23] = reinterpret_cast<void*>(&device_copy_descriptors);
    tables[device_kind][24] = reinterpret_cast<void*>(&device_copy_descriptors_simple);
    tables[device_kind][25] = reinterpret_cast<void*>(&device_resource_allocation_info);
    tables[device_kind][26] = reinterpret_cast<void*>(&device_custom_heap_properties);
    enable_create<device_kind, 27, 5, 2>();                 // CreateCommittedResource(props, flags, desc, state, clear, riid, ppv)
    enable_create<device_kind, 28, 1>();                 // CreateHeap(desc, riid, ppv)
    enable_create<device_kind, 29, 5, 2>();                 // CreatePlacedResource(heap, offset, desc, state, clear, riid, ppv)
    enable_create<device_kind, 30, 3, 0>();                 // CreateReservedResource(desc, state, clear, riid, ppv)
    tables[device_kind][34] = reinterpret_cast<void*>(&device_residency<34>);
    tables[device_kind][35] = reinterpret_cast<void*>(&device_residency<35>);
    enable_create<device_kind, 36, 2>();                 // CreateFence(initial, flags, riid, ppv)
    enable_forward<device_kind, 37>();                   // GetDeviceRemovedReason
    tables[device_kind][38] = reinterpret_cast<void*>(&device_copyable_footprints);
    enable_create<device_kind, 39, 1>();                 // CreateQueryHeap(desc, riid, ppv)
    enable_forward<device_kind, 40>();                   // SetStablePowerState
    tables[device_kind][41] = reinterpret_cast<void*>(&device_create_command_signature);                 // CreateCommandSignature(desc, root, riid, ppv)
    enable_forward<device_kind, 42>();                   // GetResourceTiling
    // Slot 43 (desktop GetAdapterLuid) is not installed: no call site has confirmed it.
    (void)&device_adapter_luid;
    // Xbox order from here.
    tables[device_kind][48] = reinterpret_cast<void*>(&device_create_command_queue_x);
    tables[device_kind][70] = reinterpret_cast<void*>(&device_memory_srv);
    tables[device_kind][73] = reinterpret_cast<void*>(&device_create_command_list_x);
    tables[device_kind][78] = reinterpret_cast<void*>(&device_frame_interval);
    tables[device_kind][79] = reinterpret_cast<void*>(&device_frame_event);
    tables[device_kind][80] = reinterpret_cast<void*>(&device_wait_frame_event);
    tables[device_kind][55] = reinterpret_cast<void*>(&device_create_derived_pipeline);
    tables[device_kind][57] = reinterpret_cast<void*>(&device_create_pipeline_placeholder<57>);
    tables[device_kind][67] = reinterpret_cast<void*>(&device_create_pipeline_placeholder<67>);
    tables[device_kind][62] = reinterpret_cast<void*>(&device_create_compute_pipeline_x);
    tables[device_kind][86] = reinterpret_cast<void*>(&device_create_pipeline_stream_x);
    for (int slot = 3; slot <= 6; ++slot) tables[pipeline_kind][slot] = reinterpret_cast<void*>(&placeholder_ok);
    for (int slot = 3; slot <= 6; ++slot) tables[command_signature_kind][slot] = reinterpret_cast<void*>(&placeholder_ok);
    tables[device_kind][108] = reinterpret_cast<void*>(&device_resource_allocation_info1);
    tables[device_kind][127] = reinterpret_cast<void*>(&device_create_committed_x);
    tables[device_kind][133] = reinterpret_cast<void*>(&device_place_resource_x);
    if (device_noop()) fill_device_noops(tables[device_kind], std::make_index_sequence<slot_count>{});
    if (list_noop()) {
        fill_noops<list_kind>(tables[list_kind], std::make_index_sequence<slot_count>{});
        tables[list_kind][7] = reinterpret_cast<void*>(&child_get_device);
    }
    if (render_mode()) {
        list_typed[9] = reinterpret_cast<ListTyped>(&list_close);
        list_typed[52] = reinterpret_cast<ListTyped>(&list_clear_target);
        list_typed[139] = reinterpret_cast<ListTyped>(&list_reset);
        list_typed[12] = reinterpret_cast<ListTyped>(&list_draw);
        list_typed[20] = reinterpret_cast<ListTyped>(&list_topology);
        list_typed[21] = reinterpret_cast<ListTyped>(&list_viewports);
        list_typed[22] = reinterpret_cast<ListTyped>(&list_scissors);
        list_typed[23] = reinterpret_cast<ListTyped>(&list_blend_factor);
        list_typed[24] = reinterpret_cast<ListTyped>(&list_stencil_ref);
        list_typed[25] = reinterpret_cast<ListTyped>(&list_pipeline);
        list_typed[32] = reinterpret_cast<ListTyped>(&list_set_descriptor_heaps);
        list_typed[33] = reinterpret_cast<ListTyped>(&list_compute_root);
        list_typed[34] = reinterpret_cast<ListTyped>(&list_graphics_root);
        list_typed[44] = reinterpret_cast<ListTyped>(&list_root_srv);
        list_typed[42] = reinterpret_cast<ListTyped>(&list_root_cbv);
        list_typed[50] = reinterpret_cast<ListTyped>(&list_targets);
        list_typed[13] = reinterpret_cast<ListTyped>(&list_draw_indexed);
        list_typed[36] = reinterpret_cast<ListTyped>(&list_table);
        list_typed[40] = reinterpret_cast<ListTyped>(&list_root_constants);
        list_typed[47] = reinterpret_cast<ListTyped>(&list_index_buffer);
        list_typed[48] = reinterpret_cast<ListTyped>(&list_vertex_buffers);
        list_typed[x2w_list_slots::execute_indirect_slot] = reinterpret_cast<ListTyped>(&list_execute_indirect);
    }
    // Slot 51 takes a float (xmm3), which the generic entry cannot pass on: the table calls the handler itself.
    if (render_mode() && list_noop()) tables[list_kind][51] = reinterpret_cast<void*>(&list_clear_depth);
    tables[device_kind][16] = reinterpret_cast<void*>(&device_create_root_signature);

    // Queue: not the desktop order (slot 18 is GetTimestampFrequency, desktop 16). Slots are
    // dispatched by argument shape until the order is known; see queue_dynamic.
    fill_queue_dynamic(tables[queue_kind], std::make_index_sequence<64>{});
    tables[queue_kind][17] = reinterpret_cast<void*>(&queue_wait);
    tables[queue_kind][18] = reinterpret_cast<void*>(&queue_timestamp_frequency);
    tables[queue_kind][32] = reinterpret_cast<void*>(&queue_present);

    // Fence: GetCompletedValue, SetEventOnCompletion, Signal, GetCreationFlags.
    tables[fence_kind][8] = reinterpret_cast<void*>(&fence_completed_value);
    // Desktop slots 9 (SetEventOnCompletion) and 11 (GetCreationFlags) wait for a confirming call site.
    tables[fence_kind][10] = reinterpret_cast<void*>(&fence_signal);
    tables[fence_kind][12] = reinterpret_cast<void*>(&fence_completed_address);

    tables[allocator_kind][8] = reinterpret_cast<void*>(&allocator_reset);  // Reset

    // Resource: Map, Unmap, GetDesc, GetGPUVirtualAddress, Write/ReadFromSubresource, GetHeapProperties.
    tables[resource_kind][8] = reinterpret_cast<void*>(&resource_map);
    tables[resource_kind][9] = reinterpret_cast<void*>(&resource_unmap);
    tables[resource_kind][11] = reinterpret_cast<void*>(&resource_gpu_address);
    // Desktop Resource 10 (GetDesc), 12-14, Heap 8 and DescriptorHeap 8 (GetDesc, struct returns) are
    // not installed until a call site confirms them: the queue's slot 18 showed what a wrong one does.
    (void)&resource_get_desc;
    (void)&heap_get_desc;
    (void)&descriptor_heap_get_desc;
    tables[descriptor_heap_kind][9] = reinterpret_cast<void*>(&descriptor_heap_cpu_start);
    tables[descriptor_heap_kind][10] = reinterpret_cast<void*>(&descriptor_heap_gpu_start);
    if (device_noop()) fill_all_remaining_noops(std::make_index_sequence<kind_count>{});
}

SRWLOCK written_lock = SRWLOCK_INIT;
std::unordered_map<u64, TextureWrite>& written_textures() { static std::unordered_map<u64, TextureWrite> m; return m; }

void note_texture_write(void* base, UINT bc7_width, UINT bc7_height)
{
    AcquireSRWLockExclusive(&written_lock);
    auto& written = written_textures()[reinterpret_cast<u64>(base)];
    ++written.generation;
    written.bc7_width = bc7_width;
    written.bc7_height = bc7_height;
    ReleaseSRWLockExclusive(&written_lock);
}
}  // namespace

void x2w_startup::attach_device(ID3D12Device* device, void** wrapped)
{
    static const bool ready = (build_tables(), true);
    (void)ready;
    removal_device = device;
    the_device = wrap(device, device_kind);
    *wrapped = the_device;
}

HRESULT x2w_startup::resolve_placed_build(void* guest_device, UINT64 dest, UINT64 scratch,
    const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& sizes,
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& build)
{
    if (build.SourceAccelerationStructureData ||
        (build.Inputs.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE))
        return E_INVALIDARG;
    constexpr UINT64 alignment = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
    const auto translate = [&](UINT64 address, UINT64 bytes, D3D12_RESOURCE_STATES state) -> UINT64 {
        if (address % alignment) return 0;
        const auto found = placed_buffer_range(static_cast<const Wrapper*>(guest_device), address, bytes);
        if (!found.owner || found.reject) return 0;
        const auto* info = resource_info(found.owner);
        if (!info || info->initial_state != state) return 0;
        const auto base = reinterpret_cast<UINT64>(found.owner->own().title_memory);
        const auto gpu = static_cast<ID3D12Resource*>(found.owner->own().real)->GetGPUVirtualAddress();
        const auto offset = address - base;
        if (!gpu || offset > UINT64_MAX - gpu || (gpu + offset) % alignment) return 0;
        return gpu + offset;
    };
    const auto out = translate(dest, sizes.ResultDataMaxSizeInBytes, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    const auto temp = translate(scratch, sizes.ScratchDataSizeInBytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!out || !temp) return E_INVALIDARG;
    build.DestAccelerationStructureData = out;
    build.ScratchAccelerationStructureData = temp;
    return S_OK;
}

// Called by the XG stand-in after it filled texture memory at a title address.
extern "C" __declspec(dllexport) void x2w_d3d12_texture_written(void* base, unsigned long long bytes)
{
    (void)bytes;
    note_texture_write(base, 0, 0);
}

extern "C" __declspec(dllexport) void x2w_d3d12_bc7_4k_s_written(void* base, UINT width, UINT height)
{
    note_texture_write(base, width, height);
}

bool x2w_d3d12_permissive()
{
    char value[4]{};
    return GetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", value, sizeof(value)) && value[0] == '1';
}

HRESULT x2w_d3d12_permissive_create_device(IUnknown* adapter, const void* parameters, REFIID iid, void** out)
{
    char text[48];
    guid_text(&iid, text, sizeof(text));
    char block[3 * 0x40 + 1];
    hex_text(parameters, 0x40, block, sizeof(block));
    if (!out) return E_POINTER;
    *out = nullptr;
    const bool debug = [] {
        char value[4]{};
        return GetEnvironmentVariableA("X2W_D3D12_DEBUG", value, sizeof(value)) && value[0] == '1';
    }();
    if (debug) {
        ID3D12Debug* layer = nullptr;
        const HRESULT debug_hr = D3D12GetDebugInterface(IID_PPV_ARGS(&layer));
        if (SUCCEEDED(debug_hr)) layer->EnableDebugLayer();
        log("x2w-d3d12 debug layer: 0x%08lx%s", static_cast<unsigned long>(debug_hr),
            SUCCEEDED(debug_hr) ? " enabled" : " (not installed?)");
    }
    ID3D12Device* device = nullptr;
    const HRESULT hr = kind_for(iid) == device_kind || iid == __uuidof(IUnknown)
        ? D3D12CreateDevice(adapter ? reinterpret_cast<IUnknown*>(unwrap(reinterpret_cast<u64>(adapter))) : nullptr,
                            D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))
        : E_NOINTERFACE;
    log("x2w-d3d12 D3D12XboxCreateDevice adapter=%p params=[%s] riid=%s -> 0x%08lx", static_cast<void*>(adapter), block,
        text, static_cast<unsigned long>(hr));
    if (FAILED(hr)) return hr;
    if (debug) device->QueryInterface(IID_PPV_ARGS(&info_queue));
    x2w_startup::attach_device(device, out);
    return S_OK;
}
