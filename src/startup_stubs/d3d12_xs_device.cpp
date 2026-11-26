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
#include "x2w_build_rev.h"
#include "xmem_page_batch.h"
#include "startup_stubs/commit_state.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/logical_view_table.h"
#include "startup_stubs/list_compute_root.h"
#include "startup_stubs/overlay_window.h"
#include "startup_stubs/draw_bmp.h"
#include "startup_stubs/draw_dump_tracking.h"
#include "startup_stubs/pixel_history.h"
#include "startup_stubs/translated_sampler_bindings.h"
#include "startup_stubs/export_write_mask.h"
#include "startup_stubs/depth_state_apply.h"
#include "startup_stubs/d16_image_format.h"
#include "startup_stubs/formatted_buffer.h"
#include "startup_stubs/heap_indexed_root.h"
#include "startup_stubs/indexed_indirect_guard.h"
#include "startup_stubs/indexed_rect_geometry.h"
#include "startup_stubs/list_dispatch.h"
#include "startup_stubs/bc7_4k_s.h"
#include "startup_stubs/bc64_chain.h"
#include "startup_stubs/address_texture.h"
#include "startup_stubs/list_upload_pages.h"
#include "startup_stubs/resource_budget.h"
#include "startup_stubs/sysmem_create.h"
#include "startup_stubs/upload_reservation.h"
#include <optional>
#include "startup_stubs/derived_blend.h"
#include "startup_stubs/placement_publish.h"
#include "startup_stubs/resource_address_index.h"
#include "startup_stubs/placed_build.h"
#include "startup_stubs/placed_buffer_range.h"
#include "startup_stubs/linear_buffer_backing.h"
#include "startup_stubs/memory_srv_report.h"
#include "startup_stubs/memory_srv_census.h"
#include "startup_stubs/gpu_time_probe.h"
#include "startup_stubs/memory_srv_as.h"
#include "startup_stubs/per_reason_refusal_census.h"
#include "startup_stubs/buffer_format_word3.h"
#include "startup_stubs/memory_srv_buffer.h"
#include "startup_stubs/srv_refusal_cache.h"
#include "startup_stubs/single_element_srv.h"
#include "startup_stubs/view_dimension.h"
#include "startup_stubs/list_copy.h"
#include "startup_stubs/d3d12_teardown.h"
#include "startup_stubs/rect_indirect_args.h"
#include "startup_stubs/split_indirect_dispatch.h"
#include "startup_stubs/profile_zone.h"

#include "startup_stubs/slice_footprints.h"
#include "startup_stubs/texture_drop_mips.h"
#include "startup_stubs/present_signature_sampler.h"
#include "startup_stubs/present_planes.h"
#include "startup_stubs/plane_compositor.h"

#include <d3d12.h>
#include "command_list_slot_names.h"
#include "command_signature_filter.h"
#include "console_plane_size.h"
#include "descriptor_heap_type.h"
#include "small_texture_4k.h"
#include "null_binding_cause.h"
#include "root_constant_store.h"
#include "null_texture_view.h"
#include "packed_target_format.h"
#include "present_view_format.h"
#include "state_block_blend.h"
#include "title_heap_memory.h"
#include "startup_stubs/xmem_section_buffer.h"
#include <d3d12sdklayers.h>
#include <d3d12shader.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// The heap-indexing bits src/d3d12_xs_shim.cpp removed for the blob it returned at exactly this
// buffer. A build that compiles this file without the shim (offscreen_check) links the fallback, 0.
extern "C" std::uint32_t x2w_stripped_root_flags(const void* blob, std::size_t bytes);
extern "C" std::uint32_t x2w_no_stripped_root_flags(const void*, std::size_t) { return 0; }
#pragma comment(linker, "/alternatename:x2w_stripped_root_flags=x2w_no_stripped_root_flags")

namespace {

using u64 = std::uint64_t;
constexpr int slot_count = 512;
// Refused slot 70 calls, "token" texture nulls (memory_srv_census.h), and indirect refusal censuses
// per (pipeline key, reason) (per_reason_refusal_census.h), printed with the five-second report.
x2w_startup::MemorySrvCensus memory_srv_census;
x2w_startup::TokenNullCensus token_null_census;
x2w_startup::MemorySrvWrites memory_srv_writes;
x2w_startup::PerReasonRefusalCensus indirect_draw_refusal_census{"indirect-draw"};
x2w_startup::PerReasonRefusalCensus indirect_other_refusal_census{"indirect-other"};

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
struct RootParameter;
enum class IndirectCommand { none, dispatch, draw, draw_indexed, dispatch_mesh };
struct IndirectGraphicsPipeline {
    ID3D12RootSignature* root{};
    DXGI_FORMAT target_format{};
    bool mesh{};
    bool cached_mesh{};
    UINT cached_target_count{};
    DXGI_FORMAT cached_targets[8]{};
    DXGI_FORMAT cached_depth{};
    bool cached_depth_writes{};
    const std::vector<RootParameter>* guest_root_params{};
    UINT cached_srv_root = UINT_MAX;
};

// Why a resource is outside the retire class (X2W_D3D12_RELEASE_RETIRED), decided at creation, and why it left it later.
// Both only feed the opt-in X2W_D3D12_RETIRE_REASONS diagnostics. never::slot is the zero value: created by a path that
// does not classify.
enum class RetireNever : std::uint8_t { slot, none, buffer, dimension, msaa, rt_ds_uav, heap_type, heap_flags, title_memory,
                                        aliased, reserved, count };
enum class RetireLeft : std::uint8_t { none, view, view_copy, residency, present, hold_missed, count };

struct alignas(16) WrapperOwn {
    IUnknown* real;
    LONG refs;
    Kind kind;
    x2w_budget::Ticket* budget;
    x2w_budget::Ticket* backing_budget;
    void* title_memory;       // Resource: the title-side memory (placement address or hook allocation), or null
    std::uint64_t title_cpu;  // DescriptorHeap: title-owned descriptor memory, or 0
    // Fence: the completed value the title polls through the address from slot 12 (monitorx/mwaitx).
    // Updated when the fence is signaled; GPU work counts as complete at once while lists are no-ops.
    volatile LONG64 completed;
    bool list_open;   // CommandList (render mode): recording between reset and close
    x2w_upload::Arena* list_uploads; // Current recording's allocations, owned by its desktop allocator.
    ID3D12Resource* list_scratch;    // Current recording's rectangle-list argument scratch, owned by its desktop allocator.
    UploadChunkMask upload_chunks;    // Ring chunks referenced by this recording, including while closed.
    bool list_close_failed; // Desktop Reset repeats a failed Close result indefinitely.
    bool tiled_address_srv; // Resource backed by 4 KiB tiled guest memory for translated draws.
    bool lazy_slot133_texture; // Slot 133 read-only texture whose desktop CreateCommittedResource is deferred.
    bool lazy_view_created;    // Resource: at least one materialized view was created referencing this wrapper.
    std::uint8_t drop_levels;  // Lazy texture: top mip levels left out of the desktop texture (texture_drop_mips.h), 0 = all kept.
    bool retire_when_idle;  // Resource: in the retire class (X2W_D3D12_RELEASE_RETIRED); cleared by any other use. Plain
                            // accesses: a race needs the guest to use a resource across its own final Release.
    bool retire_hold_missed;  // Resource: retire_hold could not record a use (no desktop object or no room), so the
                              // drain must keep it for the process instead of dropping a list's desktop object.
    std::uint8_t retire_never;  // RetireNever of this resource (set only with X2W_D3D12_RELEASE_RETIRED)
    std::uint8_t retire_left;   // RetireLeft: the first use that took it out of the class
    volatile LONG retire_view_refs;  // Live tracked slot-18 SRV descriptor slots referencing this resource.
    bool retire_has_written_gen;     // Whether written_textures()[title_memory].generation was snapshotted at Release.
    u64 retire_written_gen;          // Snapshotted written_textures() generation for stale-entry cleanup.
    bool list_ready;  // CommandList (render mode): closed with recorded work, not yet submitted
    bool compute_pipeline;   // PipelineState: a desktop compute pipeline
    u64 compute_trace_key;   // Opt-in trace: selected slot 62 container hash
    const void* compute_trace_root;
    LONG compute_trace_dispatches;
    bool root_without_bindings; // RootSignature: decoded and has no parameters or static samplers
    ID3D12RootSignature* pipeline_root; // PipelineState: desktop root signature its compute stream declared
    u64 compute_request_key;           // PipelineState: key of the slot 67 request it published, or 0
    const void* compute_request_root;  // PipelineState: root signature wrapper named by that slot 67 description
    bool root_srv_only;         // RootSignature: decoded, no static samplers, every parameter a root SRV (those are forwarded)
    volatile LONG queue_wait_seen; // CommandQueue: publish later signals only after GPU completion.
    void* pass;       // CommandList (render mode): recording state (Pass)
    Wrapper* list_device;    // CommandList: wrapper that created this list.
    Wrapper* signature_device; // CommandSignature: owner of the supported single-command shape
    IndirectCommand indirect_command;
    ID3D12RootSignature* signature_root; // CommandSignature: desktop root of a Dispatch with root constants, or null
    UINT indirect_bytes; // CommandSignature: bytes one command reads (root constants plus the Dispatch record)
    IndirectGraphicsPipeline indirect_graphics;
    u64 mesh86_key{};
    UINT list_node_mask;
    D3D12_QUERY_HEAP_DESC query_heap; // QueryHeap: the desc it was created with (device slot 39)
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
void request_translation(Wrapper* pipeline);
struct RootParameter;
void request_slot57_mesh(Wrapper* pipeline, const std::vector<RootParameter>* bound_root);
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
    x2w_trap::append_log(line, static_cast<DWORD>(n), false);
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

bool first_calls(LONG& counter, LONG limit = 3) { return x2w_local::first_calls(counter, limit); }

// One line for each of the first 32 textures created in system memory after the default heap ran out of video memory.
// The running totals are in the x2w-budget line.
void log_sysmem_fallback(const void* address, const D3D12_RESOURCE_DESC& desc, const char* path)
{
    static LONG logged;
    if (!first_calls(logged, 32)) return;
    log("x2w-d3d12 sysmem fallback %s: address=%p dim=%d %llux%ux%u mips=%u fmt=%d samples=%u -> created in system memory (L0)",
        path, address, desc.Dimension, static_cast<unsigned long long>(desc.Width), desc.Height, desc.DepthOrArraySize,
        desc.MipLevels, desc.Format, desc.SampleDesc.Count);
}

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

volatile LONG wraps_to_fail;  // fixture-only, set by x2w_d3d12_fail_wraps
constexpr u64 out_of_memory = static_cast<std::uint32_t>(E_OUTOFMEMORY);  // for slots that return u64

// Returns nullptr when the wrapper cannot be allocated or registered (#2653). The caller still owns
// `real` then: it releases what it created and returns E_OUTOFMEMORY.
Wrapper* wrap(IUnknown* real, Kind kind)
{
    for (LONG n = wraps_to_fail; n > 0; n = wraps_to_fail)
        if (InterlockedCompareExchange(&wraps_to_fail, n - 1, n) == n) return nullptr;
    const std::size_t area = kind == list_kind ? 0x10000 : sizeof(Wrapper);
    auto* block = static_cast<unsigned char*>(_aligned_malloc(sizeof(WrapperOwn) + area, 16));
    if (!block) {
        static LONG failed;
        if (first_calls(failed, 4)) log("x2w-d3d12 wrap(%s): wrapper allocation failed -> E_OUTOFMEMORY", kind_names[kind]);
        return nullptr;
    }
    std::memset(block, 0, sizeof(WrapperOwn) + area);
    auto* w = reinterpret_cast<Wrapper*>(block + sizeof(WrapperOwn));
    w->vtable = tables[kind];
    w->own().real = real;
    w->own().refs = 1;
    w->own().kind = kind;
    if (kind == fence_kind && real) w->own().completed = static_cast<LONG64>(static_cast<ID3D12Fence*>(real)->GetCompletedValue());
    AcquireSRWLockExclusive(&registry_lock);
    try {
        registry().insert(w);
    } catch (const std::bad_alloc&) {
        ReleaseSRWLockExclusive(&registry_lock);
        _aligned_free(block);
        return nullptr;
    }
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
bool compute_binding_trace_enabled();
void trace_compute_binding(Wrapper*, int slot, u64 first, u64 second, u64 third, u64 fourth);
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
    if constexpr (K == list_kind && (N == 14 || N == 35 || N == 37 || N == 39 || N == 41 || N == 43 || N == 45)) {
        if (compute_binding_trace_enabled()) trace_compute_binding(reinterpret_cast<Wrapper*>(rcx), N, rdx, r8, r9, s0);
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
ID3D12Resource* materialize_lazy_texture(Wrapper* w);

// Desktop-order forwarder for a slot whose arguments are integers or pointers (no floating-point
// argument in the first four positions, at most eight arguments). Wrapped objects among the arguments
// are replaced by their desktop objects; unused argument words are caller stack the callee ignores.
template <int K, int N>
__declspec(noinline) u64 forward_slot(Wrapper* self, u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g, u64 h)
{
    Timed timed(time_forward);
    if (!self->own().real) {
        if constexpr (K == resource_kind) {
            if constexpr (N == 12 || N == 13 || N == 14) {
                if (!self->own().lazy_slot133_texture || !materialize_lazy_texture(self)) return static_cast<u64>(E_FAIL);
            } else {
                if constexpr (N == 6) return u64{S_OK};  // ID3D12Object::SetName on an unmaterialized lazy/tiled resource
                if constexpr (N == 3) return static_cast<u64>(DXGI_ERROR_NOT_FOUND);  // GetPrivateData
                return static_cast<u64>(E_FAIL);
            }
        } else {
            return 0;
        }
    }
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
    // A packed target the adapter cannot render to or write as a UAV: packed_target_format.h.
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

// An optimized clear belongs to the actual texture format, including when the
// wrapper substitutes it. Preserve the caller's clear colour.
const D3D12_CLEAR_VALUE* packed_clear_value(const D3D12_CLEAR_VALUE* clear, DXGI_FORMAT requested,
                                          DXGI_FORMAT created, D3D12_CLEAR_VALUE& copy)
{
    const auto* mapping = x2w_startup::packed_target_mapping(requested);
    if (!clear || !mapping || created != mapping->replacement || desktop_format(clear->Format) != requested)
        return clear;
    copy = *clear;
    copy.Format = created;
    return &copy;
}

bool legacy_retire_view_stats()
{
    static const bool requested = [] {
        if (!x2w_local::env_on("X2W_D3D12_RELEASE_RETIRED_VIEWS")) return false;
        log("x2w-d3d12: X2W_D3D12_RELEASE_RETIRED_VIEWS no longer clears live descriptors; budget statistics only");
        return true;
    }();
    return requested;
}

bool release_retired_enabled()
{
    static const bool on = [] {
        (void)legacy_retire_view_stats();
        return x2w_local::env_on("X2W_D3D12_RELEASE_RETIRED");
    }();
    return on;
}

// The retire class of X2W_D3D12_RELEASE_RETIRED, decided once at creation from the desktop description: a committed
// 2D texture with one sample, no render-target, depth-stencil or unordered-access use, in a default heap with no heap
// flags. Slot 27 and non-hooked slot 127 have no title memory; slot 133 committed 2D textures with title memory are
// included (cleaned from resource_address_index, texture_uploads and written_textures on retirement), while slot 127
// resources with __XMemAllocationHooks__ title memory remain excluded. Every use reaches the shim through handlers
// that retire_hold or shadow-slot SRV ref-counting cover, or that clear the flag; see
// docs/startup-runtime.md#d3d12-resource-budget.
RetireNever retire_classify(const D3D12_HEAP_PROPERTIES& props, D3D12_HEAP_FLAGS heap_flags, const D3D12_RESOURCE_DESC& desc,
                            const void* title_memory, bool allow_title_memory)
{
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return RetireNever::buffer;
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return RetireNever::dimension;
    if (desc.SampleDesc.Count != 1) return RetireNever::msaa;
    if ((desc.Flags & ~D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) != D3D12_RESOURCE_FLAG_NONE) return RetireNever::rt_ds_uav;
    if (props.Type != D3D12_HEAP_TYPE_DEFAULT) return RetireNever::heap_type;
    if (heap_flags != D3D12_HEAP_FLAG_NONE) return RetireNever::heap_flags;
    if (title_memory && !allow_title_memory) return RetireNever::title_memory;
    return RetireNever::none;
}

// Decides the retire class of a new wrapper once, from the desktop description, and records the reason for the diagnostics.
void retire_set_class(Wrapper* w, const D3D12_HEAP_PROPERTIES& props, D3D12_HEAP_FLAGS heap_flags,
                      const D3D12_RESOURCE_DESC& desc, const void* title_memory, bool allow_title_memory = false)
{
    if (!release_retired_enabled()) return;
    const RetireNever why = retire_classify(props, heap_flags, desc, title_memory, allow_title_memory);
    w->own().retire_never = static_cast<std::uint8_t>(why);
    w->own().retire_when_idle = why == RetireNever::none;
}

template <int K, int N, int R, int Desc = -1>
__declspec(noinline) u64 create_slot(Wrapper* self, u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g, u64 h)
{
    Timed timed(time_create);
    u64 args[8] = {a, b, c, d, e, f, g, h};
    for (int i = 0; i < R; ++i) args[i] = unwrap(args[i]);
    D3D12_RESOURCE_DESC desc_copy{};
    D3D12_CLEAR_VALUE clear_copy{};
    DXGI_FORMAT requested_format = DXGI_FORMAT_UNKNOWN;
    if (Desc >= 0 && args[Desc]) {
        requested_format = desktop_format(reinterpret_cast<const D3D12_RESOURCE_DESC*>(args[Desc])->Format);
        desc_copy = sanitized(*reinterpret_cast<const D3D12_RESOURCE_DESC*>(args[Desc]));
        args[Desc] = reinterpret_cast<u64>(&desc_copy);
        if constexpr (K == device_kind && (N == 27 || N == 29))
            args[4] = reinterpret_cast<u64>(packed_clear_value(reinterpret_cast<const D3D12_CLEAR_VALUE*>(args[4]),
                                                               requested_format, desc_copy.Format, clear_copy));
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
        if (!child) {
            static_cast<IUnknown*>(*out)->Release();
            *out = nullptr;
            return out_of_memory;
        }
        if constexpr (K == device_kind && N == 27) {
            if (kind == resource_kind && args[0])
                retire_set_class(child, *reinterpret_cast<const D3D12_HEAP_PROPERTIES*>(args[0]),
                                 static_cast<D3D12_HEAP_FLAGS>(args[1]), desc_copy, nullptr);
        }
        if constexpr (K == device_kind && (N == 29 || N == 30)) {
            if (kind == resource_kind && release_retired_enabled())
                child->own().retire_never = static_cast<std::uint8_t>(N == 29 ? RetireNever::aliased : RetireNever::reserved);
        }
        if constexpr (K == device_kind && (N == 27 || N == 28 || N == 29)) {
            if (x2w_budget::enabled()) {
                using Microsoft::WRL::ComPtr;
                auto* device = static_cast<ID3D12Device*>(self->own().real);
                if constexpr (N == 28) {
                    ComPtr<ID3D12Heap> heap;
                    if (SUCCEEDED(child->own().real->QueryInterface(IID_PPV_ARGS(&heap))))
                        child->own().budget = x2w_budget::backing_heap(heap.Get());
                } else {
                    ComPtr<ID3D12Resource> resource;
                    if (SUCCEEDED(child->own().real->QueryInterface(IID_PPV_ARGS(&resource)))) {
                        if constexpr (N == 27) {
                            const auto* props = reinterpret_cast<const D3D12_HEAP_PROPERTIES*>(a);
                            child->own().budget = x2w_budget::committed(device, resource.Get(), props->Type);
                        } else {
                            child->own().budget = x2w_budget::resource(device, resource.Get(), x2w_budget::placed);
                            if (auto* heap = as_wrapper(reinterpret_cast<const void*>(a))) {
                                child->own().backing_budget = heap->own().budget;
                                if (child->own().backing_budget) child->own().backing_budget->begin_use();
                            }
                        }
                    }
                }
            }
        }
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
        const auto refs = InterlockedIncrement(&self->own().refs);
        if (self->own().kind == resource_kind && self->own().title_memory) x2w_startup::note_resource_reference(refs);
        *out = self;
        hr = S_OK;
    }
    log("x2w-d3d12 %s::QueryInterface riid=%s -> 0x%08lx", kind_names[self->own().kind], text, static_cast<unsigned long>(hr));
    return hr;
}
ULONG __stdcall add_ref(Wrapper* self)
{
    const auto refs = InterlockedIncrement(&self->own().refs);
    if (self->own().kind == resource_kind && self->own().title_memory) x2w_startup::note_resource_reference(refs);
    return static_cast<ULONG>(refs);
}
// X2W_D3D12_LINEAR_BUFFERS (opt-in): the shared or copied backing of slot 133's linear buffers
// (src/startup_stubs/linear_buffer_backing.h). Unused while the switch is off.
x2w_startup::LinearBuffers& linear_buffers()
{
    static x2w_startup::LinearBuffers buffers;
    return buffers;
}
// The wrappers and their desktop objects stay alive for the process (experiment scope). The one exception is
// X2W_D3D12_RELEASE_RETIRED=1: the desktop object of a resource of the retire class is freed once nothing refers to it.
// The wrapper stays.
void retire_placement(Wrapper* w);
void register_retire_queue(Wrapper* queue);
void retire_resource(Wrapper* w);
void retire_leave_class(const void* object, RetireLeft why);
void retire_add_view_ref(const void* object);
void retire_release_view_ref(const void* object);
void retire_drop_heap_views(Wrapper* heap);
void retire_note_title_release(Wrapper* w);
void retire_hold(Wrapper* list, Wrapper* resource);
void retire_drop_recording(Wrapper* list);
void release_upload_recording(Wrapper* list);
void retire_discard_holds(Wrapper* list);
void residency_discard_list(Wrapper* list);
void residency_title_touched(ID3D12Resource* desktop);
void retire_note_submission(ID3D12CommandQueue* queue, const std::vector<Wrapper*>& lists);
void drain_retired_resources();
bool retire_reasons_enabled();
void retire_diag_note_release(Wrapper* w);
void drop_mesh_lease_owner(Wrapper* w);
void discard_mesh_lease_recording(ID3D12CommandList* list);
void clear_unplaced_buffers(Wrapper* device);

bool lazy_textures_enabled()
{
    static const bool on = x2w_local::env_on("X2W_D3D12_LAZY_TEXTURES");
    return on;
}
// X2W_D3D12_TEXTURE_DROP_MIPS=<levels> (1 to 4, default off), X2W_D3D12_TEXTURE_DROP_MIPS_MIN_BYTES (default 4 MiB).
const x2w_startup::drop_mips::Config& drop_mips_config()
{
    static const x2w_startup::drop_mips::Config config = [] {
        char levels[16]{}, bytes[24]{};
        GetEnvironmentVariableA("X2W_D3D12_TEXTURE_DROP_MIPS", levels, sizeof(levels));
        GetEnvironmentVariableA("X2W_D3D12_TEXTURE_DROP_MIPS_MIN_BYTES", bytes, sizeof(bytes));
        return x2w_startup::drop_mips::parse_config(levels, bytes);
    }();
    return config;
}
struct LazyTextureRecord {
    Wrapper* device{};
    D3D12_RESOURCE_DESC desc{};
    DXGI_FORMAT desktop_format{};
};
std::mutex lazy_texture_mutex;  // lock order: draw_lock -> lazy_texture_mutex -> note_lock -> retire_lock
// Publications of a lazy texture in progress and finished, so a guest descriptor copy can tell whether its native copy may
// have raced one (it then writes the real view into the destination, see refresh_copied_lazy_views).
LONG lazy_publishing, lazy_published_total;
std::unordered_map<const Wrapper*, LazyTextureRecord>& lazy_textures()
{
    static std::unordered_map<const Wrapper*, LazyTextureRecord> m;
    return m;
}
bool lazy_texture_desc(const Wrapper* w, D3D12_RESOURCE_DESC& out)
{
    if (!w || !w->own().lazy_slot133_texture) return false;
    std::lock_guard lock(lazy_texture_mutex);
    const auto found = lazy_textures().find(w);
    if (found == lazy_textures().end()) return false;
    out = found->second.desc;
    return true;
}
ID3D12Resource* materialize_lazy_texture(Wrapper* w);
bool retire_unmaterialized_lazy_texture(Wrapper* w);
ULONG __stdcall release(Wrapper* self)
{
    const auto refs = InterlockedDecrement(&self->own().refs);
    if (!refs) {
        if (self->own().budget) self->own().budget->end_use();
        if (self->own().backing_budget) self->own().backing_budget->end_use();
        if (self->own().kind == descriptor_heap_kind) retire_drop_heap_views(self);
        if (self->own().kind == resource_kind && self->own().title_memory) {
            drop_mesh_lease_owner(self);
            // The last release ends the resource for existing_memory_buffer and address_resource_overlap (they skip
            // refs <= 0) whether or not it has a placement entry: a refusal cached while it was live is stale.
            x2w_startup::note_placement_change();
            retire_placement(self);
            if (self->own().retire_when_idle) retire_note_title_release(self);
            if (x2w_startup::linear_buffer_mode() != x2w_startup::LinearBufferMode::off)
                linear_buffers().retire(static_cast<ID3D12Resource*>(self->own().real));
        }
        if (self->own().kind == list_kind && self->own().real) {
            release_upload_recording(self);
            discard_mesh_lease_recording(static_cast<ID3D12CommandList*>(self->own().real));
            retire_drop_recording(self);
        }
        if (self->own().kind == resource_kind && retire_reasons_enabled()) retire_diag_note_release(self);
        const bool lazy_retired = self->own().kind == resource_kind && self->own().lazy_slot133_texture &&
            retire_unmaterialized_lazy_texture(self);
        if (self->own().retire_when_idle && !lazy_retired) retire_resource(self);
        if (self->own().kind == device_kind)
            clear_unplaced_buffers(self);
    }
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
    if (!*out) {
        queue->Release();
        return E_OUTOFMEMORY;
    }
    register_retire_queue(static_cast<Wrapper*>(*out));
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
    if (!*out) {
        queue->Release();
        return E_OUTOFMEMORY;
    }
    register_retire_queue(static_cast<Wrapper*>(*out));
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

// A single small 2D texture asked with a 4 KiB alignment gets the console's 4 KiB answer (small_texture_4k.h).
bool small_texture_answer(UINT count, const D3D12_RESOURCE_DESC* descs, D3D12_RESOURCE_ALLOCATION_INFO& info)
{
    if (count != 1 || !descs || !x2w_startup::small_texture_4k(descs[0], info)) return false;
    static LONG calls;
    if (first_calls(calls, 8))
        log("x2w-d3d12 allocation info: 4 KiB answer size 0x%llx for %llux%u mips %u format %d", info.SizeInBytes, descs[0].Width,
            descs[0].Height, descs[0].MipLevels, descs[0].Format);
    return true;
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
    small_texture_answer(count, descs, info);
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
// its argument descriptors are logged. The observed console-only type 0x69 gets a placeholder before
// desktop validation. Other types still reach the desktop even when they postdate DISPATCH_MESH.
u64 __stdcall device_create_command_signature(Wrapper* self, const D3D12_COMMAND_SIGNATURE_DESC* desc, u64 root,
                                              u64 iid, u64 out)
{
    void** output = reinterpret_cast<void**>(out);
    const bool readable_args = x2w_local::readable(desc, sizeof(*desc)) &&
        x2w_local::readable(desc->pArgumentDescs, std::size_t{desc->NumArgumentDescs} * sizeof(*desc->pArgumentDescs));
    if (output && readable_args && x2w_startup::has_console_indirect_arg_type(desc->pArgumentDescs, desc->NumArgumentDescs) &&
        x2w_local::writable(output, sizeof(*output))) {
        static LONG placeholders;
        if (first_calls(placeholders, 8)) {
            log("x2w-d3d12   CreateCommandSignature: console argument type -> placeholder (stride=%u args=%u)",
                desc->ByteStride, desc->NumArgumentDescs);
            for (UINT i = 0; i < desc->NumArgumentDescs && i < 8; ++i) {
                char bytes[3 * 16 + 1];
                hex_text(&desc->pArgumentDescs[i], sizeof(D3D12_INDIRECT_ARGUMENT_DESC), bytes, sizeof(bytes));
                log("x2w-d3d12     arg[%u]=[%s]", i, bytes);
            }
        }
        *output = wrap(nullptr, command_signature_kind);
        return *output ? u64{S_OK} : out_of_memory;
    }
    const u64 result = create_slot<device_kind, 41, 2>(self, reinterpret_cast<u64>(desc), root, iid, out, 0, 0, 0, 0);
    if (SUCCEEDED(static_cast<HRESULT>(result)) && out && desc && root && desc->NumArgumentDescs > 1) {
        // A Dispatch preceded by root constants for the signature's root (#1759 part 2). Other root argument
        // types (root views, vertex and index buffer views) and graphics commands remain unsupported here.
        auto* signature = *reinterpret_cast<Wrapper**>(out);
        UINT bytes = sizeof(D3D12_DISPATCH_ARGUMENTS);
        bool constants_only = desc->pArgumentDescs[desc->NumArgumentDescs - 1].Type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
        for (UINT i = 0; constants_only && i + 1 < desc->NumArgumentDescs; ++i) {
            constants_only = desc->pArgumentDescs[i].Type == D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
            bytes += desc->pArgumentDescs[i].Constant.Num32BitValuesToSet * sizeof(UINT);
        }
        if (constants_only && desc->ByteStride == bytes && signature && signature->own().kind == command_signature_kind) {
            auto& state = signature->own();
            state.indirect_command = IndirectCommand::dispatch;
            state.signature_root = reinterpret_cast<ID3D12RootSignature*>(unwrap(root));
            state.indirect_bytes = bytes;
            state.signature_device = self;
        }
    }
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
            state.indirect_bytes = desc->ByteStride;
            if (state.indirect_command != IndirectCommand::none) state.signature_device = self;
        }
    }
    static LONG failures;
    if (FAILED(static_cast<HRESULT>(result)) && readable_args && first_calls(failures, 8)) {
        log("x2w-d3d12   CreateCommandSignature failed: stride=%u args=%u node=0x%x root=0x%llx", desc->ByteStride,
            desc->NumArgumentDescs, desc->NodeMask, root);
        for (UINT i = 0; i < desc->NumArgumentDescs && i < 8; ++i) {
            char bytes[3 * 16 + 1];
            hex_text(&desc->pArgumentDescs[i], sizeof(D3D12_INDIRECT_ARGUMENT_DESC), bytes, sizeof(bytes));
            log("x2w-d3d12     arg[%u]=[%s]", i, bytes);
        }
    }
    // Preserve the existing fallback for unknown types only after desktop has refused the real request.
    if (FAILED(static_cast<HRESULT>(result)) && readable_args && output && x2w_local::writable(output, sizeof(*output))) {
        for (UINT i = 0; i < desc->NumArgumentDescs; ++i) {
            if (static_cast<UINT>(desc->pArgumentDescs[i].Type) > D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH) {
                *output = wrap(nullptr, command_signature_kind);
                return *output ? u64{S_OK} : out_of_memory;
            }
        }
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
    if (small_texture_answer(count, descs, info) && info1) *info1 = {0, info.Alignment, info.SizeInBytes};
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

// Device slot 15, desktop GetDescriptorHandleIncrementSize(type). A type desktop has is forwarded as before. Any
// other value makes desktop answer 0 and the debug layer report an error (#870): the stand-in logs the value once
// and answers 0 itself, with no increment made up for it.
static_assert(x2w_startup::desktop_descriptor_heap_type_count == D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES);
UINT __stdcall device_descriptor_increment_size(Wrapper* self, UINT type)
{
    if (!x2w_startup::is_valid_desktop_descriptor_heap_type(type)) {
        static LONG calls;
        if (first_calls(calls, 1)) log("x2w-d3d12 Device::slot15 out-of-range heap type 0x%x -> 0", type);
        return 0;
    }
    return static_cast<UINT>(forward_slot<device_kind, 15>(self, type, 0, 0, 0, 0, 0, 0, 0));
}

// Device slots 34/35, desktop MakeResident/Evict(count, objects).
template <int N>
HRESULT __stdcall device_residency(Wrapper* self, UINT count, void* const* objects)
{
    std::vector<ID3D12Pageable*> real;
    real.reserve(count);
    for (UINT i = 0; i < count; ++i) {
        retire_leave_class(objects[i], RetireLeft::residency);  // no hold tracks the residency of a desktop object
        if (auto* p = reinterpret_cast<ID3D12Pageable*>(unwrap(reinterpret_cast<u64>(objects[i]))))
            real.push_back(p);
        if (auto* r = reinterpret_cast<ID3D12Resource*>(unwrap(reinterpret_cast<u64>(objects[i])))) residency_title_touched(r);
    }
    if (real.empty()) return S_OK;
    const UINT active = static_cast<UINT>(real.size());
    return N == 34 ? device_of(self)->MakeResident(active, real.data()) : device_of(self)->Evict(active, real.data());
}

// ---------------------------------------------------------------------------------------------
// X2W_D3D12_RENDER=1: first stage of real rendering. Command lists are really recorded for the
// calls implemented below (reset, close, render-target clears), really submitted, fences complete
// when the desktop GPU finishes, and the presented surface is read back and shown in the window.
// Everything else on a list is still dropped.
void release_targets(Wrapper* self);
void end_open_queries(Wrapper* self, ID3D12GraphicsCommandList* list);
void reset_list_dynamic_state(Wrapper* self);
void discard_list_gpu_writes(Wrapper* self);
void commit_list_gpu_writes(Wrapper* self);
void discard_draw_sampler_heaps(ID3D12CommandList* list);
void discard_address_uploads(ID3D12CommandList* list);
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
void discard_red_producer(ID3D12GraphicsCommandList* list);
x2w_startup::DrawReadbacks& draw_dumps()
{
    static x2w_startup::DrawReadbacks tracker;
    return tracker;
}

LONG render_resets, render_closes, render_clears, render_executes, render_reset_failures;
// X2W_GPU_TIME=1: GPU time per frame of the title's direct queue (gpu_time_probe.h). Frame = the present count.
LONG gpu_time_frame;
x2w_gpu_time::Probe& gpu_time_probe()
{
    static x2w_gpu_time::Probe probe;
    return probe;
}
bool gpu_time_on()
{
    static const bool on = x2w_local::env_on("X2W_GPU_TIME");
    return on;
}
// OMSetRenderTargets calls that bound targets, and render target / depth handles that resolved to no view.
LONG render_target_binds, render_target_unresolved;

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
    std::vector<std::unique_ptr<x2w_upload::Arena>> uploads;
    std::vector<x2w_startup::ComPtr<ID3D12Resource>> indexed_validation;
    // Rectangle-list argument scratch of the recordings on this allocator. Retained with the recording, so until every
    // queue that ran it has passed its fence (the allocator is reused only when idle).
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> scratches;
};
struct TitleAllocator {
    std::vector<DesktopAllocator*> pool;
    DesktopAllocator* current = nullptr;
    D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT;  // from the lists recorded on it
};
SRWLOCK allocator_lock = SRWLOCK_INIT;
std::unordered_map<Wrapper*, TitleAllocator> title_allocators;
std::unordered_map<Wrapper*, DesktopAllocator*> list_recordings;  // list -> allocator holding its recording
std::unordered_map<ID3D12CommandQueue*, UploadQueue<ID3D12Fence>> queue_submits;
std::vector<UploadRingChunk> upload_chunks;  // allocator_lock also protects recording holds and queue fence values

LONG allocator_swaps, allocator_creates;

void release_upload_recording(Wrapper* list)
{
    AcquireSRWLockExclusive(&allocator_lock);
    discard_upload_chunks(list->own().upload_chunks, upload_chunks.data());
    ReleaseSRWLockExclusive(&allocator_lock);
}

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

// List slot 148 (Xbox order), allocate CPU-writable upload memory for this recording.
void* __stdcall list_allocate_upload(Wrapper* self, UINT64 bytes, UINT64 alignment)
{
    if (!bytes || !alignment || (alignment & (alignment - 1)) ||
        bytes > UINT64_MAX - (std::max)(alignment, UINT64{16})) return nullptr;
    x2w_upload::Allocation allocation{};
    AcquireSRWLockExclusive(&allocator_lock);
    auto& state = self->own();
    const auto found = list_recordings.find(self);
    if (state.list_open && state.list_device && found != list_recordings.end() && found->second) {
        try {
            if (!state.list_uploads) {
                auto arena = std::make_unique<x2w_upload::Arena>();
                found->second->uploads.push_back(std::move(arena));
                state.list_uploads = found->second->uploads.back().get();
            }
            allocation = state.list_uploads->allocate(device_of(state.list_device), bytes, alignment);
        } catch (const std::bad_alloc&) {
            // Refuse the allocation without invalidating any range already recorded.
        }
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    return reinterpret_cast<void*>(allocation.cpu);
}

// The rectangle-list argument scratch of this recording: created on first use, owned by the recording's desktop
// allocator (so it outlives every queue fence of the recording) and never shared with another recording.
ID3D12Resource* recording_rect_scratch(Wrapper* self)
{
    ID3D12Resource* scratch = nullptr;
    AcquireSRWLockExclusive(&allocator_lock);
    auto& state = self->own();
    const auto found = list_recordings.find(self);
    if (state.list_open && state.list_device && found != list_recordings.end() && found->second) {
        if (!state.list_scratch) {
            try {
                auto created = x2w_startup::RectIndirectArguments::make_scratch(device_of(state.list_device));
                if (created) {
                    found->second->scratches.push_back(std::move(created));
                    state.list_scratch = found->second->scratches.back().Get();
                }
            } catch (const std::bad_alloc&) {
                // No scratch: the caller refuses the draw.
            }
        }
        scratch = state.list_scratch;
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    return scratch;
}

// Keeps `resource` with this recording's desktop allocator, like the rectangle-list scratch above. False if the list
// is not recording (the caller must not use the resource then).
bool retain_with_recording(Wrapper* self, x2w_startup::ComPtr<ID3D12Resource> resource)
{
    bool retained = false;
    AcquireSRWLockExclusive(&allocator_lock);
    const auto& state = self->own();
    const auto found = list_recordings.find(self);
    if (state.list_open && found != list_recordings.end() && found->second) {
        try {
            found->second->scratches.push_back(std::move(resource));
            retained = true;
        } catch (const std::bad_alloc&) {
        }
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    return retained;
}

// Fixture query: how many scratch buffers the desktop allocator of this list's latest recording still holds (the
// rectangle-list scratch and the split indirect Dispatch records). They are released only when that allocator is
// reset after every queue that ran its recordings has passed its fence (allocator_reset).
extern "C" __declspec(dllexport) UINT x2w_d3d12_recording_scratches(void* guest_list)
{
    Wrapper* self = as_wrapper(guest_list);
    if (!self || self->own().kind != list_kind) return 0;
    AcquireSRWLockShared(&allocator_lock);
    const auto found = list_recordings.find(self);
    const UINT held = found != list_recordings.end() && found->second ? static_cast<UINT>(found->second->scratches.size()) : 0;
    ReleaseSRWLockShared(&allocator_lock);
    return held;
}

// Fixture query for two closed recordings before either is submitted.
extern "C" __declspec(dllexport) BOOL x2w_d3d12_distinct_rect_scratch(void* first, void* second)
{
    const Wrapper* a = as_wrapper(first);
    const Wrapper* b = as_wrapper(second);
    if (!a || !b || a == b || a->own().kind != list_kind || b->own().kind != list_kind) return FALSE;
    AcquireSRWLockShared(&allocator_lock);
    const auto& x = a->own();
    const auto& y = b->own();
    const bool distinct = x.list_ready && y.list_ready && x.list_scratch && y.list_scratch &&
                          x.list_scratch != y.list_scratch;
    ReleaseSRWLockShared(&allocator_lock);
    return distinct;
}

x2w_upload::Allocation list_upload_range(Wrapper* self, UINT64 address, UINT64 bytes)
{
    AcquireSRWLockShared(&allocator_lock);
    const auto& state = self->own();
    const auto allocation = state.list_open && state.list_uploads ? state.list_uploads->range(address, bytes) : x2w_upload::Allocation{};
    ReleaseSRWLockShared(&allocator_lock);
    return allocation;
}
UINT64 list_upload_gpu(Wrapper* self, UINT64 address, UINT64 bytes) { return list_upload_range(self, address, bytes).gpu; }

// Fixture lookup. The full range must belong to this list's open recording.
extern "C" __declspec(dllexport) UINT64 x2w_d3d12_upload_gpu_address(void* list, const void* cpu, UINT64 bytes)
{
    Wrapper* self = as_wrapper(list);
    return self && self->own().kind == list_kind && cpu && bytes
        ? list_upload_gpu(self, reinterpret_cast<UINT64>(cpu), bytes) : 0;
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
            next->uploads.clear();
            next->indexed_validation.clear();
            next->scratches.clear();
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
    auto& submit = queue_submits[queue];
    auto& fence = submit.fence;
    auto& value = submit.value;
    if (!fence && !submit_fence_fails()) {
        ID3D12Device* device = nullptr;
        if (SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&device)))) {
            if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) fence = nullptr;
            device->Release();
        }
    }
    UploadChunkMask chunks = 0;
    for (auto* list : lists) chunks |= list->own().upload_chunks;
    const bool signaled = fence && SUCCEEDED(queue->Signal(fence, value + 1));
    if (signaled) ++value;
    submit.submitted(chunks, signaled);
    if (signaled) {
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

// Fixtures (X2W_D3D12_LINEAR_BUFFERS): how slot 133 backed a linear buffer (false: the default backing, or
// not a placed buffer), and the totals since the process started.
extern "C" __declspec(dllexport) bool x2w_d3d12_linear_buffer_info(void* resource, X2wLinearBufferInfo* info)
{
    Wrapper* w = as_wrapper(resource);
    if (!info || !w || w->own().kind != resource_kind || !w->own().real ||
        x2w_startup::linear_buffer_mode() == x2w_startup::LinearBufferMode::off)
        return false;
    return linear_buffers().info(static_cast<ID3D12Resource*>(w->own().real), info);
}
extern "C" __declspec(dllexport) void x2w_d3d12_linear_buffer_totals(X2wLinearBufferTotals* totals)
{
    if (!totals) return;
    *totals = X2wLinearBufferTotals{};
    if (x2w_startup::linear_buffer_mode() != x2w_startup::LinearBufferMode::off) linear_buffers().totals(totals);
}
// Fixtures: the next refresh that reaches `point` (0 before staging, 1 after the title bytes are staged,
// 2 after its copy was executed, before it is published), or for point 3 the next title Wait (queue slot 17)
// after its desktop Wait and before it is recorded, sets the `reached` event, then waits up to 20 s for
// `resume`. Null events disarm the point. False with the switch off or for an unknown point.
extern "C" __declspec(dllexport) bool x2w_d3d12_linear_buffer_test_pause(unsigned point, void* reached, void* resume)
{
    return x2w_startup::linear_buffer_mode() != x2w_startup::LinearBufferMode::off &&
           linear_buffers().arm_pause(point, static_cast<HANDLE>(reached), static_cast<HANDLE>(resume));
}
// Fixtures: the internal queue that runs every refresh (AddRef'd), so a fixture can hold refreshes in
// flight behind a fence of its own. False before the first copied placement.
extern "C" __declspec(dllexport) bool x2w_d3d12_linear_buffer_internal_queue(void** queue)
{
    if (!queue) return false;
    *queue = x2w_startup::linear_buffer_mode() == x2w_startup::LinearBufferMode::off ? nullptr : linear_buffers().internal_queue();
    return *queue != nullptr;
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
                    spare->uploads.clear();
                    spare->indexed_validation.clear();
                    spare->scratches.clear();
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
            discard_upload_chunks(state.upload_chunks, upload_chunks.data());
            state.list_uploads = nullptr;
            state.list_scratch = nullptr;
            state.list_ready = false;
            state.list_open = true;
        }
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    if (SUCCEEDED(hr)) {
        reset_list_dynamic_state(self);
        discard_list_gpu_writes(self);
        discard_draw_sampler_heaps(previous);
        discard_address_uploads(previous);
        discard_mesh_lease_recording(previous);
        retire_drop_recording(self);
        if (draw_dump_frame()) draw_dumps().discard(previous);
        discard_red_producer(previous);
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
    end_open_queries(self, list);
    release_targets(self);
    const HRESULT hr = list->Close();
    state.list_open = false;
    state.list_ready = SUCCEEDED(hr);
    state.list_close_failed = FAILED(hr);
    if (FAILED(hr)) {
        discard_upload_chunks(state.upload_chunks, upload_chunks.data());
        if (const auto found = list_recordings.find(self); found != list_recordings.end() && found->second)
            found->second->close_failed = true;
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    InterlockedIncrement(&render_closes);
    if (FAILED(hr)) {
        discard_list_gpu_writes(self);
        discard_draw_sampler_heaps(list);
        discard_address_uploads(list);
        discard_mesh_lease_recording(list);
        retire_drop_recording(self);
        if (draw_dump_frame()) draw_dumps().discard(list);
        discard_red_producer(list);
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
// raw_plane < 0: the baseline capture of the presented surface (plane 0), reduced for the window and
// logged as `x2w-render presented`. raw_plane >= 0 (X2W_PRESENT_COMPOSITE=1 only): a sample of that
// plane's own surface, before compositing, logged as `x2w-render raw plane <n>`; it leaves the window
// copy, the note and the dump alone. Neither is the composited window picture.
void read_back_presented(ID3D12CommandQueue* queue, ID3D12Resource* surface, int raw_plane = -1)
{
    const bool raw = raw_plane >= 0;
    char failure[160] = "";
    const auto report_failure = [&] {
        if (!raw) {
            std::snprintf(shown_note, sizeof(shown_note), "%s", failure);
            return;
        }
        static LONG raw_failures;
        if (first_calls(raw_failures, 4)) log("x2w-render raw plane %d (not composited): %s", raw_plane, failure);
    };
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
        std::snprintf(failure, sizeof(failure), "presented surface format %d not shown yet", desc.Format);
        report_failure();
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
        if (SUCCEEDED(hr)) x2w_budget::committed(device, readback, D3D12_HEAP_TYPE_READBACK);
        readback_bytes = SUCCEEDED(hr) ? total : 0;
    }
    if (SUCCEEDED(hr)) hr = allocator->Reset();
    if (SUCCEEDED(hr)) hr = list->Reset(allocator, nullptr);
    if (FAILED(hr)) {
        std::snprintf(failure, sizeof(failure), "readback setup failed 0x%08lx", static_cast<unsigned long>(hr));
        report_failure();
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
        std::snprintf(failure, sizeof(failure), "readback timed out");
        report_failure();
        return;
    }
    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(total)};
    if (FAILED(readback->Map(0, &range, &mapped))) return;
    if (!raw) {
        AcquireSRWLockExclusive(&shown_lock);
        shown_pixels.resize(shown_width * shown_height);
    }
    std::uint64_t sum = 0, alpha_sum = 0;
    for (int y = 0; y < shown_height; ++y) {
        const auto* row = static_cast<const unsigned char*>(mapped) + footprint.Offset +
                          static_cast<UINT64>(y) * desc.Height / shown_height * footprint.Footprint.RowPitch;
        for (int x = 0; x < shown_width; ++x) {
            const unsigned char* texel = row + static_cast<UINT64>(x) * desc.Width / shown_width * 4;
            const unsigned r = rgba ? texel[0] : texel[2], g = texel[1], b = rgba ? texel[2] : texel[0];
            if (!raw) shown_pixels[y * shown_width + x] = (r << 16) | (g << 8) | b;
            sum += r + g + b;
            alpha_sum += texel[3];
        }
    }
    if (!raw) ReleaseSRWLockExclusive(&shown_lock);
    // X2W_D3D12_DUMP_SURFACE=<file>: every 40th read-back is also written at full resolution as a
    // 32-bit BMP (overwritten each time), for looking at detail the reduced overlay copy loses.
    {
        static wchar_t dump_path[MAX_PATH];
        static const bool dump = GetEnvironmentVariableW(L"X2W_D3D12_DUMP_SURFACE", dump_path, MAX_PATH) != 0;
        if (!raw && dump && (shown_frames % 4) == 3 && desc.Width <= 8192 && desc.Height <= 8192) {
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
    if (raw) {
        static LONG raw_frames;
        const LONG count = InterlockedIncrement(&raw_frames);
        if ((count > 0 && count <= 3) || count % 40 == 0)
            log("x2w-render raw plane %d (not composited): surface %llux%u format %d, sample %ld, mean level %.2f/255, mean alpha %.2f/255",
                raw_plane, desc.Width, desc.Height, desc.Format, count, sum / (3.0 * shown_width * shown_height),
                alpha_sum / (1.0 * shown_width * shown_height));
        return;
    }
    const LONG count = InterlockedIncrement(&shown_frames);
    std::snprintf(shown_note, sizeof(shown_note), "surface %llux%u format %d, readback %ld, mean level %.2f/255",
                  desc.Width, desc.Height, desc.Format, count, sum / (3.0 * shown_width * shown_height));
    // Also whenever the picture's mean level moved by more than 2 since the last line (a screen change shows in the log
    // with its time; at most 300 such lines), not only for the first 3 readbacks and every 40th.
    static double last_logged_mean = -1000;
    static LONG change_lines;
    const double mean_level = sum / (3.0 * shown_width * shown_height);
    const bool periodic = (count > 0 && count <= 3) || count % 40 == 0;
    const bool moved = std::fabs(mean_level - last_logged_mean) > 2 && change_lines < 300;
    if (periodic || moved) {
        if (moved && !periodic) {
            ++change_lines;
            log("x2w-render presented %s (mean level changed by %.2f since the last line)", shown_note, mean_level - last_logged_mean);
        } else {
            log("x2w-render presented %s", shown_note);
        }
        last_logged_mean = mean_level;
    }
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
    UINT constants_count{};
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
    if (!w) {
        root->Release();
        return E_OUTOFMEMORY;
    }
    note_root_blob(w, blob, size);
    std::vector<RootParameter> parameters;
    UINT heap_flags = 0;
    ID3D12VersionedRootSignatureDeserializer* deserializer = nullptr;
    if (SUCCEEDED(D3D12CreateVersionedRootSignatureDeserializer(blob, size, IID_PPV_ARGS(&deserializer)))) {
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc = nullptr;
        if (SUCCEEDED(deserializer->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &desc)) && desc) {
            heap_flags = static_cast<UINT>(desc->Desc_1_1.Flags) & x2w_startup::root_flags_heap_indexed;
            w->own().root_without_bindings = !heap_flags && !desc->Desc_1_1.NumParameters && !desc->Desc_1_1.NumStaticSamplers;
            w->own().root_srv_only = !heap_flags && !desc->Desc_1_1.NumStaticSamplers && desc->Desc_1_1.NumParameters;
            for (UINT i = 0; i < desc->Desc_1_1.NumParameters; ++i) {
                const auto& parameter = desc->Desc_1_1.pParameters[i];
                if (parameter.ParameterType != D3D12_ROOT_PARAMETER_TYPE_SRV) w->own().root_srv_only = false;
                RootParameter info{parameter.ParameterType, 0, 0};
                if (parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_CBV || parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_SRV ||
                    parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_UAV) {
                    info.shader_register = parameter.Descriptor.ShaderRegister;
                    info.space = parameter.Descriptor.RegisterSpace;
                } else if (parameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                    info.shader_register = parameter.Constants.ShaderRegister;
                    info.space = parameter.Constants.RegisterSpace;
                    info.constants_count = parameter.Constants.Num32BitValues;
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

// Fixtures: the copy of a root signature's blob kept for shader translation (the .root request and the
// translation key). Returns its size and copies up to capacity bytes to out; 0 for anything else. The
// heap-indexing flags slot 34 acts on (root_heap_flags, read from the desktop blob) go to heap_flags.
extern "C" __declspec(dllexport) std::size_t x2w_d3d12_root_translation_blob(void* root, void* out, std::size_t capacity, UINT* heap_flags)
{
    Wrapper* w = as_wrapper(root);
    if (!w || w->own().kind != root_signature_kind) return 0;
    if (heap_flags) *heap_flags = root_heap_flags(w);
    AcquireSRWLockShared(&note_lock);
    const auto found = root_blobs().find(w);
    const std::size_t bytes = found == root_blobs().end() ? 0 : found->second.size();
    if (bytes && out) std::memcpy(out, found->second.data(), bytes < capacity ? bytes : capacity);
    ReleaseSRWLockShared(&note_lock);
    return bytes;
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

enum class DxilMeshStage : unsigned { pixel = 0, mesh = 13, amplification = 14, unknown = 0xffff };

DxilMeshStage desktop_dxil_mesh_stage(const D3D12_SHADER_BYTECODE& code)
{
    if (!readable_shader_container(code)) return DxilMeshStage::unknown;
    const auto* data = static_cast<const unsigned char*>(code.pShaderBytecode);
    const auto word = [data](std::size_t offset) {
        UINT value;
        std::memcpy(&value, data + offset, sizeof(value));
        return value;
    };
    const std::size_t bytes = code.BytecodeLength;
    const std::size_t parts = word(28);
    if (word(24) != bytes || parts > (bytes - 32) / 4) return DxilMeshStage::unknown;
    const std::size_t header_size = 32 + parts * 4;
    DxilMeshStage stage = DxilMeshStage::unknown;
    bool found_dxil = false;
    for (std::size_t i = 0; i < parts; ++i) {
        const std::size_t part = word(32 + i * 4);
        if (part < header_size || part > bytes - 8) return DxilMeshStage::unknown;
        const std::size_t size = word(part + 4);
        if (size > bytes - part - 8) return DxilMeshStage::unknown;
        if (!std::memcmp(data + part, "XDAT", 4)) return DxilMeshStage::unknown;
        if (std::memcmp(data + part, "DXIL", 4)) continue;
        if (found_dxil || size < 24) return DxilMeshStage::unknown;
        found_dxil = true;
        const std::size_t program = part + 8;
        const std::size_t bitcode = word(program + 16);
        const std::size_t bitcode_size = word(program + 20);
        if (word(program + 4) * 4ull != size || std::memcmp(data + program + 8, "DXIL", 4) ||
            bitcode < 16 || bitcode > size - 8 || bitcode_size < 4 || bitcode_size > size - 8 - bitcode ||
            std::memcmp(data + program + 8 + bitcode, "BC\xc0\xde", 4)) return DxilMeshStage::unknown;
        const UINT version = word(program);
        if (((version >> 4) & 0xf) != 6) return DxilMeshStage::unknown;
        if (version >> 16 == static_cast<UINT>(DxilMeshStage::pixel)) stage = DxilMeshStage::pixel;
        if (version >> 16 == static_cast<UINT>(DxilMeshStage::mesh)) stage = DxilMeshStage::mesh;
        if (version >> 16 == static_cast<UINT>(DxilMeshStage::amplification)) stage = DxilMeshStage::amplification;
    }
    return stage;
}

std::size_t desktop_stream_size(UINT type)
{
    if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS || type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS) return 24;
    return type < std::size(stream_sizes) ? stream_sizes[type] : 0;
}

// Converts the supported stream layout to desktop form and creates the pipeline.
ID3D12PipelineState* build_stream_pipeline(const D3D12_PIPELINE_STATE_STREAM_DESC* desc, char* why, std::size_t why_size,
                                          bool& compute, IndirectGraphicsPipeline& graphics, ID3D12RootSignature*& pipeline_root,
                                          ID3D12RootSignature* cached_root = nullptr)
{
    const bool cached_mesh = cached_root != nullptr;
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
    bool amplification = false, console_mesh = false;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
    D3D12_RT_FORMAT_ARRAY targets{};
    DXGI_FORMAT depth = DXGI_FORMAT_UNKNOWN;
    DXGI_SAMPLE_DESC samples{1, 0};
    bool depth_enabled = false, depth_writes = false, untracked_depth_state = false;
    bool ok = true;
    // First pass: validate the layout and find the wrapped root signature.
    for (std::size_t offset = 0; offset < bytes;) {
        auto& type = *reinterpret_cast<UINT*>(stream + offset);
        const auto size = desktop_stream_size(type);
        if (!size || size > bytes - offset) {
            ok = false;
            std::snprintf(why, why_size, "unknown or truncated subobject %u at +0x%zx", type, offset);
            break;
        }
        if (type == 0) root = as_wrapper(*reinterpret_cast<void**>(stream + offset + 8));
        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS || type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS) {
            const auto& code = *reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + offset + 8);
            if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS) {
                // Xbox MS and desktop AS share tag 24. Classify the actual desktop DXIL before rewriting.
                const auto stage = desktop_dxil_mesh_stage(code);
                if (stage == DxilMeshStage::mesh && !mesh && !amplification) {
                    type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS;
                    console_mesh = true;
                } else if (stage == DxilMeshStage::amplification && !amplification && !console_mesh) {
                    amplification = true;
                } else {
                    std::snprintf(why, why_size, "ambiguous or unsupported DXIL for shared mesh tag at +0x%zx", offset);
                    ok = false;
                    break;
                }
            }
            if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS) {
                if (mesh) {
                    std::snprintf(why, why_size, "duplicate mesh stage at +0x%zx", offset);
                    ok = false;
                    break;
                }
                mesh = true;
            }
        }
        offset += size;
    }
    const bool empty_root = root && root->own().kind == root_signature_kind && root->own().root_without_bindings;
    const bool native_mesh_root = empty_root || cached_mesh;
    for (std::size_t offset = 0; ok && offset < bytes;) {
        const UINT type = *reinterpret_cast<UINT*>(stream + offset);
        if (type == 0) {
            u64& pointer = *reinterpret_cast<u64*>(stream + offset + 8);
            pointer = cached_mesh ? reinterpret_cast<u64>(cached_root) : unwrap(pointer);
        } else if (type >= 1 && type <= 5) {
            const auto* code = reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + offset + 8);
            if (code->pShaderBytecode) {
                const bool is_vertex = type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS;
                bool supported = false;
                if ((mesh ? native_mesh_root : empty_root) && type <= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS) {
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
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS || type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS) {
            const auto* code = reinterpret_cast<const D3D12_SHADER_BYTECODE*>(stream + offset + 8);
            if (!native_mesh_root || !readable_shader_container(*code)) {
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
        } else if (cached_mesh && (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL ||
                                  type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1)) {
            D3D12_DEPTH_STENCIL_DESC1 state{};
            const auto size = type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL ?
                sizeof(D3D12_DEPTH_STENCIL_DESC) : sizeof(state);
            std::memcpy(&state, stream + offset + sizeof(UINT), size);
            depth_enabled = state.DepthEnable;
            depth_writes = depth_enabled && state.DepthWriteMask != D3D12_DEPTH_WRITE_MASK_ZERO;
            untracked_depth_state = state.StencilEnable || state.DepthBoundsTestEnable;
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC) {
            samples = *reinterpret_cast<const DXGI_SAMPLE_DESC*>(stream + offset + 4);
        }
        offset += desktop_stream_size(type);
    }
    if (ok && cached_mesh && (untracked_depth_state || targets.NumRenderTargets > 8 ||
        (targets.NumRenderTargets ? !pixel : !depth_writes) ||
        (depth_enabled && depth == DXGI_FORMAT_UNKNOWN) ||
        std::any_of(targets.RTFormats, targets.RTFormats + std::min<UINT>(targets.NumRenderTargets, 8),
            [](DXGI_FORMAT value) { return value == DXGI_FORMAT_UNKNOWN; }))) {
        std::snprintf(why, why_size, "cached mesh needs tracked color attachments or writable depth-only state");
        ok = false;
    }
    if (ok && (vertex || pixel || mesh || amplification) &&
        ((!vertex && !mesh) || (amplification && !mesh) || (!pixel && !cached_mesh) || compute || input_layout ||
         topology != D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE ||
         (!cached_mesh && (targets.NumRenderTargets != 1 || targets.RTFormats[0] == DXGI_FORMAT_UNKNOWN ||
                           depth != DXGI_FORMAT_UNKNOWN)) ||
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
    if (pipeline && compute && root && root->own().kind == root_signature_kind)
        pipeline_root = static_cast<ID3D12RootSignature*>(root->own().real);
    if (pipeline && (vertex || mesh) && (pixel || cached_mesh)) {
        graphics = {cached_mesh ? cached_root : static_cast<ID3D12RootSignature*>(root->own().real),
            targets.RTFormats[0], mesh, cached_mesh};
        if (cached_mesh) {
            graphics.cached_target_count = targets.NumRenderTargets;
            std::copy_n(targets.RTFormats, targets.NumRenderTargets, graphics.cached_targets);
            graphics.cached_depth = depth;
            graphics.cached_depth_writes = depth_writes;
            graphics.guest_root_params = root_info(root);
        }
    }
    return pipeline;
}

// Transient guest data is copied into mapped upload chunks. Chunks grow from
// 16 MiB up to 256 MiB in total and stay alive for the process. A full chunk can be reused once
// no recording can submit its bytes and every queue that used it has finished.

volatile LONG64 upload_ring_reserved_bytes;
volatile LONG upload_ring_reservations, upload_ring_refusals, upload_ring_recycled;

struct UploadRingStats {
    unsigned long long reserved_bytes = 0, capacity = 0;
    long reservations = 0, refusals = 0, recycled = 0;
    std::size_t chunks = 0;
};
UploadRingStats upload_ring_stats()
{
    UploadRingStats stats;
    stats.reserved_bytes = static_cast<unsigned long long>(upload_ring_reserved_bytes);
    stats.reservations = upload_ring_reservations;
    stats.refusals = upload_ring_refusals;
    stats.recycled = upload_ring_recycled;
    AcquireSRWLockShared(&allocator_lock);
    stats.chunks = upload_chunks.size();
    for (const auto& chunk : upload_chunks) stats.capacity += chunk.capacity;
    ReleaseSRWLockShared(&allocator_lock);
    return stats;
}

UploadReservation reserve_upload_ring(Wrapper* list, std::size_t bytes)
{
    constexpr u64 max_alloc_bytes = max_upload_ring_bytes / 4;
    UploadReservation reservation;
    if (bytes == 0 || bytes > max_alloc_bytes) return reservation;
    const u64 rounded = (static_cast<u64>(bytes) + 511) & ~u64{511};

    auto& chunks = upload_chunks;
    static unsigned active = 0;
    static u64 total_bytes = 0;

    AcquireSRWLockExclusive(&allocator_lock);
    const auto finish = [&](unsigned index) {
        active = index;
        InterlockedAdd64(&upload_ring_reserved_bytes, static_cast<LONG64>(rounded));
        InterlockedIncrement(&upload_ring_reservations);
        chunks[index].hold(list->own().upload_chunks, index);
        ReleaseSRWLockExclusive(&allocator_lock);
        return reservation;
    };
    if (!chunks.empty()) {
        reservation = reserve_upload_chunk(chunks[active], rounded);
        if (reservation.mapped) return finish(active);
    }

    const u64 remaining = total_bytes < max_upload_ring_bytes ? max_upload_ring_bytes - total_bytes : 0;
    if (rounded <= remaining) {
        u64 chunk_bytes = chunks.empty() ? initial_upload_chunk_bytes : chunks.back().capacity * 2;
        if (chunk_bytes < rounded) chunk_bytes = rounded;
        if (chunk_bytes > remaining) chunk_bytes = remaining;
        auto* device = the_device ? device_of(the_device) : nullptr;
        if (device) {
            const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = chunk_bytes;
            buffer.Height = 1;
            buffer.DepthOrArraySize = 1;
            buffer.MipLevels = 1;
            buffer.SampleDesc.Count = 1;
            buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            ID3D12Resource* created = nullptr;
            void* base = nullptr;
            if (SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&created)))) {
                auto* budget = x2w_budget::committed(device, created, D3D12_HEAP_TYPE_UPLOAD);
                if (SUCCEEDED(created->Map(0, nullptr, &base))) {
                    try {
                        chunks.push_back(UploadRingChunk{created, static_cast<unsigned char*>(base), chunk_bytes, 0});
                        total_bytes += chunk_bytes;
                        reservation = reserve_upload_chunk(chunks.back(), rounded);
                        return finish(static_cast<unsigned>(chunks.size() - 1));
                    } catch (const std::bad_alloc&) {
                        created->Unmap(0, nullptr);
                        if (budget) budget->end_use();
                        created->Release();
                    }
                } else {
                    if (budget) budget->end_use();
                    created->Release();
                }
            }
        }
    }
    // Allocation failed or the cap was reached. Older unused tails are still safe.
    for (unsigned index = 0; index < chunks.size(); ++index) {
        reservation = reserve_upload_chunk(chunks[index], rounded);
        if (reservation.mapped) return finish(index);
    }

    for (unsigned index = 0; index < chunks.size(); ++index) {
        reservation = reuse_upload_chunk(chunks[index], index, rounded, queue_submits);
        if (reservation.mapped) {
            InterlockedIncrement(&upload_ring_recycled);
            return finish(index);
        }
    }

    ReleaseSRWLockExclusive(&allocator_lock);
    InterlockedIncrement(&upload_ring_refusals);
    static LONG refused;
    if (first_calls(refused, 8))
        log("x2w-draw upload ring reservation refused: no unused capacity");
    return reservation;
}

unsigned char* reserve_upload_ring(Wrapper* list, std::size_t bytes, u64* offset, ID3D12Resource** resource)
{
    if (resource) *resource = nullptr;
    if (offset) *offset = 0;
    const UploadReservation reservation = reserve_upload_ring(list, bytes);
    if (!reservation.mapped) return nullptr;
    if (offset) *offset = reservation.offset;
    if (resource) *resource = reservation.resource;
    return reservation.mapped;
}

bool ring_copy(Wrapper* list, u64 title_address, std::size_t bytes, u64* offset, ID3D12Resource** resource = nullptr)
{
    auto* dest = reserve_upload_ring(list, bytes, offset, resource);
    if (!dest) return false;
    std::memcpy(dest, reinterpret_cast<const void*>(title_address), bytes);
    return true;
}

D3D12_GPU_VIRTUAL_ADDRESS upload_from_title(Wrapper* list, u64 title_address, std::size_t bytes)
{
    u64 offset = 0;
    ID3D12Resource* resource = nullptr;
    return ring_copy(list, title_address, bytes, &offset, &resource) && resource ? resource->GetGPUVirtualAddress() + offset : 0;
}

// Number of contiguous readable bytes at address, up to wanted.
std::size_t readable_window(u64 address, std::size_t wanted)
{
    return x2w_local::probe_readable(static_cast<std::uintptr_t>(address), wanted);
}

// The opt-in slot 62 trace records setter arguments without forwarding the calls.
struct ComputeBindingTrace {
    struct Parameter {
        UINT setter{};           // The most recent non-constant setter for this parameter.
        u64 argument{};          // Address or handle. It is classified at dispatch, never printed.
        u64 constants{};         // Dwords covered by constant setters, within the first 64.
    };
    const Wrapper* root{};
    std::array<Parameter, 16> parameters{};
};

// Per-list state while recording.
struct Pass {
    struct CopyRange { ID3D12Resource* resource; u64 offset, bytes; };
    std::vector<CopyRange> copied_gpu_ranges; // exact CopyBufferRegion destinations on this recording
    bool pipeline_real = false;      // the bound pipeline has a desktop pipeline
    bool compute_pipeline = false;
    ID3D12RootSignature* compute_root = nullptr; // latest guest root if it was forwarded
    x2w_startup::ComPtr<ID3D12RootSignature> desktop_compute_root; // last root actually set on this recording
    std::array<D3D12_GPU_VIRTUAL_ADDRESS, 64> desktop_compute_upload_srvs{};
    ID3D12RootSignature* compute_pipeline_root = nullptr; // root the bound compute pipeline was built against
    UINT compute_srv_bound = 0;                           // root SRV parameters that reached the desktop list since slot 33
    // Desktop root UAVs a fixture bound through x2w_d3d12_fixture_compute_root_uav, set again after a shim compute pass.
    std::array<D3D12_GPU_VIRTUAL_ADDRESS, 64> desktop_compute_root_uavs{};
    const std::vector<RootParameter>* compute_params = nullptr;
    std::unique_ptr<ComputeBindingTrace> compute_trace;
    bool compute_root_has_arguments = false;     // that root declares parameters other than root SRVs, or static samplers
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
    Wrapper* requested_pipeline = nullptr;  // last pipeline slot 61 asked a translation for; repeats skip draw_lock
    Wrapper* requested_mesh_pipeline = nullptr;
    const std::vector<RootParameter>* requested_mesh_root = nullptr;
    UINT topology = 0;
    x2w_startup::ViewTable view_tables[16]{};
    x2w_startup::ComputeTables compute_tables;  // slot 35, read by a Dispatch handler
    u64 source_view_heap = 0;
    x2w_startup::ViewTable title_view_heap;  // slot 32: the title memory of the list's view heap (heap records)
    u64 root_address[16] = {};       // title address of each root descriptor
    std::uint32_t constants[16][8] = {};
    unsigned char constant_words_bound[16] = {};
    x2w_root_constants::Store compute_constants;  // compute root constants (slots 37 and 39)
    Wrapper* compute_root_wrapper = nullptr;      // compute root signature last set by slot 33
    D3D12_VERTEX_BUFFER_VIEW vb[8] = {};
    PlacedVertexView placed_vb[8] = {};
    D3D12_INDEX_BUFFER_VIEW ib = {};
    DXGI_FORMAT target_formats[8] = {};
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
    bool targets_single_sample = false;
    bool depth_view_writable = false;
    UINT stencil_reference = 0;  // last guest OMSetStencilRef on this recording
    UINT target_count = 0;
    ID3D12Resource* target0 = nullptr;  // desktop resource of render target 0 (for per-draw dumps)
    ID3D12Resource* trace_targets[8]{};
    ID3D12Resource* trace_depth = nullptr;
    ID3D12Resource* bound_targets[8]{};  // the last accepted bind: targets [0, target_count) and the depth texture
    ID3D12Resource* bound_depth = nullptr;
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
    // Queries begun on the desktop list by slot 28 and not yet ended (heap, type, index). The heap pointer is
    // held without a reference until Close; releasing a heap mid-recording is invalid on desktop too.
    struct OpenQuery { ID3D12QueryHeap* heap; UINT type, index; };
    std::vector<OpenQuery> open_queries;
    // Queries ended on this recording by slot 29: what a resolve may cover. A new BeginQuery on the index removes its
    // entry, and Close clears the list with the rest of the Pass.
    std::vector<OpenQuery> ended_queries;
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

bool indirect_copy_ranges()
{
    static const bool enabled = x2w_local::env_on("X2W_TRANSLATED_INDIRECT_COPY_RANGES");
    return enabled;
}

bool list_copied_gpu_range(Wrapper* self, ID3D12Resource* resource, u64 offset, u64 bytes)
{
    if (!indirect_copy_ranges()) return false;
    const auto& ranges = pass_of(self).copied_gpu_ranges;
    return std::any_of(ranges.begin(), ranges.end(), [&](const Pass::CopyRange& range) {
        return range.resource == resource && offset >= range.offset && bytes <= range.bytes &&
            offset - range.offset <= range.bytes - bytes;
    });
}

void note_copy_gpu_range(Wrapper* self, ID3D12Resource* resource, u64 offset, u64 bytes, bool authoritative)
{
    if (!indirect_copy_ranges()) return;
    auto& ranges = pass_of(self).copied_gpu_ranges;
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [&](const Pass::CopyRange& range) {
        if (range.resource != resource) return false;
        return range.offset <= offset ? offset - range.offset < range.bytes : range.offset - offset < bytes;
    }), ranges.end());
    if (authoritative && ranges.size() < 64) {
        try { ranges.push_back({resource, offset, bytes}); }
        catch (const std::bad_alloc&) { return; }
        static LONG reports;
        if (first_calls(reports, 16)) log("x2w-render recorded GPU buffer copy range: offset 0x%llx, %llu bytes", offset, bytes);
    }
}
// Placed buffers that do not alias title memory (copied placement) and that a recorded copy has written. The title's memory
// never receives those bytes, so a draw that reads the buffer's range from title memory reads what the title wrote before the
// copy, or nothing. The ranges outlive the recording: the copy and the draw that reads it are often in different lists.
// A range is dropped with its wrapper's title address changing; a recording that is discarded still leaves its range here,
// which only makes a later draw read the buffer's own copy.
struct PlacedGpuWrite { u64 address, offset, bytes; };
SRWLOCK placed_gpu_writes_lock = SRWLOCK_INIT;
LONG placed_gpu_write_buffers = 0;
std::unordered_map<Wrapper*, std::vector<PlacedGpuWrite>>& placed_gpu_writes()
{
    static std::unordered_map<Wrapper*, std::vector<PlacedGpuWrite>> m;
    return m;
}
void note_placed_gpu_write(Wrapper* buffer, u64 offset, u64 bytes)
{
    if (!buffer || !buffer->own().title_memory || !bytes) return;
    const auto* info = resource_info(buffer);
    if (!info || info->placement != Placement::copied_buffer) return;
    const u64 address = reinterpret_cast<u64>(buffer->own().title_memory);
    AcquireSRWLockExclusive(&placed_gpu_writes_lock);
    try {
        auto& writes = placed_gpu_writes()[buffer];
        if (!writes.empty() && writes.front().address != address) writes.clear();  // a new buffer at a reused wrapper
        if (writes.size() < 64) writes.push_back({address, offset, bytes});
        placed_gpu_write_buffers = static_cast<LONG>(placed_gpu_writes().size());
    } catch (const std::bad_alloc&) {
    }
    ReleaseSRWLockExclusive(&placed_gpu_writes_lock);
}
// True when a recorded copy wrote any byte of [offset, offset + bytes) of the buffer.
bool placed_gpu_written(Wrapper* buffer, u64 offset, u64 bytes)
{
    if (!InterlockedCompareExchange(&placed_gpu_write_buffers, 0, 0)) return false;
    bool written = false;
    AcquireSRWLockShared(&placed_gpu_writes_lock);
    const auto found = placed_gpu_writes().find(buffer);
    if (found != placed_gpu_writes().end() && buffer->own().title_memory &&
        found->second.front().address == reinterpret_cast<u64>(buffer->own().title_memory)) {
        for (const auto& write : found->second)
            if (offset < write.offset + write.bytes && write.offset < offset + bytes) written = true;
    }
    ReleaseSRWLockShared(&placed_gpu_writes_lock);
    return written;
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

#include "startup_stubs/mesh_xmem_handoff.inl"
#include "startup_stubs/red_producer_diagnostic.inl"

void discard_red_producer(ID3D12GraphicsCommandList* list) { red_producers().discard(list); }
void release_targets(Wrapper* self)
{
    Pass& pass = pass_of(self);
    for (const auto& known : pass.states)
        transition(static_cast<ID3D12GraphicsCommandList*>(self->own().real), known.first, known.second, D3D12_RESOURCE_STATE_COMMON);
    pass = Pass{};
}

// A dropped recording (Reset, or its list destroyed) names no resource any more. The raw pointers the pass keeps for
// Close go first, then the holds that kept those resources alive (X2W_D3D12_RELEASE_RETIRED). Without the switch
// nothing changes.
void retire_drop_recording(Wrapper* list)
{
    residency_discard_list(list);
    if (!release_retired_enabled()) return;
    if (list->own().pass) pass_of(list).states.clear();
    retire_discard_holds(list);
}

// A reset list has no viewport or scissor and uses the default stencil reference.
void reset_list_dynamic_state(Wrapper* self)
{
    if (!self->own().pass) return;
    pass_of(self).stencil_reference = 0;
    pass_of(self).viewport_set = false;
    pass_of(self).scissor_set = false;
    pass_of(self).default_viewport_w = pass_of(self).default_scissor_w = 0;
    pass_of(self).default_viewport_h = pass_of(self).default_scissor_h = 0;
}

// List slot 20, IASetPrimitiveTopology. 0x11 is a console-only rectangle list: translated draws and indirect
// non-indexed draws expand it to triangles; the topology is not forwarded as such.
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
    pass.compute_pipeline_root = pass.compute_pipeline ? w->own().pipeline_root : nullptr;
    pass.pipeline = w && w->own().kind == pipeline_kind ? w : nullptr;
    InterlockedIncrement(pass.pipeline_real ? &render_pipelines_real : &render_pipelines_missing);
    if (auto* list = recording(self); list && pass.pipeline_real) list->SetPipelineState(static_cast<ID3D12PipelineState*>(w->own().real));
}
// List slot 33 (desktop 29 + 4), SetComputeRootSignature.
// Slot 26: the console's UAV records have the desktop layout, with wrapped resource pointers.
// Desktop transitions are owned by the shim, so the other barrier types are left unhandled.
UINT __stdcall list_resource_barriers(Wrapper* self, UINT count, const D3D12_RESOURCE_BARRIER* barriers)
{
    static_assert(sizeof(D3D12_RESOURCE_BARRIER) == 32);
    UINT forwarded = 0;
    auto* list = recording(self);
    const auto type = list ? list->GetType() : D3D12_COMMAND_LIST_TYPE_COPY;
    if ((type == D3D12_COMMAND_LIST_TYPE_DIRECT || type == D3D12_COMMAND_LIST_TYPE_COMPUTE) &&
        count && barriers && x2w_local::readable(barriers, std::size_t{count} * sizeof(*barriers))) {
        for (UINT i = 0; i < count; ++i) {
            D3D12_RESOURCE_BARRIER barrier;
            std::memcpy(&barrier, barriers + i, sizeof(barrier));
            if (barrier.Type != D3D12_RESOURCE_BARRIER_TYPE_UAV || barrier.Flags != D3D12_RESOURCE_BARRIER_FLAG_NONE)
                continue;
            if (barrier.UAV.pResource) {
                Wrapper* resource = as_wrapper(barrier.UAV.pResource);
                if (!resource || resource->own().kind != resource_kind || !resource->own().real) continue;
                auto* real = static_cast<ID3D12Resource*>(resource->own().real);
                if (!(real->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) continue;
                Microsoft::WRL::ComPtr<ID3D12Device> owner;
                if (FAILED(real->GetDevice(IID_PPV_ARGS(&owner))) || owner.Get() != device_of(self->own().list_device)) continue;
                barrier.UAV.pResource = real;
            }
            list->ResourceBarrier(1, &barrier);
            ++forwarded;
        }
    }
    static LONG reports;
    if (count && first_calls(reports, 8))
        log("x2w-d3d12 CommandList::slot26(ResourceBarrier): forwarded=%u refused=%u", forwarded, count - forwarded);
    return 0; // no console barrier actions to perform on the caller's side
}

// Compute root views on the desktop list. Root SRVs come from slot 43; root UAVs only from the fixture hook
// x2w_d3d12_fixture_compute_root_uav (slot 45 is not forwarded). Both are recorded here so restore_compute_root can set
// them again after a shim compute pass has replaced the desktop compute root.
void bind_compute_root_view(Pass& pass, ID3D12GraphicsCommandList* list, UINT parameter, D3D12_ROOT_PARAMETER_TYPE type,
                            D3D12_GPU_VIRTUAL_ADDRESS address)
{
    if (type == D3D12_ROOT_PARAMETER_TYPE_SRV) {
        pass.desktop_compute_upload_srvs[parameter] = address;
        list->SetComputeRootShaderResourceView(parameter, address);
    } else if (type == D3D12_ROOT_PARAMETER_TYPE_UAV) {
        pass.desktop_compute_root_uavs[parameter] = address;
        list->SetComputeRootUnorderedAccessView(parameter, address);
    }
}
bool has_compute_root_views(const Pass& pass)
{
    const auto live = [](D3D12_GPU_VIRTUAL_ADDRESS address) { return address != 0; };
    return std::any_of(pass.desktop_compute_upload_srvs.begin(), pass.desktop_compute_upload_srvs.end(), live) ||
        std::any_of(pass.desktop_compute_root_uavs.begin(), pass.desktop_compute_root_uavs.end(), live);
}
// Sets the title's desktop compute root again, with every view bind_compute_root_view recorded under it, after a shim
// compute pass (indexed-draw validation, split indirect Dispatch) replaced the root and invalidated its arguments.
void restore_compute_root(const Pass& pass, ID3D12GraphicsCommandList* list)
{
    if (!pass.desktop_compute_root) return;
    list->SetComputeRootSignature(pass.desktop_compute_root.Get());
    for (UINT parameter = 0; parameter < pass.desktop_compute_upload_srvs.size(); ++parameter) {
        if (const auto address = pass.desktop_compute_upload_srvs[parameter])
            list->SetComputeRootShaderResourceView(parameter, address);
        if (const auto address = pass.desktop_compute_root_uavs[parameter])
            list->SetComputeRootUnorderedAccessView(parameter, address);
    }
}

void __stdcall list_compute_root(Wrapper* self, void* root)
{
    auto* list = recording(self);
    if (!list) return;
    Wrapper* w = as_wrapper(root);
    Pass& pass = pass_of(self);
    // A different root signature invalidates the title's cached compute root arguments.
    if (pass.compute_root_wrapper != w) pass.compute_constants.clear();
    pass.compute_root_wrapper = w;
    const bool real = w && w->own().kind == root_signature_kind && w->own().real && !root_heap_flags(w);
    pass.compute_root = real ? static_cast<ID3D12RootSignature*>(w->own().real) : nullptr;
    pass.compute_tables.set_signature(w ? root_info(w) : nullptr);  // a different signature leaves no table bound
    pass.compute_params = real ? root_info(w) : nullptr;
    pass.compute_srv_bound = 0;
    // Root SRV parameters are forwarded by their own slot; anything else the root declares is not.
    pass.compute_root_has_arguments = real && !w->own().root_without_bindings && !w->own().root_srv_only;
    if (compute_binding_trace_enabled() && (!pass.compute_trace || pass.compute_trace->root != w)) {
        pass.compute_trace = std::make_unique<ComputeBindingTrace>();
        pass.compute_trace->root = w;
    }
    pass.compute_srv_bound = 0;
    // Root SRV parameters are forwarded by their own slot; anything else the root declares is not.
    pass.compute_root_has_arguments = real && !w->own().root_without_bindings && !w->own().root_srv_only;
    if (pass.compute_root) {
        if (pass.desktop_compute_root.Get() != pass.compute_root) {
            pass.desktop_compute_upload_srvs.fill(0);
            pass.desktop_compute_root_uavs.fill(0);
        }
        pass.desktop_compute_root = pass.compute_root;
        list->SetComputeRootSignature(pass.compute_root);
    }
}

// Fixture binding of a desktop root UAV (a desktop GPU address) on a guest list's recording, under the compute root
// set through slot 33. Not a guest slot: titles have no compute root UAV on the desktop list (slot 45 is not
// forwarded). It records the view through bind_compute_root_view, as slot 43 does for root SRVs, so
// restore_compute_root sets it again after a shim compute pass; a binding made on the desktop list directly would be
// lost there.
extern "C" __declspec(dllexport) BOOL x2w_d3d12_fixture_compute_root_uav(void* guest_list, UINT parameter, UINT64 address)
{
    Wrapper* self = as_wrapper(guest_list);
    if (!self || self->own().kind != list_kind) return FALSE;
    auto* list = recording(self);
    auto& pass = pass_of(self);
    const auto* params = pass.compute_params;
    if (!list || !pass.compute_root || !params || parameter >= params->size() ||
        parameter >= pass.desktop_compute_root_uavs.size() || (*params)[parameter].type != D3D12_ROOT_PARAMETER_TYPE_UAV ||
        !address)
        return FALSE;
    bind_compute_root_view(pass, list, parameter, D3D12_ROOT_PARAMETER_TYPE_UAV, address);
    return TRUE;
}

// Query slots 28 to 30, as a hypothesis (command_list_slot_names.h): no fixture here establishes which console
// methods sit there, so they are bound only with X2W_D3D12_QUERY_SLOTS=1. The heap reaches device slot 39 with the
// desktop desc. Only occlusion queries are
// forwarded (shortcut: other heap types are refused until a caller needs them); a call that does not fit the
// heap it names is refused and logged with its values instead of reaching the desktop list.
enum QueryMethod { begin_query, end_query, resolve_query };
void refuse_query(QueryMethod method, const char* reason, UINT type, UINT index, UINT count)
{
    static const char* const names[] = {"BeginQuery", "EndQuery", "ResolveQueryData"};
    static LONG refusals[3];  // the first 16 refusals of each method are logged
    if (first_calls(refusals[method], 16))
        log("x2w-render %s refused: %s (type %u, index %u, count %u)", names[method], reason, type, index, count);
}
// The desktop heap behind a wrapped occlusion heap that holds [index, index + count), or null with the reason.
// Only a DIRECT recording is supported in this first slice.
ID3D12QueryHeap* occlusion_heap(ID3D12GraphicsCommandList* list, void* heap, UINT type, UINT index, UINT count,
                               const char** reason)
{
    if (!list) return *reason = "list is closed", nullptr;
    if (list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) return *reason = "requires a direct command list", nullptr;
    const Wrapper* w = as_wrapper(heap);
    if (!w || w->own().kind != query_heap_kind || !w->own().real) return *reason = "not a desktop query heap", nullptr;
    const auto& desc = w->own().query_heap;
    if (desc.Type != D3D12_QUERY_HEAP_TYPE_OCCLUSION) return *reason = "only occlusion heaps are forwarded", nullptr;
    if (type != D3D12_QUERY_TYPE_OCCLUSION && type != D3D12_QUERY_TYPE_BINARY_OCCLUSION)
        return *reason = "query type does not fit an occlusion heap", nullptr;
    if (index >= desc.Count || count > desc.Count - index) return *reason = "outside the heap's count", nullptr;
    return static_cast<ID3D12QueryHeap*>(w->own().real);
}
// List slot 28, BeginQuery(heap, type, index).
void __stdcall list_begin_query(Wrapper* self, void* heap, UINT type, UINT index)
{
    auto* list = recording(self);
    const char* reason = nullptr;
    ID3D12QueryHeap* real = occlusion_heap(list, heap, type, index, 1, &reason);
    if (!real) return refuse_query(begin_query, reason, type, index, 1);
    Pass& pass = pass_of(self);
    for (const auto& open : pass.open_queries)
        if (open.heap == real && open.index == index) return refuse_query(begin_query, "query already begun", type, index, 1);
    auto& ended = pass.ended_queries;
    ended.erase(std::remove_if(ended.begin(), ended.end(),
                               [&](const Pass::OpenQuery& q) { return q.heap == real && q.index == index; }),
                ended.end());
    pass.open_queries.push_back({real, type, index});
    list->BeginQuery(real, static_cast<D3D12_QUERY_TYPE>(type), index);
}
// List slot 29, EndQuery(heap, type, index). Only a query this list began is ended.
void __stdcall list_end_query(Wrapper* self, void* heap, UINT type, UINT index)
{
    auto* list = recording(self);
    const char* reason = nullptr;
    ID3D12QueryHeap* real = occlusion_heap(list, heap, type, index, 1, &reason);
    if (!real) return refuse_query(end_query, reason, type, index, 1);
    Pass& pass = pass_of(self);
    auto& open = pass.open_queries;
    const auto found = std::find_if(open.begin(), open.end(),
                                    [&](const Pass::OpenQuery& q) { return q.heap == real && q.type == type && q.index == index; });
    if (found == open.end()) return refuse_query(end_query, "not begun on this list", type, index, 1);
    open.erase(found);
    pass.ended_queries.push_back({real, type, index});
    list->EndQuery(real, static_cast<D3D12_QUERY_TYPE>(type), index);
}
// At Close: a query begun on the desktop list and never ended would make the desktop Close fail and drop the whole
// list, where the console call it stands for cost nothing more than that query's result.
void end_open_queries(Wrapper* self, ID3D12GraphicsCommandList* list)
{
    if (!self->own().pass) return;
    auto& open = pass_of(self).open_queries;
    for (const auto& query : open) {
        refuse_query(end_query, "missing at Close, ended there", query.type, query.index, 1);
        list->EndQuery(query.heap, static_cast<D3D12_QUERY_TYPE>(query.type), query.index);
    }
    open.clear();
}
// List slot 30, ResolveQueryData(heap, type, start, count, destination, offset). The destination is a buffer on
// a readback heap, which stays in COPY_DEST, so it is not moved through the list's COMMON state tracking.
void __stdcall list_resolve_query(Wrapper* self, void* heap, UINT type, UINT start, UINT count, void* destination, u64 offset)
{
    auto* list = recording(self);
    const char* reason = nullptr;
    ID3D12QueryHeap* real = occlusion_heap(list, heap, type, start, count, &reason);
    if (!real) return refuse_query(resolve_query, reason, type, start, count);
    // Resolving a query that was never ended, or ended with the other occlusion type, reads uninitialized data and
    // can remove the device: every index in the range must have a matching EndQuery on this recording.
    const auto& ended = pass_of(self).ended_queries;
    for (UINT i = 0; i < count; ++i)
        if (std::none_of(ended.begin(), ended.end(), [&](const Pass::OpenQuery& q) {
                return q.heap == real && q.index == start + i && q.type == type;
            }))
            return refuse_query(resolve_query, "an index in the range was not ended with this type on this recording", type,
                                start, count);
    const Wrapper* w = as_wrapper(destination);
    if (!w || w->own().kind != resource_kind || !w->own().real)
        return refuse_query(resolve_query, "destination has no desktop resource", type, start, count);
    // Map of a resource with title memory returns that memory, which a desktop resolve never reaches.
    if (w->own().title_memory)
        return refuse_query(resolve_query, "destination has title memory", type, start, count);
    auto* resource = static_cast<ID3D12Resource*>(w->own().real);
    D3D12_HEAP_PROPERTIES properties{};
    const auto desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || FAILED(resource->GetHeapProperties(&properties, nullptr)) ||
        properties.Type != D3D12_HEAP_TYPE_READBACK)
        return refuse_query(resolve_query, "destination is not a readback buffer", type, start, count);
    const u64 bytes = u64{count} * sizeof(UINT64);
    if ((offset & 7) || offset > desc.Width || bytes > desc.Width - offset)
        return refuse_query(resolve_query, "outside the destination buffer", type, start, count);
    list->ResolveQueryData(real, static_cast<D3D12_QUERY_TYPE>(type), start, count, resource, offset);
}
// Device slot 39, CreateQueryHeap(desc, riid, ppv): forwarded, and the desc kept for the query slots.
HRESULT __stdcall device_create_query_heap(Wrapper* self, const D3D12_QUERY_HEAP_DESC* desc, const GUID* iid, void** out)
{
    const auto hr = static_cast<HRESULT>(create_slot<device_kind, 39, 1>(self, reinterpret_cast<u64>(desc),
                                                                         reinterpret_cast<u64>(iid), reinterpret_cast<u64>(out),
                                                                         0, 0, 0, 0, 0));
    if (Wrapper* w = SUCCEEDED(hr) && desc && out ? as_wrapper(*out) : nullptr; w && w->own().kind == query_heap_kind)
        w->own().query_heap = *desc;
    return hr;
}

// Slot-61 calls by outcome and signature kind (IndirectCommand; none counts placeholder and unknown
// signatures), reported with the render totals. 64-bit and never reset, so they neither wrap nor stall.
enum { indirect_forwarded, indirect_graphics_pipeline, indirect_compute_pipeline, indirect_placeholder, indirect_other,
       indirect_outcomes };
volatile LONG64 indirect_calls[indirect_outcomes][5];

struct IndirectAddressAlias {
    ID3D12Resource* resource = nullptr;
    u64 offset = 0;
};

IndirectAddressAlias indirect_address_alias(Wrapper* device, u64 address, u64 bytes);

struct IndirectIndexView {
    D3D12_INDEX_BUFFER_VIEW view{};
    ID3D12Resource* resource = nullptr;
    u64 offset = 0;
    bool upload = false;
};
const char* draw_translated_indirect(Wrapper* self, ID3D12CommandSignature* signature, ID3D12Resource* argument,
                                     u64 offset, const IndirectIndexView& indices);
struct TranslatedDispatch {
    D3D12_DISPATCH_ARGUMENTS groups{};
    ID3D12CommandSignature* signature = nullptr;
    ID3D12Resource* argument = nullptr;
    u64 offset = 0;
};
const char* record_translated_dispatch(Wrapper* self, const TranslatedDispatch& dispatch);
u64 known_pipeline_key(Wrapper* pipeline);

namespace indirect_other_refusal_code {
constexpr const char* list_closed = "list_closed";
constexpr const char* invalid_list_type = "invalid_list_type";
constexpr const char* invalid_command_count = "invalid_command_count";
constexpr const char* count_offset_without_buffer = "count_offset_without_buffer";
constexpr const char* signature_device_mismatch = "signature_device_mismatch";
constexpr const char* graphics_multi_command = "graphics_multi_command";
constexpr const char* cached_mesh_attachments = "cached_mesh_attachments";
constexpr const char* no_render_target = "no_render_target";
constexpr const char* multiple_render_targets = "multiple_render_targets";
constexpr const char* depth_target_bound = "depth_target_bound";
constexpr const char* missing_viewport_or_scissor = "missing_viewport_or_scissor";
constexpr const char* target_format_mismatch = "target_format_mismatch";
constexpr const char* rect_list_indexed = "rect_list_indexed";
constexpr const char* unsupported_topology = "unsupported_topology";
constexpr const char* multisample_target = "multisample_target";
constexpr const char* compute_root_mismatch = "compute_root_mismatch";
constexpr const char* dispatch_root_multi_command = "dispatch_root_multi_command";
constexpr const char* missing_argument_resource = "missing_argument_resource";
constexpr const char* invalid_argument_offset = "invalid_argument_offset";
constexpr const char* argument_address_overflow = "argument_address_overflow";
constexpr const char* argument_outside_placed_buffer = "argument_outside_placed_buffer";
constexpr const char* argument_not_common_state = "argument_not_common_state";
constexpr const char* invalid_index_view = "invalid_index_view";
constexpr const char* index_view_overlap = "index_view_overlap";
constexpr const char* index_not_common_state = "index_not_common_state";
constexpr const char* index_unaligned_address = "index_unaligned_address";
constexpr const char* upload_index_unaligned = "upload_index_unaligned";
constexpr const char* index_no_desktop_range = "index_no_desktop_range";
constexpr const char* aliased_index_unaligned = "aliased_index_unaligned";
constexpr const char* mesh_xmem_read = "mesh_xmem_read";
constexpr const char* rect_live_compute_root = "rect_live_compute_root";
constexpr const char* rect_conversion_unavailable = "rect_conversion_unavailable";
constexpr const char* split_dispatch_unavailable = "split_dispatch_unavailable";
constexpr const char* count_missing_resource = "count_missing_resource";
constexpr const char* count_invalid_offset = "count_invalid_offset";
constexpr const char* count_outside_placed_buffer = "count_outside_placed_buffer";
constexpr const char* count_not_common_state = "count_not_common_state";
constexpr const char* unknown = "unknown";
}  // namespace indirect_other_refusal_code

// Opt-in (X2W_SPLIT_INDIRECT_DISPATCH=1): a batched or counted indirect Dispatch with a desktop compute pipeline is
// split into single-command calls (split_indirect_dispatch.h). Without it the native call runs on every adapter.
bool split_indirect_dispatch_enabled()
{
    return x2w_local::env_on("X2W_SPLIT_INDIRECT_DISPATCH");
}

// The guest method has independent resource and byte-offset arguments.
void __stdcall list_execute_indirect(Wrapper* self, void* signature, UINT count, void* argument, u64 offset,
                                     void* count_buffer, u64 count_offset)
{
    // Every refusal is counted under the signature's kind, including those before the signature is checked.
    Wrapper* sig = as_wrapper(signature);
    IndirectCommand kind = sig && sig->own().kind == command_signature_kind ? sig->own().indirect_command : IndirectCommand::none;
    Wrapper* refusal_pipeline = nullptr;
    u64 trace_call = 0;
    const auto refuse = [&kind, &refusal_pipeline, &trace_call](const char* reason, int outcome = indirect_other,
                                                               const char* code = nullptr) {
        InterlockedIncrement64(&indirect_calls[outcome][static_cast<int>(kind)]);
        if (trace_call)
            log("x2w-mesh86-trace call=%llu stage=record outcome=refused reason=%s", trace_call, reason);
        static LONG refusals;
        if (outcome == indirect_other) {
            const u64 key = known_pipeline_key(refusal_pipeline);
            const char* reason_code = code ? code : indirect_other_refusal_code::unknown;
            const bool first_pair = indirect_other_refusal_census.note(key, reason_code);
            if (first_pair || first_calls(refusals, 16))
                log("x2w-render ExecuteIndirect refused, key %016llx [%s]: %s", key, reason_code, reason);
            return;
        }
        if (first_calls(refusals, 16)) log("x2w-render ExecuteIndirect refused: %s", reason);
    };
    auto* list = recording(self);
    if (!list) return refuse("list is closed", indirect_other, indirect_other_refusal_code::list_closed);
    if (kind == IndirectCommand::dispatch_mesh && mesh_xmem_trace_ordinal()) {
        const auto& pass = pass_of(self);
        if (pass.pipeline && pass.pipeline->own().mesh86_key) trace_call = select_mesh_xmem_trace_call();
        if (trace_call) trace_mesh_xmem_inputs(self, pass, trace_call, argument, offset);
    }
    const auto type = list->GetType();
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE)
        return refuse("requires a direct or compute command list", indirect_other, indirect_other_refusal_code::invalid_list_type);
    // Only a desktop Dispatch signature runs more than one command or reads a count buffer here.
    constexpr UINT max_dispatch_commands = 1u << 16;
    if (!count || count > max_dispatch_commands)
        return refuse("command count is zero or above the supported maximum", indirect_other, indirect_other_refusal_code::invalid_command_count);
    if (!count_buffer && count_offset)
        return refuse("count offset without a count buffer", indirect_other, indirect_other_refusal_code::count_offset_without_buffer);
    if (!sig || sig->own().kind != command_signature_kind || !sig->own().real) {
        // A console-only signature has no desktop equivalent: show what the title put in its argument buffer so the
        // layout can be read from the log (first calls only; bytes of the argument range, at most 48 per call).
        static LONG traced;
        if (sig && sig->own().kind == command_signature_kind && first_calls(traced, 8)) {
            const Wrapper* arg = as_wrapper(argument);
            const auto* memory = arg && arg->own().kind == resource_kind ? static_cast<const unsigned char*>(arg->own().title_memory) : nullptr;
            char bytes[3 * 48 + 1] = "unreadable";
            if (memory && readable_window(reinterpret_cast<u64>(memory) + offset, 48) >= 48) hex_text(memory + offset, 48, bytes, sizeof(bytes));
            log("x2w-render ExecuteIndirect placeholder signature: count=%u offset=%llu count_buffer=%p argument title memory=%p bytes [%s]",
                count, static_cast<unsigned long long>(offset), count_buffer, static_cast<const void*>(memory), bytes);
        }
        return refuse("missing or placeholder signature", indirect_placeholder);
    }
    const Wrapper* device = self->own().list_device;
    if (!device || sig->own().signature_device != device)
        return refuse("requires a supported signature from this device", indirect_other, indirect_other_refusal_code::signature_device_mismatch);
    const auto command = sig->own().indirect_command;
    const bool indexed = command == IndirectCommand::draw_indexed;
    const bool mesh = command == IndirectCommand::dispatch_mesh;
    const bool graphics = indexed || mesh || command == IndirectCommand::draw;
    Pass& pass = pass_of(self);
    refusal_pipeline = pass.pipeline;
    const bool translated_compute = command == IndirectCommand::dispatch && pass.pipeline && pass.pipeline->own().compute_request_key;
    const bool translated = (command == IndirectCommand::draw || indexed) && pass.pipeline && !pass.pipeline->own().real &&
        (pass.graphics_root || pass.deferred_root) && x2w_local::env_on("X2W_TRANSLATED_INDIRECT_DRAW");
    u64 bytes = sig->own().indirect_bytes;
    if (graphics && (count != 1 || count_buffer))
        return refuse("indirect graphics requires one command and no count buffer", indirect_other, indirect_other_refusal_code::graphics_multi_command);
    if (graphics) {
        if (indexed) bytes = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
        else if (mesh) bytes = sizeof(D3D12_DISPATCH_MESH_ARGUMENTS);
        else bytes = sizeof(D3D12_DRAW_ARGUMENTS);
        if (type != D3D12_COMMAND_LIST_TYPE_DIRECT || (!translated && (!pass.indirect_graphics.root ||
            pass.indirect_graphics.root != pass.graphics_root || pass.indirect_graphics.mesh != mesh))) {
            // A console pipeline (no desktop pipeline) used only here still gets the request a direct draw writes;
            // nothing is drawn.
            if (mesh && type == D3D12_COMMAND_LIST_TYPE_DIRECT && pass.pipeline && !pass.pipeline->own().real &&
                (pass.requested_mesh_pipeline != pass.pipeline || pass.requested_mesh_root != pass.root)) {
                pass.requested_mesh_pipeline = pass.pipeline;
                pass.requested_mesh_root = pass.root;
                request_slot57_mesh(pass.pipeline, pass.root);
            } else if (!mesh && pass.pipeline && !pass.pipeline->own().real && pass.requested_pipeline != pass.pipeline) {
                pass.requested_pipeline = pass.pipeline;
                request_translation(pass.pipeline);
            }
            if (mesh && pass.pipeline && pass.pipeline->own().mesh86_key) {
                static LONG mesh_cache_refusals;
                if (first_calls(mesh_cache_refusals, 16))
                    log("x2w-render slot86 mesh %016llx: no matching desktop pipeline/root for indirect execution",
                        pass.pipeline->own().mesh86_key);
            }
            return refuse("indirect graphics requires a direct list with a matching desktop pipeline and root",
                          indirect_graphics_pipeline);
        }
        if (pass.indirect_graphics.cached_mesh) {
            const auto& pso = pass.indirect_graphics;
            if (!pass.targets_bound || !pass.viewport_set || !pass.scissor_set || !pass.targets_single_sample ||
                pass.target_count != pso.cached_target_count || pass.dsv_format != pso.cached_depth ||
                (pso.cached_depth_writes && !pass.depth_view_writable) ||
                !std::equal(pso.cached_targets, pso.cached_targets + pso.cached_target_count, pass.target_formats))
                return refuse("cached mesh requires matching color and depth attachments, viewport and scissor",
                              indirect_other, indirect_other_refusal_code::cached_mesh_attachments);
            if (!pass.root || pass.root != pso.guest_root_params || pass.root_heap_flags)
                return refuse("cached mesh requires a recorded root without heap indexing", indirect_graphics_pipeline);
            for (std::size_t i = 0; i < pass.root->size(); ++i) {
                if (i == pso.cached_srv_root) continue;
                const auto mask = (1u << (*pass.root)[i].constants_count) - 1;
                if ((pass.constant_words_bound[i] & mask) != mask)
                    return refuse("cached mesh root constants are incomplete", indirect_graphics_pipeline);
            }
        } else if (translated) {
            if (!pass.targets_bound || (!pass.target_count && pass.dsv_format == DXGI_FORMAT_UNKNOWN) ||
                !pass.viewport_set || !pass.scissor_set) {
                const char* code =
                    (!pass.targets_bound || (!pass.target_count && pass.dsv_format == DXGI_FORMAT_UNKNOWN))
                        ? indirect_other_refusal_code::no_render_target
                        : indirect_other_refusal_code::missing_viewport_or_scissor;
                return refuse("translated indirect draw requires bound target or depth view, viewport and scissor", indirect_other, code);
            }
            if (!pass.targets_single_sample)
                return refuse("draw requires single-sample targets", indirect_other, indirect_other_refusal_code::multisample_target);
        } else if (!pass.targets_bound || pass.target_count != 1 || !pass.target0 ||
            (!translated && pass.target_formats[0] != pass.indirect_graphics.target_format) ||
            pass.dsv_format != DXGI_FORMAT_UNKNOWN || !pass.viewport_set || !pass.scissor_set) {
            const char* code =
                pass.target_count > 1                                 ? indirect_other_refusal_code::multiple_render_targets
                : (!pass.targets_bound || !pass.target_count || !pass.target0) ? indirect_other_refusal_code::no_render_target
                : pass.dsv_format != DXGI_FORMAT_UNKNOWN              ? indirect_other_refusal_code::depth_target_bound
                : (!pass.viewport_set || !pass.scissor_set)           ? indirect_other_refusal_code::missing_viewport_or_scissor
                                                                      : indirect_other_refusal_code::target_format_mismatch;
            return refuse("indirect graphics requires one matching target, viewport and scissor", indirect_other, code);
        }
        const bool rect_list = !mesh && pass.topology == 0x11;
        if (rect_list && indexed && !translated)
            return refuse("indirect rectangle list requires a non-indexed draw", indirect_other, indirect_other_refusal_code::rect_list_indexed);
        if (!mesh && !rect_list && pass.topology != D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
            return refuse("indirect draw requires triangle-list or rectangle-list topology", indirect_other, indirect_other_refusal_code::unsupported_topology);
        if (!pass.indirect_graphics.cached_mesh && !translated) {
            const auto samples = pass.target0->GetDesc().SampleDesc;
            if (samples.Count != 1 || samples.Quality)
                return refuse("draw requires a single-sample render target", indirect_other, indirect_other_refusal_code::multisample_target);
        }
    } else if (translated_compute) {
        if (count != 1 || count_buffer || sig->own().signature_root)
            return refuse("translated Dispatch requires one rootless command and no count buffer", indirect_compute_pipeline);
    } else if (!pass.compute_pipeline || !pass.compute_root) {
        return refuse("requires a desktop compute pipeline and root", indirect_compute_pipeline);
    } else if (sig->own().signature_root && sig->own().signature_root != pass.compute_root) {
        return refuse("signature root constants require the bound compute root", indirect_other, indirect_other_refusal_code::compute_root_mismatch);
    } else if (sig->own().signature_root && (count != 1 || count_buffer)) {
        // Batched and count-buffer Dispatch is supported for rootless signatures only in this slice.
        return refuse("a Dispatch signature with root constants requires one command and no count buffer",
                      indirect_other, indirect_other_refusal_code::dispatch_root_multi_command);
    }
    Wrapper* buffer = as_wrapper(argument);
    if (!buffer || buffer->own().kind != resource_kind || !buffer->own().real)
        return refuse("argument has no desktop resource", indirect_other, indirect_other_refusal_code::missing_argument_resource);
    const auto base = reinterpret_cast<u64>(buffer->own().title_memory);
    if (!base || (offset & 3) || offset > ~u64{} - base)
        return refuse("invalid argument offset or unmapped buffer", indirect_other, indirect_other_refusal_code::invalid_argument_offset);
    // Every record up to the maximum must lie in the buffer: the count word can only lower the number executed.
    const u64 span = u64{count - 1} * sig->own().indirect_bytes + bytes;
    if (span < bytes || base + offset > ~u64{} - span)
        return refuse("argument records overflow the address space", indirect_other, indirect_other_refusal_code::argument_address_overflow);
    const auto found = placed_buffer_range(device, base + offset, span);
    if (found.reject || found.owner != buffer)
        return refuse("argument record is outside a unique placed buffer", indirect_other, indirect_other_refusal_code::argument_outside_placed_buffer);
    const auto* info = resource_info(buffer);
    if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON)
        return refuse("argument buffer does not use COMMON state tracking", indirect_other, indirect_other_refusal_code::argument_not_common_state);
    auto* resource = static_cast<ID3D12Resource*>(buffer->own().real);
    if (translated_compute) {
        const TranslatedDispatch dispatch{{}, static_cast<ID3D12CommandSignature*>(sig->own().real), resource, offset};
        if (const char* failure = record_translated_dispatch(self, dispatch)) return refuse(failure, indirect_compute_pipeline);
        InterlockedIncrement64(&indirect_calls[indirect_forwarded][static_cast<int>(kind)]);
        return;
    }
    IndirectIndexView index_view;
    if (indexed) {
        auto view = pass.ib;
        const UINT alignment = view.Format == DXGI_FORMAT_R16_UINT ? 2 : view.Format == DXGI_FORMAT_R32_UINT ? 4 : 0;
        if (!alignment || !view.BufferLocation || !view.SizeInBytes ||
            view.BufferLocation % alignment || view.SizeInBytes % alignment)
            return refuse("invalid index view", indirect_other, indirect_other_refusal_code::invalid_index_view);
        const auto indices = placed_buffer_range(device, view.BufferLocation, view.SizeInBytes);
        if (indices.reject)
            return refuse("index view crosses or overlaps placed buffers", indirect_other, indirect_other_refusal_code::index_view_overlap);
        if (indices.owner && indices.owner->own().real) {
            const auto* index_info = resource_info(indices.owner);
            if (!index_info || index_info->initial_state != D3D12_RESOURCE_STATE_COMMON)
                return refuse("index buffer does not use COMMON state tracking", indirect_other, indirect_other_refusal_code::index_not_common_state);
            auto* index = static_cast<ID3D12Resource*>(indices.owner->own().real);
            const auto index_base = reinterpret_cast<u64>(indices.owner->own().title_memory);
            const auto index_offset = view.BufferLocation - index_base;
            const auto native_base = index->GetGPUVirtualAddress();
            if (!native_base || index_offset > ~u64{} - native_base || (native_base + index_offset) % alignment)
                return refuse("index view has no aligned desktop address", indirect_other, indirect_other_refusal_code::index_unaligned_address);
            view.BufferLocation = native_base + index_offset;
            index_view = {view, index, index_offset};
        } else {
            const auto upload = translated && !indices.owner
                ? list_upload_range(self, view.BufferLocation, view.SizeInBytes) : x2w_upload::Allocation{};
            // The compute validator reads 32-bit words, including for R16_UINT. Do not
            // load an adjacent recording upload allocation when extracting a half-word.
            if (upload.resource) {
                if ((upload.offset | view.SizeInBytes) & 3)
                    return refuse("upload index view is not dword-aligned and dword-sized",
                                  indirect_other, indirect_other_refusal_code::upload_index_unaligned);
                view.BufferLocation = upload.gpu;
                index_view = {view, upload.resource, upload.offset, true};
            } else {
                const auto alias = translated && !indices.owner && !list_upload_range(self, view.BufferLocation, 1).resource
                    ? indirect_address_alias(self->own().list_device, view.BufferLocation, view.SizeInBytes) : IndirectAddressAlias{};
                if (!alias.resource)
                    return refuse("index view has no proven desktop buffer range", indirect_other, indirect_other_refusal_code::index_no_desktop_range);
                const auto gpu = alias.resource->GetGPUVirtualAddress();
                if (!gpu || ((alias.offset | view.SizeInBytes) & 3))
                    return refuse("aliased index view is not dword-aligned and dword-sized",
                                  indirect_other, indirect_other_refusal_code::aliased_index_unaligned);
                view.BufferLocation = gpu + alias.offset;
                index_view = {view, alias.resource, alias.offset};
            }
        }
    }
    if (translated) {
        const char* failure = draw_translated_indirect(self, static_cast<ID3D12CommandSignature*>(sig->own().real), resource,
                                                       offset, index_view);
        if (failure) return refuse(failure, indirect_graphics_pipeline);
        InterlockedIncrement64(&indirect_calls[indirect_forwarded][static_cast<int>(kind)]);
        return;
    }
    auto argument_state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    if (index_view.resource) {
        if (index_view.resource == resource) argument_state |= D3D12_RESOURCE_STATE_INDEX_BUFFER;
        else require_state(self, index_view.resource, D3D12_RESOURCE_STATE_INDEX_BUFFER);
        list->IASetIndexBuffer(&index_view.view);
    }
    const bool rect_list = graphics && !mesh && pass.topology == 0x11;
    ID3D12Resource* scratch = nullptr;
    ID3D12Resource* draw_arguments = resource;
    u64 draw_offset = offset;
    const UINT mesh_srv = mesh && pass.indirect_graphics.cached_mesh ? pass.indirect_graphics.cached_srv_root : UINT_MAX;
    const bool leased_arguments = mesh && mesh_xmem_indirect() && info->placement == Placement::copied_buffer &&
        (!pass.indirect_graphics.cached_mesh || mesh_srv != UINT_MAX);
    MeshXMemRead inputs[2]{};
    size_t input_count = 0;
    if (mesh_srv != UINT_MAX)
        inputs[input_count++] = {pass.root_address[mesh_srv], sizeof(UINT), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    if (leased_arguments)
        inputs[input_count++] = {base + offset, bytes, argument_state};
    if (input_count) {
        if (const char* failure = record_mesh_xmem_reads(self, inputs, input_count, trace_call))
            return refuse(failure, mesh_srv != UINT_MAX ? indirect_graphics_pipeline : indirect_other,
                          indirect_other_refusal_code::mesh_xmem_read);
        if (leased_arguments) {
            draw_arguments = inputs[input_count - 1].resource;
            draw_offset = inputs[input_count - 1].offset;
        }
    }
    if (rect_list) {
        if (has_compute_root_views(pass))
            return refuse("rectangle-list conversion would drop live compute root arguments",
                          indirect_other, indirect_other_refusal_code::rect_live_compute_root);
        // The producer's count names three vertices per rectangle; the desktop draw needs six. Convert on the GPU
        // into a scratch record and draw from that. The compute root signature and bindings of the list are replaced.
        // The four words are copied into the scratch first (the pass never reads the title's buffer), so the source only
        // needs COPY_SOURCE next to its argument state.
        require_state(self, resource, argument_state | D3D12_RESOURCE_STATE_COPY_SOURCE);
        scratch = recording_rect_scratch(self);
        if (!scratch || !x2w_startup::rect_indirect_arguments().convert(list, resource, offset, scratch))
            return refuse("rectangle-list argument conversion is unavailable",
                          indirect_other, indirect_other_refusal_code::rect_conversion_unavailable);
        if (pass.desktop_compute_root) list->SetComputeRootSignature(pass.desktop_compute_root.Get());
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        draw_arguments = scratch;
        draw_offset = 0;
    }
    if (graphics) {
        list->SetPipelineState(static_cast<ID3D12PipelineState*>(pass.pipeline->own().real));
        list->SetGraphicsRootSignature(pass.graphics_root);
        if (pass.indirect_graphics.cached_mesh) {
            const UINT srv = pass.indirect_graphics.cached_srv_root;
            if (srv != UINT_MAX)
                list->SetGraphicsRootShaderResourceView(srv, inputs[0].resource->GetGPUVirtualAddress() + inputs[0].offset);
            for (UINT i = 0; i < pass.root->size(); ++i) {
                if (i == srv) continue;
                list->SetGraphicsRoot32BitConstants(i, (*pass.root)[i].constants_count, pass.constants[i], 0);
            }
        }
    }
    // Opt-in split of a batched or counted Dispatch for adapters that run only its first command
    // (split_indirect_dispatch.h). Only a rootless Dispatch signature with a desktop compute pipeline and root reaches
    // here with a count above one or a count buffer; graphics and mesh signatures are refused above.
    const bool split = command == IndirectCommand::dispatch && pass.compute_pipeline && pass.compute_root &&
        (count > 1 || count_buffer) && split_indirect_dispatch_enabled();
    ID3D12Resource* counts = nullptr;
    if (count_buffer) {
        // The count word is four bytes in a unique placed buffer that also tracks state from COMMON.
        Wrapper* count_wrapper = as_wrapper(count_buffer);
        if (!count_wrapper || count_wrapper->own().kind != resource_kind || !count_wrapper->own().real)
            return refuse("count buffer has no desktop resource", indirect_other, indirect_other_refusal_code::count_missing_resource);
        const auto count_base = reinterpret_cast<u64>(count_wrapper->own().title_memory);
        if (!count_base || (count_offset & 3) || count_offset > ~u64{} - count_base - sizeof(UINT))
            return refuse("invalid count offset or unmapped count buffer", indirect_other, indirect_other_refusal_code::count_invalid_offset);
        const auto count_found = placed_buffer_range(device, count_base + count_offset, sizeof(UINT));
        if (count_found.reject || count_found.owner != count_wrapper)
            return refuse("count word is outside a unique placed buffer", indirect_other, indirect_other_refusal_code::count_outside_placed_buffer);
        const auto* count_info = resource_info(count_wrapper);
        if (!count_info || count_info->initial_state != D3D12_RESOURCE_STATE_COMMON)
            return refuse("count buffer does not use COMMON state tracking", indirect_other, indirect_other_refusal_code::count_not_common_state);
        counts = static_cast<ID3D12Resource*>(count_wrapper->own().real);
        if (counts != resource)
            require_state(self, counts, split ? D3D12_RESOURCE_STATE_COPY_SOURCE : D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    }
    if (split) {
        // The rewrite reads the title's records and count word by copies only, then replaces the compute root and
        // pipeline. The compute state the title can set on the desktop list is the pipeline (slot 25), the root
        // (slot 33) and root SRVs (slot 43); the other compute root setters are not forwarded. The pipeline is set
        // again here and the root with its views by restore_compute_root, the same code as after the indexed-draw
        // validation pass, before the single-command calls.
        auto* desktop = device_of(self->own().list_device);
        const auto stride = static_cast<UINT>(sig->own().indirect_bytes);
        auto rewritten = x2w_startup::SplitIndirectDispatch::make_scratch(desktop, count, stride);
        if (!rewritten || !retain_with_recording(self, rewritten))
            return refuse("split indirect Dispatch has no scratch buffer", indirect_other,
                          indirect_other_refusal_code::split_dispatch_unavailable);
        require_state(self, resource, D3D12_RESOURCE_STATE_COPY_SOURCE);
        if (!x2w_startup::split_indirect_dispatch().rewrite(list, resource, offset, count, stride, counts, count_offset,
                                                             rewritten.Get()))
            return refuse("split indirect Dispatch rewrite is unavailable", indirect_other,
                          indirect_other_refusal_code::split_dispatch_unavailable);
        list->SetPipelineState(static_cast<ID3D12PipelineState*>(pass.pipeline->own().real));
        restore_compute_root(pass, list);
        const auto first = x2w_startup::SplitIndirectDispatch::rewritten_offset(count, stride);
        auto* command = static_cast<ID3D12CommandSignature*>(sig->own().real);
        for (UINT i = 0; i < count; ++i)
            list->ExecuteIndirect(command, 1, rewritten.Get(), first + u64{i} * sizeof(D3D12_DISPATCH_ARGUMENTS), nullptr, 0);
        x2w_startup::SplitIndirectDispatch::finish(list, rewritten.Get());
        InterlockedIncrement64(&indirect_calls[indirect_forwarded][static_cast<int>(kind)]);
        return;
    }
    if (!leased_arguments && !rect_list) {
        require_state(self, resource, argument_state);
    }
    const auto red_samples = graphics ? red_producer_draw_before(self, list, "indirect-draw", 0) : RedDrawSamples{};
    list->ExecuteIndirect(static_cast<ID3D12CommandSignature*>(sig->own().real), count, draw_arguments, draw_offset, counts,
                          counts ? count_offset : 0);
    red_producer_draw_after(self, red_samples);
    if (trace_call) log("x2w-mesh86-trace call=%llu stage=record outcome=recorded", trace_call);
    if (scratch) x2w_startup::RectIndirectArguments::finish(list, scratch);
    InterlockedIncrement64(&indirect_calls[indirect_forwarded][static_cast<int>(kind)]);
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
        std::memset(pass.constant_words_bound, 0, sizeof(pass.constant_words_bound));
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
        if (slot == 20 || slot == 21) out.resource = view_target(handle, slot);
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
// List slot 51 (desktop 47 + 4), ClearDepthStencilView(view, flags, depth, stencil, rect count, rects). Desktop
// resources rest in COMMON here, so the target is moved to DEPTH_WRITE around the clear. The view is
// read from its shadow record, as OMSetRenderTargets does, so a descriptor the title copied clears the
// resource it now holds; a handle with no record falls back to the creation-time map.
void __stdcall list_clear_depth_stencil(Wrapper* self, u64 handle, D3D12_CLEAR_FLAGS flags, float depth, UINT8 stencil,
                                       UINT rect_count, const D3D12_RECT* rects)
{
    if (!self || !self->own().list_open) return;
    Timed timed(time_list);
    InterlockedIncrement(&draw_calls_seen);
    InterlockedIncrement64(&list_slot_calls[51]);
    InterlockedIncrement64(&list_slot_census[51]);
    if (trace_frame_active) {
        const auto ret = reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr));
        log("x2w-frame tid=%lu list=%llx slot51(0x%llx, 0x%x, %.3f, %u | %u, %p) ret=+0x%llx",
            GetCurrentThreadId(), reinterpret_cast<u64>(self), handle, static_cast<unsigned>(flags),
            depth, static_cast<unsigned>(stencil), rect_count, static_cast<const void*>(rects), ret);
    }
    const u64 desktop = desktop_cpu_handle(handle);
    const ResolvedView view = resolve_view(desktop, 21);
    ID3D12Resource* target = view.resource;
    static LONG logged;
    if (first_calls(logged, 4))
        log("x2w-render ClearDepthStencilView view=0x%llx target=%p flags=0x%x depth=%.3f stencil=%u rects=%u", handle,
            static_cast<void*>(target), static_cast<unsigned>(flags), depth, static_cast<unsigned>(stencil), rect_count);
    if (!target) {
        InterlockedIncrement(&render_target_unresolved);
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: view 0x%llx is %s", handle,
                view.wrong_kind ? "not a depth stencil view"
                                : view.skipped ? "a skipped depth stencil view" : "not a known depth stencil view");
        return;
    }
    if (!(flags & (D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL))) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: zero clear flags 0x%x", static_cast<unsigned>(flags));
        return;
    }
    if ((flags & D3D12_CLEAR_FLAG_DEPTH) && (view.dsv_flags & D3D12_DSV_FLAG_READ_ONLY_DEPTH)) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: depth clear requested on read-only depth view 0x%llx", handle);
        return;
    }
    if ((flags & D3D12_CLEAR_FLAG_STENCIL) && (view.dsv_flags & D3D12_DSV_FLAG_READ_ONLY_STENCIL)) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: stencil clear requested on read-only stencil view 0x%llx", handle);
        return;
    }
    const DXGI_FORMAT format = view_format_of(desktop, target);
    if ((flags & D3D12_CLEAR_FLAG_STENCIL) && !format_has_stencil(format)) {
        flags = static_cast<D3D12_CLEAR_FLAGS>(flags & ~D3D12_CLEAR_FLAG_STENCIL);
        if (!flags) return;
    }
    if (rect_count && !x2w_local::readable(rects, std::size_t{rect_count} * sizeof(D3D12_RECT))) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-d3d12 ClearDepthStencilView skipped: unreadable rects array (%u rects at %p)", rect_count, static_cast<const void*>(rects));
        return;
    }
    auto* list = static_cast<ID3D12GraphicsCommandList*>(self->own().real);
    require_state(self, target, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    const auto red_sample = red_producers().before(self, list, target, "clear-dsv", 0, 0);
    list->ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(desktop)}, flags, depth, stencil,
                                rect_count, rect_count ? rects : nullptr);
    red_producers().after(self, red_sample);
    InterlockedIncrement(&render_clears);
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
    const auto red_sample = red_producers().before(self, list, target, "clear-rtv", 0, 0);
    list->ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(desktop)}, colour, rect_count,
                                rect_count ? rects : nullptr);
    red_producers().after(self, red_sample);
    InterlockedIncrement(&render_clears);
}
// The desktop shader-visible CBV/SRV/UAV heap that owns the GPU handle's descriptor, or null.
ID3D12DescriptorHeap* owning_view_heap(u64 gpu)
{
    u64 owner = 0;
    AcquireSRWLockShared(&note_lock);
    for (const HeapInfo& heap : heap_infos()) {
        if (heap.type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || !heap.desktop_gpu || gpu < heap.desktop_gpu) continue;
        const u64 offset = gpu - heap.desktop_gpu;
        if (heap.increment && offset < u64{heap.count} * heap.increment && offset % heap.increment == 0) {
            owner = heap.wrapper;
            break;
        }
    }
    ReleaseSRWLockShared(&note_lock);
    return owner ? reinterpret_cast<ID3D12DescriptorHeap*>(unwrap(owner)) : nullptr;
}
// require_state assumes a resource starts in COMMON; a placed buffer made with another initial state does not.
bool tracked_from_common(u64 title_resource)
{
    const ResourceInfo* info = resource_info(as_wrapper(reinterpret_cast<const void*>(title_resource)));
    return !info || info->initial_state == D3D12_RESOURCE_STATE_COMMON;
}
// List slots 53 and 54 (desktop 49/50 + 4), ClearUnorderedAccessViewUint/Float(view GPU handle, view CPU
// handle, resource, values, rect count, rects). The CPU handle is read from its shadow record, like the
// render-target clear, so a copied descriptor clears the resource it holds now. A GPU handle that is on no
// known heap (its owning heap is bound for the clear and the translated-heap cache is reset), a resource not
// tracked from COMMON, a CPU handle that holds no unordered-access view, or a resource without unordered access is
// refused (logged, counted) instead of reaching the desktop list. The target moves to UNORDERED_ACCESS
// around the clear.
template <bool Float>
void __stdcall list_clear_uav(Wrapper* self, u64 gpu_handle, u64 cpu_handle, u64 title_resource, const void* values,
                              UINT rect_count, const D3D12_RECT* rects)
{
    if (!self->own().list_open) return;
    static LONG logged, refused;
    const char* name = Float ? "ClearUnorderedAccessViewFloat" : "ClearUnorderedAccessViewUint";
    const u64 desktop = desktop_cpu_handle(cpu_handle);
    const ResolvedView view = desktop ? resolve_view(desktop, 19) : ResolvedView{};
    ID3D12Resource* target = view.resource;
    ID3D12DescriptorHeap* view_heap = nullptr;
    const char* why = nullptr;
    if (!x2w_local::readable(values, 4 * sizeof(UINT))) why = "no clear values";
    else if (rect_count && !x2w_local::readable(rects, std::size_t{rect_count} * sizeof(D3D12_RECT))) why = "unreadable rects";
    else if (!target) why = view.wrong_kind ? "view is not an unordered-access view"
                            : view.skipped ? "view was skipped at creation" : "view holds no known resource";
    else if (!(view_heap = owning_view_heap(gpu_handle))) why = "GPU handle is on no known descriptor heap";
    else if (!(target->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) why = "resource lacks unordered access";
    else if (!tracked_from_common(title_resource)) why = "resource does not use COMMON state tracking";
    if (why) {
        InterlockedIncrement(&render_target_unresolved);
        if (first_calls(refused, 6)) log("x2w-d3d12 %s skipped: %s (view 0x%llx)", name, why, cpu_handle);
        return;
    }
    if (first_calls(logged, 4)) log("x2w-render %s view=0x%llx target=%p rects=%u", name, cpu_handle, static_cast<void*>(target), rect_count);
    auto* list = static_cast<ID3D12GraphicsCommandList*>(self->own().real);
    require_state(self, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // The clear needs the GPU descriptor's own heap bound; the next translated draw binds its heaps again.
    ID3D12DescriptorHeap* bound[] = {view_heap};
    list->SetDescriptorHeaps(1, bound);
    pass_of(self).heap_set = false;
    const D3D12_GPU_DESCRIPTOR_HANDLE gpu{gpu_handle};
    const D3D12_CPU_DESCRIPTOR_HANDLE cpu{static_cast<SIZE_T>(desktop)};
    if (Float) list->ClearUnorderedAccessViewFloat(gpu, cpu, target, static_cast<const float*>(values), rect_count, rect_count ? rects : nullptr);
    else list->ClearUnorderedAccessViewUint(gpu, cpu, target, static_cast<const UINT*>(values), rect_count, rect_count ? rects : nullptr);
    InterlockedIncrement(&render_clears);
}
// List slot 55 (desktop 51 + 4), DiscardResource(resource, region). Discard is only a hint, so a resource
// that is not a render target, depth target or unordered-access resource (needs a state the list does not
// track), an unreadable region or a region with unreadable rects is dropped; nothing reaches the desktop list.
void __stdcall list_discard(Wrapper* self, u64 title_resource, const D3D12_DISCARD_REGION* region)
{
    auto* list = recording(self);
    if (!list || !title_resource) return;
    auto* resource = reinterpret_cast<ID3D12Resource*>(unwrap(title_resource));
    if (!resource) return;
    D3D12_DISCARD_REGION safe{};
    if (region) {
        if (!x2w_local::readable(region, sizeof(*region))) return;
        safe = *region;
        if (safe.NumRects && !x2w_local::readable(safe.pRects, std::size_t{safe.NumRects} * sizeof(D3D12_RECT))) return;
    }
    if (!tracked_from_common(title_resource)) return;
    const D3D12_RESOURCE_FLAGS flags = resource->GetDesc().Flags;
    D3D12_RESOURCE_STATES state;
    if (flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    else if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    else if (flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    else return;
    require_state(self, resource, state);
    list->DiscardResource(resource, region ? &safe : nullptr);
    static LONG logged;
    if (first_calls(logged, 4)) log("x2w-render DiscardResource resource=%p rects=%u", static_cast<void*>(resource), region ? safe.NumRects : 0);
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
    pass.targets_single_sample = false;
    pass.depth_view_writable = false;
    pass.target_count = 0;
    pass.target0 = nullptr;
    std::fill_n(pass.bound_targets, 8, nullptr);
    pass.bound_depth = nullptr;
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
    if (red_producers().enabled()) {
        std::copy_n(targets, 8, pass.trace_targets);
        pass.trace_depth = dsv_target;
    }
    for (UINT i = 0; i < count; ++i) {
        use_as_target(self, targets[i]);
        pass.target_formats[i] = formats[i];
        if (i == 0) pass.target0 = targets[i];
    }
    if (dsv_target) require_state(self, dsv_target, dsv_state);
    pass.target_count = count;
    std::copy_n(targets, 8, pass.bound_targets);
    pass.bound_depth = dsv_target;
    pass.dsv_format = dsv_format;
    const auto single_sample = [](ID3D12Resource* resource) {
        const auto samples = resource->GetDesc().SampleDesc;
        return samples.Count == 1 && !samples.Quality;
    };
    pass.targets_single_sample = std::all_of(targets, targets + count, single_sample) &&
        (!dsv_target || single_sample(dsv_target));
    pass.depth_view_writable = dsv_target && dsv_state == D3D12_RESOURCE_STATE_DEPTH_WRITE;
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
    if (auto* list = recording(self)) {
        pass_of(self).stencil_reference = reference;
        list->OMSetStencilRef(reference);
    }
}
// List slot 43 (desktop 39 + 4), set a compute root SRV backed by this recording's upload pages.
void __stdcall list_compute_upload_srv(Wrapper* self, UINT parameter, UINT64 address)
{
    auto* list = recording(self);
    auto& pass = pass_of(self);
    const auto* params = pass.compute_params;
    if (!list || !params || parameter >= params->size() || parameter >= pass.desktop_compute_upload_srvs.size() ||
        (*params)[parameter].type != D3D12_ROOT_PARAMETER_TYPE_SRV) return;
    const auto gpu = list_upload_gpu(self, address, 1);
    // A rejected replacement leaves no valid binding: the desktop list may still hold the previous address.
    if (parameter < 32) pass.compute_srv_bound &= ~(1u << parameter);
    if (gpu && !(gpu % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)) {
        bind_compute_root_view(pass, list, parameter, D3D12_ROOT_PARAMETER_TYPE_SRV, gpu);
        if (parameter < 32) pass.compute_srv_bound |= 1u << parameter;
    }
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
    if (pass.indirect_graphics.cached_mesh && pass.indirect_graphics.cached_srv_root == parameter)
        return; // The slot 61 draw binds the validated lease rather than an upload snapshot.
    if (const auto gpu = list_upload_gpu(self, address, 1)) {
        if (gpu % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) return;
        list->SetGraphicsRootShaderResourceView(parameter, gpu);
        pass.srv_bound |= 1u << parameter;
        return;
    }
    const std::size_t bytes = readable_window(address, 0x10000);
    const D3D12_GPU_VIRTUAL_ADDRESS desktop = bytes ? upload_from_title(self, address, bytes) : 0;
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
// List slot 14, Dispatch. Forwarded only with a desktop compute pipeline and root bound and group counts within the limit.
void __stdcall list_dispatch(Wrapper* self, UINT x, UINT y, UINT z)
{
    auto* list = recording(self);
    const Pass& pass = pass_of(self);
    if (list && pass.pipeline && pass.pipeline->own().compute_request_key) {
        record_translated_dispatch(self, {{x, y, z}});
        return;
    }
    x2w_startup::ListDispatch request;
    request.list_open = list != nullptr;
    if (list) request.type = list->GetType();
    request.compute_pipeline = pass.compute_pipeline;
    request.compute_root = pass.compute_root != nullptr;
    request.compute_root_has_arguments = pass.compute_root_has_arguments;
    request.root_matches_pipeline = pass.compute_pipeline_root && pass.compute_pipeline_root == pass.compute_root;
    const UINT count = pass.compute_params ? static_cast<UINT>(pass.compute_params->size()) : 0;
    request.root_arguments_bound = count <= 32 && (pass.compute_srv_bound & (count >= 32 ? ~0u : (1u << count) - 1)) == (count >= 32 ? ~0u : (1u << count) - 1);
    request.x = x, request.y = y, request.z = z;
    const x2w_startup::DispatchDecision decision = x2w_startup::decide_dispatch(request);
    if (decision != x2w_startup::DispatchDecision::forward) {
        static LONG refusals;
        if (first_calls(refusals, 16))
            log("x2w-render Dispatch(%u, %u, %u) refused: %s", x, y, z, x2w_startup::dispatch_reason(decision));
        return;
    }
    list->Dispatch(x, y, z);
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
        const auto red_samples = red_producer_draw_before(self, list, "native-draw", 0);
        list->DrawInstanced(vertices, instances, first_vertex, first_instance);
        red_producer_draw_after(self, red_samples);
        const LONG n = InterlockedIncrement(&render_draws);
        if (n > 0 && n <= 3) log("x2w-render DrawInstanced(%u, %u, %u, %u) recorded", vertices, instances, first_vertex, first_instance);
    } else {
        InterlockedIncrement(&render_draws_skipped);
    }
}

extern SRWLOCK written_lock;
struct TextureWrite {
    u64 generation = 0;
    UINT bc7_width = 0, bc7_height = 0;
    bool desktop_rows = false;  // the last XG copy wrote desktop-footprint rows (not the opt-in BC7 tiles)
};
std::unordered_map<u64, TextureWrite>& written_textures();
ID3D12Resource* sampled_address_texture(Wrapper* resource);
bool console_layout_srv(const D3D12_RESOURCE_DESC& desc);
#include "startup_stubs/d3d12_draw.inl"
#include "startup_stubs/d3d12_compute_dispatch.inl"
#include "startup_stubs/mesh86_cache.inl"
#include "startup_stubs/d3d12_compute_binding_trace.inl"
#include "startup_stubs/list_copy_texture.inl"

// X2W_D3D12_LINEAR_BUFFERS: the copied linear buffers are refreshed from title memory on the internal
// refresh queue, and queue waits for those refreshes, just before the title's lists. A refresh that
// fails leaves that buffer stale (logged, counted); the lists run either way. False: a refresh had no
// safe completion order and the submission is refused (logged, counted): nothing was enqueued, and the
// caller drops the title's lists without executing them or committing their bookkeeping. The title's
// fence Signal and Wait on its queues stay as they are, so a later Signal still releases a queue that
// waits on it.
bool linear_refresh(ID3D12CommandQueue* queue)
{
    return x2w_startup::linear_buffer_mode() == x2w_startup::LinearBufferMode::off || linear_buffers().before_submit(queue);
}
// After the title's lists on queue: a later refresh's copy waits for this submission (the producer wait).
void linear_submitted(ID3D12CommandQueue* queue)
{
    if (x2w_startup::linear_buffer_mode() != x2w_startup::LinearBufferMode::off) linear_buffers().after_submit(queue);
}
// The queue's order lock (linear_buffer_backing.h), so its title waits and user fence values are recorded in
// the order the desktop queue receives them: held by a submission from its ExecuteCommandLists through
// linear_submitted, and by queue slot 17 from the desktop Wait through note_title_wait. Taken before
// draw_lock and allocator_lock and with no other lock held; nothing while the switch is off.
x2w_startup::LinearBuffers::QueueOrder linear_queue_order(ID3D12CommandQueue* queue, bool title_wait)
{
    if (x2w_startup::linear_buffer_mode() == x2w_startup::LinearBufferMode::off) return {};
    return linear_buffers().queue_order(queue, title_wait);
}

// Queue slot 10, desktop ExecuteCommandLists(count, lists).
void __stdcall queue_execute(Wrapper* self, UINT count, void* const* lists)
{
    X2W_ZONE("ExecuteCommandLists");
    static LONG calls;
    drain_retired_resources();
    if (render_mode()) {
        // A closed list stays submittable until its next Reset, as on the desktop.
        std::vector<ID3D12CommandList*> ready;
        std::vector<Wrapper*> submitted;
        auto* queue = static_cast<ID3D12CommandQueue*>(self->own().real);
        // Before the wrapper's locks: a refresh may wait for a staging fence, and recording must not
        // wait with it. A refused submission runs nothing: its recordings stay closed and submittable,
        // with their pending GPU writes (committed only after an executed submission).
        if (count && !linear_refresh(queue)) return;
        // From here through linear_submitted, a title Wait on this queue (slot 17) on another thread is
        // recorded either before this submission's user value or after it, as the desktop queue gets them.
        auto order = linear_queue_order(queue, false);
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
        bool split = false;
        const auto submit = [&](const auto& native_lists, const auto& wrappers) {
            const UINT ready_count = static_cast<UINT>(native_lists.size());
            const int gpu_slot = gpu_time_on() ? gpu_time_probe().before(queue, static_cast<std::uint64_t>(gpu_time_frame)) : -1;
            red_producers().execute(queue, native_lists, [&] {
                if (draw_dump_frame()) {
                    const HRESULT hr = draw_dumps().execute(queue, ready_count, native_lists.data());
                    if (FAILED(hr)) log("x2w-draw readback fence failed 0x%08lx", static_cast<unsigned long>(hr));
                } else {
                    queue->ExecuteCommandLists(ready_count, native_lists.data());
                }
            });
            if (gpu_slot >= 0) gpu_time_probe().after(queue, gpu_slot);
            retire_address_uploads(queue, native_lists);
            retire_draw_sampler_heaps(queue, native_lists);
            note_submits(queue, wrappers);
            retire_note_submission(queue, wrappers);
            residency_note_submission(queue, wrappers);
            for (Wrapper* w : wrappers) commit_list_gpu_writes(w);
            InterlockedIncrement(&render_executes);
            return trace_mesh_xmem_submission(queue, native_lists, split);
        };
        if (mesh_xmem_tracking()) {
            const std::lock_guard lock(mesh_xmem_mutex);
            reap_mesh_xmem_flights();
            split = std::any_of(ready.begin(), ready.end(), mesh_xmem_needs_submission);
        }
        if (split) {
            // Submit each recording in guest order so a writer's publication
            // belongs to its own completion point, even within a guest batch.
            std::vector<ID3D12CommandList*> native_list(1);
            std::vector<Wrapper*> wrapper(1);
            for (std::size_t i = 0; i < ready.size(); ++i) {
                native_list[0] = ready[i];
                wrapper[0] = submitted[i];
                execute_mesh_xmem_recording(queue, ready[i], [&] { return submit(native_list, wrapper); });
            }
        } else if (!ready.empty()) {
            submit(ready, submitted);
        }
        if (!ready.empty()) linear_submitted(queue);
        ReleaseSRWLockExclusive(&allocator_lock);
        ReleaseSRWLockExclusive(&draw_lock);
        order.release();
        check_device("CommandQueue execute", 35);
        return;
    }
    if (list_noop()) {
        if (first_calls(calls)) log("x2w-d3d12 CommandQueue::ExecuteCommandLists count=%u -> not submitted (noop lists)", count);
        return;
    }
    // No hold or flight is recorded here: the copy handlers that create holds exist only in render mode.
    std::vector<ID3D12CommandList*> real(count);
    for (UINT i = 0; i < count; ++i) real[i] = reinterpret_cast<ID3D12CommandList*>(unwrap(reinterpret_cast<u64>(lists[i])));
    auto* queue = static_cast<ID3D12CommandQueue*>(self->own().real);
    if (!linear_refresh(queue)) return;
    {
        const auto order = linear_queue_order(queue, false);  // as in render mode above
        queue->ExecuteCommandLists(count, real.data());
        if (count) linear_submitted(queue);
    }
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

// The page grants, section offsets and destination views are owned by XMem.
// Hold the installed provider for this batch; a queue can predate xmem.dll.
HRESULT WINAPI queue_page_mappings(Wrapper*, UINT count, const X2WPageBatch* batches,
                                   UINT64 pool, UINT pool_pages, UINT flags)
{
    HMODULE xmem = nullptr;
    if (!GetModuleHandleExW(0, L"xmem.dll", &xmem)) return HRESULT_FROM_WIN32(GetLastError());
    const auto map = reinterpret_cast<decltype(&X2WMapPageBatch)>(GetProcAddress(xmem, "X2WMapPageBatch"));
    const HRESULT hr = map ? map(count, batches, pool, pool_pages, flags, nullptr)
                           : HRESULT_FROM_WIN32(ERROR_PROC_NOT_FOUND);
    FreeLibrary(xmem);
    if (FAILED(hr)) {
        static LONG failures;
        if (first_calls(failures, 2))
            log("x2w-d3d12 CommandQueue::slot31 count=%u poolPages=%u flags=0x%x -> 0x%08lx",
                count, pool_pages, flags, static_cast<unsigned long>(hr));
    }
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
    auto* queue = static_cast<ID3D12CommandQueue*>(self->own().real);
    // X2W_D3D12_LINEAR_BUFFERS: until this wait is reached, a submission whose refresh would have to wait for a
    // submission behind it is refused. The queue's order lock spans the desktop Wait and its record, so a
    // submission on this queue from another thread takes its user value wholly before or after both.
    HRESULT hr;
    {
        const auto order = linear_queue_order(queue, true);
        hr = queue->Wait(static_cast<ID3D12Fence*>(fence->own().real), r8);
        if (SUCCEEDED(hr) && x2w_startup::linear_buffer_mode() != x2w_startup::LinearBufferMode::off)
            linear_buffers().note_title_wait(queue, static_cast<ID3D12Fence*>(fence->own().real), r8);
    }
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
    if (!mapped) {
        ID3D12Resource* real = materialize_lazy_texture(self);
        hr = real ? real->Map(subresource, range, &mapped) : E_FAIL;
    }
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
    if (!self->own().title_memory && self->own().real)
        static_cast<ID3D12Resource*>(self->own().real)->Unmap(subresource, range);
}

// Resource slot 11, desktop GetGPUVirtualAddress() (called with no arguments). On the console a
// resource placed at an address has that address as its GPU address, so the title gets its own
// address back; resources without title memory report the desktop address (0 for textures).
UINT64 __stdcall resource_gpu_address(Wrapper* self)
{
    const UINT64 desktop = (self->own().real &&
        static_cast<ID3D12Resource*>(self->own().real)->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
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
    if (!self || !result) return result;
    if (self->own().real) {
        *static_cast<D3D12_RESOURCE_DESC*>(result) = static_cast<ID3D12Resource*>(self->own().real)->GetDesc();
        if (self->own().drop_levels) lazy_texture_desc(self, *static_cast<D3D12_RESOURCE_DESC*>(result));  // the title's size
    } else if (!lazy_texture_desc(self, *static_cast<D3D12_RESOURCE_DESC*>(result))) {
        return result;
    }
    static LONG logged;
    if (first_calls(logged, 4)) {
        const auto& d = *static_cast<const D3D12_RESOURCE_DESC*>(result);
        log("x2w-d3d12 Resource::GetDesc res=%p dim=%d fmt=%d size=%llux%ux%u mips=%u layout=0x%x flags=0x%x",
            static_cast<void*>(self), d.Dimension, d.Format, static_cast<unsigned long long>(d.Width),
            d.Height, d.DepthOrArraySize, d.MipLevels, d.Layout, d.Flags);
    }
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
// pointer passed first. receiver() decides which argument is the object: a wrapper first is the object
// even when the caller left another live wrapper in the second register, which a call passing only
// `this` does not set. Only the (return buffer, object) form writes through the first argument.
template <class T, class Get>
std::uint64_t small_result(void* a, void* b, const char* method, Get get)
{
    void* result = nullptr;
    Wrapper* w = receiver(a, b, &result, method);
    const T value = get(w);
    if (w != a) {
        *static_cast<T*>(result) = value;
        return reinterpret_cast<std::uint64_t>(result);
    }
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
    const std::uint64_t supplied_memory = title_memory;
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
    if (!w) {
        if (title_memory != supplied_memory) VirtualFree(reinterpret_cast<void*>(title_memory), 0, MEM_RELEASE);
        heap->Release();
        return E_OUTOFMEMORY;
    }
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

// A null R8G8B8A8_UNORM SRV of a texture dimension (2 to 10; any other value gives texture 2D). A null resource
// reads no more of the description than its format and dimension.
D3D12_SHADER_RESOURCE_VIEW_DESC null_srv_desc(unsigned dimension)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.ViewDimension = static_cast<D3D12_SRV_DIMENSION>(x2w_view_dimension::refused_srv_null_dimension(dimension));
    switch (desc.ViewDimension) {
    case D3D12_SRV_DIMENSION_TEXTURE1D: desc.Texture1D.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY: desc.Texture1DArray.MipLevels = 1; desc.Texture1DArray.ArraySize = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY: desc.Texture2DArray.MipLevels = 1; desc.Texture2DArray.ArraySize = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY: desc.Texture2DMSArray.ArraySize = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE3D: desc.Texture3D.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE: desc.TextureCube.MipLevels = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: desc.TextureCubeArray.MipLevels = 1; desc.TextureCubeArray.NumCubes = 1; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMS: break;
    default: desc.Texture2D.MipLevels = 1; break;
    }
    return desc;
}

// X2W_D3D12_VIEW_DEFAULTS (view_dimension.h): which slots give an unknown-dimension view the resource's default view.
x2w_view_dimension::DefaultViews view_defaults_mode()
{
    static const x2w_view_dimension::DefaultViews mode = [] {
        char value[8]{};
        const DWORD n = GetEnvironmentVariableA("X2W_D3D12_VIEW_DEFAULTS", value, sizeof(value));
        return x2w_view_dimension::default_views_mode(n && n < sizeof(value) ? value : nullptr);
    }();
    return mode;
}

// Explicit SRV description for CreateShaderResourceView(nullptr, &desc, dest) when a slot-133
// lazy texture has not materialized yet. D3D12 requires a non-null desc with known dimension and format.
D3D12_SHADER_RESOURCE_VIEW_DESC lazy_null_srv_desc(const D3D12_RESOURCE_DESC& res_desc,
                                                  const D3D12_SHADER_RESOURCE_VIEW_DESC* requested)
{
    unsigned dimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    if (requested && requested->ViewDimension != D3D12_SRV_DIMENSION_UNKNOWN) {
        dimension = static_cast<unsigned>(requested->ViewDimension);
    } else {
        switch (res_desc.Dimension) {
        case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
            dimension = res_desc.DepthOrArraySize > 1
                ? D3D12_SRV_DIMENSION_TEXTURE1DARRAY : D3D12_SRV_DIMENSION_TEXTURE1D;
            break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
            dimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
        default:
            if (res_desc.SampleDesc.Count > 1) {
                dimension = res_desc.DepthOrArraySize > 1
                    ? D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY : D3D12_SRV_DIMENSION_TEXTURE2DMS;
            } else {
                dimension = res_desc.DepthOrArraySize > 1
                    ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY : D3D12_SRV_DIMENSION_TEXTURE2D;
            }
            break;
        }
    }
    return null_srv_desc(dimension);
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
    // Slots 18-21 name a resource in a descriptor slot. While shadow_slot() tracks the destination, note_view() owns the
    // slot's reference: the view (slot 19 for its resource and its counter resource) holds the resource in the retire
    // class until the slot is overwritten, copied over, or its heap is destroyed. Without a tracked slot the shim cannot
    // follow the descriptor, so the use takes the resource out of the class for good, as before.
    if constexpr (N >= 19 && N <= 21) {
        bool tracked = false;
        if (args[D]) {
            AcquireSRWLockShared(&note_lock);
            tracked = shadow_slot(args[D]) != nullptr;
            ReleaseSRWLockShared(&note_lock);
        }
        if (!tracked) {
            retire_leave_class(reinterpret_cast<const void*>(a), RetireLeft::view);
            if constexpr (N == 19) retire_leave_class(reinterpret_cast<const void*>(b), RetireLeft::view);
        }
    }
    // No desktop destination handle: the view stays title-side, where no shadow slot can follow it.
    if (!args[D]) {
        if constexpr (N == 18) retire_leave_class(reinterpret_cast<const void*>(a), RetireLeft::view);
        return 0;
    }
    const auto record_view = [&](bool materialized) {
        const u64 raw[4] = {a, b, c, d};
        const int described = N == 19 ? 2 : (N == 17 || N == 22) ? 0 : 1;
        const std::size_t described_bytes = N == 17 ? 16 : N == 22 ? 52 : N == 18 ? 40 : N == 19 ? 40 : N == 20 ? 20 : 16;
        note_view(N, (N >= 18 && N <= 21) ? raw[0] : 0, N == 19 ? raw[1] : 0, reinterpret_cast<const void*>(raw[described]),
                  described_bytes, title_dest, args[D], materialized);
    };
    Wrapper* const lazy_wrapper = (N >= 18 && N <= 21) ? as_wrapper(reinterpret_cast<const void*>(a)) : nullptr;
    const bool maybe_lazy = lazy_wrapper && lazy_wrapper->own().kind == resource_kind &&
        lazy_wrapper->own().lazy_slot133_texture && !lazy_wrapper->own().real;
    LazyTextureRecord lazy_rec{};
    bool is_unmaterialized_lazy = false;
    if (maybe_lazy) {
        std::lock_guard lock(lazy_texture_mutex);
        if (lazy_wrapper->own().real) {
            args[0] = reinterpret_cast<u64>(lazy_wrapper->own().real);
        } else {
            const auto found = lazy_textures().find(lazy_wrapper);
            if (found != lazy_textures().end()) {
                lazy_rec = found->second;
                is_unmaterialized_lazy = true;
            }
        }
    }
    const bool has_resource = args[0] != 0 || is_unmaterialized_lazy;
    const D3D12_RESOURCE_DESC resource_desc_for_check = is_unmaterialized_lazy ? lazy_rec.desc
        : (args[0] ? reinterpret_cast<ID3D12Resource*>(args[0])->GetDesc() : D3D12_RESOURCE_DESC{});
    const char* skip = nullptr;
    bool default_view = false;
    bool uav_denied = false;  // a UAV of a resource without unordered access: recorded as not created, as before #2649
    if (N == 17) skip = "constant buffer view (title GPU address)";
    if (N >= 18 && N <= 21 && has_resource) {
        const UINT flags = resource_desc_for_check.Flags;
        if (N == 18 && (flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) skip = "resource denies shader resource views";
        if (N == 19 && !(flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
            skip = "resource lacks unordered access";
            uav_denied = true;
        }
        if (N == 20 && !(flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) skip = "resource lacks render target";
        if (N == 21 && !(flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) skip = "resource lacks depth stencil";
        // Every view description of slots 18-21 starts with Format, then ViewDimension.
        const u64 described = N == 19 ? args[2] : args[1];
        UINT view_dimension = 0;
        if (!skip && described && x2w_local::readable(reinterpret_cast<const void*>(described), 8)) {
            std::memcpy(&view_dimension, reinterpret_cast<const unsigned char*>(described) + 4, sizeof(view_dimension));
            const auto dimension = static_cast<unsigned>(resource_desc_for_check.Dimension);
            if (view_dimension == 0) {
                // UNKNOWN: desktop refuses it; make the resource's default view where that is allowed. By default only
                // SRVs (slot 18) get one; UAV, RTV and DSV need X2W_D3D12_VIEW_DEFAULTS=all (view_dimension.h).
                const auto mode = view_defaults_mode();
                if (x2w_view_dimension::default_view_allowed(N, dimension, static_cast<unsigned>(resource_desc_for_check.Format), mode))
                    default_view = true;
                else if (x2w_view_dimension::default_view_allowed(N, dimension, static_cast<unsigned>(resource_desc_for_check.Format),
                                                                  x2w_view_dimension::DefaultViews::all))
                    skip = mode == x2w_view_dimension::DefaultViews::off
                        ? "unknown view dimension, and default views are off (X2W_D3D12_VIEW_DEFAULTS=off)"
                        : "unknown view dimension; a default view of this type needs X2W_D3D12_VIEW_DEFAULTS=all";
                else
                    skip = "unknown view dimension, and the resource has no default view";
                static LONG unknown_dimension;
                if (first_calls(unknown_dimension, 6))
                    log("x2w-d3d12 Device::slot%d view dimension 0 (unknown), view format %u, resource dim=%u fmt=%d -> %s", N,
                        *reinterpret_cast<const UINT*>(described), dimension, resource_desc_for_check.Format,
                        default_view ? "default view" : "skipped");
            } else if (!x2w_view_dimension::fits(N, view_dimension, dimension)) {
                skip = "view dimension does not suit the resource";
                static LONG mismatched;
                if (first_calls(mismatched, 6))
                    log("x2w-d3d12 Device::slot%d view dimension %u needs resource dimension %u, resource has %u", N, view_dimension,
                        x2w_view_dimension::needed(N, view_dimension), dimension);
            }
        }
    }
    // Replace a refused SRV in both the desktop heap and the descriptor record. A skipped UAV, RTV or DSV
    // does not write a replacement to the desktop heap, so its previous descriptor record must stay live.
    const auto refuse_view = [&] {
        if constexpr (N == 18) {
            UINT requested = D3D12_SRV_DIMENSION_TEXTURE2D;
            if (args[1] && x2w_local::readable(reinterpret_cast<const void*>(args[1]), 8))
                std::memcpy(&requested, reinterpret_cast<const unsigned char*>(args[1]) + 4, sizeof(requested));
            const D3D12_SHADER_RESOURCE_VIEW_DESC null_view = null_srv_desc(requested);
            device_of(self)->CreateShaderResourceView(nullptr, &null_view, D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(args[D])});
            record_view(false);
        } else if constexpr (N == 17) {
            record_view(false);  // Translated constant-buffer views use only the descriptor record.
        } else if constexpr (N == 19) {
            // A translated compute Dispatch tells this refusal from a descriptor that holds no view by the record.
            // A refused creation leaves a descriptor that still holds a live view as it is.
            if (uav_denied && !descriptor_has_live_view(args[D], title_dest)) record_view(false);
        }
    };
    if (skip) {
        refuse_view();
        static LONG skipped;
        if (first_calls(skipped, 6)) log("x2w-d3d12 Device::slot%d view skipped: %s", N, skip);
        return 0;
    }
    // The view descriptions of slots 18-21 start with a DXGI_FORMAT.
    alignas(D3D12_SHADER_RESOURCE_VIEW_DESC) unsigned char desc_copy[64];
    const int desc_index = N == 19 ? 2 : 1;
    constexpr std::size_t view_bytes = N == 18 ? sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC) :
        N == 19 ? sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC) :
        N == 20 ? sizeof(D3D12_RENDER_TARGET_VIEW_DESC) : sizeof(D3D12_DEPTH_STENCIL_VIEW_DESC);
    if (N >= 18 && N <= 21 && args[desc_index] && x2w_local::readable(reinterpret_cast<const void*>(args[desc_index]), view_bytes)) {
        std::memcpy(desc_copy, reinterpret_cast<const void*>(args[desc_index]), view_bytes);
        DXGI_FORMAT format;
        std::memcpy(&format, desc_copy, sizeof(format));
        DXGI_FORMAT mapped = desktop_format(format);
        if (has_resource && x2w_startup::packed_target_format(mapped)) {
            // A packed view follows a stand-in texture and is refused on any other texture of another format.
            const auto actual = resource_desc_for_check.Format;
            const auto* mapping = is_unmaterialized_lazy ? x2w_startup::packed_target_mapping(lazy_rec.desktop_format) : nullptr;
            const DXGI_FORMAT stand_in_for = is_unmaterialized_lazy
                ? (mapping && actual == mapping->replacement ? lazy_rec.desktop_format : DXGI_FORMAT_UNKNOWN)
                : x2w_startup::packed_stand_in_of(reinterpret_cast<ID3D12Resource*>(args[0]));
            const auto verdict = x2w_startup::packed_view(mapped, actual, stand_in_for);
            if (verdict == x2w_startup::PackedView::refused) {
                refuse_view();
                static LONG refused;
                if (first_calls(refused, 6))
                    log("x2w-d3d12 Device::slot%d view skipped: packed format %d on a texture of format %d that does not stand in for it",
                        N, mapped, actual);
                return 0;
            }
            if (verdict == x2w_startup::PackedView::stand_in) mapped = actual;
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
    if (default_view) args[desc_index] = 0;
    if (is_unmaterialized_lazy) {
        if constexpr (N == 18) {
            bool handled_lazy = false;
            bool created = false;
            HRESULT removed = S_OK;
            {
                std::lock_guard lock(lazy_texture_mutex);
                if (lazy_wrapper->own().real) {
                    args[0] = reinterpret_cast<u64>(lazy_wrapper->own().real);
                } else {
                    const D3D12_SHADER_RESOURCE_VIEW_DESC* requested =
                        args[1] && x2w_local::readable(reinterpret_cast<const void*>(args[1]), sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC))
                        ? reinterpret_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC*>(args[1])
                        : nullptr;
                    const D3D12_SHADER_RESOURCE_VIEW_DESC placeholder = lazy_null_srv_desc(lazy_rec.desc, requested);
                    device_of(self)->CreateShaderResourceView(
                        nullptr, &placeholder, D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(args[D])});
                    removed = removal_device ? removal_device->GetDeviceRemovedReason() : S_OK;
                    created = SUCCEEDED(removed);
                    record_view(created);
                    handled_lazy = true;
                }
            }
            if (handled_lazy) {
                if (!created) {
                    static LONG failed;
                    if (first_calls(failed, 6))
                        log("x2w-d3d12 Device::slot%d lazy view not created: device removed, reason 0x%08lx", N,
                            static_cast<unsigned long>(removed));
                }
                check_device("Device view", N);
                static LONG lazy_calls;
                if (first_calls(lazy_calls))
                    log("x2w-d3d12 Device::slot18 lazy SRV(0x%llx, 0x%llx, 0x%llx) dest 0x%llx -> 0x%llx (deferred)", a, b, c,
                        title_dest, args[D]);
                return 0;
            }
        } else {
            args[0] = reinterpret_cast<u64>(materialize_lazy_texture(lazy_wrapper));
        }
    }
    const LONG removed_before = removal_reported;
    const u64 result = method_of(self->own().real, N)(self->own().real, args[0], args[1], args[2], args[3], 0, 0, 0, 0);
    // A desktop view creation returns void, so the only failure it reports is a removed device. Only a created view
    // may keep the resource: on failure the slot is recorded unmaterialized, which releases the slot's previous view
    // reference and adds none.
    const HRESULT removed = removal_device ? removal_device->GetDeviceRemovedReason() : S_OK;
    const bool created = SUCCEEDED(removed);
    record_view(created);
    if (!created) {
        static LONG failed;
        if (first_calls(failed, 6))
            log("x2w-d3d12 Device::slot%d view not created: device removed, reason 0x%08lx", N,
                static_cast<unsigned long>(removed));
    }
    if ((N == 20 || N == 21) && args[0] && created)
        note_view_target(args[D], reinterpret_cast<ID3D12Resource*>(args[0]), N);
    check_device("Device view", N);
    if (!removed_before && removal_reported && N >= 18 && N <= 21) {
        // The view that removed the device: its forwarded description and the resource's shape.
        char bytes[3 * 40 + 1] = {};
        const u64 described = args[desc_index];
        if (described) hex_text(reinterpret_cast<const void*>(described), 40, bytes, sizeof(bytes));
        log("x2w-d3d12 Device::slot%d removing view desc %s", N, described ? bytes : "(null)");
        if (args[0]) {
            const D3D12_RESOURCE_DESC rd = reinterpret_cast<ID3D12Resource*>(args[0])->GetDesc();
            log("x2w-d3d12 Device::slot%d removing view resource: dim=%d %llux%u depth/array=%u mips=%u fmt=%d samples=%u flags=0x%x", N,
                rd.Dimension, static_cast<unsigned long long>(rd.Width), rd.Height, rd.DepthOrArraySize, rd.MipLevels, rd.Format,
                rd.SampleDesc.Count, static_cast<UINT>(rd.Flags));
        }
    }
    static LONG calls;
    if (first_calls(calls))
        log("x2w-d3d12 Device::slot%d view(0x%llx, 0x%llx, 0x%llx, 0x%llx) dest 0x%llx -> 0x%llx", N, a, b, c, d,
            D == 0 ? a : D == 1 ? b : D == 2 ? c : d, args[D]);
    return result;
}

// Writes the desktop SRV for a lazy texture's view record into one descriptor slot, as the title's view described it.
// The caller holds note_lock shared and has checked that the slot still names the record.
void create_lazy_srv(ID3D12Device* device, ID3D12Resource* created, const D3D12_RESOURCE_DESC& created_desc,
                     x2w_view_dimension::DefaultViews view_mode, const ViewRecord& vr, D3D12_CPU_DESCRIPTOR_HANDLE dest,
                     unsigned dropped_levels = 0)
{
    if (vr.desc_bytes < sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)) {
        device->CreateShaderResourceView(created, nullptr, dest);
        return;
    }
    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    std::memcpy(&srv_desc, vr.desc, sizeof(srv_desc));
    if (srv_desc.ViewDimension == D3D12_SRV_DIMENSION_UNKNOWN) {
        if (x2w_view_dimension::default_view_allowed(18, static_cast<unsigned>(created_desc.Dimension),
                static_cast<unsigned>(created_desc.Format), view_mode)) {
            device->CreateShaderResourceView(created, nullptr, dest);
        }
        return;
    }
    srv_desc.Format = desktop_format(srv_desc.Format);
    if (dropped_levels) {
        // A view that starts below the top level or clamps the LOD names levels that are gone: leave it unbound.
        // The others lose the same count of levels the texture lost.
        const bool array = srv_desc.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        if ((srv_desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D && !array) ||
            srv_desc.Texture2D.MostDetailedMip != 0 || (!array && srv_desc.Texture2D.ResourceMinLODClamp != 0.0f)) {
            x2w_startup::drop_mips::counters.keep(x2w_startup::drop_mips::Reason::view_clamp);
            return;
        }
        srv_desc.Texture2D.MipLevels = x2w_startup::drop_mips::view_mip_levels(srv_desc.Texture2D.MipLevels, dropped_levels);
    }
    if (x2w_startup::packed_target_format(srv_desc.Format)) {
        const auto verdict = x2w_startup::packed_view(
            srv_desc.Format, created_desc.Format, x2w_startup::packed_stand_in_of(created));
        if (verdict == x2w_startup::PackedView::refused) return;
        if (verdict == x2w_startup::PackedView::stand_in) srv_desc.Format = created_desc.Format;
    }
    device->CreateShaderResourceView(created, &srv_desc, dest);
}

// After a guest descriptor copy: a destination slot that now names a lazy texture which already has its real resource got
// the native NULL descriptor from the desktop copy when the copy ran while the texture was being published (its
// refresh had taken its snapshot already, or the copy raced the real descriptor). Write the real view into such slots.
// A texture that has no real resource yet is handled by its own refresh, which runs after this slot is recorded.
void refresh_copied_lazy_views(Wrapper* self, u64 dest_desktop, UINT count, UINT increment)
{
    if (!lazy_textures_enabled() || !count) return;
    struct Candidate {
        std::uint32_t index, id;
        Wrapper* texture;
    };
    std::vector<Candidate> candidates;
    try {
        AcquireSRWLockShared(&note_lock);
        std::uint32_t room = 0;
        const std::uint32_t* ids = shadow_slot(dest_desktop, &room);
        if (ids && count <= room) {
            for (UINT i = 0; i < count; ++i) {
                const std::uint32_t id = ids[i];
                if (!id || id - 1 >= view_records().size() || !view_materialized()[id - 1]) continue;
                const ViewRecord& vr = view_records()[id - 1];
                if (vr.slot == 18 && vr.resource) candidates.push_back({i, id, reinterpret_cast<Wrapper*>(vr.resource)});
            }
        }
        ReleaseSRWLockShared(&note_lock);
    } catch (const std::bad_alloc&) {
        ReleaseSRWLockShared(&note_lock);
        return;
    }
    for (const Candidate& candidate : candidates) {
        ID3D12Resource* real = nullptr;
        {
            // lazy_texture_mutex comes before note_lock in the lock order; a texture not in the map is not lazy.
            std::lock_guard lock(lazy_texture_mutex);
            const auto found = lazy_textures().find(candidate.texture);
            if (found == lazy_textures().end()) continue;
            real = static_cast<ID3D12Resource*>(candidate.texture->own().real);
            if (real) real->AddRef();
        }
        if (!real) continue;
        const D3D12_RESOURCE_DESC real_desc = real->GetDesc();
        const auto view_mode = view_defaults_mode();
        AcquireSRWLockShared(&note_lock);
        std::uint32_t room = 0;
        const std::uint32_t* ids = shadow_slot(dest_desktop, &room);
        // The slot must still name the record, and the record still the texture (the title may have overwritten it).
        if (ids && candidate.index < room && ids[candidate.index] == candidate.id && candidate.id - 1 < view_records().size() &&
            view_materialized()[candidate.id - 1] && view_records()[candidate.id - 1].slot == 18 &&
            view_records()[candidate.id - 1].resource == reinterpret_cast<u64>(candidate.texture)) {
            const D3D12_CPU_DESCRIPTOR_HANDLE dest{static_cast<SIZE_T>(dest_desktop + u64{candidate.index} * increment)};
            create_lazy_srv(device_of(self), real, real_desc, view_mode, view_records()[candidate.id - 1], dest,
                           candidate.texture->own().drop_levels);
        }
        ReleaseSRWLockShared(&note_lock);
        real->Release();
    }
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
        const LONG published_before = InterlockedCompareExchange(&lazy_published_total, 0, 0);
        const bool publishing_before = InterlockedCompareExchange(&lazy_publishing, 0, 0) != 0;
        device_of(self)->CopyDescriptorsSimple(chunk, {static_cast<SIZE_T>(d)}, {static_cast<SIZE_T>(s)}, type);
        note_copy(dest, src, d, s, chunk, increment);
        // A copy that overlapped the publication of a lazy texture may have copied its native NULL descriptor.
        if (publishing_before || InterlockedCompareExchange(&lazy_publishing, 0, 0) != 0 ||
            InterlockedCompareExchange(&lazy_published_total, 0, 0) != published_before)
            refresh_copied_lazy_views(self, d, chunk, increment);
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
    if (!child) {
        list->Release();
        return E_OUTOFMEMORY;
    }
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
struct ExistingHeap {
    ID3D12Heap* real;
    x2w_budget::Ticket* budget;
};
std::vector<std::pair<void*, ExistingHeap>> existing_heaps;

// Lifetime boundary: one cross-adapter page alias per (heap, page) at a title address, kept for as long as
// this module is loaded, because the GPU may still read a page the title keeps live for the device's
// lifetime. It is the function-local static of placed_page_aliases() below, so it is destroyed during
// this module's CRT teardown -- see startup_stubs/d3d12_teardown.h for the two detaches. On process exit
// the display driver may already be stopped, so the placed buffers are detached and nothing reaches it;
// FreeLibrary leaves the process and the driver alive and the pages release their objects normally. The
// heap and page words are plain addresses, not owned. Each owned buffer is accounted for on the path it
// took and, with X2W_TEARDOWN_REPORT, reported as one x2w-teardown pages line.
struct PlacedPageAliases {
    struct Page {
        ID3D12Heap* heap;
        u64 page;
        x2w_startup::ComPtr<ID3D12Resource> resource;
    };
    using iterator = std::array<Page, 64>::iterator;
    iterator begin() { return entries.begin(); }
    iterator end() { return entries.end(); }

    // Test hook (startup_teardown_unload, #2572; the review of the ownership change asked for one):
    // enters one alias for a heap and page the caller owns, where the translated indirect address route
    // would keep it after its first call. False when that pair is already there or the cache is full. No
    // real path calls it, and the entry keeps the heap as a key only, as the route does.
    bool insert_for_test(ID3D12Heap* heap, u64 page, ID3D12Resource* resource)
    {
        if (!heap || !resource) return false;
        for (const Page& entry : entries)
            if (entry.heap == heap && entry.page == page) return false;
        for (Page& entry : entries) {
            if (entry.heap) continue;
            entry.resource = resource;  // AddRef: the cache owns this reference
            entry.page = page;
            entry.heap = heap;
            return true;
        }
        return false;
    }

    ~PlacedPageAliases()
    {
        const bool detach = x2w_startup::d3d12_process_terminating && !x2w_startup::teardown_guard_disabled("pages");
        unsigned detached = 0, released = 0;
        for (auto& entry : entries) {
            if (!entry.resource) continue;
            if (detach) {
                entry.resource.Detach();
                ++detached;
            } else {
                ++released;
            }
        }
        x2w_startup::teardown_report("pages", detached, released);
    }
    std::array<Page, 64> entries{};
};

// The cache lives in its own accessor so the fixture probe can reach it, the way
// indexed_validation_pipeline(device) exposes the validation pipeline. One function-local static, built
// on the first translated indirect address.
PlacedPageAliases& placed_page_aliases()
{
    static PlacedPageAliases pages{};
    return pages;
}

IndirectAddressAlias indirect_address_alias(Wrapper* device, u64 address, u64 bytes)
{
    static const bool enabled = x2w_local::env_on("X2W_TRANSLATED_INDIRECT_ADDRESS_ALIAS");
    if (!enabled || !device || device != the_device || !address || !bytes || bytes > UINT64_MAX - address)
        return {};
    constexpr u64 page_bytes = 65536;
    const u64 page = address & ~(page_bytes - 1);
    const u64 offset = address - page;
    if (bytes > page_bytes - offset) return {};
    x2w_startup::PlacedBufferRange memory{};
    if (!x2w_startup::measure_placed_buffer(reinterpret_cast<void*>(page), page_bytes, memory) ||
        !memory.covered || memory.type != MEM_PRIVATE) return {};
    const auto nearby = placed_buffer_range(device, page, page_bytes, true);
    if (nearby.reject || nearby.owner) return {};

    // Reuse the existing allocation heap opened by slot 133. It is already held
    // for the startup device's lifetime. The caller keeps this address allocation
    // live through the GPU work and must not activate an overlapping placed alias.
    std::lock_guard lock(placed_mutex);
    const auto known = std::find_if(existing_heaps.begin(), existing_heaps.end(),
        [&](const auto& entry) { return entry.first == memory.base; });
    if (known == existing_heaps.end()) return {};
    auto* heap = known->second.real;
    using Page = PlacedPageAliases::Page;
    auto& pages = placed_page_aliases();
    auto it = std::find_if(pages.begin(), pages.end(), [&](const Page& entry) { return entry.heap == heap && entry.page == page; });
    if (it != pages.end()) return {it->resource.Get(), offset};
    it = std::find_if(pages.begin(), pages.end(), [](const Page& entry) { return !entry.heap; });
    if (it == pages.end()) return {};
    x2w_startup::ComPtr<ID3D12Device> owner;
    auto* desktop = device_of(device);
    if (FAILED(heap->GetDevice(IID_PPV_ARGS(&owner))) || owner.Get() != desktop) return {};
    const auto base = reinterpret_cast<u64>(memory.base);
    if (base > page || (page - base) % page_bytes) return {};
    const u64 heap_size = heap->GetDesc().SizeInBytes;
    if (page - base > heap_size || page_bytes > heap_size - (page - base)) return {};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, page_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
        {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER};
    if (FAILED(desktop->CreatePlacedResource(heap, page - base, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&it->resource)))) return {};
    it->heap = heap;
    it->page = page;
    return {it->resource.Get(), offset};
}
volatile LONG64 placed_buffer_calls, placed_buffer_region_info, placed_buffer_queries;  // five-second totals

// Wraps a placed buffer's desktop resource and publishes it in the placement index. aliased: the
// resource reads and writes the title memory itself, placed on a heap whose budget is backing_budget
// (or none); otherwise it is a committed default buffer.
HRESULT publish_placed_buffer(Wrapper* self, void* address, const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initial_state,
                              ID3D12Resource* resource, bool aliased, x2w_budget::Ticket* backing_budget, void** out)
{
    Wrapper* w = wrap(resource, resource_kind);
    if (!w) {
        if (x2w_startup::linear_buffer_mode() != x2w_startup::LinearBufferMode::off) linear_buffers().discard(resource);
        resource->Release();
        return E_OUTOFMEMORY;
    }
    if (release_retired_enabled()) w->own().retire_never = static_cast<std::uint8_t>(RetireNever::buffer);
    w->own().title_memory = address;
    const ResourceInfo record{*desc, desc->Width, aliased ? Placement::aliased_buffer : Placement::copied_buffer, self, initial_state};
    auto& group = placement_group(desc->Width);
    const HRESULT hr = x2w_startup::publish_placement(note_lock, resource_infos, group.by_start, group.max_width,
        static_cast<const Wrapper*>(w), reinterpret_cast<u64>(address), record, [&] { index_resource(w, record); x2w_startup::note_placement_change(); });
    if (FAILED(hr)) {
        // The caller has never received this wrapper; release its desktop resource and storage.
        AcquireSRWLockExclusive(&registry_lock);
        registry().erase(w);
        ReleaseSRWLockExclusive(&registry_lock);
        if (x2w_startup::linear_buffer_mode() != x2w_startup::LinearBufferMode::off) linear_buffers().discard(resource);
        resource->Release();
        _aligned_free(&w->own());
        return hr;
    }
    if (aliased) {
        w->own().budget = x2w_budget::resource(device_of(self), resource, x2w_budget::placed);
        w->own().backing_budget = backing_budget;
        if (backing_budget) backing_budget->begin_use();
    } else {
        w->own().budget = x2w_budget::committed(device_of(self), resource, D3D12_HEAP_TYPE_DEFAULT);
    }
    *out = w;
    return S_OK;
}

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
        const bool read_only_texture =
            (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) == 0;
        if (lazy_textures_enabled() && read_only_texture) {
            Wrapper* w = wrap(nullptr, resource_kind);
            if (!w) return E_OUTOFMEMORY;
            const DXGI_FORMAT requested_format = desktop_format(desc->Format);
            bool stored = false;
            try {
                std::lock_guard lock(lazy_texture_mutex);
                lazy_textures().emplace(w, LazyTextureRecord{self, texture, requested_format});
                stored = true;
            } catch (const std::bad_alloc&) {
                stored = false;
            }
            if (!stored) {
                AcquireSRWLockExclusive(&registry_lock);
                registry().erase(w);
                ReleaseSRWLockExclusive(&registry_lock);
                _aligned_free(&w->own());
                return E_OUTOFMEMORY;
            }
            w->own().title_memory = address;
            w->own().lazy_slot133_texture = true;
            note_resource(w, desc, 0);
            retire_set_class(w, props, D3D12_HEAP_FLAG_NONE, texture, address, /*allow_title_memory=*/true);
            static LONG lazy_created;
            if (first_calls(lazy_created, 6)) {
                log("x2w-d3d12 Device::slot133(CreatePlacedResourceX?) lazy texture address=%p dim=%d %llux%ux%u mips=%u "
                    "fmt=%d samples=%u flags=0x%x layout=0x%x state=0x%x -> deferred", address, desc->Dimension,
                    desc->Width, desc->Height, desc->DepthOrArraySize, desc->MipLevels, desc->Format,
                    desc->SampleDesc.Count, desc->Flags, desc->Layout, state);
            }
            *out = w;
            return S_OK;
        }
        D3D12_CLEAR_VALUE clear_copy{};
        const auto* clear_value = packed_clear_value(static_cast<const D3D12_CLEAR_VALUE*>(clear),
                                                    desktop_format(desc->Format), texture.Format, clear_copy);
        ID3D12Resource* created = nullptr;
        // On E_OUTOFMEMORY of the default heap, X2W_D3D12_SYSMEM_FALLBACK=1 creates the texture once more in system memory.
        bool in_system_memory = false;
        HRESULT texture_hr = x2w_sysmem::create_with_fallback(device_of(self), props, [&](const D3D12_HEAP_PROPERTIES& heap) {
            HRESULT result = device_of(self)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
                                                                      D3D12_RESOURCE_STATE_COMMON, clear_value,
                                                                      IID_PPV_ARGS(&created));
            if (FAILED(result) && clear_value)
                result = device_of(self)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture,
                                                                  D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&created));
            return result;
        }, in_system_memory);
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
        if (!w) {
            created->Release();
            return E_OUTOFMEMORY;
        }
        w->own().budget = x2w_budget::committed(device_of(self), created, props.Type, in_system_memory);
        if (in_system_memory) log_sysmem_fallback(address, texture, "eager");
        w->own().title_memory = address;
        note_resource(w, desc, 0);
        retire_set_class(w, props, D3D12_HEAP_FLAG_NONE, texture, address, /*allow_title_memory=*/true);
        *out = w;
        return S_OK;
    }
    // The allocation holding the address, and whether every byte from its base to the end of the
    // buffer is committed in it: one region-information call for a fully committed private
    // allocation, otherwise a region walk that stops at the end of the buffer. The GPU cannot fault in a lazily
    // mapped physical page (X2W_XMEM_LAZY_COMMIT), so the buffer's pages are committed first; a lazy startup
    // block (X2W_XMEM_LAZY_STARTUP) is committed whole only right before OpenExistingHeapFromAddress opens it.
    x2w_local::commit_lazy(address, static_cast<std::size_t>(desc->Width));
    x2w_startup::PlacedBufferRange range;
    const bool committed = x2w_startup::measure_placed_buffer(address, desc->Width, range);
    InterlockedIncrement64(&placed_buffer_calls);
    if (range.region_info) InterlockedIncrement64(&placed_buffer_region_info);
    InterlockedAdd64(&placed_buffer_queries, range.queries);
    if (!committed) {
        log("x2w-d3d12 Device::slot133(CreatePlacedResourceX?) address=%p not committed -> E_INVALIDARG", address);
        return E_INVALIDARG;
    }
    void* base = range.base;
    const auto initial_state = state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE ||
        state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS ? static_cast<D3D12_RESOURCE_STATES>(state) : D3D12_RESOURCE_STATE_COMMON;
    if (initial_state != D3D12_RESOURCE_STATE_COMMON && !(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
        return E_INVALIDARG;
    UINT64 offset = static_cast<char*>(address) - static_cast<char*>(base);
    // X2W_D3D12_LINEAR_BUFFERS=1 or copy: shared or copied at submit (linear_buffer_backing.h). Where
    // neither keeps its guarantees, the default backing below is kept and the refusal recorded.
    X2wLinearBufferInfo linear{};
    char linear_note[160] = "";
    if (x2w_startup::LinearBuffers::eligible(*desc, initial_state)) {
        const bool uav = (sanitized(*desc).Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0;
        ID3D12Resource* resource = nullptr;
        bool shared = false;
        x2w_budget::Ticket* linear_heap_budget = nullptr;
        // The mesh XMem lease path takes the default (copied) placements of XMem pages whose console flags
        // are 0 (note_mesh_xmem_write); while it is on, the linear backing defers to it there.
        const bool lease_owns = mesh_xmem_tracking() && range.type == MEM_MAPPED && desc->Flags == D3D12_RESOURCE_FLAG_NONE;
        const HRESULT placed = lease_owns ? x2w_startup::LinearBuffers::defer(uav, &linear)
                                          : linear_buffers().place(device_of(self), address, desc->Width, uav, range, &resource,
                                                                   &shared, &linear_heap_budget, &linear);
        if (SUCCEEDED(placed)) {
            log("x2w-d3d12 Device::slot133(CreatePlacedResourceX?) linear buffer address=%p base=%p type=0x%lx offset=0x%llx "
                "width=0x%llx flags=0x%x use=%s existing-heaps=0x%08lx share=0x%08lx -> 0x%08lx %s gpuva=0x%llx", address, base,
                static_cast<unsigned long>(range.type), offset, desc->Width, desc->Flags, uav ? "gpu-written" : "gpu-read",
                static_cast<unsigned long>(linear.feature_hr), static_cast<unsigned long>(linear.share_hr),
                static_cast<unsigned long>(placed), shared ? "shared" : "copied at submit", resource->GetGPUVirtualAddress());
            // Budgets as for the default backing: a shared buffer is a placed resource on its heap, a copied
            // one a committed default buffer (its staging is counted as upload memory when it is created).
            return publish_placed_buffer(self, address, desc, initial_state, resource, shared, linear_heap_budget, out);
        }
        if (x2w_startup::LinearBuffers::report_refusals())
            std::snprintf(linear_note, sizeof(linear_note),
                          " (kept default backing: linear refusal 0x%08lx, use=%s type=0x%lx existing-heaps=0x%08lx share=0x%08lx)",
                          static_cast<unsigned long>(placed), uav ? "gpu-written" : "gpu-read", static_cast<unsigned long>(range.type),
                          static_cast<unsigned long>(linear.feature_hr), static_cast<unsigned long>(linear.share_hr));
    }
    ID3D12Heap* heap = nullptr;
    x2w_budget::Ticket* backing_budget = nullptr;
    HRESULT hr = S_OK;
    // Desktop only accepts 64 KiB-aligned offsets, a private (VirtualAlloc) allocation, and a buffer
    // that lies inside that one allocation. Attempts that cannot succeed are skipped: each failure
    // throws inside D3D12Core. OpenExistingHeapFromAddress also wraps the entire allocation, so the
    // opt-in X2W_D3D12_CAP_EXISTING_HEAP=1 skips allocations above 256 MiB (or of unknown size) to keep
    // multi-GiB startup blocks out of WorkingSetSize/NonLocal. The cap is opt-in because a capped buffer
    // falls back to a copied placement that CopyBufferRegion (slot 15) refuses as title_backed_source,
    // losing a late guest write; the default keeps the formerly aliased path so contents stay authoritative.
    // A buffer at an aligned nonzero offset inside an eligible lazy startup block leaves [base, address)
    // MEM_RESERVE after the span-only commit_lazy; commit the eligible block before re-measuring so
    // it can alias without touching unaligned or out-of-bounds allocations (and, with the cap, > 256 MiB).
    static const bool cap_existing_heap = x2w_local::env_on("X2W_D3D12_CAP_EXISTING_HEAP");
    if (!range.covered && !range.region_info && range.type == MEM_PRIVATE &&
        initial_state != D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE &&
        x2w_startup::can_commit_lazy_placed_buffer(address, desc->Width,
                                                   cap_existing_heap ? x2w_startup::unplaced_max_span : 0)) {
        x2w_local::commit_lazy_block(address);
        if (x2w_startup::measure_placed_buffer(address, desc->Width, range)) {
            if (range.region_info) InterlockedIncrement64(&placed_buffer_region_info);
            InterlockedAdd64(&placed_buffer_queries, range.queries);
            base = range.base;
            offset = static_cast<char*>(address) - static_cast<char*>(base);
        } else {
            range.covered = false;
        }
    }
    if ((offset & 0xFFFF) || !range.covered || range.type != MEM_PRIVATE ||
        initial_state == D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE ||
        (cap_existing_heap && !x2w_startup::unplaced_existing_heap_allowed(x2w_startup::allocation_bytes(base))))
        hr = E_INVALIDARG;
    if (SUCCEEDED(hr)) {
        std::lock_guard lock(placed_mutex);
        for (const auto& [known, h] : existing_heaps) {
            if (known == base) {
                heap = h.real;
                backing_budget = h.budget;
            }
        }
        if (!heap) {
            x2w_local::commit_lazy_block(address);
            ID3D12Device3* device3 = nullptr;
            hr = device_of(self)->QueryInterface(IID_PPV_ARGS(&device3));
            if (SUCCEEDED(hr)) {
                hr = device3->OpenExistingHeapFromAddress(base, IID_PPV_ARGS(&heap));
                device3->Release();
            }
            if (SUCCEEDED(hr)) {
                backing_budget = x2w_budget::backing_heap(heap, 0);
                existing_heaps.emplace_back(base, ExistingHeap{heap, backing_budget});
            }
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
        "state=0x%x clear=%p riid=%s -> 0x%08lx %s gpuva=0x%llx%s", address, base, offset, desc->Width, desc->Flags, state,
        clear, text, static_cast<unsigned long>(hr), placement, resource ? resource->GetGPUVirtualAddress() : 0ull, linear_note);
    if (FAILED(hr)) return hr;
    if (linear.path == x2w_startup::LinearBuffers::kept_default_path) linear_buffers().kept_default(resource, linear);
    return publish_placed_buffer(self, address, desc, initial_state, resource, aliased, backing_budget, out);
}

#include "d3d12_memory_srv.inl"
#include "startup_stubs/list_copy.inl"

// X2W_D3D12_RELEASE_RETIRED=1 (docs/startup-runtime.md#d3d12-resource-budget): the desktop object of a retire class
// member is released after its final guest Release once (1) the release-time fence of every queue has passed (each queue
// owns a private fence that the Release signals) and (2) no list or submission names it: a list holds a desktop
// reference from a forwarded copy until it is Reset, fails to Close or is destroyed, and each submission holds one until
// its queue passes a fence signalled behind it. Nothing waits; the drain only try-locks and drops references outside
// the lock. What cannot be recorded stays retained for the process, the lifetime without the switch.
constexpr LONG retired_refs = -0x40000000;
SRWLOCK retire_lock = SRWLOCK_INIT;  // a leaf: taken after draw_lock and allocator_lock, nothing is taken under it
struct RetireQueue {
    ID3D12CommandQueue* queue;  // no reference: release() never frees a queue object
    ID3D12Fence* fence;  // null: no marker for this queue, so nothing retires
    UINT64 signaled;
};
struct RetiredResource {
    Wrapper* wrapper;
    std::vector<UINT64> targets;  // per RetireQueue that existed at the Release
};
struct RetireFlight {
    std::size_t queue;
    UINT64 value;
    std::vector<ID3D12Resource*> resources;  // one desktop reference each
};
std::vector<RetireQueue> retire_queues;
std::vector<RetiredResource> retired_resources;
std::vector<RetireFlight> retire_flights;
std::unordered_map<Wrapper*, std::vector<ID3D12Resource*>> retire_list_holds;  // one desktop reference per entry
volatile LONG retire_pending;  // retired_resources and retire_flights, read without the lock
volatile LONG retire_holds;    // entries of retire_list_holds, read without the lock
volatile LONG64 retire_freed, retire_kept;

using LazyRefreshHook = void (*)(void*);
LazyRefreshHook lazy_refresh_hook = nullptr;
void* lazy_refresh_hook_ctx = nullptr;

extern "C" __declspec(dllexport) void x2w_d3d12_set_lazy_refresh_hook(LazyRefreshHook fn, void* ctx)
{
    lazy_refresh_hook = fn;
    lazy_refresh_hook_ctx = ctx;
}

// Thread-safe, idempotent materialization for slot-133 read-only textures created under X2W_D3D12_LAZY_TEXTURES=1.
// Lock order: draw_lock -> lazy_texture_mutex -> note_lock -> retire_lock.
ID3D12Resource* materialize_lazy_texture(Wrapper* w)
{
    if (!w || w->own().kind != resource_kind) return nullptr;
    if (!w->own().lazy_slot133_texture) return static_cast<ID3D12Resource*>(w->own().real);
    if (InterlockedCompareExchange(&w->own().refs, 0, 0) == retired_refs) return nullptr;

    // Counts this publication (the real resource and its descriptor refresh) for the descriptor copies that overlap it.
    struct PublishScope {
        bool on = false;
        void begin() { on = true; InterlockedIncrement(&lazy_publishing); }
        ~PublishScope()
        {
            if (!on) return;
            InterlockedIncrement(&lazy_published_total);
            InterlockedDecrement(&lazy_publishing);
        }
    } publish;
    ID3D12Device* device = nullptr;
    ID3D12Resource* created = nullptr;
    {
        std::lock_guard lock(lazy_texture_mutex);
        if (auto* existing = static_cast<ID3D12Resource*>(w->own().real)) return existing;
        if (InterlockedCompareExchange(&w->own().refs, 0, 0) == retired_refs) return nullptr;

        const auto found = lazy_textures().find(w);
        if (found == lazy_textures().end()) return nullptr;
        const LazyTextureRecord record = found->second;
        device = device_of(record.device);
        if (!device) return nullptr;

        const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
        // Opt-in: a texture a translated draw binds for plain sampling may lose its top mip levels. Every other first
        // use (copy, compute view, map) has no hint and keeps all levels; the reason is counted by name.
        D3D12_RESOURCE_DESC create_desc = record.desc;
        unsigned dropped = 0;
        {
            namespace dm = x2w_startup::drop_mips;
            const dm::Hint hint = dm::current_hint;
            const dm::Decision decision = dm::decide(record.desc, drop_mips_config(), hint.use, hint.view);
            if (decision.levels) {
                dropped = decision.levels;
                create_desc = dm::reduced_desc(record.desc, dropped);
            } else if (drop_mips_config().levels) {
                dm::counters.keep(hint.draw ? decision.reason : dm::Reason::copy_or_other);
            }
        }
        bool in_system_memory = false;
        const HRESULT hr = x2w_sysmem::create_with_fallback(device, props, [&](const D3D12_HEAP_PROPERTIES& heap) {
            return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &create_desc, D3D12_RESOURCE_STATE_COMMON,
                                                   nullptr, IID_PPV_ARGS(&created));
        }, in_system_memory);
        static LONG materialized, failures;
        if (first_calls(materialized, 8) || (FAILED(hr) && first_calls(failures, 8))) {
            log("x2w-d3d12 lazy texture materialize wrapper=%p address=%p dim=%d %llux%ux%u mips=%u fmt=%d -> 0x%08lx",
                static_cast<void*>(w), w->own().title_memory, record.desc.Dimension,
                static_cast<unsigned long long>(record.desc.Width), record.desc.Height,
                record.desc.DepthOrArraySize, record.desc.MipLevels, record.desc.Format,
                static_cast<unsigned long>(hr));
        }
        if (FAILED(hr) || !created) return nullptr;
        if (dropped) {
            namespace dm = x2w_startup::drop_mips;
            w->own().drop_levels = static_cast<std::uint8_t>(dropped);
            dm::counters.dropped.fetch_add(1, std::memory_order_relaxed);
            dm::counters.saved_bytes.fetch_add(dm::saved_estimate(dm::chain_estimate(record.desc), dropped),
                                               std::memory_order_relaxed);
            static LONG shown;
            if (first_calls(shown, 6))
                log("x2w-d3d12 texture drop-mips wrapper=%p %llux%u mips=%u -> %llux%u mips=%u (top %u level(s) left out)",
                    static_cast<void*>(w), static_cast<unsigned long long>(record.desc.Width), record.desc.Height,
                    record.desc.MipLevels, static_cast<unsigned long long>(create_desc.Width), create_desc.Height,
                    create_desc.MipLevels, dropped);
        }

        publish.begin();  // before the real resource is visible to a descriptor copy
        if (release_retired_enabled()) {
            AcquireSRWLockExclusive(&retire_lock);
            if (InterlockedCompareExchange(&w->own().refs, 0, 0) == retired_refs) {
                ReleaseSRWLockExclusive(&retire_lock);
                created->Release();
                return nullptr;
            }
            w->own().real = created;
            ReleaseSRWLockExclusive(&retire_lock);
        } else {
            InterlockedExchangePointer(reinterpret_cast<void* volatile*>(&w->own().real), created);
        }

        x2w_startup::mark_packed_stand_in(created, record.desktop_format);
        x2w_budget::Ticket* const ticket = x2w_budget::committed(device, created, props.Type, in_system_memory);
        if (in_system_memory) log_sysmem_fallback(w->own().title_memory, record.desc, "lazy");
        w->own().budget = ticket;
        if (ticket && InterlockedCompareExchange(&w->own().refs, 0, 0) <= 0) {
            ticket->end_use();
        }
    }

    struct PendingSrvRefresh {
        u64 heap_wrapper;
        std::uint32_t slot_index;
        std::uint32_t record_id;
    };
    std::vector<PendingSrvRefresh> refreshes;
    const u64 target_wrapper = reinterpret_cast<u64>(w);
    try {
        AcquireSRWLockShared(&note_lock);
        for (const HeapInfo& h : heap_infos()) {
            if (!h.shadow || h.type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) continue;
            for (std::uint32_t i = 0; i < h.count; ++i) {
                const std::uint32_t id = h.shadow[i];
                if (!id || id - 1 >= view_records().size() || !view_materialized()[id - 1]) continue;
                const ViewRecord& vr = view_records()[id - 1];
                if (vr.slot == 18 && vr.resource == target_wrapper) {
                    refreshes.push_back({h.wrapper, i, id});
                }
            }
        }
        ReleaseSRWLockShared(&note_lock);
    } catch (const std::bad_alloc&) {
        ReleaseSRWLockShared(&note_lock);
    }

    if (lazy_refresh_hook) lazy_refresh_hook(lazy_refresh_hook_ctx);

    if (!refreshes.empty()) {
        const D3D12_RESOURCE_DESC created_desc = created->GetDesc();
        const auto view_mode = view_defaults_mode();
        // Hold note_lock shared across revalidation and CreateShaderResourceView so concurrent note_view /
        // note_copy / retire_drop_heap_views (all exclusive) cannot overwrite a slot or retire a heap mid-write.
        AcquireSRWLockShared(&note_lock);
        for (const auto& item : refreshes) {
            const HeapInfo* target_heap = nullptr;
            for (const HeapInfo& h : heap_infos()) {
                if (h.wrapper == item.heap_wrapper) {
                    target_heap = &h;
                    break;
                }
            }
            if (!target_heap || !target_heap->shadow || !target_heap->desktop_cpu ||
                target_heap->type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ||
                item.slot_index >= target_heap->count ||
                target_heap->shadow[item.slot_index] != item.record_id)
                continue;
            if (!item.record_id || item.record_id - 1 >= view_records().size() ||
                !view_materialized()[item.record_id - 1])
                continue;
            const ViewRecord& vr = view_records()[item.record_id - 1];
            if (vr.slot != 18 || vr.resource != target_wrapper) continue;
            const D3D12_CPU_DESCRIPTOR_HANDLE dest{
                static_cast<SIZE_T>(target_heap->desktop_cpu + u64{item.slot_index} * target_heap->increment)};
            create_lazy_srv(device, created, created_desc, view_mode, vr, dest, w->own().drop_levels);
        }
        ReleaseSRWLockShared(&note_lock);
    }
    return created;
}

bool retire_unmaterialized_lazy_texture(Wrapper* w)
{
    std::lock_guard lock(lazy_texture_mutex);
    if (w->own().real) return false;
    const bool retired_mode = release_retired_enabled();
    const bool can_drop = (retired_mode && w->own().retire_when_idle)
        ? InterlockedCompareExchange(&w->own().retire_view_refs, 0, 0) == 0
        : !w->own().lazy_view_created;
    if (!can_drop) return false;
    if (w->own().title_memory) {
        x2w_startup::ExclusiveNoteLock guard(note_lock);
        resource_address_index().erase(reinterpret_cast<u64>(w->own().title_memory), w);
    }
    lazy_textures().erase(w);
    if (retired_mode) {
        InterlockedCompareExchange(&w->own().refs, retired_refs, 0);
        return true;
    }
    return false;
}

void register_retire_queue(Wrapper* queue)
{
    if (!release_retired_enabled() || !queue || !queue->own().real) return;
    auto* real = static_cast<ID3D12CommandQueue*>(queue->own().real);
    ID3D12Fence* fence = nullptr;
    ID3D12Device* device = nullptr;
    if (SUCCEEDED(real->GetDevice(IID_PPV_ARGS(&device)))) {
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) fence = nullptr;
        device->Release();
    }
    AcquireSRWLockExclusive(&retire_lock);
    try {
        retire_queues.push_back({real, fence, 0});
    } catch (const std::bad_alloc&) {
        if (fence) fence->Release();
    }
    ReleaseSRWLockExclusive(&retire_lock);
}

// A use the holds do not cover (a view, residency, presentation) takes the resource out of the class for good.
void retire_leave_class(const void* object, RetireLeft why)
{
    if (!release_retired_enabled()) return;
    if (Wrapper* w = as_wrapper(object)) {
        if (w->own().retire_when_idle) w->own().retire_left = static_cast<std::uint8_t>(why);  // the first use counts
        w->own().retire_when_idle = false;
    }
}

void retire_add_view_ref(const void* object)
{
    if (!release_retired_enabled()) return;
    if (Wrapper* w = as_wrapper(object); w && w->own().kind == resource_kind)
        InterlockedIncrement(&w->own().retire_view_refs);
}

void retire_release_view_ref(const void* object)
{
    if (!release_retired_enabled()) return;
    if (Wrapper* w = as_wrapper(object); w && w->own().kind == resource_kind)
        InterlockedDecrement(&w->own().retire_view_refs);
}

void retire_drop_heap_views(Wrapper* heap)
{
    if (!release_retired_enabled() || !heap) return;
    AcquireSRWLockExclusive(&note_lock);
    for (HeapInfo& h : heap_infos()) {
        if (h.wrapper != reinterpret_cast<u64>(heap) || !h.shadow) continue;
        for (std::uint32_t i = 0; i < h.count; ++i) {
            const std::uint32_t id = h.shadow[i];
            if (!id) continue;
            h.shadow[i] = 0;
            // Every tracked view slot loses the ref it held: slots 18-21, slot 19 for the resource and its counter.
            if (id - 1 < view_records().size() && view_materialized()[id - 1])
                view_refs_release(view_records()[id - 1].slot, view_records()[id - 1].resource,
                                  view_records()[id - 1].counter);
        }
    }
    ReleaseSRWLockExclusive(&note_lock);
}

bool retire_address_has_live_user(u64 address, const Wrapper* skip)
{
    bool other_live = false;
    AcquireSRWLockShared(&note_lock);
    resource_address_index().for_each_reaching(address, 1, [&](const Wrapper* wrapper) {
        if (wrapper == skip || reinterpret_cast<u64>(wrapper->own().title_memory) != address) return false;
        if (InterlockedCompareExchange(const_cast<LONG*>(&wrapper->own().refs), 0, 0) > 0 ||
            InterlockedCompareExchange(const_cast<LONG*>(&wrapper->own().retire_view_refs), 0, 0) > 0) {
            other_live = true;
            return true;
        }
        return false;
    });
    ReleaseSRWLockShared(&note_lock);
    if (other_live) return true;
    std::lock_guard lock(tiled_srv_mutex);
    return tiled_srv_addresses.count(address) != 0;
}

void retire_note_title_release(Wrapper* w)
{
    const u64 address = reinterpret_cast<u64>(w->own().title_memory);
    if (!address) return;
    const bool shared_addr = retire_address_has_live_user(address, w);
    const bool no_views = InterlockedCompareExchange(&w->own().retire_view_refs, 0, 0) == 0;
    AcquireSRWLockExclusive(&written_lock);
    if (const auto found = written_textures().find(address); found != written_textures().end()) {
        if (!shared_addr && no_views) {
            written_textures().erase(found);
        } else {
            w->own().retire_has_written_gen = true;
            w->own().retire_written_gen = found->second.generation;
        }
    }
    ReleaseSRWLockExclusive(&written_lock);
}

// After a copy, texture copy or resolve was forwarded to the desktop list: the list keeps a desktop reference to the
// resource until its recording is dropped. Only wrappers are members of the class, so a raw desktop pointer is never
// held. Assumption, not proven: the guest has no raw pointer to the desktop object of a class member.
void retire_hold(Wrapper* list, Wrapper* resource)
{
    if (!resource || !resource->own().retire_when_idle) return;
    bool held = false;
    AcquireSRWLockExclusive(&retire_lock);
    // own().real is cleared by drain_retired_resources under this lock, so the read and the AddRef happen under it
    // (retire_lock is a leaf: nothing else is taken inside).
    auto* real = static_cast<ID3D12Resource*>(resource->own().real);
    if (real) {
        try {
            auto& held_by_list = retire_list_holds[list];
            if (std::find(held_by_list.begin(), held_by_list.end(), real) != held_by_list.end()) {
                held = true;
            } else {
                held_by_list.reserve(held_by_list.size() + 1);
                real->AddRef();
                held_by_list.push_back(real);
                InterlockedIncrement(&retire_holds);
                held = true;
            }
        } catch (const std::bad_alloc&) {
        }
    }
    if (!held) {
        // The use cannot be recorded: leave the class, and mark it so the drain keeps the desktop object for the
        // process (as it keeps a submission it could not fence) rather than dropping one a list still names.
        resource->own().retire_when_idle = false;
        resource->own().retire_left = static_cast<std::uint8_t>(RetireLeft::hold_missed);
        resource->own().retire_hold_missed = true;
    }
    ReleaseSRWLockExclusive(&retire_lock);
}

void retire_discard_holds(Wrapper* list)
{
    if (!InterlockedCompareExchange(&retire_holds, 0, 0)) return;
    std::vector<ID3D12Resource*> dropped;
    AcquireSRWLockExclusive(&retire_lock);
    if (const auto found = retire_list_holds.find(list); found != retire_list_holds.end()) {
        dropped = std::move(found->second);
        retire_list_holds.erase(found);
        InterlockedAdd(&retire_holds, -static_cast<LONG>(dropped.size()));
    }
    ReleaseSRWLockExclusive(&retire_lock);
    for (ID3D12Resource* resource : dropped) resource->Release();
}

// Right after ExecuteCommandLists: the lists' held resources stay referenced until the queue passes a fence signalled
// behind the submission. If that fence cannot be recorded or signalled the references are kept for the process.
void retire_note_submission(ID3D12CommandQueue* queue, const std::vector<Wrapper*>& lists)
{
    if (!InterlockedCompareExchange(&retire_holds, 0, 0)) return;
    std::vector<ID3D12Resource*> named;
    AcquireSRWLockExclusive(&retire_lock);
    bool room = true;
    try {
        for (Wrapper* list : lists) {
            const auto found = retire_list_holds.find(list);
            if (found == retire_list_holds.end()) continue;
            for (ID3D12Resource* resource : found->second)
                if (std::find(named.begin(), named.end(), resource) == named.end()) named.push_back(resource);
        }
        retire_flights.reserve(retire_flights.size() + 1);
    } catch (const std::bad_alloc&) {
        room = false;
    }
    bool recorded = false;
    if (room && !named.empty()) {
        std::size_t index = 0;
        while (index < retire_queues.size() && retire_queues[index].queue != queue) ++index;
        if (index < retire_queues.size() && retire_queues[index].fence) {
            RetireQueue& q = retire_queues[index];
            if (SUCCEEDED(queue->Signal(q.fence, q.signaled + 1))) {
                for (ID3D12Resource* resource : named) resource->AddRef();
                ++q.signaled;
                retire_flights.push_back({index, q.signaled, std::move(named)});
                InterlockedIncrement(&retire_pending);
                recorded = true;
            }
        }
    }
    if (!recorded && (!room || !named.empty())) {
        // Keep everything the lists name for the process; no allocation is needed to do so.
        for (Wrapper* list : lists)
            if (const auto found = retire_list_holds.find(list); found != retire_list_holds.end())
                for (ID3D12Resource* resource : found->second) resource->AddRef();
        InterlockedIncrement64(&retire_kept);
    }
    ReleaseSRWLockExclusive(&retire_lock);
    if (!recorded && (!room || !named.empty())) {
        static LONG reported;
        if (first_calls(reported, 4)) log("x2w-d3d12 retire: a submission could not be fenced; its resources stay retained");
    }
}

// Called by the final Release of a resource of the class.
void retire_resource(Wrapper* w)
{
    if (w->own().kind != resource_kind || (!w->own().real && !w->own().lazy_slot133_texture)) return;
    RetiredResource item{w, {}};
    bool queued = false;
    AcquireSRWLockExclusive(&retire_lock);
    try {
        // Without a registered queue nothing orders a marker: keep the resource. Every queue needs its fence.
        queued = !retire_queues.empty();
        for (const RetireQueue& q : retire_queues) queued = queued && q.fence;
        if (queued) {
            item.targets.reserve(retire_queues.size());
            retired_resources.reserve(retired_resources.size() + 1);
            for (RetireQueue& q : retire_queues) {
                if (FAILED(q.queue->Signal(q.fence, q.signaled + 1))) {
                    queued = false;
                    break;
                }
                item.targets.push_back(++q.signaled);
            }
        }
        if (queued) {
            retired_resources.push_back(std::move(item));
            InterlockedIncrement(&retire_pending);
        }
    } catch (const std::bad_alloc&) {
        queued = false;
    }
    ReleaseSRWLockExclusive(&retire_lock);
    if (!queued && w->own().real) InterlockedIncrement64(&retire_kept);  // stays retained, as without the switch
}

// True when the release-time fence of every queue has passed (retire_lock held).
bool retire_fences_reached(const RetiredResource& item)
{
    bool reached = item.targets.size() <= retire_queues.size();
    for (std::size_t q = 0; q < item.targets.size() && reached; ++q)
        reached = retire_queues[q].fence->GetCompletedValue() >= item.targets[q];
    return reached;
}

// Never blocks: a contended lock or an unfinished queue leaves the work for the next call.
void drain_retired_resources()
{
    if (!InterlockedCompareExchange(&retire_pending, 0, 0)) return;
    constexpr std::size_t batch = 256;
    struct RetiredDrop {
        Wrapper* wrapper;
        ID3D12Resource* resource;
    };
    std::vector<RetiredDrop> dropped;
    std::vector<std::vector<ID3D12Resource*>> dropped_flights;
    try {
        dropped.reserve(batch);
        dropped_flights.reserve(batch);
    } catch (const std::bad_alloc&) {
        return;
    }
    if (!TryAcquireSRWLockExclusive(&retire_lock)) return;
    for (std::size_t i = 0; i < retired_resources.size() && dropped.size() < batch;) {
        RetiredResource& item = retired_resources[i];
        Wrapper* w = item.wrapper;
        if (w->own().retire_hold_missed) {
            // retire_hold could not record a list's use of this resource: keep its desktop object for the process,
            // like a submission that could not be fenced, and drop only the retire bookkeeping.
            InterlockedIncrement64(&retire_kept);
            if (i + 1 != retired_resources.size()) item = std::move(retired_resources.back());
            retired_resources.pop_back();
            InterlockedDecrement(&retire_pending);
            continue;
        }
        if (InterlockedCompareExchange(&w->own().retire_view_refs, 0, 0) > 0) {
            ++i;
            continue;
        }
        if (!retire_fences_reached(item)) {
            ++i;
            continue;
        }
        // Claim the reference count so that a late AddRef cannot bring the wrapper back while its object goes.
        if (InterlockedCompareExchange(&w->own().refs, retired_refs, 0) == 0) {
            dropped.push_back({w, static_cast<ID3D12Resource*>(w->own().real)});
            w->own().real = nullptr;
        } else {
            InterlockedIncrement64(&retire_kept);  // referenced again after its final Release
        }
        if (i + 1 != retired_resources.size()) item = std::move(retired_resources.back());
        retired_resources.pop_back();
        InterlockedDecrement(&retire_pending);
    }
    for (std::size_t i = 0; i < retire_flights.size() && dropped_flights.size() < batch;) {
        RetireFlight& flight = retire_flights[i];
        if (retire_queues[flight.queue].fence->GetCompletedValue() < flight.value) {
            ++i;
            continue;
        }
        dropped_flights.push_back(std::move(flight.resources));
        if (i + 1 != retire_flights.size()) flight = std::move(retire_flights.back());
        retire_flights.pop_back();
        InterlockedDecrement(&retire_pending);
    }
    ReleaseSRWLockExclusive(&retire_lock);
    for (const RetiredDrop& drop : dropped) {
        if (drop.wrapper->own().title_memory) {
            const u64 address = reinterpret_cast<u64>(drop.wrapper->own().title_memory);
            {
                x2w_startup::ExclusiveNoteLock guard(note_lock);
                resource_address_index().erase(address, drop.wrapper);
                x2w_startup::note_placement_change();
            }
            if (drop.wrapper->own().retire_has_written_gen && !retire_address_has_live_user(address, drop.wrapper)) {
                AcquireSRWLockExclusive(&written_lock);
                if (const auto found = written_textures().find(address);
                    found != written_textures().end() && found->second.generation == drop.wrapper->own().retire_written_gen)
                    written_textures().erase(found);
                ReleaseSRWLockExclusive(&written_lock);
            }
        }
        if (drop.wrapper->own().lazy_slot133_texture) {
            std::lock_guard lock(lazy_texture_mutex);
            lazy_textures().erase(drop.wrapper);
        }
        if (drop.resource) {
            AcquireSRWLockExclusive(&draw_lock);
            texture_uploads().erase(drop.resource);
            ReleaseSRWLockExclusive(&draw_lock);
            drop.resource->Release();
            const LONG64 total = InterlockedIncrement64(&retire_freed);
            if (total <= 8 || total % 1000 == 0) log("x2w-d3d12 retire: %lld desktop references dropped", total);
        }
    }
    for (const auto& resources : dropped_flights) {
        for (ID3D12Resource* resource : resources) {
            resource->Release();
            const LONG64 total = InterlockedIncrement64(&retire_freed);
            if (total <= 8 || total % 1000 == 0) log("x2w-d3d12 retire: %lld desktop references dropped", total);
        }
    }
}

// X2W_D3D12_RETIRE_REASONS=1 (with X2W_D3D12_RELEASE_RETIRED=1 and X2W_D3D12_BUDGET=1): why resources that the guest
// released for good still hold their desktop object. Every final guest Release of a resource with a desktop object is
// remembered here; the heartbeat classifies the ones whose object is still alive, once each, by the first cause that
// applies, in the order of the enum below. Counts and bytes come from the budget ticket. Without the switches nothing
// is recorded and no line is built.
bool retire_reasons_enabled()
{
    static const bool on = release_retired_enabled() && x2w_budget::enabled() &&
        (legacy_retire_view_stats() || x2w_local::env_on("X2W_D3D12_RETIRE_REASONS"));
    return on;
}

struct ReleasedResource {
    ID3D12Resource* real;  // identity once the wrapper let go of it; dereferenced only while a hold keeps it alive
    LONG frame;            // frame_number at the final Release
};
SRWLOCK retire_diag_lock = SRWLOCK_INIT;  // a leaf, never held together with another lock
std::unordered_map<Wrapper*, ReleasedResource> retire_diag_released;

void retire_diag_note_release(Wrapper* w)
{
    if (!w->own().real) return;
    AcquireSRWLockExclusive(&retire_diag_lock);
    try {
        retire_diag_released[w] = {static_cast<ID3D12Resource*>(w->own().real), frame_number};
    } catch (const std::bad_alloc&) {  // a diagnostic only: the resource is then missing from the counts
    }
    ReleaseSRWLockExclusive(&retire_diag_lock);
}

enum class Held { view_refs, list_hold, flight, waiting_fence, drain_pending, unqueued, left_view, left_view_copy,
                  left_residency, left_present, left_hold_missed, never_slot, never_buffer, never_dimension, never_msaa,
                  never_rt_ds_uav, never_heap_type, never_heap_flags, never_title_memory, never_aliased, never_reserved,
                  other, count };
constexpr const char* held_names[] = {"view-refs", "list-hold", "flight", "queued-waiting-fence", "drain-pending",
    "unqueued", "left-class:view", "left-class:view-copy", "left-class:residency", "left-class:present",
    "left-class:hold-missed", "never-class:slot", "never-class:buffer", "never-class:dimension", "never-class:msaa",
    "never-class:rt-ds-uav", "never-class:heap-type", "never-class:heap-flags", "never-class:title-memory",
    "never-class:aliased", "never-class:reserved", "other"};
static_assert(std::size(held_names) == static_cast<std::size_t>(Held::count));
constexpr Held held_left[] = {Held::other, Held::left_view, Held::left_view_copy, Held::left_residency, Held::left_present,
                              Held::left_hold_missed};
constexpr Held held_never[] = {Held::never_slot, Held::other, Held::never_buffer, Held::never_dimension, Held::never_msaa,
                               Held::never_rt_ds_uav, Held::never_heap_type, Held::never_heap_flags,
                               Held::never_title_memory, Held::never_aliased, Held::never_reserved};
static_assert(std::size(held_left) == static_cast<std::size_t>(RetireLeft::count));
static_assert(std::size(held_never) == static_cast<std::size_t>(RetireNever::count));

struct RetireReasonReport {
    u64 count[static_cast<int>(Held::count)];
    u64 bytes[static_cast<int>(Held::count)];
    LONG oldest[static_cast<int>(Held::count)];  // frames since the final Release of the oldest entry
    u64 visible_count, visible_bytes;            // view-refs entries with a descriptor in a shader-visible heap
};

void retire_reasons_collect(RetireReasonReport& report)
{
    // 1: which resources a tracked slot-18 descriptor names, and whether one of those lies in a shader-visible heap.
    std::unordered_map<const Wrapper*, bool> viewed;
    AcquireSRWLockShared(&note_lock);
    try {
        for (const HeapInfo& h : heap_infos()) {
            if (!h.shadow) continue;
            for (std::uint32_t i = 0; i < h.count; ++i) {
                const std::uint32_t id = h.shadow[i];
                if (!id || id - 1 >= view_records().size() || view_records()[id - 1].slot != 18 || !view_materialized()[id - 1])
                    continue;
                bool& visible = viewed[reinterpret_cast<const Wrapper*>(view_records()[id - 1].resource)];
                visible = visible || h.desktop_gpu != 0;
            }
        }
    } catch (const std::bad_alloc&) {
    }
    ReleaseSRWLockShared(&note_lock);

    // 2: the remembered final Releases.
    std::vector<std::pair<Wrapper*, ReleasedResource>> entries;
    AcquireSRWLockShared(&retire_diag_lock);
    try {
        entries.assign(retire_diag_released.begin(), retire_diag_released.end());
    } catch (const std::bad_alloc&) {
    }
    ReleaseSRWLockShared(&retire_diag_lock);

    // 3: the retire bookkeeping. own().real is cleared by the drain under this lock, so a non-null pointer read here
    // names a desktop object that stays alive for as long as the lock is held.
    std::vector<Wrapper*> gone;
    AcquireSRWLockShared(&retire_lock);
    try {
        std::unordered_map<const Wrapper*, const RetiredResource*> queued;
        std::unordered_set<const ID3D12Resource*> listed, flying;
        for (const RetiredResource& item : retired_resources) queued.emplace(item.wrapper, &item);
        for (const auto& list : retire_list_holds) listed.insert(list.second.begin(), list.second.end());
        for (const RetireFlight& flight : retire_flights) flying.insert(flight.resources.begin(), flight.resources.end());
        const LONG now = frame_number;
        const auto add = [&](Held why, const x2w_budget::Ticket* ticket, LONG frame) {
            const int i = static_cast<int>(why);
            ++report.count[i];
            report.bytes[i] += ticket ? static_cast<u64>(ticket->bytes()) : 0;
            report.oldest[i] = (std::max)(report.oldest[i], now - frame);
        };
        for (const auto& [w, released] : entries) {
            if (InterlockedCompareExchange(&w->own().refs, 0, 0) > 0) continue;  // referenced again: not released for good
            auto* real = static_cast<ID3D12Resource*>(w->own().real);
            if (!real) {
                // The drain let go of it; only a list hold or a submission can still keep the desktop object.
                const bool by_list = listed.count(released.real) != 0;
                if (!by_list && !flying.count(released.real)) {
                    gone.push_back(w);
                    continue;
                }
                x2w_budget::Ticket* ticket = x2w_budget::ticket_of(released.real);
                add(by_list ? Held::list_hold : Held::flight, ticket, released.frame);
                if (ticket) ticket->Release();
                continue;
            }
            const x2w_budget::Ticket* ticket = w->own().budget;
            Held why = Held::other;
            const auto found = queued.find(w);
            if (!w->own().retire_when_idle) {
                if (w->own().retire_left && w->own().retire_left < static_cast<std::uint8_t>(RetireLeft::count))
                    why = held_left[w->own().retire_left];
                else if (w->own().retire_never < static_cast<std::uint8_t>(RetireNever::count))
                    why = held_never[w->own().retire_never];
            } else if (found == queued.end()) {
                why = Held::unqueued;
            } else if (InterlockedCompareExchange(&w->own().retire_view_refs, 0, 0) > 0) {
                why = Held::view_refs;
                if (const auto view = viewed.find(w); view != viewed.end() && view->second) {
                    ++report.visible_count;
                    report.visible_bytes += ticket ? static_cast<u64>(ticket->bytes()) : 0;
                }
            } else if (listed.count(real)) {
                why = Held::list_hold;
            } else if (flying.count(real)) {
                why = Held::flight;
            } else {
                const RetiredResource& item = *found->second;
                bool reached = item.targets.size() <= retire_queues.size();
                for (std::size_t q = 0; q < item.targets.size() && reached; ++q)
                    reached = retire_queues[q].fence->GetCompletedValue() >= item.targets[q];
                why = reached ? Held::drain_pending : Held::waiting_fence;
            }
            add(why, ticket, released.frame);
        }
    } catch (const std::bad_alloc&) {
    }
    ReleaseSRWLockShared(&retire_lock);

    // 4: forget entries whose desktop object is gone.
    if (!gone.empty()) {
        AcquireSRWLockExclusive(&retire_diag_lock);
        for (Wrapper* w : gone)
            if (!w->own().real) retire_diag_released.erase(w);
        ReleaseSRWLockExclusive(&retire_diag_lock);
    }
}

// The heartbeat line, or an empty string when the diagnostics are not switched on. unattributed: what the budget still
// counts as retained, less what this classification found (backing heaps, upload and readback objects, anything it cannot see).
std::size_t retire_reasons_line(char* out, std::size_t capacity)
{
    if (!out || !capacity) return 0;
    out[0] = 0;
    if (!retire_reasons_enabled()) return 0;
    RetireReasonReport report{};
    retire_reasons_collect(report);
    const x2w_budget::Snapshot budget = x2w_budget::snapshot();
    std::size_t used = 0;
    const auto append = [&](const char* format, auto... args) {
        if (used + 1 >= capacity) return;
        const int n = std::snprintf(out + used, capacity - used, format, args...);
        if (n > 0) used = (std::min)(used + static_cast<std::size_t>(n), capacity - 1);
    };
    append("x2w-retire held:");
    u64 total_count = 0, total_bytes = 0;
    for (int i = 0; i < static_cast<int>(Held::count); ++i) {
        if (!report.count[i]) continue;
        total_count += report.count[i];
        total_bytes += report.bytes[i];
        append(" %s=%llu/%llu(oldest %ldf)", held_names[i], report.count[i], report.bytes[i], report.oldest[i]);
    }
    if (!total_count) append(" none");
    if (report.visible_count) append(" view-refs-in-visible-heap=%llu/%llu", report.visible_count, report.visible_bytes);
    u64 retained_count = 0, retained_bytes = 0;
    for (const auto& retained : budget.retained) {
        retained_count += retained.objects;
        retained_bytes += retained.bytes;
    }
    append(" total=%llu/%llu unattributed=%lld/%lld", total_count, total_bytes,
           static_cast<long long>(retained_count) - static_cast<long long>(total_count),
           static_cast<long long>(retained_bytes) - static_cast<long long>(total_bytes));
    return used;
}

// Fixture export (startup_resource_retire_reasons): the line the heartbeat logs, now. Returns its length; 0 and an
// empty string without the switches.
extern "C" __declspec(dllexport) std::size_t x2w_d3d12_retire_reasons_line(char* out, std::size_t capacity)
{
    return retire_reasons_line(out, capacity);
}

// The census heartbeat line, or an empty string when X2W_D3D12_VIEW_CENSUS is not set.
std::size_t view_census_line(char* out, std::size_t capacity)
{
    if (!out || !capacity) return 0;
    out[0] = 0;
    if (!view_census_enabled()) return 0;
    AcquireSRWLockShared(&note_lock);
    const std::size_t used = view_census_format(out, capacity, "distinct by slot");
    ReleaseSRWLockShared(&note_lock);
    return used;
}

// Fixture export (startup_view_census): the line the heartbeat logs, now.
extern "C" __declspec(dllexport) std::size_t x2w_d3d12_view_census_line(char* out, std::size_t capacity)
{
    return view_census_line(out, capacity);
}

// Fixture only (startup_view_alloc): see view_allocation_gate.
extern "C" __declspec(dllexport) void x2w_d3d12_fail_view_allocations(int site, LONG count)
{
    if (site >= 0 && site < view_allocation_sites) InterlockedExchange(&view_allocations_to_fail[site], count);
}

// Fixture probe (startup_view_alloc): the sizes of the three view containers, the id + 1 that the tracked slot of a
// title descriptor handle names (0: none), and whether a resource is still in the retire class.
struct ViewStoreProbe {
    unsigned long long records, materialized, ids, records_capacity, materialized_capacity;
    unsigned shadow, view_refs, retire_when_idle;
};
extern "C" __declspec(dllexport) void x2w_d3d12_view_probe(unsigned long long title_handle, void* resource, ViewStoreProbe* out)
{
    *out = {};
    out->shadow = shadow_id(desktop_cpu_handle(title_handle));
    AcquireSRWLockShared(&note_lock);
    out->records = view_records().size();
    out->materialized = view_materialized().size();
    out->ids = view_ids().size();
    out->records_capacity = view_records().capacity();
    out->materialized_capacity = view_materialized().capacity();
    ReleaseSRWLockShared(&note_lock);
    if (Wrapper* r = as_wrapper(resource); r && r->own().kind == resource_kind) {
        out->view_refs = static_cast<unsigned>(InterlockedCompareExchange(&r->own().retire_view_refs, 0, 0));
        out->retire_when_idle = r->own().retire_when_idle;
    }
}

// Fixture-only: the next `count` wrapper allocations fail as an exhausted heap would (#2653).
extern "C" __declspec(dllexport) void x2w_d3d12_fail_wraps(LONG count)
{
    InterlockedExchange(&wraps_to_fail, count);
}

// Fixture probe (startup_resource_retire): what still names a resource of the class by its raw desktop pointer, how many
// resources the open recording of a list moved out of COMMON, how many it holds, and how many tracked descriptor slots
// hold the resource (retire_view_refs, slots 18-21). Counts entries only.
struct RetireProbe {
    unsigned cache_entries;
    unsigned list_states;
    unsigned list_holds;
    unsigned view_refs;
};
extern "C" __declspec(dllexport) BOOL x2w_d3d12_retire_probe(void* resource, void* list, RetireProbe* out)
{
    if (!out) return FALSE;
    *out = {};
    if (!release_retired_enabled()) return FALSE;  // like x2w_d3d12_budget_snapshot: nothing to report without the switch
    if (Wrapper* r = as_wrapper(resource); r && r->own().kind == resource_kind) {
        out->view_refs = static_cast<unsigned>(InterlockedCompareExchange(&r->own().retire_view_refs, 0, 0));
        if (auto* real = static_cast<ID3D12Resource*>(r->own().real)) {
            AcquireSRWLockExclusive(&draw_lock);
            out->cache_entries += static_cast<unsigned>(texture_uploads().count(real));
            ReleaseSRWLockExclusive(&draw_lock);
            AcquireSRWLockShared(&target_lock);
            for (const auto& view : view_targets()) out->cache_entries += view.second.resource == real;
            ReleaseSRWLockShared(&target_lock);
            AcquireSRWLockShared(&pending_writes_lock);
            for (const auto& writes : pending_gpu_writes())
                out->cache_entries += static_cast<unsigned>(std::count(writes.second.begin(), writes.second.end(), real));
            ReleaseSRWLockShared(&pending_writes_lock);
        }
    }
    if (Wrapper* l = as_wrapper(list); l && l->own().kind == list_kind) {
        if (l->own().pass) out->list_states = static_cast<unsigned>(pass_of(l).states.size());
        AcquireSRWLockShared(&retire_lock);
        if (const auto found = retire_list_holds.find(l); found != retire_list_holds.end())
            out->list_holds = static_cast<unsigned>(found->second.size());
        ReleaseSRWLockShared(&retire_lock);
    }
    return TRUE;
}

extern "C" __declspec(dllexport) unsigned x2w_d3d12_lazy_texture_count()
{
    std::lock_guard lock(lazy_texture_mutex);
    return static_cast<unsigned>(lazy_textures().size());
}

// Fixture probe (startup_teardown_unload, #2572) for the four caches that keep desktop objects for a
// device, one bit each: the indexed-validation pipeline (d3d12_draw.inl, bit 1) and the rectangle-list
// argument converter (rect_indirect_args.h, bit 2), the placed page aliases of the translated indirect
// address route (bit 4) and the draw-dump readbacks (bit 8). Warms the selected caches the way their
// first real call would -- the validation pipeline directly, the converter through one record-only
// conversion, an alias entry on a placed buffer of this device, a recorded readback -- and answers how
// many objects they now hold for `device`: the validation entry's device, root signature and pipeline
// (3), the converter entry's root signature and pipeline (2), the alias entry's placed buffer (1), the
// readback entry's buffer and fence (2). The caller can warm one cache at a time to see each one release
// what it owns when the module is unloaded.
//
// `placed_heap` is how bit 4 hands the heap it creates over to the caller. The alias entry keeps that heap
// as a key only -- it owns no reference, as the route's heap belongs to the device's allocation -- so the
// one reference this probe holds is the caller's from the moment the entry is entered, and the caller must
// release it after the cache released the placed buffer (a heap must outlive the resources placed in it).
// Bit 4 is not warmed when `placed_heap` is null: an entry whose backing heap nobody owns is the leak this
// handover exists to rule out. Every other bit leaves `*placed_heap` null.
extern "C" __declspec(dllexport) UINT x2w_d3d12_cached_objects_probe(ID3D12Device* device, UINT caches,
                                                                    ID3D12Heap** placed_heap)
{
    if (placed_heap) *placed_heap = nullptr;
    if (!device || !caches || (caches & ~15u)) return 0;
    UINT owned = 0;
    if (caches & 1u) {
        const auto& validation = indexed_validation_pipeline(device);
        if (validation.device) ++owned;
        if (validation.root) ++owned;
        if (validation.pipeline) ++owned;
    }
    if (caches & 2u) {
        x2w_startup::ComPtr<ID3D12CommandAllocator> allocator;
        x2w_startup::ComPtr<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                            IID_PPV_ARGS(&list))))
            return owned;
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = 256;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        // convert() copies its 16 source words; an uninitialised buffer in the state it requires is enough.
        x2w_startup::ComPtr<ID3D12Resource> source;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                   nullptr, IID_PPV_ARGS(&source))))
            return owned;
        const auto scratch = x2w_startup::RectIndirectArguments::make_scratch(device);
        if (scratch && x2w_startup::rect_indirect_arguments().convert(list.Get(), source.Get(), 0, scratch.Get()))
            owned += 2;  // the converter's root signature and pipeline for this device
    }
    if (!(caches & 12u)) return owned;

    // The two caches below hold desktop objects, as their real entries do: draw_dump_after() creates the
    // readback buffer on the desktop device, and execute() creates the fence on the desktop device it
    // reads from the desktop queue. Two shapes of device reach this probe. A title gets a wrapper from
    // D3D12XboxCreateDevice (attach_device wraps the created device), and the desktop device is then the
    // one behind it. A fixture creates its device without a parameter block, so that entry returns a
    // plain D3D12CreateDevice() result: an unwrapped desktop device. The handle itself is the desktop
    // device then. removal_device is not a fallback: attach_device is where it is written, so on the
    // unwrapped path it is null or names another device.
    Wrapper* wrapper = as_wrapper(device);
    ID3D12Device* desktop = wrapper ? device_of(wrapper) : device;

    // Diagnostics for the fixture, off unless X2W_CACHE_PROBE_DEBUG is set: one line per step, so a probe
    // that answers fewer objects than expected names the call that refused. Temporary, together with the
    // two caches they measure.
    char probe_debug[2]{};
    const bool trace = GetEnvironmentVariableA("X2W_CACHE_PROBE_DEBUG", probe_debug, sizeof(probe_debug)) != 0;
    if (trace)
        std::fprintf(stderr, "x2w-cache-probe device=%p wrapper=%p desktop=%p caches=0x%x\n",
                     static_cast<void*>(device), static_cast<void*>(wrapper), static_cast<void*>(desktop), caches);

    // Bit 4 needs an owner for the heap it creates, and that owner is the caller: with no out pointer
    // there is nowhere to put the reference, so the alias is not warmed at all rather than entered on a
    // heap nobody owns for the rest of the process.
    if ((caches & 4u) && !placed_heap && trace)
        std::fputs("x2w-cache-probe placed: no heap out pointer, bit 4 not warmed\n", stderr);
    if ((caches & 4u) && placed_heap) {
        // One placed buffer on a heap of this device, entered where the translated indirect address route
        // keeps its aliases. The page word is only the cache key: no title address is involved here, and
        // the entry never dereferences the heap. The route's buffers carry ALLOW_CROSS_ADAPTER; sharing
        // them across adapters needs a heap opened for it and plays no part in the lifetime measured
        // here, so this buffer is an ordinary placed buffer.
        D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_HEAP_DESC heap_desc{};
        heap_desc.SizeInBytes = 65536;
        heap_desc.Properties = properties;
        heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        x2w_startup::ComPtr<ID3D12Heap> heap;
        const HRESULT heap_hr = desktop->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap));
        if (trace) std::fprintf(stderr, "x2w-cache-probe heap hr=0x%08lx\n", static_cast<unsigned long>(heap_hr));
        if (FAILED(heap_hr)) return owned;
        const D3D12_RESOURCE_DESC placed{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                                         {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        x2w_startup::ComPtr<ID3D12Resource> alias;
        const HRESULT placed_hr = desktop->CreatePlacedResource(heap.Get(), 0, &placed, D3D12_RESOURCE_STATE_COMMON,
                                                               nullptr, IID_PPV_ARGS(&alias));
        if (trace) std::fprintf(stderr, "x2w-cache-probe placed hr=0x%08lx\n", static_cast<unsigned long>(placed_hr));
        if (FAILED(placed_hr)) return owned;
        {
            std::lock_guard lock(placed_mutex);
            auto& pages = placed_page_aliases();
            const bool entered = pages.insert_for_test(heap.Get(), 0x10000, alias.Get());
            unsigned used = 0;
            for (const auto& entry : pages)
                if (entry.heap) ++used;
            if (trace) std::fprintf(stderr, "x2w-cache-probe insert=%d used=%u/64\n", entered ? 1 : 0, used);
            if (entered) {
                // The route's heap belongs to the device's allocation, not to an alias entry: the shim
                // keeps it for the device's lifetime, and the entry only stores the pointer. This probe
                // created its own heap, so the one reference it holds is not abandoned here but handed to
                // the caller, who owns it from now on and releases it after the placed buffer this entry
                // keeps -- a heap must outlive the resources placed in it.
                *placed_heap = heap.Detach();
                ++owned;
            }
        }
    }
    if (caches & 8u) {
        // One recorded draw readback: the READBACK buffer draw_dump_after() keeps and the queue fence
        // execute() attaches once the copy is submitted. execute() builds both on the desktop device it
        // reads from the desktop queue, so both are desktop objects here too.
        D3D12_HEAP_PROPERTIES properties{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = 256;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        x2w_startup::ComPtr<ID3D12Resource> staging;
        x2w_startup::ComPtr<ID3D12Fence> fence;
        const HRESULT staging_hr = desktop->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &buffer,
                                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                                    IID_PPV_ARGS(&staging));
        if (trace) std::fprintf(stderr, "x2w-cache-probe readback hr=0x%08lx\n", static_cast<unsigned long>(staging_hr));
        if (FAILED(staging_hr)) return owned;
        const HRESULT fence_hr = desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (trace) std::fprintf(stderr, "x2w-cache-probe fence hr=0x%08lx\n", static_cast<unsigned long>(fence_hr));
        if (FAILED(fence_hr)) return owned;
        const bool recorded = draw_dumps().record_for_test(std::move(staging), std::move(fence));
        if (trace) std::fprintf(stderr, "x2w-cache-probe record=%d\n", recorded ? 1 : 0);
        if (recorded) owned += 2;
    }
    return owned;
}

// Fixture probe (startup_teardown_unload, #2572) for the ownership of the two caches above: hands the
// caller one reference to an object a warmed entry owns, so the fixture can watch that reference go away
// when the module is unloaded. `cache` and `slot`: 4/0 the placed buffer, 8/0 the readback buffer, 8/1 its
// fence; null for anything else. The caller's reference keeps the object alive after the module is gone,
// which is what makes its count readable then.
extern "C" __declspec(dllexport) IUnknown* x2w_d3d12_cached_object_probe(UINT cache, UINT slot)
{
    IUnknown* object = nullptr;
    if (cache == 4u) {
        std::lock_guard lock(placed_mutex);
        auto& pages = placed_page_aliases();
        for (auto& entry : pages) {
            if (!entry.resource) continue;
            object = entry.resource.Get();
            break;
        }
    } else if (cache == 8u) {
        object = draw_dumps().owned_for_test(slot);
    }
    if (!object) return nullptr;
    object->AddRef();
    return object;
}

// Fixture probes (startup_resource_retire_title) for state the retire bookkeeping maintains for one resource. They read
// the same maps the shim reads and never change them; the address index and the upload cache exist with and without
// the switch, so these answer in both modes.
extern "C" __declspec(dllexport) BOOL x2w_d3d12_resource_index_probe(void* resource)
{
    Wrapper* w = as_wrapper(resource);
    if (!w || w->own().kind != resource_kind) return FALSE;
    const u64 base = reinterpret_cast<u64>(w->own().title_memory);
    if (!base) return FALSE;
    AcquireSRWLockShared(&note_lock);
    const bool indexed = resource_address_index().contains(base, static_cast<const Wrapper*>(w));
    ReleaseSRWLockShared(&note_lock);
    return indexed ? TRUE : FALSE;
}

// The desktop pointer is not dereferenced: the fixture keeps its own desktop reference to a retired resource.
extern "C" __declspec(dllexport) BOOL x2w_d3d12_texture_upload_probe(void* desktop_resource)
{
    if (!desktop_resource) return FALSE;
    AcquireSRWLockExclusive(&draw_lock);
    const bool cached = texture_uploads().count(static_cast<ID3D12Resource*>(desktop_resource)) != 0;
    ReleaseSRWLockExclusive(&draw_lock);
    return cached ? TRUE : FALSE;
}

extern "C" __declspec(dllexport) BOOL x2w_d3d12_written_probe(void* address, unsigned long long* generation, unsigned* bc7_width,
                                                            unsigned* desktop_rows)
{
    if (!address) return FALSE;
    AcquireSRWLockShared(&written_lock);
    const auto found = written_textures().find(reinterpret_cast<u64>(address));
    const bool present = found != written_textures().end();
    if (present) {
        if (generation) *generation = found->second.generation;
        if (bc7_width) *bc7_width = found->second.bc7_width;
        if (desktop_rows) *desktop_rows = found->second.desktop_rows ? 1u : 0u;
    }
    ReleaseSRWLockShared(&written_lock);
    return present ? TRUE : FALSE;
}

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
    if (!enabled || !desc || !first_calls(dumped[slot & 511], 4)) return;
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
    if (!*out) return E_OUTOFMEMORY;
    auto* pipeline = static_cast<Wrapper*>(*out);
    note_pipeline(pipeline, 62, nullptr, desc, sizeof(*desc));
    note_compute_binding_pipeline(pipeline, *desc);
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
    D3D12_RASTERIZER_DESC rasterizer{};
    bool has_rasterizer = false;
    bool compute = false;
    ID3D12RootSignature* pipeline_root = nullptr;
    IndirectGraphicsPipeline graphics;
    u64 mesh86_key = 0;
    if (render_mode()) {
        char why[256] = "";
        // Recognized desktop stages must reach the native path before the
        // guest cache creates a pending translation for any DXBC container.
        real_pipeline = build_stream_pipeline(desc, why, sizeof(why), compute, graphics, pipeline_root);
        bool mesh_cache_request = false;
        if (!real_pipeline)
            real_pipeline = try_cached_mesh86(*desc, why, sizeof(why), graphics, mesh_cache_request, mesh86_key);
        if (!mesh_cache_request && (!real_pipeline || graphics.root))
            translation = stream_translation(*desc, blend, has_blend, depth_stencil, has_depth_stencil, dsv_format,
                                             rasterizer, has_rasterizer);
        static LONG built;
        if ((real_pipeline || !translation) && first_calls(built, 24))
            log("x2w-render stream pipeline -> %s%s", real_pipeline ? "REAL desktop pipeline" : "placeholder: ", real_pipeline ? "" : why);
    }
    *out = wrap(real_pipeline, pipeline_kind);
    if (!*out) {
        if (real_pipeline) real_pipeline->Release();
        return E_OUTOFMEMORY;
    }
    static_cast<Wrapper*>(*out)->own().compute_pipeline = real_pipeline && compute;
    static_cast<Wrapper*>(*out)->own().pipeline_root = pipeline_root;
    static_cast<Wrapper*>(*out)->own().indirect_graphics = graphics;
    static_cast<Wrapper*>(*out)->own().mesh86_key = mesh86_key;
    note_pipeline(static_cast<Wrapper*>(*out), 86, nullptr, desc->pPipelineStateSubobjectStream, desc->SizeInBytes < 0x2000 ? desc->SizeInBytes : 0x2000,
                  translation, translation && has_blend ? &blend : nullptr,
                  translation && has_depth_stencil ? &depth_stencil : nullptr,
                  translation ? dsv_format : DXGI_FORMAT_UNKNOWN,
                  translation && has_rasterizer ? &rasterizer : nullptr);
    return S_OK;
}

// The fixed prefix shared with request_slot57_mesh. Only apply the creation
// refusal to the observed bounded spans, with a registered root and no VS.
bool is_fixed_mesh57_descriptor(const void* desc)
{
    u64 words[15];
    if (!x2w_local::readable(desc, sizeof(words)) || !copy_title_bytes(words, desc, sizeof(words))) return false;
    const auto readable_stage = [](u64 address, u64 bytes) {
        return bytes >= 16 && bytes <= 0x10000 &&
            x2w_local::readable(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes));
    };
    const auto* root = as_wrapper(reinterpret_cast<const void*>(words[0]));
    return root && root->own().kind == root_signature_kind &&
        !words[9] && !words[10] &&
        readable_stage(words[13], words[14]) &&
        ((!words[5] && !words[6]) || readable_stage(words[5], words[6]));
}

// Device slots 57 and 67 (Xbox order): (desc*, riid, ppv) with the pipeline
// IID. These normally return a placeholder. In render mode slot 67 also
// publishes a compute translation request (d3d12_draw.inl).
template <int N>
HRESULT __stdcall device_create_pipeline_placeholder(Wrapper*, const void* desc, const GUID* iid, void** out)
{
    Timed timed(time_pipeline);
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    dump_pipeline("pipeline", N, desc, 0x300);
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    if constexpr (N == 57) {
        if (x2w_local::env_on("X2W_REFUSE_FIXED_MESH") && is_fixed_mesh57_descriptor(desc)) {
            static LONG refusals;
            if (first_calls(refusals, 6))
                log("x2w-d3d12 Device::slot57: fixed mesh pipeline unavailable (E_NOTIMPL)");
            return E_NOTIMPL;
        }
    }
    static LONG calls;
    if (first_calls(calls, 2)) {
        char bytes[3 * 96 + 1];
        hex_text(desc, 96, bytes, sizeof(bytes));
        log("x2w-d3d12 Device::slot%d(pipeline?) desc=[%s] riid=%s -> placeholder", N, bytes, text);
    }
    *out = wrap(nullptr, pipeline_kind);
    if (!*out) return E_OUTOFMEMORY;
    if constexpr (N == 67) {
        // A translated Dispatch finds its program by this key (d3d12_compute_dispatch.inl).
        if (const u64 key = publish_compute_request(desc)) {
            static_cast<Wrapper*>(*out)->own().compute_request_key = key;
            static_cast<Wrapper*>(*out)->own().compute_request_root = as_wrapper(*static_cast<void* const*>(desc));
        }
    }
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
    if (!*out) return E_OUTOFMEMORY;
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
// Slot 76 shares the work with the slot number in the log line.
HRESULT create_committed(int slot, Wrapper* self, const D3D12_HEAP_PROPERTIES* props, UINT heap_flags,
                         const D3D12_RESOURCE_DESC* desc, UINT state, const D3D12_CLEAR_VALUE* clear, void* session,
                         UINT castable_count, const void* castable, const GUID* iid, void** out)
{
    char text[48];
    guid_text(iid, text, sizeof(text));
    if (!out || !props || !desc || !iid) return E_POINTER;
    *out = nullptr;
    const D3D12_RESOURCE_DESC copy = sanitized(*desc);
    const auto flags = static_cast<D3D12_HEAP_FLAGS>(heap_flags);
    const D3D12_RESOURCE_ALLOCATION_INFO info = device_of(self)->GetResourceAllocationInfo(0, 1, &copy);
    check_device("Device before slot", slot);
    void* const title_memory = info.SizeInBytes != ~0ull ? title_resource_memory(info.SizeInBytes, info.Alignment) : nullptr;
    ID3D12Resource* resource = nullptr;
    const D3D12_RESOURCE_STATES initial = props->Type == D3D12_HEAP_TYPE_UPLOAD     ? D3D12_RESOURCE_STATE_GENERIC_READ
                                          : props->Type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                                    : D3D12_RESOURCE_STATE_COMMON;
    D3D12_CLEAR_VALUE clear_copy{};
    const auto* clear_value = packed_clear_value(clear, desktop_format(desc->Format), copy.Format, clear_copy);
    HRESULT hr = kind_for(*iid) == resource_kind
        ? device_of(self)->CreateCommittedResource(props, flags, &copy, initial, clear_value, IID_PPV_ARGS(&resource))
        : E_NOINTERFACE;
    bool cleared = clear != nullptr;
    if (FAILED(hr) && clear && kind_for(*iid) == resource_kind) {
        hr = device_of(self)->CreateCommittedResource(props, flags, &copy, initial, nullptr, IID_PPV_ARGS(&resource));
        cleared = false;
    }
    static LONG calls;
    if (first_calls(calls, 12))
        log("x2w-d3d12 Device::slot%d(%s) heap=%d flags=0x%x dim=%d %llux%u fmt=%d rflags=0x%x "
            "state=0x%x clear=%p%s session=%p castable=%u/%p riid=%s -> 0x%08lx", slot,
            slot == 76 ? "CreateCommittedResource" : "CreateCommittedResource3?", props->Type, heap_flags, desc->Dimension,
            desc->Width, desc->Height, desc->Format, desc->Flags, state, static_cast<const void*>(clear),
            clear && !cleared ? "(dropped)" : "", session, castable_count, castable, text, static_cast<unsigned long>(hr));
    if (FAILED(hr)) return hr;
    x2w_startup::mark_packed_stand_in(resource, desktop_format(desc->Format));
    Wrapper* w = wrap(resource, resource_kind);
    if (!w) {
        // The title-memory block is not returned here. __XMemAllocationHooks__ is documented as
        // {alloc, free, ...} (title_alloc_hook above), but the free entry's signature and attributes
        // have not been verified against the title, so this failure path still leaks the block. A
        // follow-up has to confirm those before calling it.
        resource->Release();
        return E_OUTOFMEMORY;
    }
    w->own().budget = x2w_budget::committed(device_of(self), resource, props->Type);
    w->own().title_memory = title_memory;
    retire_set_class(w, *props, flags, copy, title_memory);
    note_resource(w, desc, info.SizeInBytes != ~0ull ? info.SizeInBytes : 0);
    *out = w;
    return S_OK;
}

HRESULT __stdcall device_create_committed_x(Wrapper* self, const D3D12_HEAP_PROPERTIES* props, UINT heap_flags,
                                            const D3D12_RESOURCE_DESC* desc, UINT state, const D3D12_CLEAR_VALUE* clear,
                                            void* session, UINT castable_count, const void* castable, const GUID* iid,
                                            void** out)
{
    return create_committed(127, self, props, heap_flags, desc, state, clear, session, castable_count, castable, iid, out);
}

// Device slot 76 (Xbox order, +0x260): the seven arguments of desktop CreateCommittedResource
// (heap properties, heap flags, desc, initial state, clear value, riid, ppv). Seen from a console
// package module (an upscaler) that a title loaded only on the Series S profile. Same desktop
// creation as slot 127 (tests/startup_committed_slot76_check.cpp).
HRESULT __stdcall device_create_committed(Wrapper* self, const D3D12_HEAP_PROPERTIES* props, UINT heap_flags,
                                          const D3D12_RESOURCE_DESC* desc, UINT state, const D3D12_CLEAR_VALUE* clear,
                                          const GUID* iid, void** out)
{
    return create_committed(76, self, props, heap_flags, desc, state, clear, nullptr, 0, nullptr, iid, out);
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
        X2W_ZONE("frame wait (sleep)");
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

// Every 300 frames: GPU ms per frame (average and worst) and the share of the wall time the queue sat idle.
// The probe's submit side runs under draw_lock, so the read takes it too.
void log_gpu_time(LONG frame, ULONGLONG now_ms)
{
    static ULONGLONG last_ms;
    AcquireSRWLockExclusive(&draw_lock);
    auto& probe = gpu_time_probe();
    probe.harvest(static_cast<std::uint64_t>(frame));
    const auto r = probe.stats.take(last_ms ? static_cast<double>(now_ms - last_ms) : 0);
    const unsigned skipped = probe.skipped, failed = probe.failed;
    ReleaseSRWLockExclusive(&draw_lock);
    last_ms = now_ms;
    log("x2w-gpu-time frame %ld: %llu frames, avg %.2f ms, max %.2f ms, busy %.0f ms of %.0f ms, GPU idle %.0f%% (stamps skipped %u, failed %u)", frame,
        static_cast<unsigned long long>(r.frames), r.avg_ms, r.max_ms, r.busy_ms, r.wall_ms, r.idle_share * 100.0, skipped, failed);
}
#include "startup_stubs/d3d12_present.inl"

// Render mode: a coarse signature of the presented picture for input-script waits (present_signature.h,
// present_signature_sampler.h). The block always counts presents; samples are taken every 4 presents
// only while a script with a picture wait has asked for them (request_sampling). X2W_PRESENT_SIGNATURE_EVERY=<n>
// samples every n presents from the start without a request, and 0 turns the signature off.
// 18 one-texel rows are copied on the presenting queue and read back on a later present; nothing waits.
void present_signature(Wrapper* queue, Wrapper* surface, LONG frame)
{
    static const int explicit_every = [] {
        char value[16]{};
        return GetEnvironmentVariableA("X2W_PRESENT_SIGNATURE_EVERY", value, sizeof(value)) ? std::atoi(value) : -1;
    }();
    if (explicit_every == 0) return;
    static const int every = explicit_every > 0 ? explicit_every : 4;
    static x2w_present_signature::Block* const block = [] {
        auto* created = x2w_present_signature::create_block(GetCurrentProcessId(), static_cast<std::uint32_t>(every));
        log("x2w-present picture signature: Local\\x2w_present_signature_%lu, every %d presents %s",
            static_cast<unsigned long>(GetCurrentProcessId()), every,
            explicit_every > 0 ? "from the start (X2W_PRESENT_SIGNATURE_EVERY)" : "once a script wait asks for it");
        return created;
    }();
    // Kept until the process ends, so a copy in flight never loses the objects it uses.
    static auto* const sampler = new x2w_present_signature::Sampler(every, explicit_every > 0, red_producers().enabled());
    ID3D12Resource* real = surface && surface->own().kind == resource_kind ? static_cast<ID3D12Resource*>(surface->own().real) : nullptr;
    const bool published = sampler->present(device_of(the_device), static_cast<ID3D12CommandQueue*>(queue->own().real), real, frame, block);
    static LONG first, refused, failed;
    if (published && red_producers().enabled() && sampler->last_format() == DXGI_FORMAT_R8G8B8A8_UNORM) {
        const auto* rgba = sampler->last_rgba();
        const bool red = rgba[0] == 255 && rgba[1] == 0 && rgba[2] == 0;
        const bool matched = red_producers().matches(rgba);
        red_producers().observe(sampler->last_present(), frame, sampler->last_surface(), matched);
        static bool was_matched;
        if (matched != was_matched || red_producers().interested()) {
            log("x2w-red sample present=%lld published_at=%ld target=%p rgba=%u,%u,%u,%u red=%u match=%u tick=%llu",
                sampler->last_present(), frame, sampler->last_surface(), rgba[0], rgba[1], rgba[2], rgba[3], red, matched, GetTickCount64());
            was_matched = matched;
        }
    }
    if (published && first_calls(first, 1))
        log("x2w-present picture signature: first sample published at present %ld in Local\\x2w_present_signature_%lu (every %d presents)",
            frame, static_cast<unsigned long>(GetCurrentProcessId()), every);
    if (sampler->unsupported_format() != DXGI_FORMAT_UNKNOWN && first_calls(refused, 1))
        log("x2w-present picture signature: surface format %d is not sampled", sampler->unsupported_format());
    if (sampler->failed() && first_calls(failed, 1))
        log(sampler->stranded() ? "x2w-present picture signature: Signal failed after a copy was submitted; sampling stopped, its surface stays referenced"
                                : "x2w-present picture signature: setup failed, no samples");
}

// One line with the heartbeat while X2W_D3D12_TEXTURE_DROP_MIPS is set: how many textures lost levels, the estimated
// bytes saved, and every reason a texture kept all of them (counted once per first use).
void report_drop_mips()
{
    namespace dm = x2w_startup::drop_mips;
    if (!drop_mips_config().levels) return;
    char reasons[320]{};
    std::size_t used = 0;
    for (unsigned r = 1; r < static_cast<unsigned>(dm::Reason::count); ++r) {
        const auto n = dm::counters.kept[r].load(std::memory_order_relaxed);
        if (!n || used + 40 >= sizeof(reasons)) continue;
        used += static_cast<std::size_t>(std::snprintf(reasons + used, sizeof(reasons) - used, " %s=%llu",
                                                       dm::reason_name(static_cast<dm::Reason>(r)),
                                                       static_cast<unsigned long long>(n)));
    }
    log("x2w-drop-mips levels=%u min_bytes=%llu dropped=%llu saved_mib=%llu conflicts=%llu upload_unsupported=%llu no_drop:%s",
        drop_mips_config().levels, static_cast<unsigned long long>(drop_mips_config().min_bytes),
        static_cast<unsigned long long>(dm::counters.dropped.load()),
        static_cast<unsigned long long>(dm::counters.saved_bytes.load() >> 20),
        static_cast<unsigned long long>(dm::counters.conflicts.load()),
        static_cast<unsigned long long>(dm::counters.upload_unsupported.load()), used ? reasons : " none");
}

void report_budget(ID3D12CommandQueue* queue)
{
    using Microsoft::WRL::ComPtr;
    ComPtr<ID3D12Device> device;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter3> adapter;
    if (SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&device))) && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        const LUID luid = device->GetAdapterLuid();
        if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
            ComPtr<IDXGIAdapter1> warp;
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) && SUCCEEDED(warp->GetDesc1(&desc)) &&
                desc.AdapterLuid.LowPart == luid.LowPart && desc.AdapterLuid.HighPart == luid.HighPart)
                warp.As(&adapter);
        }
    }
    char local[48] = "n/a", nonlocal[48] = "n/a";
    const auto sample = [&](DXGI_MEMORY_SEGMENT_GROUP segment, char (&text)[48]) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (adapter && SUCCEEDED(adapter->QueryVideoMemoryInfo(0, segment, &info)))
            std::snprintf(text, sizeof(text), "%llu/%llu", info.CurrentUsage, info.Budget);
    };
    sample(DXGI_MEMORY_SEGMENT_GROUP_LOCAL, local);
    if (residency_observe()) {
        // Observation only (X2W_D3D12_RESIDENCY=1): how often the local group is over its budget at a heartbeat.
        static LONG samples, over;
        unsigned long long used_now = 0, budget_now = 0;
        const bool have = std::sscanf(local, "%llu/%llu", &used_now, &budget_now) == 2 && budget_now;
        if (have) {
            InterlockedIncrement(&samples);
            if (used_now > budget_now) InterlockedIncrement(&over);
            log("x2w-residency pressure samples %ld, over budget %ld", samples, over);
        }
    }
    sample(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, nonlocal);
    const auto counts = x2w_budget::snapshot();
    char text[640]{};
    std::size_t used = 0;
    for (int i = 0; i < x2w_budget::category_count; ++i) {
        const auto& active = counts.live[i];
        const auto& held = counts.retained[i];
        const int n = std::snprintf(text + used, sizeof(text) - used, " %s=%llu/%llu+%llu/%llu", x2w_budget::names[i],
                                    active.bytes, active.objects, held.bytes, held.objects);
        if (n < 0 || static_cast<std::size_t>(n) >= sizeof(text) - used) break;
        used += n;
    }
    // System commit and the physical-page pool behind it: a run that ends on a commit limit (error 1455) shows
    // whether the pool (sections, slots) or the resources above grew (#1199).
    char commit[80];
    x2w_local::format_commit(commit, sizeof(commit), x2w_local::system_commit());
    char job[64];
    x2w_local::format_job(job, sizeof(job), x2w_local::job_memory());
    char pool[96] = "sections=n/a slabs=n/a slots=n/a";
    using PageStats = BOOL(WINAPI*)(ULONGLONG*);
    using SlabStats = BOOL(WINAPI*)(ULONGLONG*, ULONGLONG*, ULONGLONG*);
    HMODULE xmem = nullptr;
    if (GetModuleHandleExW(0, L"xmem.dll", &xmem)) {  // adds a reference: released below
        ULONGLONG sections = 0, slabs = 0, in_use = 0, free_slots = 0;
        const auto page_stats = reinterpret_cast<PageStats>(GetProcAddress(xmem, "x2w_xmem_page_stats"));
        const auto slab_stats = reinterpret_cast<SlabStats>(GetProcAddress(xmem, "x2w_xmem_slab_stats"));
        if (page_stats && slab_stats && page_stats(&sections) && slab_stats(&slabs, &in_use, &free_slots))
            std::snprintf(pool, sizeof(pool), "sections=%llu slabs=%llu slots=%llu/%llu", sections, slabs, in_use, free_slots);
        FreeLibrary(xmem);
    }
    log("x2w-budget local=%s nonlocal=%s%s unmeasured=%llu %s %s %s sysmem-fallback=%llu bytes=%llu", local, nonlocal, text, counts.unmeasured,
        commit, pool, job, counts.sysmem_fallbacks, counts.sysmem_fallback_bytes);
}

// X2W_PRESENT_COMPOSITE=1 (experimental, docs/startup-runtime.md#presented-planes): the output
// window composites every presented plane. Unset, the window shows plane 0 alone, as before.
bool present_composite_enabled()
{
    static const bool on = x2w_local::env_on("X2W_PRESENT_COMPOSITE");
    return on;
}

// The planes the output window draws for a present, resolved to real resources it can draw.
// X2W_PRESENT_PLANE=<n> (diagnostic) keeps plane n alone (plane 0 when n is past the count). Otherwise
// plane 0 alone is kept, unless X2W_PRESENT_COMPOSITE=1 keeps every plane for the composite
// (present_planes.h). The baseline capture and readback do not use this: they read plane 0 as before.
struct PresentedPlanes {
    ID3D12Resource* drawable[window_planes] = {};  // per plane, null when left out
    x2w_startup::PlaneStep steps[window_planes];
    unsigned count = 0;          // composite steps
    unsigned top_plane = 0;      // the highest kept plane with a known resource
    ID3D12Resource* top = nullptr;  // its surface
};
PresentedPlanes presented_planes(UINT planes, const void* plane_parameters)
{
    static const long wanted = [] {
        char value[16]{};
        return GetEnvironmentVariableA("X2W_PRESENT_PLANE", value, sizeof(value)) ? std::atol(value) : -1L;
    }();
    const auto readable = [](const void* address, std::size_t bytes) { return x2w_local::readable(address, bytes); };
    PresentedPlanes out;
    std::uint64_t shown[window_planes] = {};
    const unsigned total = planes < window_planes ? planes : window_planes;
    const unsigned only = wanted >= 0 ? (static_cast<unsigned long>(wanted) < planes ? static_cast<unsigned>(wanted) : 0u)
                          : present_composite_enabled() ? ~0u
                                                        : 0u;
    for (unsigned i = 0; i < total; ++i) {
        if (only != ~0u && i != only) continue;
        const std::uint64_t resource = x2w_startup::present_plane_resource(plane_parameters, planes, i, readable);
        Wrapper* const surface = resource ? as_wrapper(reinterpret_cast<void*>(resource)) : nullptr;
        if (!surface || surface->own().kind != resource_kind || !surface->own().real) continue;
        auto* const real = static_cast<ID3D12Resource*>(surface->own().real);
        out.top_plane = i;
        out.top = real;
        if (!presentable_surface(real)) continue;
        out.drawable[i] = real;
        shown[i] = resource;
    }
    out.count = x2w_startup::composite_steps(shown, total, out.steps, window_planes);
    return out;
}

// The first presents and every 300th: each plane's first resource and its shape.
void log_present_planes(LONG frame, UINT planes, const void* plane_parameters)
{
    if (frame > 2 && frame % 300 != 0) return;
    static LONG lines;
    if (!first_calls(lines, 24)) return;
    const auto readable = [](const void* address, std::size_t bytes) { return x2w_local::readable(address, bytes); };
    for (unsigned i = 0; i < planes && i < 4; ++i) {
        const std::uint64_t resource = x2w_startup::present_plane_resource(plane_parameters, planes, i, readable);
        Wrapper* const surface = resource ? as_wrapper(reinterpret_cast<void*>(resource)) : nullptr;
        if (!surface || surface->own().kind != resource_kind || !surface->own().real) {
            log("x2w-present frame %ld plane %u/%u: resource %p is not a known resource", frame, i, planes, reinterpret_cast<void*>(resource));
            continue;
        }
        const D3D12_RESOURCE_DESC desc = static_cast<ID3D12Resource*>(surface->own().real)->GetDesc();
        log("x2w-present frame %ld plane %u/%u: %llux%u format %d flags=0x%x", frame, i, planes,
            static_cast<unsigned long long>(desc.Width), desc.Height, desc.Format, static_cast<UINT>(desc.Flags));
    }
}

// Queue slot 32 (Xbox order): (plane count, plane parameters*, present parameters*) -> HRESULT, the
// console present. Nothing is shown yet; frames are counted and a heartbeat is logged.
HRESULT __stdcall queue_present(Wrapper* self, UINT planes, const void* plane_parameters, const void* present_parameters)
{
    // First, before anything reads or copies the surface: a presented resource keeps its old lifetime.
    if (release_retired_enabled() && planes && plane_parameters) {
        const auto* words = static_cast<const u64*>(plane_parameters);
        if (x2w_local::readable(words, 0x18) && words[1] >= 1 && x2w_local::readable(reinterpret_cast<const void*>(words[2]), 8))
            retire_leave_class(*reinterpret_cast<void* const*>(words[2]), RetireLeft::present);
    }
    drain_retired_resources();
    Timed timed(time_present);
    X2W_ZONE("Present");
    X2W_FRAME_MARK;
    static LONG frames;
    static ULONGLONG started, last_report;
    const LONG frame = InterlockedIncrement(&frames);
    gpu_time_frame = frame;
    frame_number = frame;
    draw_dump_flush();
    red_producers().flush();
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
    log_present_planes(frame, planes, plane_parameters);
    // X2W_PRESENT_COMPOSITE=1: the highest plane above plane 0, sampled raw below.
    ID3D12Resource* raw_surface = nullptr;
    unsigned raw_plane = 0;
    {
        // The baseline: plane 0's first resource, captured, sampled and read back as before the planes
        // were decoded, with or without X2W_PRESENT_COMPOSITE.
        Wrapper* surface = nullptr;
        const auto* words = static_cast<const u64*>(plane_parameters);
        if (planes && words && x2w_local::readable(words, 0x18) && words[1] >= 1 && x2w_local::readable(reinterpret_cast<const void*>(words[2]), 8))
            surface = as_wrapper(*reinterpret_cast<void* const*>(words[2]));
        capture::present(frame, surface);
        if (surface && surface->own().kind == resource_kind)
            red_producers().present(static_cast<ID3D12Resource*>(surface->own().real));
        // Render mode: the presented planes go to the output window through a swap chain: plane 0 alone
        // by default, every plane with X2W_PRESENT_COMPOSITE=1. The compositor draws the steps in order.
        if (render_mode()) {
            const PresentedPlanes presented = presented_planes(planes, plane_parameters);
            if (presented.count)
                shown_in_window = present_to_window(static_cast<ID3D12CommandQueue*>(self->own().real), presented.steps, presented.count,
                                                    presented.drawable, present_composite_enabled() ? window_planes : 1);
            if (present_composite_enabled() && presented.top && presented.top_plane > 0) {
                raw_surface = presented.top;
                raw_plane = presented.top_plane;
            }
        }
        InterlockedExchange(&last_frame_shown, shown_in_window ? 1 : 0);
        if (render_mode()) present_signature(self, surface, frame);
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
        // The raw plane sample is that plane's own surface, not the composited window picture.
        if (raw_surface) read_back_presented(static_cast<ID3D12CommandQueue*>(self->own().real), raw_surface, static_cast<int>(raw_plane));
    }
    if (shown_in_window) {
        if (frame % 30 == 0) {
            // About 100 fixed characters, the revision (at most 46) and three numbers fit in 256.
            wchar_t text[256];
            swprintf(text, 256, L"Xbox2Windows %ls - frame %ld, %.1f fps, %ld translated draws (title bar is diagnostics, picture is game output)",
                     x2w_build_rev_w, frame, frame * 1000.0 / (now - started ? now - started : 1), translated_draws);
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
    if (render_mode() && frame % 300 == 0 && gpu_time_on()) log_gpu_time(frame, now);
    if (render_mode() && frame % 300 == 0 && residency_observe()) residency_report(frame);
    if (render_mode() && frame % 60 == 0) residency_tick(frame);
    if (render_mode() && frame % 300 == 0)
        log("x2w-render totals: resets %ld (failed %ld), closes %ld, clears %ld, target binds %ld (unresolved %ld), executes %ld, draws %ld (skipped %ld), "
            "translated draws %ld (not drawn %ld), pipelines translated %ld / missing %ld, null bindings %ld (entry %ld, buffer %ld, token %ld, wrapper %ld, dimension %ld, ring_full %ld), "
            "formatted-load draws refused %ld, textures uploaded %ld (%ld KiB, unreadable %ld), reads refused before a fault %ld, heap-indexed roots deferred %ld (real draws refused: view heap %ld, "
            "sampler heap %ld, both heaps %ld, state %ld), pipelines first looked up by ExecuteIndirect %ld, compute requests published %ld (already present %ld, failed %ld, pointer-like blob qwords %ld)", render_resets,
            render_reset_failures, render_closes, render_clears, render_target_binds, render_target_unresolved, render_executes, render_draws, render_draws_skipped,
            translated_draws, untranslated_draws, translation_ready, translation_missing, null_total,
            null_causes[0], null_causes[1], null_causes[2], null_causes[3], null_causes[4], null_causes[5], formatted_draws_refused, textures_uploaded,
            texture_bytes_uploaded, textures_unreadable, static_cast<long>(x2w_local::faults_avoided), heap_roots_deferred,
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::view_heap_missing)],
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::sampler_heap_missing)],
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::both_heaps_missing)],
            heap_root_refusals[static_cast<int>(x2w_startup::RealRootDecision::state_not_restorable)], indirect_first_lookups,
            compute_requests_published, compute_requests_present, compute_requests_failed, compute_blob_pointer_words);
    if (render_mode() && frame % 300 == 0) {
        UploadRingStats ring = upload_ring_stats();
        log("x2w-render upload ring: reserved %llu MiB in %ld reservations, refusals %ld, chunks %zu (%llu MiB), chunks recycled %ld",
            ring.reserved_bytes >> 20, ring.reservations, ring.refusals, ring.chunks, ring.capacity >> 20, ring.recycled);
        log("x2w-render compute dispatches: translated %ld, refused %ld, no translation %ld, not modelled %ld; placed inputs from a GPU copy %ld, from a title snapshot %ld (%ld KiB)",
            compute_dispatches_translated, compute_dispatches_refused, compute_dispatches_untranslated, compute_dispatches_unmodelled,
            compute_inputs_gpu_copied, compute_inputs_snapshot, compute_inputs_snapshot_kib);
        log_compute_census();
        log_not_drawn_causes();
        static ULONGLONG last_key_report;  // the by-pipeline ranking costs a sort: about once a minute
        if (now - last_key_report >= 60000) {
            last_key_report = now;
            log_not_drawn_keys();
        }
    }
    if (render_mode() && frame % 300 == 0) {
        // Slot-61 calls since the start, by outcome, then by signature kind (zero kinds left out).
        static const char* const outcomes[] = {"forwarded", "refused graphics-pipeline", "compute-pipeline",
                                               "placeholder-signature", "other"};
        static const char* const kinds[] = {"other", "dispatch", "draw", "drawindexed", "dispatchmesh"};
        char line[1024];  // fits every outcome with five 19-digit counts
        int n = std::snprintf(line, sizeof(line), "x2w-render indirect:");
        for (int o = 0; o < indirect_outcomes; ++o) {
            n += std::snprintf(line + n, sizeof(line) - n, "%s %s:", o ? ";" : "", outcomes[o]);
            const int start = n;
            for (int k = 0; k < 5; ++k)
                if (const LONG64 calls = indirect_calls[o][k])
                    n += std::snprintf(line + n, sizeof(line) - n, " %s %lld", kinds[k], static_cast<long long>(calls));
            if (n == start) n += std::snprintf(line + n, sizeof(line) - n, " 0");
        }
        log("%s", line);
    }
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
        log("x2w-d3d12 views noted %lld (distinct %lld), descriptors copied %lld, heaps %llu, placed buffers %lld "
            "(region-information proofs %lld, region queries %lld)", views_noted, views_distinct, descriptors_copied,
            static_cast<u64>(heap_infos().size()), static_cast<long long>(placed_buffer_calls),
            static_cast<long long>(placed_buffer_region_info), static_cast<long long>(placed_buffer_queries));
        {
            char line[640];
            if (view_census_line(line, sizeof(line))) log("%s", line);
        }
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
            log("x2w-d3d12 list copies: buffer %ld, resource %ld, resolves %ld, refused %ld, buffer copies made in title memory %ld",
                InterlockedCompareExchange(&list_buffer_copies, 0, 0), InterlockedCompareExchange(&list_resource_copies, 0, 0),
                InterlockedCompareExchange(&list_resolves, 0, 0), InterlockedCompareExchange(&list_copies_refused, 0, 0),
                InterlockedCompareExchange(&list_title_memory_copies, 0, 0));
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
        memory_srv_census.report([](const char* line) { log("%s", line); }, true);
        token_null_census.report([](const char* line) { log("%s", line); }, true);
        memory_srv_writes.report([](const char* line) { log("%s", line); }, true);
        indirect_draw_refusal_census.report([](const char* line) { log("%s", line); }, true);
        indirect_other_refusal_census.report([](const char* line) { log("%s", line); }, true);
        log("x2w-d3d12 heartbeat: %ld frames presented in %.1f s (%.1f fps overall); frame waits by type %ld/%ld/%ld/%ld, "
            "%lld ms slept", frame, (now - started) / 1000.0, frame * 1000.0 / (now - started ? now - started : 1),
            frame_waits[0], frame_waits[1], frame_waits[2], frame_waits[3], static_cast<long long>(frame_wait_ms));
        if (x2w_budget::enabled()) report_budget(static_cast<ID3D12CommandQueue*>(self->own().real));
        report_drop_mips();
        if (retire_reasons_enabled()) {
            char line[1536];
            if (retire_reasons_line(line, sizeof(line))) log("%s", line);
        }
    }
    return S_OK;
}

// Fixtures: the census lines of the five-second report, now and whether or not they changed, one per
// line. Returns the length of the whole text and copies up to capacity - 1 bytes of it, terminated.
extern "C" __declspec(dllexport) std::size_t x2w_d3d12_census_report(char* out, std::size_t capacity)
{
    std::size_t length = 0;
    const auto emit = [&](const char* line) {
        const std::size_t bytes = std::strlen(line);
        for (std::size_t i = 0; i <= bytes; ++i, ++length)
            if (out && length + 1 < capacity) out[length] = i < bytes ? line[i] : '\n';
    };
    memory_srv_census.report(emit, false);
    token_null_census.report(emit, false);
    memory_srv_writes.report(emit, false);
    if (indirect_draw_refusal_census.total()) indirect_draw_refusal_census.report(emit, false);
    if (indirect_other_refusal_census.total()) indirect_other_refusal_census.report(emit, false);
    if (out && capacity) out[length < capacity ? length : capacity - 1] = 0;
    return length;
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
    tables[device_kind][15] = reinterpret_cast<void*>(&device_descriptor_increment_size);  // GetDescriptorHandleIncrementSize(type)
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
    tables[device_kind][39] = reinterpret_cast<void*>(&device_create_query_heap);  // CreateQueryHeap(desc, riid, ppv)
    enable_forward<device_kind, 40>();                   // SetStablePowerState
    tables[device_kind][41] = reinterpret_cast<void*>(&device_create_command_signature);                 // CreateCommandSignature(desc, root, riid, ppv)
    enable_forward<device_kind, 42>();                   // GetResourceTiling
    // Slot 43 (desktop GetAdapterLuid) is not installed: no call site has confirmed it.
    (void)&device_adapter_luid;
    // Xbox order from here.
    tables[device_kind][48] = reinterpret_cast<void*>(&device_create_command_queue_x);
    tables[device_kind][70] = reinterpret_cast<void*>(&device_memory_srv);
    tables[device_kind][73] = reinterpret_cast<void*>(&device_create_command_list_x);
    tables[device_kind][76] = reinterpret_cast<void*>(&device_create_committed);
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
        list_typed[51] = reinterpret_cast<ListTyped>(&list_clear_depth_stencil);
        // Slot 51 is also entered through the generic invoke() path (the fixture calls list vtable[51]
        // directly), so it must be present in tables[list_kind] as well as list_typed; siblings 52-55
        // are only reached through list_typed and do not set both.
        tables[list_kind][51] = reinterpret_cast<void*>(&list_clear_depth_stencil);
        list_typed[52] = reinterpret_cast<ListTyped>(&list_clear_target);
        list_typed[15] = reinterpret_cast<ListTyped>(&list_copy_buffer_region);
        list_typed[17] = reinterpret_cast<ListTyped>(&list_copy_resource);
        list_typed[19] = reinterpret_cast<ListTyped>(&list_resolve_subresource);
        list_typed[53] = reinterpret_cast<ListTyped>(&list_clear_uav<false>);
        list_typed[54] = reinterpret_cast<ListTyped>(&list_clear_uav<true>);
        list_typed[55] = reinterpret_cast<ListTyped>(&list_discard);
        list_typed[139] = reinterpret_cast<ListTyped>(&list_reset);
        list_typed[148] = reinterpret_cast<ListTyped>(&list_allocate_upload);
        list_typed[12] = reinterpret_cast<ListTyped>(&list_draw);
        list_typed[14] = reinterpret_cast<ListTyped>(&list_dispatch);
        list_typed[16] = reinterpret_cast<ListTyped>(&list_copy_texture);
        list_typed[20] = reinterpret_cast<ListTyped>(&list_topology);
        list_typed[21] = reinterpret_cast<ListTyped>(&list_viewports);
        list_typed[22] = reinterpret_cast<ListTyped>(&list_scissors);
        list_typed[23] = reinterpret_cast<ListTyped>(&list_blend_factor);
        list_typed[24] = reinterpret_cast<ListTyped>(&list_stencil_ref);
        list_typed[25] = reinterpret_cast<ListTyped>(&list_pipeline);
        list_typed[x2w_list_slots::resource_barrier_slot] = reinterpret_cast<ListTyped>(&list_resource_barriers);
        list_typed[32] = reinterpret_cast<ListTyped>(&list_set_descriptor_heaps);
        list_typed[33] = reinterpret_cast<ListTyped>(&list_compute_root);
        list_typed[34] = reinterpret_cast<ListTyped>(&list_graphics_root);
        list_typed[43] = reinterpret_cast<ListTyped>(&list_compute_upload_srv);
        list_typed[44] = reinterpret_cast<ListTyped>(&list_root_srv);
        list_typed[42] = reinterpret_cast<ListTyped>(&list_root_cbv);
        list_typed[50] = reinterpret_cast<ListTyped>(&list_targets);
        list_typed[13] = reinterpret_cast<ListTyped>(&list_draw_indexed);
        list_typed[35] = reinterpret_cast<ListTyped>(&list_compute_table);
        list_typed[36] = reinterpret_cast<ListTyped>(&list_table);
        list_typed[37] = reinterpret_cast<ListTyped>(&list_compute_constant);
        list_typed[39] = reinterpret_cast<ListTyped>(&list_compute_constants);
        list_typed[40] = reinterpret_cast<ListTyped>(&list_root_constants);
        list_typed[47] = reinterpret_cast<ListTyped>(&list_index_buffer);
        list_typed[48] = reinterpret_cast<ListTyped>(&list_vertex_buffers);
        list_typed[x2w_list_slots::execute_indirect_slot] = reinterpret_cast<ListTyped>(&list_execute_indirect);
        // X2W_D3D12_QUERY_SLOTS=1 (diagnostic): which console methods sit at 28 to 30 is a hypothesis, so the query
        // bindings are opt-in. Unset, the slots stay no-ops.
        if (wchar_t on[4]{}; GetEnvironmentVariableW(L"X2W_D3D12_QUERY_SLOTS", on, 4) && on[0] == L'1') {
            list_typed[28] = reinterpret_cast<ListTyped>(&list_begin_query);
            list_typed[29] = reinterpret_cast<ListTyped>(&list_end_query);
            list_typed[30] = reinterpret_cast<ListTyped>(&list_resolve_query);
        }
    }
    tables[device_kind][16] = reinterpret_cast<void*>(&device_create_root_signature);

    // Queue: not the desktop order (slot 18 is GetTimestampFrequency, desktop 16). Slots are
    // dispatched by argument shape until the order is known; see queue_dynamic.
    fill_queue_dynamic(tables[queue_kind], std::make_index_sequence<64>{});
    tables[queue_kind][17] = reinterpret_cast<void*>(&queue_wait);
    tables[queue_kind][18] = reinterpret_cast<void*>(&queue_timestamp_frequency);
    tables[queue_kind][31] = reinterpret_cast<void*>(&queue_page_mappings);
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
    tables[resource_kind][10] = reinterpret_cast<void*>(&resource_get_desc);
    tables[resource_kind][11] = reinterpret_cast<void*>(&resource_gpu_address);
    // Desktop Resource 12-14, Heap 8 and DescriptorHeap 8 (GetDesc, struct returns) are
    // not installed until a call site confirms them: the queue's slot 18 showed what a wrong one does.
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
    written.desktop_rows = !bc7_width;
    ReleaseSRWLockExclusive(&written_lock);
}
}  // namespace

void x2w_startup::attach_device(ID3D12Device* device, void** wrapped)
{
    static const bool ready = (build_tables(), true);
    (void)ready;
    Wrapper* const candidate = wrap(device, device_kind);
    *wrapped = candidate;
    if (!candidate) {
        device->Release();
        return;
    }
    the_device = candidate;
    removal_device = device;
}

HRESULT x2w_startup::resolve_placed_build(void* guest_device, UINT64 dest, UINT64 scratch,
    const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& sizes,
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& build, x2w_startup::PlacedBuildCause* reported)
{
    using Cause = x2w_startup::PlacedBuildCause;
    // One log line per cause, for the first refusal of that kind in the process.
    static LONG logged[static_cast<std::size_t>(Cause::count)];
    // What the resolver published for each placed build (flags per structure), see PlacedBuildLedger.
    static SRWLOCK ledger_lock = SRWLOCK_INIT;
    static x2w_startup::PlacedBuildLedger ledger;
    const auto refuse = [&](Cause cause, const char* role, UINT64 address) {
        if (reported) *reported = cause;
        if (first_calls(logged[static_cast<std::size_t>(cause)], 1))
            log("x2w-dxr placed build refused: %s (%s at title 0x%llx)", x2w_startup::cause_name(cause), role,
                static_cast<unsigned long long>(address));
        return E_INVALIDARG;
    };
    const bool update = (build.Inputs.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE) != 0;
    const UINT64 source = build.SourceAccelerationStructureData;
    if (const Cause shape = x2w_startup::check_update_shape(update, source, dest, sizes.ResultDataMaxSizeInBytes);
        shape != Cause::ok)
        return refuse(shape, "source", source);
    constexpr UINT64 alignment = x2w_startup::build_alignment;
    struct Translated {
        Cause cause;
        UINT64 gpu;
    };
    const auto translate = [&](UINT64 address, UINT64 bytes, D3D12_RESOURCE_STATES state) -> Translated {
        if (address % alignment) return {Cause::unaligned, 0};
        const auto found = placed_buffer_range(static_cast<const Wrapper*>(guest_device), address, bytes);
        if (!found.owner || found.reject) return {Cause::not_placed, 0};
        const auto* info = resource_info(found.owner);
        if (!info || info->initial_state != state) return {Cause::wrong_state, 0};
        const auto base = reinterpret_cast<UINT64>(found.owner->own().title_memory);
        const auto gpu = static_cast<ID3D12Resource*>(found.owner->own().real)->GetGPUVirtualAddress();
        const auto offset = address - base;
        if (const Cause usable = x2w_startup::gpu_range_cause(gpu, offset); usable != Cause::ok) return {usable, 0};
        return {Cause::ok, gpu + offset};
    };
    constexpr auto as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    const auto out = translate(dest, sizes.ResultDataMaxSizeInBytes, as_state);
    if (out.cause != Cause::ok) return refuse(out.cause, "dest", dest);
    // An update is sized by its own scratch requirement, a full build by the build one.
    // A device may report no update scratch at all (WARP does). Then a null scratch is fine, and a scratch the title
    // still passes must be a placed buffer, checked over at least one byte.
    const UINT64 scratch_bytes = update ? sizes.UpdateScratchDataSizeInBytes : sizes.ScratchDataSizeInBytes;
    const bool no_scratch = update && !scratch_bytes && !scratch;
    const auto temp = no_scratch ? Translated{Cause::ok, 0}
                                 : translate(scratch, (std::max<UINT64>)(scratch_bytes, 1), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (temp.cause != Cause::ok) return refuse(temp.cause, "scratch", scratch);
    Translated from{Cause::ok, 0};
    if (update) {
        from = translate(source, sizes.ResultDataMaxSizeInBytes, as_state);
        if (from.cause != Cause::ok) return refuse(x2w_startup::as_source_cause(from.cause), "source", source);
    }
    // The provenance of the source and the record of this build are one step under the lock: nothing is recorded for a refusal.
    AcquireSRWLockExclusive(&ledger_lock);
    if (update) {
        if (const Cause built = ledger.check_source(source); built != Cause::ok) {
            ReleaseSRWLockExclusive(&ledger_lock);
            return refuse(built, "source", source);
        }
    }
    // The structure written here keeps ALLOW_UPDATE only if this build asked for it (an update repeats the source's flags).
    ledger.record(dest, sizes.ResultDataMaxSizeInBytes,
                  (build.Inputs.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE) != 0);
    ReleaseSRWLockExclusive(&ledger_lock);
    build.DestAccelerationStructureData = out.gpu;
    build.ScratchAccelerationStructureData = temp.gpu;
    if (update) build.SourceAccelerationStructureData = from.gpu;
    if (reported) *reported = Cause::ok;
    return S_OK;
}

extern "C" __declspec(dllexport) BOOL x2w_d3d12_budget_snapshot(x2w_budget::Snapshot* out, SIZE_T bytes)
{
    if (!out || bytes != sizeof(*out) || !x2w_budget::enabled()) return FALSE;
    *out = x2w_budget::snapshot();
    return TRUE;
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
    log("x2w-d3d12 D3D12XboxCreateDevice adapter=%p params=[%s] riid=%s -> 0x%08lx build=%s", static_cast<void*>(adapter),
        block, text, static_cast<unsigned long>(hr), x2w_build_rev);
    if (FAILED(hr)) return hr;
    x2w_startup::attach_device(device, out);
    if (!*out) return E_OUTOFMEMORY;
    if (debug) device->QueryInterface(IID_PPV_ARGS(&info_queue));
    return S_OK;
}
