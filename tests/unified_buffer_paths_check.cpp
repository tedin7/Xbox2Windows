// Source-built comparison for #1914: one ordinary linear buffer of title-style memory, written by the
// CPU and read or written by the GPU, run through the same cases on several desktop paths. Test-only:
// nothing here changes the production backing of a resource.
//
// Title memory. XMem backs title pages with 64 KiB slots of 256 MiB pagefile-backed sections, maps an
// allocation's pages as views at the title's CPU address (placeholders replaced by views), and
// zero-fills a slot when it hands it out again (src/xmem_mapped_pages.cpp). The "section" pool here
// is one 256 MiB pagefile section of 4,096 slots. Each allocation is two slots, mapped as two page
// views on a placeholder split at the page boundary, so the CPU buffer is one run across two views.
// The "private" pool is one committed 1 MiB VirtualAlloc range of 16 slots, the only kind slot 133
// (CreatePlacedResourceX) aliases today. In both, slots 0-2 hold neighbouring title data and every
// case first runs a previous user (D) on slots 3-4 and frees it, so the buffer under test lands at
// offset 0x30000 on a reused slot, which must read zero before the title fills it. A freed slot
// stays retired until the D3D12 fence of its last submission (XMem owns slots, D3D12 the fence).
//
// Paths:
//   share     The pool is opened once as a heap (OpenExistingHeapFromAddress on a whole-pool view of the
//             section, or on the allocation base for the private range, after
//             D3D12_FEATURE_EXISTING_HEAPS) and a cross-adapter buffer is placed at the slot offset.
//             The GPU reads and writes title memory; nothing is copied.
//   copy      A default-heap buffer fed from a two-slot upload ring that checks each slot's fence:
//             a slot still in flight is refused (DXGI_ERROR_WAS_STILL_DRAWING). Title memory is copied into
//             the ring at one named copy point: bind (upload_from_title), record (frame_copy) or submit
//             (just before ExecuteCommandLists, the contract point). The ring slot is copied into the
//             buffer at the start of the submission; a use that declares GPU writes copies buffer ->
//             readback at its end, and completion of that fence copies readback -> title memory.
//   fallback  Sharing is forced to fail (the section through a FILE_MAP_READ handle, or an interior
//             address of the private range) and the actual HRESULT is recorded. The path falls back
//             to copy only when copy keeps the guarantees; a GPU-written use with no completion point
//             is refused with the sharing HRESULT.
//   slot133   A model of slot 133's fallback: a non-MEM_PRIVATE address is refused before D3D12
//             (E_INVALIDARG), then a committed default buffer stands in, "not aliased", with S_OK and
//             no copy. A private range is aliased through its allocation base. The oracle must catch a
//             stand-in: success with neither the title's bytes nor a refusal.
//
// Cases (distinct sentinels A, B, C, D, E, G, P, filler and neighbour per word):
//   a    bind with A in memory; after the command list is recorded and before ExecuteCommandLists the
//        CPU writes B. The GPU must read B (copy point: submit).
//   b    the GPU writes G ^ key; after the fence the CPU alias sees it and untouched words keep theirs.
//   c    X's submission is held in flight: the queue waits on a gate fence the CPU has not signalled.
//        The title frees X and allocates Y (C), whose submission is held too. On the copy paths a third
//        allocation then wraps the ring onto X's in-flight slot. X's slots must not go to Y, the wrap
//        must be refused, the held submissions read B and C, and after the fence X's slot and ring
//        slot are handed out again.
//   d1b  as a, copy point bind (upload_from_title): the GPU reads A in every word.
//   d1r  as a, copy point record (frame_copy): the GPU reads A in every word.
//   d2   as b, held on the gate: completion queried before the gate opens returns
//        DXGI_ERROR_WAS_STILL_DRAWING, and the CPU alias still holds P in every word (the read is
//        ordered before the GPU write by the queue's Wait). After the fence the alias holds G ^ key.
//   e    fallback only: a GPU-written use with no completion point is refused with the sharing
//        HRESULT; a GPU-read use without one falls back to copy.
// Probes: refusals the fallback relies on (asserted), the address path over a section view and over
// a run across a placeholder split, and a heap cached by base address after the allocation is freed
// and recommitted at the same base (reported, not asserted).
// Unsynchronized concurrent CPU/GPU writes are outside the contract and not exercised.
//
// --mutate skip-refresh (copy point bind) | skip-copy-back | reuse-early (no fence check on slots or
// ring) removes one guarantee. The run then exits 1 only when every case that guarantee protects
// fails, and 0 otherwise (negative controls, WILL_FAIL).
// X2W_TEST_DEBUG_LAYER=1 enables the D3D12 debug layer; an error outside the deliberate refusals fails.
// Returns 0 on pass, 1 on failure, 77 without the adapter or ID3D12Device3, or when sharing is
// unsupported (after the other paths have run).
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vm_translation_cs.h"
#include "xmem_test_device.h"

using Microsoft::WRL::ComPtr;
using namespace std::literals;

namespace {
constexpr UINT64 page = 64 * 1024;
constexpr unsigned section_pages = 4096;  // a 256 MiB XMem slab
constexpr unsigned private_pages = 16;
constexpr unsigned reserved_pages = 3;
constexpr unsigned buffer_pages = 2;
constexpr UINT64 buffer_bytes = buffer_pages * page;
constexpr unsigned ring_slots = 2;
constexpr UINT words = 1024;  // one dispatch of tests/vm_translation.hlsl: dst = src ^ key
constexpr UINT64 region_bytes = words * 4;
constexpr UINT64 r0 = page - region_bytes / 2;  // the region the GPU reads; straddles the two views
constexpr UINT64 r1 = page + 0x8000;            // the region the GPU writes
constexpr std::uint32_t key = 0x5a5a5a5au;

enum Tag { neighbour, filler, A, B, C, D, E, G, P, tag_count };
const char* const tag_names[tag_count] = {"neighbour", "filler", "A", "B", "C", "D", "E", "G", "P"};
constexpr std::uint32_t seeds[tag_count] = {0x6e656967u, 0x66696c6cu, 0xa1a1a1a1u, 0xb2b2b2b2u, 0xc3c3c3c3u,
                                            0xd4d4d4d4u, 0xe5e5e5e5u, 0x47474747u, 0x50505050u};
std::uint32_t pattern(Tag tag, UINT64 word)
{
    return seeds[tag] ^ static_cast<std::uint32_t>(word * 0x9e3779b1u);
}

std::string hex(HRESULT hr)
{
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08lx", static_cast<unsigned long>(hr));
    return text;
}

// Writes `count` words of `tag` at byte `offset`; the pattern index is the word's index in the buffer.
void write(std::uint32_t* cpu, UINT64 offset, Tag tag, UINT64 count = words)
{
    for (UINT64 i = 0; i < count; ++i) cpu[offset / 4 + i] = pattern(tag, offset / 4 + i);
}

struct Tally {
    unsigned count[tag_count]{};
    unsigned zero = 0, other = 0;
    std::string text() const
    {
        std::string out;
        for (int t = 0; t < tag_count; ++t) {
            if (!count[t]) continue;
            out += (out.empty() ? "" : " ") + std::string(tag_names[t]) + "=" + std::to_string(count[t]);
        }
        if (zero) out += (out.empty() ? "" : " ") + "zero="s + std::to_string(zero);
        if (other) out += (out.empty() ? "" : " ") + "other="s + std::to_string(other);
        return out + "/" + std::to_string(words);
    }
};

// Attributes each word to the sentinel it holds (after removing `mask`).
Tally tally(const std::uint32_t* got, UINT64 first_word, std::uint32_t mask)
{
    Tally t;
    for (UINT i = 0; i < words; ++i) {
        const std::uint32_t value = got[i] ^ mask;
        bool matched = false;
        for (int tag = 0; tag < tag_count && !matched; ++tag) {
            if (value == pattern(static_cast<Tag>(tag), first_word + i)) {
                ++t.count[tag];
                matched = true;
            }
        }
        if (!matched) ++(value == 0 ? t.zero : t.other);
    }
    return t;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 size, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

constexpr UINT64 in_use = ~UINT64{0};

// Title slots, handed out first-free. A released slot stays retired until `fence` has completed,
// unless reuse_early (the reuse-early mutant) frees it at once.
struct FencedSlots {
    std::vector<UINT64> state;
    bool reuse_early = false;
    explicit FencedSlots(unsigned count) : state(count, 0) {}
    int allocate(unsigned count, UINT64 completed)
    {
        for (unsigned first = 0; first + count <= state.size(); ++first) {
            bool free = true;
            for (unsigned i = 0; i < count && free; ++i) free = state[first + i] != in_use && state[first + i] <= completed;
            if (!free) continue;
            for (unsigned i = 0; i < count; ++i) state[first + i] = in_use;
            return static_cast<int>(first);
        }
        return -1;
    }
    void release(unsigned first, unsigned count, UINT64 fence)
    {
        for (unsigned i = 0; i < count; ++i) state[first + i] = reuse_early ? 0 : fence;
    }
};

// The modeled ring wraps only when a slot's submission has completed.
struct Ring {
    UINT64 state[ring_slots]{};
    unsigned cursor = 0, wraps = 0;
    bool reuse_early = false;
    int take(UINT64 completed)
    {
        const unsigned slot = cursor;
        if (!reuse_early && (state[slot] == in_use || state[slot] > completed)) return -1;
        state[slot] = in_use;
        cursor = (cursor + 1) % ring_slots;
        wraps += cursor == 0;
        return static_cast<int>(slot);
    }
    void retire(unsigned slot, UINT64 fence) { state[slot] = fence; }
};

// ---- Device, queue, fences and the debug layer ----

struct Gpu {
    ComPtr<ID3D12Device3> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> done, gate;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12InfoQueue> info;
    bool existing_heaps = false;
    UINT64 submitted = 0, gate_value = 0;
    bool gate_armed = false;
    std::vector<std::pair<UINT64, UINT64>> refusal_windows;  // debug-layer message ranges of refusals

    bool init()
    {
        D3D12_FEATURE_DATA_EXISTING_HEAPS heaps{};
        existing_heaps = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &heaps, sizeof(heaps))) &&
                         heaps.Supported;
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        D3D12_ROOT_PARAMETER params[2]{};
        for (UINT i = 0; i < 2; ++i) {
            params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            params[i].Descriptor.ShaderRegister = i;
        }
        const D3D12_ROOT_SIGNATURE_DESC root_desc{2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob;
        if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))) ||
            FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr)) ||
            FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                               IID_PPV_ARGS(&root)))) {
            return false;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get();
        pso.CS = {vm_translation_cs, sizeof(vm_translation_cs)};
        return SUCCEEDED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)));
    }

    UINT64 completed() const { return done->GetCompletedValue(); }

    bool wait(UINT64 value)
    {
        if (completed() >= value) return true;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool ok = event && SUCCEEDED(done->SetEventOnCompletion(value, event)) &&
                        WaitForSingleObject(event, 20000) == WAIT_OBJECT_0;
        if (event) CloseHandle(event);
        return ok;
    }

    // Later submissions wait on the gate until open_gate().
    void hold()
    {
        if (!gate_armed) ++gate_value;
        gate_armed = true;
    }
    void open_gate()
    {
        if (gate_armed) gate->Signal(gate_value);
        gate_armed = false;
    }
    void idle()
    {
        open_gate();
        if (SUCCEEDED(queue->Signal(done.Get(), ++submitted))) wait(submitted);
    }

    UINT64 stored() const { return info ? info->GetNumStoredMessages() : 0; }
    // Debug-layer errors outside the deliberate refusals, printed.
    unsigned unexpected_errors()
    {
        std::vector<int> refusal_ids;
        unsigned errors = 0;
        for (UINT64 i = 0; i < stored(); ++i) {
            bool refusal = false;
            for (const auto& [begin, end] : refusal_windows) refusal |= i >= begin && i < end;
            SIZE_T bytes = 0;
            if (FAILED(info->GetMessage(i, nullptr, &bytes))) continue;
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (FAILED(info->GetMessage(i, message, &bytes))) continue;
            if (message->Severity > D3D12_MESSAGE_SEVERITY_ERROR) continue;
            if (refusal) {
                // One line per message ID provoked on purpose.
                bool seen = false;
                for (const int id : refusal_ids) seen |= id == message->ID;
                if (!seen) {
                    refusal_ids.push_back(message->ID);
                    std::printf("debug layer (deliberate refusal) id %d: %.200s\n", message->ID, message->pDescription);
                }
                continue;
            }
            if (++errors <= 8) std::printf("debug layer ERROR id %d: %.300s\n", message->ID, message->pDescription);
        }
        return errors;
    }
};

// Debug-layer messages while a refusal is provoked on purpose are reported but do not fail the run.
struct Refusal {
    Gpu& gpu;
    UINT64 begin;
    explicit Refusal(Gpu& g) : gpu(g), begin(g.stored()) {}
    ~Refusal() { gpu.refusal_windows.emplace_back(begin, gpu.stored()); }
};

HRESULT make_buffer(Gpu& gpu, D3D12_HEAP_TYPE type, UINT64 bytes, D3D12_RESOURCE_FLAGS flags,
                    D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& out)
{
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    const auto desc = buffer_desc(bytes, flags);
    return gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&out));
}

const D3D12_RESOURCE_FLAGS placed_flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER |
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

// Records into one command list. Default-heap and placed buffers rest in COMMON between operations;
// upload and readback buffers stay in their fixed states.
struct Recorder {
    Gpu& gpu;
    ID3D12GraphicsCommandList* list;
    void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        list->ResourceBarrier(1, &barrier);
    }
    void copy(ID3D12Resource* dst, UINT64 dst_offset, bool dst_common, ID3D12Resource* src, UINT64 src_offset,
              bool src_common, UINT64 bytes)
    {
        if (dst_common) transition(dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        if (src_common) transition(src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(dst, dst_offset, src, src_offset, bytes);
        if (dst_common) transition(dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        if (src_common) transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    // dst[i] = src[i] ^ key for 1,024 words, addressed only by root UAV GPU addresses.
    void dispatch(ID3D12Resource* src, D3D12_GPU_VIRTUAL_ADDRESS src_va, ID3D12Resource* dst,
                  D3D12_GPU_VIRTUAL_ADDRESS dst_va)
    {
        transition(src, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootSignature(gpu.root.Get());
        list->SetPipelineState(gpu.pipeline.Get());
        list->SetComputeRootUnorderedAccessView(0, src_va);
        list->SetComputeRootUnorderedAccessView(1, dst_va);
        list->Dispatch(words / 64, 1, 1);
        transition(src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        transition(dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    }
};

// ---- Title memory ----

enum class PoolKind { section, private_range };
const char* name(PoolKind kind) { return kind == PoolKind::section ? "section" : "private"; }

struct TitlePool {
    PoolKind kind;
    unsigned pages;
    HANDLE section = nullptr;
    std::uint8_t* base = nullptr;
    FencedSlots slots;
    std::vector<bool> used;
    ComPtr<ID3D12Heap> heap;  // the shared heap, opened once per pool as slot 133 caches it
    std::vector<std::uint8_t*> runs;  // the section's mapped CPU runs
    std::uint8_t* pool_view = nullptr;  // the section pool's whole-pool view; the shared heap is opened on it

    TitlePool(PoolKind k, bool reuse_early)
        : kind(k), pages(k == PoolKind::section ? section_pages : private_pages), slots(pages), used(pages, false)
    {
        slots.reuse_early = reuse_early;
    }
    TitlePool(const TitlePool&) = delete;
    TitlePool& operator=(const TitlePool&) = delete;
    ~TitlePool()
    {
        heap.Reset();
        if (pool_view) UnmapViewOfFile(pool_view);
        while (!runs.empty()) unmap_run(runs.back());
        if (base) VirtualFree(base, 0, MEM_RELEASE);
        if (section) CloseHandle(section);
    }
    UINT64 bytes() const { return UINT64{pages} * page; }
    // OpenExistingHeapFromFileMapping leaks Windows commit (#2234), so heaps are opened on an address.
    void* heap_address() const { return kind == PoolKind::section ? static_cast<void*>(pool_view) : base; }

    bool init()
    {
        std::uint32_t* view = nullptr;
        if (kind == PoolKind::section) {
            section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                         static_cast<DWORD>(bytes()), nullptr);
            if (section) view = static_cast<std::uint32_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0,
                                                                         reserved_pages * page));
            if (view) pool_view = static_cast<std::uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, 0));
            if (view && !pool_view) {
                UnmapViewOfFile(view);
                view = nullptr;
            }
        } else {
            base = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes(), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            view = reinterpret_cast<std::uint32_t*>(base);
        }
        if (!view) return false;
        write(view, 0, neighbour, reserved_pages * page / 4);
        if (kind == PoolKind::section) UnmapViewOfFile(view);
        for (unsigned i = 0; i < reserved_pages; ++i) used[i] = true;
        return slots.allocate(reserved_pages, 0) == 0;
    }

    // Two page views of slots [slot, slot + 2) on a placeholder split at the page boundary.
    std::uint8_t* map_run(unsigned slot)
    {
        const HANDLE process = GetCurrentProcess();
        auto* run = static_cast<std::uint8_t*>(VirtualAlloc2(process, nullptr, buffer_bytes,
                                                             MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
                                                             nullptr, 0));
        if (!run) return nullptr;
        if (!VirtualFree(run, page, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            VirtualFree(run, 0, MEM_RELEASE);
            return nullptr;
        }
        unsigned mapped = 0;
        for (; mapped < buffer_pages; ++mapped) {
            auto* at = run + mapped * page;
            if (MapViewOfFile3(section, process, at, (UINT64{slot} + mapped) * page, page, MEM_REPLACE_PLACEHOLDER,
                               PAGE_READWRITE, nullptr, 0) != at) break;
        }
        runs.push_back(run);
        if (mapped == buffer_pages) return run;
        unmap_run(run);
        return nullptr;
    }

    void unmap_run(std::uint8_t* run)
    {
        for (auto it = runs.begin(); it != runs.end(); ++it) {
            if (*it != run) continue;
            runs.erase(it);
            for (unsigned i = 0; i < buffer_pages; ++i) {
                UnmapViewOfFile2(GetCurrentProcess(), run + i * page, MEM_PRESERVE_PLACEHOLDER);
                VirtualFree(run + i * page, 0, MEM_RELEASE);
            }
            return;
        }
    }
};

struct TitleAlloc {
    TitlePool* pool = nullptr;
    int slot = -1;
    std::uint32_t* cpu = nullptr;
    bool reused = false;
    unsigned nonzero = 0;  // words not zero when a reused slot is handed out
    UINT64 offset() const { return static_cast<UINT64>(slot) * page; }
};

// Hands out two adjacent slots. A reused slot is zero-filled as XMem does (through a separate view
// for the section) and must read zero through the title's run; then the title fills it.
bool allocate(TitlePool& pool, UINT64 completed, TitleAlloc& out)
{
    out = {};
    out.pool = &pool;
    out.slot = pool.slots.allocate(buffer_pages, completed);
    if (out.slot < 0) return false;
    out.reused = pool.used[out.slot] || pool.used[out.slot + 1];
    if (pool.kind == PoolKind::section) {
        if (out.reused) {
            void* zero = MapViewOfFile(pool.section, FILE_MAP_WRITE, 0, static_cast<DWORD>(out.offset()), buffer_bytes);
            if (!zero) return false;
            std::memset(zero, 0, buffer_bytes);
            UnmapViewOfFile(zero);
        }
        out.cpu = reinterpret_cast<std::uint32_t*>(pool.map_run(static_cast<unsigned>(out.slot)));
        if (!out.cpu) return false;
    } else {
        out.cpu = reinterpret_cast<std::uint32_t*>(pool.base + out.offset());
        if (out.reused) std::memset(out.cpu, 0, buffer_bytes);
    }
    for (UINT64 w = 0; w < buffer_bytes / 4; ++w) out.nonzero += out.cpu[w] != 0;
    pool.used[out.slot] = pool.used[out.slot + 1] = true;
    write(out.cpu, 0, filler, buffer_bytes / 4);
    return true;
}

// ---- The paths ----

enum class PathKind { share, copy, fallback, slot133 };
const char* name(PathKind path)
{
    switch (path) {
    case PathKind::share: return "share";
    case PathKind::copy: return "copy";
    case PathKind::fallback: return "fallback";
    default: return "slot133";
    }
}

enum class CopyPoint { bind, record, submit };
const char* name(CopyPoint point)
{
    return point == CopyPoint::bind ? "bind" : point == CopyPoint::record ? "record" : "submit";
}

struct Policy {
    CopyPoint copy_point = CopyPoint::submit;
    bool copy_back = true;
    bool reuse_early = false;
};

enum class Backing { none, shared, copied, stand_in };

struct Unified {
    TitleAlloc title;
    Backing backing = Backing::none;
    bool fallback = false;
    HRESULT share_hr = S_OK, bind_hr = S_OK;
    ComPtr<ID3D12Resource> gpu, readback;
    int staging = -1, last_staging = -1;
    UINT64 last_fence = 0, copy_back_fence = 0;

    std::string describe() const
    {
        char text[160];
        if (backing == Backing::shared) {
            std::snprintf(text, sizeof(text), "title %s memory (heap offset 0x%llx)", name(title.pool->kind),
                          static_cast<unsigned long long>(title.offset()));
        } else if (backing == Backing::stand_in) {
            std::snprintf(text, sizeof(text), "committed default buffer, not aliased, no copy");
        } else if (last_staging < 0) {
            std::snprintf(text, sizeof(text), "%sdefault buffer (not bound)", fallback ? "fallback " : "");
        } else {
            std::snprintf(text, sizeof(text), "%sdefault buffer + ring slot %d", fallback ? "fallback " : "",
                          last_staging);
        }
        return text;
    }
};

struct Runner {
    Gpu& gpu;
    Policy policy;
    UINT64 refresh_bytes = 0, gpu_copy_bytes = 0, copy_back_bytes = 0;
    ComPtr<ID3D12Resource> upload;
    std::uint8_t* ring_cpu = nullptr;
    Ring ring;
    std::vector<std::pair<UINT64, ComPtr<IUnknown>>> retired;  // released once their fence has completed

    Runner(Gpu& g, const Policy& p) : gpu(g), policy(p) { ring.reuse_early = p.reuse_early; }
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;
    ~Runner() { gpu.idle(); }

    bool init()
    {
        void* mapped = nullptr;
        const D3D12_RANGE none{};
        if (FAILED(make_buffer(gpu, D3D12_HEAP_TYPE_UPLOAD, ring_slots * buffer_bytes, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_GENERIC_READ, upload)) ||
            FAILED(upload->Map(0, &none, &mapped))) return false;
        ring_cpu = static_cast<std::uint8_t*>(mapped);
        return true;
    }

    void retire(UINT64 fence, IUnknown* object)
    {
        if (object) retired.emplace_back(fence, object);
    }
    void reclaim()
    {
        const UINT64 completed = gpu.completed();
        for (std::size_t i = 0; i < retired.size();) {
            if (retired[i].first <= completed) {
                retired.erase(retired.begin() + static_cast<std::ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
    }

    HRESULT share(TitlePool& pool, Unified& u)
    {
        if (!gpu.existing_heaps) return DXGI_ERROR_UNSUPPORTED;
        if (!pool.heap) {
            const HRESULT hr = gpu.device->OpenExistingHeapFromAddress(pool.heap_address(), IID_PPV_ARGS(&pool.heap));
            if (FAILED(hr)) return hr;
        }
        const auto desc = buffer_desc(buffer_bytes, placed_flags);
        return gpu.device->CreatePlacedResource(pool.heap.Get(), u.title.offset(), &desc, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&u.gpu));
    }

    HRESULT make_default(Unified& u, bool with_readback)
    {
        HRESULT hr = make_buffer(gpu, D3D12_HEAP_TYPE_DEFAULT, buffer_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_COMMON, u.gpu);
        if (SUCCEEDED(hr) && with_readback)
            hr = make_buffer(gpu, D3D12_HEAP_TYPE_READBACK, buffer_bytes, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_COPY_DEST, u.readback);
        return hr;
    }

    // `completion_point`: the caller observes GPU writes only after complete() for that submission.
    HRESULT open(TitlePool& pool, PathKind path, bool gpu_writes, bool completion_point, Unified& u)
    {
        if (path == PathKind::share) {
            u.share_hr = share(pool, u);
            if (SUCCEEDED(u.share_hr)) u.backing = Backing::shared;
            return u.share_hr;
        }
        if (path == PathKind::copy) {
            u.backing = Backing::copied;
            return make_default(u, true);
        }
        if (path == PathKind::slot133) {
            // Slot 133: only MEM_PRIVATE is aliased, through its allocation base; anything else gets a
            // committed default buffer that is not aliased, and S_OK.
            MEMORY_BASIC_INFORMATION info{};
            const bool queried = VirtualQuery(u.title.cpu, &info, sizeof(info)) == sizeof(info);
            u.share_hr = queried && info.Type == MEM_PRIVATE ? share(pool, u) : E_INVALIDARG;
            if (SUCCEEDED(u.share_hr)) {
                u.backing = Backing::shared;
                return S_OK;
            }
            u.backing = Backing::stand_in;
            return make_default(u, false);
        }
        {
            const Refusal refusal(gpu);
            ComPtr<ID3D12Heap> heap;
            if (pool.kind == PoolKind::section) {
                HANDLE read_only = nullptr;
                u.share_hr = DuplicateHandle(GetCurrentProcess(), pool.section, GetCurrentProcess(), &read_only,
                                             FILE_MAP_READ, FALSE, 0)
                                 ? gpu.device->OpenExistingHeapFromFileMapping(read_only, IID_PPV_ARGS(&heap))
                                 : HRESULT_FROM_WIN32(GetLastError());
                if (read_only) CloseHandle(read_only);
            } else {
                u.share_hr = gpu.device->OpenExistingHeapFromAddress(u.title.cpu, IID_PPV_ARGS(&heap));
            }
        }
        if (SUCCEEDED(u.share_hr)) return E_UNEXPECTED;  // sharing could not be forced to fail
        // A copy cannot deliver GPU writes to a caller that never reaches a completion point.
        if (gpu_writes && !completion_point) return u.share_hr;
        u.fallback = true;
        u.backing = Backing::copied;
        return make_default(u, true);
    }

    void snapshot(Unified& u)
    {
        std::memcpy(ring_cpu + u.staging * buffer_bytes, u.title.cpu, buffer_bytes);
        refresh_bytes += buffer_bytes;
    }

    // The copy path takes the ring's next slot; copy point bind copies title memory into it now.
    // Returns 0 with bind_hr = DXGI_ERROR_WAS_STILL_DRAWING when that slot is still in flight.
    D3D12_GPU_VIRTUAL_ADDRESS bind(Unified& u)
    {
        if (u.backing == Backing::copied && u.staging < 0) {
            u.staging = ring.take(gpu.completed());
            u.bind_hr = u.staging < 0 ? DXGI_ERROR_WAS_STILL_DRAWING : S_OK;
            if (u.staging < 0) return 0;
            u.last_staging = u.staging;
            if (policy.copy_point == CopyPoint::bind) snapshot(u);
        }
        return u.gpu->GetGPUVirtualAddress();
    }

    struct Use {
        Unified* u;
        bool gpu_writes;
    };

    // Record (copy point record; ring -> buffer) -> ops -> copy-back recording -> Close -> `between`
    // -> copy point submit -> optional gate wait -> ExecuteCommandLists -> Signal.
    template <class Ops, class Between>
    UINT64 submit(std::initializer_list<Use> uses, bool gated, Ops ops, Between between)
    {
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        if (FAILED(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
            FAILED(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                                 IID_PPV_ARGS(&list)))) return 0;
        Recorder rec{gpu, list.Get()};
        for (const Use& use : uses) {
            Unified& u = *use.u;
            if (u.backing != Backing::copied) continue;
            if (u.staging < 0 && !bind(u)) return 0;
            if (policy.copy_point == CopyPoint::record) snapshot(u);
            rec.copy(u.gpu.Get(), 0, true, upload.Get(), u.staging * buffer_bytes, false, buffer_bytes);
            gpu_copy_bytes += buffer_bytes;
        }
        ops(rec);
        for (const Use& use : uses) {
            Unified& u = *use.u;
            if (u.backing != Backing::copied || !use.gpu_writes) continue;
            rec.copy(u.readback.Get(), 0, false, u.gpu.Get(), 0, true, buffer_bytes);
            gpu_copy_bytes += buffer_bytes;
        }
        if (FAILED(list->Close())) return 0;
        between();
        for (const Use& use : uses) {
            if (use.u->backing == Backing::copied && policy.copy_point == CopyPoint::submit) snapshot(*use.u);
        }
        if (gated) {
            gpu.hold();
            if (FAILED(gpu.queue->Wait(gpu.gate.Get(), gpu.gate_value))) return 0;
        }
        ID3D12CommandList* lists[] = {list.Get()};
        gpu.queue->ExecuteCommandLists(1, lists);
        if (FAILED(gpu.queue->Signal(gpu.done.Get(), ++gpu.submitted))) return 0;
        const UINT64 fence = gpu.submitted;
        retire(fence, allocator.Get());
        retire(fence, list.Get());
        for (const Use& use : uses) {
            Unified& u = *use.u;
            u.last_fence = fence;
            if (u.backing != Backing::copied) continue;
            ring.retire(static_cast<unsigned>(u.staging), fence);
            u.staging = -1;
            if (use.gpu_writes) u.copy_back_fence = fence;
        }
        return fence;
    }
    template <class Ops>
    UINT64 submit(std::initializer_list<Use> uses, bool gated, Ops ops)
    {
        return submit(uses, gated, ops, [] {});
    }

    // DXGI_ERROR_WAS_STILL_DRAWING before the buffer's last submission has completed (without waiting);
    // otherwise S_OK after the copy path's copy-back into title memory.
    HRESULT complete(Unified& u, bool wait)
    {
        if (gpu.completed() < u.last_fence) {
            if (!wait) return DXGI_ERROR_WAS_STILL_DRAWING;
            if (!gpu.wait(u.last_fence)) return HRESULT_FROM_WIN32(WAIT_TIMEOUT);
        }
        if (u.copy_back_fence && u.copy_back_fence <= gpu.completed()) {
            if (policy.copy_back) {
                void* mapped = nullptr;
                const D3D12_RANGE read{0, buffer_bytes}, none{};
                const HRESULT hr = u.readback->Map(0, &read, &mapped);
                if (FAILED(hr)) return hr;
                std::memcpy(u.title.cpu, mapped, buffer_bytes);
                u.readback->Unmap(0, &none);
                copy_back_bytes += buffer_bytes;
            }
            u.copy_back_fence = 0;
        }
        reclaim();
        return S_OK;
    }

    // The title frees the buffer: its slots and desktop resources are retired to its last fence.
    void release(Unified& u)
    {
        if (u.title.pool->kind == PoolKind::section) u.title.pool->unmap_run(reinterpret_cast<std::uint8_t*>(u.title.cpu));
        u.title.cpu = nullptr;
        u.title.pool->slots.release(static_cast<unsigned>(u.title.slot), buffer_pages, u.last_fence);
        retire(u.last_fence, u.gpu.Get());
        retire(u.last_fence, u.readback.Get());
        u.gpu.Reset();
        u.readback.Reset();
        u.copy_back_fence = 0;
    }

    std::string bytes() const
    {
        return "bytes title->ring=" + std::to_string(refresh_bytes) + " gpu_copy=" + std::to_string(gpu_copy_bytes) +
               " readback->title=" + std::to_string(copy_back_bytes);
    }
};

// Path-independent GPU output: one default buffer with three regions, captured into a readback buffer.
struct Observer {
    ComPtr<ID3D12Resource> buffer, readback;
    static constexpr unsigned regions = 3;
    bool init(Gpu& gpu)
    {
        return SUCCEEDED(make_buffer(gpu, D3D12_HEAP_TYPE_DEFAULT, regions * region_bytes,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON, buffer)) &&
               SUCCEEDED(make_buffer(gpu, D3D12_HEAP_TYPE_READBACK, regions * region_bytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_COPY_DEST, readback));
    }
    D3D12_GPU_VIRTUAL_ADDRESS va(unsigned region) const { return buffer->GetGPUVirtualAddress() + region * region_bytes; }
    void capture(Recorder& rec) { rec.copy(readback.Get(), 0, false, buffer.Get(), 0, true, regions * region_bytes); }
    Tally read(unsigned region, UINT64 first_word)
    {
        void* mapped = nullptr;
        const D3D12_RANGE range{region * region_bytes, (region + 1) * region_bytes}, none{};
        if (FAILED(readback->Map(0, &range, &mapped))) return {};
        const Tally t = tally(static_cast<const std::uint32_t*>(mapped) + region * words, first_word, key);
        readback->Unmap(0, &none);
        return t;
    }
};

// ---- Cases ----

// caught: the slot-133 stand-in failed the oracle, as it must. missed: it passed.
enum class Outcome { pass, fail, skip, setup, caught, missed };
const char* name(Outcome outcome)
{
    switch (outcome) {
    case Outcome::pass: return "ok";
    case Outcome::fail: return "FAIL";
    case Outcome::skip: return "SKIP";
    case Outcome::setup: return "SETUP";
    case Outcome::caught: return "caught";
    default: return "MISSED";
    }
}

struct Cell {
    PoolKind kind;
    PathKind path;
    std::string id;
    Outcome outcome;
};
std::vector<Cell> cells;

struct Case {
    Gpu& gpu;
    PoolKind kind;
    PathKind path;
    Policy policy;
    std::string detail;

    Outcome refused_open(HRESULT hr)
    {
        detail += " open=" + hex(hr);
        if ((path == PathKind::share || path == PathKind::slot133) &&
            (hr == E_NOTIMPL || hr == DXGI_ERROR_UNSUPPORTED)) return Outcome::skip;
        return Outcome::fail;
    }
    static Outcome verdict(bool ok, const Unified& u)
    {
        if (u.backing == Backing::stand_in) return ok ? Outcome::missed : Outcome::caught;
        return ok ? Outcome::pass : Outcome::fail;
    }
    static std::string slot_text(const TitleAlloc& t)
    {
        return "slots " + std::to_string(t.slot) + "-" + std::to_string(t.slot + 1) + (t.reused ? " reused" : " fresh") +
               " nonzero-on-handout=" + std::to_string(t.nonzero);
    }

    // A previous user D of slots 3-4 through the same path, freed after its fence: the buffer under test
    // then lands on a reused slot (and, on the copy path, the ring's next slot).
    HRESULT previous_user(TitlePool& pool, Runner& run, Observer& obs)
    {
        Unified w;
        if (!allocate(pool, gpu.completed(), w.title)) return E_FAIL;
        write(w.title.cpu, r0, D);
        const HRESULT hr = run.open(pool, path, false, true, w);
        if (FAILED(hr)) return hr;
        const D3D12_GPU_VIRTUAL_ADDRESS va = run.bind(w);
        const UINT64 fence = run.submit({{&w, false}}, false, [&](Recorder& rec) {
            rec.dispatch(w.gpu.Get(), va + r0, obs.buffer.Get(), obs.va(2));
        });
        if (!fence || FAILED(run.complete(w, true))) return E_FAIL;
        run.release(w);
        return S_OK;
    }

    // a / d1b / d1r: bind with A; after recording and before ExecuteCommandLists write B. Case a holds
    // the copy path to the contract (B); d1b and d1r expect the stale A at the bind and record points.
    Outcome cpu_write_before_submit(CopyPoint point, bool contract)
    {
        TitlePool pool(kind, policy.reuse_early);
        Unified x;
        Observer obs;
        Policy local = policy;
        local.copy_point = point;
        Runner run(gpu, local);
        if (!pool.init() || !obs.init(gpu) || !run.init()) return Outcome::setup;
        HRESULT hr = previous_user(pool, run, obs);
        if (FAILED(hr)) return refused_open(hr);
        run.refresh_bytes = run.gpu_copy_bytes = 0;
        if (!allocate(pool, gpu.completed(), x.title)) return Outcome::setup;
        write(x.title.cpu, r0, A);
        hr = run.open(pool, path, false, true, x);
        if (FAILED(hr)) return refused_open(hr);
        const D3D12_GPU_VIRTUAL_ADDRESS va = run.bind(x);
        const UINT64 fence = run.submit({{&x, false}}, false,
                                        [&](Recorder& rec) {
                                            rec.dispatch(x.gpu.Get(), va + r0, obs.buffer.Get(), obs.va(0));
                                            obs.capture(rec);
                                        },
                                        [&] { write(x.title.cpu, r0, B); });
        hr = fence ? run.complete(x, true) : E_FAIL;
        const Tally t = obs.read(0, r0 / 4);
        detail = "share_hr=" + hex(x.share_hr) + " backing=" + x.describe() + " " + slot_text(x.title) +
                 (x.backing == Backing::copied ? " copy point="s + name(point) : ""s) + " complete=" + hex(hr) +
                 " GPU read " + t.text() + " " + run.bytes();
        const Tag expected = x.backing == Backing::copied && !contract ? A : B;
        return verdict(hr == S_OK && t.count[expected] == words && x.title.reused && x.title.nonzero == 0, x);
    }

    // b / d2: the GPU writes G ^ key into R1. d2 holds the submission and queries completion first.
    Outcome gpu_write(bool early_query)
    {
        TitlePool pool(kind, policy.reuse_early);
        Unified x;
        Observer obs;
        ComPtr<ID3D12Resource> input, input_upload;
        Runner run(gpu, policy);
        if (!pool.init() || !obs.init(gpu) || !run.init()) return Outcome::setup;
        // The GPU-side source: G in the word indices of R1, copied into a default buffer first.
        {
            void* mapped = nullptr;
            const D3D12_RANGE none{};
            if (FAILED(make_buffer(gpu, D3D12_HEAP_TYPE_DEFAULT, region_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COMMON, input)) ||
                FAILED(make_buffer(gpu, D3D12_HEAP_TYPE_UPLOAD, region_bytes, D3D12_RESOURCE_FLAG_NONE,
                                   D3D12_RESOURCE_STATE_GENERIC_READ, input_upload)) ||
                FAILED(input_upload->Map(0, &none, &mapped))) return Outcome::setup;
            for (UINT i = 0; i < words; ++i) static_cast<std::uint32_t*>(mapped)[i] = pattern(G, r1 / 4 + i);
            input_upload->Unmap(0, nullptr);
            const UINT64 fence = run.submit({}, false, [&](Recorder& rec) {
                rec.copy(input.Get(), 0, true, input_upload.Get(), 0, false, region_bytes);
            });
            if (!fence || !gpu.wait(fence)) return Outcome::setup;
        }
        HRESULT hr = previous_user(pool, run, obs);
        if (FAILED(hr)) return refused_open(hr);
        run.refresh_bytes = run.gpu_copy_bytes = 0;
        if (!allocate(pool, gpu.completed(), x.title)) return Outcome::setup;
        write(x.title.cpu, r0, A);
        write(x.title.cpu, r1, P);
        hr = run.open(pool, path, true, true, x);
        if (FAILED(hr)) return refused_open(hr);
        const D3D12_GPU_VIRTUAL_ADDRESS va = run.bind(x);
        const UINT64 fence = run.submit({{&x, true}}, early_query, [&](Recorder& rec) {
            rec.dispatch(input.Get(), input->GetGPUVirtualAddress(), x.gpu.Get(), va + r1);
        });
        if (!fence) return Outcome::fail;
        bool early_ok = true;
        std::string early_text;
        if (early_query) {
            // The queue waits on the gate before this submission, so the CPU read below is ordered
            // before the GPU write; the refusal is the oracle, the P words its consequence.
            const bool held = gpu.completed() < fence;
            const HRESULT early = run.complete(x, false);
            const Tally before = tally(x.title.cpu + r1 / 4, r1 / 4, 0);
            early_text = "held=" + std::to_string(held) + " early complete=" + hex(early) + " CPU alias before fence " +
                         before.text() + "; ";
            early_ok = held && early == DXGI_ERROR_WAS_STILL_DRAWING && before.count[P] == words;
            gpu.open_gate();
        }
        hr = run.complete(x, true);
        const Tally written = tally(x.title.cpu + r1 / 4, r1 / 4, key);
        const Tally kept = tally(x.title.cpu + r0 / 4, r0 / 4, 0);
        unsigned filler_kept = 0, filler_words = 0;
        for (UINT64 w = 0; w < buffer_bytes / 4; ++w) {
            if ((w >= r0 / 4 && w < r0 / 4 + words) || (w >= r1 / 4 && w < r1 / 4 + words)) continue;
            ++filler_words;
            filler_kept += x.title.cpu[w] == pattern(filler, w);
        }
        detail = "share_hr=" + hex(x.share_hr) + " backing=" + x.describe() + " " + slot_text(x.title) + " " +
                 early_text + "complete=" + hex(hr) + " CPU alias R1 " + written.text() + " (G^key)" +
                 (written.count[G] == words ? "" : " raw " + tally(x.title.cpu + r1 / 4, r1 / 4, 0).text()) +
                 " R0 " + kept.text() + " filler " + std::to_string(filler_kept) + "/" + std::to_string(filler_words) +
                 " " + run.bytes();
        return verdict(early_ok && hr == S_OK && written.count[G] == words && kept.count[A] == words &&
                           filler_kept == filler_words && x.title.reused && x.title.nonzero == 0,
                       x);
    }

    // c: hold X's submission in flight; free X and allocate Y while it is held; on the copy paths a
    // third allocation Z wraps the ring onto X's in-flight slot.
    Outcome in_flight()
    {
        TitlePool pool(kind, policy.reuse_early);
        Unified x, y, z;
        TitleAlloc after;
        Observer obs;
        Runner run(gpu, policy);
        if (!pool.init() || !obs.init(gpu) || !run.init()) return Outcome::setup;
        HRESULT hr = previous_user(pool, run, obs);
        if (FAILED(hr)) return refused_open(hr);
        run.refresh_bytes = run.gpu_copy_bytes = 0;
        if (!allocate(pool, gpu.completed(), x.title)) return Outcome::setup;
        write(x.title.cpu, r0, B);
        hr = run.open(pool, path, false, true, x);
        if (FAILED(hr)) return refused_open(hr);
        const D3D12_GPU_VIRTUAL_ADDRESS xa = run.bind(x);
        const UINT64 f1 = run.submit({{&x, false}}, true, [&](Recorder& rec) {
            rec.dispatch(x.gpu.Get(), xa + r0, obs.buffer.Get(), obs.va(0));
        });
        const int x_slot = x.title.slot, x_ring = x.last_staging;
        const bool copied = x.backing == Backing::copied;
        run.release(x);
        bool ok = f1 && allocate(pool, gpu.completed(), y.title);
        if (ok) {
            write(y.title.cpu, r0, C);
            ok = SUCCEEDED(run.open(pool, path, false, true, y));
        }
        const D3D12_GPU_VIRTUAL_ADDRESS ya = ok ? run.bind(y) : 0;
        const UINT64 f2 = ya ? run.submit({{&y, false}}, true, [&](Recorder& rec) {
            rec.dispatch(y.gpu.Get(), ya + r0, obs.buffer.Get(), obs.va(1));
            obs.capture(rec);
        }) : 0;
        // Copy paths: the ring's next slot is X's, still in flight.
        HRESULT wrap_hr = S_OK;
        bool z_submitted = false;
        if (copied && f2 && allocate(pool, gpu.completed(), z.title)) {
            write(z.title.cpu, r0, E);
            if (SUCCEEDED(run.open(pool, path, false, true, z))) {
                const D3D12_GPU_VIRTUAL_ADDRESS za = run.bind(z);
                wrap_hr = z.bind_hr;
                if (za) {  // only without the fence check
                    z_submitted = run.submit({{&z, false}}, true, [&](Recorder& rec) {
                        rec.dispatch(z.gpu.Get(), za + r0, obs.buffer.Get(), obs.va(2));
                    }) != 0;
                }
            }
        }
        const bool held = gpu.completed() < f1;
        const bool distinct = y.title.slot != x_slot;
        gpu.open_gate();
        hr = f2 ? run.complete(y, true) : E_FAIL;
        if (z_submitted) run.complete(z, true);
        const Tally first = obs.read(0, r0 / 4), second = obs.read(1, r0 / 4);
        int z_ring_after = -1;
        if (copied && !z_submitted && z.title.cpu && run.bind(z)) z_ring_after = z.last_staging;
        // After the fence, X's slots are handed out again, zero-filled.
        const bool reused = allocate(pool, gpu.completed(), after) && after.slot == x_slot && after.reused &&
                            after.nonzero == 0;
        char text[320];
        std::snprintf(text, sizeof(text),
                      "held=%d X %s ring %d; Y (while held) slots %d-%d ring %d; wrap onto X's ring slot hr=%s%s; "
                      "after fence %s, ring slot %d; ",
                      held, slot_text(x.title).c_str(), x_ring, y.title.slot, y.title.slot + 1, y.last_staging,
                      copied ? hex(wrap_hr).c_str() : "n/a", z_submitted ? " (taken)" : "", slot_text(after).c_str(),
                      z_ring_after);
        detail = "share_hr=" + hex(x.share_hr) + " backing=" + x.describe() + " " + text + "complete=" + hex(hr) +
                 " held submission read " + first.text() + " next " + second.text() + " " + run.bytes();
        const bool ring_ok = !copied || (wrap_hr == DXGI_ERROR_WAS_STILL_DRAWING && !z_submitted && z_ring_after == x_ring);
        return verdict(held && distinct && hr == S_OK && first.count[B] == words && second.count[C] == words &&
                           reused && ring_ok,
                       x);
    }

    // e (fallback): refusal for a GPU-written use without a completion point; a GPU-read one falls back.
    Outcome eligibility()
    {
        TitlePool pool(kind, policy.reuse_early);
        Unified written, read;
        Runner run(gpu, policy);
        if (!pool.init() || !run.init() || !allocate(pool, gpu.completed(), written.title) ||
            !allocate(pool, gpu.completed(), read.title)) return Outcome::setup;
        const HRESULT refused = run.open(pool, PathKind::fallback, true, false, written);
        const HRESULT fell_back = run.open(pool, PathKind::fallback, false, false, read);
        detail = "GPU-written, no completion point: open=" + hex(refused) + " share_hr=" + hex(written.share_hr) +
                 " resource=" + (written.gpu ? "created" : "none") + "; GPU-read, no completion point: open=" +
                 hex(fell_back) + " share_hr=" + hex(read.share_hr) + " backing=" + read.describe();
        return FAILED(refused) && refused == written.share_hr && !written.gpu && fell_back == S_OK && read.fallback
                   ? Outcome::pass
                   : Outcome::fail;
    }
};

void record(Gpu& gpu, PoolKind kind, PathKind path, const char* id, const Policy& policy, Outcome (*run)(Case&))
{
    Case c{gpu, kind, path, policy, {}};
    const Outcome outcome = run(c);
    std::printf("%-6s %-7s %-8s %-3s %s\n", name(outcome), name(kind), name(path), id, c.detail.c_str());
    cells.push_back({kind, path, id, outcome});
}

// ---- Probes ----

// Sharing refusals the fallback relies on, with their actual HRESULTs. Each must fail.
unsigned refusal_probes(Gpu& gpu)
{
    unsigned bad = 0;
    const auto probe = [&](const char* what, HRESULT hr) {
        std::printf("refusal %-60s hr=%s%s\n", what, hex(hr).c_str(), SUCCEEDED(hr) ? " (FAIL: accepted)" : "");
        bad += SUCCEEDED(hr);
    };
    TitlePool section(PoolKind::section, false), committed(PoolKind::private_range, false);
    TitleAlloc view, interior;
    if (!section.init() || !committed.init() || !allocate(section, 0, view) || !allocate(committed, 0, interior)) {
        std::puts("setup error: refusal probe pools");
        return 1;
    }
    {
        const Refusal refusal(gpu);
        ComPtr<ID3D12Heap> heap;
        probe("OpenExistingHeapFromAddress(interior of private range)",
              gpu.device->OpenExistingHeapFromAddress(interior.cpu, IID_PPV_ARGS(&heap)));
        heap.Reset();
        void* reserved = VirtualAlloc(nullptr, committed.bytes(), MEM_RESERVE, PAGE_READWRITE);
        probe("OpenExistingHeapFromAddress(reserve-only range)",
              reserved ? gpu.device->OpenExistingHeapFromAddress(reserved, IID_PPV_ARGS(&heap)) : E_OUTOFMEMORY);
        heap.Reset();
        if (reserved) VirtualFree(reserved, 0, MEM_RELEASE);
        HANDLE read_only = nullptr;
        if (DuplicateHandle(GetCurrentProcess(), section.section, GetCurrentProcess(), &read_only, FILE_MAP_READ, FALSE, 0)) {
            probe("OpenExistingHeapFromFileMapping(FILE_MAP_READ handle)",
                  gpu.device->OpenExistingHeapFromFileMapping(read_only, IID_PPV_ARGS(&heap)));
            heap.Reset();
            CloseHandle(read_only);
        }
    }
    const HRESULT open_hr = gpu.device->OpenExistingHeapFromAddress(section.heap_address(), IID_PPV_ARGS(&section.heap));
    if (SUCCEEDED(open_hr)) {
        const Refusal refusal(gpu);
        const auto desc = buffer_desc(buffer_bytes, placed_flags);
        ComPtr<ID3D12Resource> placed;
        probe("CreatePlacedResource(256 MiB section heap, offset 0x31000)",
              gpu.device->CreatePlacedResource(section.heap.Get(), view.offset() + 0x1000, &desc,
                                               D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&placed)));
        placed.Reset();
        probe("CreatePlacedResource(256 MiB section heap, 128 KiB at last slot)",
              gpu.device->CreatePlacedResource(section.heap.Get(), section.bytes() - page, &desc,
                                               D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&placed)));
    } else {
        std::printf("refusal placement probes not run: OpenExistingHeapFromAddress(section pool view) hr=%s\n", hex(open_hr).c_str());
    }
    return bad;
}

// Reported, not asserted. The address path over (1) one 128 KiB view of the 256 MiB section at
// offset 0x30000, which slot 133 rules out before calling D3D12 (MEM_MAPPED), and (2) the first view
// of a run split at the page boundary. Where a 128 KiB buffer can be placed: does it read the view's
// CPU writes, and do its GPU writes reach the section (seen through another view)?
void section_view_probe(Gpu& gpu)
{
    TitlePool pool(PoolKind::section, false);
    TitleAlloc run_alloc;
    Observer obs;
    Runner run(gpu, Policy{});
    if (!pool.init() || !allocate(pool, 0, run_alloc) || !obs.init(gpu) || !run.init()) {
        std::puts("section view probe: setup error");
        return;
    }
    {
        ComPtr<ID3D12Heap> heap;
        ComPtr<ID3D12Resource> placed;
        HRESULT open_hr = S_OK, place_hr = S_OK;
        {
            const Refusal refusal(gpu);
            open_hr = gpu.device->OpenExistingHeapFromAddress(run_alloc.cpu, IID_PPV_ARGS(&heap));
            const auto desc = buffer_desc(buffer_bytes, placed_flags);
            place_hr = SUCCEEDED(open_hr) ? gpu.device->CreatePlacedResource(heap.Get(), 0, &desc,
                                                                            D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                            IID_PPV_ARGS(&placed))
                                          : open_hr;
        }
        std::printf("finding: OpenExistingHeapFromAddress(first view of a run split at the page boundary) hr=%s, "
                    "heap size %llu, 128 KiB buffer across the split hr=%s\n",
                    hex(open_hr).c_str(), heap ? static_cast<unsigned long long>(heap->GetDesc().SizeInBytes) : 0ull,
                    hex(place_hr).c_str());
    }
    auto* view = static_cast<std::uint32_t*>(MapViewOfFile(pool.section, FILE_MAP_ALL_ACCESS, 0,
                                                           static_cast<DWORD>(run_alloc.offset()), buffer_bytes));
    if (!view) {
        std::puts("section view probe: MapViewOfFile failed");
        return;
    }
    ComPtr<ID3D12Heap> heap;
    ComPtr<ID3D12Resource> placed;
    HRESULT hr = S_OK, open_hr = S_OK;
    {
        const Refusal refusal(gpu);
        hr = open_hr = gpu.device->OpenExistingHeapFromAddress(view, IID_PPV_ARGS(&heap));
        const auto desc = buffer_desc(buffer_bytes, placed_flags);
        if (SUCCEEDED(hr)) hr = gpu.device->CreatePlacedResource(heap.Get(), 0, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                                 nullptr, IID_PPV_ARGS(&placed));
    }
    std::printf("finding: OpenExistingHeapFromAddress(128 KiB view of the 256 MiB section at +0x%llx) hr=%s, "
                "placed buffer hr=%s",
                static_cast<unsigned long long>(run_alloc.offset()), hex(open_hr).c_str(), hex(hr).c_str());
    if (SUCCEEDED(hr)) {
        write(view, r0, B);
        const D3D12_GPU_VIRTUAL_ADDRESS va = placed->GetGPUVirtualAddress();
        const UINT64 fence = run.submit({}, false, [&](Recorder& rec) {
            rec.dispatch(placed.Get(), va + r0, obs.buffer.Get(), obs.va(0));  // read: B ^ key
            rec.dispatch(obs.buffer.Get(), obs.va(0), placed.Get(), va + r1);  // write back: B at R1
            obs.capture(rec);
        });
        const bool done = fence && gpu.wait(fence);
        const Tally read = obs.read(0, r0 / 4);
        const Tally seen = tally(run_alloc.cpu + r1 / 4, r0 / 4, 0);  // through the split run's views
        std::printf("; %s; GPU read of CPU writes %s; GPU write seen through the split run %s",
                    done ? "completed" : "did not complete", read.text().c_str(), seen.text().c_str());
    }
    std::puts("");
    gpu.idle();
    placed.Reset();
    heap.Reset();
    UnmapViewOfFile(view);
}

// Reported, except that a fresh heap must read the new bytes. Slot 133 caches heaps by allocation base
// and never drops them. A private range is opened, read through (D), freed and recommitted at the
// same base (B): what does a buffer placed on the cached heap read?
unsigned stale_heap_probe(Gpu& gpu)
{
    if (!gpu.existing_heaps) {
        std::puts("finding: stale heap probe not run (no existing heaps)");
        return 0;
    }
    constexpr UINT64 bytes = private_pages * page, offset = reserved_pages * page;
    Observer obs;
    Runner run(gpu, Policy{});
    if (!obs.init(gpu) || !run.init()) return 1;
    auto* base = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!base) return 1;
    ComPtr<ID3D12Heap> cached, fresh;
    ComPtr<ID3D12Resource> first, second, third;
    const auto desc = buffer_desc(buffer_bytes, placed_flags);
    const auto read_through = [&](ID3D12Resource* buffer, unsigned region) {
        const D3D12_GPU_VIRTUAL_ADDRESS va = buffer->GetGPUVirtualAddress();
        const UINT64 fence = run.submit({}, false, [&](Recorder& rec) {
            rec.dispatch(buffer, va + r0, obs.buffer.Get(), obs.va(region));
            obs.capture(rec);
        });
        return fence && gpu.wait(fence) ? obs.read(region, r0 / 4) : Tally{};
    };
    write(reinterpret_cast<std::uint32_t*>(base + offset), r0, D);
    HRESULT hr = gpu.device->OpenExistingHeapFromAddress(base, IID_PPV_ARGS(&cached));
    if (SUCCEEDED(hr)) hr = gpu.device->CreatePlacedResource(cached.Get(), offset, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                             nullptr, IID_PPV_ARGS(&first));
    if (FAILED(hr)) {
        std::printf("finding: stale heap probe not run: hr=%s\n", hex(hr).c_str());
        VirtualFree(base, 0, MEM_RELEASE);
        return 0;
    }
    const Tally before = read_through(first.Get(), 0);
    first.Reset();
    gpu.idle();
    // Release first; if the heap keeps that from succeeding, decommit and recommit in place instead.
    const BOOL freed = VirtualFree(base, 0, MEM_RELEASE);
    const DWORD free_error = freed ? 0 : GetLastError();
    BOOL decommitted = FALSE;
    DWORD decommit_error = 0;
    std::uint8_t* again = nullptr;
    if (freed) {
        again = static_cast<std::uint8_t*>(VirtualAlloc(base, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    } else {
        decommitted = VirtualFree(base, bytes, MEM_DECOMMIT);
        decommit_error = decommitted ? 0 : GetLastError();
        if (decommitted) again = static_cast<std::uint8_t*>(VirtualAlloc(base, bytes, MEM_COMMIT, PAGE_READWRITE));
    }
    std::printf("finding: cached heap by base: first read %s; with the heap alive VirtualFree(MEM_RELEASE)=%d "
                "(error %lu), MEM_DECOMMIT=%d (error %lu), committed again at the same base=%d",
                before.text().c_str(), freed, free_error, decommitted, decommit_error, again == base);
    unsigned bad = 0;
    if (again == base) {
        write(reinterpret_cast<std::uint32_t*>(base + offset), r0, B);
        hr = gpu.device->CreatePlacedResource(cached.Get(), offset, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                              IID_PPV_ARGS(&second));
        const Tally stale = SUCCEEDED(hr) ? read_through(second.Get(), 1) : Tally{};
        std::printf("; buffer on the cached heap hr=%s reads %s", hex(hr).c_str(), stale.text().c_str());
        hr = gpu.device->OpenExistingHeapFromAddress(base, IID_PPV_ARGS(&fresh));
        if (SUCCEEDED(hr)) hr = gpu.device->CreatePlacedResource(fresh.Get(), offset, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                                 nullptr, IID_PPV_ARGS(&third));
        const Tally current = SUCCEEDED(hr) ? read_through(third.Get(), 2) : Tally{};
        std::printf("; on a fresh heap hr=%s reads %s", hex(hr).c_str(), current.text().c_str());
        bad = current.count[B] != words;
    }
    std::puts(bad ? " (FAIL: the fresh heap does not read the new bytes)" : "");
    gpu.idle();
    second.Reset();
    third.Reset();
    cached.Reset();
    fresh.Reset();
    if (!freed) std::printf("finding: after the cached heap is released, VirtualFree(MEM_RELEASE)=%d\n", VirtualFree(base, 0, MEM_RELEASE));
    else if (again) VirtualFree(again, 0, MEM_RELEASE);
    return bad;
}

void print_environment(ID3D12Device* device, bool debug_layer, HRESULT debug_hr, bool existing_heaps)
{
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    OSVERSIONINFOW os{sizeof(os)};
    if (auto get = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion")))
        get(&os);
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 desc{};
    LARGE_INTEGER umd{};
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) &&
        SUCCEEDED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter)))) {
        adapter->GetDesc1(&desc);
        adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd);
    }
    const std::string layer = debug_layer ? "on" : debug_hr == S_FALSE ? "off (X2W_TEST_DEBUG_LAYER unset)"
                                                                      : "unavailable (" + hex(debug_hr) + ")";
    std::printf("adapter %ls (%04x:%04x) driver %u.%u.%u.%u, Windows %lu.%lu.%lu, debug layer %s, "
                "D3D12_FEATURE_EXISTING_HEAPS %d\n",
                desc.Description, desc.VendorId, desc.DeviceId, HIWORD(umd.HighPart), LOWORD(umd.HighPart),
                HIWORD(umd.LowPart), LOWORD(umd.LowPart), os.dwMajorVersion, os.dwMinorVersion, os.dwBuildNumber,
                layer.c_str(), existing_heaps);
}

int run(bool warp, std::string_view mutant)
{
    Policy policy;
    if (mutant == "skip-refresh"sv) policy.copy_point = CopyPoint::bind;
    if (mutant == "skip-copy-back"sv) policy.copy_back = false;
    if (mutant == "reuse-early"sv) policy.reuse_early = true;
    const bool mutated = !mutant.empty();

    bool debug_layer = false;
    HRESULT debug_hr = S_FALSE;  // not requested
    if (GetEnvironmentVariableW(L"X2W_TEST_DEBUG_LAYER", nullptr, 0)) {
        ComPtr<ID3D12Debug> debug;
        debug_hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
        if (SUCCEEDED(debug_hr) && debug) {
            debug->EnableDebugLayer();
            debug_layer = true;
        }
    }
    bool unmatched = false;
    const auto base = create_device(warp, unmatched);
    if (unmatched) return 1;
    Gpu gpu;
    if (!base || FAILED(base.As(&gpu.device))) {
        std::puts(warp ? "No ID3D12Device3 on WARP" : "No hardware adapter with ID3D12Device3");
        return 77;
    }
    if (debug_layer && SUCCEEDED(gpu.device.As(&gpu.info))) gpu.info->SetMessageCountLimit(~UINT64{0});
    if (!gpu.init()) {
        std::puts("setup error: queue, fences or compute pipeline");
        return mutated ? 77 : 1;
    }
    print_environment(gpu.device.Get(), debug_layer, debug_hr, gpu.existing_heaps);
    if (mutated) std::printf("mutant: %.*s\n", static_cast<int>(mutant.size()), mutant.data());

    unsigned probe_failures = refusal_probes(gpu);
    section_view_probe(gpu);
    probe_failures += stale_heap_probe(gpu);
    for (const PoolKind kind : {PoolKind::section, PoolKind::private_range}) {
        for (const PathKind path : {PathKind::share, PathKind::copy, PathKind::fallback, PathKind::slot133}) {
            const auto with = [&](const char* id, Outcome (*fn)(Case&)) { record(gpu, kind, path, id, policy, fn); };
            with("a", +[](Case& c) { return c.cpu_write_before_submit(c.policy.copy_point, true); });
            with("b", +[](Case& c) { return c.gpu_write(false); });
            if (path == PathKind::slot133) continue;
            with("c", +[](Case& c) { return c.in_flight(); });
            with("d1b", +[](Case& c) { return c.cpu_write_before_submit(CopyPoint::bind, false); });
            with("d1r", +[](Case& c) { return c.cpu_write_before_submit(CopyPoint::record, false); });
            with("d2", +[](Case& c) { return c.gpu_write(true); });
            if (path == PathKind::fallback) with("e", +[](Case& c) { return c.eligibility(); });
        }
    }
    gpu.idle();
    const HRESULT removed = gpu.device->GetDeviceRemovedReason();
    const unsigned debug_errors = debug_layer ? gpu.unexpected_errors() : 0;
    unsigned counts[6]{};
    for (const Cell& cell : cells) ++counts[static_cast<int>(cell.outcome)];
    std::printf("cases: %u passed, %u failed, %u skipped, %u setup errors, %u stand-ins caught, %u missed; probe "
                "failures=%u; device removed reason=%s; debug-layer errors outside refusals=%u\n",
                counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], probe_failures, hex(removed).c_str(),
                debug_errors);

    if (mutated) {
        // Every case the removed guarantee protects must fail.
        unsigned targeted = 0, caught = 0, not_run = 0;
        for (const Cell& cell : cells) {
            const bool copy_backed = cell.path == PathKind::copy || cell.path == PathKind::fallback;
            const bool target =
                (mutant == "skip-refresh"sv && copy_backed && cell.id == "a") ||
                (mutant == "skip-copy-back"sv && copy_backed && (cell.id == "b" || cell.id == "d2")) ||
                (mutant == "reuse-early"sv && cell.path != PathKind::slot133 && cell.id == "c");
            if (!target) continue;
            ++targeted;
            caught += cell.outcome == Outcome::fail;
            not_run += cell.outcome == Outcome::skip || cell.outcome == Outcome::setup;
            if (cell.outcome != Outcome::fail)
                std::printf("mutant not caught by %s %s %s\n", name(cell.kind), name(cell.path), cell.id.c_str());
        }
        std::printf("mutant %.*s: %u of %u targeted cases fail\n", static_cast<int>(mutant.size()), mutant.data(),
                    caught, targeted);
        if (not_run) return 77;
        return targeted && caught == targeted ? 1 : 0;
    }
    if (counts[1] || counts[3] || counts[5] || probe_failures || FAILED(removed) || debug_errors) return 1;
    return counts[2] ? 77 : 0;
}
}  // namespace

int main(int argc, char** argv)
{
    std::string_view mutant;
    if (argc == 4 && argv[1] == "--mutate"sv) {
        mutant = argv[2];
        if (mutant != "skip-refresh"sv && mutant != "skip-copy-back"sv && mutant != "reuse-early"sv) return 2;
    } else if (argc != 2) {
        std::fputs("usage: unified_buffer_paths_check [--mutate skip-refresh|skip-copy-back|reuse-early] "
                   "warp|hardware\n", stderr);
        return 2;
    }
    const std::string_view adapter = argv[argc - 1];
    if (adapter != "warp"sv && adapter != "hardware"sv) return 2;
    const int result = run(adapter == "warp"sv, mutant);
    std::printf("unified_buffer_paths %s%s%.*s: %s\n", adapter.data(), mutant.empty() ? "" : " --mutate ",
                static_cast<int>(mutant.size()), mutant.data(),
                result == 0 ? "PASS" : result == 77 ? "SKIP" : "FAIL");
    return result;
}
