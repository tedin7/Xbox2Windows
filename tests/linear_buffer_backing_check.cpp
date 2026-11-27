// Production slice of #1914 (Refs #874): one ordinary linear buffer, written by the CPU and read by
// the GPU, placed at a title address through the device wrapper's slot 133 (CreatePlacedResourceX)
// and submitted through the command-queue wrapper's ExecuteCommandLists, in d3d12_xs.dll. The cases
// and sentinels are those of tests/unified_buffer_paths_check.cpp (#1942); here they run against the
// production backing instead of a model of it.
//
//   linear_buffer_backing_check <warp|hardware> <path> [--mutate <name>]
//
// path selects the backing (the process environment is set here, before d3d12_xs.dll reads it):
//   default    X2W_D3D12_LINEAR_BUFFERS unset: slot 133 as before (the old behaviour, asserted)
//   share      X2W_D3D12_LINEAR_BUFFERS=1: real sharing where the adapter, extent and memory allow it
//              (a private range); an XMem view is refused for sharing by the host rule and copied
//   copy       X2W_D3D12_LINEAR_BUFFERS=copy: forced copying at submit
//   refusal    X2W_D3D12_LINEAR_BUFFERS=1 with X2W_TEST_EXISTING_HEAPS_UNSUPPORTED=1: the host refuses
//              sharing (as an adapter reporting D3D12_FEATURE_EXISTING_HEAPS = 0 does)
//   untracked  X2W_D3D12_LINEAR_BUFFERS=copy with X2W_TEST_SUBMIT_FENCE_FAIL=1: no submit fence, so
//              the copy cannot keep its lifetime guarantee: slot 133 keeps the default backing
//   stale      X2W_D3D12_LINEAR_BUFFERS=copy with X2W_TEST_LINEAR_REFRESH_FAIL=3: staging requests 2,
//              5, 8 ... fail, so one refresh per pool cannot be staged
//   contention X2W_D3D12_LINEAR_BUFFERS=copy with X2W_TEST_LINEAR_STAGING_LIMIT=1: one staging buffer,
//              so a refresh waits for the fence of the submission that holds it
//   lease      X2W_D3D12_LINEAR_BUFFERS=copy with X2W_MESH86_XMEM_SRV=1: the mesh XMem lease path is on and
//              owns XMem placements with console flags 0; the linear backing must defer to it there
//
// Where the new path cannot keep its guarantees (a GPU-written use that is not shared, no submit
// fence), slot 133 must keep the default backing and return what it returns with the switch off:
// S_OK, an XMem view not aliased, a private range aliased (on an adapter with existing heaps). The
// refusal must be reported in x2w_d3d12_linear_buffer_info (path 3, refusal_hr), in the totals
// (kept_default) and in the slot 133 log line ("kept default backing: linear refusal 0x..."), which
// the fixture reads back from X2W_TRAP_LOG (linear_buffer_backing_<path>_<adapter>.log next to the
// executable).
//
// Title memory comes from two pools. "xmem": xmem.dll physical pages (64 KiB slots of 256 MiB
// pagefile sections) mapped with X2W_XMEM_SHARED_VIEWS=1, five pages per mapping call (one view):
// three neighbour pages, then the two-page buffer at view offset 0x30000. A previous user's pages are
// freed first, so the first buffer lands on reused slots that must read zero. "private": one
// committed 8 MiB VirtualAlloc range, five pages per allocation in the same layout.
//
// Cases (sentinels A, B, C, D, E, G, P, filler and neighbour differ in every word):
//   a    bind A: slot 133 and a list that reads the buffer are recorded over A. The CPU writes B
//        before ExecuteCommandLists. The GPU must read B.
//   a2   then 16 words change to E: the next submission reads them, and a copy moves only those 64 bytes.
//   b    a GPU-written use (ALLOW_UNORDERED_ACCESS): the GPU writes G ^ key; after the fence the CPU
//        alias holds it and the words around keep theirs. The copy cannot give this: slot 133 keeps the
//        default backing and reports the sharing HRESULT (DXGI_ERROR_UNSUPPORTED when not attempted).
//   c    a submission is held in flight on a gate fence, then two more (other buffers, C and D). Each
//        held submission reads its own sentinel. Sharing: the title releases X while it is held, and
//        after the fence places a new buffer (E) at X's address, on the same cached heap generation.
//        Copying: no staging buffer is handed out again while its submission is held; after the fence
//        the next refresh takes one of them again.
//   d1   missing refresh: after a refreshed submission of A the CPU writes B, and the list is
//        executed on the desktop queue directly, past the wrapper. Copying must expose the stale A
//        (the oracle sees a missing refresh); sharing reads B. Through the wrapper the copy reads B.
//   d2   readback before the fence: the GPU write of b is held on the gate. Before the gate opens the
//        fence has not completed and the alias still holds P in every word (the read is ordered
//        before the write by the queue's Wait); after it, G ^ key. Copying keeps the default (as b).
//   stale (stale only) a refreshed submission of A; the CPU writes B and the next refresh fails: the
//        submission still runs and completes, reads A (the last upload), and the buffer is reported
//        stale with E_OUTOFMEMORY in info, totals and the log; the next submission reads B and clears it.
//   split (share only) a buffer across two XMem views: sharing is refused with E_INVALIDARG, decided
//        before any D3D12 call, and the buffer is copied; case a must still read B.
//   gate (refusal only) no heap was opened: D3D12_FEATURE_EXISTING_HEAPS gates sharing before any open.
//   q.comp, q.copy (copy only) the buffer is read by a list on a COMPUTE queue (a dispatch) and on a
//        COPY queue (CopyBufferRegion) submitted through those queues' wrappers: the refresh runs on
//        that queue, the first submission reads B, and a 16-word change then copies 64 bytes.
//   gate2q, gate2q.held (copy only) one buffer holding A, read by a list on queue 1 behind a title gate
//        and by a list on queue 2; the CPU writes B once before both submissions. Queue 2 must complete
//        while the gate is closed and read B, and queue 1 must read B after it. In .held the fixture also
//        holds the internal refresh queue: queue 2 must not complete before the refresh lands.
//   xq   (copy only) queue 1 waits on a title fence that queue 2 signals after its list (a cross-queue
//        dependency the title owns); both read the buffer the CPU changed to B. Both must complete (no
//        deadlock) and read B.
//   contention (contention only) a second queue, a release() and the info export while a submission
//        waits for the only staging buffer, whose copy the fixture holds on the internal refresh
//        queue: all three must finish while the wait is still under way. After the refreshes land the
//        waiting refresh has taken that buffer again (none is created). The gated S1 reads B: C's copy
//        waits for it (the producer wait). S2 reads C.
//   budget (share, copy) with X2W_D3D12_BUDGET=1: a shared buffer counts as placed memory (its heap
//        live), a copied one as a committed buffer; live while held, retained after the title's release.
//   gpuw (copy only) a recorded copy (list slot 15) writes G into a copied buffer, then the title changes
//        its first and last words: the buffer keeps G (it is no longer refreshed), reported as gpu_written.
//   gpuw.discard (copy only) the same copy is recorded and closed, then the list is reset without a
//        submission; the CPU writes B: the next read sees B and the buffer is not gpu_written.
//   war  (copy only) a read of A on queue 1 behind a title gate (on the desktop queue); the CPU writes B and
//        queue 2 submits a read. Its refresh's copy waits for queue 1 (producer_waits): queue 2 cannot
//        complete while the gate is closed, queue 1 reads A and queue 2 reads B.
//   xq.write (copy only) queue 1 waits through its wrapper (slot 17) on a title fence, then submits a read of
//        A; the CPU writes B and queue 2 submits a read. Its refresh has no safe completion order (it would
//        have to wait for queue 1, behind the unreached fence), so the submission is refused: unsafe refresh
//        refusals +1 and logged, its result slot keeps P. The title's Signal through queue 2's wrapper (slot
//        16) releases queue 1, which reads A; the refused recording submitted again on queue 2 reads B.
//   xq.race (copy only) xq.write with queue 1's title Wait and its read of A on two threads: the Wait is parked
//        through x2w_d3d12_linear_buffer_test_pause (point 3) after the desktop Wait and before it is recorded,
//        and the read is submitted meanwhile. The read is behind the fence on the desktop queue, so queue 2's
//        read must still be refused, queue 2 must still signal the fence, queue 1 must read A and the
//        resubmission B. A queue that does not complete within 3 s is released by a CPU Signal of the fence.
//   order (copy only) two threads refresh one buffer, parked through x2w_d3d12_linear_buffer_test_pause:
//        thread A stages C and parks before its copy executes; the CPU writes D; thread B stages D, executes
//        first and parks before publishing; A executes (the backing ends with C) and publishes, B publishes
//        last. The next submission must refresh D (reordered_publishes +1) and read it.
//   retire (copy only) a submission that reads Y snapshots the changed X and Y and parks before staging;
//        the title releases X and frees its pages. X is skipped as unreadable (no copy fault, no failed
//        refresh); Y reads C.
//   lease (lease only) an XMem buffer keeps the default backing with HRESULT_FROM_WIN32(ERROR_BUSY)
//        reported (old behaviour: reads zero); a private one is copied and reads B.
//   pin  (share, XMem; measured, not asserted) a heap opened from an XMem view and a buffer placed on
//        it, then XMem's unmap of those pages, while they live and after their release.
//
// The default path asserts what slot 133 did before: a private range is aliased (a and b pass) and
// an XMem view gets a committed buffer that does not alias it (a reads zero, b leaves P), with no
// linear-buffer bookkeeping at all.
//
// --mutate <name> sets X2W_TEST_LINEAR_BUFFER_MUTANT, which removes one guarantee inside d3d12_xs.dll:
//   skip-refresh (copy) case a, reuse-early (copy) case c, copy-gpu-writes (copy) case b,
//   ignore-feature (refusal) case gate, refusal-silent (copy) case b (the kept default is not reported),
//   hold-lock (contention) case contention (the wait keeps the backing's lock held),
//   skip-consumer-wait (copy) case gate2q.held (no submission waits for the internal fence),
//   refresh-gpu-written (copy) case gpuw (a refresh spanning the written words copies the title's A over G),
//   gpuw-at-record (copy) case gpuw.discard (the buffer stops being refreshed when the copy is recorded),
//   skip-producer-wait (copy) case war (the refresh's copy lands before queue 1's earlier read),
//   publish-out-of-order (copy) case order (B's older publish is applied after A's: no further refresh),
//   copy-after-retire (copy) case retire (no probe before staging: the fault guard catches the read),
//   accept-unsafe-refresh (copy) case xq.write (the refresh skips the blocked queue: queue 2 runs and both read B),
//   unlocked-title-wait (copy) case xq.race (slot 17 takes no order lock: the read takes its user value before the
//   Wait is recorded, queue 2's refresh is accepted and waits for queue 1, and both stay pending until the CPU
//   signals the fence).
//
// Refreshes run on an internal COPY queue that the DLL exports to fixtures
// (x2w_d3d12_linear_buffer_internal_queue); cases c, contention and gate2q.held hold it behind a fence
// of their own to keep a refresh in flight. Cases order and retire park a refresh on its thread through
// x2w_d3d12_linear_buffer_test_pause, and xq.race parks a title Wait.
// The run then exits 1 only if every targeted case failed, else 0 (negative controls, WILL_FAIL).
// X2W_TEST_DEBUG_LAYER=1 enables the D3D12 debug layer; any stored error fails the run.
// Returns 0 on pass, 1 on failure, 77 without the adapter, or on the share path when the adapter
// reports no existing-heap support.
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "xmem_test_device.h"
#include "../src/startup_stubs/linear_buffer_report.h"
#include "../src/startup_stubs/resource_budget.h"

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) bool x2w_d3d12_linear_buffer_info(void*, X2wLinearBufferInfo*);
extern "C" __declspec(dllimport) void x2w_d3d12_linear_buffer_totals(X2wLinearBufferTotals*);
extern "C" __declspec(dllimport) bool x2w_d3d12_linear_buffer_internal_queue(void**);
extern "C" __declspec(dllimport) bool x2w_d3d12_linear_buffer_test_pause(unsigned, void*, void*);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_budget_snapshot(x2w_budget::Snapshot*, SIZE_T);
extern "C" __declspec(dllimport) void* WINAPI XMemVirtualAlloc(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
extern "C" __declspec(dllimport) BOOL WINAPI XMemAllocatePhysicalPages(ULONG_PTR, PULONG_PTR, PULONG_PTR);
extern "C" __declspec(dllimport) BOOL WINAPI XMemFreePhysicalPages(ULONG_PTR, PULONG_PTR);
extern "C" __declspec(dllimport) void* WINAPI XMemMapPhysicalPages(void*, ULONG_PTR, const ULONG_PTR*);

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT64 page = 64 * 1024;
constexpr unsigned alloc_pages = 5;       // three neighbour pages, then the buffer
constexpr UINT64 buffer_offset = 3 * page;
constexpr UINT64 buffer_bytes = 2 * page;
constexpr UINT words = 1024;
constexpr UINT64 region_bytes = words * 4;
constexpr UINT64 r0 = page - region_bytes / 2;  // the region the GPU reads; straddles the buffer's two pages
constexpr UINT64 r1 = page + 0x8000;            // the region the GPU writes
constexpr std::uint32_t key = 0x5a5a5a5au;

enum Tag { neighbour, filler, A, B, C, D, E, G, P, tag_count };
const char* const tag_names[tag_count] = {"neighbour", "filler", "A", "B", "C", "D", "E", "G", "P"};
constexpr std::uint32_t seeds[tag_count] = {0x6e656967u, 0x66696c6cu, 0xa1a1a1a1u, 0xb2b2b2b2u, 0xc3c3c3c3u,
                                            0xd4d4d4d4u, 0xe5e5e5e5u, 0x47474747u, 0x50505050u};
std::uint32_t pattern(Tag tag, UINT64 word) { return seeds[tag] ^ static_cast<std::uint32_t>(word * 0x9e3779b1u); }

std::string hex(HRESULT hr)
{
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08lx", static_cast<unsigned long>(hr));
    return text;
}

// `count` words of `tag` at byte `offset` of `cpu`; the pattern index is the word's index from `cpu`.
void write(std::uint8_t* cpu, UINT64 offset, Tag tag, UINT64 count)
{
    auto* w = reinterpret_cast<std::uint32_t*>(cpu);
    for (UINT64 i = 0; i < count; ++i) w[offset / 4 + i] = pattern(tag, offset / 4 + i);
}

struct Tally {
    unsigned count[tag_count]{};
    unsigned zero = 0, other = 0;
    std::string text() const
    {
        std::string out;
        for (int t = 0; t < tag_count; ++t)
            if (count[t]) out += (out.empty() ? "" : " ") + std::string(tag_names[t]) + "=" + std::to_string(count[t]);
        if (zero) out += (out.empty() ? "" : " ") + std::string("zero=") + std::to_string(zero);
        if (other) out += (out.empty() ? "" : " ") + std::string("other=") + std::to_string(other);
        return out + "/" + std::to_string(words);
    }
};
// Attributes each word to the sentinel it holds after removing `mask`; word i has pattern index first + i.
Tally tally(const std::uint32_t* got, UINT64 first, std::uint32_t mask)
{
    Tally t;
    for (UINT i = 0; i < words; ++i) {
        const std::uint32_t value = got[i] ^ mask;
        bool matched = false;
        for (int tag = 0; tag < tag_count && !matched; ++tag)
            if (value == pattern(static_cast<Tag>(tag), first + i)) {
                ++t.count[tag];
                matched = true;
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

using Word = std::uint64_t;
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0)
{
    using Slot = Word(WINAPI*)(void*, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e);
}
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }

// The desktop object behind a wrapper: SetPrivateDataInterface unwraps its argument before the call
// reaches the shared desktop device (as in tests/startup_indirect_dispatch.cpp).
template <typename T>
ComPtr<T> underlying(ID3D12Device* desktop, ID3D12Device* guest, IUnknown* object)
{
    constexpr GUID slot{0x6c1f0a3e, 0x2b51, 0x4d0e, {0x9b, 0x31, 0x5e, 0x0c, 0x72, 0x14, 0x88, 0x19}};
    ComPtr<T> result;
    if (FAILED(guest->SetPrivateDataInterface(slot, object))) return result;
    ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT read = desktop->GetPrivateData(slot, &bytes, stored.GetAddressOf());
    guest->SetPrivateDataInterface(slot, nullptr);
    if (SUCCEEDED(read)) stored.As(&result);
    return result;
}

constexpr char shader_source[] = R"(
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer Key : register(b0) { uint key; };
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { dst.Store(id.x * 4, src.Load(id.x * 4) ^ key); }
)";

// The X2W_TRAP_LOG file d3d12_xs.dll appends its log lines to; a line with both needles.
std::wstring log_path;
bool log_has(const std::string& first, const std::string& second)
{
    HANDLE file = CreateFileW(log_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    std::string text;
    char chunk[65536];
    DWORD got = 0;
    while (ReadFile(file, chunk, sizeof(chunk), &got, nullptr) && got) text.append(chunk, got);
    CloseHandle(file);
    std::size_t at = 0;
    while (at < text.size()) {
        std::size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(at, end - at);
        if (line.find(first) != std::string::npos && line.find(second) != std::string::npos) return true;
        at = end + 1;
    }
    return false;
}
std::string pointer_text(const void* p)
{
    char text[32];
    std::snprintf(text, sizeof(text), "%p", p);
    return text;
}

enum class Path { default_backing, share, copy, refusal, untracked, stale, contention, lease };
const char* path_name(Path p)
{
    switch (p) {
    case Path::default_backing: return "default";
    case Path::share: return "share";
    case Path::copy: return "copy";
    case Path::refusal: return "refusal";
    case Path::untracked: return "untracked";
    case Path::stale: return "stale";
    case Path::contention: return "contention";
    default: return "lease";
    }
}

// ---- Title memory ----

struct Memory {
    std::uint8_t* cpu = nullptr;  // allocation start; the buffer is at cpu + buffer_offset
    std::vector<ULONG_PTR> tokens;
    std::uint8_t* buffer() const { return cpu + buffer_offset; }
};

struct Pool {
    bool xmem;
    std::uint8_t* base = nullptr;
    UINT64 cursor = 0, bytes = 0;
    explicit Pool(bool x) : xmem(x) {}
    const char* name() const { return xmem ? "xmem" : "private"; }

    bool init()
    {
        bytes = xmem ? UINT64{16} << 30 : UINT64{8} << 20;  // XMem takes the 16 GiB startup reservation shape
        base = static_cast<std::uint8_t*>(xmem ? XMemVirtualAlloc(nullptr, static_cast<SIZE_T>(bytes), 0x20402000, 0x21, PAGE_READWRITE)
                                               : VirtualAlloc(nullptr, static_cast<SIZE_T>(bytes), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!base) return false;
        if (!xmem) return true;
        // A previous user: its slots are freed, so the next allocation reuses them and must read zero.
        Memory previous;
        if (!map(previous, alloc_pages)) return false;
        write(previous.cpu, 0, D, alloc_pages * page / 4);
        return release(previous);
    }
    bool map(Memory& out, unsigned pages, bool split_last_page = false)
    {
        out.cpu = base + cursor;
        if (cursor + pages * page > bytes) return false;
        cursor += pages * page;
        if (!xmem) return true;
        out.tokens.assign(pages, 0);
        ULONG_PTR got = pages;
        if (!XMemAllocatePhysicalPages(0x20400000, &got, out.tokens.data()) || got != pages) return false;
        if (!split_last_page) return XMemMapPhysicalPages(out.cpu, pages, out.tokens.data()) == out.cpu;

        // Map the upper page first so that, with X2W_XMEM_JOIN_CALLS=1, neither call can join across the boundary
        // inside the buffer.
        auto* const last_page = out.cpu + (pages - 1) * page;
        return XMemMapPhysicalPages(last_page, 1, &out.tokens.back()) == last_page &&
               XMemMapPhysicalPages(out.cpu, pages - 1, out.tokens.data()) == out.cpu;
    }
    // One allocation in the case layout: neighbour pages, then the buffer filled with `fill`.
    bool allocate(Memory& out, Tag fill, bool* zero = nullptr)
    {
        if (!map(out, alloc_pages)) return false;
        bool clean = true;
        for (UINT64 i = 0; i < alloc_pages * page; i += 4) clean &= *reinterpret_cast<std::uint32_t*>(out.cpu + i) == 0;
        if (zero) *zero = clean;
        write(out.cpu, 0, neighbour, buffer_offset / 4);
        write(out.buffer(), 0, fill, buffer_bytes / 4);
        return true;
    }
    // Two pages of the buffer in two mapping calls (two views): the buffer crosses a view boundary.
    bool allocate_split(Memory& out, Tag fill)
    {
        if (!xmem || !map(out, alloc_pages, true)) return false;
        write(out.cpu, 0, neighbour, buffer_offset / 4);
        write(out.buffer(), 0, fill, buffer_bytes / 4);
        return true;
    }
    bool release(Memory& memory)
    {
        if (!xmem) return true;
        const auto count = static_cast<ULONG_PTR>(memory.tokens.size());
        return XMemMapPhysicalPages(memory.cpu, count, nullptr) == memory.cpu &&
               XMemFreePhysicalPages(count, memory.tokens.data());
    }
};

// ---- Device, the wrapper's queue, and recording ----

// Another queue of the title's device: its wrapper, the desktop queue behind it and a completion fence.
struct Lane {
    ComPtr<ID3D12CommandQueue> guest, native;
    ComPtr<ID3D12Fence> fence;
    UINT64 value = 0;
};

bool wait_fence(ID3D12Fence* fence, UINT64 value)
{
    if (fence->GetCompletedValue() >= value) return true;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const bool ok = event && SUCCEEDED(fence->SetEventOnCompletion(value, event)) && WaitForSingleObject(event, 20000) == WAIT_OBJECT_0;
    if (event) CloseHandle(event);
    return ok;
}
// Polls instead of registering an event, for a wait that is expected to time out in a negative control (a
// registration left on a closed handle could later set an unrelated event that reuses it).
bool poll_fence(ID3D12Fence* fence, UINT64 value, DWORD timeout_ms)
{
    const ULONGLONG deadline = GetTickCount64() + timeout_ms;
    while (fence->GetCompletedValue() < value) {
        if (GetTickCount64() >= deadline) return false;
        Sleep(1);
    }
    return true;
}

struct Gpu {
    ComPtr<ID3D12Device> desktop, guest;
    ComPtr<ID3D12CommandQueue> guest_queue, queue;  // the wrapper and the desktop queue behind it
    ComPtr<ID3D12Fence> done, gate;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    // GPU-read results (one slot per list that may be in flight at once), their CPU copy, the G words.
    static constexpr UINT slots = 4;
    ComPtr<ID3D12Resource> output, readback, source;
    ComPtr<ID3D12InfoQueue> info;
    UINT64 submitted = 0, gate_value = 0;
    bool gate_armed = false;

    bool init()
    {
        const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        if (FAILED(guest->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&guest_queue)))) return false;
        queue = underlying<ID3D12CommandQueue>(desktop.Get(), guest.Get(), guest_queue.Get());
        if (!queue || FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))) ||
            FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))))
            return false;
        D3D12_ROOT_PARAMETER params[3]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[2].Constants = {0, 0, 1};
        const D3D12_ROOT_SIGNATURE_DESC root_desc{3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob, code, errors;
        if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr)) ||
            FAILED(desktop->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root))) ||
            FAILED(D3DCompile(shader_source, sizeof(shader_source) - 1, nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0,
                              &code, &errors)))
            return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get();
        pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        if (FAILED(desktop->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)))) return false;
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        auto desc = buffer_desc(slots * region_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                    IID_PPV_ARGS(&output))))
            return false;
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&readback))))
            return false;
        heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        desc.Width = region_bytes;
        if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                    nullptr, IID_PPV_ARGS(&source))))
            return false;
        void* mapped = nullptr;
        if (FAILED(source->Map(0, nullptr, &mapped))) return false;
        // The G words are indexed as the buffer's words at r1, so the alias can be tallied in place.
        for (UINT i = 0; i < words; ++i) static_cast<std::uint32_t*>(mapped)[i] = pattern(G, r1 / 4 + i);
        source->Unmap(0, nullptr);
        return true;
    }

    // The DLL's internal refresh queue, held behind a fence of the fixture's own.
    ComPtr<ID3D12CommandQueue> internal;
    ComPtr<ID3D12Fence> internal_gate, internal_done;
    UINT64 internal_gate_value = 0, internal_done_value = 0;
    bool hold_internal()
    {
        if (!internal) {
            void* queue = nullptr;
            if (!x2w_d3d12_linear_buffer_internal_queue(&queue)) return false;
            internal.Attach(static_cast<ID3D12CommandQueue*>(queue));
        }
        if (!internal_gate && (FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&internal_gate))) ||
                               FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&internal_done)))))
            return false;
        return SUCCEEDED(internal->Wait(internal_gate.Get(), ++internal_gate_value));
    }
    void open_internal()
    {
        if (internal_gate) internal_gate->Signal(internal_gate_value);
    }
    // Every refresh enqueued so far has landed.
    bool drain_internal()
    {
        return internal && SUCCEEDED(internal->Signal(internal_done.Get(), ++internal_done_value)) &&
               wait_fence(internal_done.Get(), internal_done_value);
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
    bool lane(Lane& out, D3D12_COMMAND_LIST_TYPE type)
    {
        const D3D12_COMMAND_QUEUE_DESC desc{type};
        if (FAILED(guest->CreateCommandQueue(&desc, IID_PPV_ARGS(&out.guest)))) return false;
        out.native = underlying<ID3D12CommandQueue>(desktop.Get(), guest.Get(), out.guest.Get());
        return out.native && SUCCEEDED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&out.fence)));
    }
    bool wait(UINT64 value)
    {
        if (done->GetCompletedValue() >= value) return true;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool ok = event && SUCCEEDED(done->SetEventOnCompletion(value, event)) &&
                        WaitForSingleObject(event, 20000) == WAIT_OBJECT_0;
        if (event) CloseHandle(event);
        return ok;
    }
    std::vector<std::uint32_t> read_output(UINT slot = 0)
    {
        std::vector<std::uint32_t> got(words);
        void* mapped = nullptr;
        const D3D12_RANGE range{slot * region_bytes, (slot + 1) * region_bytes};
        if (SUCCEEDED(readback->Map(0, &range, &mapped))) {
            std::memcpy(got.data(), static_cast<const std::uint8_t*>(mapped) + slot * region_bytes, region_bytes);
            const D3D12_RANGE none{0, 0};
            readback->Unmap(0, &none);
        }
        return got;
    }
    // The CPU writes `tag` ^ key into result slot `slot` (indexed as the region at r0), so a read that never
    // ran leaves it there.
    bool fill_output(UINT slot, Tag tag)
    {
        void* mapped = nullptr;
        const D3D12_RANGE none{0, 0};
        if (FAILED(readback->Map(0, &none, &mapped))) return false;
        auto* w = reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(mapped) + slot * region_bytes);
        for (UINT i = 0; i < words; ++i) w[i] = pattern(tag, r0 / 4 + i) ^ key;
        const D3D12_RANGE written{slot * region_bytes, (slot + 1) * region_bytes};
        readback->Unmap(0, &written);
        return true;
    }
    unsigned debug_errors()
    {
        if (!info) return 0;
        unsigned errors = 0;
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
            SIZE_T bytes = 0;
            if (FAILED(info->GetMessage(i, nullptr, &bytes))) continue;
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (FAILED(info->GetMessage(i, message, &bytes)) || message->Severity > D3D12_MESSAGE_SEVERITY_ERROR) continue;
            if (++errors <= 8) std::printf("debug layer ERROR id %d: %.300s\n", message->ID, message->pDescription);
        }
        return errors;
    }
};

// One title command list: a wrapper list on its own wrapper allocator, recorded on its desktop list.
struct TitleList {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<IUnknown> list;
    ComPtr<ID3D12GraphicsCommandList> native;
    Gpu* gpu = nullptr;

    bool begin(Gpu& g, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT)
    {
        gpu = &g;
        if (FAILED(g.guest->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator)))) return false;
        using CreateList = HRESULT(WINAPI*)(IUnknown*, const void*, IUnknown*, IUnknown*, REFIID, void**);
        const UINT list_desc[]{static_cast<UINT>(type), 2, 1, 0};
        auto* create = reinterpret_cast<CreateList>((*reinterpret_cast<void***>(g.guest.Get()))[73]);
        if (FAILED(create(g.guest.Get(), list_desc, allocator.Get(), nullptr, __uuidof(ID3D12GraphicsCommandList),
                          reinterpret_cast<void**>(list.GetAddressOf()))))
            return false;
        if (FAILED(static_cast<HRESULT>(invoke(list.Get(), 139, arg(allocator.Get()))))) return false;
        native = underlying<ID3D12GraphicsCommandList>(g.desktop.Get(), g.guest.Get(), list.Get());
        return native != nullptr;
    }
    void barrier(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER b{};
        b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        native->ResourceBarrier(1, &b);
    }
    void dispatch(D3D12_GPU_VIRTUAL_ADDRESS src, D3D12_GPU_VIRTUAL_ADDRESS dst)
    {
        native->SetComputeRootSignature(gpu->root.Get());
        native->SetPipelineState(gpu->pipeline.Get());
        native->SetComputeRootShaderResourceView(0, src);
        native->SetComputeRootUnorderedAccessView(1, dst);
        native->SetComputeRoot32BitConstant(2, key, 0);
        native->Dispatch(words / 64, 1, 1);
    }
    // output[i] = buffer[r0 + 4i] ^ key, then output -> readback, in result slot `slot`.
    void read(ID3D12Resource* buffer, UINT slot = 0)
    {
        const UINT64 at = slot * region_bytes;
        barrier(buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(gpu->output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch(buffer->GetGPUVirtualAddress() + r0, gpu->output->GetGPUVirtualAddress() + at);
        barrier(buffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        barrier(gpu->output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        native->CopyBufferRegion(gpu->readback.Get(), at, gpu->output.Get(), at, region_bytes);
        barrier(gpu->output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    // For a COPY list: buffer[r0 .. r0 + 4 KiB) -> readback slot, unchanged (the buffer is promoted
    // from COMMON to COPY_SOURCE implicitly).
    void read_copy(ID3D12Resource* buffer, UINT slot = 0)
    {
        native->CopyBufferRegion(gpu->readback.Get(), slot * region_bytes, buffer, r0, region_bytes);
    }
    // buffer[r1 + 4i] = G[i] ^ key.
    void write_g(ID3D12Resource* buffer)
    {
        barrier(buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch(gpu->source->GetGPUVirtualAddress(), buffer->GetGPUVirtualAddress() + r1);
        barrier(buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    }
    bool close() { return SUCCEEDED(static_cast<HRESULT>(invoke(list.Get(), 9))); }
    // Through the queue wrapper (slot 10, ExecuteCommandLists), behind the gate if it is held.
    UINT64 submit()
    {
        if (gpu->gate_armed) gpu->queue->Wait(gpu->gate.Get(), gpu->gate_value);
        void* lists[] = {list.Get()};
        invoke(gpu->guest_queue.Get(), 10, 1, arg(lists));
        gpu->queue->Signal(gpu->done.Get(), ++gpu->submitted);
        return gpu->submitted;
    }
    // Through another queue's wrapper; the lane's fence completes it.
    UINT64 submit_on(Lane& lane)
    {
        void* lists[] = {list.Get()};
        invoke(lane.guest.Get(), 10, 1, arg(lists));
        lane.native->Signal(lane.fence.Get(), ++lane.value);
        return lane.value;
    }
    // Past the wrapper: the desktop list on the desktop queue (a submission without the refresh).
    UINT64 submit_bypassing()
    {
        ID3D12CommandList* lists[] = {native.Get()};
        gpu->queue->ExecuteCommandLists(1, lists);
        gpu->queue->Signal(gpu->done.Get(), ++gpu->submitted);
        return gpu->submitted;
    }
};

// ---- Placement through slot 133 ----

struct Placed {
    HRESULT hr = E_FAIL;
    const void* address = nullptr;
    ComPtr<IUnknown> wrapper;
    ComPtr<ID3D12Resource> backing;
    X2wLinearBufferInfo info{};
    bool linear = false;
};

Placed place(Gpu& gpu, std::uint8_t* address, bool gpu_written)
{
    Placed p;
    p.address = address;
    using Place = HRESULT(WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
    const auto desc = buffer_desc(buffer_bytes, gpu_written ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE);
    auto* slot = reinterpret_cast<Place>((*reinterpret_cast<void***>(gpu.guest.Get()))[133]);
    p.hr = slot(gpu.guest.Get(), address, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource),
                reinterpret_cast<void**>(p.wrapper.GetAddressOf()));
    if (SUCCEEDED(p.hr) && p.wrapper) {
        p.backing = underlying<ID3D12Resource>(gpu.desktop.Get(), gpu.guest.Get(), p.wrapper.Get());
        p.linear = x2w_d3d12_linear_buffer_info(p.wrapper.Get(), &p.info);
    }
    return p;
}
void refresh_info(Placed& p)
{
    if (p.wrapper) p.linear = x2w_d3d12_linear_buffer_info(p.wrapper.Get(), &p.info);
}

std::string backing_text(const Placed& p)
{
    if (FAILED(p.hr)) return "refused " + hex(p.hr);
    if (!p.linear) return "default";
    char text[160];
    if (p.info.path == 3)
        std::snprintf(text, sizeof(text), "kept default backing (refusal %s, share %s, feature %s)", hex(p.info.refusal_hr).c_str(),
                      hex(p.info.share_hr).c_str(), hex(p.info.feature_hr).c_str());
    else if (p.info.path == 1)
        std::snprintf(text, sizeof(text), "shared (heap generation %llu, feature %s)",
                      static_cast<unsigned long long>(p.info.heap_generation), hex(p.info.feature_hr).c_str());
    else
        std::snprintf(text, sizeof(text), "copied (share %s, feature %s, %llu bytes in %llu refreshes, staging %llu)",
                      hex(p.info.share_hr).c_str(), hex(p.info.feature_hr).c_str(),
                      static_cast<unsigned long long>(p.info.bytes_copied), static_cast<unsigned long long>(p.info.refreshes),
                      static_cast<unsigned long long>(p.info.last_staging));
    return text;
}

X2wLinearBufferTotals totals()
{
    X2wLinearBufferTotals t{};
    x2w_d3d12_linear_buffer_totals(&t);
    return t;
}

// ---- Cases ----

struct Run {
    Gpu& gpu;
    Path path;
    bool adapter_heaps;  // the adapter's own D3D12_FEATURE_EXISTING_HEAPS: the default aliases private ranges
    unsigned failures = 0, skips = 0, kept_rows = 0;
    std::vector<std::pair<std::string, bool>> results;  // case name -> passed (for --mutate)

    bool copying() const { return path == Path::copy || path == Path::refusal; }
    // The copy backs this pool's buffers: forced, refused by the host, or an XMem view on the share path.
    bool copied(const Pool& pool) const { return copying() || (path == Path::share && pool.xmem); }
    // The sharing HRESULT a refused sharing attempt reports for a buffer inside one allocation.
    HRESULT share_refusal(const Pool& pool) const
    {
        return path == Path::refusal ? DXGI_ERROR_UNSUPPORTED : path == Path::share && pool.xmem ? HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED) : S_FALSE;
    }
    // The default backing aliases this pool: a private range on an adapter with existing heaps.
    bool default_aliases(const Pool& pool) const { return !pool.xmem && adapter_heaps; }
    // A refusal that kept the default backing: slot 133 returned S_OK with an object, and info and the
    // slot 133 log line carry the refusal HRESULT.
    bool kept_ok(const Placed& p, HRESULT want, std::string& text)
    {
        char needle[64];
        std::snprintf(needle, sizeof(needle), "kept default backing: linear refusal 0x%08lx", static_cast<unsigned long>(want));
        const bool logged = log_has("address=" + pointer_text(p.address), needle);
        const bool ok = SUCCEEDED(p.hr) && p.wrapper && p.linear && p.info.path == 3 && p.info.refusal_hr == want && logged;
        text += std::string("; refusal ") + (p.linear ? hex(p.info.refusal_hr) : std::string("NOT IN INFO")) + " (expected " + hex(want) +
                "), log line " + (logged ? "present" : "MISSING");
        ++kept_rows;
        return ok;
    }
    void row(const Pool& pool, const char* name, bool pass, const std::string& text)
    {
        std::printf("%-8s %-8s %-6s %s: %s\n", path_name(path), pool.name(), name, pass ? "PASS" : "FAIL", text.c_str());
        results.emplace_back(name, pass);
        if (!pass) ++failures;
    }
    // The path's expected backing for a GPU-read buffer in this pool.
    bool backing_ok(const Pool& pool, const Placed& p, bool split = false) const
    {
        switch (path) {
        case Path::default_backing: return SUCCEEDED(p.hr) && !p.linear;
        case Path::share:
            if (!pool.xmem) return SUCCEEDED(p.hr) && p.linear && p.info.path == 1 && p.info.share_hr == S_OK;
            return SUCCEEDED(p.hr) && p.linear && p.info.path == 2 && p.info.feature_hr == S_OK &&
                   p.info.share_hr == (split ? E_INVALIDARG : HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED));
        case Path::copy: return SUCCEEDED(p.hr) && p.linear && p.info.path == 2 && p.info.share_hr == S_FALSE;
        case Path::refusal: return SUCCEEDED(p.hr) && p.linear && p.info.path == 2 && p.info.share_hr == DXGI_ERROR_UNSUPPORTED &&
                                   p.info.feature_hr == DXGI_ERROR_UNSUPPORTED;
        default: return false;  // untracked: kept_ok decides
        }
    }
    // What the GPU must read in case a: B, except where the default backing does not alias (zero).
    bool read_ok(const Pool& pool, const Tally& t, Tag want) const
    {
        if ((path == Path::default_backing || path == Path::untracked) && !default_aliases(pool)) return t.zero == words;
        return t.count[want] == words;
    }

    void case_a(Pool& pool, bool split = false)
    {
        const char* name = split ? "split" : "a";
        Memory memory;
        if (!(split ? pool.allocate_split(memory, A) : pool.allocate(memory, A))) return row(pool, name, false, "allocation failed");
        Placed p = place(gpu, memory.buffer(), false);
        TitleList list;
        if (FAILED(p.hr) || !p.backing || !list.begin(gpu)) return row(pool, name, false, "slot 133 -> " + hex(p.hr));
        list.read(p.backing.Get());
        if (!list.close()) return row(pool, name, false, "Close failed");
        write(memory.buffer(), r0, B, words);  // after bind and record, before ExecuteCommandLists
        if (!gpu.wait(list.submit())) return row(pool, name, false, "fence timeout");
        const Tally t = tally(gpu.read_output().data(), r0 / 4, key);
        refresh_info(p);
        std::string text = "slot 133 -> " + hex(p.hr) + ", " + backing_text(p) + "; GPU read " + t.text();
        bool pass = read_ok(pool, t, B) && (!p.linear || p.info.path != 2 || p.info.bytes_copied == buffer_bytes);
        if (path == Path::untracked)
            pass &= kept_ok(p, E_NOT_VALID_STATE, text);  // no submit fence to guard staging reuse
        else
            pass &= backing_ok(pool, p, split);
        if (path == Path::default_backing || path == Path::untracked)
            text += default_aliases(pool) ? " (old behaviour: aliased)" : " (old behaviour: not aliased, no copy)";
        row(pool, name, pass, text);
        if (split || !pass || path == Path::default_backing || path == Path::untracked) return;

        // a2: 16 words change; only they are copied.
        TitleList second;
        if (!second.begin(gpu)) return row(pool, "a2", false, "list");
        second.read(p.backing.Get());
        second.close();
        const UINT64 before = p.info.bytes_copied;
        write(memory.buffer(), r0 + 64 * 4, E, 16);
        if (!gpu.wait(second.submit())) return row(pool, "a2", false, "fence timeout");
        const Tally t2 = tally(gpu.read_output().data(), r0 / 4, key);
        refresh_info(p);
        const UINT64 copied = p.info.bytes_copied - before;
        const bool pass2 = t2.count[E] == 16 && t2.count[B] == words - 16 && (p.info.path != 2 || copied == 64);
        row(pool, "a2", pass2, "GPU read " + t2.text() + "; bytes copied for this submission " + std::to_string(copied) +
                                   (p.info.path == 2 ? " (expected 64)" : " (shared: none)"));
    }

    void case_b(Pool& pool, bool hold)
    {
        const char* name = hold ? "d2" : "b";
        Memory memory;
        if (!pool.allocate(memory, filler)) return row(pool, name, false, "allocation failed");
        write(memory.buffer(), r1, P, words);
        Placed p = place(gpu, memory.buffer(), true);
        const auto* alias = reinterpret_cast<const std::uint32_t*>(memory.buffer() + r1);
        // The copy has no point at which a GPU write reaches the title memory: slot 133 keeps the default
        // backing and reports the sharing HRESULT (the feature gate's DXGI_ERROR_UNSUPPORTED, or
        // ERROR_NOT_SUPPORTED for an XMem view), or DXGI_ERROR_UNSUPPORTED when sharing is not attempted.
        const bool kept = copied(pool) || path == Path::untracked;
        const bool on_default = kept || path == Path::default_backing;
        const HRESULT want = share_refusal(pool) == S_FALSE ? DXGI_ERROR_UNSUPPORTED : share_refusal(pool);
        TitleList list;
        if (FAILED(p.hr) || !p.backing || !list.begin(gpu)) return row(pool, name, false, "slot 133 -> " + hex(p.hr));
        list.write_g(p.backing.Get());
        if (!list.close()) return row(pool, name, false, "Close failed");
        if (hold) gpu.hold();
        const UINT64 value = list.submit();
        std::string early;
        bool early_ok = true;
        if (hold) {
            // Ordered before the GPU write by the queue's Wait on the gate: no timing assumption.
            const bool pending = gpu.done->GetCompletedValue() < value;
            const Tally before = tally(alias, r1 / 4, 0);
            early_ok = pending && before.count[P] == words;
            early = std::string("before the fence: ") + (pending ? "in flight" : "COMPLETE") + ", alias " + before.text() + "; ";
            gpu.open_gate();
        }
        if (!gpu.wait(value)) return row(pool, name, false, "fence timeout");
        const Tally t = tally(alias, r1 / 4, key), raw = tally(alias, r1 / 4, 0);
        // The words around the written region keep their filler.
        bool around = true;
        const auto* buffer = reinterpret_cast<const std::uint32_t*>(memory.buffer());
        for (UINT64 i = 0; i < buffer_bytes / 4; ++i)
            if (i < r1 / 4 || i >= r1 / 4 + words) around &= buffer[i] == pattern(filler, i);
        refresh_info(p);
        bool pass;
        std::string text = "slot 133 -> " + hex(p.hr) + ", " + backing_text(p) + "; " + early + "after the fence alias " +
                           (t.count[G] ? t.text() + " (G ^ key)" : raw.text()) + (around ? ", neighbours kept" : ", NEIGHBOURS CHANGED");
        // Visible: shared, or a default backing that aliases this pool.
        const bool visible = on_default ? default_aliases(pool) : true;
        pass = early_ok && around && (visible ? t.count[G] == words : raw.count[P] == words);
        if (path == Path::default_backing) pass &= SUCCEEDED(p.hr) && !p.linear;
        else if (kept) pass &= kept_ok(p, want, text);
        else pass &= p.linear && p.info.path == 1;
        if (on_default) text += visible ? " (old behaviour: aliased)" : " (old behaviour: not aliased, the alias keeps P)";
        row(pool, name, pass, text);
    }

    // c: held submissions and backing reuse only after completion.
    void case_c(Pool& pool)
    {
        if (path == Path::default_backing || path == Path::untracked) return;
        const bool copying = copied(pool);
        struct Held {
            Memory memory;
            Placed placed;
            TitleList list;
            UINT64 value = 0;
            Tag tag;
        };
        const Tag tags[3] = {B, C, D};
        Held held[3];
        const X2wLinearBufferTotals start = totals();
        gpu.hold();
        // Copying: the refreshes are held too, on the internal queue, so their staging stays in flight.
        if (copying && !gpu.hold_internal()) return row(pool, "c", false, "internal refresh queue not available");
        for (int i = 0; i < 3; ++i) {
            Held& h = held[i];
            h.tag = tags[i];
            if (!pool.allocate(h.memory, h.tag)) return row(pool, "c", false, "allocation failed");
            h.placed = place(gpu, h.memory.buffer(), false);
            if (FAILED(h.placed.hr) || !h.placed.backing || !h.list.begin(gpu)) return row(pool, "c", false, "slot 133 -> " + hex(h.placed.hr));
            h.list.read(h.placed.backing.Get(), static_cast<UINT>(i + 1));
            h.list.close();
            h.value = h.list.submit();
            refresh_info(h.placed);
            // The title is done with X once it is submitted: on the shared path it releases it now.
            if (i == 0 && !copying) h.placed.wrapper.Reset();
        }
        const X2wLinearBufferTotals during = totals();
        const bool pending = gpu.done->GetCompletedValue() < held[0].value;
        // Each held submission reads its own buffer into its own result slot.
        if (copying) gpu.open_internal();
        gpu.open_gate();
        if (!gpu.wait(held[2].value)) return row(pool, "c", false, "fence timeout");
        std::string reads = "held reads";
        bool reads_ok = true;
        for (int i = 0; i < 3; ++i) {
            const Tally t = tally(gpu.read_output(static_cast<UINT>(i + 1)).data(), r0 / 4, key);
            reads_ok &= t.count[held[i].tag] == words;
            reads += std::string(" ") + "XYZ"[i] + "=[" + t.text() + "]";
        }
        std::string text;
        bool pass = pending && reads_ok;
        if (copying) {
            const UINT64 s0 = held[0].placed.info.last_staging, s1 = held[1].placed.info.last_staging,
                         s2 = held[2].placed.info.last_staging;
            const bool distinct = s0 && s1 && s2 && s0 != s1 && s0 != s2 && s1 != s2;
            // After completion the next refresh takes one of those buffers again.
            Memory memory;
            Placed w;
            TitleList list;
            if (!pool.allocate(memory, E)) return row(pool, "c", false, "allocation failed");
            w = place(gpu, memory.buffer(), false);
            if (FAILED(w.hr) || !list.begin(gpu)) return row(pool, "c", false, "slot 133 -> " + hex(w.hr));
            list.read(w.backing.Get());
            list.close();
            if (!gpu.wait(list.submit())) return row(pool, "c", false, "fence timeout");
            const Tally after = tally(gpu.read_output().data(), r0 / 4, key);
            refresh_info(w);
            const X2wLinearBufferTotals end = totals();
            const bool reused_after = w.info.last_staging == s0 || w.info.last_staging == s1 || w.info.last_staging == s2;
            pass &= distinct && reused_after && after.count[E] == words && end.staging_reused > during.staging_reused;
            char line[320];
            std::snprintf(line, sizeof(line),
                          "held: staging %llu/%llu/%llu (%s; %llu created, %llu completed ones reused during the hold); after the fence the next refresh "
                          "took staging %llu (%s) and read %s",
                          static_cast<unsigned long long>(s0), static_cast<unsigned long long>(s1), static_cast<unsigned long long>(s2),
                          distinct ? "distinct" : "SHARED", static_cast<unsigned long long>(during.staging_created - start.staging_created),
                          static_cast<unsigned long long>(during.staging_reused - start.staging_reused),
                          static_cast<unsigned long long>(w.info.last_staging), reused_after ? "reused" : "NOT REUSED", after.text().c_str());
            text = line;
        } else {
            // Shared (a private range): after the fence the title places a new buffer Z at X's address.
            const std::string reuse = "after the fence Z is placed at X's address";
            Memory z = held[0].memory;
            write(z.buffer(), 0, E, buffer_bytes / 4);
            Placed zp = place(gpu, z.buffer(), false);
            TitleList list;
            if (FAILED(zp.hr) || !list.begin(gpu)) return row(pool, "c", false, "slot 133 -> " + hex(zp.hr));
            list.read(zp.backing.Get());
            list.close();
            if (!gpu.wait(list.submit())) return row(pool, "c", false, "fence timeout");
            const Tally after = tally(gpu.read_output().data(), r0 / 4, key);
            const UINT64 gx = held[0].placed.info.heap_generation, gz = zp.info.heap_generation;
            pass &= zp.linear && zp.info.path == 1 && gz == gx && after.count[E] == words;
            text = reuse + "; heap generation " + std::to_string(gx) + " -> " + std::to_string(gz) +
                   " (the range's cached heap); Z read " + after.text();
        }
        row(pool, "c", pass, std::string(pending ? "held in flight" : "NOT HELD") + "; " + reads + "; " + text);
    }

    // Measured, not asserted (share path, XMem): why slot 133 does not share a view. A heap opened from
    // the view, with a buffer placed on it, and XMem's unmap of the view's pages, before and after the
    // heap and buffer are released.
    void probe_view_pin(Pool& pool)
    {
        if (path != Path::share || !pool.xmem) return;
        Memory memory;
        ComPtr<ID3D12Device3> device3;
        if (!pool.allocate(memory, A) || FAILED(gpu.desktop.As(&device3))) return;
        ComPtr<ID3D12Heap> heap;
        ComPtr<ID3D12Resource> buffer;
        HRESULT hr = device3->OpenExistingHeapFromAddress(memory.cpu, IID_PPV_ARGS(&heap));
        const auto desc = buffer_desc(buffer_bytes, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER);
        if (SUCCEEDED(hr)) hr = gpu.desktop->CreatePlacedResource(heap.Get(), buffer_offset, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                                  nullptr, IID_PPV_ARGS(&buffer));
        const auto count = static_cast<ULONG_PTR>(memory.tokens.size());
        const bool held_unmap = XMemMapPhysicalPages(memory.cpu, count, nullptr) == memory.cpu;
        const DWORD held_error = held_unmap ? 0 : GetLastError();
        buffer.Reset();
        heap.Reset();
        const bool unmap = held_unmap || XMemMapPhysicalPages(memory.cpu, count, nullptr) == memory.cpu;
        const DWORD error = unmap ? 0 : GetLastError();
        const bool freed = unmap && XMemFreePhysicalPages(count, memory.tokens.data());
        std::printf("%-8s %-8s %-6s measured: heap from the view and placed buffer %s; XMem unmap while they live %s (error %lu); "
                    "after their release %s (error %lu), pages %s\n", path_name(path), pool.name(), "pin", hex(hr).c_str(),
                    held_unmap ? "succeeded" : "failed", held_error, unmap ? "succeeded" : "failed", error,
                    freed ? "freed" : "not freed");
    }

    // stale: a refresh that cannot be staged; the submission still runs, with the last upload.
    void case_stale(Pool& pool)
    {
        Memory memory;
        if (!pool.allocate(memory, A)) return row(pool, "stale", false, "allocation failed");
        Placed p = place(gpu, memory.buffer(), false);
        TitleList first, second, third;
        if (FAILED(p.hr) || !p.backing || !first.begin(gpu) || !second.begin(gpu) || !third.begin(gpu))
            return row(pool, "stale", false, "slot 133 -> " + hex(p.hr));
        first.read(p.backing.Get());
        first.close();
        if (!gpu.wait(first.submit())) return row(pool, "stale", false, "fence timeout (first)");
        const Tally t1 = tally(gpu.read_output().data(), r0 / 4, key);
        refresh_info(p);
        const bool fresh1 = p.linear && p.info.path == 2 && !p.info.stale;
        write(memory.buffer(), r0, B, words);
        second.read(p.backing.Get());
        second.close();
        // The refresh of this submission fails (fixture fault): it must still run and complete.
        const bool completed = gpu.wait(second.submit());
        const Tally t2 = tally(gpu.read_output().data(), r0 / 4, key);
        refresh_info(p);
        char needle[64];
        std::snprintf(needle, sizeof(needle), "failed 0x%08lx", static_cast<unsigned long>(E_OUTOFMEMORY));
        const bool logged = log_has("refresh of " + pointer_text(p.backing.Get()), needle);
        const bool stale2 = p.info.stale == 1 && p.info.refresh_failures == 1 && p.info.last_refresh_hr == E_OUTOFMEMORY;
        const unsigned stale_flag = p.info.stale;
        third.read(p.backing.Get());
        third.close();
        if (!gpu.wait(third.submit())) return row(pool, "stale", false, "fence timeout (third)");
        const Tally t3 = tally(gpu.read_output().data(), r0 / 4, key);
        refresh_info(p);
        const bool pass = fresh1 && t1.count[A] == words && completed && t2.count[A] == words && stale2 && logged &&
                          t3.count[B] == words && !p.info.stale;
        row(pool, "stale", pass,
            "refreshed " + t1.text() + "; failed refresh: submission " + (completed ? "completed" : "DID NOT COMPLETE") + ", read " +
                t2.text() + " (last upload), stale=" + std::to_string(stale_flag) + " " + (stale2 ? "reported" : "NOT REPORTED") +
                " with " + hex(E_OUTOFMEMORY) + ", log line " + (logged ? "present" : "MISSING") + "; next submission read " + t3.text() +
                (p.info.stale ? ", STILL STALE" : ", stale cleared"));
    }

    // q.comp, q.copy: the refresh on a COMPUTE and a COPY queue (implicit COMMON -> COPY_DEST there too).
    void case_queues(Pool& pool)
    {
        for (const auto type : {D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY}) {
            const bool compute = type == D3D12_COMMAND_LIST_TYPE_COMPUTE;
            const char* name = compute ? "q.comp" : "q.copy";
            Lane lane;
            Memory memory;
            if (!gpu.lane(lane, type)) return row(pool, name, false, "queue wrapper not created");
            if (!pool.allocate(memory, A)) return row(pool, name, false, "allocation failed");
            Placed p = place(gpu, memory.buffer(), false);
            TitleList first, second;
            if (FAILED(p.hr) || !p.backing || !first.begin(gpu, type) || !second.begin(gpu, type))
                return row(pool, name, false, "slot 133 -> " + hex(p.hr) + " or list creation failed");
            const std::uint32_t mask = compute ? key : 0;
            compute ? first.read(p.backing.Get()) : first.read_copy(p.backing.Get());
            if (!first.close()) return row(pool, name, false, "Close failed");
            write(memory.buffer(), r0, B, words);
            if (!wait_fence(lane.fence.Get(), first.submit_on(lane))) return row(pool, name, false, "fence timeout");
            const Tally t1 = tally(gpu.read_output().data(), r0 / 4, mask);
            refresh_info(p);
            const UINT64 after_first = p.info.bytes_copied;
            compute ? second.read(p.backing.Get()) : second.read_copy(p.backing.Get());
            second.close();
            write(memory.buffer(), r0 + 64 * 4, E, 16);
            if (!wait_fence(lane.fence.Get(), second.submit_on(lane))) return row(pool, name, false, "fence timeout");
            const Tally t2 = tally(gpu.read_output().data(), r0 / 4, mask);
            refresh_info(p);
            const UINT64 delta = p.info.bytes_copied - after_first;
            const bool pass = p.linear && p.info.path == 2 && !p.info.stale && p.info.refresh_failures == 0 && t1.count[B] == words &&
                              after_first == buffer_bytes && t2.count[E] == 16 && t2.count[B] == words - 16 && delta == 64;
            row(pool, name, pass,
                std::string(compute ? "COMPUTE queue, dispatch read: " : "COPY queue, CopyBufferRegion read: ") + "first submission " +
                    t1.text() + " (" + std::to_string(after_first) + " bytes refreshed), then " + t2.text() + " (" +
                    std::to_string(delta) + " bytes)" + (p.info.stale ? ", STALE" : ""));
        }
    }

    // gate2q: one buffer, two queues, one CPU write before both submissions (the review repro).
    void case_gate2q(Pool& pool, bool held)
    {
        const char* name = held ? "gate2q.held" : "gate2q";
        Lane q2;
        Memory memory;
        if (!gpu.lane(q2, D3D12_COMMAND_LIST_TYPE_DIRECT) || !pool.allocate(memory, A)) return row(pool, name, false, "setup failed");
        Placed x = place(gpu, memory.buffer(), false);
        TitleList l0, l1, l2;
        if (FAILED(x.hr) || !x.backing || !l0.begin(gpu) || !l1.begin(gpu) || !l2.begin(gpu))
            return row(pool, name, false, "slot 133 -> " + hex(x.hr));
        l0.read(x.backing.Get());
        l0.close();
        if (!gpu.wait(l0.submit())) return row(pool, name, false, "fence timeout");  // A uploaded and read
        l1.read(x.backing.Get(), 1);
        l1.close();
        l2.read(x.backing.Get(), 2);
        l2.close();
        write(memory.buffer(), r0, B, words);
        const X2wLinearBufferTotals before = totals();
        gpu.hold();
        if (held && !gpu.hold_internal()) return row(pool, name, false, "internal refresh queue not available");
        const UINT64 v1 = l1.submit();  // queue 1, behind the title gate: its refresh goes to the internal queue
        const UINT64 v2 = l2.submit_on(q2);
        bool q2_waited = true;
        if (held) {
            // Queue 2 has nothing to refresh itself (B is already enqueued) but must wait for that refresh.
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            q2_waited = q2.fence->GetCompletedValue() < v2;
            gpu.open_internal();
        }
        const bool q2_done = wait_fence(q2.fence.Get(), v2);
        const bool q1_pending = gpu.done->GetCompletedValue() < v1;
        const Tally t2 = tally(gpu.read_output(2).data(), r0 / 4, key);
        gpu.open_gate();
        if (!gpu.wait(v1)) return row(pool, name, false, "fence timeout (queue 1)");
        const Tally t1 = tally(gpu.read_output(1).data(), r0 / 4, key);
        refresh_info(x);
        const X2wLinearBufferTotals after = totals();
        const bool pass = q2_done && q1_pending && q2_waited && t2.count[B] == words && t1.count[B] == words && !x.info.stale &&
                          x.info.refresh_failures == 0 && after.consumer_waits > before.consumer_waits;
        row(pool, name, pass,
            std::string("queue 2 ") + (q2_done ? "completed" : "DID NOT COMPLETE") + " with queue 1 " +
                (q1_pending ? "still gated" : "NOT GATED") + (held ? (q2_waited ? ", after waiting for the held refresh" : ", WITHOUT WAITING for the held refresh") : "") +
                ": read " + t2.text() + "; queue 1 then read " + t1.text() + "; consumer waits +" +
                std::to_string(after.consumer_waits - before.consumer_waits) + (x.info.stale ? ", STALE" : ""));
    }

    // xq: queue 1 waits on a title fence that queue 2 signals after its list.
    // gpuw: a recorded copy (list slot 15, CopyBufferRegion) writes G into a copied buffer; the title then
    // changes bytes on both sides of it. The buffer keeps the GPU's bytes (no refresh spans them), and
    // info and totals report it.
    void case_gpuw(Pool& pool)
    {
        Memory memory;
        if (!pool.allocate(memory, A)) return row(pool, "gpuw", false, "allocation failed");
        Placed p = place(gpu, memory.buffer(), false);
        TitleList l0, l1, l2;
        if (FAILED(p.hr) || !p.backing || !l0.begin(gpu) || !l1.begin(gpu) || !l2.begin(gpu))
            return row(pool, "gpuw", false, "slot 133 -> " + hex(p.hr));
        l0.read(p.backing.Get());
        l0.close();
        if (!gpu.wait(l0.submit())) return row(pool, "gpuw", false, "fence timeout");  // A uploaded
        const X2wLinearBufferTotals before = totals();
        // The title's copy, through the list wrapper: G (indexed as at r1) into the buffer at r0.
        invoke(l1.list.Get(), 15, arg(p.wrapper.Get()), r0, arg(gpu.source.Get()), 0, region_bytes);
        if (!l1.close() || !gpu.wait(l1.submit())) return row(pool, "gpuw", false, "copy submission failed");
        write(memory.buffer(), 0, D, 1);                 // the first and last words, around the written region
        write(memory.buffer(), buffer_bytes - 4, D, 1);
        l2.read(p.backing.Get(), 1);
        l2.close();
        if (!gpu.wait(l2.submit())) return row(pool, "gpuw", false, "fence timeout");
        const Tally t = tally(gpu.read_output(1).data(), r1 / 4, key);  // G words carry their r1 indices
        refresh_info(p);
        const X2wLinearBufferTotals after = totals();
        const bool pass = p.linear && p.info.path == 2 && p.info.gpu_written == 1 && t.count[G] == words &&
                          after.gpu_written == before.gpu_written + 1;
        row(pool, "gpuw", pass,
            "after a recorded copy of G and title writes around it the GPU read " + t.text() + " (expected G: not refreshed); " +
                "gpu_written " + std::to_string(p.info.gpu_written) + ", totals +" + std::to_string(after.gpu_written - before.gpu_written));
    }

    // gpuw.discard (the review repro): a recorded copy into a copied buffer that is closed and then reset
    // without being submitted. The buffer must stay refreshed: the next read sees the CPU's B, and it is
    // not reported GPU-written.
    void case_gpuw_discard(Pool& pool)
    {
        const char* name = "gpuw.discard";
        Memory memory;
        if (!pool.allocate(memory, A)) return row(pool, name, false, "allocation failed");
        Placed p = place(gpu, memory.buffer(), false);
        TitleList l0, l1, l2;
        if (FAILED(p.hr) || !p.backing || !l0.begin(gpu) || !l1.begin(gpu) || !l2.begin(gpu))
            return row(pool, name, false, "slot 133 -> " + hex(p.hr));
        l0.read(p.backing.Get());
        l0.close();
        if (!gpu.wait(l0.submit())) return row(pool, name, false, "fence timeout");  // A uploaded and read
        const X2wLinearBufferTotals before = totals();
        // The title's copy of G into the buffer, recorded and closed on another list, then discarded.
        invoke(l1.list.Get(), 15, arg(p.wrapper.Get()), r0, arg(gpu.source.Get()), 0, region_bytes);
        const bool closed = l1.close();
        const bool reset = SUCCEEDED(static_cast<HRESULT>(invoke(l1.list.Get(), 139, arg(l1.allocator.Get()))));
        write(memory.buffer(), r0, B, words);
        l2.read(p.backing.Get(), 1);
        l2.close();
        if (!gpu.wait(l2.submit())) return row(pool, name, false, "fence timeout");
        const Tally t = tally(gpu.read_output(1).data(), r0 / 4, key);
        refresh_info(p);
        const X2wLinearBufferTotals after = totals();
        const bool pass = closed && reset && p.linear && p.info.path == 2 && p.info.gpu_written == 0 && t.count[B] == words &&
                          after.gpu_written == before.gpu_written && !p.info.stale;
        row(pool, name, pass,
            std::string("a copy of G recorded, closed ") + (closed ? "" : "(CLOSE FAILED) ") + "and reset " +
                (reset ? "" : "(RESET FAILED) ") + "without a submission, then the CPU wrote B: the GPU read " + t.text() +
                " (expected B: still refreshed); gpu_written " + std::to_string(p.info.gpu_written) + ", totals +" +
                std::to_string(after.gpu_written - before.gpu_written));
    }

    // war: a read of A on queue 1 is held behind a title gate (on the desktop queue, past the wrapper); the
    // CPU then writes B and queue 2 submits a read, whose refresh must not land before queue 1's read: queue
    // 1 reads A, queue 2 reads B, and the refresh's copy waited for queue 1 (producer_waits).
    void case_war(Pool& pool)
    {
        const char* name = "war";
        Lane q2;
        Memory memory;
        if (!gpu.lane(q2, D3D12_COMMAND_LIST_TYPE_DIRECT) || !pool.allocate(memory, A)) return row(pool, name, false, "setup failed");
        Placed x = place(gpu, memory.buffer(), false);
        TitleList l0, l1, l2;
        if (FAILED(x.hr) || !x.backing || !l0.begin(gpu) || !l1.begin(gpu) || !l2.begin(gpu))
            return row(pool, name, false, "slot 133 -> " + hex(x.hr));
        l0.read(x.backing.Get());
        l0.close();
        if (!gpu.wait(l0.submit())) return row(pool, name, false, "fence timeout");  // A uploaded and read
        l1.read(x.backing.Get(), 1);
        l1.close();
        l2.read(x.backing.Get(), 2);
        l2.close();
        const X2wLinearBufferTotals before = totals();
        gpu.hold();
        const UINT64 v1 = l1.submit();  // queue 1, gated: nothing to refresh (the title still holds A)
        write(memory.buffer(), r0, B, words);
        const UINT64 v2 = l2.submit_on(q2);  // its refresh (B) is ordered behind queue 1's read
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const bool q2_held = q2.fence->GetCompletedValue() < v2;
        const bool q1_pending = gpu.done->GetCompletedValue() < v1;
        gpu.open_gate();
        const bool q1_done = gpu.wait(v1), q2_done = wait_fence(q2.fence.Get(), v2);
        if (!q1_done || !q2_done) return row(pool, name, false, "fence timeout after the gate");
        const Tally t1 = tally(gpu.read_output(1).data(), r0 / 4, key), t2 = tally(gpu.read_output(2).data(), r0 / 4, key);
        refresh_info(x);
        const X2wLinearBufferTotals after = totals();
        const bool pass = q1_pending && q2_held && t1.count[A] == words && t2.count[B] == words && !x.info.stale &&
                          x.info.refresh_failures == 0 && after.producer_waits > before.producer_waits;
        row(pool, name, pass,
            std::string("queue 1 ") + (q1_pending ? "gated" : "NOT GATED") + " with a read of A; after the CPU wrote B, queue 2 " +
                (q2_held ? "waited for it" : "COMPLETED FIRST (the refresh did not wait)") + "; queue 1 read " + t1.text() +
                " (expected A), queue 2 read " + t2.text() + " (expected B); producer waits +" +
                std::to_string(after.producer_waits - before.producer_waits));
    }

    // xq.write: queue 1 waits through its wrapper (queue slot 17) on a title fence F, then submits a read of A
    // (nothing to refresh). The CPU writes B and queue 2 submits a read. Its refresh would have to wait for
    // queue 1, which sits behind F, and F may be reached only by later work (here the title signals it from
    // queue 2): there is no safe completion order, so the submission must be refused, with nothing of it
    // executed (its result slot keeps the fixture's P) and the refusal counted and logged. The title then
    // signals F through queue 2's wrapper (queue slot 16): queue 1 must complete and read A. The refused
    // recording, submitted again on queue 2 once queue 1 is done, must read B, with no further refusal.
    void case_xq_write(Pool& pool)
    {
        const char* name = "xq.write";
        Lane q2;
        Memory memory;
        ComPtr<IUnknown> fence;
        if (!gpu.lane(q2, D3D12_COMMAND_LIST_TYPE_DIRECT) || !pool.allocate(memory, A) ||
            FAILED(static_cast<HRESULT>(invoke(gpu.guest.Get(), 36, 0, D3D12_FENCE_FLAG_NONE, arg(&__uuidof(ID3D12Fence)),
                                               arg(fence.GetAddressOf())))) || !fence)
            return row(pool, name, false, "setup failed");
        Placed x = place(gpu, memory.buffer(), false);
        TitleList l0, l1, l2;
        if (FAILED(x.hr) || !x.backing || !l0.begin(gpu) || !l1.begin(gpu) || !l2.begin(gpu))
            return row(pool, name, false, "slot 133 -> " + hex(x.hr));
        l0.read(x.backing.Get());
        l0.close();
        if (!gpu.wait(l0.submit())) return row(pool, name, false, "fence timeout");  // A uploaded and read
        l1.read(x.backing.Get(), 1);
        l1.close();
        l2.read(x.backing.Get(), 2);
        l2.close();
        if (!gpu.fill_output(2, P)) return row(pool, name, false, "result slot could not be filled");
        const X2wLinearBufferTotals before = totals();
        if (FAILED(static_cast<HRESULT>(invoke(gpu.guest_queue.Get(), 17, arg(fence.Get()), 1))))
            return row(pool, name, false, "queue wait (slot 17) failed");
        const UINT64 v1 = l1.submit();  // queue 1, behind F
        write(memory.buffer(), r0, B, words);
        const UINT64 v2 = l2.submit_on(q2);  // needs a refresh of B: refused
        // The lane's own fence after the refused lists: queue 2 itself was never blocked.
        const bool q2_progress = wait_fence(q2.fence.Get(), v2);
        const bool q1_pending = gpu.done->GetCompletedValue() < v1;
        const X2wLinearBufferTotals mid = totals();
        const Tally refused = tally(gpu.read_output(2).data(), r0 / 4, key);
        const bool logged = log_has("refused: its refresh would have to wait for queue", "no safe completion order");
        // The title's ordinary Signal through queue 2's wrapper releases queue 1.
        const HRESULT signal_hr = static_cast<HRESULT>(invoke(q2.guest.Get(), 16, arg(fence.Get()), 1));
        const bool q1_done = gpu.wait(v1);
        if (!q1_done) return row(pool, name, false, "queue 1 did not complete after the title's Signal (slot 16) " + hex(signal_hr));
        const Tally t1 = tally(gpu.read_output(1).data(), r0 / 4, key);
        // A later, safe submission: the refused recording again, with no submission behind a title Wait.
        const UINT64 v3 = l2.submit_on(q2);
        if (!wait_fence(q2.fence.Get(), v3)) return row(pool, name, false, "fence timeout (queue 2, resubmission)");
        const Tally t2 = tally(gpu.read_output(2).data(), r0 / 4, key);
        refresh_info(x);
        const X2wLinearBufferTotals after = totals();
        const UINT64 refusals = mid.unsafe_refresh_refusals - before.unsafe_refresh_refusals;
        const bool pass = q1_pending && q2_progress && refusals == 1 && refused.count[P] == words && logged && SUCCEEDED(signal_hr) &&
                          t1.count[A] == words && t2.count[B] == words && after.unsafe_refresh_refusals == mid.unsafe_refresh_refusals &&
                          !x.info.stale && x.info.refresh_failures == 0;
        row(pool, name, pass,
            std::string("queue 1 ") + (q1_pending ? "behind a title Wait (slot 17)" : "NOT HELD by its title Wait") +
                " with a read of A; after the CPU wrote B, queue 2's read was " + (refusals == 1 ? "refused" : "NOT REFUSED") +
                " (refusals +" + std::to_string(refusals) + ", log line " + (logged ? "present" : "MISSING") + "), its slot " +
                refused.text() + " (expected P: nothing ran), queue 2 " + (q2_progress ? "progressed" : "STUCK") +
                "; the title's Signal on queue 2 (slot 16) -> " + hex(signal_hr) + ", queue 1 then read " + t1.text() +
                " (expected A); the refused recording resubmitted on queue 2 read " + t2.text() + " (expected B), refusals +" +
                std::to_string(after.unsafe_refresh_refusals - mid.unsafe_refresh_refusals) + (x.info.stale ? ", STALE" : ""));
    }

    // xq.race: xq.write with the title Wait and the read of A on queue 1 issued from two threads at once (the
    // review repro). Thread W calls slot 17 Wait(F, 1) on queue 1 and parks through the pause hook (point 3)
    // after the desktop Wait and before the wait is recorded. While it is parked, thread S submits the read of A
    // on queue 1; the fixture gives it up to 1 s to return before W resumes (S must not take a user value that
    // the wait would then be recorded after). Once both returned the CPU writes B and queue 2 submits a read:
    // queue 1's read is behind F on the desktop queue, so this refresh has no safe completion order and must be
    // refused, as in xq.write (refusals +1, its slot keeps P, queue 2's own fence is reached). The title's
    // Signal(F, 1) through queue 2's wrapper must succeed and release queue 1, which reads A; the refused
    // recording submitted again then reads B. Every wait is bounded: when queue 2 or queue 1 does not complete,
    // the fixture signals F from the CPU (the same fence) so both drain, and the row fails.
    void case_xq_race(Pool& pool)
    {
        const char* name = "xq.race";
        Lane q2;
        Memory memory;
        ComPtr<IUnknown> fence;
        if (!gpu.lane(q2, D3D12_COMMAND_LIST_TYPE_DIRECT) || !pool.allocate(memory, A) ||
            FAILED(static_cast<HRESULT>(invoke(gpu.guest.Get(), 36, 0, D3D12_FENCE_FLAG_NONE, arg(&__uuidof(ID3D12Fence)),
                                               arg(fence.GetAddressOf())))) || !fence)
            return row(pool, name, false, "setup failed");
        Placed x = place(gpu, memory.buffer(), false);
        TitleList l0, l1, l2;
        if (FAILED(x.hr) || !x.backing || !l0.begin(gpu) || !l1.begin(gpu) || !l2.begin(gpu))
            return row(pool, name, false, "slot 133 -> " + hex(x.hr));
        l0.read(x.backing.Get());
        l0.close();
        if (!gpu.wait(l0.submit())) return row(pool, name, false, "fence timeout");  // A uploaded and read
        l1.read(x.backing.Get(), 1);
        l1.close();
        l2.read(x.backing.Get(), 2);
        l2.close();
        if (!gpu.fill_output(1, P) || !gpu.fill_output(2, P)) return row(pool, name, false, "result slot could not be filled");
        HANDLE parked = CreateEventW(nullptr, TRUE, FALSE, nullptr), go = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        HANDLE s_returned = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const auto close_events = [&] {
            x2w_d3d12_linear_buffer_test_pause(3, nullptr, nullptr);  // disarm, if no wait reached it
            for (HANDLE h : {parked, go, s_returned})
                if (h) CloseHandle(h);
        };
        if (!parked || !go || !s_returned || !x2w_d3d12_linear_buffer_test_pause(3, parked, go)) {
            close_events();
            return row(pool, name, false, "pause hook not available");
        }
        const X2wLinearBufferTotals before = totals();
        HRESULT wait_hr = E_FAIL;
        std::thread tw([&] { wait_hr = static_cast<HRESULT>(invoke(gpu.guest_queue.Get(), 17, arg(fence.Get()), 1)); });
        const bool w_parked = WaitForSingleObject(parked, 5000) == WAIT_OBJECT_0;
        UINT64 v1 = 0;
        std::thread ts([&] {
            v1 = l1.submit();  // queue 1: the unchanged read of A, behind F on the desktop queue
            SetEvent(s_returned);
        });
        // With the order lock S cannot return while W is parked; without it, it returns at once.
        const bool s_returned_while_parked = WaitForSingleObject(s_returned, 1000) == WAIT_OBJECT_0;
        SetEvent(go);
        tw.join();
        ts.join();  // bounded: W's park ends after 20 s at most, and S waits only for W
        close_events();
        write(memory.buffer(), r0, B, words);
        const UINT64 v2 = l2.submit_on(q2);  // needs a refresh of B, which would wait for queue 1: must be refused
        const bool q2_progress = poll_fence(q2.fence.Get(), v2, 3000);
        const bool q1_pending = gpu.done->GetCompletedValue() < v1;
        const X2wLinearBufferTotals mid = totals();
        const Tally refused = tally(gpu.read_output(2).data(), r0 / 4, key);
        // The title's Signal through queue 2's wrapper releases queue 1.
        const HRESULT signal_hr = static_cast<HRESULT>(invoke(q2.guest.Get(), 16, arg(fence.Get()), 1));
        const bool q1_done = poll_fence(gpu.done.Get(), v1, 3000);
        const UINT64 refusals = mid.unsafe_refresh_refusals - before.unsafe_refresh_refusals;
        std::string text = std::string("W ") + (w_parked ? "parked" : "DID NOT PARK") + " between its desktop Wait and its record (slot 17 -> " +
                           hex(wait_hr) + "); S's read of A on queue 1 " +
                           (s_returned_while_parked ? "RETURNED WHILE W WAS PARKED" : "waited for W") + "; after the CPU wrote B, queue 2's read was " +
                           (refusals == 1 ? "refused" : "NOT REFUSED") + " (refusals +" + std::to_string(refusals) + "), its slot " +
                           refused.text() + " (expected P), queue 2 " + (q2_progress ? "progressed" : "STUCK") +
                           "; the title's Signal on queue 2 (slot 16) -> " + hex(signal_hr) + ", queue 1 " +
                           (q1_done ? "completed" : "STILL PENDING");
        if (!q2_progress || !q1_done) {
            // Cleanup: the CPU reaches F (fence slot 10), so both queues drain before the next case.
            invoke(fence.Get(), 10, 1);
            const bool drained = wait_fence(gpu.done.Get(), v1) && wait_fence(q2.fence.Get(), v2);
            const Tally t1 = tally(gpu.read_output(1).data(), r0 / 4, key), t2 = tally(gpu.read_output(2).data(), r0 / 4, key);
            return row(pool, name, false,
                       text + "; after a CPU Signal of F both " + (drained ? "drained" : "STILL DID NOT DRAIN") + ": queue 1 read " + t1.text() +
                           ", queue 2 read " + t2.text());
        }
        const Tally t1 = tally(gpu.read_output(1).data(), r0 / 4, key);
        // A later, safe submission: the refused recording again, with no submission behind a title Wait.
        const UINT64 v3 = l2.submit_on(q2);
        if (!wait_fence(q2.fence.Get(), v3)) return row(pool, name, false, text + "; fence timeout (queue 2, resubmission)");
        const Tally t2 = tally(gpu.read_output(2).data(), r0 / 4, key);
        refresh_info(x);
        const X2wLinearBufferTotals after = totals();
        const bool pass = w_parked && SUCCEEDED(wait_hr) && q1_pending && q2_progress && refusals == 1 && refused.count[P] == words &&
                          SUCCEEDED(signal_hr) && t1.count[A] == words && t2.count[B] == words &&
                          after.unsafe_refresh_refusals == mid.unsafe_refresh_refusals && !x.info.stale && x.info.refresh_failures == 0;
        row(pool, name, pass,
            text + " and read " + t1.text() + " (expected A); the refused recording resubmitted on queue 2 read " + t2.text() +
                " (expected B), refusals +" + std::to_string(after.unsafe_refresh_refusals - mid.unsafe_refresh_refusals) +
                (x.info.stale ? ", STALE" : ""));
    }

    // order: two threads' refreshes of one buffer, driven through the pause hook. Thread A stages C and parks
    // before its copy is executed; the CPU writes D; thread B stages D, executes its copy first (the lower
    // internal value) and parks before publishing; A then executes (the higher value: the backing ends with C)
    // and publishes; B publishes last. The next submission must refresh D, which the GPU then reads.
    void case_order(Pool& pool)
    {
        const char* name = "order";
        Lane q2;
        Memory memory;
        if (!gpu.lane(q2, D3D12_COMMAND_LIST_TYPE_DIRECT) || !pool.allocate(memory, A)) return row(pool, name, false, "setup failed");
        Placed x = place(gpu, memory.buffer(), false);
        TitleList l0, la, lb, lc;
        if (FAILED(x.hr) || !x.backing || !l0.begin(gpu) || !la.begin(gpu) || !lb.begin(gpu) || !lc.begin(gpu))
            return row(pool, name, false, "slot 133 -> " + hex(x.hr));
        l0.read(x.backing.Get());
        l0.close();
        if (!gpu.wait(l0.submit())) return row(pool, name, false, "fence timeout");  // A uploaded
        la.read(x.backing.Get(), 1);
        la.close();
        lb.read(x.backing.Get(), 2);
        lb.close();
        lc.read(x.backing.Get(), 3);
        lc.close();
        HANDLE staged = CreateEventW(nullptr, TRUE, FALSE, nullptr), staged_go = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        HANDLE executed = CreateEventW(nullptr, TRUE, FALSE, nullptr), executed_go = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const auto close_events = [&] {
            x2w_d3d12_linear_buffer_test_pause(1, nullptr, nullptr);  // disarm what no refresh reached
            x2w_d3d12_linear_buffer_test_pause(2, nullptr, nullptr);
            for (HANDLE h : {staged, staged_go, executed, executed_go})
                if (h) CloseHandle(h);
        };
        if (!staged || !staged_go || !executed || !executed_go || !x2w_d3d12_linear_buffer_test_pause(1, staged, staged_go)) {
            close_events();
            return row(pool, name, false, "pause hook not available");
        }
        const X2wLinearBufferTotals before = totals();
        write(memory.buffer(), r0, C, words);
        UINT64 va = 0, vb = 0;
        std::thread ta([&] { va = la.submit(); });
        const bool a_parked = WaitForSingleObject(staged, 5000) == WAIT_OBJECT_0;
        write(memory.buffer(), r0, D, words);
        const bool armed = x2w_d3d12_linear_buffer_test_pause(2, executed, executed_go);
        std::thread tb([&] { vb = lb.submit_on(q2); });
        const bool b_parked = armed && WaitForSingleObject(executed, 5000) == WAIT_OBJECT_0;
        SetEvent(staged_go);  // A executes after B and publishes first
        ta.join();
        SetEvent(executed_go);  // then B publishes
        tb.join();
        close_events();
        if (!gpu.wait(va) || !wait_fence(q2.fence.Get(), vb)) return row(pool, name, false, "fence timeout");
        const UINT64 vc = lc.submit();
        if (!gpu.wait(vc)) return row(pool, name, false, "fence timeout");
        const Tally t = tally(gpu.read_output(3).data(), r0 / 4, key);
        refresh_info(x);
        const X2wLinearBufferTotals after = totals();
        const bool pass = a_parked && b_parked && t.count[D] == words && after.reordered_publishes == before.reordered_publishes + 1 &&
                          !x.info.stale && x.info.refresh_failures == 0;
        row(pool, name, pass,
            std::string("thread A staged C ") + (a_parked ? "and parked" : "BUT DID NOT PARK") + ", thread B staged D and executed first " +
                (b_parked ? "and parked" : "BUT DID NOT PARK") + "; A executed and published, then B: the next submission read " + t.text() +
                " (expected D: the older publish dropped the upload record); reordered publishes +" +
                std::to_string(after.reordered_publishes - before.reordered_publishes));
    }

    // retire: a submission that reads Y snapshots both changed buffers X and Y and parks before staging; the
    // title releases X and frees its pages; the submission resumes. X must be skipped as unreadable (no
    // fault, no failed refresh); Y is refreshed and read.
    void case_retire(Pool& pool)
    {
        const char* name = "retire";
        Memory mx, my;
        // A private X gets an allocation of its own: the pool's range is pinned by the heap that the default
        // backing opened on it (case b), so its pages cannot be decommitted.
        if (!pool.xmem) {
            mx.cpu = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, alloc_pages * page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            if (mx.cpu) {
                write(mx.cpu, 0, neighbour, buffer_offset / 4);
                write(mx.buffer(), 0, A, buffer_bytes / 4);
            }
        }
        if ((pool.xmem ? !pool.allocate(mx, A) : !mx.cpu) || !pool.allocate(my, A)) return row(pool, name, false, "allocation failed");
        Placed x = place(gpu, mx.buffer(), false), y = place(gpu, my.buffer(), false);
        TitleList l0, la;
        if (FAILED(x.hr) || FAILED(y.hr) || !x.backing || !y.backing || !l0.begin(gpu) || !la.begin(gpu))
            return row(pool, name, false, "slot 133 -> " + hex(x.hr) + "/" + hex(y.hr));
        l0.read(x.backing.Get());
        l0.close();
        if (!gpu.wait(l0.submit())) return row(pool, name, false, "fence timeout");  // both uploaded
        la.read(y.backing.Get(), 1);
        la.close();
        HANDLE parked = CreateEventW(nullptr, TRUE, FALSE, nullptr), go = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!parked || !go || !x2w_d3d12_linear_buffer_test_pause(0, parked, go)) {
            if (parked) CloseHandle(parked);
            if (go) CloseHandle(go);
            return row(pool, name, false, "pause hook not available");
        }
        write(mx.buffer(), r0, B, words);
        write(my.buffer(), r0, C, words);
        const X2wLinearBufferTotals before = totals();
        UINT64 va = 0;
        std::thread ta([&] { va = la.submit(); });
        const bool was_parked = WaitForSingleObject(parked, 5000) == WAIT_OBJECT_0;
        auto* x_wrapper = x.wrapper.Get();
        x.wrapper.Reset();  // release(): retire X
        const bool freed = pool.xmem ? pool.release(mx) : VirtualFree(mx.cpu, 0, MEM_RELEASE) != FALSE;
        SetEvent(go);
        ta.join();
        x2w_d3d12_linear_buffer_test_pause(0, nullptr, nullptr);  // disarm, if no refresh reached it
        CloseHandle(parked);
        CloseHandle(go);
        if (!gpu.wait(va)) return row(pool, name, false, "fence timeout");
        const Tally t = tally(gpu.read_output(1).data(), r0 / 4, key);
        X2wLinearBufferInfo xi{};
        x2w_d3d12_linear_buffer_info(x_wrapper, &xi);
        refresh_info(y);
        const X2wLinearBufferTotals after = totals();
        const bool pass = was_parked && freed && t.count[C] == words && after.unreadable == before.unreadable + 1 &&
                          after.copy_faults == before.copy_faults && xi.refresh_failures == 0 && !y.info.stale;
        row(pool, name, pass,
            std::string("parked after the snapshot ") + (was_parked ? "" : "(DID NOT PARK) ") + "while the title released X and " +
                (freed ? "freed" : "COULD NOT FREE") + " its pages: unreadable +" + std::to_string(after.unreadable - before.unreadable) +
                " (expected 1), copy faults +" + std::to_string(after.copy_faults - before.copy_faults) + " (expected 0), X refresh failures " +
                std::to_string(xi.refresh_failures) + "; Y read " + t.text());
    }

    // lease: with the mesh XMem lease path on, an XMem placement with console flags 0 is the lease path's:
    // slot 133 keeps the default backing (not aliased) and records the deferral; a private one is copied.
    void case_lease(Pool& pool)
    {
        Memory memory;
        if (!pool.allocate(memory, A)) return row(pool, "lease", false, "allocation failed");
        Placed p = place(gpu, memory.buffer(), false);
        TitleList list;
        if (FAILED(p.hr) || !p.backing || !list.begin(gpu)) return row(pool, "lease", false, "slot 133 -> " + hex(p.hr));
        list.read(p.backing.Get());
        list.close();
        write(memory.buffer(), r0, B, words);
        if (!gpu.wait(list.submit())) return row(pool, "lease", false, "fence timeout");
        const Tally t = tally(gpu.read_output().data(), r0 / 4, key);
        refresh_info(p);
        std::string text = "slot 133 -> " + hex(p.hr) + ", " + backing_text(p) + "; GPU read " + t.text();
        bool pass;
        if (pool.xmem) {
            pass = kept_ok(p, HRESULT_FROM_WIN32(ERROR_BUSY), text) && t.zero == words;
            text += " (deferred to the XMem lease path; old behaviour: not aliased)";
        } else {
            pass = p.linear && p.info.path == 2 && t.count[B] == words;
        }
        row(pool, "lease", pass, text);
    }

    // budget (share, copy): X2W_D3D12_BUDGET counts a shared buffer as placed memory on its heap and a
    // copied one as a committed buffer, live while the title holds it and retained after its release.
    void case_budget(Pool& pool)
    {
        Memory memory;
        x2w_budget::Snapshot before{}, held{}, released{};
        if (!pool.allocate(memory, A) || !x2w_d3d12_budget_snapshot(&before, sizeof(before)))
            return row(pool, "budget", false, "setup failed or X2W_D3D12_BUDGET not on");
        Placed p = place(gpu, memory.buffer(), false);
        TitleList list;
        if (FAILED(p.hr) || !p.backing || !list.begin(gpu)) return row(pool, "budget", false, "slot 133 -> " + hex(p.hr));
        list.read(p.backing.Get());
        list.close();
        if (!gpu.wait(list.submit())) return row(pool, "budget", false, "fence timeout");
        x2w_d3d12_budget_snapshot(&held, sizeof(held));
        const bool shared = p.linear && p.info.path == 1;
        const int category = shared ? x2w_budget::placed : x2w_budget::buffer;
        p.wrapper.Reset();  // the title's release
        x2w_d3d12_budget_snapshot(&released, sizeof(released));
        const auto live = [&](const x2w_budget::Snapshot& snap) { return snap.live[category].objects; };
        const auto kept = [&](const x2w_budget::Snapshot& snap) { return snap.retained[category].objects; };
        const bool heap_live = !shared || held.live[x2w_budget::heap].objects >= 1;
        const bool pass = p.linear && live(held) == live(before) + 1 && live(released) == live(before) &&
                          kept(released) == kept(before) + 1 && heap_live && released.unmeasured == before.unmeasured;
        row(pool, "budget", pass,
            std::string(shared ? "shared: placed" : "copied: buffer") + " objects live " + std::to_string(live(before)) + " -> " +
                std::to_string(live(held)) + " -> " + std::to_string(live(released)) + " after release, retained +" +
                std::to_string(kept(released) - kept(before)) + (shared ? ", heap live " + std::to_string(held.live[x2w_budget::heap].objects) : "") +
                ", unmeasured +" + std::to_string(released.unmeasured - before.unmeasured));
    }

    void case_xq(Pool& pool)
    {
        Lane q2;
        Memory memory;
        ComPtr<ID3D12Fence> title_fence;
        if (!gpu.lane(q2, D3D12_COMMAND_LIST_TYPE_DIRECT) || !pool.allocate(memory, A) ||
            FAILED(gpu.desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&title_fence))))
            return row(pool, "xq", false, "setup failed");
        Placed x = place(gpu, memory.buffer(), false);
        TitleList l0, l1, l2;
        if (FAILED(x.hr) || !x.backing || !l0.begin(gpu) || !l1.begin(gpu) || !l2.begin(gpu))
            return row(pool, "xq", false, "slot 133 -> " + hex(x.hr));
        l0.read(x.backing.Get());
        l0.close();
        if (!gpu.wait(l0.submit())) return row(pool, "xq", false, "fence timeout");
        l1.read(x.backing.Get(), 1);
        l1.close();
        l2.read(x.backing.Get(), 2);
        l2.close();
        write(memory.buffer(), r0, B, words);
        gpu.queue->Wait(title_fence.Get(), 1);  // the title's own cross-queue dependency
        const UINT64 v1 = l1.submit();
        const UINT64 v2 = l2.submit_on(q2);
        q2.native->Signal(title_fence.Get(), 1);
        const bool q2_done = wait_fence(q2.fence.Get(), v2), q1_done = gpu.wait(v1);
        const Tally t1 = tally(gpu.read_output(1).data(), r0 / 4, key), t2 = tally(gpu.read_output(2).data(), r0 / 4, key);
        const bool pass = q1_done && q2_done && t1.count[B] == words && t2.count[B] == words;
        row(pool, "xq", pass,
            std::string("queue 1 waits on a fence queue 2 signals after its list: queue 2 ") + (q2_done ? "completed" : "DID NOT COMPLETE") +
                " and read " + t2.text() + ", queue 1 " + (q1_done ? "completed" : "DID NOT COMPLETE (deadlock?)") + " and read " + t1.text());
    }

    // contention: nothing waits behind a submission that waits for the only staging buffer.
    void case_contention(Pool& pool)
    {
        using clock = std::chrono::steady_clock;
        const auto ms = [](clock::duration d) { return static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(d).count()); };
        Lane second;
        Memory mx, mw;
        if (!gpu.lane(second, D3D12_COMMAND_LIST_TYPE_DIRECT) || !pool.allocate(mx, A))
            return row(pool, "contention", false, "setup failed");
        Placed x = place(gpu, mx.buffer(), false);
        TitleList l0;
        if (FAILED(x.hr) || !x.backing || !l0.begin(gpu)) return row(pool, "contention", false, "slot 133 -> " + hex(x.hr));
        l0.read(x.backing.Get());
        l0.close();
        if (!gpu.wait(l0.submit())) return row(pool, "contention", false, "fence timeout");  // X uploaded: the staging buffer exists
        if (!pool.allocate(mw, D)) return row(pool, "contention", false, "allocation failed");
        Placed w = place(gpu, mw.buffer(), false);
        TitleList lw;
        if (FAILED(w.hr) || !w.backing || !lw.begin(gpu)) return row(pool, "contention", false, "slot 133 -> " + hex(w.hr));
        lw.read(w.backing.Get());
        lw.close();
        if (!wait_fence(second.fence.Get(), lw.submit_on(second))) return row(pool, "contention", false, "fence timeout");
        const X2wLinearBufferTotals before = totals();

        // S1 reads X behind the title gate; its refresh (B) is held on the internal queue and holds the
        // only staging buffer.
        gpu.hold();
        if (!gpu.hold_internal()) return row(pool, "contention", false, "internal refresh queue not available");
        write(mx.buffer(), r0, B, words);
        TitleList s1, s2, probe;
        if (!s1.begin(gpu) || !s2.begin(gpu) || !probe.begin(gpu)) return row(pool, "contention", false, "list");
        s1.read(x.backing.Get(), 1);
        s1.close();
        s1.submit();
        // S2's refresh (C) must wait for that staging buffer's copy: on another thread.
        write(mx.buffer(), r0, C, words);
        s2.read(x.backing.Get(), 2);
        s2.close();
        std::atomic<bool> done{false};
        UINT64 s2_value = 0;
        const auto t1_start = clock::now();
        clock::duration t1_time{};
        std::thread waiter([&] {
            s2_value = s2.submit();
            t1_time = clock::now() - t1_start;
            done = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const bool waiting_before = !done;
        // Probes, each through mutex_: the info export, the title's release of X, a submission on another queue.
        auto* x_wrapper = x.wrapper.Get();
        X2wLinearBufferInfo probe_info{};
        auto t = clock::now();
        x2w_d3d12_linear_buffer_info(w.wrapper.Get(), &probe_info);
        const auto info_time = clock::now() - t;
        t = clock::now();
        x.wrapper.Reset();  // release(): retire X
        const auto release_time = clock::now() - t;
        probe.read(w.backing.Get(), 3);
        probe.close();
        t = clock::now();
        const bool probe_done = wait_fence(second.fence.Get(), probe.submit_on(second));
        const auto submit_time = clock::now() - t;
        const bool waiting_after = !done;
        const Tally w_read = tally(gpu.read_output(3).data(), r0 / 4, key);
        // Let B land and the gated reads run. C's copy waits for S1, the earlier submission (the producer
        // wait): S1 reads B, S2 then reads C, the CPU's last write.
        gpu.open_internal();
        waiter.join();
        gpu.open_gate();
        const bool drained = gpu.drain_internal();
        if (!drained || !gpu.wait(s2_value)) return row(pool, "contention", false, "fence timeout after the gates");
        const Tally r1 = tally(gpu.read_output(1).data(), r0 / 4, key), r2 = tally(gpu.read_output(2).data(), r0 / 4, key);
        X2wLinearBufferInfo xi{};
        x2w_d3d12_linear_buffer_info(x_wrapper, &xi);
        const X2wLinearBufferTotals after = totals();
        const bool reused = after.staging_created == before.staging_created && after.staging_reused >= before.staging_reused + 2;
        const bool pass = waiting_before && waiting_after && probe_done && w_read.count[D] == words && r1.count[B] == words &&
                          r2.count[C] == words && !xi.stale && xi.refresh_failures == 0 && reused &&
                          ms(t1_time) >= ms(info_time + release_time + submit_time);
        char text[400];
        std::snprintf(text, sizeof(text),
                      "while S2's refresh waited (%s before, %s after the probes): info %lld ms, release %lld ms, other-queue submit "
                      "%lld ms (read %s); S2 returned after %lld ms; S1 read %s, S2 read %s; staging created %llu -> %llu, "
                      "refresh failures %llu",
                      waiting_before ? "waiting" : "NOT WAITING", waiting_after ? "waiting" : "DONE", ms(info_time), ms(release_time),
                      ms(submit_time), w_read.text().c_str(), ms(t1_time), r1.text().c_str(), r2.text().c_str(),
                      static_cast<unsigned long long>(before.staging_created), static_cast<unsigned long long>(after.staging_created),
                      static_cast<unsigned long long>(xi.refresh_failures));
        row(pool, "contention", pass, text);
    }

    void case_d1(Pool& pool)
    {
        if (path == Path::default_backing || path == Path::untracked) return;
        Memory memory;
        if (!pool.allocate(memory, A)) return row(pool, "d1", false, "allocation failed");
        Placed p = place(gpu, memory.buffer(), false);
        TitleList first, bypass, through;
        if (FAILED(p.hr) || !first.begin(gpu) || !bypass.begin(gpu) || !through.begin(gpu))
            return row(pool, "d1", false, "slot 133 -> " + hex(p.hr));
        first.read(p.backing.Get());
        first.close();
        if (!gpu.wait(first.submit())) return row(pool, "d1", false, "fence timeout");
        write(memory.buffer(), r0, B, words);
        bypass.read(p.backing.Get());
        bypass.close();
        if (!gpu.wait(bypass.submit_bypassing())) return row(pool, "d1", false, "fence timeout");
        const Tally missing = tally(gpu.read_output().data(), r0 / 4, key);
        through.read(p.backing.Get());
        through.close();
        if (!gpu.wait(through.submit())) return row(pool, "d1", false, "fence timeout");
        const Tally refreshed = tally(gpu.read_output().data(), r0 / 4, key);
        refresh_info(p);
        const bool shared = !copied(pool);
        const bool pass = (shared ? missing.count[B] == words : missing.count[A] == words) && refreshed.count[B] == words &&
                          p.linear && p.info.path == (shared ? 1u : 2u);
        row(pool, "d1", pass, std::string("past the wrapper (no refresh) ") + missing.text() +
                                  (shared ? " (shared: no refresh needed)" : " (stale A expected: the oracle sees the missing refresh)") +
                                  "; through the wrapper " + refreshed.text());
    }
};
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::puts("usage: linear_buffer_backing_check <warp|hardware> <default|share|copy|refusal|untracked|stale|contention> [--mutate <name>]");
        return 2;
    }
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    if (!warp && std::strcmp(argv[1], "hardware") != 0) return 2;
    Path path;
    const std::string p = argv[2];
    if (p == "default") path = Path::default_backing;
    else if (p == "share") path = Path::share;
    else if (p == "copy") path = Path::copy;
    else if (p == "refusal") path = Path::refusal;
    else if (p == "untracked") path = Path::untracked;
    else if (p == "stale") path = Path::stale;
    else if (p == "contention") path = Path::contention;
    else if (p == "lease") path = Path::lease;
    else return 2;
    std::string mutant;
    if (argc == 5 && std::strcmp(argv[3], "--mutate") == 0) mutant = argv[4];
    else if (argc != 3) return 2;

    // The switches are read once, by the first call that needs them: set them before any.
    SetEnvironmentVariableW(L"X2W_D3D12_PERMISSIVE", L"1");
    SetEnvironmentVariableW(L"X2W_D3D12_BUDGET", L"1");  // budget accounting (case budget); changes no backing
    SetEnvironmentVariableW(L"X2W_D3D12_LIST_NOOP", L"1");
    SetEnvironmentVariableW(L"X2W_D3D12_RENDER", L"1");
    SetEnvironmentVariableW(L"X2W_XMEM_SHARED_VIEWS", L"1");
    SetEnvironmentVariableW(L"X2W_XMEM_JOIN_CALLS", nullptr);  // no joins across mapping calls (the default)
    SetEnvironmentVariableW(L"X2W_D3D12_LINEAR_BUFFERS", path == Path::default_backing ? nullptr
                                                         : path == Path::copy || path == Path::untracked || path == Path::stale || path == Path::contention ||
                                                                 path == Path::lease ? L"copy" : L"1");
    SetEnvironmentVariableW(L"X2W_TEST_EXISTING_HEAPS_UNSUPPORTED", path == Path::refusal ? L"1" : nullptr);
    SetEnvironmentVariableW(L"X2W_TEST_SUBMIT_FENCE_FAIL", path == Path::untracked ? L"1" : nullptr);
    SetEnvironmentVariableW(L"X2W_TEST_LINEAR_REFRESH_FAIL", path == Path::stale ? L"3" : nullptr);
    SetEnvironmentVariableW(L"X2W_TEST_LINEAR_STAGING_LIMIT", path == Path::contention ? L"1" : nullptr);
    SetEnvironmentVariableW(L"X2W_MESH86_XMEM_SRV", path == Path::lease ? L"1" : nullptr);
    SetEnvironmentVariableA("X2W_TEST_LINEAR_BUFFER_MUTANT", mutant.empty() ? nullptr : mutant.c_str());
    {
        // The DLL's log lines go to linear_buffer_backing_<path>_<adapter>[_<mutant>].log next to this executable.
        wchar_t module[MAX_PATH]{};
        const DWORD n = GetModuleFileNameW(nullptr, module, MAX_PATH);
        if (!n || n >= MAX_PATH) return 1;
        log_path.assign(module, n);
        log_path.resize(log_path.find_last_of(L"\\/") + 1);
        const std::string name = std::string("linear_buffer_backing_") + path_name(path) + "_" + argv[1] +
                                 (mutant.empty() ? "" : "_" + mutant) + ".log";
        log_path.append(name.begin(), name.end());
        DeleteFileW(log_path.c_str());
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
    }

    char debug_env[4]{};
    const bool debug_layer = GetEnvironmentVariableA("X2W_TEST_DEBUG_LAYER", debug_env, sizeof(debug_env)) && debug_env[0] == '1';
    if (debug_layer) {
        ComPtr<ID3D12Debug> debug;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            std::puts("X2W_TEST_DEBUG_LAYER=1 but the debug layer is not installed");
            return 1;
        }
        debug->EnableDebugLayer();
    }

    Gpu gpu;
    static ComPtr<ID3D12Device>* desktop_out = &gpu.desktop;
    static DXGI_ADAPTER_DESC1 adapter_desc{};
    static LARGE_INTEGER driver{};
    const auto create = [](IUnknown* adapter, REFIID iid, void** out) -> HRESULT {
        ComPtr<IDXGIAdapter1> a;
        if (SUCCEEDED(adapter->QueryInterface(IID_PPV_ARGS(&a)))) {
            a->GetDesc1(&adapter_desc);
            a->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver);
        }
        const HRESULT hr = D3D12XboxCreateDevice(adapter, nullptr, IID_PPV_ARGS(desktop_out->ReleaseAndGetAddressOf()));
        if (FAILED(hr)) return hr;
        unsigned char parameters[64]{};
        return D3D12XboxCreateDevice(adapter, parameters, iid, out);
    };
    bool unmatched = false;
    gpu.guest = create_device(warp, unmatched, create);
    if (!gpu.guest || !gpu.desktop) {
        std::puts(warp ? "WARP device not created" : "no D3D12 hardware adapter");
        return warp || unmatched ? 1 : 77;
    }
    if (debug_layer) gpu.desktop.As(&gpu.info);
    D3D12_FEATURE_DATA_EXISTING_HEAPS heaps{};
    const HRESULT heaps_hr = gpu.desktop->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &heaps, sizeof(heaps));
    std::printf("adapter %04x:%04x %ls, driver %u.%u.%u.%u, D3D12_FEATURE_EXISTING_HEAPS %s supported=%d, debug layer %s, path %s%s%s\n",
                adapter_desc.VendorId, adapter_desc.DeviceId, adapter_desc.Description, HIWORD(driver.HighPart),
                LOWORD(driver.HighPart), HIWORD(driver.LowPart), LOWORD(driver.LowPart), hex(heaps_hr).c_str(),
                heaps.Supported ? 1 : 0, debug_layer ? "on" : "off", path_name(path), mutant.empty() ? "" : ", mutant ",
                mutant.c_str());
    if (!gpu.init()) {
        std::puts("FAIL: fixture setup (queue, fences, pipeline)");
        return 1;
    }
    if (path == Path::share && (FAILED(heaps_hr) || !heaps.Supported)) {
        std::puts("SKIP share: the adapter reports no existing-heap support; the refusal path covers it");
        return 77;
    }

    Run run{gpu, path, SUCCEEDED(heaps_hr) && heaps.Supported != FALSE};
    for (const bool xmem : {true, false}) {
        Pool pool(xmem);
        if (!pool.init()) {
            run.row(pool, "pool", false, "title memory could not be set up");
            continue;
        }
        if (path == Path::stale) {
            run.case_stale(pool);
            continue;
        }
        if (path == Path::contention) {
            run.case_contention(pool);
            continue;
        }
        if (path == Path::lease) {
            run.case_lease(pool);
            continue;
        }
        run.case_a(pool);
        if (path == Path::share && xmem) run.case_a(pool, true);
        run.case_b(pool, false);
        run.case_c(pool);
        run.case_d1(pool);
        if (path != Path::default_backing) run.case_b(pool, true);
        if (path == Path::copy) {
            run.case_queues(pool);
            run.case_gate2q(pool, false);
            run.case_gate2q(pool, true);
            run.case_xq(pool);
            run.case_budget(pool);
            run.case_gpuw(pool);
            run.case_gpuw_discard(pool);
            run.case_war(pool);
            run.case_xq_write(pool);
            run.case_xq_race(pool);
            run.case_order(pool);
            run.case_retire(pool);  // last: it frees the pages of one buffer
        }
        run.probe_view_pin(pool);
        if (path == Path::share) run.case_budget(pool);
    }
    const X2wLinearBufferTotals t = totals();
    std::printf("totals: placements %llu (shared %llu, copied %llu, kept default %llu), feature queries %llu, heap opens %llu "
                "(failed %llu, cache hits %llu), placements failed %llu, submissions %llu, refreshes %llu, bytes copied %llu, "
                "staging created %llu reused %llu, unreadable %llu, refresh failures %llu, consumer waits %llu, producer waits %llu "
                "(unsafe refreshes refused %llu), gpu written %llu, reordered publishes %llu, copy faults %llu, last refusal %s, last refresh failure %s\n",
                static_cast<unsigned long long>(t.placements), static_cast<unsigned long long>(t.shared),
                static_cast<unsigned long long>(t.copied), static_cast<unsigned long long>(t.kept_default),
                static_cast<unsigned long long>(t.feature_queries), static_cast<unsigned long long>(t.heap_opens),
                static_cast<unsigned long long>(t.heap_open_failures), static_cast<unsigned long long>(t.heap_cache_hits),
                static_cast<unsigned long long>(t.place_failures), static_cast<unsigned long long>(t.submissions),
                static_cast<unsigned long long>(t.refreshes), static_cast<unsigned long long>(t.bytes_copied),
                static_cast<unsigned long long>(t.staging_created), static_cast<unsigned long long>(t.staging_reused),
                static_cast<unsigned long long>(t.unreadable), static_cast<unsigned long long>(t.refresh_failures),
                static_cast<unsigned long long>(t.consumer_waits), static_cast<unsigned long long>(t.producer_waits),
                static_cast<unsigned long long>(t.unsafe_refresh_refusals), static_cast<unsigned long long>(t.gpu_written),
                static_cast<unsigned long long>(t.reordered_publishes), static_cast<unsigned long long>(t.copy_faults),
                hex(t.last_refusal_hr).c_str(), hex(t.last_refresh_hr).c_str());
    {
        Pool none(false);
        if (path == Path::default_backing) {
            // Opt-in off: no linear-buffer bookkeeping at all.
            const bool untouched = !t.placements && !t.submissions && !t.feature_queries && !t.heap_opens;
            run.row(none, "off", untouched, untouched ? "no linear-buffer code ran (all totals zero)" : "LINEAR-BUFFER CODE RAN");
        } else if (path == Path::stale) {
            run.row(none, "totals", t.refresh_failures == 2 && t.last_refresh_hr == E_OUTOFMEMORY && t.kept_default == 0,
                    "refresh failures " + std::to_string(t.refresh_failures) + " (expected 2, one per pool), last " + hex(t.last_refresh_hr));
        }
        if (path != Path::default_backing && path != Path::stale && path != Path::contention) {
            run.row(none, "kept", t.kept_default == run.kept_rows,
                    "totals report " + std::to_string(t.kept_default) + " kept default backings for " + std::to_string(run.kept_rows) +
                        " refused placements, last refusal " + hex(t.last_refusal_hr));
        }
        if (path == Path::refusal) {
            run.row(none, "gate", t.feature_queries == 1 && t.heap_opens == 0,
                    "feature queries " + std::to_string(t.feature_queries) + ", heap opens " + std::to_string(t.heap_opens) +
                        " (expected 1 and 0: no open is attempted when D3D12_FEATURE_EXISTING_HEAPS reports no support)");
        }
    }
    const HRESULT removed = gpu.desktop->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        std::printf("FAIL: device removed %s\n", hex(removed).c_str());
        ++run.failures;
    }
    const unsigned errors = gpu.debug_errors();
    if (errors) {
        std::printf("FAIL: %u debug-layer errors\n", errors);
        run.failures += errors;
    }

    if (!mutant.empty()) {
        const char* target = mutant == "skip-refresh" ? "a" : mutant == "reuse-early" ? "c" : mutant == "copy-gpu-writes" ? "b"
                           : mutant == "ignore-feature" ? "gate" : mutant == "refusal-silent" ? "b"
                           : mutant == "hold-lock" ? "contention" : mutant == "skip-consumer-wait" ? "gate2q.held"
                           : mutant == "refresh-gpu-written" ? "gpuw" : mutant == "gpuw-at-record" ? "gpuw.discard"
                           : mutant == "skip-producer-wait" ? "war" : mutant == "publish-out-of-order" ? "order"
                           : mutant == "copy-after-retire" ? "retire" : mutant == "accept-unsafe-refresh" ? "xq.write"
                           : mutant == "unlocked-title-wait" ? "xq.race" : nullptr;
        if (!target) return 2;
        unsigned targeted = 0, failed = 0;
        for (const auto& [name, pass] : run.results)
            if (name == target) {
                ++targeted;
                failed += !pass;
            }
        std::printf("mutant %s: %u of %u targeted '%s' rows failed\n", mutant.c_str(), failed, targeted, target);
        return targeted && failed == targeted ? 1 : 0;
    }
    std::printf("linear_buffer_backing %s %s: %s (%u failed)\n", argv[1], path_name(path), run.failures ? "FAIL" : "PASS", run.failures);
    return run.failures ? 1 : 0;
}
