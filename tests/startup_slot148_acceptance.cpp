// Acceptance client for command-list slot 148, the upload allocation that the title's
// acceleration-structure builder requests (Refs #1758, #2216). It drives the actual console list and
// queue of the d3d12_xs stand-in and uses whatever slot 148 the build under test installs. It
// supplies no allocator of its own and keeps no reference to any upload page.
//
//   startup_slot148_acceptance <warp|hardware>
//       (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1, X2W_D3D12_DEBUG=0|1)
//
// The call shape is the one traced from the builder: rcx = the list, rdx = 0x280 bytes,
// r8 = 0x10 alignment. The result in rax is a pointer the caller writes. A null result is a
// refusal; the builder then reports "Failed to allocate upload buffer" (E_FAIL).
//
// Fixture resolver. The module that installs slot 148 advertises it by exporting
//
//   extern "C" unsigned long long x2w_d3d12_upload_gpu_address(void* list, const void* cpu,
//                                                             unsigned long long bytes);
//
// It is a fixture-only adapter to the production address resolver, outside the guest ABI, and holds
// no page reference for the client. It returns the desktop GPU address of [cpu, cpu + bytes) only
// for a complete, non-empty range inside one allocation of the list's open recording, and 0
// otherwise: on a closed list, for a range from before the list's last Reset, or for a range of
// another list.
//
// A module that does not export the resolver has not installed the slot. A null result from it is
// reported as "slot 148 not implemented (missing slot)" (exit 10), a failure. Once the slot is
// advertised, a null for the valid request is a failure too (exit 21), on the first call as well.
//
// GPU reads. The reads recorded on the submitted lists use the guest consumer: list slots 25
// (SetPipelineState) and 33 (SetComputeRootSignature) bind a guest compute pipeline and root
// signature. List slot 43 (SetComputeRootShaderResourceView) then binds the upload pointer at
// parameter 0. The UAV, the Dispatch and the copy to a readback buffer are recorded on the list's
// desktop list, reached the way startup_indirect_dispatch reaches it. Before slot 43, parameter 0
// points at a sentinel buffer, so a refused binding reads the sentinel instead of faulting.
// Separately, an observer queue on the same desktop device reads the resolved GPU ranges while the
// submissions are held.
//
// Preflight. The held-phase observer read needs a second queue to make progress while the console
// queue waits on an unsignalled fence. Before the steps, the console queue waits on such a fence and
// the observer reads the sentinel with a 1 s limit; the fence is then signalled and both queues
// drained. If the read did not finish, the client prints "queues serialized: held-phase checks not
// applicable" and skips only the three held-phase observer reads. All other checks still run,
// including the in-list readback staying zero while held (a CPU read). The final line then
// begins "slot 148 acceptance passed with skips". X2W_TEST_FORCE_SERIALIZED_QUEUES=1 (fixtures
// only) takes that path on any adapter.
//
// Cleanup. Every return opens the gate and drains the console queue, and the client drains both
// queues before it exits. Each wait is bounded at 10 s. If a drain does not finish, the desktop
// objects that work may still use are kept rather than released.
//
// Payloads are non-uniform: byte i of payload s is (i * 29 + 0x5b) ^ s. A and B use the #2068
// values 0xa1 and 0xb2 as seeds, the stale fill 0x5c and the sentinel 0xe7. Any two payloads
// differ at every byte, and a read shifted by 4 or 16 bytes differs at every byte.
//
// Steps, each on new lists:
//
//   traced      Reset the list on allocator a and allocate (0x280, 0x10). The CPU range must be
//               committed and writable. The GPU address must be 16-byte aligned and map the last
//               byte to GPU + 0x27f.
//   fresh       Fill A's range with the stale pattern, record the in-list read of A, then write
//               payload A. Only a read that runs at submission time sees A. Close (S_OK). Then an
//               allocation must return null and A must no longer resolve. Queue Wait on gate H,
//               submit, Signal D = 1. Reset the list on allocator b, allocate B and repeat the
//               stale, record and payload order. Close, submit and Signal D = 2. While both are held,
//               the in-list readback must still be zero, and an observer queue copies both resolved
//               GPU ranges; D must stay 0. The CPU signals H and D must reach 2. The in-list and
//               observer reads must hold A and B byte for byte.
//   allocator   The same, but the list is Reset on allocator a again and allocator a is itself Reset
//               while A is held. A's range must not be handed to B, and A must still read A.
//   overwrite   The #2068 control: as fresh, but B is also written into A's live range before the
//               gate opens. A's reads must then show B: they take the live bytes at execution.
//   reuse       A forced early-reuse control. A is written and the list Closed but never submitted,
//               then the list is Reset on the same idle allocator and B allocated. That is the only
//               client-side sequence after which a correct slot may hand A's range out again. If B
//               overlaps A's GPU range, the observer's read of A's range must show B. If B lies
//               elsewhere the control reports that it was not exercised, because the slot did not reuse.
//   foreign     The wrong-list and stale-generation controls. List 1 owns A, list 2 owns B, both open.
//               On list 2, slot 43 with A's pointer must not bind, so that read shows the sentinel,
//               and list 2's own read shows B. After list 1 is Reset, slot 43 with A's pointer on
//               list 1 must not bind either, while a new allocation reads correctly. The resolver
//               must refuse both ranges too.
//   arguments   Zero size, zero alignment, alignment 24 and a size that overflows must be refused.
//               A 256-byte alignment must be honoured on both the CPU pointer and the GPU address.
//
// Finally the desktop device must not be removed and, with the debug layer, must hold no errors.
//
// Exit codes: 0 pass; 77 no physical adapter; 10 slot 148 not implemented (missing slot); 11 slot 148
// returned memory but its module has no resolver; 12 traced request: CPU range, alignment or GPU
// address; 13 an allocation after a successful Close was not refused, or the closed range still
// resolved; 14 invalid arguments not refused or alignment not honoured; 15 observer payload
// mismatch; 16 overwrite control did not read B; 17 reuse control: B reuses A's range but A's read
// did not show B; 18 ordering (console work ran before the gate opened, a timeout, or a preflight
// that did not drain); 19 device removed or debug-layer errors; 20 A's range was handed to B while
// A was held; 21 the advertised slot refused a valid request; 22 an in-list read ran before its
// submission was released; 23 in-list payload mismatch; 24 the resolver accepted a range of another list or from before a
// Reset; 25 slot 43 bound a range of another list or from before a Reset; 2 to 5 usage and setup.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

constexpr unsigned kCloseSlot = 9, kResetSlot = 139, kUploadSlot = 148;
constexpr unsigned kPipelineSlot = 25, kComputeRootSlot = 33, kComputeSrvSlot = 43;
constexpr unsigned kAllocatorReset = 8, kFenceCompleted = 8, kFenceSignal = 10;
constexpr unsigned kQueueExecute = 10, kQueueSignal = 14, kQueueWait = 17;
constexpr unsigned kDeviceQueue = 8, kDeviceAllocator = 9, kDeviceRoot = 16, kDeviceFence = 36, kDeviceList = 73;
constexpr unsigned kDevicePrivateInterface = 5, kDevicePipelineStream = 86;
constexpr Word kBytes = 0x280, kAlignment = 0x10;  // the traced request: 16 x 40 bytes, 16-byte aligned
constexpr unsigned char kPayloadA = 0xA1, kPayloadB = 0xB2, kStale = 0x5C, kSentinel = 0xE7;
constexpr DWORD kWaitMs = 10000;
constexpr char kGpuExport[] = "x2w_d3d12_upload_gpu_address";
using GpuAddress = unsigned long long (*)(void* list, const void* cpu, unsigned long long bytes);

enum Exit {
    pass = 0, usage = 2, no_adapter = 3, no_device = 4, setup = 5, not_implemented = 10, no_resolver = 11,
    traced_shape = 12, close_refusal = 13, invalid_arguments = 14, payload = 15, overwrite_control = 16,
    reuse_control = 17, ordering = 18, device_state = 19, held_reuse = 20, advertised_refusal = 21,
    list_read_order = 22, list_payload = 23, resolver_scope = 24, consumer_scope = 25, skip = 77
};

unsigned char pattern(unsigned char seed, Word i) { return static_cast<unsigned char>((i * 29 + 0x5b) ^ seed); }
// The payloads differ from each other and from themselves shifted by 4 or 16 bytes, at every byte.
bool patterns_distinct()
{
    const unsigned char seeds[] = {kPayloadA, kPayloadB, kStale, kSentinel};
    for (unsigned char s : seeds) {
        for (Word i = 0; i < kBytes; ++i) {
            for (unsigned char t : seeds)
                if (t != s && pattern(s, i) == pattern(t, i)) return false;
            for (Word shift : {Word{4}, Word{16}})
                if (i + shift < kBytes && pattern(s, i) == pattern(s, i + shift)) return false;
        }
    }
    return true;
}

// [cpu, cpu + bytes) is committed and writable.
bool writable(Word cpu, Word bytes)
{
    constexpr DWORD write = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    for (Word at = cpu; at < cpu + bytes;) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) || info.State != MEM_COMMIT ||
            !(info.Protect & write) || (info.Protect & PAGE_GUARD))
            return false;
        at = reinterpret_cast<Word>(info.BaseAddress) + info.RegionSize;
    }
    return true;
}
// Writes the pattern, reporting an access violation instead of crashing (no C++ objects in a __try frame).
bool fill(Word cpu, unsigned char seed, Word bytes)
{
    __try {
        auto* out = reinterpret_cast<volatile unsigned char*>(cpu);
        for (Word i = 0; i < bytes; ++i) out[i] = pattern(seed, i);
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}
bool overlaps(Word a, Word b, Word bytes) { return a < b + bytes && b < a + bytes; }

struct Upload {
    Word cpu = 0, gpu = 0;
};

HRESULT make_buffer(ID3D12Device* device, D3D12_HEAP_TYPE type, Word bytes, D3D12_RESOURCE_FLAGS flags,
                    D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& out)
{
    D3D12_HEAP_PROPERTIES heap{type};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                   D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags};
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out));
}
bool map_bytes(ID3D12Resource* readback, std::vector<unsigned char>& out)
{
    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(2 * kBytes)};
    if (FAILED(readback->Map(0, &range, &mapped))) return false;
    out.assign(static_cast<const unsigned char*>(mapped), static_cast<const unsigned char*>(mapped) + 2 * kBytes);
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER barrier{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &barrier);
}

constexpr char kCopyShader[] =
    "ByteAddressBuffer src : register(t0);\n"
    "RWByteAddressBuffer dst : register(u0);\n"
    "[numthreads(64, 1, 1)] void main(uint i : SV_DispatchThreadID)\n"
    "{ if (i < 160) dst.Store(i * 4, src.Load(i * 4)); }\n";

// Where the reads recorded on a list land: a UAV with one 0x280-byte slot per read, copied to readback.
struct Reads {
    ComPtr<ID3D12Resource> target, readback;
};

struct Client {
    void* device = nullptr;  // console device
    void* queue = nullptr;   // console queue
    ComPtr<ID3D12Device> desktop;
    bool looked_up = false;
    GpuAddress gpu_address = nullptr;
    // The guest consumer: a guest root signature (SRV at parameter 0, UAV at 1) and compute pipeline.
    void* guest_root = nullptr;
    void* guest_pipeline = nullptr;
    ComPtr<ID3D12Resource> sentinel;
    // Observer: a desktop queue on the same device that copies GPU ranges into a readback buffer.
    ComPtr<ID3D12CommandQueue> observer;
    ComPtr<ID3D12CommandAllocator> observer_allocator;
    ComPtr<ID3D12GraphicsCommandList> observer_list;
    ComPtr<ID3D12Fence> observer_fence;
    UINT64 observer_value = 0;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> copy;
    Reads observed;
    bool serialized = false;  // from the preflight: the held-phase observer reads cannot run

    void* fence()
    {
        void* made = nullptr;
        return ok(invoke(device, kDeviceFence, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&made))) ? made : nullptr;
    }
    void* allocator()
    {
        void* made = nullptr;
        return ok(invoke(device, kDeviceAllocator, D3D12_COMMAND_LIST_TYPE_DIRECT,
                         arg(&__uuidof(ID3D12CommandAllocator)), arg(&made))) ? made : nullptr;
    }
    void* list(void* on)
    {
        const unsigned desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
        void* made = nullptr;
        return ok(invoke(device, kDeviceList, arg(desc), arg(on), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&made)))
                   ? made : nullptr;
    }
    bool make_reads(Reads& out)
    {
        return SUCCEEDED(make_buffer(desktop.Get(), D3D12_HEAP_TYPE_DEFAULT, 2 * kBytes,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                     out.target)) &&
               SUCCEEDED(make_buffer(desktop.Get(), D3D12_HEAP_TYPE_READBACK, 2 * kBytes, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_COPY_DEST, out.readback));
    }
    static bool reached(void* fence, Word value)
    {
        const auto deadline = GetTickCount64() + kWaitMs;
        while (invoke(fence, kFenceCompleted) < value) {
            if (GetTickCount64() >= deadline) return false;
            Sleep(1);
        }
        return true;
    }
    bool submit(void* list, void* done, Word value)
    {
        void* lists[] = {list};
        invoke(queue, kQueueExecute, 1, arg(lists));
        return ok(invoke(queue, kQueueSignal, arg(done), value));
    }
    // The desktop list behind a console list: SetPrivateDataInterface on the console device unwraps
    // its argument before the call reaches the shared desktop device (startup_indirect_dispatch).
    ComPtr<ID3D12GraphicsCommandList> desktop_list(void* list)
    {
        constexpr GUID key{0x6b1f0d2e, 0x93a4, 0x4c51, {0x8e, 0x1d, 0x14, 0x80, 0x37, 0xc2, 0x5a, 0x9b}};
        ComPtr<IUnknown> stored;
        ComPtr<ID3D12GraphicsCommandList> result;
        UINT bytes = sizeof(IUnknown*);
        if (!ok(invoke(device, kDevicePrivateInterface, arg(&key), arg(list)))) return nullptr;
        const HRESULT read = desktop->GetPrivateData(key, &bytes, stored.GetAddressOf());
        invoke(device, kDevicePrivateInterface, arg(&key), 0);
        if (FAILED(read) || !stored || FAILED(stored.As(&result))) return nullptr;
        return result;
    }
    // Records on the console list a GPU read of [cpu, cpu + 0x280) into slot `slot` of reads, through the
    // guest consumer slot 43.
    bool record_read(void* list, Word cpu, Word slot, Reads& reads)
    {
        const auto desk = desktop_list(list);
        if (!desk) return false;
        invoke(list, kPipelineSlot, arg(guest_pipeline));
        invoke(list, kComputeRootSlot, arg(guest_root));
        desk->SetComputeRootShaderResourceView(0, sentinel->GetGPUVirtualAddress());
        invoke(list, kComputeSrvSlot, 0, cpu);
        desk->SetComputeRootUnorderedAccessView(1, reads.target->GetGPUVirtualAddress() + slot * kBytes);
        desk->Dispatch(3, 1, 1);
        transition(desk.Get(), reads.target.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        desk->CopyBufferRegion(reads.readback.Get(), slot * kBytes, reads.target.Get(), slot * kBytes, kBytes);
        transition(desk.Get(), reads.target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        return true;
    }

    // The resolver of the module that installs slot 148, if it advertises the slot.
    void look_up(void* list)
    {
        if (looked_up) return;
        looked_up = true;
        HMODULE owner = nullptr;
        const void* entry = (*static_cast<void***>(list))[kUploadSlot];
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                static_cast<LPCWSTR>(entry), &owner))
            return;
        gpu_address = reinterpret_cast<GpuAddress>(GetProcAddress(owner, kGpuExport));
        wchar_t name[MAX_PATH]{};
        GetModuleFileNameW(owner, name, MAX_PATH);
        const wchar_t* base = std::wcsrchr(name, L'\\');
        std::printf("slot 148 entry in %ls, %s %s\n", base ? base + 1 : name, kGpuExport,
                    gpu_address ? "exported" : "missing");
    }
    Word resolve(void* list, Word cpu, Word bytes) const
    {
        return gpu_address(list, reinterpret_cast<const void*>(cpu), bytes);
    }

    // Allocates the traced request and checks the CPU range and its GPU address.
    int traced(void* list, const char* label, Upload& out)
    {
        look_up(list);
        const Word cpu = invoke(list, kUploadSlot, kBytes, kAlignment);
        if (!gpu_address) {
            if (!cpu) {
                std::printf("%s: slot 148 not implemented (missing slot): no %s export, and "
                            "CommandList::slot148(0x%llx, 0x%llx) returned 0x0\n",
                            label, kGpuExport, static_cast<unsigned long long>(kBytes),
                            static_cast<unsigned long long>(kAlignment));
                return not_implemented;
            }
            std::printf("%s: slot 148 returned 0x%llx, but its module exports no %s\n", label,
                        static_cast<unsigned long long>(cpu), kGpuExport);
            return no_resolver;
        }
        if (!cpu) {
            std::printf("%s: the advertised slot 148 refused the valid request (0x%llx, 0x%llx) on an open list\n", label,
                        static_cast<unsigned long long>(kBytes), static_cast<unsigned long long>(kAlignment));
            return advertised_refusal;
        }
        const Word gpu = resolve(list, cpu, kBytes);
        const Word last = resolve(list, cpu + kBytes - 1, 1);
        const bool range = writable(cpu, kBytes);
        std::printf("%s: cpu %% 16 = %llu, writable = %d, gpu %% 16 = %llu, last byte at gpu + 0x%llx\n", label,
                    static_cast<unsigned long long>(cpu % kAlignment), range ? 1 : 0,
                    static_cast<unsigned long long>(gpu % kAlignment),
                    static_cast<unsigned long long>(last && gpu ? last - gpu : 0));
        if (cpu % kAlignment || !range || !gpu || gpu % kAlignment || last != gpu + kBytes - 1) return traced_shape;
        out = {cpu, gpu};
        return pass;
    }
    // Stale bytes, the in-list read of them, then the payload: only a read at execution sees the payload.
    int write_and_record(void* list, const Upload& upload, unsigned char seed, Reads& reads, Word slot)
    {
        if (!fill(upload.cpu, kStale, kBytes)) return traced_shape;
        if (!record_read(list, upload.cpu, slot, reads)) return setup;
        return fill(upload.cpu, seed, kBytes) ? pass : traced_shape;
    }
    // A successful Close, then an allocation that must be refused and a range that must not resolve.
    int close(void* list, const char* label, const Upload* live = nullptr)
    {
        const auto closed = static_cast<HRESULT>(invoke(list, kCloseSlot));
        if (FAILED(closed)) {
            std::printf("%s: Close -> 0x%08lx\n", label, static_cast<unsigned long>(closed));
            return setup;
        }
        const Word after = invoke(list, kUploadSlot, kBytes, kAlignment);
        const Word resolved = live ? resolve(list, live->cpu, kBytes) : 0;
        std::printf("%s: after a successful Close, allocation -> 0x%llx, last range resolves to 0x%llx (expected 0x0, 0x0)\n",
                    label, static_cast<unsigned long long>(after), static_cast<unsigned long long>(resolved));
        return after || resolved ? close_refusal : pass;
    }

    bool consumer_setup()
    {
        ComPtr<ID3DBlob> code, errors, serialized;
        if (FAILED(D3DCompile(kCopyShader, sizeof(kCopyShader) - 1, "slot148_copy", nullptr, nullptr, "main", "cs_5_1", 0, 0,
                              &code, &errors)))
            return false;
        D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        const D3D12_ROOT_SIGNATURE_DESC root_desc{2, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors)) ||
            FAILED(desktop->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                                IID_PPV_ARGS(&root))) ||
            !ok(invoke(device, kDeviceRoot, 0, arg(serialized->GetBufferPointer()), serialized->GetBufferSize(),
                       arg(&__uuidof(ID3D12RootSignature)), arg(&guest_root))))
            return false;
        struct {
            alignas(void*) D3D12_PIPELINE_STATE_SUBOBJECT_TYPE root_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
            void* root;
            alignas(void*) D3D12_PIPELINE_STATE_SUBOBJECT_TYPE cs_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS;
            D3D12_SHADER_BYTECODE code;
        } stream{D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, guest_root, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS,
                 {code->GetBufferPointer(), code->GetBufferSize()}};
        const D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(stream), &stream};
        if (!ok(invoke(device, kDevicePipelineStream, arg(&stream_desc), arg(&__uuidof(ID3D12PipelineState)),
                       arg(&guest_pipeline))))
            return false;
        if (FAILED(make_buffer(desktop.Get(), D3D12_HEAP_TYPE_UPLOAD, kBytes, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_GENERIC_READ, sentinel)))
            return false;
        void* mapped = nullptr;
        const D3D12_RANGE none{};
        if (FAILED(sentinel->Map(0, &none, &mapped))) return false;
        for (Word i = 0; i < kBytes; ++i) static_cast<unsigned char*>(mapped)[i] = pattern(kSentinel, i);
        sentinel->Unmap(0, nullptr);

        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
        pipeline.pRootSignature = root.Get();
        pipeline.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        return SUCCEEDED(desktop->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&copy))) &&
               SUCCEEDED(desktop->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&observer))) &&
               SUCCEEDED(desktop->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&observer_allocator))) &&
               SUCCEEDED(desktop->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, observer_allocator.Get(), nullptr,
                                                    IID_PPV_ARGS(&observer_list))) &&
               SUCCEEDED(observer_list->Close()) &&
               SUCCEEDED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&observer_fence))) &&
               make_reads(observed);
    }
    enum class ObserverResult { complete, timeout, failure };

    static ObserverResult wait_observer(ID3D12Fence* fence, UINT64 value, DWORD ms)
    {
        if (fence->GetCompletedValue() >= value) return ObserverResult::complete;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) return ObserverResult::failure;
        const DWORD waited = SUCCEEDED(fence->SetEventOnCompletion(value, event))
            ? WaitForSingleObject(event, ms) : WAIT_FAILED;
        CloseHandle(event);
        if (waited == WAIT_OBJECT_0) return ObserverResult::complete;
        if (waited == WAIT_TIMEOUT) return ObserverResult::timeout;
        return ObserverResult::failure;
    }
    // Bounded drains: every submission on the observer or console queue so far has completed.
    bool drain_observer()
    {
        return !observer_fence ||
            wait_observer(observer_fence.Get(), observer_value, kWaitMs) == ObserverResult::complete;
    }
    bool drain_console()
    {
        void* drained = queue ? fence() : nullptr;
        return !queue || (drained && ok(invoke(queue, kQueueSignal, arg(drained), 1)) && reached(drained, 1));
    }
    // The observer: copies [first, first + 0x280) and [second, second + 0x280) on the GPU, in order. A
    // read that does not finish within `ms` stays pending; drain_observer() waits for it.
    ObserverResult read(Word first, Word second, std::vector<unsigned char>& out, DWORD ms = kWaitMs)
    {
        if (!drain_observer() || FAILED(observer_allocator->Reset()) ||
            FAILED(observer_list->Reset(observer_allocator.Get(), copy.Get())))
            return ObserverResult::failure;
        observer_list->SetComputeRootSignature(root.Get());
        const Word sources[] = {first, second};
        for (Word i = 0; i < 2; ++i) {
            observer_list->SetComputeRootShaderResourceView(0, sources[i]);
            observer_list->SetComputeRootUnorderedAccessView(1, observed.target->GetGPUVirtualAddress() + i * kBytes);
            observer_list->Dispatch(3, 1, 1);
        }
        transition(observer_list.Get(), observed.target.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        observer_list->CopyBufferRegion(observed.readback.Get(), 0, observed.target.Get(), 0, 2 * kBytes);
        transition(observer_list.Get(), observed.target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (FAILED(observer_list->Close())) return ObserverResult::failure;
        ID3D12CommandList* lists[] = {observer_list.Get()};
        observer->ExecuteCommandLists(1, lists);
        if (FAILED(observer->Signal(observer_fence.Get(), ++observer_value))) return ObserverResult::failure;
        const auto result = wait_observer(observer_fence.Get(), observer_value, ms);
        if (result != ObserverResult::complete) return result;
        return map_bytes(observed.readback.Get(), out) ? ObserverResult::complete : ObserverResult::failure;
    }

    // Preflight: can the observer queue finish a read while the console queue waits on an unsignalled
    // fence? The held phase needs that. Returns 1 if it can, 0 if the queues are serialized, and -1 if
    // the probe's work did not drain after the fence was signalled or read the wrong bytes.
    int queues_progress()
    {
        void* probe_gate = fence();
        if (!probe_gate || !ok(invoke(queue, kQueueWait, arg(probe_gate), 1))) return -1;
        std::vector<unsigned char> bytes;
        const Word base = sentinel->GetGPUVirtualAddress();
        const auto observed_result = read(base, base, bytes, 1000);
        const bool opened = ok(invoke(probe_gate, kFenceSignal, 1));
        const bool observer_drained = drain_observer();
        const bool console_drained = drain_console();
        if (!opened || !observer_drained || !console_drained || observed_result == ObserverResult::failure)
            return -1;
        if (observed_result == ObserverResult::timeout && !map_bytes(observed.readback.Get(), bytes)) return -1;
        for (Word i = 0; i < 2 * kBytes; ++i)
            if (bytes[i] != pattern(kSentinel, i % kBytes)) return -1;
        return observed_result == ObserverResult::complete ? 1 : 0;
    }

    // Leaves no work pending at exit: drains both queues (bounded). If a drain does not finish, the
    // desktop objects are kept (leaked) rather than released while the GPU may still use them.
    ~Client()
    {
        const bool observer_drained = drain_observer();
        const bool console_drained = drain_console();
        if (observer_drained && console_drained) return;
        std::printf("cleanup: queues did not drain within %lu ms; desktop objects are kept\n",
                    static_cast<unsigned long>(kWaitMs));
        observed.target.Detach();
        observed.readback.Detach();
        sentinel.Detach();
        observer_list.Detach();
        observer_allocator.Detach();
        observer.Detach();
        observer_fence.Detach();
        copy.Detach();
        root.Detach();
        desktop.Detach();
    }

    UINT64 debug_errors()
    {
        ComPtr<ID3D12InfoQueue> info;
        if (FAILED(desktop.As(&info))) return 0;
        UINT64 errors = 0;
        for (UINT64 i = 0, n = info->GetNumStoredMessages(); i < n; ++i) {
            SIZE_T bytes = 0;
            if (FAILED(info->GetMessage(i, nullptr, &bytes))) continue;
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (FAILED(info->GetMessage(i, message, &bytes))) continue;
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                ++errors;
                std::printf("debug layer: %.*s\n", static_cast<int>(message->DescriptionByteLength), message->pDescription);
            }
        }
        return errors;
    }
};

// Bytes [from, from + count) of a readback slot match payload seed; prints the first mismatch.
bool matches(const std::vector<unsigned char>& bytes, Word slot, Word from, Word count, unsigned char seed, const char* what)
{
    for (Word i = from; i < from + count; ++i) {
        if (bytes[slot * kBytes + i] != pattern(seed, i)) {
            std::printf("%s: byte %llu is 0x%02x (expected 0x%02x)\n", what, static_cast<unsigned long long>(i),
                        bytes[slot * kBytes + i], pattern(seed, i));
            return false;
        }
    }
    return true;
}

enum class Mode { fresh, allocator, overwrite };

// On every return: opens the gate and drains the console queue (bounded). If the queue does not
// drain, the read buffers are kept (leaked), so nothing the GPU may still use is released.
struct Gate {
    Client& c;
    void* fence;
    Reads& reads;
    ~Gate()
    {
        invoke(fence, kFenceSignal, 1);
        if (c.drain_console()) return;
        std::printf("cleanup: the console queue did not drain within %lu ms; the read buffers are kept\n",
                    static_cast<unsigned long>(kWaitMs));
        reads.target.Detach();
        reads.readback.Detach();
    }
};

// Submit A behind a gate, Reset, write and submit B, read both while held, then release the gate.
int held_sequence(Client& c, Mode mode)
{
    const char* label = mode == Mode::fresh ? "fresh" : mode == Mode::allocator ? "allocator" : "overwrite";
    void* a = c.allocator();
    void* b = c.allocator();
    void* list = a ? c.list(a) : nullptr;
    void* gate = c.fence();
    void* done = c.fence();
    Reads listed;  // the in-list reads: A at 0, B at 0x280
    if (!b || !list || !gate || !done || !c.make_reads(listed) || !ok(invoke(list, kResetSlot, arg(a)))) return setup;

    Upload first;
    if (const int result = c.traced(list, label, first)) return result;
    if (const int result = c.write_and_record(list, first, kPayloadA, listed, 0)) return result;
    if (const int result = c.close(list, label, &first)) return result;
    const Gate held{c, gate, listed};
    if (!ok(invoke(c.queue, kQueueWait, arg(gate), 1)) || !c.submit(list, done, 1)) return setup;

    // Reset while A is held: on another allocator, or on A's own allocator after resetting it.
    if (mode == Mode::allocator && !ok(invoke(a, kAllocatorReset))) return setup;
    if (!ok(invoke(list, kResetSlot, arg(mode == Mode::allocator ? a : b)))) return setup;
    Upload second;
    if (const int result = c.traced(list, label, second)) return result;
    if (overlaps(first.cpu, second.cpu, kBytes) || overlaps(first.gpu, second.gpu, kBytes)) {
        std::printf("%s: B (cpu 0x%llx, gpu 0x%llx) reuses A's range (cpu 0x%llx, gpu 0x%llx) while A is held\n", label,
                    static_cast<unsigned long long>(second.cpu), static_cast<unsigned long long>(second.gpu),
                    static_cast<unsigned long long>(first.cpu), static_cast<unsigned long long>(first.gpu));
        return held_reuse;
    }
    // A belongs to the recording before the Reset.
    if (const Word stale = c.resolve(list, first.cpu, kBytes)) {
        std::printf("%s: A still resolves after the Reset, to 0x%llx\n", label, static_cast<unsigned long long>(stale));
        return resolver_scope;
    }
    if (const int result = c.write_and_record(list, second, kPayloadB, listed, 1)) return result;
    if (mode == Mode::overwrite && !fill(first.cpu, kPayloadB, kBytes)) return traced_shape;
    if (const int result = c.close(list, label, &second)) return result;
    if (!c.submit(list, done, 2)) return setup;

    Sleep(50);
    std::vector<unsigned char> early, observed, recorded;
    if (!map_bytes(listed.readback.Get(), early)) return setup;
    const bool untouched = std::all_of(early.begin(), early.end(), [](unsigned char v) { return v == 0; });
    const Word before = invoke(done, kFenceCompleted);
    bool read = true;
    Word after = 0;
    if (c.serialized) {
        std::printf("%s: while held, in-list readback untouched = %d; console submissions completed %llu "
                    "(expected 1, 0); queues serialized: held-phase observer read not applicable\n",
                    label, untouched ? 1 : 0, static_cast<unsigned long long>(before));
    } else {
        read = c.read(first.gpu, second.gpu, observed) == Client::ObserverResult::complete;
        after = invoke(done, kFenceCompleted);
        std::printf("%s: while held, in-list readback untouched = %d; console submissions completed before the "
                    "observer read %llu, after it %llu (expected 1, 0, 0)\n",
                    label, untouched ? 1 : 0, static_cast<unsigned long long>(before),
                    static_cast<unsigned long long>(after));
        if (!read)
            std::printf("%s: the observer read did not finish within %lu ms while the gate was closed\n", label,
                        static_cast<unsigned long>(kWaitMs));
    }
    if (!untouched) return list_read_order;
    const bool opened = ok(invoke(gate, kFenceSignal, 1));
    if (!read || before || after || !opened || !Client::reached(done, 2)) return ordering;
    if (!map_bytes(listed.readback.Get(), recorded)) return setup;

    const unsigned char wanted = mode == Mode::overwrite ? kPayloadB : kPayloadA;
    const int a_failure = mode == Mode::overwrite ? overwrite_control : mode == Mode::allocator ? held_reuse : 0;
    if (!matches(recorded, 1, 0, kBytes, kPayloadB, "in-list B")) return list_payload;
    if (!matches(recorded, 0, 0, kBytes, wanted, "in-list A")) return a_failure ? a_failure : list_payload;
    if (!c.serialized) {
        if (!matches(observed, 1, 0, kBytes, kPayloadB, "observer B")) return payload;
        if (!matches(observed, 0, 0, kBytes, wanted, "observer A")) return a_failure ? a_failure : payload;
    }
    std::printf("%s: %s reads: A holds payload 0x%02x and B payload 0x%02x, all %llu bytes\n", label,
                c.serialized ? "in-list" : "in-list and observer", wanted, kPayloadB,
                static_cast<unsigned long long>(kBytes));
    return pass;
}

// Close without submitting, Reset on the same idle allocator: the slot may now hand A's range out again.
int forced_reuse(Client& c)
{
    void* a = c.allocator();
    void* list = a ? c.list(a) : nullptr;
    if (!list || !ok(invoke(list, kResetSlot, arg(a)))) return setup;
    Upload first, second;
    if (const int result = c.traced(list, "reuse", first)) return result;
    if (!fill(first.cpu, kPayloadA, kBytes)) return traced_shape;
    if (const int result = c.close(list, "reuse")) return result;
    if (!ok(invoke(list, kResetSlot, arg(a)))) return setup;
    if (const int result = c.traced(list, "reuse", second)) return result;
    if (!fill(second.cpu, kPayloadB, kBytes)) return traced_shape;
    if (!overlaps(first.gpu, second.gpu, kBytes)) {
        std::printf("reuse: not exercised, B (gpu 0x%llx) does not overlap the unsubmitted A (gpu 0x%llx)\n",
                    static_cast<unsigned long long>(second.gpu), static_cast<unsigned long long>(first.gpu));
        return c.close(list, "reuse");
    }
    // A's range is live again as part of B, so reading it is defined. Byte i of A's range is byte
    // i - shift of B, where shift is B's start relative to A's.
    std::vector<unsigned char> bytes;
    if (c.read(first.gpu, second.gpu, bytes) != Client::ObserverResult::complete) return ordering;
    const Word from = second.gpu > first.gpu ? second.gpu - first.gpu : 0;
    const Word to = (std::min)(kBytes, second.gpu + kBytes - first.gpu);
    for (Word i = from; i < to; ++i) {
        const Word in_b = first.gpu + i - second.gpu;
        if (bytes[i] != pattern(kPayloadB, in_b)) {
            std::printf("reuse control: byte %llu of A's range is 0x%02x (expected B's byte %llu, 0x%02x)\n",
                        static_cast<unsigned long long>(i), bytes[i], static_cast<unsigned long long>(in_b),
                        pattern(kPayloadB, in_b));
            return reuse_control;
        }
    }
    std::printf("reuse: B overlaps A at bytes [%llu, %llu) and A's read shows B there\n",
                static_cast<unsigned long long>(from), static_cast<unsigned long long>(to));
    return c.close(list, "reuse");
}

// Runs the list's recorded reads to completion and returns the readback.
bool run_reads(Client& c, void* list, Reads& reads, std::vector<unsigned char>& out)
{
    void* done = c.fence();
    const bool submitted = done && ok(invoke(list, kCloseSlot)) && c.submit(list, done, 1);
    if (submitted && Client::reached(done, 1)) return map_bytes(reads.readback.Get(), out);
    if (!c.drain_console()) {
        reads.target.Detach();
        reads.readback.Detach();
    }
    return false;
}

// The wrong-list and stale-generation controls for the resolver and the guest consumer.
int foreign(Client& c)
{
    void* a1 = c.allocator();
    void* a2 = c.allocator();
    void* first = a1 ? c.list(a1) : nullptr;
    void* second = a2 ? c.list(a2) : nullptr;
    Reads on_second, on_first;
    if (!second || !first || !c.make_reads(on_second) || !c.make_reads(on_first) ||
        !ok(invoke(first, kResetSlot, arg(a1))) || !ok(invoke(second, kResetSlot, arg(a2))))
        return setup;
    Upload a, b;
    if (const int result = c.traced(first, "foreign", a)) return result;
    if (const int result = c.traced(second, "foreign", b)) return result;
    if (!fill(a.cpu, kPayloadA, kBytes) || !fill(b.cpu, kPayloadB, kBytes)) return traced_shape;
    const Word across = c.resolve(second, a.cpu, kBytes);
    // Slot 0 of list 2: slot 43 on list 2 with list 1's A. Slot 1: list 2's own B.
    if (!c.record_read(second, a.cpu, 0, on_second) || !c.record_read(second, b.cpu, 1, on_second)) return setup;
    std::vector<unsigned char> bytes;
    if (!run_reads(c, second, on_second, bytes)) return ordering;
    const bool foreign_refused = matches(bytes, 0, 0, kBytes, kSentinel, "foreign: list 2 reading list 1's A");
    const bool own_read = matches(bytes, 1, 0, kBytes, kPayloadB, "foreign: list 2 reading its own B");

    // List 1 after a Reset: A is from the previous recording. Slot 1 reads a new allocation.
    if (const int result = c.close(first, "foreign")) return result;
    if (!ok(invoke(first, kResetSlot, arg(a1)))) return setup;
    Upload fresh;
    if (const int result = c.traced(first, "foreign", fresh)) return result;
    if (!fill(fresh.cpu, kPayloadB, kBytes)) return traced_shape;
    const Word stale = overlaps(a.cpu, fresh.cpu, kBytes) ? 0 : c.resolve(first, a.cpu, kBytes);
    if (!c.record_read(first, a.cpu, 0, on_first) || !c.record_read(first, fresh.cpu, 1, on_first)) return setup;
    if (!run_reads(c, first, on_first, bytes)) return ordering;
    const bool stale_refused = overlaps(a.cpu, fresh.cpu, kBytes) ||
                               matches(bytes, 0, 0, kBytes, kSentinel, "foreign: list 1 reading A after its Reset");
    const bool fresh_read = matches(bytes, 1, 0, kBytes, kPayloadB, "foreign: list 1 reading its new allocation");
    std::printf("foreign: resolver for list 1's A on list 2 -> 0x%llx, on list 1 after its Reset -> 0x%llx (expected 0x0, "
                "0x0); slot 43 refused both = %d, own reads correct = %d\n",
                static_cast<unsigned long long>(across), static_cast<unsigned long long>(stale),
                foreign_refused && stale_refused ? 1 : 0, own_read && fresh_read ? 1 : 0);
    if (!own_read || !fresh_read) return list_payload;
    if (!foreign_refused || !stale_refused) return consumer_scope;
    return across || stale ? resolver_scope : pass;
}

int arguments(Client& c)
{
    void* a = c.allocator();
    void* list = a ? c.list(a) : nullptr;
    if (!list || !ok(invoke(list, kResetSlot, arg(a)))) return setup;
    const struct {
        const char* name;
        Word bytes, alignment;
    } refused[] = {{"zero size", 0, kAlignment}, {"zero alignment", kBytes, 0}, {"alignment 24", kBytes, 24},
                   {"size overflow", ~Word{0} - 15, kAlignment}};
    int result = pass;
    for (const auto& request : refused) {
        const Word cpu = invoke(list, kUploadSlot, request.bytes, request.alignment);
        std::printf("arguments: %s -> 0x%llx (expected 0x0)\n", request.name, static_cast<unsigned long long>(cpu));
        if (cpu) result = invalid_arguments;
    }
    const Word cpu = invoke(list, kUploadSlot, 64, 256);
    const Word gpu = cpu ? c.resolve(list, cpu, 64) : 0;
    std::printf("arguments: (64, 256) -> cpu %% 256 = %llu, gpu %% 256 = %llu, gpu %s\n",
                static_cast<unsigned long long>(cpu % 256), static_cast<unsigned long long>(gpu % 256),
                gpu ? "found" : "missing");
    if (!cpu || cpu % 256 || !gpu || gpu % 256) result = invalid_arguments;
    const int closed = c.close(list, "arguments");
    return result ? result : closed;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware"))) return usage;
    if (!patterns_distinct()) return usage;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return no_adapter;
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(argv[1], "hardware") == 0) {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return skip;
            if (FAILED(hr)) return no_adapter;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return no_adapter;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    } else if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return no_adapter;
    }
    DXGI_ADAPTER_DESC adapter_desc{};
    if (FAILED(adapter->GetDesc(&adapter_desc))) return no_adapter;
    char debug[4]{};
    const bool debugging = GetEnvironmentVariableA("X2W_D3D12_DEBUG", debug, sizeof(debug)) && debug[0] == '1';
    std::printf("adapter=%ls debug_layer=%s\n", adapter_desc.Description, debugging ? "requested" : "off");

    Client c;
    std::array<unsigned char, 64> parameters{};
    // The console device first: with X2W_D3D12_DEBUG=1 it enables the debug layer before the desktop device exists.
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &c.device))) return no_device;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&c.desktop)))) return no_device;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    if (!ok(invoke(c.device, kDeviceQueue, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&c.queue))) ||
        !c.consumer_setup())
        return setup;

    // Preflight for the held phase. X2W_TEST_FORCE_SERIALIZED_QUEUES=1 (fixtures only) takes the
    // serialized path on any adapter, so that path is tested where the queues do make progress.
    const int progress = c.queues_progress();
    const bool forced = GetEnvironmentVariableA("X2W_TEST_FORCE_SERIALIZED_QUEUES", debug, sizeof(debug)) && debug[0] == '1';
    std::printf("preflight: the observer queue %s while the console queue waited on an unsignalled fence%s\n",
                progress > 0 ? "finished a read" : progress == 0 ? "made no progress" : "failed the probe",
                forced ? "; X2W_TEST_FORCE_SERIALIZED_QUEUES=1" : "");
    if (progress < 0) return ordering;
    c.serialized = forced || progress == 0;
    if (c.serialized) std::puts("queues serialized: held-phase checks not applicable");

    int result = held_sequence(c, Mode::fresh);
    if (!result) result = held_sequence(c, Mode::allocator);
    if (!result) result = held_sequence(c, Mode::overwrite);
    if (!result) result = forced_reuse(c);
    if (!result) result = foreign(c);
    if (!result) result = arguments(c);
    const HRESULT removed = c.desktop->GetDeviceRemovedReason();
    const UINT64 errors = c.debug_errors();
    std::printf("device removed reason: 0x%08lx, debug layer errors: %llu\n", static_cast<unsigned long>(removed),
                static_cast<unsigned long long>(errors));
    if (!result && (removed != S_OK || errors)) result = device_state;
    if (result) std::printf("slot 148 acceptance failed: exit %d\n", result);
    else if (c.serialized)
        std::puts("slot 148 acceptance passed with skips: held-phase observer reads in fresh, allocator and overwrite "
                  "(queues serialized); every other check ran");
    else std::puts("slot 148 acceptance passed");
    return result;
}
