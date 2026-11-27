#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "adapter_name_filter.h"
#include "startup_indirect_writer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_fixture_compute_root_uav(void*, UINT, UINT64);
extern "C" __declspec(dllimport) UINT x2w_d3d12_recording_scratches(void*);

namespace {
using Microsoft::WRL::ComPtr;
using Words = std::vector<std::uint32_t>;
// Word 0 pads the record to byte offset 4. The `constants` mode puts two root constants (scale, bias) in front of
// the Dispatch record; the default mode has the Dispatch record alone. The last two words are result and guard.
bool constants = false;
// The `count` mode runs a three-record argument buffer behind a GPU-written count word (max count 3).
bool counted = false;
// X2W_SPLIT_INDIRECT_DISPATCH=1 (the startup_indirect_count_split_* tests): the shim splits a batched or counted
// Dispatch into single-command calls over a scratch copy of the records (split_indirect_dispatch.h).
bool split = false;
// The follow-up case records one more Dispatch(1, 1, 2) on the desktop list after the counted ExecuteIndirect, with
// the state the shim left there. It adds 1 + 101 = 102 to the result only if the consumer pipeline, its compute root
// and the result UAV are still bound, also after the split replaced them for its own pass.
constexpr std::uint32_t follow_up_term = 102;
constexpr UINT max_commands = 3;
Words initial;
UINT argument_bytes = 0;
constexpr UINT argument_offset = sizeof(std::uint32_t);
constexpr UINT result_bytes = 2 * sizeof(std::uint32_t);

// The two entry points bind different buffers at u0. The consumer changes
// one word per group, so a missing or incorrectly shaped dispatch is visible.
// The root-constant consumer scales each group's term and adds a bias, so
// swapped or missing constants change the result.
constexpr char shader[] = R"(
RWByteAddressBuffer data : register(u0);
cbuffer Root : register(b0) { uint scale; uint bias; };

[numthreads(1, 1, 1)]
void write_constant_args() { data.Store2(4, uint2(3, 7)); data.Store3(12, uint3(4, 3, 2)); }

[numthreads(1, 1, 1)]
void write_swapped_args() { data.Store2(4, uint2(7, 3)); data.Store3(12, uint3(4, 3, 2)); }

[numthreads(1, 1, 1)]
void consume_constant_args(uint3 group : SV_GroupID) {
    uint previous;
    data.InterlockedAdd(0, scale * (1 + group.x + 10 * group.y + 100 * group.z) + bias, previous);
}
void write_records(uint count) {
    data.Store(4, count);
    data.Store3(8, uint3(2, 1, 1));
    data.Store3(20, uint3(1, 2, 1));
    data.Store3(32, uint3(1, 1, 3));
}
[numthreads(1, 1, 1)]
void write_count2() { write_records(2); }
[numthreads(1, 1, 1)]
void write_count9() { write_records(9); }
[numthreads(1, 1, 1)]
void write_count0() { write_records(0); }
[numthreads(1, 1, 1)]
void write_count1() { write_records(1); }
[numthreads(1, 1, 1)]
void write_count3() { write_records(3); }

[numthreads(1, 1, 1)]
void write_args() { data.Store3(4, uint3(4, 3, 2)); }

[numthreads(1, 1, 1)]
void write_fewer_args() { data.Store3(4, uint3(2, 3, 2)); }

[numthreads(1, 1, 1)]
void consume_args(uint3 group : SV_GroupID) {
    uint previous;
    data.InterlockedAdd(0, 1 + group.x + 10 * group.y + 100 * group.z, previous);
}
)";

ComPtr<ID3DBlob> compile(const char* entry)
{
    ComPtr<ID3DBlob> code, errors;
    const HRESULT hr = D3DCompile(shader, sizeof(shader) - 1, nullptr, nullptr, nullptr, entry, "cs_5_0",
                                  0, 0, &code, &errors);
    if (FAILED(hr) && errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
    check(hr, entry);
    return code;
}

ComPtr<ID3D12PipelineState> pipeline(ID3D12Device* device, ID3D12RootSignature* root, const char* entry)
{
    const auto code = compile(entry);
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    ComPtr<ID3D12PipelineState> result;
    check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&result)), entry);
    return result;
}

template <typename Method>
Method slot(IUnknown* object, unsigned index)
{
    return reinterpret_cast<Method>((*reinterpret_cast<void***>(object))[index]);
}

ComPtr<IUnknown> guest_pipeline(ID3D12Device* guest, ID3D12RootSignature* root)
{
    const auto code = compile(constants ? "consume_constant_args" : "consume_args");
    struct {
        alignas(void*) D3D12_PIPELINE_STATE_SUBOBJECT_TYPE root_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
        ID3D12RootSignature* root;
        alignas(void*) D3D12_PIPELINE_STATE_SUBOBJECT_TYPE cs_type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS;
        D3D12_SHADER_BYTECODE code;
    } stream{D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root,
             D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS, {code->GetBufferPointer(), code->GetBufferSize()}};
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    using Create = HRESULT (WINAPI*)(IUnknown*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    ComPtr<IUnknown> result;
    check(slot<Create>(guest, 86)(guest, &desc, __uuidof(ID3D12PipelineState),
                                 reinterpret_cast<void**>(result.GetAddressOf())), "create guest compute pipeline");
    return result;
}

// partial_record (constants mode) starts the record 12 bytes before the buffer end: a Dispatch alone would fit,
// the constants and Dispatch do not. padded_stride uses a constant signature whose stride has 4 spare bytes.
// The count mode adds no_count_buffer (accepted: every record up to the max count runs), count_outside and
// count_unaligned (count word outside the buffer or off a word boundary) and records_short (max count records overflow).
enum class Rejection { none, unmapped, short_range, placeholder, other_root, partial_record, padded_stride,
                       no_count_buffer, count_outside, count_unaligned, records_short, rooted_multi, rooted_count_buffer };

// `record` is what the writer stores after the padding word: (scale, bias) in the constants mode, then the groups.
// The constants mode also refuses a rooted signature with two records (rooted_multi) or a count buffer
// (rooted_count_buffer): batching is rootless-only in this slice.
// The other-root control binds `other_guest_root`, which lacks the signature's constants.
// Count mode only: `follow_up` adds the follow-up Dispatch above. `retire` submits through a guest queue held
// behind a gate fence, resets the guest allocator while that work is still pending, and requires the recording's
// split scratch to be held until the gate opens and the records run.
bool run(ID3D12Device* device, ID3D12Device* guest, ID3D12RootSignature* root, ID3D12RootSignature* guest_root,
         ID3D12RootSignature* other_guest_root, ID3D12PipelineState* writer, IUnknown* consumer,
         ID3D12CommandSignature* signature, const Words& record, Rejection rejection, bool follow_up = false,
         bool retire = false)
{
    const UINT total_bytes = static_cast<UINT>(initial.size() * sizeof(std::uint32_t));
    auto upload = make_buffer(device, total_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = make_buffer(device, total_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const auto free_memory = [](void* memory) { VirtualFree(memory, 0, MEM_RELEASE); };
    const std::unique_ptr<void, decltype(free_memory)> memory_backing(
        VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE), free_memory);
    if (!memory_backing) throw std::bad_alloc();
    D3D12_RESOURCE_DESC argument_desc{};
    argument_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    argument_desc.Width = argument_bytes;
    argument_desc.Height = argument_desc.DepthOrArraySize = argument_desc.MipLevels = argument_desc.SampleDesc.Count = 1;
    argument_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    argument_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    using Place = HRESULT (WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
    ComPtr<IUnknown> guest_arguments;
    check(slot<Place>(guest, 133)(guest, static_cast<char*>(memory_backing.get()) + 4, &argument_desc,
                                  D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource),
                                  reinterpret_cast<void**>(guest_arguments.GetAddressOf())), "place guest argument buffer");
    const auto arguments = indirect_native_object<ID3D12Resource>(device, guest, guest_arguments.Get());
    auto result = make_buffer(device, result_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    void* memory = nullptr;
    const D3D12_RANGE untouched{0, 0};
    check(upload->Map(0, &untouched, &memory), "map upload");
    std::memcpy(memory, initial.data(), total_bytes);
    upload->Unmap(0, nullptr);

    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create guest allocator");
    using CreateList = HRESULT (WINAPI*)(IUnknown*, const void*, IUnknown*, IUnknown*, REFIID, void**);
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    ComPtr<IUnknown> guest_list;
    check(slot<CreateList>(guest, 73)(guest, list_desc, allocator.Get(), nullptr, __uuidof(ID3D12GraphicsCommandList),
                                     reinterpret_cast<void**>(guest_list.GetAddressOf())), "create guest command list");
    using Reset = HRESULT (WINAPI*)(IUnknown*, IUnknown*, void*, void*);
    check(slot<Reset>(guest_list.Get(), 139)(guest_list.Get(), allocator.Get(), nullptr, nullptr), "reset guest command list");
    const auto native_list = indirect_native_object<ID3D12GraphicsCommandList>(device, guest, guest_list.Get());
    auto* list = native_list.Get();
    transition(list, arguments.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(arguments.Get(), 0, upload.Get(), 0, argument_bytes);
    list->CopyBufferRegion(result.Get(), 0, upload.Get(), argument_bytes, result_bytes);
    transition(list, result.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetComputeRootSignature(root);
    auto argument_state = D3D12_RESOURCE_STATE_COPY_DEST;
    if (writer) {
        transition(list, arguments.Get(), argument_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        argument_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        record_indirect_writer(list, writer, arguments.Get());
    }
    transition(list, arguments.Get(), argument_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, arguments.Get(), 0, argument_bytes);
    transition(list, arguments.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);

    using Bind = void (WINAPI*)(IUnknown*, IUnknown*);
    slot<Bind>(guest_list.Get(), 25)(guest_list.Get(), consumer);
    slot<Bind>(guest_list.Get(), 33)(guest_list.Get(), rejection == Rejection::other_root ? other_guest_root : guest_root);
    // Bound through the shim, which sets it again after a compute pass of its own (the split of a batched Dispatch).
    // If the shim has no matching UAV parameter for the bound guest root, bind on the desktop list as before.
    if (!x2w_d3d12_fixture_compute_root_uav(guest_list.Get(), 0, result->GetGPUVirtualAddress()))
        list->SetComputeRootUnorderedAccessView(0, result->GetGPUVirtualAddress());
    using Execute = void (WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
    auto* argument = rejection == Rejection::unmapped ? nullptr : guest_arguments.Get();
    UINT64 offset = counted ? 2 * sizeof(std::uint32_t) : argument_offset;
    UINT64 count_offset = sizeof(std::uint32_t);
    if (rejection == Rejection::count_outside) count_offset = argument_bytes;
    if (rejection == Rejection::count_unaligned) count_offset = 6;
    if (rejection == Rejection::records_short) offset = 3 * sizeof(std::uint32_t);
    if (rejection == Rejection::short_range) offset = argument_bytes;
    if (rejection == Rejection::partial_record) offset = argument_bytes - sizeof(D3D12_DISPATCH_ARGUMENTS);
    if (counted)
        slot<Execute>(guest_list.Get(), 61)(guest_list.Get(), signature, max_commands, argument, offset,
                                            rejection == Rejection::no_count_buffer ? nullptr : argument,
                                            rejection == Rejection::no_count_buffer ? 0 : count_offset);
    else
        slot<Execute>(guest_list.Get(), 61)(guest_list.Get(), signature, rejection == Rejection::rooted_multi ? 2 : 1,
                                            argument, offset,
                                            rejection == Rejection::rooted_count_buffer ? argument : nullptr,
                                            rejection == Rejection::rooted_count_buffer ? sizeof(std::uint32_t) : 0);
    if (follow_up) list->Dispatch(1, 1, 2);
    transition(list, result.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), argument_bytes, result.Get(), 0, result_bytes);
    using Close = HRESULT (WINAPI*)(IUnknown*);
    check(slot<Close>(guest_list.Get(), 9)(guest_list.Get()), "close guest command list");
    // Each run records on a fresh guest allocator: an accepted counted call holds one split scratch when the split is on.
    const bool accepted = rejection == Rejection::none || rejection == Rejection::no_count_buffer;
    const UINT scratches = counted && split && accepted ? 1 : 0;
    if (counted && x2w_d3d12_recording_scratches(guest_list.Get()) != scratches) {
        std::fprintf(stderr, "recording holds %u split scratch buffers after Close, expected %u\n",
                     x2w_d3d12_recording_scratches(guest_list.Get()), scratches);
        return false;
    }
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "create completion fence");
    if (retire) {
        // The shim tracks this submission's fence, so the allocator reset below must not release the scratch the
        // split calls read: the desktop queue waits on the gate before it can run them.
        ComPtr<IUnknown> guest_queue;
        check(guest->CreateCommandQueue(&queue_desc, __uuidof(ID3D12CommandQueue),
                                        reinterpret_cast<void**>(guest_queue.GetAddressOf())), "create guest queue");
        const auto queue = indirect_native_object<ID3D12CommandQueue>(device, guest, guest_queue.Get());
        ComPtr<ID3D12Fence> gate;
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)), "create gate fence");
        check(queue->Wait(gate.Get(), 1), "hold the guest queue on the gate");
        using Submit = void (WINAPI*)(IUnknown*, UINT, IUnknown* const*);
        IUnknown* submitted_list = guest_list.Get();
        slot<Submit>(guest_queue.Get(), 10)(guest_queue.Get(), 1, &submitted_list);
        const HRESULT reset = allocator->Reset();
        const UINT held = x2w_d3d12_recording_scratches(guest_list.Get());
        check(gate->Signal(1), "open the gate");
        check(queue->Signal(fence.Get(), 1), "signal completion fence");
        check(fence->SetEventOnCompletion(1, nullptr), "wait for completion");
        std::printf("retire: allocator reset while pending -> 0x%08lx, recording holds %u split scratch buffers\n",
                    static_cast<unsigned long>(reset), held);
        if (FAILED(reset) || held != scratches) {
            std::fprintf(stderr, "split scratch was released before its submission retired\n");
            return false;
        }
    } else {
        ComPtr<ID3D12CommandQueue> queue;
        check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "create queue");
        ID3D12CommandList* submitted[]{list};
        queue->ExecuteCommandLists(1, submitted);
        check(queue->Signal(fence.Get(), 1), "signal completion fence");
        check(fence->SetEventOnCompletion(1, nullptr), "wait for completion");
    }
    check(device->GetDeviceRemovedReason(), "GPU completion");

    const D3D12_RANGE read{0, total_bytes};
    check(readback->Map(0, &read, &memory), "map readback");
    Words observed(initial.size());
    std::memcpy(observed.data(), memory, total_bytes);
    readback->Unmap(0, &untouched);
    Words expected = initial;
    if (writer) std::copy(record.begin(), record.end(), expected.begin() + 1);
    if (counted) {
        // Words 2..10 hold three Dispatch records. The GPU count word (word 1) lowers the number run, never raises it.
        const std::size_t result_word = initial.size() - 2;
        const bool runs = rejection == Rejection::none || rejection == Rejection::no_count_buffer;
        const std::uint32_t executed = !runs ? 0 : rejection == Rejection::no_count_buffer ? max_commands
                                                                                       : (std::min)(expected[1], max_commands);
        for (std::uint32_t i = 0; i < executed; ++i)
            for (std::uint32_t z = 0; z < expected[4 + 3 * i]; ++z)
                for (std::uint32_t y = 0; y < expected[3 + 3 * i]; ++y)
                    for (std::uint32_t x = 0; x < expected[2 + 3 * i]; ++x)
                        expected[result_word] += 1 + x + 10 * y + 100 * z;
        if (follow_up) expected[result_word] += follow_up_term;
        std::printf("%s%s%s: count word=%u, executed=%u, result=0x%08x (expected 0x%08x)\n",
                    rejection == Rejection::none ? (writer ? "GPU writer" : "omitted writer")
                                                  : rejection == Rejection::no_count_buffer ? "no count buffer" : "refused call",
                    follow_up ? " + follow-up Dispatch" : "", retire ? " + retired allocator" : "",
                    observed[1], executed, observed[result_word], expected[result_word]);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (observed[i] == expected[i]) continue;
            std::fprintf(stderr, "word %zu: got 0x%08x, expected 0x%08x\n", i, observed[i], expected[i]);
            return false;
        }
        return true;
    }
    const std::size_t groups = constants ? 3 : 1;
    const std::size_t sum = expected.size() - 2;
    const std::uint32_t scale = constants ? expected[1] : 1;
    const std::uint32_t bias = constants ? expected[2] : 0;
    if (rejection == Rejection::none) {
        for (std::uint32_t z = 0; z < expected[groups + 2]; ++z)
            for (std::uint32_t y = 0; y < expected[groups + 1]; ++y)
                for (std::uint32_t x = 0; x < expected[groups]; ++x)
                    expected[sum] += scale * (1 + x + 10 * y + 100 * z) + bias;
    }
    std::printf("%s: constants=(%u,%u), args=(%u,%u,%u), result=0x%08x (expected 0x%08x)\n",
                rejection == Rejection::none ? (writer ? "GPU writer" : "omitted writer") : "refused call",
                constants ? observed[1] : 0, constants ? observed[2] : 0,
                observed[groups], observed[groups + 1], observed[groups + 2],
                observed[sum], expected[sum]);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (observed[i] == expected[i]) continue;
        std::fprintf(stderr, "word %zu: got 0x%08x, expected 0x%08x\n", i, observed[i], expected[i]);
        return false;
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) try
{
    const bool warp = argc >= 2 && std::strcmp(argv[1], "warp") == 0;
    if (!warp && (argc < 2 || std::strcmp(argv[1], "hardware") != 0)) return 2;
    constants = argc == 3 && std::strcmp(argv[2], "constants") == 0;
    counted = argc == 3 && std::strcmp(argv[2], "count") == 0;
    if (argc > 3 || (argc == 3 && !constants && !counted)) return 2;
    {
        char value[4]{};
        split = GetEnvironmentVariableA("X2W_SPLIT_INDIRECT_DISPATCH", value, sizeof(value)) && value[0] == '1';
    }
    initial = counted ? Words{0xa5a5a5a5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x31415926, 0x5a5a5a5a} : constants ? Words{0xa5a5a5a5, 0, 0, 0, 0, 0, 3, 5, 1, 1, 1, 0x31415926, 0x5a5a5a5a}
                        : Words{0xa5a5a5a5, 0, 0, 0, 0x31415926, 0x5a5a5a5a};
    argument_bytes = static_cast<UINT>((initial.size() - 2) * sizeof(std::uint32_t));
    ComPtr<ID3D12Debug> debug;
    const HRESULT debug_hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
    if (debug_hr != DXGI_ERROR_SDK_COMPONENT_MISSING) check(debug_hr, "get debug layer");
    if (debug) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    if (warp) {
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "find WARP");
    } else {
        AdapterNameFilter filter;
        if (!filter.read()) return 4;
        for (UINT index = 0; factory->EnumAdapters1(index, adapter.ReleaseAndGetAddressOf()) == S_OK; ++index) {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                filter.matches(desc.Description) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) break;
        }
        if (!adapter) return filter.no_match();
    }
    check(D3D12XboxCreateDevice(adapter.Get(), nullptr, IID_PPV_ARGS(&device)), "create desktop device through d3d12_xs");
    SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", "1");
    SetEnvironmentVariableA("X2W_D3D12_RENDER", "1");
    const std::array<unsigned char, 64> parameters{};
    ComPtr<ID3D12Device> guest;
    check(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), IID_PPV_ARGS(&guest)), "create guest device");
    if (!reinterpret_cast<const unsigned char*>(guest.Get())[0x18]) {
        std::fputs("guest device permits inline descriptor creation\n", stderr);
        return 1;
    }
    DXGI_ADAPTER_DESC1 adapter_desc{};
    check(adapter->GetDesc1(&adapter_desc), "get adapter description");
    std::printf("adapter: %ls, debug layer %s\n", adapter_desc.Description, debug ? "enabled" : "unavailable");
    LARGE_INTEGER driver{};
    if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver)))
        std::printf("driver: %u.%u.%u.%u\n", HIWORD(driver.HighPart), LOWORD(driver.HighPart),
                    HIWORD(driver.LowPart), LOWORD(driver.LowPart));

    // Parameter 0 is the UAV at u0. The constants mode adds parameter 1, two 32-bit constants at b0.
    D3D12_ROOT_PARAMETER parameters_desc[2]{};
    parameters_desc[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameters_desc[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    parameters_desc[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters_desc[1].Constants = {0, 0, 2};
    parameters_desc[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    const auto make_root = [&](UINT count, ID3D12Device* owner) {
        const D3D12_ROOT_SIGNATURE_DESC desc{count, parameters_desc, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> serialized;
        check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr), "serialize root");
        ComPtr<ID3D12RootSignature> result;
        check(owner->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&result)),
              "create root");
        return result;
    };
    const UINT root_parameters = constants ? 2 : 1;
    const auto root = make_root(root_parameters, device.Get());
    const auto guest_root = make_root(root_parameters, guest.Get());
    // Same parameter 0 without the constants: a root the constant signature was not created for.
    const auto other_guest_root = make_root(1, guest.Get());
    D3D12_INDIRECT_ARGUMENT_DESC argument{D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH};
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{sizeof(D3D12_DISPATCH_ARGUMENTS), 1, &argument, 0};
    D3D12_INDIRECT_ARGUMENT_DESC constant_arguments[2]{};
    constant_arguments[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
    constant_arguments[0].Constant = {1, 0, 2};
    constant_arguments[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    const D3D12_COMMAND_SIGNATURE_DESC constant_signature_desc{
        2 * sizeof(UINT) + sizeof(D3D12_DISPATCH_ARGUMENTS), 2, constant_arguments, 0};
    ComPtr<ID3D12CommandSignature> signature;
    ComPtr<ID3D12CommandSignature> padded;
    ComPtr<ID3D12CommandSignature> placeholder;
    ComPtr<ID3D12InfoQueue> info;
    if (debug) check(device.As(&info), "get debug message queue");
    if (constants) {
        check(guest->CreateCommandSignature(&constant_signature_desc, guest_root.Get(), IID_PPV_ARGS(&signature)),
              "create guest root-constant command signature");
        auto padded_desc = constant_signature_desc;
        padded_desc.ByteStride += sizeof(UINT);
        check(guest->CreateCommandSignature(&padded_desc, guest_root.Get(), IID_PPV_ARGS(&padded)),
              "create padded root-constant command signature");
    }
    else
        check(guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature)), "create guest command signature");
    // The desktop validation error during creation is expected. The guest call itself must be clean.
    D3D12_MESSAGE_SEVERITY signature_error = D3D12_MESSAGE_SEVERITY_ERROR;
    D3D12_INFO_QUEUE_FILTER signature_filter{};
    signature_filter.DenyList.NumSeverities = 1;
    signature_filter.DenyList.pSeverityList = &signature_error;
    if (info) check(info->PushStorageFilter(&signature_filter), "mute expected signature creation diagnostic");
    argument.Type = static_cast<D3D12_INDIRECT_ARGUMENT_TYPE>(0x69);
    const HRESULT placeholder_hr = guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&placeholder));
    if (info) info->PopStorageFilter();
    check(placeholder_hr, "create placeholder command signature");
    const auto writer = pipeline(device.Get(), root.Get(), counted ? "write_count2" : constants ? "write_constant_args" : "write_args");
    const auto clamped_writer = pipeline(device.Get(), root.Get(), "write_count9");
    const auto zero_writer = pipeline(device.Get(), root.Get(), "write_count0");
    const auto one_writer = pipeline(device.Get(), root.Get(), "write_count1");
    const auto three_writer = pipeline(device.Get(), root.Get(), "write_count3");
    const auto other_writer = pipeline(device.Get(), root.Get(), constants ? "write_swapped_args" : "write_fewer_args");
    const auto consumer = guest_pipeline(guest.Get(), guest_root.Get());
    const auto verify = [&](ID3D12PipelineState* producer, const Words& record, Rejection rejection,
                            ID3D12CommandSignature* command_signature, bool follow_up = false, bool retire = false) {
        return run(device.Get(), guest.Get(), root.Get(), guest_root.Get(), other_guest_root.Get(), producer,
                   consumer.Get(), command_signature, record, rejection, follow_up, retire);
    };
    const auto run_counted = [&] {
        const Words records{0, 2, 1, 1, 1, 2, 1, 1, 1, 3};
        const auto with_count = [&](std::uint32_t count) { Words r = records; r[0] = count; return r; };
        const auto& sig = signature;
        // GPU-written counts 2, above the maximum (9), 0, 1 and 3; a zero count must leave the result untouched.
        // With the split on, the last case also resets the guest allocator before its submission retires.
        return verify(writer.Get(), with_count(2), Rejection::none, sig.Get()) &&
               verify(clamped_writer.Get(), with_count(9), Rejection::none, sig.Get()) &&
               verify(zero_writer.Get(), with_count(0), Rejection::none, sig.Get()) &&
               verify(one_writer.Get(), with_count(1), Rejection::none, sig.Get()) &&
               verify(three_writer.Get(), with_count(3), Rejection::none, sig.Get()) &&
               verify(writer.Get(), with_count(2), Rejection::none, sig.Get(), true) &&
               (!split || verify(writer.Get(), with_count(2), Rejection::none, sig.Get(), false, true)) &&
               verify(nullptr, {}, Rejection::none, sig.Get()) &&
               verify(zero_writer.Get(), with_count(0), Rejection::no_count_buffer, sig.Get()) &&
               verify(writer.Get(), with_count(2), Rejection::unmapped, sig.Get()) &&
               verify(writer.Get(), with_count(2), Rejection::count_outside, sig.Get()) &&
               verify(writer.Get(), with_count(2), Rejection::count_unaligned, sig.Get()) &&
               verify(writer.Get(), with_count(2), Rejection::records_short, sig.Get()) &&
               verify(writer.Get(), with_count(2), Rejection::placeholder, placeholder.Get());
    };
    // The second writer swaps the two constants (constants mode) or halves the X groups (default mode).
    const Words record = constants ? Words{3, 7, 4, 3, 2} : Words{4, 3, 2};
    const Words other_record = constants ? Words{7, 3, 4, 3, 2} : Words{2, 3, 2};
    if (counted ? !run_counted() : (!verify(writer.Get(), record, Rejection::none, signature.Get()) ||
        !verify(other_writer.Get(), other_record, Rejection::none, signature.Get()) ||
        !verify(nullptr, {}, Rejection::none, signature.Get()) ||
        !verify(writer.Get(), record, Rejection::unmapped, signature.Get()) ||
        !verify(writer.Get(), record, Rejection::short_range, signature.Get()) ||
        !verify(writer.Get(), record, Rejection::placeholder, placeholder.Get()) ||
        (constants && (!verify(writer.Get(), record, Rejection::other_root, signature.Get()) ||
                       !verify(writer.Get(), record, Rejection::rooted_multi, signature.Get()) ||
                       !verify(writer.Get(), record, Rejection::rooted_count_buffer, signature.Get()) ||
                       !verify(writer.Get(), record, Rejection::partial_record, signature.Get()) ||
                       !verify(writer.Get(), record, Rejection::padded_stride, padded.Get()))))) return 1;
    if (info) {
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
            SIZE_T bytes = 0;
            check(info->GetMessage(i, nullptr, &bytes), "get debug message size");
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(info->GetMessage(i, message, &bytes), "get debug message");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::fprintf(stderr, "D3D12 debug message %u: %s\n", static_cast<unsigned>(message->ID), message->pDescription);
                return 1;
            }
        }
    }
    return 0;
}
catch (const std::exception&) {
    return 3;
}
