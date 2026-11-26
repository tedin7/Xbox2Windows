// Device slot 41 (CreateCommandSignature) through the startup device wrapper, with the desktop debug layer on.
//
//   startup_command_signature <warp|hardware>     (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_DEBUG=1)
//
//   control  the desktop device refuses an argument type of 0x69 and its debug layer reports message 743
//            ("unexpected parameter type"), so a missing message below means something
//   valid    one DRAW_INDEXED argument: S_OK and an object, and no debug layer message 743
//   console  one argument of type 0x69, stride 12: S_OK and an object (the placeholder), no message 743
//
// The wrapper copies the debug layer's messages into its log after forwarded calls (X2W_TRAP_LOG); the
// check reads the message ids from there. The run is on Windows only: it needs the debug layer.
//
// Exit codes: 0 pass, 77 no such adapter or no debug layer (or one that does not report the control),
// other values name the failing step.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "startup_indirect_writer.h"

#include <algorithm>
#include <array>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <share.h>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

extern "C" __declspec(dllimport) BOOL x2w_d3d12_fixture_compute_root_uav(void*, UINT, UINT64);
extern "C" __declspec(dllimport) void x2w_d3d12_fixture_console_indirect_totals();

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d);
}

constexpr int unexpected_parameter_type = 743;
constexpr unsigned console_argument_type = 0x69;

// Whether the desktop device's stored messages hold the id.
bool stored_message(ID3D12Device* device, int id)
{
    ComPtr<ID3D12InfoQueue> queue;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&queue)))) return false;
    for (UINT64 i = 0, n = queue->GetNumStoredMessages(); i < n; ++i) {
        SIZE_T bytes = 0;
        if (FAILED(queue->GetMessage(i, nullptr, &bytes))) continue;
        std::vector<char> storage(bytes);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (SUCCEEDED(queue->GetMessage(i, message, &bytes)) && static_cast<int>(message->ID) == id) return true;
    }
    return false;
}

// The wrapper's log lines that carry a debug layer message with the id.
int logged_messages(const std::wstring& path, int id)
{
    FILE* log = _wfsopen(path.c_str(), L"rb", _SH_DENYNO);
    if (!log) return -1;
    const std::string wanted = "id=" + std::to_string(id) + " ";
    int found = 0;
    char line[2048];
    while (std::fgets(line, sizeof(line), log))
        if (std::strstr(line, "x2w-d3d12-debug") && std::strstr(line, wanted.c_str())) ++found;
    if (std::ferror(log)) found = -1;
    std::fclose(log);
    return found;
}

HRESULT create_signature(void* device, const D3D12_COMMAND_SIGNATURE_DESC& desc, void** out)
{
    *out = nullptr;
    const GUID iid = __uuidof(ID3D12CommandSignature);
    return static_cast<HRESULT>(invoke(device, 41, arg(&desc), 0, arg(&iid), arg(out)));
}
template <class Method>
Method slot(void* object, unsigned index)
{
    return reinterpret_cast<Method>((*static_cast<void***>(object))[index]);
}

bool console_indirect_readback(ID3D12Device* desktop, ID3D12Device* guest, IUnknown* signature,
                               D3D12_COMMAND_SIGNATURE_DESC desc, const std::wstring& log_path,
                               bool enabled, bool mutant)
{
    constexpr char shader[] = R"(
RWByteAddressBuffer output : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    output.Store(4 * (id.x + 4 * id.y + 16 * id.z), 1 + id.x + 16 * id.y + 256 * id.z);
}
)";
    ComPtr<ID3DBlob> code;
    check(D3DCompile(shader, sizeof(shader) - 1, nullptr, nullptr, nullptr, "main", "cs_5_1",
                     0, 0, &code, nullptr), "compile indirect consumer");
    D3D12_ROOT_PARAMETER uav{};
    uav.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    uav.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    const D3D12_ROOT_SIGNATURE_DESC root_desc{1, &uav, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> serialized;
    check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr), "serialize root");
    ComPtr<ID3D12RootSignature> root;
    check(guest->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&root)),
          "create guest root");
    struct {
        alignas(void*) D3D12_PIPELINE_STATE_SUBOBJECT_TYPE root_type;
        ID3D12RootSignature* root;
        alignas(void*) D3D12_PIPELINE_STATE_SUBOBJECT_TYPE shader_type;
        D3D12_SHADER_BYTECODE shader;
    } stream{D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.Get(), D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS,
             {code->GetBufferPointer(), code->GetBufferSize()}};
    const D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(stream), &stream};
    using CreatePipeline = HRESULT (WINAPI*)(void*, const void*, const GUID*, void**);
    ComPtr<IUnknown> compute, graphics;
    const auto pipeline_iid = __uuidof(ID3D12PipelineState);
    check(slot<CreatePipeline>(guest, 86)(guest, &stream_desc, &pipeline_iid,
          reinterpret_cast<void**>(compute.GetAddressOf())), "create guest compute pipeline");
    // A synthetic graphics placeholder exercises the console graphics binding on the same list.
    const std::array<std::uint64_t, 96> graphics_desc{};
    if (!enabled)
        check(slot<CreatePipeline>(guest, 57)(guest, graphics_desc.data(), &pipeline_iid,
              reinterpret_cast<void**>(graphics.GetAddressOf())), "create guest graphics placeholder");

    desc.ByteStride = 16;
    void* padded_object = nullptr;
    check(create_signature(guest, desc, &padded_object), "create stride-16 placeholder");
    ComPtr<IUnknown> padded;
    padded.Attach(static_cast<IUnknown*>(padded_object));

    // The fourth word makes the offset mutant read (2,1,2) from valid buffer storage.
    constexpr std::array<std::uint32_t, 4> groups{3, 2, 1, 2};
    constexpr UINT argument_bytes = sizeof(groups);
    constexpr UINT region_words = 64;
    constexpr UINT region_bytes = region_words * sizeof(std::uint32_t);
    constexpr UINT result_bytes = 3 * region_bytes;
    constexpr std::uint32_t sentinel = 0x6fbe1de5;
    using Output = std::array<std::uint32_t, 3 * region_words>;
    Output initial;
    initial.fill(sentinel);
    auto upload = make_buffer(desktop, argument_bytes + result_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto result = make_buffer(desktop, result_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = make_buffer(desktop, result_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    void* mapped = nullptr;
    const D3D12_RANGE untouched{0, 0};
    check(upload->Map(0, &untouched, &mapped), "map upload");
    std::memcpy(mapped, groups.data(), argument_bytes);
    std::memcpy(static_cast<char*>(mapped) + argument_bytes, initial.data(), result_bytes);
    upload->Unmap(0, nullptr);

    const auto free_memory = [](void* memory) { VirtualFree(memory, 0, MEM_RELEASE); };
    const std::unique_ptr<void, decltype(free_memory)> backing(
        VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE), free_memory);
    if (!backing) throw std::bad_alloc();
    const D3D12_RESOURCE_DESC buffer_desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, argument_bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    using Place = HRESULT (WINAPI*)(void*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
    ComPtr<IUnknown> arguments;
    check(slot<Place>(guest, 133)(guest, static_cast<char*>(backing.get()) + 4, &buffer_desc, D3D12_RESOURCE_STATE_COMMON,
          nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(arguments.GetAddressOf())), "place arguments");

    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create guest allocator");
    using CreateList = HRESULT (WINAPI*)(void*, const void*, IUnknown*, void*, REFIID, void**);
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    ComPtr<IUnknown> guest_list;
    check(slot<CreateList>(guest, 73)(guest, list_desc, allocator.Get(), nullptr, __uuidof(ID3D12GraphicsCommandList),
          reinterpret_cast<void**>(guest_list.GetAddressOf())), "create guest list");
    using Reset = HRESULT (WINAPI*)(void*, IUnknown*, void*, void*);
    check(slot<Reset>(guest_list.Get(), 139)(guest_list.Get(), allocator.Get(), nullptr, nullptr), "reset guest list");
    const auto list = indirect_native_object<ID3D12GraphicsCommandList>(desktop, guest, guest_list.Get());
    using Copy = void (WINAPI*)(void*, IUnknown*, UINT64, IUnknown*, UINT64, UINT64);
    slot<Copy>(guest_list.Get(), 15)(guest_list.Get(), arguments.Get(), 0, upload.Get(), 0, argument_bytes);
    list->CopyBufferRegion(result.Get(), 0, upload.Get(), argument_bytes, result_bytes);
    transition(list.Get(), result.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    using Bind = void (WINAPI*)(void*, IUnknown*);
    using Execute = void (WINAPI*)(void*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
    const auto execute = slot<Execute>(guest_list.Get(), 61);
    const auto bind_pipeline = slot<Bind>(guest_list.Get(), 25);
    slot<Bind>(guest_list.Get(), 33)(guest_list.Get(), root.Get());
    const auto bind_output = [&](UINT region) {
        if (!x2w_d3d12_fixture_compute_root_uav(guest_list.Get(), 0, result->GetGPUVirtualAddress() + region * region_bytes))
            throw std::runtime_error("bind fixture root UAV");
    };
    // No pipeline has been bound on this list yet.
    if (enabled) {
        bind_output(2);
        execute(guest_list.Get(), signature, 1, arguments.Get(), 0, nullptr, 0);
    }
    bind_pipeline(guest_list.Get(), compute.Get());
    bind_output(0);
    execute(guest_list.Get(), signature, 1, arguments.Get(), mutant ? sizeof(UINT) : 0, nullptr, 0);
    bind_output(1);
    if (!enabled) bind_pipeline(guest_list.Get(), graphics.Get());
    execute(guest_list.Get(), enabled ? padded.Get() : signature, 1, arguments.Get(), 0, nullptr, 0);

    transition(list.Get(), result.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, result.Get(), 0, result_bytes);
    using Close = HRESULT (WINAPI*)(void*);
    check(slot<Close>(guest_list.Get(), 9)(guest_list.Get()), "close guest list");
    ComPtr<ID3D12CommandQueue> queue;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    check(desktop->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "create queue");
    ID3D12CommandList* submitted[]{list.Get()};
    queue->ExecuteCommandLists(1, submitted);
    ComPtr<ID3D12Fence> fence;
    check(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "create fence");
    check(queue->Signal(fence.Get(), 1), "signal fence");
    check(fence->SetEventOnCompletion(1, nullptr), "wait for readback");
    const D3D12_RANGE read{0, result_bytes};
    check(readback->Map(0, &read, &mapped), "map readback");
    Output actual;
    std::memcpy(actual.data(), mapped, result_bytes);
    readback->Unmap(0, &untouched);

    const auto reference = [&](UINT x_count, UINT y_count, UINT z_count) {
        Output expected = initial;
        for (UINT z = 0; z < z_count; ++z)
            for (UINT y = 0; y < y_count; ++y)
                for (UINT x = 0; x < x_count; ++x)
                    expected[x + 4 * y + 16 * z] = 1 + x + 16 * y + 256 * z;
        return expected;
    };
    const auto expected = enabled ? reference(3, 2, 1) : initial;
    const auto wanted = mutant ? reference(2, 1, 2) : expected;
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), wanted.begin());
    if (mismatch.first != actual.end()) {
        std::printf("readback word %td: got 0x%08x, expected 0x%08x\n", mismatch.first - actual.begin(),
                    *mismatch.first, *mismatch.second);
        return false;
    }
    if (mutant) {
        if (actual == expected) return false;
        std::puts("offset-4 control: readback differs from the unshifted CPU reference");
    }

    check(desktop->GetDeviceRemovedReason(), "get device status after readback");
    ComPtr<ID3D12InfoQueue> info;
    check(desktop->QueryInterface(IID_PPV_ARGS(&info)), "read debug messages after submission");
    for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
        SIZE_T bytes = 0;
        check(info->GetMessage(i, nullptr, &bytes), "get debug message size");
        std::vector<char> storage(bytes);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        check(info->GetMessage(i, message, &bytes), "get debug message");
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
            std::fprintf(stderr, "D3D12 debug error %u: %s\n", static_cast<unsigned>(message->ID), message->pDescription);
            return false;
        }
    }
    x2w_d3d12_fixture_console_indirect_totals();
    FILE* log = _wfsopen(log_path.c_str(), L"rb", _SH_DENYNO);
    if (!log) return false;
    unsigned rows = 0, compute_rows = 0, other_rows = 0;
    bool debug_error = false;
    const char* other = enabled ? "pipeline=none " : "pipeline=console-graphics ";
    char line[2048];
    while (std::fgets(line, sizeof(line), log)) {
        if (std::strstr(line, "x2w-d3d12-debug") && (std::strstr(line, " sev=0 ") || std::strstr(line, " sev=1 "))) {
            std::fputs(line, stderr);
            debug_error = true;
        }
        if (!std::strstr(line, "x2w-render indirect console-signature:")) continue;
        ++rows;
        if (!std::strstr(line, "list=direct ") ||
            !std::strstr(line, "desktop=1 dispatch-write=0 copy-write=1 calls=1")) continue;
        if (std::strstr(line, "pipeline=desktop-compute ")) ++compute_rows;
        if (std::strstr(line, other)) ++other_rows;
    }
    std::fclose(log);
    if (debug_error) return false;
    if (rows == 2 && compute_rows == 1 && other_rows == 1) {
        std::printf("readback and refusal totals matched (%s)\n", !enabled ? "off" : mutant ? "offset-4 control" : "on");
        return true;
    }
    std::printf("unexpected console signature totals: %u rows, %u compute, %u other\n", rows, compute_rows, other_rows);
    return false;
}

}  // namespace

int main(int argc, char** argv) try
{
    if (argc != 2 && argc != 3) return 2;
    const bool indirect = argc == 3;
    const bool mutant = indirect && std::strcmp(argv[2], "offset-control") == 0;
    const bool enabled = mutant || (indirect && std::strcmp(argv[2], "indirect-on") == 0);
    if (indirect && !enabled && std::strcmp(argv[2], "indirect-off") != 0) return 2;
    SetEnvironmentVariableA("X2W_CONSOLE_INDIRECT_12", enabled ? "1" : nullptr);
    if (indirect) SetEnvironmentVariableA("X2W_D3D12_RENDER", "1");
    ComPtr<ID3D12Debug> debug;
    if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
        std::printf("no debug layer: skipped\n");
        return 77;
    }
    debug->EnableDebugLayer();

    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(argv[1], "hardware") == 0) {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    } else if (std::strcmp(argv[1], "warp") != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 3;
    }

    wchar_t directory[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, directory)) return 4;
    const std::wstring log_path = std::wstring(directory) + L"x2w_command_signature_" + std::to_wstring(GetCurrentProcessId()) + L".log";
    DeleteFileW(log_path.c_str());
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());

    D3D12_INDIRECT_ARGUMENT_DESC console{};
    console.Type = static_cast<D3D12_INDIRECT_ARGUMENT_TYPE>(console_argument_type);
    D3D12_INDIRECT_ARGUMENT_DESC draw_indexed{};
    draw_indexed.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
    D3D12_COMMAND_SIGNATURE_DESC console_desc{12, 1, &console, 0};
    D3D12_COMMAND_SIGNATURE_DESC valid_desc{sizeof(D3D12_DRAW_INDEXED_ARGUMENTS), 1, &draw_indexed, 0};

    // Control: the layer reports the console type on a plain desktop device.
    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 4;
    if (indirect) {
        const std::array<unsigned char, 64> parameters{};
        ComPtr<ID3D12Device> guest;
        check(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), IID_PPV_ARGS(&guest)), "create guest device");
        void* object = nullptr;
        check(create_signature(guest.Get(), console_desc, &object), "create console placeholder");
        ComPtr<IUnknown> signature;
        signature.Attach(static_cast<IUnknown*>(object));
        const bool passed = console_indirect_readback(desktop.Get(), guest.Get(), signature.Get(), console_desc,
                                                      log_path, enabled, mutant);
        if (passed) DeleteFileW(log_path.c_str());
        return passed ? 0 : 1;
    }
    ComPtr<ID3D12CommandSignature> refused;
    const HRESULT desktop_hr = desktop->CreateCommandSignature(&console_desc, nullptr, IID_PPV_ARGS(&refused));
    if (SUCCEEDED(desktop_hr) || !stored_message(desktop.Get(), unexpected_parameter_type)) {
        std::printf("control: desktop hr 0x%08lx, message %d not reported: skipped\n", static_cast<unsigned long>(desktop_hr),
                    unexpected_parameter_type);
        return 77;
    }
    std::printf("control: the desktop device refuses type 0x%x and the layer reports message %d\n", console_argument_type,
                unexpected_parameter_type);
    // The guest wrapper can open the same underlying device. Remove the control's own message first.
    ComPtr<ID3D12InfoQueue> control_queue;
    if (FAILED(desktop.As(&control_queue))) return 4;
    control_queue->ClearStoredMessages();

    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device)) || !device) return 5;

    void* created = nullptr;
    HRESULT hr = create_signature(device, valid_desc, &created);
    std::printf("valid DRAW_INDEXED: hr 0x%08lx object %p\n", static_cast<unsigned long>(hr), created);
    if (FAILED(hr) || !created) return 10;
    if (const int seen = logged_messages(log_path, unexpected_parameter_type)) {
        if (seen < 0) std::puts("FAIL valid DRAW_INDEXED: could not read debug log");
        else std::printf("FAIL valid DRAW_INDEXED: %d debug layer message(s) %d\n", seen, unexpected_parameter_type);
        return 11;
    }

    hr = create_signature(device, console_desc, &created);
    std::printf("console type 0x%x: hr 0x%08lx object %p\n", console_argument_type, static_cast<unsigned long>(hr), created);
    if (FAILED(hr) || !created) return 12;
    if (const int seen = logged_messages(log_path, unexpected_parameter_type)) {
        if (seen < 0) std::puts("FAIL console type: could not read debug log");
        else std::printf("FAIL console type: %d debug layer message(s) %d\n", seen, unexpected_parameter_type);
        return 13;
    }
    DeleteFileW(log_path.c_str());
    std::printf("startup_command_signature: PASS\n");
    return 0;
}
catch (const std::exception& failure) {
    std::fprintf(stderr, "fixture setup failed: %s\n", failure.what());
    return 14;
}
