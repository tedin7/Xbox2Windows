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
#include "startup_indirect_signature.h"
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <share.h>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

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
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
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
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameter.Constants = {0, 0, 2};
    const D3D12_ROOT_SIGNATURE_DESC root_desc{1, &parameter};
    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3D12RootSignature> root;
    auto* guest = static_cast<ID3D12Device*>(device);
    if (FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, nullptr)) ||
        FAILED(guest->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(), IID_PPV_ARGS(&root))))
        return 14;
    const ConstantDispatchSignature constants_then_dispatch(0, 0, 2);
    const auto rooted_desc = constants_then_dispatch.desc();
    ComPtr<ID3D12CommandSignature> rooted;
    if (FAILED(guest->CreateCommandSignature(&rooted_desc, root.Get(), IID_PPV_ARGS(&rooted)))) return 15;

    DeleteFileW(log_path.c_str());
    std::printf("startup_command_signature: PASS\n");
    return 0;
}
