// Source-built check for the oversized shader-visible view heap shortcut.
//
// A console view heap can ask for more CBV/SRV/UAV descriptors than a desktop driver accepts. The
// device stand-in must then create the desktop heap at the largest size the driver takes, keep the
// title's full-size memory, and log the shortcut once.
//
// The case only applies on an adapter that refuses the full request; elsewhere the shortcut is not
// needed and the test skips (77).
#include "xmem_test_device.h"

#include <cstdio>
#include <share.h>
#include <cstring>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
// The count one executable asks for: above the 1,048,576 the API documents and above every driver
// limit measured so far (1,000,000 on a GTX 1660 SUPER and an RTX 3060).
constexpr UINT requested_descriptors = 1245184;
const char* const shortcut_text = "shortcut - title asked for";

HRESULT create_wrapper(IUnknown* adapter, REFIID iid, void** out)
{
    unsigned char parameters[64]{};
    return D3D12XboxCreateDevice(adapter, parameters, iid, out);
}

SIZE_T guest_cpu_base(ID3D12DescriptorHeap* heap)
{
    // The console ABI returns this one-word handle in RAX.
    using GetCpuBase = SIZE_T(__stdcall*)(void*, void*);
    auto** table = *reinterpret_cast<void***>(heap);
    return reinterpret_cast<GetCpuBase>(table[9])(heap, nullptr);
}

int count_shortcut_lines(const wchar_t* path)
{
    // The stand-ins keep the trap log open for appending (src/startup_stubs/trap.h), so read it
    // with sharing allowed: _wfopen_s opens without sharing and fails while that handle is open.
    FILE* file = _wfsopen(path, L"rb", _SH_DENYNO);
    if (!file) return -1;
    char line[1024];
    int found = 0;
    while (std::fgets(line, sizeof(line), file))
        if (std::strstr(line, shortcut_text)) ++found;
    std::fclose(file);
    return found;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) return 1;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    bool unmatched = false;
    auto device = create_device(warp, unmatched);
    if (!device) return warp || unmatched ? 1 : 77;
    const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Does this adapter refuse the full request? If it accepts it, the shortcut is not exercised.
    D3D12_DESCRIPTOR_HEAP_DESC full{};
    full.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    full.NumDescriptors = requested_descriptors;
    full.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> direct;
    if (SUCCEEDED(device->CreateDescriptorHeap(&full, IID_PPV_ARGS(&direct)))) {
        std::printf("adapter accepts %u descriptors directly; the shortcut is not needed here\n",
                    requested_descriptors);
        return 77;
    }

    // Title memory for the whole heap, as the executable passes it in the descriptor extension.
    const SIZE_T bytes = SIZE_T{requested_descriptors} * increment;
    auto* guest = static_cast<unsigned char*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!guest) return 1;

    // The shortcut must be reported once, so watch the trap log the stand-ins append to. An earlier
    // run's file is removed first: the log is appended, and a stale line would read as a second
    // report.
    wchar_t log_path[MAX_PATH]{};
    if (argc == 3) _snwprintf_s(log_path, MAX_PATH, _TRUNCATE, L"%hs", argv[2]);
    if (log_path[0]) {
        _wremove(log_path);
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path);
    }
    SetEnvironmentVariableW(L"X2W_D3D12_PERMISSIVE", L"1");

    auto wrapper = create_device(warp, unmatched, create_wrapper);
    if (!wrapper) {
        VirtualFree(guest, 0, MEM_RELEASE);
        return 1;
    }

    struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        SIZE_T guest_address;
        SIZE_T guest_bytes;
    } heap{};
    heap.desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.desc.NumDescriptors = requested_descriptors;
    heap.desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    heap.guest_address = reinterpret_cast<SIZE_T>(guest);
    heap.guest_bytes = bytes;

    void* unsupported = nullptr;
    const HRESULT unsupported_hr = wrapper->CreateDescriptorHeap(&heap.desc, __uuidof(ID3D12Resource), &unsupported);
    if (unsupported_hr != E_NOINTERFACE || unsupported) {
        std::printf("oversized request accepted an unsupported heap interface: 0x%08lx\n",
                    static_cast<unsigned long>(unsupported_hr));
        VirtualFree(guest, 0, MEM_RELEASE);
        return 1;
    }

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> wrapped;
    HRESULT hr = wrapper->CreateDescriptorHeap(&heap.desc, IID_PPV_ARGS(&wrapped));
    int status = 0;
    if (FAILED(hr)) {
        std::printf("oversized heap refused: 0x%08lx\n", static_cast<unsigned long>(hr));
        status = 1;
    } else if (guest_cpu_base(wrapped.Get()) != heap.guest_address) {
        std::puts("oversized heap lost its logical CPU descriptor base");
        status = 1;
    } else {
        // A second oversized request must not log the shortcut again. It carries its own extension
        // words, so the stand-in takes the same path rather than reading stack data.
        struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            SIZE_T guest_address;
            SIZE_T guest_bytes;
        } second_desc{};
        second_desc.desc = heap.desc;
        second_desc.guest_address = heap.guest_address;
        second_desc.guest_bytes = heap.guest_bytes;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> second;
        if (FAILED(wrapper->CreateDescriptorHeap(&second_desc.desc, IID_PPV_ARGS(&second)))) {
            std::puts("second oversized heap request failed");
            status = 1;
        } else if (log_path[0]) {
            const int lines = count_shortcut_lines(log_path);
            if (lines != 1) {
                std::printf("shortcut logged %d times, expected 1\n", lines);
                status = 1;
            } else {
                std::printf("oversized heap accepted; shortcut logged once; %u requested\n", requested_descriptors);
            }
        } else {
            // Without a log path the behavioural check still stands.
            std::printf("oversized heap accepted; %u requested\n", requested_descriptors);
        }
    }
    VirtualFree(guest, 0, MEM_RELEASE);
    return status;
}
