// Calls the desktop d3d12_xs.dll through named imports, as the package DLLs do.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

#include "pso_compute_embedded.h"
#include "pso_compute_stripped.h"

using Microsoft::WRL::ComPtr;
using namespace std::literals;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
constexpr std::uint32_t observed_console_flags = 0x03000000u;

struct DesktopD3D12 {
    HMODULE module = nullptr;
    PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE serialize = nullptr;
    PFN_D3D12_CREATE_DEVICE create_device = nullptr;

    bool load()
    {
        module = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module) return false;
        serialize = reinterpret_cast<PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE>(
            GetProcAddress(module, "D3D12SerializeVersionedRootSignature"));
        create_device = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
            GetProcAddress(module, "D3D12CreateDevice"));
        return serialize && create_device;
    }

    ~DesktopD3D12()
    {
        if (module) FreeLibrary(module);
    }
};

bool same_blob(ID3DBlob* a, ID3DBlob* b)
{
    return a && b &&
           a->GetBufferSize() == b->GetBufferSize() &&
           std::memcmp(a->GetBufferPointer(), b->GetBufferPointer(), a->GetBufferSize()) == 0;
}

D3D12_ROOT_PARAMETER1 make_constants_param1()
{
    D3D12_ROOT_PARAMETER1 param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants.Num32BitValues = 4;
    return param;
}

bool export_checks()
{
    HMODULE shim = GetModuleHandleW(L"d3d12_xs.dll");
    if (!shim) return false;
    DesktopD3D12 desktop;
    if (!desktop.load()) return false;

    unsigned found = 0;
    for (const char* name : {"D3D12SerializeVersionedRootSignature", "D3D12CreateRootSignatureDeserializer",
                             "D3D12ConfigureVirtualMemoryX"}) {
        FARPROC proc = GetProcAddress(shim, name);
        if (!proc) return false;
        HMODULE owner = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(proc), &owner) || owner != shim) {
            return false;
        }
        ++found;
    }

    // Prove this executable's named import resolves to d3d12_xs.dll rather than desktop d3d12.dll:
    // desktop d3d12.dll rejects 0x03000000 while d3d12_xs.dll strips it and succeeds.
    D3D12_ROOT_PARAMETER1 param = make_constants_param1();
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = 1;
    desc.Desc_1_1.pParameters = &param;
    desc.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(observed_console_flags);

    ComPtr<ID3DBlob> desktop_blob, shim_blob;
    const HRESULT desktop_hr = desktop.serialize(&desc, &desktop_blob, nullptr);
    const HRESULT shim_hr = D3D12SerializeVersionedRootSignature(&desc, &shim_blob, nullptr);
    std::printf("exports=%u desktop_tagged_hr=0x%08lx shim_tagged_hr=0x%08lx\n",
                found, static_cast<unsigned long>(desktop_hr), static_cast<unsigned long>(shim_hr));
    return desktop_hr == E_INVALIDARG && !desktop_blob && SUCCEEDED(shim_hr) && shim_blob;
}

bool serialize_checks()
{
    DesktopD3D12 desktop;
    if (!desktop.load()) return false;

    constexpr auto desktop_flag = D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS;
    D3D12_ROOT_PARAMETER1 param1 = make_constants_param1();
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC base_v11{};
    base_v11.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    base_v11.Desc_1_1.NumParameters = 1;
    base_v11.Desc_1_1.pParameters = &param1;
    base_v11.Desc_1_1.Flags = desktop_flag;

    ComPtr<ID3DBlob> desktop_v11, shim_v11;
    const HRESULT base_desktop_hr = desktop.serialize(&base_v11, &desktop_v11, nullptr);
    const HRESULT base_shim_hr = D3D12SerializeVersionedRootSignature(&base_v11, &shim_v11, nullptr);
    if (FAILED(base_desktop_hr) || FAILED(base_shim_hr) || !same_blob(desktop_v11.Get(), shim_v11.Get())) {
        std::fprintf(stderr, "v1.1 baseline: desktop=0x%08lx shim=0x%08lx\n",
                     static_cast<unsigned long>(base_desktop_hr), static_cast<unsigned long>(base_shim_hr));
        return false;
    }

    // Each observed console bit alone and as a pair must be rejected by desktop d3d12.dll and
    // serialized identically to the desktop baseline by d3d12_xs.dll without mutating caller input.
    for (const std::uint32_t bits : {0x01000000u, 0x02000000u, observed_console_flags}) {
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC tagged = base_v11;
        const auto expected_flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(
            static_cast<std::uint32_t>(desktop_flag) | bits);
        tagged.Desc_1_1.Flags = expected_flags;

        ComPtr<ID3DBlob> rejected, sanitized;
        const HRESULT desktop_hr = desktop.serialize(&tagged, &rejected, nullptr);
        const HRESULT shim_hr = D3D12SerializeVersionedRootSignature(&tagged, &sanitized, nullptr);
        if (desktop_hr != E_INVALIDARG || rejected || FAILED(shim_hr) ||
            !same_blob(sanitized.Get(), desktop_v11.Get()) ||
            tagged.Desc_1_1.Flags != expected_flags) {
            std::fprintf(stderr, "v1.1 tagged 0x%08x: desktop=0x%08lx shim=0x%08lx\n",
                         bits, static_cast<unsigned long>(desktop_hr), static_cast<unsigned long>(shim_hr));
            return false;
        }
    }

    // v1.0 descriptor with the same console bits at +0x28.
    D3D12_ROOT_PARAMETER param0{};
    param0.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param0.Constants.Num32BitValues = 4;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC base_v10{};
    base_v10.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    base_v10.Desc_1_0.NumParameters = 1;
    base_v10.Desc_1_0.pParameters = &param0;
    base_v10.Desc_1_0.Flags = desktop_flag;
    ComPtr<ID3DBlob> desktop_v10, rejected_v10, shim_v10;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC tagged_v10 = base_v10;
    tagged_v10.Desc_1_0.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(
        static_cast<std::uint32_t>(desktop_flag) | observed_console_flags);
    const HRESULT v10_base_hr = desktop.serialize(&base_v10, &desktop_v10, nullptr);
    const HRESULT v10_desktop_hr = desktop.serialize(&tagged_v10, &rejected_v10, nullptr);
    const HRESULT v10_shim_hr = D3D12SerializeVersionedRootSignature(&tagged_v10, &shim_v10, nullptr);
    if (FAILED(v10_base_hr) || v10_desktop_hr != E_INVALIDARG || rejected_v10 ||
        FAILED(v10_shim_hr) || !same_blob(shim_v10.Get(), desktop_v10.Get())) {
        std::fprintf(stderr, "v1.0 tagged: base=0x%08lx desktop=0x%08lx shim=0x%08lx\n",
                     static_cast<unsigned long>(v10_base_hr),
                     static_cast<unsigned long>(v10_desktop_hr),
                     static_cast<unsigned long>(v10_shim_hr));
        return false;
    }

    // Negative controls: unobserved invalid flag bits must still return E_INVALIDARG, and null
    // descriptor or unknown version must fail with the exact same HRESULT as desktop d3d12.dll.
    for (const std::uint32_t bad_bits : {0x04000000u, 0x07000000u, 0x08000000u}) {
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC bad = base_v11;
        bad.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(bad_bits);
        ComPtr<ID3DBlob> desktop_bad, shim_bad;
        const HRESULT desktop_hr = desktop.serialize(&bad, &desktop_bad, nullptr);
        const HRESULT shim_hr = D3D12SerializeVersionedRootSignature(&bad, &shim_bad, nullptr);
        if (desktop_hr != E_INVALIDARG || desktop_bad || shim_hr != E_INVALIDARG || shim_bad) {
            std::fprintf(stderr, "bad_bits 0x%08x: desktop=0x%08lx shim=0x%08lx\n",
                         bad_bits, static_cast<unsigned long>(desktop_hr), static_cast<unsigned long>(shim_hr));
            return false;
        }
    }
    ComPtr<ID3DBlob> desktop_null, shim_null;
    const HRESULT desktop_null_hr = desktop.serialize(nullptr, &desktop_null, nullptr);
    const HRESULT shim_null_hr = D3D12SerializeVersionedRootSignature(nullptr, &shim_null, nullptr);
    if (!FAILED(desktop_null_hr) || desktop_null || shim_null_hr != desktop_null_hr || shim_null) {
        std::fprintf(stderr, "null desc: desktop=0x%08lx shim=0x%08lx\n",
                     static_cast<unsigned long>(desktop_null_hr), static_cast<unsigned long>(shim_null_hr));
        return false;
    }
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC bad_ver{};
    bad_ver.Version = static_cast<D3D_ROOT_SIGNATURE_VERSION>(99);
    ComPtr<ID3DBlob> desktop_ver, shim_ver;
    const HRESULT desktop_ver_hr = desktop.serialize(&bad_ver, &desktop_ver, nullptr);
    const HRESULT shim_ver_hr = D3D12SerializeVersionedRootSignature(&bad_ver, &shim_ver, nullptr);
    if (!FAILED(desktop_ver_hr) || desktop_ver || shim_ver_hr != desktop_ver_hr || shim_ver) {
        std::fprintf(stderr, "bad version: desktop=0x%08lx shim=0x%08lx\n",
                     static_cast<unsigned long>(desktop_ver_hr), static_cast<unsigned long>(shim_ver_hr));
        return false;
    }
    std::printf("v11_cases=4 v10_cases=1 negative_cases=5 null_hr=0x%08lx bad_ver_hr=0x%08lx\n",
                static_cast<unsigned long>(shim_null_hr), static_cast<unsigned long>(shim_ver_hr));
    return true;
}

// The RTS0 flags word of a serialized root signature, ~0u if the container has none.
std::uint32_t rts0_flags(const unsigned char* data, std::size_t bytes)
{
    std::uint32_t parts = 0;
    if (bytes < 32 || std::memcmp(data, "DXBC", 4) != 0) return ~0u;
    std::memcpy(&parts, data + 28, 4);
    for (std::uint32_t i = 0; i < parts && 32 + 4 * (i + 1) <= bytes; ++i) {
        std::uint32_t at = 0, flags = 0;
        std::memcpy(&at, data + 32 + 4 * i, 4);
        if (std::size_t{at} + 8 + 24 > bytes || std::memcmp(data + at, "RTS0", 4) != 0) continue;
        std::memcpy(&flags, data + at + 8 + 20, 4);
        return flags;
    }
    return ~0u;
}

// Provenance (switch on, retry reachable). The baseline without heap-indexing bits and inputs with
// only 0x400, only 0x800 and both all serialize to the same bytes, so the bytes cannot say which call
// produced a blob. All four blobs are alive at once, in both call orders; each goes through the
// recording device's CreateRootSignature, and the copy that device keeps for shader translation must
// carry exactly its own call's bits. A copy of the bytes in other memory gets none.
// The flags command-list slot 34 acts on (src/startup_stubs/heap_indexed_root.h) come from the desktop
// blob, which has no heap-indexing bits after a retry: the desktop root signature does not index heaps,
// so slot 34 must set it directly (recorded flags 0), whatever the translation copy carries.
bool provenance_checks(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& base, std::uint32_t(__cdecl* stripped_bits)(const void*, std::size_t))
{
    using TranslationBlob = std::size_t(__cdecl*)(void*, void*, std::size_t, UINT*);
    const auto translation_blob = reinterpret_cast<TranslationBlob>(
        GetProcAddress(GetModuleHandleW(L"d3d12_xs.dll"), "x2w_d3d12_root_translation_blob"));
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> warp;
    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (!translation_blob || FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) ||
        FAILED(D3D12XboxCreateDevice(warp.Get(), parameters.data(), __uuidof(ID3D12Device), &device)) || !device) {
        std::fprintf(stderr, "provenance: no recording device (translation export %p)\n", reinterpret_cast<void*>(translation_blob));
        return false;
    }
    // Device slot 16: CreateRootSignature(node, blob, size, riid, ppv).
    const auto create_root = [device](const void* blob, std::size_t size, void** root) {
        using Slot = HRESULT(WINAPI*)(void*, UINT, const void*, SIZE_T, const GUID*, void**);
        return reinterpret_cast<Slot>((*static_cast<void***>(device))[16])(device, 0, blob, size, &__uuidof(ID3D12RootSignature), root);
    };
    // The flags of the copy the recording device kept for translation, ~0u if it kept none; desktop gets
    // the heap-indexing flags slot 34 acts on.
    const auto translation_flags = [translation_blob](void* root, UINT* desktop) {
        std::array<unsigned char, 4096> copy{};
        *desktop = ~0u;
        const std::size_t bytes = translation_blob(root, copy.data(), copy.size(), desktop);
        return bytes && bytes <= copy.size() ? rts0_flags(copy.data(), bytes) : ~0u;
    };

    const auto base_flags = static_cast<std::uint32_t>(base.Desc_1_1.Flags);
    bool ok = true;
    for (const std::array<std::uint32_t, 4>& order :
         {std::array<std::uint32_t, 4>{0u, 0x400u, 0x800u, 0xC00u}, std::array<std::uint32_t, 4>{0xC00u, 0x800u, 0x400u, 0u}}) {
        std::array<ComPtr<ID3DBlob>, 4> blobs;
        for (std::size_t i = 0; i < order.size(); ++i) {
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc = base;
            desc.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(base_flags | order[i]);
            if (FAILED(D3D12SerializeVersionedRootSignature(&desc, &blobs[i], nullptr)) || !blobs[i]) {
                std::fprintf(stderr, "provenance: serialization of 0x%x failed\n", order[i]);
                return false;
            }
        }
        for (std::size_t i = 0; i < order.size(); ++i) {
            ComPtr<IUnknown> root;
            const HRESULT hr = create_root(blobs[i]->GetBufferPointer(), blobs[i]->GetBufferSize(), reinterpret_cast<void**>(root.GetAddressOf()));
            const std::uint32_t lookup = stripped_bits(blobs[i]->GetBufferPointer(), blobs[i]->GetBufferSize());
            UINT desktop = ~0u;
            const std::uint32_t used = SUCCEEDED(hr) && root ? translation_flags(root.Get(), &desktop) : ~0u;
            const bool same_bytes = same_blob(blobs[i].Get(), blobs[0].Get());
            const bool good = SUCCEEDED(hr) && same_bytes && lookup == order[i] && used == (base_flags | order[i]) && desktop == 0;
            std::printf("provenance order=%s input=0x%03x: create=0x%08lx same_bytes=%d lookup=0x%03x translation_flags=0x%08x "
                        "slot34_heap_flags=0x%x %s\n",
                        order[0] == 0 ? "baseline-first" : "both-first", order[i], static_cast<unsigned long>(hr), same_bytes ? 1 : 0,
                        lookup, used, desktop, good ? "ok" : "WRONG");
            ok = ok && good;
        }
        // The same bytes copied elsewhere carry no provenance: the translation copy keeps the
        // desktop flags rather than guess which call they came from.
        std::vector<unsigned char> copied(static_cast<const unsigned char*>(blobs[3]->GetBufferPointer()),
                                          static_cast<const unsigned char*>(blobs[3]->GetBufferPointer()) + blobs[3]->GetBufferSize());
        ComPtr<IUnknown> root;
        const HRESULT hr = create_root(copied.data(), copied.size(), reinterpret_cast<void**>(root.GetAddressOf()));
        UINT desktop = ~0u;
        const std::uint32_t used = SUCCEEDED(hr) && root ? translation_flags(root.Get(), &desktop) : ~0u;
        const bool good = SUCCEEDED(hr) && stripped_bits(copied.data(), copied.size()) == 0 && used == base_flags && desktop == 0;
        std::printf("provenance copy of 0x%03x: translation_flags=0x%08x slot34_heap_flags=0x%x %s\n", order[3], used, desktop,
                    good ? "ok" : "WRONG");
        ok = ok && good;
    }
    static_cast<IUnknown*>(device)->Release();
    return ok;
}

// Heap-indexing flags (CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED). A desktop
// runtime older than shader model 6.6 refuses them with E_INVALIDARG; with X2W_D3D12_PERMISSIVE=1
// d3d12_xs.dll serializes again without them and records which bits it removed for that blob.
// Returns 0 (pass), 1 (fail) or 77 when the desktop runtime accepts the flags (retry unreachable).
int heap_indexing_checks()
{
    constexpr std::uint32_t heap_indexing = 0x00000C00u;
    DesktopD3D12 desktop;
    if (!desktop.load()) return 1;
    using Stripped = std::uint32_t(__cdecl*)(const void*, std::size_t);
    const auto stripped_bits = reinterpret_cast<Stripped>(
        GetProcAddress(GetModuleHandleW(L"d3d12_xs.dll"), "x2w_stripped_root_flags"));
    if (!stripped_bits) {
        std::fprintf(stderr, "x2w_stripped_root_flags not exported\n");
        return 1;
    }

    constexpr auto desktop_flag = D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS;
    D3D12_ROOT_PARAMETER1 param = make_constants_param1();
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC base{};
    base.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    base.Desc_1_1.NumParameters = 1;
    base.Desc_1_1.pParameters = &param;
    base.Desc_1_1.Flags = desktop_flag;
    ComPtr<ID3DBlob> baseline;
    if (FAILED(desktop.serialize(&base, &baseline, nullptr)) || !baseline) return 1;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC indexed = base;
    const auto indexed_flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(static_cast<std::uint32_t>(desktop_flag) | heap_indexing);
    indexed.Desc_1_1.Flags = indexed_flags;
    ComPtr<ID3DBlob> desktop_indexed;
    const HRESULT desktop_hr = desktop.serialize(&indexed, &desktop_indexed, nullptr);
    SetEnvironmentVariableW(L"X2W_D3D12_PERMISSIVE", nullptr);

    if (SUCCEEDED(desktop_hr)) {
        // A runtime with shader model 6.6 keeps the flags; the shim must pass them through unchanged.
        ComPtr<ID3DBlob> shim_indexed;
        const HRESULT shim_hr = D3D12SerializeVersionedRootSignature(&indexed, &shim_indexed, nullptr);
        const bool same = SUCCEEDED(shim_hr) && same_blob(shim_indexed.Get(), desktop_indexed.Get()) &&
                          stripped_bits(shim_indexed->GetBufferPointer(), shim_indexed->GetBufferSize()) == 0;
        std::printf("desktop accepts heap-indexing flags (hr=0x%08lx); pass-through %s; retry not reachable: skip\n",
                    static_cast<unsigned long>(desktop_hr), same ? "identical" : "DIFFERENT");
        return same ? 77 : 1;
    }

    // Negative control: without the switch the shim answers like the desktop runtime.
    ComPtr<ID3DBlob> refused;
    const HRESULT off_hr = D3D12SerializeVersionedRootSignature(&indexed, &refused, nullptr);
    if (desktop_hr != E_INVALIDARG || off_hr != E_INVALIDARG || refused) {
        std::fprintf(stderr, "switch off: desktop=0x%08lx shim=0x%08lx\n", static_cast<unsigned long>(desktop_hr),
                     static_cast<unsigned long>(off_hr));
        return 1;
    }

    SetEnvironmentVariableW(L"X2W_D3D12_PERMISSIVE", L"1");
    int failures = 0;
    // The retry: the blob equals the desktop serialization without the two bits, the caller's
    // descriptor is untouched, and the removed bits are recorded for that blob (and only for it).
    for (const std::uint32_t extra : {0u, 0x03000000u}) {
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC tagged = indexed;
        const auto tagged_flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(static_cast<std::uint32_t>(indexed_flags) | extra);
        tagged.Desc_1_1.Flags = tagged_flags;
        ComPtr<ID3DBlob> retried;
        const HRESULT hr = D3D12SerializeVersionedRootSignature(&tagged, &retried, nullptr);
        const std::uint32_t recorded = retried ? stripped_bits(retried->GetBufferPointer(), retried->GetBufferSize()) : ~0u;
        const bool ok = SUCCEEDED(hr) && same_blob(retried.Get(), baseline.Get()) && tagged.Desc_1_1.Flags == tagged_flags &&
                        recorded == heap_indexing;
        std::printf("retry extra=0x%08x: hr=0x%08lx same_as_desktop_without_bits=%d recorded=0x%x\n", extra,
                    static_cast<unsigned long>(hr), retried && same_blob(retried.Get(), baseline.Get()) ? 1 : 0, recorded);
        if (!ok) ++failures;
    }
    // A blob with other content records nothing, nor does a null blob.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC other = base;
    other.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> other_blob;
    if (FAILED(D3D12SerializeVersionedRootSignature(&other, &other_blob, nullptr)) || !other_blob ||
        stripped_bits(other_blob->GetBufferPointer(), other_blob->GetBufferSize()) != 0 || stripped_bits(nullptr, 0) != 0) {
        std::fprintf(stderr, "stripped bits recorded for a blob without a retry\n");
        ++failures;
    }
    if (!provenance_checks(base, stripped_bits)) ++failures;
    // The retry does not hide another fault: unobserved invalid bits next to 0xC00 stay refused.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC bad = indexed;
    bad.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(static_cast<std::uint32_t>(indexed_flags) | 0x04000000u);
    ComPtr<ID3DBlob> bad_blob;
    const HRESULT bad_hr = D3D12SerializeVersionedRootSignature(&bad, &bad_blob, nullptr);
    std::printf("invalid bits with heap-indexing: hr=0x%08lx\n", static_cast<unsigned long>(bad_hr));
    if (bad_hr != E_INVALIDARG || bad_blob) ++failures;
    SetEnvironmentVariableW(L"X2W_D3D12_PERMISSIVE", nullptr);
    return failures ? 1 : 0;
}

bool deserialize_checks()
{
    constexpr auto desktop_flag = D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS;
    D3D12_ROOT_PARAMETER1 param = make_constants_param1();
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC input{};
    input.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    input.Desc_1_1.NumParameters = 1;
    input.Desc_1_1.pParameters = &param;
    input.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(
        static_cast<std::uint32_t>(desktop_flag) | observed_console_flags);

    ComPtr<ID3DBlob> serialized;
    if (FAILED(D3D12SerializeVersionedRootSignature(&input, &serialized, nullptr)) || !serialized) {
        return false;
    }

    ComPtr<ID3D12RootSignatureDeserializer> reader;
    if (FAILED(D3D12CreateRootSignatureDeserializer(
            serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&reader))) || !reader) {
        return false;
    }
    const auto* roundtrip = reader->GetRootSignatureDesc();
    if (!roundtrip || roundtrip->Flags != desktop_flag || roundtrip->NumParameters != 1 ||
        roundtrip->pParameters[0].ParameterType != D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS ||
        roundtrip->pParameters[0].Constants.Num32BitValues != 4) {
        return false;
    }

    // Embedded DXBC container path (matching amd_fidelityfx_upscaler_xbox.dll at 0x4f88).
    ComPtr<ID3D12RootSignatureDeserializer> embedded_reader;
    if (FAILED(D3D12CreateRootSignatureDeserializer(
            pso_compute_embedded, sizeof(pso_compute_embedded), IID_PPV_ARGS(&embedded_reader))) || !embedded_reader) {
        return false;
    }
    const auto* embedded_desc = embedded_reader->GetRootSignatureDesc();
    if (!embedded_desc || embedded_desc->NumParameters != 3 || embedded_desc->NumStaticSamplers != 0) {
        return false;
    }

    // Negative controls: container with stripped RTS0 and empty/null input must fail.
    ComPtr<ID3D12RootSignatureDeserializer> stripped_reader, null_reader;
    const HRESULT stripped_hr = D3D12CreateRootSignatureDeserializer(
        pso_compute_stripped, sizeof(pso_compute_stripped), IID_PPV_ARGS(&stripped_reader));
    const HRESULT null_hr = D3D12CreateRootSignatureDeserializer(
        nullptr, 0, IID_PPV_ARGS(&null_reader));
    std::printf("embedded_params=%u stripped_hr=0x%08lx null_hr=0x%08lx\n",
                embedded_desc->NumParameters, static_cast<unsigned long>(stripped_hr),
                static_cast<unsigned long>(null_hr));
    return stripped_hr == E_INVALIDARG && !stripped_reader && null_hr == E_INVALIDARG && !null_reader;
}

bool iid_checks()
{
    HMODULE shim = GetModuleHandleW(L"d3d12_xs.dll");
    if (!shim) return false;

    struct IidEntry { const char* name; GUID expected; };
    const IidEntry entries[] = {
        {"IID_ID3D12CommandAllocator",          {0x6102DEE4,0xAF59,0x4B09,{0xB9,0x99,0xB4,0x4D,0x73,0xF0,0x9B,0x24}}},
        {"IID_ID3D12CommandList",               {0x7116D91C,0xE7E4,0x47CE,{0xB8,0xC6,0xEC,0x81,0x68,0xF4,0x37,0xE5}}},
        {"IID_ID3D12CommandQueue",              {0x0EC870A6,0x5D7E,0x4C22,{0x8C,0xFC,0x5B,0xAA,0xE0,0x76,0x16,0xED}}},
        {"IID_ID3D12Debug",                     {0x344488B7,0x6846,0x474B,{0xB9,0x89,0xF0,0x27,0x44,0x82,0x45,0xE0}}},
        {"IID_ID3D12DescriptorHeap",            {0x8EFB471D,0x616C,0x4F49,{0x90,0xF7,0x12,0x7B,0xB7,0x63,0xFA,0x51}}},
        {"IID_ID3D12Device",                    {0x189819F1,0x1DB6,0x4B57,{0xBE,0x54,0x18,0x21,0x33,0x9B,0x85,0xF7}}},
        {"IID_ID3D12DeviceChild",               {0x905DB94B,0xA00C,0x4140,{0x9D,0xF5,0x2B,0x64,0xCA,0x9E,0xA3,0x57}}},
        {"IID_ID3D12Fence",                     {0x0A753DCF,0xC4D8,0x4B91,{0xAD,0xF6,0xBE,0x5A,0x60,0xD9,0x5A,0x76}}},
        {"IID_ID3D12GraphicsCommandList",       {0x5B160D0F,0xAC1B,0x4185,{0x8B,0xA8,0xB3,0xAE,0x42,0xA5,0xA4,0x55}}},
        {"IID_ID3D12Heap",                      {0x6B3B2502,0x6E51,0x45B3,{0x90,0xEE,0x98,0x84,0x26,0x5E,0x8D,0xF3}}},
        {"IID_ID3D12Object",                    {0xC4FEC28F,0x7966,0x4E95,{0x9F,0x94,0xF4,0x31,0xCB,0x56,0xC3,0xB8}}},
        {"IID_ID3D12Pageable",                  {0x63EE58FB,0x1268,0x4835,{0x86,0xDA,0xF0,0x08,0xCE,0x62,0xF0,0xD6}}},
        {"IID_ID3D12PipelineState",             {0x765A30F3,0xF624,0x4C6F,{0xA8,0x28,0xAC,0xE9,0x48,0x62,0x24,0x45}}},
        {"IID_ID3D12Resource",                  {0x696442BE,0xA72E,0x4059,{0xBC,0x79,0x5B,0x5C,0x98,0x04,0x0F,0xAD}}},
        {"IID_ID3D12RootSignature",             {0xC54A6B66,0x72DF,0x4EE8,{0x8B,0xE5,0xA9,0x46,0xA1,0x42,0x92,0x14}}},
        {"IID_ID3D12RootSignatureDeserializer", {0x34AB647B,0x3CC8,0x46AC,{0x84,0x1B,0xC0,0x96,0x56,0x45,0xC0,0x46}}},
        {"IID_IDXGIAdapter",                    {0x2411E7E1,0x12AC,0x4CCF,{0xBD,0x14,0x97,0x98,0xE8,0x53,0x4D,0xC0}}},
        {"IID_IDXGIAdapter1",                   {0x29038F61,0x3839,0x4626,{0x91,0xFD,0x08,0x68,0x79,0x01,0x1A,0x05}}},
        {"IID_IDXGIAdapter2",                   {0x0AA1AE0A,0xFA0E,0x4B84,{0x86,0x44,0xE0,0x5F,0xF8,0xE5,0xAC,0xB5}}},
        {"IID_IDXGIDevice",                     {0x54EC77FA,0x1377,0x44E6,{0x8C,0x32,0x88,0xFD,0x5F,0x44,0xC8,0x4C}}},
        {"IID_IDXGIDevice1",                    {0x77DB970F,0x6276,0x48BA,{0xBA,0x28,0x07,0x01,0x43,0xB4,0x39,0x2C}}},
        {"IID_IDXGIDevice2",                    {0x05008617,0xFBFD,0x4051,{0xA7,0x90,0x14,0x48,0x84,0xB4,0xF6,0xA9}}},
        {"IID_IDXGIDeviceSubObject",            {0x3D3E0379,0xF9DE,0x4D58,{0xBB,0x6C,0x18,0xD6,0x29,0x92,0xF1,0xA6}}},
        {"IID_IDXGIFactory",                    {0x7B7166EC,0x21C7,0x44AE,{0xB2,0x1A,0xC9,0xAE,0x32,0x1A,0xE3,0x69}}},
        {"IID_IDXGIFactory1",                   {0x770AAE78,0xF26F,0x4DBA,{0xA8,0x29,0x25,0x3C,0x83,0xD1,0xB3,0x87}}},
        {"IID_IDXGIFactory2",                   {0x50C83A1C,0xE072,0x4C48,{0x87,0xB0,0x36,0x30,0xFA,0x36,0xA6,0xD0}}},
        {"IID_IDXGIObject",                     {0xAEC22FB8,0x76F3,0x4639,{0x9B,0xE0,0x28,0xEB,0x43,0xA6,0x7A,0x2E}}},
        {"IID_IDXGISwapChain",                  {0x310D36A0,0xD2E7,0x4C0A,{0xAA,0x04,0x6A,0x9D,0x23,0xB8,0x88,0x6A}}},
        {"IID_IDXGISwapChain1",                 {0x790A45F7,0x0D42,0x4876,{0x98,0x3A,0x0A,0x55,0xCF,0xE6,0xF4,0xAA}}},
        {"WKPDID_D3DDebugObjectNameW",          {0x4CCA5FD8,0x921F,0x42C8,{0x85,0x66,0x70,0xCA,0xF2,0xA9,0xB7,0x41}}},
    };

    unsigned passed = 0;
    unsigned failed = 0;
    for (const auto& entry : entries) {
        auto* exported = reinterpret_cast<const GUID*>(GetProcAddress(shim, entry.name));
        if (!exported) {
            std::fprintf(stderr, "MISSING: %s\n", entry.name);
            ++failed;
            continue;
        }
        if (std::memcmp(exported, &entry.expected, sizeof(GUID)) != 0) {
            std::fprintf(stderr, "MISMATCH: %s\n", entry.name);
            ++failed;
            continue;
        }
        ++passed;
    }

    // Negative control: a name that should not exist.
    if (GetProcAddress(shim, "IID_ID3D12XboxFakeInterface")) {
        std::fprintf(stderr, "unexpected export IID_ID3D12XboxFakeInterface\n");
        return false;
    }

    std::printf("iid_exports: %u/%u passed, %u failed\n", passed,
                static_cast<unsigned>(sizeof(entries) / sizeof(entries[0])), failed);
    return failed == 0;
}

bool warp_root_checks()
{
    DesktopD3D12 desktop;
    if (!desktop.load()) return false;

    // Layout matching tests/compute.hlsl (pso_compute_stripped): SRV(t0), UAV(u0), RootConstants(16, b0),
    // tagged with the observed Xbox root-signature flags 0x03000000.
    std::array<D3D12_ROOT_PARAMETER1, 3> params{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[0].Descriptor = {0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].Descriptor = {0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].Constants = {0, 0, 16};

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = static_cast<UINT>(params.size());
    desc.Desc_1_1.pParameters = params.data();
    desc.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(observed_console_flags);

    ComPtr<ID3DBlob> rejected, serialized;
    if (desktop.serialize(&desc, &rejected, nullptr) != E_INVALIDARG || rejected) {
        return false;
    }
    if (FAILED(D3D12SerializeVersionedRootSignature(&desc, &serialized, nullptr)) || !serialized) {
        return false;
    }

    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> warp;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) ||
        FAILED(desktop.create_device(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        return false;
    }

    ComPtr<ID3D12RootSignature> root;
    const HRESULT root_hr = device->CreateRootSignature(
        0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&root));
    if (FAILED(root_hr) || !root) {
        std::fprintf(stderr, "CreateRootSignature failed: 0x%08lx\n", static_cast<unsigned long>(root_hr));
        return false;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.Get();
    pso_desc.CS = {pso_compute_stripped, sizeof(pso_compute_stripped)};
    ComPtr<ID3D12PipelineState> pipeline;
    const HRESULT pso_hr = device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pipeline));
    std::printf("warp_root_hr=0x%08lx warp_pso_hr=0x%08lx\n",
                static_cast<unsigned long>(root_hr), static_cast<unsigned long>(pso_hr));
    return SUCCEEDED(pso_hr) && pipeline;
}
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        return 2;
    }
    const std::string_view scenario = argv[1];
    bool passed = false;
    if (scenario == "exports"sv) {
        passed = export_checks();
    } else if (scenario == "serialize"sv) {
        passed = serialize_checks();
    } else if (scenario == "deserialize"sv) {
        passed = deserialize_checks();
    } else if (scenario == "iids"sv) {
        passed = iid_checks();
    } else if (scenario == "warp_root"sv) {
        passed = warp_root_checks();
    } else if (scenario == "heap_indexing"sv) {
        const int result = heap_indexing_checks();
        std::printf("Desktop d3d12_xs.dll heap_indexing: %s\n", result == 0 ? "PASS" : result == 77 ? "SKIP" : "FAIL");
        return result;
    } else {
        return 2;
    }
    std::printf("Desktop d3d12_xs.dll %s: %s\n", argv[1], passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
