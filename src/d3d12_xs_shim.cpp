// Desktop d3d12_xs.dll exporting the two root-signature functions the package DLLs import by name.
// Observed call sites (docs/experiments.md):
//   D3D12SerializeVersionedRootSignature: XboxBvhBuilder.dll (0xc2d0/0xc32f) copies a 48-byte
//     descriptor, sets console flag bits 0x03000000 at +0x28, and passes a null error output.
//   D3D12CreateRootSignatureDeserializer: amd_fidelityfx_upscaler_xbox.dll (0x4f88) reads a
//     serialized root signature from an embedded container and retrieves the descriptor interface.
// The main executable also imports D3D12XboxCreateDevice and D3D12ConfigureVirtualMemoryX (#874).
// Device creation consults the source-owned creation-IID inventory in d3d12x_device_slots.h.
#include <d3d12.h>
#include <dxgi.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include "d3d12x_device_slots.h"
#include "startup_stubs/d3d12_teardown.h"

namespace {
// Observed console root-signature flag bits set by XboxBvhBuilder.dll.
// Unobserved bits are left untouched so desktop d3d12.dll still rejects unknown flags.
constexpr std::uint32_t observed_console_flags = 0x03000000u;

static_assert(offsetof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC, Desc_1_0) +
              offsetof(D3D12_ROOT_SIGNATURE_DESC, Flags) == 0x28);
static_assert(offsetof(D3D12_VERSIONED_ROOT_SIGNATURE_DESC, Desc_1_1) +
              offsetof(D3D12_ROOT_SIGNATURE_DESC1, Flags) == 0x28);

// The observed caller zero-fills a 32-byte block, sets only the DWORD at +4 to 3 or 4, and
// configures once per process before it creates the device. The DWORD's meaning is open.
constexpr std::size_t vm_config_bytes = 32;
constexpr std::size_t vm_config_value = 4;
// Configuration and device creation are serialized. `value` is the accepted DWORD, 0 for none. It is
// recorded for the owner of the title's device, which will reserve the matching GPU range
// (src/d3d12_xs_range.h); until that owner exists, device creation does not use it.
struct VmState {
    SRWLOCK lock = SRWLOCK_INIT;
    std::uint32_t value = 0;
    bool device_created = false;
};
VmState vm_state;

// Desktop root-signature flags for direct heap indexing (CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED and
// SAMPLER_HEAP_DIRECTLY_INDEXED). A desktop runtime older than shader model 6.6 does not know them
// and answers E_INVALIDARG; the in-box Windows 10 runtime is one. With X2W_D3D12_PERMISSIVE=1 the
// serialization is then repeated without them, so that a caller that sets them on every root
// signature still gets a blob. The result no longer declares direct heap indexing.
constexpr std::uint32_t heap_indexing_flags = 0x00000C00u;

// The serialized bytes do not say which input produced them: the legal baseline and inputs with only
// 0x400, only 0x800 or both serialize to the same blob once the bits are removed. The removed bits are
// therefore tied to the blob object a retry returns. That object registers its buffer while it lives
// and unregisters it before the buffer can be freed and reused, so x2w_stripped_root_flags answers
// for that exact buffer only. A copy of the bytes, or a blob from any other call, reports 0.
struct StrippedBlob { const void* data; std::size_t bytes; std::uint32_t bits; };
SRWLOCK stripped_lock = SRWLOCK_INIT;
std::vector<StrippedBlob> stripped_blobs;

class RetriedBlob final : public ID3DBlob {
public:
    // Take ownership only after the wrapper and its provenance record have both been allocated.
    static ID3DBlob* wrap(ID3DBlob* inner, std::uint32_t bits) noexcept
    {
        RetriedBlob* wrapper = new (std::nothrow) RetriedBlob(inner);
        if (!wrapper) return nullptr;
        bool registered = true;
        AcquireSRWLockExclusive(&stripped_lock);
        try {
            stripped_blobs.push_back({inner->GetBufferPointer(), inner->GetBufferSize(), bits});
        } catch (...) {
            registered = false;
        }
        ReleaseSRWLockExclusive(&stripped_lock);
        if (registered) return wrapper;
        wrapper->inner_ = nullptr;  // inner stays with the caller
        delete wrapper;
        return nullptr;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override
    {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (iid != __uuidof(IUnknown) && iid != __uuidof(ID3D10Blob)) return E_NOINTERFACE;
        *out = static_cast<ID3DBlob*>(this);
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(InterlockedIncrement(&refs_)); }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const LONG left = InterlockedDecrement(&refs_);
        if (!left) delete this;
        return static_cast<ULONG>(left);
    }
    LPVOID STDMETHODCALLTYPE GetBufferPointer() override { return inner_->GetBufferPointer(); }
    SIZE_T STDMETHODCALLTYPE GetBufferSize() override { return inner_->GetBufferSize(); }

private:
    explicit RetriedBlob(ID3DBlob* inner) : inner_(inner) {}
    ~RetriedBlob()
    {
        if (!inner_) return;
        const void* data = inner_->GetBufferPointer();
        AcquireSRWLockExclusive(&stripped_lock);
        for (std::size_t i = 0; i < stripped_blobs.size(); ++i)
            if (stripped_blobs[i].data == data) {
                stripped_blobs[i] = stripped_blobs.back();
                stripped_blobs.pop_back();
                break;
            }
        ReleaseSRWLockExclusive(&stripped_lock);
        inner_->Release();  // only now may the buffer be reused
    }
    ID3DBlob* inner_;
    LONG refs_ = 1;
};

bool permissive_switch()
{
    const DWORD saved = GetLastError();
    char value[4] = {};
    const bool on = GetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", value, sizeof(value)) > 0 && value[0] == '1';
    SetLastError(saved);
    return on;
}

HRESULT serialize_for_runtime(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc, ID3DBlob** blob, ID3DBlob** error_blob)
{
    const HRESULT hr = D3D12SerializeVersionedRootSignature(desc, blob, error_blob);
    if (hr != E_INVALIDARG || !desc || !blob) return hr;
    if (desc->Version != D3D_ROOT_SIGNATURE_VERSION_1_0 && desc->Version != D3D_ROOT_SIGNATURE_VERSION_1_1) return hr;
    const bool version_1_0 = desc->Version == D3D_ROOT_SIGNATURE_VERSION_1_0;
    const auto flags = static_cast<std::uint32_t>(version_1_0 ? desc->Desc_1_0.Flags : desc->Desc_1_1.Flags);
    if (!(flags & heap_indexing_flags) || !permissive_switch()) return hr;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC copy = *desc;
    auto& retry_flags = version_1_0 ? copy.Desc_1_0.Flags : copy.Desc_1_1.Flags;
    retry_flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags & ~heap_indexing_flags);
    ID3DBlob* retry_blob = nullptr;
    ID3DBlob* retry_error = nullptr;
    const HRESULT retry = D3D12SerializeVersionedRootSignature(&copy, &retry_blob, &retry_error);
    if (retry_error) retry_error->Release();
    if (FAILED(retry) || !retry_blob) {
        if (retry_blob) retry_blob->Release();
        return hr;  // something else is wrong: keep the first answer and its error blob
    }
    *blob = RetriedBlob::wrap(retry_blob, flags & heap_indexing_flags);
    if (error_blob && *error_blob) {
        (*error_blob)->Release();
        *error_blob = nullptr;
    }
    if (!*blob) {
        retry_blob->Release();
        return E_OUTOFMEMORY;
    }
    static LONG logged;
    if (InterlockedIncrement(&logged) == 1)
        std::fprintf(stderr, "x2w-d3d12 root signature flags 0x%x: this desktop runtime refuses the heap-indexing bits 0x%x; "
                             "serialized without them (X2W_D3D12_PERMISSIVE)\n", flags, flags & heap_indexing_flags);
    return retry;
}
}

// The heap-indexing bits serialize_for_runtime removed for the live blob whose buffer is exactly
// (blob, bytes), 0 for any other pointer or size.
extern "C" std::uint32_t x2w_stripped_root_flags(const void* blob, std::size_t bytes)
{
    if (!blob || !bytes) return 0;
    std::uint32_t bits = 0;
    bool same_bytes_elsewhere = false;
    // The byte comparison below only feeds a once-only line; it stops once that line is logged.
    static LONG logged;
    AcquireSRWLockShared(&stripped_lock);
    for (const StrippedBlob& entry : stripped_blobs)
        if (entry.data == blob && entry.bytes == bytes) bits = entry.bits;
    // Live entries' buffers stay valid while the lock is held (the destructor unregisters first).
    if (!bits && !InterlockedCompareExchange(&logged, 0, 0))
        for (const StrippedBlob& entry : stripped_blobs)
            if (entry.bytes == bytes && std::memcmp(entry.data, blob, bytes) == 0) {
                same_bytes_elsewhere = true;
                break;
            }
    ReleaseSRWLockShared(&stripped_lock);
    // The bytes alone cannot say which it is, so the answer stays 0; the line says why bits may be missing.
    if (same_bytes_elsewhere && InterlockedExchange(&logged, 1) == 0)
        std::fprintf(stderr, "x2w-d3d12 root signature blob %p has the bytes of a retried blob at another address: a copy "
                             "(its heap-indexing bits are not restored for translation) or a call without them\n", blob);
    return bits;
}

extern "C" HRESULT WINAPI d3d12_xs_SerializeVersionedRootSignature(
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* root_signature,
    ID3DBlob** blob,
    ID3DBlob** error_blob)
{
    if (!root_signature) {
        return serialize_for_runtime(root_signature, blob, error_blob);
    }
    if (root_signature->Version == D3D_ROOT_SIGNATURE_VERSION_1_1) {
        const auto flags = static_cast<std::uint32_t>(root_signature->Desc_1_1.Flags);
        if (flags & observed_console_flags) {
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC copy = *root_signature;
            copy.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags & ~observed_console_flags);
            return serialize_for_runtime(&copy, blob, error_blob);
        }
    } else if (root_signature->Version == D3D_ROOT_SIGNATURE_VERSION_1_0) {
        const auto flags = static_cast<std::uint32_t>(root_signature->Desc_1_0.Flags);
        if (flags & observed_console_flags) {
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC copy = *root_signature;
            copy.Desc_1_0.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags & ~observed_console_flags);
            return serialize_for_runtime(&copy, blob, error_blob);
        }
    }
    return serialize_for_runtime(root_signature, blob, error_blob);
}

extern "C" HRESULT WINAPI d3d12_xs_CreateRootSignatureDeserializer(
    LPCVOID src_data,
    SIZE_T src_data_size_in_bytes,
    REFIID deserializer_interface,
    void** deserializer)
{
    return D3D12CreateRootSignatureDeserializer(
        src_data, src_data_size_in_bytes, deserializer_interface, deserializer);
}

bool x2w_d3d12_permissive();
HRESULT x2w_d3d12_permissive_create_device(IUnknown* adapter, const void* parameters, REFIID iid, void** device);

extern "C" HRESULT WINAPI d3d12_xs_XboxCreateDevice(
    IUnknown* adapter, const void* parameters, REFIID iid, void** device)
{
    if (!device) return E_POINTER;
    *device = nullptr;
    // This entry supports an explicit desktop consumer of the returned interface.
    // The Xbox parameter block and the Xbox device vtable require their own implementation.
    // Experimental: opt-in recording wrapper, src/startup_stubs/d3d12_xs_device.cpp.
    if (parameters && x2w_d3d12_permissive())
        return x2w_d3d12_permissive_create_device(adapter, parameters, iid, device);
    if (parameters) return E_INVALIDARG;
    if (!d3d12x_device_slots::accepts_creation_iid(iid)) {
        const auto text = d3d12x_device_slots::format_iid(iid);
        std::fprintf(stderr, "D3D12XboxCreateDevice: unsupported IID %s\n", text.data());
        return E_NOINTERFACE;
    }
    AcquireSRWLockExclusive(&vm_state.lock);
    const HRESULT hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, iid, device);
    if (SUCCEEDED(hr)) vm_state.device_created = true;
    ReleaseSRWLockExclusive(&vm_state.lock);
    return hr;
}

extern "C" HRESULT WINAPI d3d12_xs_ConfigureVirtualMemoryX(const void* config)
{
    // Accepts only the observed shape, once, before this shim has created a device. A refusal
    // changes no state; the caller's block is only read.
    if (!config) return E_POINTER;
    unsigned char bytes[vm_config_bytes];
    std::memcpy(bytes, config, sizeof(bytes));
    std::uint32_t value = 0;
    std::memcpy(&value, bytes + vm_config_value, sizeof(value));
    if (value != 3 && value != 4) return E_INVALIDARG;
    for (std::size_t i = 0; i < sizeof(bytes); ++i) {
        if (bytes[i] && (i < vm_config_value || i >= vm_config_value + sizeof(value))) return E_INVALIDARG;
    }
    AcquireSRWLockExclusive(&vm_state.lock);
    const bool accepted = !vm_state.value && !vm_state.device_created;
    if (accepted) vm_state.value = value;
    ReleaseSRWLockExclusive(&vm_state.lock);
    return accepted ? S_OK : DXGI_ERROR_INVALID_CALL;
}

// Lifetime boundary of the desktop D3D12 objects this module keeps in function-local statics
// (startup_stubs/d3d12_draw.inl, startup_stubs/rect_indirect_args.h, #2572). Those caches own a device,
// a root signature and a pipeline state per device for as long as this DLL is loaded, and the loader
// detaches the module before their destructors run:
//   lpReserved != nullptr  the process is exiting. Windows has already stopped the display driver, so a
//                          Release() there faults inside the driver (amdxc64 offset 0x35f2a19, through
//                          D3D12Core!CPipelineState::FinalRelease) and the loader ends the process with
//                          __fastfail(7). The caches detach their references instead of releasing them.
//   lpReserved == nullptr  FreeLibrary unloads the module while the process and the driver stay alive:
//                          the caches release the objects they own.
// This DllMain runs before this module's atexit table: the static runtime reaches the user entry before
// it runs the table on detach, which the cpu_native_crt_lifecycle fixture measures (a global destructor
// runs after DllMain(DLL_PROCESS_DETACH), docs/experiments.md), so the destructors see the flag below.
// startup_stubs/d3d12_teardown.h has the details.
namespace x2w_startup { void report_render_process_exit(); }

extern "C" BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID reserved)
{
    if (reason == DLL_PROCESS_DETACH && reserved) {
        x2w_startup::d3d12_process_terminating = true;
        x2w_startup::report_render_process_exit();
    }
    return TRUE;
}
