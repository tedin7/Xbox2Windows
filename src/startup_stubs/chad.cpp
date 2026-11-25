// Desktop chad.dll trap stubs.
// Every export traps; see trap.h. Scaffolding, not an ABI claim.
#define X2W_TRAP_MODULE "chad.dll"
#define X2W_TRAP_MODULE_ID 2
#include "startup_stubs/trap.h"
#include "startup_stubs/local_log.h"

// Experimental (docs/startup-runtime.md). With X2W_AUDIO_PERMISSIVE=1 the exports
// stand in for the DSP stream interface; otherwise they trap. Seen:
//   XDspConnectWithMaximumStreamLimit(queue memory, 0x60000, 30, 0x80, 1, &handle) -> 0 on success,
//   E_POINTER when handle is null or unwritable.
// No stream is processed: activation and submission fail with the documented XDSP code instead of
// returning success with their outputs untouched, and deactivation and disconnection refuse null and
// unknown handles (issue #1887).
namespace {
using u64 = std::uint64_t;
constexpr long hr_ok = 0;
constexpr long hr_pointer = static_cast<long>(0x80004003);  // E_POINTER
constexpr long hr_invalidarg = static_cast<long>(0x80070057);  // E_INVALIDARG
// Microsoft GDK error codes, XDSP section: XDSP_E_NO_DEVICE_RESOURCES.
constexpr long hr_no_device_resources = static_cast<long>(0x8AC810A2);
bool permissive() noexcept { return x2w_local::env_on("X2W_AUDIO_PERMISSIVE"); }
int connection;

long accept(const char* name, unsigned index, u64 rcx, u64 rdx, u64 r8, u64 r9, void* return_address, LONG& calls)
{
    if (!permissive()) x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, index, rcx, rdx, r8, r9, return_address);
    if (x2w_local::first_calls(calls, 6))
        x2w_trap::record("x2w-audio", X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, index, rcx, rdx, r8, r9, return_address);
    return hr_ok;
}

// XDspDeactivate(stream) and XDspDisconnect(client) take a handle and write nothing back. As in
// XDspSubmitCommand, only the handle XDspConnectWithMaximumStreamLimit published is known: a null or
// unknown handle is refused instead of being reported as released.
long release_handle(const char* name, unsigned index, u64 rcx, u64 rdx, u64 r8, u64 r9, void* return_address,
                    LONG& calls)
{
    const long result = accept(name, index, rcx, rdx, r8, r9, return_address, calls);
    if (reinterpret_cast<void*>(static_cast<std::uintptr_t>(rcx)) != &connection) {
        x2w_local::logf("x2w-audio %s handle=0x%llx -> E_INVALIDARG (unknown handle)", name,
                        static_cast<unsigned long long>(rcx));
        return hr_invalidarg;
    }
    x2w_local::logf("x2w-audio %s handle=0x%llx -> 0", name, static_cast<unsigned long long>(rcx));
    return result;
}
}  // namespace

extern "C" __declspec(noinline) long XDspActivate(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    static LONG calls;
    accept("XDspActivate", 0, rcx, rdx, r8, r9, _ReturnAddress(), calls);
    // XDspActivate(client, params, XDspStatus** status, XDspStreamHandle* stream). No XDSP device exists
    // under this stand-in, so the call fails the way the documented code says instead of returning success
    // with both outputs untouched: validate them, clear them, then report the failure.
    auto** status_out = reinterpret_cast<void**>(static_cast<std::uintptr_t>(r8));
    auto** stream_out = reinterpret_cast<void**>(static_cast<std::uintptr_t>(r9));
    if (!status_out || !x2w_local::writable(status_out, sizeof(void*)) ||
        !stream_out || !x2w_local::writable(stream_out, sizeof(void*))) {
        x2w_local::logf("x2w-audio XDspActivate status_out=%p stream_out=%p -> E_POINTER",
                        static_cast<void*>(status_out), static_cast<void*>(stream_out));
        return hr_pointer;
    }
    *status_out = nullptr;
    *stream_out = nullptr;
    x2w_local::logf("x2w-audio XDspActivate status_out=%p stream_out=%p -> 0x8AC810A2 "
                    "(XDSP_E_NO_DEVICE_RESOURCES)",
                    static_cast<void*>(status_out), static_cast<void*>(stream_out));
    return hr_no_device_resources;
}

extern "C" __declspec(noinline) long XDspConnectWithMaximumStreamLimit(u64 rcx, u64 rdx, u64 r8, u64 r9, u64 flag, void** handle)
{
    static LONG calls;
    const long result = accept("XDspConnectWithMaximumStreamLimit", 1, rcx, rdx, r8, r9, _ReturnAddress(), calls);
    if (!handle || !x2w_local::writable(handle, sizeof(void*))) {
        x2w_local::logf("x2w-audio XDspConnectWithMaximumStreamLimit memory=0x%llx bytes=0x%llx streams=%llu r9=0x%llx flag=%llu "
                        "handle_out=%p -> E_POINTER",
                        static_cast<unsigned long long>(rcx), static_cast<unsigned long long>(rdx),
                        static_cast<unsigned long long>(r8), static_cast<unsigned long long>(r9),
                        static_cast<unsigned long long>(flag), static_cast<void*>(handle));
        return hr_pointer;
    }
    *handle = &connection;
    x2w_local::logf("x2w-audio XDspConnectWithMaximumStreamLimit memory=0x%llx bytes=0x%llx streams=%llu r9=0x%llx flag=%llu "
                    "handle_out=%p -> 0",
                    static_cast<unsigned long long>(rcx), static_cast<unsigned long long>(rdx),
                    static_cast<unsigned long long>(r8), static_cast<unsigned long long>(r9),
                    static_cast<unsigned long long>(flag), static_cast<void*>(handle));
    return result;
}

extern "C" __declspec(noinline) long XDspDeactivate(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    static LONG calls;
    return release_handle("XDspDeactivate", 2, rcx, rdx, r8, r9, _ReturnAddress(), calls);
}

extern "C" __declspec(noinline) long XDspDisconnect(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    static LONG calls;
    return release_handle("XDspDisconnect", 3, rcx, rdx, r8, r9, _ReturnAddress(), calls);
}

extern "C" __declspec(noinline) long XDspSubmitCommand(u64 rcx, u64 rdx, u64 r8, u64 r9)
{
    static LONG calls;
    accept("XDspSubmitCommand", 4, rcx, rdx, r8, r9, _ReturnAddress(), calls);
    // XDspSubmitCommand(stream, command, uint32_t* sequence). Only the handle XDspConnectWithMaximumStreamLimit
    // published is known; anything else is refused without writing the sequence output, and a known handle
    // still has no device to run the command, so the caller value survives every failure path.
    auto* sequence_out = reinterpret_cast<unsigned*>(static_cast<std::uintptr_t>(r8));
    if (reinterpret_cast<void*>(static_cast<std::uintptr_t>(rcx)) != &connection) {
        x2w_local::logf("x2w-audio XDspSubmitCommand stream=0x%llx sequence_out=%p -> E_INVALIDARG "
                        "(unknown stream handle)",
                        static_cast<unsigned long long>(rcx), static_cast<void*>(sequence_out));
        return hr_invalidarg;
    }
    if (!sequence_out || !x2w_local::writable(sequence_out, sizeof(*sequence_out))) {
        x2w_local::logf("x2w-audio XDspSubmitCommand stream=0x%llx sequence_out=%p -> E_POINTER",
                        static_cast<unsigned long long>(rcx), static_cast<void*>(sequence_out));
        return hr_pointer;
    }
    x2w_local::logf("x2w-audio XDspSubmitCommand stream=0x%llx sequence_out=%p -> 0x8AC810A2 "
                    "(XDSP_E_NO_DEVICE_RESOURCES, sequence preserved)",
                    static_cast<unsigned long long>(rcx), static_cast<void*>(sequence_out));
    return hr_no_device_resources;
}
