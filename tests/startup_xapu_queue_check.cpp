// Lifecycle, queue retirement, and refusal checks for the xapu.dll startup stand-in.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

using ConnectFn = long(__cdecl*)(const void* input, void* output, void** handle);
using GetResultCountFn = std::uint32_t(__cdecl*)(void* handle, std::uint64_t unused);
using DequeueResultFn = long(__cdecl*)(void* handle, void* result);
using EnqueueCommandFn = long(__cdecl*)(void* handle, const void* command, std::uint64_t r8, std::uint64_t r9);
using DisconnectFn = long(__cdecl*)(void* handle);

constexpr long hr_ok = 0;
constexpr long hr_pointer = static_cast<long>(0x80004003);
constexpr long hr_handle = static_cast<long>(0x80070006);
constexpr long hr_out_of_memory = static_cast<long>(0x8007000E);
constexpr long hr_invalid_arg = static_cast<long>(0x80070057);
constexpr long hr_pending_results = static_cast<long>(0x8AC81001);
constexpr DWORD trap_connect_exit = 0xE2570300u;
constexpr std::size_t result_bytes = 0x58;
constexpr std::size_t max_queue_entries = 16384;

int failures = 0;

void check(bool condition, const char* label)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL: %s\n", label);
}

bool all_zero(const std::array<unsigned char, result_bytes>& bytes)
{
    for (unsigned char b : bytes) {
        if (b != 0) return false;
    }
    return true;
}

void write_cmd(std::array<unsigned char, 0x60>& cmd, std::uint32_t type, std::uint32_t slot,
               std::uint32_t capacity_bytes = 0)
{
    cmd.fill(0);
    std::memcpy(cmd.data(), &type, sizeof(type));
    std::memcpy(cmd.data() + 4, &slot, sizeof(slot));
    std::memcpy(cmd.data() + 0x34, &capacity_bytes, sizeof(capacity_bytes));
}

std::uint32_t read_u32(const std::array<unsigned char, result_bytes>& result, std::size_t offset)
{
    std::uint32_t value = 0;
    std::memcpy(&value, result.data() + offset, sizeof(value));
    return value;
}

bool rest_zero(const std::array<unsigned char, result_bytes>& bytes, std::size_t from)
{
    for (std::size_t i = from; i < bytes.size(); ++i) {
        if (bytes[i] != 0) return false;
    }
    return true;
}

// Mirrors the caller-side per-slot state transitions driven by XApuDequeueResult (#1887):
// state 1 (opening) --(res 1/3)--> 4 (ready) --(res 3)--> 5 (decoded, then 4)
// state 2 (stopping) --(res 5)--> 4; res 2 --> 0 (closed).
void advance_slot_state(const std::array<unsigned char, result_bytes>& result, std::uint32_t& state,
                        std::uint32_t& bytes_out)
{
    const std::uint32_t type = read_u32(result, 0x00);
    if (state == 3 && type != 2) return;
    if (type == 3) {
        if (state == 4) {
            bytes_out = read_u32(result, 0x34);
            state = 5;
        } else if (state == 1) {
            state = 4;
        }
    } else if (type == 1) {
        if (state == 1) state = 4;
    } else if (type == 2) {
        state = 0;
    } else if (type == 5 && state == 2) {
        state = 4;
    }
}

int run_trap_control(const char* dll_path)
{
    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", nullptr);
    HMODULE module = LoadLibraryA(dll_path);
    if (!module) return 2;
    auto connect = reinterpret_cast<ConnectFn>(GetProcAddress(module, "XApuConnect"));
    if (!connect) return 3;
    std::array<unsigned char, 0x18> input{};
    std::array<unsigned char, 0x10> output{};
    void* handle = nullptr;
    connect(input.data(), output.data(), &handle);
    return 4;
}

void check_non_permissive_trap(const char* self_exe, const char* dll_path)
{
    std::string cmd = std::string("\"") + self_exe + "\" --trap-control \"" + dll_path + "\"";
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        check(false, "CreateProcessA for non-permissive trap control");
        return;
    }
    const DWORD wait_rc = WaitForSingleObject(pi.hProcess, 10000);
    DWORD exit_code = 0;
    const BOOL got_code = GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    check(wait_rc == WAIT_OBJECT_0, "non-permissive trap child exited within timeout");
    check(got_code && exit_code == trap_connect_exit, "non-permissive XApuConnect traps with 0xE2570300");
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--trap-control") == 0) return run_trap_control(argv[2]);
    if (argc != 2) return std::printf("usage: startup_xapu_queue_check <xapu.dll>\n"), 2;
    const char* dll_path = argv[1];

    check_non_permissive_trap(argv[0], dll_path);

    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_XAPU_TYPED_COMPLETIONS", nullptr);
    HMODULE module = LoadLibraryA(dll_path);
    if (!module) return std::printf("FAIL: cannot load %s\n", dll_path), 2;

    auto connect = reinterpret_cast<ConnectFn>(GetProcAddress(module, "XApuConnect"));
    auto get_count = reinterpret_cast<GetResultCountFn>(GetProcAddress(module, "XApuGetResultCount"));
    auto dequeue = reinterpret_cast<DequeueResultFn>(GetProcAddress(module, "XApuDequeueResult"));
    auto enqueue = reinterpret_cast<EnqueueCommandFn>(GetProcAddress(module, "XApuEnqueueCommand"));
    auto disconnect = reinterpret_cast<DisconnectFn>(GetProcAddress(module, "XApuDisconnect"));
    if (!connect || !get_count || !dequeue || !enqueue || !disconnect) {
        FreeLibrary(module);
        return std::printf("FAIL: missing export\n"), 2;
    }

    // XApuConnect input as the observed caller passes it: the shared buffer size is the u32 at +0x0C.
    constexpr std::uint32_t buffer_bytes = 0x104000;
    const auto sized_input = [](std::uint32_t bytes) {
        std::array<unsigned char, 0x18> in{};
        std::memcpy(in.data() + 0x0C, &bytes, sizeof(bytes));
        return in;
    };
    const auto input = sized_input(buffer_bytes);
    std::array<unsigned char, 0x10> output{};
    std::array<unsigned char, 0x60> cmd{};
    std::array<unsigned char, result_bytes> result{};
    void* bogus = reinterpret_cast<void*>(static_cast<std::uintptr_t>(0xDEADBEEFULL));

    // Invalid-handle and null-pointer negative controls before any connection exists.
    check(connect(input.data(), output.data(), nullptr) == hr_pointer, "XApuConnect(null handle*) -> E_POINTER");
    void* refused = nullptr;
    check(connect(nullptr, output.data(), &refused) == hr_pointer && !refused, "XApuConnect(null input) -> E_POINTER");
    check(connect(input.data(), nullptr, &refused) == hr_pointer && !refused, "XApuConnect(null output) -> E_POINTER");
    for (std::uint32_t bad_size : {0u, (64u << 20) + 1}) {
        const auto bad_input = sized_input(bad_size);
        output.fill(0x5A);
        check(connect(bad_input.data(), output.data(), &refused) == hr_invalid_arg && !refused,
              "XApuConnect(buffer size 0 or over 64 MiB) -> E_INVALIDARG without a handle");
        bool untouched = true;
        for (unsigned char b : output) untouched &= b == 0x5A;
        check(untouched, "refused XApuConnect leaves its output alone");
    }
    check(get_count(nullptr, 0) == 0, "XApuGetResultCount(null) -> 0");
    check(get_count(bogus, 0) == 0, "XApuGetResultCount(bogus) -> 0");
    check(enqueue(nullptr, cmd.data(), 0, 0) == hr_handle, "XApuEnqueueCommand(null) -> E_HANDLE");
    check(enqueue(bogus, cmd.data(), 0, 0) == hr_handle, "XApuEnqueueCommand(bogus) -> E_HANDLE");
    check(enqueue(nullptr, nullptr, 0, 0) == hr_pointer, "XApuEnqueueCommand(null cmd*) -> E_POINTER");
    check(dequeue(nullptr, result.data()) == hr_handle, "XApuDequeueResult(null handle) -> E_HANDLE");
    check(dequeue(bogus, result.data()) == hr_handle, "XApuDequeueResult(bogus handle) -> E_HANDLE");
    check(disconnect(nullptr) == hr_handle, "XApuDisconnect(null) -> E_HANDLE");
    check(disconnect(bogus) == hr_handle, "XApuDisconnect(bogus) -> E_HANDLE");

    // Positive lifecycle: connect, enqueue, count, and dequeue, including empty queues and pointer checks.
    std::array<unsigned char, result_bytes> sentinel;
    sentinel.fill(0xA5);
    void* h1 = nullptr;
    output.fill(0x5A);
    std::memset(output.data(), 0, sizeof(void*));  // an unwritten pointer reads as null, not as the 0x5A fill
    check(connect(input.data(), output.data(), &h1) == hr_ok && h1 != nullptr, "XApuConnect h1 succeeds");
    // The caller reads only output +0 and copies packets and samples into that buffer.
    unsigned char* shared = nullptr;
    std::memcpy(&shared, output.data(), sizeof(shared));
    bool second_untouched = true;
    for (std::size_t i = 8; i < output.size(); ++i) second_untouched &= output[i] == 0x5A;
    check(shared != nullptr && second_untouched, "XApuConnect writes the shared buffer to output +0 and leaves +8 alone");
    if (shared) {
        bool zeroed = true;
        for (std::uint32_t i = 0; i < buffer_bytes; ++i) zeroed &= shared[i] == 0;
        check(zeroed, "shared buffer starts zeroed");
        shared[0] = 1;
        shared[buffer_bytes - 1] = 1;
    }
    check(get_count(h1, 0) == 0, "initial XApuGetResultCount == 0");
    result = sentinel;
    check(dequeue(h1, result.data()) == hr_pending_results, "empty queue XApuDequeueResult -> XAPU_E_PENDING_RESULTS");
    check(result == sentinel, "empty dequeue preserves the result buffer");
    check(enqueue(h1, nullptr, 0, 0) == hr_pointer, "XApuEnqueueCommand(h1, null cmd*) -> E_POINTER");
    if (void* pages = VirtualAlloc(nullptr, 0x2000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)) {
        DWORD old_prot = 0;
        if (VirtualProtect(static_cast<unsigned char*>(pages) + 0x1000, 0x1000, PAGE_NOACCESS, &old_prot)) {
            auto* straddling = static_cast<unsigned char*>(pages) + 0x1000 - 4;
            check(enqueue(h1, straddling, 0, 0) == hr_pointer,
                  "XApuEnqueueCommand(straddling RW->NOACCESS cmd*) -> E_POINTER");
        } else {
            check(false, "VirtualProtect PAGE_NOACCESS for command negative control");
        }
        VirtualFree(pages, 0, MEM_RELEASE);
    } else {
        check(false, "VirtualAlloc for command negative control");
    }
    check(get_count(h1, 0) == 0, "rejected command headers do not enqueue entries");

    write_cmd(cmd, 1, 7);
    check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "default: enqueue an open command with nonzero type and slot");
    check(get_count(h1, 0) == 1, "default: open command enters the queue");
    result = sentinel;
    check(dequeue(h1, result.data()) == hr_ok && all_zero(result),
          "default: retire the open command without publishing a typed completion");
    check(get_count(h1, 0) == 0, "default: zeroed result still consumes the queued command");
    result = sentinel;
    check(dequeue(h1, result.data()) == hr_pending_results && result == sentinel,
          "default: empty dequeue leaves its result untouched");

    check(SetEnvironmentVariableA("X2W_XAPU_TYPED_COMPLETIONS", "1") != FALSE, "enable experimental typed completions");
    // Zero command yields all-zero 0x58-byte result; typed commands echo (+0x00 type, +0x04 slot) in FIFO order.
    write_cmd(cmd, 0, 0);
    check(enqueue(h1, cmd.data(), 0x11, 0x22) == hr_ok, "enqueue zero command on h1");
    write_cmd(cmd, 1, 7);
    check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "enqueue open (type=1, slot=7) on h1");
    write_cmd(cmd, 3, 7, 0xF00);
    check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "enqueue packet (type=3, slot=7) on h1");
    write_cmd(cmd, 5, 7);
    check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "enqueue stop (type=5, slot=7) on h1");
    write_cmd(cmd, 2, 7);
    check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "enqueue close (type=2, slot=7) on h1");
    check(get_count(h1, 0) == 5, "XApuGetResultCount == 5 after 5 enqueues");

    check(dequeue(h1, nullptr) == hr_pointer, "XApuDequeueResult(null result*) -> E_POINTER");
    if (void* ro = VirtualAlloc(nullptr, 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY)) {
        check(dequeue(h1, ro) == hr_pointer, "XApuDequeueResult(PAGE_READONLY result*) -> E_POINTER");
        VirtualFree(ro, 0, MEM_RELEASE);
    } else {
        check(false, "VirtualAlloc PAGE_READONLY for result negative control");
    }
    check(get_count(h1, 0) == 5, "failed dequeues do not consume queued entries");

    result.fill(0xAA);
    check(dequeue(h1, result.data()) == hr_ok, "dequeue #1 (zero cmd) on h1 succeeds");
    check(all_zero(result), "dequeue #1 zeroes 0x58-byte result buffer");
    check(get_count(h1, 0) == 4, "XApuGetResultCount == 4 after first dequeue");

    std::uint32_t slot_state = 1;
    std::uint32_t decoded_bytes = 0xDEADBEEFu;
    struct ExpectedStep {
        std::uint32_t type;
        std::uint32_t slot;
        std::uint32_t next_state;
        const char* label;
    };
    const ExpectedStep steps[] = {
        {1, 7, 4, "dequeue open result advances slot state 1 -> 4"},
        {3, 7, 5, "dequeue packet result advances slot state 4 -> 5 with 0 decoded bytes"},
        {5, 7, 4, "dequeue stop result restores stopping slot state 2 -> 4"},
        {2, 7, 0, "dequeue close result frees slot state -> 0"},
    };
    for (std::size_t i = 0; i < 4; ++i) {
        if (i == 2) slot_state = 2;  // simulate caller setting stopping state before dequeuing stop result
        if (i == 3) slot_state = 3;  // simulate caller setting closing state before dequeuing close result
        result.fill(0x55);
        check(dequeue(h1, result.data()) == hr_ok, steps[i].label);
        check(read_u32(result, 0x00) == steps[i].type && read_u32(result, 0x04) == steps[i].slot,
              steps[i].label);
        check(rest_zero(result, 8), "dequeue clears remaining result bytes (+0x08..+0x57, including +0x34)");
        advance_slot_state(result, slot_state, decoded_bytes);
        check(slot_state == steps[i].next_state, steps[i].label);
    }
    check(decoded_bytes == 0, "type 3 completion writes 0 decoded bytes at +0x34");
    write_cmd(cmd, 4, 9);
    check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "enqueue unobserved type=4 echoes verbatim");
    result.fill(0x55);
    check(dequeue(h1, result.data()) == hr_ok && read_u32(result, 0x00) == 4 && read_u32(result, 0x04) == 9 &&
              rest_zero(result, 8),
          "dequeue echoes type=4 and slot=9 verbatim");
    check(get_count(h1, 0) == 0, "XApuGetResultCount == 0 after draining typed commands");
    result = sentinel;
    check(dequeue(h1, result.data()) == hr_pending_results, "drained queue XApuDequeueResult -> XAPU_E_PENDING_RESULTS");
    check(result == sentinel, "dequeue after draining preserves the result buffer");

    // Burst past initial 64-entry ring capacity with wraparound.
    for (std::uint32_t i = 0; i < 40; ++i) {
        write_cmd(cmd, (i % 3) + 1, 100 + i);
        check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "ring prefill enqueue succeeds");
    }
    for (std::uint32_t i = 0; i < 20; ++i) {
        check(dequeue(h1, result.data()) == hr_ok, "ring wrap partial dequeue succeeds");
        check(read_u32(result, 0x00) == (i % 3) + 1 && read_u32(result, 0x04) == 100 + i,
              "ring wrap partial dequeue matches FIFO");
    }
    for (std::uint32_t i = 40; i < 120; ++i) {
        write_cmd(cmd, (i % 4) + 1, 100 + i);
        check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "ring growth enqueue succeeds");
    }
    check(get_count(h1, 0) == 100, "ring growth reports 100 queued entries");
    for (std::uint32_t i = 20; i < 120; ++i) {
        const std::uint32_t expected_type = i < 40 ? (i % 3) + 1 : (i % 4) + 1;
        check(dequeue(h1, result.data()) == hr_ok, "ring growth dequeue succeeds");
        check(read_u32(result, 0x00) == expected_type && read_u32(result, 0x04) == 100 + i && rest_zero(result, 8),
              "ring growth preserves FIFO (type, slot) order");
    }
    check(get_count(h1, 0) == 0, "ring drain returns count to 0");

    // Queue cap negative control: filling to max_queue_entries refuses further enqueues with E_OUTOFMEMORY.
    for (std::uint32_t i = 0; i < max_queue_entries; ++i) {
        write_cmd(cmd, 3, i);
        if (enqueue(h1, cmd.data(), 0, 0) != hr_ok) {
            check(false, "enqueue up to max_queue_entries succeeds");
            break;
        }
    }
    check(get_count(h1, 0) == max_queue_entries, "queue count reaches max_queue_entries");
    write_cmd(cmd, 5, 0xFFFF);
    check(enqueue(h1, cmd.data(), 0, 0) == hr_out_of_memory, "enqueue past max_queue_entries -> E_OUTOFMEMORY");
    check(get_count(h1, 0) == max_queue_entries, "rejected over-cap enqueue keeps count at max_queue_entries");
    check(dequeue(h1, result.data()) == hr_ok && read_u32(result, 0x00) == 3 && read_u32(result, 0x04) == 0,
          "head entry remains intact after over-cap refusal");
    write_cmd(cmd, 5, 0xFFFF);
    check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "enqueue succeeds again after draining one slot");
    for (std::size_t i = 0; i < max_queue_entries; ++i) {
        if (dequeue(h1, result.data()) != hr_ok) {
            check(false, "draining capped queue succeeds");
            break;
        }
    }
    check(get_count(h1, 0) == 0, "capped queue drains back to 0");

    // Second independent handle + disconnect invalidation.
    void* h2 = nullptr;
    check(connect(input.data(), output.data(), &h2) == hr_ok && h2 != nullptr && h2 != h1, "XApuConnect h2 distinct");
    unsigned char* shared2 = nullptr;
    std::memcpy(&shared2, output.data(), sizeof(shared2));
    check(shared2 != nullptr && shared2 != shared, "each connection gets its own shared buffer");
    write_cmd(cmd, 1, 11);
    check(enqueue(h1, cmd.data(), 0, 0) == hr_ok, "enqueue on h1 before disconnect");
    write_cmd(cmd, 3, 21);
    check(enqueue(h2, cmd.data(), 0, 0) == hr_ok, "enqueue #1 on h2");
    write_cmd(cmd, 5, 22);
    check(enqueue(h2, cmd.data(), 0, 0) == hr_ok, "enqueue #2 on h2");

    check(disconnect(h1) == hr_ok, "XApuDisconnect(h1) succeeds");
    check(disconnect(h1) == hr_handle, "repeat XApuDisconnect(h1) -> E_HANDLE");
    check(enqueue(h1, cmd.data(), 0, 0) == hr_handle, "enqueue after disconnect(h1) -> E_HANDLE");
    check(get_count(h1, 0) == 0, "get_count after disconnect(h1) -> 0");
    check(dequeue(h1, result.data()) == hr_handle, "dequeue after disconnect(h1) -> E_HANDLE");

    check(get_count(h2, 0) == 2, "disconnect(h1) preserves h2 queue");
    check(dequeue(h2, result.data()) == hr_ok && read_u32(result, 0x00) == 3 && read_u32(result, 0x04) == 21,
          "h2 dequeue #1 preserves (3, 21)");
    check(dequeue(h2, result.data()) == hr_ok && read_u32(result, 0x00) == 5 && read_u32(result, 0x04) == 22,
          "h2 dequeue #2 preserves (5, 22)");
    check(dequeue(h2, result.data()) == hr_pending_results, "h2 drained -> XAPU_E_PENDING_RESULTS");
    check(disconnect(h2) == hr_ok, "XApuDisconnect(h2) succeeds");

    FreeLibrary(module);
    std::printf(failures ? "startup_xapu_queue_check: %d failure(s)\n" : "startup_xapu_queue_check: ok\n", failures);
    return failures ? 1 : 0;
}
