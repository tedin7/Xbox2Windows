// Source-built checks for the optional XApu Opus telemetry: decoded packets, dequeued decoded
// completions, refusals by reason, and the stable summary line. Synthetic tones are encoded here
// with libopus; no game data or captured packet is used.
#include "startup_stubs/audio_telemetry.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <opus.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

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
constexpr long hr_fail = static_cast<long>(0x80004005);
constexpr std::size_t result_size = 0x58;
constexpr std::size_t max_queue_entries = 16384;
constexpr unsigned char fill = 0xCD;

int failures = 0;
ConnectFn connect;
GetResultCountFn get_count;
DequeueResultFn dequeue;
EnqueueCommandFn enqueue;
DisconnectFn disconnect;
void* handle;

void check(bool condition, const char* label)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL: %s\n", label);
}

template <typename T>
void put(std::array<unsigned char, 0x60>& cmd, std::size_t offset, T value)
{
    std::memcpy(cmd.data() + offset, &value, sizeof(value));
}

// Type 3 command with the fields one title fills.
std::array<unsigned char, 0x60> decode_cmd(std::uint32_t slot, float rate, std::uint32_t frames, const void* packet,
                                           std::uint32_t packet_bytes, void* out, std::uint32_t out_bytes)
{
    std::array<unsigned char, 0x60> cmd{};
    put(cmd, 0x00, std::uint32_t{3});
    put(cmd, 0x04, slot);
    put(cmd, 0x1C, rate);
    put(cmd, 0x24, frames);
    put(cmd, 0x28, packet_bytes);
    put(cmd, 0x2C, reinterpret_cast<std::uint64_t>(packet));
    put(cmd, 0x34, out_bytes);
    put(cmd, 0x38, reinterpret_cast<std::uint64_t>(out));
    return cmd;
}

std::array<unsigned char, 0x60> plain_cmd(std::uint32_t type, std::uint32_t slot)
{
    std::array<unsigned char, 0x60> cmd{};
    put(cmd, 0x00, type);
    put(cmd, 0x04, slot);
    return cmd;
}

std::vector<unsigned char> tone_packet(int rate, int frames)
{
    int error = OPUS_OK;
    OpusEncoder* encoder = opus_encoder_create(rate, 1, OPUS_APPLICATION_AUDIO, &error);
    if (!encoder || error != OPUS_OK) return {};
    std::vector<float> pcm(static_cast<std::size_t>(frames));
    for (int i = 0; i < frames; ++i)
        pcm[static_cast<std::size_t>(i)] =
            0.5f * static_cast<float>(std::sin(6.283185307179586 * 1000.0 * i / rate));
    std::vector<unsigned char> packet(4000);
    const int n = opus_encode_float(encoder, pcm.data(), frames, packet.data(), static_cast<opus_int32>(packet.size()));
    opus_encoder_destroy(encoder);
    if (n <= 0) return {};
    packet.resize(static_cast<std::size_t>(n));
    return packet;
}

std::vector<float> filled(std::size_t floats)
{
    std::vector<float> buffer(floats);
    std::memset(buffer.data(), fill, floats * sizeof(float));
    return buffer;
}

bool untouched(const std::vector<float>& buffer, std::size_t from = 0)
{
    const auto* bytes = reinterpret_cast<const unsigned char*>(buffer.data());
    for (std::size_t i = from * sizeof(float); i < buffer.size() * sizeof(float); ++i)
        if (bytes[i] != fill) return false;
    return true;
}

struct Result {
    std::uint32_t type = 0xFFFFFFFFu, slot = 0xFFFFFFFFu, bytes = 0xFFFFFFFFu;
};

Result pop()
{
    std::array<unsigned char, result_size> raw{};
    Result result;
    if (dequeue(handle, raw.data()) != hr_ok) return result;
    std::memcpy(&result.type, raw.data(), 4);
    std::memcpy(&result.slot, raw.data() + 4, 4);
    std::memcpy(&result.bytes, raw.data() + 0x34, 4);
    return result;
}

void expect_refused(const char* label, long want, const std::array<unsigned char, 0x60>& cmd,
                    const std::vector<float>& out)
{
    check(enqueue(handle, cmd.data(), 0, 0) == want, label);
    check(get_count(handle, 0) == 0, label);
    check(untouched(out), label);
}

std::string read_log(const char* path)
{
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return {};
    std::string text;
    char buffer[4096];
    for (;;) {
        DWORD read = 0;
        if (!ReadFile(file, buffer, sizeof(buffer), &read, nullptr) || read == 0) break;
        text.append(buffer, read);
    }
    CloseHandle(file);
    return text;
}

struct Counts {
    unsigned long long type3 = 0, packets = 0, completions = 0, refusals = 0;
    unsigned long long invalid_argument = 0, opus_error = 0, capacity = 0, queue_full = 0;
    unsigned long long short_command = 0, unknown_handle = 0, out_of_memory = 0, header = 0;
};

// Returns false on a malformed summary; counts every summary line in `lines`.
bool parse_summary(const std::string& text, Counts& counts, int& lines)
{
    counts = Counts{};
    lines = 0;
    const char prefix[] = "x2w-audio XApu opus summary:";
    for (std::size_t start = 0; start < text.size();) {
        const std::size_t end = text.find('\n', start);
        const std::string line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (line.rfind(prefix, 0) == 0) {
            Counts parsed;
            const int fields = std::sscanf(
                line.c_str(),
                "x2w-audio XApu opus summary: type3=%llu packets=%llu completions=%llu refusals=%llu "
                "invalid_argument=%llu opus_error=%llu capacity=%llu queue_full=%llu short_command=%llu "
                "unknown_handle=%llu out_of_memory=%llu header=%llu",
                &parsed.type3, &parsed.packets, &parsed.completions, &parsed.refusals, &parsed.invalid_argument,
                &parsed.opus_error, &parsed.capacity, &parsed.queue_full, &parsed.short_command,
                &parsed.unknown_handle, &parsed.out_of_memory, &parsed.header);
            if (fields != 12) return false;
            counts = parsed;
            ++lines;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return true;
}

void check_counts(const Counts& counts, const char* label)
{
    check(counts.type3 == 10, label);
    check(counts.packets == 2, label);
    check(counts.completions == 2, label);
    check(counts.refusals == 9, label);
    check(counts.invalid_argument == 3, label);
    check(counts.opus_error == 1, label);
    check(counts.capacity == 1, label);
    check(counts.queue_full == 1, label);
    check(counts.short_command == 1, label);
    check(counts.unknown_handle == 1, label);
    check(counts.out_of_memory == 0, label);
    check(counts.header == 1, label);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return std::printf("usage: startup_xapu_opus_count_check <xapu.dll>\n"), 2;

    // The saturation contract is independent of the endpoint: prove the helper stops at UINT64_MAX.
    std::atomic<std::uint64_t> bounded{(std::numeric_limits<std::uint64_t>::max)() - 1};
    x2w_audio::saturating_add(bounded, 2);
    check(bounded == (std::numeric_limits<std::uint64_t>::max)(), "counter saturates at UINT64_MAX");
    x2w_audio::saturating_add(bounded);
    check(bounded == (std::numeric_limits<std::uint64_t>::max)(), "a saturated counter stays at UINT64_MAX");

    char temp_path[MAX_PATH]{};
    char log_path[MAX_PATH]{};
    if (!GetTempPathA(MAX_PATH, temp_path) || !GetTempFileNameA(temp_path, "x2w-opus-", 0, log_path))
        return std::printf("FAIL: cannot create the telemetry log path\n"), 2;
    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_XAPU_TYPED_COMPLETIONS", "1");
    SetEnvironmentVariableA("X2W_TRAP_LOG", log_path);

    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return std::printf("FAIL: cannot load %s\n", argv[1]), 2;
    connect = reinterpret_cast<ConnectFn>(GetProcAddress(module, "XApuConnect"));
    get_count = reinterpret_cast<GetResultCountFn>(GetProcAddress(module, "XApuGetResultCount"));
    dequeue = reinterpret_cast<DequeueResultFn>(GetProcAddress(module, "XApuDequeueResult"));
    enqueue = reinterpret_cast<EnqueueCommandFn>(GetProcAddress(module, "XApuEnqueueCommand"));
    disconnect = reinterpret_cast<DisconnectFn>(GetProcAddress(module, "XApuDisconnect"));
    if (!connect || !get_count || !dequeue || !enqueue || !disconnect)
        return std::printf("FAIL: missing export\n"), 2;

    std::array<unsigned char, 0x18> input{};
    const std::uint32_t buffer_bytes = 0x10000;
    std::memcpy(input.data() + 0x0C, &buffer_bytes, sizeof(buffer_bytes));
    std::array<unsigned char, 0x10> output{};
    if (connect(input.data(), output.data(), &handle) != hr_ok || !handle)
        return std::printf("FAIL: XApuConnect\n"), 2;

    const auto packet = tone_packet(48000, 960);
    if (packet.empty()) return std::printf("FAIL: libopus produced no packet\n"), 2;

    // Two successful packets, each completion retired by the caller.
    for (int i = 0; i < 2; ++i) {
        auto out = filled(964);
        const auto cmd = decode_cmd(0, 48000.0f, 960, packet.data(), static_cast<std::uint32_t>(packet.size()),
                                    out.data(), 3840);
        check(enqueue(handle, cmd.data(), 0, 0) == hr_ok, "valid packet accepted");
        const Result result = pop();
        check(result.type == 3 && result.slot == 0 && result.bytes == 3840, "valid packet completes with 3840 bytes");
        check(untouched(out, 960), "valid packet leaves guard floats untouched");
    }

    // Malformed packet: libopus refuses it.
    {
        const unsigned char bad[] = {0x03, 0x00};
        auto out = filled(960);
        expect_refused("malformed packet -> E_FAIL", hr_fail,
                       decode_cmd(1, 48000.0f, 960, bad, sizeof(bad), out.data(), 3840), out);
    }
    // Insufficient output capacity: libopus reports OPUS_BUFFER_TOO_SMALL for a 20 ms packet in 80 frames.
    {
        auto out = filled(964);
        expect_refused("insufficient capacity -> E_FAIL", hr_fail,
                       decode_cmd(1, 48000.0f, 80, packet.data(), static_cast<std::uint32_t>(packet.size()),
                                  out.data(), 320), out);
    }
    // Invalid argument before libopus.
    {
        auto out = filled(960);
        expect_refused("rate 44100 -> E_INVALIDARG", hr_invalid_arg,
                       decode_cmd(1, 44100.0f, 960, packet.data(), static_cast<std::uint32_t>(packet.size()),
                                  out.data(), 3840), out);
    }
    // An out-of-range slot is classified in the legacy decode path.
    {
        auto out = filled(960);
        expect_refused("legacy slot UINT32_MAX -> E_INVALIDARG", hr_invalid_arg,
                       decode_cmd((std::numeric_limits<std::uint32_t>::max)(), 48000.0f, 960, packet.data(),
                                  static_cast<std::uint32_t>(packet.size()), out.data(), 3840), out);
    }
    // A type 3 header whose decode fields cross a guard page.
    if (auto* pages = static_cast<unsigned char*>(VirtualAlloc(nullptr, 0x2000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE))) {
        DWORD old = 0;
        check(VirtualProtect(pages + 0x1000, 0x1000, PAGE_NOACCESS, &old) != FALSE, "guard page for short command");
        const auto full = decode_cmd(1, 48000.0f, 960, packet.data(), static_cast<std::uint32_t>(packet.size()),
                                     pages, 3840);
        std::memcpy(pages + 0x1000 - 0x20, full.data(), 0x20);
        check(enqueue(handle, pages + 0x1000 - 0x20, 0, 0) == hr_pointer, "short decode command -> E_POINTER");
        check(get_count(handle, 0) == 0, "short decode command queues nothing");
        VirtualFree(pages, 0, MEM_RELEASE);
    } else {
        check(false, "VirtualAlloc for short command");
    }
    // A null header is a header refusal, not a type 3 one.
    check(enqueue(handle, nullptr, 0, 0) == hr_pointer, "null command header -> E_POINTER");
    // A readable type 3 command on an unknown handle.
    {
        auto out = filled(960);
        const auto cmd = decode_cmd(0, 48000.0f, 960, packet.data(), static_cast<std::uint32_t>(packet.size()),
                                    out.data(), 3840);
        check(enqueue(reinterpret_cast<void*>(0xDEADBEEFULL), cmd.data(), 0, 0) == hr_handle,
              "unknown handle -> E_HANDLE");
    }
    // Queue cap: the decode is refused before it touches the slot's decoder.
    {
        bool filled_up = true;
        for (std::uint32_t i = 0; i < max_queue_entries && filled_up; ++i)
            filled_up = enqueue(handle, plain_cmd(4, 99).data(), 0, 0) == hr_ok;
        check(filled_up && get_count(handle, 0) == max_queue_entries, "queue fills to 16384 entries");
        auto out = filled(960);
        const auto cmd = decode_cmd(1, 48000.0f, 960, packet.data(), static_cast<std::uint32_t>(packet.size()),
                                    out.data(), 3840);
        check(enqueue(handle, cmd.data(), 0, 0) == hr_out_of_memory, "full queue decode -> E_OUTOFMEMORY");
        check(untouched(out), "full queue refusal leaves output untouched");
        bool drained = true;
        std::array<unsigned char, result_size> raw{};
        for (std::size_t i = 0; i < max_queue_entries && drained; ++i)
            drained = dequeue(handle, raw.data()) == hr_ok;
        check(drained && get_count(handle, 0) == 0, "capped queue drains");
    }

    // The typed-stream path rejects an out-of-range slot before decode_packet; keep this connection
    // open until the legacy connection closes so both refusals reach one summary line.
    std::array<unsigned char, 0x18> stream_input = input;
    const std::uint32_t process = 1;
    std::memcpy(stream_input.data(), &process, sizeof(process));
    void* stream_handle = nullptr;
    check(connect(stream_input.data(), output.data(), &stream_handle) == hr_ok && stream_handle,
          "stream XApuConnect");
    {
        auto out = filled(960);
        const auto cmd = decode_cmd((std::numeric_limits<std::uint32_t>::max)(), 48000.0f, 960,
                                    packet.data(), static_cast<std::uint32_t>(packet.size()), out.data(), 3840);
        check(enqueue(stream_handle, cmd.data(), 0, 0) == hr_invalid_arg,
              "stream slot UINT32_MAX -> E_INVALIDARG");
        check(get_count(stream_handle, 0) == 0, "stream slot refusal queues nothing");
        check(untouched(out), "stream slot refusal leaves output untouched");
    }
    check(disconnect(handle) == hr_ok, "XApuDisconnect succeeds");
    check(disconnect(stream_handle) == hr_ok, "stream XApuDisconnect succeeds");

    Counts counts;
    int lines = 0;
    const std::string log = read_log(log_path);
    check(parse_summary(log, counts, lines), "summary line has the documented fields");
    check(lines == 1, "one summary line after the last disconnect");
    check_counts(counts, "summary counts successes and each refusal reason");

    // Typed completions off: the same command is queued without decoding and the counters do not move.
    SetEnvironmentVariableA("X2W_XAPU_TYPED_COMPLETIONS", nullptr);
    void* second = nullptr;
    check(connect(input.data(), output.data(), &second) == hr_ok && second != nullptr, "second XApuConnect");
    auto out = filled(960);
    const auto untyped = decode_cmd(2, 48000.0f, 960, packet.data(), static_cast<std::uint32_t>(packet.size()),
                                    out.data(), 3840);
    check(enqueue(second, untyped.data(), 0, 0) == hr_ok, "untyped type 3 accepted");
    std::array<unsigned char, result_size> raw{};
    check(dequeue(second, raw.data()) == hr_ok, "untyped type 3 completion");
    check(untouched(out), "untyped type 3 leaves output untouched");
    check(disconnect(second) == hr_ok, "second XApuDisconnect succeeds");
    Counts unchanged;
    int after = 0;
    check(parse_summary(read_log(log_path), unchanged, after), "summary still parses");
    check(after == 1, "typed completions off do not add a summary line");
    check(unchanged.packets == 2 && unchanged.completions == 2, "typed completions off do not move the counters");

    FreeLibrary(module);
    int at_exit_lines = 0;
    check(parse_summary(read_log(log_path), counts, at_exit_lines), "summary parses after unload");
    check(at_exit_lines == 1, "process exit does not repeat an unchanged summary");

    DeleteFileA(log_path);
    SetEnvironmentVariableA("X2W_TRAP_LOG", nullptr);
    SetEnvironmentVariableA("X2W_XAPU_TYPED_COMPLETIONS", nullptr);
    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", nullptr);
    if (failures) return std::printf("%d failure(s)\n", failures), 1;
    std::printf("PASS: startup_xapu_opus_count (%s)\n", opus_get_version_string());
    return 0;
}
