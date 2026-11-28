// Opus decode checks for the xapu.dll stand-in built with X2W_WITH_OPUS: generated tones, exact decoded
// byte counts, bit-exact PCM against a reference libopus decoder, a malformed packet and refusals that
// must leave the output buffer and the result queue untouched.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <opus.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using ConnectFn = long(__cdecl*)(const void* input, void* output, void** handle);
using GetResultCountFn = std::uint32_t(__cdecl*)(void* handle, std::uint64_t unused);
using DequeueResultFn = long(__cdecl*)(void* handle, void* result);
using EnqueueCommandFn = long(__cdecl*)(void* handle, const void* command, std::uint64_t r8, std::uint64_t r9);
using DisconnectFn = long(__cdecl*)(void* handle);

constexpr long hr_ok = 0;
constexpr long hr_pointer = static_cast<long>(0x80004003);
constexpr long hr_fail = static_cast<long>(0x80004005);
constexpr long hr_invalid_arg = static_cast<long>(0x80070057);
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

// Packets of a 1 kHz tone (opposite phase on the right channel).
std::vector<std::vector<unsigned char>> tone_packets(int rate, int channels, int frames, int count)
{
    std::vector<std::vector<unsigned char>> packets;
    int error = OPUS_OK;
    OpusEncoder* enc = opus_encoder_create(rate, channels, OPUS_APPLICATION_AUDIO, &error);
    if (!enc || error != OPUS_OK) return packets;
    std::vector<float> pcm(static_cast<std::size_t>(frames * channels));
    for (int p = 0; p < count; ++p) {
        for (int i = 0; i < frames; ++i) {
            const float s = 0.5f * static_cast<float>(std::sin(6.283185307179586 * 1000.0 * (p * frames + i) / rate));
            for (int c = 0; c < channels; ++c) pcm[static_cast<std::size_t>(i * channels + c)] = c ? -s : s;
        }
        std::vector<unsigned char> packet(4000);
        const int n = opus_encode_float(enc, pcm.data(), frames, packet.data(), static_cast<opus_int32>(packet.size()));
        if (n <= 0) break;
        packet.resize(static_cast<std::size_t>(n));
        packets.push_back(packet);
    }
    opus_encoder_destroy(enc);
    return packets;
}

bool untouched(const std::vector<float>& buf, std::size_t from = 0)
{
    const auto* b = reinterpret_cast<const unsigned char*>(buf.data());
    for (std::size_t i = from * sizeof(float); i < buf.size() * sizeof(float); ++i) {
        if (b[i] != fill) return false;
    }
    return true;
}

std::vector<float> filled(std::size_t floats)
{
    std::vector<float> buf(floats);
    std::memset(buf.data(), fill, floats * sizeof(float));
    return buf;
}

struct Result {
    std::uint32_t type = 0xFFFFFFFFu, slot = 0xFFFFFFFFu, bytes = 0xFFFFFFFFu;
};

Result pop()
{
    std::array<unsigned char, 0x58> raw{};
    Result r;
    if (dequeue(handle, raw.data()) != hr_ok) return r;
    std::memcpy(&r.type, raw.data(), 4);
    std::memcpy(&r.slot, raw.data() + 4, 4);
    std::memcpy(&r.bytes, raw.data() + 0x34, 4);
    return r;
}

// Decodes every packet through XApu and a reference decoder; checks byte counts, PCM and the guard floats.
void decode_stream(const char* label, std::uint32_t slot, int rate, int channels, std::uint32_t frames,
                   const std::vector<std::vector<unsigned char>>& packets, std::uint32_t expect_frames,
                   OpusDecoder* ref)
{
    check(packets.size() == 3, label);
    const std::size_t floats = static_cast<std::size_t>(frames) * channels;
    for (const auto& packet : packets) {
        auto out = filled(floats + 4);
        const auto cmd = decode_cmd(slot, static_cast<float>(rate), frames, packet.data(),
                                    static_cast<std::uint32_t>(packet.size()), out.data(),
                                    static_cast<std::uint32_t>(floats * 4));
        check(enqueue(handle, cmd.data(), 0, 0) == hr_ok, label);
        const Result r = pop();
        check(r.type == 3 && r.slot == slot && r.bytes == expect_frames * channels * 4, label);
        std::vector<float> want(floats);
        const int n = opus_decode_float(ref, packet.data(), static_cast<opus_int32>(packet.size()), want.data(),
                                        static_cast<int>(frames), 0);
        check(n == static_cast<int>(expect_frames), label);
        const std::size_t got = static_cast<std::size_t>(expect_frames) * channels;
        check(std::memcmp(out.data(), want.data(), got * sizeof(float)) == 0, label);
        check(untouched(out, got), label);
    }
}

// Decodes one mono packet through XApu and through `ref` (a fresh decoder when null); true when both agree.
bool mono_matches(std::uint32_t slot, int rate, std::uint32_t frames, const std::vector<unsigned char>& packet,
                  OpusDecoder* ref)
{
    int error = OPUS_OK;
    OpusDecoder* fresh = ref ? nullptr : opus_decoder_create(rate, 1, &error);
    OpusDecoder* dec = ref ? ref : fresh;
    if (!dec) return false;
    auto out = filled(frames);
    std::vector<float> want(frames);
    const auto cmd = decode_cmd(slot, static_cast<float>(rate), frames, packet.data(),
                                static_cast<std::uint32_t>(packet.size()), out.data(), frames * 4);
    const bool ok = enqueue(handle, cmd.data(), 0, 0) == hr_ok && pop().bytes == frames * 4 &&
                    opus_decode_float(dec, packet.data(), static_cast<opus_int32>(packet.size()), want.data(),
                                      static_cast<int>(frames), 0) == static_cast<int>(frames) &&
                    std::memcmp(out.data(), want.data(), frames * sizeof(float)) == 0;
    if (fresh) opus_decoder_destroy(fresh);
    return ok;
}

void expect_refused(const char* label, long want, const std::array<unsigned char, 0x60>& cmd,
                    const std::vector<float>& out)
{
    check(enqueue(handle, cmd.data(), 0, 0) == want, label);
    check(get_count(handle, 0) == 0, label);
    check(untouched(out), label);
}

void check_activated_stereo(std::uint32_t options)
{
    const std::array<std::uint32_t, 6> connection{1, 2, 2, 0x10000, 0, options};
    std::array<void*, 2> shared{};
    void* device = nullptr;
    if (connect(connection.data(), shared.data(), &device) != hr_ok) {
        check(false, "connect typed Opus stream");
        return;
    }
    std::array<unsigned char, 0x58> result{};
    auto open = plain_cmd(1, 0);
    put(open, 24, 2u);
    check(enqueue(device, open.data(), 0, 0) == hr_ok && dequeue(device, result.data()) == hr_ok, "activate stereo");
    auto packets = tone_packets(48000, 2, 960, 1);
    int err = OPUS_OK;
    OpusDecoder* reference = opus_decoder_create(48000, 2, &err);
    if (packets.size() == 1 && reference) {
        const auto& p = packets.front();
        float want[1920]{};
        check(opus_decode_float(reference, p.data(), static_cast<opus_int32>(p.size()), want, 960, 0) == 960,
              "decode complete reference packet");
        auto out = filled(2 * 649);
        const auto capacity = 2u * 648 * sizeof(float);
        // The decoder must consume all 960 frames before trimming the 312-frame prefix.
        auto cmd = decode_cmd(0, 1.0f, 960, p.data(), static_cast<std::uint32_t>(p.size()),
                              out.data(), static_cast<std::uint32_t>(capacity));
        put(cmd, 8, 37u);
        put(cmd, 16, std::uint64_t{0x123456789abcdef0});
        put(cmd, 32, 312u);
        result.fill(0xCD);
        check(enqueue(device, cmd.data(), 0, 0) == hr_ok && dequeue(device, result.data()) == hr_ok, "decode trimmed packet");
        std::uint32_t frames{}, bytes{};
        std::memcpy(&frames, result.data() + 0x30, 4);
        std::memcpy(&bytes, result.data() + 0x34, 4);
        check(frames == 648 && bytes == capacity && std::memcmp(cmd.data(), result.data(), 24) == 0,
              "completed frames, length and command identity");
        check(std::memcmp(result.data() + 0x38, cmd.data() + 0x38, 8) == 0 &&
              std::memcmp(result.data() + 0x44, cmd.data() + 0x28, 12) == 0, "completed packet and output spans");
        check(result[0x50] == 0xCD && result.back() == 0xCD, "next packed result is untouched");
        bool matches = true;
        for (unsigned i = 0; i < 648; ++i) {
            for (unsigned c = 0; c < 2; ++c) {
                const auto dest = (options & 4) ? c * 648 + i : 2 * i + c;
                matches &= out[dest] == want[2 * (312 + i) + c];
            }
        }
        check(matches && untouched(out, 1296), "trim and channel order match reference, tail preserved");
        auto close = plain_cmd(2, 0);
        check(enqueue(device, close.data(), 0, 0) == hr_ok && dequeue(device, result.data()) == hr_ok, "close stereo");
        check(enqueue(device, cmd.data(), 0, 0) == hr_invalid_arg && get_count(device, 0) == 0,
              "a closed stream cannot decode another packet");
    } else {
        check(false, "construct stereo reference");
    }
    if (reference) opus_decoder_destroy(reference);
    disconnect(device);
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return std::printf("usage: startup_xapu_opus_check <xapu.dll>\n"), 2;
    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_XAPU_TYPED_COMPLETIONS", "1");
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return std::printf("FAIL: cannot load %s\n", argv[1]), 2;
    connect = reinterpret_cast<ConnectFn>(GetProcAddress(module, "XApuConnect"));
    get_count = reinterpret_cast<GetResultCountFn>(GetProcAddress(module, "XApuGetResultCount"));
    dequeue = reinterpret_cast<DequeueResultFn>(GetProcAddress(module, "XApuDequeueResult"));
    enqueue = reinterpret_cast<EnqueueCommandFn>(GetProcAddress(module, "XApuEnqueueCommand"));
    disconnect = reinterpret_cast<DisconnectFn>(GetProcAddress(module, "XApuDisconnect"));
    if (!connect || !get_count || !dequeue || !enqueue || !disconnect) return std::printf("FAIL: missing export\n"), 2;
    check_activated_stereo(1);
    check_activated_stereo(5);

    std::array<unsigned char, 0x18> input{};
    const std::uint32_t buffer_bytes = 0x104000;
    std::memcpy(input.data() + 0x0C, &buffer_bytes, sizeof(buffer_bytes));
    std::array<unsigned char, 0x10> output{};
    if (connect(input.data(), output.data(), &handle) != hr_ok || !handle) return std::printf("FAIL: XApuConnect\n"), 2;

    int error = OPUS_OK;
    OpusDecoder* ref_mono = opus_decoder_create(48000, 1, &error);
    OpusDecoder* ref_stereo = opus_decoder_create(48000, 2, &error);
    OpusDecoder* ref_10ms = opus_decoder_create(48000, 1, &error);
    OpusDecoder* ref_16k = opus_decoder_create(16000, 1, &error);
    if (!ref_mono || !ref_stereo || !ref_10ms || !ref_16k) return std::printf("FAIL: reference decoders\n"), 2;

    // 20 ms mono tone at 48 kHz: 960 frames * 1 channel * 4 bytes.
    const auto mono = tone_packets(48000, 1, 960, 3);
    decode_stream("48 kHz mono 20 ms packets decode to 3840 bytes, bit-exact", 7, 48000, 1, 960, mono, 960, ref_mono);
    // Full queue: decode, reset and close are refused before they touch the slot's decoder.
    {
        bool filled_up = true;
        for (std::uint32_t i = 0; i < 16384 && filled_up; ++i) filled_up = enqueue(handle, plain_cmd(4, 99).data(), 0, 0) == hr_ok;
        check(filled_up && get_count(handle, 0) == 16384, "queue fills to 16384 entries");
        auto out = filled(960);
        const auto cmd = decode_cmd(7, 48000.0f, 960, mono[0].data(), static_cast<std::uint32_t>(mono[0].size()), out.data(), 3840);
        check(enqueue(handle, cmd.data(), 0, 0) == static_cast<long>(0x8007000E) && untouched(out),
              "full queue: decode -> E_OUTOFMEMORY, output untouched");
        check(enqueue(handle, plain_cmd(5, 7).data(), 0, 0) == static_cast<long>(0x8007000E) &&
                  enqueue(handle, plain_cmd(2, 7).data(), 0, 0) == static_cast<long>(0x8007000E),
              "full queue: reset and close -> E_OUTOFMEMORY");
        bool drained = true;
        while (drained && get_count(handle, 0)) drained = pop().type == 4;
        check(drained && get_count(handle, 0) == 0, "full queue drains");
        check(mono_matches(7, 48000, 960, mono[0], ref_mono), "after the full-queue refusals slot 7 continues its stream");
    }
    {
        auto out = filled(960);
        const auto cmd = decode_cmd(7, 48000.0f, 960, mono[2].data(), static_cast<std::uint32_t>(mono[2].size()),
                                    out.data(), 3840);
        check(enqueue(handle, cmd.data(), 0, 0) == hr_ok && pop().bytes == 3840, "mono tone decodes again");
        double energy = 0;
        for (float s : out) energy += static_cast<double>(s) * s;
        check(std::sqrt(energy / out.size()) > 0.1, "decoded tone is not silence (RMS > 0.1)");
        std::vector<float> sync(960);  // keep the reference decoder on slot 7's stream
        opus_decode_float(ref_mono, mono[2].data(), static_cast<opus_int32>(mono[2].size()), sync.data(), 960, 0);
        check(std::memcmp(out.data(), sync.data(), sizeof(float) * 960) == 0, "mono tone decodes again, bit-exact");
    }
    decode_stream("48 kHz stereo 20 ms packets decode to 7680 bytes, bit-exact", 8, 48000, 2, 960,
                  tone_packets(48000, 2, 960, 3), 960, ref_stereo);
    // 10 ms packet in a 960-frame buffer: the result carries the decoded count, not the capacity.
    const auto short_packets = tone_packets(48000, 1, 480, 4);
    decode_stream("10 ms packets in a 20 ms buffer report 1920 bytes", 9, 48000, 1, 960,
                  {short_packets.begin(), short_packets.begin() + 3}, 480, ref_10ms);
    decode_stream("16 kHz mono 20 ms packets decode to 1280 bytes, bit-exact", 10, 16000, 1, 320,
                  tone_packets(16000, 1, 320, 3), 320, ref_16k);

    // Malformed packet: code 3 with a frame count of 0. libopus refuses it; nothing is queued or written.
    {
        const unsigned char bad[] = {0x03, 0x00};
        auto out = filled(960);
        expect_refused("malformed packet -> E_FAIL, output and queue untouched", hr_fail,
                       decode_cmd(11, 48000.0f, 960, bad, sizeof(bad), out.data(), 3840), out);
    }

    // Refusals before decoding.
    {
        const auto& p = mono[0];
        const auto pb = static_cast<std::uint32_t>(p.size());
        auto out = filled(5762);
        expect_refused("rate 44100 -> E_INVALIDARG", hr_invalid_arg, decode_cmd(12, 44100.0f, 960, p.data(), pb, out.data(), 3840), out);
        expect_refused("output bytes not frames * channels * 4 -> E_INVALIDARG", hr_invalid_arg,
                       decode_cmd(12, 48000.0f, 960, p.data(), pb, out.data(), 3842), out);
        expect_refused("3 channels -> E_INVALIDARG", hr_invalid_arg, decode_cmd(12, 48000.0f, 960, p.data(), pb, out.data(), 960 * 12), out);
        expect_refused("0 frames -> E_INVALIDARG", hr_invalid_arg, decode_cmd(12, 48000.0f, 0, p.data(), pb, out.data(), 0), out);
        expect_refused("5761 frames -> E_INVALIDARG", hr_invalid_arg, decode_cmd(12, 48000.0f, 5761, p.data(), pb, out.data(), 5761 * 4), out);
        expect_refused("null packet with bytes -> E_INVALIDARG", hr_invalid_arg, decode_cmd(12, 48000.0f, 960, nullptr, pb, out.data(), 3840), out);
        expect_refused("packet with 0 bytes -> E_INVALIDARG", hr_invalid_arg, decode_cmd(12, 48000.0f, 960, p.data(), 0, out.data(), 3840), out);
        expect_refused("packet over 64 KiB -> E_INVALIDARG", hr_invalid_arg, decode_cmd(12, 48000.0f, 960, p.data(), 0x10001, out.data(), 3840), out);
        expect_refused("slot 256 -> E_INVALIDARG", hr_invalid_arg, decode_cmd(256, 48000.0f, 960, p.data(), pb, out.data(), 3840), out);
        expect_refused("null output -> E_INVALIDARG", hr_invalid_arg, decode_cmd(12, 48000.0f, 960, p.data(), pb, nullptr, 3840), out);
        if (void* ro = VirtualAlloc(nullptr, 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY)) {
            expect_refused("read-only output -> E_INVALIDARG", hr_invalid_arg, decode_cmd(12, 48000.0f, 960, p.data(), pb, ro, 3840), out);
            VirtualFree(ro, 0, MEM_RELEASE);
        } else {
            check(false, "VirtualAlloc PAGE_READONLY output");
        }
        // A command whose decode fields lie past readable memory.
        if (auto* pages = static_cast<unsigned char*>(VirtualAlloc(nullptr, 0x2000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE))) {
            DWORD old = 0;
            VirtualProtect(pages + 0x1000, 0x1000, PAGE_NOACCESS, &old);
            const auto full = decode_cmd(12, 48000.0f, 960, p.data(), pb, out.data(), 3840);
            std::memcpy(pages + 0x1000 - 0x20, full.data(), 0x20);
            check(enqueue(handle, pages + 0x1000 - 0x20, 0, 0) == hr_pointer && get_count(handle, 0) == 0,
                  "decode command cut at +0x20 -> E_POINTER, nothing queued");
            VirtualFree(pages, 0, MEM_RELEASE);
        } else {
            check(false, "VirtualAlloc guard pages");
        }
    }

    // No packet (null, 0 bytes) completes with 0 bytes and writes nothing, as without libopus.
    {
        auto out = filled(960);
        const auto cmd = decode_cmd(13, 0.0f, 0, nullptr, 0, out.data(), 3840);
        check(enqueue(handle, cmd.data(), 0, 0) == hr_ok, "no-packet command accepted");
        const Result r = pop();
        check(r.type == 3 && r.slot == 13 && r.bytes == 0 && untouched(out), "no-packet command completes with 0 bytes");
    }

    // Reset (5) and close (2) each leave slot 7 decoding like a fresh decoder, not a continuation.
    {
        check(mono_matches(7, 48000, 960, mono[1], ref_mono), "slot 7 continues before reset");
        check(enqueue(handle, plain_cmd(5, 7).data(), 0, 0) == hr_ok, "reset slot 7 accepted");
        const Result reset = pop();
        check(reset.type == 5 && reset.slot == 7, "reset result queued");
        check(mono_matches(7, 48000, 960, mono[1], nullptr), "decode after reset matches a fresh decoder");
        check(mono_matches(7, 48000, 960, mono[2], nullptr) == false, "control: a used decoder differs from a fresh one");
        check(enqueue(handle, plain_cmd(2, 7).data(), 0, 0) == hr_ok, "close slot 7 accepted");
        const Result close = pop();
        check(close.type == 2 && close.slot == 7, "close result queued");
        check(mono_matches(7, 48000, 960, mono[1], nullptr), "decode after close matches a fresh decoder");
    }

    // A new rate on a live slot starts a new decoder; a failed decode at a new rate keeps the old one.
    {
        const auto low = tone_packets(16000, 1, 320, 1);
        check(mono_matches(7, 16000, 320, low[0], nullptr), "slot 7 at 16 kHz decodes like a fresh 16 kHz decoder");
        const unsigned char bad[] = {0x03, 0x00};
        auto out = filled(960);
        expect_refused("malformed packet at a new rate -> E_FAIL", hr_fail,
                       decode_cmd(9, 16000.0f, 320, bad, sizeof(bad), out.data(), 1280), out);
        auto out9 = filled(960);
        std::vector<float> want(960);
        const auto& p = short_packets[3];
        expect_refused("malformed packet at the current rate -> E_FAIL", hr_fail,
                       decode_cmd(9, 48000.0f, 960, bad, sizeof(bad), out.data(), 3840), out);
        expect_refused("valid packet in too-small capacity -> E_FAIL", hr_fail,
                       decode_cmd(9, 48000.0f, 80, p.data(), static_cast<std::uint32_t>(p.size()), out.data(), 320), out);
        const auto cmd = decode_cmd(9, 48000.0f, 960, p.data(), static_cast<std::uint32_t>(p.size()), out9.data(), 3840);
        check(enqueue(handle, cmd.data(), 0, 0) == hr_ok && pop().bytes == 1920 &&
                  opus_decode_float(ref_10ms, p.data(), static_cast<opus_int32>(p.size()), want.data(), 960, 0) == 480 &&
                  std::memcmp(out9.data(), want.data(), 480 * sizeof(float)) == 0,
              "after the failed decode slot 9 continues its 48 kHz stream");
    }

    // Typed completions off: type 3 queues the zeroed result and decodes nothing.
    {
        SetEnvironmentVariableA("X2W_XAPU_TYPED_COMPLETIONS", nullptr);
        auto out = filled(960);
        const auto cmd = decode_cmd(14, 48000.0f, 960, mono[0].data(), static_cast<std::uint32_t>(mono[0].size()), out.data(), 3840);
        check(enqueue(handle, cmd.data(), 0, 0) == hr_ok, "untyped type 3 accepted");
        std::array<unsigned char, 0x58> raw{};
        raw.fill(0x55);
        bool zero = dequeue(handle, raw.data()) == hr_ok;
        for (unsigned char b : raw) zero = zero && b == 0;
        check(zero && untouched(out), "untyped type 3 queues a zeroed result and leaves the output untouched");
    }

    check(get_count(handle, 0) == 0, "queue drained");
    check(disconnect(handle) == hr_ok, "XApuDisconnect frees slot decoders");
    opus_decoder_destroy(ref_mono);
    opus_decoder_destroy(ref_stereo);
    opus_decoder_destroy(ref_10ms);
    opus_decoder_destroy(ref_16k);
    FreeLibrary(module);
    if (failures) return std::printf("%d failure(s)\n", failures), 1;
    std::printf("PASS: startup_xapu_opus (%s)\n", opus_get_version_string());
    return 0;
}
