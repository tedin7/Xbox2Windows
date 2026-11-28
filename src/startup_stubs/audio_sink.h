// Desktop render sink for startup audio experiments (WASAPI shared mode).
// The first half is pure (no platform headers) so its arithmetic can be built and run anywhere.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace x2w_audio {

enum class SampleKind { Float32, Pcm16 };

struct BlockFormat {
    SampleKind kind = SampleKind::Pcm16;
    uint32_t channels = 1;
    uint32_t rate = 48000;
    uint32_t frame_bytes() const { return channels * (kind == SampleKind::Float32 ? 4u : 2u); }
};

inline float read_sample(const BlockFormat& f, const void* src, size_t i)
{
    if (f.kind == SampleKind::Float32) {
        float v;
        std::memcpy(&v, static_cast<const uint8_t*>(src) + i * 4, 4);
        return v;
    }
    int16_t v;
    std::memcpy(&v, static_cast<const uint8_t*>(src) + i * 2, 2);
    return static_cast<float>(v) / 32768.0f;
}

inline void write_sample(const BlockFormat& f, void* dst, size_t i, float s)
{
    if (f.kind == SampleKind::Float32) {
        std::memcpy(static_cast<uint8_t*>(dst) + i * 4, &s, 4);
        return;
    }
    const float scaled = std::clamp(std::nearbyint(s * 32768.0f), -32768.0f, 32767.0f);
    const int16_t v = static_cast<int16_t>(scaled);
    std::memcpy(static_cast<uint8_t*>(dst) + i * 2, &v, 2);
}

// Largest absolute sample in 0..1. Exactly 0 means every sample is zero: a valid silent block.
inline float block_peak(const BlockFormat& f, const void* data, uint32_t frames)
{
    if (!data || frames == 0 || f.channels == 0) return 0.0f;
    float peak = 0.0f;
    for (size_t i = 0; i < size_t(frames) * f.channels; ++i)
        peak = (std::max)(peak, std::fabs(read_sample(f, data, i)));
    return peak;
}

inline bool is_valid_sample_rate(uint32_t rate)
{
    return rate >= 1000u && rate <= 384000u;
}

inline bool is_supported_channel_conversion(uint32_t in_ch, uint32_t out_ch)
{
    if (in_ch == 0 || out_ch == 0) return false;
    if (in_ch == out_ch || in_ch == 1u) return true;
    if (in_ch == 2u && out_ch == 1u) return true;
    if ((in_ch == 6u || in_ch == 8u) && (out_ch == 1u || out_ch == 2u)) return true;
    return false;
}

// ITU-R BS.775-1 fold-down coefficient (1 / sqrt(2)) for center and surround channels.
inline constexpr float kItuSurroundCoeff = 0.7071067811865475f;

inline float read_channel_mapped_sample(const BlockFormat& in, const void* src, size_t frame, uint32_t out_ch,
                                        uint32_t out_channels)
{
    const size_t base = frame * in.channels;
    if (in.channels == out_channels) return read_sample(in, src, base + out_ch);
    if (in.channels == 1u) return read_sample(in, src, base);
    if (in.channels == 2u && out_channels == 1u) {
        const float left = read_sample(in, src, base);
        const float right = read_sample(in, src, base + 1);
        return std::clamp(0.5f * (left + right), -1.0f, 1.0f);
    }
    // 5.1 (FL, FR, FC, LFE, BL, BR) or 7.1 (FL, FR, FC, LFE, BL, BR, SL, SR) -> stereo / mono.
    const float fl = read_sample(in, src, base + 0);
    const float fr = read_sample(in, src, base + 1);
    const float fc = read_sample(in, src, base + 2);
    const float bl = read_sample(in, src, base + 4);
    const float br = read_sample(in, src, base + 5);
    float left = fl + kItuSurroundCoeff * fc + kItuSurroundCoeff * bl;
    float right = fr + kItuSurroundCoeff * fc + kItuSurroundCoeff * br;
    if (in.channels == 8u) {
        const float sl = read_sample(in, src, base + 6);
        const float sr = read_sample(in, src, base + 7);
        left += kItuSurroundCoeff * sl;
        right += kItuSurroundCoeff * sr;
    }
    // Mono: average the unclipped stereo sums, clamp only the final value.
    if (out_channels == 1u) return std::clamp(0.5f * (left + right), -1.0f, 1.0f);
    return std::clamp(out_ch == 0u ? left : right, -1.0f, 1.0f);
}

inline uint32_t resampled_frame_count(uint32_t frames, uint32_t in_rate, uint32_t out_rate)
{
    if (frames == 0 || !is_valid_sample_rate(in_rate) || !is_valid_sample_rate(out_rate)) return 0;
    if (in_rate == out_rate) return frames;
    const uint64_t num = static_cast<uint64_t>(frames) * out_rate + (in_rate / 2u);
    const uint64_t scaled = (std::max)(uint64_t{1}, num / in_rate);
    if (scaled > (std::numeric_limits<uint32_t>::max)()) return 0;
    return static_cast<uint32_t>(scaled);
}

// Resampler memory carried between accepted blocks of one stream: the exact fractional read
// position (in 1/out_rate input-frame units, relative to the next block's first frame; -out_rate
// <= phase_num means the next output sits between `last` and the next block's first frame) and the
// last channel-mapped input frame. Cleared on any input/output format change.
struct ResampleState {
    bool valid = false;
    BlockFormat in{}, out{};
    int64_t phase_num = 0;
    std::vector<float> last;
    void reset() { *this = ResampleState{}; }
};

// Converts a submitted block to the endpoint format, supporting mono/stereo/5.1/7.1 channel mapping
// (ITU-R BS.775-1 fold-down, clamped after the final fold-down) and linear sample-rate resampling.
// Preserves exact bit-for-bit behavior for matching-rate 1:1 and mono-to-N paths.
// With state == nullptr each block is converted on its own (tail frames clamp to the last input
// frame). With a state, consecutive blocks form one stream: the fractional phase and the boundary
// frame carry over, so the output does not depend on how the input was split into blocks; the
// final interpolation point waits for the next block, so a block may yield fewer (even zero)
// frames and convert_block then returns true with an empty dst.
inline bool convert_block(const BlockFormat& in, const void* src, uint32_t frames, const BlockFormat& out,
                          std::vector<uint8_t>& dst, ResampleState* state = nullptr)
{
    if (!src || frames == 0) return false;
    if (!is_valid_sample_rate(in.rate) || !is_valid_sample_rate(out.rate)) return false;
    if (!is_supported_channel_conversion(in.channels, out.channels)) return false;

    const bool same_rate = in.rate == out.rate;
    const bool carry = state && !same_rate;
    if (state && (!state->valid || state->in.kind != in.kind || state->in.channels != in.channels ||
                  state->in.rate != in.rate || state->out.kind != out.kind ||
                  state->out.channels != out.channels || state->out.rate != out.rate || same_rate))
        state->reset();

    if (!carry) {
        const uint32_t out_frames = resampled_frame_count(frames, in.rate, out.rate);
        if (out_frames == 0) return false;
        const uint64_t total_bytes = static_cast<uint64_t>(out_frames) * out.frame_bytes();
        if (total_bytes > (std::numeric_limits<size_t>::max)()) return false;
        dst.resize(static_cast<size_t>(total_bytes));

        if (same_rate) {
            if (in.channels == out.channels || in.channels == 1u) {
                for (size_t f = 0; f < frames; ++f)
                    for (size_t c = 0; c < out.channels; ++c)
                        write_sample(out, dst.data(), f * out.channels + c,
                                     read_sample(in, src, f * in.channels + (in.channels == 1u ? 0 : c)));
                return true;
            }
            for (size_t f = 0; f < frames; ++f)
                for (uint32_t c = 0; c < out.channels; ++c)
                    write_sample(out, dst.data(), f * out.channels + c,
                                 read_channel_mapped_sample(in, src, f, c, out.channels));
            return true;
        }

        const double step = static_cast<double>(in.rate) / static_cast<double>(out.rate);
        for (uint32_t f = 0; f < out_frames; ++f) {
            const double pos = static_cast<double>(f) * step;
            const uint32_t i0 = (std::min)(static_cast<uint32_t>(pos), frames - 1u);
            const uint32_t i1 = (std::min)(i0 + 1u, frames - 1u);
            const float frac = static_cast<float>(pos - static_cast<double>(i0));
            for (uint32_t c = 0; c < out.channels; ++c) {
                const float s0 = read_channel_mapped_sample(in, src, i0, c, out.channels);
                const float s1 = (i1 == i0) ? s0 : read_channel_mapped_sample(in, src, i1, c, out.channels);
                const float interp = std::clamp(s0 + frac * (s1 - s0), -1.0f, 1.0f);
                write_sample(out, dst.data(), static_cast<size_t>(f) * out.channels + c, interp);
            }
        }
        return true;
    }

    // Streaming path. Output n reads input position n * in_rate / out_rate; phase_num is that
    // position in 1/out_rate units relative to this block's first frame (may be negative).
    const int64_t orate = out.rate;
    const int64_t span = static_cast<int64_t>(frames) * orate;
    int64_t phase = state->valid ? state->phase_num : 0;
    // Emit while the right-hand neighbour (floor(pos) + 1) lies inside this block: pos < frames - 1.
    int64_t count = 0;
    if (phase < span - orate) count = (span - orate - phase + in.rate - 1) / in.rate;
    const uint64_t total_bytes = static_cast<uint64_t>(count) * out.frame_bytes();
    if (count > static_cast<int64_t>((std::numeric_limits<uint32_t>::max)()) ||
        total_bytes > (std::numeric_limits<size_t>::max)())
        return false;
    dst.resize(static_cast<size_t>(total_bytes));
    const bool have_last = state->valid && state->last.size() == out.channels;
    for (int64_t f = 0; f < count; ++f, phase += in.rate) {
        const int64_t base = phase >= 0 ? phase / orate : -((-phase + orate - 1) / orate);
        const float frac = static_cast<float>(static_cast<double>(phase - base * orate) / static_cast<double>(orate));
        for (uint32_t c = 0; c < out.channels; ++c) {
            const float s0 = base < 0 ? (have_last ? state->last[c] : 0.0f)
                                      : read_channel_mapped_sample(in, src, static_cast<size_t>(base), c, out.channels);
            const float s1 = read_channel_mapped_sample(in, src, static_cast<size_t>(base + 1), c, out.channels);
            const float interp = std::clamp(s0 + frac * (s1 - s0), -1.0f, 1.0f);
            write_sample(out, dst.data(), static_cast<size_t>(f) * out.channels + c, interp);
        }
    }
    state->in = in;
    state->out = out;
    state->phase_num = phase - span;
    state->last.assign(out.channels, 0.0f);
    for (uint32_t c = 0; c < out.channels; ++c)
        state->last[c] = read_channel_mapped_sample(in, src, frames - 1u, c, out.channels);
    state->valid = true;
    return true;
}

inline double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Drained interval: time the playing queue sat empty while the producer withheld data.
// Recovery: time from the producer resuming to consumption advancing again. Both come from
// timestamps and frame counters only, and are independent of any glitch count.
struct StarvationTimer {
    double drained_start = -1, drained_ms = 0, resume_at = -1, recovery_ms = -1;
    uint64_t consumed_at_resume = 0;

    void sample_queue(double now, uint32_t padding, bool withholding)
    {
        if (withholding && padding == 0 && drained_start < 0) drained_start = now;
    }
    void resume(double now, uint64_t consumed)
    {
        if (drained_start >= 0) drained_ms += now - drained_start;
        drained_start = -1;
        resume_at = now;
        consumed_at_resume = consumed;
        recovery_ms = -1;
    }
    void sample_consumed(double now, uint64_t consumed)
    {
        if (resume_at >= 0 && recovery_ms < 0 && consumed > consumed_at_resume) recovery_ms = now - resume_at;
    }
};

struct Report {
    int64_t first_block = -1;         // index of the submit() call whose block was accepted first
    BlockFormat submitted{};          // format of the first accepted block
    std::vector<uint32_t> frames;     // frame count of every accepted block, in order
    std::vector<float> peaks;         // peak amplitude of every accepted block (0 = silent)
    uint64_t submitted_frames = 0;    // frames written to the endpoint
    uint64_t consumed_frames = 0;     // submitted frames minus frames still queued
    uint64_t clock_frames = 0;        // IAudioClock position converted to frames
    uint64_t accepted = 0;            // submit() calls that returned Submit::Accepted
    uint32_t unsupported = 0, rejected_full = 0, not_open = 0;
    uint32_t padding_errors = 0;      // GetCurrentPadding failed (e.g. the endpoint went away): not a measurement
    double drained_ms = 0, recovery_ms = -1;  // recovery_ms < 0: not measured
    // Glitches: the render API exposes no discontinuity flag, so this stays "not available".
    bool glitches_available = false;
};

enum class Submit { Accepted, Full, Unsupported, NotOpen };

#ifdef _WIN32
class AudioSink {
public:
    AudioSink();
    ~AudioSink();
    AudioSink(const AudioSink&) = delete;
    AudioSink& operator=(const AudioSink&) = delete;
    // Empty string on success; otherwise why there is no usable endpoint (a capability skip).
    std::string open();
    Submit submit(const BlockFormat& in, uint32_t frames, const void* bytes);
    // Samples padding and the clock. withholding = the producer is deliberately not submitting.
    void observe(bool withholding);
    // Frames still queued, or padding_failed (and padding_errors counts it) when the query fails.
    static constexpr uint32_t padding_failed = 0xFFFFFFFFu;
    uint32_t padding();
    Report report() const;
    const std::string& endpoint() const { return endpoint_; }
    const BlockFormat& mix_format() const { return mix_; }
    uint32_t buffer_frames() const { return buffer_frames_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    std::string endpoint_;
    BlockFormat mix_{};
    uint32_t buffer_frames_ = 0;
    uint32_t calls_ = 0;
    bool started_ = false;
    Report r_;
    StarvationTimer timer_;
    std::vector<uint8_t> conv_buf_;
    ResampleState resample_;
};
#endif

}  // namespace x2w_audio
