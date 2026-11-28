// Source-built WASAPI render fixture (Refs #1145). Three cases against the default endpoint:
// (a) no submission, (b) a continuous stream with one silent block, (c) producer starvation.
// Exit 77 when there is no usable render endpoint. Capture-path verification is not included.
#include "startup_audio_pure_check.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <objbase.h>

using namespace x2w_audio;

namespace {

int fail(const char* what, int code)
{
    std::printf("FAIL: %s\n", what);
    return code;
}

void print_report(const char* name, const Report& r)
{
    std::printf("%s: first_block=%lld submitted_blocks=%zu accepted=%llu submitted_frames=%llu consumed_frames=%llu clock_frames=%llu "
                "unsupported=%u rejected_full=%u not_open=%u padding_errors=%u drained_ms=%.1f recovery_ms=%.1f glitches=%s\n",
                name, static_cast<long long>(r.first_block), r.frames.size(), static_cast<unsigned long long>(r.accepted),
                static_cast<unsigned long long>(r.submitted_frames), static_cast<unsigned long long>(r.consumed_frames),
                static_cast<unsigned long long>(r.clock_frames), r.unsupported, r.rejected_full, r.not_open,
                r.padding_errors, r.drained_ms, r.recovery_ms, r.glitches_available ? "reported" : "not available");
}

// Polls until done() holds or ms elapse; every wait in this fixture goes through here.
template <class F>
bool wait_for(AudioSink& s, double ms, F done)
{
    const double end = now_ms() + ms;
    for (;;) {
        s.observe(false);
        if (done()) return true;
        if (now_ms() > end) return false;
        Sleep(2);
    }
}

// Retries a block while the endpoint buffer is full, for at most two seconds.
Submit send(AudioSink& s, const std::vector<int16_t>& pcm, const BlockFormat& fmt)
{
    const double end = now_ms() + 2000;
    for (;;) {
        const Submit r = s.submit(fmt, static_cast<uint32_t>(pcm.size()), pcm.data());
        if (r != Submit::Full || now_ms() > end) return r;
        Sleep(2);
        s.observe(false);
    }
}

void print_windows_build()
{
    using Fn = LONG(WINAPI*)(OSVERSIONINFOW*);
    OSVERSIONINFOW v{sizeof(v)};
    const auto fn = reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion")));
    if (fn && fn(&v) == 0) std::printf("windows: %lu.%lu build %lu\n", v.dwMajorVersion, v.dwMinorVersion, v.dwBuildNumber);
    else std::puts("windows: build unknown");
}

int skip(const std::string& why)
{
    std::printf("SKIP (capability): %s\n", why.c_str());
    print_windows_build();
    return 77;
}

// The three sinks live in this function, so they stop and release their COM interfaces before
// main() calls CoUninitialize().
int run()
{
    AudioSink a;
    if (const std::string why = a.open(); !why.empty()) return skip(why);
    print_windows_build();
    const BlockFormat mix = a.mix_format();
    std::printf("endpoint: %s, mix %s %u ch %u Hz, buffer %u frames\n", a.endpoint().c_str(),
                mix.kind == SampleKind::Float32 ? "float32" : "pcm16", mix.channels, mix.rate, a.buffer_frames());

    // (a) No submission: nothing accepted, nothing consumed.
    wait_for(a, 300, [] { return false; });
    const Report ra = a.report();
    print_report("case a (no submission)", ra);
    if (ra.first_block != -1 || !ra.frames.empty() || ra.accepted != 0 || ra.submitted_frames != 0 || ra.consumed_frames != 0)
        return fail("case a reported accepted or consumed frames without a submission", 1);

    // (b) Continuous stream of eight distinguishable 40 ms blocks; block 3 is all zeros.
    const BlockFormat src{SampleKind::Pcm16, 1, mix.rate};
    const uint32_t block = mix.rate / 25;
    AudioSink b;
    if (const std::string why = b.open(); !why.empty()) return skip(why);
    for (int i = 0; i < 8; ++i) {
        const auto pcm = make_tone(mix.rate, block, 220.0 * (i + 1), i == 3 ? 0.0 : 0.1 + 0.07 * i);
        const Submit r = send(b, pcm, src);
        if (r == Submit::Unsupported) return skip("block format cannot be converted to the endpoint format");
        if (r != Submit::Accepted) return fail("case b block not accepted", 2);
    }
    wait_for(b, 3000, [&] { return b.report().consumed_frames == 8ull * block; });
    const Report rb = b.report();
    print_report("case b (continuous with silence)", rb);
    if (rb.first_block != 0 || rb.frames.size() != 8 || rb.accepted != 8 || rb.submitted_frames != 8ull * block) return fail("case b submitted counts", 3);
    for (const uint32_t f : rb.frames) if (f != block) return fail("case b block frame count", 3);
    if (rb.consumed_frames != rb.submitted_frames) return fail("case b consumption incomplete", 4);
    if (rb.peaks[3] != 0.0f) return fail("case b silent block was not silent", 5);
    for (int i = 0; i < 8; ++i)
        for (int j = i + 1; j < 8; ++j)
            if (rb.peaks[i] == rb.peaks[j]) return fail("case b blocks are not distinguishable", 5);

    // (c) Starvation: let the queue drain, hold, resume.
    AudioSink c;
    if (const std::string why = c.open(); !why.empty()) return skip(why);
    for (int i = 0; i < 2; ++i)
        if (send(c, make_tone(mix.rate, block, 330.0 + 110 * i, 0.4), src) != Submit::Accepted) return fail("case c first blocks", 6);
    if (!wait_for(c, 3000, [&] { return c.padding() == 0 && c.report().consumed_frames == 2ull * block; }))
        return fail("case c queue did not drain", 7);
    for (const double hold_end = now_ms() + 300; now_ms() < hold_end;) {
        c.observe(true);
        Sleep(2);
    }
    if (c.padding() != 0) return fail("case c queue refilled while withheld", 7);  // also fails on a failed query
    if (send(c, make_tone(mix.rate, block, 550.0, 0.4), src) != Submit::Accepted) return fail("case c resume block", 8);
    wait_for(c, 3000, [&] { return c.report().recovery_ms >= 0 && c.report().consumed_frames == 3ull * block; });
    const Report rc = c.report();
    print_report("case c (starvation)", rc);
    if (rc.accepted != 3) return fail("case c accepted block count", 9);
    if (rc.drained_ms < 250) return fail("case c drained interval too short", 9);
    if (rc.recovery_ms < 0) return fail("case c consumption did not recover", 10);
    if (rc.consumed_frames != 3ull * block) return fail("case c consumption incomplete", 10);
    // A failed padding query is not a measurement: reject the run instead of counting frames as consumed.
    if (ra.padding_errors || rb.padding_errors || rc.padding_errors) return fail("GetCurrentPadding failed during a case (endpoint lost?)", 11);

    // (d) Negative control + 5.1 / 44.1 kHz downmix & resample submission when endpoint is mono or stereo.
    if (mix.channels <= 2) {
        AudioSink d;
        if (const std::string why = d.open(); !why.empty()) return skip(why);
        const BlockFormat invalid_fmt{SampleKind::Float32, 2, 0};
        const float zero_pair[2] = {0.0f, 0.0f};
        if (d.submit(invalid_fmt, 1, zero_pair) != Submit::Unsupported)
            return fail("case d zero sample rate was not rejected as Unsupported", 12);
        const BlockFormat ch6_44{SampleKind::Float32, 6, 44100};
        std::vector<float> blk6(441 * 6, 0.0f);
        for (uint32_t i = 0; i < 441; ++i) blk6[i * 6 + 0] = 0.1f;
        if (d.submit(ch6_44, 441, blk6.data()) != Submit::Accepted)
            return fail("case d 5.1 44.1 kHz block not accepted", 13);
        const Report rd = d.report();
        if (rd.unsupported != 1 || rd.accepted != 1 || rd.submitted_frames + 2 < resampled_frame_count(441, 44100, mix.rate) ||
            rd.submitted_frames > resampled_frame_count(441, 44100, mix.rate))
            return fail("case d report counts mismatch", 14);
    }
    std::puts("startup_audio_output: PASS (no capture-path verification; glitch count not available from the render API)");
    return 0;
}

}  // namespace

int main()
{
    if (const int rc = check_pure()) return fail("pure helper checks", rc);
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) return skip("COM initialisation failed");
    const int rc = run();
    CoUninitialize();
    return rc;
}
