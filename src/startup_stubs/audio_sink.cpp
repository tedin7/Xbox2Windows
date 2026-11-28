#include "startup_stubs/audio_sink.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <objbase.h>
#include <mmreg.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <wrl/client.h>

namespace x2w_audio {

using Microsoft::WRL::ComPtr;

struct AudioSink::Impl {
    ComPtr<IAudioClient> client;
    ComPtr<IAudioRenderClient> render;
    ComPtr<IAudioClock> clock;
    UINT64 clock_freq = 1;
};

AudioSink::AudioSink() = default;

AudioSink::~AudioSink()
{
    if (p_ && p_->client) p_->client->Stop();
}

static std::string friendly_name(IMMDevice* dev)
{
    // PKEY_Device_FriendlyName, spelled out to avoid the property-key header link requirements.
    const PROPERTYKEY key = {{0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}}, 14};
    ComPtr<IPropertyStore> props;
    PROPVARIANT v;
    PropVariantInit(&v);
    std::string out = "(unnamed endpoint)";
    if (SUCCEEDED(dev->OpenPropertyStore(STGM_READ, &props)) && SUCCEEDED(props->GetValue(key, &v)) &&
        v.vt == VT_LPWSTR && v.pwszVal) {
        char buf[256] = {};
        if (WideCharToMultiByte(CP_UTF8, 0, v.pwszVal, -1, buf, sizeof(buf) - 1, nullptr, nullptr) > 0) out = buf;
    }
    PropVariantClear(&v);
    return out;
}

// Accepts 32-bit float and 16-bit PCM mix formats (extensible or plain).
static bool mix_kind(const WAVEFORMATEX* wf, SampleKind& kind)
{
    WORD tag = wf->wFormatTag;
    if (tag == WAVE_FORMAT_EXTENSIBLE && wf->cbSize >= 22)
        tag = static_cast<WORD>(reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(wf)->SubFormat.Data1);
    if (tag == WAVE_FORMAT_IEEE_FLOAT && wf->wBitsPerSample == 32) kind = SampleKind::Float32;
    else if (tag == WAVE_FORMAT_PCM && wf->wBitsPerSample == 16) kind = SampleKind::Pcm16;
    else return false;
    return true;
}

std::string AudioSink::open()
{
    auto impl = std::make_unique<Impl>();
    ComPtr<IMMDeviceEnumerator> en;
    ComPtr<IMMDevice> dev;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&en));
    if (FAILED(hr)) return "no device enumerator, hr=" + std::to_string(static_cast<long>(hr));
    hr = en->GetDefaultAudioEndpoint(eRender, eConsole, &dev);
    if (FAILED(hr)) return "no default render endpoint, hr=" + std::to_string(static_cast<long>(hr));
    endpoint_ = friendly_name(dev.Get());
    hr = dev->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(impl->client.GetAddressOf()));
    if (FAILED(hr)) return "endpoint activation failed, hr=" + std::to_string(static_cast<long>(hr));
    WAVEFORMATEX* wf = nullptr;
    hr = impl->client->GetMixFormat(&wf);
    if (FAILED(hr) || !wf) return "GetMixFormat failed, hr=" + std::to_string(static_cast<long>(hr));
    SampleKind kind = SampleKind::Float32;
    const bool ok = mix_kind(wf, kind);
    if (!ok) {
        const std::string s = "unsupported mix format tag=" + std::to_string(wf->wFormatTag) +
                              " bits=" + std::to_string(wf->wBitsPerSample);
        CoTaskMemFree(wf);
        return s;
    }
    mix_ = {kind, wf->nChannels, wf->nSamplesPerSec};
    constexpr REFERENCE_TIME buffer_100ns = 5000000;  // 500 ms
    hr = impl->client->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, buffer_100ns, 0, wf, nullptr);
    CoTaskMemFree(wf);
    if (FAILED(hr)) return "IAudioClient::Initialize failed, hr=" + std::to_string(static_cast<long>(hr));
    UINT32 frames = 0;
    if (FAILED(impl->client->GetBufferSize(&frames)) || FAILED(impl->client->GetService(IID_PPV_ARGS(&impl->render))) ||
        FAILED(impl->client->GetService(IID_PPV_ARGS(&impl->clock))) || FAILED(impl->clock->GetFrequency(&impl->clock_freq)))
        return "endpoint services unavailable";
    buffer_frames_ = frames;
    p_ = std::move(impl);
    resample_.reset();
    return {};
}

uint32_t AudioSink::padding()
{
    UINT32 pad = 0;
    if (!p_) return 0;
    if (FAILED(p_->client->GetCurrentPadding(&pad))) {
        ++r_.padding_errors;
        return padding_failed;
    }
    return pad;
}

void AudioSink::observe(bool withholding)
{
    if (!p_) return;
    const double now = now_ms();
    const uint32_t pad = padding();
    if (pad == padding_failed) return;  // no observation: neither consumption nor the queue timers move
    const uint64_t consumed = r_.submitted_frames > pad ? r_.submitted_frames - pad : 0;
    r_.consumed_frames = std::max(r_.consumed_frames, consumed);
    UINT64 pos = 0;
    if (started_ && SUCCEEDED(p_->clock->GetPosition(&pos, nullptr)))
        r_.clock_frames = static_cast<uint64_t>(static_cast<double>(pos) * mix_.rate / static_cast<double>(p_->clock_freq));
    timer_.sample_queue(now, pad, withholding && r_.submitted_frames > 0);
    timer_.sample_consumed(now, r_.consumed_frames);
}

Submit AudioSink::submit(const BlockFormat& in, uint32_t frames, const void* bytes)
{
    const uint32_t call = calls_++;
    if (!p_) {
        ++r_.not_open;
        return Submit::NotOpen;
    }
    ResampleState trial = resample_;  // committed only once the block is accepted
    if (!convert_block(in, bytes, frames, mix_, conv_buf_, &trial)) {
        ++r_.unsupported;
        return Submit::Unsupported;
    }
    const uint32_t out_frames = static_cast<uint32_t>(conv_buf_.size() / mix_.frame_bytes());
    const uint32_t pad = padding();
    if (pad == padding_failed) {
        ++r_.not_open;
        return Submit::NotOpen;
    }
    if (out_frames > buffer_frames_ - pad) {
        ++r_.rejected_full;
        return Submit::Full;
    }
    observe(false);
    if (timer_.drained_start >= 0) timer_.resume(now_ms(), r_.consumed_frames);  // the producer is back
    if (out_frames > 0) {
        BYTE* dst = nullptr;
        if (FAILED(p_->render->GetBuffer(out_frames, &dst)) || !dst) {
            ++r_.not_open;
            return Submit::NotOpen;
        }
        std::memcpy(dst, conv_buf_.data(), conv_buf_.size());
        if (FAILED(p_->render->ReleaseBuffer(out_frames, 0))) {
            ++r_.not_open;
            return Submit::NotOpen;
        }
    }
    resample_ = std::move(trial);
    if (r_.first_block < 0) {
        r_.first_block = call;
        r_.submitted = in;
    }
    r_.frames.push_back(out_frames);
    r_.peaks.push_back(block_peak(in, bytes, frames));
    r_.submitted_frames += out_frames;
    ++r_.accepted;
    if (!started_ && SUCCEEDED(p_->client->Start())) started_ = true;
    return Submit::Accepted;
}

Report AudioSink::report() const
{
    Report r = r_;
    r.drained_ms = timer_.drained_ms + (timer_.drained_start >= 0 ? now_ms() - timer_.drained_start : 0);
    r.recovery_ms = timer_.recovery_ms;
    return r;
}

}  // namespace x2w_audio
