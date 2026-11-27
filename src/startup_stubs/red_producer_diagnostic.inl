// Opt-in diagnostic for writes to the title-selected presentation texture.
extern LONG frame_number;

class RedProducers {
    using Resource = Microsoft::WRL::ComPtr<ID3D12Resource>;
    using Fence = Microsoft::WRL::ComPtr<ID3D12Fence>;
    static constexpr UINT64 after_offset = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    static constexpr std::size_t history_limit = 64;
    struct Entry {
        ID3D12GraphicsCommandList* list;
        Resource readback, surface;
        Fence fence;
        ID3D12CommandQueue* queue = nullptr;
        LONG record_frame, submit_frame = 0;
        unsigned attachment, ordinal, submit_order = 0;
        const char* operation;
        u64 key;
        bool live = true;
        bool pending = false;
        bool ambiguous = false;
    };
    struct History {
        const void* list;
        const void* surface;
        LONG frame;
        unsigned attachment;
        const char* operation;
        u64 key;
    };
    std::mutex lock_;
    std::vector<Resource> surfaces_, inputs_;
    std::vector<Entry> entries_;
    LONG first_ = 0, last_ = 0;
    LONG auto_min_ = 0, auto_count_ = 0;
    volatile LONG auto_first_ = 0, auto_last_ = 0;
    unsigned target_rgb_[3]{255, 0, 0};
    bool auto_last_match_ = false, onset_ = false;
    bool black_onset_ = false, previous_nonblack_ = false;
    LONG64 previous_sample_ = 0;
    unsigned char previous_rgba_[4]{};
    History history_[history_limit]{};
    std::size_t history_next_ = 0, history_count_ = 0;
    unsigned submit_order_ = 0;
    bool full_logged_ = false;

public:
    struct Disabled {};
    explicit RedProducers(Disabled) {}

    RedProducers()
    {
        char value[48]{};
        const DWORD onset_bytes = GetEnvironmentVariableA("X2W_RED_PRODUCER_ONSET", value, sizeof(value));
        black_onset_ = onset_bytes == 5 && std::strcmp(value, "black") == 0;
        onset_ = black_onset_ || (onset_bytes == 1 && value[0] == '1');
        if (GetEnvironmentVariableA("X2W_RED_PRODUCER_FRAMES", value, sizeof(value)) &&
            std::sscanf(value, "%ld,%ld", &first_, &last_) == 2 && first_ >= 1 && last_ >= first_) {
            log("x2w-red producer capture frames %ld..%ld; max 512 center texels from known present targets", first_, last_);
        } else {
            first_ = last_ = 0;
        }
        if (GetEnvironmentVariableA("X2W_RED_PRODUCER_AUTO", value, sizeof(value)) &&
            std::sscanf(value, "%ld,%ld", &auto_min_, &auto_count_) == 2 && auto_min_ >= 1 &&
            auto_count_ >= 1 && auto_count_ <= 120) {
            const DWORD rgb_bytes = black_onset_ ? 0 : GetEnvironmentVariableA("X2W_RED_PRODUCER_RGB", value, sizeof(value));
            if (rgb_bytes) {
                unsigned rgb[3]{};
                char trailing;
                if (rgb_bytes >= sizeof(value) || std::sscanf(value, "%3u,%3u,%3u%c", &rgb[0], &rgb[1], &rgb[2], &trailing) != 3 ||
                    rgb[0] > 255 || rgb[1] > 255 || rgb[2] > 255) {
                    auto_min_ = auto_count_ = 0;
                    log("x2w-red producer auto disabled: X2W_RED_PRODUCER_RGB needs three decimal components from 0 to 255");
                    return;
                }
                std::copy_n(rgb, 3, target_rgb_);
            }
            if (black_onset_) {
                std::fill_n(target_rgb_, 3, 0);
                log("x2w-red producer waits for flat black onset at or after present %ld then captures %ld future frames",
                    auto_min_, auto_count_);
            } else log("x2w-red producer waits for RGB %u,%u,%u at or after present %ld then captures %ld future frames",
                target_rgb_[0], target_rgb_[1], target_rgb_[2], auto_min_, auto_count_);
        } else {
            auto_min_ = auto_count_ = 0;
        }
    }

    bool enabled() const { return first_ || auto_count_; }
    bool onset() const { return enabled() && onset_; }
    bool black_onset() const { return enabled() && black_onset_; }
    bool matches(const unsigned char* rgba, bool flat, bool nonblack) const
    {
        if (black_onset_) return flat && !nonblack;
        if (onset_ && !flat) return false;
        const int tolerance = onset_ ? x2w_present_signature::onset_rgb_tolerance : 0;
        for (unsigned c = 0; c < 3; ++c)
            if (std::abs(int(rgba[c]) - int(target_rgb_[c])) > tolerance) return false;
        return true;
    }
    bool wants_history() const
    {
        return auto_count_ && !auto_last_ && frame_number + 1 >= auto_min_ - (onset_ ? 1 : 0);
    }
    bool interested() const
    {
        const LONG next = frame_number + 1;
        return (next >= first_ && next <= last_) || (next >= auto_first_ && next <= auto_last_);
    }
    bool may_record() const
    {
        return interested() || wants_history();
    }

    void observe(LONG64 sampled_frame, LONG published_frame, ID3D12Resource* surface, const unsigned char* rgba, bool matched, bool nonblack)
    {
        if (!auto_count_ || auto_last_) return;
        std::lock_guard<std::mutex> guard(lock_);
        const LONG64 previous = previous_sample_;
        const bool became_matched = matched && (black_onset_ ? previous_nonblack_
                                                            : !auto_last_match_ && (!onset_ || previous));
        auto_last_match_ = matched;
        if (auto_last_ || !became_matched || sampled_frame < auto_min_ || published_frame > LONG_MAX - auto_count_) {
            if (onset_) {
                previous_sample_ = sampled_frame;
                previous_nonblack_ = nonblack;
                std::copy_n(rgba, 4, previous_rgba_);
            }
            return;
        }
        InterlockedExchange(&auto_first_, published_frame + 1);
        InterlockedExchange(&auto_last_, published_frame + auto_count_);
        log("x2w-red auto armed sampled_present=%lld published_at=%ld target=%p capture=%ld..%ld rgb=%u,%u,%u",
            sampled_frame, published_frame, surface, static_cast<LONG>(auto_first_), static_cast<LONG>(auto_last_),
            target_rgb_[0], target_rgb_[1], target_rgb_[2]);
        if (onset_) log("x2w-red %s onset previous_present=%lld sampled_present=%lld previous_rgba=%u,%u,%u,%u",
            black_onset_ ? "black" : "flat", previous, sampled_frame, previous_rgba_[0], previous_rgba_[1], previous_rgba_[2], previous_rgba_[3]);
        for (std::size_t i = 0; i < history_count_; ++i) {
            const auto& e = history_[(history_next_ + history_limit - history_count_ + i) % history_limit];
            log("x2w-red history record=%ld list=%p target=%p op=%s key=%016llx rt=%u",
                e.frame, e.list, e.surface, e.operation, e.key, e.attachment);
        }
    }

    void present(ID3D12Resource* surface)
    {
        if (!enabled() || !surface) return;
        std::lock_guard<std::mutex> guard(lock_);
        const auto found = std::find_if(surfaces_.begin(), surfaces_.end(), [=](const Resource& r) { return r.Get() == surface; });
        if (found == surfaces_.end() && surfaces_.size() < 8) {
            surfaces_.emplace_back(surface);
            const auto desc = surface->GetDesc();
            log("x2w-red present target=%p size=%llux%u format=%d", surface, desc.Width, desc.Height, desc.Format);
        }
    }

    void follow_inputs(const Pass& pass, ID3D12Resource* const* textures, std::size_t count, u64 key)
    {
        if (!count || !pass.targets_bound) return;
        std::lock_guard<std::mutex> guard(lock_);
        for (UINT i = 0; i < pass.target_count; ++i) {
            auto* target = pass.trace_targets[i];
            if (std::none_of(surfaces_.begin(), surfaces_.end(), [=](const Resource& r) { return r.Get() == target; }))
                continue;
            for (std::size_t j = 0; j < count; ++j) {
                auto* input = textures[j];
                if (input == target || std::any_of(inputs_.begin(), inputs_.end(),
                    [=](const Resource& r) { return r.Get() == input; })) continue;
                if (inputs_.size() == 32) return;
                inputs_.emplace_back(input);
                log("x2w-red input record=%ld target=%p source=%p key=%016llx", frame_number + 1, target, input, key);
            }
        }
    }

    unsigned before(Wrapper* owner, ID3D12GraphicsCommandList* list, ID3D12Resource* surface,
                    const char* operation, u64 key, unsigned attachment)
    {
        if (!may_record() || !surface || list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) return 0;
        std::lock_guard<std::mutex> guard(lock_);
        const auto found = std::find_if(surfaces_.begin(), surfaces_.end(), [=](const Resource& r) { return r.Get() == surface; });
        if (found == surfaces_.end() && std::none_of(inputs_.begin(), inputs_.end(),
            [=](const Resource& r) { return r.Get() == surface; })) return 0;
        if (wants_history()) {
            history_[history_next_] = {list, surface, frame_number + 1, attachment, operation, key};
            history_next_ = (history_next_ + 1) % history_limit;
            if (history_count_ < history_limit) ++history_count_;
        }
        if (!interested()) return 0;
        if (entries_.size() >= 512) {
            if (!full_logged_) log("x2w-red producer capture limit reached at frame %ld", frame_number + 1);
            full_logged_ = true;
            return 0;
        }
        const auto desc = surface->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
            desc.DepthOrArraySize != 1 || desc.MipLevels != 1 || desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM) return 0;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = after_offset + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        const auto ordinal = static_cast<unsigned>(entries_.size() + 1);
        Entry entry{list, {}, Resource(surface), {}, nullptr, frame_number + 1, 0, attachment, ordinal, 0, operation, key};
        const HRESULT hr = device_of(the_device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&entry.readback));
        if (FAILED(hr)) {
            log("x2w-red producer readback allocation failed hr=0x%08lx", static_cast<unsigned long>(hr));
            return 0;
        }
        x2w_budget::committed(device_of(the_device), entry.readback.Get(), D3D12_HEAP_TYPE_READBACK);
        entries_.push_back(std::move(entry));
        copy_pixel(owner, entries_.back(), 0);
        return ordinal;
    }

    void after(Wrapper* owner, unsigned ordinal)
    {
        if (!ordinal) return;
        std::lock_guard<std::mutex> guard(lock_);
        copy_pixel(owner, entries_[ordinal - 1], after_offset);
    }

private:
    void copy_pixel(Wrapper* owner, const Entry& entry, UINT64 offset)
    {
        ID3D12Resource* surface = entry.surface.Get();
        const auto desc = surface->GetDesc();
        D3D12_TEXTURE_COPY_LOCATION to{entry.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint.Offset = offset;
        to.PlacedFootprint.Footprint = {desc.Format, 1, 1, 1, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT};
        const D3D12_TEXTURE_COPY_LOCATION from{surface, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        const UINT x = static_cast<UINT>(desc.Width / 2);
        const UINT y = x2w_present_signature::sampled_row(x2w_present_signature::cells_y / 2, desc.Height, x2w_present_signature::cells_y);
        const D3D12_BOX pixel{x, y, 0, x + 1, y + 1, 1};
        require_state(owner, surface, D3D12_RESOURCE_STATE_COPY_SOURCE);
        entry.list->CopyTextureRegion(&to, 0, 0, 0, &from, &pixel);
        require_state(owner, surface, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

public:
    void discard(ID3D12GraphicsCommandList* list)
    {
        if (!enabled()) return;
        std::lock_guard<std::mutex> guard(lock_);
        for (Entry& entry : entries_) if (entry.list == list) entry.live = false;
    }

    template <class Invoke>
    void execute(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists, const Invoke& invoke)
    {
        if (!enabled()) { invoke(); return; }
        std::lock_guard<std::mutex> guard(lock_);
        flush_locked();
        invoke();
        Fence fence;
        bool tried = false;
        for (ID3D12CommandList* list : lists) {
            for (Entry& entry : entries_) {
                if (!entry.live || entry.list != list) continue;
                if (!tried) {
                    tried = true;
                    if (FAILED(device_of(the_device)->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) ||
                        FAILED(queue->Signal(fence.Get(), 1))) {
                        log("x2w-red producer fence failed; keeping recording buffers alive");
                        fence.Reset();
                    }
                }
                entry.ambiguous |= entry.pending || !fence;
                entry.fence = fence;
                entry.queue = queue;
                entry.submit_frame = frame_number + 1;
                entry.submit_order = ++submit_order_;
                entry.pending = true;
            }
        }
    }

    void flush()
    {
        if (!enabled()) return;
        std::lock_guard<std::mutex> guard(lock_);
        flush_locked();
    }

private:
    void flush_locked()
    {
        for (Entry& entry : entries_) {
            if (!entry.pending || !entry.fence || entry.fence->GetCompletedValue() != 1) continue;
            entry.pending = false;
            if (entry.ambiguous) {
                log("x2w-red producer ambiguous resubmit ordinal=%u order=%u", entry.ordinal, entry.submit_order);
                continue;
            }
            const D3D12_RANGE read{0, after_offset + 4}, none{};
            void* memory = nullptr;
            if (FAILED(entry.readback->Map(0, &read, &memory))) continue;
            unsigned char rgba[4], prior[4];
            std::memcpy(prior, memory, sizeof(prior));
            std::memcpy(rgba, static_cast<unsigned char*>(memory) + after_offset, sizeof(rgba));
            entry.readback->Unmap(0, &none);
            log("x2w-red producer order=%u ordinal=%u record=%ld submit=%ld queue=%p target=%p op=%s key=%016llx rt=%u rgba=%u,%u,%u,%u before=%u,%u,%u,%u",
                entry.submit_order, entry.ordinal, entry.record_frame, entry.submit_frame, entry.queue, entry.surface.Get(),
                entry.operation, entry.key, entry.attachment, rgba[0], rgba[1], rgba[2], rgba[3],
                prior[0], prior[1], prior[2], prior[3]);
        }
    }
};

RedProducers& red_producers()
{
    static RedProducers disabled(RedProducers::Disabled{});
    // Closed lists may be submitted again and still hold these GPU references.
    static auto* const tracker = [] {
        char value[2]{};
        if (GetEnvironmentVariableA("X2W_RED_PRODUCER_FRAMES", value, sizeof(value)) ||
            GetEnvironmentVariableA("X2W_RED_PRODUCER_AUTO", value, sizeof(value))) return new RedProducers;
        return &disabled;
    }();
    return *tracker;
}

struct RedDrawSamples {
    unsigned marks[8]{};
    unsigned count = 0;
};

RedDrawSamples red_producer_draw_before(Wrapper* self, ID3D12GraphicsCommandList* list, const char* operation, u64 key)
{
    RedDrawSamples samples;
    auto& tracker = red_producers();
    if (!tracker.may_record()) return samples;
    const Pass& pass = pass_of(self);
    if (!pass.targets_bound) return samples;
    if (tracker.interested())
        log("x2w-red attachments record=%ld list=%p op=%s key=%016llx dsv=%p/%d targets=%u rt0=%p/%d rt1=%p/%d rt2=%p/%d rt3=%p/%d rt4=%p/%d rt5=%p/%d rt6=%p/%d rt7=%p/%d",
            frame_number + 1, list, operation, key, pass.trace_depth, pass.dsv_format, pass.target_count,
            pass.trace_targets[0], pass.target_formats[0], pass.trace_targets[1], pass.target_formats[1],
            pass.trace_targets[2], pass.target_formats[2], pass.trace_targets[3], pass.target_formats[3],
            pass.trace_targets[4], pass.target_formats[4], pass.trace_targets[5], pass.target_formats[5],
            pass.trace_targets[6], pass.target_formats[6], pass.trace_targets[7], pass.target_formats[7]);
    for (UINT i = 0; i < pass.target_count; ++i) {
        if (const unsigned mark = tracker.before(self, list, pass.trace_targets[i], operation, key, i))
            samples.marks[samples.count++] = mark;
    }
    return samples;
}

void red_producer_draw_after(Wrapper* self, const RedDrawSamples& samples)
{
    for (unsigned i = 0; i < samples.count; ++i) red_producers().after(self, samples.marks[i]);
}
