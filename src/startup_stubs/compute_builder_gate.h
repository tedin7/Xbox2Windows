#pragma once

// Opt-in all-or-none gate for the console BVH builder's compute pipeline set
// (X2W_COMPUTE_BUILDER_GATE=1). Pure helpers and state, with no Windows calls, so a source-built
// control can check the decision without a device or a title module.
//
// A builder pipeline is a compute pipeline created through device slot 62 whose caller module is
// XboxBvhBuilder.dll. A builder Dispatch is allowed only while every tracked builder pipeline has a
// real desktop pipeline state. The first builder Dispatch attempts the missing creations; one failure
// latches the builder off for the rest of the process, so a partial builder cannot write through
// root views into title memory.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace x2w_startup {

// The basename of a module path, case-insensitive and accepting either slash. The console builder
// is the only module whose slot 62 compute pipelines this gate tracks.
inline bool is_compute_builder_module(std::wstring_view path) noexcept
{
    constexpr wchar_t expected[] = L"XboxBvhBuilder.dll";
    constexpr std::size_t expected_size = sizeof(expected) / sizeof(expected[0]) - 1;
    const std::size_t slash = path.find_last_of(L"\\/");
    const std::wstring_view base = slash == std::wstring_view::npos ? path : path.substr(slash + 1);
    if (base.size() != expected_size) return false;
    for (std::size_t i = 0; i < expected_size; ++i) {
        wchar_t got = base[i];
        wchar_t want = expected[i];
        if (got >= L'A' && got <= L'Z') got = static_cast<wchar_t>(got - L'A' + L'a');
        if (want >= L'A' && want <= L'Z') want = static_cast<wchar_t>(want - L'A' + L'a');
        if (got != want) return false;
    }
    return true;
}

enum class BuilderGateDecision : std::uint8_t {
    unchanged,      // gate off, or the Dispatch does not use a tracked builder pipeline
    allow,          // every tracked builder pipeline has a desktop PSO
    refuse_missing, // at least one tracked builder pipeline has no desktop PSO; the caller may attempt creation
    refuse_off,     // a creation failed earlier; the builder stays off for the rest of the run
};

class BuilderGateState {
public:
    // 52 builder programs were observed in the inspected module. The margin leaves room for a
    // second creation pass without ever allowing an untracked pipeline into the all-or-none set.
    static constexpr unsigned max_pipelines = 64;

    explicit BuilderGateState(bool enabled = false) noexcept : enabled_(enabled) {}

    bool enabled() const noexcept { return enabled_; }
    bool off() const noexcept { return off_; }
    unsigned tracked() const noexcept { return tracked_; }
    unsigned with_pso() const noexcept { return with_pso_; }
    unsigned refusals() const noexcept { return refusals_; }
    const char* first_reason() const noexcept { return reason_; }

    // Records a slot 62 creation. Only builder pipelines are tracked; a non-builder one changes
    // nothing. A repeated pointer updates its state instead of adding an entry. Returns false when
    // the builder set cannot be tracked completely, in which case the gate latches off.
    bool note_pipeline(const void* pipeline, bool builder, bool real) noexcept
    {
        if (!enabled_ || !builder || !pipeline) return true;
        for (unsigned i = 0; i < tracked_; ++i) {
            if (entries_[i].pipeline != pipeline) continue;
            if (entries_[i].real != real) {
                if (real) ++with_pso_;
                else --with_pso_;
                entries_[i].real = real;
            }
            return true;
        }
        if (tracked_ == max_pipelines) {
            note_failed("builder pipeline table is full");
            return false;
        }
        entries_[tracked_++] = Entry{pipeline, real};
        if (real) ++with_pso_;
        return true;
    }

    // Marks one tracked pipeline as having a real desktop PSO after an on-the-spot creation.
    void note_pso(const void* pipeline) noexcept
    {
        if (!enabled_ || !pipeline) return;
        for (unsigned i = 0; i < tracked_; ++i) {
            if (entries_[i].pipeline != pipeline || entries_[i].real) continue;
            entries_[i].real = true;
            ++with_pso_;
            return;
        }
    }

    // Latches the builder off. The pointer must name a static string; it is reported later.
    void note_failed(const char* reason) noexcept
    {
        if (!enabled_ || off_) return;
        off_ = true;
        reason_ = reason ? reason : "builder pipeline creation failed";
    }

    // The next pipeline without a desktop PSO at or after index. A real entry yields nullptr, so
    // callers can walk the whole table while note_pso updates it.
    const void* missing_at(unsigned index) const noexcept
    {
        return index < tracked_ && !entries_[index].real ? entries_[index].pipeline : nullptr;
    }

    BuilderGateDecision decide(bool builder_dispatch) const noexcept
    {
        if (!enabled_ || !builder_dispatch) return BuilderGateDecision::unchanged;
        if (off_) return BuilderGateDecision::refuse_off;
        if (tracked_ && with_pso_ == tracked_) return BuilderGateDecision::allow;
        return BuilderGateDecision::refuse_missing;
    }

    void note_refusal() noexcept { ++refusals_; }

private:
    struct Entry {
        const void* pipeline = nullptr;
        bool real = false;
    };
    std::array<Entry, max_pipelines> entries_{};
    unsigned tracked_ = 0;
    unsigned with_pso_ = 0;
    unsigned refusals_ = 0;
    bool enabled_ = false;
    bool off_ = false;
    const char* reason_ = nullptr;
};

inline const char* builder_gate_reason(BuilderGateDecision decision) noexcept
{
    switch (decision) {
    case BuilderGateDecision::unchanged: return "unchanged";
    case BuilderGateDecision::allow: return "allow";
    case BuilderGateDecision::refuse_missing: return "not every builder pipeline has a desktop pipeline";
    case BuilderGateDecision::refuse_off: return "the builder is off for the rest of the run";
    }
    return "unknown";
}

}  // namespace x2w_startup
