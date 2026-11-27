// Experimental (docs/startup-runtime.md): counts for two ways a sampled texture ends up without a view.
//
// MemorySrvCensus counts refused device slot 70 calls (a shader resource view whose resource is an
// address) by reason, and by distinct resource shape for the reasons that depend on the description.
// A refusal leaves the destination descriptor as it was. TokenNullCensus counts, per translated
// pipeline, the texture SRVs a draw bound null because the descriptor-table entry carries no stand-in
// view token (null_binding_cause.h, "token").
//
// MemorySrvWrites remembers which destination descriptors slot 70 has written, so the first write to
// each distinct entry can be logged.
//
// The censuses are thread-safe and format into caller buffers; only MemorySrvWrites allocates. report() hands each line to
// emit(const char*). With only_if_changed it prints nothing when no call was counted since the last
// such report.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_set>

namespace x2w_startup {

// Every refusal reason src/startup_stubs/d3d12_memory_srv.inl logs, in the order of its checks.
// tools/startup/triage_run.py keeps the same list (SLOT70_REASONS).
constexpr const char* memory_srv_reasons[] = {
    "null_destination", "unreadable_resource", "unreadable_view", "null_address",
    "unsupported_description", "destination_outside_view_heaps", "gpu_written_resource",
    // the acceleration-structure holds of publish_as_srv (memory_srv_as.h, as_srv_refusal_name)
    "as_unaligned", "as_not_placed", "as_wrong_state",
    "buffer_unavailable", "texture_unavailable",
    // the three causes texture_unavailable used to lump together (d3d12_memory_srv.inl, texture_unavailable_reason)
    "texture_no_resource", "texture_unreadable", "texture_over_64mib",
    // the partial-tile holds of device_memory_srv (upstream #2105)
    "partial_tile_xg_rows", "partial_tile_beyond_xg_size",
    // the console-layout holds of device_memory_srv (#2015)
    "unsupported_desktop_format", "console_layout_xg_rows", "console_layout_without_footprints",
    "console_layout_beyond_xg_size", "overlapping_resource",
};
constexpr int memory_srv_reason_count = static_cast<int>(sizeof(memory_srv_reasons) / sizeof(memory_srv_reasons[0]));

inline int memory_srv_reason_index(const char* reason)
{
    for (int i = 0; reason && i < memory_srv_reason_count; ++i)
        if (!std::strcmp(reason, memory_srv_reasons[i])) return i;
    return memory_srv_reason_count;  // other
}

// The reasons that depend on the resource description: their shapes are kept. The others (no
// destination, unreadable arguments, no address, a destination outside the view heaps, and the
// acceleration-structure holds, which read only the view's address) do not.
inline bool memory_srv_reason_has_shape(int reason)
{
    return reason < memory_srv_reason_count && reason != memory_srv_reason_index("null_destination") &&
           reason != memory_srv_reason_index("as_unaligned") && reason != memory_srv_reason_index("as_not_placed") &&
           reason != memory_srv_reason_index("as_wrong_state") &&
           reason != memory_srv_reason_index("unreadable_resource") && reason != memory_srv_reason_index("unreadable_view") &&
           reason != memory_srv_reason_index("null_address") && reason != memory_srv_reason_index("destination_outside_view_heaps");
}

struct MemorySrvSwitchCensus {
    using Count = std::atomic<unsigned long long>;
    Count attempts{}, bound{}, bytes{}, not_readable{}, over_limit{}, allocation{}, view_does_not_fit{};
    Count cached_skip{}, cached_rechecks{};
    Count alias_registered{}, alias_resolved{}, alias_unreadable{};
    Count one_d_published{}, one_d_bound{};
    Count one_d_refused[memory_srv_reason_count + 1]{};

    template<class Emit>
    void report(Emit emit, bool only_if_changed)
    {
        char line[4096];
        unsigned long long refused = 0;
        unsigned long long by_reason[memory_srv_reason_count + 1];
        for (int i = 0; i <= memory_srv_reason_count; ++i) {
            by_reason[i] = one_d_refused[i].load(std::memory_order_relaxed);
            refused += by_reason[i];
        }
        const auto a = attempts.load(), b = bound.load(), copied = bytes.load();
        const auto unreadable = not_readable.load(), limit = over_limit.load(), alloc = allocation.load();
        const auto fit = view_does_not_fit.load(), skip = cached_skip.load(), rechecks = cached_rechecks.load();
        const auto registered = alias_registered.load(), resolved = alias_resolved.load(), missed = alias_unreadable.load();
        const auto published = one_d_published.load(), views = one_d_bound.load();
        const auto total = a + b + copied + unreadable + limit + alloc + fit + skip + rechecks +
                           registered + resolved + missed + published + views + refused;
        if (!total || (only_if_changed && last_reported_.exchange(total) == total)) return;
        int n = std::snprintf(line, sizeof(line),
            "x2w-d3d12 memory SRV switches: snapshot attempts %llu bound %llu bytes %llu "
            "not_readable %llu over_limit %llu allocation %llu view_does_not_fit %llu cached_skip %llu cached_rechecks %llu; "
            "alias registered %llu resolved %llu unreadable %llu (texture_unreadable: CPU source range); "
            "1D published %llu bound %llu refused %llu",
            a, b, copied, unreadable, limit, alloc, fit, skip, rechecks, registered, resolved, missed, published, views, refused);
        for (int i = 0; i <= memory_srv_reason_count; ++i) {
            if (!by_reason[i]) continue;
            n += std::snprintf(line + n, sizeof(line) - n, " %s=%llu",
                               i == memory_srv_reason_count ? "other" : memory_srv_reasons[i], by_reason[i]);
        }
        emit(line);
    }

private:
    Count last_reported_{};
};

struct RefusedShape {
    int reason;
    std::uint32_t dimension, format;
    std::uint64_t width;
    std::uint32_t height, depth, mips, layout, flags;
    std::uint64_t calls;
    std::uint64_t reported_calls;  // calls at the last periodic report
};

class MemorySrvCensus {
public:
    // One reason cannot use the whole table: a reason with many shapes (buffer sizes) used to fill it and hide the
    // shapes of every reason that first appeared later. Past its share a reason's new shapes are counted as "more".
    static constexpr unsigned max_shapes_per_reason = 16;
    static constexpr unsigned max_shapes = (memory_srv_reason_count + 1) * max_shapes_per_reason;  // later distinct shapes are counted as "more"
    static constexpr unsigned detail_shapes = 8;

    // Counts one refusal. True when the call is the first of one of the first detail_shapes distinct
    // shapes of its reason: the caller then logs its detail line even when its own line limit is spent.
    bool note(const char* reason_name, const D3D12_RESOURCE_DESC* resource)
    {
        const int reason = memory_srv_reason_index(reason_name);
        bool detail = false;
        AcquireSRWLockExclusive(&lock_);
        ++reasons_[reason];
        ++total_;
        if (resource && memory_srv_reason_has_shape(reason)) {
            const RefusedShape key{reason, static_cast<std::uint32_t>(resource->Dimension), static_cast<std::uint32_t>(resource->Format),
                                   resource->Width, resource->Height, resource->DepthOrArraySize, resource->MipLevels,
                                   static_cast<std::uint32_t>(resource->Layout), static_cast<std::uint32_t>(resource->Flags), 0, 0};
            unsigned i = 0;
            while (i < shape_count_ && !same(shapes_[i], key)) ++i;
            if (i < shape_count_) {
                ++shapes_[i].calls;
            } else if (shape_count_ < max_shapes && shapes_by_reason_[reason] < max_shapes_per_reason) {
                shapes_[shape_count_] = key;
                shapes_[shape_count_].calls = 1;
                detail = shapes_by_reason_[reason] < detail_shapes;
                ++shapes_by_reason_[reason];
                ++shape_count_;
            } else {
                ++more_shape_calls_;
            }
        }
        ReleaseSRWLockExclusive(&lock_);
        return detail;
    }

    template <class Emit>
    bool report(Emit emit, bool only_if_changed)
    {
        AcquireSRWLockExclusive(&lock_);
        const bool changed = total_ != reported_;
        if (only_if_changed) reported_ = total_;  // a forced report leaves the periodic one alone
        if (only_if_changed && !changed) {
            ReleaseSRWLockExclusive(&lock_);
            return false;
        }
        char line[1024];
        int n = std::snprintf(line, sizeof(line), "x2w-d3d12 memory SRV refusals (slot 70): total %llu;",
                              static_cast<unsigned long long>(total_));
        for (int r = 0; r <= memory_srv_reason_count && n > 0 && n < static_cast<int>(sizeof(line)); ++r)
            n += std::snprintf(line + n, sizeof(line) - n, "%s %s %llu", r ? "," : "",
                               r < memory_srv_reason_count ? memory_srv_reasons[r] : "other",
                               static_cast<unsigned long long>(reasons_[r]));
        if (n > 0 && n < static_cast<int>(sizeof(line)))
            std::snprintf(line + n, sizeof(line) - n, "; distinct shapes %u (calls past the shape table %llu)", shape_count_,
                          static_cast<unsigned long long>(more_shape_calls_));
        emit(static_cast<const char*>(line));
        // The periodic report lists only the shapes counted since it last ran; a forced report lists all.
        for (unsigned i = 0; i < shape_count_; ++i) {
            RefusedShape& s = shapes_[i];
            if (only_if_changed && s.calls == s.reported_calls) continue;
            if (only_if_changed) s.reported_calls = s.calls;
            std::snprintf(line, sizeof(line),
                          "x2w-d3d12 memory SRV refused shape %u: reason=%s dim=%u fmt=%u size=%llux%ux%u mips=%u layout=0x%x flags=0x%x calls=%llu",
                          i + 1, s.reason < memory_srv_reason_count ? memory_srv_reasons[s.reason] : "other", s.dimension, s.format,
                          static_cast<unsigned long long>(s.width), s.height, s.depth, s.mips, s.layout, s.flags,
                          static_cast<unsigned long long>(s.calls));
            emit(static_cast<const char*>(line));
        }
        ReleaseSRWLockExclusive(&lock_);
        return true;
    }

private:
    static bool same(const RefusedShape& a, const RefusedShape& b)
    {
        return a.reason == b.reason && a.dimension == b.dimension && a.format == b.format && a.width == b.width &&
               a.height == b.height && a.depth == b.depth && a.mips == b.mips && a.layout == b.layout && a.flags == b.flags;
    }
    SRWLOCK lock_ = SRWLOCK_INIT;
    std::uint64_t reasons_[memory_srv_reason_count + 1] = {};
    std::uint64_t total_ = 0, reported_ = 0, more_shape_calls_ = 0;
    RefusedShape shapes_[max_shapes] = {};
    unsigned shape_count_ = 0;
    unsigned shapes_by_reason_[memory_srv_reason_count + 1] = {};
};

class TokenNullCensus {
public:
    static constexpr unsigned max_pipelines = 16;  // later pipelines are counted as "other"
    static constexpr unsigned detail_pipelines = 4;

    // Counts one texture SRV bound null for want of a view token. True for the first such binding of
    // one of the first detail_pipelines pipelines: the caller then logs the entry's words.
    bool note(std::uint64_t pipeline)
    {
        bool detail = false;
        AcquireSRWLockExclusive(&lock_);
        ++total_;
        unsigned i = 0;
        while (i < count_ && pipelines_[i].key != pipeline) ++i;
        if (i < count_) {
            ++pipelines_[i].nulls;
        } else if (count_ < max_pipelines) {
            pipelines_[count_] = {pipeline, 1};
            detail = count_ < detail_pipelines;
            ++count_;
        } else {
            ++other_;
        }
        ReleaseSRWLockExclusive(&lock_);
        return detail;
    }

    template <class Emit>
    bool report(Emit emit, bool only_if_changed)
    {
        AcquireSRWLockExclusive(&lock_);
        const bool changed = total_ != reported_;
        if (only_if_changed) reported_ = total_;  // a forced report leaves the periodic one alone
        if (only_if_changed && !changed) {
            ReleaseSRWLockExclusive(&lock_);
            return false;
        }
        char line[1024];
        int n = std::snprintf(line, sizeof(line), "x2w-draw token texture nulls: total %llu in %u pipelines;",
                              static_cast<unsigned long long>(total_), count_);
        for (unsigned i = 0; i < count_ && n > 0 && n < static_cast<int>(sizeof(line)); ++i)
            n += std::snprintf(line + n, sizeof(line) - n, " %016llx=%llu", static_cast<unsigned long long>(pipelines_[i].key),
                               static_cast<unsigned long long>(pipelines_[i].nulls));
        if (n > 0 && n < static_cast<int>(sizeof(line)))
            std::snprintf(line + n, sizeof(line) - n, " other=%llu", static_cast<unsigned long long>(other_));
        emit(static_cast<const char*>(line));
        ReleaseSRWLockExclusive(&lock_);
        return true;
    }

private:
    struct Pipeline { std::uint64_t key, nulls; };
    SRWLOCK lock_ = SRWLOCK_INIT;
    Pipeline pipelines_[max_pipelines] = {};
    unsigned count_ = 0;
    std::uint64_t total_ = 0, reported_ = 0, other_ = 0;
};

class MemorySrvWrites {
public:
    static constexpr std::uint64_t detail_entries = 8;

    // Counts one successful slot 70 write. Returns the 1-based ordinal of the destination among the
    // distinct destinations when it is new and within the first detail_entries, else 0.
    std::uint64_t note(std::uint64_t destination)
    {
        std::uint64_t ordinal = 0;
        AcquireSRWLockExclusive(&lock_);
        ++writes_;
        bool fresh = false;
        try {
            fresh = entries_.insert(destination).second;
        } catch (...) {
            ++untracked_;
        }
        if (fresh && entries_.size() <= detail_entries) ordinal = entries_.size();
        ReleaseSRWLockExclusive(&lock_);
        return ordinal;
    }

    template <class Emit>
    bool report(Emit emit, bool only_if_changed)
    {
        AcquireSRWLockExclusive(&lock_);
        const bool changed = writes_ != reported_;
        if (only_if_changed) reported_ = writes_;
        if (only_if_changed && !changed) {
            ReleaseSRWLockExclusive(&lock_);
            return false;
        }
        char line[256];
        std::snprintf(line, sizeof(line), "x2w-d3d12 memory SRV writes (slot 70): calls %llu, distinct entries %llu (untracked %llu)",
                      static_cast<unsigned long long>(writes_), static_cast<unsigned long long>(entries_.size()),
                      static_cast<unsigned long long>(untracked_));
        emit(static_cast<const char*>(line));
        ReleaseSRWLockExclusive(&lock_);
        return true;
    }

private:
    SRWLOCK lock_ = SRWLOCK_INIT;
    std::unordered_set<std::uint64_t> entries_;
    std::uint64_t writes_ = 0, reported_ = 0, untracked_ = 0;
};

// The detail line for a "token" texture null: zeros mean nothing wrote the entry, other words may be
// a descriptor the title wrote itself.
inline int report_token_entry(char* out, std::size_t size, std::uint64_t pipeline, unsigned srv, std::uint64_t entry,
                              const std::uint32_t (&words)[8])
{
    return std::snprintf(out, size,
                         "x2w-draw token texture null: pipeline %016llx srv %u entry 0x%llx words %08x %08x %08x %08x %08x %08x %08x %08x",
                         static_cast<unsigned long long>(pipeline), srv, static_cast<unsigned long long>(entry), words[0], words[1],
                         words[2], words[3], words[4], words[5], words[6], words[7]);
}

}  // namespace x2w_startup
