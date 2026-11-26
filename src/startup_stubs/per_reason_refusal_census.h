// Experimental (docs/startup-runtime.md): lock-protected, bounded per-(key, reason) refusal census.
//
// Records every distinct (key, reason) refusal pair with its own count so a later refusal on the
// same pipeline key does not overwrite or hide earlier refusal reasons. Memory is bounded by
// max_entries; calls past the table continue to advance total_ and overflow_calls_.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace x2w_startup {

class PerReasonRefusalCensus {
public:
    static constexpr unsigned max_entries = 1024;
    static constexpr unsigned report_entries = 64;
    static constexpr unsigned detail_entries = 8;
    static constexpr std::size_t max_label_bytes = 32;
    static constexpr std::size_t max_reason_bytes = 64;

    explicit PerReasonRefusalCensus(const char* label = "per-reason")
    {
        copy_sanitized(label_, sizeof(label_), label, "per-reason");
    }

    // Counts one refusal for (key, reason_name). Returns true on the first insertion of one of
    // the first detail_entries distinct (key, reason) pairs so callers can budget detail logs.
    bool note(std::uint64_t key, const char* reason_name)
    {
        char reason[max_reason_bytes] = {};
        copy_sanitized(reason, sizeof(reason), reason_name, "unknown");

        std::lock_guard<std::mutex> lock(mutex_);
        ++total_;
        // Linear scan over max_entries under mutex_ is sized for refusal paths, not a per-draw hot path.
        for (unsigned i = 0; i < entry_count_; ++i) {
#if defined(X2W_CENSUS_KEEP_LAST) || defined(X2W_CENSUS_KEEP_FIRST)
            if (entries_[i].key == key) {
                if (std::strcmp(entries_[i].reason, reason) == 0) {
                    ++entries_[i].calls;
#ifdef X2W_CENSUS_KEEP_LAST
                } else {
                    std::memcpy(entries_[i].reason, reason, sizeof(reason));
                    entries_[i].calls = 1;
#endif
                }
                return false;
            }
#else
            if (entries_[i].key == key && std::strcmp(entries_[i].reason, reason) == 0) {
                ++entries_[i].calls;
                return false;
            }
#endif
        }
        if (entry_count_ >= max_entries) {
            ++overflow_calls_;
            return false;
        }
        if (!has_key_locked(key)) ++distinct_keys_;
        Entry& entry = entries_[entry_count_];
        entry.key = key;
        std::memcpy(entry.reason, reason, sizeof(reason));
        entry.calls = 1;
        entry.reported_calls = 0;
        const bool detail = entry_count_ < detail_entries;
        ++entry_count_;
        return detail;
    }

    template <class Emit>
    bool report(Emit emit, bool only_if_changed)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool changed = total_ != reported_;
        if (only_if_changed) reported_ = total_;  // a forced report leaves periodic deltas intact
        if (only_if_changed && !changed) return false;

        char line[256];
        std::snprintf(line, sizeof(line),
                      "x2w-d3d12 %s refusals: total %llu; distinct keys %u, distinct pairs %u (overflow calls %llu)",
                      label_, static_cast<unsigned long long>(total_), distinct_keys_, entry_count_,
                      static_cast<unsigned long long>(overflow_calls_));
        emit(static_cast<const char*>(line));

        unsigned ranked[max_entries];
        unsigned count = 0;
        for (unsigned i = 0; i < entry_count_; ++i) {
            if (!only_if_changed || entries_[i].calls != entries_[i].reported_calls) ranked[count++] = i;
        }
        const unsigned shown = (std::min)(count, report_entries);
        std::partial_sort(ranked, ranked + shown, ranked + count, [&](unsigned a, unsigned b) {
            if (entries_[a].calls != entries_[b].calls) return entries_[a].calls > entries_[b].calls;
            return a < b;
        });
        for (unsigned rank = 0; rank < shown; ++rank) {
            const unsigned i = ranked[rank];
            Entry& entry = entries_[i];
            if (only_if_changed) entry.reported_calls = entry.calls;
            std::snprintf(line, sizeof(line),
                          "x2w-d3d12 %s refusal %u: key=%016llx reason=%s calls=%llu",
                          label_, i + 1, static_cast<unsigned long long>(entry.key), entry.reason,
                          static_cast<unsigned long long>(entry.calls));
            emit(static_cast<const char*>(line));
        }
        if (count > shown) {
            std::uint64_t omitted_calls = 0;
            for (unsigned rank = shown; rank < count; ++rank) {
                Entry& entry = entries_[ranked[rank]];
                omitted_calls += entry.calls;
                if (only_if_changed) entry.reported_calls = entry.calls;
            }
            std::snprintf(line, sizeof(line), "x2w-d3d12 %s refusals omitted: %u pairs, %llu calls",
                          label_, count - shown, static_cast<unsigned long long>(omitted_calls));
            emit(static_cast<const char*>(line));
        }
        return true;
    }

    std::uint64_t total() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_;
    }

    unsigned distinct_keys() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return distinct_keys_;
    }

    unsigned entry_count() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return entry_count_;
    }

    std::uint64_t overflow_calls() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return overflow_calls_;
    }

private:
    struct Entry {
        std::uint64_t key = 0;
        char reason[max_reason_bytes] = {};
        std::uint64_t calls = 0;
        std::uint64_t reported_calls = 0;
    };

    static void copy_sanitized(char* out, std::size_t out_size, const char* src, const char* fallback)
    {
        const char* input = (src && src[0] != '\0') ? src : fallback;
        std::size_t i = 0;
        for (; i + 1 < out_size && input[i] != '\0'; ++i) {
            const char c = input[i];
            out[i] = (c == '\r' || c == '\n') ? ' ' : c;
        }
        out[i] = '\0';
    }

    bool has_key_locked(std::uint64_t key) const
    {
        for (unsigned i = 0; i < entry_count_; ++i) {
            if (entries_[i].key == key) return true;
        }
        return false;
    }

    mutable std::mutex mutex_;
    char label_[max_label_bytes] = {};
    Entry entries_[max_entries] = {};
    unsigned entry_count_ = 0;
    unsigned distinct_keys_ = 0;
    std::uint64_t total_ = 0;
    std::uint64_t reported_ = 0;
    std::uint64_t overflow_calls_ = 0;
};

}  // namespace x2w_startup
