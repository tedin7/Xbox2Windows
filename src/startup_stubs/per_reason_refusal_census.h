// Experimental (docs/startup-runtime.md): lock-protected, bounded per-(key, reason) refusal census.
//
// Records every distinct (key, reason) refusal pair with its own count so a later refusal on the
// same pipeline key does not overwrite or hide earlier refusal reasons. Memory is bounded by
// max_entries; calls past the table continue to advance total_ and overflow_calls_.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace x2w_startup {

class PerReasonRefusalCensus {
public:
    static constexpr unsigned max_entries = 128;
    static constexpr unsigned detail_entries = 8;
    static constexpr std::size_t max_label_bytes = 32;
    static constexpr std::size_t max_reason_bytes = 64;

    using FormatCode = void (*)(unsigned, char*, std::size_t);

    explicit PerReasonRefusalCensus(const char* label = "per-reason", FormatCode format_code = nullptr)
        : format_code_(format_code)
    {
        copy_sanitized(label_, sizeof(label_), label, "per-reason");
    }

    // Counts one refusal for (key, reason_name). Returns true on the first insertion of one of
    // the first detail_entries distinct (key, reason) pairs so callers can budget detail logs.
    bool note(std::uint64_t key, const char* reason_name)
    {
        char reason[max_reason_bytes] = {};
        copy_sanitized(reason, sizeof(reason), reason_name, "unknown");

        return note_entry(key, reason, 0);
    }

    bool note_code(std::uint64_t key, unsigned code)
    {
        return note_entry(key, nullptr, code);
    }

private:
    bool note_entry(std::uint64_t key, const char* reason, unsigned code)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++total_;
        // Linear scan over max_entries under mutex_ is sized for refusal paths, not a per-draw hot path.
        for (unsigned i = 0; i < entry_count_; ++i) {
#if defined(X2W_CENSUS_KEEP_LAST) || defined(X2W_CENSUS_KEEP_FIRST)
            if (entries_[i].key == key) {
                if (entries_[i].code == code && (code || std::strcmp(entries_[i].reason, reason) == 0)) {
                    ++entries_[i].calls;
#ifdef X2W_CENSUS_KEEP_LAST
                } else {
                    if (!code) std::memcpy(entries_[i].reason, reason, sizeof(entries_[i].reason));
                    entries_[i].code = code;
                    entries_[i].calls = 1;
#endif
                }
                return false;
            }
#else
            if (entries_[i].key == key && entries_[i].code == code && (code || std::strcmp(entries_[i].reason, reason) == 0)) {
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
        entry.code = code;
        if (!code) std::memcpy(entry.reason, reason, sizeof(entry.reason));
        entry.calls = 1;
        entry.reported_calls = 0;
        const bool detail = entry_count_ < detail_entries;
        ++entry_count_;
        return detail;
    }

public:
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

        for (unsigned i = 0; i < entry_count_; ++i) {
            Entry& entry = entries_[i];
            if (only_if_changed && entry.calls == entry.reported_calls) continue;
            if (only_if_changed) entry.reported_calls = entry.calls;
            char reason[max_reason_bytes];
            if (entry.code) format_code_(entry.code, reason, sizeof(reason));
            std::snprintf(line, sizeof(line),
                          "x2w-d3d12 %s refusal %u: key=%016llx reason=%s calls=%llu",
                          label_, i + 1, static_cast<unsigned long long>(entry.key), entry.code ? reason : entry.reason,
                          static_cast<unsigned long long>(entry.calls));
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
        unsigned code = 0;
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

    FormatCode format_code_;
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
