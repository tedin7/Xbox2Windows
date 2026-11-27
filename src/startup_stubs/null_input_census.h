#pragma once

#include "null_binding_cause.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace x2w_startup {

inline void format_null_binding_totals(char* out, std::size_t size, long total, const long* counts)
{
    std::snprintf(out, size, "null bindings %ld (entry %ld, buffer %ld, token %ld, wrapper %ld, dimension %ld, ring_full %ld, integer_view_filtered %ld)",
                  total, counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6]);
}

struct NullInputCensus {
    struct Input {
        std::uint64_t key;
        unsigned index;
        x2w_local::NullCause cause;
        unsigned kind;
        int rtv, dsv;
        std::uint32_t words[2];
        std::size_t bytes;
        const char* reason;  // Static text from the failed binding branch.
        long first, last;
        std::uint64_t calls = 0, reported = 0;

        bool same_binding(const Input& other, bool omit_index) const
        {
            return key == other.key && (omit_index || index == other.index) && cause == other.cause;
        }
    };

    // omit_index is used only by the negative control.
    void note(Input sample, bool omit_index = false)
    {
        const auto hash = sample.key ^ (sample.key >> 32) ^
                          (std::uint64_t{omit_index ? 0u : sample.index} * 0x9e3779b9u) ^ static_cast<unsigned>(sample.cause);
        for (std::size_t probe = 0; probe < inputs.size(); ++probe) {
            auto& row = inputs[(hash + probe) % inputs.size()];
            if (!row.calls) row = sample;
            else if (!row.same_binding(sample, omit_index)) continue;
            ++row.calls;
            row.last = sample.last;
            return;
        }
        ++input_overflow;
    }

    // With the caller's draw lock held, format the changed rows into a bounded
    // batch. The caller can write the batch to the log after releasing the lock.
    struct Batch {
        std::array<std::array<char, 384>, 26> lines;
        std::size_t count = 0;

        template <typename... Args>
        void add(const char* format, Args... args)
        {
            auto& line = lines[count++];
            std::snprintf(line.data(), line.size(), format, args...);
        }

        template <typename Emit>
        void emit(Emit write) const
        {
            for (std::size_t i = 0; i < count; ++i) write(lines[i].data());
        }
    };

    void report(Batch& batch, bool changed)
    {
        std::array<Input*, 512> ranked{};
        std::size_t count = 0;
        for (auto& row : inputs)
            if (row.calls && (!changed || row.calls != row.reported)) ranked[count++] = &row;
        static constexpr std::size_t limit = 24;
        const auto shown = (std::min)(count, limit);
        std::partial_sort(ranked.begin(), ranked.begin() + shown, ranked.begin() + count,
                          [](const Input* a, const Input* b) {
                              if (a->calls != b->calls) return a->calls > b->calls;
                              if (a->key != b->key) return a->key < b->key;
                              if (a->index != b->index) return a->index < b->index;
                              return a->cause < b->cause;
                          });
        static constexpr const char* kinds[] = {"buffer", "2D", "array", "3D", "1D", "cube"};
        for (std::size_t i = 0; i < shown; ++i) {
            auto& row = *ranked[i];
            char detail[160]{};
            using x2w_local::NullCause;
            if (row.cause == NullCause::unknown_token)
                std::snprintf(detail, sizeof(detail), " words %08x %08x", row.words[0], row.words[1]);
            else if (row.cause == NullCause::buffer_unreadable)
                std::snprintf(detail, sizeof(detail), " bytes %zu reason %s", row.bytes, row.reason ? row.reason : "unspecified");
            else if (row.cause == NullCause::ring_full)
                std::snprintf(detail, sizeof(detail), " bytes %zu", row.bytes);
            batch.add("x2w-render null input: key %016llx input %u (%s) cause %s calls %llu frames %ld..%ld target format %d depth format %d%s",
                      static_cast<unsigned long long>(row.key), row.index, kinds[row.kind],
                      x2w_local::null_cause_names[static_cast<int>(row.cause)],
                      static_cast<unsigned long long>(row.calls), row.first, row.last, row.rtv, row.dsv, detail);
            if (changed) row.reported = row.calls;
        }
        if ((!changed && input_overflow) || input_overflow != reported_overflow)
            batch.add("x2w-render null census overflow: inputs %llu, target draws 0",
                      static_cast<unsigned long long>(input_overflow));
        if (changed) reported_overflow = input_overflow;
    }

private:
    std::array<Input, 512> inputs{};
    std::uint64_t input_overflow = 0, reported_overflow = 0;
};

}  // namespace x2w_startup
