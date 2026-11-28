// Experimental (docs/startup-runtime.md): bounded counters for the optional Opus decode path.
//
// Every counter saturates at UINT64_MAX instead of wrapping. The values are process-wide and are
// read without a single coherent snapshot: a summary taken while another thread enqueues can show
// one field from before that enqueue and another from after it. Each field is exact on its own.
#pragma once

#include <atomic>
#include <cstdint>
#include <limits>

namespace x2w_audio {

inline void saturating_add(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) noexcept
{
    std::uint64_t seen = counter.load(std::memory_order_relaxed);
    for (;;) {
        const std::uint64_t next = amount > (std::numeric_limits<std::uint64_t>::max)() - seen
            ? (std::numeric_limits<std::uint64_t>::max)()
            : seen + amount;
        if (next == seen || counter.compare_exchange_weak(seen, next, std::memory_order_relaxed,
                                                          std::memory_order_relaxed))
            return;
    }
}

// Counters for one process's type-3 Opus path. `type3_commands` counts every readable type-3
// command while typed completions are on. `packets_decoded` counts calls libopus accepted with a
// non-empty packet. `decoded_completions` counts type-3 results dequeued by the caller, so a packet
// that decodes but is never consumed is not in it. The refusal fields name the first check that
// refused the command; `header` covers a null or unreadable XApuEnqueueCommand header.
struct OpusCounters {
    std::atomic<std::uint64_t> type3_commands{0};
    std::atomic<std::uint64_t> packets_decoded{0};
    std::atomic<std::uint64_t> decoded_completions{0};
    std::atomic<std::uint64_t> refused_header{0};
    std::atomic<std::uint64_t> refused_short_command{0};
    std::atomic<std::uint64_t> refused_unknown_handle{0};
    std::atomic<std::uint64_t> refused_queue_full{0};
    std::atomic<std::uint64_t> refused_invalid_argument{0};
    std::atomic<std::uint64_t> refused_opus_error{0};
    std::atomic<std::uint64_t> refused_capacity{0};
    std::atomic<std::uint64_t> refused_out_of_memory{0};
};

struct OpusCounterSnapshot {
    std::uint64_t type3_commands = 0;
    std::uint64_t packets_decoded = 0;
    std::uint64_t decoded_completions = 0;
    std::uint64_t refused_header = 0;
    std::uint64_t refused_short_command = 0;
    std::uint64_t refused_unknown_handle = 0;
    std::uint64_t refused_queue_full = 0;
    std::uint64_t refused_invalid_argument = 0;
    std::uint64_t refused_opus_error = 0;
    std::uint64_t refused_capacity = 0;
    std::uint64_t refused_out_of_memory = 0;
};

inline OpusCounterSnapshot snapshot(const OpusCounters& counters) noexcept
{
    return {counters.type3_commands.load(std::memory_order_relaxed),
            counters.packets_decoded.load(std::memory_order_relaxed),
            counters.decoded_completions.load(std::memory_order_relaxed),
            counters.refused_header.load(std::memory_order_relaxed),
            counters.refused_short_command.load(std::memory_order_relaxed),
            counters.refused_unknown_handle.load(std::memory_order_relaxed),
            counters.refused_queue_full.load(std::memory_order_relaxed),
            counters.refused_invalid_argument.load(std::memory_order_relaxed),
            counters.refused_opus_error.load(std::memory_order_relaxed),
            counters.refused_capacity.load(std::memory_order_relaxed),
            counters.refused_out_of_memory.load(std::memory_order_relaxed)};
}

inline std::uint64_t saturating_total(std::uint64_t& total, std::uint64_t value) noexcept
{
    total = value > (std::numeric_limits<std::uint64_t>::max)() - total
        ? (std::numeric_limits<std::uint64_t>::max)()
        : total + value;
    return total;
}

inline std::uint64_t refused_total(const OpusCounterSnapshot& s) noexcept
{
    std::uint64_t total = 0;
    saturating_total(total, s.refused_header);
    saturating_total(total, s.refused_short_command);
    saturating_total(total, s.refused_unknown_handle);
    saturating_total(total, s.refused_queue_full);
    saturating_total(total, s.refused_invalid_argument);
    saturating_total(total, s.refused_opus_error);
    saturating_total(total, s.refused_capacity);
    saturating_total(total, s.refused_out_of_memory);
    return total;
}

}  // namespace x2w_audio
