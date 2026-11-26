#pragma once

#include <windows.h>

#include <cstdint>
#include <unordered_map>

namespace x2w_local {

// A compute pipeline whose translation is absent or unusable refuses every dispatch for the same reason until its manifest is
// retried. Deciding that took draw_lock (a flood of 45,000 such dispatches held it for 13 s). The refusal is remembered here, under
// a lock of its own, so a repeat costs a shared lookup and never touches draw_lock.
//   reason        what the caller returns; a string with static lifetime
//   kind          which counter the refusal feeds (the caller's own numbering)
//   retry_at_ms   0: until the entry is forgotten; otherwise the tick count at which the owner looks again (a miss from then on)
class RefusalCache {
public:
    struct Entry {
        const char* reason = nullptr;
        int kind = 0;
        std::uint64_t retry_at_ms = 0;
        std::uint64_t hits = 0;  // answered from the cache since the owner last took them
    };

    // True (and the entry's reason and kind) when `key` is refused now. Counts the hit.
    bool hit(std::uint64_t key, std::uint64_t now_ms, const char** reason, int* kind)
    {
        AcquireSRWLockShared(&lock_);
        const auto found = entries_.find(key);
        const bool refused = found != entries_.end() && (!found->second.retry_at_ms || now_ms < found->second.retry_at_ms);
        if (refused) {
            *reason = found->second.reason;
            *kind = found->second.kind;
            InterlockedIncrement64(reinterpret_cast<volatile LONG64*>(&found->second.hits));
        }
        ReleaseSRWLockShared(&lock_);
        return refused;
    }
    void put(std::uint64_t key, const char* reason, int kind, std::uint64_t retry_at_ms)
    {
        AcquireSRWLockExclusive(&lock_);
        Entry& entry = entries_[key];
        entry.reason = reason;
        entry.kind = kind;
        entry.retry_at_ms = retry_at_ms;
        ReleaseSRWLockExclusive(&lock_);
    }
    void forget(std::uint64_t key)
    {
        AcquireSRWLockExclusive(&lock_);
        entries_.erase(key);
        ReleaseSRWLockExclusive(&lock_);
    }
    // The hits answered for `key` since the last call (the owner folds them into its own per-key dispatch count).
    std::uint64_t take_hits(std::uint64_t key)
    {
        AcquireSRWLockExclusive(&lock_);
        std::uint64_t hits = 0;
        const auto found = entries_.find(key);
        if (found != entries_.end()) {
            hits = found->second.hits;
            found->second.hits = 0;
        }
        ReleaseSRWLockExclusive(&lock_);
        return hits;
    }

private:
    SRWLOCK lock_ = SRWLOCK_INIT;
    std::unordered_map<std::uint64_t, Entry> entries_;
};

}  // namespace x2w_local
