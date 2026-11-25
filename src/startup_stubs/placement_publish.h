#pragma once

#include <windows.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>

namespace x2w_startup {
class ExclusiveNoteLock {
public:
    explicit ExclusiveNoteLock(SRWLOCK& lock) noexcept : lock_(lock) { AcquireSRWLockExclusive(&lock_); }
    ~ExclusiveNoteLock() { ReleaseSRWLockExclusive(&lock_); }
    ExclusiveNoteLock(const ExclusiveNoteLock&) = delete;
    ExclusiveNoteLock& operator=(const ExclusiveNoteLock&) = delete;
private:
    SRWLOCK& lock_;
};

// Insert under the same lock used by binding lookup. The index is published
// only after the canonical resource record exists. Erase the new record if its
// index node cannot be allocated. Existing registrations are never replaced. `also` runs under the
// same lock hold once the record and the index node exist (a throw undoes both), so a second index
// never lags the first.
struct NoExtraIndex { void operator()() const {} };
template<class Records, class Key, class Record, class Also = NoExtraIndex>
HRESULT publish_placement(SRWLOCK& lock, Records get_records, std::multimap<std::uint64_t, Key>& by_start,
                          std::uint64_t& max_width, Key owner, std::uint64_t start, const Record& record, Also also = {})
{
    if (!start || !record.bytes || record.bytes > (std::numeric_limits<std::uint64_t>::max)() - start)
        return E_INVALIDARG;
    ExclusiveNoteLock guard(lock);
    try {
        auto& records = get_records();
        const auto [entry, added] = records.try_emplace(owner, record);
        if (!added) return HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS);
        try {
            const auto node = by_start.emplace(start, owner);
            try {
                also();
            } catch (...) {
                by_start.erase(node);
                throw;
            }
        } catch (...) {
            records.erase(entry);
            throw;
        }
        max_width = (std::max)(max_width, record.bytes);
        return S_OK;
    } catch (const std::bad_alloc&) {
        return E_OUTOFMEMORY;
    } catch (const std::length_error&) {
        return E_OUTOFMEMORY;
    }
}
} // namespace x2w_startup
