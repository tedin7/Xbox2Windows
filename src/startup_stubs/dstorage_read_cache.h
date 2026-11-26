// HDD mode of the DirectStorage stand-in (X2W_STORAGE_HDD=1): a prefetch thread reads the file ranges the next queued requests
// will need, sorted by file and offset and merged into few large reads (dstorage_read_plan.h), into a bounded cache. The queue's
// worker still runs its entries in order; before a file read it asks the cache, and on a miss it reads the file as it always did.
// So the cache changes where the bytes come from, never what the requests do.
#pragma once

#include <windows.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include "startup_stubs/dstorage_file_read.h"
#include "startup_stubs/dstorage_read_plan.h"

namespace x2w_startup {

class ReadCache {
public:
    struct Counters {
        LONG hits = 0, misses = 0, waited = 0, failed = 0;  // take(): served, not planned or failed, served after waiting; groups whose read failed
        LONG groups = 0, reads_saved = 0, evicted_unused = 0;
        LONG64 prefetched_bytes = 0;
    };
    // Called after every prefetch read with the file, offset, bytes, QueryPerformanceCounter ticks it took and whether it succeeded.
    using ReadHook = void (*)(HANDLE, unsigned long long, unsigned long long, long long, bool);

    ReadCache(ReadHook hook, std::uint64_t gap, std::uint64_t max_span, std::uint64_t max_bytes, DWORD stale_ms)
        : hook_(hook), gap_(gap), max_span_(max_span), max_bytes_(max_bytes), stale_ms_(stale_ms)
    {
    }

    // Register the reads and queue their merged groups for the prefetch thread. Reads that would push the cache past its
    // byte limit, or that are already registered, are left to the worker's own reads.
    void submit(const std::vector<PlannedRead>& reads)
    {
        std::vector<PlannedRead> fresh;
        {
            const std::lock_guard guard(mutex_);
            evict_stale_locked();
            for (const PlannedRead& r : reads) {
                const Key key{r.file, r.offset, r.size};
                if (r.size == 0 || slots_.count(key) || held_ + r.size > max_bytes_) continue;
                slots_[key] = Slot{{}, false, false, GetTickCount()};
                held_ += r.size;
                fresh.push_back(r);
            }
        }
        if (fresh.empty()) return;
        std::vector<ReadGroup> groups = plan_reads(std::move(fresh), gap_, max_span_);
        {
            const std::lock_guard guard(mutex_);
            counters_.groups += static_cast<LONG>(groups.size());
            counters_.reads_saved += static_cast<LONG>(reads_saved(groups));
            for (ReadGroup& g : groups) pending_.push_back(std::move(g));
            if (!thread_) thread_ = CreateThread(nullptr, 0, &ReadCache::run, this, 0, nullptr);
        }
        wake_.notify_all();
    }

    // The bytes of a planned read: moved into `out` when the prefetch has them (waiting up to wait_ms for a read in progress).
    // False when the read was not planned or its prefetch failed; the caller reads the file itself.
    bool take(std::uintptr_t file, std::uint64_t offset, std::uint32_t size, std::vector<unsigned char>& out, DWORD wait_ms)
    {
        std::unique_lock lock(mutex_);
        const auto found = slots_.find(Key{file, offset, size});
        if (found == slots_.end()) {
            ++counters_.misses;
            return false;
        }
        bool waited = false;
        if (!found->second.ready) {
            waited = true;
            ready_.wait_for(lock, std::chrono::milliseconds(wait_ms), [&] { return found->second.ready; });
        }
        if (!found->second.ready) {
            ++counters_.misses;  // the prefetch is still busy: read it directly, and the late bytes are dropped on arrival
            found->second.abandoned = true;
            return false;
        }
        const bool ok = found->second.ok;
        if (ok) out = std::move(found->second.data);
        held_ -= size;
        slots_.erase(found);
        if (ok) {
            ++counters_.hits;
            if (waited) ++counters_.waited;
        } else {
            ++counters_.misses;
        }
        return ok;
    }

    Counters counters() const
    {
        const std::lock_guard guard(mutex_);
        return counters_;
    }

private:
    struct Key {
        std::uintptr_t file;
        std::uint64_t offset;
        std::uint32_t size;
        bool operator<(const Key& o) const
        {
            if (file != o.file) return file < o.file;
            if (offset != o.offset) return offset < o.offset;
            return size < o.size;
        }
    };
    struct Slot {
        std::vector<unsigned char> data;
        bool ready = false, ok = false;
        DWORD tick = 0;
        bool abandoned = false;
        Slot() = default;
        Slot(std::vector<unsigned char> d, bool r, bool o, DWORD t) : data(std::move(d)), ready(r), ok(o), tick(t) {}
    };

    void evict_stale_locked()
    {
        const DWORD now = GetTickCount();
        for (auto it = slots_.begin(); it != slots_.end();) {
            if (it->second.ready && now - it->second.tick > stale_ms_) {
                held_ -= it->first.size;
                ++counters_.evicted_unused;
                it = slots_.erase(it);
            } else {
                ++it;
            }
        }
    }

    static DWORD WINAPI run(void* self) { static_cast<ReadCache*>(self)->loop(); return 0; }

    void loop()
    {
        for (;;) {
            ReadGroup group;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return !pending_.empty(); });
                group = std::move(pending_.front());
                pending_.pop_front();
            }
            std::vector<unsigned char> buffer;
            LARGE_INTEGER began, ended;
            QueryPerformanceCounter(&began);
            const bool ok = read_storage_file(reinterpret_cast<HANDLE>(group.file), group.offset, static_cast<DWORD>(group.span), buffer);
            QueryPerformanceCounter(&ended);
            if (hook_) hook_(reinterpret_cast<HANDLE>(group.file), group.offset, group.span, ended.QuadPart - began.QuadPart, ok);
            {
                const std::lock_guard guard(mutex_);
                if (!ok) ++counters_.failed;
                else counters_.prefetched_bytes += static_cast<LONG64>(group.span);
                for (const PlannedRead& m : group.members) {
                    const auto it = slots_.find(Key{m.file, m.offset, m.size});
                    if (it == slots_.end()) continue;
                    if (it->second.abandoned) {  // the worker already read it itself
                        held_ -= m.size;
                        slots_.erase(it);
                        continue;
                    }
                    if (ok) {
                        const auto from = buffer.begin() + static_cast<std::ptrdiff_t>(m.offset - group.offset);
                        it->second.data.assign(from, from + m.size);
                    }
                    it->second.ok = ok;
                    it->second.ready = true;
                    it->second.tick = GetTickCount();
                }
            }
            ready_.notify_all();
        }
    }

    ReadHook hook_;
    std::uint64_t gap_, max_span_, max_bytes_;
    DWORD stale_ms_;
    mutable std::mutex mutex_;
    std::condition_variable wake_, ready_;
    std::deque<ReadGroup> pending_;
    std::map<Key, Slot> slots_;
    std::uint64_t held_ = 0;
    Counters counters_;
    HANDLE thread_ = nullptr;
};

}  // namespace x2w_startup
