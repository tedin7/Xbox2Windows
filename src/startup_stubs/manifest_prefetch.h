#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace x2w_startup {

// Translation manifests are files in the shader cache directory, named by the pipeline key. The draw and dispatch paths
// ask for them with draw_lock held, so a file open there stalls every other thread behind the lock (a missing file is
// asked for again every second per key, and a debugger or a virus scanner makes each open milliseconds long).
//
// fetch() never touches the file system: it hands out the text of a manifest a background pass has already read, or
// records that the key is wanted and says "not there yet". scan() is the background pass: it lists the directory once,
// reads the manifests of the wanted keys that exist, and keeps their text until fetch() takes it. All file access goes
// through the two functions in Io, so a test can make them slow and count where they run.
class ManifestPrefetch {
public:
    struct Io {
        std::function<std::vector<std::uint64_t>()> list;               // keys that have a manifest right now
        std::function<bool(std::uint64_t, std::string&)> read;          // the manifest text of a key
    };

    explicit ManifestPrefetch(Io io) : io_(std::move(io)) {}

    // Any thread, any lock held: no file access, no waiting beyond the short internal mutex.
    bool fetch(std::uint64_t key, std::string& out)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = ready_.find(key);
        if (found == ready_.end()) {
            wanted_.insert(key);
            return false;
        }
        out = std::move(found->second);
        ready_.erase(found);
        return true;
    }

    // A key whose first lookup found no manifest: the background reader takes it as soon as it appears, so the first retry
    // already finds it (before, only the first retry registered the key, and a second retry was needed).
    void want(std::uint64_t key)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_.count(key)) wanted_.insert(key);
    }

    // True when some key is waiting for its manifest (the background thread has work to do).
    bool has_wanted() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return !wanted_.empty();
    }

    // The background pass. Returns how many manifests it loaded. File access happens with the mutex released.
    std::size_t scan()
    {
        std::vector<std::uint64_t> wanted;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            wanted.assign(wanted_.begin(), wanted_.end());
        }
        if (wanted.empty()) return 0;
        const std::vector<std::uint64_t> present = io_.list();
        const std::unordered_set<std::uint64_t> exists(present.begin(), present.end());
        std::size_t loaded = 0;
        for (const std::uint64_t key : wanted) {
            if (!exists.count(key)) continue;
            std::string text;
            if (!io_.read(key, text)) continue;  // vanished or still being written: stays wanted
            std::lock_guard<std::mutex> lock(mutex_);
            ready_[key] = std::move(text);
            wanted_.erase(key);
            ++loaded;
        }
        background_reads_ += loaded;
        return loaded;
    }

    long long background_reads() const { return background_reads_; }

private:
    std::atomic<long long> background_reads_{0};
    Io io_;
    mutable std::mutex mutex_;
    std::unordered_set<std::uint64_t> wanted_;
    std::unordered_map<std::uint64_t, std::string> ready_;
};

// What the draw and dispatch paths use. The first lookup of a key is claimed by exactly one caller (claim_first), who reads the
// manifest with draw_lock RELEASED (read_first) and holds nothing while the disk answers; every other lookup, and every retry
// of a key that was absent, takes what the background thread has read (load_retry) and opens nothing. Reads are counted: a key
// read twice means some retry opened the file again (retry_opens must stay 0).
class ManifestSource {
public:
    explicit ManifestSource(ManifestPrefetch::Io io) : io_(io), prefetch_(std::move(io)) {}

    // True for the first caller that asks about this key (it must then call read_first, outside any lock).
    bool claim_first(std::uint64_t key)
    {
        std::lock_guard<std::mutex> lock(opened_mutex_);
        return claimed_.insert(key).second;
    }

    // One file read for a claimed key, to be made without draw_lock. A second read of the same key counts as a retry open.
    bool read_first(std::uint64_t key, std::string& out)
    {
        {
            std::lock_guard<std::mutex> lock(opened_mutex_);
            if (read_.insert(key).second) ++first_opens_;
            else ++retry_opens_;
        }
        return io_.read(key, out);
    }

    // Retries and lookups that lost the claim: in-memory only.
    bool load_retry(std::uint64_t key, std::string& out) { return prefetch_.fetch(key, out); }

    ManifestPrefetch& prefetch() { return prefetch_; }
    long long first_opens() const { return first_opens_; }
    long long retry_opens() const { return retry_opens_; }
    long long background_reads() const { return prefetch_.background_reads(); }

private:
    ManifestPrefetch::Io io_;
    ManifestPrefetch prefetch_;
    std::mutex opened_mutex_;
    std::unordered_set<std::uint64_t> claimed_, read_;
    long long first_opens_ = 0, retry_opens_ = 0;
};

}  // namespace x2w_startup
