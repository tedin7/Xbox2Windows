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

// What the draw and dispatch paths use, with draw_lock held. The FIRST lookup of a key is one synchronous open, as it always
// was, so a translation that is already published draws on its first use. RETRIES of a key that was absent never open
// the file: they take what the background thread has read. The two kinds of open done under the lock are counted
// separately; `retry_opens_under_lock` must stay 0 (it is what made the lock busy: every absent key re-opened its
// file once a second).
class ManifestSource {
public:
    explicit ManifestSource(ManifestPrefetch::Io io) : io_(io), prefetch_(std::move(io)) {}

    // Under draw_lock. retry: the key was looked up before and was absent.
    bool load(std::uint64_t key, bool retry, std::string& out)
    {
        if (retry) return prefetch_.fetch(key, out);
        {
            // An open of a key that was already opened once is a retry that opened the file under the lock.
            std::lock_guard<std::mutex> lock(opened_mutex_);
            if (opened_.insert(key).second) ++first_opens_under_lock_;
            else ++retry_opens_under_lock_;
        }
        return io_.read(key, out);
    }

    ManifestPrefetch& prefetch() { return prefetch_; }
    long long first_opens_under_lock() const { return first_opens_under_lock_; }
    long long retry_opens_under_lock() const { return retry_opens_under_lock_; }
    long long background_reads() const { return prefetch_.background_reads(); }

protected:
    ManifestPrefetch::Io io_;
    ManifestPrefetch prefetch_;
    std::atomic<long long> first_opens_under_lock_{0};
    std::atomic<long long> retry_opens_under_lock_{0};
    std::mutex opened_mutex_;
    std::unordered_set<std::uint64_t> opened_;
};

}  // namespace x2w_startup
