#pragma once

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
        return loaded;
    }

private:
    Io io_;
    mutable std::mutex mutex_;
    std::unordered_set<std::uint64_t> wanted_;
    std::unordered_map<std::uint64_t, std::string> ready_;
};

}  // namespace x2w_startup
