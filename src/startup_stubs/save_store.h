// Experimental (docs/startup-runtime.md): on-disk store behind the local game-save stand-in.
// One directory per container, one file per blob, all below one root folder. Plain standard C++
// with no Windows types, so the logic can be compiled and tested on its own.
#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "file_lock.h"

namespace x2w_save {

namespace fs = std::filesystem;

enum class Status { ok, bad_name, too_large, not_found, io_error, busy };

constexpr std::size_t max_name_bytes = 128;
constexpr std::size_t max_blob_bytes = std::size_t{64} << 20;
constexpr std::string_view part_suffix = ".x2w-part";  // a file being written
constexpr std::string_view old_suffix = ".x2w-old";    // the file a commit replaced, kept until the commit is whole
constexpr std::string_view lock_suffix = ".x2w-lock";  // beside a container directory: the lock of its publication point
constexpr std::chrono::milliseconds default_lock_timeout{5000};

// A name becomes one path component. Anything that could address another place is refused: empty,
// too long, separators, drive letters and streams (':'), control characters, characters Windows
// does not allow in file names, "..", a trailing dot or space, device names, and the suffix the
// store uses for files it is still writing.
inline bool valid_name(std::string_view name) noexcept
{
    if (name.empty() || name.size() > max_name_bytes) return false;
    for (const unsigned char c : name) {
        if (c < 0x20 || c == 0x7f) return false;
        if (std::string_view("/\\:*?\"<>|").find(static_cast<char>(c)) != std::string_view::npos) return false;
    }
    if (name.find("..") != std::string_view::npos) return false;
    if (name.back() == '.' || name.back() == ' ' || name.front() == ' ') return false;
    // Case-insensitive: on Windows "X.X2W-PART" and "X.x2w-part" are the same file.
    for (const std::string_view suffix : {part_suffix, old_suffix, lock_suffix}) {
        if (name.size() < suffix.size()) continue;
        const std::string_view tail = name.substr(name.size() - suffix.size());
        bool same = true;
        for (std::size_t i = 0; i < tail.size() && same; ++i)
            same = std::tolower(static_cast<unsigned char>(tail[i])) == suffix[i];
        if (same) return false;
    }
    std::string stem(name.substr(0, name.find('.')));
    for (char& c : stem) c = static_cast<char>(c >= 'a' && c <= 'z' ? c - 'a' + 'A' : c);
    while (!stem.empty() && stem.back() == ' ') stem.pop_back();
    for (const char* device : {"CON", "PRN", "AUX", "NUL"})
        if (stem == device) return false;
    if (stem.size() == 4 && (stem.compare(0, 3, "COM") == 0 || stem.compare(0, 3, "LPT") == 0) && stem[3] >= '0' && stem[3] <= '9') return false;
    return true;
}

// A file name nobody can predict or share: process-wide counter plus a clock reading, then a
// reserved suffix (valid_name refuses names that end in it).
inline std::string unique_name(std::string_view suffix)
{
    static std::atomic<unsigned long long> counter{0};
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return "." + std::to_string(counter.fetch_add(1) + 1) + "-" + std::to_string(static_cast<long long>(ticks)) + std::string(suffix);
}

// A symbolic link, or on Windows a junction (both redirect a path somewhere else). Only the
// storage root itself is trusted; a container directory or blob that redirects is refused.
inline bool redirected(const fs::file_status& state) noexcept
{
#ifdef _MSC_VER
    if (state.type() == fs::file_type::junction) return true;
#endif
    return fs::is_symlink(state);
}

// One action of an update, staged in memory until it is submitted.
struct Action {
    bool remove = false;
    std::string blob;
    std::vector<unsigned char> data;
};
constexpr std::size_t max_staged_bytes = std::size_t{256} << 20;  // per update

// Container handles of the game-save stand-in: a fixed table indexed by the low bits of the handle,
// with a reference count per entry so closed containers give their slot back. The caller holds the
// lock; no Windows types here so it can be tested on its own.
class ContainerTable {
public:
    static constexpr std::size_t capacity = 64;
    // Index of the container with this name (opened again or new), or -1 when the table is full.
    long open(const std::string& name)
    {
        for (std::size_t i = 0; i < capacity; ++i)
            if (names_[i] == name && !name.empty()) {
                ++refs_[i];
                return static_cast<long>(i);
            }
        for (std::size_t i = 0; i < capacity; ++i)
            if (names_[i].empty()) {
                names_[i] = name;
                refs_[i] = 1;
                return static_cast<long>(i);
            }
        return -1;
    }
    void close(std::size_t index)
    {
        if (index >= capacity) return;
        if (refs_[index] > 0) --refs_[index];
        reclaim(index);
    }
    // Index of the update slot, or -1 (container not open, or no free update slot).
    long begin_update(std::size_t container)
    {
        if (container >= capacity || refs_[container] == 0) return -1;
        for (std::size_t i = 0; i < capacity; ++i)
            if (!updates_[i]) {
                updates_[i] = static_cast<int>(container) + 1;
                return static_cast<long>(i);
            }
        return -1;
    }
    void end_update(std::size_t update)
    {
        if (update >= capacity || !updates_[update]) return;
        const std::size_t container = static_cast<std::size_t>(updates_[update] - 1);
        updates_[update] = 0;
        staged_[update].clear();  // an update closed without a submit leaves nothing behind
        staged_bytes_[update] = 0;
        reclaim(container);
    }
    // Keeps an action with the update until it is submitted. False: no such update, or too much staged.
    bool stage(std::size_t update, Action action)
    {
        if (update >= capacity || !updates_[update]) return false;
        if (action.data.size() > max_staged_bytes - staged_bytes_[update]) return false;
        staged_bytes_[update] += action.data.size();
        staged_[update].push_back(std::move(action));
        return true;
    }
    // The staged actions in order; the update starts empty again.
    std::vector<Action> take(std::size_t update)
    {
        std::vector<Action> actions;
        if (update >= capacity || !updates_[update]) return actions;
        actions.swap(staged_[update]);
        staged_bytes_[update] = 0;
        return actions;
    }
    std::size_t staged_count(std::size_t update) const { return update < capacity ? staged_[update].size() : 0; }
    // Name of the container behind an open update, or empty.
    std::string update_container(std::size_t update) const
    {
        return update < capacity && updates_[update] ? names_[updates_[update] - 1] : std::string();
    }
    // Name of an open container, or empty. The caller holds the lock.
    std::string container_name(std::size_t index) const
    {
        return index < capacity && refs_[index] > 0 ? names_[index] : std::string();
    }
    std::size_t in_use() const
    {
        std::size_t used = 0;
        for (const std::string& name : names_) used += !name.empty();
        return used;
    }

private:
    void reclaim(std::size_t index)
    {
        if (refs_[index] > 0) return;
        for (const int update : updates_)
            if (update == static_cast<int>(index) + 1) return;
        names_[index].clear();
    }
    std::string names_[capacity];
    int refs_[capacity]{};
    int updates_[capacity]{};  // container index + 1 of an open update, 0 = free
    std::vector<Action> staged_[capacity];
    std::size_t staged_bytes_[capacity]{};
};

// One entry of an enumeration. Names are the file names on disk, so they already passed valid_name.
struct ContainerEntry {
    std::string name;
};
struct BlobEntry {
    std::string name;
    std::uint64_t bytes = 0;
};

class Store {
public:
    explicit Store(fs::path root, std::chrono::milliseconds lock_timeout = default_lock_timeout)
        : root_(std::move(root)), lock_timeout_(lock_timeout) {}

    // An existing container directory counts as success; a redirected one is refused.
    Status create_container(std::string_view container) const
    {
        const fs::path dir = container_dir(container);
        if (dir.empty()) return Status::bad_name;
        std::error_code ec;
        if (redirected(fs::symlink_status(dir, ec))) return Status::bad_name;
        if (fs::is_directory(dir, ec)) return Status::ok;
        fs::create_directories(dir, ec);
        if (ec || redirected(fs::symlink_status(dir, ec))) return Status::io_error;
        return fs::is_directory(dir, ec) ? Status::ok : Status::io_error;
    }

    Status delete_container(std::string_view container) const
    {
        const fs::path dir = container_dir(container);
        if (dir.empty()) return Status::bad_name;
        std::error_code ec;
        const fs::file_status state = fs::symlink_status(dir, ec);
        if (!fs::exists(state)) return Status::not_found;
        if (!fs::is_directory(state) || redirected(state)) return Status::bad_name;
        FileLock publication;
        if (const Status locked = lock_container(publication, container, true); locked != Status::ok) return locked;
        fs::remove_all(dir, ec);
        return ec ? Status::io_error : Status::ok;
    }

    // Publishes a whole update or nothing. Every write goes to a staging file first and every
    // delete is checked; only then are the blobs replaced, each old file moved aside first, and
    // any failure puts everything back. A later action on a blob wins over an earlier one. This
    // keeps the disk unchanged when an action fails inside this process. The replacement runs under
    // an exclusive lock on the container that readers (read_blobs, read_blob) share, so another
    // process sees the whole old state or the whole new one. A crash in the middle of the
    // replacement can still leave some blobs replaced and ".x2w-old" files behind: the lock is
    // released by the system, but nothing completes or rolls back the interrupted commit.
    Status commit(std::string_view container, const std::vector<Action>& actions) const
    {
        if (actions.empty()) return Status::ok;
        const fs::path dir = container_dir(container);
        if (dir.empty()) return Status::bad_name;
        std::vector<const Action*> last;  // the final action of each blob, in first-seen order
        for (const Action& action : actions) {
            if (!valid_name(action.blob)) return Status::bad_name;
            if (!action.remove && action.data.size() > max_blob_bytes) return Status::too_large;
            bool seen = false;
            for (const Action*& slot : last)
                if (slot->blob == action.blob) slot = &action, seen = true;
            if (!seen) last.push_back(&action);
        }
        if (const Status made = create_container(container); made != Status::ok) return made;
        static std::mutex commit_lock;  // one commit at a time in this process; the file lock below covers other processes
        const std::lock_guard<std::mutex> guard(commit_lock);
        struct Step {
            fs::path file, part, old;
            bool remove = false, moved_old = false, installed = false;
        };
        std::vector<Step> steps;
        std::error_code ec;
        const auto discard_parts = [&] {
            for (const Step& step : steps)
                if (!step.part.empty()) fs::remove(step.part, ec);
        };
        for (const Action* action : last) {
            Step step;
            step.file = dir / action->blob;
            step.remove = action->remove;
            if (const Status target = check_target(step.file, step.remove); target != Status::ok) {
                discard_parts();
                return target;
            }
            if (!action->remove) {
                step.part = step.file;
                step.part += unique_name(part_suffix);
                std::ofstream out(step.part, std::ios::binary | std::ios::trunc);
                steps.push_back(step);  // recorded first so that a failure removes the partial file
                if (!out) {
                    discard_parts();
                    return Status::io_error;
                }
                if (!action->data.empty())
                    out.write(reinterpret_cast<const char*>(action->data.data()), static_cast<std::streamsize>(action->data.size()));
                if (!out.flush()) {
                    out.close();
                    discard_parts();
                    return Status::io_error;
                }
                continue;
            }
            steps.push_back(step);
        }
        // The publication point: readers hold this lock shared, so they see the blobs before the
        // first replacement or after the last one, never in between. The staging above ran outside it.
        FileLock publication;
        if (const Status locked = lock_container(publication, container, true); locked != Status::ok) {
            discard_parts();
            return locked;
        }
        for (const Step& step : steps)
            if (const Status target = check_target(step.file, step.remove); target != Status::ok) {
                discard_parts();
                return target;
            }
        const auto undo = [&] {
            for (auto it = steps.rbegin(); it != steps.rend(); ++it) {
                if (it->installed) fs::remove(it->file, ec);
                if (it->moved_old) rename_with_retry(it->old, it->file);
            }
            discard_parts();
        };
        for (Step& step : steps) {
            if (fs::exists(fs::symlink_status(step.file, ec))) {
                step.old = step.file;
                step.old += unique_name(old_suffix);
                if (!rename_with_retry(step.file, step.old)) {
                    undo();
                    return Status::io_error;
                }
                step.moved_old = true;
            }
            // x2w-seam: moved-aside
            if (!step.remove) {
                if (!rename_with_retry(step.part, step.file)) {
                    undo();
                    return Status::io_error;
                }
                step.installed = true;
                step.part.clear();
            }
            // x2w-seam: published
        }
        publication.release();
        for (const Step& step : steps)
            if (step.moved_old) fs::remove(step.old, ec);
        return Status::ok;
    }

    // One blob, published at once: the same as a commit of a single write.
    Status write_blob(std::string_view container, std::string_view blob, const void* data, std::size_t size) const
    {
        if (size > max_blob_bytes) return Status::too_large;
        if (size && !data) return Status::io_error;
        Action action;
        action.blob = std::string(blob);
        action.data.assign(static_cast<const unsigned char*>(data), static_cast<const unsigned char*>(data) + size);
        return commit(container, {action});
    }

    // not_found only when the blob is truly absent; any other failure is io_error. busy: the
    // publication point stayed locked for the whole lock timeout.
    Status read_blob(std::string_view container, std::string_view blob, std::vector<unsigned char>& bytes) const
    {
        std::vector<std::vector<unsigned char>> one;
        const Status status = read_blobs(container, {std::string(blob)}, one);
        if (status == Status::ok) bytes = std::move(one.front());
        return status;
    }

    // Several blobs of one container as one generation: a commit from any process is seen whole or
    // not at all. All or nothing; a blob missing from the container is not_found.
    Status read_blobs(std::string_view container, const std::vector<std::string>& blobs, std::vector<std::vector<unsigned char>>& out) const
    {
        // x2w-seam: store-read
        const fs::path dir = container_dir(container);
        if (dir.empty()) return Status::bad_name;
        for (const std::string& blob : blobs)
            if (!valid_name(blob)) return Status::bad_name;
        std::error_code ec;
        const fs::file_status dir_state = fs::symlink_status(dir, ec);
        if (redirected(dir_state)) return Status::bad_name;
        if (!fs::exists(dir_state)) return Status::not_found;
        FileLock publication;
        if (const Status locked = lock_container(publication, container, false); locked != Status::ok) return locked;
        std::vector<std::vector<unsigned char>> all(blobs.size());
        for (std::size_t i = 0; i < blobs.size(); ++i) {
            if (const Status status = read_file(dir / blobs[i], all[i]); status != Status::ok) return status;
            // x2w-seam: blob-read
        }
        out = std::move(all);
        return Status::ok;
    }

    // The containers of the store: directories directly below the root, by name (byte order).
    // Links and junctions, plain files, and names valid_name refuses (the lock, staging and old
    // suffixes included) are not listed. A root that does not exist yet lists as empty. `out` is
    // replaced only on ok; on any other status it is left as it was.
    Status list_containers(std::vector<ContainerEntry>& out) const
    {
        std::error_code ec;
        const fs::file_status root_state = fs::symlink_status(root_, ec);
        if (!fs::exists(root_state)) {
            if (ec && root_state.type() != fs::file_type::not_found) return Status::io_error;
            out.clear();
            return Status::ok;
        }
        if (redirected(root_state)) return Status::bad_name;
        if (!fs::is_directory(root_state)) return Status::io_error;
        std::vector<ContainerEntry> found;
        for (fs::directory_iterator it(root_, ec), end; !ec && it != end; it.increment(ec)) {
            const fs::file_status state = it->symlink_status(ec);
            if (ec) {
                if (state.type() != fs::file_type::not_found) break;
                ec.clear();  // removed since the directory was read
                continue;
            }
            if (redirected(state) || !fs::is_directory(state)) continue;
            std::string name = it->path().filename().string();
            if (valid_name(name)) found.push_back({std::move(name)});
        }
        if (ec) return Status::io_error;
        std::sort(found.begin(), found.end(), [](const ContainerEntry& a, const ContainerEntry& b) { return a.name < b.name; });
        out = std::move(found);
        return Status::ok;
    }

    // The blobs of one container as one generation, taken under the same shared lock as read_blobs:
    // a commit from any process is listed whole or not at all. Regular files only, by name (byte
    // order), with their sizes; links, directories and names valid_name refuses are not listed.
    // not_found: no such container. busy: the publication point stayed locked for the whole lock
    // timeout. too_large: a blob is over max_blob_bytes. `out` is replaced only on ok. With
    // `payloads`, the listed blobs are also read under the same lock (one generation for names,
    // sizes and bytes); it is replaced only on ok, in the order of `out`. `read_if`, when given, sees
    // the listing first: when it returns false nothing is read and `payloads` is left empty.
    Status list_blobs(std::string_view container, std::vector<BlobEntry>& out,
                      std::vector<std::vector<unsigned char>>* payloads = nullptr,
                      const std::function<bool(const std::vector<BlobEntry>&)>& read_if = {}) const
    {
        // x2w-seam: store-read
        const fs::path dir = container_dir(container);
        if (dir.empty()) return Status::bad_name;
        std::error_code ec;
        const fs::file_status dir_state = fs::symlink_status(dir, ec);
        if (redirected(dir_state)) return Status::bad_name;
        if (!fs::exists(dir_state))
            return (ec && dir_state.type() != fs::file_type::not_found) ? Status::io_error : Status::not_found;
        if (!fs::is_directory(dir_state)) return Status::io_error;
        FileLock publication;
        if (const Status locked = lock_container(publication, container, false); locked != Status::ok) return locked;
        // Re-check under the lock: a concurrent delete_container or redirect swap can happen while waiting.
        const fs::file_status locked_state = fs::symlink_status(dir, ec);
        if (redirected(locked_state)) return Status::bad_name;
        if (!fs::exists(locked_state))
            return (ec && locked_state.type() != fs::file_type::not_found) ? Status::io_error : Status::not_found;
        if (!fs::is_directory(locked_state)) return Status::io_error;
        std::vector<BlobEntry> found;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            const fs::file_status state = it->symlink_status(ec);
            if (ec) {
                if (state.type() != fs::file_type::not_found) break;
                ec.clear();
                continue;
            }
            if (redirected(state) || !fs::is_regular_file(state)) continue;
            std::string name = it->path().filename().string();
            if (!valid_name(name)) continue;
            const std::uintmax_t size = it->file_size(ec);
            if (ec) break;
            if (size > max_blob_bytes) return Status::too_large;
            found.push_back({std::move(name), static_cast<std::uint64_t>(size)});
        }
        if (ec) return Status::io_error;
        std::sort(found.begin(), found.end(), [](const BlobEntry& a, const BlobEntry& b) { return a.name < b.name; });
        if (payloads) {
            std::vector<std::vector<unsigned char>> all;
            if (!read_if || read_if(found)) {
                all.resize(found.size());
                for (std::size_t i = 0; i < found.size(); ++i) {
                    if (const Status status = read_file(dir / found[i].name, all[i]); status != Status::ok) return status;
                    found[i].bytes = all[i].size();
                    // x2w-seam: blob-read
                }
            }
            *payloads = std::move(all);
        }
        out = std::move(found);
        return Status::ok;
    }

    Status delete_blob(std::string_view container, std::string_view blob) const
    {
        const fs::path file = blob_file(container, blob);
        if (file.empty()) return Status::bad_name;
        std::error_code ec;
        const fs::file_status dir_state = fs::symlink_status(file.parent_path(), ec);
        if (redirected(dir_state)) return Status::bad_name;
        if (!fs::exists(dir_state)) return Status::not_found;
        FileLock publication;
        if (const Status locked = lock_container(publication, container, true); locked != Status::ok) return locked;
        const fs::file_status state = fs::symlink_status(file, ec);
        if (!fs::exists(state)) return Status::not_found;
        if (redirected(state) || !fs::is_regular_file(state)) return Status::bad_name;
        return fs::remove(file, ec) && !ec ? Status::ok : Status::io_error;
    }

private:
    // A blob file that is missing is not_found; one that is not a plain file is io_error.
    static Status read_file(const fs::path& file, std::vector<unsigned char>& bytes)
    {
        std::error_code ec;
        const fs::file_status state = fs::symlink_status(file, ec);
        if (!fs::exists(state)) return Status::not_found;
        if (redirected(state) || !fs::is_regular_file(state)) return Status::io_error;
        const std::uintmax_t size = fs::file_size(file, ec);
        if (ec || size > max_blob_bytes) return ec ? Status::io_error : Status::too_large;
        std::ifstream in(file, std::ios::binary);
        if (!in) return Status::io_error;
        bytes.assign(static_cast<std::size_t>(size), 0);
        if (size && !in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size))) return Status::io_error;
        return Status::ok;
    }

    // What a commit step needs of its blob file: not a link or a directory, and present when it is a delete.
    static Status check_target(const fs::path& file, bool remove)
    {
        std::error_code ec;
        const fs::file_status state = fs::symlink_status(file, ec);
        if (redirected(state) || (fs::exists(state) && !fs::is_regular_file(state))) return Status::bad_name;
        return remove && !fs::exists(state) ? Status::not_found : Status::ok;
    }

    // Locks the publication point of a container: a file beside its directory, so deleting the
    // directory never touches it. The lock outlives nothing: the system drops it when the process ends.
    Status lock_container(FileLock& held, std::string_view container, bool exclusive) const
    {
        fs::path file = container_dir(container);
        if (file.empty()) return Status::bad_name;
        file += lock_suffix;
        switch (held.acquire(file, exclusive, lock_timeout_)) {
        case LockResult::locked: return Status::ok;
        case LockResult::timeout: return Status::busy;
        default: return Status::io_error;
        }
    }

    // Replacing or moving a file another writer holds can fail on Windows (access denied or a
    // sharing violation instead of last-writer-wins), so a few retries with a growing pause.
    static bool rename_with_retry(const fs::path& from, const fs::path& to)
    {
        std::error_code ec;
        for (int attempt = 0;; ++attempt) {
            fs::rename(from, to, ec);
            if (!ec) return true;
            if (attempt == 8) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(5 << (attempt < 4 ? attempt : 4)));
        }
    }

    // Empty path when the name is not acceptable. The result is always root/<container>.
    fs::path container_dir(std::string_view container) const
    {
        if (!valid_name(container)) return {};
        return root_ / std::string(container);
    }
    fs::path blob_file(std::string_view container, std::string_view blob) const
    {
        if (!valid_name(blob)) return {};
        const fs::path dir = container_dir(container);
        return dir.empty() ? fs::path{} : dir / std::string(blob);
    }

    fs::path root_;
    std::chrono::milliseconds lock_timeout_;
};

}  // namespace x2w_save
