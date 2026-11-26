// Checks src/startup_stubs/manifest_prefetch.h without a device: translation manifests are read by a background pass, and the
// call the draw path makes with draw_lock held never waits for a file (a fake file system whose every access takes 50 ms).
// --mutate open-under-lock: the draw-path call opens the file itself (the old behaviour); the lock-hold measurement must fail.
#include "startup_stubs/manifest_prefetch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
using x2w_startup::ManifestPrefetch;

int failures;
void check(const char* name, bool ok)
{
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

constexpr int slow_ms = 50;
std::mutex fs_mutex;
std::set<std::uint64_t> files;                 // keys that have a manifest in the fake directory
std::atomic<int> accesses_on_draw_thread{0};   // file accesses made on the thread that plays the draw path
std::thread::id draw_thread;

void slow_access()
{
    if (std::this_thread::get_id() == draw_thread) ++accesses_on_draw_thread;
    std::this_thread::sleep_for(std::chrono::milliseconds(slow_ms));
}

ManifestPrefetch::Io fake_io()
{
    ManifestPrefetch::Io io;
    io.list = [] {
        slow_access();
        std::lock_guard<std::mutex> lock(fs_mutex);
        return std::vector<std::uint64_t>(files.begin(), files.end());
    };
    io.read = [](std::uint64_t key, std::string& text) {
        slow_access();
        std::lock_guard<std::mutex> lock(fs_mutex);
        if (!files.count(key)) return false;
        text = "x2w-manifest 2 key " + std::to_string(key);
        return true;
    };
    return io;
}

double ms_since(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}  // namespace

int main(int argc, char** argv)
{
    const bool mutate = argc == 3 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "open-under-lock") == 0;
    if (argc != 1 && !mutate) {
        std::printf("usage: startup_manifest_prefetch_check [--mutate open-under-lock]\n");
        return 2;
    }
    draw_thread = std::this_thread::get_id();
    ManifestPrefetch::Io io = fake_io();
    ManifestPrefetch prefetch(io);
    // The draw path as the mutant has it: the file is opened inside the call, with the lock held.
    const auto draw_path_fetch = [&](std::uint64_t key, std::string& out) {
        if (mutate) return io.read(key, out);
        return prefetch.fetch(key, out);
    };

    {
        std::lock_guard<std::mutex> lock(fs_mutex);
        files.insert(7);
    }
    std::string text;
    check("a key nobody scanned for is not ready", !prefetch.fetch(7, text));
    check("asking registers the key as wanted", prefetch.has_wanted());

    // The background pass runs on its own thread while the draw path keeps asking; the draw path is timed.
    std::atomic<bool> stop{false};
    std::thread background([&] {
        while (!stop) {
            prefetch.scan();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });
    double worst_ms = 0;
    bool got = false;
    std::string loaded;
    for (int round = 0; round < 40; ++round) {
        std::string out;
        const auto start = Clock::now();
        const bool ready = draw_path_fetch(7, out);
        const double took = ms_since(start);
        if (took > worst_ms) worst_ms = took;
        if (ready && !got) {
            got = true;
            loaded = out;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (got && round > 20) break;
    }
    {
        std::lock_guard<std::mutex> lock(fs_mutex);
        files.insert(9);  // a manifest that appears while the run goes on
    }
    std::string late;
    bool late_ready = false;
    for (int round = 0; round < 60 && !late_ready; ++round) {
        late_ready = prefetch.fetch(9, late);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    stop = true;
    background.join();

    check("the manifest is handed over once the background pass has read it", got || mutate);
    check("with its text", mutate || loaded == "x2w-manifest 2 key 7");
    check("a manifest created later is found without the draw path doing any I/O", late_ready && late == "x2w-manifest 2 key 9");
    check("the draw path never accessed the file system", accesses_on_draw_thread == 0);
    check("one 50 ms file access never lengthens the draw-path call (under 10 ms)", worst_ms < 10.0);

    // A manifest is handed out once; asking again makes the key wanted again (it is read again if it is still there).
    std::string again;
    check("handed out once", !prefetch.fetch(9, again));
    check("a vanished file stays wanted and is not handed out", [&] {
        {
            std::lock_guard<std::mutex> lock(fs_mutex);
            files.erase(7);
        }
        ManifestPrefetch other(io);
        std::string t;
        other.fetch(7, t);
        const std::size_t n = other.scan();
        return n == 0 && other.has_wanted();
    }());

    if (failures) {
        std::printf("startup_manifest_prefetch: %d FAILED (worst draw-path call %.1f ms)\n", failures, worst_ms);
        return 1;
    }
    std::printf("startup_manifest_prefetch: PASS (worst draw-path call %.2f ms)\n", worst_ms);
    return 0;
}
