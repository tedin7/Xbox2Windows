// src/startup_stubs/copy_cache_trim.h: a storm of draws over 16 MiB buffers copies each buffer once, not once per frame, and a
// change of the title's bytes still reaches the copy. The loop uses the same decision as frame_copy: look the entry up, compare the
// snapshot with the title's memory, copy only when they differ.
// --mutate per-frame: every entry is dropped at each frame start (the old behaviour); the copy bound must fail.
#include "startup_stubs/copy_cache_trim.h"

#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {
int failures;
void check(const char* name, bool ok)
{
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}
struct Entry {
    std::vector<unsigned char> content;
};
bool per_frame;
long long copies_made, copies_reused;
std::unordered_map<int, Entry> cache;

// One draw of buffer `id` whose title bytes are `title`.
void draw(int id, const std::vector<unsigned char>& title)
{
    const auto found = cache.find(id);
    if (found != cache.end() && found->second.content == title) {
        ++copies_reused;
        return;
    }
    ++copies_made;
    cache[id] = Entry{title};
}
void frame_start()
{
    if (per_frame) cache.clear();
    else x2w_local::trim_frame_copies(cache);
}
}  // namespace

int main(int argc, char** argv)
{
    per_frame = argc == 3 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "per-frame");
    const std::size_t big = 16u << 20;
    std::vector<unsigned char> a(big, 0), b(big, 0);
    a[100] = 1;
    b[200] = 2;
    std::vector<unsigned char> small(4096, 7);
    // The storm: 40 frames, 3 draws of each of two 16 MiB buffers and a small constant buffer.
    for (int frame = 0; frame < 40; ++frame) {
        frame_start();
        for (int d = 0; d < 3; ++d) {
            draw(1, a);
            draw(2, b);
            draw(3, small);
        }
    }
    check("two unchanged 16 MiB buffers over 40 frames are copied once each (plus the small one once per frame)", copies_made <= 2 + 40);
    check("the draws after the first reuse the copy", copies_reused >= 40 * 3 * 2 - 2);
    // A change reaches the copy: the next draw copies, later draws reuse the new copy.
    const long long before = copies_made;
    a[big - 5] = 9;
    frame_start();
    draw(1, a);
    draw(1, a);
    check("a change of the title's bytes is copied exactly once", copies_made == before + 1);
    check("the new copy matches the title", cache[1].content == a);
    // The snapshots are bounded: past the budget everything is dropped and copied again.
    cache.clear();
    for (int id = 10; id < 18; ++id) cache[id] = Entry{std::vector<unsigned char>(16u << 20, 0)};
    frame_start();
    check("past the budget the kept snapshots are dropped", cache.empty());
    if (failures) return 1;
    std::printf("startup_copy_cache: PASS (made %lld, reused %lld)\n", copies_made, copies_reused);
    return 0;
}
