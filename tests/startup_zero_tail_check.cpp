// src/startup_stubs/zero_tail.h: the copy of a bound buffer drops its trailing zeros; a view over the shorter copy reads the
// same values as one over the full range (reads past the end of a raw view return zero).
// --mutate keep-all: the length never shrinks (the old behaviour); the ring-use check must fail.
// --mutate cache: the per-frame cached length is used (the 68a75e14c behaviour); a write into the tail between two draws is missed.
#include "startup_stubs/zero_tail.h"

#include <cstdio>
#include <cstring>
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
bool keep_all;
std::size_t used(const std::vector<unsigned char>& v) { return keep_all ? v.size() : x2w_local::used_length(v.data(), v.size()); }
// What a raw view of `length` bytes returns for the 32-bit word at `byte`: zero past the end.
unsigned read_word(const std::vector<unsigned char>& v, std::size_t length, std::size_t byte)
{
    unsigned w = 0;
    if (byte + 4 <= length) std::memcpy(&w, v.data() + byte, 4);
    return w;
}
}  // namespace

int main(int argc, char** argv)
{
    keep_all = argc == 3 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "keep-all");
    const std::size_t full = 16u << 20;
    std::vector<unsigned char> buffer(full, 0);
    check("an all-zero 16 MiB range shrinks to the granule", used(buffer) == 256);
    buffer[1000] = 7;
    check("data at byte 1000 keeps 1024 bytes", used(buffer) == 1024);
    buffer[full - 1] = 1;
    check("a buffer used to its end is copied whole", used(buffer) == full);
    buffer[full - 1] = 0;
    buffer[5 * 4096 + 3] = 9;
    const std::size_t length = used(buffer);
    check("the copy of a 16 MiB buffer written to 20 KiB is far below the ring size", length < (1u << 20));
    bool same = true;
    for (std::size_t at = 0; at + 4 <= full; at += 4) same = same && read_word(buffer, length, at) == read_word(buffer, full, at);
    check("every word reads the same through the shorter copy", same);
    check("a small range is returned as is", x2w_local::used_length(buffer.data(), 100) == 100);
    std::vector<unsigned char> odd(1001, 0);
    odd[1000] = 3;
    check("an odd length with data in the last byte is whole", x2w_local::used_length(odd.data(), odd.size()) == odd.size());
    // Stale-length probe: the title writes into the zero tail between two draws of the SAME frame, far from the used part and from
    // the end of the range; the second draw must see it (a cached length would hand the shader zeros).
    {
        const bool cached = argc == 3 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "cache");
        std::vector<unsigned char> b(full, 0);
        b[5 * 4096 + 3] = 9;
        x2w_local::ZeroTailCache tails;
        const std::size_t first = tails.length(7, b.data(), full, cached);
        b[9u << 20] = 5;  // 9 MiB in: outside the 64 KiB after the used part and the last 4 KiB
        const std::size_t second = tails.length(7, b.data(), full, cached);
        check("the first draw copies the used part only", first < (1u << 20));
        check("data written into the tail between two draws of a frame is copied by the second", second > (9u << 20));
        unsigned seen = 0;
        if ((9u << 20) + 4 <= second) std::memcpy(&seen, b.data() + (9u << 20), 4);
        check("the shader-visible word at 9 MiB is the new data", seen == 5);
    }
    if (failures) return 1;
    std::printf("startup_zero_tail: PASS\n");
    return 0;
}
