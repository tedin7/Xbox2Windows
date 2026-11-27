#include "startup_stubs/buffer_descriptor_range.h"

#include <cstdio>

int main()
{
    constexpr std::uint32_t strides[] = {0, 4, 16};
    constexpr std::uint32_t records = 64;
    constexpr std::uint64_t table37[] = {64, 256, 1024};
    constexpr std::uint64_t raw_mode3[] = {64, 64, 64};
    unsigned differences = 0;
    for (std::uint32_t mode = 0; mode < 4; ++mode) {
        // Test both address swizzle states with and without the cache-bank hint.
        for (std::uint32_t flags = 0; flags < 4; ++flags) {
            const bool swizzle = (flags & 2) != 0;
            for (unsigned i = 0; i < 3; ++i) {
                const std::uint32_t word1 = (flags << 30) | (strides[i] << 16) | 0xabcd;
                const std::uint32_t word3 = (mode << 28) | 0xfacu;
                const auto expected = mode == 3 && !swizzle ? raw_mode3[i] : table37[i];
                const auto actual = x2w_startup::buffer_descriptor_bytes(word1, records, word3);
                if (actual != expected) {
                    std::fprintf(stderr, "mode %u stride %u flags %u: got %llu, expected %llu (Table 37: %llu)\n",
                        mode, strides[i], flags, static_cast<unsigned long long>(actual),
                        static_cast<unsigned long long>(expected), static_cast<unsigned long long>(table37[i]));
                    return 1;
                }
                differences += actual != table37[i];
            }
        }
    }
    if (differences != 4) return 1;
    std::puts("48 V# ranges match, with 4 differences from Table 37. Always multiplying by stride fails those cases.");
}
