#pragma once

#include <windows.h>
#include <cstdint>

namespace x2w_local {

struct ReadCacheStats {
    unsigned long long lookups, hits, queries, spans, erases;
};

struct ReadCacheApi {
    bool (*query)(const void*, MEMORY_BASIC_INFORMATION*) noexcept;
    void (*record)(const void*, SIZE_T, DWORD) noexcept;
    void (*erase)(const void*, SIZE_T) noexcept;
    ReadCacheStats (*stats)() noexcept;
};

}  // namespace x2w_local
