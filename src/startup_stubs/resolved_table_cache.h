// Remembers where a root descriptor table was found, so a repeated SetXxxRootDescriptorTable with the same GPU handle
// does not take the heap lock and walk every heap again (X2W_SPEED_ROOT=1). Plain standard C++, tested on its own.
//
// Only a found table is kept. The heap list only grows and a found entry never changes, so a kept answer stays right;
// a miss is never kept because a heap created later may resolve the handle.
#pragma once

#include <cstdint>

#include "logical_view_table.h"

namespace x2w_startup {

class ResolvedTableCache {
public:
    static constexpr unsigned slots = 64;

    // The table kept for (view heap, handle), or an empty table when none is kept.
    ViewTable find(std::uint64_t view_heap, std::uint64_t handle) const noexcept
    {
        const Entry& e = entries_[index(handle)];
        return e.used && e.heap == view_heap && e.handle == handle ? e.table : ViewTable{};
    }

    void keep(std::uint64_t view_heap, std::uint64_t handle, const ViewTable& table) noexcept
    {
        if (!table.address) return;
        entries_[index(handle)] = {true, view_heap, handle, table};
    }

    // For a caller that learns a heap went away or changed.
    void clear() noexcept { *this = ResolvedTableCache{}; }

private:
    struct Entry {
        bool used = false;
        std::uint64_t heap = 0, handle = 0;
        ViewTable table;
    };
    static unsigned index(std::uint64_t handle) noexcept { return static_cast<unsigned>((handle >> 4) ^ (handle >> 11)) % slots; }
    Entry entries_[slots];
};

}  // namespace x2w_startup
