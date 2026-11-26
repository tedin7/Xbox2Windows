// The read plan of the DirectStorage stand-in's HDD mode (src/startup_stubs/dstorage_read_plan.h): order, merging, the
// gap and span limits, and what is dropped. Pure logic: no disk, no queue.
#include "startup_stubs/dstorage_read_plan.h"

#include <cstdio>

using x2w_startup::PlannedRead;
using x2w_startup::ReadGroup;
using x2w_startup::plan_reads;

static int failures = 0;

static void expect(bool ok, const char* what)
{
    std::printf("%s: %s\n", what, ok ? "ok" : "MISMATCH");
    if (!ok) ++failures;
}

static PlannedRead read(std::uintptr_t file, std::uint64_t offset, std::uint32_t size, std::size_t index)
{
    PlannedRead r;
    r.file = file;
    r.offset = offset;
    r.size = size;
    r.index = index;
    return r;
}

int main()
{
    // 1. Shuffled neighbours of one file become one forward read that serves all of them.
    {
        auto groups = plan_reads({read(1, 8192, 4096, 0), read(1, 0, 4096, 1), read(1, 4096, 4096, 2)}, 0, 1 << 20);
        expect(groups.size() == 1 && groups[0].offset == 0 && groups[0].span == 12288 && groups[0].members.size() == 3,
               "three touching reads become one read of 12288 bytes");
        expect(groups[0].members[0].index == 1 && groups[0].members[1].index == 2 && groups[0].members[2].index == 0,
               "members come out in offset order and keep the caller's numbers");
    }
    // 2. A gap up to the limit is read through; one byte more starts a new group.
    {
        auto joined = plan_reads({read(1, 0, 100, 0), read(1, 100 + 65536, 100, 1)}, 65536, 1 << 20);
        auto split = plan_reads({read(1, 0, 100, 0), read(1, 100 + 65537, 100, 1)}, 65536, 1 << 20);
        expect(joined.size() == 1 && joined[0].span == 100 + 65536 + 100, "a gap of exactly the limit is joined");
        expect(split.size() == 2, "a gap of one byte more is split");
    }
    // 3. The span limit closes a group; a read larger than the limit stands alone.
    {
        auto groups = plan_reads({read(1, 0, 3000, 0), read(1, 3000, 3000, 1), read(1, 6000, 3000, 2), read(1, 9000, 50000, 3)}, 0, 8192);
        expect(groups.size() == 3 && groups[0].members.size() == 2 && groups[1].members.size() == 1 && groups[2].members.size() == 1,
               "span limit 8192: 2 + 1 + the oversized read alone");
        expect(groups[0].span <= 8192 && groups[2].span == 50000, "no group but the single oversized read exceeds the limit");
    }
    // 4. Different files never merge, and each file is swept forward in file order.
    {
        auto groups = plan_reads({read(2, 0, 10, 0), read(1, 500, 10, 1), read(2, 10, 10, 2), read(1, 0, 10, 3)}, 1 << 20, 1 << 20);
        expect(groups.size() == 2 && groups[0].file == 1 && groups[1].file == 2, "two files give two groups, lower file first");
        expect(groups[0].members.size() == 2 && groups[1].members.size() == 2, "each file's reads share their group");
    }
    // 5. Overlapping reads: the group covers the union; identical reads are both members.
    {
        auto groups = plan_reads({read(1, 0, 100, 0), read(1, 50, 100, 1), read(1, 50, 100, 2)}, 0, 1 << 20);
        expect(groups.size() == 1 && groups[0].span == 150 && groups[0].members.size() == 3, "overlaps merge into the union (150 bytes)");
    }
    // 6. Empty reads are dropped; nothing in, nothing out; the saving is members minus groups.
    {
        expect(plan_reads({}, 0, 1).empty(), "no reads, no groups");
        auto groups = plan_reads({read(1, 0, 0, 0), read(1, 4, 4, 1), read(1, 8, 4, 2)}, 0, 1 << 20);
        expect(groups.size() == 1 && groups[0].members.size() == 2 && x2w_startup::reads_saved(groups) == 1,
               "a zero-size read is dropped; two touching reads save one disk read");
    }
    // 7. Offsets near 2^64 do not wrap into an earlier group (a read at the top of the range stays its own group).
    {
        auto groups = plan_reads({read(1, 0, 16, 0), read(1, ~0ull - 7, 4, 1)}, 65536, 1 << 20);
        expect(groups.size() == 2, "a read near 2^64 is not merged with a read at offset 0");
    }
    if (failures) {
        std::printf("FAILED: %d\n", failures);
        return 1;
    }
    std::puts("ok");
    return 0;
}
