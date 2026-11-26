#include "startup_stubs/upload_reservation.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

constexpr std::uint64_t kAlign = 512;

}  // namespace

int main()
{
    unsigned char storage_a[4 * kAlign];
    unsigned char storage_b[8 * kAlign];
    std::vector<UploadRingChunk> chunks;
    chunks.push_back({nullptr, storage_a, sizeof(storage_a), 0});

    const auto first = reserve_upload_chunk(chunks[0], 2 * kAlign);
    check(first.mapped == storage_a && first.offset == 0, "initial reservation");
    check(chunks[0].used == 2 * kAlign, "initial cursor");
    first.mapped[0] = 0xa5;
    first.mapped[2 * kAlign - 1] = 0x5a;

    const auto refused = reserve_upload_chunk(chunks[0], 3 * kAlign);
    check(!refused.mapped && !refused.resource && refused.offset == 0,
          "insufficient tail refuses the reservation");
    check(chunks[0].used == 2 * kAlign, "refusal preserves cursor");

    chunks.push_back({nullptr, storage_b, sizeof(storage_b), 0});
    const auto second = reserve_upload_chunk(chunks[1], 3 * kAlign);
    check(second.mapped == storage_b && second.offset == 0, "new chunk reservation");
    second.mapped[0] = 0xc3;
    chunks.reserve(chunks.capacity() + 1);
    check(first.mapped[0] == 0xa5 && first.mapped[2 * kAlign - 1] == 0x5a,
          "first mapped reservation survives vector growth");
    check(second.mapped[0] == 0xc3, "new mapped reservation survives vector growth");

    const auto next = reserve_upload_chunk(chunks[1], 5 * kAlign);
    check(next.mapped == storage_b + 3 * kAlign && next.offset == 3 * kAlign,
          "exact remaining active tail");
    check(chunks[1].used == chunks[1].capacity, "active cursor reaches capacity");
    check(!reserve_upload_chunk(chunks[1], kAlign).mapped, "full active chunk refuses");
    check(chunks[1].used == chunks[1].capacity, "active refusal preserves cursor");

    const auto older = reserve_upload_chunk(chunks[0], 2 * kAlign);
    check(older.mapped == storage_a + 2 * kAlign && older.offset == 2 * kAlign,
          "older chunk reserves only previously unused bytes");
    older.mapped[0] = 0x7d;
    check(chunks[0].used == chunks[0].capacity, "older cursor reaches capacity");
    check(!reserve_upload_chunk(chunks[0], kAlign).mapped, "full older chunk refuses");
    check(chunks[0].used == chunks[0].capacity && storage_a[0] == 0xa5 &&
          storage_a[2 * kAlign - 1] == 0x5a && storage_a[2 * kAlign] == 0x7d,
          "existing reservations survive exhaustion");

    // Recycling: a chunk comes back only once the GPU is done with the frame that last reserved from it, plus the slack.
    UploadRingChunk ring{nullptr, storage_a, sizeof(storage_a), 0, -1};
    check(reserve_upload_chunk(ring, 4 * kAlign, 10).mapped == storage_a && ring.epoch == 10, "stamped with the frame");
    check(!reserve_upload_chunk(ring, kAlign, 11).mapped && ring.epoch == 10, "a refused reservation does not move the stamp");
    check(!recycle_upload_chunk(ring, -1, 1), "no completed frame yet: nothing recycled");
    check(!recycle_upload_chunk(ring, 10, 1) && ring.used == 4 * kAlign, "the frame itself is still in flight");
    check(recycle_upload_chunk(ring, 10, 0) && ring.used == 0 && ring.epoch == -1, "slack 0 recycles at the completed frame");
    check(reserve_upload_chunk(ring, 4 * kAlign, 12).mapped == storage_a && ring.epoch == 12, "a recycled chunk hands out its start again");
    check(!recycle_upload_chunk(ring, 12, 1) && !recycle_upload_chunk(ring, 11, 0), "a newer reservation keeps the chunk");
    check(recycle_upload_chunk(ring, 13, 1) && ring.used == 0, "one frame of slack passes");
    UploadRingChunk idle{nullptr, storage_b, sizeof(storage_b), 0, -1};
    check(!recycle_upload_chunk(idle, 100, 1), "an unused chunk is left alone");

    std::puts("upload_reservation_check: all checks passed");
    return 0;
}
