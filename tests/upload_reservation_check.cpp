#include "startup_stubs/upload_reservation.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
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

struct TestFence {
    std::uint64_t value = 0;
    std::uint64_t GetCompletedValue() const { return value; }
};

void check_ring_reuse()
{
    // Four 64 MiB reservations fill the 256 MiB budget. The queues run independently.
    constexpr auto bytes = max_upload_ring_bytes / 4;
    std::unique_ptr<unsigned char[]> storage(new unsigned char[max_upload_ring_bytes]);
    std::array<UploadRingChunk, 4> chunks{};
    UploadChunkMask recording = 0;
    for (unsigned i = 0; i < chunks.size(); ++i) {
        chunks[i] = {nullptr, storage.get() + i * bytes, bytes};
        const auto slice = reserve_upload_chunk(chunks[i], bytes);
        check(slice.mapped != nullptr, "fill ring");
        chunks[i].hold(recording, i);
        chunks[i].hold(recording, i);
        check(chunks[i].recordings == 1, "one hold per recording and chunk");
    }

    TestFence a, b;
    std::map<unsigned, UploadQueue<TestFence>> queues{{0, {&a, 1}}, {1, {&b, 1}}};
    for (auto& queue : queues) queue.second.submitted(recording, true);
    a.value = b.value = 1;
    auto& chunk = chunks[0];
    check(!reuse_upload_chunk(chunk, 0, bytes, queues).mapped, "closed list can be resubmitted");

    ++queues[1].value;
    queues[1].submitted(recording, true);
    const auto submitted = recording;
    discard_upload_chunks(recording, chunks.data());  // reset after the second submit on queue b
    check(!reuse_upload_chunk(chunk, 0, bytes, queues).mapped, "latest submit on every queue must complete");
    b.value = UINT64_MAX;
    check(!reuse_upload_chunk(chunk, 0, bytes, queues).mapped, "device removal is not completion");
    b.value = 2;
    const std::uint64_t submitted_slice_generation = chunk.generation;  // what a slice cached before the recycle carries
    const auto next = reuse_upload_chunk(chunk, 0, bytes, queues);
    check(next.mapped == storage.get() && next.offset == 0 && next.generation == 1,
          "320 MiB cumulative reservations reuse completed storage");
    // frame_copy (d3d12_draw.inl) finds a cached slice again only while the chunk still has the slice's generation: a slice
    // made before the recycle (generation 0) names storage that now belongs to a new reservation (generation 1).
    check(next.generation == chunk.generation && submitted_slice_generation != chunk.generation,
          "a slice cached before a recycle no longer matches its chunk");

    queues[0].submitted(submitted, false);
    queues[0].submitted(submitted, true);
    check(!reuse_upload_chunk(chunk, 0, bytes, queues).mapped, "untracked submit prevents reuse");
}

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

    check_ring_reuse();
    std::puts("upload_reservation_check: all checks passed");
    return 0;
}
