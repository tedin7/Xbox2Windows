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
    const auto next = reuse_upload_chunk(chunk, 0, bytes, queues);
    check(next.mapped == storage.get() && next.offset == 0 && next.generation == 1,
          "320 MiB cumulative reservations reuse completed storage");

    queues[0].submitted(submitted, false);
    queues[0].submitted(submitted, true);
    check(!reuse_upload_chunk(chunk, 0, bytes, queues).mapped, "untracked submit prevents reuse");
}

void check_snapshot_pool()
{
    // The snapshot pool's chunks are numbered after the ring's in the same recording mask and the same per-queue table.
    check(snapshot_chunk_base == max_upload_chunks && max_snapshot_chunks == 8 && upload_chunk_bits == 40, "pool numbering");
    unsigned char storage[4 * kAlign];
    std::array<UploadRingChunk, 2> ring{};
    std::array<UploadRingChunk, 2> snap{};
    UploadRingChunk& chunk = snap[1];
    chunk = {nullptr, storage, sizeof(storage), sizeof(storage)};
    UploadChunkMask recording = 0;
    ring[1].hold(recording, 1);
    snap[0].hold(recording, snapshot_chunk_base + 0);
    snap[1].hold(recording, snapshot_chunk_base + 1);
    check(ring[1].recordings == 1 && snap[0].recordings == 1 && snap[1].recordings == 1 && ring[0].recordings == 0, "holds land in their pool");
    TestFence fence;
    std::map<unsigned, UploadQueue<TestFence>> queues{{0, {&fence, 1}}};
    queues[0].submitted(recording, true);
    check(!queues[0].completed(snapshot_chunk_base + 1), "a snapshot chunk in flight is not complete");
    fence.value = 1;
    check(queues[0].completed(snapshot_chunk_base + 1) && queues[0].completed(snapshot_chunk_base + 7), "completed, or never used");
    check(!reuse_upload_chunk(chunk, snapshot_chunk_base + 1, kAlign, queues).mapped, "a held chunk is not reused");
    discard_upload_chunks(recording, ring.data(), snap.data());
    check(recording == 0 && ring[1].recordings == 0 && snap[0].recordings == 0 && snap[1].recordings == 0, "discard releases both pools");
    const auto again = reuse_upload_chunk(chunk, snapshot_chunk_base + 1, kAlign, queues);
    check(again.mapped == storage && again.generation == 1, "a finished, unheld snapshot chunk is reused");
}

// X2W_D3D12_RING_RELEASE_AT_SUBMIT=1 bookkeeping: a closed list stops holding its chunks once its submit is fenced,
// the fence alone then guards the chunks, and a later submit of the same list before its Reset fences the intact
// chunks again and reports the reused ones.
void check_release_at_submit()
{
    constexpr std::uint64_t bytes = 4 * kAlign;
    unsigned char storage[2][bytes];
    std::array<UploadRingChunk, 2> chunks{};
    UploadChunkMask recording = 0;
    for (unsigned i = 0; i < chunks.size(); ++i) {
        chunks[i] = {nullptr, storage[i], bytes};
        check(reserve_upload_chunk(chunks[i], bytes).mapped != nullptr, "release: fill chunk");
        chunks[i].hold(recording, i);
    }

    TestFence fence;
    std::map<unsigned, UploadQueue<TestFence>> queues{{0, {&fence, 1}}};
    queues[0].submitted(recording, true);
    check(!reuse_upload_chunk(chunks[0], 0, bytes, queues).mapped, "release: a closed list holds until released");

    ReleasedUploadChunks released;
    release_upload_chunks_at_submit(recording, chunks.data(), released);
    check(recording == 0 && chunks[0].recordings == 0 && chunks[1].recordings == 0, "release: holds dropped");
    check(released.chunks == 3 && released.generation[0] == 0 && released.generation[1] == 0,
          "release: chunks and generations recorded");
    check(!reuse_upload_chunk(chunks[0], 0, bytes, queues).mapped, "release: the fence still guards the chunk");

    fence.value = 1;
    const auto reused = reuse_upload_chunk(chunks[0], 0, bytes, queues);
    check(reused.mapped == storage[0] && reused.generation == 1, "release: chunk reused after the fence");

    UploadChunkMask recycled = UINT64_MAX;
    const auto intact = resubmitted_upload_chunks(released, chunks.data(), recycled);
    check(intact == 2 && recycled == 1, "release: a resubmit finds chunk 1 intact and chunk 0 reused");

    ++queues[0].value;
    queues[0].submitted(intact, true);
    check(!reuse_upload_chunk(chunks[1], 1, bytes, queues).mapped, "release: the resubmit fences the intact chunk");
    fence.value = 2;
    check(reuse_upload_chunk(chunks[1], 1, bytes, queues).mapped != nullptr,
          "release: the intact chunk is reused after the resubmit's fence");
    check(resubmitted_upload_chunks(released, chunks.data(), recycled) == 0 && recycled == 3,
          "release: both chunks reported once reused");
}

void check_snapshot_release_at_submit()
{
    constexpr unsigned ring_index = max_upload_chunks - 1;
    constexpr unsigned snapshot_index = upload_chunk_bits - 1;
    constexpr auto ring_bit = UploadChunkMask{1} << ring_index;
    constexpr auto snapshot_bit = UploadChunkMask{1} << snapshot_index;
    constexpr auto both = ring_bit | snapshot_bit;
    unsigned char storage[2][kAlign];
    std::array<UploadRingChunk, max_upload_chunks> ring{};
    std::array<UploadRingChunk, max_snapshot_chunks> snapshots{};
    auto& ring_chunk = ring.back();
    auto& snapshot = snapshots.back();
    ring_chunk = {nullptr, storage[0], kAlign, kAlign, 2};
    snapshot = {nullptr, storage[1], kAlign, kAlign, 3};
    UploadChunkMask recording = 0;
    ring_chunk.hold(recording, ring_index);
    snapshot.hold(recording, snapshot_index);
    check(recording == both, "snapshot release: both boundary bits are recorded");

    TestFence first, second;
    std::map<unsigned, UploadQueue<TestFence>> queues{{0, {&first, 1}}, {1, {&second, 1}}};
    queues[0].submitted(recording, true);
    ReleasedUploadChunks released;
    release_upload_chunks_at_submit(recording, ring.data(), released, snapshots.data());
    check(!recording && !ring_chunk.recordings && !snapshot.recordings, "snapshot release: both holds dropped");
    check(released.chunks == both && released.generation[ring_index] == 2 && released.generation[snapshot_index] == 3,
          "snapshot release: both generations saved");
    check(!reuse_upload_chunk(snapshot, snapshot_index, kAlign, queues).mapped,
          "snapshot release: first fence guards the snapshot after release");

    UploadChunkMask recycled = UINT64_MAX;
    const auto intact = resubmitted_upload_chunks(released, ring.data(), recycled, snapshots.data());
    check(intact == both && !recycled, "snapshot release: intact resubmit includes both pools");
    queues[1].submitted(intact, true);
    first.value = 1;
    check(!reuse_upload_chunk(snapshot, snapshot_index, kAlign, queues).mapped,
          "snapshot release: the other queue's resubmit fence still guards reuse");
    second.value = 1;
    const auto reused = reuse_upload_chunk(snapshot, snapshot_index, kAlign, queues);
    check(reused.mapped == storage[1] && reused.generation == 4, "snapshot release: reuse after both fences");
    check(resubmitted_upload_chunks(released, ring.data(), recycled, snapshots.data()) == ring_bit &&
          recycled == snapshot_bit, "snapshot release: a recycled snapshot alone invalidates the recording");
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
    check_snapshot_pool();
    check_release_at_submit();
    check_snapshot_release_at_submit();
    std::puts("upload_reservation_check: all checks passed");
    return 0;
}
