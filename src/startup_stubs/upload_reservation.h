#pragma once

#include <array>
#include <cstdint>
#include <limits>

struct ID3D12Resource;

inline constexpr std::uint64_t initial_upload_chunk_bytes = 16ull << 20;
inline constexpr std::uint64_t max_upload_ring_bytes = 256ull << 20;
// X2W_D3D12_RING_RELEASE_AT_SUBMIT=1: chunks of the initial size, up to this total (all of them regular chunks).
inline constexpr std::uint64_t max_released_upload_ring_bytes = 512ull << 20;
// Each pool grows in chunks of at least the initial size, except for its final remainder.
inline constexpr unsigned max_regular_upload_chunks = max_upload_ring_bytes / initial_upload_chunk_bytes;
inline constexpr unsigned max_upload_chunks = 2 * max_regular_upload_chunks;
// Title input snapshots have a pool of their own, built like the ring (chunks reused once no recording can submit them and every
// queue that used them has finished), so draw and texture uploads cannot starve them and they cannot starve the ring. Its chunks
// are numbered after the ring's in the same recording mask: bit snapshot_chunk_base + i is chunk i of the snapshot pool.
inline constexpr std::uint64_t max_snapshot_pool_bytes = 128ull << 20;
inline constexpr unsigned max_snapshot_chunks = max_snapshot_pool_bytes / initial_upload_chunk_bytes;
inline constexpr unsigned snapshot_chunk_base = max_upload_chunks;
inline constexpr unsigned upload_chunk_bits = max_upload_chunks + max_snapshot_chunks;
using UploadChunkMask = std::uint64_t;
// Dedicated large upload buffers have their own mask (bit i = buffer i); their last_use slots follow the ring and snapshot chunks.
inline constexpr unsigned max_large_upload_buffers = 32;
inline constexpr unsigned large_upload_base = upload_chunk_bits;
static_assert(upload_chunk_bits <= std::numeric_limits<UploadChunkMask>::digits);
static_assert(max_released_upload_ring_bytes / initial_upload_chunk_bytes <= max_upload_chunks);
static_assert(max_large_upload_buffers <= std::numeric_limits<UploadChunkMask>::digits);

template <class Fence>
struct UploadQueue {
    Fence* fence = nullptr;
    std::uint64_t value = 0;
    std::array<std::uint64_t, upload_chunk_bits + max_large_upload_buffers> last_use{};

    void submitted(UploadChunkMask chunks, bool signaled, unsigned first = 0)
    {
        for (unsigned i = first; chunks; ++i, chunks >>= 1) {
            if ((chunks & 1) && last_use[i] != UINT64_MAX)
                last_use[i] = signaled ? value : UINT64_MAX;
        }
    }

    bool completed(unsigned index) const
    {
        const auto target = last_use[index];
        if (!target) return true;
        if (target == UINT64_MAX) return false;
        const auto completed = fence->GetCompletedValue();
        return completed != UINT64_MAX && completed >= target;
    }
};

struct UploadRingChunk {
    ID3D12Resource* resource = nullptr;
    unsigned char* mapped = nullptr;
    std::uint64_t capacity = 0;
    std::uint64_t used = 0;
    std::uint64_t generation = 0;
    unsigned recordings = 0;
    bool overflow = false;

    void hold(UploadChunkMask& recording, unsigned index)
    {
        const auto bit = UploadChunkMask{1} << index;
        if (!(recording & bit)) {
            recording |= bit;
            ++recordings;
        }
    }
};

// Indices below snapshot_chunk_base belong to the upload ring.
template <class Chunk>
inline Chunk& upload_chunk_at(unsigned index, Chunk* chunks, Chunk* snapshot_chunks)
{
    return index < snapshot_chunk_base ? chunks[index] : snapshot_chunks[index - snapshot_chunk_base];
}
inline void discard_upload_chunks(UploadChunkMask& recording, UploadRingChunk* chunks, UploadRingChunk* snapshot_chunks = nullptr)
{
    for (unsigned i = 0; recording; ++i, recording >>= 1)
        if (recording & 1) --upload_chunk_at(i, chunks, snapshot_chunks).recordings;
}

// X2W_D3D12_RING_RELEASE_AT_SUBMIT=1: the chunks a closed list stopped holding when its submit was fenced, and
// each chunk's generation at that moment. The queue fence keeps those chunks from reuse until the GPU passes it.
struct ReleasedUploadChunks {
    UploadChunkMask chunks = 0;
    std::array<std::uint64_t, upload_chunk_bits> generation{};
};

// Caller holds the lock, after the queue recorded a signalled fence for the recording's chunks.
inline void release_upload_chunks_at_submit(UploadChunkMask& recording, UploadRingChunk* chunks,
                                            ReleasedUploadChunks& released, UploadRingChunk* snapshot_chunks = nullptr)
{
    for (unsigned i = 0; recording; ++i, recording >>= 1) {
        if (!(recording & 1)) continue;
        released.chunks |= UploadChunkMask{1} << i;
        auto& chunk = upload_chunk_at(i, chunks, snapshot_chunks);
        released.generation[i] = chunk.generation;
        --chunk.recordings;
    }
}

// Caller holds the lock. A closed list submitted again before its Reset still reads the bytes it reserved: returns
// the released chunks that were not reused since, which the new submit must fence again, and sets `recycled` to
// those that were.
inline UploadChunkMask resubmitted_upload_chunks(const ReleasedUploadChunks& released, const UploadRingChunk* chunks,
                                                 UploadChunkMask& recycled, const UploadRingChunk* snapshot_chunks = nullptr)
{
    UploadChunkMask intact = 0;
    recycled = 0;
    for (unsigned i = 0; i < upload_chunk_bits; ++i) {
        const auto bit = UploadChunkMask{1} << i;
        if (!(released.chunks & bit)) continue;
        (upload_chunk_at(i, chunks, snapshot_chunks).generation == released.generation[i] ? intact : recycled) |= bit;
    }
    return intact;
}

struct UploadReservation {
    unsigned char* mapped = nullptr;
    std::uint64_t offset = 0;
    ID3D12Resource* resource = nullptr;
    std::uint64_t generation = 0;
    bool dedicated = false;
};

// Caller holds the lock. A successful reservation owns previously unused bytes.
inline UploadReservation reserve_upload_chunk(UploadRingChunk& chunk, std::uint64_t rounded)
{
    if (chunk.used > chunk.capacity || rounded > chunk.capacity - chunk.used) return {};
    const auto start = chunk.used;
    chunk.used += rounded;
    return {chunk.mapped + start, start, chunk.resource, chunk.generation};
}

// Rewind only after all CPU holds and GPU uses have ended.
template <class Queues>
UploadReservation reuse_upload_chunk(UploadRingChunk& chunk, unsigned fence_index, std::uint64_t rounded, const Queues& queues)
{
    if (rounded > chunk.capacity || chunk.recordings) return {};
    for (const auto& queue : queues)
        if (!queue.second.completed(fence_index)) return {};
    chunk.used = 0;
    ++chunk.generation;
    return reserve_upload_chunk(chunk, rounded);
}
