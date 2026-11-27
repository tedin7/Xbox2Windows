#pragma once

#include <array>
#include <cstdint>

struct ID3D12Resource;

inline constexpr std::uint64_t initial_upload_chunk_bytes = 16ull << 20;
inline constexpr std::uint64_t max_upload_ring_bytes = 256ull << 20;
// Each pool grows in chunks of at least the initial size, except for its final remainder.
inline constexpr unsigned max_regular_upload_chunks = max_upload_ring_bytes / initial_upload_chunk_bytes;
inline constexpr unsigned max_upload_chunks = 2 * max_regular_upload_chunks;
using UploadChunkMask = std::uint32_t;
inline constexpr unsigned max_large_upload_buffers = 32;
static_assert(max_upload_chunks <= 32);
static_assert(max_large_upload_buffers <= 32);

template <class Fence>
struct UploadQueue {
    Fence* fence = nullptr;
    std::uint64_t value = 0;
    std::array<std::uint64_t, max_upload_chunks + max_large_upload_buffers> last_use{};

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

inline void discard_upload_chunks(UploadChunkMask& recording, UploadRingChunk* chunks)
{
    for (unsigned i = 0; recording; ++i, recording >>= 1)
        if (recording & 1) --chunks[i].recordings;
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
