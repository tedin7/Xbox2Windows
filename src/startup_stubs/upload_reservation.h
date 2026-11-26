#pragma once

#include <array>
#include <cstdint>

struct ID3D12Resource;

inline constexpr std::uint64_t initial_upload_chunk_bytes = 16ull << 20;
inline constexpr std::uint64_t max_upload_ring_bytes = 256ull << 20;
// Each allocation except possibly the final remainder is at least the initial size.
inline constexpr unsigned max_upload_chunks = max_upload_ring_bytes / initial_upload_chunk_bytes;
using UploadChunkMask = std::uint32_t;
inline constexpr unsigned big_upload_slots = 8;
inline constexpr unsigned upload_slot_count = max_upload_chunks + big_upload_slots;
static_assert(upload_slot_count <= 32);

template <class Fence>
struct UploadQueue {
    Fence* fence = nullptr;
    std::uint64_t value = 0;
    std::array<std::uint64_t, upload_slot_count> last_use{};

    void submitted(UploadChunkMask chunks, bool signaled)
    {
        for (unsigned i = 0; chunks; ++i, chunks >>= 1) {
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
    unsigned pool_slot = big_upload_slots;
};

// Caller holds the lock. A successful reservation owns previously unused bytes.
inline UploadReservation reserve_upload_chunk(UploadRingChunk& chunk, std::uint64_t rounded)
{
    if (chunk.used > chunk.capacity || rounded > chunk.capacity - chunk.used) return {};
    const auto start = chunk.used;
    chunk.used += rounded;
    return {chunk.mapped + start, start, chunk.resource, chunk.generation};
}

template <class Queues>
bool upload_chunk_idle(const UploadRingChunk& chunk, unsigned index, const Queues& queues)
{
    if (chunk.recordings) return false;
    for (const auto& queue : queues)
        if (!queue.second.completed(index)) return false;
    return true;
}

// Called only after the unused tails and the capacity budget have been tried.
template <class Queues>
UploadReservation reuse_upload_chunk(UploadRingChunk& chunk, unsigned index, std::uint64_t rounded, const Queues& queues)
{
    if (rounded > chunk.capacity || !upload_chunk_idle(chunk, index, queues)) return {};
    chunk.used = 0;
    ++chunk.generation;
    return reserve_upload_chunk(chunk, rounded);
}
