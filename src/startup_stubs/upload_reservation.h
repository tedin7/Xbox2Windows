#pragma once

#include <cstdint>

struct ID3D12Resource;

struct UploadRingChunk {
    ID3D12Resource* resource = nullptr;
    unsigned char* mapped = nullptr;
    std::uint64_t capacity = 0;
    std::uint64_t used = 0;
    std::int64_t epoch = -1;  // newest frame that reserved from this chunk; -1 while unused
};

struct UploadReservation {
    unsigned char* mapped = nullptr;
    std::uint64_t offset = 0;
    ID3D12Resource* resource = nullptr;
};

// Caller holds the lock. A successful reservation owns previously unused bytes.
inline UploadReservation reserve_upload_chunk(UploadRingChunk& chunk, std::uint64_t rounded)
{
    if (chunk.used > chunk.capacity || rounded > chunk.capacity - chunk.used) return {};
    const auto start = chunk.used;
    chunk.used += rounded;
    return {chunk.mapped + start, start, chunk.resource};
}

// Recycling (X2W_UPLOAD_RING_RECYCLE=1): a chunk whose newest reservation is older than the last frame whose GPU work is
// known complete, with `slack` frames of margin, has no live reader left, so its cursor goes back to the start.
// Caller holds the lock. Returns true when the chunk was reset.
inline bool recycle_upload_chunk(UploadRingChunk& chunk, std::int64_t completed_frame, std::int64_t slack)
{
    if (!chunk.used || chunk.epoch < 0 || completed_frame < 0 || chunk.epoch + slack > completed_frame) return false;
    chunk.used = 0;
    chunk.epoch = -1;
    return true;
}

// reserve_upload_chunk that also stamps the chunk with the frame of the reservation.
inline UploadReservation reserve_upload_chunk(UploadRingChunk& chunk, std::uint64_t rounded, std::int64_t frame)
{
    const auto reservation = reserve_upload_chunk(chunk, rounded);
    if (reservation.mapped && frame > chunk.epoch) chunk.epoch = frame;
    return reservation;
}
