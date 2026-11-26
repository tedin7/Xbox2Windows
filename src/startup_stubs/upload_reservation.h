#pragma once

#include <cstdint>

struct ID3D12Resource;

struct UploadRingChunk {
    ID3D12Resource* resource = nullptr;
    unsigned char* mapped = nullptr;
    std::uint64_t capacity = 0;
    std::uint64_t used = 0;
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
