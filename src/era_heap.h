#pragma once

#include <cstddef>
#include <cstdint>

// Desktop handling of attribute words decoded with the XALLOC_ATTRIBUTES layout
// published by the GDK 261000 xmem.h: objectType:14, pageSize:2, allocatorId:8,
// alignment:5, memoryType:4, an address-below-4-GiB bit at 33 and reserved:30 above it.
// Allocator-ID meaning and GPU visibility of graphics types remain unconfirmed.
// Heap and graphics memory types are served. Physical memory types, undefined
// field values and alignments the desktop cannot honor return a null pointer.
namespace era_heap {
struct Fields {
    std::uint32_t object_type;
    std::uint32_t page_size;     // 0: 4 KB, 1: 64 KB, 2: 2 MB, 3: undefined
    std::uint32_t allocator_id;
    std::uint32_t alignment;     // log2 of the requested alignment
    std::uint32_t memory_type;   // 0: heap, 1-6: graphics, 7-9: physical
    std::uint32_t address_below_4gib;  // 1: the caller asks for an address below 4 GiB
    std::uint32_t reserved;      // 30 bits, must be zero
};

Fields decode(std::uint64_t attributes) noexcept;
void* allocate(std::size_t bytes, std::uint64_t attributes) noexcept;
// Releases a heap-type allocation (memory type 0), as before.
void release(void* allocation) noexcept;
// Releases any allocation given the attributes used to allocate it, as XMemFree callers do.
void release_with_attributes(void* allocation, std::uint64_t attributes) noexcept;
}
