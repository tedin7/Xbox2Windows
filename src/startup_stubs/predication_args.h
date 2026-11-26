// Experimental (docs/startup-runtime.md, #870): checks for the arguments of command list slot 59,
// SetPredication(buffer, offset, operation). Plain standard C++ with no Windows types, so it can be
// compiled and tested on its own.
//
// The desktop list reads one 64-bit value at `offset` of a buffer and skips draws, dispatches and copies
// while the value matches the operation. It rejects an offset that is not a multiple of 8, a value that
// does not fit in the buffer and an operation other than 0 (EQUAL_ZERO) or 1 (NOT_EQUAL_ZERO), and a
// rejected call is a debug layer error. So these are checked first and a bad call is dropped.
#pragma once

#include <cstdint>

namespace x2w_predication {

inline constexpr std::uint64_t value_bytes = 8;

enum class Verdict { clear, set, bad_operation, bad_offset, outside_buffer };

// `has_buffer` false means the title turns predication off; the offset and operation do not matter then.
inline Verdict check(bool has_buffer, std::uint64_t offset, std::uint32_t operation, std::uint64_t buffer_bytes) noexcept
{
    if (!has_buffer) return Verdict::clear;
    if (operation > 1) return Verdict::bad_operation;
    if (offset % value_bytes) return Verdict::bad_offset;
    if (offset > buffer_bytes || buffer_bytes - offset < value_bytes) return Verdict::outside_buffer;
    return Verdict::set;
}

}  // namespace x2w_predication
