// Names the reason a BCPack request fails (Refs #1198). decode_bcpack answers true or false; this header
// runs the same lane decode on the cold path with a LaneReport and returns the class of failure plus
// what the lanes had done when it stopped. It does not change decode_bcpack or any request path.
//
// The class is a reading of lane state, not a proof about the source data. A stream cut exactly at a
// fragment boundary in every lane reads as request_longer_than_stream, because it is a shorter stream.
// Every format decode_bcpack decodes (BC1, BC3, BC4, BC5 and BC7) has a lane layout here, the same alphabets and
// symbols per block; add a layout entry when decode_bcpack gains a format. The BC1, BC3, BC4 and BC5 block
// stages fail only on a destination that is too small, which this header never passes, so only BC7 has a
// control_code_unmapped class.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bcpack_decoder.h"

namespace x2w_bcpack {

enum class InspectClass : uint8_t {
    ok,
    bad_arguments,               // null or empty input, empty or oversized destination, or no lane layout fits
    unsupported_format,          // a format number without a lane layout in this header
    destination_misaligned,      // destination size is not a whole number of blocks
    header_invalid,              // a fragment header names an alphabet wider than 8 bits
    fragment_not_home,           // a flushed fragment did not end at the rANS home state
    end_state_invalid,           // all lanes finished in a state that is not a terminal state
    trailing_input,              // all lanes finished and a non-zero input byte is left
    request_shorter_than_stream, // every lane reached the requested count and the decode still did not finish
    request_longer_than_stream,  // all input read, every lane stopped at a fragment boundary short of the count
    truncated_in_fragment,       // all input read, a lane stopped inside a fragment short of the count
    cycle_limit,                 // no progress for 64 cycles in another state, or the cycle limit
    control_code_unmapped,       // lanes decoded; a BC7 block names an unmapped control code
};

struct InspectResult {
    InspectClass cls = InspectClass::bad_arguments;
    size_t bytes_read = 0;           // input bytes the lanes had taken when the decode ended
    uint32_t min_symbols_per_lane = 0;
    uint32_t max_symbols_per_lane = 0;
    uint32_t target_symbols_per_lane = 0;  // what the destination size asked for
};

// Largest destination inspected: the 16 MiB decoded-chunk cap of the chunk-window experiment. Lane symbols cost
// (output / block_bytes) * symbols_per_block * lanes * 2 bytes, at most 4x the output for BC7 (64 MiB at the
// cap). A declared size above the cap returns bad_arguments before anything is allocated.
inline constexpr size_t inspect_max_output = size_t{16} << 20;

struct InspectLayout {
    unsigned format;
    size_t lanes;
    uint8_t alphabet_bits[16];
    size_t block_bytes;
    size_t symbols_per_block;  // per lane
};

inline constexpr InspectLayout INSPECT_LAYOUTS[] = {
    { 1, 12, { 7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7 }, 8, 1 },
    { 7, 16, { 3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4 }, 16, 2 },
    { 3, 12, { 3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7 }, 16, 2 },
    { 4, 12, { 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5 }, 8, 1 },
    { 5, 12, { 3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5 }, 16, 2 },
};

inline const InspectLayout* inspect_layout(unsigned format) {
    for (const InspectLayout& layout : INSPECT_LAYOUTS) {
        if (layout.format == format) return &layout;
    }
    return nullptr;
}

inline InspectClass inspect_class(LaneFailure failure) {
    switch (failure) {
    case LaneFailure::none: return InspectClass::ok;
    case LaneFailure::bad_arguments: return InspectClass::bad_arguments;
    case LaneFailure::header_invalid: return InspectClass::header_invalid;
    case LaneFailure::fragment_not_home: return InspectClass::fragment_not_home;
    case LaneFailure::end_state_invalid: return InspectClass::end_state_invalid;
    case LaneFailure::trailing_input: return InspectClass::trailing_input;
    case LaneFailure::request_shorter: return InspectClass::request_shorter_than_stream;
    case LaneFailure::request_longer: return InspectClass::request_longer_than_stream;
    case LaneFailure::truncated: return InspectClass::truncated_in_fragment;
    case LaneFailure::no_progress: return InspectClass::cycle_limit;
    }
    return InspectClass::cycle_limit;
}

inline const char* inspect_class_name(InspectClass cls) {
    switch (cls) {
    case InspectClass::ok: return "ok";
    case InspectClass::bad_arguments: return "bad_arguments";
    case InspectClass::unsupported_format: return "unsupported_format";
    case InspectClass::destination_misaligned: return "destination_misaligned";
    case InspectClass::header_invalid: return "header_invalid";
    case InspectClass::fragment_not_home: return "fragment_not_home";
    case InspectClass::end_state_invalid: return "end_state_invalid";
    case InspectClass::trailing_input: return "trailing_input";
    case InspectClass::request_shorter_than_stream: return "request_shorter_than_stream";
    case InspectClass::request_longer_than_stream: return "request_longer_than_stream";
    case InspectClass::truncated_in_fragment: return "truncated_in_fragment";
    case InspectClass::cycle_limit: return "cycle_limit";
    case InspectClass::control_code_unmapped: return "control_code_unmapped";
    }
    return "unknown";
}

// Decodes the request like decode_bcpack would and reports the class. For a stream that decodes it
// returns ok with the bytes the lanes took. Allocates a scratch destination: call it on failures only.
inline InspectResult inspect_bcpack(const unsigned char* input, size_t input_size, unsigned bc_format,
                                    size_t output_size) {
    InspectResult result;
    if (!input || input_size == 0 || output_size == 0 || output_size > inspect_max_output) return result;
    const InspectLayout* layout = inspect_layout(bc_format);
    if (!layout) {
        result.cls = InspectClass::unsupported_format;
        return result;
    }
    if (output_size % layout->block_bytes != 0) {
        result.cls = InspectClass::destination_misaligned;
        return result;
    }
    const size_t num_blocks = output_size / layout->block_bytes;
    LaneReport report;
    std::vector<uint16_t> symbols;
    const bool lanes_ok = decode_lanes(input, input_size, layout->alphabet_bits, layout->lanes,
                                       num_blocks * layout->symbols_per_block, symbols, &report);
    result.bytes_read = report.bytes_read;
    result.min_symbols_per_lane = report.min_symbols;
    result.max_symbols_per_lane = report.max_symbols;
    result.target_symbols_per_lane = report.target_symbols;
    if (!lanes_ok) {
        result.cls = inspect_class(report.failure);
        return result;
    }
    result.cls = InspectClass::ok;
    if (bc_format == 7) {
        std::vector<unsigned char> scratch(output_size);
        if (!decode_bc7_blocks(symbols, num_blocks, scratch.data(), scratch.size())) {
            result.cls = InspectClass::control_code_unmapped;
        }
    }
    return result;
}

} // namespace x2w_bcpack
