// BCPack decoder for Xbox Series DirectStorage textures.
// Adapted from bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com).
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// New BC1/2/6H reconstruction in bcpack_blocks_extra.inl is an original MIT contribution.
// Incorporates rANS interval models derived from Microsoft silicon-rans (MIT License).
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <utility>
#include <algorithm>
#include <intrin.h>
#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#include <immintrin.h>
#endif

#include "bcpack_tables.h"

namespace x2w_bcpack {

inline void model_forward(uint8_t eab, uint8_t q, uint16_t sym, uint32_t& f, uint32_t& cf) {
    uint32_t s = sym;
    unsigned long seg = 0;
    if (s != 0) {
        _BitScanReverse(&seg, s);
        seg += 1;
    }
    uint32_t st = (seg == 0) ? 0 : (1u << (seg - 1));
    uint32_t fa = FREQ_TABLE[eab][q][seg];
    uint32_t cfa = (BASE_TABLE[eab][q][seg] + fa * (s - st)) & 0xFFFF;
    if (cfa == 0) {
        f = (fa - 1) & 0xFFFF;
        cf = 1;
    } else {
        f = fa;
        cf = cfa;
    }
}

inline uint32_t model_advance(uint8_t eab, uint8_t q, uint32_t x, uint16_t sym) {
    uint32_t f, cf;
    model_forward(eab, q, sym, f, cf);
    return f * (x >> 16) + (x & 0xFFFF) - cf;
}

inline uint16_t model_inverse_fast(uint8_t eab, uint8_t q, uint16_t cf) {
    uint32_t c = cf;
    uint32_t seg = 0;
    uint32_t nseg = eab + 2;
    for (uint32_t k = 1; k < nseg; ++k) {
        if (c >= BASE_TABLE[eab][q][k]) {
            seg = k;
        } else {
            break;
        }
    }
    uint32_t base_x = (seg == 0) ? 0 : (1u << (seg - 1));
    uint32_t dist = (c - BASE_TABLE[eab][q][seg] + RN_TABLE[eab][q][seg]) & 0xFFFF;
    uint32_t mul = dist * RF_TABLE[eab][q][seg];
    uint32_t steps = ((mul >> 15) >> RS_TABLE[eab][q][seg]) & 0x1FF;
    return static_cast<uint16_t>((base_x + steps) & 0x1FF);
}

inline int32_t unzigzag(uint16_t v) {
    int32_t iv = static_cast<int32_t>(v);
    return (iv & 1) ? -((iv + 1) >> 1) : (iv >> 1);
}

inline uint8_t residual(uint16_t sym, uint32_t prec) {
    int32_t unz = unzigzag(sym);
    return static_cast<uint8_t>(unz & ((int32_t{1} << prec) - 1));
}

inline uint8_t interleave(uint16_t even, uint16_t odd) {
    uint8_t out = 0;
    for (int k = 0; k < 4; ++k) {
        out |= static_cast<uint8_t>(((even >> k) & 1) << (2 * k));
        out |= static_cast<uint8_t>(((odd >> k) & 1) << (2 * k + 1));
    }
    return out;
}

inline void deinterleave(uint8_t b, uint16_t& even, uint16_t& odd) {
    even = 0;
    odd = 0;
    for (int k = 0; k < 4; ++k) {
        even |= static_cast<uint16_t>(((b >> (2 * k)) & 1) << k);
        odd |= static_cast<uint16_t>(((b >> (2 * k + 1)) & 1) << k);
    }
}

inline void index_pair(uint16_t sym, size_t row, uint16_t& left, uint16_t& right) {
    uint16_t even, odd;
    deinterleave(static_cast<uint8_t>(sym & 0xFF), even, odd);
    if (row < 3) {
        left = odd;
        right = even;
    } else {
        left = even;
        right = odd;
    }
}

enum LaneState : uint8_t {
    LANE_IDLE = 0,
    LANE_HDR0 = 1,
    LANE_HDR1 = 2,
    LANE_HDR2 = 3,
    LANE_PROC = 4,
    LANE_DRAIN = 5,
};

struct Lane {
    uint8_t ab = 0;
    uint32_t target = 0;
    LaneState state = LANE_IDLE;
    uint8_t ib = 0;
    bool ib_full = false;
    uint16_t sym = 0;
    bool sym_full = false;
    uint32_t in_rem = 0;
    uint32_t out_rem = 0;
    uint32_t p0 = 0;
    uint32_t p1 = 0;
    bool hdr3 = false;
    bool flush = false;
    uint8_t virtual_zeros = 0;
    uint8_t q = 0;
    uint8_t eab = 0;
    bool done = false;
    bool go = true;
    bool stall = false;
    uint32_t nsyms = 0;
    uint32_t nbytes = 0;
    uint32_t nfrags = 0;
    uint32_t final_state = 0;
};

inline bool phase1(Lane& ln) {
    switch (ln.state) {
    case LANE_IDLE:
        if (ln.go) {
            ln.go = false;
            ln.state = LANE_HDR0;
            ln.out_rem = ln.target;
            ln.p0 = 0;
            ln.p1 = 0;
        }
        break;
    case LANE_HDR0:
        if (ln.ib_full) {
            uint8_t b = ln.ib;
            ln.ib_full = false;
            ln.in_rem = 0;
            ln.state = LANE_HDR1;
            ln.virtual_zeros = 0;
            ln.q = b & 15;
            ln.hdr3 = (b & 0x20) != 0;
            ln.flush = (b & 0x10) != 0;
            int32_t e = (static_cast<int32_t>(ln.ab) - static_cast<int32_t>(b >> 6)) & 15;
            ln.eab = (e < 10) ? static_cast<uint8_t>(e) : 9;
            ln.nfrags++;
            if (ln.eab > 8) return false;
        }
        break;
    case LANE_HDR1:
        if (ln.ib_full) {
            ln.ib_full = false;
            if (ln.hdr3) {
                ln.in_rem = ln.ib;
                ln.state = LANE_HDR2;
            } else {
                ln.in_rem = static_cast<uint32_t>(ln.ib) + 1;
                ln.state = LANE_PROC;
            }
        }
        break;
    case LANE_HDR2:
        if (ln.ib_full) {
            ln.ib_full = false;
            ln.in_rem = ((static_cast<uint32_t>(ln.ib) << 8) | (ln.in_rem & 0xFF)) + 257;
            ln.state = LANE_PROC;
        }
        break;
    default: {
        bool is_proc = (ln.state == LANE_PROC);
        bool will = is_proc && (ln.p0 < 0x1000000) && (ln.in_rem > 0) && ln.ib_full;
        uint32_t x = will ? (((ln.p0 & 0xFFFFFF) << 8) | ln.ib) : ln.p0;
        uint32_t remaining = ln.in_rem - (will ? 1 : 0);
        if (ln.flush && remaining == 0) {
            while (x > 0 && x < 0x1000000 && ln.virtual_zeros < 2) {
                x <<= 8;
                ln.virtual_zeros++;
            }
        }
        bool nsf = (x > 0x1000000) && (ln.out_rem > 0);
        if (nsf) {
            ln.out_rem--;
            ln.sym = model_inverse_fast(ln.eab, ln.q, static_cast<uint16_t>(x & 0xFFFF));
        }
        uint32_t new_out = ln.out_rem;
        ln.p1 = x;
        ln.sym_full = nsf;
        if (is_proc) {
            if (will) {
                ln.in_rem--;
                ln.ib_full = false;
            }
            if (ln.in_rem == 0) {
                if (new_out > 0) {
                    if (x == 0x1000000) {
                        ln.state = LANE_HDR0;
                        ln.virtual_zeros = 0;
                        if (ln.flush) ln.p1 = 0;
                    } else {
                        ln.state = LANE_DRAIN;
                    }
                } else {
                    ln.done = true;
                    ln.state = LANE_IDLE;
                    ln.final_state = nsf ? model_advance(ln.eab, ln.q, x, ln.sym) : x;
                }
            }
        } else if (x <= 0x1000000) {
            if (new_out > 0) {
                // A flushed fragment must return to the rANS home state (0x1000000).
                // Merely being <= L could silently accept a damaged fragment.
                if (ln.flush && x != 0x1000000) return false;
                ln.state = LANE_HDR0;
                ln.virtual_zeros = 0;
                if (ln.flush) {
                    ln.p0 = 0;
                    ln.p1 = 0;
                }
            } else {
                ln.state = LANE_IDLE;
                ln.done = true;
                ln.final_state = nsf ? model_advance(ln.eab, ln.q, x, ln.sym) : x;
            }
        }
        break;
    }
    }
    return true;
}

// Why decode_lanes returned false, for inspect_bcpack (bcpack_inspect.h). Filled only when a report is passed.
enum class LaneFailure : uint8_t {
    none,
    bad_arguments,       // no lanes, more than 16, a zero target or an alphabet wider than 8 bits
    header_invalid,      // a fragment header names an alphabet wider than 8 bits
    fragment_not_home,   // a flushed fragment did not end at the rANS home state
    end_state_invalid,   // all lanes finished in a state that is not a terminal state
    trailing_input,      // all lanes finished and a non-zero input byte is left
    request_shorter,     // every lane reached its target and the decode still did not finish
    request_longer,      // all input read, every lane stopped at a fragment boundary before its target
    truncated,           // all input read, a lane stopped inside a fragment before its target
    no_progress,         // no progress for 64 cycles in another state, or the cycle limit
};

struct LaneReport {
    LaneFailure failure = LaneFailure::none;
    size_t bytes_read = 0;        // input bytes handed to lanes when the decode ended
    uint32_t min_symbols = 0;     // fewest and most symbols any lane had produced
    uint32_t max_symbols = 0;
    uint32_t target_symbols = 0;  // symbols per lane that were requested
};

// The original cycle loop. Without a report decode_lanes uses a faster path; with one it runs this loop,
// which names the failure. Tests compare every path with it.
inline bool decode_lanes_reference(const uint8_t* input, size_t input_size,
                                   const uint8_t* alphabet_bits, size_t num_lanes,
                                   size_t target_symbols_per_lane,
                                   std::vector<uint16_t>& symbols, LaneReport* report = nullptr) {
    if (report) *report = LaneReport{};
    if (num_lanes == 0 || num_lanes > 16 || target_symbols_per_lane == 0) {
        if (report) report->failure = LaneFailure::bad_arguments;
        return false;
    }
    for (size_t i = 0; i < num_lanes; ++i) {
        if (alphabet_bits[i] > 8) {
            if (report) report->failure = LaneFailure::bad_arguments;
            return false;
        }
    }
    symbols.assign(num_lanes * target_symbols_per_lane, 0);

    std::vector<Lane> lanes(num_lanes);
    for (size_t i = 0; i < num_lanes; ++i) {
        lanes[i].ab = alphabet_bits[i];
        lanes[i].target = static_cast<uint32_t>(target_symbols_per_lane);
    }

    uint32_t slots = 0;
    size_t pos = 0;
    uint32_t exhausted_mask = 0;
    uint64_t cycle = 0;
    uint64_t max_cycles = 64ull * (num_lanes * target_symbols_per_lane + input_size) + 1000;
    uint64_t total_syms = 0;
    size_t last_pos = SIZE_MAX;
    uint64_t last_total_syms = UINT64_MAX;
    uint32_t last_state_sum = UINT32_MAX;
    uint64_t last_change = 0;

    // Records why and where the decode stopped; nothing happens without a report.
    auto note = [&](LaneFailure failure) {
        if (!report) return;
        report->failure = failure;
        report->bytes_read = pos;
        report->target_symbols = static_cast<uint32_t>(target_symbols_per_lane);
        report->min_symbols = lanes[0].nsyms;
        report->max_symbols = lanes[0].nsyms;
        for (const Lane& ln : lanes) {
            if (ln.nsyms < report->min_symbols) report->min_symbols = ln.nsyms;
            if (ln.nsyms > report->max_symbols) report->max_symbols = ln.nsyms;
        }
    };
    // A stall: the request and the stream disagree about the symbol count, or the stream is damaged.
    auto stall_failure = [&]() {
        bool met = true, at_boundary = true;
        for (const Lane& ln : lanes) {
            met &= ln.nsyms == target_symbols_per_lane;
            at_boundary &= ln.state == LANE_HDR0 && !ln.ib_full;
        }
        if (met) return LaneFailure::request_shorter;
        if (!met && pos == input_size) return at_boundary ? LaneFailure::request_longer : LaneFailure::truncated;
        return LaneFailure::no_progress;
    };

    while (cycle < max_cycles) {
        size_t served = 0;
        for (size_t i = 0; i < num_lanes; ++i) {
            Lane& ln = lanes[i];
            uint32_t bit = 1u << i;
            bool blocked_out = ln.sym_full && ((slots & bit) != 0);
            bool need = !ln.ib_full && ((ln.state >= LANE_HDR0 && ln.state <= LANE_HDR2) ||
                                       (ln.state == LANE_PROC && ln.in_rem > 0));
            bool valid = false;
            uint8_t byte = 0;
            if (need && !blocked_out && served < 8 && (exhausted_mask & bit) == 0) {
                if (pos < input_size) {
                    valid = true;
                    byte = input[pos];
                } else {
                    exhausted_mask |= bit;
                }
            }
            ln.stall = (need && !valid) || blocked_out;
            if (ln.stall) continue;
            if (ln.state == LANE_PROC || ln.state == LANE_DRAIN) {
                ln.p0 = ln.sym_full ? model_advance(ln.eab, ln.q, ln.p1, ln.sym) : ln.p1;
            }
            if (valid) {
                ln.ib = byte;
                ln.ib_full = true;
                pos++;
                served++;
                ln.nbytes++;
            }
            if (ln.sym_full) {
                slots |= bit;
                symbols[i * target_symbols_per_lane + ln.nsyms] = ln.sym;
                ln.nsyms++;
                total_syms++;
                ln.sym_full = false;
            }
        }

        uint32_t active = 0;
        bool finished = true;
        uint32_t state_sum = 0;
        for (size_t i = 0; i < num_lanes; ++i) {
            Lane& ln = lanes[i];
            if (!ln.stall) {
                if (!phase1(ln)) {
                    note(ln.eab > 8 ? LaneFailure::header_invalid : LaneFailure::fragment_not_home);
                    return false;
                }
            }
            uint32_t bit = 1u << i;
            if (!(ln.done && !ln.sym_full && (slots & bit) == 0)) {
                active |= bit;
            }
            finished &= (ln.done && !ln.sym_full);
            state_sum += ln.state;
        }

        if (slots != 0 && (slots & active) == active) {
            slots = 0;
        }
        cycle++;
        if (finished) {
            for (size_t i = 0; i < num_lanes; ++i) {
                uint32_t f = lanes[i].final_state;
                if (f != 0x1000000 && f != 0x10000 && f != 0x100) {
                    note(LaneFailure::end_state_invalid);
                    return false;
                }
            }
            for (size_t p = pos; p < input_size; ++p) {
                if (input[p] != 0) {
                    note(LaneFailure::trailing_input);
                    return false;
                }
            }
            if (report) note(LaneFailure::none);
            return true;
        }

        if (pos != last_pos || total_syms != last_total_syms || state_sum != last_state_sum) {
            last_pos = pos;
            last_total_syms = total_syms;
            last_state_sum = state_sum;
            last_change = cycle;
        } else if (cycle - last_change > 64) {
            note(stall_failure());
            return false;
        }
    }
    note(LaneFailure::no_progress);
    return false;
}

// model_inverse_fast followed by model_forward of its result. The enabled BASE entries of a row
// never decrease, so the comparator chain is a count, and the reciprocal divide never leaves the
// segment it starts in. bcpack_lanes_check compares all 9 x 16 x 65536 inputs with the originals.
inline uint16_t model_inverse_interval(uint8_t eab, uint8_t q, uint32_t c, uint32_t& f, uint32_t& cf) {
    const uint16_t* base = BASE_TABLE[eab][q];
    uint32_t seg = 0;
    for (uint32_t k = 1; k < 10; ++k) {
        seg += (k <= eab + 1u) & (c >= base[k]);
    }
    uint32_t dist = (c - base[seg] + RN_TABLE[eab][q][seg]) & 0xFFFF;
    uint32_t steps = (((dist * RF_TABLE[eab][q][seg]) >> 15) >> RS_TABLE[eab][q][seg]) & 0x1FF;
    uint32_t fa = FREQ_TABLE[eab][q][seg];
    uint32_t cfa = (base[seg] + fa * steps) & 0xFFFF;
    f = cfa ? fa : ((fa - 1) & 0xFFFF);
    cf = cfa ? cfa : 1;
    return static_cast<uint16_t>((((1u << seg) >> 1) + steps) & 0x1FF);
}

// Lowest DRAIN state that decodes without a state change. bcpack_lanes_check builds a negative control
// with X2W_BCPACK_LANES_CONTROL, which also sends the home state L down the short path. Define it only
// for a whole test executable: two translation units of one binary that disagree on it would break the
// one-definition rule for every inline function below.
#ifdef X2W_BCPACK_LANES_CONTROL
inline constexpr uint32_t LANE_DRAIN_SHORT_MIN = 0x1000000;
#else
inline constexpr uint32_t LANE_DRAIN_SHORT_MIN = 0x1000001;
#endif

// decode_lanes_reference with both phases in one pass over the lanes. Exact because a lane's
// phase 1 reads and writes only that lane, and a later lane's phase 0 reads only its own state and
// the shared cursor. Steps without a header, fragment end or flush go through a short path; all
// others use phase1. A lane's f and cf are kept from the step that decoded its symbol.
inline bool decode_lanes_scalar(const uint8_t* input, size_t input_size,
                                const uint8_t* alphabet_bits, size_t num_lanes,
                                size_t target_symbols_per_lane,
                                std::vector<uint16_t>& symbols) {
    if (num_lanes == 0 || num_lanes > 16 || target_symbols_per_lane == 0) return false;
    for (size_t i = 0; i < num_lanes; ++i) {
        if (alphabet_bits[i] > 8) return false;
    }
    symbols.assign(num_lanes * target_symbols_per_lane, 0);

    Lane lanes[16];
    uint32_t lane_f[16] = {};
    uint32_t lane_cf[16] = {};
    for (size_t i = 0; i < num_lanes; ++i) {
        lanes[i].ab = alphabet_bits[i];
        lanes[i].target = static_cast<uint32_t>(target_symbols_per_lane);
    }

    uint32_t slots = 0;
    size_t pos = 0;
    uint64_t cycle = 0;
    uint64_t max_cycles = 64ull * (num_lanes * target_symbols_per_lane + input_size) + 1000;
    uint64_t total_syms = 0;
    size_t last_pos = SIZE_MAX;
    uint64_t last_total_syms = UINT64_MAX;
    uint32_t last_state_sum = UINT32_MAX;
    uint64_t last_change = 0;

    while (cycle < max_cycles) {
        size_t served = 0;
        uint32_t active = 0;
        bool finished = true;
        bool failed = false;
        uint32_t state_sum = 0;
        for (size_t i = 0; i < num_lanes; ++i) {
            Lane& ln = lanes[i];
            uint32_t bit = 1u << i;
            bool blocked_out = ln.sym_full && ((slots & bit) != 0);
            bool need = !ln.ib_full && ((ln.state >= LANE_HDR0 && ln.state <= LANE_HDR2) ||
                                       (ln.state == LANE_PROC && ln.in_rem > 0));
            // The reference's exhausted mask is implied: pos never moves back from input_size.
            bool valid = need && !blocked_out && served < 8 && pos < input_size;
            bool stall = (need && !valid) || blocked_out;
            if (!stall) {
                if (ln.state == LANE_PROC || ln.state == LANE_DRAIN) {
                    ln.p0 = ln.sym_full ? lane_f[i] * (ln.p1 >> 16) + (ln.p1 & 0xFFFF) - lane_cf[i] : ln.p1;
                }
                if (valid) {
                    ln.ib = input[pos];
                    ln.ib_full = true;
                    pos++;
                    served++;
                }
                if (ln.sym_full) {
                    slots |= bit;
                    symbols[i * target_symbols_per_lane + ln.nsyms] = ln.sym;
                    ln.nsyms++;
                    total_syms++;
                    ln.sym_full = false;
                }
            }
            // After a failed phase 1 the reference still ran phase 0 for every lane, but no phase 1.
            bool skip = stall || failed;
            bool short_step = skip;
            if (!skip && ln.state == LANE_PROC) {
                bool will = ln.p0 < 0x1000000 && ln.ib_full;
                if (ln.in_rem > static_cast<uint32_t>(will)) {
                    short_step = true;
                    uint32_t x = will ? ((ln.p0 << 8) | ln.ib) : ln.p0;
                    bool nsf = (x > 0x1000000) && (ln.out_rem > 0);
                    if (nsf) {
                        ln.out_rem--;
                        ln.sym = model_inverse_interval(ln.eab, ln.q, x & 0xFFFF, lane_f[i], lane_cf[i]);
                    }
                    ln.p1 = x;
                    ln.sym_full = nsf;
                    if (will) {
                        ln.in_rem--;
                        ln.ib_full = false;
                    }
                }
            } else if (!skip && ln.state == LANE_DRAIN && ln.p0 >= LANE_DRAIN_SHORT_MIN) {
                short_step = true;
                bool nsf = ln.out_rem > 0;
                if (nsf) {
                    ln.out_rem--;
                    ln.sym = model_inverse_interval(ln.eab, ln.q, ln.p0 & 0xFFFF, lane_f[i], lane_cf[i]);
                }
                ln.p1 = ln.p0;
                ln.sym_full = nsf;
            }
            if (!short_step) {
                if (!phase1(ln)) {
                    failed = true;
                } else if (ln.sym_full) {
                    model_forward(ln.eab, ln.q, ln.sym, lane_f[i], lane_cf[i]);
                }
            }
            if (!(ln.done && !ln.sym_full && (slots & bit) == 0)) {
                active |= bit;
            }
            finished &= (ln.done && !ln.sym_full);
            state_sum += ln.state;
        }
        if (failed) return false;

        if (slots != 0 && (slots & active) == active) {
            slots = 0;
        }
        cycle++;
        if (finished) {
            for (size_t i = 0; i < num_lanes; ++i) {
                uint32_t f = lanes[i].final_state;
                if (f != 0x1000000 && f != 0x10000 && f != 0x100) {
                    return false;
                }
            }
            for (size_t p = pos; p < input_size; ++p) {
                if (input[p] != 0) return false;
            }
            return true;
        }

        if (pos != last_pos || total_syms != last_total_syms || state_sum != last_state_sum) {
            last_pos = pos;
            last_total_syms = total_syms;
            last_state_sum = state_sum;
            last_change = cycle;
        } else if (cycle - last_change > 64) {
            return false;
        }
    }
    return false;
}

#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#define X2W_BCPACK_LANE_SIMD 1

// AVX, AVX2 and POPCNT, and an OS that saves the YMM state.
inline bool lane_avx2_supported() {
    static const bool supported = [] {
        int r[4];
        __cpuid(r, 0);
        if (r[0] < 7) return false;
        __cpuid(r, 1);
        const int popcnt_osxsave_avx = (1 << 23) | (1 << 27) | (1 << 28);
        if ((r[2] & popcnt_osxsave_avx) != popcnt_osxsave_avx) return false;
        if ((_xgetbv(0) & 0x6) != 0x6) return false;  // XMM and YMM state
        __cpuidex(r, 7, 0);
        return (r[1] & (1 << 5)) != 0;                // AVX2
    }();
    return supported;
}

// The lowest count set bits of mask. Not pdep, which is microcoded on AMD cores before Zen 3.
inline uint32_t lowest_bits(uint32_t mask, uint32_t count) {
    uint32_t kept = 0;
    for (uint32_t k = 0; k < count; ++k, mask &= mask - 1) kept |= mask & (0u - mask);
    return kept;
}

// For each 8-bit lane mask, the source byte of every lane when the set lanes take consecutive bytes.
struct LaneExpandTable {
    uint8_t index[256][8];
    constexpr LaneExpandTable() : index() {
        for (int m = 0; m < 256; ++m) {
            int next = 0;
            for (int lane = 0; lane < 8; ++lane) index[m][lane] = static_cast<uint8_t>((m >> lane) & 1 ? next++ : 0);
        }
    }
};
inline constexpr LaneExpandTable LANE_EXPAND{};

// decode_lanes_scalar with each per-lane field of up to 16 lanes held as two 8-lane AVX2 halves. The
// byte arbitration is a mask computation: the first min(8, bytes left) requesting lanes, in lane order,
// receive the next stream bytes (lowest_bits picks them; LANE_EXPAND with vpermd places them). Lanes
// whose step is a header, an idle start, a fragment end or a flush run the existing phase1 one by one,
// in lane order, after the vector step. Phase 0 of every lane precedes every phase 1, as in the
// reference, so a rejection leaves the same symbols. Within a segment the symbol's interval follows
// from the inverse (model_inverse_interval). Lane masks are kept as bits; vector masks are rebuilt from
// them, and unsigned compares use a sign flip.
inline bool decode_lanes_avx2_body(const uint8_t* input, size_t input_size,
                                   const uint8_t* alphabet_bits, size_t num_lanes,
                                   size_t target_symbols_per_lane,
                                   std::vector<uint16_t>& symbols) {
    if (num_lanes == 0 || num_lanes > 16 || target_symbols_per_lane == 0) return false;
    for (size_t i = 0; i < num_lanes; ++i) {
        if (alphabet_bits[i] > 8) return false;
    }
    symbols.assign(num_lanes * target_symbols_per_lane, 0);
    const size_t target = target_symbols_per_lane;
    const uint32_t lane_mask = (1u << num_lanes) - 1;

    Lane lanes[16];
    alignas(32) uint32_t p0[16] = {}, p1[16] = {}, in_rem[16] = {}, out_rem[16] = {};
    alignas(32) uint32_t sym[16] = {}, lane_f[16] = {}, lane_cf[16] = {}, ib[16] = {};
    alignas(32) uint32_t bound[9][16], base_rn[10][16], freq_rs[10][16], rf[10][16];
    uint32_t nsyms[16] = {};
    uint8_t row_eab[16], row_q[16];
    auto load_rows = [&](size_t i) {
        const uint8_t e = lanes[i].eab, q = lanes[i].q;
        row_eab[i] = e;
        row_q[i] = q;
        for (uint32_t k = 1; k < 10; ++k) bound[k - 1][i] = (k <= e + 1u) ? BASE_TABLE[e][q][k] : 0x10000u;
        for (uint32_t k = 0; k < 10; ++k) {
            base_rn[k][i] = BASE_TABLE[e][q][k] | (uint32_t(RN_TABLE[e][q][k]) << 16);
            freq_rs[k][i] = FREQ_TABLE[e][q][k] | (uint32_t(RS_TABLE[e][q][k]) << 16);
            rf[k][i] = RF_TABLE[e][q][k];
        }
    };
    for (size_t i = 0; i < 16; ++i) {
        if (i < num_lanes) {
            lanes[i].ab = alphabet_bits[i];
            lanes[i].target = static_cast<uint32_t>(target);
        }
        load_rows(i);
    }
    uint32_t ib_full = 0, sym_full = 0, proc = 0, drain = 0, header = 0, quiet = 0, done = 0;
    auto set_bit = [](uint32_t& m, uint32_t bit, bool on) { m = on ? (m | bit) : (m & ~bit); };

    const __m256i one = _mm256_set1_epi32(1), low16 = _mm256_set1_epi32(0xFFFF), sign = _mm256_set1_epi32(INT32_MIN);
    const __m256i home_s = _mm256_set1_epi32(static_cast<int>(0x1000000u ^ 0x80000000u));
    const __m256i drain_min_s = _mm256_set1_epi32(static_cast<int>(LANE_DRAIN_SHORT_MIN ^ 0x80000000u));
    const __m256i lane_bits = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    const __m256i lane_index = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    auto bits_of = [](__m256i lo, __m256i hi) {
        return static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(lo))) |
               (static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(hi))) << 8);
    };
    auto mask_of = [&](uint32_t m, int h) {
        return _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(static_cast<int>((m >> (8 * h)) & 0xFF)), lane_bits), lane_bits);
    };
    auto load = [](const uint32_t* a, int h) { return _mm256_load_si256(reinterpret_cast<const __m256i*>(a + 8 * h)); };
    auto store = [](uint32_t* a, int h, __m256i v) { _mm256_store_si256(reinterpret_cast<__m256i*>(a + 8 * h), v); };

    uint32_t slots = 0;
    size_t pos = 0;
    uint64_t cycle = 0;
    uint64_t max_cycles = 64ull * (num_lanes * target_symbols_per_lane + input_size) + 1000;
    uint64_t total_syms = 0;
    size_t last_pos = SIZE_MAX;
    uint64_t last_total_syms = UINT64_MAX;
    uint32_t last_state_sum = UINT32_MAX;
    uint64_t last_change = 0;
    uint32_t state_sum = 0;

    while (cycle < max_cycles) {
        __m256i v_in_rem[2], v_ib[2], v_p0[2], v_p1[2], v_x[2], v_out[2];
        for (int h = 0; h < 2; ++h) v_in_rem[h] = load(in_rem, h);
        const uint32_t in_left = ~bits_of(_mm256_cmpeq_epi32(v_in_rem[0], _mm256_setzero_si256()),
                                          _mm256_cmpeq_epi32(v_in_rem[1], _mm256_setzero_si256())) & 0xFFFF;
        const uint32_t blocked = sym_full & slots;
        const uint32_t need = ~ib_full & (header | (proc & in_left)) & lane_mask;
        const uint32_t request = need & ~blocked;
        const size_t left = input_size - pos;
        const uint32_t avail = left < 8 ? static_cast<uint32_t>(left) : 8u;
        uint32_t valid = request;
        if (static_cast<uint32_t>(_mm_popcnt_u32(request)) > avail) valid = lowest_bits(request, avail);
        const uint32_t run = lane_mask & ~((need & ~valid) | blocked);
        uint64_t raw = 0;
        if (avail != 0) std::memcpy(&raw, input + pos, avail);
        const uint32_t low_taken = static_cast<uint32_t>(_mm_popcnt_u32(valid & 0xFF));
        const uint64_t raw_half[2] = { raw, low_taken < 8 ? raw >> (8 * low_taken) : 0 };
        ib_full |= valid;
        pos += _mm_popcnt_u32(valid);

        const uint32_t pd = proc | drain;
        const uint32_t emit = run & sym_full;
        for (int h = 0; h < 2; ++h) {
            const uint32_t vh = (valid >> (8 * h)) & 0xFF;
            const __m256i bytes = _mm256_permutevar8x32_epi32(
                _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(raw_half[h]))),
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(LANE_EXPAND.index[vh]))));
            v_ib[h] = _mm256_blendv_epi8(load(ib, h), bytes, mask_of(valid, h));
            v_p1[h] = load(p1, h);
            const __m256i adv = _mm256_sub_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(load(lane_f, h), _mm256_srli_epi32(v_p1[h], 16)), _mm256_and_si256(v_p1[h], low16)),
                load(lane_cf, h));
            v_p0[h] = _mm256_blendv_epi8(load(p0, h), _mm256_blendv_epi8(v_p1[h], adv, mask_of(sym_full, h)), mask_of(run & pd, h));
        }
        for (uint32_t m = emit; m != 0; m &= m - 1) {
            unsigned long i;
            _BitScanForward(&i, m);
            symbols[i * target + nsyms[i]++] = static_cast<uint16_t>(sym[i]);
        }
        slots |= emit;
        total_syms += _mm_popcnt_u32(emit);
        sym_full &= ~run;

        __m256i below_home[2], in_is_will[2], below_drain_min[2], above_home[2], out_left[2];
        for (int h = 0; h < 2; ++h) {
            below_home[h] = _mm256_cmpgt_epi32(home_s, _mm256_xor_si256(v_p0[h], sign));
        }
        const uint32_t will = proc & run & in_left & ib_full & bits_of(below_home[0], below_home[1]);
        for (int h = 0; h < 2; ++h) {
            const __m256i wm = mask_of(will, h);
            v_x[h] = _mm256_blendv_epi8(v_p0[h], _mm256_or_si256(_mm256_slli_epi32(v_p0[h], 8), v_ib[h]), wm);
            in_is_will[h] = _mm256_cmpeq_epi32(v_in_rem[h], _mm256_and_si256(wm, one));
            const __m256i xs = _mm256_xor_si256(v_x[h], sign);
            below_drain_min[h] = _mm256_cmpgt_epi32(drain_min_s, xs);
            above_home[h] = _mm256_cmpgt_epi32(xs, home_s);
            v_out[h] = load(out_rem, h);
            out_left[h] = _mm256_cmpeq_epi32(v_out[h], _mm256_setzero_si256());
        }
        const uint32_t edge = (proc & bits_of(in_is_will[0], in_is_will[1])) | (drain & bits_of(below_drain_min[0], below_drain_min[1]));
        const uint32_t simple = run & pd & ~edge;
        const uint32_t generic = run & ((~pd & ~quiet) | (pd & edge));
        const uint32_t nsf = simple & bits_of(above_home[0], above_home[1]) & ~bits_of(out_left[0], out_left[1]);
        const uint32_t consumed = simple & will;
        for (int h = 0; h < 2; ++h) {
            if (((nsf >> (8 * h)) & 0xFF) != 0) {
                const __m256i c = _mm256_and_si256(v_x[h], low16);
                __m256i seg = _mm256_set1_epi32(9);
                for (int k = 0; k < 9; ++k) seg = _mm256_add_epi32(seg, _mm256_cmpgt_epi32(load(bound[k], h), c));
                const __m256i idx = _mm256_add_epi32(_mm256_slli_epi32(seg, 4), lane_index);
                const __m256i v_base_rn = _mm256_i32gather_epi32(reinterpret_cast<const int*>(&base_rn[0][8 * h]), idx, 4);
                const __m256i v_freq_rs = _mm256_i32gather_epi32(reinterpret_cast<const int*>(&freq_rs[0][8 * h]), idx, 4);
                const __m256i v_rf = _mm256_i32gather_epi32(reinterpret_cast<const int*>(&rf[0][8 * h]), idx, 4);
                const __m256i base = _mm256_and_si256(v_base_rn, low16);
                const __m256i freq = _mm256_and_si256(v_freq_rs, low16);
                const __m256i dist = _mm256_and_si256(_mm256_add_epi32(_mm256_sub_epi32(c, base), _mm256_srli_epi32(v_base_rn, 16)), low16);
                const __m256i steps = _mm256_and_si256(
                    _mm256_srlv_epi32(_mm256_srli_epi32(_mm256_mullo_epi32(dist, v_rf), 15), _mm256_srli_epi32(v_freq_rs, 16)),
                    _mm256_set1_epi32(0x1FF));
                const __m256i s = _mm256_and_si256(_mm256_add_epi32(_mm256_srli_epi32(_mm256_sllv_epi32(one, seg), 1), steps),
                                                   _mm256_set1_epi32(0x1FF));
                const __m256i cfa = _mm256_and_si256(_mm256_add_epi32(base, _mm256_mullo_epi32(freq, steps)), low16);
                const __m256i zero = _mm256_cmpeq_epi32(cfa, _mm256_setzero_si256());
                const __m256i k = mask_of(nsf, h);
                store(sym, h, _mm256_blendv_epi8(load(sym, h), s, k));
                store(lane_f, h, _mm256_blendv_epi8(load(lane_f, h),
                                                     _mm256_blendv_epi8(freq, _mm256_and_si256(_mm256_sub_epi32(freq, one), low16), zero), k));
                store(lane_cf, h, _mm256_blendv_epi8(load(lane_cf, h), _mm256_blendv_epi8(cfa, one, zero), k));
                v_out[h] = _mm256_add_epi32(v_out[h], k);  // k is -1 where a symbol was decoded
            }
            store(p0, h, v_p0[h]);
            store(p1, h, _mm256_blendv_epi8(v_p1[h], v_x[h], mask_of(simple, h)));
            store(in_rem, h, _mm256_add_epi32(v_in_rem[h], mask_of(consumed, h)));
            store(out_rem, h, v_out[h]);
            store(ib, h, v_ib[h]);
        }
        sym_full |= nsf;
        ib_full &= ~consumed;

        for (uint32_t m = generic; m != 0; m &= m - 1) {
            unsigned long i;
            _BitScanForward(&i, m);
            const uint32_t bit = 1u << i;
            Lane& ln = lanes[i];
            ln.p0 = p0[i];
            ln.p1 = p1[i];
            ln.in_rem = in_rem[i];
            ln.out_rem = out_rem[i];
            ln.sym = static_cast<uint16_t>(sym[i]);
            ln.ib = static_cast<uint8_t>(ib[i]);
            ln.ib_full = (ib_full & bit) != 0;
            ln.sym_full = (sym_full & bit) != 0;
            const uint32_t old_state = ln.state;
            if (!phase1(ln)) return false;
            p0[i] = ln.p0;
            p1[i] = ln.p1;
            in_rem[i] = ln.in_rem;
            out_rem[i] = ln.out_rem;
            sym[i] = ln.sym;
            set_bit(ib_full, bit, ln.ib_full);
            set_bit(sym_full, bit, ln.sym_full);
            if (ln.sym_full) model_forward(ln.eab, ln.q, ln.sym, lane_f[i], lane_cf[i]);
            state_sum += static_cast<uint32_t>(ln.state) - old_state;
            set_bit(proc, bit, ln.state == LANE_PROC);
            set_bit(drain, bit, ln.state == LANE_DRAIN);
            set_bit(header, bit, ln.state >= LANE_HDR0 && ln.state <= LANE_HDR2);
            set_bit(quiet, bit, ln.state == LANE_IDLE && !ln.go);
            set_bit(done, bit, ln.done);
            if (ln.eab != row_eab[i] || ln.q != row_q[i]) load_rows(i);
        }

        const uint32_t settled = done & ~sym_full;
        const uint32_t active = lane_mask & ~(settled & ~slots);
        if (slots != 0 && (slots & active) == active) {
            slots = 0;
        }
        cycle++;
        if ((settled & lane_mask) == lane_mask) {
            for (size_t i = 0; i < num_lanes; ++i) {
                uint32_t f = lanes[i].final_state;
                if (f != 0x1000000 && f != 0x10000 && f != 0x100) {
                    return false;
                }
            }
            for (size_t p = pos; p < input_size; ++p) {
                if (input[p] != 0) return false;
            }
            return true;
        }

        if (pos != last_pos || total_syms != last_total_syms || state_sum != last_state_sum) {
            last_pos = pos;
            last_total_syms = total_syms;
            last_state_sum = state_sum;
            last_change = cycle;
        } else if (cycle - last_change > 64) {
            return false;
        }
    }
    return false;
}

inline bool decode_lanes_avx2(const uint8_t* input, size_t input_size,
                              const uint8_t* alphabet_bits, size_t num_lanes,
                              size_t target_symbols_per_lane,
                              std::vector<uint16_t>& symbols) {
    bool ok = decode_lanes_avx2_body(input, input_size, alphabet_bits, num_lanes, target_symbols_per_lane, symbols);
    _mm256_zeroupper();
    return ok;
}
#endif

enum LanePath : int { LANE_PATH_REFERENCE, LANE_PATH_SCALAR, LANE_PATH_AVX2, LANE_PATH_COUNT };

inline bool lane_path_supported(int path) {
    if (path == LANE_PATH_REFERENCE || path == LANE_PATH_SCALAR) return true;
#ifdef X2W_BCPACK_LANE_SIMD
    if (path == LANE_PATH_AVX2) return lane_avx2_supported();
#endif
    return false;
}

inline int best_lane_path() {
    return lane_path_supported(LANE_PATH_AVX2) ? LANE_PATH_AVX2 : LANE_PATH_SCALAR;
}

#ifdef X2W_BCPACK_LANE_TESTS
// Test-only switch, compiled only into tests that define X2W_BCPACK_LANE_TESTS for the whole binary:
// when set to a supported LanePath, decode_lanes uses it.
inline int lane_path_override = -1;
#endif

inline bool decode_lanes_on(int path, const uint8_t* input, size_t input_size,
                            const uint8_t* alphabet_bits, size_t num_lanes,
                            size_t target_symbols_per_lane,
                            std::vector<uint16_t>& symbols) {
    switch (path) {
    case LANE_PATH_REFERENCE:
        return decode_lanes_reference(input, input_size, alphabet_bits, num_lanes, target_symbols_per_lane, symbols);
    case LANE_PATH_SCALAR:
        return decode_lanes_scalar(input, input_size, alphabet_bits, num_lanes, target_symbols_per_lane, symbols);
#ifdef X2W_BCPACK_LANE_SIMD
    case LANE_PATH_AVX2:
        return decode_lanes_avx2(input, input_size, alphabet_bits, num_lanes, target_symbols_per_lane, symbols);
#endif
    default:
        return false;
    }
}

inline bool decode_lanes(const uint8_t* input, size_t input_size,
                         const uint8_t* alphabet_bits, size_t num_lanes,
                         size_t target_symbols_per_lane,
                         std::vector<uint16_t>& symbols, LaneReport* report = nullptr) {
    if (report) return decode_lanes_reference(input, input_size, alphabet_bits, num_lanes, target_symbols_per_lane, symbols, report);
#ifdef X2W_BCPACK_LANE_TESTS
    int path = lane_path_supported(lane_path_override) ? lane_path_override : best_lane_path();
#else
    int path = best_lane_path();
#endif
    return decode_lanes_on(path, input, input_size, alphabet_bits, num_lanes, target_symbols_per_lane, symbols);
}

inline constexpr uint8_t SUBSETS[8] = { 3, 2, 3, 2, 1, 1, 1, 2 };
inline constexpr uint8_t CHANNEL_BITS[8][4] = {
    { 4, 4, 4, 0 },
    { 6, 6, 6, 0 },
    { 5, 5, 5, 0 },
    { 7, 7, 7, 0 },
    { 5, 5, 5, 6 },
    { 7, 7, 7, 8 },
    { 7, 7, 7, 7 },
    { 5, 5, 5, 5 },
};
inline constexpr uint8_t INDEX_BITS[8][2] = {
    { 3, 0 },
    { 3, 0 },
    { 2, 0 },
    { 2, 0 },
    { 3, 2 },
    { 2, 2 },
    { 4, 0 },
    { 2, 0 },
};
inline constexpr uint8_t PARTITION_BITS[8] = { 4, 6, 6, 6, 0, 0, 0, 6 };
inline constexpr uint8_t ANCHOR_2[64] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
    6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15
};
inline constexpr uint8_t ANCHOR_3A[64] = {
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3,
    3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
    8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15,
    3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3
};
inline constexpr uint8_t ANCHOR_3B[64] = {
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8,
    15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
    15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8,
    15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8
};
inline constexpr int INDEX_LANES[2][4] = {
    { 11, 12, 13, 14 },
    { 6, 7, 8, 10 }
};

inline int endpoint_lane(size_t subsets, size_t row, size_t ch) {
    if (subsets == 3 && ch == 3 && (row == 0 || row == 1)) return -1;
    if (row == 0) return static_cast<int>(2 + ch);
    if (row == 1) return static_cast<int>(6 + ch);
    if (row == 2 && ch == 0) return 10;
    if (row == 2 && subsets == 3 && ch == 1) return 5;
    if (row == 2 && subsets == 3 && ch == 2) return 9;
    return -1;
}

struct BitWriter128 {
    uint64_t low = 0;
    uint64_t high = 0;
    uint32_t at = 0;

    void put(uint32_t val, uint32_t n) {
        if (n == 0) return;
        uint64_t v = (n == 64) ? val : (val & ((1ull << n) - 1));
        if (at < 64) {
            low |= (v << at);
            if (at + n > 64) {
                high |= (v >> (64 - at));
            }
        } else if (at < 128) {
            high |= (v << (at - 64));
        }
        at += n;
    }

    void write_to(uint8_t* out) const {
        std::memcpy(out, &low, 8);
        std::memcpy(out + 8, &high, 8);
    }
};

struct Bc7State {
    uint8_t subsets = 0;
    uint8_t ep[3][8] = {};
    uint8_t ix[2][16] = {};
};

inline void pack_bc7_block(const Bc7Control& c, const Bc7State& st, uint8_t* out) {
    uint32_t mode = c.mode;
    uint32_t subsets = SUBSETS[mode];
    uint8_t prec[4];
    for (int ch = 0; ch < 4; ++ch) prec[ch] = CHANNEL_BITS[mode][ch];
    if (c.rotation != 0) {
        std::swap(prec[3], prec[c.rotation - 1]);
    } else if (mode == 0 || mode == 1 || mode == 3 || mode == 6 || mode == 7) {
        prec[0] += 1;
    }
    bool has_pbits = (mode == 0 || mode == 1 || mode == 3 || mode == 6 || mode == 7);

    BitWriter128 w;
    w.put(1u << mode, mode + 1);
    w.put(c.partition, PARTITION_BITS[mode]);
    if (mode == 4 || mode == 5) {
        w.put(c.rotation, 2);
    }
    if (mode == 4) {
        w.put(c.index_selection, 1);
    }
    uint8_t ch_order[4] = { 0, 1, 2, 3 };
    if (c.rotation != 0) {
        std::swap(ch_order[3], ch_order[c.rotation - 1]);
    }

    for (int i = 0; i < 4; ++i) {
        uint8_t ch = ch_order[i];
        uint8_t p = prec[ch];
        if (p == 0) continue;
        uint32_t drop = (ch == 0 && has_pbits) ? 1 : 0;
        for (size_t e = 0; e < 2 * subsets; ++e) {
            uint32_t field_val = st.ep[e / 2][2 * ch + e % 2] >> (8 - p);
            w.put(field_val >> drop, p - drop);
        }
    }
    if (mode == 1) {
        for (size_t row = 0; row < subsets; ++row) {
            w.put((st.ep[row][0] >> (8 - prec[0])) & 1, 1);
        }
    } else if (has_pbits) {
        for (size_t e = 0; e < 2 * subsets; ++e) {
            w.put((st.ep[e / 2][e % 2] >> (8 - prec[0])) & 1, 1);
        }
    }
    uint16_t anchors = 1;
    if (subsets == 2) {
        anchors = 1 | (1 << ANCHOR_2[c.partition & 63]);
    } else if (subsets == 3) {
        anchors = 1 | (1 << ANCHOR_3A[c.partition & 63]) | (1 << ANCHOR_3B[c.partition & 63]);
    }
    auto write_set = [&](size_t s) {
        uint32_t b = INDEX_BITS[mode][s];
        for (int p_idx = 0; p_idx < 16; ++p_idx) {
            uint32_t n = b - ((anchors >> p_idx) & 1);
            uint32_t val = st.ix[s][p_idx] >> (4 - b);
            w.put(val, n);
        }
    };
    if (mode == 4) {
        write_set(1);
        write_set(0);
    } else if (mode == 5) {
        write_set(0);
        write_set(1);
    } else {
        write_set(0);
    }
    w.write_to(out);
}

inline bool decode_bc7_blocks(const std::vector<uint16_t>& symbols, size_t num_blocks,
                              uint8_t* output, size_t output_capacity) {
    if (output_capacity < num_blocks * 16) return false;
    size_t target = num_blocks * 2;
    Bc7State hist[256] = {};

    for (size_t i = 0; i < num_blocks; ++i) {
        uint16_t l0_e = symbols[0 * target + 2 * i];
        uint16_t l0_o = symbols[0 * target + 2 * i + 1];
        uint16_t l1_e = symbols[1 * target + 2 * i];
        uint16_t l1_o = symbols[1 * target + 2 * i + 1];

        uint8_t d_byte = interleave(l0_e, l0_o);
        uint8_t x_byte = interleave(l1_e, l1_o);
        int32_t delta = unzigzag(d_byte);
        size_t er = static_cast<size_t>((x_byte + delta) & 0xFF);
        size_t xr = static_cast<size_t>(x_byte);

        uint16_t e15 = symbols[15 * target + 2 * i];
        uint16_t o15 = symbols[15 * target + 2 * i + 1];
        size_t code = ((o15 & 15) << 5) | (e15 & 31);
        if (code >= 512 || !BC7_CONTROL_TABLE[code].valid) {
            // The reference decoder emits a zero block for unmapped codes; for valid BC7 streams
            // all 285 mapped codes are legal. An unmapped code indicates stream corruption,
            // so fail fast per repository fail-fast policy (Refs #1139).
            return false;
        }
        const Bc7Control& c = BC7_CONTROL_TABLE[code];

        auto get_source = [&](size_t r, Bc7State& st) {
            if (r == 0) {
                st = {};
            } else if (r <= i) {
                st = hist[(i - r) & 255];
            } else {
                size_t dict_idx = r - i - 1;
                if (dict_idx < 255) {
                    const auto& entry = BC7_DICTIONARY[dict_idx];
                    st.subsets = entry.subsets;
                    std::memcpy(st.ep, entry.ep, sizeof(st.ep));
                    std::memcpy(st.ix, entry.ix, sizeof(st.ix));
                } else {
                    st = {};
                }
            }
        };

        Bc7State er_state, xr_state;
        get_source(er, er_state);
        get_source(xr, xr_state);

        Bc7State st = {};
        st.subsets = SUBSETS[c.mode];
        uint8_t prec[4];
        for (int ch = 0; ch < 4; ++ch) prec[ch] = CHANNEL_BITS[c.mode][ch];
        if (c.rotation != 0) {
            std::swap(prec[3], prec[c.rotation - 1]);
        } else if (c.mode == 0 || c.mode == 1 || c.mode == 3 || c.mode == 6 || c.mode == 7) {
            prec[0] += 1;
        }

        size_t common = (st.subsets < er_state.subsets) ? st.subsets : er_state.subsets;
        for (int ch = 0; ch < 4; ++ch) {
            uint8_t p = prec[ch];
            if (p == 0) continue;
            for (size_t row = 0; row < st.subsets; ++row) {
                size_t from = (row < common) ? row : 0;
                int lane = endpoint_lane(st.subsets, row, ch);
                for (int e = 0; e < 2; ++e) {
                    size_t col = 2 * ch + e;
                    uint8_t res = 0;
                    if (lane >= 0) {
                        res = residual(symbols[lane * target + 2 * i + e], p);
                    }
                    uint8_t base = er_state.ep[from][col] >> (8 - p);
                    st.ep[row][col] = ((base + res) & ((1 << p) - 1)) << (8 - p);
                }
            }
        }

        for (int set = 0; set < 2; ++set) {
            uint8_t b = INDEX_BITS[c.mode][set];
            if (b == 0) continue;
            for (int row = 0; row < 4; ++row) {
                int lane = INDEX_LANES[set][row];
                for (int half = 0; half < 2; ++half) {
                    uint16_t sym = symbols[lane * target + 2 * i + half];
                    uint16_t l, r;
                    index_pair(sym, row, l, r);
                    uint16_t pair[2] = { l, r };
                    for (int h = 0; h < 2; ++h) {
                        size_t p_idx = 4 * row + 2 * half + h;
                        uint8_t res = residual(pair[h], b);
                        uint8_t base = xr_state.ix[set][p_idx] >> (4 - b);
                        st.ix[set][p_idx] = ((base + res) & ((1 << b) - 1)) << (4 - b);
                    }
                }
            }
        }

        pack_bc7_block(c, st, output + i * 16);
        hist[i & 255] = st;
    }
    return true;
}

inline constexpr uint8_t ALPHA_CODE[8] = { 0, 2, 3, 4, 5, 6, 7, 1 };
inline constexpr uint8_t COLOUR_CODE[4] = { 0, 2, 3, 1 };

inline void pack_alpha(uint8_t e0, uint8_t e1, const uint8_t* ix, uint8_t* out) {
    uint64_t v = static_cast<uint64_t>(e0 ^ 0x80) | (static_cast<uint64_t>(e1 ^ 0x80) << 8);
    for (size_t p = 0; p < 16; ++p) {
        v |= static_cast<uint64_t>(ALPHA_CODE[ix[p] & 7]) << (16 + 3 * p);
    }
    std::memcpy(out, &v, 8);
}

inline void pack_colour(const uint8_t* ep, const uint8_t* ix, uint8_t* out) {
    uint16_t c0 = static_cast<uint16_t>((static_cast<uint16_t>(ep[0]) << 11) |
                                        (static_cast<uint16_t>(ep[1]) << 5) |
                                        static_cast<uint16_t>(ep[2]));
    uint16_t c1 = static_cast<uint16_t>((static_cast<uint16_t>(ep[3]) << 11) |
                                        (static_cast<uint16_t>(ep[4]) << 5) |
                                        static_cast<uint16_t>(ep[5]));
    uint32_t w = 0;
    for (size_t p = 0; p < 16; ++p) {
        w |= static_cast<uint32_t>(COLOUR_CODE[ix[p] & 3]) << (2 * p);
    }
    std::memcpy(out, &c0, 2);
    std::memcpy(out + 2, &c1, 2);
    std::memcpy(out + 4, &w, 4);
}

struct Bc4State {
    uint8_t ep[2] = {};
    uint8_t ix[16] = {};
};

inline bool decode_bc4_blocks(const std::vector<uint16_t>& symbols, size_t num_blocks,
                              uint8_t* output, size_t output_capacity) {
    if (output_capacity < num_blocks * 8) return false;
    size_t target = num_blocks;
    Bc4State hist[256] = {};

    for (size_t i = 0; i < num_blocks; ++i) {
        uint16_t d_sym = symbols[0 * target + i];
        uint16_t x_sym = symbols[1 * target + i];
        int32_t delta = unzigzag(d_sym);
        size_t er = static_cast<size_t>((x_sym + delta) & 0xFF);
        size_t xr = static_cast<size_t>(x_sym & 0xFF);

        auto get_source = [&](size_t r, Bc4State& st) {
            if (r == 0) {
                st = {};
            } else if (r <= i) {
                st = hist[(i - r) & 255];
            } else {
                size_t dict_idx = r - i - 1;
                if (dict_idx < 255) {
                    const auto& entry = BC4_DICTIONARY[dict_idx];
                    std::memcpy(st.ep, entry.ep, 2);
                    std::memcpy(st.ix, entry.ix, 16);
                } else {
                    st = {};
                }
            }
        };

        Bc4State er_state, xr_state;
        get_source(er, er_state);
        get_source(xr, xr_state);

        Bc4State st = {};
        st.ep[0] = static_cast<uint8_t>((er_state.ep[0] + residual(symbols[2 * target + i], 8)) & 0xFF);
        st.ep[1] = static_cast<uint8_t>((er_state.ep[1] + residual(symbols[3 * target + i], 8)) & 0xFF);

        for (size_t k = 0; k < 8; ++k) {
            uint16_t sym = symbols[(4 + k) * target + i] & 63;
            uint8_t pa = RANK_PAIR_BC4[sym][0];
            uint8_t pb = RANK_PAIR_BC4[sym][1];
            st.ix[2 * k + 0] = static_cast<uint8_t>((xr_state.ix[2 * k + 0] + pa) & 7);
            st.ix[2 * k + 1] = static_cast<uint8_t>((xr_state.ix[2 * k + 1] + pb) & 7);
        }

        pack_alpha(st.ep[0], st.ep[1], st.ix, output + i * 8);
        hist[i & 255] = st;
    }
    return true;
}

struct Bc5State {
    uint8_t ep[4] = {};
    uint8_t ix[32] = {};
};

inline bool decode_bc5_blocks(const std::vector<uint16_t>& symbols, size_t num_blocks,
                              uint8_t* output, size_t output_capacity) {
    if (output_capacity < num_blocks * 16) return false;
    size_t target = num_blocks * 2;
    Bc5State hist[256] = {};

    for (size_t i = 0; i < num_blocks; ++i) {
        uint16_t d0 = symbols[0 * target + 2 * i];
        uint16_t d1 = symbols[0 * target + 2 * i + 1];
        uint16_t x0 = symbols[1 * target + 2 * i];
        uint16_t x1 = symbols[1 * target + 2 * i + 1];

        uint8_t d_byte = interleave(d0, d1);
        uint8_t x_byte = interleave(x0, x1);
        int32_t delta = unzigzag(d_byte);
        size_t er = static_cast<size_t>((x_byte + delta) & 0xFF);
        size_t xr = static_cast<size_t>(x_byte);

        auto get_source = [&](size_t r, Bc5State& st) {
            if (r == 0) {
                st = {};
            } else if (r <= i) {
                st = hist[(i - r) & 255];
            } else {
                size_t dict_idx = r - i - 1;
                if (dict_idx < 255) {
                    const auto& entry = BC5_DICTIONARY[dict_idx];
                    std::memcpy(st.ep, entry.ep, 4);
                    std::memcpy(st.ix, entry.ix, 32);
                } else {
                    st = {};
                }
            }
        };

        Bc5State er_state, xr_state;
        get_source(er, er_state);
        get_source(xr, xr_state);

        Bc5State st = {};
        for (size_t c = 0; c < 2; ++c) {
            uint16_t a = symbols[(2 + c) * target + 2 * i];
            uint16_t b = symbols[(2 + c) * target + 2 * i + 1];
            st.ep[2 * c + 0] = static_cast<uint8_t>((er_state.ep[2 * c + 0] + residual(a, 8)) & 0xFF);
            st.ep[2 * c + 1] = static_cast<uint8_t>((er_state.ep[2 * c + 1] + residual(b, 8)) & 0xFF);

            size_t base = 16 * c;
            for (size_t l = 0; l < 4; ++l) {
                uint16_t sa = symbols[(4 + 4 * c + l) * target + 2 * i] & 63;
                uint16_t sb = symbols[(4 + 4 * c + l) * target + 2 * i + 1] & 63;
                uint8_t pa0 = RANK_PAIR_BC5[sa][0];
                uint8_t pa1 = RANK_PAIR_BC5[sa][1];
                uint8_t pb0 = RANK_PAIR_BC5[sb][0];
                uint8_t pb1 = RANK_PAIR_BC5[sb][1];
                st.ix[base + 4 * l + 0] = static_cast<uint8_t>((xr_state.ix[base + 4 * l + 0] + pa0) & 7);
                st.ix[base + 4 * l + 1] = static_cast<uint8_t>((xr_state.ix[base + 4 * l + 1] + pa1) & 7);
                st.ix[base + 4 * l + 2] = static_cast<uint8_t>((xr_state.ix[base + 4 * l + 2] + pb0) & 7);
                st.ix[base + 4 * l + 3] = static_cast<uint8_t>((xr_state.ix[base + 4 * l + 3] + pb1) & 7);
            }
        }

        pack_alpha(st.ep[0], st.ep[1], st.ix, output + i * 16);
        pack_alpha(st.ep[2], st.ep[3], st.ix + 16, output + i * 16 + 8);
        hist[i & 255] = st;
    }
    return true;
}

struct Bc3State {
    uint8_t ep[8] = {};
    uint8_t ix[32] = {};
};

inline bool decode_bc3_blocks(const std::vector<uint16_t>& symbols, size_t num_blocks,
                              uint8_t* output, size_t output_capacity) {
    if (output_capacity < num_blocks * 16) return false;
    size_t target = num_blocks * 2;
    Bc3State hist[256] = {};

    for (size_t i = 0; i < num_blocks; ++i) {
        uint16_t d0 = symbols[0 * target + 2 * i];
        uint16_t d1 = symbols[0 * target + 2 * i + 1];
        uint16_t x0 = symbols[1 * target + 2 * i];
        uint16_t x1 = symbols[1 * target + 2 * i + 1];

        uint8_t d_byte = interleave(d0, d1);
        uint8_t x_byte = interleave(x0, x1);
        int32_t delta = unzigzag(d_byte);
        size_t er = static_cast<size_t>((x_byte + delta) & 0xFF);
        size_t xr = static_cast<size_t>(x_byte);

        auto get_source = [&](size_t r, Bc3State& st) {
            if (r == 0) {
                st = {};
            } else if (r <= i) {
                st = hist[(i - r) & 255];
            } else {
                size_t dict_idx = r - i - 1;
                if (dict_idx < 255) {
                    const auto& entry = BC3_DICTIONARY[dict_idx];
                    std::memcpy(st.ep, entry.ep, 8);
                    std::memcpy(st.ix, entry.ix, 32);
                } else {
                    st = {};
                }
            }
        };

        Bc3State er_state, xr_state;
        get_source(er, er_state);
        get_source(xr, xr_state);

        Bc3State st = {};
        // Alpha endpoints
        uint16_t a_e0 = symbols[2 * target + 2 * i];
        uint16_t a_e1 = symbols[2 * target + 2 * i + 1];
        st.ep[0] = static_cast<uint8_t>((er_state.ep[0] + residual(a_e0, 8)) & 0xFF);
        st.ep[1] = static_cast<uint8_t>((er_state.ep[1] + residual(a_e1, 8)) & 0xFF);

        // Alpha indices
        for (size_t l = 0; l < 4; ++l) {
            uint16_t sa = symbols[(3 + l) * target + 2 * i] & 63;
            uint16_t sb = symbols[(3 + l) * target + 2 * i + 1] & 63;
            uint8_t pa0 = RANK_PAIR_BC3_ALPHA[sa][0];
            uint8_t pa1 = RANK_PAIR_BC3_ALPHA[sa][1];
            uint8_t pb0 = RANK_PAIR_BC3_ALPHA[sb][0];
            uint8_t pb1 = RANK_PAIR_BC3_ALPHA[sb][1];
            st.ix[4 * l + 0] = static_cast<uint8_t>((xr_state.ix[4 * l + 0] + pa0) & 7);
            st.ix[4 * l + 1] = static_cast<uint8_t>((xr_state.ix[4 * l + 1] + pa1) & 7);
            st.ix[4 * l + 2] = static_cast<uint8_t>((xr_state.ix[4 * l + 2] + pb0) & 7);
            st.ix[4 * l + 3] = static_cast<uint8_t>((xr_state.ix[4 * l + 3] + pb1) & 7);
        }

        // Colour endpoints (RGB565: R=5, G=6, B=5)
        for (size_t c = 0; c < 3; ++c) {
            uint16_t sa = symbols[(7 + c) * target + 2 * i];
            uint16_t sb = symbols[(7 + c) * target + 2 * i + 1];
            uint32_t bits = (c == 1) ? 6 : 5;
            uint8_t mask = static_cast<uint8_t>((1u << bits) - 1);
            st.ep[2 + c] = static_cast<uint8_t>((er_state.ep[2 + c] + residual(sa, bits)) & mask);
            st.ep[5 + c] = static_cast<uint8_t>((er_state.ep[5 + c] + residual(sb, bits)) & mask);
        }

        // Colour indices (2-bit quadrants)
        uint16_t tl = symbols[10 * target + 2 * i] & 255;
        uint16_t tr = symbols[10 * target + 2 * i + 1] & 255;
        uint16_t bl = symbols[11 * target + 2 * i] & 255;
        uint16_t br = symbols[11 * target + 2 * i + 1] & 255;
        const uint16_t quads[4] = { tl, tr, bl, br };
        for (size_t q = 0; q < 4; ++q) {
            uint16_t s = quads[q];
            size_t ox = (q & 1) * 2;
            size_t oy = (q >> 1) * 2;
            const size_t dx[4] = { 0, 1, 0, 1 };
            const size_t dy[4] = { 0, 0, 1, 1 };
            for (size_t k = 0; k < 4; ++k) {
                size_t p = 16 + 4 * (oy + dy[k]) + (ox + dx[k]);
                uint8_t qval = RANK_QUAD_BC3_COLOUR[s][k];
                st.ix[p] = static_cast<uint8_t>((xr_state.ix[p] + qval) & 3);
            }
        }

        pack_alpha(st.ep[0], st.ep[1], st.ix, output + i * 16);
        pack_colour(st.ep + 2, st.ix + 16, output + i * 16 + 8);
        hist[i & 255] = st;
    }
    return true;
}

#include "bcpack_blocks_extra.inl"

inline bool decode_bcpack(const unsigned char* input, size_t input_size, unsigned bc_format,
                          unsigned char* output, size_t output_size, size_t* produced) {
    if (!produced) return false;
    *produced = 0;
    if (!input || !output || input_size == 0 || output_size == 0 ||
        bc_format < 1 || bc_format > 7) return false;

    // Hardware alphabet indices, not literal symbol bit widths.
    static constexpr uint8_t alphabets[7][16] = {
        {7,7,4,5,4,4,5,4,7,7,7,7},
        {3,3,3,3,3,3,3,3,3,3,4,5,4,7,7},
        {3,3,7,5,5,5,5,4,5,4,7,7},
        {7,7,7,7,5,5,5,5,5,5,5,5},
        {3,3,7,7,5,5,5,5,5,5,5,5},
        {3,3,7,7,7,4,4,4,5,5,5,7,7,7,7,4},
        {3,3,7,7,7,7,7,6,6,4,5,7,7,7,7,4},
    };
    const size_t bytes_per_block = bc_format == 1 || bc_format == 4 ? 8 : 16;
    if (output_size % bytes_per_block) return false;
    const size_t blocks = output_size / bytes_per_block;
    const size_t target = output_size / 8;
    const size_t lanes = bc_format >= 6 ? 16 : bc_format == 2 ? 15 : 12;
    std::vector<uint16_t> symbols;
    if (target > symbols.max_size() / lanes) return false;
    if (!decode_lanes(input, input_size, alphabets[bc_format-1], lanes, target, symbols)) return false;
    bool ok = false;
    switch (bc_format) {
    case 1: ok = decode_bc1_blocks(symbols,blocks,output,output_size); break;
    case 2: ok = decode_bc2_blocks(symbols,blocks,output,output_size); break;
    case 3: ok = decode_bc3_blocks(symbols,blocks,output,output_size); break;
    case 4: ok = decode_bc4_blocks(symbols,blocks,output,output_size); break;
    case 5: ok = decode_bc5_blocks(symbols,blocks,output,output_size); break;
    case 6: ok = decode_bc6_blocks(symbols,blocks,output,output_size); break;
    case 7: ok = decode_bc7_blocks(symbols,blocks,output,output_size); break;
    }
    if (!ok) return false;
    *produced = output_size;
    return true;
}

} // namespace x2w_bcpack
