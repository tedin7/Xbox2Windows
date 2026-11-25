// Test-only BCPack lane encoder: the exact inverse of x2w_bcpack::decode_lanes.
// Reverse rANS fragment equation adapted from bcpack-rs by Ahmet Dedeler
// (https://github.com/Ahmet-Dedeler/bcpack-rs, src/encode.rs), as in tests/bcpack_symbols_check.py.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Each lane is encoded on its own as flushed fragments. The shared-stream order is then learnt by
// running the reference cycle loop with every lane fed from its own byte queue.
#pragma once

#include "startup_stubs/bcpack_decoder.h"

#include <cstdint>
#include <vector>

namespace bcpack_test {

struct Rng {
    uint64_t s;
    uint32_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return static_cast<uint32_t>((z ^ (z >> 31)) >> 32);
    }
    uint32_t below(uint32_t n) { return next() % n; }
};

// The decoder's interval for a symbol, or f == 0 when the symbol cannot be coded.
inline bool interval(uint8_t eab, uint8_t q, uint16_t sym, uint32_t& f, uint32_t& cf) {
    if (sym >= (1u << (eab + 1))) return false;
    x2w_bcpack::model_forward(eab, q, sym, f, cf);
    return f != 0 && x2w_bcpack::model_inverse_fast(eab, q, static_cast<uint16_t>(cf)) == sym &&
           x2w_bcpack::model_inverse_fast(eab, q, static_cast<uint16_t>(cf + f - 1)) == sym;
}

// One flushed fragment (header + payload); false when a symbol has no interval or the payload is too long.
// With split, the same rANS stream is written as unflushed fragments of random lengths instead: the lane
// state then carries across fragment headers.
inline bool encode_fragment(uint8_t ab, uint8_t narrow, uint8_t q, const uint16_t* values, size_t count,
                            bool trim, std::vector<uint8_t>& out, Rng* split = nullptr) {
    const uint8_t eab = static_cast<uint8_t>(ab - narrow);
    uint64_t state = 1u << 24;
    std::vector<uint8_t> emitted;
    size_t last_emitted = 0;
    for (size_t k = 0; k < count; ++k) {
        uint32_t f, start;
        if (!interval(eab, q, values[count - 1 - k], f, start)) return false;
        while (state >= (uint64_t(f) << 16)) {
            emitted.push_back(static_cast<uint8_t>(state));
            state >>= 8;
        }
        if (k == 0) last_emitted = emitted.size();
        state = ((state / f) << 16) + state % f + start;
    }
    size_t omit = 0;
    if (trim && !split) {
        while (omit < 2 && omit < last_emitted && emitted[omit] == 0) ++omit;
    }
    const size_t payload = 4 + emitted.size() - omit;
    if (split) {
        std::vector<uint8_t> bytes;
        for (int b = 3; b >= 0; --b) bytes.push_back(static_cast<uint8_t>(state >> (8 * b)));
        for (size_t k = emitted.size(); k > 0; --k) bytes.push_back(emitted[k - 1]);
        // The last fragment must start at or before the last byte that precedes the final zero bytes:
        // a lane that decodes its last symbol after its budget ran out finishes in DRAIN and never
        // reads a later fragment (the decoder then sees those bytes as trailing data).
        const size_t last_start = bytes.size() - last_emitted - 1;
        for (size_t at = 0; at < bytes.size();) {
            size_t len = 1 + split->below(300);
            if (at + len > last_start) len = bytes.size() - at;
            const uint8_t header = static_cast<uint8_t>((narrow << 6) | q);
            if (len <= 256) {
                out.push_back(header);
                out.push_back(static_cast<uint8_t>(len - 1));
            } else {
                out.push_back(header | 0x20);
                out.push_back(static_cast<uint8_t>(len - 257));
                out.push_back(static_cast<uint8_t>((len - 257) >> 8));
            }
            out.insert(out.end(), bytes.begin() + at, bytes.begin() + at + len);
            at += len;
        }
        return true;
    }
    if (payload > 65792) return false;
    const uint8_t header = static_cast<uint8_t>((narrow << 6) | 0x10 | q);
    if (payload <= 256) {
        out.push_back(header);
        out.push_back(static_cast<uint8_t>(payload - 1));
    } else {
        out.push_back(header | 0x20);
        out.push_back(static_cast<uint8_t>(payload - 257));
        out.push_back(static_cast<uint8_t>((payload - 257) >> 8));
    }
    for (int b = 3; b >= 0; --b) out.push_back(static_cast<uint8_t>(state >> (8 * b)));
    for (size_t k = emitted.size(); k > omit; --k) out.push_back(emitted[k - 1]);
    return true;
}

// The reference cycle loop with each lane reading its own queue; records the shared-stream order.
inline bool interleave(const std::vector<std::vector<uint8_t>>& lane_bytes, const uint8_t* ab, size_t n,
                       size_t target, std::vector<uint8_t>& stream, size_t window = 8) {
    using namespace x2w_bcpack;
    std::vector<Lane> lanes(n);
    std::vector<size_t> next(n, 0);
    for (size_t i = 0; i < n; ++i) {
        lanes[i].ab = ab[i];
        lanes[i].target = static_cast<uint32_t>(target);
    }
    uint32_t slots = 0;
    for (uint64_t cycle = 0; cycle < 64ull * n * target + 1000; ++cycle) {
        size_t served = 0;
        for (size_t i = 0; i < n; ++i) {
            Lane& ln = lanes[i];
            const uint32_t bit = 1u << i;
            const bool blocked_out = ln.sym_full && (slots & bit);
            const bool need = !ln.ib_full && ((ln.state >= LANE_HDR0 && ln.state <= LANE_HDR2) ||
                                              (ln.state == LANE_PROC && ln.in_rem > 0));
            const bool valid = need && !blocked_out && served < window;
            if (valid && next[i] >= lane_bytes[i].size()) return false;  // the encoder is wrong
            ln.stall = (need && !valid) || blocked_out;
            if (ln.stall) continue;
            if (ln.state == LANE_PROC || ln.state == LANE_DRAIN) {
                ln.p0 = ln.sym_full ? model_advance(ln.eab, ln.q, ln.p1, ln.sym) : ln.p1;
            }
            if (valid) {
                ln.ib = lane_bytes[i][next[i]++];
                ln.ib_full = true;
                stream.push_back(ln.ib);
                served++;
            }
            if (ln.sym_full) {
                slots |= bit;
                ln.sym_full = false;
            }
        }
        uint32_t active = 0;
        bool finished = true;
        for (size_t i = 0; i < n; ++i) {
            Lane& ln = lanes[i];
            if (!ln.stall && !phase1(ln)) return false;
            if (!(ln.done && !ln.sym_full && (slots & (1u << i)) == 0)) active |= 1u << i;
            finished &= ln.done && !ln.sym_full;
        }
        if (slots != 0 && (slots & active) == active) slots = 0;
        if (finished) {
            for (size_t i = 0; i < n; ++i) {
                if (next[i] != lane_bytes[i].size()) return false;
            }
            return true;
        }
    }
    return false;
}

// Codes of BC7_CONTROL_TABLE whose two lane-15 symbols (code & 31, code >> 5) both have an interval.
inline std::vector<uint16_t> bc7_codes(uint8_t eab, uint8_t q) {
    std::vector<uint16_t> codes;
    uint32_t f, cf;
    for (uint16_t c = 0; c < 512; ++c) {
        if (x2w_bcpack::BC7_CONTROL_TABLE[c].valid && interval(eab, q, c & 31, f, cf) &&
            interval(eab, q, c >> 5, f, cf)) {
            codes.push_back(c);
        }
    }
    return codes;
}

struct Options {
    uint32_t max_fragment = 150;  // symbols per fragment, at most
    bool uniform = false;         // uniform symbols instead of the fragment model's distribution
    bool narrow = true;           // random alphabet narrowing in the headers
    bool trim = true;             // randomly omit the trailing zero bytes of a fragment
    bool unflushed = false;       // code about half of the lanes as one stream cut into unflushed fragments
    int bc7_lane = -1;            // lane whose symbol pairs must form a valid BC7 control code
    size_t padding = 0;           // trailing zero bytes
    size_t window = 8;            // bytes served per cycle; only the negative control changes it
};

// A random stream and its symbols (lane-major, target per lane). False only on an encoder bug.
inline bool make_stream(const uint8_t* ab, size_t n, size_t target, Rng& rng, const Options& o,
                        std::vector<uint8_t>& stream, std::vector<uint16_t>& symbols) {
    symbols.assign(n * target, 0);
    std::vector<std::vector<uint8_t>> lane_bytes(n);
    for (size_t l = 0; l < n; ++l) {
        size_t at = 0;
        while (at < target) {
            const bool control = static_cast<int>(l) == o.bc7_lane;
            const bool whole = o.unflushed && !control && at == 0 && (rng.next() & 1);
            size_t count = whole ? target : 1 + rng.below(o.max_fragment);
            if (control) count = 2 * ((count + 1) / 2);  // keep code pairs inside one fragment
            if (count > target - at) count = target - at;
            uint8_t narrow = (o.narrow && !control) ? static_cast<uint8_t>(rng.below(ab[l] < 3 ? ab[l] + 1 : 4)) : 0;
            const uint8_t eab = static_cast<uint8_t>(ab[l] - narrow);
            uint8_t q = 0;
            std::vector<uint16_t> codes;
            for (int tries = 0;; ++tries) {
                q = static_cast<uint8_t>(rng.below(16));
                if (!control) break;
                codes = bc7_codes(eab, q);
                if (!codes.empty()) break;
                if (tries > 64) return false;
            }
            uint16_t* v = &symbols[l * target + at];
            for (size_t k = 0; k < count; ++k) {
                if (control) {
                    const uint16_t c = codes[rng.below(static_cast<uint32_t>(codes.size()))];
                    v[k] = c & 31;
                    if (k + 1 < count) v[++k] = c >> 5;
                    continue;
                }
                for (int tries = 0;; ++tries) {
                    uint16_t s;
                    if (o.uniform) {
                        s = static_cast<uint16_t>(rng.below(1u << (eab + 1)));
                    } else {
                        s = x2w_bcpack::model_inverse_fast(eab, q, static_cast<uint16_t>(1 + rng.below(65535)));
                    }
                    uint32_t f, cf;
                    if (interval(eab, q, s, f, cf)) {
                        v[k] = s;
                        break;
                    }
                    if (tries > 100000) return false;
                }
            }
            const bool trim = o.trim && (rng.next() & 1);
            if (!encode_fragment(ab[l], narrow, q, v, count, trim, lane_bytes[l], whole ? &rng : nullptr)) {
                if (count == 1) return false;
                // payload over the long-header limit: retry with fewer symbols
                for (size_t k = 0; k < count; ++k) v[k] = 0;
                continue;
            }
            at += count;
        }
    }
    stream.clear();
    if (!interleave(lane_bytes, ab, n, target, stream, o.window)) return false;
    stream.insert(stream.end(), o.padding, 0);
    return true;
}

}  // namespace bcpack_test
