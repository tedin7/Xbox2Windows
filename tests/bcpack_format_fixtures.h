// Synthetic lane inputs, independent of the decoder's predictor and rank tables.
// SPDX-License-Identifier: MIT
#pragma once
#include "bcpack_encoder.h"
#include <algorithm>

namespace bcpack_fixture {
inline constexpr uint8_t AB[7][16] = {
    {7,7,4,5,4,4,5,4,7,7,7,7},
    {3,3,3,3,3,3,3,3,3,3,4,5,4,7,7},
    {3,3,7,5,5,5,5,4,5,4,7,7},
    {7,7,7,7,5,5,5,5,5,5,5,5},
    {3,3,7,7,5,5,5,5,5,5,5,5},
    {3,3,7,7,7,4,4,4,5,5,5,7,7,7,7,4},
    {3,3,7,7,7,7,7,6,6,4,5,7,7,7,7,4},
};
inline size_t lanes(unsigned fmt) { return fmt >= 6 ? 16 : fmt == 2 ? 15 : 12; }
inline size_t per(unsigned fmt) { return fmt == 1 || fmt == 4 ? 1 : 2; }
inline size_t block_bytes(unsigned fmt) { return 8 * per(fmt); }

inline void byte_symbols(std::vector<uint16_t>& s, size_t at, unsigned value, size_t p) {
    if (p == 1) { s[at] = static_cast<uint16_t>(value); return; }
    for (unsigned h = 0; h < 2; ++h) {
        s[at+h] = 0;
        for (unsigned b = 0; b < 4; ++b) s[at+h] |= ((value >> (2*b+h)) & 1) << b;
    }
}

inline std::vector<uint16_t> reference_symbols(unsigned fmt, unsigned er, unsigned xr, unsigned mode) {
    const size_t p = per(fmt);
    std::vector<uint16_t> s(lanes(fmt)*p, 0);
    const int delta = static_cast<int>((er-xr+128)&255)-128;
    byte_symbols(s,0,delta >= 0 ? 2*delta : -2*delta-1,p);
    byte_symbols(s,p,xr,p);
    if (fmt == 6) { s[30] = static_cast<uint16_t>(mode); s[31] = er & 31; }
    return s;
}

inline std::vector<uint16_t> random_symbols(unsigned fmt, size_t blocks, uint64_t seed) {
    const size_t target = per(fmt)*blocks;
    std::vector<uint16_t> s(lanes(fmt)*target);
    bcpack_test::Rng rng{seed};
    for (size_t l = 0; l < lanes(fmt); ++l)
        for (size_t i = 0; i < target; ++i) s[l*target+i] = static_cast<uint16_t>(rng.below(1u << (AB[fmt-1][l]+1)));
    if (fmt == 6) {
        for (size_t i = 0; i < blocks; ++i) {
            s[15*target+2*i] = static_cast<uint16_t>(i%14);
            s[15*target+2*i+1] = static_cast<uint16_t>(i%32);
        }
    } else if (fmt == 7) {
        // Choose all mapped controls; expected BC7 bytes still come from the oracle.
        std::vector<unsigned> codes;
        for (unsigned c = 0; c < 512; ++c)
            if (x2w_bcpack::BC7_CONTROL_TABLE[c].valid) codes.push_back(c);
        for (size_t i = 0; i < blocks; ++i) {
            const unsigned code = codes[i%codes.size()];
            s[15*target+2*i] = code & 31;
            s[15*target+2*i+1] = code >> 5;
        }
    }
    return s;
}

inline bool encode(unsigned fmt, const std::vector<uint16_t>& s, size_t blocks, std::vector<uint8_t>& stream) {
    const size_t target = blocks*per(fmt), n = lanes(fmt);
    if (!target || s.size() != n*target) return false;
    std::vector<std::vector<uint8_t>> encoded(n);
    for (size_t l = 0; l < n; ++l) {
        for (size_t at = 0; at < target;) {
            const size_t count = std::min(size_t(128),target-at);
            bool ok = false;
            for (uint8_t q = 0; q < 16 && !ok; ++q) {
                std::vector<uint8_t> fragment;
                if (bcpack_test::encode_fragment(AB[fmt-1][l],0,q,s.data()+l*target+at,count,true,fragment)) {
                    encoded[l].insert(encoded[l].end(),fragment.begin(),fragment.end());
                    ok = true;
                }
            }
            if (!ok) return false;
            at += count;
        }
    }
    return bcpack_test::interleave(encoded,AB[fmt-1],n,target,stream);
}
} // namespace bcpack_fixture
