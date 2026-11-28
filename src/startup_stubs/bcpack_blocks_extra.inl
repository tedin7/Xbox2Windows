// Standalone BC1, BC2 and BC6H block reconstruction from BCPack lane symbols.
// SPDX-License-Identifier: MIT

template<class State, size_t Count>
inline State bcpack_predictor(size_t block, size_t reference, const State (&history)[256],
                              const State (&dictionary)[Count]) {
    if (reference == 0) return {};
    if (reference <= block) return history[(block - reference) & 255];
    const size_t index = reference - block - 1;
    return index < Count ? dictionary[index] : State{};
}

inline void bcpack_quad(const uint8_t (&rank)[256][4], uint16_t symbol, size_t quadrant,
                        const uint8_t* predictor, uint8_t* positions) {
    const size_t base = (quadrant >> 1) * 8 + (quadrant & 1) * 2;
    constexpr size_t offsets[4] = {0, 1, 4, 5};
    for (size_t k = 0; k < 4; ++k) {
        const size_t p = base + offsets[k];
        positions[p] = static_cast<uint8_t>((predictor[p] + rank[symbol & 255][k]) & 3);
    }
}

inline bool decode_bc1_blocks(const std::vector<uint16_t>& symbols, size_t blocks,
                              uint8_t* output, size_t capacity) {
    if (blocks > capacity / 8 || blocks > symbols.size() / 12) return false;
    constexpr uint8_t widths[6] = {5,6,5,5,6,5};
    Bc1FullState history[256] = {};
    for (size_t i = 0; i < blocks; ++i) {
        // Lane 1 is the index distance. Lane 0 is a zigzag delta from it to
        // the endpoint distance; they can select different historical blocks.
        const size_t xr = symbols[blocks + i] & 255;
        const size_t er = static_cast<size_t>(static_cast<int>(xr) + unzigzag(symbols[i])) & 255;
        const auto ep = bcpack_predictor(i,er,history,BC1_FULL_DICTIONARY);
        const auto ix = bcpack_predictor(i,xr,history,BC1_FULL_DICTIONARY);
        Bc1FullState state{};
        for (size_t c = 0; c < 6; ++c) {
            state.ep[c] = static_cast<uint8_t>((ep.ep[c] + unzigzag(symbols[(2+c)*blocks+i])) & ((1u<<widths[c])-1));
        }
        for (size_t q = 0; q < 4; ++q) {
            bcpack_quad(BC1_POSITION_RANK,symbols[(8+q)*blocks+i],q,ix.ix,state.ix);
        }
        pack_colour(state.ep,state.ix,output+i*8);
        history[i&255] = state;
    }
    return true;
}

inline bool decode_bc2_blocks(const std::vector<uint16_t>& symbols, size_t blocks,
                              uint8_t* output, size_t capacity) {
    if (blocks > capacity / 16 || blocks > symbols.size() / 30) return false;
    const size_t target = blocks*2;
    Bc2State history[256] = {};
    for (size_t i = 0; i < blocks; ++i) {
        const uint8_t d = interleave(symbols[2*i],symbols[2*i+1]);
        const size_t xr = interleave(symbols[target+2*i],symbols[target+2*i+1]);
        const size_t er = static_cast<size_t>(static_cast<int>(xr)+unzigzag(d)) & 255;
        const auto ep = bcpack_predictor(i,er,history,BC2_FULL_DICTIONARY);
        const auto ix = bcpack_predictor(i,xr,history,BC2_FULL_DICTIONARY);
        Bc2State state{};
        uint64_t alpha = 0;
        for (size_t p = 0; p < 16; ++p) {
            // BC2 stores explicit four-bit alpha, biased around eight. Its
            // predictor follows endpoint history, independently of colour indices.
            state.alpha[p] = static_cast<uint8_t>((ep.alpha[p]+unzigzag(symbols[(2+p/2)*target+2*i+p%2])) & 15);
            alpha |= static_cast<uint64_t>(state.alpha[p]^8) << (4*p);
        }
        for (size_t c = 0; c < 3; ++c) {
            const unsigned bits = c == 1 ? 6 : 5;
            for (size_t e = 0; e < 2; ++e) {
                state.ep[3*e+c] = static_cast<uint8_t>((ep.ep[3*e+c]+unzigzag(symbols[(10+c)*target+2*i+e])) & ((1u<<bits)-1));
            }
        }
        for (size_t q = 0; q < 4; ++q) {
            bcpack_quad(BC2_POSITION_RANK,symbols[(13+q/2)*target+2*i+q%2],q,ix.ix,state.ix);
        }
        std::memcpy(output+i*16,&alpha,8);
        pack_colour(state.ep,state.ix,output+i*16+8);
        history[i&255] = state;
    }
    return true;
}

inline uint16_t bcpack_wide(uint16_t even, uint16_t odd) {
    uint16_t value = 0;
    for (unsigned bit = 0; bit < 8; ++bit) {
        value |= static_cast<uint16_t>(((even>>bit)&1)<<(2*bit) | ((odd>>bit)&1)<<(2*bit+1));
    }
    return value;
}

inline uint16_t bcpack_add_aligned(uint16_t predictor, int32_t delta, unsigned shift) {
    const uint32_t aligned = static_cast<uint32_t>(static_cast<uint16_t>(delta)) << shift;
    return static_cast<uint16_t>((predictor+aligned) & (0xffffu<<shift));
}

inline void bcpack_put_bit(uint8_t* output, size_t at, unsigned bit) {
    output[at/8] |= static_cast<uint8_t>((bit&1)<<(at%8));
}

inline bool decode_bc6_blocks(const std::vector<uint16_t>& symbols, size_t blocks,
                              uint8_t* output, size_t capacity) {
    if (blocks > capacity/16 || blocks > symbols.size()/32) return false;
    const size_t target = blocks*2;
    constexpr uint8_t codes[14] = {0,1,2,6,10,14,18,22,26,30,3,7,11,15};
    Bc6State history[256] = {};
    for (size_t i = 0; i < blocks; ++i) {
        const unsigned mode = symbols[15*target+2*i]+1;
        if (mode < 1 || mode > 14) return false;
        const bool two = mode <= 10;
        const unsigned part = symbols[15*target+2*i+1]&31;
        const uint8_t d = interleave(symbols[2*i],symbols[2*i+1]);
        const size_t xr = interleave(symbols[target+2*i],symbols[target+2*i+1]);
        const size_t er = static_cast<size_t>(static_cast<int>(xr)+unzigzag(d)) & 255;
        const auto ep = bcpack_predictor(i,er,history,BC6_DICTIONARY);
        const auto ix = bcpack_predictor(i,xr,history,BC6_DICTIONARY);
        Bc6State state{};
        // Keep signed/bias-centred values left aligned, so changing modes can
        // reuse a predictor without confusing its original precision with this mode's.
        for (size_t c = 0; c < 3; ++c) {
            const unsigned s0 = 16-BC6_ENDPOINT_BITS[mode][0];
            const unsigned sc = 16-BC6_ENDPOINT_BITS[mode][c+1];
            const uint16_t v0 = bcpack_wide(symbols[(2+c)*target+2*i],symbols[(2+c)*target+2*i+1]);
            const uint16_t v1 = bcpack_wide(symbols[(5+c)*target+2*i],symbols[(5+c)*target+2*i+1]);
            state.ep[0][c] = bcpack_add_aligned(ep.ep[0][c],unzigzag(v0),s0);
            state.ep[1][c] = bcpack_add_aligned(ep.ep[1][c],unzigzag(v1),sc);
            if (two) {
                state.ep[2][c] = bcpack_add_aligned(ep.ep[2][c],unzigzag(symbols[(8+c)*target+2*i]),sc);
                state.ep[3][c] = bcpack_add_aligned(ep.ep[3][c],unzigzag(symbols[(8+c)*target+2*i+1]),sc);
            } else {
                // One-subset blocks still update the hidden second-subset
                // history used by a later two-subset block.
                state.ep[2][c] = static_cast<uint16_t>((state.ep[0][c]>>s0)<<sc);
                state.ep[3][c] = state.ep[1][c];
            }
        }
        const unsigned bits = two ? 3 : 4;
        for (size_t pair = 0; pair < 8; ++pair) {
            const uint16_t v = symbols[(11+pair/2)*target+2*i+pair%2];
            for (size_t h = 0; h < 2; ++h) {
                // The last two pairs swap the lane's even/odd bit assignment.
                const unsigned parity = pair < 6 ? 1-static_cast<unsigned>(h) : static_cast<unsigned>(h);
                uint16_t nibble = 0;
                for (unsigned k = 0; k < 4; ++k) nibble |= ((v>>(2*k+parity))&1)<<k;
                const size_t p = 2*pair+h;
                state.ix[p] = bcpack_add_aligned(ix.ix[p],unzigzag(nibble),16-bits);
            }
        }
        uint8_t* block = output+i*16;
        std::memset(block,0,16);
        const size_t header = two ? 82 : 65;
        for (size_t at = 0; at < header; ++at) {
            const auto wire = BC6_WIRE[mode-1][at];
            unsigned value;
            if (wire.field == 0) value = codes[mode-1];
            else if (wire.field == 1) value = part;
            else {
                const unsigned k = (wire.field-2)/3, c = (wire.field-2)%3;
                const unsigned shift = 16-BC6_ENDPOINT_BITS[mode][k == 0 ? 0 : c+1];
                value = (state.ep[k][c]^0x8000u)>>shift;
            }
            bcpack_put_bit(block,at,value>>wire.bit);
        }
        size_t at = header;
        const size_t anchor = two ? ANCHOR_2[part] : 0;
        for (size_t p = 0; p < 16; ++p) {
            const unsigned width = bits-static_cast<unsigned>(p == 0 || p == anchor);
            const unsigned value = (state.ix[p]>>(16-bits))+(1u<<(bits-1));
            for (unsigned b = 0; b < width; ++b) bcpack_put_bit(block,at++,value>>b);
        }
        history[i&255] = state;
    }
    return true;
}
