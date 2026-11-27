#pragma once
#include <cstdint>
#include <string>

namespace x2w {
struct LocalDataAccess {
    unsigned offset, address, data0, data1, destination, count;
    unsigned pair_words; // Width of each independently offset element; zero for a single address.
    bool read, stride64, return_old;
    const char *atomic_ir, *atomic_hlsl;
    bool subtract;
    unsigned byte_offset(unsigned element) const {
        if (!pair_words) return offset + 4 * element;
        unsigned encoded = (offset >> (8 * (element / pair_words))) & 255;
        return 4 * (encoded * pair_words * (stride64 ? 64 : 1) + element % pair_words);
    }
    unsigned data_register(unsigned element) const {
        return pair_words && element >= pair_words ? data1 + element - pair_words : data0 + element;
    }
};
LocalDataAccess decode_local_data(uint32_t instruction, uint32_t operands);

// Expand compute instruction words in the register-based HLSL frontend.
// Bit zero of exec is the current lane's EXEC bit. lds is groupshared uint[].
std::string expand_gfx1013_compute(const std::string &source);
} // namespace x2w
