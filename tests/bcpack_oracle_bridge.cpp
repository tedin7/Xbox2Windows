// Synthetic input bridge for the optional development-only constant recovery tool.
// SPDX-License-Identifier: MIT
#include "bcpack_format_fixtures.h"
#include <cstring>
extern "C" __declspec(dllexport) size_t encode_symbols(unsigned fmt, const uint16_t* symbols,
    size_t target, uint8_t* output, size_t capacity) {
    if (fmt < 1 || fmt > 7 || !symbols || !output || !target || target%bcpack_fixture::per(fmt)) return 0;
    std::vector<uint16_t> s(symbols,symbols+target*bcpack_fixture::lanes(fmt));
    std::vector<uint8_t> packed;
    if (!bcpack_fixture::encode(fmt,s,target/bcpack_fixture::per(fmt),packed) || packed.size() > capacity) return 0;
    std::memcpy(output,packed.data(),packed.size());
    return packed.size();
}
extern "C" __declspec(dllexport) size_t storage_symbols(unsigned fmt, uint16_t* output, size_t capacity) {
    if ((fmt != 2 && fmt != 6) || !output) return 0;
    auto s = bcpack_fixture::random_symbols(fmt,4,0x7700+fmt);
    if (capacity < s.size()) return 0;
    std::copy(s.begin(),s.end(),output);
    return s.size();
}
