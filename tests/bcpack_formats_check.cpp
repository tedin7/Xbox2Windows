// Oracle-derived golden hashes of source-built synthetic streams. No SDK at test/run time.
// Optional --oracle ABSOLUTE_DLL_PATH regenerates hashes using an installed developer oracle.
// See docs/experiments/bcpack-standalone.md for provenance and commands.
// SPDX-License-Identifier: MIT
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "bcpack_format_fixtures.h"
#include "bcpack_formats_goldens.h"
#include <cstdio>
#include <string>

namespace {
using Oracle = long (__cdecl*)(size_t,void*,size_t,const void*,size_t);
Oracle oracle = nullptr;
bool mutate = false;
size_t cases = 0;

uint64_t hash(uint64_t h, const std::vector<uint8_t>& bytes) {
    for (uint8_t b : bytes) { h ^= b; h *= 1099511628211ull; }
    return h;
}

bool check(unsigned fmt, const std::vector<uint16_t>& s, size_t blocks, uint64_t& h) {
    std::vector<uint8_t> packed;
    if (!bcpack_fixture::encode(fmt,s,blocks,packed)) return false;
    const size_t size = blocks*bcpack_fixture::block_bytes(fmt);
    std::vector<uint8_t> expected(size), guarded(size+32,0xa5);
    if (oracle) {
        if (oracle(fmt,expected.data(),size,packed.data(),packed.size()) < 0) return false;
    } else {
        bool first = true;
        for (int path = 0; path < x2w_bcpack::LANE_PATH_COUNT; ++path) {
            if (!x2w_bcpack::lane_path_supported(path)) continue;
            x2w_bcpack::lane_path_override = path;
            std::fill(guarded.begin(),guarded.end(),0xa5);
            size_t produced = 123;
            if (!x2w_bcpack::decode_bcpack(packed.data(),packed.size(),fmt,guarded.data()+16,size,&produced) || produced != size)
                return false;
            for (size_t i = 0; i < 16; ++i)
                if (guarded[i] != 0xa5 || guarded[size+16+i] != 0xa5) return false;
            if (first) {
                std::copy(guarded.begin()+16,guarded.end()-16,expected.begin());
                first = false;
            } else if (!std::equal(expected.begin(),expected.end(),guarded.begin()+16)) return false;
        }
    }
    if (mutate && fmt == 2 && cases == 0) expected[0] ^= 1;
    h = hash(h,expected);
    ++cases;
    return true;
}

bool group(unsigned fmt, int mode, uint64_t expected) {
    uint64_t h = 14695981039346656037ull;
    cases = 0;
    if (mode >= 0) {
        for (unsigned r = 0; r < 256; ++r) {
            const unsigned ep[] = {r,0,r}, ix[] = {0,r,(r*73+19)&255};
            for (unsigned k = 0; k < 3; ++k)
                if (!check(fmt,bcpack_fixture::reference_symbols(fmt,ep[k],ix[k],mode),1,h)) return false;
        }
    } else {
        for (size_t blocks : {size_t(2),size_t(17),size_t(255),size_t(256),size_t(257),size_t(1024)})
            if (!check(fmt,bcpack_fixture::random_symbols(fmt,blocks,0xbc9900+fmt+blocks),blocks,h)) return false;
    }
    std::printf("BC%u mode %d: %zu cases, %016llx%s\n",fmt,mode,cases,
                static_cast<unsigned long long>(h), oracle || h == expected ? "" : " MISMATCH");
    return oracle || h == expected;
}

bool failures() {
    for (unsigned fmt = 1; fmt <= 7; ++fmt) {
        auto s = bcpack_fixture::random_symbols(fmt,17,0x1122+fmt);
        std::vector<uint8_t> packed;
        if (!bcpack_fixture::encode(fmt,s,17,packed)) return false;
        const size_t size = 17*bcpack_fixture::block_bytes(fmt);
        std::vector<uint8_t> out(size+32,0xa5);
        auto reject = [&](size_t input_size, size_t output_size, unsigned format) {
            size_t produced = 123;
            return !x2w_bcpack::decode_bcpack(packed.data(),input_size,format,out.data()+16,output_size,&produced) && produced == 0 &&
                std::all_of(out.begin(),out.end(),[](uint8_t b){return b == 0xa5;});
        };
        if (!reject(1,size,fmt) || !reject(packed.size(),size-1,fmt) ||
            !reject(packed.size(),size,0) || !reject(packed.size(),size,8)) return false;
        packed.push_back(0xff);
        if (!reject(packed.size(),size,fmt)) return false;
    }
    // A valid entropy stream can still contain an undefined BC6H mode.
    std::vector<uint16_t> s(32,0);
    s[30] = 14;
    std::vector<uint8_t> packed;
    if (!bcpack_fixture::encode(6,s,1,packed)) return false;
    uint8_t out[16] = {}; size_t produced = 123;
    return !x2w_bcpack::decode_bcpack(packed.data(),packed.size(),6,out,sizeof(out),&produced) && produced == 0;
}
} // namespace

int main(int argc, char** argv) {
    HMODULE library = nullptr;
    if (argc == 3 && std::string(argv[1]) == "--oracle") {
        library = LoadLibraryExA(argv[2],nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!library) return 2;
        oracle = reinterpret_cast<Oracle>(GetProcAddress(library,"BCPackDecompress"));
        if (!oracle) { FreeLibrary(library); return 2; }
    } else if (argc == 2 && std::string(argv[1]) == "--mutate") mutate = true;
    else if (argc != 1) return 2;
    bool ok = true;
    for (unsigned fmt : {1u,2u,6u}) {
        const unsigned modes = fmt == 6 ? 14 : 1;
        for (unsigned mode = 0; mode < modes; ++mode)
            ok = group(fmt,mode,BCPACK_REFERENCE_GOLDENS[fmt == 1 ? 0 : fmt == 2 ? 1 : mode+2]) && ok;
    }
    for (unsigned fmt = 1; fmt <= 7; ++fmt)
        ok = group(fmt,-1,BCPACK_RANDOM_GOLDENS[fmt-1]) && ok;
    if (!oracle) ok = failures() && ok;
    if (library) FreeLibrary(library);
    std::puts(ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
