// Input for tests/startup_host_cpu_debug_check.py: a program that executes SHA-extension and RDPID
// instructions, which a host processor may lack. It prints the SHA-256 digest of "abc" computed
// with SHA256RNDS2, SHA256MSG1 and SHA256MSG2, how many threads read RDPID when released together,
// and whether two RDPID reads agree with RDTSCP.
#include <immintrin.h>
#include <intrin.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <utility>
#include <vector>

#if defined(__clang__)
#define HOST_CPU_TARGET __attribute__((target("sha,sse4.1,ssse3,rdpid")))
#else
#define HOST_CPU_TARGET
#endif

namespace {
const std::uint32_t round_constants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// One block: the padded message "abc", as sixteen big-endian words.
HOST_CPU_TARGET __declspec(noinline) void sha256_abc(std::uint32_t digest[8])
{
    const std::uint32_t initial[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::uint32_t words[16] = {0x61626380};
    words[15] = 24;
    __m128i w[16];
    for (int i = 0; i < 4; ++i) w[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(words + 4 * i));
    __m128i abef = _mm_set_epi32(static_cast<int>(initial[0]), static_cast<int>(initial[1]), static_cast<int>(initial[4]), static_cast<int>(initial[5]));
    __m128i cdgh = _mm_set_epi32(static_cast<int>(initial[2]), static_cast<int>(initial[3]), static_cast<int>(initial[6]), static_cast<int>(initial[7]));
    for (int t = 0; t < 16; ++t) {
        if (t >= 4) {
            __m128i next = _mm_sha256msg1_epu32(w[t - 4], w[t - 3]);
            next = _mm_add_epi32(next, _mm_alignr_epi8(w[t - 1], w[t - 2], 4));
            w[t] = _mm_sha256msg2_epu32(next, w[t - 1]);
        }
        __m128i wk = _mm_add_epi32(w[t], _mm_loadu_si128(reinterpret_cast<const __m128i*>(round_constants + 4 * t)));
        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, wk);
        wk = _mm_shuffle_epi32(wk, 0x0E);
        abef = _mm_sha256rnds2_epu32(abef, cdgh, wk);
    }
    std::uint32_t feba[4], hgdc[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(feba), abef);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(hgdc), cdgh);
    const std::uint32_t out[8] = {feba[3], feba[2], hgdc[3], hgdc[2], feba[1], feba[0], hgdc[1], hgdc[0]};
    for (int i = 0; i < 8; ++i) digest[i] = initial[i] + out[i];
}

HOST_CPU_TARGET __declspec(noinline) unsigned read_processor_id()
{
    return _rdpid_u32();
}

// Separate RDPID sites (the added constant keeps the linker from folding them into one function).
template <unsigned N>
HOST_CPU_TARGET __declspec(noinline) unsigned processor_id_site()
{
    return _rdpid_u32() + N;
}

template <unsigned... N>
constexpr std::array<unsigned (*)(), sizeof...(N)> make_sites(std::integer_sequence<unsigned, N...>)
{
    return {processor_id_site<N>...};
}
constexpr auto sites = make_sites(std::make_integer_sequence<unsigned, 32>{});

// Several threads reach each site together, before it has run anywhere: on a host without RDPID they
// fault at the same time, and every one has to continue after the debugger replaces the instruction.
// Returns how many threads passed every site.
int race_sites(int thread_count)
{
    std::atomic<int> arrived[sites.size()] = {};
    std::atomic<int> finished{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < thread_count; ++i)
        threads.emplace_back([&] {
            for (std::size_t site = 0; site < sites.size(); ++site) {
                arrived[site].fetch_add(1);
                while (arrived[site].load() < thread_count) _mm_pause();
                sites[site]();
            }
            finished.fetch_add(1);
        });
    for (auto& thread : threads) thread.join();
    return finished.load();
}
}  // namespace

int main()
{
    std::uint32_t digest[8];
    sha256_abc(digest);
    std::printf("sha256 abc ");
    for (std::uint32_t word : digest) std::printf("%08x", word);
    std::printf("\n");
    std::printf("rdpid threads %d of 8\n", race_sites(8));
    // A thread can move between processors, so RDPID is compared with RDTSCP on both sides.
    bool agrees = false;
    for (int attempt = 0; attempt < 8 && !agrees; ++attempt) {
        unsigned before = 0, after = 0;
        __rdtscp(&before);
        const unsigned id = read_processor_id();
        __rdtscp(&after);
        agrees = id == before || id == after;
    }
    std::printf("rdpid %s\n", agrees ? "agrees with rdtscp" : "differs from rdtscp");
    return 0;
}
