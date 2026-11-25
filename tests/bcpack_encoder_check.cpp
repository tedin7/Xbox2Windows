// Round trip of the test-only BCPack encoder (tests/bcpack_encoder.h) through decode_lanes and
// decode_bcpack: random BC1- and BC7-shaped streams with short and long fragment headers, narrowed
// alphabets, trimmed flushed fragments, unflushed fragments, uniform and model-distributed symbols and
// zero padding.
// --mutate window interleaves with a 7-byte instead of an 8-byte cycle budget; the decoder must then
// disagree, which shows the check depends on the shared-stream arbitration (negative control).
#include "bcpack_encoder.h"

#include <cstdio>
#include <cstring>

namespace {

const uint8_t AB1[12] = { 7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7 };
const uint8_t AB3[12] = { 3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7 };
const uint8_t AB4[12] = { 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5 };
const uint8_t AB5[12] = { 3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5 };
const uint8_t AB7[16] = { 3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4 };

}  // namespace

int main(int argc, char** argv) {
    const bool mutate = argc == 3 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "window") == 0;
    if (argc != 1 && !mutate) {
        std::fprintf(stderr, "usage: bcpack_encoder_check [--mutate window]\n");
        return 64;
    }
    bcpack_test::Rng rng{ 2026 };
    int failures = 0;
    size_t bytes = 0, symbols_total = 0;
    const unsigned formats[5] = { 1, 7, 4, 5, 3 };
    for (int t = 0; t < 250; ++t) {
        const unsigned fmt = formats[t % 5];
        const uint8_t* ab = (fmt == 7) ? AB7 : (fmt == 4) ? AB4 : (fmt == 5) ? AB5 : (fmt == 3) ? AB3 : AB1;
        const size_t n = (fmt == 7) ? 16 : 12;
        const size_t blocks = 1 + rng.below(t < 235 ? 96 : 3000);
        const size_t target = (fmt == 1 || fmt == 4) ? blocks : 2 * blocks;
        bcpack_test::Options o;
        o.max_fragment = 1 + rng.below(t % 3 == 0 ? 600 : 150);  // long fragments need three-byte headers
        o.uniform = t % 5 == 0;
        o.unflushed = t % 4 >= 2;
        o.bc7_lane = (fmt == 7) ? 15 : -1;
        o.padding = rng.below(3);
        if (mutate) o.window = 7;
        std::vector<uint8_t> stream;
        std::vector<uint16_t> expected, got;
        if (!bcpack_test::make_stream(ab, n, target, rng, o, stream, expected)) {
            std::fprintf(stderr, "case %d: the encoder failed\n", t);
            return 1;
        }
        bytes += stream.size();
        symbols_total += expected.size();
        const bool lanes_ok = x2w_bcpack::decode_lanes(stream.data(), stream.size(), ab, n, target, got);
        std::vector<uint8_t> out(blocks * ((fmt == 1 || fmt == 4) ? 8 : 16));
        size_t produced = 0;
        const bool bcpack_ok = x2w_bcpack::decode_bcpack(stream.data(), stream.size(), fmt, out.data(),
                                                         out.size(), &produced);
        if (!lanes_ok || got != expected || !bcpack_ok || produced != out.size()) {
            if (failures++ < 5) {
                std::fprintf(stderr, "case %d (BC%u, %zu blocks, %zu bytes): lanes=%d symbols-equal=%d bcpack=%d\n", t,
                             fmt, blocks, stream.size(), lanes_ok, got == expected, bcpack_ok);
            }
        }
    }
    std::printf("bcpack_encoder_check: 250 streams, %zu bytes, %zu symbols, %d failed\n", bytes, symbols_total, failures);
    return failures == 0 ? 0 : 1;
}
