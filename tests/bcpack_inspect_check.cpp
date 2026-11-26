// Checks src/startup_stubs/bcpack_inspect.h (Refs #1198): the class of a failed BCPack request and what the
// lanes had done when it failed, from the repository's synthetic streams. No GPU, no device, no game data.
// Each stream decodes with decode_bcpack as it is; the cases change only the destination size, the input
// length or one input byte. --mutate collapse is the negative control: every failure is reported as one
// class, which this test must reject.
#include "bcpack_encoder.h"
#include "startup_stubs/bcpack_inspect.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using x2w_bcpack::InspectClass;
using x2w_bcpack::InspectResult;
using Bytes = std::vector<unsigned char>;

int failures = 0;
int cases = 0;
bool collapse = false;

void check(bool cond, const std::string& name)
{
    ++cases;
    if (!cond) {
        std::printf("FAIL %s\n", name.c_str());
        std::fflush(stdout);
        ++failures;
    }
}

Bytes from_hex(const std::string& text)
{
    Bytes out;
    for (std::size_t i = 0; i + 1 < text.size(); i += 2)
        out.push_back(static_cast<unsigned char>(std::stoi(text.substr(i, 2), nullptr, 16)));
    return out;
}

// 64 BC1 blocks, 515 bytes: tools/_bcpack_rans/vectors/synthetic_11.json.
const std::string bc1_hex =
    "15141b1bdf1013143d3e0911032814030101f51c011005117117b7af000ffd546cd2eaa47468dc766c102bed1d6501f4"
    "1b1891e3c1741b181c27031cd50549282123f3a681061d6bab65db8a93dca83da80466f51a543ddf1a934a771b363c93"
    "6022312ac4a3259f97e1fca17966d0bba11828c4c727ae63d6dc02edbb720a0a2faef42ae4ec11a791d6c23d1f5fe9e9"
    "f57134c7521acb80d40a22fd095bf964ef543aaa20930639841501ff6dc74cba235d2b5d255e3db2451915498be65eb9"
    "082a450c74da1af0d71601bcce47f079f31a58e96adf0377271173850de4830100636bbf674385cd75223f3b0c4d0bea"
    "dcf70903010cf50ba316883d93202516b7d82c59e89f04872e7e04c16b42fc1597b515132b1860017207c456e012a7f4"
    "c79c8e7bfbfba01196077c347d05db970b3299dff62cdef05d2149643d6a7e206bf9b1792220b694644fa3d20d841e44"
    "7009518b3e6736077a9abb357994030f248f7711afcd1d0534df3985578986c5a27ad4139aed76526f2c253de0231df3"
    "a2ee66b84e333af3caa4422c882bb43fd1916af12750045d96b6c41feb95caf603010448119f6d23f61e50817efb9118"
    "0eb7a84e5609f6e835e1f352170a01ae4b3defff85a559da2e66066ff630362d99b05f8625d5b3462762f2d0d287f796"
    "009c35d22b8814856b20888760f0c10c7772d3d32003e53d9720463194e0a4c080f64c";

// One BC7 mode 6 block, 104 bytes: tests/bcpack_mode6_vector.py.
const std::string bc7_one_hex =
    "101010101010101004040303030304040101010101010140000000000000024002027546281800000101df5f4b7c0341"
    "0202024110101010040404031010101040041001030303044001080001010104000000080000000041020994e4db27a4"
    "410511fe6b020005";

// Two BC7 mode 6 blocks, 132 bytes: tests/bc7_mode6_multiblock_check.py.
const std::string bc7_two_hex =
    "101010101010101005050505050506060101010101010110000000000000042002027546281800000303df5f4b7c0521"
    "0303507823370431101010101010101006050605050505051010010101010110200801000000000400000008e4db27a4"
    "21090294fe6b020431330404c4fb37250202bd19500f034141050934f4cb070502414111";

InspectResult inspect(const Bytes& input, unsigned format, std::size_t output_size)
{
    InspectResult result = x2w_bcpack::inspect_bcpack(input.data(), input.size(), format, output_size);
    if (collapse && result.cls != InspectClass::ok) result.cls = InspectClass::cycle_limit;
    return result;
}

bool decodes(const Bytes& input, unsigned format, std::size_t output_size)
{
    Bytes out(output_size ? output_size : 1);
    std::size_t produced = 0;
    return x2w_bcpack::decode_bcpack(input.data(), input.size(), format, out.data(), output_size, &produced);
}

// One case: the class and the lane counts. The class must agree with decode_bcpack saying yes or no.
void expect(const std::string& name, const Bytes& input, unsigned format, std::size_t output_size, InspectClass cls,
            std::size_t bytes_read, uint32_t min_symbols, uint32_t max_symbols, uint32_t target)
{
    const InspectResult got = inspect(input, format, output_size);
    check(got.cls == cls, name + ": class " + x2w_bcpack::inspect_class_name(got.cls) + ", want " +
                              x2w_bcpack::inspect_class_name(cls));
    check(got.bytes_read == bytes_read, name + ": bytes_read " + std::to_string(got.bytes_read));
    check(got.min_symbols_per_lane == min_symbols && got.max_symbols_per_lane == max_symbols,
          name + ": symbols per lane " + std::to_string(got.min_symbols_per_lane) + "-" +
              std::to_string(got.max_symbols_per_lane));
    check(got.target_symbols_per_lane == target, name + ": target " + std::to_string(got.target_symbols_per_lane));
    check(decodes(input, format, output_size) == (cls == InspectClass::ok), name + ": agrees with decode_bcpack");
}

// Same, for the cases whose lane counts are not the point.
void expect_class(const std::string& name, const Bytes& input, unsigned format, std::size_t output_size,
                  InspectClass cls)
{
    const InspectResult got = inspect(input, format, output_size);
    check(got.cls == cls, name + ": class " + x2w_bcpack::inspect_class_name(got.cls) + ", want " +
                              x2w_bcpack::inspect_class_name(cls));
    check(decodes(input, format, output_size) == (cls == InspectClass::ok), name + ": agrees with decode_bcpack");
}

Bytes truncated(Bytes input, std::size_t length)
{
    input.resize(length);
    return input;
}

Bytes padded(Bytes input, unsigned char byte, std::size_t count)
{
    input.insert(input.end(), count, byte);
    return input;
}

Bytes with_byte(Bytes input, std::size_t index, unsigned char value)
{
    input[index] = value;
    return input;
}

void test_bc1()
{
    const Bytes bc1 = from_hex(bc1_hex);
    check(bc1.size() == 515, "bc1 vector size");
    using C = InspectClass;
    expect("bc1 exact", bc1, 1, 512, C::ok, 515, 64, 64, 64);
    expect("bc1 zero padding", padded(bc1, 0, 4), 1, 512, C::ok, 515, 64, 64, 64);
    expect("bc1 one byte after the end", padded(bc1, 0x5a, 1), 1, 512, C::trailing_input, 515, 64, 64, 64);
    // A request for the first part of a longer stream: every lane stops at the requested count.
    expect("bc1 1 block of 64", bc1, 1, 8, C::request_shorter_than_stream, 91, 1, 1, 1);
    expect("bc1 16 blocks of 64", bc1, 1, 128, C::request_shorter_than_stream, 196, 16, 16, 16);
    expect("bc1 63 blocks of 64", bc1, 1, 504, C::request_shorter_than_stream, 515, 63, 63, 63);
    // A request for more than the stream holds: every lane ends at the stream's own count.
    expect("bc1 65 blocks of 64", bc1, 1, 520, C::request_longer_than_stream, 515, 64, 64, 65);
    expect("bc1 1000 blocks of 64", bc1, 1, 8000, C::request_longer_than_stream, 515, 64, 64, 1000);
    // A cut stream: the input ends inside a fragment.
    expect("bc1 cut at 514", truncated(bc1, 514), 1, 512, C::truncated_in_fragment, 514, 63, 64, 64);
    expect("bc1 cut at 300", truncated(bc1, 300), 1, 512, C::truncated_in_fragment, 300, 31, 32, 64);
    expect("bc1 cut at 100", truncated(bc1, 100), 1, 512, C::truncated_in_fragment, 100, 2, 3, 64);
    expect("bc1 cut at 1", truncated(bc1, 1), 1, 512, C::truncated_in_fragment, 1, 0, 0, 64);
    // Damaged streams.
    expect_class("bc1 first byte 0", with_byte(bc1, 0, 0), 1, 512, C::fragment_not_home);
    expect_class("bc1 byte 2 set to 11", with_byte(bc1, 2, 11), 1, 512, C::cycle_limit);
    expect_class("bc1 byte 382 flipped", with_byte(bc1, 382, static_cast<unsigned char>(bc1[382] ^ 0x80)), 1, 512,
                 C::end_state_invalid);
    // A BC1 stream read with the BC7 lane layout.
    expect("bc1 stream as bc7", bc1, 7, 1024, C::fragment_not_home, 133, 4, 4, 128);
    // The request, not the stream.
    expect_class("bc1 510 bytes", bc1, 1, 510, C::destination_misaligned);
    expect_class("bc1 12 bytes (one and a half blocks)", bc1, 1, 12, C::destination_misaligned);
    expect_class("bc1 0 bytes", bc1, 1, 0, C::bad_arguments);
    expect_class("bc1 empty input", Bytes(), 1, 512, C::bad_arguments);
    expect_class("bc1 format 6", bc1, 6, 512, C::unsupported_format);
    expect_class("bc1 format 0", bc1, 0, 512, C::unsupported_format);
    check(x2w_bcpack::inspect_bcpack(nullptr, 515, 1, 512).cls == C::bad_arguments, "null input");
    check(x2w_bcpack::inspect_bcpack(bc1.data(), bc1.size(), 1, x2w_bcpack::inspect_max_output + 8).cls ==
              C::bad_arguments,
          "destination over the inspection limit");
    // A tiny source declaring a 256 MiB BC7 output is refused before the ~1 GiB lane-symbol allocation.
    const Bytes bc7_tiny = from_hex(bc7_one_hex);
    for (const std::size_t declared : { std::size_t{256} << 20, x2w_bcpack::inspect_max_output + 16 }) {
        const InspectResult over = x2w_bcpack::inspect_bcpack(bc7_tiny.data(), bc7_tiny.size(), 7, declared);
        check(over.cls == C::bad_arguments && over.bytes_read == 0 && over.target_symbols_per_lane == 0,
              "bc7 declared " + std::to_string(declared) + " bytes refused before scratch");
    }
    check(x2w_bcpack::inspect_max_output == (std::size_t{16} << 20), "inspection cap is the 16 MiB chunk window");
}

void test_bc7()
{
    const Bytes one = from_hex(bc7_one_hex), two = from_hex(bc7_two_hex);
    check(one.size() == 104 && two.size() == 132, "bc7 vector sizes");
    using C = InspectClass;
    expect("bc7 one block exact", one, 7, 16, C::ok, 104, 2, 2, 2);
    expect("bc7 one block zero padding", padded(one, 0, 3), 7, 16, C::ok, 104, 2, 2, 2);
    expect("bc7 two blocks exact", two, 7, 32, C::ok, 132, 4, 4, 4);
    expect("bc7 two blocks one byte after the end", padded(two, 0x5a, 1), 7, 32, C::trailing_input, 132, 4, 4, 4);
    expect("bc7 1 block of 2", two, 7, 16, C::request_shorter_than_stream, 132, 2, 2, 2);
    expect("bc7 2 blocks of 1", one, 7, 32, C::request_longer_than_stream, 104, 2, 2, 4);
    expect("bc7 10 blocks of 1", one, 7, 160, C::request_longer_than_stream, 104, 2, 2, 20);
    expect("bc7 3 blocks of 2", two, 7, 48, C::request_longer_than_stream, 132, 4, 4, 6);
    expect("bc7 cut at 103", truncated(one, 103), 7, 16, C::truncated_in_fragment, 103, 0, 1, 2);
    expect("bc7 cut at 60", truncated(one, 60), 7, 16, C::truncated_in_fragment, 60, 0, 1, 2);
    expect("bc7 cut at 1", truncated(one, 1), 7, 16, C::truncated_in_fragment, 1, 0, 0, 2);
    expect_class("bc7 first byte 1", with_byte(one, 0, 1), 7, 16, C::end_state_invalid);
    expect_class("bc7 byte 8 zero", with_byte(one, 8, 0), 7, 16, C::fragment_not_home);
    expect("bc7 stream as bc1", two, 1, 64, C::fragment_not_home, 132, 6, 6, 8);
    expect_class("bc7 15 bytes", one, 7, 15, C::destination_misaligned);
    expect_class("bc7 24 bytes (one and a half blocks)", one, 7, 24, C::destination_misaligned);
    // The single mode 6 block decodes to the block the vector was made from.
    Bytes out(16);
    std::size_t produced = 0;
    const Bytes block = from_hex("40317204a56819be023456789abcdef0");
    check(x2w_bcpack::decode_bcpack(one.data(), one.size(), 7, out.data(), out.size(), &produced) && produced == 16 &&
              out == block,
          "bc7 one block decodes to the vector's block");
}

// BC3, BC4 and BC5 have lane layouts too (the alphabets bcpack_encoder_check round-trips through decode_bcpack).
// Streams come from the test-only encoder, which is not a stream a title wrote: the cases check that the class is
// the lane state's reading and never unsupported_format, and that every answer agrees with decode_bcpack.
void test_bc345()
{
    using C = InspectClass;
    struct Layout {
        unsigned format;
        std::size_t lanes, block_bytes, symbols_per_block;
        uint8_t ab[12];
    };
    const Layout layouts[3] = {
        { 3, 12, 16, 2, { 3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7 } },
        { 4, 12, 8, 1, { 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5 } },
        { 5, 12, 16, 2, { 3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5 } },
    };
    bcpack_test::Rng rng{ 77 };
    for (const Layout& l : layouts) {
        const std::string name = "bc" + std::to_string(l.format);
        const std::size_t blocks = 40, size = blocks * l.block_bytes;
        bcpack_test::Options options;
        options.unflushed = true;
        Bytes stream;
        std::vector<uint16_t> symbols;
        check(bcpack_test::make_stream(l.ab, l.lanes, blocks * l.symbols_per_block, rng, options, stream, symbols),
              name + " encoder");
        const InspectResult exact = inspect(stream, l.format, size);
        check(exact.cls == C::ok && exact.bytes_read == stream.size() &&
                  exact.min_symbols_per_lane == blocks * l.symbols_per_block &&
                  exact.max_symbols_per_lane == blocks * l.symbols_per_block,
              name + " exact: " + x2w_bcpack::inspect_class_name(exact.cls));
        expect_class(name + " exact agrees", stream, l.format, size, C::ok);
        expect_class(name + " half the blocks", stream, l.format, size / 2, C::request_shorter_than_stream);
        expect_class(name + " one byte after the end", padded(stream, 0x5a, 1), l.format, size, C::trailing_input);
        expect_class(name + " misaligned", stream, l.format, size - 1, C::destination_misaligned);
        const InspectResult longer = inspect(stream, l.format, size * 2);
        check((longer.cls == C::request_longer_than_stream || longer.cls == C::truncated_in_fragment) &&
                  !decodes(stream, l.format, size * 2),
              name + " twice the blocks: " + x2w_bcpack::inspect_class_name(longer.cls));
        const InspectResult cut = inspect(truncated(stream, stream.size() / 2), l.format, size);
        check((cut.cls == C::truncated_in_fragment || cut.cls == C::request_longer_than_stream) &&
                  !decodes(truncated(stream, stream.size() / 2), l.format, size),
              name + " half the stream: " + x2w_bcpack::inspect_class_name(cut.cls));
        // The BC1 layout is not this one: a stream of another format is named by the lanes, not refused as a format.
        check(inspect(stream, l.format == 4 ? 5 : 4, size).cls != C::unsupported_format, name + " read as another format");
    }
}

// A fragment header whose first byte names an alphabet wider than 8 bits cannot come from the BC1 or BC7
// layouts, whose alphabets never go below 3 bits. A one-lane decode with a 2-bit alphabet reaches it.
void test_lane_report()
{
    using x2w_bcpack::LaneFailure;
    using x2w_bcpack::LaneReport;
    std::vector<uint16_t> symbols;
    LaneReport report;
    const uint8_t narrow[1] = { 2 };
    const uint8_t input[3] = { 0xC0, 0x00, 0x00 };
    check(!x2w_bcpack::decode_lanes(input, sizeof(input), narrow, 1, 1, symbols, &report) &&
              report.failure == LaneFailure::header_invalid,
          "lane report: header names a wider alphabet");
    check(!x2w_bcpack::decode_lanes(input, sizeof(input), narrow, 0, 1, symbols, &report) &&
              report.failure == LaneFailure::bad_arguments,
          "lane report: no lanes");
    const uint8_t wide[1] = { 9 };
    check(!x2w_bcpack::decode_lanes(input, sizeof(input), wide, 1, 1, symbols, &report) &&
              report.failure == LaneFailure::bad_arguments,
          "lane report: alphabet wider than 8 bits");
    // Without a report the call answers as before.
    check(!x2w_bcpack::decode_lanes(input, sizeof(input), narrow, 1, 1, symbols), "lane decode without a report");
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 1) {
        if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "collapse") == 0) {
            collapse = true;
        } else {
            std::fprintf(stderr, "usage: bcpack_inspect_check [--mutate collapse]\n");
            return 2;
        }
    }
    test_bc1();
    test_bc7();
    test_bc345();
    test_lane_report();
    std::printf("bcpack_inspect_check: %d cases, %d failed%s\n", cases, failures, collapse ? " (collapsed control)" : "");
    return failures ? 1 : 0;
}
