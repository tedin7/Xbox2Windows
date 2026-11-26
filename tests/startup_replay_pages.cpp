// Page store for record + replay: a range captured twice is rebuilt byte for byte from the chunks, and the second capture
// writes only what changed. Modes (the stream a mutant would produce, to prove the checks bite):
//   (none)      the chunks as written
//   drop-map    the second MAP chunk never reaches the reader
//   drop-page   one PAGE chunk never reaches the reader
//   corrupt     one stored page has a byte flipped on the way
#include "../src/startup_stubs/replay_pages.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace x2w_replay;

namespace {
struct Chunk { char tag[4]; std::string header, data; };

int fail(const char* what)
{
    std::fprintf(stderr, "startup_replay_pages: FAIL %s\n", what);
    return 1;
}

// The chunk framing the capture file uses, through a byte string: what a file round trip would do.
void frame(std::string& out, const char* tag, const void* header, std::size_t hb, const void* data, std::size_t db)
{
    out.append(tag, 4);
    const std::uint32_t h = static_cast<std::uint32_t>(hb);
    const std::uint64_t d = db;
    out.append(reinterpret_cast<const char*>(&h), 4);
    out.append(reinterpret_cast<const char*>(&d), 8);
    out.append(static_cast<const char*>(header), hb);
    if (db) out.append(static_cast<const char*>(data), db);
}
std::vector<Chunk> unframe(const std::string& in)
{
    std::vector<Chunk> chunks;
    std::size_t at = 0;
    while (at + 16 <= in.size()) {
        Chunk c{};
        std::uint32_t h;
        std::uint64_t d;
        std::memcpy(c.tag, in.data() + at, 4);
        std::memcpy(&h, in.data() + at + 4, 4);
        std::memcpy(&d, in.data() + at + 8, 8);
        at += 16;
        if (at + h + d > in.size()) break;
        c.header.assign(in, at, h);
        c.data.assign(in, at + h, static_cast<std::size_t>(d));
        at += h + d;
        chunks.push_back(std::move(c));
    }
    return chunks;
}
}  // namespace

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "";
    constexpr std::size_t pages = 64;
    constexpr std::uint64_t address = 0x7000000000ull;
    std::vector<unsigned char> memory(pages * page_bytes);
    std::uint32_t seed = 12345;
    const auto next = [&] { seed = seed * 1664525u + 1013904223u; return static_cast<unsigned char>(seed >> 24); };
    // Pages 0-9 random, 10-19 zero, 20-29 copies of page 0 (duplicates), 30-63 random.
    for (std::size_t i = 0; i < memory.size(); ++i) memory[i] = next();
    std::memset(memory.data() + 10 * page_bytes, 0, 10 * page_bytes);
    for (std::size_t i = 20; i < 30; ++i) std::memcpy(memory.data() + i * page_bytes, memory.data(), page_bytes);

    std::string stream;
    PageWriter writer([&](const char* tag, const void* h, std::size_t hb, const void* d, std::size_t db) { frame(stream, tag, h, hb, d, db); });

    if (writer.capture(address + 1, memory.data(), memory.size(), 1)) return fail("an unaligned address was accepted");
    if (writer.capture(address, memory.data(), memory.size() - 1, 1)) return fail("an unaligned size was accepted");
    if (!writer.capture(address, memory.data(), memory.size(), 1)) return fail("first capture");
    const PageStats first = writer.stats();
    const std::vector<unsigned char> before = memory;
    // Change page 3 (random), page 12 (zero -> data), page 40 (data -> zero) and page 50 (becomes a copy of page 0).
    for (std::size_t i = 0; i < page_bytes; ++i) memory[3 * page_bytes + i] ^= 0x5a;
    for (std::size_t i = 0; i < page_bytes; ++i) memory[12 * page_bytes + i] = static_cast<unsigned char>(i | 1);
    std::memset(memory.data() + 40 * page_bytes, 0, page_bytes);
    std::memcpy(memory.data() + 50 * page_bytes, memory.data(), page_bytes);
    const std::size_t after_first = stream.size();
    if (!writer.capture(address, memory.data(), memory.size(), 2)) return fail("second capture");
    const PageStats second = writer.stats();

    // Counts: 54 non-zero pages in the first capture, of which 10 are copies of page 0 (pages 20-29): 44 distinct + 10 duplicates... page 0
    // itself is distinct, so distinct = 64 - 10 zero - 10 duplicate = 44; the zero pages are 10.
    if (first.zero != 10) return fail("first capture zero pages");
    if (first.distinct != 44 || first.duplicate != 10) return fail("first capture distinct / duplicate pages");
    // Second capture: 4 pages changed; page 3 and 12 are new content (2 PAGE chunks), page 40 is zero, page 50 repeats page 0.
    if (second.unchanged - first.unchanged != 60) return fail("second capture should leave 60 pages unchanged");
    if (second.distinct - first.distinct != 2) return fail("second capture should store 2 new pages");
    if (second.bytes_written - first.bytes_written != 2 * page_bytes) return fail("second capture bytes");
    if (stream.size() - after_first > 2 * (page_bytes + 16 + 16) + 16 + 32 + 4 * 64) return fail("second capture wrote more than the changes");

    // The same memory again changes nothing: no chunk at all.
    const std::size_t before_again = stream.size();
    if (!writer.capture(address, memory.data(), memory.size(), 3)) return fail("third capture");
    if (stream.size() != before_again) return fail("an unchanged range wrote a chunk");

    // Replay the stream through the reader (with the mutant applied) and compare the image at each MAP.
    auto chunks = unframe(stream);
    ImageReader reader;
    std::vector<unsigned char> image(memory.size());
    int maps = 0, pages_seen = 0;
    for (auto& c : chunks) {
        const bool is_map = !std::memcmp(c.tag, "MAP ", 4);
        if (!std::memcmp(c.tag, "PAGE", 4)) {
            ++pages_seen;
            if (mode == "drop-page" && pages_seen == 5) continue;
            if (mode == "corrupt" && pages_seen == 5) c.data[100] ^= 1;
        }
        if (is_map && ++maps == 2 && mode == "drop-map") continue;
        if (!reader.apply(c.tag, c.header.data(), c.header.size(), c.data.data(), c.data.size())) {
            if (mode == "drop-page") return fail("reader refused a MAP that names a page it never got (expected for this mutant)");
            return fail("reader refused a chunk");
        }
        if (is_map) {
            if (!reader.read(address, image.data(), image.size())) return fail("image has unmapped pages");
            const std::vector<unsigned char>& want = maps == 1 ? before : memory;
            if (std::memcmp(image.data(), want.data(), want.size()) != 0) {
                std::size_t at = 0;
                while (image[at] == want[at]) ++at;
                std::fprintf(stderr, "startup_replay_pages: FAIL image after MAP %d differs from the memory at page %zu\n", maps, at / page_bytes);
                return 1;
            }
        }
    }
    if (maps != 2) return fail("expected two MAP chunks");
    if (!reader.read(address, image.data(), image.size()) || std::memcmp(image.data(), memory.data(), memory.size()) != 0)
        return fail("the final image is not the memory after the second capture");
    std::printf("startup_replay_pages: ok (first: %llu distinct %llu dup %llu zero; second: %llu unchanged, %llu new)\n",
                static_cast<unsigned long long>(first.distinct), static_cast<unsigned long long>(first.duplicate),
                static_cast<unsigned long long>(first.zero), static_cast<unsigned long long>(second.unchanged - first.unchanged),
                static_cast<unsigned long long>(second.distinct - first.distinct));
    return 0;
}
