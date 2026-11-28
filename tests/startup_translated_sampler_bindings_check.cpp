// Checks src/startup_stubs/translated_sampler_bindings.h without a device: how 16-byte title sampler
// words decode to desktop samplers, which manifest lines the stand-in accepts, and what a snapshot of
// title sampler memory keeps.
#include "startup_stubs/sampler_census.h"

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace {
using namespace x2w_startup;
int failures, cases;
bool shared_axis;  // --mutate shared-axis

void fail(const char* name, const char* what, unsigned got, unsigned expected)
{
    std::printf("FAIL %s: %s 0x%x, expected 0x%x\n", name, what, got, expected);
    ++failures;
}

void expect(const char* name, const char* what, unsigned got, unsigned expected)
{
    if (got != expected) fail(name, what, got, expected);
}

// D3D12 values written out, not taken from the header's expressions.
enum : unsigned { wrap = 1, mirror = 2, clamp = 3, border = 4, mirror_once = 5 };
enum : unsigned {
    point = 0x0, mip_linear = 0x1, mag_linear = 0x4, min_linear = 0x10, trilinear = 0x15, anisotropic = 0x55
};

// GCN SQ_TEX_CLAMP values 0 to 7: WRAP, MIRROR, CLAMP_LAST_TEXEL, MIRROR_ONCE_LAST_TEXEL,
// CLAMP_HALF_BORDER, MIRROR_ONCE_HALF_BORDER, CLAMP_BORDER, MIRROR_ONCE_BORDER. The desktop has no
// half-border or mirror-once-border mode: the half-border modes keep their edge texel and the last
// becomes BORDER.
constexpr unsigned desktop_mode[8] = {wrap, mirror, clamp, mirror_once, clamp, mirror_once, border, border};

// Word 0: CLAMP_X bits 0-2, CLAMP_Y 3-5, CLAMP_Z 6-8, MAX_ANISO_RATIO 9-11.
// Word 2: XY_MAG_FILTER bits 20-21, XY_MIN_FILTER 22-23, Z_FILTER 24-25, MIP_FILTER 26-27.
std::uint32_t word0(unsigned u, unsigned v, unsigned w, unsigned aniso)
{
    return u | v << 3 | w << 6 | aniso << 9;
}

void check_filter_reduction()
{
    // Encodings are independent of the decoder. Include the reserved mode and telemetry-only fields.
    constexpr unsigned reduction[]{0x000, 0x100, 0x180, 0x000};
    for (unsigned mode = 0; mode < 4; ++mode) {
        for (bool aniso : {false, true}) {
            ++cases;
            const SamplerWords words{(mode << 29) | 0x98189000u, 0xfff000u,
                                     aniso ? 0x0aa04000u : 0x0a504000u, 0};
            const unsigned base = aniso ? anisotropic : trilinear;
            const unsigned compare_base = aniso ? mip_linear : trilinear;
            TranslatedSamplers bindings;
            for (bool comparison : {false, false, true}) {
                bindings.statics.push_back(words);
                bindings.comparison.push_back(comparison);
                bindings.lod_limits.push_back(bindings.statics.size() == 2);
            }
            for (bool supported : {false, true}) {
                const unsigned expected = base | (supported ? reduction[mode] : 0);
                const auto statics = decode_statics(bindings, supported);
                expect("reduction", "heap filter", dynamic_sampler(words, supported).Filter, expected);
                expect("reduction", "smp filter", statics[0].Filter, expected);
                expect("reduction", "smpl filter", statics[1].Filter, expected);
                expect("reduction", "smpc filter", statics[2].Filter, compare_base | 0x80);
            }
            if (mode == 0) {
                const auto heap_default = dynamic_sampler(words);
                const auto heap_enabled = dynamic_sampler(words, true);
                expect("blend", "same heap descriptor", std::memcmp(&heap_default, &heap_enabled, sizeof(heap_default)) == 0, 1);
                const auto statics_default = decode_statics(bindings);
                const auto statics_enabled = decode_statics(bindings, true);
                for (unsigned i = 0; i < statics_default.size(); ++i)
                    expect("blend", "same static descriptor",
                           std::memcmp(&statics_default[i], &statics_enabled[i], sizeof(statics_default[i])) == 0, 1);
            }
        }
    }
}

void check_sampler_census()
{
    ++cases;
    static SamplerCensus census;
    const SamplerWords first{0xb8188000u, 0, 0x01004000u, 0};
    const SamplerWords second{0x40000000u, 0, 0x02080000u, 0};
    expect("census", "log at one", census.observe(first), 1);
    expect("census", "duplicate before two", census.observe(first), 0);
    expect("census", "log at two", census.observe(second), 1);
    expect("census", "quiet at three", census.observe({0, 0, 0, 0}), 0);
    expect("census", "duplicate before four", census.observe(second), 0);
    expect("census", "log at four (distinct word 3)", census.observe({0, 0, 0, 1}), 1);
    const auto& c = census.counts();
    expect("census", "distinct samplers", c.samplers, 4);
    expect("census", "filter_min", c.filter_min, 1);
    expect("census", "filter_max", c.filter_max, 1);
    expect("census", "force_unnorm", c.force_unnorm, 1);
    expect("census", "force_degamma", c.force_degamma, 1);
    expect("census", "skip_degamma", c.skip_degamma, 1);
    expect("census", "trunc_coord", c.trunc_coord, 1);
    expect("census", "mc_trunc", c.mc_trunc, 1);
    expect("census", "no_cube_wrap", c.no_cube_wrap, 1);
    expect("census", "z_filter_point", c.z_filter_point, 1);
    expect("census", "z_filter_linear", c.z_filter_linear, 1);
    expect("census", "lod_bias_sec", c.lod_bias_sec, 2);
    bool logged = false;
    for (unsigned i = 1; i <= 8188; ++i) logged = census.observe({0, i, 0, 0});
    expect("census", "log at capacity", logged, 1);
    expect("census", "quiet past capacity", census.observe({0, 0, 0, 2}), 0);
    expect("census", "frozen total", c.samplers, 8192);
}

void lod_range(const char* name, const char* kind, float bias, float min_lod, float max_lod, float expected_min,
               float expected_max, float expected_bias = 0.0f)
{
    if (bias == expected_bias && min_lod == expected_min && max_lod == expected_max) return;
    std::printf("FAIL %s: %s LOD bias %g, range %g..%g, expected %g, %g..%g\n", name, kind, bias, min_lod, max_lod,
                expected_bias, expected_min, expected_max);
    ++failures;
}

// min_lod and max_lod are the expected limits for table entries and smpl static samplers.
// Ordinary static samplers have an unbounded MaxLOD.
void decoded(const char* name, const SamplerWords& words, unsigned u, unsigned v, unsigned w, unsigned filter,
             unsigned anisotropy, UINT reg = 0, float min_lod = 0.0f, float max_lod = 0.0f)
{
    ++cases;
    const int before = failures;
    if (shared_axis) v = w = u;
    const D3D12_STATIC_SAMPLER_DESC s = decode_sampler(words, reg);
    expect(name, "static Filter", s.Filter, filter);
    expect(name, "static AddressU", s.AddressU, u);
    expect(name, "static AddressV", s.AddressV, v);
    expect(name, "static AddressW", s.AddressW, w);
    expect(name, "static MaxAnisotropy", s.MaxAnisotropy, anisotropy);
    expect(name, "static ComparisonFunc", s.ComparisonFunc, 1);  // NEVER for ordinary filters
    expect(name, "static BorderColor", s.BorderColor, 0);       // TRANSPARENT_BLACK
    expect(name, "static ShaderRegister", s.ShaderRegister, reg);
    expect(name, "static RegisterSpace", s.RegisterSpace, 0);
    expect(name, "static ShaderVisibility", s.ShaderVisibility, 0);  // ALL
    lod_range(name, "static", s.MipLODBias, s.MinLOD, s.MaxLOD, 0.0f, FLT_MAX);
    const D3D12_STATIC_SAMPLER_DESC limited = decode_sampler(words, reg, false, true);
    lod_range(name, "static smpl", limited.MipLODBias, limited.MinLOD, limited.MaxLOD, min_lod, max_lod);

    const D3D12_SAMPLER_DESC d = dynamic_sampler(words);
    expect(name, "dynamic Filter", d.Filter, filter);
    expect(name, "dynamic AddressU", d.AddressU, u);
    expect(name, "dynamic AddressV", d.AddressV, v);
    expect(name, "dynamic AddressW", d.AddressW, w);
    expect(name, "dynamic MaxAnisotropy", d.MaxAnisotropy, anisotropy);
    expect(name, "dynamic ComparisonFunc", d.ComparisonFunc, 1);
    for (float component : d.BorderColor)
        if (component != 0.0f) fail(name, "dynamic BorderColor", 1, 0);
    lod_range(name, "dynamic", d.MipLODBias, d.MinLOD, d.MaxLOD, (std::min)(min_lod, max_lod), max_lod);
    if (failures != before) std::printf("  words %08x %08x %08x %08x\n", words[0], words[1], words[2], words[3]);
}

void manifest(const char* name, std::initializer_list<const char*> lines, bool accepted, bool heap, bool fallback,
              std::size_t statics = 0)
{
    ++cases;
    TranslatedSamplers samplers;
    bool ok = true;
    for (const char* line : lines) ok = samplers.read_manifest_line(line) && ok;
    expect(name, "accepted", ok, accepted);
    expect(name, "heap", samplers.heap, heap);
    expect(name, "reserved_fallback", samplers.reserved_fallback, fallback);
    expect(name, "statics", static_cast<unsigned>(samplers.statics.size()), static_cast<unsigned>(statics));
}

struct Pages {
    unsigned char* base = nullptr;
    std::size_t page = 0;
    Pages(unsigned readable, unsigned blocked)
    {
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        page = system.dwPageSize;
        base = static_cast<unsigned char*>(
            VirtualAlloc(nullptr, (readable + blocked) * page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        DWORD old = 0;
        if (!base || (blocked && !VirtualProtect(base + readable * page, blocked * page, PAGE_NOACCESS, &old))) {
            std::printf("FAIL cannot allocate fixture pages\n");
            std::exit(1);
        }
    }
    ~Pages() { VirtualFree(base, 0, MEM_RELEASE); }
};

// Record i holds {0x1000 * (i + 1) + 0..3}, so a record read from the wrong place or index differs.
SamplerWords record(UINT i)
{
    const std::uint32_t base = 0x1000 * (i + 1);
    return {base, base + 1, base + 2, base + 3};
}

void fill(unsigned char* at, UINT records)
{
    for (UINT i = 0; i < records; ++i) {
        const SamplerWords words = record(i);
        std::memcpy(at + i * 16, words.data(), 16);
    }
}

// readable_end: records [0, readable_end) are readable; the rest of [0, count) are not and are zero.
void snapshot(const char* name, std::uintptr_t address, UINT count, bool reserve, UINT expected_count,
              UINT readable_end)
{
    ++cases;
    const int before = failures;
    const SamplerSnapshot got = snapshot_samplers({address, count}, reserve);
    expect(name, "count", got.count, expected_count);
    expect(name, "reserved_fallback", got.reserved_fallback, reserve);
    expect(name, "readable entries", static_cast<unsigned>(got.readable.count()), readable_end);
    for (UINT i = 0; i < 2048 && failures == before; ++i) {
        const bool readable = i < readable_end;
        expect(name, "readable bit", got.readable.test(i), readable);
        const SamplerWords want = readable ? record(i) : SamplerWords{};
        if (got.words[i] != want) {
            std::printf("FAIL %s: entry %u holds %08x %08x %08x %08x\n", name, i, got.words[i][0], got.words[i][1],
                        got.words[i][2], got.words[i][3]);
            ++failures;
        }
    }
}
}  // namespace

int main(int argc, char** argv)
{
    // --mutate shared-axis: the expectations of a decoder that applies the U field to all three axes
    // (a negative control: the check must then fail on every case with distinct axes).
    shared_axis = argc > 2 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "shared-axis") == 0;
    if (argc > 1 && !shared_axis) {
        std::printf("unknown mutation\n");
        return 2;
    }

    const auto reversed_lod = dynamic_sampler({0, 1024u | (512u << 12), 0, 0});
    expect("reversed lod", "minimum clamped", reversed_lod.MinLOD <= reversed_lod.MaxLOD, true);
    expect("constants", "table entries", translated_sampler_count, 2048);
    expect("constants", "table space", translated_sampler_space, 1);
    expect("constants", "fallback entry", translated_sampler_fallback, 2047);
    expect("constants", "title stride", title_sampler_stride, 16);
    expect("root", "sampler table parameter without views", sampler_root_parameter(0), 1);
    expect("root", "sampler table parameter with views", sampler_root_parameter(1), 2);
    expect("root", "sampler table parameter with four views", sampler_root_parameter(4), 2);

    // Every clamp value on each axis, with the three axes different from one another.
    char name[64];
    for (unsigned k = 0; k < 8; ++k) {
        std::snprintf(name, sizeof(name), "clamp modes %u %u %u", k, (k + 3) % 8, (k + 6) % 8);
        decoded(name, {word0(k, (k + 3) % 8, (k + 6) % 8, 0), 0, 0, 0}, desktop_mode[k], desktop_mode[(k + 3) % 8],
                desktop_mode[(k + 6) % 8], point, 1, k);
    }
    // MAX_ANISO_RATIO n is 2^n samples, at most the desktop's 16.
    const unsigned ratios[8] = {1, 2, 4, 8, 16, 16, 16, 16};
    for (unsigned n = 0; n < 8; ++n) {
        std::snprintf(name, sizeof(name), "aniso ratio %u", n);
        decoded(name, {word0(0, 0, 0, n), 0, 0, 0}, wrap, wrap, wrap, point, ratios[n]);
    }
    // Filters: point and bilinear choices for magnification, minification and mips; any anisotropic
    // XY filter (2 or 3) makes the sampler anisotropic. Mip filter 1 is point.
    decoded("point", {0, 0, 0, 0}, wrap, wrap, wrap, point, 1);
    decoded("linear mag", {0, 0, 1u << 20, 0}, wrap, wrap, wrap, mag_linear, 1);
    decoded("linear min", {0, 0, 1u << 22, 0}, wrap, wrap, wrap, min_linear, 1);
    decoded("point mip", {0, 0, 1u << 26, 0}, wrap, wrap, wrap, point, 1);
    decoded("linear mip", {0, 0, 2u << 26, 0}, wrap, wrap, wrap, mip_linear, 1);
    decoded("trilinear", {0, 0, 1u << 20 | 1u << 22 | 2u << 26, 0}, wrap, wrap, wrap, trilinear, 1);
    decoded("anisotropic mag", {0, 0, 2u << 20, 0}, wrap, wrap, wrap, anisotropic, 1);
    decoded("anisotropic min", {0, 0, 2u << 22, 0}, wrap, wrap, wrap, anisotropic, 1);
    decoded("anisotropic mag 3, ratio 4", {word0(0, 0, 0, 4), 0, 3u << 20, 0}, wrap, wrap, wrap, anisotropic, 16);
    constexpr std::uint32_t point_mip = 1u << 26;
    // Bits the decoder does not read: PERF_MIP and PERF_Z in word 1, word 3 bits 29:0 (and BORDER_COLOR_TYPE 3
    // falls back to transparent black), Z_FILTER and the high bits of words 0 and 2. MIN_LOD and MAX_LOD are
    // both 0xfff, 4095 / 256, for a table entry.
    decoded("ignored bits", {word0(2, 1, 3, 0) | 0xfffff000u, ~0u, 1u << 22 | 3u << 24 | point_mip | 0xf0000000u | 0xfc000u, ~0u},
            clamp, mirror, mirror_once, min_linear, 1, 7, 15.99609375f, 15.99609375f);
    // Word 1: MIN_LOD bits 11:0 and MAX_LOD bits 23:12, unsigned 4.8 fixed point. Each field is read on its
    // own: a whole level, a fraction, the smallest step and the largest value.
    decoded("min lod 2", {0, 0x200u | 0xfffu << 12, point_mip, 0}, wrap, wrap, wrap, point, 1, 0, 2.0f, 15.99609375f);
    decoded("max lod 1", {0, 0x100u << 12, point_mip, 0}, wrap, wrap, wrap, point, 1, 0, 0.0f, 1.0f);
    decoded("lod fractions", {0, 0x140u | 0x1c0u << 12, point_mip, 0}, wrap, wrap, wrap, point, 1, 0, 1.25f, 1.75f);
    // The decoder preserves the two fields; dynamic creation clamps the minimum to the maximum.
    decoded("min lod above max lod", {0, 0x1c0u | 0x140u << 12, point_mip, 0}, wrap, wrap, wrap, point, 1, 0, 1.75f, 1.25f);
    decoded("lod smallest steps", {0, 0x001u | 0x001u << 12, point_mip, 0}, wrap, wrap, wrap, point, 1, 0, 0.00390625f,
            0.00390625f);
    decoded("lod with perf bits", {0, 0xff000000u | 0x080u | 0x300u << 12, point_mip, 0}, wrap, wrap, wrap, point, 1, 0,
            0.5f, 3.0f);

    {
        ++cases;
        // The fixed static sampler keeps its full range even when MIP_FILTER is zero.
        const SamplerWords words{0, 0x100u | 0x300u << 12, 0, 0};
        const auto s = decode_sampler(words, 0);
        lod_range("mip none", "static", s.MipLODBias, s.MinLOD, s.MaxLOD, 0.0f, D3D12_FLOAT32_MAX);
        const auto d = dynamic_sampler(words);
        lod_range("mip none", "dynamic", d.MipLODBias, d.MinLOD, d.MaxLOD, 0.0f, 0.0f);
    }

    const struct {
        const char* name;
        std::uint32_t field;
        float expected;
    } biases[] = {
        {"bias +2", 0x0200u, 2.0f},
        {"bias -1", 0x3f00u, -1.0f},
        {"bias +31 clamped", 0x1f00u, 15.99f},
        {"bias -32 clamped", 0x2000u, -16.0f},
    };
    for (const auto& test : biases) {
        ++cases;
        const auto s = dynamic_sampler({0, 0x300u << 12, point_mip | test.field, 0});
        lod_range(test.name, "dynamic", s.MipLODBias, s.MinLOD, s.MaxLOD, 0.0f, 3.0f, test.expected);
    }
    {
        ++cases;
        TranslatedSamplers samplers;
        if (!samplers.read_manifest_line("smpl 0 0x0 0x300100 0x04003f00 0x0") || samplers.statics.size() != 1) {
            fail("smpl bias", "manifest sampler count", static_cast<unsigned>(samplers.statics.size()), 1);
        } else {
            const auto s = decode_statics(samplers).front();
            lod_range("smpl bias", "static", s.MipLODBias, s.MinLOD, s.MaxLOD, 1.0f, 3.0f, -1.0f);
        }
    }

    // Word 3 bits 31:30 (BORDER_COLOR_TYPE): 0 = TRANS_BLACK, 1 = OPAQUE_BLACK, 2 = OPAQUE_WHITE, and 3 = REGISTER
    // (falls back to TRANS_BLACK). BORDER_COLOR_PTR in bits 11:0 does not alter the fixed border colors.
    {
        const struct {
            const char* label;
            std::uint32_t w3;
            unsigned static_border;
            float rgba[4];
        } border_cases[] = {
            {"border trans black", 0u << 30 | 0xfffu, 0, {0.0f, 0.0f, 0.0f, 0.0f}},
            {"border opaque black", 1u << 30 | 0xfffu, 1, {0.0f, 0.0f, 0.0f, 1.0f}},
            {"border opaque white", 2u << 30 | 0xfffu, 2, {1.0f, 1.0f, 1.0f, 1.0f}},
            {"border register fallback", 3u << 30 | 0xfffu, 0, {0.0f, 0.0f, 0.0f, 0.0f}},
        };
        for (const auto& bc : border_cases) {
            ++cases;
            const SamplerWords words{word0(6, 6, 6, 0), 0, 0, bc.w3};
            const D3D12_STATIC_SAMPLER_DESC s = decode_sampler(words, 2);
            expect(bc.label, "static BorderColor", s.BorderColor, bc.static_border);
            expect(bc.label, "static AddressU", s.AddressU, border);
            expect(bc.label, "static AddressV", s.AddressV, border);
            const D3D12_SAMPLER_DESC d = dynamic_sampler(words);
            for (unsigned ch = 0; ch < 4; ++ch)
                if (d.BorderColor[ch] != bc.rgba[ch]) fail(bc.label, "dynamic BorderColor", unsigned(d.BorderColor[ch]), unsigned(bc.rgba[ch]));
        }
    }

    // The border sampler a missing table entry uses: border on all three axes, point, transparent black.
    {
        ++cases;
        const D3D12_SAMPLER_DESC b = border_sampler();
        expect("border sampler", "Filter", b.Filter, point);
        expect("border sampler", "MaxAnisotropy", b.MaxAnisotropy, 1);
        expect("border sampler", "ComparisonFunc", b.ComparisonFunc, 1);
        expect("border sampler", "AddressU", b.AddressU, border);
        expect("border sampler", "AddressV", b.AddressV, border);
        expect("border sampler", "AddressW", b.AddressW, border);
        lod_range("border sampler", "dynamic", b.MipLODBias, b.MinLOD, b.MaxLOD, 0.0f, FLT_MAX);  // not a title descriptor
        for (float component : b.BorderColor)
            if (component != 0.0f) fail("border sampler", "BorderColor", 1, 0);
    }

    // Manifest lines.
    manifest("table", {"smpheap 2048 1"}, true, true, false);
    manifest("table with fallback", {"smpheap 2048 1 fallback 2047"}, true, true, true);
    manifest("tab separator", {"smpheap\t2048 1 fallback 2047"}, true, true, true);
    manifest("other fallback entry", {"smpheap 2048 1 fallback 2046"}, false, false, false);
    manifest("fallback without entry", {"smpheap 2048 1 fallback"}, false, false, false);
    manifest("other policy", {"smpheap 2048 1 reserve 2047"}, false, false, false);
    manifest("trailing field", {"smpheap 2048 1 fallback 2047 x"}, false, false, false);
    manifest("2049 entries", {"smpheap 2049 1"}, false, false, false);
    manifest("2047 entries", {"smpheap 2047 1"}, false, false, false);
    manifest("space 0", {"smpheap 2048 0"}, false, false, false);
    manifest("space 2", {"smpheap 2048 2 fallback 2047"}, false, false, false);
    manifest("no space", {"smpheap 2048"}, false, false, false);
    manifest("joined keyword", {"smpheap2048 1"}, false, false, false);
    manifest("second table", {"smpheap 2048 1", "smpheap 2048 1 fallback 2047"}, false, true, false);
    manifest("static samplers", {"smp 0 0x2 0x0 0x100000 0x0", "smp 1 0 0 0 0", "srv 0 tex2d table 0 0"}, true,
             false, false, 2);
    manifest("short static sampler ignored", {"smp 0 0x2 0x0 0x100000"}, true, false, false, 0);
    manifest("statics beside the table", {"smp 0 0 0 0 0", "smpheap 2048 1 fallback 2047"}, true, true, true, 1);
    {
        ++cases;
        TranslatedSamplers samplers;
        samplers.read_manifest_line("smp 3 0x2 0xa 0x400000 0xb");
        if (samplers.statics.size() != 1 || samplers.statics[0] != SamplerWords{2, 0xa, 0x400000, 0xb})
            fail("static sampler words", "words", 0, 0);
    }

    // Snapshots. Record i is 0x1000 * (i + 1) + 0..3.
    {
        Pages pages(9, 0);
        fill(pages.base, 2048);
        const auto at = reinterpret_cast<std::uintptr_t>(pages.base);
        snapshot("three records", at, 3, false, 3, 3);
        snapshot("full table", at, 2048, false, 2048, 2048);
        snapshot("full table, reserved entry", at, 2048, true, 2047, 2047);
        snapshot("count above the table", at, 5000, false, 2048, 2048);
        snapshot("count above the table, reserved entry", at, 5000, true, 2047, 2047);
        snapshot("no records", at, 0, true, 0, 0);
        snapshot("null address", 0, 3, true, 0, 0);
        snapshot("table past the top of memory", UINTPTR_MAX - 2048 * 16 + 1, 1, false, 0, 0);
        // A table that ends exactly at the top is read; that kernel address is unreadable.
        snapshot("table ending at the top of memory", UINTPTR_MAX - 2048 * 16, 1, false, 1, 0);
        // Not 16-byte aligned: runs still stop at page ends and read whole records.
        fill(pages.base + 8, 600);
        snapshot("unaligned records across pages", at + 8, 600, false, 600, 600);
    }
    {
        // Two readable records end 8 bytes before an inaccessible page. The third crosses into it
        // and the fourth lies inside it; both are unreadable and zero.
        Pages pages(1, 1);
        unsigned char* const begin = pages.base + pages.page - 40;
        fill(begin, 2);
        std::memcpy(begin + 32, record(2).data(), 8);  // the readable half of record 2
        snapshot("records up to an inaccessible page", reinterpret_cast<std::uintptr_t>(begin), 4, false, 4, 2);
    }
    {
        // A draw reuses its sampler table only when the new snapshot equals the last one, so each
        // field on its own must make two snapshots differ.
        ++cases;
        Pages pages(1, 0);
        fill(pages.base, 4);
        const SamplerSource source{reinterpret_cast<std::uintptr_t>(pages.base), 4};
        const SamplerSnapshot first = snapshot_samplers(source, false);
        if (!(snapshot_samplers(source, false) == first)) fail("equal snapshots", "equal", 0, 1);
        pages.base[3 * 16 + 8] ^= 1;  // one bit of record 3, word 2
        if (snapshot_samplers(source, false) == first) fail("changed sampler word", "equal", 1, 0);
        SamplerSnapshot other = first;
        other.readable.reset(1);
        if (other == first) fail("changed readable bit", "equal", 1, 0);
        other = first;
        other.count = 3;
        if (other == first) fail("changed count", "equal", 1, 0);
        other = first;
        other.reserved_fallback = true;
        if (other == first) fail("changed reserved entry", "equal", 1, 0);
    }

    check_filter_reduction();
    check_sampler_census();
    if (failures) {
        std::printf("startup_translated_sampler_bindings: %d FAILED in %d cases\n", failures, cases);
        return 1;
    }
    std::printf("startup_translated_sampler_bindings: PASS (%d cases)\n", cases);
    return 0;
}
