#include "guest_shader_container.h"
#include "startup_stubs/inflate.h"
#include <cstring>
#include <stdexcept>

// Xbox DXIL/XDAT, XBCH and FSSM record fields follow the native-stage layout
// documented in victory-shader native.rs at 9376fd6. The parser and bounds
// checks below are independently implemented using the repository's inflater.
namespace x2w {
namespace {
using Span = std::string_view;
Span slice(Span data, uint64_t offset, uint64_t size) {
    if (offset > data.size() || size > data.size() - offset)
        throw std::runtime_error("shader record outside its parent");
    return data.substr(static_cast<size_t>(offset), static_cast<size_t>(size));
}
template <class T> T le(Span data, uint64_t offset) {
    T v;
    std::memcpy(&v, slice(data, offset, sizeof v).data(), sizeof v);
    return v;
}
uint32_t u32(Span s, uint64_t off) { return le<uint32_t>(s, off); }
uint64_t u64(Span s, uint64_t off) { return le<uint64_t>(s, off); }
std::vector<ShaderSemantic> read_signature(Span part, bool extended) {
    std::vector<ShaderSemantic> result;
    auto count = u32(part, 0);
    if (count > 128)
        throw std::runtime_error("invalid signature element count");
    const unsigned stride = extended ? 32 : 24;
    auto rows = slice(part, u32(part, 4), size_t(count) * stride);
    for (unsigned i = 0; i < count; ++i) {
        auto row = rows.substr(i * stride, stride);
        if (extended && u32(row, 0) != 0)
            throw std::runtime_error("unsupported signature stream");
        if (extended)
            row.remove_prefix(4);
        auto start = u32(row, 0);
        auto end = part.find(char(0), start);
        if (end == Span::npos || end == start || end - start > 255)
            throw std::runtime_error("invalid shader semantic name");
        auto name = part.substr(start, end - start);
        result.push_back({std::string(name), u32(row, 4), u32(row, 8), u32(row, 12), u32(row, 16), uint8_t(row[20]),
                          uint8_t(row[21])});
    }
    return result;
}
void graphics_io(GuestShaderStage &stage, Span io) {
    const bool pixel = stage.stage == GpuStage::ps;
    auto attributes = [&](unsigned offset, unsigned size, unsigned count_at, bool output) {
        auto count = u32(io, count_at);
        if (count > 32)
            throw std::runtime_error("too many native shader attributes");
        auto records = slice(io, offset, uint64_t(count) * size);
        for (unsigned n = 0; n < count; ++n) {
            auto e = records.substr(n * size, size);
            ShaderAttribute a{};
            a.usage = uint8_t(e[0]);
            a.row = uint8_t(e[1]);
            if (output) {
                a.location = uint8_t(e[2]);
                a.flags = uint8_t(e[5]);
            } else {
                a.mask = uint8_t(e[2]) & 15;
                a.location = uint8_t(e[3]);
                if (pixel) {
                    a.second_row = uint8_t(e[5]);
                    a.flags = uint8_t(e[7]);
                } else a.last_component = uint8_t(e[4]);
            }
            (output ? stage.outputs : stage.attributes).push_back(a);
        }
    };
    if (pixel) {
        attributes(0x24, 8, 0x124, false);
        stage.pixel_input_address = u32(io, 0x134);
        stage.depth_control = u32(io, 0x138);
        unsigned words = u32(io, 0x148);
        if (words > 10)
            throw std::runtime_error("invalid pixel export patch size");
        for (unsigned mrt = 0; mrt < 8; ++mrt) {
            unsigned at = 0x14c + mrt * 404, where = u32(io, at);
            if (where == UINT32_MAX)
                continue;
            slice(stage.code, uint64_t(where) * 4, 4 * words);
            for (const auto &other : stage.exports)
                if (words && uint64_t(where) < other.word_offset + other.full_precision.size() / 4 &&
                    uint64_t(other.word_offset) < uint64_t(where) + words)
                    throw std::runtime_error("overlapping pixel export patches");
            stage.exports.push_back({mrt, where, std::string(slice(io, at + 4 + 360, 4 * words))});
        }
    } else {
        attributes(0x24, 5, 0xc4, false);
        attributes(0xc8, 6, 0x188, true);
        if (stage.stage == GpuStage::ms) {
            stage.mesh_clip_control = u32(io, 0x194);
            stage.threads[0] = le<uint16_t>(io, 0x448);
            stage.max_vertices = le<uint16_t>(io, 0x44a);
            stage.max_primitives = le<uint16_t>(io, 0x44c);
            stage.topology = le<uint8_t>(io, 0x44f);
        }
    }
}
void stages(Span packed, std::vector<GuestShaderStage> &result, size_t &total) {
    if (slice(packed, 0, 4) != "XBCH" || u32(packed, 4) != 3)
        throw std::runtime_error("unsupported native blob version");
    auto payload = slice(packed, 24, u64(packed, 8));
    auto size = u64(packed, 16);
    if (size < 16 || size > 16u << 20 || total + size > 64u << 20)
        throw std::runtime_error("native stage size out of range");
    total += static_cast<size_t>(size);
    std::string inflated(static_cast<size_t>(size), 0);
    size_t produced = 0;
    if (!x2w_inflate::inflate_zlib(reinterpret_cast<const unsigned char *>(payload.data()), payload.size(),
                                   reinterpret_cast<unsigned char *>(inflated.data()), inflated.size(),
                                   &produced) ||
        produced != size)
        throw std::runtime_error("native stage inflation failed");
    if (slice(inflated, 0, 4) != "FSSM" || u32(inflated, 4) != 0x101 || u64(inflated, 8) != size - 16)
        throw std::runtime_error("invalid FSSM header");
    auto body = slice(inflated, 16, size - 16);
    if (u64(body, 0) != 6)
        throw std::runtime_error("unknown FSSM slot layout");
    constexpr unsigned offsets[]{8, 16, 24, 32, 40, 48, 72, 96};
    for (unsigned i = 0; i < 8; ++i) {
        auto off = u64(body, offsets[i]);
        if (off == 0xdeadbeefdeadbeefull)
            continue;
        auto record = slice(body, off, 84);
        GuestShaderStage stage{};
        stage.stage = static_cast<GpuStage>(i);
        stage.wave32 = (u32(record, 20) & 32) != 0;
        stage.vgprs = u32(record, 80) & 1023;
        stage.lds_bytes = u32(record, 76);
        if (stage.vgprs > 256)
            throw std::runtime_error("invalid native register count");
        auto code = slice(body, u64(record, 0), u64(record, 8));
        if (code.size() > 1u << 20)
            throw std::runtime_error("native instruction stream too large");
        stage.code = code;
        if (stage.code.empty() || stage.code.size() % 4)
            throw std::runtime_error("invalid native code size");
        for (unsigned t = 0; t < 2; ++t) {
            auto count = u32(record, 56 + t * 4);
            if (!count)
                continue;
            if (count > 65536)
                throw std::runtime_error("native resource layout too large");
            auto entries = slice(body, u64(record, 40 + t * 8), uint64_t(count) * 20);
            for (unsigned j = 0; j < count; ++j) {
                auto e = entries.substr(20ull * j, 20);
                if (t == 0)
                    stage.inputs.push_back(
                        {u32(e, 0), u32(e, 4), u32(e, 8), u32(e, 12), uint8_t(e[16]), uint8_t(e[17])});
                else
                    stage.resources.push_back({uint8_t(e[0]), uint8_t(e[1]), u32(e, 4), u32(e, 8), u32(e, 12),
                                               uint8_t(e[16]), uint8_t(e[17]), uint8_t(e[18])});
            }
        }
        if (stage.stage == GpuStage::cs) {
            auto io = slice(body, off + 84 + 36, 16);
            stage.rsrc2 = u32(io, 0);
            for (unsigned j = 0; j < 3; ++j)
                stage.threads[j] = u32(io, 4 + j * 4);
        }
        if (stage.stage == GpuStage::vs || stage.stage == GpuStage::ms || stage.stage == GpuStage::ps)
            graphics_io(stage, body.substr(static_cast<size_t>(off) + 84));
        result.push_back(std::move(stage));
    }
}
} // namespace
std::string GuestShaderStage::full_precision_code() const {
    auto patched = code;
    for (const auto &p : exports)
        patched.replace(size_t(p.word_offset) * 4, p.full_precision.size(), p.full_precision);
    return patched;
}
std::vector<GuestShaderStage> read_guest_shader_container(std::string_view bytes) {
    if (slice(bytes, 0, 4) != "DXBC" || u32(bytes, 24) != bytes.size())
        throw std::runtime_error("invalid DXBC length");
    std::vector<GuestShaderStage> result;
    size_t total = 0;
    std::vector<ShaderSemantic> inputs, outputs;
    const auto count = u32(bytes, 28);
    if (count > 4096)
        throw std::runtime_error("invalid DXBC part count");
    auto offsets = slice(bytes, 32, 4ull * count);
    for (unsigned i = 0; i < count; ++i) {
        uint64_t part = u32(offsets, 4ull * i);
        auto data = slice(bytes, part + 8, u32(bytes, part + 4));
        auto kind = slice(bytes, part, 4);
        if (kind == "ISG1" || kind == "ISGN")
            inputs = read_signature(data, kind == "ISG1");
        if (kind == "OSG1" || kind == "OSGN")
            outputs = read_signature(data, kind == "OSG1");
        if (kind != "DXIL" || data.size() < 36 || data.substr(24, 4) != "XDAT")
            continue;
        auto n = u32(data, 28);
        if (n > 64)
            throw std::runtime_error("invalid XDAT entry count");
        auto entries = slice(data, 36, 12ull * n);
        for (unsigned j = 0; j < n; ++j) {
            auto row = entries.substr(12ull * j, 12);
            auto chunk = slice(data, u32(row, 4), u32(row, 8));
            if (u32(row, 0) == 13)
                stages(chunk, result, total);
        }
    }
    for (auto &stage : result)
        if (stage.stage == GpuStage::vs || stage.stage == GpuStage::ps || stage.stage == GpuStage::ms) {
            stage.input_signature = inputs;
            stage.output_signature = outputs;
        }
    return result;
}
} // namespace x2w
