#include "shader_compute_request.h"
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace {
using Bytes = std::string;
template <class T> T read(const Bytes &data, size_t offset) {
    if (offset > data.size() || sizeof(T) > data.size() - offset)
        throw std::runtime_error("truncated compute request");
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}
Bytes part(fs::path path, const char *suffix, size_t limit) {
    path.replace_extension(suffix);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    auto size = file.tellg();
    if (size < 0 || size > static_cast<std::streamoff>(limit))
        throw std::runtime_error(std::string("missing or oversized ") + suffix);
    Bytes bytes(static_cast<size_t>(size), 0);
    file.seekg(0);
    if (size && !file.read(bytes.data(), size))
        throw std::runtime_error("short read");
    return bytes;
}
void hash_bytes(uint64_t &key, const void *data, size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i)
        key = (key ^ bytes[i]) * 1099511628211ull;
}
bool lower_hex(std::string_view text) {
    if (text.empty()) return false;
    for (char c : text)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    return true;
}
uint64_t parse_request_stem(std::string_view stem, const char *error_message) {
    const std::string_view digits = stem.substr(4);
    uint64_t expected = 0;
    if (!lower_hex(digits))
        throw std::runtime_error(error_message);
    const auto end = digits.data() + digits.size();
    const auto parsed = std::from_chars(digits.data(), end, expected, 16);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::runtime_error(error_message);
    return expected;
}
} // namespace

std::string x2w::format_pixel_input_description(uint64_t key, uint32_t ena, uint32_t addr) {
    char text[64];
    std::snprintf(text, sizeof(text), "psin1 %016llx ena=0x%04x addr=0x%04x\n",
                  static_cast<unsigned long long>(key), ena, addr);
    return text;
}

x2w::PixelInputDescription x2w::parse_pixel_input_description(std::string_view raw, uint64_t expected) {
    std::string_view line = raw;
    if (!line.empty() && line.back() == '\n')
        line.remove_suffix(1);
    if (line.size() < 22 || line.substr(0, 6) != "psin1 " || line.find_first_of("\r\n") != std::string_view::npos)
        throw std::runtime_error("invalid pixel input description");
    const std::string_view key_hex = line.substr(6, 16);
    if (!lower_hex(key_hex))
        throw std::runtime_error("invalid pixel input description");
    uint64_t described_key = 0;
    const auto key_end = key_hex.data() + key_hex.size();
    if (std::from_chars(key_hex.data(), key_end, described_key, 16).ptr != key_end)
        throw std::runtime_error("invalid pixel input description");
    bool has_ena = false, has_addr = false;
    uint32_t ena = 0, addr = 0;
    unsigned fields = 0;
    size_t pos = 22;
    while (pos < line.size()) {
        if (line[pos] != ' ' || ++fields > 2)
            throw std::runtime_error("invalid pixel input description");
        ++pos;
        const size_t next = line.find(' ', pos);
        const std::string_view token = line.substr(pos, next == std::string_view::npos ? line.size() - pos : next - pos);
        pos = next == std::string_view::npos ? line.size() : next;
        const bool is_ena = token.substr(0, 6) == "ena=0x";
        const bool is_addr = token.substr(0, 7) == "addr=0x";
        if (!is_ena && !is_addr)
            throw std::runtime_error("invalid pixel input description");
        const std::string_view mask_hex = token.substr(is_ena ? 6 : 7);
        if (mask_hex.empty() || mask_hex.size() > 4 || !lower_hex(mask_hex) || (is_ena ? has_ena : has_addr))
            throw std::runtime_error("invalid pixel input description");
        uint32_t value = 0;
        const auto mask_end = mask_hex.data() + mask_hex.size();
        if (std::from_chars(mask_hex.data(), mask_end, value, 16).ptr != mask_end)
            throw std::runtime_error("invalid pixel input description");
        if (is_ena) {
            has_ena = true;
            ena = value;
        } else {
            has_addr = true;
            addr = value;
        }
    }
    if (described_key != expected)
        throw std::runtime_error("pixel input description names another pipeline");
    if (!has_ena)
        throw std::runtime_error("pixel input enable mask missing from the stream description");
    if (!has_addr)
        throw std::runtime_error("pixel input address mask missing from the stream description");
    if (ena & ~addr)
        throw std::runtime_error("contradictory pixel input enable/address masks");
    PixelInputDescription parsed;
    parsed.ena = ena;
    parsed.addr = addr;
    parsed.entry.assign("psin1", 5);
    const uint32_t words[2]{ena, addr};
    parsed.entry.append(reinterpret_cast<const char *>(words), sizeof(words));
    parsed.canonical_text = format_pixel_input_description(expected, ena, addr);
    return parsed;
}

x2w::RtvVariantDescription x2w::parse_rtv_variant_text(std::string_view text, uint64_t expected) {
    if (text.size() > 256)
        throw std::runtime_error("invalid RTV variant request");
    const uint64_t key = fnv1a_64(text);
    std::vector<std::string_view> tokens;
    std::string_view rest = text;
    bool valid = rest.substr(0, 5) == "rtv1 ";
    if (valid) rest.remove_prefix(5);
    while (valid) {
        const size_t end = rest.find(' ');
        const auto token = rest.substr(0, end);
        valid = !token.empty() && token.size() <= (tokens.empty() ? 20u : 10u) &&
                token.find_first_not_of("0123456789") == token.npos;
        tokens.push_back(token);
        if (end == rest.npos) break;
        rest.remove_prefix(end + 1);
    }
    if (!valid || tokens.size() < 2 || tokens.size() > 9 || key != expected)
        throw std::runtime_error("invalid RTV variant request");
    RtvVariantDescription result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        uint64_t value = 0;
        const auto parsed = std::from_chars(tokens[i].data(), tokens[i].data() + tokens[i].size(), value);
        if (parsed.ec != std::errc{} || (i && value > 0xffffffffull))
            throw std::runtime_error("invalid RTV variant values");
        if (!i) result.base_key = value;
        else result.formats.push_back(static_cast<unsigned>(value));
    }
    return result;
}

std::string x2w::format_rtv_variant_text(uint64_t base_key, const std::vector<unsigned> &formats) {
    std::string out = "rtv1 " + std::to_string(base_key);
    for (unsigned format : formats) {
        out.push_back(' ');
        out += std::to_string(format);
    }
    return out;
}

uint64_t x2w::fnv1a_64(std::string_view bytes) {
    uint64_t key = 1469598103934665603ull;
    hash_bytes(key, bytes.data(), bytes.size());
    return key;
}

uint64_t x2w::graphics_request_key(std::string_view a, std::string_view b, std::string_view root,
                                   std::string_view input_layout, std::string_view pixel_entry,
                                   std::string_view revision_override) {
    constexpr char revision[] = "x2w-startup-shader-29";
    uint64_t key = fnv1a_64(revision_override.empty() ? std::string_view(revision, sizeof(revision) - 1)
                                                      : revision_override);
    for (const auto part_view : {a, b, root, input_layout, pixel_entry})
        hash_bytes(key, part_view.data(), part_view.size());
    return key;
}

uint64_t x2w::compute_request_key(std::string_view code, std::string_view data, std::string_view root,
                                  std::string_view description, std::string_view sa, std::string_view sb,
                                  std::string_view revision_override) {
    constexpr char revision[] = "x2w-startup-shader-29cs";
    uint64_t key = fnv1a_64(revision_override.empty() ? std::string_view(revision, sizeof(revision) - 1)
                                                      : revision_override);
    for (const auto part_view : {code, data, root, description, sa, sb}) {
        const uint64_t size = part_view.size();
        hash_bytes(key, &size, sizeof(size));
        hash_bytes(key, part_view.data(), part_view.size());
    }
    return key;
}

uint64_t x2w::mesh57_request_key(std::string_view code, std::string_view ps57, std::string_view root) {
    constexpr char revision[] = "x2w-startup-mesh-57-3";
    uint64_t key = fnv1a_64(std::string_view(revision, sizeof(revision) - 1));
    for (const auto part_view : {code, ps57, root}) {
        const uint64_t size = part_view.size();
        hash_bytes(key, &size, sizeof(size));
        hash_bytes(key, part_view.data(), part_view.size());
    }
    return key;
}

namespace {
void parse_pixel_inputs(const fs::path &marker, uint64_t expected, Bytes &entry, Bytes &description) {
    fs::path path = marker;
    path.replace_extension(".psin");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto size = file.tellg();
    if (size < 0 || size > 128)
        throw std::runtime_error("invalid pixel input description");
    Bytes raw(static_cast<size_t>(size), 0);
    file.seekg(0);
    if (size && !file.read(raw.data(), size))
        throw std::runtime_error("short read");
    auto parsed = x2w::parse_pixel_input_description(raw, expected);
    entry = std::move(parsed.entry);
    description = std::move(parsed.canonical_text);
}
// tools/startup/gfx2hlsl.py request_source: at most 256 bytes of "rtv1 <base key> <format> ..." in decimal with one to
// eight formats and nothing else, whose FNV-1a hash is the key in the file name. The variant is translated from its
// base request's parts, which are read and checked against the base key like any base request.
x2w::GraphicsRequest read_rtv_variant(const fs::path &marker) {
    const auto stem = marker.stem().string();
    if (stem.size() != 20 || stem.compare(0, 4, "req_"))
        throw std::runtime_error("expected req_<16 hex digits>.rtv");
    const uint64_t expected = parse_request_stem(stem, "invalid RTV variant request key");
    Bytes text;
    try {
        text = part(marker, ".rtv", 256);
    } catch (const std::runtime_error &) {
        throw std::runtime_error("invalid RTV variant request");
    }
    auto parsed = x2w::parse_rtv_variant_text(text, expected);
    char name[32];
    std::snprintf(name, sizeof(name), "req_%016llx.a", static_cast<unsigned long long>(parsed.base_key));
    x2w::GraphicsRequest request;
    try {
        request = x2w::read_graphics_request(marker.parent_path() / name);
    } catch (const std::runtime_error &error) {
        throw std::runtime_error(std::string("RTV variant source: ") + error.what());
    }
    request.key = expected;
    request.rtv_request = std::move(text);
    request.rtv_formats = std::move(parsed.formats);
    return request;
}
} // namespace

x2w::GraphicsRequest x2w::read_graphics_request(const fs::path &marker) {
    if (marker.extension() == ".rtv") return read_rtv_variant(marker);
    auto stem = marker.stem().string();
    if (marker.extension() != ".a" || stem.size() != 20 || stem.compare(0, 4, "req_"))
        throw std::runtime_error("expected req_<16 hex digits>.a");
    const uint64_t expected = parse_request_stem(stem, "invalid graphics request key");
    GraphicsRequest request{part(marker, ".a", 64 << 20), part(marker, ".b", 64 << 20),
                            part(marker, ".root", 16 << 20), {}, {}, {}, expected};
    if (fs::exists(fs::path(marker).replace_extension(".ia")))
        request.input_layout = part(marker, ".ia", 1 << 20);
    if (fs::exists(fs::path(marker).replace_extension(".psin")))
        parse_pixel_inputs(marker, expected, request.pixel_entry, request.pixel_input_description);
    // The runtime hashes a feature tag after the revision while extended buffer formats are on (the default): accept both.
    constexpr char feature[] = "extended-buffer-formats1";
    const auto request_key = [&](std::string_view revision, bool tagged = false) {
        uint64_t key = 1469598103934665603ull;
        hash_bytes(key, revision.data(), revision.size());
        if (tagged) hash_bytes(key, feature, sizeof(feature) - 1);
        for (const auto *p : {&request.a, &request.b, &request.root, &request.input_layout, &request.pixel_entry})
            hash_bytes(key, p->data(), p->size());
        return key;
    };
    if (request.root.empty())
        throw std::runtime_error("graphics request key mismatch or missing root");
    // Match new runtime captures and keep revision-28 archives replayable.
    constexpr char revision[] = "x2w-startup-shader-29";
    if (request_key(revision) != expected && request_key(revision, true) != expected &&
        request_key("x2w-startup-shader-28") != expected && request_key("x2w-startup-shader-28", true) != expected) {
        if (!request.pixel_entry.empty())
            throw std::runtime_error("stale pixel input description: graphics request key mismatch");
        throw std::runtime_error("graphics request key mismatch or missing root");
    }
    return request;
}

x2w::ComputeRequest x2w::read_compute_request(const fs::path &marker) {
    auto name = marker.stem().string();
    if (marker.extension() != ".c67" || name.size() != 20 || name.compare(0, 4, "req_"))
        throw std::runtime_error("expected req_<16 hex digits>.c67");
    const uint64_t expected = parse_request_stem(name, "invalid request key");
    auto code = part(marker, ".c67", 1 << 20), data = part(marker, ".data", 1 << 20),
         root = part(marker, ".root", 16 << 20);
    auto desc = part(marker, ".desc", 96), sa = part(marker, ".sa", 4096), sb = part(marker, ".sb", 4096);
    if (desc.size() != 96 || read<uint64_t>(desc, 48) != code.size() || read<uint64_t>(desc, 64) != data.size() ||
        read<uint64_t>(desc, 16) != sa.size() || read<uint64_t>(desc, 32) != sb.size())
        throw std::runtime_error("inconsistent compute request");
    constexpr char feature[] = "extended-buffer-formats1";  // hashed after the revision while extended buffer formats are on
    const auto request_key = [&](std::string_view revision, bool tagged = false) {
        uint64_t key = 1469598103934665603ull;
        hash_bytes(key, revision.data(), revision.size());
        if (tagged) hash_bytes(key, feature, sizeof(feature) - 1);
        for (const auto *p : {&code, &data, &root, &desc, &sa, &sb}) {
            uint64_t size = p->size();
            hash_bytes(key, &size, sizeof(size));
            hash_bytes(key, p->data(), p->size());
        }
        return key;
    };
    constexpr char revision[] = "x2w-startup-shader-29cs";
    if (request_key(revision) != expected && request_key(revision, true) != expected &&
        request_key("x2w-startup-shader-28cs") != expected && request_key("x2w-startup-shader-28cs", true) != expected)
        throw std::runtime_error("compute request key mismatch");
    if (root.empty() || sa.size() != 28 || code.empty() || data.empty() || code.size() % 4)
        throw std::runtime_error(sa.size() != 28 ? "compute state block A is not 28 bytes"
                                                 : "inconsistent compute request");
    // Bit 20 of dword 4 (bits 15:0 are Z): set for wave32, clear for wave64. Inferred from captured requests
    // (docs/experiments/native-slot67-wave-size.md).
    const bool wave32 = (static_cast<unsigned char>(sa[18]) & 0x10) != 0;
    return {std::move(code), std::move(data), std::move(root), std::move(desc), std::move(sa), std::move(sb), expected,
            wave32};
}

std::string x2w::read_shader_container_bytes(const fs::path &file) {
    std::ifstream stream(file, std::ios::binary | std::ios::ate);
    auto length = stream.tellg();
    if (length < 32 || length > 64 * 1024 * 1024)
        throw std::runtime_error("missing or oversized shader container");
    std::string bytes(static_cast<size_t>(length), 0);
    stream.seekg(0);
    if (!stream.read(bytes.data(), length)) throw std::runtime_error("truncated shader container");
    return bytes;
}
x2w::ComputeContainer x2w::read_compute_container(const fs::path &file) {
    auto bytes=read_shader_container_bytes(file);
    const auto stem=file.stem().string();
    if (stem.compare(0,4,"req_") != 0) return {std::move(bytes), {}};
    uint64_t expected=0;
    if (stem.size()!=20 || file.extension()!=".cdx")
        throw std::runtime_error("invalid container request filename");
    auto parsed=std::from_chars(stem.data()+4,stem.data()+stem.size(),expected,16);
    if(parsed.ec!=std::errc{} || parsed.ptr!=stem.data()+stem.size())
        throw std::runtime_error("invalid container request key");
    auto root=part(file,".root",16<<20);
    uint64_t key=1469598103934665603ull, size=root.size();
    constexpr char revision[]="native-container-1";
    hash_bytes(key,revision,sizeof(revision)-1);
    hash_bytes(key,&size,sizeof(size));
    hash_bytes(key,root.data(),root.size());
    hash_bytes(key,bytes.data(),bytes.size());
    if (root.empty() || key!=expected) throw std::runtime_error("container request key mismatch");
    return {std::move(bytes),std::move(root)};
}
