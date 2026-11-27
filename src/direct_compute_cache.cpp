#include "direct_compute_cache.h"
#include "guest_shader_container.h"
#include "gfx1013_isa.h"
#include "native_graphics_bindings.h"
#include "shader_compute_request.h"
#include "native_shader_revision.h"
#include "shader_dxil_compiler.h"
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <charconv>
#include <sstream>
#include <set>
#include <map>
#include <memory>
#include <mutex>
#include <d3d12.h>
#include <wrl/client.h>

namespace x2w {
namespace {
namespace fs = std::filesystem;
using Bytes = std::string;
void word(Bytes &data, uint32_t value) { data.append(reinterpret_cast<const char *>(&value), sizeof(value)); }
bool embedded_graphics_id(std::string_view raw) {
    // Alternative compiler variants can share the embedded ID encoded in two
    // literal-bearing V_NOPs. A common executable prologue is not an ID.
    if (raw.size() < 24) return false;
    for (unsigned pc : {0u,12u}) {
        if (static_cast<uint8_t>(raw[pc+2]) != 0x80 || static_cast<uint8_t>(raw[pc+3]) != 0xd5) return false;
        const auto op = gfx1013::instruction(raw,pc);
        if (op.form != gfx1013::Form::vop3 || op.name != "v_nop" || op.size != 12) return false;
    }
    return true;
}
GuestShaderStage match_graphics_stage(std::string_view raw, const Bytes &blob, bool &exact,
                                      std::optional<GpuStage> required = {}) {
    auto stages = read_guest_shader_container(blob);
    int best = 0;
    size_t selected = 0;
    bool tie = false;
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto &stage = stages[i];
        if (required ? stage.stage != *required : (stage.stage != GpuStage::vs && stage.stage != GpuStage::ps))
            continue;
        // A pixel container can carry materialized export instructions in a
        // patch table. The running PS already contains the selected export.
        int rank = 0;
        if (raw == stage.code) rank = stage.exports.empty() ? 3 : 2;
        if (!stage.exports.empty() && raw == stage.full_precision_code()) rank = 3;
        if (!rank && raw.substr(0,24) == std::string_view(stage.code).substr(0,24) && embedded_graphics_id(raw) &&
            (!required || (stage.container_identifier && native_shader_identifier(raw) == stage.container_identifier)))
            rank = 1;
        if (rank > best) { best = rank; selected = i; tie = false; }
        else if (rank && rank == best) tie = true;
    }
    if (!best) throw UnsupportedComputeShader("raw stage and interface identity disagree");
    if (tie) throw UnsupportedComputeShader("ambiguous native stage in graphics interface");
    exact = best == 3;
    return std::move(stages[selected]);
}
struct Reader {
    std::string_view remaining;
    std::string_view take(size_t size) {
        if (size > remaining.size())
            throw std::runtime_error("short frontend cache");
        auto bytes = remaining.substr(0, size);
        remaining.remove_prefix(size);
        return bytes;
    }
    uint32_t word() {
        uint32_t n;
        std::memcpy(&n, take(sizeof n).data(), sizeof n);
        return n;
    }
    uint32_t count(uint32_t max) {
        auto n = word();
        if (n > max)
            throw std::runtime_error("invalid frontend cache count");
        return n;
    }
};

Bytes encode(const NativeComputeShader &program) {
    Bytes out;
    word(out, program.instruction_count);
    for (unsigned n : program.group)
        word(out, n);
    word(out, static_cast<uint32_t>(program.bindings.size()));
    for (const auto &b : program.bindings) {
        word(out, static_cast<uint32_t>(b.kind));
        for (unsigned n : {b.root_parameter, b.byte_offset, b.shader_register, b.required_bytes, b.array_count,
                           b.array_position, b.image_dimension, unsigned(b.scalar_table)})
            word(out, n);
    }
    word(out, static_cast<uint32_t>(program.constants.size()));
    for (const auto &c : program.constants)
        for (unsigned n : {c.root_parameter, c.first_sgpr, c.count})
            word(out, n);
    word(out, program.sampler_heap);
    word(out, static_cast<uint32_t>(program.samplers.size()));
    for (const auto &s : program.samplers) {
        for (auto n : s.words)
            word(out, n);
        word(out, s.explicit_lod);
        word(out, s.comparison);
    }
    for (const auto *signature : {&program.inputs, &program.outputs}) {
        word(out, static_cast<unsigned>(signature->size()));
        for (const auto &row : *signature) {
            word(out, static_cast<unsigned>(row.name.size()));
            out += row.name;
            for (auto field :
                 {row.index, row.row, row.first, row.width, row.component, row.used, row.kind, row.interpolation})
                word(out, field);
        }
    }
    word(out, program.vertex_fetch);
    for (auto mask : program.pixel_writes)
        word(out, mask);
    out += program.dxil_ir;
    return out;
}
NativeComputeShader decode(std::string_view bytes) {
    Reader in{bytes};
    NativeComputeShader shader{};
    shader.instruction_count = in.count(1 << 18);
    shader.group = {in.count(1024), in.count(1024), in.count(64)};
    unsigned n = in.count(1000000);
    shader.bindings.reserve(n);
    for (unsigned j = 0; j < n; ++j) {
        auto kind = static_cast<ComputeViewKind>(in.count(static_cast<unsigned>(ComputeViewKind::indexed_raw_srv)));
        ComputeViewBinding b{kind, in.word(), in.word(), in.word(), in.word(), in.word(), in.word(), in.count(7)};
        b.scalar_table = static_cast<ComputeViewBinding::ScalarTable>(in.count(unsigned(ComputeViewBinding::ScalarTable::srv)));
        const bool sampler_heap = kind == ComputeViewKind::heap_sampler;
        if (!b.array_count || b.array_position >= b.array_count ||
            (sampler_heap ? b.root_parameter != compute_sampler_heap
                          : b.root_parameter >= 16 && b.root_parameter != compute_resource_heap))
            throw std::runtime_error("invalid cached binding");
        shader.bindings.push_back(b);
    }
    n = in.count(32);
    for (unsigned j = 0; j < n; ++j) {
        ComputeConstants c{in.count(15), in.count(105), in.count(32)};
        if (!c.count || c.count > 106 - c.first_sgpr)
            throw std::runtime_error("invalid cached constants");
        shader.constants.push_back(c);
    }
    shader.sampler_heap = in.count(1) != 0;
    n = in.count(2048);
    for (unsigned j = 0; j < n; ++j) {
        ComputeSampler sampler{};
        for (auto &w : sampler.words)
            w = in.word();
        sampler.explicit_lod = in.count(1) != 0;
        sampler.comparison = in.count(1) != 0;
        shader.samplers.push_back(sampler);
    }
    for (auto *signature : {&shader.inputs, &shader.outputs}) {
        unsigned count = in.count(128);
        for (unsigned j = 0; j < count; ++j) {
            GraphicsLinkage row{};
            row.name = in.take(in.count(4096));
            row.index = in.word();
            row.row = in.word();
            row.first = in.count(3);
            row.width = in.count(4);
            row.component = in.word();
            row.used = in.count(15);
            row.kind = in.count(31);
            row.interpolation = in.count(7);
            const bool unpacked = row.row == UINT_MAX && (signature == &shader.outputs ||
                (row.name == "SV_Barycentrics" && row.kind == 28 && row.component == 9 &&
                 row.width == 3 && !row.first && row.index < 2 && row.interpolation >= 2));
            if (row.row >= 32 && !unpacked)
                throw std::runtime_error("invalid cached signature register");
            if (row.name.empty() || !row.width || row.first + row.width > 4 || row.used >= (1u << row.width))
                throw std::runtime_error("invalid cached signature");
            signature->push_back(std::move(row));
        }
    }
    shader.vertex_fetch = in.count(1) != 0;
    for (auto &mask : shader.pixel_writes)
        mask = in.count(15);
    if (in.remaining.empty())
        throw std::runtime_error("missing cached DXIL IR");
    shader.dxil_ir = in.remaining;
    return shader;
}
bool lookup(const fs::path &path, const Bytes &key, Bytes &payload) {
    HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
        return false;
    LARGE_INTEGER size{};
    bool ok = false;
    if (GetFileSizeEx(handle, &size) && size.QuadPart > 128 && size.QuadPart <= 256ll * 1024 * 1024) {
        Bytes id(64, 0), checksum(64, 0);
        payload.resize(static_cast<size_t>(size.QuadPart) - 128);
        DWORD got_id = 0, got_sum = 0, got_body = 0;
        ok = ReadFile(handle, id.data(), 64, &got_id, nullptr) && got_id == 64 && id == key &&
             ReadFile(handle, checksum.data(), 64, &got_sum, nullptr) && got_sum == 64 &&
             ReadFile(handle, payload.data(), static_cast<DWORD>(payload.size()), &got_body, nullptr) &&
             got_body == payload.size() && shader_digest(payload) == checksum;
    }
    CloseHandle(handle);
    return ok;
}
void publish(const fs::path &path, const Bytes &key, const Bytes &payload) {
    fs::create_directories(path.parent_path());
    static std::atomic<unsigned> serial{};
    auto temp = path;
    temp += L"." + std::to_wstring(GetCurrentProcessId()) + L"." + std::to_wstring(serial++) + L".tmp";
    try {
        std::ofstream file(temp, std::ios::binary);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << key << shader_digest(payload);
        file.write(payload.data(), payload.size());
        file.close();
        if (!MoveFileExW(temp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
            Bytes winner;
            if (!lookup(path, key, winner) || winner != payload)
                throw std::runtime_error("could not publish direct frontend cache");
            std::error_code ignored;
            fs::remove(temp, ignored);
        }
    } catch (...) {
        std::error_code ignored;
        fs::remove(temp, ignored);
        throw;
    }
}
using InterfaceMap = std::map<Bytes, Bytes>;
const Bytes &catalog_version() {
    static const Bytes version = shader_digest("x2w graphics interface catalog 1");
    return version;
}
bool digest_text(std::string_view s) {
    return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == s.npos;
}
std::shared_ptr<const InterfaceMap> load_interfaces(const fs::path &dir) {
    auto marker = dir / "catalog";
    if (!fs::exists(marker)) return {};
    Bytes generation;
    if (!lookup(marker, catalog_version(), generation) || !digest_text(generation))
        throw std::runtime_error("invalid graphics interface catalog marker");
    static std::mutex lock;
    static Bytes resident_id;
    static std::shared_ptr<const InterfaceMap> resident;
    std::lock_guard guard(lock);
    if (resident && generation == resident_id) return resident;
    Bytes index;
    if (!lookup(dir / (generation + ".index"), generation, index) || shader_digest(index) != generation)
        throw std::runtime_error("invalid graphics interface catalog index");
    auto entries = std::make_shared<InterfaceMap>();
    Reader reader{index};
    while (!reader.remaining.empty()) {
        auto key = reader.take(65), value = reader.take(64);
        if (std::string_view("FIMN").find(key.front()) == std::string_view::npos || !digest_text(key.substr(1)) ||
            (!digest_text(value) && value != Bytes(64,'-')) || !entries->emplace(key,value).second)
            throw std::runtime_error("invalid graphics interface catalog entry");
    }
    resident_id = generation;
    resident = entries;
    return resident;
}
Bytes indexed_interface(const fs::path &dir, const InterfaceMap &entries, std::string_view raw, const Bytes &hash, bool mesh = false) {
    auto it = entries.find((mesh ? "M" : "F") + hash);
    if (it == entries.end() && embedded_graphics_id(raw))
        it = entries.find((mesh ? "N" : "I") + shader_digest(raw.substr(0,24)));
    if (it == entries.end()) return {};
    if (it->second.front() == '-') throw UnsupportedComputeShader("ambiguous reviewed graphics interfaces");
    Bytes container;
    if (!lookup(dir / (it->second + ".stage"), it->second, container) || shader_digest(container) != it->second)
        throw std::runtime_error("invalid indexed graphics interface container");
    return container;
}
} // namespace

GuestShaderStage reviewed_graphics_stage(std::string_view raw, GpuStage kind, const fs::path &cache, bool &exact) {
    const auto dir = cache / "interfaces" / "library";
    const auto entries = load_interfaces(dir);
    if (!entries) throw UnsupportedComputeShader("missing reviewed graphics interfaces");
    const auto blob = indexed_interface(dir, *entries, raw, shader_digest(raw), kind == GpuStage::ms);
    if (blob.empty()) throw UnsupportedComputeShader("stage absent from reviewed graphics interfaces");
    return match_graphics_stage(raw, blob, exact, kind);
}

size_t index_graphics_interfaces(const fs::path &source, const fs::path &cache) {
    const auto dir = cache / "interfaces" / "library";
    InterfaceMap index;
    size_t count = 0;
    for (const auto &entry : fs::recursive_directory_iterator(source)) {
        if (!entry.is_regular_file() || entry.path().extension() != L".cdxil") continue;
        auto blob = read_shader_container_bytes(entry.path());
        auto stages = read_guest_shader_container(blob);
        const auto id = shader_digest(blob);
        auto add = [&](const Bytes &key) {
            auto [it, fresh] = index.emplace(key,id);
            if (!fresh && it->second != id) it->second.assign(64,'-');
        };
        bool graphics = false;
        for (const auto &stage : stages) {
            const bool mesh = stage.stage == GpuStage::ms;
            if (!mesh && stage.stage != GpuStage::vs && stage.stage != GpuStage::ps) continue;
            graphics = true;
            add((mesh ? "M" : "F") + shader_digest(stage.code));
            if (!stage.exports.empty()) add("F" + shader_digest(stage.full_precision_code()));
            if (embedded_graphics_id(stage.code))
                add((mesh ? "N" : "I") + shader_digest(std::string_view(stage.code).substr(0,24)));
        }
        if (!graphics) continue;
        publish(dir / (id + ".stage"), id, blob);
        ++count;
    }
    if (!count) throw std::runtime_error("no graphics .cdxil containers in interface directory");
    Bytes encoded;
    for (const auto &[key,id] : index) { encoded += key; encoded += id; }
    const auto generation = shader_digest(encoded);
    publish(dir / (generation + ".index"), generation, encoded);
    publish(dir / "catalog", catalog_version(), generation);
    return count;
}
namespace {
Bytes slot67_digest(const ComputeRequest &request, unsigned entry) {
    Bytes identity = X2W_NATIVE_SHADER_REVISION "/slot67/cs_6_6";
    word(identity, entry);
    for (const auto *part : {&request.code, &request.data, &request.root, &request.description, &request.sa, &request.sb})
        identity += shader_digest(*part);
    return shader_digest(identity);
}
template<class Translate>
NativeComputeShader cached_compute_ir(const fs::path &cache, const Bytes &key, bool &hit, Translate translate) {
    const auto path = cache / (key + ".csir");
    Bytes payload;
    hit = lookup(path, key, payload);
    if (hit) {
        try { return decode(payload); }
        catch (const std::runtime_error &) { hit = false; }
    }
    auto program = translate();
    publish(path, key, encode(program));
    return program;
}
Bytes pack_version(unsigned entry) {
    Bytes version = X2W_NATIVE_SHADER_REVISION "/slot67/cs_6_6/prepared-1";
    word(version, entry);
    return shader_digest(version);
}
}
NativeComputeShader prepare_direct_compute(const fs::path &input, unsigned entry, bool container,
                                           const fs::path &cache, bool &cache_hit, HeapViewContract heap_view) {
    if (!container) {
        if (heap_view != HeapViewContract::none)
            throw std::runtime_error("a heap view contract needs a container request");
        const auto request = read_compute_request(input);
        return cached_compute_ir(cache, slot67_digest(request, entry), cache_hit,
                                 [&] { return translate_compute(request, entry, true); });
    }
    auto packed = read_compute_container(input);
    Bytes identity = X2W_NATIVE_SHADER_REVISION "/container/cs_6_6";
    word(identity, 0);
    if (heap_view != HeapViewContract::none)
        identity += heap_view == HeapViewContract::raw_buffer    ? "/heap-view/raw_buffer"
                    : heap_view == HeapViewContract::typed_dword ? "/heap-view/typed_dword"
                                                                 : "/heap-view/other";
    identity += shader_digest(packed.bytes) + shader_digest(packed.root);
    return cached_compute_ir(cache, shader_digest(identity), cache_hit,
                             [&] { return translate_compute_container(std::move(packed), heap_view); });
}

ComputePackCounts build_compute_frontend_pack(const fs::path &requests, const fs::path &cache, unsigned entry) {
    std::vector<fs::path> markers;
    for (const auto &file : fs::directory_iterator(requests))
        if (file.is_regular_file() && file.path().extension() == ".c67" && file.path().stem().wstring().rfind(L"req_",0) == 0)
            markers.push_back(file.path());
    if (markers.empty() || markers.size() > 100000)
        throw std::runtime_error("invalid compute pack request count");
    std::sort(markers.begin(), markers.end());
    Bytes pack;
    word(pack, 0);
    ComputePackCounts counts;
    for (const auto &marker : markers) {
        try {
            const auto request = read_compute_request(marker);
            const auto digest = slot67_digest(request, entry);
            Bytes payload;
            bool unsupported = false, hit = false;
            try {
                auto program = cached_compute_ir(cache, digest, hit, [&] { return translate_compute(request, entry, true); });
                payload = encode(program);

            } catch (const UnsupportedComputeShader &error) {
                payload = error.what();
                unsupported = true;

            }
            constexpr size_t limit = (256u << 20) - 128;
            if (payload.empty() || payload.size() > limit || pack.size() + 80 + payload.size() > limit)
                throw std::runtime_error("invalid compute pack size");
            word(pack, static_cast<uint32_t>(request.key));
            word(pack, static_cast<uint32_t>(request.key >> 32));
            pack += digest;
            word(pack, unsupported);
            word(pack, static_cast<uint32_t>(payload.size()));
            pack += payload;
            if (unsupported) ++counts.unsupported;
            else ++counts.prepared;
        } catch (const std::runtime_error &e) {
            counts.errors.push_back(marker.filename().string() + ": " + e.what());
        }
    }
    const uint32_t total = static_cast<uint32_t>(counts.prepared + counts.unsupported);
    if (!total) throw std::runtime_error(counts.errors.at(0));
    std::memcpy(pack.data(), &total, sizeof(total));
    publish(cache / ("compute-" + std::to_string(entry) + ".pack"), pack_version(entry), pack);
    return counts;
}

ComputeFrontendPack ComputeFrontendPack::load(const fs::path &file, unsigned entry) {
    Bytes bytes;
    if (!x2w::lookup(file, pack_version(entry), bytes))
        throw std::runtime_error("missing, stale or damaged compute pack");
    Reader in{bytes};
    ComputeFrontendPack pack;
    pack.entry_ = entry;
    auto count = in.count(100000);
    if (!count) throw std::runtime_error("empty compute pack");
    pack.records_.reserve(count);
    for (unsigned j = 0; j < count; ++j) {
        uint64_t key = in.word();
        key |= uint64_t(in.word()) << 32;
        auto digest = in.take(64);
        bool unsupported = in.count(1) != 0;
        auto payload = in.take(in.word());
        if (!digest_text(digest) || payload.empty() || (unsupported && payload.size() > 4096) ||
            !pack.records_.emplace(key, Record{Bytes(digest), Bytes(payload), unsupported}).second)
            throw std::runtime_error("invalid compute pack record");
    }
    if (!in.remaining.empty()) throw std::runtime_error("trailing compute pack data");
    return pack;
}

ComputePackResult ComputeFrontendPack::lookup(const ComputeRequest &request, unsigned entry,
                                               NativeComputeShader &program, Bytes &diagnostic) const {
    auto it = records_.find(request.key);
    if (it == records_.end() || entry != entry_ || it->second.digest != slot67_digest(request, entry))
        return ComputePackResult::miss;
    if (it->second.unsupported) {
        diagnostic = it->second.payload;
        return ComputePackResult::unsupported;
    }
    try { program = decode(it->second.payload); }
    catch (const std::runtime_error &) { return ComputePackResult::miss; }
    return ComputePackResult::prepared;
}

unsigned dxil_io_component(unsigned format) {
    switch (format) {
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        return 9;
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R10G10B10A2_UINT:
        return 5;
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R32G32B32_SINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
        return 4;
    }
    return 0;
}

// The capture writer serializes the guest's D3D12_INPUT_ELEMENT_DESC array as ia1.
// Keep APPEND_ALIGNED_ELEMENT intact so D3D12 uses the preceding element of its slot.
static std::vector<NativeVertexElement> vertex_layout(std::string_view text, const NativeComputeShader &vs) {
    std::vector<NativeVertexElement> elements;
    std::set<std::pair<std::string, unsigned>> names;
    auto upper = [](std::string name) {
        for (char &c : name)
            if (c >= 'a' && c <= 'z')
                c -= 'a' - 'A';
        return name;
    };
    if (!text.empty()) {
        std::istringstream in{std::string(text)};
        std::string token;
        auto number = [&]() {
            unsigned value = 0;
            if (!(in >> token))
                throw UnsupportedComputeShader("truncated input layout");
            const auto end = token.data() + token.size();
            auto parsed = std::from_chars(token.data(), end, value);
            if (parsed.ec != std::errc{} || parsed.ptr != end)
                throw UnsupportedComputeShader("invalid input layout number");
            return value;
        };
        if (!(in >> token) || token != "ia1")
            throw UnsupportedComputeShader("invalid input layout version");
        const unsigned count = number();
        if (!count || count > 32)
            throw UnsupportedComputeShader("invalid input element count");
        for (unsigned i = 0; i < count; ++i) {
            NativeVertexElement e{};
            if (!(in >> e.semantic) || e.semantic.empty() || e.semantic.size() > 63 ||
                e.semantic.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") !=
                    std::string::npos ||
                (e.semantic.front() >= '0' && e.semantic.front() <= '9'))
                throw UnsupportedComputeShader("invalid input semantic");
            e.semantic = upper(e.semantic);
            e.index = number();
            e.format = number();
            e.slot = number();
            e.offset = number();
            e.classification = number();
            e.step_rate = number();
            if (!dxil_io_component(e.format))
                throw UnsupportedComputeShader("unsupported vertex input format");
            if (e.slot >= 8 || e.classification > 1 || (!e.classification && e.step_rate) ||
                (e.offset != UINT_MAX && e.offset >= D3D12_REQ_MULTI_ELEMENT_STRUCTURE_SIZE_IN_BYTES))
                throw UnsupportedComputeShader("unsupported vertex input slot, offset or classification");
            if (!names.emplace(e.semantic, e.index).second)
                throw UnsupportedComputeShader("duplicate input semantic");
            for (const auto &prior : elements)
                if (e.slot == prior.slot &&
                    (e.classification != prior.classification || e.step_rate != prior.step_rate))
                    throw UnsupportedComputeShader("inconsistent input slot classification");
            elements.push_back(std::move(e));
        }
        if (in >> token)
            throw UnsupportedComputeShader("trailing input layout elements");
    }
    for (const auto &row : vs.inputs) {
        const auto name = upper(row.name);
        if (name.rfind("SV_", 0) == 0)
            continue;
        auto it = std::find_if(elements.begin(), elements.end(),
                               [&](const auto &e) { return e.semantic == name && e.index == row.index; });
        if (it == elements.end())
            throw UnsupportedComputeShader("missing vertex input element " + name);
        if (dxil_io_component(it->format) != row.component)
            throw UnsupportedComputeShader("vertex format disagrees with shader input " + name);
        it->used = true;
    }
    return elements;
}

DirectGraphicsPair prepare_direct_graphics(std::string_view a, std::string_view b, const fs::path &cache,
                                           bool &cache_hit, std::string_view root, std::string_view input_layout,
                                           std::string_view pixel_entry, bool write_cache,
                                           const std::vector<unsigned> &rtv_formats) {
    // An RTV variant types each color target by its format, as tools/startup/gfx2hlsl.py mrt_kind does: u32 or i32
    // for the integer formats, f32 for every other format.
    std::array<unsigned char, 8> targets{};
    if (rtv_formats.size() > targets.size()) throw UnsupportedComputeShader("invalid RTV variant values");
    for (size_t i = 0; i < rtv_formats.size(); ++i) {
        const unsigned component = dxil_io_component(rtv_formats[i]);
        targets[i] = static_cast<unsigned char>(component == 5 || component == 4 ? component : 9);
    }
    // The two three-channel 32-bit integer formats have no verified readback there. As there, one is refused by
    // name only when the pixel stage exports to its target; the stage's cached form does not depend on which
    // format of a type was asked for, so this is checked on a cache hit too.
    const auto verified = [&](DirectGraphicsPair pair) {
        for (size_t i = 0; i < rtv_formats.size(); ++i)
            if (pair.pixel.pixel_writes[i] &&
                (rtv_formats[i] == DXGI_FORMAT_R32G32B32_UINT || rtv_formats[i] == DXGI_FORMAT_R32G32B32_SINT))
                throw UnsupportedComputeShader(std::string("unverified ") + (targets[i] == 5 ? "UINT" : "SINT") +
                                               " RTV format " + std::to_string(rtv_formats[i]));
        return pair;
    };
    // A fixed PSO can expose a shorter serialized desktop root than the
    // guest's hardware descriptor table. An explicitly supplied companion
    // describes the guest ABI. Never use its executable code in the pair.
    std::array<Bytes, 2> companions;
    const std::array<std::string_view, 2> raw{a, b};
    const std::array<Bytes, 2> digests{shader_digest(a), shader_digest(b)};
    Bytes identity = Bytes(X2W_NATIVE_SHADER_REVISION) + "/linked/vs_ps_6_6/";
    if (!cache.empty() && a.substr(0,4) != "DXBC" && b.substr(0,4) != "DXBC") {
        for (size_t i = 0; i < raw.size(); ++i) {
            auto file = cache / "interfaces" / (digests[i] + ".cdxil");
            if (std::filesystem::exists(file)) companions[i] = read_shader_container_bytes(file);
        }
        if (companions[0].empty() || companions[1].empty()) {
            const bool pinned = !companions[0].empty() || !companions[1].empty();
            const auto library = cache / "interfaces" / "library";
            if (auto index = load_interfaces(library)) {
                for (size_t i = 0; i < raw.size(); ++i)
                    if (companions[i].empty()) companions[i] = indexed_interface(library,*index,raw[i],digests[i]);
                // An incomplete library must still allow a self-describing raw pair.
                if (!pinned && (companions[0].empty() || companions[1].empty())) companions = {};
            }
        }
    }
    for (size_t i = 0; i < raw.size(); ++i) {
        identity += digests[i];
        identity += shader_digest(companions[i]);
    }
    const bool reflected = !companions[0].empty();
    if (reflected != !companions[1].empty()) throw UnsupportedComputeShader("incomplete raw graphics interface pair");
    if (reflected && root.empty()) throw UnsupportedComputeShader("missing captured root for reflected graphics");
    // The base pair keeps its key. A variant adds its target types, so formats of one type share an entry.
    const Bytes typed = rtv_formats.empty() ? Bytes{}
        : shader_digest(std::string_view(reinterpret_cast<const char *>(targets.data()), targets.size()));
    const auto key = shader_digest(identity + shader_digest(root) + shader_digest(input_layout) + shader_digest(pixel_entry) + typed);
    const auto path = cache / (key + ".pairir");
    Bytes payload;
    cache_hit = write_cache && !cache.empty() && lookup(path, key, payload);
    auto effective_layout = [&](const NativeComputeShader &vs) {
        if (!input_layout.empty() || !vs.vertex_fetch) return std::string(input_layout);
        unsigned count = 0;
        std::ostringstream elements;
        for (const auto &r : vs.inputs) {
            if (r.name != "X2W_RAW_VERTEX") continue;
            unsigned format = 0;
            if (r.component == 5 && r.width == 1) format = DXGI_FORMAT_R8_UINT;
            if (r.component == 5 && r.width == 2) format = DXGI_FORMAT_R32G32_UINT;
            if (r.component == 5 && r.width == 3) format = DXGI_FORMAT_R32G32B32_UINT;
            if (r.component == 5 && r.width == 4) format = DXGI_FORMAT_R32G32B32A32_UINT;
            if (r.component == 9 && r.width == 1) format = DXGI_FORMAT_R8_UNORM;
            if (r.component == 9 && r.width == 2) format = DXGI_FORMAT_R8G8_UNORM;
            if (r.component == 9 && r.width == 4) format = DXGI_FORMAT_R8G8B8A8_UNORM;
            if (!format) throw UnsupportedComputeShader("unsupported fixed vertex format");
            elements << " X2W_RAW_VERTEX " << r.index << ' ' << format << " 0 " << r.index << " 0 0";
            ++count;
        }
        return count ? "ia1 " + std::to_string(count) + elements.str() : std::string{};
    };
    if (cache_hit) {
        try {
            Reader in{payload};
            auto vertex = decode(in.take(in.word()));
            auto elements = vertex_layout(effective_layout(vertex), vertex);
            return verified({std::move(vertex), decode(in.remaining), std::move(elements), targets});
        } catch (const UnsupportedComputeShader &) {
            throw;
        } catch (const std::runtime_error &) {
            cache_hit = false;
        }
    }
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC empty{};
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC *desc = &empty;
    Microsoft::WRL::ComPtr<ID3D12VersionedRootSignatureDeserializer> sig;
    if (!root.empty()) {
        if (FAILED(D3D12CreateVersionedRootSignatureDeserializer(root.data(), root.size(), IID_PPV_ARGS(&sig))) ||
            FAILED(sig->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &desc)))
            throw UnsupportedComputeShader("invalid captured graphics root signature");
    }
    const auto &layout = desc->Desc_1_1;
    std::map<GpuStage, GuestShaderStage> interfaces;
    std::optional<GpuStage> first_stage;
    bool exact_pair = reflected;
    bool exact_pixel_code = false;
    Bytes entry;
    const bool fixed_entry = pixel_entry.size() == 12 && pixel_entry.substr(0,4) == "PSI1";
    if (fixed_entry && !reflected)
        throw UnsupportedComputeShader("fixed graphics entry requires matching stage interfaces");
    bool packed_xy = false;
    std::optional<unsigned> captured_interp_count;
    std::optional<std::array<bool,2>> stage_wave32;
    if (reflected) {
        for (size_t i = 0; i < companions.size(); ++i) {
            bool exact = false;
            auto matched = match_graphics_stage(raw[i],companions[i],exact);
            exact_pair &= exact;
            auto stage = matched.stage;
            if (stage == GpuStage::ps) exact_pixel_code = exact;
            if (i == 0) first_stage = stage;
            if (!interfaces.emplace(stage, std::move(matched)).second)
                throw UnsupportedComputeShader("duplicate reflected graphics stage");
        }
        if (!interfaces.count(GpuStage::vs) || !interfaces.count(GpuStage::ps))
            throw UnsupportedComputeShader("missing vertex or pixel interface");
        const auto &pixel_defaults = interfaces.at(GpuStage::ps);
        stage_wave32 = {{interfaces.at(GpuStage::vs).wave32, pixel_defaults.wave32}};
        if (fixed_entry) {
            // Fixed slot 57 supplies ADDR and NUM_INTERP, not ENA. Intersect
            // the observed ABI with the reviewed stage and the raw consumers.
            uint32_t address, interpolants;
            std::memcpy(&address, pixel_entry.data() + 4, 4);
            std::memcpy(&interpolants, pixel_entry.data() + 8, 4);
            if (!interpolants || interpolants > 7 || interpolants > pixel_defaults.attributes.size())
                throw UnsupportedComputeShader("captured interpolation count disagrees with pixel interface");
            captured_interp_count = interpolants;
            packed_xy = address == 0x8002 && pixel_defaults.pixel_input_address == 0x302;
            if (!(address == 2 && pixel_defaults.pixel_input_address == 2) && !packed_xy)
                throw UnsupportedComputeShader("unsupported fixed pixel entry layout");
            // Center I/J and, in the specialized position variant, packed XY
            // are the only allocated inputs. The raw frontend checks each
            // register use; the interpolants are checked below after linking.
            entry = "psin1";
            word(entry, address);
            word(entry, address);
            pixel_entry = entry;
        }
        if (exact_pair && !pixel_entry.empty()) {
            auto &ps = interfaces.at(GpuStage::ps);
            uint32_t enable_mask = 0, address_mask = 0;
            if (pixel_entry.size() != 13 || pixel_entry.substr(0, 5) != "psin1")
                throw UnsupportedComputeShader("invalid raw pixel entry record");
            std::memcpy(&enable_mask, pixel_entry.data() + 5, 4);
            std::memcpy(&address_mask, pixel_entry.data() + 9, 4);
            if (enable_mask > 0xffff || address_mask > 0xffff || (enable_mask & ~address_mask))
                throw UnsupportedComputeShader("contradictory pixel input enable/address masks");
            exact_pair = ps.pixel_input_address == address_mask;
            if (exact_pair)
                ps.pixel_input_enable = enable_mask;
        }
        if (pixel_entry.empty() && !exact_pair) {
            const auto &ps = interfaces.at(GpuStage::ps);
            if (ps.pixel_input_address != 2 || ps.attributes.empty() || ps.attributes.size() > 7)
                throw UnsupportedComputeShader("raw graphics requires its captured pixel entry");
            entry = "psin1";
            word(entry, ps.pixel_input_address);
            word(entry, ps.pixel_input_address);
            pixel_entry = entry;
        }
    }
    DirectGraphicsPair pair;
    if (exact_pair) {
        for (size_t i = 0; i < raw.size(); ++i) {
            auto type = i == 0 ? *first_stage : *first_stage == GpuStage::vs ? GpuStage::ps : GpuStage::vs;
            auto &stage = interfaces.at(type);
            stage.code.assign(raw[i]);
            stage.exports.clear();
        }
        interfaces.at(GpuStage::ps).target_components = targets;
        pair = translate_reflected_graphics(interfaces.at(GpuStage::vs),interfaces.at(GpuStage::ps),layout);
    } else pair = translate_graphics_pair(a, b, layout, pixel_entry, first_stage, targets, stage_wave32);
    pair.target_components = targets;
    if (!entry.empty()) {
        const auto &attrs = interfaces.at(GpuStage::ps).attributes;
        std::vector<unsigned> channels(attrs.size());
        for (const auto &r : pair.pixel.inputs) {
            if (r.kind == 28 && r.interpolation == 2) continue;
            if (packed_xy && r.kind == 3 && r.name == "SV_Position" && r.first == 0 &&
                r.index == 0 && r.width == 4 && r.interpolation == 4) continue;
            // Fixed pipelines may specialize individual components from
            // P1/P2 to VINTRP_MOV P0. The raw front end links each observed
            // component to a VS PARAM channel and emits flat mode only for
            // that P0 instruction. It may drop unused reflected rows, too.
            if (r.name != "TEXCOORD" || r.index / 4 >= channels.size() ||
                (captured_interp_count && r.index / 4 >= *captured_interp_count) ||
                (r.interpolation != 2 && !(fixed_entry && r.interpolation == 1)) ||
                r.width != 1 || r.first != r.index % 4)
                throw UnsupportedComputeShader("raw and reflected interpolation inputs disagree");
            channels[r.index / 4] |= 1u << r.first;
        }
        for (const auto &attr : attrs) {
            // A specialized raw PS can read more components of the same PARAM
            // than its library variant. The raw VS/PS link above has already
            // checked every consumed component against a matching VS export.
            const bool linked_raw_row = fixed_entry && !exact_pixel_code && attr.row == attr.location;
            if (attr.usage != 6 || attr.location >= channels.size() || attr.flags != 2 ||
                (!linked_raw_row && (channels[attr.location] & ~attr.mask)) || !attr.mask)
                throw UnsupportedComputeShader("raw and reflected interpolation channels disagree at location " +
                    std::to_string(attr.location) + " (usage " + std::to_string(attr.usage) + ", flags " +
                    std::to_string(attr.flags) + ", reflected mask " + std::to_string(attr.mask) +
                    ", raw mask " + std::to_string(attr.location < channels.size() ? channels[attr.location] : 0) + ")");
            channels[attr.location] = 0;
        }
        if (std::any_of(channels.begin(), channels.end(), [](unsigned bits) { return bits != 0; }))
            throw UnsupportedComputeShader("raw interpolation has no reflected attribute");
    }
    pair.input_layout = vertex_layout(effective_layout(pair.vertex), pair.vertex);
    if (!root.empty()) {
        for (auto *stage : {&pair.vertex, &pair.pixel}) {
            const bool vertex = stage == &pair.vertex;
            const auto visibility = vertex ? D3D12_SHADER_VISIBILITY_VERTEX : D3D12_SHADER_VISIBILITY_PIXEL;
            auto parameter = [&](unsigned index) -> const D3D12_ROOT_PARAMETER1 & {
                if (index >= layout.NumParameters ||
                    layout.Flags & (vertex ? D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS
                                           : D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS))
                    throw UnsupportedComputeShader("graphics root parameter inaccessible");
                const auto &p = layout.pParameters[index];
                if (p.ShaderVisibility != D3D12_SHADER_VISIBILITY_ALL && p.ShaderVisibility != visibility)
                    throw UnsupportedComputeShader("graphics root parameter stage visibility mismatch");
                return p;
            };
            for (const auto &c : stage->constants) {
                const auto &p = parameter(c.root_parameter);
                if (p.ParameterType != D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS ||
                    c.count > p.Constants.Num32BitValues)
                    throw UnsupportedComputeShader("graphics root constant span mismatch");
            }
            for (auto &binding : stage->bindings) {
                const auto &p = parameter(binding.root_parameter);
                if (binding.kind == ComputeViewKind::root_buffer_srv) {
                    if (p.ParameterType != D3D12_ROOT_PARAMETER_TYPE_CBV && p.ParameterType != D3D12_ROOT_PARAMETER_TYPE_SRV)
                        throw UnsupportedComputeShader("graphics scalar load requires a root CBV or SRV");
                    continue;
                }
                if (p.ParameterType != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE || binding.byte_offset % 32)
                    throw UnsupportedComputeShader("graphics binding is outside a descriptor table");
                const auto slot = binding.byte_offset / 32;
                const auto kind =
                    compute_uav(binding.kind) ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
                uint64_t next = 0;
                bool found = false;
                for (unsigned i = 0; i < p.DescriptorTable.NumDescriptorRanges; ++i) {
                    const auto &range = p.DescriptorTable.pDescriptorRanges[i];
                    const uint64_t first =
                        range.OffsetInDescriptorsFromTableStart == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
                            ? next
                            : range.OffsetInDescriptorsFromTableStart;
                    next = first + range.NumDescriptors;
                    // Scalar buffer loads also consume CBV table records. The
                    // resolver supplies a raw SRV over the recorded CBV window.
                    if (first <= slot && slot < next && (kind == range.RangeType ||
                        (accepts_cbv_window(binding.kind) && range.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_CBV)))
                        found = true;
                }
                if (reflected && binding.array_count == 1) {
                    const auto &iface = interfaces.at(vertex ? GpuStage::vs : GpuStage::ps);
                    unsigned sgpr = 8;
                    for (unsigned k = 0; k < binding.root_parameter; ++k) {
                        const auto &prev = layout.pParameters[k];
                        sgpr += prev.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE ? 1 :
                                prev.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS ? prev.Constants.Num32BitValues : 2;
                    }
                    const bool used = apply_reflected_table_use(binding, iface,
                        exact_pair ? std::optional<unsigned>{} : std::optional<unsigned>{sgpr});
                    const bool needs_record = !found || (!exact_pair && (accepts_cbv_window(binding.kind) ||
                                                                         binding.kind == ComputeViewKind::texture_srv ||
                                                                         binding.kind == ComputeViewKind::d16_sample_srv));
                    if (!used && needs_record && binding.kind != ComputeViewKind::heap_stride_srv)
                        throw UnsupportedComputeShader(std::string("missing reflected resource: ") + (vertex ? "VS" : "PS") + " root " + std::to_string(binding.root_parameter) + " offset " + std::to_string(binding.byte_offset) + " kind " + std::to_string(static_cast<unsigned>(binding.kind)) + " user SGPR " + std::to_string(sgpr));
                    found |= used;
                }
                apply_scalar_table_layout(binding, p.DescriptorTable);
                if (!found)
                    throw UnsupportedComputeShader(std::string("graphics descriptor range does not cover resource: ") +
                        (vertex ? "VS" : "PS") + " root " + std::to_string(binding.root_parameter) +
                        " byte offset " + std::to_string(binding.byte_offset) + " slot " + std::to_string(slot));
            }
        }
    }
    if (write_cache && !cache.empty()) {
        auto vertex = encode(pair.vertex);
        payload.clear();
        word(payload, static_cast<uint32_t>(vertex.size()));
        payload += vertex;
        payload += encode(pair.pixel);
        publish(path, key, payload);
    }
    return verified(std::move(pair));
}

std::optional<std::string> format_compute_runtime_manifest(const NativeComputeShader &program) {
    if (!program.group[0] || !program.group[1] || !program.group[2])
        return std::nullopt;
    unsigned ud_count = 0;
    for (const auto &block : program.constants) {
        if (block.root_parameter >= 16 || block.count > 8)
            return std::nullopt;
        ud_count += block.count;
    }
    if (ud_count > 32)
        return std::nullopt;

    auto sorted = program.bindings;
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
        const bool ua = compute_uav(a.kind), ub = compute_uav(b.kind);
        if (ua != ub)
            return !ua;
        return a.shader_register < b.shader_register;
    });

    unsigned srv_count = 0, uav_count = 0;
    std::ostringstream srv_lines, uav_lines, sampler_lines;
    for (const auto &b : sorted) {
        if (b.root_parameter >= 16)
            return std::nullopt;
        if (!compute_uav(b.kind)) {
            if (b.shader_register != srv_count++)
                return std::nullopt;
            if (b.kind == ComputeViewKind::buffer_srv || b.kind == ComputeViewKind::raw_vector_srv ||
                b.kind == ComputeViewKind::byte_window_srv) {
                srv_lines << "srv " << b.shader_register << " buf table " << b.root_parameter << ' ' << b.byte_offset
                          << '\n';
            } else if (b.kind == ComputeViewKind::texture_srv && b.image_dimension == 1 && b.array_count == 1) {
                srv_lines << "srv " << b.shader_register << " tex2d table " << b.root_parameter << ' ' << b.byte_offset
                          << '\n';
            } else {
                return std::nullopt;
            }
        } else {
            if (b.shader_register != uav_count++)
                return std::nullopt;
            if (b.kind == ComputeViewKind::raw_buffer_uav) {
                uav_lines << "uav " << b.shader_register << " table " << b.root_parameter << ' ' << b.byte_offset
                          << '\n';
            } else if ((b.kind == ComputeViewKind::texture_uav || b.kind == ComputeViewKind::half4_texture_uav) &&
                       (b.image_dimension == 1 || b.image_dimension == 5 ||
                        (b.kind == ComputeViewKind::texture_uav && b.image_dimension == 2)) && b.array_count == 1) {
                uav_lines << "uav " << b.shader_register << (b.image_dimension == 5 ? " tex2da" : b.image_dimension == 2 ? " tex3d" : " tex2d")
                          << " float table " << b.root_parameter << ' ' << b.byte_offset << " channels 4\n";
            } else {
                return std::nullopt;
            }
        }
    }
    if (srv_count > 32 || uav_count > 8)
        return std::nullopt;

    for (size_t i = 0; i < program.samplers.size(); ++i) {
        const auto &s = program.samplers[i];
        if (s.comparison)
            return std::nullopt;
        char line[64];
        std::snprintf(line, sizeof(line), "%s %zu %08x %08x %08x %08x\n", s.explicit_lod ? "smpl" : "smp", i,
                      s.words[0], s.words[1], s.words[2], s.words[3]);
        sampler_lines << line;
    }

    std::ostringstream out;
    out << "x2w-manifest 2\nstage cs\ngroup " << program.group[0] << ' ' << program.group[1] << ' ' << program.group[2]
        << "\nud " << ud_count << '\n';
    unsigned ud_index = 0;
    for (const auto &block : program.constants)
        for (unsigned j = 0; j < block.count; ++j)
            out << "udsrc " << ud_index++ << " rootconst " << block.root_parameter << ' ' << j << '\n';
    out << srv_lines.str() << uav_lines.str() << sampler_lines.str();
    return out.str();
}

std::optional<std::string> format_graphics_runtime_manifest(const DirectGraphicsPair &pair, std::string_view rtv_request) {
    const bool fixed_stream = pair.vertex.vertex_fetch && !pair.input_layout.empty();
    if (pair.vertex.vertex_fetch != !pair.input_layout.empty())
        return std::nullopt;
    for (const auto &e : pair.input_layout)
        if (e.semantic != "X2W_RAW_VERTEX" || e.slot || e.offset != e.index || e.classification || e.step_rate ||
            !e.used || (e.format != DXGI_FORMAT_R8_UNORM && e.format != DXGI_FORMAT_R8G8_UNORM &&
                        e.format != DXGI_FORMAT_R8G8B8A8_UNORM && e.format != DXGI_FORMAT_R32G32_UINT &&
                        e.format != DXGI_FORMAT_R32G32B32_UINT && e.format != DXGI_FORMAT_R32G32B32A32_UINT &&
                        e.format != DXGI_FORMAT_R8_UINT))
            return std::nullopt;
    for (const auto &in : pair.vertex.inputs)
        if (in.name != "SV_VertexID" && in.name != "SV_InstanceID" && !(fixed_stream && in.name == "X2W_RAW_VERTEX"))
            return std::nullopt;
    // An integer target is published only for the RTV variant whose format asked for that type.
    for (const auto &out : pair.pixel.outputs)
        if (out.name == "SV_Target" && out.component != 9 &&
            (out.index >= pair.target_components.size() || out.component != pair.target_components[out.index]))
            return std::nullopt;

    std::optional<NativeGraphicsBindings> maybe_bindings;
    GraphicsConstantLayout constants;
    try {
        maybe_bindings.emplace(pair);
        constants = pair.constant_layout();
    } catch (const std::exception &) {
        return std::nullopt;
    }
    const auto &bindings = *maybe_bindings;
    const unsigned uav_count = static_cast<unsigned>(bindings.views.size()) - bindings.srv_count;
    if (bindings.srv_count > 32 || uav_count > 8)
        return std::nullopt;
    for (unsigned count : constants.counts)
        if (count > 8)
            return std::nullopt;
    const unsigned total_constants = constants.offset(static_cast<unsigned>(constants.counts.size()));
    if (total_constants > 32)
        return std::nullopt;

    std::ostringstream srv_lines, uav_lines, sampler_lines;
    for (unsigned i = 0; i < bindings.srv_count; ++i) {
        const auto &b = bindings.views[i];
        if (b.root_parameter >= 16)
            return std::nullopt;
        if (b.kind == ComputeViewKind::buffer_srv || b.kind == ComputeViewKind::raw_vector_srv ||
            b.kind == ComputeViewKind::byte_window_srv) {
            srv_lines << "srv " << i << " buf table " << b.root_parameter << ' ' << b.byte_offset << '\n';
        } else if (b.kind == ComputeViewKind::indexed_raw_srv && b.array_count == 1 &&
                   b.required_bytes && !(b.required_bytes % 4) && b.required_bytes <= 2048) {
            srv_lines << "srv " << i << " idxbuf table " << b.root_parameter << ' ' << b.byte_offset << ' ' << b.required_bytes << '\n';
        } else if (b.kind == ComputeViewKind::root_buffer_srv && b.byte_offset == 0 && b.array_count == 1) {
            srv_lines << "srv " << i << " buf root " << b.root_parameter << '\n';
        } else if (b.kind == ComputeViewKind::texture_srv || b.kind == ComputeViewKind::comparison_texture_srv) {
            const char *tex = b.image_dimension == 0   ? "tex1d"
                              : b.image_dimension == 1 ? "tex2d"
                              : b.image_dimension == 2 ? "tex3d"
                              : b.image_dimension == 5 ? "tex2da"
                                                       : nullptr;
            if (!tex)
                return std::nullopt;
            srv_lines << "srv " << i << ' ' << tex << " table " << b.root_parameter << ' ' << b.byte_offset << '\n';
        } else {
            return std::nullopt;
        }
    }
    for (unsigned j = 0; j < uav_count; ++j) {
        const auto &b = bindings.views[bindings.srv_count + j];
        if (b.root_parameter >= 16 || b.array_count != 1)
            return std::nullopt;
        if (b.kind != ComputeViewKind::texture_uav && b.kind != ComputeViewKind::half4_texture_uav)
            return std::nullopt;
        if (b.image_dimension != 1 && b.image_dimension != 5)
            return std::nullopt;
        uav_lines << "uav " << j << (b.image_dimension == 5 ? " tex2da table " : " tex2d table ") << b.root_parameter
                  << ' ' << b.byte_offset << '\n';
    }
    if (bindings.samplers.heap) sampler_lines << "smpheap 2048 1 fallback 2047\n";
    for (size_t i = 0; i < bindings.samplers.statics.size(); ++i) {
        const auto &w = bindings.samplers.statics[i];
        const bool cmp = i < bindings.samplers.comparison.size() && bindings.samplers.comparison[i];
        const bool lod = !cmp && i < bindings.samplers.lod_limits.size() && bindings.samplers.lod_limits[i];
        char line[64];
        std::snprintf(line, sizeof(line), "%s %zu %08x %08x %08x %08x\n", cmp ? "smpc" : lod ? "smpl" : "smp", i, w[0],
                      w[1], w[2], w[3]);
        sampler_lines << line;
    }

    unsigned params = 0;
    for (const auto &out : pair.vertex.outputs)
        if (out.name == "TEXCOORD")
            params = (std::max)(params, out.index / 4 + 1);

    std::ostringstream out;
    out << "x2w-manifest " << (fixed_stream ? 3 : uav_count ? 2 : 1) << "\nud " << (total_constants ? total_constants : 1u) << '\n';
    for (const auto &e : pair.input_layout)
        out << "fixedvb " << e.offset << ' ' << e.format << '\n';
    if (!total_constants) {
        out << "udsrc 0 rect\n";
    } else {
        unsigned ud_index = 0;
        for (unsigned r = 0; r < constants.counts.size(); ++r)
            for (unsigned j = 0; j < constants.counts[r]; ++j)
                out << "udsrc " << ud_index++ << " rootconst " << r << ' ' << j << '\n';
    }
    out << srv_lines.str() << uav_lines.str() << sampler_lines.str() << "params " << params << "\ntargets";
    bool any_target = false;
    for (unsigned t = 0; t < pair.pixel.pixel_writes.size(); ++t) {
        if (!pair.pixel.pixel_writes[t])
            continue;
        out << ' ' << t;
        any_target = true;
    }
    if (!any_target)
        out << ' ';
    out << '\n';
    for (unsigned t = 0; t < pair.pixel.pixel_writes.size(); ++t) {
        const unsigned mask = pair.pixel.pixel_writes[t];
        if (mask > 0 && mask < 0xf) {
            char line[16];
            std::snprintf(line, sizeof(line), "wmask %u 0x%x\n", t, mask);
            out << line;
        }
    }
    if (!rtv_request.empty())
        out << rtv_request << '\n';
    return out.str();
}
} // namespace x2w
