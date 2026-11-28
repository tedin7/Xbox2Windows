// Mesh replay oracle (opt-in, off by default): X2W_REPLAY_ORACLE=mesh draws a sampled slot 57 DispatchMesh twice from
// the same inputs, once with the published proof translation (<key>.ms57.cso) and once with the serial wave-loop
// translation of the same key (<key>.ms57w.cso, published by gfx2hlsl.py with X2W_MESH_WAVELOOP=oracle), and compares
// the two images of every bound render target and of the depth view. Per sampled draw, in the title's own command list:
//   1. the bound targets and depth are copied aside (snapshot);
//   2. the normal draw runs as it always does, and each target is copied aside again (normal result) and read back;
//   3. the snapshot is copied back, the serial pipeline draws, and each target is read back;
//   4. the normal result is copied back and the normal pipeline is set again, with every resource state as it was, so
//      what the list presents is the normal draw's result.
// When the list is submitted, the queue is waited for and one JSON line per sample goes to X2W_REPLAY_ORACLE_FILE
// (x2w_replay sets <out dir>\oracle.jsonl; default .\oracle.jsonl): verdict equal, differs (first differing texel,
// differing texel count, largest float distance) or oracle_unavailable with the reason. The comparison rule is
// mesh_oracle_compare.h (bit-exact, f32 within 4 ULP, f16 within 2 ULP).
//   X2W_REPLAY_ORACLE=mesh            every slot 57 key that has a serial sibling
//   X2W_REPLAY_ORACLE=mesh:<hex>,...  only these translation keys
//   X2W_REPLAY_ORACLE_SAMPLES=N       sampled draws per key (default 4)
//   X2W_REPLAY_ORACLE_MAX_MB=N        bytes one sample may allocate (default 1024)
// A draw is never sampled when it would not be repeatable: the pipeline writes UAVs, a query is open on the list, a
// target is multisampled or not a texture. Those draws say oracle_unavailable once per key and reason.
// startup_stubs/mesh_oracle_compare.h is included at the top of d3d12_xs_device.cpp (outside its anonymous namespace).

#include <mutex>

namespace mesh_oracle {

struct Config {
    bool enabled = false;
    bool all_keys = true;
    std::vector<u64> keys;
    UINT samples = 4;
    u64 max_bytes = u64{1024} << 20;
    std::wstring file = L"oracle.jsonl";
};

inline const Config& config()
{
    static const Config value = [] {
        Config c;
        char text[4096]{};
        const DWORD n = GetEnvironmentVariableA("X2W_REPLAY_ORACLE", text, sizeof(text));
        if (!n || n >= sizeof(text)) return c;
        const std::string mode(text);
        if (mode == "mesh") {
            c.enabled = true;
        } else if (mode.compare(0, 5, "mesh:") == 0) {
            c.enabled = true;
            c.all_keys = false;
            std::size_t at = 5;
            while (at < mode.size()) {
                const std::size_t end = mode.find(',', at);
                const std::string item = mode.substr(at, end == std::string::npos ? std::string::npos : end - at);
                if (!item.empty()) c.keys.push_back(std::strtoull(item.c_str(), nullptr, 16));
                if (end == std::string::npos) break;
                at = end + 1;
            }
        }
        char number[32]{};
        if (GetEnvironmentVariableA("X2W_REPLAY_ORACLE_SAMPLES", number, sizeof(number)))
            c.samples = static_cast<UINT>(std::strtoul(number, nullptr, 10));
        if (GetEnvironmentVariableA("X2W_REPLAY_ORACLE_MAX_MB", number, sizeof(number)))
            c.max_bytes = u64{std::strtoull(number, nullptr, 10)} << 20;
        wchar_t path[MAX_PATH]{};
        const DWORD w = GetEnvironmentVariableW(L"X2W_REPLAY_ORACLE_FILE", path, MAX_PATH);
        if (w && w < MAX_PATH) c.file = path;
        if (c.enabled)
            log("x2w-oracle mesh: on (%s keys, %u samples per key, f32 %u ULP, f16 %u ULP, other formats exact)",
                c.all_keys ? "all" : "listed", c.samples, x2w_startup::mesh_oracle_f32_ulp, x2w_startup::mesh_oracle_f16_ulp);
        return c;
    }();
    return value;
}

inline bool selected(u64 key)
{
    const Config& c = config();
    return c.enabled && (c.all_keys || std::find(c.keys.begin(), c.keys.end(), key) != c.keys.end());
}

// The serial stages of one key, read once from the shader cache.
struct Stages {
    bool loaded = false;
    std::string unavailable;  // empty when ms (and ps for a pipeline with a pixel stage) are usable
    x2w_startup::ComPtr<ID3DBlob> ms, ps;
};

struct Readback {
    UINT subresource = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0;
    x2w_startup::OracleElement element = x2w_startup::OracleElement::exact;
};

struct Target {
    std::string name;  // rt<N> or depth
    DXGI_FORMAT view = DXGI_FORMAT_UNKNOWN;
    x2w_startup::ComPtr<ID3D12Resource> normal, serial;  // readback buffers
    std::vector<Readback> parts;
};

struct Sample {
    ID3D12CommandList* list = nullptr;
    u64 key = 0;
    UINT key_draw = 0;  // the key's mesh draw ordinal in this process (0 = first)
    UINT sample = 0;
    LONG frame = 0;
    u64 mesh_draw = 0;  // the process's slot 57 mesh draw ordinal
    std::vector<Target> targets;
};

struct KeyState {
    UINT draws = 0;
    UINT sampled = 0;
    Stages stages;
    std::vector<std::string> reported;  // unavailable reasons already written
};

struct State {
    std::mutex mutex;
    std::unordered_map<u64, KeyState> keys;
    std::unordered_map<ID3D12PipelineState*, x2w_startup::ComPtr<ID3D12PipelineState>> serial_of;
    std::vector<Sample> pending;
    u64 mesh_draws = 0;
    bool header = false;
};

inline State& state()
{
    static State* value = new State;  // never destroyed: the process may end with GPU objects still referenced
    return *value;
}

inline std::string json_text(const std::string& text)
{
    std::string out;
    for (const char ch : text) {
        if (ch == '"' || ch == '\\') out += '\\';
        if (static_cast<unsigned char>(ch) < 0x20) continue;
        out += ch;
    }
    return out;
}

// state().mutex held.
inline void write_line(State& s, const std::string& line)
{
    FILE* file = _wfopen(config().file.c_str(), L"ab");
    if (!file) {
        static LONG failed;
        if (first_calls(failed, 4)) log("x2w-oracle mesh: cannot append to the oracle file");
        return;
    }
    if (!s.header) {
        s.header = true;
        std::fprintf(file, "{\"oracle\":\"mesh\",\"f32_ulp\":%u,\"f16_ulp\":%u,\"other\":\"exact\"}\n",
                     x2w_startup::mesh_oracle_f32_ulp, x2w_startup::mesh_oracle_f16_ulp);
    }
    std::fwrite(line.data(), 1, line.size(), file);
    std::fputc('\n', file);
    std::fclose(file);
}

inline std::string sample_head(u64 key, UINT key_draw, u64 mesh_draw, LONG frame)
{
    char text[160];
    std::snprintf(text, sizeof(text), "{\"key\":\"%016llx\",\"draw\":%u,\"mesh_draw\":%llu,\"frame\":%ld", key, key_draw,
                  static_cast<unsigned long long>(mesh_draw), static_cast<long>(frame));
    return text;
}

// state().mutex held. One line per (key, reason).
inline void unavailable(State& s, KeyState& k, u64 key, UINT key_draw, const std::string& reason)
{
    if (std::find(k.reported.begin(), k.reported.end(), reason) != k.reported.end()) return;
    k.reported.push_back(reason);
    write_line(s, sample_head(key, key_draw, s.mesh_draws, frame_number) +
                      ",\"verdict\":\"oracle_unavailable\",\"reason\":\"" + json_text(reason) + "\"}");
}

// state().mutex held.
inline void load_stages(KeyState& k, const Translation& tr)
{
    if (k.stages.loaded) return;
    k.stages.loaded = true;
    wchar_t path[MAX_PATH];
    std::string status;
    if (!cache_path(path, L"", tr.key, L".ms57w.txt") || !read_text(path, status)) {
        k.stages.unavailable = "no serial sibling (translate with X2W_MESH_WAVELOOP=oracle)";
        return;
    }
    while (!status.empty() && (status.back() == '\n' || status.back() == '\r')) status.pop_back();
    if (status != "ok") {
        k.stages.unavailable = status.empty() ? "empty serial sibling status" : status;
        return;
    }
    const auto load = [&](const wchar_t* suffix, x2w_startup::ComPtr<ID3DBlob>& blob) {
        return cache_path(path, L"", tr.key, suffix) && SUCCEEDED(D3DReadFileToBlob(path, &blob)) && blob;
    };
    if (!load(L".ms57w.cso", k.stages.ms) || (tr.ps && !load(L".ps57w.cso", k.stages.ps))) {
        k.stages.unavailable = "serial sibling bytecode missing";
        return;
    }
    const D3D12_SHADER_BYTECODE ms{k.stages.ms->GetBufferPointer(), k.stages.ms->GetBufferSize()};
    if (desktop_dxil_mesh_stage(ms) != DxilMeshStage::mesh ||
        (k.stages.ps && desktop_dxil_mesh_stage({k.stages.ps->GetBufferPointer(), k.stages.ps->GetBufferSize()}) !=
                            DxilMeshStage::pixel)) {
        k.stages.ms.Reset();
        k.stages.ps.Reset();
        k.stages.unavailable = "serial sibling bytecode stage mismatch";
    }
}

}  // namespace mesh_oracle

inline bool mesh_oracle_enabled() { return mesh_oracle::config().enabled; }

// The serial twin of a slot 57 pipeline: the same pipeline stream with the serial mesh and pixel stages. Called right
// after the normal pipeline was created from `stream`.
template <class Stream>
void mesh_oracle_pipeline(ID3D12Device2* device, const Translation& tr, const Stream& stream, ID3D12PipelineState* normal)
{
    if (!tr.mesh || !normal || !mesh_oracle::selected(tr.key)) return;
    auto& s = mesh_oracle::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto& k = s.keys[tr.key];
    mesh_oracle::load_stages(k, tr);
    if (!k.stages.unavailable.empty() || !k.stages.ms) return;
    Stream serial = stream;
    serial.ms.value = {k.stages.ms->GetBufferPointer(), k.stages.ms->GetBufferSize()};
    if (k.stages.ps) serial.ps.value = {k.stages.ps->GetBufferPointer(), k.stages.ps->GetBufferSize()};
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(serial), &serial};
    x2w_startup::ComPtr<ID3D12PipelineState> pipeline;
    const HRESULT hr = device->CreatePipelineState(&desc, IID_PPV_ARGS(&pipeline));
    log("x2w-oracle mesh %016llx: serial pipeline -> 0x%08lx", tr.key, static_cast<unsigned long>(hr));
    if (SUCCEEDED(hr) && pipeline) s.serial_of[normal] = pipeline;
}

namespace mesh_oracle {

// The tracked state of one resource on this list, to put back after the oracle's copies.
struct SavedState {
    ID3D12Resource* resource = nullptr;
    bool split = false;
    x2w_substates::Split subresources{};
    D3D12_RESOURCE_STATES whole = D3D12_RESOURCE_STATE_COMMON;
};

inline SavedState save_state(Pass& pass, ID3D12Resource* resource)
{
    SavedState saved;
    saved.resource = resource;
    const auto split = find_split(pass, resource);
    if (split != pass.split.end()) {
        saved.split = true;
        saved.subresources = *split;
        return saved;
    }
    const auto known = find_state(pass, resource);
    if (known != pass.states.end()) saved.whole = known->second;
    return saved;
}

inline void restore_state(Wrapper* self, ID3D12GraphicsCommandList* list, const SavedState& saved)
{
    if (!saved.split) return require_state(self, saved.resource, saved.whole);
    auto& split = split_state(pass_of(self), saved.resource);
    x2w_substates::move_each(list, split, x2w_substates::whole(saved.resource->GetDesc()),
                             [&](UINT index, D3D12_RESOURCE_STATES from) {
                                 return index < saved.subresources.state.size() ? saved.subresources.state[index] : from;
                             });
}

// The subresources a bound view covers, as D3D12 subresource indices.
inline std::vector<UINT> view_subresources(const D3D12_RESOURCE_DESC& desc, const x2w_substates::Range& range)
{
    std::vector<UINT> out;
    const UINT mips = desc.MipLevels ? desc.MipLevels : 1;
    const UINT array = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : desc.DepthOrArraySize;
    const UINT slices = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : range.slices;
    const UINT first_slice = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 0u : range.first_slice;
    for (UINT p = range.first_plane; p < range.first_plane + range.planes; ++p)
        for (UINT a = first_slice; a < first_slice + slices; ++a)
            for (UINT m = range.first_mip; m < range.first_mip + range.mips; ++m)
                if (m < mips && a < array) out.push_back(m + a * mips + p * mips * array);
    return out;
}

struct Prepared {
    ID3D12Resource* resource = nullptr;
    x2w_startup::ComPtr<ID3D12Resource> snapshot, result;  // whole-resource copies on the default heap
    Target target;
    SavedState saved;
};

// Allocates the copies one target needs; empty string on success, else the reason the draw is not sampled.
inline std::string prepare(ID3D12Device* device, ID3D12Resource* resource, const x2w_substates::Range& range,
                           DXGI_FORMAT view, const char* name, u64& bytes, Prepared& out)
{
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || desc.Dimension == D3D12_RESOURCE_DIMENSION_UNKNOWN)
        return std::string(name) + " is not a texture";
    if (desc.SampleDesc.Count != 1) return std::string(name) + " is multisampled";
    out.resource = resource;
    out.target.name = name;
    out.target.view = view != DXGI_FORMAT_UNKNOWN ? view : desc.Format;
    const D3D12_RESOURCE_ALLOCATION_INFO info = device->GetResourceAllocationInfo(0, 1, &desc);
    bytes += 2 * info.SizeInBytes;
    UINT64 offset = 0;
    for (const UINT sub : view_subresources(desc, range)) {
        Readback part;
        part.subresource = sub;
        device->GetCopyableFootprints(&desc, sub, 1, offset, &part.footprint, &part.rows, &part.row_bytes, nullptr);
        const UINT mips = desc.MipLevels ? desc.MipLevels : 1;
        const UINT array = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : desc.DepthOrArraySize;
        part.element = x2w_startup::oracle_element(out.target.view, sub / (mips * array));
        offset = part.footprint.Offset + u64{part.footprint.Footprint.RowPitch} * part.rows * part.footprint.Footprint.Depth;
        offset = (offset + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) & ~u64{D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1};
        out.target.parts.push_back(part);
    }
    if (out.target.parts.empty()) return std::string(name) + " view covers no subresource";
    bytes += 2 * offset;
    if (bytes > config().max_bytes) return "targets exceed X2W_REPLAY_ORACLE_MAX_MB";
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = offset;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, IID_PPV_ARGS(&out.snapshot))) ||
        FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, IID_PPV_ARGS(&out.result))) ||
        FAILED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&out.target.normal))) ||
        FAILED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&out.target.serial))))
        return "oracle copies could not be allocated";
    x2w_budget::committed(device, out.snapshot.Get(), D3D12_HEAP_TYPE_DEFAULT);
    x2w_budget::committed(device, out.result.Get(), D3D12_HEAP_TYPE_DEFAULT);
    x2w_budget::committed(device, out.target.normal.Get(), D3D12_HEAP_TYPE_READBACK);
    x2w_budget::committed(device, out.target.serial.Get(), D3D12_HEAP_TYPE_READBACK);
    return {};
}

// The copies live with the list's allocator until it is reset idle, like the list's other scratch resources.
inline bool retain(Wrapper* self, const std::vector<Prepared>& prepared)
{
    bool retained = false;
    AcquireSRWLockExclusive(&allocator_lock);
    const auto found = list_recordings.find(self);
    if (found != list_recordings.end() && found->second) {
        try {
            for (const Prepared& p : prepared) {
                for (ID3D12Resource* r : {p.snapshot.Get(), p.result.Get(), p.target.normal.Get(), p.target.serial.Get()})
                    found->second->scratches.emplace_back(r);
            }
            retained = true;
        } catch (const std::bad_alloc&) {
        }
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    return retained;
}

inline void copy_whole(ID3D12GraphicsCommandList* list, ID3D12Resource* to, ID3D12Resource* from) { list->CopyResource(to, from); }

inline void oracle_barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES from,
                       D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &barrier);
}

inline void read_back(ID3D12GraphicsCommandList* list, ID3D12Resource* source, ID3D12Resource* buffer, const Target& target)
{
    for (const Readback& part : target.parts) {
        D3D12_TEXTURE_COPY_LOCATION to{buffer, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint = part.footprint;
        D3D12_TEXTURE_COPY_LOCATION from{source, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        from.SubresourceIndex = part.subresource;
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
}

// The verdict line of one completed sample.
inline std::string compare(const Sample& sample)
{
    std::string line = sample_head(sample.key, sample.key_draw, sample.mesh_draw, sample.frame);
    std::string targets;
    bool differs = false;
    for (const Target& target : sample.targets) {
        void* a = nullptr;
        void* b = nullptr;
        if (FAILED(target.normal->Map(0, nullptr, &a)) || FAILED(target.serial->Map(0, nullptr, &b))) {
            if (a) target.normal->Unmap(0, nullptr);
            return line + ",\"verdict\":\"oracle_unavailable\",\"reason\":\"readback map failed\"}";
        }
        x2w_startup::OracleDiff diff;
        UINT first_sub = 0;
        for (const Readback& part : target.parts) {
            const auto& fp = part.footprint;
            const bool before = diff.any;
            x2w_startup::oracle_compare(static_cast<const unsigned char*>(a) + fp.Offset,
                                        static_cast<const unsigned char*>(b) + fp.Offset, part.row_bytes, part.rows,
                                        fp.Footprint.RowPitch, fp.Footprint.Depth,
                                        fp.Footprint.Width ? static_cast<unsigned>(part.row_bytes / fp.Footprint.Width) : 0u,
                                        part.element, diff);
            if (!before && diff.any) first_sub = part.subresource;
        }
        const D3D12_RANGE nothing{};
        target.normal->Unmap(0, &nothing);
        target.serial->Unmap(0, &nothing);
        char text[400];
        if (diff.any) {
            differs = true;
            std::snprintf(text, sizeof(text),
                          "%s{\"target\":\"%s\",\"format\":%d,\"texels\":%llu,\"differing\":%llu,\"max_ulp\":%u,"
                          "\"first\":{\"subresource\":%u,\"x\":%u,\"y\":%u,\"z\":%u,\"offset\":%llu}}",
                          targets.empty() ? "" : ",", target.name.c_str(), static_cast<int>(target.view),
                          static_cast<unsigned long long>(diff.texels), static_cast<unsigned long long>(diff.differing),
                          diff.max_ulp, first_sub, diff.first_x, diff.first_y, diff.first_z,
                          static_cast<unsigned long long>(diff.first_offset));
        } else {
            std::snprintf(text, sizeof(text), "%s{\"target\":\"%s\",\"format\":%d,\"texels\":%llu,\"differing\":0,\"max_ulp\":%u}",
                          targets.empty() ? "" : ",", target.name.c_str(), static_cast<int>(target.view),
                          static_cast<unsigned long long>(diff.texels), diff.max_ulp);
        }
        targets += text;
    }
    char sample_text[32];
    std::snprintf(sample_text, sizeof(sample_text), ",\"sample\":%u", sample.sample);
    return line + sample_text + ",\"verdict\":\"" + (differs ? "differs" : "equal") + "\",\"targets\":[" + targets + "]}";
}

}  // namespace mesh_oracle

// The draw path's ExecuteIndirect of a slot 57 mesh draw. Returns false when the oracle does not sample this draw: the
// caller then issues the draw itself, exactly as without the oracle. Returns true when it issued the normal draw (and
// the serial one around it). draw_lock held exclusively.
template <class Issue>
bool mesh_oracle_draw(Wrapper* self, ID3D12GraphicsCommandList* list, const Translation& tr,
                      ID3D12PipelineState* pipeline, const Issue& issue)
{
    if (!tr.mesh || !mesh_oracle::selected(tr.key)) return false;
    auto& s = mesh_oracle::state();
    Pass& pass = pass_of(self);
    ID3D12PipelineState* serial = nullptr;
    UINT key_draw = 0, sample = 0;
    u64 mesh_draw = 0;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        mesh_draw = s.mesh_draws++;
        auto& k = s.keys[tr.key];
        key_draw = k.draws++;
        if (k.sampled >= mesh_oracle::config().samples) return false;
        mesh_oracle::load_stages(k, tr);
        std::string reason = k.stages.unavailable;
        if (reason.empty()) {
            const auto found = s.serial_of.find(pipeline);
            if (found != s.serial_of.end()) serial = found->second.Get();
            else reason = "no serial pipeline for this pipeline state";
        }
        if (reason.empty() && !tr.uavs.empty()) reason = "the pipeline writes UAVs (a second draw would write them twice)";
        if (reason.empty() && !pass.open_queries.empty()) reason = "a query is open on the list";
        if (reason.empty() && !pass.target_count && !pass.bound_depth) reason = "no bound target";
        if (!reason.empty()) {
            mesh_oracle::unavailable(s, k, tr.key, key_draw, reason);
            return false;
        }
        sample = k.sampled++;
    }
    ID3D12Device* device = device_of(the_device);
    std::vector<mesh_oracle::Prepared> prepared;
    u64 bytes = 0;
    std::string reason;
    for (UINT t = 0; t < pass.target_count && reason.empty(); ++t) {
        if (!pass.bound_targets[t]) continue;
        char name[8];
        std::snprintf(name, sizeof(name), "rt%u", t);
        prepared.emplace_back();
        reason = mesh_oracle::prepare(device, pass.bound_targets[t], pass.bound_target_ranges[t], pass.target_formats[t],
                                      name, bytes, prepared.back());
    }
    if (reason.empty() && pass.bound_depth) {
        prepared.emplace_back();
        reason = mesh_oracle::prepare(device, pass.bound_depth, pass.bound_depth_view.range, pass.dsv_format, "depth",
                                      bytes, prepared.back());
    }
    if (reason.empty() && !mesh_oracle::retain(self, prepared)) reason = "the list has no allocator to hold the copies";
    if (!reason.empty()) {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto& k = s.keys[tr.key];
        --k.sampled;
        mesh_oracle::unavailable(s, k, tr.key, key_draw, reason);
        return false;
    }

    for (auto& p : prepared) p.saved = mesh_oracle::save_state(pass, p.resource);
    const auto restore_all = [&] {
        for (const auto& p : prepared) mesh_oracle::restore_state(self, list, p.saved);
    };
    // 1. snapshot
    for (auto& p : prepared) {
        require_state(self, p.resource, D3D12_RESOURCE_STATE_COPY_SOURCE);
        mesh_oracle::oracle_barrier(list, p.snapshot.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        mesh_oracle::copy_whole(list, p.snapshot.Get(), p.resource);
        mesh_oracle::oracle_barrier(list, p.snapshot.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    restore_all();
    // 2. the normal draw, its result kept and read back
    issue();
    for (auto& p : prepared) {
        require_state(self, p.resource, D3D12_RESOURCE_STATE_COPY_SOURCE);
        mesh_oracle::oracle_barrier(list, p.result.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        mesh_oracle::copy_whole(list, p.result.Get(), p.resource);
        mesh_oracle::oracle_barrier(list, p.result.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        mesh_oracle::read_back(list, p.resource, p.target.normal.Get(), p.target);
        // 3. back to the inputs for the serial draw
        require_state(self, p.resource, D3D12_RESOURCE_STATE_COPY_DEST);
        mesh_oracle::copy_whole(list, p.resource, p.snapshot.Get());
    }
    restore_all();
    list->SetPipelineState(serial);
    issue();
    for (auto& p : prepared) {
        require_state(self, p.resource, D3D12_RESOURCE_STATE_COPY_SOURCE);
        mesh_oracle::read_back(list, p.resource, p.target.serial.Get(), p.target);
        // 4. the normal result is what the list goes on with
        require_state(self, p.resource, D3D12_RESOURCE_STATE_COPY_DEST);
        mesh_oracle::copy_whole(list, p.resource, p.result.Get());
        // A resubmitted list starts these copies from COMMON again.
        mesh_oracle::oracle_barrier(list, p.snapshot.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        mesh_oracle::oracle_barrier(list, p.result.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    restore_all();
    list->SetPipelineState(pipeline);
    mesh_oracle::Sample pending;
    pending.list = list;
    pending.key = tr.key;
    pending.key_draw = key_draw;
    pending.sample = sample;
    pending.frame = frame_number;
    pending.mesh_draw = mesh_draw;
    for (auto& p : prepared) pending.targets.push_back(std::move(p.target));
    std::lock_guard<std::mutex> lock(s.mutex);
    s.pending.push_back(std::move(pending));
    return true;
}

// After a queue ran `lists`: waits for the queue when one of them carries oracle samples and writes their verdicts.
void mesh_oracle_executed(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists)
{
    if (!mesh_oracle_enabled()) return;
    auto& s = mesh_oracle::state();
    std::vector<mesh_oracle::Sample> ran;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        for (auto it = s.pending.begin(); it != s.pending.end();) {
            if (std::find(lists, lists + count, it->list) == lists + count) {
                ++it;
                continue;
            }
            ran.push_back(std::move(*it));
            it = s.pending.erase(it);
        }
    }
    if (ran.empty()) return;
    x2w_startup::ComPtr<ID3D12Device> device;
    x2w_startup::ComPtr<ID3D12Fence> fence;
    HRESULT hr = queue->GetDevice(IID_PPV_ARGS(&device));
    if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (SUCCEEDED(hr)) hr = queue->Signal(fence.Get(), 1);
    bool done = false;
    if (SUCCEEDED(hr)) {
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (event && SUCCEEDED(fence->SetEventOnCompletion(1, event))) done = WaitForSingleObject(event, 20000) == WAIT_OBJECT_0;
        if (event) CloseHandle(event);
        done = done || fence->GetCompletedValue() >= 1;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    for (const auto& sample : ran) {
        if (!done) {
            mesh_oracle::write_line(s, mesh_oracle::sample_head(sample.key, sample.key_draw, sample.mesh_draw, sample.frame) +
                                           ",\"verdict\":\"oracle_unavailable\",\"reason\":\"queue wait failed\"}");
            continue;
        }
        const std::string line = mesh_oracle::compare(sample);
        mesh_oracle::write_line(s, line);
        log("x2w-oracle mesh %016llx draw %u: %s", sample.key, sample.key_draw,
            line.find("\"differs\"") != std::string::npos ? "differs" :
            line.find("\"equal\"") != std::string::npos ? "equal" : "unavailable");
    }
}

// A list reset or failed Close drops the samples it recorded (their copies never ran).
void mesh_oracle_discard(ID3D12CommandList* list)
{
    if (!mesh_oracle_enabled()) return;
    auto& s = mesh_oracle::state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.pending.erase(std::remove_if(s.pending.begin(), s.pending.end(),
                                   [list](const mesh_oracle::Sample& sample) { return sample.list == list; }),
                    s.pending.end());
}
