// Experimental (docs/startup-runtime.md): draws through translated pipelines.
// Included inside d3d12_xs_device.cpp's anonymous namespace, after the Pass helpers.
//
// A machine-code pipeline is looked up by the hash of its two code blobs and its root signature in
// X2W_SHADER_CACHE: <key>.txt (manifest), <key>.vs.cso and <key>.ps.cso, produced offline by
// out/gamerun/gfx2hlsl.py. A miss writes req_<key>.{a,b,root} there for the next offline pass.
// The translated shaders use their own root signature: [0] 32-bit constants ("entry" values such
// as the base vertex), an optional SRV table and an optional sampler table. Each draw copies what the title bound (vertex streams,
// buffers named by the hardware descriptors in its tables, textures behind view tokens) into an
// upload ring / a private descriptor heap.

struct Translation {
    struct Srv { int kind, source, a, b; };  // kind 0 buffer, 1 2D texture, 2 2D array, 3 3D texture, 4 1D texture; source 0 table(param a, byte b), 1 vertex stream a, 2 root parameter a
    enum UdKind { rect, base_vertex, start_instance, root_constant, stride, mrt_bits, d16_unorm16, desc_word3 };
    struct Ud { int kind, a, b; };
    int state = 0;                           // 1 ready, 2 unavailable
    ULONGLONG retry_manifest_at = 0;         // Only an absent manifest can become ready on a later draw.
    std::vector<Srv> srvs;
    std::vector<Ud> uds;
    x2w_startup::TranslatedSamplers samplers;
    x2w_startup::ExportWriteMasks write_masks;  // manifest 'wmask': channels each export leaves unwritten (#1674)
    x2w_startup::D16ImageSrvs d16_srvs;         // manifest 'd16srv': texture views that must match proven D16 formats (#2023)
    ID3DBlob* vs = nullptr;
    ID3DBlob* ps = nullptr;
    ID3D12RootSignature* root = nullptr;
    std::optional<bool> independent_indirect_entry;
    std::optional<bool> indexed_indirect_entry;
    bool indexed_rect_geometry_attempted = false;
    x2w_startup::ComPtr<ID3DBlob> indexed_rect_geometry;
    int indirect_base_word = -1;
    x2w_startup::ComPtr<ID3D12CommandSignature> indexed_signature;
    std::unordered_map<u64, ID3D12PipelineState*> pipelines;  // by target formats
    u64 key = 0;
    LONG logged_draws = 0, logged_bindings = 0, logged_large = 0;  // diagnostics budget per translation
};

// Per title pipeline: its translation and the blend state found in its state block (if any).
struct BoundPipeline {
    Translation* translation = nullptr;
    const std::vector<RootParameter>* guest_root = nullptr;
    bool has_blend = false;
    D3D12_BLEND_DESC blend{};
    bool has_depth_stencil = false;
    D3D12_DEPTH_STENCIL_DESC depth_stencil{};
    bool depth_from_registers = false;  // X2W_DEPTH_STATE: depth_stencil was decoded from state block A (slot 57)
    bool has_stencil_ref = false;       // ... and its stencil test is on: stencil_ref goes on the list with each draw
    std::uint8_t stencil_ref = 0;
    bool has_rasterizer = false;
    D3D12_RASTERIZER_DESC rasterizer{};
    DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
    std::unordered_map<u64, ID3D12PipelineState*> pipelines;  // by target formats
};

// Looks for a desktop-layout D3D12_BLEND_DESC inside a state block: two flags, then eight render
// target entries whose fields are all inside their enumerations.
bool find_blend(const std::vector<unsigned char>& block, D3D12_BLEND_DESC* out, std::size_t* where)
{
    for (std::size_t at = 0; at + sizeof(D3D12_BLEND_DESC) <= block.size(); at += 4) {
        D3D12_BLEND_DESC d;
        std::memcpy(&d, block.data() + at, sizeof(d));
        bool ok = (d.AlphaToCoverageEnable == 0 || d.AlphaToCoverageEnable == 1) && (d.IndependentBlendEnable == 0 || d.IndependentBlendEnable == 1);
        for (int i = 0; ok && i < 8; ++i) {
            const auto& t = d.RenderTarget[i];
            ok = (t.BlendEnable == 0 || t.BlendEnable == 1) && (t.LogicOpEnable == 0 || t.LogicOpEnable == 1) && t.SrcBlend >= 1 &&
                 t.SrcBlend <= 19 && t.DestBlend >= 1 && t.DestBlend <= 19 && t.BlendOp >= 1 && t.BlendOp <= 5 && t.SrcBlendAlpha >= 1 &&
                 t.SrcBlendAlpha <= 19 && t.DestBlendAlpha >= 1 && t.DestBlendAlpha <= 19 && t.BlendOpAlpha >= 1 && t.BlendOpAlpha <= 5 &&
                 static_cast<unsigned>(t.LogicOp) <= 15 && t.RenderTargetWriteMask <= 15;
        }
        if (ok) {
            *out = d;
            *where = at;
            return true;
        }
    }
    return false;
}

SRWLOCK draw_lock = SRWLOCK_INIT;
std::unordered_map<const Wrapper*, BoundPipeline*>& translation_by_pipeline() { static std::unordered_map<const Wrapper*, BoundPipeline*> m; return m; }
std::unordered_map<u64, Translation*>& translation_by_key() { static std::unordered_map<u64, Translation*> m; return m; }
LONG translated_draws, untranslated_draws, translation_ready, translation_missing, null_bindings;
LONG null_by_cause[x2w_local::null_cause_count];  // null_bindings split by cause; the parts add up to it

void count_null_binding(x2w_local::NullCause cause)
{
    if (cause == x2w_local::NullCause::none) return;
    InterlockedIncrement(&null_bindings);
    InterlockedIncrement(&null_by_cause[static_cast<int>(cause)]);
}
LONG frame_number;

u64 fnv_bytes(const void* data, std::size_t bytes, u64 hash = 1469598103934665603ull)
{
    for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ static_cast<const unsigned char*>(data)[i]) * 1099511628211ull;
    return hash;
}

bool cache_path(wchar_t* out, const wchar_t* prefix, u64 key, const wchar_t* extension)
{
    wchar_t dir[MAX_PATH];
    const DWORD n = GetEnvironmentVariableW(L"X2W_SHADER_CACHE", dir, MAX_PATH);
    if (!n || n >= MAX_PATH) return false;
    swprintf(out, MAX_PATH, L"%s\\%s%016llx%s", dir, prefix, key, extension);
    return true;
}

bool read_text(const wchar_t* path, std::string& out)
{
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    char buffer[8192];
    DWORD got = 0;
    while (ReadFile(file, buffer, sizeof(buffer), &got, nullptr) && got) out.append(buffer, got);
    CloseHandle(file);
    return true;
}

// Publish each input atomically. The callers write the request's ready marker last.
bool publish_shader_request(u64 key, const wchar_t* ext, const void* data, std::size_t bytes)
{
    wchar_t path[MAX_PATH], temporary[MAX_PATH], suffix[32];
    if (!cache_path(path, L"req_", key, ext)) return false;
    if (GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES) return true;
    swprintf(suffix, std::size(suffix), L"%s.%lu.tmp", ext, GetCurrentProcessId());
    if (!cache_path(temporary, L"req_", key, suffix)) return false;
    HANDLE file = CreateFileW(temporary, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    DWORD written = 0;
    const bool complete = WriteFile(file, data, static_cast<DWORD>(bytes), &written, nullptr) && written == bytes;
    CloseHandle(file);
    if (complete && MoveFileExW(temporary, path, 0)) return true;
    DeleteFileW(temporary);
    return GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES;
}

void load_translation(Translation& tr, const std::string& rtv_request = {})
{
    tr.state = 2;
    wchar_t path[MAX_PATH];
    std::string manifest;
    if (!cache_path(path, L"", tr.key, L".txt")) return;
    if (!read_text(path, manifest)) {
        if (!tr.retry_manifest_at) {
            InterlockedIncrement(&translation_missing);
            static LONG missing;
            if (rtv_request.empty() && first_calls(missing, 12))
                log("x2w-draw no translation for pipeline %016llx", tr.key);
        }
        tr.retry_manifest_at = GetTickCount64() + 1000;
        return;
    }
    // The writer publishes this marker last. A malformed published result stays rejected.
    const bool previously_missing = tr.retry_manifest_at != 0;
    tr.retry_manifest_at = 0;
    bool matched_rtv = false;
    for (std::size_t at = 0; at < manifest.size();) {
        std::size_t end = manifest.find('\n', at);
        if (end == std::string::npos) end = manifest.size();
        std::string line = manifest.substr(at, end - at);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        at = end + 1;
        int i = 0, x = 0, y = 0;
        char kind[16] = "", source[16] = "";
        if (line.compare(0, 5, "rtv1 ") == 0) {
            if (matched_rtv || line != rtv_request) {
                log("x2w-draw RTV manifest mismatch for %016llx", tr.key);
                return;
            }
            matched_rtv = true;
        } else if (line.compare(0, 13, "x2w-manifest ") == 0 && line != "x2w-manifest 1") {
            // Version 2 adds `uav` lines for vertex-stage stores. This runtime does not bind them,
            // so it must not draw a translation whose stores it would silently skip (#1723).
            log("x2w-draw unsupported manifest version for %016llx", tr.key);
            return;
        } else if (line.compare(0, 4, "uav ") == 0) {
            log("x2w-draw unsupported uav line for %016llx", tr.key);
            return;
        } else if (std::sscanf(line.c_str(), "udsrc %d %15s %d %d", &i, source, &x, &y) >= 2) {
            Translation::Ud ud{-1, x, y};
            if (!std::strcmp(source, "rect")) ud.kind = Translation::rect;
            else if (!std::strcmp(source, "basevertex")) ud.kind = Translation::base_vertex;
            else if (!std::strcmp(source, "startinstance")) ud.kind = Translation::start_instance;
            else if (!std::strcmp(source, "rootconst")) ud.kind = Translation::root_constant;
            else if (!std::strcmp(source, "stride")) ud.kind = Translation::stride;
            else if (!std::strcmp(source, "d16unorm16") && x >= 0 && x < 32) ud.kind = Translation::d16_unorm16;
            else if (!std::strcmp(source, "descword3")) ud.kind = Translation::desc_word3;
            else if (!std::strcmp(source, "mrtbits") && !rtv_request.empty()) ud.kind = Translation::mrt_bits;
            if (static_cast<std::size_t>(i) == tr.uds.size() && ud.kind >= 0) tr.uds.push_back(ud);
        } else if (std::sscanf(line.c_str(), "srv %d %15s %15s %d %d", &i, kind, source, &x, &y) >= 4) {
            const int texture_kind = !std::strcmp(kind, "tex2d") ? 1 : !std::strcmp(kind, "tex2da") ? 2 :
                                     !std::strcmp(kind, "tex3d") ? 3 : !std::strcmp(kind, "tex1d") ? 4 : 0;
            Translation::Srv srv{texture_kind, !std::strcmp(source, "vb") ? 1 : !std::strcmp(source, "root") ? 2 : 0, x, y};
            if (static_cast<std::size_t>(i) == tr.srvs.size()) tr.srvs.push_back(srv);
        } else if (line.compare(0, 6, "wmask ") == 0) {
            if (!tr.write_masks.read_manifest_line(line)) {
                log("x2w-draw invalid write mask manifest line for %016llx: %s", tr.key, line.c_str());
                return;
            }
        } else if (line.compare(0, 7, "d16srv ") == 0) {
            if (!tr.d16_srvs.read_manifest_line(line)) {
                log("x2w-draw invalid d16srv manifest line for %016llx: %s", tr.key, line.c_str());
                return;
            }
        } else if (line.compare(0, 4, "uav ") == 0) {
            // tools/startup/gfx2hlsl.py lowers a pixel-stage image_store to a u-register store, but the
            // translated root signature has no UAV range and no view is made from the title's descriptor yet.
            log("x2w-draw pipeline %016llx stores to an image (%s); pixel-stage UAVs are not bound yet, translation not used",
                tr.key, line.c_str());
            return;
        } else if (!tr.samplers.read_manifest_line(line.c_str())) {
            log("x2w-draw invalid sampler heap manifest for %016llx", tr.key);
            return;
        }
    }
    if (matched_rtv != !rtv_request.empty()) {
        log("x2w-draw RTV manifest missing for %016llx", tr.key);
        return;
    }
    if (tr.uds.empty() || tr.uds.size() > 32 || tr.srvs.size() > 32 || !tr.d16_srvs.valid_for_srvs(tr.srvs)) return;
    for (const Translation::Ud& ud : tr.uds) {
        if (ud.kind == Translation::d16_unorm16 &&
            (ud.a < 0 || static_cast<std::size_t>(ud.a) >= tr.srvs.size() || !tr.d16_srvs.requires_exact(static_cast<unsigned>(ud.a))))
            return;
    }
    if (!cache_path(path, L"", tr.key, L".vs.cso") || FAILED(D3DReadFileToBlob(path, &tr.vs))) return;
    if (!cache_path(path, L"", tr.key, L".ps.cso") || FAILED(D3DReadFileToBlob(path, &tr.ps))) return;
    const HRESULT hr = x2w_startup::create_translated_root(device_of(the_device), static_cast<UINT>(tr.uds.size()),
        static_cast<UINT>(tr.srvs.size()), tr.samplers, &tr.root);
    if (FAILED(hr)) {
        log("x2w-draw root signature for %016llx failed 0x%08lx", tr.key, static_cast<unsigned long>(hr));
        return;
    }
    tr.state = 1;
    if (previously_missing) InterlockedDecrement(&translation_missing);
    InterlockedIncrement(&translation_ready);
    log("x2w-draw pipeline %016llx translated: %zu entry values, %zu resources, %zu static samplers, sampler_heap=%u",
        tr.key, tr.uds.size(), tr.srvs.size(), tr.samplers.statics.size(), tr.samplers.heap ? 1u : 0u);
}

// draw_lock held. Both pipeline forms use the same request writer and shader cache.
Translation* cached_translation(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes,
                                const std::vector<unsigned char>& root_blob, const std::string& layout = {})
{
    // Keep pre-fix desktop bytecode and manifests out of this ABI's cache.
    // tools/startup/gfx2hlsl.py hashes the same bytes, without a trailing NUL.
    constexpr char revision[] = "x2w-startup-shader-14";
    const u64 seed = fnv_bytes(revision, sizeof(revision) - 1);
    u64 key = fnv_bytes(root_blob.data(), root_blob.size(), fnv_bytes(b, b_bytes, fnv_bytes(a, a_bytes, seed)));
    // A stream's input layout (stream_input_layout) feeds the vertex fetch, so it is part of the key.
    if (!layout.empty()) key = fnv_bytes(layout.data(), layout.size(), key);
    auto& entry = translation_by_key()[key];
    if (!entry) {
        entry = new Translation;
        entry->key = key;
        // A legacy cache hit still needs source blobs for late RTV specialization.
        if (publish_shader_request(key, L".b", b, b_bytes) &&
            publish_shader_request(key, L".root", root_blob.data(), root_blob.size()) &&
            (layout.empty() || publish_shader_request(key, L".ia", layout.data(), layout.size())))
            publish_shader_request(key, L".a", a, a_bytes);
        load_translation(*entry);
    }
    return entry;
}

LONG compute_requests_published, compute_requests_present, compute_requests_failed, compute_blob_pointer_words;

// Device slot 67 (console compute program), 12 64-bit words: 0 (root signature), 1-2 and 3-4 (two state
// blocks and sizes, a size of 0 meaning absent), 5-6 (program code and size), 7-8 (a second blob and size),
// 9 and 10 (pointers whose targets are not decoded) and 11 (a value). The request carries .desc (the 12
// words with the seven pointer words always zeroed, so no process address reaches a cache file or the key),
// .sa, .sb, .data and .root, then .c67 (the code) last as the ready marker, for an offline compute stage
// (tools/startup/gfx2hlsl.py). Any declared part that cannot be read refuses the whole request before a
// file is written. The pipeline stays
// a placeholder. Like the graphics path, a key is published once per cache whether or not a result exists.
constexpr u64 compute_code_limit = 1u << 20, compute_data_limit = 1u << 20, compute_state_limit = 0x1000;

// The key frames every part with its length, so moving bytes from one part to the next changes it.
u64 fnv_part(const std::vector<unsigned char>& part, u64 key)
{
    const u64 bytes = part.size();
    return fnv_bytes(part.data(), part.size(), fnv_bytes(&bytes, sizeof(bytes), key));
}

// A fault-safe copy of title memory: the parts are read once, then hashed, scanned and written from the copy,
// so a title write in between cannot leave a file that does not match its key.
bool copy_title_bytes(void* to, const void* from, std::size_t bytes) noexcept
{
    __try {
        std::memcpy(to, from, bytes);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return true;
}

// Aligned qwords of a published blob that name committed memory of this process (one VirtualQuery each, so
// no fault is taken): such a value makes the key run-dependent. They are counted and published, not refused
// (shortcut: hash-like values can name memory by chance; refuse once a decoder names the pointer fields).
LONG pointer_like_qwords(const std::vector<unsigned char>& blob)
{
    LONG found = 0;
    for (std::size_t at = 0; at + 8 <= blob.size(); at += 8) {
        u64 value;
        std::memcpy(&value, blob.data() + at, sizeof(value));
        MEMORY_BASIC_INFORMATION info{};
        if (value >= 0x10000 && value < 0x800000000000ull &&
            VirtualQuery(reinterpret_cast<const void*>(value), &info, sizeof(info)) == sizeof(info) && info.State == MEM_COMMIT)
            ++found;
    }
    return found;
}

SRWLOCK compute_request_lock = SRWLOCK_INIT;
std::unordered_set<u64>& compute_keys_seen() { static std::unordered_set<u64> s; return s; }

void publish_compute_request(const void* desc)
{
    wchar_t marker[MAX_PATH];
    if (!render_mode() || !cache_path(marker, L"req_", 0, L".c67")) return;  // publishing disabled
    static LONG logged;
    const char* refusal = nullptr;
    u64 words[12]{};
    std::vector<unsigned char> root_blob;
    if (!desc || !x2w_local::readable(desc, sizeof(words)) || !copy_title_bytes(words, desc, sizeof(words)))
        refusal = "description unreadable";
    else {
        if (Wrapper* root = as_wrapper(reinterpret_cast<const void*>(words[0]))) {
            AcquireSRWLockShared(&note_lock);
            const auto found = root_blobs().find(root);
            if (found != root_blobs().end()) root_blob = found->second;
            ReleaseSRWLockShared(&note_lock);
        }
        if (root_blob.empty()) refusal = "no root signature blob";
    }
    struct Part { int pointer; u64 limit; bool required; const char *zero, *over, *unreadable; };
    static constexpr Part parts[4] = {
        {5, compute_code_limit, true, "code size 0", "code size over 1 MiB", "code unreadable"},
        {7, compute_data_limit, true, "data size 0", "data size over 1 MiB", "data unreadable"},
        {1, compute_state_limit, false, nullptr, "state block A size over 4 KiB", "state block A unreadable"},
        {3, compute_state_limit, false, nullptr, "state block B size over 4 KiB", "state block B unreadable"},
    };
    std::vector<unsigned char> copies[4];  // code, data, state A, state B
    for (int i = 0; i < 4 && !refusal; ++i) {
        const Part& p = parts[i];
        const u64 bytes = words[p.pointer + 1];
        const auto* at = reinterpret_cast<const void*>(words[p.pointer]);
        if (!bytes) refusal = p.required ? p.zero : nullptr;
        else if (bytes > p.limit) refusal = p.over;
        else {
            copies[i].resize(static_cast<std::size_t>(bytes));
            if (!x2w_local::readable(at, copies[i].size()) || !copy_title_bytes(copies[i].data(), at, copies[i].size()))
                refusal = p.unreadable;
        }
    }
    if (refusal) {
        InterlockedIncrement(&compute_requests_failed);
        if (first_calls(logged, 12)) log("x2w-draw compute request not written: %s", refusal);
        return;
    }
    const auto& [code, data, state_a, state_b] = copies;
    // Pointer words never reach .desc or the key; 9 and 10 are not followed (shortcut: two descriptions that
    // differ only in what they point to share a key; publish those targets once they are decoded).
    std::vector<unsigned char> fields(sizeof(words));
    words[0] = words[1] = words[3] = words[5] = words[7] = words[9] = words[10] = 0;
    std::memcpy(fields.data(), words, sizeof(words));
    constexpr char compute_revision[] = "x2w-startup-shader-14cs";  // tools/startup/gfx2hlsl.py compute_key
    u64 key = fnv_bytes(compute_revision, sizeof(compute_revision) - 1);
    const std::vector<unsigned char>* order[] = {&code, &data, &root_blob, &fields, &state_a, &state_b};
    for (const auto* part : order) key = fnv_part(*part, key);
    // A dedicated lock and an in-memory set: a key seen in this process costs no file access, and draws
    // under draw_lock are not held up by request writes.
    const auto write = [key](const wchar_t* ext, const std::vector<unsigned char>& part) {
        static const unsigned char none = 0;  // a valid buffer for a 0-byte part
        return publish_shader_request(key, ext, part.empty() ? &none : part.data(), part.size());
    };
    AcquireSRWLockExclusive(&compute_request_lock);
    bool present = compute_keys_seen().count(key) != 0;
    bool written = present;
    if (!present) {
        present = cache_path(marker, L"req_", key, L".c67") && GetFileAttributesW(marker) != INVALID_FILE_ATTRIBUTES;
        // Parts left by an interrupted earlier write are kept; the marker comes last, so the request completes here.
        written = present || (write(L".desc", fields) && write(L".sa", state_a) && write(L".sb", state_b) &&
                              write(L".data", data) && write(L".root", root_blob) && write(L".c67", code));
        if (written) compute_keys_seen().insert(key);
    }
    ReleaseSRWLockExclusive(&compute_request_lock);
    const LONG pointers = written && !present ? pointer_like_qwords(data) + pointer_like_qwords(state_a) + pointer_like_qwords(state_b) : 0;
    if (pointers) InterlockedAdd(&compute_blob_pointer_words, pointers);
    InterlockedIncrement(present ? &compute_requests_present : written ? &compute_requests_published : &compute_requests_failed);
    if (first_calls(logged, 12))
        log("x2w-draw compute request %016llx %s (code %llu bytes, data %llu bytes, state blocks %llu + %llu bytes, "
            "pointer-like blob qwords %ld)", key, present ? "already present" : written ? "published" : "not written",
            words[6], words[8], words[2], words[4], pointers);
}

// Slot 57 fixed descriptors consumed by a decoded DispatchMesh signature.
// The vertex span must be absent. A null/zero PS span is valid for depth-only work.
void request_slot57_mesh(Wrapper* pipeline, const std::vector<RootParameter>* bound_root)
{
    wchar_t marker[MAX_PATH];
    if (!bound_root || !cache_path(marker, L"req_", 0, L".m57")) return;
    const PipelineInfo* info = pipeline_info(pipeline);
    for (int depth = 0; info && info->path == 55 && info->base && depth < 8; ++depth)
        info = pipeline_info(info->base);
    constexpr std::size_t fields = 15;  // through the MS length at +0x70
    if (!info || info->path != 57 || info->desc.size() < fields * sizeof(u64)) return;
    u64 words[fields];
    std::memcpy(words, info->desc.data(), sizeof(words));
    if (words[9] || words[10]) return;  // VS at +0x48
    const Wrapper* root = as_wrapper(reinterpret_cast<const void*>(words[0]));
    if (!root || root_info(root) != bound_root) return;
    std::vector<unsigned char> root_blob;
    {
        AcquireSRWLockShared(&note_lock);
        if (const auto found = root_blobs().find(root); found != root_blobs().end()) root_blob = found->second;
        ReleaseSRWLockShared(&note_lock);
    }
    if (root_blob.empty()) return;
    std::vector<unsigned char> mesh, pixel;
    const auto copy_stage = [&words](int field, bool required, std::vector<unsigned char>& out) {
        const auto* pointer = reinterpret_cast<const void*>(words[field]);
        const auto bytes = words[field + 1];
        if (!bytes) return !required && !pointer;
        if (bytes > (1u << 20) || !x2w_local::readable(pointer, bytes)) return false;
        out.resize(static_cast<std::size_t>(bytes));
        return copy_title_bytes(out.data(), pointer, out.size());
    };
    if (!copy_stage(13, true, mesh) || !copy_stage(5, false, pixel)) return;
    constexpr char revision[] = "x2w-startup-mesh-57-1";
    const u64 key = fnv_part(root_blob, fnv_part(pixel, fnv_part(mesh, fnv_bytes(revision, sizeof(revision) - 1))));
    static SRWLOCK request_lock = SRWLOCK_INIT;
    static std::unordered_set<u64> written;
    x2w_startup::ExclusiveNoteLock guard(request_lock);
    if (written.count(key)) return;
    const unsigned char none = 0;
    if (publish_shader_request(key, L".ps57", pixel.empty() ? &none : pixel.data(), pixel.size()) &&
        publish_shader_request(key, L".root", root_blob.data(), root_blob.size()) &&
        publish_shader_request(key, L".m57", mesh.data(), mesh.size())) {
        written.insert(key);
        static LONG logged;
        if (first_calls(logged, 12))
            log("x2w-draw mesh request %016llx: slot57, MS %zu bytes, PS %zu bytes",
                key, mesh.size(), pixel.size());
    }
}

// The INPUT_LAYOUT subobject as the request text tools/startup/gfx2hlsl.py reads: "ia1 <count>", then one
// "<semantic> <index> <format> <slot> <offset> <class> <step>" line per element. False when it cannot be read.
bool stream_input_layout(const unsigned char* subobject, std::string& text)
{
    D3D12_INPUT_LAYOUT_DESC layout;
    std::memcpy(&layout, subobject + 8, sizeof(layout));
    if (!layout.NumElements) return true;
    if (layout.NumElements > D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT ||
        !x2w_local::readable(layout.pInputElementDescs, sizeof(D3D12_INPUT_ELEMENT_DESC) * layout.NumElements)) return false;
    text = "ia1 " + std::to_string(layout.NumElements) + "\n";
    std::string shown;
    for (UINT i = 0; i < layout.NumElements; ++i) {
        const D3D12_INPUT_ELEMENT_DESC& e = layout.pInputElementDescs[i];
        std::size_t length = 0;
        while (length < 64 && x2w_local::readable(e.SemanticName + length, 1) && e.SemanticName[length]) {
            const char c = e.SemanticName[length];
            if (!(std::isalpha(static_cast<unsigned char>(c)) || c == '_' || (length && std::isdigit(static_cast<unsigned char>(c))))) return false;
            ++length;
        }
        if (!length || length == 64) return false;
        char line[160];
        std::snprintf(line, sizeof(line), "%.*s %u %u %u %u %u %u\n", static_cast<int>(length), e.SemanticName, e.SemanticIndex,
                      static_cast<unsigned>(e.Format), e.InputSlot, e.AlignedByteOffset, static_cast<unsigned>(e.InputSlotClass),
                      e.InstanceDataStepRate);
        text += line;
        std::snprintf(line, sizeof(line), "%s%u/%u/%u/%u", i ? " " : "", static_cast<unsigned>(e.Format), e.InputSlot,
                      e.AlignedByteOffset, static_cast<unsigned>(e.InputSlotClass));
        shown += line;
    }
    static LONG logged;
    if (first_calls(logged, 24))
        log("x2w-draw stream input layout: %u elements (format/slot/offset/class) %s", layout.NumElements, shown.c_str());
    return true;
}

// RASTERIZER1 stores DepthBias as a float. The desktop PSO takes an int, so only an integral value
// in the int range is representable.
bool integral_depth_bias(float bias, INT& out)
{
    if (!(bias == bias)) return false;
    const double value = static_cast<double>(bias);
    if (value < -2147483648.0 || value > 2147483647.0) return false;
    const INT integral = static_cast<INT>(value);
    if (static_cast<double>(integral) != value) return false;
    out = integral;
    return true;
}

// Resolve the shader pointers and wrapped root while the caller still owns the stream.
Translation* stream_translation(const D3D12_PIPELINE_STATE_STREAM_DESC& desc, D3D12_BLEND_DESC& blend, bool& has_blend,
                                D3D12_DEPTH_STENCIL_DESC& depth_stencil, bool& has_depth_stencil, DXGI_FORMAT& dsv_format,
                                D3D12_RASTERIZER_DESC& rasterizer, bool& has_rasterizer)
{
    // RASTERIZER1 (type 27) is the same width as RASTERIZER. The width comes from the table.
    static_assert(stream_sizes[D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER] == sizeof(unsigned) + sizeof(D3D12_RASTERIZER_DESC));
    const auto* stream = static_cast<const unsigned char*>(desc.pPipelineStateSubobjectStream);
    if (!desc.SizeInBytes || desc.SizeInBytes > 0x2000 || desc.SizeInBytes % 8 ||
        !x2w_local::readable(stream, desc.SizeInBytes)) return nullptr;

    const Wrapper* root = nullptr;
    D3D12_SHADER_BYTECODE stages[2]{};
    std::string layout;
    unsigned seen = 0;
    for (std::size_t at = 0; at < desc.SizeInBytes;) {
        unsigned type;
        std::memcpy(&type, stream + at, sizeof(type));
        // Type 27 is RASTERIZER1. It is outside the table; its width is the RASTERIZER entry.
        const bool rasterizer1 = type == 27;
        if (!rasterizer1 && type >= std::size(stream_sizes)) return nullptr;
        if (type >= 32) return nullptr;
        const std::size_t size = rasterizer1 ? stream_sizes[D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER] : stream_sizes[type];
        if (size > desc.SizeInBytes - at || (seen & (1u << type))) return nullptr;
        seen |= 1u << type;
        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE) {
            std::memcpy(&root, stream + at + 8, sizeof(root));
        } else if (type >= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS && type <= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS) {
            D3D12_SHADER_BYTECODE shader;
            std::memcpy(&shader, stream + at + 8, sizeof(shader));
            if (shader.pShaderBytecode || shader.BytecodeLength) {
                if (type > D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS || shader.BytecodeLength < 32 ||
                    shader.BytecodeLength > (1u << 20) || !x2w_local::readable(shader.pShaderBytecode, shader.BytecodeLength) ||
                    std::memcmp(shader.pShaderBytecode, "DXBC", 4)) return nullptr;
                stages[type - D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS] = shader;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND) {
            std::memcpy(&blend, stream + at + sizeof(type), sizeof(blend));
            has_blend = true;
        } else if (type == 11 || type == 21) {  // DEPTH_STENCIL (56B) / DEPTH_STENCIL1 (64B)
            std::memcpy(&depth_stencil, stream + at + sizeof(type), sizeof(depth_stencil));
            has_depth_stencil = true;
            if (type == 21) {  // DEPTH_STENCIL1 appends DepthBoundsTestEnable; it has no translated equivalent
                BOOL bounds = FALSE;
                std::memcpy(&bounds, stream + at + sizeof(type) + sizeof(depth_stencil), sizeof(bounds));
                if (bounds) return nullptr;
            }
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT) {
            // Unreadable: translate without it; gfx2hlsl refuses a vertex shader that needs it.
            if (!stream_input_layout(stream + at, layout)) {
                layout.clear();
                log("x2w-draw stream input layout unreadable; requesting the translation without it");
            }
        } else if (type == 16) {  // DEPTH_STENCIL_FORMAT (8B)
            DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
            std::memcpy(&fmt, stream + at + sizeof(type), sizeof(fmt));
            dsv_format = desktop_format(fmt);
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER || type == 27) {
            // Both encodings describe the same rasterizer state.
            if (has_rasterizer) return nullptr;
            // RASTERIZER (48B from stream_sizes) / RASTERIZER1 (same width, float depth bias).
            D3D12_RASTERIZER_DESC parsed{};
            if (sizeof(type) + sizeof(parsed) > size) return nullptr;
            std::memcpy(&parsed, stream + at + sizeof(type), sizeof(parsed));
            if (type == 27) {
                float bias = 0;
                std::memcpy(&bias, &parsed.DepthBias, sizeof(bias));
                INT integral = 0;
                if (!integral_depth_bias(bias, integral)) return nullptr;
                parsed.DepthBias = integral;
            }
            rasterizer = parsed;
            has_rasterizer = true;
        }
        at += size;
    }
    if (!root || !stages[0].pShaderBytecode || !stages[1].pShaderBytecode) return nullptr;
    std::vector<unsigned char> root_blob;
    AcquireSRWLockShared(&note_lock);
    const auto found = root_blobs().find(root);
    if (found != root_blobs().end()) root_blob = found->second;
    ReleaseSRWLockShared(&note_lock);
    if (root_blob.empty()) return nullptr;
    AcquireSRWLockExclusive(&draw_lock);
    auto* result = cached_translation(stages[0].pShaderBytecode, stages[0].BytecodeLength,
                                      stages[1].pShaderBytecode, stages[1].BytecodeLength, root_blob, layout);
    ReleaseSRWLockExclusive(&draw_lock);
    return result;
}

// draw_lock held.
BoundPipeline* translation_for(Wrapper* pipeline)
{
    auto& by_pipeline = translation_by_pipeline();
    const auto known = by_pipeline.find(pipeline);
    if (known != by_pipeline.end()) return known->second;
    auto* bound = new BoundPipeline;
    Translation*& result = bound->translation;
    const PipelineInfo* info = pipeline_info(pipeline);
    const PipelineInfo* own = info;
    {
        // The pipeline's own description first (a derived pipeline overrides state), then its base's state block.
        std::size_t where = 0;
        const char* from = "none";
        const char* refusal = nullptr;
        unsigned refused_targets = 0;
        for (int depth = 0; info && info->path == 55 && info->base && depth < 8; ++depth) {
            if (!bound->has_blend && x2w_derived_blend::last_blend(info->desc.data(), info->desc.size(), bound->blend, where)) {
                from = "derived blend patch";
                bound->has_blend = true;
            }
            info = pipeline_info(info->base);
        }
        if (!bound->has_blend && info && info->has_stream_blend) {
            from = "stream blend subobject";
            bound->blend = info->stream_blend;
        } else if (!bound->has_blend && info && info->path == 57 &&
                   x2w_startup::state_block_blend(info->state_a.data(), info->state_a.size(), false, &bound->blend, &refusal, &refused_targets)) {
            // The colour-blend register words of a slot 57 pipeline (state_block_blend.h, #1216).
            from = "colour-blend registers";
            where = x2w_startup::state_block_blend_offset;
        } else if (!bound->has_blend && info && find_blend(info->state_a, &bound->blend, &where)) from = "state block A";
        else if (!bound->has_blend && info && find_blend(info->state_b, &bound->blend, &where)) from = "state block B";
        bound->has_blend = from[0] != 'n';
        static LONG logged;
        if (first_calls(logged, 24)) {
            const auto& t = bound->blend.RenderTarget[0];
            log("x2w-draw pipeline %p (creation slot %d, base slot %d): blend from %s at +0x%zx: enable %d src %d dest %d op %d / alpha %d %d %d "
                "mask 0x%x; blocks %zu + %zu bytes", static_cast<void*>(pipeline), own ? own->path : 0, info ? info->path : 0, from, where,
                t.BlendEnable, t.SrcBlend, t.DestBlend, t.BlendOp, t.SrcBlendAlpha, t.DestBlendAlpha, t.BlendOpAlpha, t.RenderTargetWriteMask,
                info ? info->state_a.size() : 0, info ? info->state_b.size() : 0);
            if (refusal || refused_targets)
                log("x2w-draw   colour-blend registers not used: %s; refused targets 0x%x", refusal ? refusal : "refused", refused_targets);
            if (info && !bound->has_blend) {
                char hex[3 * 0x1a0 + 1];
                hex_text(info->state_a.data(), info->state_a.size() < 0x1a0 ? info->state_a.size() : 0x1a0, hex, sizeof(hex));
                log("x2w-draw   state A: %s", hex);
                hex_text(info->state_b.data(), info->state_b.size() < 0x1a0 ? info->state_b.size() : 0x1a0, hex, sizeof(hex));
                log("x2w-draw   state B: %s", hex);
                if (own != info && own) {
                    hex_text(own->desc.data(), own->desc.size() < 0x100 ? own->desc.size() : 0x100, hex, sizeof(hex));
                    log("x2w-draw   derived: %s", hex);
                }
            }
        }
    }
    if (info && info->path == 86) {
        result = info->stream_translation;
        if (info->has_stream_depth_stencil) {
            bound->depth_stencil = info->stream_depth_stencil;
            bound->has_depth_stencil = true;
        }
        if (info->has_stream_rasterizer) {
            bound->rasterizer = info->stream_rasterizer;
            bound->has_rasterizer = true;
        }
        bound->dsv_format = info->stream_dsv_format;
    }
    // X2W_DEPTH_STATE=1 (#1999, item A): the depth-stencil registers of a slot 57 pipeline's state block A. A slot 57
    // pipeline has no desktop DEPTH_STENCIL subobject, so has_depth_stencil can only be set here; the test keeps any
    // desktop state that was found first. A block the decoder refuses leaves the desktop default (depth and stencil off).
    if (info && info->path == 57 && !bound->has_depth_stencil && x2w_local::env_on("X2W_DEPTH_STATE")) {
        const auto chosen = x2w_pipeline_state::select_depth_stencil(true, info->state_a.data(), info->state_a.size());
        static LONG logged_applied, logged_refused;
        if (chosen.applied) {
            bound->depth_stencil = chosen.desc;
            bound->has_depth_stencil = bound->depth_from_registers = true;
            bound->has_stencil_ref = chosen.has_stencil_ref;
            bound->stencil_ref = chosen.stencil_ref;
        }
        if (first_calls(chosen.applied ? logged_applied : logged_refused, 24)) {
            if (chosen.applied)
                log("x2w-draw pipeline %p: depth-stencil from state block A: depth %d func %d write %d, stencil %d reference 0x%x%s%s",
                    static_cast<void*>(pipeline), chosen.desc.DepthEnable, chosen.desc.DepthFunc, chosen.desc.DepthWriteMask,
                    chosen.desc.StencilEnable, chosen.stencil_ref, chosen.depth_bounds ? "; depth bounds not applied" : "",
                    chosen.masks_differ ? "; back-face masks or reference differ, front used" : "");
            else
                log("x2w-draw pipeline %p: depth-stencil registers not used (%s); keeping the desktop state", static_cast<void*>(pipeline),
                    chosen.reason);
        }
    }
    if (info && info->path == 57 && info->desc.size() >= 0x60) {
        u64 words[12];
        std::memcpy(words, info->desc.data(), sizeof(words));
        Wrapper* root = as_wrapper(reinterpret_cast<const void*>(words[0]));
        bound->guest_root = root ? root_info(root) : nullptr;
        std::vector<unsigned char> root_blob;
        if (root) {
            AcquireSRWLockShared(&note_lock);
            const auto found = root_blobs().find(root);
            if (found != root_blobs().end()) root_blob = found->second;
            ReleaseSRWLockShared(&note_lock);
        }
        const auto* a = reinterpret_cast<const void*>(words[5]);
        const auto* b = reinterpret_cast<const void*>(words[9]);
        const std::size_t a_bytes = static_cast<std::size_t>(words[6]), b_bytes = static_cast<std::size_t>(words[10]);
        if (!root_blob.empty() && a_bytes && b_bytes && a_bytes <= (1u << 20) && b_bytes <= (1u << 20) && x2w_local::readable(a, a_bytes) &&
            x2w_local::readable(b, b_bytes)) {
            result = cached_translation(a, a_bytes, b, b_bytes, root_blob);
        }
    }
    by_pipeline[pipeline] = bound;
    return bound;
}

// The request translation_for writes on a direct draw, for a pipeline that slot 61 refuses. Only direct draws poll
// for the manifest afterwards. Count first lookups separately from published request files.
LONG indirect_first_lookups;
void request_translation(Wrapper* pipeline)
{
    AcquireSRWLockExclusive(&draw_lock);
    const bool known = translation_by_pipeline().count(pipeline) != 0;
    translation_for(pipeline);
    ReleaseSRWLockExclusive(&draw_lock);
    if (!known) InterlockedIncrement(&indirect_first_lookups);
    static LONG lookups;
    if (first_calls(lookups, 16))
        log("x2w-render ExecuteIndirect looks up translation for pipeline %p (%s)", static_cast<void*>(pipeline),
            known ? "already known" : "new");
}

bool needs_rtv_variant(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT:
    case DXGI_FORMAT_R32G32B32_UINT: case DXGI_FORMAT_R32G32B32_SINT:
    case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R8G8_UINT: case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R8_UINT: case DXGI_FORMAT_R8_SINT:
        return true;
    default:
        return false;
    }
}

Translation* translation_for_targets(Translation* base, const Pass& pass)
{
    if (!base) return nullptr;
    if (std::none_of(pass.target_formats, pass.target_formats + pass.target_count, needs_rtv_variant)) {
        if (base->retry_manifest_at && GetTickCount64() >= base->retry_manifest_at)
            load_translation(*base);
        return base;
    }
    std::string request = "rtv1 " + std::to_string(base->key);
    for (UINT i = 0; i < pass.target_count; ++i)
        request += " " + std::to_string(static_cast<unsigned>(pass.target_formats[i]));
    const u64 key = fnv_bytes(request.data(), request.size());
    auto& entry = translation_by_key()[key];
    if (!entry) {
        entry = new Translation;
        entry->key = key;
        load_translation(*entry, request);
        if (entry->state != 1) {
            publish_shader_request(key, L".rtv", request.data(), request.size());
            log("x2w-draw no RTV translation for %016llx (%s)", key, request.c_str());
        }
    } else if (entry->retry_manifest_at && GetTickCount64() >= entry->retry_manifest_at) {
        load_translation(*entry, request);
    }
    return entry;
}

// The depth format a translated draw uses: the pass's own view, else the one the pipeline was created with.
inline DXGI_FORMAT pass_dsv_format(const BoundPipeline& bound, const Pass& pass)
{
    return pass.dsv_format != DXGI_FORMAT_UNKNOWN ? pass.dsv_format : bound.dsv_format;
}

ID3D12PipelineState* pipeline_for(BoundPipeline& bound, Translation& tr, const Pass& pass, bool indexed_rect)
{
    const DXGI_FORMAT dsv_format = pass_dsv_format(bound, pass);
    u64 key = pass.target_count;
    for (UINT i = 0; i < pass.target_count; ++i) key = key * 131 + static_cast<u64>(pass.target_formats[i]);
    key = fnv_bytes(&dsv_format, sizeof(dsv_format), key);
    // A state decoded from registers needs a depth view to act on; with none bound the pass keeps the desktop default.
    const bool use_depth_stencil = bound.has_depth_stencil && (!bound.depth_from_registers || dsv_format != DXGI_FORMAT_UNKNOWN);
    if (use_depth_stencil) key = fnv_bytes(&bound.depth_stencil, sizeof(bound.depth_stencil), key);
    if (bound.has_rasterizer) key = fnv_bytes(&bound.rasterizer, sizeof(bound.rasterizer), key);
    key = fnv_bytes(&indexed_rect, sizeof(indexed_rect), key);
    const auto found = bound.pipelines.find(key);
    if (found != bound.pipelines.end()) return found->second;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = tr.root;
    desc.VS = {tr.vs->GetBufferPointer(), tr.vs->GetBufferSize()};
    desc.PS = {tr.ps->GetBufferPointer(), tr.ps->GetBufferSize()};
    if (indexed_rect) {
        if (!tr.indexed_rect_geometry_attempted) {
            tr.indexed_rect_geometry_attempted = true;
            x2w_startup::ComPtr<ID3DBlob> errors;
            const HRESULT hr = x2w_startup::compile_indexed_rect_geometry(tr.vs, tr.ps, &tr.indexed_rect_geometry, &errors);
            if (FAILED(hr))
                log("x2w-draw %016llx: indexed rectangle geometry unavailable (0x%08lx) %s", tr.key,
                    static_cast<unsigned long>(hr), errors ? static_cast<const char*>(errors->GetBufferPointer()) : "");
        }
        if (!tr.indexed_rect_geometry) return nullptr;
        desc.GS = {tr.indexed_rect_geometry->GetBufferPointer(), tr.indexed_rect_geometry->GetBufferSize()};
    }
    // Blend state as found in the pipeline's state block, else opaque. Depth-stencil state and
    // DSVFormat come from the stream PSO (or bound DSV), else disabled. Raster state comes from
    // the stream RASTERIZER / RASTERIZER1 subobject, else solid fill, no culling, depth clip on.
    for (auto& target : desc.BlendState.RenderTarget)
        target = {FALSE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE, D3D12_BLEND_ZERO,
                  D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
    if (bound.has_blend) desc.BlendState = bound.blend;
    // Channels the translated exports leave unwritten stay out of the write mask (#1674).
    if (!x2w_startup::apply_export_write_masks(desc.BlendState, tr.write_masks, pass.target_count)) {
        log("x2w-draw export write masks incompatible with shared logic op for %016llx", tr.key);
        bound.pipelines[key] = nullptr;
        return nullptr;
    }
    if (use_depth_stencil) desc.DepthStencilState = bound.depth_stencil;
    desc.SampleMask = ~0u;
    if (bound.has_rasterizer) {
        desc.RasterizerState = bound.rasterizer;
    } else {
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable = TRUE;
    }
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = pass.target_count;
    for (UINT i = 0; i < pass.target_count; ++i) desc.RTVFormats[i] = pass.target_formats[i];
    desc.DSVFormat = dsv_format;
    desc.SampleDesc.Count = 1;
    ID3D12PipelineState* pipeline = nullptr;
    X2W_ZONE("desktop pipeline creation");
    const HRESULT hr = device_of(the_device)->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline));
    log("x2w-draw desktop pipeline for %016llx, %u targets (format %d) -> 0x%08lx (dsv %d, depth %d)", tr.key,
        pass.target_count, pass.target_count ? pass.target_formats[0] : 0, static_cast<unsigned long>(hr),
        static_cast<int>(dsv_format), desc.DepthStencilState.DepthEnable ? 1 : 0);
    check_device("translated CreateGraphicsPipelineState", 0);
    bound.pipelines[key] = pipeline;
    return pipeline;
}

// Private shader-visible heap for the translated draws' tables, used as a ring.
constexpr UINT draw_heap_size = 1u << 18;
ID3D12DescriptorHeap* draw_heap;
Wrapper* draw_heap_device;
UINT draw_heap_increment;
volatile LONG draw_heap_cursor;

bool ensure_draw_heap(Wrapper* device)
{
    // The translated pipeline and upload ring are scoped to the current
    // startup render device. A list from another device cannot use this heap.
    if (!device || device != the_device) return false;
    if (draw_heap) return device == draw_heap_device;
    const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, draw_heap_size, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(device_of(the_device)->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&draw_heap)))) return false;
    draw_heap_device = device;
    draw_heap_increment = device_of(the_device)->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

// A recorded table is immutable. Keep it alive until its submission completes,
// including when the title resets the command list with a different allocator.
using DrawSamplerHeap = x2w_startup::ComPtr<ID3D12DescriptorHeap>;
std::unordered_map<ID3D12CommandList*, std::vector<DrawSamplerHeap>> recorded_draw_sampler_heaps;
struct SubmittedDrawSamplerHeaps {
    x2w_startup::ComPtr<ID3D12Fence> fence;
    std::vector<DrawSamplerHeap> heaps;
};
std::vector<SubmittedDrawSamplerHeaps> submitted_draw_sampler_heaps;

void discard_draw_sampler_heaps(ID3D12CommandList* list)
{
    AcquireSRWLockExclusive(&draw_lock);
    recorded_draw_sampler_heaps.erase(list);
    ReleaseSRWLockExclusive(&draw_lock);
}

// draw_lock held. The lists have already been executed on this queue. A closed list can be
// submitted again until its Reset, so each submit takes its own references and the recording
// keeps its own until discard_draw_sampler_heaps.
void retire_draw_sampler_heaps(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists)
{
    submitted_draw_sampler_heaps.erase(std::remove_if(submitted_draw_sampler_heaps.begin(),
        submitted_draw_sampler_heaps.end(), [](const SubmittedDrawSamplerHeaps& batch) {
            return batch.fence && batch.fence->GetCompletedValue() != 0;
        }), submitted_draw_sampler_heaps.end());
    SubmittedDrawSamplerHeaps batch;
    for (auto* list : lists) {
        auto it = recorded_draw_sampler_heaps.find(list);
        if (it == recorded_draw_sampler_heaps.end()) continue;
        for (const auto& heap : it->second) batch.heaps.push_back(heap);
    }
    if (batch.heaps.empty()) return;
    x2w_startup::ComPtr<ID3D12Device> device;
    HRESULT hr = queue->GetDevice(IID_PPV_ARGS(&device));
    if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&batch.fence));
    if (SUCCEEDED(hr)) hr = queue->Signal(batch.fence.Get(), 1);
    if (FAILED(hr)) {
        // Without a completion point the submitted descriptors still belong
        // to the queue. Retain them instead of reclaiming them early.
        batch.fence.Reset();
        log("x2w-draw sampler heap retirement fence failed 0x%08lx", static_cast<unsigned long>(hr));
    }
    submitted_draw_sampler_heaps.push_back(std::move(batch));
}

struct TextureUpload {
    u64 hash = 0, generation = ~0ull;
    LONG checked_frame = -1000;
    bool done = false, tiled = false;
    ID3D12CommandList* pending_list = nullptr;
    u64 revision = 0;
    bool staged = false;       // the last upload attempt put the title memory into the desktop texture
    bool gpu_written = false;  // a list copy or resolve wrote the texture; its title memory is stale and must not be uploaded
};
std::unordered_map<ID3D12Resource*, TextureUpload>& texture_uploads() { static std::unordered_map<ID3D12Resource*, TextureUpload> m; return m; }
u64 address_upload_revision;

// A texture published in a newly supported console layout is not filled through the shared upload ring.
// The ring is reused without waiting for any queue, and a queue held behind a fence can execute a
// recorded copy long after later uploads have passed over its source bytes. Each such upload owns
// its upload buffer instead. The buffer and the texture it fills stay referenced by the recording,
// and by every submit of that recording until the queue it went to has completed the submit, as
// for the sampler heaps above.
struct AddressUpload {
    x2w_startup::ComPtr<ID3D12Resource> buffer, texture;
    u64 revision;
    bool pinned = false;
};
std::unordered_map<ID3D12CommandList*, std::vector<AddressUpload>> recorded_address_uploads;
struct SubmittedAddressUploads {
    x2w_startup::ComPtr<ID3D12Fence> fence;
    std::vector<AddressUpload> uploads;
};
std::vector<SubmittedAddressUploads> submitted_address_uploads;

// X2W_TEST_ADDRESS_UPLOAD_RING=1 (fixtures only): fill by-address textures from the shared upload ring,
// without the buffers above. The control of startup_address_console_gate runs with it.
bool address_uploads_from_ring()
{
    static const bool on = GetEnvironmentVariableW(L"X2W_TEST_ADDRESS_UPLOAD_RING", nullptr, 0) != 0;
    return on;
}

void discard_address_uploads(ID3D12CommandList* list)
{
    AcquireSRWLockExclusive(&draw_lock);
    const auto recorded = recorded_address_uploads.find(list);
    if (recorded != recorded_address_uploads.end()) {
        for (const auto& upload : recorded->second) {
            const auto state = texture_uploads().find(upload.texture.Get());
            if (state != texture_uploads().end() && state->second.pending_list == list) {
                state->second.done = false;
                state->second.pending_list = nullptr;
            }
        }
        recorded_address_uploads.erase(recorded);
    }
    ReleaseSRWLockExclusive(&draw_lock);
}

// draw_lock held. The lists have already been executed on this queue. Each submit takes its own
// references; they are dropped by a later call, once the fence signalled behind the submit has
// completed. A submit whose fence could not be made or signalled keeps its references.
void retire_address_uploads(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists)
{
    static std::size_t peak, released;
    const std::size_t before = submitted_address_uploads.size();
    submitted_address_uploads.erase(std::remove_if(submitted_address_uploads.begin(),
        submitted_address_uploads.end(), [](const SubmittedAddressUploads& batch) {
            return batch.fence && batch.fence->GetCompletedValue() != 0;
        }), submitted_address_uploads.end());
    if (const std::size_t gone = before - submitted_address_uploads.size()) {
        released += gone;
        static LONG reports;
        if (first_calls(reports, 12))
            log("x2w-draw by-address uploads: %zu submits completed and released (%zu so far, %zu still pending, at most %zu pending)",
                gone, released, submitted_address_uploads.size(), peak);
    }
    for (auto* list : lists) {
        const auto it = recorded_address_uploads.find(list);
        if (it == recorded_address_uploads.end()) continue;
        for (const auto& upload : it->second) {
            const auto state = texture_uploads().find(upload.texture.Get());
            if (state == texture_uploads().end()) continue;
            // A recording that writes an older revision may be submitted again. When the
            // last write in a batch differs from the cache, make the next draw restage it.
            auto& cached = state->second;
            cached.done = cached.revision == upload.revision;
            if (cached.pending_list == list) cached.pending_list = nullptr;
        }
    }
    try {
        SubmittedAddressUploads batch;
        for (auto* list : lists) {
            const auto it = recorded_address_uploads.find(list);
            if (it == recorded_address_uploads.end()) continue;
            for (const auto& upload : it->second)
                if (!upload.pinned) batch.uploads.push_back(upload);
        }
        if (batch.uploads.empty()) return;
        x2w_startup::ComPtr<ID3D12Device> device;
        HRESULT hr = queue->GetDevice(IID_PPV_ARGS(&device));
        if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&batch.fence));
        if (SUCCEEDED(hr)) hr = queue->Signal(batch.fence.Get(), 1);
        if (FAILED(hr)) {
            batch.fence.Reset();
            log("x2w-draw by-address upload retirement fence failed 0x%08lx", static_cast<unsigned long>(hr));
        }
        submitted_address_uploads.push_back(std::move(batch));
        peak = (std::max)(peak, submitted_address_uploads.size());
    } catch (const std::bad_alloc&) {
        // Execution already happened. Pin each affected resource once if bookkeeping fails.
        for (auto* list : lists) {
            const auto it = recorded_address_uploads.find(list);
            if (it == recorded_address_uploads.end()) continue;
            for (auto& upload : it->second) {
                if (upload.pinned) continue;
                upload.buffer->AddRef();
                upload.texture->AddRef();
                upload.pinned = true;
            }
        }
        log("x2w-draw by-address upload retirement storage exhausted; uploads pinned");
    }
}

bool resolve_draw_sampler(const x2w_startup::SamplerWords& words, D3D12_SAMPLER_DESC& sampler)
{
    if (words[0] != view_token_magic) {
        sampler = x2w_startup::dynamic_sampler(words);
        return true;
    }
    AcquireSRWLockShared(&note_lock);
    const auto& records = view_records();
    const bool valid = words[2] == 22 && words[1] < records.size() && records[words[1]].slot == 22 &&
                       records[words[1]].desc_bytes == sizeof(sampler);
    if (valid) std::memcpy(&sampler, records[words[1]].desc, sizeof(sampler));
    ReleaseSRWLockShared(&note_lock);
    return valid;
}

// draw_lock held.
bool prepare_draw_sampler_heap(Pass& pass, ID3D12GraphicsCommandList* list, bool reserved_fallback)
{
    auto snapshot = x2w_startup::snapshot_samplers(pass.title_samplers, reserved_fallback);
    if (pass.sampler_snapshot && *pass.sampler_snapshot == snapshot) return true;
    x2w_startup::NullSamplerCounts counts;
    DrawSamplerHeap heap;
    const HRESULT hr = x2w_startup::create_sampler_snapshot_heap(device_of(the_device), snapshot,
        resolve_draw_sampler, heap.GetAddressOf(), counts);
    if (FAILED(hr)) {
        log("x2w-draw sampler heap creation failed 0x%08lx", static_cast<unsigned long>(hr));
        return false;
    }
    static LONG reports;
    if (first_calls(reports, 12))
        log("x2w-draw sampler heap: null_sampler_unbound=%u null_sampler_out_of_range=%u "
            "null_sampler_unreadable=%u null_sampler_invalid_token=%u",
            counts.unbound, counts.out_of_range, counts.unreadable, counts.invalid_token);
    recorded_draw_sampler_heaps[list].push_back(heap);
    pass.sampler_heap = std::move(heap);
    pass.sampler_snapshot = std::make_unique<x2w_startup::SamplerSnapshot>(std::move(snapshot));
    return true;
}

// Copies of title memory made for the current frame, found by (address, bytes). A title may rewrite the same
// address between two draws of one frame (a dynamic constant buffer, a reused vertex or index range), so an
// entry keeps the bytes it was made from and is reused only while the title's memory still matches them. The
// upload ring is write-combined memory that is slow to read, so the comparison is against this copy, not the ring.
struct FrameCopy {
    u64 address = 0;
    u64 offset = 0;
    std::vector<unsigned char> content;
};
std::unordered_map<u64, FrameCopy>& frame_copies() { static std::unordered_map<u64, FrameCopy> m; return m; }
LONG frame_copies_frame = -1;

// Callers have just probed that [address, address + bytes) is readable (readable_window), and draw_lock is held.
bool frame_copy(u64 address, std::size_t bytes, u64* offset)
{
    if (frame_copies_frame != frame_number) {
        frame_copies().clear();
        frame_copies_frame = frame_number;
    }
    const u64 key = address * 1099511628211ull ^ bytes;
    const auto* title = reinterpret_cast<const unsigned char*>(address);
    const auto found = frame_copies().find(key);
    const bool known = found != frame_copies().end() && found->second.address == address && found->second.content.size() == bytes;
    if (known && std::memcmp(found->second.content.data(), title, bytes) == 0) {
        *offset = found->second.offset;
        return true;
    }
    u64 fresh = 0;
    auto* ring = reserve_upload_ring(bytes, &fresh);
    if (!ring) return false;
    // Cache insertion and snapshot allocation are optional. Fill the ring directly if they fail.
    const unsigned char* source = title;
    try {
        std::vector<unsigned char> content(title, title + bytes);
        // Use the stored snapshot to fill the ring so the GPU bytes and subsequent comparisons agree.
        auto entry = frame_copies().insert_or_assign(key, FrameCopy{address, fresh, std::move(content)});
        source = entry.first->second.content.data();
    } catch (const std::bad_alloc&) {
        // The ring slice is already reserved. The draw can proceed without another CPU allocation.
    }
    std::memcpy(ring, source, bytes);
    if (known) {
        static LONG refreshed;
        if (first_calls(refreshed, 4))
            log("x2w-draw frame copy of 0x%llx (%zu bytes) refreshed: the title changed it since an earlier draw of this frame", address, bytes);
    }
    *offset = fresh;
    return true;
}

void buffer_view(D3D12_CPU_DESCRIPTOR_HANDLE dest, ID3D12Resource* buffer, u64 offset, std::size_t bytes)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Buffer.FirstElement = buffer ? offset / 4 : 0;
    desc.Buffer.NumElements = buffer ? static_cast<UINT>((bytes + 3) / 4) : 1;
    desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    device_of(the_device)->CreateShaderResourceView(buffer, &desc, dest);
}

// Texture contents. Linear backing uses desktop copyable footprints; an announced BC7 4 KiB
// surface must be detiled into the footprint first. A texture is uploaded on its first bind and
// again when its memory changed: XG copies announce themselves (x2w_d3d12_texture_written), and
// CPU-backed textures up to 32 MiB are also re-hashed for writes the title makes on its own.
// Recheck single-mip R8 and A8 planes on their first bind each frame and other textures every 15 frames.
// Targets and UAVs are GPU-written and never uploaded.
LONG textures_uploaded, texture_bytes_uploaded, textures_unreadable;
u64 hash_memory(const unsigned char* data, std::size_t bytes)
{
    u64 h = 0xcbf29ce484222325ull;
    const std::size_t words = bytes / 8;
    const u64* w = reinterpret_cast<const u64*>(data);
    for (std::size_t i = 0; i < words; ++i) h = (h ^ w[i]) * 0x100000001b3ull;
    for (std::size_t i = words * 8; i < bytes; ++i) h = (h ^ data[i]) * 0x100000001b3ull;
    return h;
}
bool list_wrote(Wrapper* list, ID3D12Resource* resource);  // list_copy.inl
bool upload_texture(Wrapper* self, Wrapper* resource, ID3D12Resource* texture)
{
    auto* const memory = static_cast<const unsigned char*>(resource->own().title_memory);
    auto* list = recording(self);
    const D3D12_RESOURCE_DESC desc = texture->GetDesc();
    const auto* original = resource_info(resource);
    const bool tiled_address = resource->own().tiled_address_srv;
    const bool console = tiled_address && original && console_layout_srv(original->console);
    const bool dedicated = console && !address_uploads_from_ring();
    // An address-only single-element texture (2D or 3D) uploads its one texel through the footprint path.
    const bool single = resource->own().tiled_address_srv && x2w_startup::single_element_bytes(desc) != 0;
    const bool uploadable = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D || single ||
        (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && desc.DepthOrArraySize == 1 && desc.Height == 1);
    {
        static LONG reasons;
        if ((!memory || (desc.Flags & 0x7) || !uploadable) && !texture_uploads().count(texture) &&
            first_calls(reasons, 12)) {
            texture_uploads()[texture].done = true;
            log("x2w-draw texture not uploaded %p: %llux%u format %d flags 0x%x dim %d title memory %p", static_cast<void*>(texture),
                desc.Width, desc.Height, desc.Format, desc.Flags, desc.Dimension, static_cast<const void*>(memory));
        }
    }
    if (!memory || !list || !uploadable) return !console;
    if (list_wrote(self, texture)) return !console;  // this recording's own copy; not yet submitted
    if (const auto written_by_gpu = texture_uploads().find(texture); written_by_gpu != texture_uploads().end() && written_by_gpu->second.gpu_written) return !console;
    if (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return !console;
    const UINT mips = desc.MipLevels ? desc.MipLevels : 1;
    // For Texture3D, depth slices are carried in Footprint.Depth and spanned by SliceFootprints::bytes().
    const UINT slices = (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D || !desc.DepthOrArraySize) ? 1 : desc.DepthOrArraySize;
    if (mips > 15) return !console;
    const x2w_startup::SliceFootprints plan(device_of(the_device), desc, mips);
    const UINT64 slice_total = plan.size;
    if (slice_total == ~0ull || !slice_total) return !console;
    const u64 stride = (slice_total + 0xFFFF) & ~u64{0xFFFF};
    TextureUpload* cached;
    try {
        cached = &texture_uploads()[texture];
    } catch (const std::bad_alloc&) {
        if (!console) throw;
        return false;
    }
    TextureUpload& state = *cached;
    TextureWrite written;
    AcquireSRWLockShared(&written_lock);
    if (const auto found = written_textures().find(reinterpret_cast<u64>(memory)); found != written_textures().end())
        written = found->second;
    ReleaseSRWLockShared(&written_lock);
    const bool tiled_chain = original && x2w_startup::bc64::supports(original->console);
    // Any other texture slot 70 publishes holds its texels in the console layout its description names
    // (gfx10_layout.h, address_texture.h).
    const bool tiled_bc7 = !tiled_chain && !tiled_address && written.bc7_width != 0;
    const bool tiled = tiled_bc7 || tiled_address || tiled_chain;
    std::optional<x2w_startup::gfx10::Layout> console_layout;
    if (console) console_layout.emplace(original->console);
    const auto generation = written.generation;
    const x2w_startup::Bc16Surface surface(tiled_bc7 ? written.bc7_width : static_cast<UINT>(desc.Width),
                                         tiled_bc7 ? written.bc7_height : desc.Height);
    if (tiled_bc7) {
        if (!original || !x2w_startup::supports_bc7_4k_s(original->console) ||
            original->console.Width != written.bc7_width || original->console.Height != written.bc7_height ||
            desc.Format != DXGI_FORMAT_BC7_UNORM || desc.Width != surface.columns * surface.block_texels ||
            desc.Height != surface.rows * surface.block_texels) return true;
    }
    if (tiled && !tiled_chain && !single && !console && (plan.count != 1 || plan.rows[0] != surface.rows ||
                  plan.row_bytes[0] != u64{surface.columns} * surface.block_bytes)) return true;
    if (console) {
        // The desktop footprints must hold exactly the elements the console layout addresses.
        if (console_layout->mips != mips || console_layout->slices != slices || plan.count != mips) return false;
        for (UINT i = 0; i < mips; ++i)
            if (!plan.bytes(i) || plan.rows[i] != console_layout->rows(i) ||
                plan.row_bytes[i] != u64{console_layout->columns(i)} * console_layout->bytes_per_element) return false;
    }
    // A texture given the 4 KiB allocation answer is read in the compact plan XG describes and writes for it
    // (small_texture_4k.h); its rows must be the footprint's rows, level by level. Address-only textures
    // (single-element, or a console layout named in Layout) are tiled and keep their own paths.
    x2w_startup::SmallTexturePlan compact;
    const bool is_compact = original && !tiled && x2w_startup::small_texture_4k_plan(original->console, compact);
    if (is_compact) {
        bool rows_match = compact.levels == plan.count;
        for (UINT i = 0; rows_match && i < plan.count; ++i)
            rows_match = compact.level[i].rows == plan.rows[i] && compact.level[i].row_bytes == plan.row_bytes[i];
        if (!rows_match) return true;
    }
    const u64 total = tiled_chain ? x2w_startup::bc64::Chain(original->console).bytes() : console ? console_layout->bytes()
        : tiled && !single ? surface.size() : is_compact ? compact.bytes : stride * (slices - 1) + slice_total;
    const bool announced = state.generation != generation;
    const UINT elapsed = static_cast<UINT>(frame_number) - static_cast<UINT>(state.checked_frame);
    // A single texel (at most 16 bytes) is rehashed on every bind, so a write between two draws is seen.
    const bool recheck = single || ((tiled_address || tiled_chain) ? elapsed != 0
        : x2w_startup::recheck_cpu_texture(desc, total, elapsed));
    const bool reusable = state.done && (!state.pending_list || state.pending_list == list);
    if (reusable && !announced && !recheck) return true;
    if (readable_window(reinterpret_cast<u64>(memory), static_cast<std::size_t>(total)) < total) {
        if (!state.done) InterlockedIncrement(&textures_unreadable);
        if (console) return false;
        state.done = true;
        state.staged = false;
        state.generation = generation;
        return true;
    }
    if (!console) state.checked_frame = frame_number;
    const u64 hash = hash_memory(memory, static_cast<std::size_t>(total));
    if (reusable && hash == state.hash && state.tiled == tiled) {
        state.checked_frame = frame_number;
        state.generation = generation;
        return true;
    }
    constexpr u64 placement = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    x2w_upload::Page owned;
    if (dedicated) {
        u64 bytes = 0;
        for (UINT i = 0; i < mips; ++i) bytes += (plan.bytes(i) + placement - 1) & ~(placement - 1);
        bytes *= slices;
        if (FAILED(x2w_upload::make_page(device_of(the_device), bytes, owned))) return false;
        // Store the owning reference before issuing any D3D12 commands that name the buffer.
        try {
            recorded_address_uploads[list].push_back({owned.resource, texture, ++address_upload_revision});
        } catch (const std::bad_alloc&) {
            return false;
        }
        std::memset(owned.cpu, 0, static_cast<std::size_t>(bytes));
    }
    state.staged = false;
    require_state(self, texture, D3D12_RESOURCE_STATE_COPY_DEST);
    if (console) {
        for (UINT slice = 0; slice < slices; ++slice) {
            for (UINT i = 0; i < mips; ++i) {
                const u64 bytes = plan.bytes(i);
                u64 offset = owned.cursor;
                unsigned char* rows = owned.cpu ? owned.cpu + offset : nullptr;
                if (owned.cpu) {
                    owned.cursor += (bytes + placement - 1) & ~(placement - 1);
                } else {
                    rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset);
                    if (!rows) {
                        state.done = false;
                        return false;
                    }
                    std::memset(rows, 0, static_cast<std::size_t>(bytes));
                }
                console_layout->detile(i, slice, memory, rows, plan.placed[i].Footprint.RowPitch);
                plan.upload(list, texture, owned.cpu ? owned.resource.Get() : upload_ring, mips, slices, slice, i, offset);
            }
        }
        if (dedicated) {
            state.pending_list = list;
            state.revision = address_upload_revision;
        }
    } else if (tiled_chain) {
        const x2w_startup::bc64::Chain chain(original->console);
        for (UINT slice = 0; slice < slices; ++slice) {
            for (UINT i = 0; i < mips; ++i) {
                const u64 bytes = plan.bytes(i);
                u64 offset = 0;
                auto* rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset);
                if (!rows) {
                    state.done = false;
                    return true;
                }
                std::memset(rows, 0, static_cast<std::size_t>(bytes));
                chain.copy<false>(i, slice, rows, memory, plan.placed[i].Footprint.RowPitch);
                plan.upload(list, texture, upload_ring, mips, slices, slice, i, offset);
            }
        }
    } else if (tiled && !single) {
        u64 offset = 0;
        const auto bytes = plan.bytes(0);
        auto* rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset);
        if (!rows) {
            state.done = false;
            return true;
        }
        std::memset(rows, 0, static_cast<std::size_t>(bytes));
        surface.detile(memory, rows, plan.placed[0].Footprint.RowPitch);
        plan.upload(list, texture, upload_ring, 1, 1, 0, 0, offset);
    } else {
        for (UINT s = 0; s < slices; ++s)
            for (UINT i = 0; i < plan.count; ++i) {
                const auto& fp = plan.placed[i];
                const u64 bytes = plan.bytes(i);
                u64 offset = 0;
                if (is_compact) {
                    auto* const rows = reserve_upload_ring(static_cast<std::size_t>(bytes), &offset);
                    if (!rows) {
                        state.done = false;
                        return true;
                    }
                    const auto& level = compact.level[i];
                    for (u64 y = 0; y < level.rows; ++y)
                        std::memcpy(rows + y * fp.Footprint.RowPitch, memory + level.offset + y * level.row_bytes,
                                    static_cast<std::size_t>(level.row_bytes));
                } else if (!ring_copy(reinterpret_cast<u64>(memory) + stride * s + fp.Offset, static_cast<std::size_t>(bytes), &offset)) {
                    state.done = false;
                    return true;
                }
                plan.upload(list, texture, upload_ring, mips, slices, s, i, offset);
            }
    }
    state.hash = hash;
    state.generation = generation;
    state.checked_frame = frame_number;
    state.tiled = tiled;
    state.done = true;
    state.staged = true;
    InterlockedIncrement(&textures_uploaded);
    InterlockedExchangeAdd(&texture_bytes_uploaded, static_cast<LONG>(total >> 10));
    static LONG shown;
    if (first_calls(shown, 12))
        log("x2w-draw texture upload %p %llux%u x%u mips %u format %d from %p (%llu KiB)%s%s", static_cast<void*>(texture), desc.Width,
            desc.Height, slices, mips, desc.Format, static_cast<const void*>(memory), total >> 10, announced ? " (announced write)" : "",
            is_compact ? " (compact plan of the 4 KiB allocation answer)" : "");
    return true;
}

// Per-draw dumps (X2W_DRAW_DUMP_FRAME=<frame>, X2W_DRAW_DUMP_DIR=<folder>): during that frame, render
// target 0 is copied to a readback buffer after each translated draw (at most 48).
// Poll the fence of the queue that actually executed the recording before writing its image.
using x2w_startup::DrawDump;
LONG draw_dump_frame()
{
    static const LONG frame = [] {
        char value[16]{};
        return GetEnvironmentVariableA("X2W_DRAW_DUMP_FRAME", value, sizeof(value)) ? std::atol(value) : 0L;
    }();
    return frame;
}
void draw_dump_after(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key, UINT count)
{
    Pass& pass = pass_of(self);
    const LONG wanted = draw_dump_frame();
    if (!wanted || frame_number < wanted || frame_number > wanted + 1 || !pass.target0 || draw_dumps().full()) return;
    const D3D12_RESOURCE_DESC desc = pass.target0->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
        !x2w_draw_bmp::supported(desc.Format)) return;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device_of(the_device)->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* readback = nullptr;
    if (FAILED(device_of(the_device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                              IID_PPV_ARGS(&readback)))) return;
    x2w_budget::committed(device_of(the_device), readback, D3D12_HEAP_TYPE_READBACK);
    DrawDump dump{};
    dump.buffer.Attach(readback);
    dump.footprint = footprint;
    dump.rows = rows;
    dump.key = key;
    dump.count = count;
    dump.format = desc.Format;
    if (!draw_dumps().record(list, std::move(dump))) return;
    require_state(self, pass.target0, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION target{readback, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    target.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION source{pass.target0, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    source.SubresourceIndex = 0;
    list->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
    require_state(self, pass.target0, D3D12_RESOURCE_STATE_RENDER_TARGET);
}
void draw_dump_flush()
{
    if (!draw_dump_frame()) return;
    auto completed = draw_dumps().take_completed();
    if (completed.empty()) return;
    char folder[MAX_PATH] = ".";
    GetEnvironmentVariableA("X2W_DRAW_DUMP_DIR", folder, sizeof(folder));
    int written = 0;
    for (const DrawDump& dump : completed) {
        const int index = static_cast<int>(dump.index);
        void* mapped = nullptr;
        if (FAILED(dump.buffer->Map(0, nullptr, &mapped))) continue;
        char path[MAX_PATH + 64];
        std::snprintf(path, sizeof(path), "%s\\draw_%02d_%016llx_%u.bmp", folder, index, dump.key, dump.count);
        if (FILE* file = std::fopen(path, "wb")) {
            const auto& fp = dump.footprint.Footprint;
            const auto* pixels = static_cast<const unsigned char*>(mapped) + dump.footprint.Offset;
            const bool complete = x2w_draw_bmp::write(file, pixels, fp.Width, dump.rows, fp.RowPitch, dump.format);
            if (std::fclose(file) == 0 && complete) ++written;
            else std::remove(path);
        }
        const D3D12_RANGE no_writes{};
        dump.buffer->Unmap(0, &no_writes);
    }
    log("x2w-draw per-draw dump: %d of %zu completed target copies written to %s", written, completed.size(), folder);
}

bool d16_srvs_ready(const Translation& tr, const Pass& pass, std::uint32_t (&d16_unorm16)[32])
{
    if (!tr.d16_srvs.any()) return true;
    const auto refuse = [&](UINT slot, const char* reason, int detail) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-draw %016llx srv %u: d16 exact refused (%s %d)", tr.key, slot, reason, detail);
        return false;
    };
    for (UINT i = 0; i < tr.srvs.size(); ++i) {
        if (!tr.d16_srvs.requires_exact(i)) continue;
        const Translation::Srv& srv = tr.srvs[i];
        if (srv.kind < 1 || srv.kind > 4 || srv.source != 0 || srv.a < 0 || srv.a >= 16 || srv.b < 0)
            return refuse(i, "non-table texture binding", srv.kind);
        const auto dimension = srv.kind == 1 ? D3D12_SRV_DIMENSION_TEXTURE2D :
                               srv.kind == 2 ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY :
                               srv.kind == 4 ? D3D12_SRV_DIMENSION_TEXTURE1D : D3D12_SRV_DIMENSION_TEXTURE3D;
        const u64 entry = pass.view_tables[srv.a].record(static_cast<u64>(srv.b), 32);
        std::uint32_t words[8] = {};
        if (!entry || !x2w_local::readable(reinterpret_cast<const void*>(entry), 32))
            return refuse(i, "unreadable table entry", srv.a);
        std::memcpy(words, reinterpret_cast<const void*>(entry), 32);
        if (words[0] != view_token_magic) return refuse(i, "missing view token", 0);
        ViewRecord record{};
        bool known = false;
        AcquireSRWLockShared(&note_lock);
        known = words[1] < view_records().size();
        if (known) record = view_records()[words[1]];
        ReleaseSRWLockShared(&note_lock);
        if (!known || record.slot != 18) return refuse(i, "view slot", static_cast<int>(record.slot));
        Wrapper* resource = as_wrapper(reinterpret_cast<const void*>(record.resource));
        ID3D12Resource* texture = sampled_address_texture(resource);
        if (!texture) return refuse(i, "missing texture resource", 0);
        if (x2w_startup::packed_stand_in_of(texture) != DXGI_FORMAT_UNKNOWN)
            return refuse(i, "packed stand-in resource", static_cast<int>(x2w_startup::packed_stand_in_of(texture)));
        const D3D12_RESOURCE_DESC tex_desc = texture->GetDesc();
        if (srv.kind == 4 && (tex_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D || tex_desc.DepthOrArraySize != 1))
            return refuse(i, "1D resource mismatch", static_cast<int>(tex_desc.Dimension));
        const auto* original = resource_info(resource);
        DXGI_FORMAT effective = original ? original->console.Format : tex_desc.Format;
        if (record.desc_bytes >= 40) {
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            std::memcpy(&desc, record.desc, 40);
            if (desc.ViewDimension != dimension)
                return refuse(i, "view dimension", static_cast<int>(desc.ViewDimension));
            if (desc.Format != DXGI_FORMAT_UNKNOWN) effective = desc.Format;
        } else if (record.desc_bytes == 0) {
            const bool dim_ok =
                (srv.kind == 1 && tex_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && tex_desc.DepthOrArraySize == 1) ||
                (srv.kind == 2 && tex_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) ||
                (srv.kind == 3 && tex_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) ||
                (srv.kind == 4 && tex_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && tex_desc.DepthOrArraySize == 1);
            if (!dim_ok) return refuse(i, "default view dimension", static_cast<int>(tex_desc.Dimension));
        } else {
            return refuse(i, "truncated view desc", static_cast<int>(record.desc_bytes));
        }
        const auto mode = x2w_startup::d16_srv_mode(effective);
        if (mode == x2w_startup::D16SrvMode::refused)
            return refuse(i, "format", static_cast<int>(effective));
        d16_unorm16[i] = mode == x2w_startup::D16SrvMode::unorm16 ? 1u : 0u;
    }
    return true;
}

// FXC keeps unused entry declarations in reflection. Check actual uses so a
// generated shader which reads the draw's start vertex or instance is refused.
bool indirect_shader_entry(ID3DBlob* blob, const Translation& translation, bool indexed = false)
{
    x2w_startup::ComPtr<ID3D12ShaderReflection> shader;
    if (!blob || FAILED(D3DReflect(blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(shader.GetAddressOf())))) return false;
    D3D12_SHADER_DESC desc{};
    if (FAILED(shader->GetDesc(&desc)) || ((desc.Version >> 4) & 0xf) != 5) return false;
    for (UINT binding = 0; binding < desc.BoundResources; ++binding) {
        D3D12_SHADER_INPUT_BIND_DESC input{};
        if (FAILED(shader->GetResourceBindingDesc(binding, &input))) return false;
        if (input.Type != D3D_SIT_CBUFFER) continue;
        if (!input.Name || std::strcmp(input.Name, "entry") || input.Space || input.BindPoint || input.BindCount != 1) return false;
        auto* buffer = shader->GetConstantBufferByName(input.Name);
        D3D12_SHADER_BUFFER_DESC cb{};
        if (!buffer || FAILED(buffer->GetDesc(&cb))) return false;
        for (UINT i = 0; i < cb.Variables; ++i) {
            D3D12_SHADER_VARIABLE_DESC var{};
            if (FAILED(buffer->GetVariableByIndex(i)->GetDesc(&var))) return false;
            if (!(var.uFlags & D3D_SVF_USED)) continue;
            const auto word = var.StartOffset / sizeof(UINT);
            char name[16];
            std::snprintf(name, sizeof(name), "ud%u", static_cast<unsigned>(word));
            if (var.StartOffset % sizeof(UINT) || var.Size != sizeof(UINT) || word >= translation.uds.size() ||
                !var.Name || std::strcmp(var.Name, name)) return false;
            const auto& ud = translation.uds[word];
            if (indexed && ud.kind == Translation::base_vertex) continue;
            if (ud.kind == Translation::stride && ud.a >= 0 && static_cast<std::size_t>(ud.a) < translation.srvs.size() &&
                (translation.srvs[ud.a].source == 0 || (indexed && translation.srvs[ud.a].source == 1)) &&
                translation.srvs[ud.a].kind == 0) continue;
            if (ud.kind != Translation::root_constant && ud.kind != Translation::rect && ud.kind != Translation::d16_unorm16)
                return false;
        }
    }
    return true;
}

namespace indirect_refusal_code {
constexpr const char* pipeline_or_target_unavailable = "pipeline_or_target_unavailable";
constexpr const char* translation_not_ready = "translation_not_ready";
constexpr const char* d16_srv_not_ready = "d16_srv_not_ready";
constexpr const char* formatted_buffer_not_ready = "formatted_buffer_not_ready";
constexpr const char* missing_guest_root = "missing_guest_root";
constexpr const char* live_compute_pipeline = "live_compute_pipeline";
constexpr const char* unsupported_resource_or_entry = "unsupported_resource_or_entry";
constexpr const char* indexed_rect_unavailable = "indexed_rect_unavailable";
constexpr const char* pipeline_unavailable = "pipeline_unavailable";
constexpr const char* sampler_heap_unavailable = "sampler_heap_unavailable";
constexpr const char* sampled_texture_mismatch = "sampled_texture_mismatch";
constexpr const char* sampled_texture_unbindable = "sampled_texture_unbindable";
constexpr const char* incomplete_vertex_stream = "incomplete_vertex_stream";
constexpr const char* incomplete_table_buffer = "incomplete_table_buffer";
constexpr const char* indexed_validation_failed = "indexed_validation_failed";
}  // namespace indirect_refusal_code

struct TranslatedIndirectDraw {
    ID3D12CommandSignature* signature;
    ID3D12Resource* argument;
    u64 offset;
    IndirectIndexView indices;
    u64 shader_key = 0;
    const char* refusal = "translated pipeline or target unavailable";
    const char* refusal_code = indirect_refusal_code::pipeline_or_target_unavailable;

    // Keep prose refusal text and census code updated together.
    void set_refusal(const char* message, const char* code)
    {
        refusal = message;
        refusal_code = code;
    }
};

struct IndexedValidationPipeline {
    x2w_startup::ComPtr<ID3D12Device> device;
    x2w_startup::ComPtr<ID3D12RootSignature> root;
    x2w_startup::ComPtr<ID3D12PipelineState> pipeline;
};

IndexedValidationPipeline& indexed_validation_pipeline(ID3D12Device* device)
{
    // Hold the compute pipeline for the life of the shim and keep incompatible devices out of the route.
    static IndexedValidationPipeline validation = [device] {
        IndexedValidationPipeline result;
        result.device = device;
        D3D12_ROOT_PARAMETER params[4]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues = 8;
        params[1].ParameterType = params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[2].Descriptor.ShaderRegister = 1;
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        const D3D12_ROOT_SIGNATURE_DESC desc{4, params};
        x2w_startup::ComPtr<ID3DBlob> signature, code;
        HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, nullptr);
        if (SUCCEEDED(hr)) hr = device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(),
                                                            IID_PPV_ARGS(&result.root));
        if (SUCCEEDED(hr)) hr = D3DCompile(x2w_startup::indexed_indirect_guard, sizeof(x2w_startup::indexed_indirect_guard) - 1,
                                           "indexed indirect validation", nullptr, nullptr, "guard", "cs_5_0", 0, 0, &code, nullptr);
        if (SUCCEEDED(hr)) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
            pipeline.pRootSignature = result.root.Get();
            pipeline.CS = {code->GetBufferPointer(), code->GetBufferSize()};
            hr = device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&result.pipeline));
        }
        if (FAILED(hr)) log("x2w-render translated indexed argument validation setup failed 0x%08lx", static_cast<unsigned long>(hr));
        return result;
    }();
    return validation;
}

ID3D12Resource* validate_indexed_arguments(Wrapper* self, ID3D12GraphicsCommandList* list, Translation& translation,
                                         const TranslatedIndirectDraw& draw, UINT vertices)
{
    auto* device = device_of(self->own().list_device);
    auto& compute = indexed_validation_pipeline(device);
    if (compute.device.Get() != device || !compute.pipeline) return nullptr;
    if (!translation.indexed_signature) {
        D3D12_INDIRECT_ARGUMENT_DESC fields[2]{};
        fields[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        fields[0].Constant = {0, static_cast<UINT>(translation.indirect_base_word), 1};
        fields[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
        const D3D12_COMMAND_SIGNATURE_DESC signature{24, 2, fields, 0};
        if (FAILED(device->CreateCommandSignature(&signature, translation.root, IID_PPV_ARGS(&translation.indexed_signature))))
            return nullptr;
    }
    const auto argument_desc = draw.argument->GetDesc();
    const auto index_desc = draw.indices.resource->GetDesc();
    if (argument_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || index_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
        (index_desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) || (argument_desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) ||
        argument_desc.Width > UINT_MAX || index_desc.Width > UINT_MAX || draw.offset > UINT_MAX || draw.indices.offset > UINT_MAX)
        return nullptr;

    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = 28;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    x2w_startup::ComPtr<ID3D12Resource> scratch;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&scratch)))) return nullptr;
    const auto keep = [&] {
        AcquireSRWLockExclusive(&allocator_lock);
        const auto found = list_recordings.find(self);
        bool retained = false;
        if (found != list_recordings.end() && found->second) {
            try {
                found->second->indexed_validation.push_back(scratch);
                retained = true;
            } catch (const std::bad_alloc&) {
            }
        }
        ReleaseSRWLockExclusive(&allocator_lock);
        return retained;
    };
    if (!keep()) return nullptr;

    auto* indices = draw.indices.resource;
    const auto before = [&](ID3D12Resource* resource) {
        for (const auto& state : pass_of(self).states)
            if (state.first == resource) return state.second;
        return D3D12_RESOURCE_STATE_COMMON;
    };
    const auto argument_previous = before(draw.argument);
    const auto index_previous = before(indices);
    require_state(self, draw.argument, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (indices != draw.argument && !draw.indices.upload)
        require_state(self, indices, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    require_state(self, scratch.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const UINT bounds[]{static_cast<UINT>(draw.offset), static_cast<UINT>(argument_desc.Width), static_cast<UINT>(draw.indices.offset),
        draw.indices.view.SizeInBytes, static_cast<UINT>(index_desc.Width), draw.indices.view.Format == DXGI_FORMAT_R16_UINT ? 2u : 4u,
        vertices, 0};
    list->SetPipelineState(compute.pipeline.Get());
    list->SetComputeRootSignature(compute.root.Get());
    list->SetComputeRoot32BitConstants(0, 8, bounds, 0);
    list->SetComputeRootShaderResourceView(1, draw.argument->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(2, indices->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(3, scratch->GetGPUVirtualAddress());
    list->Dispatch(1, 1, 1);
    require_state(self, scratch.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    if (draw.argument != indices && argument_previous != D3D12_RESOURCE_STATE_COMMON)
        require_state(self, draw.argument, argument_previous);
    constexpr auto shader_reads = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (!draw.indices.upload)
        require_state(self, indices, (index_previous & shader_reads) | D3D12_RESOURCE_STATE_INDEX_BUFFER);

    // The clamp changes the native compute root and invalidates its arguments.
    // In render mode slot 43 is the forwarded compute parameter setter.
    const auto& pass = pass_of(self);
    if (pass.desktop_compute_root) {
        list->SetComputeRootSignature(pass.desktop_compute_root.Get());
        for (UINT parameter = 0; parameter < pass.desktop_compute_upload_srvs.size(); ++parameter) {
            const auto address = pass.desktop_compute_upload_srvs[parameter];
            if (address) list->SetComputeRootShaderResourceView(parameter, address);
        }
    }
    return scratch.Get();
}

struct IndirectBufferView {
    ID3D12Resource* resource = nullptr;
    u64 offset = 0, bytes = 0;
    UINT stride = 0;
    bool upload = false;
};

// For indirect draws, bind the complete descriptor range on a known GPU backing. There is no
// fallback to copying an unrelated or stale title allocation into the draw upload ring.
IndirectBufferView indirect_table_buffer(Wrapper* self, const std::uint32_t* words, bool allow_address_alias = false)
{
    constexpr u64 limit = 16u << 20;
    u64 address = 0, offset = 0, bytes = 0;
    UINT stride = 0;
    const Wrapper* owner = nullptr;
    if (words[0] == view_token_magic) {
        ViewRecord record{};
        AcquireSRWLockShared(&note_lock);
        if (words[1] < view_records().size()) record = view_records()[words[1]];
        ReleaseSRWLockShared(&note_lock);
        if (record.slot == 17 && record.desc_bytes >= 12) {
            UINT size = 0;
            std::memcpy(&address, record.desc, 8);
            std::memcpy(&size, record.desc + 8, 4);
            bytes = size;
        } else if (record.slot == 18 && record.desc_bytes >= 40) {
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            std::memcpy(&view, record.desc, 40);
            owner = as_wrapper(reinterpret_cast<const void*>(record.resource));
            if (!owner || owner->own().kind != resource_kind || !owner->own().real ||
                view.ViewDimension != D3D12_SRV_DIMENSION_BUFFER) return {};
            const bool raw = view.Buffer.Flags == D3D12_BUFFER_SRV_FLAG_RAW && !view.Buffer.StructureByteStride &&
                view.Format == DXGI_FORMAT_R32_TYPELESS;
            const bool structured = view.Buffer.Flags == D3D12_BUFFER_SRV_FLAG_NONE && view.Buffer.StructureByteStride &&
                view.Format == DXGI_FORMAT_UNKNOWN;
            const bool typed = view.Buffer.Flags == D3D12_BUFFER_SRV_FLAG_NONE && !view.Buffer.StructureByteStride;
            const UINT element = raw ? 4 : structured ? view.Buffer.StructureByteStride :
                typed ? x2w_startup::typed_buffer_format_bytes(view.Format) : 0;
            if (!element || view.Buffer.FirstElement > UINT64_MAX / element) return {};
            offset = view.Buffer.FirstElement * element;
            bytes = u64{view.Buffer.NumElements} * element;
            stride = raw ? 0 : element;
            if (const auto base = reinterpret_cast<u64>(owner->own().title_memory)) {
                if (offset > UINT64_MAX - base) return {};
                address = base + offset;
            }
        }
    } else {
        address = words[0] | (u64{words[1] & 0xffff} << 32);
        stride = (words[1] >> 16) & 0x3fff;
        bytes = u64{stride ? stride : 1} * words[2];
    }
    if (!bytes || bytes > limit || bytes % 4) return {};
    ID3D12Resource* native = nullptr;
    bool upload = false;
    Wrapper* device = self->own().list_device;
    if (address) {
        if (const auto allocation = list_upload_range(self, address, bytes); allocation.resource) {
            if (!owner) return {allocation.resource, allocation.offset, bytes, stride, true};
            return {};
        }
        const auto placed = placed_buffer_range(device, address, bytes);
        if (placed.reject || (owner && owner != placed.owner)) return {};
        if (!placed.owner) {
            const auto alias = allow_address_alias && !list_upload_range(self, address, 1).resource
                ? indirect_address_alias(device, address, bytes) : IndirectAddressAlias{};
            return {alias.resource, alias.offset, bytes, stride};
        }
        owner = placed.owner;
        native = static_cast<ID3D12Resource*>(owner->own().real);
        offset = address - reinterpret_cast<u64>(owner->own().title_memory);
        if (!placed.alias && (!native || !list_copied_gpu_range(self, native, offset, bytes))) return {};
    } else if (owner && !owner->own().title_memory) {
        native = static_cast<ID3D12Resource*>(owner->own().real);
        D3D12_HEAP_PROPERTIES props{};
        D3D12_HEAP_FLAGS flags{};
        if (FAILED(native->GetHeapProperties(&props, &flags)) ||
            (props.Type != D3D12_HEAP_TYPE_DEFAULT && props.Type != D3D12_HEAP_TYPE_UPLOAD)) return {};
        upload = props.Type == D3D12_HEAP_TYPE_UPLOAD;
    }
    if (!native || offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT ||
        !InterlockedCompareExchange(&owner->own().refs, 0, 0)) return {};
    const auto* info = resource_info(owner);
    if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON) return {};
    x2w_startup::ComPtr<ID3D12Device> native_device;
    if (FAILED(native->GetDevice(IID_PPV_ARGS(native_device.GetAddressOf()))) || native_device.Get() != device_of(device)) return {};
    const auto desc = native->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || (desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) ||
        offset > desc.Width || bytes > desc.Width - offset) return {};
    return {native, offset, bytes, stride, upload};
}

struct FormattedBufferDescriptor {
    u64 entry = 0;
    std::uint32_t words[8] = {};
};

bool formatted_buffers_ready(const Translation& tr, const Pass& pass, FormattedBufferDescriptor (&descriptors)[32])
{
    const auto refuse = [&](int slot, const char* reason, std::uint32_t detail = 0) {
        static LONG refused;
        if (first_calls(refused, 6))
            log("x2w-draw %016llx srv %d: formatted buffer refused (%s %u)", tr.key, slot, reason, detail);
        return false;
    };
    for (const auto& ud : tr.uds) {
        if (ud.kind != Translation::desc_word3) continue;
        if (ud.a < 0 || ud.a >= 32 || static_cast<std::size_t>(ud.a) >= tr.srvs.size())
            return refuse(ud.a, "invalid SRV index");
        auto& descriptor = descriptors[ud.a];
        if (descriptor.entry) continue;
        const auto& srv = tr.srvs[ud.a];
        if (srv.kind != 0 || srv.source != 0 || srv.a < 0 || srv.a >= 16 || srv.b < 0)
            return refuse(ud.a, "missing hardware descriptor");
        const u64 entry = pass.view_tables[srv.a].record(static_cast<u64>(srv.b), sizeof(descriptor.words));
        if (!entry || !x2w_local::readable(reinterpret_cast<const void*>(entry), sizeof(descriptor.words)))
            return refuse(ud.a, "missing hardware descriptor");
        std::memcpy(descriptor.words, reinterpret_cast<const void*>(entry), sizeof(descriptor.words));
        if (descriptor.words[0] == view_token_magic)
            return refuse(ud.a, "missing hardware descriptor");
        const auto format = x2w_startup::formatted_buffer_code(descriptor.words[3]);
        if (!x2w_startup::formatted_buffer_supported(format))
            return refuse(ud.a, "unsupported FORMAT", format);
        descriptor.entry = entry;
    }
    return true;
}

bool draw_translated(Wrapper* self, bool indexed, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance,
                     TranslatedIndirectDraw* indirect)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    const bool rect = pass.topology == 0x11;
    const bool indexed_rect = rect && indexed && indirect;
    if (!list || !pass.pipeline || !pass.targets_bound || !(rect || pass.topology == 4 || pass.topology == 5) ||
        (rect && indexed && !indexed_rect)) return false;
    AcquireSRWLockExclusive(&draw_lock);
    BoundPipeline* bound = translation_for(pass.pipeline);
    Translation* tr = bound ? translation_for_targets(bound->translation, pass) : nullptr;
    if (indirect && tr) indirect->shader_key = tr->key;
    std::uint32_t d16_unorm16[32] = {};
    FormattedBufferDescriptor formatted_buffers[32] = {};
    const bool translation_ok = tr && tr->state == 1;
    const bool d16_ok = translation_ok && d16_srvs_ready(*tr, pass, d16_unorm16);
    const bool formatted_ok = d16_ok && formatted_buffers_ready(*tr, pass, formatted_buffers);
    if (!formatted_ok) {
        if (indirect && tr)
            indirect->refusal_code = !translation_ok ? indirect_refusal_code::translation_not_ready
                                     : !d16_ok       ? indirect_refusal_code::d16_srv_not_ready
                                                     : indirect_refusal_code::formatted_buffer_not_ready;
        ReleaseSRWLockExclusive(&draw_lock);
        return false;
    }
    if (indirect) {
        if (!bound->guest_root || bound->guest_root != pass.root) {
            indirect->set_refusal("translated draw requires the guest pipeline's root signature",
                                  indirect_refusal_code::missing_guest_root);
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
        if (indexed && pass.compute_pipeline) {
            indirect->set_refusal("indexed validation cannot replace a live guest compute pipeline",
                                  indirect_refusal_code::live_compute_pipeline);
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
        auto& entry = indexed ? tr->indexed_indirect_entry : tr->independent_indirect_entry;
        if (!entry.has_value()) {
            bool found_base = !indexed;
            if (indexed) {
                for (std::size_t i = 0; i < tr->uds.size(); ++i) {
                    if (tr->uds[i].kind != Translation::base_vertex) continue;
                    if (found_base) { found_base = false; break; }
                    found_base = true;
                    tr->indirect_base_word = static_cast<int>(i);
                }
            }
            entry = found_base && indirect_shader_entry(tr->vs, *tr, indexed) && indirect_shader_entry(tr->ps, *tr, indexed) &&
                std::all_of(tr->srvs.begin(), tr->srvs.end(), [indexed](const Translation::Srv& srv) {
                    return srv.source == 0 || (indexed && srv.kind == 0 && srv.source == 1 && srv.a == 0);
                });
        }
        if (!*entry || tr->samplers.heap) {
            indirect->set_refusal("translated draw uses an unsupported resource or draw-dependent entry value",
                                  indirect_refusal_code::unsupported_resource_or_entry);
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
    }
    ID3D12PipelineState* pipeline = ensure_draw_heap(self->own().list_device) ? pipeline_for(*bound, *tr, pass, indexed_rect) : nullptr;
    if (indirect && indexed_rect && !pipeline) {
        indirect->set_refusal("translated indexed rectangle geometry unavailable",
                              indirect_refusal_code::indexed_rect_unavailable);
    }
    if (!pipeline || (tr->samplers.heap && !prepare_draw_sampler_heap(pass, list, tr->samplers.reserved_fallback))) {
        if (indirect && !(indexed_rect && !pipeline))
            indirect->refusal_code = !pipeline ? indirect_refusal_code::pipeline_unavailable
                                               : indirect_refusal_code::sampler_heap_unavailable;
        ReleaseSRWLockExclusive(&draw_lock);
        return false;
    }
    const UINT n = static_cast<UINT>(tr->srvs.size());
    UINT first = static_cast<UINT>(InterlockedExchangeAdd(&draw_heap_cursor, static_cast<LONG>(n))) % draw_heap_size;
    if (first + n > draw_heap_size) {
        first = 0;
        InterlockedExchange(&draw_heap_cursor, static_cast<LONG>(n));
    }
    std::uint32_t strides[32] = {};
    // Only the vertices this draw can reference are copied: [vertex_first, vertex_first + vertex_count).
    // The shader adds the "base vertex" entry value to the vertex index, so that value is biased by
    // -vertex_first and each stream's view starts at vertex_first.
    const UINT index_bytes = pass.ib.Format == DXGI_FORMAT_R16_UINT ? 2 : 4;
    const u64 index_address = pass.ib.BufferLocation + u64{start} * index_bytes;
    const std::size_t index_total = std::size_t{count} * index_bytes;
    const bool indices_ok = indexed && !indirect && pass.ib.BufferLocation && readable_window(index_address, index_total) >= index_total;
    INT vertex_first = indexed ? 0 : static_cast<INT>(start);
    UINT vertex_count = indexed ? 0 : count;
    if (indices_ok) {
        std::uint32_t low = ~0u, high = 0;
        for (UINT k = 0; k < count; ++k) {
            const std::uint32_t value = index_bytes == 2 ? reinterpret_cast<const std::uint16_t*>(index_address)[k]
                                                         : reinterpret_cast<const std::uint32_t*>(index_address)[k];
            if (value < low) low = value;
            if (value > high) high = value;
        }
        if (low <= high) {
            vertex_first = base_vertex + static_cast<INT>(low);
            vertex_count = high - low + 1;
        }
    }
    if (vertex_first < 0) vertex_first = 0;
    UINT indexed_vertices = UINT_MAX;
    for (UINT i = 0; i < n; ++i) {
        const Translation::Srv& srv = tr->srvs[i];
        const D3D12_CPU_DESCRIPTOR_HANDLE dest{draw_heap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T{first + i} * draw_heap_increment};
        u64 address = 0, offset = 0;
        std::size_t bytes = 0;
        PlacedVertexView placed{};
        if (srv.source == 1 && srv.a >= 0 && srv.a < 8) {
            placed = pass.placed_vb[srv.a];
            address = pass.vb[srv.a].BufferLocation;
            bytes = pass.vb[srv.a].SizeInBytes;
            strides[i] = pass.vb[srv.a].StrideInBytes;
            const u64 skip = u64{static_cast<UINT>(vertex_first)} * strides[i];
            if (strides[i] && vertex_count && skip < bytes) {
                const u64 wanted = u64{vertex_count} * strides[i];
                address += skip;
                bytes = static_cast<std::size_t>(wanted < bytes - skip ? wanted : bytes - skip);
            }
        } else if (srv.source == 2 && srv.a < 16) {
            address = pass.root_address[srv.a];
            bytes = readable_window(address, 0x10000);
        }
        const auto& formatted = formatted_buffers[i];
        const u64 entry = formatted.entry ? formatted.entry :
            (srv.source == 0 && srv.a >= 0 && srv.a < 16 && srv.b >= 0 ?
                pass.view_tables[srv.a].record(static_cast<u64>(srv.b), 32) : 0);
        std::uint32_t words[8] = {};
        const bool entry_ok = formatted.entry ||
            (entry && x2w_local::readable(reinterpret_cast<const void*>(entry), 32));
        if (entry_ok) std::memcpy(words, formatted.entry ? formatted.words : reinterpret_cast<const void*>(entry), 32);
        bool buffer_token_known = false;
        if (srv.kind >= 1 && srv.kind <= 4) {
            const auto dimension = srv.kind == 1 ? D3D12_SRV_DIMENSION_TEXTURE2D :
                                   srv.kind == 2 ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY :
                                   srv.kind == 4 ? D3D12_SRV_DIMENSION_TEXTURE1D : D3D12_SRV_DIMENSION_TEXTURE3D;
            // A texture: the title's descriptor memory carries the id of the view it created there.
            ViewRecord record{};
            bool known = false;
            if (words[0] == view_token_magic) {
                AcquireSRWLockShared(&note_lock);
                known = words[1] < view_records().size();
                if (known) record = view_records()[words[1]];
                ReleaseSRWLockShared(&note_lock);
            }
            Wrapper* resource = known ? as_wrapper(reinterpret_cast<const void*>(record.resource)) : nullptr;
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.ViewDimension = dimension;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.Texture2D.MipLevels = 1;  // shared MostDetailedMip and MipLevels in all supported texture views
            if (srv.kind == 2) desc.Texture2DArray.ArraySize = 1;
            ID3D12Resource* texture = sampled_address_texture(resource);
            if (texture && record.desc_bytes >= 40) {
                std::memcpy(&desc, record.desc, 40);
                desc.Format = desktop_format(desc.Format);
            }
            if (texture && x2w_startup::packed_target_format(desc.Format)) {
                // The view recorded from the title names a packed format: follow a stand-in texture,
                // and bind nothing for a texture the view does not fit (packed_target_format.h).
                const auto verdict = x2w_startup::packed_view(desc.Format, texture->GetDesc().Format, x2w_startup::packed_stand_in_of(texture));
                if (verdict == x2w_startup::PackedView::stand_in) desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                if (verdict == x2w_startup::PackedView::refused) {
                    static LONG refused;
                    if (first_calls(refused, 6))
                        log("x2w-draw   %016llx srv %u: packed format %d on a texture of format %d that does not stand in for it; binding null",
                            tr->key, i, desc.Format, texture->GetDesc().Format);
                    texture = nullptr;
                    desc = {};
                    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                    desc.ViewDimension = dimension;
                    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    desc.Texture2D.MipLevels = 1;
                    if (srv.kind == 2) desc.Texture2DArray.ArraySize = 1;
                }
            }
            if (indirect && (!texture || record.slot != 18 || texture == pass.target0)) {
                indirect->set_refusal("translated draw requires a matching sampled texture distinct from the target",
                                      indirect_refusal_code::sampled_texture_mismatch);
                ReleaseSRWLockExclusive(&draw_lock);
                return false;
            }
            if (texture && !upload_texture(self, resource, texture)) texture = nullptr;
            if (texture) require_state(self, texture, sampled_state(self, texture));
            const bool had_texture = texture != nullptr;
            bool mismatched = false;
            static LONG shown;
            if (first_calls(shown, 6))
                log("x2w-draw   %016llx srv %u texture: table entry 0x%llx words %08x %08x -> view %s, resource %p, desc %u bytes format %d",
                    tr->key, i, entry, words[0], words[1], known ? "known" : "UNKNOWN", static_cast<void*>(texture), record.desc_bytes, desc.Format);
            const bool wrong_1d_resource = texture && srv.kind == 4 &&
                (texture->GetDesc().Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D || texture->GetDesc().DepthOrArraySize != 1);
            if (texture && ((record.desc_bytes >= 40 && desc.ViewDimension != dimension) || wrong_1d_resource)) {
                static LONG mismatch;
                if (first_calls(mismatch, 6)) log("x2w-draw   %016llx srv %u: view dimension %d does not match the shader; binding null", tr->key, i, desc.ViewDimension);
                texture = nullptr;
                mismatched = true;
                desc.ViewDimension = dimension;
                desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                desc.Texture2DArray = {};
                desc.Texture2D.MipLevels = 1;
                if (srv.kind == 2) desc.Texture2DArray.ArraySize = 1;
            }
            if (indirect && !texture) {
                indirect->set_refusal("translated draw could not bind its sampled texture",
                                      indirect_refusal_code::sampled_texture_unbindable);
                ReleaseSRWLockExclusive(&draw_lock);
                return false;
            }
            x2w_local::BindingFacts facts;
            facts.texture = facts.from_table = true;
            facts.entry_readable = entry_ok;
            facts.token_magic = words[0] == view_token_magic;
            facts.token_known = known;
            facts.resource_wrapper = had_texture;
            facts.dimension_matches = !mismatched;
            const auto cause = x2w_local::classify_null_binding(facts);
            count_null_binding(cause);
            if (cause == x2w_local::NullCause::unknown_token && token_null_census.note(tr->key)) {
                char line[256];
                x2w_startup::report_token_entry(line, sizeof(line), tr->key, i, entry, words);
                log("%s", line);
            }
            device_of(the_device)->CreateShaderResourceView(texture, texture && record.desc_bytes < 40 ? nullptr : &desc, dest);
            continue;
        }
        if (indirect) {
            IndirectBufferView view{};
            if (srv.source == 1 && indexed) {
                const auto& vb = pass.vb[0];
                if (vb.StrideInBytes && vb.StrideInBytes <= 0x3fff && vb.SizeInBytes % vb.StrideInBytes == 0 &&
                    vb.BufferLocation <= 0xffffffffffffull) {
                    const std::uint32_t stream[]{static_cast<UINT>(vb.BufferLocation),
                        static_cast<UINT>(vb.BufferLocation >> 32) | vb.StrideInBytes << 16, vb.SizeInBytes / vb.StrideInBytes, 0};
                    view = indirect_table_buffer(self, stream, true);
                    indexed_vertices = vb.SizeInBytes / vb.StrideInBytes;
                }
            } else if (entry_ok) {
                view = indirect_table_buffer(self, words);
            }
            if (!view.resource || view.resource == indirect->argument || view.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) {
                indirect->set_refusal(
                    srv.source == 1 ? "translated indexed draw requires a complete GPU-backed vertex stream"
                                    : "translated draw requires a complete GPU-backed table buffer range",
                    srv.source == 1 ? indirect_refusal_code::incomplete_vertex_stream
                                    : indirect_refusal_code::incomplete_table_buffer);
                ReleaseSRWLockExclusive(&draw_lock);
                return false;
            }
            if (!view.upload)
                require_state(self, view.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            strides[i] = view.stride;
            buffer_view(dest, view.resource, view.offset, static_cast<std::size_t>(view.bytes));
            continue;
        }
        if (srv.source == 0 && words[0] == view_token_magic && entry) {
            // A view the title created: a constant buffer view names its memory directly; a buffer
            // shader-resource view names a range of its resource's title memory.
            ViewRecord record{};
            AcquireSRWLockShared(&note_lock);
            buffer_token_known = words[1] < view_records().size();
            if (buffer_token_known) record = view_records()[words[1]];
            ReleaseSRWLockShared(&note_lock);
            if (record.slot == 17 && record.desc_bytes >= 12) {
                std::uint32_t size = 0;
                std::memcpy(&address, record.desc, 8);
                std::memcpy(&size, record.desc + 8, 4);
                bytes = size;
            } else if (record.slot == 18 && record.desc_bytes >= 40) {
                D3D12_SHADER_RESOURCE_VIEW_DESC view;
                std::memcpy(&view, record.desc, 40);
                Wrapper* resource = as_wrapper(reinterpret_cast<const void*>(record.resource));
                if (view.ViewDimension == D3D12_SRV_DIMENSION_BUFFER && resource && resource->own().title_memory) {
                    const bool typed_view = !view.Buffer.StructureByteStride && view.Format != DXGI_FORMAT_R32_TYPELESS;
                    const UINT element = view.Buffer.StructureByteStride ? view.Buffer.StructureByteStride
                                         : typed_view ? x2w_startup::typed_buffer_format_bytes(view.Format) : 4;
                    address = reinterpret_cast<u64>(resource->own().title_memory) + view.Buffer.FirstElement * element;
                    bytes = std::size_t{view.Buffer.NumElements} * element;
                    strides[i] = typed_view ? element : view.Buffer.StructureByteStride;
                }
            }
        }
        if (srv.source == 0 && words[0] != view_token_magic && entry) {
            // A hardware buffer descriptor the title wrote itself: base address, stride and record count.
            address = words[0] | (static_cast<u64>(words[1] & 0xffff) << 32);
            strides[i] = (words[1] >> 16) & 0x3fff;
            bytes = static_cast<std::size_t>(strides[i] ? strides[i] : 1) * words[2];
        }
        if (bytes > (16u << 20)) bytes = 16u << 20;
        // A buffer binding that ends up without a readable range counts as a null binding with its cause.
        const auto count_buffer = [&](bool bound) {
            x2w_local::BindingFacts buffer_facts;
            buffer_facts.from_table = srv.source == 0;
            buffer_facts.entry_readable = entry_ok;
            buffer_facts.token_magic = entry_ok && words[0] == view_token_magic;
            buffer_facts.token_known = buffer_token_known;
            buffer_facts.bound = bound;
            count_null_binding(x2w_local::classify_null_binding(buffer_facts));
        };
        if (placed.reject || (placed.owner && !InterlockedCompareExchange(&placed.owner->own().refs, 0, 0))) {
            buffer_view(dest, nullptr, 0, 0);
            count_buffer(false);
            continue;
        }
        if (placed.owner && bytes % 4 != 0) {
            // Preserve tight byte-sized streams. The desktop raw view is rounded
            // to a DWORD, so initialise its final padding without reading the guest.
            const auto padded = (bytes + 3) & ~std::size_t{3};
            auto* target = readable_window(address, bytes) >= bytes ? reserve_upload_ring(padded, &offset) : nullptr;
            if (target) {
                std::memcpy(target, reinterpret_cast<const void*>(address), bytes);
                std::memset(target + bytes, 0, padded - bytes);
            }
            buffer_view(dest, target ? upload_ring : nullptr, offset, padded);
            count_buffer(target != nullptr);
            continue;
        }
        if (placed.alias && address % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT == 0 && bytes) {
            require_state(self, placed.alias, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            buffer_view(dest, placed.alias, address - reinterpret_cast<u64>(placed.owner->own().title_memory), bytes);
            continue;
        }
        bool have = address && bytes && readable_window(address, bytes) >= bytes && frame_copy(address, bytes, &offset);
        // Experiment (X2W_DRAW_NULL_IDENTITY=1): an empty buffer slot reads as repeated identity
        // matrices instead of zeros. The start-up screen's last pass takes its world matrix from a
        // slot the title leaves empty here; why is not known yet.
        static const bool identity_for_null = x2w_local::env_on("X2W_DRAW_NULL_IDENTITY");
        if (!have && formatted.entry) {
            static LONG refused;
            if (first_calls(refused, 6))
                log("x2w-draw %016llx srv %u: formatted buffer refused (unresolved buffer)", tr->key, i);
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
        if (!have && identity_for_null) {
            static float identity[64];
            for (int k = 0; k < 64; ++k) identity[k] = (k % 16) % 5 == 0 ? 1.0f : 0.0f;
            address = reinterpret_cast<u64>(identity);
            bytes = sizeof(identity);
            have = frame_copy(address, bytes, &offset);
        }
        buffer_view(dest, have ? upload_ring : nullptr, offset, bytes);
        count_buffer(have);
        if (first_calls(tr->logged_bindings, 8)) {
            char head[3 * 48 + 1] = "";
            if (have) hex_text(reinterpret_cast<const void*>(address), bytes < 48 ? bytes : 48, head, sizeof(head));
            log("x2w-draw   %016llx srv %u buffer source %d(%d,%d): entry 0x%llx words %08x %08x %08x %08x -> address 0x%llx, %zu bytes, stride %u: %s [%s]",
                tr->key, i, srv.source, srv.a, srv.b, entry, words[0], words[1], words[2], words[3], address, bytes, strides[i],
                have ? "bound" : "NULL", head);
        }
    }
    static LONG checks;
    if ((InterlockedIncrement(&checks) & 63) == 1) check_device("translated draw views", 0);
    std::uint32_t values[32] = {};
    for (std::size_t i = 0; i < tr->uds.size(); ++i) {
        const Translation::Ud& ud = tr->uds[i];
        values[i] = ud.kind == Translation::rect ? (rect && !indexed_rect ? 1u : 0u)
            : ud.kind == Translation::base_vertex ? static_cast<std::uint32_t>((indexed ? base_vertex : static_cast<INT>(start)) - vertex_first)
            : ud.kind == Translation::start_instance ? start_instance
            : ud.kind == Translation::root_constant ? ((ud.a >= 0 && ud.a < 16 && ud.b >= 0 && ud.b < 8) ? pass.constants[ud.a][ud.b] : 0)
            : ud.kind == Translation::stride && ud.a >= 0 && ud.a < 32 ? strides[ud.a]
            : ud.kind == Translation::d16_unorm16 && ud.a >= 0 && ud.a < 32 ? d16_unorm16[ud.a]
            : ud.kind == Translation::desc_word3 ? formatted_buffers[ud.a].words[3] : 0;
    }
    // A desktop list with no viewport or scissor draws nothing: cover the first render target unless the title set its own.
    if (pass.target0 && (!pass.viewport_set || !pass.scissor_set)) {
        const D3D12_RESOURCE_DESC target = pass.target0->GetDesc();
        // Only the title's own calls set the flags; an automatic default is regenerated when render target 0 changes size.
        if (!pass.viewport_set && (pass.default_viewport_w != target.Width || pass.default_viewport_h != target.Height)) {
            const D3D12_VIEWPORT full{0.0f, 0.0f, static_cast<float>(target.Width), static_cast<float>(target.Height), 0.0f, 1.0f};
            list->RSSetViewports(1, &full);
            pass.default_viewport_w = target.Width;
            pass.default_viewport_h = target.Height;
        }
        if (!pass.scissor_set && (pass.default_scissor_w != target.Width || pass.default_scissor_h != target.Height)) {
            const D3D12_RECT full{0, 0, static_cast<LONG>(target.Width), static_cast<LONG>(target.Height)};
            list->RSSetScissorRects(1, &full);
            pass.default_scissor_w = target.Width;
            pass.default_scissor_h = target.Height;
        }
    }
    ID3D12Resource* validated = nullptr;
    if (indirect && indexed) {
        validated = validate_indexed_arguments(self, list, *tr, *indirect, indexed_vertices);
        if (!validated) {
            indirect->set_refusal("translated indexed argument validation could not be recorded",
                                  indirect_refusal_code::indexed_validation_failed);
            ReleaseSRWLockExclusive(&draw_lock);
            return false;
        }
    }
    list->SetPipelineState(pipeline);
    // X2W_DEPTH_STATE: the reference of the decoded stencil test is list state in D3D12. It replaces the one a
    // preceding OMSetStencilRef set, for draws of this pipeline only when a depth view is bound.
    const bool scoped_stencil_ref = bound->depth_from_registers && bound->has_stencil_ref &&
                                    pass_dsv_format(*bound, pass) != DXGI_FORMAT_UNKNOWN &&
                                    bound->stencil_ref != pass.stencil_reference;
    if (scoped_stencil_ref) list->OMSetStencilRef(bound->stencil_ref);
    list->SetGraphicsRootSignature(tr->root);
    ID3D12DescriptorHeap* samplers = tr->samplers.heap ? pass.sampler_heap.Get() : nullptr;
    if (!pass.heap_set || pass.bound_sampler_heap != samplers) {
        x2w_startup::bind_translated_heaps(list, draw_heap, samplers);
        pass.heap_set = true;
        pass.bound_sampler_heap = samplers;
    }
    list->SetGraphicsRoot32BitConstants(0, static_cast<UINT>(tr->uds.size()), values, 0);
    if (n) list->SetGraphicsRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{draw_heap->GetGPUDescriptorHandleForHeapStart().ptr + u64{first} * draw_heap_increment});
    if (samplers) list->SetGraphicsRootDescriptorTable(x2w_startup::sampler_root_parameter(n), samplers->GetGPUDescriptorHandleForHeapStart());
    list->IASetPrimitiveTopology(pass.topology == 5 ? D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP : D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    bool drawn = false;
    if (indirect) {
        if (validated) {
            list->IASetIndexBuffer(&indirect->indices.view);
            list->ExecuteIndirect(tr->indexed_signature.Get(), 1, validated, 4, validated, 0);
        } else {
            require_state(self, indirect->argument, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
            list->ExecuteIndirect(indirect->signature, 1, indirect->argument, indirect->offset, nullptr, 0);
        }
        // The translated root invalidated the desktop arguments of the tracked guest root.
        if (pass.graphics_root) list->SetGraphicsRootSignature(pass.graphics_root);
        pass.srv_bound = 0;
        drawn = true;
    } else if (indexed) {
        u64 offset = 0;
        const u64 address = index_address;
        const std::size_t bytes = index_total;
        if (indices_ok && frame_copy(address, bytes, &offset)) {
            const D3D12_INDEX_BUFFER_VIEW view{upload_ring->GetGPUVirtualAddress() + offset, static_cast<UINT>(bytes),
                                               index_bytes == 2 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT};
            list->IASetIndexBuffer(&view);
            const auto red_samples = red_producer_draw_before(self, list, "translated-draw", tr->key);
            list->DrawIndexedInstanced(count, instances ? instances : 1, 0, 0, start_instance);
            red_producer_draw_after(self, red_samples);
            drawn = true;
        }
    } else {
        const auto red_samples = red_producer_draw_before(self, list, "translated-draw", tr->key);
        list->DrawInstanced(rect ? count / 3 * 6 : count, instances ? instances : 1, 0, start_instance);
        red_producer_draw_after(self, red_samples);
        drawn = true;
    }
    if (scoped_stencil_ref) list->OMSetStencilRef(pass.stencil_reference);
    if (drawn && !indirect) draw_dump_after(self, list, tr->key, count);
    ReleaseSRWLockExclusive(&draw_lock);
    if (indirect) return drawn;
    if (first_calls(tr->logged_draws, 10) || (count > 100 && first_calls(tr->logged_large, 12))) {
        char indices[3 * 24 + 1] = "";
        if (indexed && pass.ib.BufferLocation) hex_text(reinterpret_cast<const void*>(pass.ib.BufferLocation + u64{start} * 2), 24, indices, sizeof(indices));
        log("x2w-draw %s(%u, %u, start %u, base %d) pipeline %016llx topology 0x%x -> %s; entry [%x %x %x %x %x %x]; ib 0x%llx+%u fmt %d [%s]; "
            "vb 0x%llx %u/%u; targets %u", indexed ? "indexed" : "draw", count, instances, start, base_vertex, tr->key, pass.topology,
            drawn ? "drawn" : "skipped (index data)", values[0], values[1], values[2], values[3], values[4], values[5], pass.ib.BufferLocation,
            pass.ib.SizeInBytes, pass.ib.Format, indices, pass.vb[0].BufferLocation, pass.vb[0].SizeInBytes, pass.vb[0].StrideInBytes,
            pass.target_count);
    }
    return drawn;
}

bool draw_translated(Wrapper* self, bool indexed, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance)
{
    return draw_translated(self, indexed, count, instances, start, base_vertex, start_instance, nullptr);
}

const char* draw_translated_indirect(Wrapper* self, ID3D12CommandSignature* signature, ID3D12Resource* argument, u64 offset,
                                     const IndirectIndexView& indices)
{
    TranslatedIndirectDraw draw{signature, argument, offset, indices};
    if (!draw_translated(self, indices.resource != nullptr, 0, 0, 0, 0, 0, &draw)) {
        static LONG refused;
        if (draw.shader_key) {
            const bool first_pair = indirect_draw_refusal_census.note(draw.shader_key, draw.refusal_code);
            if (first_pair || first_calls(refused, 32))
                log("x2w-render translated ExecuteIndirect Draw refused, shader %016llx: %s", draw.shader_key, draw.refusal);
        }
        return draw.refusal;
    }
    InterlockedIncrement(&translated_draws);
    static LONG reported;
    if (first_calls(reported, 16))
        log("x2w-render translated ExecuteIndirect %s recorded, shader %016llx, offset 0x%llx",
            indices.resource ? "DrawIndexed" : "Draw", draw.shader_key, offset);
    return nullptr;
}

// List slot 32, SetDescriptorHeaps. Track the title's shader-visible sampler
// source while translated draws use their own physical descriptor heaps.
void __stdcall list_set_descriptor_heaps(Wrapper* self, UINT count, void* const* heaps)
{
    x2w_startup::SamplerSource source;
    u64 view_heap = 0;
    if (count && count <= 4 && x2w_local::readable(heaps, std::size_t{count} * sizeof(*heaps))) {
        AcquireSRWLockShared(&note_lock);
        for (UINT i = 0; i < count; ++i) {
            for (const HeapInfo& heap : heap_infos()) {
                if (heap.wrapper == reinterpret_cast<u64>(heaps[i]) &&
                    heap.type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV &&
                    heap.logical_count > heap.count)
                    view_heap = heap.wrapper;
                if (heap.wrapper == reinterpret_cast<u64>(heaps[i]) &&
                    heap.type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER && heap.desktop_gpu)
                    source = {static_cast<std::uintptr_t>(heap.title_sampler_cpu), heap.count};
            }
        }
        ReleaseSRWLockShared(&note_lock);
    }
    auto& pass = pass_of(self);
    if (pass.source_view_heap != view_heap)
        for (auto& table : pass.view_tables) table = {};
    pass.source_view_heap = view_heap;
    pass.title_samplers = source;
}

// List slot 36 (desktop 32 + 4), SetGraphicsRootDescriptorTable(parameter, GPU handle): the table's
// address in the title's descriptor memory is kept for the draw.
void __stdcall list_table(Wrapper* self, UINT parameter, u64 gpu_handle)
{
    if (parameter >= 16) return;
    auto& pass = pass_of(self);
    x2w_startup::ViewTable table;
    AcquireSRWLockShared(&note_lock);
    for (const HeapInfo& heap : heap_infos()) {
        if (heap.type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) continue;
        if (pass.source_view_heap && heap.wrapper != pass.source_view_heap) continue;
        if (!pass.source_view_heap && heap.logical_count > heap.count) continue;
        table = x2w_startup::resolve_view_table(heap.desktop_gpu, heap.title_cpu, heap.logical_count, heap.increment, gpu_handle);
        if (table.address) break;
    }
    ReleaseSRWLockShared(&note_lock);
    pass.view_tables[parameter] = table;
}
// List slot 40 (desktop 36 + 4), SetGraphicsRoot32BitConstants(parameter, count, data, offset).
void __stdcall list_root_constants(Wrapper* self, UINT parameter, UINT count, const std::uint32_t* data, UINT offset)
{
    static LONG dropped;
    if (count && (parameter >= 16 || offset >= 8 || count > 8 - offset) && first_calls(dropped, 4))
        log("x2w-render root constants %s: parameter %u, offset %u, count %u (16 parameters of 8 words are kept)",
            parameter >= 16 || offset >= 8 ? "dropped" : "truncated", parameter, offset, count);
    if (parameter >= 16 || offset >= 8 || !count || !x2w_local::readable(data, std::size_t{count} * 4)) return;
    Pass& pass = pass_of(self);
    for (UINT i = 0; i < count && offset + i < 8; ++i) pass.constants[parameter][offset + i] = data[i];
    if (pass.root && parameter < pass.root->size() &&
        (*pass.root)[parameter].type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS &&
        count <= 8 - offset && offset + count <= (*pass.root)[parameter].constants_count)
        pass.constant_words_bound[parameter] |= static_cast<unsigned char>(((1u << count) - 1) << offset);
}
// List slot 47 (desktop 43 + 4), IASetIndexBuffer(view).
void __stdcall list_index_buffer(Wrapper* self, const D3D12_INDEX_BUFFER_VIEW* view)
{
    pass_of(self).ib = view && x2w_local::readable(view, sizeof(*view)) ? *view : D3D12_INDEX_BUFFER_VIEW{};
}
// List slot 48 (desktop 44 + 4), IASetVertexBuffers(start, count, views).
void __stdcall list_vertex_buffers(Wrapper* self, UINT start, UINT count, const D3D12_VERTEX_BUFFER_VIEW* views)
{
    if (!views || !x2w_local::readable(views, std::size_t{count} * sizeof(*views))) return;
    if (start >= 8) return;
    Pass& pass = pass_of(self);
    for (UINT i = 0; i < count && i < 8 - start; ++i) {
        pass.vb[start + i] = views[i];
        pass.placed_vb[start + i] = placed_buffer_range(self->own().list_device, views[i].BufferLocation, views[i].SizeInBytes);
    }
}
// List slot 13, DrawIndexedInstanced(index count, instances, start index, base vertex, start instance).
void __stdcall list_draw_indexed(Wrapper* self, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance)
{
    InterlockedIncrement(draw_translated(self, true, count, instances, start, base_vertex, start_instance) ? &translated_draws : &untranslated_draws);
}
