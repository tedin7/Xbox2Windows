// Translated compute programs at Dispatch (list slot 14). A device slot 67 pipeline is a placeholder that keeps
// the key of its published request; gfx2hlsl.py --pending answers the request with <key>.txt (`stage cs`) and
// <key>.cs.cso. Bound from the compute tables recorded by slot 35 and compute root SRVs recorded by slot 43: raw
// buffers (`srv <i> buf table <param> <byte>` and `srv <i> buf root <param>` inputs, `uav <i> table <param> <byte>`
// outputs) named by hardware buffer descriptors or 64-bit root SRV addresses, and textures
// (`srv <i> tex2d table <param> <byte>` inputs, `uav <i> tex1d/tex2d/tex2da/tex3d float table <param> <byte> channels <n>` outputs) named
// by the token of a view the title created (device slots 18 and 19). `smp`, `smpl` and `smpc` lines become static samplers. Entry
// values come from the compute root constants of slots 37 and 39, or from the descriptor a buffer input or output is
// bound from: its stride and word 3 (`stride`, `descword3`), and word 3, the upper half of word 1 and the record
// count of an output (`uav_descword3`, `uav_stride`, `uav_numrecords`). Any other manifest line makes the translation
// unavailable; a dispatch that cannot be bound completely is refused with one reason and records no Dispatch and
// binds nothing (the upload of an input texture the title filled in memory may already be recorded: it only
// syncs content).

const char* volume_uav_range_error(const D3D12_RESOURCE_DESC& resource, const D3D12_TEX3D_UAV& view)
{
    if (view.MipSlice >= resource.MipLevels) return "an output 3D texture mip is outside the resource";
    const UINT depth = (std::max)(1u, UINT{resource.DepthOrArraySize} >> (std::min)(view.MipSlice, 15u));
    if (view.FirstWSlice >= depth || !view.WSize || (view.WSize != ~0u && view.WSize > depth - view.FirstWSlice))
        return "an output 3D texture W range is outside the mip";
    return nullptr;
}

struct ComputeTranslation {
    // Root parameter and byte offset (views), or parameter and word (entry values). A heap view names the root
    // constant (parameter and word) that indexes its 32-byte record in the list's view heap, or, with word_param
    // set, the table buffer (word_param, word_byte) whose word at byte at + root constant (a, b) * times (a < 0:
    // at alone) is the index. A fixed heap view uses at directly as a byte offset. Non-root entry values name a
    // descriptor by source and slot number.
    enum Source { root_constant, input_stride, input_word3, output_word3, output_stride, output_records };
    struct Slot {
        int a, b;
        bool texture, heap;
        int word_param = -1, word_byte = 0;
        std::uint32_t at = 0, times = 0;
        Source source = root_constant;
        D3D12_SRV_DIMENSION srv_dimension = D3D12_SRV_DIMENSION_UNKNOWN;
        bool load = false;  // typed UAV load for a partial image store
        D3D12_UAV_DIMENSION uav_dimension = D3D12_UAV_DIMENSION_UNKNOWN;
        bool root = false;
        bool atomic_uint_texture = false;
        unsigned channels = 4;
    };
    u64 key = 0;
    int state = 0;                    // 0 not loaded, 1 ready, 2 unavailable
    x2w_startup::ManifestRetry manifest_retry;  // absent and stale manifests can become ready later
    std::vector<Slot> uds, srvs, uavs;
    bool dynamic_heap = false;
    bool dynamic_heap_formats = false;
    x2w_startup::IntegerTextureSrvs integer_srvs;
    x2w_startup::TranslatedSamplers samplers;  // static ones only
    x2w_startup::ComPtr<ID3D12RootSignature> root;
    x2w_startup::ComPtr<ID3D12PipelineState> pipeline;
    const char* unmodelled = nullptr;      // why a manifest that exists cannot be used
    u64 dispatches = 0, translated = 0;    // census: Dispatch calls with this key bound, and those recorded
    KeyUse use;                            // when a Dispatch first and last had this key bound
    const char* first_refusal = nullptr;   // reason of the first refused call
    bool unfilled_heap_slot_logged = false;
    bool input_path_logged = false;        // the first time a placed input came from a GPU copy or a title snapshot
    std::uint32_t refusal_words[4] = {};   // the table descriptor that reason was about, if any; a view token's
                                           // last word is replaced by its view format | dimension << 16
    ~ComputeTranslation()
    {
        // The per-process cache is destroyed during DLL detach. A driver may already have shut
        // down when the process is terminating. Explicit unload still releases both references.
        if (x2w_startup::d3d12_process_terminating) {
            pipeline.Detach();
            root.Detach();
        }
    }
};

// Device support is adapter-specific. The shim keeps a singleton desktop device.
unsigned compute_output_format_support(ID3D12Device* device, DXGI_FORMAT format)
{
    static std::mutex mutex;
    static ID3D12Device* cached_device = nullptr;
    static unsigned bits[256]{};
    static bool filled[256]{};
    const unsigned index = static_cast<unsigned>(format);
    if (index >= 256) return 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (cached_device != device) {
        std::memset(filled, 0, sizeof(filled));
        cached_device = device;
    }
    if (!filled[index]) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
        bits[index] = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)))
            ? static_cast<unsigned>(support.Support2) : 0;
        filled[index] = true;
    }
    return bits[index];
}

// Component counts of the float and normalized formats accepted for typed outputs.
unsigned float_output_channels(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_SNORM: case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_SNORM: case DXGI_FORMAT_B8G8R8A8_UNORM:
        return 4;
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return 3;
    case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_SNORM: case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_SNORM:
        return 2;
    case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_SNORM: case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_SNORM:
        return 1;
    default:
        return 0;
    }
}
// X2W_INDIRECT_DISPATCH_TEXTURES=1: a translated indirect Dispatch also binds input textures (off: fixed-table raw buffers only).
bool indirect_dispatch_textures_enabled()
{
    static const bool enabled = x2w_local::env_on("X2W_INDIRECT_DISPATCH_TEXTURES");
    return enabled;
}
std::unordered_map<u64, ComputeTranslation>& compute_translations() { static std::unordered_map<u64, ComputeTranslation> m; return m; }
LONG compute_dispatches_translated, compute_dispatches_refused, compute_dispatches_untranslated, compute_dispatches_unmodelled;
bool compute_texture_srv_range_fits(const D3D12_SHADER_RESOURCE_VIEW_DESC& view, const D3D12_RESOURCE_DESC& resource)
{
    const auto mips_fit = [&](UINT first, UINT count) {
        return first < resource.MipLevels && count && (count == UINT_MAX || count <= resource.MipLevels - first);
    };
    if (resource.SampleDesc.Count != 1) return false;
    switch (view.ViewDimension) {
    case D3D12_SRV_DIMENSION_TEXTURE1D:
        return resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && resource.DepthOrArraySize == 1 &&
            mips_fit(view.Texture1D.MostDetailedMip, view.Texture1D.MipLevels);
    case D3D12_SRV_DIMENSION_TEXTURE2D:
        return resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && resource.DepthOrArraySize == 1 &&
            mips_fit(view.Texture2D.MostDetailedMip, view.Texture2D.MipLevels);
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY: {
        const auto& array = view.Texture2DArray;
        return resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
            mips_fit(array.MostDetailedMip, array.MipLevels) && array.FirstArraySlice < resource.DepthOrArraySize &&
            array.ArraySize && array.ArraySize <= resource.DepthOrArraySize - array.FirstArraySlice;
    }
    case D3D12_SRV_DIMENSION_TEXTURE3D:
        return resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D &&
            mips_fit(view.Texture3D.MostDetailedMip, view.Texture3D.MipLevels);
    case D3D12_SRV_DIMENSION_TEXTURECUBE:
        return resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && resource.DepthOrArraySize == 6 &&
            mips_fit(view.TextureCube.MostDetailedMip, view.TextureCube.MipLevels);
    default:
        return false;
    }
}

// An output given as the token of an unordered access view of a buffer: bind the token's own desktop resource over the
// view's range and give the program the descriptor words 1 to 3 the hardware descriptor would have held (stride, records
// and the unified FORMAT with identity channel selection), so a program that reads them runs too.
const char* compute_uav_token(Wrapper* self, const std::uint32_t* words, Wrapper*& wrapper,
                              ID3D12Resource*& resource, u64& offset, u64& bytes, std::uint32_t (&out)[4], u64 entry, u64 key)
{
    ViewRecord record{};
    bool materialized = false;
    if (!read_view_token(words, record, &materialized, entry, key))
        return "an output view token has an unknown kind and record";
    if (record.slot == 18) return "an output shader resource view token cannot supply unordered access";
    if (record.slot != 19) return "an output view token is not an unordered access view";
    if (!materialized || record.desc_bytes < sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC))
        return "an output unordered access view token has no materialized description";

    D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
    std::memcpy(&desc, record.desc, sizeof(desc));
    if (desc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER)
        return "an output unordered access view token describes a texture, but the shader needs a buffer";
    if (record.counter || desc.Buffer.CounterOffsetInBytes)
        return "an output unordered access view token has a counter";

    wrapper = as_wrapper(reinterpret_cast<const void*>(record.resource));
    if (!wrapper || wrapper->own().kind != resource_kind || !wrapper->own().real)
        return "an output unordered access view token has no desktop resource";
    resource = static_cast<ID3D12Resource*>(wrapper->own().real);
    const auto native = resource->GetDesc();
    x2w_startup::ComPtr<ID3D12Device> native_device;
    const auto* info = resource_info(wrapper);
    if (native.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || !(native.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ||
        !info || info->initial_state != D3D12_RESOURCE_STATE_COMMON ||
        FAILED(resource->GetDevice(IID_PPV_ARGS(native_device.GetAddressOf()))) ||
        native_device.Get() != device_of(self->own().list_device))
        return "an output unordered access view token needs a desktop UAV buffer created in common state on this device";

    const DXGI_FORMAT format = desktop_format(desc.Format);
    const UINT stride = desc.Buffer.StructureByteStride;
    const bool raw = desc.Buffer.Flags == D3D12_BUFFER_UAV_FLAG_RAW && !stride && format == DXGI_FORMAT_R32_TYPELESS;
    const bool structured = desc.Buffer.Flags == D3D12_BUFFER_UAV_FLAG_NONE && stride && format == DXGI_FORMAT_UNKNOWN;
    const bool typed = desc.Buffer.Flags == D3D12_BUFFER_UAV_FLAG_NONE && !stride && format != DXGI_FORMAT_UNKNOWN;
    const UINT element = raw ? 4 : structured ? stride : typed ? x2w_startup::typed_buffer_format_bytes(format) : 0;
    if (!element || element > 0x3fff)
        return "an output unordered access view token is not a raw, structured or typed buffer view of a known element size";
    const UINT code = typed ? x2w_startup::buffer_format_code(format) : 0;
    if (typed && !code) return "an output typed buffer view has a format with no decoded buffer FORMAT";
    if (desc.Buffer.FirstElement > native.Width / element)
        return "an output unordered access view token starts outside its desktop buffer";
    offset = desc.Buffer.FirstElement * element;
    bytes = u64{desc.Buffer.NumElements} * element;
    if (!bytes || bytes > (16u << 20) || bytes % 4 || bytes > native.Width - offset || offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)
        return "an output unordered access view token has an invalid buffer range";

    out[0] = 0;
    out[1] = raw ? 0 : element << 16;
    out[2] = raw ? static_cast<std::uint32_t>(bytes) : desc.Buffer.NumElements;
    out[3] = 0xfacu | (raw ? 3u << 28 : 0) | (code ? (1u << 24) | (code << 12) : 0);
    return nullptr;
}

LONG compute_inputs_plain;
LONG compute_inputs_gpu_copied, compute_inputs_snapshot, compute_inputs_snapshot_kib;  // inputs by path: this list's GPU copy, a title snapshot

// Both wrappers and the prefetch worker use stable entries, even after a wrapper is released.
struct ComputeManifest {
    x2w_startup::ManifestPrefetch::Publication publication;
    std::once_flag probe;
};

bool nolock_compute_refuse()
{
    static const bool enabled = x2w_local::env_on("X2W_NOLOCK_REFUSE");
    return enabled;
}

ComputeManifest& compute_manifest_for(u64 key)
{
    static auto* const entries = new std::unordered_map<u64, ComputeManifest>;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    return (*entries)[key];
}

bool compute_manifest_missing(ComputeManifest& manifest, u64 key)
{
    using State = x2w_startup::ManifestPrefetch::Publication::State;
    auto& publication = manifest.publication;
    const State state = publication.state.load(std::memory_order_acquire);
    if (state != State::unprobed) return state == State::missing;
    // Competing callers for this key wait for the same first read with draw_lock free.
    std::call_once(manifest.probe, [&] {
        auto& source = manifest_source();
        if (source.read_first(key, publication.text))
            publication.state.store(State::ready, std::memory_order_release);
        else
            source.prefetch().watch_missing(key, publication);
    });
    return publication.state.load(std::memory_order_acquire) == State::missing;
}

// The root signature and pipeline of a translation whose entry values and views are filled in. Returns why it is
// unavailable, or null when it is ready.
const char* create_compute_pipeline(ComputeTranslation& tr, const void* code, std::size_t bytes)
{
    // Parameter 0 keeps one constant for a program without entry values: a constants parameter cannot be empty.
    const UINT entries = static_cast<UINT>(tr.uds.size() + tr.integer_srvs.twins.size());
    const UINT constants = (std::max)(1u, entries + (tr.dynamic_heap ? 1u : 0u));
    if (FAILED(create_title_translated_root(device_of(the_device), constants,
            static_cast<UINT>(tr.srvs.size()) + tr.integer_srvs.descriptors(), tr.samplers, &tr.root, static_cast<UINT>(tr.uavs.size()), tr.dynamic_heap)))
        return "root signature creation failed";
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = tr.root.Get();
    desc.CS = {code, bytes};
    if (FAILED(device_of(the_device)->CreateComputePipelineState(&desc, IID_PPV_ARGS(&tr.pipeline))))
        return "pipeline creation failed";
    tr.state = 1;
    return nullptr;
}

// draw_lock held. Returns the reason the translation is unavailable, or null when it is ready or still absent.
const char* load_compute_translation(ComputeTranslation& tr, ComputeManifest* prefetched)
{
    tr.state = 2;
    wchar_t path[MAX_PATH];
    std::string manifest;
    if (prefetched) {
        // The manifest is ready and draw_lock serializes its consumption and PSO creation.
        manifest = std::move(prefetched->publication.text);
    } else if (!cache_path(path, L"", tr.key, L".txt") || !lookup_manifest(tr.key, tr.manifest_retry.pending(), manifest, true)) {
        tr.manifest_retry.missed(GetTickCount64());
        return nullptr;
    }
    // A native manifest from another compiler revision, or an unmarked one without the Python translator's
    // <key>.cs.hlsl, is stale: it is looked for again on the same schedule as an absent one.
    const auto output = x2w::native_compute_output(manifest);
    if (output == x2w::NativeComputeOutput::incompatible) {
        tr.manifest_retry.missed(GetTickCount64());
        return "native translation from another compiler revision; rebuild it";
    }
    if (output == x2w::NativeComputeOutput::unmarked) {
        const DWORD attributes = cache_path(path, L"", tr.key, L".cs.hlsl")
            ? GetFileAttributesW(path) : INVALID_FILE_ATTRIBUTES;
        if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
            tr.manifest_retry.missed(GetTickCount64());
            return "native translation without a revision; rebuild it";
        }
    }
    tr.manifest_retry.found();
    std::string approx_marker;
    const u64 output_key = translation_output_key(manifest, tr.key, approx_marker);
    unsigned declared_uds = 0, line_number = 0;
    for (std::size_t at = 0; at < manifest.size(); ++line_number) {
        std::size_t end = manifest.find('\n', at);
        if (end == std::string::npos) end = manifest.size();
        std::string line = manifest.substr(at, end - at);
        at = end + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line_number == 0) {
            if (line != "x2w-manifest 2") return "manifest version";
            continue;
        }
        if (line_number == 1) {
            if (line != "stage cs") return "not a compute manifest";
            continue;
        }
        if (line.compare(0, 7, "native ") == 0) continue;
        unsigned i = 0, x = 0, y = 0, z = 0, w = 0;
        int a = 0, b = 0, c = 0, d = 0, used = 0;
        char name[16] = "";
        const auto whole = [&] { return used == static_cast<int>(line.size()); };
        // The manifest names of the Source values after root_constant, in their order.
        const auto source = [&] {
            static const char* const names[] = {"stride", "descword3", "uav_descword3", "uav_stride", "uav_numrecords"};
            for (int s = 0; s < 5; ++s)
                if (!std::strcmp(name, names[s])) return static_cast<ComputeTranslation::Source>(s + 1);
            return ComputeTranslation::root_constant;
        };
        if (line == "dynamic_heap") { tr.dynamic_heap = true; continue; }
        if (line == "dynamic_heap_formats") { tr.dynamic_heap_formats = true; continue; }
        const auto input_dimension = [&] {
            return !std::strcmp(name, "tex2d") ? D3D12_SRV_DIMENSION_TEXTURE2D :
                !std::strcmp(name, "texcube") ? D3D12_SRV_DIMENSION_TEXTURECUBE :
                tr.integer_srvs.typed && !std::strcmp(name, "tex1d") ? D3D12_SRV_DIMENSION_TEXTURE1D :
                tr.integer_srvs.typed && !std::strcmp(name, "tex2da") ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY :
                tr.integer_srvs.typed && !std::strcmp(name, "tex3d") ? D3D12_SRV_DIMENSION_TEXTURE3D :
                D3D12_SRV_DIMENSION_UNKNOWN;
        };
        const auto uav_dimension = [&] {
            if (!std::strcmp(name, "tex1d")) return D3D12_UAV_DIMENSION_TEXTURE1D;
            if (!std::strcmp(name, "tex2d")) return D3D12_UAV_DIMENSION_TEXTURE2D;
            if (!std::strcmp(name, "tex2da")) return D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            if (!std::strcmp(name, "tex3d")) return D3D12_UAV_DIMENSION_TEXTURE3D;
            return D3D12_UAV_DIMENSION_UNKNOWN;
        };
        if (line.empty() || (std::sscanf(line.c_str(), "group %u %u %u%n", &x, &y, &z, &used) == 3 && whole())) continue;
        if (std::sscanf(line.c_str(), "ud %u%n", &declared_uds, &used) == 1 && whole()) continue;
        if (line == "srvtypes 1") {
            tr.integer_srvs.typed = true;
            continue;
        }
        if (line.compare(0, 7, "srvint ") == 0) {
            if (!tr.integer_srvs.read_twin(line)) return "invalid srvint";
            continue;
        }
        if (std::sscanf(line.c_str(), "udsrc %u rootconst %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.uds.size())
            tr.uds.push_back({a, b});
        else if (std::sscanf(line.c_str(), "udsrc %u %15s %d%n", &i, name, &a, &used) == 3 && whole() && i == tr.uds.size() &&
                 source() != ComputeTranslation::root_constant)
            tr.uds.push_back({a, 0, false, false, -1, 0, 0, 0, source()});
        else if (std::sscanf(line.c_str(), "srv %u buf table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.srvs.size())
            tr.srvs.push_back({a, b});
        else if (std::sscanf(line.c_str(), "srv %u buf root %d%n", &i, &a, &used) == 2 && whole() && a >= 0 && a < 64 &&
                 i == tr.srvs.size()) {
            tr.srvs.push_back({a, 0});
            tr.srvs.back().root = true;
        }
        else if (std::sscanf(line.c_str(), "srv %u buf heap fixed %u%n", &i, &x, &used) == 2 && whole() &&
                 tr.dynamic_heap && i == tr.srvs.size() && x <= 0xfffff && !(x % 32))
            tr.srvs.push_back({-1, 0, false, true, -1, 0, x});
        else if (std::sscanf(line.c_str(), "srv %u buf heap rootconst %d %d%n", &i, &a, &b, &used) == 3 && whole() &&
                 i == tr.srvs.size())
            tr.srvs.push_back({a, b, false, true});
        else if (std::sscanf(line.c_str(), "srv %u buf heap word table %d %d at %u%n", &i, &a, &b, &x, &used) == 4 && whole() &&
                 i == tr.srvs.size())
            tr.srvs.push_back({-1, 0, false, true, a, b, x});
        else if (std::sscanf(line.c_str(), "srv %u buf heap word table %d %d at %u rootconst %d %d times %u%n", &i, &a, &b, &x,
                     &c, &d, &y, &used) == 7 && whole() && i == tr.srvs.size())
            tr.srvs.push_back({c, d, false, true, a, b, x, y});
        else if (std::sscanf(line.c_str(), "uav %u table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.uavs.size())
            tr.uavs.push_back({a, b});
        else if (std::sscanf(line.c_str(), "uav %u tex2d uint table %d %d channels 1%n", &i, &a, &b, &used) == 3 &&
                 whole() && i == tr.uavs.size()) {
            tr.uavs.push_back({a, b, true});
            tr.uavs.back().uav_dimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            tr.uavs.back().atomic_uint_texture = true;
        }
        else if (std::sscanf(line.c_str(), "uav %u %15s float table %d %d channels %u%n", &i, name, &a, &b, &x, &used) == 5 &&
                 whole() && i == tr.uavs.size() && x >= 1 && x <= 4 && uav_dimension() != D3D12_UAV_DIMENSION_UNKNOWN &&
                 (x == 4 || uav_dimension() != D3D12_UAV_DIMENSION_TEXTURE1D)) {
            tr.uavs.push_back({a, b, true});
            tr.uavs.back().uav_dimension = uav_dimension();
            tr.uavs.back().channels = x;
        }
        else if (std::sscanf(line.c_str(), "uavload %u%n", &i, &used) == 1 && whole() &&
                 i < tr.uavs.size() && tr.uavs[i].texture && !tr.uavs[i].load)
            tr.uavs[i].load = true;
        else if (std::sscanf(line.c_str(), "srv %u %15s table %d %d%n", &i, name, &a, &b, &used) == 4 &&
                 whole() && i == tr.srvs.size() && input_dimension() != D3D12_SRV_DIMENSION_UNKNOWN) {
            tr.srvs.push_back({a, b, true});
            tr.srvs.back().srv_dimension = input_dimension();
        }
        // A static sampler, decoded as the draw path does: plain (smp), with the descriptor's LOD clamp (smpl) or a
        // comparison one (smpc). Not modelled here: the run-time sampler heap table (smpheap).
        else if (line.rfind("smpheap", 0) == 0) return "a sampler from the sampler heap";
        else if ((std::sscanf(line.c_str(), "smp %u %x %x %x %x%n", &i, &x, &y, &z, &w, &used) == 5 ||
                  std::sscanf(line.c_str(), "smpl %u %x %x %x %x%n", &i, &x, &y, &z, &w, &used) == 5 ||
                  std::sscanf(line.c_str(), "smpc %u %x %x %x %x%n", &i, &x, &y, &z, &w, &used) == 5) && whole() &&
                 i == tr.samplers.statics.size())
            tr.samplers.read_manifest_line(line.c_str());
        else {
            // The first line no pattern took (the whole line, not only its kind), so the next run names the form.
            const char* const why = line.rfind("srv", 0) == 0 ? "a resource that is not a raw table buffer, a 2D texture or a cube map"
                  : line.rfind("uav", 0) == 0 ? "an output that is not a raw table buffer or a 1D, 2D, 2D-array or 3D float texture"
                  : line.rfind("udsrc", 0) == 0 ? "an entry value that is not a root constant or a buffer descriptor word"
                  : line.rfind("smp", 0) == 0 ? "a sampler that is not a plain static one" : "an unknown manifest line";
            log("x2w-draw compute pipeline %016llx manifest line %u rejected (%s; %zu inputs and %zu outputs read so far): %.120s", tr.key,
                line_number, why, tr.srvs.size(), tr.uavs.size(), line.c_str());
            return why;
        }
    }
    if ((tr.dynamic_heap_formats && !tr.dynamic_heap) || tr.uds.size() != declared_uds || tr.uds.size() > 32 || tr.srvs.size() > 32 || tr.uavs.size() > 8)
        return "entry value, input or output count";
    for (const ComputeTranslation::Slot& ud : tr.uds) {
        const auto& slots = ud.source >= ComputeTranslation::output_word3 ? tr.uavs : tr.srvs;
        if (ud.source != ComputeTranslation::root_constant &&
            (ud.a < 0 || static_cast<std::size_t>(ud.a) >= slots.size() || slots[ud.a].texture || slots[ud.a].root))
            return "an entry value names a descriptor that is not a buffer input or output";
    }
    if (!tr.integer_srvs.valid(static_cast<UINT>(tr.srvs.size()), static_cast<UINT>(tr.uds.size()),
                              [&](UINT i) { return tr.srvs[i].texture && tr.srvs[i].srv_dimension != D3D12_SRV_DIMENSION_TEXTURECUBE; }))
        return "invalid srvint mapping";
    x2w_startup::ComPtr<ID3DBlob> code;
    if (!cache_path(path, L"", output_key, L".cs.cso") || FAILED(D3DReadFileToBlob(path, &code))) return "no compiled program";
    if (const char* failure = create_compute_pipeline(tr, code->GetBufferPointer(), code->GetBufferSize())) return failure;
    note_approx_translation(output_key, approx_marker);
    log("x2w-draw compute pipeline %016llx translated: %zu entry values, %zu inputs, %zu outputs", tr.key, tr.uds.size(),
        tr.srvs.size(), tr.uavs.size());
    return nullptr;
}

// A copy of an input texture made on the recording before a dispatch whose output is the same texture: the shader reads
// the copy (what the title's input read) while the output unordered access view writes the original. Null when the copy
// cannot be made or kept with the recording (the dispatch is refused then). The copy has no render or unordered-access flags.
LONG compute_aliased_input_copies;
ID3D12Resource* snapshot_input_texture(Wrapper* self, ID3D12Resource* source)
{
    auto* list = recording(self);
    if (!list || !source) return nullptr;
    D3D12_RESOURCE_DESC desc = source->GetDesc();
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || desc.SampleDesc.Count > 1) return nullptr;
    desc.Flags &= ~(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
    desc.Alignment = 0;
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    x2w_startup::ComPtr<ID3D12Resource> copy;
    if (FAILED(device_of(the_device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                              nullptr, IID_PPV_ARGS(&copy)))) return nullptr;
    ID3D12Resource* const held = copy.Get();
    if (!retain_with_recording(self, std::move(copy))) return nullptr;
    require_state(self, source, D3D12_RESOURCE_STATE_COPY_SOURCE);
    require_state(self, held, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyResource(held, source);
    static LONG logged;
    if (first_calls(logged, 4)) log("x2w-draw compute input texture copied before its own output is written");
    return held;
}

// X2W_NATIVE_SHADERS=<directory of dxcompiler.dll and dxil.dll>: the first Dispatch of a slot 67 request translates
// it in this process with the C++ compiler (src/gfx1013_compute_native.cpp), before a published <key>.txt is read
// and with no offline translator running. The dispatch path below binds the result, so a program is accepted only
// with root constants, raw table buffers and exact-width typed texture outputs.
// A refusal is logged and leaves the key to the published translations. draw_lock held. Returns whether tr is ready.
bool translate_compute_natively(ComputeTranslation& tr)
{
#ifdef X2W_NATIVE_SHADERS
    wchar_t sdk[MAX_PATH], path[MAX_PATH];
    const DWORD length = GetEnvironmentVariableW(L"X2W_NATIVE_SHADERS", sdk, MAX_PATH);
    if (!length || length >= MAX_PATH) return false;
    if (!cache_path(path, L"req_", tr.key, L".c67")) return false;
    try {
        // The request as this runtime published it. The reader checks the six parts against the key in the name.
        // Entry 0: a source-built program starts there, and the frontend reads a capture's v_nop prefix as code.
        auto program = x2w::translate_compute(x2w::read_compute_request(path), 0, true);
        for (const auto& block : program.constants)
            for (unsigned word = 0; word < block.count; ++word)
                tr.uds.push_back({static_cast<int>(block.root_parameter), static_cast<int>(word)});
        for (unsigned srv : program.stride_srvs) {
            const bool present = std::any_of(program.bindings.begin(), program.bindings.end(), [srv](const auto& view) {
                return view.kind == x2w::ComputeViewKind::raw_vector_srv && view.shader_register == srv;
            });
            if (!present) throw std::runtime_error("descriptor stride source is not a bound SRV");
            ComputeTranslation::Slot slot{};
            slot.a = static_cast<int>(srv);
            slot.source = ComputeTranslation::input_stride;
            tr.uds.push_back(slot);
        }
        std::sort(program.bindings.begin(), program.bindings.end(),
                  [](const auto& a, const auto& b) { return a.shader_register < b.shader_register; });
        for (const auto& view : program.bindings) {
            const unsigned channels = x2w::exact_texture_uav_channels(view.kind);
            const bool output = channels || view.kind == x2w::ComputeViewKind::raw_buffer_uav;
            auto& slots = output ? tr.uavs : tr.srvs;
            if ((!output && view.kind != x2w::ComputeViewKind::raw_vector_srv) || view.array_count != 1 ||
                view.root_parameter >= 16 || view.shader_register != slots.size())
                throw std::runtime_error("view kind " + std::to_string(static_cast<unsigned>(view.kind)) +
                                         " is not bound for a natively translated program");
            slots.push_back({static_cast<int>(view.root_parameter), static_cast<int>(view.byte_offset)});
            if (channels) {
                auto& slot = slots.back();
                slot.texture = true;
                slot.channels = channels;
                slot.uav_dimension = view.image_dimension == 2 ? D3D12_UAV_DIMENSION_TEXTURE3D :
                                     view.image_dimension == 5 ? D3D12_UAV_DIMENSION_TEXTURE2DARRAY :
                                                                 D3D12_UAV_DIMENSION_TEXTURE2D;
            }
        }
        if (!program.samplers.empty() || tr.uds.size() > 32 || tr.srvs.size() > 32 || tr.uavs.size() > 8)
            throw std::runtime_error("a sampler, or more entry values, inputs or outputs than the dispatch path binds");
        // A direct module declares wave size 32 (src/dxil_module.h): without it the pipeline cannot be created.
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 waves{};
        if (FAILED(device_of(the_device)->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &waves, sizeof(waves))) ||
            !waves.WaveOps || waves.WaveLaneCountMin > 32 || waves.WaveLaneCountMax < 32)
            throw std::runtime_error("the adapter has no 32-lane waves");
        // Never released: a static destructor would release DXC after its module may have detached.
        static x2w::ShaderDxilCompiler* const compiler =
            new x2w::ShaderDxilCompiler(x2w::ShaderDxilCompiler::runtime_worker(sdk));
        const auto shader = compiler->compile(program.dxil_ir, L"cs_6_6", std::filesystem::path(path).parent_path() / L"native",
                                              x2w::ShaderSourceFormat::dxil_ir);
        if (const char* failure = create_compute_pipeline(tr, shader.dxil.data(), shader.dxil.size()))
            throw std::runtime_error(failure);
        log("x2w-draw compute pipeline %016llx translated natively: %zu entry values, %zu inputs, %zu outputs", tr.key,
            tr.uds.size(), tr.srvs.size(), tr.uavs.size());
        return true;
    } catch (const std::exception& error) {
        tr.uds.clear();
        tr.srvs.clear();
        tr.uavs.clear();
        tr.root.Reset();
        tr.pipeline.Reset();
        static LONG logged;
        if (first_calls(logged, 16))
            log("x2w-draw compute pipeline %016llx not translated natively: %s", tr.key, error.what());
        return false;
    }
#else
    (void)tr;
    return false;
#endif
}

// Resolve a compute root SRV (`srv <i> buf root <param>`) from the guest address recorded by slot 43 for <param>
// to the end of its backing upload allocation or aliased placed buffer (up to the 16 MiB compute raw buffer limit).
IndirectBufferView compute_root_srv_buffer(Wrapper* self, const Pass& pass, int param, std::uint32_t (&words)[8],
                                           const char** why)
{
    const auto refuse = [why](const char* reason) {
        if (why) *why = reason;
        return IndirectBufferView{};
    };
    if (why) *why = nullptr;
    const auto* root_params = pass.compute_root_wrapper ? root_info(pass.compute_root_wrapper) : nullptr;
    if (!root_params || param < 0 || static_cast<std::size_t>(param) >= root_params->size() ||
        static_cast<std::size_t>(param) >= pass.compute_root_srvs.size() ||
        (*root_params)[param].type != D3D12_ROOT_PARAMETER_TYPE_SRV)
        return refuse("a compute root input is not a shader resource view parameter");
    if (!(pass.compute_root_srv_set & (u64{1} << param)))
        return refuse("a compute root SRV was not set since the compute root");
    const u64 address = pass.compute_root_srvs[param];
    words[0] = static_cast<std::uint32_t>(address);
    words[1] = static_cast<std::uint32_t>(address >> 32);
    if (!address) return refuse("a compute root SRV has a null address");
    if (address % 4) return refuse("a compute root SRV address is not 4-byte aligned");
    constexpr u64 limit = 16u << 20;
    if (const auto allocation = list_upload_tail(self, address); allocation.resource) {
        if (allocation.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)
            return refuse("a compute root SRV range is not 16-byte aligned in its resource");
        const u64 bytes = (std::min)(allocation.bytes, limit) & ~u64{3};
        if (bytes < 4)
            return refuse("a compute root SRV has fewer than 4 aligned bytes in its upload allocation");
        words[2] = static_cast<std::uint32_t>(bytes);
        return {allocation.resource, allocation.offset, bytes, 0, true};
    }
    Wrapper* device = self->own().list_device;
    const auto first = placed_buffer_range(device, address, 4);
    if (first.reject) return refuse("a compute root SRV crosses the end of a placed buffer");
    if (!first.owner) return refuse("a compute root SRV is in title memory outside any placed buffer");
    if (!InterlockedCompareExchange(&first.owner->own().refs, 0, 0))
        return refuse("a compute root SRV's resource is released");
    const auto* info = resource_info(first.owner);
    if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON)
        return refuse("a compute root SRV's resource was not created in the common state");
    const u64 base = reinterpret_cast<u64>(first.owner->own().title_memory);
    if (!base || address < base || address - base >= info->bytes)
        return refuse("a compute root SRV range is outside its desktop buffer");
    const u64 offset = address - base;
    if (offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)
        return refuse("a compute root SRV range is not 16-byte aligned in its resource");
    const u64 bytes = (std::min)(info->bytes - offset, limit) & ~u64{3};
    if (bytes < 4)
        return refuse("a compute root SRV has fewer than 4 aligned bytes in its placed buffer");
    const auto placed = placed_buffer_range(device, address, bytes);
    if (placed.reject || placed.owner != first.owner)
        return refuse("a compute root SRV crosses the end of a placed buffer");
    auto* native = static_cast<ID3D12Resource*>(placed.owner->own().real);
    if (!placed.alias && (!native || !list_copied_gpu_range(self, native, offset, bytes)))
        return refuse("a compute root SRV is in a placed buffer that does not alias title memory");
    x2w_startup::ComPtr<ID3D12Device> native_device;
    if (!native || FAILED(native->GetDevice(IID_PPV_ARGS(native_device.GetAddressOf()))) ||
        native_device.Get() != device_of(device))
        return refuse("a compute root SRV's resource is on another device");
    const auto desc = native->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || (desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) ||
        offset > desc.Width || bytes > desc.Width - offset)
        return refuse("a compute root SRV range is outside its desktop buffer");
    words[2] = static_cast<std::uint32_t>(bytes);
    return {native, offset, bytes, 0, false};
}

// Both callers have an open list and a slot 67 request. The indirect caller has checked the signature and argument range.
x2w_phase::Totals dispatch_phase_totals;

// Every 300 frames with the render totals: where translated dispatches spent their time since the last line.
void log_dispatch_phases()
{
    const auto window = dispatch_phase_totals.take();
    if (!window.calls[x2w_phase::direct] && !window.calls[x2w_phase::indirect] && !window.calls[x2w_phase::refused]) return;
    char line[1024];
    x2w_phase::format_window(window, line, sizeof(line));
    log("%s", line);
}

const char* record_translated_dispatch(Wrapper* self, const TranslatedDispatch& dispatch)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    const auto& pipeline = pass.pipeline->own();
    const u64 key = pipeline.compute_request_key;
    const auto [x, y, z] = dispatch.groups;
    const auto type = list->GetType();
    ComputeManifest* const manifest = pipeline.compute_manifest;
    const bool valid_type = type == D3D12_COMMAND_LIST_TYPE_DIRECT || type == D3D12_COMMAND_LIST_TYPE_COMPUTE;
    const bool valid_groups = x <= x2w_startup::max_dispatch_groups && y <= x2w_startup::max_dispatch_groups &&
                              z <= x2w_startup::max_dispatch_groups;
    const bool valid_root = pass.compute_root_wrapper == pipeline.compute_request_root;
    if (manifest && valid_type && valid_groups && valid_root && compute_manifest_missing(*manifest, key)) {
        InterlockedIncrement(&compute_dispatches_untranslated);
        note_refused_compute_dispatch(key, "no translation");
        return "no translation";
    }
    const char* reason = nullptr;
    LONG* counter = &compute_dispatches_refused;
    struct View {
        ID3D12Resource* resource;
        u64 offset, bytes;
        bool transition, texture, described;
        union {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv;
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav;
        } desc;  // a texture view's description, as the title gave it with a desktop format
        Wrapper* wrapper;
        std::uint32_t words[4];  // a buffer's descriptor words as it was bound from, and its stride
        UINT stride;
        bool aliased_output;          // an input texture that one of the dispatch's outputs also names
        ID3D12Resource* snapshot;     // its copy from before the dispatch, bound as the input (held by the recording)
    } views[40]{};
    bool formatted_inputs[32]{};  // input count is bounded by the manifest parser
    static_assert(sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC) >= ViewRecord::srv_desc_bytes);
    static_assert(sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC) >= ViewRecord::uav_desc_bytes);
    x2w_phase::Timer timer;
    timer.start(dispatch_phase_totals);
    draw_lock_acquire_exclusive();
    timer.got_lock();
    DispatchPhases phases;
    ComputeTranslation* const tr = &compute_translations()[key];
    tr->key = key;
    ++tr->dispatches;
    tr->use.note(process_ms());
    std::uint32_t words[8] = {};
    if (!valid_type) reason = "the list type has no Dispatch";
    else if (!valid_groups) reason = "a group count is over the limit";
    else if (!valid_root)
        reason = "the compute root set is not the one the pipeline was created with";
    else if (!ensure_draw_heap(self->own().list_device)) reason = "no descriptor heap on this device";
    else {
        if (!tr->state || tr->manifest_retry.due(GetTickCount64())) {
            // The in-process translation is tried once per key, before a published one.
            if (tr->state || !translate_compute_natively(*tr)) {
                tr->unmodelled = load_compute_translation(*tr, manifest);
                if (tr->unmodelled) log("x2w-draw compute pipeline %016llx not translated: %s", key, tr->unmodelled);
            }
        }
        if (tr->state != 1) {
            // A manifest the runtime cannot bind is the dispatch side's work; an absent one is the translator's.
            reason = tr->unmodelled ? tr->unmodelled : "no translation";
            counter = tr->unmodelled ? &compute_dispatches_unmodelled : &compute_dispatches_untranslated;
        }
    }
    // A word no slot 37 or 39 call wrote since the compute root was set has no value to pass (it is not 0).
    for (std::size_t i = 0; !reason && i < tr->uds.size(); ++i)
        if (tr->uds[i].source == ComputeTranslation::root_constant && !pass.compute_constants.is_bound(tr->uds[i].a, tr->uds[i].b, 1))
            reason = "an entry value was not set since the compute root";
    const std::size_t inputs = reason ? 0 : tr->srvs.size(), count = reason ? 0 : inputs + tr->uavs.size();
    if (!reason)
        for (const auto& ud : tr->uds)
            if (ud.source == ComputeTranslation::input_word3) formatted_inputs[ud.a] = true;
    phases.to(1);
    for (std::size_t i = 0; i < count && !reason; ++i) {
        const bool output = i >= inputs;
        const ComputeTranslation::Slot slot = output ? tr->uavs[i - inputs] : tr->srvs[i];
        // An input texture is bound exactly as for a direct Dispatch; only the group counts come from the GPU.
        const bool indirect_input_texture = indirect_dispatch_textures_enabled() && !output && slot.texture &&
                                            !slot.heap && !slot.root;
        if (dispatch.signature && (slot.heap || slot.texture || slot.root) && !indirect_input_texture) {
            reason = "translated indirect Dispatch requires fixed-table raw buffers";
            break;
        }
        std::uint32_t index = slot.a < 0 || slot.b < 0 ? 0 : pass.compute_constants.word(slot.a, slot.b);
        std::memset(words, 0, sizeof(words));
        if (!output && slot.root) {
            // A root constant buffer view (dynamic-heap programs only; other programs refuse it below): its address was recorded by the list's slot 43/44 calls
            // and must lie in this list's upload memory. A root SRV goes through the checks below.
            const auto* root_params = pass.compute_root_wrapper ? root_info(pass.compute_root_wrapper) : nullptr;
            if (tr->dynamic_heap && root_params && slot.a >= 0 && static_cast<std::size_t>(slot.a) < root_params->size() &&
                static_cast<std::size_t>(slot.a) < pass.compute_root_addresses.size() &&
                (*root_params)[slot.a].type == D3D12_ROOT_PARAMETER_TYPE_CBV) {
                const auto range = list_upload_range(self, pass.compute_root_addresses[slot.a], 1, true);
                if (!range.resource || range.bytes < 4 || range.bytes > (16u << 20) ||
                    range.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)
                    reason = "a root input is not an aligned range in this list's upload memory";
                else views[i] = {range.resource, range.offset, range.bytes & ~u64{3}, false};
                continue;
            }
            const IndirectBufferView view = compute_root_srv_buffer(self, pass, slot.a, words, &reason);
            if (reason && !tr->first_refusal)
                log("x2w-draw compute %016llx first refused input %zu (root %d), address 0x%llx", key, i, slot.a,
                    u64{words[0]} | (u64{words[1]} << 32));
            views[i] = {view.resource, view.offset, view.bytes, !view.upload};
            views[i].stride = view.stride;
            std::memcpy(views[i].words, words, sizeof(views[i].words));
            continue;
        }
        if (slot.word_param >= 0) {
            // The index word is read now from the recording's upload memory, where the title wrote it (no GPU command
            // writes there), at the dword-aligned 32-bit byte offset the program reads it at.
            const u64 table_entry = replay_record::draw_table_entry(pass.compute_tables.table(slot.word_param),
                                                                   static_cast<u64>(slot.word_byte), 16);
            if (!table_entry || !x2w_local::readable(reinterpret_cast<const void*>(table_entry), 16)) {
                reason = "a table descriptor is not bound";
                break;
            }
            std::memcpy(words, reinterpret_cast<const void*>(table_entry), 16);
            if ((reason = refuse_native_buffer_addressing(words))) break;
            const std::uint32_t at = slot.at + index * slot.times;
            const u64 address = words[0] | (u64{words[1] & 0xffff} << 32);
            if (!(words[0] | words[1] | words[2] | words[3])) {
                // A null buffer descriptor (all four words zero, no records): every read of it is out of range and the
                // hardware returns zero, so the program's index word is 0 and the dispatch reads heap record 0.
                index = 0;
            } else {
                if (words[0] == view_token_magic || !list_upload_range(self, address + at, 4).resource) {
                    reason = "a heap index word is not in the list's upload memory";
                    break;
                }
                if (u64{at} + 4 > x2w_startup::buffer_descriptor_bytes(words[1], words[2], words[3])) {
                    reason = "a heap index word is outside its table buffer";
                    break;
                }
                replay_record::draw_pages_note(reinterpret_cast<u64>(self), address + at, 4, false);
                std::memcpy(&index, reinterpret_cast<const void*>(address + at), 4);
            }
        }
        // A heap record sits at the 32-bit byte offset the program computes, the index shifted left by 5, in the
        // title memory of the list's view heap: a literal byte offset, as a table's is.
        const UINT heap_offset = slot.word_param >= 0 || slot.a >= 0 ? std::uint32_t{index << 5} : slot.at;
        const u64 entry = slot.heap ? pass.title_view_heap.record(heap_offset, 16)
            : slot.a < 0 || slot.b < 0 ? 0 : replay_record::draw_table_entry(pass.compute_tables.table(slot.a),
                                                                           static_cast<u64>(slot.b), slot.texture ? 32 : 16);
        const std::size_t entry_bytes = slot.texture ? 32 : 16;
        if (!entry || !x2w_local::readable(reinterpret_cast<const void*>(entry), entry_bytes)) {
            reason = slot.heap ? "a heap record is outside the list's view heap" : "a table descriptor is not bound";
            break;
        }
        std::memcpy(words, reinterpret_cast<const void*>(entry), entry_bytes);
        if (slot.texture) {
            // The title's descriptor memory carries the id of the view it created there. Nothing else is bound:
            // a null view would read zeros or drop the program's stores without a trace.
            ViewRecord record{};
            bool created = false;
            const bool known = read_view_token(words, record, &created, entry, key);
            Wrapper* const wrapper = known ? as_wrapper(reinterpret_cast<const void*>(record.resource)) : nullptr;
            ID3D12Resource* const texture = !wrapper || wrapper->own().kind != resource_kind ? nullptr
                : output ? static_cast<ID3D12Resource*>(wrapper->own().real) : sampled_address_texture(wrapper);
            const D3D12_RESOURCE_DESC resource = texture ? texture->GetDesc() : D3D12_RESOURCE_DESC{};
            if (!output) {
                D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
                const bool described = record.desc_bytes >= ViewRecord::srv_desc_bytes;
                if (described) std::memcpy(&desc, record.desc, ViewRecord::srv_desc_bytes);
                if (desc.Format) desc.Format = desktop_format(desc.Format);
                words[3] = static_cast<std::uint32_t>(desc.Format ? desc.Format : resource.Format) | static_cast<std::uint32_t>(desc.ViewDimension) << 16;
                if (!known || record.slot != 18) reason = "an input texture entry is not the token of a shader resource view";
                else if (!texture || (resource.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
                    reason = "an input texture has no desktop texture to sample";
                else if (slot.srv_dimension == D3D12_SRV_DIMENSION_TEXTURECUBE &&
                         (!described || desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURECUBE))
                    reason = "an input texture view is not a cube map";
                else if (slot.srv_dimension == D3D12_SRV_DIMENSION_TEXTURECUBE &&
                         (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.DepthOrArraySize != 6))
                    reason = "a cube map view's texture is not six 2D faces";
                else if (slot.srv_dimension == D3D12_SRV_DIMENSION_TEXTURE2D &&
                         (described ? desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D
                                    : resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.DepthOrArraySize != 1))
                    reason = "an input texture view is not 2D";
                else if (slot.srv_dimension == D3D12_SRV_DIMENSION_TEXTURE1D &&
                         (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D || resource.DepthOrArraySize != 1 ||
                          (described && desc.ViewDimension != slot.srv_dimension)))
                    reason = "an input texture view is not 1D";
                else if (slot.srv_dimension == D3D12_SRV_DIMENSION_TEXTURE3D &&
                         (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D ||
                          (described && desc.ViewDimension != slot.srv_dimension)))
                    reason = "an input texture view is not 3D";
                else if (slot.srv_dimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY &&
                         (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
                          (described ? desc.ViewDimension != slot.srv_dimension : resource.DepthOrArraySize <= 1)))
                    reason = "an input texture view is not a 2D array";
                else if (compute_view_tokens_enabled() && described &&
                         !compute_texture_srv_range_fits(desc, resource))
                    reason = "an input texture shader resource view token has a range outside its desktop resource";
                // shortcut: a packed format needs the stand-in rules of the draw path; add them with its first user.
                else if (x2w_startup::packed_target_format(desc.Format)) reason = "an input texture view has a packed format";
                views[i] = {texture, 0, 0, true, true, described};
                views[i].desc.srv = desc;
                views[i].wrapper = wrapper;
                continue;
            }
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            const bool described = record.desc_bytes >= ViewRecord::uav_desc_bytes;
            if (described) std::memcpy(&desc, record.desc, ViewRecord::uav_desc_bytes);
            if (desc.Format) desc.Format = desktop_format(desc.Format);
            const DXGI_FORMAT format = desc.Format ? desc.Format : resource.Format;
            words[3] = static_cast<std::uint32_t>(format) | static_cast<std::uint32_t>(desc.ViewDimension) << 16;
            if (!known || record.slot != 19) reason = "an output texture entry is not the token of an unordered access view";
            else if (slot.uav_dimension == D3D12_UAV_DIMENSION_TEXTURE3D &&
                     texture && (resource.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
                const auto& view = desc.Texture3D;
                if (!described || desc.ViewDimension != slot.uav_dimension || resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D)
                    reason = "an output texture view or resource is not 3D";
                else {
                    // Slot 19 can retain a rejected range as an unmaterialized view.
                    reason = volume_uav_range_error(resource, view);
                    if (!reason && !created) reason = "an output texture view was not created on the desktop";
                }
            } else if (slot.uav_dimension == D3D12_UAV_DIMENSION_TEXTURE1D &&
                       texture && (resource.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
                // Checked before `created`, as for 3D: slot 19 records a 1D view with an out-of-range mip as refused.
                if (!described || desc.ViewDimension != slot.uav_dimension || resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D)
                    reason = "an output texture view or resource is not 1D";
                else if (desc.Texture1D.MipSlice >= resource.MipLevels)
                    reason = "an output 1D texture mip is outside the resource";
                else if (!created)
                    reason = "an output texture view was not created on the desktop";
            } else if (!created || !texture || !(resource.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
                reason = "an output texture view was not created on the desktop";
            else reason = x2w_startup::compute_output_shape_reason(resource, desc, described,
                                                                slot.uav_dimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY, 0);
            if (!reason && x2w_startup::packed_target_format(format)) reason = "an output texture view has a packed format";
            if (!reason && slot.atomic_uint_texture && format != DXGI_FORMAT_R32_UINT)
                reason = "an image atomic requires an R32_UINT UAV";
            if (!reason && slot.channels < 4) {
                if (float_output_channels(format) != slot.channels || float_output_channels(resource.Format) != slot.channels)
                    reason = "an exact-width image store requires a matching output texture channel count";
                else {
                    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{resource.Format};
                    if (FAILED(device_of(the_device)->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support,
                                                                         sizeof(support))) ||
                        !(support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE))
                        reason = "an exact-width image store requires typed UAV store support";
                    desc.Format = resource.Format;
                }
            }
            if (!reason && !slot.atomic_uint_texture && !float_output_channels(format))
                reason = "an output texture format is not a float or normalized one";
            if (!reason) {
                const unsigned bits = compute_output_format_support(device_of(self->own().list_device), format);
                reason = x2w_startup::compute_output_support_reason(bits, slot.load);
            }
            // A texture bound as both SRV and UAV needs two states at once. The input is bound from a copy taken before
            // the dispatch (below), which reads what the title's own input read. Two outputs of one texture (two mips)
            // share the unordered access state.
            for (std::size_t input = 0; input < inputs && !reason; ++input)
                if (views[input].wrapper == wrapper) views[input].aliased_output = true;
            views[i] = {texture, 0, 0, true, true, described};
            views[i].desc.uav = desc;
            views[i].wrapper = wrapper;
            continue;
        }
        if (!output) {
            // A heap token supplies the view format and mapping. Native heap records keep their
            // word verbatim, including unknown formats handled by the generated switch.
            if (slot.heap && formatted_inputs[i] && words[0] == view_token_magic) {
                std::uint32_t format = 0;
                if (const char* refusal = formatted_buffer_word3(words, format, extended_buffer_formats())) {
                    ViewRecord record{};
                    read_view_token(words, record, nullptr, entry, key);
                    reason = refusal == x2w_startup::undecoded_view_format
                        ? "a formatted input view has no buffer format code"
                        : "a formatted input has no word 3 the program decodes";
                    break;
                }
            }
            // Each condition of a complete GPU-backed buffer range has its own reason.
            const IndirectBufferView view = indirect_table_buffer(self, words, false, &reason, 16u << 20, true, entry, key);
            if (unfilled_heap_slot(words) && !tr->unfilled_heap_slot_logged) {
                tr->unfilled_heap_slot_logged = true;
                log("x2w-draw unfilled heap slot %u key %016llx", words[0] - 1, key);
            }
            if (view.source == 1) InterlockedIncrement(&compute_inputs_gpu_copied);
            if (view.source == 4) InterlockedIncrement(&compute_inputs_plain);
            if (view.source == 2) {
                InterlockedIncrement(&compute_inputs_snapshot);
                InterlockedExchangeAdd(&compute_inputs_snapshot_kib, static_cast<LONG>((view.bytes + 1023) / 1024));
            }
            if (view.source && !tr->input_path_logged) {
                tr->input_path_logged = true;
                log("x2w-draw compute %016llx input %zu (%s %d %d) comes from %s (%llu bytes)", key, i, slot.heap ? "heap" : "table", slot.a, slot.b,
                    view.source == 1 ? "a GPU copy this list made" : view.source == 3 ? "zeros for a null descriptor" :
                    view.source == 4 ? "plain title memory" : "a snapshot of title memory", view.bytes);
            }
            if (view.resource && view.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) reason = "an input buffer range is not 16-byte aligned in its resource";
            if (reason && !tr->first_refusal)  // once per key: which input, and where its descriptor was read
                log("x2w-draw compute %016llx first refused input %zu (%s %d %d), descriptor at 0x%llx", key, i,
                    slot.heap ? "heap" : "table", slot.a, slot.b, entry);
            // buffer_view creates a native null SRV when the token has no resource.
            views[i] = {view.resource, view.offset, view.bytes, !view.upload && !view.null_token};
            views[i].stride = view.stride;
            std::memcpy(views[i].words, words, sizeof(views[i].words));
            continue;
        }
        if (words[0] == view_token_magic && compute_view_tokens_enabled()) {
            auto& view = views[i];
            reason = compute_uav_token(self, words, view.wrapper, view.resource, view.offset, view.bytes, view.words, entry, key);
            for (std::size_t input = 0; input < inputs && !reason; ++input)
                if (views[input].resource == view.resource)
                    reason = "an output unordered access view token names a buffer that is also an input";
            continue;
        }
        // Native buffer descriptors must cover a placement that aliases title memory.
        const u64 address = words[0] | (u64{words[1] & 0xffff} << 32);
        const UINT stride = (words[1] >> 16) & 0x3fff;
        const u64 bytes = x2w_startup::buffer_descriptor_bytes(words[1], words[2], words[3]);
        const bool sized = words[0] != view_token_magic && bytes && bytes % 4 == 0 && bytes <= (16u << 20);
        const auto placed = sized ? placed_buffer_range(self->own().list_device, address, bytes) : PlacedVertexView{};
        const ResourceInfo* info = placed.alias ? resource_info(placed.owner) : nullptr;
        const u64 offset = placed.alias ? address - reinterpret_cast<u64>(placed.owner->own().title_memory) : 0;
        // One reason per condition, so a census says which one an output failed on.
        if (words[0] == view_token_magic) reason = "an output buffer descriptor is a view token";
        else if (!bytes) {}  // no records: a null view, whose writes are dropped as a D3D12 null descriptor drops them
        else if (!sized) reason = "an output buffer range is over 16 MiB or not whole dwords";
        else if (placed.reject) reason = "an output range crosses the end of a placed buffer";
        else if (!placed.owner) reason = "an output is in title memory outside any placed buffer";
        else if (!placed.alias) reason = "an output is in a placed buffer that does not alias title memory";
        else if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON)
            reason = "an output buffer's resource was not created in the common state";
        else if (offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) reason = "an output buffer range is not 16-byte aligned in its resource";
        else if (!(placed.alias->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            reason = "an output buffer has no unordered access";
        static LONG size_logged;
        if (reason && !sized && first_calls(size_logged, 16))
            log("x2w-draw compute %016llx output %zu has %llu bytes (stride %u, records %u): %s", key, i, static_cast<unsigned long long>(bytes),
                stride, words[2], reason);
        const bool null_output = !bytes && words[0] != view_token_magic;
        for (std::size_t input = 0; input < inputs && !reason && !null_output; ++input)
            if (views[input].resource == placed.alias) reason = "an output buffer is also an input";
        views[i] = {placed.alias, offset, bytes, true};
        std::memcpy(views[i].words, words, sizeof(views[i].words));
    }
    timer.resolved();
    if (dispatch.argument)
        for (std::size_t i = 0; i < count && !reason; ++i)
            if (views[i].resource == dispatch.argument) reason = "the indirect argument buffer is also a bound shader resource";
    phases.to(2);
    // An entry value from a descriptor comes from the words its view was bound from in the loop above.
    std::uint32_t values[32] = {};
    for (std::size_t i = 0; !reason && i < tr->uds.size(); ++i) {
        const ComputeTranslation::Slot& ud = tr->uds[i];
        const View& view = views[ud.source >= ComputeTranslation::output_word3 ? inputs + ud.a : ud.a];
        std::uint32_t format = 0, word[4];
        switch (ud.source) {
        case ComputeTranslation::root_constant: values[i] = pass.compute_constants.word(ud.a, ud.b); break;
        case ComputeTranslation::input_stride: values[i] = view.stride; break;
        case ComputeTranslation::input_word3:
            if (tr->srvs[ud.a].heap) {
                values[i] = view.words[3];
                break;
            }
            std::memcpy(word, view.words, sizeof(word));
            if (formatted_buffer_word3(word, format, extended_buffer_formats())) {
                reason = "a formatted input has no word 3 the program decodes";
                std::memcpy(words, view.words, sizeof(view.words));
            }
            values[i] = word[3];
            break;
        case ComputeTranslation::output_word3: values[i] = view.words[3]; break;
        case ComputeTranslation::output_stride: values[i] = view.words[1] >> 16; break;  // unmasked: the program checks the swizzle bits
        case ComputeTranslation::output_records: values[i] = view.words[2]; break;
        }
    }
    phases.to(3);
    // A texture the title filled in memory is brought up to date first. An upload recorded before a later one
    // fails only syncs content, as a draw sampling that texture would.
    for (std::size_t i = 0; i < inputs && !reason; ++i)
        if (views[i].texture && !upload_texture(self, views[i].wrapper, views[i].resource)) {
            reason = "an input texture could not be uploaded";
            std::memset(words, 0, sizeof(words));
        }
    // An input that an output also names is read from a copy of it made before the dispatch writes.
    for (std::size_t i = 0; i < inputs && !reason; ++i) {
        if (!views[i].aliased_output) continue;
        views[i].snapshot = snapshot_input_texture(self, views[i].resource);
        if (!views[i].snapshot) {
            reason = "an input texture could not be copied before its own output";
            std::memset(words, 0, sizeof(words));
        } else {
            InterlockedIncrement(&compute_aliased_input_copies);
        }
    }
    ID3D12DescriptorHeap* view_heap = draw_heaps[0].heap;
    UINT first = 0, heap_count = 0;
    if (!reason && tr->dynamic_heap) {
        if (dispatch.signature) reason = "translated indirect Dispatch requires fixed-table raw buffers";
        else {
            view_heap = snapshot_buffer_heap(self, static_cast<UINT>(count) + tr->integer_srvs.descriptors(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                [&](ID3D12Resource* resource) {
                    for (std::size_t i = inputs; i < count; ++i)
                        if (views[i].resource == resource) return true;
                    return false;
                }, tr->dynamic_heap_formats, heap_count, false);
            first = heap_count + 1;
            if (!view_heap) reason = "the dynamic resource heap could not be bound";
        }
    }
    const UINT twins = static_cast<UINT>(tr->integer_srvs.twins.size());
    const UINT extra_srvs = tr->integer_srvs.descriptors();
    const UINT n = static_cast<UINT>(count) + extra_srvs;
    if (!reason && !tr->dynamic_heap) reason = allocate_draw_table(self, n, view_heap, first);
    timer.valued();
    if (reason) {
        if (!tr->first_refusal) {
            tr->first_refusal = reason;
            std::memcpy(tr->refusal_words, words, sizeof(tr->refusal_words));
        }
        phases.end();
        timer.finish(x2w_phase::refused);
        draw_lock_release_exclusive();
        {
            // The outputs the loop reached, for the exposure check: native buffer descriptors give a title address and a size.
            std::pair<u64, u64> outputs[16];
            unsigned output_count = 0;
            for (std::size_t i = inputs; i < count && output_count < 16; ++i) {
                const auto& w = views[i].words;
                if (w[0] == view_token_magic || !(w[0] | w[1])) continue;
                const u64 bytes_here = x2w_startup::buffer_descriptor_bytes(w[1], w[2], w[3]);
                if (bytes_here) outputs[output_count++] = {w[0] | (u64{w[1] & 0xffff} << 32), bytes_here};
            }
            provenance_note_refusal(key, outputs, output_count);
        }
        note_refused_compute_dispatch(key, reason);
        InterlockedIncrement(counter);
        // Dispatches that arrive before the offline translation must not use up the lines for bind refusals.
        static LONG untranslated, refusals;
        if (!dispatch.signature && first_calls(counter == &compute_dispatches_refused ? refusals : untranslated, 16))
            log("x2w-draw compute %016llx Dispatch(%u, %u, %u) refused: %s", key, x, y, z, reason);
        return reason;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE dest{view_heap->GetCPUDescriptorHandleForHeapStart().ptr + std::size_t{first} * draw_heap_increment};
    phases.to(4);
    for (std::size_t i = 0; i < count; ++i, dest.ptr += draw_heap_increment) {
        if (i == inputs) dest.ptr += SIZE_T{extra_srvs} * draw_heap_increment;
        const View& view = views[i];
        if (i < inputs && view.texture) {
            // A compute list accepts no pixel-shader or depth state in a transition.
            ID3D12Resource* const shader_input = view.snapshot ? view.snapshot : view.resource;
            require_state(self, shader_input, type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                : sampled_state(self, shader_input) | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            const auto* desc = view.described ? &view.desc.srv : nullptr;
            const auto dimension = tr->srvs[i].srv_dimension;
            const UINT integer = shader_input ? integer_view_type(desc ? desc->Format : shader_input->GetDesc().Format) : 0;
            if (const auto* twin = tr->integer_srvs.find(static_cast<UINT>(i))) {
                values[twin->entry] = integer;
                const D3D12_CPU_DESCRIPTOR_HANDLE twin_dest{dest.ptr + (twin->reg - i) * draw_heap_increment};
                bind_typed_texture_srv(shader_input, desc, dimension, integer, dest, twin_dest, draw_heap_increment);
            } else if (integer && tr->integer_srvs.typed) {
                count_filtered_integer_srv(tr->integer_srvs, tr->key, static_cast<UINT>(i));
                const auto null_view = x2w_startup::null_texture_srv(dimension, DXGI_FORMAT_R32_FLOAT);
                x2w_create_srv(device_of(the_device), nullptr, &null_view, dest, "compute-filtered-integer", log);
            } else {
                x2w_create_srv(device_of(the_device), shader_input, desc, dest, "compute-texture", log);
            }
            if (view.wrapper && !view.wrapper->own().tiled_address_srv) retire_hold(self, view.wrapper);
            continue;
        }
        if (i < inputs) {
            if (view.transition) require_state(self, view.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            buffer_view(dest, view.resource, view.offset, static_cast<std::size_t>(view.bytes));
            continue;
        }
        if (view.resource) require_state(self, view.resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (view.texture) {
            device_of(the_device)->CreateUnorderedAccessView(view.resource, nullptr,
                view.described ? &view.desc.uav : nullptr, dest);
            continue;
        }
        if (view.wrapper) retire_hold(self, view.wrapper);
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = view.offset / 4;
        desc.Buffer.NumElements = view.resource ? static_cast<UINT>(view.bytes / 4) : 1;  // a null view keeps one element
        desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_of(the_device)->CreateUnorderedAccessView(view.resource, nullptr, &desc, dest);
    }
    if (tr->dynamic_heap || pass.bound_view_heap != view_heap) {
        x2w_startup::bind_translated_heaps(list, view_heap, nullptr);
        pass.bound_view_heap = tr->dynamic_heap ? nullptr : view_heap;
        pass.bound_sampler_heap = nullptr;
    }
    list->SetPipelineState(tr->pipeline.Get());
    list->SetComputeRootSignature(tr->root.Get());
    const UINT entries_now = static_cast<UINT>(tr->uds.size()) + twins;
    if (entries_now) list->SetComputeRoot32BitConstants(0, entries_now, values, 0);
    if (tr->dynamic_heap) list->SetComputeRoot32BitConstant(0, heap_count, static_cast<UINT>(tr->uds.size()) + twins);
    if (n) list->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{view_heap->GetGPUDescriptorHandleForHeapStart().ptr + u64{first} * draw_heap_increment});
    if (dispatch.signature) {
        require_state(self, dispatch.argument, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        list->ExecuteIndirect(dispatch.signature, 1, dispatch.argument, dispatch.offset, nullptr, 0);
    } else {
        list->Dispatch(x, y, z);
    }
    for (std::size_t i = inputs; i < count; ++i) {  // a later dispatch on this list reads what this one wrote
        if (!views[i].resource) continue;  // a null view wrote nothing
        replay_dump::note(self, views[i].resource);
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = views[i].resource;
        list->ResourceBarrier(1, &barrier);
    }
    // The translated root replaced the desktop compute root of the title and its arguments.
    if (pass.desktop_compute_root) {
        list->SetComputeRootSignature(pass.desktop_compute_root.Get());
        for (UINT parameter = 0; parameter < pass.desktop_compute_upload_srvs.size(); ++parameter)
            if (const auto address = pass.desktop_compute_upload_srvs[parameter]) list->SetComputeRootShaderResourceView(parameter, address);
    }
    ++tr->translated;
    phases.end();
    timer.finish(dispatch.signature ? x2w_phase::indirect : x2w_phase::direct);
    draw_lock_release_exclusive();
    static LONG checks;  // as translated draws do: the queue's execute drains the rest
    if ((InterlockedIncrement(&checks) & 63) == 1) check_device("translated dispatch", dispatch.signature ? 61 : 14);
    InterlockedIncrement(&compute_dispatches_translated);
    static LONG logged;
    if (!dispatch.signature && first_calls(logged, 8))
        log("x2w-draw compute %016llx Dispatch(%u, %u, %u) translated: entry [%x %x %x %x], %zu inputs, %zu outputs", key, x, y, z,
            values[0], values[1], values[2], values[3], inputs, count - inputs);
    return nullptr;
}

// With the periodic totals: refused dispatches by the first refusal reason of their key, then every key by
// dispatch count. The per-call log lines above share one small budget, so they cannot rank keys.
void log_compute_census()
{
    struct Reason { const char* text; u64 keys, dispatches; };
    std::vector<Reason> reasons;
    std::vector<ComputeTranslation> keys;
    AcquireSRWLockShared(&draw_lock);
    for (const auto& item : compute_translations()) {
        const ComputeTranslation& tr = item.second;
        ComputeTranslation copy;
        copy.key = tr.key;
        copy.dispatches = tr.dispatches;
        copy.translated = tr.translated;
        copy.first_refusal = tr.first_refusal;
        std::memcpy(copy.refusal_words, tr.refusal_words, sizeof(copy.refusal_words));
        keys.push_back(copy);
        if (!tr.first_refusal) continue;
        auto found = std::find_if(reasons.begin(), reasons.end(), [&](const Reason& r) { return !std::strcmp(r.text, tr.first_refusal); });
        if (found == reasons.end()) found = reasons.insert(reasons.end(), Reason{tr.first_refusal, 0, 0});
        ++found->keys;
        found->dispatches += tr.dispatches - tr.translated;
    }
    ReleaseSRWLockShared(&draw_lock);
    if (keys.empty()) return;
    std::sort(reasons.begin(), reasons.end(), [](const Reason& a, const Reason& b) { return a.dispatches > b.dispatches; });
    std::sort(keys.begin(), keys.end(), [](const ComputeTranslation& a, const ComputeTranslation& b) { return a.dispatches > b.dispatches; });
    log("x2w-render compute census: %zu keys dispatched", keys.size());
    for (const Reason& r : reasons)
        log("x2w-render compute census reason \"%s\": %llu keys, %llu dispatches", r.text, r.keys, r.dispatches);
    for (std::size_t i = 0; i < keys.size(); ++i)
        log("x2w-render compute census key %016llx: %llu dispatches, %llu translated, first refusal: %s [%08x %08x %08x %08x]",
            keys[i].key, keys[i].dispatches, keys[i].translated, keys[i].first_refusal ? keys[i].first_refusal : "none",
            keys[i].refusal_words[0], keys[i].refusal_words[1], keys[i].refusal_words[2], keys[i].refusal_words[3]);
}

// One line per pipeline key (graphics and compute) whose use count changed since the last line: first and last use in
// seconds since the process started. The report script keeps the earliest first and the latest last per key.
void log_key_use_census()
{
    struct Row { const char* kind; u64 key; KeyUse use; };
    std::vector<Row> rows;
    AcquireSRWLockExclusive(&draw_lock);
    for (auto& item : compute_translations())
        if (item.second.use.uses != item.second.use.reported_uses) {
            rows.push_back({"compute", item.second.key, item.second.use});
            item.second.use.reported_uses = item.second.use.uses;
        }
    for (auto& item : translation_by_key())
        if (item.second && item.second->use.uses != item.second->use.reported_uses) {
            rows.push_back({"graphics", item.second->key, item.second->use});
            item.second->use.reported_uses = item.second->use.uses;
        }
    ReleaseSRWLockExclusive(&draw_lock);
    for (const Row& row : rows)
        log("x2w-render key use %s %016llx first %.1f s last %.1f s uses %llu", row.kind, row.key, row.use.first_ms / 1000.0,
            row.use.last_ms / 1000.0, row.use.uses);
}

void log_compute_dispatch_totals()
{
    constexpr auto format = "x2w-render compute dispatches: translated %ld, refused %ld, no translation %ld, not modelled %ld; placed inputs from a GPU copy %ld, from a title snapshot %ld (%ld KiB)";
    if (compute_title_inputs_enabled()) {
        log("x2w-render compute dispatches: translated %ld, refused %ld, no translation %ld, not modelled %ld; placed inputs from a GPU copy %ld, from a title snapshot %ld (%ld KiB), from plain title memory %ld, ring retries %ld, ring full %ld",
            compute_dispatches_translated, compute_dispatches_refused, compute_dispatches_untranslated, compute_dispatches_unmodelled,
            compute_inputs_gpu_copied, compute_inputs_snapshot, compute_inputs_snapshot_kib, compute_inputs_plain,
            compute_inputs_ring_retried, compute_inputs_ring_full);
    } else {
        log(format, compute_dispatches_translated, compute_dispatches_refused, compute_dispatches_untranslated, compute_dispatches_unmodelled,
            compute_inputs_gpu_copied, compute_inputs_snapshot, compute_inputs_snapshot_kib);
    }
    if (compute_builder_gate_refusals)
        log("x2w-render compute dispatches: builder gate refused %ld", compute_builder_gate_refusals);
}

// Fixtures: the census without waiting for the frame that logs it.
extern "C" __declspec(dllexport) void x2w_d3d12_compute_census()
{
    if (compute_title_inputs_enabled()) log_compute_dispatch_totals();
    log_compute_census();
    log_key_use_census();
}

// Fixtures: snapshot the counters without acquiring or resetting draw_lock.
extern "C" __declspec(dllexport) void x2w_d3d12_compute_refusal_stats(
    LONG64* holds, LONG64* repeated_reads, LONG64* background_reads, LONG* untranslated)
{
    *holds = InterlockedCompareExchange64(&draw_lock_stats.hold_calls, 0, 0);
    *repeated_reads = manifest_source().retry_opens();
    *background_reads = manifest_source().background_reads();
    *untranslated = InterlockedCompareExchange(&compute_dispatches_untranslated, 0, 0);
}
