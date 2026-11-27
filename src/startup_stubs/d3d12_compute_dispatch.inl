// Translated compute programs at Dispatch (list slot 14). A device slot 67 pipeline is a placeholder that keeps
// the key of its published request; gfx2hlsl.py --pending answers the request with <key>.txt (`stage cs`) and
// <key>.cs.cso. Bound from the compute tables recorded by slot 35 and compute root SRVs recorded by slot 43: raw
// buffers (`srv <i> buf table <param> <byte>` and `srv <i> buf root <param>` inputs, `uav <i> table <param> <byte>`
// outputs) named by hardware buffer descriptors or 64-bit root SRV addresses, and textures
// (`srv <i> tex2d table <param> <byte>` inputs, `uav <i> tex1d/tex2d/tex2da/tex3d float table <param> <byte> channels 4` outputs) named
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
    // at alone) is the index. An entry value that is not a root constant names input or output a by its source.
    enum Source { root_constant, input_stride, input_word3, output_word3, output_stride, output_records };
    struct Slot {
        int a, b;
        bool texture, heap;
        int word_param = -1, word_byte = 0;
        std::uint32_t at = 0, times = 0;
        Source source = root_constant;
        D3D12_UAV_DIMENSION uav_dimension = D3D12_UAV_DIMENSION_UNKNOWN;
        bool root = false;
        bool array = false;
        unsigned half_typed_bytes = 0;
        bool packed_d16_x = false;
        unsigned scalar_window = 0;
    };
    u64 key = 0;
    int state = 0;                    // 0 not loaded, 1 ready, 2 unavailable
    x2w_startup::ManifestRetry manifest_retry;  // only an absent manifest can become ready later
    std::vector<Slot> uds, srvs, uavs;
    x2w_startup::TranslatedSamplers samplers;  // static ones only
    x2w_startup::ComPtr<ID3D12RootSignature> root;
    x2w_startup::ComPtr<ID3D12PipelineState> pipeline;
    const char* unmodelled = nullptr;      // why a manifest that exists cannot be used
    bool logged_token_output = false, logged_bound_token_output = false;
    u64 dispatches = 0, translated = 0;    // census: Dispatch calls with this key bound, and those recorded
    const char* first_refusal = nullptr;   // reason of the first refused call
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

// Formats a `float` typed output may be viewed with: the program stores float4.
bool float_output_format(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_SNORM: case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R11G11B10_FLOAT: case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_UNORM: case DXGI_FORMAT_R16G16_SNORM: case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_SNORM: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_SNORM: case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_SNORM: case DXGI_FORMAT_B8G8R8A8_UNORM:
        return true;
    default:
        return false;
    }
}
std::unordered_map<u64, ComputeTranslation>& compute_translations() { static std::unordered_map<u64, ComputeTranslation> m; return m; }
LONG compute_dispatches_translated, compute_dispatches_refused, compute_dispatches_untranslated, compute_dispatches_unmodelled;

// The root signature and pipeline of a translation whose entry values and views are filled in. Returns why it is
// unavailable, or null when it is ready.
const char* create_compute_pipeline(ComputeTranslation& tr, const void* code, std::size_t bytes)
{
    // Parameter 0 keeps one constant for a program without entry values: a constants parameter cannot be empty.
    if (FAILED(x2w_startup::create_translated_root(device_of(the_device), static_cast<UINT>(tr.uds.empty() ? 1 : tr.uds.size()),
            static_cast<UINT>(tr.srvs.size()), tr.samplers, &tr.root, static_cast<UINT>(tr.uavs.size()))))
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
const char* load_compute_translation(ComputeTranslation& tr)
{
    tr.state = 2;
    if (native_shaders_only()) {
        tr.manifest_retry.found();
        return nullptr;
    }
    wchar_t path[MAX_PATH];
    std::string manifest;
    if (!cache_path(path, L"", tr.key, L".txt") || !read_text(path, manifest)) {
        tr.manifest_retry.missed(GetTickCount64());
        return nullptr;
    }
    tr.manifest_retry.found();
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
        const auto uav_dimension = [&] {
            if (!std::strcmp(name, "tex1d")) return D3D12_UAV_DIMENSION_TEXTURE1D;
            if (!std::strcmp(name, "tex2d")) return D3D12_UAV_DIMENSION_TEXTURE2D;
            if (!std::strcmp(name, "tex2da")) return D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            if (!std::strcmp(name, "tex3d")) return D3D12_UAV_DIMENSION_TEXTURE3D;
            return D3D12_UAV_DIMENSION_UNKNOWN;
        };
        if (line.empty() || (std::sscanf(line.c_str(), "group %u %u %u%n", &x, &y, &z, &used) == 3 && whole())) continue;
        if (std::sscanf(line.c_str(), "ud %u%n", &declared_uds, &used) == 1 && whole()) continue;
        if (std::sscanf(line.c_str(), "udsrc %u rootconst %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.uds.size())
            tr.uds.push_back({a, b});
        else if (std::sscanf(line.c_str(), "udsrc %u %15s %d%n", &i, name, &a, &used) == 3 && whole() && i == tr.uds.size() &&
                 source() != ComputeTranslation::root_constant)
            tr.uds.push_back({a, 0, false, false, -1, 0, 0, 0, source()});
        else if (std::sscanf(line.c_str(), "srv %u buf table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.srvs.size())
            tr.srvs.push_back({a, b});
        else if (std::sscanf(line.c_str(), "srv %u buf root %d%n", &i, &a, &used) == 2 && whole() && a >= 0 &&
                 i == tr.srvs.size()) {
            tr.srvs.push_back({a, 0});
            tr.srvs.back().root = true;
        }
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
        else if (std::sscanf(line.c_str(), "uav %u %15s float table %d %d channels 4%n", &i, name, &a, &b, &used) == 4 && whole() &&
                 i == tr.uavs.size() && uav_dimension() != D3D12_UAV_DIMENSION_UNKNOWN) {
            tr.uavs.push_back({a, b, true});
            tr.uavs.back().uav_dimension = uav_dimension();
        }
        else if (std::sscanf(line.c_str(), "srv %u tex2d table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.srvs.size())
            tr.srvs.push_back({a, b, true});
        // A static sampler, decoded as the draw path does: plain (smp), with the descriptor's LOD clamp (smpl) or a
        // comparison one (smpc). Not modelled here: the run-time sampler heap table (smpheap).
        else if (line.rfind("smpheap", 0) == 0) return "a sampler from the sampler heap";
        else if ((std::sscanf(line.c_str(), "smp %u %x %x %x %x%n", &i, &x, &y, &z, &w, &used) == 5 ||
                  std::sscanf(line.c_str(), "smpl %u %x %x %x %x%n", &i, &x, &y, &z, &w, &used) == 5 ||
                  std::sscanf(line.c_str(), "smpc %u %x %x %x %x%n", &i, &x, &y, &z, &w, &used) == 5) && whole() &&
                 i == tr.samplers.statics.size())
            tr.samplers.read_manifest_line(line.c_str());
        else return line.rfind("srv", 0) == 0 ? "a resource that is not a raw table buffer or a 2D texture"
                  : line.rfind("uav", 0) == 0 ? "an output that is not a raw table buffer or a 1D/2D/2D-array/3D float texture"
                  : line.rfind("udsrc", 0) == 0 ? "an entry value that is not a root constant or a buffer descriptor word"
                  : line.rfind("smp", 0) == 0 ? "a sampler that is not a plain static one" : "an unknown manifest line";
    }
    if (tr.uds.size() != declared_uds || tr.uds.size() > 32 || tr.srvs.size() > 32 || tr.uavs.size() > 8)
        return "entry value, input or output count";
    for (const ComputeTranslation::Slot& ud : tr.uds) {
        const auto& slots = ud.source >= ComputeTranslation::output_word3 ? tr.uavs : tr.srvs;
        if (ud.source != ComputeTranslation::root_constant &&
            (ud.a < 0 || static_cast<std::size_t>(ud.a) >= slots.size() || slots[ud.a].texture || slots[ud.a].root))
            return "an entry value names a descriptor that is not a buffer input or output";
    }
    x2w_startup::ComPtr<ID3DBlob> code;
    if (!cache_path(path, L"", tr.key, L".cs.cso") || FAILED(D3DReadFileToBlob(path, &code))) return "no compiled program";
    if (const char* failure = create_compute_pipeline(tr, code->GetBufferPointer(), code->GetBufferSize())) return failure;
    log("x2w-draw compute pipeline %016llx translated: %zu entry values, %zu inputs, %zu outputs", tr.key, tr.uds.size(),
        tr.srvs.size(), tr.uavs.size());
    return nullptr;
}

// X2W_NATIVE_SHADERS=<directory of dxcompiler.dll and dxil.dll>: the first Dispatch of a slot 67 request translates
// it in this process with the C++ compiler (src/gfx1013_compute_native.cpp), before a published <key>.txt is read
// and with no offline translator running. The dispatch path below binds the result, so a program is accepted only
// with root constants, raw table buffers it loads per lane (buffer_load_dword) and raw table buffers it stores to.
// A refusal is logged and leaves the key to the published translations. draw_lock held. Returns whether tr is ready.
bool translate_compute_natively(ComputeTranslation& tr)
{
#ifdef X2W_NATIVE_SHADERS
    wchar_t sdk[MAX_PATH], path[MAX_PATH];
    const DWORD length = GetEnvironmentVariableW(L"X2W_NATIVE_SHADERS", sdk, MAX_PATH);
    if (!length || length >= MAX_PATH) return false;
    if (!cache_path(path, L"req_", tr.key, L".c67")) return false;
    auto refuse_native = [&](const char* reason) {
        tr.uds.clear();
        tr.srvs.clear();
        tr.uavs.clear();
        tr.root.Reset();
        tr.pipeline.Reset();
        static LONG logged;
        if (first_calls(logged, 16))
            log("x2w-draw compute pipeline %016llx not translated natively: %s", tr.key, reason);
        return false;
    };
    try {
        // The request as this runtime published it. The reader checks the six parts against the key in the name.
        // Entry 0: a source-built program starts there, and the frontend reads a capture's v_nop prefix as code.
        auto frozen = frozen_compute_requests().take(tr.key);
        if (frozen) x2w::validate_compute_request_shape(*frozen);
        else frozen.emplace(x2w::read_compute_request(path));
        x2w::NativeComputeShader program;
        auto result = x2w::ComputePackResult::miss;
        if (const auto* pack = runtime_compute_pack()) {
            std::string diagnostic;
            result = pack->lookup(*frozen, 0, program, diagnostic);
            if (result == x2w::ComputePackResult::unsupported) {
                InterlockedIncrement(&compute_pack_refused);
                return refuse_native(diagnostic.c_str());
            }
            InterlockedIncrement(result == x2w::ComputePackResult::prepared ? &compute_pack_prepared : &compute_pack_misses);
        }
        if (result == x2w::ComputePackResult::miss)
            program = x2w::translate_compute(*frozen, 0, true);
        for (const auto& block : program.constants)
            for (unsigned word = 0; word < block.count; ++word)
                tr.uds.push_back({static_cast<int>(block.root_parameter), static_cast<int>(word)});
        std::sort(program.bindings.begin(), program.bindings.end(),
                  [](const auto& a, const auto& b) { return a.shader_register < b.shader_register; });
        for (const auto& view : program.bindings) {
            const bool packed_x = view.kind == x2w::ComputeViewKind::packed_d16_x_uav;
            const bool half_typed = packed_x || view.kind == x2w::ComputeViewKind::half_buffer_uav;
            const bool texture_input = view.kind == x2w::ComputeViewKind::texture_srv && view.image_dimension == 1;
            const bool volume = view.kind == x2w::ComputeViewKind::texture_uav && view.image_dimension == 2;
            const bool output = view.kind == x2w::ComputeViewKind::raw_buffer_uav || half_typed || volume;
            auto& slots = output ? tr.uavs : tr.srvs;
            if ((!output && !texture_input && view.kind != x2w::ComputeViewKind::raw_vector_srv && !x2w::accepts_cbv_window(view.kind)) || view.array_count != 1 ||
                view.root_parameter >= 16 || view.shader_register != slots.size())
                throw std::runtime_error("view kind " + std::to_string(static_cast<unsigned>(view.kind)) +
                                         " is not bound for a natively translated program");
            slots.push_back({static_cast<int>(view.root_parameter), static_cast<int>(view.byte_offset), volume || texture_input});
            if (volume) slots.back().uav_dimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            slots.back().half_typed_bytes = half_typed ? view.required_bytes : 0;
            slots.back().packed_d16_x = packed_x;
            if (x2w::accepts_cbv_window(view.kind))
                slots.back().scalar_window = view.required_bytes;
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
        log("x2w-draw compute pipeline %016llx translated natively: %zu entry values, %zu inputs, %zu outputs (frontend %s)", tr.key,
            tr.uds.size(), tr.srvs.size(), tr.uavs.size(),
            result == x2w::ComputePackResult::prepared ? "prepared" : "live");
        return true;
    } catch (const std::exception& error) {
        return refuse_native(error.what());
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

// A slot-19 token identifies a view, not a four-dword gfx1013 buffer descriptor.
struct ComputeTokenOutput {
    ID3D12Resource* resource{};
    u64 offset{}, bytes{};
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    D3D12_RESOURCE_DESC backing{};
    Placement placement{};
    bool created{};
    u64 guest_base{}, guest_resource{}, counter{}, desktop_gpuva{};
    UINT description_bytes{};
};

const char* compute_token_output(Wrapper* list, const std::uint32_t* words, ComputeTokenOutput& output)
{
    ViewRecord record{};
    AcquireSRWLockShared(&note_lock);
    if (words[1] < view_records().size()) {
        record = view_records()[words[1]];
        output.created = view_materialized()[words[1]] != 0;
    }
    ReleaseSRWLockShared(&note_lock);
    if (words[2] != 19 || words[3] || record.slot != 19)
        return "an output token does not identify a slot-19 UAV";
    output.guest_resource = record.resource;
    output.description_bytes = record.desc_bytes;
    output.counter = record.counter;
    if (record.desc_bytes >= sizeof(output.view)) std::memcpy(&output.view, record.desc, sizeof(output.view));
    auto* owner = as_wrapper(reinterpret_cast<void*>(record.resource));
    auto* resource = owner && owner->own().kind == resource_kind ? static_cast<ID3D12Resource*>(owner->own().real) : nullptr;
    const ResourceInfo* info = owner && owner->own().kind == resource_kind ? resource_info(owner) : nullptr;
    if (info) {
        output.placement = info->placement;
        output.guest_base = reinterpret_cast<u64>(owner->own().title_memory);
        output.backing = info->console;
    }
    if (!output.created || !record.resource || record.counter || record.desc_bytes < sizeof(output.view))
        return "an output token lacks a materialized explicit counterless UAV";
    if (output.view.ViewDimension != D3D12_UAV_DIMENSION_BUFFER || !info ||
        info->console.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
        return "a raw buffer writer was bound to a non-buffer UAV";
    if (!resource) return "an output token has no desktop resource";
    output.backing = resource->GetDesc();
    if (output.backing.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
        return "a raw buffer writer was bound to a non-buffer backing";
    if (!(output.backing.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ||
        info->initial_state != D3D12_RESOURCE_STATE_COMMON)
        return "an output token's backing lacks tracked UAV access";
    if (trace_compute_outputs()) output.desktop_gpuva = resource->GetGPUVirtualAddress();
    const auto& view = output.view.Buffer;
    const bool raw = view.Flags == D3D12_BUFFER_UAV_FLAG_RAW && output.view.Format == DXGI_FORMAT_R32_TYPELESS &&
                     !view.StructureByteStride;
    const bool structured = view.Flags == D3D12_BUFFER_UAV_FLAG_NONE && output.view.Format == DXGI_FORMAT_UNKNOWN &&
                            view.StructureByteStride;
    const bool typed = view.Flags == D3D12_BUFFER_UAV_FLAG_NONE && !view.StructureByteStride;
    const UINT element = raw ? 4 : structured ? view.StructureByteStride :
                         typed ? x2w_startup::typed_buffer_format_bytes(desktop_format(output.view.Format)) : 0;
    if (!element || view.CounterOffsetInBytes || view.FirstElement > UINT64_MAX / element)
        return "an output UAV has no supported byte interval";
    output.offset = view.FirstElement * element;
    output.bytes = u64{view.NumElements} * element;
    if (!output.bytes || output.bytes % 4 || output.bytes > (16u << 20) ||
        output.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT || output.offset > output.backing.Width ||
        output.bytes > output.backing.Width - output.offset)
        return "an output UAV interval is unaligned, oversized or outside its buffer";
    const auto base = reinterpret_cast<u64>(owner->own().title_memory);
    if (base) {
        if (output.offset > UINT64_MAX - base) return "an output UAV placement address overflows";
        const auto placed = placed_buffer_range(list->own().list_device, base + output.offset, output.bytes);
        if (placed.reject) return "an output UAV range overlaps or crosses active placements";
        if (placed.owner != owner) return "an output UAV range no longer resolves to its token owner";
        // A slot-19 token identifies an object. For a complete view its committed
        // backing is the host buffer used by subsequent object-based GPU commands.
        // This mode does not imply CPU aliasing or make a partial view authoritative.
        static const bool copied_views = x2w_local::env_on("X2W_COPIED_OBJECT_UAV");
        const bool whole_copy = copied_views && info->placement == Placement::copied_buffer &&
            output.offset == 0 && output.bytes == info->bytes && output.bytes == output.backing.Width;
        if (placed.alias != resource && !whole_copy)
            return "an output token's backing does not uniquely alias its guest byte interval";
    } else {
        D3D12_HEAP_PROPERTIES heap{};
        D3D12_HEAP_FLAGS flags{};
        if (FAILED(resource->GetHeapProperties(&heap, &flags)) || heap.Type != D3D12_HEAP_TYPE_DEFAULT)
            return "an output UAV without guest shadow is not on the default heap";
    }
    x2w_startup::ComPtr<ID3D12Device> device;
    if (FAILED(resource->GetDevice(IID_PPV_ARGS(device.GetAddressOf()))) || device.Get() != device_of(list->own().list_device))
        return "an output UAV belongs to a different device";
    output.resource = resource;
    return nullptr;
}

// Both callers have an open list and a slot 67 request. The indirect caller has checked the signature and argument range.
const char* record_translated_dispatch(Wrapper* self, const TranslatedDispatch& dispatch)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    const u64 key = pass.pipeline->own().compute_request_key;
    const auto [x, y, z] = dispatch.groups;
    const char* reason = nullptr;
    LONG* counter = &compute_dispatches_refused;
    struct View {
        ID3D12Resource* resource;
        u64 offset, bytes;
        bool transition, texture, described;
        unsigned char desc[40];  // a texture view's description, as the title gave it with a desktop format
        Wrapper* wrapper;
        std::uint32_t words[4];  // a buffer's descriptor words as it was bound from, and its stride
        UINT stride;
    } views[40]{};
    static_assert(sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC) == 40 && sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC) == 40);
    const auto type = list->GetType();
    AcquireSRWLockExclusive(&draw_lock);
    ComputeTranslation* const tr = &compute_translations()[key];
    tr->key = key;
    ++tr->dispatches;
    std::uint32_t words[8] = {};
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE) reason = "the list type has no Dispatch";
    else if (x > x2w_startup::max_dispatch_groups || y > x2w_startup::max_dispatch_groups || z > x2w_startup::max_dispatch_groups)
        reason = "a group count is over the limit";
    else if (pass.compute_root_wrapper != pass.pipeline->own().compute_request_root)
        reason = "the compute root set is not the one the pipeline was created with";
    else if (!ensure_draw_heap(self->own().list_device)) reason = "no descriptor heap on this device";
    else {
        if (!tr->state || tr->manifest_retry.due(GetTickCount64())) {
            // The in-process translation is tried once per key, before a published one.
            if (tr->state || !translate_compute_natively(*tr)) {
                tr->unmodelled = load_compute_translation(*tr);
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
    for (std::size_t i = 0; i < count && !reason; ++i) {
        const bool output = i >= inputs;
        const ComputeTranslation::Slot slot = output ? tr->uavs[i - inputs] : tr->srvs[i];
        if (dispatch.signature && (slot.heap || slot.texture || slot.root)) {
            reason = "translated indirect Dispatch requires fixed-table raw buffers";
            break;
        }
        std::uint32_t index = slot.a < 0 || slot.b < 0 ? 0 : pass.compute_constants.word(slot.a, slot.b);
        std::memset(words, 0, sizeof(words));
        if (!output && slot.root) {
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
            const u64 table_entry = pass.compute_tables.table(slot.word_param).record(static_cast<u64>(slot.word_byte), 16);
            if (!table_entry || !x2w_local::readable(reinterpret_cast<const void*>(table_entry), 16)) {
                reason = "a table descriptor is not bound";
                break;
            }
            std::memcpy(words, reinterpret_cast<const void*>(table_entry), 16);
            const std::uint32_t at = slot.at + index * slot.times;
            const u64 address = words[0] | (u64{words[1] & 0xffff} << 32);
            const UINT stride = (words[1] >> 16) & 0x3fff;
            if (words[0] == view_token_magic || !list_upload_range(self, address + at, 4).resource) {
                reason = "a heap index word is not in the list's upload memory";
                break;
            }
            if (u64{at} + 4 > u64{stride ? stride : 1} * words[2]) {
                reason = "a heap index word is outside its table buffer";
                break;
            }
            std::memcpy(&index, reinterpret_cast<const void*>(address + at), 4);
        }
        // A heap record sits at the 32-bit byte offset the program computes, the index shifted left by 5, in the
        // title memory of the list's view heap: a literal byte offset, as a table's is.
        const u64 entry = slot.heap ? (slot.word_param >= 0 || slot.a >= 0 ? pass.title_view_heap.record(std::uint32_t{index << 5}, 16) : 0)
            : slot.a < 0 || slot.b < 0 ? 0 : pass.compute_tables.table(slot.a).record(static_cast<u64>(slot.b), slot.texture ? 32 : 16);
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
            bool known = false, created = false;
            if (words[0] == view_token_magic) {
                AcquireSRWLockShared(&note_lock);
                known = words[1] < view_records().size();
                if (known) {
                    record = view_records()[words[1]];
                    created = view_materialized()[words[1]] != 0;
                }
                ReleaseSRWLockShared(&note_lock);
            }
            Wrapper* const wrapper = known ? as_wrapper(reinterpret_cast<const void*>(record.resource)) : nullptr;
            ID3D12Resource* const texture = !wrapper || wrapper->own().kind != resource_kind ? nullptr
                : output ? static_cast<ID3D12Resource*>(wrapper->own().real) : sampled_address_texture(wrapper);
            const D3D12_RESOURCE_DESC resource = texture ? texture->GetDesc() : D3D12_RESOURCE_DESC{};
            if (!output) {
                D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
                const bool described = record.desc_bytes >= sizeof(desc);
                if (described) std::memcpy(&desc, record.desc, sizeof(desc));
                if (desc.Format) desc.Format = desktop_format(desc.Format);
                words[3] = static_cast<std::uint32_t>(desc.Format ? desc.Format : resource.Format) | static_cast<std::uint32_t>(desc.ViewDimension) << 16;
                if (!known || record.slot != 18) reason = "an input texture entry is not the token of a shader resource view";
                else if (!texture || (resource.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
                    reason = "an input texture has no desktop texture to sample";
                else if (described ? desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D
                                   : resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.DepthOrArraySize != 1)
                    reason = "an input texture view is not 2D";
                // shortcut: a packed format needs the stand-in rules of the draw path; add them with its first user.
                else if (x2w_startup::packed_target_format(desc.Format)) reason = "an input texture view has a packed format";
                views[i] = {texture, 0, 0, true, true, described};
                std::memcpy(views[i].desc, &desc, sizeof(desc));
                views[i].wrapper = wrapper;
                continue;
            }
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            const bool described = record.desc_bytes >= sizeof(desc);
            if (described) std::memcpy(&desc, record.desc, sizeof(desc));
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
            else if (slot.uav_dimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY) {
                // The layer in HLSL is relative to FirstArraySlice: retain the caller's complete view.
                const auto& a = desc.Texture2DArray;
                if (!described || desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE2DARRAY)
                    reason = "an output texture view is not a 2D array";
                else if (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.SampleDesc.Count != 1 ||
                         a.PlaneSlice || a.MipSlice >= resource.MipLevels || a.FirstArraySlice >= resource.DepthOrArraySize ||
                         !a.ArraySize || a.ArraySize > resource.DepthOrArraySize - a.FirstArraySlice)
                    reason = "an output texture array view is outside the resource";
            } else if (described ? desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE2D || desc.Texture2D.PlaneSlice
                               : resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.DepthOrArraySize != 1)
                reason = "an output texture view is not 2D";
            if (!reason && x2w_startup::packed_target_format(format)) reason = "an output texture view has a packed format";
            if (!reason && !float_output_format(format)) reason = "an output texture format is not a float or normalized one";
            // One state per resource here: a texture read and written by the same call is refused. Two outputs
            // of one texture (two mips) share the unordered access state.
            for (std::size_t input = 0; input < inputs && !reason; ++input)
                if (views[input].wrapper == wrapper) reason = "an output texture is also an input of the dispatch";
            views[i] = {texture, 0, 0, true, true, described};
            std::memcpy(views[i].desc, &desc, sizeof(desc));
            views[i].wrapper = wrapper;
            continue;
        }
        if (!output) {
            // Each condition of a complete GPU-backed buffer range has its own reason.
            auto view = indirect_table_buffer(self, words, false, &reason, 16u << 20);
            if (reason && slot.scalar_window && words[0] != view_token_magic &&
                !std::strcmp(reason, "an input is in title memory outside any placed buffer")) {
                const u64 address = words[0] | (u64{words[1] & 0xffff} << 32);
                const UINT stride = (words[1] >> 16) & 0x3fff;
                const u64 extent = u64{stride ? stride : 1} * words[2];
                const u64 bytes = extent;
                u64 start{};
                ID3D12Resource* chunk{};
                if (bytes >= slot.scalar_window && bytes <= 65536 && bytes % 4 == 0 &&
                    readable_window(address, bytes) >= bytes && frame_copy(self, address, bytes, &start, &chunk) && chunk) {
                    view = {chunk, start, bytes, stride, true};
                    reason = nullptr;
                }
            }
            if (view.resource && view.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) reason = "an input buffer range is not 16-byte aligned in its resource";
            if (reason && !tr->first_refusal)  // once per key: which input, and where its descriptor was read
                log("x2w-draw compute %016llx first refused input %zu (%s %d %d), descriptor at 0x%llx", key, i,
                    slot.heap ? "heap" : "table", slot.a, slot.b, entry);
            views[i] = {view.resource, view.offset, view.bytes, !view.upload};
            views[i].stride = view.stride;
            std::memcpy(views[i].words, words, sizeof(views[i].words));
            continue;
        }
        if (words[0] == view_token_magic) {
            ComputeTokenOutput token{};
            reason = compute_token_output(self, words, token);
#ifdef X2W_NATIVE_SHADERS
            if (!reason && slot.half_typed_bytes) {
                const auto kind = slot.packed_d16_x ? x2w::ComputeViewKind::packed_d16_x_uav : x2w::ComputeViewKind::half_buffer_uav;
                const x2w::ComputeViewBinding binding{kind, 0, 0, 0, slot.half_typed_bytes};
                token.view.Format = desktop_format(token.view.Format);
                if (!x2w::native_buffer_store_view(binding, token.view, token.backing.Width))
                    reason = "formatted D16 write has an incompatible typed buffer UAV";
            }
#endif
            if (reason && token.placement == Placement::copied_buffer &&
                trace_copied_compute_output(key, token.guest_resource, token.guest_base, token.offset, token.bytes)) {
                log("x2w-compute-output-watch key=%016llx groups=%u,%u,%u indirect=%u owner=0x%llx gpuva=0x%llx",
                    key, x, y, z, dispatch.signature != nullptr, token.guest_resource, token.desktop_gpuva);
                for (const auto& ud : tr->uds)
                    if (ud.source == ComputeTranslation::root_constant)
                        log("x2w-compute-output-constant key=%016llx root=%d word=%d value=0x%08x", key, ud.a, ud.b,
                            pass.compute_constants.word(ud.a, ud.b));
            }
            for (const auto& ud : tr->uds)
                if (ud.source >= ComputeTranslation::output_word3 && ud.a == static_cast<int>(i - inputs) && !reason)
                    reason = "an output token has no original ISA descriptor words for entry constants";
            for (std::size_t input = 0; input < inputs && !reason; ++input)
                if (views[input].resource == token.resource) reason = "an output buffer is also an input";
            if (!reason && !retain_with_recording(self, x2w_startup::ComPtr<ID3D12Resource>(token.resource)))
                reason = "an output token's backing could not be retained with the recording";
            if (!tr->logged_token_output || (!reason && !tr->logged_bound_token_output)) {
                tr->logged_token_output = true;
                if (!reason) tr->logged_bound_token_output = true;
                log("x2w-draw compute UAV %016llx table=%d+%d entry=0x%llx token=%u created=%u dim=%u fmt=%u "
                    "first=%llu elements=%u stride=%u flags=%u backing-dim=%u backing-bytes=%llu backing-flags=%u "
                    "placement=%u owner=0x%llx guest-base=0x%llx counter=0x%llx desc-bytes=%u offset=%llu bytes=%llu result=%s",
                    key, slot.a, slot.b, entry, words[1], token.created, token.view.ViewDimension, token.view.Format,
                    token.view.Buffer.FirstElement, token.view.Buffer.NumElements, token.view.Buffer.StructureByteStride,
                    token.view.Buffer.Flags, token.backing.Dimension, token.backing.Width, token.backing.Flags,
                    static_cast<unsigned>(token.placement), token.guest_resource, token.guest_base, token.counter,
                    token.description_bytes, token.offset, token.bytes, reason ? reason : "bound");
            }
            views[i] = {token.resource, token.offset, token.bytes, true};
            if (!reason && slot.half_typed_bytes) {
                views[i].described = true;
                std::memcpy(views[i].desc, &token.view, sizeof(token.view));
            }
            continue;
        }
        if (slot.half_typed_bytes) {
            reason = "formatted D16 write needs the original typed buffer view";
            break;
        }
        // Hardware descriptors address guest memory directly and must name a unique writable alias.
        const u64 address = words[0] | (u64{words[1] & 0xffff} << 32);
        const UINT stride = (words[1] >> 16) & 0x3fff;
        const u64 bytes = u64{stride ? stride : 1} * words[2];
        const auto placed = !bytes || bytes % 4 || bytes > (16u << 20)
            ? PlacedVertexView{} : placed_buffer_range(self->own().list_device, address, bytes);
        const ResourceInfo* info = placed.alias ? resource_info(placed.owner) : nullptr;
        const u64 offset = placed.alias ? address - reinterpret_cast<u64>(placed.owner->own().title_memory) : 0;
        if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON || offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT ||
            !(placed.alias->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            reason = "an output is not an aliased placed buffer range with unordered access";
        for (std::size_t input = 0; input < inputs && !reason; ++input)
            if (views[input].resource == placed.alias) reason = "an output buffer is also an input";
        if (reason && !tr->first_refusal)
            log("x2w-draw compute raw UAV %016llx table=%d+%d entry=0x%llx words=%08x/%08x/%08x/%08x "
                "stride=%u bytes=%llu result=%s", key, slot.a, slot.b, entry, words[0], words[1], words[2], words[3],
                stride, bytes, reason);
        views[i] = {placed.alias, offset, bytes, true};
        std::memcpy(views[i].words, words, sizeof(views[i].words));
    }
    if (dispatch.argument)
        for (std::size_t i = 0; i < count && !reason; ++i)
            if (views[i].resource == dispatch.argument) reason = "the indirect argument buffer is also a bound shader resource";
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
            std::memcpy(word, view.words, sizeof(word));
            if (formatted_buffer_word3(word, format)) {
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
    // A texture the title filled in memory is brought up to date first. An upload recorded before a later one
    // fails only syncs content, as a draw sampling that texture would.
    for (std::size_t i = 0; i < inputs && !reason; ++i)
        if (views[i].texture && !upload_texture(self, views[i].wrapper, views[i].resource)) {
            reason = "an input texture could not be uploaded";
            std::memset(words, 0, sizeof(words));
        }
    if (reason) {
        if (!tr->first_refusal) {
            tr->first_refusal = reason;
            std::memcpy(tr->refusal_words, words, sizeof(tr->refusal_words));
        }
        ReleaseSRWLockExclusive(&draw_lock);
        InterlockedIncrement(counter);
        // Dispatches that arrive before the offline translation must not use up the lines for bind refusals.
        static LONG untranslated, refusals;
        if (!dispatch.signature && first_calls(counter == &compute_dispatches_refused ? refusals : untranslated, 16))
            log("x2w-draw compute %016llx Dispatch(%u, %u, %u) refused: %s", key, x, y, z, reason);
        return reason;
    }
    const UINT n = static_cast<UINT>(count);
    UINT first = static_cast<UINT>(InterlockedExchangeAdd(&draw_heap_cursor, static_cast<LONG>(n))) % draw_heap_size;
    if (first + n > draw_heap_size) {
        first = 0;
        InterlockedExchange(&draw_heap_cursor, static_cast<LONG>(n));
    }
    D3D12_CPU_DESCRIPTOR_HANDLE dest{draw_heap->GetCPUDescriptorHandleForHeapStart().ptr + std::size_t{first} * draw_heap_increment};
    for (std::size_t i = 0; i < count; ++i, dest.ptr += draw_heap_increment) {
        const View& view = views[i];
        if (i < inputs && view.texture) {
            // A compute list accepts no pixel-shader or depth state in a transition.
            require_state(self, view.resource, type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                : sampled_state(self, view.resource) | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            device_of(the_device)->CreateShaderResourceView(view.resource,
                view.described ? reinterpret_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC*>(view.desc) : nullptr, dest);
            continue;
        }
        if (i < inputs) {
            if (view.transition) require_state(self, view.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            buffer_view(dest, view.resource, view.offset, static_cast<std::size_t>(view.bytes));
            continue;
        }
        require_state(self, view.resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (view.texture || view.described) {
            device_of(the_device)->CreateUnorderedAccessView(view.resource, nullptr,
                view.described ? reinterpret_cast<const D3D12_UNORDERED_ACCESS_VIEW_DESC*>(view.desc) : nullptr, dest);
            continue;
        }
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = view.offset / 4;
        desc.Buffer.NumElements = static_cast<UINT>(view.bytes / 4);
        desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_of(the_device)->CreateUnorderedAccessView(view.resource, nullptr, &desc, dest);
    }
    if (!pass.heap_set) {
        x2w_startup::bind_translated_heaps(list, draw_heap, nullptr);
        pass.heap_set = true;
        pass.bound_sampler_heap = nullptr;
    }
    list->SetPipelineState(tr->pipeline.Get());
    list->SetComputeRootSignature(tr->root.Get());
    if (!tr->uds.empty()) list->SetComputeRoot32BitConstants(0, static_cast<UINT>(tr->uds.size()), values, 0);
    if (n) list->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{draw_heap->GetGPUDescriptorHandleForHeapStart().ptr + u64{first} * draw_heap_increment});
    if (dispatch.signature) {
        require_state(self, dispatch.argument, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        list->ExecuteIndirect(dispatch.signature, 1, dispatch.argument, dispatch.offset, nullptr, 0);
    } else {
        list->Dispatch(x, y, z);
    }
    for (std::size_t i = inputs; i < count; ++i) {  // a later dispatch on this list reads what this one wrote
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
    ReleaseSRWLockExclusive(&draw_lock);
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

// Fixtures: the census without waiting for the frame that logs it.
extern "C" __declspec(dllexport) void x2w_d3d12_compute_census() { log_compute_census(); }

void scene_argument_probe(Wrapper* self, unsigned event, void* argument, u64 offset, const char* refused) {
    static const bool enabled = GetEnvironmentVariableW(L"X2W_SCENE_ARGUMENT_PROBE",nullptr,0) != 0;
    if (!enabled || !recording(self) || !draw_dump_frame() || frame_number != draw_dump_frame()) return;
    auto& pass = pass_of(self);
    if (!pass.pipeline) return;
    const u64 key = pass.pipeline->own().compute_request_key;
    static const u64 input_key = probe_key("X2W_ARGUMENT_INPUT_KEY");
    static const u64 output_key = probe_key("X2W_ARGUMENT_OUTPUT_KEY");
    static const u64 indirect_key = probe_key("X2W_ARGUMENT_INDIRECT_KEY");
    if (!key || (event == 17 && key != input_key) || (event == 22 && key != output_key) ||
        (event == 27 && key != indirect_key)) return;
    static volatile LONG seen[28]{};
    LONG previous = seen[event];
    if (previous == frame_number || InterlockedCompareExchange(&seen[event],frame_number,previous)!=previous) return;
    ID3D12Resource* resource{}; bool upload = false;
    u64 entry = 0; UINT words[4]{}; UINT bytes = event == 17 ? 4 : 12;
    if (event != 27) {
        entry = pass.compute_tables.table(1).record(event == 17 ? 64 : 32,16);
        if (!entry || !x2w_local::readable(reinterpret_cast<void*>(entry),16)) return;
        std::memcpy(words,reinterpret_cast<void*>(entry),16);
        if (event == 17) {
            const char* why{};
            auto v = indirect_table_buffer(self,words,false,&why);
            const u64 guest = words[0] | (u64{words[1]&0xffff}<<32);
            UINT scalar = 0; SIZE_T got = 0;
            if (words[0]!=view_token_magic)
                ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<void*>(guest+36),&scalar,4,&got);
            log("x2w-argument-probe input frame=%ld guest=%llx cpuBytes=%zu cpu=%08x resolve=%s",frame_number,guest,got,scalar,why?why:"bound");
            if (v.bytes < 40) return;
            resource=v.resource; offset=v.offset+36; upload=v.upload;
        } else {
            ComputeTokenOutput token{};
            const char* why = words[0]==view_token_magic ? compute_token_output(self,words,token) : "no token";
            log("x2w-argument-probe after frame=%ld token=%u owner=%llx dispatch=%s view=%s",frame_number,words[1],token.guest_resource,refused?refused:"recorded",why?why:"bound");
            if (token.view.Buffer.FirstElement != 0 || token.backing.Dimension!=D3D12_RESOURCE_DIMENSION_BUFFER) return;
            auto* w=as_wrapper(reinterpret_cast<void*>(token.guest_resource));
            if (!w || w->own().kind != resource_kind) return;
            resource=static_cast<ID3D12Resource*>(w->own().real);
        }
    } else {
        auto* w=as_wrapper(argument);
        if (!w || w->own().kind != resource_kind) return;
        resource=static_cast<ID3D12Resource*>(w->own().real);
    }
    if (!resource || draw_dumps().full()) return;
    auto desc=resource->GetDesc();
    if (desc.Dimension!=D3D12_RESOURCE_DIMENSION_BUFFER || offset>desc.Width || bytes>desc.Width-offset) return;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC rd{}; rd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width=bytes; rd.Height=1; rd.DepthOrArraySize=1; rd.MipLevels=1; rd.SampleDesc.Count=1; rd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    DrawDump dump{};
    if(FAILED(device_of(the_device)->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(dump.buffer.GetAddressOf())))) return;
    auto* dst=dump.buffer.Get(); dump.frame=frame_number; dump.count=event; dump.key=key;
    auto* list=recording(self);
    if (!retain_with_recording(self,x2w_startup::ComPtr<ID3D12Resource>(resource)) || !draw_dumps().record(list,std::move(dump))) return;
    auto state=D3D12_RESOURCE_STATE_COMMON;
    for (auto& st:pass.states) if (st.first==resource) state=st.second;
    if (!upload) require_state(self,resource,D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(dst,0,resource,offset,bytes);
    if (!upload) require_state(self,resource,state);
    log("x2w-argument-probe copied frame=%ld event=%u owner=%p gpu=%llx offset=%llu bytes=%u",frame_number,event,argument,resource->GetGPUVirtualAddress(),offset,bytes);
}
