// Translated compute programs at Dispatch (list slot 14). A device slot 67 pipeline is a placeholder that keeps
// the key of its published request; gfx2hlsl.py --pending answers the request with <key>.txt (`stage cs`) and
// <key>.cs.cso. Bound from the compute tables recorded by slot 35: raw buffers (`srv <i> buf table <param> <byte>`
// inputs, `uav <i> table <param> <byte>` outputs) named by hardware buffer descriptors, and 2D textures
// (`srv <i> tex2d table <param> <byte>` inputs, `uav <i> tex2d/tex2da float table <param> <byte> channels 4` outputs) named
// by the token of a view the title created (device slots 18 and 19). `smp` lines become static samplers. Entry
// values come from the compute root constants of slots 37 and 39, or from the descriptor a buffer input or output is
// bound from: its stride and word 3 (`stride`, `descword3`), and word 3, the upper half of word 1 and the record
// count of an output (`uav_descword3`, `uav_stride`, `uav_numrecords`). Any other manifest line makes the translation
// unavailable; a dispatch that cannot be bound completely is refused with one reason and records no Dispatch and
// binds nothing (the upload of an input texture the title filled in memory may already be recorded: it only
// syncs content).

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
        bool array = false;
        // An output texture of another shape than 2D: 1 = Texture1D, 3 = Texture3D (a 3D view keeps the caller's W range).
        int dimension = 0;
        bool cube = false;  // an input texture sampled as a TextureCube (`srv <i> texcube table <param> <byte>`)
    };
    u64 key = 0;
    int state = 0;                    // 0 not loaded, 1 ready, 2 unavailable
    ULONGLONG retry_manifest_at = 0;  // only an absent manifest can become ready later
    std::vector<Slot> uds, srvs, uavs;
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

// draw_lock held. Returns the reason the translation is unavailable, or null when it is ready or still absent.
const char* load_compute_translation(ComputeTranslation& tr, ComputeManifest* prefetched)
{
    tr.state = 2;
    wchar_t path[MAX_PATH];
    std::string manifest;
    if (prefetched) {
        // The manifest is ready and draw_lock serializes its consumption and PSO creation.
        manifest = std::move(prefetched->publication.text);
    } else if (!cache_path(path, L"", tr.key, L".txt") || !lookup_manifest(tr.key, tr.retry_manifest_at != 0, manifest, true)) {
        tr.retry_manifest_at = GetTickCount64() + manifest_retry_ms;
        return nullptr;
    }
    tr.retry_manifest_at = 0;
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
        if (line.empty() || (std::sscanf(line.c_str(), "group %u %u %u%n", &x, &y, &z, &used) == 3 && whole())) continue;
        if (std::sscanf(line.c_str(), "ud %u%n", &declared_uds, &used) == 1 && whole()) continue;
        if (std::sscanf(line.c_str(), "udsrc %u rootconst %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.uds.size())
            tr.uds.push_back({a, b});
        else if (std::sscanf(line.c_str(), "udsrc %u %15s %d%n", &i, name, &a, &used) == 3 && whole() && i == tr.uds.size() &&
                 source() != ComputeTranslation::root_constant)
            tr.uds.push_back({a, 0, false, false, -1, 0, 0, 0, source()});
        else if (std::sscanf(line.c_str(), "srv %u buf table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.srvs.size())
            tr.srvs.push_back({a, b});
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
        else if (std::sscanf(line.c_str(), "uav %u tex2d float table %d %d channels 4%n", &i, &a, &b, &used) == 3 && whole() &&
                 i == tr.uavs.size())
            tr.uavs.push_back({a, b, true});
        else if (std::sscanf(line.c_str(), "uav %u tex2da float table %d %d channels 4%n", &i, &a, &b, &used) == 3 && whole() &&
                 i == tr.uavs.size()) {
            tr.uavs.push_back({a, b, true});
            tr.uavs.back().array = true;
        }
        else if (std::sscanf(line.c_str(), "uav %u tex1d float table %d %d channels 4%n", &i, &a, &b, &used) == 3 && whole() &&
                 i == tr.uavs.size()) {
            tr.uavs.push_back({a, b, true});
            tr.uavs.back().dimension = 1;
        }
        else if (std::sscanf(line.c_str(), "uav %u tex3d float table %d %d channels 4%n", &i, &a, &b, &used) == 3 && whole() &&
                 i == tr.uavs.size()) {
            tr.uavs.push_back({a, b, true});
            tr.uavs.back().dimension = 3;
        }
        else if (std::sscanf(line.c_str(), "srv %u tex2d table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.srvs.size())
            tr.srvs.push_back({a, b, true});
        else if (std::sscanf(line.c_str(), "srv %u texcube table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.srvs.size()) {
            tr.srvs.push_back({a, b, true});
            tr.srvs.back().cube = true;
        }
        // A static sampler, plain (smp) or with the descriptor's LOD clamp (smpl), decoded as the draw path does.
        // Not modelled here: comparison samplers (smpc) and the run-time sampler heap table (smpheap).
        else if (line.rfind("smpc", 0) == 0) return "a comparison sampler";
        else if (line.rfind("smpheap", 0) == 0) return "a sampler from the sampler heap";
        else if ((std::sscanf(line.c_str(), "smp %u %x %x %x %x%n", &i, &x, &y, &z, &w, &used) == 5 ||
                  std::sscanf(line.c_str(), "smpl %u %x %x %x %x%n", &i, &x, &y, &z, &w, &used) == 5) && whole() &&
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
    if (tr.uds.size() != declared_uds || tr.uds.size() > 32 || tr.srvs.size() > 32 || tr.uavs.size() > 8)
        return "entry value, input or output count";
    for (const ComputeTranslation::Slot& ud : tr.uds) {
        const auto& slots = ud.source >= ComputeTranslation::output_word3 ? tr.uavs : tr.srvs;
        if (ud.source != ComputeTranslation::root_constant &&
            (ud.a < 0 || static_cast<std::size_t>(ud.a) >= slots.size() || slots[ud.a].texture))
            return "an entry value names a descriptor that is not a buffer input or output";
    }
    x2w_startup::ComPtr<ID3DBlob> code;
    if (!cache_path(path, L"", tr.key, L".cs.cso") || FAILED(D3DReadFileToBlob(path, &code))) return "no compiled program";
    // Parameter 0 keeps one constant for a program without entry values: a constants parameter cannot be empty.
    if (FAILED(x2w_startup::create_translated_root(device_of(the_device), static_cast<UINT>(tr.uds.empty() ? 1 : tr.uds.size()),
            static_cast<UINT>(tr.srvs.size()), tr.samplers, &tr.root, static_cast<UINT>(tr.uavs.size()))))
        return "root signature creation failed";
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = tr.root.Get();
    desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    if (FAILED(device_of(the_device)->CreateComputePipelineState(&desc, IID_PPV_ARGS(&tr.pipeline))))
        return "pipeline creation failed";
    tr.state = 1;
    log("x2w-draw compute pipeline %016llx translated: %zu entry values, %zu inputs, %zu outputs", tr.key, tr.uds.size(),
        tr.srvs.size(), tr.uavs.size());
    return nullptr;
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

// Buffer translations use RWByteAddressBuffer. Rebuild the token's exact byte range as
// a raw view in the translated heap, independent of where the title stored or copied its token.
const char* resolve_compute_buffer_uav(Wrapper* device, const std::uint32_t* token, Wrapper*& wrapper,
                                      ID3D12Resource*& resource, u64& offset, u64& bytes, std::uint32_t (&words)[4])
{
    ViewRecord record{};
    bool created = false;
    note_lock_acquire(false);
    if (token[1] < view_records().size()) {
        record = view_records()[token[1]];
        created = view_materialized()[token[1]] != 0;
    }
    ReleaseSRWLockShared(&note_lock);
    if (record.slot != 19) return "an output token is not an unordered access view";
    if (!created || record.desc_bytes < sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC))
        return "an output token has no materialized UAV description";

    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    std::memcpy(&view, record.desc, sizeof(view));
    if (view.ViewDimension != D3D12_UAV_DIMENSION_BUFFER)
        return "an output token does not match the buffer manifest";
    if (record.counter || view.Buffer.CounterOffsetInBytes)
        return "an output buffer view has a UAV counter";
    wrapper = as_wrapper(reinterpret_cast<const void*>(record.resource));
    if (!wrapper || wrapper->own().kind != resource_kind || !wrapper->own().real ||
        !InterlockedCompareExchange(&wrapper->own().refs, 0, 0))
        return "an output buffer view has no live desktop resource";
    const auto* info = resource_info(wrapper);
    if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON)
        return "an output buffer view resource was not created in the common state";
    resource = static_cast<ID3D12Resource*>(wrapper->own().real);
    x2w_startup::ComPtr<ID3D12Device> native_device;
    if (FAILED(resource->GetDevice(IID_PPV_ARGS(native_device.GetAddressOf()))) || native_device.Get() != device_of(device))
        return "an output buffer view belongs to another device";
    const auto desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
        return "an output view resource is not a UAV buffer";

    const auto format = desktop_format(view.Format);
    const UINT stride = view.Buffer.StructureByteStride;
    const bool raw = view.Buffer.Flags == D3D12_BUFFER_UAV_FLAG_RAW && !stride && format == DXGI_FORMAT_R32_TYPELESS;
    const bool structured = view.Buffer.Flags == D3D12_BUFFER_UAV_FLAG_NONE && stride && format == DXGI_FORMAT_UNKNOWN;
    const bool typed = view.Buffer.Flags == D3D12_BUFFER_UAV_FLAG_NONE && !stride;
    const UINT element = raw ? 4 : structured ? stride : typed ? x2w_startup::typed_buffer_format_bytes(format) : 0;
    if (!element || element > 0x3fff || view.Buffer.FirstElement > desc.Width / element)
        return "an output buffer view has an unsupported element size or start";
    offset = view.Buffer.FirstElement * element;
    bytes = u64{view.Buffer.NumElements} * element;
    if (!bytes || bytes > (16u << 20) || bytes > desc.Width - offset ||
        bytes % 4 || offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)
        return "an output buffer view range cannot be bound as a raw UAV";

    words[1] = raw ? 0 : element << 16;
    words[2] = raw ? static_cast<UINT>(bytes) : view.Buffer.NumElements;
    const UINT code = typed ? x2w_startup::buffer_format_code(format) : 0;
    // Identity channel selection, unified FORMAT for typed views, and byte bounds for raw views.
    words[3] = 0xfacu | (raw ? 3u << 28 : 0) | (code ? (1u << 24) | (code << 12) : 0);
    return nullptr;
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
        return "no translation";
    }
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
        if (!tr->state || (tr->retry_manifest_at && GetTickCount64() >= tr->retry_manifest_at)) {
            tr->unmodelled = load_compute_translation(*tr, manifest);
            if (tr->unmodelled) log("x2w-draw compute pipeline %016llx not translated: %s", key, tr->unmodelled);
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
    phases.to(1);
    for (std::size_t i = 0; i < count && !reason; ++i) {
        const bool output = i >= inputs;
        const ComputeTranslation::Slot slot = output ? tr->uavs[i - inputs] : tr->srvs[i];
        if (dispatch.signature && (slot.heap || slot.texture)) {
            reason = "translated indirect Dispatch requires fixed-table raw buffers";
            break;
        }
        std::uint32_t index = slot.a < 0 || slot.b < 0 ? 0 : pass.compute_constants.word(slot.a, slot.b);
        std::memset(words, 0, sizeof(words));
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
            if (!(words[0] | words[1] | words[2] | words[3])) {
                // A null buffer descriptor (all four words zero, no records): every read of it is out of range and the
                // hardware returns zero, so the program's index word is 0 and the dispatch reads heap record 0.
                index = 0;
            } else {
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
                note_lock_acquire(false);
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
                else if (slot.cube && (!described || desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURECUBE))
                    reason = "an input texture view is not a cube map";
                else if (slot.cube && (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.DepthOrArraySize != 6))
                    reason = "a cube map view's texture is not six 2D faces";
                else if (!slot.cube && (described ? desc.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D
                                                  : resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.DepthOrArraySize != 1))
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
            else if (!created || !texture || !(resource.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
                reason = "an output texture view was not created on the desktop";
            else if (slot.array) {
                // The layer in HLSL is relative to FirstArraySlice: retain the caller's complete view.
                const auto& a = desc.Texture2DArray;
                if (!described || desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE2DARRAY)
                    reason = "an output texture view is not a 2D array";
                else if (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.SampleDesc.Count != 1 ||
                         a.PlaneSlice || a.MipSlice >= resource.MipLevels || a.FirstArraySlice >= resource.DepthOrArraySize ||
                         !a.ArraySize || a.ArraySize > resource.DepthOrArraySize - a.FirstArraySlice)
                    reason = "an output texture array view is outside the resource";
            } else if (slot.dimension == 1) {
                if (!described || desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE1D)
                    reason = "an output texture view is not 1D";
                else if (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D || desc.Texture1D.MipSlice >= resource.MipLevels)
                    reason = "an output texture view is outside the resource";
            } else if (slot.dimension == 3) {
                // The W coordinate in HLSL is relative to FirstWSlice: retain the caller's complete view.
                const auto& v = desc.Texture3D;
                if (!described || desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE3D)
                    reason = "an output texture view is not 3D";
                else if (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D || v.MipSlice >= resource.MipLevels ||
                         !v.WSize || v.FirstWSlice >= (resource.DepthOrArraySize >> v.MipSlice ? resource.DepthOrArraySize >> v.MipSlice : 1u) ||
                         (v.WSize != 0xffffffffu && v.WSize > (resource.DepthOrArraySize >> v.MipSlice ? resource.DepthOrArraySize >> v.MipSlice : 1u) - v.FirstWSlice))
                    reason = "an output texture view is outside the resource";
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
            const IndirectBufferView view = indirect_table_buffer(self, words, false, &reason, 16u << 20, true);
            if (unfilled_heap_slot(words) && !tr->unfilled_heap_slot_logged) {
                tr->unfilled_heap_slot_logged = true;
                log("x2w-draw unfilled heap slot %u key %016llx", words[0] - 1, key);
            }
            if (view.source == 1) InterlockedIncrement(&compute_inputs_gpu_copied);
            if (view.source == 2) {
                InterlockedIncrement(&compute_inputs_snapshot);
                InterlockedExchangeAdd(&compute_inputs_snapshot_kib, static_cast<LONG>((view.bytes + 1023) / 1024));
            }
            if (view.source && !tr->input_path_logged) {
                tr->input_path_logged = true;
                log("x2w-draw compute %016llx input %zu (%s %d %d) comes from %s (%llu bytes)", key, i, slot.heap ? "heap" : "table", slot.a, slot.b,
                    view.source == 1 ? "a GPU copy this list made" : view.source == 3 ? "zeros for a null descriptor" : "a snapshot of title memory", view.bytes);
            }
            if (view.resource && view.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) reason = "an input buffer range is not 16-byte aligned in its resource";
            if (reason && !tr->first_refusal)  // once per key: which input, and where its descriptor was read
                log("x2w-draw compute %016llx first refused input %zu (%s %d %d), descriptor at 0x%llx", key, i,
                    slot.heap ? "heap" : "table", slot.a, slot.b, entry);
            // buffer_view creates a native null SRV when the token has no resource.
            views[i] = {view.resource, view.offset, view.bytes, !view.upload && !view.null_token};
            views[i].wrapper = view.token_owner;
            views[i].stride = view.stride;
            std::memcpy(views[i].words, words, sizeof(views[i].words));
            continue;
        }
        if (words[0] == view_token_magic && view_token_uav_enabled()) {
            const auto view = uav_token_buffer(self, words, 16u << 20, true, reason);
            for (std::size_t input = 0; input < inputs && !reason; ++input)
                if (views[input].resource == view.resource) reason = "an output buffer is also an input";
            views[i] = {view.resource, view.offset, view.bytes, true};
            views[i].wrapper = view.token_owner;
            views[i].words[0] = view_token_magic;
            if (!reason) {
                // The descriptor words a typed or raw view gives the shader: channel selection, format and the byte bound.
                Wrapper* owner = nullptr;
                ID3D12Resource* resource = nullptr;
                u64 offset = 0, bytes = 0;
                std::uint32_t described[4]{};
                reason = resolve_compute_buffer_uav(self->own().list_device, words, owner, resource, offset, bytes, described);
                if (!reason) views[i].words[3] = described[3];
            }
            views[i].words[1] = view.stride << 16;
            // A raw view (stride 0) counts records in bytes, a structured or typed view in elements.
            views[i].words[2] = static_cast<UINT>(view.stride ? view.bytes / view.stride : view.bytes);
            continue;
        }
        // A hardware buffer descriptor must name a placed buffer that aliases title memory.
        const u64 address = words[0] | (u64{words[1] & 0xffff} << 32);
        const UINT stride = (words[1] >> 16) & 0x3fff;
        const u64 bytes = u64{stride ? stride : 1} * words[2];
        const auto placed = words[0] == view_token_magic || !bytes || bytes % 4 || bytes > (16u << 20)
            ? PlacedVertexView{} : placed_buffer_range(self->own().list_device, address, bytes);
        const ResourceInfo* info = placed.alias ? resource_info(placed.owner) : nullptr;
        const u64 offset = placed.alias ? address - reinterpret_cast<u64>(placed.owner->own().title_memory) : 0;
        // One reason per condition, so a census says which one an output failed on.
        if (words[0] == view_token_magic) reason = "an output is a view token, not a buffer descriptor";
        else if (!bytes) reason = "an output range has zero bytes";
        else if (bytes % 4) reason = "an output range is not a whole number of dwords";
        else if (bytes > (16u << 20)) reason = "an output range is over 16 MiB";
        else if (!placed.alias) reason = "an output is not in a placed buffer that aliases title memory";
        else if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON) reason = "an output buffer was not created in the common state";
        else if (offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) reason = "an output range is not 16-byte aligned in its resource";
        else if (!(placed.alias->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            reason = "an output buffer was created without unordered access";
        static LONG size_logged;
        if (reason && (!bytes || bytes % 4 || bytes > (16u << 20)) && first_calls(size_logged, 16))
            log("x2w-draw compute %016llx output %zu has %llu bytes (stride %u, records %u): %s", key, i, static_cast<unsigned long long>(bytes),
                stride, words[2], reason);
        for (std::size_t input = 0; input < inputs && !reason; ++input)
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
            std::memcpy(word, view.words, sizeof(word));
            if (formatted_buffer_word3(word, format)) {
                reason = "a formatted input has no word 3 the program decodes";
                std::memcpy(words, view.words, sizeof(view.words));
            }
            values[i] = word[3];
            break;
        case ComputeTranslation::output_word3:
            if (view.wrapper && !view.texture && view.words[0] == view_token_magic && !view.words[3])  // word 3 is filled from the UAV description
                reason = "a UAV buffer view token has no hardware word 3";
            values[i] = view.words[3];
            break;
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
    timer.valued();
    if (reason) {
        if (!tr->first_refusal) {
            tr->first_refusal = reason;
            std::memcpy(tr->refusal_words, words, sizeof(tr->refusal_words));
        }
        phases.end();
        timer.finish(x2w_phase::refused);
        draw_lock_release_exclusive();
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
    phases.to(4);
    for (std::size_t i = 0; i < count; ++i, dest.ptr += draw_heap_increment) {
        const View& view = views[i];
        if (i < inputs && view.texture) {
            // A compute list accepts no pixel-shader or depth state in a transition.
            require_state(self, view.resource, type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                : sampled_state(self, view.resource) | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            device_of(the_device)->CreateShaderResourceView(view.resource,
                view.described ? reinterpret_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC*>(view.desc) : nullptr, dest);
            if (view.wrapper && !view.wrapper->own().tiled_address_srv) retire_hold(self, view.wrapper);
            continue;
        }
        if (i < inputs) {
            if (view.transition) require_state(self, view.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (view.wrapper) retire_hold(self, view.wrapper);
            buffer_view(dest, view.resource, view.offset, static_cast<std::size_t>(view.bytes));
            continue;
        }
        require_state(self, view.resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (view.texture) {
            device_of(the_device)->CreateUnorderedAccessView(view.resource, nullptr,
                view.described ? reinterpret_cast<const D3D12_UNORDERED_ACCESS_VIEW_DESC*>(view.desc) : nullptr, dest);
            continue;
        }
        if (view.wrapper) retire_hold(self, view.wrapper);
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = view.offset / 4;
        desc.Buffer.NumElements = static_cast<UINT>(view.bytes / 4);
        desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        if (view.wrapper) retire_hold(self, view.wrapper);
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
    // Order later UAV accesses even if the resource stays in UAV state. SRV consumers transition it with require_state.
    for (std::size_t i = inputs; i < count; ++i) {
        if (views[i].wrapper && views[i].words[0] == view_token_magic)
            note_placed_gpu_write(views[i].wrapper, views[i].offset, views[i].bytes);  // title memory is stale now
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

// Mesh requests have no entry in either draw or compute translation accounting.
auto& indirect_mesh_key_uses()
{
    static std::unordered_map<u64, KeyUse> uses;
    return uses;
}

template <class Emit>
void emit_key_use(Emit emit, const char* kind, u64 key, const KeyUse& use)
{
    char line[160];
    std::snprintf(line, sizeof(line), "x2w-render key use %s %016llx first %.1f s last %.1f s uses %llu",
                  kind, key, use.first_ms / 1000.0, use.last_ms / 1000.0, use.uses);
    emit(line);
}

void note_indirect_mesh_use(u64 key)
{
    KeyUse first_use;
    {
        x2w_startup::ExclusiveNoteLock guard(draw_lock);
        auto& use = indirect_mesh_key_uses()[key];
        // Publish the first use even if the title never reaches the periodic report.
        use.note(process_ms());
        if (use.uses == 1) {
            use.reported_uses = use.uses;
            first_use = use;
        }
    }
    if (first_use.uses)
        emit_key_use([](const char* line) { log("%s", line); }, "mesh", key, first_use);
}

// The report script keeps the earliest first and the latest last per key.
template <class Emit>
void report_key_use_census(Emit emit, bool only_if_changed)
{
    struct Row { const char* kind; u64 key; KeyUse use; };
    std::vector<Row> rows;
    {
        x2w_startup::ExclusiveNoteLock guard(draw_lock);
        const auto collect = [&](const char* kind, u64 key, KeyUse& use) {
            if (!use.uses || (only_if_changed && use.uses == use.reported_uses)) return;
            rows.push_back({kind, key, use});
            if (only_if_changed) use.reported_uses = use.uses;
        };
        for (auto& item : compute_translations())
            collect("compute", item.second.key, item.second.use);
        for (auto& item : translation_by_key())
            if (item.second) collect("graphics", item.second->key, item.second->use);
        for (auto& item : indirect_mesh_key_uses())
            collect("mesh", item.first, item.second);
    }
    for (const Row& row : rows)
        emit_key_use(emit, row.kind, row.key, row.use);
}

void log_key_use_census()
{
    report_key_use_census([](const char* line) { log("%s", line); }, true);
}

// Fixtures: the census without waiting for the frame that logs it.
extern "C" __declspec(dllexport) void x2w_d3d12_compute_census()
{
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
