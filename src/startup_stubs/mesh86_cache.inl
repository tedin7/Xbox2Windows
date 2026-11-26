// Include after d3d12_draw.inl to reuse its cache and title-memory helpers.
namespace {
UINT mesh86_word(const unsigned char* data)
{
    UINT value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

bool xbox_mesh86_container(const std::vector<unsigned char>& code, UINT expected_kind)
{
    const auto* data = code.data();
    const std::size_t bytes = code.size();
    if (bytes < 32 || std::memcmp(data, "DXBC", 4) || mesh86_word(data + 24) != bytes)
        return false;
    const std::size_t parts = mesh86_word(data + 28);
    if (parts > (bytes - 32) / 4) return false;
    bool native = false;
    for (std::size_t i = 0; i < parts; ++i) {
        const std::size_t part = mesh86_word(data + 32 + 4 * i);
        if (part < 32 + 4 * parts || part > bytes - 8) return false;
        const std::size_t size = mesh86_word(data + part + 4);
        if (size > bytes - part - 8) return false;
        if (std::memcmp(data + part, "DXIL", 4)) continue;
        if (native || size < 36) return false;
        const auto* program = data + part + 8;
        if (mesh86_word(program) >> 16 != expected_kind) return false;
        std::size_t table = 24;
        while (table <= size - 12 && std::memcmp(program + table, "XDAT", 4)) table += 4;
        if (table > size - 12) return false;
        const std::size_t count = mesh86_word(program + table + 4);
        if (count > (size - table - 12) / 12) return false;
        const std::size_t end = table + 12 + 12 * count;
        for (std::size_t entry = table + 12; entry < end; entry += 12) {
            if (mesh86_word(program + entry) != 13) continue;
            const std::size_t start = mesh86_word(program + entry + 4);
            const std::size_t length = mesh86_word(program + entry + 8);
            if (native || start < end || start > size || length < 24 || length > size - start ||
                std::memcmp(program + start, "XBCH", 4)) return false;
            native = true;
        }
        if (!native) return false;
    }
    return native;
}

struct Mesh86Request {
    const Wrapper* root{};
    std::size_t root_offset{}, ms_offset{}, ps_offset{};
    std::vector<unsigned char> ms, ps, stream;
};

bool mesh86_hardware_depth()
{
    x2w_startup::ComPtr<IDXGIFactory4> factory;
    x2w_startup::ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 desc{};
    const auto luid = device_of(the_device)->GetAdapterLuid();
    return SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf()))) &&
        SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(adapter.GetAddressOf()))) &&
        SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
        !(desc.VendorId == 0x1414 && desc.DeviceId == 0x008c);
}

void mesh86_clear_state_padding(unsigned char* record, UINT type, std::size_t size)
{
    constexpr std::size_t tag_size = sizeof(UINT);
    if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND) {
        constexpr auto targets = tag_size + offsetof(D3D12_BLEND_DESC, RenderTarget);
        constexpr auto stride = sizeof(D3D12_RENDER_TARGET_BLEND_DESC);
        constexpr auto fields = offsetof(D3D12_RENDER_TARGET_BLEND_DESC, RenderTargetWriteMask) + 1;
        static_assert(stride == 40 && fields == 37);
        for (UINT i = 0; i < 8; ++i)
            std::memset(record + targets + i * stride + fields, 0, stride - fields);
        constexpr auto used = tag_size + sizeof(D3D12_BLEND_DESC);
        std::memset(record + used, 0, size - used);
    } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL ||
               type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1) {
        constexpr auto masks = offsetof(D3D12_DEPTH_STENCIL_DESC, StencilWriteMask) + 1;
        constexpr auto faces = offsetof(D3D12_DEPTH_STENCIL_DESC, FrontFace);
        static_assert(masks == offsetof(D3D12_DEPTH_STENCIL_DESC1, StencilWriteMask) + 1 &&
                      faces == offsetof(D3D12_DEPTH_STENCIL_DESC1, FrontFace));
        std::memset(record + tag_size + masks, 0, faces - masks);
        const auto used = tag_size + (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL
            ? sizeof(D3D12_DEPTH_STENCIL_DESC) : sizeof(D3D12_DEPTH_STENCIL_DESC1));
        std::memset(record + used, 0, size - used);
    } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC) {
        constexpr auto used = tag_size + sizeof(DXGI_SAMPLE_DESC);
        std::memset(record + used, 0, size - used);
    }
}

// Parse the entire stream and keep a pointer-free copy. The shared tag is only
// recognized as Xbox MS when its shader container declares mesh and native code.
bool mesh86_request(const D3D12_PIPELINE_STATE_STREAM_DESC& desc, Mesh86Request& result, bool& recognized)
{
    const auto bytes = desc.SizeInBytes;
    if (bytes < 16 || bytes > 0x2000 || bytes % 8 ||
        !x2w_local::readable(desc.pPipelineStateSubobjectStream, bytes)) return false;
    result.stream.resize(bytes);
    if (!copy_title_bytes(result.stream.data(), desc.pPipelineStateSubobjectStream, bytes)) return false;
    UINT seen = 0;
    for (std::size_t offset = 0; offset < bytes;) {
        auto* record = result.stream.data() + offset;
        const UINT type = mesh86_word(record);
        const bool mesh = type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS; // Xbox MS tag 24.
        const auto size = mesh ? 24 : type < std::size(stream_sizes) ? stream_sizes[type] : 0;
        if (!size || size > bytes - offset || (seen & (1u << type))) return false;
        seen |= 1u << type;
        auto* payload = record + 8;
        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE) {
            const void* root;
            std::memcpy(&root, payload, sizeof(root));
            result.root = as_wrapper(root);
            result.root_offset = offset;
            std::memset(payload, 0, sizeof(root));
        } else if (mesh || type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS) {
            D3D12_SHADER_BYTECODE source;
            std::memcpy(&source, payload, sizeof(source));
            if (!source.BytecodeLength) {
                if (mesh || source.pShaderBytecode) return false;
            } else {
                if (!readable_shader_container(source)) return false;
                auto& stage = mesh ? result.ms : result.ps;
                stage.resize(source.BytecodeLength);
                if (!copy_title_bytes(stage.data(), source.pShaderBytecode, stage.size()) ||
                    !xbox_mesh86_container(stage, mesh ? 13 : 0)) return false;
                if (mesh) {
                    result.ms_offset = offset;
                    recognized = true;
                } else {
                    result.ps_offset = offset;
                }
            }
            std::memset(payload, 0, sizeof(source));
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT ||
                   type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT ||
                   type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO) {
            auto* end = record + size;
            if (std::any_of(payload, end, [](unsigned char value) { return value != 0; }))
                return false;
        } else if (type >= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS &&
                   type <= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS) {
            if (std::any_of(payload, record + size, [](unsigned char value) { return value != 0; }))
                return false;
        } else if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING) {
            return false;
        }
        // Alignment padding following the tag is outside the key's ABI.
        if (type == D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE || mesh ||
            (type >= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS && type <= D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS))
            std::memset(record + 4, 0, 4);
        mesh86_clear_state_padding(record, type, size);
        offset += size;
    }
    return recognized && result.root && result.root->own().kind == root_signature_kind;
}

ID3D12PipelineState* try_cached_mesh86(const D3D12_PIPELINE_STATE_STREAM_DESC& desc, char* why, std::size_t why_size,
                                      IndirectGraphicsPipeline& graphics, bool& recognized, u64& published_key)
{
    wchar_t path[MAX_PATH];
    if (!cache_path(path, L"req_", 0, L".m86")) return nullptr;
    Mesh86Request request;
    if (!mesh86_request(desc, request, recognized)) {
        if (recognized) std::snprintf(why, why_size, "slot86 mesh stream has unsupported subobjects, stages or root");
        return nullptr;
    }
    std::vector<unsigned char> root_blob;
    {
        note_lock_acquire(false);
        if (const auto root = root_blobs().find(request.root); root != root_blobs().end()) root_blob = root->second;
        ReleaseSRWLockShared(&note_lock);
    }
    if (root_blob.empty()) {
        std::snprintf(why, why_size, "slot86 mesh creation root has no recorded blob");
        return nullptr;
    }
    constexpr char revision[] = "x2w-startup-mesh-86-1";
    auto key = fnv_bytes(revision, sizeof(revision) - 1);
    for (const auto* part : {&request.ms, &request.ps, &root_blob, &request.stream}) key = fnv_part(*part, key);
    static SRWLOCK lock = SRWLOCK_INIT;
    x2w_startup::ExclusiveNoteLock guard(lock);
    const unsigned char none = 0;
    if (!publish_shader_request(key, L".ps86", request.ps.empty() ? &none : request.ps.data(), request.ps.size()) ||
        !publish_shader_request(key, L".root", root_blob.data(), root_blob.size()) ||
        !publish_shader_request(key, L".stream86", request.stream.data(), request.stream.size()) ||
        !publish_shader_request(key, L".m86", request.ms.data(), request.ms.size())) {
        std::snprintf(why, why_size, "slot86 mesh request %016llx publication failed", key);
        return nullptr;
    }
    published_key = key;
    const auto* params = root_info(request.root);
    if (const UINT flags = root_heap_flags(request.root)) {
        std::snprintf(why, why_size, "slot86 mesh request %016llx: root heap indexing (0x%x) has no verified binding", key, flags);
        return nullptr;
    }
    if (!params || !request.root->own().real || params->size() > 16) {
        std::snprintf(why, why_size, "slot86 mesh request %016llx: untracked root", key);
        return nullptr;
    }
    UINT srv_root = UINT_MAX;
    for (UINT i = 0; i < params->size(); ++i) {
        const auto& parameter = (*params)[i];
        if (parameter.type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS &&
            parameter.constants_count && parameter.constants_count <= 8) continue;
        if ((mesh_xmem_handoff() || mesh_xmem_loose_srv()) && srv_root == UINT_MAX && parameter.type == D3D12_ROOT_PARAMETER_TYPE_SRV &&
            parameter.shader_register == 0 && parameter.space == 0 && !request.ps.empty()) {
            srv_root = i;
            continue;
        }
        std::snprintf(why, why_size, "slot86 mesh request %016llx: root bindings outside supported profile", key);
        return nullptr;
    }
    std::string ready;
    WIN32_FILE_ATTRIBUTE_DATA marker{};
    if (!cache_path(path, L"", key, L".mesh86.txt") || !GetFileAttributesExW(path, GetFileExInfoStandard, &marker) ||
        marker.nFileSizeHigh || marker.nFileSizeLow > 96 || !read_text(path, ready)) {
        std::snprintf(why, why_size, "slot86 mesh request %016llx pending", key);
        return nullptr;
    }
    x2w_startup::ComPtr<ID3DBlob> ms, ps;
    const auto load_stage = [&](const wchar_t* ext, x2w_startup::ComPtr<ID3DBlob>& stage) {
        WIN32_FILE_ATTRIBUTE_DATA info{};
        return cache_path(path, L"", key, ext) && GetFileAttributesExW(path, GetFileExInfoStandard, &info) &&
            !info.nFileSizeHigh && info.nFileSizeLow <= 1u << 20 &&
            SUCCEEDED(D3DReadFileToBlob(path, stage.GetAddressOf()));
    };
    if (!load_stage(L".ms.cso", ms) || (!request.ps.empty() && !load_stage(L".ps.cso", ps))) {
        std::snprintf(why, why_size, "slot86 mesh request %016llx: incomplete bytecode", key);
        return nullptr;
    }
    const D3D12_SHADER_BYTECODE mesh{ms->GetBufferPointer(), ms->GetBufferSize()};
    const D3D12_SHADER_BYTECODE pixel = ps ? D3D12_SHADER_BYTECODE{ps->GetBufferPointer(), ps->GetBufferSize()} :
        D3D12_SHADER_BYTECODE{};
    char expected[96];
    std::snprintf(expected, sizeof(expected), "x2w-mesh86 %u %016llx %016llx %016llx%s\n",
        srv_root == UINT_MAX ? 1u : 2u, key,
        fnv_bytes(mesh.pShaderBytecode, mesh.BytecodeLength),
        ps ? fnv_bytes(pixel.pShaderBytecode, pixel.BytecodeLength) : 0,
        srv_root == UINT_MAX ? "" : " srv 0 0 4");
    if (ready != expected || desktop_dxil_mesh_stage(mesh) != DxilMeshStage::mesh ||
        (ps && desktop_dxil_mesh_stage(pixel) != DxilMeshStage::pixel)) {
        std::snprintf(why, why_size, "slot86 mesh request %016llx: cache key, stage or bytecode mismatch", key);
        return nullptr;
    }
    if (!ps && !mesh86_hardware_depth()) {
        std::snprintf(why, why_size, "slot86 mesh request %016llx: depth-only requires a physical adapter", key);
        return nullptr;
    }
    const UINT tag = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS;
    std::memcpy(request.stream.data() + request.ms_offset, &tag, sizeof(tag));
    std::memcpy(request.stream.data() + request.ms_offset + 8, &mesh, sizeof(mesh));
    if (ps) std::memcpy(request.stream.data() + request.ps_offset + 8, &pixel, sizeof(pixel));
    std::memcpy(request.stream.data() + request.root_offset + 8, &request.root, sizeof(request.root));
    const D3D12_PIPELINE_STATE_STREAM_DESC desktop{request.stream.size(), request.stream.data()};
    bool compute = false;
    ID3D12RootSignature* pipeline_root = nullptr;
    auto* result = build_stream_pipeline(&desktop, why, why_size, compute, graphics, pipeline_root,
        static_cast<ID3D12RootSignature*>(request.root->own().real));
    if (result) graphics.cached_srv_root = srv_root;
    static LONG loaded;
    if (result && first_calls(loaded, 24))
        log("x2w-render slot86 mesh %016llx: cached pipeline loaded, %u color targets, depth format %u",
            key, graphics.cached_target_count, static_cast<UINT>(graphics.cached_depth));
    return result;
}
}  // namespace
