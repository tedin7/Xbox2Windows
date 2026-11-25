// Optional observation of root arguments where a slot 62 pipeline is consumed.
// This runs before the slot handlers and never submits GPU work.
std::string_view compute_binding_filter()
{
    static const std::string filter = [] {
        char value[192]{};
        const DWORD error = GetLastError();
        const DWORD size = GetEnvironmentVariableA("X2W_D3D12_COMPUTE_BINDINGS", value, sizeof(value));
        SetLastError(error);
        return size && size < sizeof(value) ? std::string(value, size) : std::string();
    }();
    return filter;
}

bool compute_binding_trace_enabled() { return !compute_binding_filter().empty(); }

void note_compute_binding_pipeline(Wrapper* pipeline, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc)
{
    auto filter = compute_binding_filter();
    if (filter.empty() || !desc.CS.BytecodeLength || desc.CS.BytecodeLength > (1u << 20) ||
        !x2w_local::readable(desc.CS.pShaderBytecode, desc.CS.BytecodeLength)) return;
    const u64 key = fnv_bytes(desc.CS.pShaderBytecode, desc.CS.BytecodeLength);
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", key);
    for (;;) {
        const auto comma = filter.find(',');
        if (filter == "1" || filter.substr(0, comma) == hex) {
            pipeline->own().compute_trace_key = key;
            pipeline->own().compute_trace_root = desc.pRootSignature;
            static LONG selected;
            if (first_calls(selected, 32))
                x2w_local::logf("x2w-compute-bindings key=%016llx Create slot=62 bytes=%zu", key, desc.CS.BytecodeLength);
            return;
        }
        if (comma == std::string_view::npos) return;
        filter.remove_prefix(comma + 1);
    }
}

const char* compute_binding_source(Wrapper* list, unsigned slot, u64 address)
{
    if (!address) return "zero";
    if (slot == 35) return "table-handle";
    if (const u64 gpu = list_upload_gpu(list, address, 1))
        return gpu % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT ? "upload-unaligned" : "upload";
    const auto placed = placed_buffer_range(list->own().list_device, address, 1);
    if (placed.reject) return "ambiguous-placement";
    if (placed.alias) return "placed-alias";
    if (placed.owner) return "placed-copy";
    return "unresolved";
}

void trace_compute_dispatch(Wrapper* list, const Pass& pass, UINT x, UINT y, UINT z)
{
    if (!pass.pipeline || !pass.pipeline->own().compute_trace_key) return;
    auto& pipeline = pass.pipeline->own();
    static LONG samples;
    if (!first_calls(pipeline.compute_trace_dispatches, 16) || !first_calls(samples, 256)) return;

    const auto* trace = pass.compute_trace.get();
    const char* root_match = !trace || !trace->root ? "absent" :
        trace->root == pipeline.compute_trace_root ? "match" : "different";
    const auto* params = trace && trace->root ? root_info(trace->root) : nullptr;
    std::string bindings;
    const std::size_t count = params ? (std::min)(params->size(), std::size_t{16}) : 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& param = (*params)[i];
        const auto& seen = trace->parameters[i];
        char item[128];
        std::snprintf(item, sizeof(item), " p%zu=%u:r%u:s%u:n%u/set%u/%s/low%u/mask%016llx", i,
            static_cast<unsigned>(param.type), param.shader_register, param.space, param.constants_count, seen.setter,
            seen.setter ? compute_binding_source(list, seen.setter, seen.argument) : "none",
            static_cast<unsigned>(seen.setter && seen.setter != 35 ? seen.argument & 15 : 0), seen.constants);
        bindings += item;
    }
    x2w_local::logf("x2w-compute-bindings key=%016llx Dispatch=%ux%ux%u root=%s desktop-root=%u desktop-pso=%u params=%zu%s",
        pipeline.compute_trace_key, x, y, z, root_match, pass.compute_root != nullptr, pass.compute_pipeline,
        params ? params->size() : 0, bindings.c_str());
}

void trace_compute_binding(Wrapper* list, int slot, u64 first, u64 second, u64 third, u64 fourth)
{
    if (!recording(list)) return;
    Pass& pass = pass_of(list);
    if (slot == 14) {
        trace_compute_dispatch(list, pass, static_cast<UINT>(first), static_cast<UINT>(second), static_cast<UINT>(third));
        return;
    }
    auto* trace = pass.compute_trace.get();
    const UINT parameter = static_cast<UINT>(first);
    if (!trace || !trace->root || parameter >= trace->parameters.size()) return;
    auto& seen = trace->parameters[parameter];
    if (slot == 37 || slot == 39) {
        const UINT offset = static_cast<UINT>(slot == 37 ? third : fourth);
        const UINT count = slot == 37 ? 1 : static_cast<UINT>(second);
        if (!count || offset >= 64 || count > 64 - offset ||
            (slot == 39 && !x2w_local::readable(reinterpret_cast<const void*>(third), sizeof(UINT) * count))) return;
        seen.constants |= (~u64{} >> (64 - count)) << offset;
    } else {
        seen.setter = static_cast<UINT>(slot);
        seen.argument = second;
    }
}
