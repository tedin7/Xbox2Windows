// Included after both the native graphics compiler and the desktop stream validator.
// Resolve every SRV before recording barriers or a descriptor table. A guest root address
// has no descriptor length; use the remaining proven GPU allocation, including for indexed reads.
const char* bind_mesh57_buffers(Wrapper* self, Pass& pass, ID3D12GraphicsCommandList* list, ID3D12Resource* arguments)
{
    const auto* info = pipeline_info(pass.pipeline);
    if (!info) return "missing MS creation metadata";
    const auto& reads = info->mesh57_reads;
    const auto& spans = info->mesh57_pixel_spans;
    if (reads.empty() && spans.empty()) {
        list->SetGraphicsRootSignature(pass.indirect_graphics.root);
        return nullptr;
    }
    std::array<IndirectBufferView,32> views{};
    if (reads.size() > views.size()) return "too many MS buffer reads";
    for (size_t i = 0; i < reads.size(); ++i) {
        const auto& read = reads[i];
        auto& view = views[i];
        if (read.root_address) {
            const auto address = pass.root_address[read.root_parameter];
            if (!address) return "missing MS root SRV address";
            if (auto range = list_upload_range(self, address, read.minimum_bytes, true); range.resource) {
                view = {range.resource, range.offset, range.bytes, 0, true};
            } else {
                const auto placed = placed_buffer_range(self->own().list_device, address, read.minimum_bytes);
                if (placed.reject || !placed.owner || !placed.owner->own().real)
                    return "MS root SRV lacks a complete GPU-backed range";
                auto* backing = static_cast<ID3D12Resource*>(placed.owner->own().real);
                const auto offset = address - reinterpret_cast<u64>(placed.owner->own().title_memory);
                const auto width = backing->GetDesc().Width;
                if (address >> 48 || offset >= width || width - offset > UINT_MAX)
                    return "MS root SRV range outside raw buffer limits";
                const UINT words[]{static_cast<UINT>(address), static_cast<UINT>(address >> 32),
                                   static_cast<UINT>(width - offset), 0};
                view = indirect_table_buffer(self, words);
            }
        } else {
            const auto entry = pass.view_tables[read.root_parameter].record(read.offset, 32);
            if (!entry || !x2w_local::readable(reinterpret_cast<const void*>(entry), 32))
                return "MS descriptor table slot is absent";
            UINT words[8]{};
            std::memcpy(words, reinterpret_cast<const void*>(entry), sizeof(words));
            if (words[0] == view_token_magic) {
                x2w_startup::ExclusiveNoteLock lock(note_lock);
                if (words[1] >= view_records().size() || view_records()[words[1]].slot != 18)
                    return "MS descriptor table slot is not an SRV";
            } else if (words[3] != 0x31016fac || (words[1] & 0xffff0000)) {
                return "unsupported MS native byte buffer descriptor";
            }
            view = indirect_table_buffer(self, words);
        }
        if (!view.resource || view.resource == arguments || view.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT ||
            view.bytes < read.minimum_bytes || view.bytes % 4 || view.bytes > UINT_MAX)
            return "MS buffer read exceeds its usable GPU-backed range";
    }
    UINT index = UINT_MAX;
    IndirectBufferView typed{};
    if (info->mesh57_heap_word != UINT_MAX) {
        if (!pass.mesh_heap_slots) return "MS direct read requires one bound shader-visible view heap";
        index = pass.constants[pass.indirect_graphics.constant_guest][info->mesh57_heap_word] & 1023u;
        if (index >= pass.mesh_heap_slots) return "selected MS heap index exceeds its logical descriptor count";
        const auto entry = pass.title_view_heap.record(UINT64{index} * 32, 32);
        if (!entry || !x2w_local::readable(reinterpret_cast<const void*>(entry), 32))
            return "selected MS heap slot is outside the bound shader-visible heap";
        UINT words[8]{};
        std::memcpy(words, reinterpret_cast<const void*>(entry), sizeof(words));
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        {
            x2w_startup::ExclusiveNoteLock guard(note_lock);
            if (words[0] != view_token_magic || words[1] >= view_records().size())
                return "selected MS heap slot lacks a registered SRV";
            const auto& record = view_records()[words[1]];
            if (record.slot != 18 || record.desc_bytes < sizeof(view)) return "selected MS heap slot is not an SRV";
            std::memcpy(&view,record.desc,sizeof(view));
        }
        if (view.Format != DXGI_FORMAT_R32_UINT || view.ViewDimension != D3D12_SRV_DIMENSION_BUFFER ||
            view.Buffer.Flags != D3D12_BUFFER_SRV_FLAG_NONE || view.Buffer.StructureByteStride ||
            !view.Buffer.NumElements || view.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING)
            return "selected MS heap SRV requires an R32_UINT buffer";
        typed = indirect_table_buffer(self, words, false, nullptr, u64{4} << D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP, 4);
        if (!typed.resource || typed.resource == arguments || typed.stride != 4 ||
            typed.bytes != UINT64{view.Buffer.NumElements} * 4)
            return "selected MS heap SRV lacks its complete GPU-backed range";
        if (!retain_with_recording(self, x2w_startup::ComPtr<ID3D12Resource>(typed.resource)))
            return "selected MS heap SRV could not be retained";
    }
    struct Sampled { Wrapper* owner; x2w_startup::ComPtr<ID3D12Resource> resource; D3D12_SHADER_RESOURCE_VIEW_DESC view; UINT slot; };
    std::vector<Sampled> textures;
    auto* device = device_of(self->own().list_device);
    for (const auto& span : spans) {
        if (span.last >= pass.mesh_heap_slots) return "PS texture index exceeds the bound guest heap";
        for (UINT slot=span.first; slot<=span.last; ++slot) {
            const auto at=pass.title_view_heap.record(UINT64{slot}*32,32);
            if (!at || !x2w_local::readable(reinterpret_cast<const void*>(at),32)) return "PS texture heap slot inaccessible";
            UINT token[2];
            std::memcpy(token,reinterpret_cast<const void*>(at),sizeof(token));
            ViewRecord record{};
            {
                x2w_startup::ExclusiveNoteLock guard(note_lock);
                if (token[0]!=view_token_magic || token[1]>=view_records().size()) return "PS heap slot lacks a registered SRV";
                record=view_records()[token[1]];
            }
            if (record.slot!=18 || record.desc_bytes<sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)) return "PS texture heap requires an explicit SRV";
            Sampled sampled{};
            auto& srv=sampled.view;
            std::memcpy(&srv,record.desc,sizeof(srv));
            srv.Format=desktop_format(srv.Format);
            const bool array=span.dimension==5;
            if (srv.ViewDimension!=(array?D3D12_SRV_DIMENSION_TEXTURE2DARRAY:D3D12_SRV_DIMENSION_TEXTURE2D)) return "PS heap texture dimension mismatch";
            const auto family=[](DXGI_FORMAT f) {
                if (f>=DXGI_FORMAT_BC1_TYPELESS && f<=DXGI_FORMAT_BC1_UNORM_SRGB) return DXGI_FORMAT_BC1_TYPELESS;
                if (f>=DXGI_FORMAT_BC7_TYPELESS && f<=DXGI_FORMAT_BC7_UNORM_SRGB) return DXGI_FORMAT_BC7_TYPELESS;
                if (f>=DXGI_FORMAT_R8G8B8A8_TYPELESS && f<=DXGI_FORMAT_R8G8B8A8_SINT) return DXGI_FORMAT_R8G8B8A8_TYPELESS;
                return f;
            };
            if (!float_uav_format(srv.Format) && srv.Format!=DXGI_FORMAT_R8G8B8A8_UNORM_SRGB &&
                srv.Format!=DXGI_FORMAT_BC1_UNORM && srv.Format!=DXGI_FORMAT_BC1_UNORM_SRGB &&
                srv.Format!=DXGI_FORMAT_BC7_UNORM && srv.Format!=DXGI_FORMAT_BC7_UNORM_SRGB)
                return "PS heap texture requires a supported float or normalized SRV";
            if (record.resource) {
                sampled.owner=as_wrapper(reinterpret_cast<const void*>(record.resource));
                auto* tex=sampled_address_texture(sampled.owner);
                if (!tex || tex==arguments || std::find(pass.bound_targets,pass.bound_targets+pass.target_count,tex)!=pass.bound_targets+pass.target_count ||
                    (tex==pass.bound_depth && pass.depth_view_writable)) return "PS heap texture missing or aliased to a writable attachment";
                const auto d=tex->GetDesc();
                const UINT first=srv.Texture2D.MostDetailedMip, levels=srv.Texture2D.MipLevels;
                if (d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count!=1 ||
                    (d.Format!=srv.Format && (d.Format!=family(d.Format) || family(srv.Format)!=d.Format)) ||
                    first>=d.MipLevels || !levels || (levels!=UINT_MAX && levels>d.MipLevels-first) ||
                    (array ? (!srv.Texture2DArray.ArraySize || srv.Texture2DArray.FirstArraySlice>=d.DepthOrArraySize ||
                        srv.Texture2DArray.ArraySize>d.DepthOrArraySize-srv.Texture2DArray.FirstArraySlice || srv.Texture2DArray.PlaneSlice)
                        : (d.DepthOrArraySize!=1 || srv.Texture2D.PlaneSlice))) return "PS texture SRV outside resource format, mip or array range";
                sampled.resource=tex;
                if (!retain_with_recording(self,sampled.resource)) return "PS texture lifetime retention failed";
            }
            sampled.slot=span.host_first+slot-span.first;
            textures.push_back(std::move(sampled));
        }
    }
    for (auto& texture : textures)
        if (texture.resource && !upload_texture(self,texture.owner,texture.resource.Get())) return "PS heap texture upload failed";
    const UINT prefix=spans.empty() ? (typed.resource ? index+1 : 0) :
        spans.back().host_first+(spans.back().last-spans.back().first+1);
    constexpr UINT capacity = 256;
    const auto increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    DrawSamplerHeap heap;
    UINT start;
    {
        x2w_startup::ExclusiveNoteLock guard(allocator_lock);
        const auto recording = list_recordings.find(self);
        if (!self->own().list_open || recording == list_recordings.end() || !recording->second)
            return "MS descriptor table requires an open recording";
        auto& allocator = *recording->second;
        const bool changed = index != allocator.mesh_heap_index ||
            (typed.resource && (allocator.mesh_heap_buffer != typed.resource || allocator.mesh_heap_begin != typed.offset ||
                                allocator.mesh_heap_size != typed.bytes));
        if (!spans.empty() || allocator.mesh_view_pages.empty() || changed || reads.size() > capacity - allocator.mesh_view_cursor) {
            const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, capacity + prefix,
                                                 D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
            if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)))) return "MS descriptor heap allocation failed";
            try { allocator.mesh_view_pages.push_back(heap); }
            catch (const std::bad_alloc&) { return "MS descriptor heap could not be retained with its recording"; }
            allocator.mesh_view_cursor = 0;
            allocator.mesh_heap_index = index;
            allocator.mesh_heap_buffer = typed.resource;
            allocator.mesh_heap_begin = typed.offset;
            allocator.mesh_heap_size = typed.bytes;
            if (typed.resource) {
                D3D12_SHADER_RESOURCE_VIEW_DESC srv{DXGI_FORMAT_R32_UINT, D3D12_SRV_DIMENSION_BUFFER,
                                                    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
                srv.Buffer.FirstElement = typed.offset / 4;
                srv.Buffer.NumElements = static_cast<UINT>(typed.bytes / 4);
                auto selected = heap->GetCPUDescriptorHandleForHeapStart();
                selected.ptr += SIZE_T{index} * increment;
                device->CreateShaderResourceView(typed.resource,&srv,selected);
            }
        } else {
            heap = allocator.mesh_view_pages.back();
        }
        start = allocator.mesh_view_cursor + prefix;
        allocator.mesh_view_cursor = spans.empty() ? allocator.mesh_view_cursor + static_cast<UINT>(reads.size()) : capacity;
    }
    for (auto& texture : textures) {
        auto dst=heap->GetCPUDescriptorHandleForHeapStart();
        dst.ptr+=SIZE_T{texture.slot}*increment;
        device->CreateShaderResourceView(texture.resource.Get(),&texture.view,dst);
        if (texture.resource) {
            require_state(self, texture.resource.Get(), sampled_state(self, texture.resource.Get()));
            if (!texture.owner->own().tiled_address_srv) retire_hold(self, texture.owner);
        }
    }
    auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += SIZE_T{start} * increment;
    for (size_t i = 0; i < reads.size(); ++i) {
        const auto& view = views[i];
        if (!retain_with_recording(self, x2w_startup::ComPtr<ID3D12Resource>(view.resource)))
            return "MS buffer could not be retained with its recording";
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{DXGI_FORMAT_R32_TYPELESS, D3D12_SRV_DIMENSION_BUFFER,
                                            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        srv.Buffer = {view.offset / 4, static_cast<UINT>(view.bytes / 4), 0, D3D12_BUFFER_SRV_FLAG_RAW};
        device->CreateShaderResourceView(view.resource, &srv, handle);
        handle.ptr += increment;
    }
    for (size_t i = 0; i < reads.size(); ++i)
        if (!views[i].upload) require_state(self, views[i].resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (typed.resource && !typed.upload) require_state(self, typed.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto* native_heap = heap.Get();
    list->SetDescriptorHeaps(1, &native_heap);
    list->SetGraphicsRootSignature(pass.indirect_graphics.root);
    auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += UINT64{start} * increment;
    if (!reads.empty()) list->SetGraphicsRootDescriptorTable(1, gpu);
    return nullptr;
}
#ifdef X2W_NATIVE_SHADERS
void prepare_native_mesh57(Wrapper* pipeline)
{
    wchar_t sdk[MAX_PATH], path[MAX_PATH];
    const DWORD sdk_length = GetEnvironmentVariableW(L"X2W_NATIVE_SHADERS", sdk, MAX_PATH);
    if (!sdk_length || sdk_length >= MAX_PATH)
        return;
    const auto* info = pipeline_info(pipeline);
    if (!info || !info->fixed_ms || !info->fixed_ps || info->desc.size() != 0x300)
        return;
    const auto& ms_bytes = *info->fixed_ms;
    const auto& ps_bytes = *info->fixed_ps;
    u64 fields[0x300 / sizeof(u64)];
    std::memcpy(fields, info->desc.data(), sizeof(fields));
    auto* root = as_wrapper(reinterpret_cast<const void*>(fields[0]));
    if (!root || root->own().kind != root_signature_kind || !root->own().real || !root_info(root) ||
        fields[2] != info->state_a.size() || fields[4] != info->state_b.size())
        return;
    std::vector<unsigned char> root_blob;
    {
        x2w_startup::ExclusiveNoteLock guard(note_lock);
        const auto it = root_blobs().find(root);
        if (it == root_blobs().end())
            return;
        root_blob = it->second;
    }
    constexpr char revision[] = "x2w-startup-mesh-57-3";
    const u64 key = fnv_part(root_blob, fnv_part(ps_bytes, fnv_part(ms_bytes, fnv_bytes(revision, sizeof(revision) - 1))));
    const auto normalized = x2w_startup::normalize_mesh57_descriptor(info->desc);
    const auto variant = fnv_part(info->state_b, fnv_part(info->state_a, fnv_part(normalized, key)));
    if (!cache_path(path, L"", variant, L".mesh57-state"))
        return;
    std::ifstream file(std::filesystem::path(path), std::ios::binary);
    if (!file)
        return;
    try {
        x2w_startup::Mesh57Profile profile{};
        if (!file.read(reinterpret_cast<char*>(&profile), sizeof(profile)))
            throw std::runtime_error("invalid profile header");
        const bool pixel_heap = !std::memcmp(profile.magic,"X2WM576",8);
        const bool direct_heap = pixel_heap || !std::memcmp(profile.magic, "X2WM575", 8);
        const bool buffers = direct_heap || !std::memcmp(profile.magic, "X2WM574", 8);
        const bool mapped = buffers || !std::memcmp(profile.magic, "X2WM573", 8);
        const bool varying_pixel = mapped || !std::memcmp(profile.magic, "X2WM572", 8);
        x2w_startup::Mesh57PixelEntry entry{};
        if (varying_pixel) {
            if (!file.read(reinterpret_cast<char*>(&entry), sizeof(entry)))
                throw std::runtime_error("truncated fixed pixel entry");
        } else if (std::memcmp(profile.magic, "X2WM571", 8)) {
            throw std::runtime_error("unknown fixed profile version");
        }
        x2w_startup::Mesh57ConstantEntry mapping{};
        if (mapped) {
            if (!file.read(reinterpret_cast<char*>(&mapping), sizeof(mapping)) || mapping.root_parameter >= 16 ||
                !mapping.count || mapping.count > 8 || !root_heap_flags(root))
                throw std::runtime_error("invalid MS constant mapping or guest root is not deferred");
        } else if (!root_info(root)->empty() || root_heap_flags(root)) {
            throw std::runtime_error("fixed mesh profile lacks guest root mapping");
        }
        std::vector<x2w_startup::Mesh57ViewEntry> view_entries;
        if (buffers) {
            UINT count{};
            if (!file.read(reinterpret_cast<char*>(&count), sizeof(count)) || !count || count > 15)
                throw std::runtime_error("invalid MS view entry count");
            view_entries.resize(count);
            if (!file.read(reinterpret_cast<char*>(view_entries.data()), count * sizeof(view_entries[0])))
                throw std::runtime_error("truncated MS view mappings");
            for (const auto& e : view_entries)
                if (e.root_address > 1) throw std::runtime_error("invalid MS view mapping");
        }
        UINT heap_sgpr{};
        if (direct_heap && (!file.read(reinterpret_cast<char*>(&heap_sgpr), sizeof(heap_sgpr)) ||
                            !(root_heap_flags(root) & D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED)))
            throw std::runtime_error("missing MS resource heap mapping");
        const auto match = [&](UINT part, const void* bytes, size_t size) {
            if (size != profile.lengths[part])
                return false;
            if (!size)
                return true;
            std::string copy(size, 0);
            return file.read(copy.data(), size) && !std::memcmp(copy.data(), bytes, size);
        };
        if (!match(0, normalized.data(), normalized.size()) || !match(1, info->state_a.data(), info->state_a.size()) ||
            !match(2, info->state_b.data(), info->state_b.size()) || !match(3, root_blob.data(), root_blob.size()) ||
            !match(4, ms_bytes.data(), ms_bytes.size()) || !match(5, ps_bytes.data(), ps_bytes.size()) || file.peek() != EOF)
            throw std::runtime_error("profile input bytes differ");
        if (!profile.targets.NumRenderTargets || profile.targets.NumRenderTargets > D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT ||
            ((profile.depth.DepthEnable || profile.depth.StencilEnable) && profile.depth_format == DXGI_FORMAT_UNKNOWN) ||
            profile.samples.Count != 1 ||
            profile.samples.Quality || profile.sample_mask != UINT_MAX ||
            (profile.wave_size != 32 && profile.wave_size != 64))
            throw std::runtime_error("unsupported fixed attachment or sample state");
        const std::string ms(ms_bytes.begin(), ms_bytes.end()), ps(ps_bytes.begin(), ps_bytes.end());
        x2w::MeshStageLayout mesh_layout{profile.threads, profile.vertices, profile.primitives,
                                         profile.topology, profile.group_stride, profile.wave_size == 32};
        mesh_layout.packed_parameters = varying_pixel;
        if (direct_heap) {
            mesh_layout.resource_heap_sgpr = heap_sgpr;
            mesh_layout.validate_uint_heap_reads = true;
        }
        x2w_startup::ComPtr<ID3D12VersionedRootSignatureDeserializer> deserializer;
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* guest_desc = nullptr;
        if (mapped) {
            if (FAILED(D3D12CreateVersionedRootSignatureDeserializer(root_blob.data(), root_blob.size(), IID_PPV_ARGS(&deserializer))) ||
                FAILED(deserializer->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &guest_desc)))
                throw std::runtime_error("invalid MS creation root signature");
            mesh_layout.root_inputs.push_back({mapping.sgpr, mapping.root_parameter, mapping.count});
            for (const auto& e : view_entries)
                mesh_layout.root_inputs.push_back({e.sgpr, e.root_parameter, 0, e.root_address != 0});
        }
        auto native = mapped ? x2w::translate_mesh_program(ms, mesh_layout, guest_desc->Desc_1_1)
                             : x2w::translate_mesh_program(ms, mesh_layout);
        const bool mapped_constants = mapped && native.constants.size() == 1 &&
            native.constants[0].root_parameter == mapping.root_parameter && native.constants[0].count == mapping.count &&
            native.constants[0].first_sgpr == mapping.sgpr;
        if ((mapped ? !mapped_constants : !native.constants.empty()) || (!buffers && !native.bindings.empty()) || !native.samplers.empty() || native.sampler_heap ||
            (direct_heap ? native.mesh_heap_reads.empty() : !native.mesh_heap_reads.empty()))
            throw std::runtime_error("MS needs root or heap bindings");
        UINT heap_word = UINT_MAX;
        for (const auto& guard : native.mesh_heap_reads) {
            if (guard.root_parameter != mapping.root_parameter || guard.dword >= mapping.count || guard.mask != 1023 ||
                (heap_word != UINT_MAX && heap_word != guard.dword))
                throw std::runtime_error("unsupported MS resource heap index contract");
            heap_word = guard.dword;
        }
        std::vector<x2w_startup::Mesh57BufferRead> reads;
        if (buffers) {
            if (native.bindings.empty() || native.bindings.size() > 32)
                throw std::runtime_error("unsupported MS buffer binding count");
            for (auto b : native.bindings) {
                if (b.shader_register != reads.size() || b.root_parameter >= guest_desc->Desc_1_1.NumParameters ||
                    b.root_parameter >= 16 || !b.required_bytes || b.array_count != 1 || b.array_position)
                    throw std::runtime_error("unsupported MS buffer register or span");
                const bool direct = b.kind == x2w::ComputeViewKind::root_buffer_srv;
                const auto& param = guest_desc->Desc_1_1.pParameters[b.root_parameter];
                if (direct) {
                    if (param.ParameterType != D3D12_ROOT_PARAMETER_TYPE_SRV || b.byte_offset)
                        throw std::runtime_error("MS root buffer requires a root SRV");
                } else {
                    if (b.kind != x2w::ComputeViewKind::buffer_srv || param.ParameterType != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
                        throw std::runtime_error("MS table read requires a scalar SRV");
                    // Validate this creation root's range even when reflection already classified the slot.
                    using Table = x2w::ComputeViewBinding::ScalarTable;
                    if (b.scalar_table != Table::unknown && b.scalar_table != Table::srv)
                        throw std::runtime_error("MS table binding is not an SRV");
                    b.scalar_table = Table::unknown;
                    x2w::apply_scalar_table_layout(b, param.DescriptorTable);
                    if (b.scalar_table != x2w::ComputeViewBinding::ScalarTable::srv || b.byte_offset % 32)
                        throw std::runtime_error("MS buffer slot outside a verified SRV range");
                }
                reads.push_back({b.root_parameter, b.byte_offset, b.required_bytes, direct});
            }
        }
        std::string ps_ir;
        std::vector<x2w::PixelHeapSpan> spans;
        std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
        if (varying_pixel) {
            auto pixel = x2w::translate_mesh_pixel_program(ps, entry.enable, entry.address, entry.wave_size, pixel_heap ? &guest_desc->Desc_1_1 : nullptr);
            if (pixel_heap) {
                if (pixel.pixel_heap_spans.empty()) throw std::runtime_error("missing PS texture heap reads");
                spans=std::move(pixel.pixel_heap_spans);
                for (const auto& c:pixel.constants)
                    if (c.root_parameter!=mapping.root_parameter || c.count!=mapping.count)
                        throw std::runtime_error("PS constants differ from the bound MS constant block");
                for (const auto& s:pixel.samplers)
                    samplers.push_back(x2w_startup::decode_sampler(s.words,static_cast<UINT>(samplers.size()),s.comparison,s.explicit_lod));
            }
            if ((!pixel_heap && (!pixel.constants.empty() || !pixel.samplers.empty())) || !pixel.bindings.empty() || pixel.sampler_heap)
                throw std::runtime_error("PS needs root or heap bindings");
            x2w::validate_raster_link(native, pixel);
            ps_ir = std::move(pixel.dxil_ir);
        } else {
            ps_ir = x2w::direct_gfx1013_constant_pixel(ps);
        }
        auto compiler = x2w::ShaderDxilCompiler::runtime_worker(sdk);
        const auto cache = std::filesystem::path(path).parent_path();
        const auto ms_output = compiler.compile(native.dxil_ir, L"ms_6_6", cache, x2w::ShaderSourceFormat::dxil_ir);
        const auto ps_output =
            compiler.compile(ps_ir, L"ps_6_6", cache, x2w::ShaderSourceFormat::dxil_ir);
        x2w::meshlet::MeshOnlyPipelineStream stream{};
        stream.root_signature.value = reinterpret_cast<ID3D12RootSignature*>(root);
        stream.ms.value = {ms_output.dxil.data(), ms_output.dxil.size()};
        stream.ps.value = {ps_output.dxil.data(), ps_output.dxil.size()};
        stream.blend.value = profile.blend;
        stream.sample_mask.value = profile.sample_mask;
        stream.rasterizer.value = profile.rasterizer;
        stream.depth_stencil.value = profile.depth;
        stream.rtv_formats.value = profile.targets;
        stream.dsv_format.value = profile.depth_format;
        stream.sample_desc.value = profile.samples;
        stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
        bool compute = false;
        char why[256]{};
        auto& state = pipeline->own();
        x2w_startup::ComPtr<ID3D12RootSignature> host;
        if (mapped) {
            D3D12_ROOT_PARAMETER params[2]{};
            auto& block = params[0];
            block.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            block.Constants.Num32BitValues = (mapping.count + 3) & ~3u;
            D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, static_cast<UINT>(reads.size())};
            params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            params[1].DescriptorTable = {1, &range};
            D3D12_ROOT_SIGNATURE_DESC host_desc{reads.empty() ? 1u : 2u, params, static_cast<UINT>(samplers.size()), samplers.data()};
            if (direct_heap) host_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
            x2w_startup::ComPtr<ID3DBlob> blob;
            if (FAILED(D3D12SerializeRootSignature(&host_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr)) ||
                FAILED(device_of(the_device)->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                                   IID_PPV_ARGS(&host))))
                throw std::runtime_error("failed to create MS b0 constant root");
        }
        state.real = build_stream_pipeline(&desc, why, sizeof(why), compute, state.indirect_graphics, state.pipeline_root,
                                          host ? host.Get() : static_cast<ID3D12RootSignature*>(root->own().real), true);
        if (!state.real)
            throw std::runtime_error(why);
        if (mapped) {
            static constexpr GUID lifetime{0xf74c6928,0xb063,0x4a00,{0xa5,0xf0,0x79,0xe3,0x60,0xa4,0xf1,0x09}};
            auto* desktop = static_cast<ID3D12PipelineState*>(state.real);
            if (FAILED(desktop->SetPrivateDataInterface(lifetime, host.Get()))) {
                desktop->Release();
                state.real = nullptr;
                state.indirect_graphics = {};
                throw std::runtime_error("could not retain MS constant root");
            }
            state.indirect_graphics.constant_guest = mapping.root_parameter;
            info->mesh57_pixel_spans = std::move(spans);
            info->mesh57_reads = std::move(reads);
            info->mesh57_heap_word = heap_word;
        }
        log("x2w-render native slot57 mesh %016llx ready", key);
    } catch (const std::exception& e) {
        log("x2w-render native slot57 mesh %016llx refused: %s", key, e.what());
    }
}
#endif
