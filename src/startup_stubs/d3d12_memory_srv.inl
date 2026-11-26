// Included after the resource creation and descriptor handlers.
struct MemoryShaderResourceView {
    D3D12_SHADER_RESOURCE_VIEW_DESC view;
    u64 address;
};
static_assert(offsetof(MemoryShaderResourceView, address) == 0x28);

bool same_texture_description(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b)
{
    return a.Dimension == b.Dimension && a.Alignment == b.Alignment && a.Width == b.Width &&
           a.Height == b.Height && a.DepthOrArraySize == b.DepthOrArraySize && a.MipLevels == b.MipLevels &&
           a.Format == b.Format && a.SampleDesc.Count == b.SampleDesc.Count &&
           a.SampleDesc.Quality == b.SampleDesc.Quality && a.Layout == b.Layout && a.Flags == b.Flags;
}

bool tiled_bc_srv(const D3D12_RESOURCE_DESC& r)
{
    return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && r.DepthOrArraySize == 1 &&
           r.MipLevels == 1 && r.SampleDesc.Count == 1 && r.SampleDesc.Quality == 0 &&
           (r.Format == DXGI_FORMAT_BC3_UNORM || r.Format == DXGI_FORMAT_BC7_UNORM) &&
           static_cast<UINT>(r.Layout) == 0x105 && r.Alignment == 0 &&
           (r.Flags == D3D12_RESOURCE_FLAG_NONE || static_cast<UINT>(r.Flags) == 0x300000) &&
           r.Width <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION && r.Height <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
           // A single mip pads its pitch and height to whole 4 KiB tiles (Bc16Surface); the desktop
           // texture needs whole 4x4 blocks.
           r.Width && r.Height && r.Width % 4 == 0 && r.Height % 4 == 0;
}

// One line for the first publication of a view whose last tile row or column is partial. A published
// view gets a content summary, so a title run can tell console-tiled data from zeros; a held one names
// why it stays refused.
void log_partial_tile_view(const D3D12_RESOURCE_DESC& r, u64 address, const char* held, bool published)
{
    static LONG lines, unpublished;
    const auto bytes = x2w_startup::Bc16Surface(r).size();
    if (!published) {
        if (first_calls(unpublished, 8))
            log("x2w-d3d12 Device::slot70 partial-tile BC view %llux%u format %d layout 0x%x: %s, %llu tiled bytes",
                r.Width, r.Height, r.Format, static_cast<unsigned>(r.Layout), held ? held : "texture_unavailable", bytes);
        return;
    }
    if (!first_calls(lines, 64)) return;
    const auto* data = reinterpret_cast<const unsigned char*>(address);
    const auto nonzero = [&](u64 from) {
        unsigned count = 0;
        for (u64 i = from; i < from + 4096; ++i) count += data[i] != 0;
        return count;
    };
    // publish_tiled_texture already found the whole range readable.
    log("x2w-d3d12 Device::slot70 partial-tile BC view %llux%u format %d layout 0x%x: published, %llu tiled bytes, "
        "nonzero first/last 4 KiB %u/%u hash %016llx", r.Width, r.Height, r.Format, static_cast<unsigned>(r.Layout),
        bytes, nonzero(0), nonzero(bytes - 4096), hash_memory(data, static_cast<std::size_t>(bytes)));
}

// The size the XG stand-in reports for a description: desktop copyable footprints rounded to 64 KiB.
u64 xg_reported_bytes(Wrapper* device, const D3D12_RESOURCE_DESC& desc)
{
    const x2w_startup::SliceFootprints plan(device_of(device), sanitized(desc), 1);
    return plan.size == UINT64_MAX ? 0 : (plan.size + 0xFFFF) & ~u64{0xFFFF};
}

bool supports_placed_memory_srv(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v);
// The shapes slot 70 took before address_texture.h, or a texture in a console layout gfx10_layout.h
// addresses with a view that stays inside it.
bool supports_memory_srv(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v)
{
    if (supports_placed_memory_srv(r, v)) return true;
    return console_layout_srv(r) && x2w_startup::address_view_fits(r, v);
}

bool supports_placed_memory_srv(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v)
{
    if (r.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
        return x2w_startup::supports_buffer_memory_srv(r, v);
    if (r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D) {
        const auto format = desktop_format(r.Format);
        if (!placed_srv_1d_enabled() || !r.Width || r.Height != 1 || !r.DepthOrArraySize ||
            r.SampleDesc.Count != 1 || r.SampleDesc.Quality != 0 ||
            (r.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) ||
            v.Format != r.Format || format == DXGI_FORMAT_UNKNOWN ||
            x2w_view_dimension::typeless(format) || x2w_view_dimension::depth_format(format) ||
            v.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING)
            return false;
        const auto mip_range_fits = [&](UINT first, UINT count) {
            return first < r.MipLevels && count &&
                   (count == UINT_MAX || count <= r.MipLevels - first);
        };
        if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE1D) {
            const auto& t = v.Texture1D;
            return r.DepthOrArraySize == 1 && mip_range_fits(t.MostDetailedMip, t.MipLevels) &&
                   t.ResourceMinLODClamp == 0;
        }
        if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE1DARRAY) {
            const auto& t = v.Texture1DArray;
            return t.FirstArraySlice < r.DepthOrArraySize && t.ArraySize &&
                   t.ArraySize <= r.DepthOrArraySize - t.FirstArraySlice &&
                   mip_range_fits(t.MostDetailedMip, t.MipLevels) && t.ResourceMinLODClamp == 0;
        }
        return false;
    }
    if (x2w_startup::bc64::supports(r)) {
        if (v.Format != r.Format || v.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING)
            return false;
        if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY) {
            const auto& a = v.Texture2DArray;
            return r.DepthOrArraySize == 2 && a.FirstArraySlice < r.DepthOrArraySize && a.ArraySize > 0 &&
                   a.ArraySize <= r.DepthOrArraySize - a.FirstArraySlice && a.MostDetailedMip < r.MipLevels &&
                   a.MipLevels > 0 && a.MipLevels <= r.MipLevels - a.MostDetailedMip &&
                   a.PlaneSlice == 0 && a.ResourceMinLODClamp == 0;
        }
        return r.DepthOrArraySize == 1 && v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D &&
               v.Texture2D.MostDetailedMip < r.MipLevels && v.Texture2D.MipLevels > 0 &&
               v.Texture2D.MipLevels <= r.MipLevels - v.Texture2D.MostDetailedMip &&
               !v.Texture2D.PlaneSlice && v.Texture2D.ResourceMinLODClamp == 0;
    }
    if (x2w_startup::supports_single_element_srv(r, v)) return true;
    if (x2w_startup::supports_volume_memory_srv(r, v)) return true;
    return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && r.Width && r.Height &&
           r.DepthOrArraySize == 1 && r.MipLevels == 1 && r.SampleDesc.Count == 1 && r.SampleDesc.Quality == 0 &&
           (r.Format == DXGI_FORMAT_R8_UNORM || tiled_bc_srv(r)) && v.Format == r.Format && !(r.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) &&
           v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D &&
           v.Shader4ComponentMapping == D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING &&
           v.Texture2D.MostDetailedMip == 0 && v.Texture2D.MipLevels == 1 && v.Texture2D.PlaneSlice == 0 &&
           v.Texture2D.ResourceMinLODClamp == 0.0f;
}

// The layout tables describe the bytes independently of the desktop adapter's format support.
bool desktop_address_view_supported(Wrapper* device, const D3D12_RESOURCE_DESC& r,
                                    const D3D12_SHADER_RESOURCE_VIEW_DESC& v)
{
    auto* const desktop = device_of(device);
    D3D12_FEATURE_DATA_FORMAT_INFO planes{r.Format};
    if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &planes, sizeof(planes))) || planes.PlaneCount != 1)
        return false;

    D3D12_FEATURE_DATA_FORMAT_SUPPORT view{v.Format};
    if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &view, sizeof(view))) ||
        !(view.Support1 & (D3D12_FORMAT_SUPPORT1_SHADER_LOAD | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE)))
        return false;

    D3D12_FEATURE_DATA_FORMAT_SUPPORT storage{r.Format};
    if (r.Format == v.Format) storage = view;
    else if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &storage, sizeof(storage)))) return false;
    const UINT req = r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D   ? D3D12_FORMAT_SUPPORT1_TEXTURE3D
                     : r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D ? D3D12_FORMAT_SUPPORT1_TEXTURE1D
                                                                         : D3D12_FORMAT_SUPPORT1_TEXTURE2D;
    return (storage.Support1 & req) &&
           (r.MipLevels == 1 || (storage.Support1 & D3D12_FORMAT_SUPPORT1_MIP));
}

bool memory_srv_destination(u64 handle)
{
    bool valid = false;
    note_lock_acquire(false);
    for (const auto& heap : heap_infos()) {
        if (heap.type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || !heap.increment) continue;
        const u64 base = heap.title_cpu ? heap.title_cpu : heap.desktop_cpu;
        if (handle >= base && (handle - base) % heap.increment == 0 &&
            (handle - base) / heap.increment < heap.count) {
            valid = true;
            break;
        }
    }
    ReleaseSRWLockShared(&note_lock);
    return valid;
}

// With X2W_RESOURCE_INDEX_CHECK=1 (or 2, see resource_index_check_mode), every memory SRV call (a full pass, for tests) checks that each live resource with title
// memory is in the address index (entries are never removed, so it stays there after a release). A miss is logged and ends the process with code 3; this is a test mode only.
void check_resource_address_index()
{
    if (resource_index_check_mode() == 0) return;
    note_lock_acquire(false);
    std::optional<u64> missing;
    for (const auto& [wrapper, info] : resource_infos()) {
        const u64 base = reinterpret_cast<u64>(wrapper->own().title_memory);
        if (!base || InterlockedCompareExchange(&wrapper->own().refs, 0, 0) <= 0 || resource_address_index().contains(base, wrapper))
            continue;
        missing = base;
        break;
    }
    const std::size_t entries = resource_address_index().size();
    ReleaseSRWLockShared(&note_lock);
    if (!missing) return;
    log("x2w-d3d12 resource address index misses a live resource at 0x%llx (index %zu entries)", *missing, entries);
    ExitProcess(3);
}

Wrapper* existing_memory_texture(u64 address, const D3D12_RESOURCE_DESC& desc)
{
    X2W_ZONE("srv existing_memory_texture scan");
    Wrapper* match = nullptr;
    note_lock_acquire(false);
    // A null address never gets here (memory_srv_refusal), and the index holds no null base.
    resource_address_index().for_each_reaching(address, 1, [&](const Wrapper* wrapper) {
        const auto& info = resource_infos().at(wrapper);
        if (reinterpret_cast<u64>(wrapper->own().title_memory) != address ||
            wrapper->own().tiled_address_srv ||
            InterlockedCompareExchange(&wrapper->own().refs, 0, 0) <= 0 ||
            !same_texture_description(desc, info.console)) return false;
        if (match) {
            match = nullptr;
            return true;
        }
        match = const_cast<Wrapper*>(wrapper);
        return false;
    });
    ReleaseSRWLockShared(&note_lock);
    return match;
}

// An unmatched or interior view must not reinterpret bytes owned by a live resource.
bool address_resource_overlap(u64 address, u64 span)
{
    X2W_ZONE("srv address_resource_overlap scan");
    bool found = false;
    note_lock_acquire(false);
    resource_address_index().for_each_reaching(address, span, [&](const Wrapper* wrapper) {
        if (wrapper->own().tiled_address_srv || InterlockedCompareExchange(&wrapper->own().refs, 0, 0) <= 0)
            return false;
        const auto& info = resource_infos().at(wrapper);
        const u64 bytes = info.bytes ? info.bytes
            : info.console.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? info.console.Width : 0;
        return found = x2w_startup::address_span_touches(address, span, reinterpret_cast<u64>(wrapper->own().title_memory), bytes);
    });
    ReleaseSRWLockShared(&note_lock);
    return found;
}

// A live render-target, depth or UAV resource overlapping the texel is written on the desktop side, so its
// title memory does not hold the texel. For a resource of unknown size, its base is the only known byte.
bool gpu_written_overlap(u64 address, UINT texel_bytes)
{
    X2W_ZONE("srv gpu_written_overlap scan");
    constexpr UINT gpu_written = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    bool found = false;
    note_lock_acquire(false);
    resource_address_index().for_each_reaching(address, texel_bytes, [&](const Wrapper* wrapper) {
        const auto& info = resource_infos().at(wrapper);
        const u64 bytes = info.bytes ? info.bytes : info.console.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? info.console.Width : 0;
        return found = x2w_startup::address_span_touches(address, texel_bytes, reinterpret_cast<u64>(wrapper->own().title_memory), bytes) &&
            (static_cast<UINT>(info.console.Flags) & gpu_written) && InterlockedCompareExchange(&wrapper->own().refs, 0, 0) > 0;
    });
    ReleaseSRWLockShared(&note_lock);
    return found;
}

Wrapper* existing_memory_buffer(Wrapper* device, u64 address, u64 bytes)
{
    X2W_ZONE("srv existing_memory_buffer scan");
    if (!address || !bytes) return nullptr;
    const auto placed = placed_buffer_range(device, address, bytes);
    if (placed.reject) return nullptr;
    if (placed.owner) {
        if (InterlockedCompareExchange(&placed.owner->own().refs, 0, 0) <= 0) return nullptr;
        const auto* info = resource_info(placed.owner);
        if (!info || info->console.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
            (info->console.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
            return nullptr;
        return const_cast<Wrapper*>(placed.owner);
    }
    Wrapper* match = nullptr;
    note_lock_acquire(false);
    resource_address_index().for_each_reaching(address, 1, [&](const Wrapper* wrapper) {
        const auto& info = resource_infos().at(wrapper);
        if (info.placement != Placement::none ||
            info.console.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
            (info.console.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) ||
            InterlockedCompareExchange(&wrapper->own().refs, 0, 0) <= 0)
            return false;
        const u64 base = reinterpret_cast<u64>(wrapper->own().title_memory);
        const u64 width = info.console.Width;
        if (!base || address < base || address - base >= width || bytes > width - (address - base))
            return false;
        if (match) {
            match = nullptr;
            return true;
        }
        match = const_cast<Wrapper*>(wrapper);
        return false;
    });
    ReleaseSRWLockShared(&note_lock);
    return match;
}

// Backing for unplaced buffer memory SRVs: 64 KiB page-aligned placed resource on an existing heap
// (OpenExistingHeapFromAddress zero-copy aliasing on hardware, allocations up to 256 MiB) or committed UPLOAD
// buffer (fallback).
struct UnplacedBufferEntry {
    Wrapper* device = nullptr;
    void* allocation_base = nullptr;
    u64 page_start = 0;
    u64 page_span = 0;
    bool is_upload = false;
    bool exact = false;  // starts at the view address (a structured view whose stride does not divide the page offset)
    x2w_startup::ComPtr<ID3D12Resource> resource;
    Wrapper* wrapper = nullptr;
};

std::mutex unplaced_buffer_mutex;
std::vector<UnplacedBufferEntry> unplaced_buffers;

void clear_unplaced_buffers(Wrapper* device)
{
    std::lock_guard lock(unplaced_buffer_mutex);
    for (auto it = unplaced_buffers.begin(); it != unplaced_buffers.end();) {
        if (it->device == device) {
            if (it->wrapper) {
                retire_placement(it->wrapper);
                if (it->wrapper->own().budget) it->wrapper->own().budget->end_use();
                if (it->wrapper->own().backing_budget) it->wrapper->own().backing_budget->end_use();
                it->wrapper->own().real = nullptr;
                InterlockedDecrement(&it->wrapper->own().refs);
            }
            it->resource.Reset();
            it = unplaced_buffers.erase(it);
            x2w_startup::note_placement_change();
        } else {
            ++it;
        }
    }
}

// refusal (out, optional): when a nullptr answer is exactly provable, the proof that keeps a cached copy of it exact
// (srv_refusal_cache.h): decided by the request and stand-in state alone (independent), by a fully committed
// allocation the buffer does not fit (fully_committed), or by one page that is not committed (witness_page).
// A nullptr from a later step carries no proof (proven stays false).
Wrapper* unplaced_memory_buffer(Wrapper* device, u64 address, u64 span_bytes, const D3D12_RESOURCE_DESC& requested_desc,
                                u64 elem_bytes, x2w_startup::SrvRefusalProof* refusal = nullptr)
{
    if (refusal) *refusal = x2w_startup::SrvRefusalProof{};
    const auto prove = [&](x2w_startup::SrvRefusalKind kind, u64 witness = 0, u64 base = 0) {
        if (refusal) *refusal = x2w_startup::SrvRefusalProof{kind, witness, base, true};
    };
    if (address_resource_overlap(address, span_bytes)) {
        prove(x2w_startup::SrvRefusalKind::independent);
        return nullptr;
    }
    u64 page_start = 0, page_span = 0;
    if (!x2w_startup::unplaced_buffer_extent(address, span_bytes, page_start, page_span)) {
        prove(x2w_startup::SrvRefusalKind::independent);
        return nullptr;
    }

    x2w_startup::PlacedBufferRange memory{};
    if (!x2w_startup::measure_placed_buffer(reinterpret_cast<void*>(page_start), page_span, memory) ||
        !memory.covered) {
        if (memory.region_info)
            prove(x2w_startup::SrvRefusalKind::fully_committed);
        else if (memory.witness)
            prove(x2w_startup::SrvRefusalKind::witness_page, memory.witness, reinterpret_cast<u64>(memory.base));
        return nullptr;
    }

    const auto nearby = placed_buffer_range(device, page_start, page_span, true);
    if (nearby.reject || nearby.owner) {
        prove(x2w_startup::SrvRefusalKind::independent);
        return nullptr;
    }

    // A buffer view addresses whole elements from its base. When the base is the 64 KiB page start and the view address is not
    // a multiple of the element size past it, no FirstElement can name the address: back the view with an upload copy that
    // starts at the address itself (checked above to lie inside memory the title owns).
    const bool exact = x2w_startup::unplaced_view_needs_exact_start(address, page_start, elem_bytes);
    if (exact) {
        page_start = address;
        page_span = span_bytes;
    }

    auto* desktop = device_of(device);
    if (!desktop) return nullptr;

    std::lock_guard lock(unplaced_buffer_mutex);
    for (auto& entry : unplaced_buffers) {
        if (entry.device == device && entry.allocation_base == memory.base && entry.exact == exact &&
            x2w_startup::unplaced_buffer_covers(entry.page_start, entry.page_span, address, span_bytes)) {
            x2w_startup::PlacedBufferRange verify{};
            if (!x2w_startup::measure_placed_buffer(reinterpret_cast<void*>(entry.page_start), entry.page_span, verify) ||
                !verify.covered || verify.base != memory.base) {
                continue;
            }
            if (entry.is_upload && entry.resource) {
                void* mapped = nullptr;
                D3D12_RANGE read_range{0, 0};
                if (SUCCEEDED(entry.resource->Map(0, &read_range, &mapped))) {
                    std::memcpy(mapped, reinterpret_cast<const void*>(entry.page_start), static_cast<std::size_t>(entry.page_span));
                    entry.resource->Unmap(0, nullptr);
                }
            }
            return entry.wrapper;
        }
    }

    x2w_startup::ComPtr<ID3D12Resource> native_resource;
    bool is_upload = false;
    ID3D12Heap* heap = nullptr;
    x2w_budget::Ticket* backing_budget = nullptr;

    D3D12_FEATURE_DATA_EXISTING_HEAPS existing_support{};
    const bool existing_heaps_supported =
        SUCCEEDED(desktop->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &existing_support, sizeof(existing_support))) &&
        existing_support.Supported;

    if (existing_heaps_supported && memory.type == MEM_PRIVATE && !exact) {
        std::lock_guard placed_lock(placed_mutex);
        for (const auto& [known, h] : existing_heaps) {
            if (known == memory.base) {
                heap = h.real;
                backing_budget = h.budget;
                break;
            }
        }
        if (!heap && x2w_startup::unplaced_existing_heap_allowed(x2w_startup::allocation_bytes(memory.base))) {
            ID3D12Device3* device3 = nullptr;
            if (SUCCEEDED(desktop->QueryInterface(IID_PPV_ARGS(&device3)))) {
                if (SUCCEEDED(device3->OpenExistingHeapFromAddress(memory.base, IID_PPV_ARGS(&heap)))) {
                    backing_budget = x2w_budget::backing_heap(heap, 0);
                    existing_heaps.emplace_back(memory.base, ExistingHeap{heap, backing_budget});
                }
                device3->Release();
            }
        }
        if (heap) {
            const u64 base_u64 = reinterpret_cast<u64>(memory.base);
            const u64 heap_offset = page_start - base_u64;
            const u64 heap_size = heap->GetDesc().SizeInBytes;
            if (heap_offset <= heap_size && page_span <= heap_size - heap_offset) {
                D3D12_RESOURCE_DESC desc{};
                desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                desc.Width = page_span;
                desc.Height = 1;
                desc.DepthOrArraySize = 1;
                desc.MipLevels = 1;
                desc.Format = DXGI_FORMAT_UNKNOWN;
                desc.SampleDesc.Count = 1;
                desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

                desktop->CreatePlacedResource(heap, heap_offset, &desc, D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, IID_PPV_ARGS(&native_resource));
            }
        }
    }

    if (!native_resource) {
        D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_UPLOAD};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = page_span;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;

        if (SUCCEEDED(desktop->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&native_resource)))) {
            is_upload = true;
            void* mapped = nullptr;
            D3D12_RANGE read_range{0, 0};
            if (SUCCEEDED(native_resource->Map(0, &read_range, &mapped))) {
                std::memcpy(mapped, reinterpret_cast<const void*>(page_start), static_cast<std::size_t>(page_span));
                native_resource->Unmap(0, nullptr);
            }
        }
    }

    if (!native_resource) return nullptr;

    Wrapper* w = wrap(native_resource.Get(), resource_kind);
    if (!w) return nullptr;
    w->own().title_memory = reinterpret_cast<void*>(page_start);
    D3D12_RESOURCE_DESC rdesc{};
    rdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rdesc.Width = page_span;
    rdesc.Height = 1;
    rdesc.DepthOrArraySize = 1;
    rdesc.MipLevels = 1;
    rdesc.Format = DXGI_FORMAT_UNKNOWN;
    rdesc.SampleDesc.Count = 1;
    rdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rdesc.Flags = is_upload ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

    note_resource(w, &rdesc, page_span);
    if (!is_upload) {
        w->own().budget = x2w_budget::resource(desktop, native_resource.Get(), x2w_budget::placed);
        w->own().backing_budget = backing_budget;
        if (backing_budget) backing_budget->begin_use();
    } else {
        w->own().budget = x2w_budget::committed(desktop, native_resource.Get(), D3D12_HEAP_TYPE_UPLOAD);
    }

    unplaced_buffers.push_back(UnplacedBufferEntry{device, memory.base, page_start, page_span, is_upload, exact, native_resource, w});
    x2w_startup::note_placement_change();
    return w;
}

// Keep metadata for published tiled SRVs; allocate GPU memory only when a
// translated draw actually samples one. A later write to the same address is
// picked up by the uploader even when the description and view token are reused.
std::mutex tiled_srv_mutex;
struct TiledAddressTexture {
    Wrapper* device;
    Wrapper* resource;
    D3D12_RESOURCE_DESC desc;
    x2w_startup::ComPtr<ID3D12Resource> desktop;
};
std::unordered_map<Wrapper*, TiledAddressTexture> tiled_srv_textures;
std::unordered_multimap<u64, Wrapper*> tiled_srv_addresses;

bool partial_tiled_bc(const D3D12_RESOURCE_DESC& desc)
{
    return tiled_bc_srv(desc) && (desc.Width % 64 || desc.Height % 64);
}

bool latest_texture_write_is_rows(u64 address)
{
    AcquireSRWLockShared(&written_lock);
    const auto found = written_textures().find(address);
    const bool rows = found != written_textures().end() && found->second.desktop_rows;
    ReleaseSRWLockShared(&written_lock);
    return rows;
}

// A texture slot 70 reads in a console layout of gfx10_layout.h: not one of the shapes it read before
// (the single texel, the 0x105 BC3/BC7 surface and the 0x109 BC chains keep their own paths).
bool console_layout_srv(const D3D12_RESOURCE_DESC& desc)
{
    return x2w_startup::console_address_texture(desc) && !x2w_startup::single_element_bytes(desc) &&
           !tiled_bc_srv(desc) && !x2w_startup::bc64::supports(desc);
}

// One line for each new kind of slot-70 texture call (format, layout, shape class, view and outcome), so
// a run shows every shape a title asks for without a line per texture.
void log_address_kind(const char* outcome, const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v, u64 address)
{
    static std::mutex mutex;
    static std::unordered_map<u64, LONG> kinds;
    const u64 words[] = {static_cast<u64>(r.Format), static_cast<u64>(r.Layout), r.MipLevels > 1 ? 2u : 1u,
                         r.DepthOrArraySize > 1 ? 2u : 1u, static_cast<u64>(r.Flags), static_cast<u64>(r.Dimension),
                         static_cast<u64>(v.Format), static_cast<u64>(v.ViewDimension), reinterpret_cast<u64>(outcome)};
    const u64 key = fnv_bytes(words, sizeof(words));
    std::lock_guard lock(mutex);
    if (kinds.size() >= 200 || kinds[key]++) return;
    log("x2w-d3d12 slot70 texture kind %zu: %s; first seen res{dim=%d fmt=%d size=%llux%ux%u mips=%u layout=0x%x flags=0x%x "
        "align=%llu} view{dim=%d fmt=%d map=0x%x} address=0x%llx", kinds.size(), outcome, r.Dimension, r.Format, r.Width,
        r.Height, r.DepthOrArraySize, r.MipLevels, static_cast<UINT>(r.Layout), static_cast<UINT>(r.Flags), r.Alignment,
        v.ViewDimension, v.Format, v.Shader4ComponentMapping, address);
}

// Bytes the texels of a by-address texture span at their address.
u64 address_texture_bytes(Wrapper* device, const D3D12_RESOURCE_DESC& desc)
{
    if (const UINT element = x2w_startup::single_element_bytes(desc)) return element;
    if (x2w_startup::bc64::supports(desc)) return x2w_startup::bc64::Chain(desc).bytes();
    if (console_layout_srv(desc)) return x2w_startup::gfx10::Layout(desc).bytes();
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
        if (device) {
            const x2w_startup::SliceFootprints plan(device_of(device), sanitized(desc), desc.MipLevels ? desc.MipLevels : 1);
            if (plan.size != UINT64_MAX && plan.size) return plan.size;
        }
        return x2w_startup::volume_texture_bytes(desc);
    }
    return x2w_startup::Bc16Surface(desc).size();
}

// Why a console-layout description is not read from an address, or null. The two cases a partial-tile
// BC view is held for:
// - the latest XG stand-in copy to the address wrote desktop-footprint rows (only the opt-in BC7 4 KiB
//   copy records a tiled write; the 0x109 BC chains, which XG writes in their layout, keep their own
//   path and never come through here);
// - the layout runs past the size the XG stand-in reports for the description (the desktop copyable
//   footprints of a slice rounded up to 64 KiB, times the slices), which the title may not own.
const char* console_address_hold(Wrapper* device, u64 address, const D3D12_RESOURCE_DESC& desc)
{
    if (latest_texture_write_is_rows(address)) return "console_layout_xg_rows";
    const x2w_startup::SliceFootprints plan(device_of(device), sanitized(desc), desc.MipLevels);
    if (plan.size == UINT64_MAX || !plan.size) return "console_layout_without_footprints";
    const u64 reported = ((plan.size + 0xFFFF) & ~u64{0xFFFF}) * desc.DepthOrArraySize;
    return x2w_startup::gfx10::Layout(desc).bytes() > reported ? "console_layout_beyond_xg_size" : nullptr;
}

// A texture range the probe found unreadable is not probed again for a short time: the title republishes the same view
// every frame, and each probe of an unmapped or protected range costs a first-chance exception (the whole process stops
// on each one while a debugger is attached). A failure for (address, bytes) holds for any longer range at the address;
// the entry expires, because the memory may be mapped or committed later.
struct UnreadableTextureRange {
    u64 address = 0, bytes = 0;
    ULONGLONG tick = 0;
};
constexpr ULONGLONG unreadable_texture_hold_ms = 1000;
std::mutex unreadable_texture_mutex;
UnreadableTextureRange unreadable_textures[64];
unsigned unreadable_texture_next;
volatile LONG unreadable_texture_skips;
volatile LONG untouched_texture_refusals;  // ranges refused by VirtualQuery state without touching them

bool texture_range_readable(u64 address, u64 bytes)
{
    const ULONGLONG now = GetTickCount64();
    {
        std::lock_guard lock(unreadable_texture_mutex);
        for (const auto& entry : unreadable_textures) {
            if (entry.address == address && bytes >= entry.bytes && now - entry.tick < unreadable_texture_hold_ms) {
                const LONG skips = InterlockedIncrement(&unreadable_texture_skips);
                if (skips == 1 || (skips & (skips - 1)) == 0)
                    log("x2w-d3d12 slot70 texture range probes skipped (recently unreadable): %ld", skips);
                return false;
            }
        }
    }
    // Not touched: a range the title has not committed yet (lazily committed memory is MEM_RESERVE until the title writes it)
    // is refused from its VirtualQuery state alone, so it costs no first-chance exception however many distinct ranges ask.
    if (x2w_local::committed_range(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes))) return true;
    const LONG refusals = InterlockedIncrement(&untouched_texture_refusals);
    if (refusals == 1 || (refusals & (refusals - 1)) == 0)
        log("x2w-d3d12 slot70 texture ranges refused without touching them (not committed yet): %ld", refusals);
    std::lock_guard lock(unreadable_texture_mutex);
    unreadable_textures[unreadable_texture_next++ % 64] = UnreadableTextureRange{address, bytes, now};
    return false;
}

// Why publish_tiled_texture (or the lookup of a live resource) gave no texture: the three causes the census
// keeps apart. `published` is true when a publication was attempted (BC, console layout, single element), false when
// only a live resource with this base and description could have served the view.
constexpr u64 max_published_texture_bytes = 64ull << 20;  // the transient upload ring's single-copy limit
const char* texture_unavailable_reason(Wrapper* device, u64 address, const D3D12_RESOURCE_DESC& desc, bool published)
{
    if (!published) return "texture_no_resource";
    const auto bytes = address_texture_bytes(device, desc);
    if (bytes > max_published_texture_bytes) return "texture_over_64mib";
    if (!texture_range_readable(address, bytes)) return "texture_unreadable";
    return "texture_unavailable";  // publication failed for another reason (no wrapper)
}

Wrapper* publish_tiled_texture(Wrapper* device, u64 address, const D3D12_RESOURCE_DESC& desc)
{
    const auto bytes = address_texture_bytes(device, desc);
    // A console layout is read only up to its last addressed element: a small texture in a big block need not have the
    // block's padding mapped.
    const auto read_bytes = console_layout_srv(desc) ? x2w_startup::gfx10::Layout(desc).used_bytes() : bytes;
    // The transient upload ring accepts individual copies up to 64 MiB.
    if (bytes > max_published_texture_bytes || !texture_range_readable(address, read_bytes)) return nullptr;
    std::lock_guard lock(tiled_srv_mutex);
    const auto [first, last] = tiled_srv_addresses.equal_range(address);
    for (auto it = first; it != last; ++it) {
        const auto& known = tiled_srv_textures.at(it->second);
        if (known.device == device && same_texture_description(known.desc, desc)) {
            AcquireSRWLockExclusive(&written_lock);
            ++written_textures()[address].generation;
            ReleaseSRWLockExclusive(&written_lock);
            return known.resource;
        }
    }
    Wrapper* resource = wrap(nullptr, resource_kind);
    if (!resource) return nullptr;
    resource->own().title_memory = reinterpret_cast<void*>(address);
    resource->own().tiled_address_srv = true;
    note_resource(resource, &desc, bytes);
    tiled_srv_textures.emplace(resource, TiledAddressTexture{device, resource, desc, {}});
    tiled_srv_addresses.emplace(address, resource);
    // address_resource_overlap reads tiled_address_srv: a buffer refused for overlap may be publishable now.
    x2w_startup::note_placement_change();
    return resource;
}

// A texture that lost mip levels (texture_drop_mips.h) goes only to a bind that said it samples it; every other reader
// (a compute view, a bind of a texel-addressed shader) gets no texture and the refusal is counted.
ID3D12Resource* texture_for_reader(Wrapper* resource, ID3D12Resource* texture)
{
    namespace dm = x2w_startup::drop_mips;
    if (!texture || !resource->own().drop_levels || dm::may_receive_dropped(dm::current_hint)) return texture;
    dm::counters.conflicts.fetch_add(1, std::memory_order_relaxed);
    static LONG shown;
    if (first_calls(shown, 6))
        log("x2w-d3d12 texture %p lost %u mip level(s) and is asked for by a reader that does not only sample it: refused",
            static_cast<void*>(resource), static_cast<unsigned>(resource->own().drop_levels));
    return nullptr;
}

ID3D12Resource* sampled_address_texture(Wrapper* resource)
{
    if (!resource || resource->own().kind != resource_kind) return nullptr;
    if (resource->own().lazy_slot133_texture && !resource->own().real)
        return texture_for_reader(resource, materialize_lazy_texture(resource));
    if (!resource->own().tiled_address_srv)
        return texture_for_reader(resource, static_cast<ID3D12Resource*>(resource->own().real));
    std::lock_guard lock(tiled_srv_mutex);
    auto& sampled = tiled_srv_textures.at(resource);
    // An already-published token can outlive a write in a different layout.
    if ((partial_tiled_bc(sampled.desc) || console_layout_srv(sampled.desc)) &&
        latest_texture_write_is_rows(reinterpret_cast<u64>(resource->own().title_memory)))
        return nullptr;
    if (!sampled.desktop) {
        const auto desc = sanitized(sampled.desc);
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        const HRESULT hr = device_of(sampled.device)->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&sampled.desktop));
        if (FAILED(hr)) {
            static LONG failures;
            if (first_calls(failures, 8)) log("x2w-d3d12 failed to materialize tiled memory SRV: 0x%08lx", hr);
            return nullptr;
        }
        x2w_budget::committed(device_of(sampled.device), sampled.desktop.Get(), D3D12_HEAP_TYPE_DEFAULT);
        x2w_startup::mark_packed_stand_in(sampled.desktop.Get(), desktop_format(sampled.desc.Format));
    }
    return sampled.desktop.Get();
}

// The first check that refuses a call, in the order the checks run; null when every check passes.
const char* memory_srv_refusal(const D3D12_RESOURCE_DESC* resource, const MemoryShaderResourceView* input, u64 destination)
{
    if (!destination) return "null_destination";
    const bool readable_input = x2w_local::readable(input, sizeof(*input));
    // An acceleration-structure view names its structure by address only; the resource description is not read.
    const bool as_view = readable_input && x2w_startup::is_as_srv(input->view);
    if (!as_view && !x2w_local::readable(resource, sizeof(*resource))) return "unreadable_resource";
    if (!readable_input) return "unreadable_view";
    if (as_view) {
        if (!x2w_startup::as_srv_location(input->view, input->address)) return "null_address";
        if (!x2w_startup::supports_as_memory_srv(input->view)) return "unsupported_description";
    } else {
        if (!input->address) return "null_address";
        if (!supports_memory_srv(*resource, input->view)) return "unsupported_description";
    }
    if (!memory_srv_destination(destination)) return "destination_outside_view_heaps";
    return nullptr;
}

// One line with every field the checks read; the caller rate-limits it.
void log_refused_memory_srv(const char* what, const char* reason, const D3D12_RESOURCE_DESC* resource,
                            const MemoryShaderResourceView* input, u64 destination)
{
    const bool resource_ok = resource && x2w_local::readable(resource, sizeof(*resource));
    const bool input_ok = x2w_local::readable(input, sizeof(*input));
    const u64 address = input_ok ? input->address : 0;
    char line[640];
    x2w_startup::report_refused_memory_srv(line, sizeof(line), what, reason, resource_ok ? resource : nullptr,
        input_ok ? &input->view : nullptr, address, address && x2w_local::readable(reinterpret_cast<const void*>(address), 1),
        destination, destination && memory_srv_destination(destination));
    log("%s", line);
}

// Diagnostic for textures refused with the desktop "unknown" layout (Layout 0): nothing says how their rows are packed.
// For the first 32 of them one line gives what a reader needs to infer the pitch: format, size, mips, the address
// alignment, how many bytes after the address are readable (a lower bound of the allocation), the tightly packed byte
// size, and a hash plus non-zero counts of the first rows. Only a hash and counts are logged, never pixel data.
void log_layout0_texture(const char* reason, const D3D12_RESOURCE_DESC* resource, const MemoryShaderResourceView* input)
{
    if (!reason || std::strcmp(reason, "unsupported_description") != 0) return;
    if (!x2w_local::readable(resource, sizeof(*resource)) || !x2w_local::readable(input, sizeof(*input))) return;
    if (resource->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource->Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN) return;
    static LONG lines;
    if (!first_calls(lines, 32)) return;
    const u64 address = input->address;
    const auto element = x2w_startup::gfx10::element_of(resource->Format);
    const u64 blocks_w = element.bytes ? (resource->Width + element.texels - 1) / element.texels : 0;
    const u64 blocks_h = element.bytes ? (resource->Height + element.texels - 1) / element.texels : 0;
    const u64 row_bytes = blocks_w * element.bytes;
    const u64 first_mip_bytes = row_bytes * blocks_h;
    constexpr u64 probe_cap = u64{64} << 20;   // one touch per page; the title's regions can be gigabytes, so only a lower bound
    const u64 readable_bytes = address ? x2w_local::probe_readable(static_cast<std::uintptr_t>(address), static_cast<std::size_t>(probe_cap)) : 0;
    u64 hash = 0;
    unsigned nonzero_rows[4] = {};
    if (row_bytes && readable_bytes >= row_bytes * 4) {
        const auto* data = reinterpret_cast<const unsigned char*>(address);
        hash = hash_memory(data, static_cast<std::size_t>(row_bytes * 4));
        for (unsigned row = 0; row < 4; ++row)
            for (u64 i = 0; i < row_bytes; ++i) nonzero_rows[row] += data[row * row_bytes + i] != 0;
    }
    log("x2w-d3d12 layout0 texture: fmt=%d %llux%ux%u mips=%u flags=0x%x align=%llu view{dim=%d fmt=%d} address=0x%llx "
        "addr_mod64k=0x%llx readable=%llu%s bpe=%u tight_mip0=%llu row_bytes=%llu first4rows_hash=%016llx nonzero_rows=%u/%u/%u/%u",
        resource->Format, static_cast<unsigned long long>(resource->Width), resource->Height, resource->DepthOrArraySize,
        resource->MipLevels, static_cast<UINT>(resource->Flags), static_cast<unsigned long long>(resource->Alignment),
        input->view.ViewDimension, input->view.Format, address, address & 0xFFFF, readable_bytes,
        readable_bytes >= probe_cap ? "+" : "", element.bytes, first_mip_bytes, row_bytes, hash, nonzero_rows[0],
        nonzero_rows[1], nonzero_rows[2], nonzero_rows[3]);
}

// Counts a refusal (memory_srv_census) and logs its line while the first 8 refusals last, and also for
// the first refusal of each of the first few distinct shapes (MemorySrvCensus::detail_shapes).
void refuse_memory_srv(LONG& refused, const char* what, const char* reason, const D3D12_RESOURCE_DESC* resource,
                       const MemoryShaderResourceView* input, u64 destination)
{
    const bool within_limit = first_calls(refused, 8);
    const bool new_shape = memory_srv_census.note(reason, x2w_local::readable(resource, sizeof(*resource)) ? resource : nullptr);
    if (within_limit || new_shape) log_refused_memory_srv(what, reason, resource, input, destination);
    log_layout0_texture(reason, resource, input);
}

// Counts a successful write (memory_srv_writes) and logs the first write to each of the first few
// distinct destination descriptors.
void note_memory_srv_write(const D3D12_RESOURCE_DESC* resource, const MemoryShaderResourceView* input, u64 destination)
{
    const u64 ordinal = memory_srv_writes.note(destination);
    if (!ordinal) return;
    char resource_text[224], view_text[256];
    x2w_startup::report_resource(resource_text, sizeof(resource_text), resource);
    x2w_startup::report_view(view_text, sizeof(view_text), &input->view);
    log("x2w-d3d12 memory SRV written %llu: dest=0x%llx address=0x%llx res{%s} view{%s}", ordinal, destination, input->address,
        resource_text, view_text);
}

// The placed buffer that holds `address`, as the pure resolver wants it; nothing found when no live one does.
x2w_startup::PlacedAsBuffer placed_as_buffer(const Wrapper* device, u64 address)
{
    x2w_startup::PlacedAsBuffer result;
    const auto placed = placed_buffer_range(device, address, 1);
    if (placed.reject) {
        result.ambiguous = true;
        return result;
    }
    if (!placed.owner || InterlockedCompareExchange(&placed.owner->own().refs, 0, 0) <= 0) return result;
    const auto* info = resource_info(placed.owner);
    auto* real = static_cast<ID3D12Resource*>(placed.owner->own().real);
    if (!info || !real) return result;
    result.found = true;
    result.title_base = reinterpret_cast<u64>(placed.owner->own().title_memory);
    result.gpu_base = real->GetGPUVirtualAddress();
    result.bytes = info->bytes;
    result.initial_state = info->initial_state;
    return result;
}

// Slot 70 with an acceleration-structure view: published through slot 18 with no resource and the
// desktop address; a refusal leaves the destination descriptor as it was.
void publish_as_srv(Wrapper* self, const D3D12_RESOURCE_DESC* resource, const MemoryShaderResourceView* input, u64 destination)
{
    static LONG refused;
    const u64 location = x2w_startup::as_srv_location(input->view, input->address);
    D3D12_SHADER_RESOURCE_VIEW_DESC resolved{};
    const auto refusal = x2w_startup::resolve_as_srv_desc(location, placed_as_buffer(self, location), input->view, resolved);
    if (refusal != x2w_startup::AsSrvRefusal::none) {
        refuse_memory_srv(refused, "unavailable acceleration structure at the supplied address",
                          x2w_startup::as_srv_refusal_name(refusal), resource, input, destination);
        return;
    }
    view_slot<18, 2>(self, 0, reinterpret_cast<u64>(&resolved), destination, 0);
}

// Refused buffer SRVs are remembered for a short time (srv_refusal_cache.h): the same view of the same address is
// published every frame, and each refusal used to repeat the placement scans and the memory questions.
x2w_startup::SrvRefusalCache& buffer_refusal_cache()
{
    static x2w_startup::SrvRefusalCache cache;
    return cache;
}

x2w_startup::SrvRefusalKey buffer_refusal_key(const Wrapper* device, const D3D12_RESOURCE_DESC& resource,
                                              const MemoryShaderResourceView& input, u64 span_bytes)
{
    const auto& view = input.view;
    x2w_startup::SrvRefusalKey key;
    key.device = reinterpret_cast<u64>(device);
    key.address = input.address;
    key.span_bytes = span_bytes;
    key.view_a = (u64{static_cast<UINT>(view.Format)} << 32) | (u64{static_cast<UINT>(view.ViewDimension)} << 16) | view.Shader4ComponentMapping;
    key.view_b = view.Buffer.FirstElement;
    key.view_c = (u64{view.Buffer.NumElements} << 32) | (u64{view.Buffer.StructureByteStride} << 8) | static_cast<UINT>(view.Buffer.Flags);
    key.resource_a = (u64{static_cast<UINT>(resource.Dimension)} << 32) | static_cast<UINT>(resource.Flags);
    key.resource_b = resource.Width;
    return key;
}

// The generation first, then the allocation: a change between the two reads makes the entry stale, never fresh.
x2w_startup::SrvRefusalStamp buffer_refusal_stamp(u64 address)
{
    x2w_startup::SrvRefusalStamp stamp;
    stamp.generation = x2w_startup::srv_placement_generation().load(std::memory_order_acquire);
    stamp.valid = x2w_startup::region_stamp(reinterpret_cast<const void*>(address), stamp.base, stamp.size, stamp.commit, stamp.flags);
    return stamp;
}

// A cached witness_page refusal still holds while its page is not committed in the allocation the walk started from.
bool buffer_refusal_page_still_uncommitted(const x2w_startup::SrvRefusalProof& proof)
{
    MEMORY_BASIC_INFORMATION page{};
    if (VirtualQuery(reinterpret_cast<const void*>(proof.witness), &page, sizeof(page)) != sizeof(page)) return true;
    return page.State != MEM_COMMIT || (proof.base != 0 && reinterpret_cast<u64>(page.AllocationBase) != proof.base);
}

void __stdcall device_memory_srv(Wrapper* self, const D3D12_RESOURCE_DESC* resource,
                                 const MemoryShaderResourceView* input, u64 destination)
{
    X2W_ZONE("device_memory_srv");
    static LONG refused;
    check_resource_address_index();
    if (const char* reason = memory_srv_refusal(resource, input, destination)) {
        refuse_memory_srv(refused, "unsupported memory SRV arguments", reason, resource, input, destination);
        return;
    }
    // A single-element texture is published by address like a tiled view: its texel is read at offset 0.
    if (x2w_startup::supports_single_element_srv(*resource, input->view)) {
        if (gpu_written_overlap(input->address, x2w_startup::single_element_bytes(*resource))) {
            refuse_memory_srv(refused, "GPU-written resource at the supplied address", "gpu_written_resource", resource, input,
                              destination);
        } else if (Wrapper* texture = publish_tiled_texture(self, input->address, *resource)) {
            view_slot<18, 2>(self, reinterpret_cast<u64>(texture), reinterpret_cast<u64>(&input->view), destination, 0);
            note_memory_srv_write(resource, input, destination);
        } else {
            refuse_memory_srv(refused, "unavailable texture at the supplied base and layout",
                              texture_unavailable_reason(self, input->address, *resource, true), resource, input, destination);
        }
        return;
    }
    if (x2w_startup::is_as_srv(input->view)) {
        publish_as_srv(self, resource, input, destination);
        return;
    }
    if (resource->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        const u64 elem_bytes = x2w_startup::buffer_srv_element_bytes(input->view);
        const u64 span_bytes = (input->view.Buffer.FirstElement + input->view.Buffer.NumElements) * elem_bytes;
        const auto refusal_key = buffer_refusal_key(self, *resource, *input, span_bytes);
        const auto refusal_stamp = buffer_refusal_stamp(input->address);
        if (buffer_refusal_cache().hit(refusal_key, refusal_stamp, GetTickCount64(), buffer_refusal_page_still_uncommitted)) {
            refuse_memory_srv(refused, "unavailable buffer at the supplied address", "buffer_unavailable", resource, input,
                              destination);
            return;
        }
        Wrapper* buffer = existing_memory_buffer(self, input->address, span_bytes);
        x2w_startup::SrvRefusalProof refusal_proof;
        if (!buffer) {
            buffer = unplaced_memory_buffer(self, input->address, span_bytes, *resource, elem_bytes, &refusal_proof);
        }
        const ResourceInfo* info = buffer ? resource_info(buffer) : nullptr;
        D3D12_SHADER_RESOURCE_VIEW_DESC adjusted{};
        if (!buffer || !info ||
            !x2w_startup::resolve_buffer_srv_desc(reinterpret_cast<u64>(buffer->own().title_memory),
                                                  info->console.Width, input->address, input->view, adjusted)) {
            // A buffer that was found (placed or unplaced) and does not fit the view is decided by its recorded size.
            if (buffer) refusal_proof = x2w_startup::SrvRefusalProof{x2w_startup::SrvRefusalKind::independent, 0, 0, true};
            buffer_refusal_cache().note(refusal_key, refusal_stamp, GetTickCount64(), refusal_proof);
            refuse_memory_srv(refused, "unavailable buffer at the supplied address", "buffer_unavailable", resource, input,
                              destination);
            return;
        }
        view_slot<18, 2>(self, reinterpret_cast<u64>(buffer), reinterpret_cast<u64>(&adjusted), destination, 0);
        note_memory_srv_write(resource, input, destination);
        return;
    }
    // A partial-tile view stays refused when the last XG stand-in copy to the address wrote desktop-footprint
    // rows, or when its padded tiles run past the size the stand-in reported, which the title may not own.
    bool first = false;
    const char* held = nullptr;
    if (partial_tiled_bc(*resource)) {
        {
            std::lock_guard lock(tiled_srv_mutex);
            first = !tiled_srv_addresses.count(input->address);
        }
        if (latest_texture_write_is_rows(input->address)) held = "partial_tile_xg_rows";
        else if (x2w_startup::Bc16Surface(*resource).size() > xg_reported_bytes(self, *resource))
            held = "partial_tile_beyond_xg_size";
    }
    Wrapper* texture = held ? nullptr
        : (tiled_bc_srv(*resource) || x2w_startup::bc64::supports(*resource)) ? publish_tiled_texture(self, input->address, *resource)
                                              : existing_memory_texture(input->address, *resource);
    if (first) log_partial_tile_view(*resource, input->address, held, texture != nullptr);
    // A new console-layout view reads the address only when there is no matching live resource.
    const char* outcome = tiled_bc_srv(*resource) || x2w_startup::bc64::supports(*resource) ? "published BC shape" : "existing resource";
    if (console_layout_srv(*resource)) {
        if ((!texture || !supports_placed_memory_srv(*resource, input->view)) &&
            !desktop_address_view_supported(self, *resource, input->view)) {
            held = "unsupported_desktop_format";
            texture = nullptr;
        } else if (!texture) {
            held = console_address_hold(self, input->address, *resource);
            if (!held && address_resource_overlap(input->address, address_texture_bytes(self, *resource)))
                held = "overlapping_resource";
            texture = held ? nullptr : publish_tiled_texture(self, input->address, *resource);
            outcome = "published console layout";
        }
    }
    // A publication was attempted for a BC / console-layout view; any other view only looks up a live resource.
    const char* unavailable = texture || held ? nullptr
        : texture_unavailable_reason(self, input->address, *resource,
                                     tiled_bc_srv(*resource) || x2w_startup::bc64::supports(*resource) || console_layout_srv(*resource));
    if (!texture) outcome = held ? held : unavailable;
    log_address_kind(outcome, *resource, input->view, input->address);
    if (!texture) {
        refuse_memory_srv(refused, "unavailable texture at the supplied base and layout", held ? held : unavailable,
                          resource, input, destination);
        return;
    }
    view_slot<18, 2>(self, reinterpret_cast<u64>(texture), reinterpret_cast<u64>(&input->view), destination, 0);
    note_memory_srv_write(resource, input, destination);
}
