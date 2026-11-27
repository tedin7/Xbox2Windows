// Native compute fixture: samplers read from the directly indexed sampler heap at a fixed index (issue #3076).
//
// usage: native_sampler_heap SDK CACHE REQ.c67 ADAPTER MODE
//   REQ.c67  a source-built compute request (tests/native_sampler_heap_check.py) whose root has
//            CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED and one SRV+UAV table.
//   CACHE    a fresh frontend cache directory: the fixture asserts a miss, then a hit on the same request.
//   ADAPTER  warp | hardware
//   MODE     sample   one s_load_dwordx4 at 0x430 (entry 67), image_sample_lz at a coordinate outside [0, 1]
//            two      one s_load_dwordx8 at 0x430, two samples with entries 67 and 68
//            lod      image_sample_l at LOD 1 of a two-mip texture; MaxLOD of the entry decides the mip
//            mixed    one literal static sampler and heap entry 67 in one program
//            unbound  bind() without a sampler heap throws
// The sampler heap has 128 entries, every one but the entries under test a decoy; each decoy is dispatched in place of
// the entry under test and must change the result. Entries are made with dynamic_sampler(), as the runtime does.
// Exit 0: every assertion held. 1: a result differs. 4: setup error. 77: capability skip.
#include "desktop_gpu.h"
#include "direct_compute_cache.h"
#include "native_compute_pipeline.h"
#include "startup_stubs/translated_sampler_bindings.h"
#include <dxgi1_4.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using x2w_startup::SamplerWords;

namespace {
constexpr unsigned lanes = 32, heap_entries = 128;
constexpr float u_coordinate = 1.375f, v_coordinate = -0.375f;
constexpr std::uint32_t sentinel = 0x69696969u;
using Texel = std::array<float, 4>;

// A 4x4 first mip and a 2x2 second mip with a distinct value in every texel and channel.
Texel texel(unsigned level, int x, int y) {
    if (level == 0) return {10.f * x + 1, 10.f * y + 2, 100.f + x + 4 * y, 1000.f};
    return {500.f + x, 600.f + y, 700.f + x + 2 * y, 2000.f};
}

enum Address : unsigned { wrap = 0, mirror = 1, clamp = 2, border = 6 };
// Title SQ_IMG_SAMP words in the encoding decode_sampler reads: clamp modes in word 0, MAX_LOD in word 1 (4.8 fixed
// point), filters and mip filter in word 2 (mip filter 1 = point), border colour type in word 3 (2 = opaque white).
SamplerWords title_sampler(Address address, bool bilinear, unsigned max_lod, bool white_border = false) {
    const unsigned a = address, filter = bilinear ? 1u : 0u;
    return {a | a << 3 | a << 6, (max_lod * 256u) << 12, filter << 20 | filter << 22 | 1u << 26,
            white_border ? 2u << 30 : 0u};
}

// CPU model of a point-filtered sample at (u_coordinate, v_coordinate): the texel the address mode picks at the level the
// requested LOD gives after the sampler's MAX_LOD clamp.
Texel model(const SamplerWords &words, unsigned requested_lod) {
    const unsigned level = (std::min)(requested_lod, (words[1] >> 12 & 0xfffu) / 256u);
    const int n = 4 >> level;
    int index[2]{int(std::floor(u_coordinate * n)), int(std::floor(v_coordinate * n))};
    bool outside = false;
    for (int &i : index) {
        switch (words[0] & 7u) {
        case wrap: i = ((i % n) + n) % n; break;
        case mirror: {
            const int m = ((i % (2 * n)) + 2 * n) % (2 * n);
            i = m < n ? m : 2 * n - 1 - m;
            break;
        }
        case clamp: i = (std::max)(0, (std::min)(i, n - 1)); break;
        default: outside |= i < 0 || i >= n; break;
        }
    }
    if (outside) {
        const float white = (words[3] >> 30) == 2 ? 1.f : 0.f;
        return {white, white, white, white};
    }
    return texel(level, index[0], index[1]);
}

// One heap entry under test: the float4 of `result` must equal `expected` and a decoy in its place must change it.
struct Probe {
    unsigned entry, result;
    SamplerWords words;
    unsigned lod;
};
struct Mode {
    std::vector<Probe> probes;
    std::vector<SamplerWords> decoys;
    unsigned results = 1, statics = 0;
    std::map<unsigned, Texel> fixed; // results that do not come from the heap (the static sampler)
    unsigned highest_entry = 0;
};

Mode make_mode(const std::wstring &name) {
    Mode mode;
    const auto wrap_point = title_sampler(wrap, false, 0), clamp_point = title_sampler(clamp, false, 0),
               mirror_point = title_sampler(mirror, false, 0), border_clear = title_sampler(border, false, 0),
               border_white = title_sampler(border, true, 0, true);
    if (name == L"sample") {
        mode.probes = {{67, 0, wrap_point, 0}};
        mode.decoys = {clamp_point, mirror_point, border_clear, border_white};
    } else if (name == L"two") {
        mode.results = 2;
        mode.probes = {{67, 0, wrap_point, 0}, {68, 1, clamp_point, 0}};
        mode.decoys = {mirror_point, border_clear, border_white};
    } else if (name == L"lod") {
        mode.probes = {{67, 0, title_sampler(wrap, false, 1), 1}};
        // MaxLOD 0 with a bilinear and a point filter (the LOD clamp must matter), and a wrong address mode at MaxLOD 1.
        mode.decoys = {title_sampler(wrap, true, 0), title_sampler(wrap, false, 0), title_sampler(clamp, false, 1)};
    } else if (name == L"mixed") {
        mode.results = 2;
        mode.statics = 1;
        mode.probes = {{67, 1, wrap_point, 0}};
        mode.decoys = {clamp_point, mirror_point, border_clear};
        mode.fixed[0] = model(SamplerWords{0x92u, 0, 0, 0}, 0); // the literal static sampler: clamp, point
    }
    for (const auto &probe : mode.probes) mode.highest_entry = (std::max)(mode.highest_entry, probe.entry);
    return mode;
}

struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    UINT64 fence_value = 0;
    ComPtr<ID3D12DescriptorHeap> views;
    ComPtr<ID3D12Resource> texture, output, readback, clear;
    UINT view_increment = 0, sampler_increment = 0;
    UINT output_bytes = 0;

    ~Gpu() {
        if (event) CloseHandle(event);
    }

    // Returns 77 when the adapter cannot run the fixture; 0 when ready.
    int open(const std::wstring &adapter) {
        ComPtr<IDXGIFactory4> factory;
        ComPtr<IDXGIAdapter> chosen;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 77;
        if (adapter == L"warp") {
            if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&chosen)))) return 77;
        } else {
            for (UINT i = 0; i < 8; ++i) {
                ComPtr<IDXGIAdapter1> candidate;
                if (factory->EnumAdapters1(i, &candidate) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 desc{};
                candidate->GetDesc1(&desc);
                if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                    chosen = candidate;
                    std::printf("adapter=%ls\n", desc.Description);
                    break;
                }
            }
            if (!chosen) { std::puts("no hardware adapter"); return 77; }
        }
        if (FAILED(D3D12CreateDevice(chosen.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
            std::puts("no device");
            return 77;
        }
        D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        if (FAILED(check_feature(device.Get(), D3D12_FEATURE_SHADER_MODEL, &model, sizeof model)) ||
            model.HighestShaderModel < D3D_SHADER_MODEL_6_6 ||
            FAILED(check_feature(device.Get(), D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof options)) ||
            options.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
            std::printf("adapter lacks shader model 6.6 or resource binding tier 3 (tier %d)\n",
                        int(options.ResourceBindingTier));
            return 77;
        }
        // The native module declares WaveSize=32 (the guest wave), so an adapter without 32-lane waves cannot run it.
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 waves{};
        if (FAILED(check_feature(device.Get(), D3D12_FEATURE_D3D12_OPTIONS1, &waves, sizeof waves)) || !waves.WaveOps ||
            waves.WaveLaneCountMin > 32 || waves.WaveLaneCountMax < 32) {
            std::printf("adapter has no 32-lane waves (%u..%u)\n", waves.WaveLaneCountMin, waves.WaveLaneCountMax);
            return 77;
        }
        // An older runtime refuses roots with the heap-indexing flags (E_INVALIDARG): a capability skip, not a failure.
        {
            D3D12_ROOT_SIGNATURE_DESC probe{0, nullptr, 0, nullptr, static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(0xc00)};
            ComPtr<ID3DBlob> blob;
            ComPtr<ID3D12RootSignature> root;
            if (FAILED(D3D12SerializeRootSignature(&probe, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr)) ||
                FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                   IID_PPV_ARGS(&root)))) {
                std::puts("adapter refuses a root with the heap-indexing flags 0xc00");
                return 77;
            }
        }
        D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "queue");
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "allocator");
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
              "list");
        check(list->Close(), "close");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "fence");
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) throw std::runtime_error("event");
        view_increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        sampler_increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        return 0;
    }

    void wait() {
        check(queue->Signal(fence.Get(), ++fence_value), "signal");
        check(fence->SetEventOnCompletion(fence_value, event), "fence event");
        if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0) throw std::runtime_error("fence wait timed out");
    }
    template<class Body> void run(Body body) {
        check(allocator->Reset(), "allocator reset");
        check(list->Reset(allocator.Get(), nullptr), "list reset");
        body(list.Get());
        check(list->Close(), "list close");
        ID3D12CommandList *lists[]{list.Get()};
        queue->ExecuteCommandLists(1, lists);
        wait();
        check(device->GetDeviceRemovedReason(), "device removed");
    }
    static D3D12_RESOURCE_BARRIER barrier(ID3D12Resource *resource, D3D12_RESOURCE_STATES before,
                                          D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        return b;
    }

    // t0 of the guest table: a 4x4 two-mip texture; u0: one float4 per result and lane, as a raw buffer.
    void create_resources(unsigned results) {
        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = td.Height = 4;
        td.DepthOrArraySize = 1;
        td.MipLevels = 2;
        td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        td.SampleDesc.Count = 1;
        texture = resource(device.Get(), td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[2]{};
        UINT64 total = 0;
        device->GetCopyableFootprints(&td, 0, 2, 0, footprints, nullptr, nullptr, &total);
        auto upload = make_buffer(device.Get(), total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void *mapped = nullptr;
        check(upload->Map(0, nullptr, &mapped), "texture upload map");
        for (unsigned level = 0; level < 2; ++level)
            for (int y = 0; y < (4 >> level); ++y)
                for (int x = 0; x < (4 >> level); ++x)
                    std::memcpy(static_cast<char *>(mapped) + footprints[level].Offset +
                                    y * footprints[level].Footprint.RowPitch + x * sizeof(Texel),
                                texel(level, x, y).data(), sizeof(Texel));
        upload->Unmap(0, nullptr);
        run([&](ID3D12GraphicsCommandList *l) {
            for (UINT level = 0; level < 2; ++level) {
                D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT},
                    dst{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
                src.PlacedFootprint = footprints[level];
                dst.SubresourceIndex = level;
                l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            auto to_read = barrier(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            l->ResourceBarrier(1, &to_read);
        });
        output_bytes = lanes * results * sizeof(Texel);
        output = make_buffer(device.Get(), output_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        readback = make_buffer(device.Get(), output_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        clear = make_buffer(device.Get(), output_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        check(clear->Map(0, nullptr, &mapped), "sentinel map");
        for (unsigned i = 0; i < output_bytes / 4; ++i) static_cast<std::uint32_t *>(mapped)[i] = sentinel;
        clear->Unmap(0, nullptr);
        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "view heap");
        auto cpu = views->GetCPUDescriptorHandleForHeapStart();
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = td.Format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 2;
        device->CreateShaderResourceView(texture.Get(), &srv, cpu);
        cpu.ptr += view_increment;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = output_bytes / 4;
        uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(output.Get(), nullptr, &uav, cpu);
    }

    // The sampler heap of the run: every entry a decoy (cycling through `decoys`), `entries` replaced.
    ComPtr<ID3D12DescriptorHeap> sampler_heap(const std::vector<SamplerWords> &decoys,
                                              const std::map<unsigned, SamplerWords> &entries) {
        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, heap_entries, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        ComPtr<ID3D12DescriptorHeap> heap;
        check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "sampler heap");
        auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
        for (unsigned i = 0; i < heap_entries; ++i) {
            auto found = entries.find(i);
            const auto desc = x2w_startup::dynamic_sampler(found != entries.end() ? found->second : decoys[i % decoys.size()]);
            device->CreateSampler(&desc, cpu);
            cpu.ptr += sampler_increment;
        }
        return heap;
    }

    // Fills u0 with a sentinel, dispatches one group of 32 lanes, reads u0 back as floats.
    std::vector<float> dispatch(const x2w::NativeComputePipeline &pipeline, ID3D12DescriptorHeap *samplers) {
        run([&](ID3D12GraphicsCommandList *l) {
            auto to_copy = barrier(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
            l->ResourceBarrier(1, &to_copy);
            l->CopyBufferRegion(output.Get(), 0, clear.Get(), 0, output_bytes);
            auto to_uav = barrier(output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            l->ResourceBarrier(1, &to_uav);
            pipeline.bind(l, views.Get(), views->GetGPUDescriptorHandleForHeapStart(), view_increment, nullptr, {},
                          samplers);
            l->Dispatch(1, 1, 1);
            auto to_source = barrier(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            l->ResourceBarrier(1, &to_source);
            l->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, output_bytes);
            auto back = barrier(output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            l->ResourceBarrier(1, &back);
        });
        std::vector<float> out(output_bytes / 4);
        void *mapped = nullptr;
        D3D12_RANGE range{0, output_bytes};
        check(readback->Map(0, &range, &mapped), "readback map");
        std::memcpy(out.data(), mapped, output_bytes);
        D3D12_RANGE written{};
        readback->Unmap(0, &written);
        return out;
    }
};

bool equal(const Texel &a, const Texel &b) { return std::memcmp(a.data(), b.data(), sizeof a) == 0; }
const char *texel_text(const Texel &t, char (&buffer)[96]) {
    std::snprintf(buffer, sizeof buffer, "(%g, %g, %g, %g)", t[0], t[1], t[2], t[3]);
    return buffer;
}

// 0: every lane holds `expected` in float4 `result`. Otherwise the first mismatch is printed.
bool lanes_hold(const std::vector<float> &values, unsigned results, unsigned result, const Texel &expected,
                const char *what) {
    for (unsigned lane = 0; lane < lanes; ++lane) {
        Texel got;
        std::memcpy(got.data(), &values[(lane * results + result) * 4], sizeof got);
        if (!equal(got, expected)) {
            char a[96], b[96];
            std::printf("FAIL: %s: lane %u result %u is %s, expected %s\n", what, lane, result, texel_text(got, a),
                        texel_text(expected, b));
            return false;
        }
    }
    return true;
}
// Every lane must hold a result other than `expected`, and not the untouched output sentinel (the dispatch ran).
bool lanes_differ(const std::vector<float> &values, unsigned results, unsigned result, const Texel &expected) {
    for (unsigned lane = 0; lane < lanes; ++lane) {
        Texel got;
        std::memcpy(got.data(), &values[(lane * results + result) * 4], sizeof got);
        std::uint32_t bits[4];
        std::memcpy(bits, got.data(), sizeof bits);
        if (equal(got, expected) || (bits[0] == sentinel && bits[1] == sentinel && bits[2] == sentinel &&
                                     bits[3] == sentinel))
            return false;
    }
    return true;
}

int run_mode(const std::wstring &sdk, const std::wstring &cache, const std::filesystem::path &marker, Gpu &gpu,
             const std::wstring &name) {
    const bool unbound = name == L"unbound";
    const Mode mode = make_mode(unbound ? L"sample" : name);
    if (mode.probes.empty()) {
        std::puts("unknown mode");
        return 2;
    }
    // Frontend cache: a miss on the fresh directory, then a hit with the same binding record.
    bool first_hit = true, second_hit = false;
    const auto first = x2w::prepare_direct_compute(marker, 0, false, cache, first_hit);
    const auto second = x2w::prepare_direct_compute(marker, 0, false, cache, second_hit);
    if (first_hit || !second_hit) {
        std::printf("FAIL: frontend cache first call hit=%d second call hit=%d\n", int(first_hit), int(second_hit));
        return 1;
    }
    std::puts("pass: prepare_direct_compute misses on a fresh cache and hits on the second call");
    for (const auto *program : {&first, &second}) {
        unsigned count = 0;
        for (const auto &b : program->bindings) {
            if (b.kind != x2w::ComputeViewKind::heap_sampler) continue;
            ++count;
            if (b.root_parameter != x2w::compute_sampler_heap || b.byte_offset != 0 || b.shader_register != 0 ||
                b.required_bytes != 16 * (mode.highest_entry + 1) || b.array_count != 1 || b.array_position != 0 ||
                b.image_dimension != 1 || b.scalar_table != x2w::ComputeViewBinding::ScalarTable::unknown) {
                std::printf("FAIL: heap_sampler binding root=%u offset=%u register=%u bytes=%u\n", b.root_parameter,
                            b.byte_offset, b.shader_register, b.required_bytes);
                return 1;
            }
        }
        if (count != 1 || program->samplers.size() != mode.statics) {
            std::printf("FAIL: %u heap_sampler bindings, %zu static samplers\n", count, program->samplers.size());
            return 1;
        }
    }
    std::printf("pass: one heap_sampler binding {compute_sampler_heap, 0, 0, %u bytes} in the computed and the cached "
                "program, %u static samplers\n", 16 * (mode.highest_entry + 1), mode.statics);

    x2w::ShaderDxilCompiler compiler(sdk);
    auto pipeline = x2w::create_native_compute_pipeline(gpu.device.Get(), compiler, marker, 0, cache, true);
    if (!pipeline.sampler_heap_indexed || pipeline.heap_indexed || pipeline.srv_count != 1 || pipeline.uav_count != 1 ||
        !pipeline.frontend_cache_hit) {
        std::printf("FAIL: pipeline shape sampler_heap_indexed=%d heap_indexed=%d srv=%u uav=%u frontend_hit=%d\n",
                    int(pipeline.sampler_heap_indexed), int(pipeline.heap_indexed), pipeline.srv_count,
                    pipeline.uav_count, int(pipeline.frontend_cache_hit));
        return 1;
    }
    std::puts("pass: pipeline has sampler_heap_indexed, no resource-heap indexing, one SRV and one UAV");
    gpu.create_resources(mode.results);

    if (unbound) {
        // bind() without a sampler heap must throw before anything is recorded for it.
        bool threw = false;
        try {
            gpu.run([&](ID3D12GraphicsCommandList *l) {
                pipeline.bind(l, gpu.views.Get(), gpu.views->GetGPUDescriptorHandleForHeapStart(), gpu.view_increment);
            });
        } catch (const std::runtime_error &error) {
            threw = std::string(error.what()) == "a sampler heap is needed";
            std::printf("bind without a sampler heap: %s\n", error.what());
        }
        if (!threw) {
            std::puts("FAIL: bind without a sampler heap did not throw");
            return 1;
        }
        std::puts("pass: bind without a sampler heap throws \"a sampler heap is needed\"");
        return 0;
    }

    std::map<unsigned, SamplerWords> entries;
    for (const auto &probe : mode.probes) entries[probe.entry] = probe.words;
    const auto values = gpu.dispatch(pipeline, gpu.sampler_heap(mode.decoys, entries).Get());
    for (const auto &[result, expected] : mode.fixed)
        if (!lanes_hold(values, mode.results, result, expected, "static sampler")) return 1;
    if (!mode.fixed.empty()) std::puts("pass: the literal static sampler result matches the CPU model");
    for (const auto &probe : mode.probes) {
        const Texel expected = model(probe.words, probe.lod);
        if (!lanes_hold(values, mode.results, probe.result, expected, "heap entry")) return 1;
        std::printf("pass: entry %u (result %u) equals the CPU model in all %u lanes\n", probe.entry, probe.result, lanes);
        // The same program with a decoy in place of the entry under test must give another result.
        for (size_t d = 0; d < mode.decoys.size(); ++d) {
            auto swapped = entries;
            swapped[probe.entry] = mode.decoys[d];
            const auto other = gpu.dispatch(pipeline, gpu.sampler_heap(mode.decoys, swapped).Get());
            if (!lanes_differ(other, mode.results, probe.result, expected)) {
                std::printf("FAIL: decoy %zu in place of entry %u leaves result %u unchanged\n", d, probe.entry,
                            probe.result);
                return 1;
            }
            std::printf("pass: decoy %zu in place of entry %u changes result %u\n", d, probe.entry, probe.result);
        }
    }
    return 0;
}
} // namespace

int wmain(int argc, wchar_t **argv) {
    if (argc != 6) {
        std::fputs("usage: native_sampler_heap SDK CACHE REQ.c67 ADAPTER sample|two|lod|mixed|unbound\n", stderr);
        return 2;
    }
    Gpu gpu;
    try {
        if (int skip = gpu.open(argv[4])) return skip;
        return run_mode(argv[1], argv[2], argv[3], gpu, argv[5]);
    } catch (const std::exception &error) {
        std::printf("setup error: %s\n", error.what());
        return 4;
    }
}
