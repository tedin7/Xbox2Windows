#pragma once

#include "desktop_gpu.h"
#include "startup_stubs/placed_build.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace dxr_fixture {
// These are the existing slot-133 and IUnknown contracts, called on the
// same shim object that owns the production placement index.
class PlacedBuildScene {
    struct Buffer {
        void* memory{};
        Microsoft::WRL::ComPtr<IUnknown> wrapper;
        ~Buffer() { wrapper.Reset(); if (memory) VirtualFree(memory, 0, MEM_RELEASE); }
    };
    Microsoft::WRL::ComPtr<IUnknown> guest_;
    Microsoft::WRL::ComPtr<IUnknown> cpu_heap_;
    std::array<Buffer, 9> buffers_;
    UINT64 big_as_{};  // an acceleration-structure buffer larger than any scratch range the controls use
    UINT64 tlas_title_{}, scratch_title_{};
    struct Placed {
        UINT64 dest{}, scratch{};
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO sizes{};
    };
    std::array<Placed, 2> titles_; // BLAS, TLAS as built by prepare()

public:
    using Cause = x2w_startup::PlacedBuildCause;
    // Wrong-routing control (issue #874): also build a decoy BLAS from another triangle into its own placed
    // destination and scratch, and route the TLAS instance to the decoy instead of the BLAS under test.
    bool misroute = false;
    // Slot-70 variant (issue #1938): the TLAS is handed to the trace as an acceleration-structure view that
    // device slot 70 published, instead of as a root address.
    bool slot70 = false;
    void attach(ID3D12Device* device) {
        device->AddRef();
        x2w_startup::attach_device(device, reinterpret_cast<void**>(guest_.GetAddressOf()));
    }
    void* guest() const { return guest_.Get(); }
    unsigned release() {
        unsigned count = 0;
        cpu_heap_.Reset();
        for (auto& buffer : buffers_) {
            if (auto* wrapper = buffer.wrapper.Detach()) {
                if (wrapper->Release()) throw std::runtime_error("Placed resource still referenced");
                ++count;
            }
        }
        return count;
    }
    UINT64 place(unsigned index, UINT64 bytes, D3D12_RESOURCE_STATES state) {
        auto& buffer = buffers_.at(index);
        buffer.memory = VirtualAlloc(nullptr, static_cast<SIZE_T>(bytes), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!buffer.memory) throw std::bad_alloc();
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        using Place = HRESULT (WINAPI*)(void*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, const GUID*, void**);
        auto slot133 = reinterpret_cast<Place>((*reinterpret_cast<void***>(guest_.Get()))[133]);
        check(slot133(guest_.Get(), buffer.memory, &desc, state, nullptr, &__uuidof(ID3D12Resource),
            reinterpret_cast<void**>(buffer.wrapper.GetAddressOf())), "Register slot-133 DXR buffer");
        return reinterpret_cast<UINT64>(buffer.memory);
    }

    void prepare(ID3D12Device* device,
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& bottom_size,
        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& top_size,
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& bottom,
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& top) {
        attach(device);
        constexpr UINT64 dest_offset = 512, scratch_offset = 768;
        constexpr auto dest_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        constexpr auto scratch_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        if (!bottom_size.ScratchDataSizeInBytes || !top_size.ScratchDataSizeInBytes)
            throw std::runtime_error("This scratch-boundary fixture requires nonzero BLAS and TLAS scratch sizes");
        // A scratch buffer serves the build and, later, the update, so it holds the larger of the two.
        const auto scratch_bytes = [](const auto& sizes) {
            return (std::max)(sizes.ScratchDataSizeInBytes, sizes.UpdateScratchDataSizeInBytes);
        };
        const auto blas = place(0, dest_offset + bottom_size.ResultDataMaxSizeInBytes, dest_state) + dest_offset;
        const auto tlas = place(1, dest_offset + top_size.ResultDataMaxSizeInBytes, dest_state) + dest_offset;
        const auto bs = place(2, scratch_offset + scratch_bytes(bottom_size), scratch_state) + scratch_offset;
        const auto ts = place(3, scratch_offset + scratch_bytes(top_size), scratch_state) + scratch_offset;
        titles_[0] = {blas, bs, bottom_size};
        titles_[1] = {tlas, ts, top_size};
        big_as_ = place(8, 4096, dest_state);
        const auto short_scratch = place(4, scratch_offset + top_size.ScratchDataSizeInBytes - 1, scratch_state) + scratch_offset;
        constexpr UINT64 unmapped = 0x1000; // outside the user allocation range
        // Each refusal names its cause (not only E_INVALIDARG) and publishes nothing.
        const auto must_reject = [&](UINT64 dest, UINT64 scratch, Cause expected, const char* what) {
            auto candidate = top;
            Cause cause = Cause::ok;
            const HRESULT hr = x2w_startup::resolve_placed_build(guest_.Get(), dest, scratch, top_size, candidate, &cause);
            if (hr != E_INVALIDARG || candidate.DestAccelerationStructureData || candidate.ScratchAccelerationStructureData)
                throw std::runtime_error(std::string("Invalid DXR build was accepted or partially published: ") + what);
            if (cause != expected)
                throw std::runtime_error(std::string("DXR build refusal names the wrong cause (") + x2w_startup::cause_name(cause) +
                                         " for " + what + ", expected " + x2w_startup::cause_name(expected) + ")");
        };
        must_reject(unmapped, ts, Cause::not_placed, "destination outside the placed buffers");
        must_reject(tlas + 16, ts, Cause::unaligned, "unaligned destination");
        must_reject(ts, tlas, Cause::wrong_state, "destination in a buffer that is not an acceleration structure");
        must_reject(tlas, short_scratch, Cause::not_placed, "scratch shorter than the build needs");
        must_reject(tlas, unmapped, Cause::not_placed, "scratch outside the placed buffers");
        must_reject(tlas, ts + 16, Cause::unaligned, "unaligned scratch");
        must_reject(tlas, big_as_, Cause::wrong_state, "scratch in an acceleration-structure buffer");
        check(x2w_startup::resolve_placed_build(guest_.Get(), blas, bs, bottom_size, bottom), "Resolve placed BLAS");
        check(x2w_startup::resolve_placed_build(guest_.Get(), tlas, ts, top_size, top), "Resolve placed TLAS");
        tlas_title_ = tlas;
        scratch_title_ = ts;
        const auto verify_offset = [&](const auto& build, UINT64 dest, UINT64 scratch, const auto& sizes) {
            auto base = build;
            check(x2w_startup::resolve_placed_build(guest_.Get(), dest - dest_offset, scratch - scratch_offset,
                sizes, base), "Resolve placement bases");
            if (build.DestAccelerationStructureData != base.DestAccelerationStructureData + dest_offset ||
                build.ScratchAccelerationStructureData != base.ScratchAccelerationStructureData + scratch_offset)
                throw std::runtime_error("The translated DXR VA lost its interior offset");
        };
        verify_offset(bottom, blas, bs, bottom_size);
        verify_offset(top, tlas, ts, top_size);
        // The base resolves above are builds over the same memory, so the resolver's record of the real builds was replaced:
        // resolve the real ones again, then run the update controls against them.
        check(x2w_startup::resolve_placed_build(guest_.Get(), blas, bs, bottom_size, bottom), "Resolve placed BLAS again");
        check(x2w_startup::resolve_placed_build(guest_.Get(), tlas, ts, top_size, top), "Resolve placed TLAS again");
        check_updates(top, top_size, tlas, ts);
        // check_updates records the TLAS as built with ALLOW_UPDATE whatever flags the caller gave it: record the builds as given.
        check(x2w_startup::resolve_placed_build(guest_.Get(), blas, bs, bottom_size, bottom), "Resolve placed BLAS as given");
        check(x2w_startup::resolve_placed_build(guest_.Get(), tlas, ts, top_size, top), "Resolve placed TLAS as given");
        std::printf("Placed DXR bytes dest/scratch: BLAS=%llu/%llu TLAS=%llu/%llu; short scratch, missing dest and swapped states rejected before recording\n",
            bottom_size.ResultDataMaxSizeInBytes, bottom_size.ScratchDataSizeInBytes,
            top_size.ResultDataMaxSizeInBytes, top_size.ScratchDataSizeInBytes);
    }

    // PERFORM_UPDATE with SourceAccelerationStructureData, through the same resolver. The scratch of an update
    // is sized by UpdateScratchDataSizeInBytes; the fixture sets it to the build scratch size so that the one
    // scratch buffer placed for the build fits. Needs the BLAS/TLAS builds of prepare() to be recorded already.
    void check_updates(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& top,
                       const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& top_size, UINT64 tlas, UINT64 scratch) {
        constexpr auto update_flag = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        constexpr auto allow_flag = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
        constexpr UINT64 dest_offset = 512, unmapped = 0x1000;
        auto sizes = top_size;
        sizes.UpdateScratchDataSizeInBytes = top_size.ScratchDataSizeInBytes;
        const auto request = [&](UINT64 source, bool update) {
            auto build = top;
            // A valid update repeats the source's flags, ALLOW_UPDATE included (the structure it writes stays updatable).
            build.Inputs.Flags = update ? (build.Inputs.Flags | update_flag | allow_flag) : (build.Inputs.Flags & ~update_flag);
            build.SourceAccelerationStructureData = source;
            build.DestAccelerationStructureData = build.ScratchAccelerationStructureData = 0;
            return build;
        };
        // Each refusal returns E_INVALIDARG, names its cause, publishes neither address and keeps the source as the title gave it.
        const auto must_reject = [&](UINT64 dest, UINT64 source, UINT64 scratch_address, const auto& use, bool update,
                                     Cause expected, const char* what) {
            auto build = request(source, update);
            Cause cause = Cause::ok;
            const HRESULT hr = x2w_startup::resolve_placed_build(guest_.Get(), dest, scratch_address, use, build, &cause);
            if (hr != E_INVALIDARG || build.DestAccelerationStructureData || build.ScratchAccelerationStructureData ||
                build.SourceAccelerationStructureData != source)
                throw std::runtime_error(std::string("Invalid DXR update was accepted or partially published: ") + what);
            if (cause != expected)
                throw std::runtime_error(std::string("DXR update refusal names the wrong cause (") + x2w_startup::cause_name(cause) +
                                         " for " + what + ", expected " + x2w_startup::cause_name(expected) + ")");
        };
        must_reject(tlas, 0, scratch, sizes, true, Cause::update_without_source, "update without a source");
        must_reject(tlas, tlas, scratch, sizes, false, Cause::source_without_update, "source without PERFORM_UPDATE");
        must_reject(tlas, tlas + 1, scratch, sizes, true, Cause::source_unaligned, "unaligned source");
        must_reject(tlas, unmapped, scratch, sizes, true, Cause::source_not_placed, "source outside the placed buffers");
        must_reject(tlas, scratch, scratch, sizes, true, Cause::source_wrong_state,
                    "source in a buffer that is not an acceleration structure");
        // An aligned source one stride (256 bytes) past the destination: where the structure is longer than that the two
        // partly overlap (the shape rule, checked by value in placed_build_causes names it); on a device that sizes it
        // shorter, the source lies past the end of the destination's buffer.
        const Cause stride_cause = x2w_startup::check_update_shape(true, tlas + 256, tlas, sizes.ResultDataMaxSizeInBytes);
        must_reject(tlas, tlas + 256, scratch, sizes, true,
                    stride_cause != Cause::ok ? stride_cause : Cause::source_not_placed, "source one stride past the destination");
        if (stride_cause != Cause::ok && stride_cause != Cause::source_dest_partial_overlap)
            throw std::runtime_error("The shape rule named an unexpected cause for a source one stride past the destination");
        // The scratch rule applies to an update as to a build.
        must_reject(tlas, tlas, big_as_, sizes, true, Cause::wrong_state, "update with a scratch in an acceleration-structure buffer");

        // The update controls below need the TLAS recorded as built with ALLOW_UPDATE, whatever flags the caller built it with.
        auto recorded = request(0, false);
        recorded.Inputs.Flags |= allow_flag;
        check(x2w_startup::resolve_placed_build(guest_.Get(), tlas, scratch, top_size, recorded), "Record the TLAS with ALLOW_UPDATE");

        // In place: the source is the destination.
        auto in_place = request(tlas, true);
        check(x2w_startup::resolve_placed_build(guest_.Get(), tlas, scratch, sizes, in_place), "Resolve in-place update");
        if (!in_place.DestAccelerationStructureData || in_place.SourceAccelerationStructureData != in_place.DestAccelerationStructureData)
            throw std::runtime_error("An in-place update must translate source and destination to the same address");

        // A device that reports no update scratch (WARP): a placed scratch and a null scratch both resolve, and a
        // scratch outside every placed buffer is still refused.
        auto none = sizes;
        none.UpdateScratchDataSizeInBytes = 0;
        for (const UINT64 given : {scratch, UINT64{0}}) {
            auto build = request(tlas, true);
            check(x2w_startup::resolve_placed_build(guest_.Get(), tlas, given, none, build), "Resolve update without scratch");
            if (!build.DestAccelerationStructureData || bool(build.ScratchAccelerationStructureData) != bool(given))
                throw std::runtime_error("A zero-size update scratch was not published as given");
        }
        must_reject(tlas, tlas, unmapped, none, true, Cause::not_placed, "a stray scratch for a zero-size update scratch");

        // Provenance: an update needs a source that the resolver saw built with ALLOW_UPDATE.
        const auto other = place(7, dest_offset + top_size.ResultDataMaxSizeInBytes,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE) + dest_offset;
        must_reject(tlas, other, scratch, sizes, true, Cause::source_not_built, "source never built");
        const auto build_other = [&](bool allow_update) {
            auto build = request(0, false);
            build.Inputs.Flags = allow_update ? (build.Inputs.Flags | allow_flag) : (build.Inputs.Flags & ~allow_flag);
            check(x2w_startup::resolve_placed_build(guest_.Get(), other, scratch, top_size, build), "Resolve a build of the other structure");
        };
        build_other(false);
        must_reject(tlas, other, scratch, sizes, true, Cause::source_not_updatable, "source built without ALLOW_UPDATE");
        // A refused update records nothing: the in-place TLAS update still works, and the other source is still refused.
        auto again = request(tlas, true);
        check(x2w_startup::resolve_placed_build(guest_.Get(), tlas, scratch, sizes, again), "Resolve in-place update after refusals");
        must_reject(tlas, other, scratch, sizes, true, Cause::source_not_updatable, "source built without ALLOW_UPDATE, again");

        // Out of place: another placed acceleration structure built with ALLOW_UPDATE, its interior offset kept.
        build_other(true);
        auto moved = request(other, true);
        check(x2w_startup::resolve_placed_build(guest_.Get(), tlas, scratch, sizes, moved), "Resolve out-of-place update");
        auto base = request(0, false);
        check(x2w_startup::resolve_placed_build(guest_.Get(), other - dest_offset, scratch - 768, sizes, base),
            "Resolve the source placement base");
        if (moved.SourceAccelerationStructureData != base.DestAccelerationStructureData + dest_offset ||
            moved.SourceAccelerationStructureData == moved.DestAccelerationStructureData)
            throw std::runtime_error("An out-of-place update lost the source address or aliased the destination");
        // The update's destination (the TLAS) is a recorded build again, with the flags of the update.
        auto after = request(tlas, true);
        check(x2w_startup::resolve_placed_build(guest_.Get(), tlas, scratch, sizes, after), "Resolve in-place update of the updated structure");
    }

    // Re-resolve one placed structure (0 = BLAS, 1 = TLAS) as an in-place PERFORM_UPDATE of itself. The source is the
    // title address the structure was built at, so the update runs through the same slot-133 placement lookup as the
    // build. The caller keeps ALLOW_UPDATE from the initial build; the resolver cannot see it.
    void resolve_update(unsigned which, D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& build) {
        const auto& placed = titles_.at(which);
        build.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        build.SourceAccelerationStructureData = placed.dest;
        check(x2w_startup::resolve_placed_build(guest_.Get(), placed.dest, placed.scratch, placed.sizes, build),
            "Resolve placed in-place update");
    }

    // Publishes the placed TLAS through guest device slot 70 into a CPU-only guest view heap, then three refused
    // calls at the same destination (wrong buffer state, unaligned, outside every placed buffer), and returns a
    // shader-visible desktop heap holding a copy of that one descriptor. Call after prepare().
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> publish_as_view(ID3D12Device* device) {
        struct Extended {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            UINT64 address, bytes;
        } extended{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0}, 0, 0};
        using Create = HRESULT (WINAPI*)(void*, const void*, const GUID*, void**);
        auto create = reinterpret_cast<Create>((*reinterpret_cast<void***>(guest_.Get()))[14]);
        check(create(guest_.Get(), &extended, &__uuidof(ID3D12DescriptorHeap),
            reinterpret_cast<void**>(cpu_heap_.ReleaseAndGetAddressOf())), "Create the guest view heap");
        using CpuStart = UINT64 (WINAPI*)(void*);
        const UINT64 destination = reinterpret_cast<CpuStart>((*reinterpret_cast<void***>(cpu_heap_.Get()))[9])(cpu_heap_.Get());
        D3D12_RESOURCE_DESC resource{};
        resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resource.Width = 256;
        resource.Height = resource.DepthOrArraySize = resource.MipLevels = resource.SampleDesc.Count = 1;
        resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        struct ViewAt {
            D3D12_SHADER_RESOURCE_VIEW_DESC view;
            UINT64 address, tail[2];
        } at{};
        at.view.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        at.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        using Publish = void (WINAPI*)(void*, const D3D12_RESOURCE_DESC*, const void*, UINT64);
        auto slot70 = reinterpret_cast<Publish>((*reinterpret_cast<void***>(guest_.Get()))[70]);
        const auto publish = [&](UINT64 address) {
            at.address = address;
            slot70(guest_.Get(), &resource, &at, destination);
        };
        publish(tlas_title_);
        publish(scratch_title_);   // as_wrong_state: a placed buffer, but not in the acceleration-structure state
        publish(tlas_title_ + 16); // as_unaligned
        publish(0x1000);           // as_not_placed
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> visible;
        D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&visible)), "Create the shader-visible view heap");
        // The published descriptor lives in a title-side handle range; the guest device copy (slot 24) translates it.
        using Copy = void (WINAPI*)(void*, UINT, UINT64, UINT64, UINT);
        reinterpret_cast<Copy>((*reinterpret_cast<void***>(guest_.Get()))[24])(guest_.Get(), 1,
            visible->GetCPUDescriptorHandleForHeapStart().ptr, destination, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return visible;
    }

    // The decoy BLAS for the wrong-routing control, placed and resolved through the same slot-133 path.
    void prepare_decoy(const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& bottom_size,
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& decoy) {
        constexpr UINT64 dest_offset = 512, scratch_offset = 768;
        const auto dest = place(5, dest_offset + bottom_size.ResultDataMaxSizeInBytes,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE) + dest_offset;
        const auto scratch = place(6, scratch_offset + bottom_size.ScratchDataSizeInBytes,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS) + scratch_offset;
        check(x2w_startup::resolve_placed_build(guest_.Get(), dest, scratch, bottom_size, decoy), "Resolve decoy BLAS");
    }
};

struct ResolvedBuild {
    UINT64 dest, scratch;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO sizes;
};
using ResolvedBatch = std::array<ResolvedBuild, 2>;
struct BuildCounts { UINT records{}, submissions{}; };
std::vector<UINT32> trace_placed_triangle(ID3D12Device* device, const ResolvedBatch* resolved,
                                          PlacedBuildScene* placement = nullptr, BuildCounts* counts = nullptr);
}
