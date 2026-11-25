#pragma once

#include "desktop_gpu.h"
#include "startup_stubs/placed_build.h"
#include <array>
#include <cstdio>

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
    std::array<Buffer, 7> buffers_;
    UINT64 tlas_title_{}, scratch_title_{};

public:
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
        const auto blas = place(0, dest_offset + bottom_size.ResultDataMaxSizeInBytes, dest_state) + dest_offset;
        const auto tlas = place(1, dest_offset + top_size.ResultDataMaxSizeInBytes, dest_state) + dest_offset;
        const auto bs = place(2, scratch_offset + bottom_size.ScratchDataSizeInBytes, scratch_state) + scratch_offset;
        const auto ts = place(3, scratch_offset + top_size.ScratchDataSizeInBytes, scratch_state) + scratch_offset;
        const auto short_scratch = place(4, scratch_offset + top_size.ScratchDataSizeInBytes - 1, scratch_state) + scratch_offset;
        constexpr UINT64 unmapped = 0x1000; // outside the user allocation range
        const auto must_reject = [&](UINT64 dest, UINT64 scratch) {
            auto candidate = top;
            if (x2w_startup::resolve_placed_build(guest_.Get(), dest, scratch, top_size, candidate) != E_INVALIDARG ||
                candidate.DestAccelerationStructureData || candidate.ScratchAccelerationStructureData)
                throw std::runtime_error("Invalid DXR build was accepted or partially published");
        };
        must_reject(unmapped, ts);
        must_reject(tlas, short_scratch);
        must_reject(ts, tlas);
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
        std::printf("Placed DXR bytes dest/scratch: BLAS=%llu/%llu TLAS=%llu/%llu; short scratch, missing dest and swapped states rejected before recording\n",
            bottom_size.ResultDataMaxSizeInBytes, bottom_size.ScratchDataSizeInBytes,
            top_size.ResultDataMaxSizeInBytes, top_size.ScratchDataSizeInBytes);
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
