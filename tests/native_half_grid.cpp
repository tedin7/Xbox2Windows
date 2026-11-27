#include "desktop_gpu.h"
#include "native_compute_pipeline.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace {
constexpr UINT width = 40, height = 54;
struct Surface {
    ComPtr<ID3D12Resource> texture, upload, readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 size{};
    bool half{}, writable{};
};
Surface make_surface(ID3D12Device *device, ID3D12GraphicsCommandList *list, bool writable, bool half, bool bit_input, bool prefix) {
    Surface s;
    s.writable = writable;
    s.half = half;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Format = half ? DXGI_FORMAT_R16_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.Flags = writable ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    s.texture = resource(device, desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    device->GetCopyableFootprints(&desc, 0, 1, 0, &s.footprint, nullptr, nullptr, &s.size);
    s.upload = make_buffer(device, s.size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void *mapped;
    check(s.upload->Map(0, nullptr, &mapped), "upload");
    for (UINT y = 0; y < height; ++y) {
        auto row = static_cast<char*>(mapped) + y * s.footprint.Footprint.RowPitch;
        for (UINT x = 0; x < width; ++x) {
            if (half) {
                reinterpret_cast<uint16_t*>(row)[x] = 0x5400;  // +64 sentinel
            } else {
                auto pixel = reinterpret_cast<float*>(row) + 4*x;
                for (UINT k = 0; k < 4; ++k)
                    pixel[k] = writable ? 64.f : float(1+x+16*y+k);
                if (!writable && prefix)
                    pixel[0] = x == 1+y ? 2.f : x == 9-y ? -2.f : x == 14 ? float(y+1) : 0.f;
                if (!writable && bit_input && ((x+y)&1)) {
                    UINT bits;
                    std::memcpy(&bits, pixel, 4);
                    bits |= 2;
                    std::memcpy(pixel, &bits, 4);
                }
            }
        }
    }
    s.upload->Unmap(0, nullptr);
    D3D12_TEXTURE_COPY_LOCATION src{s.upload.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    D3D12_TEXTURE_COPY_LOCATION dst{s.texture.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    src.PlacedFootprint = s.footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    transition(list, s.texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, writable ?
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (writable)
        s.readback = make_buffer(device, s.size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    return s;
}
}

int check_half_grid(ID3D12Device *device, x2w::ShaderDxilCompiler &compiler, const wchar_t *marker,
                    const wchar_t *cache, const std::wstring &mode) {
    const bool shift = mode == L"half-shift", bits = mode == L"half-bit", offset = mode == L"half-offset";
    const bool fixture = mode == L"half-value" || mode == L"half-mutant";
    if (!shift && !bits && !offset && !fixture && mode != L"half-clear" && mode != L"half-dual")
        return 2;
    auto request = x2w::read_compute_request(marker);
    UINT shape;
    std::memcpy(&shape, request.sa.data()+12, 4);
    const UINT gx = shape & 65535, gy = shape >> 16;
    auto p = x2w::create_native_compute_pipeline(device, compiler, marker, fixture ? 0 : 36, cache);
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, UINT(p.bindings.size()),
                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)), "half heap");
    UINT step = device->GetDescriptorHandleIncrementSize(hd.Type);
    auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    ComputeCommands cmd(device,p.state.Get(),D3D12_COMMAND_LIST_TYPE_DIRECT);
    // Multiple desktop views of one guest slot must see the same underlying storage.
    using Slot = std::pair<UINT,UINT>;
    std::map<Slot, Surface> surfaces;
    ComPtr<ID3D12Resource> params;
    for (const auto &b : p.bindings) {
        const bool write = x2w::compute_uav(b.kind);
        Slot slot{b.root_parameter,b.byte_offset};
        if (b.kind == x2w::ComputeViewKind::buffer_srv && offset && slot.second == 32) {
            params = make_buffer(device, 24, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            const UINT words[]{91,98,87,77,3,5};
            void *data;
            check(params->Map(0,nullptr,&data),"offset buffer");
            std::memcpy(data, words,sizeof(words));
            params->Unmap(0,nullptr);
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Format = DXGI_FORMAT_R32_TYPELESS;
            srv.Buffer.NumElements = 6;
            srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            device->CreateShaderResourceView(params.Get(),&srv,cpu);
        } else {
            bool half = b.kind == x2w::ComputeViewKind::half_scalar_uav || (shift && slot.second == 64);
            if (!surfaces.count(slot))
                surfaces.emplace(slot,make_surface(device,cmd.list.Get(),write,half,bits,shift));
            auto &s = surfaces.at(slot);
            if (write)
                device->CreateUnorderedAccessView(s.texture.Get(),nullptr,nullptr,cpu);
            else
                device->CreateShaderResourceView(s.texture.Get(),nullptr,cpu);
        }
        cpu.ptr += step;
    }
    if (surfaces.size() != (shift || bits ? 3u : mode == L"half-dual" ? 2u : 1u))
        return 2;
    const UINT groups_x = shift ? 1 : 2, groups_y = 3;
    std::vector<UINT> constants(p.constant_words,0);
    p.bind(cmd.list.Get(),heap.Get(),heap->GetGPUDescriptorHandleForHeapStart(),step,constants.data());
    cmd.list->Dispatch(groups_x,groups_y,1);
    for (auto &[slot,s] : surfaces) {
        if (!s.writable) continue;
        transition(cmd.list.Get(),s.texture.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{s.texture.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION dst{s.readback.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = s.footprint;
        cmd.list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    }
    cmd.submit_and_wait(device);
    check(device->GetDeviceRemovedReason(),"half completion");
    UINT differences = 0, checked = 0;
    for (auto &[slot,s] : surfaces) {
        if (!s.writable) continue;
        void *mapped;
        check(s.readback->Map(0,nullptr,&mapped),"half readback");
        for (UINT y = 0; y < height; ++y) {
            auto row = static_cast<const char*>(mapped) + y*s.footprint.Footprint.RowPitch;
            for (UINT x = 0; x < width; ++x) {
                UINT ox = offset ? 3 : 0, oy = offset ? 5 : 0;
                bool covered = x >= ox && y >= oy && x-ox < gx*groups_x && y-oy < gy*groups_y;
                if (s.half) {
                    UINT expected = covered ? (fixture ? 0xc100u : 0u) : 0x5400u;
                    if (shift) {
                        expected = 0x5400;
                        if (x < 16 && y < groups_y) {
                            // Exclusive u32 prefix of the bit patterns for +2 and -2.
                            // Their modulo-2^32 sum is zero. The last source value is y+1.
                            constexpr uint16_t tail[]{0x3c00, 0x4000, 0x4200};
                            expected = x == 15 ? tail[y] : x > 1+y && x <= 9-y ? 0x4000 : 0;
                        }
                    }
                    differences += reinterpret_cast<const uint16_t*>(row)[x] != expected;
                    ++checked;
                } else {
                    float expected = 64.f;
                    if (shift && y == 0 && x < groups_y) expected = float(x+1);
                    if (bits && covered) {
                        UINT value = ((x+y)&1) * 2;
                        std::memcpy(&expected,&value,4);
                    }
                    for (UINT k = 0; k < 4; ++k) {
                        differences += std::memcmp(row+(4*x+k)*4,&expected,4) != 0;
                        ++checked;
                    }
                }
            }
        }
        s.readback->Unmap(0,nullptr);
    }
    std::printf("components=%u differing_components=%u cache_hit=%d frontend_ms=%.4f backend_ms=%.4f pso_ms=%.4f\n",
                checked,differences,p.cache_hit,p.frontend_ms,p.backend_ms,p.pipeline_ms);
    return differences ? 1 : 0;
}
