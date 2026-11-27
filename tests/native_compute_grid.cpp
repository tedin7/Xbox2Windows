#include "desktop_gpu.h"
#include "native_compute_pipeline.h"
#include "startup_stubs/native_buffer_stride.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;
namespace {
constexpr UINT width = 64, height = 32, pixels = width * height;
constexpr UINT raw_words = pixels * 2 + 16;
UINT raw_word(UINT i) {
    // Every even word is another valid pair's byte address; odd words identify the pair.
    return i % 2 ? 0x1234u + 11 * (i / 2) : 8 * ((i / 2 + 1) % pixels);
}
using Pixel = std::array<float, 4>;
Pixel pattern(UINT x, UINT y) {
    return {float(1 + x + 128 * y), float(2000 + 2 * x + y), float(3 + x * y), float(10000 - x - 2 * y)};
}
std::array<UINT, 2> sdwa_inputs(UINT x) {
    return {0x93ac38e7u ^ (x * 0x0305070bu), 0x5ad2419bu ^ (x * 0x09010305u)};
}
Pixel sdwa_expected(UINT x, const std::wstring &mode) {
    auto [a, b] = sdwa_inputs(x);
    UINT first, second;
    if (mode == L"sdwa-class") {
        first = b;
        second = ((b >> 24) & 32u) ? a : b;
    } else if (mode == L"sdwa-alias") {
        first = (a >> 16) ^ a;
        second = ((b >> 16) & 255u) ^ ((a >> 8) & 255u);
    } else {
        // Sign-extended source byte XOR a 32-bit operand. Then place the low byte
        // of a separate XOR in byte 1, retaining or sign-filling unused bits.
        first = UINT(int32_t(int8_t(a >> 16))) ^ b;
        const UINT byte = ((a >> 16) ^ b) & 255u;
        second = mode == L"sdwa-preserve" ? ((b & 0xffff00ffu) | (byte << 8))
                                         : UINT(int32_t(int8_t(byte))) << 8;
    }
    return {float(first & 65535), float(first >> 16), float(second & 65535), float(second >> 16)};
}
} // namespace
int check_native_compute_grid(ID3D12Device *device, x2w::ShaderDxilCompiler &compiler, const wchar_t *marker,
                              const wchar_t *cache, const std::wstring &mode) {
    const bool sdwa = mode == L"sdwa-alias" || mode == L"sdwa-preserve" || mode == L"sdwa-sext" || mode == L"sdwa-class";
    const bool typed = mode == L"typed" || sdwa;
    const bool global_stride = mode == L"global-stride";
    const bool indexed = mode == L"indexed";
    const bool raw = global_stride || indexed || mode == L"raw-offset" || mode == L"raw-edge" || mode == L"raw-three" ||
                     mode == L"raw-four" || mode == L"raw-math";
    const bool row = mode == L"row";
    if (!typed && !raw && !row && mode != L"copy" && mode != L"red" && mode != L"xyxx" && mode != L"negative")
        return 2;
    auto req = x2w::read_compute_request(marker);
    UINT shape = 0;
    std::memcpy(&shape, req.sa.data() + 12, sizeof(shape));
    const UINT gx = shape & 65535, gy = shape >> 16;
    if (!gx || !gy || width % gx || height % gy)
        return 2;
    auto pipeline = x2w::create_native_compute_pipeline(device, compiler, marker, typed || raw ? 0 : 36, cache);
    if (pipeline.srv_count != (row || indexed || global_stride ? 2u : 1u) || pipeline.uav_count != 1)
        return 2;
    auto kind = global_stride ? x2w::ComputeViewKind::buffer_srv : raw ? x2w::ComputeViewKind::raw_vector_srv :
                typed ? x2w::ComputeViewKind::typed_buffer_srv : x2w::ComputeViewKind::texture_srv;
    if ((row ? pipeline.bindings[1] : pipeline.bindings.front()).kind != kind)
        return 2;
    std::vector<UINT> constants(pipeline.constant_words, 0);
    if (typed || raw) {
        if (constants.size() != 4 || pipeline.constants[0].root_parameter != 2 ||
            pipeline.bindings[0].byte_offset != 64 || pipeline.bindings[indexed || global_stride ? 2 : 1].byte_offset != 224)
            return 2;
        if (indexed && (pipeline.bindings[0].root_parameter != 0 || pipeline.bindings[0].array_count != 2 ||
                        pipeline.bindings[1].root_parameter != 0 || pipeline.bindings[1].byte_offset != 96 ||
                        pipeline.bindings[1].array_position != 1 || pipeline.bindings[2].root_parameter != 1))
            return 2;
        if (global_stride && (pipeline.bindings[1].kind != x2w::ComputeViewKind::heap_stride_srv ||
                              pipeline.bindings[1].root_parameter != x2w::compute_resource_heap))
            return 2;
        constants[0] = 3;
        constants[1] = 5;
    }
    constexpr unsigned copied_columns = 47, source_row = 13;
    if (row) {
        if (constants.size() != 2 || pipeline.bindings.front().kind != x2w::ComputeViewKind::buffer_srv ||
            pipeline.bindings[0].byte_offset != 0 || pipeline.bindings[1].byte_offset != 32 ||
            pipeline.bindings[2].byte_offset != 64)
            return 2;
        constants[0] = copied_columns;
        constants[1] = 1;
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, pipeline.srv_count + 1,
                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "grid heap");
    UINT step = device->GetDescriptorHandleIncrementSize(hd.Type);
    auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    ComputeCommands cmd(device, pipeline.state.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = width;
    td.Height = height;
    td.DepthOrArraySize = td.MipLevels = td.SampleDesc.Count = 1;
    td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 bytes;
    device->GetCopyableFootprints(&td, 0, 1, 0, &fp, nullptr, nullptr, &bytes);
    auto upload = make_buffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void *mapped;
    check(upload->Map(0, nullptr, &mapped), "grid upload");
    ComPtr<ID3D12Resource> input;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (raw) {
        constexpr UINT first = 8;
        for (UINT i = 0; i < first + raw_words + 1; ++i)
            static_cast<UINT*>(mapped)[i] = i < first ? 0x654321u : raw_word(i - first);
        if (global_stride) {
            static_cast<UINT*>(mapped)[first + 3] = 4;
            static_cast<UINT*>(mapped)[first + 11] = 9;
        }
        srv.Format = DXGI_FORMAT_R32_TYPELESS;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Buffer.FirstElement = first;
        srv.Buffer.NumElements = raw_words;
        srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        input = upload;
    } else if (typed) {
        for (UINT i = 0; i < width; ++i) {
            const auto values = sdwa ? sdwa_inputs(i) : std::array<UINT, 2>{33 * i + 7, 5 * i + 2};
            std::memcpy(static_cast<UINT *>(mapped) + 2 * i, values.data(), sizeof(values));
        }
        srv.Format = DXGI_FORMAT_R32G32_UINT;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Buffer.NumElements = width;
        input = upload;
    } else {
        for (UINT y = 0; y < height; ++y)
            for (UINT x = 0; x < width; ++x)
                std::memcpy(static_cast<char *>(mapped) + y * fp.Footprint.RowPitch + x * sizeof(Pixel),
                            pattern(x, y).data(), sizeof(Pixel));
        srv.Format = td.Format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MipLevels = 1;
        input = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT},
            dst{input.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src.PlacedFootprint = fp;
        cmd.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(cmd.list.Get(), input.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    upload->Unmap(0, nullptr);
    ComPtr<ID3D12Resource> parameters;
    if (row) {
        constexpr unsigned words = 130;
        parameters = make_buffer(device, words * 4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        check(parameters->Map(0, nullptr, &mapped), "row parameters");
        std::memset(mapped, 0, words * 4);
        static_cast<UINT *>(mapped)[129] = source_row << 8;
        parameters->Unmap(0, nullptr);
        D3D12_SHADER_RESOURCE_VIEW_DESC raw{};
        raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        raw.Format = DXGI_FORMAT_R32_TYPELESS;
        raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        raw.Buffer.NumElements = words;
        raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        device->CreateShaderResourceView(parameters.Get(), &raw, cpu);
        cpu.ptr += step;
    }
    device->CreateShaderResourceView(input.Get(), &srv, cpu);
    ComPtr<ID3D12Resource> second;
    if (indexed || global_stride) {
        second = make_buffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        check(second->Map(0, nullptr, &mapped), "second indexed SRV");
        for (UINT i = 0; i < raw_words; ++i)
            static_cast<UINT*>(mapped)[srv.Buffer.FirstElement + i] = 100000 + raw_word(i);
        if (global_stride) {
            auto *strides = static_cast<UINT*>(mapped) + srv.Buffer.FirstElement;
            // Supply the metadata for a typed float3 SRV and an eleven-word structured UAV
            // using the same decoder used for materialized guest slots at dispatch.
            D3D12_SHADER_RESOURCE_VIEW_DESC typed_view{};
            typed_view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            typed_view.Format = DXGI_FORMAT_R32G32B32_FLOAT;
            D3D12_UNORDERED_ACCESS_VIEW_DESC structured{};
            structured.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            structured.Buffer.StructureByteStride = 44;
            if (!x2w_startup::native_view_stride(18, &typed_view, sizeof(typed_view), strides[4]) ||
                !x2w_startup::native_view_stride(19, &structured, sizeof(structured), strides[9]))
                return 2;
        }
        second->Unmap(0, nullptr);
        cpu.ptr += step;
        device->CreateShaderResourceView(second.Get(), &srv, cpu);
    }
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    auto output = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (row) {
        transition(cmd.list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        D3D12_TEXTURE_COPY_LOCATION dst{output.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src.PlacedFootprint = fp;
        cmd.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(cmd.list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    cpu.ptr += step;
    device->CreateUnorderedAccessView(output.Get(), nullptr, nullptr, cpu);
    auto rb = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    auto ticks = make_buffer(device, 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2};
    ComPtr<ID3D12QueryHeap> query;
    check(device->CreateQueryHeap(&qd, IID_PPV_ARGS(&query)), "grid timestamps");
    pipeline.bind(cmd.list.Get(), heap.Get(), heap->GetGPUDescriptorHandleForHeapStart(), step, constants.data());
    cmd.list->EndQuery(query.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    cmd.list->Dispatch(width / gx, row ? 1u : height / gy, 1);
    cmd.list->EndQuery(query.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    cmd.list->ResolveQueryData(query.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, ticks.Get(), 0);
    transition(cmd.list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{output.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX},
        dst{rb.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint = fp;
    cmd.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    check(cmd.list->Close(), "close");
    ID3D12CommandList *lists[]{cmd.list.Get()};
    auto start = Clock::now();
    cmd.queue->ExecuteCommandLists(1, lists);
    double submit_us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "fence");
    check(cmd.queue->Signal(fence.Get(), 1), "signal");
    check(fence->SetEventOnCompletion(1, nullptr), "await grid");
    check(device->GetDeviceRemovedReason(), "grid completion");
    check(ticks->Map(0, nullptr, &mapped), "read times");
    UINT64 t[2], hz;
    std::memcpy(t, mapped, 16);
    ticks->Unmap(0, nullptr);
    check(cmd.queue->GetTimestampFrequency(&hz), "frequency");
    check(rb->Map(0, nullptr, &mapped), "read pixels");
    unsigned differences = 0;
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x) {
            Pixel expected = pattern(x, y);
            if (raw) {
                const UINT lane = y * width + x;
                const UINT pair = mode == L"raw-edge" ? raw_words - 1 : 2 * lane;
                expected = {float(raw_word(2 * lane + 13)), float(raw_word(pair)),
                            pair + 1 < raw_words ? float(raw_word(pair + 1)) : 0.f, float(lane)};
                if (mode == L"raw-three" || mode == L"raw-four")
                    expected = {float(raw_word(pair)), float(raw_word(pair + 1)), float(raw_word(pair + 2)),
                                mode == L"raw-four" ? float(raw_word(pair + 3)) : float(raw_word(pair + 13))};
                if (indexed && x >= gx)
                    for (unsigned component = 0; component < 3; ++component)
                        expected[component] += 100000;
                if (mode == L"raw-math") {
                    constexpr float cosine[]{1.f, 0.f, -1.f, 0.f};
                    expected[0] = cosine[lane % 4];
                    const UINT bits = raw_word(pair), exponent = bits >> 10, fraction = bits & 1023;
                    expected[3] = exponent ? std::ldexp(float(1024 + fraction), int(exponent) - 25)
                                           : std::ldexp(float(fraction), -24);
                }
            }
            if (global_stride) expected.fill(x < gx ? 12.f : 44.f);
            if (typed)
                expected = {float(33 * x + 7), float(16*(y*3+5)+1+5*x+2), float(x+(x<32?3:9)), float(y/2+((y&3)==3))};
            if (sdwa)
                expected = sdwa_expected(x, mode);
            if (row && y == 0 && x < copied_columns)
                expected = pattern(x, source_row);
            if (mode == L"red")
                expected.fill(expected[0]);
            if (mode == L"xyxx")
                expected[2] = expected[3] = expected[0];
            if (mode == L"negative")
                expected[0] += 1;
            auto *actual =
                reinterpret_cast<const float *>(static_cast<const char *>(mapped) + y * fp.Footprint.RowPitch) + 4 * x;
            for (UINT i = 0; i < 4; ++i)
                differences += actual[i] != expected[i];
        }
    rb->Unmap(0, nullptr);
    std::printf("pixels=%u differing_components=%u cache_hit=%d frontend_ms=%.4f backend_ms=%.4f pso_ms=%.4f "
                "cpu_submit_us=%.3f gpu_dispatch_us=%.3f\n",
                pixels, differences, pipeline.cache_hit, pipeline.frontend_ms, pipeline.backend_ms,
                pipeline.pipeline_ms, submit_us, double(t[1] - t[0]) * 1e6 / hz);
    return differences ? 1 : 0;
}
