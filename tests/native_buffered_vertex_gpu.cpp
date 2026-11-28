#include "desktop_gpu.h"
#include "gfx1013_vertex_id.h"
#include "shader_dxil_compiler.h"
#include "xmem_test_device.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>

using Microsoft::WRL::ComPtr;
using Position = std::array<float, 4>;
using Bytes = std::vector<unsigned char>;
using Word = std::uint32_t;
namespace fs = std::filesystem;

namespace {
template <class T> Bytes bytes_of(const std::vector<T> &input) {
    auto begin = reinterpret_cast<const unsigned char *>(input.data());
    return {begin, begin + input.size() * sizeof(T)};
}
struct Case {
    std::string program, bindings;
    std::array<Word, 4> entry{};
    std::vector<Bytes> buffers;
    Bytes indices;
    std::vector<Position> expected;
};
Case fixture(bool overwrite_descriptor = false) {
    Case data;
    // Different SGPRs, VB descriptor position, root-table number and byte offset
    // from the buffered draw. Use a nonzero base vertex and a vertex byte offset.
    std::vector<Word> words{0xbe980308,
                            0xbe990380, // s24 = table0, s25 = 0
                            0xbe9a0400,
                            0x939bff1b,
                            0x00080000, // vertex table s[26:27], mask high
                            0xf4080000u | (32u << 6) | 12u,
                            0xfa000040,
                            0xf4080000u | (40u << 6) | 13u,
                            0xfa000000,
                            0x4a000002, // v0 += s2
                            0xe03c2004,
                            0x800a0600, // x/y/z at stride*id + 4
                            0xf4280000u | (32u << 6) | 16u,
                            0xfa000010, // replace descriptor with 4 values
                            0xd5030009,
                            32u | (262u << 9), // x + constant.x
                            0xd54b000a,
                            242u | (263u << 9) | (33u << 18) | (1u << 31), // y - constant.y
                            0x7e1602f2,                                    // w = 1
                            0xf80008cf,
                            0x0b080a09,
                            0xf8000201,
                            0x80808009,
                            0xbf810000};
    if (overwrite_descriptor) {
        // Overwrite word 2 of the vertex descriptor before the first vertex load.
        const auto load = std::find(words.begin(), words.end(), Word{0xe03c2004});
        words.insert(load, 0xbeaa0380);
    }
    data.program.assign(reinterpret_cast<const char *>(words.data()), words.size() * sizeof(Word));
    data.entry = {0, 1, 0, 16};
    data.buffers = {bytes_of(std::vector<float>{99, 99, 99, 99, 7, 1, 2, 3, 7, 4, 5, 6, 7, 7, 8, 9}),
                    bytes_of(std::vector<float>{0, 0, 0, 0, .5f, 1.5f, 2, 1})};
    data.indices = bytes_of(std::vector<std::uint16_t>{1, 0, 2, 2, 0, 1});
    data.expected = {{4.5f, 3.5f, 6, 1}, {1.5f, .5f, 3, 1}, {7.5f, 6.5f, 9, 1},
                     {7.5f, 6.5f, 9, 1}, {1.5f, .5f, 3, 1}, {4.5f, 3.5f, 6, 1}};
    data.bindings = "srv 0 buf vb 0\nsrv 1 buf table 0 64\n";
    return data;
}
Bytes read_file(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("missing replay input: " + path.filename().string());
    return {std::istreambuf_iterator<char>(file), {}};
}
Case load(const fs::path &dir) {
    Case data;
    auto program = read_file(dir / "program.bin"), bindings = read_file(dir / "bindings.txt");
    data.program.assign(program.begin(), program.end());
    data.bindings.assign(bindings.begin(), bindings.end());
    data.indices = read_file(dir / "indices.bin");
    auto entry = read_file(dir / "entry.bin"), expected = read_file(dir / "positions.bin");
    if (entry.size() != sizeof(data.entry) || expected.empty() || expected.size() % sizeof(Position) ||
        expected.size() / sizeof(Position) * 2 != data.indices.size())
        throw std::runtime_error("invalid replay dimensions");
    std::memcpy(data.entry.data(), entry.data(), entry.size());
    data.expected.resize(expected.size() / sizeof(Position));
    std::memcpy(data.expected.data(), expected.data(), expected.size());
    for (unsigned i = 0; i < 32; ++i) {
        auto path = dir / ("t" + std::to_string(i) + ".bin");
        if (!fs::exists(path)) break;
        data.buffers.push_back(read_file(path));
    }
    return data;
}
ComPtr<ID3D12Resource> upload(ID3D12Device *device, const Bytes &bytes) {
    auto obj = make_buffer(device, bytes.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void *mapped = nullptr;
    check(obj->Map(0, nullptr, &mapped), "map upload");
    std::memcpy(mapped, bytes.data(), bytes.size());
    obj->Unmap(0, nullptr);
    return obj;
}
void verify(ID3D12Device *dev, const Bytes &vs, const Case &data) {
    if (data.buffers.empty() || data.buffers.size() > 32) throw std::runtime_error("invalid descriptor count");
    const UINT count = static_cast<UINT>(data.buffers.size());
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, count, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 4;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &range};
    D3D12_ROOT_SIGNATURE_DESC root_desc{2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT};
    ComPtr<ID3DBlob> serialized;
    check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr),
          "serialize SO root");
    ComPtr<ID3D12RootSignature> root;
    check(dev->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                   IID_PPV_ARGS(&root)),
          "SO root");
    D3D12_SO_DECLARATION_ENTRY output{0, "SV_Position", 0, 0, 4, 0};
    const UINT stride = sizeof(Position);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = root.Get();
    pso.VS = {vs.data(), vs.size()};
    pso.StreamOutput = {&output, 1, &stride, 1, D3D12_SO_NO_RASTERIZED_STREAM};
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.SampleMask = UINT_MAX;
    pso.SampleDesc.Count = 1;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    ComPtr<ID3D12PipelineState> pipeline;
    const auto pso_start = std::chrono::steady_clock::now();
    check(dev->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline)), "SO graphics pipeline");
    const auto pso_time = std::chrono::steady_clock::now() - pso_start;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count,
                                         D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> heap;
    check(dev->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "SRV heap");
    auto dest = heap->GetCPUDescriptorHandleForHeapStart();
    std::vector<ComPtr<ID3D12Resource>> buffers;
    for (const auto &bytes : data.buffers) {
        if (bytes.empty() || bytes.size() % 4) throw std::runtime_error("invalid raw SRV length");
        buffers.push_back(upload(dev, bytes));
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Format = DXGI_FORMAT_R32_TYPELESS;
        view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        view.Buffer.NumElements = static_cast<UINT>(bytes.size() / 4);
        view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        dev->CreateShaderResourceView(buffers.back().Get(), &view, dest);
        dest.ptr += dev->GetDescriptorHandleIncrementSize(heap_desc.Type);
    }
    auto ib = upload(dev, data.indices), zero = upload(dev, Bytes(16));
    D3D12_INDEX_BUFFER_VIEW index{ib->GetGPUVirtualAddress(), static_cast<UINT>(data.indices.size()),
                                  DXGI_FORMAT_R16_UINT};
    const UINT64 payload = data.expected.size() * sizeof(Position);
    auto out = make_buffer(dev, 16 + payload, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    auto readback = make_buffer(dev, 16 + payload, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands commands(dev, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto *list = commands.list.Get();
    D3D12_QUERY_HEAP_DESC queries{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
    ComPtr<ID3D12QueryHeap> timestamps;
    check(dev->CreateQueryHeap(&queries, IID_PPV_ARGS(&timestamps)), "draw timestamps");
    auto timing = make_buffer(dev, 2 * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    UINT64 frequency = 0;
    check(commands.queue->GetTimestampFrequency(&frequency), "timestamp frequency");
    list->CopyBufferRegion(out.Get(), 0, zero.Get(), 0, 16);
    transition(list, out.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_STREAM_OUT);
    D3D12_STREAM_OUTPUT_BUFFER_VIEW so{out->GetGPUVirtualAddress() + 16, payload, out->GetGPUVirtualAddress()};
    list->SOSetTargets(0, 1, &so);
    list->SetGraphicsRootSignature(root.Get());
    ID3D12DescriptorHeap *heaps[]{heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetGraphicsRoot32BitConstants(0, 4, data.entry.data(), 0);
    list->SetGraphicsRootDescriptorTable(1, heap->GetGPUDescriptorHandleForHeapStart());
    list->IASetIndexBuffer(&index);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->EndQuery(timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    list->DrawIndexedInstanced(static_cast<UINT>(data.indices.size() / 2), 1, 0, 0, 0);
    list->EndQuery(timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    list->ResolveQueryData(timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, timing.Get(), 0);
    transition(list, out.Get(), D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyResource(readback.Get(), out.Get());
    commands.submit_and_wait(dev);
    void *memory = nullptr;
    D3D12_RANGE read{0, static_cast<SIZE_T>(16 + payload)};
    check(readback->Map(0, &read, &memory), "map SO result");
    UINT64 written = 0;
    std::memcpy(&written, memory, sizeof(written));
    if (written != payload) throw std::runtime_error("wrong output vertex count");
    const auto *actual = reinterpret_cast<const Position *>(static_cast<const char *>(memory) + 16);
    double max_error = 0;
    unsigned identical = 0;
    for (size_t v = 0; v < data.expected.size(); ++v)
        for (unsigned c = 0; c < 4; ++c) {
            const float a = actual[v][c], e = data.expected[v][c];
            if (!std::isfinite(a) || !std::isfinite(e) || std::abs(a - e) > 1.e-5f * (std::max)(std::abs(e), 1.0f))
                throw std::runtime_error("position mismatch at vertex " + std::to_string(v) + " component " +
                                         std::to_string(c) + " actual=" + std::to_string(a) +
                                         " expected=" + std::to_string(e));
            identical += !std::memcmp(&a, &e, sizeof(a));
            max_error = (std::max)(max_error, std::abs(double(a) - e));
        }
    D3D12_RANGE none{};
    readback->Unmap(0, &none);
    D3D12_RANGE time_range{0, 2 * sizeof(UINT64)};
    check(timing->Map(0, &time_range, &memory), "read timestamps");
    const auto *ticks = static_cast<const UINT64 *>(memory);
    if (!frequency || ticks[1] < ticks[0]) throw std::runtime_error("invalid GPU timestamps");
    std::printf("pso_ms=%.6f draw_gpu_us=%.6f\n", std::chrono::duration<double, std::milli>(pso_time).count(),
                1e6 * double(ticks[1] - ticks[0]) / frequency);
    timing->Unmap(0, &none);
    std::printf("%zu indexed vertices, %u/%zu identical components, max absolute error %.9g\n",
                data.expected.size(), identical, data.expected.size() * 4, max_error);
}
} // namespace
int wmain(int argc, wchar_t **argv) {
    if (argc != 4 && argc != 5) return 2;
    try {
        bool unmatched = false;
        const auto dev = create_device(std::wstring_view(argv[1]) == L"warp", unmatched);
        if (!dev) return unmatched ? 1 : 77;
        D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_0};
        if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) ||
            model.HighestShaderModel < D3D_SHADER_MODEL_6_0)
            return 77;
        auto data = argc == 5 ? load(argv[4]) : fixture();
        if (argc == 4) {
            bool refused = false;
            try {
                x2w::translate_gfx1013_buffered_vertex(fixture(true).program);
            } catch (const std::runtime_error &error) {
                refused = std::strstr(error.what(), "overwritten buffer descriptor") != nullptr;
            }
            if (!refused) throw std::runtime_error("accepted overwritten vertex descriptor or wrong diagnostic");
        }
        const auto program = x2w::translate_gfx1013_buffered_vertex(data.program);
        if (program.bindings != data.bindings) throw std::runtime_error("vertex resource manifest mismatch");
        x2w::ShaderDxilCompiler compiler(argv[2]);
        verify(dev.Get(), compiler.compile(program.hlsl, L"vs_6_0", argv[3]).dxil, data);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
