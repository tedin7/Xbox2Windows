#include "startup_slot67_indirect.h"
#include "startup_indirect_writer.h"
#include "startup_stream_setup.h"
#include <d3dcompiler.h>
#include <array>
#include <cstdio>
#include <cstring>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
using Microsoft::WRL::ComPtr;

namespace {
using Record = std::array<UINT, 5>;
constexpr Record initial{0xa5a5a5a5, 0, 0, 0, 0x5a5a5a5a};
using FullRecord = std::array<UINT, 8>;
constexpr FullRecord full_record{2, 1, 1, 0x137u, 5, 3, 1, 0x245u};
constexpr UINT full_offset = 4 * sizeof(UINT);
constexpr UINT offset = sizeof(UINT);  // guards on either side of the three-DWORD record
constexpr char producer[] = R"(
RWByteAddressBuffer args : register(u0);
cbuffer Size : register(b0) { uint groups; };
[numthreads(1, 1, 1)]
void main() { args.Store3(4, uint3(groups, 1, 1)); }
)";
StreamWord word(const void* value) { return reinterpret_cast<StreamWord>(value); }
}

void Slot67Indirect::prepare(IDXGIAdapter* adapter, void* guest_device, void* guest_list, void* backing, UINT groups, bool full)
{
    auto* guest = static_cast<ID3D12Device*>(guest_device);
    ComPtr<ID3D12Device> device;
    check(D3D12XboxCreateDevice(adapter, nullptr, IID_PPV_ARGS(&device)), "open producer's desktop device");
    auto list = indirect_native_object<ID3D12GraphicsCommandList>(device.Get(), guest, static_cast<IUnknown*>(guest_list));
    record_offset = full ? full_offset : offset;
    const UINT bytes = full ? sizeof(full_record) : sizeof(initial);
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    check(static_cast<HRESULT>(invoke(guest, 133, word(backing), word(&desc), D3D12_RESOURCE_STATE_COMMON, 0,
          word(&__uuidof(ID3D12Resource)), word(placed.GetAddressOf()))), "place indirect arguments");
    auto args = indirect_native_object<ID3D12Resource>(device.Get(), guest, placed.Get());
    upload = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    readback = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const D3D12_RANGE empty{0, 0};
    void* mapped = nullptr;
    check(upload->Map(0, &empty, &mapped), "map zero dispatch record");
    std::memcpy(mapped, full ? full_record.data() : initial.data(), bytes);
    upload->Unmap(0, nullptr);
    transition(list.Get(), args.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(args.Get(), 0, upload.Get(), 0, bytes);
    auto state = D3D12_RESOURCE_STATE_COPY_DEST;
    if (groups && !full) {
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants = {0, 0, 1};
        const D3D12_ROOT_SIGNATURE_DESC root_desc{2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob, code;
        check(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "serialize producer root");
        check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "create producer root");
        check(D3DCompile(producer, sizeof(producer) - 1, nullptr, nullptr, nullptr, "main", "cs_5_0", 0, 0,
                         &code, nullptr), "compile indirect producer");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get();
        pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        check(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&writer)), "create indirect producer");
        transition(list.Get(), args.Get(), state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        list->SetComputeRootSignature(root.Get());
        list->SetComputeRoot32BitConstant(1, groups, 0);
        record_indirect_writer(list.Get(), writer.Get(), args.Get());
    }
    transition(list.Get(), args.Get(), state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, args.Get(), 0, bytes);
    transition(list.Get(), args.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    D3D12_INDIRECT_ARGUMENT_DESC command{D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH};
    const D3D12_COMMAND_SIGNATURE_DESC sig{sizeof(D3D12_DISPATCH_ARGUMENTS), 1, &command, 0};
    check(guest->CreateCommandSignature(&sig, nullptr, IID_PPV_ARGS(&signature)), "create guest dispatch signature");
}

void Slot67Indirect::execute(void* list, UINT commands, bool count_buffer, bool outside, bool zero_offset) const
{
    invoke(list, 61, word(signature.Get()), commands, word(placed.Get()), outside ? sizeof(initial) - sizeof(UINT) : zero_offset ? 0 : record_offset,
           count_buffer ? word(placed.Get()) : 0, count_buffer ? offset : 0);
}

bool Slot67Indirect::check_record(UINT groups, bool full) const
{
    const D3D12_RANGE range{0, full ? sizeof(full_record) : sizeof(initial)}, empty{0, 0};
    void* mapped = nullptr;
    check(readback->Map(0, &range, &mapped), "map GPU argument readback after consumer fence");
    if (full) {
        FullRecord actual{};
        std::memcpy(actual.data(), mapped, sizeof(actual));
        readback->Unmap(0, &empty);
        return actual == full_record;
    }
    Record actual{};
    std::memcpy(actual.data(), mapped, sizeof(actual));
    readback->Unmap(0, &empty);
    const Record expected{initial.front(), groups, groups ? 1u : 0u, groups ? 1u : 0u, initial.back()};
    std::printf("GPU argument record: %08x [%u,%u,%u] %08x\n", actual[0], actual[1], actual[2], actual[3], actual[4]);
    std::fflush(stdout);
    return actual == expected;
}
