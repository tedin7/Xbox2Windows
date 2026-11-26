#include "startup_slot67_indirect.h"
#include "startup_indirect_writer.h"
#include "startup_indirect_signature.h"
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
constexpr UINT offset = sizeof(UINT);  // the record follows a leading guard
std::array<UINT, 3> record_constants(UINT count)
{
    return count == 3 ? std::array<UINT, 3>{0x1234, 0xd7, 0x5678} : std::array<UINT, 3>{0x1234, 0x5678, 0};
}
constexpr char producer[] = R"(
RWByteAddressBuffer args : register(u0);
cbuffer Size : register(b0) { uint groups; };
[numthreads(1, 1, 1)]
void main() { args.Store3(4, uint3(groups, 1, 1)); }
)";
StreamWord word(const void* value) { return reinterpret_cast<StreamWord>(value); }
}

void Slot67Indirect::prepare(IDXGIAdapter* adapter, void* guest_device, void* guest_list, void* backing, UINT groups,
                             void* title_root, UINT constants, void* guest_output)
{
    auto* guest = static_cast<ID3D12Device*>(guest_device);
    ComPtr<ID3D12Device> device;
    check(D3D12XboxCreateDevice(adapter, nullptr, IID_PPV_ARGS(&device)), "open producer's desktop device");
    auto list = indirect_native_object<ID3D12GraphicsCommandList>(device.Get(), guest, static_cast<IUnknown*>(guest_list));
    constant_count = constants;
    const UINT size = sizeof(initial) + constant_count * sizeof(UINT);
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, size, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    check(static_cast<HRESULT>(invoke(guest, 133, word(backing), word(&desc), D3D12_RESOURCE_STATE_COMMON, 0,
          word(&__uuidof(ID3D12Resource)), word(placed.GetAddressOf()))), "place indirect arguments");
    auto args = indirect_native_object<ID3D12Resource>(device.Get(), guest, placed.Get());
    upload = make_buffer(device.Get(), size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    readback = make_buffer(device.Get(), size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const D3D12_RANGE empty{0, 0};
    void* mapped = nullptr;
    check(upload->Map(0, &empty, &mapped), "map dispatch record upload");
    if (constant_count) {
        const auto constants = record_constants(constant_count);
        const UINT suffix[]{groups, 1, 1, initial.back()};
        auto* dest = static_cast<UINT*>(mapped);
        dest[0] = initial.front();
        std::memcpy(dest + 1, constants.data(), constant_count * sizeof(UINT));
        std::memcpy(dest + 1 + constant_count, suffix, sizeof(suffix));
    } else {
        std::memcpy(mapped, initial.data(), sizeof(initial));
    }
    upload->Unmap(0, nullptr);
    transition(list.Get(), args.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(args.Get(), 0, upload.Get(), 0, size);
    auto state = D3D12_RESOURCE_STATE_COPY_DEST;
    if (groups && !constant_count) {
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
    list->CopyBufferRegion(readback.Get(), 0, args.Get(), 0, size);
    transition(list.Get(), args.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    if (constant_count) {
        const ConstantDispatchSignature rooted(0, 0, constant_count);
        auto sig = rooted.desc();
        if (constant_count == 3) sig.ByteStride += sizeof(UINT);  // the trailing guard also serves as stride padding
        check(guest->CreateCommandSignature(&sig, static_cast<ID3D12RootSignature*>(title_root),
                                            IID_PPV_ARGS(&signature)), "create rooted guest dispatch signature");
        native_list = list;
        output = indirect_native_object<ID3D12Resource>(device.Get(), guest, static_cast<IUnknown*>(guest_output));
        const auto bytes = output->GetDesc().Width;
        output_upload = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        output_readback = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        check(output_upload->Map(0, &empty, &mapped), "map sentinel output upload");
        std::memset(mapped, 0xc5, static_cast<std::size_t>(bytes));
        output_upload->Unmap(0, nullptr);
        transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(output.Get(), 0, output_upload.Get(), 0, bytes);
        transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        return;
    }
    D3D12_INDIRECT_ARGUMENT_DESC command{D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH};
    const D3D12_COMMAND_SIGNATURE_DESC sig{sizeof(D3D12_DISPATCH_ARGUMENTS), 1, &command, 0};
    check(guest->CreateCommandSignature(&sig, nullptr, IID_PPV_ARGS(&signature)), "create guest dispatch signature");
}

void Slot67Indirect::execute(void* list, UINT commands, bool count_buffer, bool outside) const
{
    invoke(list, 61, word(signature.Get()), commands, word(placed.Get()), outside ? sizeof(initial) - sizeof(UINT) : offset,
           count_buffer ? word(placed.Get()) : 0, count_buffer ? offset : 0);
}

bool Slot67Indirect::check_record(UINT groups) const
{
    if (constant_count) {
        const UINT bytes = sizeof(initial) + constant_count * sizeof(UINT);
        const D3D12_RANGE range{0, bytes}, empty{0, 0};
        void* mapped = nullptr;
        check(readback->Map(0, &range, &mapped), "map rooted GPU argument readback");
        const auto* actual = static_cast<const UINT*>(mapped);
        const auto constants = record_constants(constant_count);
        const bool match = actual[0] == initial.front() &&
            std::memcmp(actual + 1, constants.data(), constant_count * sizeof(UINT)) == 0 &&
            actual[1 + constant_count] == groups && actual[2 + constant_count] == 1 &&
            actual[3 + constant_count] == 1 && actual[4 + constant_count] == initial.back();
        readback->Unmap(0, &empty);
        std::printf("GPU rooted argument record: %u constants [%u,1,1]\n", constant_count, groups);
        return match;
    }
    const D3D12_RANGE range{0, sizeof(initial)}, empty{0, 0};
    void* mapped = nullptr;
    check(readback->Map(0, &range, &mapped), "map GPU argument readback after consumer fence");
    Record actual{};
    std::memcpy(actual.data(), mapped, sizeof(actual));
    readback->Unmap(0, &empty);
    const Record expected{initial.front(), groups, groups ? 1u : 0u, groups ? 1u : 0u, initial.back()};
    std::printf("GPU argument record: %08x [%u,%u,%u] %08x\n", actual[0], actual[1], actual[2], actual[3], actual[4]);
    std::fflush(stdout);
    return actual == expected;
}

void Slot67Indirect::capture_output(bool dispatched) const
{
    // Preserve the state the shim will use when it closes the recording.
    const auto state = dispatched ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COMMON;
    transition(native_list.Get(), output.Get(), state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    native_list->CopyBufferRegion(output_readback.Get(), 0, output.Get(), 0, output->GetDesc().Width);
    transition(native_list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, state);
}

void Slot67Indirect::read_output(UINT* words) const
{
    const auto bytes = static_cast<std::size_t>(output->GetDesc().Width);
    const D3D12_RANGE range{0, bytes}, empty{0, 0};
    void* mapped = nullptr;
    check(output_readback->Map(0, &range, &mapped), "map translated output readback");
    std::memcpy(words, mapped, bytes);
    output_readback->Unmap(0, &empty);
}
