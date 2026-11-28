// Cycle 1 of the global SRV work: can a native compute module read through ResourceDescriptorHeap[index]?
//
// Two shaders select a heap slot from a GPU word and copy that slot's raw buffer to a UAV: one is HLSL compiled by DXC
// (the reference), the other is the textual DXIL module the native compiler builds (dx.op.createHandleFromHeap, opcode
// 218). Both run on the same pipeline shape (root flag CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED) with two different selector
// words: the readbacks must be the distinct data of slot A and slot B, and a UAV guard word past the copied range must
// stay intact. Usage: native_heap_handle_probe ADAPTER SDK_DIRECTORY WORK_DIRECTORY SHADER_FLAGS
// ADAPTER is warp or hardware. Exit 77 when the adapter lacks shader model 6.6 or resource binding tier 3.
#include "dxil_module.h"
#include "shader_dxil_compiler.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT words = 64;
constexpr UINT guard_word = 0xC0FFEE11u;
constexpr UINT sentinel = 0xCAFEF00Du;

UINT slot_word(UINT slot, UINT i) { return (slot == 0 ? 0x1000u : 0x9000u) + i * (slot == 0 ? 7u : 13u); }

ComPtr<ID3D12Resource> make_buffer(ID3D12Device *device, UINT64 bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state,
                                   D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_HEAP_PROPERTIES heap{type};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    ComPtr<ID3D12Resource> resource;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource))))
        return {};
    return resource;
}

struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    UINT64 value = 0;
    bool open(ID3D12Device *d) {
        device = d;
        D3D12_COMMAND_QUEUE_DESC q{D3D12_COMMAND_LIST_TYPE_DIRECT};
        return SUCCEEDED(d->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) &&
               SUCCEEDED(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) &&
               SUCCEEDED(d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list))) &&
               SUCCEEDED(list->Close()) && SUCCEEDED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    }
    template <class Record> bool run(Record &&record) {
        if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.Get(), nullptr))) return false;
        record(list.Get());
        if (FAILED(list->Close())) return false;
        ID3D12CommandList *lists[]{list.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence.Get(), ++value))) return false;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        fence->SetEventOnCompletion(value, event);
        WaitForSingleObject(event, 30000);
        CloseHandle(event);
        return fence->GetCompletedValue() >= value;
    }
};

void transition(ID3D12GraphicsCommandList *list, ID3D12Resource *r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &b);
}

// The dxil module of the native compiler: output[tid] = ResourceDescriptorHeap[selection[0]][tid].
std::string native_module(unsigned shader_flags) {
    x2w::DxilModule m;
    m.features = shader_flags | 16;
    const auto selection = m.handle({0, 0, 11, 0, 0, 0});
    const auto output = m.handle({1, 0, 11, 0, 0, 0});
    const auto thread = m.invoke("i32", "threadId.i32", 93, {{"i32", "0"}}, "nounwind readnone");
    const auto address = m.binary("shl", thread, "2");
    const auto word = m.invoke("%dx.types.ResRet.i32", "rawBufferLoad.i32", 139,
                               {{"%dx.types.Handle", selection}, {"i32", "0"}, {"i32", "undef"}, {"i8", "1"}, {"i32", "4"}},
                               "nounwind readonly");
    const auto slot = m.value("extractvalue %dx.types.ResRet.i32 " + word + ", 0");
    const auto data = m.heap_handle(slot, 11, 0);
    const auto loaded = m.invoke("%dx.types.ResRet.i32", "rawBufferLoad.i32", 139,
                                 {{"%dx.types.Handle", data}, {"i32", address}, {"i32", "undef"}, {"i8", "1"}, {"i32", "4"}},
                                 "nounwind readonly");
    const auto value = m.value("extractvalue %dx.types.ResRet.i32 " + loaded + ", 0");
    m.invoke("void", "rawBufferStore.i32", 140,
             {{"%dx.types.Handle", output}, {"i32", address}, {"i32", "undef"}, {"i32", value}, {"i32", "undef"},
              {"i32", "undef"}, {"i32", "undef"}, {"i8", "1"}, {"i32", "4"}},
             "nounwind");
    m.body << "  ret void\n";
    return m.finish_compute(words, 1, 1);
}

const char *reference_hlsl =
    "ByteAddressBuffer selection : register(t0);\n"
    "RWByteAddressBuffer output : register(u0);\n"
    "[numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n"
    "    ByteAddressBuffer data = ResourceDescriptorHeap[selection.Load(0)];\n"
    "    output.Store(id.x * 4, data.Load(id.x * 4));\n"
    "}\n";
} // namespace

int wmain(int argc, wchar_t **argv) {
    if (argc != 5) {
        std::fputs("usage: native_heap_handle_probe ADAPTER SDK WORK SHADER_FLAGS\n", stderr);
        return 2;
    }
    const std::wstring adapter = argv[1];
    const unsigned shader_flags = static_cast<unsigned>(std::stoull(argv[4]));
    const std::filesystem::path work = argv[3];
    std::filesystem::create_directories(work);

    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> chosen;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 77;
    if (adapter == L"warp") {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&chosen)))) return 77;
    } else {
        for (UINT i = 0; i < 8; ++i) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapters1(i, &candidate) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{};
            candidate->GetDesc1(&desc);
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { chosen = candidate; break; }
        }
        if (!chosen) { std::puts("no hardware adapter"); return 77; }
    }
    if (FAILED(D3D12CreateDevice(chosen.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) { std::puts("no device"); return 77; }
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof model)) ||
        model.HighestShaderModel < D3D_SHADER_MODEL_6_6 ||
        FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof options)) ||
        options.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
        std::printf("adapter lacks shader model 6.6 or resource binding tier 3 (tier %d)\n", int(options.ResourceBindingTier));
        return 77;
    }
    // The native module declares WaveSize=32 (the guest wave), so an adapter without 32-lane waves cannot create it.
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 waves{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &waves, sizeof waves)) || !waves.WaveOps ||
        waves.WaveLaneCountMin > 32 || waves.WaveLaneCountMax < 32) {
        std::printf("adapter has no 32-lane waves (%u..%u)\n", waves.WaveLaneCountMin, waves.WaveLaneCountMax);
        return 77;
    }
    try {
        x2w::ShaderDxilCompiler compiler(argv[2]);
        const auto module = native_module(shader_flags);
        std::ofstream(work / "native_module.ll") << module;
        x2w::CompiledShader reference, native;
        reference = compiler.compile(reference_hlsl, L"cs_6_6", work);
        try {
            native = compiler.compile(module, L"cs_6_6", work, x2w::ShaderSourceFormat::dxil_ir);
        } catch (const std::exception &error) {
            std::printf("FAIL: the native module was refused by the compiler or validator: %s\n", error.what());
            return 3;
        }

        // Both containers go to the work directory so the check script can print how they differ.
        for (const auto &[name, shader] : {std::pair<const char *, const x2w::CompiledShader *>{"reference.dxil", &reference},
                                           std::pair<const char *, const x2w::CompiledShader *>{"native.dxil", &native}})
            std::ofstream(work / name, std::ios::binary)
                .write(reinterpret_cast<const char *>(shader->dxil.data()), static_cast<std::streamsize>(shader->dxil.size()));

        // Root signature: root SRV t0 (selector), root UAV u0 (output), and the heap indexed directly.
        D3D12_ROOT_PARAMETER1 parameters[2]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC versioned{};
        versioned.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        versioned.Desc_1_1.NumParameters = 2;
        versioned.Desc_1_1.pParameters = parameters;
        versioned.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(0x400); // CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
        ComPtr<ID3DBlob> blob, error;
        ComPtr<ID3D12RootSignature> root;
        if (FAILED(D3D12SerializeVersionedRootSignature(&versioned, &blob, &error)) ||
            FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)))) {
            std::puts("FAIL: root signature with the heap-indexed flag");
            return 4;
        }
        ComPtr<ID3D12PipelineState> states[2];
        const x2w::CompiledShader *shaders[2]{&reference, &native};
        for (int k = 0; k < 2; ++k) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
            pso.pRootSignature = root.Get();
            pso.CS = {shaders[k]->dxil.data(), shaders[k]->dxil.size()};
            if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&states[k])))) {
                std::printf("FAIL: pipeline state of the %s shader\n", k ? "native" : "reference");
                return 5;
            }
        }

        // Heap slots 0 and 1: two raw buffers with different data. Slot 2: a buffer of another size, never selected.
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        ComPtr<ID3D12DescriptorHeap> heap;
        if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)))) return 6;
        const UINT step = device->GetDescriptorHandleIncrementSize(heap_desc.Type);
        ComPtr<ID3D12Resource> data[2], decoy;
        void *mapped = nullptr;
        for (UINT s = 0; s < 2; ++s) {
            data[s] = make_buffer(device.Get(), words * 4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            if (!data[s] || FAILED(data[s]->Map(0, nullptr, &mapped))) return 6;
            for (UINT i = 0; i < words; ++i) static_cast<UINT *>(mapped)[i] = slot_word(s, i);
            data[s]->Unmap(0, nullptr);
        }
        decoy = make_buffer(device.Get(), words * 4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R32_TYPELESS;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Buffer.NumElements = words;
        srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
        for (ID3D12Resource *r : {data[0].Get(), data[1].Get(), decoy.Get()}) {
            device->CreateShaderResourceView(r, &srv, cpu);
            cpu.ptr += step;
        }

        auto selection = make_buffer(device.Get(), 256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto output = make_buffer(device.Get(), (words + 1) * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto seed = make_buffer(device.Get(), (words + 1) * 4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback = make_buffer(device.Get(), (words + 1) * 4, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!selection || !output || !seed || !readback) return 6;
        Gpu gpu;
        if (!gpu.open(device.Get())) return 6;
        int failures = 0;
        for (int k = 0; k < 2; ++k) {
            const char *name = k ? "native" : "reference";
            std::vector<UINT> seen[2];
            for (UINT selected = 0; selected < 2; ++selected) {
                seed->Map(0, nullptr, &mapped);
                for (UINT i = 0; i < words; ++i) static_cast<UINT *>(mapped)[i] = sentinel;
                static_cast<UINT *>(mapped)[words] = guard_word;
                seed->Unmap(0, nullptr);
                selection->Map(0, nullptr, &mapped);
                static_cast<UINT *>(mapped)[0] = selected;
                selection->Unmap(0, nullptr);
                const bool ok = gpu.run([&](ID3D12GraphicsCommandList *list) {
                    transition(list, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
                    list->CopyResource(output.Get(), seed.Get());
                    transition(list, output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    ID3D12DescriptorHeap *heaps[]{heap.Get()};
                    list->SetDescriptorHeaps(1, heaps);
                    list->SetComputeRootSignature(root.Get());
                    list->SetPipelineState(states[k].Get());
                    list->SetComputeRootShaderResourceView(0, selection->GetGPUVirtualAddress());
                    list->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
                    list->Dispatch(1, 1, 1);
                    transition(list, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    list->CopyResource(readback.Get(), output.Get());
                    transition(list, output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                });
                if (!ok || FAILED(device->GetDeviceRemovedReason())) {
                    std::printf("FAIL: %s dispatch selecting slot %u did not complete\n", name, selected);
                    return 7;
                }
                readback->Map(0, nullptr, &mapped);
                seen[selected].assign(static_cast<UINT *>(mapped), static_cast<UINT *>(mapped) + words + 1);
                readback->Unmap(0, nullptr);
                bool match = true;
                for (UINT i = 0; i < words; ++i) match &= seen[selected][i] == slot_word(selected, i);
                const bool guard = seen[selected][words] == guard_word;
                std::printf("%s selector %u: data %s, guard %s\n", name, selected, match ? "matches the CPU reference" : "DIFFERS",
                            guard ? "intact" : "OVERWRITTEN");
                failures += !match || !guard;
            }
            if (seen[0] == seen[1]) {
                std::printf("FAIL: %s read the same data for both selectors\n", name);
                ++failures;
            }
        }
        std::puts(failures ? "native_heap_handle_probe: FAIL" : "native_heap_handle_probe: PASS");
        return failures ? 1 : 0;
    } catch (const std::exception &error) {
        std::printf("FAIL: %s\n", error.what());
        return 8;
    }
}
