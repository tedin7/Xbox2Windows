#include "desktop_gpu.h"
#include "native_buffer_store_view.h"
#include "native_compute_pipeline.h"
#include "xmem_test_device.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
using namespace x2w;
namespace {
constexpr UINT elements = 2048, guard = 8;
struct Window {
    ComPtr<ID3D12Resource> resource, upload, readback;
    UINT bytes = 0;
    bool write = false;
};
UINT constant(UINT index) {
    constexpr UINT words[]{3,9,37,5};
    return words[index % 4];
}
UINT format_width(ComputeViewKind kind, UINT bytes) {
    if (kind == ComputeViewKind::typed_buffer_srv) return 16;
    if (kind == ComputeViewKind::buffer_srv || kind == ComputeViewKind::root_buffer_srv ||
        kind == ComputeViewKind::raw_buffer_uav) return 4;
    return bytes;
}
DXGI_FORMAT typed_format(unsigned bytes, bool half, bool reference) {
    const DXGI_FORMAT uints[]{DXGI_FORMAT_R32_UINT,DXGI_FORMAT_R32G32_UINT,DXGI_FORMAT_R32G32B32_UINT,DXGI_FORMAT_R32G32B32A32_UINT};
    const DXGI_FORMAT floats[]{DXGI_FORMAT_R32_FLOAT,DXGI_FORMAT_R32G32_FLOAT,DXGI_FORMAT_R32G32B32_FLOAT,DXGI_FORMAT_R32G32B32A32_FLOAT};
    if (half) return bytes == 2 ? DXGI_FORMAT_R16_FLOAT : bytes == 4 ? DXGI_FORMAT_R16G16_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
    return (reference ? floats : uints)[bytes/4-1];
}
void report_adapter(ID3D12Device* device) {
    const LUID luid = device->GetAdapterLuid();
    std::fprintf(stderr, "adapter_luid=%08lx:%08lx\n", static_cast<unsigned long>(luid.HighPart), luid.LowPart);
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 desc{};
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) ||
        FAILED(adapter->GetDesc1(&desc))) {
        std::fputs("adapter=unavailable via DXGI\n", stderr);
        return;
    }
    std::fprintf(stderr, "adapter=%ls vendor=%04x device=%04x\n", desc.Description, desc.VendorId, desc.DeviceId);
    LARGE_INTEGER driver{};
    if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver)))
        std::fprintf(stderr, "driver=%u.%u.%u.%u\n", HIWORD(driver.HighPart), LOWORD(driver.HighPart),
                     HIWORD(driver.LowPart), LOWORD(driver.LowPart));
    else
        std::fputs("driver=unavailable via DXGI\n", stderr);
}
}

// A reference directory supplies independently compiled DXIL and its user-data layout.
// --direct exercises the same direct DXIL frontend used by the runtime.
int wmain(int argc, wchar_t **argv) {
    if (argc != 6 && argc != 7) return 2;
    try {
        ComPtr<ID3D12Device> device;
        if (GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", nullptr, 0)) {
            bool unmatched = false;
            device = create_device(false, unmatched, [](IUnknown* adapter, REFIID iid, void** result) {
                return D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, iid, result);
            }, stderr);
            if (!device) {
                if (!unmatched)
                    std::fputs("Unable to enumerate D3D12 hardware adapters\n", stderr);
                return 2;
            }
        } else {
            const auto device_hr = D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device));
            if (device_hr == DXGI_ERROR_UNSUPPORTED) return 77;
            check(device_hr, "device");
        }
        report_adapter(device.Get());
        ShaderDxilCompiler compiler(argv[1]);
        fs::path marker=argv[2], dest=argv[5];
        const bool null_array = argc == 7 && std::wstring_view(argv[6]) == L"--direct-null-srv";
        const bool direct = null_array || (argc == 7 && std::wstring_view(argv[6]) == L"--direct");
        if (direct) {
            D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
            D3D12_FEATURE_DATA_D3D12_OPTIONS1 waves{};
            if (FAILED(check_feature(device.Get(), D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) ||
                model.HighestShaderModel < D3D_SHADER_MODEL_6_6 ||
                FAILED(check_feature(device.Get(), D3D12_FEATURE_D3D12_OPTIONS1, &waves, sizeof(waves))) ||
                !waves.WaveOps || waves.WaveLaneCountMin > 32 || waves.WaveLaneCountMax < 32) {
                std::fprintf(stderr, "requires shader model 6.6 with wave32\n");
                return 77;
            }
        }
        const bool reference = argc == 7 && !direct;
        auto p=create_native_compute_pipeline(device.Get(),compiler,marker,std::stoul(argv[4]),argv[3],direct);
        std::vector<UINT> args(p.constant_words);
        UINT index=0;
        for (auto &block:p.constants)
            for (unsigned j=0;j<block.count;++j) args[index++]=constant(j);
        UINT roots[64]{};
        if (reference) {
            fs::path base=fs::path(argv[6])/marker.stem().wstring().substr(4);
            std::ifstream input(base.wstring()+L".dxil",std::ios::binary);
            std::vector<char> dxil{std::istreambuf_iterator<char>(input),{}};
            std::ifstream layout(base.wstring()+L".roots");
            UINT addr, slot, dword; std::string type;
            while (layout>>addr>>slot>>type>>dword) {
                if (addr>=64 || slot>=16 || (type!="Table" && type!="Constants")) return 2;
                roots[addr]= type=="Table" ? slot*256*32 : constant(dword);
            }
            D3D12_ROOT_PARAMETER param{};
            param.ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            param.Constants.Num32BitValues=64;
            D3D12_ROOT_SIGNATURE_DESC rd{1,&param,0,nullptr,D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED};
            ComPtr<ID3DBlob> blob;
            check(D3D12SerializeRootSignature(&rd,D3D_ROOT_SIGNATURE_VERSION_1,&blob,nullptr),"reference root");
            p.root.Reset();p.state.Reset();
            check(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&p.root)),"reference root");
            D3D12_COMPUTE_PIPELINE_STATE_DESC sd{};sd.pRootSignature=p.root.Get();sd.CS={dxil.data(),dxil.size()};
            check(device->CreateComputePipelineState(&sd,IID_PPV_ARGS(&p.state)),"reference PSO");
        }
        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,reference?4096u:UINT(p.bindings.size()),D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        ComPtr<ID3D12DescriptorHeap> heap;
        check(device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)),"heap");
        const UINT step=device->GetDescriptorHandleIncrementSize(hd.Type);
        ComputeCommands cmd(device.Get(),p.state.Get(),D3D12_COMMAND_LIST_TYPE_DIRECT);
        std::vector<Window> windows;
        std::vector<IndexedAtomicView> atomic;
        for (const auto &b:p.bindings) {
            const UINT position=reference?b.root_parameter*256+b.byte_offset/32:UINT(windows.size());
            if (position>=hd.NumDescriptors) return 2;
            auto cpu=heap->GetCPUDescriptorHandleForHeapStart();cpu.ptr+=UINT64(position)*step;
            Window w;w.write=compute_buffer_uav(b.kind);
            // A byte window is the byte-address rewrite of the same raw table view.
            const bool vector_bytes = b.kind == ComputeViewKind::raw_vector_srv || b.kind == ComputeViewKind::byte_window_srv;
            const bool root_bytes = b.kind == ComputeViewKind::root_buffer_srv;
            if (!w.write && b.kind!=ComputeViewKind::typed_buffer_srv && b.kind!=ComputeViewKind::buffer_srv && !vector_bytes && !root_bytes) return 2;
            const bool raw=root_bytes || vector_bytes || b.kind==ComputeViewKind::buffer_srv || b.kind==ComputeViewKind::raw_buffer_uav;
            const bool indexed = b.kind == ComputeViewKind::indexed_atomic_uav;
            const UINT stride = indexed && b.byte_offset == 32 ? 16 :
                                indexed && b.byte_offset == 64 ? 12 : format_width(b.kind,b.required_bytes);
            w.bytes=(elements+2*guard)*stride;
            w.upload=make_buffer(device.Get(),w.bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
            void *mapped;
            check(w.upload->Map(0,nullptr,&mapped),"seed");
            std::memset(mapped,0x69,w.bytes);
            if (!w.write) {
                auto *words=reinterpret_cast<UINT*>(static_cast<char*>(mapped)+guard*stride);
                constexpr UINT packed[]{0x7fff8000, 0x40008001, 0x007fff81, 0xfc003c00, 0x0001bc00, 0x42987ec5};
                for (UINT j=0;j<elements*stride/4;++j)
                    words[j]=root_bytes ? 0x51aa0000u+j : vector_bytes ? packed[j%6] : raw ? constant(j) : 0x3f800000u+(b.byte_offset/32)*256+j;
            }
            w.upload->Unmap(0,nullptr);
            if (w.write) {
                auto desc=w.upload->GetDesc();desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                w.resource=resource(device.Get(),desc,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST);
                cmd.list->CopyResource(w.resource.Get(),w.upload.Get());
                transition(cmd.list.Get(),w.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
                v.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;v.Buffer.FirstElement=guard;v.Buffer.NumElements=elements;
                v.Format=raw?DXGI_FORMAT_R32_TYPELESS:typed_format(stride,b.kind==ComputeViewKind::half_buffer_uav ||
                                                                    b.kind==ComputeViewKind::packed_d16_x_uav,false);
                if(raw)v.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_RAW;
                if (indexed) {
                    if (stride > 4) {
                        v.Format = DXGI_FORMAT_UNKNOWN;
                        v.Buffer.StructureByteStride = stride;
                    } else if (b.byte_offset == 128) {
                        v.Format = DXGI_FORMAT_R32_TYPELESS;
                        v.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                    }
                }
                auto truncated = v;
                IndexedAtomicView shape;
                if(!native_buffer_store_view(b,v,w.bytes,&shape) ||
                   native_buffer_store_view(b,truncated,w.bytes-guard*stride-1,&shape)) return 2;
                if (indexed) atomic.push_back(shape);
                if(reference&&!raw)v.Format=typed_format(stride,b.kind==ComputeViewKind::half_buffer_uav ||
                                                         b.kind==ComputeViewKind::packed_d16_x_uav,true);
                device->CreateUnorderedAccessView(w.resource.Get(),nullptr,&v,cpu);
                w.readback=make_buffer(device.Get(),w.bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
            } else {
                w.resource=w.upload;
                D3D12_SHADER_RESOURCE_VIEW_DESC v{};v.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;
                v.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                v.Format=raw?DXGI_FORMAT_R32_TYPELESS:typed_format(16,false,reference);v.Buffer.FirstElement=guard;v.Buffer.NumElements=elements;
                if (root_bytes) {
                    if (b.required_bytes % 4 || b.required_bytes > elements * 4) return 2;
                    v.Buffer.NumElements = b.required_bytes / 4;
                }
                if(raw)v.Buffer.Flags=D3D12_BUFFER_SRV_FLAG_RAW;
                auto *source = w.resource.Get();
                if (null_array && b.kind == ComputeViewKind::typed_buffer_srv && b.array_position == 1) {
                    auto guest = v;
                    guest.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
                    auto bad = guest;
                    bad.Format = DXGI_FORMAT_R8G8B8A8_UINT;
                    if (x2w_startup::native_null_typed_array_srv(0,0,&guest,sizeof(guest),b.required_bytes,v) ||
                        x2w_startup::native_null_typed_array_srv(18,1,&guest,sizeof(guest),b.required_bytes,v) ||
                        x2w_startup::native_null_typed_array_srv(18,0,&bad,sizeof(bad),b.required_bytes,v) ||
                        x2w_startup::native_null_typed_array_srv(18,0,&guest,sizeof(guest)-1,b.required_bytes,v) ||
                        !x2w_startup::native_null_typed_array_srv(18,0,&guest,sizeof(guest),b.required_bytes,v) ||
                        v.Format != DXGI_FORMAT_R32G32B32A32_UINT || v.Buffer.FirstElement != guard ||
                        v.Buffer.NumElements != elements) return 2;
                    source = nullptr;
                }
                device->CreateShaderResourceView(source,&v,cpu);
            }
            std::printf("%u %u %u %u %u\n", b.root_parameter,b.byte_offset,unsigned(b.kind),stride,w.write);
            windows.push_back(std::move(w));
        }
        if(reference){
            ID3D12DescriptorHeap* h=heap.Get();cmd.list->SetDescriptorHeaps(1,&h);cmd.list->SetComputeRootSignature(p.root.Get());
            cmd.list->SetComputeRoot32BitConstants(0,64,roots,0);
        } else p.bind(cmd.list.Get(),heap.Get(),heap->GetGPUDescriptorHandleForHeapStart(),step,args.data(),atomic);
        cmd.list->Dispatch(2,1,1);
        for(auto &w:windows) if(w.write) {
            transition(cmd.list.Get(),w.resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
            cmd.list->CopyResource(w.readback.Get(),w.resource.Get());
        }
        cmd.submit_and_wait(device.Get());check(device->GetDeviceRemovedReason(),"completion");
        std::ofstream out(dest,std::ios::binary);
        for(auto &w:windows)if(w.write){
            void *mapped;check(w.readback->Map(0,nullptr,&mapped),"readback");
            out.write(static_cast<const char*>(mapped),w.bytes);w.readback->Unmap(0,nullptr);
        }
        return out.good()?0:1;
    } catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
