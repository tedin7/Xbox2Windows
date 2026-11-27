#include "desktop_gpu.h"
#include "shader_dxil_compiler.h"
#include <dxgi1_4.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

using Microsoft::WRL::ComPtr;
int wmain(int argc,wchar_t** argv) {
    if(argc!=5) return 2;
    try {
        std::ifstream input(argv[3],std::ios::binary);
        if(!input) return 2;
        std::string source{std::istreambuf_iterator<char>(input),{}};
        x2w::ShaderDxilCompiler compiler(argv[2]);
        auto shader=compiler.compile(source,L"cs_6_0",argv[4]);
        ComPtr<IDXGIFactory4> factory;
        ComPtr<IDXGIAdapter> adapter;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"factory");
        const bool warp=std::wstring(argv[1])==L"warp";
        if(warp) check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)),"warp");
        else {
            ComPtr<IDXGIAdapter1> candidate;
            for(UINT i=0;factory->EnumAdapters1(i,candidate.ReleaseAndGetAddressOf())==S_OK;++i) {
                DXGI_ADAPTER_DESC1 description{};
                check(candidate->GetDesc1(&description),"adapter description");
                if(!(description.Flags&DXGI_ADAPTER_FLAG_SOFTWARE) &&
                   SUCCEEDED(D3D12CreateDevice(candidate.Get(),D3D_FEATURE_LEVEL_11_0,__uuidof(ID3D12Device),nullptr))) {
                    adapter=candidate;
                    break;
                }
            }
            if(!adapter) { std::puts("no D3D12 hardware adapter"); return 77; }
        }
        ComPtr<ID3D12Device> device;
        check(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)),"device");
        D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_0};
        if(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,&model,sizeof(model))) ||
           model.HighestShaderModel<D3D_SHADER_MODEL_6_0) { std::puts("requires shader model 6"); return 77; }
        D3D12_ROOT_PARAMETER param{};
        param.ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
        D3D12_ROOT_SIGNATURE_DESC layout{1,&param};
        ComPtr<ID3DBlob> serialized;
        check(D3D12SerializeRootSignature(&layout,D3D_ROOT_SIGNATURE_VERSION_1,&serialized,nullptr),"serialize");
        ComPtr<ID3D12RootSignature> root;
        check(device->CreateRootSignature(0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(&root)),"root");
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature=root.Get();
        desc.CS={shader.dxil.data(),shader.dxil.size()};
        ComPtr<ID3D12PipelineState> pipeline;
        check(device->CreateComputePipelineState(&desc,IID_PPV_ARGS(&pipeline)),"pipeline");
        ComputeCommands cmd(device.Get(),pipeline.Get());
        constexpr unsigned words=29;
        constexpr size_t bytes=32*words*sizeof(unsigned);
        auto output=make_buffer(device.Get(),bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto readback=make_buffer(device.Get(),bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
        cmd.list->SetComputeRootSignature(root.Get());
        cmd.list->SetComputeRootUnorderedAccessView(0,output->GetGPUVirtualAddress());
        cmd.list->Dispatch(1,1,1);
        transition(cmd.list.Get(),output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd.list->CopyResource(readback.Get(),output.Get());
        cmd.submit_and_wait(device.Get());
        void* data=nullptr;
        D3D12_RANGE range{0,bytes};
        check(readback->Map(0,&range,&data),"readback");
        for(unsigned lane=0;lane<32;++lane) {
            unsigned other=(lane+1)%32;
            auto bits=[](float f){ unsigned u; std::memcpy(&u,&f,sizeof(u)); return u; };
            float a=static_cast<float>(lane+1);
            std::array<unsigned,words> expected{100+other,200+other,300+other,400+other,100,231,500+other,600+other,
                700+other,800+other,900+other,bits(a+2),bits(a-2),bits(2-a),bits(2*a),
                lane*3,lane*2,(0x80000000u+lane)-1,lane+11,700+other,800+other,900+other,1000+other,
                303+other,404+other,101+other,202+other,lane+100,99};
            for(unsigned i=0;i<words;++i) {
                auto actual=static_cast<const unsigned*>(data)[words*lane+i];
                if(actual!=expected[i]) {
                    std::fprintf(stderr,"lane %u component %u expected %u got %u\n",lane,i,expected[i],actual);
                    return 1;
                }
            }
        }
        D3D12_RANGE empty{};
        readback->Unmap(0,&empty);
        std::printf("%u LDS and vector ALU readback words matched\n",32*words);
    } catch(const std::exception& e) {
        std::fprintf(stderr,"%s\n",e.what());
        return 1;
    }
}
