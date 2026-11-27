#include "desktop_gpu.h"
#include "gfx1013_sampled_pixel.h"
#include "gfx1013_vertex_id.h"
#include "shader_dxil_compiler.h"
#include "xmem_test_device.h"
#include <d3d12sdklayers.h>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iterator>

using Microsoft::WRL::ComPtr;
namespace {
using Color=std::array<float,4>;
constexpr unsigned width=8,height=4;
constexpr Color left{0.125f,1,2,1},right{2,0.5f,1,1};
constexpr Color first{0.25f,0,0,1},last{4,0,0,1};
constexpr Color sentinel{8,8,8,8};
constexpr DXGI_FORMAT formats[]{DXGI_FORMAT_R32G32B32A32_FLOAT,DXGI_FORMAT_R32_FLOAT,
                                DXGI_FORMAT_R11G11B10_FLOAT,DXGI_FORMAT_R16_FLOAT};

// Two independently sampled inputs. Coordinates include a runtime offset.
// Rename registers and use table 0 to exercise decoded operands and bindings.
std::string packet(unsigned color_mask=7, bool implicit=false, unsigned sampler_word2=0) {
    std::vector<unsigned> w;
    auto emit=[&](std::initializer_list<unsigned> a){w.insert(w.end(),a);};
    auto mov=[&](unsigned d,unsigned s){emit({0xbe800300u|d<<16|s});};
    auto interp=[&](unsigned d,unsigned c){for(unsigned i=0;i<2;++i)emit({0xc8000000u|d<<18|i<<16|c<<8|i});};
    auto sample=[&](unsigned d,unsigned mask,unsigned tex){
        emit({(implicit ? 0xf0800008u : 0xf09c0008u)|mask<<8,6u|d<<8|(tex/4)<<16|(44u/4)<<21});
    };
    auto mix=[&](unsigned d,unsigned src,bool high){
        emit({(high?0xcc224000u:0xcc214000u)|d,0x82000000u|(256+src)<<9|(256+84)});
    };
    mov(20,8);mov(21,128);
    emit({0xf4080000u|32u<<6|10,0xfa000060u,0xf4100000u|16u<<6|10,0xfa000020u});
    emit({0xf4240000u|12u<<6|16,0xfa000038u}); // offsets into t0
    emit({0x92800000u|44u<<16|182u<<8|130u}); // s_bfm_b64 s[44:45], 2, 54
    emit({0xbe800400u|46u<<16|255u,sampler_word2}); // s_mov_b64 with a 32-bit literal
    mov(124,10);interp(4,0);interp(5,1);
    emit({0x06000000u|6u<<17|4u<<9|12u,0x06000000u|7u<<17|5u<<9|13u});
    sample(32,color_mask,16);sample(12,1,24);
    // Two VOP3 sources share a single trailing literal. The next dword is an opcode.
    emit({0xd54b0000u|84u,255u|(255u<<9)|(128u<<18),0x3f800000u});
    mix(80,32,false);mix(80,33,true);mix(81,34,false);mix(47,12,false);
    emit({0xf800040fu,0x80800000u|81u<<8|80u,0xf8001c13u,0x80808000u|47u,0xbf810000u});
    return {reinterpret_cast<const char*>(w.data()),w.size()*4};
}

std::string vertex_packet() {
    const auto vop2=[](unsigned op,unsigned dst,unsigned a,unsigned b) {
        return op<<25 | dst<<17 | b<<9 | a;
    };
    const auto vop1=[](unsigned op,unsigned dst,unsigned src) {
        return 0x7e000000u | dst<<17 | op<<9 | src;
    };
    const auto args=[](unsigned a,unsigned b,unsigned c){ return a | b<<9 | c<<18; };
    const unsigned words[]{
        vop2(26,7,129,0), vop2(27,6,130,0), vop2(27,7,130,7),
        vop1(5,7,263), vop1(5,6,262), vop1(1,10,128), vop1(1,11,242),
        0xd54b0008u,args(244,263,243), 0xd54b0009u,args(245,262,242),
        0xf80008cfu,0x0b0a0908u, 0xf8000203u,0x80800607u, 0xbf810000u
    };
    return {reinterpret_cast<const char*>(words),sizeof(words)};
}

void write_fixture(const std::filesystem::path& folder) {
    const auto write=[&](const char* name,const std::string& bytes) {
        std::ofstream out(folder/name,std::ios::binary);
        out.exceptions(std::ios::failbit|std::ios::badbit);
        out.write(bytes.data(),bytes.size());
    };
    write("vertex.bin",vertex_packet());
    write("pixel.bin",packet());
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    D3D12_ROOT_PARAMETER tables[2]{};
    for(unsigned i=0;i<2;++i) {
        ranges[i]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,4,0,i,0};
        tables[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        tables[i].DescriptorTable={1,&ranges[i]};
    }
    D3D12_ROOT_SIGNATURE_DESC desc{2,tables,0,nullptr,D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    ComPtr<ID3DBlob> root;
    check(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&root,nullptr),"fixture root");
    write("root.bin",{static_cast<const char*>(root->GetBufferPointer()),root->GetBufferSize()});
}

ComPtr<ID3D12DescriptorHeap> heap(ID3D12Device* dev,D3D12_DESCRIPTOR_HEAP_TYPE type,unsigned count,bool visible) {
    D3D12_DESCRIPTOR_HEAP_DESC d{type,count,visible?D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE:D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    ComPtr<ID3D12DescriptorHeap> h;check(dev->CreateDescriptorHeap(&d,IID_PPV_ARGS(&h)),"descriptor heap");return h;
}
ComPtr<ID3D12RootSignature> signature(ID3D12Device* dev,bool graphics) {
    D3D12_DESCRIPTOR_RANGE ranges[]{{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,graphics?3u:2u,0,0,0},
                                    {D3D12_DESCRIPTOR_RANGE_TYPE_UAV,1,0,0,2}};
    D3D12_ROOT_PARAMETER p[2]{};p[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    p[0].DescriptorTable={graphics?1u:2u,ranges};
    p[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[1].Constants.Num32BitValues=3;
    p[1].ShaderVisibility=D3D12_SHADER_VISIBILITY_VERTEX;
    D3D12_STATIC_SAMPLER_DESC s{};s.Filter=D3D12_FILTER_MIN_MAG_MIP_POINT;
    s.AddressU=s.AddressV=s.AddressW=D3D12_TEXTURE_ADDRESS_MODE_CLAMP;s.MaxLOD=D3D12_FLOAT32_MAX;
    s.MaxAnisotropy=1;s.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    D3D12_ROOT_SIGNATURE_DESC d{graphics?2u:1u,p,graphics?1u:0u,graphics?&s:nullptr};
    ComPtr<ID3DBlob> blob;check(D3D12SerializeRootSignature(&d,D3D_ROOT_SIGNATURE_VERSION_1,&blob,nullptr),"root serialization");
    ComPtr<ID3D12RootSignature> root;
    check(dev->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)),"root");
    return root;
}

void run(ID3D12Device* dev,x2w::ShaderDxilCompiler& compiler,const wchar_t* cache,
         const std::vector<unsigned char>& vs,const std::vector<unsigned char>& ps) {
    auto graphics=signature(dev,true),compute=signature(dev,false);
    auto cs=compiler.compile(R"(
Texture2D<float3> a:register(t0);Texture2D<float> b:register(t1);RWByteAddressBuffer dst:register(u0);
[numthreads(8,4,1)]void main(uint3 id:SV_DispatchThreadID){
dst.Store4(16*(id.y*8+id.x),asuint(float4(a.Load(int3(id.xy,0)),b.Load(int3(id.xy,0)))));})",L"cs_6_0",cache);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC g{};g.pRootSignature=graphics.Get();
    g.VS={vs.data(),vs.size()};g.PS={ps.data(),ps.size()};g.SampleMask=UINT_MAX;
    g.BlendState.RenderTarget[0].RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
    g.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;g.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;
    g.RasterizerState.DepthClipEnable=TRUE;g.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    g.NumRenderTargets=2;g.RTVFormats[0]=formats[2];g.RTVFormats[1]=formats[3];g.SampleDesc.Count=1;
    ComPtr<ID3D12PipelineState> gp,cp;
    const auto start=std::chrono::steady_clock::now();
    auto hr=dev->CreateGraphicsPipelineState(&g,IID_PPV_ARGS(&gp));
    const auto pso_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(FAILED(hr)){
        ComPtr<ID3D12InfoQueue> queue;
        if(SUCCEEDED(dev->QueryInterface(IID_PPV_ARGS(&queue)))){
            for(UINT64 i=0;i<queue->GetNumStoredMessages();++i){
                SIZE_T size=0;queue->GetMessage(i,nullptr,&size);
                std::vector<unsigned char> bytes(size);
                auto* msg=reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
                if(SUCCEEDED(queue->GetMessage(i,msg,&size)))std::fprintf(stderr,"%s\n",msg->pDescription);
            }
        }
    }
    check(hr,"graphics PSO");
    D3D12_COMPUTE_PIPELINE_STATE_DESC c{};c.pRootSignature=compute.Get();c.CS={cs.dxil.data(),cs.dxil.size()};
    check(dev->CreateComputePipelineState(&c,IID_PPV_ARGS(&cp)),"compute PSO");
    auto rtvs=heap(dev,D3D12_DESCRIPTOR_HEAP_TYPE_RTV,4,false);
    auto views=heap(dev,D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,6,true);
    auto cpu=[&](ID3D12DescriptorHeap* h,unsigned i){auto v=h->GetCPUDescriptorHandleForHeapStart();
        v.ptr+=SIZE_T(i)*dev->GetDescriptorHandleIncrementSize(h->GetDesc().Type);return v;};
    auto gpu=[&](unsigned i){auto v=views->GetGPUDescriptorHandleForHeapStart();
        v.ptr+=UINT64(i)*dev->GetDescriptorHandleIncrementSize(views->GetDesc().Type);return v;};
    auto constants=make_buffer(dev,64,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    void* data{};check(constants->Map(0,nullptr,&data),"map offsets");std::memset(data,0,64);constants->Unmap(0,nullptr);
    D3D12_SHADER_RESOURCE_VIEW_DESC buf{};buf.Format=DXGI_FORMAT_R32_TYPELESS;buf.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;
    buf.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;buf.Buffer.NumElements=16;buf.Buffer.Flags=D3D12_BUFFER_SRV_FLAG_RAW;
    dev->CreateShaderResourceView(constants.Get(),&buf,cpu(views.Get(),0));
    std::array<ComPtr<ID3D12Resource>,4> surfaces;
    for(unsigned i=0;i<4;++i){
        D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=i<2?2*width:width;
        d.Height=i<2?2*height:height;d.DepthOrArraySize=d.MipLevels=d.SampleDesc.Count=1;
        d.Format=formats[i];d.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        surfaces[i]=resource(dev,d,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_RENDER_TARGET);
        dev->CreateRenderTargetView(surfaces[i].Get(),nullptr,cpu(rtvs.Get(),i));
        dev->CreateShaderResourceView(surfaces[i].Get(),nullptr,cpu(views.Get(),1+i));
    }
    constexpr unsigned bytes=width*height*sizeof(Color);
    auto dest=make_buffer(dev,bytes,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback=make_buffer(dev,bytes,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=DXGI_FORMAT_R32_TYPELESS;u.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;
    u.Buffer.NumElements=bytes/4;u.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_RAW;
    dev->CreateUnorderedAccessView(dest.Get(),nullptr,&u,cpu(views.Get(),5));
    ComputeCommands cmd(dev,gp.Get(),D3D12_COMMAND_LIST_TYPE_DIRECT);
    UINT64 frequency=0;
    check(cmd.queue->GetTimestampFrequency(&frequency),"timestamp frequency");
    D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_TIMESTAMP,2,0};
    ComPtr<ID3D12QueryHeap> timestamps;
    check(dev->CreateQueryHeap(&query_desc,IID_PPV_ARGS(&timestamps)),"timestamp heap");
    auto timing=make_buffer(dev,2*sizeof(UINT64),D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    auto* list=cmd.list.Get();ID3D12DescriptorHeap* heaps[]{views.Get()};list->SetDescriptorHeaps(1,heaps);
    D3D12_RECT right_half{width,0,2*width,2*height},bottom_half{0,height,2*width,2*height};
    list->ClearRenderTargetView(cpu(rtvs.Get(),0),left.data(),0,nullptr);
    list->ClearRenderTargetView(cpu(rtvs.Get(),0),right.data(),1,&right_half);
    list->ClearRenderTargetView(cpu(rtvs.Get(),1),first.data(),0,nullptr);
    list->ClearRenderTargetView(cpu(rtvs.Get(),1),last.data(),1,&bottom_half);
    for(unsigned i=0;i<2;++i){
        transition(list,surfaces[i].Get(),D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        list->ClearRenderTargetView(cpu(rtvs.Get(),2+i),sentinel.data(),0,nullptr);
    }
    const auto targets=cpu(rtvs.Get(),2);list->OMSetRenderTargets(2,&targets,TRUE,nullptr);
    D3D12_VIEWPORT viewport{0,0,width,height,0,1};D3D12_RECT scissor{0,0,width-1,height};
    list->RSSetViewports(1,&viewport);list->RSSetScissorRects(1,&scissor);list->SetGraphicsRootSignature(graphics.Get());
    list->SetGraphicsRootDescriptorTable(0,gpu(0));list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->SetGraphicsRoot32BitConstant(1,0,0);
    list->EndQuery(timestamps.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
    list->DrawInstanced(3,1,0,0);
    list->EndQuery(timestamps.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);
    list->ResolveQueryData(timestamps.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,timing.Get(),0);
    for(unsigned i=2;i<4;++i)transition(list,surfaces[i].Get(),D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->SetPipelineState(cp.Get());list->SetComputeRootSignature(compute.Get());list->SetComputeRootDescriptorTable(0,gpu(3));
    list->Dispatch(1,1,1);
    transition(list,dest.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyResource(readback.Get(),dest.Get());cmd.submit_and_wait(dev);
    D3D12_RANGE range{0,bytes};check(readback->Map(0,&range,&data),"map compute result");
    for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x){
        Color expected=x<width/2?left:right;expected[3]=y<height/2?first[0]:last[0];
        if(x==width-1)expected=sentinel;
        auto actual=static_cast<const Color*>(data)[y*width+x];
        if(actual!=expected)throw std::runtime_error("sampled MRT mismatch at "+std::to_string(x)+","+std::to_string(y));
    }
    D3D12_RANGE empty{};readback->Unmap(0,&empty);
    D3D12_RANGE time_range{0,2*sizeof(UINT64)};
    check(timing->Map(0,&time_range,&data),"read timestamps");
    const auto* ticks=static_cast<const UINT64*>(data);
    if(!frequency || ticks[1]<ticks[0]) throw std::runtime_error("invalid GPU timestamps");
    std::printf("pso_ms=%.6f draw_gpu_us=%.6f\n",pso_ms,1e6*(ticks[1]-ticks[0])/frequency);
    timing->Unmap(0,&empty);
}
}

int wmain(int argc,wchar_t** argv) {
    try {
        if(argc==3 && std::wstring_view(argv[1])==L"--write-fixture") {
            write_fixture(argv[2]);
            return 0;
        }
        if(argc<4 || argc>6)return 2;
        if(GetEnvironmentVariableW(L"X2W_TEST_DEBUG_LAYER",nullptr,0)){
            ComPtr<ID3D12Debug> debug;
            check(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),"debug layer");debug->EnableDebugLayer();
        }
        bool unmatched=false;
        auto device=create_device(std::wstring_view(argv[1])==L"warp",unmatched);
        if(!device)return unmatched?1:77;
        D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_0};
        if(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,&model,sizeof(model))) ||
            model.HighestShaderModel<D3D_SHADER_MODEL_6_0)return 77;
        auto code=packet();
        auto shader=x2w::translate_gfx1013_sampled_pixel(code);
        if(shader.bindings!="srv 0 buf table 0 96\nsrv 1 tex2d table 0 32\nsampler 0 0 12582912 0 0\nsrv 2 tex2d table 0 64\n")
            throw std::runtime_error("unexpected bindings");
        const auto require_refusal=[](const std::string& bad,const char* diagnostic) {
            try { x2w::translate_gfx1013_sampled_pixel(bad); }
            catch(const std::runtime_error& e) {
                if(std::strstr(e.what(),diagnostic))return;
                throw;
            }
            throw std::runtime_error("accepted invalid sampled program");
        };
        require_refusal(code.substr(0,code.size()-4),"missing endpgm");
        require_refusal(packet(0),"invalid image_sample_lz");
        require_refusal(packet(3),"undefined float VGPR");
        require_refusal(packet(7,true,0x04004000),"secondary sampler bias");
        require_refusal(packet(7,true,0x04001000),"primary sampler bias");
        require_refusal(packet(7,false,0x04003f80),"primary sampler bias");
        x2w::ShaderDxilCompiler compiler(argv[2]);
        auto object=[&](int index,const std::string& hlsl,const wchar_t* profile){
            if(argc<=index)return compiler.compile(hlsl,profile,argv[3]).dxil;
            std::ifstream input(argv[index],std::ios::binary);
            if(!input)throw std::runtime_error("external stage file missing");
            return std::vector<unsigned char>(std::istreambuf_iterator<char>(input),{});
        };
        auto pixel=object(4,shader.hlsl,L"ps_6_0");
        auto vertex=object(5,x2w::translate_gfx1013_vertex_id(vertex_packet()),L"vs_6_0");
        run(device.Get(),compiler,argv[3],vertex,pixel);
        if(argc==4) {
            const auto biased=x2w::translate_gfx1013_sampled_pixel(packet(7,true,0x04003f80));
            if(biased.hlsl.find("-128.0/256.0") == std::string::npos ||
               biased.bindings.find("sampler 0 0 12582912 67108864 0") == std::string::npos)
                throw std::runtime_error("implicit sampler bias or normalized manifest differs");
            run(device.Get(),compiler,argv[3],vertex,compiler.compile(biased.hlsl,L"ps_6_0",argv[3]).dxil);
        }
        std::puts("112 covered components and 16 uncovered sentinel components matched after compute read");
    } catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
