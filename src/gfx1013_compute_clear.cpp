#include "gfx1013_compute_clear.h"
#include "shader_compute_request.h"
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <tuple>

namespace fs = std::filesystem;
namespace {
using Bytes = std::string;
template<class T> T read(const Bytes& data,size_t offset) {
    if(offset>data.size() || sizeof(T)>data.size()-offset) throw std::runtime_error("truncated compute request");
    T value;
    std::memcpy(&value,data.data()+offset,sizeof(value));
    return value;
}
enum class Kind { unknown, zero, table, descriptor, tg_size, wave_base, low_count, bin };
struct Value {
    Kind kind{};
    unsigned parameter{},offset{};
};
}

x2w::ComputeClear x2w::replay_group_clear(const fs::path& marker,unsigned entry_offset) {
    const auto request=read_compute_request(marker);
    const auto& code=request.code;
    const auto& root=request.root;
    const auto& sa=request.sa;
    if(entry_offset%4) throw std::runtime_error("unaligned entry offset");
    const unsigned xy=read<uint32_t>(sa,12),x=xy&65535,y=xy>>16,z=read<uint32_t>(sa,16)&65535;
    if(!x || !y || !z || uint64_t(x)*y*z!=256 || read<uint32_t>(sa,24)!=((6u<<1)|(1u<<10)))
        throw std::runtime_error("unsupported group-clear entry layout");
    Microsoft::WRL::ComPtr<ID3D12VersionedRootSignatureDeserializer> signature;
    if(FAILED(D3D12CreateVersionedRootSignatureDeserializer(root.data(),root.size(),IID_PPV_ARGS(&signature))))
        throw std::runtime_error("invalid compute root signature");
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* layout=nullptr;
    if(FAILED(signature->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1,&layout)) ||
       layout->Desc_1_1.NumParameters!=2) throw std::runtime_error("expected two descriptor tables");
    std::array<Value,128> scalar{};
    std::array<Value,256> vector{};
    // A descriptor occupies four SGPRs. Updating any component makes the old base unusable.
    auto write_scalar=[&](unsigned reg,Value value) {
        auto& base=scalar[reg & ~3u];
        if(base.kind==Kind::descriptor) base={};
        scalar[reg]=value;
    };
    for(unsigned i=0;i<2;++i) {
        if(layout->Desc_1_1.pParameters[i].ParameterType!=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
            throw std::runtime_error("compute parameter is not a descriptor table");
        scalar[4+i]={Kind::table,i};
    }
    scalar[6]={Kind::tg_size};
    unsigned pc=entry_offset;
    auto word=[&](){ auto v=read<uint32_t>(code,pc);pc+=4;return v; };
    ComputeClear result;
    std::vector<Value> stores;
    for(;;) {
        unsigned at=pc;
        uint32_t op=word();
        auto require=[&](bool ok){if(!ok) throw std::runtime_error("unsupported group-clear operand or instruction at byte "+std::to_string(at));};
        if(op==0xbf810000) break;
        if((op>>16)==0xbfa0 || (op>>16)==0xbfa1 || op==0xbf8cc07f) continue; // prefetch, clause, wait for scalar loads
        if((op&0xff80ff00)==0xbe800300) {
            unsigned source=op&255;
            require(((op>>16)&127)<102 && (source==128 || source<102));
            require(source==128 || (scalar[source].kind!=Kind::descriptor && scalar[source].kind!=Kind::unknown));
            write_scalar((op>>16)&127,source==128?Value{Kind::zero}:scalar[source]);
        } else if((op&~0x1fffu)==0xf4080000) { // s_load_dwordx4
            unsigned base=(op&63)*2,dst=(op>>6)&127;
            auto imm=word();
            require((imm&0xfff00000)==0xfa000000 && dst+3<102 && !(dst%4) && scalar[base].kind==Kind::table && scalar[base+1].kind==Kind::zero);
            auto table=scalar[base];
            std::fill_n(scalar.begin()+dst,4,Value{});
            scalar[dst]={Kind::descriptor,table.parameter,imm&0xfffff};
        } else if((op&0xff80ff00)==0x8700ff00) { // s_and_b32 with literal wave-base mask
            require(((op>>16)&127)<102 && (op&255)<102 && scalar[op&255].kind==Kind::tg_size && word()==0xfc0);
            write_scalar((op>>16)&127,{Kind::wave_base});
        } else if((op&0xffffff00)==0xd7000000) {
            unsigned dst=op&255;
            require(dst<255 && word()==0x10080); // v_lshrrev_b64 with zero shift and zero source
            vector[dst]=vector[dst+1]={Kind::zero};
        } else if((op>>8)==0xd76500 || (op>>8)==0xd76600) {
            auto operands=word();
            require((operands&0xfffc01ff)==193);
            unsigned source=(operands>>9)&511;
            bool low=(op>>8)==0xd76500;
            require(low ? source<128 && scalar[source].kind==Kind::wave_base :
                    source>=256 && vector[source-256].kind==Kind::low_count);
            vector[op&255]={low?Kind::low_count:Kind::bin};
        } else if(op==0xe2102000) { // buffer_store_format_d16_x, idxen only, offset zero
            auto args=word();
            require((args&0xffe00000)==0x80000000);
            auto resource=scalar[((args>>16)&31)*4];
            require(vector[args&255].kind==Kind::bin && vector[(args>>8)&255].kind==Kind::zero &&
                    resource.kind==Kind::descriptor && !(resource.offset%32));
            stores.push_back(resource);
        } else require(false);
    }
    if(stores.empty()) throw std::runtime_error("compute program has no clears");
    std::sort(stores.begin(),stores.end(),[](Value a,Value b){return std::tie(a.parameter,a.offset)<std::tie(b.parameter,b.offset);});
    for(auto value:stores) {
        if(!result.bindings.empty() && value.parameter==result.bindings.back().parameter && value.offset==result.bindings.back().byte_offset) continue;
        unsigned reg=static_cast<unsigned>(result.bindings.size());
        result.bindings.push_back({value.parameter,value.offset});
        auto id=std::to_string(reg);
        result.hlsl+="RWBuffer<uint> out"+id+" : register(u"+id+");\n";
    }
    result.hlsl+="[numthreads("+std::to_string(x)+","+std::to_string(y)+","+std::to_string(z)+")]\nvoid main(uint bin:SV_GroupIndex){\n";
    for(unsigned reg=0;reg<result.bindings.size();++reg) result.hlsl+="out"+std::to_string(reg)+"[bin]=0;\n";
    result.hlsl+="}\n";
    return result;
}
