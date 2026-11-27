#include "gfx1013_sampled_pixel.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {
using Word = uint32_t;
using std::string;
enum class Type { undefined, constant, table, descriptor_word, value, primitive_mask };
struct Scalar {
    Type type{};
    Word number{}, offset{};
    string value;
};
struct Vector {
    string value;
    std::array<string,2> halves{};
    int interpolation = -1;
};
struct Binding {
    Word table, offset;
    bool texture;
};

class PixelProgram {
    std::string_view code;
    size_t pc{}, start{};
    std::optional<Word> instruction_literal;
    unsigned temporary{};
    std::array<Scalar,102> scalars{};
    std::array<Vector,256> vectors{};
    std::vector<Binding> resources;
    std::vector<std::array<Word,4>> samplers;
    bool primitive_mask=false;
    std::array<bool,2> exported{};
    bool done=false;
    string declarations, body, bindings;

    [[noreturn]] void fail(const char* why) const {
        throw std::runtime_error(string(why)+" at pixel byte "+std::to_string(start));
    }
    void require(bool ok,const char* why) const { if(!ok) fail(why); }
    Word word() {
        require(code.size()-pc>=4,"truncated instruction");
        Word v; std::memcpy(&v,code.data()+pc,4); pc+=4; return v;
    }
    string symbol(const char* type,const string& expression) {
        auto name="p"+std::to_string(temporary++);
        body+="    "+string(type)+" "+name+" = "+expression+";\n";
        return name;
    }
    Scalar source(Word index) {
        if(index<scalars.size()) return scalars[index];
        Word bits{};
        if(index>=128 && index<=192) bits=index-128;
        else if(index>=193 && index<=208) bits=192u-index;
        else if(index>=240 && index<=247) {
            constexpr Word floats[]{0x3f000000,0xbf000000,0x3f800000,0xbf800000,0x40000000,0xc0000000,0x40800000,0xc0800000};
            bits=floats[index-240];
        } else if(index==255) {
            // Every literal selector in the instruction names the same trailing dword.
            if(!instruction_literal) instruction_literal=word();
            bits=*instruction_literal;
        }
        else fail("unsupported scalar operand");
        return {Type::constant,bits};
    }
    static string literal(Word bits) {return std::to_string(bits)+"u";}
    string as_float(Word index) {
        if(index>=256) {
            const auto& v=vectors[index-256];
            require(!v.value.empty(),"undefined float VGPR");
            return v.value;
        }
        auto s=source(index);
        if(s.type==Type::value) return "asfloat("+s.value+")";
        require(s.type==Type::constant,"expected float operand");
        return "asfloat("+literal(s.number)+")";
    }
    void put(Word reg,const string& expr) {
        vectors[reg]=Vector{symbol("float",expr)};
    }
    string resource(Word base,unsigned words,bool texture) {
        require(base+words<=scalars.size(),"invalid descriptor SGPR range");
        auto first=scalars[base];
        require(first.type==Type::descriptor_word && first.offset%32==0,"unresolved descriptor base");
        for(unsigned i=1;i<words;++i) {
            auto s=scalars[base+i];
            require(s.type==first.type && s.number==first.number && s.offset==first.offset+4*i,"incomplete descriptor");
        }
        unsigned i=0;
        for(auto b:resources) {
            if(b.table==first.number && b.offset==first.offset) {
                require(b.texture==texture,"conflicting descriptor type");
                return "r"+std::to_string(i);
            }
            ++i;
        }
        auto n=std::to_string(i);
        declarations+=(texture?"Texture2D<float4>":"ByteAddressBuffer")+string(" r")+n+" : register(t"+n+");\n";
        bindings+="srv "+n+(texture?" tex2d table ":" buf table ")+std::to_string(first.number)+" "+std::to_string(first.offset)+"\n";
        resources.push_back({first.number,first.offset,texture});
        return "r"+n;
    }
    std::array<Word,4> sampler_words(Word base) {
        require(base+4<=scalars.size(),"invalid sampler SGPR range");
        std::array<Word,4> bits{};
        for(unsigned i=0;i<4;++i) {
            require(scalars[base+i].type==Type::constant,"unresolved sampler");
            bits[i]=scalars[base+i].number;
        }
        return bits;
    }
    string sampler(const std::array<Word,4>& bits) {
        unsigned i=0;
        for(const auto& s:samplers) {if(s==bits) return "q"+std::to_string(i); ++i;}
        auto n=std::to_string(i);
        declarations+="SamplerState q"+n+" : register(s"+n+");\n";
        bindings+="sampler "+n;
        for(auto b:bits) bindings+=" "+std::to_string(b);
        bindings+="\n";
        samplers.push_back(bits);
        return "q"+n;
    }
    void scalar_load(Word op) {
        auto base=(op&63)*2, dst=(op>>6)&127;
        auto imm=word();
        require((imm&0xfff00000)==0xfa000000,"unsupported scalar byte offset");
        unsigned off=imm&0xfffff;
        if((op&~0x1fffu)==0xf4240000) {
            require(dst+2<=scalars.size() && off%4==0,"invalid buffer load");
            auto r=resource(base,4,false);
            for(unsigned i=0;i<2;++i)
                scalars[dst+i]={Type::value,0,0,symbol("uint",r+".Load("+literal(off+4*i)+")")};
            return;
        }
        unsigned count=(op&~0x1fffu)==0xf4080000 ? 4 : 16;
        require(base+1<scalars.size() && dst+count<=scalars.size(),"invalid descriptor load");
        auto ptr=scalars[base],hi=scalars[base+1];
        require(ptr.type==Type::table && hi.type==Type::constant && hi.number==0,"expected a descriptor table");
        for(unsigned i=0;i<count;++i) scalars[dst+i]={Type::descriptor_word,ptr.number,off+4*i};
    }
    void sample(Word op) {
        auto args=word(), addr=args&255, dst=(args>>8)&255, mask=(op>>8)&15;
        require(!(args&0xfc000000) && mask && addr<255,"invalid image_sample_lz");
        auto r=resource(((args>>16)&31)*4,8,true);
        auto bits=sampler_words(((args>>21)&31)*4);
        const bool implicit = (op & 0xfffff0ff) == 0xf0800008;
        const int bias = int((bits[2] & 0x3fff) ^ 0x2000) - 0x2000;
        const unsigned mip_filter = (bits[2] >> 26) & 3;
        require(!(bits[2] & 0xfc000), "unsupported secondary sampler bias");
        require(!implicit || mip_filter == 1 || mip_filter == 2, "unsupported implicit mip filter");
        // The primary s6.8 field must fit the desktop SampleBias range [-16, 15.99].
        require(!bias || (implicit && bias >= -4096 && bias <= 4093),
                "unsupported primary sampler bias");
        require((bits[1]&0xfff) <= ((bits[1]>>12)&0xfff), "inverted sampler LOD limits");
        bits[2] &= ~0x3fffu;
        auto q=sampler(bits);
        auto uv="float2("+as_float(256+addr)+","+as_float(257+addr)+")";
        string call;
        if (bias) call=r+".SampleBias("+q+","+uv+","+std::to_string(bias)+".0/256.0)";
        else if (implicit) call=r+".Sample("+q+","+uv+")";
        else call=r+".SampleLevel("+q+","+uv+",0)";
        auto value=symbol("float4",call);
        for(unsigned bit=0;bit<4;++bit) if(mask&(1u<<bit)) {
            require(dst<256,"sample destination overflows VGPR file");
            vectors[dst++]=Vector{value+"."+"xyzw"[bit]};
        }
    }
    void export_mrt(Word op) {
        auto args=word();
        unsigned mrt=(op>>4)&63;
        require(mrt<2 && !exported[mrt],"unexpected render target export");
        require((op&15)==(mrt?3u:15u) && (args&0xffff0000)==0x80800000,"unsupported compressed export mask");
        unsigned channels=mrt?1:3;
        string value;
        for(unsigned i=0;i<channels;++i) {
            auto& v=vectors[(args>>(8*(i/2)))&255];
            auto half=v.halves[i%2];
            require(!half.empty(),"undefined live render target channel");
            if(i) value+=",";
            value+="f16tof32("+half+")";
        }
        body+="    o.c"+std::to_string(mrt)+" = "+(mrt?value:"float3("+value+")")+";\n";
        exported[mrt]=true;
        done=(op&0x800)!=0;
    }
    void instruction(Word op) {
        auto dst=(op>>16)&127;
        if((op>>16)==0xbfa0 || (op>>16)==0xbfa1 || (op>>16)==0xbf8c) return; // prefetch, clause and waitcnt
        if((op&0xff80ff00)==0xbe800300 || (op&0xff80ff00)==0xbe800400) {
            auto s=source(op&255);
            if(dst==124) {
                require(((op>>8)&255)==3 && s.type==Type::primitive_mask,"unsupported m0 write");
                primitive_mask=true;return;
            }
            require(dst<scalars.size() && s.type!=Type::undefined,"invalid scalar move");
            if(((op>>8)&255)==4) {
                const unsigned selector=op&255;
                require(dst+1<scalars.size(),"invalid scalar pair destination");
                if(selector<scalars.size()-1) {
                    require(scalars[selector+1].type!=Type::undefined,"undefined scalar pair source");
                    scalars[dst+1]=scalars[selector+1];
                } else if(selector==255 && !(s.number>>31)) {
                    scalars[dst+1]={Type::constant,0};
                } else {
                    require(selector>=128 && selector<=208,"unsupported 64-bit scalar constant");
                    scalars[dst+1]={Type::constant,selector<=192?0u:0xffffffffu};
                }
            }
            scalars[dst]=s;
        } else if((op&0xff800000)==0xb0000000) {
            require(dst<scalars.size(),"invalid movk destination");
            scalars[dst]={Type::constant,static_cast<Word>(static_cast<int16_t>(op))};
        } else if((op&0xff800000)==0x92000000 || (op&0xff800000)==0x92800000) {
            auto a=source(op&255),b=source((op>>8)&255);
            require(dst<scalars.size() && a.type==Type::constant && b.type==Type::constant,"unresolved bfm operands");
            if (op&0x800000) {
                require(!(dst&1) && dst+1<scalars.size(),"invalid bfm scalar pair");
                const uint64_t mask=((uint64_t{1}<<(a.number&63))-1)<<(b.number&63);
                scalars[dst]={Type::constant,static_cast<Word>(mask)};
                scalars[dst+1]={Type::constant,static_cast<Word>(mask>>32)};
            } else scalars[dst]={Type::constant,((1u<<(a.number&31))-1)<<(b.number&31)};
        } else if((op&~0x1fffu)==0xf4080000 || (op&~0x1fffu)==0xf4100000 || (op&~0x1fffu)==0xf4240000) {
            scalar_load(op);
        } else if((op&0xfc000000)==0xc8000000) {
            unsigned d=(op>>18)&255, attr=(op>>10)&63, component=(op>>8)&3, phase=(op>>16)&3;
            require(primitive_mask && attr==0 && phase<2 && (op&255)==phase,"unsupported interpolation interface");
            require(vectors[phase].value.empty() && vectors[phase].interpolation==-1 &&
                    vectors[phase].halves[0].empty() && vectors[phase].halves[1].empty(),"overwritten IJ input");
            if(!phase) { vectors[d]={}; vectors[d].interpolation=component; }
            else {
                require(vectors[d].interpolation==int(component),"interpolation phase mismatch");
                put(d,"param0."+string(1,"xyzw"[component]));
            }
        } else if((op&0xfffff0ff)==0xf09c0008 || (op&0xfffff0ff)==0xf0800008) {
            sample(op);
        } else if((op&0xffffff00)==0xd54b0000 || (op&0xffffff00)==0xcc214000 || (op&0xffffff00)==0xcc224000) {
            auto args=word(), d=op&255;
            auto a=as_float(args&511), b=as_float((args>>9)&511);
            if((op>>16)==0xd54b) {
                require(!(args&0xf8000000),"unsupported FMA modifiers");
                auto c=as_float((args>>18)&511);
                put(d,"mad("+a+","+b+","+c+")");
            } else {
                require((args&0xfffc0000)==0x82000000,"expected negated zero half addend");
                unsigned half=(op>>16)==0xcc22;
                auto& v=vectors[d];
                if(!v.value.empty()) {
                    v.halves[0]="(asuint("+v.value+") & 65535u)";
                    v.halves[1]="(asuint("+v.value+") >> 16)";
                }
                v.value.clear();v.interpolation=-1;
                v.halves[half]=symbol("uint","f32tof16(mad("+a+","+b+",asfloat(0x80000000u)))");
            }
        } else if(op>>25==3 || op>>25==5 || op>>25==43) {
            auto d=(op>>17)&255;
            auto a=as_float(op&511), b=as_float(256+((op>>9)&255));
            if(op>>25==43) put(d,"mad("+a+","+b+","+as_float(256+d)+")");
            else put(d,op>>25==3?a+"+"+b:b+"-"+a);
        } else if(op==0xf800040f || op==0xf8001c13) export_mrt(op);
        else fail("unsupported sampled pixel instruction");
    }
public:
    explicit PixelProgram(std::string_view bytes) : code(bytes) {}
    x2w::SampledPixel translate() {
        scalars[8]={Type::table,0};scalars[9]={Type::table,1};scalars[10]={Type::primitive_mask};
        while(pc<code.size()) {
            start=pc;instruction_literal.reset();auto op=word();
            if(op==0xbf810000) {
                require(done && exported[0] && exported[1] && pc==code.size(),"incomplete pixel exports or trailing instructions");
                return {declarations+"struct Out {float3 c0:SV_Target0; float c1:SV_Target1;};\n"
                       "Out main(float4 param0:PARAM0){\n    Out o;\n"+body+"    return o;\n}\n",bindings};
            }
            require(!done,"instruction after final export");
            instruction(op);
        }
        fail("missing endpgm");
    }
};
}

x2w::SampledPixel x2w::translate_gfx1013_sampled_pixel(std::string_view code) {
    return PixelProgram(code).translate();
}
