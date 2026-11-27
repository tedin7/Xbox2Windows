// RDNA2 merged-wave mesh entry, GS_ALLOC_REQ and EXP encoding. The slot-57
// exchange proof is also described in tools/startup/gfx2hlsl.py:mesh_skeleton.
// The desktop MS writes lane-indexed vertices and triangle indices directly.
NativeComputeShader lower_mesh(const string &code, const MeshStageLayout &layout,
                               const GuestShaderStage *reflected = nullptr,
                               const D3D12_ROOT_SIGNATURE_DESC1 &root = {}) {
    if (!layout.threads || layout.threads > 128 || !layout.max_vertices || layout.max_vertices > 256 ||
        !layout.max_primitives || layout.max_primitives > 256 || layout.topology != 2 ||
        code.empty() || code.size() % 4)
        refuse(0,"unsupported mesh group, output dimensions or topology");
    Program program;
    // This subset has a single forward path after the proven exchange.
    // The shared ALU's SGPR provenance lookup uses the current state.
    program.graphics_wave64=!layout.wave32;
    program.before.emplace_back();
    for (auto &s : program.before.front()) s.type=Symbol::Type::unbound;
    Emit emitter{{},program,root};
    DirectCompute alu{emitter};
    GuestShaderStage fixed_outputs{};
    fixed_outputs.output_signature.push_back({"SV_Position",0,1,3,0,15,15});
    GraphicsStage graphics{reflected ? *reflected : fixed_outputs,alu,false};
    auto &ir = alu.ir;
    ir.mesh_vertices = layout.max_vertices;
    ir.mesh_primitives = layout.max_primitives;
    auto thread = ir.invoke("i32","flattenedThreadIdInGroup.i32",96,{},"nounwind readnone");
    const unsigned width = layout.wave32 ? 32 : 64;
    const auto wave_lane = ir.binary("and",thread,to_string(width-1));
    const auto wave_number = ir.binary("lshr",thread,layout.wave32 ? "5" : "6");
    alu.graphics_input = [&](unsigned r) -> string {
        refuse(alu.graphics_pc,"unproven mesh entry register " + to_string(r));
    };
    alu.after_graphics_write = [&](unsigned r) {
        if (r<106) program.before.front()[r].type=alu.graphics_registers.at(r).empty() ?
            Symbol::Type::unbound : Symbol::Type::unknown;
    };
    auto &state=program.before.front();
    auto inputs=layout.root_inputs;
    auto heap=layout.resource_heap_sgpr;
    if (reflected) {
        for (const auto &input : reflected->inputs) {
            if (input.type==5) {
                if (heap) refuse(0,"duplicate mesh resource heap");
                heap=reflected->physical_user_sgpr(input.user_data);
            }
            if (input.type==1) inputs.push_back({reflected->physical_user_sgpr(input.user_data),input.root_parameter,0,true});
            if (input.type==0 || input.type==3)
                inputs.push_back({reflected->physical_user_sgpr(input.user_data),input.root_parameter,
                                  input.type==0 ? input.sgpr_count : 0});
        }
    }
    for (const auto &input : inputs) {
        const auto at=input.sgpr, param=input.root_parameter, count=input.constant_count;
        if (at<8 || at>=106 || count>106-at || param>=16)
            refuse(0,"mesh root input outside user SGPR range");
        if (!reflected || root.NumParameters) {
            if (param>=root.NumParameters) refuse(0,"missing mesh root parameter");
            const auto &binding=root.pParameters[param];
            if (input.root_address ? binding.ParameterType!=D3D12_ROOT_PARAMETER_TYPE_SRV :
                count ? (binding.ParameterType!=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS ||
                         binding.Constants.Num32BitValues!=count) :
                        binding.ParameterType!=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
                refuse(0,"mesh root parameter kind or size disagrees with entry");
        }
        if (input.root_address) {
            if (count || at+1>=106 || state[at].type!=Symbol::Type::unbound || state[at+1].type!=Symbol::Type::unbound)
                refuse(0,"invalid mesh root address pair");
            state[at]={Symbol::Type::root_address,param,0};
            state[at+1]={Symbol::Type::root_address,param,1};
        } else if (!count) {
            const Symbol table{Symbol::Type::table,param};
            if (state[at].type!=Symbol::Type::unbound && !(state[at]==table))
                refuse(0,"overlapping mesh entry table");
            state[at]=table;
        } else {
            for (unsigned j=0;j<count;++j) {
                if (state[at+j].type!=Symbol::Type::unbound) refuse(0,"overlapping mesh entry constants");
                state[at+j]={};
            }
            emitter.result.constants.push_back({param,at,count});
        }
    }
    if (heap) {
        if (*heap<8 || *heap>=106 || state[*heap].type!=Symbol::Type::unbound) refuse(0,"invalid mesh resource heap SGPR");
        state[*heap]={Symbol::Type::table,compute_resource_heap};
    }
    alu.initialize_constants();
    struct RootBits { unsigned param, dword, mask, shift; };
    std::map<string,RootBits> root_values;
    std::map<unsigned,RootBits> heap_fetches;
    for (const auto &block : emitter.result.constants)
        for (unsigned j=0;j<block.count;++j)
            root_values.emplace(alu.read(block.first_sgpr+j),RootBits{block.root_parameter,j,~0u,0});
    std::set<string> uniform_values;
    string exec_value[2], wave_active, lane_active;
    std::map<string,std::pair<string,string>> active_uniform, mask_uniform;
    auto uniform=[&](const string &v) -> string {
        if (v.empty()) return {};
        if (v.find_first_not_of("0123456789-")==string::npos || uniform_values.count(v)) return v;
        auto it=active_uniform.find(v); return it==active_uniform.end() || it->second.first!=lane_active ? string{} : it->second.second;
    };
    auto remember_scalar=[&](unsigned r) {
        const auto it=alu.graphics_registers.find(r);
        if (it!=alu.graphics_registers.end() && !it->second.empty()) uniform_values.insert(it->second);
    };
    std::set<string> wave_bases, low_group, low_lane, whole_lane, linear_threads;
    std::set<unsigned> partial_writes;
    bool full_exec = true, allocated = false, position = false, indices = false;
    std::set<unsigned> parameter_exports;
    std::optional<Word> allocation_word;
    unsigned vertices = 0, primitives = 0;
    string vertex_count, primitive_count, export_limit;
    std::map<string,unsigned> upper_bound;
    auto bound=[&](const Inst &op,unsigned source) {
        if (auto c=DirectCompute::constant32(op,source)) return *c;
        auto it=alu.graphics_registers.find(source);
        if (it!=alu.graphics_registers.end() && upper_bound.count(it->second)) return upper_bound.at(it->second);
        return ~0u;
    };
    string exchanged_group;
    if (layout.group_stride) {
        const auto group = ir.invoke("i32","groupId.i32",94,{{"i32","0"}},"nounwind readnone");
        exchanged_group = ir.binary("mul",group,to_string(layout.group_stride));
    }
    unsigned scope = 0;
    auto guarded = [&](const string &limit, auto body) {
        auto tag = to_string(scope++);
        auto pred = alu.cmp("ult",thread,limit);
        ir.body << "  br i1 " << pred << ", label %mesh_write" << tag << ", label %mesh_next" << tag
                << "\nmesh_write" << tag << ":\n";
        body();
        ir.body << "  br label %mesh_next" << tag << "\nmesh_next" << tag << ":\n";
    };
    // Scalar forward if/else only. Keep both predecessor register files so
    // a value live after the join is selected from the edge that wrote it.
    struct Edge {
        State symbols;
        decltype(alu.graphics_registers) regs;
        string group;
        std::set<unsigned> partial;
    };
    struct Diamond {
        unsigned start, alternate, join;
        Edge entry, first;
    };
    std::optional<Diamond> branch;
    unsigned lane_else=0;
    constexpr unsigned vcc_words=0x70000000, vcc_nonempty=vcc_words+2;
    std::map<string,string> mask_parent;
    auto covered_by=[&](string current,const string &ancestor) {
        if (ancestor.empty()) return true;
        while (current!=ancestor) {
            auto it=mask_parent.find(current);
            if (it==mask_parent.end()) return false;
            current=it->second;
        }
        return true;
    };
    auto terminate_arm=[&](unsigned start,bool first) {
        const string edge=(first ? "mesh_then_end" : "mesh_else_end")+to_string(start);
        ir.body<<"  br label %"<<edge<<"\n"<<edge<<":\n  br label %mesh_join"<<start<<"\n";
    };
    struct MaskState { bool full; string lane,wave,lo,hi; decltype(alu.graphics_registers) regs; std::set<unsigned> partial; };
    std::map<unsigned,MaskState> saved_exec;
    std::map<string,Edge> before_narrowing;
    auto saved_mask=[&](unsigned source)->const MaskState * {
        if (source>104 || (source&1)) return nullptr;
        const auto &lo=state[source], &hi=state[source+1];
        if (lo.type!=Symbol::Type::lane_mask || hi.type!=lo.type || lo.offset || hi.offset!=1 || lo.value!=hi.value)
            return nullptr;
        const auto it=saved_exec.find(lo.value); return it==saved_exec.end() ? nullptr : &it->second;
    };
    auto pure_lane_arm=[&](unsigned begin,unsigned end) {
        for (auto at=begin;at<end;) {
            const auto inst=decode(code,at,true,true);
            const bool vector=inst.dst>=256 && (inst.enc==Encoding::Vop1 || inst.enc==Encoding::Vop2 || inst.enc==Encoding::Vop3);
            const bool compare=inst.enc==Encoding::Vopc && inst.dst==106 && inst.name.substr(0,6)=="v_cmp_";
            const bool literal=inst.name=="s_mov_b32" && inst.dst<106 && DirectCompute::constant32(inst,inst.src[0]);
            if ((!vector && !compare && !literal && inst.name!="s_nop") || inst.next<=at || inst.next>end)
                return false;
            at=inst.next;
        }
        return true;
    };
    // Prove scalar dependencies when two mutually exclusive lane arms
    // gather uniform buffer records. Scalar registers execute per wave, so
    // the inverse arm may observe the first arm's writes unless it defines
    // them itself before first use.
    struct GatherArm { std::set<unsigned> writes, inputs; };
    auto gather_arm=[&](unsigned begin,unsigned end)->std::optional<GatherArm> {
        GatherArm arm;
        auto use=[&](unsigned r) { if (r<=107 && !arm.writes.count(r)) arm.inputs.insert(r); };
        for (unsigned at=begin;at<end;) {
            auto op=decode(code,at,true,true);
            if (op.next<=at || op.next>end || op.subword || op.dpp_control) return {};
            if (op.name=="s_add_u32" && op.dst<106) {
                use(op.src[0]); use(op.src[1]); arm.writes.insert(op.dst); arm.writes.insert(128);
            } else if (op.enc==Encoding::Smem && op.op==12 && op.dst+16<=106) {
                for (unsigned j=0;j<4;++j) use(op.src[0]+j);
                use(op.b>>25);
                for (unsigned j=0;j<16;++j) arm.writes.insert(op.dst+j);
            } else if (op.enc==Encoding::Vopc && op.name=="v_cmp_gt_i32" && op.dst<=106 && !(op.dst&1)) {
                use(op.src[0]); use(op.src[1]); arm.writes.insert(op.dst); arm.writes.insert(op.dst+1);
            } else if (op.name=="v_mov_b32" && op.dst>=256) use(op.src[0]);
            else if (op.name=="v_cndmask_b32" && op.dst>=256 && op.src[2]<=106) {
                use(op.src[0]); use(op.src[1]); use(op.src[2]); use(op.src[2]+1);
            } else if (op.name!="s_waitcnt") return {};
            at=op.next;
        }
        return arm;
    };
    std::set<unsigned> lane_scalar_clobbers;
    struct Skip { unsigned start,end; Edge entry; bool per_lane; string guard; };
    std::vector<Skip> skip_parents;
    std::optional<Skip> exec_skip;
    unsigned mask_generation=0x1000000;
    auto merge_edges=[&](const Edge &first, const string &l, const string &r) {
        auto phi=[&](const string &a,const string &b) {return ir.value("phi i32 [ "+a+", %"+l+" ], [ "+b+", %"+r+" ]");};
        std::map<unsigned,std::pair<Symbol,string>> masks;
        for (unsigned reg=0;reg<=104;reg+=2) {
            const auto bit=[&](const Symbol &s,const auto &regs) -> string {
                if ((s.type!=Symbol::Type::lane_mask && s.type!=Symbol::Type::uniform_mask) || s.offset) return {};
                auto it=regs.find(1024+s.value);return it==regs.end() ? string{} : it->second;
            };
            const auto a=bit(first.symbols[reg],first.regs), b=bit(state[reg],alu.graphics_registers);
            if (a.empty() || b.empty() || first.symbols[reg]==state[reg]) continue;
            if (!layout.wave32 && (first.symbols[reg+1].value!=first.symbols[reg].value ||
                state[reg+1].value!=state[reg].value || first.symbols[reg+1].offset!=1 || state[reg+1].offset!=1 ||
                first.symbols[reg+1].type!=first.symbols[reg].type || state[reg+1].type!=state[reg].type)) continue;
            const auto id=mask_generation++;
            masks.emplace(reg,std::make_pair(Symbol{Symbol::Type::lane_mask,id,0},a==b ? a : phi(a,b)));
        }
        for (const auto &[reg,v] : first.regs) alu.graphics_registers.try_emplace(reg);
        for (auto &[reg,v] : alu.graphics_registers) {
            const auto a=first.regs.find(reg);
            if (a==first.regs.end() || a->second.empty() || v.empty()) v.clear();
            else if (a->second!=v) {
                const auto av=uniform(a->second), bv=uniform(v);
                v=phi(a->second,v);
                if (!lane_else && !av.empty() && !bv.empty()) active_uniform[v]={lane_active,av==bv ? av : phi(av,bv)};
            }
        }
        for (unsigned j=0;j<state.size();++j) {
            if (!(first.symbols[j]==state[j]))
                state[j].type=j<106 && alu.graphics_registers.count(j) &&
                    !alu.graphics_registers.at(j).empty() ? Symbol::Type::unknown : Symbol::Type::unbound;
        }
        for (const auto &[r,entry] : masks) {
            state[r]=entry.first;
            if (!layout.wave32) {state[r+1]=entry.first;state[r+1].offset=1;}
            alu.graphics_registers[1024+entry.first.value]=entry.second;
        }
        if (exchanged_group!=first.group) exchanged_group.clear();
        partial_writes.insert(first.partial.begin(),first.partial.end());
    };
    auto save_edge=[&]() { return Edge{state,alu.graphics_registers,exchanged_group,partial_writes}; };
    struct Incoming {
        Edge edge;
        string label, lane, wave;
        bool full;
    };
    std::map<unsigned,std::vector<Incoming>> forward;
    bool live=true;
    unsigned next_edge=0;
    auto arrive=[&](unsigned target) {
        string label="mesh_edge"+to_string(next_edge++);
        forward[target].push_back({save_edge(),label,lane_active,wave_active,full_exec});
        return label;
    };
    auto restore=[&](const Edge &e) {
        state=e.symbols; alu.graphics_registers=e.regs; exchanged_group=e.group; partial_writes=e.partial;
    };
    unsigned pc = 0;
    while (pc < code.size()) {
        // All handled scalar branches are forward, so every predecessor is
        // available when we reach its target. Funnel arbitrary fan-in into
        // binary PHI blocks, preserving descriptor and mask provenance too.
        if (!live && !forward.count(pc)) {
            auto next=forward.lower_bound(pc);
            if (next==forward.end()) refuse(pc,"mesh forward branch has no reachable successor");
            pc=next->first;
        }
        if (forward.count(pc)) {
            if (live) ir.body<<"  br label %"<<arrive(pc)<<"\n";
            auto inputs=std::move(forward.at(pc)); forward.erase(pc);
            for (const auto &in : inputs)
                if (in.full!=full_exec || in.lane!=lane_active || in.wave!=wave_active)
                    refuse(pc,"mesh scalar branch merges different EXEC definitions");
            restore(inputs[0].edge);
            string current=inputs[0].label;
            ir.body<<current<<":\n";
            for (size_t j=1;j<inputs.size();++j) {
                const auto join="mesh_phi"+to_string(next_edge++);
                ir.body<<"  br label %"<<join<<"\n"<<inputs[j].label<<":\n  br label %"<<join<<"\n"<<join<<":\n";
                const auto first=save_edge();
                restore(inputs[j].edge);
                merge_edges(first,current,inputs[j].label);
                current=join;
            }
            live=true;
        }
        if (exec_skip && pc==exec_skip->end) {
            const auto tag=to_string(exec_skip->start);
            ir.body<<"  br label %mesh_done"<<tag<<"\nmesh_done"<<tag<<":\n  br label %mesh_resume"<<tag
                   <<"\nmesh_resume"<<tag<<":\n";
            if (exec_skip->per_lane) {
                // A SALU write was made by each active wave. Inactive lanes
                // skipped that invocation; deny later reads of changed scalar
                // definitions. Vector values keep their actual incoming edge.
                for (unsigned r=0;r<=128;++r) {
                    auto now=alu.graphics_registers.find(r), old=exec_skip->entry.regs.find(r);
                    if (now!=alu.graphics_registers.end() && (old==exec_skip->entry.regs.end() || now->second!=old->second)) {
                        now->second.clear();
                        if (r<128) state[r].type=Symbol::Type::unbound;
                    }
                }
            }
            merge_edges(exec_skip->entry,"mesh_empty"+tag,"mesh_done"+tag);
            exec_skip.reset();
            if (!skip_parents.empty()) {
                exec_skip=std::move(skip_parents.back());skip_parents.pop_back();
                if (pc==exec_skip->end) continue;
            }
        }
        if (branch && pc==branch->alternate) {
            if (pc==branch->join || lane_else) {
                branch->first=save_edge();
                terminate_arm(branch->start,true);
            }
            const auto &entry=branch->entry;
            state=entry.symbols;
            alu.graphics_registers=entry.regs;
            exchanged_group=entry.group;
            partial_writes=entry.partial;
            ir.body<<"\nmesh_else"<<branch->start<<":\n";
            if (lane_else) { pc=lane_else; continue; }
        }
        if (branch && pc==branch->join) {
            const auto &first=branch->first;
            terminate_arm(branch->start,false);
            ir.body<<"mesh_join"<<branch->start<<":\n";
            if (lane_else) {
                // A scalar literal in a nonempty wave may be written even
                // for its inactive lanes. Refuse later numeric use unless a
                // subsequent definition overwrites it. VCC is wave-wide too.
                for (unsigned r=0;r<=128;++r) {
                    const auto old=branch->entry.regs.find(r);
                    const auto differs=[&](const auto &regs) {
                        auto v=regs.find(r);
                        return v!=regs.end() && (old==branch->entry.regs.end() || v->second!=old->second);
                    };
                    if (differs(first.regs) || differs(alu.graphics_registers)) {
                        alu.graphics_registers[r].clear();
                        if (r<106) state[r].type=Symbol::Type::unbound;
                    }
                }
                lane_else=0;
            }
            merge_edges(first,"mesh_then_end"+to_string(branch->start),"mesh_else_end"+to_string(branch->start));
            for (auto r:lane_scalar_clobbers) {
                alu.write(r,{});
                if (r<state.size()) state[r].type=Symbol::Type::unbound;
            }
            if (lane_scalar_clobbers.count(106)) {
                alu.write(vcc_words,{}); alu.write(vcc_words+1,{}); alu.write(vcc_nonempty,{});
            }
            lane_scalar_clobbers.clear();
            branch.reset();
        }
        program.at[pc]=0;
        alu.graphics_pc = pc;
        const auto raw = word(code,pc);
        // All paths through this diamond rendezvous exactly once. Wave zero
        // writes its first lane's mesh-group identifier before the rendezvous,
        // the other waves read the same LDS word after it. Only v_readfirstlane
        // may use the result; v0 and the inactive lanes of v5 are not uniform.
        if (raw == 0xbefe0481 && code.size()-pc >= 64) {
            const Word save = word(code,pc+4), mask = word(code,pc+8);
            constexpr Word tail[]{0x0f000000,0xbf850007,0x2c0a0a87,0x7e000280,0xd8340008,0x500,
                                  0xbfa3ffe3,0xbf8a0000,0xbf820004,0xbf8a0000,0x7e000280,0xd8d80008,0x05000000};
            bool exchange = (save & 0xff80ffff) == 0xbe80047e && (mask & 0xff80ffff) == 0x8700ff03;
            for (unsigned k=0;k<std::size(tail);++k) exchange &= word(code,pc+12+4*k)==tail[k];
            const unsigned saved = (save>>16)&127, temp = (mask>>16)&127;
            if (exchange && !branch && !exec_skip && full_exec && !layout.wave32 && layout.group_stride && !alu.graphics_registers.count(3) &&
                !alu.graphics_registers.count(261) && saved<=104 && !(saved&1) && temp<106 &&
                temp!=saved && temp!=saved+1) {
                alu.write(saved,"1");
                alu.write(saved+1,"0");
                const auto merged = ir.binary("shl",wave_number,"24");
                alu.write(temp,merged);
                alu.write(128,ir.value("zext i1 "+alu.cmp("ne",merged,"0")+" to i32"));
                if (exchanged_group.empty()) refuse(pc,"mesh exchange lost the entry group VGPR");
                exchanged_group = ir.binary("lshr",exchanged_group,"7");
                alu.write(256,{});
                alu.write(261,{});
                full_exec=false;
                pc+=64;
                continue;
            }
        }
        // A wave-zero producer sends GS_ALLOC_REQ and shares its two count
        // SGPRs with the other waves through LDS. Verify the complete
        // sequence, including both rendezvous, every LDS offset and every
        // def-use edge before replacing it with group-shared DXIL.
        if (raw==0xbf8a0000 && pc+128<=code.size() && !allocated && !branch && !exec_skip && forward.empty()) {
            auto w=[&](unsigned n) { return word(code,pc+4*n); };
            const unsigned wave=w(1)&127, mask=w(3)&127, p=w(5)&255, v=w(7)&255;
            const unsigned data=(w(5)>>17)&255, addr=(w(6)>>17)&255, vdata=(w(7)>>17)&255;
            const unsigned temp=(w(15)>>16)&127, rp=(w(24)>>24)&255, rv=(w(26)>>24)&255;
            const unsigned raddr=(w(22)>>17)&255;
            const auto vector_mov=[](unsigned dst,unsigned src) { return 0x7e000200u|(dst<<17)|src; };
            bool match=w(1)==(0xbf008000u|wave) && w(2)==0xbf840010 && w(3)==(0xbefe0400u|mask) &&
                w(4)==0xbf880007 && w(5)==vector_mov(data,p) && w(6)==vector_mov(addr,128) &&
                w(7)==vector_mov(vdata,v) && data!=addr && addr!=vdata && vdata!=data &&
                w(8)==0xd8340000 && w(9)==((data<<8)|addr) && w(10)==0xd8340004 && w(11)==((vdata<<8)|addr) &&
                w(12)==0xbefe04c1 && w(13)==0xbfa3ffe3 && w(14)==0xbf8a0000 &&
                w(15)==(0x8f008c00u|(temp<<16)|p) && w(16)==(0x887c0000u|(v<<8)|temp) &&
                w(17)==0xbf900009 && w(18)==0xbf82000d && w(19)==0xbf8a0000 && w(20)==w(3) && w(21)==0xbf880009 &&
                w(22)==vector_mov(raddr,128) && w(23)==0xd8d80000 && w(24)==((rp<<24)|raddr) &&
                w(25)==0xd8d80004 && w(26)==((rv<<24)|raddr) && rp!=rv && rp!=raddr &&
                w(27)==0xbf8cc17f && w(28)==(0x7e000500u|(p<<17)|rp) && w(29)==0xbf8cc07f &&
                w(30)==(0x7e000500u|(v<<17)|rv) && w(31)==0xbefe04c1;
            if (match) {
                if (mask>104 || (mask&1) || p>=106 || v>=106 || p==v || temp>=106 || temp==v ||
                    alu.read(wave)!=wave_number || alu.read(mask)!="1" || alu.read(mask+1)!="0")
                    refuse(pc,"unproven mesh count exchange wave or lane-zero mask");
                auto input_v=alu.read(v), input_p=alu.read(p);
                if (!upper_bound.count(input_v) || !upper_bound.count(input_p) ||
                    upper_bound.at(input_v)>(std::min)(layout.max_vertices,layout.threads) ||
                    upper_bound.at(input_p)>(std::min)(layout.max_primitives,layout.threads))
                    refuse(pc,"mesh count exchange exceeds declared range or lacks an unsigned bound");
                ir.globals<<"@meshCounts = internal addrspace(3) global [2 x i32] undef, align 4\n";
                auto ptr0=alu.shared_address("meshCounts",2,"0"),ptr1=alu.shared_address("meshCounts",2,"1");
                alu.barrier();
                guarded("1",[&] {
                    ir.body<<"  store i32 "<<input_v<<", i32 addrspace(3)* "<<ptr0<<", align 4\n"
                           <<"  store i32 "<<input_p<<", i32 addrspace(3)* "<<ptr1<<", align 4\n";
                });
                alu.barrier();
                vertex_count=ir.value("load i32, i32 addrspace(3)* "+ptr0+", align 4");
                primitive_count=ir.value("load i32, i32 addrspace(3)* "+ptr1+", align 4");
                ir.invoke("void","setMeshOutputCounts",168,{{"i32",vertex_count},{"i32",primitive_count}},"nounwind");
                alu.write(v,vertex_count); alu.write(p,primitive_count);
                for (auto r:{data,addr,vdata,rp,rv,raddr}) alu.write(256+r,{});
                alu.write(temp,{}); alu.write(128,{});
                full_exec=true; lane_active.clear(); wave_active.clear(); exec_value[0].clear(); exec_value[1].clear();
                allocated=true; pc+=128; continue;
            }
        }
        const auto op = gfx1013::instruction(code,pc);
        if ((branch || exec_skip || !forward.empty()) && ((op.form==gfx1013::Form::exp && export_limit.empty()) || op.name=="s_endpgm" || raw==0xbf900009 ||
                       raw==0xbefe0481 || raw==0xbefe04c1 || (op.name=="s_mov_b32" && ((raw>>16)&127)==124)))
            refuse(pc,"mesh scalar branch crosses allocation, EXEC or export");
        if (op.name=="s_endpgm") {
            if (!allocated || !position || !indices) refuse(pc,"mesh ended without complete position and triangle exports");
            ir.body << "  ret void\n";
            emitter.result.dxil_ir=ir.finish_compute(layout.threads,1,1,"ms");
            emitter.result.group={layout.threads,1,1};
            for (const auto &sig : ir.outputs)
                emitter.result.outputs.push_back({sig.name,sig.semantic_index,sig.row,sig.first,sig.width,
                                                  sig.component,sig.used,sig.kind,sig.interpolation});
            return std::move(emitter.result);
        }
        if (op.name=="s_nop" || op.name=="v_nop" || op.name=="s_clause" || op.name=="s_inst_prefetch" ||
            op.name=="s_waitcnt" || op.name=="s_waitcnt_depctr" || op.name=="s_setprio") {
            pc+=op.size;
            continue;
        }
        // HW_REG_FLAT_SCR_LO / HI are not numeric SGPRs. Scratch accesses
        // and hardware-register reads are still rejected at their own PCs.
        if (raw == 0xba80f814 || raw == 0xba80f815) {
            pc += op.size;
            continue;
        }
        if (raw==0xbf900009) {
            if (allocated || !allocation_word) refuse(pc,"mesh allocation requires one uniform, bounded m0 count word");
            vertices=*allocation_word & 4095; primitives=(*allocation_word>>12)&4095;
            if ((*allocation_word>>24) || vertices>layout.max_vertices || primitives>layout.max_primitives ||
                vertices>layout.threads || primitives>layout.threads || (!vertices && primitives))
                refuse(pc,"mesh allocation exceeds declared lane-indexed output range");
            ir.invoke("void","setMeshOutputCounts",168,{{"i32",to_string(vertices)},{"i32",to_string(primitives)}},"nounwind");
            vertex_count=to_string(vertices); primitive_count=to_string(primitives);
            allocated=true; pc+=4; continue;
        }
        if (op.form==gfx1013::Form::exp) {
            const unsigned target=(raw>>4)&63, channels=raw&15;
            if ((!full_exec && (export_limit.empty() || !exec_skip || exec_skip->guard!=lane_active ||
                 export_limit!=(target==20 ? primitive_count : vertex_count))) || !allocated || (raw&0x400))
                refuse(pc,"unsupported mesh export mask or allocation");
            const auto sources=word(code,pc+4);
            if ((target==12 && channels==15 && !position) ||
                (target>=32 && channels && parameter_exports.insert(target).second)) {
                position |= target==12;
                graphics.pc=pc;
                guarded(vertex_count,[&] {
                    for (unsigned c=0;c<4;++c) {
                        if (!(channels & (1u<<c))) continue;
                        if (!reflected && target>=32) {
                            // Identical fixed-pipeline convention to the raw PS
                            // interpolator: semantic index = 4 * PARAM + channel.
                            const unsigned id=4*(target-32)+c;
                            const unsigned reg=1+(layout.packed_parameters ? target-32 : id);
                            if (reg>=32) refuse(pc,"raw mesh PARAM exceeds signature registers");
                            const auto row=static_cast<uint8_t>(reg), bit=static_cast<uint8_t>(1u<<c);
                            fixed_outputs.outputs.push_back({6,row,bit,static_cast<uint8_t>(target-32),0,0});
                            fixed_outputs.output_signature.push_back({"TEXCOORD",id,0,3,row,bit,bit});
                        }
                        unsigned channel=c;
                        const auto id=graphics.output(target,channel);
                        auto &sig=ir.outputs.at(id);
                        const unsigned col=channel-sig.first;
                        auto bits=alu.read(256+((sources>>(8*c))&255));
                        const bool fp=sig.component==9;
                        ir.invoke("void",fp ? "storeVertexOutput.f32" : "storeVertexOutput.i32",171,
                                  {{"i32",to_string(id)},{"i32","0"},{"i8",to_string(col)},
                                   {fp ? "float" : "i32",fp ? ir.value("bitcast i32 "+bits+" to float") : bits},
                                   {"i32",thread}},"nounwind");
                        sig.used|=1u<<col;
                    }
                });
            } else if (target==20 && channels==1 && !indices) {
                indices=true;
                guarded(primitive_count,[&] {
                    auto packed=alu.read(256+(sources&255));
                    auto culled=alu.cmp("slt",packed,"0");
                    std::array<string,3> triangle;
                    for (unsigned c=0;c<3;++c) {
                        triangle[c]=ir.binary("and",ir.binary("lshr",packed,to_string(10*c)),"1023");
                        culled=ir.value("or i1 "+culled+", "+alu.cmp("uge",triangle[c],to_string(vertices)));
                    }
                    ir.invoke("void","emitIndices",169,{{"i32",thread},{"i32",alu.select(culled,"0",triangle[0])},
                              {"i32",alu.select(culled,"0",triangle[1])},{"i32",alu.select(culled,"0",triangle[2])}},"nounwind");
                });
            } else refuse(pc,"unmapped mesh export target "+to_string(target));
            pc+=op.size; continue;
        }
        const auto i=decode(code,pc,true,op.form!=gfx1013::Form::mtbuf);
        std::optional<unsigned> result_bound;
        if (i.name=="s_and_b32") result_bound=(std::min)(bound(i,i.src[0]),bound(i,i.src[1]));
        if (i.name=="s_bfe_u32") {
            if (auto field=DirectCompute::constant32(i,i.src[1]); field && ((*field>>16)&127)<32)
                result_bound=(1u<<((*field>>16)&127))-1;
        }
        if (i.name=="s_add_u32") {
            uint64_t sum=uint64_t(bound(i,i.src[0]))+bound(i,i.src[1]);
            if (sum<=UINT32_MAX) result_bound=static_cast<unsigned>(sum);
        }
        if (i.name=="s_cselect_b32") result_bound=(std::max)(bound(i,i.src[0]),bound(i,i.src[1]));
        const unsigned vec_dst=i.enc==Encoding::Mubuf ? 256+i.dst : i.dst;
        std::array<string,4> before_vec;
        std::array<bool,4> before_partial{};
        if (!full_exec && !lane_active.empty() && vec_dst>=256 && vec_dst<512) {
            for (unsigned j=0;j<4;++j) {
                const auto old=alu.graphics_registers.find(vec_dst+j);
                if (old!=alu.graphics_registers.end()) before_vec[j]=old->second;
                before_partial[j]=partial_writes.count(vec_dst+j)!=0;
            }
        }
        const bool export_cmp=i.name=="v_cmpx_gt_i32" && allocated && i.src[0]<106 && full_exec &&
            (alu.read(i.src[0])==vertex_count || alu.read(i.src[0])==primitive_count);
        if ((i.name=="v_cmpx_gt_u32" || export_cmp) && full_exec && !branch && !exec_skip && i.enc==Encoding::Vopc && !i.wide && !i.subword && !i.dpp_control &&
            i.src[0]<106 && linear_threads.count(alu.operand32(i,i.src[1]))) {
            // Lane-linear count comparison in a merged workgroup. Construct
            // both halves from the guest wave start, independent of native
            // WaveGetLaneCount. Saturation avoids shifts by 32 or unsigned
            // underflow when the last group wave is empty.
            const auto limit=alu.operand32(i,i.src[0]);
            export_limit=export_cmp ? limit : string{};
            const auto capped=alu.select(alu.cmp("ult",limit,to_string(layout.threads)),limit,to_string(layout.threads));
            const auto base=ir.binary("sub",thread,wave_lane);
            wave_active=alu.cmp("ugt",capped,base);
            lane_active=alu.cmp("ult",thread,capped);
            for (unsigned j=0;j<2;++j) {
                if (j && layout.wave32) { exec_value[j]="0"; continue; }
                const auto begin=j ? ir.binary("add",base,"32") : base;
                const auto n=alu.select(alu.cmp("ugt",capped,begin),ir.binary("sub",capped,begin),"0");
                const auto bits=ir.binary("sub",ir.binary("shl","1",ir.binary("and",n,"31")),"1");
                exec_value[j]=alu.select(alu.cmp("uge",n,"32"),"4294967295",bits);
            }
            full_exec=false;
        } else if (i.name=="v_cmpx_gt_i32" && !branch && exec_skip && !lane_active.empty() &&
                   i.enc==Encoding::Vopc && !i.wide && !i.subword && !i.dpp_control) {
            const auto a=decode(code,i.next,true,true);
            if (a.name!="s_cbranch_execz" || a.target()<=a.next || a.target()>=code.size())
                refuse(pc,"mesh signed CMPX requires paired EXECZ arms");
            const auto inverse=decode(code,a.target(),true,true);
            const auto b=decode(code,inverse.next,true,true);
            if (inverse.name!="s_andn2_b64" || inverse.dst!=126 || inverse.src[1]!=126 || inverse.src[0]>104 ||
                b.name!="s_cbranch_execz" || b.target()<=b.next || b.target()>=code.size() ||
                !pure_lane_arm(a.next,a.target()) || !pure_lane_arm(b.next,b.target()))
                refuse(pc,"mesh CMPX arms contain an unproven wave effect");
            const auto mask=saved_mask(inverse.src[0]);
            if (!mask || mask->lane!=lane_active || mask->wave!=wave_active || mask->full!=full_exec)
                refuse(pc,"mesh CMPX inverse lacks the saved parent EXEC");
            branch=Diamond{pc,a.target(),b.target(),save_edge(),{}};
            lane_else=b.next;
            auto take=alu.comparison(i);
            ir.body<<"  br i1 "<<take<<", label %mesh_then"<<pc<<", label %mesh_else"<<pc
                   <<"\nmesh_then"<<pc<<":\n";
            pc=a.next; continue;
        } else if (i.name=="v_cmpx_eq_u32" && !branch && !i.dpp_control && !i.subword && !i.abs && !i.neg &&
                   !i.scale && !i.clamp && (full_exec || !lane_active.empty())) {
            const auto before=lane_active;
            auto pred=alu.comparison(i);
            lane_active=full_exec ? pred : ir.value("and i1 "+before+", "+pred);
            mask_parent[lane_active]=before;
            full_exec=false; wave_active.clear(); exec_value[0].clear();exec_value[1].clear();
        } else if (i.name=="s_cbranch_execz" && !full_exec && !lane_active.empty() && !branch &&
                   i.target()>i.next && i.target()<code.size()) {
            if (exec_skip) {
                if (i.target()>exec_skip->end) refuse(pc,"overlapping mesh EXECZ exits");
                skip_parents.push_back(std::move(*exec_skip));
            }
            exec_skip=Skip{pc,i.target(),save_edge(),wave_active.empty(),lane_active};
            ir.body<<"  br i1 "<<(wave_active.empty() ? lane_active : wave_active)<<", label %mesh_work"<<pc<<", label %mesh_empty"<<pc
                   <<"\nmesh_empty"<<pc<<":\n  br label %mesh_resume"<<pc<<"\nmesh_work"<<pc<<":\n";
        } else if (i.name=="s_cbranch_scc0" || i.name=="s_cbranch_scc1" ||
                   i.name=="s_cbranch_vccz" || i.name=="s_cbranch_vccnz") {
            const auto other=i.target();
            if (lane_else || (!full_exec && !exec_skip) || allocated || other<=i.next || other>=code.size() ||
                (exec_skip && other>exec_skip->end))
                refuse(pc,"unsupported mesh scalar branch destination");
            for (unsigned at=i.next;at<other;) {
                const auto next=at+gfx1013::instruction(code,at).size;
                if (next<=at || next>other) refuse(pc,"mesh branch target splits instruction");
                at=next;
            }
            const bool vcc=i.name=="s_cbranch_vccz" || i.name=="s_cbranch_vccnz";
            string value;
            if (!vcc) value=alu.read(128);
            else {
                auto nonempty=alu.graphics_registers.find(vcc_nonempty);
                if (nonempty!=alu.graphics_registers.end() && !nonempty->second.empty()) value=nonempty->second;
                else {
                    auto lo=alu.graphics_registers.find(vcc_words), hi=alu.graphics_registers.find(vcc_words+1);
                    if (lo==alu.graphics_registers.end() || lo->second.empty() ||
                        hi==alu.graphics_registers.end() || hi->second.empty())
                        refuse(pc,"mesh VCC branch lacks a proven guest whole-wave mask");
                    value=ir.binary("or",lo->second,hi->second);
                }
            }
            auto take=alu.cmp(i.name=="s_cbranch_scc0" || i.name=="s_cbranch_vccz" ? "eq" : "ne",value,"0");
            const auto edge=arrive(other), fall="mesh_fall"+to_string(pc);
            ir.body<<"  br i1 "<<take<<", label %"<<edge<<", label %"<<fall<<"\n"<<fall<<":\n";
        } else if (i.name=="s_branch" && !lane_else && i.target()>=i.next && i.target()<code.size() &&
                   (!exec_skip || i.target()<=exec_skip->end)) {
            const auto target=i.target();
            for (unsigned at=i.next;at<target;) {
                const auto next=at+gfx1013::instruction(code,at).size;
                if (next<=at || next>target) refuse(pc,"mesh branch target splits instruction");
                at=next;
            }
            ir.body<<"  br label %"<<arrive(target)<<"\n";
            live=false;
        } else if (auto field=byte_descriptor_field(i,state)) {
            alu.write(i.dst,{});
            state[i.dst]=*field;
            if (i.name!="s_mul_i32")
                alu.write(128,field->type==Symbol::Type::byte_oob || field->type==Symbol::Type::root_stride_one_high ?
                              "1" : string{});
        } else if (i.enc==Encoding::Smem) {
            const auto offset_reg=i.b>>25;
            if ((offset_reg!=125 && offset_reg>=106) || i.src[0]>=105 || i.dst+memory_count(i)>106)
                refuse(pc,"unsupported mesh scalar memory address");
            if (i.op<8 && state[i.src[0]].type!=Symbol::Type::table)
                refuse(pc,"mesh scalar memory needs a verified guest root mapping: "+string(op.name));
            if (i.op<8 && offset_reg!=125 && state[i.src[0]].value!=compute_resource_heap)
                refuse(pc,"indexed mesh descriptor table lacks a binding contract");
            auto next=transfer(i,state,!layout.wave32,true);
            const bool global=state[i.src[0]].value==compute_resource_heap && state[i.src[0]].type==Symbol::Type::table;
            if (global && layout.validate_uint_heap_reads) {
                if (i.op!=2 || offset_reg>=106 || (i.b&0xfffff) ||
                    !(state[i.src[0]+1]==Symbol{Symbol::Type::constant,0}))
                    refuse(pc,"unsupported mesh heap descriptor address");
                const auto bytes=alu.operand32(i,offset_reg);
                const auto bits=root_values.find(bytes);
                if (bits==root_values.end() || bits->second.shift!=5)
                    refuse(pc,"mesh heap slot lacks masked root constant provenance");
                heap_fetches[pc]=bits->second;
            } else alu.memory(i,state);
            if (i.op<8) {
                for (unsigned j=0;j<memory_count(i);++j) alu.write(i.dst+j,{});
            }
            state=next;
        } else if (i.name=="buffer_load_format" && i.op==0 && full_exec && (i.a&0x3ffff)==0x2000) {
            const auto base=((i.b>>16)&31)*4;
            const auto sym=state[base];
            if (sym.type!=Symbol::Type::indexed_descriptor || sym.value!=compute_resource_heap ||
                !heap_fetches.count(sym.definition) || sym.offset)
                refuse(pc,"formatted mesh load lacks an R32_UINT resource heap contract");
            for (unsigned j=1;j<4;++j) {
                auto expected=sym;
                expected.offset=j*4;
                if (!(expected==state[base+j])) refuse(pc,"overwritten mesh heap descriptor");
            }
            const auto guard=heap_fetches.at(sym.definition);
            const auto entry=std::find_if(root_values.begin(),root_values.end(),[&](const auto &x) {
                return x.second.param==guard.param && x.second.dword==guard.dword && !x.second.shift && x.second.mask==~0u;
            });
            if (entry==root_values.end()) refuse(pc,"missing mesh heap selector");
            emitter.result.mesh_heap_reads.push_back({guard.param,guard.dword,guard.mask});
            const auto descriptor=ir.binary("and",entry->first,to_string(guard.mask));
            ir.features|=0x40000000;
            const auto handle=ir.heap_handle(descriptor,10,261); // TypedBuffer, one U32 component.
            const auto index=alu.read(256+(i.b&255));
            const auto word=ir.value("extractvalue %dx.types.ResRet.i32 "+alu.load_buffer(handle,index,false,1)+", 0");
            alu.write(256+i.dst,word);
            if (uniform_values.count(index)) uniform_values.insert(word);
        } else if (op.form==gfx1013::Form::mtbuf && (full_exec || exec_skip)) {
            const auto base=((i.b>>16)&31)*4;
            const auto low=state[base];
            string window;
            if (low.type==Symbol::Type::root_address) {
                const bool zero_stride=state[base+1]==Symbol{Symbol::Type::root_address,low.value,1} &&
                                       state[base+3]==Symbol{Symbol::Type::constant,0x31016fac};
                const bool byte_stride=state[base+1]==Symbol{Symbol::Type::root_stride_one_high,low.value,1} &&
                                       state[base+3]==Symbol{Symbol::Type::constant,0x11014fac};
                if (low.offset || (!zero_stride && !byte_stride) ||
                    !(state[base+2]==Symbol{Symbol::Type::constant,0xffffffff}))
                    refuse(pc,"mesh formatted root load requires an unbounded byte window");
                window=alu.root_buffer_handle(low.value,4);
            }
            alu.formatted_byte_load(i,state,window);
            if (!full_exec) for (unsigned j=0;j<=i.op;++j) partial_writes.insert(256+i.dst+j);
        } else if (i.name=="buffer_load_dword" && (full_exec || exec_skip)) {
            alu.memory(i,state);
            if (!full_exec) {
                constexpr unsigned counts[]{1,2,4,3};
                for (unsigned j=0;j<counts[i.op&3];++j) partial_writes.insert(256+i.dst+j);
            }
        } else if (i.enc==Encoding::Vopc && i.name.substr(0,6)=="v_cmp_" && i.dst<=106 &&
                   !i.dpp_control && !i.scale && !i.clamp && (full_exec || !lane_active.empty())) {
            auto next=transfer(i,state,!layout.wave32,true);
            auto pred=alu.comparison(i);
            if (!full_exec) pred=ir.value("and i1 "+pred+", "+lane_active);
            const auto bit=ir.value("zext i1 "+pred+" to i32");
            if (!i.subword) {
                const auto lhs=uniform(alu.operand32(i,i.src[0])), rhs=uniform(alu.operand32(i,i.src[1]));
                if (!lhs.empty() && !rhs.empty()) mask_uniform[bit]={lane_active,alu.comparison(i,lhs,rhs)};
            }
            if (i.dst==106) {
                alu.write(106,bit);
                alu.write(vcc_words,{}); alu.write(vcc_words+1,{}); alu.write(vcc_nonempty,{});
                next[106].type=next[107].type=Symbol::Type::unbound;
            } else {
                alu.write(i.dst,{});
                if (!layout.wave32) alu.write(i.dst+1,{});
                alu.write(1024+i.pc,bit);
            }
            state=next;
        } else if (i.name=="v_cndmask_b32" && !i.scale && !i.clamp && !i.dpp_control &&
                   (full_exec || !lane_active.empty())) {
            if ((i.wide && (i.a&0x7800)) || ((i.abs|i.neg)&4)) refuse(pc,"unsupported mesh CNDMASK operand selection");
            auto a=alu.operand32(i,i.src[0]), b=alu.operand32(i,i.src[1]);
            if (i.subword) {
                a=alu.subword(a,i.src_select[0],i.src_sext[0]);
                b=alu.subword(b,i.src_select[1],i.src_sext[1]);
            }
            if (i.abs&1) a=ir.binary("and",a,"2147483647");
            if (i.abs&2) b=ir.binary("and",b,"2147483647");
            if (i.neg&1) a=ir.binary("xor",a,"2147483648");
            if (i.neg&2) b=ir.binary("xor",b,"2147483648");
            auto bit=alu.graphics_lane_mask(i,state,i.src[2],"unproven mesh CNDMASK");
            const auto value=alu.place_subword(i,alu.select(alu.cmp("ne",bit,"0"),b,a));
            alu.write(i.dst,value);
            if (mask_uniform.count(bit) && mask_uniform.at(bit).first==lane_active && !uniform(a).empty() && !uniform(b).empty() && !i.subword)
                active_uniform[value]={lane_active,alu.select(mask_uniform.at(bit).second,uniform(b),uniform(a))};
            if (!full_exec) partial_writes.insert(i.dst);
        } else if (i.name=="s_cselect_b64" && !(i.dst&1) && i.dst<=106 &&
                   (i.src[0]==128 || i.src[0]==193) && (i.src[1]==128 || i.src[1]==193)) {
            auto bit=alu.select(alu.cmp("ne",alu.read(128),"0"),i.src[0]==193 ? "1" : "0",i.src[1]==193 ? "1" : "0");
            if (i.dst==106) {
                alu.write(106,bit);
                auto word=ir.binary("sub","0",bit);
                mask_uniform[bit]={"",alu.cmp("ne",bit,"0")};
                alu.write(vcc_words,word); alu.write(vcc_words+1,word); alu.write(vcc_nonempty,word);
                state[106]={}; state[107]={};
            } else {
                auto next=transfer(i,state,!layout.wave32,true);
                const auto word=ir.binary("sub","0",bit);
                alu.write(i.dst,word); alu.write(i.dst+1,word);
                alu.write(1024+i.pc,bit);
                state=next;
            }
        } else if (i.name=="s_cselect_b64" && i.dst<=104 && !(i.dst&1) && i.src[0]==126 &&
                   i.src[1]==128 && (full_exec || !lane_active.empty())) {
            const auto cond=alu.cmp("ne",alu.read(128),"0");
            const auto pred=full_exec ? cond : ir.value("and i1 "+cond+", "+lane_active);
            alu.write(i.dst,alu.select(cond,full_exec ? "4294967295" : exec_value[0],"0"));
            alu.write(i.dst+1,alu.select(cond,full_exec ? (layout.wave32 ? "0" : "4294967295") : exec_value[1],"0"));
            state[i.dst]={Symbol::Type::lane_mask,pc,0};
            state[i.dst+1]={Symbol::Type::lane_mask,pc,1};
            const auto lane=ir.value("zext i1 "+pred+" to i32");
            alu.write(1024+pc,lane); mask_uniform[lane]={lane_active,cond};
        } else if (i.name=="s_not_b64" && i.dst<=104 && !(i.dst&1) && i.src[0]<=104 && !(i.src[0]&1)) {
            const auto a=alu.graphics_lane_mask(i,state,i.src[0],"unproven mesh NOT mask");
            std::array<string,2> data{};
            for (unsigned j=0;j<2;++j) {
                auto it=alu.graphics_registers.find(i.src[0]+j);
                if (it!=alu.graphics_registers.end() && !it->second.empty()) data[j]=ir.binary("xor",it->second,"4294967295");
            }
            alu.write(i.dst,data[0]); alu.write(i.dst+1,data[1]);
            state[i.dst]={Symbol::Type::lane_mask,pc,0}; state[i.dst+1]={Symbol::Type::lane_mask,pc,1};
            alu.write(1024+pc,ir.binary("xor",a,"1")); alu.write(128,{});
        } else if (i.name=="s_and_b64" && i.dst==126 && i.src[1]==126 && i.src[0]<=104 &&
                   !full_exec && exec_skip && !branch && !lane_active.empty() &&
                   [&] {
                       auto a=decode(code,i.next,true,true);
                       if (a.name!="s_cbranch_execz" || a.target()<=a.next || a.target()+8>=code.size()) return false;
                       auto inv=decode(code,a.target(),true,true);
                       auto b=decode(code,inv.next,true,true);
                       if (inv.name!="s_andn2_b64" || inv.dst!=126 || inv.src[1]!=126 ||
                           b.name!="s_cbranch_execz" || b.target()<=b.next || b.target()+4>code.size()) return false;
                       auto reset=decode(code,b.target(),true,true);
                       auto parent=saved_mask(inv.src[0]);
                       if (!parent || parent->lane!=lane_active || parent->wave!=wave_active ||
                           reset.name!="s_mov_b64" || reset.dst!=126 || reset.src[0]!=inv.src[0]) return false;
                       auto left=gather_arm(a.next,a.target()),right=gather_arm(b.next,b.target());
                       if (!left || !right) return false;
                       for (auto r : right->inputs) if (left->writes.count(r)) return false;
                       for (auto r : {inv.src[0],inv.src[0]+1}) if (left->writes.count(r) || right->writes.count(r)) return false;
                       branch=Diamond{pc,a.target(),b.target(),save_edge(),{}};
                       lane_scalar_clobbers=left->writes;
                       lane_scalar_clobbers.insert(right->writes.begin(),right->writes.end());
                       lane_scalar_clobbers.insert(128);
                       lane_else=b.next;
                       auto take=alu.cmp("ne",alu.graphics_lane_mask(i,state,i.src[0],"unproven mesh gather mask"),"0");
                       ir.body<<"  br i1 "<<take<<", label %mesh_then"<<pc<<", label %mesh_else"<<pc
                              <<"\nmesh_then"<<pc<<":\n";
                       pc=a.next;
                       return true;
                   }()) {
            continue;
        } else if ((i.name=="s_or_b64" || i.name=="s_and_b64" || i.name=="s_andn2_b64") &&
                   (i.dst<=106 || i.dst==126) && !(i.dst&1)) {
            auto lane_bit=[&](unsigned src) {
                if (src==126 && (full_exec || !lane_active.empty()))
                    return full_exec ? string("1") : ir.value("zext i1 "+lane_active+" to i32");
                return alu.graphics_lane_mask(i,state,src,"unproven mesh mask operand");
            };
            auto mask_word=[&](unsigned src,unsigned j) -> string {
                if (src==126) return full_exec ? (j && layout.wave32 ? "0" : "4294967295") : exec_value[j];
                if (src==106) src=vcc_words;
                auto v=alu.graphics_registers.find(src+j);
                return v==alu.graphics_registers.end() ? string{} : v->second;
            };
            const auto a=lane_bit(i.src[0]),b=lane_bit(i.src[1]);
            auto combine=[&](const string &x,const string &y,bool lane) {
                if (x.empty() || y.empty()) return string{};
                return ir.binary(i.name=="s_or_b64" ? "or" : "and",x,i.name=="s_andn2_b64" ?
                                 ir.binary("xor",y,lane ? "1" : "4294967295") : y);
            };
            const auto lo=combine(mask_word(i.src[0],0),mask_word(i.src[1],0),false);
            const auto hi=combine(mask_word(i.src[0],1),mask_word(i.src[1],1),false);
            const auto bit=combine(a,b,true);
            string any;
            if (!lo.empty() && !hi.empty()) any=ir.binary("or",lo,hi);
            else if (((i.src[0]==126) || (i.src[1]==126 && i.name=="s_and_b64")) && (i.name=="s_andn2_b64" || i.name=="s_and_b64") && mask_uniform.count(i.src[0]==126 ? b : a) &&
                     covered_by(lane_active,mask_uniform.at(i.src[0]==126 ? b : a).first) &&
                     (full_exec || !wave_active.empty() || (exec_skip && exec_skip->guard==lane_active))) {
                auto flag=mask_uniform.at(i.src[0]==126 ? b : a).second;
                if (i.name=="s_andn2_b64") flag=ir.value("xor i1 "+flag+", true");
                if (!wave_active.empty()) flag=ir.value("and i1 "+flag+", "+wave_active);
                any=ir.value("zext i1 "+flag+" to i32");
            }
            if (i.dst==126) {
                const auto before=lane_active;
                if (i.name=="s_and_b64" && (i.src[0]==126 || i.src[1]==126))
                    before_narrowing[before]=save_edge();
                full_exec=false;
                lane_active=alu.cmp("ne",bit,"0");
                if (i.name!="s_or_b64" && (i.src[0]==126 || (i.name=="s_and_b64" && i.src[1]==126))) mask_parent[lane_active]=before;
                wave_active=lo.empty() || hi.empty() ? string{} : alu.cmp("ne",ir.binary("or",lo,hi),"0");
                exec_value[0]=lo; exec_value[1]=hi;
            } else if (i.dst==106) {
                alu.write(106,bit);
                alu.write(vcc_words,lo); alu.write(vcc_words+1,hi); alu.write(vcc_nonempty,any);
                state[106].type=lo.empty()?Symbol::Type::unbound:Symbol::Type::unknown;
                state[107].type=hi.empty()?Symbol::Type::unbound:Symbol::Type::unknown;
            } else {
                alu.write(i.dst,lo); alu.write(i.dst+1,hi);
                state[i.dst]={Symbol::Type::lane_mask,pc,0};
                state[i.dst+1]={Symbol::Type::lane_mask,pc,1};
                alu.write(1024+pc,bit);
            }
            alu.write(128,any.empty() ? string{} : ir.value("zext i1 "+alu.cmp("ne",any,"0")+" to i32"));
        } else if ((i.name=="s_bitset0_b32" || i.name=="s_bitset1_b32") && i.dst<106) {
            auto bit=ir.binary("shl","1",ir.binary("and",alu.operand32(i,i.src[0]),"31"));
            const auto old=alu.operand32(i,i.dst);
            alu.write(i.dst,i.name=="s_bitset0_b32" ? ir.binary("and",old,ir.binary("xor",bit,"4294967295")) :
                      ir.binary("or",old,bit));
        } else if (i.name=="s_bitcmp0_b32" || i.name=="s_bitcmp1_b32" || i.name=="s_cmp_lt_u32" ||
                   i.name=="s_mulk_i32" || i.name=="s_addk_i32" || i.name=="s_mul_i32" || i.name=="s_add_u32" || i.name=="s_add_i32" ||
                   i.name=="s_cmp_gt_i32" || i.name=="s_cmp_lt_i32" || i.name=="s_cmp_ge_i32" || i.name=="s_cmp_eq_u32" || i.name=="s_cmp_eq_i32" || i.name=="s_cmp_lg_u32" || i.name=="s_cmp_ge_u32" || i.name=="s_cmp_lg_i32" || i.name=="s_cmp_gt_u32" || i.name=="s_cselect_b32") {
            alu.arithmetic(i,state);
        } else if (raw==0xbefe04c1) {
            full_exec=true;
            export_limit.clear();
            for (auto r : partial_writes) alu.write(r,{});
            partial_writes.clear();
            wave_active.clear(); lane_active.clear();
            exec_value[0].clear(); exec_value[1].clear();
        } else if (i.name=="s_mov_b64" && i.dst==106 && (i.src[0]==128 || i.src[0]==193)) {
            const bool set=i.src[0]==193;
            auto value=set ? "4294967295" : "0";
            alu.write(106,set ? "1" : "0"); alu.write(vcc_words,value); alu.write(vcc_words+1,value);
            alu.write(vcc_nonempty,value);
            state[106]=state[107]={Symbol::Type::constant,set ? ~0u : 0u};
            // Thread the known-zero predecessor through a mask test before
            // merging with paths that computed its optional buffer values.
            if (!set && i.next+8<=code.size() && !branch && gfx1013::instruction(code,i.next).name=="s_and_b64") {
                auto test=decode(code,i.next,true,true);
                auto jump=decode(code,test.next,true,true);
                if (test.name=="s_and_b64" && test.dst<=104 && !(test.dst&1) &&
                    ((test.src[0]==106 && test.src[1]==126) || (test.src[0]==126 && test.src[1]==106)) &&
                    jump.name=="s_cbranch_scc0" && jump.target()>=jump.next && jump.target()<code.size() &&
                    (!exec_skip || jump.target()<=exec_skip->end)) {
                    alu.write(test.dst,"0"); alu.write(test.dst+1,"0"); alu.write(128,"0");
                    state[test.dst]={Symbol::Type::uniform_mask,test.pc,0};
                    state[test.dst+1]={Symbol::Type::uniform_mask,test.pc,1};
                    alu.write(1024+test.pc,"0");
                    ir.body<<"  br label %"<<arrive(jump.target())<<"\n";
                    live=false;
                }
            }
        } else if (i.name=="s_mov_b64" && i.dst==106 && i.src[0]<=104 && !(i.src[0]&1)) {
            const auto src=i.src[0];
            const auto old=state;
            const auto next=transfer(i,state,!layout.wave32,true);
            auto physical=[&](unsigned r) {
                auto it=alu.graphics_registers.find(r); return it==alu.graphics_registers.end() ? string{} : it->second;
            };
            const auto lo=physical(src), hi=physical(src+1);
            string bit;
            if (old[src].type==Symbol::Type::lane_mask || old[src].type==Symbol::Type::uniform_mask)
                bit=alu.graphics_lane_mask(i,old,src,"unproven mesh VCC move");
            else if (!lo.empty() && !hi.empty()) {
                auto half=alu.select(alu.cmp("ult",wave_lane,"32"),lo,hi);
                bit=ir.binary("and",ir.binary("lshr",half,ir.binary("and",wave_lane,"31")),"1");
            }
            alu.write(106,bit); alu.write(vcc_words,lo); alu.write(vcc_words+1,hi);
            alu.write(vcc_nonempty,lo.empty() || hi.empty() ? string{} : ir.binary("or",lo,hi));
            state=next;
        } else if (i.name=="s_mov_b64" && i.dst<=104 && !(i.dst&1) && i.src[0]==126 &&
                   (full_exec || !lane_active.empty())) {
            alu.write(i.dst,full_exec ? "4294967295" : exec_value[0]);
            alu.write(i.dst+1,full_exec ? (layout.wave32 ? "0" : "4294967295") : exec_value[1]);
            state[i.dst]={Symbol::Type::lane_mask,pc,0}; state[i.dst+1]={Symbol::Type::lane_mask,pc,1};
            alu.write(1024+pc,full_exec ? "1" : ir.value("zext i1 "+lane_active+" to i32"));
            saved_exec[pc]={full_exec,lane_active,wave_active,exec_value[0],exec_value[1],alu.graphics_registers,partial_writes};
        } else if (i.name=="s_mov_b64" && i.dst==126 && i.src[0]<=104 && !(i.src[0]&1)) {
            const auto mask=saved_mask(i.src[0]);
            if (!mask) refuse(pc,"mesh EXEC restore lacks a still-valid reaching mask");
            if (mask->lane!=lane_active) {
                auto at_narrow=before_narrowing.find(mask->lane);
                const auto &regs=at_narrow==before_narrowing.end() ? mask->regs : at_narrow->second.regs;
                const auto &valid=at_narrow==before_narrowing.end() ? mask->partial : at_narrow->second.partial;
                std::set<unsigned> remaining;
                for (auto r:partial_writes) {
                    auto at=regs.find(r);
                    if (at==regs.end() || at->second.empty() || at->second!=alu.graphics_registers.at(r))
                        alu.write(r,{});
                    else if (valid.count(r)) remaining.insert(r);
                }
                partial_writes=std::move(remaining);
            }
            full_exec=mask->full; lane_active=mask->lane; wave_active=mask->wave;
            exec_value[0]=mask->lo; exec_value[1]=mask->hi;
        } else if (i.name=="s_mov_b64" && i.dst<=104 && !(i.dst&1) && i.src[0]<=104 && !(i.src[0]&1)) {
            auto next=transfer(i,state,!layout.wave32,true);
            auto src=alu.graphics_registers.find(i.src[0]), high=alu.graphics_registers.find(i.src[0]+1);
            auto lo=src==alu.graphics_registers.end() ? string{} : src->second;
            auto hi=high==alu.graphics_registers.end() ? string{} : high->second;
            alu.write(i.dst,lo);
            alu.write(i.dst+1,hi);
            state=next;
        } else if (i.name=="s_mov_b64" && i.dst<=104 && !(i.dst&1) && i.src[0]>=128 && i.src[0]<=208) {
            auto next=transfer(i,state,!layout.wave32,true);
            alu.write(i.dst,to_string(next[i.dst].value)); alu.write(i.dst+1,to_string(next[i.dst+1].value));
            state=next;
        } else if (i.name=="s_mov_b32" && i.dst==125 && DirectCompute::constant32(i,i.src[0])) {
            // NULL discards the scalar move; MOV does not write SCC.
        } else if (i.name=="s_mov_b32" && i.dst==124) {
            allocation_word=DirectCompute::constant32(i,i.src[0]);
        } else if (i.name=="s_mov_b32" && i.dst<106) {
            const auto provenance=source(i,i.src[0],state);
            if (auto value=DirectCompute::constant32(i,i.src[0])) alu.write(i.dst,to_string(*value));
            else if (i.src[0]<=107) {
                const auto source=alu.graphics_registers.find(i.src[0]<106 ? i.src[0] : vcc_words+i.src[0]-106);
                alu.write(i.dst,source==alu.graphics_registers.end() ? string{} : source->second);
            } else if (i.src[0]==253) alu.write(i.dst,alu.read(128));
            else refuse(pc,"unsupported scalar mesh entry operand");
            state[i.dst]=provenance;
        } else if (i.name=="s_bfe_u32" && i.src[0]==3 && !alu.graphics_registers.count(3) &&
                   i.src[1]==255 && i.literal==0x40018 && i.dst<106) {
            alu.write(i.dst,wave_number);
            alu.write(128,ir.value("zext i1 "+alu.cmp("ne",wave_number,"0")+" to i32"));
        } else if (i.name=="s_bcnt1_i32_b64" && i.src[0]<=104 && !(i.src[0]&1) && i.dst<106) {
            auto pop=[&](unsigned r) {
                return ir.invoke("i32","unaryBits.i32",31,{{"i32",alu.read(r)}},"nounwind readnone");
            };
            const auto lo=pop(i.src[0]), hi=pop(i.src[0]+1);
            const auto sum=ir.binary("add",lo,hi);
            alu.write(i.dst,sum);
            alu.write(128,ir.value("zext i1 "+alu.cmp("ne",sum,"0")+" to i32"));
        } else if (i.name=="s_bfe_u32" || i.name=="s_min_u32") {
            alu.arithmetic(i,state);
        } else if (i.name=="v_readfirstlane_b32" && i.dst<106 && i.src[0]==261 && !exchanged_group.empty() && full_exec) {
            alu.write(i.dst,exchanged_group);
        } else if (i.name=="v_readfirstlane_b32" && i.dst<106 && (full_exec || exec_skip) &&
                   !uniform(alu.read(i.src[0])).empty()) {
            alu.write(i.dst,uniform(alu.read(i.src[0])));
        } else if (i.name=="s_pack_ll_b32_b16" || i.name=="s_pack_hh_b32_b16") {
            const auto a=alu.operand32(i,i.src[0]),b=alu.operand32(i,i.src[1]);
            const bool high=i.name=="s_pack_hh_b32_b16";
            alu.write(i.dst,ir.binary("or",high ? ir.binary("lshr",a,"16") : ir.binary("and",a,"65535"),
                         high ? ir.binary("and",b,"4294901760") : ir.binary("shl",b,"16")));
        } else if (i.enc==Encoding::Sop2 && i.dst<106 && (i.name=="s_lshr_b32" || i.name=="s_lshl_b32" ||
                   i.name=="s_or_b32" || i.name=="s_and_b32")) {
            const auto known=[&](unsigned r) {
                return DirectCompute::constant32(i,r).has_value() ||
                    (alu.graphics_registers.count(r) && !alu.graphics_registers.at(r).empty());
            };
            if (known(i.src[0]) && known(i.src[1])) {
                const auto a=alu.operand32(i,i.src[0]), b=alu.operand32(i,i.src[1]);
                const bool shift=i.name=="s_lshl_b32" || i.name=="s_lshr_b32";
                const auto value=ir.binary(i.name=="s_lshl_b32" ? "shl" : i.name=="s_lshr_b32" ? "lshr" :
                                           i.name=="s_or_b32" ? "or" : "and",a,shift ? ir.binary("and",b,"31") : b);
                if (i.name=="s_lshl_b32" && a==wave_number && DirectCompute::constant32(i,i.src[1])==(layout.wave32 ? 5u : 6u))
                    wave_bases.insert(value);
                const auto bits=root_values.find(a);
                const auto imm=DirectCompute::constant32(i,i.src[1]);
                if (bits!=root_values.end() && imm && !bits->second.shift) {
                    auto out=bits->second;
                    if (i.name=="s_and_b32") { out.mask&=*imm; root_values.emplace(value,out); }
                    if (i.name=="s_lshl_b32" && *imm==5 && out.mask<=0x7ffffffu) {
                        out.shift=5; root_values.emplace(value,out);
                    }
                }
                const auto next=transfer(i,state,!layout.wave32,true);
                alu.write(i.dst,value);
                state[i.dst]=next[i.dst];
                alu.write(128,ir.value("zext i1 "+alu.cmp("ne",value,"0")+" to i32"));
            }
            else {
                alu.write(i.dst,{});
                alu.write(128,{});
            }
        } else if (full_exec && (i.name=="v_mbcnt_lo_u32_b32" || i.name=="v_mbcnt_hi_u32_b32") && i.src[0]==193) {
            const auto add=i.name=="v_mbcnt_lo_u32_b32" ? alu.select(alu.cmp("ult",wave_lane,"32"),wave_lane,"32") :
                alu.select(alu.cmp("ult",wave_lane,"32"),"0",ir.binary("sub",wave_lane,"32"));
            const auto base=alu.operand32(i,i.src[1]);
            const auto value=ir.binary("add",base,add);
            if (i.name=="v_mbcnt_lo_u32_b32") {
                if (base=="0") low_lane.insert(value);
                if (wave_bases.count(base)) low_group.insert(value);
            } else {
                if (low_lane.count(base)) whole_lane.insert(value);
                if (low_group.count(base)) linear_threads.insert(value);
            }
            alu.write(i.dst,value);
        } else if (i.name=="v_perm_b32" && !i.abs && !i.neg && !i.clamp && !i.scale && !i.subword &&
                   !i.dpp_control && (full_exec || exec_skip)) {
            // RDNA V_PERM_B32: selectors 0..3 index src1, 4..7 index src0,
            // 8..11 replicate the sign of each halfword; 12 zero-fills.
            const auto selectors=DirectCompute::constant32(i,i.src[2]);
            if (!selectors) refuse(pc,"dynamic mesh byte permutation");
            const auto a=alu.operand32(i,i.src[0]), b=alu.operand32(i,i.src[1]);
            string result="0";
            for (unsigned j=0;j<4;++j) {
                const unsigned sel=(*selectors>>(j*8))&255;
                string byte;
                if (sel<8) byte=ir.binary("and",ir.binary("lshr",sel<4 ? b : a,to_string((sel&3)*8)),"255");
                else if (sel<12) {
                    const auto bit=ir.binary("and",ir.binary("lshr",sel<10 ? b : a,to_string((sel&1)*16+15)),"1");
                    byte=alu.select(alu.cmp("ne",bit,"0"),"255","0");
                } else byte=sel==12 ? "0" : "255";
                result=ir.binary("or",result,ir.binary("shl",byte,to_string(j*8)));
            }
            alu.write(i.dst,result);
            if (!full_exec) partial_writes.insert(i.dst);
        } else if ((full_exec || exec_skip) && (i.name=="v_trunc_f32" || i.name=="v_floor_f32" ||
                    i.name=="v_ceil_f32" || i.name=="v_rndne_f32" || i.name=="v_fract_f32" ||
                    i.name=="v_cvt_u32_f32" || i.name=="v_mul_u32_u24" || i.name=="v_lshl_or_b32" ||
                    i.name=="v_add_lshl_u32" || i.name=="v_lshrrev_b64" || i.name=="v_lshlrev_b64" || i.name=="v_mad_legacy_f32" ||
                    i.name=="v_mul_lo_u32" || i.name=="v_mac_f32" || i.name=="v_mul_legacy_f32" || i.name=="v_sub_f32" ||
                    i.name=="v_subrev_f32" || i.name=="v_fmac_f32" || i.name=="v_fmaak_f32" ||
                    i.name=="v_rsq_f32" || i.name=="v_rcp_f32" || i.name=="v_sin_f32" || i.name=="v_cos_f32" ||
                    i.name.substr(0,15)=="v_cvt_f32_ubyte" || i.name=="v_min_u32" || i.name=="v_max_f32" ||
                    i.name=="v_mad_u32_u24" || i.name=="v_add3_u32" || i.name=="v_bfe_u32" || i.name=="v_mov_b32" || i.name=="v_add_nc_u32" || i.name=="v_lshlrev_b32" ||
                    i.name=="v_lshrrev_b32" || i.name=="v_or_b32" || i.name=="v_and_b32" ||
                    i.name=="v_xor_b32" || i.name=="v_lshl_add_u32" || i.name=="v_cvt_f32_u32" || i.name=="v_cvt_f32_i32" ||
                    i.name=="v_mul_f32" || i.name=="v_add_f32" || i.name=="v_fma_f32")) {
            const bool flat=i.name=="v_lshl_add_u32" && alu.operand32(i,i.src[0])==wave_number &&
                            DirectCompute::constant32(i,i.src[1])==(layout.wave32 ? 5u : 6u) &&
                            whole_lane.count(alu.operand32(i,i.src[2]));
            alu.arithmetic(i,state);
            if (flat) linear_threads.insert(alu.read(i.dst));
            if (!full_exec) {
                partial_writes.insert(i.dst);
                if (i.name=="v_lshrrev_b64" || i.name=="v_lshlrev_b64") partial_writes.insert(i.dst+1);
            }
        } else refuse(pc,"unsupported mesh instruction or EXEC state: "+string(op.name));
        for (unsigned j=0;j<4;++j) {
            if (!before_vec[j].empty() && !before_partial[j] && partial_writes.count(vec_dst+j)) {
                auto &value=alu.graphics_registers.at(vec_dst+j);
                auto u=uniform(value);
                value=alu.select(lane_active,value,before_vec[j]);
                if (!u.empty()) active_uniform[value]={lane_active,u};
                partial_writes.erase(vec_dst+j);
            }
        }
        if (result_bound && *result_bound<UINT32_MAX && i.dst<106) {
            auto it=alu.graphics_registers.find(i.dst);
            if (it!=alu.graphics_registers.end() && !it->second.empty()) upper_bound[it->second]=*result_bound;
        }
        if (i.dst<106 && i.enc!=Encoding::Smem) remember_scalar(i.dst);
        if (i.name=="v_mov_b32" && i.src[0]<106) remember_scalar(i.src[0]);
        if (i.dst==261 || (i.enc==Encoding::Mubuf && i.dst==5)) exchanged_group.clear();
        pc=i.next;
    }
    refuse(pc,"unterminated mesh shader");
}
