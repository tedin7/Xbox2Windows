// Single-block read-only descriptor waterfalls. See victory-shader/resource.rs
// (waterfall) and tools/startup/gfx2hlsl.py (waterfall_refusal). Every pass
// removes exactly the lanes it sampled. No other loop effect can be projected
// onto an individual invocation.
struct PixelWaterfalls {
    struct Access { unsigned key, descriptor_pc; bool key_is_bytes; };
    std::map<unsigned,Access> accesses;
    std::map<unsigned,std::vector<unsigned>> poisoned;
    std::set<unsigned> exits, saves, restores;

    void find(Program &p, bool wave32) {
        auto &code=p.code;
        const auto mov=wave32 ? "s_mov_b32" : "s_mov_b64";
        const auto remaining_op=wave32 ? "s_andn2_wrexec_b32" : "s_andn2_wrexec_b64";
        auto idle=[](const Inst &i) {return i.name=="s_waitcnt" || i.name=="s_nop";};
        auto pair=[&](unsigned r) {return wave32 ? r<=105 : r<=104 && !(r&1);};
        for (size_t tail=2; tail+1<code.size(); ++tail) {
            auto &retry=code[tail];
            const auto &advance=code[tail-1], &restore=code[tail+1];
            if (!retry.branch() || (retry.op!=5 && retry.op!=9) || retry.target()>=retry.pc ||
                advance.name!=remaining_op || !pair(advance.dst) || advance.src[0]!=advance.dst ||
                restore.name!=mov || restore.dst!=126 || !pair(restore.src[0]) || restore.src[0]==advance.dst) continue;
            const size_t head=p.at.at(retry.target());
            std::vector<size_t> body;
            for (size_t j=head;j<tail-1;++j) if (!idle(code[j])) body.push_back(j);
            if (body.size()!=4 && body.size()!=5) continue;
            const auto &read=code[body[0]], &cmp=code[body[1]], &load=code[body[body.size()-2]],
                       &image=code[body.back()];
            if (read.name!="v_readfirstlane_b32" || read.src[0]<256 || read.subword || read.dpp_control ||
                (cmp.name!="v_cmpx_eq_i32" && cmp.name!="v_cmpx_eq_u32") || cmp.dst!=126 ||
                cmp.src[0]!=read.dst || cmp.src[1]!=read.src[0] || cmp.abs || cmp.neg || cmp.scale ||
                cmp.clamp || cmp.subword || cmp.dpp_control || load.enc!=Encoding::Smem || load.op!=3 ||
                load.dst>98 || (load.b>>25)!=read.dst || (load.b&0xfffff) ||
                image.enc!=Encoding::Image || (image.op!=32 && image.op!=39 && image.op!=0) ||
                ((image.b>>16)&31)*4!=load.dst) continue;
            const auto &table=p.before[body[body.size()-2]].at(load.src[0]);
            if (table.type!=Symbol::Type::table || table.value!=compute_resource_heap) continue;
            // The slot or the already shifted byte offset may be the key.
            bool bytes=body.size()==4;
            if (!bytes) {
                const auto &shift=code[body[2]];
                if (shift.name!="s_lshl_b32" || shift.dst!=read.dst || shift.src[0]!=read.dst ||
                    !(source(shift,shift.src[1],p.before[body[2]])==Symbol{Symbol::Type::constant,5})) continue;
            }
            const unsigned width=wave32 ? 1u : 2u;
            auto overlaps=[](unsigned a,unsigned na,unsigned b,unsigned nb) {return a<b+nb && b<a+na;};
            if (overlaps(read.dst,1,advance.dst,width) || overlaps(read.dst,1,restore.src[0],width) ||
                overlaps(load.dst,8,advance.dst,width) || overlaps(load.dst,8,restore.src[0],width) ||
                overlaps(load.src[0],2,load.dst,8) || overlaps(load.src[0],2,read.dst,1) ||
                overlaps(load.dst,8,read.dst,1)) continue;
            // Copies of exactly the entering EXEC must reach both the retry and
            // the restore. No alternate edge may enter past either copy.
            auto anchor=[&](unsigned r)->std::optional<size_t> {
                for (size_t at=head; at;) {
                    const auto &save=code[--at];
                    if (save.name!=mov || save.dst!=r || save.src[0]!=126) continue;
                    if (!GraphicsMaskRegions::preserves_exec(p,at+1,head,r)) return {};
                    for (const auto &edge:code) {
                        if (!edge.branch() || edge.pc==retry.pc) continue;
                        if (edge.target()>save.pc && edge.target()<=restore.pc &&
                            (edge.pc<save.pc || edge.pc>=retry.pc || edge.target()>code[head].pc)) return {};
                    }
                    return at;
                }
                return {};
            };
            auto parent=anchor(restore.src[0]), left=anchor(advance.dst);
            if (!parent || !left) continue;
            saves.insert(code[*parent].pc);
            saves.insert(code[*left].pc);
            restores.insert(restore.pc);
            auto forget=[&](unsigned pc,unsigned r,unsigned n) {
                for (unsigned j=0;j<n;++j) poisoned[pc].push_back(r+j);
            };
            // Keep the actual saves in the instruction stream. They can have
            // consumers earlier in the shader; only this validated restore
            // is an identity on the original coverage.
            forget(read.pc,read.dst,1);
            forget(load.pc,load.dst,8);
            forget(advance.pc,advance.dst,width);
            exits.insert(advance.pc);
            accesses.emplace(image.pc,Access{read.src[0],load.pc,bytes});
            for (size_t j=head;j<=tail;++j)
                if (code[j].pc!=image.pc) GraphicsMaskRegions::skip_proven(code[j]);
            // The final iteration's uniform descriptor cannot escape the loop
            // as a varying handle. Later genuine loads have a new definition.
            for (size_t j=tail+2;j<code.size();++j)
                for (auto &sym:p.before[j]) if (sym.type==Symbol::Type::indexed_descriptor && sym.definition==load.pc)
                    sym.type=Symbol::Type::unbound;
        }
    }
};
