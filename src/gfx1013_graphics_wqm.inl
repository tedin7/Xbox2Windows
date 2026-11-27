// A numeric SGPR mask contains every guest lane, independently of the
// desktop wave size. A raster coverage mask instead has one known bit per
// invocation. Keep these two representations distinct.
// See victory-shader 9376fd6, structure.rs (Atom::Quad) and emit/lower.rs
// (wave_wide). The raster path below requires a quad-local live arm.
struct PixelWqm {
    std::set<unsigned> packed, packed_reads, packed_restores;
    std::set<unsigned> filters, quad_guards;
    unsigned exact_restore = UINT_MAX;
    string survivor;
    // Cross-quad EXECZ (#2907): each write of the branch mask, taken from the
    // filtered coverage or the whole-quad EXEC, or narrowing the mask again.
    enum class MaskSource { survivors, whole_quad, narrowed };
    std::map<unsigned, MaskSource> vote_masks;
    unsigned vote_wqm = UINT_MAX, vote_skip = UINT_MAX;
    string vote_lane;

    static bool local(const Inst &i) {
        return GraphicsMaskRegions::local_alu(i) || i.name == "s_waitcnt" || i.name == "s_nop" ||
               (i.dpp_control && !i.mirror_mask() && i.dst >= 256);
    }
    static bool is_wqm(const Inst &i, bool narrow) { return i.name == (narrow ? "s_wqm_b32" : "s_wqm_b64"); }
    // Within a scalar-only diamond the lane bits can be inspected without
    // assigning any desktop pixel a guest lane index. The original EXEC must
    // be restored before committing vector results.
    void numeric_guards(Program &p, bool narrow) {
        const auto mov = narrow ? "s_mov_b32" : "s_mov_b64";
        std::multimap<unsigned, unsigned> incoming;
        for (const auto &op : p.code)
            if (op.branch())
                incoming.emplace(op.target(), op.pc);
        for (size_t n = 0; n + 2 < p.code.size(); ++n) {
            const auto &wqm = p.code[n];
            auto &skip = p.code[n + 1];
            if (!is_wqm(wqm, narrow) || wqm.dst != 126 || wqm.src[0] >= 106 || !skip.branch() ||
                (skip.op != 8 && skip.op != 9) || skip.target() <= skip.pc)
                continue;
            size_t end = n + 2;
            std::vector<unsigned> reads;
            for (; end < p.code.size(); ++end) {
                const auto &i = p.code[end];
                if (i.name == mov && i.dst == 126)
                    break;
                if (i.name == mov && i.src[0] == 126 && i.dst < 106) {
                    reads.push_back(i.pc);
                    continue;
                }
                if (i.branch() && i.target() > i.pc && (i.op == 2 || i.op == 4 || i.op == 5))
                    continue;
                if (i.name == "s_nop")
                    continue;
                if ((i.enc != Encoding::Sop1 && i.enc != Encoding::Sop2 && i.enc != Encoding::Sopc &&
                     i.enc != Encoding::Sopk) ||
                    i.dst >= 106 || i.name.find("exec") != Name::npos ||
                    std::any_of(i.src.begin(), i.src.end(), [](unsigned r) { return r == 126 || r == 127; }))
                    break;
            }
            if (end == p.code.size())
                continue;
            const auto &restore = p.code[end];
            if (restore.name != mov || restore.dst != 126 || restore.src[0] >= 106 || skip.target() > restore.pc)
                continue;
            bool closed = true;
            for (auto it = incoming.upper_bound(wqm.pc); it != incoming.end() && it->first <= restore.pc; ++it)
                if (it->second < wqm.pc || it->second >= restore.pc)
                    closed = false;
            for (size_t k = n + 1; k < end; ++k)
                if (p.code[k].branch() && p.code[k].target() > restore.pc)
                    closed = false;
            if (!closed)
                continue;
            packed.insert(wqm.pc);
            packed_reads.insert(reads.begin(), reads.end());
            packed_restores.insert(restore.pc);
            // SCC from WQM is exactly (lo | hi) != 0. No instruction
            // between WQM and the branch can overwrite SCC.
            skip.op = skip.op == 8 ? 4 : 5;
            skip.name = skip.op == 4 ? "s_cbranch_scc0" : "s_cbranch_scc1";
        }
    }
    void raster_guard(Program &p, bool narrow) {
        auto &code = p.code;
        if (code.size() < 8)
            return;
        const size_t null = code.size() - 2;
        if (code[null].enc != Encoding::Export || (code[null].a & 0x1fff) != 0x1890 ||
            code.back().name != "s_endpgm" || code[null - 1].name != "s_endpgm")
            return;
        std::vector<size_t> guards, updates;
        unsigned mask = UINT_MAX;
        for (size_t k = 0; k < null; ++k) {
            if (!code[k].branch())
                continue;
            if (k < 2 || code[k].op != 8 || code[k].target() != code[null].pc)
                return;
            const auto &wqm = code[k - 1];
            if (!is_wqm(wqm, narrow) || wqm.dst != 126 || wqm.src[0] > (narrow ? 105u : 104u) ||
                (!narrow && (wqm.src[0] & 1)) || (mask != UINT_MAX && mask != wqm.src[0]))
                return;
            mask = wqm.src[0];
            size_t f = k - 1;
            while (f > 0 && local(code[f - 1]))
                --f;
            if (!f)
                return;
            const auto &keep = code[f - 1];
            if (keep.name != (narrow ? "s_andn2_b32" : "s_andn2_b64") || keep.dst != mask || keep.src[0] != mask ||
                keep.src[1] != 106)
                return;
            if (!guards.empty()) {
                // Only intact quads reach another filter. Comparing again
                // can remove more covered lanes, never revive a dead quad.
                for (size_t j = guards.back() + 1; j < f - 1; ++j)
                    if (!local(code[j]) && !(code[j].enc == Encoding::Vopc && code[j].dst == 106 &&
                                             code[j].name.substr(0, 6) == "v_cmp_"))
                        return;
            }
            guards.push_back(k);
            updates.push_back(f - 1);
        }
        if (guards.empty())
            return;
        size_t r = guards.back() + 1;
        while (r < null - 1 && local(code[r]))
            ++r;
        if (r >= null - 1 || code[r].name != (narrow ? "s_mov_b32" : "s_mov_b64") || code[r].dst != 126 ||
            code[r].src[0] != mask)
            return;
        for (size_t j = r + 1; j < null - 1; ++j) {
            const auto &i = code[j];
            if (!(GraphicsMaskRegions::local_alu(i) || i.name == "s_waitcnt" ||
                  (i.enc == Encoding::Export && ((i.a >> 4) & 63) < 8)))
                return;
        }
        // Each surviving quad runs the same finite local instructions as in
        // the nonempty guest wave. An empty quad has no observable result
        // on either edge: the only exports use the restored survivor bits.
        for (size_t j = 0; j < guards.size(); ++j) {
            filters.insert(code[updates[j]].pc);
            quad_guards.insert(code[guards[j] - 1].pc);
            GraphicsMaskRegions::skip_proven(code[guards[j]]);
        }
        exact_restore = code[r].pc;
        p.at.erase(code[null].pc);
        p.at.erase(code.back().pc);
        code.resize(null);
    }
    // A scalar instruction that neither reads nor writes EXEC or VCC.
    static bool plain_scalar(const Inst &i) {
        if (i.enc != Encoding::Sop1 && i.enc != Encoding::Sop2 && i.enc != Encoding::Sopc &&
            i.enc != Encoding::Sopk)
            return false;
        return (i.enc == Encoding::Sopc || i.dst < 106) && i.name.find("exec") == Name::npos &&
               std::none_of(i.src.begin(), i.src.end(), [](unsigned r) { return r >= 106 && r <= 127; });
    }
    // Cross-quad EXECZ under WQM (#2907):
    //   s_mov sE, exec; s_wqm exec, exec
    //   one or more coverage filters  v_cmp_* vcc; s_andn2 sE, sE, vcc
    //   v_cmp_* vcc; s_andn2 sM, sX, vcc     sX = sE (survivors) or exec (whole quads)
    //   s_wqm exec, sM; s_cbranch_execz J    the region writes only scalars
    //   J: s_mov exec, sE                    then a terminal colour export
    // Lanes are skipped only when EXEC is zero in the whole guest wave, so the
    // branch is a vote over the wave. It is lowered over the desktop wave,
    // under the #2835 packing contract: a correct program cannot depend on
    // which quads share a guest wave. The vote reads EXEC as modelled at the
    // branch, OR-ed across the quad, so helper lanes count as on the guest.
    void raster_vote(Program &p, bool narrow) {
        auto &code = p.code;
        const auto mov = narrow ? "s_mov_b32" : "s_mov_b64", andn2 = narrow ? "s_andn2_b32" : "s_andn2_b64";
        const unsigned last_reg = narrow ? 105u : 104u;
        auto aligned = [&](unsigned r) { return r <= last_reg && (narrow || !(r & 1)); };
        auto overlaps = [&](const Inst &i, unsigned r) {
            const unsigned width = i.name.find("_b64") != Name::npos ? 2 : 1;
            return i.enc != Encoding::Sopc && i.dst < r + (narrow ? 1u : 2u) && r < i.dst + width;
        };
        auto compare = [](const Inst &i) {
            return i.enc == Encoding::Vopc && i.dst == 106 && i.name.substr(0, 6) == "v_cmp_" &&
                   i.name.find("class") == Name::npos;
        };
        if (!filters.empty() || code.size() < 10 || code[0].name != mov || code[0].src[0] != 126 ||
            !aligned(code[0].dst) || !is_wqm(code[1], narrow) || code[1].dst != 126 || code[1].src[0] != 126)
            return;
        const unsigned entry = code[0].dst;
        size_t n = 2;
        while (n + 1 < code.size() && !code[n].branch() && !(is_wqm(code[n], narrow) && code[n].dst == 126))
            ++n;
        if (n + 1 >= code.size() || code[n].branch() || !code[n + 1].branch() || code[n + 1].op != 8 ||
            code[n + 1].target() <= code[n + 1].pc)
            return;
        const auto &wqm = code[n];
        auto &skip = code[n + 1];
        if (!aligned(wqm.src[0]) || wqm.src[0] == entry)
            return;
        const unsigned mask = wqm.src[0];
        // Before the vote: filters on sE, the mask write and lane-local work.
        std::vector<unsigned> filter_pcs;
        std::map<unsigned, MaskSource> mask_writes;
        bool vcc_from_compare = false;
        for (size_t k = 2; k < n; ++k) {
            const auto &i = code[k];
            if (compare(i)) {
                vcc_from_compare = true;
                continue;
            }
            if (i.name == andn2 && i.src[1] == 106) {
                if (!vcc_from_compare)
                    return;
                if (i.dst == entry && i.src[0] == entry && mask_writes.empty())
                    filter_pcs.push_back(i.pc);
                else if (i.dst == mask && mask_writes.empty() && (i.src[0] == entry || i.src[0] == 126))
                    mask_writes.emplace(i.pc, i.src[0] == 126 ? MaskSource::whole_quad : MaskSource::survivors);
                else if (i.dst == mask && i.src[0] == mask && !mask_writes.empty())
                    mask_writes.emplace(i.pc, MaskSource::narrowed);
                else
                    return;
                vcc_from_compare = false;
                continue;
            }
            if (local(i) && !(i.dst < 256 && (i.dst == 106 || i.dst == 107)))
                continue;
            if (plain_scalar(i) && !overlaps(i, entry) && !overlaps(i, mask))
                continue;
            return;
        }
        if (filter_pcs.empty() || mask_writes.empty())
            return;
        // The region: from here on the shape is claimed, so refuse by name.
        size_t r = n + 2;
        for (; r < code.size() && !(code[r].name == mov && code[r].dst == 126); ++r) {
            const auto &i = code[r];
            if (i.name == "s_nop" || i.name == "s_waitcnt")
                continue;
            if (i.branch())
                refuse(i.pc, "cross-quad EXECZ region contains a branch");
            if (i.name.find("s_setreg") == 0 || i.name.find("s_getreg") == 0)
                refuse(i.pc, "cross-quad EXECZ region accesses a hardware register with " + string(i.name));
            if (!plain_scalar(i))
                refuse(i.pc, "cross-quad EXECZ region may only write scalars; " + string(i.name) +
                                 " depends on the lanes the vote leaves out");
            if (overlaps(i, entry))
                refuse(i.pc, "cross-quad EXECZ region overwrites the saved coverage");
        }
        if (r == code.size() || code[r].src[0] != entry || skip.target() != code[r].pc)
            refuse(skip.pc, "cross-quad EXECZ must branch to the exact coverage restore");
        for (size_t j = r + 1; j < code.size(); ++j) {
            const auto &i = code[j];
            if (i.name == "s_endpgm" && j + 1 == code.size())
                break;
            if (!(GraphicsMaskRegions::local_alu(i) || i.name == "s_waitcnt" ||
                  (i.enc == Encoding::Export && ((i.a >> 4) & 63) < 8)))
                refuse(i.pc, "cross-quad EXECZ needs a terminal colour export after the restore");
        }
        for (const auto &i : code)
            if (i.branch() && i.pc != skip.pc)
                refuse(i.pc, "cross-quad EXECZ program has another branch");
        filters.insert(filter_pcs.begin(), filter_pcs.end());
        vote_masks.insert(mask_writes.begin(), mask_writes.end());
        vote_wqm = wqm.pc;
        vote_skip = skip.pc;
        exact_restore = code[r].pc;
        // SCC carries the vote; it is uniform over the desktop wave.
        skip.op = 4;
        skip.name = "s_cbranch_scc0";
    }
    string expand(DxilModule &ir, const string &mask) const {
        auto pair = ir.binary("or", mask, ir.binary("lshr", mask, "1"));
        auto nibble = ir.binary("or", pair, ir.binary("lshr", pair, "2"));
        return ir.binary("mul", ir.binary("and", nibble, "286331153"), "15");
    }
    bool emit_packed(DirectCompute &alu, const Inst &i, bool narrow) const {
        if (packed_restores.count(i.pc)) {
            auto tag = alu.graphics_registers.find(512 + i.src[0]);
            if (tag == alu.graphics_registers.end() || tag->second != "0")
                refuse(i.pc, "numeric EXEC region requires saved raster entry coverage");
            alu.write(126, {});
            if (!narrow)
                alu.write(127, {});
            return true;
        }
        if (packed_reads.count(i.pc)) {
            if (!narrow && ((i.dst & 1) || i.dst > 104))
                refuse(i.pc, "unaligned EXEC read");
            alu.write(i.dst, alu.read(126));
            if (!narrow)
                alu.write(i.dst + 1, alu.read(127));
            return true;
        }
        const bool to_exec = packed.count(i.pc) != 0;
        const bool word = i.name == "s_wqm_b32";
        if (!to_exec && ((!word && i.name != "s_wqm_b64") || i.dst >= 106))
            return false;
        if (!alu.graphics_predicate.empty())
            refuse(i.pc, "WQM scalar result inside a lane branch");
        if (i.src[0] >= 106 || (!word && ((i.src[0] | i.dst) & 1)))
            refuse(i.pc, "numeric WQM requires an aligned general register source");
        auto low = expand(alu.ir, alu.operand32(i, i.src[0]));
        auto high = word ? string{} : expand(alu.ir, alu.operand32(i, i.src[0] + 1));
        auto either = word ? low : alu.ir.binary("or", low, high);
        alu.write(i.dst, low);
        if (!word)
            alu.write(i.dst + 1, high);
        alu.write(128, alu.ir.value("zext i1 " + alu.cmp("ne", either, "0") + " to i32"));
        return true;
    }
};
