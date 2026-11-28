// Graphics stages share decoded operands, scalar provenance,
// vector arithmetic, memory operations and binary resource declarations with CS.
// Signature rows describe how native exports meet the desktop rasterizer.
struct GraphicsStage {
    const GuestShaderStage &stage;
    DirectCompute &alu;
    bool pixel;
    std::map<unsigned, unsigned> exported;
    unsigned pc = 0;

    struct Interpolation {
        unsigned attribute, channel, source, mode;
    };
    std::map<unsigned, Interpolation> partial;
    std::set<unsigned> overwritten_inputs;
    std::map<unsigned, string> branch_inputs;
    // SPI_PS_INPUT_ADDR order and DXIL interpolation modes, following
    // victory-shader (9376fd6), io.rs and emit/lower.rs::barycentric.
    static constexpr unsigned bary_modes[] = {6, 2, 3, 0, 7, 4, 5};
    const string &incoming_input(unsigned reg) {
        auto [it, new_value] = branch_inputs.try_emplace(reg);
        if (!new_value || reg < 256 || reg >= 512)
            return it->second;
        // An input untouched on one predecessor still has its entry value.
        // Its load must dominate every edge of the phi at the join.
        alu.ir.setup.swap(alu.ir.body);
        try {
            it->second = input(reg);
        } catch (const UnsupportedComputeShader &) {
            // A temporary may legitimately be undefined on a dead edge.
            // A subsequent read of this merged register reports the refusal.
        }
        alu.ir.setup.swap(alu.ir.body);
        return it->second;
    }
    void interpolate(Word w) {
        unsigned op = (w >> 16) & 3, attr = (w >> 10) & 63, channel = (w >> 8) & 3, src = w & 255,
                 dest = 256 + ((w >> 18) & 255);
        if (!pixel)
            refuse(pc, "interpolation outside pixel stage");
        const auto it = std::find_if(stage.attributes.begin(), stage.attributes.end(),
                                     [&](auto &a) { return a.location == attr && (a.mask & (1u << channel)); });
        if (it == stage.attributes.end() || it->usage != 6 || (it->flags & 0x18))
            refuse(pc, "missing or packed interpolation attribute");
        unsigned mode = 1;
        if (op == 2) {
            if (src != 2 || !(it->flags & 1))
                refuse(pc, "unsupported provoking vertex or smooth flat fetch");
        } else {
            unsigned base = 0, first = UINT_MAX;
            for (unsigned bit = 0; bit < 7; ++bit)
                if (stage.pixel_input_address & (1u << bit)) {
                    if ((stage.pixel_input_enable & (1u << bit)) && src == base + (op == 1)) {
                        mode = bary_modes[bit];
                        first = base;
                        break;
                    }
                    base += bit == 3 ? 3 : 2;
                }
            if (op > 1 || !mode || first == UINT_MAX || (it->flags & 1) || overwritten_inputs.count(256 + src))
                refuse(pc, "interpolation barycentric source is not an entry register");
            if (op == 0) {
                partial[dest] = {attr, channel, src, mode};
                alu.graphics_registers.erase(dest);
                alu.guarded_ij.erase(dest);
                overwritten_inputs.insert(dest);
                return;
            }
            const auto prev = partial.find(dest);
            if (prev == partial.end() || prev->second.attribute != attr || prev->second.channel != channel ||
                prev->second.source != first || prev->second.mode != mode)
                refuse(pc, "unmatched interpolation pair");
        }
        partial.erase(dest);
        alu.write(dest, load(generic(true, it->row, channel, mode), channel));
        overwritten_inputs.insert(dest);
    }
    unsigned system(const string &name) {
        if (name == "SV_Position")
            return 3;
        if (name == "SV_VertexID")
            return 1;
        if (name == "SV_InstanceID")
            return 2;
        if (name == "SV_Target")
            return 16;
        if (name.rfind("SV_", 0) == 0)
            refuse(pc, "unsupported graphics system semantic " + name);
        return 0;
    }
    unsigned row(bool input, const string &name, unsigned index, unsigned kind, unsigned component, unsigned reg,
                 unsigned mask, unsigned interpolation) {
        auto &rows = input ? alu.ir.inputs : alu.ir.outputs;
        const bool unpacked = pixel && reg == UINT_MAX && (input ? kind == 28 && mask == 7 : mask == 1);
        if ((reg >= 32 && !unpacked) || !mask || mask > 15)
            refuse(pc, "invalid signature register or component mask");
        unsigned first = 0, width = 0;
        while (!(mask & (1u << first)))
            ++first;
        while (mask & (1u << (first + width)))
            ++width;
        if (mask != ((1u << width) - 1) << first)
            refuse(pc, "noncontiguous signature channels");
        for (unsigned j = 0; j < rows.size(); ++j) {
            const auto &r = rows[j];
            if (r.name == name && r.semantic_index == index) {
                if (r.row != reg || r.first != first || r.width != width || r.interpolation != interpolation ||
                    r.component != component || r.kind != kind)
                    refuse(pc, "inconsistent graphics signature row");
                return j;
            }
            if (reg != UINT_MAX && r.row == reg && mask & (((1u << r.width) - 1) << r.first))
                refuse(pc, "overlapping graphics signature rows");
        }
        rows.push_back({name, index, kind, component, reg, first, width, interpolation});
        return static_cast<unsigned>(rows.size() - 1);
    }
    unsigned generic(bool input, unsigned reg, unsigned channel, unsigned interpolation) {
        const auto &signature = input ? stage.input_signature : stage.output_signature;
        for (const auto &s : signature) {
            if (s.row != reg || !(s.mask & (1u << channel)) || s.system)
                continue;
            unsigned component = s.component == 1 ? 5 : s.component == 2 ? 4 : s.component == 3 ? 9 : 0;
            if (!component)
                refuse(pc, "unknown signature component");
            return row(input, s.name, s.index, system(s.name), component, s.row, s.mask,
                       !input && component != 9 ? 1 : interpolation);
        }
        refuse(pc, "missing generic graphics signature row");
    }
    string load(unsigned id, unsigned channel) {
        auto &s = alu.ir.inputs.at(id);
        const unsigned col = channel - s.first;
        const bool fp = s.component == 9;
        auto v = alu.ir.invoke(fp ? "float" : "i32", fp ? "loadInput.f32" : "loadInput.i32", 4,
                               {{"i32", to_string(id)}, {"i32", "0"}, {"i8", to_string(col)}, {"i32", "undef"}},
                               "nounwind readnone");
        alu.ir.dependencies[v].set(s.row == UINT_MAX ? 128 : s.row * 4 + channel);
        s.used |= 1u << col;
        return fp ? alu.ir.value("bitcast float " + v + " to i32") : v;
    }
    string input(unsigned reg) {
        if (partial.count(reg))
            refuse(pc, "intermediate interpolation value used outside its pair");
        if (reg < 256)
            refuse(pc, "undefined graphics register " + to_string(reg));
        unsigned vgpr = reg - 256;
        if (pixel) {
            // SPI_PS_INPUT_ADDR allocates enabled inputs consecutively. The
            // pull-model barycentrics consume three VGPRs and line stipple one.
            unsigned first = 0;
            for (unsigned bit = 0; bit < 16; ++bit) {
                if (!(stage.pixel_input_address & (1u << bit)))
                    continue;
                const unsigned width = bit == 3 ? 3 : bit < 7 ? 2 : 1;
                if (vgpr >= first && vgpr < first + width) {
                    if (!(stage.pixel_input_enable & (1u << bit)))
                        break;
                    if (bit < 7 && bary_modes[bit]) {
                        // I/J weight vertices one and two. Distinct guest
                        // pairs may need distinct host system-value elements.
                        const unsigned mode = bary_modes[bit];
                        auto perspective = [](unsigned m) { return m == 2 || m == 3 || m == 6; };
                        unsigned index = 0;
                        for (const auto &r : alu.ir.inputs) {
                            if (r.kind != 28) continue;
                            if (r.interpolation == mode) {
                                index = r.semantic_index;
                                break;
                            }
                            if (perspective(r.interpolation) == perspective(mode))
                                refuse(pc, "two barycentric locations in the same perspective mode");
                            ++index;
                        }
                        alu.ir.features |= 1u << 22;
                        auto id = row(true, "SV_Barycentrics", index, 28, 9, UINT_MAX, 7, mode);
                        return load(id, 1 + vgpr - first);
                    }
                    if (bit == 15) {
                        for (const auto &s : stage.input_signature) {
                            if (s.system != D3D_NAME_POSITION || s.component != 3 || (s.mask & 3) != 3)
                                continue;
                            auto id = row(true,"SV_Position",0,3,9,s.row,s.mask,4);
                            auto coord = [&](unsigned axis) {
                                auto fp = alu.ir.value("bitcast i32 " + load(id,axis) + " to float");
                                return alu.ir.value("fptoui float " + fp + " to i32");
                            };
                            auto x=coord(0), y=coord(1);
                            return alu.ir.binary("or", alu.ir.binary("and",x,"65535"), alu.ir.binary("shl",y,"16"));
                        }
                        refuse(pc,"packed position requires screen XY");
                    }
                    if (bit == 12) {
                        for (const auto &s : stage.input_signature)
                            if (s.system == D3D_NAME_IS_FRONT_FACE && s.mask == 1 && s.component == 1)
                                return load(row(true, "SV_IsFrontFace", 0, 13, 5, s.row, 1, 1), 0);
                        refuse(pc, "front face absent from pixel signature");
                    }
                    if (bit < 8 || bit > 11)
                        break;
                    unsigned channel = bit - 8;
                    for (const auto &signature : stage.input_signature) {
                        if (signature.system != 1 || signature.component != 3 || !(signature.mask & (1u << channel)))
                            continue;
                        if (stage.pixel_input_address & 0x11)
                            refuse(pc, "sample-frequency position input");
                        const bool conservative = std::any_of(stage.output_signature.begin(),
                                                              stage.output_signature.end(), [](const auto &out) {
                                                                  return out.system == D3D_NAME_DEPTH_GREATER_EQUAL ||
                                                                         out.system == D3D_NAME_DEPTH_LESS_EQUAL;
                                                              });
                        return load(
                            row(true, "SV_Position", 0, 3, 9, signature.row, signature.mask, conservative ? 5 : 4),
                            channel);
                    }
                    refuse(pc, "position absent from pixel input signature");
                }
                first += width;
            }
            refuse(pc, "unsupported pixel entry VGPR " + to_string(vgpr));
        }
        if (alu.analysis.result.vertex_fetch)
            for (const auto &a : stage.attributes) {
                if (vgpr < a.location || vgpr - a.location > a.last_component)
                    continue;
                if (a.usage != 6)
                    refuse(pc, "unsupported vertex fetch semantic");
                unsigned channel = vgpr - a.location;
                return load(generic(true, a.row, channel, 0), channel);
            }
        if (vgpr != 0 && vgpr != 3)
            refuse(pc, "unbound vertex VGPR " + to_string(vgpr));
        string name = vgpr ? "SV_InstanceID" : "SV_VertexID";
        unsigned free = 0;
        for (const auto &s : stage.input_signature) {
            if (s.name == name)
                return load(row(true, name, 0, system(name), 5, s.row, 1, 0), 0);
            free = (std::max)(free, s.row + 1);
        }
        for (unsigned id = 0; id < alu.ir.inputs.size(); ++id) {
            const auto &s = alu.ir.inputs[id];
            if (s.name == name && !s.semantic_index && s.kind == system(name))
                return load(id, 0);
            free = (std::max)(free, s.row + 1);
        }
        return load(row(true, name, 0, system(name), 5, free, 1, 0), 0);
    }
    string quad_source(const Inst &i) {
        const unsigned perm = *i.dpp_control;
        string word = alu.operand32(i, i.src[0]);
        if (perm == 0xe4)
            return word;
        alu.ir.uses_wave = true;
        auto pick = [&](unsigned n) {
            return alu.ir.invoke("i32", "quadReadLaneAt.i32", 122, {{"i32", word}, {"i32", to_string(n)}}, "nounwind");
        };
        if (perm == (perm & 3) * 0x55u)
            return pick(perm & 3);
        if (perm == 0xb1 || perm == 0x4e || perm == 0x1b)
            return alu.ir.invoke("i32", "quadOp.i32", 123,
                                 {{"i32", word},
                                  {"i8", perm == 0xb1   ? "0"
                                         : perm == 0x4e ? "1"
                                                        : "2"}},
                                 "nounwind");
        for (const auto &s : stage.input_signature) {
            if (s.system != D3D_NAME_POSITION || s.component != 3 || (s.mask & 3) != 3)
                continue;
            const bool conservative =
                std::any_of(stage.output_signature.begin(), stage.output_signature.end(), [](const auto &out) {
                    return out.system == D3D_NAME_DEPTH_GREATER_EQUAL || out.system == D3D_NAME_DEPTH_LESS_EQUAL;
                });
            auto id = row(true, "SV_Position", 0, 3, 9, s.row, s.mask, conservative ? 5 : 4);
            auto parity = [&](unsigned channel) {
                auto pos = alu.ir.value("bitcast i32 " + load(id, channel) + " to float");
                auto pixel = alu.ir.invoke("float", "unary.f32", 27, {{"float", pos}}, "nounwind readnone");
                return alu.ir.binary("and", alu.ir.value("fptosi float " + pixel + " to i32"), "1");
            };
            auto x = parity(0), y = parity(1);
            auto lane = alu.ir.binary("or", x, alu.ir.binary("shl", y, "1"));
            std::array<string, 4> picked;
            for (unsigned j = 0; j < 4; ++j) {
                unsigned source = (perm >> (2 * j)) & 3;
                if (picked[source].empty())
                    picked[source] = pick(source);
            }
            auto result = picked[(perm >> 6) & 3];
            for (int j = 2; j >= 0; --j)
                result = alu.select(alu.cmp("eq", lane, to_string(j)), picked[(perm >> (2 * j)) & 3], result);
            return result;
        }
        refuse(i.pc, "general pixel quad permutation requires position XY");
    }
    unsigned output(unsigned target, unsigned &channel) {
        if (pixel && target < 8) {
            // An RTV variant types the target by its format. A signature that says otherwise is not reinterpreted.
            const unsigned asked = stage.target_components[target];
            for (const auto &s : stage.output_signature) {
                if (s.system != D3D_NAME_TARGET || s.index != target)
                    continue;
                if (s.row != target || !(s.mask & (1u << channel)) || !s.component || s.component > 3)
                    refuse(pc, "color export incompatible with pixel signature");
                const unsigned component = s.component == 3 ? 9 : s.component == 2 ? 4 : 5;
                if (asked && asked != component)
                    refuse(pc, "RTV variant format disagrees with the pixel signature of target " + to_string(target));
                return row(false, "SV_Target", target, 16, component, s.row, s.mask, 0);
            }
            if (!stage.output_signature.empty())
                refuse(pc, "color target absent from pixel signature");
            return row(false, "SV_Target", target, 16, asked ? asked : 9, target, 15, 0);
        }
        if (pixel && target == 8) {
            for (const auto &s : stage.output_signature) {
                unsigned kind = 0;
                if (!channel && s.system == D3D_NAME_DEPTH)
                    kind = 17;
                if (!channel && s.system == D3D_NAME_DEPTH_LESS_EQUAL)
                    kind = 18;
                if (!channel && s.system == D3D_NAME_DEPTH_GREATER_EQUAL)
                    kind = 19;
                if (channel == 1 && s.system == D3D_NAME_STENCIL_REF)
                    kind = 20;
                if (channel == 2 && s.system == D3D_NAME_COVERAGE)
                    kind = 14;
                if (!kind)
                    continue;
                if (s.row != UINT_MAX || s.mask != 1 || s.index || s.component != (channel ? 1u : 3u))
                    refuse(pc, "invalid depth, stencil or coverage signature");
                if (kind == 20)
                    alu.ir.features |= 2048;
                return row(false, s.name, 0, kind, channel ? 5 : 9, UINT_MAX, 1, 0);
            }
            refuse(pc, "unmapped depth, stencil or coverage export");
        }
        if (!pixel && target == 12) {
            for (const auto &s : stage.output_signature)
                if (s.system == 1 || s.name == "SV_Position")
                    return row(false, "SV_Position", 0, 3, 9, s.row, s.mask, 4);
            if (!stage.outputs.empty())
                refuse(pc, "missing position register");
            return row(false, "SV_Position", 0, 3, 9, 0, 15, 4);
        }
        if (!pixel && (target == 13 || target == 14)) {
            // With no misc position vector, POS1 and POS2 pack clip distances
            // followed by cull distances, in semantic index/component order.
            std::vector<const ShaderSemantic *> distances;
            for (const auto &s : stage.output_signature) {
                if (s.system == D3D_NAME_RENDER_TARGET_ARRAY_INDEX || s.system == D3D_NAME_VIEWPORT_ARRAY_INDEX ||
                    s.name == "PSIZE")
                    refuse(pc, "ambiguous clip/cull position export");
                if (s.system == D3D_NAME_CLIP_DISTANCE || s.system == D3D_NAME_CULL_DISTANCE)
                    distances.push_back(&s);
            }
            std::sort(distances.begin(), distances.end(), [](const auto *a, const auto *b) {
                return std::pair{a->system, a->index} < std::pair{b->system, b->index};
            });
            unsigned index = (target - 13) * 4 + channel;
            for (const auto *s : distances) {
                if (s->component != 3)
                    refuse(pc, "non-float clip/cull distance");
                for (unsigned c = 0; c < 4; ++c)
                    if (s->mask & (1u << c)) {
                        if (index == 0) {
                            channel = c;
                            return row(false, s->name, s->index, s->system == D3D_NAME_CLIP_DISTANCE ? 6 : 7, 9, s->row,
                                       s->mask, 2);
                        }
                        --index;
                    }
            }
            refuse(pc, "unmapped position export channel");
        }
        if (!pixel && target >= 32)
            for (const auto &a : stage.outputs)
                if (a.location == target - 32 && (!a.mask || (a.mask & (1u << channel)))) {
                    if (a.usage != 6 || (a.flags & 3))
                        refuse(pc, "unsupported packed vertex export");
                    return generic(false, a.row, channel, 2);
                }
        refuse(pc, "unmapped graphics export");
    }
    void emit_export(Word first, Word sources) {
        const unsigned target = (first >> 4) & 63, mask = first & 15;
        const bool compressed = (first & 0x400) != 0;
        if ((first & 0x03ffe000) || (compressed && (!pixel || target >= 8 || (mask != 3 && mask != 12 && mask != 15))) ||
            !exported.emplace(target, mask).second)
            refuse(pc, "unsupported graphics export mask or repeated target");
        if (target == 9 || (!pixel && target == 32 && stage.outputs.empty() &&
                            std::none_of(stage.output_signature.begin(), stage.output_signature.end(),
                                         [](const auto &s) { return !s.system; })))
            return;
        if (!mask)
            refuse(pc, "empty enabled graphics export");
        unsigned signature_mask = 15;
        if (!pixel && target >= 32) {
            // PARAM exports have four hardware lanes, including padding when
            // the logical signature row is narrower. Preserve packed semantics
            // belonging to the same row and let stage linkage check all PS reads.
            unsigned channels = 0;
            for (const auto &a : stage.outputs) {
                if (a.location != target - 32 || a.usage != 6 || (a.flags & 3)) continue;
                unsigned declared = 0;
                for (const auto &s : stage.output_signature)
                    if (s.row == a.row && !s.system) declared |= s.mask;
                if (!declared) refuse(pc, "vertex parameter has no output signature");
                channels |= declared;
            }
            if (channels) signature_mask = channels;
        }
        if (pixel && target < 8)
            alu.analysis.result.pixel_writes[target] = mask;
        for (unsigned c = 0; c < 4; ++c)
            if (mask & signature_mask & (1u << c)) {
                auto bits = alu.read(256 + ((sources >> (8 * (compressed ? c / 2 : c))) & 255));
                unsigned channel = c;
                auto id = output(target, channel);
                auto &s = alu.ir.outputs[id];
                if (s.row == UINT_MAX)
                    channel = 0;
                if (channel < s.first || channel >= s.first + s.width)
                    refuse(pc, "export outside signature");
                const bool fp = s.component == 9;
                if (compressed) {
                    if (!fp) refuse(pc, "packed color export requires floating target");
                    auto half = alu.subword(bits, 4 + (c & 1), false);
                    auto value = alu.ir.invoke("float", "legacyF16ToF32", 131, {{"i32",half}}, "nounwind readnone");
                    bits = alu.ir.value("bitcast float " + value + " to i32");
                }
                auto v = fp ? alu.ir.value("bitcast i32 " + bits + " to float") : bits;
                alu.ir.invoke("void", fp ? "storeOutput.f32" : "storeOutput.i32", 5,
                              {{"i32", to_string(id)},
                               {"i32", "0"},
                               {"i8", to_string(channel - s.first)},
                               {fp ? "float" : "i32", v}},
                              "nounwind");
                s.used |= 1u << (channel - s.first);
                if (s.row == UINT_MAX)
                    continue;
                if (alu.ir.loop_dependencies)
                    alu.ir.output_values[s.row * 4 + channel].push_back(v);
                if (auto d = alu.ir.dependencies.find(v); d != alu.ir.dependencies.end())
                    alu.ir.output_dependencies[s.row * 4 + channel] |= d->second;
            }
    }
};

// Acyclic scalar branches preserve the rasterizer's per-invocation SSA values.
// An SGPR comparison can select a path without introducing a guest EXEC mask.
struct GraphicsBlock {
    size_t begin, end;
    std::vector<unsigned> successors;
    unsigned predecessors = 0;
};
// An EXECZ skip over lane-local ALU with an exact restore has a per-lane
// equivalent. Mark that region before building the scalar CFG so the merge
// is a select on each VGPR, including the old value of inactive lanes.
struct GraphicsMaskRegions {
    std::set<unsigned> saves, complements, restores;
    std::set<unsigned> sample_begins, sample_ends;
    std::set<unsigned> compares, cmpx_starts, unions, intersections, inner_saves, inner_restores;
    std::map<unsigned, std::set<unsigned>> expired;
    std::map<unsigned, unsigned> enclosing_mask;
    std::map<unsigned, unsigned> branch_saves;
    std::map<unsigned, std::pair<unsigned,bool>> lane_branches;
    std::map<unsigned, unsigned> branch_merges;
    struct MaskCapture { unsigned skip, value; };
    std::map<unsigned, MaskCapture> captures;
    std::set<unsigned> vertex_anchors, vertex_partitions;
    std::map<unsigned,unsigned> mask_inputs;
    std::set<unsigned> vertex_mask_compares, uniform_mask_tests;
    std::set<unsigned> scalar_mask_selects;
    std::map<unsigned,unsigned> then_fallthroughs;
    std::set<unsigned> scalar_tail_merges;
    static unsigned scalar_width(const Inst &i) {
        if (i.enc == Encoding::Smem) return memory_count(i);
        if (i.enc == Encoding::Vopc) return 2;
        if (i.enc == Encoding::Sop1 || i.enc == Encoding::Sop2 || i.enc == Encoding::Sopk)
            return i.name.find("64") == Name::npos ? 1 : 2;
        return i.name == "v_readfirstlane_b32" || i.name == "v_readlane_b32" ? 1 : 0;
    }
    // Track pairs equal to the initial EXEC, on every path. An inner save
    // followed by a restore may preserve the value even across nested skips.
    static bool preserves_exec(const Program &p, size_t begin, size_t end, unsigned parent = 126) {
        using Pairs = std::bitset<128>;
        std::vector<std::optional<Pairs>> incoming(end-begin+1);
        incoming.front().emplace().set(126).set(parent);
        for (size_t j = begin; j < end; ++j) {
            const auto &at = incoming[j-begin];
            if (!at) continue;
            auto known = *at;
            const auto &i = p.code[j];
            const bool was_exec = known[126];
            const bool copy = i.name == "s_mov_b64" && i.src[0] < 128 && known[i.src[0]];
            auto clobber = [&](unsigned dst,unsigned count) {
                for (unsigned n = dst; n < dst + count && n < 128; ++n)
                    known.reset(n & ~1u);
            };
            clobber(i.dst, scalar_width(i));
            if (auto carry = carry_destination(i)) clobber(*carry,2);
            if (i.name.find("saveexec") != Name::npos) {
                known[126] = false;
                known[i.dst] = was_exec;
            } else if (copy) known[i.dst] = true;
            auto edge = [&](size_t target) {
                if (target <= j) return false;
                if (target > end) return parent != 126;
                auto &destination = incoming[target-begin];
                if (destination) *destination &= known;
                else destination = known;
                return true;
            };
            if (i.branch() && !edge(p.at.at(i.target()))) return false;
            if ((!i.branch() || i.op != 2) && !edge(j+1)) return false;
        }
        return incoming.back() && (*incoming.back())[126] && (*incoming.back())[parent];
    }
    static bool local_alu(const Inst &i) {
        return (i.enc == Encoding::Vop1 || i.enc == Encoding::Vop2 || i.enc == Encoding::Vop3) &&
            i.dst >= 256 && !i.dpp_control && i.name.find("lane") == Name::npos &&
            i.name.find("mbcnt") == Name::npos && i.name.find("_co_") == Name::npos;
    }
    static void skip_proven(Inst &skip) {
        skip.enc = Encoding::Sopp;
        skip.op = 0;
        skip.name = "s_nop"; // VGPR writes use the corresponding lane predicate.
    }
    void vertex_scalar_tails(const Program &p, const std::multimap<unsigned,unsigned> &entries) {
        const auto &code = p.code;
        for (size_t k=0; k+2<code.size(); ++k) {
            const auto &mask=code[k], &skip=code[k+1];
            if (!skip.branch() || skip.op != 8 || skip.target() <= skip.next || lane_branches.count(skip.pc)) continue;
            const bool cmpx = mask.enc == Encoding::Vopc && mask.dst == 126 && mask.name.substr(0,7) == "v_cmpx_" &&
                !mask.scale && !mask.clamp && !mask.subword && mask.name.find("class") == Name::npos;
            if (!cmpx && !(mask.name == "s_and_b64" && mask.dst == 126 &&
                ((mask.src[0] == 106 && mask.src[1] == 126) || (mask.src[0] == 126 && mask.src[1] == 106)))) continue;
            const size_t join = p.at.at(skip.target());
            const auto &inv = code[join];
            const bool has_else = inv.name == "s_andn2_b64" && inv.dst == 126 && inv.src[1] == 126;
            size_t tail = join + has_else;
            if (has_else) while (tail < code.size() && local_alu(code[tail])) ++tail;
            size_t end = tail;
            while (end < code.size() && (code[end].enc == Encoding::Smem ||
                code[end].name == "s_waitcnt" || code[end].name == "s_nop")) ++end;
            if (end == code.size() || (!has_else && end == tail)) continue;
            const auto &restore = code[end];
            const unsigned parent = restore.src[0];
            if (restore.name != "s_mov_b64" || restore.dst != 126 || parent > 104 || (parent & 1) ||
                (has_else && inv.src[0] != parent)) continue;
            // An earlier restore may dominate across a uniform diamond.
            // Exclude outside entries and prove that both EXEC and the saved
            // pair retain the anchor's value on all paths reaching the mask.
            size_t anchor = k;
            bool anchored = false;
            while (anchor) {
                const auto &a = code[--anchor];
                if (a.name != "s_mov_b64" || !((a.dst == 126 && a.src[0] == parent) ||
                    (a.dst == parent && a.src[0] == 126))) continue;
                auto entry = entries.upper_bound(a.pc);
                for (; entry != entries.end() && entry->first <= mask.pc; ++entry)
                    if (entry->second < a.pc || entry->second >= mask.pc) break;
                anchored = (entry == entries.end() || entry->first > mask.pc) && preserves_exec(p,anchor+1,k,parent);
                break;
            }
            if (!anchored) continue;
            for (auto entry=entries.upper_bound(mask.pc); entry!=entries.end() && entry->first<=restore.pc; ++entry)
                if (entry->second != skip.pc)
                    refuse(entry->first,"branch enters a vertex EXEC scalar-tail region");
            std::set<unsigned> modified{106,107,128};
            for (size_t j=k+2; j<end; ++j) {
                const auto &op=code[j];
                if (has_else && j==join) continue;
                const bool scalar = op.enc == Encoding::Smem || op.enc == Encoding::Sopc ||
                    ((op.enc == Encoding::Sop1 || op.enc == Encoding::Sop2 || op.enc == Encoding::Sopk) && op.dst<106);
                if (op.name.find("exec") != Name::npos || !(local_alu(op) || scalar || op.name=="s_waitcnt" || op.name=="s_nop" ||
                    (op.enc==Encoding::Mubuf && op.name.find("load")!=Name::npos) ||
                    (op.enc==Encoding::Image && (op.op==0 || op.op==1 || op.op==36 || op.op==39))))
                    refuse(op.pc,"unsupported operation inside a vertex EXEC scalar-tail region");
                const unsigned width = scalar_width(op);
                if (width && op.dst < parent+2 && parent < op.dst+width)
                    refuse(op.pc,"vertex scalar-tail region overwrites its saved EXEC");
                if (j<join)
                    for (unsigned r=op.dst; r<op.dst+width && r<128; ++r) modified.insert(r);
            }
            const unsigned condition = 1024 + mask.pc;
            branch_saves.emplace(mask.pc,condition);
            lane_branches.emplace(skip.pc,std::make_pair(condition,false));
            if (code[anchor].dst == parent && !branch_saves.count(code[anchor].pc) && !saves.count(code[anchor].pc))
                vertex_anchors.insert(code[anchor].pc);
            if (has_else) {
                // Only the vector prefix obeys complementary EXEC. The wave
                // runs the scalar tail even when all ELSE lanes are disabled.
                then_fallthroughs.emplace(code[join-1].pc,code[tail].pc);
                branch_merges.emplace(inv.pc,condition);
                expired[inv.pc].insert(modified.begin(),modified.end());
            }
            branch_merges.emplace(code[tail].pc,condition);
            if (tail != end) scalar_tail_merges.insert(code[tail].pc);
            expired[code[tail].pc].insert(modified.begin(),modified.end());
            branch_merges.emplace(restore.pc,condition);
        }
    }
    explicit GraphicsMaskRegions(Program &program, bool wave32, bool vertex) {
        std::multimap<unsigned, unsigned> entries;
        for (const auto &i : program.code)
            if (i.branch()) entries.emplace(i.target(), i.pc);
        // Only carry SCC into a full numeric mask if its reaching producer
        // compares ordinary scalar operands, with no join or intervening
        // scalar writer. Per-lane predicates cannot establish those 64 bits.
        bool scalar_test = false;
        auto scalar_data = [](unsigned r) { return r <= 105 || (r >= 128 && r <= 248) || r == 255; };
        for (const auto &i : program.code) {
            if (entries.count(i.pc)) scalar_test = false;
            if (i.name == "s_cselect_b64" && scalar_test) scalar_mask_selects.insert(i.pc);
            if (i.enc == Encoding::Sopc)
                scalar_test = (i.name.substr(0,6) == "s_cmp_" || i.name.substr(0,8) == "s_bitcmp") &&
                    scalar_data(i.src[0]) && scalar_data(i.src[1]);
            else if (i.branch() || i.enc == Encoding::Sop1 || i.enc == Encoding::Sop2 || i.enc == Encoding::Sopk)
                scalar_test = false;
        }
        // Follow the selected lane's bit from a compare into an explicit
        // SGPR pair, through VCC intersections, up to its AND_SAVEEXEC.
        // Restrict the chain to one block and the same reaching EXEC.
        if (vertex && !wave32) {
            const auto &code = program.code;
            for (size_t j=0; j<code.size(); ++j) {
                const auto &cmp = code[j];
                if (cmp.enc != Encoding::Vopc || cmp.dst > 104 || (cmp.dst & 1) ||
                    cmp.name.substr(0,6) != "v_cmp_" || cmp.name.find("class") != Name::npos ||
                    cmp.clamp || cmp.scale || cmp.subword) continue;
                unsigned bit = 1024 + cmp.pc;
                std::map<unsigned,unsigned> chain;
                for (size_t k=j+1; k<code.size() && !entries.count(code[k].pc); ++k) {
                    const auto &op = code[k];
                    if (op.name == "s_and_b64" && op.dst == cmp.dst &&
                        op.src[0] == 106 && op.src[1] == cmp.dst) {
                        chain[op.pc] = bit;
                        bit = 1024 + op.pc;
                        continue;
                    }
                    if (op.name == "s_and_saveexec_b64" && op.src[0] == cmp.dst) {
                        vertex_mask_compares.insert(cmp.pc);
                        chain[op.pc] = bit;
                        mask_inputs.insert(chain.begin(),chain.end());
                        break;
                    }
                    const auto width = scalar_width(op);
                    auto clobbers = [&](unsigned r) {return width && op.dst < r+2 && r < op.dst+width;};
                    if (op.branch() || clobbers(126) || clobbers(cmp.dst) || carry_destination(op) ||
                        op.name.find("saveexec") != Name::npos) break;
                }
            }
        }
        for (size_t k = 0; k + 2 < program.code.size(); ++k) {
            auto save = program.code[k];
            const auto mov = wave32 ? "s_mov_b32" : "s_mov_b64";
            const auto &narrow = program.code[k + 1];
            const bool split = save.name == mov && save.src[0] == 126 &&
                narrow.name == (wave32 ? "s_and_b32" : "s_and_b64") && narrow.dst == 126 &&
                ((narrow.src[0] == 126 && narrow.src[1] == 106) || (narrow.src[0] == 106 && narrow.src[1] == 126));
            const bool cmpx = vertex && !wave32 && save.enc == Encoding::Vopc &&
                save.name.substr(0,7) == "v_cmpx_" && save.dst == 126 && !save.subword && !save.scale &&
                !save.clamp && save.name.find("class") == Name::npos;
            // Input is an individual bit proven by an earlier scalar mask
            // intersection. Its numeric SGPR pair remains inaccessible.
            const bool from_mask = mask_inputs.count(save.pc) && save.dst == 126;
            const bool saved_mask = mask_inputs.count(save.pc) && save.name == "s_and_saveexec_b64";
            if (!cmpx && !from_mask && !saved_mask && !split && (save.name != (wave32 ? "s_and_saveexec_b32" : "s_and_saveexec_b64") ||
                           save.src[0] != 106)) continue;
            auto &skip = program.code[k + 1 + split];
            if (!skip.branch() || skip.op != 8 || skip.target() <= skip.next) continue;
            const auto join = program.at.at(skip.target());
            const auto &capture = program.code[join];
            const bool captured = cmpx && capture.name == "s_and_b64" && capture.dst <= 104 && !(capture.dst & 1) &&
                ((capture.src[0] == 106 && capture.src[1] == 126) || (capture.src[0] == 126 && capture.src[1] == 106));
            const auto middle = join + captured;
            if (middle >= program.code.size()) continue;
            const auto &inverse = program.code[middle];
            if (cmpx || from_mask) {
                if ((inverse.name != "s_andn2_b64" && inverse.name != mov) || inverse.dst != 126 ||
                    inverse.src[0] > 104 || (inverse.src[0] & 1)) continue;
                save.dst = inverse.src[0];
                // A dominating save or restore with no subsequent writer
                // identifies the exact parent of this CMPX, including nested
                // regions where the currently executing lane is already active.
                bool parent = false;
                for (size_t back = k; back && !entries.count(program.code[back].pc);) {
                    const auto &prev = program.code[--back];
                    if (prev.name == mov && ((prev.dst == 126 && prev.src[0] == save.dst) ||
                        (prev.dst == save.dst && prev.src[0] == 126))) {
                        parent = true;
                        if (prev.src[0] == 126) vertex_anchors.insert(prev.pc);
                        break;
                    }
                    const unsigned width = scalar_width(prev);
                    auto overlaps = [&](unsigned r) {return width && prev.dst < r+2 && r < prev.dst+width;};
                    if (prev.branch() || overlaps(126) || overlaps(save.dst) || carry_destination(prev) ||
                        prev.name.find("saveexec") != Name::npos) break;
                }
                if (!parent || (captured && capture.dst == save.dst)) continue;
            }
            if (save.dst > (wave32 ? 105u : 104u) || (!wave32 && (save.dst & 1))) continue;
            const bool two_arms = inverse.name == (wave32 ? "s_andn2_b32" : "s_andn2_b64") &&
                inverse.dst == 126 && inverse.src[0] == save.dst && inverse.src[1] == 126;
            if (two_arms && middle + 1 >= program.code.size()) continue;
            auto &else_skip = program.code[two_arms ? middle + 1 : middle];
            if (two_arms && (!else_skip.branch() || else_skip.op != 8 || else_skip.target() <= else_skip.next)) continue;
            const auto end = two_arms ? program.at.at(else_skip.target()) : middle;
            const auto &restore = program.code[end];
            if (restore.name != mov || restore.dst != 126 || restore.src[0] != save.dst) continue;
            const auto begin = k + 2 + split;
            // For a VS with control flow inside the masked region, keep a
            // real lane branch. Victory 9376fd6 ir.rs::without_skips and
            // gfx2hlsl.py's EXECZ analysis both require that scalar effects
            // of a skipped arm cannot escape. Here the branch guards reads
            // and the merge invalidates scalar definitions from either arm.
            const bool branched = cmpx || from_mask || saved_mask || (vertex && std::any_of(program.code.begin()+begin,program.code.begin()+end,
                [&](const Inst &i) {return i.branch() && (!two_arms || i.pc != else_skip.pc);}));
            if (branched) {
                for (auto it = entries.upper_bound(save.pc); it != entries.end() && it->first <= restore.pc; ++it) {
                    const unsigned dst = it->first, src = it->second;
                    if (src == skip.pc || (two_arms && src == else_skip.pc)) continue;
                    const unsigned start = two_arms && src >= else_skip.next ? else_skip.next : skip.next;
                    const unsigned limit = two_arms && src < inverse.pc ? capture.pc : restore.pc;
                    if (src < start || src >= limit || dst < start || dst > limit || dst <= src)
                        refuse(dst, "branch enters or crosses a masked vertex arm");
                }
                std::set<unsigned> first_writes{106,107,128}, second_writes{106,107,128};
                for (size_t j=begin; j<end; ++j) {
                    const auto &v=program.code[j];
                    if (two_arms && (j == middle || j == middle+1)) continue;
                    if ((v.branch() && (v.target() < skip.next || v.target() > restore.pc)) ||
                        v.name == "s_endpgm" || v.enc == Encoding::Export || v.enc == Encoding::Lds ||
                        v.enc == Encoding::Fetch || v.enc == Encoding::FixedFetch ||
                        (v.enc == Encoding::Image && v.op != 0 && v.op != 1 && v.op != 14 &&
                         v.op != 34 && v.op != 36 && v.op != 39 && v.op != 47 && v.op != 71) ||
                        (v.enc == Encoding::Mubuf && v.name.find("load") == Name::npos))
                        refuse(v.pc,"masked vertex arm has an unsupported exit or memory effect");
                    auto &writes = two_arms && j > middle ? second_writes : first_writes;
                    auto record = [&](unsigned dst,unsigned width) {
                        if (width && dst < save.dst + (wave32 ? 1u : 2u) && save.dst < dst + width)
                            refuse(v.pc,"masked vertex arm overwrites its saved EXEC");
                        for (unsigned x = dst; x < dst + width && x < 128; ++x) writes.insert(x);
                    };
                    record(v.dst,scalar_width(v));
                    if (auto carry = carry_destination(v)) record(*carry,2);
                    const auto arm_start = two_arms && j > middle ? middle+2 : begin;
                    if (!wave32 && v.name == "s_and_b64" && v.dst <= 104 && v.src[0] == v.dst &&
                        v.src[1] == 126 && j > arm_start) {
                        const auto &cmp = program.code[j-1];
                        auto uniform_source = [](unsigned s) {
                            return s <= 105 || (s >= 128 && s <= 248) || s == 255;
                        };
                        if (cmp.enc == Encoding::Vopc && cmp.dst == v.dst && cmp.name.substr(0,6) == "v_cmp_" &&
                            cmp.name.find("class") == Name::npos && !cmp.subword && !cmp.clamp && !cmp.scale &&
                            uniform_source(cmp.src[0]) && uniform_source(cmp.src[1]) && !entries.count(v.pc) &&
                            preserves_exec(program,arm_start,j-1)) {
                            vertex_mask_compares.insert(cmp.pc);
                            mask_inputs[v.pc] = 1024 + cmp.pc;
                            uniform_mask_tests.insert(v.pc);
                        }
                    }
                }
                if (two_arms && !preserves_exec(program,begin,join))
                    refuse(capture.pc, captured ? "VCC capture requires the reaching CMPX mask on every path" :
                          "vertex ELSE requires its THEN mask on every path");
                if (captured) {
                    if (!two_arms) refuse(capture.pc,"VCC capture requires a matching ELSE and restore");
                    const unsigned bit = 1024 + capture.pc;
                    captures.emplace(capture.pc, MaskCapture{skip.pc,bit});
                    // Captured pair is numeric poison. A following AND with
                    // VCC may feed another EXEC-only arm through its bit.
                    branch_merges.emplace(capture.pc,1024 + save.pc);
                    if (end+3 < program.code.size() && !second_writes.count(capture.dst) &&
                        !second_writes.count(capture.dst+1)) {
                        const auto &cmp = program.code[end+1], &andv = program.code[end+2], &act = program.code[end+3];
                        if (cmp.enc == Encoding::Vopc && cmp.dst == 106 && andv.name == "s_and_b64" &&
                            andv.dst == capture.dst && andv.src[0] == 106 && andv.src[1] == capture.dst &&
                            act.name == "s_and_b64" && act.dst == 126 && act.src[0] == capture.dst &&
                            act.src[1] == 126 && !entries.count(cmp.pc) && !entries.count(andv.pc) && !entries.count(act.pc)) {
                            mask_inputs[andv.pc] = bit;
                            mask_inputs[act.pc] = 1024 + andv.pc;
                        }
                    }
                }
                // Unique virtual register; no guest write aliases it.
                const unsigned condition = 1024 + save.pc;
                branch_saves.emplace(save.pc,condition);
                lane_branches.emplace(skip.pc,std::make_pair(condition,false));
                if (split) skip_proven(program.code[k+1]);
                if (two_arms) {
                    branch_merges.emplace(inverse.pc, condition);
                    lane_branches.emplace(else_skip.pc,std::make_pair(condition,true));
                    expired[inverse.pc].insert(first_writes.begin(),first_writes.end());
                }
                branch_merges.emplace(restore.pc, condition);
                expired[restore.pc].insert(first_writes.begin(),first_writes.end());
                expired[restore.pc].insert(second_writes.begin(),second_writes.end());
                continue; // inner structured regions still need classification
            }
            for (auto it = entries.upper_bound(save.pc); it != entries.end() && it->first <= restore.pc; ++it)
                if (it->second != skip.pc && (!two_arms || it->second != else_skip.pc))
                    refuse(it->first, "branch enters a masked graphics region");
            for (size_t j = k + 2 + split; j < end; ++j) {
                const auto &body = program.code[j];
                if (two_arms && (j == middle || j == middle + 1)) continue;
                if (body.pc != program.code[j - 1].next || body.next != program.code[j + 1].pc ||
                    !(local_alu(body) || body.name == "s_waitcnt" || body.name == "s_nop" ||
                      (!vertex && body.name == "buffer_atomic_add" && !(body.a & 0x2000)) ||
                      (body.enc == Encoding::Image && body.op == 0x27)))
                    refuse(body.pc, "masked graphics region requires lane-local vector arithmetic");
            }
            saves.insert(save.pc);
            restores.insert(restore.pc);
            if (split) skip_proven(program.code[k + 1]);
            skip_proven(skip);
            if (two_arms) {
                complements.insert(inverse.pc);
                skip_proven(else_skip);
            }
            k = end;
        }
        // GFX10 CMPX changes EXEC alone. With no wave-wide skip these arms
        // still partition a proven enclosing mask. No entry may bypass CMPX.
        const auto &code = program.code;
        for (size_t k = 0; k < code.size(); ++k) {
            const auto &start = code[k];
            auto restore_mask = [wave32](const Inst &i) {
                return i.dst == 126 && ((wave32 && i.name == "s_mov_b32" && i.src[0] <= 105) ||
                    (i.name == "s_mov_b64" && i.src[0] <= 104 && !(i.src[0] & 1)));
            };
            if (!restore_mask(start)) continue;
            size_t end = k + 1;
            bool sampled = false;
            for (; end < code.size(); ++end) {
                const auto &i = code[end];
                if (i.enc == Encoding::Image && i.op >= 0x20 && i.op <= 0x3f) sampled = true;
                else if (i.name != "s_waitcnt" && i.name != "s_nop") break;
            }
            if (!sampled || end == code.size() || !restore_mask(code[end]) || code[end].src[0] == start.src[0]) continue;
            auto entry = entries.upper_bound(start.pc);
            if (entry != entries.end() && entry->first <= code[end].pc) continue;
            sample_begins.insert(start.pc);
            sample_ends.insert(code[end].pc);
            k = end;
        }
        for (size_t k = 0; !wave32 && k < code.size(); ++k) {
            const auto &cmp = code[k];
            if (cmp.enc != Encoding::Vopc || cmp.name.substr(0,7) != "v_cmpx_" ||
                cmp.dst != 126 || cmp.scale || cmp.clamp || cmp.subword) continue;
            auto local = [&](const Inst &i) {
                return local_alu(i) || (i.enc == Encoding::Vopc && i.dst == 106 &&
                    i.name.substr(0,6) == "v_cmp_" && !i.clamp && !i.scale && !i.subword &&
                    i.name.find("class") == Name::npos);
            };
            size_t mid = k + 1;
            while (mid < code.size() && local(code[mid])) ++mid;
            if (mid == code.size()) continue;
            const auto &inverse = code[mid];
            if (inverse.name != "s_andn2_b64" || inverse.dst != 126 || inverse.src[0] > 104 ||
                (inverse.src[0] & 1) || inverse.src[1] != 126) continue;
            size_t end = mid + 1;
            while (end < code.size() && local(code[end])) ++end;
            if (end == code.size()) continue;
            const auto &restore = code[end];
            if (restore.name != "s_mov_b64" || restore.dst != 126 || restore.src[0] != inverse.src[0]) continue;
            auto entry = entries.upper_bound(cmp.pc);
            if (entry != entries.end() && entry->first <= restore.pc)
                refuse(entry->first,"branch enters a CMPX region");
            if (vertex) {
                // Both arms contain only vector instructions. Each VS
                // invocation reaches this point with its own enclosing bit
                // set. Prove the complement uses that same saved EXEC.
                size_t anchor = k;
                while (anchor && (code[anchor-1].name == "s_waitcnt" || code[anchor-1].name == "s_nop"))
                    --anchor;
                if (!anchor) continue;
                const auto &save = code[anchor-1];
                const auto entrant = entries.upper_bound(save.pc);
                if (save.name != "s_mov_b64" || save.src[0] != 126 || save.dst != inverse.src[0] ||
                    (entrant != entries.end() && entrant->first <= cmp.pc)) continue;
                vertex_anchors.insert(save.pc);
                vertex_partitions.insert(cmp.pc);
            }
            cmpx_starts.insert(cmp.pc);
            enclosing_mask.emplace(cmp.pc,inverse.src[0]);
            complements.insert(inverse.pc);
            restores.insert(restore.pc);
            k = end;
        }
        // A second compare may target an SGPR mask and be ORed with VCC.
        // Each arm is a straight-line region. Conditional SMEM results are
        // usable within their arm only; poison them before lowering its join.
        for (size_t k = 0; k + 4 < program.code.size(); ++k) {
            auto &code = program.code;
            const auto &cmp = code[k], &combine = code[k+1], &active = code[k+2];
            auto &skip = code[k+3];
            if (wave32 || cmp.enc != Encoding::Vopc || cmp.name.substr(0,6) != "v_cmp_" ||
                cmp.dst > 104 || (cmp.dst & 1) || cmp.subword || cmp.clamp || cmp.scale ||
                combine.name != "s_or_b64" || combine.src[0] != 106 || combine.src[1] != cmp.dst ||
                combine.dst != cmp.dst || active.name != "s_and_b64" || active.dst != 126 ||
                active.src[0] != cmp.dst || active.src[1] != 126 || !skip.branch() || skip.op != 8 ||
                skip.target() <= skip.next) continue;
            const size_t mid = program.at.at(skip.target());
            if (mid + 1 >= code.size()) continue;
            const auto &inverse = code[mid];
            auto &skip_else = code[mid+1];
            if (inverse.name != "s_andn2_b64" || inverse.dst != 126 || inverse.src[0] > 104 ||
                (inverse.src[0] & 1) || inverse.src[1] != 126 || !skip_else.branch() ||
                skip_else.op != 8 || skip_else.target() <= skip_else.next) continue;
            const size_t end = program.at.at(skip_else.target());
            const auto &restore = code[end];
            if (restore.name != "s_mov_b64" || restore.dst != 126 || restore.src[0] != inverse.src[0]) continue;
            for (auto it = entries.upper_bound(cmp.pc); it != entries.end() && it->first <= restore.pc; ++it)
                if (it->second != skip.pc && it->second != skip_else.pc)
                    refuse(it->first, "branch enters a partitioned EXEC region");
            for (auto [begin, limit] : {std::pair{k+4,mid}, std::pair{mid+2,end}}) {
                unsigned saved = UINT_MAX;
                auto &dead = expired[code[limit].pc];
                for (size_t j = begin; j < limit; ++j) {
                    const auto &i = code[j];
                    if (i.next != code[j+1].pc) refuse(i.pc,"noncontiguous EXEC region");
                    if (local_alu(i) || i.name == "s_waitcnt" || i.name == "s_nop" ||
                        (i.enc == Encoding::Image && i.op == 0x27)) continue;
                    unsigned width = 0;
                    if (i.enc == Encoding::Smem) {
                        width = memory_count(i);
                    } else if (i.name == "s_and_saveexec_b64" && saved == UINT_MAX &&
                               i.src[0] <= 104 && !(i.src[0] & 1) && !(i.dst & 1)) {
                        saved = i.dst; inner_saves.insert(i.pc); width = 2;
                    } else if (i.name == "s_or_saveexec_b64" && i.src[0] == saved && !(i.dst & 1)) {
                        saved = UINT_MAX; inner_restores.insert(i.pc); width = 2;
                    } else refuse(i.pc,"unsupported operation inside partitioned EXEC");
                    if (i.dst + width > 106 ||
                        (i.dst < inverse.src[0]+2 && inverse.src[0] < i.dst+width) ||
                        (saved != UINT_MAX && i.enc == Encoding::Smem && i.dst < saved+2 && saved < i.dst+width))
                        refuse(i.pc,"clobbered enclosing EXEC mask");
                    for (unsigned r = i.dst; r < i.dst + width; ++r) dead.insert(r);
                }
                if (saved != UINT_MAX) refuse(code[limit].pc, "unbalanced inner EXEC");
            }
            compares.insert(cmp.pc); unions.insert(combine.pc); intersections.insert(active.pc);
            enclosing_mask.emplace(active.pc,inverse.src[0]);
            complements.insert(inverse.pc); restores.insert(restore.pc);
            skip_proven(skip); skip_proven(skip_else);
            k = end;
        }
        if (vertex && !wave32) vertex_scalar_tails(program,entries);
    }
};
std::map<unsigned, GraphicsBlock> graphics_blocks(const Program &program, const GraphicsMaskRegions &regions) {
    std::set<unsigned> leaders{0};
    for (auto edge : regions.then_fallthroughs) leaders.insert(edge.second);
    for (const auto &i : program.code) {
        if (!i.branch())
            continue;
        if (i.op != 2 && (i.op < 4 || i.op > 7) && !regions.lane_branches.count(i.pc)) {
            const auto k = program.at.at(i.pc);
            if (k && (i.op == 8 || i.op == 9)) {
                const auto &writer = program.code[k - 1];
                if (writer.next == i.pc && writer.dst == 126 &&
                    (writer.name == "s_wqm_b32" || writer.name == "s_wqm_b64"))
                    refuse(i.pc, "WQM branch on a wave mask needs the complete guest vote or a quad-local continuation");
            }
            refuse(i.pc, "graphics branch on a wave mask");
        }
        leaders.insert(i.target());
        if (i.op != 2)
            leaders.insert(i.next);
    }
    std::map<unsigned, GraphicsBlock> blocks;
    for (auto it = leaders.begin(); it != leaders.end(); ++it) {
        const auto next = std::next(it);
        size_t end = next == leaders.end() ? program.code.size() : program.at.at(*next);
        auto &block = blocks.emplace(*it, GraphicsBlock{program.at.at(*it), end}).first->second;
        const auto &last = program.code[end - 1];
        if (last.branch())
            block.successors.push_back(last.target());
        if (last.name != "s_endpgm" && (!last.branch() || (last.op != 2 && last.target() != last.next)))
            block.successors.push_back(regions.then_fallthroughs.count(last.pc) ?
                regions.then_fallthroughs.at(last.pc) : last.next);
    }
    for (const auto &[label, block] : blocks)
        for (auto next : block.successors)
            ++blocks.at(next).predecessors;
    return blocks;
}

struct GraphicsEdge {
    unsigned from;
    std::map<unsigned, string> values;
    std::set<unsigned> written;
    std::map<unsigned, GraphicsStage::Interpolation> partial;
    std::map<unsigned, unsigned> exports;
};

void join_graphics(GraphicsStage &g, std::vector<GraphicsEdge> &edges, const GraphicsMaskRegions &regions) {
    auto &alu = g.alu;
    alu.guarded_ij.clear();
    if (edges.size() == 1) {
        auto &edge = edges.front();
        alu.graphics_registers = std::move(edge.values);
        g.overwritten_inputs = std::move(edge.written);
        g.partial = std::move(edge.partial);
        g.exported = std::move(edge.exports);
        return;
    }
    alu.graphics_registers.clear();
    g.partial.clear();
    g.exported = edges.front().exports;
    g.overwritten_inputs.clear();
    for (const auto &edge : edges) {
        if (edge.exports != g.exported)
            refuse(g.pc, "inconsistent graphics exports at branch join");
        g.overwritten_inputs.insert(edge.written.begin(), edge.written.end());
    }
    for (const auto &[reg, part] : edges.front().partial) {
        bool same = std::all_of(edges.begin() + 1, edges.end(), [&](const auto &edge) {
            auto it = edge.partial.find(reg);
            if (it == edge.partial.end())
                return false;
            const auto &v = it->second;
            return v.attribute == part.attribute && v.channel == part.channel && v.source == part.source &&
                   v.mode == part.mode;
        });
        if (same)
            g.partial.emplace(reg, part);
    }
    DxilModule::InputMask control_inputs;
    if (auto merge = regions.branch_merges.find(g.pc); merge != regions.branch_merges.end()) {
        for (const auto &edge : edges)
            control_inputs |= alu.ir.dependencies[edge.values.at(merge->second)];
    }
    for (unsigned r : g.overwritten_inputs) {
        if (g.partial.count(r))
            continue;
        string first, incoming;
        bool same = true, defined = true;
        for (const auto &edge : edges) {
            auto it = edge.values.find(r);
            const string absent;
            const auto &value = it != edge.values.end() ? it->second
                                : edge.written.count(r) ? absent
                                                        : g.incoming_input(r);
            if (value.empty()) {
                defined = false;
                break;
            }
            if (incoming.empty())
                first = value;
            else {
                same &= value == first;
                incoming += ", ";
            }
            incoming += "[ " + value + ", %g" + to_string(edge.from) + " ]";
        }
        alu.graphics_registers[r] = !defined ? string{} : same ? first : alu.ir.value("phi i32 " + incoming);
        if (defined && !same)
            alu.ir.dependencies[alu.graphics_registers[r]] |= control_inputs;
    }
}

// A loop may consume a VGPR defined on a backedge. Give each requested
// block-entry value a name, then populate its phi once all exits are known.
// Only registers actually read through a block boundary acquire phis.
template <class EmitInstruction>
void emit_graphics_loops(GraphicsStage &g, const Program &program, const std::map<unsigned, GraphicsBlock> &blocks,
                         EmitInstruction emit) {
    auto &alu = g.alu;
    auto &ir = alu.ir;
    using Values = std::map<unsigned, string>;
    using Entry = std::pair<unsigned, unsigned>;
    std::map<unsigned, std::vector<unsigned>> preds;
    std::map<unsigned, Values> exits;
    std::map<unsigned, string> body, phis;
    std::set<Entry> requested;
    std::deque<Entry> pending;
    std::vector<string> predicates;
    std::optional<std::map<unsigned, unsigned>> exports;
    auto initial = alu.graphics_registers;
    auto entry_input = alu.graphics_input;
    std::ostringstream prologue;
    prologue.swap(ir.body);
    ir.loop_dependencies = true;
    auto phi = [&](unsigned block, unsigned reg) {
        Entry key{block, reg};
        if (requested.insert(key).second)
            pending.push_back(key);
        return "%p" + to_string(block) + "r" + to_string(reg);
    };
    for (const auto &[label, block] : blocks)
        for (auto target : block.successors)
            preds[target].push_back(label);
    for (const auto &[label, block] : blocks) {
        g.exported.clear();
        g.partial.clear();
        g.overwritten_inputs.clear();
        alu.graphics_registers.clear();
        alu.graphics_input = [&, label](unsigned r) {
            if (g.partial.count(r))
                refuse(g.pc, "intermediate interpolation value consumed in loop");
            return phi(label, r);
        };
        for (size_t k = block.begin; k < block.end; ++k) {
            const auto &i = program.code[k];
            if (i.branch() || i.name == "s_endpgm")
                continue;
            if (i.enc == Encoding::Export && !block.successors.empty())
                refuse(i.pc, "looping graphics shader exports before its exit block");
            if ((i.enc == Encoding::Interp || i.enc == Encoding::Fetch) && (label || !preds[label].empty()))
                refuse(i.pc, "looping graphics shader uses an entry input operation after a branch");
            emit(i, program.before[k]);
        }
        if (!g.partial.empty())
            refuse(label, "interpolation pair crosses a loop block boundary");
        const auto &last = program.code[block.end - 1];
        if (block.successors.empty()) {
            if (g.exported.empty() || (!g.pixel && !g.exported.count(12)) || (exports && *exports != g.exported))
                refuse(last.pc, "inconsistent loop exit exports");
            exports = g.exported;
            ir.body << "  br label %graphicsEnd\n";
        } else if (block.successors.size() == 1) {
            ir.body << "  br label %g" << block.successors.front() << "\n";
        } else {
            auto cond = alu.cmp((last.op & 1) ? "ne" : "eq", alu.read(last.op >= 6 ? 106 : 128), "0");
            predicates.push_back(cond);
            ir.body << "  br i1 " << cond << ", label %g" << last.target() << ", label %g" << last.next << "\n";
        }
        exits.emplace(label, std::move(alu.graphics_registers));
        body.emplace(label, ir.body.str());
        ir.body.str({});
    }
    if (!exports)
        refuse(0, "graphics loop has no reachable exit");
    alu.graphics_input = std::move(entry_input);
    ir.body.swap(prologue);
    g.partial.clear();
    while (!pending.empty()) {
        auto [label, reg] = pending.front();
        pending.pop_front();
        auto name = phi(label, reg);
        string incoming;
        auto add = [&](const string &v, const string &from) {
            if (v.empty())
                refuse(label, "graphics register undefined on an incoming loop edge");
            if (!incoming.empty())
                incoming += ", ";
            incoming += "[ " + v + ", %" + from + " ]";
            if (!v.empty() && v.front() == '%')
                ir.consumers[v].push_back(name);
        };
        if (label == 0) {
            auto value = initial.find(reg);
            if (value == initial.end())
                value = initial.emplace(reg, g.input(reg)).first;
            add(value->second, "entry");
        }
        for (unsigned pred : preds[label]) {
            const auto &values = exits.at(pred);
            auto value = values.find(reg);
            add(value != values.end() ? value->second : phi(pred, reg), "g" + to_string(pred));
        }
        phis[label] += "  " + name + " = phi i32 " + incoming + "\n";
    }
    ir.finish_loop_dependencies();
    for (const auto &condition : predicates)
        if (ir.dependencies[condition].any())
            refuse(0, "graphics loop condition depends on a varying input");
    ir.body << "  br label %g0\n";
    for (const auto &[label, code] : body)
        ir.body << "g" << label << ":\n" << phis[label] << code;
}

// An all-ones EXEC island can use VGPRs as a wave-wide register array.
// Keep separately defined guest lanes in SSA. This also works when the
// desktop PS runs on wave32 or has only a single covered quad. A varying
// pre-island pixel value has no proven address in this guest lane array.
struct GraphicsFullWave {
    DirectCompute &alu;
    unsigned width;
    std::optional<unsigned> saved;
    using Lanes = std::vector<string>;
    std::map<unsigned, Lanes> vectors;

    bool emit(const Inst &i, const Program &program) {
        const bool narrow = width == 32;
        const auto mov = narrow ? "s_mov_b32" : "s_mov_b64";
        if (!saved) {
            if (i.name != (narrow ? "s_or_saveexec_b32" : "s_or_saveexec_b64") || i.src[0] != 193)
                return false;
            if (i.dst > (narrow ? 105u : 104u) || (!narrow && (i.dst & 1)) || !alu.graphics_predicate.empty())
                refuse(i.pc, "invalid full-wave EXEC save");
            auto close = std::find_if(program.code.begin()+program.at.at(i.pc)+1, program.code.end(),
                [&](const Inst &j) { return j.branch() || j.name == "s_endpgm" || (j.name == mov && j.dst == 126); });
            if (close == program.code.end() || close->name != mov || close->src[0] != i.dst ||
                std::any_of(program.code.begin(), program.code.end(), [&](const Inst &j) {
                    return j.branch() && j.target() > i.pc && j.target() <= close->pc;
                }))
                refuse(i.pc, "full-wave EXEC island needs a matching restore in the same block");
            saved = i.dst;
            alu.write(*saved, {});
            if (!narrow) alu.write(*saved+1, {});
            alu.write(128, "1"); // Nonempty EXEC after OR with all ones.
            return true;
        }
        if (i.name == mov && i.dst == 126 && i.src[0] == *saved) {
            for (const auto &[reg, values] : vectors) {
                const bool uniform = std::all_of(values.begin()+1, values.end(),
                    [&](const string &v) { return v == values.front(); });
                alu.write(reg, uniform ? values.front() : string{});
            }
            vectors.clear();
            saved.reset();
            return true;
        }
        if (i.subword || i.abs || i.neg || i.scale || i.clamp || (i.wide && (i.a & 0x7800u)))
            refuse(i.pc, "unsupported full-wave operand modifier");
        auto scalar = [&](unsigned r) {
            if (r >= 256 || r == *saved || (!narrow && r == *saved+1))
                refuse(i.pc, "full-wave source needs the original pixel lane values");
            if (r == 126 || (!narrow && r == 127)) return to_string(~Word{0});
            auto value = alu.operand32(i,r);
            if (alu.ir.loop_dependencies || alu.ir.dependencies[value].any())
                refuse(i.pc, "full-wave scalar source depends on pixel input");
            return value;
        };
        auto constant = [&](unsigned r) {
            if (r == 126 || (!narrow && r == 127)) return ~Word{0};
            if (r == *saved || (!narrow && r == *saved+1)) refuse(i.pc, "saved EXEC used as a lane selector");
            const auto value = source(i, r, program.before[program.at.at(i.pc)]);
            if (value.type != Symbol::Type::constant) refuse(i.pc, "dynamic pixel wave lane selector");
            return value.value;
        };
        auto lane_value = [&](unsigned r, unsigned lane) -> string {
            if (r < 256) return scalar(r);
            auto found = vectors.find(r);
            if (found == vectors.end() || found->second[lane].empty())
                refuse(i.pc, "pixel wave register needs an unobserved source lane");
            return found->second[lane];
        };
        if (i.name == "v_writelane_b32") {
            auto &lanes = vectors.try_emplace(i.dst, width).first->second;
            lanes[constant(i.src[1]) & (width-1)] = scalar(i.src[0]);
            return true;
        }
        if (i.name == "v_readlane_b32") {
            if (i.dst == *saved || (!narrow && i.dst == *saved+1))
                refuse(i.pc, "lane read overwrites saved EXEC");
            alu.write(i.dst, lane_value(i.src[0],constant(i.src[1]) & (width-1)));
            return true;
        }
        if (i.name == "v_permlane16_b32" || i.name == "v_permlanex16_b32") {
            const Word selectors[]{constant(i.src[1]),constant(i.src[2])};
            const unsigned other = i.name == "v_permlanex16_b32" ? 16 : 0;
            Lanes result(width);
            for (unsigned lane=0; lane<width; ++lane) {
                unsigned pick = (selectors[(lane >> 3) & 1] >> (4*(lane & 7))) & 15;
                result[lane] = lane_value(i.src[0], ((lane & ~15u) ^ other) + pick);
            }
            vectors[i.dst] = std::move(result);
            return true;
        }
        if (i.name != "v_mov_b32" && i.name != "v_min_u32" && i.name != "v_max_u32" && i.name != "v_xor_b32")
            refuse(i.pc, "unsupported pixel full-wave operation " + string(i.name));
        Lanes result(width);
        // Broadcasts and reductions frequently leave many equal SSA pairs.
        // Reuse each expression so a wave-uniform result can leave the island.
        std::map<std::pair<string,string>,string> terms;
        for (unsigned lane=0; lane<width; ++lane) {
            unsigned from = lane;
            if (i.mirror_mask()) from ^= i.mirror_mask();
            else if (i.dpp_control) from = (lane & ~3u) | ((*i.dpp_control >> (2*(lane & 3))) & 3);
            auto a = lane_value(i.src[0],from);
            if (i.name == "v_mov_b32") result[lane] = a;
            else {
                auto b = lane_value(i.src[1],lane);
                auto &term = terms[{a,b}];
                if (term.empty())
                    term = i.name == "v_xor_b32" ? alu.ir.binary("xor",a,b) : a == b ? a :
                        alu.select(alu.cmp(i.name == "v_min_u32" ? "ult" : "ugt",a,b),a,b);
                result[lane] = term;
            }
        }
        vectors[i.dst] = std::move(result);
        return true;
    }
};

#include "gfx1013_graphics_wqm.inl"

NativeComputeShader lower_graphics(GuestShaderStage &stage, unsigned first_srv = 0, unsigned first_sampler = 0,
                                   const GraphicsConstantLayout *linked_constants = nullptr,
                                   const D3D12_ROOT_SIGNATURE_DESC1 &layout = {}, unsigned first_uav = 0) {
    Program program;
    Emit emitter{{}, program, layout};
    emitter.srv = first_srv;
    emitter.uav = first_uav;
    emitter.first_sampler = first_sampler;
    DirectCompute alu{emitter};
    const bool pixel = stage.stage == GpuStage::ps;
    alu.pixel_stage = pixel;
    alu.perspective_center_pair = pixel && (stage.pixel_input_address & 3) == 2 && (stage.pixel_input_enable & 2) != 0;
    if (pixel && !(stage.pixel_input_address >> 16))
        for (unsigned b = 0; b < 16; ++b)
            if (stage.pixel_input_address & (1u << b))
                alu.pixel_entry_vgprs += b == 3 ? 3 : b < 7 ? 2 : 1;
    GraphicsStage graphics{stage, alu, pixel};
    alu.untouched_pixel_register = [&](unsigned r) { return !graphics.overwritten_inputs.count(r); };
    alu.ir.track_inputs = true;
    alu.graphics_input = [&](unsigned r) { return graphics.input(r); };
    State initial{};
    for (auto &s : initial)
        s.type = Symbol::Type::unbound;
    std::set<unsigned> assigned;
    for (const auto &input : stage.inputs) {
        unsigned at = input.user_data;
        if (input.type == 1) {
            bind_root_address(initial, input, 106);
            assigned.insert(at);
            assigned.insert(at + 1);
            continue;
        }
        if (input.type != 0 && input.type != 3)
            continue;
        if (at >= 106 || input.root_parameter >= 16)
            refuse(0, "graphics root input outside ABI");
        if (input.type == 3) {
            Symbol table{Symbol::Type::table, input.root_parameter};
            if (!assigned.insert(at).second && !(initial[at] == table))
                refuse(0, "ambiguous graphics table");
            initial[at] = table;
        } else {
            if (!input.sgpr_count || input.sgpr_count > 32 || input.sgpr_count > 106 - at)
                refuse(0, "invalid graphics constant count");
            emitter.result.constants.push_back({input.root_parameter, at, input.sgpr_count});
            for (unsigned j = 0; j < input.sgpr_count; ++j) {
                if (!assigned.insert(at + j).second)
                    refuse(0, "overlapping graphics constants");
                initial[at + j] = {};
            }
        }
    }
    alu.after_graphics_write = [&](unsigned r) {
        // The provenance of an entry EXEC pair is valid only while both
        // halves keep their reaching definition on every predecessor.
        if (r <= 105) {
            auto tag = 512u + (stage.wave32 ? r : (r & ~1u));
            if (alu.graphics_registers.erase(tag)) graphics.overwritten_inputs.insert(tag);
        }
        graphics.partial.erase(r);
        graphics.overwritten_inputs.insert(r);
    };
    // The PS primitive mask follows user data (the legacy graphics entry ABI).
    // Keep it symbolic: desktop interpolation supplies the primitive, and a
    // numeric use of this SGPR or of M0 must still be refused.
    if (pixel && layout.NumParameters) {
        unsigned end = 8;
        for (unsigned p = 0; p < layout.NumParameters && end < 106; ++p) {
            const auto &param = layout.pParameters[p];
            end += param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE ? 1 :
                   param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS ? param.Constants.Num32BitValues : 2;
        }
        end += (layout.Flags & D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED) != 0;
        end += (layout.Flags & D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED) != 0;
        if (end < 106 && initial[end].type == Symbol::Type::unbound &&
            std::none_of(stage.inputs.begin(), stage.inputs.end(), [end](auto &p) {return p.user_data >= end || p.sgpr_count > end - p.user_data;}))
            initial[end].type = Symbol::Type::primitive_mask;
    }
    alu.initialize_constants(linked_constants);
    const auto code = stage.full_precision_code();
    program = parse(code, 0, initial, true, true, !stage.wave32, stage.fixed_vertex_abi);
    // A pixel discards when its invocation bit is removed from EXEC and the
    // only remaining work is local ALU plus its color export. The wave-empty
    // shortcut then reaches an empty EXP_NULL/end block. See Victory's pixel
    // export/discard lowering and gfx2hlsl.py's masked color export.
    unsigned discard_pc = UINT_MAX, combine_pc = UINT_MAX, mask_reg = UINT_MAX;
    if (pixel && !stage.wave32 && program.code.size() >= 6) {
        auto &ops = program.code;
        const size_t null = ops.size() - 2;
        const auto &output = ops[null];
        unsigned branches = 0;
        for (const auto &i : ops) branches += i.branch();
        if (branches == 1 && output.enc == Encoding::Export && (output.a & 0x1fff) == 0x1890 &&
            ops.back().name == "s_endpgm" && ops[null-1].name == "s_endpgm") {
            for (size_t n = 2; n + 1 < null; ++n) {
                auto &skip = ops[n];
                const auto &remove = ops[n-1], &combine = ops[n-2];
                if (!skip.branch() || skip.op != 8 || skip.target() != output.pc ||
                    remove.name != "s_andn2_b64" || remove.dst != 126 || remove.src[0] != 126 ||
                    combine.name != "s_or_b64" || combine.dst > 104 || (combine.dst & 1) ||
                    remove.src[1] != combine.dst ||
                    !((combine.src[0] == combine.dst && combine.src[1] == 106) ||
                      (combine.src[1] == combine.dst && combine.src[0] == 106))) continue;
                const bool local = std::all_of(ops.begin()+n+1, ops.begin()+null-1, [](const auto &op) {
                    return GraphicsMaskRegions::local_alu(op) || op.name == "s_waitcnt" ||
                           (op.enc == Encoding::Export && ((op.a >> 4) & 63) < 8);
                });
                if (!local) break;
                mask_reg = combine.dst;
                combine_pc = combine.pc;
                discard_pc = remove.pc;
                GraphicsMaskRegions::skip_proven(skip);
                program.at.erase(output.pc);
                program.at.erase(ops.back().pc);
                ops.resize(null);
                break;
            }
        }
    }
    PixelWqm wqm;
    if (pixel) {
        wqm.raster_guard(program,stage.wave32);
        wqm.raster_vote(program,stage.wave32);
        wqm.numeric_guards(program,stage.wave32);
    }
    GraphicsMaskRegions regions(program, stage.wave32, !pixel);
    const auto first_branch = std::find_if(program.code.begin(), program.code.end(),
                                            [](const auto &i) { return i.branch(); });
    const bool straight = first_branch == program.code.end();
    const bool entry_backedge = std::any_of(program.code.begin(), program.code.end(),
        [](const auto &i) { return i.branch() && i.target() <= i.pc; });
    const unsigned prologue_end = straight ? UINT_MAX : entry_backedge ? 0 : first_branch->pc;
    // Once EXEC returns to coverage for the last time, helper results are
    // unobservable if the remaining code only reads local data at explicit
    // LOD, packs colors, and exports them. A forward uniform branch can
    // cross this suffix without converting coverage into a wave-wide vote.
    auto tail_instruction = [](const Inst &i) {
        if (i.enc == Encoding::Sopp)
            return !i.branch() || i.op == 2 || i.op == 4 || i.op == 5;
        return i.enc == Encoding::Interp || i.enc == Encoding::Export ||
               (i.enc == Encoding::Smem && i.dst + memory_count(i) <= 106) ||
               (i.enc == Encoding::Image && i.op == 0x27) || i.name == "v_cvt_pkrtz_f16_f32";
    };
    auto last_observer = std::find_if_not(program.code.rbegin(), program.code.rend(), tail_instruction);
    // In a terminal straight-line suffix, helper values cannot reach a
    // neighboring lane or an externally visible store. Only covered lanes
    // commit the color export. Implicit LOD and all mask reductions end it.
    auto color_tail = [](const Inst &i) {
        if (GraphicsMaskRegions::local_alu(i)) return true;
        if (i.enc == Encoding::Export) return true;
        if (i.enc == Encoding::Smem) return i.dst + memory_count(i) <= 106;
        if (i.enc == Encoding::Image) return i.op == 0 || i.op == 0x27;
        if (i.enc == Encoding::Sopp)
            return i.name == "s_endpgm" || i.name == "s_waitcnt" || i.name == "s_clause" || i.name == "s_nop";
        if (i.enc == Encoding::Vopc) return i.dst == 106 && i.name.substr(0,6) == "v_cmp_";
        if (i.name == "s_cselect_b64" && i.dst == 106)
            return (i.src[0] == 128 || i.src[0] == 193) && (i.src[1] == 128 || i.src[1] == 193);
        return (i.enc == Encoding::Sopc || (i.enc == Encoding::Sop2 && i.dst < 106)) &&
            std::none_of(i.src.begin(),i.src.end(),[](unsigned s) { return s >= 106 && s <= 127; });
    };
    auto terminal_mask = std::find_if_not(program.code.rbegin(),program.code.rend(),color_tail);
    auto lane_local = [](const Inst &i) {
        if (i.enc == Encoding::Sopp) return !i.branch();
        if (i.enc == Encoding::Interp || i.enc == Encoding::Export) return true;
        if (i.enc == Encoding::Sop1) return i.name == "s_mov_b64" || i.name == "s_mov_b32";
        if (i.enc == Encoding::Vop1 || i.enc == Encoding::Vop2 || i.enc == Encoding::Vop3 || i.enc == Encoding::Vopc)
            return !i.dpp_control && i.name.find("lane") == Name::npos && i.name.find("mbcnt") == Name::npos;
        return false;
    };
    const auto last_lane_observer = std::find_if_not(program.code.rbegin(), program.code.rend(), lane_local);
    bool whole_quad = false;
    bool exact_after_wqm = false;
    bool exact_sample_region = false;
    std::map<unsigned,bool> saved_exec;
    std::string covered;
    string discard_lane;
    std::string parent_predicate, arm_condition, inner_parent;
    bool scalar_entry = true;
    std::set<string> uniform_vector_values;
    std::set<string> uniform_vcc_values;
    auto scalar_operand = [](unsigned s) { return s <= 105 || (s >= 128 && s <= 248) || s == 255; };
    auto covered_lane = [&] {
        auto helper = alu.ir.invoke("i1","isHelperLane.i1",221,{},"nounwind readonly");
        alu.ir.dependencies[helper].set(128);
        return alu.ir.value("xor i1 " + helper + ", true");
    };
    GraphicsFullWave full_wave{alu, stage.wave32 ? 32u : 64u};
    auto instruction = [&](const Inst &i, const State &state) {
        const auto pc = alu.graphics_pc = graphics.pc = i.pc;
        if (pixel && full_wave.emit(i,program)) return;
        if (wqm.packed.count(pc) && (whole_quad || exact_after_wqm || !alu.graphics_predicate.empty()))
            refuse(pc, "numeric EXEC region requires exact pixel entry coverage");
        if (pixel && wqm.emit_packed(alu,i,stage.wave32)) return;
        if (wqm.filters.count(pc)) {
            const auto tag=alu.graphics_registers.find(512+i.dst);
            if (!whole_quad || exact_after_wqm || !alu.graphics_predicate.empty() ||
                (wqm.survivor.empty() && (tag==alu.graphics_registers.end() || tag->second!="0")))
                refuse(pc,"WQM coverage filter requires saved exact entry EXEC");
            const auto keep=alu.cmp("eq",alu.read(106),"0");
            const auto previous=wqm.survivor.empty() ? covered_lane() : wqm.survivor;
            wqm.survivor=alu.ir.value("and i1 "+previous+", "+keep);
            alu.write(i.dst,{}); if (!stage.wave32) alu.write(i.dst+1,{});
            alu.write(128,{});
            return;
        }
        if (auto vote=wqm.vote_masks.find(pc); vote!=wqm.vote_masks.end()) {
            if (!whole_quad || exact_after_wqm || !alu.graphics_predicate.empty() || wqm.survivor.empty())
                refuse(pc,"cross-quad EXECZ mask requires whole-quad EXEC after a coverage filter");
            // Every lane of a covered quad, helpers included, is in the
            // whole-quad EXEC; the filtered coverage holds only survivors.
            const auto keep=alu.cmp("eq",alu.read(106),"0");
            using Source=PixelWqm::MaskSource;
            if (vote->second==Source::narrowed && wqm.vote_lane.empty())
                refuse(pc,"cross-quad EXECZ mask narrowed before it is written");
            wqm.vote_lane=vote->second==Source::whole_quad ? keep :
                alu.ir.value("and i1 "+(vote->second==Source::survivors ? wqm.survivor : wqm.vote_lane)+", "+keep);
            alu.write(i.dst,{}); if (!stage.wave32) alu.write(i.dst+1,{});
            alu.write(128,{});
            return;
        }
        if (pc==wqm.vote_wqm) {
            if (wqm.vote_lane.empty()) refuse(pc,"cross-quad EXECZ vote has no modelled EXEC");
            // EXEC at the branch is the WQM expansion of the mask: OR it over
            // the quad (quad reads see helper lanes), then vote over the wave.
            alu.ir.uses_wave=true;
            auto bit=alu.ir.value("zext i1 "+wqm.vote_lane+" to i32");
            auto quad=bit;
            for (unsigned op=0;op<3;++op)
                quad=alu.ir.binary("or",quad,
                    alu.ir.invoke("i32","quadOp.i32",123,{{"i32",bit},{"i8",to_string(op)}},"nounwind"));
            auto vote=alu.ir.invoke("i1","waveAnyTrue",113,{{"i1",alu.cmp("ne",quad,"0")}},"nounwind");
            alu.write(128,alu.ir.value("zext i1 "+vote+" to i32"));
            return;
        }
        if (wqm.quad_guards.count(pc)) {
            // Helpers have no input coverage bit. Broadcast the four survivor
            // bits while the quad is intact. Only a terminal colour region
            // proven independent across quads can use this projection.
            auto bit=alu.ir.value("zext i1 "+wqm.survivor+" to i32");
            auto any=bit;
            alu.ir.uses_wave=true;
            for (unsigned axis=0;axis<2;++axis) {
                auto peer=alu.ir.invoke("i32","quadOp.i32",123,{{"i32",any},{"i8",to_string(axis)}},"nounwind");
                any=alu.ir.binary("or",any,peer);
            }
            auto live=alu.cmp("ne",any,"0");
            const auto label=to_string(pc);
            alu.ir.body<<"  br i1 "<<live<<", label %wqmLive"<<label<<", label %wqmEmpty"<<label<<"\nwqmEmpty"<<label<<":\n";
            alu.ir.invoke("void","discard",82,{{"i1","true"}},"nounwind");
            alu.ir.body<<"  ret void\nwqmLive"<<label<<":\n";
            alu.write(128,{}); // Whole-wave SCC is not the quad vote.
            return;
        }
        if (pc==wqm.exact_restore) {
            auto killed=alu.ir.value("xor i1 "+wqm.survivor+", true");
            alu.ir.invoke("void","discard",82,{{"i1",killed}},"nounwind");
            return;
        }
        if (i.enc == Encoding::Vopc && i.dst == mask_reg && pc < combine_pc &&
            i.name.substr(0,6) == "v_cmp_" && !i.subword && !i.scale && !i.clamp && !i.abs && !i.neg &&
            i.name.find("class") == Name::npos && alu.graphics_predicate.empty() && !whole_quad) {
            auto lhs = alu.operand32(i, i.src[0]), rhs = alu.operand32(i, i.src[1]);
            alu.write(i.dst, {}); alu.write(i.dst + 1, {});
            discard_lane = alu.comparison(i, lhs, rhs);
            return;
        }
        if (pc == combine_pc) {
            if (discard_lane.empty()) refuse(pc, "discard mask has no reaching lane comparison");
            discard_lane = alu.ir.value("or i1 " + discard_lane + ", " + alu.cmp("ne",alu.read(106),"0"));
            alu.write(128, {});
            return;
        }
        if (pc == discard_pc) {
            if (whole_quad || !alu.graphics_predicate.empty()) refuse(pc, "discard requires exact pixel coverage");
            alu.ir.invoke("void", "discard", 82, {{"i1", discard_lane}}, "nounwind");
            alu.write(128, {});
            return;
        }
        if (!discard_lane.empty() && pc < combine_pc && i.dst < 106) {
            const unsigned width = i.enc == Encoding::Smem ? memory_count(i) :
                i.name.find("_b64") != Name::npos || i.enc == Encoding::Vopc ? 2 : 1;
            if (i.dst < mask_reg + 2 && mask_reg < i.dst + width) discard_lane.clear();
        }
        alu.helper_result_unobserved = straight &&
            (last_lane_observer == program.code.rend() || pc >= last_lane_observer->pc);
        if (auto it = regions.expired.find(pc); it != regions.expired.end())
            for (auto reg : it->second) alu.write(reg, {});
        if (regions.vertex_anchors.count(pc) || regions.captures.count(pc)) {
            // A single invocation observes one bit of this saved wave mask.
            // Never substitute that bit for a numeric 64-bit SGPR read.
            alu.write(i.dst,{});
            alu.write(i.dst+1,{});
            if (regions.captures.count(pc)) alu.write(128,{});
            return;
        }
        auto vertex_compare_bit = [&] {
            return alu.ir.value("zext i1 " + alu.comparison(i) + " to i32");
        };
        if (regions.vertex_mask_compares.count(pc)) {
            alu.write(1024+pc,vertex_compare_bit());
            alu.write(i.dst,{}); alu.write(i.dst+1,{});
            return;
        }
        if (auto it = regions.branch_saves.find(pc); it != regions.branch_saves.end()) {
            if (!alu.graphics_predicate.empty())
                refuse(pc,"vertex EXEC branch requires an active enclosing invocation");
            string bit;
            if (i.enc == Encoding::Vopc) {
                bit = vertex_compare_bit();
            } else if (auto input = regions.mask_inputs.find(pc); input != regions.mask_inputs.end()) {
                bit = alu.read(input->second);
            } else {
                bit = alu.read(106);
            }
            if (i.dst != 126) {
                alu.write(i.dst,{});
                if (!stage.wave32) alu.write(i.dst+1,{});
            }
            alu.write(128,{});
            alu.write(it->second, bit);
            return;
        }
        if (auto input = regions.mask_inputs.find(pc); input != regions.mask_inputs.end()) {
            if (regions.uniform_mask_tests.count(pc)) {
                auto bit = alu.read(input->second);
                if (alu.ir.dependencies[bit].any())
                    refuse(pc,"non-uniform compare needs a full guest-wave SCC reduction");
                // We are inside a proven nonempty EXEC arm. Every active
                // lane agrees on the comparison, so any(mask) equals it.
                alu.write(i.dst,{}); alu.write(i.dst+1,{}); alu.write(128,bit);
                return;
            }
            if (i.dst == 126 || i.name != "s_and_b64") refuse(pc,"mask-fed EXEC has no matching restore");
            auto bit = alu.ir.binary("and",alu.read(106),alu.read(input->second));
            alu.write(1024+pc,bit);
            alu.write(i.dst,{}); alu.write(i.dst+1,{}); alu.write(128,{});
            return;
        }
        if (regions.branch_merges.count(pc) && !regions.scalar_tail_merges.count(pc)) return;
        const bool cmpx = regions.cmpx_starts.count(pc) != 0;
        if (cmpx || regions.compares.count(pc)) {
            if (i.name.find("class") != Name::npos ||
                ((i.abs || i.neg) && i.name.find("_f") == Name::npos))
                refuse(pc,"unsupported partitioned compare modifiers");
            auto lhs = alu.operand32(i,i.src[0]), rhs = alu.operand32(i,i.src[1]);
            arm_condition = alu.comparison(i,lhs,rhs);
            if (!cmpx) {
                alu.write(i.dst, {}); alu.write(i.dst+1, {});
                return;
            }
        }
        if (regions.unions.count(pc)) {
            auto vcc = alu.cmp("ne",alu.read(106),"0");
            arm_condition = alu.ir.value("or i1 " + vcc + ", " + arm_condition);
            alu.write(128, {});
            return;
        }
        if (cmpx || regions.intersections.count(pc)) {
            const auto tag = alu.graphics_registers.find(512 + regions.enclosing_mask.at(pc));
            const bool known_parent = regions.vertex_partitions.count(pc) ||
                (whole_quad && !exact_after_wqm && tag != alu.graphics_registers.end() && tag->second == "1");
            if (!known_parent || !alu.graphics_predicate.empty())
                refuse(pc,"unproven enclosing whole-quad EXEC");
            parent_predicate.clear();
            alu.graphics_predicate = arm_condition;
            if (!cmpx) alu.write(128, {});
            return;
        }
        if (regions.inner_saves.count(pc)) {
            const auto tag = alu.graphics_registers.find(512 + i.src[0]);
            if (tag == alu.graphics_registers.end() || tag->second != "0")
                refuse(pc,"inner EXEC requires saved entry coverage");
            inner_parent = alu.graphics_predicate;
            auto exact = covered_lane();
            alu.graphics_predicate = alu.ir.value("and i1 " + inner_parent + ", " + exact);
            alu.write(i.dst, {}); alu.write(i.dst+1, {}); alu.write(128, {});
            return;
        }
        if (regions.inner_restores.count(pc)) {
            alu.graphics_predicate = inner_parent;
            alu.write(i.dst, {}); alu.write(i.dst+1, {}); alu.write(128, {});
            return;
        }
        if (regions.saves.count(pc)) {
            parent_predicate = alu.graphics_predicate;
            arm_condition = alu.cmp("ne", alu.read(106), "0");
            alu.graphics_predicate = parent_predicate.empty() ? arm_condition :
                alu.ir.value("and i1 " + parent_predicate + ", " + arm_condition);
            saved_exec.erase(i.dst);
            alu.write(i.dst, {});
            if (!stage.wave32) alu.write(i.dst + 1, {});
            alu.write(128, {});
            return;
        }
        if (regions.complements.count(pc)) {
            auto other = alu.ir.value("xor i1 " + arm_condition + ", true");
            alu.graphics_predicate = parent_predicate.empty() ? other :
                alu.ir.value("and i1 " + parent_predicate + ", " + other);
            alu.write(128, {});
            return;
        }
        if (regions.restores.count(pc)) {
            alu.graphics_predicate = parent_predicate;
            return;
        }
        const bool short_mask = stage.wave32 && i.name == "s_mov_b32";
        const bool move_exec = pixel && (short_mask || i.name == "s_mov_b64") && (i.dst == 126 || i.src[0] == 126);
        if (move_exec && i.dst == 126 && pc >= prologue_end && !entry_backedge && whole_quad &&
            alu.graphics_predicate.empty() && terminal_mask != program.code.rend() && terminal_mask->pc == pc) {
            const auto tag = alu.graphics_registers.find(512 + i.src[0]);
            if (tag == alu.graphics_registers.end() || tag->second != "0")
                refuse(pc,"terminal pixel export requires saved entry coverage");
            return;
        }
        if (pixel && pc >= prologue_end && (regions.sample_begins.count(pc) || regions.sample_ends.count(pc))) {
            const bool begin = regions.sample_begins.count(pc) != 0;
            const auto saved = alu.graphics_registers.find(512 + i.src[0]);
            if (!whole_quad || saved == alu.graphics_registers.end() || saved->second != (begin ? "0" : "1") ||
                (begin ? !alu.graphics_predicate.empty() : !exact_sample_region))
                refuse(pc,"unproven EXEC for a conditional sample");
            exact_after_wqm = alu.exact_pixel_entry = exact_sample_region = begin;
            if (begin) {
                // Run read-only sampling with the quad intact for derivatives,
                // then merge only the writes made under the guest coverage.
                alu.graphics_predicate = covered_lane();
            } else alu.graphics_predicate.clear();
            return;
        }
        if (!straight && pc >= prologue_end && exact_after_wqm && !exact_sample_region)
            refuse(pc, "exact pixel mask crosses a control-flow edge");
        if (move_exec && (short_mask ? i.dst <= 105 : i.dst <= 104 && !(i.dst & 1)) &&
            i.src[0] == 126 && pc < prologue_end) {
            saved_exec[i.dst] = whole_quad && !exact_after_wqm;
            // Numeric use of either half needs actual wave-mask lowering.
            alu.write(i.dst,{});
            if (!short_mask) alu.write(i.dst+1,{});
            alu.write(512+i.dst, saved_exec.at(i.dst) ? "1" : "0");
            return;
        }
        if (move_exec && i.dst == 126 && pc < prologue_end && saved_exec.count(i.src[0]) &&
            alu.graphics_registers.count(512 + i.src[0])) {
            exact_after_wqm = whole_quad && !saved_exec.at(i.src[0]);
            if (whole_quad && exact_after_wqm && !entry_backedge &&
                last_observer != program.code.rend() && pc == last_observer->pc) {
                while (!alu.guarded_ij.empty()) {
                    auto it = alu.guarded_ij.begin();
                    if (!covered.empty() && it->second.first == covered) {
                        alu.graphics_registers[it->first] = it->second.second;
                        alu.guarded_ij.erase(it);
                    } else {
                        alu.materialize_ij(it->first);
                    }
                }
                whole_quad = false;
                exact_after_wqm = false;
                alu.exact_pixel_entry = false;
                alu.graphics_predicate.clear();
                return;
            }
            if (exact_after_wqm) {
                if (covered.empty()) {
                    covered = covered_lane();
                }
                alu.graphics_predicate = covered;
                alu.exact_pixel_entry = true;
            } else {
                alu.graphics_predicate.clear();
                alu.exact_pixel_entry = false;
            }
            return;
        }
        if (i.enc == Encoding::Sop1 || i.enc == Encoding::Sop2 || i.enc == Encoding::Sopk ||
            i.enc == Encoding::Smem || i.enc == Encoding::Vop3 || i.enc == Encoding::Vop1) {
            // Invalidate both halves of a saved mask when scalar code reuses them.
            const unsigned width = i.enc == Encoding::Smem ? memory_count(i) :
                i.name.find("_b64") != Name::npos ? 2 : 1;
            if (i.dst<128)
                for (auto it=saved_exec.begin(); it!=saved_exec.end();)
                    if (i.dst<it->first+(stage.wave32 ? 1u : 2u) && it->first<i.dst+width) it=saved_exec.erase(it);
                    else ++it;
        }
        if (exact_after_wqm && i.enc != Encoding::Smem && i.enc != Encoding::Sopp &&
            i.enc != Encoding::Sop1 && i.enc != Encoding::Sop2 && i.enc != Encoding::Sopk &&
            i.enc != Encoding::Export &&
            !(i.name == "v_mov_b32" && i.dst >= 256 && !i.dpp_control && !i.subword) &&
            !(i.enc==Encoding::Image && i.op>=0x20 && i.op<=0x3f) &&
            !(i.name == "buffer_atomic_add" && !(i.a & 0x2000)))
            refuse(pc,"unsupported exact-mask operation after whole-quad entry");
        if ((i.name == "s_wqm_b32" || i.name == "s_wqm_b64") && i.dst == 126 && i.src[0] == 126) {
            if (!stage.wave32 && i.name == "s_wqm_b32")
                refuse(pc, "whole-quad operation affects only half of the guest EXEC mask");
            if (!pixel || pc >= prologue_end || !scalar_entry || whole_quad)
                refuse(pc, "whole-quad mode requires an unmodified pixel entry mask");
            whole_quad = true;
            alu.graphics_registers[128] = {}; // SCC describes the physical wave mask.
            return;
        }
        if (i.branch() || !((i.enc == Encoding::Sop1 || i.enc == Encoding::Sop2 || i.enc == Encoding::Smem ||
                             i.enc == Encoding::Sopk) && i.dst < 106) &&
            i.name != "v_nop" && i.name != "s_nop" && i.name != "s_clause" &&
            i.name != "s_setprio" && i.name != "s_inst_prefetch" && i.name != "s_waitcnt")
            scalar_entry = false;
        if (i.dpp_control) {
            if (i.mirror_mask())
                refuse(pc, i.mirror_mask() == 7 ? "DPP half-row mirror requires a proven eight-lane pixel layout"
                                                : "DPP row mirror requires a proven sixteen-lane pixel layout");
            // In whole-quad mode all four possible sources participate. At
            // exact coverage the desktop helper's VGPR contents are unproven.
            if (i.dpp_fetch_inactive && (!whole_quad || exact_after_wqm))
                refuse(pc, "DPP fetch-inactive source requires a proven whole quad");
            if (!whole_quad || exact_after_wqm)
                refuse(pc, "pixel quad permutation without whole-quad entry");
            const auto op = alu_rules.find(i.name);
            if (i.name != "v_mov_b32" && (op == alu_rules.end() || op->second.arity > 2))
                refuse(pc, "unsupported DPP arithmetic");
            if (i.enc == Encoding::Vop1 && (i.neg & 2 || i.abs & 2))
                refuse(pc, "unexpected second DPP source modifier");
            auto first = graphics.quad_source(i);
            alu.arithmetic(i, state, &first);
            return;
        }
        if (i.enc == Encoding::FixedSetup) {
            if (pixel || !straight) refuse(pc,"inline vertex fetch requires a straight-line VS");
            for (unsigned r = i.dst; r < i.dst + i.op; ++r) alu.write(r, {});
            if (i.next - pc == 8 && i.op == 1) alu.write(128, {}); // BFE sets SCC.
            return;
        }
        if (i.enc == Encoding::FixedFetch) {
            if (pixel || !straight) refuse(pc,"inline vertex fetch requires a straight-line VS");
            const auto &inputs = alu.ir.inputs;
            auto prior = std::find_if(inputs.begin(), inputs.end(), [&](const auto &r) {
                return r.name == "X2W_RAW_VERTEX" && r.semantic_index == i.literal;
            });
            unsigned row = prior != inputs.end() ? prior->row : static_cast<unsigned>(inputs.size());
            const auto id = graphics.row(true, "X2W_RAW_VERTEX", i.literal, 0, i.op == 4 ? 9 : 5,
                                         row, (1u << i.op) - 1, 0);
            for (unsigned c = 0; c < i.op; ++c) alu.write(i.dst + c, graphics.load(id, c));
            emitter.result.vertex_fetch=true;
            return;
        }
        if (i.enc == Encoding::Fetch) {
            if (pixel)
                refuse(pc, "vertex fetch call in a pixel shader");
            if (std::any_of(program.code.begin(),program.code.begin()+program.at.at(pc),
                            [](const Inst &prefix) { return prefix.name != "v_nop"; }))
                refuse(pc, "vertex fetch call outside the entry");
            // The stream fetch ABI supplies each declared input VGPR through
            // the input assembler. Its return address is opaque to guest ALU.
            if (stage.attributes.empty())
                refuse(pc, "vertex fetch call without fetch records");
            std::bitset<256> destinations;
            for (const auto &a : stage.attributes) {
                if (a.usage != 6 || !a.location || a.last_component > 3 || (a.mask >> a.last_component) != 1 ||
                    a.location + a.last_component >= 256)
                    refuse(pc, "invalid vertex fetch record");
                for (unsigned j = 0; j <= a.last_component; ++j) {
                    if (destinations.test(a.location + j))
                        refuse(pc, "overlapping vertex fetch destinations");
                    destinations.set(a.location + j);
                }
            }
            emitter.result.vertex_fetch = true;
            for (unsigned r : {4u, 5u}) {
                alu.graphics_registers[r] = {};
                graphics.overwritten_inputs.insert(r);
            }
            return;
        }
        if (i.enc == Encoding::Export) {
            graphics.emit_export(i.a, i.b);
            return;
        }
        if (i.enc == Encoding::Interp) {
            graphics.interpolate(i.a);
            return;
        }
        if (i.name == "s_nop" || i.name == "v_nop" || i.name == "s_setprio" || i.name == "s_clause" || i.name == "s_inst_prefetch" ||
            i.name == "s_waitcnt")
            return;
        if (i.enc == Encoding::Vopc && i.dst <= 106 && i.name.substr(0, 6) == "v_cmp_" && !i.scale && !i.clamp) {
            // Keep the invocation bit for CNDMASK and predicated regions.
            // A whole-wave VCC branch also requires a uniform condition.
            auto pred = alu.comparison(i);
            if (!alu.graphics_predicate.empty())
                pred = alu.ir.value("and i1 " + alu.graphics_predicate + ", " + pred);
            auto bit = alu.ir.value("zext i1 " + pred + " to i32");
            if (i.dst == 106) alu.write(106,bit);
            else {
                alu.write(i.dst,{});
                if (!stage.wave32) alu.write(i.dst+1,{});
                alu.write(1024+pc,bit);
            }
            if (i.dst == 106 && alu.graphics_predicate.empty() && scalar_operand(i.src[0]) && scalar_operand(i.src[1]) &&
                alu.ir.dependencies[bit].none()) uniform_vcc_values.insert(bit);
            return;
        }
        if (i.name == "v_cndmask_b32" && !i.abs && !i.neg && !i.scale && !i.clamp) {
            if (i.wide && (i.a & 0x7800)) refuse(pc,"unsupported graphics CNDMASK operand selection");
            auto a = alu.operand32(i, i.src[0]), b = alu.operand32(i, i.src[1]);
            if (i.subword) {
                a = alu.subword(a, i.src_select[0], i.src_sext[0]);
                b = alu.subword(b, i.src_select[1], i.src_sext[1]);
            }
            auto mask = alu.graphics_lane_mask(i,state,i.src[2],"unproven graphics CNDMASK source");
            alu.vector_write(i.dst, alu.place_subword(i, alu.select(alu.cmp("ne",mask,"0"), b, a)));
            return;
        }
        if (pixel && i.name == "s_mov_b32" && i.dst == 124 && i.src[0] < 106 &&
            state[i.src[0]].type == Symbol::Type::primitive_mask)
            return;
        if (i.name == "v_readfirstlane_b32" && i.dst < 106 && i.src[0] >= 256 &&
            !i.subword && !i.scale && !i.clamp && !i.abs && !i.neg && alu.graphics_predicate.empty()) {
            auto value = alu.read(i.src[0]);
            if (!uniform_vector_values.count(value))
                refuse(pc,"graphics readfirstlane requires a wave-uniform reaching value");
            alu.write(i.dst,value);
            return;
        }
        if (whole_quad && !exact_after_wqm && alu.graphics_predicate.empty() &&
            i.name == "s_and_b64" && i.dst <= 104 && !(i.dst & 1) &&
            ((i.src[0] == 106 && i.src[1] == 126) || (i.src[1] == 106 && i.src[0] == 126))) {
            auto bit = alu.read(106);
            if (!uniform_vcc_values.count(bit))
                refuse(pc,"varying VCC requires a full-wave nonzero reduction");
            // An executing invocation belongs to a nonempty guest wave. For
            // a uniform comparison VCC & EXEC is nonzero iff its bit is set.
            // The numeric SGPR pair still depends on the actual wave mask.
            alu.write(i.dst, {}); alu.write(i.dst + 1, {}); alu.write(128,bit);
            return;
        }
        if (i.name == "s_cselect_b64" && i.dst == 106 &&
            (i.src[0] == 128 || i.src[0] == 193) && (i.src[1] == 128 || i.src[1] == 193)) {
            auto scc = alu.read(128);
            alu.write(106, alu.select(alu.cmp("ne",scc,"0"),i.src[0] == 193 ? "1" : "0",
                                     i.src[1] == 193 ? "1" : "0"));
            alu.write(107, {});
            return;
        }
        if (i.name == "s_cselect_b64" && i.dst <= 104 && !(i.dst & 1) &&
            (i.src[0] == 128 || i.src[0] == 193) && (i.src[1] == 128 || i.src[1] == 193)) {
            if (!regions.scalar_mask_selects.count(pc) || alu.ir.loop_dependencies)
                refuse(pc,"full scalar mask requires a reaching scalar SCC comparison");
            auto scc = alu.read(128);
            if (alu.ir.dependencies[scc].any())
                refuse(pc,"full scalar mask depends on a varying SCC");
            auto word = alu.select(alu.cmp("ne",scc,"0"),i.src[0] == 193 ? "4294967295" : "0",
                                   i.src[1] == 193 ? "4294967295" : "0");
            alu.write(i.dst,word); alu.write(i.dst+1,word);
            alu.write(1024+pc,word);
            return;
        }
        if (((i.enc == Encoding::Sop1 || i.enc == Encoding::Sop2 || i.enc == Encoding::Sopk) && i.dst >= 106) ||
            i.branch() || i.enc == Encoding::Sopp || i.enc == Encoding::Lds || i.enc == Encoding::Vopc ||
            i.name.find("exec") != Name::npos || i.name.find("lane") != Name::npos ||
            i.name.find("mbcnt") != Name::npos || i.name == "v_cndmask_b32")
            refuse(i.pc, "graphics wave or control-flow instruction");
        if (i.enc == Encoding::Smem || i.enc == Encoding::Mubuf) {
            if (i.name.find("atomic") != Name::npos) {
                // ponytail: branch splitting stays inside one guest block; general
                // graphics CFG edges would need to track the new predecessor labels.
                if (!pixel || i.name != "buffer_atomic_add" || (i.a & 0x2000) || !straight)
                    refuse(i.pc, "graphics buffer atomic requires a straight-line pixel raw DWORD add");
                if (whole_quad && !exact_after_wqm)
                    refuse(i.pc, "pixel atomic requires exact coverage after WQM");
                auto pred = covered_lane();
                if (!alu.graphics_predicate.empty())
                    pred = alu.ir.value("and i1 " + pred + ", " + alu.graphics_predicate);
                alu.buffer_atomic(i, state, pred);
                return;
            }
            alu.memory(i, state);
            if (i.name == "buffer_load_format" && (i.a & 0x3ffffu) == 0 && alu.graphics_predicate.empty()) {
                // A read-only uniform table entry and the literal element zero
                // supply the same bits to every active invocation.
                for (unsigned n = 0; n <= (i.op & 3); ++n) {
                    auto value = alu.read(256 + i.dst + n);
                    if (alu.ir.dependencies[value].none()) uniform_vector_values.insert(value);
                }
            }
        } else if (i.enc == Encoding::Image) {
            alu.image_io(i, state);
        } else {
            // SGPR values and inline constants are common to the wave.
            // Propagate that fact through pure VS arithmetic before a write
            // can replace an overlapping source. An FMAC also reads its dst.
            bool uniform_vertex = false;
            auto rule = alu_rules.find(i.name);
            if (!pixel && !alu.ir.loop_dependencies && alu.graphics_predicate.empty() &&
                !i.subword && GraphicsMaskRegions::local_alu(i) && rule != alu_rules.end()) {
                auto uniform = [&](unsigned r) {
                    if (scalar_operand(r)) return true;
                    auto source = alu.graphics_registers.find(r);
                    return r >= 256 && source != alu.graphics_registers.end() &&
                           uniform_vector_values.count(source->second) != 0;
                };
                const bool mac = i.name == "v_fmac_f32";
                const unsigned count = mac || i.name == "v_fmamk_f32" || i.name == "v_fmaak_f32" ? 2 : rule->second.arity;
                uniform_vertex = std::all_of(i.src.begin(),i.src.begin()+count,uniform) && (!mac || uniform(i.dst));
            }
            alu.arithmetic(i, state);
            if (uniform_vertex || (pixel && i.name == "v_med3_f32" && alu.graphics_predicate.empty() &&
                std::all_of(i.src.begin(), i.src.end(), scalar_operand))) {
                auto value = alu.read(i.dst);
                if (alu.ir.dependencies[value].none()) uniform_vector_values.insert(value);
            }
        }
    };
    auto blocks = graphics_blocks(program, regions);
    std::map<unsigned, std::vector<GraphicsEdge>> edges;
    std::vector<unsigned> ready;
    if (!blocks.at(0).predecessors)
        ready.push_back(0);
    std::vector<unsigned> order;
    while (!ready.empty()) {
        unsigned label = ready.back();
        ready.pop_back();
        order.push_back(label);
        for (auto next : blocks.at(label).successors)
            if (!--blocks.at(next).predecessors)
                ready.push_back(next);
    }
    if (order.size() != blocks.size()) {
        if (!regions.lane_branches.empty()) refuse(0, "masked vertex control flow requires an acyclic graph");
        emit_graphics_loops(graphics, program, blocks, instruction);
    } else {
        alu.ir.body << "  br label %g0\n";
        std::optional<std::map<unsigned, unsigned>> final_exports;
        for (unsigned label : order) {
            auto &block = blocks.at(label);
            alu.ir.body << "g" << label << ":\n";
            graphics.pc = label;
            if (label) {
                join_graphics(graphics, edges.at(label), regions);
                edges.erase(label);
            }
            for (size_t k = block.begin; k < block.end; ++k) {
                const auto &i = program.code[k];
                if (!i.branch() && i.name != "s_endpgm")
                    instruction(i, program.before[k]);
            }
            const auto &last = program.code[block.end - 1];
            alu.graphics_pc = last.pc;
            if (!block.successors.empty()) {
                // Each predecessor must carry the value for covered and helper
                // lanes before its definitions become incoming phi operands.
                while (!alu.guarded_ij.empty()) alu.materialize_ij(alu.guarded_ij.begin()->first);
            }
            if (block.successors.empty()) {
                if (graphics.exported.empty() || (!pixel && !graphics.exported.count(12)) ||
                    (final_exports && *final_exports != graphics.exported))
                    refuse(last.pc, "incomplete or inconsistent terminal graphics exports");
                final_exports = graphics.exported;
                alu.ir.body << "  br label %graphicsEnd\n";
            } else if (block.successors.size() == 2) {
                const auto branch = regions.lane_branches.find(last.pc);
                auto condition = branch != regions.lane_branches.end() ?
                    alu.cmp(branch->second.second ? "ne" : "eq", alu.read(branch->second.first), "0") :
                    alu.cmp((last.op & 1) ? "ne" : "eq", alu.read(last.op >= 6 ? 106 : 128), "0");
                // The cross-quad EXECZ vote is uniform over the desktop wave;
                // its input dependencies still reach the values merged after it.
                // Wave intrinsics are undefined on helper lanes, so a helper may
                // take the other edge: safe only while nothing after the branch
                // reads another lane (raster_vote keeps the tail lane-local).
                if (branch == regions.lane_branches.end() && alu.ir.dependencies[condition].any() &&
                    last.pc != wqm.vote_skip)
                    refuse(last.pc, "graphics scalar condition depends on a varying input");
                alu.ir.body << "  br i1 " << condition << ", label %g" << last.target() << ", label %g" << last.next
                            << "\n";
            } else
                alu.ir.body << "  br label %g" << block.successors.front() << "\n";
            for (size_t j = 0; j < block.successors.size(); ++j) {
                auto &incoming = edges[block.successors[j]];
                if (j + 1 == block.successors.size())
                    incoming.push_back({label, std::move(alu.graphics_registers),
                                        std::move(graphics.overwritten_inputs), std::move(graphics.partial),
                                        std::move(graphics.exported)});
                else
                    incoming.push_back({label, alu.graphics_registers, graphics.overwritten_inputs, graphics.partial,
                                        graphics.exported});
                if (auto capture = regions.captures.find(block.successors[j]); capture != regions.captures.end()) {
                    auto &edge = incoming.back();
                    auto vcc = edge.values.find(106);
                    const bool skip = last.pc == capture->second.skip;
                    if (!skip && (vcc == edge.values.end() || vcc->second.empty()))
                        refuse(last.pc,"missing active VCC contribution to EXEC intersection");
                    edge.values[capture->second.value] = skip ? "0" : vcc->second;
                    edge.written.insert(capture->second.value);
                }
            }
        }
    }
    emitter.result.instruction_count = static_cast<unsigned>(program.code.size());
    alu.ir.body << "graphicsEnd:\n  ret void\n";
    auto copy_signature = [](const auto &rows, auto &out) {
        for (const auto &r : rows)
            out.push_back(
                {r.name, r.semantic_index, r.row, r.first, r.width, r.component, r.used, r.kind, r.interpolation});
    };
    copy_signature(alu.ir.inputs, emitter.result.inputs);
    copy_signature(alu.ir.outputs, emitter.result.outputs);
    if (!pixel && emitter.uav != first_uav) alu.ir.features |= 1u << 16; // UAVsAtEveryStage
    emitter.result.dxil_ir = alu.ir.finish_compute(0, 0, 0, pixel ? "ps" : "vs");
    return std::move(emitter.result);
}
