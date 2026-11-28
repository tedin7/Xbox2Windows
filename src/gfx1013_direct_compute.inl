// Lower the shared decoded CFG to DXIL operations. A scalar register holds one
// 32-bit word per wave. Ordinary vector writes retain their old value on inactive lanes.
struct DirectCompute {
    // The nine inline float constants as binary16, which an f16 operand reads instead of the f32 pattern.
    static constexpr Word inline_half_words[]{0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000,
                                              0xc000, 0x4400, 0xc400, 0x3118};
    static void validate_lane_shape(const Inst &i, unsigned threads, unsigned width) {
        const auto constant = constant32(i, i.src[1]);
        const auto remaining = threads % width;
        if (((i.src[1] <= 105 || (width == 32 && i.src[1] == 107)) && threads % 32) ||
            (constant && remaining && (*constant & (width - 1)) >= remaining))
            refuse(i.pc, "lane read/write needs full waves");
    }
    Emit &analysis;
    DxilModule ir;
    std::set<unsigned> allocated;
    string lane, lane_bit, half_index, upper_half, exchange_base;
    unsigned exchange_words = 0;
    bool uses_exchange = false;
    bool wave64 = false;
    // The view class a GPU-selected heap slot is read as; see HeapViewContract.
    HeapViewContract heap_view = HeapViewContract::none;
    // heap_view is raw_buffer and the path is direct: the byte-address rewrite of a heap descriptor is tracked.
    bool heap_window = false;
    // Binary16 add, sub, subrev and mul are lowered only while MODE stays at
    // the contract this path implements: round-to-nearest-even, f16 denormals
    // kept on input and output, and FP16_OVFL clear. mode_bits holds the
    // fields this path tracks (round in [3:0], denorm in [7:4], FP16_OVFL in
    // bit 23). A guest branch together with a MODE write makes the value at
    // the arithmetic unprovable.
    bool half_binary = false;
    bool half_mode_known = true;
    unsigned mode_bits = 0xf0;
    std::vector<bool> scc_live;
    unsigned label_number = 0;
    bool pixel_stage = false;
    // Straight-line graphics stages keep exact SSA definitions; a missing
    // entry value is resolved through the container's stage signature.
    std::function<string(unsigned)> graphics_input;
    std::function<void(unsigned)> after_graphics_write;
    std::map<unsigned, string> graphics_registers;
    string graphics_predicate;
    unsigned graphics_pc = 0;
    const State *compute_state = nullptr;
    unsigned compute_pc = 0;
    bool perspective_center_pair = false;
    bool exact_pixel_entry = false;
    bool helper_result_unobserved = false;
    unsigned pixel_entry_vgprs = 0;
    std::function<bool(unsigned)> untouched_pixel_register;
    // P1/P2 consume the entry IJ pair symbolically. After an exact-mask
    // write, covered lanes can use the replacement. Materialize the entry
    // barycentric only when an observer can also see the unmodified lanes.
    std::map<unsigned, std::pair<string, string>> guarded_ij;
    string materialize_ij(unsigned r) {
        auto [predicate, updated] = guarded_ij.at(r);
        auto value = select(predicate, updated, graphics_input(r));
        graphics_registers[r] = value;
        guarded_ij.erase(r);
        return value;
    }
    string address(unsigned r) {
        const auto name = "%r" + to_string(r);
        if (allocated.insert(r).second)
            ir.setup << "  " << name << " = alloca i32, align 4\n";
        return name;
    }
    string read_storage(unsigned r) {
        if (graphics_input) {
            auto guarded = guarded_ij.find(r);
            if (guarded != guarded_ij.end())
                return guarded->second.first == graphics_predicate ? guarded->second.second : materialize_ij(r);
            auto found = graphics_registers.find(r);
            if (found != graphics_registers.end()) {
                if (found->second.empty())
                    refuse(graphics_pc, "graphics register undefined on an incoming path: " + to_string(r));
                return found->second;
            }
            auto value = graphics_input(r);
            graphics_registers.emplace(r, value);
            return value;
        }
        return ir.value("load i32, i32* " + address(r) + ", align 4");
    }
    string read(unsigned r) {
        if (compute_state && compute_state->has_spill(r))
            refuse(compute_pc, "symbolic VGPR lane read of " + reg(r));
        return read_storage(r);
    }
    bool lane_write = false;  // v_writelane_b32 updates one lane of a VGPR that may hold symbolic lanes
    void write(unsigned r, const string &value) {
        // A VGPR write the analysis did not model as clearing its symbolic lanes
        // would let a later v_readlane_b32 return a stale identity.
        if (compute_state && !lane_write && compute_state->has_spill(r))
            refuse(compute_pc, "write to a VGPR holding symbolic lanes: " + reg(r));
        if (graphics_input) {
            guarded_ij.erase(r);
            graphics_registers[r] = value;
            after_graphics_write(r);
            return;
        }
        const auto ptr = address(r);
        ir.body << "  store i32 " << value << ", i32* " << ptr << ", align 4\n";
    }
    string cmp(const string &predicate, const string &a, const string &b) {
        return ir.value("icmp " + predicate + " i32 " + a + ", " + b);
    }
    string select(const string &predicate, const string &yes, const string &no) {
        return ir.value("select i1 " + predicate + ", i32 " + yes + ", i32 " + no);
    }
    string active() {
        if (graphics_input)
            return "true";
        return cmp("ne", ir.binary("and", wave64 ? select(upper_half, read(127), read(126)) : read(126), lane_bit), "0");
    }
    void vector_write(unsigned r, const string &value) {
        if (graphics_input && graphics_predicate.empty())
            write(r, value);
        else if (perspective_center_pair && (r == 256 || r == 257) &&
                 (!graphics_registers.count(r) || (guarded_ij.count(r) && guarded_ij.at(r).first == graphics_predicate))) {
            write(r, {});
            guarded_ij.emplace(r, std::make_pair(graphics_predicate, value));
        } else if (exact_pixel_entry && helper_result_unobserved && pixel_entry_vgprs && r >= 256 + pixel_entry_vgprs &&
                   !graphics_registers.count(r) && untouched_pixel_register(r)) {
            // Only covered lanes can observe this result. In particular no
            // subsequent implicit LOD or quad operation can inspect helpers.
            write(r, value);
        } else
            write(r, select(graphics_input ? graphics_predicate : active(), value, read_storage(r)));
    }
    string operand32(const Inst &i, unsigned n, unsigned sgpr_limit = 106) {
        if (graphics_input && n == 106)
            refuse(i.pc, "graphics instruction consumes a full wave mask");
        // M0 holds ordinary numeric ALU bits here, not an implicit memory/indexing operand.
        const bool numeric_m0 = !graphics_input && n == 124 && i.enc != Encoding::Mubuf && i.enc != Encoding::Smem;
        if (n <= 106 || n == 126 || n == 127 || (!graphics_input && n == 107) || numeric_m0 || (n >= 256 && n < 512)) {
            analysis.require_numeric(i, n, sgpr_limit);
            return read(n);
        }
        if (n == 253)
            return read(128); // internal slot for SCC
        if (auto constant = constant32(i, n))
            return to_string(*constant);
        refuse(i.pc, "direct DXIL operand encoding");
    }
    string ballot(const string &pred) {
        ir.uses_wave = true;
        auto b = ir.invoke("%dx.types.fouri32", "waveActiveBallot", 116, {{"i1", pred}}, "nounwind");
        return ir.value("extractvalue %dx.types.fouri32 " + b + ", 0");
    }
    string group_intrinsic(const string &name, unsigned opcode, unsigned axis) {
        return ir.invoke("i32", name, opcode, {{"i32", to_string(axis)}}, "nounwind readnone");
    }
    string shared_address(const string &name, unsigned words, const string &index) {
        const string type = "[" + to_string(words) + " x i32]";
        return ir.value("getelementptr " + type + ", " + type + " addrspace(3)* @" + name + ", i32 0, i32 " + index);
    }
    void barrier(unsigned mode = 9) {
        ir.invoke("void", "barrier", 80, {{"i32", to_string(mode)}}, "nounwind noduplicate");
    }
    // Each guest wave owns two scratch words. Multi-wave groups that use
    // exchange must reach these workgroup barriers in the same order.
    std::array<string, 2> exchange_halves(const string &word) {
        const auto tag = to_string(label_number++);
        if (!uses_exchange) {
            ir.globals << "@waveExchange = internal addrspace(3) global [" << exchange_words << " x i32] undef, align 4\n";
            uses_exchange = true;
        }
        auto ptr = shared_address("waveExchange", exchange_words, half_index);
        auto first = cmp("eq", lane, "0");
        ir.body << "  br i1 " << first << ", label %publish" << tag << ", label %published" << tag
                << "\npublish" << tag << ":\n  store i32 " << word << ", i32 addrspace(3)* " << ptr
                << ", align 4\n  br label %published" << tag << "\npublished" << tag << ":\n";
        barrier();
        std::array<string, 2> pair;
        for (unsigned j = 0; j != 2; ++j)
            pair[j] = ir.value("load i32, i32 addrspace(3)* " + shared_address("waveExchange", exchange_words, ir.binary("add", exchange_base, to_string(j))) + ", align 4");
        barrier();
        return pair;
    }
    void write_ballot(unsigned dst, const string &condition) {
        auto mask = ballot(ir.value("and i1 " + active() + ", " + condition));
        if (wave64) {
            if ((dst & 1) || (dst > 104 && dst != 106 && dst != 126))
                throw UnsupportedComputeShader("invalid wave64 mask destination");
            auto pair = exchange_halves(mask);
            write(dst, pair[0]);
            write(dst + 1, pair[1]);
        } else
            write(dst, mask);
    }
    string pair_high(const Inst &i, unsigned src) {
        if (src <= 104 || src == 106 || src == 126) {
            if (src & 1) refuse(i.pc, "unaligned scalar mask pair");
            return operand32(i, src + 1);
        }
        if ((src >= 128 && src <= 192) || src == 255) return "0";
        if (src >= 193 && src <= 208) return "4294967295";
        refuse(i.pc, "unsupported wave64 scalar pair");
    }
    void initialize(unsigned user, unsigned flags, unsigned x, unsigned y, unsigned z) {
        unsigned threads = x * y * z;
        lane = ir.invoke("i32", "waveGetLaneIndex", 111, {}, "nounwind readnone");
        lane_bit = ir.binary("shl", "1", lane);
        write(126, ballot("true"));
        if (!wave64) {
            write(127, "0");
            // s107 is ordinary scalar data in wave32; a VCC pair read sees zero until it is written.
            write(107, "0");
        }
        string wave = "0";
        if (threads > 32) {
            auto flat = ir.invoke("i32", "flattenedThreadIdInGroup.i32", 96, {}, "nounwind readnone");
            auto first =
                ir.invoke("i32", "waveActiveOp.i32", 119, {{"i32", flat}, {"i8", "2"}, {"i8", "1"}}, "nounwind");
            if (threads == 64)
                wave = ir.value("zext i1 " + cmp("ne", first, "0") + " to i32");
            else {
                ir.globals << "@waveStarts = internal addrspace(3) global [" << threads << " x i32] undef, align 4\n";
                auto ptr = shared_address("waveStarts", threads, flat);
                auto leader = ir.value("zext i1 " + cmp("eq", first, flat) + " to i32");
                ir.body << "  store i32 " << leader << ", i32 addrspace(3)* " << ptr << ", align 4\n";
                barrier();
                ir.body << "  br label %rankLoop\nrankLoop:\n  %idx = phi i32 [0, %entry], [%step, %count]\n"
                           "  %rank = phi i32 [0, %entry], [%sum, %count]\n";
                auto more = cmp("ult", "%idx", first);
                ir.body << "  br i1 " << more << ", label %count, label %rankDone\ncount:\n";
                auto addr = shared_address("waveStarts", threads, "%idx");
                auto flag = ir.value("load i32, i32 addrspace(3)* " + addr + ", align 4");
                ir.body << "  %sum = add i32 %rank, " << flag
                        << "\n  %step = add i32 %idx, 1\n  br label %rankLoop\nrankDone:\n";
                wave = "%rank";
            }
        }
        if (wave64) {
            exchange_words = threads / 32;
            half_index = wave;
            exchange_base = ir.binary("and", wave, "4294967294");
            upper_half = cmp("ne", ir.binary("and", wave, "1"), "0");
            write(127, "4294967295");
        }
        auto logical = ir.binary("add", ir.binary("mul", wave, "32"), lane);
        write(256, ir.binary("urem", logical, to_string(x)));
        if (((flags >> 11) & 3) >= 1)
            write(257, ir.binary("urem", ir.binary("udiv", logical, to_string(x)), to_string(y)));
        if (((flags >> 11) & 3) >= 2)
            write(258, ir.binary("udiv", logical, to_string(x * y)));
        for (unsigned j = 0; j < 3; ++j)
            if (flags & (128u << j))
                write(user++, group_intrinsic("groupId.i32", 94, j));
        if (flags & 1024) {
            if (wave64) wave = ir.binary("lshr", wave, "1");
            auto info = ir.binary("or", ir.binary("shl", wave, "20"), to_string(wave64 ? threads / 64 : (threads + 31) / 32));
            write(user, ir.binary("or", info, select(cmp("eq", wave, "0"), "2147483648", "0")));
        }
        initialize_constants();
    }
    void initialize_constants(const GraphicsConstantLayout *linked = nullptr) {
        unsigned words = 0;
        for (const auto &block : analysis.result.constants)
            words += block.count;
        if (words) {
            if (linked)
                words = linked->offset(static_cast<unsigned>(linked->counts.size()));
            auto h = ir.handle({2, 0, 13, 0, 0, (words + 3) / 4 * 16});
            unsigned pos = 0;
            std::map<unsigned, string> vectors;
            for (const auto &block : analysis.result.constants) {
                if (linked)
                    pos = linked->offset(block.root_parameter);
                for (unsigned j = 0; j < block.count; ++j, ++pos) {
                    auto &vec = vectors[pos / 4];
                    if (vec.empty())
                        vec = ir.invoke("%dx.types.CBufRet.i32", "cbufferLoadLegacy.i32", 59,
                                        {{"%dx.types.Handle", h}, {"i32", to_string(pos / 4)}}, "nounwind readonly");
                    write(block.first_sgpr + j,
                          ir.value("extractvalue %dx.types.CBufRet.i32 " + vec + ", " + to_string(pos % 4)));
                }
            }
        }
    }
    // Refuses a GPU-selected heap read that the contract does not cover. raw_buffer is the only class that is translated.
    void refuse_heap_contract(unsigned pc) const {
        if (heap_view == HeapViewContract::none)
            refuse(pc, "direct resource-heap load requires the selected GPU slot and its view type");
        if (heap_view == HeapViewContract::other)
            refuse(pc, "GPU-selected heap slot view type is not raw_buffer; only raw buffer heap reads are translated");
    }
    // A raw buffer SRV chosen by a GPU value in the whole descriptor heap: ResourceDescriptorHeap[slot]. The view class the
    // shader reads is the one this handle is annotated with (raw buffer); a slot holding another class is refused by the
    // dispatch before the shader runs. The binding names the heap (no table register); the pipeline indexes it directly.
    // A byte window is the byte-address rewrite of that descriptor (word 0 copied, word 1 without stride, word 2 the
    // byte count, word 3 with the raw out-of-bounds mode; see byte_descriptor_field): read as the same raw slot view.
    string heap_resource(const Inst &i, const State &s, unsigned base, ComputeViewKind kind, unsigned bytes) {
        const auto &symbol = s.at(base);
        refuse_heap_contract(i.pc);
        const bool window = kind == ComputeViewKind::byte_window_srv;
        if (kind != ComputeViewKind::raw_vector_srv && kind != ComputeViewKind::buffer_srv && !window)
            refuse(i.pc, "a GPU-selected heap slot is read as a raw buffer SRV only");
        if (symbol.offset % 32 || s[base].type != Symbol::Type::indexed_descriptor)
            refuse(i.pc, "unaligned heap descriptor");
        // Precaution: the rewritten words are compared by the pc of their heap load (symbol.definition), which a loop
        // around that load executes again for another selection. A loop that does not contain the load runs it once.
        // The scan also sees unreached code up to s_endpgm, which can only refuse more.
        if (window)
            for (const auto &inst : analysis.program.code)
                if (inst.branch() && inst.target() <= symbol.definition && symbol.definition <= inst.pc)
                    refuse(i.pc, "heap byte window loaded inside a loop");
        constexpr Symbol::Type window_words[] = {Symbol::Type::indexed_descriptor, Symbol::Type::zero_stride_word,
                                                 Symbol::Type::byte_count, Symbol::Type::byte_oob};
        for (unsigned j = 1; j < 4; ++j) {
            Symbol expected = symbol;
            if (window)
                expected.type = window_words[j];  // field() already cleared the word offset
            else
                expected.offset += 4 * j;
            if (!(s.at(base + j) == expected))
                refuse(i.pc, "non-contiguous resource descriptor");
        }
        auto &bindings = analysis.result.bindings;
        auto it = std::find_if(bindings.begin(), bindings.end(), [](const auto &b) {
            return b.kind == ComputeViewKind::heap_buffer_srv;
        });
        if (it == bindings.end())
            bindings.push_back({ComputeViewKind::heap_buffer_srv, compute_resource_heap, 0, 0, bytes});
        else
            it->required_bytes = (std::max)(it->required_bytes, bytes);
        // 16: raw buffers. 0x40000000: the module reads ResourceDescriptorHeap (DXC sets both for the HLSL reference,
        // see tests/native_heap_handle_probe_check.py); the validator refuses a module whose flags do not match.
        ir.features |= 16 | 0x40000000;
        return ir.heap_handle(ir.binary("add", read(base), to_string(symbol.offset / 32)), 11, 0);
    }
    std::set<std::pair<unsigned,unsigned>> read_write_raw_views;
    string resource(const Inst &i, const State &s, unsigned base, ComputeViewKind kind, unsigned bytes,
                    unsigned *bound_register = nullptr) {
        const auto &symbol = s.at(base);
        const bool indexed = symbol.type == Symbol::Type::indexed_descriptor;
        if (indexed && symbol.value == compute_resource_heap)
            return heap_resource(i, s, base, kind, bytes);
        if (!indexed && symbol.type == Symbol::Type::descriptor &&
            (kind == ComputeViewKind::buffer_srv || kind == ComputeViewKind::raw_vector_srv ||
             kind == ComputeViewKind::byte_window_srv) &&
            read_write_raw_views.count({symbol.value, symbol.offset})) {
            // Both instructions read the same descriptor table entry. A
            // second, SRV binding would conflict with its writable resource state.
            kind = ComputeViewKind::raw_buffer_uav;
        }
        const unsigned count = indexed ? 1 + symbol.index_mask / 32 : 1;
        analysis.binding(i, s, base, 4, kind, bytes);
        auto it = std::find_if(analysis.result.bindings.begin(), analysis.result.bindings.end(), [&](const auto &b) {
            return !b.array_position && b.array_count == count && b.kind == kind && b.root_parameter == s[base].value &&
                   b.byte_offset == s[base].offset;
        });
        if (it == analysis.result.bindings.end())
            throw std::logic_error("missing direct binding");
        if (bound_register) *bound_register = it->shader_register;
        const bool raw = kind == ComputeViewKind::raw_buffer_uav || kind == ComputeViewKind::buffer_srv ||
                         kind == ComputeViewKind::raw_vector_srv || kind == ComputeViewKind::byte_window_srv ||
                         kind == ComputeViewKind::indexed_atomic_uav;
        const bool half = kind == ComputeViewKind::half_buffer_uav;
        const unsigned width = kind == ComputeViewKind::typed_buffer_srv ? 4 : bytes / (half ? 2 : 4);
        if (raw)
            ir.features |= 16;
        return ir.handle({unsigned(compute_uav(kind)), it->shader_register, raw ? 11u : 10u,
                          raw    ? 0u
                          : half ? 9u
                                 : 5u,
                          raw ? 0u : width, 0, count},
                         indexed && count > 1 ? read(base) : "");
    }
    string load_buffer(const string &h, const string &addr, bool raw, unsigned count) {
        using Arg = std::pair<string, string>;
        std::vector<Arg> args{{"%dx.types.Handle", h}, {"i32", addr}, {"i32", "undef"}};
        if (raw) {
            args.emplace_back("i8", to_string((1u << count) - 1));
            args.emplace_back("i32", "4");
        }
        return ir.invoke("%dx.types.ResRet.i32", raw ? "rawBufferLoad.i32" : "bufferLoad.i32", raw ? 139 : 68, args,
                         "nounwind readonly");
    }
    void buffer_atomic(const Inst &i, const State &s, const string &pixel_predicate = {}) {
        // The operation set and VGPR ordering follow gfx2hlsl.MUBUF_ATOMIC_OPS.
        static const std::map<Name, unsigned> operations{
            {"buffer_atomic_add", 0},  {"buffer_atomic_sub", 0},  {"buffer_atomic_and", 1},  {"buffer_atomic_or", 2},
            {"buffer_atomic_xor", 3},  {"buffer_atomic_smin", 4}, {"buffer_atomic_smax", 5}, {"buffer_atomic_umin", 6},
            {"buffer_atomic_umax", 7}, {"buffer_atomic_swap", 8}};
        const bool exchange = i.name == "buffer_atomic_cmpswap";
        const auto op = operations.find(i.name);
        if ((!exchange && op == operations.end()) || (exchange && i.dst == 255))
            refuse(i.pc, "unsupported buffer atomic width or operation");
        const unsigned data = 256 + i.dst;
        const bool indexed = (i.a & 0x2000) != 0;
        unsigned reg;
        auto h = resource(i, s, ((i.b >> 16) & 31) * 4,
                          indexed ? ComputeViewKind::indexed_atomic_uav : ComputeViewKind::raw_buffer_uav, 4, &reg);
        string address, in_range;
        if (indexed) {
            // b(1 + uav register) is filled from the exact recorded buffer view.
            // Test the element count before multiplication to reject wrapped indices.
            auto cb = ir.handle({2, reg + 1, 13, 0, 0, 16});
            auto shape = ir.invoke("%dx.types.CBufRet.i32", "cbufferLoadLegacy.i32", 59,
                                   {{"%dx.types.Handle", cb}, {"i32", "0"}}, "nounwind readonly");
            auto stride = ir.value("extractvalue %dx.types.CBufRet.i32 " + shape + ", 0");
            auto count = ir.value("extractvalue %dx.types.CBufRet.i32 " + shape + ", 1");
            auto index = read(256 + (i.b & 255));
            in_range = cmp("ult", index, count);
            address = ir.binary("mul", index, stride);
        } else {
            address = ir.binary("add", operand32(i, i.b >> 24), to_string(i.a & 4095));
            if (i.a & 0x1000)
                address = ir.binary("add", address, read(256 + (i.b & 255)));
        }
        address = ir.binary("and", address, "4294967292");
        auto value = read(data);
        if (i.name == "buffer_atomic_sub")
            value = ir.binary("sub", "0", value);
        std::vector<std::pair<string, string>> args{{"%dx.types.Handle", h}};
        if (!exchange)
            args.emplace_back("i32", to_string(op->second));
        args.emplace_back("i32", address);
        args.emplace_back("i32", "undef");
        args.emplace_back("i32", "undef");
        if (exchange)
            args.emplace_back("i32", read(data + 1));
        args.emplace_back("i32", value);
        auto tag = to_string(label_number++), pred = active();
        if (!pixel_predicate.empty()) {
            // Graphics uses SSA rather than the compute VGPR allocas. The caller
            // proves a single guest block and supplies exact coverage and EXEC.
            // Keep the original VGPR for inactive lanes, including helpers.
            const auto size = ir.invoke("%dx.types.Dimensions", "getDimensions", 72,
                {{"%dx.types.Handle", h}, {"i32", "undef"}}, "nounwind readonly");
            const auto bytes = ir.value("extractvalue %dx.types.Dimensions " + size + ", 0");
            const auto in_view = cmp("ult", address, bytes); // aligned DWORD, raw view length is a multiple of 4
            const auto fallback = (i.a & 0x4000) ? select(pixel_predicate, "0", value) : "";
            const auto execute = ir.value("and i1 " + pixel_predicate + ", " + in_view);
            ir.body << "  br i1 " << execute << ", label %atomic" << tag << ", label %atomicSkip" << tag
                    << "\natomic" << tag << ":\n";
            const auto previous = ir.invoke("i32", "atomicBinOp.i32", 78, args, "nounwind");
            ir.body << "  br label %atomicEnd" << tag << "\natomicSkip" << tag
                    << ":\n  br label %atomicEnd" << tag << "\natomicEnd" << tag << ":\n";
            if (i.a & 0x4000) {
                auto returned = ir.value("phi i32 [ " + previous + ", %atomic" + tag + " ], [ " +
                                         fallback + ", %atomicSkip" + tag + " ]");
                // Contention makes the old value varying even with uniform operands.
                ir.dependencies[returned].set(128);
                write(data, returned);
            }
            return;
        }
        if (indexed) {
            if (i.a & 0x4000) {
                auto out_of_range = ir.value("and i1 " + pred + ", " + ir.value("xor i1 " + in_range + ", true"));
                write(data, select(out_of_range, "0", read(data)));
            }
            pred = ir.value("and i1 " + pred + ", " + in_range);
        }
        ir.body << "  br i1 " << pred << ", label %atomic" << tag << ", label %atomicEnd" << tag << "\natomic" << tag
                << ":\n";
        auto previous = ir.invoke("i32", exchange ? "atomicCompareExchange.i32" : "atomicBinOp.i32", exchange ? 79 : 78,
                                  args, "nounwind");
        if (i.a & 0x4000)
            write(data, previous);
        ir.body << "  br label %atomicEnd" << tag << "\natomicEnd" << tag << ":\n";
    }
    #include "gfx1013_compute_mtbuf.inl"

    // Global DWORD loads in the saddr form: ADDR = SADDR + zext(VADDR) + sext(OFFSET[11:0]) (RDNA 2 ISA,
    // FLAT/GLOBAL addressing), with SADDR the unadvanced root address of a root SRV, as for the scalar root loads.
    // Guard the first and last DWORD offsets against 32-bit wrap. The low two address bits are dropped,
    // as for buffer_load_dword in this path (DWORD alignment mode).
    void global_load(const Inst &i, const State &s) {
        constexpr unsigned widths[]{1, 2, 4, 3}; // GFX10 opcodes 12..15.
        const unsigned count = widths[i.op - 12];
        if (i.dst + count > 256)
            refuse(i.pc, "global load destination register range");
        const unsigned base = (i.b >> 16) & 127;
        if (base > 104 || (base & 1))
            refuse(i.pc, "global load base must be an aligned SGPR pair");
        const auto &low = s.at(base);
        if (low.type != Symbol::Type::root_address)
            refuse(i.pc, "global load base is not a root buffer address");
        const Symbol high{Symbol::Type::root_address, low.value, 1};
        if (low.offset || !(s.at(base + 1) == high))
            refuse(i.pc, "global load base is an advanced or mismatched root address");
        const int offset = int(Word(i.a) << 20) >> 20;
        if (offset % 4)
            refuse(i.pc, "global load at an unaligned immediate offset");
        const unsigned span = static_cast<unsigned>((std::max)(offset, 0)) + count * 4;
        auto &bindings = analysis.result.bindings;
        auto binding = std::find_if(bindings.begin(), bindings.end(), [&](const auto &b) {
            return b.kind == ComputeViewKind::root_buffer_srv && b.root_parameter == low.value;
        });
        if (binding == bindings.end()) {
            bindings.push_back({ComputeViewKind::root_buffer_srv, low.value, 0, analysis.srv++, span});
            binding = std::prev(bindings.end());
        } else
            binding->required_bytes = (std::max)(binding->required_bytes, span);
        ir.features |= 16;
        const auto h = ir.handle({0, binding->shader_register, 11});
        const auto vaddr = read(256 + (i.b & 255));
        string in_range;
        if (offset < 0)
            in_range = cmp("uge", vaddr, to_string(-offset));
        const int last_offset = offset + static_cast<int>((count - 1) * 4);
        if (last_offset > 0) {
            const auto upper = cmp("ule", vaddr, to_string(0xffffffffu - static_cast<unsigned>(last_offset)));
            in_range = in_range.empty() ? upper : ir.value("and i1 " + in_range + ", " + upper);
        }
        const auto addr = offset ? ir.binary("add", vaddr, to_string(offset)) : vaddr;
        const auto ret = load_buffer(h, ir.binary("and", addr, "4294967292"), true, count);
        for (unsigned j = 0; j < count; ++j) {
            auto value = ir.value("extractvalue %dx.types.ResRet.i32 " + ret + ", " + to_string(j));
            if (!in_range.empty())
                value = ir.value("select i1 " + in_range + ", i32 " + value + ", i32 0");
            vector_write(256 + i.dst + j, value);
        }
    }
    void memory(const Inst &i, const State &s) {
        if (i.name.substr(0, 17) == "global_load_dword") {
            global_load(i, s);
            return;
        }
        if (i.enc == Encoding::Mubuf && i.name.substr(0, 8) == "tbuffer_") {
            formatted_byte_load(i, s);
            return;
        }
        if (i.enc == Encoding::Mubuf && i.name.substr(0, 14) == "buffer_atomic_") {
            buffer_atomic(i, s);
            return;
        }
        if (i.enc == Encoding::Smem) {
            unsigned so = i.b >> 25;
            const unsigned limit = smem_sgpr_limit(!graphics_input && !wave64);
            const auto &low = s.at(i.src[0]);
            const bool root = low.type == Symbol::Type::root_address;
            if (!root && low.type == Symbol::Type::table && low.value == compute_resource_heap) {
                // Reads from the whole descriptor heap: only the indexed descriptor load of a captured raw buffer slot.
                if (i.op >= 8)
                    refuse(i.pc, "scalar buffer load from the descriptor heap is not translated");
                refuse_heap_contract(i.pc);
                if (so == 125)
                    refuse(i.pc, "heap descriptor load needs a GPU-selected index (SGPR offset)");
                // The captured selection names the slot the GPU data selects; a constant index or an immediate offset
                // would address another slot than the one the guard checked.
                if (s.at(so).type == Symbol::Type::constant)
                    refuse(i.pc, "heap descriptor index is a constant; the captured selection must come from GPU data");
                if (i.b & 0xfffff)
                    refuse(i.pc, "heap descriptor load with an immediate offset addresses another slot than the captured selection");
            }
            if (!root && low.type == Symbol::Type::table && low.value == compute_sampler_heap) {
                // Reads from the directly indexed sampler heap: one or two samplers at a constant 16-byte aligned offset.
                if (i.op >= 8)
                    refuse(i.pc, "scalar buffer load from the sampler heap is not translated");
                if (so != 125)
                    refuse(i.pc, "sampler heap load with an SGPR offset (a GPU-selected sampler) is not translated");
                if (i.op != 2 && i.op != 3)
                    refuse(i.pc, "sampler heap load must read one or two samplers");
                if ((i.b & 0xfffff) % 16)
                    refuse(i.pc, "sampler heap offset is not a multiple of 16");
                if ((i.b & 0xfffff) / 16 + (i.op == 3 ? 2 : 1) - 1 > 2046)
                    refuse(i.pc, "sampler heap index beyond 2046 (2047 is the runtime fallback entry)");
            }
            if (i.op < 8 && !root) {
                if (s.at(i.src[0]).type != Symbol::Type::table ||
                    s.at(i.src[0] + 1).type != Symbol::Type::constant || s.at(i.src[0] + 1).value != 0)
                    refuse(i.pc, "direct descriptor address outside bound table");
                if (so != 125) {
                    if (i.op != 2 || so >= limit)
                        refuse(i.pc, "direct indexed descriptor load shape");
                    auto index = ir.binary("lshr", operand32(i, so, limit), "5");
                    for (unsigned j = 0; j < 4; ++j)
                        write(i.dst + j, index);
                }
                return; // The CFG transfer already associated every destination word with its table slot.
            }
            unsigned off = i.b & 0xfffff, n = memory_count(i);
            string h;
            if (root) {
                const Symbol high{Symbol::Type::root_address, low.value, 1};
                if (low.offset || !(s.at(i.src[0] + 1) == high))
                    refuse(i.pc, "mismatched root address high half");
                // Like Victory's Source::Built, retain the root identity through
                // SGPR copies. Scalar buffer loads read dwords at byte offsets.
                // Accept a finite byte bound or the unbounded NUM_RECORDS form
                // only for this verified zero-stride, byte-address V# layout.
                const bool built = i.op >= 8;
                if (built && (s.at(i.src[0] + 2).type != Symbol::Type::constant ||
                              !(s.at(i.src[0] + 3) == Symbol{Symbol::Type::constant, 0x31016fac})))
                    refuse(i.pc, "unsupported synthesized scalar buffer descriptor fields");
                unsigned span = off + 4 * n;
                if (off % 4 || span > 65536 || i.dst + n > limit)
                    refuse(i.pc, "unsupported root descriptor load range");
                if (so != 125) {
                    const auto &index = s.at(so);
                    if ((index.type != Symbol::Type::masked && index.type != Symbol::Type::constant) ||
                        (index.value & 3) || index.value > 65536 - span)
                        refuse(i.pc, "unbounded or unaligned root descriptor index");
                    span += index.value;
                }
                if (built && span > s.at(i.src[0] + 2).value)
                    refuse(i.pc, "scalar load exceeds synthesized descriptor byte bound");
                auto &bindings = analysis.result.bindings;
                auto binding = std::find_if(bindings.begin(), bindings.end(), [&](const auto &b) {
                    return b.kind == ComputeViewKind::root_buffer_srv && b.root_parameter == low.value;
                });
                if (binding == bindings.end()) {
                    bindings.push_back({ComputeViewKind::root_buffer_srv, low.value, 0, analysis.srv++, span});
                    binding = std::prev(bindings.end());
                } else
                    binding->required_bytes = (std::max)(binding->required_bytes, span);
                ir.features |= 16;
                h = ir.handle({0, binding->shader_register, 11});
            } else
                h = resource(i, s, i.src[0], ComputeViewKind::buffer_srv, so == 125 ? off + 4 * n : 4);
            auto addr = so == 125 ? to_string(off)
                                  : ir.binary("and", ir.binary("add", operand32(i, so, limit), to_string(off)), "4294967292");
            for (unsigned j = 0; j < n; j += 4) {
                unsigned width = (std::min)(4u, n - j);
                auto chunk = so == 125 ? to_string(off + j * 4) :
                             j ? ir.binary("add", addr, to_string(j * 4)) : addr;
                auto ret = load_buffer(h, chunk, true, width);
                for (unsigned k = 0; k < width; ++k)
                    write(i.dst + j + k, ir.value("extractvalue %dx.types.ResRet.i32 " + ret + ", " + to_string(k)));
            }
            return;
        }
        bool store = i.name == "buffer_store_raw" || i.name == "buffer_store_format";
        bool raw = i.name == "buffer_load_dword" || i.name == "buffer_store_raw";
        bool half = i.name == "buffer_store_format" && (i.op & 128);
        constexpr unsigned widths[]{1, 2, 4, 3};
        unsigned n = raw ? widths[i.op & 3] : (i.op & 3) + 1, data = 256 + i.dst;
        if (i.dst + (half ? (n + 1) / 2 : n) > 256)
            refuse(i.pc, "direct buffer data overflow");
        auto kind = store ? (raw    ? ComputeViewKind::raw_buffer_uav
                             : half ? ComputeViewKind::half_buffer_uav
                                    : ComputeViewKind::typed_buffer_uav)
                    : raw && s.at(((i.b >> 16) & 31) * 4 + 1).type == Symbol::Type::zero_stride_word
                          ? ComputeViewKind::byte_window_srv
                    : raw ? ComputeViewKind::raw_vector_srv
                          : ComputeViewKind::typed_buffer_srv;
        auto h = resource(i, s, ((i.b >> 16) & 31) * 4, kind, raw ? 4 : n * (half ? 2 : 4));
        auto addr = (i.a & (raw ? 0x1000 : 0x2000)) ? read(256 + (i.b & 255)) : "0";
        if (raw) {
            addr = ir.binary("add", addr, to_string(i.a & 4095));
            addr = ir.binary("and", ir.binary("add", addr, operand32(i, i.b >> 24)), "4294967292");
        }
        if (!store) {
            auto ret = load_buffer(h, addr, raw, n);
            for (unsigned k = 0; k < n; ++k)
                vector_write(data + k, ir.value("extractvalue %dx.types.ResRet.i32 " + ret + ", " + to_string(k)));
            return;
        }
        const string tag = to_string(label_number++), pred = active();
        if (!graphics_input)
            ir.body << "  br i1 " << pred << ", label %store" << tag << ", label %after" << tag << "\nstore" << tag
                << ":\n";
        std::vector<std::pair<string, string>> args{{"%dx.types.Handle", h}, {"i32", addr}, {"i32", "undef"}};
        for (unsigned j = 0; j < 4; ++j) {
            string value = "undef";
            if (j < n || !raw) {
                const unsigned component = (std::min)(j, n - 1);
                value = read(data + (half ? component / 2 : component));
                if (half) {
                    if (component % 2)
                        value = ir.binary("lshr", value, "16");
                    value = ir.invoke("float", "legacyF16ToF32", 131, {{"i32", value}}, "nounwind readnone");
                }
            }
            args.emplace_back(half ? "float" : "i32", value);
        }
        args.emplace_back("i8", to_string(raw ? (1u << n) - 1 : 15));
        if (raw)
            args.emplace_back("i32", "4");
        ir.invoke("void",
                  raw    ? "rawBufferStore.i32"
                  : half ? "bufferStore.f32"
                         : "bufferStore.i32",
                  raw ? 140 : 69, args, "nounwind");
        if (!graphics_input)
            ir.body << "  br label %after" << tag << "\nafter" << tag << ":\n";
    }
    string select_float(const string &condition, const string &yes, const string &no) {
        return ir.value("select i1 " + condition + ", float " + yes + ", float " + no);
    }
    // Invert the cube projection. Each pair of faces has a common major
    // axis, with the negative face second. The two minor axes span [-1,1].
    std::array<string, 3> cube_axes(const string &face, const string &u, const string &v, bool tangent) {
        const auto nu = ir.value("fsub float -0.0, " + u), nv = ir.value("fsub float -0.0, " + v);
        const auto axis = ir.binary("lshr", ir.binary("and", face, "7"), "1");
        const auto is_x = cmp("eq", axis, "0"), is_y = cmp("eq", axis, "1");
        const auto negative = cmp("ne", ir.binary("and", face, "1"), "0");
        const auto major = tangent ? "0.0" : select_float(negative, "-1.0", "1.0");
        return {select_float(is_x, major, select_float(is_y, u, select_float(negative, nu, u))),
                select_float(is_y, major, nv),
                select_float(is_x, select_float(negative, u, nu), select_float(is_y, select_float(negative, nv, v), major))};
    }
    string texture_dimensions(const string &handle, const string &mip) {
        return ir.invoke("%dx.types.Dimensions", "getDimensions", 72,
                         {{"%dx.types.Handle", handle}, {"i32", mip}}, "nounwind readonly");
    }
    void explicit_gather_2d(const Inst &i, const State &s, const string &handle,
                            const std::array<string, 3> &coords, unsigned mask) {
        const unsigned sampler_base = ((i.b >> 21) & 31) * 4;
        std::array<Word, 4> words{};
        for (unsigned j = 0; j < words.size(); ++j) {
            const auto &word = s.at(sampler_base + j);
            if (word.type != Symbol::Type::constant)
                refuse(i.pc, "explicit 2D gather requires a constant sampler");
            words[j] = word.value;
        }
        // Match gfx2hlsl's normalized clamp profile. Point/bilinear XY and
        // CLAMP_Z do not change this 2D footprint; the mip filter must be point.
        constexpr Word xy_filter = (3u << 20) | (3u << 22);
        if ((words[0] & ~(7u << 6)) != 0x12 || (words[1] & ~0x00fff000u) ||
            (words[2] & ~xy_filter) != 0x04000000 || ((words[2] >> 20) & 3) > 1 ||
            ((words[2] >> 22) & 3) > 1 || words[3])
            refuse(i.pc, "unsupported explicit 2D gather sampler");
        auto finite = [&](const string &bits) {
            return cmp("ne", ir.binary("and", bits, "2139095040"), "2139095040");
        };
        auto valid = finite(coords[0]);
        for (unsigned j = 1; j < coords.size(); ++j)
            valid = ir.value("and i1 " + valid + ", " + finite(coords[j]));
        auto safe_float = [&](unsigned j) {
            return ir.value("bitcast i32 " + select(valid, coords[j], "0") + " to float");
        };
        auto clamp_float = [&](const string &value, const string &max) {
            auto low = ir.invoke("float", "binary.f32", 35, {{"float", value}, {"float", "0.0"}}, "nounwind readnone");
            return ir.invoke("float", "binary.f32", 36, {{"float", low}, {"float", max}}, "nounwind readnone");
        };
        auto floor = [&](const string &value) {
            return ir.invoke("float", "unary.f32", 27, {{"float", value}}, "nounwind readnone");
        };
        auto dimension = [&](const string &dims, unsigned j) {
            return ir.value("extractvalue %dx.types.Dimensions " + dims + ", " + to_string(j));
        };
        auto last = [&](const string &count) {
            return select(cmp("ne", count, "0"), ir.binary("sub", count, "1"), "0");
        };
        auto max_lod = ir.value("uitofp i32 " + to_string(words[1] >> 12) + " to float");
        max_lod = ir.value("fmul float " + max_lod + ", 3.906250e-03"); // unsigned 4.8 MAX_LOD
        auto bounded = clamp_float(safe_float(2), max_lod);
        auto rounded = ir.value("fptoui float " + floor(ir.value("fadd float " + bounded + ", 0.5")) + " to i32");
        auto count = dimension(texture_dimensions(handle, "0"), 3);
        auto last_mip = last(count);
        auto mip = select(cmp("ult", rounded, last_mip), rounded, last_mip);
        auto dimensions = texture_dimensions(handle, mip);
        auto have_view = ir.value("and i1 " + valid + ", " + cmp("ne", count, "0"));
        std::array<std::array<string, 2>, 2> taps;
        for (unsigned axis = 0; axis < 2; ++axis) {
            auto size = dimension(dimensions, axis);
            auto edge = last(size);
            // Clamping UV first keeps the conversion in range. Outside [0,1]
            // both taps clamp to the edge, as with clamping after flooring.
            auto uv = clamp_float(safe_float(axis), "1.0");
            auto scaled = ir.value("fmul float " + uv + ", " + ir.value("uitofp i32 " + size + " to float"));
            auto origin = ir.value("fptosi float " + floor(ir.value("fsub float " + scaled + ", 0.5")) + " to i32");
            for (unsigned j = 0; j < 2; ++j) {
                auto index = j ? ir.binary("add", origin, "1") : origin;
                auto nonnegative = select(cmp("slt", index, "0"), "0", index);
                taps[axis][j] = select(cmp("ugt", nonnegative, edge), edge, nonnegative);
            }
        }
        unsigned channel = 0;
        while (!(mask & (1u << channel))) ++channel;
        constexpr unsigned x[]{0, 1, 1, 0}, y[]{1, 1, 0, 0};
        for (unsigned j = 0; j < 4; ++j) {
            auto texel = ir.invoke("%dx.types.ResRet.f32", "textureLoad.f32", 66,
                {{"%dx.types.Handle", handle}, {"i32", mip}, {"i32", taps[0][x[j]]}, {"i32", taps[1][y[j]]},
                 {"i32", "undef"}, {"i32", "undef"}, {"i32", "undef"}, {"i32", "undef"}}, "nounwind readonly");
            auto value = ir.value("extractvalue %dx.types.ResRet.f32 " + texel + ", " + to_string(channel));
            vector_write(256 + i.dst + j, select(have_view, ir.value("bitcast float " + value + " to i32"), "0"));
        }
    }
    static bool is_image_atomic(const Inst &i) {
        return i.enc == Encoding::Image && (i.op == 16 || i.op == 17 || i.op == 23);
    }
    void image_atomic(const Inst &i, const State &s) {
        const bool compare_exchange = i.op == 16;
        const bool return_old = (i.a & 0x2000u) != 0; // GLC returns the pre-operation value.
        const unsigned coord = i.b & 255, base = ((i.b >> 16) & 31) * 4;
        const unsigned mask = (i.a >> 8) & 15, dim = (i.a >> 3) & 7;
        if (graphics_input || dim != 1 || (i.a & 0x3c0c6u) || (i.b & 0xffe00000u) ||
            mask != (compare_exchange ? 3u : 1u) || coord == 255 || (compare_exchange && i.dst == 255) ||
            s.at(base).type != Symbol::Type::descriptor)
            refuse(i.pc, "unsupported typed 2D image atomic operands");
        constexpr auto kind = ComputeViewKind::uint_texture_uav;
        analysis.binding(i, s, base, 8, kind, 0, dim);
        const auto &bindings = analysis.result.bindings;
        const auto &descriptor = s.at(base);
        const auto it = std::find_if(bindings.begin(), bindings.end(), [&](const auto &b) {
            return b.kind == kind && b.root_parameter == descriptor.value && b.byte_offset == descriptor.offset;
        });
        if (it == bindings.end())
            throw std::logic_error("missing image atomic binding");
        const auto handle = ir.handle({1, it->shader_register, 2, 5, 1, 0}); // RWTexture2D<uint>
        const auto label = to_string(label_number++);
        const auto enabled = active();
        ir.body << "  br i1 " << enabled << ", label %imageAtomic" << label << ", label %imageAtomicEnd" << label
                << "\nimageAtomic" << label << ":\n";
        std::vector<std::pair<string, string>> args{{"%dx.types.Handle", handle}};
        if (!compare_exchange)
            args.emplace_back("i32", i.op == 17 ? "0" : "7"); // AtomicBinOp IADD or UMAX.
        args.emplace_back("i32", read(256 + coord));
        args.emplace_back("i32", read(257 + coord));
        args.emplace_back("i32", "undef");
        if (compare_exchange)
            args.emplace_back("i32", read(257 + i.dst));
        args.emplace_back("i32", read(256 + i.dst));
        const auto previous = ir.invoke("i32", compare_exchange ? "atomicCompareExchange.i32" : "atomicBinOp.i32",
                                        compare_exchange ? 79 : 78, args, "nounwind");
        if (return_old)
            write(256 + i.dst, previous);
        ir.body << "  br label %imageAtomicEnd" << label << "\nimageAtomicEnd" << label << ":\n";
    }
    void image_io(const Inst &i, const State &s) {
        if (is_image_atomic(i)) {
            image_atomic(i, s);
            return;
        }
        const bool bias = i.op == 37, gradient = i.op == 34, query = i.op == 96, dimensions = i.op == 14;
        // image_gather4_c_lz (79) is the comparison gather at the view's first mip (#2969): DXIL textureGatherCmp.
        const bool compare_gather = i.op == 79;
        const bool compare = i.op == 40 || i.op == 47 || compare_gather;
        const bool implicit = i.op == 32 || i.op == 40 || bias || query || i.op == 64;
        if (implicit && !pixel_stage)
            refuse(i.pc, "implicit derivatives require a pixel stage");
        const bool gather = i.op == 64 || i.op == 71 || compare_gather;
        const bool gather_lod = i.op == 68;
        const bool sample = (!gather && implicit) || (compare && !gather) || gradient || i.op == 39 || i.op == 36,
                   store = i.op == 8,
                   half = store && (i.b >> 31);
        const bool lod = i.op == 1 || i.op == 36;
        const unsigned mask = (i.a >> 8) & 15, addr = i.b & 255, base = ((i.b >> 16) & 31) * 4, dim = (i.a >> 3) & 7;
        constexpr unsigned dx_kind[]{1, 2, 4, 5, 6, 7, 3, 8};
        constexpr unsigned coord_count[]{1, 2, 3, 3, 2, 3, 2, 3};
        const unsigned coords = coord_count[dim];
        if ((i.op != 0 && i.op != 1 && !store && !sample && !gather && !gather_lod && !dimensions) || !mask || (i.a & 0x3e0c0u) ||
            (i.b & 0x3c000000u) || (gather && ((mask & (mask - 1)) || (dim != 1 && dim != 3 && dim != 5))) ||
            (query && ((dim != 1 && dim != 3) || (mask & ~3u))) ||
            (compare && (mask != 1 || (dim != 1 && dim != 3 && dim != 5) || (i.a & 0x1000u))) ||
            (compare_gather && dim == 3) ||
            ((sample || gather) && (i.a & 0x1000u) && gradient) ||
            (dimensions && (dim == 4 || dim >= 6 || (i.b & 0x40000000u))) ||
            (store && !half && mask != 15) || (half && mask != 1 && mask != 3 && mask != 15) ||
            s.at(base).type != Symbol::Type::descriptor)
            refuse(i.pc, "direct image opcode or descriptor modifiers");
        if (gather_lod && (dim != 1 || (mask & (mask - 1)) || (i.a & 0x1006u) || (i.b & 0xc0000000u) || i.dst > 252))
            refuse(i.pc, "unsupported explicit 2D gather operands");
        if ((dim == 3 && ((!sample && !gather && !dimensions) || (i.a & 0x1000u))) ||
            (dim >= 6 && (sample || gather || store || lod)))
            refuse(i.pc, "unsupported cube or multisample operation");
        const bool d16_sample = !store && (i.b >> 31);
        const unsigned channels = half && mask == 1 ? 1 : half && mask == 3 ? 2 : 4;
        const bool cube_sample = dim == 3 && (sample || gather);
        const auto kind = half    ? (channels == 1 ? ComputeViewKind::half_scalar_uav :
                                     channels == 2 ? ComputeViewKind::half2_texture_uav : ComputeViewKind::half4_texture_uav)
                          : store ? ComputeViewKind::texture_uav
                          : cube_sample ? (compare ? ComputeViewKind::cube_comparison_srv : ComputeViewKind::cube_sample_srv)
                          : d16_sample ? ComputeViewKind::d16_sample_srv
                          : compare ? ComputeViewKind::comparison_texture_srv
                                  : ComputeViewKind::texture_srv;
        analysis.binding(i, s, base, 8, kind, 0, dim);
        const auto &bs = analysis.result.bindings;
        auto it = std::find_if(bs.begin(), bs.end(), [&](const auto &b) {
            return !b.array_position && b.image_dimension == dim && b.kind == kind &&
                   b.root_parameter == s[base].value && b.byte_offset == s[base].offset;
        });
        if (it == bs.end())
            throw std::logic_error("missing image binding");
        auto h = ir.handle({unsigned(store), it->shader_register, cube_sample ? 9u : dx_kind[dim], 9, compare ? 1u : channels, 0});
        auto coordinate = [&](unsigned j) {
            unsigned r = addr + j;
            if (j && (i.a & 6)) {
                const unsigned word_index = (j - 1) / 4;
                if (word_index >= ((i.a >> 1) & 3))
                    refuse(i.pc, "incomplete image NSA address list");
                r = (i.image_address_words[word_index] >> (8 * ((j - 1) % 4))) & 255;
            }
            if (r > 255)
                refuse(i.pc, "image coordinate register overflow");
            return read(256 + r);
        };
        if (gather_lod) {
            explicit_gather_2d(i, s, h, {coordinate(0), coordinate(1), coordinate(2)}, mask);
            return;
        }
        if (dimensions) {
            auto info = texture_dimensions(h, coordinate(0));
            unsigned dst = i.dst;
            for (unsigned ch = 0; ch < 4; ++ch) {
                if (!(mask & (1u << ch))) continue;
                if (dst > 255) refuse(i.pc, "image destination overflow");
                auto value = ch == 3 || ch < (dim == 3 ? 2 : coords) ? ir.value("extractvalue %dx.types.Dimensions " + info + ", " + to_string(ch)) : "1";
                vector_write(256 + dst++, value);
            }
            return;
        }
        std::array<string, 4> c{"undef", "undef", "undef", "undef"};
        string level = "0";
        const unsigned spatial = coords - unsigned(dim == 3 || dim == 4 || dim == 5);
        const bool a16 = (i.b & 0x40000000u) != 0;
        if (a16) {
            if (dim != 1 || lod || (i.a & 6) || (sample && i.op != 32 && i.op != 39) || gather || (store && !half))
                refuse(i.pc, "packed image addressing");
            auto packed = coordinate(0);
            c[0] = ir.binary("and", packed, "65535");
            c[1] = ir.binary("lshr", packed, "16");
        } else {
            for (unsigned j = 0; j < coords; ++j)
                c[j] = coordinate(j + unsigned(bias || compare) + (gradient ? 2 * spatial : 0));
            if (bias || compare)
                level = coordinate(0);
            if (lod || dim >= 6)
                level = coordinate(coords);
        }
        if (!store) {
            string r;
            if (sample || gather) {
                const auto sampler_ref = analysis.sampler_index(i, s, sample || implicit, i.op == 71 || compare_gather,
                                                                implicit || gradient, compare);
                if (i.op == 64 && !sampler_ref.heap &&
                    (analysis.result.samplers.at(sampler_ref.index - analysis.first_sampler).words[1] & 0xffffff))
                    refuse(i.pc, "implicit gather requires a sampler fixed to mip zero");
                string sampler;
                if (sampler_ref.heap) {
                    ir.features |= 0x80000000;
                    sampler = ir.sampler_heap_handle(to_string(sampler_ref.index));
                } else
                    sampler = ir.handle({3, sampler_ref.index, 14, 0, 0, 0, 1, compare});
                using Arg = std::pair<string, string>;
                std::vector<Arg> args{{"%dx.types.Handle", h}, {"%dx.types.Handle", sampler}};
                const auto size = (i.a & 0x1000u) ? texture_dimensions(h, "0") : string{};
                for (unsigned j = 0; j < coords; ++j) {
                    c[j] = a16 ? ir.invoke("float", "legacyF16ToF32", 131, {{"i32", c[j]}}, "nounwind readnone") :
                                 ir.value("bitcast i32 " + c[j] + " to float");
                    if (!size.empty() && j < spatial) {
                        auto extent = ir.value("extractvalue %dx.types.Dimensions " + size + ", " + to_string(j));
                        auto scale = ir.value("uitofp i32 " + extent + " to float");
                        c[j] = ir.value("fdiv float " + c[j] + ", " + scale);
                    }
                }
                string face;
                if (dim == 3) {
                    // GFX packs face + 8 * layer. Desktop cube arrays take a
                    // direction plus the layer relative to the bound view.
                    face = ir.value("fptosi float " + c[2] + " to i32");
                    auto projected = [&](const string &a) {
                        return ir.value("fsub float " + ir.value("fmul float " + a + ", 2.0") + ", 3.0");
                    };
                    auto direction = cube_axes(face, projected(c[0]), projected(c[1]), false);
                    std::copy(direction.begin(), direction.end(), c.begin());
                    c[3] = ir.value("sitofp i32 " + ir.binary("ashr", face, "3") + " to float");
                }
                if (query) {
                    for (unsigned j = 0; j < 3; ++j)
                        args.emplace_back("float", c[j]);
                    args.emplace_back("i1", "true");
                    unsigned dst = i.dst;
                    for (unsigned ch = 0; ch < 2; ++ch) {
                        if (!(mask & (1u << ch))) continue;
                        if (dst > 255) refuse(i.pc, "image destination overflow");
                        args.back().second = ch ? "false" : "true";
                        auto result = ir.invoke("float", "calculateLOD.f32", 81, args, "nounwind readonly");
                        vector_write(256 + dst++, ir.value("bitcast float " + result + " to i32"));
                    }
                    return;
                }
                for (unsigned j = 0; j < 4; ++j)
                    args.emplace_back("float", c[j]);
                const unsigned offsets = dim == 3 ? 0 : spatial;
                for (unsigned j = 0; j < (sample ? 3u : 2u); ++j)
                    args.emplace_back("i32", j < offsets ? "0" : "undef");
                if (gather) {
                    unsigned channel = 0;
                    while (!(mask & (1u << channel))) ++channel;
                    args.emplace_back("i32", to_string(channel));
                    // textureGatherCmp takes the reference after the channel; the comparison reads red.
                    if (compare_gather)
                        args.emplace_back("float", ir.value("bitcast i32 " + level + " to float"));
                } else if (compare) {
                    args.emplace_back("float", ir.value("bitcast i32 " + level + " to float"));
                    if (implicit)
                        args.emplace_back("float", "undef");
                } else if (gradient) {
                    // Victory emit/lower.rs: d/dx and then d/dy of spatial
                    // coordinates precede the coordinates, excluding array slices.
                    for (unsigned axis = 0; axis < 2; ++axis) {
                        std::array<string, 3> d{"undef", "undef", "undef"};
                        for (unsigned j = 0; j < spatial; ++j)
                            d[j] = ir.value("bitcast i32 " + coordinate(axis * spatial + j) + " to float");
                        if (dim == 3)
                            d = cube_axes(face, ir.value("fmul float " + d[0] + ", 2.0"),
                                          ir.value("fmul float " + d[1] + ", 2.0"), true);
                        for (const auto &component : d) args.emplace_back("float", component);
                    }
                    args.emplace_back("float", "undef");
                } else if (implicit) {
                    if (bias)
                        args.emplace_back("float", ir.value("bitcast i32 " + level + " to float"));
                    args.emplace_back("float", "undef"); // no per-instruction minimum LOD clamp
                } else
                    args.emplace_back("float", lod ? ir.value("bitcast i32 " + level + " to float") : "0.0");
                const char *method = compare_gather ? "textureGatherCmp.f32" :
                                     compare ? (implicit ? "sampleCmp.f32" : "sampleCmpLevelZero.f32") :
                                     gather ? "textureGather.f32" : gradient ? "sampleGrad.f32" :
                                     bias ? "sampleBias.f32" : implicit ? "sample.f32" : "sampleLevel.f32";
                const unsigned opcode = compare_gather ? 74 : compare ? (implicit ? 64 : 65) : gather ? 73 :
                                        gradient ? 63 : bias ? 61 : implicit ? 60 : 62;
                r = ir.invoke("%dx.types.ResRet.f32", method, opcode, args, "nounwind readonly");
            } else
                r = ir.invoke("%dx.types.ResRet.f32", "textureLoad.f32", 66,
                              {{"%dx.types.Handle", h},
                               {"i32", level},
                               {"i32", c[0]},
                               {"i32", c[1]},
                               {"i32", c[2]},
                               {"i32", "undef"},
                               {"i32", "undef"},
                               {"i32", "undef"}},
                              "nounwind readonly");
            unsigned dst = i.dst;
            string pending_half;
            for (unsigned c = 0; c < 4; ++c)
                if (gather || (mask & (1u << c))) {
                    if (dst > 255)
                        refuse(i.pc, "image destination overflow");
                    auto value = ir.value("extractvalue %dx.types.ResRet.f32 " + r + ", " + to_string(c));
                    auto bits = ir.value("bitcast float " + value + " to i32");
                    if (d16_sample) {
                        bits = packed_half_rne(bits);
                        if (pending_half.empty()) {
                            pending_half = bits;
                            continue;
                        }
                        bits = ir.binary("or", pending_half, ir.binary("shl", bits, "16"));
                        pending_half.clear();
                    }
                    vector_write(256 + dst++, bits);
                }
            if (!pending_half.empty()) vector_write(256 + dst, pending_half);
        } else {
            if (i.dst + (half ? (channels + 1) / 2 : channels) > 256)
                refuse(i.pc, "image data overflow");
            string label = to_string(label_number++), pred = active();
            if (!graphics_input)
                ir.body << "  br i1 " << pred << ", label %img" << label << ", label %imgover" << label << "\nimg" << label
                    << ":\n";
            std::vector<std::pair<string, string>> args{
                {"%dx.types.Handle", h}, {"i32", c[0]}, {"i32", c[1]}, {"i32", c[2]}};
            std::array<string, 4> values;
            string packed;
            for (unsigned c = 0; c < channels; ++c) {
                if (half) {
                    if (!(c & 1)) packed = read(256 + i.dst + c / 2);
                    const auto bits = c & 1 ? ir.binary("lshr", packed, "16") : packed;
                    values[c] = ir.invoke("float", "legacyF16ToF32", 131, {{"i32", bits}}, "nounwind readnone");
                } else
                    values[c] = ir.value("bitcast i32 " + read(256 + i.dst + c) + " to float");
            }
            // DXIL typed stores use a full write mask; narrower views ignore the tail.
            for (unsigned c = 0; c < 4; ++c)
                args.emplace_back("float", values[(std::min)(c, channels - 1)]);
            args.emplace_back("i8", "15");
            ir.invoke("void", "textureStore.f32", 67, args, "nounwind");
            if (!graphics_input)
                ir.body << "  br label %imgover" << label << "\nimgover" << label << ":\n";
        }
    }
    void local_memory(const Inst &i, unsigned words) {
        const auto op = decode_local_data(i.a, i.b);
        // The host executes a guest wave64 on two wave32s. Meet before
        // each operation, outside its lane predicate, as in Victory split_lds.
        if (wave64)
            barrier();
        const auto label = to_string(label_number++);
        auto enabled = active();
        ir.body << "  br i1 " << enabled << ", label %ds" << label << ", label %dsDone" << label << "\nds" << label
                << ":\n";
        const auto address = read(256 + op.address);
        for (unsigned j = 0; j < op.count; ++j) {
            auto index = ir.binary("lshr", ir.binary("add", address, to_string(op.byte_offset(j))), "2");
            auto ptr = shared_address("lds", words, index);
            if (op.read)
                write(256 + op.destination + j, ir.value("load i32, i32 addrspace(3)* " + ptr + ", align 4"));
            else {
                auto data = read(256 + op.data_register(j));
                if (op.atomic_ir) {
                    if (op.subtract)
                        data = ir.binary("sub", "0", data);
                    auto old = ir.value(string("atomicrmw ") + op.atomic_ir + " i32 addrspace(3)* " + ptr +
                                        ", i32 " + data + " monotonic");
                    if (op.return_old)
                        write(256 + op.destination, old);
                } else
                    ir.body << "  store i32 " << data << ", i32 addrspace(3)* " << ptr << ", align 4\n";
            }
        }
        ir.body << "  br label %dsDone" << label << "\ndsDone" << label << ":\n";
    }
    string shift(const string &op, const string &a, const string &b) {
        return ir.binary(op, a, ir.binary("and", b, "31"));
    }
    string subword(const string &value, unsigned sel, bool sign) {
        if (sel == 6)
            return value;
        unsigned width = sel < 4 ? 8 : 16, bit = sel < 4 ? 8 * sel : 16 * (sel - 4);
        auto slice = ir.binary("lshr", value, to_string(bit));
        if (!sign)
            return ir.binary("and", slice, to_string((1u << width) - 1));
        auto left = ir.binary("shl", slice, to_string(32 - width));
        return ir.binary("ashr", left, to_string(32 - width));
    }
    string place_subword(const Inst &i, const string &value) {
        if (i.dst_select == 6)
            return value;
        unsigned sel = i.dst_select, width = sel < 4 ? 8 : 16, bit = sel < 4 ? 8 * sel : 16 * (sel - 4);
        Word mask = (1u << width) - 1;
        string result = subword(value, sel < 4 ? 0 : 4, i.dst_unused == 1);
        if (bit)
            result = ir.binary("shl", result, to_string(bit));
        if (i.dst_unused == 2)
            result = ir.binary("or", result, ir.binary("and", read(i.dst), to_string(~(mask << bit))));
        return result;
    }
    string cube_alu(Name op, const string &x, const string &y, const string &z) {
        auto magnitude = [&](const string &v) {
            return ir.invoke("float", "unary.f32", 6, {{"float", v}}, "nounwind readnone");
        };
        auto at_least = [&](const string &a, const string &b) { return ir.value("fcmp oge float " + a + ", " + b); };
        auto negative = [&](const string &a) { return ir.value("fcmp olt float " + a + ", 0.0"); };
        auto negate = [&](const string &a) { return ir.value("fsub float -0.0, " + a); };
        const auto ax = magnitude(x), ay = magnitude(y), az = magnitude(z);
        // The tie order is Z, Y, X, as in pixel_aot's cube lowering.
        const auto major_z = ir.value("and i1 " + at_least(az, ax) + ", " + at_least(az, ay));
        const auto major_y = at_least(ay, ax);
        auto axis = [&](const string &xf, const string &yf, const string &zf) {
            return select_float(major_z, zf, select_float(major_y, yf, xf));
        };
        if (op == "v_cubeid_f32")
            return axis(select_float(negative(x), "1.0", "0.0"), select_float(negative(y), "3.0", "2.0"),
                        select_float(negative(z), "5.0", "4.0"));
        if (op == "v_cubesc_f32")
            return axis(select_float(negative(x), z, negate(z)), x, select_float(negative(z), negate(x), x));
        if (op == "v_cubetc_f32") {
            const auto ny = negate(y);
            return axis(ny, select_float(negative(y), negate(z), z), ny);
        }
        return ir.value("fmul float " + axis(x, y, z) + ", 2.0");
    }
    string packed_half_rne(const string &bits) {
        // Round the integer significand to nearest even. Some desktop
        // implementations truncate legacyF32ToF16 (Victory 9376fd6,
        // emit/lower.rs::f16_nearest). This also preserves subnormals.
        const auto sign = ir.binary("and", ir.binary("lshr", bits, "16"), "32768");
        const auto mag = ir.binary("and", bits, "2147483647");
        const auto exp = ir.binary("lshr", mag, "23");
        const auto frac = ir.binary("and", mag, "8388607");
        auto round_shift = [&](const string &v, const string &shift, const string &half) {
            const auto odd = ir.binary("and", ir.binary("lshr", v, shift), "1");
            const auto bias = ir.binary("add", ir.binary("sub", half, "1"), odd);
            return ir.binary("lshr", ir.binary("add", v, bias), shift);
        };
        const auto normal = round_shift(ir.binary("sub", mag, "939524096"), "13", "4096");
        const auto sub_range = ir.value("and i1 " + cmp("uge", exp, "102") + ", " + cmp("ule", exp, "112"));
        const auto shift = select(sub_range, ir.binary("sub", "126", exp), "14");
        const auto half = ir.binary("shl", "1", ir.binary("sub", shift, "1"));
        const auto sub = round_shift(ir.binary("or", frac, "8388608"), shift, half);
        auto h = select(cmp("ule", exp, "112"), select(sub_range, sub, "0"), normal);
        h = select(cmp("ugt", exp, "142"), "31744", h);
        const auto nan = cmp("ugt", mag, "2139095040");
        h = select(nan, ir.binary("or", ir.binary("lshr", frac, "13"), "32256"), h);
        return ir.binary("or", h, sign);
    }
    static bool packed_f16_arithmetic(Name n) {
        return n == "v_pk_add_f16" || n == "v_pk_mul_f16" || n == "v_pk_fma_f16";
    }
    string fused_half_rne(const string &a, const string &b, const string &c) {
        auto both = [&](const string &x, const string &y) { return ir.value("and i1 " + x + ", " + y); };
        auto either = [&](const string &x, const string &y) { return ir.value("or i1 " + x + ", " + y); };
        auto clz = [&](const string &value) {
            // FirstbitHi counts from the most significant bit. OR-ing in bit
            // zero leaves every nonzero count unchanged and bounds zero's shift.
            return ir.invoke("i32", "unaryBits.i32", 33,
                             {{"i32", ir.binary("or", value, "1")}}, "nounwind readnone");
        };
        struct Half { string magnitude, sign, significand, exponent; };
        auto decode = [&](const string &word) {
            auto magnitude = ir.binary("and", word, "32767");
            auto field = ir.binary("lshr", magnitude, "10");
            auto normal = cmp("ne", field, "0");
            auto hidden = select(normal, "1024", "0");
            return Half{magnitude, ir.binary("and", word, "32768"),
                        ir.binary("or", ir.binary("and", word, "1023"), hidden),
                        ir.binary("sub", select(normal, field, "1"), "25")};
        };
        auto x = decode(a), y = decode(b), z = decode(c);
        auto product_sign = ir.binary("xor", x.sign, y.sign);
        auto product = ir.binary("mul", x.significand, y.significand);
        auto product_exp = ir.binary("add", x.exponent, y.exponent);

        // Normalize both exact terms at bit 30, leaving space for a carry.
        // The product has at most 22 bits and the addend at most 11. Close
        // exponents therefore align exactly, including after cancellation.
        auto normalize = [&](const string &mantissa, const string &exponent) {
            auto leading = clz(mantissa);
            return std::pair{
                ir.binary("shl", mantissa, ir.binary("sub", leading, "1")),
                select(cmp("eq", mantissa, "0"), "-128",
                       ir.binary("add", exponent, ir.binary("sub", "31", leading)))};
        };
        auto [product_mantissa, product_top] = normalize(product, product_exp);
        auto [addend_mantissa, addend_top] = normalize(z.significand, z.exponent);
        auto product_larger = either(cmp("sgt", product_top, addend_top),
            both(cmp("eq", product_top, addend_top), cmp("uge", product_mantissa, addend_mantissa)));
        auto big = select(product_larger, product_mantissa, addend_mantissa);
        auto smaller_mantissa = select(product_larger, addend_mantissa, product_mantissa);
        auto big_exp = select(product_larger, product_top, addend_top);
        auto small_exp = select(product_larger, addend_top, product_top);
        auto sign = select(product_larger, product_sign, z.sign);
        auto distance = ir.binary("sub", big_exp, small_exp);
        auto shift = select(cmp("ult", distance, "31"), distance, "31");
        auto aligned = ir.binary("lshr", smaller_mantissa, shift);
        // A wide alignment has no significant cancellation. Retain a sticky
        // bit for discarded data so a nearby midpoint still rounds correctly.
        auto sticky = cmp("ne", ir.binary("shl", aligned, shift), smaller_mantissa);
        aligned = ir.binary("or", aligned, ir.value("zext i1 " + sticky + " to i32"));
        auto same_sign = cmp("eq", product_sign, z.sign);
        auto magnitude = select(same_sign, ir.binary("add", big, aligned), ir.binary("sub", big, aligned));
        auto leading = clz(magnitude);
        auto normalized = ir.binary("shl", magnitude, leading);
        auto exponent = ir.binary("sub", ir.binary("add", big_exp, "128"), leading);

        // Encode the exact sum rounded to odd at binary32 precision. If any
        // bits were lost, set the last retained bit. The final binary16
        // nearest-even conversion then sees the correct side of every tie.
        auto odd = cmp("ne", ir.binary("and", normalized, "255"), "0");
        auto significand = ir.binary("or", ir.binary("lshr", normalized, "8"),
                                    ir.value("zext i1 " + odd + " to i32"));
        auto binary32 = ir.binary("or", ir.binary("shl", sign, "16"),
            ir.binary("or", ir.binary("shl", exponent, "23"), ir.binary("and", significand, "8388607")));
        auto zero = select(both(cmp("eq", product, "0"), cmp("eq", z.magnitude, "0")),
                           ir.binary("and", product_sign, z.sign), "0");
        auto finite = select(cmp("eq", magnitude, "0"), zero, packed_half_rne(binary32));

        auto x_inf = cmp("eq", x.magnitude, "31744"), y_inf = cmp("eq", y.magnitude, "31744");
        auto product_inf = either(x_inf, y_inf), c_inf = cmp("eq", z.magnitude, "31744");
        auto invalid_product = either(both(x_inf, cmp("eq", y.magnitude, "0")),
                                      both(y_inf, cmp("eq", x.magnitude, "0")));
        auto invalid_sum = both(both(product_inf, c_inf), cmp("ne", product_sign, z.sign));
        auto nan = either(either(cmp("ugt", x.magnitude, "31744"), cmp("ugt", y.magnitude, "31744")),
                          either(cmp("ugt", z.magnitude, "31744"), either(invalid_product, invalid_sum)));
        return select(nan, "32256",
            select(product_inf, ir.binary("or", product_sign, "31744"), select(c_inf, c, finite)));
    }
    void packed_f16_alu(const Inst &i) {
        const bool fused = i.name == "v_pk_fma_f16";
        if (i.enc != Encoding::Mix || i.clamp || (!fused && (i.src[2] || (i.a & 0x2400) || (i.neg & 4))))
            refuse(i.pc, "unsupported packed binary16 arithmetic encoding");
        if (!half_mode_known || ((mode_bits ^ 0xf0u) & 0x8000ccu))
            refuse(i.pc, "unsupported binary16 rounding or denormal mode");
        std::array<string, 3> words;
        for (unsigned j = 0; j < (fused ? 3u : 2u); ++j) {
            unsigned r = i.src[j];
            if (r > 105 && r < 256 && r != 128 && r != 255)
                refuse(i.pc, "unsupported packed binary16 arithmetic source");
            words[j] = operand32(i, r);
        }
        auto lane = [&](unsigned selectors, unsigned negate) {
            auto operand = [&](unsigned j) {
                auto bits = (selectors & (1u << j)) ? ir.binary("lshr", words[j], "16")
                                                    : ir.binary("and", words[j], "65535");
                return (negate & (1u << j)) ? ir.binary("xor", bits, "32768") : bits;
            };
            auto a = operand(0), b = operand(1);
            if (fused)
                return fused_half_rne(a, b, operand(2));
            auto widen = [&](const string &bits) {
                return ir.invoke("float", "legacyF16ToF32", 131, {{"i32", bits}}, "nounwind readnone");
            };
            const auto op = i.name == "v_pk_mul_f16" ? "fmul" : "fadd";
            auto result = ir.value(string(op) + " float " + widen(a) + ", " + widen(b));
            auto bits = packed_half_rne(ir.value("bitcast float " + result + " to i32"));
            return select(cmp("ugt", ir.binary("and", bits, "32767"), "31744"), "32256", bits);
        };
        // In VOP3P, NEG_HI occupies the ABS field. OP_SEL_HI[2] lives in word 0.
        auto low = lane((i.a >> 11) & 7, i.neg);
        auto high = lane(((i.b >> 27) & 3) | ((i.a >> 12) & 4), i.abs);
        vector_write(i.dst, ir.binary("or", low, ir.binary("shl", high, "16")));
    }
    bool float_alu(const Inst &i, const string *permuted = nullptr) {
        auto n = i.name;
        if (packed_f16_arithmetic(n)) {
            packed_f16_alu(i);
            return true;
        }
        if ((i.subword || (!i.wide && !i.dpp_control && !permuted)) &&
            (n == "v_cvt_f16_f32" || n == "v_cvt_f32_f16")) {
            if (i.scale || i.clamp || i.src_sext[0])
                refuse(i.pc, "unsupported SDWA half conversion modifier");
            string bits = operand32(i, i.src[0]);
            if (i.subword) bits = subword(bits, i.src_select[0], false);
            const bool narrow = n == "v_cvt_f16_f32";
            if (i.abs) bits = ir.binary("and", bits, narrow ? "2147483647" : "32767");
            if (i.neg) bits = ir.binary("xor", bits, narrow ? "2147483648" : "32768");
            if (narrow) {
                bits = packed_half_rne(bits);
                if (!i.subword) bits = ir.binary("or", bits, ir.binary("and", read(i.dst), "4294901760"));
            } else {
                auto f = ir.invoke("float", "legacyF16ToF32", 131, {{"i32",bits}}, "nounwind readnone");
                bits = ir.value("bitcast float " + f + " to i32");
            }
            vector_write(i.dst, place_subword(i, bits));
            return true;
        }
        if (n == "v_pack_b32_f16") {
            // VOP3 only: D[15:0] = S0.f16, D[31:16] = S1.f16. Every step is an
            // integer bit operation so signed zeros and NaN payloads survive.
            // Unsupported forms: clamp, omod, a destination or third-source
            // op_sel bit, third-source abs/neg, inline constants, literals,
            // SDWA and DPP. They are refused rather than approximated.
            if (!i.wide || i.enc != Encoding::Vop3 || i.subword || i.dpp_control || permuted)
                refuse(i.pc, "v_pack_b32_f16 requires the VOP3 encoding");
            if (i.clamp || i.scale)
                refuse(i.pc, "unsupported v_pack_b32_f16 clamp or omod");
            if ((i.a & 0x6000) || ((i.abs | i.neg) & 4))
                refuse(i.pc, "unsupported v_pack_b32_f16 destination or third-source modifier");
            auto half = [&](unsigned j) {
                const unsigned r = i.src[j];
                if (r >= 128 && r < 256)
                    refuse(i.pc, "unsupported v_pack_b32_f16 constant source");
                auto bits = operand32(i, r);
                if ((i.a >> (11 + j)) & 1) bits = ir.binary("lshr", bits, "16");
                bits = ir.binary("and", bits, "65535");
                // Clear the sign for abs first, then toggle it for neg.
                if (i.abs & (1u << j)) bits = ir.binary("and", bits, "32767");
                if (i.neg & (1u << j)) bits = ir.binary("xor", bits, "32768");
                return bits;
            };
            auto low = half(0), high = half(1);
            vector_write(i.dst, ir.binary("or", low, ir.binary("shl", high, "16")));
            return true;
        }
        if (n == "v_frexp_exp_i16_f16" || n == "v_frexp_mant_f16") {
            const bool exp_op = n == "v_frexp_exp_i16_f16";
            if (i.subword || i.dpp_control || permuted || i.scale || (exp_op && i.clamp) ||
                (i.wide && ((i.a & 0x7800) || ((i.abs | i.neg) & 6) || i.src[1] || i.src[2])))
                refuse(i.pc, "unsupported " + string(n) + " operand modifiers");
            // An inline float (240-248) reads its binary16 value, as v_add_f16 does.
            const unsigned r = i.src[0];
            auto bits = r >= 240 && r <= 248 ? to_string(inline_half_words[r - 240])
                                             : ir.binary("and", operand32(i, r), "65535");
            if (i.abs & 1) bits = ir.binary("and", bits, "32767");
            if (i.neg & 1) bits = ir.binary("xor", bits, "32768");
            auto sign = ir.binary("and", bits, "32768");
            auto exponent = ir.binary("and", ir.binary("lshr", bits, "10"), "31");
            auto mantissa = ir.binary("and", bits, "1023");
            auto keep = ir.value("or i1 " + cmp("eq", exponent, "31") + ", " +
                                 cmp("eq", ir.binary("and", bits, "32767"), "0"));
            string x = mantissa, steps = "0";
            for (unsigned k : {8u, 4u, 2u, 1u}) {
                auto low = cmp("ult", x, to_string(1u << (11 - k)));
                x = select(low, ir.binary("shl", x, to_string(k)), x);
                steps = select(low, ir.binary("add", steps, to_string(k)), steps);
            }
            auto denormal = cmp("eq", exponent, "0");
            string result;
            if (exp_op) {
                auto e = select(denormal, ir.binary("sub", "1", steps), exponent);
                result = select(keep, "0", ir.binary("sub", e, "14"));
            } else {
                auto frac = select(denormal, ir.binary("and", x, "1023"), mantissa);
                result = select(keep, bits, ir.binary("or", sign, ir.binary("or", "14336", frac)));
                if (i.clamp) {
                    auto nan = cmp("ugt", ir.binary("and", result, "32767"), "31744");
                    auto low = ir.value("or i1 " + nan + ", " + cmp("ne", ir.binary("and", result, "32768"), "0"));
                    result = select(low, "0", select(cmp("ugt", result, "15359"), "15360", result));
                }
            }
            auto kept = ir.binary("and", read(i.dst), "4294901760");
            vector_write(i.dst, ir.binary("or", kept, ir.binary("and", result, "65535")));
            return true;
        }
        if (n == "v_cvt_f16_u16" || n == "v_cvt_f16_i16" || n == "v_cvt_u16_f16" || n == "v_cvt_i16_f16") {
            if (graphics_input)
                return false;
            const bool to_f16 = n == "v_cvt_f16_u16" || n == "v_cvt_f16_i16";
            const bool is_signed = n == "v_cvt_f16_i16" || n == "v_cvt_i16_f16";
            if (i.dpp_control || permuted)
                refuse(i.pc, "unsupported " + string(n) + " DPP form");
            if (i.subword)
                refuse(i.pc, "unsupported " + string(n) + " SDWA form");
            if (i.scale || (!to_f16 && i.clamp))
                refuse(i.pc, "unsupported " + string(n) + " clamp or omod");
            if (i.wide && (i.a & 0x7800))
                refuse(i.pc, "unsupported " + string(n) + " high-half op_sel");
            if ((to_f16 && (i.abs || i.neg)) || (i.wide && (((i.abs | i.neg) & 6) || i.src[1] || i.src[2])))
                refuse(i.pc, "unsupported " + string(n) + " operand modifiers");
            if (!half_mode_known || ((mode_bits ^ 0xf0u) & 0x8000ccu))
                refuse(i.pc, "unsupported binary16 rounding or denormal mode");
            const unsigned r = i.src[0];
            string bits;
            if (r >= 240 && r <= 248) {
                constexpr unsigned halves[]{0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000,
                                            0xc000, 0x4400, 0xc400, 0x3118};
                bits = to_string(halves[r - 240]);
            } else if (r == 255) {
                bits = to_string(i.literal & 65535u);
            } else if (r >= 128 && r <= 192) {
                bits = to_string((r - 128) & 65535u);
            } else if (r >= 193 && r <= 208) {
                bits = to_string((192u - r) & 65535u);
            } else {
                bits = ir.binary("and", operand32(i, r), "65535");
            }
            string result;
            if (to_f16) {
                string f;
                if (is_signed) {
                    auto s16 = ir.binary("ashr", ir.binary("shl", bits, "16"), "16");
                    f = ir.value("sitofp i32 " + s16 + " to float");
                } else {
                    f = ir.value("uitofp i32 " + bits + " to float");
                }
                auto fbits = ir.value("bitcast float " + f + " to i32");
                result = packed_half_rne(fbits);
                if (i.clamp) {
                    auto nan = cmp("ugt", ir.binary("and", result, "32767"), "31744");
                    auto low = ir.value("or i1 " + nan + ", " + cmp("ne", ir.binary("and", result, "32768"), "0"));
                    result = select(low, "0", select(cmp("ugt", result, "15359"), "15360", result));
                }
            } else {
                if (i.abs & 1)
                    bits = ir.binary("and", bits, "32767");
                if (i.neg & 1)
                    bits = ir.binary("xor", bits, "32768");
                auto f = ir.invoke("float", "legacyF16ToF32", 131, {{"i32", bits}}, "nounwind readnone");
                auto lower = ir.value("fcmp oge float " + f + ", " + (is_signed ? "-32768.0" : "0.0"));
                auto upper = ir.value("fcmp olt float " + f + ", " + (is_signed ? "32768.0" : "65536.0"));
                auto safe = select_float(ir.value("and i1 " + lower + ", " + upper), f, "0.0");
                auto converted = ir.value(string(is_signed ? "fptosi" : "fptoui") + " float " + safe + " to i32");
                auto bounded = select(upper, converted, is_signed ? "32767" : "65535");
                bounded = select(lower, bounded, is_signed ? "32768" : "0");
                auto ordered = ir.value("fcmp ord float " + f + ", " + f);
                result = ir.binary("and", select(ordered, bounded, "0"), "65535");
            }
            auto kept = ir.binary("and", read(i.dst), "4294901760");
            vector_write(i.dst, ir.binary("or", kept, ir.binary("and", result, "65535")));
            return true;
        }
        if (n == "v_ldexp_f32") {
            // pixel_aot.py ldexp32 (#3013): exact binary32 scaling of S0 by 2^S1. Zero, Inf and NaN pass unchanged,
            // a denormal S0 is normalized rather than flushed, and a denormal result rounds to nearest even. S1 is
            // a signed integer clamped to [-512, 512]. clamp saturates the result with integer tests: NaN and
            // negative values (also -0) give +0, values from 1.0 give 1.0, and a denormal stays.
            if (!i.wide || i.subword || i.dpp_control || permuted || i.scale || (i.a & 0x7800) || ((i.abs | i.neg) & 6) ||
                i.src[2])
                refuse(i.pc, "unsupported v_ldexp_f32 operand modifiers");
            auto bits = operand32(i, i.src[0]);
            if (i.abs & 1) bits = ir.binary("and", bits, "2147483647");
            if (i.neg & 1) bits = ir.binary("xor", bits, "2147483648");
            auto delta = operand32(i, i.src[1]);
            auto sign = ir.binary("and", bits, "2147483648");
            auto exponent = ir.binary("and", ir.binary("lshr", bits, "23"), "255");
            auto mantissa = ir.binary("and", bits, "8388607");
            auto keep = ir.value("or i1 " + cmp("eq", exponent, "255") + ", " +
                                 cmp("eq", ir.binary("and", bits, "2147483647"), "0"));
            // Normalize a denormal mantissa: shift it left until bit 23 is set (5 halving steps, at most 23).
            string x = mantissa, steps = "0";
            for (unsigned k : {16u, 8u, 4u, 2u, 1u}) {
                auto low = cmp("ult", x, to_string(1u << (24 - k)));
                x = select(low, ir.binary("shl", x, to_string(k)), x);
                steps = select(low, ir.binary("add", steps, to_string(k)), steps);
            }
            auto denormal = cmp("eq", exponent, "0");
            mantissa = select(denormal, ir.binary("and", x, "8388607"), mantissa);
            auto e = select(denormal, ir.binary("sub", "1", steps), exponent);
            delta = select(cmp("slt", delta, "-512"), "-512", select(cmp("sgt", delta, "512"), "512", delta));
            e = ir.binary("add", e, delta);
            auto normal = ir.binary("or", sign, ir.binary("or", ir.binary("shl", e, "23"), mantissa));
            // A result below the normal range: shift the full significand right by 1 - e (1..24) and round to
            // nearest even.
            auto full = ir.binary("or", mantissa, "8388608");
            // Every shift stays in 1..24, also in the lanes whose result another select discards.
            auto amount = select(cmp("sgt", e, "0"), "1", select(cmp("slt", e, "-23"), "24", ir.binary("sub", "1", e)));
            auto half = ir.binary("shl", "1", ir.binary("sub", amount, "1"));
            auto rest = ir.binary("and", full, ir.binary("sub", ir.binary("shl", half, "1"), "1"));
            auto q = ir.binary("lshr", full, amount);
            auto up = ir.value("or i1 " + cmp("ugt", rest, half) + ", " +
                               ir.value("and i1 " + cmp("eq", rest, half) + ", " +
                                        cmp("ne", ir.binary("and", q, "1"), "0")));
            auto subnormal = ir.binary("or", sign, ir.binary("add", q, ir.value("zext i1 " + up + " to i32")));
            string result = select(cmp("sgt", e, "0"), normal, select(cmp("slt", e, "-23"), sign, subnormal));
            result = select(cmp("sgt", e, "254"), ir.binary("or", sign, "2139095040"), result);
            result = select(keep, bits, result);
            if (i.clamp) {
                auto nan = cmp("ugt", ir.binary("and", result, "2147483647"), "2139095040");
                auto low = ir.value("or i1 " + nan + ", " + cmp("slt", result, "0"));
                result = select(low, "0", select(cmp("sgt", result, "1065353215"), "1065353216", result));
            }
            vector_write(i.dst, result);
            return true;
        }
        if (n == "v_frexp_exp_i32_f32" || n == "v_frexp_mant_f32") {
            const bool exp_op = n == "v_frexp_exp_i32_f32";
            if (i.subword || i.dpp_control || permuted || i.scale || (exp_op && i.clamp) ||
                (i.wide && ((i.a & 0x7800) || ((i.abs | i.neg) & 6) || i.src[1] || i.src[2])))
                refuse(i.pc, "unsupported " + string(n) + " operand modifiers");
            auto bits = operand32(i, i.src[0]);
            if (i.abs & 1) bits = ir.binary("and", bits, "2147483647");
            if (i.neg & 1) bits = ir.binary("xor", bits, "2147483648");
            auto sign = ir.binary("and", bits, "2147483648");
            auto exponent = ir.binary("and", ir.binary("lshr", bits, "23"), "255");
            auto mantissa = ir.binary("and", bits, "8388607");
            auto keep = ir.value("or i1 " + cmp("eq", exponent, "255") + ", " +
                                 cmp("eq", ir.binary("and", bits, "2147483647"), "0"));
            string x = mantissa, steps = "0";
            for (unsigned k : {16u, 8u, 4u, 2u, 1u}) {
                auto low = cmp("ult", x, to_string(1u << (24 - k)));
                x = select(low, ir.binary("shl", x, to_string(k)), x);
                steps = select(low, ir.binary("add", steps, to_string(k)), steps);
            }
            auto denormal = cmp("eq", exponent, "0");
            string result;
            if (exp_op) {
                auto e = select(denormal, ir.binary("sub", "1", steps), exponent);
                result = select(keep, "0", ir.binary("sub", e, "126"));
            } else {
                auto frac = select(denormal, ir.binary("and", x, "8388607"), mantissa);
                result = select(keep, bits, ir.binary("or", sign, ir.binary("or", "1056964608", frac)));
                if (i.clamp) {
                    auto nan = cmp("ugt", ir.binary("and", result, "2147483647"), "2139095040");
                    auto low = ir.value("or i1 " + nan + ", " + cmp("slt", result, "0"));
                    result = select(low, "0", select(cmp("sgt", result, "1065353215"), "1065353216", result));
                }
            }
            vector_write(i.dst, result);
            return true;
        }
        if (n == "v_cvt_pkrtz_f16_f32") {
            if (i.clamp || i.scale || (i.wide && (i.a & 0x7800)))
                refuse(i.pc, "unsupported packed half conversion modifier");
            // SDWA: both sources are full f32 words (src_sel DWORD); only the destination placement differs.
            if (i.subword && (i.src_select[0] != 6 || i.src_select[1] != 6))
                refuse(i.pc, "unsupported packed half conversion SDWA source selection");
            auto pack = [&](unsigned j) {
                auto bits = operand32(i, i.src[j]);
                if (i.abs & (1u << j)) bits = ir.binary("and", bits, "2147483647");
                if (i.neg & (1u << j)) bits = ir.binary("xor", bits, "2147483648");
                auto f = ir.value("bitcast i32 " + bits + " to float");
                auto h = ir.invoke("i32", "legacyF32ToF16", 130, {{"float",f}}, "nounwind readnone");
                auto mag = ir.binary("and", h, "32767");
                auto back = ir.invoke("float", "legacyF16ToF32", 131, {{"i32",mag}}, "nounwind readnone");
                auto abs = ir.value("bitcast i32 " + ir.binary("and", bits, "2147483647") + " to float");
                auto rounded_up = ir.value("fcmp ogt float " + back + ", " + abs);
                return ir.binary("and", ir.binary("sub", h, ir.value("zext i1 " + rounded_up + " to i32")), "65535");
            };
            auto low = pack(0), high = pack(1);
            auto result = ir.binary("or", low, ir.binary("shl", high, "16"));
            vector_write(i.dst, place_subword(i, result));
            return true;
        }
        if (n == "v_add_f16" || n == "v_sub_f16" || n == "v_subrev_f16" || n == "v_mul_f16") {
            // The graphics frontend shares this ALU. Leave those programs on the
            // existing refusal; this lowering is the direct compute path.
            if (graphics_input)
                return false;
            if (i.clamp || i.scale)
                refuse(i.pc, "unsupported " + string(n) + " clamp or omod");
            // op_sel bits 11..14 select a high half. Bit 15 is clamp, already refused.
            if (i.wide && (i.a & 0x7800))
                refuse(i.pc, "unsupported " + string(n) + " high-half op_sel");
            if (!half_mode_known || ((mode_bits ^ 0xf0u) & 0x8000ccu))
                refuse(i.pc, "unsupported binary16 rounding or denormal mode");
            auto half_bits = [&](unsigned j) {
                const unsigned r = i.src[j];
                string bits;
                if (!j && permuted) {
                    bits = ir.binary("and", *permuted, "65535");
                } else if (i.subword) {
                    // A DWORD selection reads the low half for binary16 arithmetic.
                    const unsigned sel = i.src_select[j];
                    bits = subword(operand32(i, r), sel == 6 ? 4 : sel, false);
                } else if (r >= 240 && r <= 248) {
                    // Inline floats are f32 patterns. The f16 operation consumes the
                    // binary16 rounding of that value.
                    bits = to_string(inline_half_words[r - 240]);
                } else if (r == 255)
                    bits = to_string(i.literal & 65535u);
                else if (r >= 128 && r <= 192)
                    bits = to_string((r - 128) & 65535u);
                else if (r >= 193 && r <= 208)
                    bits = to_string((192u - r) & 65535u);
                else
                    bits = ir.binary("and", operand32(i, r), "65535");
                if (i.abs & (1u << j))
                    bits = ir.binary("and", bits, "32767");
                if (i.neg & (1u << j))
                    bits = ir.binary("xor", bits, "32768");
                return bits;
            };
            auto widen = [&](const string &bits) {
                return ir.invoke("float", "legacyF16ToF32", 131, {{"i32", bits}}, "nounwind readnone");
            };
            auto a = widen(half_bits(0));
            auto b = widen(half_bits(1));
            if (n == "v_subrev_f16")
                std::swap(a, b);
            const char *op = n == "v_mul_f16" ? "fmul" : n == "v_add_f16" ? "fadd" : "fsub";
            auto widened = ir.value(string(op) + " float " + a + ", " + b);
            auto bits = ir.value("bitcast float " + widened + " to i32");
            // f32 has a 24-bit significand, which covers a correctly rounded f16
            // add, sub or mul (2*11+2). Round with the same integer nearest-even
            // conversion v_cvt_f16_f32 uses.
            auto rounded = packed_half_rne(bits);
            // D[31:16] keeps its value: LLVM GCNSubtarget::zeroesHigh16BitsOfDest lists these four ops with
            // "On gfx10, all 16-bit instructions preserve the high bits", as v_cvt_f16_f32 above already does.
            vector_write(i.dst, i.subword ? place_subword(i, rounded)
                                          : ir.binary("or", ir.binary("and", read(i.dst), "4294901760"),
                                                      ir.binary("and", rounded, "65535")));
            return true;
        }
        if (n == "v_fma_f16" || n == "v_fmac_f16" || n == "v_fmamk_f16" || n == "v_fmaak_f16") {
            if (graphics_input)
                return false;
            if (i.dpp_control || permuted)
                refuse(i.pc, "unsupported " + string(n) + " DPP form");
            if (i.subword)
                refuse(i.pc, "unsupported " + string(n) + " SDWA form");
            if (i.scale)
                refuse(i.pc, "unsupported " + string(n) + " omod");
            if (i.wide && (i.a & 0x7800))
                refuse(i.pc, "unsupported " + string(n) + " high-half op_sel");
            if (n == "v_fmac_f16" && i.wide && (((i.abs | i.neg) & 4) || i.src[2]))
                refuse(i.pc, "unsupported v_fmac_f16 third-source modifier");
            if (!half_mode_known || ((mode_bits ^ 0xf0u) & 0x8000ccu))
                refuse(i.pc, "unsupported binary16 rounding or denormal mode");
            auto half_bits = [&](unsigned j) {
                const unsigned r = i.src[j];
                string bits;
                if (r >= 240 && r <= 248) {
                    constexpr unsigned halves[]{0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000,
                                                0xc000, 0x4400, 0xc400, 0x3118};
                    bits = to_string(halves[r - 240]);
                } else if (r == 255)
                    bits = to_string(i.literal & 65535u);
                else if (r >= 128 && r <= 192)
                    bits = to_string((r - 128) & 65535u);
                else if (r >= 193 && r <= 208)
                    bits = to_string((192u - r) & 65535u);
                else
                    bits = ir.binary("and", operand32(i, r), "65535");
                if (i.abs & (1u << j))
                    bits = ir.binary("and", bits, "32767");
                if (i.neg & (1u << j))
                    bits = ir.binary("xor", bits, "32768");
                return bits;
            };
            auto a = half_bits(0);
            auto b = n == "v_fmamk_f16" ? to_string(i.literal & 65535u) : half_bits(1);
            auto c = n == "v_fma_f16"   ? half_bits(2)
                   : n == "v_fmac_f16"  ? ir.binary("and", read(i.dst), "65535")
                   : n == "v_fmamk_f16" ? half_bits(1)
                                        : to_string(i.literal & 65535u);
            auto result = fused_half_rne(a, b, c);
            if (i.clamp) {
                auto nan = cmp("ugt", ir.binary("and", result, "32767"), "31744");
                auto low = ir.value("or i1 " + nan + ", " + cmp("ne", ir.binary("and", result, "32768"), "0"));
                result = select(low, "0", select(cmp("ugt", result, "15359"), "15360", result));
            }
            auto kept = ir.binary("and", read(i.dst), "4294901760");
            vector_write(i.dst, ir.binary("or", kept, ir.binary("and", result, "65535")));
            return true;
        }
        const auto rule = alu_rules.find(n);
        const bool byte_to_float =
            n.size() == 16 && n.substr(0, 15) == "v_cvt_f32_ubyte" && n.back() >= '0' && n.back() <= '3';
        const bool off_i4_to_float = n == "v_cvt_off_f32_i4";
        const bool round = n == "v_fract_f32" || n == "v_trunc_f32" || n == "v_ceil_f32";
        const bool to_integer = n == "v_cvt_i32_f32" || n == "v_cvt_u32_f32" ||
                                n == "v_cvt_flr_i32_f32" || n == "v_cvt_rpi_i32_f32";
        const bool cube = n == "v_cubeid_f32" || n == "v_cubesc_f32" || n == "v_cubetc_f32" || n == "v_cubema_f32";
        const bool mixed = i.enc == Encoding::Mix && (n == "v_fma_mix_f32" || n == "v_fma_mixlo_f16" || n == "v_fma_mixhi_f16");
        const bool unfused_mad = n == "v_mad_f32" || n == "v_mad_legacy_f32" || n == "v_mac_f32" || n == "v_mac_legacy_f32";
        if (!mixed && !to_integer && !byte_to_float && !off_i4_to_float && !round && !cube && !unfused_mad &&
            n != "v_mul_legacy_f32" &&
            n != "v_cos_f32" && (rule == alu_rules.end() || rule->second.type == AluType::Bits))
            return false;
        auto source = [&](unsigned j) {
            auto v = !j && permuted ? *permuted : operand32(i, i.src[j]);
            if (i.subword && j < 2)
                v = subword(v, i.src_select[j], i.src_sext[j]);
            return v;
        };
        auto fp = [&](unsigned j) {
            auto bits = source(j);
            if (i.abs & (1u << j))
                bits = ir.binary("and", bits, "2147483647");
            if (i.neg & (1u << j))
                bits = ir.binary("xor", bits, "2147483648");
            return ir.value("bitcast i32 " + bits + " to float");
        };
        auto unary = [&](unsigned op, const string &v) {
            return ir.invoke("float", "unary.f32", op, {{"float", v}}, "nounwind readnone");
        };
        auto binary = [&](unsigned op, const string &a, const string &b) {
            return ir.invoke("float", "binary.f32", op, {{"float", a}, {"float", b}}, "nounwind readnone");
        };
        string result;
        if (to_integer) {
            if (i.scale || i.clamp)
                refuse(i.pc, "integer conversion output modifier");
            auto a = fp(0);
            if (n == "v_cvt_rpi_i32_f32")
                a = ir.value("fadd float " + a + ", 0.5");
            if (n == "v_cvt_rpi_i32_f32" || n == "v_cvt_flr_i32_f32")
                a = unary(27, a);
            const bool sign = n != "v_cvt_u32_f32";
            // Only convert a representable input. Select the ISA's saturated
            // result (and zero for NaN) independently of LLVM poison behavior.
            auto lower = ir.value("fcmp oge float " + a + ", " + (sign ? "-2147483648.0" : "0.0"));
            auto upper = ir.value("fcmp olt float " + a + ", " + (sign ? "2147483648.0" : "4294967296.0"));
            auto safe = select_float(ir.value("and i1 " + lower + ", " + upper), a, "0.0");
            auto converted = ir.value(string(sign ? "fptosi" : "fptoui") + " float " + safe + " to i32");
            auto bounded = select(upper, converted, sign ? "2147483647" : "4294967295");
            if (sign)
                bounded = select(lower, bounded, "2147483648");
            else
                bounded = select(lower, bounded, "0");
            auto ordered = ir.value("fcmp ord float " + a + ", " + a);
            vector_write(i.dst, place_subword(i, select(ordered, bounded, "0")));
            return true;
        }
        if (mixed) {
            // GFX10 VOP3P splits OP_SEL_HI between the two instruction words.
            const unsigned half = ((i.b >> 27) & 3) | ((i.a >> 12) & 4);
            std::vector<std::pair<string,string>> args;
            for (unsigned j = 0; j < 3; ++j) {
                auto bits = source(j);
                if (half & (1u << j)) {
                    bits = subword(bits, 4 + ((i.a >> (11 + j)) & 1), false);
                    const auto value = ir.invoke("float", "legacyF16ToF32", 131, {{"i32", bits}}, "nounwind readnone");
                    bits = ir.value("bitcast float " + value + " to i32");
                }
                if (i.abs & (1u << j)) bits = ir.binary("and", bits, "2147483647");
                if (i.neg & (1u << j)) bits = ir.binary("xor", bits, "2147483648");
                args.emplace_back("float", ir.value("bitcast i32 " + bits + " to float"));
            }
            result = ir.invoke("float", "tertiary.f32", 46, args, "nounwind readnone");
            if (i.clamp) result = unary(7, result);
            if (n == "v_fma_mix_f32") {
                vector_write(i.dst, ir.value("bitcast float " + result + " to i32"));
            } else {
                auto h = ir.invoke("i32", "legacyF32ToF16", 130, {{"float", result}}, "nounwind readnone");
                const bool high = n == "v_fma_mixhi_f16";
                auto keep = ir.binary("and", read(i.dst), high ? "65535" : "4294901760");
                vector_write(i.dst, ir.binary("or", keep, high ? ir.binary("shl", h, "16") : ir.binary("and", h, "65535")));
            }
            return true;
        }
        if (byte_to_float || off_i4_to_float || (rule != alu_rules.end() && rule->second.type == AluType::ToFloat)) {
            if (i.abs || i.neg)
                refuse(i.pc, "integer conversion source modifier");
            if (off_i4_to_float && i.wide && (i.src[1] != 0 || i.src[2] != 0))
                refuse(i.pc, "v_cvt_off_f32_i4 with unused src1/src2 field set");
            if (off_i4_to_float && i.wide && (i.a & 0x7800))
                refuse(i.pc, "v_cvt_off_f32_i4 with op_sel");
            auto a = source(0);
            if (byte_to_float) {
                unsigned shift_bits = 8u * (n.back() - '0');
                if (shift_bits)
                    a = ir.binary("lshr", a, to_string(shift_bits));
                a = ir.binary("and", a, "255");
            }
            if (off_i4_to_float) {
                auto nibble = ir.binary("ashr", ir.binary("shl", a, "28"), "28");
                auto f = ir.value("sitofp i32 " + nibble + " to float");
                result = ir.value("fmul float " + f + ", 0.0625");
            } else {
                result = ir.value(string(n == "v_cvt_f32_i32" ? "sitofp" : "uitofp") + " i32 " + a + " to float");
            }
        } else {
            auto a = fp(0);
            if (n == "v_fma_f32" || n == "v_fmac_f32" || n == "v_fmamk_f32" || n == "v_fmaak_f32") {
                // Victory emit/lower.rs uses DXIL FMad for gfx10 f32 FMA.
                auto as_float = [&](const string &bits) { return ir.value("bitcast i32 " + bits + " to float"); };
                auto b = n == "v_fmamk_f32" ? as_float(to_string(i.literal)) : fp(1);
                auto c = n == "v_fmac_f32" ? as_float(read(i.dst)) : n == "v_fmaak_f32" ? as_float(to_string(i.literal)) :
                         n == "v_fmamk_f32" ? fp(1) : fp(2);
                result = ir.invoke("float", "tertiary.f32", 46, {{"float", a}, {"float", b}, {"float", c}}, "nounwind readnone");
            } else if (unfused_mad) {
                // gfx2hlsl.py mad32 and pixel_aot (#3005): flush denormal inputs to signed zero, round the product on
                // its own (the legacy form makes 0 * anything 0), add the flushed addend and flush the sum. Plain
                // fmul and fadd carry no fast-math flags, so DXIL does not allow them to be contracted into FMad.
                auto ftz = [&](const string &x) {
                    auto bits = ir.value("bitcast float " + x + " to i32");
                    auto tiny = cmp("ult", ir.binary("and", bits, "2147483647"), "8388608");
                    auto kept = ir.binary("and", bits, select(tiny, "2147483648", "4294967295"));
                    return ir.value("bitcast i32 " + kept + " to float");
                };
                const bool legacy = n == "v_mad_legacy_f32" || n == "v_mac_legacy_f32";
                if (n.substr(0, 5) == "v_mac" && ((i.abs | i.neg) & 4))
                    refuse(i.pc, "unsupported " + string(n) + " accumulator modifier");
                auto x = ftz(a), y = ftz(fp(1));
                auto c = n.substr(0, 5) == "v_mac" ? ir.value("bitcast i32 " + read(i.dst) + " to float") : fp(2);
                auto product = ir.value("fmul float " + x + ", " + y);
                if (legacy) {
                    auto zero = ir.value("or i1 " + ir.value("fcmp oeq float " + x + ", 0.0") + ", " +
                                         ir.value("fcmp oeq float " + y + ", 0.0"));
                    product = select_float(zero, "0.0", product);
                }
                result = ftz(ir.value("fadd float " + ftz(product) + ", " + ftz(c)));
            } else if (n == "v_mul_legacy_f32") {
                auto b = fp(1);
                auto zero = ir.value("or i1 " + ir.value("fcmp oeq float " + a + ", 0.0") + ", " +
                                     ir.value("fcmp oeq float " + b + ", 0.0"));
                result = select_float(zero, "0.0", ir.value("fmul float " + a + ", " + b));
            } else if (n == "v_sin_f32" || n == "v_cos_f32") {
                // Victory trig_turns: reduce in turns before converting to
                // radians, preserving signed zero and exact quarter-turn zeros.
                auto part = ir.value("fsub float " + a + ", " + unary(26, a));
                part = select_float(ir.value("fcmp oeq float " + a + ", 0.0"), a, part);
                const bool cosine = n == "v_cos_f32";
                auto radians = ir.value("fmul float " + part + ", 0x401921FB60000000");
                auto exact = ir.value("fcmp oeq float " + unary(6, part) + (cosine ? ", 0.25" : ", 0.5"));
                result = select_float(exact, "0.0", unary(cosine ? 12 : 13, radians));
            } else if (cube)
                result = cube_alu(n, a, fp(1), fp(2));
            else if (n == "v_add_f32" || n == "v_sub_f32" || n == "v_subrev_f32" || n == "v_mul_f32") {
                auto b = fp(1);
                if (n == "v_subrev_f32")
                    std::swap(a, b);
                result = ir.value(string(n == "v_add_f32"   ? "fadd"
                                         : n == "v_mul_f32" ? "fmul"
                                                            : "fsub") +
                                  " float " + a + ", " + b);
            } else if (n == "v_med3_f32") {
                auto b = fp(1), c = fp(2);
                auto lo = binary(36, a, b), hi = binary(35, a, b);
                auto median = binary(35, lo, binary(36, hi, c));
                auto nan = ir.value("or i1 " + ir.value("fcmp uno float " + a + ", " + b) + ", " +
                                    ir.value("fcmp uno float " + c + ", " + c));
                result = select_float(nan, binary(36, lo, c), median);
            } else if (n == "v_min_f32" || n == "v_max_f32" || n == "v_min3_f32" || n == "v_max3_f32") {
                unsigned op = n.substr(2, 3) == "min" ? 36 : 35;
                result = binary(op, a, fp(1));
                if (rule->second.arity == 3)
                    result = binary(op, result, fp(2));
            } else if (n == "v_exp_f32")
                result = unary(21, a);
            else if (n == "v_log_f32")
                result = unary(23, a);
            else if (n == "v_rndne_f32")
                result = unary(26, a);
            else if (n == "v_floor_f32")
                result = unary(27, a);
            else if (round)
                result = unary(n == "v_fract_f32" ? 22 : n == "v_ceil_f32" ? 28 : 29, a);
            else if (n == "v_sqrt_f32")
                result = unary(24, a);
            else if (n == "v_rsq_f32")
                result = unary(25, a);
            else if (n == "v_rcp_f32")
                result = ir.value("fdiv float 1.000000e+00, " + a);
            else
                return false;
        }
        if (i.scale)
            result = ir.value("fmul float " + result + ", " + (i.scale == 1 ? "2.0" : i.scale == 2 ? "4.0" : "0.5"));
        if (i.clamp)
            result = unary(7, result);
        vector_write(i.dst, place_subword(i, ir.value("bitcast float " + result + " to i32")));
        return true;
    }
    string graphics_lane_mask(const Inst &i, const State &state, unsigned src, const char *failure) {
        if (src == 106) return read(106);
        if (src < 106) {
            const auto &mask = state[src];
            if ((mask.type == Symbol::Type::lane_mask || mask.type == Symbol::Type::uniform_mask) && !mask.offset &&
                (!analysis.program.graphics_wave64 || (!(src & 1) &&
                    state[src+1] == Symbol{mask.type,mask.value,1})))
                return read(1024+mask.value);
        }
        refuse(i.pc,failure);
    }
    string comparison(const Inst &i) {
        if (graphics_input && i.wide && (i.a & 0x7800))
            refuse(i.pc, "unsupported graphics compare operand selection");
        if ((i.abs || i.neg) && i.name.find("_f16") == Name::npos && i.name.find("_f32") == Name::npos)
            refuse(i.pc, "integer comparison with floating modifier");
        auto input = [&](unsigned j) {
            auto value = operand32(i,i.src[j]);
            return i.subword ? subword(value,i.src_select[j],i.src_sext[j]) : value;
        };
        auto a = input(0), b = input(1);
        return comparison(i,a,b);
    }
    string integer_comparison64(const Inst &i, Name op, bool sign) {
        if (i.subword || i.abs || i.neg || i.clamp || i.scale || (i.wide && (i.a & 0x7800)))
            refuse(i.pc, "unsupported 64-bit integer comparison modifiers");
        string predicate(op);
        if (op == "lt" || op == "le" || op == "gt" || op == "ge")
            predicate.insert(0, 1, sign ? 's' : 'u');
        else if (op != "eq" && op != "ne")
            refuse(i.pc, "direct comparison width");

        auto extend = [&](const char *how, unsigned src) {
            return ir.value(string(how) + " i32 " + operand32(i, src) + " to i64");
        };
        auto input = [&](unsigned src) {
            if ((src <= 104 && !(src & 1)) || (src >= 256 && src < 511)) {
                auto low = extend("zext", src), high = extend("zext", src + 1);
                auto upper = ir.value("shl i64 " + high + ", 32");
                return ir.value("or i64 " + upper + ", " + low);
            }
            // A 64-bit integer operand sign-extends an inline constant for both i64 and u64.
            if (src >= 128 && src <= 208)
                return extend("sext", src);
            refuse(i.pc, "direct 64-bit integer comparison operand");
        };
        auto lhs = input(i.src[0]), rhs = input(i.src[1]);
        ir.features |= 1048576;
        return ir.value("icmp " + predicate + " i64 " + lhs + ", " + rhs);
    }
    string comparison(const Inst &i, const string &lhs, const string &rhs) {
        Name n = i.name;
        const auto at = n.rfind('_');
        if (at == n.npos)
            refuse(i.pc, "comparison missing type");
        Name type = n.substr(at + 1), stem = n.substr(0, at);
        Name op = stem.substr(stem.rfind('_') + 1);
        bool fp = type == "f16" || type == "f32";
        bool sign = type == "i16" || type == "i32";
        bool half = type == "f16" || type == "i16" || type == "u16";
        if (!fp && !sign && type != "u16" && type != "u32")
            refuse(i.pc, "direct comparison width");
        if (op == "f")
            return "false";
        if (op == "t" || op == "tru")
            return "true";
        string a = lhs, b = rhs;
        auto bits = [&](string v, unsigned j) {
            if (i.abs & (1u << j))
                v = ir.binary("and", v, half ? "32767" : "2147483647");
            if (i.neg & (1u << j))
                v = ir.binary("xor", v, half ? "32768" : "2147483648");
            return v;
        };
        if (op == "class" && fp) {
            if ((i.abs | i.neg) & 2 ||
                (i.subword && (half || i.src_select[0] != 6 || i.src_sext[0])) ||
                (i.wide && (i.a & 0x7800)))
                refuse(i.pc, "unsupported floating class operand selection");
            // The class mask is integer data. Testing raw bits keeps sNaNs,
            // subnormals and negative zero distinct (gfx2hlsl's class16/32).
            auto word = bits(a,0);
            auto neg = cmp("ne", ir.binary("and",word,half ? "32768" : "2147483648"), "0");
            auto exp = ir.binary("and", word, half ? "31744" : "2139095040");
            auto frac = ir.binary("and", word, half ? "1023" : "8388607");
            auto empty = cmp("eq", frac, "0");
            auto nan = select(cmp("ne",ir.binary("and",frac,half ? "512" : "4194304"),"0"), "2", "1");
            auto zero_or_tiny = select(empty, select(neg,"32","64"), select(neg,"16","128"));
            auto normal = select(neg,"8","256");
            auto cls = select(cmp("eq",exp,half ? "31744" : "2139095040"),
                              select(empty,select(neg,"4","512"),nan),
                              select(cmp("eq",exp,"0"),zero_or_tiny,normal));
            return cmp("ne",ir.binary("and",cls,b),"0");
        }
        if (fp) {
            auto convert = [&](const string &v, unsigned j) {
                auto value = bits(v, j);
                return half ? ir.invoke("float", "legacyF16ToF32", 131, {{"i32", ir.binary("and", value, "65535")}},
                                        "nounwind readnone")
                            : ir.value("bitcast i32 " + value + " to float");
            };
            a = convert(a, 0);
            b = convert(b, 1);
            static const std::map<Name, Name> predicates{{"lt", "olt"},  {"le", "ole"},  {"eq", "oeq"},  {"lg", "one"},
                                                         {"gt", "ogt"},  {"ge", "oge"},  {"o", "ord"},   {"u", "uno"},
                                                         {"nge", "ult"}, {"nlg", "ueq"}, {"ngt", "ule"}, {"nle", "ugt"},
                                                         {"neq", "une"}, {"nlt", "uge"}};
            auto it = predicates.find(op);
            if (it == predicates.end())
                refuse(i.pc, "unknown floating comparison predicate");
            return ir.value("fcmp " + string(it->second) + " float " + a + ", " + b);
        }
        if (half) {
            a = subword(a, 4, sign);
            b = subword(b, 4, sign);
        }
        if (op == "eq")
            return cmp("eq", a, b);
        if (op == "lg" || op == "ne")
            return cmp("ne", a, b);
        if (op != "lt" && op != "le" && op != "gt" && op != "ge")
            refuse(i.pc, "unknown integer comparison predicate");
        return cmp(string(sign ? "s" : "u") + string(op), a, b);
    }
    // Compute SCC liveness once, only for programs that construct buffer
    // descriptors symbolically. A join can observe the old SCC through either
    // predecessor. Closed loops without a read do not make it live.
    bool uses_scc_after(const Inst &definition) {
        const auto &program = analysis.program;
        if (scc_live.empty()) {
            scc_live.resize(program.code.size());
            std::vector<std::vector<size_t>> predecessors(program.code.size());
            std::vector<size_t> pending;
            for (size_t j = 0; j < program.code.size(); ++j) {
                const auto &inst = program.code[j];
                const auto n = inst.name;
                const bool reads_scc = n == "s_cselect_b32" || n == "s_cselect_b64" ||
                    n == "s_cmov_b32" || n == "s_cmov_b64" || n == "s_cmovk_i32" ||
                    std::find(inst.src.begin(), inst.src.end(), 253u) != inst.src.end();
                // These scalar mask operations set SCC from their entire
                // result and do not consume the incoming SCC. An explicit
                // SRC_SCC operand still takes precedence below.
                const bool mask_scc = n == "s_andn2_b64" || n == "s_and_b64" || n == "s_or_b64" ||
                    n == "s_xor_b64" || n == "s_not_b64" || n == "s_wqm_b32" || n == "s_wqm_b64" ||
                    n == "s_quadmask_b32" || n == "s_quadmask_b64" ||
                    n == "s_and_saveexec_b64" || n == "s_or_saveexec_b64" || n == "s_xor_saveexec_b64" ||
                    n == "s_andn2_saveexec_b64" || n == "s_orn2_saveexec_b64" ||
                    n == "s_nand_saveexec_b64" || n == "s_nor_saveexec_b64" || n == "s_xnor_saveexec_b64";
                const bool overwrites = mask_scc || n == "s_abs_i32" ||
                    n == "s_bcnt0_i32_b32" || n == "s_bcnt1_i32_b32" ||
                    n == "s_bcnt0_i32_b64" || n == "s_bcnt1_i32_b64" ||
                    inst.enc == Encoding::Sopc || n == "s_bfe_u32" || n == "s_and_b32" ||
                    n == "s_or_b32" || n == "s_max_u32" || n == "s_add_u32" || n == "s_sub_u32" ||
                    n == "s_add_i32" || n == "s_sub_i32" || n == "s_and_saveexec_b32" ||
                    n == "s_andn2_wrexec_b32" || n == "s_andn2_b32" || n == "s_orn2_b32" ||
                    n == "s_nor_b32" || n == "s_nand_b32" || n == "s_xnor_b32" || n == "s_not_b32" ||
                    n == "s_lshr_b32" || n == "s_lshl_b32" || n == "s_ashr_i32" ||
                    n == "s_lshl_b64" || n == "s_lshr_b64" || n == "s_ashr_i64" || n == "s_xor_b32" ||
                    n == "s_bfe_u64" || n == "s_bfe_i64" ||
                    n == "s_lshl1_add_u32" || n == "s_lshl2_add_u32" || n == "s_lshl3_add_u32" || n == "s_lshl4_add_u32";
                const bool preserves_pair = n == "s_mul_i32" || n == "s_mul_hi_i32" || n == "s_mul_hi_u32" ||
                    n == "s_bfm_b32" || n == "s_bfm_b64" || n == "s_pack_ll_b32_b16" || n == "s_pack_lh_b32_b16" ||
                    n == "s_pack_hh_b32_b16";
                const bool preserves_sop1 = n == "s_mov_b32" || n == "s_mov_b64" ||
                    n == "s_brev_b32" || n == "s_brev_b64" ||
                    n == "s_ff0_i32_b32" || n == "s_ff0_i32_b64" ||
                    n == "s_ff1_i32_b32" || n == "s_ff1_i32_b64" ||
                    n == "s_flbit_i32_b32" || n == "s_flbit_i32_b64" ||
                    n == "s_flbit_i32" || n == "s_flbit_i32_i64" ||
                    n == "s_sext_i32_i8" || n == "s_sext_i32_i16" ||
                    n == "s_bitset0_b32" || n == "s_bitset0_b64" ||
                    n == "s_bitset1_b32" || n == "s_bitset1_b64" || n == "s_bitreplicate_b64_b32";
                const bool uncertain_scalar = (inst.enc == Encoding::Sopk && n != "s_movk_i32") ||
                    (inst.enc == Encoding::Sop2 && !overwrites && !preserves_pair) ||
                    (inst.enc == Encoding::Sop1 && !overwrites && !preserves_sop1);
                if (reads_scc || uncertain_scalar || (inst.branch() && (inst.op == 4 || inst.op == 5))) {
                    scc_live[j] = true;
                    pending.push_back(j);
                    continue;
                }
                if (overwrites || n == "s_endpgm") continue;
                if (inst.branch()) predecessors[program.at.at(inst.target())].push_back(j);
                if (!inst.branch() || inst.op != 2)
                    predecessors[program.at.at(inst.next)].push_back(j);
            }
            while (!pending.empty()) {
                const auto j = pending.back();
                pending.pop_back();
                for (auto prev : predecessors[j]) {
                    if (scc_live[prev]) continue;
                    scc_live[prev] = true;
                    pending.push_back(prev);
                }
            }
        }
        return scc_live[program.at.at(definition.next)];
    }
    string saturated_add(const string &a, const string &b, bool sign) {
        auto sum = ir.binary("add", a, b);
        if (!sign)
            return select(cmp("ult", sum, a), "4294967295", sum);
        auto overflow = ir.binary("and", ir.binary("xor", a, sum), ir.binary("xor", b, sum));
        return select(cmp("slt", overflow, "0"), select(cmp("slt", a, "0"), "2147483648", "2147483647"), sum);
    }
    bool packed_minmax(const Inst &i, const State &state) {
        if (i.enc != Encoding::Mix)
            return false;
        const auto n = i.name;
        const bool floating = n == "v_pk_min_f16" || n == "v_pk_max_f16";
        const bool signed_int = n == "v_pk_min_i16" || n == "v_pk_max_i16";
        if (!floating && !signed_int && n != "v_pk_min_u16" && n != "v_pk_max_u16")
            return false;
        // VOP3P NEG_HI/LO occupy the usual ABS/NEG fields. The third
        // OP_SEL_HI bit is unused here, like the packed add and shift forms.
        if (i.clamp || i.abs || i.neg || i.src[2] || (i.a & 0x2000))
            refuse(i.pc, "unsupported packed 16-bit min/max encoding");
        if (floating && (!half_mode_known || (mode_bits & 0xc0u) != 0xc0u))
            refuse(i.pc, "unsupported packed binary16 min/max denormal mode");
        auto source = [&](unsigned src) {
            if (src <= 105) {
                if (state[src].type == Symbol::Type::masked || state[src].type == Symbol::Type::uniform_mask)
                    refuse(i.pc, "packed 16-bit min/max uses a lane mask");
            } else if (src < 256 && src != 128 && src != 255) {
                refuse(i.pc, "unsupported packed 16-bit min/max source");
            }
            return operand32(i, src);
        };
        const auto a = source(i.src[0]), b = source(i.src[1]);
        const bool maximum = n.substr(5, 3) == "max";
        auto half = [&](const string &word, bool high) {
            return high ? ir.binary("lshr", word, "16") : ir.binary("and", word, "65535");
        };
        auto order_key = [&](const string &bits) -> string {
            if (floating) {
                // Binary16 encodings are monotone within each sign. Reverse
                // the negative half of that order and keep -0 below +0.
                auto negative = cmp("uge", bits, "32768");
                return ir.binary("xor", bits, select(negative, "65535", "32768"));
            }
            return signed_int ? ir.binary("xor", bits, "32768") : bits;
        };
        auto lane = [&](unsigned selectors) {
            auto x = half(a, selectors & 1), y = half(b, selectors & 2);
            auto result = select(cmp(maximum ? "ugt" : "ult", order_key(x), order_key(y)), x, y);
            if (floating) {
                // Follow the existing startup contract. One NaN selects
                // the numeric side. Two NaNs use the canonical quiet NaN.
                auto x_nan = cmp("ugt", ir.binary("and", x, "32767"), "31744");
                auto y_nan = cmp("ugt", ir.binary("and", y, "32767"), "31744");
                result = select(x_nan, select(y_nan, "32256", y), select(y_nan, x, result));
            }
            return result;
        };
        auto low = lane((i.a >> 11) & 3), high = lane((i.b >> 27) & 3);
        vector_write(i.dst, ir.binary("or", low, ir.binary("shl", high, "16")));
        return true;
    }
    void arithmetic(const Inst &i, const State &state, const string *permuted = nullptr, bool compute = false) {
        auto n = i.name;
        if (compute && (n == "v_dot4_i32_i8" || n == "v_dot4_u32_u8")) {
            // Integer dot products use both packed words as-is. Require the
            // canonical VOP3P OP_SEL/OP_SEL_HI and zero NEG_LO/NEG_HI.
            if (i.enc != Encoding::Mix || (i.a & 0x7f00) != 0x4000 || (i.b >> 27) != 3)
                refuse(i.pc, "unsupported packed byte dot operand modifiers");
            const bool sign = n == "v_dot4_i32_i8";
            auto a = operand32(i, i.src[0]), b = operand32(i, i.src[1]), acc = operand32(i, i.src[2]);
            // The dot alone fits into i32 even for four unsigned 255*255
            // products. Saturate only the final addition to the accumulator.
            auto dot = ir.invoke("i32", "dot4AddPacked.i32", sign ? 163 : 164,
                                 {{"i32", i.clamp ? "0" : acc}, {"i32", a}, {"i32", b}}, "nounwind readnone");
            vector_write(i.dst, i.clamp ? saturated_add(dot, acc, sign) : dot);
            return;
        }
        const bool signed_add_sub = n == "v_add_nc_i32" || n == "v_sub_nc_i32";
        if (signed_add_sub || (i.clamp && (n == "v_add_nc_u32" || n == "v_sub_nc_u32" || n == "v_subrev_nc_u32"))) {
            // AMD RDNA 2 ISA: the VOP3-only signed forms wrap like the unsigned ones, and clamp saturates the
            // result: signed for v_*_nc_i32, unsigned for v_*_nc_u32. LLVM 15 selects these clamp forms for
            // llvm.sadd.sat, llvm.ssub.sat, llvm.uadd.sat and llvm.usub.sat (#3009).
            if ((signed_add_sub && !i.wide) || i.subword || i.dpp_control || permuted || i.abs || i.neg || i.scale ||
                (i.a & 0x7800))
                refuse(i.pc, "unsupported " + string(n) + " operand modifiers");
            auto a = operand32(i, i.src[0]), b = operand32(i, i.src[1]);
            if (n == "v_subrev_nc_u32") std::swap(a, b);
            const bool add = n.substr(0, 5) == "v_add";
            auto result = ir.binary(add ? "add" : "sub", a, b);
            if (i.clamp) {
                if (signed_add_sub) {
                    // Overflow when the operands' signs make the true result leave the i32 range; it saturates
                    // toward the sign of the first operand.
                    auto overflow = add ? ir.binary("and", ir.binary("xor", a, result), ir.binary("xor", b, result))
                                        : ir.binary("and", ir.binary("xor", a, b), ir.binary("xor", a, result));
                    result = select(cmp("slt", overflow, "0"),
                                    select(cmp("slt", a, "0"), "2147483648", "2147483647"), result);
                } else if (add)
                    result = select(cmp("ult", result, a), "4294967295", result);
                else
                    result = select(cmp("ult", a, b), "0", result);
            }
            vector_write(i.dst, result);
            return;
        }
        if (compute && (n == "v_dot8_i32_i4" || n == "v_dot8_u32_u4")) {
            // Integer dot products use both packed words as-is. Require the
            // canonical VOP3P OP_SEL/OP_SEL_HI and zero NEG_LO/NEG_HI.
            if (i.enc != Encoding::Mix || (i.a & 0x7f00) != 0x4000 || (i.b >> 27) != 3)
                refuse(i.pc, "unsupported packed nibble dot operand modifiers");
            const bool sign = n == "v_dot8_i32_i4";
            auto a = operand32(i, i.src[0]), b = operand32(i, i.src[1]), acc = operand32(i, i.src[2]);
            auto nibble = [&](const string &word, unsigned shift) {
                if (sign) {
                    auto aligned = shift < 28 ? ir.binary("shl", word, to_string(28 - shift)) : word;
                    return ir.binary("ashr", aligned, "28");
                }
                auto shifted = shift ? ir.binary("lshr", word, to_string(shift)) : word;
                return shift < 28 ? ir.binary("and", shifted, "15") : shifted;
            };
            // Eight 4-bit products sum to [-448, 512] (signed) or [0, 1800] (unsigned),
            // which always fits in i32. Saturate only the final addition to the accumulator.
            string dot = ir.binary("mul", nibble(a, 0), nibble(b, 0));
            for (unsigned shift = 4; shift < 32; shift += 4)
                dot = ir.binary("add", dot, ir.binary("mul", nibble(a, shift), nibble(b, shift)));
            vector_write(i.dst, i.clamp ? saturated_add(dot, acc, sign) : ir.binary("add", dot, acc));
            return;
        }
        if (n == "v_mad_u32_u16" || n == "v_mad_i32_i16") {
            // AMD RDNA 2 ISA, VOP3A opcodes 883/885. Only the two
            // multiplicands are halfwords. The addend and result are dwords.
            if (!i.wide || i.abs || i.neg || i.scale || (i.a & 0x6000))
                refuse(i.pc, "unsupported halfword MAD operand modifiers");
            const bool sign = n == "v_mad_i32_i16";
            auto half = [&](unsigned j) {
                auto value = operand32(i, i.src[j]);
                if (i.a & (1u << (11 + j))) return ir.binary(sign ? "ashr" : "lshr", value, "16");
                return sign ? ir.binary("ashr", ir.binary("shl", value, "16"), "16")
                            : ir.binary("and", value, "65535");
            };
            auto product = ir.binary("mul", half(0), half(1));
            auto addend = operand32(i, i.src[2]);
            auto sum = i.clamp ? saturated_add(product, addend, sign) : ir.binary("add", product, addend);
            vector_write(i.dst, sum);
            return;
        }
        if (compute && (n == "v_pk_mul_lo_u16" || n == "v_pk_mad_u16" || n == "v_pk_mad_i16")) {
            const bool mad = n != "v_pk_mul_lo_u16";
            const bool sign = n == "v_pk_mad_i16";
            if (i.enc != Encoding::Mix || (i.a & 0x0700) || (i.b >> 29) ||
                (!mad && (i.clamp || (i.a & 0x2000) || i.src[2])))
                refuse(i.pc, "unsupported packed 16-bit mul/mad operand modifiers");
            auto a = operand32(i, i.src[0]), b = operand32(i, i.src[1]);
            string c = mad ? operand32(i, i.src[2]) : string();
            auto half = [&](const string &word, bool hi, bool sext) {
                if (sext)
                    return hi ? ir.binary("ashr", word, "16")
                              : ir.binary("ashr", ir.binary("shl", word, "16"), "16");
                return hi ? ir.binary("lshr", word, "16") : ir.binary("and", word, "65535");
            };
            auto lane = [&](bool hi_out) {
                const bool sel_a = hi_out ? ((i.b >> 27) & 1) != 0 : ((i.a >> 11) & 1) != 0;
                const bool sel_b = hi_out ? ((i.b >> 28) & 1) != 0 : ((i.a >> 12) & 1) != 0;
                const bool sel_c = hi_out ? ((i.a >> 14) & 1) != 0 : ((i.a >> 13) & 1) != 0;
                const bool sext = i.clamp && sign;
                auto res = ir.binary("mul", half(a, sel_a, sext), half(b, sel_b, sext));
                if (mad) {
                    res = ir.binary("add", res, half(c, sel_c, sext));
                    if (i.clamp) {
                        if (sign) {
                            res = select(cmp("sgt", res, "32767"), "32767", res);
                            res = select(cmp("slt", res, "-32768"), "-32768", res);
                        } else {
                            res = select(cmp("ugt", res, "65535"), "65535", res);
                        }
                    }
                }
                return ir.binary("and", res, "65535");
            };
            auto lo = lane(false), hi = lane(true);
            vector_write(i.dst, ir.binary("or", lo, ir.binary("shl", hi, "16")));
            return;
        }
        if (n == "v_mad_u64_u32" || n == "v_mad_i64_i32") {
            // D = S0 * S1 + S2 modulo 2^64 with 32-bit factors and a 64-bit S2 (#3000, gfx2hlsl.py mad64). The
            // unsigned form writes the carry-out of the 64-bit add as a lane mask; the signed carry-out is not
            // established, so the signed form is lowered only with the discarding null SDST. In VOP3B the ABS field
            // holds SDST, so only NEG, clamp and the output modifier are refused here.
            const bool sign = n == "v_mad_i64_i32";
            const unsigned sdst = (i.a >> 8) & 127, addend = i.src[2];
            if (pixel_stage || graphics_input || i.subword || i.dpp_control || i.neg || i.clamp || i.scale)
                refuse(i.pc, "unsupported " + string(n) + " stage or operand modifiers");
            if (sign ? sdst != 125 : (sdst > 105 && sdst != 106))
                refuse(i.pc, string(n) + " carry-out destination");
            const bool vgpr_pair = addend >= 256 && addend < 511, sgpr_pair = addend <= 104 && !(addend & 1);
            if (!(vgpr_pair || sgpr_pair) || i.dst >= 511)
                refuse(i.pc, string(n) + " addend or destination operand");
            for (unsigned j = 0; j < 2; ++j)  // gfx2hlsl.py refuses an inline float as a factor
                if (i.src[j] >= 240 && i.src[j] <= 248)
                    refuse(i.pc, string(n) + " with an inline float factor");
            const char *extend = sign ? "sext" : "zext";
            auto wide = [&](const char *how, const string &x) { return ir.value(string(how) + " i32 " + x + " to i64"); };
            auto c = ir.value("or i64 " + ir.value("shl i64 " + wide("zext", operand32(i, addend + 1)) + ", 32") + ", " +
                              wide("zext", operand32(i, addend)));
            auto product = ir.value("mul i64 " + wide(extend, operand32(i, i.src[0])) + ", " +
                                    wide(extend, operand32(i, i.src[1])));
            auto sum = ir.value("add i64 " + product + ", " + c);
            auto carry = ir.value("icmp ult i64 " + sum + ", " + c);
            ir.features |= 1048576;
            vector_write(i.dst, ir.value("trunc i64 " + sum + " to i32"));
            vector_write(i.dst + 1, ir.value("trunc i64 " + ir.value("lshr i64 " + sum + ", 32") + " to i32"));
            if (!sign)
                write_ballot(sdst, carry);
            return;
        }
        if (auto dst = carry_destination(i)) {
            if (pixel_stage || (graphics_input && ir.loop_dependencies) || i.dpp_control || i.clamp || i.neg || i.scale)
                refuse(i.pc, "unsupported carry instruction stage or operand modifiers");
            // Victory add_sub_carry and pixel_aot model unsigned carry/borrow
            // as lane bits. Snapshot both operands and carry-in before writing
            // the result VGPR or its possibly aliased scalar mask.
            const bool add = n.substr(0, 5) == "v_add";
            auto a = operand32(i, i.src[0]), b = operand32(i, i.src[1]);
            if (i.subword) {
                a = subword(a, i.src_select[0], i.src_sext[0]);
                b = subword(b, i.src_select[1], i.src_sext[1]);
            }
            if (n.substr(0, 8) == "v_subrev") std::swap(a, b);
            auto result = ir.binary(add ? "add" : "sub", a, b);
            auto carry = add ? cmp("ult", result, a) : cmp("ult", a, b);
            if (n.find("_ci_") != n.npos) {
                unsigned src = i.src[2];
                if (src > 106) refuse(i.pc, "unsupported carry-in mask source");
                string bit;
                if (graphics_input) {
                    bit = cmp("ne",graphics_lane_mask(i,state,src,"unproven vertex carry-in mask"),"0");
                } else {
                    auto mask = operand32(i, src);
                    if (wave64) mask = select(upper_half, pair_high(i, src), mask);
                    bit = cmp("ne", ir.binary("and", mask, lane_bit), "0");
                }
                auto edge = cmp("eq", result, add ? "4294967295" : "0");
                auto extra = ir.value("and i1 " + edge + ", " + bit);
                carry = ir.value("or i1 " + carry + ", " + extra);
                result = ir.binary(add ? "add" : "sub", result, ir.value("zext i1 " + bit + " to i32"));
            }
            vector_write(i.dst, place_subword(i, result));
            if (graphics_input) {
                if (!graphics_predicate.empty()) carry = ir.value("and i1 " + graphics_predicate + ", " + carry);
                auto bit = ir.value("zext i1 " + carry + " to i32");
                // An explicit VOP3B SDST leaves VCC untouched. Its per-lane
                // bit may feed another carry, but neither dword is numeric.
                if (*dst != 106) write(*dst,{});
                if (analysis.program.graphics_wave64) write(*dst+1,{});
                write(*dst == 106 ? 106 : 1024+i.pc,bit);
            } else write_ballot(*dst, carry);
            return;
        }
        if (auto field = byte_descriptor_field(i,state,heap_window)) {
            // Keep derived words symbolic, including the status bit until a
            // later instruction overwrites it. Never let a numeric branch
            // observe a placeholder for this field or its SCC.
            // The byte-address OOB selector ORs in two nonzero bits.
            // Its SCC is known for every source descriptor.
            if (field->type == Symbol::Type::byte_oob)
                write(128, "1");
            else if (n != "s_mul_i32" && uses_scc_after(i))
                refuse(i.pc, "numeric SCC from a symbolic descriptor field");
            if (graphics_input) write(i.dst, {});
            return;
        }
        if (packed_minmax(i, state) || float_alu(i, permuted))
            return;
        if (compute && i.enc == Encoding::Vopc && n.size() >= 10) {
            const auto type = n.substr(n.size() - 4);
            if (type == "_i64" || type == "_u64") {
                const auto stem = n.substr(0, n.size() - 4);
                auto condition = integer_comparison64(i, stem.substr(stem.rfind('_') + 1), type == "_i64");
                write_ballot(i.dst, condition);
                return;
            }
        }
        if (i.enc == Encoding::Vopc && !i.clamp && !i.scale &&
            (n.find("class") == n.npos || n.find("_f16") != n.npos || n.find("_f32") != n.npos)) {
            auto condition = comparison(i);
            write_ballot(i.dst, condition);
            return;
        }
        if (compute && (n == "v_pk_add_i16" || n == "v_pk_sub_i16" ||
                        n == "v_pk_add_u16" || n == "v_pk_sub_u16")) {
            // Two-source VOP3P packed 16-bit add/sub accept op_sel[1:0], op_sel_hi[1:0]
            // and clamp (ignoring LLVM's default op_sel_hi[2] bit 14) and refuse neg and reserved bits.
            if (i.enc != Encoding::Mix || (i.a & 0x2700) || (i.b >> 29) || i.src[2])
                refuse(i.pc, "unsupported packed 16-bit add/sub operand modifiers");
            const bool sign = n == "v_pk_add_i16" || n == "v_pk_sub_i16";
            const bool sub = n == "v_pk_sub_i16" || n == "v_pk_sub_u16";
            auto a = operand32(i, i.src[0]), b = operand32(i, i.src[1]);
            auto half = [&](const string &word, bool hi, bool sext) {
                if (sext)
                    return hi ? ir.binary("ashr", word, "16")
                              : ir.binary("ashr", ir.binary("shl", word, "16"), "16");
                return hi ? ir.binary("lshr", word, "16") : ir.binary("and", word, "65535");
            };
            auto lane = [&](bool hi_out) {
                const bool sel_a = hi_out ? ((i.b >> 27) & 1) != 0 : ((i.a >> 11) & 1) != 0;
                const bool sel_b = hi_out ? ((i.b >> 28) & 1) != 0 : ((i.a >> 12) & 1) != 0;
                auto x = half(a, sel_a, i.clamp && sign);
                auto y = half(b, sel_b, i.clamp && sign);
                auto res = ir.binary(sub ? "sub" : "add", x, y);
                if (i.clamp) {
                    if (sign) {
                        auto low = select(cmp("slt", res, "-32768"), "32768", ir.binary("and", res, "65535"));
                        return select(cmp("sgt", res, "32767"), "32767", low);
                    } else if (sub) {
                        res = select(cmp("ugt", x, y), res, "0");
                    } else {
                        res = select(cmp("ugt", res, "65535"), "65535", res);
                    }
                }
                return ir.binary("and", res, "65535");
            };
            auto lo = lane(false), hi = lane(true);
            vector_write(i.dst, ir.binary("or", lo, ir.binary("shl", hi, "16")));
            return;
        }
        if (compute && (n == "v_dot2_i32_i16" || n == "v_dot2_u32_u16")) {
            // These VOP3P dot forms consume both halfwords as-is.
            if (i.enc != Encoding::Mix || (i.a & 0x7f00) != 0x4000 || (i.b >> 27) != 3)
                refuse(i.pc, "unsupported packed halfword dot operand modifiers");
            const bool sign = n == "v_dot2_i32_i16";
            auto a = operand32(i, i.src[0]), b = operand32(i, i.src[1]), acc = operand32(i, i.src[2]);
            auto low_half = [&](const string &word) {
                return sign ? ir.binary("ashr", ir.binary("shl", word, "16"), "16")
                            : ir.binary("and", word, "65535");
            };
            auto low_product = ir.binary("mul", low_half(a), low_half(b));
            auto high_half = sign ? "ashr" : "lshr";
            auto high_product = ir.binary("mul", ir.binary(high_half, a, "16"), ir.binary(high_half, b, "16"));
            auto pair = ir.binary("add", low_product, high_product);
            auto sum = ir.binary("add", pair, acc);
            if (i.clamp) {
                // Accumulate both carries and the three sign extensions in
                // the upper word. Clamp only after the accumulator is added.
                auto carry = [&](const string &result, const string &left) {
                    return ir.value("zext i1 " + cmp("ult", result, left) + " to i32");
                };
                auto upper = ir.binary("add", carry(pair, low_product), carry(sum, pair));
                if (sign) {
                    for (const auto &word : {low_product, high_product, acc})
                        upper = ir.binary("add", upper, ir.binary("ashr", word, "31"));
                    auto fits = cmp("eq", upper, ir.binary("ashr", sum, "31"));
                    auto bound = select(cmp("slt", upper, "0"), "2147483648", "2147483647");
                    sum = select(fits, sum, bound);
                } else {
                    sum = select(cmp("eq", upper, "0"), sum, "4294967295");
                }
            }
            vector_write(i.dst, sum);
            return;
        }
        if (compute && (n == "v_pk_lshlrev_b16" || n == "v_pk_lshrrev_b16" || n == "v_pk_ashrrev_i16")) {
            // Two-source VOP3P packed 16-bit shifts accept op_sel[1:0] and op_sel_hi[1:0]
            // (ignoring LLVM's default op_sel_hi[2] bit 14) and refuse clamp, neg and reserved bits.
            if (i.enc != Encoding::Mix || i.clamp || (i.a & 0x2700) || (i.b >> 29) || i.src[2])
                refuse(i.pc, "unsupported packed 16-bit shift operand modifiers");
            auto a = operand32(i, i.src[0]), b = operand32(i, i.src[1]);
            auto lane = [&](bool hi_out) {
                const bool sel_a = hi_out ? ((i.b >> 27) & 1) != 0 : ((i.a >> 11) & 1) != 0;
                const bool sel_b = hi_out ? ((i.b >> 28) & 1) != 0 : ((i.a >> 12) & 1) != 0;
                auto amt = ir.binary("and", sel_a ? ir.binary("lshr", a, "16") : a, "15");
                if (n == "v_pk_ashrrev_i16") {
                    auto aligned = sel_b ? b : ir.binary("shl", b, "16");
                    auto shift = ir.binary("add", amt, "16");
                    return ir.binary("and", ir.binary("ashr", aligned, shift), "65535");
                }
                auto val = sel_b ? ir.binary("lshr", b, "16") : ir.binary("and", b, "65535");
                if (n == "v_pk_lshrrev_b16")
                    return ir.binary("lshr", val, amt);
                return ir.binary("and", ir.binary("shl", val, amt), "65535");
            };
            auto lo = lane(false), hi = lane(true);
            vector_write(i.dst, ir.binary("or", lo, ir.binary("shl", hi, "16")));
            return;
        }
        const bool select_mods = n == "v_cndmask_b32" && i.wide && !((i.abs | i.neg) & 4);
        if ((i.subword && i.enc != Encoding::Vop1 && i.enc != Encoding::Vop2) || ((i.abs || i.neg) && !select_mods) || i.clamp || i.scale)
            refuse(i.pc, "direct ALU operand modifiers");
        auto val = [&](unsigned j) {
            auto value = !j && permuted ? *permuted : operand32(i, i.src[j]);
            if (select_mods && (i.abs & (1u << j))) value = ir.binary("and", value, "2147483647");
            if (select_mods && (i.neg & (1u << j))) value = ir.binary("xor", value, "2147483648");
            return i.subword && j < 2 ? subword(value, i.src_select[j], i.src_sext[j]) : value;
        };
        if (n == "s_mov_b32" || n == "s_mov_b64") {
            const bool exec_hi = i.dst == 127 && (wave64 || (compute && n == "s_mov_b32"));
            if (i.dst > 106 && i.dst != 126 && !(compute && i.dst == 107) && !exec_hi &&
                !(compute && i.dst == 124 && n == "s_mov_b32"))
                refuse(i.pc, "direct scalar move destination");
            if (i.src[0] <= 105 && !state[i.src[0]].numeric()) {
                if (i.dst > 105)
                    refuse(i.pc, "descriptor used as mask");
                if (n == "s_mov_b64" && ((i.dst | i.src[0]) & 1))
                    refuse(i.pc, "unaligned scalar pair");
                if (state[i.src[0]].type == Symbol::Type::indexed_descriptor)
                    write(i.dst, read(i.src[0]));
                if (n == "s_mov_b64" && state.at(i.src[0] + 1).type == Symbol::Type::indexed_descriptor)
                    write(i.dst + 1, read(i.src[0] + 1));
                // Transfer tracks symbolic low words; preserve numeric high words of a pair independently.
                if (n == "s_mov_b64" && state.at(i.src[0] + 1).numeric())
                    write(i.dst + 1, operand32(i, i.src[0] + 1));
                return;
            }
            string a = val(0);
            if (n == "s_mov_b64") {
                unsigned src = i.src[0];
                if (src >= 240 && src != 255)
                    refuse(i.pc, "floating inline constant in 64-bit move");
                if ((i.dst & 1) || (src < 128 && (src & 1)))
                    refuse(i.pc, "unaligned scalar pair");
                auto hi = graphics_input && !wave64 && src == 106 ? "0" : pair_high(i, src);
                write(i.dst + 1, hi);
            }
            write(i.dst, a);
            return;
        }
        if (i.enc == Encoding::Sopk && (n.substr(0, 7) == "s_cmpk_" || n == "s_addk_i32" || n == "s_mulk_i32")) {
            auto a = operand32(i, i.dst);
            auto immediate = to_string(n.find("_u32") != n.npos ? Word(uint16_t(i.a)) : Word(int16_t(i.a)));
            if (n.substr(0, 7) == "s_cmpk_") {
                write(128, ir.value("zext i1 " + comparison(i, a, immediate) + " to i32"));
                return;
            }
            auto result = ir.binary(n == "s_mulk_i32" ? "mul" : "add", a, immediate);
            if (n == "s_addk_i32") {
                auto overflow = ir.binary("and", ir.binary("xor", a, result), ir.binary("xor", immediate, result));
                write(128, ir.binary("lshr", overflow, "31"));
            }
            write(i.dst, result);
            return;
        }
        if (n == "s_movk_i32") {
            write(i.dst, to_string(Word(int16_t(i.a))));
            return;
        }
        if (compute && (n == "s_cmov_b32" || n == "s_cmov_b64" || n == "s_cmovk_i32")) {
            // RDNA 2 ISA SOP1/SOPK (#3070): D = SCC ? S0 : D; SCC is unchanged.
            const bool pair = n == "s_cmov_b64";
            if (pair ? ((i.dst & 1) || (i.dst > 104 && i.dst != 106))
                     : (i.dst > 106 && !(!wave64 && i.dst == 107)))
                refuse(i.pc, "invalid " + string(n) + " destination");
            const unsigned s0 = i.src[0];
            if (pair) {
                if (!((s0 <= 104 && !(s0 & 1)) || (s0 >= 128 && s0 <= 208) || s0 == 255))
                    refuse(i.pc, "invalid s_cmov_b64 source");
            } else if (n == "s_cmov_b32") {
                if (s0 > 106 && !(!wave64 && s0 == 107) && !constant32(i, s0))
                    refuse(i.pc, "invalid s_cmov_b32 source");
            }
            auto cond = cmp("ne", read(128), "0");
            auto old_lo = operand32(i, i.dst);
            if (pair) {
                auto old_hi = operand32(i, i.dst + 1);
                auto new_lo = operand32(i, s0);
                auto new_hi = pair_high(i, s0);
                write(i.dst, select(cond, new_lo, old_lo));
                write(i.dst + 1, select(cond, new_hi, old_hi));
            } else {
                auto next = n == "s_cmovk_i32" ? to_string(Word(int16_t(i.a))) : operand32(i, s0);
                write(i.dst, select(cond, next, old_lo));
            }
            return;
        }
        if (compute && (n == "v_readlane_b32" || n == "v_writelane_b32")) {
            if (i.enc != Encoding::Vop3 || i.dpp_control || (i.a & 0x7800u) || i.src[2])
                refuse(i.pc, "lane read/write encoding or operand modifiers");
            const bool reading = n == "v_readlane_b32";
            if ((reading && (i.src[0] < 256 || i.src[0] >= 512 || i.dst > 107)) ||
                (!reading && (i.src[0] >= 256 || i.dst < 256 || i.dst >= 512)))
                refuse(i.pc, "lane read/write register operand");
            const unsigned width = wave64 ? 64 : 32;
            const unsigned selector = i.src[1];
            const auto guest = guest_spill_lane(i, wave64);
            const unsigned vector = reading ? i.src[0] : i.dst;
            if (guest && reading) {
                const auto slot = state.lane_spills.find({vector, *guest});
                if (slot != state.lane_spills.end()) {
                    if (!slot->second)
                        refuse(i.pc, "symbolic VGPR lane read has conflicting incoming values");
                    if (i.dst > 105)
                        refuse(i.pc, "symbolic VGPR lane read needs an ordinary SGPR destination");
                    return; // Transfer associated the destination SGPR with this identity.
                }
            }
            if (!reading && i.src[0] <= 105 && identity_spill(state[i.src[0]])) {
                if (!guest)
                    refuse(i.pc, "symbolic VGPR lane write needs a constant lane");
                return;
            }
            string selected;
            if (selector <= 105 || (!wave64 && selector == 107)) {
                if (state.has_spill(vector))
                    refuse(i.pc, "symbolic VGPR lane read or write needs a constant lane");
                selected = ir.binary("and", val(1), to_string(width - 1));
            } else {
                if (!guest)
                    refuse(i.pc, "lane read/write needs a constant or SGPR lane");
                selected = to_string(*guest);
            }
            if (reading) {
                const auto physical = wave64 ? ir.binary("and", selected, "31") : selected;
                auto value = ir.invoke("i32", "waveReadLaneAt.i32", 117,
                                       {{"i32", read_storage(vector)}, {"i32", physical}}, "nounwind");
                if (wave64) {
                    const auto halves = exchange_halves(value);
                    value = select(cmp("uge", selected, "32"), halves[1], halves[0]);
                }
                write(i.dst, value);
            } else {
                const auto guest_lane = wave64 ? ir.binary("or", lane, select(upper_half, "32", "0")) : lane;
                // The lane selector is the sole write predicate, independent of guest EXEC.
                lane_write = true;
                write(i.dst, select(cmp("eq", guest_lane, selected), val(0), read_storage(i.dst)));
                lane_write = false;
            }
            return;
        }
        if (n == "v_readfirstlane_b32") {
            if (compute && (i.dpp_control || (i.wide && (i.a & 0x7800u))))
                refuse(i.pc, "readfirstlane modifiers");
            // Guest EXEC selects the first lane, even if it is not physical lane zero.
            auto low = read(126);
            auto mask = wave64 ? select(upper_half, read(127), low) : low;
            auto first = ir.invoke("i32", "unaryBits.i32", 32, {{"i32", mask}}, "nounwind readnone");
            first = select(cmp("eq", mask, "0"), "0", first);
            auto value = ir.invoke("i32", "waveReadLaneAt.i32", 117, {{"i32", val(0)}, {"i32", first}}, "nounwind");
            if (wave64) {
                auto candidates = exchange_halves(value);
                value = select(cmp("ne", low, "0"), candidates[0], candidates[1]);
            }
            write(i.dst, value);
            return;
        }
        if (n == "v_lshrrev_b64" || n == "v_lshlrev_b64" || n == "v_ashrrev_i64") {
            // v_ashrrev_i64 (#2991) sign-extends a 32-bit literal, as gfx2hlsl.py's i64_words(signed=True) does.
            const bool arithmetic = n == "v_ashrrev_i64";
            unsigned src = i.src[1];
            const bool sgpr_pair = src <= 104 && (src & 1) == 0;
            const bool vgpr_pair = src >= 256 && src < 511;
            const bool inline_int = src >= 128 && src <= 208;
            if (i.dst >= 511 || !(sgpr_pair || vgpr_pair || inline_int || src == 255))
                refuse(i.pc, "direct 64-bit shift operand");
            string lo = "0", hi = "0";
            if (src != 128) {
                lo = operand32(i, src);
                if (sgpr_pair || vgpr_pair)
                    hi = operand32(i, src + 1);
                else if (inline_int && src >= 193)
                    hi = "4294967295";
                else if (src == 255 && arithmetic)
                    hi = ir.binary("ashr", lo, "31");
                auto low = ir.value("zext i32 " + lo + " to i64"), high = ir.value("zext i32 " + hi + " to i64");
                auto bits = ir.value("shl i64 " + high + ", 32");
                bits = ir.value("or i64 " + bits + ", " + low);
                auto amt = ir.value("zext i32 " + ir.binary("and", val(0), "63") + " to i64");
                bits = ir.value(string(arithmetic ? "ashr" : n == "v_lshrrev_b64" ? "lshr" : "shl") + " i64 " + bits +
                                ", " + amt);
                lo = ir.value("trunc i64 " + bits + " to i32");
                hi = ir.value("trunc i64 " + ir.value("lshr i64 " + bits + ", 32") + " to i32");
                ir.features |= 1048576;
            }
            vector_write(i.dst, lo);
            vector_write(i.dst + 1, hi);
            return;
        }
        if (n == "s_bfe_u32" && i.src[0] <= 105) {
            const auto &descriptor = state[i.src[0]];
            const auto field = source(i, i.src[1], state);
            if (descriptor.type == Symbol::Type::indexed_descriptor && descriptor.value == compute_resource_heap &&
                descriptor.offset % 32 == 4 && field.type == Symbol::Type::constant && field.value == 0xe0010) {
                // Reachable only through a heap descriptor load, which needs the raw_buffer contract. A raw buffer view
                // has no stride to read, so the stride of a GPU-selected slot stays untranslated in the direct path.
                // With the raw_buffer contract a stride read into s0-s105 is tracked symbolically
                // (byte_descriptor_field) and does not reach here.
                refuse_heap_contract(i.pc);
                refuse(i.pc, "a GPU-selected heap slot stride read is not translated with a view contract");
            }
        }
        if (compute && (n == "s_cmp_eq_u64" || n == "s_cmp_lg_u64" || n == "s_bitcmp0_b64" || n == "s_bitcmp1_b64")) {
            // Issue #3078: lower 64-bit SOPC scalar comparisons to pure 32-bit DXIL writes of SCC.
            auto scalar64 = [&](unsigned src) -> std::pair<string, string> {
                if (src == 125)
                    return {"0", "0"};
                if (src == 106 || src == 126)
                    refuse(i.pc, "invalid " + string(n) + " source");
                return {operand32(i, src), pair_high(i, src)};
            };
            auto [a_lo, a_hi] = scalar64(i.src[0]);
            string cond;
            if (n == "s_cmp_eq_u64" || n == "s_cmp_lg_u64") {
                auto [b_lo, b_hi] = scalar64(i.src[1]);
                const bool eq = n == "s_cmp_eq_u64";
                const auto op = eq ? "eq" : "ne";
                const auto combine = eq ? "and" : "or";
                cond = ir.value(string(combine) + " i1 " + cmp(op, a_lo, b_lo) + ", " + cmp(op, a_hi, b_hi));
            } else {
                const unsigned index_src = i.src[1];
                if (!(index_src <= 105 || (index_src >= 128 && index_src <= 208) || index_src == 255))
                    refuse(i.pc, "invalid " + string(n) + " bit index");
                auto idx = operand32(i, index_src);
                auto upper = cmp("ne", ir.binary("and", idx, "32"), "0");
                auto word = select(upper, a_hi, a_lo);
                auto bit = ir.binary("and", shift("lshr", word, idx), "1");
                cond = cmp(n == "s_bitcmp0_b64" ? "eq" : "ne", bit, "0");
            }
            write(128, ir.value("zext i1 " + cond + " to i32"));
            return;
        }
        if (compute && n == "s_bfm_b64") {
            if ((i.dst & 1) || i.dst > 104)
                refuse(i.pc, "invalid s_bfm_b64 destination");
            if ((i.src[0] > 105 && !constant32(i, i.src[0])) ||
                (i.src[1] > 105 && !constant32(i, i.src[1])))
                refuse(i.pc, "invalid s_bfm_b64 source");
            auto width = ir.binary("and", val(0), "63");
            auto offset = ir.binary("and", val(1), "63");
            auto wide_width = ir.value("zext i32 " + width + " to i64");
            auto wide_offset = ir.value("zext i32 " + offset + " to i64");
            // Width and offset are six-bit fields, so both shifts stay below 64 and a raw
            // width of 64 is an empty mask.
            auto mask = ir.value("sub i64 " + ir.value("shl i64 1, " + wide_width) + ", 1");
            auto bits = ir.value("shl i64 " + mask + ", " + wide_offset);
            write(i.dst, ir.value("trunc i64 " + bits + " to i32"));
            write(i.dst + 1, ir.value("trunc i64 " + ir.value("lshr i64 " + bits + ", 32") + " to i32"));
            ir.features |= 1048576;
            return;
        }
        if (compute && (n == "s_cselect_b64" || n == "s_not_b64")) {
            if ((i.dst & 1) || (i.dst > 104 && i.dst != 106))
                refuse(i.pc, "invalid " + string(n) + " destination");
            auto high = [&](unsigned src) {
                if (src == 106 || src == 126)
                    refuse(i.pc, "invalid " + string(n) + " source");
                return pair_high(i, src);
            };
            auto lo = val(0), hi = high(i.src[0]);
            if (n == "s_cselect_b64") {
                auto condition = cmp("ne", read(128), "0");
                lo = select(condition, lo, val(1));
                hi = select(condition, hi, high(i.src[1]));
            } else {
                lo = ir.binary("xor", lo, "4294967295");
                hi = ir.binary("xor", hi, "4294967295");
                write(128, ir.value("zext i1 " + cmp("ne", ir.binary("or", lo, hi), "0") + " to i32"));
            }
            write(i.dst, lo);
            write(i.dst + 1, hi);
            return;
        }
        if (compute && (n == "s_lshl_b64" || n == "s_lshr_b64" || n == "s_ashr_i64")) {
            // gfx2hlsl.py scalar_wide_op (#3062): D = f(S0, S1 & 63); SCC = (D != 0). Read all sources before
            // writing the destination pair so an overlapping destination cannot observe a half-written input.
            if ((i.dst & 1) || (i.dst > 104 && i.dst != 106))
                refuse(i.pc, "invalid " + string(n) + " destination");
            const unsigned s0 = i.src[0], s1 = i.src[1];
            const bool s0_pair = s0 <= 104 && !(s0 & 1);
            const bool s0_int = s0 >= 128 && s0 <= 208;
            const bool s1_ok = s1 <= 105 || (s1 >= 128 && s1 <= 208) || s1 == 255;
            if (!(s0_pair || s0_int || s0 == 255) || !s1_ok)
                refuse(i.pc, "invalid " + string(n) + " source");
            const bool arithmetic = n == "s_ashr_i64";
            auto lo = operand32(i, s0);
            string hi = "0";
            if (s0_pair)
                hi = operand32(i, s0 + 1);
            else if (s0 >= 193 && s0 <= 208)
                hi = "4294967295";
            else if (s0 == 255 && arithmetic && (i.literal & 0x80000000u))
                hi = "4294967295";
            auto amt = ir.value("zext i32 " + ir.binary("and", operand32(i, s1), "63") + " to i64");
            auto wide = ir.value("or i64 " + ir.value("shl i64 " + ir.value("zext i32 " + hi + " to i64") + ", 32") +
                                 ", " + ir.value("zext i32 " + lo + " to i64"));
            auto shifted = ir.value(string(arithmetic ? "ashr" : n == "s_lshr_b64" ? "lshr" : "shl") +
                                    " i64 " + wide + ", " + amt);
            auto out_lo = ir.value("trunc i64 " + shifted + " to i32");
            auto out_hi = ir.value("trunc i64 " + ir.value("lshr i64 " + shifted + ", 32") + " to i32");
            ir.features |= 1048576;
            write(i.dst, out_lo);
            write(i.dst + 1, out_hi);
            write(128, ir.value("zext i1 " + cmp("ne", ir.binary("or", out_lo, out_hi), "0") + " to i32"));
            return;
        }
        if (compute && (n == "s_quadmask_b32" || n == "s_quadmask_b64")) {
            auto quadmask32 = [&](const string &x) {
                auto t = ir.binary("or", x, ir.binary("lshr", x, "2"));
                t = ir.binary("and", ir.binary("or", t, ir.binary("lshr", t, "1")), "286331153");
                t = ir.binary("and", ir.binary("or", t, ir.binary("lshr", t, "3")), "50529027");
                t = ir.binary("and", ir.binary("or", t, ir.binary("lshr", t, "6")), "983055");
                return ir.binary("and", ir.binary("or", t, ir.binary("lshr", t, "12")), "255");
            };
            if (n == "s_quadmask_b32") {
                auto q = quadmask32(operand32(i, i.src[0]));
                write(i.dst, q);
                write(128, ir.value("zext i1 " + cmp("ne", q, "0") + " to i32"));
                return;
            }
            if ((i.dst & 1) || i.dst > 104)
                refuse(i.pc, "invalid s_quadmask_b64 destination");
            const unsigned s0 = i.src[0];
            const bool s0_pair = s0 <= 104 && !(s0 & 1);
            const bool s0_int = s0 >= 128 && s0 <= 208;
            if (!(s0_pair || s0 == 125 || s0_int || s0 == 255))
                refuse(i.pc, "invalid s_quadmask_b64 source");
            auto lo = s0 == 125 ? "0" : operand32(i, s0);
            string hi = "0";
            if (s0_pair)
                hi = operand32(i, s0 + 1);
            else if (s0 >= 193 && s0 <= 208)
                hi = "4294967295";
            auto out_lo = ir.binary("or", quadmask32(lo), ir.binary("shl", quadmask32(hi), "8"));
            write(i.dst, out_lo);
            write(i.dst + 1, "0");
            write(128, ir.value("zext i1 " + cmp("ne", out_lo, "0") + " to i32"));
            return;
        }
        if (compute && (n == "s_brev_b64" || n == "s_bfe_u64" || n == "s_bfe_i64")) {
            if ((i.dst & 1) || i.dst > 104)
                refuse(i.pc, "invalid " + string(n) + " destination");
            const unsigned s0 = i.src[0];
            const bool s0_pair = s0 <= 104 && !(s0 & 1);
            const bool s0_int = s0 >= 128 && s0 <= 208;
            const bool sign = n == "s_bfe_i64";
            if (!(s0_pair || s0 == 125 || s0_int || s0 == 255))
                refuse(i.pc, "invalid " + string(n) + " source");
            if (n != "s_brev_b64") {
                const unsigned s1 = i.src[1];
                if (!(s1 <= 105 || s1 == 125 || (s1 >= 128 && s1 <= 208) || s1 == 255))
                    refuse(i.pc, "invalid " + string(n) + " source");
            }
            // Read all inputs before writing the destination pair so overlapping registers stay intact (#3086).
            auto lo = s0 == 125 ? "0" : operand32(i, s0);
            string hi = "0";
            if (s0_pair)
                hi = operand32(i, s0 + 1);
            else if (s0 >= 193 && s0 <= 208)
                hi = "4294967295";
            else if (s0 == 255 && sign && (i.literal & 0x80000000u))
                hi = "4294967295";
            if (n == "s_brev_b64") {
                auto rev_lo = ir.invoke("i32", "unary.i32", 30, {{"i32", hi}}, "nounwind readnone");
                auto rev_hi = ir.invoke("i32", "unary.i32", 30, {{"i32", lo}}, "nounwind readnone");
                write(i.dst, rev_lo);
                write(i.dst + 1, rev_hi);
                return;
            }
            auto ctrl = i.src[1] == 125 ? "0" : operand32(i, i.src[1]);
            auto offset = ir.binary("and", ctrl, "63");
            auto width = ir.binary("and", ir.binary("lshr", ctrl, "16"), "127");
            auto wide = ir.value("or i64 " + ir.value("shl i64 " + ir.value("zext i32 " + hi + " to i64") + ", 32") +
                                 ", " + ir.value("zext i32 " + lo + " to i64"));
            auto off64 = ir.value("zext i32 " + offset + " to i64");
            auto shifted = ir.value(string(sign ? "ashr" : "lshr") + " i64 " + wide + ", " + off64);
            auto pad = ir.value("zext i32 " + ir.binary("and", ir.binary("sub", "64", width), "63") + " to i64");
            auto narrow = ir.value(string(sign ? "ashr" : "lshr") + " i64 " + ir.value("shl i64 " + shifted + ", " + pad) +
                                   ", " + pad);
            auto non_zero = ir.value("select i1 " + cmp("uge", width, "64") + ", i64 " + shifted + ", i64 " + narrow);
            auto result = ir.value("select i1 " + cmp("eq", width, "0") + ", i64 0, i64 " + non_zero);
            auto out_lo = ir.value("trunc i64 " + result + " to i32");
            auto out_hi = ir.value("trunc i64 " + ir.value("lshr i64 " + result + ", 32") + " to i32");
            ir.features |= 1048576;
            write(i.dst, out_lo);
            write(i.dst + 1, out_hi);
            write(128, ir.value("zext i1 " + cmp("ne", ir.binary("or", out_lo, out_hi), "0") + " to i32"));
            return;
        }
        if (compute && (n == "s_bcnt0_i32_b64" || n == "s_bcnt1_i32_b64" ||
                        n == "s_ff0_i32_b64" || n == "s_ff1_i32_b64" ||
                        n == "s_flbit_i32_b64" || n == "s_flbit_i32_i64")) {
            // RDNA 2 ISA SOP1 / gfx2hlsl.py SCALAR_BITS (#3072): 64-bit source, 32-bit i32 destination.
            // Uses only 32-bit DXIL unaryBits.i32 (31=Countbits, 32=FirstbitLo, 33=FirstbitHi).
            // Only the counts write SCC (D != 0); all four scans preserve SCC and return ~0 when no bit qualifies.
            if (i.dst > 105)
                refuse(i.pc, "invalid " + string(n) + " destination");
            const unsigned s0 = i.src[0];
            const bool s0_pair = s0 <= 104 && !(s0 & 1);
            const bool s0_int = s0 >= 128 && s0 <= 208;
            if (!(s0_pair || s0_int || s0 == 255))
                refuse(i.pc, "invalid " + string(n) + " source");
            const bool signed_flbit = n == "s_flbit_i32_i64";
            auto lo = operand32(i, s0);
            string hi = "0";
            if (s0_pair)
                hi = operand32(i, s0 + 1);
            else if (s0 >= 193 && s0 <= 208)
                hi = "4294967295";
            else if (s0 == 255 && signed_flbit && (i.literal & 0x80000000u))
                hi = "4294967295";
            if (n == "s_bcnt0_i32_b64" || n == "s_bcnt1_i32_b64") {
                if (n == "s_bcnt0_i32_b64") {
                    lo = ir.binary("xor", lo, "4294967295");
                    hi = ir.binary("xor", hi, "4294967295");
                }
                auto clo = ir.invoke("i32", "unaryBits.i32", 31, {{"i32", lo}}, "nounwind readnone");
                auto chi = ir.invoke("i32", "unaryBits.i32", 31, {{"i32", hi}}, "nounwind readnone");
                auto count = ir.binary("add", clo, chi);
                write(i.dst, count);
                write(128, ir.value("zext i1 " + cmp("ne", count, "0") + " to i32"));
            } else if (n == "s_ff0_i32_b64" || n == "s_ff1_i32_b64") {
                if (n == "s_ff0_i32_b64") {
                    lo = ir.binary("xor", lo, "4294967295");
                    hi = ir.binary("xor", hi, "4294967295");
                }
                auto ilo = ir.invoke("i32", "unaryBits.i32", 32, {{"i32", lo}}, "nounwind readnone");
                auto ihi = ir.invoke("i32", "unaryBits.i32", 32, {{"i32", hi}}, "nounwind readnone");
                auto hi_pos = select(cmp("ne", hi, "0"), ir.binary("add", ihi, "32"), "4294967295");
                write(i.dst, select(cmp("ne", lo, "0"), ilo, hi_pos));
            } else {
                if (signed_flbit) {
                    auto neg = cmp("slt", hi, "0");
                    lo = select(neg, ir.binary("xor", lo, "4294967295"), lo);
                    hi = select(neg, ir.binary("xor", hi, "4294967295"), hi);
                }
                auto lzhi = ir.invoke("i32", "unaryBits.i32", 33, {{"i32", hi}}, "nounwind readnone");
                auto lzlo = ir.invoke("i32", "unaryBits.i32", 33, {{"i32", lo}}, "nounwind readnone");
                auto lo_pos = select(cmp("ne", lo, "0"), ir.binary("add", lzlo, "32"), "4294967295");
                write(i.dst, select(cmp("ne", hi, "0"), lzhi, lo_pos));
            }
            return;
        }
        auto a = val(0);
        string b, z;
        const bool scalar = n.substr(0, 2) == "s_";
        if (i.enc != Encoding::Vop1 && i.enc != Encoding::Sop1)
            b = val(1);
        // As in Victory's SALU lowering and pixel_aot's SGPR-pair rules,
        // bitset reads its previous destination and leaves SCC alone.
        if (n == "s_bitset0_b32" || n == "s_bitset1_b32" || n == "s_bitset0_b64" || n == "s_bitset1_b64") {
            const bool pair = n.substr(n.size() - 3) == "b64";
            if (i.dst > 105 || (pair && (i.dst & 1)))
                refuse(i.pc, "bitset requires a general scalar destination or aligned pair");
            auto old = operand32(i, i.dst);
            auto bit = shift("shl", "1", a);
            auto replace = [&](const string &value) {
                return n[8] == '1' ? ir.binary("or", value, bit)
                                   : ir.binary("and", value, ir.binary("xor", bit, "4294967295"));
            };
            if (pair) {
                auto high = operand32(i, i.dst + 1);
                auto upper = cmp("ne", ir.binary("and", a, "32"), "0");
                write(i.dst, select(upper, old, replace(old)));
                write(i.dst + 1, select(upper, replace(high), high));
            } else
                write(i.dst, replace(old));
            return;
        }
        if (n == "s_bitreplicate_b64_b32") {
            if ((i.dst > 104 && (graphics_input || (i.dst != 106 && i.dst != 126))) || (i.dst & 1))
                refuse(i.pc, "bitreplicate requires an aligned general scalar pair or compute VCC/EXEC");
            // pixel_aot expands each 16-bit half into the even positions,
            // then duplicates each of the original bits into its neighbour.
            auto replicate = [&](string half) {
                unsigned shift = 8;
                for (Word mask : {0x00ff00ffu, 0x0f0f0f0fu, 0x33333333u, 0x55555555u}) {
                    half = ir.binary("and", ir.binary("or", half, ir.binary("shl", half, to_string(shift))), to_string(mask));
                    shift >>= 1;
                }
                return ir.binary("or", half, ir.binary("shl", half, "1"));
            };
            write(i.dst, replicate(ir.binary("and", a, "65535")));
            write(i.dst + 1, replicate(ir.binary("lshr", a, "16")));
            return;
        }
        if (n == "s_pack_ll_b32_b16" || n == "s_pack_lh_b32_b16" || n == "s_pack_hh_b32_b16") {
            const auto lo = n == "s_pack_hh_b32_b16" ? ir.binary("lshr", a, "16") : ir.binary("and", a, "65535");
            const auto hi = n == "s_pack_ll_b32_b16" ? ir.binary("shl", b, "16") : ir.binary("and", b, "4294901760");
            write(i.dst, ir.binary("or", lo, hi));
            return;
        }
        if (i.enc == Encoding::Sop2 && i.op >= 0x2e && i.op <= 0x31) {
            unsigned amount = i.op - 0x2d;
            auto shifted = ir.binary("shl", a, to_string(amount));
            auto sum = ir.binary("add", shifted, b);
            auto shifted_out = cmp("ne", ir.binary("lshr", a, to_string(32 - amount)), "0");
            auto carry = ir.value("or i1 " + shifted_out + ", " + cmp("ult", sum, shifted));
            write(i.dst, sum);
            write(128, ir.value("zext i1 " + carry + " to i32"));
            return;
        }
        if ((i.enc == Encoding::Sop1 || i.enc == Encoding::Sop2) && n.size() > 6 && n.substr(n.size() - 4) == "_b64") {
            bool save = n.find("_saveexec_") != Name::npos;
            auto op = n.substr(2, n.find('_', 2) - 2);
            if ((i.dst & 1) || (i.dst > 104 && i.dst != 106 && i.dst != 126) ||
                (save && i.enc != Encoding::Sop1) || (!save && i.enc != Encoding::Sop2))
                refuse(i.pc, "unsupported wave64 scalar operation");
            auto ah = pair_high(i, i.src[0]);
            auto bl = save ? read(126) : b, bh = save ? read(127) : pair_high(i, i.src[1]);
            auto combine = [&](const string &left, const string &right) -> string {
                if (op == "and" || op == "or" || op == "xor") return ir.binary(string(op), left, right);
                if (op == "andn2" || op == "orn2")
                    return ir.binary(op == "andn2" ? "and" : "or", left, ir.binary("xor", right, "4294967295"));
                if (op == "nand" || op == "nor" || op == "xnor")
                    return ir.binary("xor", ir.binary(op == "nand" ? "and" : op == "nor" ? "or" : "xor", left, right), "4294967295");
                refuse(i.pc, "unsupported wave64 scalar mask operation");
            };
            // SAVEEXEC combines its source operand with the old EXEC in ISA operand order.
            auto lo = combine(a, bl), hi = combine(ah, bh);
            write(i.dst, save ? bl : lo);
            write(i.dst + 1, save ? bh : hi);
            if (save) { write(126, lo); write(127, hi); }
            write(128, ir.value("zext i1 " + cmp("ne", ir.binary("or", lo, hi), "0") + " to i32"));
            return;
        }
        if (n == "s_and_saveexec_b32" || n == "s_andn2_wrexec_b32") {
            auto old = read(126);
            auto mask = ir.binary("and", a, n == "s_and_saveexec_b32" ? old : ir.binary("xor", old, "4294967295"));
            write(i.dst, n == "s_and_saveexec_b32" ? old : mask);
            write(126, mask);
            write(128, ir.value("zext i1 " + cmp("ne", mask, "0") + " to i32"));
            return;
        }
        if (n == "s_add_i32" || n == "s_sub_i32") {
            auto z = ir.binary(n == "s_add_i32" ? "add" : "sub", a, b);
            auto left = ir.binary("xor", a, z);
            auto right = ir.binary("xor", n == "s_add_i32" ? b : a, n == "s_add_i32" ? z : b);
            write(128, ir.binary("lshr", ir.binary("and", left, right), "31"));
            write(i.dst, z);
            return;
        }
        if (n == "s_addc_u32" || n == "s_subb_u32") {
            auto carry = read(128);
            bool add = n == "s_addc_u32";
            auto partial = ir.binary(add ? "add" : "sub", a, b),
                 result = ir.binary(add ? "add" : "sub", partial, carry);
            auto flag = add ? cmp("ult", partial, a) : cmp("ult", a, b);
            auto carry2 = add ? cmp("ult", result, partial) : cmp("ult", partial, carry);
            write(i.dst, result);
            write(128, ir.value("zext i1 " + ir.value("or i1 " + flag + ", " + carry2) + " to i32"));
            return;
        }
        if (n == "s_brev_b32" || n == "v_bfrev_b32" || n == "v_bcnt_u32_b32") {
            bool count = n == "v_bcnt_u32_b32";
            auto value = ir.invoke("i32", count ? "unaryBits.i32" : "unary.i32", count ? 31 : 30, {{"i32", a}},
                                   "nounwind readnone");
            if (scalar)
                write(i.dst, value);
            else
                vector_write(i.dst, place_subword(i, count ? ir.binary("add", value, b) : value));
            return;
        }
        if (n == "s_bcnt0_i32_b32" || n == "s_bcnt1_i32_b32") {
            auto bits = n == "s_bcnt0_i32_b32" ? ir.binary("xor", a, "4294967295") : a;
            auto count = ir.invoke("i32", "unaryBits.i32", 31, {{"i32", bits}}, "nounwind readnone");
            write(i.dst, count);
            write(128, ir.value("zext i1 " + cmp("ne", count, "0") + " to i32"));
            return;
        }
        if (n == "s_ff0_i32_b32" || n == "s_ff1_i32_b32" || n == "v_ffbl_b32" || n == "s_flbit_i32_b32" ||
            n == "v_ffbh_u32" || n == "s_flbit_i32" || n == "v_ffbh_i32") {
            // DXIL FirstbitLo (32) counts from bit 0; FirstbitHi (33) and FirstbitSHi (34) count from bit 31,
            // as these gfx10 scans do. All three return ~0 when no bit qualifies, and none of the scans writes SCC.
            bool low = n == "s_ff0_i32_b32" || n == "s_ff1_i32_b32" || n == "v_ffbl_b32";
            bool sign = n == "s_flbit_i32" || n == "v_ffbh_i32";
            auto bits = n == "s_ff0_i32_b32" ? ir.binary("xor", a, "4294967295") : a;
            auto index = ir.invoke("i32", "unaryBits.i32", low ? 32 : sign ? 34 : 33, {{"i32", bits}},
                                   "nounwind readnone");
            if (scalar)
                write(i.dst, index);
            else
                vector_write(i.dst, place_subword(i, index));
            return;
        }
        if (n == "v_swap_b32") {
            if (i.wide || i.src[0] < 256)
                refuse(i.pc, "invalid vector swap source");
            auto old = read(i.dst);
            vector_write(i.dst, a);
            vector_write(i.src[0], old);
            return;
        }
        if (n == "s_cselect_b32") {
            write(i.dst, select(cmp("ne", read(128), "0"), a, b));
            return;
        }
        if (n == "s_bitcmp1_b32" || n == "s_bitcmp0_b32") {
            auto bit = ir.binary("and", shift("lshr", a, b), "1");
            write(128, n == "s_bitcmp0_b32" ? ir.binary("xor", bit, "1") : bit);
            return;
        }
        if (n == "s_min_u32" || n == "s_max_u32" || n == "s_min_i32" || n == "s_max_i32" ||
            n == "v_min_u32" || n == "v_max_u32" || n == "v_min_i32" || n == "v_max_i32") {
            const string sign = n.substr(n.size() - 3) == "i32" ? "s" : "u";
            auto pred = cmp(sign + (n.substr(2, 3) == "min" ? "lt" : "gt"), a, b);
            auto result = select(pred, a, b);
            if (scalar) {
                write(i.dst, result);
                write(128, ir.value("zext i1 " + pred + " to i32"));
            } else {
                vector_write(i.dst, place_subword(i, result));
            }
            return;
        }
        if (n == "v_cndmask_b32") {
            if (!wave64 && i.src[2] == 107)
                refuse(i.pc, "unsupported wave32 conditional mask source");
            z = select(cmp("ne", ir.binary("and", wave64 ? select(upper_half, pair_high(i, i.src[2]), val(2)) : val(2), lane_bit), "0"), b, a);
        } else if (n == "v_mov_b32")
            z = a;
        else if (n == "s_absdiff_i32") {
            // D = S0 - S1 wrapped to 32 bits, negated when negative, so 0x80000000 stays; SCC = D != 0 below.
            auto difference = ir.binary("sub", a, b);
            z = select(cmp("slt", difference, "0"), ir.binary("sub", "0", difference), difference);
        } else if (n == "s_mul_i32" || n == "v_mul_lo_u32" || n == "v_mul_lo_i32")
            z = ir.binary("mul", a, b);
        else if (n == "v_mul_hi_u32" || n == "v_mul_hi_i32" || n == "s_mul_hi_u32" || n == "s_mul_hi_i32") {
            // Accumulate 16-bit partial products without requiring Int64ShaderOps.
            auto lo = [&](const string &v) { return ir.binary("and", v, "65535"); };
            auto hi = [&](const string &v) { return ir.binary("lshr", v, "16"); };
            auto al = lo(a), ah = hi(a), bl = lo(b), bh = hi(b);
            auto middle = ir.binary("add", ir.binary("mul", ah, bl), hi(ir.binary("mul", al, bl)));
            auto carry = hi(ir.binary("add", lo(middle), ir.binary("mul", al, bh)));
            z = ir.binary("add", ir.binary("add", ir.binary("mul", ah, bh), hi(middle)), carry);
            if (n == "v_mul_hi_i32" || n == "s_mul_hi_i32") {
                z = ir.binary("sub", z, select(cmp("slt", a, "0"), b, "0"));
                z = ir.binary("sub", z, select(cmp("slt", b, "0"), a, "0"));
            }
        } else if (n == "s_abs_i32")
            z = select(cmp("slt", a, "0"), ir.binary("sub", "0", a), a);
        else if (n == "s_sext_i32_i8" || n == "s_sext_i32_i16") {
            const char *bits = n == "s_sext_i32_i8" ? "24" : "16";
            write(i.dst, ir.binary("ashr", ir.binary("shl", a, bits), bits));
            return;
        } else if (n == "v_add_nc_u32" || n == "s_add_u32")
            z = ir.binary("add", a, b);
        else if (n == "s_lshl1_add_u32" || n == "s_lshl2_add_u32" ||
                 n == "s_lshl3_add_u32" || n == "s_lshl4_add_u32") {
            const unsigned k = static_cast<unsigned>(n[6] - '0');
            auto shifted = ir.binary("shl", a, to_string(k));
            z = ir.binary("add", shifted, b);
            auto shift_overflow = cmp("ne", ir.binary("lshr", a, to_string(32 - k)), "0");
            auto add_carry = cmp("ult", z, shifted);
            auto flag = ir.value("or i1 " + shift_overflow + ", " + add_carry);
            write(i.dst, z);
            write(128, ir.value("zext i1 " + flag + " to i32"));
            return;
        }
        else if (n == "s_sub_u32" || n == "v_sub_nc_u32")
            z = ir.binary("sub", a, b);
        else if (n == "v_subrev_nc_u32")
            z = ir.binary("sub", b, a);
        else if (n == "v_not_b32")
            z = ir.binary("xor", a, "4294967295");
        else if (n == "s_and_b32" || n == "v_and_b32")
            z = ir.binary("and", a, b);
        else if (n == "s_or_b32" || n == "v_or_b32")
            z = ir.binary("or", a, b);
        else if (n == "v_xor_b32" || n == "s_xor_b32")
            z = ir.binary("xor", a, b);
        else if (n == "v_xnor_b32")
            z = ir.binary("xor", ir.binary("xor", a, b), "4294967295");
        else if (n == "v_and_or_b32")
            z = ir.binary("or", ir.binary("and", a, b), val(2));
        else if (n == "v_xad_u32")
            z = ir.binary("add", ir.binary("xor", a, b), val(2));
        else if (n == "v_or3_b32" || n == "v_xor3_b32" || n == "v_add3_u32") {
            const char *op = n == "v_or3_b32" ? "or" : n == "v_xor3_b32" ? "xor" : "add";
            z = ir.binary(op, ir.binary(op, a, b), val(2));
        } else if (n == "v_add_lshl_u32")
            z = shift("shl", ir.binary("add", a, b), val(2));
        else if (n == "v_mul_i32_i24" || n == "v_mad_i32_i24") {
            if (n == "v_mul_i32_i24" && i.wide && i.src[2] != 0)
                refuse(i.pc, "v_mul_i32_i24 with unused src2 field set");
            if (n == "v_mul_i32_i24" && i.wide && (i.a & 0x7800))
                refuse(i.pc, "v_mul_i32_i24 with op_sel");
            // sext(S0[23:0]) * sext(S1[23:0]) (+ S2) modulo 2^32.
            auto sext24 = [&](const string &x) { return ir.binary("ashr", ir.binary("shl", x, "8"), "8"); };
            z = ir.binary("mul", sext24(a), sext24(b));
            if (n == "v_mad_i32_i24")
                z = ir.binary("add", z, val(2));
        } else if (n == "v_mul_hi_i32_i24" || n == "v_mul_hi_u32_u24") {
            if (i.wide && i.src[2] != 0)
                refuse(i.pc, string(n) + " with unused src2 field set");
            if (i.wide && (i.a & 0x7800))
                refuse(i.pc, string(n) + " with op_sel");
            const bool sign = n == "v_mul_hi_i32_i24";
            // Split each 24-bit operand as top * 2^12 + low (low in 0..4095, top signed or unsigned 12-bit).
            // All intermediates stay inside 32 bits; floor division nests exactly for both signs.
            auto top = [&](const string &x) {
                return sign ? ir.binary("ashr", ir.binary("shl", x, "8"), "20")
                            : ir.binary("and", ir.binary("lshr", x, "12"), "4095");
            };
            auto low = [&](const string &x) { return ir.binary("and", x, "4095"); };
            auto ah = top(a), al = low(a), bh = top(b), bl = low(b);
            auto hh = ir.binary("mul", ah, bh);
            auto mid = ir.binary("add", ir.binary("mul", ah, bl), ir.binary("mul", al, bh));
            auto t = ir.binary("add", mid, ir.binary("lshr", ir.binary("mul", al, bl), "12"));
            const char *rshift = sign ? "ashr" : "lshr";
            z = ir.binary(rshift, ir.binary("add", hh, ir.binary(rshift, t, "12")), "8");
        } else if (n == "v_mul_u32_u24" || n == "v_mad_u32_u24") {
            z = ir.binary("mul", ir.binary("and", a, "16777215"), ir.binary("and", b, "16777215"));
            if (n == "v_mad_u32_u24")
                z = ir.binary("add", z, val(2));
        } else if ((n.substr(2, 3) == "min" || n.substr(2, 3) == "max" || n.substr(2, 4) == "med3") &&
                   (n.substr(n.size() - 4) == "_u32" || n.substr(n.size() - 4) == "_i32") && !scalar) {
            const bool signed_order = n.substr(n.size() - 4) == "_i32";
            auto bound = [&](const string &x, const string &y, bool minimum) {
                return select(cmp(signed_order ? (minimum ? "slt" : "sgt") : (minimum ? "ult" : "ugt"), x, y), x, y);
            };
            bool minimum = n.substr(2, 3) == "min";
            if (n.substr(2, 4) == "med3") {
                auto c = val(2);
                z = bound(bound(a, b, true), bound(bound(a, b, false), c, true), false);
            } else {
                z = bound(a, b, minimum);
                if (n[5] == '3')
                    z = bound(z, val(2), minimum);
            }
        } else if (!scalar && (n == "v_mad_u16" || n == "v_mad_i16" ||
                               n == "v_min3_u16" || n == "v_min3_i16" ||
                               n == "v_max3_u16" || n == "v_max3_i16" ||
                               n == "v_med3_u16" || n == "v_med3_i16")) {
            // Three-operand 16-bit integer VOP3 operations (#669): each computes D[15:0] from
            // S0[15:0], S1[15:0] and S2[15:0], and D[31:16] keeps its value (measured on RDNA 2
            // silicon; LLVM GCNSubtarget: "On gfx10, all 16-bit instructions preserve the high bits").
            // op_sel would select other halves and stays refused. Modifiers (clamp, scale, neg, abs)
            // are refused by the path's common rule.
            if (i.wide && (i.a & 0x7800))
                refuse(i.pc, "unsupported " + string(n) + " op_sel");
            auto low = [&](const string &x) { return ir.binary("and", x, "65535"); };
            auto sext = [&](const string &x) { return ir.binary("ashr", ir.binary("shl", x, "16"), "16"); };
            auto c = val(2);
            string r;
            if (n == "v_mad_u16" || n == "v_mad_i16") {
                r = ir.binary("add", ir.binary("mul", low(a), low(b)), low(c));
            } else {
                const bool signed_order = n.substr(n.size() - 4) == "_i16";
                auto bound = [&](const string &x, const string &y, bool minimum) {
                    return select(cmp(signed_order ? (minimum ? "slt" : "sgt") : (minimum ? "ult" : "ugt"), x, y), x, y);
                };
                auto x = signed_order ? sext(a) : low(a);
                auto y = signed_order ? sext(b) : low(b);
                auto z_val = signed_order ? sext(c) : low(c);
                if (n.substr(2, 4) == "med3") {
                    r = bound(bound(x, y, true), bound(bound(x, y, false), z_val, true), false);
                } else {
                    bool minimum = n.substr(2, 3) == "min";
                    r = bound(bound(x, y, minimum), z_val, minimum);
                }
            }
            z = ir.binary("or", ir.binary("and", read(i.dst), "4294901760"), low(r));
        } else if (n == "v_sad_u8" || n == "v_sad_hi_u8" || n == "v_sad_u16" || n == "v_sad_u32" || n == "v_msad_u8") {
            // Sums of unsigned absolute differences plus S2, modulo 2^32 (#2987), as gfx2hlsl.py's sad_u8, sad4_u8,
            // sad_u16 and sad_u32 helpers and tools/pixel_aot.py compute them. v_msad_u8 skips each byte whose S1
            // byte is zero; v_sad_hi_u8 places the byte sum in the high half.
            auto absdiff = [&](const string &x, const string &y) {
                return select(cmp("ult", x, y), ir.binary("sub", y, x), ir.binary("sub", x, y));
            };
            auto field = [&](const string &x, unsigned at, unsigned mask) {
                return ir.binary("and", at ? ir.binary("lshr", x, to_string(at)) : x, to_string(mask));
            };
            string sum;
            if (n == "v_sad_u32")
                sum = absdiff(a, b);
            else {
                const bool half = n == "v_sad_u16";
                for (unsigned at = 0; at < 32; at += half ? 16 : 8) {
                    const unsigned mask = half ? 65535 : 255;
                    auto x = field(a, at, mask), y = field(b, at, mask);
                    auto term = absdiff(x, y);
                    if (n == "v_msad_u8")
                        term = select(cmp("eq", y, "0"), "0", term);
                    sum = sum.empty() ? term : ir.binary("add", sum, term);
                }
                if (n == "v_sad_hi_u8")
                    sum = ir.binary("shl", sum, "16");
            }
            z = ir.binary("add", sum, val(2));
        } else if (n == "v_alignbit_b32" || n == "v_alignbyte_b32") {
            // Bits [k + 31 : k] of the 64-bit value {S0, S1}, with k = S2[4:0] or 8 * S2[1:0] (#2985, the
            // gfx2hlsl.py VOP3 table). A zero k returns S1; the masked shift keeps the other arm in range.
            auto k = n == "v_alignbit_b32" ? ir.binary("and", val(2), "31")
                                           : ir.binary("shl", ir.binary("and", val(2), "3"), "3");
            auto joined = ir.binary("or", shift("lshr", b, k), shift("shl", a, ir.binary("sub", "32", k)));
            z = select(cmp("eq", k, "0"), b, joined);
        } else if (n == "v_perm_b32" || n == "v_lerp_u8") {
            // Byte-wise (RDNA 2 ISA V_PERM_B32 and V_LERP_U8; gfx2hlsl.py #2875 and #2873). v_perm_b32: selector byte
            // S2[8i+7:8i] picks byte 0..7 of {S0, S1} (S1 low), 8..11 the sign of bit 15 or 31 of S1 or S0 as 0x00 or
            // 0xff, 12 gives 0x00 and 13..255 give 0xff. v_lerp_u8: (S0 + S1 + S2[8i]) >> 1 for each byte i.
            auto byte = [&](const string &x, const string &at) { return ir.binary("and", ir.binary("lshr", x, at), "255"); };
            for (unsigned i = 0; i < 4; ++i) {
                const string at = to_string(8 * i);
                string v;
                if (n == "v_lerp_u8") {
                    auto round = ir.binary("and", ir.binary("lshr", val(2), at), "1");
                    v = ir.binary("lshr", ir.binary("add", ir.binary("add", byte(a, at), byte(b, at)), round), "1");
                } else {
                    auto sel = byte(val(2), at);
                    auto picked = byte(select(cmp("ult", sel, "4"), b, a), ir.binary("shl", ir.binary("and", sel, "3"), "3"));
                    auto sign = ir.binary("and", ir.binary("lshr", select(cmp("ult", sel, "10"), b, a),
                                                           ir.binary("or", ir.binary("shl", ir.binary("and", sel, "1"), "4"), "15")), "1");
                    v = select(cmp("ult", sel, "8"), picked,
                               select(cmp("ult", sel, "12"), ir.binary("mul", sign, "255"),
                                      select(cmp("eq", sel, "12"), "0", "255")));
                }
                v = i ? ir.binary("shl", v, at) : v;
                z = i ? ir.binary("or", z, v) : v;
            }
        } else if (n == "v_bfm_b32")
            z = shift("shl", ir.binary("sub", shift("shl", "1", a), "1"), b);  // like s_bfm_b32 (#2985)
        else if (n == "v_bfi_b32")
            z = ir.binary("or", ir.binary("and", a, b), ir.binary("and", ir.binary("xor", a, "4294967295"), val(2)));
        else if (n == "s_bfe_u32" || n == "s_bfe_i32") {
            const bool sign = n == "s_bfe_i32";
            auto offset = ir.binary("and", b, "31");
            auto width = ir.binary("and", ir.binary("lshr", b, "16"), "127");
            auto shifted = ir.binary(sign ? "ashr" : "lshr", a, offset);
            auto pad = ir.binary("and", ir.binary("sub", "32", width), "31");
            auto narrow = ir.binary(sign ? "ashr" : "lshr", ir.binary("shl", shifted, pad), pad);
            z = select(cmp("eq", width, "0"), "0", select(cmp("uge", width, "32"), shifted, narrow));
        }
        else if (n == "v_cvt_pk_u16_u32" || n == "v_cvt_pk_i16_i32") {
            // Saturate S0 and S1 to 16 bits and pack S0 low, S1 high (#2995, gfx2hlsl.py's VOP3 table). The
            // instructions have two sources; like gfx2hlsl.py, refuse a nonzero unused SRC2 field.
            if (i.src[2] != 0)
                refuse(i.pc, string(n) + " with unused src2 field set");
            if (i.a & 0x7800)
                refuse(i.pc, string(n) + " with op_sel");
            auto narrow = [&](const string &x) {
                if (n == "v_cvt_pk_u16_u32")
                    return select(cmp("ugt", x, "65535"), "65535", x);
                auto low = select(cmp("slt", x, "-32768"), "-32768", x);
                return ir.binary("and", select(cmp("sgt", low, "32767"), "32767", low), "65535");
            };
            z = ir.binary("or", narrow(a), ir.binary("shl", narrow(b), "16"));
        }
        else if (n == "v_sat_pk_u8_i16") {
            // RDNA 2 ISA VOP1 0x62 / VOP3A 0x1e2: clamp each signed 16-bit half of S0 to [0, 255],
            // pack into D[7:0] and D[15:8], and zero D[31:16].
            if (i.wide && (i.src[1] != 0 || i.src[2] != 0))
                refuse(i.pc, "v_sat_pk_u8_i16 with unused src1/src2 field set");
            if (i.wide && (i.a & 0x7800))
                refuse(i.pc, "v_sat_pk_u8_i16 with op_sel");
            auto clamp_u8 = [&](const string &s16) {
                auto nonneg = select(cmp("slt", s16, "0"), "0", s16);
                return select(cmp("sgt", nonneg, "255"), "255", nonneg);
            };
            auto lo = clamp_u8(ir.binary("ashr", ir.binary("shl", a, "16"), "16"));
            auto hi = clamp_u8(ir.binary("ashr", a, "16"));
            z = ir.binary("or", lo, ir.binary("shl", hi, "8"));
        }
        else if (n == "v_bfe_u32")
            z = ir.binary("and", shift("lshr", a, b), ir.binary("sub", shift("shl", "1", val(2)), "1"));
        else if (n == "v_bfe_i32") {
            auto width = ir.binary("and", val(2), "31");
            auto pad = ir.binary("sub", "32", width);
            auto field = shift("ashr", a, b);
            z = select(cmp("eq", width, "0"), "0", shift("ashr", shift("shl", field, pad), pad));
        } else if (n == "v_cvt_pk_u8_f32") {
            // Round and saturate the binary32 bits directly. Literal 0.5 through DXIL Round_ne
            // and fptoui read back as 1 on some NVIDIA drivers (#3052).
            constexpr auto half_bits = "1056964608";      // 0x3f000000
            constexpr auto max_byte_bits = "1132396544";  // 0x437f0000 (255.0)
            auto positive = cmp("ult", ir.binary("sub", a, "1"), "2139095040");  // subnormal through +inf
            auto bounded = select(cmp("ugt", a, max_byte_bits), max_byte_bits, a);
            // Values at or below 0.5 yield zero, so clamp them and invalid inputs to
            // exactly 0.5 to keep the significand shift in the range 16..24.
            auto in_range = ir.value("and i1 " + positive + ", " + cmp("uge", bounded, half_bits));
            bounded = select(in_range, bounded, half_bits);
            auto discard = ir.binary("sub", "150", ir.binary("lshr", bounded, "23"));  // 23 + exponent bias
            auto significand = ir.binary("or", ir.binary("and", bounded, "8388607"), "8388608");
            auto truncated = ir.binary("lshr", significand, discard);
            auto halfway = ir.binary("shl", "1", ir.binary("sub", discard, "1"));
            // Bias by halfway - 1, then add the low retained bit so only odd ties round up.
            auto bias = ir.binary("add", ir.binary("sub", halfway, "1"), ir.binary("and", truncated, "1"));
            auto byte = ir.binary("lshr", ir.binary("add", significand, bias), discard);
            auto amount = ir.binary("shl", ir.binary("and", b, "3"), "3");
            auto keep = ir.binary("xor", ir.binary("shl", "255", amount), "4294967295");
            z = ir.binary("or", ir.binary("and", val(2), keep), ir.binary("shl", byte, amount));
        } else if (n == "s_orn2_b32")
            z = ir.binary("or", a, ir.binary("xor", b, "4294967295"));
        else if (n == "s_nand_b32" || n == "s_xnor_b32")
            z = ir.binary("xor", ir.binary(n == "s_nand_b32" ? "and" : "xor", a, b), "4294967295");
        else if (n == "s_ashr_i32")
            z = shift("ashr", a, b);
        else if (n == "s_bfm_b32")
            z = shift("shl", ir.binary("sub", shift("shl", "1", a), "1"), b);
        else if (n == "s_not_b32")
            z = ir.binary("xor", a, "4294967295");
        else if (n == "s_andn2_b32" || n == "s_nor_b32")
            z = n == "s_andn2_b32" ? ir.binary("and", a, ir.binary("xor", b, "4294967295"))
                                   : ir.binary("xor", ir.binary("or", a, b), "4294967295");
        else if (n == "s_lshl_b32" || n == "s_lshr_b32")
            z = shift(n == "s_lshl_b32" ? "shl" : "lshr", a, b);
        else if (n == "v_lshlrev_b32" || n == "v_lshrrev_b32" || n == "v_ashrrev_i32")
            z = shift(n == "v_lshlrev_b32" ? "shl" : n == "v_lshrrev_b32" ? "lshr" : "ashr", b, a);
        else if (n == "v_lshl_add_u32" || n == "v_lshl_or_b32")
            z = ir.binary(n == "v_lshl_add_u32" ? "add" : "or", shift("shl", a, b), val(2));
        else if (n == "v_mbcnt_hi_u32_b32" || n == "v_mbcnt_lo_u32_b32") {
            bool high = n == "v_mbcnt_hi_u32_b32";
            auto before = ir.binary("sub", lane_bit, "1");
            auto mask = wave64 ? select(upper_half, high ? before : "4294967295", high ? "0" : before)
                               : high ? "0" : before;
            auto lower = ir.binary("and", a, mask);
            z = ir.binary("add", b, ir.invoke("i32", "unaryBits.i32", 31, {{"i32", lower}}, "nounwind readnone"));
        } else if (!scalar && (n == "v_add_nc_u16" || n == "v_sub_nc_u16" || n == "v_add_nc_i16" || n == "v_sub_nc_i16" ||
                               n == "v_mul_lo_u16" || n == "v_lshlrev_b16" || n == "v_lshrrev_b16" ||
                               n == "v_ashrrev_i16" || n == "v_min_u16" || n == "v_max_u16" || n == "v_min_i16" ||
                               n == "v_max_i16")) {
            // Non-packed 16-bit integer VOP3 (#669): the result is D[15:0], computed from S0[15:0] and S1[15:0],
            // and D[31:16] keeps its value (measured on RDNA 2 silicon; LLVM: "On gfx10, all 16-bit instructions
            // preserve the high bits"). op_sel would select other halves and stays refused.
            if (i.wide && (i.a & 0x7800))
                refuse(i.pc, "unsupported " + string(n) + " op_sel");
            auto low = [&](const string &x) { return ir.binary("and", x, "65535"); };
            auto sext = [&](const string &x) { return ir.binary("ashr", ir.binary("shl", x, "16"), "16"); };
            const bool sign = n.substr(n.size() - 3) == "i16";
            string r;
            if (n == "v_add_nc_u16" || n == "v_add_nc_i16")
                r = ir.binary("add", a, b);
            else if (n == "v_sub_nc_u16" || n == "v_sub_nc_i16")
                r = ir.binary("sub", a, b);
            else if (n == "v_mul_lo_u16")
                r = ir.binary("mul", a, b);
            else if (n == "v_lshlrev_b16" || n == "v_lshrrev_b16" || n == "v_ashrrev_i16") {
                // D = S1 shifted by S0[3:0].
                auto amount = ir.binary("and", a, "15");
                r = n == "v_lshlrev_b16"   ? ir.binary("shl", b, amount)
                    : n == "v_lshrrev_b16" ? ir.binary("lshr", low(b), amount)
                                           : ir.binary("ashr", sext(b), amount);
            } else {
                auto x = sign ? sext(a) : low(a), y = sign ? sext(b) : low(b);
                const bool minimum = n.substr(2, 3) == "min";
                r = select(cmp(sign ? (minimum ? "slt" : "sgt") : (minimum ? "ult" : "ugt"), x, y), x, y);
            }
            z = ir.binary("or", ir.binary("and", read(i.dst), "4294901760"), low(r));
        } else if (i.enc == Encoding::Sopc || i.enc == Encoding::Vopc || n.substr(0, 6) == "v_cmpx" ||
                   n.substr(0, 6) == "v_cmp_") {
            auto condition = comparison(i, a, b);
            if (scalar)
                write(128, ir.value("zext i1 " + condition + " to i32"));
            else
                write_ballot(i.dst, condition);
            return;
        } else
            refuse(i.pc, "direct DXIL lowering for " + string(n));
        if (scalar) {
            write(i.dst, z);
            if (n != "s_mul_i32" && n != "s_mul_hi_u32" && n != "s_mul_hi_i32" && n != "s_bfm_b32") {
                auto flag = n == "s_add_u32"   ? cmp("ult", z, a)
                            : n == "s_sub_u32" ? cmp("ult", a, b)
                                               : cmp("ne", z, "0");
                write(128, ir.value("zext i1 " + flag + " to i32"));
            }
        } else
            vector_write(i.dst, place_subword(i, z));
    }
    void lower_permlane(const Inst &i, unsigned threads) {
        const bool cross_row = i.name == "v_permlanex16_b32";
        if (threads % (cross_row ? 32 : 16))
            refuse(i.pc, string(i.name) + " needs a thread count divisible by " + (cross_row ? "32" : "16"));

        // VOP3A retains op_sel in word 0 bits 11..14: bit 0 is FI and bit 1 is BOUND_CTRL.
        const unsigned op_sel = (i.a >> 11) & 15;
        if ((op_sel & ~3u) || i.neg || i.abs || i.clamp || i.scale)
            refuse(i.pc, "unsupported permlane modifier or op_sel");
        const bool fi = op_sel & 1;
        const bool bound_zero = op_sel & 2;

        // D and S0 are VGPRs. S1 and S2 are uniform SGPRs or constants.
        if (i.src[0] < 256)
            refuse(i.pc, "permlane S0 must be a VGPR");
        for (unsigned reg : {i.src[1], i.src[2]})
            if (reg > 105 && (wave64 || reg != 107) && !constant32(i, reg))
                refuse(i.pc, "permlane selectors must be uniform SGPRs or constants");

        // S1 packs the nibbles for row lanes 0..7 and S2 packs those for 8..15.
        auto row_lane = ir.binary("and", lane, "15");
        auto selectors = select(cmp("uge", row_lane, "8"), operand32(i, i.src[2]), operand32(i, i.src[1]));
        auto shift = ir.binary("shl", ir.binary("and", row_lane, "7"), "2");
        auto pick = ir.binary("and", ir.binary("lshr", selectors, shift), "15");

        // Permlane16 uses the current row; permlanex16 toggles row bit 4.
        // Both rows of the pair are in the same physical wave32, even for guest wave64.
        auto row_base = ir.binary("and", lane, "4294967280");
        if (cross_row)
            row_base = ir.binary("xor", row_base, "16");
        auto source_lane = ir.binary("or", row_base, pick);
        auto src = operand32(i, i.src[0]);
        auto value = ir.invoke("i32", "waveReadLaneAt.i32", 117, {{"i32",src},{"i32",source_lane}}, "nounwind");

        // Assume DPP's inactive-source rule: FI bypasses the guest EXEC test.
        // Otherwise an inactive source yields zero with BOUND_CTRL or preserves D.
        if (!fi) {
            auto exec = wave64 ? select(upper_half, read(127), read(126)) : read(126);
            auto source_bit = ir.binary("shl", "1", source_lane);
            auto source_usable = cmp("ne", ir.binary("and", exec, source_bit), "0");
            value = select(source_usable, value, bound_zero ? "0" : read(i.dst));
        }
        // Guest EXEC still predicates the destination write.
        vector_write(i.dst, value);
    }
    void lower_dpp(const Inst &i, const State &state, unsigned threads) {
        // All lane selections stay in one physical wave32 for split guest wave64.
        // Shader control flow stays wave-uniform. Guest EXEC predicates VGPR
        // writes, so FI can fetch the last VGPR value from a disabled thread.
        const unsigned mask = i.mirror_mask();
        const auto row = i.dpp_row();
        const bool row_op = row.op != Inst::DppRowOp::None;
        const unsigned group_size = row_op ? 16 : mask ? mask + 1 : 4;
        if (threads % group_size)
            refuse(i.pc, row_op ? "DPP row shift or rotate needs a thread count divisible by 16"
                              : mask ? "DPP mirror needs complete source lane groups"
                              : "DPP quad_perm needs a thread count divisible by four");
        const auto rule = alu_rules.find(i.name);
        if (!binary16_alu(i.name) && (rule == alu_rules.end() || rule->second.arity > 2))
            refuse(i.pc, "unsupported DPP arithmetic");
        if (i.enc == Encoding::Vop1 && ((i.neg | i.abs) & 2))
            refuse(i.pc, "unexpected second DPP source modifier");
        string source_lane, source_usable;
        if (row_op) {
            auto row_base = ir.binary("and", lane, "4294967280");
            auto row_lane = ir.binary("and", lane, "15");
            auto amount = to_string(row.amount);
            const bool shl = row.op == Inst::DppRowOp::Shl;
            auto peer = ir.binary(shl ? "add" : "sub", row_lane, amount);
            if (row.op != Inst::DppRowOp::Ror)
                source_usable = cmp("ult", peer, "16");
            // Keep the physical lane index safe even when the shift is out of row.
            source_lane = ir.binary("or", row_base, ir.binary("and", peer, "15"));
        } else if (mask)
            source_lane = ir.binary("xor", lane, to_string(mask));
        else {
            auto quad_base = ir.binary("and", lane, "4294967292");
            auto quad_lane = ir.binary("and", lane, "3");
            auto shift = ir.binary("shl", quad_lane, "1");
            auto fields = ir.binary("lshr", to_string(*i.dpp_control), shift);
            source_lane = ir.binary("or", quad_base, ir.binary("and", fields, "3"));
        }
        auto src = operand32(i, i.src[0]);
        auto word = ir.invoke("i32", "waveReadLaneAt.i32", 117, {{"i32",src},{"i32",source_lane}}, "nounwind");
        if (!i.dpp_fetch_inactive) {
            auto exec = wave64 ? select(upper_half, read(127), read(126)) : read(126);
            auto source_enabled = cmp("ne", ir.binary("and", exec, ir.binary("shl", "1", source_lane)), "0");
            source_usable = source_usable.empty() ? source_enabled
                          : ir.value("and i1 " + source_usable + ", " + source_enabled);
        }
        string write_enabled;
        if (!source_usable.empty()) {
            if (i.dpp_bound_zero) word = select(source_usable, word, "0");
            else write_enabled = source_usable;
        }
        // Row and bank masks gate destination writes independently of the source and BOUND_CTRL.
        auto apply_mask = [&](unsigned mask, const string &index) {
            auto enabled = cmp("ne", ir.binary("and", to_string(mask), ir.binary("shl", "1", index)), "0");
            write_enabled = write_enabled.empty() ? enabled : ir.value("and i1 " + write_enabled + ", " + enabled);
        };
        if (i.dpp_row_mask != 15) {
            auto row_index = ir.binary("lshr", lane, "4");
            if (wave64) row_index = ir.binary("or", row_index, select(upper_half, "2", "0"));
            apply_mask(i.dpp_row_mask, row_index);
        }
        if (i.dpp_bank_mask != 15)
            apply_mask(i.dpp_bank_mask, ir.binary("and", ir.binary("lshr", lane, "2"), "3"));
        string previous;
        if (!write_enabled.empty()) previous = read(i.dst);
        arithmetic(i, state, &word);
        if (!previous.empty())
            write(i.dst, select(write_enabled, read(i.dst), previous));
    }
    // Returns true when the instruction is a MODE write this path has accepted
    // (including a no-op that leaves the implemented contract in place).
    bool consume_half_mode(const Inst &i) {
        if (!half_binary || graphics_input)
            return false;
        constexpr unsigned f16_bits = 0x8000ccu; // round16 [3:2], denorm16 [7:6], FP16_OVFL
        constexpr unsigned contract = 0xf0u;     // denorm32 = denorm16 = 3 (keep)
        auto bad = [&]() { refuse(i.pc, "unsupported binary16 rounding or denormal mode"); };
        auto accept = [&](unsigned updated) {
            if ((updated ^ contract) & f16_bits)
                bad();
            // A write that moves a bit this path does not model (f32 round or
            // f32 denorm) keeps the previous refusal.
            if ((updated ^ mode_bits) & ~f16_bits)
                return false;
            mode_bits = updated;
            return true;
        };
        const auto &n = i.name;
        if (n == "s_round_mode" || n == "s_denorm_mode") {
            // Both immediates use [1:0] for f32 and [3:2] for f16/f64. Bits above
            // the low nibble are ignored by the instruction.
            const unsigned nibble = i.a & 15u;
            const unsigned mask = n == "s_round_mode" ? 0xfu : 0xf0u;
            const unsigned field = n == "s_round_mode" ? nibble : nibble << 4;
            if (!half_mode_known && ((field ^ mode_bits) & mask & f16_bits))
                bad();
            return accept((mode_bits & ~mask) | field);
        }
        if (n != "s_setreg_b32" && n != "s_setreg_imm32_b32")
            return false;
        // SIMM16 = {size[4:0], offset[4:0], hwRegId[5:0]}. hwRegId 1 is MODE.
        const unsigned simm = i.a & 0xffffu;
        const unsigned hw = simm & 63u;
        const unsigned offset = (simm >> 6) & 31u;
        const unsigned size = ((simm >> 11) & 31u) + 1u;
        if (hw != 1u || offset >= 32u || size > 32u - offset)
            return false;
        const unsigned mask = size == 32u ? 0xffffffffu : ((1u << size) - 1u) << offset;
        if ((mask & f16_bits) == 0)
            return false;
        // The SGPR form, and any form reached from more than one path, does not
        // prove the bits that this lowering implements.
        if (n == "s_setreg_b32" || !half_mode_known)
            bad();
        const unsigned updated = (mode_bits & ~mask) | ((i.literal << offset) & mask);
        return accept(updated);
    }
    NativeComputeShader lower(unsigned entry, unsigned user, unsigned flags, unsigned x, unsigned y, unsigned z) {
        const unsigned lds_words = ((flags >> 15) & 511) * 128;
        const bool uses_lds = std::any_of(analysis.program.code.begin(), analysis.program.code.end(),
                                          [](const Inst &i) { return i.enc == Encoding::Lds; });
        if (uses_lds)
            ir.globals << "@lds = internal addrspace(3) global [" << lds_words << " x i32] undef, align 4\n";
        initialize(user, flags, x, y, z);
        const auto &p = analysis.program;
        mode_bits = 0xf0;
        bool mode_write = false, conditional = false;
        for (size_t k = 0; k < p.code.size(); ++k) {
            if (!p.reached[k])
                continue;
            const auto &op = p.code[k];
        if (binary16_alu(op.name) ||
            op.name == "v_pk_min_f16" ||
            op.name == "v_pk_max_f16" ||
            op.name == "v_cvt_f16_u16" ||
            op.name == "v_cvt_f16_i16" ||
            op.name == "v_cvt_u16_f16" ||
            op.name == "v_cvt_i16_f16" ||
            op.name == "v_fma_f16" ||
            op.name == "v_fmac_f16" ||
            op.name == "v_fmamk_f16" ||
            op.name == "v_fmaak_f16")
                half_binary = true;
            if (op.name == "s_round_mode" || op.name == "s_denorm_mode" || op.name == "s_setreg_b32" ||
                op.name == "s_setreg_imm32_b32")
                mode_write = true;
            if (packed_f16_arithmetic(op.name))
                half_binary = true;
            if (op.branch() && op.op != 2)
                conditional = true;
        }
        half_mode_known = !(half_binary && mode_write && conditional);
        const bool writes_uavs = std::any_of(p.code.begin(), p.code.end(), [](const Inst &i) {
            return i.name.substr(0, 12) == "buffer_store" || i.name.substr(0, 13) == "buffer_atomic" ||
                   (i.enc == Encoding::Image && i.op == 8) || is_image_atomic(i);
        });
        std::set<unsigned> labels{entry};
        ir.body << "  br label %pc" << entry << "\n";
        for (auto &i : p.code)
            if (i.branch()) {
                labels.insert(i.target());
                if (i.op != 2)
                    labels.insert(i.next);
            }
        for (size_t k = 0; k < p.code.size(); ++k) {
            const auto &i = p.code[k];
            compute_pc = i.pc;
            compute_state = &p.before[k];
            if (labels.count(i.pc)) {
                if (k && p.code[k - 1].name != "s_endpgm" && !p.code[k - 1].branch())
                    ir.body << "  br label %pc" << i.pc << "\n";
                ir.body << "pc" << i.pc << ":\n";
            }
            if (!p.reached[k]) {
                ir.body << "  unreachable\n";
                while (k + 1 < p.code.size() && !labels.count(p.code[k + 1].pc))
                    ++k;
                continue;
            }
            if (i.branch()) {
                if (i.op == 2)
                    ir.body << "  br label %pc" << i.target() << "\n";
                else {
                    unsigned r = i.op == 4 || i.op == 5 ? 128 : i.op == 6 || i.op == 7 ? 106 : 126;
                    auto mask = wave64 && r != 128 ? ir.binary("or", read(r), read(r + 1)) : read(r);
                    auto pred = cmp((i.op & 1) ? "ne" : "eq", mask, "0");
                    ir.body << "  br i1 " << pred << ", label %pc" << i.target() << ", label %pc" << i.next << "\n";
                }
            } else if (i.dpp_control)
                lower_dpp(i, p.before[k], x*y*z);
            else if (i.name == "v_permlane16_b32" || i.name == "v_permlanex16_b32")
                lower_permlane(i, x*y*z);
            else if (i.name == "s_endpgm")
                ir.body << "  ret void\n";
            // Hardware scheduling and register-dependency waits have no SGPR
            // operands. SSA dependencies carry these constraints into DXIL.
            else if (i.name == "v_nop" || i.name == "s_inst_prefetch" || i.name == "s_clause" || i.name == "s_nop" ||
                     i.name == "s_waitcnt" || i.name == "s_waitcnt_depctr" || i.name == "s_setprio")
                continue;
            else if (i.enc == Encoding::Mubuf || i.enc == Encoding::Smem)
                memory(i, p.before[k]);
            else if (i.enc == Encoding::Lds)
                local_memory(i, lds_words);
            else if (i.name == "s_barrier")
                barrier(writes_uavs ? 11 : 9); // group rendezvous, LDS and, when used, UAV memory
            else if (i.name == "s_waitcnt_vscnt" || i.name == "s_waitcnt_vmcnt" ||
                     i.name == "s_waitcnt_lgkmcnt" || i.name == "s_waitcnt_expcnt") {
                // null,0 drains this wave's counter. Memory-only barriers
                // preserve ordering without rendezvousing other workgroup waves.
                if (i.dst != 125 || (i.a & 65535))
                    refuse(i.pc, "compute counter wait requires null and a zero threshold");
                if (i.name == "s_waitcnt_lgkmcnt" && uses_lds)
                    barrier(8);
                if ((i.name == "s_waitcnt_vscnt" || i.name == "s_waitcnt_vmcnt") && writes_uavs)
                    barrier(2);
            }
            else if (consume_half_mode(i))
                continue;
            else if (i.enc == Encoding::Sopp)
                refuse(i.pc, "unsupported compute control instruction " + string(i.name));
            else if (i.enc == Encoding::Image)
                image_io(i, p.before[k]);
            else
                arithmetic(i, p.before[k], nullptr, true);
        }
        auto &r = analysis.result;
        r.instruction_count = unsigned(p.code.size());
        const unsigned rank_words = x * y * z > 64 ? x * y * z : 0;
        if ((uses_lds ? lds_words : 0) + rank_words + (uses_exchange ? exchange_words : 0) > 8192)
            refuse(entry, "combined shared storage exceeds 32 KiB");
        r.dxil_ir = ir.finish_compute(x, y, z);
        return std::move(r);
    }
};
