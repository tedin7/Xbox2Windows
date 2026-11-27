#include "gfx1013_compute_native.h"
#include "gfx1013_compute_ir.h"
#include "dxil_module.h"
#include "guest_shader_container.h"
#include "gfx1013_isa.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <iomanip>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace x2w;
using std::string;
using std::to_string;
using Word = uint32_t;
using Name = std::string_view;

static bool binary16_alu(std::string_view name) {
    return name == "v_add_f16" || name == "v_sub_f16" || name == "v_subrev_f16" || name == "v_mul_f16";
}

[[noreturn]] void refuse(unsigned pc, const string &reason) {
    std::ostringstream out;
    out << "native compute +0x" << std::hex << pc << ": " << reason;
    throw UnsupportedComputeShader(out.str());
}
Word word(const string &bytes, unsigned at) {
    if (at > bytes.size() || bytes.size() - at < 4)
        refuse(at, "truncated word");
    Word n;
    std::memcpy(&n, bytes.data() + at, 4);
    return n;
}
string hexword(Word n) {
    char s[12];
    snprintf(s, sizeof(s), "0x%08xu", n);
    return s;
}
enum class Encoding { Sopp, Sop1, Sop2, Sopc, Sopk, Vop1, Vop2, Vopc, QuadCompare, Vop3, Mix, Smem, Image, Mubuf, Lds, Export, Interp, Fetch, FixedSetup, FixedFetch };
#include "gfx1013_compute_opcodes.inl"
struct Inst {
    unsigned pc = 0, next = 0, op = 0, dst = 0;
    Encoding enc{};
    Word a = 0, b = 0, literal = 0;
    std::array<Word, 3> image_address_words{};
    std::array<unsigned, 3> src{};
    Name name;
    unsigned abs = 0, neg = 0, scale = 0;
    bool clamp = false, wide = false, subword = false;
    // 0..255 selects within a quad. Row operations and mirrors cross quads.
    std::optional<unsigned> dpp_control;
    bool dpp_fetch_inactive = false;
    bool dpp_bound_zero = false;
    unsigned mirror_mask() const { return dpp_control == 0x141 ? 7 : dpp_control == 0x140 ? 15 : 0; }
    enum class DppRowOp { None, Shl, Shr, Ror };
    struct DppRow {
        DppRowOp op;
        unsigned amount;
    };
    DppRow dpp_row() const {
        const unsigned control = dpp_control.value_or(0);
        const unsigned amount = control & 15;
        if (amount) {
            switch (control & ~15u) {
            case 0x100: return {DppRowOp::Shl, amount};
            case 0x110: return {DppRowOp::Shr, amount};
            case 0x120: return {DppRowOp::Ror, amount};
            }
        }
        return {DppRowOp::None, 0};
    }
    std::array<unsigned, 2> src_select{6, 6};
    std::array<bool, 2> src_sext{};
    unsigned dst_select = 6, dst_unused = 0;
    bool branch() const {
        return enc == Encoding::Sopp && (op == 2 || op == 4 || op == 5 || op == 6 || op == 7 || op == 8 || op == 9);
    }
    unsigned target() const { return next + 4 * static_cast<int16_t>(a & 65535); }
};
Inst decode(const string &bytes, unsigned pc,bool direct=false, bool graphics=false, bool wave64=false) {
    const auto encoded=gfx1013::instruction(bytes,pc);
    Inst i;
    i.pc = pc;
    i.next = pc;
    auto take = [&]() {
        auto n = word(bytes, i.next);
        i.next += 4;
        return n;
    };
    Word a = i.a = take();
    if (graphics && (encoded.form == gfx1013::Form::exp || encoded.form == gfx1013::Form::vintrp ||
                     a == 0xbe842104)) {
        if (encoded.form == gfx1013::Form::exp) {
            i.enc = Encoding::Export;
            i.b = take();
        } else i.enc = encoded.form == gfx1013::Form::vintrp ? Encoding::Interp : Encoding::Fetch;
        i.name = encoded.name;
        return i;
    }
    auto form = a & 0xff800000u;
    bool lit = false;
    if (form == 0xbf800000) {
        i.enc = Encoding::Sopp;
        i.op = (a >> 16) & 127;
    } else if (form == 0xbe800000) {
        i.enc = Encoding::Sop1;
        i.op = (a >> 8) & 255;
        i.dst = (a >> 16) & 127;
        i.src[0] = a & 255;
        lit = i.src[0] == 255;
    } else if (form == 0xbf000000) {
        i.enc = Encoding::Sopc;
        i.op = (a >> 16) & 127;
        i.src = {a & 255, (a >> 8) & 255, 0};
        lit = i.src[0] == 255 || i.src[1] == 255;
    } else if ((a >> 28) == 11) {
        i.enc = Encoding::Sopk;
        i.op = (a >> 23) & 31;
        i.dst = (a >> 16) & 127;
        // s_setreg_imm32_b32 carries the written value in a literal word.
        lit = i.op == 0x15;
    } else if ((a >> 30) == 2) {
        i.enc = Encoding::Sop2;
        i.op = (a >> 23) & 127;
        i.dst = (a >> 16) & 127;
        i.src = {a & 255, (a >> 8) & 255, 0};
        lit = i.src[0] == 255 || i.src[1] == 255;
    } else if ((a >> 31) == 0) {
        auto k = a >> 25;
        i.dst = 256 + ((a >> 17) & 255);
        i.src = {a & 511, 256 + ((a >> 9) & 255), 106};
        if (k == 63) {
            i.enc = Encoding::Vop1;
            i.op = (a >> 9) & 255;
        } else if (k == 62) {
            i.enc = Encoding::Vopc;
            i.op = (a >> 17) & 255;
            i.dst = 106;
        } else {
            i.enc = Encoding::Vop2;
            i.op = k;
        }
        lit = i.src[0] == 255 ||
              (i.enc == Encoding::Vop2 && (k == 0x20 || k == 0x21 || k == 0x2c || k == 0x2d || k == 0x37 || k == 0x38));
    } else if ((a & 0xfc000000u) == 0xf4000000) {
        i.enc = Encoding::Smem;
        i.b = take();
        i.op = (a >> 18) & 255;
        i.dst = (a >> 6) & 127;
        i.src[0] = (a & 63) * 2;
        const unsigned soffset = i.b >> 25;
        if ((i.b & 0x01f00000u) || (a & 0x14000) || (i.op & ~8u) > 4 ||
            (soffset != 125 && ((!direct && i.op < 8 && i.op != 2) || soffset > 105)))
            refuse(pc, "unsupported scalar memory addressing");
        i.name = "smem";
    } else if ((a & 0xfc000000u) == 0xf0000000) {
        i.enc = Encoding::Image;
        i.b = take();
        i.op = ((a >> 18) & 127) | ((a & 1) << 7);
        i.dst = (i.b >> 8) & 255;
        unsigned extra = (a >> 1) & 3;
        if (!direct && extra > 1)
            refuse(pc, "image NSA count");
        for (unsigned j = 0; j < extra; ++j)
            i.image_address_words[j] = take();
        const bool sample_d16 = direct && (i.op == 32 || i.op == 34 || i.op == 36 || i.op == 37 || i.op == 39) &&
                                ((a >> 3) & 7) != 3;
        if ((!direct && ((a >> 3) & 7) != 1) || ((i.b & 0x80000000u) && !sample_d16 &&
            (i.op != 8 || (!direct && ((a >> 8) & 15) != 1))))
            refuse(pc, "unsupported image dimension or D16");
        i.name = "image";
    } else if ((a & 0xfc000000u) == 0xe0000000u) {
        i.b = take();
        i.op = (a >> 18) & 127;
        i.op |= (a >> 18) & 0x80;
        if (direct && encoded.name.substr(0, 14) == "buffer_atomic_") {
            // Byte offset plus optional VGPR address and GLC old-value return.
            // Indexed atomics currently access the first dword of the chosen element.
            if ((a & 0x00038000u) || (i.b & 0x00e00000u))
                refuse(pc, "unsupported buffer atomic addressing or cache policy");
            if ((a & 0x2000u) && (graphics || (a & 0x1fffu) ||
                ((i.b >> 24) != 128 && (i.b >> 24) != 125)))
                refuse(pc, "indexed compute atomic requires zero byte offsets");
            i.name = encoded.name;
        } else if ((i.op >= 4 && i.op <= 7) || (i.op >= 0x84 && i.op <= 0x87)) {
            if ((a & 0x3ffffu) != 0x2000u || (i.b & 0xffe00000u) != 0x80000000u || i.op == 0x86)
                refuse(pc, "formatted store needs an index-only linear view and supported component count");
            i.name = "buffer_store_format";
        } else if (i.op >= 0x1c && i.op <= 0x1f) {
            if ((a & (direct && !graphics ? 0x32000u : 0x3e000u)) ||
                (i.b & (direct && !graphics ? 0xa00000u : 0xe00000u)))
                refuse(pc, "raw store needs a linear byte offset");
            i.name = "buffer_store_raw";
        } else if (i.op >= 0x0c && i.op <= 0x0f) {
            const unsigned soffset = i.b >> 24;
            if (direct && (!graphics || !(a & 0x2000u))) {
                // Linear MUBUF uses an optional OFFEN VGPR, scalar byte offset and OFFSET12.
                if ((a & 0x02032000u) || (i.b & 0x00a00000u) || (soffset > 105 && soffset != 128))
                    refuse(pc, graphics ? "raw graphics load requires linear addressing without LDS or TFE"
                                        : "raw compute load requires linear addressing without LDS or TFE");
            } else if (!(direct && graphics && (a & 0x0203f000u) == 0x2000u &&
                        (i.b & 0xffe00000u) == 0x80000000u) &&
                       ((a & 0x0203f000u) != 0x1000u || (i.b & 0xffe00000u) != 0x80000000u))
                refuse(pc, "raw buffer load needs offen with zero scalar offset and no modifiers");
            i.name = "buffer_load_dword";
        } else {
            if (i.op > 3)
                refuse(pc, "unsupported buffer operation");
            const auto addressing = a & 0x3ffffu;
            if ((addressing != 0x2000u && !(direct && addressing == 0)) ||
                (a & 0x02000000u) || (i.b & 0xffe00000u) != 0x80000000u)
                refuse(pc, "formatted load requires zero or a single linear index");
            i.name = "buffer_load_format";
        }
        i.enc = Encoding::Mubuf;
        i.dst = (i.b >> 8) & 255;
    } else if (direct && !graphics && encoded.form == gfx1013::Form::mtbuf) {
        i.b = take();
        i.op = encoded.opcode;
        if (i.op > 3 || (a & 0xe000u) || (i.b & 0xe00000u))
            refuse(pc, "MTBUF requires a non-D16 load without index, cache modifiers or TFE");
        i.enc = Encoding::Mubuf;
        i.name = encoded.name;
        i.dst = (i.b >> 8) & 255;
    } else if (direct && !graphics && (a & 0xfc000000u) == 0xdc000000u) {
        // FLAT encoding, segment 2 (GLOBAL). Only the one-dword load with an SGPR-pair base (the saddr form,
        // ADDR = SADDR + zext(VADDR) + sext(OFFSET)) is lowered, as gfx2hlsl.py op_global_load_dword does.
        // SADDR 125 is `off` on gfx10. LDS, the other segments and opcodes stay refused.
        i.b = take();
        i.op = (a >> 18) & 127;
        const unsigned saddr = (i.b >> 16) & 127;
        if (((a >> 14) & 3) != 2 || i.op != 12 || (a & 0x2000u) || saddr == 125 || saddr == 127)
            refuse(pc, "direct global memory supports only global_load_dword with a scalar address pair");
        i.enc = Encoding::Mubuf;
        i.name = "global_load_dword";
        i.dst = i.b >> 24;
    } else if ((a & 0xfc000000u) == 0xd8000000) {
        i.enc = Encoding::Lds;
        i.b = take();
        i.name = "lds";
    } else if ((a & 0xfc000000u) == 0xd4000000 || (a & 0xff800000u) == 0xcc000000) {
        i.wide = true;
        i.b = take();
        i.enc = (a >> 26) == 0x35 ? Encoding::Vop3 : Encoding::Mix;
        i.op = (a >> 16) & (i.enc == Encoding::Mix ? 0x7f : 0x3ff);
        i.src = {i.b & 511, (i.b >> 9) & 511, (i.b >> 18) & 511};
        i.abs = (a >> 8) & 7;
        i.neg = i.b >> 29;
        i.scale = (i.b >> 27) & 3;
        i.clamp = (a & 0x8000) != 0;
        i.dst = 256 + (a & 255);
        if (i.enc == Encoding::Vop3 && i.op < 256) {
            i.enc = Encoding::Vopc;
            i.dst = a & 255;
        } else if (i.enc == Encoding::Vop3 && i.op >= 0x100 && i.op < 0x140) {
            i.enc = Encoding::Vop2;
            i.op -= 0x100;
        } else if (i.enc == Encoding::Vop3 && i.op >= 0x180 && i.op < 0x200) {
            i.enc = Encoding::Vop1;
            i.op -= 0x180;
        }
        lit = std::find(i.src.begin(), i.src.end(), 255) != i.src.end();
    } else
        refuse(pc, "unsupported instruction " + string(encoded.name) + " " + hexword(a));
    if (!i.wide && (i.enc == Encoding::Vop1 || i.enc == Encoding::Vop2 || i.enc == Encoding::Vopc)) {
        const bool half_alu = !graphics && i.enc == Encoding::Vop2 && binary16_alu(encoded.name);
        if (direct && half_alu && i.src[0] == 250)
            refuse(pc, "unsupported " + string(encoded.name) + " DPP form");
        const bool sdwa_logic = i.enc == Encoding::Vop2 && i.op == 0x1d;
        const bool sdwa_class = i.enc == Encoding::Vopc && (i.op == 0x88 || (direct && i.op == 0x8f));
        const bool sdwa_compare = i.enc == Encoding::Vopc && !sdwa_class;
        const unsigned sdwa_mask_limit = direct && !graphics && !wave64 ? 107 : 106;
        if (direct && i.src[0] == 249 && (sdwa_compare || i.enc == Encoding::Vop1 || i.enc == Encoding::Vop2)) {
            // Operand profiles from disasm_rdna2's GFX10 VOP1/VOP2 SDWA
            // decoder. Admit only profiles with a direct result placement.
            static const std::set<Name> bits{
                "v_mov_b32", "v_not_b32", "v_cndmask_b32", "v_bfrev_b32", "v_mul_u32_u24",
                "v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32",
                "v_and_b32", "v_or_b32", "v_xor_b32", "v_xnor_b32", "v_min_u32", "v_max_u32",
                "v_min_i32", "v_max_i32", "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32"};
            static const std::set<Name> floats{
                "v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mul_f32", "v_min_f32", "v_max_f32",
                "v_floor_f32", "v_ceil_f32", "v_fract_f32", "v_trunc_f32", "v_rndne_f32",
                "v_exp_f32", "v_log_f32", "v_rcp_f32", "v_rsq_f32", "v_sqrt_f32"};
            const auto name = encoded.name;
            const bool fp_input = half_alu || floats.count(name) || name == "v_cvt_u32_f32" || name == "v_cvt_i32_f32" ||
                                  name == "v_cvt_f16_f32" || name == "v_cvt_f32_f16" ||
                                  (sdwa_compare && (name.substr(name.size()-4) == "_f32" || name.substr(name.size()-4) == "_f16"));
            const bool conversion = name == "v_cvt_f32_i32" || name == "v_cvt_f32_u32" ||
                                    name.substr(0,15) == "v_cvt_f32_ubyte";
            if (!(fp_input || conversion || bits.count(name) || sdwa_compare))
                refuse(pc, "unimplemented direct SDWA operand profile");
            const Word ext = take();
            const unsigned sources = i.enc == Encoding::Vop1 ? 1 : 2;
            if (sdwa_compare) {
                if ((ext & 0x8000) && (name.substr(0,7) == "v_cmpx_" || ((ext >> 8) & 127) > sdwa_mask_limit))
                    refuse(pc, "invalid SDWA compare destination");
                i.dst = ext & 0x8000 ? (ext >> 8) & 127 : 106;
            } else {
                i.dst_select = (ext >> 8) & 7;
                i.dst_unused = (ext >> 11) & 3;
                i.scale = (ext >> 14) & 3;
                i.clamp = (ext & (1u << 13)) != 0;
                if (half_alu && ((i.dst_select != 4 && i.dst_select != 5) ||
                                 (i.dst_unused != 0 && i.dst_unused != 2)))
                    refuse(pc, "unsupported SDWA destination modifier");
                if (i.dst_select == 7 || i.dst_unused == 3 || ((i.clamp || i.scale) && !floats.count(name) && !conversion) ||
                    (sources == 1 && (ext & 0xbf000000u)))
                    refuse(pc, "unsupported SDWA destination modifier");
            }
            for (unsigned j = 0; j < sources; ++j) {
                const unsigned control = (ext >> (16 + 8*j)) & 255;
                i.src_select[j] = control & 7;
                i.src_sext[j] = (control & 8) != 0;
                if (half_alu && (control & 128))
                    refuse(pc, "SDWA binary16 source must be a VGPR");
                if (half_alu && i.src_select[j] < 4)
                    refuse(pc, "invalid SDWA source modifier");
                if (i.src_select[j] == 7 || (control & 64) || (fp_input && i.src_sext[j]) || (!fp_input && (control & 0x30)))
                    refuse(pc, "invalid SDWA source modifier");
                i.abs |= ((control >> 5) & 1) << j;
                i.neg |= ((control >> 4) & 1) << j;
                const unsigned reg = j ? (a >> 9) & 255 : ext & 255;
                if ((control & 128) && reg == 255)
                    refuse(pc, "literal is unavailable in SDWA");
                i.src[j] = reg + ((control & 128) ? 0 : 256);
            }
            i.subword = true;
            lit = false;
        } else if (i.src[0] == 249 && (sdwa_logic || sdwa_class)) {
            const Word ext = take();
            i.src_select = {(ext >> 16) & 7, (ext >> 24) & 7};
            i.src_sext = {bool(ext & (1u << 19)), bool(ext & (1u << 27))};
            if (sdwa_logic) {
                i.dst_select = (ext >> 8) & 7;
                i.dst_unused = (ext >> 11) & 3;
                if ((ext & 0x7070e000u) || i.dst_select == 7 || i.dst_unused == 3)
                    refuse(pc, "invalid or unsupported integer SDWA modifiers");
            } else {
                // VOPC has a scalar mask destination in the same bits that VOP2 uses for placement.
                i.dst = ext & 0x8000 ? (ext >> 8) & 127 : 106;
                if ((ext & 0x70480000u) || i.dst > sdwa_mask_limit)
                    refuse(pc, "invalid SDWA class modifiers or mask destination");
                i.neg = (ext >> 20) & 1;
                i.abs = (ext >> 21) & 1;
            }
            if (i.src_select[0] == 7 || i.src_select[1] == 7)
                refuse(pc, "invalid or unsupported integer SDWA modifiers");
            i.src[0] = (ext & 255) + (ext & (1u << 23) ? 0 : 256);
            i.src[1] = ((a >> 9) & 255) + (ext & (1u << 31) ? 0 : 256);
            if (i.src[0] == 255 || i.src[1] == 255)
                refuse(pc, "literal is unavailable in SDWA");
            i.subword = true;
            lit = false;
        } else if (i.src[0] == 249 && i.enc == Encoding::Vop2 && (i.op == 1 || i.op == 0x25)) {
            auto ext = take();
            if ((ext & 0xfff8ff00u) != 0x06000600u || ((ext >> 16) & 7u) > 6)
                refuse(pc, "SDWA source or destination modifier");
            i.subword = true;
            i.src_select[0] = (ext >> 16) & 7u;
            i.src[0] = 256 + (ext & 255);
            lit = false;
        } else if (direct && i.src[0]==250 && (graphics || i.enc!=Encoding::Vopc) && !lit) {
            auto control=take();
            // DPP_CTRL[8] selects patterns outside a quad; bit 17 is
            // reserved. FI and BOUND_CTRL are independent of the pattern.
            if ((control >> 24) != 0xff) refuse(pc,"DPP requires a full-row, full-bank mask");
            if (control & (1u << 17)) refuse(pc,"reserved DPP16 control bit");
            const auto pattern = (control >> 8) & 0x1ff;
            i.dpp_control = pattern;
            if (pattern > 255 && !i.mirror_mask() && i.dpp_row().op == Inst::DppRowOp::None)
                refuse(pc,"unsupported DPP lane selection");
            i.dpp_fetch_inactive = (control & (1u << 18)) != 0;
            i.dpp_bound_zero = (control & (1u << 19)) != 0;
            i.src[0]=256+(control&255);
            i.neg=((control>>20)&1)|((control>>21)&2);
            i.abs=((control>>21)&1)|((control>>22)&2);
        } else if (graphics && (i.src[0] == 233 || i.src[0] == 234) && i.enc != Encoding::Vopc && !lit) {
            // pixel_aot.parse_dpp: both halves of an octet must select the
            // same local 2x2 permutation to map to desktop quad operations.
            const Word control = take();
            unsigned perm = 0;
            for (unsigned lane = 0; lane < 4; ++lane) {
                const unsigned low = (control >> (8 + 3 * lane)) & 7;
                const unsigned high = (control >> (20 + 3 * lane)) & 7;
                if (low >= 4 || high != low + 4) refuse(pc, "DPP8 requires repeating quad-local selections");
                perm |= low << (2 * lane);
            }
            i.dpp_control = perm;
            i.dpp_fetch_inactive = i.src[0] == 234;
            i.src[0] = 256 + (control & 255);
        } else if (i.src[0] == 249 || i.src[0] == 250 || i.src[0] == 233 || i.src[0] == 234)
            refuse(pc, "SDWA or DPP vector operands");
    }
    if (lit)
        i.literal = take();
    if (i.enc == Encoding::Vop1 && !i.op) i.name = "v_nop";
    if (i.name.empty()) {
        const char *name = mnemonic(i.enc,i.op);
        if (!name && (direct || encoded.name == "v_readlane_b32") && !encoded.name.empty())
            name = encoded.name.data();
        if (!name)
            refuse(pc, "unsupported opcode " + hexword(a));
        i.name = name;
        if (i.name.substr(0, 7) == "v_cmpx_")
            i.dst = 126;
        if (i.name == "v_readfirstlane_b32" || i.name == "v_readlane_b32") {
            i.dst -= 256;
            if (i.dst > (graphics ? 105u : 107u))
                refuse(pc, "invalid scalar lane read destination");
        }
    }
    // Keep permuted VCC compares separate from unpermuted compare/EXEC proofs.
    if (i.enc == Encoding::Vopc && i.dpp_control) {
        if (i.dst != 106 || i.name.substr(0,6) != "v_cmp_" || i.name.substr(i.name.size()-4) != "_f32" ||
            i.name.find("_class_") != Name::npos || *i.dpp_control > 255)
            refuse(pc,"unsupported DPP pixel compare");
        i.enc = Encoding::QuadCompare;
    }
    if(i.next-pc!=encoded.size)refuse(pc,"instruction extent mismatch for "+string(encoded.name));
    return i;
}
// VOP3B carries a scalar destination in the field used for ABS in VOP3A.
static std::optional<unsigned> carry_destination(const Inst &i) {
    if (i.name == "v_add_co_u32" || i.name == "v_sub_co_u32" || i.name == "v_subrev_co_u32" ||
        i.name == "v_add_co_ci_u32" || i.name == "v_sub_co_ci_u32" || i.name == "v_subrev_co_ci_u32")
        return i.wide ? (i.a >> 8) & 127 : 106;
    if (i.name == "v_mad_u64_u32")  // VOP3B: the carry-out of the 64-bit add (#3000)
        return (i.a >> 8) & 127;
    return {};
}
struct Symbol {
    enum class Type { unknown, constant, masked, table, descriptor, indexed_descriptor, sampler_heap, sampler_word, root_address, root_stride_one_high, primitive_mask,
                      stride, byte_stride, zero_stride_word, byte_count, clear_oob, byte_oob, lane_mask, uniform_mask, unbound } type{};
    Word value = 0, offset = 0;
    unsigned index_mask = 0, definition = 0;
    bool numeric() const { return type == Type::unknown || type == Type::constant || type == Type::masked || type == Type::uniform_mask; }
    bool operator==(const Symbol &r) const { return type == r.type && value == r.value && offset == r.offset &&
        index_mask == r.index_mask && definition == r.definition; }
};
using State = std::array<Symbol, 128>;
// FSSM root descriptors carry a guest address in two SGPRs. Preserve the
// identity of each half through scalar moves and CFG joins (Victory resource.rs).
void bind_root_address(State &state, const ShaderRootInput &input, unsigned limit) {
    const unsigned at = input.user_data;
    if (input.sgpr_count != 2 || at >= limit || limit - at < 2 || input.root_parameter >= 16 ||
        input.offset_dwords || state[at].type != Symbol::Type::unbound || state[at+1].type != Symbol::Type::unbound)
        refuse(0, "invalid FSSM root descriptor pair");
    for (unsigned half = 0; half < 2; ++half)
        state[at + half] = {Symbol::Type::root_address, input.root_parameter, half};
}
Symbol source(const Inst &i, unsigned code, const State &s) {
    if (code < 128)
        return s[code];
    if (code <= 192)
        return {Symbol::Type::constant, code - 128};
    if (code <= 208)
        return {Symbol::Type::constant, 192u - code};
    if (code == 255)
        return {Symbol::Type::constant, i.literal};
    return {};
}
unsigned memory_count(const Inst &i) { return 1u << (i.op & 7); }
// Derive the rewritten byte-address V# symbolically. Each field carries its
// source table/slot identity. Numeric consumption still requires real bits.
// heap_window: the direct path under HeapViewContract::raw_buffer also rewrites a
// descriptor loaded from the GPU-selected heap slot (its words keep the load pc).
static std::optional<Symbol> byte_descriptor_field(const Inst &i, const State &s, bool heap_window = false) {
    using T = Symbol::Type;
    if (i.enc != Encoding::Sop2 || i.dst > 105) return {};
    auto a = source(i, i.src[0], s), b = source(i, i.src[1], s);
    auto component = [heap_window](const Symbol &v, unsigned ch) {
        return v.offset % 32 == ch*4 && ((v.type == T::descriptor && v.value != compute_resource_heap) ||
               (heap_window && v.type == T::indexed_descriptor && v.value == compute_resource_heap));
    };
    auto field = [](Symbol v, T kind) {
        v.type = kind;
        v.offset &= ~31u;
        return v;
    };
    if (i.name == "s_bfe_u32" && component(a, 1) && b.type == T::constant && b.value == 0xe0010)
        return field(a, T::stride);
    if (i.name == "s_mul_i32") {
        if (a.type == T::byte_stride) std::swap(a,b);
        if (component(a, 2) && b == field(a,T::byte_stride)) return field(a, T::byte_count);
    }
    if (a.type == T::constant) std::swap(a,b);
    if (b.type != T::constant) return {};
    // Address bits occupy the low halfword. Setting bit 16 in the high
    // address word constructs a buffer descriptor with byte stride one.
    if (i.name == "s_or_b32" && a.type == T::root_address && a.offset == 1 && b.value == 0x10000) {
        a.type = T::root_stride_one_high;
        return a;
    }
    if (i.name == "s_max_u32" && a.type == T::stride && b.value == 1) return field(a, T::byte_stride);
    if (i.name == "s_and_b32" && b.value == 0xc000ffff && component(a,1)) return field(a,T::zero_stride_word);
    if (i.name == "s_and_b32" && b.value == 0xcfffffff && component(a,3)) return field(a,T::clear_oob);
    if (i.name == "s_or_b32" && b.value == 0x30000000 && (a.type == T::clear_oob || component(a,3)))
        return field(a,T::byte_oob);
    return {};
}
State transfer(const Inst &i, State s, bool wave64, bool graphics, bool heap_window = false) {
    auto set = [&](unsigned n, Symbol v) {
        if (n >= s.size())
            refuse(i.pc, "invalid SGPR");
        s[n] = v;
    };
    auto mask = carry_destination(i);
    const bool compared = graphics && i.enc == Encoding::Vopc && i.dst < 106 && i.name.substr(0,6) == "v_cmp_";
    if (mask || compared) {
        const unsigned dst = mask.value_or(i.dst);
        if ((dst > 105 && dst != 106) || (wave64 && (dst & 1)))
            refuse(i.pc, compared ? "unsupported comparison mask destination" : "unsupported carry mask destination");
        const bool tagged = graphics && dst != 106;
        set(dst, tagged ? Symbol{Symbol::Type::lane_mask,i.pc,0} : Symbol{});
        if (wave64) set(dst + 1, tagged ? Symbol{Symbol::Type::lane_mask,i.pc,1} : Symbol{});
    } else if (graphics && i.name == "s_cselect_b64" && i.dst <= 104 && !(i.dst & 1) &&
               (i.src[0] == 193 || i.src[0] == 128) && (i.src[1] == 193 || i.src[1] == 128)) {
        set(i.dst,{Symbol::Type::uniform_mask,i.pc,0});
        set(i.dst+1,{Symbol::Type::uniform_mask,i.pc,1});
    } else if (auto field = byte_descriptor_field(i,s,heap_window)) {
        set(i.dst,*field);
    } else if (i.enc == Encoding::FixedSetup) {
        for (unsigned r = i.dst; r < i.dst + i.op && r < s.size(); ++r) s[r] = {};
    } else if (i.enc == Encoding::Fetch) {
        s[4].type = s[5].type = Symbol::Type::unbound;
    } else if (i.enc == Encoding::Smem) {
        auto base = s[i.src[0]];
        const unsigned offset_reg = i.b >> 25;
        const Symbol index = offset_reg <= 105 ? s[offset_reg] : Symbol{};
        unsigned n = memory_count(i);
        for (unsigned j = 0; j < n; ++j) {
            Symbol v{};
            if (graphics && i.op == 3 && base.type == Symbol::Type::table && base.value == compute_resource_heap) {
                const bool fixed = offset_reg == 125 || index.type == Symbol::Type::constant;
                const Word offset = (i.b & 0xfffff) + (offset_reg != 125 && fixed ? index.value : 0);
                v = {Symbol::Type::indexed_descriptor, base.value, offset + 4*j,
                     fixed ? 0u : index.type == Symbol::Type::masked ? index.value : ~0u, i.pc};
            } else if (i.op == 2 && base.type == Symbol::Type::sampler_heap) {
                v = {Symbol::Type::sampler_word, i.dst, j, 0, i.pc};
            } else if ((i.op == 2 || (graphics && i.op == 3)) && offset_reg != 125 && base.type == Symbol::Type::table) {
                v = {Symbol::Type::indexed_descriptor, base.value, (i.b & 0xfffff) + 4*j,
                     index.type == Symbol::Type::masked || index.type == Symbol::Type::constant ? index.value : ~0u, i.pc};
            } else if (i.op < 8 && base.type == Symbol::Type::table)
                v = {Symbol::Type::descriptor, base.value, (i.b & 0xfffff) + 4 * j};
            set(i.dst + j, v);
        }
    } else if (i.name == "s_mov_b32" || i.name == "s_mov_b64") {
        auto v = source(i, i.src[0], s);
        Symbol high = i.src[0] < 127 ? s[i.src[0] + 1]
                                     : Symbol{Symbol::Type::constant, (i.src[0] >= 193 && i.src[0] <= 208) ? ~0u : 0u};
        set(i.dst, v);
        if (i.name == "s_mov_b64")
            set(i.dst + 1, high);
    } else if (i.name == "s_and_b32" || i.name == "s_lshl_b32") {
        const auto a = source(i, i.src[0], s), b = source(i, i.src[1], s);
        Symbol out{};
        if (i.name == "s_and_b32" && (a.type == Symbol::Type::constant || b.type == Symbol::Type::constant))
            out = {Symbol::Type::masked, a.type == Symbol::Type::constant ? a.value : b.value};
        if (i.name == "s_lshl_b32" && b.type == Symbol::Type::constant && a.numeric()) {
            const Word mask = a.type == Symbol::Type::masked ? a.value : ~0u;
            out = {Symbol::Type::masked, mask << (b.value & 31)};
        }
        if (a.type == Symbol::Type::constant && b.type == Symbol::Type::constant)
            out = {Symbol::Type::constant,i.name=="s_and_b32" ? a.value & b.value : a.value << (b.value & 31)};
        set(i.dst, out);
    } else if (!graphics && i.name == "s_bfm_b64") {
        auto a = source(i, i.src[0], s), b = source(i, i.src[1], s);
        Symbol lo{}, hi{};
        if (a.type == Symbol::Type::constant && b.type == a.type) {
            const auto mask = ((uint64_t{1} << (a.value & 63)) - 1) << (b.value & 63);
            lo = {a.type, Word(mask)};
            hi = {a.type, Word(mask >> 32)};
        }
        set(i.dst, lo);
        set(i.dst + 1, hi);
    } else if (i.name == "s_bfm_b32") {
        auto a = source(i, i.src[0], s), b = source(i, i.src[1], s);
        Symbol v{};
        if (a.type == Symbol::Type::constant && b.type == a.type)
            v = {a.type, ((1u << (a.value & 31)) - 1u) << (b.value & 31)};
        set(i.dst, v);
    } else if (i.name == "s_movk_i32") {
        set(i.dst, {Symbol::Type::constant, Word(int16_t(i.a))});
    } else if (i.enc == Encoding::Sop1 || i.enc == Encoding::Sop2 || i.enc == Encoding::Vopc ||
               i.enc == Encoding::QuadCompare ||
               i.name == "s_mulk_i32" || i.name == "s_addk_i32" || i.name == "s_cmovk_i32" ||
               i.name == "v_readfirstlane_b32" || i.name == "v_readlane_b32")
    {
        set(i.dst, {});
        if ((wave64 && (i.enc == Encoding::Vopc || i.enc == Encoding::QuadCompare)) || ((i.enc == Encoding::Sop1 || i.enc == Encoding::Sop2) &&
            i.name.size() >= 4 &&
            (i.name.substr(i.name.size() - 4) == "_b64" || i.name.substr(i.name.size() - 4) == "_i64" ||
             i.name.substr(i.name.size() - 4) == "_u64") &&
            i.name.find("_i32_") == Name::npos) || i.name == "s_bitreplicate_b64_b32")
            set(i.dst + 1, {});
    }
    return s;
}
struct Program {
    bool graphics_wave64 = false;
    std::vector<Inst> code;
    std::map<unsigned, size_t> at;
    std::vector<State> before;
    std::vector<bool> reached;
};
// Recognize slot-zero vertex-buffer loads as a group. Other scalar table
// loads interleaved by the scheduler stay in the normal instruction stream.
std::map<unsigned, Inst> fixed_vertex_loads(const string &code) {
    std::map<unsigned, Inst> result;
    unsigned pc = 0;
    while (pc + 12 <= code.size()) {
        const Word a = word(code, pc);
        if ((a & 0xff80ffffu) == 0xbe800400u) break;
        const auto i = decode(code, pc, true, true);
        if (!(i.name == "v_nop" || i.name == "s_nop" || i.name == "s_setprio" ||
              i.name == "s_inst_prefetch" || i.name == "s_clause" ||
              ((i.name == "s_mov_b32" || i.enc == Encoding::Smem) && i.dst >= 3 && i.dst < 106)))
            return {};
        pc = i.next;
    }
    if (pc + 12 > code.size()) return {};
    const unsigned ptr = (word(code, pc) >> 16) & 127;
    if (ptr < 4 || ptr > 104 || (ptr & 1) || word(code,pc+4) != (0x9380ff00u | ((ptr+1)<<16) | (ptr+1)) ||
        word(code,pc+8) != 0x80000u) return {};
    auto setup = [&](unsigned bytes, unsigned dst, unsigned count) {
        Inst i{};
        i.pc = pc; i.next = pc + bytes; i.enc = Encoding::FixedSetup; i.name = "fixed_vertex_setup";
        i.dst = dst; i.op = count; result.emplace(pc,i); pc = i.next;
    };
    setup(4, ptr, 2);
    setup(8, ptr + 1, 1);
    unsigned desc = UINT_MAX;
    while (pc + 8 <= code.size()) {
        Word a = word(code,pc), b = word(code,pc+4);
        if ((a & 0xffff0000u) == 0xbfa10000u) { pc += 4; continue; }
        if ((a & 0xffffe000u) != 0xf4080000u) return {};
        const unsigned dst = (a >> 6) & 127, base = (a & 63) * 2;
        if (base == ptr && b == 0xfa000000u && !(dst & 3) && dst <= 100) {
            desc = dst;
            setup(8, dst, 4);
            break;
        }
        if (base == ptr || (b >> 25) != 125 || (dst < ptr + 2 && ptr < dst + 4) || dst < 3)
            return {};
        pc += 8;
    }
    if (desc == UINT_MAX || pc + 8 > code.size() || word(code,pc) != 0x4a000002u ||
        word(code,pc+4) != 0xbf8cc07fu) return {};
    setup(4, 256, 1); // v0 is the adjusted fetch index, consumed by the IA.
    pc += 4;
    if (pc + 4 <= code.size() && (word(code,pc) & 0xffff0000u) == 0xbfa10000u) pc += 4;
    bool found = false, overwrote_index = false;
    while (pc + 8 <= code.size()) {
        const Word a = word(code,pc), b = word(code,pc+4);
        const Word form = a & ~4095u;
        unsigned width, alignment;
        bool normalized;
        switch (form) {
        case 0xe0342000u: width = 2; alignment = 4; normalized = false; break;
        case 0xe0382000u: width = 4; alignment = 4; normalized = false; break; // buffer_load_dwordx4 idxen
        case 0xe03c2000u: width = 3; alignment = 4; normalized = false; break;
        case 0xe8282000u: width = 1; alignment = 1; normalized = false; break; // BUF_FMT_8_UINT
        case 0xe8082000u: width = 1; alignment = 1; normalized = true; break; // BUF_FMT_8_UNORM
        case 0xe8712000u: width = 2; alignment = 2; normalized = true; break; // BUF_FMT_8_8_UNORM
        case 0xe9c32000u: width = 4; alignment = 4; normalized = true; break; // BUF_FMT_8_8_8_8_UNORM
        default: return found ? result : std::map<unsigned, Inst>{};
        }
        if (overwrote_index || (b & 0xffff00ffu) != (0x80000000u | ((desc / 4) << 16))) return {};
        Inst i{};
        i.pc = pc; i.next = pc + 8; i.enc = Encoding::FixedFetch; i.name = "fixed_vertex_fetch";
        i.dst = 256 + ((b >> 8) & 255);
        i.op = width;
        i.a = normalized;
        i.literal = a & 4095u;
        if (i.literal % alignment || i.literal >= 2048 || i.dst+i.op > 512) return {};
        result.emplace(pc, i);
        found = true; overwrote_index = i.dst == 256; pc += 8;
    }
    return found ? result : std::map<unsigned, Inst>{};
}
Program parse(const string &code, unsigned entry, const State &initial, bool graph, bool graphics=false, bool wave64=false, bool fixed_vertex=false,
              bool heap_window=false) {
    Program p;
    p.graphics_wave64 = graphics && wave64;
    if ((entry & 3) || entry >= code.size() || code.size() > 1u << 20)
        refuse(entry, "invalid entry");
    if (graph) {
        const auto fixed = fixed_vertex ? fixed_vertex_loads(code) : std::map<unsigned, Inst>{};
        std::map<unsigned, Inst> instructions;
        std::deque<unsigned> pending{entry};
        while (!pending.empty()) {
            unsigned pc = pending.front();
            pending.pop_front();
            auto next = instructions.lower_bound(pc);
            if (next != instructions.end() && next->first == pc) continue;
            if (pc >= code.size() || (next != instructions.begin() && std::prev(next)->second.next > pc))
                refuse(pc, "branch outside code or into an instruction");
            const auto fetch = fixed.find(pc);
            auto instruction = fetch != fixed.end() ? fetch->second : decode(code, pc,true,graphics,wave64);
            if (next != instructions.end() && instruction.next > next->first)
                refuse(pc, "overlapping instruction boundaries");
            instructions.emplace_hint(next, pc, instruction);
            if (instruction.branch()) pending.push_back(instruction.target());
            if (instruction.name != "s_endpgm" && !(instruction.branch() && instruction.op == 2))
                pending.push_back(instruction.next);
        }
        for (const auto &[pc, instruction] : instructions) {
            p.at.emplace(pc, p.code.size());
            p.code.push_back(instruction);
        }
    } else {
        for (unsigned pc = entry;;) {
            auto i = decode(code, pc);
            p.at.emplace(pc, p.code.size());
            p.code.push_back(i);
            if (i.name == "s_endpgm") break;
            pc = i.next;
        }
    }
    // Include the loop's entry and all branch joins when proving descriptor provenance.
    p.before.resize(p.code.size());
    p.reached.resize(p.code.size());
    const size_t start = p.at.at(entry);
    p.before[start] = initial;
    p.reached[start] = true;
    std::deque<size_t> work{start};
    while (!work.empty()) {
        auto k = work.front();
        work.pop_front();
        auto &i = p.code[k];
        auto after = transfer(i, p.before[k], wave64, graphics, heap_window);
        auto merge = [&](size_t t) {
            bool changed = !p.reached[t];
            auto &in = p.before[t];
            if (changed) {
                in = after;
                p.reached[t] = true;
            } else
                for (unsigned n = 0; n < in.size(); ++n)
                    if (!(in[n] == after[n])) {
                        Symbol merged{};
                        if (!in[n].numeric() || !after[n].numeric())
                            merged.type = Symbol::Type::unbound;
                        if (!(in[n] == merged)) {
                            in[n] = merged;
                            changed = true;
                        }
                    }
            if (changed)
                work.push_back(t);
        };
        if (i.branch()) {
            auto t = p.at.find(i.target());
            if (t == p.at.end())
                refuse(i.pc, "invalid branch target");
            merge(t->second);
        }
        if (i.name != "s_endpgm" && !(i.branch() && i.op == 2))
            merge(p.at.at(i.next));
    }
    return p;
}
string reg(unsigned n) { return n >= 256 ? "v" + to_string(n - 256) : "s" + to_string(n); }
string operand(const Inst &i, unsigned n) {
    if ((n <= 105) || n == 106 || n == 126 || n >= 256)
        return reg(n);
    if (n >= 128 && n <= 208)
        return hexword(n <= 192 ? n - 128 : 192u - n);
    if (n == 253)
        return "uint(scc)";
    if (n == 255)
        return hexword(i.literal);
    constexpr Word f[] = {0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x40000000,
                          0xc0000000, 0x40800000, 0xc0800000, 0x3e22f983};
    if (n >= 240 && n <= 248)
        return hexword(f[n - 240]);
    refuse(i.pc, "unsupported operand " + to_string(n));
}
enum class AluType { Bits, Float, ToFloat };
struct AluRule {
    unsigned arity;
    AluType type;
    Name expression;
};
#include "gfx1013_compute_alu.inl"

struct Emit {
    NativeComputeShader result{};
    const Program &program;
    const D3D12_ROOT_SIGNATURE_DESC1 &layout;
    std::ostringstream body;
    unsigned srv = 0, uav = 0, first_sampler = 0;
    string binding(const Inst &i, const State &state, unsigned base, unsigned width, ComputeViewKind kind,
                   unsigned bytes = 0,unsigned dimension = 1) {
        auto sym = state.at(base);
        const bool indexed = sym.type == Symbol::Type::indexed_descriptor;
        if ((!indexed && sym.type != Symbol::Type::descriptor) || base + width > state.size())
            refuse(i.pc, "unproven resource descriptor s" + to_string(base));
        if (sym.value == compute_sampler_heap)
            refuse(i.pc, "a sampler heap descriptor is read as a resource");
        const bool byte_window = kind == ComputeViewKind::byte_window_srv ||
                                 (kind == ComputeViewKind::raw_buffer_uav && width == 4 &&
                                  (i.name == "buffer_store_raw" ||
                                   (i.name.substr(0, 14) == "buffer_atomic_" && !(i.a & 0x2000u))) &&
                                  state[base + 1].type == Symbol::Type::zero_stride_word);
        for (unsigned j = 1; j < width; ++j) {
            Symbol expected = sym;
            if (byte_window) {
                constexpr Symbol::Type fields[] = {Symbol::Type::descriptor, Symbol::Type::zero_stride_word,
                                                   Symbol::Type::byte_count, Symbol::Type::byte_oob};
                expected.type = fields[j];
            } else expected.offset += j * 4;
            if (!(state[base + j] == expected))
                refuse(i.pc, "non-contiguous resource descriptor");
        }
        if (sym.offset % 32)
            refuse(i.pc, "unaligned descriptor table entry");
        auto &list = result.bindings;
        if (sym.value == compute_resource_heap)
            refuse(i.pc, "whole-heap buffer access needs a matching descriptor at the GPU-selected index");
        if (indexed) {
            if ((width != 4 && width != 8) || compute_uav(kind) || (sym.index_mask & 31))
                refuse(i.pc, "indexed descriptor requires a bounded 32-byte-aligned SRV offset");
            const unsigned first = sym.offset / 32, count = 1 + sym.index_mask / 32;
            if (sym.value >= layout.NumParameters || layout.pParameters[sym.value].ParameterType != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
                refuse(i.pc, "missing indexed descriptor table range");
            const auto &table = layout.pParameters[sym.value].DescriptorTable;
            const uint64_t end = uint64_t(first) + count;
            if (count > D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1 ||
                end > (uint64_t(UINT_MAX) + 1) / 32)
                refuse(i.pc, "indexed SRV span exceeds its declared root range");
            std::vector<std::pair<uint64_t, uint64_t>> covered;
            uint64_t next = 0;
            for (unsigned j = 0; j < table.NumDescriptorRanges; ++j) {
                const auto &range = table.pDescriptorRanges[j];
                uint64_t start = range.OffsetInDescriptorsFromTableStart == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
                                     ? next : range.OffsetInDescriptorsFromTableStart;
                next = start + range.NumDescriptors;
                if (start >= end || next <= first) continue;
                if (range.RangeType != D3D12_DESCRIPTOR_RANGE_TYPE_SRV)
                    refuse(i.pc, "indexed SRV span overlaps a non-SRV root range");
                covered.emplace_back((std::max)(uint64_t(first), start), (std::min)(end, next));
            }
            // These are physical slots relative to the bound table. Adjacent
            // SRV ranges may name unrelated guest registers or register spaces.
            std::sort(covered.begin(), covered.end());
            uint64_t cursor = first;
            for (const auto &[lo, hi] : covered) {
                if (lo > cursor) break;
                cursor = (std::max)(cursor, hi);
            }
            if (cursor != end)
                refuse(i.pc, "indexed SRV span exceeds its declared root range");
            auto span = std::find_if(list.begin(), list.end(), [&](const auto &b) {
                return b.kind == kind && b.root_parameter == sym.value && b.byte_offset == sym.offset &&
                       b.array_count == count && b.image_dimension == dimension && !b.array_position;
            });
            unsigned first_reg = srv;
            if (span == list.end()) {
                for (unsigned j = 0; j < count; ++j)
                    list.push_back({kind, sym.value, sym.offset + j * 32, srv++, bytes, count, j, dimension});
            } else {
                first_reg = span->shader_register;
                for (unsigned j = 0; j < count; ++j)
                    span[j].required_bytes = (std::max)(span[j].required_bytes, bytes);
            }
            return "r"+to_string(first_reg)+(count==1 ? "" : "[NonUniformResourceIndex("+reg(base)+")]");
        }
        auto it = std::find_if(list.begin(), list.end(), [&](const auto &b) {
            if (compute_buffer_uav(kind) && b.kind == kind && b.root_parameter == sym.value &&
                b.byte_offset == sym.offset && kind != ComputeViewKind::raw_buffer_uav && b.required_bytes != bytes)
                refuse(i.pc, "incompatible formatted store widths on one slot");
            return b.array_count == 1 && b.image_dimension==dimension && b.kind == kind && b.root_parameter == sym.value && b.byte_offset == sym.offset;
        });
        bool write = compute_uav(kind);
        if (it == list.end()) {
            list.push_back({kind, sym.value, sym.offset, write ? uav++ : srv++, bytes,1,0,dimension});
            it = std::prev(list.end());
        }
        it->required_bytes = (std::max)(it->required_bytes, bytes);
        return (write ? "w" : "r") + to_string(it->shader_register);
    }
    // heap: the sampler is entry `index` of the directly indexed sampler heap. Otherwise `index` is the static sampler register.
    struct SamplerRef {
        bool heap;
        unsigned index;
    };
    SamplerRef sampler_index(const Inst &i, const State &state, bool lod, bool gather_base_level = false,
                             bool implicit_sample = false, bool comparison = false) {
        const unsigned base = ((i.b >> 21) & 31) * 4;
        ComputeSampler sampler{{}, lod, comparison};
        unsigned literal = 0, heap = 0;
        const auto &first = state.at(base);
        for (unsigned j = 0; j < 4; ++j) {
            const auto &s = state.at(base + j);
            literal += s.type == Symbol::Type::constant;
            // The heap content is fixed within a dispatch, so contiguity of the four words of one 16-byte entry is enough.
            heap += s.type == Symbol::Type::descriptor && s.value == compute_sampler_heap && !s.definition &&
                    first.offset % 16 == 0 && s.offset == first.offset + 4 * j;
            if (s.type == Symbol::Type::constant)
                sampler.words[j] = s.value;
        }
        if (literal != 4 && heap != 4)
            refuse(i.pc, "nonconstant sampler");
        if (heap == 4) {
            if (gather_base_level)
                refuse(i.pc, "gather through the sampler heap is not translated");
            if (comparison)
                refuse(i.pc, "comparison sampler from the sampler heap is not translated");
            if (implicit_sample)
                refuse(i.pc, "sampler from the sampler heap needs image_sample_l or image_sample_lz");
            const unsigned entry = first.offset / 16, bytes = 16 * (entry + 1);
            auto &list = result.bindings;
            auto found = std::find_if(list.begin(), list.end(),
                                      [](const auto &b) { return b.kind == ComputeViewKind::heap_sampler; });
            if (found == list.end())
                list.push_back({ComputeViewKind::heap_sampler, compute_sampler_heap, 0, 0, bytes});
            else
                found->required_bytes = (std::max)(found->required_bytes, bytes);
            return {true, entry};
        }
        if (comparison) {
            // Field constraints from pixel_aot.comparison_sampler_state. The
            // native static sampler also preserves the mip filter and LOD limits.
            auto &[address, limits, filter, border] = sampler.words;
            const unsigned mag = (filter >> 20) & 3, min = (filter >> 22) & 3, mip = (filter >> 26) & 3;
            if ((address & 63) != 0x12 || (address & ((7u << 9) | (1u << 15) | (3u << 19) | (3u << 29))) ||
                mag != min || mag > 1 || mip == 3 || ((address >> 27) & 1) != (mag == 0) ||
                (filter & 0xfffff) || (filter >> 28 & (implicit_sample ? 15 : 8)) || (border & 0x3ffff000))
                refuse(i.pc, "unsupported comparison sampler fields");
            if (limits & 0xfff)
                refuse(i.pc, "comparison sampler requires minimum LOD zero");
            if (!mip)
                limits &= ~0x00fff000u;
        } else if (implicit_sample) {
            // The static sampler carries MIN_LOD and MAX_LOD. Only encode a
            // gradient-derived LOD when its mip filter and bias are representable.
            unsigned mip = (sampler.words[2] >> 26) & 3;
            if ((sampler.words[2] & 0xfffff) || (mip != 1 && mip != 2) ||
                (sampler.words[1] & 0xfff) > ((sampler.words[1] >> 12) & 0xfff))
                refuse(i.pc, "implicit sample with unsupported sampler LOD fields");
        }
        if (gather_base_level) {
            // DXIL textureGather has no explicit LOD operand. A dedicated static
            // sampler fixes the gather footprint to the first mip of the SRV.
            if (sampler.words[1] & 0xfffu)
                refuse(i.pc, "base-level gather with a nonzero minimum LOD");
            sampler.words[1] &= ~0x00fff000u;
            sampler.explicit_lod = true;
        }
        auto &all = result.samplers;
        auto it = std::find_if(all.begin(), all.end(),
                               [&](const auto &s) { return s.words == sampler.words && s.explicit_lod == sampler.explicit_lod &&
                                                           s.comparison == sampler.comparison; });
        if (it == all.end()) {
            all.push_back(sampler);
            it = std::prev(all.end());
        }
        return {false, first_sampler + static_cast<unsigned>(it - all.begin())};
    }
    void image(const Inst &i, const State &state) {
        unsigned mask = (i.a >> 8) & 15, base = ((i.b >> 16) & 31) * 4, addr = i.b & 255;
        const bool half_store = i.op == 8 && (i.b & 0x80000000u);
        if (!mask || (i.a & 0x3e0c0u) || (i.b & 0x3c000000u))
            refuse(i.pc, "unsupported image flags");
        bool store = i.op == 8, lod = i.op == 39, gather = i.op == 71;
        if (i.op != 0 && !store && !lod && !gather)
            refuse(i.pc, "unsupported image opcode");
        auto tex = binding(i, state, base, 8, half_store ? ComputeViewKind::half_scalar_uav :
                           store ? ComputeViewKind::texture_uav : ComputeViewKind::texture_srv);
        string x = reg(256 + addr), y = reg(256 + (((i.a >> 1) & 3) ? (i.image_address_words[0] & 255) : addr + 1));
        if (i.b & 0x40000000u) {
            if (lod || gather || (store && !half_store) || (half_store && ((i.a >> 1) & 3)))
                refuse(i.pc, "packed sample address");
            y = "(" + x + ">>16)";
            x = "(" + x + "&65535u)";
        }
        if (store) {
            if (half_store) {
                body << "if(active) " << tex << "[uint2(" << x << "," << y << ")]=f16tof32("
                     << reg(256+i.dst) << "&65535u);\n";
                return;
            }
            if (mask != 15 || i.dst > 252)
                refuse(i.pc, "unsupported image store mask");
            body << "if(active) " << tex << "[uint2(" << x << "," << y << ")]=asfloat(uint4(";
            for (unsigned k = 0; k < 4; ++k)
                body << (k ? "," : "") << reg(256 + i.dst + k);
            body << "));\n";
            return;
        }
        if (lod || gather) {
            if (gather && mask != 1)
                refuse(i.pc, "unsupported gather channel");
            const auto sampler = sampler_index(i, state, lod);
            if (sampler.heap)
                refuse(i.pc, "sampler heap reads need the direct DXIL path");
            body << "t=" << tex << (lod ? ".SampleLevel(" : ".GatherRed(") << "q" << sampler.index
                 << ",float2(asfloat(" << x << "),asfloat(" << y << "))" << (lod ? ",0);\n" : ");\n");
        } else
            body << "t=" << tex << ".Load(int3(" << x << "," << y << ",0));\n";
        unsigned dst = i.dst;
        for (unsigned k = 0; k < 4; ++k)
            if (gather || ((mask >> k) & 1)) {
                if (dst >= 256)
                    refuse(i.pc, "image destination overflow");
                body << "if(active) " << reg(256 + dst++) << "=asuint(t." << "xyzw"[k] << ");\n";
            }
    }
    void require_numeric(const Inst &i, unsigned reg) const {
        if (reg <= 105 && !program.before[program.at.at(i.pc)][reg].numeric())
            refuse(i.pc, "unbound numeric SGPR s" + to_string(reg));
    }
    void scalar(const Inst &i) {
        auto n = i.name;
        string a = operand(i, i.src[0]), b = operand(i, i.src[1]), d = reg(i.dst), e;
        if (n == "s_mov_b32" || n == "s_mov_b64") {
            if (n == "s_mov_b64") {
                if (i.dst == 126)
                    require_numeric(i, i.src[0]);
                if ((i.dst & 1) || i.dst + 1 >= 128 || (i.src[0] < 128 && (i.src[0] & 1)))
                    refuse(i.pc, "scalar pair overflow");
                body << reg(i.dst + 1) << "="
                     << (i.src[0] < 105                       ? reg(i.src[0] + 1)
                         : i.src[0] >= 193 && i.src[0] <= 208 ? "0xffffffffu"
                                                              : "0u")
                     << ";\n";
            }
            if (i.dst == 106 || i.dst == 126)
                require_numeric(i, i.src[0]);
            body << d << "=" << a << ";\n";
            if (i.dst == 126)
                body << "active=(s126&laneBit)!=0;\n";
            return;
        }
        if (n == "s_bfe_u32" && i.src[0] <= 105) {
            const auto &state = program.before[program.at.at(i.pc)];
            const auto word = state[i.src[0]];
            const auto bits = source(i, i.src[1], state);
            if (word.type == Symbol::Type::indexed_descriptor && word.value == compute_resource_heap &&
                (word.offset & 31) == 4 && bits.type == Symbol::Type::constant && bits.value == 0xe0010) {
                auto &bindings = result.bindings;
                auto it = std::find_if(bindings.begin(), bindings.end(), [](const auto &b) {
                    return b.kind == ComputeViewKind::heap_stride_srv;
                });
                if (it == bindings.end()) {
                    bindings.push_back({ComputeViewKind::heap_stride_srv, compute_resource_heap, 0, srv++, 4});
                    it = std::prev(bindings.end());
                }
                body << d << "=r" << it->shader_register << ".Load((" << a << "+"
                     << word.offset / 32 << "u)*4u)&16383u; scc=(" << d << "!=0);\n";
                return;
            }
        }
        if (i.enc == Encoding::Sopk && n != "s_movk_i32")
            require_numeric(i, i.dst);
        if (i.enc != Encoding::Sopk)
            require_numeric(i, i.src[0]);
        if (i.enc == Encoding::Sop2 || i.enc == Encoding::Sopc)
            require_numeric(i, i.src[1]);
        if (n == "s_movk_i32") {
            body << d << "=" << hexword(Word(int16_t(i.a))) << ";\n";
            return;
        }
        if (n == "s_and_saveexec_b32") {
            body << "{uint old=s126;s126&=" << a << ";" << d << "=old;active=(s126&laneBit)!=0;scc=s126!=0;}\n";
            return;
        }
        if (n == "s_andn2_wrexec_b32") {
            body << "s126=" << a << "&~s126;" << d << "=s126;active=(s126&laneBit)!=0;scc=s126!=0;\n";
            return;
        }
        if (n == "s_cmp_eq_i32" || n == "s_cmp_eq_u32")
            e = a + "==" + b;
        else if (n == "s_cmp_lt_u32")
            e = a + "<" + b;
        else if (n == "s_cmp_lg_i32" || n == "s_cmp_lg_u32")
            e = a + "!=" + b;
        else if (n == "s_cmp_lt_i32")
            e = "asint(" + a + ")<asint(" + b + ")";
        else if (n == "s_bitcmp1_b32")
            e = "((" + a + ">>(" + b + "&31))&1)!=0";
        if (!e.empty()) {
            body << "scc=" << e << ";\n";
            return;
        }
        if (n == "s_mulk_i32") {
            body << d << "*=" << hexword(Word(int16_t(i.a))) << ";\n";
            return;
        }
        if (n == "s_cmpk_lg_i32") {
            body << "scc=" << d << "!=" << hexword(Word(int16_t(i.a))) << ";\n";
            return;
        }
        if (n == "s_min_u32" || n == "s_max_u32") {
            body << "{uint lhs=" << a << ",rhs=" << b << ";scc=lhs" << (n == "s_min_u32" ? "<" : ">") << "rhs;" << d
                 << "=scc?lhs:rhs;}\n";
            return;
        }
        if (n == "s_sub_u32") {
            body << "{uint lhs=" << a << ",rhs=" << b << ";scc=lhs<rhs;" << d << "=lhs-rhs;}\n";
            return;
        }
        if (n == "s_mul_hi_u32") {
            body << d << "=uint((uint64_t(" << a << ")*uint64_t(" << b << "))>>32);\n";
            return;
        }
        if (n == "s_mul_i32") {
            body << d << "=" << a << "*" << b << ";\n";
            return;
        }
        if (n == "s_add_u32") {
            body << "{uint lhs=" << a << ", sum=lhs+" << b << ";scc=sum<lhs;" << d << "=sum;}\n";
            return;
        }
        if (n == "s_and_b32")
            e = a + "&" + b;
        else if (n == "s_andn2_b32")
            e = a + "&~" + b;
        else if (n == "s_or_b32")
            e = a + "|" + b;
        else if (n == "s_nor_b32")
            e = "~(" + a + "|" + b + ")";
        else if (n == "s_not_b32")
            e = "~" + a;
        else if (n == "s_lshr_b32")
            e = a + ">>(" + b + "&31)";
        else if (n == "s_lshl_b32")
            e = a + "<<(" + b + "&31)";
        else if (n == "s_bfe_u32")
            e = "extractBits(" + a + "," + b + ",(" + b + ">>16)&127u)";
        else if (n == "s_bfm_b32")
            e = "((1u<<(" + a + "&31))-1u)<<(" + b + "&31)";
        else if (n == "s_absdiff_i32")
            e = "uint(abs(asint(" + a + ")-asint(" + b + ")))";
        else if (n == "s_cselect_b32") {
            body << d << "=scc?" << a << ":" << b << ";\n";
            return;
        } else if (i.enc == Encoding::Sop2 && i.op >= 0x2e && i.op <= 0x31) {
            const unsigned shift = i.op - 0x2d;
            body << "scc=(" << a << ">>" << (32 - shift) << ")!=0||((" << a << "<<" << shift << ")+" << b << ")<" << b
                 << ";\n"
                 << d << "=(" << a << "<<" << shift << ")+" << b << ";\n";
            return;
        } else
            refuse(i.pc, string(n));
        body << d << "=" << e << ";scc=" << d << "!=0;\n";
    }
    void vector(const Inst &i) {
        auto n = i.name;
        if ((n == "v_readfirstlane_b32" || n == "v_readlane_b32") && i.dst >= 106)
            refuse(i.pc, "invalid scalar lane read destination");
        if (n == "v_readlane_b32")
            refuse(i.pc, "unsupported HLSL instruction v_readlane_b32");
        auto selected = [&](unsigned j) {
            require_numeric(i, i.src[j]);
            auto s = operand(i, i.src[j]);
            if (i.subword && j < 2 && i.src_select[j] < 6) {
                unsigned sel = i.src_select[j];
                unsigned shift = sel < 4 ? sel * 8 : (sel - 4) * 16;
                s = "((" + s + ">>" + to_string(shift) + ")&" + to_string(sel < 4 ? 255 : 65535) + "u)";
                if (i.src_sext[j]) {
                    const auto sign = sel < 4 ? "128u" : "32768u";
                    s = "((" + s + "^" + sign + ")-" + sign + ")";
                }
            }
            return s;
        };
        auto val = [&](unsigned j, bool fp) {
            auto s = selected(j);
            if (fp) {
                s = "asfloat(" + s + ")";
                if ((i.abs >> j) & 1)
                    s = "abs(" + s + ")";
                if ((i.neg >> j) & 1)
                    s = "(-" + s + ")";
            } else if (((i.abs | i.neg) >> j) & 1) {
                if (n != "v_fma_mix_f32")
                    refuse(i.pc, "unexpected integer modifier");
            }
            return s;
        };
        auto d = reg(i.dst);
        if (n == "v_cos_f32" || n == "v_cvt_f32_f16") {
            if (i.wide)
                refuse(i.pc, "unsupported wide conversion or cosine modifiers");
            if (n == "v_cvt_f32_f16")
                body << "if(active) " << d << "=asuint(f16tof32(" << selected(0) << "&65535u));\n";
            else {
                body << "if(active){float turns=" << val(0,true)
                     << ";float phase=turns-round(turns);" << d
                     << "=asuint(abs(phase)==0.25f?0.0f:cos(phase*6.283185307f));}\n";
            }
            return;
        }
        if (n == "v_swap_b32") {
            if (i.wide || i.src[0] < 256 || i.src[0] > 511)
                refuse(i.pc, "swap requires two VGPRs in a VOP1 encoding");
            const auto src = reg(i.src[0]);
            body << "if(active){uint old=" << d << ";" << d << "=" << src << ";" << src << "=old;}\n";
            return;
        }
        if (n == "v_readfirstlane_b32") {
            if (i.abs || i.neg || i.scale || i.clamp)
                refuse(i.pc, "readfirstlane modifiers");
            body << d << "=WaveReadLaneAt(" << val(0, false) << ",s126==0?0u:uint(firstbitlow(s126)));\n";
            return;
        }
        if (i.enc == Encoding::Vopc) {
            bool fp = n.find("f32") != n.npos;
            auto a = val(0, fp), b = val(1, fp);
            string pred;
            if (!fp && n.find("_u32") == n.npos) {
                a = "asint(" + a + ")";
                b = "asint(" + b + ")";
            }
            if (n == "v_cmp_class_f32") {
                // Apply floating-point abs/neg to the raw sign bit, including for signed zero and NaNs.
                auto bits = selected(0);
                if (i.abs & 1) bits = "(" + bits + "&0x7fffffffu)";
                if (i.neg & 1) bits = "(" + bits + "^0x80000000u)";
                if ((i.abs | i.neg) & 2) refuse(i.pc, "unexpected class mask modifier");
                pred = "(fpClass(" + bits + ")&" + selected(1) + ")!=0u";
            }
            else if (n.find("_nle_") != n.npos)
                pred = "!(" + a + "<=" + b + ")";
            else {
                string op = n.find("_lt_") != n.npos   ? "<"
                            : n.find("_le_") != n.npos ? "<="
                            : n.find("_gt_") != n.npos ? ">"
                            : n.find("_ge_") != n.npos ? ">="
                            : n.find("_eq_") != n.npos ? "=="
                                                       : "!=";
                pred = "(" + a + op + b + ")";
            }
            body << d << "=WaveActiveBallot((uint(active)&uint(" << pred << "))!=0u).x;\n";
            if (i.dst == 126)
                body << "active=(s126&laneBit)!=0;\n";
            return;
        }
        if (n == "v_cndmask_b32") {
            require_numeric(i, i.src[2]);
            auto a = val(0, true), b = val(1, true);
            body << "if(active) " << d << "=((" << operand(i, i.src[2]) << "&laneBit)!=0)?asuint(" << b << "):asuint("
                 << a << ");\n";
            return;
        }
        if (n == "v_lshrrev_b64") {
            auto src = i.src[1];
            if (src == 128 && i.dst < 511) {
                body << "if(active){" << d << "=0;" << reg(i.dst + 1) << "=0;}\n";
                return;
            }
            if (src < 256 || src >= 511 || i.dst >= 511)
                refuse(i.pc, "invalid shift pair");
            body << "if(active){uint64_t w=(uint64_t(" << reg(src + 1) << ")<<32)|" << reg(src) << "; w>>=("
                 << val(0, false) << "&63u);" << d << "=uint(w);" << reg(i.dst + 1) << "=uint(w>>32);}\n";
            return;
        }
        if (n == "v_fma_mix_f32") {
            std::array<string, 3> args;
            for (unsigned j = 0; j < 3; ++j) {
                require_numeric(i, i.src[j]);
                auto a = operand(i, i.src[j]);
                bool half = j == 2 ? (i.a & 0x4000) != 0 : ((i.b >> (27 + j)) & 1) != 0;
                args[j] = half ? "f16tof32((" + a + ">>" + to_string(((i.a >> (11 + j)) & 1) * 16) + ")&65535u)"
                               : "asfloat(" + a + ")";
                if ((i.abs >> j) & 1)
                    args[j] = "abs(" + args[j] + ")";
                if ((i.neg >> j) & 1)
                    args[j] = "-" + args[j];
            }
            body << "if(active) " << d << "=asuint(mad(" << args[0] << "," << args[1] << "," << args[2] << "));\n";
            return;
        }
        auto it = alu_rules.find(n);
        if (it == alu_rules.end())
            refuse(i.pc, string(n));
        const auto &rule = it->second;
        bool fp = rule.type == AluType::Float;
        std::array<string, 3> args;
        for (unsigned j = 0; j < rule.arity; ++j) {
            if (j == 2 && n == "v_fmac_f32")
                args[j] = "asfloat(" + d + ")";
            else if ((n == "v_fmamk_f32" && j == 1) || (n == "v_fmaak_f32" && j == 2))
                args[j] = "asfloat(" + hexword(i.literal) + ")";
            else
                args[j] = val(n == "v_fmamk_f32" && j == 2 ? 1 : j, fp);
        }
        string e;
        for (size_t k = 0; k < rule.expression.size(); ++k) {
            if (rule.expression[k] == '{') {
                e += args[rule.expression[k + 1] - '0'];
                k += 2;
            } else
                e += rule.expression[k];
        }
        if (i.scale)
            e = "(" + e + ")*" + (i.scale == 1 ? "2.0" : i.scale == 2 ? "4.0" : "0.5");
        if (i.clamp)
            e = "saturate(" + e + ")";
        if (i.dst_select < 6) {
            const unsigned width = i.dst_select < 4 ? 8 : 16;
            const unsigned shift = width == 8 ? i.dst_select * 8 : (i.dst_select - 4) * 16;
            const Word mask = (1u << width) - 1u;
            e = "((" + e + ")&" + hexword(mask) + ")";
            if (i.dst_unused == 1) {
                const auto sign = hexword(1u << (width - 1));
                e = "((" + e + "^" + sign + ")-" + sign + ")";
            }
            e = "(" + e + "<<" + to_string(shift) + ")";
            if (i.dst_unused == 2)
                e += "|(" + d + "&" + hexword(~(mask << shift)) + ")";
        }
        body << "if(active) " << d << "=" << (rule.type != AluType::Bits ? "asuint(" + e + ")" : e) << ";\n";
    }
    void instruction(size_t index) {
        const auto &i = program.code[index];
        const auto &s = program.before[index];
        if (i.enc == Encoding::Mubuf) {
            if (i.name == "buffer_store_format" || i.name == "buffer_store_raw") {
                const bool raw = i.name == "buffer_store_raw";
                const bool half = (i.op & 0x80) != 0;
                constexpr unsigned raw_widths[]{1, 2, 4, 3};
                const unsigned count = raw ? raw_widths[i.op & 3] : (i.op & 3) + 1;
                if (i.dst + (half ? (count + 1) / 2 : count) > 256)
                    refuse(i.pc, "buffer store data register overflow");
                const auto kind = raw ? ComputeViewKind::raw_buffer_uav :
                                  half ? ComputeViewKind::half_buffer_uav : ComputeViewKind::typed_buffer_uav;
                auto view = binding(i, s, ((i.b >> 16) & 31) * 4, 4, kind, raw ? 4 : count * (half ? 2 : 4));
                auto address = reg(256 + (i.b & 255));
                if (raw) {
                    const unsigned scalar = i.b >> 24;
                    if (scalar > 105 && (scalar < 128 || scalar > 192))
                        refuse(i.pc, "unsupported raw store scalar offset");
                    require_numeric(i, scalar);
                    address = "((" + ((i.a & 0x1000) ? address : "0u") + "+" + operand(i, scalar) +
                              "+" + to_string(i.a & 4095) + "u)&~3u)";
                }
                const string suffix = count == 1 ? "" : to_string(count);
                string data;
                for (unsigned j = 0; j < count; ++j) {
                    if (j) data += ",";
                    auto src = reg(256 + i.dst + (half ? j / 2 : j));
                    data += half ? "f16tof32((" + src + ">>" + to_string((j%2)*16) + ")&65535u)" : src;
                }
                if (count > 1) data = (half ? "float" : "uint") + suffix + "(" + data + ")";
                body << "if(active) " << view;
                if (raw) body << ".Store" << suffix << "(" << address << "," << data << ");\n";
                else body << "[" << address << "]=" << data << ";\n";
                return;
            }
            if (i.name == "buffer_load_dword") {
                constexpr unsigned widths[] = {1, 2, 4, 3};
                const unsigned count = widths[i.op - 0x0c];
                if (i.dst + count > 256)
                    refuse(i.pc, "raw load destination overflow");
                auto r = binding(i, s, ((i.b >> 16) & 31) * 4, 4, ComputeViewKind::raw_vector_srv, 4);
                // Address generation precedes every writeback, including when vaddr is a destination.
                body << "if(active){uint address=(" << reg(256 + (i.b & 255)) << "+" << (i.a & 4095)
                     << "u)&~3u;";
                const string suffix = count == 1 ? "" : to_string(count);
                body << "uint" << suffix << " loaded=" << r << ".Load" << suffix << "(address);";
                for (unsigned k = 0; k < count; ++k) {
                    body << reg(256 + i.dst + k) << "=loaded";
                    if (count > 1) body << "." << "xyzw"[k];
                    body << ";";
                }
                body << "}\n";
                return;
            }
            const auto buffer =
                binding(i, s, ((i.b >> 16) & 31) * 4, 4, ComputeViewKind::typed_buffer_srv, 4 * (1 + i.op));
            if (i.dst + i.op >= 256)
                refuse(i.pc, "formatted load destination overflow");
            body << "if(active){uint4 loaded=" << buffer << ".Load(" << reg(256 + (i.b & 255)) << ");";
            for (unsigned j = 0; j <= i.op; ++j)
                body << reg(256 + i.dst + j) << "=loaded[" << j << "];";
            body << "}\n";
            return;
        }
        if (i.enc == Encoding::Image) {
            image(i, s);
            return;
        }
        if (i.enc == Encoding::Smem) {
            if (i.op < 8) {
                if (s.at(i.src[0]).type != Symbol::Type::table)
                    refuse(i.pc, "unknown scalar table base");
                if (s.at(i.src[0]).value == compute_sampler_heap)
                    refuse(i.pc, "sampler heap reads need the direct DXIL path");
                if (s.at(i.src[0]).value == compute_resource_heap && i.b >> 25 == 125)
                    refuse(i.pc, "unsupported immediate whole-heap descriptor");
                if (i.b >> 25 != 125) {
                    const auto high = s.at(i.src[0]+1);
                    if (high.type != Symbol::Type::constant || high.value || ((i.b & 0xfffff) % 32))
                        refuse(i.pc,"unproven indexed table address");
                    require_numeric(i,i.b>>25);
                    if (s.at(i.src[0]).value == compute_resource_heap) {
                        const auto offset = s.at(i.b >> 25);
                        if ((offset.type != Symbol::Type::masked && offset.type != Symbol::Type::constant) ||
                            (offset.value & 31))
                            refuse(i.pc, "whole-heap descriptor offset is not proven 32-byte aligned");
                    }
                    body << "{uint slot=" << reg(i.b>>25) << ">>5;";
                    for (unsigned j=0;j<4;++j) body << reg(i.dst+j) << "=slot;";
                    body << "}\n";
                }
                return;
            }
            unsigned off = i.b & 0xfffff, count = memory_count(i);
            const unsigned offset_reg = i.b >> 25;
            const bool dynamic = offset_reg != 125;
            if (dynamic)
                require_numeric(i, offset_reg);
            auto r = binding(i, s, i.src[0], 4, ComputeViewKind::buffer_srv, dynamic ? 4 : off + count * 4);
            // The offset can alias any destination SGPR. Snapshot it before committing the first word.
            if (dynamic)
                body << "{uint address=(" << reg(offset_reg) << "+" << off << "u)&~3u;\n";
            for (unsigned j = 0; j < count; ++j)
                body << reg(i.dst + j) << "=" << r << ".Load(" << (dynamic ? "address+" : "")
                     << (dynamic ? 0 : off) + 4 * j << "u);\n";
            if (dynamic)
                body << "}\n";
            return;
        }
        if (i.enc == Encoding::Lds) {
            std::ostringstream line;
            line << "#pragma x2w_lds " << std::hex << std::setfill('0') << std::setw(8) << i.a << " " << std::setw(8)
                 << i.b;
            body << "{uint exec=uint(active);" << expand_gfx1013_compute(line.str()) << "}\n";
            return;
        }
        if (i.enc == Encoding::Sopp) {
            if (i.name == "s_barrier")
                body << "GroupMemoryBarrierWithGroupSync();\n";
            return;
        }
        if (i.enc == Encoding::Sop1 || i.enc == Encoding::Sop2 || i.enc == Encoding::Sopk || i.enc == Encoding::Sopc)
            scalar(i);
        else
            vector(i);
    }
};
string branch_test(const Inst &i) {
    if (i.op == 2)
        return "true";
    if (i.op == 4)
        return "!scc";
    if (i.op == 5)
        return "scc";
    if (i.op == 6 || i.op == 7)
        return i.op == 6 ? "s106==0" : "s106!=0";
    return i.op == 8 ? "s126==0" : "s126!=0";
}
void lower_blocks(Emit &e) {
    const auto &p = e.program;
    std::set<unsigned> starts{p.code.front().pc};
    for (const auto &i : p.code)
        if (i.branch()) {
            starts.insert(i.target());
            starts.insert(i.next);
        }
    std::map<unsigned, unsigned> ends;
    for (const auto &i : p.code)
        if (i.branch() && i.target() < i.pc) {
            if (*std::prev(starts.upper_bound(i.pc)) != i.target() || i.op == 2)
                refuse(i.pc, "unsupported loop shape");
            ends[i.target()] = i.pc;
        }
    auto &out = e.body;
    out << "uint pc=" << p.code.front().pc << "u;\n";
    for (size_t k = 0; k < p.code.size(); ++k) {
        auto &i = p.code[k];
        if (starts.count(i.pc)) {
            out << "if(pc==" << i.pc << "u){\n";
            if (ends.count(i.pc))
                out << "[loop] do {\n";
        }
        if (i.branch()) {
            if (i.target() < i.pc)
                out << "}while(" << branch_test(i) << "); pc=" << i.next << "u;\n";
            else
                out << "pc=(" << branch_test(i) << ")?" << i.target() << "u:" << i.next << "u;\n";
        } else {
            if (i.name != "v_nop") e.instruction(k);
            if (starts.count(i.next))
                out << "pc=" << i.next << "u;\n";
        }
        if (k + 1 == p.code.size() || starts.count(i.next))
            out << "}\n";
    }
}
#include "gfx1013_direct_compute.inl"
#include "gfx1013_direct_graphics.inl"
#include "gfx1013_direct_mesh.inl"
} // namespace
static NativeComputeShader translate_program(const ComputeRequest &request, unsigned entry, bool direct_dxil, const GuestShaderStage *stage,
                                             HeapViewContract heap_view) {
    if (entry == 36 && ((word(request.code, 0) >> 16) != 0xd580 || (word(request.code, 12) >> 16) != 0xd580 ||
                        (word(request.code, 24) >> 16) != 0xd580 || (word(request.code, 36) >> 16) != 0xbfa0))
        refuse(entry, "unrecognized compute header");
    if(direct_dxil && entry==36){
        for(unsigned at=0;at<entry;){
            auto prefix=gfx1013::instruction(request.code,at);
            if(prefix.name!="v_nop" || prefix.size>entry-at)refuse(at,"executable instruction in capture prefix");
            at+=prefix.size;
        }
    }
    unsigned xy = stage ? 0 : word(request.sa, 12);
    unsigned z = stage ? stage->threads[2] : word(request.sa, 16) & 65535;
    unsigned rsrc2 = stage ? stage->rsrc2 : word(request.sa, 24);
    unsigned lds = (rsrc2 >> 15) & 511;
    unsigned x = stage ? stage->threads[0] : xy & 65535;
    unsigned y = stage ? stage->threads[1] : xy >> 16;
    if (x > 1024 || y > 1024 || z > 64) refuse(entry, "compute dimension exceeds desktop limit");
    const uint64_t threads = uint64_t(x) * y * z;
    unsigned local_axes = 1 + ((rsrc2 >> 11) & 3);
    unsigned user = (rsrc2 >> 1) & 31;
    if (!threads || threads > 1024 || z > 64 || local_axes > 3)
        refuse(entry, "invalid compute group shape");
    if (rsrc2 & ~0xff9fbeu || lds > 64)
        refuse(entry, "unsupported compute resource flags");
    if (!direct_dxil && threads > 32 && threads % 32)
        refuse(entry, "partial final wave in multi-wave group");
    Microsoft::WRL::ComPtr<ID3D12VersionedRootSignatureDeserializer> sig;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC empty{};
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC *layout = &empty;
    if (!stage || !request.root.empty()) {
        HRESULT hr = D3D12CreateVersionedRootSignatureDeserializer(request.root.data(), request.root.size(),
                                                                   IID_PPV_ARGS(&sig));
        if (SUCCEEDED(hr)) hr = sig->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &layout);
        if (hr == E_INVALIDARG) refuse(entry, "invalid root layout");
        if (FAILED(hr)) throw std::runtime_error("invalid root layout: deserializer HRESULT " + std::to_string(hr));
        if (layout->Desc_1_1.NumParameters > 16) refuse(entry, "invalid root layout");
    }
    unsigned next = 4;
    State initial{};
    for (unsigned k = 0; k <= 105; ++k)
        initial[k].type = Symbol::Type::unbound;
    std::vector<ComputeConstants> constants;
    if (stage) {
        std::array<bool,32> assigned{};
        for (const auto &input : stage->inputs) {
            if (input.type == 1) {
                bind_root_address(initial, input, user);
                assigned[input.user_data] = assigned[input.user_data + 1] = true;
                continue;
            }
            if (input.type != 0 && input.type != 3 && input.type != 5) continue;
            unsigned at = input.user_data;
            unsigned count = input.type == 0 ? input.sgpr_count : 1;
            if (!count || at >= user || count > user - at || (input.type != 5 && input.root_parameter >= 16))
                refuse(entry, "invalid FSSM user-data range");
            if (input.type == 0) {
                constants.push_back({input.root_parameter, at, count});
                for (unsigned j = at; j < at + count; ++j) {
                    if (assigned[j]) refuse(entry, "overlapping FSSM constants");
                    assigned[j] = true;
                    initial[j] = {};
                }
            } else {
                Symbol table{Symbol::Type::table, input.type == 5 ? compute_resource_heap : input.root_parameter};
                if (assigned[at] && !(initial[at] == table)) refuse(entry, "ambiguous FSSM table");
                initial[at] = table;
                assigned[at] = true;
            }
        }
    }
    unsigned system = user;
    for (unsigned flag : {128u, 256u, 512u, 1024u})
        if (rsrc2 & flag)
            initial[system++] = {};
    for (unsigned i = 0; !stage && i < layout->Desc_1_1.NumParameters; ++i) {
        const auto &param = layout->Desc_1_1.pParameters[i];
        if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
            if (next >= user)
                refuse(entry, "descriptor table exceeds user SGPRs");
            initial[next++] = {Symbol::Type::table, i};
        } else if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
            unsigned count = param.Constants.Num32BitValues;
            if (next > user || count > user - next)
                refuse(entry, "root constants exceed user SGPRs");
            constants.push_back({i, next, count});
            for (unsigned j = 0; j < count; ++j)
                initial[next + j] = {};
            next += count;
        } else if (direct_dxil && (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_CBV ||
                                   param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_SRV ||
                                   param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_UAV)) {
            // Same packed entry layout as gfx2hlsl.Stage.setup: a root address
            // occupies two consecutive SGPRs, even when only later parameters are used.
            if (next > user || user - next < 2)
                refuse(entry, "root descriptor pair exceeds user SGPRs");
            bind_root_address(initial, {1, 0, 2, 0, static_cast<uint8_t>(next), static_cast<uint8_t>(i)}, user);
            next += 2;
        } else
            refuse(entry, "root-address parameter");
    }
    // In this compute ABI the two directly indexed heaps follow the user root parameters.
    // Require both flags and exactly two remaining user SGPRs, as in the matched native layouts.
    if (!stage && (UINT(layout->Desc_1_1.Flags) & 0xc00u) == 0xc00u && next + 2 == user) {
        initial[next] = {Symbol::Type::table, compute_resource_heap};
        initial[next + 1] = {Symbol::Type::table, compute_sampler_heap};
    }
    // Tagged slot-67 CS state packs group Z into the low 16 bits of word 4
    // and wave32 into bit 20. Leave untagged synthetic request ABIs alone.
    const bool packed_state = !stage && direct_dxil && word(request.sa,0)==0x4005cf00 &&
                              word(request.sa,4)==0x6c090102;
    const bool wave64 = stage ? !stage->wave32 : packed_state && !(word(request.sa,16)&(1u<<20));
    // The byte-address rewrite of a GPU-selected heap descriptor is read only under the raw_buffer contract.
    const bool heap_window = direct_dxil && heap_view == HeapViewContract::raw_buffer;
    auto program = parse(request.code, entry, initial, direct_dxil, false, wave64, false, heap_window);
    Emit emit{{}, program,layout->Desc_1_1};
    emit.result.constants = constants;
    if (!lds)
        for (const auto &inst : program.code)
            if (inst.enc == Encoding::Lds)
                refuse(inst.pc, "LDS instruction without allocation");
    if (direct_dxil) {
        DirectCompute compiler{emit};
        compiler.wave64 = wave64;
        compiler.heap_view = heap_view;
        compiler.heap_window = heap_window;
        if (threads % (wave64 ? 64 : 32))
            for (const auto &i : program.code)
                if (i.name == "v_readlane_b32" || i.name == "v_writelane_b32")
                    DirectCompute::validate_lane_shape(i, static_cast<unsigned>(threads), wave64 ? 64 : 32);
        if (threads > 32 && threads % 32)
            refuse(entry, "partial final wave in multi-wave group");
        if (wave64 && threads % 64) refuse(entry, "wave64 exchange requires complete guest waves");
        if (wave64 && threads > 64) {
            const auto &code = program.code;
            const bool branches = std::any_of(code.begin(), code.end(),
                [](const Inst &i) { return i.branch() && i.op != 2; });
            const bool synchronizes = std::any_of(code.begin(), code.end(), [](const Inst &i) {
                return i.enc == Encoding::Vopc || i.enc == Encoding::Lds || carry_destination(i) ||
                       i.name == "v_readfirstlane_b32" || i.name == "v_readlane_b32" || i.name == "s_barrier";
            });
            if (branches && synchronizes) {
                if (!packed_state) refuse(entry, "multi-wave64 exchange requires converged group control flow");
                compiler.virtual_waves = static_cast<unsigned>(threads / 64);
            }
        }
        auto translated = compiler.lower(entry,user,rsrc2,x,y,z);
        for (auto &binding : translated.bindings) {
            if (stage) apply_reflected_table_use(binding, *stage);
            if (!request.root.empty()) {
                const auto &desc = layout->Desc_1_1;
                if (binding.kind == ComputeViewKind::buffer_srv && binding.root_parameter < desc.NumParameters) {
                    const auto &param = desc.pParameters[binding.root_parameter];
                    if (param.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
                        apply_scalar_table_layout(binding, param.DescriptorTable);
                }
                if (binding.kind != ComputeViewKind::root_buffer_srv) continue;
                if (binding.root_parameter >= desc.NumParameters)
                    refuse(entry, "missing compute root descriptor");
                const auto &param = desc.pParameters[binding.root_parameter];
                if ((param.ParameterType != D3D12_ROOT_PARAMETER_TYPE_CBV &&
                     param.ParameterType != D3D12_ROOT_PARAMETER_TYPE_SRV) ||
                    param.ShaderVisibility != D3D12_SHADER_VISIBILITY_ALL)
                    refuse(entry, "compute root descriptor type or visibility");
            }
        }
        translated.group = {x, y, z};
        return translated;
    }
    lower_blocks(emit);
    std::ostringstream h;
    h << R"(
uint extractBits(uint v,uint offset,uint width){return width>=32?v>>(offset&31u):(v>>(offset&31u))&((1u<<width)-1u);}
float med3(float a,float b,float c){return max(min(a,b),min(max(a,b),c));}
uint fpClass(uint v){
    uint a=v&0x7fffffffu;bool neg=(v>>31)!=0;
    if(a>0x7f800000u)return (a&0x400000u)?2u:1u;
    if(a==0x7f800000u)return neg?4u:512u;
    if(a>=0x800000u)return neg?8u:256u;
    if(a!=0)return neg?16u:128u;
    return neg?32u:64u;
}
)";
    for (auto &b : emit.result.bindings) {
        auto r = b.shader_register;
        if (b.array_position) continue;
        if (b.kind == ComputeViewKind::raw_buffer_uav)
            h << "RWByteAddressBuffer w" << r << ":register(u" << r << ");\n";
        else if (compute_buffer_uav(b.kind)) {
            const bool half = b.kind == ComputeViewKind::half_buffer_uav;
            const unsigned n = b.required_bytes / (half ? 2 : 4);
            h << "RWBuffer<" << (half ? "float" : "uint") << (n == 1 ? "" : to_string(n)) << "> w"
              << r << ":register(u" << r << ");\n";
        } else if (b.kind == ComputeViewKind::half_scalar_uav)
            h << "RWTexture2D<float> w" << r << ":register(u" << r << ");\n";
        else if (b.kind == ComputeViewKind::texture_uav)
            h << "RWTexture2D<float4> w" << r << ":register(u" << r << ");\n";
        else
            h << (b.kind == ComputeViewKind::heap_stride_srv || b.kind == ComputeViewKind::buffer_srv || b.kind == ComputeViewKind::raw_vector_srv ? "ByteAddressBuffer"
                  : b.kind == ComputeViewKind::typed_buffer_srv ? "Buffer<uint4>"
                                                                : "Texture2D<float4>")
              << " r" << r << (b.array_count>1 ? "["+to_string(b.array_count)+"]" : "") << ":register(t" << r << ");\n";
    }
    for (size_t j = 0; j < emit.result.samplers.size(); ++j)
        h << "SamplerState q" << j << ":register(s" << j << ");\n";
    unsigned constant_words = 0;
    for (const auto &block : constants)
        constant_words += block.count;
    if (constant_words)
        h << "cbuffer EntryConstants:register(b0){uint4 rootWords[" << (constant_words + 3) / 4 << "];};\n";
    if (lds)
        h << "groupshared uint lds[" << lds * 128 << "];\n";
    if (threads > 64)
        h << "groupshared uint waveStarts[" << threads << "];\n";
    h << "[WaveSize(32)] [numthreads(" << x << "," << y << "," << z
      << ")] void main(uint3 g:SV_GroupID,uint idx:SV_GroupIndex){\n";
    for (unsigned j = 0; j < 128; ++j)
        h << "uint s" << j << "=0;\n";
    for (unsigned j = 0; j < 256; ++j)
        h << "uint v" << j << "=0;\n";
    if (threads <= 32)
        h << "uint wave=0u;\n";
    else if (threads == 64)
        h << "uint wave=WaveActiveMin(idx)==0u?0u:1u;\n";
    else {
        // Order host waves by their lowest group index. Physical lane order need not be row-major.
        h << "uint first=WaveActiveMin(idx);waveStarts[idx]=uint(idx==first);GroupMemoryBarrierWithGroupSync();"
             "uint wave=0;for(uint p=0;p<first;++p)wave+=waveStarts[p];\n";
    }
    h << "uint lane=WaveGetLaneIndex();uint guestIndex=wave*32u+lane;uint laneBit=1u<<lane;"
         "bool active=true,scc=false;float4 t=0;s126=WaveActiveBallot(true).x;v0=guestIndex%"
      << x << "u;\n";
    if (local_axes >= 2)
        h << "v1=(guestIndex/" << x << "u)%" << y << "u;\n";
    if (local_axes == 3)
        h << "v2=guestIndex/" << x * y << "u;\n";
    unsigned sys = user;
    for (unsigned axis = 0; axis < 3; ++axis)
        if (rsrc2 & (1u << (7 + axis)))
            h << "s" << sys++ << "=g." << "xyz"[axis] << ";\n";
    if (rsrc2 & 1024)
        h << "s" << sys << "=" << (threads + 31) / 32 << "u|(wave<<20)|(wave==0u?0x80000000u:0u);\n";
    unsigned offset = 0;
    for (const auto &block : constants)
        for (unsigned j = 0; j < block.count; ++j, ++offset)
            h << "s" << block.first_sgpr + j << "=rootWords[" << offset / 4 << "][" << offset % 4 << "];\n";
    h << emit.body.str() << "}\n";
    emit.result.hlsl = h.str();
    emit.result.instruction_count = static_cast<unsigned>(program.code.size());
    emit.result.group = {x, y, z};
    return std::move(emit.result);
}

// Native FSSM UsedResource and RootParam use dword table offsets and an
// explicit user SGPR. See victory-shader native.rs and victory-gpu draw.rs.
bool x2w::apply_reflected_table_use(ComputeViewBinding &binding, const GuestShaderStage &stage,
                                    std::optional<unsigned> packed_sgpr) {
    if (binding.array_count != 1 || binding.kind == ComputeViewKind::heap_stride_srv ||
        binding.kind == ComputeViewKind::root_buffer_srv || binding.root_parameter == compute_resource_heap)
        return false;
    const ShaderResourceUse *match = nullptr;
    for (const auto &use : stage.resources) {
        if (use.root_type != 3 || use.root_parameter != binding.root_parameter ||
            uint64_t(use.offset_dwords) * 4 != binding.byte_offset) continue;
        if (packed_sgpr) {
            if (use.user_data != *packed_sgpr) continue;
        } else if (std::none_of(stage.inputs.begin(), stage.inputs.end(), [&](const auto &input) {
            return input.type == 3 && input.root_parameter == use.root_parameter && input.user_data == use.user_data;
        })) {
            throw UnsupportedComputeShader("reflected table record has no matching user SGPR");
        }
        if (match) throw UnsupportedComputeShader("ambiguous reflected table record");
        match = &use;
    }
    if (!match) return false;
    const bool scalar = accepts_cbv_window(binding.kind);
    const bool buffer = scalar || compute_buffer_uav(binding.kind) || binding.kind == ComputeViewKind::raw_vector_srv ||
                        binding.kind == ComputeViewKind::typed_buffer_srv || binding.kind == ComputeViewKind::byte_window_srv ||
                        binding.kind == ComputeViewKind::indexed_raw_srv;
    if (match->dwords != (buffer ? 4u : 8u) ||
        (match->type != (compute_uav(binding.kind) ? 3u : 0u) && !(scalar && match->type == 2)))
        throw UnsupportedComputeShader("raw load disagrees with reflected table record at root " + std::to_string(binding.root_parameter) + " offset " + std::to_string(binding.byte_offset) + " kind " + std::to_string(static_cast<unsigned>(binding.kind)) + " reflected type " + std::to_string(match->type) + " width " + std::to_string(match->dwords));
    if (scalar)
        binding.scalar_table = match->type == 2 ? ComputeViewBinding::ScalarTable::reflected_cbv
                                                : ComputeViewBinding::ScalarTable::srv;
    return true;
}

void x2w::apply_scalar_table_layout(ComputeViewBinding &binding, const D3D12_ROOT_DESCRIPTOR_TABLE1 &table) {
    using Table = ComputeViewBinding::ScalarTable;
    if (!accepts_cbv_window(binding.kind) || binding.scalar_table != Table::unknown || binding.byte_offset % 32)
        return;
    uint64_t end = 0;
    const auto slot = binding.byte_offset / 32;
    bool cbv = false, srv = false;
    for (unsigned i = 0; i < table.NumDescriptorRanges; ++i) {
        const auto &range = table.pDescriptorRanges[i];
        const uint64_t first = range.OffsetInDescriptorsFromTableStart == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
                                   ? end : range.OffsetInDescriptorsFromTableStart;
        end = first + range.NumDescriptors;
        if (first <= slot && slot < end) {
            cbv |= range.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            srv |= range.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        }
    }
    if (cbv != srv) binding.scalar_table = cbv ? Table::declared_cbv : Table::srv;
}

x2w::NativeComputeShader x2w::translate_compute(const ComputeRequest &request, unsigned entry, bool direct_dxil) {
    return translate_program(request, entry, direct_dxil, nullptr, HeapViewContract::none);
}

x2w::NativeComputeShader x2w::translate_compute_container(ComputeContainer container, HeapViewContract heap_view) {
    auto stages = read_guest_shader_container(container.bytes);
    GuestShaderStage *compute = nullptr;
    for (auto &stage : stages) {
        if (stage.stage != GpuStage::cs) continue;
        if (compute) refuse(0, "ambiguous native compute program");
        compute = &stage;
    }
    if (!compute) refuse(0, "expected one compute program");
    ComputeRequest request{};
    request.code = std::move(compute->code);
    request.root = std::move(container.root);
    return translate_program(request, 0, true, compute, heap_view);
}

x2w::NativeComputeShader x2w::translate_graphics_container(const std::string &bytes,bool pixel) {
    auto stages=read_guest_shader_container(bytes);
    GuestShaderStage *selected=nullptr;
    for(auto &stage:stages)if(stage.stage==(pixel?GpuStage::ps:GpuStage::vs)){
        if(selected)refuse(0,"ambiguous native graphics stage");
        selected=&stage;
    }
    if(!selected)refuse(0,"missing native graphics stage");
    return lower_graphics(*selected);
}

x2w::NativeComputeShader x2w::translate_mesh_program(const std::string &code, const MeshStageLayout &layout) {
    return lower_mesh(code,layout);
}

x2w::NativeComputeShader x2w::translate_mesh_program(const std::string &code, const MeshStageLayout &layout,
                                                      const D3D12_ROOT_SIGNATURE_DESC1 &root) {
    return lower_mesh(code,layout,nullptr,root);
}

x2w::NativeComputeShader x2w::translate_mesh_container(const std::string &bytes, bool validate_uint_heap_reads) {
    const auto stages=read_guest_shader_container(bytes);
    const GuestShaderStage *mesh=nullptr;
    for (const auto &s : stages) if (s.stage==GpuStage::ms) {
        if (mesh) refuse(0,"ambiguous mesh stage");
        mesh=&s;
    }
    if (!mesh) refuse(0,"missing mesh stage");
    MeshStageLayout layout{mesh->threads[0],mesh->max_vertices,mesh->max_primitives,
                           mesh->topology,mesh->mesh_group_stride,mesh->wave32};
    layout.validate_uint_heap_reads=validate_uint_heap_reads;
    return lower_mesh(mesh->code,layout,mesh);
}

x2w::DirectGraphicsPair x2w::translate_graphics_pair(std::string_view a,std::string_view b) {
    return translate_graphics_pair(a, b, {});
}

// Fixed-descriptor draws publish the bounded native stages in export order,
// without FSSM signatures. Link PARAM exports to interpolator reads using
// the captured entry layout and the channels actually named by each stage.
static bool raw_center_inputs(unsigned enable_mask) {
    // SPI_PS_INPUT_ADDR allocates the enabled X/Y/Z/W components after
    // the center I/J pair. The packed screen coordinate is a later VGPR.
    // victory-shader 9376fd6 io.rs::PS_VGPRS uses the same allocation.
    constexpr unsigned position_inputs = (0xfu << 8) | (1u << 15);
    return (enable_mask & ~position_inputs) == 2;
}
static GuestShaderStage read_raw_graphics(std::string_view bytes, const D3D12_ROOT_SIGNATURE_DESC1 &layout,
                                         unsigned enable_mask, unsigned address_mask, bool mesh_pixel = false) {
    if (bytes.empty() || bytes.size() % 4 || bytes.size() > (1u << 20))
        throw UnsupportedComputeShader("invalid raw graphics extent");
    GuestShaderStage stage{};
    unsigned extent = 0;
    bool position = false, color = false, ended = false;
    struct Parameter { unsigned exported = 0, smooth = 0, flat = 0; };
    std::vector<Parameter> params(mesh_pixel ? 31 : 7);
    const bool has_params = raw_center_inputs(enable_mask) || mesh_pixel;
    std::map<unsigned, unsigned> visited;
    std::deque<unsigned> pending{0};
    while (!pending.empty()) {
        const unsigned pc = pending.front();
        pending.pop_front();
        auto next = visited.lower_bound(pc);
        if (next != visited.end() && next->first == pc) continue;
        if (pc >= bytes.size() || (next != visited.begin() && std::prev(next)->second > pc))
            refuse(pc, "raw graphics branch outside code or into an instruction");
        auto op = gfx1013::instruction(bytes, pc);
        if (next != visited.end() && pc + op.size > next->first)
            refuse(pc, "overlapping raw graphics instructions");
        visited.emplace_hint(next, pc, pc + op.size);
        extent = (std::max)(extent, pc + op.size);
        if (op.form == gfx1013::Form::ds || op.name == "s_swappc_b64" || (!has_params && (op.form == gfx1013::Form::vintrp ||
            op.name.substr(0,9) == "s_cbranch" || op.name == "s_branch" || op.name.find("exec") != Name::npos ||
            op.name.substr(0,7) == "v_cmpx_" || op.name.find("wqm") != Name::npos)))
            refuse(pc, "raw graphics requires a verified stage interface or wave state");
        if (op.form == gfx1013::Form::vintrp) {
            Word w; std::memcpy(&w, bytes.data()+pc, sizeof(w));
            const unsigned attr = (w >> 10) & 63;
            if (attr >= params.size()) refuse(pc, "raw interpolation attribute exceeds entry count");
            auto &mask = ((w >> 16) & 3) == 2 ? params[attr].flat : params[attr].smooth;
            mask |= 1u << ((w >> 8) & 3);
        }
        if (op.form == gfx1013::Form::exp) {
            Word w; std::memcpy(&w, bytes.data()+pc, sizeof(w));
            const unsigned target = (w >> 4) & 63;
            if (target == 12) position = true;
            else if (target < 8) color = true;
            else if (target == 9 && !(w & 15)) {} // Null export on a pixel kill path.
            else if (has_params && target >= 32 && target - 32 < params.size() &&
                     !params[target - 32].exported && !(w & 0x400)) params[target - 32].exported = w & 15;
            else refuse(pc, "raw graphics varying or system export requires a stage interface");
        }
        if (op.name == "s_endpgm") {
            ended = true;
            continue;
        }
        if (op.name == "s_branch" || op.name.substr(0,9) == "s_cbranch") {
            int16_t displacement;
            std::memcpy(&displacement, bytes.data() + pc, sizeof(displacement));
            pending.push_back(pc + op.size + 4 * displacement);
        }
        if (op.name != "s_branch") pending.push_back(pc + op.size);
    }
    if (!ended || position == color) refuse(extent, "ambiguous raw graphics stage");
    stage.stage = position ? GpuStage::vs : GpuStage::ps;
    stage.fixed_vertex_abi = position && has_params;
    stage.code.assign(bytes.substr(0,extent));
    if (color) {
        stage.pixel_input_address = address_mask;
        stage.pixel_input_enable = enable_mask;
    }
    if ((color && (enable_mask & 0x8f00)) || (position && has_params))
        (color ? stage.input_signature : stage.output_signature).push_back({"SV_Position",0,1,3,0,15,15});
    for (uint8_t attr = 0; attr < params.size(); ++attr) {
        const auto &p = params[attr];
        if ((position && (p.smooth || p.flat)) || (color && p.exported) || (p.smooth & p.flat))
            refuse(0, "ambiguous raw varying interface");
        if (mesh_pixel && p.smooth && p.flat)
            refuse(0, "mixed interpolation modes in one packed mesh PARAM");
        // A PARAM may mix smooth channels and provoking-vertex fetches.
        for (unsigned ch = 0; ch < 4; ++ch) {
            const uint8_t bit = uint8_t(1u << ch);
            if (!((p.exported | p.smooth | p.flat) & bit)) continue;
            const unsigned index = 4 * attr + ch;
            const uint8_t row = uint8_t(1 + (mesh_pixel ? attr : index));
            (position ? stage.outputs : stage.attributes).push_back({6,row,bit,attr,0,uint8_t(bool(p.flat & bit))});
            (position ? stage.output_signature : stage.input_signature).push_back({"TEXCOORD",index,0,3,row,bit,bit});
        }
    }
    // Same fixed-descriptor user data layout as the legacy graphics path.
    unsigned sgpr = 8;
    for (unsigned j=0; j<layout.NumParameters; ++j) {
        const auto &p=layout.pParameters[j];
        const bool table=p.ParameterType==D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        const bool constants=p.ParameterType==D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        const unsigned count=table ? 1 : constants ? p.Constants.Num32BitValues : 2;
        if (!count || count>106-sgpr || j>=16) refuse(0,"raw graphics user data exceeds register range");
        if (p.ShaderVisibility==D3D12_SHADER_VISIBILITY_ALL ||
            p.ShaderVisibility==(position ? D3D12_SHADER_VISIBILITY_VERTEX : D3D12_SHADER_VISIBILITY_PIXEL))
            stage.inputs.push_back({table ? 3u : constants ? 0u : 1u,0,count,0,uint8_t(sgpr),uint8_t(j)});
        sgpr+=count;
    }
    if (layout.Flags & D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED) {
        if (sgpr >= 106) refuse(0, "resource heap outside graphics user data");
        stage.inputs.push_back({5,0,1,0,uint8_t(sgpr++),0});
    }
    if (layout.Flags & D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED) {
        if (sgpr >= 106) refuse(0, "sampler heap outside graphics user data");
        stage.inputs.push_back({6,0,1,0,uint8_t(sgpr),0});
    }
    return stage;
}

x2w::NativeComputeShader x2w::translate_mesh_pixel_program(std::string_view code, unsigned enable, unsigned address,
                                                           unsigned wave_size, const D3D12_ROOT_SIGNATURE_DESC1 *root) {
    if (address > 0xffff || (enable & ~address) || (wave_size != 32 && wave_size != 64))
        refuse(0,"invalid fixed mesh pixel entry");
    const D3D12_ROOT_SIGNATURE_DESC1 layout = root ? *root : D3D12_ROOT_SIGNATURE_DESC1{};
    auto ps = read_raw_graphics(code, layout, enable, address, true);
    if (ps.stage != GpuStage::ps) refuse(0,"mesh pixel program has no color export");
    ps.wave32 = wave_size == 32;
    return lower_graphics(ps, 0, 0, nullptr, layout, 0, root != nullptr);
}

x2w::DirectGraphicsPair x2w::translate_graphics_pair(std::string_view a,std::string_view b,
                                                     const D3D12_ROOT_SIGNATURE_DESC1 &layout, std::string_view pixel_entry,
                                                     std::optional<GpuStage> first_stage,
                                                     const std::array<unsigned char, 8> &target_components,
                                                     std::optional<std::array<bool,2>> stage_wave32) {
    const bool raw = a.substr(0,4)!="DXBC" && b.substr(0,4)!="DXBC";
    unsigned enable_mask=0, address_mask=0;
    if (!pixel_entry.empty()) {
        if (!raw || pixel_entry.size()!=13 || pixel_entry.substr(0,5)!="psin1")
            refuse(0,"invalid raw pixel entry record");
        std::memcpy(&enable_mask,pixel_entry.data()+5,4);
        std::memcpy(&address_mask,pixel_entry.data()+9,4);
        if (enable_mask > 0xffff || address_mask > 0xffff || (enable_mask & ~address_mask))
            refuse(0,"contradictory pixel input enable/address masks");
    }
    auto read_stage = [&](std::string_view bytes) {
        return raw ? std::vector<GuestShaderStage>{read_raw_graphics(bytes,layout,enable_mask,address_mask)} : read_guest_shader_container(bytes);
    };
    auto left=read_stage(a),right=read_stage(b);
    if(left.size()!=1 || right.size()!=1)refuse(0,"expected a distinct stage per graphics container");
    if (first_stage && left.front().stage != *first_stage)
        refuse(0,"raw exports disagree with reflected stage");
    auto *vs=&left.front(),*ps=&right.front();
    if(vs->stage==GpuStage::ps)std::swap(vs,ps);
    if(vs->stage!=GpuStage::vs || ps->stage!=GpuStage::ps)refuse(0,"expected vertex and pixel stages");
    ps->target_components=target_components;
    if (raw && stage_wave32) {
        vs->wave32 = (*stage_wave32)[0];
        ps->wave32 = (*stage_wave32)[1];
    }
    return translate_reflected_graphics(*vs,*ps,layout);
}
x2w::DirectGraphicsPair x2w::translate_reflected_graphics(GuestShaderStage &vs, GuestShaderStage &ps,
                                                           const D3D12_ROOT_SIGNATURE_DESC1 &layout) {
    GraphicsConstantLayout constants;
    for (const auto *stage:{&vs,&ps})
        for(const auto &input:stage->inputs)
            if(input.type==0) constants.include(input.root_parameter,input.sgpr_count);
    auto vertex=lower_graphics(vs,0,0,&constants,layout);
    const unsigned vertex_uavs = static_cast<unsigned>(std::count_if(vertex.bindings.begin(), vertex.bindings.end(),
        [](const auto &binding) { return compute_uav(binding.kind); }));
    auto pixel = lower_graphics(ps, static_cast<unsigned>(vertex.bindings.size()) - vertex_uavs,
                                static_cast<unsigned>(vertex.samplers.size()), &constants, layout, vertex_uavs);
    for (auto &binding : vertex.bindings) apply_reflected_table_use(binding, vs);
    for (auto &binding : pixel.bindings) apply_reflected_table_use(binding, ps);
    validate_raster_link(vertex, pixel);
    return {std::move(vertex),std::move(pixel)};
}
void x2w::validate_raster_link(const NativeComputeShader &vertex, const NativeComputeShader &pixel) {
    for(const auto &input:pixel.inputs){
        if(input.name=="SV_IsFrontFace" || input.kind==28) continue;
        auto producer=std::find_if(vertex.outputs.begin(),vertex.outputs.end(),[&](auto &out){
            return out.name==input.name && out.index==input.index && out.row==input.row && out.component==input.component &&
                ((out.used<<out.first)&(input.used<<input.first))==(input.used<<input.first);
        });
        if(producer==vertex.outputs.end())refuse(0,"graphics stages disagree on "+input.name+to_string(input.index));
    }
}
