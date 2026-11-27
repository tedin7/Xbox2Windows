#include "gfx1013_isa.h"
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>

namespace x2w::gfx1013 {
namespace {
#include "gfx1013_isa_tables.inl"
struct Names {
    const char *const *data;
    size_t size;
};
template <size_t N> constexpr Names names(const char *const (&a)[N]) { return {a, N}; }
constexpr Names table[]{names(sopp_names),  names(sop1_names),   names(sop2_names),   names(sopc_names),
                        names(sopk_names),  names(vop1_names),   names(vop2_names),   names(vopc_names),
                        names(vop3_names),  names(vop3p_names),  names(vintrp_names), names(smem_names),
                        names(mubuf_names), names(mtbuf_names),  names(mimg_names),   names(ds_names),
                        names(flat_names),  names(global_names), names(scratch_names)};
std::string_view lookup(Form form, unsigned index) {
    if (form == Form::exp)
        return "exp";
    const auto &t = table[static_cast<unsigned>(form)];
    return index < t.size && t.data[index] ? t.data[index] : "";
}
} // namespace
Instruction instruction(std::string_view code, unsigned pc) {
    auto word = [&](unsigned index) {
        uint64_t at = uint64_t(pc) + 4 * index;
        if ((pc & 3) || at > code.size() || code.size() - at < 4)
            throw std::runtime_error("truncated or misaligned gfx1013 instruction at byte " + std::to_string(pc));
        uint32_t n;
        std::memcpy(&n, code.data() + at, 4);
        return n;
    };
    const auto a = word(0);
    const unsigned top = a >> 26, major = a >> 25;
    Form form;
    unsigned op, count = 1;
    if (!(a >> 31)) {
        form = major == 63 ? Form::vop1 : major == 62 ? Form::vopc : Form::vop2;
        op = form == Form::vop1 ? (a >> 9) & 255 : form == Form::vopc ? (a >> 17) & 255 : major;
        const auto src = a & 511;
        bool extension = src == 233 || src == 234 || src == 249 || src == 250 || src == 255;
        if (form == Form::vop2) {
            const auto name = lookup(form, op);
            extension |= name.find("madmk") != name.npos || name.find("madak") != name.npos ||
                         name.find("fmamk") != name.npos || name.find("fmaak") != name.npos;
        }
        count += extension;
    } else if ((a >> 30) == 2) {
        switch (a >> 23) {
        case 0x17d:
            form = Form::sop1;
            op = (a >> 8) & 255;
            count += (a & 255) == 255;
            break;
        case 0x17e:
            form = Form::sopc;
            op = (a >> 16) & 127;
            count += (a & 255) == 255 || ((a >> 8) & 255) == 255;
            break;
        case 0x17f:
            form = Form::sopp;
            op = (a >> 16) & 127;
            break;
        default:
            if ((a >> 28) == 11) {
                form = Form::sopk;
                op = (a >> 23) & 31;
                count += lookup(form, op) == "s_setreg_imm32_b32";
            } else {
                form = Form::sop2;
                op = (a >> 23) & 127;
                count += (a & 255) == 255 || ((a >> 8) & 255) == 255;
            }
        }
    } else {
        count = 2;
        switch (top) {
        case 0x32:
            form = Form::vintrp;
            op = (a >> 16) & 3;
            count = 1;
            break;
        case 0x33:
        case 0x35: {
            form = top == 0x33 ? Form::vop3p : Form::vop3;
            op = (a >> 16) & (top == 0x33 ? 127 : 1023);
            auto b = word(1);
            count += ((b & 511) == 255 || ((b >> 9) & 511) == 255 || ((b >> 18) & 511) == 255);
            break;
        }
        case 0x36:
            form = Form::ds;
            op = (a >> 18) & 255;
            break;
        case 0x37: {
            unsigned segment = (a >> 14) & 3;
            if (segment == 3)
                throw std::runtime_error("reserved FLAT segment at byte " + std::to_string(pc));
            form = segment == 0 ? Form::flat : segment == 1 ? Form::scratch : Form::global;
            op = (a >> 18) & 127;
            break;
        }
        case 0x38:
            form = Form::mubuf;
            op = (a >> 18) & 255;
            break;
        case 0x3a:
            form = Form::mtbuf;
            op = ((a >> 16) & 7) | ((word(1) >> 18) & 8);
            break;
        case 0x3c:
            form = Form::mimg;
            op = ((a >> 18) & 127) | ((a & 1) << 7);
            count += (a >> 1) & 3;
            break;
        case 0x3d:
            form = Form::smem;
            op = (a >> 18) & 255;
            break;
        case 0x3e:
            form = Form::exp;
            op = 0;
            break;
        default:
            throw std::runtime_error("unknown gfx1013 instruction format at byte " + std::to_string(pc));
        }
    }
    word(count - 1);
    auto name = lookup(form, op);
    if (form == Form::vop3 && name.empty()) {
        if (op < 0x100)
            name = lookup(Form::vopc, op);
        else if (op < 0x140)
            name = lookup(Form::vop2, op - 0x100);
        else if (op >= 0x180 && op < 0x200)
            name = lookup(Form::vop1, op - 0x180);
    }
    return {form, op, 4 * count, name};
}
} // namespace x2w::gfx1013
