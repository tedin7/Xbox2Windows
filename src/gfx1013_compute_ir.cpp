#include "gfx1013_compute_ir.h"
#include <charconv>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace x2w {
LocalDataAccess decode_local_data(uint32_t instruction, uint32_t operands) {
    if ((instruction & 0xfc030000u) != 0xd8000000u)
        throw std::runtime_error("expected a local DS instruction with GDS clear");
    const unsigned op = (instruction >> 18) & 255, offset = instruction & 65535;
    const unsigned addr = operands & 255, data0 = (operands >> 8) & 255, data1 = (operands >> 16) & 255,
                   dst = operands >> 24;
    LocalDataAccess access{offset, addr, data0, data1, dst, 1};
    // The DS opcode/tuple layout follows Victory 9376fd6 gfx10/{opcodes,fields}.rs.
    // The 32-bit read-modify-write operations follow compute_store_aot.py LDS_RMW.
    switch (op) {
    case 0x00: case 0x20: case 0x01: case 0x21:
        access.atomic_ir = "add"; access.atomic_hlsl = "Add"; access.subtract = (op & 1) != 0;
        break;
    case 0x07: case 0x27: access.atomic_ir = "umin"; access.atomic_hlsl = "Min"; break;
    case 0x08: case 0x28: access.atomic_ir = "umax"; access.atomic_hlsl = "Max"; break;
    case 0x09: case 0x29: access.atomic_ir = "and"; access.atomic_hlsl = "And"; break;
    case 0x0a: case 0x2a: access.atomic_ir = "or"; access.atomic_hlsl = "Or"; break;
    case 0x0b: case 0x2b: access.atomic_ir = "xor"; access.atomic_hlsl = "Xor"; break;
    case 0x2d: access.atomic_ir = "xchg"; access.atomic_hlsl = "Exchange"; break;
    case 0x36: access.read = true; break; // read_b32
    case 0x76: access.read = true; access.count = 2; break;
    case 0xfe: access.read = true; access.count = 3; break;
    case 0xff: access.read = true; access.count = 4; break;
    case 0x0d: break; // write_b32
    case 0x4d: access.count = 2; break;
    case 0xde: access.count = 3; break;
    case 0xdf: access.count = 4; break;
    case 0x0e: case 0x0f: case 0x37: case 0x38:
    case 0x4e: case 0x4f: case 0x77: case 0x78:
        access.pair_words = op >= 0x40 ? 2 : 1;
        access.count = 2 * access.pair_words;
        access.read = (op & 0x3f) >= 0x37;
        access.stride64 = (op & 0x3f) == 0x0f || (op & 0x3f) == 0x38;
        break;
    default:
        throw std::runtime_error("unsupported local DS opcode " + std::to_string(op));
    }
    access.return_old = access.atomic_ir && op >= 0x20;
    const unsigned first_width = access.pair_words ? access.pair_words : access.count;
    if (access.read ? data0 || data1 || dst + access.count > 256 :
        (!access.return_old && dst) || (!access.pair_words && data1) ||
        data0 + first_width > 256 || data1 + access.pair_words > 256)
        throw std::runtime_error("invalid local DS register tuple");
    return access;
}

static std::string lower_lds(uint32_t instruction, uint32_t operands) {
    const auto access = decode_local_data(instruction, operands);
    // Preserve the input address when a read writes back to the address VGPR.
    std::string out = "if (exec & 1u) { uint ds_addr = v" + std::to_string(access.address) + "; ";
    for (unsigned i = 0; i < access.count; ++i) {
        auto location = "lds[(ds_addr + " + std::to_string(access.byte_offset(i)) + "u) >> 2]";
        auto reg = "v" + std::to_string(access.read ? access.destination + i : access.data_register(i));
        if (access.atomic_ir) {
            auto value = access.subtract ? "(0u - " + reg + ")" : reg;
            out += std::string("Interlocked") + access.atomic_hlsl + "(" + location + ", " + value;
            if (access.return_old)
                out += ", v" + std::to_string(access.destination);
            out += "); ";
        } else
            out += access.read ? reg + " = " + location + "; " : location + " = " + reg + "; ";
    }
    return out + "}";
}

// Single-word VOP2 with an optional 32-bit literal. SGPR and lane-special
// operands require provenance from the entry layout and stay in the frontend.
static std::string lower_vop2(uint32_t word, std::optional<uint32_t> literal) {
    const auto src0 = word & 511u;
    if ((word >> 31) || (src0 == 255) != literal.has_value())
        throw std::runtime_error("invalid VOP2 word or literal length");
    std::string a;
    if (src0 >= 256)
        a = "v" + std::to_string(src0 - 256);
    else if (literal)
        a = std::to_string(*literal) + "u";
    else if (src0 >= 128 && src0 <= 192)
        a = std::to_string(src0 - 128) + "u";
    else if (src0 >= 193 && src0 <= 208)
        a = std::to_string(192u - src0) + "u";
    else
        throw std::runtime_error("unsupported VOP2 source");
    const std::string b = "v" + std::to_string((word >> 9) & 255);
    const std::string dst = "v" + std::to_string((word >> 17) & 255);
    std::string expr;
    switch (word >> 25) {
    case 3:
        expr = "asuint(asfloat(" + a + ") + asfloat(" + b + "))";
        break;
    case 4:
        expr = "asuint(asfloat(" + a + ") - asfloat(" + b + "))";
        break;
    case 5:
        expr = "asuint(asfloat(" + b + ") - asfloat(" + a + "))";
        break;
    case 8:
        expr = "asuint(asfloat(" + a + ") * asfloat(" + b + "))";
        break;
    case 11:
        expr = "(" + a + " & 0xffffffu) * (" + b + " & 0xffffffu)";
        break;
    case 26:
        expr = b + " << (" + a + " & 31u)";
        break;
    case 37:
        expr = a + " + " + b;
        break;
    default:
        throw std::runtime_error("unsupported VOP2 opcode");
    }
    return "if (exec & 1u) " + dst + " = " + expr + ";";
}

std::string expand_gfx1013_compute(const std::string &source) {
    constexpr std::string_view lds = "#pragma x2w_lds ", alu = "#pragma x2w_vop2 ";
    std::string output;
    size_t pos = 0;
    while (pos < source.size()) {
        auto end = source.find('\n', pos);
        if (end == source.npos)
            end = source.size();
        std::string_view line(source.data() + pos, end - pos);
        auto first = line.find_first_not_of(" \t");
        const auto trimmed = first == line.npos ? std::string_view{} : line.substr(first);
        const bool ds = trimmed.substr(0, lds.size()) == lds;
        if (ds || trimmed.substr(0, alu.size()) == alu) {
            auto args = trimmed.substr(ds ? lds.size() : alu.size());
            if (!args.empty() && args.back() == '\r')
                args.remove_suffix(1);
            if ((args.size() != 17 || args[8] != ' ') && (ds || args.size() != 8))
                throw std::runtime_error("invalid native compute instruction length");
            auto parse = [](std::string_view token) {
                uint32_t word{};
                auto r = std::from_chars(token.data(), token.data() + token.size(), word, 16);
                if (r.ec != std::errc{} || r.ptr != token.data() + token.size())
                    throw std::runtime_error("invalid compute word");
                return word;
            };
            auto instruction = parse(args.substr(0, 8));
            std::optional<uint32_t> second;
            if (args.size() == 17)
                second = parse(args.substr(9));
            output += ds ? lower_lds(instruction, *second) : lower_vop2(instruction, second);
        } else {
            if (trimmed.substr(0, 12) == "#pragma x2w_")
                throw std::runtime_error("unknown native compute pragma");
            output.append(line);
        }
        if (end < source.size())
            output += '\n';
        pos = end + 1;
    }
    return output;
}
} // namespace x2w
