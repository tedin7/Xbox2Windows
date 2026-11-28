#include "gfx1013_vertex_id.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {
using Word = std::uint32_t;

class VertexProgram {
    std::string_view code;
    std::size_t pc = 0, address = 0;
    std::array<std::string, 256> values;
    std::optional<Word> literal;
    bool position = false;
    std::array<bool, 2> parameters{};
    const bool buffered;
    enum class Kind { undefined, value, table, vb_table, vb_tagged_high, vb_high, zero_high, desc };
    struct Scalar {
        Kind kind{};
        int table{};
        unsigned offset{};
        std::string value;
    };
    std::array<Scalar, 102> scalars{};
    std::vector<std::pair<int, unsigned>> buffers;
    std::string declarations, bindings;
    unsigned next = 0;
    std::string body;

    [[noreturn]] void fail(const char *reason) const {
        throw std::runtime_error(std::string(reason) + " at vertex byte " + std::to_string(address));
    }
    Word read() {
        if (code.size() - pc < sizeof(Word)) fail("truncated instruction");
        Word result;
        std::memcpy(&result, code.data() + pc, sizeof(result));
        pc += sizeof(result);
        return result;
    }
    std::string operand(Word selector) {
        if (selector >= 256) {
            const auto &value = values[selector - 256];
            if (value.empty()) fail("undefined VGPR");
            return value;
        }
        if (buffered && selector < scalars.size()) {
            if (scalars[selector].kind == Kind::zero_high) return "0u";
            if (scalars[selector].kind != Kind::value) fail("unresolved scalar input");
            return scalars[selector].value;
        }
        Word bits = 0;
        if (selector >= 128 && selector <= 192)
            bits = selector - 128;
        else if (selector >= 193 && selector <= 208)
            bits = 192u - selector;
        else if (selector >= 240 && selector <= 247) {
            constexpr Word constants[]{0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
                                       0x40000000, 0xc0000000, 0x40800000, 0xc0800000};
            bits = constants[selector - 240];
        } else if (selector == 255) {
            if (!literal) literal = read();
            bits = *literal;
        } else
            fail("unsupported input selector");
        return std::to_string(bits) + "u";
    }
    std::string temporary(const std::string &expression) {
        auto name = "r" + std::to_string(next++);
        body += "    uint " + name + " = " + expression + ";\n";
        return name;
    }
    void assign(Word reg, const std::string &expression) { values[reg] = temporary(expression); }
    std::string buffer(Word base, bool vertex) {
        if (base + 4 > scalars.size()) fail("invalid buffer descriptor range");
        const auto &first = scalars[base];
        if (first.kind != Kind::desc || (first.table < 0) != vertex) fail("unexpected buffer descriptor");
        for (unsigned i = 1; i < 4; ++i) {
            const auto &part = scalars[base + i];
            if (part.kind != Kind::desc || part.table != first.table || part.offset != first.offset + 4 * i)
                fail("overwritten buffer descriptor");
        }
        const auto id = std::make_pair(first.table, first.offset);
        unsigned reg = 0;
        for (const auto &entry : buffers) {
            if (entry == id) break;
            ++reg;
        }
        const auto number = std::to_string(reg);
        if (reg == buffers.size()) {
            buffers.push_back(id);
            declarations += "ByteAddressBuffer b" + number + " : register(t" + number + ");\n";
            bindings +=
                "srv " + number + " buf " +
                (vertex ? "vb 0" : "table " + std::to_string(first.table) + " " + std::to_string(first.offset)) +
                "\n";
        }
        return "b" + number;
    }
    void scalar_load(Word op) {
        const Word base = (op & 63) * 2, dest = (op >> 6) & 127, offset_word = read();
        if ((offset_word & 0xfff00003) != 0xfa000000) fail("unsupported scalar load offset");
        const Word offset = offset_word & 0xfffff;
        const auto opcode = op & ~0x1fffu;
        const unsigned count = opcode == 0xf4300000 ? 16 : 4;
        if (base + 1 >= scalars.size() || dest + count > scalars.size()) fail("scalar load register range");
        if (opcode == 0xf4080000) {
            const auto pointer = scalars[base];
            const auto high = scalars[base + 1].kind;
            if (pointer.kind == Kind::vb_table) {
                if (offset || high != Kind::vb_high) fail("expected vertex stream zero");
            } else if (pointer.kind != Kind::table || high != Kind::zero_high || offset % 32)
                fail("unresolved root table");
            for (unsigned i = 0; i < count; ++i)
                scalars[dest + i] = {Kind::desc, pointer.table, offset + 4 * i};
        } else {
            const auto resource = buffer(base, false);
            for (unsigned i = 0; i < count; ++i)
                scalars[dest + i] = {Kind::value, 0, 0,
                                     temporary(resource + ".Load(" + std::to_string(offset + 4 * i) + "u)")};
        }
    }
    void load_vertex(Word op) {
        const auto args = read();
        if ((args & 0xffe00000) != 0x80000000) fail("unsupported vertex load operands");
        const auto resource = buffer(((args >> 16) & 31) * 4, true);
        const unsigned count = (op & 0xfffff000) == 0xe03c2000 ? 3 : 2;
        const unsigned dest = (args >> 8) & 255;
        if (dest + count > values.size()) fail("vertex load destination range");
        const auto address =
            temporary(operand(256 + (args & 255)) + " * ud3 + " + std::to_string(op & 4095) + "u");
        for (unsigned i = 0; i < count; ++i)
            assign(dest + i, resource + ".Load(" + address + "+" + std::to_string(4 * i) + "u)");
    }
    bool buffer_instruction(Word op) {
        const auto dest = (op >> 16) & 127, src = op & 255;
        if ((op & 0xff80ff00) == 0xbe800300 || (op & 0xff80ff00) == 0xbe800400) {
            if (dest >= scalars.size()) fail("invalid scalar destination");
            if (((op >> 8) & 255) == 4) {
                if (dest + 1 >= scalars.size() || src + 1 >= scalars.size() || src % 2 || dest % 2 ||
                    scalars[src].kind == Kind::undefined || scalars[src + 1].kind == Kind::undefined)
                    fail("unsupported scalar pair");
                const auto low = scalars[src], high = scalars[src + 1];
                scalars[dest] = low;
                scalars[dest + 1] = high;
            } else if (src == 128)
                scalars[dest] = {Kind::zero_high};
            else if (src < scalars.size() && scalars[src].kind != Kind::undefined)
                scalars[dest] = scalars[src];
            else
                fail("unsupported scalar move");
        } else if ((op & 0xff80ff00) == 0x9380ff00) {
            if (read() != 0x80000 || dest >= scalars.size() || src >= scalars.size() ||
                scalars[src].kind != Kind::vb_tagged_high)
                fail("unsupported vertex address high word");
            scalars[dest] = {Kind::vb_high};
        } else if ((op & ~0x1fffu) == 0xf4080000 || (op & ~0x1fffu) == 0xf4280000 ||
                   (op & ~0x1fffu) == 0xf4300000) {
            scalar_load(op);
        } else if ((op & 0xfffff000) == 0xe03c2000 || (op & 0xfffff000) == 0xe0342000) {
            load_vertex(op);
        } else if ((op & 0xfe01fe00) == 0x7e005c00) {
            assign((op >> 17) & 255, "asuint(rsqrt(asfloat(" + operand(op & 511) + ")))");
        } else if ((op & 0xffffff00) == 0xd5030000) {
            const auto args = read();
            if (args & 0x9ffc0000) fail("unsupported add modifiers");
            const auto a = operand(args & 511), b = operand((args >> 9) & 511);
            assign(op & 255, std::string("asuint(") + ((args >> 29) & 1 ? "-" : "") + "asfloat(" + a + ") + " +
                                 ((args >> 30) & 1 ? "-" : "") + "asfloat(" + b + "))");
        } else if (op >> 25 == 3 || op >> 25 == 4 || op >> 25 == 5 || op >> 25 == 8 || op >> 25 == 37 ||
                   op >> 25 == 43) {
            const auto d = (op >> 17) & 255;
            auto a = operand(op & 511), b = operand(256 + ((op >> 9) & 255));
            if (op >> 25 == 37)
                assign(d, a + " + " + b);
            else {
                a = "asfloat(" + a + ")";
                b = "asfloat(" + b + ")";
                if (op >> 25 == 5) std::swap(a, b);
                const auto value = op >> 25 == 43 ? "mad(" + a + "," + b + ",asfloat(" + operand(256 + d) + "))"
                                                  : a +
                                                        (op >> 25 == 8   ? " * "
                                                         : op >> 25 == 3 ? " + "
                                                                         : " - ") +
                                                        b;
                assign(d, "asuint(" + value + ")");
            }
        } else
            return (op >> 16) == 0xbfa1 || (op >> 16) == 0xbf8c; // clause and waitcnt
        return true;
    }
    void export_value(bool pos, Word mask, unsigned param = 0) {
        if (pos ? position : parameters[param]) fail("duplicate vertex export");
        const Word sources = read();
        std::string vector;
        for (unsigned i = 0; i < 4; ++i) {
            if (i) vector += ", ";
            const Word reg = (sources >> (8 * i)) & 255;
            if (mask & (1u << i))
                vector += "asfloat(" + operand(256 + reg) + ")";
            else {
                if (reg != 128) fail("unexpected disabled export operand");
                vector += "0.0";
            }
        }
        body += (pos ? "    o.pos" : "    o.param" + std::to_string(param)) + " = float4(" + vector + ");\n";
        (pos ? position : parameters[param]) = true;
    }

  public:
    VertexProgram(std::string_view bytes, bool uses_buffers) : code(bytes), buffered(uses_buffers) {
        values[0] = "vid";
        scalars[0] = {Kind::vb_table, -1};
        scalars[1] = {Kind::vb_tagged_high};
        scalars[2] = {Kind::value, 0, 0, "ud1"};
        scalars[8] = {Kind::table, 0};
        scalars[9] = {Kind::table, 1};
    }
    x2w::BufferedVertex translate() {
        while (pc < code.size()) {
            address = pc;
            literal.reset();
            const auto op = read();
            if (op == 0xbf810000) {
                if (!position || !parameters[0] || pc != code.size())
                    fail("incomplete exports or trailing instructions");
                const std::string param1 = parameters[1] ? "float4 param1 : PARAM1; " : "";
                return {declarations + "cbuffer entry : register(b0) { uint ud0; uint ud1; uint ud2; " +
                            (buffered ? "uint ud3; " : "") + "};\nstruct Out { float4 param0 : PARAM0; " + param1 +
                            "float4 pos : SV_Position; };\nOut vertex(uint vid) {\n    Out o;\n" + body +
                            R"(    return o;
}
Out main(uint vid : SV_VertexID) {
    if (!ud0) return vertex(vid);
    uint first = (vid / 6) * 3, corner = vid % 6;
    if (corner < 3) return vertex(first + corner);
    if (corner == 3) return vertex(first);
    if (corner == 4) return vertex(first + 2);
    Out a = vertex(first), b = vertex(first + 1), c = vertex(first + 2), o;
    o.pos = a.pos + c.pos - b.pos;
    o.param0 = a.param0 + c.param0 - b.param0;
)" + (parameters[1] ? "    o.param1 = a.param1 + c.param1 - b.param1;\n" : "") +
                            R"(    return o;
}
)",
                        bindings};
            }
            if ((op >> 16) == 0xbfa0 || (op & ~3u) == 0xbf8f0000) continue; // prefetch and priority
            if (buffered && buffer_instruction(op)) continue;
            if ((op & 0xfe01fe00) == 0x7e000200 || (op & 0xfe01fe00) == 0x7e000a00) {
                auto value = operand(op & 511);
                if ((op & 0x1fe00) == 0xa00) value = "asuint(float(asint(" + value + ")))";
                assign((op >> 17) & 255, value);
            } else if (op >> 25 == 26 || op >> 25 == 27) {
                auto a = operand(op & 511);
                auto b = operand(256 + ((op >> 9) & 255));
                assign((op >> 17) & 255, op >> 25 == 26 ? b + " << (" + a + " & 31u)" : a + " & " + b);
            } else if ((op & 0xffffff00) == 0xd54b0000) {
                const auto args = read();
                if (args & (buffered ? 0x18000000u : 0xf8000000u)) fail("unsupported FMA modifiers");
                auto a = operand(args & 511);
                auto b = operand((args >> 9) & 511);
                auto c = operand((args >> 18) & 511);
                const auto f = [&](const std::string &v, unsigned bit) {
                    return ((args >> bit) & 1 ? "-asfloat(" : "asfloat(") + v + ")";
                };
                assign(op & 255, "asuint(mad(" + f(a, 29) + ", " + f(b, 30) + ", " + f(c, 31) + "))");
            } else if (op == 0xf80008cf) {
                export_value(true, 15);
            } else if ((op & ~15u) == 0xf8000200 && (op & 15)) {
                export_value(false, op & 15);
            } else if (buffered && (op & ~15u) == 0xf8000210 && (op & 15)) {
                export_value(false, op & 15, 1);
            } else
                fail("unsupported vertex instruction");
        }
        fail("missing endpgm");
    }
};
} // namespace

std::string x2w::translate_gfx1013_vertex_id(std::string_view code) {
    return VertexProgram(code, false).translate().hlsl;
}

x2w::BufferedVertex x2w::translate_gfx1013_buffered_vertex(std::string_view code) {
    return VertexProgram(code, true).translate();
}
