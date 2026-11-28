#include "gfx1013_pixel.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>

static std::array<uint32_t, 4> decode_color(std::string_view code) {
    std::array<std::optional<uint32_t>, 256> vgpr{};
    size_t pc = 0;
    auto word = [&]() {
        if (code.size() - pc < 4)
            throw std::runtime_error("truncated pixel program at " + std::to_string(pc));
        uint32_t value;
        std::memcpy(&value, code.data() + pc, 4);
        pc += 4;
        return value;
    };
    std::optional<std::array<uint32_t, 4>> result;
    while (pc < code.size()) {
        const auto address = pc;
        const auto op = word();
        if (op == 0xbf810000 && result.has_value() && pc == code.size())
            return *result;
        if (result.has_value())
            throw std::runtime_error("expected final s_endpgm at " + std::to_string(address));
        if ((op & 0xfe01fe00u) == 0x7e000200u) {
            const auto src = op & 511;
            auto &dst = vgpr[(op >> 17) & 255];
            if (src >= 256 && vgpr[src - 256])
                dst = vgpr[src - 256];
            else if (src == 255)
                dst = word();
            else if (src >= 128 && src <= 192)
                dst = src - 128;
            else if (src >= 193 && src <= 208)
                dst = 192u - src;
            else if (src >= 240 && src <= 247) {
                constexpr uint32_t floats[]{0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
                                            0x40000000, 0xc0000000, 0x40800000, 0xc0800000};
                dst = floats[src - 240];
            } else
                throw std::runtime_error("unresolved v_mov source at " + std::to_string(address));
        } else if (op == 0xf800180f) {
            const auto sources = word();
            result.emplace();
            for (unsigned component = 0; component < 4; ++component) {
                const auto &value = vgpr[(sources >> (component * 8)) & 255];
                if (!value)
                    throw std::runtime_error("uninitialized export VGPR at " + std::to_string(address));
                (*result)[component] = *value;
            }
        } else
            throw std::runtime_error("unsupported pixel instruction at " + std::to_string(address));
    }
    throw std::runtime_error("missing final MRT0 export or s_endpgm");
}

std::string x2w::translate_gfx1013_constant_pixel(std::string_view code) {
    std::string out = "float4 main() : SV_Target0 { return asfloat(uint4(";
    for (auto word : decode_color(code))
        out += std::to_string(word) + "u,";
    out.back() = ')';
    return out + "); }";
}

std::string x2w::direct_gfx1013_constant_pixel(std::string_view code) {
    auto color = decode_color(code);
    std::ostringstream out;
    out << "target triple = \"dxil-ms-dx\"\ndefine void @main() {\n";
    for (unsigned c = 0; c < color.size(); ++c)
        out << "  call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 " << c << ", float bitcast (i32 "
            << color[c] << " to float))\n";
    out << R"(  ret void
}
declare void @dx.op.storeOutput.f32(i32,i32,i32,i8,float) nounwind
!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!1}
!dx.shaderModel = !{!2}
!dx.viewIdState = !{!3}
!dx.entryPoints = !{!4}
!0 = !{!"x2w"}
!1 = !{i32 1, i32 0}
!2 = !{!"ps", i32 6, i32 0}
!3 = !{[2 x i32] [i32 0, i32 4]}
!4 = !{void ()* @main, !"main", !5, null, null}
!5 = !{null, !6, null}
!6 = !{!7}
!7 = !{i32 0, !"SV_Target", i8 9, i8 16, !8, i8 0, i32 1, i8 4, i32 0, i8 0, !9}
!8 = !{i32 0}
!9 = !{i32 3, i32 15}
)";
    return out.str();
}
