#pragma once
#include <cstdint>
#include <string_view>

namespace x2w::gfx1013 {
enum class Form {
    sopp,
    sop1,
    sop2,
    sopc,
    sopk,
    vop1,
    vop2,
    vopc,
    vop3,
    vop3p,
    vintrp,
    smem,
    mubuf,
    mtbuf,
    mimg,
    ds,
    flat,
    global,
    scratch,
    exp
};
struct Instruction {
    Form form;
    unsigned opcode, size;
    std::string_view name;
};
Instruction instruction(std::string_view code, unsigned pc);
} // namespace x2w::gfx1013
