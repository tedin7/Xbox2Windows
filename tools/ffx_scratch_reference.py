"""CPU reference for the i8 dataflow that reaches a flat 8x4 scene/history blend.

The oracle uses row/column tensor addressing and Python signed-byte arithmetic.
It evaluates only the integer network and its quantizers. No generated HLSL is
parsed or executed. The four final logits suffice for spatially constant colors.
"""
from collections.abc import Mapping
import re
import struct


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def real(x):
    return struct.unpack("<f", struct.pack("<I", x))[0]


def signed(x):
    return x if x < 0x80000000 else x - 0x100000000


def simm16(token):
    if not isinstance(token, str):
        raise ValueError(f"Invalid 16-bit signed immediate: {token!r}")
    value = int(token, 0)
    if -0x8000 <= value < 0:
        return value & 0xffffffff
    if 0 <= value <= 0xffff:
        return (value - 0x10000 if value & 0x8000 else value) & 0xffffffff
    raise ValueError(f"Invalid 16-bit signed immediate: {token}")


def seed(x, y, c):
    return ((x + 13*y + 19*c + 1)*0x19660d + 0x3c6ef35f) & 0xffffffff


def network(program, taps, initializer, stage):
    if isinstance(stage, bool) or not isinstance(stage, int) or stage not in (24, 26):
        raise ValueError(f"Expected network stage 24 or 26, got {stage!r}")
    regs = {"s16": 0}  # group z
    logits = []
    tap = 0
    for pc, line in program:
        if stage == 26 and (0x2414 <= pc < 0x26dc or 0x2a40 <= pc < 0x3c48):
            continue
        mn, _, operands = line.partition(" ")
        operands, _, mods = operands.partition(" op_sel")
        dst, *src = operands.split(", ")
        if mn == "buffer_load_dwordx4":
            first = int(dst.split(":")[0][2:])
            for j in range(4):
                regs[f"v{first+j}"] = taps[tap][j]
            tap += 1
            continue
        if mn == "s_buffer_load_dwordx16":
            if not src:
                raise ValueError("Missing constant-buffer offset for s_buffer_load_dwordx16")
            first = int(dst.split(":")[0][2:])
            offset = int(src[-1], 0)
            if offset < 0 or offset % 4 != 0 or offset + 64 > len(initializer):
                raise ValueError(
                    f"Constant-buffer offset {offset} out of bounds for {len(initializer)}-byte initializer"
                )
            for i, val in enumerate(struct.unpack_from("<16I", initializer, offset)):
                regs[f"s{first+i}"] = val
            continue
        if stage == 24 and mn == "buffer_store_dwordx4":
            return [regs[f"v{i}"] for i in range(4)]
        if not re.fullmatch("[vs][0-9]+", dst):
            continue
        def value(s):
            if s.startswith(("s","v")):
                return regs.get(s)
            try:
                return bits(float(s)) if "." in s else int(s, 0) & 0xffffffff
            except ValueError:
                return None
        op = mn.removesuffix("_e32")
        args = [value(token) for token in src]
        result = None
        if None not in args:
            if op in ("s_mov_b32", "v_mov_b32"):
                result = args[0]
            elif op == "s_movk_i32":
                result = simm16(src[0])
            elif op == "s_add_u32":
                result = sum(args)
            elif op == "s_addk_i32" and regs.get(dst) is not None:
                result = regs[dst] + simm16(src[0])
            elif op == "s_mulk_i32" and regs.get(dst) is not None:
                result = regs[dst] * simm16(src[0])
            elif op == "s_brev_b32":
                result = int(f"{args[0]:032b}"[::-1],2)
            elif op == "v_add_f32":
                result = bits(real(args[0])+real(args[1]))
            elif op == "v_min_f32":
                result = bits(min(real(args[0]),real(args[1])))
            elif op == "v_xor_b32":
                result = args[0]^args[1]
            elif op == "v_cvt_pk_u8_f32":
                channel = min(255,max(0,round(real(args[0]))))
                shift = 8*args[1]
                result = (args[2] & ~(255<<shift)) | (channel<<shift)
            elif op in ("v_dot4_i32_i8", "v_dot4c_i32_i8"):
                lhs, rhs = args[:2]
                if mods:
                    high = rhs>>16 if mods.startswith(":[0,1,0]") else rhs&65535
                    rhs = ((rhs&65535)<<16) | high
                accum = args[2] if len(args)==3 else regs[dst]
                a = struct.unpack("<4b",lhs.to_bytes(4,"little"))
                b = struct.unpack("<4b",rhs.to_bytes(4,"little"))
                result = accum + sum(x*y for x,y in zip(a,b))
        regs[dst] = None if result is None else result & 0xffffffff
        if stage == 26 and pc in (0x2880,0x4370,0x4ac0,0x51ec):
            if regs[dst] is None:
                raise ValueError(f"Unresolved history logit at {pc:#x}")
            logits.append(signed(regs[dst]))
            if len(logits)==4:
                return logits
    raise ValueError("Incomplete tensor reference")


def make_reference(programs, initializer):
    if not isinstance(programs, Mapping) or "pass24" not in programs or "producer" not in programs:
        raise ValueError("Expected programs mapping with 'pass24' and 'producer'")
    rows, gates = [], []
    for mode in range(3):
        # The 6x4 union includes a pass-24 overdispatch column at x=5.
        # Pass 25 subsequently clears this column, the top row and the left edge.
        values = {}
        before = []
        for y in range(4):
            for x in range(6):
                value = [0]*4
                if mode and x and y:
                    taps = [[seed(x-1+dx,y-1+dy,c) if mode==2 else 0 for c in range(4)]
                            for dy in range(3) for dx in range(3)]
                    value = network(programs["pass24"], taps, initializer, 24)
                before.extend(value)
                values[x,y] = value if 1<=x<=4 and y else [0]*4
        rows.extend(before)
        grid = [0]*32
        for y in range(2):
            for x in range(4):
                taps = [values[x+dx,y+dy] for dy in range(3) for dx in range(3)]
                logits = network(programs["producer"], taps, initializer, 26)
                for i, logit in enumerate(logits):
                    grid[(2*y+i//2)*8+2*x+i%2] = logit
        gates.extend(grid)
    if rows[96:192] == rows[192:] or gates[32:64] == gates[64:]:
        raise ValueError("Fixture does not distinguish the two upstream inputs")
    return ("static constexpr uint32_t before_padding[] = {" + ",".join(hex(n)+"u" for n in rows) +
            "};\nstatic constexpr int history_logits[] = {" + ",".join(map(str,gates)) + "};\n")
