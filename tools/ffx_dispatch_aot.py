"""Lower the four pinned compute programs for the bounded 8x4 dispatch experiment."""
import argparse
import hashlib
from pathlib import Path
import re
import struct

from pixel_aot import disassemble
from ffx_scratch_reference import make_reference, simm16


DIGEST = {
    "pass24": "e158269f335449abd33ffccebf7a4652a4e08bd8a4d3e39ce6d765807c049669",
    "pass25": "97c5afa22f8d10d2b602118067161026f618e5a63b4d0a9ee490620db0846cbf",
    "producer": "407c213a5cb6c92c08509f7a8783550d775fe5518399eab3eecf4c736a78e4ed",
    "rcas": "35f9cb0f301653b64a28fd3a0f2a6509abc4a95b565f10d2251711c0ff0c349a",
}


def word(token, packed=False):
    if re.fullmatch(r"[sv]\d+|exec_lo|vcc_lo", token):
        return token
    if token == "src_scc":
        return "scc"
    if "." in token:
        fmt = "<e" if packed else "<f"
        value = int.from_bytes(struct.pack(fmt, float(token)), "little")
    else:
        value = int(token, 0) & 0xffffffff
    return f"0x{value:x}u"


def registers(token):
    match = re.fullmatch(r"([vs])\[(\d+):(\d+)\]", token)
    if match:
        prefix, start, end = match.groups()
        return [prefix + str(i) for i in range(int(start), int(end)+1)]
    return re.findall(r"[sv]\d+", token)


def assign(names, values):
    if len(names) != len(values):
        raise ValueError(f"Register count mismatch: {names} {values}")
    return " ".join(f"{n} = {v};" for n, v in zip(names, values))


# Operands remain raw DWORDs across integer, IEEE float and packed-half instructions.
OPS = {
    "mulk_i32": "{d}*{0}", "sub_nc_u32": "{0}-{1}", "min_u32": "min({0},{1})",
    "mul_hi_u32": "mul_high({0},{1})",
    "add_lshl_u32": "({0}+{1})<<{2}", "cndmask_b32": "{2}?{1}:{0}", "cselect_b32": "scc?{0}:{1}",
    "mov_b32": "{0}", "movk_i32": "{0}", "add_u32": "{0}+{1}", "add_nc_u32": "{0}+{1}",
    "addk_i32": "{d}+{0}", "sub_i32": "{0}-{1}", "subrev_nc_u32": "{1}-{0}",
    "mul_i32": "{0}*{1}", "mul_lo_u32": "{0}*{1}", "mul_u32_u24": "({0}&0xffffff)*({1}&0xffffff)",
    "add3_u32": "{0}+{1}+{2}", "or_b32": "{0}|{1}", "and_b32": "{0}&{1}",
    "xor_b32": "{0}^{1}", "and_or_b32": "({0}&{1})|{2}",
    "lshl_or_b32": "({0}<<{1})|{2}", "lshl4_add_u32": "({0}<<4)+{1}",
    "lshl_add_u32": "({0}<<{1})+{2}", "lshlrev_b32": "{1}<<{0}", "lshrrev_b32": "{1}>>{0}",
    "lshr_b32": "{0}>>({1}&31u)", "lshl_b32": "{0}<<({1}&31u)", "brev_b32": "reversebits({0})",
    "max_i32": "max(asint({0}),asint({1}))", "min_i32": "min(asint({0}),asint({1}))",
    "max_u32": "max({0},{1})", "abs_i32": "abs(asint({0}))",
    "med3_i32": "median_i(asint({0}),asint({1}),asint({2}))",
    "bfe_u32": "bfe_u({0},{1},{2})", "bfe_i32": "bfe_i({0},{1},{2})",
    "cvt_f32_i32": "asuint(float(asint({0})))", "cvt_f32_u32": "asuint(float({0}))",
    "cvt_i32_f32": "cvt_i32_f32({0})", "cvt_u32_f32": "cvt_u32_f32({0})",
    "cvt_pk_u8_f32": "insert_byte({2},{1},uint(round(clamp(asfloat({0}),0.0,255.0))))",
    "dot4_i32_i8": "idot({0},{1},{2})", "dot4c_i32_i8": "idot({0},{1},{d})",
    "pack_b32_f16": "({0}&65535u)|({1}<<16)",
}

SALU_SCC = {
    "add_u32": "sadd_u({0},{1},scc)", "lshl4_add_u32": "slshl4_add_u({0},{1},scc)",
    "addk_i32": "sadd_i({d},{0},scc)", "sub_i32": "ssub_i({0},{1},scc)",
    "min_u32": "smin_u({0},{1},scc)", "max_u32": "smax_u({0},{1},scc)",
    "min_i32": "smin_i({0},{1},scc)", "max_i32": "smax_i({0},{1},scc)",
    "abs_i32": "sabs_i({0},scc)",
}
SALU_NZ_SCC = {"and_b32", "or_b32", "xor_b32", "lshl_b32", "lshr_b32", "bfe_u32", "bfe_i32"}

FLOAT = {
    "add_f32": "{0}+{1}", "sub_f32": "{0}-{1}", "mul_f32": "{0}*{1}", "exp_f32": "exp2({0})",
    "log_f32": "log2({0})", "rndne_f32": "round({0})", "rcp_f32": "rcp({0})",
    "min_f32": "min({0},{1})", "max_f32": "max({0},{1})", "med3_f32": "median_f({0},{1},{2})",
    "fma_f32": "mad({0},{1},{2})", "fmamk_f32": "mad({0},{1},{2})", "fmac_f32": "mad({0},{1},{d})",
    "subrev_f32": "{1}-{0}", "sqrt_f32": "sqrt({0})", "rsq_f32": "rsqrt({0})",
    "floor_f32": "floor({0})", "ceil_f32": "ceil({0})", "trunc_f32": "trunc({0})", "fract_f32": "(frac({0}) >= 1.0f ? asfloat(0x3f7fffffu) : frac({0}))",
}
# Source count of the operations added with the clamp modifier. A wrong count is refused
# instead of silently dropping or inventing a source.
FLOAT_SOURCES = {"subrev_f32": 2, "sqrt_f32": 1, "rsq_f32": 1, "floor_f32": 1, "ceil_f32": 1, "trunc_f32": 1, "fract_f32": 1}
# VOP3 output multipliers, as the factor applied to the binary32 result. The translation is a plain
# multiply by a power of two, which is exact for normal results. The hardware output modifier also
# flushes denormals and quiets NaN differently from a multiply; neither is modelled here.
OMOD = {"mul:2": "2.0", "mul:4": "4.0", "div:2": "0.5"}


def packed_op(op, dest, inputs, mods):
    if op == "pk_fmac_f16":
        inputs = [*inputs, dest]
    opts = {}
    clamped = False
    for token in mods.split():
        if token == "clamp":
            if clamped:
                raise ValueError(f"Duplicate packed modifier: {token}")
            clamped = True
        elif match := re.fullmatch(r"(op_sel|op_sel_hi|neg_lo|neg_hi):\[([01](?:,[01])*)\]", token):
            key, raw = match.groups()
            values = [int(v) for v in raw.split(",")]
            if key in opts or len(values) != len(inputs):
                raise ValueError(f"Unsupported packed modifier: {token}")
            opts[key] = values
        else:
            raise ValueError(f"Unsupported packed modifier: {token}")
    operations = {"add": "{0}+{1}", "sub": "{0}-{1}", "min": "min({0},{1})",
                  "max": "max({0},{1})", "mul": "{0}*{1}", "fma": "mad({0},{1},{2})",
                  "fmac": "mad({0},{1},{2})"}
    _, operation, kind = op.split("_")
    if clamped and kind != "f16":
        raise ValueError(f"Unsupported clamp modifier on {op}")
    result = []
    for lane in (0, 1):
        args = []
        select = opts.get(("op_sel", "op_sel_hi")[lane], [lane]*len(inputs))
        negate = opts.get(("neg_lo", "neg_hi")[lane], [0]*len(inputs))
        for token, high, neg in zip(inputs, select, negate):
            arg = f"(({word(token, packed=True)} >> {16*high}) & 65535u)"
            if kind == "f16":
                arg = f"f16tof32({arg})"
            args.append(("-" if neg else "") + arg)
        expr = operations[operation].format(*args)
        if clamped:
            expr = f"saturate({expr})"
        result.append(f"f32tof16({expr})" if kind == "f16" else f"(({expr})&65535u)")
    return f"{dest} = {result[0]} | ({result[1]}<<16);"


def alu(mn, tokens, mods):
    op = re.sub(r"_e(?:32|64)$", "", mn[2:])
    dest, *inputs = tokens
    if op.startswith("pk_"):
        return packed_op(op, dest, inputs, mods)
    if op.endswith("_sdwa") or op == "rcp_f16":
        def high(field):
            return int(f"{field}:WORD_1" in mods)
        args = [f"(({reg} >> {16*high(f'src{i}_sel')}) & 65535u)" for i, reg in enumerate(inputs)]
        expr = args[0]
        if op.startswith("rcp"):
            expr = f"f32tof16(rcp(f16tof32({expr})))"
        elif op.startswith("xor"):
            expr += "^" + args[1]
        elif op != "mov_b32_sdwa":
            raise ValueError(op)
        lane = high("dst_sel")
        return f"{dest} = ({dest}&{(65535, 0xffff0000)[1-lane]}u) | (({expr})<<{16*lane});"
    if op == "swap_b32":
        return f"{{ uint t={dest}; {dest}={inputs[0]}; {inputs[0]}=t; }}"
    if op == "fma_mix_f32":
        if inputs[2] != "neg(0)" or "op_sel_hi:[0,1,0]" not in mods:
            raise ValueError("Unexpected mixed-precision operands")
        shift = 16 if "op_sel:[0,1,0]" in mods else 0
        return f"{dest}=asuint(asfloat({inputs[0]})*f16tof32({inputs[1]}>>{shift}));"
    if op in ("mov_b64", "lshrrev_b64"):
        if op == "lshrrev_b64" and inputs[0] != "0":
            raise ValueError("Unsupported 64-bit shift")
        a, b = ["0", "0"] if inputs[-1] == "0" else registers(inputs[-1])
        x, y = registers(dest)
        return f"{{ uint2 t=uint2({a},{b}); {x}=t.x; {y}=t.y; }}"
    if mn in ("s_movk_i32", "s_addk_i32", "s_mulk_i32"):
        if len(inputs) != 1:
            raise ValueError(f"Expected one SOPK immediate for {mn}")
        args = [f"0x{simm16(inputs[0]):x}u"]
    else:
        args = list(map(word, inputs))
    if op == "dot4_i32_i8" and mods:
        if mods == "op_sel_hi:[1,0,1]":
            args[1] = f"(({args[1]}&65535u)*65537u)"
        elif mods == "op_sel:[0,1,0] op_sel_hi:[1,0,1]":
            args[1] = f"(({args[1]}<<16)|({args[1]}>>16))"
        else:
            raise ValueError(mods)
    if op in ("bfe_u32", "bfe_i32"):
        args = [args[0], args[1], f"(({args[1]}>>16)&127u)" if mn.startswith("s_") else f"({args[2]}&31u)"]
    if op in FLOAT:
        # clamp is a VOP3 output modifier; it saturates the binary32 result to [0, 1]
        # (NaN results are not modelled) and excludes the output multiplier.
        if mods not in ("", "clamp", *OMOD) or (mods == "clamp" and mn.endswith("_e32")):
            raise ValueError(f"Unsupported float modifier for {mn}: {mods}")
        # mul:2 on an e32 spelling stays accepted as before; mul:4 and div:2 are VOP3-only.
        if mods in ("mul:4", "div:2") and mn.endswith("_e32"):
            raise ValueError(f"Unsupported float modifier for {mn}: {mods}")
        if len(args) != FLOAT_SOURCES.get(op, len(args)):
            raise ValueError(f"Expected {FLOAT_SOURCES[op]} sources for {mn}")
        expr = FLOAT[op].format(*[f"asfloat({a})" for a in args], d=f"asfloat({dest})")
        if mods in OMOD:
            expr = f"{OMOD[mods]}*({expr})"
        elif mods == "clamp":
            expr = f"saturate({expr})"
        return f"{dest} = asuint({expr});"
    if mods and op != "dot4_i32_i8":
        raise ValueError(f"Unsupported modifier for {mn}: {mods}")
    if mn.startswith("s_") and op in SALU_SCC:
        return f"{dest} = {SALU_SCC[op].format(*args, d=dest)};"
    stmt = f"{dest} = {OPS[op].format(*args, d=dest)};"
    if mn.startswith("s_") and op in SALU_NZ_SCC:
        stmt += f" scc = {dest} != 0u;"
    return stmt


# An 8x4 active tile uses an aligned 8x8 output tensor. Exposure is one
# and the sharpen flag at b1+0x5c is set.
PARAMS = {0x10: [8,4,8,8], 0x28: [8,8], 0x1fc0: [1.0], 0x1fc8: [4,1.0,1.0,1],
          0x1ff8: [1.0,1.0,0,0], 0x26f0: [1.0,1.0],
          0x178: [0x38003800], 0x180: [1.0], 0x188: [1.0]}


def image_op(name, mn, tokens, mods):
    if mn not in ("image_load", "image_store"):
        raise ValueError(f"Unsupported image access: {mn}")
    data, addr, desc = tokens
    regs = registers(data)
    xy = registers(addr)
    coord = f"uint2({','.join(xy)})" if len(xy) == 2 else f"uint2({xy[0]} & 65535u, {xy[0]}>>16)"
    if mn == "image_load":
        if name == "rcas":
            if desc != "s[24:31]" or "dmask:0x7" not in mods or "d16" not in mods:
                raise ValueError("Unexpected RCAS load")
            return "{ uint3 h=f32tof16(load16(" + coord + ").rgb); " + assign(regs, ["h.x|(h.y<<16)", "h.z"]) + " }"
        source = {"s[32:39]": "historyInput", "s[56:63]": "sceneInput"}[desc]
        return "{ uint3 t=asuint(" + source + ".Load3(12*(" + coord + ".y*8+" + coord + ".x))); " + assign(regs, ["t.x","t.y","t.z"]) + " }"
    value = f"asfloat(uint4({','.join(regs)}))"
    if name == "rcas":
        if desc != "s[16:23]":
            raise ValueError("Unexpected RCAS output")
        return f"finalOutput[{coord}]={value};"
    target = {"s[40:47]": "recurrent", "s[48:55]": "historyOutput", "s[64:71]": "intermediate"}[desc]
    return f"write16({coord},{value});" if target == "intermediate" else f"{target}[{coord}]={value};"


def literal_operands(code, pc, tokens):
    """LLVM prints a 16-bit literal 0x0000ffff as -1. Keep both encoded halves."""
    sources, = struct.unpack_from("<I", code, pc + 4)
    for i in range(1, len(tokens)):
        if (sources >> (9 * (i-1))) & 511 == 255:
            literal, = struct.unpack_from("<I", code, pc + 8)
            tokens[i] = hex(literal)


def load_scratch(tokens, trace=None):
    dst, addr, _, _ = tokens
    audit = "" if trace is None else f"audit_load(uint2(v1,v2),{trace},{addr},t); "
    return "{ uint4 t = work.Load4(" + addr + "); " + audit + assign(registers(dst), ["t.x","t.y","t.z","t.w"]) + " }"


def predecessor(name, instructions, initializer):
    padding = name == "pass25"
    threads = 32 if padding else 64
    lines = ['#include "ffx_dispatch_io.hlsli"', '[RootSignature(SCRATCH_ROOT)]',
             f'[numthreads({threads},1,1)] void main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {{',
             "uint " + ",".join(f"s{i}=0" for i in range(64)) + ";",
             "uint " + ",".join(f"v{i}=0" for i in range(64)) + ";",
             "uint scc=0,vcc_lo=0,exec_lo=1; v0=lane; s14=group.x; s15=group.y;"]
    for pc, instruction in instructions:
        mn, _, rest = instruction.partition(" ")
        pieces = re.split(r" (?=\w+:|clamp(?: |$))", rest, maxsplit=1)
        args = pieces[0].split(", ")
        mods = pieces[1] if len(pieces) == 2 else ""
        if mn in ("s_endpgm", "s_inst_prefetch", "s_clause", "s_waitcnt") or mn.startswith("s_load_dword"):
            continue
        if not padding and 0xc <= pc <= 0x20:
            if pc == 0xc:
                lines.append("if (s15>539 || v0>959) return;")
            continue
        if mn == "s_buffer_load_dwordx16":
            values = struct.unpack_from("<16I", initializer, int(args[2],0))
            stmt = assign(registers(args[0]), [hex(v)+"u" for v in values])
        elif mn == "s_buffer_load_dwordx2":
            if not padding or args[2] != "0xd0":
                raise ValueError("Unexpected tensor size read")
            stmt = "s8=4; s9=4;"  # 8x4 aligned up to 8x8, then scaled by 1/2.
        elif mn == "buffer_load_dwordx4":
            stmt = load_scratch(args)
        elif mn == "buffer_store_dwordx4":
            stmt = f"work.Store4({args[1]},uint4({','.join(registers(args[0]))}));"
        elif mn.startswith("v_cmp"):
            cond = {"gt": ">", "le": "<=", "ge": ">="}[mn.split("_")[2]]
            cast = "asint" if "_i32_" in mn else "uint"
            a, b = map(word,args[-2:])
            comp = f"{cast}({a}) {cond} {cast}({b})"
            stmt = f"exec_lo = exec_lo && ({comp});" if "cmpx" in mn else f"{args[0]} = {comp};"
        elif mn == "s_cbranch_execz":
            # The inner branch rejoins at the saved mask. Per-lane predication
            # below also handles waves spanning two different border regions.
            if pc == 0x6c:
                continue
            stmt = "if (!exec_lo) return;"
        elif mn == "s_andn2_b32" and args[0] == "exec_lo":
            stmt = f"exec_lo = ({word(args[1])} != 0) && !{word(args[2])};"
        elif mn.startswith("v_add_co_ci"):
            # Both carry outputs are dead before their next use.
            stmt = f"{args[0]} = {word(args[2])}+{word(args[3])}+{word(args[4])};"
        elif mn.startswith("s_cmp_"):
            cmp = {"s_cmp_eq_i32": "==", "s_cmp_lg_i32": "!="}[mn]
            stmt = f"scc = {word(args[0])}{cmp}{word(args[1])};"
        else:
            stmt = alu(mn, args, mods)
        if padding and (mn.startswith("v_") and not mn.startswith("v_cmpx") or mn.startswith("buffer_store")):
            stmt = "if (exec_lo) { " + stmt + " }"
        lines.append("    " + stmt)
    return "\n".join([*lines, "}", ""])


def translate(name, instructions, machine_code):
    producer = name == "producer"
    signature = "PRODUCER" if producer else "CONSUMER"
    numthreads = 256 if producer else 64
    lines = ['#include "ffx_dispatch_io.hlsli"', f'[RootSignature({signature}_ROOT)]',
             f'[numthreads({numthreads},1,1)] void main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {{',
             "uint " + ",".join(f"s{i}=0" for i in range(102)) + ";",
             "uint " + ",".join(f"v{i}=0" for i in range(84)) + ";",
             "uint scc=0; v0=lane;", "s14=group.x; s15=group.y;" if producer else "s8=group.x; s9=group.y;"]
    skip_until = 0
    for pc, instruction in instructions:
        if pc < skip_until:
            continue
        mn, _, rest = instruction.partition(" ")
        # Bracketed image coordinates contain a comma and a space.
        rest = re.sub(r"\[(v\d+), (v\d+)\]", r"[\1,\2]", rest.strip())
        pieces = re.split(r" (?=\w+:|clamp(?: |$))", rest, maxsplit=1)
        tokens = pieces[0].split(", ")
        mods = pieces[1] if len(pieces) == 2 else ""
        if mn.startswith("v_pk_") and not mn.startswith("v_pk_fmac_"):
            literal_operands(machine_code, pc, tokens)
        if mn in ("s_endpgm", "s_clause", "s_waitcnt", "s_inst_prefetch") or mn.startswith("s_load_dword"):
            continue  # Descriptor loads map to the D3D12 bindings in image_op.
        if mn.startswith("s_buffer_load"):
            code = assign(registers(tokens[0]), [word(str(x)) for x in PARAMS[pc]])
        elif mn == "buffer_load_dwordx4":
            if pc not in range(0x14c,0x194,8):
                raise ValueError("Unexpected pass 26 scratch read")
            code = load_scratch(tokens,(pc-0x14c)//8)
        elif mn.startswith("image_"):
            code = image_op(name, mn, tokens, mods)
        elif producer and pc == 0xa4:
            # Per-lane form of the two compares, NOR exec mask and branch to endpgm.
            code = "if (v1>=s20 || v2>=s21) return;"
            skip_until = 0xb8
        elif producer and pc in (0x2410, 0x2a3c, 0x40c8, 0x4828, 0x4f84, 0x568c):
            # At most eight lanes survive. The full-wave LDS path is unreachable.
            # With sharpening enabled all four outputs retain their log-domain values.
            if mn != "s_cbranch_scc0":
                raise ValueError("Changed producer branch")
            skip_until = pc + 4 + 4*int(rest)
            code = "scc=0;"
        elif producer and (mn.startswith(("v_readfirstlane", "v_readlane", "s_bcnt")) or
                           (0x231c <= pc < 0x2410 and mn.startswith("s_"))):
            continue  # Scalar wave extents feed only the excluded LDS path.
        elif mn.startswith("s_cmp_"):
            cmp = {"s_cmp_eq_i32": "==", "s_cmp_lg_i32": "!="}[mn]
            code = f"scc = {word(tokens[0])}{cmp}{word(tokens[1])};"
        else:
            code = alu(mn, tokens, mods)
        lines.append("    " + code)
    return "\n".join([*lines, "}", ""])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--llvm", required=True)
    args = parser.parse_args(argv)
    try:
        initializer = (args.directory / "initializer.bin").read_bytes()
        if hashlib.sha256(initializer).hexdigest() != "a54e552ff69a3f7199861417f7ae461d1f67f636ab532c6ec9d8ea5d2963735a":
            raise ValueError("Unknown model initializer")
        programs = {}
        for name, digest in DIGEST.items():
            code = (args.directory / (name + ".bin")).read_bytes()
            if hashlib.sha256(code).hexdigest() != digest:
                raise ValueError(f"Unknown {name} machine code")
            ins = disassemble(code, args.llvm, cpu=b"gfx1013" if name == "rcas" else b"gfx1030", offsets=True)
            programs[name] = ins
            if name == "rcas":
                accesses = [line.split()[0] for _, line in ins if line.startswith("image_")]
                if accesses.count("image_load") != 20 or accesses.count("image_store") != 4 or len(accesses) != 24:
                    raise ValueError("Expected twenty integer texture loads and four output stores")
            shader = predecessor(name, ins, initializer) if name.startswith("pass") else translate(name, ins, code)
            (args.directory / (name + ".hlsl")).write_text(shader, encoding="ascii")
            print(f"{name} instructions={len(ins)}")
        (args.directory / "scratch_reference.h").write_text(make_reference(programs, initializer), encoding="ascii")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
