"""Specialize a two-table, depth-only gfx1013 PS for an explicit Windows binding."""

import argparse
import re
import struct
import tempfile
import zlib
from pathlib import Path

from pixel_aot import container_code, disassemble
from resource_inventory import parts


# Plain binary32 VOP2 forms. src0 may be a constant, scalar word or VGPR; vsrc1 is
# always a VGPR. subrev swaps the operands. The fixtures keep values finite and
# normal, so NaN and subnormal behavior is not claimed.
BINARY_F32 = {
    "v_add_f32_e32": lambda x, y: f"({x} + {y})",
    "v_sub_f32_e32": lambda x, y: f"({x} - {y})",
    "v_subrev_f32_e32": lambda x, y: f"({y} - {x})",
    "v_mul_f32_e32": lambda x, y: f"({x} * {y})",
    "v_min_f32_e32": lambda x, y: f"min({x}, {y})",
    "v_max_f32_e32": lambda x, y: f"max({x}, {y})",
}


# One-source binary32 e32 forms with a VGPR, constant or scalar word source. round is round-half-even, like
# v_rndne_f32. The result of rcp, rsq, sqrt, exp and log is whatever the host computes: a caller may only rely on inputs
# whose exact result is an integer, and a zero or negative operand (or an overflow) is not claimed. Source modifiers, clamp,
# output modifiers and a second source stay rejected; the e64 spelling is accepted only without modifiers (see plain).
UNARY_F32 = {
    "v_rcp_f32_e32": "(1.0f / {x})",
    "v_rsq_f32_e32": "rsqrt({x})",
    "v_sqrt_f32_e32": "sqrt({x})",
    "v_floor_f32_e32": "floor({x})",
    "v_trunc_f32_e32": "trunc({x})",
    "v_rndne_f32_e32": "round({x})",
    "v_ceil_f32_e32": "ceil({x})",
    "v_exp_f32_e32": "exp2({x})",
    "v_log_f32_e32": "log2({x})",
}


# Ordered binary32 compares that write EXEC with the lanes where the compare holds.
# A lane for which src0 OP vsrc1 holds is discarded. Only the four ordered forms are
# claimed: a NaN operand makes every one of them false, as the HLSL operators do. The
# unordered (n*) forms are true for NaN and stay rejected, as do other widths and the VOP3 compares that write a scalar mask.
CMPX_F32 = {
    "v_cmpx_lt_f32_e32": "<",
    "v_cmpx_le_f32_e32": "<=",
    "v_cmpx_gt_f32_e32": ">",
    "v_cmpx_ge_f32_e32": ">=",
}


# Plain VOP3 spellings. An _e64 instruction of exactly the forms above (the binary32 VOP2 and
# VOP1 math, v_fract_f32, v_mul_legacy_f32, v_fmac_f32 and the four ordered v_cmpx_*_f32
# compares) is the same operation as its _e32 spelling and is rewritten to it before lowering,
# so both spellings give identical HLSL and take the same operand rules (a binary32 VOP2
# vsrc1 stays a VGPR even though VOP3 would allow a scalar or constant there). Rejected VOP3
# fields: clamp, omod (mul:2, div:2, mul:4), the abs ("|v1|") and neg ("-v1", "neg(v1)")
# source modifiers, op_sel and every other suffix. Only a plain register, scalar word or
# constant operand is accepted. A VOP3 compare with a scalar destination (v_cmp_*_e64) and
# every other _e64 instruction are different instructions and are rejected.
E64_FORMS = frozenset(
    (
        *BINARY_F32,
        *UNARY_F32,
        *CMPX_F32,
        "v_fract_f32_e32",
        "v_mul_legacy_f32_e32",
        "v_fmac_f32_e32",
    )
)
PLAIN_OPERAND = r"[vs]\d+|-?\d+(?:\.\d+)?|0x[0-9a-f]+"


def plain(line):
    op, _, operands = line.partition(" ")
    if not op.endswith("_e64"):
        return line
    narrow = op[: -len("_e64")] + "_e32"
    if narrow not in E64_FORMS:
        raise ValueError(f"Unsupported depth instruction: {op}")
    if not all(re.fullmatch(PLAIN_OPERAND, a) for a in operands.split(", ")):
        raise ValueError(f"VOP3 modifier or unsupported operand: {line}")
    return f"{narrow} {operands}"


def lower(decoded, tables):
    if (
        len(tables) != 2
        or len({reg for reg, _ in tables}) != 2
        or any(
            not 0 <= reg <= 105 or offset < 0 or offset % 4 for reg, offset in tables
        )
    ):
        raise ValueError(
            "Expected two distinct table registers and aligned descriptor offsets"
        )
    scalar = {reg: ("table", n) for n, (reg, _) in enumerate(tables)}
    vector = {2: "pos.x", 3: "pos.y", 4: "pos.z"}
    pending = set()
    used = set()
    lines = [
        "cbuffer First : register(b0) { uint4 first[128]; };",
        "cbuffer Second : register(b1) { uint4 second[128]; };",
        "float main(float4 pos : SV_Position) : SV_Depth {",
    ]

    def word(token):
        if token.startswith("s"):
            n = int(token[1:])
            val = scalar.get(n)
            if n in pending or not isinstance(val, str):
                raise ValueError("Unresolved scalar word")
            return val
        return f"{int(token, 0) & 0xFFFFFFFF}u"

    def fp(token):
        if re.fullmatch(r"v\d+", token):
            if int(token[1:]) not in vector:
                raise ValueError("Unwritten vector register")
            return vector[int(token[1:])]
        if re.fullmatch(r"-?\d+\.\d+", token):
            return token + "f"
        return f"asfloat({word(token)})"

    def sgprs(start, stop):
        if start % 2 or stop > 105 or stop < start:
            raise ValueError("Invalid scalar register span")
        return list(range(start, stop + 1))

    index = 0
    for index, (_, line) in enumerate(decoded):
        line = plain(line)
        if line.startswith("v_cmpx_"):
            break
        if re.fullmatch(r"s_(?:inst_prefetch|nop) (?:0x[0-9a-f]+|\d+)", line):
            continue
        if line == "s_waitcnt lgkmcnt(0)":
            pending.clear()
            continue
        if m := re.fullmatch(r"s_mov_b64 s\[(\d+):(\d+)\], exec", line):
            regs = sgprs(*map(int, m.groups()))
            if len(regs) != 2 or pending.intersection(regs):
                raise ValueError("Unverified EXEC save")
            for part, r in enumerate(regs):
                scalar[r] = ("entry", part)
        elif m := re.fullmatch(r"s_mov_b32 s(\d+), (0|s\d+)", line):
            dest, src = int(m[1]), m[2]
            if dest in pending or (src != "0" and int(src[1:]) in pending):
                raise ValueError("Scalar move before wait")
            scalar[dest] = 0 if src == "0" else scalar[int(src[1:])]
        elif m := re.fullmatch(
            r"s_load_dwordx4 s\[(\d+):(\d+)\], s\[(\d+):(\d+)\], (null|0x[0-9a-f]+)",
            line,
        ):
            dest = sgprs(int(m[1]), int(m[2]))
            base = sgprs(int(m[3]), int(m[4]))
            offset = 0 if m[5] == "null" else int(m[5], 0)
            if (
                len(dest) != 4
                or dest[0] % 4
                or len(base) != 2
                or pending.intersection(dest + base)
            ):
                raise ValueError("Unresolved descriptor load")
            ptr = scalar.get(base[0])
            if (
                not isinstance(ptr, tuple)
                or ptr[0] != "table"
                or scalar.get(base[1]) != 0
                or offset != tables[ptr[1]][1]
            ):
                raise ValueError("Unbound descriptor address")
            for part, reg in enumerate(dest):
                scalar[reg] = ("cb", ptr[1], part)
            pending.update(dest)
        elif m := re.fullmatch(
            r"s_buffer_load_dword s(\d+), s\[(\d+):(\d+)\], (null|0x[0-9a-f]+)", line
        ):
            dest, lo, hi = map(int, m.groups()[:3])
            source = sgprs(lo, hi)
            desc = scalar.get(lo)
            offset = 0 if m[4] == "null" else int(m[4], 0)
            if (
                not isinstance(desc, tuple)
                or desc[0] != "cb"
                or len(source) != 4
                or pending.intersection([dest, *source])
                or offset % 4
                or offset >= 2048
                or [scalar.get(r) for r in source]
                != [("cb", desc[1], j) for j in range(4)]
            ):
                raise ValueError("Unbound or pending constant read")
            used.add(desc[1])
            name = ("first", "second")[desc[1]]
            scalar[dest] = f"{name}[{offset // 16}].{'xyzw'[offset // 4 % 4]}"
            pending.add(dest)
        elif m := re.fullmatch(r"s_(add_i32|and_b32) s(\d+), (s\d+), (\d+)", line):
            if int(m[2]) in pending:
                raise ValueError("Pending scalar destination")
            op = "+" if m[1] == "add_i32" else "&"
            expr = f"({word(m[3])} {op} {word(m[4])})"
            scalar[int(m[2])] = expr
        else:
            op, _, arglist = line.partition(" ")
            args = arglist.split(", ")
            if not re.fullmatch(r"v\d+", args[0]):
                raise ValueError(f"Unsupported depth instruction: {op}")
            dest = int(args[0][1:])
            if op == "v_cvt_f32_ubyte0_e32" and len(args) == 2:
                expr = f"float({word(args[1])} & 255u)"
            elif op == "v_mul_legacy_f32_e32" and len(args) == 3:
                x, y = map(fp, args[1:])
                expr = f"(({x} == 0 || {y} == 0) ? 0.0f : {x} * {y})"
            elif op in BINARY_F32 and len(args) == 3 and re.fullmatch(r"v\d+", args[2]):
                x, y = map(fp, args[1:])
                expr = BINARY_F32[op](x, y)
            elif op == "v_fmac_f32_e32" and len(args) == 3:
                if dest not in vector:
                    raise ValueError("Unwritten vector register")
                expr = f"mad({fp(args[1])}, {fp(args[2])}, {vector[dest]})"
            elif op in UNARY_F32 and len(args) == 2:
                expr = UNARY_F32[op].format(x=fp(args[1]))
            elif op == "v_fract_f32_e32" and len(args) == 2:
                # D3D min returns the other operand for a NaN, so a min cap would turn a NaN input
                # into the finite cap; a comparison fails for NaN and passes it through.
                frac = f"frac({fp(args[1])})"
                expr = f"({frac} >= 1.0f ? asfloat(0x3f7fffffu) : {frac})"
            else:
                raise ValueError(f"Unsupported depth instruction: {op}")
            vector[dest] = f"value{index}"
            lines.append(f"    precise float value{index} = {expr};")
    tail = decoded[index:]
    if len(tail) != 9 or used != {0, 1} or pending:
        raise ValueError("Expected two completed buffer reads and depth/discard exits")
    compare, invert, branch, restore, depth, end, empty, null, last = (
        s for _, s in tail
    )
    compare = plain(compare)
    cmp = re.fullmatch(r"(v_cmpx_(?:lt|le|gt|ge)_f32_e32) (s\d+), (v\d+)", compare)
    inv = re.fullmatch(r"s_andn2_b64 (s\[(\d+):(\d+)\]), \1, exec", invert)
    take = re.fullmatch(r"s_cbranch_scc0 (-?\d+)", branch)
    mask = re.fullmatch(r"s_and_b64 exec, (s\[(\d+):(\d+)\]), (s\[\d+:\d+\])", restore)
    exp = re.fullmatch(r"exp mrtz,? (v\d+), off, off, off done vm", depth)
    if not (
        cmp
        and inv
        and take
        and mask
        and exp
        and mask[4] == inv[1]
        and mask[1] != inv[1]
        and tail[2][0] + 4 + 4 * int(take[1]) == tail[6][0]
        and empty == f"s_mov_b64 exec, {inv[1]}"
        and end == last == "s_endpgm"
        and re.fullmatch(r"exp null,? off, off, off, off done vm", null)
    ):
        raise ValueError("Unverified lane mask or branch target")
    for first, final in ((inv[2], inv[3]), (mask[2], mask[3])):
        if [scalar.get(r) for r in sgprs(int(first), int(final))] != [
            ("entry", 0),
            ("entry", 1),
        ]:
            raise ValueError("Expected the original entry mask")
    lines += [
        f"    if ({fp(cmp[2])} {CMPX_F32[cmp[1]]} {fp(cmp[3])}) discard;",
        f"    return {fp(exp[1])};",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _parse_int(token: str) -> int:
    return int(token, 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", type=Path)
    parser.add_argument("--llvm", required=True)
    parser.add_argument(
        "--table",
        action="append",
        nargs=2,
        type=_parse_int,
        required=True,
        metavar=("ENTRY_SGPR", "BYTE_OFFSET"),
        help="b0 then b1",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.table or any(sgpr < 0 or offset < 0 for sgpr, offset in args.table):
        parser.error(
            "Expected non-empty --table pairs with non-negative SGPR and byte offset"
        )
    try:
        dest = args.output.resolve()
        src = args.container.resolve()
    except OSError as exc:
        parser.exit(1, f"{exc}\n")
    if dest.suffix.lower() != ".hlsl" or dest == src:
        parser.error("Choose an HLSL destination distinct from the input")
    tmp_path = None
    try:
        sections = parts(args.container.read_bytes())
        if b"DXIL" not in sections:
            raise ValueError("DXBC container has no DXIL part")
        code = container_code(sections[b"DXIL"], stage=0, llvm=args.llvm)
        source = lower(disassemble(code, args.llvm, offsets=True), args.table)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=dest.parent,
            prefix=f".{dest.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(source)
        tmp_path.replace(dest)
    except (OSError, ValueError, struct.error, zlib.error) as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        parser.exit(1, f"{exc}\n")
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
