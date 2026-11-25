"""Specialize a sampled, coverage-only gfx1013 PS for a supplied descriptor snapshot."""

import argparse
import json
import re
import struct
import tempfile
import zlib
from pathlib import Path

from pixel_aot import container_code, disassemble
from resource_inventory import parts


# Coverage comparisons: the compare kills the lanes for which it is true, so
# the generated discard uses the same relation with the scalar limit first.
KILL_RELATIONS = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}
# Both the VOP2 (e32) and the VOP3 (e64) spellings are accepted. Only the plain
# register form is lowered: the e64 encoding also carries clamp, output scale
# and input negate or absolute value, which the lowering does not model.
FORMS = ("e32", "e64")
SAMPLED_PRODUCTS = tuple(
    f"v_mul{kind}_f32_{form}" for kind in ("_legacy", "") for form in FORMS
)
COVERAGE_COMPARES = tuple(
    f"v_cmpx_{name}_f32_{form}" for name in KILL_RELATIONS for form in FORMS
)


# The explicit-level zero form takes the same operands as the implicit one, so
# the same pattern accepts it. It reads mip 0 without a derivative, hence
# SampleLevel with a literal zero rather than Sample.
SAMPLE_CALLS = {
    "image_sample": "Sample(samp,uv)",
    "image_sample_lz": "SampleLevel(samp,uv,0)",
}
# The explicit-level form takes the level as a third address register after the
# two coordinates. The biased form takes the bias as the first address register,
# before them (bias, s, t). Only a register copied from a buffered float is
# modeled for either, so the value is a GPU resource like the other comparison
# inputs.
LEVEL_SAMPLE = "image_sample_l"
BIAS_SAMPLE = "image_sample_b"
EXTRA_SAMPLES = {LEVEL_SAMPLE: "SampleLevel", BIAS_SAMPLE: "SampleBias"}


def plain(bank, *tokens):
    # One unmodified register of the bank, so a modifier suffix or prefix, a
    # special register or an inline constant is rejected rather than ignored.
    return all(re.fullmatch(bank + r"\d+", token) for token in tokens)


def regs(token, bank, width):
    match = re.fullmatch(bank + r"\[(\d+):(\d+)\]", token)
    if not match:
        raise ValueError("Invalid register range")
    lo, hi = map(int, match.groups())
    if hi - lo + 1 != width or (bank == "s" and (lo % min(width, 4) or hi > 105)):
        raise ValueError("Unexpected register alignment or width")
    return [f"{bank}{i}" for i in range(lo, hi + 1)]


def coverage_exit(tail, scalar):
    # For nonempty survivors, expand them to quads, intersect with the original
    # WQM mask, then remove helpers. For no survivors both exits export zero lanes.
    if len(tail) != 11:
        raise ValueError("Unexpected coverage exit extent")
    lines = [line.replace("null,", "null") for _, line in tail]
    pair = r"(s\[\d+:\d+\])"
    keep = re.fullmatch(r"s_andn2_b64 " + pair + r", \1, exec", lines[0])
    jump = re.fullmatch(r"s_cbranch_scc0 (-?\d+)", lines[1])
    quad = re.fullmatch(r"s_wqm_b64 " + pair + ", " + pair, lines[2])
    intersect = re.fullmatch("s_and_b64 exec, " + pair + ", " + pair, lines[3])
    if not (keep and jump and quad and intersect):
        raise ValueError("Unverified coverage branch")
    saved, expanded, original_quad = keep[1], quad[1], intersect[1]
    if (
        len({saved, expanded, original_quad}) != 3
        or quad[2] != saved
        or intersect[2] != expanded
    ):
        raise ValueError("Overlapping or unrelated mask operands")
    for token, kind in ((saved, "entry"), (original_quad, "quad")):
        if [scalar.get(r) for r in regs(token, "s", 2)] != [(kind, 0), (kind, 1)]:
            raise ValueError("Original execution mask was overwritten")
    intersect_live = "s_and_b64 exec, exec, " + saved
    null = "exp null off, off, off, off done vm"
    if tail[1][0] + 4 + 4 * int(jump[1]) != tail[7][0] or lines[4:] != [
        intersect_live,
        null,
        "s_endpgm",
        intersect_live,
        "s_mov_b64 exec, " + saved,
        null,
        "s_endpgm",
    ]:
        raise ValueError("Unverified coverage export or branch target")


def _validate_bindings(bindings):
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Bindings must be a non-empty JSON object")
    for key in ("selector_byte", "descriptors", "words", "interp"):
        if key not in bindings:
            raise ValueError(f"Missing bindings field: {key}")
    descriptors = bindings["descriptors"]
    if (
        not isinstance(descriptors, (list, tuple))
        or not descriptors
        or any(
            not isinstance(entry, (list, tuple)) or len(entry) != 4
            for entry in descriptors
        )
    ):
        raise ValueError(
            "Bindings descriptors must be a non-empty list of 4-item entries"
        )
    words = bindings["words"]
    if not isinstance(words, (list, tuple)) or any(
        not isinstance(entry, (list, tuple)) or len(entry) != 2 for entry in words
    ):
        raise ValueError("Bindings words must be a list of 2-item entries")


def lower(decoded, bindings):
    # Locations are keyed by low table-pointer register and byte offset. A
    # descriptor may refer to a static selector, a desktop CB slice, one SRV,
    # or the fixture's point-clamp sampler.
    _validate_bindings(bindings)
    scalar = {}
    slots = {}
    selector_byte = bindings["selector_byte"]
    if (
        type(selector_byte) != int
        or not 0 <= selector_byte <= 0xFFFFFFFC
        or selector_byte % 4
    ):
        raise ValueError(
            "Selector byte address must be an aligned unsigned word offset"
        )
    for table, offset, kind, value in bindings["descriptors"]:
        width = 32 if kind == "texture" else 16
        if (
            type(table) != int
            or not 0 <= table <= 105
            or type(offset) != int
            or not 0 <= offset <= 0xFFFFFFFF - width + 1
            or offset % 4
            or kind not in ("texture", "cb", "selector", "sampler")
            or type(value) != int
            or not 0
            <= value
            <= {"texture": 1, "cb": 63, "selector": 0xFFFFFFFF, "sampler": 0}[kind]
        ):
            raise ValueError("Invalid descriptor mapping")
        for (reg, start), (other, _) in slots.items():
            if (
                reg == table
                and start < offset + width
                and offset < start + (32 if other == "texture" else 16)
            ):
                raise ValueError("Overlapping descriptors")
        slots[table, offset] = (kind, value)
        scalar[f"s{table}"] = ("table", table)
    for reg, value in bindings["words"]:
        if (
            type(reg) != int
            or not 0 <= reg <= 105
            or type(value) != int
            or not 0 <= value <= 0xFFFFFFFF
            or f"s{reg}" in scalar
        ):
            raise ValueError("Invalid entry word")
        scalar[f"s{reg}"] = value
    interp = bindings["interp"]
    if type(interp) != int or not 0 <= interp <= 105 or f"s{interp}" in scalar:
        raise ValueError("Invalid interpolation entry")
    scalar[f"s{interp}"] = "interpolation"
    vector = {"v0": "i", "v1": "j"}
    mask = "entry"
    pending = set()
    pending_sample = set()
    samples = 0

    def read(token):
        if token == "null":
            return 0
        if token in pending:
            raise ValueError("Scalar load consumed before wait")
        if token.startswith("s"):
            return scalar[token]
        return int(token, 0)

    def write(dest, values, load=False):
        if pending.intersection(dest):
            raise ValueError("Overlapping outstanding scalar load")
        scalar.update(zip(dest, values, strict=True))
        if load:
            pending.update(dest)

    for i, (_, line) in enumerate(decoded):
        op, _, operands = line.partition(" ")
        args = operands.split(", ")
        if op in ("s_nop", "s_inst_prefetch", "v_nop"):
            continue
        if line == "s_waitcnt lgkmcnt(0)":
            pending.clear()
        elif line == "s_waitcnt vmcnt(0)":
            pending_sample.clear()
        elif op == "s_mov_b32":
            write([args[0]], [read(args[1])])
        elif op == "s_mov_b64" and args[1] == "exec":
            write(regs(args[0], "s", 2), [(mask, 0), (mask, 1)])
        elif line == "s_wqm_b64 exec, exec" and mask == "entry":
            mask = "quad"
        elif op in ("s_load_dwordx4", "s_load_dwordx8"):
            count = 4 if op.endswith("x4") else 8
            lo, hi = map(read, regs(args[1], "s", 2))
            offset = read(args[2])
            if (
                not isinstance(lo, tuple)
                or lo[0] != "table"
                or hi != 0
                or type(offset) != int
            ):
                raise ValueError("Unbound descriptor address")
            descriptor = slots[lo[1], offset]
            if (count == 8) != (descriptor[0] == "texture"):
                raise ValueError("Descriptor width mismatch")
            write(
                regs(args[0], "s", count), [(descriptor, j) for j in range(count)], True
            )
        elif op == "s_buffer_load_dword":
            parts = list(map(read, regs(args[1], "s", 4)))
            (kind, value), _ = parts[0]
            if parts != [((kind, value), j) for j in range(4)]:
                raise ValueError("Incomplete buffer descriptor")
            offset = read(args[2])
            if type(offset) != int or offset < 0 or offset % 4:
                raise ValueError("Unresolved buffer offset")
            if kind == "selector" and offset == selector_byte:
                word = value
            elif kind == "cb" and 0 <= value + offset // 4 < 64:
                index = value + offset // 4
                word = ("float", f"params[{index // 4}].{'xyzw'[index % 4]}")
            else:
                raise ValueError("Unbound scalar read")
            write([args[0]], [word], True)
        elif op == "s_lshl_b32":
            value, shift = map(read, args[1:])
            if type(value) != int or type(shift) != int:
                raise ValueError("Descriptor index requires specialization")
            write([args[0]], [(value << (shift & 31)) & 0xFFFFFFFF])
        elif op in ("v_interp_p1_f32_e32", "v_interp_p2_f32_e32"):
            dst, src, attr = args
            if (
                mask != "quad"
                or scalar.get("m0") != "interpolation"
                or attr not in ("attr0.x", "attr0.y")
                or dst in pending_sample
                or src in pending_sample
            ):
                raise ValueError("Invalid interpolation inputs")
            if op.startswith("v_interp_p1") and vector.get(src) == "i":
                vector[dst] = ("partial", attr)
            elif (
                op.startswith("v_interp_p2")
                and vector.get(src) == "j"
                and vector.get(dst) == ("partial", attr)
            ):
                vector[dst] = "uv." + attr[-1]
            else:
                raise ValueError("Unmatched interpolation pair")
        elif op == "v_mov_b32_e32":
            if len(args) != 2 or not (plain("v", args[0]) and plain("s", args[1])):
                raise ValueError("Unsupported move operands")
            source = read(args[1])
            if (
                args[0] in pending_sample
                or not isinstance(source, tuple)
                or source[0] != "float"
            ):
                raise ValueError("Unresolved move source")
            vector[args[0]] = ("level", source[1])
        elif op in SAMPLE_CALLS or op in EXTRA_SAMPLES:
            match = re.fullmatch(
                r"(v\d+), (v\[\d+:\d+\]), (s\[\d+:\d+\]), (s\[\d+:\d+\]) dmask:0x([1248]) dim:SQ_RSRC_IMG_2D",
                operands,
            )
            if not match or mask != "quad" or samples:
                raise ValueError("Unsupported sample")
            explicit = op in EXTRA_SAMPLES
            coords = regs(match[2], "v", 3 if explicit else 2)
            extra = coords[0] if op == BIAS_SAMPLE else coords[-1]
            uv_regs = coords[1:] if op == BIAS_SAMPLE else coords[:2]
            if [vector.get(v) for v in uv_regs] != ["uv.x", "uv.y"]:
                raise ValueError("Unresolved sample coordinates")
            call = SAMPLE_CALLS.get(op)
            if explicit:
                level = vector.get(extra)
                if (
                    extra in pending_sample
                    or not isinstance(level, tuple)
                    or level[0] != "level"
                ):
                    raise ValueError("Unresolved sample level")
                call = f"{EXTRA_SAMPLES[op]}(samp,uv,{level[1]})"
            texture = list(map(read, regs(match[3], "s", 8)))
            descriptor, _ = texture[0]
            if (
                descriptor[0] != "texture"
                or texture != [(descriptor, j) for j in range(8)]
                or list(map(read, regs(match[4], "s", 4)))
                != [(("sampler", 0), j) for j in range(4)]
            ):
                raise ValueError("Unbound texture or sampler")
            channel = int(match[5]).bit_length() - 1
            vector[match[1]] = (
                "sample",
                f"tex{descriptor[1]}.{call}.{'rgba'[channel]}",
            )
            pending_sample.add(match[1])
            samples += 1
        elif op in SAMPLED_PRODUCTS:
            if len(args) != 3 or not (
                plain("v", args[0], args[2]) and plain("s", args[1])
            ):
                raise ValueError("Unsupported multiply operands or modifiers")
            dst, src, v = args
            scale, value = read(src), vector[v]
            if (
                dst in pending_sample
                or v in pending_sample
                or not isinstance(scale, tuple)
                or scale[0] != "float"
                or not isinstance(value, tuple)
                or value[0] != "sample"
            ):
                raise ValueError("Unresolved sampled product")
            # The legacy multiply returns zero when either input is zero; the
            # IEEE multiply keeps NaN for zero times infinity.
            vector[dst] = ("product", scale[1], value[1], op.startswith("v_mul_legacy_"))
        elif op in COVERAGE_COMPARES:
            if len(args) != 2 or not (plain("s", args[0]) and plain("v", args[1])):
                raise ValueError("Unsupported compare operands or modifiers")
            limit, product = read(args[0]), vector[args[1]]
            if (
                pending
                or pending_sample
                or samples != 1
                or mask != "quad"
                or not isinstance(limit, tuple)
                or limit[0] != "float"
                or not isinstance(product, tuple)
                or product[0] != "product"
            ):
                raise ValueError("Unresolved coverage predicate")
            coverage_exit(decoded[i + 1 :], scalar)
            _, scale, sampled, legacy = product
            relation = KILL_RELATIONS[op.split("_")[2]]
            multiply = (
                f"({scale} == 0 || alpha == 0) ? 0.0f : {scale} * alpha"
                if legacy
                else f"{scale} * alpha"
            )
            return f"""Texture2D<float4> tex0 : register(t0);
Texture2D<float4> tex1 : register(t1);
SamplerState samp : register(s0);
cbuffer Constants : register(b0) {{ float4 params[16]; }};
void main(float2 uv : TEXCOORD0) {{
    float alpha = {sampled};
    precise float product = {multiply};
    if ({limit[1]} {relation} product) discard;
}}
"""
        else:
            raise ValueError(f"Unsupported coverage instruction {op}")
    raise ValueError("Missing coverage exit")


def translate(blob, bindings, llvm):
    sections = parts(blob)
    if b"DXIL" not in sections:
        raise ValueError("DXBC container has no DXIL part")
    code = container_code(sections[b"DXIL"], stage=0, llvm=llvm)
    return lower(disassemble(code, llvm, offsets=True), bindings)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", type=Path)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--llvm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        dest = args.output.resolve()
        src = args.container.resolve()
        bindings_path = args.bindings.resolve()
    except OSError as error:
        parser.error(f"Unsupported coverage input: {error}")
    if dest.suffix.lower() != ".hlsl" or dest in (src, bindings_path):
        parser.error("Expected a separate HLSL output")
    tmp_path = None
    try:
        blob = args.container.read_bytes()
        bindings = json.loads(args.bindings.read_text(encoding="utf-8"))
        _validate_bindings(bindings)
        source = translate(blob, bindings, args.llvm)
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
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        struct.error,
        zlib.error,
    ) as error:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        parser.error(f"Unsupported coverage input: {error}")
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
