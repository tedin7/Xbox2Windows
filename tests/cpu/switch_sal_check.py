"""Differential checks (native vs dev lifted vs /O2 lifted vs Python reference),
mutated-C++ controls, unit/PE rejections, and FORM_COVERAGE_RECORD for SAL (/6) in
tools/cpu_switch_aot.py.
"""

import ctypes
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Decoder, Mnemonic

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva
from cpu_switch_aot import Lifter, translate
from direct_check import GUARDS, MASK, SENTINELS, input_pairs, observe, register_probe

M32 = (1 << 32) - 1
ALL = "o no b ae e ne be a s ns p np l ge le g".split()
DEFINED = "b ae e ne be a s ns p np".split()
MNEMONIC_NAMES = {
    v: k.lower() for k, v in vars(Mnemonic).items() if not k.startswith("_") and isinstance(v, int)
}

ONE = {
    "sal1_8": (8, 1, "rax"),
    "sal1_16": (16, 1, "r9"),
    "sal1_32": (32, 1, "rax"),
    "sal1_64": (64, 1, "r10"),
    "sal1_32_imm1": (32, 1, "rax"),
    "sal1_16m": (16, 1, None),
}
MANY = {
    "saln_8": (8, 3, "rax"),
    "saln_16": (16, 9, None),
    "saln_32": (32, 31, "rax"),
    "saln_64": (64, 13, None),
    "salw_64": (64, 63, "r10"),
    "saln_32m": (32, 5, None),
}
CL_AND_MEM = (
    "sal_reg_cl_8",
    "sal_reg_cl_16",
    "sal_reg_cl_32",
    "sal_reg_cl_64",
    "sal_mem_8_16",
    "sal_mem_32_64",
    "sal_mem_imm_edge",
    "sal_nonvolatile_frame",
)
JUMPS = {
    "jcc_sal64_o": ("x", 64, 1, "o"),
    "jcc_sal8_be": ("y", 8, 3, "be"),
}
WIDTHS = {
    **{name: 8 for name in (*CL_AND_MEM, *ONE, *MANY, *JUMPS, "cmov_sal")},
    "sal_narrow_ret": 1,
}

EXPECTED_SAL_CODES = {
    Code.SAL_RM8_1, Code.SAL_RM8_CL, Code.SAL_RM8_IMM8,
    Code.SAL_RM16_1, Code.SAL_RM16_CL, Code.SAL_RM16_IMM8,
    Code.SAL_RM32_1, Code.SAL_RM32_CL, Code.SAL_RM32_IMM8,
    Code.SAL_RM64_1, Code.SAL_RM64_CL, Code.SAL_RM64_IMM8,
}

REJECTIONS = (
    ("neg_sal_cl_setcc", "a shift by CL may leave the flags unchanged"),
    ("neg_cmp_sal_cl_setcc", "a shift by CL may leave the flags unchanged"),
    ("neg_sal_zero_setcc", "a zero-count shift leaves the flags unchanged"),
    ("neg_sal8_wide_setcc", "a shift by the operand width or more"),
    ("neg_sal16_wide_cmov", "a shift by the operand width or more"),
    ("neg_saln_seto", "OF is undefined after a shift by more than 1"),
    ("neg_saln_cmovg", "OF is undefined after a shift by more than 1"),
    ("neg_saln_jl", "OF is undefined after a shift by more than 1"),
    ("neg_sal_ah", "unsupported register"),
    ("neg_sal_undef", "undefined register r10"),
    ("neg_sal_rsp", "dynamic frame"),
)


def even_parity(value):
    return bin(value & 0xff).count("1") % 2 == 0


def sal_value(bits, count, value):
    mask = (1 << bits) - 1
    masked_count = count & (63 if bits == 64 else 31)
    return ((value & mask) << masked_count) & mask


def sal_flags(bits, count, value):
    """Result and defined condition flags for SAL with 1 <= count < bits."""
    mask = (1 << bits) - 1
    value &= mask
    msb = bits - 1
    result = (value << count) & mask
    carry = (value >> (bits - count)) & 1
    overflow = (((result >> msb) & 1) ^ carry) if count == 1 else None
    zero, sign = result == 0, bool((result >> msb) & 1)
    conditions = {"b": bool(carry), "e": zero, "be": bool(carry) or zero, "s": sign, "p": even_parity(result)}
    if overflow is not None:
        conditions.update({"o": bool(overflow), "l": sign != bool(overflow), "le": zero or sign != bool(overflow)})
    for negated, base in (("no", "o"), ("ae", "b"), ("ne", "e"), ("a", "be"), ("ns", "s"), ("np", "p"),
                          ("ge", "l"), ("g", "le")):
        if base in conditions:
            conditions[negated] = not conditions[base]
    return result, conditions


def merge(register, value, bits):
    if bits >= 32:
        return value & ((1 << bits) - 1)
    low = (1 << bits) - 1
    return (register & ~low | value & low) & MASK


def flag_bytes(conditions, names):
    data = bytearray(b"\xff" * 16)
    for number, name in enumerate(names):
        data[number] = int(conditions[name])
    return int.from_bytes(data[:8], "little"), int.from_bytes(data[8:], "little")


def reference(name, x, y, caller_rax):
    if name == "sal_reg_cl_8":
        r0 = merge(y, sal_value(8, x, y), 1 * 8)
        r1 = r0
        r2 = merge(r0, sal_value(8, 1, r0), 8)
        return r0, (GUARDS[0], r0, r1, r2, GUARDS[1])
    if name == "sal_reg_cl_16":
        r0 = merge(y, sal_value(16, x, y), 16)
        return r0, (GUARDS[0], r0, r0, y, GUARDS[1])
    if name == "sal_reg_cl_32":
        r0 = sal_value(32, x, y)
        return r0, (GUARDS[0], r0, r0, y, GUARDS[1])
    if name == "sal_reg_cl_64":
        r0 = sal_value(64, x, y)
        return r0, (GUARDS[0], r0, r0, y, GUARDS[1])
    if name == "sal_mem_8_16":
        q0 = (y & ~0xff | sal_value(8, x, y)) & MASK
        q1 = (y & ~0xff | sal_value(8, 3, y)) & MASK
        q2 = (y & ~0xffff | sal_value(16, x, y)) & MASK
        rax = merge(y, sal_value(16, 1, y), 16)
        return rax, (GUARDS[0], q0, q1, q2, GUARDS[1])
    if name == "sal_mem_32_64":
        q0 = (y & ~M32 | sal_value(32, x, y)) & MASK
        q1 = sal_value(64, x, y)
        q2 = sal_value(64, 1, y)
        return q2, (GUARDS[0], q0, q1, q2, GUARDS[1])
    if name == "sal_mem_imm_edge":
        q0 = (y & ~0xff | sal_value(8, 0, y)) & MASK
        q1 = (y & ~0xff | sal_value(8, 8, y)) & MASK
        q2 = (y & ~0xffff | sal_value(16, 16, y)) & MASK
        rax = sal_value(32, 32, x)
        return rax, (GUARDS[0], q0, q1, q2, GUARDS[1])
    if name == "sal_nonvolatile_frame":
        q0 = sal_value(32, x, y)
        q1 = sal_value(32, 5, y)
        return q0, (GUARDS[0], q0, q1, y, GUARDS[1])
    if name == "sal_narrow_ret":
        return merge(caller_rax, sal_value(8, x, y), 8), (GUARDS[0], MASK, MASK, MASK, GUARDS[1])
    if name in ONE or name in MANY:
        bits, count, register = (ONE.get(name) or MANY[name])
        result, conditions = sal_flags(bits, count, y)
        low, high = flag_bytes(conditions, ALL if name in ONE else DEFINED)
        stored = merge(y, result, bits) if register else (y & ~((1 << bits) - 1) | result) & MASK
        rax = stored if register else x
        return rax, (GUARDS[0], low, high, stored, GUARDS[1])
    if name in JUMPS:
        source, bits, count, condition = JUMPS[name]
        initial = x if source == "x" else y
        result, conditions = sal_flags(bits, count, initial)
        rax = merge(initial, result, bits)
        if conditions[condition]:
            rax ^= MASK
        return rax, (GUARDS[0], MASK, MASK, MASK, GUARDS[1])
    if name == "cmov_sal":
        result, conditions = sal_flags(32, 1, x)
        first = y if conditions["l"] else result
        stored = (y << 13) & MASK
        _, conditions = sal_flags(64, 13, y)
        rax = stored if conditions["np"] else x
        return rax, (GUARDS[0], first, MASK, stored, GUARDS[1])
    raise AssertionError(f"unknown export {name}")


def sal_inputs():
    values = {0, 1, 2, 3, 0x40, 0x7f, 0x80, 0x81, 0xc0, 0xfe, 0xff, 0x100, 0x3fff, 0x4000, 0x7fff, 0x8000,
              0x8001, 0xc000, 0xfffe, 0xffff, 0x10000, 0x3fffffff, 0x40000000, 0x7fffffff, 0x80000000,
              0x80000001, 0xc0000000, 0xfffffffe, 0xffffffff, 0x100000000, 0x4000000000000000,
              0x7fffffffffffffff, 0x8000000000000000, 0x8000000000000001, 0xc000000000000000,
              0xfffffffffffffffe, MASK, 0x123456789abcdef0, 0xdeadbeefcafebabe, 0x0000000100000080}
    highs = (0, 0xdeadbeef00000000)
    pairs = [(x, (v | high) & MASK) for v in sorted(values) for high in highs for x in (0, 1, 3, 8, 16, 31, 32, 63, 64, 0x5555aaaa33cc0ff0)]
    rng = random.Random(0x53414C)
    pairs.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(128))
    pairs.extend((bit, 1 << bit) for bit in range(64))
    pairs.extend((1, MASK ^ (1 << bit)) for bit in range(64))
    return pairs + input_pairs()


def cl_inputs():
    counts = sorted({0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 0x80, 0xff})
    values = (0, 1, 0x7f, 0x80, 0x81, 0xff, 0x7fff, 0x8000, 0xffff, 0x7fffffff, 0x80000000, 0xffffffff,
              0x123456789abcdef0, 0x8000000000000000, MASK)
    highs = (0, 0xdeadbeef00000000, 0xffffff00)
    pairs = [(c | high, v) for c in counts for high in highs for v in values]
    rng = random.Random(0x53414C43)
    pairs.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(128))
    return pairs


def compile_cpp_dll(cxx_path, cpp_path, dll_path, opt_flag):
    obj_path = dll_path.with_suffix(".obj")
    cmd = [str(cxx_path), "/nologo", opt_flag, "/W4", "/LD", str(cpp_path), f"/Fo{obj_path}", f"/Fe{dll_path}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise AssertionError(f"failed to compile {cpp_path.name}:\n{res.stdout}\n{res.stderr}")


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def check_generated(original_path):
    source, functions = translate(original_path, list(WIDTHS))
    widths = {name: fn.return_width for name, fn in zip(WIDTHS, functions)}
    if widths != WIDTHS:
        raise AssertionError(f"return widths {widths} != {WIDTHS}")
    for name, expected_loads in (("sal_mem_8_16", 3), ("sal_mem_32_64", 4), ("saln_16", 1), ("saln_64", 1), ("cmov_sal", 2)):
        body = source.split(f"lifted_{name}(", 1)[1].split("\n}", 1)[0]
        if body.count("load_always<") != expected_loads or re.search(r"\bload<", body):
            raise AssertionError(f"unexpected load count in {name}:\n{body}")
    sal_codes = set()
    for fn in functions:
        for ins in fn.instructions.values():
            if ins.mnemonic == Mnemonic.SHL:
                raise AssertionError(f"unexpected SHL (/4) instruction in {fn.symbol}: {ins}")
            if ins.mnemonic == Mnemonic.SAL:
                sal_codes.add(ins.code)
    if sal_codes != EXPECTED_SAL_CODES:
        missing = sorted(str(c) for c in EXPECTED_SAL_CODES - sal_codes)
        raise AssertionError(f"missing SAL encodings in translated exports: {missing}")
    return source, functions


def build_mutated_sources(base_source):
    """Create deterministic C++ mutants breaking shift direction, CF mask, OF mask, and CL mask."""
    mutants = {}

    # 1) Mutate left shift to right shift inside lifted_sal_reg_cl_64.
    marker = "lifted_sal_reg_cl_64("
    head, rest = base_source.split(marker, 1)
    body, tail = rest.split("\n}", 1)
    body_dir, n1 = re.subn(r"<<", ">>", body)
    if n1 < 2:
        raise AssertionError(f"shift_dir mutant expected >= 2 replacements, got {n1}")
    mutants["shift_dir"] = head + marker + body_dir + "\n}" + tail

    # 2) Mutate CF shift position inside lifted_sal1_32 (bits - n = 31 -> 0).
    marker = "lifted_sal1_32("
    head, rest = base_source.split(marker, 1)
    body, tail = rest.split("\n}", 1)
    body_cf, n2 = re.subn(r">> 31\b", ">> 0", body, count=1)
    if n2 != 1:
        raise AssertionError(f"cf_mask mutant expected 1 replacement, got {n2}")
    mutants["cf_mask"] = head + marker + body_cf + "\n}" + tail

    # 3) Mutate OF computation inside lifted_sal1_64 to constant 0u.
    marker = "lifted_sal1_64("
    head, rest = base_source.split(marker, 1)
    body, tail = rest.split("\n}", 1)
    body_of, n3 = re.subn(r"\| \([^;]+<< 1\);", "| (0u << 1);", body, count=1)
    if n3 != 1:
        raise AssertionError(f"of_mask mutant expected 1 replacement in:\n{body}")
    mutants["of_mask"] = head + marker + body_of + "\n}" + tail

    # 4) Mutate CL count mask (& 31u -> & 15u) inside lifted_sal_reg_cl_32.
    marker = "lifted_sal_reg_cl_32("
    head, rest = base_source.split(marker, 1)
    body, tail = rest.split("\n}", 1)
    body_cl, n4 = re.subn(r"& 31u", "& 15u", body)
    if n4 < 2:
        raise AssertionError(f"cl_mask mutant expected >= 2 replacements, got {n4}")
    mutants["cl_mask"] = head + marker + body_cl + "\n}" + tail

    return mutants


def verify_differential(probe, native, target, label, imm_pairs, cl_pairs):
    total = 0
    for name, width in WIDTHS.items():
        source, translation = getattr(native, name), getattr(target, "lifted_" + name)
        caller_rax = ctypes.cast(source, ctypes.c_void_p).value
        low = (1 << width * 8) - 1
        pairs = cl_pairs if "_cl_" in name or name in ("sal_mem_8_16", "sal_mem_32_64", "sal_narrow_ret", "sal_nonvolatile_frame") else imm_pairs
        for x, y in pairs:
            want, trace = reference(name, x, y, caller_rax)
            original = observe(probe, source, x, y)
            actual = observe(probe, translation, x, y)
            if original != (want, trace, SENTINELS):
                raise AssertionError(f"original {name}({x:#x}, {y:#x}): {original} != ({want:#x}, {trace})")
            if (actual[0] & low, *actual[1:]) != (want & low, trace, SENTINELS):
                raise AssertionError(f"{label} {name}({x:#x}, {y:#x}): {actual} != ({want:#x} & {low:#x}, {trace})")
            total += 1
    return total


def check_rejections(original_path):
    for name, reason in REJECTIONS:
        try:
            translate(original_path, [name])
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{name}: expected {reason!r}, got {exc}") from exc
        else:
            raise AssertionError(f"accepted negative control {name}")

    for text, raw, reason in (
        ("sal eax, cl", "d3 f0", "a shift by CL may leave the flags unchanged"),
        ("sal al, 8", "c0 f0 08", "a shift by the operand width or more"),
        ("sal eax, 32", "c1 f0 20", "a zero-count shift leaves the flags unchanged"),
        ("sal ah, 1", "d0 f4", "unsupported register"),
    ):
        ins = next(Decoder(64, bytes.fromhex(raw), ip=0x1000))
        try:
            Lifter().lower(ins, sets_flags=(reason != "unsupported register"))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{text}: expected {reason!r}, got {exc}") from exc
        else:
            raise AssertionError(f"accepted unit negative {text}")

    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "sal1_32")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12)
                       if int.from_bytes(records[i:i + 4], "little") == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = int.from_bytes(records[rec_idx + 4:rec_idx + 8], "little") - entry

    pe_patches = (
        ("pe_sal_cl_setcc", bytes.fromhex("89 d0 d3 f0 0f 92 c0 c3"), "a shift by CL may leave the flags unchanged"),
        ("pe_sal_zero_setcc", bytes.fromhex("89 d0 c1 f0 20 0f 94 c0 c3"), "a zero-count shift leaves the flags unchanged"),
        ("pe_sal8_wide_setcc", bytes.fromhex("89 d0 c0 f0 08 0f 92 c0 c3"), "a shift by the operand width or more"),
        ("pe_saln_seto", bytes.fromhex("89 d0 c1 f0 02 0f 90 c0 c3"), "OF is undefined after a shift by more than 1"),
        ("pe_sal_seg_mem", bytes.fromhex("4c 89 c0 65 d0 30 c3"), "unsupported memory operand"),
        ("pe_sal_rep", bytes.fromhex("89 d0 f3 d1 f0 c3"), "unsupported prefix"),
        ("pe_sal_lock", bytes.fromhex("f0 d0 f0 c3"), "invalid"),
    )
    with TemporaryDirectory() as folder:
        patched_path = Path(folder) / "patched.dll"
        for label, patch, reason in pe_patches:
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch exceeds max_len ({len(patch)} > {max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            damaged[pdata_end_off:pdata_end_off + 4] = (entry + len(patch)).to_bytes(4, "little")
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["sal1_32"])
            except ValueError as exc:
                if reason.lower() not in str(exc).lower():
                    raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: PE translate unexpectedly accepted patched bytes")

    print(f"{len(REJECTIONS)} MASM, 4 unit, and {len(pe_patches)} PE-patched rejection controls passed")


def record_covered_forms(functions):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    forms = {MNEMONIC_NAMES[ins.mnemonic] for fn in functions for ins in fn.instructions.values()}
    assert "sal" in forms, sorted(forms)
    out_path = Path(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(word + "\n" for word in sorted(forms)), encoding="utf-8")


def check(probe_path, original_path, lifted_path, cxx_path):
    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)
    source_text, functions = check_generated(original_path)
    probe = register_probe(probe_path)
    native, lifted = ctypes.CDLL(original_path), ctypes.CDLL(lifted_path)
    imm_pairs, cl_pairs = sal_inputs(), cl_inputs()

    total_dev = verify_differential(probe, native, lifted, "lifted_dev", imm_pairs, cl_pairs)

    mutants = build_mutated_sources(source_text)
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        o2_cpp = tmp / "lifted_o2.cpp"
        o2_dll = tmp / "lifted_o2.dll"
        o2_cpp.write_text(source_text, encoding="utf-8")
        compile_cpp_dll(Path(cxx_path), o2_cpp, o2_dll, "/O2")
        o2_lib = ctypes.CDLL(str(o2_dll))
        try:
            total_o2 = verify_differential(probe, native, o2_lib, "lifted_o2", imm_pairs, cl_pairs)
        finally:
            unload_lib(o2_lib)

        for mut_name, mut_src in mutants.items():
            mut_cpp = tmp / f"{mut_name}.cpp"
            mut_dll = tmp / f"{mut_name}.dll"
            mut_cpp.write_text(mut_src, encoding="utf-8")
            compile_cpp_dll(Path(cxx_path), mut_cpp, mut_dll, "/O2")
            mut_lib = ctypes.CDLL(str(mut_dll))
            try:
                try:
                    verify_differential(probe, native, mut_lib, f"mutant_{mut_name}", imm_pairs, cl_pairs)
                except AssertionError as exc:
                    print(f"mutant {mut_name} caught: {exc}")
                else:
                    raise AssertionError(f"mutant {mut_name} unexpectedly passed all checks")
            finally:
                unload_lib(mut_lib)

    check_rejections(original_path)
    record_covered_forms(functions)
    print(f"{total_dev} dev + {total_o2} /O2 differential comparisons passed across all 12 SAL (/6) encodings")


if __name__ == "__main__":
    check(*sys.argv[1:])
