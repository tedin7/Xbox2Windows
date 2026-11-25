"""Differential, /O2, flag-preservation, mutant, and rejection checks for cpu_bmi2_shift_aot."""

import ctypes
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from iced_x86 import Code, Decoder, Encoder, Formatter, FormatterSyntax, Instruction, MemoryOperand, Register as R
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_bmi2_shift_aot import lift_code, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXPORTS = (
    "op_rorx32",
    "op_rorx32_imm32",
    "op_rorx32_imm37",
    "op_rorx64",
    "op_rorx64_imm64",
    "op_rorx64_imm73",
    "op_shlx32",
    "op_shlx64",
    "op_shrx32",
    "op_shrx64",
    "op_sarx32",
    "op_sarx64",
    "op_rorx32_mem",
    "op_rorx64_mem",
    "op_shlx32_mem",
    "op_shlx64_mem",
    "op_shrx32_mem",
    "op_shrx64_mem",
    "op_sarx32_mem",
    "op_sarx64_mem",
    "op_mem_r8_source",
    "op_flags_preserve32",
    "op_flags_preserve64",
    "op_flags_preserve_mem",
    "op_bmi2_discarded_read",
)

ALTERED_EXPORTS = (
    "op_rorx32",
    "op_rorx64",
    "op_shlx32",
    "op_shlx64",
    "op_shrx32",
    "op_shrx64",
    "op_sarx32",
    "op_sarx64",
)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))
OUT_FILL = b"\x5A" * 16


# --- Independent Python reference model ---

def ref_rorx(val, imm8, bits, *, bad_mask=False):
    mask = (1 << bits) - 1
    v = val & mask
    c = (imm8 & 0xFF) if bad_mask else (imm8 & (bits - 1))
    if c == 0 or c >= bits:
        return v if c == 0 else ((v >> (c % bits)) | (v << (bits - (c % bits)))) & mask
    return ((v >> c) | (v << (bits - c))) & mask


def ref_shlx(val, cnt, bits, *, bad_mask=False):
    mask = (1 << bits) - 1
    c = (cnt & 0x3F) if (bad_mask and bits == 32) else (cnt & (bits - 1))
    if c >= bits:
        return 0
    return ((val & mask) << c) & mask


def ref_shrx(val, cnt, bits, *, bad_mask=False):
    mask = (1 << bits) - 1
    c = (cnt & 0x3F) if (bad_mask and bits == 32) else (cnt & (bits - 1))
    if c >= bits:
        return 0
    return (val & mask) >> c


def ref_sarx(val, cnt, bits, *, logical=False, bad_mask=False):
    mask = (1 << bits) - 1
    c = (cnt & 0x3F) if (bad_mask and bits == 32) else (cnt & (bits - 1))
    if logical:
        return 0 if c >= bits else ((val & mask) >> c)
    sign = 1 << (bits - 1)
    u = val & mask
    s = u - (1 << bits) if (u & sign) else u
    if c >= bits:
        return mask if s < 0 else 0
    return (s >> c) & mask


def ref_cmp_flags(a, b, bits):
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    ua, ub = a & mask, b & mask
    r = (ua - ub) & mask
    cf = 1 if ua < ub else 0
    zf = 1 if r == 0 else 0
    sf = 1 if (r & sign) else 0
    of = 1 if (((ua ^ ub) & (ua ^ r)) & sign) else 0
    pf = 1 if (bin(r & 0xFF).count("1") % 2 == 0) else 0
    packed = (cf << 0) | (zf << 1) | (sf << 2) | (of << 3) | (pf << 4)
    return (cf, zf, sf, of, pf), packed


def ref_reg(name, a, b, *, altered=False):
    if name == "op_rorx32":
        return ref_rorx(a, 7 if altered else 13, 32)
    if name == "op_rorx32_imm32":
        return ref_rorx(a, 32, 32)
    if name == "op_rorx32_imm37":
        return ref_rorx(a, 37, 32)
    if name == "op_rorx64":
        return ref_rorx(a, 11 if altered else 19, 64)
    if name == "op_rorx64_imm64":
        return ref_rorx(a, 64, 64)
    if name == "op_rorx64_imm73":
        return ref_rorx(a, 73, 64)
    if name == "op_shlx32":
        return ref_shrx(a, b, 32) if altered else ref_shlx(a, b, 32)
    if name == "op_shlx64":
        return ref_shrx(a, b, 64) if altered else ref_shlx(a, b, 64)
    if name == "op_shrx32":
        return ref_sarx(a, b, 32) if altered else ref_shrx(a, b, 32)
    if name == "op_shrx64":
        return ref_sarx(a, b, 64) if altered else ref_shrx(a, b, 64)
    if name == "op_sarx32":
        return ref_shrx(a, b, 32) if altered else ref_sarx(a, b, 32)
    if name == "op_sarx64":
        return ref_shrx(a, b, 64) if altered else ref_sarx(a, b, 64)
    raise KeyError(name)


VALUES = (
    0x0000000000000000,
    0x0000000000000001,
    0x0000000000000002,
    0x0000000000000003,
    0x0000000055AA55AA,
    0x000000007FFFFFFF,
    0x0000000080000000,
    0x0000000080000001,
    0x00000000FFFFFFFF,
    0xDEADBEEF00000000,
    0xDEADBEEF80000001,
    0x7FFFFFFFFFFFFFFF,
    0x8000000000000000,
    0x8000000000000001,
    0xFFFFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFFFFF,
    0x0123456789ABCDEF,
    0xFEDCBA9876543210,
)
COUNTS = (0, 1, 2, 5, 7, 15, 31, 32, 33, 63, 64, 65, 95, 127, 255, 0x100, 0xFFFFFFFF, 0x8000000000000020)


def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path):
    """Verify EVEX, legacy shifts, BMI1/other BMI2, ABI, addressing, prefix, and state rejections."""
    ret = Instruction.create(Code.RETNQ)
    mov_rax_rcx = Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.RCX)
    mov_r9_rcx = Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R9, R.RCX)
    cmp_rcx_rdx = Instruction.create_reg_reg(Code.CMP_R64_RM64, R.RCX, R.RDX)

    # Alternate valid encodings accepted cleanly.
    alt_accept = [
        ("rorx32_r11_r8_disp", (Instruction.create_reg_mem_u32(
            Code.VEX_RORX_R32_RM32_IMM8, R.R11D, MemoryOperand(R.R8, displ=4, displ_size=1), 37),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R11), ret),
         "r11 = op_rorx32(ld_1000_1, static_cast<std::uint8_t>(0x25u));"),
        ("sarx64_r10_rdx_neg_disp", (Instruction.create_reg_mem_reg(
            Code.VEX_SARX_R64_RM64_R64, R.R10, MemoryOperand(R.RDX, displ=-8, displ_size=1), R.RCX),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R10), ret),
         "r10 = op_sarx64(ld_1000_1, rcx);"),
        ("shlx32_r9_rcx_r8d", (Instruction.create_reg_reg_reg(
            Code.VEX_SHLX_R32_RM32_R32, R.R9D, R.ECX, R.R8D),
            Instruction.create_reg_reg(Code.MOV_R32_RM32, R.EAX, R.R9D), ret),
         "r9 = op_shlx32((rcx & 0xFFFFFFFFull), (r8 & 0xFFFFFFFFull));"),
        ("shrx64_rax_r9_r10", (mov_r9_rcx, Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R10, R.RDX),
            Instruction.create_reg_reg_reg(Code.VEX_SHRX_R64_RM64_R64, R.RAX, R.R9, R.R10), ret),
         "rax = op_shrx64(r9, r10);"),
    ]
    for label, seq, expected in alt_accept:
        src, count = lift_code(label, 0x1000, encode_seq(*seq))
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if expected not in src:
            raise AssertionError(f"alt {label}: expected {expected!r} in:\n{src}")

    seg_rorx = Instruction.create_reg_mem_u32(Code.VEX_RORX_R64_RM64_IMM8, R.RAX, MemoryOperand(R.RCX), 5)
    seg_rorx.segment_prefix = R.GS
    evex_ins = Instruction.create_reg_reg_reg(Code.EVEX_VPADDD_XMM_K1Z_XMM_XMMM128B32, R.XMM0, R.XMM1, R.XMM2)
    evex_ins.op_mask = R.K1

    unit_cases = [
        # EVEX / XOP
        ("evex_vpaddd", (evex_ins, mov_rax_rcx, ret), "unsupported EVEX instruction"),
        ("xop_bextr", (Instruction.create_reg_reg_u32(Code.XOP_BEXTR_R32_RM32_IMM32, R.EAX, R.ECX, 0x0804), ret),
         "unsupported XOP instruction"),
        # Legacy shifts/rotates (both cl and imm/1 forms)
        ("legacy_shl_cl", (mov_rax_rcx, Instruction.create_reg_reg(Code.SHL_RM32_CL, R.EAX, R.CL), ret),
         "unsupported instruction"),
        ("legacy_shl_imm", (mov_rax_rcx, Instruction.create_reg_u32(Code.SHL_RM32_IMM8, R.EAX, 3), ret),
         "unsupported instruction"),
        ("legacy_shl_1", (mov_rax_rcx, Instruction.create_reg_u32(Code.SHL_RM32_1, R.EAX, 1), ret),
         "unsupported instruction"),
        ("legacy_sar_cl", (mov_rax_rcx, Instruction.create_reg_reg(Code.SAR_RM32_CL, R.EAX, R.CL), ret),
         "unsupported instruction"),
        ("legacy_shr_imm", (mov_rax_rcx, Instruction.create_reg_u32(Code.SHR_RM32_IMM8, R.EAX, 4), ret),
         "unsupported instruction"),
        ("legacy_ror_imm", (mov_rax_rcx, Instruction.create_reg_u32(Code.ROR_RM32_IMM8, R.EAX, 5), ret),
         "unsupported instruction"),
        # BMI1 & other BMI2 instructions
        ("bmi1_andn", (Instruction.create_reg_reg_reg(Code.VEX_ANDN_R32_R32_RM32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        ("bmi1_bextr", (Instruction.create_reg_reg_reg(Code.VEX_BEXTR_R32_RM32_R32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        ("bmi2_mulx", (Instruction.create_reg_reg_reg(Code.VEX_MULX_R32_R32_RM32, R.EAX, R.R9D, R.ECX), ret),
         "unsupported VEX instruction"),
        ("bmi2_pdep", (Instruction.create_reg_reg_reg(Code.VEX_PDEP_R32_R32_RM32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        ("bmi2_pext", (Instruction.create_reg_reg_reg(Code.VEX_PEXT_R32_R32_RM32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        ("bmi2_bzhi", (Instruction.create_reg_reg_reg(Code.VEX_BZHI_R32_RM32_R32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        # Nonvolatile GPRs
        ("nonvol_rbx_dst", (Instruction.create_reg_reg_u32(Code.VEX_RORX_R64_RM64_IMM8, R.RBX, R.RCX, 5), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("nonvol_rsi_src", (Instruction.create_reg_reg_reg(Code.VEX_SHLX_R64_RM64_R64, R.RAX, R.RSI, R.RDX), ret),
         "nonvolatile GPR"),
        ("nonvol_r12_cnt", (Instruction.create_reg_reg_reg(Code.VEX_SARX_R64_RM64_R64, R.RAX, R.RCX, R.R12), ret),
         "nonvolatile GPR"),
        ("nonvol_rsp_dst", (Instruction.create_reg_reg_reg(Code.VEX_SHRX_R64_RM64_R64, R.RSP, R.RCX, R.RDX), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        # Writes to argument GPRs (rcx/rdx/r8)
        ("write_arg_rcx", (Instruction.create_reg_reg_u32(Code.VEX_RORX_R64_RM64_IMM8, R.RCX, R.RDX, 5), mov_rax_rcx, ret),
         "cannot write argument GPR rcx"),
        ("write_arg_edx", (Instruction.create_reg_reg_reg(Code.VEX_SHLX_R32_RM32_R32, R.EDX, R.ECX, R.R8D), mov_rax_rcx, ret),
         "cannot write argument GPR rdx"),
        ("write_arg_r8", (Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R8, R.RCX), mov_rax_rcx, ret),
         "cannot write argument GPR r8"),
        # Undefined GPR / flag reads
        ("undef_gpr_src", (Instruction.create_reg_reg_u32(Code.VEX_RORX_R64_RM64_IMM8, R.RAX, R.R9, 5), ret),
         "undefined GPR r9"),
        ("undef_gpr_cnt", (Instruction.create_reg_reg_reg(Code.VEX_SARX_R64_RM64_R64, R.RAX, R.RCX, R.R10), ret),
         "undefined GPR r10"),
        ("undef_flags_setb", (mov_rax_rcx, Instruction.create_mem(Code.SETB_RM8, MemoryOperand(R.R8)), ret),
         "undefined flags read"),
        ("undef_flags_seto_after_bmi2", (Instruction.create_reg_reg_reg(Code.VEX_SHLX_R64_RM64_R64, R.RAX, R.RCX, R.RDX),
            Instruction.create_mem(Code.SETO_RM8, MemoryOperand(R.R8)), ret), "undefined flags read"),
        ("undef_rax_ret", (Instruction.create_reg_reg_u32(Code.VEX_RORX_R64_RM64_IMM8, R.R9, R.RCX, 3), ret),
         "return with undefined RAX"),
        # Memory addressing & store base restrictions
        ("rip_rel_rorx", (Instruction.create_reg_mem_u32(
            Code.VEX_RORX_R64_RM64_IMM8, R.RAX, MemoryOperand(R.RIP, displ=0x1020, displ_size=4), 3), ret),
         "RIP-relative memory operand"),
        ("sib_idx_shlx", (Instruction.create_reg_mem_reg(
            Code.VEX_SHLX_R64_RM64_R64, R.RAX, MemoryOperand(R.RCX, R.RDX, 8), R.RDX), ret),
         "SIB index memory operand"),
        ("bad_base_rbx", (Instruction.create_reg_mem_reg(
            Code.VEX_SARX_R64_RM64_R64, R.RAX, MemoryOperand(R.RBX), R.RDX), ret),
         "bad base register in memory operand"),
        ("bad_base_rsp", (Instruction.create_reg_mem_reg(
            Code.VEX_SHRX_R32_RM32_R32, R.EAX, MemoryOperand(R.RSP), R.EDX), ret),
         "bad base register in memory operand"),
        ("non_r8_store_mov", (mov_rax_rcx, Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RCX), R.RAX), ret),
         "non-r8 store memory operand"),
        ("non_r8_store_setcc", (cmp_rcx_rdx, mov_rax_rcx, Instruction.create_mem(Code.SETE_RM8, MemoryOperand(R.RDX)), ret),
         "non-r8 store memory operand"),
        # Prefixes, branches, missing/trailing ret
        ("seg_prefix_rorx", (seg_rorx, ret), "unsupported instruction prefix"),
        ("branch_jmp", (mov_rax_rcx, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret),
         "non-leaf branch or call"),
        ("missing_ret", (mov_rax_rcx,), "missing terminal ret"),
        ("after_ret", (mov_rax_rcx, ret, mov_r9_rcx), "unreachable instruction after ret"),
    ]

    raw_cases = [
        ("lock_prefix", bytes.fromhex("f00fb109") + encode_seq(mov_rax_rcx, ret), "unsupported instruction prefix"),
        ("rep_prefix", bytes.fromhex("f3a4") + encode_seq(mov_rax_rcx, ret), "unsupported instruction prefix"),
        ("prefixed_vex_66_rorx", bytes.fromhex("66c4e37bf0c107c3"), "invalid instruction"),
        ("empty_stream", b"", "truncated instruction stream"),
    ]

    for label, seq, reason in unit_cases:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted rejected control")

    for label, raw, reason in raw_cases:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted raw rejected control")

    # End-to-end PE patching and CLI non-traceback rejection controls.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_flags_preserve64")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry

    by_label = {lbl: (seq, reason) for lbl, seq, reason in unit_cases}
    pe_labels = ("evex_vpaddd", "legacy_shl_cl", "bmi1_andn", "bmi2_bzhi", "nonvol_rbx_dst",
                 "write_arg_rcx", "undef_flags_seto_after_bmi2", "non_r8_store_setcc", "seg_prefix_rorx")
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_bmi2_shift_aot.py"
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        patched_path = tmp / "patched.dll"
        out_cpp = tmp / "rejected.cpp"
        for label in pe_labels:
            seq, reason = by_label[label]
            patch = encode_seq(*seq)
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds max_len ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off : file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["op_flags_preserve64"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched_path), "op_flags_preserve64", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 1 or "Traceback" in proc.stderr or "unsupported instruction prefix" not in proc.stderr:
            raise AssertionError(f"CLI rejection unexpected: rc={proc.returncode}, stderr={proc.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure left behind output file")

    print(
        f"rejections passed: {len(alt_accept)} alt + {len(unit_cases)} unit + {len(raw_cases)} raw + "
        f"{len(pe_labels)} PE-patched + CLI"
    )


def bind_exports(dll_path, prefix=""):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = ctypes.c_uint64
        funcs[name] = fn
    return lib, funcs


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def host_has_bmi2(lifted_lib):
    if hasattr(lifted_lib, "host_has_bmi2"):
        lifted_lib.host_has_bmi2.argtypes = []
        lifted_lib.host_has_bmi2.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_bmi2())
    try:
        return " bmi2" in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace")
    except OSError:
        return False


def alloc_guarded(payload, offset=0):
    data = GUARD_PRE + b"\x77" * offset + payload + b"\x77" * (8 - offset) + GUARD_POST
    buf = ctypes.create_string_buffer(data, len(data))
    ptr = ctypes.addressof(buf) + len(GUARD_PRE) + offset
    return buf, ptr, offset, len(payload)


def read_guarded(buf, offset, length, label):
    raw = buf.raw
    if raw[:16] != GUARD_PRE or raw[-16:] != GUARD_POST:
        raise AssertionError(f"guard corrupted around {label}")
    if raw[16 : 16 + offset] != b"\x77" * offset or raw[16 + offset + length : -16] != b"\x77" * (8 - offset):
        raise AssertionError(f"padding corrupted around {label}")
    return raw[16 + offset : 16 + offset + length]


def check_discarded_read(fns, label):
    buf = ctypes.create_string_buffer(struct.pack("<Q", 0x0123456789ABCDEF))
    got = fns["op_bmi2_discarded_read"](0xCAFEBABE12345678, ctypes.addressof(buf), 0)
    if got != 0xCAFEBABE12345678:
        raise AssertionError(f"{label} op_bmi2_discarded_read(valid) = {got:#x}")
    try:
        fns["op_bmi2_discarded_read"](0xCAFEBABE12345678, 0, 0)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} op_bmi2_discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_bmi2_discarded_read(NULL) dropped faulting m64 read")


def verify_positive_suite(native_fns, target_fns, label):
    cases = 0
    # 1) Register forms
    reg_only = ("op_rorx32", "op_rorx32_imm32", "op_rorx32_imm37", "op_rorx64", "op_rorx64_imm64", "op_rorx64_imm73")
    for name in reg_only:
        for a in VALUES:
            want = ref_reg(name, a, 0)
            got_n = native_fns[name](a, 0, 0)
            got_t = target_fns[name](a, 0, 0)
            if got_n != want or got_t != want:
                raise AssertionError(f"{label} {name}({a:#x}): n={got_n:#x} t={got_t:#x} exp={want:#x}")
            cases += 1

    shift_reg = ("op_shlx32", "op_shlx64", "op_shrx32", "op_shrx64", "op_sarx32", "op_sarx64")
    for name in shift_reg:
        for a in VALUES:
            for c in COUNTS:
                want = ref_reg(name, a, c)
                got_n = native_fns[name](a, c, 0)
                got_t = target_fns[name](a, c, 0)
                if got_n != want or got_t != want:
                    raise AssertionError(f"{label} {name}({a:#x}, {c:#x}): n={got_n:#x} t={got_t:#x} exp={want:#x}")
                cases += 1

    # 2) Flat [rcx|rdx|r8 + disp] memory forms (including unaligned offsets)
    rng = random.Random(0xB2541F7)
    mem_inputs = [
        (struct.pack("<4Q", 0x800000017FFFFFFF, 0xFEDCBA9876543210, 0x8000000000000001, 0x0123456789ABCDEF),
         struct.pack("<4Q", 32, 33, 63, 64)),
        (struct.pack("<4Q", 0xFFFFFFFF80000000, 0x0000000100000000, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000),
         struct.pack("<4Q", 0, 1, 31, 65)),
    ] + [(rng.randbytes(32), rng.randbytes(32)) for _ in range(24)]

    mem_specs = (
        ("op_rorx32_mem", lambda a, b: ref_rorx(struct.unpack_from("<I", a, 4)[0], 9, 32)),
        ("op_rorx64_mem", lambda a, b: ref_rorx(struct.unpack_from("<Q", a, 8)[0], 23, 64)),
        ("op_shlx32_mem", lambda a, b: ref_shlx(struct.unpack_from("<I", a, 4)[0], struct.unpack_from("<I", b, 4)[0], 32)),
        ("op_shlx64_mem", lambda a, b: ref_shlx(struct.unpack_from("<Q", a, 8)[0], struct.unpack_from("<Q", b, 8)[0], 64)),
        ("op_shrx32_mem", lambda a, b: ref_shrx(struct.unpack_from("<I", a, 8)[0], struct.unpack_from("<I", b, 0)[0], 32)),
        ("op_shrx64_mem", lambda a, b: ref_shrx(struct.unpack_from("<Q", a, 16)[0], struct.unpack_from("<Q", b, 0)[0], 64)),
        ("op_sarx32_mem", lambda a, b: ref_sarx(struct.unpack_from("<I", a, 12)[0], struct.unpack_from("<I", b, 8)[0], 32)),
        ("op_sarx64_mem", lambda a, b: ref_sarx(struct.unpack_from("<Q", a, 24)[0], struct.unpack_from("<Q", b, 16)[0], 64)),
    )
    for a_bytes, b_bytes in mem_inputs:
        for mis in (0, 3):
            ba, pa, oa, la = alloc_guarded(a_bytes, mis)
            bb, pb, ob, lb = alloc_guarded(b_bytes, mis)
            for name, ref_fn in mem_specs:
                want = ref_fn(a_bytes, b_bytes)
                got_n = native_fns[name](pa, pb, 0)
                got_t = target_fns[name](pa, pb, 0)
                if got_n != want or got_t != want:
                    raise AssertionError(f"{label} {name} mis={mis}: n={got_n:#x} t={got_t:#x} exp={want:#x}")
                cases += 1
            # op_mem_r8_source: stores rorx([rcx], 17) to [r8] then returns sarx([r8], [rdx])
            want_mid = ref_rorx(struct.unpack_from("<Q", a_bytes, 0)[0], 17, 64)
            want_r8 = ref_sarx(want_mid, struct.unpack_from("<Q", b_bytes, 0)[0], 64)
            bo_n, po_n, oo_n, lo_n = alloc_guarded(OUT_FILL, mis)
            bo_t, po_t, oo_t, lo_t = alloc_guarded(OUT_FILL, mis)
            got_n = native_fns["op_mem_r8_source"](pa, pb, po_n)
            got_t = target_fns["op_mem_r8_source"](pa, pb, po_t)
            out_n = read_guarded(bo_n, oo_n, lo_n, "r8_native")
            out_t = read_guarded(bo_t, oo_t, lo_t, "r8_target")
            exp_out = struct.pack("<Q", want_mid) + OUT_FILL[8:]
            if got_n != want_r8 or got_t != want_r8 or out_n != exp_out or out_t != exp_out:
                raise AssertionError(f"{label} op_mem_r8_source mismatch")
            assert read_guarded(ba, oa, la, "a") == a_bytes and read_guarded(bb, ob, lb, "b") == b_bytes
            cases += 1

    # 3) Flag preservation across CMP -> RORX/SHLX/SHRX/SARX -> SETcc (CF/ZF/SF/OF/PF)
    flag_pairs = [
        (0x0000000000000000, 0x0000000000000000),  # ZF=1, PF=1
        (0x0000000000000000, 0x0000000000000001),  # CF=1, SF=1, PF=1
        (0x0000000000000003, 0x0000000000000002),  # all 0 (PF=0)
        (0x0000000080000000, 0x0000000000000001),  # 32-bit OF=1, PF=0
        (0x000000007FFFFFFF, 0x00000000FFFFFFFF),  # 32-bit CF=1, SF=1, OF=1
        (0x8000000000000000, 0x0000000000000001),  # 64-bit OF=1
        (0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),  # 64-bit CF=1, SF=1, OF=1
        (0x0000000000000020, 0x0000000000000020),  # count=32 -> ZF=1
        (0xFFFFFFFFFFFFFFFC, 0x0000000000000004),  # negative SARX witness
    ] + [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(32)]

    seen_flags_32, seen_flags_64 = set(), set()
    for a, b in flag_pairs:
        for bits, name, seen in ((32, "op_flags_preserve32", seen_flags_32),
                                 (64, "op_flags_preserve64", seen_flags_64)):
            f_tuple, f_packed = ref_cmp_flags(a, b, bits)
            seen.add(f_tuple)
            r9 = ref_rorx(a, 5 if bits == 32 else 11, bits)
            r10 = ref_shlx(r9, b, bits)
            r11 = ref_shrx(r10, b, bits)
            want_rax = ref_sarx(r11, b, bits)
            want_buf = bytes(f_tuple) + OUT_FILL[5:8] + struct.pack("<Q", f_packed)
            bo_n, po_n, oo_n, lo_n = alloc_guarded(OUT_FILL)
            bo_t, po_t, oo_t, lo_t = alloc_guarded(OUT_FILL)
            got_n = native_fns[name](a, b, po_n)
            got_t = target_fns[name](a, b, po_t)
            buf_n = read_guarded(bo_n, oo_n, lo_n, f"{name}_n")
            buf_t = read_guarded(bo_t, oo_t, lo_t, f"{name}_t")
            if got_n != want_rax or got_t != want_rax or buf_n != want_buf or buf_t != want_buf:
                raise AssertionError(
                    f"{label} {name}({a:#x}, {b:#x}): rax n={got_n:#x} t={got_t:#x} exp={want_rax:#x}, "
                    f"buf n={buf_n.hex()} t={buf_t.hex()} exp={want_buf.hex()}"
                )
            cases += 1

    # Confirm every individual flag (CF, ZF, SF, OF, PF) took both 0 and 1 in both 32-bit and 64-bit checks.
    for bits, seen in ((32, seen_flags_32), (64, seen_flags_64)):
        for idx, flag_name in enumerate(("CF", "ZF", "SF", "OF", "PF")):
            vals = {t[idx] for t in seen}
            if vals != {0, 1}:
                raise AssertionError(f"{bits}-bit flag {flag_name} did not exercise both 0 and 1: {vals}")

    # op_flags_preserve_mem
    for a_bytes, b_bytes in mem_inputs[:8]:
        ba, pa, _, _ = alloc_guarded(a_bytes)
        bb, pb, _, _ = alloc_guarded(b_bytes)
        r9, r10 = struct.unpack_from("<Q", a_bytes, 0)[0], struct.unpack_from("<Q", b_bytes, 0)[0]
        f_tuple, _ = ref_cmp_flags(r9, r10, 64)
        want_rax = ref_sarx(struct.unpack_from("<Q", a_bytes, 8)[0], r10, 64)
        want_buf = bytes(f_tuple) + OUT_FILL[5:]
        bo_n, po_n, oo_n, lo_n = alloc_guarded(OUT_FILL)
        bo_t, po_t, oo_t, lo_t = alloc_guarded(OUT_FILL)
        got_n = native_fns["op_flags_preserve_mem"](pa, pb, po_n)
        got_t = target_fns["op_flags_preserve_mem"](pa, pb, po_t)
        buf_n = read_guarded(bo_n, oo_n, lo_n, "flags_mem_n")
        buf_t = read_guarded(bo_t, oo_t, lo_t, "flags_mem_t")
        if got_n != want_rax or got_t != want_rax or buf_n != want_buf or buf_t != want_buf:
            raise AssertionError(f"{label} op_flags_preserve_mem mismatch")
        cases += 1

    check_discarded_read(native_fns, "native")
    check_discarded_read(target_fns, label)
    return cases


def check_altered_and_mutants(lifted_fns, alt_native_fns, alt_lifted_fns):
    diverged = {}
    for name in ALTERED_EXPORTS:
        diffs = 0
        counts = (0,) if "rorx" in name else COUNTS
        for a in VALUES:
            for c in counts:
                want_alt = ref_reg(name, a, c, altered=True)
                got_an = alt_native_fns[name](a, c, 0)
                got_al = alt_lifted_fns[name](a, c, 0)
                if got_an != want_alt or got_al != want_alt:
                    raise AssertionError(f"altered {name}({a:#x}, {c:#x}): n={got_an:#x} l={got_al:#x} exp={want_alt:#x}")
                if lifted_fns[name](a, c, 0) != got_al:
                    diffs += 1
        if diffs == 0:
            raise AssertionError(f"ALTER_BMI2_SHIFT failed to diverge on {name}")
        diverged[name] = diffs

    # Mutant models (wrong count mask, logical-for-arithmetic SARX, flag-clobbering BMI2) must fail!
    # 1) 6-bit mask instead of 5-bit mask on 32-bit SHLX/SHRX/SARX (count=32 yields 0/sign instead of val)
    if lifted_fns["op_shlx32"](0x12345678, 32, 0) == ref_shlx(0x12345678, 32, 32, bad_mask=True):
        raise AssertionError("mutant bad_mask on shlx32 survived")
    if lifted_fns["op_shrx32"](0x12345678, 32, 0) == ref_shrx(0x12345678, 32, 32, bad_mask=True):
        raise AssertionError("mutant bad_mask on shrx32 survived")
    if lifted_fns["op_sarx32"](0x80000001, 32, 0) == ref_sarx(0x80000001, 32, 32, bad_mask=True):
        raise AssertionError("mutant bad_mask on sarx32 survived")
    # Unmasked RORX32 with imm8=37 (e.g. UB/non-wrapping shift >= 32 returning 0) must differ from lifted 37 & 31 = 5.
    if lifted_fns["op_rorx32_imm37"](0x12345678, 0, 0) == 0:
        raise AssertionError("mutant bad_mask on rorx32_imm37 survived")
    # 2) Logical right shift instead of arithmetic on negative SARX32 / SARX64
    if lifted_fns["op_sarx32"](0x80000000, 4, 0) == ref_sarx(0x80000000, 4, 32, logical=True):
        raise AssertionError("mutant logical_sarx32 survived")
    if lifted_fns["op_sarx64"](0x8000000000000000, 4, 0) == ref_sarx(0x8000000000000000, 4, 64, logical=True):
        raise AssertionError("mutant logical_sarx64 survived")
    # 3) Flag-clobbering mutant: if BMI2 overwrote ZF=1 when shift result is 0, op_flags_preserve32 would fail
    bo, po, oo, lo = alloc_guarded(OUT_FILL)
    rax = lifted_fns["op_flags_preserve32"](0x00000001, 20, po)
    # 1 < 20 -> CF=1, ZF=0, SF=1, OF=0, PF=0; yet rorx(1,5)=0x08000000, shlx(_,20)=0 -> rax=0!
    assert rax == 0
    buf = read_guarded(bo, oo, lo, "flag_mutant")
    if buf[1] != 0:
        raise AssertionError("flag-preservation witness failed: ZF was clobbered to 1")
    return diverged


def compile_o2_and_verify(original_path, native_fns, cxx_path):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "bmi2_shift_o2.cpp"
        dll_path = tmp / "bmi2_shift_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'bmi2_shift_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_exports(dll_path, "lifted_")
        try:
            return verify_positive_suite(native_fns, o2_fns, "lifted_o2")
        finally:
            o2_fns.clear()
            unload_lib(o2_lib)


def record_covered_forms(original_path):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    formatter = Formatter(FormatterSyntax.INTEL)
    forms = set()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        for name in EXPORTS:
            entry = export_rva(pe, name)
            raw = bounded_code(pe, entry)
            for ins in Decoder(64, raw, ip=entry):
                forms.add(formatter.format_mnemonic(ins).lower())
    out_path = Path(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(word + "\n" for word in sorted(forms)), encoding="utf-8")


def main():
    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)

    original_path, lifted_path, altered_path, altered_lifted_path, cxx_path = (
        Path(p) for p in sys.argv[1:6]
    )
    check_rejections(original_path)

    lifted_lib, lifted_fns = bind_exports(lifted_path, "lifted_")
    if not host_has_bmi2(lifted_lib):
        lifted_fns.clear()
        unload_lib(lifted_lib)
        print("Host lacks BMI2 (CPUID leaf 7 subleaf 0 EBX bit 8); skipping native differential check")
        sys.exit(77)

    orig_lib, orig_fns = bind_exports(original_path)
    alt_lib, alt_fns = bind_exports(altered_path)
    alt_lifted_lib, alt_lifted_fns = bind_exports(altered_lifted_path, "lifted_")
    try:
        lifted_cases = verify_positive_suite(orig_fns, lifted_fns, "lifted")
        diverged = check_altered_and_mutants(lifted_fns, alt_fns, alt_lifted_fns)
        o2_cases = compile_o2_and_verify(original_path, orig_fns, cxx_path)
    finally:
        for fn_map, lib in (
            (orig_fns, orig_lib),
            (lifted_fns, lifted_lib),
            (alt_fns, alt_lib),
            (alt_lifted_fns, alt_lifted_lib),
        ):
            fn_map.clear()
            unload_lib(lib)

    record_covered_forms(original_path)
    print(
        f"bmi2_shift passed: {lifted_cases} lifted + {o2_cases} /O2 cases across {len(EXPORTS)} exports; "
        f"ALTER_BMI2_SHIFT diverged on {diverged}"
    )


if __name__ == "__main__":
    main()
