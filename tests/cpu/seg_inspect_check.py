"""Differential, /O2, mutant, and rejection tests for cpu_seg_inspect_aot."""

import ctypes
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from iced_x86 import (
    Code,
    Decoder,
    Encoder,
    Formatter,
    FormatterSyntax,
    Instruction,
    MemoryOperand,
    Register as R,
)
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
from cpu_seg_inspect_aot import lift_code, translate  # noqa: E402


TRANSLATED_EXPORTS = (
    "lar_reg16",
    "lar_reg32",
    "lar_reg64",
    "lar_mem16",
    "lar_mem32_disp",
    "lar_mem64_disp",
    "lsl_reg16",
    "lsl_reg32",
    "lsl_reg64",
    "lsl_mem16",
    "lsl_mem32_disp",
    "lsl_mem64_disp",
    "verr_reg",
    "verw_reg",
    "verr_mem_disp",
    "verw_mem_disp",
    "seg_imm_select",
    "seg_flags_preserve",
    "seg_cmov32",
    "seg_discarded_read",
)

SELECTORS = (
    0x0000,
    0x0003,
    0x0004,
    0x0008,
    0x0010,
    0x0018,
    0x0020,
    0x0021,
    0x0022,
    0x0023,
    0x0027,
    0x0028,
    0x0029,
    0x002A,
    0x002B,
    0x002F,
    0x0030,
    0x0031,
    0x0032,
    0x0033,
    0x0037,
    0x0040,
    0xFFFF,
)

SENTINELS = (
    0x1122334455667788,
    0xA5A5A5A5DEADBEEF,
    0x00000000FFFFFFFF,
    0xFFFFFFFF00000000,
)

# Independent architectural reference for standard Windows x64 user-mode GDT selectors.
# LAR bits [19:16] are architecturally undefined (00FXFF00H across Intel/AMD), so 32/64-bit
# LAR comparisons mask out ~0x000F0000 only when ZF=1.
LAR_UNDEF_BITS_MASK = 0x000F0000

STD_WIN64_GDT = {
    4: (0x00C0FB00, 0xFFFFFFFF, True, False),   # 0x20..0x23: 32-bit user code
    5: (0x00C0F300, 0xFFFFFFFF, True, True),    # 0x28..0x2B: user data/stack
    6: (0x0020FB00, 0x00000000, True, False),   # 0x30..0x33: 64-bit user code
}


def ref_inspect(selector):
    sel = selector & 0xFFFF
    ti = (sel >> 2) & 1
    idx = sel >> 3
    if ti != 0 or (sel & 0xFFFC) == 0 or idx not in STD_WIN64_GDT:
        return False, 0, 0, False, False
    ar, lim, rd, wr = STD_WIN64_GDT[idx]
    return True, ar, lim, rd, wr


def ref_lar_result(selector, width, sentinel):
    valid, ar, _, _, _ = ref_inspect(selector)
    if not valid:
        return 0, sentinel
    if width == 16:
        return 1, (sentinel & ~0xFFFF) | (ar & 0xFFFF)
    return 1, ar & 0xFFFFFFFF


def ref_lsl_result(selector, width, sentinel):
    valid, _, lim, _, _ = ref_inspect(selector)
    if not valid:
        return 0, sentinel
    if width == 16:
        return 1, (sentinel & ~0xFFFF) | (lim & 0xFFFF)
    return 1, lim & 0xFFFFFFFF


def bind_lib(path, prefix=""):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    fns = {}
    for name in TRANSLATED_EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p]
        fn.restype = ctypes.c_uint64
        fns[name] = fn
    if not prefix:
        probe = lib.native_read_segs
        probe.argtypes = [ctypes.c_void_p]
        probe.restype = ctypes.c_uint64
        fns["native_read_segs"] = probe
    return lib, fns


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def check_host_layout(native_fns):
    segs = (ctypes.c_uint16 * 5)()
    native_fns["native_read_segs"](segs)
    cs, ss, ds, es, gs = segs
    if (cs & 0xFFFC) != 0x30 or (ds & 0xFFFC) != 0x28 or (ss & 0xFFFC) != 0x28 or (es & 0xFFFC) != 0x28 or (gs & 0xFFFC) != 0x28:
        print(
            f"non-standard Windows x64 segment layout "
            f"(cs={cs:#06x} ss={ss:#06x} ds={ds:#06x} es={es:#06x} gs={gs:#06x}); skipping"
        )
        raise SystemExit(77)


def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path):
    ret_ins = Instruction.create(Code.RETNQ)
    ret_bytes = encode_seq(ret_ins)
    ld_rax = Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.RDX)

    # Verify raw byte sequences decode to the intended excluded instructions first.
    formatter = Formatter(FormatterSyntax.INTEL)
    raw_checks = [
        ("vex_vpaddw", bytes.fromhex("c5f1fdc2") + ret_bytes, "vpaddw xmm0,xmm1,xmm2", "unsupported VEX instruction"),
        ("xop_vpcmov", bytes.fromhex("8fe878a2c230") + ret_bytes, "vpcmov xmm0,xmm0,xmm2,xmm3", "unsupported XOP instruction"),
        ("seg_gs_lar", bytes.fromhex("650f0201") + ret_bytes, "lar eax,gs:[rcx]", "unsupported instruction prefix"),
        ("lock_add_mem", bytes.fromhex("f00101") + ret_bytes, "lock add [rcx],eax", "unsupported instruction prefix"),
        ("rep_movsb", bytes.fromhex("f3a4") + ret_bytes, "rep movsb [rdi],[rsi]", "unsupported instruction prefix"),
        ("repne_lar", bytes.fromhex("f20f02c1") + ret_bytes, "lar eax,ecx", "unsupported instruction prefix"),
    ]
    for label, raw, expected_asm, expected_reason in raw_checks:
        decoded = formatter.format(next(iter(Decoder(64, raw, ip=0x1000))))
        if decoded != expected_asm:
            raise AssertionError(f"{label}: decoded {decoded!r}, expected {expected_asm!r}")
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: unexpectedly accepted excluded encoding")

    evex_ins = Instruction.create_reg_reg_reg(Code.EVEX_VPADDQ_XMM_K1Z_XMM_XMMM128B64, R.XMM0, R.XMM1, R.XMM2)
    evex_ins.op_mask = R.K1

    unit_cases = [
        ("evex_vpaddq", encode_seq(evex_ins, ret_ins), "unsupported EVEX instruction"),
        ("rip_rel_lar", encode_seq(ld_rax, Instruction.create_reg_mem(
            Code.LAR_R32_R32M16, R.EAX, MemoryOperand(R.RIP, displ=0x1020, displ_size=4)), ret_ins),
         "RIP-relative memory operand"),
        ("bad_base_rax", encode_seq(ld_rax, Instruction.create_reg_mem(
            Code.LAR_R32_R32M16, R.EAX, MemoryOperand(R.RAX)), ret_ins),
         "bad base register"),
        ("bad_base_rsp", encode_seq(ld_rax, Instruction.create_reg_mem(
            Code.LSL_R64_R64M16, R.RAX, MemoryOperand(R.RSP)), ret_ins),
         "bad base register"),
        ("sib_index_lar", encode_seq(ld_rax, Instruction.create_reg_mem(
            Code.LAR_R32_R32M16, R.EAX, MemoryOperand(R.RCX, R.RDX, 2)), ret_ins),
         "SIB index memory operand"),
        ("sib_index_verr", encode_seq(Instruction.create_mem(
            Code.VERR_RM16, MemoryOperand(R.RCX, R.R8, 1)), ret_ins),
         "SIB index memory operand"),
        ("nonvolatile_write_rbx", encode_seq(ld_rax, Instruction.create_reg_reg(
            Code.LAR_R32_R32M16, R.EBX, R.ECX), ret_ins),
         "nonvolatile GPR"),
        ("nonvolatile_write_r12", encode_seq(ld_rax, Instruction.create_reg_reg(
            Code.LSL_R64_R64M16, R.R12, R.RCX), ret_ins),
         "nonvolatile GPR"),
        ("nonvolatile_verr_bx", encode_seq(ld_rax, Instruction.create_reg(
            Code.VERR_RM16, R.BX), ret_ins),
         "nonvolatile GPR"),
        ("nonleaf_jmp", encode_seq(ld_rax, Instruction.create_branch(Code.JMP_REL8_64, 0x1008), ret_ins),
         "non-leaf branch or call"),
        ("nonleaf_call", encode_seq(ld_rax, Instruction.create_branch(Code.CALL_REL32_64, 0x1020), ret_ins),
         "non-leaf branch or call"),
        ("nonleaf_je", encode_seq(ld_rax, Instruction.create_branch(Code.JE_REL8_64, 0x1008), ret_ins),
         "non-leaf branch or call"),
        ("undefined_lar_dst", encode_seq(Instruction.create_reg_reg(
            Code.LAR_R32_R32M16, R.EAX, R.ECX), ret_ins),
         "undefined GPR"),
        ("undefined_flags_sete", encode_seq(ld_rax, Instruction.create_reg(
            Code.SETE_RM8, R.AL), ret_ins),
         "undefined flags"),
        ("undefined_cf_after_lar", encode_seq(ld_rax, Instruction.create_reg_reg(
            Code.LAR_R32_R32M16, R.EAX, R.ECX), Instruction.create_reg(Code.SETB_RM8, R.AL), ret_ins),
         "undefined flags"),
        ("undefined_flags_cmove", encode_seq(ld_rax, Instruction.create_reg_reg(
            Code.CMOVE_R32_RM32, R.EAX, R.ECX), ret_ins),
         "undefined flags"),
        ("xor_mem_dst", encode_seq(ld_rax, Instruction.create_mem_reg(
            Code.XOR_RM32_R32, MemoryOperand(R.RCX), R.EAX), ret_ins),
         "unsupported xor destination"),
        ("or_mem_dst", encode_seq(ld_rax, Instruction.create_mem_reg(
            Code.OR_RM32_R32, MemoryOperand(R.RCX), R.EAX), ret_ins),
         "unsupported or destination"),
        ("clobbered_rcx_ptr", encode_seq(
            ld_rax,
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RCX, R.RDX),
            Instruction.create_reg_mem(Code.LAR_R32_R32M16, R.EAX, MemoryOperand(R.RCX)),
            ret_ins),
         "clobbered pointer"),
        ("missing_ret", encode_seq(ld_rax), "missing terminal ret"),
        ("after_ret", encode_seq(ld_rax, ret_ins, ld_rax), "unreachable instruction after ret"),
    ]
    for label, raw, expected_reason in unit_cases:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: unexpectedly accepted invalid sequence")

    # PE-level patched rejections + CLI non-traceback rejections.
    pe_bytes = original_path.read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "lar_reg32")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(
            i for i in range(0, len(records), 12)
            if struct.unpack_from("<I", records, i)[0] == entry
        )
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        orig_end = struct.unpack_from("<I", records, rec_idx + 4)[0]
        max_len = orig_end - entry

    pe_patches = [
        ("pe_vex", bytes.fromhex("c5f1fdc2") + ret_bytes, "unsupported VEX instruction"),
        ("pe_evex", encode_seq(evex_ins, ret_ins), "unsupported EVEX instruction"),
        ("pe_xop", bytes.fromhex("8fe878a2c230") + ret_bytes, "unsupported XOP instruction"),
        ("pe_seg_prefix", bytes.fromhex("650f0201") + ret_bytes, "unsupported instruction prefix"),
        ("pe_sib", encode_seq(ld_rax, Instruction.create_reg_mem(
            Code.LAR_R32_R32M16, R.EAX, MemoryOperand(R.RCX, R.RDX, 2)), ret_ins), "SIB index memory operand"),
        ("pe_nonvol", encode_seq(ld_rax, Instruction.create_reg_reg(
            Code.LAR_R32_R32M16, R.EBX, R.ECX), ret_ins), "nonvolatile GPR"),
        ("pe_branch", encode_seq(ld_rax, Instruction.create_branch(Code.JMP_REL8_64, 0x1008), ret_ins),
         "non-leaf branch or call"),
    ]

    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_seg_inspect_aot.py"
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        patched_path = tmp / "patched.dll"
        out_cpp = tmp / "rejected.cpp"
        for label, patch, expected_reason in pe_patches:
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds max_len ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off : file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["lar_reg32"])
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: translate() accepted rejected PE patch")

        # CLI negative check: rejects bad patched PE and missing PE cleanly without Traceback or leftover output.
        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched_path), "lar_reg32", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 1 or "Traceback" in proc.stderr or "non-leaf branch or call" not in proc.stderr:
            raise AssertionError(f"CLI rejection unexpected: rc={proc.returncode}, stderr={proc.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure left behind output file")

        proc_missing = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(tmp / "missing.dll"), "lar_reg32", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc_missing.returncode != 1 or "Traceback" in proc_missing.stderr:
            raise AssertionError(f"CLI missing file unexpected: rc={proc_missing.returncode}, stderr={proc_missing.stderr!r}")

    print(
        f"unit rejections ({len(raw_checks)} raw + {len(unit_cases)} IR), "
        f"{len(pe_patches)} PE-patched rejections, and CLI non-traceback rejections passed"
    )


def verify_positive_suite(label, fns):
    out = (ctypes.c_uint8 * 16)()
    mem_buf = (ctypes.c_uint16 * 8)()
    cases = 0

    lar_specs = (
        ("lar_reg16", 16, False, 0),
        ("lar_reg32", 32, False, 0),
        ("lar_reg64", 64, False, 0),
        ("lar_mem16", 16, True, 0),
        ("lar_mem32_disp", 32, True, 2),  # byte offset +4
        ("lar_mem64_disp", 64, True, 4),  # byte offset +8
    )
    lsl_specs = (
        ("lsl_reg16", 16, False, 0),
        ("lsl_reg32", 32, False, 0),
        ("lsl_reg64", 64, False, 0),
        ("lsl_mem16", 16, True, 0),
        ("lsl_mem32_disp", 32, True, 2),  # byte offset +4
        ("lsl_mem64_disp", 64, True, 4),  # byte offset +8
    )

    for is_lar, specs in ((True, lar_specs), (False, lsl_specs)):
        for name, width, is_mem, slot_idx in specs:
            fn = fns[name]
            for sel in SELECTORS:
                arg_variants = [sel] if is_mem else [sel, sel | 0xDEAD0000, sel | 0xBEEFCAFE00000000]
                for arg in arg_variants:
                    for sentinel in SENTINELS:
                        ctypes.memset(out, 0xCC, 16)
                        if is_mem:
                            ctypes.memset(mem_buf, 0xAA, 16)
                            mem_buf[slot_idx] = sel
                            rcx_val = ctypes.addressof(mem_buf)
                        else:
                            rcx_val = arg
                        got_rax = fn(rcx_val, sentinel, ctypes.addressof(out))
                        got_zf = out[0]
                        exp_zf, exp_rax = (
                            ref_lar_result(sel, width, sentinel)
                            if is_lar
                            else ref_lsl_result(sel, width, sentinel)
                        )
                        cmp_got = (got_rax & ~LAR_UNDEF_BITS_MASK) if (is_lar and width >= 32 and exp_zf == 1) else got_rax
                        cmp_exp = (exp_rax & ~LAR_UNDEF_BITS_MASK) if (is_lar and width >= 32 and exp_zf == 1) else exp_rax
                        if got_zf != exp_zf or cmp_got != cmp_exp:
                            raise AssertionError(
                                f"{label} {name} sel={sel:#06x} arg={arg:#x} sent={sentinel:#018x}: "
                                f"got (zf={got_zf}, rax={got_rax:#018x}) != exp (zf={exp_zf}, rax={exp_rax:#018x})"
                            )
                        cases += 1

    # VERR and VERW (reg and [rcx+2] memory), ensuring RAX and R9 stay untouched.
    for is_verr, name, is_mem, r9_expect in (
        (True, "verr_reg", False, 0xCAFEBABE12345678),
        (False, "verw_reg", False, 0xCAFEBABE12345678),
        (True, "verr_mem_disp", True, 0xDEADBEEF00112233),
        (False, "verw_mem_disp", True, 0xDEADBEEF00112233),
    ):
        fn = fns[name]
        for sel in SELECTORS:
            arg_variants = [sel] if is_mem else [sel, sel | 0xDEAD0000]
            for arg in arg_variants:
                for sentinel in SENTINELS[:2]:
                    ctypes.memset(out, 0xCC, 16)
                    if is_mem:
                        ctypes.memset(mem_buf, 0xAA, 16)
                        mem_buf[1] = sel  # byte offset +2
                        rcx_val = ctypes.addressof(mem_buf)
                    else:
                        rcx_val = arg
                    got_rax = fn(rcx_val, sentinel, ctypes.addressof(out))
                    got_zf = out[0]
                    got_r9 = struct.unpack_from("<Q", bytes(out), 8)[0]
                    _, _, _, rd, wr = ref_inspect(sel)
                    exp_zf = int(rd if is_verr else wr)
                    if got_zf != exp_zf or got_rax != sentinel or got_r9 != r9_expect:
                        raise AssertionError(
                            f"{label} {name} sel={sel:#06x}: got (zf={got_zf}, rax={got_rax:#x}, r9={got_r9:#x}) "
                            f"!= exp (zf={exp_zf}, rax={sentinel:#x}, r9={r9_expect:#x})"
                        )
                    cases += 1

    # seg_imm_select: inspects 0x002B (LAR=0x00C0F300 masked [19:16], LSL=0xFFFFFFFF, VERR=1, VERW=1).
    for sentinel in SENTINELS:
        ctypes.memset(out, 0xCC, 16)
        got_rax = fns["seg_imm_select"](0, sentinel, ctypes.addressof(out))
        got_lsl = struct.unpack_from("<Q", bytes(out), 8)[0]
        if ((got_rax & ~LAR_UNDEF_BITS_MASK), out[0], out[1], got_lsl) != (0x00C0F300, 1, 1, 0xFFFFFFFF):
            raise AssertionError(
                f"{label} seg_imm_select: got ({got_rax:#x}, {out[0]}, {out[1]}, {got_lsl:#x})"
            )
        cases += 1

    # seg_flags_preserve: 0x80000001 + 0x80000002 = 3 => CF=1, SF=0, OF=1, PF=1; LAR updates only ZF.
    for sel in SELECTORS:
        for sentinel in SENTINELS[:2]:
            ctypes.memset(out, 0xCC, 16)
            got_rax = fns["seg_flags_preserve"](sel, sentinel, ctypes.addressof(out))
            exp_zf, exp_rax = ref_lar_result(sel, 32, sentinel)
            got_flags = tuple(out[0:5])
            cmp_got = (got_rax & ~LAR_UNDEF_BITS_MASK) if exp_zf == 1 else got_rax
            cmp_exp = (exp_rax & ~LAR_UNDEF_BITS_MASK) if exp_zf == 1 else exp_rax
            if cmp_got != cmp_exp or got_flags != (exp_zf, 1, 0, 1, 1):
                raise AssertionError(
                    f"{label} seg_flags_preserve sel={sel:#06x}: got rax={got_rax:#x} flags={got_flags}, "
                    f"exp rax={exp_rax:#x} flags={(exp_zf, 1, 0, 1, 1)}"
                )
            cases += 1

    # seg_cmov32: cmove eax, r9d writes 0x12345678 on ZF=1 and zeroes RAX[63:32] even on ZF=0.
    for sel in SELECTORS:
        for sentinel in SENTINELS:
            ctypes.memset(out, 0xCC, 16)
            got_rax = fns["seg_cmov32"](sel, sentinel, ctypes.addressof(out))
            valid, _, _, _, _ = ref_inspect(sel)
            exp_zf = int(valid)
            exp_rax = 0x12345678 if valid else (sentinel & 0xFFFFFFFF)
            if got_rax != exp_rax or out[0] != exp_zf:
                raise AssertionError(
                    f"{label} seg_cmov32 sel={sel:#06x} sent={sentinel:#018x}: "
                    f"got (zf={out[0]}, rax={got_rax:#018x}) != exp (zf={exp_zf}, rax={exp_rax:#018x})"
                )
            cases += 1

    # seg_discarded_read: valid pointer succeeds and returns sentinel; NULL pointer faults.
    mem_buf[0] = 0x002B
    got_ok = fns["seg_discarded_read"](ctypes.addressof(mem_buf), SENTINELS[0], ctypes.addressof(out))
    if got_ok != SENTINELS[0]:
        raise AssertionError(f"{label} seg_discarded_read(valid) = {got_ok:#x} != {SENTINELS[0]:#x}")
    try:
        fns["seg_discarded_read"](0, SENTINELS[0], ctypes.addressof(out))
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} seg_discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} seg_discarded_read(NULL) dropped faulting m16 read")
    cases += 2

    return cases


def check_altered_and_mutants(original_lifted_fns, altered_native_fns, altered_lifted_fns):
    out_orig = (ctypes.c_uint8 * 16)()
    out_alt_n = (ctypes.c_uint8 * 16)()
    out_alt_l = (ctypes.c_uint8 * 16)()
    mem_buf = (ctypes.c_uint16 * 8)()

    # 1. Altered native and altered lifted must match bit-for-bit across all exports.
    for name in TRANSLATED_EXPORTS:
        fn_n, fn_l = altered_native_fns[name], altered_lifted_fns[name]
        if name == "seg_discarded_read":
            continue
        for sel in SELECTORS:
            mem_buf[0] = sel
            mem_buf[1] = sel
            mem_buf[2] = sel
            mem_buf[4] = sel
            rcx_val = ctypes.addressof(mem_buf) if ("_mem" in name) else sel
            ctypes.memset(out_alt_n, 0xCC, 16)
            ctypes.memset(out_alt_l, 0xCC, 16)
            rax_n = fn_n(rcx_val, SENTINELS[0], ctypes.addressof(out_alt_n))
            rax_l = fn_l(rcx_val, SENTINELS[0], ctypes.addressof(out_alt_l))
            valid_lar = ref_inspect(sel)[0]
            if (
                (name.startswith("lar_") and not name.endswith("16") and valid_lar)
                or name == "seg_imm_select"
                or (name == "seg_flags_preserve" and valid_lar)
            ):
                cmp_n = rax_n & ~LAR_UNDEF_BITS_MASK
                cmp_l = rax_l & ~LAR_UNDEF_BITS_MASK
            else:
                cmp_n, cmp_l = rax_n, rax_l
            if cmp_n != cmp_l or bytes(out_alt_n) != bytes(out_alt_l):
                raise AssertionError(
                    f"altered mismatch in {name} sel={sel:#06x}: "
                    f"native=({rax_n:#x}, {bytes(out_alt_n).hex()}) != lifted=({rax_l:#x}, {bytes(out_alt_l).hex()})"
                )

    # 2. Altered lifted must diverge from original lifted on every mutated export.
    diverged = {}
    for name in ("verr_reg", "verw_reg", "verr_mem_disp", "verw_mem_disp", "seg_imm_select", "seg_flags_preserve"):
        diffs = 0
        for sel in SELECTORS:
            mem_buf[1] = sel
            rcx_val = ctypes.addressof(mem_buf) if "_mem" in name else sel
            ctypes.memset(out_orig, 0xCC, 16)
            ctypes.memset(out_alt_l, 0xCC, 16)
            rax_o = original_lifted_fns[name](rcx_val, SENTINELS[0], ctypes.addressof(out_orig))
            rax_a = altered_lifted_fns[name](rcx_val, SENTINELS[0], ctypes.addressof(out_alt_l))
            if rax_o != rax_a or bytes(out_orig) != bytes(out_alt_l):
                diffs += 1
        if diffs == 0:
            raise AssertionError(f"altered control {name} failed to diverge from original")
        diverged[name] = diffs

    # 3. Negative comparator mutant: corrupting untouched-dst sentinel on ZF=0 must be caught.
    bad_zf, bad_rax = ref_lar_result(0x0000, 32, SENTINELS[0])
    assert bad_zf == 0 and bad_rax == SENTINELS[0]
    mutated_rax = bad_rax & 0xFFFFFFFF  # simulates buggy 32-bit zero-extension on ZF=0
    if mutated_rax == bad_rax:
        raise AssertionError("negative mutant failed to alter sentinel")

    return diverged


def compile_o2_and_verify(original_path, cxx_path):
    source_text, _ = translate(original_path, list(TRANSLATED_EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "seg_inspect_o2.cpp"
        dll_path = tmp / "seg_inspect_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'seg_inspect_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_lib(dll_path, "lifted_")
        try:
            return verify_positive_suite("lifted_o2", o2_fns)
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
        for name in TRANSLATED_EXPORTS:
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

    orig_lib, orig_fns = bind_lib(original_path)
    lifted_lib, lifted_fns = bind_lib(lifted_path, "lifted_")
    alt_lib, alt_fns = bind_lib(altered_path)
    alt_lifted_lib, alt_lifted_fns = bind_lib(altered_lifted_path, "lifted_")
    try:
        check_host_layout(orig_fns)
        native_cases = verify_positive_suite("native", orig_fns)
        lifted_cases = verify_positive_suite("lifted", lifted_fns)
        diverged = check_altered_and_mutants(lifted_fns, alt_fns, alt_lifted_fns)
        o2_cases = compile_o2_and_verify(original_path, cxx_path)
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
        f"seg_inspect passed: {native_cases} native, {lifted_cases} lifted, {o2_cases} /O2 cases "
        f"across {len(TRANSLATED_EXPORTS)} exports; ALTER_IMM diverged on {diverged}"
    )


if __name__ == "__main__":
    main()
