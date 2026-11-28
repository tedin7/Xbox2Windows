"""Differential, /O2-barrier, fault-ordering, mutation, rejection, and AMD CPUID checks
for AMD MONITORX, MWAITX, and RDPRU AOT lifting (`tools/cpu_amd_monitor_rdpru_aot.py`).

Unconditional checks (translator acceptance/rejections, Debug & /O2 lifted execution,
/O2 disassembly anti-CSE/anti-DCE verification, NULL-pointer MONITORX fault ordering,
ALTER_AMD_EXT altered-lifted checks, and mutant controls) run on every host BEFORE
checking AMD CPUID Fn8000_0001:ECX[29] (MWAITX) and Fn8000_0008:EBX[4] (RDPRU).
On non-AMD or pre-Zen2 hosts lacking either feature, the script exits 77 without
writing FORM_COVERAGE_RECORD (matching repository form_coverage conventions).
"""

import ctypes
import os
from pathlib import Path
import struct
import sys

from iced_x86 import (
    Code,
    Decoder,
    Encoder,
    FlowControl,
    Formatter,
    FormatterSyntax,
    Instruction,
    MemoryOperand,
    Register as R,
)
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
import cpu_amd_monitor_rdpru_aot  # noqa: E402
from cpu_amd_monitor_rdpru_aot import lift_code  # noqa: E402

EXPORTS = (
    "op_rdpru_mperf",
    "op_rdpru_aperf",
    "op_rdpru_twice",
    "op_rdpru_unused",
    "op_monitorx_mwaitx_r8",
    "op_monitorx_mwaitx_intbrk",
    "op_monitorx_rcx_rdx",
    "op_add32_rdpru",
    "op_cmp32_mwaitx",
)

ALTERED_EXPORTS = (
    "op_rdpru_mperf",
    "op_rdpru_aperf",
    "op_rdpru_twice",
    "op_rdpru_unused",
    "op_monitorx_mwaitx_r8",
    "op_monitorx_mwaitx_intbrk",
)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))
OUT_BYTES = 64
SENTINEL64 = 0x5A5A5A5A5A5A5A5A

# Pairs exercising all CF/ZF/SF/OF combinations under both ADD and CMP.
FLAG_PAIRS = (
    (0, 0),
    (1, 1),
    (5, 9),
    (9, 5),
    (0x7FFFFFFFFFFFFFFF, 1),
    (0x8000000000000000, 0x8000000000000000),
    (0xFFFFFFFFFFFFFFFF, 1),
    (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
    (0x8000000000000000, 1),
    (1, 0x8000000000000000),
)

FLAG_PAIRS_32 = (
    (0, 0),
    (1, 1),
    (5, 9),
    (9, 5),
    (0x7FFFFFFF, 1),
    (0x80000000, 0x80000000),
    (0xFFFFFFFF, 1),
    (0xFFFFFFFF, 0xFFFFFFFF),
    (0x80000000, 1),
    (1, 0x80000000),
    (0x100000005, 0x200000009),
    (0x900000009, 0x100000005),
)


def add32_flags(a, b):
    a &= MASK32
    b &= MASK32
    r = (a + b) & MASK32
    cf = 1 if (a + b) > MASK32 else 0
    zf = 1 if r == 0 else 0
    sf = 1 if (r >> 31) else 0
    of = 1 if (((~(a ^ b)) & (a ^ r)) >> 31) & 1 else 0
    return cf | (zf << 1) | (sf << 2) | (of << 3)


def cmp32_flags(a, b):
    a &= MASK32
    b &= MASK32
    r = (a - b) & MASK32
    cf = 1 if a < b else 0
    zf = 1 if r == 0 else 0
    sf = 1 if (r >> 31) else 0
    of = 1 if (((a ^ b) & (a ^ r)) >> 31) & 1 else 0
    return cf | (zf << 1) | (sf << 2) | (of << 3)


def add64_flags(a, b):
    a &= MASK64
    b &= MASK64
    r = (a + b) & MASK64
    cf = 1 if (a + b) > MASK64 else 0
    zf = 1 if r == 0 else 0
    sf = 1 if (r >> 63) else 0
    of = 1 if (((~(a ^ b)) & (a ^ r)) >> 63) & 1 else 0
    return cf | (zf << 1) | (sf << 2) | (of << 3)


def cmp64_flags(a, b):
    a &= MASK64
    b &= MASK64
    r = (a - b) & MASK64
    cf = 1 if a < b else 0
    zf = 1 if r == 0 else 0
    sf = 1 if (r >> 63) else 0
    of = 1 if (((a ^ b) & (a ^ r)) >> 63) & 1 else 0
    return cf | (zf << 1) | (sf << 2) | (of << 3)


def alloc_guarded(payload):
    size = 32 + len(GUARD_PRE) + len(payload) + len(GUARD_POST)
    raw = ctypes.create_string_buffer(size)
    base = ctypes.addressof(raw)
    data_addr = (base + len(GUARD_PRE) + 15) & ~15
    pre_off = (data_addr - len(GUARD_PRE)) - base
    data_off = data_addr - base
    post_off = data_off + len(payload)
    raw[pre_off:data_off] = GUARD_PRE
    raw[data_off:post_off] = payload
    raw[post_off:post_off + len(GUARD_POST)] = GUARD_POST
    return raw, pre_off, data_off, post_off, data_addr


def call_export(fn, a_u64, b_u64):
    raw, pre, off, post, addr = alloc_guarded(b"\x5a" * OUT_BYTES)
    ret = fn(a_u64, b_u64, addr)
    assert raw.raw[pre:off] == GUARD_PRE and raw.raw[post:post + 16] == GUARD_POST, "output guards corrupted"
    slots = struct.unpack("<8Q", raw.raw[off:off + OUT_BYTES])
    return ret, slots


def bind(path, prefix):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = ctypes.c_uint64
        funcs[name] = fn
    return lib, funcs


def pin_current_thread():
    """Pin current thread to one logical CPU so TSC brackets are strictly monotonic."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
    proc_mask, sys_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if kernel32.GetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.byref(proc_mask), ctypes.byref(sys_mask)):
        bit = proc_mask.value & -proc_mask.value
        if bit:
            kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), bit)


# --- Translator positive and negative controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    rip = 0x1000
    for ins in instructions:
        rip += enc.encode(ins, rip)
    return enc.take_buffer()


def check_translator():
    ret = Instruction.create(Code.RETNQ)
    rdpru = Instruction.create(Code.RDPRU)
    monitorx = Instruction.create(Code.MONITORXQ)
    mwaitx = Instruction.create(Code.MWAITX)
    xor_ecx = Instruction.create_reg_reg(Code.XOR_R32_RM32, R.ECX, R.ECX)
    xor_edx = Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EDX, R.EDX)
    mov_ecx_0 = Instruction.create_reg_u32(Code.MOV_R32_IMM32, R.ECX, 0)
    mov_ecx_1 = Instruction.create_reg_u32(Code.MOV_R32_IMM32, R.ECX, 1)
    mov_ecx_2 = Instruction.create_reg_u32(Code.MOV_R32_IMM32, R.ECX, 2)
    mov_eax_0 = Instruction.create_reg_u32(Code.MOV_R32_IMM32, R.EAX, 0)
    lea_rax_rcx = Instruction.create_reg_mem(Code.LEA_R64_M, R.RAX, MemoryOperand(R.RCX))
    lea_rax_rdx = Instruction.create_reg_mem(Code.LEA_R64_M, R.RAX, MemoryOperand(R.RDX))
    lea_rax_r8 = Instruction.create_reg_mem(Code.LEA_R64_M, R.RAX, MemoryOperand(R.R8))
    st_r8_rax = Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8), R.RAX)

    # Positive acceptance checks.
    positive = [
        ("rdpru_0", (xor_ecx, rdpru, st_r8_rax, ret), "op_rdpru_read(0u)"),
        ("rdpru_1", (mov_ecx_1, rdpru, st_r8_rax, ret), "op_rdpru_read(1u)"),
        ("monitorx_rcx", (lea_rax_rcx, xor_ecx, xor_edx, monitorx, ret), "op_monitorx(rax);"),
        ("monitorx_rdx", (lea_rax_rdx, xor_ecx, xor_edx, monitorx, ret), "op_monitorx(rax);"),
        ("monitorx_r8_mwaitx_0", (lea_rax_r8, xor_ecx, xor_edx, monitorx, mov_eax_0, mov_ecx_0, mwaitx, ret), "op_mwaitx("),
        ("mwaitx_intbrk_1", (mov_eax_0, mov_ecx_1, mwaitx, ret), "op_mwaitx(static_cast<std::uint32_t>(rax), 1u);"),
    ]
    for label, seq, needle in positive:
        src, _ = lift_code(label, 0x1000, encode_seq(*seq))
        assert needle in src, f"{label}: missing {needle!r}"

    # Negative rejection controls.
    rep_ret = Instruction.create(Code.RETNQ)
    rep_ret.has_rep_prefix = True
    fs_store = Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8), R.RAX)
    fs_store.segment_prefix = R.FS

    rejections = [
        ("legacy_monitor", (lea_rax_rcx, xor_ecx, xor_edx, Instruction.create(Code.MONITORQ), ret),
         "unsupported legacy monitor/mwait/rdtsc instruction"),
        ("legacy_mwait", (mov_eax_0, mov_ecx_0, Instruction.create(Code.MWAIT), ret),
         "unsupported legacy monitor/mwait/rdtsc instruction"),
        ("legacy_rdtsc", (Instruction.create(Code.RDTSC), ret),
         "unsupported legacy monitor/mwait/rdtsc instruction"),
        ("legacy_rdtscp", (Instruction.create(Code.RDTSCP), ret),
         "unsupported legacy monitor/mwait/rdtsc instruction"),
        ("monitorx_addr32", (lea_rax_rcx, xor_ecx, xor_edx, Instruction.create(Code.MONITORXD), ret),
         "unsupported non-64-bit monitorx address size"),
        ("vex_vmovdqu", (Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), mov_eax_0, ret),
         "unsupported VEX encoding"),
        ("evex_vmovdqu32", (Instruction.create_reg_mem(Code.EVEX_VMOVDQU32_XMM_K1Z_XMMM128, R.XMM0, MemoryOperand(R.RCX)), mov_eax_0, ret),
         "unsupported EVEX encoding"),
        ("sse_movaps", (Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), mov_eax_0, ret),
         "unsupported instruction"),
        ("nonvol_rbx", (Instruction.create_reg_u32(Code.MOV_R32_IMM32, R.EBX, 0), mov_eax_0, ret),
         "nonvolatile GPR"),
        ("nonvol_r12", (Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R12), ret),
         "nonvolatile GPR"),
        ("nonvol_mem_rbx", (mov_eax_0, Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RBX), R.RAX), ret),
         "nonvolatile GPR base"),
        ("rdpru_unproven_ecx", (rdpru, ret),
         "non-constant/unproven ECX for rdpru"),
        ("rdpru_oor_ecx_2", (mov_ecx_2, rdpru, ret),
         "out-of-range ECX=0x2 for rdpru"),
        ("rdpru_oor_ecx_neg1", (Instruction.create_reg_u32(Code.MOV_R32_IMM32, R.ECX, 0xFFFFFFFF), rdpru, ret),
         "out-of-range ECX=0xffffffff for rdpru"),
        ("monitorx_undef_rax", (xor_ecx, xor_edx, monitorx, mov_eax_0, ret),
         "non-flat or unowned RAX address for monitorx"),
        ("monitorx_const_rax", (Instruction.create_reg_u32(Code.MOV_R32_IMM32, R.EAX, 0x1000), xor_ecx, xor_edx, monitorx, ret),
         "non-flat or unowned RAX address for monitorx"),
        ("monitorx_arith_rax", (lea_rax_rcx, Instruction.create_reg_i32(Code.ADD_RM64_IMM8, R.RAX, 1), xor_ecx, xor_edx, monitorx, ret),
         "non-flat or unowned RAX address for monitorx"),
        ("monitorx_nonzero_ecx", (lea_rax_rcx, mov_ecx_1, xor_edx, monitorx, ret),
         "undefined or non-zero ECX for monitorx"),
        ("monitorx_unproven_ecx", (lea_rax_rcx, xor_edx, monitorx, ret),
         "undefined or non-zero ECX for monitorx"),
        ("monitorx_nonzero_edx", (lea_rax_rcx, xor_ecx, Instruction.create_reg_u32(Code.MOV_R32_IMM32, R.EDX, 1), monitorx, ret),
         "undefined or non-zero EDX for monitorx"),
        ("monitorx_unproven_edx", (lea_rax_rcx, xor_ecx, monitorx, ret),
         "undefined or non-zero EDX for monitorx"),
        ("mwaitx_undef_rax", (mov_ecx_0, mwaitx, mov_eax_0, ret),
         "undefined RAX hints for mwaitx"),
        ("mwaitx_unproven_ecx", (mov_eax_0, mwaitx, ret),
         "undefined or non-constant ECX extensions for mwaitx"),
        ("mwaitx_timer_ecx_2", (mov_eax_0, mov_ecx_2, mwaitx, ret),
         "unsupported MWAITX timer extension ECX[1]"),
        ("mwaitx_timer_ecx_3", (mov_eax_0, Instruction.create_reg_u32(Code.MOV_R32_IMM32, R.ECX, 3), mwaitx, ret),
         "unsupported MWAITX timer extension ECX[1]"),
        ("mwaitx_reserved_ecx_4", (mov_eax_0, Instruction.create_reg_u32(Code.MOV_R32_IMM32, R.ECX, 4), mwaitx, ret),
         "unsupported reserved MWAITX extension bits"),
        ("undef_gpr_read", (Instruction.create_reg_reg(Code.ADD_R64_RM64, R.R9, R.RCX), mov_eax_0, ret),
         "undefined register r9"),
        ("undef_ret_rax", (xor_ecx, ret),
         "undefined return register rax"),
        ("undef_flag_setb", (mov_eax_0, Instruction.create_reg(Code.SETB_RM8, R.AL), ret),
         "undefined flag cf read"),
        ("undef_flag_seto_after_shl2", (mov_eax_0, Instruction.create_reg_u32(Code.SHL_RM64_IMM8, R.RAX, 2),
                                        Instruction.create_reg(Code.SETO_RM8, R.AL), ret),
         "undefined flag of read"),
        ("store_outside_r8_rcx", (mov_eax_0, Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RCX), R.RAX), ret),
         "store outside output buffer r8"),
        ("store_outside_r8_rdx", (mov_eax_0, Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RDX), R.RAX), ret),
         "store outside output buffer r8"),
        ("store_oob_r8_disp64", (mov_eax_0, Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8, displ=64, displ_size=1), R.RAX), ret),
         "out-of-buffer store displacement"),
        ("store_indexed", (mov_eax_0, Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8, index=R.RCX), R.RAX), ret),
         "unsupported memory operand"),
        ("lea_displaced", (Instruction.create_reg_mem(Code.LEA_R64_M, R.RAX, MemoryOperand(R.RCX, displ=8, displ_size=1)), ret),
         "unsupported displaced lea operand"),
        ("lea_indexed", (Instruction.create_reg_mem(Code.LEA_R64_M, R.RAX, MemoryOperand(R.RCX, index=R.RDX)), ret),
         "unsupported lea memory operand"),
        ("prefix_fs", (mov_eax_0, fs_store, ret),
         "unsupported instruction prefix"),
        ("prefix_rep_ret", (mov_eax_0, rep_ret),
         "unsupported instruction prefix"),
        ("branch_jmp", (mov_eax_0, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret),
         "unsupported control-flow instruction"),
        ("missing_ret", (xor_ecx, rdpru),
         "missing terminal ret"),
        ("trailing_after_ret", (mov_eax_0, ret, mov_eax_0),
         "unreachable instruction after ret"),
    ]
    for label, seq, reason in rejections:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted rejected sequence")
    return len(positive), len(rejections)


# --- /O2 binary anti-CSE / anti-DCE inspection ---

def walk_export_codes(dll_path, symbol):
    """Collect reachable instruction Codes from a compiled export without relying on .pdata."""
    with pefile.PE(str(dll_path), fast_load=True) as pe:
        base = pe.OPTIONAL_HEADER.ImageBase
        image = pe.get_memory_mapped_image()
        pending = [base + export_rva(pe, symbol)]
        seen = set()
        found = {}
        while pending:
            ip = pending.pop()
            while ip not in seen:
                seen.add(ip)
                assert base <= ip < base + len(image), f"{symbol}: branch left image at {ip:#x}"
                ins = Decoder(64, image[ip - base:ip - base + 15], ip=ip).decode()
                found[ip] = ins.code
                flow = ins.flow_control
                if flow == FlowControl.RETURN:
                    break
                if flow in (FlowControl.CONDITIONAL_BRANCH, FlowControl.UNCONDITIONAL_BRANCH):
                    pending.append(ins.near_branch_target)
                    if flow == FlowControl.UNCONDITIONAL_BRANCH:
                        break
                ip = ins.next_ip
    return [found[ip] for ip in sorted(found)]


def check_o2_barriers(o2_path):
    """Verify /O2 did not CSE consecutive RDPRU reads, DCE overwritten RDPRU, or elide MWAITX."""
    expected_rdtsc = {
        "lifted_op_rdpru_mperf": 1,
        "lifted_op_rdpru_aperf": 1,
        "lifted_op_rdpru_twice": 2,
        "lifted_op_rdpru_unused": 1,
    }
    for sym, want in expected_rdtsc.items():
        codes = walk_export_codes(o2_path, sym)
        got = codes.count(Code.RDTSC)
        if got != want:
            raise AssertionError(f"/O2 {sym}: expected {want} RDTSC instruction(s), found {got}")
    for sym in ("lifted_op_monitorx_mwaitx_r8", "lifted_op_monitorx_mwaitx_intbrk"):
        codes = walk_export_codes(o2_path, sym)
        if codes.count(Code.PAUSE) < 1:
            raise AssertionError(f"/O2 {sym}: expected PAUSE barrier from MWAITX, found 0")


# --- Execution checks (run on Debug lifted, /O2 lifted, and optionally native AMD) ---

def check_runtime_suite(label, funcs, serial_tsc, *, check_tsc_bracket, check_null_fault):
    total = 0
    # 1. RDPRU MPERF (ECX=0) and APERF (ECX=1)
    for name, want_ecx, flag_fn in (
        ("op_rdpru_mperf", 0, add64_flags),
        ("op_rdpru_aperf", 1, cmp64_flags),
    ):
        for a, b in FLAG_PAIRS:
            before = serial_tsc() if check_tsc_bracket else 0
            ret, slots = call_export(funcs[name], a, b)
            after = serial_tsc() if check_tsc_bracket else MASK64
            lo, hi, ecx_out, flags_out = slots[0], slots[1], slots[2], slots[3]
            assert lo <= MASK32 and hi <= MASK32, f"{label} {name}: upper 32 bits not zeroed ({lo:#x}, {hi:#x})"
            assert ret == ((hi << 32) | lo), f"{label} {name}: return mismatch"
            if check_tsc_bracket:
                assert before <= ret <= after, f"{label} {name}: {ret:#x} outside TSC bracket {before:#x}..{after:#x}"
            assert ecx_out == want_ecx, f"{label} {name}: RCX clobbered ({ecx_out:#x} != {want_ecx:#x})"
            want_flags = flag_fn(a, b)
            assert flags_out == want_flags, f"{label} {name} ({a:#x},{b:#x}): flags {flags_out:#x} != {want_flags:#x}"
            total += 1

    # 2. Consecutive RDPRU monotonicity (op_rdpru_twice)
    advanced = False
    for _ in range(512):
        ret, slots = call_export(funcs["op_rdpru_twice"], 0, 0)
        lo1, hi1, lo2, hi2, ecx_out = slots[0], slots[1], slots[2], slots[3], slots[4]
        assert max(lo1, hi1, lo2, hi2) <= MASK32, f"{label} op_rdpru_twice: upper 32 bits not zeroed"
        t1, t2 = (hi1 << 32) | lo1, (hi2 << 32) | lo2
        assert t2 >= t1, f"{label} op_rdpru_twice went backwards: {t1:#x} -> {t2:#x}"
        assert ret == t2 and ecx_out == 0
        advanced |= (t2 > t1)
        total += 1
    assert advanced, f"{label} op_rdpru_twice never observed counter advancement in 512 rounds"

    # 3. Overwritten RDPRU (op_rdpru_unused)
    ret, _ = call_export(funcs["op_rdpru_unused"], 0x1111, 0x2222)
    assert ret == 7, f"{label} op_rdpru_unused returned {ret}, expected 7"
    total += 1

    # 4. MONITORX + triggering store + MWAITX (ECX=0 and ECX=1 interrupt-break)
    for a, b in FLAG_PAIRS:
        ret, slots = call_export(funcs["op_monitorx_mwaitx_r8"], a, b)
        want_flags = cmp64_flags(a, b)
        assert slots[0] == a and slots[1] == b and slots[2] == 0 and slots[3] == want_flags and slots[4] == 0
        assert ret == want_flags
        ret2, slots2 = call_export(funcs["op_monitorx_mwaitx_intbrk"], a, b)
        want_flags2 = add64_flags(a, b)
        assert slots2[0] == a and slots2[1] == 0x10 and slots2[2] == 1 and slots2[3] == 0 and slots2[4] == want_flags2
        assert ret2 == want_flags2
        total += 2

    # 5. MONITORX on flat [rcx] and [rdx] pointers + NULL fault ordering
    buf_a, a_pre, a_off, a_post, a_addr = alloc_guarded(b"\x11" * 16)
    buf_b, b_pre, b_off, b_post, b_addr = alloc_guarded(b"\x22" * 16)
    ret, slots = call_export(funcs["op_monitorx_rcx_rdx"], a_addr, b_addr)
    want_flags = cmp64_flags(a_addr, b_addr)
    assert slots[0] == b_addr and slots[1] == a_addr and slots[2] == b_addr and slots[3] == want_flags
    assert ret == want_flags
    assert buf_a.raw[a_pre:a_off] == GUARD_PRE and buf_a.raw[a_post:a_post + 16] == GUARD_POST
    assert buf_b.raw[b_pre:b_off] == GUARD_PRE and buf_b.raw[b_post:b_post + 16] == GUARD_POST
    total += 1

    if check_null_fault:
        out_raw, o_pre, o_off, o_post, o_addr = alloc_guarded(b"\x5a" * OUT_BYTES)
        try:
            funcs["op_monitorx_rcx_rdx"](0, b_addr, o_addr)
        except OSError as exc:
            assert "access violation" in str(exc).lower(), (label, exc)
        else:
            raise AssertionError(f"{label} op_monitorx_rcx_rdx did not fault on NULL rcx")
        assert out_raw.raw[o_pre:o_off] == GUARD_PRE and out_raw.raw[o_post:o_post + 16] == GUARD_POST
        fault_slots = struct.unpack("<8Q", out_raw.raw[o_off:o_off + OUT_BYTES])
        assert fault_slots[0] == b_addr, f"{label}: pre-MONITORX store was elided/sunk across fault"
        assert fault_slots[1] == SENTINEL64, f"{label}: post-MONITORX store hoisted before fault"
        total += 1

    # 6. 32-bit ADD and CMP flag paths across RDPRU and MONITORX/MWAITX
    for a, b in FLAG_PAIRS_32:
        before = serial_tsc() if check_tsc_bracket else 0
        ret, slots = call_export(funcs["op_add32_rdpru"], a, b)
        after = serial_tsc() if check_tsc_bracket else MASK64
        lo, hi, ecx_out, flags_out = slots[0], slots[1], slots[2], slots[3]
        assert lo <= MASK32 and hi <= MASK32
        assert ret == ((hi << 32) | lo)
        if check_tsc_bracket:
            assert before <= ret <= after
        assert ecx_out == 0 and flags_out == add32_flags(a, b)

        ret_c, slots_c = call_export(funcs["op_cmp32_mwaitx"], a, b)
        want_c = cmp32_flags(a, b)
        assert slots_c[0] == (a & MASK32) and slots_c[1] == 0 and slots_c[2] == 0 and slots_c[3] == want_c
        assert ret_c == want_c
        total += 2

    return total


MUTANT_SPECS = (
    ("mut_rdpru_no_zext", "op_rdpru_mperf", "rax = static_cast<std::uint32_t>(pru); rdx = pru >> 32;",
     "rax = pru | 0x100000000ull; rdx = (pru >> 32) | 0x100000000ull;"),
    ("mut_rdpru_swap_halves", "op_rdpru_mperf", "rax = static_cast<std::uint32_t>(pru); rdx = pru >> 32;",
     "rax = pru >> 32; rdx = static_cast<std::uint32_t>(pru);"),
    ("mut_rdpru_clobber_flags", "op_rdpru_mperf", "rdx = pru >> 32; }",
     "rdx = pru >> 32; cf = false; zf = false; sf = false; of = false; }"),
    ("mut_mwaitx_clobber_flags", "op_monitorx_mwaitx_r8", "op_mwaitx(static_cast<std::uint32_t>(rax), 0u);",
     "op_mwaitx(static_cast<std::uint32_t>(rax), 0u); cf = false; zf = false; sf = false; of = false;"),
    ("mut_mwaitx_drop_hint_ext", "op_monitorx_mwaitx_intbrk", "op_mwaitx(static_cast<std::uint32_t>(rax), 1u);",
     "op_mwaitx(static_cast<std::uint32_t>(rax), 1u); rax = 0ull; rcx = 0ull;"),
    ("mut_cmp32_flip_cf", "op_cmp32_mwaitx", "op_cmp32(r9, r10, cf, zf, sf, of);",
     "op_cmp32(r9, r10, cf, zf, sf, of); cf = static_cast<std::uint32_t>(r9) > static_cast<std::uint32_t>(r10);"),
)


def emit_mutants(original_path, out_cpp):
    """Emit compiled C++ mutation controls from translated functions."""
    unique_exports = []
    for _, src_name, _, _ in MUTANT_SPECS:
        if src_name not in unique_exports:
            unique_exports.append(src_name)
    base_src, _ = cpu_amd_monitor_rdpru_aot.translate(original_path, unique_exports)
    chunks = [base_src]
    with pefile.PE(str(original_path), fast_load=True) as pe:
        by_name = {}
        for name in unique_exports:
            entry = export_rva(pe, name)
            fn_src, _ = lift_code(name, entry, bounded_code(pe, entry))
            by_name[name] = fn_src
    for mut_name, src_name, needle, repl in MUTANT_SPECS:
        src = by_name[src_name]
        if needle not in src:
            raise AssertionError(f"mutant {mut_name}: needle {needle!r} not found in {src_name}")
        mutated = src.replace(f"lifted_{src_name}", mut_name, 1).replace(needle, repl, 1)
        chunks.append(mutated)
    out_cpp.parent.mkdir(parents=True, exist_ok=True)
    out_cpp.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def check_altered_and_mutants(lifted, altered_lifted, mutants_path=None, serial_tsc=None):
    # Altered-lifted build must diverge on every altered export.
    _, s_orig_m = call_export(lifted["op_rdpru_mperf"], 5, 9)
    _, s_alt_m = call_export(altered_lifted["op_rdpru_mperf"], 5, 9)
    assert s_orig_m[2] == 0 and s_alt_m[2] == 1

    _, s_orig_a = call_export(lifted["op_rdpru_aperf"], 5, 9)
    _, s_alt_a = call_export(altered_lifted["op_rdpru_aperf"], 5, 9)
    assert s_orig_a[2] == 1 and s_alt_a[2] == 0

    _, s_orig_t = call_export(lifted["op_rdpru_twice"], 0, 0)
    _, s_alt_t = call_export(altered_lifted["op_rdpru_twice"], 0, 0)
    assert s_orig_t[4] == 0 and s_alt_t[4] == 1

    r_orig_u, _ = call_export(lifted["op_rdpru_unused"], 0, 0)
    r_alt_u, _ = call_export(altered_lifted["op_rdpru_unused"], 0, 0)
    assert r_orig_u == 7 and r_alt_u == 42

    _, s_orig_mw = call_export(lifted["op_monitorx_mwaitx_r8"], 0x1234, 0x5678)
    _, s_alt_mw = call_export(altered_lifted["op_monitorx_mwaitx_r8"], 0x1234, 0x5678)
    assert s_orig_mw[1] == 0x5678 and s_alt_mw[1] == 0x1234

    _, s_orig_ib = call_export(lifted["op_monitorx_mwaitx_intbrk"], 1, 2)
    _, s_alt_ib = call_export(altered_lifted["op_monitorx_mwaitx_intbrk"], 1, 2)
    assert s_orig_ib[1] == 0x10 and s_alt_ib[1] == 0x20

    mutants = 0
    if mutants_path is not None:
        mut_lib = ctypes.CDLL(str(Path(mutants_path).resolve()))
        mut = {}
        for mut_name, _, _, _ in MUTANT_SPECS:
            fn = getattr(mut_lib, mut_name)
            fn.argtypes = [ctypes.c_uint64] * 3
            fn.restype = ctypes.c_uint64
            mut[mut_name] = fn

        # 1. Missing 32->64 zero extension on RAX/RDX after RDPRU
        _, sm1 = call_export(mut["mut_rdpru_no_zext"], 1, 2)
        assert (sm1[0] >> 32) != 0 and (sm1[1] >> 32) != 0
        mutants += 1
        # 2. Swapped RAX/RDX halves in returned 64-bit counter
        before = serial_tsc()
        rm2, sm2 = call_export(mut["mut_rdpru_swap_halves"], 1, 2)
        after = serial_tsc()
        unswapped = (sm2[0] << 32) | sm2[1]
        assert sm2[0] != sm2[1] and before <= unswapped <= after and not (before <= rm2 <= after)
        mutants += 1
        # 3. Clobbering RFLAGS on RDPRU
        _, sm3 = call_export(mut["mut_rdpru_clobber_flags"], 0x7FFFFFFFFFFFFFFF, 1)
        assert sm3[3] != add64_flags(0x7FFFFFFFFFFFFFFF, 1)
        mutants += 1
        # 4. Clobbering RFLAGS on MWAITX
        _, sm4 = call_export(mut["mut_mwaitx_clobber_flags"], 5, 9)
        assert sm4[3] != cmp64_flags(5, 9)
        mutants += 1
        # 5. Dropping MWAITX EAX hint and ECX=1 extension
        _, sm5 = call_export(mut["mut_mwaitx_drop_hint_ext"], 1, 2)
        assert sm5[1] != 0x10 or sm5[2] != 1
        mutants += 1
        # 6. Flipping CF in op_cmp32 (a < b -> a > b)
        rc6, _ = call_export(mut["mut_cmp32_flip_cf"], 5, 9)
        assert rc6 != cmp32_flags(5, 9)
        mutants += 1
    return len(ALTERED_EXPORTS), mutants


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
    if len(sys.argv) >= 4 and sys.argv[1] == "emit-mutants":
        emit_mutants(Path(sys.argv[2]), Path(sys.argv[3]))
        return

    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)

    original_path, lifted_path, lifted_o2_path, altered_path, altered_lifted_path, mutants_path = (
        Path(p) for p in sys.argv[1:7]
    )

    accepted, rejected = check_translator()
    print(f"{accepted} AMD MONITORX/MWAITX/RDPRU sequences accepted; {rejected} rejection controls passed")

    check_o2_barriers(lifted_o2_path)
    print("/O2 disassembly verified: RDPRU anti-CSE (2 reads in twice), anti-DCE (1 read in unused), MWAITX PAUSE barriers kept")

    pin_current_thread()
    lifted_lib, lifted = bind(lifted_path, "lifted_")
    _, lifted_o2 = bind(lifted_o2_path, "lifted_")
    _, altered_lifted = bind(altered_lifted_path, "lifted_")

    serial_tsc = lifted_lib.host_serial_tsc
    serial_tsc.argtypes, serial_tsc.restype = [], ctypes.c_uint64
    has_mwaitx_fn = lifted_lib.host_has_mwaitx
    has_mwaitx_fn.argtypes, has_mwaitx_fn.restype = [], ctypes.c_uint32
    has_rdpru_fn = lifted_lib.host_has_rdpru
    has_rdpru_fn.argtypes, has_rdpru_fn.restype = [], ctypes.c_uint32

    dbg_total = check_runtime_suite("lifted", lifted, serial_tsc, check_tsc_bracket=True, check_null_fault=True)
    o2_total = check_runtime_suite("lifted_o2", lifted_o2, serial_tsc, check_tsc_bracket=True, check_null_fault=True)
    alt_count, mut_count = check_altered_and_mutants(lifted, altered_lifted, mutants_path, serial_tsc)
    print(
        f"lifted Debug ({dbg_total} checks) + /O2 ({o2_total} checks) + NULL fault-ordering passed; "
        f"{alt_count} altered-build + {mut_count} compiled C++ mutant controls caught"
    )

    has_mwaitx = bool(has_mwaitx_fn())
    has_rdpru = bool(has_rdpru_fn())
    if not (has_mwaitx and has_rdpru):
        print(
            f"Host lacks AMD MWAITX (CPUID Fn8000_0001:ECX[29]={int(has_mwaitx)}) and/or "
            f"RDPRU (CPUID Fn8000_0008:EBX[4]={int(has_rdpru)}); skipping native differential check"
        )
        sys.exit(77)

    _, native = bind(original_path, "")
    _, altered = bind(altered_path, "")
    nat_total = check_runtime_suite("native", native, serial_tsc, check_tsc_bracket=False, check_null_fault=True)
    alt_nat_count, _ = check_altered_and_mutants(native, altered)
    print(f"native AMD hardware differential passed ({nat_total} checks, {alt_nat_count} altered-native controls caught)")

    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
