"""Differential, /O2, mutant, and rejection checks for legacy MMX/SSE2 packed unsigned integer ops."""

import ctypes
from pathlib import Path
import random
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Decoder, Encoder, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva  # noqa: E402
from cpu_sse_mmx_aot import lift_code, translate  # noqa: E402
import sse_mmx_check as base  # noqa: E402


EXPORTS = (
    "op_pavgb_mm_reg",
    "op_pavgb_mm_mem",
    "op_pavgb_mm_mem_plus1",
    "op_pavgb_mm_self",
    "op_pavgw_mm_reg",
    "op_pavgw_mm_mem",
    "op_pavgw_mm_mem_plus1",
    "op_pavgw_mm_self",
    "op_pmulhuw_mm_reg",
    "op_pmulhuw_mm_mem",
    "op_pmulhuw_mm_mem_plus1",
    "op_pmulhuw_mm_self",
    "op_psadbw_mm_reg",
    "op_psadbw_mm_mem",
    "op_psadbw_mm_mem_plus1",
    "op_psadbw_mm_self",
    "op_pavgb_xmm_reg",
    "op_pavgb_xmm_mem",
    "op_pavgb_xmm_mem_disp",
    "op_pavgb_xmm_self",
    "op_pavgw_xmm_reg",
    "op_pavgw_xmm_mem",
    "op_pavgw_xmm_mem_disp",
    "op_pavgw_xmm_self",
    "op_pmulhuw_xmm_reg",
    "op_pmulhuw_xmm_mem",
    "op_pmulhuw_xmm_mem_disp",
    "op_pmulhuw_xmm_self",
    "op_psadbw_xmm_reg",
    "op_psadbw_xmm_mem",
    "op_psadbw_xmm_mem_disp",
    "op_psadbw_xmm_self",
    "op_mmx_int_pipeline",
    "op_mmx_int_discarded_read",
)

ALTERED_EXPORTS = (
    "op_pavgb_mm_reg",
    "op_pavgw_mm_reg",
    "op_pmulhuw_mm_reg",
    "op_psadbw_mm_reg",
    "op_pavgb_xmm_reg",
    "op_pavgw_xmm_reg",
    "op_pmulhuw_xmm_reg",
    "op_psadbw_xmm_reg",
    "op_psadbw_xmm_mem_disp",
    "op_mmx_int_pipeline",
)

FILL = base.FILL
GUARD_PRE = base.GUARD_PRE
GUARD_POST = base.GUARD_POST


def ref_pavgb(a, b, drop_plus1=False):
    bias = 0 if drop_plus1 else 1
    return bytes(((x + y + bias) >> 1) & 0xFF for x, y in zip(a, b))


def ref_pavgw(a, b, drop_plus1=False):
    bias = 0 if drop_plus1 else 1
    aw = struct.unpack("<" + "H" * (len(a) // 2), a)
    bw = struct.unpack("<" + "H" * (len(b) // 2), b)
    return struct.pack("<" + "H" * len(aw), *(((x + y + bias) >> 1) & 0xFFFF for x, y in zip(aw, bw)))


def ref_pmulhuw(a, b, signed=False):
    fmt = "<" + ("h" if signed else "H") * (len(a) // 2)
    aw = struct.unpack(fmt, a)
    bw = struct.unpack(fmt, b)
    return struct.pack("<" + "H" * len(aw), *(((x * y) >> 16) & 0xFFFF for x, y in zip(aw, bw)))


def ref_psadbw(a, b, merge_halves=False):
    if len(a) == 8:
        return struct.pack("<Q", sum(abs(x - y) for x, y in zip(a, b)))
    lo = sum(abs(x - y) for x, y in zip(a[:8], b[:8]))
    hi = sum(abs(x - y) for x, y in zip(a[8:16], b[8:16]))
    if merge_halves:
        return struct.pack("<QQ", lo + hi, 0)
    return struct.pack("<QQ", lo, hi)


def packed_pairs():
    """Edge byte/word combinations (carry, rounding, signed-vs-unsigned MSB, max SAD) + random."""
    byte_edges = (0x00, 0x01, 0x02, 0x7F, 0x80, 0x81, 0xFE, 0xFF)
    word_edges = (0x0000, 0x0001, 0x0002, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF)
    pairs = [
        (bytes([0xFF] * 16 + [0x00] * 16), bytes([0xFF] * 16 + [0x01] * 16)),
        (bytes([0x00] * 16 + [0xFF] * 16), bytes([0x01] * 16 + [0x00] * 16)),
        (struct.pack("<16H", *([0xFFFF] * 8 + [0x8000] * 8)),
         struct.pack("<16H", *([0xFFFF] * 8 + [0x8001] * 8))),
        (bytes(byte_edges[(i * 3) % len(byte_edges)] for i in range(32)),
         b"\xA5" + bytes(byte_edges[(i * 5 + 1) % len(byte_edges)] for i in range(31))),
        (struct.pack("<16H", *(word_edges[(i * 3) % len(word_edges)] for i in range(16))),
         b"\x5A" + struct.pack("<15H", *(word_edges[(i * 5 + 1) % len(word_edges)] for i in range(15))) + b"\x99"),
    ]
    for sh in range(8):
        a = bytes(byte_edges[(i + sh) % len(byte_edges)] for i in range(32))
        b = bytes(byte_edges[(3 * i + sh + 1) % len(byte_edges)] for i in range(32))
        pairs.append((a, b))
    rng = random.Random(0x55E177)
    for _ in range(24):
        pairs.append((rng.randbytes(32), rng.randbytes(32)))
    return pairs


def reference(name, a, b, altered=False):
    out = bytearray(FILL)
    a8, a16, a8_hi = a[0:8], a[0:16], a[8:16]
    b8, b8_p1, b16, b16_hi = b[0:8], b[1:9], b[0:16], b[16:32]

    if name == "op_pavgb_mm_reg":
        out[0:8] = ref_pavgw(a8, b8) if altered else ref_pavgb(a8, b8)
    elif name == "op_pavgb_mm_mem":
        out[0:8] = ref_pavgb(a8, b8)
    elif name == "op_pavgb_mm_mem_plus1":
        out[0:8] = ref_pavgb(a8, b8_p1)
    elif name == "op_pavgb_mm_self":
        out[0:8] = ref_pavgb(a8, a8)
    elif name == "op_pavgw_mm_reg":
        out[0:8] = ref_pavgb(a8, b8) if altered else ref_pavgw(a8, b8)
    elif name == "op_pavgw_mm_mem":
        out[0:8] = ref_pavgw(a8, b8)
    elif name == "op_pavgw_mm_mem_plus1":
        out[0:8] = ref_pavgw(a8, b8_p1)
    elif name == "op_pavgw_mm_self":
        out[0:8] = ref_pavgw(a8, a8)
    elif name == "op_pmulhuw_mm_reg":
        out[0:8] = ref_pavgw(a8, b8) if altered else ref_pmulhuw(a8, b8)
    elif name == "op_pmulhuw_mm_mem":
        out[0:8] = ref_pmulhuw(a8, b8)
    elif name == "op_pmulhuw_mm_mem_plus1":
        out[0:8] = ref_pmulhuw(a8, b8_p1)
    elif name == "op_pmulhuw_mm_self":
        out[0:8] = ref_pmulhuw(a8, a8)
    elif name == "op_psadbw_mm_reg":
        out[0:8] = ref_pavgb(a8, b8) if altered else ref_psadbw(a8, b8)
    elif name == "op_psadbw_mm_mem":
        out[0:8] = ref_psadbw(a8, b8)
    elif name == "op_psadbw_mm_mem_plus1":
        out[0:8] = ref_psadbw(a8, b8_p1)
    elif name == "op_psadbw_mm_self":
        out[0:8] = ref_psadbw(a8, a8)
    elif name == "op_pavgb_xmm_reg":
        out[0:16] = ref_pavgw(a16, b16) if altered else ref_pavgb(a16, b16)
    elif name == "op_pavgb_xmm_mem":
        out[0:16] = ref_pavgb(a16, b16)
    elif name == "op_pavgb_xmm_mem_disp":
        out[0:16] = ref_pavgb(a16, b16_hi)
    elif name == "op_pavgb_xmm_self":
        out[0:16] = ref_pavgb(a16, a16)
    elif name == "op_pavgw_xmm_reg":
        out[0:16] = ref_pavgb(a16, b16) if altered else ref_pavgw(a16, b16)
    elif name == "op_pavgw_xmm_mem":
        out[0:16] = ref_pavgw(a16, b16)
    elif name == "op_pavgw_xmm_mem_disp":
        out[0:16] = ref_pavgw(a16, b16_hi)
    elif name == "op_pavgw_xmm_self":
        out[0:16] = ref_pavgw(a16, a16)
    elif name == "op_pmulhuw_xmm_reg":
        out[0:16] = ref_pavgw(a16, b16) if altered else ref_pmulhuw(a16, b16)
    elif name == "op_pmulhuw_xmm_mem":
        out[0:16] = ref_pmulhuw(a16, b16)
    elif name == "op_pmulhuw_xmm_mem_disp":
        out[0:16] = ref_pmulhuw(a16, b16_hi)
    elif name == "op_pmulhuw_xmm_self":
        out[0:16] = ref_pmulhuw(a16, a16)
    elif name == "op_psadbw_xmm_reg":
        out[0:16] = ref_pavgb(a16, b16) if altered else ref_psadbw(a16, b16)
    elif name == "op_psadbw_xmm_mem":
        out[0:16] = ref_psadbw(a16, b16)
    elif name == "op_psadbw_xmm_mem_disp":
        out[0:16] = ref_psadbw(a16, b16 if altered else b16_hi)
    elif name == "op_psadbw_xmm_self":
        out[0:16] = ref_psadbw(a16, a16)
    elif name == "op_mmx_int_pipeline":
        mm0 = ref_pmulhuw(ref_pavgb(a8, b8_p1), a8_hi)
        mm1 = ref_psadbw(a8_hi, b8_p1)
        x0 = ref_pavgw(a16, b16_hi)
        x0 = ref_pavgb(x0, b16) if altered else ref_psadbw(x0, b16)
        out[0:8] = mm0
        out[8:16] = mm1
        out[16:32] = x0
    elif name == "op_mmx_int_discarded_read":
        out[0:16] = a16
    else:
        raise ValueError(name)
    return bytes(out)


def bind_exports(dll_path, prefix=""):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def check_discarded_read(fns, label):
    payload = packed_pairs()[0][0]
    buf_a, ptr_a, _, _, _ = base.alloc_aligned_buffer(payload)
    buf_out, ptr_out, o_pre, o_off, o_post = base.alloc_aligned_buffer(FILL)
    fns["op_mmx_int_discarded_read"](ptr_a, ptr_a, ptr_out)
    got_ok = base.check_guards_and_read(buf_out, o_pre, o_off, o_post, "output")
    if got_ok != payload[:16] + FILL[16:]:
        raise AssertionError(f"{label} op_mmx_int_discarded_read(valid): {got_ok.hex()}")

    buf_out2, ptr_out2, o_pre2, o_off2, o_post2 = base.alloc_aligned_buffer(FILL)
    try:
        fns["op_mmx_int_discarded_read"](ptr_a, 0, ptr_out2)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} op_mmx_int_discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_mmx_int_discarded_read(NULL) dropped faulting m128 read")
    if base.check_guards_and_read(buf_out2, o_pre2, o_off2, o_post2, "output") != FILL:
        raise AssertionError(f"{label} op_mmx_int_discarded_read(NULL) wrote output after fault")


def verify_positive_suite(native_fns, target_fns, label, pairs):
    cases = 0
    for name in EXPORTS:
        fn_n, fn_t = native_fns[name], target_fns[name]
        for idx, (a, b) in enumerate(pairs):
            want = reference(name, a, b, altered=False)
            got_n = base.invoke(fn_n, a, b)
            got_t = base.invoke(fn_t, a, b)
            if got_n != want or got_t != want:
                raise AssertionError(
                    f"{label} {name} case {idx}: native={got_n.hex()} target={got_t.hex()} exp={want.hex()}"
                )
            cases += 1
    check_discarded_read(native_fns, "native")
    check_discarded_read(target_fns, label)
    return cases


def check_altered_and_mutants(original_lifted_fns, altered_native_fns, altered_lifted_fns, pairs):
    diverged = {}
    for name in ALTERED_EXPORTS:
        diffs = 0
        for idx, (a, b) in enumerate(pairs):
            want_alt = reference(name, a, b, altered=True)
            got_an = base.invoke(altered_native_fns[name], a, b)
            got_al = base.invoke(altered_lifted_fns[name], a, b)
            if got_an != want_alt or got_al != want_alt:
                raise AssertionError(
                    f"altered {name} case {idx}: native={got_an.hex()} lifted={got_al.hex()} exp={want_alt.hex()}"
                )
            got_ol = base.invoke(original_lifted_fns[name], a, b)
            if got_ol != got_al:
                diffs += 1
        if diffs == 0:
            raise AssertionError(f"altered control {name} failed to diverge from original")
        diverged[name] = diffs

    # Negative software mutants: dropping +1, signed pmulhw, and merging XMM psadbw halves must diverge.
    a1, b1 = pairs[1]
    a2, b2 = pairs[2]
    a3, b3 = pairs[3]
    if ref_pavgb(a1[:8], b1[:8]) == ref_pavgb(a1[:8], b1[:8], drop_plus1=True):
        raise AssertionError("drop +1 pavgb mutant failed to diverge")
    if ref_pavgw(a1[:16], b1[:16]) == ref_pavgw(a1[:16], b1[:16], drop_plus1=True):
        raise AssertionError("drop +1 pavgw mutant failed to diverge")
    if ref_pmulhuw(a2[:16], b2[:16]) == ref_pmulhuw(a2[:16], b2[:16], signed=True):
        raise AssertionError("signed pmulhw mutant failed to diverge")
    if ref_psadbw(a3[:16], b3[:16]) == ref_psadbw(a3[:16], b3[:16], merge_halves=True):
        raise AssertionError("merged psadbw halves mutant failed to diverge")
    return diverged


def compile_o2_and_verify(original_path, native_fns, cxx_path, pairs):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "sse_mmx_int_o2.cpp"
        dll_path = tmp / "sse_mmx_int_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'sse_mmx_int_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_exports(dll_path, "lifted_")
        try:
            return verify_positive_suite(native_fns, o2_fns, "lifted_o2", pairs)
        finally:
            o2_fns.clear()
            unload_lib(o2_lib)


def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path):
    ld_xmm0 = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_xmm1 = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld_mm0 = Instruction.create_reg_mem(Code.MOVQ_MM_MMM64, R.MM0, MemoryOperand(R.RCX))
    ld_mm1 = Instruction.create_reg_mem(Code.MOVQ_MM_MMM64, R.MM1, MemoryOperand(R.RDX))
    st_mm0 = Instruction.create_mem_reg(Code.MOVQ_MMM64_MM, MemoryOperand(R.R8), R.MM0)
    emms_ins = Instruction.create(Code.EMMS)
    ret_ins = Instruction.create(Code.RETNQ)
    ret_bytes = encode_seq(ret_ins)

    # 1. VEX / EVEX rejections.
    nonlegacy_cases = [
        ("vpavgb_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VPAVGB_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vpavgw_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VPAVGW_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vpmulhuw_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VPMULHUW_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vpsadbw_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VPSADBW_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vpavgb_evex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.EVEX_VPAVGB_XMM_K1Z_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
    ]

    # 2. Missing emms, undefined mm/xmm, unaligned XMM, nonvolatile XMM, bad addressing, prefixes.
    seg_pavgb = Instruction.create_reg_mem(Code.PAVGB_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX))
    seg_pavgb.segment_prefix = R.GS
    other_cases = [
        ("missing_emms_pavgb", (ld_mm0, ld_mm1, Instruction.create_reg_reg(
            Code.PAVGB_MM_MMM64, R.MM0, R.MM1), st_mm0, ret_ins), "missing emms"),
        ("missing_emms_psadbw", (ld_mm0, Instruction.create_reg_reg(
            Code.PSADBW_MM_MMM64, R.MM0, R.MM0), ret_ins), "missing emms"),
        ("undef_mm0_pavgb_dst", (ld_mm1, Instruction.create_reg_reg(
            Code.PAVGB_MM_MMM64, R.MM0, R.MM1), emms_ins, ret_ins), "undefined MMX register mm0"),
        ("undef_mm2_pmulhuw_src", (ld_mm0, Instruction.create_reg_reg(
            Code.PMULHUW_MM_MMM64, R.MM0, R.MM2), emms_ins, ret_ins), "undefined MMX register mm2"),
        ("undef_mm_after_emms", (ld_mm0, emms_ins, Instruction.create_reg_reg(
            Code.PAVGW_MM_MMM64, R.MM0, R.MM0), emms_ins, ret_ins), "undefined MMX register mm0"),
        ("undef_xmm0_pavgw_dst", (ld_xmm1, Instruction.create_reg_reg(
            Code.PAVGW_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins), "undefined vector register xmm0"),
        ("undef_xmm3_psadbw_src", (ld_xmm0, Instruction.create_reg_reg(
            Code.PSADBW_XMM_XMMM128, R.XMM0, R.XMM3), ret_ins), "undefined vector register xmm3"),
        ("nonvol_xmm6_pavgb", (ld_xmm0, Instruction.create_reg_reg(
            Code.PAVGB_XMM_XMMM128, R.XMM6, R.XMM0), ret_ins), "nonvolatile vector register"),
        ("nonvol_xmm8_pmulhuw_src", (ld_xmm0, Instruction.create_reg_reg(
            Code.PMULHUW_XMM_XMMM128, R.XMM0, R.XMM8), ret_ins), "nonvolatile vector register"),
        ("unaligned_xmm_pavgb_1", (ld_xmm0, Instruction.create_reg_mem(
            Code.PAVGB_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=1, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("unaligned_xmm_pavgw_8", (ld_xmm0, Instruction.create_reg_mem(
            Code.PAVGW_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=8, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("unaligned_xmm_pmulhuw_neg4", (ld_xmm0, Instruction.create_reg_mem(
            Code.PMULHUW_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=-4, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("unaligned_xmm_psadbw_4", (ld_xmm0, Instruction.create_reg_mem(
            Code.PSADBW_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=4, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("rip_rel_pmulhuw", (ld_xmm0, Instruction.create_reg_mem(
            Code.PMULHUW_XMM_XMMM128, R.XMM0, MemoryOperand(R.RIP, displ=0x1020, displ_size=4)), ret_ins),
            "unsupported memory operand"),
        ("sib_index_psadbw", (ld_mm0, Instruction.create_reg_mem(
            Code.PSADBW_MM_MMM64, R.MM0, MemoryOperand(R.RCX, R.RDX, 2)), emms_ins, ret_ins),
            "unsupported memory operand"),
        ("bad_base_rax_pavgb", (ld_mm0, Instruction.create_reg_mem(
            Code.PAVGB_MM_MMM64, R.MM0, MemoryOperand(R.RAX)), emms_ins, ret_ins),
            "unsupported memory base"),
        ("nonvol_base_rbx", (ld_xmm0, Instruction.create_reg_mem(
            Code.PAVGW_XMM_XMMM128, R.XMM0, MemoryOperand(R.RBX)), ret_ins),
            "nonvolatile GPR base"),
        ("non_r8_store_movq", (ld_mm0, Instruction.create_mem_reg(
            Code.MOVQ_MMM64_MM, MemoryOperand(R.RCX), R.MM0), emms_ins, ret_ins),
            "non-r8 store memory operand"),
        ("segment_prefix_pavgb", (ld_xmm0, seg_pavgb, ret_ins), "unsupported instruction prefix"),
        ("nonleaf_jmp", (ld_xmm0, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret_ins),
            "unsupported instruction"),
        ("missing_ret", (ld_xmm0,), "missing terminal ret"),
        ("after_ret", (ret_ins, ld_xmm0), "unreachable instruction after ret"),
    ]

    raw_cases = [
        ("xop_vpcmov", bytes.fromhex("8fe878a2c230") + ret_bytes, "non-legacy encoding"),
        ("lock_cmpxchg", bytes.fromhex("f00fb109") + ret_bytes, "unsupported instruction prefix"),
        ("rep_movsb", bytes.fromhex("f3a4") + ret_bytes, "unsupported instruction prefix"),
    ]

    for label, seq, expected_reason in nonlegacy_cases + other_cases:
        raw = encode_seq(*seq)
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: unexpectedly accepted rejected sequence")

    for label, raw, expected_reason in raw_cases:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: unexpectedly accepted raw rejected sequence")

    # End-to-end PE-patched rejections & CLI non-traceback rejection.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_mmx_int_pipeline")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        orig_end = struct.unpack_from("<I", records, rec_idx + 4)[0]
        max_len = orig_end - entry

    other_map = {lbl: seq for lbl, seq, _ in other_cases}
    pe_patches = [
        ("pe_vpavgb_vex", encode_seq(*nonlegacy_cases[0][1]), "non-legacy encoding"),
        ("pe_vpavgb_evex", encode_seq(*nonlegacy_cases[4][1]), "non-legacy encoding"),
        ("pe_missing_emms", encode_seq(*other_map["missing_emms_pavgb"]), "missing emms"),
        ("pe_unaligned_xmm", encode_seq(*other_map["unaligned_xmm_pavgb_1"]), "unaligned 16-byte memory operand"),
        ("pe_undef_mm", encode_seq(*other_map["undef_mm0_pavgb_dst"]), "undefined MMX register mm0"),
        ("pe_seg_prefix", encode_seq(*other_map["segment_prefix_pavgb"]), "unsupported instruction prefix"),
    ]
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_sse_mmx_aot.py"
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
                translate(patched_path, ["op_mmx_int_pipeline"])
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched_path), "op_mmx_int_pipeline", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 1 or "Traceback" in proc.stderr or "unsupported instruction prefix" not in proc.stderr:
            raise AssertionError(f"CLI rejection unexpected: rc={proc.returncode}, stderr={proc.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure left behind output file")

    print(
        f"rejections passed: {len(nonlegacy_cases)} VEX/EVEX + {len(other_cases)} IR + "
        f"{len(raw_cases)} raw + {len(pe_patches)} PE-patched + CLI"
    )


def main():
    original_path, lifted_path, altered_path, altered_lifted_path, cxx_path = (
        Path(p) for p in sys.argv[1:6]
    )
    check_rejections(original_path)

    pairs = packed_pairs()
    orig_lib, orig_fns = bind_exports(original_path)
    lifted_lib, lifted_fns = bind_exports(lifted_path, "lifted_")
    alt_lib, alt_fns = bind_exports(altered_path)
    alt_lifted_lib, alt_lifted_fns = bind_exports(altered_lifted_path, "lifted_")
    try:
        lifted_cases = verify_positive_suite(orig_fns, lifted_fns, "lifted", pairs)
        diverged = check_altered_and_mutants(lifted_fns, alt_fns, alt_lifted_fns, pairs)
        o2_cases = compile_o2_and_verify(original_path, orig_fns, cxx_path, pairs)
    finally:
        for fn_map, lib in (
            (orig_fns, orig_lib),
            (lifted_fns, lifted_lib),
            (alt_fns, alt_lib),
            (alt_lifted_fns, alt_lifted_lib),
        ):
            fn_map.clear()
            unload_lib(lib)

    print(
        f"sse_mmx_int passed: {lifted_cases} lifted + {o2_cases} /O2 cases across {len(EXPORTS)} exports; "
        f"ALTER_IMM diverged on {diverged}"
    )


if __name__ == "__main__":
    main()
