"""Differential, /O2, mutant, and rejection checks for legacy SSE packed bitwise FP logic."""

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
from cpu_sse_fp_aot import lift_code, translate  # noqa: E402
import sse_fp_check as base  # noqa: E402
import sse_fp_ref as ref  # noqa: E402


EXPORTS = (
    "op_andps",
    "op_andps_mem",
    "op_andps_mem_disp",
    "op_andps_self",
    "op_andnps",
    "op_andnps_mem",
    "op_andnps_mem_disp",
    "op_andnps_self",
    "op_orps",
    "op_orps_mem",
    "op_orps_mem_disp",
    "op_orps_self",
    "op_xorps",
    "op_xorps_mem",
    "op_xorps_mem_disp",
    "op_xorps_self",
    "op_xorps_zero_logic",
    "op_xorps_zero_arith",
    "op_logic_pipeline",
    "op_logic_discarded_read",
)

ALTERED_EXPORTS = (
    "op_andps",
    "op_andnps",
    "op_andnps_mem",
    "op_orps",
    "op_xorps",
    "op_logic_pipeline",
)

FILL = base.FILL
MASK32 = 0xFFFFFFFF

LOGIC_SPECIAL_WORDS = (
    0x00000000,  # +0.0
    0x80000000,  # -0.0
    0x3F800000,  # 1.0
    0xBF800000,  # -1.0
    0x7F800000,  # +inf
    0xFF800000,  # -inf
    0x7FC00000,  # canonical qNaN
    0x7FC00001,  # payload qNaN
    0xFFC00000,  # negative qNaN
    0x7F800001,  # positive sNaN
    0xFF800001,  # negative sNaN
    0x7FBFFFFF,  # max sNaN payload
    0x00000001,  # min positive denormal
    0x007FFFFF,  # max positive denormal
    0x80000001,  # min negative denormal
    0x7F7FFFFF,  # max finite
    0xFFFFFFFF,  # all ones / negative qNaN
    0x55555555,  # alternating bits
    0xAAAAAAAA,  # alternating bits
)


def pk_d(*vals):
    return struct.pack("<" + "I" * len(vals), *(v & MASK32 for v in vals))


def u32s(buf):
    return struct.unpack("<" + "I" * (len(buf) // 4), buf)


def logic_pairs():
    """Bit-pattern edge pairs covering NaNs, +/-0.0, denormals, Inf, bitmasks, and random vectors."""
    pairs = []
    e = LOGIC_SPECIAL_WORDS
    for i in range(len(e)):
        a_lo = pk_d(e[i], e[(i + 3) % len(e)], e[(i + 7) % len(e)], e[(i + 11) % len(e)])
        a_hi = pk_d(e[(i + 5) % len(e)], e[(i + 9) % len(e)], e[(i + 13) % len(e)], e[(i + 17) % len(e)])
        b_lo = pk_d(e[(2 * i + 1) % len(e)], e[(2 * i + 5) % len(e)], e[(2 * i + 9) % len(e)], e[(2 * i + 13) % len(e)])
        b_hi = pk_d(e[(3 * i + 2) % len(e)], e[(3 * i + 6) % len(e)], e[(3 * i + 10) % len(e)], e[(3 * i + 14) % len(e)])
        pairs.append((a_lo + a_hi, b_lo + b_hi))
    # Explicit asymmetric masks to distinguish (~dst) & src from dst & (~src).
    pairs.extend([
        (pk_d(0xF0F0F0F0, 0x7FC00001, 0x80000000, 0x00000001) + pk_d(0x0F0F0F0F, 0xFF800001, 0x00000000, 0x7F7FFFFF),
         pk_d(0x0FF00FF0, 0x7F800001, 0x00000000, 0xFFFFFFFF) + pk_d(0xFF00FF00, 0xFFC00000, 0x80000000, 0x007FFFFF)),
        (pk_d(0x00000000, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA) * 2,
         pk_d(0xFFFFFFFF, 0x00000000, 0x33333333, 0xCCCCCCCC) * 2),
    ])
    rng = random.Random(0x55EF9106)
    for _ in range(32):
        pairs.append((rng.randbytes(32), rng.randbytes(32)))
    return pairs


def _bit_op(kind, dst16, src16):
    out = []
    for d, s in zip(u32s(dst16), u32s(src16)):
        if kind == "and":
            out.append(d & s)
        elif kind == "andn":
            out.append((~d) & s)
        elif kind == "or":
            out.append(d | s)
        elif kind == "xor":
            out.append(d ^ s)
        else:
            raise ValueError(kind)
    return pk_d(*out)


def reference(name, a, b, altered=False):
    out = bytearray(FILL)
    a0, a1 = a[0:16], a[16:32]
    b0, b1 = b[0:16], b[16:32]

    if name == "op_andps":
        out[0:16] = _bit_op("or" if altered else "and", a0, b0)
    elif name == "op_andps_mem":
        out[0:16] = _bit_op("and", a0, b0)
    elif name == "op_andps_mem_disp":
        out[0:16] = _bit_op("and", a0, b1)
    elif name == "op_andps_self":
        out[0:16] = _bit_op("and", a0, a0)
    elif name == "op_andnps":
        out[0:16] = _bit_op("andn", b0, a0) if altered else _bit_op("andn", a0, b0)
    elif name == "op_andnps_mem":
        out[0:16] = _bit_op("and" if altered else "andn", a0, b0)
    elif name == "op_andnps_mem_disp":
        out[0:16] = _bit_op("andn", a0, b1)
    elif name == "op_andnps_self":
        out[0:16] = _bit_op("andn", a0, a0)
    elif name == "op_orps":
        out[0:16] = _bit_op("and" if altered else "or", a0, b0)
    elif name == "op_orps_mem":
        out[0:16] = _bit_op("or", a0, b0)
    elif name == "op_orps_mem_disp":
        out[0:16] = _bit_op("or", a0, b1)
    elif name == "op_orps_self":
        out[0:16] = _bit_op("or", a0, a0)
    elif name == "op_xorps":
        out[0:16] = _bit_op("or" if altered else "xor", a0, b0)
    elif name == "op_xorps_mem":
        out[0:16] = _bit_op("xor", a0, b0)
    elif name == "op_xorps_mem_disp":
        out[0:16] = _bit_op("xor", a0, b1)
    elif name == "op_xorps_self":
        out[0:16] = bytes(16)
    elif name == "op_xorps_zero_logic":
        t = _bit_op("andn", bytes(16), a0)
        out[0:16] = _bit_op("or", t, b0)
    elif name == "op_xorps_zero_arith":
        zero = bytes(16)
        added, wild1 = ref.pack(ref.apply("add", zero, a0, ref.DEFAULT_MODE))
        mulled, wild2 = ref.pack(ref.apply("mul", added, b0, ref.DEFAULT_MODE))
        assert not any(wild1) and not any(wild2)
        out[0:16] = mulled
    elif name == "op_logic_pipeline":
        x0 = _bit_op("and", a0, b0)
        x2 = _bit_op("and" if altered else "andn", a1, b1)
        x5 = _bit_op("xor", _bit_op("or", bytes(16), x0), x2)
        low, wild1 = ref.pack(ref.apply("add", x5, b0, ref.DEFAULT_MODE))
        high, wild2 = ref.pack(ref.apply("sub", x0, x2, ref.DEFAULT_MODE))
        assert not any(wild1) and not any(wild2)
        out[0:16] = low
        out[16:32] = high
    elif name == "op_logic_discarded_read":
        out[0:16] = a0
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
    payload = logic_pairs()[0][0]
    buf_a, ptr_a, _, _, _ = base.alloc_aligned_buffer(payload)
    buf_out, ptr_out, o_pre, o_off, o_post = base.alloc_aligned_buffer(FILL)
    fns["op_logic_discarded_read"](ptr_a, ptr_a, ptr_out)
    got_ok = base.check_guards_and_read(buf_out, o_pre, o_off, o_post, "output")
    if got_ok != payload[:16] + FILL[16:]:
        raise AssertionError(f"{label} op_logic_discarded_read(valid): {got_ok.hex()}")

    buf_out2, ptr_out2, o_pre2, o_off2, o_post2 = base.alloc_aligned_buffer(FILL)
    try:
        fns["op_logic_discarded_read"](ptr_a, 0, ptr_out2)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} op_logic_discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_logic_discarded_read(NULL) dropped faulting m128 read")
    if base.check_guards_and_read(buf_out2, o_pre2, o_off2, o_post2, "output") != FILL:
        raise AssertionError(f"{label} op_logic_discarded_read(NULL) wrote output after fault")


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

    # Negative comparator mutant: swapping (~dst) & src to dst & (~src) must diverge on reference.
    a0 = pk_d(0xF0F0F0F0, 0x7FC00001, 0x80000000, 0x00000001)
    b0 = pk_d(0x0FF00FF0, 0x7F800001, 0x00000000, 0xFFFFFFFF)
    if _bit_op("andn", a0, b0) == _bit_op("andn", b0, a0):
        raise AssertionError("negative andnps symmetry mutant failed to diverge")
    return diverged


def compile_o2_and_verify(original_path, native_fns, cxx_path, pairs):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "sse_fp_logic_o2.cpp"
        dll_path = tmp / "sse_fp_logic_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'sse_fp_logic_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",  # cl and link can use different encodings for localized output.
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
    st_xmm0 = Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    ret_ins = Instruction.create(Code.RETNQ)
    ret_bytes = encode_seq(ret_ins)

    # 1. VEX / EVEX / XOP rejections.
    nonlegacy_cases = [
        ("vandps_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VANDPS_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vandnps_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VANDNPS_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vorps_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VORPS_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vxorps_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VXORPS_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vandps_evex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.EVEX_VANDPS_XMM_K1Z_XMM_XMMM128B32, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
    ]

    # 2. Undefined destination, nonvolatile XMM, alignment, memory base, store, and branch checks.
    seg_andps = Instruction.create_reg_mem(Code.ANDPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX))
    seg_andps.segment_prefix = R.GS
    other_cases = [
        ("undef_andps_dst", (ld_xmm1, Instruction.create_reg_reg(
            Code.ANDPS_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins), "undefined vector register xmm0"),
        ("undef_andnps_dst", (ld_xmm1, Instruction.create_reg_reg(
            Code.ANDNPS_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins), "undefined vector register xmm0"),
        ("undef_orps_dst", (ld_xmm1, Instruction.create_reg_reg(
            Code.ORPS_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins), "undefined vector register xmm0"),
        ("undef_xorps_nonself_dst", (ld_xmm1, Instruction.create_reg_reg(
            Code.XORPS_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins), "undefined vector register xmm0"),
        ("undef_andps_self", (Instruction.create_reg_reg(
            Code.ANDPS_XMM_XMMM128, R.XMM0, R.XMM0), ret_ins), "undefined vector register xmm0"),
        ("undef_andnps_self", (Instruction.create_reg_reg(
            Code.ANDNPS_XMM_XMMM128, R.XMM0, R.XMM0), ret_ins), "undefined vector register xmm0"),
        ("undef_orps_self", (Instruction.create_reg_reg(
            Code.ORPS_XMM_XMMM128, R.XMM0, R.XMM0), ret_ins), "undefined vector register xmm0"),
        ("undef_andps_src", (ld_xmm0, Instruction.create_reg_reg(
            Code.ANDPS_XMM_XMMM128, R.XMM0, R.XMM3), ret_ins), "undefined vector register xmm3"),
        ("nonvol_xorps_self_xmm6", (Instruction.create_reg_reg(
            Code.XORPS_XMM_XMMM128, R.XMM6, R.XMM6), ret_ins), "nonvolatile vector register"),
        ("nonvol_andps_xmm7", (ld_xmm0, Instruction.create_reg_reg(
            Code.ANDPS_XMM_XMMM128, R.XMM7, R.XMM0), ret_ins), "nonvolatile vector register"),
        ("nonvol_orps_src_xmm8", (ld_xmm0, Instruction.create_reg_reg(
            Code.ORPS_XMM_XMMM128, R.XMM0, R.XMM8), ret_ins), "nonvolatile vector register"),
        ("unaligned_andps_m128", (ld_xmm0, Instruction.create_reg_mem(
            Code.ANDPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=4, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("unaligned_andnps_m128", (ld_xmm0, Instruction.create_reg_mem(
            Code.ANDNPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=8, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("unaligned_orps_m128", (ld_xmm0, Instruction.create_reg_mem(
            Code.ORPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=-4, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("unaligned_xorps_m128", (ld_xmm0, Instruction.create_reg_mem(
            Code.XORPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=1, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("non_r8_store", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.RCX), R.XMM0), ret_ins), "non-r8 store memory operand"),
        ("rip_rel_andps", (ld_xmm0, Instruction.create_reg_mem(
            Code.ANDPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RIP, displ=0x1020, displ_size=4)), ret_ins),
            "unsupported memory operand"),
        ("sib_index_orps", (ld_xmm0, Instruction.create_reg_mem(
            Code.ORPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, R.RDX, 4)), ret_ins),
            "unsupported memory operand"),
        ("bad_base_rax", (ld_xmm0, Instruction.create_reg_mem(
            Code.XORPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RAX)), ret_ins), "unsupported memory base"),
        ("nonvol_base_rbx", (ld_xmm0, Instruction.create_reg_mem(
            Code.ANDNPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RBX)), ret_ins), "nonvolatile GPR base"),
        ("segment_prefix_andps", (ld_xmm0, seg_andps, st_xmm0, ret_ins), "unsupported instruction prefix"),
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
        entry = export_rva(pe, "op_logic_pipeline")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        orig_end = struct.unpack_from("<I", records, rec_idx + 4)[0]
        max_len = orig_end - entry

    other_map = {lbl: seq for lbl, seq, _ in other_cases}
    pe_patches = [
        ("pe_vandps_vex", encode_seq(*nonlegacy_cases[0][1]), "non-legacy encoding"),
        ("pe_vandps_evex", encode_seq(*nonlegacy_cases[4][1]), "non-legacy encoding"),
        ("pe_undef_andnps", encode_seq(*other_map["undef_andnps_dst"]), "undefined vector register xmm0"),
        ("pe_nonvol_xorps_xmm6", encode_seq(*other_map["nonvol_xorps_self_xmm6"]), "nonvolatile vector register"),
        ("pe_unaligned_andps", encode_seq(*other_map["unaligned_andps_m128"]), "unaligned 16-byte memory operand"),
        ("pe_non_r8_store", encode_seq(*other_map["non_r8_store"]), "non-r8 store memory operand"),
        ("pe_seg_prefix", encode_seq(*other_map["segment_prefix_andps"]), "unsupported instruction prefix"),
    ]
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_sse_fp_aot.py"
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
                translate(patched_path, ["op_logic_pipeline"])
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched_path), "op_logic_pipeline", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 1 or "Traceback" in proc.stderr or "unsupported instruction prefix" not in proc.stderr:
            raise AssertionError(f"CLI rejection unexpected: rc={proc.returncode}, stderr={proc.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure left behind output file")

    print(
        f"rejections passed: {len(nonlegacy_cases)} VEX/EVEX + {len(other_cases)} IR + "
        f"3 raw + {len(pe_patches)} PE-patched + CLI"
    )


def main():
    original_path, lifted_path, altered_path, altered_lifted_path, cxx_path = (
        Path(p) for p in sys.argv[1:6]
    )
    check_rejections(original_path)

    pairs = logic_pairs()
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
        f"sse_fp_logic passed: {lifted_cases} lifted + {o2_cases} /O2 cases across {len(EXPORTS)} exports; "
        f"ALTER_IMM diverged on {diverged}"
    )


if __name__ == "__main__":
    main()
