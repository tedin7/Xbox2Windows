"""Differential checks, ALTER_IMM/mutation controls and rejection controls for AVX VEX.128 pcmp*str* AOT."""

import ctypes
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Encoder, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_avx_pcmpstr_aot import lift_code, translate
from cpu_leaf_aot import export_rva


GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))
SENTINEL_HI = bytes(range(0xA0, 0xB0))

# (name, orig_imm, alt_imm, is_explicit, is_mask, is_word, is_alias_self, check_xmm1, check_unaligned_rdx)
SPECS = (
    ("vpcmpestri_ub_eqany_pos_lsb", 0x00, 0x01, True, False, False, False, False, False),
    ("vpcmpestri_sw_ranges_neg_msb", 0x57, None, True, False, True, False, False, False),
    ("vpcmpestri_sb_eqeach_mpos_msb", 0x6A, None, True, False, False, False, False, False),
    ("vpcmpestri_uw_eqord_mneg_lsb", 0x3D, None, True, False, True, False, False, False),
    ("vpcmpestri_mem_ub_eqany_pos_lsb", 0x00, None, True, False, False, False, False, False),
    ("vpcmpestri_alias_xmm0", 0x1C, None, True, False, False, True, False, False),
    ("vpcmpestrm_ub_eqany_pos_bit", 0x00, None, True, True, False, False, True, False),
    ("vpcmpestrm_sb_ranges_neg_exp", 0x56, 0x16, True, True, False, False, False, False),
    ("vpcmpestrm_uw_eqeach_mpos_exp", 0x69, None, True, True, True, False, True, False),
    ("vpcmpestrm_sw_eqord_mneg_bit", 0x3F, None, True, True, True, False, False, False),
    ("vpcmpestrm_mem_ub_eqany_exp", 0x40, None, True, True, False, False, False, False),
    ("vpcmpestrm_xmm0_self", 0x40, None, True, True, False, False, False, False),
    ("vpcmpistri_ub_eqord_pos_lsb", 0x8C, 0xCC, False, False, False, False, False, False),
    ("vpcmpistri_sb_ranges_mneg_msb", 0x76, None, False, False, False, False, False, False),
    ("vpcmpistri_uw_eqany_neg_msb", 0x51, None, False, False, True, False, False, False),
    ("vpcmpistri_sw_eqeach_mpos_lsb", 0x2B, None, False, False, True, False, False, False),
    ("vpcmpistri_mem_ub_eqany_pos_lsb", 0x00, None, False, False, False, False, False, True),
    ("vpcmpistrm_ub_eqeach_pos_exp", 0x48, None, False, True, False, False, False, False),
    ("vpcmpistrm_sw_eqany_mneg_exp", 0x73, 0x33, False, True, True, False, True, False),
    ("vpcmpistrm_uw_ranges_mpos_bit", 0x25, None, False, True, True, False, False, False),
    ("vpcmpistrm_sb_eqord_neg_bit", 0x1E, None, False, True, False, False, False, False),
    ("vpcmpistrm_mem_ub_eqord_exp", 0x4C, None, False, True, False, False, False, True),
    ("vpcmpestri64_ub_eqany_pos_lsb", 0x00, 0x01, True, False, False, False, False, False),
    ("vpcmpestri64_sb_eqeach_mpos_msb", 0x6A, None, True, False, False, False, False, False),
    ("vpcmpestri64_mem_sw_ranges_neg_msb", 0x57, None, True, False, True, False, False, False),
    ("vpcmpestrm64_ub_eqany_pos_bit", 0x00, None, True, True, False, False, True, False),
    ("vpcmpestrm64_uw_eqeach_mpos_exp", 0x69, 0x29, True, True, True, False, True, False),
)

ALL_EXPORTS = tuple(row[0] for row in SPECS) + ("vpcmpistrm_discarded_read",)


def is_len64(name: str) -> bool:
    """VEX.W=1 exports read full signed 64-bit RAX/RDX lengths."""
    return "64_" in name



BYTE_PAIRS = (
    (b"aeiou\x00xxxxxxxxxx", b"hello world!\x00xyz"),
    (b"azAZ09\x80\x7f\x00abcdefg", b"mZ!\x80\x00hello\x7f\xff09ab"),
    (b"ab" + b"\x00" * 14, b"xxabxabyab" + b"\x00" * 6),
    (b"abc" + b"\x00" * 13, b"0123456789abcdab"),
    (b"\x00" * 16, b"\x00" * 16),
    (b"za" + b"\x80\x7f" + b"\x00" * 12, b"\x80\xff\x00\x01\x7f\x81azbcdefghi"),
    (b"0123456789abcdef", b"0123456789abcdef"),
    (b"0123456789abcdef", b"012345X789abcdef"),
    (b"A" * 15 + b"\x00", b"A" * 15 + b"\x00"),
)

WORD_PAIRS = (
    (
        struct.pack("<8H", 10, 20, 30, 0x8000, 0, 1, 2, 3),
        struct.pack("<8H", 99, 20, 0x8000, 10, 0, 5, 10, 30),
    ),
    (
        struct.pack("<8h", -100, -1, 10, 50, -32768, 32767, 0, 0),
        struct.pack("<8h", -101, -50, 0, 25, -32768, 32767, 100, -1),
    ),
    (
        struct.pack("<8H", 1, 2, 3, 0, 0, 0, 0, 0),
        struct.pack("<8H", 9, 1, 2, 3, 9, 9, 1, 2),
    ),
    (
        struct.pack("<8H", 0, 0, 0, 0, 0, 0, 0, 0),
        struct.pack("<8H", 1, 2, 3, 4, 5, 6, 7, 8),
    ),
    (
        struct.pack("<8H", 10, 10, 0, 0, 0, 0, 0, 0),
        struct.pack("<8H", 10, 0, 0, 0, 0, 0, 0, 0),
    ),
)

EXPLICIT_LEN_PAIRS = (
    (0, 0),
    (0, 5),
    (1, 5),
    (1, 15),
    (3, 16),
    (4, 8),
    (8, 8),
    (16, 16),
    (17, 5),
    (-1, 5),
    (4, -5),
    (-17, -17),
    (-0x80000000, 6),
)

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

# Lengths whose low 32 bits differ from the full signed 64-bit value, so a 32-bit
# truncation of RAX/RDX gives a different clamped length than the real W=1 semantics.
EXPLICIT_LEN64_PAIRS = EXPLICIT_LEN_PAIRS + (
    (0x1_0000_0003, 5),
    (5, 0x1_0000_0003),
    (0x1_0000_0000, 0x1_0000_0000),
    (0xFFFFFFFF, 6),
    (7, 0xFFFFFFFF),
    (-0xFFFFFFFB, 6),
    (4, -0x1_0000_0004),
    (INT64_MIN, 4),
    (4, INT64_MIN),
    (INT64_MAX, INT64_MAX),
    (-1, INT64_MIN),
)


def sdm_pcmp(a_bytes: bytes, b_bytes: bytes, imm: int, is_explicit: bool, raw_la: int = 0, raw_lb: int = 0,
             len64: bool = False):
    """Independent Intel SDM reference model for VPCMPESTRI/M (W=0 or W=1) and VPCMPISTRI/M."""
    is_word = bool(imm & 1)
    is_signed = bool(imm & 2)
    limit = 8 if is_word else 16
    agg = (imm >> 2) & 3
    pol = (imm >> 4) & 3
    high_sel = bool(imm & 0x40)

    fmt = f"<{limit}{'h' if is_signed else 'H'}" if is_word else f"<{limit}{'b' if is_signed else 'B'}"
    elems_a = list(struct.unpack(fmt, a_bytes))
    elems_b = list(struct.unpack(fmt, b_bytes))

    if is_explicit:
        def clamp_len(raw: int) -> int:
            if len64:
                signed = struct.unpack("<q", struct.pack("<Q", raw & 0xffffffffffffffff))[0]
            else:
                signed = struct.unpack("<i", struct.pack("<I", raw & 0xffffffff))[0]
            return min(abs(signed), limit)

        len1 = clamp_len(raw_la)
        len2 = clamp_len(raw_lb)
    else:
        len1 = next((i for i, val in enumerate(elems_a) if val == 0), limit)
        len2 = next((i for i, val in enumerate(elems_b) if val == 0), limit)

    int_res1 = 0
    for i in range(limit):
        if agg == 0:
            bit = i < len2 and any(elems_a[j] == elems_b[i] for j in range(len1))
        elif agg == 1:
            bit = i < len2 and any(
                elems_a[j] <= elems_b[i] <= elems_a[j + 1]
                for j in range(0, len1 - 1, 2)
            )
        elif agg == 2:
            if i < len1 and i < len2:
                bit = elems_a[i] == elems_b[i]
            else:
                bit = i >= len1 and i >= len2
        else:
            bit = all(
                (i + j) < len2 and elems_a[j] == elems_b[i + j]
                for j in range(min(len1, limit - i))
            )
        if bit:
            int_res1 |= 1 << i

    if pol == 1:
        int_res2 = int_res1 ^ ((1 << limit) - 1)
    elif pol == 3:
        int_res2 = int_res1 ^ ((1 << len2) - 1)
    else:
        int_res2 = int_res1

    if int_res2 == 0:
        index = limit
    elif not high_sel:
        index = (int_res2 & -int_res2).bit_length() - 1
    else:
        index = int_res2.bit_length() - 1

    if not high_sel:
        mask_bytes = struct.pack("<H", int_res2) + b"\x00" * 14
    elif not is_word:
        mask_bytes = bytes(0xff if ((int_res2 >> k) & 1) else 0x00 for k in range(16))
    else:
        mask_bytes = b"".join(struct.pack("<H", 0xffff if ((int_res2 >> k) & 1) else 0x0000) for k in range(8))

    # Packed EFLAGS: CF(0), ZF(1), SF(2), OF(3); PF(4) must always be cleared to 0.
    flags = (
        (1 if int_res2 != 0 else 0)
        | (2 if len2 < limit else 0)
        | (4 if len1 < limit else 0)
        | (8 if (int_res2 & 1) else 0)
    )
    return index, mask_bytes, flags


def bind_exports(dll_path: Path, prefix: str = ""):
    lib = ctypes.CDLL(str(dll_path.resolve()))
    funcs = {}
    for name in ALL_EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = ctypes.c_uint64
        funcs[name] = fn
    return lib, funcs


def invoke_case(fn, a_bytes: bytes, b_bytes: bytes, is_exp: bool, la: int, lb: int, b_align: int = 0,
                len64: bool = False):
    a_buf = ctypes.create_string_buffer(a_bytes + SENTINEL_HI, 32)
    b_raw = ctypes.create_string_buffer(b"\xCC" * b_align + b_bytes + b"\xCC" * 8, 32)
    if len64:
        header = b"\x5A" * 16 + SENTINEL_HI + b"\xAA" * 16 + struct.pack("<qq", la, lb)
    else:
        header = struct.pack("<ii", la, lb) + b"\x5A" * 8 + SENTINEL_HI + b"\xAA" * 32
    guarded = ctypes.create_string_buffer(GUARD_PRE + header + GUARD_POST, 96)
    io_ptr = ctypes.addressof(guarded) + len(GUARD_PRE)
    flags = fn(ctypes.addressof(a_buf), ctypes.addressof(b_raw) + b_align, io_ptr)
    raw = guarded.raw
    if raw[:16] != GUARD_PRE or raw[80:] != GUARD_POST:
        raise AssertionError("guard bytes corrupted around 64-byte io buffer")
    return flags, raw[16:80]


def verify_output(label: str, name: str, is_mask: bool, check_xmm1: bool,
                  a_bytes: bytes, b_bytes: bytes, imm: int, is_exp: bool, la: int, lb: int,
                  got_flags: int, io_bytes: bytes):
    b_eff = a_bytes if "alias_xmm0" in name else b_bytes
    exp_idx, exp_mask, exp_flags = sdm_pcmp(a_bytes, b_eff, imm, is_exp, la, lb, is_len64(name))
    if got_flags != exp_flags:
        raise AssertionError(
            f"{label} {name} imm={imm:#x} la={la} lb={lb}: flags={got_flags:#x} != exp={exp_flags:#x}"
        )
    if is_mask:
        got_ymm = io_bytes[:32]
        exp_ymm = exp_mask + b"\x00" * 16
        if got_ymm != exp_ymm:
            raise AssertionError(
                f"{label} {name} imm={imm:#x} la={la} lb={lb}: ymm0={got_ymm.hex()} != exp={exp_ymm.hex()}"
            )
        if check_xmm1 and io_bytes[32:48] != a_bytes:
            raise AssertionError(f"{label} {name}: xmm1 clobbered during implicit XMM0 write")
    else:
        got_rcx = struct.unpack("<Q", io_bytes[:8])[0]
        if got_rcx != exp_idx:
            raise AssertionError(
                f"{label} {name} imm={imm:#x} la={la} lb={lb}: rcx={got_rcx:#x} != exp={exp_idx:#x}"
            )


def check_differential(native_path: Path, lifted_path: Path, altered_path: Path) -> None:
    lifted_lib, lifted = bind_exports(lifted_path, "lifted_")
    lifted_lib.host_has_avx.restype = ctypes.c_uint32
    if not lifted_lib.host_has_avx():
        print("host lacks AVX/OSXSAVE; skipping native differential check")
        raise SystemExit(77)

    _, native = bind_exports(native_path)
    _, altered = bind_exports(altered_path, "lifted_")

    for label, functions in (("native", native), ("lifted", lifted)):
        fn = functions["vpcmpistrm_discarded_read"]
        source = ctypes.create_string_buffer(16)
        result = ctypes.c_uint32()
        ptr = ctypes.addressof(source)
        assert fn(ptr, ptr, ctypes.addressof(result)) == 0x55667788
        assert result.value == 0x55667788
        result.value = 0
        try:
            fn(ptr, 0, ctypes.addressof(result))
        except OSError as error:
            assert "access violation reading" in str(error).lower(), (label, error)
        else:
            raise AssertionError(f"{label} discarded the faulting memory compare")
        assert result.value == 0x11223344, (label, result.value)

    total = 0
    for name, imm, _, is_exp, is_mask, is_word, is_alias, check_xmm1, check_unaligned in SPECS:
        vec_pairs = WORD_PAIRS if is_word else BYTE_PAIRS
        len_pairs = (EXPLICIT_LEN64_PAIRS if is_len64(name) else EXPLICIT_LEN_PAIRS) if is_exp else ((0, 0),)
        for a_bytes, b_bytes in vec_pairs:
            for la, lb in len_pairs:
                fn_n, fn_l = native[name], lifted[name]
                flags_n, io_n = invoke_case(fn_n, a_bytes, b_bytes, is_exp, la, lb, len64=is_len64(name))
                flags_l, io_l = invoke_case(fn_l, a_bytes, b_bytes, is_exp, la, lb, len64=is_len64(name))
                verify_output("native", name, is_mask, check_xmm1, a_bytes, b_bytes, imm, is_exp, la, lb, flags_n, io_n)
                verify_output("lifted", name, is_mask, check_xmm1, a_bytes, b_bytes, imm, is_exp, la, lb, flags_l, io_l)
                total += 1
        if check_unaligned:
            a_bytes, b_bytes = vec_pairs[0]
            for off in (1, 3):
                flags_n, io_n = invoke_case(native[name], a_bytes, b_bytes, is_exp, 0, 0, b_align=off)
                flags_l, io_l = invoke_case(lifted[name], a_bytes, b_bytes, is_exp, 0, 0, b_align=off)
                verify_output("native-unaligned", name, is_mask, False, a_bytes, b_bytes, imm, is_exp, 0, 0, flags_n, io_n)
                verify_output("lifted-unaligned", name, is_mask, False, a_bytes, b_bytes, imm, is_exp, 0, 0, flags_l, io_l)
                total += 1

    # Verify ALTER_IMM exports match altered reference and diverge from original reference.
    diverged = {}
    for name, orig_imm, alt_imm, is_exp, is_mask, is_word, _, _, _ in SPECS:
        if alt_imm is None:
            continue
        vec_pairs = WORD_PAIRS if is_word else BYTE_PAIRS
        len_pairs = (EXPLICIT_LEN64_PAIRS if is_len64(name) else EXPLICIT_LEN_PAIRS) if is_exp else ((0, 0),)
        diffs = 0
        for a_bytes, b_bytes in vec_pairs:
            for la, lb in len_pairs:
                flags_a, io_a = invoke_case(altered[name], a_bytes, b_bytes, is_exp, la, lb, len64=is_len64(name))
                verify_output("altered", name, is_mask, False, a_bytes, b_bytes, alt_imm, is_exp, la, lb, flags_a, io_a)
                orig_idx, orig_mask, orig_flags = sdm_pcmp(a_bytes, b_bytes, orig_imm, is_exp, la, lb, is_len64(name))
                alt_out = io_a[:16] if is_mask else struct.unpack("<Q", io_a[:8])[0]
                orig_out = orig_mask if is_mask else orig_idx
                if flags_a != orig_flags or alt_out != orig_out:
                    diffs += 1
        if not diffs:
            raise AssertionError(f"altered control {name} never diverged from original")
        diverged[name] = diffs

    # Negative comparator control: mutating one bit of expected output must raise AssertionError.
    sample_a, sample_b = BYTE_PAIRS[0]
    flags_n, io_n = invoke_case(native[ALL_EXPORTS[6]], sample_a, sample_b, True, 5, 12)
    mutated = bytearray(io_n)
    mutated[0] ^= 0x01
    try:
        verify_output("mutation", ALL_EXPORTS[6], True, False, sample_a, sample_b, 0x00, True, 5, 12, flags_n, bytes(mutated))
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output was not caught by comparator")

    # W=1 negative control: a reference that truncates RAX/RDX to 32 bits must disagree with
    # the native and lifted W=1 exports on the lengths above 32 bits, or the cases prove nothing.
    truncated = {}
    for name, imm, _, _, is_mask, is_word, _, _, _ in SPECS:
        if not is_len64(name):
            continue
        vec_pairs = WORD_PAIRS if is_word else BYTE_PAIRS
        differing = 0
        for a_bytes, b_bytes in vec_pairs:
            for la, lb in EXPLICIT_LEN64_PAIRS:
                for functions in (native, lifted):
                    flags, io = invoke_case(functions[name], a_bytes, b_bytes, True, la, lb, len64=True)
                    b_eff = b_bytes
                    idx, mask, exp_flags = sdm_pcmp(a_bytes, b_eff, imm, True, la, lb, False)
                    got = io[:16] if is_mask else struct.unpack("<Q", io[:8])[0]
                    if flags != exp_flags or got != (mask if is_mask else idx):
                        differing += 1
        if not differing:
            raise AssertionError(f"{name}: 32-bit length truncation control never diverged")
        truncated[name] = differing

    w1_name = "vpcmpestrm64_ub_eqany_pos_bit"
    flags_w, io_w = invoke_case(native[w1_name], sample_a, sample_b, True, 5, 12, len64=True)
    mutated = bytearray(io_w)
    mutated[0] ^= 0x01
    try:
        verify_output("mutation-w1", w1_name, True, False, sample_a, sample_b, 0x00, True, 5, 12, flags_w, bytes(mutated))
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated W=1 output was not caught by comparator")

    print(
        f"{total} VEX.128 AVX packed string compare cases across {len(SPECS)} exports matched native and reference; "
        f"ALTER_IMM diverged on {diverged}; "
        f"32-bit length truncation of the W=1 reference diverged on {truncated}; output mutation controls caught"
    )


def encode_seq(*instructions) -> bytes:
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path: Path) -> None:
    ld_xmm0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_xmm1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RCX))
    ld_xmm2 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM2, MemoryOperand(R.RDX))
    ld_ymm1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM1, MemoryOperand(R.RCX))
    ld_ymm2 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM2, MemoryOperand(R.RDX))
    ld_eax = Instruction.create_reg_mem(Code.MOV_R32_RM32, R.EAX, MemoryOperand(R.R8))
    ld_edx = Instruction.create_reg_mem(Code.MOV_R32_RM32, R.EDX, MemoryOperand(R.R8, displ=4, displ_size=1))
    ret_ins = Instruction.create(Code.RETNQ)
    ld_preamble = encode_seq(ld_xmm1, ld_xmm2, ld_eax, ld_edx)
    ret_bytes = encode_seq(ret_ins)

    evex_ins = Instruction.create_reg_reg_reg(Code.EVEX_VPADDQ_XMM_K1Z_XMM_XMMM128B64, R.XMM0, R.XMM1, R.XMM2)
    evex_ins.op_mask = R.K1
    seg_ins = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RCX))
    seg_ins.segment_prefix = R.GS

    unit_cases = [
        ("undefined_rax", ret_bytes, "return with undefined RAX"),
        # Invalid VEX.256 (L=1) pcpm*str* encodings (#UD in x86-64)
        ("vex256_vpcmpestri", ld_preamble + bytes.fromhex("c4e37d61ca00") + ret_bytes, "invalid instruction"),
        ("vex256_vpcmpestrm", ld_preamble + bytes.fromhex("c4e37d60ca00") + ret_bytes, "invalid instruction"),
        ("vex256_vpcmpistri", ld_preamble + bytes.fromhex("c4e37d63ca00") + ret_bytes, "invalid instruction"),
        ("vex256_vpcmpistrm", ld_preamble + bytes.fromhex("c4e37d62ca00") + ret_bytes, "invalid instruction"),
        # Valid VEX.256 instruction out of scope
        ("vex256_vpaddq", encode_seq(ld_ymm1, ld_ymm2, Instruction.create_reg_reg_reg(
            Code.VEX_VPADDQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret_ins), "unsupported VEX.256 (L=1)"),
        # VEX.W=1 64-bit forms: supported, but only with a defined RAX length and L=0
        ("vex_w1_vpcmpestri64_undefined_rax", encode_seq(ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(
            Code.VEX_VPCMPESTRI64_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins),
         "undefined explicit length register"),
        ("vex_w1_vpcmpestrm64_undefined_rax", encode_seq(ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(
            Code.VEX_VPCMPESTRM64_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins),
         "undefined explicit length register"),
        ("vex256_w1_vpcmpestri64", ld_preamble + bytes.fromhex("c4e3fd61ca00") + ret_bytes, "invalid instruction"),
        ("vex256_w1_vpcmpestrm64", ld_preamble + bytes.fromhex("c4e3fd60ca00") + ret_bytes, "invalid instruction"),
        # Legacy SSE4.2 forms
        ("legacy_pcmpestri", encode_seq(ld_xmm1, ld_xmm2, ld_eax, ld_edx, Instruction.create_reg_reg_u32(
            Code.PCMPESTRI_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins),
         "unsupported legacy SSE4.2 packed string compare"),
        ("legacy_pcmpestrm", encode_seq(ld_xmm1, ld_xmm2, ld_eax, ld_edx, Instruction.create_reg_reg_u32(
            Code.PCMPESTRM_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins),
         "unsupported legacy SSE4.2 packed string compare"),
        ("legacy_pcmpistri", encode_seq(ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(
            Code.PCMPISTRI_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins),
         "unsupported legacy SSE4.2 packed string compare"),
        ("legacy_pcmpistrm", encode_seq(ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(
            Code.PCMPISTRM_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins),
         "unsupported legacy SSE4.2 packed string compare"),
        # EVEX
        ("evex_vpaddq", encode_seq(ld_xmm1, ld_xmm2, evex_ins, ret_ins), "unsupported EVEX instruction"),
        # Nonvolatile vector and GPR registers
        ("nonvolatile_xmm6", encode_seq(Instruction.create_reg_mem(
            Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM6, MemoryOperand(R.RCX)), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_ymm6", encode_seq(Instruction.create_reg_mem(
            Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM6, MemoryOperand(R.RCX)), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_ebx", encode_seq(Instruction.create_reg_mem(
            Code.MOV_R32_RM32, R.EBX, MemoryOperand(R.R8)), ret_ins), "nonvolatile GPR"),
        ("write_r8d", encode_seq(Instruction.create_reg_i32(
            Code.MOV_R32_IMM32, R.R8D, 1), ret_ins), "unsupported write to output buffer pointer r8"),
        # Undefined registers and flags
        ("undef_xmm3", encode_seq(ld_xmm1, Instruction.create_reg_reg_u32(
            Code.VEX_VPCMPISTRI_XMM_XMMM128_IMM8, R.XMM1, R.XMM3, 0), ret_ins), "undefined vector register ymm3"),
        ("undef_eax_est", encode_seq(ld_xmm1, ld_xmm2, ld_edx, Instruction.create_reg_reg_u32(
            Code.VEX_VPCMPESTRI_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins),
         "undefined explicit length register"),
        ("undef_flags_setb", encode_seq(Instruction.create_reg(Code.SETB_RM8, R.AL), ret_ins), "undefined flags"),
        # Bad addressing and clobbered pointer reads
        ("rip_relative", encode_seq(Instruction.create_reg_mem(
            Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RIP, displ=0x1010, displ_size=1)), ret_ins),
         "unsupported memory operand"),
        ("rsp_base", encode_seq(Instruction.create_reg_mem(
            Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RSP)), ret_ins),
         "unsupported memory operand"),
        ("indexed_mem", encode_seq(ld_xmm1, Instruction.create_reg_mem_u32(
            Code.VEX_VPCMPISTRI_XMM_XMMM128_IMM8, R.XMM1, MemoryOperand(R.RDX, R.RCX, 4), 0), ret_ins),
         "unsupported memory operand"),
        ("unscaled_indexed_mem", encode_seq(ld_xmm1, ld_eax, Instruction.create_reg_mem_u32(
            Code.VEX_VPCMPISTRI_XMM_XMMM128_IMM8, R.XMM1, MemoryOperand(R.RDX, R.RCX, 1), 0), ret_ins),
         "unsupported memory operand"),
        ("non_arg_base", encode_seq(ld_xmm1, Instruction.create_reg_mem_u32(
            Code.VEX_VPCMPISTRI_XMM_XMMM128_IMM8, R.XMM1, MemoryOperand(R.RAX), 0), ret_ins),
         "unsupported memory operand"),
        ("clobbered_rdx_mem", encode_seq(ld_xmm1, ld_edx, Instruction.create_reg_mem_u32(
            Code.VEX_VPCMPISTRI_XMM_XMMM128_IMM8, R.XMM1, MemoryOperand(R.RDX), 0), ret_ins),
         "memory operand uses clobbered pointer rdx"),
        ("clobbered_rcx_mem", encode_seq(ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(
            Code.VEX_VPCMPISTRI_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ld_xmm0, ret_ins),
         "memory operand uses clobbered pointer rcx"),
        # Prefix, out-of-scope, control-flow / ret checks
        ("segment_prefix", encode_seq(seg_ins, ret_ins), "unsupported instruction prefix"),
        ("out_of_scope_vaddps", encode_seq(ld_xmm1, ld_xmm2, Instruction.create_reg_reg_reg(
            Code.VEX_VADDPS_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2), ret_ins), "unsupported instruction"),
        ("out_of_scope_jmp", encode_seq(Instruction.create_branch(Code.JMP_REL8_64, 0x1020), ret_ins),
         "unsupported instruction"),
        ("missing_ret", encode_seq(ld_xmm0), "missing terminal ret"),
        ("after_ret", encode_seq(ld_eax, ret_ins, ld_xmm0), "unreachable instruction after ret"),
    ]

    for label, raw, expected_reason in unit_cases:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator unexpectedly accepted rejected control")

    # End-to-end PE patching controls on a copy of the built native DLL.
    pe_bytes = original_path.read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, ALL_EXPORTS[0])
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
        ("pe_vex_w1_vpcmpestri64_undefined_rax", encode_seq(ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(
            Code.VEX_VPCMPESTRI64_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins),
         "undefined explicit length register"),
        ("pe_legacy_pcmpistrm", encode_seq(ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(
            Code.PCMPISTRM_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins),
         "unsupported legacy SSE4.2 packed string compare"),
        ("pe_evex_vpaddq", encode_seq(ld_xmm1, ld_xmm2, evex_ins, ret_ins), "unsupported EVEX instruction"),
        ("pe_nonvolatile_xmm6", encode_seq(Instruction.create_reg_mem(
            Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM6, MemoryOperand(R.RCX)), ret_ins), "nonvolatile vector register"),
        ("pe_missing_ret", encode_seq(ld_xmm0), "missing terminal ret"),
    ]
    with TemporaryDirectory() as folder:
        patched_path = Path(folder) / "patched.dll"
        for label, patch, expected_reason in pe_patches:
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds export size ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, [ALL_EXPORTS[0]])
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: PE translate accepted rejected patch")

    print(f"{len(unit_cases)} unit rejection controls and {len(pe_patches)} PE-patched rejection controls passed")


if __name__ == "__main__":
    orig, lifted, alt = (Path(p) for p in sys.argv[1:4])
    check_differential(orig, lifted, alt)
    check_rejections(orig)
