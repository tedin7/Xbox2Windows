"""Differential, discarded-read, mutation, and rejection checks for opt-in AVX2 memory-operand lowering (--mem-ops)."""

import ctypes
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory

from iced_x86 import Code, Decoder, Formatter, FormatterSyntax, Instruction, MemoryOperand, Register as R
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from avx2_regs_check import (  # noqa: E402
    FILL,
    GUARD_POST,
    GUARD_PRE,
    assert_equal_output,
    bind_exports,
    encode_seq,
    general_pairs,
    invoke,
    perm32_pairs,
    ref_vpblendd,
    ref_vpbroadcast,
    ref_vperm2i128,
    ref_vpermd,
    ref_vpermq,
    ref_vextracti128,
    ref_vinserti128,
    ref_vpsllvd,
    ref_vpsllvq,
    ref_vpsravd,
    ref_vpsrlvd,
    ref_vpsrlvq,
    shift32_pairs,
    shift64_pairs,
)
from cpu_avx2_aot import lift_code, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXPORTS = (
    "op_extracti128_mem_lo",
    "op_extracti128_mem_hi",
    "op_extracti128_mem_disp",
    "op_inserti128_mem_lo",
    "op_inserti128_mem_hi_disp",
    "op_perm2i128_mem",
    "op_pbroadcastb_xmm_mem",
    "op_pbroadcastb_ymm_mem",
    "op_pbroadcastw_xmm_mem",
    "op_pbroadcastw_ymm_mem",
    "op_pbroadcastd_xmm_mem",
    "op_pbroadcastd_ymm_mem",
    "op_pbroadcastq_xmm_mem",
    "op_pbroadcastq_ymm_mem",
    "op_permd_mem",
    "op_permps_mem",
    "op_permq_mem",
    "op_permpd_mem",
    "op_pblendd_xmm_mem",
    "op_pblendd_ymm_mem",
    "op_psllvd_xmm_mem",
    "op_psllvd_ymm_mem",
    "op_psllvq_xmm_mem",
    "op_psllvq_ymm_mem",
    "op_psravd_xmm_mem",
    "op_psravd_ymm_mem",
    "op_psrlvd_xmm_mem",
    "op_psrlvd_ymm_mem",
    "op_psrlvq_xmm_mem",
    "op_psrlvq_ymm_mem",
    "op_discarded_mem_read",
    "op_mem_pipeline",
)


def ref_mem_pipeline(a, b):
    ymm2 = ref_vpbroadcast(b[7:], 4, True)
    ymm3 = ref_vpermq(a, 0xD8)
    ymm4 = ref_vpblendd(ymm3, b, 0x5A, True)
    ymm5 = ref_vpsravd(ymm4, b, True)
    ymm5 = ref_vinserti128(ymm5, a[3:19], 1)
    ymm0 = ref_vpermd(ymm2, ymm5)
    return ymm0[16:32] + ymm0[:16]


def reference(name, a, b, altered=False):
    if name == "op_extracti128_mem_lo":
        return a[:16] + FILL[16:]
    if name == "op_extracti128_mem_hi":
        return a[16:32] + FILL[16:]
    if name == "op_extracti128_mem_disp":
        return a[16:32] + a[:16]
    if name == "op_inserti128_mem_lo":
        return ref_vinserti128(a, b, 0xFE)
    if name == "op_inserti128_mem_hi_disp":
        return ref_vinserti128(a, b[3:19], 0xFF)
    if name == "op_perm2i128_mem":
        return ref_vperm2i128(a, b, 0x31 if altered else 0x20)
    if name == "op_pbroadcastb_xmm_mem":
        return ref_vpbroadcast(a[3:], 1, False)
    if name == "op_pbroadcastb_ymm_mem":
        return ref_vpbroadcast(a[3:], 1, True)
    if name == "op_pbroadcastw_xmm_mem":
        return ref_vpbroadcast(a[5:], 2, False)
    if name == "op_pbroadcastw_ymm_mem":
        return ref_vpbroadcast(a[5:], 2, True)
    if name == "op_pbroadcastd_xmm_mem":
        return ref_vpbroadcast(a[7:], 4, False)
    if name == "op_pbroadcastd_ymm_mem":
        return ref_vpbroadcast(a[7:], 4, True)
    if name == "op_pbroadcastq_xmm_mem":
        return ref_vpbroadcast(a[9:], 8, False)
    if name == "op_pbroadcastq_ymm_mem":
        return ref_vpbroadcast(a[9:], 8, True)
    if name in ("op_permd_mem", "op_permps_mem"):
        return ref_vpermd(a, b)
    if name == "op_permq_mem":
        return ref_vpermq(a, 0x4E if altered else 0x1B)
    if name == "op_permpd_mem":
        return ref_vpermq(a, 0x93)
    if name == "op_pblendd_xmm_mem":
        return ref_vpblendd(a, b, 0xFA, False)
    if name == "op_pblendd_ymm_mem":
        return ref_vpblendd(a, b, 0x55 if altered else 0xAA, True)
    if name == "op_psllvd_xmm_mem":
        return ref_vpsllvd(a, b, False)
    if name == "op_psllvd_ymm_mem":
        return ref_vpsllvd(a, b, True)
    if name == "op_psllvq_xmm_mem":
        return ref_vpsllvq(a, b, False)
    if name == "op_psllvq_ymm_mem":
        return ref_vpsllvq(a, b, True)
    if name == "op_psravd_xmm_mem":
        return ref_vpsravd(a, b, False)
    if name == "op_psravd_ymm_mem":
        return ref_vpsravd(a, b, True)
    if name == "op_psrlvd_xmm_mem":
        return ref_vpsrlvd(a, b, False)
    if name == "op_psrlvd_ymm_mem":
        return ref_vpsrlvd(a, b, True)
    if name == "op_psrlvq_xmm_mem":
        return ref_vpsrlvq(a, b, False)
    if name == "op_psrlvq_ymm_mem":
        return ref_vpsrlvq(a, b, True)
    if name == "op_discarded_mem_read":
        return a
    if name == "op_mem_pipeline":
        return ref_mem_pipeline(a, b)
    raise ValueError(f"unknown export {name}")


def inputs_for(name):
    if name in ("op_permd_mem", "op_permps_mem", "op_mem_pipeline"):
        return perm32_pairs() + general_pairs()
    if name in ("op_psllvd_xmm_mem", "op_psllvd_ymm_mem", "op_psravd_xmm_mem", "op_psravd_ymm_mem",
                "op_psrlvd_xmm_mem", "op_psrlvd_ymm_mem"):
        return shift32_pairs() + general_pairs()
    if name in ("op_psllvq_xmm_mem", "op_psllvq_ymm_mem", "op_psrlvq_xmm_mem", "op_psrlvq_ymm_mem"):
        return shift64_pairs() + general_pairs()
    return general_pairs()


def host_has_avx2(lifted_lib):
    if hasattr(lifted_lib, "host_has_avx2"):
        lifted_lib.host_has_avx2.argtypes = []
        lifted_lib.host_has_avx2.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_avx2())
    try:
        return " avx2" in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace")
    except OSError:
        return False


def check_translator(original_path):
    """Validate default allow_memory=False rejections and --mem-ops acceptance/rejections."""
    source_cpp, _ = translate(original_path, EXPORTS, allow_memory=True)
    if "host_has_avx2" not in source_cpp:
        raise AssertionError("translated C++ missing host_has_avx2")

    ld_ymm0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX))
    ld_ymm1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM1, MemoryOperand(R.RCX))
    ret_ins = Instruction.create(Code.RETNQ)

    # All 17 memory forms must be rejected when allow_memory=False (default).
    seventeen_mem_forms = [
        ("vextracti128_mem", (ld_ymm0, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTI128_XMMM128_YMM_IMM8, MemoryOperand(R.R8), R.YMM0, 1), ret_ins)),
        ("vinserti128_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VINSERTI128_YMM_YMM_XMMM128_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RDX), 1), ret_ins)),
        ("vperm2i128_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VPERM2I128_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RDX), 0x20), ret_ins)),
        ("vpbroadcastb_mem", (Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTB_YMM_XMMM8, R.YMM0, MemoryOperand(R.RCX)), ret_ins)),
        ("vpbroadcastw_mem", (Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTW_YMM_XMMM16, R.YMM0, MemoryOperand(R.RCX)), ret_ins)),
        ("vpbroadcastd_mem", (Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTD_YMM_XMMM32, R.YMM0, MemoryOperand(R.RCX)), ret_ins)),
        ("vpbroadcastq_mem", (Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTQ_YMM_XMMM64, R.YMM0, MemoryOperand(R.RCX)), ret_ins)),
        ("vpermd_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VPERMD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX)), ret_ins)),
        ("vpermpd_mem", (Instruction.create_reg_mem_u32(
            Code.VEX_VPERMPD_YMM_YMMM256_IMM8, R.YMM0, MemoryOperand(R.RCX), 0x1B), ret_ins)),
        ("vpermps_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VPERMPS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX)), ret_ins)),
        ("vpermq_mem", (Instruction.create_reg_mem_u32(
            Code.VEX_VPERMQ_YMM_YMMM256_IMM8, R.YMM0, MemoryOperand(R.RCX), 0x1B), ret_ins)),
        ("vpblendd_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VPBLENDD_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RDX), 0xAA), ret_ins)),
        ("vpsllvd_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VPSLLVD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX)), ret_ins)),
        ("vpsllvq_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VPSLLVQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX)), ret_ins)),
        ("vpsravd_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VPSRAVD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX)), ret_ins)),
        ("vpsrlvd_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VPSRLVD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX)), ret_ins)),
        ("vpsrlvq_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VPSRLVQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX)), ret_ins)),
    ]
    for label, seq in seventeen_mem_forms:
        raw = encode_seq(*seq)
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if "unsupported memory operand for register-domain AVX2 instruction" not in str(exc):
                raise AssertionError(f"{label} default reject: got {exc!r}") from exc
        else:
            raise AssertionError(f"{label} unexpectedly accepted with allow_memory=False")
        # Under allow_memory=True, all 17 must lift cleanly.
        src, _ = lift_code(label, 0x1000, raw, allow_memory=True)
        assert f"lifted_{label}" in src, label

    # --mem-ops genuine rejection controls: invalid store base, addressing, EVEX, YMM6..15, undef src, missing ret.
    evex_bcast = Instruction.create_reg_mem(
        Code.EVEX_VPBROADCASTD_YMM_K1Z_XMMM32, R.YMM0, MemoryOperand(R.RCX))
    evex_bcast.op_mask = R.K1
    mem_rejects = [
        ("extract_store_rcx", (ld_ymm0, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTI128_XMMM128_YMM_IMM8, MemoryOperand(R.RCX), R.YMM0, 0), ret_ins),
         "unsupported memory operand"),
        ("extract_store_rdx", (ld_ymm0, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTI128_XMMM128_YMM_IMM8, MemoryOperand(R.RDX), R.YMM0, 1), ret_ins),
         "unsupported memory operand"),
        ("extract_store_rax", (ld_ymm0, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTI128_XMMM128_YMM_IMM8, MemoryOperand(R.RAX), R.YMM0, 0), ret_ins),
         "unsupported memory operand"),
        ("rip_rel_mem", (Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTD_YMM_XMMM32, R.YMM0, MemoryOperand(R.RIP, displ=0x1010, displ_size=1)), ret_ins),
         "unsupported memory operand"),
        ("indexed_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VPERMD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RDX, R.RCX, 4)), ret_ins),
         "unsupported memory operand"),
        ("addr32_mem", (Instruction.create_reg_mem_u32(
            Code.VEX_VPERMQ_YMM_YMMM256_IMM8, R.YMM0, MemoryOperand(R.ECX), 0x1B), ret_ins),
         "unsupported memory operand"),
        ("non_arg_base_rax", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VPSLLVD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RAX)), ret_ins),
         "unsupported memory operand"),
        ("evex_bcast", (evex_bcast, ret_ins), "unsupported EVEX instruction"),
        ("nonvol_ymm6", (Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTQ_YMM_XMMM64, R.YMM6, MemoryOperand(R.RCX)), ret_ins),
         "nonvolatile vector register"),
        ("nonvol_ymm15", (ld_ymm0, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTI128_XMMM128_YMM_IMM8, MemoryOperand(R.R8), R.YMM15, 0), ret_ins),
         "undefined vector register" if False else "nonvolatile vector register"),
        ("undef_ymm3", (Instruction.create_reg_reg_mem(
            Code.VEX_VPERMD_YMM_YMM_YMMM256, R.YMM0, R.YMM3, MemoryOperand(R.RDX)), ret_ins),
         "undefined vector register ymm3"),
        ("missing_ret", (Instruction.create_reg_mem(
            Code.VEX_VPBROADCASTB_YMM_XMMM8, R.YMM0, MemoryOperand(R.RCX)),),
         "missing terminal ret"),
    ]
    for label, seq, reason in mem_rejects:
        try:
            lift_code(label, 0x1000, encode_seq(*seq), allow_memory=True)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: accepted under allow_memory=True")

    # Patched PE negative control on original_path.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_mem_pipeline")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12)
                       if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry
    with TemporaryDirectory() as folder:
        patched = Path(folder) / "patched.dll"
        for label, seq, reason in mem_rejects[:4]:
            patch = encode_seq(*seq)
            assert len(patch) <= max_len
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched.write_bytes(damaged)
            try:
                translate(patched, ["op_mem_pipeline"], allow_memory=True)
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"PE {label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"PE {label}: accepted rejected patch")

    print(f"translator validated: 17 default mem rejections, 17 --mem-ops acceptances, "
          f"{len(mem_rejects)} unit + 4 PE --mem-ops rejections")


def check_discarded_read(native, lifted):
    """Verify op_discarded_mem_read faults on NULL rdx after storing the low 16 bytes."""
    payload = bytes(range(1, 33))
    for label, funcs in (("native", native), ("lifted", lifted)):
        fn = funcs["op_discarded_mem_read"]
        buf_a = ctypes.create_string_buffer(payload, 32)
        for readable in (True, False):
            guarded = ctypes.create_string_buffer(GUARD_PRE + FILL + GUARD_POST, 64)
            out_ptr = ctypes.addressof(guarded) + len(GUARD_PRE)
            try:
                fn(ctypes.addressof(buf_a), ctypes.addressof(buf_a) if readable else 0, out_ptr)
            except OSError as exc:
                assert not readable and "access violation reading" in str(exc).lower(), (label, exc)
            else:
                assert readable, f"{label} op_discarded_mem_read discarded faulting read"
            raw = guarded.raw
            assert raw[:16] == GUARD_PRE and raw[48:] == GUARD_POST, f"{label} guard corruption"
            expected = payload if readable else (payload[:16] + FILL[16:])
            assert_equal_output(label, "op_discarded_mem_read", readable, raw[16:48], expected)


def check_differential(original_path, lifted_path, altered_lifted_path):
    native = bind_exports(original_path, "", EXPORTS)
    lifted = bind_exports(lifted_path, "lifted_", EXPORTS)
    altered = bind_exports(altered_lifted_path, "lifted_", EXPORTS)

    total = 0
    for name in EXPORTS:
        for idx, (a, b) in enumerate(inputs_for(name)):
            want = reference(name, a, b)
            assert_equal_output("native", name, idx, invoke(native[name], a, b), want)
            assert_equal_output("lifted", name, idx, invoke(lifted[name], a, b), want)
            total += 1

    # Altered build checks: op_perm2i128_mem, op_permq_mem, op_pblendd_ymm_mem diverge from original.
    altered_diffs = {}
    for name in ("op_perm2i128_mem", "op_permq_mem", "op_pblendd_ymm_mem"):
        diffs = 0
        for idx, (a, b) in enumerate(inputs_for(name)):
            want_alt = reference(name, a, b, altered=True)
            want_orig = reference(name, a, b, altered=False)
            got_alt = invoke(altered[name], a, b)
            assert_equal_output("altered", name, idx, got_alt, want_alt)
            diffs += got_alt != want_orig
        assert diffs > 0, f"altered {name} did not diverge"
        altered_diffs[name] = diffs

    check_discarded_read(native, lifted)

    # Mutant checks on XMM zero-extension and extracti128 lane order.
    a, b = general_pairs()[0]
    if invoke(lifted["op_pbroadcastb_xmm_mem"], a, b)[16:] != b"\x00" * 16:
        raise AssertionError("vpbroadcastb xmm failed to zero upper 128 bits")
    if invoke(lifted["op_pblendd_xmm_mem"], a, b)[16:] != b"\x00" * 16:
        raise AssertionError("vpblendd xmm failed to zero upper 128 bits")
    if invoke(lifted["op_extracti128_mem_disp"], a, b) == a:
        raise AssertionError("op_extracti128_mem_disp did not swap 128-bit lanes")

    print(f"{total} native/lifted/reference cases across {len(EXPORTS)} exports matched bit-for-bit; "
          f"discarded-read ordering + XMM zero-ext + altered controls ({altered_diffs}) passed")


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

    original_path, lifted_path, altered_lifted_path = (Path(p) for p in sys.argv[1:4])
    check_translator(original_path)

    lifted_lib = ctypes.CDLL(str(lifted_path.resolve()))
    if not host_has_avx2(lifted_lib):
        print("Host lacks CPUID.01H:ECX[28] + XGETBV + CPUID.07H:EBX[5] AVX2; skipping native differential check")
        sys.exit(77)

    check_differential(original_path, lifted_path, altered_lifted_path)
    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
