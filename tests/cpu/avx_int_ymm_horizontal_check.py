"""Differential, boundary, mutation, and rejection checks for 256-bit AVX2 horizontal packed-integer ops."""

import ctypes
import os
from pathlib import Path
import random
import struct
import sys

from iced_x86 import Code, Decoder, Formatter, FormatterSyntax, Instruction, MemoryOperand, Register as R
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from avx_int_ymm_minmax_check import encode_seq, first_bad_word, invoke, load  # noqa: E402
from cpu_avx_int_aot import lift_code  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXPORTS = (
    "op_vphaddw_ymm",
    "op_vphaddd_ymm",
    "op_vphaddsw_ymm",
    "op_vphsubw_ymm",
    "op_vphsubd_ymm",
    "op_vphsubsw_ymm",
    "op_vphaddw_ymm_dst_src1",
    "op_vphaddd_ymm_dst_src2",
    "op_vphaddsw_ymm_all",
    "op_vphsubw_ymm_dst_src2",
    "op_vphaddsw_ymm_then_xmm_vphsubsw",
    "op_chain_ymm_horizontal",
)


# --- Independent Python reference: operates on each 16-byte 128-bit half independently ---

def _sat_s16(val):
    if val > 32767:
        return 32767
    if val < -32768:
        return -32768
    return val


def _hword_half(a, b, *, subtract=False, saturate=False):
    fmt = "<8h" if saturate else "<8H"
    wa, wb = struct.unpack(fmt, a), struct.unpack(fmt, b)
    out = []
    for src in (wa, wb):
        for i in range(4):
            val = src[2 * i] - src[2 * i + 1] if subtract else src[2 * i] + src[2 * i + 1]
            out.append(_sat_s16(val) if saturate else (val & 0xFFFF))
    return struct.pack(fmt, *out)


def _hdword_half(a, b, *, subtract=False):
    da, db = struct.unpack("<4I", a), struct.unpack("<4I", b)
    out = []
    for src in (da, db):
        for i in range(2):
            val = src[2 * i] - src[2 * i + 1] if subtract else src[2 * i] + src[2 * i + 1]
            out.append(val & 0xFFFFFFFF)
    return struct.pack("<4I", *out)


def _halves(fn, a, b):
    return fn(a[:16], b[:16]) + fn(a[16:32], b[16:32])


def reference(name, a, b):
    if name in ("op_vphaddw_ymm", "op_vphaddw_ymm_dst_src1"):
        return _halves(lambda x, y: _hword_half(x, y, subtract=False, saturate=False), a, b)
    if name in ("op_vphaddd_ymm", "op_vphaddd_ymm_dst_src2"):
        return _halves(lambda x, y: _hdword_half(x, y, subtract=False), a, b)
    if name == "op_vphaddsw_ymm":
        return _halves(lambda x, y: _hword_half(x, y, subtract=False, saturate=True), a, b)
    if name == "op_vphaddsw_ymm_all":
        return _halves(lambda x, y: _hword_half(x, y, subtract=False, saturate=True), a, a)
    if name in ("op_vphsubw_ymm", "op_vphsubw_ymm_dst_src2"):
        return _halves(lambda x, y: _hword_half(x, y, subtract=True, saturate=False), a, b)
    if name == "op_vphsubd_ymm":
        return _halves(lambda x, y: _hdword_half(x, y, subtract=True), a, b)
    if name == "op_vphsubsw_ymm":
        return _halves(lambda x, y: _hword_half(x, y, subtract=True, saturate=True), a, b)
    if name == "op_vphaddsw_ymm_then_xmm_vphsubsw":
        top = _halves(lambda x, y: _hword_half(x, y, subtract=False, saturate=True), a, b)
        return _hword_half(top[:16], b[:16], subtract=True, saturate=True) + b"\x00" * 16
    if name == "op_chain_ymm_horizontal":
        t2 = _halves(lambda x, y: _hword_half(x, y, subtract=False, saturate=False), a, b)
        t3 = _halves(lambda x, y: _hword_half(x, y, subtract=True, saturate=False), b, a)
        t4 = _halves(lambda x, y: _hword_half(x, y, subtract=False, saturate=True), t2, t3)
        t5 = _halves(lambda x, y: _hword_half(x, y, subtract=True, saturate=True), t4, a)
        t2b = _halves(lambda x, y: _hdword_half(x, y, subtract=False), t5, t3)
        return _halves(lambda x, y: _hdword_half(x, y, subtract=True), t2b, b)
    raise KeyError(name)


def all_pairs():
    """Construct edge-case 32-byte input pairs with distinct 128-bit halves plus random coverage."""
    pairs = []
    words = (0x0000, 0x0001, 0x3FFF, 0x4000, 0x7FFF, 0x8000, 0x8001, 0xFFFF)
    for shift in range(len(words)):
        a = [words[(i + shift) % len(words)] for i in range(16)]
        b = [words[(3 * i + shift + 1) % len(words)] for i in range(16)]
        pairs.append((struct.pack("<16H", *a), struct.pack("<16H", *b)))

    dwords = (0x00000000, 0x00000001, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF)
    for shift in range(len(dwords)):
        a = [dwords[(i + shift) % len(dwords)] for i in range(8)]
        b = [dwords[(3 * i + shift + 2) % len(dwords)] for i in range(8)]
        pairs.append((struct.pack("<8I", *a), struct.pack("<8I", *b)))

    # Explicit signed-16 saturation vs wraparound & dword overflow witnesses across both 128-bit halves.
    pairs.append((
        struct.pack("<16h", 32767, 32767, -32768, -32768, 32767, -1, -32768, 1,
                    32767, 1, -32768, -1, 16384, 16384, -16384, -16385),
        struct.pack("<16h", -32768, 32767, 32767, -32768, 20000, 20000, -20000, -20000,
                    0, -32768, 32767, 0, -1, -32768, 1, 32767),
    ))
    pairs.append((
        struct.pack("<8I", 0x7FFFFFFF, 0x7FFFFFFF, 0x80000000, 0x00000001,
                    0xFFFFFFFF, 0x00000001, 0x12345678, 0x87654321),
        struct.pack("<8I", 0x80000000, 0x80000000, 0x00000000, 0x80000000,
                    0xCAFEBABE, 0xDEADBEEF, 0x00000001, 0xFFFFFFFF),
    ))
    pairs.append((bytes(range(32)), bytes(255 - i for i in range(32))))
    pairs.append((bytes(32), bytes(32)))
    pairs.append((b"\xff" * 32, b"\xff" * 32))
    pairs.append((b"\x00" * 16 + b"\xff" * 16, b"\xff" * 16 + b"\x00" * 16))

    rng = random.Random(0x6D7A)
    pairs += [(rng.randbytes(32), rng.randbytes(32)) for _ in range(160)]
    return pairs


def bind(path, prefix):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


def host_has_avx2(lifted_lib):
    if hasattr(lifted_lib, "host_has_avx2"):
        lifted_lib.host_has_avx2.argtypes = []
        lifted_lib.host_has_avx2.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_avx2())
    try:
        return " avx2" in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace")
    except OSError:
        return False


def check_translator():
    ret = Instruction.create(Code.RETNQ)
    loads = (load(R.YMM1, R.RCX), load(R.YMM2, R.RDX))
    xmm_loads = (
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RCX)),
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM2, MemoryOperand(R.RDX)),
    )

    accepted = 0
    for name, helper in (
        ("VPHADDW", "op_vphaddw"),
        ("VPHADDD", "op_vphaddd"),
        ("VPHADDSW", "op_vphaddsw"),
        ("VPHSUBW", "op_vphsubw"),
        ("VPHSUBD", "op_vphsubd"),
        ("VPHSUBSW", "op_vphsubsw"),
    ):
        code = getattr(Code, f"VEX_{name}_YMM_YMM_YMMM256")
        source, _ = lift_code(
            name, 0x1000, encode_seq(*loads, Instruction.create_reg_reg_reg(code, R.YMM0, R.YMM1, R.YMM2), ret))
        assert f"wide128({helper}, ymm1, ymm2)" in source, name
        accepted += 1

    # 128-bit VPHADDW keeps its single-helper path (and cpu_vex_aot.py stays untouched).
    src128, _ = lift_code(
        "xmm_vphaddw", 0x1000,
        encode_seq(*xmm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPHADDW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2), ret))
    assert "ymm0 = op_vphaddw(ymm1, ymm2);" in src128 and "wide128" not in src128

    mem_err = "unsupported memory operand for register-domain AVX integer instruction"
    mem = MemoryOperand(R.RDX)
    cases = [
        # m256 memory operands on all 6 new YMM horizontal forms
        ("vphaddw_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPHADDW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        ("vphaddd_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPHADDD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        ("vphaddsw_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPHADDSW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        ("vphsubw_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPHSUBW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        ("vphsubd_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPHSUBD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        ("vphsubsw_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPHSUBSW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        # EVEX
        ("vpaddw_evex", (*loads, Instruction.create_reg_reg_reg(Code.EVEX_VPADDB_YMM_K1Z_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported EVEX instruction"),
        ("vpaddd_evex", (*loads, Instruction.create_reg_reg_reg(Code.EVEX_VPADDD_YMM_K1Z_YMM_YMMM256B32, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported EVEX instruction"),
        # Nonvolatile YMM6..YMM15
        ("vphaddw_nonvol_dst", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPHADDW_YMM_YMM_YMMM256, R.YMM6, R.YMM1, R.YMM2), ret),
         "nonvolatile vector register"),
        ("vphsubd_nonvol_src1", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPHSUBD_YMM_YMM_YMMM256, R.YMM0, R.YMM7, R.YMM2), ret),
         "nonvolatile vector register"),
        ("vphsubsw_nonvol_src2", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPHSUBSW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM15), ret),
         "nonvolatile vector register"),
        # Undefined vector source
        ("vphaddsw_undef_src2", (*loads[:1], Instruction.create_reg_reg_reg(Code.VEX_VPHADDSW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM3), ret),
         "undefined vector register ymm3"),
        ("vphsubw_undef_src1", (*loads[1:], Instruction.create_reg_reg_reg(Code.VEX_VPHSUBW_YMM_YMM_YMMM256, R.YMM0, R.YMM4, R.YMM2), ret),
         "undefined vector register ymm4"),
        # Missing terminal ret
        ("vphaddd_missing_ret", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPHADDD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
         "missing terminal ret"),
    ]
    for label, seq, reason in cases:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted a rejected control")
    return accepted, len(cases)


def check_differential(native, lifted, altered, altered_lifted):
    pairs = all_pairs()
    total = 0
    for name in EXPORTS:
        for index, (a, b) in enumerate(pairs):
            want = reference(name, a, b)
            for label, funcs in (("native", native), ("lifted", lifted)):
                got = invoke(funcs[name], a, b)
                if got != want:
                    raise AssertionError(
                        f"{label} {name} case {index} word {first_bad_word(got, want)}: "
                        f"got {got.hex()} expected {want.hex()}")
            total += 1

    # Unaligned input and output buffers; guards must remain intact.
    a, b = pairs[len(pairs) // 2]
    for name in EXPORTS:
        for a_offset, out_offset in ((1, 0), (0, 3), (7, 5)):
            want = reference(name, a, b)
            for label, funcs in (("native", native), ("lifted", lifted)):
                got = invoke(funcs[name], a, b, a_offset, out_offset)
                if got != want:
                    raise AssertionError(f"{label} {name} offsets {a_offset}/{out_offset}")
            total += 1

    # Altered build checks: swapped ops match altered native and diverge from original.
    altered_names = (
        "op_vphaddw_ymm", "op_vphaddd_ymm", "op_vphaddsw_ymm",
        "op_vphsubw_ymm", "op_vphsubd_ymm", "op_vphsubsw_ymm",
    )
    for name in altered_names:
        changed = 0
        for a, b in pairs:
            alt_nat = invoke(altered[name], a, b)
            alt_lif = invoke(altered_lifted[name], a, b)
            if alt_lif != alt_nat:
                raise AssertionError(f"altered {name}: lifted differs from native")
            changed += alt_nat != invoke(native[name], a, b)
        if not changed:
            raise AssertionError(f"altered build did not diverge on {name}")

    # Mutant Python models: must fail against lifted in the expected 64-bit words.
    zero16 = b"\x00" * 16
    mutants = (
        ("op_vphaddw_ymm", lambda x, y: reference("op_vphaddw_ymm", x, y)[:16] + zero16, {2, 3}),
        ("op_vphaddd_ymm", lambda x, y: reference("op_vphsubd_ymm", x, y), {0, 1, 2, 3}),
        ("op_vphaddsw_ymm", lambda x, y: reference("op_vphaddw_ymm", x, y), {0, 1, 2, 3}),
        ("op_vphsubw_ymm", lambda x, y: reference("op_vphsubw_ymm", x, y)[16:] + reference("op_vphsubw_ymm", x, y)[:16], {0, 1, 2, 3}),
        ("op_vphsubsw_ymm", lambda x, y: reference("op_vphsubw_ymm", x, y), {0, 1, 2, 3}),
        ("op_vphaddsw_ymm_then_xmm_vphsubsw",
         lambda x, y: reference("op_vphaddsw_ymm_then_xmm_vphsubsw", x, y)[:16] + reference("op_vphaddsw_ymm", x, y)[16:],
         {2, 3}),
    )
    for name, wrong, allowed_words in mutants:
        bad_words = set()
        for x, y in pairs:
            w = first_bad_word(invoke(lifted[name], x, y), wrong(x, y))
            if w is not None:
                bad_words.add(w)
        if not bad_words or not bad_words <= allowed_words:
            raise AssertionError(f"mutant on {name}: bad words {sorted(bad_words)}, expected subset of {sorted(allowed_words)}")

    return total, len(mutants), len(altered_names)


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

    original_path, lifted_path, altered_path, altered_lifted_path = (Path(p) for p in sys.argv[1:5])

    accepted, rejected = check_translator()
    print(f"{accepted} 256-bit horizontal forms accepted; {rejected} rejection controls passed")

    lifted_lib, lifted = bind(lifted_path, "lifted_")
    if not host_has_avx2(lifted_lib):
        print("Host lacks CPUID.01H:ECX[28] + XGETBV + CPUID.07H:EBX[5] AVX2; skipping native differential check")
        sys.exit(77)

    _, native = bind(original_path, "")
    _, altered = bind(altered_path, "")
    _, altered_lifted = bind(altered_lifted_path, "lifted_")
    total, caught, altered_count = check_differential(native, lifted, altered, altered_lifted)
    print(f"{total} native/lifted/reference cases across {len(EXPORTS)} exports matched bit for bit; "
          f"{caught} mutant controls + {altered_count} altered-build controls caught")

    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
