"""Differential, boundary, mutation, and rejection checks for 256-bit AVX2 VPMULDQ, VPMULUDQ, VPMULHRSW, and VPSADBW."""

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
    "op_vpmuldq_ymm",
    "op_vpmuludq_ymm",
    "op_vpmulhrsw_ymm",
    "op_vpsadbw_ymm",
    "op_vpmuldq_ymm_dst_src1",
    "op_vpmuludq_ymm_dst_src2",
    "op_vpmulhrsw_ymm_all",
    "op_vpsadbw_ymm_dst_src2",
    "op_vpmuldq_ymm_then_xmm_vpmuludq",
    "op_chain_ymm_mul_sad",
)


# --- Independent Python reference: operates on each 16-byte 128-bit half independently ---

def _muldq_half(a, b, signed_mul):
    fmt = "<4i" if signed_mul else "<4I"
    va, vb = struct.unpack(fmt, a), struct.unpack(fmt, b)
    return struct.pack("<2q" if signed_mul else "<2Q", va[0] * vb[0], va[2] * vb[2])


def _mulhrsw_half(a, b, *, drop_round=False):
    bias = 0 if drop_round else 0x4000
    wa, wb = struct.unpack("<8h", a), struct.unpack("<8h", b)
    out = [(((sa * sb) + bias) >> 15) & 0xFFFF for sa, sb in zip(wa, wb)]
    return struct.pack("<8H", *out)


def _sadbw_half(a, b):
    s0 = sum(abs(x - y) for x, y in zip(a[:8], b[:8]))
    s1 = sum(abs(x - y) for x, y in zip(a[8:16], b[8:16]))
    return struct.pack("<2Q", s0, s1)


def _halves(fn, a, b):
    return fn(a[:16], b[:16]) + fn(a[16:32], b[16:32])


def reference(name, a, b):
    if name in ("op_vpmuldq_ymm", "op_vpmuldq_ymm_dst_src1"):
        return _halves(lambda x, y: _muldq_half(x, y, True), a, b)
    if name in ("op_vpmuludq_ymm", "op_vpmuludq_ymm_dst_src2"):
        return _halves(lambda x, y: _muldq_half(x, y, False), a, b)
    if name == "op_vpmulhrsw_ymm":
        return _halves(_mulhrsw_half, a, b)
    if name == "op_vpmulhrsw_ymm_all":
        return _halves(_mulhrsw_half, a, a)
    if name in ("op_vpsadbw_ymm", "op_vpsadbw_ymm_dst_src2"):
        return _halves(_sadbw_half, a, b)
    if name == "op_vpmuldq_ymm_then_xmm_vpmuludq":
        top = _halves(lambda x, y: _muldq_half(x, y, True), a, b)
        return _muldq_half(top[:16], b[:16], False) + b"\x00" * 16
    if name == "op_chain_ymm_mul_sad":
        t2 = _halves(lambda x, y: _muldq_half(x, y, True), a, b)
        t3 = _halves(lambda x, y: _muldq_half(x, y, False), b, a)
        t4 = _halves(_mulhrsw_half, t2, t3)
        return _halves(_sadbw_half, t4, a)
    raise KeyError(name)


def all_pairs():
    """Construct edge-case 32-byte input pairs with distinct 128-bit halves plus random coverage."""
    pairs = []
    dwords = (0x00000000, 0x00000001, 0x00000002, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF)
    for shift in range(len(dwords)):
        a = [dwords[(i + shift) % len(dwords)] for i in range(8)]
        b = [dwords[(3 * i + shift + 1) % len(dwords)] for i in range(8)]
        pairs.append((struct.pack("<8I", *a), struct.pack("<8I", *b)))

    words = (0x0000, 0x0001, 0x3FFF, 0x4000, 0x7FFF, 0x8000, 0x8001, 0xFFFF)
    for shift in range(len(words)):
        a = [words[(i + shift) % len(words)] for i in range(16)]
        b = [words[(3 * i + shift + 2) % len(words)] for i in range(16)]
        pairs.append((struct.pack("<16H", *a), struct.pack("<16H", *b)))

    # Explicit boundary witnesses:
    # - 0x80000000 * 2: signed (-2^32) vs unsigned (+2^32)
    # - 0x8000 * 0x8000 -> 0x8000 in VPMULHRSW; 1 * 0x4000 -> 1
    # - asymmetric byte differences across 64-bit qwords and 128-bit lanes
    pairs.append((
        struct.pack("<8I", 0x80000000, 0x12345678, 0x80000000, 0xDEADBEEF,
                    0x7FFFFFFF, 0xCAFEBABE, 0xFFFFFFFF, 0x00000000),
        struct.pack("<8I", 2, 0x87654321, 0x80000000, 0x11111111,
                    0x80000000, 0x22222222, 0xFFFFFFFF, 0x33333333),
    ))
    pairs.append((
        struct.pack("<16h", 1, -1, 0x4000, -32768, 32767, 0, -2, 3,
                    -32768, 0x4000, 1, -1, 100, -200, 300, -400),
        struct.pack("<16h", 0x4000, 0x4000, 1, -32768, 32767, 0, 0, -1,
                    -32768, 2, 0x4000, 0x4000, -50, 60, -70, 80),
    ))
    pairs.append((bytes(range(32)), bytes(255 - i for i in range(32))))
    pairs.append((bytes(32), bytes(32)))
    pairs.append((b"\xff" * 32, b"\xff" * 32))
    pairs.append((b"\x00" * 16 + b"\xff" * 16, b"\xff" * 16 + b"\x00" * 16))

    rng = random.Random(0x6D5A)
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
        ("VPMULDQ", "op_vpmuldq"),
        ("VPMULUDQ", "op_vpmuludq"),
        ("VPMULHRSW", "op_vpmulhrsw"),
        ("VPSADBW", "op_vpsadbw"),
    ):
        code = getattr(Code, f"VEX_{name}_YMM_YMM_YMMM256")
        source, _ = lift_code(
            name, 0x1000, encode_seq(*loads, Instruction.create_reg_reg_reg(code, R.YMM0, R.YMM1, R.YMM2), ret))
        assert f"wide128({helper}, ymm1, ymm2)" in source, name
        accepted += 1

    # 128-bit VPMULDQ/VPMULUDQ/VPMULHRSW/VPSADBW keep their single-helper path.
    src128, _ = lift_code(
        "xmm_muldq", 0x1000,
        encode_seq(*xmm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPMULDQ_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2), ret))
    assert "ymm0 = op_vpmuldq(ymm1, ymm2);" in src128 and "wide128" not in src128

    mem_err = "unsupported memory operand for register-domain AVX integer instruction"
    mem = MemoryOperand(R.RDX)
    cases = [
        # m256 memory operands on the 4 new YMM forms
        ("vpmuldq_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPMULDQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        ("vpmuludq_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPMULUDQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        ("vpmulhrsw_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPMULHRSW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        ("vpsadbw_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPSADBW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        # EVEX
        ("vpmuldq_evex", (*loads, Instruction.create_reg_reg_reg(Code.EVEX_VPMULDQ_YMM_K1Z_YMM_YMMM256B64, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported EVEX instruction"),
        ("vpmuludq_evex", (*loads, Instruction.create_reg_reg_reg(Code.EVEX_VPMULUDQ_YMM_K1Z_YMM_YMMM256B64, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported EVEX instruction"),
        ("vpmulhrsw_evex", (*loads, Instruction.create_reg_reg_reg(Code.EVEX_VPMULHRSW_YMM_K1Z_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported EVEX instruction"),
        # Nonvolatile YMM6..YMM15
        ("vpmuldq_nonvol_dst", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPMULDQ_YMM_YMM_YMMM256, R.YMM6, R.YMM1, R.YMM2), ret),
         "nonvolatile vector register"),
        ("vpsadbw_nonvol_src2", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPSADBW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM15), ret),
         "nonvolatile vector register"),
        # Undefined vector source
        ("vpmulhrsw_undef_src", (*loads[:1], Instruction.create_reg_reg_reg(Code.VEX_VPMULHRSW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM3), ret),
         "undefined vector register ymm3"),
        ("vpmuludq_undef_src1", (*loads[1:], Instruction.create_reg_reg_reg(Code.VEX_VPMULUDQ_YMM_YMM_YMMM256, R.YMM0, R.YMM4, R.YMM2), ret),
         "undefined vector register ymm4"),
        # Missing terminal ret
        ("vpsadbw_missing_ret", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPSADBW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
         "missing terminal ret"),
        # Carved-out 128-bit forms (owned by cpu_vex_aot.py)
        ("carved_vpaddw_128", (*xmm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPADDW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2), ret),
         "unsupported instruction"),
        ("carved_vpmulhw_128", (*xmm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPMULHW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2), ret),
         "unsupported instruction"),
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
    for name in ("op_vpmuldq_ymm", "op_vpmuludq_ymm", "op_vpmulhrsw_ymm", "op_vpsadbw_ymm"):
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
        ("op_vpmuldq_ymm", lambda x, y: reference("op_vpmuldq_ymm", x, y)[:16] + zero16, {2, 3}),
        ("op_vpmuludq_ymm", lambda x, y: reference("op_vpmuldq_ymm", x, y), {0, 1, 2, 3}),
        ("op_vpmulhrsw_ymm", lambda x, y: _halves(lambda u, v: _mulhrsw_half(u, v, drop_round=True), x, y), {0, 1, 2, 3}),
        ("op_vpsadbw_ymm", lambda x, y: reference("op_vpsadbw_ymm", x, y)[16:] + reference("op_vpsadbw_ymm", x, y)[:16], {0, 1, 2, 3}),
        ("op_vpmuldq_ymm_then_xmm_vpmuludq", lambda x, y: reference("op_vpmuldq_ymm_then_xmm_vpmuludq", x, y)[:16] + reference("op_vpmuldq_ymm", x, y)[16:], {2, 3}),
    )
    for name, wrong, allowed_words in mutants:
        bad_words = set()
        for x, y in pairs:
            w = first_bad_word(invoke(lifted[name], x, y), wrong(x, y))
            if w is not None:
                bad_words.add(w)
        if not bad_words or not bad_words <= allowed_words:
            raise AssertionError(f"mutant on {name}: bad words {sorted(bad_words)}, expected subset of {sorted(allowed_words)}")

    return total, len(mutants)


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
    print(f"{accepted} 256-bit forms accepted; {rejected} rejection controls passed")

    lifted_lib, lifted = bind(lifted_path, "lifted_")
    if not host_has_avx2(lifted_lib):
        print("Host lacks CPUID.01H:ECX[28] + XGETBV + CPUID.07H:EBX[5] AVX2; skipping native differential check")
        sys.exit(77)

    _, native = bind(original_path, "")
    _, altered = bind(altered_path, "")
    _, altered_lifted = bind(altered_lifted_path, "lifted_")
    total, caught = check_differential(native, lifted, altered, altered_lifted)
    print(f"{total} native/lifted/reference cases across {len(EXPORTS)} exports matched bit for bit; "
          f"{caught} mutant controls + 4 altered-build controls caught")

    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
