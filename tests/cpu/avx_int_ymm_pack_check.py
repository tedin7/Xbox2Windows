"""Differential, boundary, mutation, and rejection checks for 256-bit AVX2 packed-integer pack ops."""

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
    "op_vpacksswb_ymm",
    "op_vpackssdw_ymm",
    "op_vpackusdw_ymm",
    "op_vpacksswb_ymm_dst_src1",
    "op_vpackssdw_ymm_dst_src2",
    "op_vpackusdw_ymm_all",
    "op_vpacksswb_ymm_dst_src2",
    "op_vpackusdw_ymm_then_xmm_vpacksswb",
    "op_chain_ymm_pack",
)


# --- Independent Python reference: operates on each 16-byte 128-bit half independently ---

def _sat_s8(val):
    return max(-128, min(127, val))


def _sat_s16(val):
    return max(-32768, min(32767, val))


def _sat_u16(val):
    return max(0, min(65535, val))


def _pack_half(kind, a, b):
    if kind == "sswb":
        vals = [_sat_s8(v) for v in (*struct.unpack("<8h", a), *struct.unpack("<8h", b))]
        return struct.pack("<16b", *vals)
    if kind == "ssdw":
        vals = [_sat_s16(v) for v in (*struct.unpack("<4i", a), *struct.unpack("<4i", b))]
        return struct.pack("<8h", *vals)
    if kind == "usdw":
        vals = [_sat_u16(v) for v in (*struct.unpack("<4i", a), *struct.unpack("<4i", b))]
        return struct.pack("<8H", *vals)
    raise ValueError(kind)


def _halves(fn, a, b):
    return fn(a[:16], b[:16]) + fn(a[16:32], b[16:32])


def reference(name, a, b):
    if name in ("op_vpacksswb_ymm", "op_vpacksswb_ymm_dst_src1", "op_vpacksswb_ymm_dst_src2"):
        return _halves(lambda x, y: _pack_half("sswb", x, y), a, b)
    if name in ("op_vpackssdw_ymm", "op_vpackssdw_ymm_dst_src2"):
        return _halves(lambda x, y: _pack_half("ssdw", x, y), a, b)
    if name == "op_vpackusdw_ymm":
        return _halves(lambda x, y: _pack_half("usdw", x, y), a, b)
    if name == "op_vpackusdw_ymm_all":
        return _halves(lambda x, y: _pack_half("usdw", x, y), a, a)
    if name == "op_vpackusdw_ymm_then_xmm_vpacksswb":
        top = _halves(lambda x, y: _pack_half("usdw", x, y), a, b)
        return _pack_half("sswb", top[:16], b[:16]) + b"\x00" * 16
    if name == "op_chain_ymm_pack":
        t2 = _halves(lambda x, y: _pack_half("ssdw", x, y), a, b)
        t3 = _halves(lambda x, y: _pack_half("usdw", x, y), b, a)
        return _halves(lambda x, y: _pack_half("sswb", x, y), t2, t3)
    raise KeyError(name)


def all_pairs():
    """Construct exact clamping-boundary 32-byte input pairs across both 128-bit halves plus random."""
    pairs = []
    words = (-32768, -129, -128, -127, -1, 0, 1, 127, 128, 129, 32767)
    for shift in range(len(words)):
        a = [words[(i + shift) % len(words)] for i in range(16)]
        b = [words[(3 * i + shift + 1) % len(words)] for i in range(16)]
        pairs.append((struct.pack("<16h", *a), struct.pack("<16h", *b)))

    dwords = (-2147483648, -32769, -32768, -32767, -1, 0, 1, 32767, 32768, 65535, 65536, 2147483647)
    for shift in range(len(dwords)):
        a = [dwords[(i + shift) % len(dwords)] for i in range(8)]
        b = [dwords[(3 * i + shift + 2) % len(dwords)] for i in range(8)]
        pairs.append((struct.pack("<8i", *a), struct.pack("<8i", *b)))

    # Explicit boundary witnesses spanning low and high 128-bit halves differently.
    pairs.append((
        struct.pack("<16h", -32768, -129, -128, -127, 127, 128, 129, 32767,
                    -1, 0, 1, -200, 200, -128, 127, 30000),
        struct.pack("<16h", 32767, 129, 128, 127, -127, -128, -129, -32768,
                    -500, 500, -1, 0, 1, 128, -129, -30000),
    ))
    pairs.append((
        struct.pack("<8i", -2147483648, -32769, -32768, -1, 0, 32767, 32768, 65536),
        struct.pack("<8i", 2147483647, 65535, 65536, -32768, -1, 1, -32769, 32768),
    ))
    pairs.append((bytes(range(32)), bytes(255 - i for i in range(32))))
    pairs.append((bytes(32), bytes(32)))
    pairs.append((b"\xff" * 32, b"\xff" * 32))
    pairs.append((b"\x00" * 16 + b"\xff" * 16, b"\xff" * 16 + b"\x00" * 16))

    rng = random.Random(0x50AC4)
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
        ("VPACKSSWB", "op_vpacksswb"),
        ("VPACKSSDW", "op_vpackssdw"),
        ("VPACKUSDW", "op_vpackusdw"),
    ):
        code = getattr(Code, f"VEX_{name}_YMM_YMM_YMMM256")
        source, _ = lift_code(
            name, 0x1000, encode_seq(*loads, Instruction.create_reg_reg_reg(code, R.YMM0, R.YMM1, R.YMM2), ret))
        assert f"wide128({helper}, ymm1, ymm2)" in source, name
        accepted += 1

    # 128-bit VPACKSSWB keeps its single-helper path (and cpu_vex_aot.py stays untouched).
    src128, _ = lift_code(
        "xmm_vpacksswb", 0x1000,
        encode_seq(*xmm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPACKSSWB_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2), ret))
    assert "ymm0 = op_vpacksswb(ymm1, ymm2);" in src128 and "wide128" not in src128

    mem_err = "unsupported memory operand for register-domain AVX integer instruction"
    mem = MemoryOperand(R.RDX)
    cases = [
        # m256 memory operands on all 3 new YMM pack forms
        ("vpacksswb_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPACKSSWB_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        ("vpackssdw_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPACKSSDW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        ("vpackusdw_m256", (*loads[:1], Instruction.create_reg_reg_mem(Code.VEX_VPACKUSDW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem), ret), mem_err),
        # EVEX
        ("vpacksswb_evex", (*loads, Instruction.create_reg_reg_reg(Code.EVEX_VPACKSSWB_YMM_K1Z_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported EVEX instruction"),
        ("vpackusdw_evex", (*loads, Instruction.create_reg_reg_reg(Code.EVEX_VPACKUSDW_YMM_K1Z_YMM_YMMM256B32, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported EVEX instruction"),
        # Nonvolatile YMM6..YMM15
        ("vpacksswb_nonvol_dst", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPACKSSWB_YMM_YMM_YMMM256, R.YMM6, R.YMM1, R.YMM2), ret),
         "nonvolatile vector register"),
        ("vpackssdw_nonvol_src1", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPACKSSDW_YMM_YMM_YMMM256, R.YMM0, R.YMM7, R.YMM2), ret),
         "nonvolatile vector register"),
        ("vpackusdw_nonvol_src2", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPACKUSDW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM15), ret),
         "nonvolatile vector register"),
        # Undefined vector source
        ("vpacksswb_undef_src2", (*loads[:1], Instruction.create_reg_reg_reg(Code.VEX_VPACKSSWB_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM3), ret),
         "undefined vector register ymm3"),
        ("vpackssdw_undef_src1", (*loads[1:], Instruction.create_reg_reg_reg(Code.VEX_VPACKSSDW_YMM_YMM_YMMM256, R.YMM0, R.YMM4, R.YMM2), ret),
         "undefined vector register ymm4"),
        # 128-bit & 256-bit VPACKUSWB stay rejected (128-bit carved out to cpu_vex_aot.py)
        ("vpackuswb_xmm", (*xmm_loads, Instruction.create_reg_reg_reg(Code.VEX_VPACKUSWB_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2), ret),
         "unsupported instruction"),
        ("vpackuswb_ymm", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPACKUSWB_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret),
         "unsupported instruction"),
        # Missing terminal ret
        ("vpackusdw_missing_ret", (*loads, Instruction.create_reg_reg_reg(Code.VEX_VPACKUSDW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
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
    altered_names = ("op_vpacksswb_ymm", "op_vpackssdw_ymm", "op_vpackusdw_ymm")
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
        ("op_vpacksswb_ymm", lambda x, y: reference("op_vpacksswb_ymm", x, y)[:16] + zero16, {2, 3}),
        ("op_vpacksswb_ymm",
         lambda x, y: _pack_half("sswb", x[:16], x[16:]) + _pack_half("sswb", y[:16], y[16:]), {1, 2}),
        ("op_vpackssdw_ymm", lambda x, y: reference("op_vpackusdw_ymm", x, y), {0, 1, 2, 3}),
        ("op_vpackusdw_ymm", lambda x, y: reference("op_vpackssdw_ymm", x, y), {0, 1, 2, 3}),
        ("op_vpacksswb_ymm",
         lambda x, y: _halves(lambda a16, b16: struct.pack("<16b", *(
             ((v + 128) & 0xFF) - 128 for v in (*struct.unpack("<8h", a16), *struct.unpack("<8h", b16)))), x, y),
         {0, 1, 2, 3}),
        ("op_vpackusdw_ymm_then_xmm_vpacksswb",
         lambda x, y: reference("op_vpackusdw_ymm_then_xmm_vpacksswb", x, y)[:16] + reference("op_vpackusdw_ymm", x, y)[16:],
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
    print(f"{accepted} 256-bit pack forms accepted; {rejected} rejection controls passed")

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
