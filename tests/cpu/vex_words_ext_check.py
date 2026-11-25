"""Compare VEX.128 extended word-form leaf functions (vpabsw, vpsignw, vpmulhrsw, vphaddw) with their recompiled C++."""

import argparse
import ctypes
from itertools import product
import os
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

from iced_x86 import Decoder, Formatter, FormatterSyntax, Mnemonic, OpKind
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
from cpu_vex_aot import lift, translate  # noqa: E402

FUNCTIONS = (
    "pabsw_reg",
    "pabsw_mem",
    "signw_reg",
    "signw_mem",
    "mulhrsw_reg",
    "mulhrsw_mem",
    "phaddw_reg",
    "phaddw_mem",
    "ext_mem_r8_chain",
    "ext_aligned",
)
UNALIGNED = tuple(name for name in FUNCTIONS if name != "ext_aligned")

# Arena: guard | A | guard | B | guard | OUT | guard, each 16 bytes, starting 16-byte aligned.
ZONES = {"a": 16, "b": 48, "out": 80}
ARENA = 112
OUT = slice(ZONES["out"], ZONES["out"] + 16)
ALIAS_PAIRS = 256
EDGES = (
    0x0000, 0x0001, 0x0002, 0x3FFF, 0x4000, 0x4001,
    0x7FFE, 0x7FFF, 0x8000, 0x8001, 0xC000, 0xFFFF,
)
PAIRS = 1024


def inputs():
    """1024 pairs of 16-byte inputs covering edge pairings, rounding boundaries, and random lanes."""
    pairs = []
    for a, b in product(EDGES, repeat=2):
        pairs.append((struct.pack("<8H", *[a] * 8), struct.pack("<8H", *[b] * 8)))
    for shift in range(len(EDGES)):
        row = [EDGES[(i + shift) % len(EDGES)] for i in range(8)]
        col = [EDGES[(3 * i + shift + 1) % len(EDGES)] for i in range(8)]
        pairs.append((struct.pack("<8H", *row), struct.pack("<8H", *col)))
    # Explicit witness row for rounding boundaries, INT16_MIN, and sign classes.
    pairs.append((
        struct.pack("<8h", 1, -1, 0x4000, -32768, 32767, 0, -2, 3),
        struct.pack("<8h", 0x4000, 0x4000, 1, -32768, 32767, 0, 0, -1),
    ))
    generator = random.Random(0xE716)
    while len(pairs) < PAIRS:
        pairs.append((generator.randbytes(16), generator.randbytes(16)))
    return pairs


def call(function, a, b, output="out", misalign=0, same_inputs=False):
    """Run function(A, B, output) inside a guarded 16-byte-aligned (+ misalign) arena."""
    raw = (ctypes.c_uint8 * (ARENA + 32))()
    base = (ctypes.addressof(raw) + 15) // 16 * 16 + misalign
    ctypes.memset(base, 0xCC, ARENA)
    ctypes.memmove(base + ZONES["a"], a, 16)
    ctypes.memmove(base + ZONES["b"], b, 16)
    second = ZONES["a"] if same_inputs else ZONES["b"]
    function(base + ZONES["a"], base + second, base + ZONES[output])
    return ctypes.string_at(base, ARENA)


def outside_output(arena, output="out"):
    start = ZONES[output]
    return arena[:start] + arena[start + 16:]


def op_pabsw(a, *, sat_abs=False):
    lanes = []
    for w in struct.unpack("<8h", a):
        val = abs(w)
        if sat_abs and val > 32767:
            val = 32767
        lanes.append(val & 0xFFFF)
    return struct.pack("<8H", *lanes)


def op_signw(a, b, *, sign_zero_identity=False):
    lanes = []
    for sa, sb in zip(struct.unpack("<8h", a), struct.unpack("<8h", b)):
        if sb < 0:
            lanes.append((-sa) & 0xFFFF)
        elif sb == 0:
            lanes.append((sa if sign_zero_identity else 0) & 0xFFFF)
        else:
            lanes.append(sa & 0xFFFF)
    return struct.pack("<8H", *lanes)


def op_mulhrsw(a, b, *, drop_round=False):
    bias = 0 if drop_round else 0x4000
    lanes = [
        (((sa * sb) + bias) >> 15) & 0xFFFF
        for sa, sb in zip(struct.unpack("<8h", a), struct.unpack("<8h", b))
    ]
    return struct.pack("<8H", *lanes)


def op_phaddw(a, b, *, swap_hadd=False):
    if swap_hadd:
        a, b = b, a
    wa, wb = struct.unpack("<8h", a), struct.unpack("<8h", b)
    low = [(wa[2 * i] + wa[2 * i + 1]) & 0xFFFF for i in range(4)]
    high = [(wb[2 * i] + wb[2 * i + 1]) & 0xFFFF for i in range(4)]
    return struct.pack("<8H", *(low + high))


def reference(name, a, b, *, drop_round=False, swap_hadd=False, sign_zero_identity=False, sat_abs=False):
    """Independent per-lane Python reference model for every exported fixture."""
    if name in ("pabsw_reg", "pabsw_mem"):
        return op_pabsw(a, sat_abs=sat_abs)
    if name in ("signw_reg", "signw_mem"):
        return op_signw(a, b, sign_zero_identity=sign_zero_identity)
    if name in ("mulhrsw_reg", "mulhrsw_mem"):
        return op_mulhrsw(a, b, drop_round=drop_round)
    if name in ("phaddw_reg", "phaddw_mem"):
        return op_phaddw(a, b, swap_hadd=swap_hadd)
    if name == "ext_mem_r8_chain":
        r8 = op_mulhrsw(a, b)
        x3 = op_pabsw(r8)
        x4 = op_signw(x3, a)
        x5 = op_phaddw(x4, r8)
        x0 = op_mulhrsw(x5, a)
        x1 = op_phaddw(x0, b)
        return op_signw(x1, r8)
    if name == "ext_aligned":
        x1 = op_pabsw(b)
        x2 = op_mulhrsw(a, x1)
        x3 = op_signw(x2, a)
        return op_phaddw(x3, x1)
    raise ValueError(name)


def bind(path):
    library = ctypes.CDLL(str(path))
    functions = {}
    for name in FUNCTIONS:
        fn = getattr(library, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        fn.restype = None
        functions[name] = fn
    return library, functions


def code_offset(image, name):
    with pefile.PE(str(image), fast_load=True) as pe:
        return pe.get_offset_from_rva(export_rva(pe, name))


def expect_rejected(image, patch, message):
    with TemporaryDirectory() as directory:
        copy = Path(directory) / image.name
        data = bytearray(image.read_bytes())
        patch(data)
        copy.write_bytes(data)
        try:
            translate(copy, FUNCTIONS)
        except ValueError as error:
            if message not in str(error):
                raise AssertionError(f"expected {message!r}, got {error}") from None
            return
        raise AssertionError(f"translated a copy that should fail with {message!r}")


# Excluded encodings checked via Decoder + Formatter before calling lift().
REJECTED = (
    # VEX.256 (L=1)
    ("C4 E2 7D 1D C1", "vpabsw ymm0,ymm1", "VEX.256 form"),
    ("C4 E2 75 09 C2", "vpsignw ymm0,ymm1,ymm2", "VEX.256 form"),
    ("C4 E2 75 0B C2", "vpmulhrsw ymm0,ymm1,ymm2", "VEX.256 form"),
    ("C4 E2 75 01 C2", "vphaddw ymm0,ymm1,ymm2", "VEX.256 form"),
    # EVEX
    ("62 F2 7D 09 1D C1", "vpabsw xmm0{k1},xmm1", "non-VEX encoding"),
    ("62 F2 75 09 0B C2", "vpmulhrsw xmm0{k1},xmm1,xmm2", "non-VEX encoding"),
    # Legacy SSSE3
    ("66 0F 38 1D C1", "pabsw xmm0,xmm1", "non-VEX encoding"),
    ("66 0F 38 09 C1", "psignw xmm0,xmm1", "non-VEX encoding"),
    ("66 0F 38 0B C1", "pmulhrsw xmm0,xmm1", "non-VEX encoding"),
    ("66 0F 38 01 C1", "phaddw xmm0,xmm1", "non-VEX encoding"),
    # Nonvolatile XMM6..XMM15
    ("C4 E2 79 1D F0", "vpabsw xmm6,xmm0", "nonvolatile XMM register"),
    ("C4 E2 79 1D C6", "vpabsw xmm0,xmm6", "nonvolatile XMM register"),
    ("C4 E2 79 09 F9", "vpsignw xmm7,xmm0,xmm1", "nonvolatile XMM register"),
    ("C4 C2 79 09 C0", "vpsignw xmm0,xmm0,xmm8", "nonvolatile XMM register"),
    ("C4 E2 31 0B C1", "vpmulhrsw xmm0,xmm9,xmm1", "nonvolatile XMM register"),
    ("C4 62 79 01 F9", "vphaddw xmm15,xmm0,xmm1", "nonvolatile XMM register"),
    # Undefined XMM sources
    ("C4 E2 79 1D C1", "vpabsw xmm0,xmm1", "undefined XMM source"),
    ("C4 E2 79 1D C0", "vpabsw xmm0,xmm0", "undefined XMM source"),
    ("C4 E2 71 09 C2", "vpsignw xmm0,xmm1,xmm2", "undefined XMM source"),
    ("C4 E2 71 0B C2", "vpmulhrsw xmm0,xmm1,xmm2", "undefined XMM source"),
    ("C4 E2 71 01 C2", "vphaddw xmm0,xmm1,xmm2", "undefined XMM source"),
    # Displaced, indexed, or unsupported base memory operands
    ("C4 E2 79 1D 41 10", "vpabsw xmm0,[rcx+10h]", "memory operand outside the 16-byte buffers"),
    ("C4 E2 79 09 04 02", "vpsignw xmm0,xmm0,[rdx+rax]", "memory operand outside the 16-byte buffers"),
    ("C4 C2 79 0B 44 48 10", "vpmulhrsw xmm0,xmm0,[r8+rcx*2+10h]", "memory operand outside the 16-byte buffers"),
    ("C4 E2 79 01 00", "vphaddw xmm0,xmm0,[rax]", "memory operand outside the 16-byte buffers"),
    # Direct / non-r8 / displaced stores
    ("C5 FA 7F 02", "vmovdqu [rdx],xmm0", "store outside the output buffer"),
    ("C5 F9 7F 01", "vmovdqa [rcx],xmm0", "store outside the output buffer"),
    ("C4 C1 7A 7F 40 10", "vmovdqu [r8+10h],xmm0", "memory operand outside the 16-byte buffers"),
    ("C4 C3 79 15 00 00", "vpextrw [r8],xmm0,0", "unsupported instruction"),
)


def check_aot_lowering(original_image):
    """Validate AOT translation and negative rejection cases regardless of host AVX support."""
    source_cpp = translate(original_image, FUNCTIONS)
    if "host_has_avx" not in source_cpp:
        raise AssertionError("translated C++ missing host_has_avx")

    formatter = Formatter(FormatterSyntax.INTEL)
    for code, text, reason in REJECTED:
        body = bytes.fromhex(code) + b"\xc3"
        decoded = formatter.format(next(iter(Decoder(64, body))))
        if decoded != text:
            raise AssertionError(f"test encoding {code} decodes as {decoded!r}, expected {text!r}")
        try:
            lift(0x1000, body)
        except ValueError as error:
            if reason not in str(error):
                raise AssertionError(f"{text}: expected {reason!r}, got {error}") from None
            continue
        raise AssertionError(f"lifted {text}, expected {reason!r}")

    # Multi-instruction defined-register checks: vpabsw defines its destination; partial-def fails.
    defined_chain = bytes.fromhex("C5 FA 6F 01 C4 E2 79 1D C8 C4 E2 79 1D D1") + b"\xc3"
    lifted_lines = lift(0x1000, defined_chain)
    if len(lifted_lines) != 3:
        raise AssertionError(f"expected 3 lifted lines for vpabsw chain, got {lifted_lines}")
    partial_def = bytes.fromhex("C5 FA 6F 01 C4 E2 79 09 C1") + b"\xc3"  # vmovdqu xmm0,[rcx]; vpsignw xmm0,xmm0,xmm1
    try:
        lift(0x1000, partial_def)
    except ValueError as error:
        if "undefined XMM source" not in str(error):
            raise AssertionError(f"expected undefined XMM source, got {error}") from None
    else:
        raise AssertionError("lifted vpsignw with undefined xmm1")

    # Patched PE negative controls on original_image.
    data = original_image.read_bytes()
    pabsw_start = code_offset(original_image, "pabsw_reg")
    mul_start = code_offset(original_image, "mulhrsw_reg")

    def set_vex_l_pabsw(buffer):
        decoder = Decoder(64, bytes(data[pabsw_start:pabsw_start + 64]), ip=0)
        for ins in decoder:
            if ins.mnemonic == Mnemonic.VPABSW:
                pos = pabsw_start + ins.ip
                assert buffer[pos] == 0xC4, hex(buffer[pos])
                buffer[pos + 2] |= 0x04
                return
        raise AssertionError("no vpabsw in pabsw_reg")

    def store_to_rcx(buffer):
        decoder = Decoder(64, bytes(data[pabsw_start:pabsw_start + 64]), ip=0)
        for ins in decoder:
            if ins.mnemonic == Mnemonic.VMOVDQU and ins.op0_kind == OpKind.MEMORY:
                pos = pabsw_start + ins.ip
                # C4 C1 7A 7F 10 (vmovdqu [r8], xmm2) -> clear VEX.B (bit 5 of byte 1) and change modrm rm=001 ([rcx]).
                assert buffer[pos] == 0xC4, hex(buffer[pos])
                buffer[pos + 1] |= 0x20  # VEX.B is inverted: 1 clears REX.B
                buffer[pos + 4] = (buffer[pos + 4] & ~0x07) | 0x01
                return
        raise AssertionError("no vmovdqu store in pabsw_reg")

    def mulhrsw_to_unsupported(buffer):
        decoder = Decoder(64, bytes(data[mul_start:mul_start + 64]), ip=0)
        for ins in decoder:
            if ins.mnemonic == Mnemonic.VPMULHRSW:
                pos = mul_start + ins.ip
                assert buffer[pos] == 0xC4 and buffer[pos + 3] == 0x0B
                buffer[pos + 3] = 0x29  # vpmuldq -> unsupported
                return
        raise AssertionError("no vpmulhrsw in mulhrsw_reg")

    expect_rejected(original_image, set_vex_l_pabsw, "VEX.256 form")
    expect_rejected(original_image, store_to_rcx, "store outside the output buffer")
    expect_rejected(original_image, mulhrsw_to_unsupported, "unsupported instruction")
    print(f"AOT lowering validated ({len(REJECTED)} rejected encodings + 3 patched PE rejections)")


def record_covered_forms(original_path):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    formatter = Formatter(FormatterSyntax.INTEL)
    forms = set()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        for name in FUNCTIONS:
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

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "original_lifted", "altered", "altered_lifted"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()

    check_aot_lowering(args.original)

    lifted_lib, lifted = bind(args.original_lifted)
    lifted_lib.host_has_avx.argtypes = []
    lifted_lib.host_has_avx.restype = ctypes.c_uint32
    if not lifted_lib.host_has_avx():
        print("Host lacks CPUID.01H:ECX[28] or OS XGETBV XMM/YMM; skipping native differential check")
        sys.exit(77)

    pairs = inputs()
    _, native = bind(args.original)
    _, altered = bind(args.altered)
    _, altered_lifted = bind(args.altered_lifted)

    for name in FUNCTIONS:
        for index, (a, b) in enumerate(pairs):
            expected = call(native[name], a, b)
            pristine = call(lambda *args: None, a, b)
            if outside_output(expected) != outside_output(pristine):
                raise AssertionError(f"{name}: native wrote outside its 16-byte output")
            observed = call(lifted[name], a, b)
            if observed != expected:
                raise AssertionError(f"{name} pair {index}: lifted {observed[OUT].hex()} "
                                     f"!= native {expected[OUT].hex()} for {a.hex()} {b.hex()}")
            want = reference(name, a, b)
            if observed[OUT] != want:
                raise AssertionError(f"{name} pair {index}: lifted {observed[OUT].hex()} "
                                     f"!= reference {want.hex()} for {a.hex()} {b.hex()}")
        print(f"{name}: {len(pairs)} pairs match native and reference bit for bit, guards intact")

    # Explicit edge-case assertions.
    min16 = struct.pack("<8H", *([0x8000] * 8))
    if call(lifted["pabsw_reg"], min16, min16)[OUT] != min16:
        raise AssertionError("vpabsw(0x8000) must wrap to 0x8000")
    sign_masks = struct.pack("<8h", -1, 0, 1, -32768, 32767, 0, -2, 2)
    sign_expected = struct.pack("<8H", 0x8000, 0x0000, 0x8000, 0x8000, 0x8000, 0x0000, 0x8000, 0x8000)
    if call(lifted["signw_reg"], min16, sign_masks)[OUT] != sign_expected:
        raise AssertionError("vpsignw(0x8000, mask) failed edge check")
    if call(lifted["mulhrsw_reg"], min16, min16)[OUT] != min16:
        raise AssertionError("vpmulhrsw(0x8000, 0x8000) must yield 0x8000")

    # Exact aliases, RCX == RDX, and 8-byte misalignment on vmovdqu-only exports.
    for name in FUNCTIONS:
        for output in ("a", "b"):
            for a, b in pairs[:ALIAS_PAIRS]:
                if call(lifted[name], a, b, output) != call(native[name], a, b, output):
                    raise AssertionError(f"{name}: lifted differs from native with output aliasing {output}")
        for output in ("out", "a"):
            for a, b in pairs[:ALIAS_PAIRS]:
                if call(lifted[name], a, b, output, same_inputs=True) != call(native[name], a, b, output, same_inputs=True):
                    raise AssertionError(f"{name}: lifted differs from native with RCX == RDX (output {output})")
    for name in UNALIGNED:
        for a, b in pairs[:ALIAS_PAIRS]:
            if call(lifted[name], a, b, misalign=8) != call(native[name], a, b, misalign=8):
                raise AssertionError(f"{name}: lifted differs from native at 8-byte misalignment")
    print("aliased outputs, RCX == RDX, and unaligned buffers match native")

    # Mutation negative controls: Python faulty models must be distinguished from lifted outputs.
    mutants = (
        ("pabsw_reg", {"sat_abs": True}, "sat_abs"),
        ("signw_reg", {"sign_zero_identity": True}, "sign_zero_identity"),
        ("mulhrsw_reg", {"drop_round": True}, "drop_round"),
        ("phaddw_reg", {"swap_hadd": True}, "swap_hadd"),
    )
    for name, kwargs, label in mutants:
        diffs = sum(call(lifted[name], a, b)[OUT] != reference(name, a, b, **kwargs) for a, b in pairs)
        if not diffs:
            raise AssertionError(f"mutant model {label} was not distinguished on {name}")
        print(f"mutant {label} on {name}: distinguished on {diffs}/{len(pairs)} pairs")

    # Altered build differential controls (drop rounding in mulhrsw_reg, swap halves in phaddw_reg,
    # zero-mask bug in signw_reg). Altered native and altered lifted must match each other and differ from original.
    for name in ("signw_reg", "mulhrsw_reg", "phaddw_reg"):
        changed = 0
        for a, b in pairs:
            alt_nat = call(altered[name], a, b)
            alt_lif = call(altered_lifted[name], a, b)
            if alt_lif != alt_nat:
                raise AssertionError(f"altered {name}: lifted differs from altered native")
            changed += alt_nat != call(native[name], a, b)
        if not changed:
            raise AssertionError(f"altered build did not change {name}")
        print(f"altered {name}: altered lifted matches altered native and diverges from original on {changed}/{len(pairs)} pairs")

    for name in ("pabsw_reg", "pabsw_mem", "signw_mem", "mulhrsw_mem", "phaddw_mem", "ext_mem_r8_chain", "ext_aligned"):
        for a, b in pairs[:256]:
            if call(altered[name], a, b) != call(native[name], a, b):
                raise AssertionError(f"altered build unexpectedly changed {name}")

    record_covered_forms(args.original)


if __name__ == "__main__":
    main()
