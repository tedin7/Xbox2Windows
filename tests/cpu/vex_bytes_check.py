"""Compare VEX.128 byte-domain leaf functions (vpandn, vpavgb, vpmaddubsw, vpshufb) with their recompiled C++."""

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
    "pandn_reg",
    "pandn_mem",
    "pavgb_reg",
    "pavgb_mem",
    "pmaddubsw_reg",
    "pmaddubsw_mem",
    "pshufb_reg",
    "pshufb_mem",
    "bytes_mem_r8_chain",
    "bytes_aligned",
)
UNALIGNED = tuple(name for name in FUNCTIONS if name != "bytes_aligned")

# Arena: guard | A | guard | B | guard | OUT | guard, each 16 bytes, starting 16-byte aligned.
ZONES = {"a": 16, "b": 48, "out": 80}
ARENA = 112
OUT = slice(ZONES["out"], ZONES["out"] + 16)
ALIAS_PAIRS = 256
BYTE_EDGES = (0x00, 0x01, 0x03, 0x0F, 0x13, 0x7E, 0x7F, 0x80, 0x81, 0x8F, 0xFE, 0xFF)
PAIRS = 1024


def inputs():
    """1024 pairs of 16-byte inputs covering edge pairings, saturation/shuffle boundaries, and random bytes."""
    pairs = []
    for a, b in product(BYTE_EDGES, repeat=2):
        pairs.append((bytes([a] * 16), bytes([b] * 16)))
    for shift in range(len(BYTE_EDGES)):
        row = [BYTE_EDGES[(i + shift) % len(BYTE_EDGES)] for i in range(16)]
        col = [BYTE_EDGES[(3 * i + shift + 1) % len(BYTE_EDGES)] for i in range(16)]
        pairs.append((bytes(row), bytes(col)))
    # Explicit witness pairs: signed 16-bit saturation boundaries, pavgb carry, pshufb bit-7 zeroing and bits 4..6 ignored.
    pairs.append((
        bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x80, 0x80, 0x00, 0x01, 0xFE, 0xFF, 0x10, 0x20, 0x30, 0x40]),
        bytes([0x80, 0x80, 0x7F, 0x7F, 0x80, 0x00, 0x7F, 0x7F, 0x00, 0x01, 0xFF, 0xFF, 0x80, 0x8F, 0xFF, 0x13]),
    ))
    generator = random.Random(0xB7E5)
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


def _s8(byte):
    return byte - 256 if byte & 0x80 else byte


def _sat_s16(val):
    if val > 32767:
        return 32767
    if val < -32768:
        return -32768
    return val


def op_pandn(a, b, *, swap=False):
    if swap:
        a, b = b, a
    return bytes(((~x) & y) & 0xFF for x, y in zip(a, b))


def op_pavgb(a, b, *, truncate=False):
    bias = 0 if truncate else 1
    return bytes((x + y + bias) >> 1 for x, y in zip(a, b))


def op_pmaddubsw(a, b, *, wrap=False, signed_a=False):
    out = []
    for i in range(8):
        a0 = _s8(a[2 * i]) if signed_a else a[2 * i]
        a1 = _s8(a[2 * i + 1]) if signed_a else a[2 * i + 1]
        val = a0 * _s8(b[2 * i]) + a1 * _s8(b[2 * i + 1])
        out.append((val if wrap else _sat_s16(val)) & 0xFFFF)
    return struct.pack("<8H", *out)


def op_pshufb(a, b, *, ignore_bit7=False, mask7=False):
    idx_mask = 0x07 if mask7 else 0x0F
    out = []
    for m in b:
        if not ignore_bit7 and (m & 0x80):
            out.append(0)
        else:
            out.append(a[m & idx_mask])
    return bytes(out)


def reference(name, a, b, *, swap_pandn=False, truncate_pavgb=False, wrap_pmadd=False,
              signed_pmadd=False, ignore_bit7=False, mask7=False):
    """Independent byte/lane Python reference model for every exported fixture."""
    if name in ("pandn_reg", "pandn_mem"):
        return op_pandn(a, b, swap=swap_pandn)
    if name in ("pavgb_reg", "pavgb_mem"):
        return op_pavgb(a, b, truncate=truncate_pavgb)
    if name in ("pmaddubsw_reg", "pmaddubsw_mem"):
        return op_pmaddubsw(a, b, wrap=wrap_pmadd, signed_a=signed_pmadd)
    if name in ("pshufb_reg", "pshufb_mem"):
        return op_pshufb(a, b, ignore_bit7=ignore_bit7, mask7=mask7)
    if name == "bytes_mem_r8_chain":
        r8 = op_pmaddubsw(a, b)
        x3 = op_pandn(r8, a)
        x4 = op_pavgb(x3, r8)
        x5 = op_pshufb(x4, b)
        x0 = op_pandn(x5, a)
        x1 = op_pmaddubsw(x0, r8)
        return op_pshufb(x1, b)
    if name == "bytes_aligned":
        x1 = op_pandn(a, b)
        x2 = op_pavgb(x1, a)
        x3 = op_pmaddubsw(x2, b)
        return op_pshufb(x3, x1)
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
    ("C5 F5 DF C2", "vpandn ymm0,ymm1,ymm2", "VEX.256 form"),
    ("C5 F5 E0 C2", "vpavgb ymm0,ymm1,ymm2", "VEX.256 form"),
    ("C4 E2 75 04 C2", "vpmaddubsw ymm0,ymm1,ymm2", "VEX.256 form"),
    ("C4 E2 75 00 C2", "vpshufb ymm0,ymm1,ymm2", "VEX.256 form"),
    # EVEX
    ("62 F1 7D 09 E0 C2", "vpavgb xmm0{k1},xmm0,xmm2", "non-VEX encoding"),
    ("62 F2 7D 09 04 C2", "vpmaddubsw xmm0{k1},xmm0,xmm2", "non-VEX encoding"),
    ("62 F2 7D 09 00 C2", "vpshufb xmm0{k1},xmm0,xmm2", "non-VEX encoding"),
    # Legacy SSE2 / SSSE3
    ("66 0F DF C1", "pandn xmm0,xmm1", "non-VEX encoding"),
    ("66 0F E0 C1", "pavgb xmm0,xmm1", "non-VEX encoding"),
    ("66 0F 38 04 C1", "pmaddubsw xmm0,xmm1", "non-VEX encoding"),
    ("66 0F 38 00 C1", "pshufb xmm0,xmm1", "non-VEX encoding"),
    # Nonvolatile XMM6..XMM15
    ("C5 F9 DF F1", "vpandn xmm6,xmm0,xmm1", "nonvolatile XMM register"),
    ("C5 C1 E0 C1", "vpavgb xmm0,xmm7,xmm1", "nonvolatile XMM register"),
    ("C4 C2 71 04 C0", "vpmaddubsw xmm0,xmm1,xmm8", "nonvolatile XMM register"),
    ("C4 62 79 00 F9", "vpshufb xmm15,xmm0,xmm1", "nonvolatile XMM register"),
    # Undefined XMM sources
    ("C5 F1 DF C2", "vpandn xmm0,xmm1,xmm2", "undefined XMM source"),
    ("C5 F1 E0 C2", "vpavgb xmm0,xmm1,xmm2", "undefined XMM source"),
    ("C4 E2 71 04 C2", "vpmaddubsw xmm0,xmm1,xmm2", "undefined XMM source"),
    ("C4 E2 71 00 C2", "vpshufb xmm0,xmm1,xmm2", "undefined XMM source"),
    # Displaced, indexed, or unsupported base memory operands
    ("C5 F9 DF 41 10", "vpandn xmm0,xmm0,[rcx+10h]", "memory operand outside the 16-byte buffers"),
    ("C5 F9 E0 04 02", "vpavgb xmm0,xmm0,[rdx+rax]", "memory operand outside the 16-byte buffers"),
    ("C4 C2 79 04 44 48 10", "vpmaddubsw xmm0,xmm0,[r8+rcx*2+10h]", "memory operand outside the 16-byte buffers"),
    ("C4 E2 79 00 00", "vpshufb xmm0,xmm0,[rax]", "memory operand outside the 16-byte buffers"),
    # Non-16-byte memory operand (qword load) and direct non-r8 store
    ("C4 E2 79 18 02", "vbroadcastss xmm0,[rdx]", "unsupported instruction"),
    ("C5 FA 7F 02", "vmovdqu [rdx],xmm0", "store outside the output buffer"),
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

    # Patched PE negative controls on original_image.
    data = original_image.read_bytes()
    pshufb_start = code_offset(original_image, "pshufb_reg")
    pmadd_start = code_offset(original_image, "pmaddubsw_reg")

    def set_vex_l_pshufb(buffer):
        decoder = Decoder(64, bytes(data[pshufb_start:pshufb_start + 64]), ip=0)
        for ins in decoder:
            if ins.mnemonic == Mnemonic.VPSHUFB:
                pos = pshufb_start + ins.ip
                assert buffer[pos] == 0xC4, hex(buffer[pos])
                buffer[pos + 2] |= 0x04
                return
        raise AssertionError("no vpshufb in pshufb_reg")

    def store_to_rcx(buffer):
        decoder = Decoder(64, bytes(data[pshufb_start:pshufb_start + 64]), ip=0)
        for ins in decoder:
            if ins.mnemonic == Mnemonic.VMOVDQU and ins.op0_kind == OpKind.MEMORY:
                pos = pshufb_start + ins.ip
                assert buffer[pos] == 0xC4, hex(buffer[pos])
                buffer[pos + 1] |= 0x20  # clear VEX.B (~REX.B)
                buffer[pos + 4] = (buffer[pos + 4] & ~0x07) | 0x01
                return
        raise AssertionError("no vmovdqu store in pshufb_reg")

    def pmaddubsw_to_unsupported(buffer):
        decoder = Decoder(64, bytes(data[pmadd_start:pmadd_start + 64]), ip=0)
        for ins in decoder:
            if ins.mnemonic == Mnemonic.VPMADDUBSW:
                pos = pmadd_start + ins.ip
                assert buffer[pos] == 0xC4 and buffer[pos + 3] == 0x04
                buffer[pos + 3] = 0x29  # vpmuldq -> unsupported
                return
        raise AssertionError("no vpmaddubsw in pmaddubsw_reg")

    expect_rejected(original_image, set_vex_l_pshufb, "VEX.256 form")
    expect_rejected(original_image, store_to_rcx, "store outside the output buffer")
    expect_rejected(original_image, pmaddubsw_to_unsupported, "unsupported instruction")
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

    # Explicit boundary assertions: VPMADDUBSW saturation, VPAVGB carry/no-overflow, VPANDN asymmetry, VPSHUFB bit-7 zero.
    sat_a = bytes([0xFF] * 8 + [0xFF] * 8)
    sat_b = bytes([0x80] * 8 + [0x7F] * 8)
    sat_want = struct.pack("<8h", -32768, -32768, -32768, -32768, 32767, 32767, 32767, 32767)
    if call(lifted["pmaddubsw_reg"], sat_a, sat_b)[OUT] != sat_want:
        raise AssertionError("vpmaddubsw signed 16-bit saturation mismatch")
    if call(lifted["pavgb_reg"], b"\xff\x00\xfe\x01" * 4, b"\xff\x01\xff\x01" * 4)[OUT] != b"\xff\x01\xff\x01" * 4:
        raise AssertionError("vpavgb rounding/no-overflow mismatch")
    if call(lifted["pandn_reg"], b"\xf0\x0f" * 8, b"\xff\xff" * 8)[OUT] != b"\x0f\xf0" * 8:
        raise AssertionError("vpandn (~a) & b mismatch")
    shuf_data = bytes(range(16))
    shuf_mask = bytes([0x80, 0x8F, 0xFF, 0x13, 0x00, 0x7F, 0x25, 0x0F] * 2)
    shuf_want = bytes([0, 0, 0, 3, 0, 15, 5, 15] * 2)
    if call(lifted["pshufb_reg"], shuf_data, shuf_mask)[OUT] != shuf_want:
        raise AssertionError("vpshufb bit-7 zero / low-nibble select mismatch")

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
        ("pandn_reg", {"swap_pandn": True}, "swap_pandn"),
        ("pavgb_reg", {"truncate_pavgb": True}, "truncate_pavgb"),
        ("pmaddubsw_reg", {"wrap_pmadd": True}, "wrap_pmadd"),
        ("pmaddubsw_reg", {"signed_pmadd": True}, "signed_pmadd"),
        ("pshufb_reg", {"ignore_bit7": True}, "ignore_bit7"),
        ("pshufb_reg", {"mask7": True}, "mask7"),
    )
    for name, kwargs, label in mutants:
        diffs = sum(call(lifted[name], a, b)[OUT] != reference(name, a, b, **kwargs) for a, b in pairs)
        if not diffs:
            raise AssertionError(f"mutant model {label} was not distinguished on {name}")
        print(f"mutant {label} on {name}: distinguished on {diffs}/{len(pairs)} pairs")

    # Altered build differential controls.
    for name in ("pandn_reg", "pavgb_reg", "pmaddubsw_reg", "pshufb_reg"):
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

    for name in ("pandn_mem", "pavgb_mem", "pmaddubsw_mem", "pshufb_mem", "bytes_mem_r8_chain", "bytes_aligned"):
        for a, b in pairs[:256]:
            if call(altered[name], a, b) != call(native[name], a, b):
                raise AssertionError(f"altered build unexpectedly changed {name}")

    record_covered_forms(args.original)


if __name__ == "__main__":
    main()
