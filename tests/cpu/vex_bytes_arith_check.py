"""Compare VEX.128 byte arithmetic/compare leaf functions (vpaddb, vpsubb, vpcmpeqb, vpcmpgtb) with lifted C++."""

import argparse
import ctypes
from itertools import product
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

from iced_x86 import Decoder, Formatter, FormatterSyntax, Mnemonic, OpKind
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
from cpu_vex_aot import lift, translate  # noqa: E402

FUNCTIONS = (
    "paddb_reg",
    "paddb_mem",
    "psubb_reg",
    "psubb_mem",
    "psubb_self",
    "pcmpeqb_reg",
    "pcmpeqb_mem",
    "pcmpgtb_reg",
    "pcmpgtb_mem",
    "bytes_arith_mem_r8_chain",
    "bytes_arith_aligned",
)
UNALIGNED = tuple(name for name in FUNCTIONS if name != "bytes_arith_aligned")

# Arena: guard | A | guard | B | guard | OUT | guard, each 16 bytes, starting 16-byte aligned.
ZONES = {"a": 16, "b": 48, "out": 80}
ARENA = 112
OUT = slice(ZONES["out"], ZONES["out"] + 16)
ALIAS_PAIRS = 256
BYTE_EDGES = (0x00, 0x01, 0x03, 0x0F, 0x13, 0x7E, 0x7F, 0x80, 0x81, 0x8F, 0xFE, 0xFF)
PAIRS = 1024


def inputs():
    """1024 pairs of 16-byte inputs covering edge pairings, wrap/borrow, signed boundaries, and random bytes."""
    pairs = []
    for a, b in product(BYTE_EDGES, repeat=2):
        pairs.append((bytes([a] * 16), bytes([b] * 16)))
    for shift in range(len(BYTE_EDGES)):
        row = [BYTE_EDGES[(i + shift) % len(BYTE_EDGES)] for i in range(16)]
        col = [BYTE_EDGES[(3 * i + shift + 1) % len(BYTE_EDGES)] for i in range(16)]
        pairs.append((bytes(row), bytes(col)))
    # Explicit witness pairs: wrap/borrow boundaries and signed -128..127 comparisons.
    pairs.append((
        bytes([0xFF, 0x7F, 0x00, 0x80, 0x80, 0xFF, 0xFE, 0x7F, 0x00, 0x01, 0x81, 0x7E, 0x10, 0x20, 0x30, 0x40]),
        bytes([0x01, 0x01, 0x01, 0x01, 0x7F, 0xFE, 0xFF, 0x80, 0x00, 0x01, 0x80, 0x7F, 0x10, 0x80, 0x7F, 0x13]),
    ))
    generator = random.Random(0xB7EA71)
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


def op_paddb(a, b, *, saturate=False):
    if saturate:
        return bytes(min(255, x + y) for x, y in zip(a, b))
    return bytes((x + y) & 0xFF for x, y in zip(a, b))


def op_psubb(a, b, *, swap=False, saturate=False):
    if swap:
        a, b = b, a
    if saturate:
        return bytes(max(0, x - y) for x, y in zip(a, b))
    return bytes((x - y) & 0xFF for x, y in zip(a, b))


def op_pcmpeqb(a, b, *, eq_one=False):
    true_byte = 0x01 if eq_one else 0xFF
    return bytes(true_byte if x == y else 0x00 for x, y in zip(a, b))


def op_pcmpgtb(a, b, *, unsigned=False, swap=False):
    if swap:
        a, b = b, a
    if unsigned:
        return bytes(0xFF if x > y else 0x00 for x, y in zip(a, b))
    return bytes(0xFF if _s8(x) > _s8(y) else 0x00 for x, y in zip(a, b))


def reference(name, a, b, *, saturate_add=False, swap_sub=False, saturate_sub=False,
              eq_one=False, unsigned_gt=False, swap_gt=False):
    """Independent byte/lane Python reference model for every exported fixture."""
    if name in ("paddb_reg", "paddb_mem"):
        return op_paddb(a, b, saturate=saturate_add)
    if name in ("psubb_reg", "psubb_mem"):
        return op_psubb(a, b, swap=swap_sub, saturate=saturate_sub)
    if name == "psubb_self":
        return b"\x00" * 16
    if name in ("pcmpeqb_reg", "pcmpeqb_mem"):
        return op_pcmpeqb(a, b, eq_one=eq_one)
    if name in ("pcmpgtb_reg", "pcmpgtb_mem"):
        return op_pcmpgtb(a, b, unsigned=unsigned_gt, swap=swap_gt)
    if name == "bytes_arith_mem_r8_chain":
        r8 = op_paddb(a, b)
        x3 = op_psubb(r8, a)
        x4 = op_pcmpeqb(x3, r8)
        x5 = op_pcmpgtb(x4, b)
        x0 = op_paddb(x5, a)
        x1 = op_psubb(x0, r8)
        return op_pcmpgtb(x1, b)
    if name == "bytes_arith_aligned":
        x1 = op_paddb(a, b)
        x2 = op_psubb(x1, a)
        x3 = op_pcmpeqb(x2, b)
        return op_pcmpgtb(x3, x1)
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
    ("C5 F5 FC C2", "vpaddb ymm0,ymm1,ymm2", "VEX.256 form"),
    ("C5 F5 F8 C2", "vpsubb ymm0,ymm1,ymm2", "VEX.256 form"),
    ("C5 F5 74 C2", "vpcmpeqb ymm0,ymm1,ymm2", "VEX.256 form"),
    ("C5 F5 64 C2", "vpcmpgtb ymm0,ymm1,ymm2", "VEX.256 form"),
    # EVEX
    ("62 F1 75 09 FC C2", "vpaddb xmm0{k1},xmm1,xmm2", "non-VEX encoding"),
    ("62 F1 75 09 F8 C2", "vpsubb xmm0{k1},xmm1,xmm2", "non-VEX encoding"),
    ("62 F1 7D 08 74 C9", "vpcmpeqb k1,xmm0,xmm1", "non-VEX encoding"),
    ("62 F1 7D 08 64 C9", "vpcmpgtb k1,xmm0,xmm1", "non-VEX encoding"),
    # Legacy SSE2
    ("66 0F FC C1", "paddb xmm0,xmm1", "non-VEX encoding"),
    ("66 0F F8 C1", "psubb xmm0,xmm1", "non-VEX encoding"),
    ("66 0F 74 C1", "pcmpeqb xmm0,xmm1", "non-VEX encoding"),
    ("66 0F 64 C1", "pcmpgtb xmm0,xmm1", "non-VEX encoding"),
    # Nonvolatile XMM6..XMM15
    ("C5 F9 FC F1", "vpaddb xmm6,xmm0,xmm1", "nonvolatile XMM register"),
    ("C5 C1 F8 C1", "vpsubb xmm0,xmm7,xmm1", "nonvolatile XMM register"),
    ("C4 C1 71 74 C0", "vpcmpeqb xmm0,xmm1,xmm8", "nonvolatile XMM register"),
    ("C4 61 79 64 F9", "vpcmpgtb xmm15,xmm0,xmm1", "nonvolatile XMM register"),
    # Undefined XMM sources
    ("C5 F1 FC C2", "vpaddb xmm0,xmm1,xmm2", "undefined XMM source"),
    ("C5 F1 F8 C2", "vpsubb xmm0,xmm1,xmm2", "undefined XMM source"),
    ("C5 F1 74 C2", "vpcmpeqb xmm0,xmm1,xmm2", "undefined XMM source"),
    ("C5 F1 64 C2", "vpcmpgtb xmm0,xmm1,xmm2", "undefined XMM source"),
    # Displaced, indexed, or unsupported base memory operands
    ("C5 F9 FC 41 10", "vpaddb xmm0,xmm0,[rcx+10h]", "memory operand outside the 16-byte buffers"),
    ("C5 F9 F8 04 02", "vpsubb xmm0,xmm0,[rdx+rax]", "memory operand outside the 16-byte buffers"),
    ("C4 C1 79 74 44 48 10", "vpcmpeqb xmm0,xmm0,[r8+rcx*2+10h]", "memory operand outside the 16-byte buffers"),
    ("C5 F9 64 00", "vpcmpgtb xmm0,xmm0,[rax]", "memory operand outside the 16-byte buffers"),
    # Store outside [r8]
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
    paddb_start = code_offset(original_image, "paddb_reg")

    def set_vex_l_paddb(buffer):
        decoder = Decoder(64, bytes(data[paddb_start:paddb_start + 64]), ip=0)
        for ins in decoder:
            if ins.mnemonic == Mnemonic.VPADDB:
                pos = paddb_start + ins.ip
                assert buffer[pos] == 0xC5, hex(buffer[pos])
                buffer[pos + 1] |= 0x04
                return
        raise AssertionError("no vpaddb in paddb_reg")

    def store_to_rcx(buffer):
        decoder = Decoder(64, bytes(data[paddb_start:paddb_start + 64]), ip=0)
        for ins in decoder:
            if ins.mnemonic == Mnemonic.VMOVDQU and ins.op0_kind == OpKind.MEMORY:
                pos = paddb_start + ins.ip
                assert buffer[pos] == 0xC4, hex(buffer[pos])
                buffer[pos + 1] |= 0x20  # clear VEX.B (~REX.B)
                buffer[pos + 4] = (buffer[pos + 4] & ~0x07) | 0x01
                return
        raise AssertionError("no vmovdqu store in paddb_reg")

    def paddb_to_unsupported(buffer):
        decoder = Decoder(64, bytes(data[paddb_start:paddb_start + 64]), ip=0)
        for ins in decoder:
            if ins.mnemonic == Mnemonic.VPADDB:
                pos = paddb_start + ins.ip
                assert buffer[pos] == 0xC5 and buffer[pos + 2] == 0xFC
                buffer[pos + 2] = 0xDC  # vpaddusb -> unsupported
                return
        raise AssertionError("no vpaddb in paddb_reg")

    expect_rejected(original_image, set_vex_l_paddb, "VEX.256 form")
    expect_rejected(original_image, store_to_rcx, "store outside the output buffer")
    expect_rejected(original_image, paddb_to_unsupported, "unsupported instruction")
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

    # Explicit boundary assertions: wrap/borrow, signed -128..127 compare, dst==src1==src2 zeroing.
    if call(lifted["paddb_reg"], b"\xFF\x7F\x80\x00" * 4, b"\x01\x01\x80\xFF" * 4)[OUT] != b"\x00\x80\x00\xFF" * 4:
        raise AssertionError("vpaddb wraparound mismatch")
    if call(lifted["psubb_reg"], b"\x00\x80\x7F\x01" * 4, b"\x01\x01\x80\xFF" * 4)[OUT] != b"\xFF\x7F\xFF\x02" * 4:
        raise AssertionError("vpsubb borrow/wraparound mismatch")
    if call(lifted["psubb_self"], b"\xA5\x5A" * 8, b"\xFF" * 16)[OUT] != b"\x00" * 16:
        raise AssertionError("vpsubb self-zeroing mismatch")
    if call(lifted["pcmpeqb_reg"], b"\x00\x7F\x80\xFF" * 4, b"\x00\x80\x80\xFE" * 4)[OUT] != b"\xFF\x00\xFF\x00" * 4:
        raise AssertionError("vpcmpeqb mask mismatch")
    gt_want = b"\x00\xFF\xFF\x00" * 4
    if call(lifted["pcmpgtb_reg"], b"\x80\xFF\x7F\x80" * 4, b"\x7F\xFE\x80\x80" * 4)[OUT] != gt_want:
        raise AssertionError("vpcmpgtb signed compare mismatch")

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
        ("paddb_reg", {"saturate_add": True}, "saturate_add"),
        ("psubb_reg", {"swap_sub": True}, "swap_sub"),
        ("psubb_reg", {"saturate_sub": True}, "saturate_sub"),
        ("pcmpeqb_reg", {"eq_one": True}, "eq_one"),
        ("pcmpgtb_reg", {"unsigned_gt": True}, "unsigned_gt"),
        ("pcmpgtb_reg", {"swap_gt": True}, "swap_gt"),
    )
    for name, kwargs, label in mutants:
        diffs = sum(call(lifted[name], a, b)[OUT] != reference(name, a, b, **kwargs) for a, b in pairs)
        if not diffs:
            raise AssertionError(f"mutant model {label} was not distinguished on {name}")
        print(f"mutant {label} on {name}: distinguished on {diffs}/{len(pairs)} pairs")

    # Altered build differential controls.
    for name in ("paddb_reg", "psubb_reg", "pcmpeqb_reg", "pcmpgtb_reg"):
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

    for name in ("paddb_mem", "psubb_mem", "psubb_self", "pcmpeqb_mem", "pcmpgtb_mem",
                 "bytes_arith_mem_r8_chain", "bytes_arith_aligned"):
        for a, b in pairs[:256]:
            if call(altered[name], a, b) != call(native[name], a, b):
                raise AssertionError(f"altered build unexpectedly changed {name}")

    record_covered_forms(args.original)


if __name__ == "__main__":
    main()
