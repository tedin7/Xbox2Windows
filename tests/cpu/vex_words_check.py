"""Compare VEX.128 word-form leaf functions with their recompiled C++ over edge and random lanes."""

import argparse
import ctypes
from itertools import product
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

from iced_x86 import Decoder, Formatter, FormatterSyntax, Mnemonic, OpKind
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva  # noqa: E402
from cpu_vex_aot import lift, translate  # noqa: E402

FUNCTIONS = ("widen_mix", "word_ops", "shift_edges", "store_reload", "aligned_ops", "byte_pack_reload",
             "sat_words", "mul_minmax_words", "cmp_words")
UNALIGNED = ("widen_mix", "word_ops", "shift_edges", "store_reload", "byte_pack_reload",
             "sat_words", "mul_minmax_words", "cmp_words")  # vmovdqu-only functions
# Arena: guard | A | guard | B | guard | OUT | guard, each 16 bytes, starting 16-byte aligned.
ZONES = {"a": 16, "b": 48, "out": 80}
ARENA = 112
ALIAS_PAIRS = 512
EDGES = (0x7FFF, 0x8000, 0x0000, 0xFFFF, 0x00FF, 0xFF00, 0x0001, 0x8001)
PAIRS = 4096
GUARD = 16


def inputs():
    """4096 pairs of 16-byte inputs: every pairing of edge words in every lane, then random lanes."""
    pairs = []
    for a, b in product(EDGES, repeat=2):
        pairs.append((struct.pack("<8H", *[a] * 8), struct.pack("<8H", *[b] * 8)))
    # Edge words rotated across lanes, so each lane sees each pairing.
    for shift in range(8):
        row = [EDGES[(i + shift) % len(EDGES)] for i in range(8)]
        col = [EDGES[(3 * i + shift) % len(EDGES)] for i in range(8)]
        pairs.append((struct.pack("<8H", *row), struct.pack("<8H", *col)))
    generator = random.Random(0x5EED)
    while len(pairs) < PAIRS:
        pairs.append((generator.randbytes(16), generator.randbytes(16)))
    return pairs


def call(function, a, b, output="out", misalign=0, same_inputs=False):
    """Run function(A, B, output) inside one arena and return the whole arena.

    The arena starts 16-byte aligned plus `misalign`, so every zone shares that alignment. The
    output is its own zone, or aliases A or B exactly. Guard zones are filled with 0xCC, so any
    write outside the output, or any difference in read/write order under aliasing, shows up in
    a byte-for-byte comparison of the arena.
    """
    raw = (ctypes.c_uint8 * (ARENA + 32))()
    base = (ctypes.addressof(raw) + 15) // 16 * 16 + misalign
    ctypes.memset(base, 0xCC, ARENA)
    ctypes.memmove(base + ZONES["a"], a, 16)
    ctypes.memmove(base + ZONES["b"], b, 16)
    second = ZONES["a"] if same_inputs else ZONES["b"]
    function(base + ZONES["a"], base + second, base + ZONES[output])
    return ctypes.string_at(base, ARENA)


def outside_output(arena, output="out"):
    """The arena bytes that the function must not change: everything except the output zone."""
    start = ZONES[output]
    return arena[:start] + arena[start + 16:]


def reference(name, a, b, altered=False):
    """Expected output of the saturating, multiply, min/max and compare functions, from the
    instruction definitions alone: exact integer arithmetic per lane, no code shared with the
    translator. With altered=True, the high product is the signed one, as in the altered build."""
    signed_a, signed_b = struct.unpack("<8h", a), struct.unpack("<8h", b)
    unsigned_a, unsigned_b = struct.unpack("<8H", a), struct.unpack("<8H", b)
    lanes = []
    for sa, sb, ua, ub in zip(signed_a, signed_b, unsigned_a, unsigned_b):
        if name == "sat_words":
            adds = max(-32768, min(32767, sa + sb))
            subs = max(-32768, min(32767, sa - sb))
            lane = (adds ^ (min(65535, ua + ub) + subs) ^ max(0, ua - ub)) & 0xFFFF
        elif name == "mul_minmax_words":
            high = (sa * sb) >> 16 if altered else (ua * ub) >> 16
            lane = (((sa * sb + high) & 0xFFFF) ^ max(ua, ub)) - min(ua, ub)
            lane &= 0xFFFF
        elif name == "cmp_words":
            lane = 1 if sa == sb else 3 if sa > sb else 7
        else:
            raise ValueError(name)
        lanes.append(lane)
    return struct.pack("<8H", *lanes)


REFERENCED = ("sat_words", "mul_minmax_words", "cmp_words")
OUT = slice(ZONES["out"], ZONES["out"] + 16)


def bind(path):
    library = ctypes.CDLL(str(path))
    functions = {}
    for name in FUNCTIONS:
        function = getattr(library, name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        function.restype = None
        functions[name] = function
    return library, functions


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


def code_offsets(image, name):
    with pefile.PE(str(image), fast_load=True) as pe:
        return pe.get_offset_from_rva(export_rva(pe, name))


# Hand-encoded leaf bodies that the profile excludes, each with the disassembly iced must report
# (checked first, so a wrong encoding fails the test instead of passing vacuously) and the reason.
REJECTED = (
    ("C5 F1 FD C2", "vpaddw xmm0,xmm1,xmm2", "undefined XMM source"),
    ("62 F1 75 09 FD C2", "vpaddw xmm0{k1},xmm1,xmm2", "non-VEX encoding"),
    ("66 0F FD C1", "paddw xmm0,xmm1", "non-VEX encoding"),
    ("C5 F9 FD F1", "vpaddw xmm6,xmm0,xmm1", "nonvolatile XMM register"),
    ("C5 F9 FD C6", "vpaddw xmm0,xmm0,xmm6", "nonvolatile XMM register"),
    ("C5 FA 6F 41 10", "vmovdqu xmm0,[rcx+10h]", "memory operand outside the 16-byte buffers"),
    ("C5 FA 6F 00", "vmovdqu xmm0,[rax]", "memory operand outside the 16-byte buffers"),
    ("C5 FA 7F 01", "vmovdqu [rcx],xmm0", "store outside the output buffer"),
    ("C5 F1 E1 C2", "vpsraw xmm0,xmm1,xmm2", "unsupported instruction"),
)


def check_rejected_encodings():
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
    print(f"rejected {len(REJECTED)} excluded encodings, including an EVEX masked vpaddw")


def check_byte_pack_reload(functions):
    a = bytes((5, 7, 9, 0, 0, 0, 0, 0, 254, 6)) + bytes(6)
    b = bytes((0, 1, 0, 0, 0, 0, 0, 0, 255)) + bytes(7)
    out = slice(ZONES["out"], ZONES["out"] + 16)
    cases = ((a, b, 0), (a, b, 1), (bytes(16), bytes(16), 1))
    for left, right, misalign in cases:
        words = [int.from_bytes(bytes(pair), "little", signed=True)
                 for pair in zip(left, right)]
        # Only the upper eight words are halved. The logical shift by 32 contributes zero.
        words[8:] = [word // 2 for word in words[8:]]
        expected = bytes(max(0, min(word, 255)) for word in words)
        pristine = call(lambda *args: None, left, right, misalign=misalign)
        for variant, exports in functions:
            arena = call(exports["byte_pack_reload"], left, right, misalign=misalign)
            if arena[out] != expected:
                raise AssertionError(f"{variant} byte_pack_reload at offset {misalign}: "
                                     f"{arena[out].hex()} != {expected.hex()}")
            if outside_output(arena) != outside_output(pristine):
                raise AssertionError(f"{variant} byte_pack_reload wrote outside its output")
    print(f"byte_pack_reload: {len(cases)} reference cases match in all four builds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "original_lifted", "altered", "altered_lifted"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if not ctypes.windll.kernel32.IsProcessorFeaturePresent(39):  # PF_AVX_INSTRUCTIONS_AVAILABLE
        check_rejected_encodings()
        print("Native VEX differential requires OS-enabled AVX")
        sys.exit(77)
    pairs = inputs()
    _, native = bind(args.original)
    _, lifted = bind(args.original_lifted)
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
                out = slice(ZONES["out"], ZONES["out"] + 16)
                raise AssertionError(f"{name} pair {index}: lifted {observed[out].hex()} "
                                     f"!= native {expected[out].hex()} for {a.hex()} {b.hex()}")
        print(f"{name}: {len(pairs)} input pairs match bit for bit, guards intact")

    # The new word forms against an independent per-lane reference, in the native and lifted DLLs.
    for name in REFERENCED:
        for index, (a, b) in enumerate(pairs):
            expected = reference(name, a, b)
            for variant, exports in (("native", native), ("lifted", lifted)):
                observed = call(exports[name], a, b)[OUT]
                if observed != expected:
                    raise AssertionError(f"{name} pair {index}: {variant} {observed.hex()} != "
                                         f"reference {expected.hex()} for {a.hex()} {b.hex()}")
        print(f"{name}: native and lifted match the reference on {len(pairs)} pairs")

    # Exact aliases: the output is input A or input B. The whole arena must match, so a lifted
    # function that read an input after the native one wrote it (or before) would differ.
    reordered = 0
    for name in FUNCTIONS:
        for output in ("a", "b"):
            for a, b in pairs[:ALIAS_PAIRS]:
                expected = call(native[name], a, b, output)
                if call(lifted[name], a, b, output) != expected:
                    raise AssertionError(f"{name}: lifted differs from native with the output aliasing {output}")
                if name == "store_reload":
                    # With the output aliasing an input, the second reads see the first store.
                    separate = call(native[name], a, b)[ZONES["out"]:ZONES["out"] + 16]
                    reordered += expected[ZONES[output]:ZONES[output] + 16] != separate
    if not reordered:
        raise AssertionError("aliasing never changed store_reload's result; the order is not observed")
    print(f"aliased outputs match for all functions; store_reload reads its own store in {reordered} cases")

    # The two inputs may be the same buffer, alone or with the output aliasing it too.
    for name in FUNCTIONS:
        for output in ("out", "a"):
            for a, b in pairs[:ALIAS_PAIRS]:
                expected = call(native[name], a, b, output, same_inputs=True)
                if call(lifted[name], a, b, output, same_inputs=True) != expected:
                    raise AssertionError(f"{name}: lifted differs from native with RCX == RDX (output {output})")
    print("RCX == RDX matches for all functions, with a separate output and with the output aliasing it")

    # vmovdqu has no alignment requirement: run the unaligned-form functions 8 bytes off alignment.
    for name in UNALIGNED:
        for a, b in pairs[:ALIAS_PAIRS]:
            if call(lifted[name], a, b, misalign=8) != call(native[name], a, b, misalign=8):
                raise AssertionError(f"{name}: lifted differs from native on 8-byte-aligned buffers")
    print("vmovdqu functions match on buffers 8 bytes off 16-byte alignment")

    # The altered fixture replaces one arithmetic shift with a logical one in shift_edges.
    differing = 0
    for a, b in pairs:
        reference = call(altered["shift_edges"], a, b)
        if call(altered_lifted["shift_edges"], a, b) != reference:
            raise AssertionError("altered shift_edges: lifted differs from its native build")
        differing += reference != call(native["shift_edges"], a, b)
    if not differing:
        raise AssertionError("replacing vpsraw with vpsrlw changed no output")
    for name in ("widen_mix", "word_ops", "store_reload", "aligned_ops", "byte_pack_reload",
                 "sat_words", "cmp_words"):
        for a, b in pairs[:256]:
            if call(altered[name], a, b) != call(native[name], a, b):
                raise AssertionError(f"altered fixture changed {name}")
    print(f"vpsraw replaced by vpsrlw: {differing} of {len(pairs)} shift_edges outputs differ")

    # The altered build also swaps vpmulhuw for vpmulhw in mul_minmax_words. Its lifted DLL must
    # follow the altered native build and the altered reference, and both must differ from the
    # original, so the check fails if the unsigned high product is lowered as the signed one.
    changed = 0
    for a, b in pairs:
        reference_altered = reference("mul_minmax_words", a, b, altered=True)
        if call(altered["mul_minmax_words"], a, b)[OUT] != reference_altered:
            raise AssertionError("altered mul_minmax_words: native differs from the altered reference")
        if call(altered_lifted["mul_minmax_words"], a, b)[OUT] != reference_altered:
            raise AssertionError("altered mul_minmax_words: lifted differs from the altered reference")
        changed += reference_altered != reference("mul_minmax_words", a, b)
    if not changed:
        raise AssertionError("replacing vpmulhuw with vpmulhw changed no output")
    print(f"vpmulhuw replaced by vpmulhw: {changed} of {len(pairs)} mul_minmax_words outputs differ")

    check_byte_pack_reload((
        ("original", native), ("original lifted", lifted),
        ("altered", altered), ("altered lifted", altered_lifted),
    ))

    # Translation rejects a VEX.256 form and an unsupported mnemonic in patched copies.
    start = code_offsets(args.original, "widen_mix")
    data = args.original.read_bytes()

    def set_vex_l(buffer):
        # The first two-byte VEX prefix (C5) after the entry: bit 2 of its payload is VEX.L.
        prefix = buffer.index(0xC5, start)
        assert prefix == start, "widen_mix must start with a two-byte VEX instruction"
        buffer[prefix + 1] |= 0x04

    def vpaddw_to_vpaddd(buffer):
        # vpaddw is VEX opcode FD and vpaddd FE; patch the opcode byte of the first vpaddw.
        decoder = Decoder(64, bytes(data[start:start + 256]), ip=0)
        for instruction in decoder:
            if instruction.mnemonic == Mnemonic.VPADDW:
                vex_length = 2 if data[start + instruction.ip] == 0xC5 else 3
                opcode = start + instruction.ip + vex_length
                assert buffer[opcode] == 0xFD, hex(buffer[opcode])
                buffer[opcode] = 0xFE
                return
        raise AssertionError("no vpaddw in widen_mix")

    def set_vex_l_on_store(buffer):
        # VEX.L on the store makes its source a YMM register while the destination stays memory.
        decoder = Decoder(64, bytes(data[start:start + 256]), ip=0)
        for instruction in decoder:
            if instruction.mnemonic == Mnemonic.VMOVDQU and instruction.op0_kind == OpKind.MEMORY:
                prefix = start + instruction.ip
                # VEX.L is bit 2 of the last prefix byte: byte 1 of C5 xx, byte 2 of C4 xx xx.
                # [R8] needs VEX.B, so this store uses the three-byte form.
                assert buffer[prefix] in (0xC4, 0xC5), hex(buffer[prefix])
                buffer[prefix + (2 if buffer[prefix] == 0xC4 else 1)] |= 0x04
                return
        raise AssertionError("no vmovdqu store in widen_mix")

    expect_rejected(args.original, set_vex_l, "VEX.256 form")
    expect_rejected(args.original, set_vex_l_on_store, "VEX.256 form")
    expect_rejected(args.original, vpaddw_to_vpaddd, "unsupported instruction")
    print("rejected: a VEX.256 load, a VEX.256 store and vpaddd")
    check_rejected_encodings()


if __name__ == "__main__":
    main()
