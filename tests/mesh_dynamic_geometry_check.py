"""Check CPU export modeling for a source-built mesh with runtime geometry words."""

from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from disasm_rdna2 import decode_instruction
from mesh_entry_aot import Wave
from pixel_aot import disassemble


SOURCE = Path(__file__).with_name("mesh_dynamic_geometry.s")
WAVE64 = "+wavefrontsize64,-wavefrontsize32"
RETIRED_PREFIX = bytes.fromhex("00008abf 03ff8893 18000400 088006bf 020084bf 00008abf 010082bf 4c0082bf")
MESH_LIMITS = (4, 2)
SECOND = 0x300402  # (2, 1, 3)
DUPLICATE = 0x402  # (2, 1, 0)
# Name, input allocation word, input second index word, expected vertices and primitives.
CASES = (
    ("triangle", 0x1003, SECOND, 3, ((0, 1, 2),)),
    ("quad", 0x2004, SECOND, 4, ((0, 1, 2), (2, 1, 3))),
    ("zero", 0, SECOND, 0, ()),
    ("oversized count", (2 << 12) | (MESH_LIMITS[0] + 1), SECOND, 0, ()),
    ("out-of-range index", 0x2004, 0x400402, 0, ()),
    ("duplicate triangle", 0x2004, DUPLICATE, 4, ((0, 1, 2), (2, 1, 0))),
)

# LLVM 15 .text for SOURCE. Keep these bytes in source so CPU checks need no LLVM.
CODE = bytes.fromhex(
    """
    11038abe 80038bbe 050608f4 200000fa
    7fc08cbf cc0320f4 000000fa 8c0320f4
    040000fa 7fc08cbf 0fff06bf 03100000
    010084bf 0c0082bf 0fff06bf 04200000
    080084bf 0eff06bf 02043000 010084bf
    050082bf 0eff06bf 02040000 010084bf
    010082bf 80038fbe 0fff9093 00000c00
    0f8c1390 000065d7 c1000100 000066d7
    c1000200 03ff8893 18000400 000046d7
    080c0104 7e0496be 8000a47d 020088bf
    0f03fcbe 090090bf 1604febe 00008abf
    1000a87d 81001436 82001636 1400c2d4
    81140200 0c00c2d4 82160200 010001d5
    f3e45100 020001d5 f3e43100 050001d5
    80e45100 060001d5 80e43100 8002067e
    f202087e f0020e7e cf0800f8 01020304
    0f0200f8 05060704 1604febe 1300a87d
    1400c2d4 81000200 ff02127e 00042000
    090001d5 091d5000 410900f8 09000000
    000081bf
    """
)

# Wave 0 exports the first 64 vertices and ends. Wave 1 passes a second barrier
# and exports the last four. The second triangle uses its top-right corner (67).
WIDE_COUNTS, WIDE_SECOND = 0x2044, 0x4300402
SURVIVOR_BARRIER = """s_barrier
s_cmp_eq_u32 s8, 1
s_cbranch_scc0 export_vertices
s_barrier
export_vertices:"""


def surviving_wave_code():
    barrier = bytes.fromhex("00008abf")
    assert CODE.count(barrier) == 1
    code = CODE.replace(barrier, barrier + bytes.fromhex("088106bf 010084bf 00008abf"))
    return code.replace(struct.pack("<I", 0x2004), struct.pack("<I", WIDE_COUNTS)).replace(
        struct.pack("<I", SECOND), struct.pack("<I", WIDE_SECOND))


def wave64_pair(token):
    match = re.fullmatch(r"s(\d+)", token)
    assert match and int(match[1]) % 2 == 0, token
    low = int(match[1])
    return f"s[{low}:{low + 1}]"


def decode(code):
    """Adapt Python disassembler spellings to the model's LLVM spellings."""
    instructions = []
    pc = 0
    while pc < len(code):
        size, _, text, _ = decode_instruction(code, pc)
        assert size and pc + size <= len(code), (pc, size)
        op, _, operands = text.partition(" ")
        if op == "s_branch" or op.startswith("s_cbranch_"):
            target = re.fullmatch(r"label_(0x[0-9a-f]+)", operands)
            assert target, (pc, text)
            displacement = int(target[1], 16) - pc - size
            assert displacement % 4 == 0, (pc, text)
            text = f"{op} {displacement // 4}"
        elif text == "s_sendmsg 0x9":
            text = "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)"
        elif text == "s_barrier 0x0":
            text = "s_barrier"
        elif text.startswith("exp target_20 "):
            text = "exp prim " + operands.removeprefix("target_20 ")
        elif op in ("v_cmp_eq_u32_e64", "v_cndmask_b32_e64"):
            args = operands.split(", ")
            mask = 0 if op == "v_cmp_eq_u32_e64" else -1
            args[mask] = wave64_pair(args[mask])
            text = op + " " + ", ".join(args)
        elif op == "s_buffer_load_dword" and operands.endswith(", 0x0"):
            text = text.removesuffix("0x0") + "null"
        instructions.append((pc, text))
        pc += size
    return instructions


def reassemble(assembler, objcopy, library, *, retiring_wave=None):
    with tempfile.TemporaryDirectory() as directory:
        source = SOURCE
        expected = CODE
        if retiring_wave == 1:
            source = Path(directory) / "retired.s"
            prefix = SOURCE.with_name("mesh_retired_wave_prefix.s").read_text(encoding="ascii")
            source.write_text(prefix + "\n" + SOURCE.read_text(encoding="ascii"), encoding="ascii")
            expected = RETIRED_PREFIX + CODE
        elif retiring_wave == 0:
            source = Path(directory) / "wide.s"
            program = SOURCE.read_text(encoding="ascii").replace("s_barrier\n", SURVIVOR_BARRIER + "\n")
            program = program.replace("0x2004\n", f"{WIDE_COUNTS:#x}\n").replace("0x300402\n", f"{WIDE_SECOND:#x}\n")
            source.write_text(program, encoding="ascii")
            expected = surviving_wave_code()
        obj, raw = (Path(directory) / name for name in ("fixture.o", "fixture.bin"))
        subprocess.run(
            [assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-mattr=" + WAVE64,
             "-filetype=obj", source, "-o", obj],
            check=True,
        )
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
        code = raw.read_bytes()
    assert code == expected, "Embedded bytes differ from the assembled mesh"
    native = disassemble(code, library, offsets=True, features=WAVE64.encode("ascii"))
    for llvm_instruction, python_instruction in zip(native, decode(code), strict=True):
        assert llvm_instruction == python_instruction, (llvm_instruction, python_instruction)


def words(values):
    return struct.unpack("<4I", struct.pack("<4f", *values))


def check_cases(decoded):
    for name, packed_counts, second, vertex_count, triangles in CASES:
        waves = [Wave(index, [packed_counts, second]) for index in range(2)]
        for wave in waves:
            wave.run(decoded, {})
        first_wave, second_wave = waves
        assert first_wave.counts == [(vertex_count, len(triangles))], (name, first_wave.counts)
        assert not second_wave.counts, (name, second_wave.counts)
        assert first_wave.barriers == second_wave.barriers and len(first_wave.barriers) == 1, name
        assert not (second_wave.positions or second_wave.attributes or second_wave.indices), name

        position, attribute = {}, {}
        for index in range(vertex_count):
            x, y = index % 2, index // 2
            position[index] = words((2 * x - 1, 2 * y - 1, 0, 1))
            attribute[index] = words((x, y, 0.5, 1))
        assert {i: tuple(v) for i, v in first_wave.positions.items()} == position, name
        assert {i: tuple(v) for i, v in first_wave.attributes.items()} == attribute, name
        expected_indices = {i: list(triangle) for i, triangle in enumerate(triangles)}
        assert first_wave.indices == expected_indices, (name, first_wave.indices)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "binary":
        reassemble(*sys.argv[2:])
        reassemble(*sys.argv[2:], retiring_wave=1)
        reassemble(*sys.argv[2:], retiring_wave=0)
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: mesh_dynamic_geometry_check.py [binary llvm-mc llvm-objcopy libLLVM]")
    check_cases(decode(CODE))
    print("Six mesh inputs matched the CPU allocation and export model")
