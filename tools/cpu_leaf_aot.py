"""Lift a bounded AMD64 leaf function with the fixture's (uint32, uint32*) ABI to C++."""

import argparse
import os
from pathlib import Path
import struct
import sys
import tempfile

import pefile

from cpu_ranges import function_ranges, image_bytes


def export_rva(pe, name):
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    pe.parse_data_directories(directories=[0, 3])  # exports and runtime functions
    exports = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
    ordinal = isinstance(name, int)
    key = name if ordinal else name.encode("ascii")
    matches = [e.address for e in exports.symbols if (e.ordinal if ordinal else e.name) == key
               and e.address and not e.forwarder] if exports else []
    if len(matches) != 1:
        raise ValueError(f"expected one local export for {name}")
    return matches[0]


def bounded_code(pe, start):
    ranges = function_ranges(pe, start)
    if len(ranges) != 1:
        raise ValueError(f"multiple owned ranges at {start:#x}; use cpu_chained_aot")
    begin, end = ranges[0]
    return image_bytes(pe, begin, end - begin, executable=True)


def function_bytes(image, name):
    with pefile.PE(str(image), fast_load=True) as pe:
        start = export_rva(pe, name)
        return start, bounded_code(pe, start)


CONDITIONS = {
    0x72: ("<", False),
    0x73: (">=", False),
    0x74: ("==", False),
    0x75: ("!=", False),
    0x76: ("<=", False),
    0x77: (">", False),
    0x7c: ("<", True),
    0x7d: (">=", True),
    0x7e: ("<=", True),
    0x7f: (">", True),
}


def translate(start, code):
    if isinstance(start, bool) or not isinstance(start, int):
        raise ValueError(f"expected integer start RVA, got {type(start).__name__}")
    if start < 0:
        raise ValueError(f"expected non-negative start RVA, got {start}")
    if not isinstance(code, bytes):
        try:
            code = memoryview(code).tobytes()
        except TypeError as exc:
            raise ValueError(f"expected bytes-like code, got {type(code).__name__}") from exc
    # This profile initializes EAX from ECX before either branch. A compare and
    # its immediate consumer form one operation, so flags never cross a join.
    if not code.startswith((b"\x8b\xc1", b"\x89\xc8")):
        raise ValueError(f"expected mov eax, ecx at RVA {start:#x}")
    statements = {}
    targets = set()
    offset = 2
    while offset < len(code):
        pos = offset
        rva = start + pos
        prefix = code[pos:pos + 2]
        opcode = code[pos]
        jump = False
        if prefix == b"\x83\xf8" and pos + 5 <= len(code) and code[pos + 3] in CONDITIONS:
            limit = struct.unpack_from("b", code, pos + 2)[0] & 0xffffffff
            condition, signed = CONDITIONS[code[pos + 3]]
            offset += 5
            jump = True
            statement = (f"if (static_cast<std::int32_t>(eax) {condition} static_cast<std::int32_t>({limit}u)) "
                         if signed else f"if (eax {condition} {limit}u) ")
        elif opcode == 0x3d and pos + 7 <= len(code) and code[pos + 5] in CONDITIONS:
            limit = struct.unpack_from("<I", code, pos + 1)[0]
            condition, signed = CONDITIONS[code[pos + 5]]
            offset += 7
            jump = True
            statement = (f"if (static_cast<std::int32_t>(eax) {condition} static_cast<std::int32_t>({limit}u)) "
                         if signed else f"if (eax {condition} {limit}u) ")
        elif opcode == 0xeb and pos + 2 <= len(code):
            offset += 2
            jump = True
            statement = ""
        elif prefix in (b"\x83\xc0", b"\x83\xe8") and pos + 3 <= len(code):
            immediate = struct.unpack_from("b", code, pos + 2)[0] & 0xffffffff
            operator = "+=" if prefix[1] == 0xc0 else "-="
            statement = f"eax {operator} {immediate}u;"
            offset += 3
        elif opcode in (0x05, 0x2d) and pos + 5 <= len(code):
            immediate = struct.unpack_from("<I", code, pos + 1)[0]
            operator = "+=" if opcode == 0x05 else "-="
            statement = f"eax {operator} {immediate}u;"
            offset += 5
        elif prefix == b"\x89\x02":
            statement = "*output = eax;"
            offset += 2
        elif opcode == 0xc3:
            statement = "return eax;"
            offset += 1
        else:
            raise ValueError(f"unsupported or incomplete instruction at RVA {rva:#x}")
        if jump:
            displacement = struct.unpack_from("b", code, offset - 1)[0]
            target = offset + displacement
            if target < offset or target >= len(code):
                raise ValueError(f"expected forward in-function branch at RVA {rva:#x}")
            targets.add(target)
            statement += f"goto loc_{target:x};"
        statements[pos] = statement

    invalid = targets.difference(statements)
    if invalid:
        raise ValueError(f"branch into an instruction at RVA {start + min(invalid):#x}")
    if not statements or statements[pos] != "return eax;":
        raise ValueError(f"missing terminal return at RVA {start + len(code):#x}")
    lines = ["#include <cstdint>",
             'extern "C" __declspec(dllexport) std::uint32_t lifted(std::uint32_t input, std::uint32_t* output) {',
             "    std::uint32_t eax = input;"]
    for pos, statement in statements.items():
        if pos in targets:
            lines.append(f"loc_{pos:x}:")
        lines.append("    " + statement)
    return "\n".join([*lines, "}", ""])


def _validate_paths(image: Path, export: str, output: Path) -> None:
    if image.is_dir():
        raise IsADirectoryError(f"image path is a directory: {image}")
    if not image.is_file():
        raise FileNotFoundError(f"image file not found: {image}")
    if not isinstance(export, str) or not export.strip():
        raise ValueError("export name must not be empty")
    if output.suffix.lower() != ".cpp":
        raise ValueError(f"output path must have .cpp suffix: {output}")
    if output.is_symlink():
        raise ValueError(f"output path must not be a symlink: {output}")
    if output.is_dir():
        raise IsADirectoryError(f"output path is a directory: {output}")
    for parent in output.parents:
        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):
            raise NotADirectoryError(f"output parent is not a directory: {parent}")
    if output.resolve() == image.resolve() or (output.exists() and image.samefile(output)):
        raise ValueError(f"output path must not match input image: {output}")


def _write_atomic(path: Path, content: str) -> None:
    """Write UTF-8 text to a sibling temp file and atomically replace path, unlinking temp files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("export")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        _validate_paths(args.image, args.export, args.output)
        rva, code = function_bytes(args.image, args.export)
        _write_atomic(args.output, translate(rva, code))
    except (ValueError, OSError, pefile.PEFormatError) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
