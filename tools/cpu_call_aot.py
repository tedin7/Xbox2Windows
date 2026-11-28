"""Recompile a bounded Windows x64 caller with direct calls or one ordinary import call."""

import argparse
import os
from pathlib import Path
import struct
import sys
import tempfile

import pefile

from cpu_leaf_aot import bounded_code, export_rva


GPR = {0: "rax", 1: "rcx", 2: "rdx", 3: "rbx", 6: "rsi"}
OPERATORS = {0x89: "=", 0x8b: "=", 0x01: "+=", 0x03: "+=",
             0x29: "-=", 0x2b: "-=", 0x31: "^=", 0x33: "^="}


def r8_register_move(body, pos):
    """Decode a 64-bit register move between R8 and a profile GPR as (write, read)."""
    if pos + 3 > len(body) or body[pos + 2] >> 6 != 3:
        return None
    rex, opcode, modrm = body[pos:pos + 3]
    reg, rm = GPR.get((modrm >> 3) & 7), GPR.get(modrm & 7)
    # REX.B selects R8 in the r/m field and REX.R selects it in the reg field.
    if rex == 0x49 and modrm & 7 == 0 and reg:
        return {0x8b: (reg, "r8"), 0x89: ("r8", reg)}.get(opcode)
    if rex == 0x4c and (modrm >> 3) & 7 == 0 and rm:
        return {0x8b: ("r8", rm), 0x89: (rm, "r8")}.get(opcode)
    return None


def callee_name(index):
    # The first callee keeps the name used before more than one target was supported.
    return "lifted_callee" if index == 0 else f"lifted_callee{index + 1}"


def lift_body(start, code, *, caller, bind_import=None, name=None,
              allow_distinct_imports=False, bind_direct=None):
    # In this profile only RBX/RSI survive the call. The host compiler owns
    # physical register allocation and emits its own ABI frame for each function.
    frame = 40 if caller else 8
    prologue = bytes.fromhex("53 56 48 83 ec") + bytes([frame])
    epilogue = bytes.fromhex("48 83 c4") + bytes([frame]) + bytes.fromhex("5e 5b c3")
    if not code.startswith(prologue) or not code.endswith(epilogue):
        raise ValueError(f"unsupported save/restore frame at RVA {start:#x}")
    body = code[len(prologue):-len(epilogue)]
    available = {"rcx", "rdx", "r8"}
    # Registers that hold the entry R8 value, so stores stay inside the trace.
    pointer = {"r8"}
    lines = []
    target = None
    imported_symbols = {}
    pos = 0
    while pos < len(body):
        rva = start + len(prologue) + pos
        reads = set()
        write = None
        via_iat = body[pos:pos + 2] == b"\xff\x15"
        is_call = via_iat or body[pos] == 0xe8
        r8_move = r8_register_move(body, pos)
        if is_call:
            # Direct calls may repeat or name further targets, and each distinct
            # target is lifted once. Import slots require an explicit binding.
            if not caller:
                raise ValueError(f"unexpected additional call at RVA {rva:#x}")
            if target is not None and (target[0] == "import") != via_iat:
                raise ValueError(f"mixed import and direct calls at RVA {rva:#x}")
            length = 6 if via_iat else 5
            if pos + length > len(body):
                raise ValueError(f"truncated call at RVA {rva:#x}")
            address = rva + length + struct.unpack_from("<i", body, pos + length - 4)[0]
            if via_iat:
                if bind_import is None:
                    raise ValueError(f"no provider binding for IAT slot RVA {address:#x}")
                if target is not None and target[1] != address and not allow_distinct_imports:
                    raise ValueError(f"call to a second IAT slot at RVA {rva:#x}")
                if address not in imported_symbols:
                    imported_symbols[address] = bind_import(address)
                symbol = imported_symbols[address]
                if target is None:
                    target = ("import", address, symbol)
                callee = f"::{symbol}"
            else:
                if target is None:
                    target = ("direct", [])
                if address not in target[1]:
                    target[1].append(address)
                callee = bind_direct(address) if bind_direct else callee_name(target[1].index(address))
            reads = {"rcx", "rdx", "r8"}
            statement = f"rax = {callee}(rcx, rdx, r8);"
            write = "rax"
            pos += length
        elif r8_move:
            write, source = r8_move
            reads = {source}
            if write == "r8":
                statement = f"r8 = reinterpret_cast<word*>({source});"
            else:
                statement = f"{write} = reinterpret_cast<word>(r8);"
            pos += 3
        elif body[pos:pos + 2] == b"\x49\x89" and pos + 3 <= len(body):
            modrm = body[pos + 2]
            src = GPR.get((modrm >> 3) & 7)
            mode = modrm >> 6
            if modrm & 7 or mode not in (0, 1) or not src or pos + 3 + mode > len(body):
                raise ValueError(f"unsupported output store at RVA {rva:#x}")
            displacement = body[pos + 3] if mode else 0
            if displacement not in (0, 8, 16):
                raise ValueError(f"store outside trace at RVA {rva:#x}")
            reads = {"r8", src}
            statement = f"r8[{displacement // 8}] = {src};"
            pos += 3 + mode
        elif body[pos] == 0x48 and pos + 3 <= len(body):
            opcode, modrm = body[pos + 1:pos + 3]
            reg, rm = (modrm >> 3) & 7, modrm & 7
            if modrm >> 6 != 3 or rm not in GPR:
                raise ValueError(f"unsupported register operand at RVA {rva:#x}")
            write = GPR[rm]
            if opcode == 0x83 and reg in (0, 5) and pos + 4 <= len(body):
                imm = struct.unpack_from("b", body, pos + 3)[0]
                operator = "+=" if reg == 0 else "-="
                source = f"static_cast<word>({imm})"
                pos += 4
            elif opcode in OPERATORS and reg in GPR:
                operator = OPERATORS[opcode]
                source = GPR[reg]
                if opcode & 2:
                    write, source = source, write
                reads.add(source)
                pos += 3
            else:
                raise ValueError(f"unsupported register instruction at RVA {rva:#x}")
            if operator != "=":
                reads.add(write)
            statement = f"{write} {operator} {source};"
        else:
            raise ValueError(f"unsupported or incomplete instruction at RVA {rva:#x}")

        if not reads <= available:
            raise ValueError(f"undefined register at RVA {rva:#x}: {sorted(reads - available)}")
        if r8_move and write == "r8" and source not in pointer:
            raise ValueError(f"R8 reloaded without the output pointer at RVA {rva:#x}")
        if is_call:
            available.intersection_update({"rbx", "rsi"})
            pointer.intersection_update({"rbx", "rsi"})
        if write:
            available.add(write)
            # Only an R8 move copies the output pointer; any other write replaces it.
            if r8_move:
                pointer.add(write)
            else:
                pointer.discard(write)
        lines.append("    " + statement)

    if "rax" not in available or (caller and target is None):
        raise ValueError(f"missing return value or call at RVA {start:#x}")
    if name is None:
        name = "lifted" if caller else "lifted_callee"
    signature = f'extern "C" __declspec(dllexport) __declspec(noinline) word {name}(word rcx, word rdx, word* r8)'
    return "\n".join([signature + " {", "    word rax, rbx, rsi;", *lines, "    return rax;", "}"]), target


def resolve_import(pe, slot, provider, symbol, link_name=None):
    provider = Path(provider)
    if link_name is not None and (not isinstance(link_name, str) or not link_name.strip()):
        raise ValueError("link name must not be empty")
    pe.parse_data_directories(directories=[1])  # ordinary import directory
    matches = [(desc.dll, imp) for desc in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
               for imp in desc.imports if imp.address - pe.OPTIONAL_HEADER.ImageBase == slot]
    if len(matches) != 1:
        raise ValueError(f"no unique ordinary import at IAT slot RVA {slot:#x}")
    module, imp = matches[0]
    dll = module.decode("ascii")
    # pefile names some system ordinal imports from a built-in table; keep the stored ordinal.
    by_ordinal = imp.import_by_ordinal
    name = f"#{imp.ordinal}" if by_ordinal else imp.name.decode("ascii")
    identity = f"{dll}!{name}"
    if dll.casefold() != provider.name.casefold() or symbol != name:
        raise ValueError(f"unbound import {identity} at IAT slot RVA {slot:#x}")
    if by_ordinal and not link_name:
        raise ValueError(f"import-library link name required for {identity}")
    if not by_ordinal and link_name is not None:
        raise ValueError(f"unexpected link name for named import {identity}")
    identifier = link_name if by_ordinal else name
    if not identifier.isascii() or not identifier.isidentifier():
        raise ValueError(f"unsupported C import name {identity}")
    if not provider.is_file():
        raise ValueError(f"missing provider for {identity}")
    with pefile.PE(str(provider), fast_load=True) as library:
        try:
            rva = export_rva(library, imp.ordinal if by_ordinal else name)
        except ValueError as exc:
            raise ValueError(f"unresolved import {identity}: {exc}") from exc
        section = library.get_section_by_rva(rva)
        if not section or not section.Characteristics & 0x20000000:
            raise ValueError(f"non-executable export for {identity}")
    return identifier


def translate(image, entry, binding=None, link_name=None):
    if link_name is not None:
        if not binding:
            raise ValueError("link name requires a provider binding")
        if not isinstance(link_name, str) or not link_name.strip():
            raise ValueError("link name must not be empty")
    if binding:
        if len(binding) != 2:
            raise ValueError("expected provider path and symbol for --bind")
        provider_raw, bind_symbol = binding
        if not str(provider_raw).strip() or not isinstance(bind_symbol, str) or not bind_symbol.strip():
            raise ValueError("provider path and symbol must not be empty")
        provider_path = Path(provider_raw)
        if not provider_path.is_file():
            raise ValueError(f"missing provider for {provider_path.name}!{bind_symbol}")
    with pefile.PE(str(image), fast_load=True) as pe:
        start = export_rva(pe, entry)
        code = bounded_code(pe, start)
        def bind_import(slot):
            return resolve_import(pe, slot, Path(binding[0]), binding[1], link_name)

        caller, target_info = lift_body(start, code, caller=True, bind_import=bind_import if binding else None)
        kind = target_info[0]
        if kind == "import":
            callees = [f'extern "C" __declspec(dllimport) word {target_info[2]}(word, word, word*);']
        else:
            target = target_info[1]
            if binding:
                raise ValueError("provider binding supplied for a direct call")
            intervals = [(start, start + len(code))]
            callees = []
            for index, address in enumerate(target):
                callee_code = bounded_code(pe, address)
                end = address + len(callee_code)
                # Each target must own an interval apart from the caller and the other callees.
                if any(address < high and low < end for low, high in intervals):
                    raise ValueError(f"overlapping call target at RVA {address:#x}")
                intervals.append((address, end))
                callee, _ = lift_body(address, callee_code, caller=False, name=callee_name(index))
                callees.append(callee)
    body = "\n\n".join([*callees, caller])
    return f"#include <cstdint>\nusing word = std::uint64_t;\n{body}\n"


def _validate_paths(
    image: Path,
    export: str,
    output: Path,
    bind: list[str] | None = None,
    link_name: str | None = None,
) -> None:
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
    if link_name is not None:
        if not bind:
            raise ValueError("link name requires a provider binding")
        if not isinstance(link_name, str) or not link_name.strip():
            raise ValueError("link name must not be empty")
    if bind:
        if len(bind) != 2:
            raise ValueError("expected provider path and symbol for --bind")
        provider_raw, bind_symbol = bind
        if not str(provider_raw).strip() or not isinstance(bind_symbol, str) or not bind_symbol.strip():
            raise ValueError("provider path and symbol must not be empty")
        provider = Path(provider_raw)
        if not provider.is_file():
            raise ValueError(f"missing provider for {provider.name}!{bind_symbol}")
        inputs = (image, provider)
    else:
        inputs = (image,)
    out_resolved = output.resolve()
    for input_path in inputs:
        if out_resolved == input_path.resolve() or (output.exists() and input_path.samefile(output)):
            raise ValueError(f"output path must not match input path: {output}")


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
    parser.add_argument("--bind", nargs=2, metavar=("PROVIDER_DLL", "SYMBOL"),
                        help="export name or #ordinal with the fixture's two-word and output-pointer ABI")
    parser.add_argument("--link-name", help="local import-library symbol for an ordinal binding")
    args = parser.parse_args(argv)
    try:
        _validate_paths(args.image, args.export, args.output, args.bind, args.link_name)
        _write_atomic(args.output, translate(args.image, args.export, args.bind, args.link_name))
    except (OSError, ValueError, pefile.PEFormatError) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
