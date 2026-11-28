"""Lift acyclic pointer checks and early returns across AMD64 chained unwind ranges."""

import argparse
import os
from pathlib import Path
import re
import sys
import tempfile

import pefile
from iced_x86 import Code, Decoder, OpKind, Register as R, RegisterExt

from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges, image_bytes


SAVED = {reg: f"saved_{reg}" for reg in (R.RBX, R.RBP, R.RDI, R.RSI, R.R12, R.R13, R.R14, R.R15)}
GPRS = set(SAVED) | {R.RAX, R.RCX, R.RDX, R.R8, R.R9, R.R10, R.R11}


def translate(image, name, partial=False, callback_offset=None, ffx_destroy=False):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("export name must be non-empty")
    if ffx_destroy and (partial or callback_offset is not None):
        raise ValueError("the destroy contract requires a complete translation and its own typed bindings")
    if callback_offset is not None and (
        type(callback_offset) is not int
        or not 0 <= callback_offset <= 0x7fffffff
        or callback_offset % 8
    ):
        raise ValueError("callback slot must be a nonnegative aligned table offset")
    with pefile.PE(str(image), fast_load=True) as pe:
        entry = export_rva(pe, name)
        free_slots = set()
        if ffx_destroy:
            pe.parse_data_directories(directories=[1])
            free_slots = {imp.address - pe.OPTIONAL_HEADER.ImageBase
                          for dll in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
                          if dll.dll.lower() == b"api-ms-win-crt-heap-l1-1-0.dll"
                          for imp in dll.imports if imp.name == b"free"}
        ranges = function_ranges(pe, entry)
        instructions = {}
        for begin, end in ranges:
            for ins in Decoder(64, image_bytes(pe, begin, end - begin, executable=True), ip=begin):
                if ins.is_invalid:
                    raise ValueError(f"invalid instruction in owned range at {ins.ip:#x}")
                instructions[ins.ip] = ins

    stops = set()
    returns = []
    lines = []
    steps = 0
    def walk(ip, regs, stack, sp, visited, indent, stage="entry", allocator_null=None):
        nonlocal steps
        while True:
            if ip not in instructions:
                raise ValueError(f"edge outside owned instruction boundaries at {ip:#x}")
            steps += 1
            if ip in visited or len(visited) >= 256 or steps > 2048:
                raise ValueError(f"cycle or early-return profile limit at {ip:#x}")
            visited.add(ip)
            ins = instructions[ip]
            code = ins.code
            dst, src = ins.op0_register, ins.op1_register
            reg_dst, reg_src = dst in GPRS, src in regs
            if ins.has_segment_prefix or ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix:
                raise ValueError(f"unsupported instruction prefix at {ip:#x}")
            base, disp = ins.memory_base, ins.memory_displacement
            simple = ins.memory_index == R.NONE
            slot = sp + disp
            if code == Code.PUSH_R64 and dst in regs:
                sp -= 8
                stack[sp] = regs[dst]
            elif code == Code.POP_R64 and reg_dst and sp in stack:
                regs[dst] = stack.pop(sp)
                sp += 8
            elif code in (Code.ADD_RM64_IMM8, Code.SUB_RM64_IMM8, Code.ADD_RM64_IMM32, Code.SUB_RM64_IMM32) and dst == R.RSP:
                delta = ins.immediate8to64 if code in (Code.ADD_RM64_IMM8, Code.SUB_RM64_IMM8) else ins.immediate32to64
                sp += delta if code in (Code.ADD_RM64_IMM8, Code.ADD_RM64_IMM32) else -delta
                if not -4096 <= sp <= 0 or sp % 8:
                    raise ValueError(f"unsupported stack adjustment at {ip:#x}")
                stack = {k: v for k, v in stack.items() if k >= sp}
            elif code in (Code.MOV_RM64_R64, Code.MOV_R64_RM64) and reg_dst and reg_src:
                regs[dst] = regs[src]
            elif (code == Code.MOV_RM64_R64 and ins.op0_kind == OpKind.MEMORY
                  and simple and base == R.RSP and reg_src and sp <= slot <= 32 and slot and slot % 8 == 0):
                stack[slot] = regs[src]
            elif code == Code.MOV_R64_RM64 and reg_dst and simple and base == R.RSP and slot in stack:
                regs[dst] = stack[slot]
            elif (code in (Code.MOV_R64_RM64, Code.LEA_R64_M) and reg_dst and simple
                  and base in regs and regs[base] not in SAVED.values()):
                address = f"({regs[base]} + {disp}ull)"
                if code == Code.MOV_R64_RM64:
                    # The operand is read once, in guest order, into a local that every later use names. The
                    # read is volatile: the original faults on an inaccessible operand even when the value is
                    # overwritten before use, and the host compiler may drop an ordinary load whose value is unused.
                    var = f"load_{ip:x}"
                    lines.append(indent + f"const word {var} = load_always<word>({address});")
                    regs[dst] = var
                else:
                    regs[dst] = address
            elif code == Code.MOV_R32_IMM32 and RegisterExt.full_register(dst) in GPRS:
                regs[RegisterExt.full_register(dst)] = f"{ins.immediate32}ull"
            elif (code in (Code.XOR_RM32_R32, Code.XOR_R32_RM32, Code.XOR_RM64_R64, Code.XOR_R64_RM64)
                  and ins.op0_kind == ins.op1_kind == OpKind.REGISTER and dst == src
                  and RegisterExt.full_register(dst) in GPRS):
                regs[RegisterExt.full_register(dst)] = "0ull"
            elif code in (Code.MOV_RM32_R32, Code.MOV_R32_RM32) and ins.op0_kind == ins.op1_kind == OpKind.REGISTER:
                dest, source = RegisterExt.full_register(dst), RegisterExt.full_register(src)
                if dest not in GPRS or regs.get(source) in (None, *SAVED.values()):
                    raise ValueError(f"unknown 32-bit register at {ip:#x}")
                regs[dest] = f"static_cast<std::uint32_t>({regs[source]})"
            elif code == Code.TEST_RM64_R64 and dst == src and reg_src and regs[src] not in SAVED.values():
                branch = instructions.get(ins.next_ip)
                if branch is None or branch.code not in (
                        Code.JE_REL8_64, Code.JE_REL32_64, Code.JNE_REL8_64, Code.JNE_REL32_64):
                    raise ValueError(f"expected JE or JNE after TEST at {ip:#x}")
                taken_zero = branch.code in (Code.JE_REL8_64, Code.JE_REL32_64)
                op = "==" if taken_zero else "!="
                lines.append(indent + f"if ({regs[src]} {op} 0) {{")
                checks_allocator = ffx_destroy and regs[src] == "second"
                walk(branch.near_branch_target, regs.copy(), stack.copy(), sp, visited.copy(), indent + "    ",
                     stage, taken_zero if checks_allocator else allocator_null)
                lines.append(indent + "}")
                if checks_allocator:
                    allocator_null = not taken_zero
                ip = branch.next_ip
                continue
            elif code in (Code.JMP_REL8_64, Code.JMP_REL32_64):
                ip = ins.near_branch_target
                continue
            elif code == Code.RETNQ:
                if stage == "destroyed":
                    raise ValueError(f"missing wrapper cleanup at {ip:#x}")
                if sp or any(regs[r] != v for r, v in SAVED.items()) or regs.get(R.RAX) in (None, *SAVED.values()):
                    raise ValueError(f"unrestored state or unknown return at {ip:#x}")
                returns.append(ip)
                lines.extend([indent + f"*output = {regs[R.RAX]};", indent + "return 1;"])
                return
            elif code == Code.CALL_RM64 and ffx_destroy and ins.op0_kind == OpKind.MEMORY and simple:
                first_arg = regs.get(R.RCX)
                second_arg = regs.get(R.RDX)
                if sp > -40 or (sp + 8) % 16 or first_arg in (None, *SAVED.values()):
                    raise ValueError(f"invalid destroy call frame or argument at {ip:#x}")
                result = None
                if (stage == "entry" and disp == 8 and base in regs and regs[base] not in SAVED.values()
                        and second_arg == "second"):
                    lines.extend([
                        indent + f"if (!known || !known->destroy || !known->table || known->table != {regs[base]}) return 0;",
                        indent + "if (read_word(known->table + 8) != reinterpret_cast<word>(known->destroy)) return 0;",
                        indent + "if (second != known->allocator) return 0;",
                        indent + "if (second && (!known->dealloc || read_word(second + 16) != reinterpret_cast<word>(known->dealloc))) return 0;",
                        indent + "if (!second && !known->release) return 0;",
                    ])
                    result = f"status_{ip:x}"
                    lines.append(indent + f"const auto {result} = known->destroy(reinterpret_cast<void**>({first_arg}), reinterpret_cast<const Allocator*>(second));")
                    stage = "destroyed"
                elif (stage == "destroyed" and allocator_null is False and regs.get(base) == "second"
                      and disp == 16 and second_arg not in (None, *SAVED.values())):
                    lines.append(indent + f"known->dealloc(reinterpret_cast<void*>({first_arg}), reinterpret_cast<void*>({second_arg}));")
                    stage = "released"
                elif (stage == "destroyed" and allocator_null is True and ins.is_ip_rel_memory_operand
                      and ins.ip_rel_memory_address in free_slots):
                    lines.append(indent + f"known->release(reinterpret_cast<void*>({first_arg}));")
                    stage = "released"
                else:
                    raise ValueError(f"unbound destroy call at {ip:#x}")
                regs = {r: v for r, v in regs.items() if r in SAVED}
                if result is not None:
                    regs[R.RAX] = result
                stack = {k: v for k, v in stack.items() if not sp <= k < sp + 32}
            elif (code == Code.CALL_RM64 and callback_offset is not None
                  and ins.op0_kind == OpKind.MEMORY and simple and base in regs
                  and regs[base] not in SAVED.values() and disp == callback_offset):
                if (sp > -40 or (sp + 8) % 16 or
                        any(regs.get(r) in (None, *SAVED.values()) for r in (R.RCX, R.RDX))):
                    raise ValueError(f"unknown callback arguments or invalid call frame at {ip:#x}")
                table = regs[base]
                lines.extend([
                    indent + f"if (!binding || !binding->target || !binding->table || binding->table != {table}) return 0;",
                    indent + f"if (read_word(binding->table + {disp}ull) != reinterpret_cast<word>(binding->target)) return 0;",
                    indent + f"const word call_{ip:x} = binding->target({regs[R.RCX]}, {regs[R.RDX]});",
                ])
                regs = {r: v for r, v in regs.items() if r in SAVED}
                regs[R.RAX] = f"call_{ip:x}"
                stack = {k: v for k, v in stack.items() if not sp <= k < sp + 32}
            elif code == Code.CALL_RM64 and partial:
                stops.add(ip)
                lines.append(indent + "return 0;")
                return
            else:
                raise ValueError(f"unsupported operation at {ip:#x}: {ins}")
            ip = ins.next_ip

    walk(entry, {**SAVED, R.RCX: "first", R.RDX: "second"}, {}, 0, set(), "    ")
    for index, line in enumerate(lines):
        # A load whose value nothing uses is still performed.
        if " = load_always<word>(" in line:
            var = line.split("const word ", 1)[1].split(" ", 1)[0]
            if not any(re.search(rf"\b{var}\b", other) for other in lines[index + 1:]):
                lines[index] = f"{line} static_cast<void>({var});"
    source = """#include <cstdint>
#include <cstring>
using word = std::uint64_t;
// Every guest MOV load, once per instruction: the compiler must perform it even when its value is unused,
// because the original faults on an inaccessible address either way.
template <class T> static T load_always(word address) {
    return *reinterpret_cast<const volatile T __unaligned*>(address);
}
"""
    if ffx_destroy or callback_offset is not None:
        source += """// A check of a bound table slot or allocator word; its value is always compared.
static word read_word(word address) {
    word value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    return value;
}
"""
    if ffx_destroy:
        source += """struct Allocator;
using Destroy = std::uint32_t (*)(void**, const Allocator*);
using Dealloc = void (*)(void*, void*);
struct DestroyBinding { word table; Destroy destroy; word allocator; Dealloc dealloc; void (*release)(void*); };
extern "C" __declspec(dllexport) word try_destroy(word first, word second, word* output, const DestroyBinding* known) {
"""
    elif callback_offset is None:
        source += 'extern "C" __declspec(dllexport) word try_lifted(word first, word second, word* output) {\n'
    else:
        source += """using Callback = word (*)(word, word);
struct CallbackBinding { word table; Callback target; };
extern "C" __declspec(dllexport) word try_lifted_callback(word first, word second, word* output, const CallbackBinding* binding) {
"""
    return source + "\n".join(lines) + "\n}\n", ranges, returns, stops


def _validate_cli_inputs(image: Path, export: str, output: Path) -> None:
    if not image.is_file():
        raise ValueError(f"input image is not an existing file: {image}")
    if not export.strip():
        raise ValueError("export name must not be empty")
    if output.suffix.lower() != ".cpp":
        raise ValueError(f"output path must have a .cpp suffix: {output}")
    if output.is_symlink():
        raise ValueError(f"output path must not be a symbolic link: {output}")
    if output.is_dir():
        raise ValueError(f"output path is a directory: {output}")
    if output.parent.is_file() or (output.parent.exists() and not output.parent.is_dir()):
        raise ValueError(f"output parent is not a directory: {output.parent}")
    try:
        same = output.resolve() == image.resolve()
    except OSError:
        same = False
    if not same and output.exists():
        same = output.samefile(image)
    if same:
        raise ValueError(f"output path must not collide with input image: {output}")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    fd_open = True
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            fd_open = False
            handle.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        if fd_open:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("export")
    parser.add_argument("output", type=Path)
    parser.add_argument("--partial", action="store_true", help="return status 0 at an indirect call")
    parser.add_argument("--ffx-destroy", action="store_true", help="use the verified FidelityFX destroy and cleanup ABI")
    parser.add_argument("--callback-offset", type=lambda s: int(s, 0),
                        help="bind a table slot with a known Windows x64 word(word, word) callback ABI")
    args = parser.parse_args(argv)
    try:
        _validate_cli_inputs(args.image, args.export, args.output)
        source, ranges, returns, stops = translate(
            args.image, args.export, args.partial, args.callback_offset, args.ffx_destroy
        )
        _write_atomic(args.output, source)
        print(f"{len(ranges)} owned ranges, {len(returns)} return paths, indirect stops: {sorted(stops)}")
        return 0
    except (ValueError, OSError, pefile.PEFormatError) as error:
        print(f"{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
