"""Deterministic differential fuzz smoke test for the shared CPU lifter."""

import argparse
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
import random
import re
import struct
import sys

WIDTHS = {
    "b": ("al", "dl", "BYTE PTR", 1),
    "w": ("ax", "dx", "WORD PTR", 2),
    "d": ("eax", "edx", "DWORD PTR", 4),
    "q": ("rax", "rdx", "QWORD PTR", 8),
}
CMP_CONDITIONS = ("e", "ne", "l", "le", "g", "ge", "s", "ns", "o", "no", "p", "np", "b", "ae", "be", "a")
TEST_CONDITIONS = ("e", "ne", "l", "le", "g", "ge", "s", "ns", "p", "np")


@dataclass
class Function:
    name: str
    lines: list[str]
    prolog: list[str] = field(default_factory=list)
    # The functions this one calls or tail-jumps to, in order of first use, printed with it on a mismatch.
    callees: list["Function"] = field(default_factory=list)
    # For a negative control: the reason the translator must give for refusing it. It is assembled
    # and exported but neither translated with the others nor run.
    refused: str | None = None
    # A callee that only other functions of the image reach is not exported, so it is not run on its own.
    export: bool = True
    # Only AL is defined in RAX when it returns, so a caller widens it and no tail jump reaches it.
    narrow: bool = False
    # A scalar float export: '<ret>:<slots>' as cpu_sse_aot names it, for example 'd:ffq'. None for three 64-bit integer slots.
    signature: str | None = None

    def asm(self):
        head = [f"    {line}" for line in self.prolog]
        body = [f"    {line}" if not line.endswith(":") else line for line in self.lines]
        return "\n".join([f"{self.name} PROC FRAME", *head, "    .endprolog", *body, f"{self.name} ENDP"])


def alu(name, width, variant, rng):
    reg, rhs, ptr, size = WIDTHS[width]
    imm = rng.choice((1, 3, 7, 0x1f, 0x7f, 0xff))
    lines = ["mov rax, rcx"]
    if variant == 0:
        lines += [
            f"add {reg}, {rhs}",
            f"sub {reg}, {imm}",
            f"and {reg}, {rhs}",
            f"xor {reg}, {rng.choice((1, 3, 5, 9))}",
        ]
    else:
        lines += [
            f"or {reg}, {rhs}",
            f"add {reg}, {imm}",
            f"sub {reg}, {rhs}",
            f"xor {reg}, {rng.choice((1, 3, 5, 9))}",
        ]
    lines += [f"mov {ptr} [r8], {reg}", f"shr {reg}, {rng.choice((1, 3, 5))}", f"mov {ptr} [r8+{size}], {reg}", "ret"]
    return Function(name, lines)


def shift(name, width, rng):
    reg, _, ptr, size = WIDTHS[width]
    op = rng.choice(("shl", "shr", "sar", "rol", "ror"))
    count = rng.choice((1, 3, 5, 7))
    lines = ["mov rax, rcx", f"{op} {reg}, {count}", f"mov {ptr} [r8], {reg}", "ret"]
    return Function(name, lines)


def flags(name, width, rng):
    reg, rhs, ptr, _ = WIDTHS[width]
    producer = rng.choice(("cmp", "test"))
    cc = rng.choice(CMP_CONDITIONS if producer == "cmp" else TEST_CONDITIONS)
    label = f"{name}_taken"
    lines = [
        "mov rax, rcx",
        f"{producer} {reg}, {rhs}",
        "mov rax, 0",
        "mov rdx, 0",
        f"set{cc} BYTE PTR [r8]",
        "mov eax, 42",
        "mov r10d, 17",
        f"cmov{cc} eax, r10d",
        "mov DWORD PTR [r8+4], eax",
        f"j{cc} {label}",
        "mov eax, 1",
        "mov DWORD PTR [r8+8], eax",
        "ret",
        f"{label}:",
        "mov eax, 2",
        "mov DWORD PTR [r8+8], eax",
        "ret",
    ]
    return Function(name, lines)


def branch_diamond(name, rng):
    label = f"{name}_even"
    add = rng.choice((1, 3, 7))
    lines = [
        "mov eax, ecx",
        "test eax, 1",
        f"jz {label}",
        f"add eax, {add}",
        "mov DWORD PTR [r8], eax",
        "mov eax, 3",
        "ret",
        f"{label}:",
        "sub eax, edx",
        "mov DWORD PTR [r8], eax",
        "mov eax, 4",
        "ret",
    ]
    return Function(name, lines)


def branch_fallthrough(name, rng):
    label = f"{name}_next"
    add = rng.choice((1, 3, 7))
    lines = [
        "mov eax, ecx",
        "cmp eax, edx",
        f"jne {label}",
        f"{label}:",
        f"add eax, {add}",
        "mov DWORD PTR [r8], eax",
        "ret",
    ]
    return Function(name, lines)


def mixed(name, width, rng):
    reg, rhs, _, _ = WIDTHS[width]
    cc = rng.choice(TEST_CONDITIONS)
    lines = [
        "mov rax, rcx",
        f"test {reg}, {rhs}",
        "mov rax, 0",
        "mov rdx, 0",
        f"set{cc} BYTE PTR [r8]",
        "mov eax, 123456789",
        "mov r10d, 987654321",
        f"cmov{cc} eax, r10d",
        "mov DWORD PTR [r8+4], eax",
        "ret",
    ]
    return Function(name, lines)


def generate(seed, count):
    rng = random.Random(seed)
    recipes = [
        ("alu", "b", 0), ("alu", "b", 1),
        ("alu", "w", 0), ("alu", "w", 1),
        ("alu", "d", 0), ("alu", "d", 1),
        ("alu", "q", 0), ("alu", "q", 1),
        ("shift", "d", 0), ("shift", "d", 1),
        ("shift", "q", 0), ("shift", "q", 1),
        ("flags", "b", 0), ("flags", "w", 0),
        ("flags", "d", 0), ("flags", "q", 0),
        ("branch_diamond", None, None),
        ("branch_fallthrough", None, None),
        ("mixed", "b", 0), ("mixed", "d", 0),
    ]
    functions = []
    for index in range(count):
        kind, width, variant = recipes[index % len(recipes)]
        name = f"fuzz_{index}"
        if kind == "alu":
            functions.append(alu(name, width, variant, rng))
        elif kind == "shift":
            functions.append(shift(name, width, rng))
        elif kind == "flags":
            functions.append(flags(name, width, rng))
        elif kind == "branch_diamond":
            functions.append(branch_diamond(name, rng))
        elif kind == "branch_fallthrough":
            functions.append(branch_fallthrough(name, rng))
        else:
            functions.append(mixed(name, width, rng))
    return functions


REGISTER_PARTS = {
    "rax": ("eax", "ax", "al"), "rcx": ("ecx", "cx", "cl"), "rdx": ("edx", "dx", "dl"),
    "rbx": ("ebx", "bx", "bl"), "rsi": ("esi", "si", "sil"), "rdi": ("edi", "di", "dil"),
    **{f"r{n}": (f"r{n}d", f"r{n}w", f"r{n}b") for n in range(9, 16)},
}
PTR = {8: "QWORD PTR", 4: "DWORD PTR", 2: "WORD PTR", 1: "BYTE PTR"}
NONVOLATILE = ("rbx", "rsi", "rdi", "r12", "r13", "r14", "r15")
# The partitions of a 16-byte spill region into consecutive stores, so stores of every width
# meet at unaligned boundaries and a reload can straddle two or three of them.
SPILL_WIDTHS = (8, 4, 2, 1)


def part(register, width):
    return register if width == 8 else REGISTER_PARTS[register][{4: 0, 2: 1, 1: 2}[width]]


def partition(rng, total):
    widths = []
    while total:
        width = rng.choice([width for width in SPILL_WIDTHS if width <= total])
        widths.append(width)
        total -= width
    return widths


def frame_function(name, rng):
    """A fixed RSP frame: saved nonvolatile registers, a 16-byte spill region and partial reloads.

    Each saved register is overwritten from the arguments before it is read, so the function never
    reads a nonvolatile register's entry value, which cpu_switch_aot rejects.
    """
    prolog, lines = [], []
    home = rng.random() < 0.3
    if home:
        # The MSVC /O2 shape: RBX and RSI saved into home space before the allocation, restored by MOV.
        pushed, moved = ["rdi"], ["rbx", "rsi"]
        prolog += ["mov QWORD PTR [rsp+8], rbx", "mov QWORD PTR [rsp+10h], rsi"]
    else:
        pushed, moved = rng.sample(NONVOLATILE, rng.choice((1, 2, 3))), []
    for register in pushed:
        prolog += [f"push {register}", f".pushreg {register}"]
    # Entry RSP is 8 mod 16; keep RSP 16-byte aligned after the allocation.
    size = rng.choice((16, 32)) + (8 if len(pushed) % 2 == 0 else 0)
    prolog += [f"sub rsp, {size}", f".allocstack {size}"]
    for index, register in enumerate(moved):
        prolog.append(f".savereg {register}, {size + 8 * len(pushed) + 8 * (index + 1)}")

    saved = pushed + moved
    for register in saved:
        lines.append(rng.choice((f"mov {register}, rcx", f"mov {register}, rdx",
                                 f"lea {register}, [rcx+rdx*2]", f"lea {register}, [rdx+rcx*4+7]")))
        lines.append(rng.choice((f"xor {register}, rdx", f"rol {register}, 13", f"add {register}, rcx")))
    sources = ["rcx", "rdx", *saved]
    base = rng.choice([offset for offset in (0, 8) if offset + 16 <= size])

    offset, boundaries = base, []
    for width in partition(rng, 16):
        if offset != base:
            boundaries.append(offset)
        if rng.random() < 0.2:
            lines.append(f"mov {PTR[width]} [rsp+{offset}], {rng.choice((0x5A, 0x7F, 0x81, 0x33))}")
        else:
            lines.append(f"mov {PTR[width]} [rsp+{offset}], {part(rng.choice(sources), width)}")
        offset += width

    if rng.random() < 0.5:
        width = rng.choice((1, 2, 4, 8))
        at = base + rng.randrange(16 - width + 1)
        lines.append(f"add {PTR[width]} [rsp+{at}], {part(rng.choice(('rdx', *saved)), width)}")
    if rng.random() < 0.5:
        label, join = f"{name}_odd", f"{name}_join"
        lines += [
            "test cl, 1",
            f"jnz {label}",
            f"mov BYTE PTR [rsp+{base + rng.randrange(16)}], 11h",
            f"jmp {join}",
            f"{label}:",
            f"mov WORD PTR [rsp+{base + rng.randrange(15)}], dx",
            f"{join}:",
        ]

    lines.append("mov rax, rcx")
    for slot in range(rng.choice((4, 5, 6))):
        if slot == 0:
            # At least one reload crosses a boundary between two of the stores above.
            boundary = rng.choice(boundaries)
            width = rng.choice((2, 4, 8))
            at = rng.randrange(max(base, boundary - width + 1), min(boundary - 1, base + 16 - width) + 1)
        else:
            width = rng.choice((1, 2, 4, 8))
            at = base + rng.randrange(16 - width + 1)
        signed = rng.random() < 0.5
        if width == 8:
            lines.append(f"mov r10, QWORD PTR [rsp+{at}]")
        elif width == 4:
            lines.append(f"movsxd r10, DWORD PTR [rsp+{at}]" if signed else f"mov r10d, DWORD PTR [rsp+{at}]")
        else:
            lines.append(f"{'movsx r10' if signed else 'movzx r10d'}, {PTR[width]} [rsp+{at}]")
        lines += [f"mov QWORD PTR [r8+{8 * slot}], r10", rng.choice(("xor rax, r10", "add rax, r10", "sub rax, r10")),
                  "rol rax, 7"]

    for index, register in enumerate(moved):
        lines.append(f"mov {register}, QWORD PTR [rsp+{size + 8 * len(pushed) + 8 * (index + 1)}]")
    lines.append(f"add rsp, {size}")
    lines += [f"pop {register}" for register in reversed(pushed)]
    lines.append("ret")
    return Function(name, lines, prolog)


def generate_frames(seed, count):
    rng = random.Random(seed)
    return [frame_function(f"fuzz_frame_{index}", rng) for index in range(count)]


# Each call writes its callee's output into its own 48-byte segment of the buffer (a frame function
# writes at most six words); the caller stores RAX after each call from byte 192 on.
CALL_SEGMENT = 48
CALL_RESULTS = 4 * CALL_SEGMENT


def clobber_function(name):
    """A callee that overwrites every volatile general-purpose register (RAX, RCX, RDX and R8 to R11)
    with known values derived from its arguments.

    It writes each value to its output segment before the next one replaces it, so the clobbers are
    observed through memory and RAX, the only results a caller may read after the call.
    """
    lines = [
        "lea rax, [rcx+rdx*2]",
        "lea r9, [rcx+11h]",
        "lea r10, [rdx+2222h]",
        "lea r11, [rcx+rdx+333h]",
        "mov QWORD PTR [r8], r9",
        "mov QWORD PTR [r8+8], r10",
        "mov QWORD PTR [r8+16], r11",
        "mov rcx, r10",
        "xor rcx, rax",
        "mov rdx, r11",
        "sub rdx, r9",
        "mov QWORD PTR [r8+24], rcx",
        "mov QWORD PTR [r8+32], rdx",
        "mov r8, rax",
        "ret",
    ]
    return Function(name, lines)


def clobber_read(name, clobber, register):
    """A negative control: a caller that reads a volatile register after the call, which the
    translator must refuse because the callee may have overwritten it."""
    prolog = ["push rbx", ".pushreg rbx", "sub rsp, 32", ".allocstack 32"]
    lines = ["mov rbx, r8", f"call {clobber.name}", f"mov QWORD PTR [rbx+{CALL_RESULTS}], rax",
             f"mov rax, {register}", "add rsp, 32", "pop rbx", "ret"]
    return Function(name, lines, prolog, [clobber], refused=f"undefined register {register}")


def caller_function(name, callees, first, rng, displacements):
    """A fixed RSP frame that makes one to three direct calls and may end with a tail jump.

    Every callee may overwrite the volatile registers, and a frame callee overwrites the nonvolatile
    registers it saves, so the caller keeps its state in three saved registers and defines RCX, RDX
    and R8 again before each call. After a call it reads only RAX: cpu_switch_aot treats the other
    volatile registers and the flags as undefined there. The displacement added for RDX is unique in
    the program, so callers that share a callee pass it different arguments.
    """
    output, left, right = rng.sample(NONVOLATILE, 3)
    # Three pushes and a 32- or 48-byte allocation leave RSP 8 mod 16 from entry with the callee's
    # home space at RSP, below the return address.
    size = rng.choice((32, 48))
    prolog = []
    for register in (output, left, right):
        prolog += [f"push {register}", f".pushreg {register}"]
    prolog += [f"sub rsp, {size}", f".allocstack {size}"]
    lines = [f"mov {output}, r8", f"mov {left}, rcx", f"mov {right}, rdx"]
    # A spill above the callee's home space is kept across the calls.
    spill = size == 48 and rng.random() < 0.5
    if spill:
        lines.append(f"mov QWORD PTR [rsp+32], {right}")

    def arguments(index, after_call):
        displacement = rng.randrange(1, 0x8000)
        while displacement in displacements:
            displacement = rng.randrange(1, 0x8000)
        displacements.add(displacement)
        sources = [left, right, f"{left}+{right}*2"] + (["rax"] if after_call else [])
        source = rng.choice(sources)
        return [f"lea rcx, [{source}]" if "+" in source else f"mov rcx, {source}",
                f"lea rdx, [{rng.choice((left, right, 'rax') if after_call else (left, right))}+{displacement}]",
                f"lea r8, [{output}+{CALL_SEGMENT * index}]" if index else f"mov r8, {output}"]

    targets = []
    for index in range(rng.choice((1, 2, 3))):
        if index == 0:
            target = first
        elif rng.random() < 0.3:
            target = targets[-1]  # the same callee again, with the previous result as an argument
        else:
            target = rng.choice(callees)
        lines += arguments(index, index > 0)
        lines += [f"call {target.name}", f"mov QWORD PTR [{output}+{CALL_RESULTS + 8 * index}], rax",
                  rng.choice((f"xor {left}, rax", f"add {right}, rax", f"sub {left}, rax")), f"rol {left}, 9"]
        targets.append(target)
    if spill:
        lines += ["mov r10, QWORD PTR [rsp+32]", f"xor {left}, r10"]

    tail = rng.random() < 0.5
    if tail:
        target = rng.choice(callees)
        targets.append(target)
        lines += arguments(len(targets) - 1, True)
    else:
        lines += [f"mov rax, {left}", f"xor rax, {right}"]
    lines.append(f"add rsp, {size}")
    lines += [f"pop {register}" for register in (right, left, output)]
    lines.append(f"jmp {target.name}" if tail else "ret")
    return Function(name, lines, prolog, list({target.name: target for target in targets}.values()))


def generate_calls(seed, count):
    """Callers with direct calls and tail jumps to callees from the leaf and frame generators.

    Half as many callees as callers, alternating leaf and frame functions. Each callee is the first
    call of one or two callers, and later calls and tail jumps reach it again.

    The first two callers instead start by calling fuzz_clobber with different arguments, which
    overwrites every volatile general-purpose register with known values. Two controls read R10 and RCX after
    calling it, and the translator must refuse both.
    """
    rng = random.Random(seed)
    pool = max(2, count // 2)
    leaves = rng.sample(generate(seed ^ 0x1EAF, max(20, pool)), (pool + 1) // 2)
    frames = generate_frames(seed ^ 0xF4A3E, pool // 2)
    callees = [function for pair in zip(leaves, frames + [None]) for function in pair if function]
    clobber = clobber_function("fuzz_clobber")
    displacements = set()
    callers = [caller_function(f"fuzz_call_{index}", callees, clobber if index < 2 else callees[index % pool],
                               rng, displacements)
               for index in range(count)]
    controls = [clobber_read(f"fuzz_clobber_read_{register}", clobber, register) for register in ("r10", "rcx")]
    return [clobber, *callees, *callers, *controls]


# Chain shapes: deeper call chains, skipped calls and unexported callees, beyond the direct calls of
# generate_calls. Each exported function is fn(x, y, out) with x in RCX, y in RDX and the output buffer in R8.
# The callees are not exported: they live in a pool that every exported function shares, so one callee
# is reached by several callers, by calls at different depths and by tail jumps. Callee i only calls a
# callee below it, so the pool has no cycles.
CALLEE_KINDS = ("mix", "branch", "narrow", "store", "dword", "mid_registers", "mid_spill")
# Output words 0 to 7 belong to the exported functions. A storing callee writes its own word above them.
CALLEE_STORE_OFFSET = 0x40
MAX_CHAIN_CALLS = 4


def reached(targets):
    """The targets and everything they call, in order of first use, for the failure report."""
    found = {}
    for target in targets:
        for function in (target, *target.callees):
            found.setdefault(function.name, function)
    return list(found.values())


def callee_leaf(name, kind, index, rng):
    imm = rng.choice((1, 5, 0x21, 0x7f))
    count = rng.choice((1, 3, 5, 9))
    if kind == "mix":
        lines = ["lea rax, [rcx+rcx*2]", "add rax, rdx", f"xor rax, {imm}", f"rol rax, {count}", "ret"]
    elif kind == "branch":
        label = f"{name}_below"
        lines = ["cmp rcx, rdx", f"jb {label}", "mov rax, rcx", "sub rax, rdx", f"add rax, {imm}", "ret",
                 f"{label}:", "mov rax, rdx", "sub rax, rcx", f"xor rax, {imm}", "ret"]
    elif kind == "narrow":
        # Only AL is defined at the RET, so every caller must widen it before use.
        lines = ["mov al, cl", "add al, dl", f"xor al, {imm}", "ret"]
    elif kind == "store":
        lines = ["mov rax, rcx", "add rax, rdx", f"mov QWORD PTR [r8+{CALLEE_STORE_OFFSET + 8 * index}], rax",
                 f"xor rax, {imm}", "ret"]
    else:
        # Writing EAX clears the upper half of RAX.
        lines = ["lea eax, [rcx+rdx*4]", f"rol eax, {count}", "ret"]
    return Function(name, lines, narrow=kind == "narrow", export=False)


def call_lines(callee):
    """A call followed by the widening that a byte-sized return needs."""
    return [f"call {callee.name}", *(["movzx eax, al"] if callee.narrow else [])]


def callee_mid(name, kind, pool, rng):
    """A callee with its own frame that calls one or two callees below it and combines the results."""
    targets = [rng.choice(pool) for _ in range(rng.choice((1, 2)))]
    if kind == "mid_registers":
        pushed, size, keep, spill = ["rbx", "rsi"], 0x28, "rbx", None
    else:
        pushed, size, keep, spill = ["rsi"], 0x30, None, "[rsp+20h]"
    prolog = []
    for register in pushed:
        prolog += [f"push {register}", f".pushreg {register}"]
    prolog += [f"sub rsp, {size}", f".allocstack {size}"]
    lines = [f"mov {keep}, rdx" if keep else f"mov QWORD PTR {spill}, rdx", "mov rsi, r8"]
    held = keep or "r10"
    for position, callee in enumerate(targets):
        if position == 0:
            lines.append(rng.choice(("xor rcx, rdx", "lea rcx, [rcx+rdx*2]", "add rcx, rdx")))
        else:
            lines += ["mov rcx, rax", f"mov rdx, {keep}" if keep else f"mov rdx, QWORD PTR {spill}"]
        lines += ["mov r8, rsi", *call_lines(callee)]
        if keep is None:
            lines.append(f"mov r10, QWORD PTR {spill}")
        lines.append(rng.choice((f"add rax, {held}", f"xor rax, {held}", f"sub rax, {held}")))
    lines += [f"add rsp, {size}", *[f"pop {register}" for register in reversed(pushed)], "ret"]
    return Function(name, lines, prolog, reached(targets), export=False)


def callee_pool(rng):
    pool = []
    for index, kind in enumerate(CALLEE_KINDS):
        name = f"fuzz_callee_{index}"
        if kind.startswith("mid"):
            pool.append(callee_mid(name, kind, pool, rng))
        else:
            pool.append(callee_leaf(name, kind, index, rng))
    return pool


def argument_setup(rng, held_x, held_y, first):
    """RCX and RDX for the next call: from the previous result in RAX, or from the saved arguments."""
    if first:
        rcx = rng.choice((f"mov rcx, {held_x}", f"lea rcx, [{held_x}+7]", f"lea rcx, [{held_x}+{held_x}*2]"))
    else:
        rcx = rng.choice(("mov rcx, rax", f"lea rcx, [rax+{held_x}]", "lea rcx, [rax+rax*4]"))
    rdx = [f"mov rdx, {held_y}"]
    if rng.random() < 0.5:
        rdx.append("add rdx, 11")
    return [rcx, *rdx]


def call_chain(name, pool, rng, conditional):
    """Two to four calls from a frame that keeps the output pointer, x and y across them."""
    registers = rng.sample(NONVOLATILE, 3)
    use_registers = rng.random() < 0.6
    pushed = registers if use_registers else registers[:2]
    out, held_x = pushed[0], pushed[1]
    held_y = pushed[2] if use_registers else "QWORD PTR [rsp+20h]"
    size = 0x20 if use_registers else 0x28
    prolog = []
    for register in pushed:
        prolog += [f"push {register}", f".pushreg {register}"]
    prolog += [f"sub rsp, {size}", f".allocstack {size}"]
    lines = [f"mov {out}, r8", f"mov {held_x}, rcx", f"mov {held_y}, rdx"]
    calls = rng.randint(2, MAX_CHAIN_CALLS)
    used = []
    for position in range(calls):
        callee = rng.choice(pool)
        used.append(callee)
        body = [*argument_setup(rng, held_x, held_y, position == 0), f"mov r8, {out}", *call_lines(callee),
                f"mov QWORD PTR [{out}+{8 * position}], rax",
                rng.choice((f"add {held_x}, rax", f"xor {held_x}, rax", f"lea {held_x}, [{held_x}+rax*2]"))]
        if conditional and position > 0 and rng.random() < 0.6:
            skip = f"{name}_skip{position}"
            body = [f"test {part(held_x, 1)}, 1", f"jz {skip}", *body, f"{skip}:"]
        lines += body
    lines += [f"mov rax, {held_x}", f"add rsp, {size}", *[f"pop {register}" for register in reversed(pushed)], "ret"]
    return Function(name, lines, prolog, reached(used))


def tail_function(name, pool, rng, shape):
    """A function that ends in a jump to another function's entry instead of a RET."""
    tails = [callee for callee in pool if not callee.narrow]
    first, second = rng.choice(tails), rng.choice(tails)
    imm = rng.choice((3, 5, 9, 0x10))
    if shape == "plain":
        lines = [f"add rcx, {imm}", f"xor rdx, {imm + 1}", f"jmp {first.name}"]
        return Function(name, lines, callees=reached([first]))
    if shape == "frame":
        # The saved register is popped before the jump, so the callee returns straight to the caller.
        lines = ["mov rbx, rcx", "mov QWORD PTR [r8], rbx", f"lea rcx, [rbx+{imm}]", "xor rdx, rbx", "pop rbx",
                 f"jmp {first.name}"]
        return Function(name, lines, ["push rbx", ".pushreg rbx"], reached([first]))
    odd = f"{name}_odd"
    lines = ["test cl, 1", f"jnz {odd}", f"add rcx, {imm}", f"jmp {first.name}",
             f"{odd}:", "lea rdx, [rdx+rcx*2]", f"jmp {second.name}"]
    return Function(name, lines, callees=reached([first, second]))


def generate_chains(seed, count):
    """Exports that call and jump to a pool of unexported callees that nest two levels deep."""
    rng = random.Random(seed)
    pool = callee_pool(rng)
    recipes = (("chain", False), ("tail", "plain"), ("chain", True), ("tail", "frame"), ("chain", True),
               ("tail", "arms"))
    functions = []
    for index in range(count):
        kind, option = recipes[index % len(recipes)]
        name = f"fuzz_call_{index}"
        functions.append(call_chain(name, pool, rng, option) if kind == "chain" else tail_function(name, pool, rng, option))
    return [*pool, *functions]


# Switch shapes: a jump table behind an unsigned bound check, in the layouts cpu_switch_aot accepts (see
# tests/cpu/switch.asm and tests/cpu/switch_inverted.asm). Each export is fn(selector, y, out): the
# selector in RCX picks an arm that rewrites RDX, every arm writes a marker word of its own, and the common
# tail stores RDX at the start of the output and returns the selector plus RDX. Every combination of table
# layout, bound branch and index source appears once in 48 exports.
SWITCH_COUNTS = (2, 3, 4, 5, 6, 8, 9, 12, 16, 24)
SWITCH_LAYOUTS = ("rva", "rel")
# JA and JAE branch to the default; JBE and JB branch to the dispatch block, which a JMP to the default precedes.
SWITCH_BOUNDS = ("ja", "jae", "jbe", "jb")
SWITCH_SOURCES = ("plain", "wide", "mask", "byte", "word", "floor")
SWITCH_MARKERS = 8
# Selectors 0 to 39 reach every arm, and the default, of the largest table.
SWITCH_SELECTORS = 40
# Negative controls: the shape that the translator must refuse, and the reason it must give.
SWITCH_CONTROLS = {
    "unbounded": "unbounded jump-table index",
    "signed": "jump-table bound is not an unsigned upper limit",
    "unextended": "is not the bounded value",
    "base": "is not a RIP-relative address",
}


def arm_operations(rng, wide, count):
    """Operations on RDX that read only RDX and the selector in RAX, so that no arm depends on the
    registers that the dispatch sequence overwrites."""
    d, a = ("rdx", "rax") if wide else ("edx", "eax")
    imm = lambda: rng.choice((1, 3, 7, 17, 127, 23130))
    shifts = ("shl", "shr", "sar", "rol")
    choices = (lambda: f"add {d}, {imm()}", lambda: f"sub {d}, {imm()}", lambda: f"xor {d}, {imm()}",
               lambda: f"imul {d}, {d}, {rng.choice((3, 5, 7, 11))}",
               lambda: f"{rng.choice(shifts)} {d}, {rng.choice((1, 3, 5))}", lambda: f"neg {d}", lambda: f"not {d}",
               lambda: f"lea {d}, [rdx+rdx*{rng.choice((2, 4, 8))}]",
               lambda: f"lea {d}, [rdx+rax*{rng.choice((2, 4))}+{imm()}]",
               lambda: f"{rng.choice(('add', 'xor', 'sub'))} {d}, {a}")
    return [rng.choice(choices)() for _ in range(count)]


def switch_entries(rng, count):
    """The arm of each table entry and the order the arms are laid out in: every entry its own arm in
    order, runs of entries that share an arm in order, or shared arms with entries and arms shuffled."""
    kind = rng.choice(("distinct", "grouped", "shuffled") if count > 2 else ("distinct", "shuffled"))
    if kind == "distinct":
        return list(range(count)), list(range(count))
    arms = rng.randint(max(2, count // 2), count - 1 if kind == "grouped" else count)
    entries = list(range(arms)) + [rng.randrange(arms) for _ in range(count - arms)]
    layout = list(range(arms))
    if kind == "grouped":
        return sorted(entries), layout
    rng.shuffle(entries)
    rng.shuffle(layout)
    return entries, layout


def switch_function(name, rng, layout, bound, source):
    count = rng.choice(SWITCH_COUNTS)
    wide = rng.random() < 0.3
    d, a, ptr = ("rdx", "rax", "QWORD PTR") if wide else ("edx", "eax", "DWORD PTR")
    entries, order = switch_entries(rng, count)
    arms = max(entries) + 1
    # JA and JBE admit selectors up to the compared limit, JAE and JB stop one short of it.
    limit = count - 1 if bound in ("ja", "jbe") else count
    default, store, table = f"{name}_default", f"{name}_store", f"{name}_table"
    epilogue = [f"mov {ptr} [r8], {d}", f"add {a}, {d}", "ret"]

    def block(label, slot, tail):
        lines = [f"{label}:", f"mov DWORD PTR [r8+{SWITCH_MARKERS + 4 * slot}], {0x100 + slot}",
                 *arm_operations(rng, wide, rng.choice((1, 2)))]
        return lines + {"jump": [f"jmp {store}"], "fall": [], "ret": epilogue, "const": [f"mov {a}, -1", "ret"]}[tail]

    # What the selector becomes before the bound check, the register that is compared, and which
    # 32-bit register the dispatch block copies the index from.
    mask = (1 << count.bit_length()) - 1
    prepare, compared = {
        "plain": (["mov eax, ecx"], "ecx"), "wide": (["mov eax, ecx"], "rcx"),
        "mask": (["mov eax, ecx", f"and eax, {mask}"], "eax"), "byte": (["movzx eax, cl"], "eax"),
        "word": (["movzx eax, cx"], "eax"),
        "floor": (["mov eax, ecx", "xor r11d, r11d", "test eax, eax", "cmovs eax, r11d"], "eax"),
    }[source]
    hoisted = []
    if layout == "rva":
        # MSVC: image-relative entries, loaded through a register that holds the image base.
        index = rng.choice((f"movsxd r9, {'ecx' if source in ('plain', 'wide') else 'eax'}",
                            f"mov r9d, {'ecx' if source in ('plain', 'wide') else 'eax'}"))
        if rng.random() < 0.25:
            hoisted, dispatch = ["lea r10, __ImageBase"], [index]
        else:
            dispatch = rng.sample([index, "lea r10, __ImageBase"], 2)
        dispatch += [f"mov r9d, DWORD PTR [(IMAGEREL {table}) + r10 + r9*4]", "add r9, r10", "jmp r9"]
        table_lines = [f"{table} LABEL DWORD", *(f"DD IMAGEREL {name}_arm{arm}" for arm in entries)]
    else:
        # LLVM: signed offsets from the table, whose address is added back. A 64-bit compare of RCX
        # needs no copy; a 32-bit one zero-extends the compared value into RCX first.
        base = rng.choice(("r9", "r10"))
        dispatch = [*([] if source == "wide" else ["mov ecx, eax"]), f"lea {base}, {table}"]
        rng.shuffle(dispatch)
        dispatch += [f"movsxd rcx, DWORD PTR [{base}+rcx*4]", f"add rcx, {base}", "jmp rcx"]
        table_lines = [f"{table}:", *(f"DD {name}_arm{arm} - {table}" for arm in entries)]
    check = [f"cmp {compared}, {limit}"]
    if bound in ("jbe", "jb"):
        check += [f"{bound} {name}_in_range", f"jmp {default}", f"{name}_in_range:"]
    else:
        check.append(f"{bound} {default}")

    guard = rng.random() < 0.25
    default_tail = rng.choice(("jump", "jump", "ret", "const"))
    default_lines = block(default, arms, default_tail)
    lines = [*hoisted]
    if guard:
        # The default in front of the check: the bound branch goes backward.
        lines += [f"jmp {name}_guard", *default_lines, f"{name}_guard:"]
    lines += [*prepare, *check, *dispatch]
    for position, arm in enumerate(order):
        last = position == len(order) - 1
        tail = "fall" if not last and rng.random() < 0.25 else rng.choice(("jump",) * 4 + ("ret",))
        arm_lines = block(f"{name}_arm{arm}", arm, tail)
        if guard and last and tail == "jump":
            arm_lines.pop()  # falls into the common tail
        lines += arm_lines
    if not guard:
        # The default after the arms; as the last block it falls into the common tail.
        lines += default_lines[:-1] if default_tail == "jump" else default_lines
    return Function(name, [*lines, f"{store}:", *epilogue, "ALIGN 4", *table_lines])


def switch_control(name, kind):
    """A negative control: a two-entry LLVM-layout switch that breaks one rule the translator checks."""
    check = {"unbounded": [], "signed": ["cmp ecx, 1", f"jg {name}_default"]}.get(kind, ["cmp ecx, 1", f"ja {name}_default"])
    copy = [] if kind in ("unbounded", "unextended") else ["mov ecx, eax"]
    base = "mov r9, rdx" if kind == "base" else f"lea r9, {name}_table"
    lines = ["mov eax, ecx", *check, *copy, base, "movsxd rcx, DWORD PTR [r9+rcx*4]", "add rcx, r9", "jmp rcx",
             f"{name}_arm0:", "add edx, 1", f"jmp {name}_store", f"{name}_arm1:", "xor edx, 5", f"jmp {name}_store",
             f"{name}_default:", "not edx", f"{name}_store:", "mov DWORD PTR [r8], edx", "add eax, edx", "ret",
             "ALIGN 4", f"{name}_table:", f"DD {name}_arm0 - {name}_table", f"DD {name}_arm1 - {name}_table"]
    return Function(name, lines, refused=SWITCH_CONTROLS[kind])


def generate_switches(seed, count):
    """Exports that switch on the selector in RCX through a jump table, then four refused controls.

    The first 48 exports take the 48 combinations of layout, bound branch and index source in a
    seeded order; later ones repeat them with other tables.
    """
    rng = random.Random(seed)
    shapes = [(layout, bound, source) for layout in SWITCH_LAYOUTS for bound in SWITCH_BOUNDS
              for source in SWITCH_SOURCES]
    rng.shuffle(shapes)
    functions = [switch_function(f"fuzz_switch_{index}", rng, *shapes[index % len(shapes)]) for index in range(count)]
    return [*functions, *(switch_control(f"fuzz_switch_refused_{kind}", kind) for kind in SWITCH_CONTROLS)]


# Scalar float and double shapes: random leaf functions over the scalar SSE and SSE2 instructions that
# cpu_sse_aot lifts. Each export is fn(x, y, out) with the slots numbered as in the Windows x64 convention: x and
# y are a float (f), a double (d) or a 64-bit integer (q), in XMM0 and XMM1 or in RCX and RDX by slot number, and
# the output buffer is the third slot, in R8. The Function's signature is the descriptor '<ret>:<slots>' that
# names its export for cpu_sse_aot, such as 'd:fdq'. Float and double results are compared bit for bit, with every
# NaN payload, because the translator models those bits; a float result is its low dword.
FP_WIDTH = {"f": 4, "d": 8}
FP_OPERAND = {4: "DWORD PTR", 8: "QWORD PTR"}
FP_REGISTERS = tuple(f"xmm{number}" for number in range(6))
FP_INTEGER_REGISTERS = ("rax", "rcx", "rdx", "r9", "r10", "r11")
FP_ARITHMETIC = ("add", "sub", "mul", "div")
FP_EXTREMA = ("min", "max")
# Only the conditional jumps that cpu_sse_aot accepts after a float compare: the unordered result sets ZF, PF and CF.
FP_CONDITIONS = ("e", "ne", "b", "ae", "be", "a")
# Every argument combination, then the combinations that have at least one floating-point slot.
FP_KINDS = tuple(a + b for a in "fdq" for b in "fdq")
FP_COMPARE_KINDS = tuple(kinds for kinds in FP_KINDS if kinds != "qq")
FP_COUNT_OPERATIONS = (4, 9)
FP_SINGLES = (0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3fc00000, 0x00000001, 0x007fffff, 0x00800000,
              0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc12345, 0x7f800001, 0x4f000000, 0xcf000000,
              0x5f000000)
FP_DOUBLES = (0x0000000000000000, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000, 0x3ff8000000000000,
              0x0000000000000001, 0x000fffffffffffff, 0x0010000000000000, 0x7fefffffffffffff, 0x7ff0000000000000,
              0xfff0000000000000, 0x7ff8000000000000, 0xfff8000012345678, 0x7ff0000000000001, 0x41e0000000000000,
              0xc1e0000000000000, 0x43e0000000000000)
FP_INTEGERS = (0, 1, 2, 3, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x100000000,
               0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFF80000000, (1 << 53) + 1, 0x1000001, 0xDEADBEEF,
               0x12345678, 0xA5A5A5A5A5A5A5A5)
# Values around the edges of the conversions that the full cross of the lists above leaves out.
FP_EXTRA = {"f": (0x4effffff, 0xcf000001, 0x5effffff, 0xdf000000, 0x3dcccccd, 0x33800000, 0x4b800001, 0xffa00000,
                  0xff7fffff, 0x807fffff),
            "d": (0x41dfffffffe00000, 0xc1e0000000200000, 0x43dfffffffffffff, 0xc3e0000000000000, 0x3fb999999999999a,
                  0x800fffffffffffff, 0xffefffffffffffff, 0xfff7ffffffffffff),
            "q": (0x100000001, 0xFFFFFFFE, 0x7FFFFFFE, 0x8000000000000001)}
FP_POOLS = {"f": FP_SINGLES, "d": FP_DOUBLES, "q": FP_INTEGERS}
FP_JUNK = 0xDEADBEEF << 32
# Each profile: the argument combinations, the result kinds and the weight of every operation.
FP_PROFILES = (
    ("single", ("ff",), "ffq", {"arith4": 6, "extrema4": 2, "root4": 2, "copy": 2, "store": 1, "load": 1, "zero": 1,
                                "compare": 1}, ()),
    ("double", ("dd",), "ddq", {"arith8": 6, "extrema8": 2, "root8": 2, "copy": 2, "store": 1, "load": 1, "zero": 1,
                                "compare": 1}, ()),
    ("extrema", ("ff", "dd", "fd", "df"), "fdq", {"extrema4": 3, "extrema8": 3, "root4": 1, "root8": 1, "arith4": 1,
                                                  "arith8": 1, "resize": 2, "copy": 1, "compare": 1}, ("extrema4",)),
    ("compare", FP_COMPARE_KINDS, "fdq", {"compare": 4, "arith4": 1, "arith8": 1, "resize": 1, "constant": 1,
                                          "to_int": 1}, ("compare",)),
    ("convert", FP_KINDS, "fdq", {"from_int": 4, "to_int": 4, "resize": 3, "constant": 2, "movd": 2, "copy": 1,
                                  "arith4": 1, "arith8": 1, "root4": 1, "root8": 1, "spill": 1, "compare": 1},
     ("from_int", "to_int")),
    ("mixed", FP_KINDS, "fdq", {"arith4": 2, "arith8": 2, "extrema4": 1, "extrema8": 1, "root4": 1, "root8": 1,
                                "copy": 2, "load": 1, "store": 1, "zero": 1, "resize": 2, "from_int": 2, "to_int": 2,
                                "constant": 1, "movd": 1, "spill": 1, "compare": 2}, ()),
)
# Negative controls: the shape that the translator must refuse, the signature and the reason it must give.
FP_CONTROLS = {
    "setcc": ("q:ffq", ["ucomiss xmm0, xmm1", "seta al", "movzx eax, al", "ret"], "unsupported flag producer"),
    "condition": ("q:ffq", ["ucomiss xmm0, xmm1", "jle {name}_le", "mov eax, 1", "ret", "{name}_le:", "xor eax, eax",
                            "ret"], "unsupported condition after a float compare"),
    "unwritten": ("f:ffq", ["addss xmm0, xmm2", "ret"], "undefined register xmm2"),
    "narrow": ("d:ffq", ["addsd xmm0, xmm1", "ret"], "undefined register xmm0[63:32]"),
}


class FpBuilder:
    """Emits one scalar float function and follows which low bytes of each XMM register and which general
    registers hold a defined value, so that it reads only what cpu_sse_aot accepts: the lifter refuses a read
    of a register that is undefined on some path, and the upper dword of a float argument is not part of it."""

    def __init__(self, name, rng, kinds, result, weights):
        self.name, self.rng, self.kinds, self.result, self.weights = name, rng, kinds, result, weights
        self.lines = []
        self.labels = 0
        self.defined = {register: 0 for register in FP_REGISTERS}
        self.ints = set()
        for number, kind in enumerate(kinds):
            if kind == "q":
                self.ints.add(("rcx", "rdx")[number])
            else:
                self.defined[f"xmm{number}"] = FP_WIDTH[kind]

    def emit(self, line):
        self.lines.append(line)

    def state(self):
        return dict(self.defined), set(self.ints)

    def restore(self, state):
        self.defined, self.ints = dict(state[0]), set(state[1])

    def live(self, least):
        return [register for register in FP_REGISTERS if self.defined[register] >= least]

    def memory(self, size, low=0, high=0xE0):
        """An operand in the output buffer, at a multiple of 4 that also straddles earlier stores."""
        return f"{FP_OPERAND[size]} [r8+{self.rng.randrange(low, high - size + 1, 4)}]"

    def source(self, width):
        """A register that holds at least width bytes, or a memory operand of that size. Memory often reads
        the arguments that the function stored first, so edge values reach it without a register."""
        registers = self.live(width)
        if registers and self.rng.random() < 0.65:
            return self.rng.choice(registers)
        if self.rng.random() < 0.5:
            return f"{FP_OPERAND[width]} [r8+{self.rng.choice((0, 8) if width == 8 else (0, 4, 8, 12))}]"
        return self.memory(width)

    def store_arguments(self):
        for number, kind in enumerate(self.kinds):
            target = f"[r8+{8 * number}]"
            if kind == "q":
                self.emit(f"mov QWORD PTR {target}, {('rcx', 'rdx')[number]}")
            else:
                self.emit(f"{'movss' if kind == 'f' else 'movsd'} {FP_OPERAND[FP_WIDTH[kind]]} {target}, xmm{number}")

    def arithmetic(self, width, names):
        destinations = self.live(width)
        if not destinations:
            return False
        suffix = "ss" if width == 4 else "sd"
        self.emit(f"{self.rng.choice(names)}{suffix} {self.rng.choice(destinations)}, {self.source(width)}")
        return True

    def root(self, width):
        destination = self.rng.choice(FP_REGISTERS)
        self.emit(f"sqrt{'ss' if width == 4 else 'sd'} {destination}, {self.source(width)}")
        self.defined[destination] = max(self.defined[destination], width)
        return True

    def copy(self):
        form = self.rng.choice(("movaps", "movss", "movsd", "movq"))
        sources = self.live(4 if form in ("movaps", "movss") else 8)
        if not sources:
            return False
        source, destination = self.rng.choice(sources), self.rng.choice(FP_REGISTERS)
        self.emit(f"{form} {destination}, {source}")
        # A register copy keeps what is defined; MOVSS between registers writes the low dword only.
        written = {"movaps": self.defined[source], "movss": max(self.defined[destination], 4)}.get(form, 8)
        self.defined[destination] = written
        return True

    def load(self):
        form, width = self.rng.choice((("movss", 4), ("movsd", 8), ("movq", 8)))
        destination = self.rng.choice(FP_REGISTERS)
        self.emit(f"{form} {destination}, {self.memory(width)}")
        self.defined[destination] = 8  # a load clears the rest of the register
        return True

    def store(self):
        form, width = self.rng.choice((("movss", 4), ("movsd", 8), ("movd", 4), ("movq", 8)))
        sources = self.live(width)
        if not sources:
            return False
        self.emit(f"{form} {self.memory(width, 0x10, 0x80)}, {self.rng.choice(sources)}")
        return True

    def zero(self):
        destination = self.rng.choice(FP_REGISTERS)
        self.emit(f"{self.rng.choice(('xorps', 'xorpd', 'pxor'))} {destination}, {destination}")
        self.defined[destination] = 8
        return True

    def resize(self):
        narrow = self.rng.random() < 0.5
        destination = self.rng.choice(FP_REGISTERS)
        width = 8 if narrow else 4
        self.emit(f"{'cvtsd2ss' if narrow else 'cvtss2sd'} {destination}, {self.source(width)}")
        self.defined[destination] = max(self.defined[destination], 12 - width)
        return True

    def from_integer(self):
        suffix, size = self.rng.choice(("ss", "sd")), self.rng.choice((4, 8))
        if self.ints and self.rng.random() < 0.7:
            source = part(self.rng.choice(sorted(self.ints)), size)
        else:
            source = self.memory(size)
        destination = self.rng.choice(FP_REGISTERS)
        self.emit(f"cvtsi2{suffix} {destination}, {source}")
        self.defined[destination] = max(self.defined[destination], 4 if suffix == "ss" else 8)
        return True

    def to_integer(self):
        form, width = self.rng.choice((("cvttss2si", 4), ("cvttsd2si", 8)))
        sources = self.live(width)
        source = self.rng.choice(sources) if sources and self.rng.random() < 0.65 else self.memory(width)
        destination = self.rng.choice(FP_INTEGER_REGISTERS)
        self.emit(f"{form} {part(destination, self.rng.choice((4, 8)))}, {source}")
        self.ints.add(destination)
        return True

    def constant(self):
        destination = self.rng.choice(FP_INTEGER_REGISTERS)
        value = self.rng.choice((0, 1, 5, 0x7FFFFFFF, -0x80000000, -1, 0x12345678))
        self.emit(f"mov {part(destination, 4)}, {value}")
        self.ints.add(destination)
        return True

    def spill(self):
        if not self.ints:
            return False
        size = self.rng.choice((4, 8))
        register = part(self.rng.choice(sorted(self.ints)), size)
        self.emit(f"mov {self.memory(size, 0x10, 0x80)}, {register}")
        return True

    def movd(self):
        if self.ints and self.rng.random() < 0.5:
            destination = self.rng.choice(FP_REGISTERS)
            self.emit(f"movd {destination}, {part(self.rng.choice(sorted(self.ints)), 4)}")
            self.defined[destination] = 8
            return True
        sources = self.live(4)
        if not sources:
            return False
        destination = self.rng.choice(FP_INTEGER_REGISTERS)
        self.emit(f"movd {part(destination, 4)}, {self.rng.choice(sources)}")
        self.ints.add(destination)
        return True

    def compare(self):
        """A float compare, one or two conditional jumps and an arm for each outcome, joined afterwards. A
        value that only some arms define is undefined after the join."""
        width = self.rng.choice((4, 8))
        registers = self.live(width)
        if not registers:
            return False
        kind = f"{self.rng.choice(('ucomis', 'comis'))}{'s' if width == 4 else 'd'}"
        self.emit(f"{kind} {self.rng.choice(registers)}, {self.source(width)}")
        self.labels += 1
        stem = f"{self.name}_{self.labels}"
        taken, unordered, join = f"{stem}_t", f"{stem}_u", f"{stem}_j"
        if self.rng.random() < 0.4:
            # The idiom of a compiler's equality test: the parity jump first, then the ordered condition.
            self.emit(f"j{self.rng.choice(('p', 'np'))} {unordered}")
            self.emit(f"j{self.rng.choice(FP_CONDITIONS)} {taken}")
            arms = [None, unordered, taken]
        else:
            self.emit(f"j{self.rng.choice(FP_CONDITIONS + ('p', 'np'))} {taken}")
            arms = [None, taken]
        start, outcomes = self.state(), []
        for position, label in enumerate(arms):
            self.restore(start)
            if label:
                self.emit(f"{label}:")
            for _ in range(self.rng.randint(1, 3)):
                self.straight()
            if self.result == "q":
                self.emit(f"mov eax, {self.rng.randrange(1, 1000)}")
                self.ints.add("rax")
            outcomes.append(self.state())
            if position < len(arms) - 1:
                self.emit(f"jmp {join}")
        self.emit(f"{join}:")
        self.defined = {register: min(state[0][register] for state in outcomes) for register in FP_REGISTERS}
        self.ints = set.intersection(*(state[1] for state in outcomes))
        return True

    def operation(self, name):
        table = {"arith4": lambda: self.arithmetic(4, FP_ARITHMETIC), "arith8": lambda: self.arithmetic(8, FP_ARITHMETIC),
                 "extrema4": lambda: self.arithmetic(4, FP_EXTREMA), "extrema8": lambda: self.arithmetic(8, FP_EXTREMA),
                 "root4": lambda: self.root(4), "root8": lambda: self.root(8), "copy": self.copy, "load": self.load,
                 "store": self.store, "zero": self.zero, "resize": self.resize, "from_int": self.from_integer,
                 "to_int": self.to_integer, "constant": self.constant, "spill": self.spill, "movd": self.movd,
                 "compare": self.compare}
        return table[name]()

    def straight(self):
        """One operation that is not a compare, tried again with another choice when it has no operands."""
        names = [name for name in self.weights if name != "compare"]
        for _ in range(12):
            if self.operation(self.rng.choices(names, [self.weights[name] for name in names])[0]):
                return

    def finish(self):
        """Show every live register in the buffer, then put the result in XMM0 or RAX."""
        rng = self.rng
        for slot, register in enumerate(FP_REGISTERS):
            if self.defined[register] and rng.random() < 0.85:
                form, width = ("movsd", 8) if self.defined[register] == 8 and rng.random() < 0.7 else ("movss", 4)
                self.emit(f"{form} {FP_OPERAND[width]} [r8+{0x80 + 8 * slot}], {register}")
        for slot, register in enumerate(sorted(self.ints)):
            self.emit(f"mov QWORD PTR [r8+{0xC0 + 8 * slot}], {register}")
        if self.result == "q":
            if "rax" not in self.ints or (len(self.ints) > 1 and rng.random() < 0.4):
                others = sorted(self.ints - {"rax"})
                self.emit(f"mov rax, {rng.choice(others)}" if others else f"mov eax, {rng.randrange(1, 1000)}")
        else:
            width = FP_WIDTH[self.result]
            sources = self.live(width)
            if not sources:
                self.emit(f"{'movss' if width == 4 else 'movsd'} xmm0, {self.memory(width)}")
            elif (source := rng.choice(sources)) != "xmm0":
                self.emit(f"{rng.choice(('movaps', 'movss' if width == 4 else 'movsd'))} xmm0, {source}")
        self.emit("ret")


def scalar_fp_function(name, rng, profile):
    _, kinds, results, weights, required = profile
    kinds, result = rng.choice(kinds), rng.choice(results)
    builder = FpBuilder(name, rng, kinds, result, weights)
    builder.store_arguments()
    for operation in required:
        builder.operation(operation)
    for _ in range(rng.randint(*FP_COUNT_OPERATIONS)):
        for _ in range(12):
            if builder.operation(rng.choices(list(weights), list(weights.values()))[0]):
                break
    builder.finish()
    return Function(name, builder.lines, signature=f"{result}:{kinds}q")


def scalar_fp_control(name, kind):
    """A negative control: a function with one construct that cpu_sse_aot must refuse."""
    signature, lines, reason = FP_CONTROLS[kind]
    return Function(name, [line.format(name=name) for line in lines], signature=signature, refused=reason)


def generate_scalar_fp(seed, count):
    """Exports over scalar float and double arithmetic, then four refused controls.

    The exports cycle through six profiles (float arithmetic, double arithmetic, MIN and MAX, compares with
    jumps, conversions, and a mixture), each with its own argument kinds and operation weights.
    """
    rng = random.Random(seed)
    functions = [scalar_fp_function(f"fuzz_fp_{index}", rng, FP_PROFILES[index % len(FP_PROFILES)])
                 for index in range(count)]
    return [*functions, *(scalar_fp_control(f"fuzz_fp_refused_{kind}", kind) for kind in FP_CONTROLS)]


def fp_inputs(seed, kinds):
    """313 pairs for the first two slots of kinds: every combination of 17 edge values (zeros, ones, the
    smallest and largest denormals and normals, infinities, quiet and signalling NaNs with payloads, and the
    values at the integer conversion limits), then 24 pairs of extra edges and random bits."""
    first, second = FP_POOLS[kinds[0]], FP_POOLS[kinds[1]]
    values = [(a, b) for a in first for b in second]
    rng = random.Random(seed ^ 0xF10A7)
    while len(values) < 313:
        pair = []
        for kind in kinds:
            pool = FP_POOLS[kind] + FP_EXTRA[kind]
            pair.append(rng.choice(pool) if rng.random() < 0.6 else rng.getrandbits(32 if kind == "f" else 64))
        values.append(tuple(pair))
    return values


# Mid-body RSP changes: functions whose RSP moves after the prologue, by SUB and ADD RSP, by PUSH and POP of a
# saved register that holds data, differently in the two arms of a branch, around a direct call, and below an RBP
# anchor that LEAVE or POP RBP releases. Each export is fn(x, y, out) with x in RCX, y in RDX and the output
# buffer in R8. cpu_switch_aot gives every [RSP+d] and [RBP+d] operand a constant index into one frame array,
# so a value stored at one depth and read back at another, through either base, must reach the same bytes.
RSP_SHAPES = ("adjust", "push", "branch", "call", "anchor_leave", "anchor_pop")
RSP_STEPS = (8, 16, 24, 32, 40, 48)
# RSP stays within 256 bytes below entry RSP.
RSP_DEEPEST = -0x100
# Output words 0 to 19 take reloaded values and call results, the words from RSP_REGISTERS on the saved
# registers' final values, and each call's callee writes its own word from RSP_CALLEE on.
RSP_LOADS = 20
RSP_MIN_LOADS = 4
RSP_REGISTERS = 0xA0
RSP_CALLEE = 0xC0
# Each shape: the weight of every body operation, and the operations that it makes at least once.
RSP_WEIGHTS = {
    "adjust": ({"store": 3, "load": 3, "rmw": 1, "grow": 2, "shrink": 2}, ()),
    "push": ({"store": 2, "load": 3, "push": 3, "pop": 3, "grow": 1, "shrink": 1}, ("push",)),
    "branch": ({"store": 2, "load": 2, "rmw": 1, "grow": 1, "shrink": 1, "push": 1, "pop": 1}, ("branch",)),
    "call": ({"store": 2, "load": 2, "grow": 1, "shrink": 1}, ("call",)),
    "anchor_leave": ({"store": 3, "load": 3, "rmw": 1, "grow": 2, "shrink": 1, "push": 1, "pop": 1}, ("grow",)),
    "anchor_pop": ({"store": 3, "load": 3, "rmw": 1, "grow": 2, "shrink": 1, "push": 1, "pop": 1}, ("grow",)),
}
# Negative controls: a body after a 16-byte allocation that the translator must refuse, and the reason it must give.
RSP_CONTROLS = {
    "released": (["sub rsp, 16", "mov QWORD PTR [rsp], rcx", "add rsp, 16", "sub rsp, 16", "mov rax, QWORD PTR [rsp]",
                  "add rsp, 16"], "undefined stack byte"),
    "join": (["test cl, 1", "jz {name}_join", "sub rsp, 8", "{name}_join:", "mov QWORD PTR [rsp], rdx", "mov rax, rdx"],
             "stack depth differs at join"),
    "below": (["sub rsp, 16", "mov QWORD PTR [rsp], rcx", "add rsp, 16", "mov rax, QWORD PTR [rsp-8]"],
              "stack access below RSP"),
    "unaligned": (["sub rsp, 4", "mov DWORD PTR [rsp], ecx", "add rsp, 4", "mov rax, rcx"],
                  "unsupported stack adjustment"),
}


class RspBuilder:
    """Emits one function whose RSP moves after the prologue, and follows which stack bytes hold a defined
    value, so that it reads only what cpu_switch_aot accepts: a read needs each byte written on every path, and
    the bytes that ADD RSP and POP leave below RSP, and the callee's home space after a call, are undefined.

    Offsets are from entry RSP. The prologue pushes one to three saved registers (and RBP for an anchored
    shape) and allocates 16 to 48 bytes; the body may use that allocation, the caller's home space and what it
    allocates itself, but never the pushed slots or the return address. Every saved register is overwritten
    from the arguments before it is read, and the body pushes and pops only those registers."""

    def __init__(self, name, rng, shape, callees):
        self.name, self.rng, self.shape, self.callees = name, rng, shape, callees
        self.prolog, self.lines, self.used = [], [], []
        self.saved = rng.sample(NONVOLATILE, 3 if shape == "call" else rng.choice((1, 2, 3)))
        for register in self.saved:
            self.prolog += [f"push {register}", f".pushreg {register}"]
        pushed = 8 * len(self.saved)
        self.anchor = None
        if shape.startswith("anchor"):
            pushed += 8
            self.anchor = -pushed
            self.prolog += ["push rbp", ".pushreg rbp", "mov rbp, rsp", ".setframe rbp, 0"]
        self.size = rng.choice((16, 24, 32, 40, 48))
        self.prolog += [f"sub rsp, {self.size}", f".allocstack {self.size}"]
        # RSP after the prologue; the body never releases bytes above it.
        self.floor = self.depth = -pushed - self.size
        self.blocked = set(range(-pushed, 8))
        self.defined, self.stored_at = set(), {}
        # A caller keeps its output pointer in a saved register, because the call overwrites R8.
        self.out = self.saved[0] if shape == "call" else "r8"
        self.data = [register for register in self.saved if register != self.out]
        self.arguments = True  # RCX and RDX still hold x and y
        self.slots = self.labels = self.calls = self.loads = 0
        self.cross = 0  # reloads of bytes stored at another RSP

    def emit(self, line):
        self.lines.append(line)

    def state(self):
        return self.depth, set(self.defined), dict(self.stored_at)

    def restore(self, state):
        self.depth, self.defined, self.stored_at = state[0], set(state[1]), dict(state[2])

    def operand(self, offset, width):
        """[RSP+d] at the current RSP or, in an anchored function, often [RBP+d] for the same bytes."""
        if self.anchor is not None and self.rng.random() < 0.5:
            return f"{PTR[width]} [rbp{offset - self.anchor:+d}]"
        return f"{PTR[width]} [rsp+{offset - self.depth}]"

    def sources(self):
        return [*self.data, *(("rcx", "rdx") if self.arguments else ()), "rax"]

    def release(self, low, high):
        for offset in range(low, high):
            self.defined.discard(offset)
            self.stored_at.pop(offset, None)

    def spots(self, width, defined):
        """Offsets where width bytes lie above RSP and below the end of the home space, outside the pushed
        slots and the return address, and are all defined when defined is set."""
        found = []
        for offset in range(self.depth, 40 - width + 1):
            span = range(offset, offset + width)
            if any(byte in self.blocked for byte in span):
                continue
            if defined and not all(byte in self.defined for byte in span):
                continue
            found.append(offset)
        return found

    def store(self):
        width = self.rng.choice((1, 2, 4, 8))
        spots = self.spots(width, False)
        if not spots:
            return False
        # Mostly near RSP, where a later ADD RSP or POP releases the bytes and a deeper RSP reads them back.
        near = [offset for offset in spots if offset < self.depth + 24]
        at = self.rng.choice(near if near and self.rng.random() < 0.6 else spots)
        if self.rng.random() < 0.2:
            source = self.rng.choice((0x5A, 0x7F, 0x81, 0x33))
        else:
            source = part(self.rng.choice(self.sources()), width)
        self.emit(f"mov {self.operand(at, width)}, {source}")
        for byte in range(at, at + width):
            self.defined.add(byte)
            self.stored_at[byte] = self.depth
        return True

    def load(self):
        # Room is kept for the forced cross-depth reload and two call results.
        if self.slots >= RSP_LOADS - 3:
            return False
        width = self.rng.choice((1, 2, 4, 8))
        spots = self.spots(width, True)
        if not spots:
            return False
        cross = [offset for offset in spots
                 if any(self.stored_at[byte] != self.depth for byte in range(offset, offset + width))]
        at = self.rng.choice(cross if cross and self.rng.random() < 0.8 else spots)
        self.cross += at in cross
        self.loads += 1
        operand, signed = self.operand(at, width), self.rng.random() < 0.5
        if width == 8:
            self.emit(f"mov r10, {operand}")
        elif width == 4:
            self.emit(f"movsxd r10, {operand}" if signed else f"mov r10d, {operand}")
        else:
            self.emit(f"{'movsx r10' if signed else 'movzx r10d'}, {operand}")
        self.result("r10")
        return True

    def result(self, register):
        self.emit(f"mov QWORD PTR [{self.out}+{8 * self.slots}], {register}")
        self.slots += 1
        if register != "rax":
            self.emit(self.rng.choice((f"xor rax, {register}", f"add rax, {register}", f"sub rax, {register}")))
        self.emit("rol rax, 7")

    def rmw(self):
        width = self.rng.choice((1, 2, 4, 8))
        spots = self.spots(width, True)
        if not spots:
            return False
        at = self.rng.choice(spots)
        source = part(self.rng.choice(self.sources()), width)
        self.emit(f"{self.rng.choice(('add', 'xor', 'sub'))} {self.operand(at, width)}, {source}")
        for byte in range(at, at + width):
            self.stored_at[byte] = self.depth
        return True

    def grow(self, steps=RSP_STEPS):
        steps = [step for step in steps if self.depth - step >= RSP_DEEPEST]
        if not steps:
            return False
        step = self.rng.choice(steps)
        self.emit(f"sub rsp, {step}")
        self.depth -= step
        return True

    def shrink(self):
        room = self.floor - self.depth
        if not room:
            return False
        step = room if self.rng.random() < 0.4 else self.rng.choice(range(8, room + 1, 8))
        self.emit(f"add rsp, {step}")
        self.release(self.depth, self.depth + step)
        self.depth += step
        return True

    def push(self):
        if not self.data or self.depth - 8 < RSP_DEEPEST:
            return False
        self.emit(f"push {self.rng.choice(self.data)}")
        self.depth -= 8
        for byte in range(self.depth, self.depth + 8):
            self.defined.add(byte)
            self.stored_at[byte] = self.depth
        return True

    def pop(self):
        if (not self.data or self.depth >= self.floor
                or not all(byte in self.defined for byte in range(self.depth, self.depth + 8))):
            return False
        self.cross += any(self.stored_at[byte] != self.depth for byte in range(self.depth, self.depth + 8))
        self.emit(f"pop {self.rng.choice(self.data)}")
        self.release(self.depth, self.depth + 8)
        self.depth += 8
        return True

    def straight(self):
        """A store, a reload or an in-place update at the current RSP."""
        for _ in range(12):
            if getattr(self, self.rng.choice(("store", "store", "load", "load", "rmw")))():
                return

    def branch(self):
        """A TEST and a conditional jump on a saved register, and two arms that move RSP by the same amount in
        different ways (SUB RSP or PUSH, with or without a balanced excursion), so the join sees one depth.
        A byte that only one arm defines is undefined after the join."""
        if not self.data:
            return False
        drops = [drop for drop in (0, 8, 16, 24) if self.depth - drop - 48 >= RSP_DEEPEST]
        if not drops:
            return False
        self.labels += 1
        other, join = f"{self.name}_other{self.labels}", f"{self.name}_join{self.labels}"
        self.emit(f"test {part(self.rng.choice(self.data), 1)}, {self.rng.choice((1, 2, 4))}")
        self.emit(f"j{self.rng.choice(('z', 'nz'))} {other}")
        drop, start, outcomes = self.rng.choice(drops), self.state(), []
        for arm in (0, 1):
            self.restore(start)
            if arm:
                self.emit(f"{other}:")
            target = self.depth - drop
            self.straight()
            while self.depth > target:
                if self.depth - 8 == target or self.rng.random() < 0.4:
                    self.push()
                else:
                    self.grow([step for step in RSP_STEPS if self.depth - step >= target])
                if self.rng.random() < 0.6:
                    self.straight()
            if self.rng.random() < 0.3:
                # An excursion that the arm releases again before the join.
                step = self.rng.choice((8, 16, 32))
                self.emit(f"sub rsp, {step}")
                self.depth -= step
                self.straight()
                self.straight()
                self.emit(f"add rsp, {step}")
                self.release(self.depth, self.depth + step)
                self.depth += step
            outcomes.append(self.state())
            if not arm:
                self.emit(f"jmp {join}")
        self.emit(f"{join}:")
        self.defined = outcomes[0][1] & outcomes[1][1]
        self.stored_at = {byte: min(outcome[2][byte] for outcome in outcomes) for byte in self.defined}
        return True

    def call(self):
        """A mid-body SUB RSP that leaves RSP 16-byte aligned with room for the callee's home space, a direct
        call, and the callee's result in RAX. After the call the volatile registers and the 32 bytes at RSP are
        undefined; the rest of the frame is kept."""
        first = (self.depth - 8) % 16
        step = first + 16 * self.rng.choice((1, 2) if not first else (0, 1, 2))
        if self.depth - step < RSP_DEEPEST:
            return False
        self.emit(f"sub rsp, {step}")
        self.depth -= step
        callee = self.rng.choice(self.callees)
        if callee not in self.used:
            self.used.append(callee)
        x, y = self.rng.choice((*self.data, "rax")), self.rng.choice(self.data)
        self.emit(f"mov rcx, {x}")
        self.emit(f"lea rdx, [{y}+{self.rng.randrange(1, 0x8000)}]")
        self.emit(f"lea r8, [{self.out}+{RSP_CALLEE + 8 * self.calls}]")
        self.emit(f"call {callee.name}")
        self.calls += 1
        self.arguments = False
        self.release(self.depth, self.depth + 32)
        self.result("rax")
        return True

    def operation(self, name):
        return getattr(self, name)()

    def cross_depth(self):
        """A store at the current RSP, a move of RSP and a reload of those bytes at the new RSP."""
        if self.depth - 8 < RSP_DEEPEST:
            self.emit(f"add rsp, {self.floor - self.depth}")
            self.release(self.depth, self.floor)
            self.depth = self.floor
        width = self.rng.choice((2, 4, 8))
        at = self.rng.choice(self.spots(width, False))
        self.emit(f"mov {self.operand(at, width)}, {part(self.rng.choice(self.sources()), width)}")
        for byte in range(at, at + width):
            self.defined.add(byte)
            self.stored_at[byte] = self.depth
        self.grow([step for step in RSP_STEPS if self.depth - step >= RSP_DEEPEST] or [8])
        self.cross += 1
        self.loads += 1
        operand = self.operand(at, width)
        self.emit(f"mov r10, {operand}" if width == 8 else f"movzx r10d, {operand}" if width == 2
                  else f"mov r10d, {operand}")
        self.result("r10")

    def build(self):
        rng = self.rng
        if self.out != "r8":
            self.emit(f"mov {self.out}, r8")
        for register in self.data:
            self.emit(rng.choice((f"mov {register}, rcx", f"mov {register}, rdx", f"lea {register}, [rcx+rdx*2]",
                                  f"lea {register}, [rdx+rcx*4+7]")))
            self.emit(rng.choice((f"xor {register}, rdx", f"rol {register}, 13", f"add {register}, rcx")))
        self.emit("mov rax, rcx")
        for _ in range(rng.randint(2, 4)):
            self.store()
        weights, required = RSP_WEIGHTS[self.shape]
        names = list(weights)
        operations = [rng.choices(names, [weights[name] for name in names])[0] for _ in range(rng.randint(6, 12))]
        for name in required * rng.choice((1, 2)):
            operations.insert(rng.randrange(len(operations) + 1), name)
        for name in operations:
            if not self.operation(name):
                for _ in range(12):
                    if self.operation(rng.choices(names, [weights[name] for name in names])[0]):
                        break
        if not self.cross:
            self.cross_depth()
        for _ in range(48):
            if self.loads >= RSP_MIN_LOADS:
                break
            # More reloads at the final RSP, after a store when nothing is defined.
            self.load() or self.store()
        for slot, register in enumerate(self.data):
            self.emit(f"mov QWORD PTR [{self.out}+{RSP_REGISTERS + 8 * slot}], {register}")
        if self.shape == "anchor_leave":
            # LEAVE restores RSP from the anchor, so the body may end below the prologue's RSP.
            self.emit("leave")
        else:
            while self.depth < self.floor:
                if not (rng.random() < 0.4 and self.pop()):
                    self.shrink()
            self.emit(f"add rsp, {self.size}")
            if self.anchor is not None:
                self.emit("pop rbp")
        self.lines += [f"pop {register}" for register in reversed(self.saved)]
        self.emit("ret")
        return Function(self.name, self.lines, self.prolog, self.used)


def rsp_callee(name, index):
    """A frameless callee that only the generated callers reach: it writes one word through R8 and returns a
    value that is not symmetric in RCX and RDX."""
    if index == 0:
        lines = ["lea rax, [rcx+rdx*2]", "xor rax, 5A5Ah", "mov QWORD PTR [r8], rax", "rol rax, 5", "ret"]
    else:
        lines = ["mov rax, rdx", "sub rax, rcx", "mov DWORD PTR [r8+4], eax", "imul rax, rax, 7", "ret"]
    return Function(name, lines, export=False)


def rsp_control(name, kind):
    """A negative control: a mid-body RSP change that the translator must refuse."""
    body, reason = RSP_CONTROLS[kind]
    lines = [*(line.format(name=name) for line in body), "mov QWORD PTR [r8], rax", "add rsp, 16", "ret"]
    return Function(name, lines, ["sub rsp, 16", ".allocstack 16"], refused=reason)


def generate_rsp(seed, count):
    """Exports whose RSP moves after the prologue, cycling through six shapes, then four refused controls.

    Two frameless callees, which no export names, serve the calling shape."""
    rng = random.Random(seed)
    callees = [rsp_callee(f"fuzz_rsp_callee_{index}", index) for index in range(2)]
    functions = [RspBuilder(f"fuzz_rsp_{index}", rng, RSP_SHAPES[index % len(RSP_SHAPES)], callees).build()
                 for index in range(count)]
    return [*callees, *functions, *(rsp_control(f"fuzz_rsp_refused_{kind}", kind) for kind in RSP_CONTROLS)]


# Counted loops: leaf functions, some with pushed registers, whose loops end in a backward conditional branch.
# Every trip count comes from x masked to 4 bits (3 bits for a nested loop) and the counter is written only by
# the loop's own update, so each input terminates within 16 iterations, or 8 times 8 when nested. The output
# buffer keeps the final carried registers at 0 to 23, a value carried through memory at 24, an iteration count
# at 32, the final counter (the count in the up form) at 36, an early-exit record at 40 to 55, the first saved
# register at 56, one word per outer iteration from 64 to 191 and one word per inner iteration from 192 to 255.
LOOP_FORMS = ("dec", "up", "guarded", "step")
LOOP_CARRIED = ("rax", "rdx", "r10")
LOOP_SAVED = ("rbx", "rsi", "rdi")
LOOP_MEMORY, LOOP_COUNT, LOOP_FINAL, LOOP_EARLY, LOOP_SAVES, LOOP_TRACE, LOOP_INNER = 24, 32, 36, 40, 56, 64, 192


def loop_operation(rng, target, sources):
    """One update of a carried register from itself and the other carried values."""
    source = rng.choice(sources)
    dword = rng.random() < 0.2
    a, b = (REGISTER_PARTS[target][0], REGISTER_PARTS[source][0]) if dword else (target, source)
    return rng.choice((f"add {a}, {b}", f"sub {a}, {b}", f"xor {a}, {b}", f"imul {a}, {b}, {rng.choice((3, 5, 7, 31))}",
                       f"rol {target}, {rng.choice((5, 13, 29))}", f"ror {target}, {rng.choice((3, 11))}",
                       f"lea {target}, [{target}+{source}*{rng.choice((2, 4, 8))}+{rng.choice((1, 7, 0x51))}]",
                       f"add {a}, {REGISTER_PARTS['r9'][0] if dword else 'r9'}", f"not {target}", f"neg {target}"))


def loop_function(name, rng, form, nested, early):
    """A counted loop of one of four forms, with an optional data-dependent early exit and nested loop.

    dec: the counter runs from n down to 1 (DEC or SUB 1, then JNZ) or from n-1 down to 0 (DEC, then JNS).
    up: an index runs from 0 up to n-1 (INC, CMP with the count, then JB, JNE or JL).
    guarded: as dec from n, but n may be 0 and a TEST and JZ in front skips the loop.
    step: the counter drops by 2 or 3 while it stays above 0 (SUB, then JG).
    """
    saved = rng.sample(LOOP_SAVED, rng.choice((0, 0, 1, 2)))
    prolog = [line for register in saved for line in (f"push {register}", f".pushreg {register}")]
    carried = [*LOOP_CARRIED, *saved]
    top, exit_, out = f"{name}_top", f"{name}_exit", f"{name}_early"
    # The count n is x masked, plus 1 where the loop has no guard; a dec loop that ends at 0 starts from n-1.
    to_zero = form == "dec" and rng.random() < 0.5
    lines = ["mov r9d, ecx", f"and r9d, {7 if nested else 15}"]
    if form != "guarded" and not to_zero:
        lines.append("inc r9d")
    # Carried values defined from both arguments; each saved register is overwritten before it is read.
    lines += ["mov rax, rcx", "lea r10, [rdx+rcx*2+3]", f"mov QWORD PTR [r8+{LOOP_MEMORY}], rdx",
              f"mov DWORD PTR [r8+{LOOP_COUNT}], 0"]
    for register in saved:
        lines += [f"mov {register}, rdx", rng.choice((f"xor {register}, rcx", f"rol {register}, 17", f"add {register}, rcx"))]
    if form == "up":
        lines.append("xor r11d, r11d")
    if form == "guarded":
        lines += ["test r9d, r9d", f"jz {exit_}"]
    # The outer trace: one word per iteration indexed by the counter or index, from LOOP_TRACE.
    if form == "up":
        trace = f"mov QWORD PTR [r8+r11*8+{LOOP_TRACE}], rax"
    elif to_zero:
        trace = f"mov QWORD PTR [r8+r9*8+{LOOP_TRACE}], rax"
    else:
        trace = f"mov QWORD PTR [r8+r9*8+{LOOP_TRACE - 8}], rax"

    # RAX and a saved register, when there is one, change in every iteration.
    targets = ["rax", *saved[:1], *rng.sample(carried, rng.choice((1, 2)))]
    body = [loop_operation(rng, target, [r for r in carried if r != target]) for target in targets]
    body.append(rng.choice((f"add QWORD PTR [r8+{LOOP_MEMORY}], rax", f"xor QWORD PTR [r8+{LOOP_MEMORY}], r10",
                            f"sub QWORD PTR [r8+{LOOP_MEMORY}], rdx")))
    body.append(rng.choice((f"add rax, QWORD PTR [r8+{LOOP_MEMORY}]", f"xor r10, QWORD PTR [r8+{LOOP_MEMORY}]",
                            f"add rdx, QWORD PTR [r8+{LOOP_MEMORY}]")))
    body += [f"add DWORD PTR [r8+{LOOP_COUNT}], 1", trace]
    if early:
        producer = rng.choice((f"test al, {rng.choice((3, 7))}", f"test r10d, {rng.choice((0xF, 0x1F))}",
                               f"cmp eax, {rng.choice((0x10000000, 0x20000000))}"))
        # After the first update, so the flags never cross the loop head.
        at = rng.randrange(1, len(body) + 1)
        body[at:at] = [producer, f"{'jb' if producer.startswith('cmp') else 'jz'} {out}"]
    if nested:
        # The inner count comes from the carried RAX, so it differs from one outer iteration to the next.
        inner = f"{name}_inner"
        body += ["mov ecx, eax", "and ecx, 7", "inc ecx", f"{inner}:",
                 *(loop_operation(rng, target, [r for r in carried if r != target])
                   for target in rng.sample(carried, rng.choice((1, 2)))),
                 f"add QWORD PTR [r8+rcx*8+{LOOP_INNER - 8}], r10", "dec ecx", f"jnz {inner}"]
    if form == "dec":
        if to_zero:
            body += ["dec r9d", f"jns {top}"]
        else:
            body += [rng.choice(("dec r9d", "sub r9d, 1")), f"jnz {top}"]
    elif form == "guarded":
        body += ["dec r9d", f"jnz {top}"]
    elif form == "step":
        body += [f"sub r9d, {rng.choice((2, 3))}", f"jg {top}"]
    else:
        body += ["inc r11d", "cmp r11d, r9d", f"{rng.choice(('jb', 'jne', 'jl'))} {top}"]

    restore = [f"pop {register}" for register in reversed(saved)]
    stores = [f"mov QWORD PTR [r8+{LOOP_SAVES}], {saved[0]}", f"add rax, {saved[0]}"] if saved else ["xor rax, r10"]
    lines += [f"{top}:", *body, f"{exit_}:", "mov QWORD PTR [r8], rax", "mov QWORD PTR [r8+8], r10",
              "mov QWORD PTR [r8+16], rdx", f"mov DWORD PTR [r8+{LOOP_FINAL}], r9d", *stores, *restore, "ret"]
    if early:
        lines += [f"{out}:", f"mov DWORD PTR [r8+{LOOP_EARLY}], 0EA51h", f"mov QWORD PTR [r8+{LOOP_EARLY + 8}], r9",
                  "mov QWORD PTR [r8+16], rdx", "lea rax, [rax+r10*2]", *restore, "ret"]
    return Function(name, lines, prolog)


def generate_loops(seed, count):
    """Exports with counted loops: the 16 combinations of form, nesting and early exit in a seeded order,
    repeated with other bodies."""
    rng = random.Random(seed)
    shapes = [(form, nested, early) for form in LOOP_FORMS for nested in (False, True) for early in (False, True)]
    rng.shuffle(shapes)
    return [loop_function(f"fuzz_loop_{index}", rng, *shapes[index % len(shapes)]) for index in range(count)]


AREAS = {"leaf": (generate, "cpu_fuzz"), "frames": (generate_frames, "cpu_fuzz_frames"),
         "calls": (generate_calls, "cpu_fuzz_calls"), "chains": (generate_chains, "cpu_fuzz_chains"),
         "switches": (generate_switches, "cpu_fuzz_switches"),
         "scalar_fp": (generate_scalar_fp, "cpu_fuzz_scalar_fp"), "rsp": (generate_rsp, "cpu_fuzz_rsp"),
         "loops": (generate_loops, "cpu_fuzz_loops")}


def inputs_for(seed, area="leaf", signature=None):
    if area == "scalar_fp":
        return fp_inputs(seed, signature.split(":")[1][:2])
    boundary = [
        0, 1, 2, 3, 7, 0x0F, 0x10, 0x1F, 0x20, 0x3F, 0x40, 0x7F, 0x80, 0x81, 0xFF, 0x100,
        0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x100000000,
        0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF, 0xDEADBEEF, 0x12345678,
        0x0FEDCBA9, 0xA5A5A5A5A5A5A5A5, 0x5A5A5A5A5A5A5A5A,
    ]
    values = [(value, 0) for value in boundary] + [(0, value) for value in boundary] + [(value, value) for value in boundary]
    rng = random.Random(seed ^ 0x5EED)
    while len(values) < 313:
        values.append((rng.getrandbits(64), rng.getrandbits(64)))
    if area == "switches":
        # Selectors that reach every arm and the default, then the same selectors with upper bits set: a
        # 32-bit compare ignores them, a 64-bit compare does not, and a signed clamp sees bit 31.
        rng = random.Random(seed ^ 0x5E1EC7)
        values = ([(selector, rng.getrandbits(64)) for selector in range(SWITCH_SELECTORS)]
                  + [(upper | selector, rng.getrandbits(64)) for selector in range(4)
                     for upper in (1 << 32, 0x80000000, 0xFFFFFFFF00000000)] + values)
    return values[:313]


def bind_symbol(symbol, function):
    """Declare a loaded export's prototype: three 64-bit integer slots, or the function's own signature,
    where a float or double slot is passed as a double so that every bit pattern reaches the XMM register."""
    result, kinds = function.signature.split(":") if function.signature else ("q", "qqq")
    symbol.argtypes = [ctypes.c_uint64 if kind == "q" else ctypes.c_double for kind in kinds]
    symbol.restype = ctypes.c_uint64 if result == "q" else ctypes.c_double


def call_symbol(symbol, function, x, y, out):
    """Call an export on x, y and the output pointer. A float takes the low 32 bits of its value, with the high
    half set to a pattern that the function must ignore; a float result is its low 32 bits."""
    if function.signature is None:
        return symbol(x, y, out)
    result, kinds = function.signature.split(":")
    slots = []
    for kind, value in zip(kinds, (x, y, out)):
        bits = (value & 0xFFFFFFFF | FP_JUNK) if kind == "f" else value
        slots.append(value if kind == "q" else struct.unpack("<d", struct.pack("<Q", bits))[0])
    answer = symbol(*slots)
    if result == "q":
        return answer
    bits = struct.unpack("<Q", struct.pack("<d", answer))[0]
    return bits & 0xFFFFFFFF if result == "f" else bits


def translator_module(area):
    """The module that lifts the functions of an area: the scalar SSE lifter, or the shared one."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if area == "scalar_fp":
        import cpu_sse_aot

        return cpu_sse_aot
    import cpu_switch_aot

    return cpu_switch_aot


def export_name(function):
    """The name a translator takes for an export: with its signature when it has one."""
    return f"{function.name}={function.signature}" if function.signature else function.name


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def buffer_text(data, start, stop):
    return " ".join(f"{byte:02x}" for byte in data[start:stop])


def run_check(seed, count, native, lifted, o2, only=None, input_index=None, area="leaf", mutant=None):
    if first_difference(b"\x01\x02", b"\x01\x02") is not None:
        raise SystemExit("comparator self-test failed on equal bytes")
    if first_difference(b"\x01\x02", b"\x01\x03") != 1:
        raise SystemExit("comparator self-test failed on a changed byte")
    if mutant is not None and area not in MUTANTS:
        raise SystemExit(f"no planted translator fault for --area {area}")

    functions = AREAS[area][0](seed, count)
    controls = [function for function in functions if function.refused]
    functions = [function for function in functions if not function.refused and function.export]
    libraries = [(native, ""), (lifted, "lifted_"), (o2, "lifted_")]
    if mutant is not None:
        libraries.append((mutant, "lifted_"))
    loaded = []
    for path, prefix in libraries:
        library = ctypes.CDLL(str(path))
        for function in functions:
            symbol = getattr(library, prefix + function.name)
            bind_symbol(symbol, function)
        loaded.append((library, prefix))

    native_library, lifted_library, o2_library = (item[0] for item in loaded[:3])
    if count <= 0:
        raise SystemExit("CPU_FUZZ_FUNCTIONS must be positive")
    selected = [function for function in functions if only is None or function.name == only]

    def function_inputs(function):
        return [(index, value) for index, value in enumerate(inputs_for(seed, area, function.signature))
                if input_index is None or index == input_index]

    if not selected:
        raise SystemExit(f"no generated function matches --only {only!r}")
    if not function_inputs(selected[0]):
        raise SystemExit(f"no input matches --input {input_index}")
    checks = 0
    for function in selected:
        native_symbol = getattr(native_library, function.name)
        cases = function_inputs(function)
        for variant, library in (("default", lifted_library), ("/O2", o2_library)):
            lifted_symbol = getattr(library, "lifted_" + function.name)
            for index, (x, y) in cases:
                native_buffer = ctypes.create_string_buffer(288)
                lifted_buffer = ctypes.create_string_buffer(288)
                ctypes.memset(native_buffer, 0x5A, len(native_buffer))
                ctypes.memset(lifted_buffer, 0x5A, len(lifted_buffer))
                native_return = call_symbol(native_symbol, function, x, y, ctypes.addressof(native_buffer) + 16)
                lifted_return = call_symbol(lifted_symbol, function, x, y, ctypes.addressof(lifted_buffer) + 16)
                checks += 1
                difference = first_difference(native_buffer.raw, lifted_buffer.raw)
                if native_return != lifted_return or difference is not None:
                    print(f"seed={seed:#x} functions={count} function={function.name} input={index} x={x:#x} y={y:#x}", file=sys.stderr)
                    print(f"native return={native_return:#x} {variant} return={lifted_return:#x}", file=sys.stderr)
                    if difference is not None:
                        low = max(0, difference - 8)
                        high = min(len(native_buffer), difference + 8)
                        print(f"first differing byte={difference} (output offset {difference - 16})", file=sys.stderr)
                        print(f"native {buffer_text(native_buffer.raw, low, high)}", file=sys.stderr)
                        print(f"lifted {buffer_text(lifted_buffer.raw, low, high)}", file=sys.stderr)
                    print(function.asm(), file=sys.stderr)
                    for callee in function.callees:
                        print(callee.asm(), file=sys.stderr)
                    command = (f"python -B tools/fuzz_cpu.py check {seed:#x} {count} {native} {lifted} {o2} "
                               f"--area {area} --only {function.name} --input {index}")
                    print(f"reproduce: {command}", file=sys.stderr)
                    return 1
    prefix = "" if area == "leaf" else f"area={area} "
    print(f"{prefix}seed={seed:#x} functions={len(selected)} inputs={len(function_inputs(selected[0]))} "
          f"comparisons={checks} default and /O2 matched")
    if controls and only is None:
        lifter = translator_module(area)
        for control in controls:
            try:
                lifter.translate(Path(native), [export_name(control)])
            except ValueError as error:
                if control.refused not in str(error):
                    print(f"{control.name} refused for another reason: {error}", file=sys.stderr)
                    return 1
            else:
                print(f"{control.name} was translated; expected refusal: {control.refused}", file=sys.stderr)
                print(control.asm(), file=sys.stderr)
                return 1
        print(f"controls refused: {', '.join(f'{control.name} ({control.refused})' for control in controls)}")
    if mutant is not None:
        mutant_library = loaded[3][0]
        mismatches = total = 0
        failing = set()
        for function in selected:
            native_symbol = getattr(native_library, function.name)
            mutant_symbol = getattr(mutant_library, "lifted_" + function.name)
            for _, (x, y) in function_inputs(function):
                native_buffer = ctypes.create_string_buffer(288)
                mutant_buffer = ctypes.create_string_buffer(288)
                ctypes.memset(native_buffer, 0x5A, len(native_buffer))
                ctypes.memset(mutant_buffer, 0x5A, len(mutant_buffer))
                native_return = call_symbol(native_symbol, function, x, y, ctypes.addressof(native_buffer) + 16)
                mutant_return = call_symbol(mutant_symbol, function, x, y, ctypes.addressof(mutant_buffer) + 16)
                total += 1
                if native_return != mutant_return or native_buffer.raw != mutant_buffer.raw:
                    mismatches += 1
                    failing.add(function.name)
        print(f"mutant {MUTANTS[area][0]}: {mismatches}/{total} mismatches in {len(failing)}/{len(selected)} functions")
        if not mismatches:
            print("the planted translator fault was not detected", file=sys.stderr)
            return 1
    return 0


def emit(seed, count, asm_path, def_path, area="leaf"):
    generator, library = AREAS[area]
    functions = generator(seed, count)
    body = "\n\n".join(function.asm() for function in functions)
    asm = ("EXTERN __ImageBase:BYTE\n" if "__ImageBase" in body else "") + ".code\n\n" + body + "\n\nEND\n"
    definition = (f"LIBRARY {library}\nEXPORTS\n"
                  + "".join(f"    {function.name}\n" for function in functions if function.export))
    Path(asm_path).write_text(asm, encoding="utf-8")
    Path(def_path).write_text(definition, encoding="utf-8")


def plant_reload_shift(cpu_switch_aot, image, functions):
    """Make every frame reload narrower than 8 bytes read one byte higher in the frame array.

    This is a planted translator fault for the negative control: a partial reload at the wrong
    index changes only the value read back, which the return value and output buffer show.
    """
    from iced_x86 import Mnemonic as M, MemorySizeExt, OpKind

    original = cpu_switch_aot.Lifter.slot

    def slot(self, ins):
        text = original(self, ins)
        if (ins.mnemonic in (M.MOV, M.MOVZX, M.MOVSX, M.MOVSXD) and ins.op0_kind == OpKind.REGISTER
                and ins.op1_kind == OpKind.MEMORY and MemorySizeExt.size(ins.memory_size) < 8):
            index = self.frame[ins.ip]
            text = text.replace(f"frame + {index})", f"frame + {index + 1})")
        return text

    cpu_switch_aot.Lifter.slot = slot


def plant_call_retarget(cpu_switch_aot, image, functions):
    """Send the first call of the first caller to the next callee in the generated order.

    This is a planted translator fault for the negative control: the caller's own instructions are
    unchanged, but the wrong callee computes RAX and writes the call's output segment.
    """
    caller = next(function for function in functions if function.callees)
    callees = [function for function in functions if not function.callees]
    wrong = callees[(callees.index(caller.callees[0]) + 1) % len(callees)]
    with cpu_switch_aot.pefile.PE(str(image), fast_load=True) as pe:
        caller_rva = cpu_switch_aot.export_rva(pe, caller.name)
        wrong_rva = cpu_switch_aot.export_rva(pe, wrong.name)
    original = cpu_switch_aot.Program.callee
    planted = []

    def callee(self, target, ip):
        # Callees are resolved in address order, so the first one asked for by the caller is its first call.
        if not planted and self.active[-1] == caller_rva:
            planted.append(ip)
            target = wrong_rva
        return original(self, target, ip)

    cpu_switch_aot.Program.callee = callee


CALL_SITE = re.compile(r"\b(rax = |return )(fn_[0-9a-f]+)\(rcx, rdx, ")


def plant_swapped_arguments(cpu_switch_aot, image, functions):
    """Pass RDX first and RCX second at every direct call and tail jump of the translation.

    This is a planted translator fault for the negative control: every callee that is not
    symmetric in its two arguments returns another value.
    """
    original = cpu_switch_aot.translate

    def translate(*args, **kwargs):
        source, rest = original(*args, **kwargs)
        swapped = CALL_SITE.sub(lambda match: f"{match.group(1)}{match.group(2)}(rdx, rcx, ", source)
        if swapped == source:
            raise SystemExit("the swapped_arguments fault found nothing to change")
        return swapped, rest

    cpu_switch_aot.translate = translate


def plant_table_rotate(cpu_switch_aot, image, functions):
    """Read every jump-table entry one slot late, so each selector reaches the arm of the next entry.

    This is a planted translator fault for the negative control: the bound check and the control flow
    are unchanged, but a selector runs another arm, which the markers, RDX and the return value show.
    """
    original = cpu_switch_aot.Function.dispatch

    def dispatch(self, ip):
        found = original(self, ip)
        index, targets, *rest = self.dispatches[ip]
        self.dispatches[ip] = (index, targets[1:] + targets[:1], *rest)
        return found

    cpu_switch_aot.Function.dispatch = dispatch


def plant_minmax_swap(cpu_sse_aot, image, functions):
    """Make MINSS and MINSD keep the larger operand and MAXSS and MAXSD the smaller one.

    This is a planted translator fault for the negative control: only a compare that ends in the
    destination or the source operand changes, so equal operands and NaNs still agree and every other
    instruction is untouched.
    """
    from iced_x86 import Mnemonic as M

    extrema = cpu_sse_aot.EXTREMA
    for low, high in ((M.MINSS, M.MAXSS), (M.MINSD, M.MAXSD)):
        extrema[low], extrema[high] = extrema[high], extrema[low]


def plant_stale_rsp(cpu_switch_aot, image, functions):
    """Resolve every [RSP+d] operand against the RSP that the prologue's allocation leaves (after the first
    SUB RSP in address order), as if no later SUB, ADD, PUSH or POP had moved RSP.

    This is a planted translator fault for the negative control: the frame proof, PUSH, POP and every
    [RBP+d] operand are unchanged, but a value stored at one RSP and read back at another reaches other
    frame bytes. An operand whose moved bytes would leave the frame array keeps its index.
    """
    from iced_x86 import Mnemonic as M, MemorySizeExt, Register as R

    shifts = {}  # RVA -> (RSP at the instruction minus RSP after the prologue, bytes of the frame array)
    original_frame = cpu_switch_aot.Frame.__init__

    def frame_init(self, function, edges, internal=False):
        original_frame(self, function, edges, internal)
        if self.adjustments:
            floor = self.offsets[function.instructions[min(self.adjustments)].next_ip]
            for ip, offset in self.offsets.items():
                if offset != floor and ip in self.accesses:
                    shifts[ip] = (offset - floor, self.size)

    original_slot = cpu_switch_aot.Lifter.slot

    def slot(self, ins):
        text = original_slot(self, ins)
        if ins.ip in shifts and ins.memory_base == R.RSP and ins.mnemonic not in (M.PUSH, M.POP):
            shift, size = shifts[ins.ip]
            index = self.frame[ins.ip]
            if 0 <= index - shift <= size - MemorySizeExt.size(ins.memory_size):
                text = text.replace(f"frame + {index})", f"frame + {index - shift})")
        return text

    cpu_switch_aot.Frame.__init__ = frame_init
    cpu_switch_aot.Lifter.slot = slot


def plant_back_edge_exit(cpu_switch_aot, image, functions):
    """Retarget every backward conditional branch to its own fallthrough, the instruction after the loop.

    This is a planted translator fault for the negative control: each loop body runs once, so any input
    with more than one iteration leaves another trace, count and carried value. Every input still terminates.
    """
    from iced_x86 import FlowControl

    original = cpu_switch_aot.Function.decode
    planted = []

    def decode(self, ip):
        ins = original(self, ip)
        if ins.flow_control == FlowControl.CONDITIONAL_BRANCH and ins.near_branch_target <= ip:
            ins.near_branch64 = ins.next_ip
            planted.append(ip)
        return ins

    cpu_switch_aot.Function.decode = decode
    translate_original = cpu_switch_aot.translate

    def translate(*args, **kwargs):
        result = translate_original(*args, **kwargs)
        if not planted:
            raise SystemExit("the back_edge_exit fault found no backward branch")
        return result

    cpu_switch_aot.translate = translate


MUTANTS = {"frames": ("reload_shift", plant_reload_shift), "calls": ("call_retarget", plant_call_retarget),
           "chains": ("swapped_arguments", plant_swapped_arguments),
           "switches": ("table_rotate", plant_table_rotate), "scalar_fp": ("minmax_swap", plant_minmax_swap),
           "rsp": ("stale_rsp", plant_stale_rsp), "loops": ("back_edge_exit", plant_back_edge_exit)}


def translate(seed, count, image, output, area="leaf", mutant=False):
    lifter = translator_module(area)
    functions = AREAS[area][0](seed, count)
    if mutant:
        if area not in MUTANTS:
            raise SystemExit(f"no planted translator fault for --area {area}")
        MUTANTS[area][1](lifter, image, functions)
    # Negative controls stay in the image for the check to translate on their own.
    functions = [function for function in functions if not function.refused and function.export]
    names = [export_name(function) for function in functions]
    try:
        source, _ = lifter.translate(Path(image), names)
    except ValueError as error:
        print(f"seed={seed:#x} functions={count}: {error}", file=sys.stderr)
        for function in functions:
            try:
                lifter.translate(Path(image), [export_name(function)])
            except ValueError as inner:
                print(f"refused {function.name}: {inner}", file=sys.stderr)
                print(function.asm(), file=sys.stderr)
                break
        return 1
    Path(output).write_text(source, encoding="utf-8")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    emit_parser = commands.add_parser("emit")
    emit_parser.add_argument("seed", type=lambda value: int(value, 0))
    emit_parser.add_argument("count", type=int)
    emit_parser.add_argument("asm", type=Path)
    emit_parser.add_argument("definition", type=Path)
    emit_parser.add_argument("--area", choices=AREAS, default="leaf")
    translate_parser = commands.add_parser("translate")
    translate_parser.add_argument("seed", type=lambda value: int(value, 0))
    translate_parser.add_argument("count", type=int)
    translate_parser.add_argument("image", type=Path)
    translate_parser.add_argument("output", type=Path)
    translate_parser.add_argument("--area", choices=AREAS, default="leaf")
    translate_parser.add_argument("--mutant", action="store_true",
                                  help="plant the area's translator fault for the negative control: "
                                       + ", ".join(f"{name} for {area}" for area, (name, _) in MUTANTS.items()))
    check_parser = commands.add_parser("check")
    check_parser.add_argument("seed", type=lambda value: int(value, 0))
    check_parser.add_argument("count", type=int)
    check_parser.add_argument("native", type=Path)
    check_parser.add_argument("lifted", type=Path)
    check_parser.add_argument("o2", type=Path)
    check_parser.add_argument("--only")
    check_parser.add_argument("--input", type=int, dest="input_index")
    check_parser.add_argument("--area", choices=AREAS, default="leaf")
    check_parser.add_argument("--mutant", type=Path, help="a translation with the planted fault, which must mismatch")
    args = parser.parse_args(argv)

    if args.command == "emit":
        emit(args.seed, args.count, args.asm, args.definition, args.area)
        return 0
    if args.command == "translate":
        return translate(args.seed, args.count, args.image, args.output, args.area, args.mutant)
    return run_check(args.seed, args.count, args.native, args.lifted, args.o2, args.only, args.input_index,
                     args.area, args.mutant)


if __name__ == "__main__":
    raise SystemExit(main())