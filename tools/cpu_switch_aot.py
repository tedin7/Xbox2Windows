"""Recompile bounded Windows x64 leaf functions with jump-table dispatch to C++.

Each export takes three integer words in RCX, RDX and R8 and returns RAX. The
translator follows branches and jump-table entries from the export, so data
placed after the code (as MSVC does with its tables) is never decoded. An
indirect JMP is accepted only at the end of one of these sequences, reached by
the fallthrough of an unsigned bound check on the index (JA or JAE to the default),
or by the taken branch of the inverted form (JB or JBE to the dispatch block,
which a JMP to the default precedes and no other edge enters):

    mov E32, [B+I*4+table]; add E, B; jmp E   B = image base (MSVC)
    movsxd E, [B+I*4];      add E, B; jmp E   B = table      (LLVM)

B must hold the same RIP-relative address on every path, and the index must be
shown to carry the compared value. The generated C++ replaces the dispatch with
a switch over the table targets.

An export listed as "name:4" also takes a fourth integer word in R9: R9 is then
defined at entry and the generated function has a fourth parameter. "name" and
"name:3" take three, and R9 is a scratch register that must be written before it
is read; any other count is rejected. The count is chosen by whoever invokes the
translator, not derived from the machine code, and the host must call the
translated function with the same number of arguments. An export's stack
arguments (the fifth and later) and floating-point arguments are not supported;
an internal callee may take stack arguments, as described below.

Flags are modeled only between a producer and its consumers (Jcc, SETcc,
CMOVcc) in one straight-line run that no edge enters. Overflow and low-byte
parity conditions use the producer operand width, including narrow arithmetic.
Instructions that leave the flags unchanged (moves, LEA, NOT, BSWAP, SETcc, CMOVcc, and
CBW, CWDE, CDQE, CDQ and CQO) may sit between them.

SHL, SAL (its /6 encoding alias), SHR and SAR produce flags only with an immediate
count whose masked value (five bits, six for a 64-bit operand) is 1 to the operand
width minus 1. A count of 0 leaves the flags of an earlier instruction, which a
shift by CL may do, and CF is undefined for SHL, SAL and SHR by the operand width
or more, so those are refused, and SAR by such a count with them. CF is the last
bit shifted out and ZF, SF and PF come from the result. OF is defined only for a
count of 1 (MSB of the result XOR CF for SHL and SAL, MSB of the operand for SHR,
0 for SAR), so O, NO, L, GE, LE and G are refused after a larger count.

BT, BTS, BTR and BTC take a 16-, 32- or 64-bit register or memory bit base and
a register or immediate bit offset. Register bases and immediate offsets use
the offset modulo the operand width. A register offset into ordinary memory is
signed and addresses a bit string; frame, image-relative and 32-bit-address
forms are refused. CF is the selected bit before the
write; the Intel SDM leaves ZF unaffected and the other flags undefined, so
only the B and AE conditions may read it.

POPCNT, LZCNT and TZCNT take a 16-, 32- or 64-bit register or memory source.
LZCNT and TZCNT return the operand width for a zero source. As the Intel SDM
defines their flags, POPCNT sets ZF for a zero source and clears OF, SF, AF, CF
and PF, so every condition may read them. LZCNT and TZCNT set CF for a zero
source and ZF for a zero result and leave OF, SF, AF and PF undefined, so only
B, AE, E, NE, BE and A may read them. The host must support the instructions
whose results the original computes: without ABM and BMI1 a processor runs
LZCNT and TZCNT as BSR and BSF.

BSF and BSR take a 16-, 32- or 64-bit register destination and a register or
memory source of the same width. A nonzero source gives the index of its lowest
(BSF) or highest (BSR) set bit and clears ZF. For a zero source ZF is set and
the destination is left unchanged. The Intel SDM calls that destination
undefined; AMD documents it as unchanged, and the translation follows the
unchanged behaviour of the processors measured, 32-bit destinations included:
a zero source does not clear bits 63:32. A nonzero source zero-extends a 32-bit
result and keeps bits 63:16 of a 16-bit destination. CF, OF, SF, PF and AF are
undefined, so only the E and NE conditions may read them and every other
condition is refused. The destination is read, so it must be defined.

BSWAP takes a 32- or 64-bit register and reverses its bytes in place; no flag changes.
A 32-bit BSWAP zero-extends its result into the 64-bit register like any other 32-bit
write. The 16-bit form (operand-size prefix) is refused: the Intel SDM leaves its
result undefined.

One-operand MUL, IMUL, DIV and IDIV and the sign extensions read and write RAX
and RDX without naming them. Products of up to 32-bit factors are computed in
std::uint64_t; 64-bit products and all quotients use the MSVC intrinsics
_umul128, _mul128, _udiv64, _div64, _udiv128 and _div128. The divide
intrinsics execute DIV or IDIV, so a zero divisor or a quotient that does not
fit raises the same divide error. The quotient is also stored to a volatile
local, so the divide is performed, and its error raised, when the function
overwrites RAX and RDX without reading them. 8- and 16-bit divides are rejected, and so is a flag
consumer after a multiply or divide.

A forward pass records how many low bytes (0, 1, 2 or 8) of each register every
path defines. A read needs as many defined bytes as it uses, and an 8- or 16-bit
write keeps the other bytes of its register. The return width is the number of
consecutive low RAX bytes that are definitely defined at every reachable RET or
RET 0: 0, 1, 2 or 8, the lower bound over joins and over returns. It counts
defined bytes and does not establish a source-language type. It is None when no
RET is reachable, which describes the reachable control flow and makes no
termination claim. The generated source states the width unless it is 8.

RIP-relative data is refused unless the invoker passes --image-data
(image_data=True). The translation then addresses a live mapping of the input
image that the host binds at run time through lifted_bind_image, which checks
that the mapping is that image; the translated functions keep their signatures.
A RIP-relative memory operand becomes a load or store at base + RVA. Its whole
width must lie inside one section's virtual extent (VirtualSize, zero-filled
past the file-backed bytes); a read needs a readable section and a write a
writable, non-executable one. A RIP-relative LEA yields base + RVA for any RVA
inside the image, except one that loads a recognised jump-table base: that stays
a guest address, which the switch replaces, and never needs the mapping. The
generated source states the binding contract.
A function may use its stack. A forward pass gives RSP at
every instruction as entry RSP plus a constant that must agree at joins and be
0 at every RET. PUSH and POP of a nonvolatile register, SUB or ADD RSP by a
multiple of 8, and LEAVE where the RBP anchor is proven are the only changes of
RSP accepted. LEAVE sets RSP to the anchor and pops RBP from it. ENTER, MOV RSP,
RBP and any other write of RSP is a dynamic frame, and any use of RSP as a value (MOV RAX, RSP or LEA RAX, [RSP+8])
would let the frame's address escape. Both are rejected. The frame becomes one
byte array per call, from the lowest RSP up to the end of the caller's 32-byte
home space at entry RSP + 40, at most MAX_FRAME bytes, and each RSP-based
operand a load or store at a constant index into it. An access below the
current RSP (Windows x64 has no red zone), of the return address or, in an
export, at or above entry RSP + 40 (a stack argument) is rejected. A second pass follows
every stack byte: a read needs each byte written on every path, so home space
must be written before it is read. RBX, RBP, RSI, RDI and R12-R15 may be used
when every RET is reached with each one holding its entry value again, saved
and restored by PUSH and POP or by a 64-bit MOV to and from the frame. Their
entry values are not arguments: one may only be saved, and its saved bytes
only restored into the same register. The generated function is C++ that the
host compiler gives its own frame, so exception unwinding through the original
frame layout is not reproduced.

A direct CALL, and a direct JMP that leaves the function for another function's
entry (a tail jump), are translated by RVA into the same source: each callee
once, as a static function named fn_<rva> that takes RCX, RDX, R8 and R9,
defined before its first caller. A target must be the first byte of a verified
primary runtime function (unchained, version 1 or 2 unwind data, executable) or
a non-forwarded code export that no other function's interval holds; chained
ranges, interior labels, forwarded and data exports are refused. A caller
is refused when any callee is. A callee records how many low bytes of each argument
register it may read before writing it, and the caller must define them at the
call. After a call RAX holds the callee's return width of defined bytes; the
other volatile registers, the flags and the callee's 32-byte home space at RSP
are undefined, and the nonvolatile registers and the rest of the frame,
including the caller's own home space above entry RSP, are kept. The return
address and the callee's frame lie below RSP, where every byte is already
undefined because the frame proof forgets the bytes that ADD RSP, POP and LEAVE
leave below RSP. RSP must be 8
mod 16 from entry at a call, with the callee's home space below the return
address. A tail jump is checked as a RET and contributes the callee's return
width. The callee's frame is its own array, so no frame address crosses a call.

An internal function may read and write up to MAX_STACK_ARGUMENTS stack
arguments, the 8-byte slots from entry RSP + 40, each access inside one slot. Its
frame array extends over them, copied at entry from one parameter per slot after
R9, and it records how many low bytes of each slot some read reaches. A caller
must have those bytes written at RSP + 32 + 8 * slot when the CALL executes, below
its own return address, and passes the 8 bytes of each slot. Afterwards the slots
are undefined, like the callee's home space, because the callee may write them.
A tail jump to a function with stack arguments is refused.

A CALL may close a call cycle, an export's own entry included, which becomes an
internal function too. A call into a function whose translation is in progress
uses an assumed return width and demands, at first 8 bytes and none, and the
functions of the cycle are translated again from the meet of the assumed and
computed values until no computed width is lower and no demand higher. Each
function of a cycle is declared before the definitions. A tail jump that closes
a cycle is refused: the original runs such a cycle in constant stack. The host
stack bounds the depth of recursion, at a depth that differs from the original.

A target without unwind data or export is refused, with a message saying that
a range binding is needed, unless the invoker supplies one: --bind-ranges names
a JSON manifest of trusted [start, end) RVAs from the source labels of this
exact image, bound to its SHA-256. Each range must lie in executable code and
overlap no runtime function and no other binding. A call or tail jump may enter
a bound range only at its start; the callee is walked within that range alone,
may not move RSP or call, and may not be an import thunk.

A call or tail jump through the import address table is translated only through
an explicit binding (--bind, --import-arguments, --return-bytes). One binding
covers one IAT slot for all functions of a run, which must hold the import that
the binding names. With several --bind options, each names a distinct ordinary
import of the image and binds the slot that holds it; a reached slot that no
binding names is refused, as is a binding that no call or tail jump uses.
It may be reached by call or jmp QWORD PTR [rip+slot], or by a direct CALL or
JMP of an import thunk: the six bytes FF 25 disp32 at a target outside every
unwind interval and bound range, read as a call or tail jump through their
slot. A call or thunk counts only when its slot is one of the ordinary import
directory; the slot must be eight bytes of one section marked readable.
Those checks establish the slot identity and expected initial mapping. During
execution of the original and its translation, the caller must keep the image
and bound target valid and ensure the slot stays readable and contains that
target. The translation directly calls the binding. Delay-loaded imports,
rebinding and page protection changes are outside this contract. A call becomes
"rax = import(...);" under the rules of a direct call above: RSP 8
mod 16, the home space below the return address, and each register of the
binding's argument count (RCX, RDX, R8, then R9) fully defined. After it RAX
holds the binding's number of guaranteed low bytes, none for 0, and the other
volatile registers, the flags and the 32 bytes at RSP are undefined; the
nonvolatile registers and the rest of the frame are kept. That is a term of the
binding: the import is taken to follow the Windows x64 calling convention,
which the translator cannot check. A tail jump becomes "return import(...);"
and contributes the binding's number to the return width; a call contributes
only through the RAX bytes defined at a later RET.
"""

import re
import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import NamedTuple

from cpu_call_aot import resolve_import

import pefile
from iced_x86 import (Code, ConditionCode as CC, Decoder, FlowControl, InstructionInfoFactory, MemorySizeExt,
                      Mnemonic as M, OpAccess, OpKind, Register as R, RegisterExt)

from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges, image_bytes


NAMES = {R.RAX: "rax", R.RCX: "rcx", R.RDX: "rdx", R.RBX: "rbx", R.RSP: "rsp", R.RBP: "rbp",
         R.RSI: "rsi", R.RDI: "rdi", R.R8: "r8", R.R9: "r9", R.R10: "r10", R.R11: "r11",
         R.R12: "r12", R.R13: "r13", R.R14: "r14", R.R15: "r15"}
# Integer argument registers in order. An entry takes the first three unless it is listed as "name:4".
ARGUMENT_REGISTERS = (R.RCX, R.RDX, R.R8, R.R9)
ARGUMENT_COUNTS = (3, 4)
IMPORT_ARGUMENT_COUNTS = range(len(ARGUMENT_REGISTERS) + 1)
IMPORT_RETURN_WIDTHS = (0, 1, 2, 4, 8)
STACK_OUTPUT_WIDTHS = (1, 2, 4, 8)
DEFAULT_ARGUMENTS = 3
VOLATILE = {R.RAX, R.RCX, R.RDX, R.R8, R.R9, R.R10, R.R11}
NONVOLATILE = tuple(sorted(set(NAMES) - VOLATILE - {R.RSP}))  # RBX, RBP, RSI, RDI, R12-R15
# The caller's home space ends 40 bytes above entry RSP: 8 for the return address and 32 reserved.
HOME_END = 40
# Bytes of the frame array. MSVC calls __chkstk before it allocates a page (4096 bytes) or more, and that
# call's target has no unwind entry, so an accepted MSVC frame stays below a page.
MAX_FRAME = 4096
# The home space a callee may write: the 32 bytes at RSP when the CALL executes.
HOME_SIZE = 32
# Stack arguments an internal function may take: the fifth to the twelfth, 8 bytes each from entry RSP + 40.
MAX_STACK_ARGUMENTS = 8
HIGH_BYTES = {R.AH, R.BH, R.CH, R.DH}
# Flow that continues at the next instruction (a direct call does once its callee returns).
STRAIGHT = (FlowControl.NEXT, FlowControl.CONDITIONAL_BRANCH, FlowControl.CALL, FlowControl.INDIRECT_CALL)
MAX_CALLEES = 256
ALU = {M.ADD: "+", M.SUB: "-", M.AND: "&", M.OR: "|", M.XOR: "^"}
SHIFTS = {M.SHL: "<<", M.SAL: "<<", M.SHR: ">>", M.SAR: ">>"}
ROTATES = {M.ROL: "true", M.ROR: "false"}
# Bit tests, by the C++ operator that writes the selected bit (BT writes nothing).
BIT_TESTS = {M.BT: None, M.BTS: "|", M.BTR: "& ~", M.BTC: "^"}
# Bit scans, by the PRELUDE helper that computes the index of the bit found in a nonzero source.
BIT_SCANS = {M.BSF: "trailing_zeros", M.BSR: "highest_bit"}
# Bit counts, by the PRELUDE helper that computes each.
BIT_COUNTS = {M.POPCNT: "count_ones", M.LZCNT: "leading_zeros", M.TZCNT: "trailing_zeros"}
_CODES = "O NO B AE E NE BE A S NS P NP L GE LE G".split()
SETCC = {getattr(M, "SET" + code) for code in _CODES}
CMOVCC = {getattr(M, "CMOV" + code) for code in _CODES}
# Sign extensions inside RDX:RAX, as (source, destination). The Intel SDM lists "Flags Affected: None".
SIGN_EXTENSIONS = {M.CBW: (R.AL, R.AX), M.CWDE: (R.AX, R.EAX), M.CDQE: (R.EAX, R.RAX),
                   M.CDQ: (R.EAX, R.EDX), M.CQO: (R.RAX, R.RDX)}
# Implicit operands of one-operand MUL, IMUL, DIV and IDIV, by operand width.
ACCUMULATOR = {1: R.AL, 2: R.AX, 4: R.EAX, 8: R.RAX}
DATA = {2: R.DX, 4: R.EDX, 8: R.RDX}
# Key by decoded Code, not Mnemonic: MOVSD also names an unrelated SSE instruction.
STRING_TRANSFERS = {
    Code.MOVSB_M8_M8: ("movs", 1), Code.MOVSW_M16_M16: ("movs", 2),
    Code.MOVSD_M32_M32: ("movs", 4), Code.MOVSQ_M64_M64: ("movs", 8),
    Code.STOSB_M8_AL: ("stos", 1), Code.STOSW_M16_AX: ("stos", 2),
    Code.STOSD_M32_EAX: ("stos", 4), Code.STOSQ_M64_RAX: ("stos", 8),
}
# Instructions that neither read nor write the arithmetic flags, besides SETcc and CMOVcc,
# which only read them.
FLAG_NEUTRAL = {M.CLZERO, M.MOV, M.MOVZX, M.MOVSX, M.MOVSXD, M.LEA, M.NOP, M.NOT, M.BSWAP, *SETCC, *CMOVCC, *SIGN_EXTENSIONS,
                M.MOVSB, M.MOVSW, M.MOVSD, M.MOVSQ, M.STOSB, M.STOSW, M.STOSD, M.STOSQ, M.CLD, M.STD, M.LEAVE,
                M.RDFSBASE, M.RDGSBASE, M.RDPID}
HOST_CPU_READ_BASE_CODES = {
    Code.RDFSBASE_R32: ("_readfsbase_u32", 4),
    Code.RDFSBASE_R64: ("_readfsbase_u64", 8),
    Code.RDGSBASE_R32: ("_readgsbase_u32", 4),
    Code.RDGSBASE_R64: ("_readgsbase_u64", 8),
}
FLAG_PRODUCERS = {*ALU, M.CMP, M.TEST, M.INC, M.DEC, M.NEG, *SHIFTS, *BIT_TESTS, *BIT_COUNTS, *BIT_SCANS}
MAX_FLAG_DISTANCE = 16
CONDITIONS = {CC.O: "o", CC.NO: "no", CC.P: "p", CC.NP: "np",
              CC.E: "e", CC.NE: "ne", CC.B: "b", CC.AE: "ae", CC.BE: "be", CC.A: "a",
              CC.L: "l", CC.GE: "ge", CC.LE: "le", CC.G: "g", CC.S: "s", CC.NS: "ns"}
MAX_CASES = 1024
MAX_INSTRUCTIONS = 4096
SCN_EXECUTE, SCN_READ, SCN_WRITE = 0x20000000, 0x40000000, 0x80000000
READS = {OpAccess.READ, OpAccess.COND_READ, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE}
WRITES = {OpAccess.WRITE, OpAccess.COND_WRITE, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE}
INFO = InstructionInfoFactory()


def unsigned(width):
    return f"std::uint{width * 8}_t"


def signed(width):
    return f"std::int{width * 8}_t"


def full(register):
    full_register = RegisterExt.full_register(register)
    if full_register not in NAMES:
        raise ValueError(f"unsupported register {register}")
    return full_register


def implicit_writes(ins):
    """Full registers that an instruction writes without naming them as operand 0. RDX counts for
    every one-operand multiply or divide, although an 8-bit multiply leaves it unchanged."""
    if ins.code in STRING_TRANSFERS:
        kind, _ = STRING_TRANSFERS[ins.code]
        return {R.RDI} | ({R.RSI} if kind == "movs" else set()) | ({R.RCX} if ins.has_rep_prefix else set())
    if ins.mnemonic in SIGN_EXTENSIONS:
        return {full(SIGN_EXTENSIONS[ins.mnemonic][1])}
    if ins.mnemonic in (M.MUL, M.IMUL, M.DIV, M.IDIV) and ins.op_count == 1:
        return {R.RAX, R.RDX}
    return set()


def extend(value, width, sign):
    """value (width bytes) widened to std::uint64_t. (x ^ s) - s sign-extends x from its top bit s;
    unsigned arithmetic wraps by definition, so no signed conversion or shift is involved."""
    value = f"static_cast<std::uint64_t>({value})"
    if not sign:
        return value
    top = 1 << width * 8 - 1
    return f"(({value} ^ {top:#x}ull) - {top:#x}ull)"


def runtime_functions(pe):
    table = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    if not table.Size:
        return []
    if table.Size % 12:
        raise ValueError("incomplete runtime-function record")
    return [(begin, end) for begin, end, _ in
            struct.iter_unpack("<III", image_bytes(pe, table.VirtualAddress, table.Size))]


def owned_ranges(pe, entry):
    """Unwind-owned intervals, or the export's section for a leaf without unwind data."""
    records = runtime_functions(pe)
    if any(begin <= entry < end for begin, end in records):
        return function_ranges(pe, entry), []
    # Windows x64 leaf functions may omit .pdata. Such a function may use any
    # file-backed bytes of its section that no runtime-function entry owns.
    section = pe.get_section_by_rva(entry)
    if not section or not section.Characteristics & 0x20000000:
        raise ValueError(f"export at RVA {entry:#x} is not in an executable section")
    end = section.VirtualAddress + min(section.Misc_VirtualSize, section.SizeOfRawData)
    return [(section.VirtualAddress, end)], records


def executable(pe, rva):
    section = pe.get_section_by_rva(rva)
    return bool(section and section.Characteristics & 0x20000000)


class Entries:
    """The RVAs of this image that a direct call or tail jump may treat as a function.

    An entry is the first byte of a verified primary runtime function (an unchained record with a
    version 1 or 2 unwind header, a non-empty interval and executable code) or a non-forwarded code
    export (an export outside the export directory, in an executable section and not inside any
    runtime-function interval except at a primary start). Chained secondary ranges, interior labels,
    forwarded exports and data exports are not entries."""

    MAX_CHAIN = 32

    def __init__(self, pe, bindings=None):
        table = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        data = image_bytes(pe, table.VirtualAddress, table.Size - table.Size % 12) if table.Size >= 12 else b""
        self.records = list(struct.iter_unpack("<III", data))
        self.primary = set()
        self.unverified = set()  # unchained records whose unwind header or interval fails the check
        self.parents = {}  # begin of a chained record -> begin of the record it continues
        for begin, end, info in self.records:
            try:
                header = image_bytes(pe, info, 4)
            except ValueError:
                self.unverified.add(begin)
                continue
            version, flags = header[0] & 7, header[0] >> 3
            if flags & 4:
                trailer = info + 4 + ((header[2] + 1) & ~1) * 2
                try:
                    self.parents[begin] = struct.unpack("<III", image_bytes(pe, trailer, 12))[0]
                except ValueError:
                    self.parents[begin] = None
            elif version in (1, 2) and begin < end and executable(pe, begin):
                self.primary.add(begin)
            else:
                self.unverified.add(begin)
        pe.parse_data_directories(directories=[0])
        directory = pe.OPTIONAL_HEADER.DATA_DIRECTORY[0]
        exports = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
        self.exports = {}
        for symbol in (exports.symbols if exports else ()):
            rva = symbol.address
            if not rva:
                continue
            if symbol.forwarder or directory.VirtualAddress <= rva < directory.VirtualAddress + directory.Size:
                kind = "forward"
            elif not executable(pe, rva):
                kind = "data"
            else:
                kind = "code"
            # A forwarder or data kind wins over code for the same RVA.
            if self.exports.get(rva) in (None, "code"):
                self.exports[rva] = kind
        self.entries = set(self.primary)
        self.entries.update(rva for rva, kind in self.exports.items()
                            if kind == "code" and self.owner(rva) is None)
        # The IAT slots of the ordinary import directory: a call through one, or a direct call or jump to a thunk
        # through one, goes to the import binding.
        pe.parse_data_directories(directories=[1])
        self.import_slots = {imp.address - pe.OPTIONAL_HEADER.ImageBase
                             for desc in getattr(pe, "DIRECTORY_ENTRY_IMPORT", ()) for imp in desc.imports}
        self.bound = {}  # start -> (end, name) of each verified range binding
        if bindings is not None:
            self.bind(pe, bindings)

    def bind(self, pe, bindings):
        """Verify a range manifest for this image: its hash, executable bounds and no overlap with
        another binding or a runtime function."""
        digest = hashlib.sha256(bytes(pe.__data__)).hexdigest()
        if bindings["image_sha256"] != digest:
            raise ValueError(f"range binding is for another image: manifest sha256 {bindings['image_sha256']}, "
                             f"image sha256 {digest}")
        previous = None
        for start, end, name in sorted(bindings["functions"]):
            section = pe.get_section_by_rva(start)
            if (not start < end or not section or not section.Characteristics & 0x20000000 or end >
                    section.VirtualAddress + min(section.Misc_VirtualSize, section.SizeOfRawData)):
                raise ValueError(f"range binding {name} [{start:#x}, {end:#x}) is not inside executable code")
            for begin, stop, _ in self.records:
                if begin < end and start < stop:
                    raise ValueError(f"range binding {name} [{start:#x}, {end:#x}) overlaps the runtime "
                                     f"function at {self.root(begin):#x}")
            if previous is not None and start < previous[1]:
                raise ValueError(f"range binding {name} [{start:#x}, {end:#x}) overlaps range binding "
                                 f"{previous[2]} [{previous[0]:#x}, {previous[1]:#x})")
            previous = (start, end, name)
            self.bound[start] = (end, name)

    def bound_range(self, rva):
        """(start, end, name) of the binding that starts at rva, or None."""
        if rva in self.bound:
            return (rva, *self.bound[rva])
        return None

    def bound_holder(self, rva):
        """(start, end, name) of the binding whose range holds rva, or None."""
        for start, (end, name) in self.bound.items():
            if start <= rva < end:
                return start, end, name
        return None

    def root(self, begin):
        for _ in range(self.MAX_CHAIN):
            if begin not in self.parents or self.parents[begin] is None:
                return begin
            begin = self.parents[begin]
        return begin

    def owner(self, rva):
        """The primary function whose runtime-function interval holds rva, or None."""
        for begin, end, _ in self.records:
            if begin <= rva < end:
                return self.root(begin)
        return None

    def __contains__(self, rva):
        return rva in self.entries or self.bound_range(rva) is not None

    def why_not(self, rva):
        kind = self.exports.get(rva)
        if kind == "forward":
            return "a forwarded export"
        if kind == "data":
            return "a data export in a non-executable section"
        if rva in self.parents:
            return f"the start of a chained range of the function at {self.root(rva):#x}"
        if rva in self.unverified:
            return "a runtime function whose unwind data is not version 1 or 2 in executable code"
        owner = self.owner(rva)
        if owner is not None:
            label = "an exported label" if kind else "an address"
            return f"{label} inside the function at {owner:#x}"
        holder = self.bound_holder(rva)
        if holder is not None:
            return f"inside the bound range {holder[2]} [{holder[0]:#x}, {holder[1]:#x}), not its start"
        return ("no runtime-function entry or export; a range binding [start, end) for this image "
                "(--bind-ranges) is needed")


class Function:
    def __init__(self, pe, entry, entries=None):
        self.pe = pe
        self.entry = entry
        self.entries = entries if entries is not None else Entries(pe)
        self.unwound = any(begin <= entry < end for begin, end in runtime_functions(pe))
        self.bound = self.entries.bound_range(entry)
        others = [(start, end) for start, (end, _) in self.entries.bound.items()
                  if self.bound is None or start != self.bound[0]]
        if self.bound is not None:
            # Trusted [start, end): every reachable edge must stay inside it.
            ranges, self.foreign = [self.bound[:2]], runtime_functions(pe) + others
        else:
            ranges, self.foreign = owned_ranges(pe, entry)
            if not self.unwound:
                # A leaf without unwind data may not enter another bound range either.
                self.foreign = self.foreign + others
        self.ranges = [(begin, end, image_bytes(pe, begin, end - begin, executable=True))
                       for begin, end in ranges]
        self.instructions = {}
        self.owner = {}
        self.ending = {}
        self.successors = {}
        self.dispatches = {}
        # Direct calls and tail jumps to other functions, by the RVA of the CALL or JMP.
        self.calls = {}
        self.tails = {}
        self.host_calls = {}

    def leaves_for(self, ins):
        """Whether a JMP out of this function is a tail jump. A target the image names as a function
        start (a runtime-function record or an export) that is not an entry is refused here."""
        target = ins.near_branch_target
        if target in self.entries:
            return True
        named = (target in self.entries.exports or any(begin == target for begin, _, _ in self.entries.records)
                 or self.entries.bound_holder(target) is not None)
        # Executable code outside every runtime function and binding: say which binding would be needed.
        unbound = executable(self.pe, target) and self.entries.owner(target) is None
        if (named or unbound) and target != self.entry:
            raise ValueError(f"jump target {target:#x} is not a function entry ({self.entries.why_not(target)}) "
                             f"at RVA {ins.ip:#x}")
        return False

    def owns(self, ip):
        return (any(begin <= ip < end for begin, end, _ in self.ranges)
                and not any(start <= ip < stop for start, stop in self.foreign))

    def callee(self, ins):
        """Validate the target of a direct CALL as the entry of another function in the image."""
        target = ins.near_branch_target
        if not 0 <= target < self.pe.OPTIONAL_HEADER.SizeOfImage:
            raise ValueError(f"call target outside the image at RVA {ins.ip:#x}")
        section = self.pe.get_section_by_rva(target)
        if target not in self.entries and self.entries.exports.get(target) in ("forward", "data"):
            raise ValueError(f"call target {target:#x} is not a function entry ({self.entries.why_not(target)}) "
                             f"at RVA {ins.ip:#x}")
        if not section or not section.Characteristics & 0x20000000:
            raise ValueError(f"call target {target:#x} is not in an executable section at RVA {ins.ip:#x}")
        if target not in self.entries:
            # A leaf without unwind data or an import thunk has no entry that the image records.
            raise ValueError(f"call target {target:#x} is not a function entry ({self.entries.why_not(target)}) "
                             f"at RVA {ins.ip:#x}")
        return target

    def decode(self, ip):
        for begin, end, data in self.ranges:
            if begin <= ip < end:
                ins = Decoder(64, data[ip - begin:], ip=ip).decode()
                if ins.is_invalid or ins.next_ip > end:
                    raise ValueError(f"invalid or truncated instruction at RVA {ip:#x}")
                if any(ip < stop and start < ins.next_ip for start, stop in self.foreign):
                    raise ValueError(f"edge outside owned ranges at RVA {ip:#x}: another function's unwind interval")
                return ins
        if self.bound is not None:
            # A bound function's edges must stay inside its range; name what the edge reaches instead.
            if any(begin <= ip < end for begin, end in runtime_functions(self.pe)):
                raise ValueError(f"edge outside owned ranges at RVA {ip:#x}: another function's unwind interval")
            holder = self.entries.bound_holder(ip)
            if holder is not None:
                raise ValueError(f"edge outside owned ranges at RVA {ip:#x}: the bound range {holder[2]}")
            raise ValueError(f"edge outside owned ranges at RVA {ip:#x}: outside the bound range {self.bound[2]}")
        raise ValueError(f"edge outside owned ranges at RVA {ip:#x}")

    def claim(self, begin, end, owner, what):
        for address in range(begin, end):
            if self.owner.get(address, owner) != owner:
                raise ValueError(f"{what} at RVA {begin:#x} overlaps RVA {self.owner[address]:#x}")
        for address in range(begin, end):
            self.owner[address] = owner

    def thunk_slot(self, ins):
        """Recognize the complete x64 FF 25 disp32 linker thunk in executable bytes."""
        target = ins.near_branch_target
        if not executable(self.pe, target):
            return None
        if self.pe.get_data(target, 2) != b"\xff\x25":
            return None
        if any(self.entries.owner(rva) is not None or self.entries.bound_holder(rva) is not None
               for rva in range(target, target + 6)):
            return None
        code = image_bytes(self.pe, target, 6, executable=True)
        return target + 6 + int.from_bytes(code[2:], "little", signed=True)

    def bound_import(self, resolver, slot, ip, what):
        """The import that the call or tail jump at ip reaches through IAT slot slot, given by the binding."""
        if resolver is None:
            raise ValueError(f"no binding for the {what} through IAT slot RVA {slot:#x} at RVA {ip:#x}")
        return resolver(slot)

    def explore(self, imports=None, *, callback_calls=False, import_resolver=None):
        pending = [self.entry]
        indirect = []
        while pending or indirect:
            if not pending:
                ip = indirect.pop()
                pending.extend(self.dispatch(ip))
                continue
            ip = pending.pop()
            if ip in self.instructions:
                continue
            if self.owner.get(ip, ip) != ip:
                raise ValueError(f"edge at RVA {ip:#x} overlaps an instruction")
            if len(self.instructions) >= MAX_INSTRUCTIONS:
                raise ValueError(f"instruction limit reached at RVA {ip:#x}")
            ins = self.decode(ip)
            self.claim(ip, ins.next_ip, ip, "instruction")
            self.instructions[ip] = ins
            self.ending[ins.next_ip] = ins
            flow = ins.flow_control
            if flow == FlowControl.NEXT:
                targets = [ins.next_ip]
            elif flow == FlowControl.CONDITIONAL_BRANCH and ins.is_jcc_short_or_near:
                targets = [ins.near_branch_target, ins.next_ip]
            elif (flow == FlowControl.UNCONDITIONAL_BRANCH and ins.is_jmp_short_or_near
                  and self.entries.import_slots and not self.owns(ins.near_branch_target)
                  and self.thunk_slot(ins) in self.entries.import_slots):
                # A jump to an import thunk is a tail jump through the thunk's slot.
                slot = self.thunk_slot(ins)
                self.host_calls[ip] = self.bound_import(
                    import_resolver, slot, ip, f"tail jump to the thunk at RVA {ins.near_branch_target:#x}")
                self.tails[ip] = slot
                targets = []
            elif (flow == FlowControl.UNCONDITIONAL_BRANCH and ins.is_jmp_short_or_near
                  and not self.owns(ins.near_branch_target) and self.leaves_for(ins)):
                # A jump to another function's entry outside this one is a tail call.
                self.tails[ip] = ins.near_branch_target
                targets = []
            elif flow == FlowControl.UNCONDITIONAL_BRANCH and ins.is_jmp_short_or_near:
                targets = [ins.near_branch_target]
            elif (ins.code == Code.JMP_RM64 and ins.memory_base == R.RIP and not ins.has_segment_prefix):
                slot = ins.ip_rel_memory_address
                self.host_calls[ip] = self.bound_import(import_resolver, slot, ip, "tail jump")
                self.tails[ip] = slot
                targets = []
            elif flow == FlowControl.CALL and ins.is_call_near:
                slot = self.thunk_slot(ins) if imports or self.entries.import_slots else None
                if slot is not None and imports and slot in imports:
                    self.host_calls[ip] = imports[slot]
                    self.calls[ip] = slot
                elif slot in self.entries.import_slots:
                    # A call of an import thunk is a call through the thunk's slot.
                    self.host_calls[ip] = self.bound_import(
                        import_resolver, slot, ip, f"call of the thunk at RVA {ins.near_branch_target:#x}")
                    self.calls[ip] = slot
                else:
                    self.calls[ip] = self.callee(ins)
                targets = [ins.next_ip]
            elif (ins.code == Code.CALL_RM64 and imports and ins.memory_base == R.RIP
                  and not ins.has_segment_prefix and ins.ip_rel_memory_address in imports):
                self.host_calls[ip] = imports[ins.ip_rel_memory_address]
                self.calls[ip] = ins.ip_rel_memory_address
                targets = [ins.next_ip]
            elif (ins.code == Code.CALL_RM64 and ins.memory_base == R.RIP and not ins.has_segment_prefix
                  and ins.ip_rel_memory_address in self.entries.import_slots):
                # An ordinary call through an IAT slot.
                slot = ins.ip_rel_memory_address
                self.host_calls[ip] = self.bound_import(import_resolver, slot, ip, "call")
                self.calls[ip] = slot
                targets = [ins.next_ip]
            elif (callback_calls and ins.code == Code.CALL_RM64 and not ins.has_segment_prefix
                  and ins.op0_kind == OpKind.MEMORY and ins.memory_base != R.RIP):
                # The callback-aware emitter must check the loaded pointer against its typed bindings.
                self.calls[ip] = None
                targets = [ins.next_ip]
            elif flow == FlowControl.RETURN and ins.mnemonic == M.RET and (ins.op_count == 0 or ins.immediate(0) == 0):
                targets = []
            elif flow == FlowControl.INDIRECT_BRANCH and ins.op0_kind == OpKind.REGISTER:
                targets = []
                indirect.append(ip)
            else:
                raise ValueError(f"unsupported control flow at RVA {ip:#x}: {ins}")
            if ip in self.host_calls and self.host_calls[ip].noreturn:
                targets = []
            self.successors[ip] = targets
            pending.extend(reversed(targets))
        edges = {target for ip, targets in self.successors.items() for target in targets
                 if target != self.instructions[ip].next_ip
                 or self.instructions[ip].flow_control not in STRAIGHT}
        # A conditional branch to its own fallthrough still emits a goto, so its target needs a label.
        edges |= {ins.near_branch_target for ins in self.instructions.values()
                  if ins.flow_control == FlowControl.CONDITIONAL_BRANCH}
        constants = self.constants()
        for ip in self.dispatches:
            self.prove(ip, edges, constants)
        return edges

    def previous(self, ins, what):
        before = self.ending.get(ins.ip)
        if before is None or before.flow_control != FlowControl.NEXT:
            raise ValueError(f"missing {what} before RVA {ins.ip:#x}")
        return before

    def constants(self):
        """RIP-relative addresses that every explored path leaves in a register, per instruction."""
        state = {self.entry: {}}
        work = [self.entry]
        while work:
            ip = work.pop()
            ins = self.instructions.get(ip)
            if ins is None:
                continue
            known = dict(state[ip])
            if ip in self.dispatches:
                known.pop(self.dispatches[ip][4], None)
            elif ip in self.calls:
                known = {r: v for r, v in known.items() if r not in VOLATILE}
            elif ins.mnemonic == M.LEA and ins.is_ip_rel_memory_operand and RegisterExt.is_gpr64(ins.op0_register):
                known[ins.op0_register] = ins.ip_rel_memory_address
            elif ins.op_count and ins.op0_kind == OpKind.REGISTER and ins.mnemonic not in (M.CMP, M.TEST):
                known.pop(full(ins.op0_register), None)
            for register in implicit_writes(ins):
                known.pop(register, None)
            for successor in self.successors.get(ip, ()):
                if successor not in self.instructions:
                    continue
                if successor == self.entry:
                    merged = {}
                elif successor in state:
                    merged = {r: v for r, v in state[successor].items() if known.get(r) == v}
                else:
                    merged = known
                if successor not in state or merged != state[successor]:
                    state[successor] = merged
                    work.append(successor)
        return state

    def dispatch(self, ip):
        jump = self.instructions[ip]
        add = self.previous(jump, "table address add")
        load = self.previous(add, "table entry load")
        target = full(jump.op0_register)
        base = add.op1_register if add.op1_kind == OpKind.REGISTER else None
        if (add.mnemonic != M.ADD or add.op0_kind != OpKind.REGISTER or add.op0_register != target
                or not RegisterExt.is_gpr64(target) or base is None or not RegisterExt.is_gpr64(base)
                or base == target):
            raise ValueError(f"unrecognized indirect jump at RVA {ip:#x}")
        index = load.memory_index
        if (load.op1_kind != OpKind.MEMORY or load.memory_base != base or load.memory_index_scale != 4
                or not RegisterExt.is_gpr64(index) or index == base or load.has_segment_prefix
                or full(load.op0_register) != target):
            raise ValueError(f"unrecognized table load at RVA {load.ip:#x}")
        anchor = self.constants().get(load.ip, {}).get(base)
        if anchor is None:
            raise ValueError(f"table base {NAMES[base]} is not a RIP-relative address at RVA {load.ip:#x}")
        if load.mnemonic == M.MOV and RegisterExt.is_gpr32(load.op0_register):
            # Image-relative entries: the base register holds the image base.
            table, kind = load.memory_displacement, "image"
            if anchor != 0 or not 0 < table < 1 << 31:
                raise ValueError(f"image-relative table without an image base at RVA {load.ip:#x}")
        elif load.mnemonic == M.MOVSXD and load.memory_displacement == 0:
            table, kind = anchor, "relative"
        else:
            raise ValueError(f"unrecognized table load at RVA {load.ip:#x}")

        # The bound check falls through into the dispatch block, or (inverted) branches to it with a
        # JMP to the default in front of it.
        bound = load
        between = []
        inverted = False
        while True:
            bound = self.ending.get(bound.ip)
            if bound is None:
                raise ValueError(f"unbounded jump-table index at RVA {ip:#x}")
            if bound.flow_control == FlowControl.CONDITIONAL_BRANCH:
                break
            if (bound.flow_control == FlowControl.UNCONDITIONAL_BRANCH and bound.is_jmp_short_or_near
                    and not inverted):
                inverted = True
                first = between[-1] if between else load
                bound = self.ending.get(bound.ip)
                if (bound is None or bound.flow_control != FlowControl.CONDITIONAL_BRANCH
                        or not bound.is_jcc_short_or_near or bound.near_branch_target != first.ip):
                    raise ValueError(f"unbounded jump-table index at RVA {ip:#x}")
                break
            if bound.flow_control != FlowControl.NEXT or len(between) >= 8:
                raise ValueError(f"unbounded jump-table index at RVA {ip:#x}")
            between.append(bound)
        compare = self.ending.get(bound.ip)
        # The branch that enters the table is the one the index passed: JA/JAE when the default is
        # taken, JB/JBE when the table is.
        if bound.condition_code not in ((CC.B, CC.BE) if inverted else (CC.A, CC.AE)):
            raise ValueError(f"jump-table bound is not an unsigned upper limit at RVA {bound.ip:#x}")
        if (compare is None or compare.mnemonic != M.CMP or compare.op0_kind != OpKind.REGISTER
                or compare.op1_kind not in (OpKind.IMMEDIATE8TO32, OpKind.IMMEDIATE8TO64,
                                            OpKind.IMMEDIATE32, OpKind.IMMEDIATE32TO64)
                or RegisterExt.size(compare.op0_register) not in (4, 8)):
            raise ValueError(f"unbounded jump-table index at RVA {bound.ip:#x}")
        limit = compare.immediate(1) & ((1 << RegisterExt.size(compare.op0_register) * 8) - 1)
        count = limit + 1 if bound.condition_code in (CC.A, CC.BE) else limit
        if not 0 < count <= MAX_CASES:
            raise ValueError(f"unsupported jump-table size {count} at RVA {compare.ip:#x}")
        entries = struct.unpack(f"<{count}{'I' if kind == 'image' else 'i'}",
                                image_bytes(self.pe, table, 4 * count))
        self.claim(table, table + 4 * count, -table, "jump table")
        targets = [entry if kind == "image" else (table + entry) & 0xffffffff for entry in entries]
        self.dispatches[ip] = (index, targets, (load.ip, add.ip), list(reversed(between)),
                               target, (compare, bound, base, anchor))
        self.successors[ip] = targets
        return list(reversed(targets))

    def prove(self, ip, edges, constants):
        index, _, sequence, window, _, (compare, bound, base, anchor) = self.dispatches[ip]
        block = [*(i.ip for i in window), *sequence, ip]
        if bound.condition_code in (CC.B, CC.BE):
            # The inverted bound branch is the one way into the block, and it enters at the top.
            entries = {source for source, targets in self.successors.items() if block[0] in targets}
            if entries != {bound.ip}:
                raise ValueError(f"second entry into the dispatch block at RVA {ip:#x}")
            block = block[1:]
        if any(address in edges for address in (bound.ip, *block)):
            raise ValueError(f"second entry into the dispatch block at RVA {ip:#x}")
        if constants.get(sequence[0], {}).get(base) != anchor:
            raise ValueError(f"table base {NAMES[base]} differs on another path to RVA {sequence[0]:#x}")
        # Extend the straight-line window back from the compare while it has a single entry.
        start = compare
        prefix = []
        while start.ip not in edges and start.ip != self.entry and len(prefix) < 16:
            before = self.ending.get(start.ip)
            if before is None or before.flow_control != FlowControl.NEXT:
                break
            prefix.append(before)
            start = before
        values = {}
        counter = iter(range(1 << 30))

        def value(reg):
            if reg not in values:
                values[reg] = (next(counter), "full")
            return values[reg]

        def step(ins):
            for register in implicit_writes(ins):
                values[register] = (next(counter), "full")
            if ins.op_count == 0 or ins.op0_kind != OpKind.REGISTER:
                return
            dst = ins.op0_register
            width = RegisterExt.size(dst)
            if ins.mnemonic == M.CMP:
                return
            if ins.mnemonic == M.MOV and ins.op1_kind == OpKind.REGISTER and width in (4, 8):
                number, form = value(full(ins.op1_register))
                values[full(dst)] = (number, form) if width == 8 else (number, "zext32")
            elif (ins.mnemonic == M.MOVSXD and ins.op1_kind == OpKind.REGISTER
                  and RegisterExt.size(ins.op1_register) == 4):
                values[full(dst)] = (value(full(ins.op1_register))[0], "sext32")
            elif ins.mnemonic not in (M.CMP, M.TEST):
                values[full(dst)] = (next(counter), "zext32" if width == 4 else "full")

        for ins in reversed(prefix):
            step(ins)
        bounded, bounded_form = value(full(compare.op0_register))
        compared_width = RegisterExt.size(compare.op0_register)
        for ins in window:
            step(ins)
        number, form = value(index)
        if number != bounded or not (form in ("zext32", "sext32") or (compared_width == 8 and form == bounded_form)):
            raise ValueError(f"jump-table index {NAMES[index]} is not the bounded value at RVA {ip:#x}")


def extent(section):
    """Bytes that a mapping gives a section, as Windows and tools/pe_map.cpp map it: VirtualSize, or
    SizeOfRawData when VirtualSize is 0. Bytes past the file-backed part are zero-filled."""
    return section.Misc_VirtualSize or section.SizeOfRawData


def image_target(pe, ins):
    """The RVA that a RIP-relative operand addresses in the bound mapping, checked against the image."""
    rva = ins.ip_rel_memory_address
    if rva >= pe.OPTIONAL_HEADER.SizeOfImage:
        raise ValueError(f"RIP-relative target {rva:#x} outside the image at RVA {ins.ip:#x}")
    if ins.mnemonic == M.LEA:
        # An address, not an access: it may name any byte of the image, such as __ImageBase in the headers.
        return rva
    width = _memory_width(ins)
    section = next((s for s in pe.sections if s.VirtualAddress <= rva < s.VirtualAddress + extent(s)), None)
    if section is None:
        raise ValueError(f"RIP-relative target {rva:#x} outside every section at RVA {ins.ip:#x}")
    name = section.Name.rstrip(b"\0").decode("ascii", "replace")
    if rva + width > section.VirtualAddress + extent(section):
        raise ValueError(f"RIP-relative access of {width} bytes at {rva:#x} crosses the end of section {name} "
                         f"at RVA {ins.ip:#x}")
    access = INFO.info(ins).op_access(next(n for n in range(ins.op_count) if ins.op_kind(n) == OpKind.MEMORY))
    flags = section.Characteristics
    if access in READS and not flags & SCN_READ:
        raise ValueError(f"RIP-relative read from section {name} without read access at RVA {ins.ip:#x}")
    # The translation was made from the original code bytes and cannot follow a write into code.
    if access in WRITES and (not flags & SCN_WRITE or flags & SCN_EXECUTE):
        kind = "executable" if flags & SCN_WRITE else "read-only"
        raise ValueError(f"RIP-relative write to {kind} section {name} at RVA {ins.ip:#x}")
    return rva


class Lifter:
    def __init__(self, image=None, *, host_cpu=False):
        self.used = set()
        self.flags = False
        self.host_cpu = host_cpu
        # The input image when RIP-relative operands address the bound mapping, and whether one does.
        self.image = image
        self.image_used = False
        self.frame = {}  # RVA -> index into the frame array of the stack bytes it accesses (from Frame)
        self.adjustments = set()  # RVAs of SUB or ADD RSP, imm
        self.captured = {}  # RVA -> declaration of the local that holds the instruction's one guest read

    def slot(self, ins):
        if ins.ip not in self.frame:
            raise ValueError(f"stack access outside the frame model at RVA {ins.ip:#x}")
        return f"reinterpret_cast<word>(frame + {self.frame[ins.ip]})"

    def reg(self, register):
        width = RegisterExt.size(register)
        if register in HIGH_BYTES or width not in (1, 2, 4, 8):
            raise ValueError(f"unsupported register {register}")
        name = NAMES[full(register)]
        self.used.add(full(register))
        return name if width == 8 else f"static_cast<{unsigned(width)}>({name})"

    def address(self, ins):
        if ins.memory_base == R.RIP and self.image is not None and not ins.has_segment_prefix:
            self.image_used = True
            return f"(image + {image_target(self.image, ins):#x}ull)"
        if ins.ip in self.frame:
            # A constant index into the frame array: stack bases never become C++ values.
            return self.slot(ins)
        if (ins.memory_base in (R.RIP, R.EIP, R.RSP, R.ESP) or ins.memory_index in (R.RSP, R.ESP)
                or ins.has_segment_prefix or ins.memory_base == R.NONE):
            raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}")
        parts = [self.reg(ins.memory_base)]
        if ins.memory_index != R.NONE:
            if not RegisterExt.is_gpr64(ins.memory_index):
                raise ValueError(f"unsupported memory index at RVA {ins.ip:#x}")
            parts.append(f"{self.reg(ins.memory_index)} * {ins.memory_index_scale}u")
        displacement = ins.memory_displacement & ((1 << 64) - 1)
        if displacement:
            parts.append(f"{displacement:#x}ull")
        if RegisterExt.is_gpr32(ins.memory_base):
            # A 67h prefix selects 32-bit addressing: the sum wraps at 32 bits and is zero-extended.
            return f"static_cast<std::uint64_t>(static_cast<std::uint32_t>({' + '.join(parts)}))"
        return "(" + " + ".join(parts) + ")"

    def capture(self, ins, text):
        """The statement of ins with its guest read, if any, performed first and exactly once."""
        declaration = self.captured.pop(ins.ip, None)
        if declaration is None:
            return text
        # A read whose value nothing uses (CMP or TEST without a flag consumer) is still performed.
        parts = [declaration, text] if re.search(r"\bmem\b", text) else [declaration, "static_cast<void>(mem);", text]
        return "{ " + " ".join(part for part in parts if part) + " }"

    def operand(self, ins, number, width):
        kind = ins.op_kind(number)
        if kind == OpKind.REGISTER:
            return self.reg(ins.op_register(number))
        if kind == OpKind.MEMORY:
            if ins.ip in self.frame:
                # The function's own frame array: a read of it cannot fault.
                return f"load<{unsigned(width)}>({self.address(ins)})"
            # The instruction's memory operand is read once, at its own width, into a local declared before
            # the rest of the statement (see capture()); every use of the operand names that local. The read
            # is volatile: the original faults on an inaccessible operand even when the value is never
            # used, and the host compiler may drop an ordinary load whose value is unused.
            declaration = f"const {unsigned(width)} mem = load_always<{unsigned(width)}>({self.address(ins)});"
            if self.captured.setdefault(ins.ip, declaration) != declaration:
                raise ValueError(f"memory operand read at two widths at RVA {ins.ip:#x}")
            return "mem"
        if kind in (OpKind.IMMEDIATE8, OpKind.IMMEDIATE8TO16, OpKind.IMMEDIATE8TO32, OpKind.IMMEDIATE8TO64,
                    OpKind.IMMEDIATE16, OpKind.IMMEDIATE32, OpKind.IMMEDIATE32TO64, OpKind.IMMEDIATE64):
            return f"static_cast<{unsigned(width)}>({ins.immediate(number) & ((1 << width * 8) - 1):#x}ull)"
        raise ValueError(f"unsupported operand at RVA {ins.ip:#x}")

    def write(self, ins, width, expression, register=None):
        """C++ that writes expression to operand 0, or to register when one is given."""
        if register is None:
            if ins.op0_kind == OpKind.MEMORY:
                return f"store<{unsigned(width)}>({self.address(ins)}, static_cast<{unsigned(width)}>({expression}));"
            register = ins.op0_register
        if register in HIGH_BYTES:
            raise ValueError(f"unsupported register {register}")
        if width not in (1, 2, 4, 8) or RegisterExt.size(register) != width:
            raise ValueError(f"unsupported partial register write at RVA {ins.ip:#x}")
        self.used.add(full(register))
        name = NAMES[full(register)]
        if width < 4:
            # An 8- or 16-bit destination leaves the rest of the register unchanged.
            mask = (1 << width * 8) - 1
            return f"{name} = ({name} & ~{mask:#x}ull) | static_cast<{unsigned(width)}>({expression});"
        return f"{name} = static_cast<{unsigned(width)}>({expression});"

    def width(self, ins):
        if ins.op0_kind == OpKind.REGISTER:
            return RegisterExt.size(ins.op0_register)
        if ins.op0_kind == OpKind.MEMORY:
            return _memory_width(ins)
        return 0

    def condition(self, producer, jcc):
        code = CONDITIONS.get(jcc.condition_code)
        if producer.mnemonic in BIT_TESTS:
            # fb is CF, the selected bit. ZF is unaffected and OF, SF, AF and PF are undefined.
            if code not in ("b", "ae"):
                raise ValueError(f"only CF is defined after {str(producer).split()[0].upper()} at RVA {jcc.ip:#x}")
            return "fb != 0" if code == "b" else "fb == 0"
        if producer.mnemonic in BIT_SCANS:
            # fa is the source. ZF is set for a zero source; CF, OF, SF, PF and AF are undefined.
            if code not in ("e", "ne"):
                name = str(producer).split()[0].upper()
                raise ValueError(f"only ZF is defined after {name} at RVA {jcc.ip:#x}")
            source_zero = f"static_cast<{unsigned(self.width(producer))}>(fa) == 0"
            return source_zero if code == "e" else f"!({source_zero})"
        if producer.mnemonic in BIT_COUNTS:
            # fa is the source and fb the result.
            source_zero = f"static_cast<{unsigned(self.width(producer))}>(fa) == 0"
            if producer.mnemonic == M.POPCNT:
                # ZF is set for a zero source; OF, SF, AF, CF and PF are cleared.
                table = {"e": source_zero, "be": source_zero, "le": source_zero, "o": "false", "b": "false",
                         "s": "false", "p": "false", "l": "false"}
            else:
                result_zero = f"static_cast<{unsigned(self.width(producer))}>(fb) == 0"
                table = {"b": source_zero, "e": result_zero, "be": f"{source_zero} || {result_zero}"}
            for negated, base in (("no", "o"), ("ae", "b"), ("ne", "e"), ("a", "be"), ("ns", "s"), ("np", "p"),
                                  ("ge", "l"), ("g", "le")):
                if base in table:
                    table[negated] = f"!({table[base]})"
            if code not in table:
                name = str(producer).split()[0].upper()
                raise ValueError(f"OF, SF and PF are undefined after {name} at RVA {jcc.ip:#x}")
            return table[code]
        width = self.width(producer)
        u, s = unsigned(width), signed(width)
        a, b = f"static_cast<{u}>(fa)", f"static_cast<{u}>(fb)"
        sign = f"{1 << (width * 8 - 1):#x}ull"
        if producer.mnemonic in (M.CMP, M.SUB):
            flag_result = f"static_cast<{u}>({a} - {b})"
            overflow = f"((({a} ^ {b}) & ({a} ^ {flag_result}) & {sign}) != 0)"
            table = {"e": f"{a} == {b}", "ne": f"{a} != {b}", "b": f"{a} < {b}", "ae": f"{a} >= {b}",
                     "be": f"{a} <= {b}", "a": f"{a} > {b}", "l": f"static_cast<{s}>({a}) < static_cast<{s}>({b})",
                     "ge": f"static_cast<{s}>({a}) >= static_cast<{s}>({b})",
                     "le": f"static_cast<{s}>({a}) <= static_cast<{s}>({b})",
                     "g": f"static_cast<{s}>({a}) > static_cast<{s}>({b})",
                     "s": f"static_cast<{s}>({flag_result}) < 0", "ns": f"static_cast<{s}>({flag_result}) >= 0"}
        elif producer.mnemonic in SHIFTS:
            # fa is the result; bit 0 of fb is CF and bit 1 is OF (see lower()).
            flag_result = f"static_cast<{u}>(fa)"
            carry, zero = "((fb & 1u) != 0)", f"{flag_result} == 0"
            s_cond = f"static_cast<{s}>({flag_result}) < 0"
            table = {"e": zero, "ne": f"{flag_result} != 0", "s": s_cond,
                     "ns": f"static_cast<{s}>({flag_result}) >= 0",
                     "b": carry, "ae": f"!{carry}", "be": f"{carry} || {zero}", "a": f"!{carry} && !({zero})"}
            overflow = None
            if shift_count(producer) == 1:
                overflow = "((fb & 2u) != 0)"
                table.update({"l": f"({s_cond}) != {overflow}", "ge": f"({s_cond}) == {overflow}",
                              "le": f"{zero} || ({s_cond}) != {overflow}",
                              "g": f"!({zero}) && ({s_cond}) == {overflow}"})
            elif code in ("o", "no", "l", "ge", "le", "g"):
                raise ValueError(f"OF is undefined after a shift by more than 1 at RVA {jcc.ip:#x}")
        else:
            if producer.mnemonic == M.ADD:
                flag_result = f"static_cast<{u}>({a} + {b})"
                overflow = f"((~({a} ^ {b}) & ({a} ^ {flag_result}) & {sign}) != 0)"
            else:
                flag_result = f"static_cast<{u}>(fa)"
                if producer.mnemonic in (M.INC, M.NEG):
                    overflow = f"{flag_result} == {sign}"
                elif producer.mnemonic == M.DEC:
                    overflow = f"{flag_result} == {((1 << (width * 8 - 1)) - 1):#x}ull"
                else:
                    overflow = "false"  # TEST, AND, OR and XOR clear OF.
            s_cond = f"static_cast<{s}>({flag_result}) < 0"
            ns_cond = f"static_cast<{s}>({flag_result}) >= 0"
            table = {"e": f"{flag_result} == 0", "ne": f"{flag_result} != 0", "s": s_cond, "ns": ns_cond}
            if producer.mnemonic in (M.TEST, M.AND, M.OR, M.XOR):
                # Logical operations clear CF and OF.
                table.update({"b": "false", "ae": "true", "be": f"{flag_result} == 0", "a": f"{flag_result} != 0",
                              "l": s_cond, "ge": ns_cond,
                              "le": f"{flag_result} == 0 || {s_cond}", "g": f"{flag_result} != 0 && {ns_cond}"})
            else:
                # ADD, INC, DEC and NEG share SF != OF signed comparisons; only ADD and NEG define CF.
                table.update({"l": f"({s_cond}) != ({overflow})", "ge": f"({s_cond}) == ({overflow})",
                              "le": f"{flag_result} == 0 || ({s_cond}) != ({overflow})",
                              "g": f"{flag_result} != 0 && ({s_cond}) == ({overflow})"})
                if producer.mnemonic == M.ADD:
                    table.update({"b": f"{flag_result} < {a}", "ae": f"{flag_result} >= {a}",
                                  "be": f"{flag_result} < {a} || {flag_result} == 0",
                                  "a": f"{flag_result} >= {a} && {flag_result} != 0"})
                elif producer.mnemonic == M.NEG:
                    table.update({"b": f"{flag_result} != 0", "ae": f"{flag_result} == 0",
                                  "be": "true", "a": "false"})
        # PF is even parity of only the low result byte. OF is signed overflow,
        # independent of the host's signed arithmetic and its overflow rules.
        table.update({"p": f"even_parity({flag_result})", "np": f"!even_parity({flag_result})"})
        if overflow is not None:
            table.update({"o": overflow, "no": f"!({overflow})"})
        if code not in table:
            raise ValueError(f"unsupported condition after {producer.mnemonic} at RVA {jcc.ip:#x}")
        return table[code]

    def statement(self, ins, sets_flags=False, producer=None):
        """lower(), with the instruction's guest memory read performed first and exactly once."""
        text, reads, writes = self.lower(ins, sets_flags, producer)
        return self.capture(ins, text), reads, writes

    def lower(self, ins, sets_flags=False, producer=None):
        """Return C++ for one instruction, the low bytes it reads from each full register and the
        width it writes to each full register.

        sets_flags records the flag inputs for a later consumer; producer is the
        instruction whose flags a SETcc or CMOVcc reads."""
        mnemonic = ins.mnemonic
        if mnemonic == M.CLZERO:
            if ins.len != 3:
                raise ValueError(f"unsupported CLZERO prefix or address size at RVA {ins.ip:#x}")
            address = self.reg(R.RAX)
            # One aligned cache line per executed instruction. Volatile retains writes
            # even when a following iteration or caller overwrites the same line.
            return (f"{{ auto* line = reinterpret_cast<volatile unsigned char*>({address} & ~63ull); "
                    "for (unsigned i = 0; i < 64; ++i) line[i] = 0; }", {R.RAX: 8}, {})
        if mnemonic in (M.RDFSBASE, M.RDGSBASE, M.RDPID):
            if not self.host_cpu:
                raise ValueError(f"{str(ins).split()[0].upper()} requires host_cpu=True (--host-cpu) at RVA {ins.ip:#x}")
            if (ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix
                    or ins.has_segment_prefix or ins.has_xacquire_prefix or ins.has_xrelease_prefix):
                raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}")
            if ins.op_count != 1 or ins.op0_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported operand at RVA {ins.ip:#x}: {ins}")
            dest_reg = ins.op0_register
            if full(dest_reg) == R.RSP:
                raise ValueError(f"unsupported use of rsp at RVA {ins.ip:#x}")
            if mnemonic == M.RDPID:
                if ins.code != Code.RDPID_R64:
                    raise ValueError(f"unsupported RDPID encoding at RVA {ins.ip:#x}: {ins}")
                line = f"{{ const volatile std::uint32_t pid = _rdpid_u32(); {self.write(ins, 8, 'pid')} }}"
                return line, {}, {full(dest_reg): 8}
            if ins.code not in HOST_CPU_READ_BASE_CODES:
                raise ValueError(f"unsupported FSGSBASE encoding at RVA {ins.ip:#x}: {ins}")
            intrinsic, op_width = HOST_CPU_READ_BASE_CODES[ins.code]
            u = unsigned(op_width)
            line = f"{{ const volatile {u} base = {intrinsic}(); {self.write(ins, op_width, 'base')} }}"
            return line, {}, {full(dest_reg): op_width}
        if ins.code in STRING_TRANSFERS:
            kind, width = STRING_TRANSFERS[ins.code]
            if (ins.has_segment_prefix or ins.has_repne_prefix or ins.has_lock_prefix
                    or any(ins.op_kind(n) not in (OpKind.REGISTER, OpKind.MEMORY_ESRDI, OpKind.MEMORY_SEG_RSI)
                           for n in range(ins.op_count))):
                raise ValueError(f"unsupported string addressing or prefix at RVA {ins.ip:#x}")
            registers = {R.RDI} | ({R.RSI} if kind == "movs" else set())
            self.used.update(registers)
            reads = dict.fromkeys(registers, 8)
            writes = dict.fromkeys(registers, 8)
            if kind == "movs":
                value = f"load_always<{unsigned(width)}>(rsi)"
            else:
                self.used.add(R.RAX)
                reads[R.RAX] = width
                value = f"static_cast<{unsigned(width)}>(rax)"
            # One element at a time: neither memcpy nor memmove reproduces a forward
            # overlapping MOVS that reads the bytes just written by earlier iterations.
            alignment = " __unaligned" if width > 1 else ""
            step = (f"*reinterpret_cast<volatile {unsigned(width)}{alignment}*>(rdi) = {value}; "
                    + ("rsi += step; " if kind == "movs" else "") + "rdi += step;")
            if ins.has_rep_prefix:
                self.used.add(R.RCX)
                reads[R.RCX] = writes[R.RCX] = 8
                step = f"while (rcx != 0) {{ {step} --rcx; }}"
            return f"{{ const word step = direction ? 0ull - {width}ull : {width}ull; {step} }}", reads, writes
        if mnemonic in (M.CLD, M.STD):
            return f"direction = {'true' if mnemonic == M.STD else 'false'};", {}, {}
        width = self.width(ins)
        reads, writes = {}, {}

        def read(register):
            reads[full(register)] = max(reads.get(full(register), 0), RegisterExt.size(register))

        def note(*numbers):
            for number in numbers:
                kind = ins.op_kind(number)
                if kind == OpKind.REGISTER:
                    read(ins.op_register(number))
                elif kind == OpKind.MEMORY and ins.ip not in self.frame:
                    for register in (ins.memory_base, ins.memory_index):
                        if register not in (R.NONE, R.RSP) and (register != R.RIP or self.image is None):
                            read(register)

        if ins.op_count and ins.op0_kind == OpKind.REGISTER:
            writes[full(ins.op0_register)] = width
        elif ins.op_count and ins.op0_kind == OpKind.MEMORY:
            note(0)
        if mnemonic == M.NOP:
            return "", {}, {}
        if mnemonic in (M.PUSH, M.POP):
            # Frame accepts only PUSH and POP of a 64-bit nonvolatile register.
            if mnemonic == M.PUSH:
                note(0)
                return f"store<std::uint64_t>({self.slot(ins)}, {self.reg(ins.op0_register)});", reads, {}
            return self.write(ins, 8, f"load<std::uint64_t>({self.slot(ins)})"), {}, writes
        if mnemonic == M.LEAVE:
            # Frame proved the anchor: RSP stays a constant offset, and RBP is reloaded from the anchor slot.
            return self.write(ins, 8, f"load<std::uint64_t>({self.slot(ins)})", R.RBP), {}, {R.RBP: 8}
        if ins.ip in self.adjustments:
            # SUB or ADD RSP, imm: Frame knows RSP as a constant offset, so nothing is computed.
            if sets_flags:
                raise ValueError(f"flags of a stack adjustment are read at RVA {ins.ip:#x}")
            return "", {}, {}
        if mnemonic == M.MOV:
            note(1)
            return self.write(ins, width, self.operand(ins, 1, width)), reads, writes
        if mnemonic in (M.MOVZX, M.MOVSX, M.MOVSXD):
            note(1)
            source_width = (RegisterExt.size(ins.op1_register) if ins.op1_kind == OpKind.REGISTER
                            else _memory_width(ins))
            value = self.operand(ins, 1, source_width)
            if mnemonic != M.MOVZX:
                value = f"static_cast<{unsigned(width)}>(static_cast<{signed(width)}>(static_cast<{signed(source_width)}>({value})))"
            return self.write(ins, width, value), reads, writes
        if mnemonic == M.LEA:
            note(1)
            return self.write(ins, width, self.address(ins)), reads, writes
        if mnemonic in (*ALU, M.CMP, M.TEST):
            same = (mnemonic in (M.XOR, M.SUB) and ins.op0_kind == ins.op1_kind == OpKind.REGISTER
                    and ins.op0_register == ins.op1_register)
            if not same:
                note(0, 1)
            a = self.operand(ins, 0, width) if not same else f"static_cast<{unsigned(width)}>(0)"
            b = self.operand(ins, 1, width) if not same else a
            operator = ALU.get(mnemonic, "&")
            lines = []
            if sets_flags:
                self.flags = True
                if mnemonic in (M.CMP, M.SUB, M.ADD):
                    lines.append(f"fa = {a}; fb = {b};")
                else:
                    lines.append(f"fa = static_cast<{unsigned(width)}>({a} {operator} {b});")
            if mnemonic in ALU:
                lines.append(self.write(ins, width, f"{a} {operator} {b}"))
            else:
                writes.clear()
            return " ".join(lines), reads, writes
        if mnemonic in (M.INC, M.DEC, M.NEG, M.NOT):
            note(0)
            a = self.operand(ins, 0, width)
            value = {M.INC: f"{a} + 1u", M.DEC: f"{a} - 1u", M.NEG: f"0u - {a}", M.NOT: f"~{a}"}[mnemonic]
            line = self.write(ins, width, value)
            if sets_flags:
                if mnemonic not in (M.INC, M.DEC, M.NEG):
                    raise ValueError(f"unsupported flag producer at RVA {ins.ip:#x}")
                self.flags = True
                line = f"fa = static_cast<{unsigned(width)}>({value}); " + line
            return line, reads, writes
        if mnemonic in BIT_TESTS:
            if ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                # Frame and image proofs cover the nominal address only; the dynamic bit offset
                # can reach outside that range. Address-size overrides need separate wrap proof.
                if (ins.ip in self.frame or ins.memory_base in (R.RIP, R.EIP)
                        or RegisterExt.is_gpr32(ins.memory_base)):
                    raise ValueError(f"unsupported bit-string operand at RVA {ins.ip:#x}")
                note(0, 1)
                bits = width * 8
                shift = bits.bit_length() - 1
                raw = self.operand(ins, 1, width)
                # Unsigned arithmetic implements a sign-extended arithmetic shift without
                # signed overflow or an implementation-defined right shift of a negative value.
                lines = [f"const std::uint64_t bit_index = {raw};",
                         f"const std::uint64_t sign = 0ull - (bit_index >> {bits - 1});",
                         f"const std::uint64_t word_index = (bit_index >> {shift}) | (sign << {bits - shift});",
                         f"const word address = {self.address(ins)} + word_index * {width}ull;",
                         f"const {unsigned(width)} value = load_always<{unsigned(width)}>(address);",
                         f"const std::uint64_t bit = bit_index & {bits - 1}u;"]
                if sets_flags:
                    self.flags = True
                    lines.append("fb = (static_cast<std::uint64_t>(value) >> bit) & 1u;")
                if mnemonic != M.BT:
                    lines.append(f"store<{unsigned(width)}>(address, static_cast<{unsigned(width)}>(value {BIT_TESTS[mnemonic]}(1ull << bit)));")
                return "{ " + " ".join(lines) + " }", reads, writes
            note(0, 1)
            bits = width * 8
            a = self.operand(ins, 0, width)
            offset = (f"(static_cast<std::uint64_t>({self.operand(ins, 1, width)}) & {bits - 1}u)"
                      if ins.op1_kind == OpKind.REGISTER else f"{ins.immediate(1) & (bits - 1)}u")
            lines = []
            if sets_flags:
                self.flags = True
                lines.append(f"fb = (static_cast<std::uint64_t>({a}) >> {offset}) & 1u;")
            if mnemonic == M.BT:
                writes.clear()
            else:
                lines.append(self.write(ins, width, f"{a} {BIT_TESTS[mnemonic]}(1ull << {offset})"))
            return " ".join(lines), reads, writes
        if mnemonic == M.BSWAP:
            if width not in (4, 8) or ins.op0_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported BSWAP operands at RVA {ins.ip:#x}: the 16-bit form is undefined")
            note(0)
            return self.write(ins, width, f"byte_swap<{unsigned(width)}>({self.operand(ins, 0, width)})"), reads, writes
        if mnemonic in BIT_SCANS:
            if width not in (2, 4, 8) or ins.op0_kind != OpKind.REGISTER or ins.op1_kind not in (OpKind.REGISTER, OpKind.MEMORY):
                raise ValueError(f"unsupported {str(ins).split()[0].upper()} operands at RVA {ins.ip:#x}")
            note(1)
            # A zero source leaves the whole destination register as it was, so it is read: 16 bits for a
            # 16-bit destination, all 64 otherwise (a 32-bit result zero-extends only for a nonzero source).
            destination = full(ins.op0_register)
            reads[destination] = max(reads.get(destination, 0), 2 if width == 2 else 8)
            u = unsigned(width)
            source = self.operand(ins, 1, width)
            found = self.write(ins, width, f"{BIT_SCANS[mnemonic]}<{u}>(source)")
            if sets_flags:
                self.flags = True
                return f"fa = {source}; {{ const {u} source = static_cast<{u}>(fa); if (source != 0) {{ {found} }} }}", reads, writes
            return f"{{ const {u} source = {source}; if (source != 0) {{ {found} }} }}", reads, writes
        if mnemonic in BIT_COUNTS:
            note(1)
            source = self.operand(ins, 1, width)
            value = f"{BIT_COUNTS[mnemonic]}<{unsigned(width)}>({source})"
            if not sets_flags:
                return self.write(ins, width, value), reads, writes
            self.flags = True
            return (f"fa = {source}; fb = {value}; " + self.write(ins, width, "fb")), reads, writes
        if mnemonic in (*SHIFTS, *ROTATES):
            if sets_flags:
                if mnemonic in ROTATES:
                    raise ValueError(f"unsupported flag producer at RVA {ins.ip:#x}")
                if ins.op1_kind == OpKind.REGISTER:
                    # A masked count of 0 leaves the flags unchanged, so they may come from an earlier instruction.
                    raise ValueError(f"unsupported flag producer at RVA {ins.ip:#x}: a shift by CL may leave the flags unchanged")
                if shift_count(ins) == 0:
                    raise ValueError(f"unsupported flag producer at RVA {ins.ip:#x}: a zero-count shift leaves the flags unchanged")
                if shift_count(ins) >= width * 8:
                    # CF is undefined for SHL and SHR (Intel SDM, SAL/SAR/SHL/SHR); SAR is refused with them.
                    raise ValueError(f"unsupported flag producer at RVA {ins.ip:#x}: "
                                     "a shift by the operand width or more (CF is undefined for SHL and SHR)")
            note(0)
            a = source = self.operand(ins, 0, width)
            # x86 masks the count to six bits for 64-bit operands and to five bits otherwise.
            # Shifts widen to 64 bits (count < 64, no host overflow) and truncate on write;
            # rotates mask modulo the operand width inside rotate_bits.
            mask = 63 if width == 8 else 31
            if ins.op1_kind == OpKind.REGISTER:
                if ins.op1_register != R.CL:
                    raise ValueError(f"unsupported shift count at RVA {ins.ip:#x}")
                read(R.CL)
                count = f"({self.reg(R.RCX)} & {mask}u)"
            else:
                count = f"{ins.immediate(1) & mask}u"
            if mnemonic in ROTATES:
                value = f"rotate_bits<{unsigned(width)}>({a}, {count}, {ROTATES[mnemonic]})"
            else:
                if mnemonic == M.SAR:
                    a = f"static_cast<std::int64_t>(static_cast<{signed(width)}>({a}))"
                else:
                    a = f"static_cast<std::uint64_t>({a})"
                value = f"{a} {SHIFTS[mnemonic]} {count}"
            if sets_flags:
                # CF is the last bit shifted out. OF, defined only for a count of 1, is the MSB of the
                # result XOR CF for SHL/SAL, the MSB of the operand for SHR and 0 for SAR.
                self.flags = True
                bits, n, wide = width * 8, shift_count(ins), f"static_cast<std::uint64_t>({source})"
                carry = f"(({wide} >> {bits - n if mnemonic in (M.SHL, M.SAL) else n - 1}) & 1u)"
                overflow = "0u"
                if n == 1 and mnemonic in (M.SHL, M.SAL):
                    overflow = f"((({wide} >> {bits - 1}) ^ ({wide} >> {bits - 2})) & 1u)"
                elif n == 1 and mnemonic == M.SHR:
                    overflow = f"(({wide} >> {bits - 1}) & 1u)"
                return (f"fa = static_cast<{unsigned(width)}>({value}); fb = {carry} | ({overflow} << 1); "
                        + self.write(ins, width, "fa")), reads, writes
            return self.write(ins, width, value), reads, writes
        if mnemonic == M.IMUL and ins.op_count in (2, 3):
            # The two-operand form reads its destination; the three-operand form only writes it.
            if ins.op_count == 2:
                note(0, 1)
                a, b = self.operand(ins, 0, width), self.operand(ins, 1, width)
            else:
                note(1, 2)
                a, b = self.operand(ins, 1, width), self.operand(ins, 2, width)
            if width == 2:
                # Two uint16_t factors would be promoted to int, where 0xffff * 0xffff overflows.
                a, b = f"static_cast<std::uint32_t>({a})", f"static_cast<std::uint32_t>({b})"
            return self.write(ins, width, f"{a} * {b}"), reads, writes
        if mnemonic in SIGN_EXTENSIONS:
            source, destination = SIGN_EXTENSIONS[mnemonic]
            size = RegisterExt.size(destination)
            read(source)
            writes[full(destination)] = size
            if full(destination) == R.RAX:
                value = extend(self.reg(source), RegisterExt.size(source), True)
            else:
                # Every bit of EDX or RDX becomes the sign bit of the source: 0 - 0 or 0 - 1, wrapped.
                value = f"0ull - (static_cast<std::uint64_t>({self.reg(source)}) >> {RegisterExt.size(source) * 8 - 1})"
            return self.write(ins, size, value, destination), reads, writes
        if mnemonic in (M.MUL, M.IMUL, M.DIV, M.IDIV) and ins.op_count == 1:
            # Operand 0 is a source, not the destination. Every input is copied into a const temporary
            # before RAX or RDX is written, since operand 0 may be one of them or address memory through them.
            writes.clear()
            note(0)
            read(ACCUMULATOR[width])
            accumulator, source = self.reg(ACCUMULATOR[width]), self.operand(ins, 0, width)
            sign = mnemonic in (M.IMUL, M.IDIV)
            if mnemonic in (M.DIV, M.IDIV):
                # An 8-bit divide writes AH, which is not modelled; no intrinsic executes a 16-bit divide.
                if width not in (4, 8):
                    raise ValueError(f"unsupported divide width at RVA {ins.ip:#x}")
                read(DATA[width])
                # The intrinsics execute DIV or IDIV: a zero divisor or a quotient that does not fit raises
                # the same divide error as the original, and no C++ division is involved.
                kind, call = (signed if sign else unsigned)(width), "_div" if sign else "_udiv"
                if width == 8:
                    lines = [f"const {kind} high = static_cast<{kind}>({self.reg(DATA[8])}), "
                             f"low = static_cast<{kind}>({accumulator}), divisor = static_cast<{kind}>({source});",
                             f"{kind} remainder;", f"const {kind} quotient = {call}128(high, low, divisor, &remainder);"]
                else:
                    wide = signed(8) if sign else unsigned(8)
                    lines = [f"const {wide} dividend = static_cast<{wide}>((static_cast<std::uint64_t>("
                             f"{self.reg(DATA[4])}) << 32) | {accumulator});",
                             f"const {kind} divisor = static_cast<{kind}>({source});",
                             f"{kind} remainder;", f"const {kind} quotient = {call}64(dividend, divisor, &remainder);"]
                # Reaching the divide raises its error even when RAX and RDX are overwritten unread: the store
                # to a volatile local makes the compiler compute the quotient, which only the divide can do.
                lines.append(f"volatile {kind} performed = quotient; static_cast<void>(performed);")
                low, high = "quotient", "remainder"
            elif width == 8:
                kind = signed(8) if sign else unsigned(8)
                lines = [f"const {kind} a = static_cast<{kind}>({accumulator}), b = static_cast<{kind}>({source});",
                         f"{kind} high;", f"const {kind} low = {'_mul128' if sign else '_umul128'}(a, b, &high);"]
                low, high = "low", "high"
            else:
                # Factors of up to 32 bits, sign- or zero-extended: their product fits in 64 bits, so the
                # wrapped std::uint64_t product is exact (in two's complement for IMUL).
                lines = [f"const std::uint64_t a = {extend(accumulator, width, sign)}, "
                         f"b = {extend(source, width, sign)};", "const std::uint64_t product = a * b;"]
                low, high = "product", f"product >> {width * 8}"
            if width == 1:
                # MUL and IMUL r/m8 write the whole 16-bit product to AX and leave RDX alone.
                lines.append(self.write(ins, 2, low, R.AX))
                writes[R.RAX] = 2
            else:
                lines += [self.write(ins, width, low, ACCUMULATOR[width]), self.write(ins, width, high, DATA[width])]
                writes.update({R.RAX: width, R.RDX: width})
            return "{ " + " ".join(lines) + " }", reads, writes
        if mnemonic in SETCC:
            condition = self.condition(producer, ins)
            if ins.op0_kind == OpKind.MEMORY:
                return f"store<std::uint8_t>({self.address(ins)}, static_cast<std::uint8_t>({condition}));", reads, writes
            # A byte write: it keeps bits 8..63 of the full register and does not read them.
            return self.write(ins, 1, condition), reads, writes
        if mnemonic in CMOVCC:
            if width not in (4, 8):
                raise ValueError(f"unsupported partial register write at RVA {ins.ip:#x}")
            note(0, 1)
            condition = self.condition(producer, ins)
            source, destination = self.operand(ins, 1, width), self.operand(ins, 0, width)
            # A 32-bit CMOVcc zero-extends the destination even when the condition is false, and a memory
            # source is read either way: capture() reads it before the condition is evaluated.
            return self.write(ins, width, f"({condition}) ? {source} : {destination}"), reads, writes
        if mnemonic == M.RET:
            # Which bytes of RAX are a return value is decided by the definedness pass in lift().
            return "return rax;", {}, {}
        raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")


def shift_count(ins):
    """The immediate count of a shift, masked as x86 does: to six bits for a 64-bit operand, else five."""
    width = RegisterExt.size(ins.op0_register) if ins.op0_kind == OpKind.REGISTER else _memory_width(ins)
    return ins.immediate(1) & (63 if width == 8 else 31)


def _memory_width(ins):
    width = MemorySizeExt.size(ins.memory_size)
    if width not in (1, 2, 4, 8):
        raise ValueError(f"unsupported memory width at RVA {ins.ip:#x}")
    return width


INFO = InstructionInfoFactory()
WRITE_ACCESS = {OpAccess.WRITE, OpAccess.READ_WRITE, OpAccess.COND_WRITE, OpAccess.READ_COND_WRITE}


class Frame:
    """RSP as entry RSP + a constant at each instruction, and the stack bytes each instruction accesses.

    offsets: RVA -> RSP - entry RSP before the instruction. accesses: RVA -> (offset of the first byte from entry
    RSP, width, reads, writes). released: RVA -> (RSP before, RSP after) for an ADD RSP, POP or LEAVE, which leaves
    those bytes below RSP, where their contents may change. The frame array starts at entry RSP + start, the lowest offset rounded down to
    8 mod 16: with RSP + 8 a multiple of 16 at entry, as the ABI requires, every array byte then has the address
    residue mod 16 of the stack byte it stands for. It ends with the home space at entry RSP + 40."""

    frame_limit = MAX_FRAME
    incoming_end = HOME_END

    def __init__(self, function, edges, internal=False):
        if internal:
            # An internal function may read the stack arguments its callers write above its home space.
            self.incoming_end = HOME_END + 8 * MAX_STACK_ARGUMENTS
        self.offsets, self.accesses, self.adjustments, self.released = {function.entry: 0}, {}, set(), {}
        self.output_calls = {}
        self.output_addresses = {}
        self.outputs = {}
        for ip, host in function.host_calls.items():
            if host.output is None:
                continue
            register, width = host.output
            if ip not in function.calls or host.noreturn or register not in host.demands or width not in STACK_OUTPUT_WIDTHS:
                raise ValueError(f"unsupported stack output contract at RVA {ip:#x}")
            before = ip
            while before not in edges and before != function.entry:
                ins = function.ending.get(before)
                if ins is None or ins.flow_control != FlowControl.NEXT:
                    break
                if (ins.mnemonic == M.LEA and ins.op0_register == register and ins.memory_base in (R.RSP, R.RBP)
                        and ins.memory_index == R.NONE and not ins.has_segment_prefix):
                    self.output_calls[ip] = (ins, register, width)
                    self.output_addresses[ins.ip] = width
                    break
                if any(RegisterExt.full_register(used.register) == register
                       or RegisterExt.full_register(used.register) == R.RSP and used.access in WRITE_ACCESS
                       for used in INFO.info(ins).used_registers()):
                    break
                before = ins.ip
            if ip not in self.output_calls:
                raise ValueError(f"stack output call at RVA {ip:#x} needs a proven frame address")
        anchors, self.set_anchor = {function.entry: None}, set()
        work = [function.entry]
        while work:
            ip = work.pop()
            anchor = anchors[ip]
            if ip in function.calls or ip in function.tails:
                # The callee pops its return address; call sites are checked by check_call_sites.
                after = self.offsets[ip]
            else:
                after, anchor = self.step(function.instructions[ip], self.offsets[ip], anchor)
                if after > self.offsets[ip]:
                    self.released[ip] = (self.offsets[ip], after)
            for successor in function.successors[ip]:
                if successor not in self.offsets:
                    self.offsets[successor] = after
                    anchors[successor] = anchor
                    work.append(successor)
                elif self.offsets[successor] != after:
                    raise ValueError(f"stack depth differs at join RVA {successor:#x}: entry RSP"
                                     f"{self.offsets[successor]:+#x} and entry RSP{after:+#x}")
                elif anchors[successor] != anchor:
                    raise ValueError(f"frame pointer differs at join RVA {successor:#x}")
        for ip in sorted(self.offsets):
            if function.instructions[ip].mnemonic == M.RET and self.offsets[ip]:
                raise ValueError(f"stack not balanced at RET RVA {ip:#x}: RSP is entry RSP{self.offsets[ip]:+#x}")
        for ip, (ins, _, width) in self.output_calls.items():
            offset = self.accesses[ins.ip][0]
            if offset < self.offsets[ip] + HOME_SIZE or offset % width:
                raise ValueError(f"misaligned stack output or overlap with callee home space at RVA {ip:#x}")
            self.outputs[ip] = (offset, width)
        # Per stack argument slot: the low bytes that some read reaches, 0 for a slot only written.
        slots = {}
        for offset, width, reads, _ in self.accesses.values():
            if offset >= HOME_END:
                slot = (offset - HOME_END) // 8
                slots[slot] = max(slots.get(slot, 0), offset + width - HOME_END - 8 * slot if reads else 0)
        self.stack = tuple(slots.get(slot, 0) for slot in range(max(slots, default=-1) + 1))
        low = min(self.offsets.values())
        self.start = low - (low + 8) % 16
        self.end = HOME_END + 8 * len(self.stack)
        self.size = self.end - self.start
        if self.size > self.frame_limit:
            raise ValueError(f"frame of {self.size} bytes exceeds the {self.frame_limit}-byte bound")

    def access(self, ins, rsp, offset, width, reads, writes):
        if offset < rsp:
            raise ValueError(f"stack access below RSP at RVA {ins.ip:#x}")
        if offset < 8 and offset + width > 0:
            raise ValueError(f"return address slot accessed at RVA {ins.ip:#x}")
        if offset + width > HOME_END and (offset + width > self.incoming_end or offset < HOME_END
                                          or (offset - HOME_END) // 8 != (offset + width - 1 - HOME_END) // 8):
            raise ValueError(f"unsupported stack argument at RVA {ins.ip:#x}: entry RSP{offset:+#x}")
        self.accesses[ins.ip] = (offset, width, reads, writes)

    def step(self, ins, rsp, anchor):
        """RSP and the optional RBP anchor as offsets from entry RSP, after ins."""
        ip, mnemonic = ins.ip, ins.mnemonic
        if mnemonic == M.RET:
            return rsp, anchor
        if mnemonic == M.LEAVE:
            if ins.code != Code.LEAVEQ:
                raise ValueError(f"unsupported LEAVE operand size at RVA {ip:#x}")
            # RSP becomes the anchor and RBP is popped from it, so only a proven anchor gives the new RSP.
            # __init__ then releases every byte from the old RSP through anchor + 8, the pop slot included.
            if anchor is None:
                raise ValueError(f"LEAVE without a proven frame anchor at RVA {ip:#x}")
            if anchor < rsp:
                raise ValueError(f"LEAVE with the frame anchor below RSP at RVA {ip:#x}: "
                                 f"anchor entry RSP{anchor:+#x}, RSP entry RSP{rsp:+#x}")
            if anchor + 8 > 0:
                raise ValueError(f"LEAVE would raise RSP above entry RSP at RVA {ip:#x}: "
                                 f"anchor entry RSP{anchor:+#x}")
            self.access(ins, rsp, anchor, 8, True, False)
            return anchor + 8, None
        if ins.op0_kind == OpKind.REGISTER and ins.op0_register == R.RBP:
            if mnemonic == M.MOV and ins.op1_kind == OpKind.REGISTER and ins.op1_register == R.RSP:
                self.set_anchor.add(ip)
                return rsp, rsp
            if (mnemonic == M.LEA and ins.memory_base == R.RSP and ins.memory_index == R.NONE
                    and not ins.has_segment_prefix):
                self.set_anchor.add(ip)
                displacement = ins.memory_displacement - ((ins.memory_displacement >> 63) << 64)
                return rsp, rsp + displacement
        if mnemonic in (M.PUSH, M.POP):
            if ins.op0_kind != OpKind.REGISTER or ins.op0_register not in NONVOLATILE:
                raise ValueError(f"unsupported push or pop at RVA {ip:#x}: {ins}")
            if ins.op0_register == R.RBP and anchor is not None:
                if mnemonic == M.PUSH:
                    raise ValueError(f"frame pointer escapes at RVA {ip:#x}: {ins}")
                anchor = None
            pushed = rsp - 8 if mnemonic == M.PUSH else rsp
            self.access(ins, pushed, pushed, 8, mnemonic == M.POP, mnemonic == M.PUSH)
            return (pushed if mnemonic == M.PUSH else rsp + 8), anchor
        if (mnemonic in (M.SUB, M.ADD) and ins.op0_kind == OpKind.REGISTER and ins.op0_register == R.RSP
                and ins.op1_kind in (OpKind.IMMEDIATE8TO64, OpKind.IMMEDIATE32TO64)):
            change = ins.immediate(1) & ((1 << 64) - 1)
            change -= (change >> 63) << 64
            after = rsp - change if mnemonic == M.SUB else rsp + change
            if change % 8 or after > 0:
                raise ValueError(f"unsupported stack adjustment at RVA {ip:#x}: {ins}")
            self.adjustments.add(ip)
            return after, anchor
        info = INFO.info(ins)
        if ip in self.output_addresses:
            base = rsp if ins.memory_base == R.RSP else anchor
            if base is None:
                raise ValueError(f"stack output uses an unproven frame anchor at RVA {ip:#x}")
            displacement = ins.memory_displacement - ((ins.memory_displacement >> 63) << 64)
            self.access(ins, rsp, base + displacement, self.output_addresses[ip], False, False)
            return rsp, anchor
        if any(RegisterExt.full_register(used.register) == R.RSP and used.access in WRITE_ACCESS
               for used in info.used_registers()):
            raise ValueError(f"dynamic frame at RVA {ip:#x}: {ins}")
        if (any(ins.op_kind(n) == OpKind.REGISTER and RegisterExt.full_register(ins.op_register(n)) == R.RSP
                for n in range(ins.op_count))
                or mnemonic == M.LEA and RegisterExt.full_register(ins.memory_base) == R.RSP):
            raise ValueError(f"frame address escapes at RVA {ip:#x}: {ins}")
        has_memory = OpKind.MEMORY in map(ins.op_kind, range(ins.op_count))
        anchored_memory = anchor is not None and ins.memory_base == R.RBP and has_memory
        stack_base = anchor if anchored_memory else rsp
        if mnemonic != M.NOP and has_memory and (ins.memory_base == R.RSP or anchored_memory):
            used = [memory for memory in info.used_memory() if memory.base == ins.memory_base]
            if (ins.memory_index != R.NONE or ins.has_segment_prefix or len(used) != 1
                    or used[0].access not in (OpAccess.READ, OpAccess.WRITE, OpAccess.READ_WRITE)):
                raise ValueError(f"unsupported stack operand at RVA {ip:#x}: {ins}")
            displacement = ins.memory_displacement - ((ins.memory_displacement >> 63) << 64)
            self.access(ins, rsp, stack_base + displacement, _memory_width(ins), used[0].access != OpAccess.WRITE,
                        used[0].access != OpAccess.READ)
        if anchor is not None and mnemonic != M.NOP:
            restores_rbp = (mnemonic == M.MOV and ins.op0_kind == OpKind.REGISTER
                            and ins.op0_register == R.RBP and ip in self.accesses)
            if (any(ins.op_kind(n) == OpKind.REGISTER and RegisterExt.full_register(ins.op_register(n)) == R.RBP
                    and not (n == 0 and restores_rbp) for n in range(ins.op_count))
                    or has_memory and RegisterExt.full_register(ins.memory_index) == R.RBP
                    or has_memory and RegisterExt.full_register(ins.memory_base) == R.RBP and not anchored_memory):
                raise ValueError(f"frame pointer escapes at RVA {ip:#x}: {ins}")
            if restores_rbp:
                anchor = None
        return rsp, anchor


def slot_register(ins):
    """The nonvolatile register that PUSH, POP or a 64-bit MOV copies whole to or from the frame, else None."""
    if ins.mnemonic in (M.PUSH, M.POP):
        return ins.op0_register
    if ins.mnemonic == M.LEAVE:
        return R.RBP
    if ins.mnemonic == M.MOV and ins.memory_base in (R.RSP, R.RBP):
        for number in (0, 1):
            if ins.op_kind(number) == OpKind.REGISTER and ins.op_register(number) in NONVOLATILE:
                return ins.op_register(number)
    return None


def prove_frame(function, order, effects, frame, stacks=None):
    """Check that every stack byte is written before it is read and that every RET is reached with each
    nonvolatile register holding its entry value, on every path.

    Per stack byte: 1 once a store writes it; 2 + 8 * i + k while it holds byte k of the entry value of
    NONVOLATILE[i], saved by PUSH or MOV [RSP+d] while that register holds its entry value; 0 when not
    written on some path, or released since: when ADD RSP, POP or LEAVE raises RSP, the bytes it leaves below RSP
    become 0 after the POP has read its source, because their contents may change before a later allocation
    brings them back into the frame. Per nonvolatile register: "entry", "written" since entry or the last restore, or
    None when paths disagree. A join keeps what both sides agree on. A save is not a read of the entry value
    as data, so its register read is removed from effects before the register pass."""
    instructions = function.instructions
    saved = {register: bytes(range(2 + 8 * i, 10 + 8 * i)) for i, register in enumerate(NONVOLATILE)}
    # Per call: the low bytes of each stack argument that the callee may read.
    stacks = stacks or {}

    def step(ip, status, data):
        if ip in function.calls:
            # The callee may write its home space, the 32 bytes at RSP. The CALL's return address and the
            # callee's frame lie below RSP, where every byte is already 0: no access is below RSP, and the
            # bytes that ADD RSP and POP leave below RSP become 0. The rest of the frame, including this
            # function's own home space above entry RSP, is kept.
            # It may also write the stack arguments above that home space.
            index = frame.offsets[ip] - frame.start
            size = HOME_SIZE + 8 * len(stacks.get(ip, ()))
            data = data[:index] + bytes(size) + data[index + size:]
            if ip in frame.outputs:
                offset, width = frame.outputs[ip]
                index = offset - frame.start
                data = data[:index] + b"\1" * width + data[index + width:]
            return status, data
        status, restored = dict(zip(NONVOLATILE, status)), None
        if ip in frame.accesses:
            offset, width, reads, writes = frame.accesses[ip]
            index, register = offset - frame.start, slot_register(instructions[ip])
            if reads and register is not None and data[index:index + 8] == saved[register]:
                restored = register
            if writes:
                value = saved[register] if register is not None and status[register] == "entry" else b"\1" * width
                data = data[:index] + value + data[index + width:]
        if ip in frame.released:
            low, high = (offset - frame.start for offset in frame.released[ip])
            data = data[:low] + bytes(high - low) + data[high:]
        for register in (*effects[ip][1], *effects[ip][2]):
            if register in status:
                status[register] = "entry" if register == restored else "written"
        return tuple(status.values()), data

    # The caller writes the stack arguments before the call.
    state = {function.entry: (("entry",) * len(NONVOLATILE),
                              bytes(HOME_END - frame.start) + b"\1" * (frame.end - HOME_END))}
    work = [function.entry]
    while work:
        ip = work.pop()
        out = step(ip, *state[ip])
        for successor in function.successors[ip]:
            old = state.get(successor)
            new = out if old is None else (tuple(a if a == b else None for a, b in zip(old[0], out[0])),
                                           bytes(a if a == b else 0 for a, b in zip(old[1], out[1])))
            if new != old:
                state[successor] = new
                work.append(successor)
    for ip in order:
        ins, (status, data) = instructions[ip], state[ip]
        status, reads, register = dict(zip(NONVOLATILE, status)), effects[ip][0], slot_register(ins)
        offset, width, loads, stores = frame.accesses.get(ip, (0, 0, False, False))
        if stores and register is not None and status[register] == "entry":
            del reads[register]
        for read in reads:
            if status.get(read, "written") != "written":
                raise ValueError(f"nonvolatile register {NAMES[read]} may hold the caller's value where RVA "
                                 f"{ip:#x} reads it")
        for slot, needed in enumerate(stacks.get(ip, ())):
            index = frame.offsets[ip] + HOME_SIZE + 8 * slot - frame.start
            passed = data[index:index + needed]
            if passed != b"\1" * needed:
                at = index + frame.start + (passed.index(0) if 0 in passed else 0)
                raise ValueError(f"stack argument {5 + slot} of the call at RVA {ip:#x} is undefined at entry "
                                 f"RSP{at:+#x}" if 0 in passed else
                                 f"stack argument {5 + slot} of the call at RVA {ip:#x} holds a saved register")
        got = data[offset - frame.start:offset - frame.start + width]
        if loads and got != b"\1" * width and not (register is not None and got == saved[register]):
            if 0 in got:
                raise ValueError(f"undefined stack byte at entry RSP{offset + got.index(0):+#x} at RVA {ip:#x}")
            raise ValueError(f"stack bytes at entry RSP{offset:+#x} hold the saved "
                             f"{NAMES[NONVOLATILE[(max(got) - 2) // 8]]}, read as data at RVA {ip:#x}")
        if ins.mnemonic == M.RET or ip in function.tails:
            site = "tail jump" if ip in function.tails else "RET"
            for register, value in status.items():
                if value != "entry":
                    raise ValueError(f"nonvolatile register {NAMES[register]} is not restored on every path to "
                                     f"the {site} at RVA {ip:#x}")


def flag_producer(function, edges, consumer):
    """The instruction whose flags consumer reads, within one straight-line run."""
    current = consumer
    for _ in range(MAX_FLAG_DISTANCE):
        before = function.ending.get(current.ip)
        if before is not None and before.ip in function.calls:
            raise ValueError(f"flags clobbered by the call at RVA {before.ip:#x} are read at RVA {consumer.ip:#x}")
        if (current.ip in edges or current.ip == function.entry or before is None
                or before.flow_control != FlowControl.NEXT):
            raise ValueError(f"flags cross a join before RVA {consumer.ip:#x}")
        if before.mnemonic not in FLAG_NEUTRAL:
            if before.mnemonic not in FLAG_PRODUCERS:
                raise ValueError(f"unsupported flag producer at RVA {before.ip:#x}")
            return before
        current = before
    raise ValueError(f"flag producer too far before RVA {consumer.ip:#x}")


def check_call_sites(function, offsets):
    """The frame conditions of direct calls and tail jumps, given RSP before each instruction as an
    offset from entry RSP."""
    for ip in sorted({**function.calls, **function.tails}):
        if ip in function.tails and offsets[ip] != 0:
            raise ValueError(f"stack not balanced at tail jump RVA {ip:#x}: RSP is entry RSP{offsets[ip]:+#x}")
        if ip in function.calls and (offsets[ip] % 16 != 8 or offsets[ip] + HOME_SIZE > 0):
            # Entry RSP is 8 mod 16, and the callee's home space must lie below the return address.
            raise ValueError(f"call without an aligned home space at entry RSP{offsets[ip]:+#x} at RVA {ip:#x}")


def prove_direction(function):
    """The Windows ABI enters with DF clear and requires it clear at calls and returns.

    Keep both possibilities at joins. The emitted local follows the actual path;
    this proof prevents a backwards-copy region from leaking DF across an ABI edge.
    """
    states = {function.entry: frozenset({False})}
    work = [function.entry]
    while work:
        ip = work.pop()
        ins = function.instructions[ip]
        incoming = states[ip]
        if (ins.mnemonic == M.RET or ip in function.calls or ip in function.tails) and True in incoming:
            raise ValueError(f"direction flag may be set at call or return RVA {ip:#x}")
        outgoing = (frozenset({ins.mnemonic == M.STD}) if ins.mnemonic in (M.CLD, M.STD) else incoming)
        for target in function.successors[ip]:
            merged = states.get(target, frozenset()) | outgoing
            if merged != states.get(target):
                states[target] = merged
                work.append(target)


def lift(pe, name, arguments=DEFAULT_ARGUMENTS, program=None, *, image_data=False, host_cpu=False):
    """Translate one export that takes the given number of integer arguments, 3 or 4.

    Functions it reaches through direct calls and tail jumps are translated into program."""
    if not isinstance(arguments, int) or arguments not in ARGUMENT_COUNTS:
        raise ValueError(f"unsupported argument count {arguments!r} for export {name}: expected 3 or 4")
    entry = export_rva(pe, name)
    program = program or Program(pe, image_data=image_data, host_cpu=host_cpu)
    return program.lift(entry, f"lifted_{name}", ARGUMENT_REGISTERS[:arguments])


class Binding(NamedTuple):
    provider: Path
    symbol: str
    link_name: str | None
    arguments: int
    return_bytes: int
    output: tuple[int, int] | None = None


class HostImport(NamedTuple):
    symbol: str
    widths: tuple
    return_width: int
    noreturn: bool = False
    # An explicitly bound synchronous import always initializes this output and does not retain its address.
    output: tuple[int, int] | None = None

    @property
    def demands(self):
        return dict(zip(ARGUMENT_REGISTERS, self.widths))


class Assumed(NamedTuple):
    """The summary that calls into a function still being translated use: its return width,
    argument demands and stack argument demands, assumed until its translation computes them."""
    entry: int
    return_width: int
    demands: dict
    stack: tuple = ()


def parameters(registers, stack=()):
    """The C++ parameters of a function: its argument registers, then one word per stack argument."""
    return ", ".join([f"word {NAMES[r]}" for r in registers] + [f"word arg{5 + slot}" for slot in range(len(stack))])


def at_least_as_good(computed, assumed):
    """Whether a computed summary defines at least the assumed RAX bytes and demands no more."""
    return (computed.return_width >= assumed.return_width
            and all(width <= assumed.demands.get(r, 0) for r, width in computed.demands.items())
            and len(computed.stack) <= len(assumed.stack)
            and all(width <= assumed.stack[slot] for slot, width in enumerate(computed.stack)))


class Program:
    """The functions of one image reached from the requested exports through direct calls and tail
    jumps. Each callee is translated once, in internal mode, and its source precedes its callers'.

    A call to an internal function whose translation is in progress closes a call cycle. It uses an
    assumed summary, at first 8 defined RAX bytes and no demands. Each function of the cycle is
    translated again with the meet of the assumed and computed summaries until each computed
    summary defines at least the assumed bytes and demands no more. Its returns then define those
    bytes and it reads no more, by induction on the depth of calls that return. The functions of a
    cycle are kept once its first function is done, after declarations, so each may call any other."""

    def __init__(self, pe, bindings=None, *, image_data=False, imports=None, host_cpu=False):
        self.pe = pe
        self.image_data = image_data
        self.host_cpu = host_cpu
        self.imports = imports or {}
        self.entries = Entries(pe, bindings)
        self.active = []  # entry of each translation in progress, outermost first
        self.internal = []  # whether each of those is an internal function
        self.lows = []  # per active translation: the lowest active depth it has used a summary from
        self.assumed = {}
        self.pending = []  # (entry, source, function, low) of cycle functions not yet kept
        self.callees = {}
        self.refused = {}
        self.sources = []
        self.import_resolver = None

    def lift(self, entry, symbol, arguments, internal=False):
        self.active.append(entry)
        self.internal.append(internal)
        self.lows.append(len(self.active) - 1)
        try:
            return lift_function(self, entry, symbol, arguments, internal)
        finally:
            self.active.pop()
            self.internal.pop()
            self.low = self.lows.pop()
            if self.lows:
                self.lows[-1] = min(self.lows[-1], self.low)

    def depends(self, depth):
        self.lows[-1] = min(self.lows[-1], depth)

    def committed(self, callee):
        return isinstance(callee, HostImport) or self.callees.get(callee.entry) is callee

    def callee(self, target, ip):
        if target in self.refused:
            raise ValueError(f"callee at RVA {target:#x} refused: {self.refused[target]}")
        if target in self.callees:
            return self.callees[target]
        for entry, _, function, low in self.pending:
            if entry == target:
                self.depends(low)
                return function
        depth = next((d for d, (entry, internal) in enumerate(zip(self.active, self.internal))
                      if internal and entry == target), None)
        if depth is not None:
            self.depends(depth)
            return self.assumed.setdefault(target, Assumed(target, 8, {}))
        if len(self.callees) + len(self.pending) >= MAX_CALLEES:
            raise ValueError(f"callee limit reached at RVA {ip:#x}")
        depth = len(self.active)
        start = len(self.pending)
        while True:
            try:
                source, function = self.lift(target, f"fn_{target:x}", ARGUMENT_REGISTERS, internal=True)
                if function.return_width is None:
                    raise ValueError("no reachable normal return")
            except ValueError as error:
                del self.pending[start:]
                self.assumed.pop(target, None)
                self.refused[target] = str(error)
                raise ValueError(f"callee at RVA {target:#x} refused: {error}") from None
            assumed = self.assumed.get(target)
            if assumed is None or at_least_as_good(function, assumed):
                break
            # Translate the cycle again from a summary that both bounds.
            demands = {r: max(width, function.demands.get(r, 0))
                       for r, width in assumed.demands.items()} | {
                       r: width for r, width in function.demands.items() if r not in assumed.demands}
            slots = max(len(assumed.stack), len(function.stack))
            stack = tuple(max(assumed.stack[slot] if slot < len(assumed.stack) else 0,
                              function.stack[slot] if slot < len(function.stack) else 0) for slot in range(slots))
            self.assumed[target] = Assumed(target, min(assumed.return_width, function.return_width), demands, stack)
            del self.pending[start:]
        if self.low < depth:
            # Part of a cycle through a function below it on the stack, which keeps it when that is done.
            # The functions pending from its translation now depend on what it depends on: a later
            # translation at this depth that reuses one is in the same cycle.
            self.pending[start:] = [(e, t, f, min(l, self.low)) for e, t, f, l in self.pending[start:]]
            self.pending.append((target, source, function, self.low))
            return function
        members = [(entry, text, member) for entry, text, member, _ in self.pending[start:]] + [(target, source, function)]
        del self.pending[start:]
        cyclic = [self.assumed.pop(entry, None) for entry, _, _ in members]
        if any(cyclic) or len(members) > 1:
            self.sources.append("\n".join(f"static word fn_{entry:x}({parameters(ARGUMENT_REGISTERS, member.stack)});"
                                          for entry, _, member in sorted(members, key=lambda m: m[0])))
        for entry, text, member in members:
            self.callees[entry] = member
            self.sources.append(text)
        return function


def argument_demands(function, order, effects):
    """Low bytes of each argument register that the function may read while it may still hold the
    value passed at entry. Any write of 32 or 64 bits ends that value; an 8- or 16-bit write keeps
    the rest of it, so a later wider read still counts. Joins take the union."""
    holding = {function.entry: frozenset(function.arguments)}
    demands = {}
    changed = True
    while changed:
        changed = False
        for ip in order:
            if ip not in holding:
                continue
            reads, writes, kills = effects[ip]
            for r, width in reads.items():
                if r in holding[ip]:
                    demands[r] = max(demands.get(r, 0), width)
            out = set(holding[ip]) - set(kills)
            out -= {r for r, width in writes.items() if width >= 4}
            if ip in function.calls:
                out -= VOLATILE
            for successor in function.successors[ip]:
                merged = holding.get(successor, frozenset()) | out
                if merged != holding.get(successor):
                    holding[successor] = merged
                    changed = True
    return demands


def lift_function(program, entry, symbol, arguments, internal):
    """Translate the function at entry. An internal function takes all four argument registers, and
    its demands record how many low bytes of each one it may read before writing it."""
    pe = program.pe
    function = Function(pe, entry, program.entries)
    function.arguments = arguments
    if function.bound is not None:
        first = function.decode(entry)
        if first.flow_control == FlowControl.INDIRECT_BRANCH and first.op0_kind == OpKind.MEMORY:
            raise ValueError(f"bound function {function.bound[2]} at RVA {entry:#x} is an import thunk "
                             "(an indirect jump through memory)")
    edges = function.explore(program.imports, import_resolver=program.import_resolver)
    image_data = program.image_data
    lifter = Lifter(pe if image_data else None, host_cpu=program.host_cpu)
    # A RIP-relative LEA that loads a recognised table base stays a guest address, with or without image data.
    bases = {(base, anchor) for *_, (_, _, base, anchor) in function.dispatches.values()}
    order = sorted(function.instructions)
    instructions = function.instructions
    frame = Frame(function, edges, internal)
    if internal and not function.unwound:
        # Without unwind data a function may not move RSP, so it cannot push, allocate or call.
        moved = sorted(ip for ip, offset in frame.offsets.items() if offset or ip in function.calls)
        if moved:
            what = "call" if moved[0] in function.calls else "stack change"
            raise ValueError(f"{what} in a function without unwind data at RVA {moved[0]:#x}")
    check_call_sites(function, frame.offsets)
    prove_direction(function)
    lifter.frame = {ip: access[0] - frame.start for ip, access in frame.accesses.items()}
    lifter.adjustments = frame.adjustments
    producers = {ip: flag_producer(function, edges, ins) for ip, ins in instructions.items()
                 if ins.flow_control == FlowControl.CONDITIONAL_BRANCH or ins.mnemonic in SETCC | CMOVCC}
    sets_flags = {producer.ip for producer in producers.values()}
    # Callees are translated first: the caller's definedness pass needs their return widths and demands.
    callees = {ip: (function.host_calls[ip] if ip in function.host_calls
                    else program.callee(target, ip))
               for ip, target in sorted({**function.calls, **function.tails}.items())}
    for ip in sorted(function.tails):
        if not program.committed(callees[ip]):
            # The original runs a cycle of tail jumps in constant stack; the host call does not.
            raise ValueError(f"tail jump within a call cycle at RVA {ip:#x}")
    # Per call: the low bytes of each stack argument that the callee may read.
    stacks = {ip: callee.stack for ip, callee in callees.items() if getattr(callee, "stack", ())}
    for ip, stack in sorted(stacks.items()):
        if ip in function.tails:
            # The stack arguments would be this function's own, which its caller wrote.
            raise ValueError(f"tail jump to a function with stack arguments at RVA {ip:#x}")
        if frame.offsets[ip] + HOME_SIZE + 8 * len(stack) > 0:
            raise ValueError(f"stack arguments of the call at RVA {ip:#x} reach the return address")
    loads = {sequence[0]: index for index, _, sequence, _, _, _ in function.dispatches.values()}
    skipped = {ip for _, _, sequence, _, _, _ in function.dispatches.values() for ip in sequence}
    body = {}
    effects = {}
    for ip in order:
        ins = instructions[ip]
        if ins.has_lock_prefix or ins.has_repne_prefix or (ins.has_rep_prefix and ins.code not in STRING_TRANSFERS):
            raise ValueError(f"unsupported prefix at RVA {ip:#x}")
        body[ip] = ""
        if ip in callees:
            callee = callees[ip]
            if isinstance(callee, HostImport):
                output = frame.output_calls.get(ip)
                call_args = [lifter.slot(output[0]) if output and r == output[1] else lifter.reg(r)
                             for r in callee.demands]
                call = f"{callee.symbol}({', '.join(call_args)})"
            else:
                # Stack arguments are passed by value from the bytes above the callee's home space. A slot
                # that the callee never reads is passed as 0; the caller then need not have a frame.
                passed = [lifter.reg(r) for r in ARGUMENT_REGISTERS] + [
                    f"load<word>(reinterpret_cast<word>(frame + {frame.offsets[ip] + HOME_SIZE + 8 * slot - frame.start}))"
                    if width else "0" for slot, width in enumerate(stacks.get(ip, ()))]
                call = f"fn_{callee.entry:x}({', '.join(passed)})"
            # A tail jump returns the callee's RAX; a call leaves it in RAX.
            if isinstance(callee, HostImport) and callee.return_width == 0:
                body[ip] = f"{call}; return 0;" if ip in function.tails else f"{call};"
            else:
                lifter.used.add(R.RAX)
                body[ip] = f"return {call};" if ip in function.tails else f"rax = {call};"
            reads = dict(callee.demands)
            if ip in frame.output_calls:
                del reads[frame.output_calls[ip][1]]
            effects[ip] = (reads, {}, set())
        elif ins.flow_control == FlowControl.CONDITIONAL_BRANCH:
            body[ip] = f"if ({lifter.condition(producers[ip], ins)}) goto loc_{ins.near_branch_target:x};"
            effects[ip] = ({}, {}, set())
        elif ins.flow_control == FlowControl.UNCONDITIONAL_BRANCH:
            body[ip] = f"goto loc_{ins.near_branch_target:x};"
            effects[ip] = ({}, {}, set())
        elif ip in function.dispatches:
            index, targets, _, _, target, _ = function.dispatches[ip]
            cases = {}
            for number, destination in enumerate(targets):
                cases.setdefault(destination, []).append(number)
            lines = [f"switch ({lifter.reg(index)}) {{"]
            for destination, numbers in cases.items():
                labels = " ".join(f"case {n}:" for n in numbers)
                lines.append(f"    {labels} goto loc_{destination:x};")
            lines += ["    default: std::abort();", "}"]
            body[ip] = "\n    ".join(lines)
            # The guest jump target has no host meaning after the dispatch.
            effects[ip] = ({}, {}, {target})
        elif ip in skipped:
            # The table load reads the 64-bit index; it and the add leave guest addresses behind.
            effects[ip] = ({loads[ip]: 8} if ip in loads else {}, {}, {full(ins.op0_register)})
        elif ip in frame.set_anchor:
            # Only the frame proof observes the anchor. No host pointer stands in for RBP.
            effects[ip] = ({}, {R.RBP: 8}, set())
        elif ip in frame.output_addresses:
            effects[ip] = ({}, {}, {full(ins.op0_register)})
        elif (ins.mnemonic == M.LEA and ins.is_ip_rel_memory_operand and RegisterExt.is_gpr64(ins.op0_register)
              and (not image_data or (ins.op0_register, ins.ip_rel_memory_address) in bases)):
            # A guest address is only usable as a jump-table base, which the switch replaces.
            effects[ip] = ({}, {}, {full(ins.op0_register)})
        else:
            text, reads, writes = lifter.statement(ins, ip in sets_flags, producers.get(ip))
            body[ip] = text
            effects[ip] = (reads, writes, set())
        if R.RSP in set().union(*effects[ip]):
            # Frame rejects every use of RSP whose offset it does not know, so RSP is never a C++ variable.
            raise ValueError(f"unsupported use of rsp at RVA {ip:#x}")
    # Compilers may copy a byte-sized return through a wider register move. Carry
    # the defined low width forward and validate the eventual consumer separately.
    copies = {ip: (full(ins.op0_register), full(ins.op1_register), RegisterExt.size(ins.op0_register))
              for ip, ins in instructions.items()
              if ins.mnemonic == M.MOV and ins.op0_kind == ins.op1_kind == OpKind.REGISTER
              and RegisterExt.size(ins.op0_register) >= 4 and full(ins.op0_register) in effects[ip][1]}
    prove_frame(function, order, effects, frame, stacks)

    # Per register: (low bytes defined on every path: 0, 1, 2 or 8; may hold a guest address that
    # the translation does not reproduce). Arguments are defined at entry; joins meet as (min, or).
    state = {ip: dict.fromkeys(NAMES, (8, False)) for ip in order}
    state[entry] = {r: (8 if r in function.arguments else 0, False) for r in NAMES}
    changed = True
    while changed:
        changed = False
        for ip in order:
            _, writes, kills = effects[ip]
            out = dict(state[ip])
            for r, width in writes.items():
                # A 32- or 64-bit write defines all eight bytes; an 8- or 16-bit write keeps the rest.
                out[r] = (8, False) if width >= 4 else (max(out[r][0], width), out[r][1])
            if ip in copies:
                dst, src, width = copies[ip]
                defined, guest = state[ip][src]
                out[dst] = (8 if defined >= width else defined, guest)
            for r in kills:
                out[r] = (0, True)
            if ip in function.calls:
                # The callee may change every volatile register. RAX keeps the low bytes that the
                # callee defines at every reachable return; nonvolatile registers are kept.
                for r in VOLATILE:
                    out[r] = (0, False)
                out[R.RAX] = (callees[ip].return_width, False)
            for successor in function.successors[ip]:
                old = state[successor]
                merged = {r: (min(old[r][0], out[r][0]), old[r][1] or out[r][1]) for r in NAMES}
                if merged != old:
                    state[successor] = merged
                    changed = True
    widths = []
    ret_widths = []
    for ip in order:
        missing = [r for r, width in effects[ip][0].items() if state[ip][r][0] < width]
        if ip in copies:
            _, src, _ = copies[ip]
            if state[ip][src][0] and not state[ip][src][1]:
                missing = [r for r in missing if r != src]
        if missing:
            if ip in callees:
                raise ValueError(f"callee argument {NAMES[min(missing)]} is undefined at RVA {ip:#x}")
            raise ValueError(f"undefined register {NAMES[min(missing)]} at RVA {ip:#x}")
        if ip in function.tails:
            # The callee returns to this function's caller with its own defined RAX bytes.
            widths.append(callees[ip].return_width)
        elif instructions[ip].mnemonic == M.RET:
            defined, guest = state[ip][R.RAX]
            if guest:
                raise ValueError(f"return value may hold a guest address at RVA {ip:#x}")
            widths.append(defined)
            ret_widths.append(defined)
    # Consecutive low RAX bytes defined at every reachable RET or tail jump; None when there is none.
    function.return_width = min(widths) if widths else None
    function.demands = argument_demands(function, order, effects)
    # Calls in a cycle pass the stack arguments the assumed summary names; the definition takes them all.
    assumed = program.assumed.get(entry) if internal else None
    stack = frame.stack + (0,) * max(0, len(assumed.stack) - len(frame.stack)) if assumed else frame.stack
    function.stack = tuple(max(width, assumed.stack[slot] if assumed and slot < len(assumed.stack) else 0)
                           for slot, width in enumerate(stack))

    labels = set(edges)
    if order[0] != entry:
        # Emission follows address order. Code that the function reaches below its entry
        # comes first in the text, so execution has to start with a jump to the entry.
        labels.add(entry)
    for position, ip in enumerate(order):
        ins = instructions[ip]
        following = order[position + 1] if position + 1 < len(order) else None
        if ins.flow_control in STRAIGHT and function.successors[ip] and following != ins.next_ip:
            labels.add(ins.next_ip)
    locals_ = sorted((lifter.used | {R.RAX}) - set(function.arguments), key=lambda r: NAMES[r])
    declarations = [f"word {NAMES[r]} = 0;" for r in locals_]
    if any(ins.code in STRING_TRANSFERS or ins.mnemonic in (M.CLD, M.STD) for ins in instructions.values()):
        declarations.append("bool direction = false;  // Windows x64 ABI entry state")
    function.image_data = lifter.image_used
    if lifter.image_used:
        # Read once at entry: a call while nothing is bound fails here, before any access.
        declarations.insert(0, "const word image = bound_image();")
    if lifter.flags:
        declarations.append("word fa = 0, fb = 0;")
    if lifter.frame:
        declarations.append(f"alignas(16) std::uint8_t frame[{frame.size}] = {{}};  "
                            f"// entry RSP{frame.start:+#x} to entry RSP+{frame.end:#x}")
    for slot, width in enumerate(frame.stack):
        if width:
            declarations.append(f"std::memcpy(frame + {HOME_END + 8 * slot - frame.start}, &arg{5 + slot}, 8);")
    text = [f"    goto loc_{entry:x};"] if order[0] != entry else []
    for position, ip in enumerate(order):
        if ip in labels:
            text.append(f"loc_{ip:x}:")
        if body[ip]:
            text.append("    " + body[ip])
        ins = instructions[ip]
        following = order[position + 1] if position + 1 < len(order) else None
        if ins.flow_control in STRAIGHT and function.successors[ip] and following != ins.next_ip:
            text.append(f"    goto loc_{ins.next_ip:x};")
    linkage = "static" if internal else 'extern "C" __declspec(dllexport)'
    signature = f'{linkage} word {symbol}({parameters(function.arguments, function.stack)})'
    comment = []
    if function.return_width is None:
        comment = ["// RAX: no reachable normal return"]
    elif function.return_width < 8:
        comment = [f"// RAX: {function.return_width} low byte(s) are defined at every reachable RET"]
    imported_tails = [function.host_calls[ip] for ip in function.tails if ip in function.host_calls]
    if imported_tails:
        target = imported_tails[0]
        rets = f"{min(ret_widths)} defined at every reachable RET" if ret_widths else "no RET is reachable"
        comment = [f"// RAX: {function.return_width} low byte(s) at every reachable normal return: {rets}, "
                   f"{target.return_width} guaranteed after every tail jump to {target.symbol} as the binding states"]
    source = [*comment, signature + " {", *("    " + d for d in declarations), *text, "}"]
    return "\n".join(source), function


PRELUDE = """#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <intrin.h>
using word = std::uint64_t;
static inline bool even_parity(word value) {
    // Fold the low byte into a nibble; bit n of 0x6996 is its odd parity.
    const auto folded = static_cast<unsigned>((value ^ (value >> 4)) & 15u);
    return ((0x6996u >> folded) & 1u) == 0;
}
template <class T> static inline T rotate_bits(T value, word count, bool left) {
    const unsigned bits = static_cast<unsigned>(sizeof(T) * 8);
    const unsigned shift = static_cast<unsigned>(count & (bits - 1));
    if (!shift) return value;
    const unsigned amount = left ? shift : bits - shift;
    const word wide = value;
    return static_cast<T>((wide << amount) | (wide >> (bits - amount)));
}
// POPCNT, LZCNT and TZCNT, at the width of T; LZCNT and TZCNT of 0 are the width.
template <class T> static inline T count_ones(T value) {
    unsigned count = 0;
    for (word bits = value; bits != 0; bits &= bits - 1) ++count;
    return static_cast<T>(count);
}
template <class T> static inline T leading_zeros(T value) {
    unsigned count = 0;
    while (count < sizeof(T) * 8 && ((static_cast<word>(value) >> (sizeof(T) * 8 - 1 - count)) & 1u) == 0) ++count;
    return static_cast<T>(count);
}
template <class T> static inline T trailing_zeros(T value) {
    unsigned count = 0;
    while (count < sizeof(T) * 8 && ((static_cast<word>(value) >> count) & 1u) == 0) ++count;
    return static_cast<T>(count);
}
// BSR of a nonzero value: the index of its highest set bit.
template <class T> static inline T highest_bit(T value) {
    return static_cast<T>(sizeof(T) * 8 - 1 - leading_zeros<T>(value));
}
// BSWAP: reverse the bytes of a 32- or 64-bit value.
template <class T> static inline T byte_swap(T value) {
    T result = 0;
    for (unsigned i = 0; i < sizeof(T); ++i) result = static_cast<T>((result << 8) | ((static_cast<word>(value) >> (8 * i)) & 0xffu));
    return result;
}
// A read of the function's own frame array, which cannot fault.
template <class T> static T load(word address) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    return value;
}
// Every guest read, once per instruction: the compiler must perform it even when its value is unused,
// because the original faults on an inaccessible address either way.
template <class T> static T load_always(word address) {
    return *reinterpret_cast<const volatile T __unaligned*>(address);
}
template <class T> static void store(word address, T value) {
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(value));
}
"""


IMAGE_DATA = """
// RIP-relative data addresses a live mapping of the input image, bound at run time (--image-data).
// lifted_bind_image(base) binds the mapping at base and returns 0 only when its headers and read-only
// contents match the input image. It returns 1 for a null base, 2 when the headers differ and 3 when the
// read-only contents differ, and a refused bind leaves the current binding in place. base must be null or
// the start of a live image mapping. lifted_unbind_image() ends the binding. While nothing is bound, a
// translated function that addresses image data raises the noncontinuable structured exception 0xE0494D47
// at entry, before any access. An address that a translated function returns or stores is valid only while
// that mapping stays bound and mapped; a later bind does not move it. Bind and unbind must not run
// concurrently with each other or with a translated function.
extern "C" __declspec(dllimport) void __stdcall RaiseException(unsigned long, unsigned long, unsigned long,
                                                               const unsigned long long*);
static word image_base = 0;
static inline word bound_image() {
    if (!image_base) RaiseException(0xE0494D47ul, 1ul /* EXCEPTION_NONCONTINUABLE */, 0ul, nullptr);
    return image_base;
}
// FNV-1a over the mapped bytes [begin, end) of each span, compared with the value computed from the file.
template <std::size_t N> static std::uint64_t image_hash(word base, const std::uint32_t (&spans)[N][2]) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const auto& span : spans)
        for (word rva = span[0]; rva < span[1]; ++rva)
            hash = (hash ^ load<std::uint8_t>(base + rva)) * 0x100000001b3ull;
    return hash;
}
// The DOS signature and e_lfanew, the PE signature and file header (TimeDateStamp included), SizeOfImage,
// CheckSum and the section table. A loader rewrites ImageBase, which is not included.
static const std::uint32_t image_headers[][2] = {%s};
// The file-backed bytes of every readable, non-writable section, except the slots that DIR64 relocations
// rewrite and the import address table.
static const std::uint32_t image_contents[][2] = {%s};
extern "C" __declspec(dllexport) int lifted_bind_image(word base) {
    if (!base) return 1;
    if (image_hash(base, image_headers) != %#xull) return 2;
    if (image_hash(base, image_contents) != %#xull) return 3;
    image_base = base;
    return 0;
}
extern "C" __declspec(dllexport) void lifted_unbind_image() { image_base = 0; }
"""


def relocated(pe):
    """RVAs of the 8-byte slots that DIR64 base relocations rewrite when the image is mapped elsewhere."""
    directory = pe.OPTIONAL_HEADER.DATA_DIRECTORY[5]
    data = pe.get_data(directory.VirtualAddress, directory.Size) if directory.Size else b""
    slots, offset = [], 0
    while offset + 8 <= len(data):
        page, size = struct.unpack_from("<II", data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError(f"invalid base relocation block at RVA {directory.VirtualAddress + offset:#x}")
        for (entry,) in struct.iter_unpack("<H", data[offset + 8:offset + size - size % 2]):
            if entry >> 12 == 10:  # IMAGE_REL_BASED_DIR64
                slots.append(page + (entry & 0xfff))
            elif entry >> 12:  # anything but IMAGE_REL_BASED_ABSOLUTE padding
                raise ValueError(f"unsupported base relocation type {entry >> 12} in page {page:#x}")
        offset += size
    return slots


def image_block(pe):
    """The binding code of IMAGE_DATA, with the identity of this image computed from its file."""
    header = pe.DOS_HEADER.e_lfanew
    optional = header + 24
    table = optional + pe.FILE_HEADER.SizeOfOptionalHeader
    headers = [(0, 2), (0x3c, 0x40), (header, optional), (optional + 56, optional + 60),
               (optional + 64, optional + 68), (table, table + 40 * len(pe.sections))]
    iat = pe.OPTIONAL_HEADER.DATA_DIRECTORY[12]
    skip = sorted([(slot, slot + 8) for slot in relocated(pe)] + [(iat.VirtualAddress, iat.VirtualAddress + iat.Size)])
    contents = []
    for section in pe.sections:
        if section.Characteristics & SCN_READ and not section.Characteristics & SCN_WRITE:
            begin = section.VirtualAddress
            end = begin + min(extent(section), section.SizeOfRawData)
            for low, high in skip:
                if low < end and high > begin:
                    if low > begin:
                        contents.append((begin, low))
                    begin = max(begin, high)
            if begin < end:
                contents.append((begin, end))

    def fnv(spans):
        value = 0xcbf29ce484222325
        for begin, end in spans:
            data = pe.get_data(begin, end - begin)
            if len(data) != end - begin:
                raise ValueError(f"truncated image bytes at RVA {begin:#x}")
            for byte in data:
                value = (value ^ byte) * 0x100000001b3 & ((1 << 64) - 1)
        return value

    text = lambda spans: ", ".join(f"{{{begin:#x}, {end:#x}}}" for begin, end in spans or [(0, 0)])
    return IMAGE_DATA % (text(headers), text(contents), fnv(headers), fnv(contents))


def parse_entry(spec):
    """Split an export-list item, "name" or "name:N", into the name and its integer argument count."""
    name, colon, count = spec.partition(":")
    if not colon:
        return name, DEFAULT_ARGUMENTS
    if not (count.isascii() and count.isdigit()):
        raise ValueError(f"unsupported argument count {count!r} for export {name}: expected 3 or 4")
    return name, int(count)


def load_bindings(source):
    """A range manifest: {"image_sha256": hex, "functions": [{"name", "start", "end"}, ...]}, where start
    and end are RVAs (integers or hex strings) of a function's first byte and the byte after it."""
    manifest = json.loads(Path(source).read_text(encoding="utf-8")) if not isinstance(source, dict) else source
    digest = manifest.get("image_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or digest.strip("0123456789abcdef"):
        raise ValueError("range manifest needs a lowercase hex image_sha256")
    functions = []
    for item in manifest.get("functions", []):
        name, start, end = item.get("name"), item.get("start"), item.get("end")
        start, end = (int(v, 16) if isinstance(v, str) else v for v in (start, end))
        if not isinstance(name, str) or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"malformed range manifest entry {item!r}")
        functions.append((start, end, name))
    return {"image_sha256": digest, "functions": functions}


def import_identities(pe):
    """(DLL, name or "#ordinal") of each ordinary import by its IAT slot RVA, named as resolve_import names it."""
    pe.parse_data_directories(directories=[1])  # ordinary import directory
    return {imp.address - pe.OPTIONAL_HEADER.ImageBase:
            (desc.dll.decode("ascii"), f"#{imp.ordinal}" if imp.import_by_ordinal else imp.name.decode("ascii"))
            for desc in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) for imp in desc.imports}


def translate_pe(pe, names, bindings=None, *, image_data=False, binding=None, host_cpu=False):
    """binding is one Binding or a sequence of them; with several, each binds the IAT slot of the import it names."""
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    if image_data and {"bind_image", "unbind_image"} & {parse_entry(name)[0] for name in names}:
        raise ValueError("an export named bind_image or unbind_image collides with the image binding functions")
    program = Program(pe, None if bindings is None else load_bindings(bindings), image_data=image_data, host_cpu=host_cpu)
    bound = () if binding is None else (binding,) if isinstance(binding, Binding) else tuple(binding)
    for item in bound:
        if item.arguments not in IMPORT_ARGUMENT_COUNTS or item.return_bytes not in IMPORT_RETURN_WIDTHS:
            raise ValueError(f"unsupported binding of {item.symbol}")
        if item.output is not None:
            position, width = item.output
            if not 1 <= position <= item.arguments or width not in STACK_OUTPUT_WIDTHS:
                raise ValueError(f"invalid stack output argument or width for {item.symbol}")
    # With several bindings, each names one ordinary import of the image and a slot takes the binding of its import.
    imported = import_identities(pe) if len(bound) > 1 else {}
    by_import = {}
    for index, item in enumerate(bound if len(bound) > 1 else ()):
        key = (Path(item.provider).name.casefold(), item.symbol)
        if key in by_import:
            raise ValueError(f"two bindings of {Path(item.provider).name}!{item.symbol}")
        if key not in {(dll.casefold(), name) for dll, name in imported.values()}:
            raise ValueError(f"binding {Path(item.provider).name}!{item.symbol} names no ordinary import of the image")
        by_import[key] = index
    slots, chosen = {}, {}  # slot -> HostImport, slot -> index of its binding
    if bound:
        def resolve_slot(slot):
            if len(bound) == 1:
                if slots and slot not in slots:
                    raise ValueError(f"call or tail jump through a second IAT slot RVA {slot:#x}")
                index = 0
            else:
                if slot not in imported:
                    raise ValueError(f"RIP-relative target RVA {slot:#x} is not an ordinary IAT slot")
                dll, name = imported[slot]
                index = by_import.get((dll.casefold(), name))
                if index is None:
                    raise ValueError(f"call or tail jump through IAT slot RVA {slot:#x} ({dll}!{name}) matches "
                                     "no binding")
            item = bound[index]
            symbol = resolve_import(pe, slot, Path(item.provider), item.symbol, item.link_name)
            # The translation calls the import instead of reading the slot; the original's read must not fault.
            if not any(s.VirtualAddress <= slot and slot + 8 <= s.VirtualAddress + extent(s)
                       and s.Characteristics & SCN_READ for s in pe.sections):
                raise ValueError(f"IAT slot RVA {slot:#x} is not 8 bytes of one readable section")
            output = None
            if item.output is not None:
                position, width = item.output
                output = (ARGUMENT_REGISTERS[position - 1], width)
            slots[slot] = HostImport(symbol, (8,) * item.arguments, item.return_bytes, output=output)
            chosen[slot] = index
            return slots[slot]
        program.import_resolver = resolve_slot
    functions = [lift(pe, *parse_entry(name), program=program) for name in names]
    # Callees precede every caller, so each internal function is defined before its first use.
    sources = program.sources + [source for source, _ in functions]
    lines = []
    for index, item in enumerate(bound):
        used = [slot for slot, owner in chosen.items() if owner == index]
        if not used:
            raise ValueError(f"binding {Path(item.provider).name}!{item.symbol} supplied but no call or "
                             "tail jump uses it")
        lines.append((min(used), slots[min(used)].symbol, item))
    # Declared in IAT slot order, so the source does not depend on the order of the bindings.
    lines = [(symbol, item) for _, symbol, item in sorted(lines, key=lambda line: line[0])]
    for (symbol, item), (other, earlier) in ((a, b) for n, a in enumerate(lines) for b in lines[:n]):
        if symbol == other:
            raise ValueError(f"bindings {Path(earlier.provider).name}!{earlier.symbol} and "
                             f"{Path(item.provider).name}!{item.symbol} both declare the C name {symbol}")
    declarations = "".join(f'\nextern "C" __declspec(dllimport) {"word" if item.return_bytes else "void"} '
                           f'{symbol}({", ".join(["word"] * item.arguments)});'
                           for symbol, item in lines) + "\n" if lines else ""
    return PRELUDE + declarations + (image_block(pe) if image_data else "") + "\n".join(f"\n{source}" for source in sources) + "\n", [f for _, f in functions]


class BindingOption(argparse.Action):
    """Records options for each import in command-line order."""

    def __call__(self, parser, namespace, values, option_string=None):
        # The canonical spelling: argparse also accepts an abbreviation such as --return.
        namespace.binding_options = [*namespace.binding_options, (self.option_strings[0], values)]


def parse_binding_options(options):
    """None, one Binding or a tuple of them from the recorded binding options. With one --bind the other options
    may appear anywhere, the last of each counting, as before; with several, each belongs to the --bind before it."""
    binds = sum(option == "--bind" for option, _ in options)
    if binds < 2:
        group = dict(options)  # the last value of each option counts
        if not binds:
            if group:
                raise ValueError("--link-name, --import-arguments, --return-bytes and --stack-output require --bind")
            return None
        groups = [group]
    else:
        groups = []
        for option, value in options:
            if option == "--bind":
                groups.append({option: value})
            elif not groups:
                raise ValueError(f"{option} comes before the first of several --bind options")
            elif option in groups[-1]:
                raise ValueError(f"{option} given twice for --bind {' '.join(groups[-1]['--bind'])}")
            else:
                groups[-1][option] = value
    found = []
    for group in groups:
        if "--import-arguments" not in group or "--return-bytes" not in group:
            raise ValueError("--bind requires --import-arguments and --return-bytes")
        provider, symbol = group["--bind"]
        output = group.get("--stack-output")
        found.append(Binding(Path(provider), symbol, group.get("--link-name"), group["--import-arguments"],
                             group["--return-bytes"], None if output is None else tuple(output)))
    return found[0] if len(found) == 1 else tuple(found)


def translate(image, names, bindings=None, *, image_data=False, binding=None, host_cpu=False):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names, bindings, image_data=image_data, binding=binding, host_cpu=host_cpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help='comma-separated export names; "name:4" translates that export with a '
                                        "fourth integer argument in R9, and the host must pass four")
    parser.add_argument("output", type=Path)
    parser.add_argument("--host-cpu", action="store_true",
                        help="opt into read-only host-CPU instructions (rdfsbase, rdgsbase, rdpid)")
    parser.add_argument("--image-data", action="store_true",
                        help="translate RIP-relative data against a live mapping of the image that the host "
                             "binds at run time with lifted_bind_image; the generated source states the contract")
    parser.add_argument("--bind-ranges", type=Path, metavar="MANIFEST",
                        help="trusted [start, end) ranges of functions without unwind data, bound to this image's sha256")
    parser.set_defaults(binding_options=[])
    parser.add_argument("--bind", nargs=2, metavar=("PROVIDER_DLL", "SYMBOL"), action=BindingOption,
                        help="provider and export name or #ordinal of the import that calls and tail jumps "
                             "through the import address table reach. The image and target must stay valid and "
                             "the original slot must stay readable and contain that target throughout execution. "
                             "Repeat it to bind several imports, each followed by its own options below")
    parser.add_argument("--link-name", action=BindingOption,
                        help="local import-library symbol for an ordinal binding")
    parser.add_argument("--import-arguments", type=int, choices=IMPORT_ARGUMENT_COUNTS, action=BindingOption,
                        help="integer arguments the bound import takes; stated by the binding, not derived from "
                             "the provider (required with --bind)")
    parser.add_argument("--return-bytes", type=int, choices=IMPORT_RETURN_WIDTHS, action=BindingOption,
                        help="guaranteed low RAX bytes of the bound import's return, 0 for none; stated by "
                             "the binding, not derived from the provider (required with --bind)")
    parser.add_argument("--stack-output", nargs=2, type=int, metavar=("ARG", "WIDTH"), action=BindingOption,
                        help="1-based argument that the import fully writes before returning; WIDTH is 1, 2, "
                             "4 or 8. It must not read the initial contents or retain the address")
    args = parser.parse_args()
    try:
        binding = parse_binding_options(args.binding_options)
    except ValueError as error:
        parser.error(str(error))
    try:
        source, functions = translate(args.image, args.exports.split(","), args.bind_ranges,
                                      image_data=args.image_data, binding=binding, host_cpu=args.host_cpu)
        args.output.write_text(source, encoding="utf-8")
        for function in functions:
            print(f"RVA {function.entry:#x}: {len(function.instructions)} instructions, "
                  f"{sum(len(d[1]) for d in function.dispatches.values())} table entries, "
                  + ("no reachable normal return" if function.return_width is None else
                     f"{function.return_width} defined RAX byte(s) at every reachable normal return"))
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
