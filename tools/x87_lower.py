"""Opt-in x87 leaf-function profile for ffx_version_query.lower(x87=True).

The profile accepts:
- FLD and FSTP with an m32, m64 or m80 operand, and FST with an m32 or m64 operand;
- FLD, FST, FSTP and FXCH with ST(i), FCHS and FABS, FSQRT and FRNDINT;
- FXTRACT, which replaces ST(0) by its exponent and pushes its significand, and FSCALE;
- the constant loads FLD1, FLDZ, FLDPI, FLDL2T, FLDL2E, FLDLG2 and FLDLN2;
- FILD, FISTP and FISTTP with an m16, m32 or m64 integer operand, and FIST with m16 or m32;
- FCOMI, FCOMIP, FUCOMI and FUCOMIP ST(0), ST(i), which set ZF and CF and clear OF for the integer
  lowering, and PF in a function with a JP, JNP, FCMOVU or FCMOVNU;
- FCMOVB, FCMOVE, FCMOVBE, FCMOVNB, FCMOVNE, FCMOVNBE, FCMOVU and FCMOVNU ST(0), ST(i), which read ZF,
  CF and PF;
- FCOM and FCOMP with ST(i) or an m32 or m64 operand, FCOMPP, FUCOM and FUCOMP ST(i) and FUCOMPP, which
  set C0, C2 and C3 of the status word, and FNSTSW AX, which copies the status word to AX (see below);
- FADD, FSUB, FSUBR, FMUL, FDIV and FDIVR with an m32 or m64 operand, with ST(0), ST(i) and with
  ST(i), ST(0), and their popping forms FADDP, FSUBP, FSUBRP, FMULP, FDIVP and FDIVRP ST(i), ST(0) for
  i from 1 to 7;
- FIADD, FISUB, FISUBR, FIMUL, FIDIV and FIDIVR with an m16 or m32 integer operand, which is converted
  exactly, as FILD converts it, before the arithmetic;
- FNSTCW m16, which stores the profile's control word 0x027F, and WAIT, which does nothing because
  every exception is masked, so none can be pending (FSTCW m16 decodes as WAIT and FNSTCW);
- FTST, and FICOM and FICOMP with an m16 or m32 integer, which set C0, C2 and C3 as FCOM does, and FXAM,
  which sets them from the class of ST(0) (an empty ST(0) is known at translation time; C1 stays unknown);
- FNOP and FNCLEX, which change nothing the profile models (FCLEX decodes as WAIT and FNCLEX), and the 8087/287
  controls FNENI, FNDISI and FNSETPM, which later processors execute as FNOP (FENI and FDISI decode as WAIT and
  FNENI or FNDISI);
- FFREEP ST(0) on a nonempty stack, which tags ST(0) empty and pops it as FSTP ST(0) does without storing it,
  and FFREE ST(i) of a register that is already empty (i at least the depth), which changes neither occupancy
  nor TOP, and FFREE ST(0) directly followed by FINCSTP on a nonempty stack, which pops ST(0) as FFREEP ST(0)
  does when nothing branches to the FINCSTP (any other FFREE of an occupied register, FFREEP with another
  register, FINCSTP on its own and FDECSTP stay outside the profile);
A function starts with control word 0x027F, TOP 0, eight empty registers and a clear status word, and
must return with the stack empty. The stack depth before every instruction is fixed at translation
time, so each register maps to a C++ slot and no TOP is needed at run time. Translation rejects any x87
instruction outside the profile (including FLDCW, FNINIT, FBLD, FNSTSW with an operand other than AX or an m16, FNSAVE, FLDENV, FRSTOR, a
popping form with ST(0) as the destination, and the undocumented FSTP, FXCH, FCOM and FCOMP alias
encodings), a stack overflow or underflow, a depth that differs where control flow joins, an x87
instruction it cannot reach from the entry through direct edges, and x87 state that is live at a call, a
return, an indirect branch or an external edge. The helpers are in tools/x87_soft.h, which the generated
source must include before the lowered functions (X87_HEADER).

Flags and the status word. In a function with an FCMOV, FNSTSW, SAHF, JP or JNP, each status flag that
any instruction reads must be defined on every path from the entry: the caller's flags are not an input
of the lowered function, a call leaves them undefined, and ADC, SBB, RCL, RCR and CMC carry an undefined
CF on into the flags and registers they write. The integer lowering models PF only after SAHF, the FCOMI
family and an 8-bit TEST or AND, so a reader of PF that another instruction may have written last is
rejected. SAHF,
JP and JNP are accepted only in such a function. C0, C2 and C3 are known after a comparison into the
status word, FPREM, FPREM1 or FNINIT, and unknown at the entry. FCOMI, FCOMIP, FUCOMI and FUCOMIP leave them as
they were, and every other x87 instruction leaves them undefined, as the SDM states, including WAIT (so
FSTSW AX, which is WAIT and FNSTSW AX, sees none of them), FNSTCW and FNSTSW itself; a call does too. C1 is
known only after FPREM, FPREM1 and FNINIT, until the next x87 instruction. TOP is the depth before FNSTSW
subtracted from 8, modulo 8. The exception flags, the stack fault and the busy bit are never known. FNSTSW AX leaves AX with only the known bits known. Any reader of AL, AH, AX, EAX or RAX that
sees an unknown bit is rejected, and so are a return, a call, an indirect branch and an external edge
while AX holds one, except:
- SAHF copies C0 to CF, C2 to PF, C3 to ZF and the busy bit to SF, each defined only when its bit is
  known (AF, from bit 1 of TOP, is left undefined);
- TEST AH, imm8 and AND AH, imm8 leave ZF, PF and SF undefined when the immediate selects an unknown bit,
  and AND makes the bits it clears known.
The flag rule then rejects a reader of a flag that an unknown bit left undefined.

FNSTSW m16 (and FSTSW m16, which is WAIT and FNSTSW m16) stores the status word to memory, which cannot
record an unknown bit, so it is accepted only where every bit but TOP is known on every path from the
entry: C0 to C3, C1, the exception flags IE to PE, SF, ES and the busy bit. Only FNINIT leaves them all known
(the whole word is clear), so only the checked entry reaches it, and only through instructions that are not
x87 (WAIT, FNSTCW, FNSTSW, a comparison and every other x87 instruction leave C0 to C3 undefined, and FCOMI,
FPREM and FPREM1 may raise an exception flag). Anywhere else the translation is rejected with the unknown
bits named, and the fixed profile rejects it. The stored word is x87_sw with TOP, which is 8 minus the
depth, modulo 8, in bits 11 to 13; it is a guest store of two bytes like FNSTCW m16.

FNSTENV and FNSAVE (the 28-byte and 108-byte 32-bit protected-mode layouts; the 16-bit operand-size forms are
rejected) store the status word too, so they follow the FNSTSW m16 rule: only right after FNINIT, where the
stack is empty and the word, the instruction and data pointers and the opcode are all clear. They store the
control word, the status word 0, the tag word 0xFFFF and zero pointers; FNSAVE then stores ST(0) to ST(7), the
profile's slots 7 to 0 (the SDM does not define the bytes of an empty register), and initializes as FNINIT does
(control word 0x037F, status word clear), so only the checked entry accepts it and the function must restore
0x027F before it returns. FSTENV and FSAVE are WAIT and the store; the WAIT leaves C0 to C3 undefined, as for
FSTSW, so they are rejected with the unknown bits named. FINIT then FNSTENV is accepted.

The checked entry, lower(..., x87=Checked(reject)), also accepts FLDCW m16. The control word becomes a
local of the lowered function, and every rounding helper, FNSTCW and the constant loads follow it. FLDCW
accepts a word with all six exceptions masked and precision control 00, 10 or 11. It keeps the word as
FNSTCW reads it back: bits 6 and 7 and 13 to 15 are reserved, bit 6 reads as 1 and the others as 0, and
bit 12 (the obsolete infinity control) is kept. Any other word, and a return with a control word other
than 0x027F, leaves the function through reject(status), the caller's statement, with a status from
STATUS. Guest stores before that point stay done. A function that loads the control word must not
contain a call, an indirect branch or an external edge, because the word may be live there.

The checked entry also accepts FNINIT (and FINIT, which is WAIT and FNINIT). It loads control word
0x037F, clears C0 to C3, and empties every register, so the depth after it is 0. Since every exception
is masked, no exception is pending and WAIT does nothing. The function must then restore 0x027F before
it returns, and it counts as loading the control word for the call and edge rule.

FBSTP m80bcd stores ST(0), rounded to an integer by the rounding control, as 18 packed BCD digits and
the sign of ST(0) (also for a value that rounds to zero), then pops. A NaN, an infinity, an unsupported
encoding or a rounded magnitude of 10^18 or more stores the packed BCD indefinite, the masked
invalid-operation response (x87_to_bcd in x87_soft.h). The checked entry also accepts FBLD m80bcd, which
pushes the 18 digits exactly, with the sign from bit 7 of byte 9 and the other bits of that byte
ignored. The SDM leaves the result undefined for a digit above 9, and the function leaves with status
"bcddigit" there. The fixed profile rejects FBLD.

The checked entry also accepts FPREM and FPREM1, which replace ST(0) by its remainder by ST(1) with the
quotient truncated (FPREM) or rounded to nearest-even (FPREM1), and set C0, C3 and C1 to bits 2, 1 and 0
of the quotient and C2 to 0 (x87_prem in x87_soft.h). Neither reads the control word. When the exponent
of ST(0) exceeds that of ST(1) by 64 or more, the instruction reduces ST(0) only partially, by an amount
the SDM leaves to the implementation, and sets C2; the function leaves with status "partial". A NaN
result (a NaN or unsupported operand, an infinite ST(0) or a zero ST(1)) follows the arithmetic NaN
rules, but the SDM does not give the status bits after it, so in a function with FNSTSW it leaves with
status "nanremainder". The usual fmod loop, FPREM; FNSTSW AX; SAHF; JP back, is accepted; in an
accepted run its back edge is never taken. The fixed profile rejects both instructions.
"""
from pathlib import Path

from iced_x86 import Code as C, FlowControl, MemorySize as S, MemorySizeExt, Mnemonic as M, OpKind as K, Register as R
from iced_x86 import InstructionInfoFactory, OpAccess, RegisterExt, RflagsBits as F

X87_HEADER = Path(__file__).with_name("x87_soft.h").read_text(encoding="ascii")
# Arithmetic with an m16 or m32 integer operand: mnemonic -> helper, with ST(0) as the destination.
IARITHMETIC = {M.FIADD: "x87_add", M.FISUB: "x87_sub", M.FISUBR: "x87_subr", M.FIMUL: "x87_mul",
               M.FIDIV: "x87_div", M.FIDIVR: "x87_divr"}
# Conditional register moves and the C++ condition each reads from the integer lowering's flags.
FCMOV = {M.FCMOVB: "carry", M.FCMOVE: "zero", M.FCMOVBE: "carry || zero", M.FCMOVNB: "!carry",
         M.FCMOVNE: "!zero", M.FCMOVNBE: "!carry && !zero", M.FCMOVU: "parity", M.FCMOVNU: "!parity"}
# 8087/287 interrupt-mask and protected-mode controls; the 80387 and later execute them as FNOP.
LEGACY_CONTROLS = {M.FNENI: "fneni", M.FNDISI: "fndisi", M.FNSETPM: "fnsetpm"}
X87_OPS = frozenset({M.FLD, M.FST, M.FSTP, M.FXCH, M.FCHS, M.FABS, M.FSQRT, M.FRNDINT, M.FLD1, M.FLDZ, M.FLDPI, M.FLDL2T, M.FLDL2E,
                     M.FLDLG2, M.FLDLN2, M.FILD, M.FIST, M.FISTP, M.FISTTP, M.FXTRACT, M.FSCALE, *FCMOV, M.FCOMI,
                     M.FCOMIP, M.FUCOMI, M.FUCOMIP, M.FADD, M.FADDP, M.FSUB, M.FSUBP, M.FSUBR, M.FSUBRP, M.FMUL, M.FMULP,
                     M.FDIV, M.FDIVP, M.FDIVR, M.FDIVRP, *IARITHMETIC, M.FNSTCW, M.WAIT, M.FLDCW, M.FCOM, M.FINCSTP,
                     M.FCOMP, M.FCOMPP, M.FUCOM, M.FUCOMP, M.FUCOMPP, M.FNSTSW, M.FTST, M.FICOM, M.FICOMP,
                     M.FXAM, M.FNOP, M.FNCLEX, M.FPREM, M.FPREM1, M.FNINIT, M.FNSTENV, M.FNSAVE, M.FBLD, M.FBSTP, *LEGACY_CONTROLS, M.FFREEP, M.FFREE})
# Comparisons into C0, C2 and C3 of the status word; FCOMPP and FUCOMPP pop twice, the other P forms once.
STATUS_COMPARES = frozenset({M.FCOM, M.FCOMP, M.FCOMPP, M.FUCOM, M.FUCOMP, M.FUCOMPP})
STATUS_POPS = {M.FCOM: 0, M.FCOMP: 1, M.FCOMPP: 2, M.FUCOM: 0, M.FUCOMP: 1, M.FUCOMPP: 2}
# FICOM and FICOMP compare ST(0) with an m16 or m32 integer, converted exactly as for FILD; FICOMP pops.
ICOMPARES = {M.FICOM: 0, M.FICOMP: 1}
# Every instruction that leaves C0, C2 and C3 known: the comparisons, FTST (ST(0) with +0.0) and FXAM.
STATUS_WRITERS = STATUS_COMPARES | {M.FTST, M.FXAM, *ICOMPARES}
# C3, C2 and C0 from x87_compare's result: 0 0 1 less, 1 0 0 equal, 0 0 0 greater, 1 1 1 unordered.
COMPARE_SW = "x87_sw = static_cast<uint16_t>(r == 0 ? 0x0100 : r == 1 ? 0x4000 : r == 3 ? 0x4500 : 0);"
# Status-word bits: C0, C2 and C3 (the comparison result), C1 (bit 0 of an FPREM quotient), and TOP.
C_BITS = 0x4500
C1_BIT = 0x0200
TOP_BITS = 0x3800
# IE, DE, ZE, OE, UE, PE, SF and ES (bits 0 to 7) and the busy bit (bit 15): the bits that FNINIT clears and that
# no other instruction of the profile leaves known. With C0 to C3 they are every bit of the status word but TOP.
EXCEPTION_BITS = 0x80FF
STATUS_BITS = C_BITS | C1_BIT | EXCEPTION_BITS
# The slots and the control word, which the fixed profile keeps at 0x027F.
DECLARATION = "X87Value x87_slot[8]{}; const uint16_t x87_cw = 0x027f; (void)x87_cw;"
# The checked entry's control word, which FLDCW changes.
CHECKED_DECLARATION = "X87Value x87_slot[8]{}; uint16_t x87_cw = 0x027f;"
# Profile-rejection statuses of the checked entry: an FLDCW operand with an exception unmasked, an FLDCW
# operand with the reserved precision control 01, a return with a control word other than 0x027F, an FPREM
# or FPREM1 that reduces only partially, and an FPREM or FPREM1 with a NaN result in a function that reads
# the status word.
STATUS = {"unmasked": 1, "precision": 2, "unrestored": 3, "partial": 4, "nanremainder": 5, "bcddigit": 6}
# The checked entry's partial remainders.
REMAINDERS = frozenset({M.FPREM, M.FPREM1})
# FNSTENV and FNSAVE store the 32-bit protected-mode environment (28 bytes: control word, status word, tag word,
# instruction and data pointers); FNSAVE follows it with the eight registers (108 bytes) and then initializes.
ENVIRONMENT = {M.FNSTENV: S.FPU_ENV28, M.FNSAVE: S.FPU_STATE108}


def declaration(body, checked=None):
    """The x87 locals of a lowered function: the slots, the control word and, in a function that compares
    into the status word or stores it, the condition bits of the status word."""
    text = DECLARATION if checked is None else CHECKED_DECLARATION
    if any(i.mnemonic in STATUS_WRITERS or i.mnemonic in REMAINDERS or i.mnemonic in (M.FNSTSW, M.FNINIT, *ENVIRONMENT) for i in body):
        text += " uint16_t x87_sw = 0; (void)x87_sw;"
    return text


# PF of the integer lowering: even parity of the low byte of v.
def parity_statement(v):
    return f"parity = ((0x6996u >> ((static_cast<unsigned>(static_cast<uint8_t>({v})) ^ " \
           f"(static_cast<unsigned>(static_cast<uint8_t>({v})) >> 4)) & 0xfu)) & 1u) == 0;"


class Checked:
    """The checked entry's error channel. reject is a format string with a {status} field: C++ that leaves
    the lowered function and hands the status to the caller, for example by setting a variable the
    enclosing entry declares and returning."""

    def __init__(self, reject):
        self.reject = reject

    def leave(self, name):
        return self.reject.format(status=STATUS[name])


def return_statement(checked):
    """A RET of the checked entry: the function must return with the profile's control word."""
    return f"if (x87_cw != 0x027f) {checked.leave('unrestored')} return rax;"

LOADS = {S.FLOAT32: "x87_from_f32(load<uint32_t>(a))", S.FLOAT64: "x87_from_f64(load<uint64_t>(a))",
         S.FLOAT80: "x87_from_m80(load<uint64_t>(a), load<uint16_t>(a + 8))"}
# An arithmetic memory operand keeps a signaling NaN signaling until the NaN rules have seen it.
OPERANDS = {S.FLOAT32: "x87_operand_f32(load<uint32_t>(a))", S.FLOAT64: "x87_operand_f64(load<uint64_t>(a))"}
# Two-operand register arithmetic: mnemonic -> helper. The P forms pop after writing ST(i).
ARITHMETIC = {M.FADD: "x87_add", M.FADDP: "x87_add", M.FSUB: "x87_sub", M.FSUBP: "x87_sub", M.FSUBR: "x87_subr",
              M.FSUBRP: "x87_subr", M.FMUL: "x87_mul", M.FMULP: "x87_mul", M.FDIV: "x87_div", M.FDIVP: "x87_div",
              M.FDIVR: "x87_divr", M.FDIVRP: "x87_divr"}
# Arithmetic with a memory operand or with ST(0), ST(i) / ST(i), ST(0), which keeps the stack depth.
TWO_FORM = frozenset({M.FADD, M.FSUB, M.FSUBR, M.FMUL, M.FDIV, M.FDIVR})
POPPING = frozenset({M.FADDP, M.FSUBP, M.FSUBRP, M.FMULP, M.FDIVP, M.FDIVRP})
# The constant loads push their value rounded to 64 significand bits under the rounding control, which
# the profile fixes at round to nearest-even; the precision control does not apply to them. Each entry is
# the (significand, sign and exponent) encoding.
CONSTANTS = {M.FLD1: (0x8000000000000000, 0x3FFF), M.FLDZ: (0, 0), M.FLDPI: (0xC90FDAA22168C235, 0x4000),
             M.FLDL2T: (0xD49A784BCD1B8AFE, 0x4000), M.FLDL2E: (0xB8AA3B295C17F0BC, 0x3FFF),
             M.FLDLG2: (0x9A209A84FBCFF799, 0x3FFD), M.FLDLN2: (0xB17217F7D1CF79AC, 0x3FFE)}
# The checked entry rounds the five inexact constants under its rounding control. All are positive, so
# toward negative infinity and toward zero give the significand below the exact value and toward positive
# infinity the one above; to nearest gives the CONSTANTS entry, which is one of the two. Measured on the
# AMD Ryzen 7 7700 under every precision and rounding control; the precision control has no effect.
BELOW = {M.FLDPI: 0xC90FDAA22168C234, M.FLDL2T: 0xD49A784BCD1B8AFE, M.FLDL2E: 0xB8AA3B295C17F0BB,
         M.FLDLG2: 0x9A209A84FBCFF798, M.FLDLN2: 0xB17217F7D1CF79AB}
# Integer memory operands: (signed type, unsigned type, bits).
INTEGERS = {S.INT16: ("int16_t", "uint16_t", 16), S.INT32: ("int32_t", "uint32_t", 32),
            S.INT64: ("int64_t", "uint64_t", 64)}
# Comparisons into EFLAGS; the P forms pop after comparing.
COMPARES = frozenset({M.FCOMI, M.FCOMIP, M.FUCOMI, M.FUCOMIP})
# Undocumented aliases of FSTP ST(i) (DF D0+i, DF D8+i), FXCH ST(i) (DD C8+i, DF C8+i), FCOM ST(i)
# (DC D0+i) and FCOMP ST(i) (DC D8+i, DE D0+i). Assemblers do not emit them and the fixtures do not measure
# them, so they stay outside the profile.
ALIASES = frozenset({C.FSTP_STI_DFD0, C.FSTP_STI_DFD8, C.FXCH_ST0_STI_DDC8, C.FXCH_ST0_STI_DFC8,
                     C.FCOM_ST0_STI_DCD0, C.FCOMP_ST0_STI_DCD8, C.FCOMP_ST0_STI_DED0})


# Every x87 mnemonic in iced starts with F (including FXSAVE/FXRSTOR, which store the x87 state, and
# FEMMS, which empties it); WAIT reports pending x87 exceptions. All of them touch the modelled state.
X87_STATE_OPS = frozenset(value for name, value in vars(M).items() if name.startswith("F")) | {M.WAIT}


def is_x87(i):
    return i.mnemonic in X87_STATE_OPS


def _discard_pairs(body):
    """FFREE ST(0) followed directly by FINCSTP: ip of the FINCSTP -> ip of the FFREE. Together they tag ST(0)
    empty and move TOP up by one, a pop without a store, so the stack stays contiguous."""
    index = {i.ip: i for i in body}
    return {i.next_ip: i.ip for i in body if i.mnemonic == M.FFREE and i.op0_register == R.ST0
            and i.next_ip in index and index[i.next_ip].mnemonic == M.FINCSTP}


def _effect(i, depth, checked=None, paired=False):
    """Stack depth after i, or a ValueError for an instruction or operand outside the profile. paired is
    whether i is a half of an FFREE ST(0), FINCSTP pair (_discard_pairs)."""
    op = i.mnemonic
    if paired and op == M.FFREE:  # ST(0) is freed; the FINCSTP right after it pops
        return depth
    if paired and op == M.FINCSTP:  # TOP moves up past the freed ST(0), as FFREEP ST(0) does
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth - 1
    if i.code in ALIASES:
        raise ValueError(f"unsupported x87 encoding at {i.ip:#x}: {i}")
    if op in (M.FNSTCW, M.WAIT) or (op == M.FLDCW and checked is not None):  # no register stack access
        return depth
    if op == M.FNINIT and checked is not None:  # every register becomes empty
        return 0
    if op == M.FNSTENV or (op == M.FNSAVE and checked is not None):
        if i.op0_kind != K.MEMORY or i.memory_size != ENVIRONMENT[op]:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if depth:  # the status word is known only after FNINIT, which leaves the stack empty (check_flags)
            raise ValueError(f"{i} at {i.ip:#x} stores the tag word of a non-empty x87 stack")
        return 0
    if op in (M.FNOP, M.FNCLEX, M.FXAM, *LEGACY_CONTROLS):  # no register stack change; FXAM also classifies an empty ST(0)
        return depth
    if op == M.FFREEP:  # ST(0) only: freeing another register would leave a hole below ST(0)
        if i.op0_register != R.ST0:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth - 1
    if op == M.FFREE:  # an occupied register would leave a hole, or keep TOP while the stack shrinks
        if i.op0_register - R.ST0 < depth:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i} frees an occupied register")
        return depth
    if op == M.FTST:
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op in ICOMPARES:
        if i.op0_kind != K.MEMORY or i.memory_size not in (S.INT16, S.INT32):
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth - ICOMPARES[op]
    if op in (M.FLD, M.FST, M.FSTP) and i.op0_kind == K.REGISTER:  # ST(i) copies
        if depth <= i.op0_register - R.ST0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        if op == M.FLD and depth == 8:
            raise ValueError(f"x87 stack overflow at {i.ip:#x}: {i}")
        return depth + {M.FLD: 1, M.FST: 0, M.FSTP: -1}[op]
    if op == M.FXCH and i.op_count == 2 and i.op0_register == R.ST0:
        if depth <= i.op1_register - R.ST0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op in FCMOV:
        if i.op_count != 2 or i.op0_register != R.ST0:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if depth <= i.op1_register - R.ST0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op in COMPARES:
        if i.op_count != 2 or i.op0_register != R.ST0:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if depth <= i.op1_register - R.ST0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth - 1 if op in (M.FCOMIP, M.FUCOMIP) else depth
    if op == M.FNSTSW:  # AX, or an m16 that check_flags accepts only where every status-word bit is known
        if not (i.op0_kind == K.REGISTER and i.op0_register == R.AX) \
                and not (i.op0_kind == K.MEMORY and i.memory_size == S.UINT16):
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        return depth
    if op in STATUS_COMPARES:
        if i.op_count == 0:  # FCOMPP and FUCOMPP compare ST(0) with ST(1)
            needed = 2
        elif i.op_count == 1:  # FCOM and FCOMP m32/m64
            if i.op0_kind != K.MEMORY or i.memory_size not in OPERANDS:
                raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
            needed = 1
        elif i.op0_register == R.ST0:
            needed = i.op1_register - R.ST0 + 1
        else:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if depth < max(needed, STATUS_POPS[op]):
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth - STATUS_POPS[op]
    if op in (M.FILD, M.FIST, M.FISTP, M.FISTTP):
        if i.op0_kind != K.MEMORY or i.memory_size not in INTEGERS:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if op == M.FILD:
            if depth == 8:
                raise ValueError(f"x87 stack overflow at {i.ip:#x}: {i}")
            return depth + 1
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth if op == M.FIST else depth - 1
    if op == M.FSCALE or (op in REMAINDERS and checked is not None):  # ST(0) and ST(1) stay on the stack
        if depth < 2:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op == M.FXTRACT:
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        if depth == 8:
            raise ValueError(f"x87 stack overflow at {i.ip:#x}: {i}")
        return depth + 1
    if op in CONSTANTS:
        if depth == 8:
            raise ValueError(f"x87 stack overflow at {i.ip:#x}: {i}")
        return depth + 1
    if op in (M.FCHS, M.FABS, M.FSQRT, M.FRNDINT):
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op == M.FST:  # FST m32/m64; there is no FST m80
        if i.op0_kind != K.MEMORY or i.memory_size not in OPERANDS:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op in (M.FLD, M.FSTP):
        if i.op0_kind != K.MEMORY or i.memory_size not in LOADS:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if op == M.FLD:
            if depth == 8:
                raise ValueError(f"x87 stack overflow at {i.ip:#x}: {i}")
            return depth + 1
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth - 1
    if op in IARITHMETIC:
        if i.op0_kind != K.MEMORY or i.memory_size not in (S.INT16, S.INT32):
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op in TWO_FORM and i.op_count == 1:
        if i.op0_kind != K.MEMORY or i.memory_size not in OPERANDS:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op in TWO_FORM and i.op_count == 2 and R.ST0 in (i.op0_register, i.op1_register):
        if depth <= max(i.op0_register, i.op1_register) - R.ST0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth
    if op in POPPING and i.op_count == 2 and i.op1_register == R.ST0 and i.op0_register != R.ST0:
        if depth <= i.op0_register - R.ST0 or depth < 2:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth - 1
    if op == M.FBSTP or (op == M.FBLD and checked is not None):  # packed BCD m80
        if i.op0_kind != K.MEMORY or i.memory_size != S.BCD:
            raise ValueError(f"unsupported x87 operand at {i.ip:#x}: {i}")
        if op == M.FBLD:
            if depth == 8:
                raise ValueError(f"x87 stack overflow at {i.ip:#x}: {i}")
            return depth + 1
        if depth == 0:
            raise ValueError(f"x87 stack underflow at {i.ip:#x}: {i}")
        return depth - 1
    raise ValueError(f"unsupported x87 operation at {i.ip:#x}: {i}")


def depths(body, exits, checked=None):
    """The x87 stack depth before each instruction of body, from an empty stack at body[0]. checked is the
    Checked entry's error channel, or None for the fixed control word."""
    index = {i.ip: i for i in body}
    pairs = _discard_pairs(body)
    depth = {body[0].ip: 0}
    work = [body[0].ip]
    loads_cw = checked is not None and any(i.mnemonic in (M.FLDCW, M.FNINIT, M.FNSAVE) for i in body)
    while work:
        i = index[work.pop()]
        before = depth[i.ip]
        paired = i.ip in pairs or i.ip in pairs.values()
        after = _effect(i, before, checked, paired) if is_x87(i) else before
        flow = i.flow_control
        if loads_cw and flow in (FlowControl.CALL, FlowControl.INDIRECT_CALL, FlowControl.INDIRECT_BRANCH):
            raise ValueError(f"x87 control word may be live at {i} at {i.ip:#x}")
        if flow == FlowControl.RETURN and before:
            raise ValueError(f"x87 stack not empty at return {i.ip:#x}")
        if flow in (FlowControl.CALL, FlowControl.INDIRECT_CALL) and before:
            raise ValueError(f"x87 state live across the call at {i.ip:#x}")
        if flow == FlowControl.INDIRECT_BRANCH and before:
            raise ValueError(f"x87 state live at the indirect branch {i.ip:#x}")
        successors = []
        if flow not in (FlowControl.RETURN, FlowControl.UNCONDITIONAL_BRANCH, FlowControl.INDIRECT_BRANCH):
            successors.append(i.next_ip)
        if flow in (FlowControl.CONDITIONAL_BRANCH, FlowControl.UNCONDITIONAL_BRANCH):
            successors.append(i.near_branch_target)
        for target in successors:
            if target in pairs and pairs[target] != i.ip:  # only the FFREE may reach the FINCSTP of a pair
                raise ValueError(f"x87 branch into the FFREE ST(0), FINCSTP pair at {i.ip:#x} -> {target:#x}")
            if target in index:
                if target not in depth:
                    depth[target] = after
                    work.append(target)
                elif depth[target] != after:
                    raise ValueError(f"x87 stack depth {depth[target]} and {after} meet at {target:#x}")
            elif target in exits and after:
                raise ValueError(f"x87 state live at the external edge {i.ip:#x} -> {target:#x}")
            elif target in exits and loads_cw:
                raise ValueError(f"x87 control word may be live at the external edge {i.ip:#x} -> {target:#x}")
    for i in body:
        if i.ip not in depth and is_x87(i):
            raise ValueError(f"unknown x87 stack depth at {i.ip:#x}: {i}")
    if any(i.mnemonic in GATED for i in body):
        check_flags(body, exits)
    return depth


# Instructions that put a function under the flag and status-word rules. SETP, SETNP, CMOVP and CMOVNP read PF as
# JP and JNP do (ffx_version_query.PF_CONSUMERS).
GATED = frozenset({*FCMOV, M.FNSTSW, *ENVIRONMENT, M.SAHF, M.JP, M.JNP, M.SETP, M.SETNP, M.CMOVP, M.CMOVNP})


# The six status flags. FCMOV reads ZF and CF; DF is not tracked, because the Windows x64 calling
# convention clears it at every call and return.
FLAG_NAMES = (("CF", F.CF), ("PF", F.PF), ("AF", F.AF), ("ZF", F.ZF), ("SF", F.SF), ("OF", F.OF))
STATUS_FLAGS = F.CF | F.PF | F.AF | F.ZF | F.SF | F.OF
# Shifts and rotates leave the flags unchanged when the masked count is 0 (an immediate 0 already writes
# no flags in iced).
SHIFTS = frozenset({M.SHL, M.SAL, M.SHR, M.SAR, M.ROL, M.ROR, M.RCL, M.RCR, M.SHLD, M.SHRD})


def _width(i):
    return RegisterExt.size(i.op0_register) * 8 if i.op0_kind == K.REGISTER else MemorySizeExt.size(i.memory_size) * 8


def _flags_after(i, before):
    """The status flags (as RflagsBits) defined after i, given those defined before it."""
    if i.flow_control in (FlowControl.CALL, FlowControl.INDIRECT_CALL):
        return 0  # the callee may leave any flags
    undefined = i.rflags_undefined & STATUS_FLAGS
    defines = (i.rflags_written | i.rflags_set | i.rflags_cleared) & STATUS_FLAGS & ~undefined
    if i.mnemonic in SHIFTS and defines:
        count = i.op2_kind if i.mnemonic in (M.SHLD, M.SHRD) else i.op1_kind
        width = _width(i)
        if count == K.REGISTER:  # a CL count of 0 changes nothing, so only flags already defined stay so
            if i.mnemonic in (M.SHLD, M.SHRD) and width == 16:
                return 0  # a count above 16 leaves the result and the flags undefined
            if i.mnemonic in (M.SHL, M.SAL, M.SHR, M.SAR) and width < 32:
                return before & ~undefined & ~F.CF  # CF is undefined for a count of the width or more
            return before & ~undefined
        n = i.immediate8 & (0x3F if width == 64 else 0x1F)
        if i.mnemonic in (M.SHLD, M.SHRD) and width == 16 and n > 16:
            return 0
        if i.mnemonic in (M.SHL, M.SAL, M.SHR, M.SAR) and n >= width:
            undefined |= F.CF
            defines &= ~F.CF
    if (i.has_rep_prefix or i.has_repe_prefix or i.has_repne_prefix) and defines:
        return before & ~undefined  # a zero count compares nothing and keeps the flags
    return (before & ~undefined) | defines


# The bits of AX that each part of RAX covers.
AX_PARTS = {R.AL: 0x00FF, R.AH: 0xFF00, R.AX: 0xFFFF, R.EAX: 0xFFFF, R.RAX: 0xFFFF}
# SAHF loads SF, ZF, AF, PF and CF from bits 7, 6, 4, 2 and 0 of AH. AF (bit 1 of TOP) is left undefined.
SAHF_BITS = ((F.CF, 0x0100), (F.PF, 0x0400), (F.ZF, 0x4000), (F.SF, 0x8000))
SAHF_FLAGS = F.CF | F.PF | F.AF | F.ZF | F.SF
# The flags of TEST and AND that depend on the bits the immediate selects.
RESULT_FLAGS = F.ZF | F.PF | F.SF
_INFO = InstructionInfoFactory()


def assigns_parity(i):
    """Whether the integer lowering sets PF for i: SAHF, the FCOMI family, the VEX scalar compares, and TEST and
    AND with an 8-bit destination."""
    if i.mnemonic in (M.SAHF, M.VCOMISS, M.VUCOMISS, M.VCOMISD, M.VUCOMISD) or i.mnemonic in COMPARES:
        return True
    if i.mnemonic not in (M.TEST, M.AND):
        return False
    size = MemorySizeExt.size(i.memory_size) if i.op0_kind == K.MEMORY else RegisterExt.size(i.op0_register)
    return size == 1


def _ah_immediate(i):
    """The immediate of TEST AH, imm8 or AND AH, imm8 as AX bits, or None for any other instruction."""
    if i.mnemonic in (M.TEST, M.AND) and i.op0_kind == K.REGISTER and i.op0_register == R.AH \
            and i.op1_kind == K.IMMEDIATE8:
        return (i.immediate8 & 0xFF) << 8
    return None


class _State:
    """What is known before an instruction: the defined status flags, the flags left undefined by unknown
    status-word bits (for the diagnostic), whether the integer lowering models PF, the known C0, C2 and C3
    bits of the status word, and the known bits of AX."""
    __slots__ = ("flags", "tainted", "parity", "c", "ax")

    def __init__(self, flags, tainted, parity, c, ax):
        self.flags, self.tainted, self.parity, self.c, self.ax = flags, tainted, parity, c, ax

    def key(self):
        return self.flags, self.tainted, self.parity, self.c, self.ax

    def join(self, other):
        return _State(self.flags & other.flags, self.tainted | other.tainted, self.parity and other.parity,
                      self.c & other.c, self.ax & other.ax)


def _ax_access(i):
    """The AX bits i reads and the AX bits it writes."""
    read = written = 0
    for used in _INFO.info(i).used_registers():
        bits = AX_PARTS.get(used.register, 0)
        if used.access in (OpAccess.READ, OpAccess.COND_READ, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE):
            read |= bits
        if used.access in (OpAccess.WRITE, OpAccess.READ_WRITE):
            written |= bits
    return read, written


def _after(i, s):
    """The state after i (the conditions it may raise are checked separately, in check_flags)."""
    op = i.mnemonic
    call = i.flow_control in (FlowControl.CALL, FlowControl.INDIRECT_CALL)
    flags = _flags_after(i, s.flags)
    modified = i.rflags_modified & STATUS_FLAGS
    tainted = 0 if call else s.tainted & ~modified
    parity = assigns_parity(i) if modified & F.PF else s.parity and not call
    ax = s.ax
    immediate = _ah_immediate(i)
    if op == M.SAHF:
        flags = s.flags & ~SAHF_FLAGS
        for flag, bit in SAHF_BITS:
            if s.ax & bit == bit:
                flags |= flag
            else:
                tainted |= flag
    elif immediate is not None:
        if immediate & ~s.ax:
            flags &= ~RESULT_FLAGS
            tainted |= RESULT_FLAGS
        if op == M.AND:  # a cleared bit is known to be 0
            ax |= 0xFF00 & ~immediate
    elif op == M.FNSTSW and i.op0_kind == K.REGISTER:
        ax = (s.c & ~EXCEPTION_BITS) | TOP_BITS
    elif call:
        ax = 0xFFFF  # the return value
    else:
        ax |= _ax_access(i)[1]
    if op in STATUS_WRITERS:
        c = C_BITS
    elif op in (M.FNINIT, M.FNSAVE):  # the whole status word is clear
        c = STATUS_BITS
    elif op in REMAINDERS:  # quotient bits and C2 = 0; the exception flags may be raised
        c = C_BITS | C1_BIT
    elif op in COMPARES:  # FCOMI leaves C0, C2 and C3, but may raise an exception flag
        c = s.c & ~EXCEPTION_BITS
    elif not (is_x87(i) or call):  # integer instructions change none of it
        c = s.c
    else:
        c = 0
    return _State(flags, tainted, parity, c, ax)


def _successors(i):
    flow = i.flow_control
    successors = []
    if flow not in (FlowControl.RETURN, FlowControl.UNCONDITIONAL_BRANCH, FlowControl.INDIRECT_BRANCH):
        successors.append(i.next_ip)
    if flow in (FlowControl.CONDITIONAL_BRANCH, FlowControl.UNCONDITIONAL_BRANCH):
        successors.append(i.near_branch_target)
    return successors


def states(body):
    """The _State before each reachable instruction of body, from body[0], where no flag, PF model or
    condition bit is known and AX holds no status word. The integer lowering starts its flags as false,
    which is not what a native caller passes in, so a reader of an undefined flag would see that
    initializer. The meet over joining paths keeps only what is known on all of them."""
    index = {i.ip: i for i in body}
    known = {body[0].ip: _State(0, 0, True, 0, 0xFFFF)}
    work = [body[0].ip]
    while work:
        i = index[work.pop()]
        after = _after(i, known[i.ip])
        for target in _successors(i):
            if target in index:
                joined = after if target not in known else known[target].join(after)
                if target not in known or joined.key() != known[target].key():
                    known[target] = joined
                    work.append(target)
    return known


def _bits(mask):
    return f"{mask:#06x}"


def check_flags(body, exits):
    """Reject a read of an undefined status flag, a read of PF that the integer lowering does not model, and
    a reader of AX, or a point where RAX leaves the function, that sees a status-word bit not known."""
    known = states(body)
    for i in body:
        s = known.get(i.ip)
        if s is None:
            continue  # not reachable from the entry
        missing = i.rflags_read & STATUS_FLAGS & ~s.flags
        if missing:
            names = " and ".join(name for name, bit in FLAG_NAMES if missing & bit)
            if missing & s.tainted == missing:
                raise ValueError(f"{i} at {i.ip:#x} reads {names}, which depends on a status-word bit that is not "
                                 "known on every path from the entry")
            raise ValueError(f"{i} at {i.ip:#x} reads {names}, which no instruction defines on every path from "
                             "the entry")
        if i.rflags_read & F.PF and not s.parity:
            raise ValueError(f"{i} at {i.ip:#x} reads PF, which the integer lowering models only after SAHF, "
                             "FCOMI, FCOMIP, FUCOMI, FUCOMIP and an 8-bit TEST or AND")
        if (i.mnemonic == M.FNSTSW or i.mnemonic in ENVIRONMENT) and i.op0_kind == K.MEMORY \
                and s.c & STATUS_BITS != STATUS_BITS:
            raise ValueError(f"{i} at {i.ip:#x} stores status-word bits {_bits(STATUS_BITS & ~s.c)}, which are not "
                             "known on every path from the entry (only FNINIT leaves all of them known)")
        unknown = 0xFFFF & ~s.ax
        if not unknown:
            continue
        immediate = _ah_immediate(i)
        if i.mnemonic != M.SAHF and immediate is None:
            read = _ax_access(i)[0] & unknown
            if read:
                raise ValueError(f"{i} at {i.ip:#x} reads AX bits {_bits(read)}, which FNSTSW AX leaves unknown "
                                 "on some path from the entry")
        flow = i.flow_control
        if flow in (FlowControl.RETURN, FlowControl.CALL, FlowControl.INDIRECT_CALL, FlowControl.INDIRECT_BRANCH):
            raise ValueError(f"RAX holds unknown status-word bits {_bits(unknown)} at {i} at {i.ip:#x}")
        for target in _successors(i):
            if target in exits:
                raise ValueError(f"RAX holds unknown status-word bits {_bits(unknown)} at the external edge "
                                 f"{i.ip:#x} -> {target:#x}")


def _environment(i, address, checked):
    """FNSTENV and FNSAVE right after FNINIT: control word, status word 0, tag word 0xFFFF (every register empty)
    and the instruction and data pointers and opcode, which FNINIT clears; the reserved halfwords read 0xFFFF. FNSAVE adds ST(0) to ST(7), which are
    the slots 7 down to 0, and initializes as FNINIT does."""
    word = "0x027f" if checked is None else "x87_cw"
    # Native FNINIT; FNSTENV/FNSAVE write 0xffff in the reserved halfwords at 2, 6, 10 and 26 (measured on two hosts).
    stores = [f"store<uint16_t>(a, {word});", "store<uint16_t>(a + 2, 0xffff);",
              "store<uint16_t>(a + 4, 0);", "store<uint16_t>(a + 6, 0xffff);",
              "store<uint16_t>(a + 8, 0xffff);", "store<uint16_t>(a + 10, 0xffff);"]
    stores += [f"store<uint32_t>(a + {offset}, 0);" for offset in (12, 16, 20)]
    stores += ["store<uint16_t>(a + 24, 0);", "store<uint16_t>(a + 26, 0xffff);"]
    if i.mnemonic == M.FNSTENV:
        return f"{{ const word a = {address()}; {' '.join(stores)} }}  // fnstenv"
    for k in range(8):
        stores.append(f"store<uint64_t>(a + {28 + 10 * k}, x87_slot[{7 - k}].sig); "
                      f"store<uint16_t>(a + {36 + 10 * k}, x87_slot[{7 - k}].se);")
    return f"{{ const word a = {address()}; {' '.join(stores)} }} x87_cw = 0x037f; x87_sw = 0;  // fnsave"


def statement(i, depth, address, checked=None, parity=False, read="load", status_word=False):
    """C++ for one profile instruction at a known stack depth; address() gives the memory operand. checked
    is the Checked entry's error channel, or None for the fixed control word. parity is whether the function
    declares the integer lowering's PF, which the FCOMI family then sets. read names the template that reads
    a memory operand: the integer lowering passes load_always for guest memory, which the compiler must read
    even when the value is unused, and load for the function's own frame. status_word is whether the
    function contains FNSTSW, the only reader of the status word."""
    return _statement(i, depth, address, checked, parity, status_word).replace("load<", f"{read}<")


def _statement(i, depth, address, checked, parity, status_word=False):
    op = i.mnemonic
    # depths() accepts FINCSTP, and FFREE of an occupied ST(0), only as an FFREE ST(0), FINCSTP pair.
    _effect(i, depth, checked, op == M.FINCSTP or (op == M.FFREE and i.op0_register == R.ST0))
    top = depth - 1  # ST(k) is x87_slot[top - k]
    if op == M.FNSTCW:  # the fixed profile's control word is 0x027F
        word = "0x027f" if checked is None else "x87_cw"
        return f"{{ const word a = {address()}; store<uint16_t>(a, {word}); }}  // fnstcw"
    if op == M.FLDCW:  # only the checked entry reaches this
        return (f"{{ const word a = {address()}; const uint16_t w = load<uint16_t>(a); "
                f"if ((w & 0x3f) != 0x3f) {checked.leave('unmasked')} "
                f"if ((w & 0x300) == 0x100) {checked.leave('precision')} "
                "x87_cw = static_cast<uint16_t>((w & 0x1f3f) | 0x40); }  // fldcw")
    if op == M.FNINIT:  # only the checked entry reaches this; the registers are empty from here on
        return "x87_cw = 0x037f; x87_sw = 0;  // fninit"
    if op in ENVIRONMENT:  # only after FNINIT (check_flags), so the stack is empty and the word is exact
        return _environment(i, address, checked)
    if op == M.WAIT:  # every exception is masked, so none is pending
        return ";  // wait"
    if op in (M.FNOP, M.FNCLEX):  # the profile keeps no exception flag, so clearing them changes nothing
        return f";  // {'fnop' if op == M.FNOP else 'fnclex'}"
    if op in LEGACY_CONTROLS:  # executed as FNOP since the 80387
        return f";  // {LEGACY_CONTROLS[op]}"
    if op == M.FFREEP:  # the pop is the depth change; the freed slot is dead from here on
        return ";  // ffreep st(0)"
    if op == M.FFREE:  # the register is already empty, or ST(0) of a pair whose FINCSTP pops it
        return f";  // ffree st({i.op0_register - R.ST0})"
    if op == M.FINCSTP:  # the pop is the depth change; the freed slot is dead from here on
        return ";  // fincstp"
    if op == M.FTST:  # ST(0) compared with +0.0
        return f"{{ const int r = x87_compare(x87_slot[{top}], X87Value{{0, 0}}); {COMPARE_SW} }}  // ftst"
    if op in ICOMPARES:
        signed, unsigned, bits = INTEGERS[i.memory_size]
        name = "ficom" if op == M.FICOM else "ficomp"
        return (f"{{ const word a = {address()}; const int r = x87_compare(x87_slot[{top}], "
                f"x87_from_integer(static_cast<{signed}>(load<{unsigned}>(a)))); {COMPARE_SW} }}  // {name} m{bits}")
    if op == M.FXAM:  # an empty ST(0) is C3 and C0
        if depth == 0:
            return "x87_sw = 0x4100;  // fxam, empty"
        return f"x87_sw = x87_examine(x87_slot[{top}]);  // fxam"
    if op in (M.FLD, M.FST, M.FSTP) and i.op0_kind == K.REGISTER:
        k = i.op0_register - R.ST0
        if op == M.FLD:  # push a copy of ST(k): every encoding is copied as it is
            return f"x87_slot[{depth}] = x87_slot[{top - k}];  // fld st({k})"
        return f"x87_slot[{top - k}] = x87_slot[{top}];  // {'fst' if op == M.FST else 'fstp'} st({k})"
    if op == M.FXCH:
        k = top - (i.op1_register - R.ST0)
        return f"{{ const X87Value t = x87_slot[{top}]; x87_slot[{top}] = x87_slot[{k}]; x87_slot[{k}] = t; }}  // fxch"
    if op in FCMOV:  # the whole 80-bit encoding is copied, as for FLD ST(i)
        k = i.op1_register - R.ST0
        name = next(n for n, v in vars(M).items() if v == op).lower()
        return f"if ({FCMOV[op]}) x87_slot[{top}] = x87_slot[{top - k}];  // {name} st({k})"
    if op == M.FNSTSW:  # C0, C2 and C3 of the last comparison, and TOP; the other bits read as 0
        stack_top = (8 - depth) & 7
        if i.op0_kind == K.MEMORY:  # every bit is known here (check_flags), so the word is exact
            return (f"{{ const word a = {address()}; store<uint16_t>(a, static_cast<uint16_t>(x87_sw | "
                    f"{stack_top << 11:#06x})); }}  // fnstsw m16, top {stack_top}")
        return (f"rax = (rax & ~0xffffull) | static_cast<uint16_t>(x87_sw | {stack_top << 11:#06x});  "
                f"// fnstsw ax, top {stack_top}")
    if op in STATUS_COMPARES:  # C3, C2 and C0: 0 0 1 less, 1 0 0 equal, 0 0 0 greater, 1 1 1 unordered
        name = next(n for n, v in vars(M).items() if v == op).lower()
        if i.op_count == 1:
            prefix, other = f"const word a = {address()}; ", OPERANDS[i.memory_size]
        else:
            k = 1 if i.op_count == 0 else i.op1_register - R.ST0
            prefix, other = "", f"x87_slot[{top - k}]"
        return (f"{{ {prefix}const int r = x87_compare(x87_slot[{top}], {other}); "
                "x87_sw = static_cast<uint16_t>(r == 0 ? 0x0100 : r == 1 ? 0x4000 : r == 3 ? 0x4500 : 0); }  "
                f"// {name}")
    if op in COMPARES:  # ZF and CF: 1 and 0 for equal, 0 and 1 for less, 0 and 0 for greater, 1 and 1 unordered
        k = top - (i.op1_register - R.ST0)  # OF is always cleared, and PF is 1 only for unordered
        return (f"{{ const int r = x87_compare(x87_slot[{top}], x87_slot[{k}]); "
                f"{'parity = r == 3; ' if parity else ''}zero = r == 1 || r == 3; carry = r == 0 || r == 3; "
                "overflow = false; }")
    if op in (M.FILD, M.FIST, M.FISTP, M.FISTTP):
        signed, unsigned, bits = INTEGERS[i.memory_size]
        if op == M.FILD:
            return (f"{{ const word a = {address()}; "
                    f"x87_slot[{depth}] = x87_from_integer(static_cast<{signed}>(load<{unsigned}>(a))); }}")
        chop = "true" if op == M.FISTTP else "false"
        return (f"{{ const word a = {address()}; "
                f"store<{unsigned}>(a, static_cast<{unsigned}>(x87_to_integer(x87_slot[{top}], {bits}, {chop}, x87_cw))); }}")
    if op == M.FBLD:  # only the checked entry reaches this; a digit above 9 leaves the function
        return (f"{{ const word a = {address()}; if (!x87_from_bcd(load<uint64_t>(a), load<uint16_t>(a + 8), "
                f"x87_slot[{depth}])) {checked.leave('bcddigit')} }}  // fbld")
    if op == M.FBSTP:
        return (f"{{ const word a = {address()}; uint64_t low; uint16_t high; "
                f"x87_to_bcd(x87_slot[{top}], x87_cw, low, high); store<uint64_t>(a, low); "
                "store<uint16_t>(a + 8, high); }  // fbstp")
    if op in IARITHMETIC:  # ST(0) = ST(0) op integer, the integer converted exactly as for FILD
        signed, unsigned, bits = INTEGERS[i.memory_size]
        name = next(n for n, v in vars(M).items() if v == op).lower()
        return (f"{{ const word a = {address()}; x87_slot[{top}] = {IARITHMETIC[op]}(x87_slot[{top}], "
                f"x87_from_integer(static_cast<{signed}>(load<{unsigned}>(a))), x87_cw); }}  // {name} m{bits}")
    if op == M.FSCALE:
        return f"x87_slot[{top}] = x87_scale(x87_slot[{top}], x87_slot[{top - 1}], x87_cw);"
    if op in REMAINDERS:  # only the checked entry reaches this; C0, C3 and C1 are bits 2, 1 and 0 of q, C2 0
        name, nearest = ("fprem1", "true") if op == M.FPREM1 else ("fprem", "false")
        nan = f" if (q == x87_prem_nan) {checked.leave('nanremainder')}" if status_word else ""
        return (f"{{ const int q = x87_prem(x87_slot[{top}], x87_slot[{top - 1}], {nearest}, x87_slot[{top}]); "
                f"if (q == x87_prem_partial) {checked.leave('partial')}{nan} "
                "x87_sw = static_cast<uint16_t>(q < 0 ? 0 : (q & 4 ? 0x0100 : 0) | (q & 2 ? 0x4000 : 0) | "
                f"(q & 1 ? 0x0200 : 0)); }}  // {name}")
    if op == M.FXTRACT:  # ST(1) becomes the exponent and the pushed ST(0) the significand
        return (f"{{ X87Value e; const X87Value s = x87_xtract(x87_slot[{top}], e); "
                f"x87_slot[{top}] = e; x87_slot[{depth}] = s; }}  // fxtract")
    if op in CONSTANTS:
        sig, se = CONSTANTS[op]
        name = next(n for n, v in vars(M).items() if v == op).lower()
        if checked is not None and op in BELOW:  # up for toward +inf, CONSTANTS for nearest, else BELOW
            below = BELOW[op]
            return (f"x87_slot[{depth}] = X87Value{{x87_rounding(x87_cw) == 2 ? {below + 1:#018x}ull : "
                    f"x87_rounding(x87_cw) == 0 ? {sig:#018x}ull : {below:#018x}ull, {se:#06x}}};  // {name}")
        return f"x87_slot[{depth}] = X87Value{{{sig:#018x}ull, {se:#06x}}};  // {name}"
    if op in (M.FSQRT, M.FRNDINT):
        return f"x87_slot[{top}] = {'x87_sqrt' if op == M.FSQRT else 'x87_rndint'}(x87_slot[{top}], x87_cw);"
    if op in (M.FCHS, M.FABS):
        return f"x87_slot[{top}] = {'x87_chs' if op == M.FCHS else 'x87_abs'}(x87_slot[{top}]);"
    if op == M.FLD:
        return f"{{ const word a = {address()}; x87_slot[{depth}] = {LOADS[i.memory_size]}; }}"
    if op in (M.FST, M.FSTP):  # FST m32/m64 stores without popping
        v = f"x87_slot[{top}]"
        store = {S.FLOAT32: f"store<uint32_t>(a, x87_to_f32({v}, x87_cw));",
                 S.FLOAT64: f"store<uint64_t>(a, x87_to_f64({v}, x87_cw));",
                 S.FLOAT80: f"store<uint64_t>(a, {v}.sig); store<uint16_t>(a + 8, {v}.se);"}[i.memory_size]
        return f"{{ const word a = {address()}; {store} }}" + ("  // fst m" if op == M.FST else "")
    if i.op_count == 1:  # arithmetic with an m32/m64 operand: ST(0) = ST(0) op operand
        return (f"{{ const word a = {address()}; "
                f"x87_slot[{top}] = {ARITHMETIC[op]}(x87_slot[{top}], {OPERANDS[i.memory_size]}, x87_cw); }}")
    # The destination is the first operand: ST(i) for FMUL ST(i), ST(0) and the P forms, ST(0) otherwise.
    destination, source = (top - (i.op0_register - R.ST0), top - (i.op1_register - R.ST0))
    return f"x87_slot[{destination}] = {ARITHMETIC[op]}(x87_slot[{destination}], x87_slot[{source}], x87_cw);"
