#!/usr/bin/env python3
"""Statically resolved RDNA2 far early-out control flow through the two-exit profile.

  (no arguments)                       pure-Python controls, extraction and negative controls
                                       (the extraction runs when an LLVM 15 library is found, and
                                       is skipped with a line otherwise)
  binary <llvm-mc> [libLLVM]
                                       reassemble the fixture sources and compare with the words below;
                                       with a library the extraction runs against it as well
  controls <libLLVM>                 the same full run with the extraction driven through the
                                       given LLVM 15 library (LLVM-21 is accepted for a local run;
                                       a recorded CI run should use an LLVM 15 build)

The fixture builds one gfx1013 pixel program in three variants, all with the same per-pixel arithmetic
and the same two exports: a uniform root-constant compare selects an early path or a normal path, and
the early path transfers to a pure terminating continuation (NOPs, then the program's final s_endpgm):

  - short: the transfer is a plain s_branch onto the continuation;
  - calculated: the documented far-transfer chain s_getpc_b64 / s_add_u32 displacement /
    s_addc_u32 sign extension / s_setpc_b64 over a short distance;
  - far: the same chain over a distance the SOPP branch word cannot encode (>= 32768 reachable
    NOP DWORDs sit in the normal path, and the displacement measured from the actual assembled
    offsets exceeds the 131068-byte positive s_branch range).

Every variant is checked three ways: the startup Stage must translate it (its blocks show the jump),
the emitted dispatcher must reproduce the independent per-lane arithmetic oracle for both selector
values, and the bounded FSSM v0x101 extraction of pixel_aot must return the original code exactly.
The negative matrix refuses a register-derived displacement, a missing ADDC, a wrong sign extension,
a signed ADD, a competing entry into the chain, pair clobbering, an address escape, an SCC escape, a
stale scratch pair, an effectful continuation (export, store, barrier, EXEC write), a continuation
ending at the first rather than the final s_endpgm, a target off an
instruction boundary and a target outside the program, each with a named diagnostic.
"""
import re
import struct
import subprocess
import ctypes.util
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))

import gfx2hlsl as G  # noqa: E402
import pixel_aot  # noqa: E402
import disasm_rdna2 as D  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402


def resolve_libllvm():
    """The LLVM 15 shared library for the extraction check, or None when this host has none.

    The binary mode takes the library as an optional argument; the plain mode searches the common
    install paths, following the discovery order tests/rdna2_tools_check.py uses.
    """
    candidates = ['/usr/lib64/llvm15/lib/libLLVM-15.so', 'libLLVM-15.so.1', 'libLLVM-15.so',
                  'libLLVM.so.15']
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return str(path)
        found = ctypes.util.find_library(candidate)
        if found and Path(found).is_file():
            return found
        for directory in ('/usr/lib/x86_64-linux-gnu', '/usr/lib64/llvm15/lib', '/usr/lib64',
                          '/usr/lib', '/usr/local/lib'):
            option = Path(directory) / candidate
            if option.is_file():
                return str(option)
    return None


LIBRARY = resolve_libllvm()


class Pipe:
    root = {'params': [{'type': 1, 'count': 1}], 'flags': 0}

    def ud(self, key):
        return 0

    def mrt_kind(self, index):
        assert index == 0
        return 'uint'

# The shared per-pixel arithmetic: p = (x & 0xffff) | (y << 16) over the entry I/J words, then the
# early export (p, p+3, p^0x80000000, p+3) and the normal export (p, p+19, p^0x80000000, p+19),
# all modulo 2^32 (R32G32B32A32_UINT).
PREFIX = (
    ('v_mov_b32_e32 v4, v0', '0003087e'),
    ('v_and_b32_e32 v4, 0xffff, v4', 'ff080836ffff0000'),
    ('v_lshlrev_b32_e32 v5, 16, v1', '90020a34'),
    ('v_or_b32_e32 v4, v4, v5', '040b0838'),
    ('v_mov_b32_e32 v1, v4', '0403027e'),
    ('v_add_nc_u32_e32 v6, 3, v4', '83080c4a'),
    ('v_xor_b32_e32 v7, 0x80000000, v4', 'ff080e3a00000080'),
    ('v_add_nc_u32_e32 v8, 19, v4', '9308104a'),
)
COMPARE = ('s_cmp_eq_u32 s8, 0', '088006bf')
EXP_A = ('exp mrt0 v4, v6, v7, v6 done vm', '0f1800f804060706')
EXP_B = ('exp mrt0 v4, v8, v7, v8 done vm', '0f1800f804080708')
END = ('s_endpgm', '000081bf')
NOP = ('s_nop 0', '000080bf')
GETPC = ('s_getpc_b64 s[20:21]', '001f94be')
ADDC_0 = ('s_addc_u32 s21, s21, 0', '15801582')
SETPC = ('s_setpc_b64 s[20:21]', '142080be')
# The SOPP branch word is a signed 16-bit count of DWORDs from the instruction after the branch, so
# 131068 bytes is the largest forward distance one s_branch can encode.
SOPP_LIMIT = 131068
# The far variant's normal path holds this many reachable NOP DWORDs.
FAR_NOP_DWORDS = 32768


def sopp(name, opcode, dwords):
    return ('%s %d' % (name, dwords), struct.pack('<I', 0xbf800000 | (opcode << 16) | (dwords & 0xffff)).hex())


def cbranch_scc0(dwords):
    return sopp('s_cbranch_scc0', 4, dwords)


def s_branch(dwords):
    return sopp('s_branch', 2, dwords)


def sop2_literal(opcode, sdst, ssrc0, literal):
    word = 0x80000000 | (opcode << 23) | (sdst << 16) | (0xff << 8) | ssrc0
    return struct.pack('<I', word).hex() + struct.pack('<i', literal).hex()


def sop2_inline(opcode, sdst, ssrc0, inline):
    word = 0x80000000 | (opcode << 23) | (sdst << 16) | ((128 + inline) << 8) | ssrc0
    return struct.pack('<I', word).hex()


def add_u32(imm):
    """s_add_u32 s20, s20, imm as llvm-mc encodes it: an inline constant up to 64, a literal after."""
    if 0 <= imm <= 64:
        return ('s_add_u32 s20, s20, %d' % imm, sop2_inline(0, 20, 20, imm))
    return ('s_add_u32 s20, s20, %d' % imm, sop2_literal(0, 20, 20, imm))


def add_i32(imm):
    """s_add_i32 s20, s20, imm: the signed SOP2 form the negative matrix substitutes."""
    if 0 <= imm <= 64:
        return ('s_add_i32 s20, s20, %d' % imm, sop2_inline(2, 20, 20, imm))
    return ('s_add_i32 s20, s20, %d' % imm, sop2_literal(2, 20, 20, imm))


# A transfer is a list of rows; the marker 'ADD' names the low-word ADD whose immediate is the
# displacement from the GETPC successor to the continuation, resolved by the layout solver.
CHAIN = [GETPC, 'ADD', ADDC_0, SETPC]
BRANCH = ['BRANCH']


def variant(transfer, nops=0):
    """(rows, code, offsets) of one variant.

    Layout: prefix; compare; entry branch over the early path to the normal block; early export; the
    transfer; normal export; NOPs; first s_endpgm; continuation (NOP, final s_endpgm). The entry
    branch and the symbolic ADD are re-encoded from the offsets until both are stable, because a
    displacement above 64 widens the ADD from 4 to 8 bytes and moves the continuation.
    """
    rows = list(PREFIX) + [COMPARE, 'CBRANCH', EXP_A] + list(transfer) \
        + [EXP_B] + [NOP] * nops + [END, NOP, END]
    entry = len(PREFIX) + 1
    add_at = next((entry + 2 + i for i, row in enumerate(transfer) if row == 'ADD'), None)
    branch_at = entry + 2 if not transfer or transfer[0] == 'BRANCH' else None
    normal = entry + 2 + len(transfer)
    continuation = normal + 1 + nops + 1
    displacement = 0
    for _ in range(8):
        materialized = []
        for row in rows:
            if row == 'CBRANCH' or row == 'BRANCH':
                materialized.append(None)  # a placeholder of the same 4-byte width
            elif row == 'ADD':
                materialized.append(add_u32(displacement))  # keeps the width of the last pass
            else:
                materialized.append(row)
        offsets, words = [], bytearray()
        for row in materialized:
            offsets.append(len(words))
            words += bytes(4) if row is None else bytes.fromhex(row[1])
        if add_at is not None:
            displacement = offsets[continuation] - offsets[add_at]
            materialized[add_at] = add_u32(displacement)
        else:
            materialized[branch_at] = s_branch((offsets[continuation] - offsets[branch_at] - 4) // 4)
        materialized[entry] = cbranch_scc0((offsets[normal] - offsets[entry] - 4) // 4)
        offsets, words = [], bytearray()
        for row in materialized:
            offsets.append(len(words))
            words += bytes.fromhex(row[1])
        if add_at is None or materialized[add_at][1] == add_u32(offsets[continuation] - offsets[add_at])[1]:
            return list(materialized), bytes(words), offsets
    raise AssertionError('the layout solver did not settle')


def words_of(rows):
    return b''.join(bytes.fromhex(row[1]) for row in rows)


def source_of(rows):
    return '\n'.join(row[0] for row in rows) + '\n'


def decode_all(code):
    out, off = [], 0
    while off < len(code):
        n, name, asm, _ = D.decode_instruction(code, off)
        out.append((off, n, name, asm))
        off += n
    return out


# ---- independent oracle: documented RDNA 2 semantics of the fixture ------------------------------
def oracle(selector, x, y):
    """The four exported words for one selector value, from plain arithmetic over the instructions."""
    p = (x & 0xffff) | ((y << 16) & 0xffffffff)
    if selector != 0:  # s8 != 0: the compare is false, SCC0 is taken, the normal path runs
        return (p, (p + 19) & 0xffffffff, p ^ 0x80000000, (p + 19) & 0xffffffff)
    return (p, (p + 3) & 0xffffffff, p ^ 0x80000000, (p + 3) & 0xffffffff)


# ---- interpreter of the emitted dispatcher: the pc machine Stage.body actually emits -------------
def block_statements(body, pc):
    statements, inside = [], False
    for line in body:
        head = line.strip()
        if head in ('}', ''):
            if inside and head == '}':
                return statements
            continue
        if head.startswith('if (pc == ') or head.startswith('else if (pc == '):
            if inside:
                return statements
            index = head.index('pc == ') + len('pc == ')
            if int(head[index:head.rindex(')')]) == pc:
                inside = True
            continue
        if inside:
            statements.append(head)
    raise AssertionError('pc %d has no dispatcher block' % pc)


def run_dispatcher(body, ud0, x, y, steps=64):
    """Execute the emitted if/else pc chain directly and return the exported uint4 words."""
    values = {'v0': x, 'v1': y, 'ud0': ud0, 'scc': 0, 'vcc': 0, 'exec': 1, 'm0': 0}
    for i in (4, 5, 6, 7, 8):
        values['v%d' % i] = 0
    mrt = None
    pc = 0
    for _ in range(steps):
        if pc == 0xffff:
            return mrt
        for statement in block_statements(body, pc):
            if statement.startswith('pc = ('):
                cond = statement[len('pc = ('):statement.index(') ?')]
                taken = Expr(cond).run(values)[0]
                pc = int(statement[statement.index('? ') + 2:statement.index(' :')], 0) if taken else \
                    int(statement[statement.rindex(': ') + 2:-1], 0)
            elif statement.startswith('pc = '):
                pc = int(statement[len('pc = '):-2], 0) if statement.endswith('u;') \
                    else int(statement[len('pc = '):-1], 0)
            elif statement.startswith('o.mrt0'):
                mrt = tuple(Expr(t).run(values)[0] for t in
                            statement[statement.index('(') + 1:statement.rindex(')')].split(', '))
            else:
                name, _, expr = statement.partition(' = ')
                values[name] = Expr(expr[:-1]).run(values)[0]
    raise AssertionError('dispatcher did not reach pc == 0xffff')


# ---- bounded FSSM v0x101 envelope ---------------------------------------------------------------
def shader_record(blob: bytes, name: bytes) -> bytes:
    encoder = zlib.compressobj(wbits=-15)
    payload = encoder.compress(blob) + encoder.flush()
    return (bytes([len(name) + 1]) + name + b'\0\0'
            + struct.pack('<4I', len(payload), 0, len(payload), len(blob)) + payload)


def envelope(code: bytes, name: bytes = b'fixture') -> bytes:
    program_start = 288
    base = program_start - 40
    pad = (-(40 + len(code)) % 16) or 16
    body = code + b'\x00\x00\x9f\xbf' * (pad // 4)
    extent = 40 + len(body)
    image = bytearray(program_start) + body
    image[:4] = b'FSSM'
    struct.pack_into('<I', image, 192, program_start - 40)
    struct.pack_into('<I', image, 4, 0x101)
    struct.pack_into('<Q', image, 200, extent)
    struct.pack_into('<Q', image, 232, base + extent)
    native = b'XBCH' + bytes(12) + struct.pack('<II', len(image), 0) + zlib.compress(image)
    program = (bytes(24) + b'XDAT' + struct.pack('<II', 1, 0)
               + struct.pack('<III', 13, 48, len(native)) + native)
    sections = [b'DXIL' + struct.pack('<I', len(program)) + program]
    offset = 32 + 4 * len(sections)
    offsets = bytearray()
    for section in sections:
        offsets.extend(struct.pack('<I', offset))
        offset += len(section)
    blob = b'DXBC' + bytes(16) + struct.pack('<3I', 1, offset, len(sections)) + offsets + b''.join(sections)
    return shader_record(blob, name)


def extract(directory: Path, code: bytes, library: str | None = LIBRARY) -> bytes:
    archive = directory / 'far-early-out.rpf'
    archive.write_bytes(envelope(code))
    return pixel_aot.native_program(archive, 'fixture', llvm=library)[1]


# ---- checks ---------------------------------------------------------------------------------------
def check_variant(name, rows, code, offsets, continuation, directory, *, displacement=None,
                  library=LIBRARY):
    stage = G.Stage(Pipe(), 'ps', code)
    body = stage.body()
    # The emitted CFG: the entry branches on the root constant; the transfer is a jump block.
    terms = [term for _off, _lines, term in stage.blocks if term]
    assert terms[0][0] == 'branch', (name, stage.blocks)
    jumps = [term for term in terms if term[0] == 'jump']
    assert jumps and jumps[-1][1] == offsets[continuation], (name, stage.blocks)
    # Both selector values reproduce the independent arithmetic oracle on several entry words.
    for ud0 in (0, 1):
        for x, y in ((0, 0), (5, 9), (0x1234, 0x5678), (0xffff, 0xffff)):
            got = run_dispatcher(body, ud0, x, y)
            assert got == oracle(ud0, x, y), (name, ud0, x, y, got)
    # The bounded FSSM v0x101 extraction returns the original code exactly; without an LLVM 15
    # library on this host it is skipped, and the binary mode of a machine that has one covers it.
    extracted = 'extraction skipped (no LLVM 15 library found)' if library is None else (
        'extracted exactly' if extract(directory, code, library) == code else 'MISMATCH')
    assert extracted in ('extracted exactly', 'extraction skipped (no LLVM 15 library found)'), \
        (name, extracted)
    if displacement is not None:
        assert displacement > SOPP_LIMIT, (name, displacement)
    print('%s: translated, dispatcher oracle matched both selectors, %s (%d bytes)'
          % (name, extracted, len(code)))


def negative(name, code, expected, extractor):
    """A refusal with the named diagnostic, from the startup Stage and the pixel_aot extraction."""
    try:
        G.Stage(Pipe(), 'ps', code)
    except G.Unsupported as reason:
        assert expected in str(reason), (name, str(reason))
    else:
        raise AssertionError('%s: the startup translator accepted a refused program' % name)
    if extractor is None:
        print('%s: refused by the translator (%s); extraction skipped (no LLVM 15 library found)'
              % (name, expected))
        return
    try:
        extractor(code)
    except ValueError as reason:
        assert expected in str(reason), (name, str(reason))
    else:
        raise AssertionError('%s: the extractor accepted a refused program' % name)
    print('%s: refused (%s)' % (name, expected))


def replace_row(rows, index, row):
    out = list(rows)
    out[index] = row
    return out


def drop_row(rows, index):
    return rows[:index] + rows[index + 1:]


def check_negatives(directory, rows, code, far_rows, far_code, library=LIBRARY):
    ins = decode_all(code)
    index_of = {off: i for i, (off, _n, _name, _asm) in enumerate(ins)}
    asm_of = {i: asm for i, (_off, _n, _name, asm) in enumerate(ins)}
    getpc = next(off for off, _n, name, _asm in ins if name == 's_getpc_b64')
    add = index_of[getpc + 4]
    addc = next(i for i, (off, _n, name, _asm) in enumerate(ins) if name == 's_addc_u32')
    setpc = next(i for i, (off, _n, name, _asm) in enumerate(ins) if name == 's_setpc_b64')
    entry = next(i for i, (off, _n, name, _asm) in enumerate(ins) if name == 's_cbranch_scc0')
    continuation = len(rows) - 2
    displacement = int(asm_of[add].split(', ')[-1], 0)

    def stage_and_extract(built):
        if library is None:
            return None
        def run(_code):
            return extract(directory, built, library)
        return run

    # A register-derived displacement: the ADD reads s8, so the target is not a constant.
    rows_reg = replace_row(rows, add, ('s_add_u32 s20, s20, s8', '14081480'))
    negative('register-derived displacement', words_of(rows_reg),
             's_setpc_b64: unsupported arithmetic on calculated PC words',
             stage_and_extract(words_of(rows_reg)))

    # A missing ADDC: an equally long no-op sits where the high-word addition belonged, so the SETPC
    # consumes the incomplete low-word pair and every other offset stays as it was.
    rows_missing = replace_row(rows, addc, NOP)
    negative('missing ADDC', words_of(rows_missing),
             's_setpc_b64: incomplete calculated target (missing high-word addition with carry)',
             stage_and_extract(words_of(rows_missing)))

    # A wrong high sign extension: the ADDC adds 1 while the displacement is positive.
    rows_wrong = replace_row(rows, addc, ('s_addc_u32 s21, s21, 1', sop2_inline(4, 21, 21, 1)))
    negative('wrong sign extension', words_of(rows_wrong),
             's_setpc_b64: calculated PC high word is not the signed displacement extension',
             stage_and_extract(words_of(rows_wrong)))

    # A signed ADD in the low word: SOP2 s_add_i32 instead of s_add_u32.
    rows_signed = replace_row(rows, add, add_i32(displacement))
    negative('signed low-word ADD', words_of(rows_signed),
             's_setpc_b64: calculated PC word escapes into s_add_i32',
             stage_and_extract(words_of(rows_signed)))

    # A competing entry: the entry branch targets the ADD, entering the chain without its GETPC base.
    branch_dwords = (ins[add][0] - ins[entry][0] - 4) // 4
    rows_competing = replace_row(rows, entry, cbranch_scc0(branch_dwords))
    negative('competing entry into the chain', words_of(rows_competing),
             's_setpc_b64: unknown or ambiguous input pair for calculated PC low-word addition',
             stage_and_extract(words_of(rows_competing)))

    # Pair clobbering: an s_mov_b32 of the high word between the ADDC and the SETPC.
    rows_clobber = rows[:addc + 1] + [('s_mov_b32 s21, 0', '800395be')] + rows[addc + 1:]
    negative('pair clobbering', words_of(rows_clobber),
             's_setpc_b64: unknown or ambiguous calculated target',
             stage_and_extract(words_of(rows_clobber)))

    # An address escape: the early block reads the low word as data between the ADDC and the SETPC.
    rows_escape = rows[:setpc] + [('v_mov_b32_e32 v9, s20', '1402127e')] + rows[setpc:]
    negative('address escape', words_of(rows_escape),
             's_setpc_b64: calculated PC word escapes into v_mov_b32_e32',
             stage_and_extract(words_of(rows_escape)))

    # An SCC escape: an SCC-reading branch sits between the ADDC and the SETPC.
    rows_scc = rows[:setpc] + [('s_cbranch_scc1 0', '000085bf')] + rows[setpc:]
    negative('SCC escape', words_of(rows_scc),
             's_setpc_b64: calculated PC SCC escapes into s_cbranch_scc1',
             stage_and_extract(words_of(rows_scc)))

    # A stale scratch pair: the program writes s21 before the chain, so the pair is not fresh.
    rows_stale = replace_row(rows, 0, ('s_mov_b32 s21, 5', '850395be'))
    negative('stale scratch pair', words_of(rows_stale),
             'far early-out: the scratch pair s[20:21] is not fresh',
             stage_and_extract(words_of(rows_stale)))

    # Effectful continuations: an export, a store, a barrier and an EXEC write each refuse.
    for row, label in ((('exp mrt0 v4, v4, v4, v4 done vm', '0f1800f804040404'), 'an export'),
                       (('buffer_store_dword v5, v6, s[16:19], 0 offen', '001070e006050480'), 'a store'),
                       (('s_barrier', '00008abf'), 'a barrier'),
                       (('s_mov_b32 exec_lo, 0', '8003febe'), 'an EXEC write')):
        rows_effect = replace_row(rows, continuation, row)
        negative('effectful continuation (%s)' % label, words_of(rows_effect),
                 'far early-out: the continuation instruction', stage_and_extract(words_of(rows_effect)))

    # A continuation ending at the first rather than the final s_endpgm: one s_nop is inserted
    # before the first exit and the ADD re-aimed four bytes earlier onto it, so the transfer is
    # proven and its continuation is pure, but it terminates before the program's final exit.
    first_end = next(i for i, row in enumerate(rows) if row == END)
    rows_first = rows[:first_end] + [NOP] + rows[first_end:]
    rows_first = replace_row(rows_first, add, add_u32(displacement - 4))
    negative('continuation ends before the final exit', words_of(rows_first),
             "far early-out: the continuation ends at",
             stage_and_extract(words_of(rows_first)))

    # A target off an instruction boundary: one byte past the continuation start is unaligned.
    rows_boundary = replace_row(rows, add, add_u32(displacement + 1))
    negative('target off an instruction boundary', words_of(rows_boundary),
             's_setpc_b64: unaligned calculated branch target',
             stage_and_extract(words_of(rows_boundary)))

    # A target outside the program: the displacement points past the final exit. The far rows hold
    # the same shape with an already-literal ADD, so the wider displacement moves no instruction.
    far_ins = decode_all(far_code)
    far_add = next(i for i, (off, n, name, asm) in enumerate(far_ins) if name == 's_add_u32')
    far_displacement = int(far_ins[far_add][3].split(', ')[-1], 0)
    rows_outside = replace_row(far_rows, far_add, add_u32(far_displacement + 4096))
    negative('target outside the program', words_of(rows_outside),
             's_setpc_b64: target',
             stage_and_extract(words_of(rows_outside)))


def main():
    args = sys.argv[1:]
    if args and args[0] == 'binary':
        if len(args) not in (2, 3):
            raise SystemExit(__doc__)
        return check_binary(args[1], args[2] if len(args) == 3 else None)
    library = LIBRARY
    if args and args[0] == 'controls':
        # The explicit selection drives the extraction half of every control below; it is not
        # captured globally, so check_binary and the no-argument auto-discovery keep their own.
        if len(args) != 2 or not Path(args[1]).is_file():
            raise SystemExit(__doc__)
        library = args[1]
    elif args:
        raise SystemExit(__doc__)

    short_rows, short_code, short_offsets = variant(BRANCH)
    calc_rows, calc_code, calc_offsets = variant(CHAIN)
    far_rows, far_code, far_offsets = variant(CHAIN, FAR_NOP_DWORDS)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        check_variant('short s_branch', short_rows, short_code, short_offsets, len(short_rows) - 2, directory,
                      library=library)
        check_variant('short calculated', calc_rows, calc_code, calc_offsets, len(calc_rows) - 2, directory,
                      library=library)
        # The far variant: the displacement measured from the actual assembled offsets must exceed
        # the positive SOPP s_branch range, and its NOP run must be reachable in the normal path.
        continuation = len(far_rows) - 2
        ins = decode_all(far_code)
        getpc = next(off for off, _n, name, _asm in ins if name == 's_getpc_b64')
        displacement = far_offsets[continuation] - (getpc + 4)
        nops = sum(1 for _off, _n, name, _asm in ins if name == 's_nop')
        assert nops >= FAR_NOP_DWORDS, (nops, FAR_NOP_DWORDS)
        check_variant('far calculated', far_rows, far_code, far_offsets, continuation, directory,
                      displacement=displacement, library=library)
        print('far displacement %#x exceeds the SOPP positive range %d' % (displacement, SOPP_LIMIT))
        check_negatives(directory, calc_rows, calc_code, far_rows, far_code, library=library)
    print('Far early-out: three variants, dispatcher oracle and refusal controls passed')


def assemble(llvm_mc, source):
    run = subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-show-encoding'], input=source,
                         capture_output=True, text=True, check=True)
    words = bytearray()
    for line in run.stdout.splitlines():
        m = re.search(r'encoding: \[([^\]]*)\]', line)
        if m:
            words += bytes(int(b, 16) for b in m.group(1).replace(' ', '').split(','))
    return bytes(words)


def check_binary(llvm_mc, library=None):
    for name, (rows, code, _offsets) in (('short s_branch', variant(BRANCH)),
                                         ('short calculated', variant(CHAIN)),
                                         ('far calculated', variant(CHAIN, FAR_NOP_DWORDS))):
        built = assemble(llvm_mc, source_of(rows))
        assert built == code, (name, built[:80].hex(), code[:80].hex(), len(built), len(code))
        if library is not None:
            with tempfile.TemporaryDirectory() as tmp:
                assert extract(Path(tmp), code, library) == code, name
    if library is not None:
        print('fixture words match llvm-mc, and the extraction of each variant through %s is exact' % library)
    else:
        print('fixture words match llvm-mc')
    return 0


if __name__ == '__main__':
    main()
