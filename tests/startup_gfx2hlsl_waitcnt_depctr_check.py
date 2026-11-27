"""s_waitcnt_depctr is a wait without architectural effect in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_waitcnt_depctr_check.py
       startup_gfx2hlsl_waitcnt_depctr_check.py binary <llvm-mc>

LLVM defines S_WAITCNT_DEPCTR (SOPP 0x23, gfx10 and later) with one immediate operand and no register
definition, use or side effect (llvm/lib/Target/AMDGPU/SOPInstructions.td) and emits it only as a hazard delay
(GCNHazardRecognizer.cpp, fixVMEMtoScalarWriteHazards, immediate 0xffe3). The translator therefore lowers it
like s_nop: a program with the wait must translate to the same text and state as the program without it.

Controls: s_wait_idle, s_sleep and a binary32 s_round_mode stay refused, and replacing the wait by a scalar move changes
the compared text, so the comparison can fail.
"""
from pathlib import Path
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_scalar_m0_check import ProbeStage, encode_sop1, encode_sop2  # noqa: E402


def sopp(op, imm=0):
    return struct.pack('<I', 0xbf800000 | op << 16 | imm)


MOV = encode_sop1(0x03, 2, 8)              # s_mov_b32 s2, s8
ADD = encode_sop2(0x00, 3, 2, 128 + 5)     # s_add_u32 s3, s2, 5
WAITS = (0xffe3, 0xfffe, 0x0fff, 0x0000, 0xffff)


def run(*chunks):
    stage = ProbeStage()
    results = []
    for code in chunks:
        _, name, asm, _ = D.decode_instruction(code, 0)
        results.append(stage.one(name, asm))
    return stage, results


def state(stage):
    return stage.lines, stage.sym, stage.used_s, stage.used_v, stage.exports, stage.masks


def controls():
    plain, _ = run(MOV, ADD)
    assert plain.lines, 'the reference program must emit text'
    for imm in WAITS:
        _, name, asm, _ = D.decode_instruction(sopp(0x23, imm), 0)
        assert (name, asm) == ('s_waitcnt_depctr', 's_waitcnt_depctr 0x%x' % imm), (name, asm)
        for order in ((sopp(0x23, imm), MOV, ADD), (MOV, sopp(0x23, imm), ADD), (MOV, ADD, sopp(0x23, imm))):
            waited, results = run(*order)
            assert results[order.index(sopp(0x23, imm))] is None
            assert state(waited) == state(plain), 's_waitcnt_depctr 0x%x changed the translation' % imm
    # The comparison notices an instruction with an effect in the same position.
    moved, _ = run(MOV, encode_sop1(0x03, 2, 5), ADD)
    assert state(moved) != state(plain)
    # Neighbouring SOPP forms keep their refusal (s_round_mode only where it leaves binary32 nearest even).
    for op, imm, name in ((0x22, 0, 's_wait_idle'), (0x0e, 0, 's_sleep'), (0x24, 0x3, 's_round_mode')):
        try:
            run(MOV, sopp(op, imm), ADD)
        except G.Unsupported as reason:
            assert name in str(reason), reason
        else:
            raise AssertionError('%s must stay refused' % name)
    print('s_waitcnt_depctr: %d immediates leave the translation unchanged; 3 neighbouring forms refused' % len(WAITS))
    return 0


def check_binary(llvm_mc):
    source = ''.join('s_waitcnt_depctr 0x%x\n' % imm for imm in WAITS)
    proc = subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-show-encoding'],
                          input=source, capture_output=True, text=True, check=True)
    built = b''.join(bytes(int(b, 16) for b in m.split(','))
                     for m in re.findall(r'encoding: \[([^\]]*)\]', proc.stdout))
    want = b''.join(sopp(0x23, imm) for imm in WAITS)
    if built != want:
        raise AssertionError('llvm-mc encoding mismatch: got %s, expected %s' % (built.hex(), want.hex()))
    print('s_waitcnt_depctr encodings match llvm-mc')
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'binary':
        sys.exit(check_binary(sys.argv[2]))
    if len(sys.argv) != 1:
        raise SystemExit(__doc__)
    sys.exit(controls())
