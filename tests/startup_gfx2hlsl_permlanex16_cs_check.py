#!/usr/bin/env python3
"""Cross-row lane permutes: Python controls or dispatch <dxc> <runner> <adapter>."""
import hashlib
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_gfx2hlsl_permlane16_cs_check as P
import startup_gfx2hlsl_wave_reduce_check as W

G, BASE = P.G, P.BASE
GROUP = (4, 4, 4)
LOW, HIGH = P.SEL_LOW, P.SEL_HIGH

def cross(dst, src, low=P.S_LOW, high=P.S_HIGH, flags=0):
    w0 = 0xd4000000 | 0x378 << 16 | flags << 11 | dst
    return struct.pack('<2I', w0, (256 + src) | low << 9 | high << 18)

PROGRAM = b''.join([
    P.HEAD,
    P.sopc(P.S_CMP_EQ_U32, 7, BASE.INLINE_0), P.sopp(0x04, 1),
    P.vop1(P.V_MOV_B32, 6, 256),
    P.sop1(P.S_MOV_B32, P.S_LOW, P.LITERAL, LOW),
    P.sop1(P.S_MOV_B32, P.S_HIGH, P.LITERAL, HIGH),
    cross(8, 5, flags=2),
    P.permlane(9, 8, P.S_HIGH, P.S_LOW),
    cross(10, 9),
    BASE.S_WAITCNT_LGKM0, BASE.PROGRAM_TAIL_STORE,
])

def translate(code=PROGRAM, group=GROUP):
    return P.translate(code=code, group=group).hlsl_cs()

def oracle(same_row=False, swapped=False):
    # RDNA 2 ISA guide 12.12: select a nibble in each row, then exchange row pairs.
    original = [i * 7 + 1 for i in range(64)]
    def perm(values, low, high, cross_row):
        result = []
        for lane in range(64):
            pos = lane % 16
            nibble = ((low if pos < 8 else high) >> (4 * (pos % 8))) & 15
            row = lane // 16
            if cross_row:
                row ^= 1
            result.append(values[row * 16 + nibble])
        return result
    a = perm(original, HIGH if swapped else LOW, LOW if swapped else HIGH, not same_row)
    b = perm(a, HIGH, LOW, False)
    c = perm(b, LOW, HIGH, True)
    result = [P.SENTINEL] * P.WORDS
    for lane in range(64):
        result[4*lane:4*lane+4] = [a[lane], b[lane], c[lane], lane]
    return result

def check_controls():
    # Golden outputs captured from the parent revision before the new handler.
    assert hashlib.sha256(P.translate().hlsl_cs().encode()).hexdigest() == '39bb86f9db4cd3ef4b457a00a0b42802f4c1ab5fd6f90b006d09ce82ed650ea2'
    wave, reason = W.translate(W.wave64_sum())
    assert reason is None
    assert hashlib.sha256(wave.encode()).hexdigest() == '7144b18bb68b8f79011396beb615fef0243b757fe01eb25c949daa16a28efee4'
    text = translate()
    flat = '(tid.x + 4u * (tid.y + 4u * tid.z))'
    assert text.count('(%s ^ 16u) & ~15u' % flat) == 2
    assert text.count('GroupMemoryBarrierWithGroupSync();') == 6
    assert oracle() != oracle(same_row=True)
    assert oracle() != oracle(swapped=True)
    one = cross(8, 5) + P.S_ENDPGM
    P.refused('v_permlanex16_b32 in a compute program requires a group of whole 32-lane row pairs',
              lambda: translate(one, (16, 1, 1)))
    branch = b''.join([P.sopc(P.S_CMP_EQ_U32, 7, BASE.INLINE_0), P.sopp(0x04, 1),
                       P.vop1(P.V_MOV_B32, 3, 256), one])
    flow = translate(branch)
    assert 'GroupMemoryBarrierWithGroupSync();' in flow
    skipped = b''.join([P.sopc(P.S_CMP_EQ_U32, 7, BASE.INLINE_0), P.sopp(0x04, 2), one])
    P.refused('requires a straight-line compute program with entry EXEC', lambda: translate(skipped))
    bad = bytearray(one); bad[1] |= 0x20
    P.refused('unsupported', lambda: translate(bytes(bad)))
    P.refused('unsupported', lambda: translate(cross(8, 5, 256, P.S_HIGH) + P.S_ENDPGM))
    _, reason = W.translate(one)
    assert reason and 'v_permlanex16_b32 outside a compute program' in reason, reason
    print('v_permlanex16 compute controls pass; both mutant oracles differ')

def dispatch(dxc, runner, adapter):
    P.ROOT.joinpath('out').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=P.ROOT / 'out') as tmp:
        path = Path(tmp)
        src, cso, case = path/'shader.hlsl', path/'shader.cso', path/'case.bin'
        src.write_text(translate(), encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
        case.write_bytes(struct.pack('<I', 1 | 1 << 8 | 1 << 16) +
                         struct.pack('<%dI' % P.WORDS, *oracle()) + struct.pack('<16I', *([0]*16)))
        run = subprocess.run([str(Path(runner).resolve()), adapter, '--compute-store-case', str(cso), str(case)],
                             capture_output=True, text=True)
        print(run.stdout, end='')
        if run.returncode == 77:
            return 77
        diagnostics = [line for line in run.stderr.splitlines()
                       if not (adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '))]
        if run.returncode or diagnostics:
            raise AssertionError(run.stderr or 'exit %d' % run.returncode)
        print('permlanex16 dispatch: all %d words match' % P.WORDS)
    return 0

if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(dispatch(*sys.argv[2:]))
    else:
        sys.exit(__doc__)
