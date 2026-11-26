#!/usr/bin/env python3
"""DPP16 quad swaps in a compute program of tools/startup/gfx2hlsl.py (sibling of startup_gfx2hlsl_dpp_quad_swap_check.py).

  (no arguments)                         translation controls and refusals, pure Python
  dispatch <dxc> <offscreen_check> <warp|hardware>
                                         translate, compile with DXC cs_6_0, dispatch and compare all 320 words
  control <wrong-neighbour|no-exchange> <dxc> <offscreen_check> <warp|hardware>
                                         mutated translation must mismatch on readback at the named word

In a DPP16 `quad_perm:[a,b,c,d]` source, lane i of each aligned group of four reads lane perm[i] of that group (AMD RDNA 2
ISA reference, "Data-Parallel Primitives"). The swaps [1,0,3,2], [2,3,0,1] and [3,2,1,0] flip lane bit 0, bit 1 and both.
A wave holds 32 or 64 consecutive threads in x-major order, so lane bits 1:0 are the bits 1:0 of the flat thread index.
A compute thread has no 2x2 quad of its own in SM 6, so the exchange goes through a groupshared array behind a group barrier
(`x2w_quad[flat ^ k]`), and only when every thread reaches it with entry EXEC (straight-line code) in a group of whole quads.
The reference below is written from the lane rule with integers and a float32 sum, not from the translated text.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))

import gfx2hlsl as G  # noqa: E402
from startup_compute_tg_size_check import (  # noqa: E402
    BUFFER_STORE_X4_V8_V4_S16, EXEC_LO, INLINE_0, INLINE_4, INLINE_NEG1, S_ENDPGM, S_LOAD_X8_S12_S10, S_WAITCNT_LGKM0,
    S_MOV_B32, S_CMP_EQ_U32, V_MAD_U32_U24, V_LSHLREV_B32, V_MOV_B32, sop1, sopc, sopp, vop1, vop2, vop3)
from startup_gfx2hlsl_compute_check import SENTINEL, TGID_X, USER7, WORDS, refused, root_blob, tidig  # noqa: E402
from startup_gfx2hlsl_dpp_quad_swap_check import vop1_dpp, vop2_dpp  # noqa: E402

GROUP = (4, 4, 2)  # 32 threads: one wave32 or half a wave64
RSRC2 = USER7 | TGID_X | tidig(3)
V_CVT_F32_U32 = 0x06
V_ADD_F32 = 0x03
INLINE_7, INLINE_1 = 128 + 7, 128 + 1
SWAP_X, SWAP_Y, SWAP_DIAGONAL = (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)
ROW_ROR_8 = 0x128  # dpp_ctrl of row_ror:8 (RDNA 2 ISA, DPP16: 0x121..0x12f rotate the 16-lane row right by 1..15)

# flat = tid.x + 4 * (tid.y + 4 * tid.z); v5 = flat * 7 + 1; v6 = float(flat)
# v8 = swap X of v5 (v_mov_b32_dpp), v9 = swap Y of v5, v10 = swap diagonal of v6 + v6 (v_add_f32_dpp), v11 = flat
PROGRAM = b''.join([
    sop1(S_MOV_B32, 10, 6),
    sop1(S_MOV_B32, 11, INLINE_0),
    S_LOAD_X8_S12_S10,
    vop3(V_MAD_U32_U24, 3, 256 + 2, INLINE_4, 256 + 1),
    vop3(V_MAD_U32_U24, 3, 256 + 3, INLINE_4, 256 + 0),
    vop3(V_MAD_U32_U24, 5, 256 + 3, INLINE_7, INLINE_1),
    vop1(V_CVT_F32_U32, 6, 256 + 3),
    vop1(V_MOV_B32, 11, 256 + 3),
    vop2(V_LSHLREV_B32, 4, INLINE_4, 3),
    vop1_dpp(0x01, 8, 5, SWAP_X),
    vop1_dpp(0x01, 9, 5, SWAP_Y),
    vop2_dpp(V_ADD_F32, 10, 6, 6, SWAP_DIAGONAL),
    S_WAITCNT_LGKM0,
])
PROGRAM_TAIL_STORE = BUFFER_STORE_X4_V8_V4_S16 + S_ENDPGM
PROGRAM += PROGRAM_TAIL_STORE


def dpp_ror8_mov(vdst, src):
    return struct.pack('<2I', 0x7e000000 | vdst << 17 | 0x01 << 9 | 250, src | ROW_ROR_8 << 8 | 0xf << 24 | 0xf << 28)


def dpp_ror8_add(vdst, src0, vsrc1):
    return struct.pack('<2I', V_ADD_F32 << 25 | vdst << 17 | vsrc1 << 9 | 250, src0 | ROW_ROR_8 << 8 | 0xf << 24 | 0xf << 28)


# row_ror:8 on the same lanes: v8 = ror8 of v5 (v_mov_b32_dpp), v9 = ror8 of v6 + v6 (v_add_f32_dpp), v10 = ror8 of v5 + the
# swap X of v5 is not needed: v10 = ror8 of v11 (the flat index itself), v11 = flat
PROGRAM_ROR = PROGRAM[:PROGRAM.index(vop1_dpp(0x01, 8, 5, SWAP_X))] + b''.join([
    dpp_ror8_mov(8, 5),
    dpp_ror8_add(9, 6, 6),
    dpp_ror8_mov(10, 11),
    S_WAITCNT_LGKM0,
]) + PROGRAM_TAIL_STORE


def expected_words():
    out = [SENTINEL] * WORDS
    for lane in range(32):
        f = struct.unpack('<I', struct.pack('<f', float((lane ^ 3) + lane)))[0]
        out[4 * lane:4 * lane + 4] = [(lane ^ 1) * 7 + 1, (lane ^ 2) * 7 + 1, f, lane]
    return out


def expected_words_ror():
    out = [SENTINEL] * WORDS
    for lane in range(32):
        f = struct.unpack('<I', struct.pack('<f', float((lane ^ 8) + lane)))[0]
        out[4 * lane:4 * lane + 4] = [(lane ^ 8) * 7 + 1, f, lane ^ 8, lane]
    return out


def translate(code=None, group=GROUP, rsrc2=RSRC2):
    return G.ComputePipeline(PROGRAM if code is None else code, root_blob(), rsrc2, group)


def check_controls():
    pipe = translate()
    hlsl = pipe.hlsl_cs()
    flat = '(tid.x + 4u * (tid.y + 4u * tid.z))'
    assert 'groupshared uint x2w_quad[32];' in hlsl, hlsl
    for k in (1, 2, 3):
        assert 'x2w_quad[%s ^ %du]' % (flat, k) in hlsl, (k, hlsl)
    assert hlsl.count('GroupMemoryBarrierWithGroupSync();') == 6, hlsl  # one before and one after each of the 3 exchanges
    assert 'QuadReadAcross' not in hlsl, hlsl
    asm = [a for _, _, _, a in G.decode(PROGRAM)]
    assert 'v_mov_b32_dpp v8, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf' in asm, asm
    assert any(a.startswith('v_add_f32_dpp v10, v6, v6 quad_perm:[3,2,1,0]') for a in asm), asm

    dpp = vop1_dpp(0x01, 8, 5, SWAP_X)
    # whole quads only
    for bad in ((6, 1, 1), (3, 3, 1), (2, 1, 1)):
        refused('requires a group of whole quads', lambda g=bad: translate(code=dpp + S_ENDPGM, group=g).hlsl_cs())
    # straight-line code with entry EXEC only
    branchy = b''.join([sopc(S_CMP_EQ_U32, 7, INLINE_0), sopp(0x04, 1), vop1(V_MOV_B32, 3, 256 + 0), dpp, S_ENDPGM])
    refused('requires a straight-line compute program with entry EXEC', lambda: translate(code=branchy).hlsl_cs())
    exec_write = b''.join([sop1(S_MOV_B32, EXEC_LO, INLINE_NEG1), dpp, S_ENDPGM])
    refused('requires a straight-line compute program with entry EXEC', lambda: translate(code=exec_write).hlsl_cs())
    # a permutation other than the three swaps keeps its refusal
    refused('only the swaps', lambda: translate(code=vop1_dpp(0x01, 8, 5, (1, 2, 3, 0)) + S_ENDPGM).hlsl_cs())
    check_ror_controls()
    print('dpp quad swap in compute: translation controls and refusals pass')


def check_ror_controls():
    flat = '(tid.x + 4u * (tid.y + 4u * tid.z))'
    hlsl = translate(code=PROGRAM_ROR).hlsl_cs()
    assert hlsl.count('x2w_quad[%s ^ 8u]' % flat) == 3, hlsl
    asm = [a for _, _, _, a in G.decode(PROGRAM_ROR)]
    assert 'v_mov_b32_dpp v8, v5 row_ror:8 row_mask:0xf bank_mask:0xf' in asm, asm
    ror = dpp_ror8_mov(8, 5) + S_ENDPGM
    refused('requires a group of whole rows', lambda: translate(code=ror, group=(4, 1, 1)).hlsl_cs())
    refused('requires a group of whole rows', lambda: translate(code=ror, group=(12, 1, 1)).hlsl_cs())
    branchy = b''.join([sopc(S_CMP_EQ_U32, 7, INLINE_0), sopp(0x04, 1), vop1(V_MOV_B32, 3, 256 + 0), ror])
    refused('requires a straight-line compute program with entry EXEC', lambda: translate(code=branchy).hlsl_cs())
    # the other rotations keep their refusal: only row_ror:8 has a direction-free partner
    other = struct.pack('<2I', 0x7e000000 | 8 << 17 | 0x01 << 9 | 250, 5 | 0x123 << 8 | 0xf << 24 | 0xf << 28)
    refused('row_ror:3', lambda: translate(code=other + S_ENDPGM).hlsl_cs())


CONTROLS = {
    # the X exchange reads the Y neighbour: lane 0 gets lane 2's word
    'wrong-neighbour': ('x2w_quad[%s ^ 1u]', 'x2w_quad[%s ^ 2u]'),
    # the diagonal exchange reads the lane's own word: v10 becomes 2 * lane
    'no-exchange': ('x2w_quad[%s ^ 3u]', 'x2w_quad[%s ^ 0u]'),
    # the row rotation reads the next lane instead of lane ^ 8 (first of its three exchanges)
    'ror8-wrong-lane': ('x2w_quad[%s ^ 8u]', 'x2w_quad[%s ^ 9u]'),
}
CONTROL_MISMATCH = {
    'wrong-neighbour': 'word=0 expected=00000008 actual=0000000f',
    'no-exchange': 'word=2 expected=40400000 actual=00000000',
    'ror8-wrong-lane': 'word=0 expected=00000039 actual=00000040',
}


def mutate(hlsl, control):
    flat = '(tid.x + 4u * (tid.y + 4u * tid.z))'
    old, new = (t % flat for t in CONTROLS[control])
    assert hlsl.count(old) >= 1, (control, hlsl)
    return hlsl.replace(old, new, 1)


def check_dispatch(dxc, runner, adapter, control=None, ror=False):
    runner = str(Path(runner).resolve())
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    hlsl = translate(code=PROGRAM_ROR if ror else None).hlsl_cs()
    if control:
        if control not in CONTROLS:
            raise SystemExit('unknown control %r' % control)
        hlsl = mutate(hlsl, control)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        d = Path(tmp)
        src, cso, case = d / 'quad.hlsl', d / 'quad.cso', d / 'quad.case'
        src.write_text(hlsl, encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
        case.write_bytes(struct.pack('<I', 1 | (1 << 8) | (1 << 16)) + struct.pack('<%dI' % WORDS, *(expected_words_ror() if ror else expected_words()))
                         + struct.pack('<16I', *([0] * 16)))
        run = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)
        print(run.stdout, end='')
        if run.returncode == 77:
            return 77
        diagnostics = [line for line in run.stderr.splitlines()
                       if not (adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '))]
        if control:
            if run.returncode != 1 or diagnostics[:1] != [CONTROL_MISMATCH[control]]:
                raise AssertionError('control %s: unexpected result (exit %d) %s' % (control, run.returncode, run.stderr))
            print('control %s: rejected at %s' % (control, diagnostics[0]))
        elif run.returncode or diagnostics:
            raise AssertionError('quad swap dispatch: %s' % (run.stderr or 'exit %d' % run.returncode))
        else:
            print('quad swap dispatch: all %d words match (1 group x %dx%dx%d threads)' % (WORDS, *GROUP))
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(check_dispatch(*sys.argv[2:]))
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch-ror8':
        sys.exit(check_dispatch(*sys.argv[2:], ror=True))
    elif len(sys.argv) == 6 and sys.argv[1] == 'control':
        sys.exit(check_dispatch(*sys.argv[3:], control=sys.argv[2], ror=sys.argv[2].startswith('ror8')))
    else:
        sys.exit(__doc__)
