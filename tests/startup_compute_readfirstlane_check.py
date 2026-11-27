"""Source-built gfx1013 compute fixture for one-path SGPRs and uniform loop counters with v_readfirstlane_b32 (#2274).

  (no arguments)                              decode, translate and check controls + negative controls (pure Python)
  --llvm-mc <llvm-mc> / binary <llvm-mc>      also reassemble the fixture and patched instructions with llvm-mc
  --dispatch <dxc> <offscreen_check> <adapter>  compile translated HLSL with DXC, run GPU readback and negative controls

The shader builds a one-path SGPR ``s20`` across a uniform forward branch whose taken path does not write ``s20``,
reads it through ``v_readfirstlane_b32`` at the join, runs a four-iteration uniform loop whose counter ``s22`` is
incremented uniformly and read back via ``v_readfirstlane_b32`` both inside the loop and after loop exit, and stores
one combined word per thread across two four-thread groups ``(4, 1, 1) x (2, 1, 1)`` so both arms of the branch
run on the GPU. The waterfall fixtures also exercise a lane-varying key. They restore the incoming threads
after selecting zero to four lanes, then read back per-key arithmetic and a load from one fixed raw buffer.
"""
import argparse
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch as mock_patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))

import disasm_rdna2 as D  # noqa: E402
import startup_compute_waterfall as waterfall
from startup_gfx2hlsl_compute_check import (  # noqa: E402
    CASES, G, KEY, MASK, SENTINEL, STRIDE, WORDS, patch, refused, root_blob,
)

RSRC2 = CASES['D'][2]  # USER7 | TGID_X | tidig(0)
LOCAL = (4, 1, 1)
GROUPS = (2, 1, 1)
ONE_PATH_LINE = 's25 = uint(max(F((U(2.0 * F((U(float(s20))))))), 0.0));'
LOOP_LINE = 's24 = uint(max(F((U(2.0 * F((U(float(s22))))))), 0.0));'
POST_LOOP_LINE = 's26 = s22;'
CONST_SHIFT_LINE = 's29 = (0x0000003cu) >> (0x00000002u & 31);'
INT_LINE = 's28 = (s27 + (s27)) << (0x00000001u & 31);'


def opcode(table, name):
    return next(op for op, info in table.items() if info[0] == name)


def ssrc(token):
    if isinstance(token, str):
        if token.startswith('v') and token != 'vcc_lo':
            return 256 + int(token[1:])
        return {'2.0': 244, '0': 128, 'vcc_lo': 106}.get(token) or int(token[1:])
    if 0 <= token <= 64:
        return 128 + token
    raise ValueError(token)


def sop1(name, dst, src):
    return [0xbe800000 | ssrc(dst) << 16 | opcode(D.SOP1_OPS, name) << 8 | ssrc(src)]


def sop2(name, dst, a, b):
    return [0x80000000 | opcode(D.SOP2_OPS, name) << 23 | ssrc(dst) << 16 | ssrc(b) << 8 | ssrc(a)]


def sopc(name, a, b):
    return [0xbf000000 | opcode(D.SOPC_OPS, name) << 16 | ssrc(b) << 8 | ssrc(a)]


def sopp(name, imm=0):
    return [0xbf800000 | opcode(D.SOPP_OPS, name) << 16 | (imm & 0xffff)]


def vop1(name, dst, src):
    dv = int(dst[1:]) if dst.startswith('v') else ssrc(dst)
    return [0x7e000000 | dv << 17 | opcode(D.VOP1_OPS, name) << 9 | ssrc(src)]


def vop2(name, dst, a, b):
    return [opcode(D.VOP2_OPS, name) << 25 | int(dst[1:]) << 17 | int(b[1:]) << 9 | ssrc(a)]


def build_words():
    # s_mov_b32 s10, s6; s_mov_b32 s11, 0; s_load_dwordx8 s[12:19], s[10:11], 0x0; s_waitcnt lgkmcnt(0)
    ws = list(struct.unpack('<5I', bytes.fromhex('06038abe80038bbe05030cf4000000fa7fc08cbf')))
    ws += sop2('s_and_b32', 's20', 's5', 63) + sop2('s_add_u32', 's20', 's20', 's7')
    ws += sop1('s_mov_b32', 's21', 5) + sopc('s_cmp_eq_u32', 's7', 0)
    ws += sopp('s_cbranch_scc0', 1) + sop1('s_mov_b32', 's21', 9)
    ws += (vop1('v_cvt_f32_u32_e32', 'v7', 's20') + vop2('v_mul_f32_e32', 'v7', '2.0', 'v7')
           + vop1('v_cvt_u32_f32_e32', 'v7', 'v7') + vop1('v_readfirstlane_b32', 's25', 'v7'))
    ws += sop1('s_mov_b32', 's22', 0) + sop1('s_mov_b32', 's23', 0)
    loop_idx = len(ws)
    ws += (vop1('v_cvt_f32_u32_e32', 'v8', 's22') + vop2('v_mul_f32_e32', 'v8', '2.0', 'v8')
           + vop1('v_cvt_u32_f32_e32', 'v8', 'v8') + vop1('v_readfirstlane_b32', 's24', 'v8'))
    ws += sop2('s_add_u32', 's23', 's23', 's24') + sop2('s_add_u32', 's22', 's22', 1)
    ws += sopc('s_cmp_lt_u32', 's22', 4)
    ws += sopp('s_cbranch_scc1', loop_idx - (len(ws) + 1))
    ws += vop1('v_mov_b32_e32', 'v9', 's22') + vop1('v_readfirstlane_b32', 's26', 'v9')
    ws += sop2('s_lshl_b32', 's27', 's7', 2) + vop2('v_add_nc_u32_e32', 'v4', 's27', 'v0')
    # Pure integer ops on a uniform scalar, read past the entry block: (s27 + s27) << 1 (RF1).
    ws += (vop1('v_mov_b32_e32', 'v10', 's27') + vop2('v_add_nc_u32_e32', 'v10', 's27', 'v10')
           + vop2('v_lshlrev_b32_e32', 'v10', 1, 'v10') + vop1('v_readfirstlane_b32', 's28', 'v10'))
    # A VGPR overwritten by a constant, then shifted: 60 >> 2 is the same in every lane (RF1).
    ws += (vop1('v_mov_b32_e32', 'v11', 60) + vop2('v_lshrrev_b32_e32', 'v11', 2, 'v11')
           + vop1('v_readfirstlane_b32', 's29', 'v11'))
    ws += sop2('s_add_u32', 's25', 's25', 's21') + sop2('s_add_u32', 's25', 's25', 's23')
    ws += sop2('s_add_u32', 's25', 's25', 's26') + sop2('s_add_u32', 's25', 's25', 's28') + sop2('s_add_u32', 's25', 's25', 's29')
    ws += vop2('v_add_nc_u32_e32', 'v5', 's25', 'v4') + vop2('v_lshlrev_b32_e32', 'v4', 2, 'v4')
    # buffer_store_dword v5, v4, s[16:19], 0 offen; s_endpgm
    ws += list(struct.unpack('<2I', bytes.fromhex('001070e004050480'))) + sopp('s_endpgm')
    return struct.pack('<%dI' % len(ws), *ws)


PROGRAM_WORDS = build_words()
DECODED = G.decode(PROGRAM_WORDS)


def pack_hex(*ws):
    return struct.pack('<%dI' % len(ws), *ws).hex()


PATCHES = (
    ('non-uniform loop back-edge',
     [(pack_hex(*sopp('s_cbranch_scc1', -8)), pack_hex(*sopp('s_cbranch_vccnz', -8)))],
     'v_readfirstlane_b32 requires a proven uniform source'),
    ('divergent then arm writes one-path scalar',
     [(pack_hex(*sopp('s_cbranch_scc0', 1)), pack_hex(*sopp('s_cbranch_vccz', 1))),
      (pack_hex(*sop1('s_mov_b32', 's21', 9)), pack_hex(*sop1('s_mov_b32', 's20', 9)))],
     'v_readfirstlane_b32 requires a proven uniform source'),
    ('loop counter updated from lane mask',
     [(pack_hex(*sop2('s_add_u32', 's22', 's22', 1)), pack_hex(*sop1('s_mov_b32', 's22', 'vcc_lo')))],
     'v_readfirstlane_b32 requires a proven uniform source'),
    # RF1 negative controls: the integer chain stays refused when a source differs between lanes.
    ('integer chain from the thread id',
     [(pack_hex(*vop1('v_mov_b32_e32', 'v10', 's27')), pack_hex(*vop1('v_mov_b32_e32', 'v10', 'v0')))],
     'v_readfirstlane_b32 requires a proven uniform source'),
    # the lane-mask read as a number is refused first (#2702); the chain is not reached
    ('integer chain with a lane-mask scalar',
     [(pack_hex(*vop2('v_add_nc_u32_e32', 'v10', 's27', 'v10')), pack_hex(*vop2('v_add_nc_u32_e32', 'v10', 'vcc_lo', 'v10')))],
     'v_add_nc_u32 reads lane mask vcc_lo as numeric data'),
)


def translate(code=None, *, group=LOCAL):
    return G.ComputePipeline(code or PROGRAM_WORDS, root_blob(), RSRC2, group)


def expected():
    out = [SENTINEL] * WORDS
    base_stride = STRIDE & 63
    for gx in range(GROUPS[0]):
        s20 = base_stride + gx
        s25 = s20 * 2
        s21 = 9 if gx == 0 else 5
        s23 = 0 + 2 + 4 + 6  # sum over uniform loop iterations i=0..3
        s26 = 4
        s28 = ((gx * 4) * 2) << 1
        combined = (s25 + s21 + s23 + s26 + s28 + (60 >> 2)) & MASK
        for tx in range(LOCAL[0]):
            idx = gx * LOCAL[0] + tx
            out[idx] = (combined + idx) & MASK
    return out


def llvm_decoded_lines(decoded):
    """Format decoded instructions for llvm-mc, resolving label branches to dword offsets."""
    lines = []
    for off, _n, _name, asm in decoded:
        m = re.fullmatch(r'(s_branch|s_cbranch_\w+) label_0x([0-9a-f]+)', asm)
        if m:
            lines.append('%s %d' % (m.group(1), (int(m.group(2), 16) - off - 4) // 4))
        else:
            lines.append(asm.replace('vcc_lo', 'vcc'))
    return lines


def llvm_lines():
    return llvm_decoded_lines(DECODED) + [
        's_cbranch_vccnz -8',
        's_cbranch_vccz 1',
        's_mov_b32 s20, 9',
        's_mov_b32 s22, vcc_lo',
    ]


def check_binary(llvm_mc):
    lines = llvm_lines()
    proc = subprocess.run(
        [llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=+wavefrontsize64,-wavefrontsize32', '-show-encoding'],
        input='\n'.join(lines) + '\n', capture_output=True, text=True, check=True)
    built = [bytes(int(b, 16) for b in m.group(1).split(','))
             for m in re.finditer(r'encoding: \[([^\]]*)\]', proc.stdout)]
    assert len(built) == len(lines), (len(built), len(lines), proc.stderr)
    assembled = b''.join(built[:len(DECODED)])
    assert assembled == PROGRAM_WORDS, (assembled.hex(), PROGRAM_WORDS.hex())
    extra = [
        (built[-4], bytes.fromhex(pack_hex(*sopp('s_cbranch_vccnz', -8)))),
        (built[-3], bytes.fromhex(pack_hex(*sopp('s_cbranch_vccz', 1)))),
        (built[-2], bytes.fromhex(pack_hex(*sop1('s_mov_b32', 's20', 9)))),
        (built[-1], bytes.fromhex(pack_hex(*sop1('s_mov_b32', 's22', 'vcc_lo')))),
    ]
    for got, want in extra:
        assert got == want, (got.hex(), want.hex())
    print('%d fixture and negative-control instructions match %s' % (len(lines), llvm_mc))


def controls(llvm_mc=None):
    pipe = translate()
    hlsl = pipe.hlsl_cs()
    for snippet in (ONE_PATH_LINE, LOOP_LINE, POST_LOOP_LINE, INT_LINE, CONST_SHIFT_LINE, 'u0.Store(a, v5);'):
        assert hlsl.count(snippet) == 1, (snippet, hlsl)
    assert 'uav 0 table 1 16' in pipe.manifest().splitlines(), pipe.manifest()

    for label, replacements, reason in PATCHES:
        code = PROGRAM_WORDS
        for old_hex, new_hex in replacements:
            code = patch(code, old_hex, new_hex)
        refused(reason, lambda code=code: translate(code))

    if llvm_mc:
        check_binary(llvm_mc)
    log_ceil_controls(llvm_mc)
    img_load_controls(llvm_mc)
    waterfall.controls(llvm_mc)
    print('Compute readfirstlane one-path, loop-counter, log2/ceil, image_load, waterfall and refusal checks passed')


LOG_CEIL_LOG_LINE = 's21 = v8;'
LOG_CEIL_CEIL_LINE = 's22 = v9;'


def build_log_ceil_words():
    ws = list(struct.unpack('<5I', bytes.fromhex('06038abe80038bbe05030cf4000000fa7fc08cbf')))
    ws += sop2('s_lshl_b32', 's20', 's7', 2) + sop2('s_add_u32', 's20', 's20', 5)
    ws += vop1('v_cvt_f32_u32_e32', 'v7', 's20') + vop1('v_log_f32_e32', 'v7', 'v7')
    ws += vop2('v_mul_f32_e32', 'v8', '2.0', 'v7') + vop1('v_cvt_u32_f32_e32', 'v8', 'v8')
    ws += vop1('v_readfirstlane_b32', 's21', 'v8')
    ws += vop1('v_ceil_f32_e32', 'v9', 'v7') + vop1('v_cvt_u32_f32_e32', 'v9', 'v9')
    ws += vop1('v_readfirstlane_b32', 's22', 'v9')
    ws += sop2('s_lshl_b32', 's21', 's21', 8) + sop2('s_or_b32', 's21', 's21', 's22')
    ws += sop2('s_lshl_b32', 's23', 's7', 2) + vop2('v_add_nc_u32_e32', 'v4', 's23', 'v0')
    ws += vop2('v_add_nc_u32_e32', 'v5', 's21', 'v4') + vop2('v_lshlrev_b32_e32', 'v4', 2, 'v4')
    ws += list(struct.unpack('<2I', bytes.fromhex('001070e004050480'))) + sopp('s_endpgm')
    return struct.pack('<%dI' % len(ws), *ws)


LOG_CEIL_WORDS = build_log_ceil_words()


def log_ceil_expected():
    out = [SENTINEL] * WORDS
    for gx, base_val in ((0, 0x403), (1, 0x604)):
        for tx in range(LOCAL[0]):
            idx = gx * LOCAL[0] + tx
            out[idx] = (base_val + idx) & MASK
    return out


def log_ceil_controls(llvm_mc=None):
    hlsl = translate(LOG_CEIL_WORDS).hlsl_cs()
    for snippet in ('v7 = U(log2(F(v7)));', 'v9 = U(ceil(F(v7)));', LOG_CEIL_LOG_LINE, LOG_CEIL_CEIL_LINE, 'u0.Store(a, v5);'):
        assert hlsl.count(snippet) == 1, (snippet, hlsl)
    for old_ws, new_ws in (
            (vop1('v_log_f32_e32', 'v7', 'v7'), vop1('v_log_f32_e32', 'v7', 'v0')),
            (vop1('v_ceil_f32_e32', 'v9', 'v7'), vop1('v_ceil_f32_e32', 'v9', 'v0'))):
        refused('v_readfirstlane_b32 requires a proven uniform source',
                lambda code=patch(LOG_CEIL_WORDS, pack_hex(*old_ws), pack_hex(*new_ws)): translate(code, group=(8, 1, 1)))
    # Isolate the original proof's opcode dependencies from the new retry.
    retry = G.ComputePipeline.retry_readfirstlane
    with mock_patch.object(G.ComputePipeline, 'retry_readfirstlane',
                           lambda pipe: not pipe.retry_entry_exec and retry(pipe)):
        for removed in ('v_log_f32', 'v_ceil_f32'):
            with mock_patch.object(G, 'UNIFORM_VALU', G.UNIFORM_VALU - {removed}):
                refused('v_readfirstlane_b32 requires a proven uniform source', lambda: translate(LOG_CEIL_WORDS))
    if llvm_mc:
        decoded = G.decode(LOG_CEIL_WORDS)
        lines = [asm for _, _, _, asm in decoded]
        proc = subprocess.run(
            [llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=+wavefrontsize64,-wavefrontsize32', '-show-encoding'],
            input='\n'.join(lines) + '\n', capture_output=True, text=True, check=True)
        built = b''.join(bytes(int(b, 16) for b in m.group(1).split(','))
                         for m in re.finditer(r'encoding: \[([^\]]*)\]', proc.stdout))
        assert built == LOG_CEIL_WORDS, (built.hex(), LOG_CEIL_WORDS.hex())


LOG_CEIL_CONTROLS = {
    'dropped-log2-read': (LOG_CEIL_LOG_LINE, 's21 = 0u;', 'word=0 expected=00000403 actual=00000003'),
    'dropped-ceil-read': (LOG_CEIL_CEIL_LINE, 's22 = 0u;', 'word=0 expected=00000403 actual=00000400'),
}


def log_ceil_dispatch(td, dxc, runner, adapter):
    hlsl = translate(LOG_CEIL_WORDS).hlsl_cs()
    case = td / 'log_ceil.case'
    case.write_bytes(
        struct.pack('<I', GROUPS[0] | (GROUPS[1] << 8) | (GROUPS[2] << 16))
        + struct.pack('<%dI' % WORDS, *log_ceil_expected())
        + struct.pack('<16I', KEY, STRIDE, *([0] * 14)))
    for label, mutation in [('log_ceil_baseline', None)] + [(k, k) for k in LOG_CEIL_CONTROLS]:
        text = hlsl
        if mutation:
            old, new, _ = LOG_CEIL_CONTROLS[mutation]
            assert text.count(old) == 1, mutation
            text = text.replace(old, new)
        src, cso = td / f'{label}.hlsl', td / f'{label}.cso'
        src.write_text(text, encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
        proc = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)],
                              capture_output=True, text=True)
        print(proc.stdout, end='')
        if proc.returncode == 77:
            return 77
        diagnostics = []
        for line in proc.stderr.splitlines():
            if adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '):
                print(line, file=sys.stderr)
            else:
                diagnostics.append(line)
        if mutation:
            want = LOG_CEIL_CONTROLS[mutation][2]
            first = diagnostics[:1]
            if proc.returncode != 1 or first != [want]:
                raise AssertionError('control %s: expected exit 1 with %r, got exit %d %r'
                                     % (mutation, want, proc.returncode, proc.stderr))
            print('control %s (%s): rejected at %s' % (mutation, adapter, first[0]))
        elif proc.returncode or diagnostics:
            raise AssertionError('log_ceil baseline (%s): %s' % (adapter, proc.stderr or 'exit %d' % proc.returncode))
        else:
            print('log_ceil baseline (%s): all %d words match' % (adapter, WORDS))
    return 0


IMG_LOAD_X_LINE = 's21 = ul30_7;'
IMG_LOAD_Y_LINE = 's22 = ul30_8;'
IMG_TEX_WIDTH = 32
IMG_TEX_HEIGHT = 8
IMG_TEX_RGBA32_FLOAT = 1 << 30


def build_img_load_words():
    ws = sop1('s_mov_b32', 's10', 's6') + sop1('s_mov_b32', 's11', 0)
    ws += [0xf4080305, 0xfa000000]  # s_load_dwordx4 s[12:15], s[10:11], 0x0 (u0)
    ws += [0xf40c0405, 0xfa000020]  # s_load_dwordx8 s[16:23], s[10:11], 0x20 (r0)
    ws += [0xbf8cc07f]              # s_waitcnt lgkmcnt(0)
    ws += sopc('s_cmp_eq_u32', 's7', 0) + sopp('s_cbranch_scc0', 1) + sopp('s_nop', 0)
    ws += vop1('v_mov_b32_e32', 'v2', 's7') + vop1('v_mov_b32_e32', 'v3', 2)
    ws += [0xf0000308, 0x00040702]  # image_load v[7:8], v[2:3], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D
    ws += [0xbf8c3f70]              # s_waitcnt vmcnt(0)
    ws += vop1('v_readfirstlane_b32', 's21', 'v7') + vop1('v_readfirstlane_b32', 's22', 'v8')
    ws += sop2('s_xor_b32', 's21', 's21', 's22')
    ws += sop2('s_lshl_b32', 's23', 's7', 2) + vop2('v_add_nc_u32_e32', 'v4', 's23', 'v0')
    ws += vop2('v_add_nc_u32_e32', 'v5', 's21', 'v4') + vop2('v_lshlrev_b32_e32', 'v4', 2, 'v4')
    ws += [0xe0701000, 0x80030504]  # buffer_store_dword v5, v4, s[12:15], 0 offen
    ws += sopp('s_endpgm')
    return struct.pack('<%dI' % len(ws), *ws)


IMG_LOAD_WORDS = build_img_load_words()


def img_load_texels():
    """Distinct finite float32 bit patterns across (x, y) so groups 0 and 1 read distinct uniform x^y pairs."""
    words = []
    for y in range(IMG_TEX_HEIGHT):
        for x in range(IMG_TEX_WIDTH):
            base = 0x3f800000 + ((y * IMG_TEX_WIDTH + x) << 12)
            words.extend((base, base + 0x100, base + 0x200, base + 0x300))
    return words


def img_load_expected(texels):
    out = [SENTINEL] * WORDS
    for gx in range(GROUPS[0]):
        off = (2 * IMG_TEX_WIDTH + gx) * 4
        combined = texels[off] ^ texels[off + 1]
        for tx in range(LOCAL[0]):
            idx = gx * LOCAL[0] + tx
            out[idx] = (combined + idx) & MASK
    return out


def img_load_controls(llvm_mc=None):
    hlsl = translate(IMG_LOAD_WORDS).hlsl_cs()
    for snippet in ('t = r0.Load(int3(v2, v3, 0));', IMG_LOAD_X_LINE, IMG_LOAD_Y_LINE, 'u0.Store(a, v5);'):
        assert hlsl.count(snippet) == 1, (snippet, hlsl)
    # Varying coordinate control: image_load v[7:8], v[0:1], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D.
    refused('v_readfirstlane_b32 requires a proven uniform source',
            lambda: translate(patch(IMG_LOAD_WORDS, '080300f002070400', '080300f000070400')))
    # Image store on a different descriptor slot in the same dispatch must block the uniform image_load proof.
    store_prefix = (sop1('s_mov_b32', 's10', 's6') + sop1('s_mov_b32', 's11', 0)
                    + [0xf40c0605, 0xfa000040, 0xbf8cc07f]  # s_load_dwordx8 s[24:31], s[10:11], 0x40; s_waitcnt lgkmcnt(0)
                    + [0xf0200f08, 0x00060a02])             # image_store v[10:13], v[2:3], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D
    refused('v_readfirstlane_b32 requires a proven uniform source',
            lambda: translate(struct.pack('<%dI' % len(store_prefix), *store_prefix) + IMG_LOAD_WORDS[8:]))
    saved = G.UNIFORM_IMAGE_LOADS
    G.UNIFORM_IMAGE_LOADS = frozenset()
    try:
        refused('v_readfirstlane_b32 requires a proven uniform source', lambda: translate(IMG_LOAD_WORDS))
    finally:
        G.UNIFORM_IMAGE_LOADS = saved
    if llvm_mc:
        lines = llvm_decoded_lines(G.decode(IMG_LOAD_WORDS))
        proc = subprocess.run(
            [llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=+wavefrontsize64,-wavefrontsize32', '-show-encoding'],
            input='\n'.join(lines) + '\n', capture_output=True, text=True, check=True)
        built = b''.join(bytes(int(b, 16) for b in m.group(1).split(','))
                         for m in re.finditer(r'encoding: \[([^\]]*)\]', proc.stdout))
        assert built == IMG_LOAD_WORDS, (built.hex(), IMG_LOAD_WORDS.hex())


IMG_LOAD_CONTROLS = {
    'dropped-img-x-read': (IMG_LOAD_X_LINE, 's21 = 0u;', 'word=0 expected=00000100 actual=3f840100'),
    'dropped-img-y-read': (IMG_LOAD_Y_LINE, 's22 = 0u;', 'word=0 expected=00000100 actual=3f840000'),
}


def img_load_dispatch(td, dxc, runner, adapter):
    hlsl = translate(IMG_LOAD_WORDS).hlsl_cs()
    texels = img_load_texels()
    expected_words = img_load_expected(texels)
    payload = expected_words + texels
    case = td / 'img_load.case'
    header = GROUPS[0] | (GROUPS[1] << 8) | (GROUPS[2] << 16) | IMG_TEX_RGBA32_FLOAT
    case.write_bytes(struct.pack('<I%dI' % len(payload), header, *payload))
    for label, mutation in [('img_load_baseline', None)] + [(k, k) for k in IMG_LOAD_CONTROLS]:
        text = hlsl
        if mutation:
            old, new, _ = IMG_LOAD_CONTROLS[mutation]
            assert text.count(old) == 1, mutation
            text = text.replace(old, new)
        src, cso = td / f'{label}.hlsl', td / f'{label}.cso'
        src.write_text(text, encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
        proc = subprocess.run([runner, adapter, '--compute-texture-case', str(cso), str(case)],
                              capture_output=True, text=True)
        print(proc.stdout, end='')
        if proc.returncode == 77:
            return 77
        diagnostics = []
        for line in proc.stderr.splitlines():
            if adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '):
                print(line, file=sys.stderr)
            else:
                diagnostics.append(line)
        if mutation:
            want = IMG_LOAD_CONTROLS[mutation][2]
            first = diagnostics[:1]
            if proc.returncode != 1 or first != [want]:
                raise AssertionError('control %s: expected exit 1 with %r, got exit %d %r'
                                     % (mutation, want, proc.returncode, proc.stderr))
            print('control %s (%s): rejected at %s' % (mutation, adapter, first[0]))
        elif proc.returncode or diagnostics:
            raise AssertionError('img_load baseline (%s): %s' % (adapter, proc.stderr or 'exit %d' % proc.returncode))
        else:
            print('img_load baseline (%s): all %d words match' % (adapter, WORDS))
    return 0


CONTROLS = {
    'dropped-one-path': (ONE_PATH_LINE, 's25 = 0u;', 'word=0 expected=0000009a actual=00000028'),
    'dropped-loop-read': (LOOP_LINE, 's24 = 0u;', 'word=0 expected=0000009a actual=0000008e'),
    'dropped-int-read': (INT_LINE, 's28 = 0u;', 'word=4 expected=000000ac actual=0000009c'),
    'dropped-const-shift': (CONST_SHIFT_LINE, 's29 = 0u;', 'word=0 expected=0000009a actual=0000008b'),
    'dropped-post-loop': (POST_LOOP_LINE, 's26 = 0u;', 'word=0 expected=0000009a actual=00000096'),
}


def dispatch(dxc, runner, adapter):
    controls()
    hlsl = translate().hlsl_cs()
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    case_bytes = (struct.pack('<I', GROUPS[0] | (GROUPS[1] << 8) | (GROUPS[2] << 16))
                  + struct.pack('<%dI' % WORDS, *expected())
                  + struct.pack('<16I', KEY, STRIDE, *([0] * 14)))
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        td = Path(tmp)
        case = td / 'readfirstlane.case'
        case.write_bytes(case_bytes)

        for label, mutation in [('baseline', None)] + [(k, k) for k in CONTROLS]:
            text = hlsl
            if mutation:
                old, new, _ = CONTROLS[mutation]
                assert text.count(old) == 1, mutation
                text = text.replace(old, new)
            src, cso = td / f'{label}.hlsl', td / f'{label}.cso'
            src.write_text(text, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
            proc = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)],
                                  capture_output=True, text=True)
            print(proc.stdout, end='')
            if proc.returncode == 77:
                return 77
            diagnostics = []
            for line in proc.stderr.splitlines():
                if adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '):
                    print(line, file=sys.stderr)
                else:
                    diagnostics.append(line)
            if mutation:
                want = CONTROLS[mutation][2]
                first = diagnostics[:1]
                if proc.returncode != 1 or first != [want]:
                    raise AssertionError('control %s: expected exit 1 with %r, got exit %d %r'
                                         % (mutation, want, proc.returncode, proc.stderr))
                print('control %s (%s): rejected at %s' % (mutation, adapter, first[0]))
            elif proc.returncode or diagnostics:
                raise AssertionError('baseline (%s): %s' % (adapter, proc.stderr or 'exit %d' % proc.returncode))
            else:
                print('baseline (%s): all %d words match' % (adapter, WORDS))
        rc = log_ceil_dispatch(td, dxc, runner, adapter)
        if rc:
            return rc
        rc = img_load_dispatch(td, dxc, runner, adapter)
        if rc:
            return rc
        return waterfall.dispatch(td, dxc, runner, adapter)


def main(argv):
    if not argv:
        controls()
        return 0
    if len(argv) == 2 and argv[0] in ('binary', '--llvm-mc'):
        controls(llvm_mc=argv[1])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args(argv)
    if args.dispatch:
        return dispatch(*args.dispatch)
    controls(llvm_mc=args.llvm_mc)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
