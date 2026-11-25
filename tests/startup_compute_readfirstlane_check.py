"""Source-built gfx1013 compute fixture for one-path SGPRs and uniform loop counters with v_readfirstlane_b32 (#2274).

  (no arguments)                              decode, translate and check controls + negative controls (pure Python)
  --llvm-mc <llvm-mc> / binary <llvm-mc>      also reassemble the fixture and patched instructions with llvm-mc
  --dispatch <dxc> <offscreen_check> <adapter>  compile translated HLSL with DXC, run GPU readback and negative controls

The shader builds a one-path SGPR ``s20`` across a uniform forward branch whose taken path does not write ``s20``,
reads it through ``v_readfirstlane_b32`` at the join, runs a four-iteration uniform loop whose counter ``s22`` is
incremented uniformly and read back via ``v_readfirstlane_b32`` both inside the loop and after loop exit, and stores
one combined word per thread across two four-thread groups ``(4, 1, 1) x (2, 1, 1)`` so both arms of the branch
run on the GPU.
"""
import argparse
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))

import disasm_rdna2 as D  # noqa: E402
from startup_gfx2hlsl_compute_check import (  # noqa: E402
    CASES, G, KEY, MASK, SENTINEL, STRIDE, WORDS, patch, refused, root_blob,
)

RSRC2 = CASES['D'][2]  # USER7 | TGID_X | tidig(0)
LOCAL = (4, 1, 1)
GROUPS = (2, 1, 1)
ONE_PATH_LINE = 's25 = uint(max(F((U(2.0 * F((U(float(s20))))))), 0.0));'
LOOP_LINE = 's24 = uint(max(F((U(2.0 * F((U(float(s22))))))), 0.0));'
POST_LOOP_LINE = 's26 = s22;'


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
    ws += sop2('s_add_u32', 's25', 's25', 's21') + sop2('s_add_u32', 's25', 's25', 's23')
    ws += sop2('s_add_u32', 's25', 's25', 's26')
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
)


def translate(code=None):
    return G.ComputePipeline(code or PROGRAM_WORDS, root_blob(), RSRC2, LOCAL)


def expected():
    out = [SENTINEL] * WORDS
    base_stride = STRIDE & 63
    for gx in range(GROUPS[0]):
        s20 = base_stride + gx
        s25 = s20 * 2
        s21 = 9 if gx == 0 else 5
        s23 = 0 + 2 + 4 + 6  # sum over uniform loop iterations i=0..3
        s26 = 4
        combined = (s25 + s21 + s23 + s26) & MASK
        for tx in range(LOCAL[0]):
            idx = gx * LOCAL[0] + tx
            out[idx] = (combined + idx) & MASK
    return out


def llvm_lines():
    lines = []
    for off, _n, _name, asm in DECODED:
        m = re.fullmatch(r'(s_branch|s_cbranch_\w+) label_0x([0-9a-f]+)', asm)
        if m:
            lines.append('%s %d' % (m.group(1), (int(m.group(2), 16) - off - 4) // 4))
        else:
            lines.append(asm.replace('vcc_lo', 'vcc'))
    lines += [
        's_cbranch_vccnz -8',
        's_cbranch_vccz 1',
        's_mov_b32 s20, 9',
        's_mov_b32 s22, vcc_lo',
    ]
    return lines


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
    for snippet in (ONE_PATH_LINE, LOOP_LINE, POST_LOOP_LINE, 'u0.Store(a, v5);'):
        assert hlsl.count(snippet) == 1, (snippet, hlsl)
    assert 'uav 0 table 1 16' in pipe.manifest().splitlines(), pipe.manifest()

    for label, replacements, reason in PATCHES:
        code = PROGRAM_WORDS
        for old_hex, new_hex in replacements:
            code = patch(code, old_hex, new_hex)
        refused(reason, lambda code=code: translate(code))

    if llvm_mc:
        check_binary(llvm_mc)
    print('Compute readfirstlane one-path, loop-counter and refusal checks passed')


CONTROLS = {
    'dropped-one-path': (ONE_PATH_LINE, 's25 = 0u;', 'word=0 expected=0000008b actual=00000019'),
    'dropped-loop-read': (LOOP_LINE, 's24 = 0u;', 'word=0 expected=0000008b actual=0000007f'),
    'dropped-post-loop': (POST_LOOP_LINE, 's26 = 0u;', 'word=0 expected=0000008b actual=00000087'),
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
            if mutation:
                want = CONTROLS[mutation][2]
                first = proc.stderr.strip().splitlines()[:1]
                if proc.returncode != 1 or first != [want]:
                    raise AssertionError('control %s: expected exit 1 with %r, got exit %d %r'
                                         % (mutation, want, proc.returncode, proc.stderr))
                print('control %s (%s): rejected at %s' % (mutation, adapter, first[0]))
            elif proc.returncode or proc.stderr:
                raise AssertionError('baseline (%s): %s' % (adapter, proc.stderr or 'exit %d' % proc.returncode))
            else:
                print('baseline (%s): all %d words match' % (adapter, WORDS))
    return 0


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
