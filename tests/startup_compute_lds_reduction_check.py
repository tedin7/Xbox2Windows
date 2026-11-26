#!/usr/bin/env python3
"""Check a synthetic six-phase gfx1013 LDS reduction on D3D12 WARP."""

import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_compute_lds_b64_check import G, ROOT, SENTINEL, WORDS, assemble
from startup_gfx2hlsl_compute_check import TGID_X, USER7, refused, renderer_diagnostics, root_blob

HALVES = (128, 64, 32, 16, 8, 4)
RSRC2 = USER7 | TGID_X | (1 << 11) | (4 << 15)

HEADER = """\
s_mov_b32 s14, s6
s_mov_b32 s15, 0
s_load_dwordx8 s[16:23], s[14:15], 0x0
s_waitcnt lgkmcnt(0)
s_mov_b64 s[10:11], exec
v_lshl_add_u32 v2, v1, 2, v0
v_lshlrev_b32 v4, 3, v2
v_mov_b32 v5, 0x40003c00
v_lshlrev_b32 v6, 14, v2
v_and_b32 v6, 0x200000, v6
v_or_b32 v6, 0x42003c00, v6
v_fma_f16 v5, v5, v6, v6 op_sel:[1,0,1,1]
v_pk_add_f16 v6, v6, 0x3c003800
ds_write_b64 v4, v[5:6]
s_waitcnt lgkmcnt(0)
s_barrier
"""

PHASE = """\
v_cmpx_gt_u32 {half}, v2
s_cbranch_execz 10
ds_read_b64 v[8:9], v4 offset:{stride}
s_waitcnt lgkmcnt(0)
v_pk_add_f16 v5, v5, v8
v_pk_add_f16 v6, v6, v9
ds_write_b64 v4, v[5:6]
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, s[10:11]
s_barrier
"""

TAIL = """\
v_mov_b32 v10, v5
ds_read_u16_d16 v10, v4 offset:6
s_waitcnt lgkmcnt(0)
v_lshlrev_b32 v13, 2, v2
buffer_store_dword v10, v13, s[20:23], 0 offen
s_endpgm
"""
SOURCE = HEADER + ''.join(PHASE.format(half=n, stride=8 * n) for n in HALVES) + TAIL

# Source-built c67 fixture bytes. Binary mode checks these against the assembly above.
C67 = bytes.fromhex(
    '06038ebe80038fbe07040cf4000000fa7fc08cbf7e048abe020046d70105010483040834ff020a7e003c0040'
    '8e040c34ff0c0c3600002000ff0c0c38003c004205684bd7050d1a0406400fcc06ff01180038003c000034d9'
    '040500007fc08cbf00008abfff04a87d800000000a0088bf0004d8d9040000087fc08cbf05400fcc05110218'
    '06400fcc06130218000034d9040500007fc08cbf0a04febe00008abfc004a87d0a0088bf0002d8d904000008'
    '7fc08cbf05400fcc0511021806400fcc06130218000034d9040500007fc08cbf0a04febe00008abfa004a87d'
    '0a0088bf0001d8d9040000087fc08cbf05400fcc0511021806400fcc06130218000034d9040500007fc08cbf'
    '0a04febe00008abf9004a87d0a0088bf8000d8d9040000087fc08cbf05400fcc0511021806400fcc06130218'
    '000034d9040500007fc08cbf0a04febe00008abf8804a87d0a0088bf4000d8d9040000087fc08cbf05400fcc'
    '0511021806400fcc06130218000034d9040500007fc08cbf0a04febe00008abf8404a87d0a0088bf2000d8d9'
    '040000087fc08cbf05400fcc0511021806400fcc06130218000034d9040500007fc08cbf0a04febe00008abf'
    '0503147e060098da0400000a7fc08cbf82041a34001070e00d0a0580000081bf'
)


def translate(code=C67, rsrc2=RSRC2):
    return G.ComputePipeline(code, root_blob(), rsrc2, (4, 64, 1)).hlsl_cs()


def half_bits(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def half(value):
    return struct.unpack('<e', struct.pack('<e', value))[0]


def expected_words():
    # Store one word per lane. Its upper half exposes the FMA accumulator.
    # Its lower half comes from the high half of the second LDS dword.
    fma = []
    lds = []
    for lane in range(256):
        high = 3 + (lane >> 7 & 1) / 16
        fma.append(half(2 * 1 + high))
        lds.append(half(high + 1))
    for active in HALVES:
        for lane in range(active):
            fma[lane] = half(fma[lane] + fma[lane + active])
            lds[lane] = half(lds[lane] + lds[lane + active])
    words = [(half_bits(a) << 16) | half_bits(b) for a, b in zip(fma, lds)]
    return words + [SENTINEL] * (WORDS - len(words))


CONTROLS = {
    'exchange': ('a = (v4 + 1024u) >> 2;', 'a = (v4 + 0u) >> 2;'),
    'fma-select': ('f16tof32((v5 >> 16u) & 0xffffu)', 'f16tof32((v5 >> 0u) & 0xffffu)'),
    'literal-select': ('f16tof32(0x3c00u)', 'f16tof32(0x3800u)'),
    'half-address': ('((a & 2u) * 8u)', '0u'),
    'half-preserve': ('(v10 & 0xffff0000u) | ', ''),
}


def mutate(hlsl, control):
    old, new = CONTROLS[control]
    assert hlsl.count(old) == 1, (control, old)
    return hlsl.replace(old, new)


def check_proof():
    hlsl = translate()
    main = hlsl[hlsl.index('[numthreads'):].splitlines()
    assert main.count('    GroupMemoryBarrierWithGroupSync();') == 7
    assert sum(line.startswith('    if (exec & 1) lds[') for line in main) == 12
    assert sum(' = lds[a' in line and line.startswith('    if (exec & 1)') for line in main) == 12
    assert all('switch' not in line and 'WaveActive' not in line for line in main)
    for control in CONTROLS:
        mutate(hlsl, control)

    # Unsafe shapes retain the legacy refusal. Each edit changes only one
    # instruction and exercises a different condition of the proof.
    for old, new in (
        ('020046d701050104', '020046d701070104'),  # wrong X/Y stride
        ('0004d8d904000008', '0804d8d904000008'),  # unproven peer
        ('0a0088bf', '0b0088bf'),  # skips the EXEC restore
        ('0a04febe', '0c04febe'),  # restores a different SGPR pair
        ('040000087fc08cbf', '04000008000080bf'),  # consuming an LDS result without a wait
        ('060098da', '070098da'),  # unaligned D16 access
    ):
        before, after = bytes.fromhex(old), bytes.fromhex(new)
        assert before in C67, old
        refused('LDS exchange needs a 1D group', lambda: translate(C67.replace(before, after, 1)))
    refused('LDS exchange needs a 1D group', lambda: translate(rsrc2=RSRC2 - (1 << 15)))
    refused('LDS exchange needs a 1D group', lambda: translate(rsrc2=RSRC2 ^ (1 << 11)))
    print('six-phase LDS proof and unsafe controls pass')


def dispatch(dxc, runner, adapter, control=None):
    hlsl = translate()
    if control:
        hlsl = mutate(hlsl, control)
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        src, cso, case = (directory / ('reduction.' + ext) for ext in ('hlsl', 'cso', 'case'))
        src.write_text(hlsl, encoding='ascii')
        subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
        case.write_bytes(struct.pack('<%dI' % (1 + WORDS + 16),
                                     0x010101, *expected_words(), *([0] * 16)))
        run = subprocess.run([str(Path(runner).resolve()), adapter, '--compute-store-case', str(cso), str(case)],
                             capture_output=True, text=True)
    print(run.stdout, end='')
    if run.returncode == 77:
        return 77
    diagnostics = renderer_diagnostics(run.stderr, adapter)
    if control:
        assert run.returncode == 1 and diagnostics and re.fullmatch(
            r'word=\d+ expected=[0-9a-f]{8} actual=[0-9a-f]{8}', diagnostics[0]), (control, run.stderr)
        print('control %s: rejected at %s' % (control, diagnostics[0]))
    else:
        assert run.returncode == 0 and not diagnostics, (run.returncode, run.stderr)
        print('all %d readback words match the CPU reference' % WORDS)
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_proof()
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        assert assemble(sys.argv[2], SOURCE) == C67
        print('LDS reduction fixture bytes match the assembly')
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(dispatch(*sys.argv[2:]))
    elif len(sys.argv) == 6 and sys.argv[1] == 'control':
        sys.exit(dispatch(*sys.argv[3:], control=sys.argv[2]))
    else:
        sys.exit('expected binary, dispatch, or control arguments')
