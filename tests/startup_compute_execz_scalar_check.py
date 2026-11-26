"""Wave-wide scalar results at an EXECZ join, with compute readback controls."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

from compute_load_format_check import check_references
from startup_compute_global_load_check import root_blob_cs
from startup_gfx2hlsl_compute_check import G, SENTINEL, WORDS, assemble, refused, root_blob

GROUP = (4, 1, 1)
OLD = (41, 42, 43, 44)
INPUT = tuple(0x831052 + 37 * i for i in range(WORDS))
ARG = 23
CASES = {
    'and': ('s_and_b32 s32, s5, 10', (ARG & 10,)),
    'bfm': ('s_bfm_b32 s32, s5, 2', (((1 << ARG) - 1) << 2,)),
    'lshl': ('s_lshl_b32 s32, s5, 4', (ARG << 4,)),
    'buffer-x2': ('s_buffer_load_dwordx2 s[32:33], s[20:23], 24', INPUT[6:8]),
    'buffer-x4': ('s_buffer_load_dwordx4 s[32:35], s[20:23], 24', INPUT[6:10]),
    'root-x2': ('s_load_dwordx2 s[32:33], s[6:7], 40', INPUT[10:12]),
    'root-x4': ('s_load_dwordx4 s[32:35], s[6:7], 40', INPUT[10:14]),
}


# The bodies below were independently assembled from source() with gfx1013 encodings.
# Keeping the words here lets the Windows fixture run without an assembler.
BODY_WORDS = {
    'and': '058a2087',
    'bfm': '05822092',
    'lshl': '0584208f',
    'buffer-x2': '0a0824f4180000fa',
    'buffer-x4': '0a0828f4180000fa',
    'root-x2': '030804f4280000fa',
    'root-x4': '030808f4280000fa',
}


def binary(label):
    prefix = ('08038cbe80038dbe060608f4000000fa' if label.startswith('root')
              else '06038cbe80038dbe06050cf4000000fa')
    setup = '7fc08cbfa903a0beaa03a1beab03a2beac03a3be7e039ebe0400a87d'
    body = BODY_WORDS[label]
    branch = struct.pack('<I', 0xbf880000 | (1 + len(body) // 8)).hex()
    suffix = '7fc08cbf1e03febe84000c342002107e2102127e2202147e2302167e001078e006080680000081bf'
    return bytes.fromhex(prefix + setup + branch + body + suffix)


def source(label, op):
    root = label.startswith('root')
    table = 8 if root else 6
    descriptors = 's_load_dwordx4 s[24:27], s[12:13], 0' if root else (
        's_load_dwordx8 s[20:27], s[12:13], 0')
    initialize = '\n'.join(f's_mov_b32 s{32 + i}, {word}' for i, word in enumerate(OLD))
    copy = '\n'.join(f'v_mov_b32 v{8 + i}, s{32 + i}' for i in range(4))
    skip = 3 if 'load' in op else 2
    return f"""\
s_mov_b32 s12, s{table}
s_mov_b32 s13, 0
{descriptors}
s_waitcnt lgkmcnt(0)
{initialize}
s_mov_b32 s30, exec_lo
v_cmpx_gt_u32_e32 s4, v0
s_cbranch_execz {skip}
{op}
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, s30
v_lshlrev_b32_e32 v6, 4, v0
{copy}
buffer_store_dwordx4 v[8:11], v6, s[24:27], 0 offen
s_endpgm
"""


def translate(label, code, group=GROUP, refusals=None):
    root = label.startswith('root')
    return G.ComputePipeline(code, root_blob_cs() if root else root_blob(),
                             (9 if root else 7) << 1, group,
                             refusals=refusals).hlsl_cs()


def controls(llvm_mc):
    shaders = {}
    for label, (op, _) in CASES.items():
        code = binary(label)
        if llvm_mc:
            assert code == assemble(llvm_mc, source(label, op)), label
        hlsl = translate(label, code)
        assert hlsl.count('WaveActiveAnyTrue(') == 1, (label, hlsl)
        reasons = []
        assert translate(label, code, refusals=reasons) == hlsl and not reasons, reasons
        refused('live where the block joins', lambda: translate(label, code, (33, 1, 1)))
        shaders[label] = hlsl
        print(f'{label}: wave vote and wide-group refusal verified')
    return shaders


def reference(label, active):
    new = CASES[label][1]
    record = (*new, *OLD[len(new):]) if active else OLD
    return (*record * GROUP[0], *([SENTINEL] * (WORDS - 4 * GROUP[0])))


def dispatch(directory, shaders, dxc, runner, adapter):
    for label, hlsl in shaders.items():
        cases = []
        for active in (2, 0):
            case = directory / f'{label}-{active}.case'
            contents = (1, *reference(label, active), active, ARG, *([0] * 14), *INPUT)
            case.write_bytes(struct.pack(f'<{len(contents)}I', *contents))
            cases.append((case, f'{active} active lanes', None))
        for variant, code, selected in (
            ('correct', hlsl, cases),
            ('lane-local', hlsl.replace('WaveActiveAnyTrue((exec & 1u) != 0u)', '((exec & 1u) != 0u)'),
             [(cases[0][0], 'inactive lane must see updated result',
               f'word=8 expected={CASES[label][1][0]:08x} actual={OLD[0]:08x}')]),
            ('always-run', hlsl.replace('WaveActiveAnyTrue((exec & 1u) != 0u)', 'true'),
             [(cases[1][0], 'empty wave must keep old result',
               f'word=0 expected={OLD[0]:08x} actual={CASES[label][1][0]:08x}')]),
        ):
            assert variant == 'correct' or code != hlsl, label
            src, shader = directory / f'{label}-{variant}.hlsl', directory / f'{label}-{variant}.cso'
            src.write_text(code, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
            check_references(f'{label} {variant}', runner, adapter, shader, selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args()
    shaders = controls(args.llvm_mc)
    if args.dispatch:
        with tempfile.TemporaryDirectory() as directory:
            dispatch(Path(directory), shaders, *args.dispatch)
