"""Synthetic compute EXECZ exits with a D3D12 readback against lane-wise references."""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, assemble, refused, root_blob

PREFIX = (
    ('s_mov_b32 s10, s6', '06038abe'),
    ('s_mov_b32 s11, 0', '80038bbe'),
    ('s_load_dwordx8 s[12:19], s[10:11], 0', '05030cf4000000fa'),
    ('s_waitcnt lgkmcnt(0)', '7fc08cbf'),
    ('v_lshlrev_b32_e32 v4, 2, v0', '82000834'),
    ('v_mov_b32_e32 v7, 3', '83020e7e'),
    ('s_mov_b32 s26, exec_lo', '7e039abe'),
    ('v_cmpx_gt_u32_e32 s4, v0', '0400a87d'),
)
CMP = ('s_cmp_eq_u32 s5, 0', '058006bf')
EARLY = ('v_mov_b32_e32 v7, 7', '87020e7e')
ARM = ('v_mov_b32_e32 v7, 9', '89020e7e')
LATE = ('v_mov_b32_e32 v7, 11', '8b020e7e')
NOP = ('s_nop 0', '000080bf')
RESTORE = ('s_mov_b32 exec_lo, s26', '1a03febe')
STORE = ('buffer_store_dword v7, v4, s[16:19], 0 offen', '001070e004070480')
END = ('s_endpgm', '000081bf')
BRANCH = {'s_branch': 0xbf82, 's_cbranch_scc1': 0xbf85, 's_cbranch_execz': 0xbf88}


def program(kind, *, mutant=False, join_op=LATE, tail_ops=()):
    rows = [*PREFIX, ('s_cbranch_execz', '@join'), CMP, ('s_cbranch_scc1', '@late'), EARLY,
            NOP if mutant else ('s_branch', '@' + {'join': 'join', 'shared': 'tail', 'split': 'arm'}[kind]),
            'late']
    if kind == 'join':
        rows += [LATE, 'join']
    else:
        rows += [NOP, 'join', join_op]
        if kind == 'split':
            rows += [('s_branch', '@tail'), 'arm', ARM]
    rows += ['tail', RESTORE, *tail_ops, STORE, END]
    offsets, instructions = {}, []
    offset = 0
    for row in rows:
        if isinstance(row, str):
            offsets[row] = offset
        else:
            instructions.append((offset, *row))
            offset += 1 if row[1].startswith('@') else len(row[1]) // 8
    for offset, asm, encoded in instructions:
        if encoded.startswith('@'):
            delta = offsets[encoded[1:]] - offset - 1
            encoded = struct.pack('<I', BRANCH[asm] << 16 | (delta & 0xffff)).hex()
            asm = f'{asm} {delta}'
        yield asm, encoded


def translate(rows, llvm_mc=None, group=(4, 1, 1)):
    rows = tuple(rows)
    binary = bytes.fromhex(''.join(code for _, code in rows))
    if llvm_mc:
        source = '\n'.join(asm for asm, _ in rows) + '\n'
        assert assemble(llvm_mc, source) == binary, source
    return G.ComputePipeline(binary, root_blob(), USER7, group).hlsl_cs()


KINDS = ('join', 'shared', 'split')


def controls(llvm_mc):
    for kind in KINDS:
        good = translate(program(kind), llvm_mc)
        bad = translate(program(kind, mutant=True), llvm_mc)
        assert good != bad, kind
    multi_wave = (65, 1, 1)
    refused('exec write', lambda: translate(program('shared', join_op=RESTORE), llvm_mc, multi_wave))
    refused('branch out of the block', lambda: translate(program('shared', join_op=END), llvm_mc, multi_wave))
    refused('branch out of the block',
            lambda: translate(program('shared', join_op=('s_branch', '@late')), llvm_mc, multi_wave))
    refused('definition of s27', lambda: translate(program(
        'shared', join_op=('s_mov_b32 s27, 1', '81039bbe'),
        tail_ops=(('v_mov_b32_e32 v7, s27', '1b020e7e'),)), llvm_mc, multi_wave))
    print('EXECZ exit translations and unsafe-path refusals passed')


def dispatch(directory, dxc, runner, adapter):
    for kind in KINDS:
        checks, mutants = [], []
        early = 9 if kind == 'split' else 7
        for active in (0, 1, 3, 4):
            for take_exit in (0, 1):
                result = early if take_exit else 11
                expected = [result if lane < active else 3 for lane in range(4)]
                case = directory / f'{kind}-{active}-{take_exit}.case'
                words = [1, *expected, *([SENTINEL] * (WORDS - 4)), active, take_exit,
                         *([0] * 14), *([0] * WORDS)]
                case.write_bytes(struct.pack('<%dI' % len(words), *words))
                label = f'active={active} early_exit={take_exit}'
                checks.append((case, label, None))
                diagnostic = f'word=0 expected={early:08x} actual=0000000b' if active and take_exit else None
                mutants.append((case, label, diagnostic))
        for mutant, references in ((False, checks), (True, mutants)):
            name = f'{kind}-mutant' if mutant else kind
            src, shader = directory / f'{name}.hlsl', directory / f'{name}.cso'
            src.write_text(translate(program(kind, mutant=mutant)), encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
            if check_references(name, runner, adapter, shader, references, allow_skip=True) == 77:
                return 77
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args()
    controls(args.llvm_mc)
    if args.dispatch:
        out = Path(__file__).resolve().parents[1] / 'out'
        out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            return dispatch(Path(tmp), *args.dispatch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
