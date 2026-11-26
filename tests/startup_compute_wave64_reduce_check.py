"""Synthetic gfx1013 wave64 reductions with a D3D12 WARP readback and a losing mutant per operation."""
import argparse
from functools import reduce
from operator import add, and_, or_, xor
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import G, MASK, SENTINEL, USER7, WORDS, refused, root_blob

GROUP = (64, 1, 1)
NATIVE_BRANCH = 'if (WaveGetLaneCount() >= 64u && WaveActiveCountBits(true) == 64u)'
OPS = {
    # VALU encoding byte, scalar combine, inline identity, CPU reference
    'or': ('38', '1a1b1c88', '80', 0, or_),
    'and': ('36', '1a1b1c87', 'c1', MASK, and_),
    'xor': ('3a', '1a1b1c89', '80', 0, xor),
    'sum': ('4a', '1a1b1c80', '80', 0, add),
    'min': ('26', '1a1b9c83', 'c1', MASK, min),
    'max': ('28', '1a1b9c84', '80', 0, max),
}

# Input v5 is loaded before EXEC changes. s4 selects active lanes.
PREFIX = ('06039ebe 80039fbe 0f080cf4000000fa 7fc08cbf 82000c34 001030e006050880 '
          '703f8cbf 7e0496be 0400a87d')


def program(op, *, upper='1b0060d709530100', result='1c020e7e', dpp16=False):
    valu, scalar, identity, _, _ = OPS[op]
    if dpp16:
        dpp = [f'fa1212{valu}09{control}ff' for control in ('b100', '4e00', '4101', '4001')]
    else:
        # DPP8 exchanges lanes at distances 4, 2, 1; DPP16 finishes each row.
        dpp = [f'e91212{valu}09{selectors}' for selectors in ('ac8f68', '1ae2b3', 'c154de')]
        dpp.append(f'fa1212{valu}094001ff')
    rows = [PREFIX, 'c12598be', f'090001d5{identity}0a6200', *dpp,
            '0a0078d709010102', f'0a1312{valu}', '1804febe', '1a0060d709070100', upper, scalar,
            '1604febe', result, '001070e006070980', '000081bf']
    return bytes.fromhex(' '.join(rows))


def translate(code, group=GROUP, **kwargs):
    return G.ComputePipeline(code, root_blob(), USER7, group, **kwargs).hlsl_cs()


def reference(op, words):
    _, _, _, identity, fold = OPS[op]
    return reduce(fold, words, identity) & MASK


def controls():
    for op in OPS:
        assert any(reference(op, SOURCE[:active]) != reference(op, SOURCE[:min(active, 32)])
                   for active in ACTIVE_COUNTS), op
        code = program(op)
        hlsl = translate(code)
        assert hlsl == translate(code, wave_size=64), op
        assert 'groupshared uint x2w_wave64[64];' in hlsl
        assert NATIVE_BRANCH in hlsl
        assert 'GroupMemoryBarrierWithGroupSync();' in hlsl
        assert 's26 = wr' in hlsl and 's27 = wr' in hlsl, hlsl
        assert translate(program(op, dpp16=True)) == hlsl, op
        refused('wave64 requires straight-line compute', lambda: translate(code, wave_size=32))
        refused('readlanes must read opposite halves',
                lambda: translate(program(op, upper='1b0060d709130100')))
    code = program('sum')
    assert 'groupshared uint x2w_wave64[128];' in translate(code, group=(16, 8, 1))
    assert 'WaveGetLaneCount' not in translate(code, group=(16, 8, 1))
    tail = bytes.fromhex('1804febe 1a0060d709070100 1b0060d709530100 1a1b1c80')
    assert translate(code.replace(tail, tail[4:] + tail[:4])) == translate(code)
    refused('permlane temporary v10 is live',
            lambda: translate(code.replace(bytes.fromhex('1c020e7e'), bytes.fromhex('0a030e7e'))))
    refused('expected s_add_u32 combining both halves',
            lambda: translate(code.replace(bytes.fromhex('1a1b1c80'), bytes.fromhex('1a1a1c80'))))
    refused('DPP steps do not reduce each 32-lane half',
            lambda: translate(code.replace(bytes.fromhex('09ac8f68'), bytes.fromhex('09c154de'))))
    refused('expected a full EXEC restore',
            lambda: translate(code.replace(bytes.fromhex('1804febe'), bytes.fromhex('1803febe'))))
    print('PASS: six wave64 reductions, DPP8 and DPP16, and malformed-chain refusals')


# The upper half contributes a new high bit and a smaller unsigned minimum.
SOURCE = [0x100 + (i % 3) for i in range(32)] + [0x80000000, 2]
SOURCE += [0x500 + i for i in range(34, 64)] + [0] * (WORDS - 64)
ACTIVE_COUNTS = (0, 19, 33, 64)


def dispatch(directory, dxc, runner):
    for op in OPS:
        cases, mutations = [], []
        for active in ACTIVE_COUNTS:
            expected = reference(op, SOURCE[:active])
            wrong = reference(op, SOURCE[:min(active, 32)])
            case = directory / f'{op}-{active}.case'
            data = [1, *([expected] * 64), *([SENTINEL] * (WORDS - 64)), active, *([0] * 15), *SOURCE]
            case.write_bytes(struct.pack('<%dI' % len(data), *data))
            cases.append((case, f'{active} active lanes', None))
            if wrong != expected:
                mutations.append((case, f'{active} active lanes, low half only',
                                  f'word=0 expected={expected:08x} actual={wrong:08x}'))
        assert mutations, op
        correct = translate(program(op))
        forced_shared = correct.replace(NATIVE_BRANCH, 'if (false)')
        mutant = translate(program(op, result='1a020e7e'))
        for variant, hlsl, checks in (('auto', correct, cases), ('shared', forced_shared, cases),
                                     ('mutant', mutant, mutations)):
            src = directory / f'{op}-{variant}.hlsl'
            shader = src.with_suffix('.cso')
            src.write_text(hlsl, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
            if check_references(f'{op}-{variant}', runner, 'warp', shader, checks, allow_skip=True) == 77:
                return 77
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch', nargs=2, metavar=('DXC', 'RUNNER'))
    args = parser.parse_args()
    controls()
    if args.dispatch:
        output = Path(__file__).resolve().parents[1] / 'out'
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            return dispatch(Path(tmp), *args.dispatch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
