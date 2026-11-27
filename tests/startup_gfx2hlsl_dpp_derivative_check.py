"""Translate synthetic DPP16 pixel gradients and optionally compare DXC/WARP readback."""
import struct
import sys
import tempfile
from pathlib import Path

from startup_gfx2hlsl_dpp_quad_swap_check import (
    EXPORT, FULLSCREEN_VS, G, HEIGHT, WIDTH, WQM, check, compile_hlsl, draw,
    failures, pipeline, vop1_dpp, vop2_dpp, vop2_plain, words_file,
)

KINDS = ('fine', 'fine_split', 'coarse', 'coarse_split')
FLOWS = ('straight', 'uniform')
SWAP_X = (1, 0, 3, 2)
SWAP_Y = (2, 3, 0, 1)
COARSE = (
    ((1, 0, 1, 0), (0, 1, 0, 1)),
    ((3, 2, 3, 2), (2, 3, 2, 3)),
    ((2, 2, 0, 0), (0, 0, 2, 2)),
    ((3, 3, 1, 1), (1, 1, 3, 3)),
)


def inline_vop2(op, dst, src0, src1):
    return struct.pack('<I', op << 25 | dst << 17 | src1 << 9 | src0)


def mov(dst, src):
    return struct.pack('<I', 0x7e000200 | dst << 17 | src)


def fixture(kind, flow, mutant=None):
    coarse = kind.startswith('coarse')
    words = struct.pack('<I', 0xbe810382)  # s_mov_b32 s1, 2
    if flow != 'straight':
        if flow == 'divergent':
            # Compare the packed position to 4, then branch on VCC.
            words += struct.pack('<I', 0x7d880084)
            branch = 0xbf870001
        else:
            # s_mov_b32 s0, 3; s_cmp_eq_u32 s0, 3; s_cbranch_scc1.
            compare = 0xbf068400 if mutant == 'branch' else 0xbf068300
            words += struct.pack('<2I', 0xbe800383, compare)
            branch = 0xbf850002 if flow == 'skip-wqm' else 0xbf850001
        fallthrough = mov(0, 128) if flow in ('pre-wqm-write', 'divergent') else struct.pack('<I', 0xbe810384)
        words += struct.pack('<I', branch) + fallthrough
    if flow != 'no-wqm':
        words += WQM
    # v8 = float((x + 2) * (y + 5)), with x and y unpacked from v0.
    words += inline_vop2(0x1b, 6, 255, 0) + struct.pack('<I', 0xffff)
    words += inline_vop2(0x16, 7, 144, 0)
    words += inline_vop2(0x25, 6, 1, 6) + inline_vop2(0x25, 7, 133, 7)
    words += vop2_plain(0x0b, 8, 6, 7)
    words += struct.pack('<I', 0x7e000000 | 8 << 17 | 6 << 9 | 264)
    for index in range(4):
        axis = index // 2 if coarse else index % 2
        swap = (SWAP_X, SWAP_Y)[axis]
        reverse = index % 2 if coarse else index >= 2
        op = 0x05 if reverse else 0x04
        if coarse:
            projected, own = COARSE[index]
            # Read the local endpoint through a pair broadcast, then subtract
            # the other endpoint on the same selected row or column.
            words += vop1_dpp(1, 10, 8, own)
            selected = swap if index == mutant else projected
            local = 10
        else:
            other_axis = (SWAP_Y, SWAP_X)[axis]
            selected = other_axis if index == mutant else swap
            local = 8
        if kind.endswith('_split'):
            words += vop1_dpp(1, 11, 8, selected)
            words += vop2_plain(op, index + 1, 11, local)
        else:
            words += vop2_dpp(op, index + 1, 8, local, selected)
    return words + EXPORT


def gradient(x, y):
    return (x + 2) * (y + 5)


def image(kind):
    coarse = kind.startswith('coarse')
    values = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            for index in range(4):
                if coarse and index < 2:
                    row = (y & ~1) + index
                    delta = gradient(x ^ 1, row) - gradient(x, row)
                elif coarse:
                    col = (x & ~1) + (index - 2)
                    delta = gradient(col, y ^ 1) - gradient(col, y)
                else:
                    nx, ny = (x ^ 1, y) if index % 2 == 0 else (x, y ^ 1)
                    delta = gradient(nx, ny) - gradient(x, y)
                reverse = index % 2 if coarse else index >= 2
                values.append(-delta if reverse else delta)
    return struct.unpack('<%dI' % len(values), struct.pack('<%df' % len(values), *values))


def translate(kind, flow, mutant=None):
    pipe = pipeline(fixture(kind, flow, mutant))
    source = pipe.hlsl_ps()
    assert pipe.ps.needs_sm6
    return source


def mutants(flow):
    return (0, 1, 2, 3, 'branch') if flow == 'uniform' else (0, 1, 2, 3)


def controls():
    for flow in FLOWS:
        for kind in KINDS:
            source = translate(kind, flow)
            if kind.startswith('fine'):
                ok = source.count('QuadReadAcrossX(v8)') == 2 and source.count('QuadReadAcrossY(v8)') == 2
            else:
                ok = source.count('QuadReadLaneAt(v8,') == 32
            check(f'{flow} {kind} difference', ok)
            for mutant in mutants(flow):
                check(f'{flow} {kind} mutant {mutant} changes HLSL', source != translate(kind, flow, mutant))
    for flow, diagnostic in (
        ('skip-wqm', 'does not run before every branch'),
        ('divergent', 'is not proven wave-uniform'),
        ('no-wqm', 'whole-quad exec mask'),
        ('pre-wqm-write', 'before whole-quad EXEC'),
    ):
        try:
            translate('coarse', flow)
        except G.Unsupported as error:
            check(flow + ' refusal', diagnostic in str(error), str(error))
        else:
            check(flow + ' refusal', False)


def warp(dxc, renderer):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs', 'vs_6_0', directory)
        for kind in KINDS:
            expected = words_file(directory, kind, image(kind))
            for flow in FLOWS:
                label = kind + '_' + flow
                for mutant in (None, *mutants(flow)):
                    source = translate(kind, flow, mutant)
                    ps = compile_hlsl(dxc, 'dxc', source, f'{label}_{mutant}', 'ps_6_0', directory)
                    result = draw(renderer, 'warp', vs, ps, expected)
                    if mutant is None:
                        print(result.stdout)
                        ok = result.returncode == 0 and f'all {WIDTH * HEIGHT} pixels matched' in result.stdout
                    else:
                        ok = result.returncode == 1 and 'pixel mismatch' in result.stderr
                    check(f'{label} mutant={mutant}', ok, result.stdout + result.stderr)


if __name__ == '__main__':
    controls()
    if len(sys.argv) > 1:
        warp(*sys.argv[1:])
    sys.exit(bool(failures))
