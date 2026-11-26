"""Graphics spill slots, mask bits and descriptor provenance through WARP readback.

Run without arguments for synthetic instruction-stream checks. The WARP mode
takes DXC and offscreen_check, and compares the translated pixel shaders and
their mutants against identical CPU references.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_gfx2hlsl_spill_wave64_check as W
from startup_gfx2hlsl_spill_branches_check import program
from startup_image_3d_check import texel

G = W.G
WAVE32_EXEC = "s_mov_b32 s30, exec_lo\ns_mov_b32 exec_lo, s30\n"

LANES = """\
s_mov_b32 s2, 0x3f800000
s_mov_b32 s3, 0x40000000
s_or_b32 s2, s2, 0
s_or_b32 s3, s3, 0
v_writelane_b32 v10, s2, 0
v_writelane_b32 v10, s3, 32
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_readlane_b32 s4, v10, 0
v_readlane_b32 s5, v10, 32
v_mov_b32 v1, s4
v_mov_b32 v2, s5
exp mrt0 v1, v2, v1, v2 done vm
s_endpgm
"""

MASK = """\
v_and_b32 v1, 1, v0
v_cmp_eq_u32_e64 s7, 0, v1
v_writelane_b32 v10, s7, 3
s_mov_b32 s7, 0
v_readlane_b32 s9, v10, 3
v_cndmask_b32_e64 v2, 1.0, 2.0, s9
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
"""

DESCRIPTOR = """\
v_writelane_b32 v10, s8, 4
s_mov_b32 s8, 0
v_readlane_b32 s8, v10, 4
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_waitcnt lgkmcnt(0)
v_writelane_b32 v10, s16, 5
s_mov_b32 s16, 0
v_readlane_b32 s16, v10, 5
s_mov_b32 s24, 0
s_mov_b32 s25, 0
s_mov_b32 s26, 0
s_mov_b32 s27, 0
v_mov_b32 v0, 0x3e800000
v_mov_b32 v1, 0x3f400000
v_mov_b32 v2, s9
image_sample v[0:3], v[0:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_3D
s_waitcnt vmcnt(0)
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
"""


def translate(text, wave=None, packed=False):
    refusals = []
    pipeline = G.Pipeline(W.VS_WORDS, b'', W.root_blob(), ps_hw_inputs=W.PACKED if packed else None,
                          decoded={'ps': program(text)}, refusals=refusals, stage_wave_sizes={'ps': wave})
    return refusals or pipeline.hlsl_ps()


def accepted(text, wave=None, packed=False):
    source = translate(text, wave, packed)
    assert isinstance(source, str), source
    return source


def refused(text, fragment, wave=None, packed=False):
    result = translate(text, wave, packed)
    assert isinstance(result, list) and any(fragment in reason for reason, _ in result), fragment


def lane_reference(wave):
    slots = {}
    slots[0] = 0x3f800000
    slots[32 % wave] = 0x40000000
    return [slots[0], slots[32 % wave]] * 2 * (W.WIDTH * W.HEIGHT)


def mask_reference():
    return [0x3f800000 if x & 1 else 0x40000000
            for _ in range(W.HEIGHT) for x in range(W.WIDTH) for _ in range(4)]


def cases():
    wave32 = accepted(WAVE32_EXEC + LANES)
    wave64 = accepted(LANES, 64)
    mask = accepted(WAVE32_EXEC + MASK, packed=True)
    descriptor = accepted(DESCRIPTOR)
    return [
        ('wave32', wave32, W.mutate(wave32, r'sp10_0 = s3;', 'sp10_0 = s2;'), lane_reference(32)),
        ('wave64', wave64, W.mutate(wave64, r's4 = sp10_0;', 's4 = sp10_32;'), lane_reference(64)),
        ('mask', mask, W.mutate(mask, r'sp10_3 = s7;', 'sp10_3 = s7 ^ 1u;'), mask_reference()),
        ('descriptor', descriptor, W.mutate(descriptor, r'r0\.Sample\(q0, float3\(F\(v0\), F\(v1\), F\(v2\)\)\)',
                                            'float4(0, 0, 0, 0)'), None),
    ]


def controls():
    refused(LANES, 'one slot in wave32 and two in wave64')
    refused(MASK, 'holds a lane mask or descriptor', packed=True)
    refused(MASK.replace('v_cndmask_b32_e64 v2, 1.0, 2.0, s9', 'v_mov_b32 v2, s9'),
            'reloads a lane mask that v_mov_b32 reads as data', 32, packed=True)
    refused(LANES.replace('v_writelane_b32 v10, s2, 0\n', ''), 'before any', 64)
    assert 's4 = sp10_0;' in accepted(LANES, 32)
    assert 's5 = sp10_0;' in accepted(LANES, 32)
    assert 's5 = sp10_32;' in accepted(LANES, 64)
    cross_lane = LANES.replace('v_writelane_b32 v10, s2, 0\n', '')
    assert 's4 = sp10_0;' in accepted(cross_lane, 32)
    branch = cross_lane.replace('v_readlane_b32 s4', 's_branch Lreload\nLreload:\nv_readlane_b32 s4')
    assert 's4 = sp10_0;' in accepted(branch, 32)
    assert 's4 = sp10_0;' in accepted(WAVE32_EXEC + LANES)
    assert '    sp10_' not in accepted(DESCRIPTOR)
    refused(DESCRIPTOR.replace('v_writelane_b32 v10, s8, 4', 'v_writelane_b32 v10, 0, 4'),
            's_load through s8')
    print('Graphics spill controls passed')


def draw(dxc, renderer, shaders):
    out = Path(__file__).resolve().parents[1] / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        vs = W.compile_hlsl(dxc, 'dxc', W.FULLSCREEN_VS, 'vs', 'vs_6_0', directory)
        for name, source, mutant, reference in shaders:
            for bad, hlsl in enumerate((source, mutant)):
                ps = W.compile_hlsl(dxc, 'dxc', hlsl, '%s_%d' % (name, bad), 'ps_6_0', directory)
                if reference is None:
                    args = ['--startup-image-3d', str(vs), str(ps), '0.75', texel(0, 1, 1)]
                    diagnostic = 'startup_image mismatch'
                else:
                    path = directory / 'reference.bin'
                    path.write_bytes(struct.pack('<%dI' % len(reference), *reference))
                    args = ['--startup-pixel-words', str(vs), str(ps), str(W.WIDTH), str(W.HEIGHT), str(path)]
                    diagnostic = 'pixel mismatch'
                run = subprocess.run([renderer, 'warp', *args], capture_output=True, text=True)
                print(run.stdout + run.stderr, end='')
                if run.returncode == 77:
                    sys.exit(77)
                if bad:
                    assert run.returncode != 0 and diagnostic in run.stderr, (name, run.returncode)
                else:
                    assert run.returncode == 0, (name, run.returncode)
            print('%s: readback matched and mutant mismatched' % name)


if __name__ == '__main__':
    controls()
    shaders = cases()
    if len(sys.argv) > 1:
        dxc, renderer = sys.argv[1:]
        draw(dxc, renderer, shaders)
