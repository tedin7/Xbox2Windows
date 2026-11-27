"""Synthetic mixed-use VGPR writes and D3D12 WARP readback."""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_gfx2hlsl_compute_check as C
from startup_flat_param_check import assemble
from startup_gfx2hlsl_wave_reduce_check import CLEAR, G, SIZE, TRIANGLE_VS, dump, translate
from startup_image_3d_check import compile_hlsl

LANE = 3
VALUE = 13
VECTOR_BODY = """\
v_cvt_f32_u32 v13, v0
s_mov_b32 s20, 13.0
v_writelane_b32 v13, s20, 3
v_mul_f32 v14, 2.0, v13
"""
PIXEL = VECTOR_BODY + """\
exp mrt0 v14, v14, v14, v14 done vm
s_endpgm
"""
COMPUTE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx4 s[16:19], s[10:11], 0x10
s_waitcnt lgkmcnt(0)
v_mbcnt_lo_u32_b32 v12, -1, 0
v_lshlrev_b32 v15, 2, v12
""" + VECTOR_BODY.replace(
    'v_writelane_b32 v13, s20, 3\n',
    'v_cmp_ne_u32 vcc_lo, 3, v12\ns_and_saveexec_b32 s21, vcc_lo\n'
    'v_writelane_b32 v13, s20, 3\ns_mov_b32 exec_lo, s21\n'
) + """\
buffer_store_dword v14, v15, s[16:19], 0 offen
s_endpgm
"""
PIXEL_WORDS = bytes.fromhex('000d1a7eff0394be000050410d0061d714060100f41a1c100f1800f80e0e0e0e000081bf')
COMPUTE_WORDS = bytes.fromhex('06038abe80038bbe050408f4100000fa7fc08cbf0c0065d7c100010082181e34000d1a7eff0394be0000504183188a7d6a3c95be0d0061d7140601001503febef41a1c10001070e00f0e0480000081bf')


def shaders():
    ps, reason = translate(PIXEL_WORDS)
    assert ps is not None, reason
    cs = G.ComputePipeline(COMPUTE_WORDS, C.root_blob(), C.USER7 | C.tidig(1), (4, 2, 1)).hlsl_cs()
    return ps, cs


def mutant(text, index):
    write = f'v13 = ({index} == {LANE}u) ? 0x41500000u : v13;'
    assert text.count(write) == 1, text
    return text.replace(write, write.replace(f'== {LANE}u', f'== {LANE + 1}u'))


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def reference(lane, original):
    return bits(2 * (VALUE if lane == LANE else original))


def check_pixels(path):
    raw = path.read_bytes()
    assert len(raw) == SIZE * SIZE * 16, len(raw)
    selected, retained, mismatches = 0, 0, []
    for i, (r, g, b, a) in enumerate(struct.iter_unpack('<4I', raw)):
        if (r, g, b, a) == (CLEAR,) * 4:
            continue
        x, y = i % SIZE, i // SIZE
        lane_value, = struct.unpack('<f', struct.pack('<I', a))
        assert 0 <= lane_value < 128 and lane_value.is_integer(), (x, y, lane_value)
        lane = int(lane_value)
        original = x | (y << 16)
        selected += lane == LANE and original != VALUE
        retained += lane != LANE and original != VALUE
        want = reference(lane, original)
        if (r, g, b) != (want,) * 3:
            mismatches.append((x, y, lane, hex(want), hex(r)))
    assert selected and retained, (selected, retained)
    return mismatches


def warp(dxc, renderer, ps, cs, ps_bad, cs_bad):
    out = Path(__file__).resolve().parents[1] / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', TRIANGLE_VS, 'vs', 'vs_6_0', directory)
        observe = '    o.mrt0.w = float(WaveGetLaneIndex());\n    return o;'
        for bad, text in enumerate((ps, ps_bad)):
            assert text.count('    return o;') == 1
            pixel = compile_hlsl(dxc, 'dxc', text.replace('    return o;', observe), f'ps{bad}', 'ps_6_0', directory)
            output = directory / f'pixels{bad}.bin'
            run = dump(renderer, 'warp', vs, pixel, output)
            print(run.stdout, end='')
            assert run.returncode == 0, run.stdout + run.stderr
            if 'debug_layer=enabled' in run.stdout:
                assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
            mismatches = check_pixels(output)
            assert bool(mismatches) == bool(bad), mismatches[:3]
            if bad:
                print('pixel mutant mismatch', mismatches[0])

        # The guest's x-major lanes form rows of four. The target lane had EXEC clear during the write.
        expected = [reference(lane, lane % 4) for lane in range(8)] + [C.SENTINEL] * (C.WORDS - 8)
        case = directory / 'compute.case'
        case.write_bytes(struct.pack('<I', 0x010101) + struct.pack(f'<{C.WORDS}I', *expected) + bytes(64))
        for bad, text in enumerate((cs, cs_bad)):
            compute = compile_hlsl(dxc, 'dxc', text, f'cs{bad}', 'cs_6_0', directory)
            run = subprocess.run([renderer, 'warp', '--compute-store-case', str(compute), str(case)],
                                 capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                raise SystemExit(77)
            if bad:
                assert run.returncode == 1 and re.search(r'word=3 expected=[0-9a-f]+ actual=[0-9a-f]+',
                                                         run.stderr), run.stdout + run.stderr
                print(run.stderr, end='')
            else:
                assert run.returncode == 0 and not run.stderr, run.stdout + run.stderr


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'binary':
        assert assemble(sys.argv[2], PIXEL) == PIXEL_WORDS
        assert assemble(sys.argv[2], COMPUTE) == COMPUTE_WORDS
    else:
        assert G.lifted_family('v_writelane_b32: v13 is also used as a vector (v_mul_f32_e32)') == 'writelane-vector'
        ps, cs = shaders()
        ps_bad = mutant(ps, 'WaveGetLaneIndex()')
        cs_bad = mutant(cs, '(tid.x + 4u * (tid.y + 2u * tid.z))')
        if len(sys.argv) > 1:
            dxc, renderer = sys.argv[1:]
            warp(dxc, renderer, ps, cs, ps_bad, cs_bad)
    print('PASS')


if __name__ == '__main__':
    main()
