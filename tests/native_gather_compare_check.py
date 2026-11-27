"""image_gather4_c_lz in the direct C++ image path (#2969).

  native_gather_compare_check.py <compile_dxil> <sdk bin dir> <native_recording_check>

image_gather4_c_lz compares the red component of the four texels of a gather footprint at the view's first mip
against a reference, the first address dword, and returns the four results. The direct frontend lowers
it to DXIL textureGatherCmp (opcode 74) with channel 0, the comparison sampler under the existing comparison field
checks and the base-level sampler fix of image_gather4_lz, on 2D and 2D-array images with dmask 0x1. The programs
reuse comparison_program() of native_direct_container_check.py (clamp X/Y, point with TRUNC_COORD, LESS).

Checks: a compute container compiles and its module calls textureGatherCmp with channel 0, u and v from v1 and v2
and the reference v0 as compare value (the same program with image_sample_c_lz compiles as a control); refusals by
message for dmask 0x2, a cube and a 3D image, d16, image_gather4_c (implicit LOD), _c_l and _c_b; and,
through native_recording_check --graphics-sampler, WARP creates the graphics pipeline of the program of
native_graphics_comparison_pso_warp with the comparison sample replaced by this gather.
"""
import pathlib
import re
import subprocess
import sys
import tempfile

from native_direct_container_check import comparison_program, compute, patched_pixel, vertex, words

REFUSED = 'direct image opcode or descriptor modifiers'


def main(compiler, sdk, probe):
    sdk = pathlib.Path(sdk)
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)

        def compile_cs(label, code):
            marker = folder / (label + '.cdx')
            marker.write_bytes(compute(words(*code, 0x7e0a0300, 0xbf810000), table=True))
            run = subprocess.run([compiler, sdk, folder / 'cache', 'gfx1013-container-cs', marker,
                                  marker.with_suffix('.cso')], capture_output=True, text=True)
            return marker.with_suffix('.cso'), run

        cso, run = compile_cs('gather', comparison_program(op=79))
        assert run.returncode == 0, run.stderr
        dump = subprocess.run([sdk / 'dxc.exe', '-dumpbin', cso], capture_output=True, text=True, check=True).stdout
        calls = [line for line in dump.splitlines() if '@dx.op.textureGatherCmp.f32(i32 74' in line]
        assert len(calls) == 1, dump
        # The module keeps VGPRs in allocas %r256 (v0) and up. Each float argument must be a bitcast of a load of the
        # matching register: u = v1, v = v2, two undef components, then zero offsets, channel 0 and the reference v0.
        casts = dict(re.findall(r'(%[\w.]+) = bitcast i32 (%[\w.]+) to float', dump))
        loads = dict(re.findall(r'(%[\w.]+) = load i32, i32\* %r(\d+),', dump))
        args = re.search(r'textureGatherCmp\.f32\(i32 74, ([^)]*)\)', calls[0]).group(1).split(', ')
        floats = [a.split(' ', 1)[1] for a in args if a.startswith('float ')]
        sources = ['v%d' % (int(loads[casts[f]]) - 256) if f in casts else f for f in floats]
        assert sources == ['v1', 'v2', 'undef', 'undef', 'v0'], (sources, calls[0])
        assert args[-4:-1] == ['i32 0', 'i32 0', 'i32 0'], args
        assert 'textureGather.f32' not in dump and 'sampleCmp' not in dump, dump
        _, run = compile_cs('sample_c_lz', comparison_program(op=47))
        assert run.returncode == 0, run.stderr
        print('module: one textureGatherCmp with channel 0')

        base = comparison_program(op=79)
        image = base[-2]
        cases = [('dmask 0x2', image & ~0xf00 | 0x200, base[-1], REFUSED),
                 ('cube', image & ~0x38 | 3 << 3, base[-1], REFUSED),
                 ('3D', image & ~0x38 | 2 << 3, base[-1], REFUSED),
                 ('d16', image, base[-1] | 0x80000000, 'unsupported image dimension or D16')]  # refused at decode
        cases += [(name, image & ~(0x7f << 18) | op << 18, base[-1], REFUSED)
                  for name, op in (('image_gather4_c', 72), ('image_gather4_c_l', 76), ('image_gather4_c_b', 77))]
        for label, word, second, message in cases:
            _, run = compile_cs(label.replace(' ', '_'), [*base[:-2], word, second])
            assert run.returncode and message in run.stderr, (label, run.stdout, run.stderr)
        _, run = compile_cs('array', [*base[:-2], image & ~0x38 | 5 << 3, base[-1]])
        assert run.returncode == 0, run.stderr
        print('refused: dmask 0x2, cube, 3D, d16, image_gather4_c, _c_l and _c_b; a 2D array compiles')

        # The program of native_graphics_comparison_pso_warp, with image_sample_c_lz (47) replaced by the gather.
        code = [*comparison_program(op=79), 0xf09c0108, 0x00220500, 0x06000b04]
        vs, ps, root = (folder / name for name in ('vertex', 'pixel', 'root.bin'))
        vs_code = [0xbe810380, 0xf40c0200, 0xfa000040, 0x7e020280, 0x7e040280,
                   0xf0200108, 0x80020201, 0x7e020b00, 0xf80008cf, 0x01010101, 0xbf810000]
        vs.write_bytes(vertex(program=vs_code, table=True))
        code += [0xf40c0200, 0xfa000020, 0xf0200108, 0x80020400]
        ps.write_bytes(patched_pixel(len(code), code, True))
        source = folder / 'root.hlsl'
        source.write_text('#define RS "DescriptorTable(SRV(t0), UAV(u0, numDescriptors=2))"\n')
        subprocess.run([sdk / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'RS', source, '-Fo', root], check=True)
        run = subprocess.run([probe, '--graphics-sampler', sdk, vs, ps, root, folder / 'cache'],
                             capture_output=True, text=True)
        print(run.stdout.strip())
        if run.returncode == 77:
            sys.exit(77)
        assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)


if __name__ == '__main__':
    main(*sys.argv[1:])
