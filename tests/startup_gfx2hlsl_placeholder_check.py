"""Synthetic pipeline cache checks, plus optional 8x8 D3D12 WARP readback."""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from startup_gfx2hlsl_export_null_draw_check import G, I3, root_blob
import tester_pack


def export(slot, regs, *, mask=15, done=True):
    header = 0xf8000000 | slot << 4 | mask | int(done) << 11
    return struct.pack('<I4B', header, *regs)


def mov(reg, value):
    return struct.pack('<If', 0x7e0002ff | reg << 17, value)


def corner(vertex, dest):
    compare = 0x7c000000 | 0xc2 << 17 | (128 + vertex)  # v_cmp_eq_u32 vcc_lo, vertex, v0
    select = 1 << 25 | dest << 17 | 5 << 9 | 243     # v_cndmask_b32 dest, -1.0, v5, vcc_lo
    return struct.pack('<II', compare, select)


END = struct.pack('<I', 0xbf810000)
TRAP = struct.pack('<I', 0xbf920005)
VS = (mov(5, 3.0) + corner(1, 1) + corner(2, 2) + mov(3, 0.25) + mov(4, 1.0)
      + export(12, (1, 2, 3, 4)) + END)
VALUES = mov(6, 0.5)
MRT0 = export(0, (6, 6, 6, 6))
SINGLE = VALUES + TRAP + MRT0 + END
MIXED = VALUES + TRAP + export(0, (6, 6, 6, 6), done=False) + export(1, (6, 6, 6, 6)) + END
REASON = 's_trap'
DRAWN = REASON + ' (placeholder drawn)'


def request(directory, vs, ps, formats=()):
    root = root_blob()
    base = G.pipeline_key(vs, ps, root)
    for extension, blob in (('.b', ps), ('.root', root), ('.a', vs)):
        (directory / ('req_%016x' % base + extension)).write_bytes(blob)
    if not formats:
        return base
    line = 'rtv1 %d %s' % (base, ' '.join(map(str, formats)))
    key = G.fnv(line.encode('ascii'))
    (directory / ('req_%016x.rtv' % key)).write_text(line, encoding='ascii')
    return key


def result_file(directory, key, extension):
    return directory / ('%016x' % key + extension)


def check_requests(directory):
    supported = request(directory, VS, VALUES + MRT0 + END)
    single = request(directory, VS, SINGLE)
    mixed = request(directory, VS, MIXED, (28, 30))
    bad_vs = request(directory, TRAP + VS, SINGLE)
    bad_export = request(directory, VS, VALUES + TRAP + export(10, (6, 6, 6, 6)) + MRT0 + END)

    def translate(key):
        return G.translate_request(str(directory), key)

    def content(key, extension):
        return result_file(directory, key, extension).read_text(encoding='ascii')

    with patch.dict(os.environ, X2W_FXC='', X2W_PLACEHOLDER_RGB='0,255,0'):
        os.environ.pop('X2W_PLACEHOLDER_PS', None)
        assert translate(supported) == 'ok'
        suffixes = ('.vs.hlsl', '.ps.hlsl', '.txt')
        unchanged = [result_file(directory, supported, ext).read_bytes() for ext in suffixes]
        for key in (single, mixed, bad_vs):
            assert translate(key) == 'unsupported: ' + REASON
            assert content(key, '.fail').splitlines()[0] == REASON
            assert not result_file(directory, key, '.txt').exists()
        assert single not in G.pending_keys(str(directory), True)

        os.environ['X2W_PLACEHOLDER_PS'] = '1'
        assert single in G.pending_keys(str(directory), True)
        assert translate(supported) == 'ok'
        assert unchanged == [result_file(directory, supported, ext).read_bytes() for ext in suffixes]
        for key, targets in ((single, '0'), (mixed, '0 1')):
            assert translate(key) == 'ok'
            manifest = content(key, '.txt').splitlines()
            assert 'placeholder ps ' + REASON in manifest and 'targets ' + targets in manifest
            assert content(key, '.fail').splitlines().count(DRAWN) == 1
            assert 'SV_Depth' not in content(key, '.ps.hlsl')
            assert key not in G.pending_keys(str(directory), True)
        assert 'uint4 mrt1 : SV_Target1;' in content(mixed, '.ps.hlsl')
        assert translate(bad_vs) == translate(bad_export) == 'unsupported: ' + REASON
        for key in (bad_vs, bad_export):
            assert not result_file(directory, key, '.txt').exists()
        assert bad_vs not in G.pending_keys(str(directory), True)
        rows = tester_pack.cache_results(str(directory))
        assert [(status, reason) for key, _, status, reason in rows if key == '%016x' % single] == [('refused', DRAWN)]
        assert tester_pack.clean_reason('long reason ' * 30 + DRAWN).endswith(G.PLACEHOLDER_DRAWN)
        with patch.dict(os.environ, X2W_FXC='fxc.exe'):
            assert single in G.pending_keys(str(directory), True)  # source-only result needs bytecode

        depth_code = VALUES + TRAP + export(8, (6, 0, 0, 0), mask=1, done=False) + MRT0 + END
        depth = G.Pipeline(VS, depth_code, root_blob()).hlsl_ps()
        assert 'float depth : SV_Depth;' in depth and 'o.depth = pos.z;' in depth
        os.environ['X2W_PLACEHOLDER_RGB'] = 'bad'
        assert G.placeholder_rgb() == (255, 0, 255)
        assert single in G.pending_keys(str(directory), True)
        assert '255.0/255.0, 0.0/255.0, 255.0/255.0' in G.Pipeline(VS, SINGLE, root_blob()).hlsl_ps()
        os.environ['X2W_PLACEHOLDER_PS'] = '0'
        assert single in G.pending_keys(str(directory), True)
        shaders = tuple(content(key, '.ps.hlsl') for key in (single, mixed))
        assert translate(single) == 'unsupported: ' + REASON
        assert not result_file(directory, single, '.txt').exists()
        assert content(single, '.fail').splitlines()[0] == REASON
    return content(supported, '.vs.hlsl'), shaders


def compare(data, reference, tolerance, label, expected_mismatches=0):
    assert len(data) == len(reference) * 64, f'{label}: expected 256 bytes, got {len(data)}'
    mismatches = []
    for index in range(64):
        pixel = tuple(data[index * 4:index * 4 + 4])
        if any(abs(actual - want) > tolerance for actual, want in zip(pixel, reference)):
            mismatches.append((index % 8, index // 8, pixel))
    assert len(mismatches) == expected_mismatches, (
        f'{label}: {len(mismatches)} differing pixels, expected {expected_mismatches}. '
        f'Reference RGBA {reference}, first mismatch (x, y, RGBA) {mismatches[:1]}'
    )


def draw_checks(directory, vs_source, shaders, dxc, renderer):
    def compile_shader(source, name, profile):
        try:
            return I3.compile_hlsl(dxc, 'dxc', source, name, profile, directory)
        except subprocess.CalledProcessError as error:
            raise SystemExit(error.stderr) from error

    vs = compile_shader(vs_source, 'triangle', 'vs_6_0')
    single, mixed = (compile_shader(source, name, 'ps_6_0') for source, name in zip(shaders, ('single', 'mixed')))
    output = directory / 'readback.bin'

    def readback(ps, layout):
        run = subprocess.run([renderer, 'warp', '--startup-placeholder-pixels', str(vs), str(ps), layout, str(output)],
                             capture_output=True, text=True)
        if run.returncode:
            print(run.stdout + run.stderr)
            raise SystemExit(run.returncode)
        print(run.stdout, end='')
        return output.read_bytes()

    clear, green, integer = (0, 0, 0, 255), (0, 255, 0, 255), (1, 2, 3, 1)
    compare(readback('-', 'color'), clear, 0, 'refused')
    compare(readback(single, 'color'), green, 1, 'single MRT0')
    valid = readback(mixed, 'mixed')
    compare(valid[:256], green, 1, 'mixed MRT0')
    compare(valid[256:], integer, 0, 'mixed MRT1')
    # Substitute the one-export placeholder into the two-target draw.
    mutant = readback(single, 'mixed')
    compare(mutant[:256], green, 1, 'mutant MRT0')
    compare(mutant[256:], integer, 0, 'mutant MRT1', expected_mismatches=64)
    print('64 pixels match each CPU reference. The MRT0-only mutant fails all 64 MRT1 comparisons.')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs, shaders = check_requests(directory)
        print('placeholder request, cache, export signature, and refusal checks passed')
        if len(sys.argv) == 4 and sys.argv[1] == 'draw':
            draw_checks(directory, vs, shaders, *sys.argv[2:])
        elif len(sys.argv) != 1:
            raise SystemExit('usage: startup_gfx2hlsl_placeholder_check.py [draw <dxc> <offscreen_check>]')
