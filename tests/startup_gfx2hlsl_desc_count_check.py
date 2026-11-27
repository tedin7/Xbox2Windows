"""Translate a synthetic table descriptor count and check a WARP pixel against a CPU reference."""

import json
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_gfx2hlsl_refusals_check import cli, put_request
from startup_gfx2hlsl_sload_s104_check import TABLE_ROOT
from startup_stream_draw_check import container, gfx2hlsl, vertex, vop1, vop2

TARGET = 2  # DXGI_FORMAT_R32G32B32A32_FLOAT
COUNTS = (3, 11)
COLOR = (2.0, -0.5, 1.0, 0.5)
COUNT_BINDING = 'udsrc 3 tablenumrecords 0 32'


def pixel(word=2, offset=32):
    # Load the second buffer record, convert its count to float and scale an RGBA constant.
    return container(0, [
        0xf4080000 | (52 << 6) | 4, 0xfa000000 | offset,  # s_load_dwordx4 s[52:55], s[8:9], offset
        0xbf8cc07f,                             # s_waitcnt lgkmcnt(0)
        vop1(6, 6, 52 + word),                  # v_cvt_f32_u32 v6, s54
        *[vop2(8, out, inline, 6) for out, inline in enumerate((244, 241, 242, 240))],
        0xf800180f, 0x03020100, 0xbf810000,      # exp mrt0; s_endpgm
    ])


def diagnostics(cache, shader):
    cache.mkdir()
    put_request(cache, vertex(), shader, TABLE_ROOT)
    before = {p.name: p.read_bytes() for p in cache.iterdir()}
    result = json.loads(cli(cache, '--json'))['per_request']
    assert result['examined'] == 1
    assert {p.name: p.read_bytes() for p in cache.iterdir()} == before
    return result


def controls(work):
    report = diagnostics(work / 'count', pixel())
    assert report['no_refusal'] == 1, report
    for label, shader in [('word%d' % word, pixel(word)) for word in (0, 1, 3)] + [
        ('unaligned', pixel(offset=33)),
    ]:
        report = diagnostics(work / label, shader)
        assert [(row['reason'], row['scope']) for row in report['reasons']] == [
            ('numeric use of sN holding desc', 'ps')], report

    key = gfx2hlsl.pipeline_key(vertex(), pixel(), TABLE_ROOT)
    kinds, root, _, _, _, _, _ = gfx2hlsl.load_request(str(work / 'count'), key)
    pipe = gfx2hlsl.Pipeline(kinds['vs'], kinds['ps'], root, rtv_formats=(TARGET,))
    assert 'float(ud3)' in pipe.hlsl_ps()
    assert COUNT_BINDING in pipe.manifest().splitlines()
    assert not pipe.resources
    assert gfx2hlsl.lifted_family('numeric use of s54 holding desc') == 'table-record-count'


def expected(count):
    return struct.pack('<4f', *(channel * count for channel in COLOR))


def check_draw(work, consumer, fxc):
    cache = work / 'cache'
    cache.mkdir()
    vs, ps, root, output = (work / name for name in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel())
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1',
               X2W_D3D12_RENDER='1', X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache), X2W_FXC=fxc)
    args = [consumer, vs, ps, root, 'warp', output, str(TARGET)]

    def draw(count, *, require_translated=True):
        result = subprocess.run(args, env=dict(env, X2W_TEST_DESCRIPTOR_COUNT=str(count)),
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0, (result.returncode, result.stderr[-3000:])
        assert not re.search(r'x2w-d3d12-debug.*sev=[012]\b', result.stderr), result.stderr[-3000:]
        if require_translated:
            assert '-> drawn;' in result.stderr, result.stderr[-3000:]
        return output.read_bytes()[:16]

    # The first draw publishes the base request. The second publishes the float target variant.
    for _ in range(2):
        draw(COUNTS[0], require_translated=False)
        subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', cache], env=env, check=True)
    manifests = [p for p in cache.glob('*.txt') if 'rtv1 ' in p.read_text()]
    assert len(manifests) == 1, manifests
    manifest = manifests[0].read_text()
    assert 'x2w-manifest 3\n' in manifest
    assert COUNT_BINDING in manifest.splitlines(), manifest
    assert '\nsrv ' not in manifest, manifest

    for count in COUNTS:
        got = draw(count)
        assert got == expected(count), (count, got.hex(), expected(count).hex())
        print(f'WARP: pixel matches CPU reference for {count} records')

    # Point the binding at the adjacent descriptor, whose count is always one.
    # The draw must reach readback and disagree with both CPU reference colors.
    manifests[0].write_text(manifest.replace(COUNT_BINDING, 'udsrc 3 tablenumrecords 0 0'))
    for count in COUNTS:
        got = draw(count)
        assert got == expected(1) and got != expected(count), (count, got.hex())
    print('WARP control: wrong descriptor offset fails both pixel comparisons')


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) not in (0, 2):
        raise SystemExit('usage: startup_gfx2hlsl_desc_count_check.py [stream_consumer fxc]')
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        controls(work)
        if args:
            if not Path(args[1]).is_file():
                print('fxc unavailable')
                sys.exit(77)
            check_draw(work, *args)
