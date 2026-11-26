"""Read typed buffer views the title created through a translated buffer_load_format_xyzw, bit for bit.

usage: startup_formatted_view_draw_check.py <startup_stream_consumer> <warp|hardware> <fxc>

A source-built gfx1013 pixel stage loads its descriptor from table entry 0, reads element 1 with
buffer_load_format_xyzw (idxen) and exports the four registers to an R32G32B32A32_FLOAT target. The view behind the
entry is made by tests/startup_address_srv.cpp (draw_formatted_view) through the title-side CreateShaderResourceView,
so the draw has no hardware descriptor word 3: it builds FORMAT and DST_SEL from the view's DXGI format and component
mapping (src/startup_stubs/buffer_format_word3.h). One translation serves every case; each draw is compared as four
32-bit words. Views the translation cannot decode refuse the draw with one logged reason, and the
target keeps its clear value. A translation whose W selector reads the Z field must fail.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from startup_stream_check import container, gfx2hlsl
from startup_stream_draw_check import vertex, vop1

def pixel(element):
    return container(0, [
        0xf4080304, 0xfa000000,  # s_load_dwordx4 s[12:15], s[8:9], 0x0
        0xbf8cc07f,              # s_waitcnt lgkmcnt(0)
        vop1(1, 4, 128 + element),
        0xe00c2000, 0x80030004,  # buffer_load_format_xyzw v[0:3], v4, s[12:15], 0 idxen
        0xbf8c3f70,              # s_waitcnt vmcnt(0)
        0xf800180f, 0x03020100,  # exp mrt0 v0, v1, v2, v3 done vm
        0xbf810000])             # s_endpgm


TARGET = 2  # DXGI_FORMAT_R32G32B32A32_FLOAT
IDENTITY = 0x1688
F = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]  # noqa: E731
ONE, CLEAR = F(1.0), (0, 0, F(1.0), F(1.0))
SENTINELS = (1.5, -2.25, 3.125, 0.0625)  # distinct per channel; exact in binary32 and binary16
HALVES = (0x3e00, 0xc080, 0x4240, 0x2c00)  # the same four values in binary16


def mapping(*sources):
    """D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING: 0-3 channels, 4 forced 0, 5 forced 1."""
    return sum(s << 3 * i for i, s in enumerate(sources)) | 1 << 12


def data(offset, payload):
    raw = bytearray(64)
    raw[offset:offset + len(payload)] = payload
    return bytes(raw)


# (label, DXGI format, mapping, raw view, 64 buffer bytes, expected words or a refusal reason). Element 1 sits at
# byte 16 (16-byte elements), 8 (8-byte) or 4 (4-byte).
CASES = (
    ('R32G32B32A32_FLOAT', 2, IDENTITY, 0, data(16, struct.pack('<4f', *SENTINELS)), tuple(map(F, SENTINELS))),
    ('R16G16B16A16_FLOAT', 10, IDENTITY, 0, data(8, struct.pack('<4H', *HALVES)), tuple(map(F, SENTINELS))),
    ('R16G16_FLOAT, missing z and w', 34, IDENTITY, 0, data(4, struct.pack('<2H', *HALVES[:2])),
     (F(1.5), F(-2.25), 0, ONE)),
    ('R16G16_SNORM, -32768 and 0', 37, IDENTITY, 0, data(4, struct.pack('<2H', 0x8000, 0)), (F(-1.0), 0, 0, ONE)),
    ('R32G32B32A32_FLOAT as (w, 0, x, 1)', 2, mapping(3, 4, 0, 5), 0, data(16, struct.pack('<4f', *SENTINELS)),
     (F(0.0625), 0, F(1.5), ONE)),
    ('R8G8B8A8_UNORM as (z, 1, 0, w)', 28, mapping(2, 5, 4, 3), 0, data(4, bytes(4)), (0, ONE, 0, 0)),
    ('R32_UINT', 42, IDENTITY, 0, data(4, b'\x2a\0\0\0'), 'view format has no decoded buffer FORMAT'),
    ('raw view', 39, IDENTITY, 1, data(16, struct.pack('<4f', *SENTINELS)), 'not a typed buffer view'),
)


def spec(case):
    _, fmt, mapping_word, raw, payload, _ = case
    return '%d:%x:%d:%s' % (fmt, mapping_word, raw, payload.hex())


def check(work, consumer, backend, fxc):
    cache = work / 'cache'
    cache.mkdir()
    vs, ps, root, output = (work / name for name in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(1))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache), X2W_FXC=fxc, X2W_TEST_FORMATTED_VIEW=spec(CASES[0]))
    args = [consumer, vs, ps, root, backend, output, str(TARGET)]

    def draw(case, source_env=None):
        result = subprocess.run(args, env=dict(source_env or env, X2W_TEST_FORMATTED_VIEW=spec(case)),
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0, (result.returncode, result.stderr[-1500:])
        assert not re.search(r'x2w-d3d12-debug.*sev=[012]\b', result.stderr), result.stderr[-3000:]
        return struct.unpack('<4I', output.read_bytes()), result.stderr + result.stdout

    # The first draw publishes the base request and the second its R32G32B32A32_FLOAT target variant.
    for _ in range(2):
        draw(CASES[0])
        subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', cache], env=env, check=True)
    manifests = [p for p in cache.glob('*.txt') if 'descword3' in p.read_text()]
    assert manifests and all('srv 0 buf table 0 0' in p.read_text() for p in manifests), [p.read_text() for p in manifests]
    for case in CASES:
        label, expected = case[0], case[5]
        words, log = draw(case)
        refusals = re.findall(r'srv 0 not ready: (.+)', log)
        if isinstance(expected, str):
            assert '-> drawn;' not in log and refusals == [expected], (label, refusals)
            assert words == CLEAR, (label, [hex(w) for w in words])
        else:
            assert '-> drawn;' in log and not refusals, (label, refusals, log[-2000:])
            assert words == expected, (label, [hex(w) for w in words], [hex(w) for w in expected])
        print('%s: %s' % (label, 'refused and logged' if isinstance(expected, str) else 'bit-exact'))

    if backend == 'warp':
        readiness_draws(work, cache, ps, draw, env, fxc)

    # Negative control: the same translation with the W selector read from the Z field must fail the first case.
    ps.write_bytes(pixel(1))
    variant = next(p for p in manifests if 'rtv1' in p.read_text())
    hlsl = variant.with_suffix('.ps.hlsl')
    source = hlsl.read_text()
    selector = re.search(r'\(\((ud\d+) >> 9\) & 7u\)', source)
    assert selector, 'no W selector in the translation'
    hlsl.write_text(source.replace(selector.group(0), '((%s >> 6) & 7u)' % selector.group(1)))
    subprocess.run([fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo', variant.with_suffix('.ps.cso'), hlsl],
                   check=True, capture_output=True)
    words, _ = draw(CASES[0])
    assert words != CASES[0][5] and words[3] == CASES[0][5][2], [hex(w) for w in words]
    print('control: a W selector read from the Z field fails (w = z)')
    print('%s: %d typed-view cases through the translated buffer_load_format_xyzw' % (backend, len(CASES)))
    return 0


CAUSES = (
    'translation_pending', 'translation_refused', 'translation_failed_to_load',
    'd16_view_not_ready', 'formatted_buffer_not_ready',
)

# A fixture shader that reads the exact-depth resource declared as SRV 1.
D16_PIXEL = """\
Texture2D<float> depth_view : register(t1);
float4 main() : SV_Target {
    return depth_view.Load(int3(0, 0, 0));
}
"""


def check_draw_report(log, key, expected_cause, resource, availability):
    cause_line, = re.findall(r'^x2w-render not drawn by cause:(.*)$', log, re.M)
    expected_counts = {expected_cause: 1} if expected_cause else {}
    counts = {name: int(count) for name, count in re.findall(r'([a-z0-9_]+) ([0-9]+),', cause_line)}
    assert counts == expected_counts, (expected_cause, counts)
    expected_key_line = [(key, '1', f'{expected_cause} 1')] if expected_cause else []
    key_lines = re.findall(r'^x2w-render not drawn key ([0-9a-f]{16}): ([0-9]+) \((.*)\)$', log, re.M)
    assert key_lines == expected_key_line, (expected_key_line, key_lines)
    resource_lines = re.findall(r'^x2w-draw ([0-9a-f]{16}) srv ([0-9]+) not ready: (.*)$', log, re.M)
    assert resource_lines == ([(key, *resource)] if resource else []), resource_lines
    ready, pending, refused, failed = availability
    totals = f'pipelines translated {ready} / pending {pending} / refused {refused} / failed to load {failed}'
    assert totals + '\n' in log, (totals, log[-1500:])
    legacy_count, = re.findall(r'^untranslated draws (\d+)$', log, re.M)
    return sum(counts.values()), int(legacy_count)


def readiness_draws(work, cache, ps, draw, env, fxc):
    report_env = dict(env, X2W_TEST_DRAW_CAUSE_REPORT='1')
    ready_variant, = cache.glob('req_*.rtv')
    words, log = draw(CASES[0], report_env)
    assert words == CASES[0][5], words
    split_total, legacy_total = check_draw_report(log, ready_variant.stem[4:], None, None, (2, 0, 0, 0))

    # Each case gets a different source key and a fresh consumer. The target is always one pixel.
    for element, cause in zip((0, 2, 3, 4, 5), CAUSES):
        ps.write_bytes(pixel(element))
        case_cache = work / cause
        case_cache.mkdir()
        case_env = dict(report_env, X2W_SHADER_CACHE=str(case_cache))
        words, log = draw(CASES[0], case_env)
        request, = case_cache.glob('req_*.rtv')
        key = request.stem[4:]
        resource = None
        availability = (0, 2, 0, 0)
        if cause != 'translation_pending':
            if cause == 'translation_refused':
                gfx2hlsl.write_failure(str(case_cache / f'{key}.fail'), 'synthetic unsupported operation')
                availability = (0, 1, 1, 0)
            else:
                subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', case_cache],
                               env=case_env, check=True, timeout=30)
                manifest = case_cache / f'{key}.txt'
                assert 'srv 0 buf table 0 0' in manifest.read_text(), manifest
                availability = (2, 0, 0, 0)
                if cause == 'translation_failed_to_load':
                    manifest.with_suffix('.ps.cso').unlink()
                    availability = (1, 0, 0, 1)
                elif cause == 'd16_view_not_ready':
                    manifest.write_text(manifest.read_text() + 'srv 1 tex2d table 15 0\nd16srv 1 exact\n')
                    source = case_cache / 'depth.ps.hlsl'
                    source.write_text(D16_PIXEL, encoding='ascii')
                    subprocess.run([fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo',
                                    manifest.with_suffix('.ps.cso'), source],
                                   check=True, capture_output=True, timeout=30)
                    resource = ('1', 'unreadable table entry')
                else:
                    resource = ('0', CASES[-2][5])
            case = CASES[-2] if cause == 'formatted_buffer_not_ready' else CASES[0]
            words, log = draw(case, case_env)
        assert words == CLEAR and '-> drawn;' not in log, (cause, words)
        split_count, legacy_count = check_draw_report(log, key, cause, resource, availability)
        split_total += split_count
        legacy_total += legacy_count
        print(f'warp: {cause} counted once, target stayed clear')
        if cause == 'translation_failed_to_load':
            # A compiler error publishes a sidecar without publishing the manifest.
            manifest.unlink()
            gfx2hlsl.write_failure(str(case_cache / f'{key}.fail'), 'fxc ps: synthetic compiler failure')
            compiler_words, compiler_log = draw(case, case_env)
            assert compiler_words == CLEAR and '-> drawn;' not in compiler_log, compiler_words
            assert check_draw_report(compiler_log, key, cause, None, availability) == (1, 1)

    # These draws reach the same early gate as the old single bucket. Its population is
    # independently counted by the caller's untranslated-draw counter on this fixture.
    assert split_total == legacy_total == 5, (split_total, legacy_total)
    print('warp: the five readiness buckets account for all five untranslated draws')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    if not Path(sys.argv[3]).is_file():
        print('fxc unavailable')
        sys.exit(77)
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
