"""RTV variant requests (req_<key>.rtv) resolved, compiled and cached by compile_dxil --requests (#2980).

usage:
  native_rtv_variant_check.py controls <compile_dxil> <DXC directory>
  native_rtv_variant_check.py e2e <compile_dxil> <startup_stream_consumer> <DXC directory> <warp|hardware>

The pair is source-built: the vertex container of native_startup_request_cache_check.py, which covers the viewport
from the vertex index, and a pixel container that moves four literal words into v0..v3 and exports them to MRT0.
A variant request is the ASCII text "rtv1 <base key> <format> ...", in a file named by the FNV-1a hash of that text
(tools/startup/gfx2hlsl.py request_source). The reader checks it and the base parts it names, the formats type the
pixel stage's color targets, and the variant is published as <variant key>.{vs.cso,ps.cso,txt} with the request as
the manifest's last line, which is what the runtime matches.

controls (no device):
1. One directory holds the base request and four variants of it: R16G16B16A16_UINT, R16G16B16A16_SINT and
   R16G16_UINT, each with an R8G8B8A8_UNORM second target, and R32G32B32A32_FLOAT. With an empty compiler cache
   the tool compiles all five with four parallel workers; the two UINT variants have the same target types and
   share one frontend cache entry. Each variant's manifest is the base manifest plus its request line, the report
   names the target types (u32, i32, f32), the compiled pixel stages declare SV_Target0 as uint, int and float,
   and the base manifest and stages are those of the base alone. A second run hits the frontend and bytecode
   caches for every request and publishes the same bytes. gfx2hlsl.pending_keys then finds nothing left to do.
2. Refusals, each in a fresh directory with only <key>.fail and its reason under the variant key: a forged variant key, base
   content that no longer matches the base key, no format, nine formats, a format above 32 bits, a trailing
   newline, a missing base, the two integer formats without a verified readback (7 and 8) in the exported slot,
   and a forged variant file under the base's own key. Wherever the base request is intact it still compiles.
   Format 7 in a slot the pixel stage does not export to is accepted, and refused again when it is moved to the
   exported slot although the frontend entry of those target types is then cached. A base with an input layout
   publishes its stages without a manifest, as before; its variant is refused for having none, and again when
   the layout no longer matches the base key.
3. A published variant whose base then changes is refused on the next run, and every manifest of the directory
   is withdrawn.
4. A pixel container whose own output signature says uint: the UINT variant compiles, the FLOAT variant is
   refused by name, and the base pair publishes no manifest (as before this change).

e2e: the consumer creates the pair through device slot 86 and draws one pixel into a target of each format, which
makes the stand-in publish the base request and the three variant requests. The tool compiles the directory with
an empty compiler cache and again with the populated one. The consumer then draws each format in a new process:
the stand-in loads <variant key>.txt, creates the desktop pipeline with the bound formats and the target is read
back after the fence. Expected, for the exported words ffff7ffe, 15, eb, 68: UINT 16 clamps to ffff 0015 00eb 0068,
SINT 16 clamps to 8000 0015 00eb 0068, FLOAT 32 keeps all four words bit for bit (the first is a NaN with a
payload), and the untyped RGBA8 base draw stores 00000000 over the blue clear. Exit 77 without the adapter.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from native_startup_request_cache_check import compile_graphics_table_root, put_graphics_request, stream_vertex_container
from native_direct_container_check import patched_pixel, publish, signature, vertex
from startup_stream_draw_check import vop1
import gfx2hlsl as G

WORDS = (0xffff7ffe, 0x15, 0xeb, 0x68)
UINT16, SINT16, FLOAT32, UNORM8, UINT16_RG = 12, 14, 2, 28, 36
# request formats, report types, SV_Target0 type, MRT0 bytes, bytes of the second (RGBA8, cleared) target
VARIANTS = (
    (UINT16, (UINT16, UNORM8), 'u32,f32', 'uint', 'ffff1500eb006800', '0000ffff'),
    (SINT16, (SINT16, UNORM8), 'i32,f32', 'int', '00801500eb006800', '0000ffff'),
    (FLOAT32, (FLOAT32,), 'f32', 'float', struct.pack('<4I', *WORDS).hex(), ''),
)


def pixel_stage(words=WORDS, parts=()):
    prefix = []
    for register, value in enumerate(words):
        prefix += [vop1(1, register, 255), value]
    return patched_pixel(len(prefix), prefix, export=(0xf800180f, 0x03020100), input_signature=parts)


def put_variant(cache, base, formats, name_key=None, text=None):
    line = text if text is not None else 'rtv1 ' + ' '.join(map(str, (base, *formats)))
    key = G.fnv(line.encode('ascii')) if name_key is None else name_key
    (cache / f'req_{key:016x}.rtv').write_bytes(line.encode('ascii'))
    return key, line


def run_tool(cli, sdk, frontend, cache, jobs=()):
    return subprocess.run([cli, sdk, frontend, *jobs, 'gfx1013-container-pair', '--requests', cache, cache],
                          capture_output=True, text=True)


def target_types(sdk, blob):
    listing = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', blob], capture_output=True, text=True, check=True).stdout
    found = re.findall(r'^\s*;\s+SV_Target\s+(\d+)\s+\S+\s+\d+\s+\S+\s+(float|uint|int)\b', listing, re.M)
    assert found, listing[:2000]
    return dict(found)


def published(cache, key):
    return sorted(p.name for p in cache.glob(f'{key:016x}.*'))


def only_failure(cache, key, reason):
    """A refused request leaves <key>.fail with its reason, and no stage or manifest."""
    assert published(cache, key) == [f'{key:016x}.fail'], published(cache, key)
    assert reason in (cache / f'{key:016x}.fail').read_text(encoding='ascii'), (cache / f'{key:016x}.fail').read_text()


def check_published(cli, sdk, frontend, cache, base, variants, jobs=(), shared=0):
    """variants: (key, line, report types, SV_Target0 type). Cold run, checks, warm run. Returns the cold report.

    shared: how many variants have the target types of another one. Such a pair has one frontend cache entry, so
    with parallel jobs the later one may already hit it in the cold run."""
    cold = run_tool(cli, sdk, frontend, cache, jobs)
    count = len(variants) + 1
    assert cold.returncode == 0 and f'requests={count} compiled={count} refused=0' in cold.stdout, (cold.stdout, cold.stderr)
    # Every pair is linked anew. The stage bytecode cache already hits within this run: the requests share one vertex
    # stage, and a pixel stage with float targets is the base's.
    assert count - shared <= cold.stdout.count('graphics_pair frontend_cache_hit=0') <= count, cold.stdout
    # One vertex stage and one pixel stage per distinct set of target types; parallel workers can compile one twice.
    stages = 1 + len({types for _, _, types, _ in variants} | {'f32'})
    compiled = cold.stdout.count('cold_compile_and_publish_ms=')
    assert compiled >= stages if jobs else compiled == stages, cold.stdout
    base_manifest = (cache / f'{base:016x}.txt').read_text(encoding='ascii')
    assert 'rtv1' not in base_manifest and base_manifest.startswith('x2w-manifest 1\n'), base_manifest
    assert target_types(sdk, cache / f'{base:016x}.ps.cso') == {'0': 'float'}
    for key, line, types, target in variants:
        assert published(cache, key) == [f'{key:016x}.ps.cso', f'{key:016x}.txt', f'{key:016x}.vs.cso'], published(cache, key)
        manifest = cache / f'{key:016x}.txt'
        assert manifest.read_text(encoding='ascii') == base_manifest + line + '\n', manifest.read_text(encoding='ascii')
        assert f'rtv_variant {line} types={types}\n' in cold.stdout, cold.stdout
        assert target_types(sdk, cache / f'{key:016x}.ps.cso') == {'0': target}, (line, target)
        assert manifest.stat().st_mtime_ns >= max((cache / f'{key:016x}{e}').stat().st_mtime_ns for e in ('.vs.cso', '.ps.cso'))
    before = {p.name: p.read_bytes() for p in cache.iterdir() if not p.name.startswith('req_')}
    warm = run_tool(cli, sdk, frontend, cache, jobs)
    assert warm.returncode == 0 and f'requests={count} compiled={count} refused=0' in warm.stdout, (warm.stdout, warm.stderr)
    assert warm.stdout.count('graphics_pair frontend_cache_hit=1') == count, warm.stdout
    assert warm.stdout.count('cache_hit_ms=') == 2 * count and 'cold_compile_and_publish_ms=' not in warm.stdout, warm.stdout
    assert {p.name: p.read_bytes() for p in cache.iterdir() if not p.name.startswith('req_')} == before
    assert G.pending_keys(str(cache), skip_failed=False) == [], 'published entries are not recognised as complete'
    return cold


def check_controls(cli, sdk):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        frontend = work / 'frontend'
        root = compile_graphics_table_root(Path(sdk), work)
        vs, ps = stream_vertex_container(), pixel_stage()

        cache = work / 'cache'
        cache.mkdir()
        base = put_graphics_request(cache, vs, ps, root)
        variants = [put_variant(cache, base, formats) + (types, target) for _, formats, types, target, _, _ in VARIANTS]
        # R16G16_UINT has the target types of the first variant, so the two share a frontend cache entry. Four
        # workers compile the five requests side by side.
        variants.append(put_variant(cache, base, (UINT16_RG, UNORM8)) + ('u32,f32', 'uint'))
        check_published(cli, sdk, frontend, cache, base, variants, jobs=('--jobs', '4'), shared=1)

        # 3. The base changes under a published variant: the next run refuses it and withdraws its manifest.
        (cache / f'req_{base:016x}.b').write_bytes(pixel_stage((1, 2, 3, 4)))
        stale = run_tool(cli, sdk, frontend, cache)
        assert stale.returncode != 0 and 'requests=5 compiled=0 refused=5' in stale.stdout, (stale.stdout, stale.stderr)
        assert stale.stderr.count('RTV variant source: graphics request key mismatch') == len(variants), stale.stderr
        assert not list(cache.glob('*.txt')), 'a request with a changed base kept a loadable manifest'

        refusals = 0

        def refuse(label, expected, setup, base_intact=True):
            nonlocal refusals
            folder = work / label
            folder.mkdir()
            base_key = put_graphics_request(folder, vs, ps, root)
            key = setup(folder, base_key)
            result = run_tool(cli, sdk, frontend, folder)
            assert result.returncode != 0 and expected in result.stderr, (label, result.returncode, result.stderr)
            only_failure(folder, key, expected)
            if base_intact:
                assert 'compiled=1 refused=1' in result.stdout, (label, result.stdout)
                assert (folder / f'{base_key:016x}.txt').is_file(), f'{label}: the base pair no longer compiles'
            refusals += 1

        def remove_base(folder, base_key):
            for extension in ('.a', '.b', '.root'):
                (folder / f'req_{base_key:016x}{extension}').unlink()
            return put_variant(folder, base_key, (UINT16, UNORM8))[0]

        def change_base(folder, base_key):
            key = put_variant(folder, base_key, (UINT16, UNORM8))[0]
            (folder / f'req_{base_key:016x}.b').write_bytes(pixel_stage((1, 2, 3, 4)))
            return key

        invalid = 'invalid RTV variant request'
        refuse('forged-key', invalid,
               lambda d, b: put_variant(d, b, (UINT16, UNORM8), name_key=G.fnv(f'rtv1 {b} 12 28'.encode()) ^ 1)[0])
        refuse('base-changed', 'RTV variant source: graphics request key mismatch', change_base, base_intact=False)
        refuse('no-format', invalid, lambda d, b: put_variant(d, b, ())[0])
        refuse('nine-formats', invalid, lambda d, b: put_variant(d, b, (UNORM8,) * 9)[0])
        refuse('wide-format', 'invalid RTV variant values', lambda d, b: put_variant(d, b, (1 << 32,))[0])
        refuse('trailing-newline', invalid, lambda d, b: put_variant(d, b, (), text=f'rtv1 {b} 12 28\n')[0])
        refuse('missing-base', 'RTV variant source: missing or oversized .a', remove_base, base_intact=False)
        refuse('unverified-uint', 'unverified UINT RTV format 7', lambda d, b: put_variant(d, b, (7, UNORM8))[0])
        refuse('unverified-sint', 'unverified SINT RTV format 8', lambda d, b: put_variant(d, b, (8, UNORM8))[0])
        # Both markers under one key are one request, the variant: a forged one refuses that key and nothing else runs
        # under it.
        refuse('forged-base-key', invalid, lambda d, b: put_variant(d, b, (UINT16, UNORM8), name_key=b)[0], base_intact=False)

        # As in gfx2hlsl.py, an unverified format in a slot the pixel stage does not export to is not refused. The
        # frontend entry of these target types is cached by the first request, so the second is refused on a hit.
        unwritten = work / 'unwritten'
        unwritten.mkdir()
        slot_base = put_graphics_request(unwritten, vs, ps, root)
        accepted, line = put_variant(unwritten, slot_base, (UINT16, 7))
        result = run_tool(cli, sdk, frontend, unwritten)
        assert result.returncode == 0 and f'rtv_variant {line} types=u32,u32\n' in result.stdout, (result.stdout, result.stderr)
        assert (unwritten / f'{accepted:016x}.txt').is_file()
        (unwritten / f'req_{accepted:016x}.rtv').unlink()
        rejected, _ = put_variant(unwritten, slot_base, (7, UINT16))
        result = run_tool(cli, sdk, frontend, unwritten)
        assert result.returncode != 0 and 'unverified UINT RTV format 7' in result.stderr, (result.stdout, result.stderr)
        only_failure(unwritten, rejected, 'unverified UINT RTV format 7')
        refusals += 1

        # A base with an input layout: the layout is read and checked with the base key, the base publishes its
        # stages without a manifest (as before), and its variant is refused because it would have none either.
        fetched = work / 'layout'
        layout_base = int(publish(fetched, vertex(True), ps, root, b'ia1 1\nposition 0 2 3 4294967295 0 0\n'), 16)
        layout_variant, _ = put_variant(fetched, layout_base, (UINT16, UNORM8))
        result = run_tool(cli, sdk, frontend, fetched)
        assert result.returncode != 0 and 'requests=2 compiled=1 refused=1' in result.stdout, (result.stdout, result.stderr)
        assert 'RTV variant has no runtime manifest' in result.stderr, result.stderr
        assert published(fetched, layout_base) == [f'{layout_base:016x}.ps.cso', f'{layout_base:016x}.vs.cso']
        only_failure(fetched, layout_variant, 'RTV variant has no runtime manifest')
        (fetched / f'req_{layout_base:016x}.ia').write_bytes(b'ia1 1\nposition 0 2 3 0 0 0\n')
        result = run_tool(cli, sdk, frontend, fetched)
        assert 'RTV variant source: graphics request key mismatch' in result.stderr, result.stderr
        only_failure(fetched, layout_variant, 'RTV variant source: graphics request key mismatch')
        refusals += 2

        # 4. A container whose own signature types the target: the format has to agree with it.
        typed = work / 'signature'
        typed.mkdir()
        uint_ps = pixel_stage(parts=[(b'OSG1', signature(b'SV_Target', system=64, component=1))])
        typed_base = put_graphics_request(typed, vs, uint_ps, root)
        agree, _ = put_variant(typed, typed_base, (UINT16, UNORM8))
        disagree, _ = put_variant(typed, typed_base, (FLOAT32,))
        result = run_tool(cli, sdk, frontend, typed)
        assert result.returncode != 0 and 'requests=3 compiled=2 refused=1' in result.stdout, (result.stdout, result.stderr)
        assert 'RTV variant format disagrees with the pixel signature of target 0' in result.stderr, result.stderr
        assert (typed / f'{agree:016x}.txt').is_file() and target_types(sdk, typed / f'{agree:016x}.ps.cso') == {'0': 'uint'}
        only_failure(typed, disagree, 'RTV variant format disagrees with the pixel signature of target 0')
        assert not (typed / f'{typed_base:016x}.txt').exists(), 'a base pair with an integer target published a manifest'

    print(f'native_rtv_variant controls: the base and 4 variants (uint, int, float SV_Target0) compiled cold and warm '
          f'by 4 workers with identical bytes; {refusals} refusals left only a .fail; a changed base withdrew every manifest; '
          f'a signature that disagrees with the format is refused')
    return 0


def check_e2e(cli, consumer, sdk, backend):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        cache = work / 'cache'
        cache.mkdir()
        vs, ps, root, pixels = (work / name for name in ('vs', 'ps', 'root', 'pixels'))
        vs.write_bytes(stream_vertex_container())
        ps.write_bytes(pixel_stage())
        env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
                   X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache))
        env.pop('X2W_NATIVE_SHADERS', None)

        def draw(fmt=None):
            pixels.unlink(missing_ok=True)
            result = subprocess.run([consumer, vs, ps, root, backend, pixels] + ([str(fmt)] if fmt else []),
                                    env=env, capture_output=True, text=True)
            if result.returncode == 77:
                print(result.stdout + result.stderr[-2000:], end='')
                print(f'native_rtv_variant e2e: SKIP on {backend}: the adapter is not available')
                sys.exit(77)
            assert result.returncode == 0, f'consumer exit {result.returncode}\n{result.stderr[-3000:]}'
            errors = [line for line in result.stderr.splitlines() if 'x2w-d3d12-debug ' in line and (' sev=0 ' in line or ' sev=1 ' in line)]
            assert not errors, errors[0]
            return pixels.read_bytes(), result

        # The stand-in publishes the requests: no translation exists, so nothing is drawn.
        cleared = {}
        for fmt, *_ in VARIANTS:
            cleared[fmt], result = draw(fmt)
            assert '-> drawn;' not in result.stderr and 'no RTV translation' in result.stderr, result.stderr[-3000:]
        print(result.stdout, end='')
        base = G.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())
        variants = []
        for fmt, formats, types, target, _, _ in VARIANTS:
            line = 'rtv1 ' + ' '.join(map(str, (base, *formats)))
            key = G.fnv(line.encode('ascii'))
            assert (cache / f'req_{key:016x}.rtv').read_bytes() == line.encode('ascii'), line
            variants.append((key, line, types, target))
        assert sorted(p.name for p in cache.iterdir()) == sorted(
            [f'req_{base:016x}{e}' for e in ('.a', '.b', '.root')] + [f'req_{key:016x}.rtv' for key, *_ in variants])

        check_published(cli, sdk, work / 'frontend', cache, base, variants)

        for (fmt, _, _, _, first, second), (key, line, _, _) in zip(VARIANTS, variants):
            image, result = draw(fmt)
            log = result.stderr
            assert image.hex() == first + second, f'format {fmt}: {image.hex()} expected {first + second}\n{log[-3000:]}'
            assert image != cleared[fmt], f'format {fmt}: the expected value equals the cleared target'
            targets = 2 if second else 1
            assert f'desktop pipeline for {key:016x}, {targets} targets (format {fmt}) -> 0x00000000' in log, log[-3000:]
            assert f'pipeline {key:016x} topology 0x4 -> drawn;' in log, log[-3000:]
            print(f'{backend} format {fmt} variant {key:016x}: MRT0={first}' + (f' MRT1={second}' if second else ''))
        image, result = draw()
        assert image == (bytes(4) * 4 + bytes((0, 0, 255, 255)) * 4) * 4, image.hex()
        assert f'pipeline {base:016x} topology 0x4 -> drawn;' in result.stderr, result.stderr[-3000:]
        layer = [line for line in result.stderr.splitlines() if 'x2w-d3d12 debug layer: ' in line]
        assert layer, result.stderr[-3000:]
        print('debug layer: ' + ('enabled, no error message' if '0x00000000' in layer[0] else 'unavailable'))

    print(f'native_rtv_variant e2e: PASS on {backend} (3 runtime-published variants of one base compiled cold and warm; '
          f'UINT16, SINT16 and FLOAT32 readbacks match; the base pair draws)')
    return 0


def main(argv):
    if len(argv) == 3 and argv[0] == 'controls':
        return check_controls(*argv[1:])
    if len(argv) == 5 and argv[0] == 'e2e' and argv[4] in ('warp', 'hardware'):
        return check_e2e(*argv[1:])
    raise SystemExit(__doc__)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
