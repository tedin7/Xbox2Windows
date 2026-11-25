"""Check typed gfx1013 colour exports through the wrapped draw and offline cache."""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from startup_stream_check import container, gfx2hlsl
from startup_stream_draw_check import vertex, vop1


FLOAT_WORDS = (1, 2, 0x7fc00001, 0x80000001)
INTEGER_WORD = 0x80000001
# Typed formats whose channels are all 32 bits: DXGI value -> (shader output type, channels).
INTEGER_FORMATS = ((42, 'uint', 1), (43, 'int', 1), (17, 'uint', 2), (18, 'int', 2), (3, 'uint', 4), (4, 'int', 4))
REFUSED_INTEGER_FORMATS = (12, 14, 57, 59, 62, 64)


def pixel(values, unorm=False):
    words = []
    for reg, value in enumerate(values):
        words += [vop1(1, reg, 255), value]
    words += [0xf800000f if unorm else 0xf800180f, 0x03020100]
    if unorm:
        words += [vop1(1, 0, 128), vop1(1, 1, 242), 0xf800181f, 0x01000001]
    return container(0, words + [0xbf810000])


def compile_shader(compiler, kind, stage, source, blob, listing, unoptimized=False):
    profile = stage + ('_5_1' if kind == 'fxc' else '_6_0')
    if kind == 'fxc':
        cmd = [compiler, '/nologo', '/T', profile, '/E', 'main', '/Fo', blob, '/Fc', listing]
        if unoptimized:
            cmd.append('/Od')
    else:
        cmd = [compiler, '-T', profile, '-E', 'main', '-Fo', blob, '-Fc', listing]
    subprocess.run(cmd + [source], check=True)


def signature(listing):
    targets = re.findall(r'^\s*(?://|;)\s+SV_Target\s+(\d+)\s+\S+\s+\d+\s+\S+\s+(float|uint|s?int)\b',
                         listing.read_text(errors='replace'), re.M)
    return {slot: 'int' if kind == 'sint' else kind for slot, kind in targets}


def check(work, consumer, backend, kind, dxc, fxc):
    compiler = fxc if kind == 'fxc' else dxc
    if not Path(compiler).is_file():
        print(f'{kind} compiler unavailable')
        return 77
    cache = work / 'cache'
    cache.mkdir()
    vs, ps, root, output = (work / name for name in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(vertex())
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache), X2W_FXC=fxc if kind == 'fxc' else '')
    args = [consumer, vs, ps, root, backend, output]

    def draw(fmt, word=0, wanted_draw=True):
        result = subprocess.run(args + [str(fmt), str(word)], env=env, capture_output=True, text=True)
        if result.returncode == 77:
            raise SystemExit(77)
        print(result.stdout, end='')
        debug = next((line for line in result.stderr.splitlines() if 'debug layer:' in line), '')
        assert debug, result.stderr
        print(debug)
        assert not re.search(r'x2w-d3d12-debug.*sev=[012]\b', result.stderr), result.stderr
        actual_draw = '-> drawn;' in result.stderr
        assert result.returncode == 0 and actual_draw == wanted_draw, result.stderr
        return output.read_bytes(), result.stderr

    def request(fmt, formats):
        pixels, log = draw(fmt, wanted_draw=False)
        assert 'no RTV translation' in log, log
        base = gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())
        line = 'rtv1 ' + ' '.join(map(str, (base, *formats)))
        key = f'{gfx2hlsl.fnv(line.encode("ascii")):016x}'
        assert (cache / f'req_{key}.rtv').read_text() == line
        subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', cache], env=env, check=True)
        assert (cache / f'{key}.txt').read_text().splitlines()[-1] == line
        return key, pixels

    def compile_pair(key):
        for stage in ('vs', 'ps'):
            compile_shader(compiler, kind, stage, cache / f'{key}.{stage}.hlsl', cache / f'{key}.{stage}.cso',
                           cache / f'{key}.{stage}.asm')
        return signature(cache / f'{key}.ps.asm')

    ps.write_bytes(pixel((INTEGER_WORD, 2, 3, 4), unorm=True))
    variants = {}
    for fmt, expected_type, channels in INTEGER_FORMATS:
        key, cleared = request(fmt, (fmt, 28))
        variants[fmt] = key
        # The harness clears every target to (0, 0, 1, 1); integer channels hold those values as words.
        assert cleared == struct.pack(f'<{channels}I', *(0, 0, 1, 1)[:channels]) + bytes.fromhex('0000ffff'), cleared.hex()
        outputs = compile_pair(key)
        assert outputs == {'0': expected_type, '1': 'float'}, outputs
        manifest = cache / f'{key}.txt'
        manifest.rename(str(manifest) + '.pending')
        env['X2W_TEST_PUBLISH_MANIFEST'] = str(manifest)
        pixels, log = draw(fmt)
        del env['X2W_TEST_PUBLISH_MANIFEST']
        assert Path(str(output) + '.first').read_bytes() == cleared
        assert f'2 targets (format {fmt}) -> 0x00000000' in log, log
        words = (INTEGER_WORD, 2, 3, 4)[:channels]
        assert pixels == struct.pack(f'<{channels}I4B', *words, 255, 0, 0, 255), pixels.hex()
        print(f'{kind} {backend} {expected_type} format {fmt} MRT0={pixels[:4 * channels].hex()}; UNORM MRT1=ff0000ff')
        if fmt == 42:
            for narrow in REFUSED_INTEGER_FORMATS:
                line = f'rtv1 {gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())} {narrow} 28'
                refused = f'{gfx2hlsl.fnv(line.encode("ascii")):016x}'
                marker = cache / f'req_{refused}.rtv'
                marker.write_text(line)
                result = gfx2hlsl.translate_request(cache, int(refused, 16))
                assert re.fullmatch(rf'unsupported: unverified (UINT|SINT) RTV format {narrow}', result), result
                assert not (cache / f'{refused}.txt').exists()
                # Later --pending runs would retry this synthetic request on every valid format.
                marker.unlink()
            print(f'{kind} {backend} narrow integer formats {REFUSED_INTEGER_FORMATS} refused at translation')
            # A pre-existing base translation may lack its original request files.
            base = f'{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes()):016x}'
            compile_pair(base)
            for ext in ('a', 'b', 'root'):
                (cache / f'req_{base}.{ext}').unlink()
    assert len({variants[f] for f, _, _ in INTEGER_FORMATS}) == len(INTEGER_FORMATS)
    # A SINT artifact under the UINT cache key must be refused before PSO creation.
    unsigned = cache / f'{variants[42]}.txt'
    unsigned.write_bytes((cache / f'{variants[43]}.txt').read_bytes())
    pixels, log = draw(42, wanted_draw=False)
    assert 'RTV manifest mismatch' in log and 'desktop pipeline for' not in log, log
    assert pixels.hex() == '000000000000ffff', pixels.hex()
    print(f'{kind} {backend} stale signedness control: refused before PSO creation')

    # Distinguish signed and unsigned saturation from writing the low eight bits.
    ps.write_bytes(pixel((0xffffffff, 0xffffff7f, 0x80, 0x100), unorm=True))
    for fmt, expected_type, rgba in ((30, 'uint', 'ffff80ff'), (32, 'int', 'ff807f7f')):
        key, cleared = request(fmt, (fmt, 28))
        assert cleared == bytes.fromhex('000001010000ffff'), cleared.hex()
        outputs = compile_pair(key)
        assert outputs == {'0': expected_type, '1': 'float'}, outputs
        pixels, _ = draw(fmt)
        assert pixels == bytes.fromhex(rgba + 'ff0000ff'), pixels.hex()
        source = (cache / f'{key}.ps.hlsl').read_text()
        end = '    return o;'
        assert source.count(end) == 1
        trunc = 'o.mrt0 & 255u' if fmt == 30 else 'asint(asuint(o.mrt0) << 24u) >> 24u'
        wrong = cache / f'{key}.low-byte.hlsl'
        wrong.write_text(source.replace(end, f'    o.mrt0 = {trunc};\n{end}'))
        compile_shader(compiler, kind, 'ps', wrong, cache / f'{key}.ps.cso', cache / f'{key}.low-byte.asm')
        control, _ = draw(fmt)
        assert control == bytes.fromhex('ff7f8000ff0000ff'), control.hex()
        print(f'{kind} {backend} RGBA8 {expected_type} MRT0={rgba}; low-byte control={control[:4].hex()}')

    ps.write_bytes(pixel(FLOAT_WORDS))
    key, _ = request(2, (2,))
    assert compile_pair(key) == {'0': 'float'}
    exact = struct.pack('<4I', *FLOAT_WORDS)
    pixels, _ = draw(2)
    assert pixels == exact, pixels.hex()
    manifest = cache / f'{key}.txt'
    original = manifest.read_text()
    assert re.search(r'udsrc \d+ mrtbits', original)
    manifest.write_text(re.sub(r'mrtbits', 'rootconst 0 0', original))
    word = 0x00800000
    pixels, _ = draw(2, word)
    assert pixels == struct.pack('<4I', *(value ^ word for value in FLOAT_WORDS)), pixels.hex()
    manifest.write_text(original)
    print(f'{kind} {backend} optimized FLOAT exact {exact.hex()}; nonzero constant readback {pixels.hex()}')

    base = f'{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes()):016x}'
    ps_source, ps_blob = cache / f'{base}.ps.hlsl', cache / f'{key}.ps.cso'
    compile_shader(compiler, kind, 'ps', ps_source, ps_blob, cache / f'{key}.folded.asm')
    pixels, _ = draw(2)
    if kind == 'fxc':
        folded = struct.pack('<4I', 0, 0, 0x7fc00001, 0)
        assert pixels == folded, pixels.hex()
        compile_shader(compiler, kind, 'ps', ps_source, ps_blob, cache / f'{key}.unoptimized.asm', unoptimized=True)
        pixels, _ = draw(2)
        assert pixels == exact, pixels.hex()
        print(f'{kind} {backend} folding control lost three words; /Od kept all four')
    else:
        assert pixels == exact, pixels.hex()
        print(f'{kind} {backend} constant baseline kept all four words')

    # Narrower binary32 targets store the leading words only; a NaN payload and a negative denormal must survive.
    narrow_words = (0x7fc00001, 0x80000001, 1, 2)
    ps.write_bytes(pixel(narrow_words))
    for fmt, channels in ((16, 2), (41, 1)):
        key, _ = request(fmt, (fmt,))
        assert compile_pair(key) == {'0': 'float'}
        pixels, _ = draw(fmt)
        assert pixels == struct.pack(f'<{channels}I', *narrow_words[:channels]), pixels.hex()
        print(f'{kind} {backend} FLOAT format {fmt} exact {pixels.hex()}')
    return 0


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        sys.exit(check(Path(directory), *sys.argv[1:]))
