"""Check typed gfx1013 colour exports through the wrapped draw and offline cache."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from startup_stream_check import container, gfx2hlsl
from startup_stream_draw_check import vertex, vop1


FLOAT_WORDS = (1, 2, 0x7fc00001, 0x80000001)
INTEGER_WORD = 0x80000001
# Typed formats whose channels are all 32 bits: DXGI value -> (shader output type, channels).
INTEGER_FORMATS = ((42, 'uint', 1), (43, 'int', 1), (17, 'uint', 2), (18, 'int', 2), (3, 'uint', 4), (4, 'int', 4))
REFUSED_INTEGER_FORMATS = (7, 8)  # Three 32-bit channels have no verified RTV readback.
# MRT0 bytes for export words (ffff7ffe, 21, 235, 104), ordered R to A in little-endian storage.
NARROW_INTEGER_FORMATS = (
    (12, 'uint', 'ffff210035020401'), (14, 'int', '0080210035020401'),  # RGBA16
    (25, 'uint', 'ff8750e3'),  # RGB10A2: R=1023, G=33, B=565, A=3
    (36, 'uint', 'ffff2100'), (38, 'int', '00802100'),  # RG16
    (50, 'uint', 'ff21'), (52, 'int', '8021'),  # RG8
    (57, 'uint', 'ffff'), (59, 'int', '0080'),  # R16
    (62, 'uint', 'ff'), (64, 'int', '80'),  # R8
)


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
    vs, ps = work / 'vs', work / 'ps'
    vs.write_bytes(vertex())

    def session(name):
        """Private cache, root and output files, so sessions can run side by side."""
        cache = work / name
        cache.mkdir()
        root, output = cache / 'root', cache / 'pixels'
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

        def queue_request(fmt, formats):
            pixels, log = draw(fmt, wanted_draw=False)
            assert 'no RTV translation' in log, log
            base = gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())
            line = 'rtv1 ' + ' '.join(map(str, (base, *formats)))
            key = f'{gfx2hlsl.fnv(line.encode("ascii")):016x}'
            assert (cache / f'req_{key}.rtv').read_text() == line
            return key, line, pixels

        def request(fmt, formats):
            key, line, pixels = queue_request(fmt, formats)
            subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', cache], env=env, check=True)
            assert (cache / f'{key}.txt').read_text().splitlines()[-1] == line
            return key, pixels

        def compile_pair(key):
            for stage in ('vs', 'ps'):
                compile_shader(compiler, kind, stage, cache / f'{key}.{stage}.hlsl', cache / f'{key}.{stage}.cso',
                               cache / f'{key}.{stage}.asm')
            return signature(cache / f'{key}.ps.asm')

        return SimpleNamespace(cache=cache, root=root, output=output, env=env, draw=draw,
                               queue_request=queue_request, request=request, compile_pair=compile_pair)

    main = session('cache')
    cache, root, output, env = main.cache, main.root, main.output, main.env
    draw, queue_request, request, compile_pair = main.draw, main.queue_request, main.request, main.compile_pair

    ps.write_bytes(pixel((INTEGER_WORD, 2, 3, 4), unorm=True))
    integer_sessions = {fmt: session(f'int{fmt}') for fmt, _, _ in INTEGER_FORMATS}

    def integer_format(fmt, expected_type, channels):
        s = integer_sessions[fmt]
        key, cleared = s.request(fmt, (fmt, 28))
        # The harness clears every target to (0, 0, 1, 1); integer channels hold those values as words.
        assert cleared == struct.pack(f'<{channels}I', *(0, 0, 1, 1)[:channels]) + bytes.fromhex('0000ffff'), cleared.hex()
        outputs = s.compile_pair(key)
        assert outputs == {'0': expected_type, '1': 'float'}, outputs
        manifest = s.cache / f'{key}.txt'
        manifest.rename(str(manifest) + '.pending')
        s.env['X2W_TEST_PUBLISH_MANIFEST'] = str(manifest)
        pixels, log = s.draw(fmt)
        del s.env['X2W_TEST_PUBLISH_MANIFEST']
        assert Path(str(s.output) + '.first').read_bytes() == cleared
        assert f'2 targets (format {fmt}) -> 0x00000000' in log, log
        words = (INTEGER_WORD, 2, 3, 4)[:channels]
        assert pixels == struct.pack(f'<{channels}I4B', *words, 255, 0, 0, 255), pixels.hex()
        print(f'{kind} {backend} {expected_type} format {fmt} MRT0={pixels[:4 * channels].hex()}; UNORM MRT1=ff0000ff')
        if fmt == 42:
            for unverified in REFUSED_INTEGER_FORMATS:
                line = f'rtv1 {gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), s.root.read_bytes())} {unverified} 28'
                refused = f'{gfx2hlsl.fnv(line.encode("ascii")):016x}'
                marker = s.cache / f'req_{refused}.rtv'
                marker.write_text(line)
                result = gfx2hlsl.translate_request(s.cache, int(refused, 16))
                assert re.fullmatch(rf'unsupported: unverified (UINT|SINT) RTV format {unverified}', result), result
                assert not (s.cache / f'{refused}.txt').exists()
                # Later --pending runs would retry this synthetic request on every valid format.
                marker.unlink()
            print(f'{kind} {backend} three-channel integer formats {REFUSED_INTEGER_FORMATS} refused at translation')
            # A pre-existing base translation may lack its original request files.
            base = f'{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), s.root.read_bytes()):016x}'
            s.compile_pair(base)
            for ext in ('a', 'b', 'root'):
                (s.cache / f'req_{base}.{ext}').unlink()
        return key

    # Each format needs two device creations and two compiles; run them side by side in private caches.
    with ThreadPoolExecutor(max_workers=len(INTEGER_FORMATS)) as pool:
        futures = [pool.submit(integer_format, *entry) for entry in INTEGER_FORMATS]
        variants = {entry[0]: future.result() for entry, future in zip(INTEGER_FORMATS, futures)}
    assert len({variants[f] for f, _, _ in INTEGER_FORMATS}) == len(INTEGER_FORMATS)
    # This cache still has the compiled base and UINT pair. Substitute only the signed manifest.
    # On a base hit the runtime must also republish the three source blobs removed above.
    uint = integer_sessions[42]
    unsigned = uint.cache / f'{variants[42]}.txt'
    unsigned.write_bytes((integer_sessions[43].cache / f'{variants[43]}.txt').read_bytes())
    pixels, log = uint.draw(42, wanted_draw=False)
    assert 'RTV manifest mismatch' in log and 'desktop pipeline for' not in log, log
    assert pixels.hex() == '000000000000ffff', pixels.hex()
    base = f'{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), uint.root.read_bytes()):016x}'
    for ext, source in (('a', vs), ('b', ps), ('root', uint.root)):
        assert (uint.cache / f'req_{base}.{ext}').read_bytes() == source.read_bytes(), ext
    print(f'{kind} {backend} stale signedness control: refused before PSO creation')

    def low_bits_control(key, fmt, trunc, expected):
        source = (cache / f'{key}.ps.hlsl').read_text()
        end = '    return o;'
        assert source.count(end) == 1
        wrong = cache / f'{key}.low-bits.hlsl'
        wrong.write_text(source.replace(end, f'    o.mrt0 = {trunc};\n{end}'))
        compile_shader(compiler, kind, 'ps', wrong, cache / f'{key}.ps.cso', cache / f'{key}.low-bits.asm')
        control, _ = draw(fmt)
        assert control == bytes.fromhex(expected + 'ff0000ff'), control.hex()
        return control[:-4].hex()

    # Distinguish signed and unsigned saturation from writing the low eight bits.
    ps.write_bytes(pixel((0xffffffff, 0xffffff7f, 0x80, 0x100), unorm=True))
    for fmt, expected_type, rgba in ((30, 'uint', 'ffff80ff'), (32, 'int', 'ff807f7f')):
        key, cleared = request(fmt, (fmt, 28))
        assert cleared == bytes.fromhex('000001010000ffff'), cleared.hex()
        outputs = compile_pair(key)
        assert outputs == {'0': expected_type, '1': 'float'}, outputs
        pixels, _ = draw(fmt)
        assert pixels == bytes.fromhex(rgba + 'ff0000ff'), pixels.hex()
        trunc = 'o.mrt0 & 255u' if fmt == 30 else 'asint(asuint(o.mrt0) << 24u) >> 24u'
        control = low_bits_control(key, fmt, trunc, 'ff7f8000')
        print(f'{kind} {backend} RGBA8 {expected_type} MRT0={rgba}; low-byte control={control}')

    # All views share this shader, so one pending scan can publish their variants.
    ps.write_bytes(pixel((0xffff7ffe, 0x21, 0x235, 0x104), unorm=True))
    queued = {fmt: queue_request(fmt, (fmt, 28)) for fmt, _, _ in NARROW_INTEGER_FORMATS}
    subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', cache], env=env, check=True)
    for fmt, expected_type, rgba in NARROW_INTEGER_FORMATS:
        key, record, _ = queued[fmt]
        assert (cache / f'{key}.txt').read_text().splitlines()[-1] == record
        assert compile_pair(key) == {'0': expected_type, '1': 'float'}
        pixels, _ = draw(fmt)
        assert pixels == bytes.fromhex(rgba + 'ff0000ff'), (fmt, pixels.hex())
        if fmt == 14:
            low_bits_control(key, fmt, 'asint(asuint(o.mrt0) << 16u) >> 16u', 'fe7f210035020401')
        if fmt == 25:
            low_bits_control(key, fmt, 'o.mrt0 & uint4(1023u, 1023u, 1023u, 3u)', 'fe875023')
        print(f'{kind} {backend} narrow format {fmt} MRT0={rgba}; UNORM MRT1=ff0000ff')

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

    # SetGraphicsRoot32BitConstant (list slot 38) feeds the same root constant words as slot 40. The manifest
    # names the (parameter, offset) the shader reads; the title call either lands there or it does not.
    flipped = struct.pack('<4I', *(value ^ word for value in FLOAT_WORDS))
    preserved_offset3 = struct.pack('<4I', *(value ^ 0x01800000 for value in FLOAT_WORDS))
    cases = (
        ('set', '0 0', flipped),
        ('overwrite', '0 0', flipped),            # the second call replaces the first
        ('offset3', '0 3', flipped),
        ('offset3', '0 0', exact),                # a different offset does not reach word 0
        ('preserve', '0 0', flipped),             # slot 38 at offset 3 keeps slot 40's word at offset 0
        ('preserve', '0 3', preserved_offset3),   # and stores its own word at offset 3
        ('corner', '15 7', flipped),              # the last stored slot
        ('outside', '0 0', exact),                # parameter 16, parameter -1, offset 8 and offset -1 are dropped
        ('same_root', '0 0', flipped),            # the same root signature again keeps the word
        ('switch_root', '0 0', exact),            # another root signature clears it
        ('unbound', '0 0', exact),
    )
    for mode, source, expected in cases:
        manifest.write_text(re.sub(r'mrtbits', 'rootconst ' + source, original))
        env['X2W_TEST_ROOT_CONSTANT'] = mode
        pixels, _ = draw(2, word)
        assert pixels == expected, (mode, source, pixels.hex(), expected.hex())
    del env['X2W_TEST_ROOT_CONSTANT']
    manifest.write_text(original)
    print(f'{kind} {backend} slot 38 root constant: {len(cases)} cases (set, overwrite, offset, corner, bounds, same and switched root)')

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
