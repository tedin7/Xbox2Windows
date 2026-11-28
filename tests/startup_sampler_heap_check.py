"""Run-time sampler heap indexing in tools/startup/gfx2hlsl.py (#1143).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture source and compare it with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given) and
                                       draw: the readback must name the sampler table entry the offset selects

The fixture is a source-built gfx1013 vertex+pixel pair. Its pixel stage loads a texture descriptor
from table 0 and four sampler words from the sampler heap at the byte offset in root constant 1 (s9),
then samples at (1.25, 0.5) and exports the result. tests/startup_sampler_heap.cpp binds a 2048-entry
point-sampler table: entry 0 wraps (0.25), entry 1 clamps (0.75), and missing entries use a black
border (0.0). Entry 2047 is reserved for a distinguishable fallback border (0.5).
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

# Pixel stage: s8 = descriptor table 0, s9 = root constant 1 (the byte offset), s10 = sampler heap.
PS_SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_load_dwordx4 s[24:27], s[10:11], s9
v_mov_b32 v0, 0x3fa00000
v_mov_b32 v1, 0.5
s_waitcnt lgkmcnt(0)
image_sample v[0:3], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
"""
PS_WORDS = bytes.fromhex(
    '04040cf4000000fa' '050608f400000012' 'ff02007e0000a03f' 'f002027e' '7fc08cbf'
    '080f80f00000c400' '703f8cbf' '0f1800f800010203' '000081bf')

# The same load at the constant byte offset 0x10. gfx2hlsl still refuses a constant sampler heap
# offset in both modes; on in-box WARP 10.0.19041.5794 a constant index into the sampler table read
# entry 0 with DXC and FXC while any run-time index read the right entry, so constant offsets are a
# separate step.
CONST_SOURCE = PS_SOURCE.replace('s[10:11], s9', 's[10:11], 0x10')
CONST_WORDS = PS_WORDS.replace(bytes.fromhex('050608f400000012'), bytes.fromhex('050608f4100000fa'))

# The run-time load, then s_mov_b32 s25, 0: the sample's s[24:27] no longer holds one heap load.
MIXED_SOURCE = PS_SOURCE.replace('v_mov_b32 v0, 0x3fa00000\n', 's_mov_b32 s25, 0\nv_mov_b32 v0, 0x3fa00000\n')
MIXED_WORDS = PS_WORDS.replace(bytes.fromhex('ff02007e0000a03f'), bytes.fromhex('800399be' 'ff02007e0000a03f'))

VS_SOURCE = """\
v_mov_b32 v4, 1.0
v_mov_b32 v5, 0
exp pos0 v5, v5, v5, v4 done
s_endpgm
"""
VS_WORDS = bytes.fromhex('f202087e' '80020a7e' 'cf0800f805050504' '000081bf')

# Desktop full-screen triangle for the draw; the translated pixel stage reads only SV_Position.
FULLSCREEN_VS = """\
float4 main(uint id : SV_VertexID) : SV_Position
{
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
}
"""

SAMPLER_HEAP_FLAG = 0x800  # D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED


def root_blob(flags=SAMPLER_HEAP_FLAG):
    """A DXBC container with one RTS0 part: parameter 0 a descriptor table, parameter 1 one root constant."""
    params = [(0, 0), (1, 0)]  # (type, visibility)
    poff = 24
    coff = poff + 12 * len(params)
    rts0 = struct.pack('<6I', 2, len(params), poff, 0, 0, flags)
    rts0 += struct.pack('<3I', 0, 0, coff + 12) + struct.pack('<3I', 1, 0, coff)
    rts0 += struct.pack('<3I', 0, 0, 1)        # constants: b0, space 0, one value
    rts0 += struct.pack('<5I', 0, 0, 0, 0, 0)  # table body (gfx2hlsl does not read it)
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return (b'DXBC' + bytes(16) + struct.pack('<III', 1, 32 + 4 + len(part), 1) + struct.pack('<I', 36) + part)


def translate(ps_words, sampler_heap):
    return G.Pipeline(VS_WORDS, ps_words, root_blob(), sampler_heap=sampler_heap)


def controls():
    # Opt-in: the run-time offset indexes the table; no shortcut note, no static sampler.
    pipe = translate(PS_WORDS, True)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'SamplerState qh[2048] : register(s0, space1);' in ps, ps
    m = re.search(r'(\w+)\.Sample\(qh\[x2w_sampler_entry\((s\d+)\)\], ', ps)
    assert m, ps
    assert re.search(r'\b%s = ud1;' % m.group(2), ps), ps
    guard = ('uint x2w_sampler_entry(uint offset) { return (offset & 15u) != 0 || offset >= 32752u ? 2047u : '
             'offset >> 4; }')
    assert guard in ps, ps
    assert 'udsrc 1 rootconst 1 0' in manifest and 'smpheap 2048 1 fallback 2047' in manifest, manifest
    assert not re.search(r'^smp ', manifest, re.M) and not pipe.notes, (manifest, pipe.notes)
    print('run-time offset: qh[x2w_sampler_entry(%s)] with %s = ud1 (root constant 1); unaligned or >= 32752 reads '
          'entry 2047; manifest smpheap 2048 1 fallback 2047' % (m.group(2), m.group(2)))

    # A constant heap offset: refused without the table, a table lookup with it.
    try:
        translate(CONST_WORDS, False)
    except G.Unsupported as error:
        assert 'sampler from a descriptor' in str(error), error
    else:
        raise AssertionError('constant sampler heap offset accepted without the table')
    pipe = translate(CONST_WORDS, True)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    m = re.search(r'(\w+)\.Sample\(qh\[x2w_sampler_entry\((s\d+)\)\], ', ps)
    assert m and re.search(r'\b%s = 16u;' % m.group(2), ps), ps
    assert 'smpheap 2048 1 fallback 2047' in manifest and not pipe.notes, (manifest, pipe.notes)
    print('constant offset 0x10: %s = 16u, qh[x2w_sampler_entry(%s)]; refused without the table' % (m.group(2), m.group(2)))

    # Default (X2W_SAMPLER_HEAP unset): today's shortcut, a static all-zero sampler and a note.
    pipe = translate(PS_WORDS, False)
    assert 'qh[' not in pipe.hlsl_ps() and 'smpheap' not in pipe.manifest(), pipe.manifest()
    assert 'smp 0 0x0 0x0 0x0 0x0' in pipe.manifest(), pipe.manifest()
    assert pipe.notes == {'dynamic sampler index replaced by a default sampler'}, pipe.notes
    print('opt-out: default sampler shortcut unchanged')
    os.environ['X2W_SAMPLER_HEAP'] = '1'
    try:
        assert G.Pipeline(VS_WORDS, PS_WORDS, root_blob()).uses_sampler_heap
    finally:
        del os.environ['X2W_SAMPLER_HEAP']
    print('X2W_SAMPLER_HEAP=1 selects the table')

    # A sampler operand whose words no longer all come from the heap load is refused.
    try:
        translate(MIXED_WORDS, True)
    except G.Unsupported as error:
        assert 'mixes sampler heap words' in str(error), error
    else:
        raise AssertionError('sampler operand with an overwritten heap word accepted')
    print('refused: s25 overwritten between the heap load and the sample')


def assemble(llvm_mc, source):
    run = subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-show-encoding'], input=source,
                         capture_output=True, text=True, check=True)
    words = bytearray()
    for line in run.stdout.splitlines():
        m = re.search(r'encoding: \[([^\]]*)\]', line)
        if m:
            words += bytes(int(b, 16) for b in m.group(1).split(','))
    return bytes(words)


def check_binary(llvm_mc):
    for name, source, words in (('pixel', PS_SOURCE, PS_WORDS), ('constant pixel', CONST_SOURCE, CONST_WORDS),
                                ('mixed pixel', MIXED_SOURCE, MIXED_WORDS),
                                ('vertex', VS_SOURCE, VS_WORDS)):
        built = assemble(llvm_mc, source)
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    print('fixture words match llvm-mc')


def compile_hlsl(compiler, kind, source, out, profile, directory):
    path = directory / ('%s.hlsl' % out)
    path.write_text(source, encoding='ascii')
    if kind == 'fxc':
        args = [compiler, '/nologo', '/T', profile, '/E', 'main', '/Fo', str(directory / ('%s.cso' % out)), str(path)]
    else:
        args = [compiler, '-T', profile, '-E', 'main', '-Fo', str(directory / ('%s.cso' % out)), str(path)]
    subprocess.run(args, check=True, capture_output=True, text=True)
    return directory / ('%s.cso' % out)


def check_draw(dxc, renderer, adapter, fxc=None):
    # (pixel words, table byte offset, expected red bits)
    cases = [(0, 0x3e800000), (16, 0x3f400000), (32, 0x00000000), (2046 * 16, 0x00000000),
             (2047 * 16, 0x3f000000), (0xfffffff0, 0x3f000000), (8, 0x3f000000)]
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        dynamic = translate(PS_WORDS, True)
        static = translate(PS_WORDS, False)
        manifest = directory / 'dynamic.txt'
        # Include a space-zero sampler alongside the translated space-one table.
        manifest.write_text(dynamic.manifest().rstrip() + '\nsmp 0 0 0 0 0\n', encoding='ascii')
        static_manifest = directory / 'static.txt'
        static_manifest.write_text(static.manifest(), encoding='ascii')
        legacy_manifest = directory / 'legacy.txt'
        legacy_manifest.write_text(dynamic.manifest().replace(' fallback 2047', ''), encoding='ascii')
        bad_manifest = directory / 'bad.txt'
        bad_manifest.write_text(dynamic.manifest().replace('smpheap 2048 1', 'smpheap 2049 1'), encoding='ascii')
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'fullscreen_%s' % kind, vs_profile, directory)
            shader = compile_hlsl(compiler, kind, dynamic.hlsl_ps(), 'ps_%s' % kind, ps_profile, directory)
            static_shader = compile_hlsl(compiler, kind, static.hlsl_ps(), 'ps_static_%s' % kind, ps_profile, directory)
            # Negative control: the run-time shader with its index dropped must lose entry 1.
            guarded = dynamic.hlsl_ps()
            dropped = re.sub(r'qh\[x2w_sampler_entry\(s\d+\)\]', 'qh[0]', guarded)
            assert dropped != guarded
            mutant = compile_hlsl(compiler, kind, dropped, 'ps_dropped_%s' % kind, ps_profile, directory)
            for offset, expected, dropped_index in [c + (False,) for c in cases] + [(16, 0x3f400000, True)]:
                run = subprocess.run([renderer, adapter, '--startup-sampler-heap', str(vs),
                                      str(mutant if dropped_index else shader), str(offset), '%08x' % expected, str(manifest)],
                                     capture_output=True, text=True)
                sys.stdout.write(run.stdout)
                # offscreen_check prefixes each info-queue message with its numeric D3D12 ID.
                if 'D3D12 1361:' in run.stderr:
                    raise AssertionError('ordinary sampler heap emitted D3D12 warning 1361')
                label = 'dropped index' if dropped_index else 'run-time offset'
                if dropped_index:
                    if run.returncode == 0 or 'mismatch offset=16 expected=3f400000 actual=3e800000' not in run.stderr:
                        raise AssertionError('%s %s: control did not fail at entry 1\n%s\n%s'
                                             % (kind, label, run.stdout, run.stderr))
                    print('%s %s: fails at entry 1 as expected' % (kind, label), flush=True)
                    continue
                if run.returncode == 77:
                    print(run.stdout + run.stderr)
                    sys.exit(77)
                if run.returncode:
                    raise AssertionError('%s %s offset %d\n%s\n%s' % (kind, label, offset, run.stdout, run.stderr))
                assert 'null_sampler_unreadable=1 null_sampler_out_of_range=2044' in run.stdout, run.stdout
                print('%s %s offset %d: %08x' % (kind, label, offset, expected), flush=True)

            # A constant byte offset 0x10 names entry 1 whatever the root constant says; dropping the index reads entry 0.
            constant = translate(CONST_WORDS, True).hlsl_ps()
            constant_shader = compile_hlsl(compiler, kind, constant, 'ps_const_%s' % kind, ps_profile, directory)
            lost = re.sub(r'qh\[x2w_sampler_entry\(s\d+\)\]', 'qh[0]', constant)
            assert lost != constant
            constant_mutant = compile_hlsl(compiler, kind, lost, 'ps_const_lost_%s' % kind, ps_profile, directory)
            for root_offset in (0, 32):
                run = subprocess.run([renderer, adapter, '--startup-sampler-heap', str(vs), str(constant_shader),
                                      str(root_offset), '3f400000', str(manifest)], capture_output=True, text=True)
                sys.stdout.write(run.stdout)
                assert run.returncode == 0, f'{kind} constant offset (root {root_offset})\n{run.stdout}\n{run.stderr}'
                print(f'{kind} constant offset 0x10 (root constant {root_offset}): 3f400000', flush=True)
            run = subprocess.run([renderer, adapter, '--startup-sampler-heap', str(vs), str(constant_mutant),
                                  '0', '3f400000', str(manifest)], capture_output=True, text=True)
            assert run.returncode != 0 and 'mismatch offset=0 expected=3f400000 actual=3e800000' in run.stderr, \
                f'{kind} constant index dropped\n{run.stdout}\n{run.stderr}'
            print(f'{kind} constant offset with the index dropped: fails at entry 1 as expected', flush=True)

            for label, pixel, bindings, extra, failure in (
                ('static only', static_shader, static_manifest, [], None),
                ('legacy manifest', shader, legacy_manifest, ['rebind'], None),
                ('changed binding', shader, manifest, ['rebind'], None),
                ('stale binding', shader, manifest, ['stale'], 'mismatch offset=16 expected=3e800000 actual=3f400000'),
                ('invalid manifest', shader, bad_manifest, [], 'Invalid sampler heap manifest'),
            ):
                run = subprocess.run([renderer, adapter, '--startup-sampler-heap', str(vs), str(pixel),
                                      '16', '3e800000', str(bindings), *extra], capture_output=True, text=True)
                sys.stdout.write(run.stdout)
                if failure:
                    assert run.returncode == 1 and failure in run.stderr, f'{kind} {label}\n{run.stdout}\n{run.stderr}'
                else:
                    assert run.returncode == 0, f'{kind} {label}\n{run.stdout}\n{run.stderr}'
                print(f'{kind} {label}: {"failed at the expected check" if failure else "passed"}', flush=True)


def check_wrapped_heap(dll_path, backend):
    import ctypes as c
    import struct
    import uuid

    ptr, uint, word, hr = c.c_void_p, c.c_uint32, c.c_uint64, c.c_int32

    def guid(value):
        return (c.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le)

    def method(obj, slot, result, *args):
        table = c.cast(obj, c.POINTER(c.POINTER(ptr)))[0]
        return c.WINFUNCTYPE(result, ptr, *args)(table[slot])

    adapter = ptr()
    if backend == 'warp':
        factory = ptr()
        dxgi = c.WinDLL('dxgi.dll')
        dxgi.CreateDXGIFactory2.argtypes = (uint, ptr, c.POINTER(ptr))
        dxgi.CreateDXGIFactory2.restype = hr
        assert dxgi.CreateDXGIFactory2(
            0, guid('1bc6ea02-ef36-464f-bf0c-21ca39e5168a'), c.byref(factory)) == 0
        assert method(factory, 27, hr, ptr, c.POINTER(ptr))(
            factory, guid('29038f61-3839-4626-91fd-086879011a05'), c.byref(adapter)) == 0

    dll = c.WinDLL(dll_path)
    create = dll.D3D12XboxCreateDevice
    create.argtypes = (ptr, ptr, ptr, c.POINTER(ptr))
    create.restype = hr
    device = ptr()
    parameters = c.create_string_buffer(64)
    assert create(adapter, parameters, guid('189819f1-1db6-4b57-be54-1821339b85f7'), c.byref(device)) == 0
    native_stride = method(device, 15, uint, uint)(device, 1)

    def heap(flags):
        sentinel = 0xABCDEF00
        memory = (uint * 12)(*([sentinel] * 12))
        start = c.addressof(memory)
        # SAMPLER heap, three 16-byte records, with the stand-in's explicit
        # title-memory extension. Both arrays retain their storage through the copies.
        desc = c.create_string_buffer(struct.pack('<4I2Q', 1, 3, flags, 0, start, 48))
        result = ptr()
        assert method(device, 14, hr, ptr, ptr, c.POINTER(ptr))(
            device, desc, guid('8efb471d-616c-4f49-90f7-127bb763fa51'), c.byref(result)) == 0
        actual = method(result, 9, word, ptr)(result, None)
        assert actual == start, f'title sampler CPU handle was not returned (desktop increment {native_stride})'
        return memory

    source, target = heap(0), heap(1)
    src, dst = c.addressof(source), c.addressof(target)
    create_sampler = method(device, 22, None, ptr, word, word, word)
    for slot, address_u in ((1, 1), (2, 3)):
        desc = c.create_string_buffer(struct.pack('<4If2I6f', 0, address_u, 3, 3, 0.0, 1, 1, *([0.0] * 6)))
        create_sampler(device, desc, src + 16 * slot, 0, 0)
    # The console method and the desktop method must publish identical tokens
    # for equivalent samplers, including when the U address mode changes.
    console_sampler = method(device, 61, None, ptr, word)
    console = (uint * 27)()
    console[7] = console[8] = 3
    for slot, address_u in ((1, 1), (2, 3)):
        console[6] = address_u
        console_sampler(device, console, src)
        assert source[:4] == source[4 * slot:4 * slot + 4], list(source)
    # Independently encode linear min/mip + point mag, three address modes,
    # explicit bias, border and LOD range in each ABI.
    mixed = c.create_string_buffer(struct.pack('<4If2I6f', 0x11, 2, 3, 4, -.5, 1, 1,
                                               .25, .5, .75, 1., 1., 3.))
    create_sampler(device, mixed, src, 0, 0)
    expected = list(source[:4])
    assert expected != list(source[4:8]) and expected != list(source[8:12])
    console[2], console[3] = 1, 2
    console[6:9] = 2, 3, 4
    console[9:12] = struct.unpack('<3I', struct.pack('<3f', 1., 3., -.5))
    console[17:21] = struct.unpack('<4I', struct.pack('<4f', .25, .5, .75, 1.))
    console_sampler(device, console, src)
    assert list(source[:4]) == expected, list(source)
    assert source[4] == source[8] == 0x56573258, list(source)
    assert source[5] != source[9], 'distinct samplers produced the same token'
    method(device, 24, None, uint, word, word, uint)(device, 1, dst + 32, src + 16, 1)
    assert target[8:12] == source[4:8] and target[:8] == [0xABCDEF00] * 8, list(target)

    dest_ranges, source_ranges = (word * 1)(dst), (word * 2)(src + 16, src + 32)
    dest_sizes = (uint * 1)(2)
    method(device, 23, None, uint, ptr, ptr, uint, ptr, ptr, uint)(
        device, 1, dest_ranges, dest_sizes, 2, source_ranges, None, 1)
    assert target[:8] == source[4:12] and target[8:12] == source[4:8], list(target)
    assert method(device, 37, hr)(device) == 0
    print(f'wrapped sampler heap {backend}: title_stride=16 desktop_stride={native_stride}, token copies match')


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        controls()
        check_draw(*args)
    elif mode == 'wrapped':
        check_wrapped_heap(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
