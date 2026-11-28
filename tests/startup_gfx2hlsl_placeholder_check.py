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
TAIL = struct.pack('<3I', 0xfc123456, 0xffffffff, 0xfd135791)
MIXED = (mov(7, 0.75) + TRAP + export(0, (7, 6, 7, 6), done=False)
         + export(1, (6, 7, 6, 7)) + END + TAIL)
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


def check_export_guards():
    def refused(code, reason):
        try:
            G.pixel_export_targets(code)
        except G.Unsupported as error:
            assert str(error) == reason, str(error)
        else:
            raise AssertionError('placeholder accepted an invalid export signature: ' + reason)

    unreadable = 'unreadable instruction in placeholder pixel program'
    invalid_export = 'unreadable placeholder pixel export'
    refused(VALUES + TAIL + MRT0 + END, unreadable)
    refused(SINGLE[:-len(END)], 'placeholder pixel program has no end')
    refused(SINGLE + TAIL + MRT0, 'placeholder export after final program end')
    enter_tail = struct.pack('<I', 0xbf820003)  # Branch across the export and end.
    back_edge = struct.pack('<I', 0xbf82fffa)
    refused(enter_tail + MRT0 + END + TAIL[:4] + back_edge, 'placeholder pixel program has no end')
    refused(SINGLE + export(1, (6, 6, 6, 6), mask=1) + END, invalid_export)
    refused(VALUES + export(0, (6, 6, 6, 6), mask=7) + END, invalid_export)
    refused(VALUES + export(10, (6, 6, 6, 6)) + END, invalid_export + ' target')
    assert G.pixel_export_targets(SINGLE + export(7, (6, 6, 6, 6)) + END + TAIL) == {'mrt0', 'mrt7'}
    assert G.pixel_export_targets(export(9, (0, 0, 0, 0), mask=0) + END + TAIL) == set()

    decode_instruction = G.D.decode_instruction

    def nonregister(code, offset):
        size, name, asm, comment = decode_instruction(code, offset)
        return size, name, asm.replace('v6', '1.0', 1) if name == 'exp' else asm, comment

    # Hardware sources encode VGPR indices. Inject a non-register disassembly operand at the reader boundary.
    with patch.object(G.D, 'decode_instruction', nonregister):
        refused(SINGLE, invalid_export)


def check_tail_mutant(directory):
    decode = G.decode

    def reject_tail(code, *args, **kwargs):
        if kwargs.get('exports_only') and code.endswith(TAIL):
            raise G.Unsupported('unreadable instruction in placeholder pixel program')
        return decode(code, *args, **kwargs)

    with patch.object(G, 'decode', reject_tail), patch.dict(os.environ, X2W_FXC='', X2W_PLACEHOLDER_PS='1'):
        key = request(directory, VS, MIXED, (2, 30))
        assert G.translate_request(str(directory), key) == 'unsupported: ' + REASON
        assert not result_file(directory, key, '.txt').exists()
        assert result_file(directory, key, '.fail').read_text().splitlines()[0] == REASON


def check_requests(directory, family_retry):
    supported = request(directory, VS, VALUES + MRT0 + END)
    supported_tail = request(directory, VS, VALUES + MRT0 + END + TAIL)
    single = request(directory, VS, SINGLE)
    mixed = request(directory, VS, MIXED, (2, 30))
    base = G.pipeline_key(VS, MIXED, root_blob())
    assert G.decode(MIXED)[-1][2] == 's_endpgm'
    assert G.pixel_export_targets(MIXED) == {'mrt0', 'mrt1'}
    bad_vertices = [request(directory, struct.pack('<I', 0xbf920000 | n) + VS, SINGLE)
                    for n in range(G.FAMILY_PROBES)]
    bad_vs = bad_vertices[0]
    bad_export = request(directory, VS, VALUES + TRAP + export(10, (6, 6, 6, 6)) + MRT0 + END)

    def translate(key):
        return G.translate_request(str(directory), key)

    def content(key, extension):
        return result_file(directory, key, extension).read_text(encoding='ascii')

    with patch.dict(os.environ, X2W_FXC='', X2W_PLACEHOLDER_RGB='0,255,0',
                    X2W_PRE_RUN_FAMILY_RETRY=family_retry, X2W_RETRY_FAILED='0'):
        os.environ.pop('X2W_PLACEHOLDER_PS', None)
        assert translate(supported) == translate(supported_tail) == 'ok'
        suffixes = ('.vs.hlsl', '.ps.hlsl', '.txt')
        unchanged = {(key, ext): result_file(directory, key, ext).read_bytes()
                     for key in (supported, supported_tail) for ext in suffixes}
        for key in (single, mixed, base, *bad_vertices):
            assert translate(key) == 'unsupported: ' + REASON
            assert content(key, '.fail').splitlines()[0] == REASON
            assert not result_file(directory, key, '.txt').exists()
        assert single not in G.pending_keys(str(directory), True)

        os.environ['X2W_PLACEHOLDER_PS'] = '1'
        assert single in G.pending_keys(str(directory), True)
        assert translate(supported) == translate(supported_tail) == 'ok'
        assert all(result_file(directory, key, ext).read_bytes() == data for (key, ext), data in unchanged.items())
        weights = dict.fromkeys(bad_vertices, 1)
        report = G.retry_lifted(str(directory), jobs=1, weights=weights)
        assert (report.keys, report.ok, report.refused, report.skipped) == (len(bad_vertices) + 3, 0,
                                                                            len(bad_vertices) + 3, 0)
        assert G.retry_lifted(str(directory), jobs=1).keys == 0
        for key, targets in ((single, '0'), (mixed, '0 1')):
            manifest = content(key, '.txt').splitlines()
            assert 'placeholder ps ' + REASON in manifest and 'targets ' + targets in manifest
            assert content(key, '.fail').splitlines().count(DRAWN) == 1
            assert 'SV_Depth' not in content(key, '.ps.hlsl')
            assert key not in G.pending_keys(str(directory), True)
        assert 'uint4 mrt1 : SV_Target1;' in content(mixed, '.ps.hlsl')
        assert translate(bad_vs) == translate(bad_export) == 'unsupported: ' + REASON
        for key in (*bad_vertices, bad_export):
            assert not result_file(directory, key, '.txt').exists()
            assert G.PLACEHOLDER_ATTEMPT in content(key, '.fail').splitlines()
        assert bad_vs not in G.pending_keys(str(directory), True)
        rows = tester_pack.cache_results(str(directory))
        assert [(status, reason) for key, _, status, reason in rows if key == '%016x' % mixed] == [('refused', DRAWN)]
        assert tester_pack.clean_reason('long reason ' * 30 + DRAWN).endswith(G.PLACEHOLDER_DRAWN)
        with patch.dict(os.environ, X2W_FXC='fxc.exe'):
            assert single in G.pending_keys(str(directory), True)  # source-only result needs bytecode

        depth_code = VALUES + TRAP + export(8, (6, 0, 0, 0), mask=1, done=False) + MRT0 + END
        depth = G.Pipeline(VS, depth_code, root_blob()).hlsl_ps()
        assert 'float depth : SV_Depth;' in depth and 'o.depth = pos.z;' in depth
        shaders = tuple(content(key, '.ps.hlsl') for key in (single, mixed))
        os.environ['X2W_PLACEHOLDER_RGB'] = 'bad'
        assert G.placeholder_rgb() == (255, 0, 255)
        assert mixed in G.pending_keys(str(directory), True)
        assert G.retry_lifted(str(directory), jobs=1).refused == 3
        assert 'placeholder rgb 255,0,255' in content(mixed, '.txt')
        assert '255.0/255.0, 0.0/255.0, 255.0/255.0' in content(mixed, '.ps.hlsl')
        assert content(mixed, '.fail').splitlines()[0] == DRAWN
        os.environ['X2W_PLACEHOLDER_PS'] = '0'
        assert mixed in G.pending_keys(str(directory), True)
        assert G.retry_lifted(str(directory), jobs=1).refused == 3
        for key in (single, mixed, base):
            assert not result_file(directory, key, '.txt').exists()
            assert content(key, '.fail').splitlines()[0] == REASON
        assert all(result_file(directory, key, ext).read_bytes() == data for (key, ext), data in unchanged.items())
        rows = tester_pack.cache_results(str(directory))
        assert [(status, reason) for key, _, status, reason in rows if key == '%016x' % mixed] == [('refused', REASON)]
    return content(supported, '.vs.hlsl'), shaders


def check_magenta_switch(directory):
    """X2W_MAGENTA_REFUSED=1 alone publishes the magenta placeholder; off, or with X2W_PLACEHOLDER_PS=0, the refusal stays.
    Returns the magenta pixel shader source."""
    def run(key_directory, **environment):
        with patch.dict(os.environ, X2W_FXC='', **environment):
            for name in ('X2W_PLACEHOLDER_PS', 'X2W_PLACEHOLDER_RGB', 'X2W_MAGENTA_REFUSED'):
                if name not in environment:
                    os.environ.pop(name, None)
            key = request(key_directory, VS, SINGLE)
            G.translate_request(str(key_directory), key)
            return key

    off = directory / 'off'
    off.mkdir()
    key = run(off)
    assert result_file(off, key, '.fail').read_text().splitlines()[0] == REASON, 'switch off: the refusal must stand'
    assert not result_file(off, key, '.txt').exists()
    explicit_off = directory / 'explicit_off'
    explicit_off.mkdir()
    key = run(explicit_off, X2W_MAGENTA_REFUSED='1', X2W_PLACEHOLDER_PS='0')
    assert not result_file(explicit_off, key, '.txt').exists(), 'X2W_PLACEHOLDER_PS=0 wins over the short switch'
    on = directory / 'on'
    on.mkdir()
    key = run(on, X2W_MAGENTA_REFUSED='1')
    manifest = result_file(on, key, '.txt').read_text().splitlines()
    assert 'placeholder ps ' + REASON in manifest and 'placeholder rgb 255,0,255' in manifest, manifest
    shader = result_file(on, key, '.ps.hlsl').read_text(encoding='ascii')
    assert '255.0/255.0, 0.0/255.0, 255.0/255.0' in shader, shader[-400:]
    return shader


def compare(data, layout, reference, label, expected_mismatches=0):
    size = struct.calcsize(layout) * 64
    assert len(data) == size, f'{label}: expected {size} bytes, got {len(data)}'
    mismatches = []
    for index, pixel in enumerate(struct.iter_unpack(layout, data)):
        if pixel != reference:
            mismatches.append((index % 8, index // 8, pixel))
    assert len(mismatches) == expected_mismatches, (
        f'{label}: {len(mismatches)} differing pixels, expected {expected_mismatches}. '
        f'Reference RGBA {reference}, first mismatch (x, y, RGBA) {mismatches[:1]}'
    )


def draw_checks(directory, vs_source, shaders, dxc, renderer, magenta_source=None):
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

    clear, green, integer = (-1.0,) * 4, (0.0, 1.0, 0.0, 1.0), (1, 2, 3, 1)
    magenta_ps = compile_shader(magenta_source, 'magenta', 'ps_6_0') if magenta_source else None
    compare(readback('-', 'color'), '<4f', clear, 'refused')
    compare(readback(single, 'color'), '<4f', green, 'single MRT0')
    if magenta_ps:
        # X2W_MAGENTA_REFUSED=1: a draw whose pixel shader was refused covers the triangle in magenta. The off control is the
        # first readback above (no pixel shader: the clear value stays).
        compare(readback(magenta_ps, 'color'), '<4f', (1.0, 0.0, 1.0, 1.0), 'magenta single MRT0')
    float_bytes = struct.calcsize('<4f') * 64
    valid = readback(mixed, 'mixed')
    compare(valid[:float_bytes], '<4f', green, 'mixed MRT0')
    compare(valid[float_bytes:], '<4B', integer, 'mixed MRT1')
    # The tail-rejection mutant published no shader. Both targets retain their clear values.
    mutant = readback('-', 'mixed')
    compare(mutant[:float_bytes], '<4f', clear, 'tail mutant MRT0 clear')
    compare(mutant[float_bytes:], '<4B', (0, 0, 0, 1), 'tail mutant MRT1 clear')
    compare(mutant[:float_bytes], '<4f', green, 'tail mutant MRT0', expected_mismatches=64)
    compare(mutant[float_bytes:], '<4B', integer, 'tail mutant MRT1', expected_mismatches=64)
    one_export = readback(single, 'mixed')
    compare(one_export[:float_bytes], '<4f', green, 'single-export mutant MRT0')
    compare(one_export[float_bytes:], '<4B', integer, 'single-export mutant MRT1', expected_mismatches=64)
    print('64 pixels match each CPU reference. The tail-rejection mutant differs at all 64 pixels on both targets.')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        check_export_guards()
        mutant_dir = directory / 'mutant'
        mutant_dir.mkdir()
        check_tail_mutant(mutant_dir)
        magenta_dir = directory / 'magenta'
        magenta_dir.mkdir()
        magenta_source = check_magenta_switch(magenta_dir)
        legacy = directory / 'legacy'
        legacy.mkdir()
        check_requests(legacy, '0')
        vs, shaders = check_requests(directory, '1')
        print('placeholder request, cache, export signature, and refusal checks passed')
        if len(sys.argv) == 4 and sys.argv[1] == 'draw':
            draw_checks(directory, vs, shaders, *sys.argv[2:], magenta_source=magenta_source)
        elif len(sys.argv) != 1:
            raise SystemExit('usage: startup_gfx2hlsl_placeholder_check.py [draw <dxc> <offscreen_check>]')
