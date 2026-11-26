"""Bind a translated pixel-stage image store through the runtime (#1996 item F, runtime slice).

The pixel program is the hand-assembled gfx1013 one of tests/startup_pixel_image_store_check.py: it loads an
image descriptor from table 0, stores 1.0, 2.0, 3.0 and 4.0 at texel (1, 2) and exports the same values. It is
published by startup_stream_consumer, translated by tools/startup/gfx2hlsl.py --pending (manifest line
'uav 0 tex2d table 0 0') and drawn through d3d12_xs.dll with X2W_TEST_PIXEL_UAV=<case>
(tests/startup_pixel_uav.cpp). Each case is its own process. After the fence the fixture writes every word of
both mips of four 4x4 textures (A and B writable, C without ALLOW_UNORDERED_ACCESS, D typeless), each word a
distinct sentinel, and the debug layer's error count when the layer is on.

Drawn: the table names A (case a) or B (case b, a view without a description). Texel (1, 2) of mip 0 of that
texture holds the four values; every other word of all four textures, both mips, keeps its sentinel.
Refused, each by its logged reason, with every word unchanged and no translated draw: a table entry with no
view, an SRV, a texture without ALLOW_UNORDERED_ACCESS, an integer view, a view of mip 1, a 2D-array view, an
entry outside the bound table, no table at the parameter, the same texture also bound as an SRV of the draw, and
a texture E that is also the draw's render target (bound through list slot 50).
Manifest lines: a view qualifier (float, uint, and the compute line with 'channels 4'), another image kind, a buffer
line, a 'channels' field and a view index out of order are refused when the manifest is loaded.
Exec: the v_cmpx variants of the program, through the same runtime path: exec on writes the texel, exec off
leaves it. Control: the exec-off shader recompiled with the guard removed from the store must write the texel,
and the check must report it.
"""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from startup_stream_check import container, gfx2hlsl  # noqa: E402
from startup_stream_draw_check import vertex  # noqa: E402
from startup_pixel_image_store_check import (  # noqa: E402
    ARRAY_EXEC_OFF, ARRAY_EXEC_ON, ARRAY_GUARDED_LINE, ARRAY_STORE_LINE, ARRAY_WORDS, ARRAY_WORDS_LAST,
    BITS, EXEC_OFF, EXEC_ON, GUARDED_LINE, STORE_LINE, WORDS,
)

SIDE, MIPS, TEXTURES, ARRAY_SLICES = 4, 2, 4, 6
WRITTEN = [int(b, 16) for b in BITS]
CLEAR, WHITE = bytes((0, 0, 255, 255)), bytes((255, 255, 255, 255))
DRAWN_LINE = 'x2w-draw draw(3, 1, start 0, base 0) pipeline'

# case: (table entry the fixture binds, texture written or None, reason logged, manifest edit)
CASES = {
    'a': (0, 0, None, None),
    'b': (1, 1, None, None),
    'missing': (4, None, 'uav 0 refused: table entry holds no view', None),
    'srv': (5, None, 'uav 0 refused: table entry is not an unordered-access view', None),
    # The device skips a UAV over a resource without ALLOW_UNORDERED_ACCESS and records it as not created, so the draw names the resource.
    'no-ua': (2, None, 'uav 0 refused: resource lacks ALLOW_UNORDERED_ACCESS', None),
    'uint': (3, None, 'uav 0 refused: integer format for a float view', None),
    'mip1': (6, None, 'uav 0 refused: view names a mip other than 0', None),
    'array': (7, None, 'uav 0 refused: view is not a 2D texture view', None),
    'out-of-range': (7, None, 'uav 0 refused: table entry outside the bound table',
                     ('uav 0 tex2d table 0 0', 'uav 0 tex2d table 0 4096')),
    'no-table': (None, None, 'uav 0 refused: no descriptor table at the parameter', None),
    'srv-and-uav': (0, None, 'uav 0 refused: the texture is also an SRV of this draw',
                    ('uav 0 tex2d table 0 0', 'uav 0 tex2d table 0 0\nsrv 0 tex2d table 1 0')),
    'target': (8, None, 'uav 0 refused: the texture is also a render or depth target of this draw', None),
}

# case: (replacement of 'uav 0 tex2d table 0 0', texture written or None, reason logged)
LINES = (
    ('a.float-line', 'uav 0 tex2d float table 0 0', None, 'uav line refused (a view qualifier is not read for pixel stores)'),
    ('a.compute-line', 'uav 0 tex2d float table 0 0 channels 4', None,
     'uav line refused (a view qualifier is not read for pixel stores)'),
    ('a.tex3d-line', 'uav 0 tex3d table 0 0', None, 'uav line refused (only tex2d and tex2da views are bound)'),
    ('a.buffer-line', 'uav 0 raw table 0 0', None, 'uav line refused (buffer view (raw uav) is not bound)'),
    ('a.uint-line', 'uav 0 tex2d uint table 0 0', None, 'uav line refused (a view qualifier is not read for pixel stores)'),
    ('a.channels-field', 'uav 0 tex2d table 0 0 channels 4', None,
     'uav line refused (unexpected field after the table offset)'),
    ('a.second-view', 'uav 1 tex2d table 0 0', None, 'uav line refused (view index out of order)'),
    # 'uavload N' (a partial-dmask store reads the texel back): a view that does not exist, a repeat or junk.
    ('a.uavload-range', 'uav 0 tex2d table 0 0\nuavload 1', None, 'uavload line refused: uavload 1'),
    ('a.uavload-twice', 'uav 0 tex2d table 0 0\nuavload 0\nuavload 0', None, 'uavload line refused: uavload 0'),
    ('a.uavload-junk', 'uav 0 tex2d table 0 0\nuavload x', None, 'uavload line refused: uavload x'),
)


ARRAY_CASES = (
    ('arr-2d-view', 'uav 0 refused: view is not a 2D array texture view', None),
    ('arr-default', 'uav 0 refused: view is not a 2D array texture view', None),
    ('arr-mip1', 'uav 0 refused: view names a mip other than 0', None),
    ('arr-srv-and-uav', 'uav 0 refused: the texture is also an SRV of this draw',
     ('uav 0 tex2da table 0 0', 'uav 0 tex2da table 0 0\nsrv 0 tex2da table 1 0')),
    ('arr-no-ua', 'uav 0 refused: resource lacks ALLOW_UNORDERED_ACCESS', None),
    ('arr-uint', 'uav 0 refused: integer format for a float view', None),
)


def sentinel(t, m, texel, c):
    return 0xc0000000 | t << 12 | m << 8 | texel << 2 | c


def array_sentinel(s, m, texel, c):
    return 0xd0000000 | s << 12 | m << 8 | texel << 2 | c


def expected_array_words(written_slice):
    words = []
    for s in range(ARRAY_SLICES):
        for m in range(MIPS):
            n = SIDE >> m
            for texel in range(n * n):
                for c in range(4):
                    hit = s == written_slice and m == 0 and texel == 2 * n + 1
                    words.append(WRITTEN[c] if hit else array_sentinel(s, m, texel, c))
    return words


def describe_array(index):
    for s in range(ARRAY_SLICES):
        for m in range(MIPS):
            n = SIDE >> m
            if index < n * n * 4:
                texel, c = divmod(index, 4)
                return 'slice %d mip %d texel (%d, %d) word %d' % (s, m, texel % n, texel // n, c)
            index -= n * n * 4
    return 'word past the end'


def expected_words(written):
    words = []
    for t in range(TEXTURES):
        for m in range(MIPS):
            n = SIDE >> m
            for texel in range(n * n):
                for c in range(4):
                    hit = t == written and m == 0 and texel == 2 * n + 1
                    words.append(WRITTEN[c] if hit else sentinel(t, m, texel, c))
    return words


def describe(index):
    """texture, mip, texel and channel of word index of the .uav file."""
    for t in range(TEXTURES):
        for m in range(MIPS):
            n = SIDE >> m
            if index < n * n * 4:
                texel, c = divmod(index, 4)
                return 'texture %s mip %d texel (%d, %d) word %d' % ('ABCD'[t], m, texel % n, texel // n, c)
            index -= n * n * 4
    return 'word past the end'


class Program:
    """One pixel program published and translated into its own cache."""

    def __init__(self, directory, consumer, words, env, kind='tex2d'):
        self.directory, self.consumer, self.kind = directory, consumer, kind
        directory.mkdir()
        self.cache = directory / 'cache'
        self.cache.mkdir()
        self.vs, self.ps, self.root = (directory / name for name in ('vs', 'ps', 'root'))
        self.vs.write_bytes(vertex())
        self.ps.write_bytes(container(0, list(struct.unpack('<%dI' % (len(words) // 4), words))))
        publish = dict(env, X2W_SHADER_CACHE=str(self.cache), X2W_TEST_PIXEL_UAV='arr-ok' if kind == 'tex2da' else 'a')
        publish.pop('X2W_TEST_DEBUG_LAYER', None)
        subprocess.run([consumer, self.vs, self.ps, self.root], env=publish, check=True)
        subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', self.cache], env=publish, check=True)
        self.manifest, = (path for path in self.cache.glob('*.txt') if path.name != 'revision.txt')
        lines = self.manifest.read_text(encoding='ascii').splitlines()
        assert lines[0] == 'x2w-manifest 2' and ('uav 0 %s table 0 0' % kind) in lines and not any(
            line.startswith('srv ') for line in lines), lines


def run_case(program, name, env, backend, cache=None):
    """Returns (returncode, stdout, trap log, render target bytes, texture words)."""
    folder = program.directory / name
    folder.mkdir()
    output, trap = folder / 'pixels', folder / 'trap.log'
    case_env = dict(env, X2W_SHADER_CACHE=str(cache or program.cache), X2W_TEST_PIXEL_UAV=name.split('.')[0],
                    X2W_TRAP_LOG=str(trap))
    run = subprocess.run([program.consumer, program.vs, program.ps, program.root, backend, output], env=case_env,
                         capture_output=True, text=True)
    if run.returncode == 77 and 'no debug layer: skipped' in run.stdout and 'X2W_TEST_DEBUG_LAYER' in env:
        # Rule of the neighbouring fixtures: the debug layer is a condition, not a requirement.
        print('%s: no debug layer on this machine, run without it' % name)
        shutil.rmtree(folder)
        env.pop('X2W_TEST_DEBUG_LAYER')
        return run_case(program, name, env, backend, cache)
    log = trap.read_text(errors='replace') if trap.exists() else ''
    pixels = output.read_bytes() if output.exists() else b''
    uav = Path(str(output) + '.uav')
    words = list(struct.unpack('<%dI' % (uav.stat().st_size // 4), uav.read_bytes())) if uav.exists() else []
    return run.returncode, run.stdout + run.stderr, log, pixels, words


def verdict(name, result, written, reason, drawn, array=False):
    """None when the case holds, otherwise the first difference."""
    returncode, out, log, pixels, words = result
    if returncode:
        return 'consumer exited %d\n%s' % (returncode, out[-3000:])
    if 'device removed reason: 0x00000000' not in out:
        return 'device removed or no report\n%s' % out[-3000:]
    if 'debug layer: on' in out and 'debug layer: on, errors: 0' not in out:
        return 'debug layer errors\n%s' % out[-3000:]
    want = expected_array_words(written) if array else expected_words(written)
    if len(words) != len(want):
        return '%d texture words read back, expected %d' % (len(words), len(want))
    for i, (got, expected) in enumerate(zip(words, want)):
        if got != expected:
            return 'texel_mismatch %s: got %08x, expected %08x' % (
                describe_array(i) if array else describe(i), got, expected)
    if (DRAWN_LINE in log) != drawn:
        return 'translated draw %s' % ('missing' if drawn else 'present although refused')
    if reason and reason not in log:
        return 'reason "%s" not logged\n%s' % (reason, log[-3000:])
    if drawn is not None and name in ('a', 'b', 'arr-ok'):
        rows = [pixels[i:i + 32] for i in range(0, len(pixels), 32)]
        want_row = WHITE * 4 + CLEAR * 4
        if len(rows) != 4 or any(row != want_row for row in rows):
            return 'render target %s, expected the left half white' % pixels.hex()
    if not drawn and pixels != CLEAR * 32:
        return 'render target changed by a refused draw: %s' % pixels.hex()
    return None


def recompile_mutated(program, folder, tag, old, new):
    """A copy of the cache whose pixel shader has old replaced with new."""
    cache = folder / tag
    shutil.copytree(program.cache, cache)
    hlsl, = cache.glob('*.ps.hlsl')
    text = hlsl.read_text()
    assert old in text, (old, text)
    hlsl.write_text(text.replace(old, new))
    fxc = gfx2hlsl.find_fxc()
    if not fxc:
        return None
    cso = hlsl.with_name(hlsl.name.replace('.hlsl', '.cso'))
    subprocess.run([fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo', str(cso), str(hlsl)], check=True,
                   capture_output=True)
    return cache


def recompile_without_guard(program, folder):
    return recompile_mutated(program, folder, 'cache_unguarded', GUARDED_LINE, STORE_LINE)


def check(directory, consumer, backend):
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_D3D12_DEBUG='0', X2W_D3D12_DEVICE_NOOP='1', X2W_TEST_DEBUG_LAYER='1')
    if os.environ.get('X2W_PIXEL_UAV_NO_DEBUG') == '1':
        env.pop('X2W_TEST_DEBUG_LAYER')
    store = Program(directory / 'store', consumer, WORDS, env)
    failures = []

    def expect(program, name, written, reason, drawn, cache=None, array=False):
        result = run_case(program, name, env, backend, cache)
        if result[0] == 77:
            print(result[1])
            sys.exit(77)
        problem = verdict(name, result, written, reason, drawn, array=array)
        adapter = next((line for line in result[1].splitlines() if line.startswith('adapter=')), '')
        layer = next((line for line in result[1].splitlines() if line.startswith('debug layer:')), 'debug layer: ?')
        stride = next((line.rsplit(' ', 1)[1] for line in result[1].splitlines() if 'title descriptor stride' in line), '?')
        layer += '; descriptor stride ' + stride
        if problem:
            failures.append('%s: %s' % (name, problem))
            print('FAIL %s: %s' % (name, problem.splitlines()[0]))
            for line in result[1].splitlines():
                if line.startswith(('debug layer', 'device removed')):
                    print('    ' + line)
        else:
            if array:
                what = ('texel (1, 2) of slice %d written, all other words of 6 slices unchanged' % written
                        if written is not None else 'all 6 slices unchanged')
            else:
                what = ('texel (1, 2) of texture %s written, every other word unchanged' % 'ABCD'[written]
                        if written is not None else 'every word unchanged')
            print('%-16s %s; %s%s [%s; %s]' % (name + ':', 'drawn' if drawn else 'refused', what,
                                              (' (' + reason + ')') if reason else '', adapter, layer))
        return problem

    for name, (_, written, reason, edit) in CASES.items():
        if edit:
            text = store.manifest.read_text(encoding='ascii')
            assert edit[0] in text
            store.manifest.write_text(text.replace(edit[0], edit[1]), encoding='ascii')
        try:
            expect(store, name, written, reason, written is not None)
        finally:
            if edit:
                store.manifest.write_text(text, encoding='ascii')
    # Other forms of the manifest line, table entry 0: load_translation refuses the translation and nothing is drawn.
    text = store.manifest.read_text(encoding='ascii')
    for name, replacement, written, reason in LINES:
        store.manifest.write_text(text.replace('uav 0 tex2d table 0 0', replacement), encoding='ascii')
        try:
            expect(store, name, written, reason, written is not None)
        finally:
            store.manifest.write_text(text, encoding='ascii')

    exec_on = Program(directory / 'exec_on', consumer, EXEC_ON, env)
    exec_off = Program(directory / 'exec_off', consumer, EXEC_OFF, env)
    for program in (exec_on, exec_off):
        assert ('    ' + GUARDED_LINE) in next(program.cache.glob('*.ps.hlsl')).read_text()
    expect(exec_on, 'a.exec-on', 0, None, True)
    expect(exec_off, 'a.exec-off', None, None, True)
    # Control: the exec-off shader without the guard writes the texel, and the check above must catch it.
    unguarded = recompile_without_guard(exec_off, directory)
    if unguarded is None:
        print('control exec guard removed: no fxc, not run')
    else:
        result = run_case(exec_off, 'a.exec-off-unguarded', env, backend, unguarded)
        problem = verdict('a', result, None, None, True)
        needle = 'texel_mismatch texture A mip 0 texel (1, 2) word 0: got %08x, expected %08x' % (WRITTEN[0], sentinel(0, 0, 9, 0))
        if not problem or not problem.startswith(needle):
            failures.append('control exec guard removed was not detected: %s' % problem)
        else:
            print('control exec guard removed: fails as expected (%s)' % problem)
    # 2D-array pixel UAV path (#2742): FirstArraySlice=1, ArraySize=4 -> relative layers 1 and 3 hit physical slices 2 and 4.
    arr_mid = Program(directory / 'arr_mid', consumer, ARRAY_WORDS, env, kind='tex2da')
    arr_last = Program(directory / 'arr_last', consumer, ARRAY_WORDS_LAST, env, kind='tex2da')
    expect(arr_mid, 'arr-ok', 2, None, True, array=True)
    expect(arr_last, 'arr-ok.last', 4, None, True, array=True)
    wrong_layer = recompile_mutated(arr_mid, directory, 'cache_wrong_layer',
                                    'uint3(v0, v1, v2)', 'uint3(v0, v1, v2 + 1u)')
    if wrong_layer is None:
        print('control array wrong layer: no fxc, not run')
    else:
        result = run_case(arr_mid, 'arr-ok.wrong-layer', env, backend, wrong_layer)
        problem = verdict('arr-ok', result, 2, None, True, array=True)
        needle = 'texel_mismatch slice 2 mip 0 texel (1, 2) word 0: got %08x, expected %08x' % (
            array_sentinel(2, 0, 9, 0), WRITTEN[0])
        if not problem or not problem.startswith(needle):
            failures.append('control array wrong layer was not detected: %s' % problem)
        else:
            print('control array wrong layer: fails as expected (%s)' % problem)
    for name, reason, edit in ARRAY_CASES:
        text = arr_mid.manifest.read_text(encoding='ascii')
        if edit:
            assert edit[0] in text
            arr_mid.manifest.write_text(text.replace(edit[0], edit[1]), encoding='ascii')
        try:
            expect(arr_mid, name, None, reason, False, array=True)
        finally:
            if edit:
                arr_mid.manifest.write_text(text, encoding='ascii')
    arr_exec_on = Program(directory / 'arr_exec_on', consumer, ARRAY_EXEC_ON, env, kind='tex2da')
    arr_exec_off = Program(directory / 'arr_exec_off', consumer, ARRAY_EXEC_OFF, env, kind='tex2da')
    for program in (arr_exec_on, arr_exec_off):
        assert ('    ' + ARRAY_GUARDED_LINE) in next(program.cache.glob('*.ps.hlsl')).read_text()
    expect(arr_exec_on, 'arr-ok.exec-on', 2, None, True, array=True)
    expect(arr_exec_off, 'arr-ok.exec-off', None, None, True, array=True)
    arr_unguarded = recompile_mutated(arr_exec_off, directory, 'cache_arr_unguarded',
                                      ARRAY_GUARDED_LINE, ARRAY_STORE_LINE)
    if arr_unguarded is None:
        print('control array exec guard removed: no fxc, not run')
    else:
        result = run_case(arr_exec_off, 'arr-ok.exec-off-unguarded', env, backend, arr_unguarded)
        problem = verdict('arr-ok', result, None, None, True, array=True)
        needle = 'texel_mismatch slice 2 mip 0 texel (1, 2) word 0: got %08x, expected %08x' % (
            WRITTEN[0], array_sentinel(2, 0, 9, 0))
        if not problem or not problem.startswith(needle):
            failures.append('control array exec guard removed was not detected: %s' % problem)
        else:
            print('control array exec guard removed: fails as expected (%s)' % problem)

    if failures:
        raise SystemExit('\n'.join(failures))
    print('%s: %d cases and the exec/layer controls' % (backend, len(CASES) + len(LINES) + len(ARRAY_CASES) + 6))
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
