"""One front door for compute requests: the default translator first, the engine when it refuses.

tools/startup/gfx2hlsl.py translate_compute() runs the default translator; on an `unsupported:` refusal it hands the same request,
under the same key, to tools/startup/gfx2hlsl_engine.py and writes whichever translation succeeds (it prints
"translated via engine"). Graphics and mesh requests never reach the engine. Opt-in: the second try runs only with X2W_ENGINE_FALLBACK=1 (default off = today's behaviour).

Part 1 (routing, stand-in engine): the engine is called only after a default refusal, with an empty key suffix during the call (the
request keeps the default key) and the real suffix restored after; a default translation never reaches it; a crash or a refusal
of the engine leaves the default's .fail text on record; an engine success removes the .fail.
Part 2 (real engine, no stand-in): a compute program the default refuses (a scalar written in a block s_cbranch_execz skips and read
after the join, the shape of startup_engine_compute_check.py case JOIN-S-after) is translated through the front door to a
`stage cs` manifest and HLSL in compute tick mode; with the fallback off the same request is refused and publishes nothing.
The dispatch of the same program on a GPU is startup_engine_compute_check.py case JOIN-S-after.
"""
import contextlib
import io
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx2hlsl as G  # noqa: E402  (the default translator: nothing here imports the engine shim)
import startup_gfx2hlsl_compute_check as C  # noqa: E402
from startup_compute_translate_check import state_a, write_request  # noqa: E402

FAILED = []
JOIN_WORDS = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf0783148f1400064a82060834870396be7e0395be8300887d6a03febe020088bf'
    '07051693160416801503febe16000c4a001070e004060480000081bf')
JOIN_RSRC2 = C.USER7 | C.TGID_X | C.tidig(1)
JOIN_GROUP = (8, 1, 1)


def expect(name, got, want):
    if got != want:
        FAILED.append(name)
        print('FAIL %s: got %r, want %r' % (name, got, want))


def run(call):
    """(result, what was logged on stderr) of one call: stdout stays the translator's result lines."""
    out = io.StringIO()
    with contextlib.redirect_stderr(out):
        result = call()
    return result, out.getvalue()


def routing_checks():
    seen = []
    engine = types.ModuleType('gfx2hlsl_engine')
    engine.ENGINE_KEY_SUFFIX = b'-engine1'
    engine.mode = 'ok'

    def translate(out_dir, key):
        seen.append((key, engine.ENGINE_KEY_SUFFIX))
        if engine.mode == 'crash':
            raise NameError('stand-in')
        if engine.mode == 'refuse':
            Path(out_dir, '%016x.fail' % key).write_text('engine reason\n')
            return 'unsupported: engine reason'
        return 'ok'
    engine.translate_compute = translate
    real = sys.modules.get('gfx2hlsl_engine')
    sys.modules['gfx2hlsl_engine'] = engine
    os.environ['X2W_ENGINE_FALLBACK'] = '1'
    os.environ['X2W_FXC'] = ''
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            # A program the default translates: the engine is never asked.
            _, words, rsrc2, group, _ = C.CASES['A']
            good = write_request(cache, words, state_a(group, rsrc2))
            result, _ = run(lambda: G.translate_compute(str(cache), good))
            expect('a default translation does not reach the engine', (result, seen), ('ok', []))
            # A request the default refuses (no files): the engine gets the same key, with no suffix during the call.
            missing = 0x1234
            result, text = run(lambda: G.translate_compute(str(cache), missing))
            expect('a refusal goes to the engine under the same key, suffix off during the call', (result, seen), ('ok', [(missing, b'')]))
            expect('the real suffix is restored', engine.ENGINE_KEY_SUFFIX, b'-engine1')
            expect('the log names the engine', 'translated via engine' in text, True)
            expect('an engine success leaves no .fail', Path(cache, '%016x.fail' % missing).exists(), False)
            # The engine crashes: a refusal, the default's text stays, the translator lives.
            seen.clear()
            engine.mode = 'crash'
            result, text = run(lambda: G.translate_compute(str(cache), missing))
            expect('an engine crash is a refusal', (result.startswith('unsupported:'), 'engine also refused' in text), (True, True))
            expect('the default reason stays on record after a crash', 'unreadable' in Path(cache, '%016x.fail' % missing).read_text(), True)
            # The engine refuses too: the default's reason stays, not the engine's.
            engine.mode = 'refuse'
            result, _ = run(lambda: G.translate_compute(str(cache), missing))
            expect('the default reason stays after an engine refusal', 'unreadable' in Path(cache, '%016x.fail' % missing).read_text(), True)
            # Kill switch.
            seen.clear()
            os.environ.pop('X2W_ENGINE_FALLBACK', None)
            result, _ = run(lambda: G.translate_compute(str(cache), missing))
            expect('fallback off: the engine is never asked', (result.startswith('unsupported:'), seen), (True, []))
            os.environ['X2W_ENGINE_FALLBACK'] = '1'
            # A graphics key never reaches the engine.
            result, _ = run(lambda: G.translate_request(str(cache), 0x5678))
            expect('a graphics request does not reach the engine', seen, [])
    finally:
        if real is None:
            sys.modules.pop('gfx2hlsl_engine', None)
        else:
            sys.modules['gfx2hlsl_engine'] = real


def real_engine_checks():
    os.environ['X2W_FXC'] = ''
    sys.modules.pop('gfx2hlsl_engine', None)
    for fallback in ('on', 'off'):
        if fallback == 'off':
            os.environ.pop('X2W_ENGINE_FALLBACK', None)
        else:
            os.environ['X2W_ENGINE_FALLBACK'] = '1'
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            key = write_request(cache, JOIN_WORDS, state_a(JOIN_GROUP, JOIN_RSRC2))
            result, text = run(lambda: G.translate_request(str(cache), key))
            manifest, hlsl = cache / ('%016x.txt' % key), cache / ('%016x.cs.hlsl' % key)
            if fallback == 'on':
                expect('real engine: the refused program translates through the front door', result, 'ok')
                expect('real engine: the log says via engine', 'translated via engine' in text, True)
                expect('real engine: a stage cs manifest is published', manifest.read_text().startswith('x2w-manifest 2\nstage cs\n'), True)
                expect('real engine: the HLSL is compute tick mode', hlsl.read_text().startswith('// x2w compute tick mode'), True)
                expect('real engine: no .fail is left', (cache / ('%016x.fail' % key)).exists(), False)
            else:
                expect('fallback off: the default refusal stands', result.startswith('unsupported:'), True)
                expect('fallback off: nothing is published', (manifest.exists(), hlsl.exists()), (False, False))
                expect('fallback off: the refusal is the live-at-join text', 'live where the block joins' in (cache / ('%016x.fail' % key)).read_text(), True)
    os.environ.pop('X2W_ENGINE_FALLBACK', None)


def main():
    routing_checks()
    real_engine_checks()
    if FAILED:
        print('FAILED: %d' % len(FAILED))
        return 1
    print('startup_engine_front_door: PASS (default first, the engine on a refusal under the same key; graphics and mesh never reach it)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
