"""Pixel input masks from the captured stream description of the exact pipeline (#1629).

  (no arguments)                       association controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate the requests below through translate_request, compile the
                                       pixel stage (DXC ps_6_0, and FXC ps_5_1 when given) and draw it with the
                                       #1634 gradient: every pixel must read back its own packed x | (y << 16);
                                       with fxc, also the compiled-result path of a rewritten description

Two source-built requests share one cache directory. Each has a stream description, req_<key>.psin:
  target  exports v4; enable 0x8002 (PERSP_CENTER, POS_FIXED_PT) and address 0x8003 (PERSP_SAMPLE allocated
          but not enabled). The address mask lays out v0..v1 PERSP_SAMPLE, v2..v3 PERSP_CENTER, v4 the packed
          position. Laying out by the enable mask would put the position in v2.
  decoy   exports v2; enable = address = 0x8002, so its position is v2.
Both draw correctly only when each translation takes the masks of its own description, so no fixed register
and no mask taken from the other description passes. The controls also refuse, by name, a description that
names another pipeline, a stale description whose stage blobs changed, a missing enable or address mask,
contradictory masks (an enabled input not allocated) and a malformed line.

The masks are part of the request identity (pipeline_key with ps_inputs). In the same cache, pipelines with
the target's exact stage bytes, root signature and input layout but another mask pair are requests of their
own: 0x8003/0x8003 also places the position in v4 and draws the gradient; 0x8002/0x8002 places it in v2, so
the v4 this code reads is not the position and the draw fails at (1,0) as expected. They arrive after the
target's result exists and are translated as new requests, leaving that result untouched. A description
rewritten in place with another pair after its result was published is checked again, refused by name and
its old result withdrawn. The watcher follows the description each translation reads (an RTV variant reads its
source pipeline's): a wrong pair in a base description withdraws the base and its variant, and the corrected
description, newer than their failures, publishes both again without touching the request files. A result
translated with a description records it (<key>.psdep): removing only the base description withdraws the base
and its variant and refuses both by name, while a legacy request that never had a description has no record
and is not checked again.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, G, compile_hlsl, root_blob

TARGET = (0x8002, 0x8003)  # (enable, address): the packed position in v4
DECOY = (0x8002, 0x8002)   # the packed position in v2
SAME_V4 = (0x8003, 0x8003)  # another valid pair with the packed position in v4 (PERSP_SAMPLE also enabled)


def ps_words(reg):
    # exp mrt0 vN, vN, vN, vN done vm; s_endpgm (the #1634 fixture's pixel stage)
    return struct.pack('<III', 0xf800180f, reg * 0x01010101, 0xbf810000)


def vop2(op, vdst, src0, vsrc1):
    return struct.pack('<I', op << 25 | vdst << 17 | vsrc1 << 9 | src0)


def split_words(reg, low=1, high=2, out=5):
    """A pixel stage that splits the packed position register `reg` into its 16-bit halves and puts them back together
    (v_and_b32 low, 0xffff, reg; v_lshrrev_b32 high, 16, reg; v_lshlrev_b32 high, 16, high; v_or_b32 out, low, high), then
    exports `out`. The result is the packed position again, and the program's own register use says which register is
    the position: the halves are the evidence a request without a stream description has."""
    return b''.join((vop2(0x1b, low, 255, reg) + struct.pack('<I', 0xffff), vop2(0x16, high, 144, reg),
                     vop2(0x1a, high, 144, high), vop2(0x1c, out, 256 + low, high),
                     struct.pack('<III', 0xf800180f, out * 0x01010101, 0xbf810000)))


def put_request(cache, ps, masks=None, line=None):
    """Write req_<key>.a/.b/.root for VS_WORDS and `ps` under pipeline_key() with `masks`, and the stream
    description: `line` when given, else the line naming that key and `masks` (none without masks)."""
    key = G.pipeline_key(VS_WORDS, ps, root_blob(), ps_inputs=masks)
    name = cache / ('req_%016x' % key)
    for ext, blob in (('.b', ps), ('.root', root_blob()), ('.a', VS_WORDS)):
        name.with_suffix(ext).write_bytes(blob)
    description = line if line is not None else describe(key, masks) if masks else None
    if description is not None:
        name.with_suffix('.psin').write_bytes(description.encode('ascii'))
    return key


def describe(key, masks):
    return 'psin1 %016x ena=0x%04x addr=0x%04x\n' % (key, *masks)


@contextmanager
def fxc_setting(fxc=''):
    """X2W_FXC for the translator: by default the manifest-only path (''), where a published manifest is a
    finished result without bytecode; with an fxc path, the compile path."""
    previous = os.environ.get('X2W_FXC')
    os.environ['X2W_FXC'] = fxc
    try:
        yield
    finally:
        if previous is None:
            del os.environ['X2W_FXC']
        else:
            os.environ['X2W_FXC'] = previous


def pending(cache, fxc=''):
    """The keys a later --watch pass would translate (failures already written are skipped)."""
    with fxc_setting(fxc):
        return G.pending_keys(str(cache), True)


def watch_pass(cache):
    """One translate_pending pass as --watch runs it, on the manifest-only path."""
    with fxc_setting():
        G.translate_pending(str(cache), skip_failed=True, jobs=1)


def translated_ps(cache, key, through_pending=False):
    """translate_request with the manifest-only path: the pixel HLSL, or the refusal. With through_pending, a
    translate_pending pass (as --pending and --watch run it) translates whatever it finds pending instead."""
    with fxc_setting():
        if through_pending:
            G.translate_pending(str(cache), skip_failed=True, jobs=1)
            fail = cache / ('%016x.fail' % key)
            result = 'ok' if (cache / ('%016x.txt' % key)).is_file() else (
                'unsupported: ' + fail.read_text().strip() if fail.is_file() else 'not translated')
        else:
            result = G.translate_request(str(cache), key)
    if result != 'ok':
        return None, result
    return (cache / ('%016x.ps.hlsl' % key)).read_text(), result


def packed_line(text):
    found = re.findall(r'\bv(\d+) = \(uint\(i\.pos\.x\) & 0xffffu\)', text)
    return [int(n) for n in found]


def build_pair(cache, target=TARGET, decoy=DECOY):
    return put_request(cache, ps_words(4), target), put_request(cache, ps_words(2), decoy)


def controls():
    failures = []

    def check(label, ok):
        print(('ok   ' if ok else 'FAIL ') + label)
        if not ok:
            failures.append(label)

    def refused(label, cache, key, reason):
        text, result = translated_ps(cache, key)
        rows = G.request_refusals(str(cache), key)
        check(label, text is None and reason in result and any(reason in r for r, _ in rows))

    # The translator lays out by the address mask and loads by the enable mask.
    pipe = G.Pipeline(VS_WORDS, ps_words(4), root_blob(), ps_hw_inputs=TARGET)
    check('distinct masks: position after the allocated PERSP_SAMPLE and PERSP_CENTER pairs (v4)',
          pipe.packed_position_vgpr == 4 and pipe.persp_center_vgpr == 2)
    # No description (a bare request): the position register comes from the program's own 16-bit split of one live-in register.
    pipe = G.Pipeline(VS_WORDS, split_words(4), root_blob())
    text = pipe.hlsl_ps()
    check('a bare request that splits v4 takes the packed position from v4', pipe.packed_position_vgpr == 4
          and packed_line(text) == [4])
    check('another register: the same program shape on v3 takes v3',
          G.Pipeline(VS_WORDS, split_words(3), root_blob()).packed_position_vgpr == 3)
    check('a bare request with no half split evidence is left alone', G.Pipeline(VS_WORDS, ps_words(4), root_blob()).packed_position_vgpr is None
          and not packed_line(G.Pipeline(VS_WORDS, ps_words(4), root_blob()).hlsl_ps()))
    two = b''.join((vop2(0x1b, 1, 255, 4) + struct.pack('<I', 0xffff), vop2(0x16, 2, 144, 4),
                    vop2(0x1b, 6, 255, 3) + struct.pack('<I', 0xffff), vop2(0x16, 7, 144, 3),
                    struct.pack('<III', 0xf800180f, 0x01010101, 0xbf810000)))
    check('two split registers are ambiguous and leave the pipeline alone',
          G.Pipeline(VS_WORDS, two, root_blob()).packed_position_vgpr is None)
    check('published masks win over the program: the decoy masks still put the position in v2',
          G.Pipeline(VS_WORDS, split_words(4), root_blob(), ps_hw_inputs=DECOY).packed_position_vgpr == 2)
    pipe = G.Pipeline(VS_WORDS, ps_words(4), root_blob(), ps_hw_inputs=(0x0002, 0x8002))
    check('position allocated but not enabled is not modelled', pipe.packed_position_vgpr is None)
    for masks in ((0x8003, 0x8002), (0x8100, 0x8000)):
        try:
            G.Pipeline(VS_WORDS, ps_words(4), root_blob(), ps_hw_inputs=masks)
        except G.Unsupported as err:
            check('contradictory masks %04x/%04x refused by name' % masks,
                  'contradictory pixel input enable/address masks' in str(err))
        else:
            check('contradictory masks %04x/%04x refused by name' % masks, False)

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        t, d = build_pair(cache)
        text, _ = translated_ps(cache, t)
        check('target takes its own description: packed position in v4', text is not None and packed_line(text) == [4])
        text, _ = translated_ps(cache, d)
        check('decoy takes its own description: packed position in v2', text is not None and packed_line(text) == [2])
        check('refusal report sees no refusal for either request',
              G.request_refusals(str(cache), t) == [] and G.request_refusals(str(cache), d) == [])
        # An RTV variant reads its source pipeline's description.
        request = 'rtv1 %d 28' % t
        variant = G.fnv(request.encode('ascii'))
        (cache / ('req_%016x.rtv' % variant)).write_text(request, encoding='ascii')
        text, _ = translated_ps(cache, variant)
        check('RTV variant takes its source pipeline description: v4', text is not None and packed_line(text) == [4])

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        t = put_request(cache, ps_words(4))
        text, _ = translated_ps(cache, t)
        check('no captured description: no hardware inputs, position not modelled (today)',
              text is not None and packed_line(text) == [])

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        t, d = build_pair(cache)
        # The decoy's description copied over the target's.
        (cache / ('req_%016x.psin' % t)).write_bytes((cache / ('req_%016x.psin' % d)).read_bytes())
        refused('decoy description under the target name refused', cache, t,
                'pixel input description names another pipeline')
        # A stale description: the line names the key, but the stored pixel stage changed after capture.
        (cache / ('req_%016x.psin' % t)).write_bytes(describe(t, TARGET).encode('ascii'))
        (cache / ('req_%016x.b' % t)).write_bytes(ps_words(2))
        refused('stale description (stage blob changed) refused', cache, t,
                'pixel input description names another pipeline')

    for label, line, reason in (
        ('missing address mask', 'psin1 {k} ena=0x8002\n', 'pixel input address mask missing from the stream description'),
        ('missing enable mask', 'psin1 {k} addr=0x8003\n', 'pixel input enable mask missing from the stream description'),
        ('no masks', 'psin1 {k}\n', 'pixel input enable mask missing from the stream description'),
        ('contradictory masks', 'psin1 {k} ena=0x8003 addr=0x8002\n', 'contradictory pixel input enable/address masks'),
        ('duplicate mask', 'psin1 {k} ena=0x8002 ena=0x8002\n', 'invalid pixel input description'),
        ('malformed line', 'psin1 {k} ena=8002 addr=8003\n', 'invalid pixel input description'),
        ('unknown version', 'psin2 {k} ena=0x8002 addr=0x8003\n', 'invalid pixel input description'),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            masks = tuple(int(m, 16) for m in re.findall(r'(?:ena|addr)=(0x[0-9a-f]+)', line))
            masks = masks if len(masks) == 2 else None
            t = G.pipeline_key(VS_WORDS, ps_words(4), root_blob(), ps_inputs=masks)
            put_request(cache, ps_words(4), masks, line=line.format(k='%016x' % t))
            refused(label + ' refused by name', cache, t, reason)

    # One cache, pipelines with identical stage bytes, root signature and input layout but different masks.
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        bare = G.pipeline_key(VS_WORDS, ps_words(4), root_blob())
        check('without a description the key is unchanged',
              bare == G.fnv(root_blob(), G.fnv(ps_words(4), G.fnv(VS_WORDS, G.fnv(b'x2w-startup-shader-29')))))
        keys = [put_request(cache, ps_words(4), masks) for masks in (TARGET, SAME_V4, DECOY)]
        check('same stages, root and layout with three mask pairs: three request identities',
              len(set(keys)) == 3 and bare not in keys
              and all(len(list(cache.glob('req_%016x.*' % key))) == 4 for key in keys))
        texts = [translated_ps(cache, key)[0] for key in keys]
        # Under the third pair the position is v2, so the v4 this code reads is not the position.
        check('same cache: each pipeline takes its own masks (position v4, v4, not v4), each result under its own key',
              None not in texts and [packed_line(text) for text in texts] == [[4], [4], []]
              and all((cache / ('%016x.txt' % key)).is_file() for key in keys))

    # A different mask pair for the same stage bytes after a result exists.
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        t = put_request(cache, ps_words(4), TARGET)
        translated_ps(cache, t, through_pending=True)
        published = {ext: (cache / ('%016x%s' % (t, ext))).read_bytes() for ext in ('.txt', '.ps.hlsl')}
        w = put_request(cache, ps_words(4), DECOY)
        check('another mask pair after the result exists is a pending request of its own', pending(cache) == [w])
        text, _ = translated_ps(cache, w, through_pending=True)
        check('it is translated with its own masks (position not v4), the existing result untouched',
              text is not None and packed_line(text) == []
              and all((cache / ('%016x%s' % (t, ext))).read_bytes() == data for ext, data in published.items()))
        # The target's own description rewritten in place with the other pair, its line still naming the target.
        manifest = cache / ('%016x.txt' % t)
        stamp = manifest.stat().st_mtime - 100
        os.utime(manifest, (stamp, stamp))
        (cache / ('req_%016x.psin' % t)).write_bytes(describe(t, DECOY).encode('ascii'))
        check('a description rewritten after the result is published is checked again', pending(cache) == [t])
        text, result = translated_ps(cache, t, through_pending=True)
        check('the rewritten mask pair is refused by name and the old result withdrawn, not reused',
              text is None and 'pixel input description names another pipeline' in result and not manifest.is_file())
        check('the refusal is not retried while the request stays the same', pending(cache) == [])
        # A description rewritten with the same masks is checked once, then the result counts as done again.
        manifest = cache / ('%016x.txt' % w)
        stamp = manifest.stat().st_mtime - 100
        os.utime(manifest, (stamp, stamp))
        (cache / ('req_%016x.psin' % w)).write_bytes(describe(w, DECOY).encode('ascii'))
        first = pending(cache)
        text, _ = translated_ps(cache, w, through_pending=True)
        check('a description rewritten unchanged is checked once, then done',
              first == [w] and text is not None and packed_line(text) == [] and pending(cache) == [])

    # The watcher follows the description each translation reads: an RTV variant depends on its source's.
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        t = put_request(cache, ps_words(4), TARGET)
        request = 'rtv1 %d 28' % t
        v = G.fnv(request.encode('ascii'))
        (cache / ('req_%016x.rtv' % v)).write_text(request, encoding='ascii')
        for path in cache.glob('req_*'):  # captured well before anything below
            stamp = path.stat().st_mtime - 1000
            os.utime(path, (stamp, stamp))
        sources = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                   for path in (cache / ('req_%016x.a' % t), cache / ('req_%016x.rtv' % v))}
        watch_pass(cache)
        manifests = [cache / ('%016x.txt' % key) for key in (t, v)]
        description = cache / ('req_%016x.psin' % t)
        check('base and RTV variant published, nothing pending', all(m.is_file() for m in manifests) and pending(cache) == [])
        # The base description rewritten with the wrong pair (its line still naming the base), newer than both manifests.
        for path in manifests:
            if path.is_file():
                stamp = path.stat().st_mtime - 100
                os.utime(path, (stamp, stamp))
        description.write_bytes(describe(t, DECOY).encode('ascii'))
        check('a wrong pair in the base description queues the base and its RTV variant', pending(cache) == sorted([t, v]))
        watch_pass(cache)
        fails = [cache / ('%016x.fail' % key) for key in (t, v)]
        check('both refused by name and both published results withdrawn',
              not any(m.is_file() for m in manifests) and all(f.is_file() for f in fails)
              and 'pixel input description names another pipeline' in fails[0].read_text()
              and 'RTV variant source key mismatch' in fails[1].read_text())
        check('the refusals are not retried while the description stays the same', pending(cache) == [])
        # The valid description restored, newer than both failures.
        for path in fails:
            if path.is_file():
                stamp = path.stat().st_mtime - 100
                os.utime(path, (stamp, stamp))
        description.write_bytes(describe(t, TARGET).encode('ascii'))
        check('the restored description, newer than the failures, queues both again', pending(cache) == sorted([t, v]))
        watch_pass(cache)
        texts = [(cache / ('%016x.ps.hlsl' % key)).read_text() if m.is_file() else '' for key, m in zip((t, v), manifests)]
        check('both republished from the restored description (position v4), .a and .rtv untouched',
              all(m.is_file() for m in manifests) and [packed_line(text) for text in texts] == [[4], [4]]
              and all((path.read_bytes(), path.stat().st_mtime_ns) == state for path, state in sources.items())
              and pending(cache) == [])
        # Only the base description removed: both results were translated with it, so both are withdrawn.
        records = [cache / ('%016x.psdep' % key) for key in (t, v)]
        check('both results record the description line they were translated with',
              all(r.is_file() and r.read_text() == describe(t, TARGET) for r in records))
        # The failures of the wrong-pair cycle stay: on the manifest-only path a success does not remove them.
        description.unlink()
        check('a removed description queues the base and its RTV variant', pending(cache) == sorted([t, v]))
        watch_pass(cache)
        check('both refused by name as missing their description and both published results withdrawn',
              not any(m.is_file() for m in manifests) and all(f.is_file() for f in fails)
              and all('pixel input description missing for a request translated with one' in f.read_text()
                      for f in fails))
        check('the refusals are not retried while the description stays missing', pending(cache) == [])
        for path in fails:
            if path.is_file():
                stamp = path.stat().st_mtime - 100
                os.utime(path, (stamp, stamp))
        description.write_bytes(describe(t, TARGET).encode('ascii'))
        check('the description restored again queues both', pending(cache) == sorted([t, v]))
        watch_pass(cache)
        check('both republished from the restored description',
              all(m.is_file() for m in manifests) and pending(cache) == [])

    # A legacy request that never had a description: no dependency record, and its result stays done.
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        legacy = put_request(cache, ps_words(4))
        watch_pass(cache)
        check('a legacy request publishes without a dependency record and is not checked again',
              (cache / ('%016x.txt' % legacy)).is_file() and not (cache / ('%016x.psdep' % legacy)).exists()
              and pending(cache) == [])

    # Re-keying a cache keeps both pipelines and their descriptions, each naming its own key.
    with tempfile.TemporaryDirectory() as tmp:
        old, new = Path(tmp) / 'old', Path(tmp) / 'new'
        old.mkdir()
        keys = [put_request(old, ps_words(4), masks) for masks in (TARGET, DECOY)]
        import io
        import shader_cache_rekey
        status = shader_cache_rekey.rekey(str(old), str(new), jobs=1, out=io.StringIO())
        check('re-keying keeps both mask pairs as requests with their own descriptions',
              status == 0 and all((new / ('req_%016x.psin' % key)).read_bytes() == describe(key, masks).encode('ascii')
                                  for key, masks in zip(keys, (TARGET, DECOY)))
              and sorted(p.name for p in new.glob('req_*.psin')) == sorted('req_%016x.psin' % key for key in keys))

    if failures:
        print('%d control(s) failed' % len(failures))
        return 1
    print('controls: masks from the exact pipeline description; decoy, stale, missing and contradictory refused; '
          'masks in the request identity')
    return 0


def draw(dxc, renderer, adapter, fxc=None):
    compilers = [('dxc', dxc, '6_0')]
    if fxc:
        compilers.append(('fxc', fxc, '5_1'))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        cache = folder / 'cache'
        cache.mkdir()
        t, d = build_pair(cache)
        sources = {}
        for label, key in (('target', t), ('decoy', d)):
            text, result = translated_ps(cache, key)
            assert text is not None, (label, result)
            sources[label] = text
        # A request with no description at all (the runtime publishes none): the position register is taken from the program's
        # own 16-bit split of v4. With the decoy's masks (v2) the same code reads a register that holds no position.
        bare = put_request(cache, split_words(4))
        text, result = translated_ps(cache, bare)
        assert text is not None and packed_line(text) == [4], (result, text)
        sources['bare request, position derived from the program (v4)'] = text
        # After those results exist, two more pipelines with the target's exact stage bytes, root signature and
        # input layout arrive in the same cache: one whose masks also place the position in v4, and one whose
        # masks place it in v2 (wrong register for this code). Each is a request of its own and is translated.
        same = put_request(cache, ps_words(4), SAME_V4)
        wrong = put_request(cache, ps_words(4), DECOY)
        assert len({t, d, same, wrong}) == 4 and pending(cache) == sorted([same, wrong]), 'identities'
        for label, key in (('same bytes, other masks (v4)', same), ('same bytes, other masks (v2)', wrong)):
            text, result = translated_ps(cache, key, through_pending=True)
            assert text is not None, (label, result)
            sources[label] = text
        for kind, compiler, profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs', 'vs_' + profile, folder)
            for label, good in (('target', True), ('decoy', True), ('same bytes, other masks (v4)', True),
                                ('bare request, position derived from the program (v4)', True),
                                ('same bytes, other masks (v2)', False)):
                ps = compile_hlsl(compiler, kind, sources[label], 'ps', 'ps_' + profile, folder)
                run = subprocess.run([renderer, adapter, '--startup-packed-position', vs, ps],
                                     capture_output=True, text=True)
                if run.returncode == 77:
                    print(run.stdout + run.stderr)
                    return 77
                if good:
                    assert run.returncode == 0 and 'all 91 packed positions matched' in run.stdout, (label, run.stdout, run.stderr)
                else:
                    assert run.returncode != 0 and 'position mismatch (1,0): expected 00000001 actual 00000000' in run.stderr, (label, run.stderr)
                device = [line for line in run.stdout.splitlines() if line.startswith(('backend=', 'adapter=', 'debug_'))]
                print(f'{kind} {adapter} {label}: ' + ('91 pixels matched' if good else 'expected mismatch at (1,0)')
                      + ' [%s]' % '; '.join(device))
        if fxc:
            compiled_results(folder / 'compiled', fxc)
    return 0


def compiled_results(cache, fxc):
    """With bytecode (FXC), an unchanged manifest is not rewritten. A description rewritten unchanged is still
    checked once and then done, and one rewritten with another mask pair withdraws the published result."""
    cache.mkdir()
    t = put_request(cache, ps_words(4), TARGET)
    with fxc_setting(fxc):
        G.translate_pending(str(cache), skip_failed=True, jobs=1)
    manifest = cache / ('%016x.txt' % t)
    assert manifest.is_file() and (cache / ('%016x.ps.cso' % t)).is_file() and pending(cache, fxc) == [], 'compiled'
    for masks, done in ((TARGET, True), (DECOY, False)):
        stamp = manifest.stat().st_mtime - 100
        os.utime(manifest, (stamp, stamp))
        (cache / ('req_%016x.psin' % t)).write_bytes(describe(t, masks).encode('ascii'))
        assert pending(cache, fxc) == [t], ('checked again', masks)
        with fxc_setting(fxc):
            G.translate_pending(str(cache), skip_failed=True, jobs=1)
        assert pending(cache, fxc) == [] and manifest.is_file() == done, ('checked once', masks)
    assert 'pixel input description names another pipeline' in (cache / ('%016x.fail' % t)).read_text()
    print('fxc compiled result: rewritten description checked once; another mask pair withdraws the result')


def main(argv):
    if not argv:
        return controls()
    mode, *args = argv
    if mode == 'draw':
        status = controls()
        return status or draw(*args)
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
