"""Check the startup translator's diagnostic refusal census without game data or fxc."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []
FIRST, SECOND = 'unsupported_first', 'unsupported_second'
ENCODING = 'synthetic encoding refusal'
END = 0xbf810000


def check(label, condition):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)


def refusal(call):
    try:
        call()
    except G.Unsupported as reason:
        return str(reason)
    return None


def stage(refusals=None):
    s = G.Stage.__new__(G.Stage)
    s.kind, s.refusals = 'ps', refusals
    s.sym, s.lines, s.used_v = {}, [], set()
    s.exec_known = True
    s.ins = [(0, 4, FIRST, FIRST),
             (4, 4, 'v_mov_b32', 'v_mov_b32 v0, 0'),
             (8, 4, SECOND, SECOND)]
    return s


def check_stage():
    check('normal stage reports the first refusal', refusal(stage().translate) == FIRST)
    found = []
    s = stage(found)
    s.translate()
    check('diagnostic stage records both refusals in order', found == [(FIRST, 'ps'), (SECOND, 'ps')])
    check('supported instruction still emits HLSL', s.lines == ['v0 = 0x00000000u;'])

    def broken(self, ops, mods):
        raise TypeError('opaque handler detail')

    with patch.object(G.Stage, 'op_v_census_probe', broken, create=True):
        probe = (0, 4, 'v_census_probe', 'v_census_probe v0, v1')
        s = stage()
        s.ins = [probe]
        try:
            s.translate()
            check('normal stage re-raises a handler error', False)
        except TypeError:
            check('normal stage re-raises a handler error', True)
        for preceding, label in (([], 'translator crash'), ([(FIRST, 'decode')], 'internal error')):
            found = list(preceding)
            s = stage(found)
            s.ins = [probe, s.ins[-1]]
            s.translate()
            check('handler error classified as ' + label,
                  found == preceding + [(label + ' in v_census_probe: TypeError', 'ps'), (SECOND, 'ps')])


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for decoded instructions to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True


def vop3(op, srcs, vdst=4, sdst=0, neg=0, absolute=0, literal=None):
    """One VOP3 instruction: w0 = 0b110101 | op | abs | SDST (VOP3b) or VDST, w1 = three 9-bit sources | NEG."""
    w0 = (0b110101 << 26) | (op << 16) | (absolute << 8) | (sdst << 8) | vdst
    w1 = srcs[0] | (srcs[1] << 9) | (srcs[2] << 18) | (neg << 29)
    return struct.pack('<2I', w0, w1) + (b'' if literal is None else struct.pack('<I', literal))


def translate_one(code):
    """(decoded text, generated lines) of a single instruction, or the refusal text."""
    try:
        decoded = G.decode(code)
        assert len(decoded) == 1 and decoded[0][1] == len(code), decoded
        probe = Probe()
        probe.one(decoded[0][2], decoded[0][3])
        return decoded[0][3], probe.lines
    except G.Unsupported as reason:
        return str(reason)
    except (AttributeError, KeyError, IndexError, ValueError) as error:
        # A handler that needs operands the synthetic probe does not provide: not an accepted form.
        return 'probe error %s' % type(error).__name__


def check_neg_integer_sources():
    v = lambda n: 256 + n
    sentinels = [v(10), v(12), v(14)]
    inline = lambda n: 128 + n
    names = {}
    for op, entry in G.D.VOP3_NATIVE_OPS.items():
        names[entry[0]] = op
    cndmask = 0x101
    accepted = [op for op in range(0x400) if not isinstance(translate_one(vop3(op, sentinels)), str)]
    check('the sweep reaches the accepted VOP3 opcodes', len(accepted) > 100 and cndmask in accepted)
    mismatches = []
    for op in accepted:
        for slot in range(3):
            for source, literal, expected in ((inline(1), None, '0x80000001'), (255, 5, '0x80000005')):
                srcs = list(sentinels)
                srcs[slot] = source
                code = vop3(op, srcs, literal=literal)
                if literal is not None and G.D.decode_instruction(code, 0)[0] != len(code):
                    # An unused source field does not consume a literal dword.
                    continue
                plain = translate_one(code)
                if isinstance(plain, str) or (op == cndmask and slot < 2):
                    continue
                result = translate_one(vop3(op, srcs, neg=1 << slot, literal=literal))
                name = plain[0].split()[0].removesuffix('_e64')
                compare = name.startswith(('v_cmp_', 'v_cmpx_')) and name.endswith('_f32')
                float_slot = (name in G.VEC_F and slot < G.VEC_F[name][0]
                              or compare and slot < (1 if '_class_' in name else 2))
                if float_slot:
                    # Always/never compares consume no source in the emitted predicate.
                    constant_compare = name.split('_')[-2] in ('t', 'f')
                    ok = (not isinstance(result, str) and expected in result[0]
                          and (constant_compare or expected + 'u' in ' '.join(result[1])))
                else:
                    # Dedicated encoding refusals run before the general NEG check.
                    ok = (isinstance(result, str) and ('NEG on an integer or literal source %d' % slot in result
                                                       or ' with neg/abs' in result))
                if not ok:
                    mismatches.append((hex(op), slot, expected, result))
    check('VOP3 NEG on inline 1 and a literal follows the source type in every lowerable slot', not mismatches)
    if mismatches:
        print('  first misses: %s' % mismatches[:4])

    def refused(code, slot):
        text = translate_one(code)
        return isinstance(text, str) and 'NEG on an integer or literal source %d' % slot in text

    iadd, fadd, add_co, shl64 = names['v_mad_u32_u24'], 0x103, names['v_add_co_u32'], names['v_lshlrev_b64']
    cmp_lt = next(op for op, entry in G.D.VOPC_OPS.items() if entry[0] == 'v_cmp_lt_f32_e32')
    for slot in range(3):
        srcs = list(sentinels)
        srcs[slot] = inline(1)
        check('v_mad_u32_u24 slot %d: inline 1 with NEG refused' % slot,
              refused(vop3(iadd, srcs, neg=1 << slot), slot))
        check('v_mad_u32_u24 slot %d: the same word without NEG still translates' % slot,
              not isinstance(translate_one(vop3(iadd, srcs)), str))
    check('inline 0 with NEG refused', refused(vop3(iadd, [inline(0), sentinels[1], sentinels[2]], neg=1), 0))
    check('inline 64 with NEG refused', refused(vop3(iadd, [inline(64), sentinels[1], sentinels[2]], neg=1), 0))
    check('literal with NEG refused', refused(vop3(iadd, [255, sentinels[1], sentinels[2]], neg=1, literal=5), 0))
    check('literal without NEG translates', not isinstance(
        translate_one(vop3(iadd, [255, sentinels[1], sentinels[2]], literal=5)), str))
    def exact(code, word):
        result = translate_one(code)
        return not isinstance(result, str) and 'F(0x%08xu)' % word in ' '.join(result[1])

    check('v_add_f32_e64 inline 0 with NEG is negative zero',
          exact(vop3(fadd, [inline(0), v(12), 0], neg=1), 0x80000000))
    check('v_add_f32_e64 inline 64 with NEG stays subnormal',
          exact(vop3(fadd, [inline(64), v(12), 0], neg=1), 0x80000040))
    check('v_add_f32_e64 NEG flips the sign of an encoded NaN payload',
          exact(vop3(fadd, [255, v(12), 0], neg=1, literal=0xffc12345), 0x7fc12345))
    check('v_add_f32_e64 immediate ABS with NEG is refused',
          refused(vop3(fadd, [inline(1), v(12), 0], absolute=1, neg=1), 0))
    word0, word1 = struct.unpack('<2I', vop3(fadd, [v(10), inline(1), 0], neg=2))
    # The rewritten final operand must keep both supported and refused output modifiers.
    clamp = translate_one(struct.pack('<2I', word0 | (1 << 15), word1))
    check('rewriting the last source preserves clamp',
          not isinstance(clamp, str) and 'saturate(F(v10) + F(0x80000001u))' in ' '.join(clamp[1]))
    op_sel = translate_one(struct.pack('<2I', word0 | (1 << 11), word1))
    check('rewriting the last source preserves the op_sel refusal',
          isinstance(op_sel, str) and 'modifier op_sel' in op_sel)
    check('float compare slot 0 with inline 1 and NEG reads its word',
          exact(vop3(cmp_lt, [inline(1), v(12), 0], neg=1), 0x80000001))
    check('float compare slot 1 with literal and NEG reads its word',
          exact(vop3(cmp_lt, [v(10), 255, 0], neg=2, literal=7), 0x80000007))
    check('class compare slot 0 with literal and NEG reads its word',
          exact(vop3(0x88, [255, inline(1), 0], neg=1, literal=7), 0x80000007))
    check('class compare slot 1 with literal and NEG is refused',
          refused(vop3(0x88, [v(10), 255, 0], neg=2, literal=7), 1))
    check('integer compare slot 0 with inline and NEG is refused',
          refused(vop3(0x81, [inline(1), v(12), 0], neg=1), 0))
    check('v_add_co_u32 (carry out) with inline 1 and NEG refused',
          refused(vop3(add_co, [inline(1), v(12), 0], sdst=4, neg=1), 0))
    check('v_lshlrev_b64 with NEG on the shift inline refused', refused(vop3(shl64, [inline(1), v(12), 0], neg=1), 0))

    # Negative controls.
    minus_one = translate_one(vop3(iadd, [193, sentinels[1], sentinels[2]]))
    check('plain inline -1 still reads 0xffffffff',
          not isinstance(minus_one, str) and '0xffffffffu' in ' '.join(minus_one[1]))
    negative = translate_one(vop3(iadd, [193, sentinels[1], sentinels[2]], neg=1))
    check('inline -1 with NEG is not refused by this rule',
          not (isinstance(negative, str) and 'NEG on an integer' in negative))
    for code in range(240, 249):
        result = translate_one(vop3(fadd, [code, v(12), 0], neg=1))
        check('inline float %d with NEG is not refused by this rule' % code,
              not (isinstance(result, str) and 'NEG on an integer' in result))
    check('v_add_co_u32 with sdst s4 and register sources translates',
          not isinstance(translate_one(vop3(add_co, [v(10), v(12), 0], sdst=4)), str))
    check('v_add_co_u32 with sdst s4 and an inline source translates',
          not isinstance(translate_one(vop3(add_co, [inline(1), v(12), 0], sdst=4)), str))
    for slot in range(2):
        srcs = [v(10), v(12), 106]
        srcs[slot] = inline(1)
        result = translate_one(vop3(cndmask, srcs, neg=1 << slot))
        check('v_cndmask_b32_e64 slot %d: inline 1 with NEG still translates' % slot,
              not isinstance(result, str) and '0x80000001u' in ' '.join(result[1]))
    check('v_cndmask_b32_e64 mask slot: inline 1 with NEG is refused',
          refused(vop3(cndmask, [v(10), v(12), inline(1)], neg=4), 2))


def check_decode():
    code = struct.pack('<2I', 0, END)
    with patch.object(G.D, 'decode_instruction', decode_instruction), patch.object(
            G, 'check_encoding', side_effect=[G.Unsupported(ENCODING), 's_endpgm']):
        found = []
        ins = G.decode(code, found)
        check('encoding refusal carries decode scope', found == [(ENCODING, 'decode')])
        check('decode resumes at the next instruction', [row[:3] for row in ins] == [(4, 4, 's_endpgm')])

    # GFX10 VOP3 v_cndmask: inline 1 (129), v1 (257), vcc (106), NEG on src0.
    word1 = 129 | (257 << 9) | (106 << 18) | (1 << 29)
    code = struct.pack('<3I', 0xd5010000, word1, END)
    found = []
    ins = G.decode(code, found)
    check('upstream NEG lowering is retained', not found and '0x80000001' in ins[0][3])


def decode_instruction(blob, offset):
    instructions = {
        0: ('s_nop', 's_nop 0'),
        1: ('exp', 'exp pos0 off, off, off, off done'),
        2: (FIRST, FIRST),
        3: ('v_mov_b32', 'v_mov_b32 v0, 0'),
        4: (SECOND, SECOND),
        END: ('s_endpgm', 's_endpgm'),
    }
    name, asm = instructions[struct.unpack_from('<I', blob, offset)[0]]
    return 4, name, asm, None


def cli(cache, *flags):
    out = io.StringIO()
    with patch.object(sys, 'argv', ['gfx2hlsl.py', *flags, '--refusals', str(cache)]), contextlib.redirect_stdout(out):
        assert G.main() is None
    return out.getvalue()


def put_request(cache, vs, ps, root):
    key = G.pipeline_key(vs, ps, root)
    for suffix, blob in (('.a', vs), ('.b', ps), ('.root', root)):
        (cache / ('req_%016x%s' % (key, suffix))).write_bytes(blob)
    return key


def root_blob(flags=0):
    return b'DXBC' + b'\0' * 24 + struct.pack('<2I', 1, 36) + b'RTS0' + struct.pack('<7I', 24, 1, 0, 0, 0, 0, flags)


class PublishingEntered(BaseException):
    pass


def check_cli():
    vs = struct.pack('<2I', 1, END)
    ps = struct.pack('<4I', 2, 3, 4, END)
    missing_base = 0xabcdef0123456789
    with tempfile.TemporaryDirectory() as work, patch.object(G.D, 'decode_instruction', decode_instruction):
        cache = Path(work)
        base = put_request(cache, vs, ps, root_blob())
        variant_request = ('rtv1 %d 28' % base).encode('ascii')
        variant = G.fnv(variant_request)
        dangling_request = ('rtv1 %d 28' % missing_base).encode('ascii')
        dangling = G.fnv(dangling_request)
        (cache / ('req_%016x.rtv' % variant)).write_bytes(variant_request)
        (cache / ('req_%016x.rtv' % dangling)).write_bytes(dangling_request)
        (cache / ('%016x.txt' % base)).write_bytes(b'previous success')
        (cache / ('%016x.fail' % variant)).write_bytes(b'previous failure')
        before = {p.name: p.read_bytes() for p in cache.iterdir()}

        with patch.object(G, 'translate_to', side_effect=PublishingEntered):
            with patch.dict(os.environ, {'X2W_REPLAY_SHA': 'replay-from-env'}):
                report_json = cli(cache, '--json', '--replay-sha', 'replay-from-flag')
                report_text = cli(cache)
            report = json.loads(report_json)
            translator = G.translator_version()[:12]
            seed = G.key_seed_line()
            check('matching unstamped cache reports JSON header and the replay flag takes precedence',
                  report['header'] == dict(translator=translator, cache_key_seed='unstamped',
                                           translator_key_seed=seed, replay='replay-from-flag'))
            check('text header precedes the report and uses the environment replay', report_text.startswith(
                f'Translator: {translator}\nCache key seed: unstamped\nTranslator key seed: {seed}\n'
                'Replay: replay-from-env\nPer request key\n'))
            requests = report['per_request']
            check('base, RTV variant and dangling variant are each counted', requests['examined'] == 3)
            check('histogram counts sites per request', requests['no_refusal'] == 0 and
                  requests['refusal_sites_per_request'] == [dict(sites=1, requests=1), dict(sites=2, requests=2)])
            check('request rows include scope, containing, first and only', requests['reasons'] == [
                dict(reason=FIRST, scope='ps', containing=2, first=2, only=0),
                dict(reason=SECOND, scope='ps', containing=2, first=0, only=0),
                dict(reason='unreadable shader request', scope='pipeline', containing=1, first=1, only=1),
            ])
            check('RTV variants do not add code pairs or roots', report['per_code_pair'] == dict(examined=1, reasons=[
                dict(reason=FIRST, scope='ps', under_every_root=1, under_some_roots=0),
                dict(reason=SECOND, scope='ps', under_every_root=1, under_some_roots=0),
            ]))
            check('text reports the request, site and code pair views',
                  'Requests examined: 3' in report_text and 'Code pairs examined: 1' in report_text and
                  'Refusal sites per request' in report_text and '2\t2\t0\tps\t' + FIRST in report_text and
                  'unreadable shader request' in report_text)
            check('neither report leaks request keys, paths or raw disassembly',
                  all(str(cache) not in output and str(cache).replace('\\', '\\\\') not in output and
                      all('%016x' % key not in output for key in (base, variant, dangling, missing_base)) and
                      'exp pos0' not in output for output in (report_text, report_json)))

            translate = G.Stage.translate

            def stop_at_first(s):
                s.refusals = None
                translate(s)

            with patch.object(G.Stage, 'translate', stop_at_first):
                control = json.loads(cli(cache, '--json'))['per_request']
            check('disabling skip removes the second reason from the report', control['reasons'] == [
                dict(reason=FIRST, scope='pipeline', containing=2, first=2, only=2),
                dict(reason='unreadable shader request', scope='pipeline', containing=1, first=1, only=1),
            ])

            missing = cache / 'nonexistent'
            try:
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    cli(missing, '--json')
                check('missing directory fails with a parser error', False)
            except SystemExit as error:
                check('missing directory fails with a parser error', error.code == 2 and 'existing directory' in stderr.getvalue())
            check('missing directory is never created', not missing.exists())

            # Every spelling and order of the flag stays in the diagnostic and writes nothing.
            out = io.StringIO()
            with patch.object(sys, 'argv', ['gfx2hlsl.py', '--refusals', str(cache), '--json']), \
                    contextlib.redirect_stdout(out):
                G.main()
            check('flag after the directory is accepted', json.loads(out.getvalue())['per_request']['examined'] == 3)
            out = io.StringIO()
            with patch.object(sys, 'argv', ['gfx2hlsl.py', '--refusals=' + str(cache)]), \
                    contextlib.redirect_stdout(out):
                G.main()
            check('--refusals=DIR stays in the diagnostic', 'Requests examined: 3' in out.getvalue())

        check('diagnostics leave cache filenames and contents unchanged',
              {p.name: p.read_bytes() for p in cache.iterdir()} == before)
        for key in (base, variant):
            check('normal request publishes the upstream first-refusal result',
                  G.translate_request(work, key) == 'unsupported: ' + FIRST and
                  (cache / ('%016x.fail' % key)).read_text() == FIRST + '\n')
        try:
            with open(os.path.join(work, 'req_%016x.a' % missing_base), 'rb'):
                pass
        except OSError as error:
            check('normal dangling RTV retains the upstream raw file error',
                  G.translate_request(work, dangling) == 'unsupported: ' + str(error) and
                  (cache / ('%016x.fail' % dangling)).read_text() == str(error) + '\n')
        else:
            check('normal dangling RTV retains the upstream raw file error', False)


def check_request_failures():
    vs = struct.pack('<2I', 1, END)
    ps = struct.pack('<2I', 0, END)
    with tempfile.TemporaryDirectory() as work, patch.object(G.D, 'decode_instruction', decode_instruction):
        cache = Path(work)
        base = put_request(cache, vs, ps, root_blob())
        stages = put_request(cache, ps, ps, root_blob())
        truncated = put_request(cache, vs, struct.pack('<I', 0) + b'\xff', root_blob())
        undecodable = put_request(cache, struct.pack('<2I', 0, 255), ps, root_blob())
        # The census follows the translator: a short tail is accepted on both paths, and a
        # disassembler struct.error is the same undecodable-word refusal the default path writes.
        census = G.request_refusals(cache, truncated)
        check('short tail is not a census-only refusal',
              all('truncated' not in reason and 'invalid binary' not in reason for reason, _ in census))
        default = G.translate_request(work, truncated)
        check('short tail: census first site matches the default result',
              (census[0][0] if census else 'ok') == default.replace('unsupported: ', '', 1))
        with patch.object(G.D, 'decode_instruction', side_effect=struct.error('short instruction')):
            found = G.request_refusals(cache, base)
            check('disassembler struct.error is the default undecodable-word refusal',
                  [(G.normalize_refusal(reason), scope) for reason, scope in found] == [('undecodable word', 'pipeline')])
        found = G.request_refusals(cache, undecodable)
        check('undecodable word before exports ends the pipeline',
              [(G.normalize_refusal(reason), scope) for reason, scope in found] == [('undecodable word', 'pipeline')])
        check('bad stage set is a pipeline refusal', G.request_refusals(cache, stages) == [("stages ['ps']", 'pipeline')])
        check('normal bad stage set writes upstream fail text',
              G.translate_request(work, stages) == "stages ['ps']" and
              (cache / ('%016x.fail' % stages)).read_text() == "stages ['ps']\n")
        with patch.object(G.Pipeline, 'manifest', side_effect=TypeError('opaque pipeline detail')):
            check('unexpected pipeline error without an earlier refusal is a crash',
                  G.request_refusals(cache, base) == [('translator crash in manifest: TypeError', 'pipeline')])
        broken = put_request(cache, vs, struct.pack('<2I', 2, END), root_blob())
        with patch.object(G.Pipeline, 'manifest', side_effect=TypeError('opaque pipeline detail')):
            check('unexpected pipeline error after a refusal is a follow-on row',
                  G.request_refusals(cache, broken) == [(FIRST, 'ps'), ('internal error in manifest: TypeError', 'pipeline')])
        with patch.object(G, 'load_request', side_effect=RuntimeError('opaque loader detail')):
            check('unexpected loader error has pipeline scope',
                  G.request_refusals(cache, base) == [('translator crash in request loading: RuntimeError', 'pipeline')])


def check_pairs():
    vs = struct.pack('<2I', 1, END)
    ps = struct.pack('<3I', 2, 2, END)
    with tempfile.TemporaryDirectory() as work, patch.object(G.D, 'decode_instruction', decode_instruction):
        cache = Path(work)
        put_request(cache, vs, ps, root_blob())
        put_request(cache, vs, ps, root_blob(0x400))
        clean = put_request(cache, vs, struct.pack('<I', END), root_blob())
        translate_one = G.Stage.one

        def root_dependent(s, name, asm):
            if name == FIRST and s.pipe.root['flags'] & 0x400:
                raise G.Unsupported('root-dependent reason')
            return translate_one(s, name, asm)

        with patch.object(G.Stage, 'one', root_dependent):
            report = G.refusals_report(cache)
        check('a single reason at multiple sites still counts as only',
              report['per_request']['reasons'] == [
                  dict(reason='root-dependent reason', scope='ps', containing=1, first=1, only=1),
                  dict(reason=FIRST, scope='ps', containing=1, first=1, only=1),
              ])
        check('same stage bytes under two roots count as one pair with root-dependent reasons',
              report['per_code_pair'] == dict(examined=2, reasons=[
                  dict(reason='root-dependent reason', scope='ps', under_every_root=0, under_some_roots=1),
                  dict(reason=FIRST, scope='ps', under_every_root=0, under_some_roots=1),
              ]))
        check('a clean request enters the zero bucket', not G.request_refusals(cache, clean) and
              report['per_request']['no_refusal'] == 1)


def check_key_seed_mismatch():
    vs = struct.pack('<2I', 1, END)
    ps = struct.pack('<I', END)
    with tempfile.TemporaryDirectory() as work, patch.dict(os.environ, {'X2W_EXTENDED_BUFFER_FORMATS': '1'}):
        cache = Path(work)
        stamp = cache / G.KEY_SEED_NAME
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            G.translate_pending(cache, jobs=1)
        ours = G.key_seed_line()
        check('pending stamps a fresh cache quietly', stamp.read_text(encoding='ascii') == ours + '\n'
              and not stderr.getvalue())

        stamp.unlink()
        with patch.dict(os.environ, {'X2W_EXTENDED_BUFFER_FORMATS': '0'}):
            theirs = G.key_seed_line()
            old_key = put_request(cache, vs, ps, root_blob())
        stamp.write_text(theirs + '\nextra line\n', encoding='ascii')
        check('key seed reader returns the first line', G.read_key_seed(cache) == theirs)
        argv = [sys.executable, '-B', G.__file__, '--refusals', str(cache), '--json']
        env = {key: value for key, value in os.environ.items() if key != 'X2W_REPLAY_SHA'}
        result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60)
        check('census exits 3 without a report for a request written with another seed',
              result.returncode == 3 and not result.stdout and
              'refusals: 1 of 1 requests (100.0%) are request-key mismatches' in result.stderr and
              theirs in result.stderr and ours in result.stderr and
              f'tools/startup/shader_cache_rekey.py {cache} <new dir>' in result.stderr and
              '--allow-key-mismatch reports anyway' in result.stderr)

        result = subprocess.run([*argv, '--allow-key-mismatch'], capture_output=True, text=True, env=env, timeout=60)
        report = json.loads(result.stdout) if result.returncode == 0 else {}
        reason = 'shader request source key mismatch'
        check('override reports the request-key refusal and identifies the old cache seed',
              result.returncode == 0 and not result.stderr and report['header']['cache_key_seed'] == theirs and
              report['header']['replay'] == 'unknown' and G.is_key_mismatch(G.Unsupported(reason)) and
              report['per_request']['reasons'] == [dict(reason=reason, scope='pipeline', containing=1, first=1, only=1)])

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            G.stamp_key_seed(cache)
        check('stamp warns once and preserves the existing different seed',
              stderr.getvalue().count('x2w-gfx key seed mismatch:') == 1 and
              stamp.read_text(encoding='ascii') == theirs + '\nextra line\n')

        stamp.unlink()
        current_key = put_request(cache, vs, ps, root_blob())
        stderr = io.StringIO()
        with patch.object(G.D, 'decode_instruction', decode_instruction), contextlib.redirect_stderr(stderr):
            G.translate_pending(cache, jobs=1)
        warning = stderr.getvalue()
        check('pending warns once for an unstamped cache with stale keys and still translates current requests',
              warning.count('x2w-gfx key seed mismatch:') == 1 and
              '1 of 2 requests (50.0%)' in warning and "(cache seed: 'unstamped')" in warning and
              (cache / ('%016x.fail' % old_key)).read_text() == reason + '\n' and
              (cache / ('%016x.txt' % current_key)).is_file())


def check_normalization():
    for raw, normalized in (
            ('undecodable word at +0x10: exp v3', 'undecodable word'),
            ('invalid stream shader: opaque stream 0x01234567', 'invalid stream shader'),
            ('unverified vertex fetch record deadbeef00', 'unverified vertex fetch record'),
            ('unreadable shader request C:/cache/req_deadbeef', 'unreadable shader request'),
            ('input layout format 28', 'input layout format 28'),
            ('64-bit shift amount 32', '64-bit shift amount N'),
            ('s_load of 4 dwords from (2, 3)', 's_load of 4 dwords from (N, N)'),
            ('SampleCmpLevel needs shader model 6.7', 'SampleCmpLevel needs shader model 6.7'),
            ('s_load_dword s[4:7], v8, 0x10, 16, -1.5', 's_load_dword s[N:N], vN, N, N, N'),
            ('operand s4, v[2:5]', 'operand sN, v[N:N]'),
            ('word_0xdeadbeef', 'word_N'),
            # A bare literal outside an operand position: the operand rule above does not reach it.
            ('v_mov_b32 with literal 0x3f800000', 'v_mov_b32 with literal N'),
            ('unexpected value 0x7fc00000 in source', 'unexpected value N in source')):
        check('normalization of ' + normalized, G.normalize_refusal(raw) == normalized)


if __name__ == '__main__':
    # An empty X2W_FXC publishes the manifest only, so no default-path call here ever starts fxc
    # and the check behaves the same on hosts with and without the Windows SDK.
    with patch.dict(os.environ, {'X2W_FXC': ''}):
        check_stage()
        check_decode()
        check_neg_integer_sources()
        check_cli()
        check_request_failures()
        check_pairs()
        check_key_seed_mismatch()
        check_normalization()
    sys.exit(bool(failures))
