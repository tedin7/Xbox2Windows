"""64-bit VOPC controls; draw <dxc> <offscreen_check> <warp|hardware> [fxc]."""
import hashlib
import hlsl_pin  # noqa: E402
import re
import sys
import tempfile
from pathlib import Path

import startup_gfx2hlsl_cmp16_check as S

G, D, I3 = S.G, S.D, S.I3
OPS = {}
for code, (name, _) in D.VOPC_OPS.items():
    match = re.fullmatch(r'v_cmp(x?)_(\w+)_([iu]64)_e32', name)
    if match:
        OPS[bool(match[1]), match[2], match[3]] = code

VALUES = [0, 1, 0xffffffff, 0x100000000, 0xffffffff00000000,
          0x8000000000000000, 0x7fffffffffffffff, 0xffffffffffffffff]

def signed(n):
    return n - (1 << 64) if n & (1 << 63) else n

def evaluate(expr, values):
    expr = re.sub(r'0x([0-9a-f]+)u', r'0x\1', expr)
    expr = expr.replace('&&', ' and ').replace('||', ' or ')
    expr = re.sub(r'!(?!=)', ' not ', expr)
    expr = expr.replace('asint(', 'i32(')
    # Strip the outer HLSL ternary; comparisons themselves use only boolean operators.
    assert expr.startswith('((') and expr.endswith(') ? 1u : 0u)'), expr
    expr = expr[2:-len(') ? 1u : 0u)')]
    env = dict(values, i32=lambda n: n - (1 << 32) if n & 0x80000000 else n)
    return int(eval(expr, {'__builtins__': {}}, env))

def controls():
    for typ in ('i64', 'u64'):
        for cond in S.CONDITIONS:
            for cmpx in (False, True):
                for encoding in ('e32', 'e64'):
                    op = OPS[cmpx, cond, typ]
                    code = (S.e32(op, S.V(2), 4) if encoding == 'e32' else
                            S.e64(op, 4, S.V(2), S.V(4)))
                    name, asm = S.decode_one(code)
                    stage = S.Probe()
                    stage.one(name, asm)
                    line = stage.lines[0]
                    expr = line.split(' = ', 1)[1][:-1]
                    for a in VALUES:
                        for b in VALUES:
                            env = {'v2': a & S.MASK, 'v3': a >> 32, 'v4': b & S.MASK, 'v5': b >> 32}
                            want = S.RELATION[cond](signed(a) if typ == 'i64' else a,
                                                    signed(b) if typ == 'i64' else b)
                            assert evaluate(expr, env) == want, (asm, a, b, expr)
                    if cmpx and encoding == 'e64':
                        assert len(stage.lines) == 1 and stage.lines[0].startswith('exec = '), stage.lines
                        assert 4 not in stage.masks and 4 not in stage.used_s, stage.lines
    for dst in (0, 4, S.VCC_LO):
        code = S.e64(OPS[True, 'ne', 'i64'], dst, S.V(2), S.V(4))
        name, asm = S.decode_one(code)
        stage = S.Probe()
        stage.one(name, asm)
        assert len(stage.lines) == 1 and stage.lines[0].startswith('exec = '), stage.lines
        assert not stage.masks and not stage.used_s, stage.lines
    for typ in ('i64', 'u64'):
        for constant in (-16, -1, 0, 1, 64):
            code = S.e32(OPS[False, 'lt', typ], S.inline(constant), 4)
            _, _, expr = S.translate(code)
            a = constant & ((1 << 64) - 1)
            for b in VALUES:
                env = {'v4': b & S.MASK, 'v5': b >> 32}
                assert evaluate(expr, env) == ((signed(a) < signed(b)) if typ == 'i64' else (a < b))
    for typ in ('i64', 'u64'):
        for source, value in ((6, 0xffffffff00000000), (S.LITERAL, 0x7fffffff)):
            code = S.e32(OPS[False, 'gt', typ], source, 4, value if source == S.LITERAL else None)
            _, _, expr = S.translate(code)
            for b in VALUES:
                a = value
                env = {'s6': a & S.MASK, 's7': a >> 32, 'v4': b & S.MASK, 'v5': b >> 32}
                assert evaluate(expr, env) == ((signed(a) > signed(b)) if typ == 'i64' else (a > b))
    code = S.e32(OPS[True, 'ne', 'i64'], S.V(2), 4)
    _, _, expr = S.translate(code, exec_known=False)
    assert expr.startswith('(exec & '), expr
    for typ in ('i64', 'u64'):
        for token in ('0x80000000', '2147483648', '0xffffffff', '4294967295'):
            reason = S.refusal(name='v_cmp_ne_%s_e32' % typ,
                               asm='v_cmp_ne_%s_e32 vcc_lo, %s, v[4:5]' % (typ, token))
            assert reason == 'v_cmp_ne_%s source %s: ambiguous 64-bit literal' % (typ, token), reason
        for token in ('0x7fffffff', '2147483647'):
            stage = S.Probe()
            stage.one('v_cmp_eq_%s_e32' % typ, 'v_cmp_eq_%s_e32 vcc_lo, %s, v[4:5]' % (typ, token))
            expr = stage.lines[0].split(' = ', 1)[1][:-1]
            assert evaluate(expr, {'v4': 0x7fffffff, 'v5': 0}) == 1
            assert evaluate(expr, {'v4': 0x7fffffff, 'v5': 0xffffffff}) == 0
    for token in ('0x80000000', 'v2', 'vcc', 'neg(v[2:3])'):
        reason = S.refusal(name='v_cmp_ne_i64_e32', asm='v_cmp_ne_i64_e32 vcc_lo, %s, v[4:5]' % token)
        assert reason and ('source' in reason or 'operand' in reason), (token, reason)
    for mod in ('op_sel:[1,0,0]', 'clamp'):
        reason = S.refusal(name='v_cmp_ne_i64_e64', asm='v_cmp_ne_i64_e64 s4, v[2:3], v[4:5] ' + mod)
        assert reason and 'with' in reason, reason
    assert S.refusal(name='v_cmp_ne_i64_sdwa', asm='v_cmp_ne_i64_sdwa vcc_lo, v2, v4') == 'v_cmp_ne_i64_sdwa'
    old = S.e32(S.OPCODES[False, 'lt', 'i16'], S.V(1), 2)
    assert S.translate(old)[2] == '(((asint(v1 << 16u) >> 16u) < (asint(v2 << 16u) >> 16u)) ? 1u : 0u)'
    pipeline = G.Pipeline(I3.VS_WORDS, S.K_WORDS, I3.root_blob(), ps_hw_inputs=S.PACKED)
    assert hlsl_pin.digest(pipeline.hlsl_ps()) == 'bf67b970cf344cfbdce23a79f132724226821df1102251465962962c801dc090'
    assert hashlib.sha256(pipeline.manifest().encode()).hexdigest() == '43ded0ab436e974b155ca5b6e9248c264be3dd66388e9ee8b65d5e0312032c92'
    assert sum(a != b for a, b in zip(image(), image(True))) > 0
    assert 'exec = ' in G.Pipeline(I3.VS_WORDS, WORDS, I3.root_blob(), ps_hw_inputs=S.PACKED).hlsl_ps()
    print('64-bit compare controls PASS')

# v2:v3 = {0, packed screen position}; the first pixel alone is discarded.
WORDS = (bytes.fromhex('8002047e0003067e') + S.e32(OPS[True, 'ne', 'i64'], S.inline(0), 2)
         + bytes.fromhex('f20006380f1800f803030303000081bf'))

def image(wrong=False):
    words = []
    for y in range(S.HEIGHT):
        for x in range(S.WIDTH):
            p = x | y << 16
            pixel = (S.ONE | p) if (False if wrong else p != 0) else S.CLEAR
            words.extend([pixel] * 4)
    return words

def draw(dxc, renderer, adapter, fxc=None):
    text = G.Pipeline(I3.VS_WORDS, WORDS, I3.root_blob(), ps_hw_inputs=S.PACKED).hlsl_ps()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        reference = S.words_file(directory, 'cmp64', image())
        for kind, compiler, vp, pp in (([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) +
                                       [('dxc', dxc, 'vs_6_0', 'ps_6_0')]):
            vs = I3.compile_hlsl(compiler, kind, I3.FULLSCREEN_VS, 'vs', vp, directory)
            ps = I3.compile_hlsl(compiler, kind, text, 'ps', pp, directory)
            run = S.draw(renderer, adapter, vs, ps, reference)
            print(run.stdout + run.stderr)
            assert run.returncode == 0 and 'all 256 pixels matched' in run.stdout
    print('64-bit compare draw PASS')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif sys.argv[1] == 'draw' and len(sys.argv) in (5, 6):
        controls()
        draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
