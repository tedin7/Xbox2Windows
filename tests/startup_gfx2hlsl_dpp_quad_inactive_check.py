"""A DPP16 quad read inside an if (EXEC narrowed by s_and_saveexec): the contract a lane-exact EXEC model in
tools/startup/gfx2hlsl.py must meet, read back from a real D3D12 draw.

  (no arguments)                       translation state and CPU reference controls, pure Python
  binary <llvm-mc>                     assemble the fixture sources with LLVM (gfx1013) and compare with the words here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw 16 x 16 pixels with (a) hand-written HLSL that follows the contract, which
                                       must match the CPU reference, (b) wrong lowerings, which must not, and (c) the
                                       translator's own lowering once it produces one; DXC ps_6_0 (quad reads need SM 6.0;
                                       the optional fxc argument is accepted and ignored)

The pixel stage (both fixtures):
  s_wqm_b64 exec, exec                                 whole quads on
  v1 = (x | y << 16) * 0x9b5d ; v2 = 0x5a5a
  if (v1 & 0x80) {                                     v_cmp_ne_u32 + s_and_saveexec_b64: a lane mask per pixel
    v2 = v_min_u32_dpp(v1 quad_perm:[1,0,3,2], v1)     src0 from the pixel at x ^ 1, src1 the lane's own v1
  }                                                    s_or_b64 exec, exec, s[4:5]
  exp mrt0 v2, v1, v0, v0

The contract (AMD RDNA 2 ISA reference, DPP16 instruction fields FI and BOUND_CTRL, decoded from the instruction's own
bits: FI is bit 18 and BOUND_CTRL bit 19 of the DPP word): per lane src_valid = in range (a quad swap always is) and
(FI = 1 or the source lane is enabled by EXEC). An invalid source reads 0 with BOUND_CTRL = 1, and with BOUND_CTRL = 0
the destination lane is not written (v2 keeps 0x5a5a). A lane disabled by EXEC is not written either way. The four
fixtures cover FI x BOUND_CTRL: MIN_IN_IF (0, 0), MIN_IN_IF_BC (0, 1), MIN_IN_IF_FI (1, 0), MIN_IN_IF_FI_BC (1, 1);
with FI = 1 the disabled neighbour's v1 is read, so BOUND_CTRL does not matter there. The bit-7 condition gives 40 partly covered quads, 8 whole ones and
16 empty ones over 16 x 16 pixels, and 160 pixels whose x ^ 1 neighbour is on the other side of the if.

Today the translator refuses both fixtures (the read meets the narrowed EXEC); the controls accept that refusal or a
translation. Once the lane-exact EXEC model lowers them, the draw checks the translation against the same reference,
and the model's owner then makes the controls require it (REQUIRE_TRANSLATION). The wrong lowerings are the ones a
model can fall into: ignoring FI (each fixture drawn with the opposite FI rule), and, where it matters (FI = 0), the
other BOUND_CTRL rule.
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

MASK = 0xffffffff
WIDTH = HEIGHT = 16
MULTIPLIER, CONDITION_BIT, UNWRITTEN = 0x9b5d, 0x80, 0x5a5a
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
REQUIRE_TRANSLATION = False  # set by the lane-exact EXEC model's owner when the model lowers these fixtures
failures = []

SOURCE = '''s_wqm_b64 exec, exec
v_mul_u32_u24_e32 v1, 0x9b5d, v0
v_mov_b32_e32 v2, 0x5a5a
v_and_b32_e32 v3, 0x80, v1
v_cmp_ne_u32_e32 vcc_lo, 0, v3
s_and_saveexec_b64 s[4:5], vcc
v_min_u32_dpp v2, v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf%s
s_or_b64 exec, exec, s[4:5]
exp mrt0 v2, v1, v0, v0 done vm
s_endpgm
'''
PREFIX = '7e0afebeff0002165d9b0000ff02047e5a5a0000ff0206368000000080068a7d6a2484befa02042601b1'
SUFFIX = 'ff7e04fe880f1800f802010000000081bf'
# name -> (source, words assembled by llvm-mc 19 for gfx1013, (FI, BOUND_CTRL)); FI is DPP16 bit 18, BOUND_CTRL bit 19
FIXTURES = {
    'MIN_IN_IF': (SOURCE % '', PREFIX + '00' + SUFFIX, (0, 0)),
    'MIN_IN_IF_BC': (SOURCE % ' bound_ctrl:1', PREFIX + '08' + SUFFIX, (0, 1)),
    'MIN_IN_IF_FI': (SOURCE % ' fi:1', PREFIX + '04' + SUFFIX, (1, 0)),
    'MIN_IN_IF_FI_BC': (SOURCE % ' bound_ctrl:1 fi:1', PREFIX + '0c' + SUFFIX, (1, 1)),
}
REFUSAL = 'v_min_u32_dpp requires a straight-line pixel shader with entry EXEC: EXEC written at +0x20 by s_and_saveexec_b64'


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + detail[:2000]))
    if not ok:
        failures.append(label)


def translate(name):
    try:
        return G.Pipeline(VS_WORDS, bytes.fromhex(FIXTURES[name][1]), root_blob(), ps_hw_inputs=PACKED).hlsl_ps(), None
    except G.Unsupported as reason:
        return None, str(reason)


# ---- CPU reference ----------------------------------------------------------------------------------------------------
def word(x, y):
    return ((x | (y << 16)) * MULTIPLIER) & MASK


def active(x, y):
    return bool(word(x, y) & CONDITION_BIT)


def image(bits):
    """Words per pixel: v2, v1, v0, v0, for the instruction's (FI, BOUND_CTRL): per lane src_valid = in range (always,
    a quad swap) and (FI or the source lane enabled); an invalid source reads 0 with BOUND_CTRL = 1, else the lane is
    not written. A disabled lane is not written either way."""
    fetch_inactive, bound_ctrl = bits
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            own, v2 = word(x, y), UNWRITTEN
            if active(x, y):
                if active(x ^ 1, y) or fetch_inactive:
                    v2 = min(word(x ^ 1, y), own)
                elif bound_ctrl:
                    v2 = min(0, own)
            out += [v2, own, x | (y << 16), x | (y << 16)]
    return out


# ---- hand-written lowerings: the contract and the wrong ones ----------------------------------------------------------
HAND = '''struct PSIn { float4 pos : SV_Position; };
float4 main(PSIn i) : SV_Target0 {
    uint v0 = (uint(i.pos.x) & 0xffffu) | ((uint(i.pos.y) & 0xffffu) << 16);
    uint v1 = (0x9b5du & 0xffffffu) * (v0 & 0xffffffu);
    bool lane = (v1 & 0x80u) != 0u;
    // Quad reads in uniform flow, before the if: the neighbour's word and whether its lane is enabled.
    uint other = QuadReadAcrossX(v1);
    bool other_lane = QuadReadAcrossX(lane ? 1u : 0u) != 0u;
    uint v2 = 0x5a5au;
    if (lane) {
        %s
    }
    return float4(asfloat(v2), asfloat(v1), asfloat(v0), asfloat(v0));
}
'''
CONTRACT = {(0, 0): 'if (other_lane) v2 = min(other, v1);', (0, 1): 'v2 = other_lane ? min(other, v1) : min(0u, v1);',
            (1, 0): 'v2 = min(other, v1);', (1, 1): 'v2 = min(other, v1);'}


def wrong_lowerings(bits):
    """{what it gets wrong: body} for the fixture's bits: ignoring FI (the mutant c13's model must not be), and the
    other BOUND_CTRL rule where BOUND_CTRL matters (FI = 0)."""
    fi, bc = bits
    out = {'ignores FI': CONTRACT[(1 - fi, bc)]}
    if not fi:
        out['takes the other BOUND_CTRL rule'] = CONTRACT[(fi, 1 - bc)]
    return out


def controls():
    for name, (_source, _words, bits) in FIXTURES.items():
        text, reason = translate(name)
        if REQUIRE_TRANSLATION:
            check('%s translates' % name, text is not None, str(reason))
        else:
            check('%s translates or is refused for its narrowed EXEC' % name,
                  text is not None or (reason is not None and REFUSAL in reason), str(reason))
    reference, other = image((0, 0)), image((0, 1))
    pixels = WIDTH * HEIGHT
    split = sum(active(x, y) and not active(x ^ 1, y) for y in range(HEIGHT) for x in range(WIDTH))
    check('the condition leaves %d enabled pixels with a disabled x ^ 1 neighbour' % split, split > pixels // 8)
    check('the two BOUND_CTRL rules give different images', reference[0::4] != other[0::4])
    check('FI = 1 (read the disabled neighbour) gives another image', image((1, 0))[0::4] != reference[0::4])
    check('with FI = 1 BOUND_CTRL does not matter', image((1, 0)) == image((1, 1)))


# ---- draw -------------------------------------------------------------------------------------------------------------
def words_file(directory, name, values):
    path = directory / ('%s.bin' % name)
    path.write_bytes(struct.pack('<%dI' % len(values), *values))
    return path


def draw(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


def matched(run):
    return run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout


def mismatched(run):
    lines = [ln for ln in run.stderr.splitlines() if 'pixel' in ln]
    return run.returncode != 0 and bool(lines) and lines[0].startswith('pixel mismatch')


def check_draw(dxc, renderer, adapter, fxc=None):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs_dxc', 'vs_6_0', directory)
        for name, (_source, _words, bits) in FIXTURES.items():
            tag = '%d%d' % bits
            reference = words_file(directory, 'reference_%s' % name, image(bits))
            good = compile_hlsl(dxc, 'dxc', HAND % CONTRACT[bits], 'ps_contract_%s' % tag, 'ps_6_0', directory)
            run = draw(renderer, adapter, vs, good, reference)
            print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                                  if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
            check('dxc %s %s: the contract lowering matches all %d pixels' % (adapter, name, WIDTH * HEIGHT),
                  matched(run), run.stdout + run.stderr)
            for how, body in wrong_lowerings(bits).items():
                bad = compile_hlsl(dxc, 'dxc', HAND % body, 'ps_wrong_%s_%d' % (tag, len(how)), 'ps_6_0', directory)
                run = draw(renderer, adapter, vs, bad, reference)
                check('dxc %s %s: a lowering that %s fails' % (adapter, name, how), mismatched(run),
                      run.stdout + run.stderr)
            text, reason = translate(name)
            if text is None:
                print('  %s: not lowered by the translator yet (%s); contract and wrong lowerings drawn' % (name, reason))
                continue
            ps = compile_hlsl(dxc, 'dxc', text, 'ps_%s' % name.lower(), 'ps_6_0', directory)
            run = draw(renderer, adapter, vs, ps, reference)
            check('dxc %s %s: the translation matches all %d pixels' % (adapter, name, WIDTH * HEIGHT),
                  matched(run), run.stdout + run.stderr)


def check_binary(llvm_mc):
    for name, (source, hexwords, _bits) in FIXTURES.items():
        check('llvm-mc assembles the %s fixture to the same words' % name,
              assemble(llvm_mc, source) == bytes.fromhex(hexwords))


def main():
    if len(sys.argv) == 1:
        controls()
    else:
        mode, *args = sys.argv[1:]
        if mode == 'binary':
            check_binary(*args)
        elif mode == 'draw':
            controls()
            check_draw(*args)
        else:
            raise SystemExit(__doc__)
    if failures:
        print('FAIL: %d check(s): %s' % (len(failures), ', '.join(failures)))
        return 1
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
