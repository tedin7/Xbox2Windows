"""A loop over the set bits of a wave-uniform word, laid out latch first, in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw the triangle of tests/startup_gfx2hlsl_wave_reduce_check.py with the
                                       translated pixel stage and compare every covered pixel with a CPU reference

Compiled pixel code iterates over the bits of a wave OR (RDNA 2 ISA: S_FF1_I32_B32 gives the index of the lowest set
bit, S_LSHL_B32 / S_XOR_B32 clear it, V_READLANE_B32 ignores EXEC) with the back edge going to a latch placed before
the loop head:

    s_cmp_lg_u32 s37, 0 / s_cbranch_scc0 EXIT / s_branch H
  L: s_mov_b32 s37, s40
  H: s_ff1_i32_b32 s25, s37 / s_lshl_b32 s39, 1, s25 / s_xor_b32 s40, s39, s37 / v5 = v5 * 3 + s25 /
     s_cmp_lg_u32 s40, 0 / s_cbranch_scc1 L

s40 also held a saved EXEC before (an earlier, unrelated lane-mask use), so the program-order state at L is wrong:
the translator used to carry that lane-mask mark into s37 and refuse `s_ff1_i32_b32 on a lane mask`. The latch retry
(Stage.retry_latch_join) drops a mark the back edges do not bring, keeps symbols whose last write dominates the join,
and judges uniformity at the joins by reaching definitions over the CFG. The draw shows the result: per desktop wave
(WaveReadLaneFirst of the pixel id, as in the wave_reduce check) v5 is the fold over the bits of the OR of
1 << (x & 7) of the covered pixels, ascending, and the third channel is the highest of those bits. A planted error
that walks the bits from the top (firstbithigh) must fail.
"""
import collections
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import startup_gfx2hlsl_wave_reduce_check as W

G = W.G
# gfx1013, wave32, assembled with LLVM: a mask use of s40, v1 = 1 << (x & 7), then (after the OR chain) the loop.
PREFIX = bytes.fromhex(
    '0001887d6a3ca8be2803febe8700023601001ad501030100')
SUFFIX = bytes.fromhex(
    '80020a7e258007bf090084bf010082bf2803a5be251399be'
    '8119278f27252889830a0a16190a0a4a288007bff8ff85bf'
    '2502087e0f1800f805010400000081bf')
CHAIN = W.program(W.chain('v_or_b32', sd=37))[1]
WORDS = PREFIX + CHAIN + SUFFIX
failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + detail))
    if not ok:
        failures.append(label)


def reference(bits):
    """The loop of the fixture from the ISA text: (v5, last bit) for the OR word `bits` (never 0 here)."""
    acc, word, last = 0, bits, 0
    while word:
        index = (word & -word).bit_length() - 1      # S_FF1_I32_B32
        acc = (acc * 3 + index) & 0xffffffff
        last = word
        word ^= 1 << index                           # S_LSHL_B32 1, idx / S_XOR_B32
    return acc, last


def controls():
    decoded = [asm for _, _, _, asm in G.decode(WORDS)]
    check('the fixture holds the latch-first loop', any('s_ff1_i32_b32 s25, s37' in line for line in decoded)
          and any('s_mov_b32 s37, s40' in line for line in decoded), str(decoded[-14:]))
    with patch.object(G.Stage, 'retry_latch_join', return_value=False):
        _, reason = W.translate(WORDS)
    check('without the latch retry the stale mark refuses as before', reason == 's_ff1_i32_b32 on a lane mask',
          str(reason))
    text, reason = W.translate(WORDS)
    check('the latch-first loop translates', text is not None, str(reason))
    if text:
        check('s_ff1_i32_b32 reads the plain word', 's25 = firstbitlow(s37);' in text, '')
        check('the latch copies the word and the loop clears the bit in it, not as lane bits',
              's37 = s40;' in text and 's40 = s39 ^ s37;' in text, '')
    # A loop body that leaves a lane mask in s40 (s_and_saveexec_b32 s40, vcc_lo in place of the xor) brings the mark
    # back over the edge: the loop must not then read s37 as a plain word.
    xor = struct.pack('<I', 0x89000000 | (40 << 16) | (37 << 8) | 39)          # s_xor_b32 s40, s39, s37
    saveexec = struct.pack('<I', 0xbe800000 | (40 << 16) | (0x3c << 8) | 106)  # s_and_saveexec_b32 s40, vcc_lo
    check('the xor of the loop is where the fixture has it', SUFFIX.count(xor) == 1, SUFFIX.hex())
    marked, reason = W.translate(PREFIX + CHAIN + SUFFIX.replace(xor, saveexec))
    check('a mark the back edge brings keeps the loop refused', marked is None, str(reason))
    for bits, expect in ((0b1, (0, 1)), (0b1010, (1 * 3 + 3, 8)), (0b10000101, ((0 * 3 + 2) * 3 + 7, 128))):
        check('reference loop over 0x%x' % bits, reference(bits) == expect, str(reference(bits)))


def verify(words):
    if len(words) != W.SIZE * W.SIZE * 16:
        return ['readback size is %d bytes' % len(words)]
    pixels = struct.unpack('<%dI' % (len(words) // 4), words)
    groups, errors = collections.defaultdict(list), []
    for y in range(W.SIZE):
        for x in range(W.SIZE):
            p = pixels[(y * W.SIZE + x) * 4:(y * W.SIZE + x) * 4 + 4]
            if p != (W.CLEAR,) * 4:
                groups[p[3] >> 16].append((x, y, p))
    for members in groups.values():
        bits = 0
        for x, _y, _p in members:
            bits |= 1 << (x & 7)
        acc, last = reference(bits)
        for x, y, p in members:
            if list(p[:3]) != [acc, 1 << (x & 7), last]:
                errors.append('pixel (%d,%d): %08x / %08x / %08x, reference %08x / %08x / %08x over %d pixels'
                              % (x, y, p[0], p[1], p[2], acc, 1 << (x & 7), last, len(members)))
    covered = sum(len(m) for m in groups.values())
    if covered < 100:
        errors.append('only %d covered pixels' % covered)
    return errors


def draw(dxc, renderer, adapter):
    text, reason = W.translate(WORDS)
    if text is None:
        check('the latch-first loop translates', False, str(reason))
        return
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = W.compile_hlsl(dxc, 'dxc', W.TRIANGLE_VS, 'vs_latch', 'vs_6_0', directory)
        for label, source in (('translated', text), ('highest bit first', text.replace('firstbitlow(', 'firstbithigh('))):
            tag = 'latch_' + label.replace(' ', '_')
            ps = W.compile_hlsl(dxc, 'dxc', W.observed(source), tag, 'ps_6_0', directory)
            out = directory / (tag + '.bin')
            run = W.dump(renderer, adapter, vs, ps, out)
            clean = 'debug_layer=enabled' not in run.stdout or 'debug_errors=0 debug_warnings=0' in run.stdout
            if run.returncode or not out.exists() or not clean:
                check(tag + ' draws', False, run.stdout + run.stderr)
                continue
            errors = verify(out.read_bytes())
            if label == 'translated':
                print(run.stdout)
                check(adapter + ' every covered pixel matches its desktop wave', not errors, ' / '.join(errors[:3]))
            else:
                check(adapter + ' the highest-bit-first mutation fails', any(e.startswith('pixel (') for e in errors),
                      ' / '.join(errors[:3]))


def main():
    if len(sys.argv) >= 5 and sys.argv[1] == 'draw':
        draw(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        controls()
    if failures:
        print('FAIL')
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
