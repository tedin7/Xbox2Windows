"""Verify that compute buffer loads through a modified or overwritten descriptor word are refused.

Refs #2901, #2618.
"""
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))

import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_compute_check as C  # noqa: E402


def sop2(op, d, a, b, lit=None):
    return struct.pack('<I', 0x80000000 | op << 23 | d << 16 | b << 8 | a) + (b'' if lit is None else struct.pack('<I', lit))


def sop1(op, d, a):
    return struct.pack('<I', 0xbe800000 | d << 16 | op << 8 | a)


W = C.WORDS_A
LOAD_SITE = W.index(bytes.fromhex('001030e004050380'))
STORE_SITE = C.STORE_A

# r = first register of the descriptor (12 for load s[12:15], 16 for store s[16:19])
# Word 1: stride; Word 2: NUM_RECORDS; Word 3: format/OOB
EDITS = {
    'descnum copy w1': lambda r: sop2(0x20, 21, r + 2, 130) + sop1(0x03, r + 1, 21),
    'descnum copy w2': lambda r: sop2(0x20, 21, r + 2, 130) + sop1(0x03, r + 2, 21),
    'descnum copy w3': lambda r: sop2(0x20, 21, r + 2, 130) + sop1(0x03, r + 3, 21),
    'descnum alu w2': lambda r: sop2(0x20, 21, r + 2, 130) + sop2(0x20, r + 2, 21, 128),
    'in-place and 0 w2': lambda r: sop2(0x0e, r + 2, r + 2, 128),
    'constant 0 w2': lambda r: sop1(0x03, r + 2, 128),
}


def translate_code(code):
    return G.ComputePipeline(code, C.root_blob(), C.CASES['A'][2], (2, 3, 4)).hlsl_cs()


def check_refusals():
    # 1. Base case A translates cleanly without edits
    base_hlsl = translate_code(W)
    assert 'v5 = r0.Load(a + 0);' in base_hlsl, 'base case A load missing'
    assert 'u0.Store(a, v5);' in base_hlsl, 'base case A store missing'

    # 2. Check each edit refused for both load and store
    for label, edit in EDITS.items():
        # Check load refusal
        load_code = W[:LOAD_SITE] + edit(12) + W[LOAD_SITE:]
        try:
            translate_code(load_code)
            raise AssertionError(f'expected load refusal for {label}, but translation succeeded')
        except G.Unsupported as e:
            assert 'buffer load through a modified descriptor' in str(e), (
                f'unexpected refusal for load {label}: {e}'
            )

        # Check store refusal
        store_code = W[:STORE_SITE] + edit(16) + W[STORE_SITE:]
        try:
            translate_code(store_code)
            raise AssertionError(f'expected store refusal for {label}, but translation succeeded')
        except G.Unsupported as e:
            assert 'buffer store through a modified descriptor' in str(e), (
                f'unexpected refusal for store {label}: {e}'
            )

    # 3. Proven stride-0 clear on raw load is accepted (stride0_ok=True for non-indexed load)
    # s_and_b32 s13, s13, 0xc000ffff
    stride0_inst = sop2(0x0e, 13, 13, 255) + struct.pack('<I', 0xc000ffff)
    stride0_hlsl = translate_code(W[:LOAD_SITE] + stride0_inst + W[LOAD_SITE:])
    assert 'v5 = r0.Load(a + 0);' in stride0_hlsl, 'stride0 load failed to translate'

    # 4. Incorrect stride mask (0xc000fff0) is refused on load
    bad_stride_inst = sop2(0x0e, 13, 13, 255) + struct.pack('<I', 0xc000fff0)
    try:
        translate_code(W[:LOAD_SITE] + bad_stride_inst + W[LOAD_SITE:])
        raise AssertionError('expected refusal for bad stride mask on load, but succeeded')
    except G.Unsupported as e:
        assert 'buffer load through a modified descriptor' in str(e), f'unexpected refusal: {e}'

    # 5. Proven structured-to-raw conversion on load descriptor is accepted (raw_conversion_ok=True)
    # Extracts stride from s13, clears stride in s13, scales s14 by s21, sets raw OOB in s15.
    raw_conv = (
        sop2(0x27, 21, 13, 255, 0xe0010) +
        sop2(0x0e, 13, 13, 255, 0xc000ffff) +
        sop2(0x26, 14, 14, 21) +
        sop2(0x0e, 15, 15, 255, 0xcfffffff) +
        sop2(0x10, 15, 15, 255, 0x30000000)
    )
    raw_hlsl = translate_code(W[:LOAD_SITE] + raw_conv + W[LOAD_SITE:])
    assert 'v5 = r0.Load(a + 0);' in raw_hlsl, 'raw conversion load failed to translate'

    # 6. Incomplete or invalid raw conversion (e.g. constant multiplier 4 instead of extracted stride s21) is refused
    bad_raw_conv = (
        sop2(0x27, 21, 13, 255, 0xe0010) +
        sop2(0x0e, 13, 13, 255, 0xc000ffff) +
        sop2(0x26, 14, 14, 132) +  # literal 4
        sop2(0x0e, 15, 15, 255, 0xcfffffff) +
        sop2(0x10, 15, 15, 255, 0x30000000)
    )
    try:
        translate_code(W[:LOAD_SITE] + bad_raw_conv + W[LOAD_SITE:])
        raise AssertionError('expected refusal for bad raw conversion on load, but succeeded')
    except G.Unsupported as e:
        assert 'buffer load through a modified descriptor' in str(e), f'unexpected refusal: {e}'

    # 7. Formatted indexed buffer load through a modified descriptor is refused.
    # Inserting `s_and_b32 s21, s13, 0xc000ffff` immediately before LOAD_L writes
    # a derived descriptor word to s21, leaving the four operand registers s[12:15]
    # intact. The indexed formatted load must be refused with 'indexed load through a modified descriptor'.
    prefix_l = C.WORDS_L[:C.WORDS_L.index(C.LOAD_L)]
    end_l = bytes.fromhex('000081bf')
    s_and_s21_s13 = sop2(0x0e, 21, 13, 255, 0xc000ffff)
    formatted_mod_code = prefix_l + s_and_s21_s13 + C.LOAD_L + end_l
    try:
        G.ComputePipeline(formatted_mod_code, C.root_blob(), C.CASES['L-uint'][2], (2, 3, 4)).hlsl_cs()
        raise AssertionError('expected refusal for formatted indexed load with modified descriptor, but translated')
    except G.Unsupported as e:
        assert 'indexed load through a modified descriptor' in str(e), f'unexpected refusal: {e}'

    # 8. Negative control: pre-fix mutant (bypassing load descriptor validation)
    # Under this mutant, unvalidated loads must translate (escaping detection).
    real_load_desc = G.Stage.load_descriptor
    try:
        G.Stage.load_descriptor = lambda self, tok, what, **kw: self.descriptor(tok)
        for label, edit in EDITS.items():
            mutant_code = W[:LOAD_SITE] + edit(12) + W[LOAD_SITE:]
            res = translate_code(mutant_code)
            assert 'v5 = r0.Load(a + 0);' in res, f'mutant failed to reproduce issue on {label}'
    finally:
        G.Stage.load_descriptor = real_load_desc

    print('compute buffer load modified descriptor checks passed')


if __name__ == '__main__':
    check_refusals()
