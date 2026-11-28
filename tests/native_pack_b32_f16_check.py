"""v_pack_b32_f16 (VOP3 0x311) in the direct C++ DXIL path.

D[15:0] = S0.f16 and D[31:16] = S1.f16. Per source, op_sel picks the high or low half, abs clears bit 15 of the
half, then neg toggles it. Only integer bit operations are allowed, so signed zero and NaN payloads are kept.
The reference is `ref` of startup_gfx2hlsl_pack_b32_f16_check.py, the model of the Python translator.

  native_pack_b32_f16_check.py ir DUMPER            no GPU: build pixel containers, dump the emitted IR with
                                                    native_pixel_ir_dump, evaluate it and compare with the model
  native_pack_b32_f16_check.py gpu STORE_REPLAY SDK  compute readback of the packed bits (exit 77: no adapter)

Refused forms (explicit diagnostics): clamp, omod, a destination op_sel bit, op_sel[2], abs/neg on source 2, inline
constants and a literal source.
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from native_direct_container_check import patched_pixel
from startup_gfx2hlsl_pack_b32_f16_check import ref
from native_compute_clear_check import write_request
from native_scalar_lds_check import scalar, sop1, vmove

GROUPS = 2
WORDS = 2064
BASE = 8  # words before group 0's results (as in half_mad_readback)
STRIDE = 128  # words per group: 512 bytes, so the 66 results of one group never reach the next
SHIFT = 9  # s_lshl_b32 amount giving the per-group byte base
assert 4 * STRIDE == 1 << SHIFT

PAIRS = [(0x3c00bc00, 0x80017fff), (0xfc007e00, 0x7d557c01), (0x12345678, 0xffff0001), (0x00008000, 0x80000000)]


def pack(dst, s0, s1, opsel=0, absm=0, neg=0, clamp=0, omod=0):
    return [0xd7110000 | clamp << 15 | opsel << 11 | absm << 8 | dst, s0 | s1 << 9 | omod << 27 | neg << 29]


def mov(reg, value):
    return [0x7e0002ff | reg << 17, value]


def mov_s(reg, value):
    return [0xbe8003ff | reg << 16, value]


def container(code, outputs=(4, 5, 6, 7)):
    return patched_pixel(len(code), code, export=(0xf800180f, sum(r << 8 * i for i, r in enumerate(outputs))))


def evaluate(ir):
    """Evaluate the straight-line integer IR; return the four exported words. Anything else is an error."""
    values, out = {}, {}

    def val(token):
        return values[token] if token.startswith('%') else int(token) & 0xffffffff

    for line in ir.splitlines():
        line = line.strip()
        m = re.fullmatch(r'(%\w+) = (and|or|xor|shl|lshr|add|sub) i32 (\S+), (\S+)', line)
        if m:
            a, b = val(m.group(3)), val(m.group(4))
            values[m.group(1)] = {'and': a & b, 'or': a | b, 'xor': a ^ b, 'shl': a << b, 'lshr': a >> b,
                                  'add': a + b, 'sub': a - b}[m.group(2)] & 0xffffffff
            continue
        m = re.fullmatch(r'(%\w+) = bitcast i32 (\S+) to float', line)
        if m:
            values[m.group(1)] = val(m.group(2))
            continue
        m = re.fullmatch(r'call void @dx\.op\.storeOutput\.f32\(i32 5, i32 0, i32 0, i8 (\d), float (\S+)\)', line)
        if m:
            out[int(m.group(1))] = val(m.group(2))
            continue
        # Any float operation, conversion call or select would break the bit-exactness requirement.
        assert not re.search(r'\b(fadd|fmul|fcmp|fptrunc|fpext|select|legacy|call float)', line), line
    return [out[c] for c in range(4)]


def dump(dumper, work, code, name='case'):
    path = work / (name + '.cdxil')
    path.write_bytes(container(code))
    run = subprocess.run([dumper, path], capture_output=True, text=True)
    return run


def run_ir(dumper):
    cases = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Every op_sel x abs x neg form, four value pairs per container (v1/v2 in, v4..v7 out).
        for sel in range(4):
            for absm in range(4):
                for neg in range(4):
                    code, want = [], []
                    for k, (a, b) in enumerate(PAIRS):
                        code += mov(1, a) + mov(2, b) + pack(4 + k, 257, 258, sel, absm, neg)
                        want.append(ref(a, b, sel, absm, neg))
                    run = dump(dumper, work, code)
                    assert run.returncode == 0, (sel, absm, neg, run.stderr)
                    got = evaluate(run.stdout)
                    assert got == want, (sel, absm, neg, [hex(g) for g in got], [hex(w) for w in want])
                    cases += len(want)
                    # Controls: the order of abs and neg, and the half order, must be observable by this check.
                    if absm & neg:
                        assert got != [ref(a, b, sel, absm, neg, neg_first=True) for a, b in PAIRS], (sel, absm, neg)
                    if sel == 1 and not absm and not neg:
                        swapped = [(w & 0xffff) << 16 | w >> 16 for w in want]
                        assert got != swapped
        # The issue reproduction: v4 = 0x77883344 and v5 is untouched. Without the pack both are 0x55667788.
        issue = [0x7e0802ff, 0x11223344, 0x7e0a02ff, 0x55667788, 0xd7110004, 0x00020b04]
        path = work / 'issue.cdxil'
        path.write_bytes(patched_pixel(len(issue), issue, export=(0xf800180f, 0x05040504)))
        run = subprocess.run([dumper, path], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert evaluate(run.stdout) == [0x77883344, 0x55667788, 0x77883344, 0x55667788], run.stdout
        control = patched_pixel(4, issue[:4], export=(0xf800180f, 0x05040504))
        path.write_bytes(control)
        run = subprocess.run([dumper, path], capture_output=True, text=True)
        assert run.returncode == 0 and evaluate(run.stdout) == [0x11223344, 0x55667788] * 2, run.stderr
        # Destination overlap, SGPR sources (including a high half) and one source used twice.
        a, b = 0x80017e55, 0xfc00bc01
        shapes = [
            (mov(4, a) + mov(5, b) + pack(4, 260, 261, 1, 1, 2), a, b, 1, 1, 2),    # dst = src0
            (mov(4, a) + mov(5, b) + pack(5, 260, 261, 2, 0, 1), a, b, 2, 0, 1),    # dst = src1
            (mov(4, a) + pack(4, 260, 260, 1, 3, 3), a, a, 1, 3, 3),                 # one reg twice
            (mov_s(5, a) + mov(5, b) + pack(4, 5, 261, 1, 1, 0), a, b, 1, 1, 0),     # SGPR src0
            (mov_s(5, b) + mov(4, a) + pack(4, 260, 5, 2, 2, 3), a, b, 2, 2, 3),     # SGPR src1
        ]
        # The SGPR shapes reuse number 5 for an unrelated VGPR; keep each case independent.
        for code, x, y, sel, absm, neg in shapes:
            run = dump(dumper, work, sum((mov(r, 0) for r in range(4, 8)), []) + code)
            assert run.returncode == 0, (code, run.stderr)
            dst = code[-2] & 255 if code[-2] >> 16 == 0xd711 else 4
            assert evaluate(run.stdout)[dst - 4] == ref(x, y, sel, absm, neg), (hex(evaluate(run.stdout)[dst - 4]), code)
            cases += 1
        # Unsupported forms are refused with a named diagnostic and no IR.
        base = mov(1, a) + mov(2, b)
        for words, why in [(pack(4, 257, 258, clamp=1), 'clamp or omod'), (pack(4, 257, 258, omod=1), 'clamp or omod'),
                           (pack(4, 257, 258, opsel=4), 'destination or third-source'),
                           (pack(4, 257, 258, opsel=8), 'destination or third-source'),
                           (pack(4, 257, 258, absm=4), 'destination or third-source'),
                           (pack(4, 257, 258, neg=4), 'destination or third-source'),
                           (pack(4, 242, 258), 'constant source'), (pack(4, 129, 258), 'constant source'),
                           (pack(4, 257, 255) + [0x3c00], 'constant source')]:
            run = dump(dumper, work, base + words, 'refused')
            assert run.returncode != 0 and 'v_pack_b32_f16' in run.stderr and why in run.stderr, (why, run.stderr)
    print('pack_b32_f16 IR: %d results match the model; issue reproduction, control, overlap, SGPR and refusals '
          'pass; swapped-half and neg-before-abs variants differ from the emitted bits' % cases)


def group_offsets(count):
    """Word indices written by each workgroup; they must be disjoint and inside the result buffer."""
    offsets = [[BASE + group * STRIDE + i for i in range(count)] for group in range(GROUPS)]
    flat = [i for group in offsets for i in group]
    assert len(set(flat)) == len(flat), 'two workgroups write the same word'
    assert 0 <= min(flat) and max(flat) < WORDS, 'a store falls outside the result buffer'
    return offsets


def run_gpu(gpu, sdk):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src, root = work / 'root.hlsl', work / 'root.bin'
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, src], check=True)
        sa = struct.pack('<7I', 0, 0, 0, 32 | 1 << 16, 1 | (1 << 20), 0, 0x80 | 5 << 1)
        checked = 0
        for a, b in PAIRS:
            # Same preamble as half_mad_readback in native_scalar_lds_check.py: s20:21 table, v8 = group * 512 bytes (STRIDE words).
            words = [sop1(3, 20, 4), sop1(3, 21, 128), 0xf408080a, 0xfa000020,
                     scalar(0x1e, 7, 5, 128 + SHIFT), vmove(8, 7), sop1(3, 126, 129)]
            words += mov(1, a) + mov(2, b)
            want = []
            for sel in range(4):
                for absm in range(4):
                    for neg in range(4):
                        # v3 is both the destination and (through the preceding move) a pre-existing value.
                        words += mov(3, 0x69696969) + pack(3, 257, 258, sel, absm, neg)
                        words += [0xe0701000 | 4 * len(want), 0x80080308]
                        want.append(ref(a, b, sel, absm, neg))
            # SGPR source and destination overlapping a source.
            words += mov_s(9, a) + pack(3, 9, 258, 1, 1, 1) + [0xe0701000 | 4 * len(want), 0x80080308]
            want.append(ref(a, b, 1, 1, 1))
            words += mov(3, a) + pack(3, 259, 258, 2, 2, 1) + [0xe0701000 | 4 * len(want), 0x80080308]
            want.append(ref(a, b, 2, 2, 1))
            marker = write_request(work, words + [0xbf810000], root.read_bytes(), sa)
            dest = work / 'pack-readback.bin'
            run = subprocess.run([gpu, sdk, marker, work / 'cache', '0', dest, '--direct'])
            if run.returncode == 77:
                sys.exit(77)
            run.check_returncode()
            offsets = group_offsets(len(want))
            expected = [0x69696969] * WORDS
            for group in range(GROUPS):
                expected[offsets[group][0]:offsets[group][-1] + 1] = want
            actual = struct.unpack('<2064I', dest.read_bytes())
            bad = [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, expected)) if x != y][:10]
            assert not bad, (hex(a), hex(b), bad)
            checked += GROUPS * len(want)
    print('pack_b32_f16 GPU: %d packed words matched the model bit for bit' % checked)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'ir':
        run_ir(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == 'gpu':
        run_gpu(*sys.argv[2:])
    else:
        sys.exit(__doc__)
