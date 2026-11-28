"""Direct compute SDWA arithmetic and conversion. Use ir DUMPER or gpu STORE_REPLAY SDK."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import f16_packed_model as model
import native_f16_alu_check as fixture
from native_compute_clear_check import write_request

BYTE_1, WORD_0, WORD_1, DWORD = 1, 4, 5, 6
PAD, SEXT, PRESERVE = 0, 1, 2
NEG, ABS, SCALAR = 0x10, 0x20, 0x80
CLAMP = 1 << 13
OLD = 0xa55a1357
GUARD = 0x69696969
ADD_CO_CI, PKRTZ = 0x28, 0x2f
LANE_STRIDE = fixture.STRIDE // 2
LANE_BYTE_SHIFT = fixture.SHIFT - 1


def sdwa(opcode, dst, unused, src0, src1, reg0=1, reg1=2, dst_mod=0):
    ext = (reg0 | (dst << 8) | (unused << 11) | dst_mod |
           (src0 << 16) | (src1 << 24))
    return fixture.vop2(opcode, 3, 249, reg1) + [ext]


def cvt_sdwa(src0, dst_mod=CLAMP):
    # VOP1 opcode 0x0b, v3 destination, SDWA escape, then v1 with DWORD/PAD.
    return [0x7e000000 | (3 << 17) | (0x0b << 9) | 249,
            1 | (DWORD << 8) | dst_mod | (src0 << 16)]


class ClampCase:
    op = 'v_cvt_f32_f16 clamp'
    dest = 3

    def __init__(self, half, selector):
        self.selector = selector
        # Put a distinct in-range value in the other half to expose a wrong selector.
        other = 0x3400
        self.word = (half << 16) | other if selector == WORD_1 else (other << 16) | half
        value = struct.unpack('<e', struct.pack('<H', half))[0]
        # An ordered > comparison maps NaN, negative values and -0.0 to +0.0.
        result = min(value, 1.0) if value > 0.0 else 0.0
        self.want = struct.unpack('<I', struct.pack('<f', result))[0]

    def emit(self):
        return fixture.mov(1, self.word) + fixture.mov(self.dest, OLD) + cvt_sdwa(self.selector)

    def matches(self, got):
        return got == self.want


class Case:
    dest = 3

    def __init__(self, op, a, b, dst, unused, src0, src1):
        self.op, self.a, self.b = op, a, b
        self.dst, self.unused = dst, unused
        self.src0, self.src1 = src0, src1
        self.shift = 16 if dst == WORD_1 else 0
        self.mask = 0xffff << self.shift
        self.want = self.expected(src0, src1, unused)

    def expected(self, src0, src1, unused):
        def read(value, control):
            picked = model.pick(value, (control & 7) == WORD_1)
            return fixture.modified(picked, bool(control & ABS), bool(control & NEG), 0)

        result = fixture.binary16(self.op, read(self.a, src0), read(self.b, src1))
        other = OLD & ~self.mask if unused == PRESERVE else 0
        return (result << self.shift) | other

    def emit(self):
        return (fixture.mov(1, self.a) + fixture.mov(2, self.b) + fixture.mov(self.dest, OLD) +
                sdwa(fixture.OP_CODE[self.op], self.dst, self.unused, self.src0, self.src1))

    def matches(self, got):
        return (got & ~self.mask == self.want & ~self.mask and
                fixture.same_binary16((got & self.mask) >> self.shift, (self.want & self.mask) >> self.shift))


def cases():
    shapes = [
        (0x3c004400, 0x45004000, WORD_0, PRESERVE, WORD_1, DWORD),
        (0x44004200, 0x45003c00, WORD_0, PAD, DWORD, WORD_0),
        (0x4400bc00, 0x40004500, WORD_1, PRESERVE, WORD_0 | ABS | NEG, WORD_1),
    ]
    items = [Case(op, *shape) for shape in shapes for op in fixture.OPS]
    items += [
        Case('add', 0x3c014400, 0x45001000, WORD_0, PRESERVE, WORD_1, DWORD),
        Case('mul', 0x44000003, 0x45003800, WORD_0, PAD, DWORD, WORD_0),
        Case('mul', 0x44007bff, 0x40004500, WORD_1, PAD, WORD_0, WORD_1),
        Case('add', 0x7e014400, 0x45003c00, WORD_0, PRESERVE, WORD_1, DWORD),
        # neg on one source and abs on the other: -(1.0) + |-2.0| = 1.0; dropping either modifier changes the sum.
        Case('add', 0x44003c00, 0xc0004500, WORD_1, PAD, WORD_0 | NEG, WORD_1 | ABS),
    ]
    assert len(items) < fixture.STRIDE
    return items


def clamp_cases():
    # -1, 2, quiet NaN, -0, smallest positive subnormal, 0.5, and 1.
    halves = (0xbc00, 0x4000, 0x7e01, 0x8000, 0x0001, 0x3800, 0x3c00)
    return [ClampCase(half, selector) for selector in (WORD_0, WORD_1) for half in halves]


def check_mutants(items):
    # Manual lowering mutants that must fail: swap WORD_0/WORD_1 on either
    # source, or ignore UNUSED_PRESERVE by zeroing the other destination half.
    def swap(control):
        return control ^ 1 if (control & 7) in (WORD_0, WORD_1) else control

    # Without this, a comparator that rejects every word would pass the mutant checks below.
    assert all(case.matches(case.want) for case in items)
    assert any(not case.matches(case.expected(swap(case.src0), case.src1, case.unused)) for case in items)
    assert any(not case.matches(case.expected(case.src0, swap(case.src1), case.unused)) for case in items)
    assert any(not case.matches(case.expected(case.src0, case.src1, PAD)) for case in items)


def refusals():
    add = fixture.OP_CODE['add']
    dest_error = 'unsupported SDWA destination modifier'
    src_error = 'invalid SDWA source modifier'
    return [
        (sdwa(add, DWORD, PAD, WORD_0, WORD_0), dest_error),
        (sdwa(add, WORD_0, PAD, BYTE_1, WORD_0), src_error),
        (sdwa(add, WORD_0, PAD, WORD_0, WORD_0 | 8), src_error),
        (sdwa(add, WORD_0, SEXT, WORD_0, WORD_0), dest_error),
        (sdwa(add, WORD_0, PAD, WORD_0, WORD_0, dst_mod=1 << 13), dest_error),
        (sdwa(add, WORD_0, PAD, WORD_0, WORD_0, dst_mod=1 << 14), dest_error),
        (cvt_sdwa(WORD_0, 1 << 14), dest_error),
        (cvt_sdwa(WORD_1, CLAMP | (1 << 14)), dest_error),
        (sdwa(0x39, WORD_0, PAD, WORD_0, WORD_0), 'unimplemented direct SDWA operand profile'),  # max
        (sdwa(0x3a, WORD_0, PAD, WORD_0, WORD_0), 'unimplemented direct SDWA operand profile'),  # min
    ] + [
        (sdwa(op, WORD_0, PAD, DWORD, DWORD, dst_mod=1 << modifier), dest_error)
        for op in (PKRTZ, ADD_CO_CI) for modifier in (13, 14)
    ]


def run_ir(dumper, items, code):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = fixture.dump(dumper, work, code, 'sdwa-f16')
        assert run.returncode == 0, run.stderr
        stores = sorted(fixture.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(items))]
        for (addr, got), case in zip(stores, items):
            assert case.matches(got), (addr, case.op, hex(got), hex(case.want))
        for index, (instruction, why) in enumerate(refusals()):
            run = fixture.dump(dumper, work, instruction + [0xbf810000], 'refused-%d' % index)
            assert run.returncode != 0 and why in run.stderr, (why, run.returncode, run.stderr)
    print('native_sdwa_f16 IR: %d results and %d refusals' % (len(items), len(refusals())))


def extend_gpu_program(code, items):
    code = code[:-1]  # Append cases just before s_endpgm.
    expected = {(0, slot): (case.want, None) for slot, case in enumerate(items)}
    slot = len(items)
    # The expected half is the RTZ conversion of the first source after abs/neg.
    for a, dst, unused, src0, half in (
            (0x3f801800, WORD_0, PAD, DWORD, 0x3c00),
            (0xbf801800, WORD_1, PRESERVE, DWORD | ABS, 0x3c00),
            (0x7f800000, WORD_0, PAD, DWORD | NEG, 0xfc00),
            (0x7fc00001, WORD_1, PRESERVE, DWORD, 0x7e00)):
        code += fixture.mov(1, a) + fixture.mov(3, OLD)
        code += sdwa(PKRTZ, dst, unused, src0, DWORD | SCALAR, reg1=128)
        code += fixture.store(4 * slot, 3)
        shift = 16 if dst == WORD_1 else 0
        expected[0, slot] = ((half << shift) | (OLD & 0xffff if unused == PRESERVE else 0), shift)
        slot += 1

    # Enable lanes 0 and 1. v8 is the group base and v0 is the local thread ID.
    code += [fixture.sop1(3, 126, 131)]
    code += fixture.vop2(0x1a, 7, 128 + LANE_BYTE_SHIFT, 0)
    code += fixture.vop2(0x25, 8, 256 + 7, 8)
    carry_in = 0b10
    for a, b, dst, src0, src1 in (
            (0xffffffff, 0, WORD_1, DWORD, DWORD),
            (0xffffffff, 0x1235, WORD_1, DWORD, DWORD),
            (0xfffffffe, 0xffff0001, WORD_1, WORD_0, DWORD),
            (0xaaaa1234, 0xbeefffff, DWORD, WORD_0, WORD_1)):
        code += fixture.mov(1, a) + fixture.mov(2, b) + fixture.mov(4, 1)
        code += [fixture.sop1(3, 106, 128 + carry_in)]
        code += sdwa(ADD_CO_CI, dst, PAD, src0, src1)
        code += fixture.vop2(0x01, 4, 128, 4)  # v_cndmask_b32 v4, 0, v4, vcc_lo
        code += fixture.store(4 * slot, 3) + fixture.store(4 * (slot + 1), 4)
        left = a if src0 == DWORD else a & 0xffff
        right = b if src1 == DWORD else b >> 16
        for lane in range(2):
            total = left + right + ((carry_in >> lane) & 1)
            result = (total & 0xffff) << 16 if dst == WORD_1 else total & 0xffffffff
            expected[lane, slot] = (result, None)
            expected[lane, slot + 1] = (total >> 32, None)
        slot += 2
    assert slot <= LANE_STRIDE
    return code + [0xbf810000], expected


def run_gpu(gpu, sdk, items, code):
    code, expected = extend_gpu_program(code, items)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src, root = work / 'root.hlsl', work / 'root.bin'
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([str(Path(sdk) / 'dxc.exe'), '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, src], check=True)
        sa = struct.pack('<7I', 0, 0, 0, 32 | (1 << 16), 1 | (1 << 20), 0, 0x80 | (5 << 1))
        marker = write_request(work, code, root.read_bytes(), sa)
        dest = work / 'sdwa-f16-readback.bin'
        run = subprocess.run([gpu, sdk, str(marker), str(work / 'cache'), '0', str(dest), '--direct'])
        if run.returncode == 77:
            sys.exit(77)
        run.check_returncode()
        stores = {base + lane * LANE_STRIDE + slot: (slot, want, shift)
                  for (base,) in fixture.group_offsets(1)
                  for (lane, slot), (want, shift) in expected.items()}
        actual = struct.unpack(f'<{fixture.WORDS}I', dest.read_bytes())
        for index, got in enumerate(actual):
            if index in stores:
                slot, want, shift = stores[index]
                if slot < len(items):
                    matches = items[slot].matches(got)
                elif shift is not None:
                    mask = 0xffff << shift
                    matches = (got & ~mask == want & ~mask and
                               fixture.same_binary16((got & mask) >> shift, (want & mask) >> shift))
                else:
                    matches = got == want
                assert matches, (index, slot, hex(got), hex(want))
            else:
                assert got == GUARD, (index, hex(got), hex(GUARD))
    print('native_sdwa_f16 GPU: %d results and %d guards' % (len(stores), fixture.WORDS - len(stores)))


if __name__ == '__main__':
    items = cases()
    check_mutants(items)
    items += clamp_cases()
    assert len(items) < fixture.STRIDE
    code, _ = fixture.program(items)
    if len(sys.argv) == 3 and sys.argv[1] == 'ir':
        run_ir(sys.argv[2], items, code)
    elif len(sys.argv) == 4 and sys.argv[1] == 'gpu':
        run_gpu(sys.argv[2], sys.argv[3], items, code)
    else:
        sys.exit(__doc__)
