"""Direct compute SDWA binary16 arithmetic. Use ir DUMPER or gpu STORE_REPLAY SDK."""
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
OLD = 0xa55a1357
GUARD = 0x69696969


def sdwa(opcode, dst, unused, src0, src1, reg0=1, reg1=2, dst_mod=0):
    ext = (reg0 | (dst << 8) | (unused << 11) | dst_mod |
           (src0 << 16) | (src1 << 24))
    return fixture.vop2(opcode, 3, 249, reg1) + [ext]


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
    vgpr_error = 'SDWA binary16 source must be a VGPR'
    return [
        (sdwa(add, DWORD, PAD, WORD_0, WORD_0), dest_error),
        (sdwa(add, WORD_0, PAD, BYTE_1, WORD_0), src_error),
        (sdwa(add, WORD_0, PAD, WORD_0 | SCALAR, WORD_0), vgpr_error),
        (sdwa(add, WORD_0, PAD, WORD_0, WORD_0 | SCALAR, reg1=242), vgpr_error),
        (sdwa(add, WORD_0, PAD, WORD_0, WORD_0 | 8), src_error),
        (sdwa(add, WORD_0, SEXT, WORD_0, WORD_0), dest_error),
        (sdwa(add, WORD_0, PAD, WORD_0, WORD_0, dst_mod=1 << 13), dest_error),
        (sdwa(add, WORD_0, PAD, WORD_0, WORD_0, dst_mod=1 << 14), dest_error),
        (sdwa(0x39, WORD_0, PAD, WORD_0, WORD_0), 'unimplemented direct SDWA operand profile'),  # max
        (sdwa(0x3a, WORD_0, PAD, WORD_0, WORD_0), 'unimplemented direct SDWA operand profile'),  # min
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


def run_gpu(gpu, sdk, items, code):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src, root = work / 'root.hlsl', work / 'root.bin'
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([str(Path(sdk) / 'dxc.exe'), '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, src], check=True)
        sa = struct.pack('<7I', 0, 0, 0, 32 | (1 << 16), 1, 0, 0x80 | (5 << 1))
        marker = write_request(work, code, root.read_bytes(), sa)
        dest = work / 'sdwa-f16-readback.bin'
        run = subprocess.run([gpu, sdk, str(marker), str(work / 'cache'), '0', str(dest), '--direct'])
        if run.returncode == 77:
            sys.exit(77)
        run.check_returncode()
        offsets = fixture.group_offsets(len(items))
        stores = {slot: case for group in offsets for slot, case in zip(group, items)}
        actual = struct.unpack(f'<{fixture.WORDS}I', dest.read_bytes())
        for index, got in enumerate(actual):
            if index in stores:
                case = stores[index]
                assert case.matches(got), (index, case.op, hex(got), hex(case.want))
            else:
                assert got == GUARD, (index, hex(got), hex(GUARD))
    print('native_sdwa_f16 GPU: %d results and %d guards' % (len(stores), fixture.WORDS - len(stores)))


if __name__ == '__main__':
    items = cases()
    check_mutants(items)
    code, _ = fixture.program(items)
    if len(sys.argv) == 3 and sys.argv[1] == 'ir':
        run_ir(sys.argv[2], items, code)
    elif len(sys.argv) == 4 and sys.argv[1] == 'gpu':
        run_gpu(sys.argv[2], sys.argv[3], items, code)
    else:
        sys.exit(__doc__)
