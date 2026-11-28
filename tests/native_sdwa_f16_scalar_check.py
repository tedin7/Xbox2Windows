"""Direct compute binary16 SDWA scalar sources. Use ir DUMPER or gpu STORE_REPLAY SDK."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import f16_packed_model as model
import native_f16_alu_check as fixture
from native_compute_clear_check import write_request
from native_sdwa_f16_check import DWORD, OLD, PAD, PRESERVE, SCALAR, WORD_0, WORD_1, sdwa

SGPR = 9
WORDS = {SGPR: 0x44004000, 256 + 1: 0x48003c00, 256 + 2: 0x45004200}
SOURCE_WORDS = {**WORDS, **fixture.INLINE_F16, 128: 0}


def instruction(op, unused, left, left_sel, right, right_sel):
    def control(source, selector):
        return selector | (SCALAR if source < 256 else 0)

    return sdwa(op, WORD_0, unused, control(left, left_sel), control(right, right_sel),
                reg0=left & 255, reg1=right & 255)


class Case:
    dest = 3

    def __init__(self, op, unused, left, left_sel, right, right_sel):
        self.instruction = instruction(fixture.OP_CODE[op], unused, left, left_sel, right, right_sel)

        def half(source, selector):
            return model.pick(SOURCE_WORDS[source], selector == WORD_1)

        result = fixture.binary16(op, half(left, left_sel), half(right, right_sel))
        self.want = (OLD & 0xffff0000 if unused == PRESERVE else 0) | result

    def emit(self):
        return (fixture.smov(SGPR, WORDS[SGPR]) +
                fixture.mov(1, WORDS[257]) + fixture.mov(2, WORDS[258]) +
                fixture.mov(self.dest, OLD) + self.instruction)


def cases():
    return [
        Case('add', PAD, SGPR, DWORD, 258, WORD_0),
        Case('add', PRESERVE, 242, WORD_0, 128, DWORD),
        Case('mul', PRESERVE, 257, WORD_0, SGPR, WORD_1),
        Case('sub', PAD, 257, WORD_1, SGPR, DWORD),
        Case('subrev', PAD, 257, WORD_1, SGPR, WORD_0),
    ]


def refusals():
    add = fixture.OP_CODE['add']
    return [
        (instruction(add, PAD, 255, WORD_0, 258, WORD_0), 'literal is unavailable in SDWA'),
        (instruction(add, PAD, 257, WORD_0, SGPR, WORD_1 | 8), 'invalid SDWA source modifier'),
        (instruction(add, PAD, 258, WORD_0, 242, WORD_1), 'unsupported SDWA binary16 inline constant selection'),
        # These captured min/max forms also need an arithmetic lowering.
        (instruction(0x39, PRESERVE, SGPR, DWORD, 258, WORD_1), 'unimplemented direct SDWA operand profile'),
        (instruction(0x3a, PAD, 257, WORD_0, 258, WORD_0), 'unimplemented direct SDWA operand profile'),
    ]


def run_ir(dumper, items):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Isolate each shape so one decode refusal cannot hide another.
        for index, case in enumerate(items):
            code, _ = fixture.program([case])
            run = fixture.dump(dumper, work, code, f'case-{index}')
            assert run.returncode == 0, (index, run.stderr)
            stores = fixture.interpret(run.stdout)
            assert stores == [(0, case.want)], (index, stores, hex(case.want))
        for index, (words, why) in enumerate(refusals()):
            run = fixture.dump(dumper, work, words + [0xbf810000], f'refused-{index}')
            assert run.returncode != 0 and why in run.stderr, (why, run.returncode, run.stderr)
    print(f'native_sdwa_f16_scalar IR: {len(items)} results and {len(refusals())} refusals')


def run_gpu(gpu, sdk, items):
    code, expected = fixture.program(items)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src, root = work / 'root.hlsl', work / 'root.bin'
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([str(Path(sdk) / 'dxc.exe'), '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, src], check=True)
        sa = struct.pack('<7I', 0, 0, 0, 32 | (1 << 16), 1 | (1 << 20), 0, 0x80 | (5 << 1))
        marker = write_request(work, code, root.read_bytes(), sa)
        dest = work / 'sdwa-f16-scalar-readback.bin'
        run = subprocess.run([gpu, sdk, str(marker), str(work / 'cache'), '0', str(dest), '--direct'])
        if run.returncode == 77:
            sys.exit(77)
        run.check_returncode()
        # The preamble sets EXEC_LO=1. Group IDs 0 and 1 in s5 select
        # wave32 records at DWORDs 8 and 136, with contiguous lane-0 results.
        stores = {slot: want for offsets in fixture.group_offsets(len(items))
                  for slot, want in zip(offsets, expected)}
        actual = struct.unpack(f'<{fixture.WORDS}I', dest.read_bytes())
        for index, got in enumerate(actual):
            want = stores.get(index, fixture.SENTINEL)
            assert got == want, (index, hex(got), hex(want))
    print(f'native_sdwa_f16_scalar GPU: {len(stores)} results and {fixture.WORDS - len(stores)} guards')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'ir':
        run_ir(sys.argv[2], cases())
    elif len(sys.argv) == 4 and sys.argv[1] == 'gpu':
        run_gpu(sys.argv[2], sys.argv[3], cases())
    else:
        sys.exit(__doc__)
