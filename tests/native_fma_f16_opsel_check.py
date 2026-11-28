"""Direct VOP3 v_fma_f16 source and destination half selection."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import f16_packed_model as model
from native_compute_clear_check import write_request
import native_f16_alu_check as F
from native_fma_f16_check import OP_FMA_VOP3, OP_FMAC, vop3_tern


SOURCES = (0x40003C00, 0x44004200, 0x4C004500)  # (high, low): (2, 1), (4, 3), (16, 5)
DST = 0xABCD1234
END = 0xBF810000


def fma(srcs, op_sel, *, clamp=0):
    return vop3_tern(OP_FMA_VOP3, 3, *srcs, op_sel=op_sel, clamp=clamp)


def program():
    # s5 supplies the group X ID (0 or 1). Lane 0 alone writes each dword.
    # Case i in group g stores at byte 512*g + 4*i relative to the UAV window,
    # which starts at readback word 8, so the file word index is 8 + 128*g + i.
    # The preamble sets EXEC to 1 before the per-lane VGPR initialization.
    code = F.preamble() + F.mov(1, SOURCES[0]) + F.mov(2, SOURCES[1]) + F.mov(4, SOURCES[2])
    want = []
    for op_sel in (1, 2, 3, 15, 14, 12, 8):
        srcs = (257, 258, 260)
        words = SOURCES
        if op_sel == 8:
            srcs = (244, 258, 243)  # inline binary16 2.0 and -1.0
            words = (0x4000, SOURCES[1], 0xBC00)
        halves = [model.pick(word, op_sel & (1 << j)) for j, word in enumerate(words)]
        result = model.half_op('fma', *halves)
        code += F.mov(3, DST) + fma(srcs, op_sel) + F.store(4 * len(want), 3)
        want.append(((result << 16) | (DST & 0xFFFF)) if op_sel & 8 else ((DST & 0xFFFF0000) | result))
    assert want == [0xABCD4980, 0xABCD4880, 0xABCD4A80, 0x4E001234, 0x4D001234, 0x4CC01234, 0x45001234]
    return code + [END], want


def refusals():
    return (
        (F.smov(10, SOURCES[0]) + fma((10, 258, 260), 1),
         'unsupported v_fma_f16 high-half op_sel on non-VGPR source'),
        (F.vop2(OP_FMAC, 3, 250, 2) + [0xFF00B101], 'unsupported DPP arithmetic'),
        (fma((257, 258, 260), 8, clamp=1), 'unsupported v_fma_f16 high-half op_sel'),
    )


def ir(dumper, work, code, want):
    run = F.dump(dumper, work, code, 'fma-f16-opsel')
    assert run.returncode == 0, run.stderr
    stores = sorted(F.interpret(run.stdout))
    assert stores == [(4 * i, word) for i, word in enumerate(want)], stores
    for i, (snippet, why) in enumerate(refusals()):
        run = F.dump(dumper, work, F.preamble() + snippet + [END], f'refused-{i}')
        assert run.returncode != 0 and why in run.stderr, (why, run.returncode, run.stderr)
    print(f'native_fma_f16_opsel IR matched {len(want)} cases and {len(refusals())} refusals')
    return 0


def hardware(replay, sdk, work, code, want):
    shader, root = work / 'root.hlsl', work / 'root.bin'
    shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
    subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
    # .sa dword 4, bit 20 requests wave32.
    sa = struct.pack('<7I', 0, 0, 0, 32 | (1 << 16), 1 | (1 << 20), 0, 0x8A)
    marker = write_request(work, code, root.read_bytes(), sa)
    output = work / 'out.bin'
    run = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if run.returncode == 77:
        return 77
    run.check_returncode()
    got = struct.unpack(f'<{F.WORDS}I', output.read_bytes())
    expected = {slot: value for slots in F.group_offsets(len(want))
                for slot, value in zip(slots, want)}
    bad = [(i, hex(word), hex(expected.get(i, F.SENTINEL)))
           for i, word in enumerate(got) if word != expected.get(i, F.SENTINEL)]
    assert not bad, bad[:8]
    print(f'native_fma_f16_opsel hardware matched {len(want)} cases across {F.GROUPS} groups')
    return 0


def main(mode, tool, sdk=None):
    code, want = program()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if mode == 'ir':
            return ir(tool, work, code, want)
        assert mode == 'hardware' and sdk is not None
        return hardware(tool, sdk, work, code, want)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
