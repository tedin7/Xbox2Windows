"""Direct compute v_xnor_b32 (VOP2, VOP2 literal, VOP3A, and VOP2 SDWA)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969

CASES = (
    (0x00000000, 0x00000000),  # identical zeros -> 0xFFFFFFFF
    (0xFFFFFFFF, 0xFFFFFFFF),  # identical ones -> 0xFFFFFFFF
    (0xFFFFFFFF, 0x00000000),  # complementary -> 0x00000000
    (0xAAAAAAAA, 0x55555555),  # alternating complements -> 0x00000000
    (0x12345678, 0x9ABCDEF0),  # general 32-bit pattern (byte0=0x78 positive)
    (0x12345680, 0x8000FF7F),  # byte0=0x80 sign-extends to 0xFFFFFF80 in SDWA
    (0x00FF00FF, 0xF0F0F0F0),  # byte0=0xFF sign-extends to 0xFFFFFFFF in SDWA
    (0x76543210, 0x76543210),  # identical non-trivial -> 0xFFFFFFFF (VOP2/VOP3A)
)


def v_mov_lit(vdst, lit):
    return [0x7E0002FF | ((vdst & 0xFF) << 17), lit & MASK]


def vop2_xnor(vdst, src0, vsrc1):
    return (0x1E << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)


def vop3_xnor(vdst, src0, src1):
    return [0xD4000000 | (0x11E << 16) | (vdst & 0xFF), (src0 & 0x1FF) | ((src1 & 0x1FF) << 9)]


def vop2_xnor_sdwa(vdst, vsrc0, vsrc1, dst_sel=6, dst_unused=0, src0_ctrl=0x08, src1_ctrl=0x04):
    ext = ((vsrc0 & 0xFF) | ((dst_sel & 7) << 8) | ((dst_unused & 3) << 11) |
           ((src0_ctrl & 0xFF) << 16) | ((src1_ctrl & 0xFF) << 24))
    return [vop2_xnor(vdst, 249, vsrc1), ext]


def program():
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for ci, (a, b) in enumerate(CASES):
        code += v_mov_lit(2, a) + v_mov_lit(3, b)
        # 0: VOP2 VGPR + VGPR
        code += [vop2_xnor(4, 258, 3), 0xE0701000 | (4 * (ci * 4 + 0)), 0x80080408]
        # 1: VOP2 literal + VGPR
        code += [vop2_xnor(4, 255, 3), a, 0xE0701000 | (4 * (ci * 4 + 1)), 0x80080408]
        # 2: VOP3A VGPR + VGPR
        code += vop3_xnor(4, 258, 259) + [0xE0701000 | (4 * (ci * 4 + 2)), 0x80080408]
        # 3: VOP2 SDWA (src0=SEXT(BYTE_0), src1=WORD_0, dst=DWORD)
        code += vop2_xnor_sdwa(4, 2, 3) + [0xE0701000 | (4 * (ci * 4 + 3)), 0x80080408]
    return code + [0xBF810000]


def xnor32(a, b, omit_invert=False):
    x = (a ^ b) & MASK
    return x if omit_invert else (~x) & MASK


def sext8(v):
    b = v & 0xFF
    return (b - 0x100) & MASK if (b & 0x80) else b


def reference(omit_invert=False):
    words = [SENTINEL] * 8
    for _lane in range(64):
        for a, b in CASES:
            plain = xnor32(a, b, omit_invert=omit_invert)
            sdwa = xnor32(sext8(a), b & 0xFFFF, omit_invert=omit_invert)
            words.extend((plain, plain, plain, sdwa))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    marker = request(work, root, program())
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                          text=True, check=True).stdout
    assert 'xor i32 ' in dump and ('-1' in dump or '4294967295' in dump), dump
    assert reference() != reference(omit_invert=True)
    print('v_xnor_b32 DXIL verified; xor-without-invert control differs')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    expected = reference()
    actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
    mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
    if mismatch is not None:
        raise AssertionError(f'word {mismatch}: got {actual[mismatch]:08x}, expected {expected[mismatch]:08x}')
    print('v_xnor_b32: 2048 result words and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_v_xnor_b32_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
