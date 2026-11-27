"""global_load_dword through a root SRV address in the direct C++ compute path (#2962).

  native_global_load_check.py <compile_dxil> <native_store_replay> <sdk bin dir>

The saddr form reads ADDR = SADDR + zext(VADDR) + sext(OFFSET[11:0]) (RDNA 2 ISA, FLAT/GLOBAL addressing). The
direct frontend accepts SADDR only as the unadvanced root address of a root SRV, the binding its scalar root loads
already use, guards the 32-bit byte offset against wrapping, and drops the low two address bits like
buffer_load_dword in this path. All words below were assembled with LLVM 15 (clang -cc1as, gfx1013).

The dispatch (two groups of 32 lanes) reads the root SRV that native_store_replay fills with 0x51aa0000 + dword
index. An s_load_dword at byte 252 sizes the view to 64 dwords. Each lane L stores five words:
  A  v1 = 4L                          -> word L
  B  v1, offset:16                    -> word L + 4
  C  v1, offset:-8                    -> word L - 2, or 0 for L < 2 (the byte offset would be negative)
  D  v2 = 0xfffffff0 | 4(L & 3), +16  -> 0: zext(v2) + 16 is at least 2^32 (unguarded it would wrap in bounds)
  E  v3 = 4L + 1                      -> word L (low address bits dropped)
References with the offset dropped or the wrap guard dropped must each differ from the readback. The lower guard
of C is not observable here: a wrapped negative offset lies near 2^32, outside any view below 4 GiB, and reads 0
either way. Refusals are checked by message, and the module's loads and guards are counted before the
GPU run, which needs shader model 6.6 with wave32 (native_store_replay returns 77 otherwise).
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import sop1, vmove


def words(hexes):
    out = []
    for h in hexes.split():
        out += [int.from_bytes(bytes.fromhex(h[i:i + 8]), 'little') for i in range(0, len(h), 8)]
    return out


ROOT = 'SRV(t0),DescriptorTable(UAV(u0,numDescriptors=8))'
SA = struct.pack('<7I', 0, 0, 0, 32 | 1 << 16, 1, 0, 0x80 | 7 << 1)
# s_mov_b32 s20, s6 / s_mov_b32 s21, 0 / s_load_dwordx4 s[40:43], s[20:21], 0x20 / s_load_dword s30, s[4:5], 0xfc /
# s_lshl_b32 s12, s7, 10 / v_lshlrev_b32 v8, 5, v0 / v_add_nc_u32 v8, s12, v8 / v_lshlrev_b32 v1, 2, v0 /
# v_and_b32 v2, 3, v0 / v_lshlrev_b32 v2, 2, v2 / v_or_b32 v2, -16, v2 / v_add_nc_u32 v3, 1, v1
PROLOGUE = words('060394be 800395be 0a0a08f4200000fa 820700f4fc0000fa 078a0c8f 85001034 0c10104a 82000234 '
                 '83000436 82040434 d0040438 8102064a')
LOADS = {
    'A': '008030dc01000414',  # global_load_dword v20, v1, s[4:5]
    'B': '108030dc01000415',  # global_load_dword v21, v1, s[4:5] offset:16
    'C': 'f88f30dc01000416',  # global_load_dword v22, v1, s[4:5] offset:-8
    'D': '108030dc02000417',  # global_load_dword v23, v2, s[4:5] offset:16
    'E': '008030dc03000418',  # global_load_dword v24, v3, s[4:5]
}
# s_waitcnt vmcnt(0) lgkmcnt(0), five buffer_store_dword v20..v24, v8, s[40:43], 0 offen offset:0..16, s_endpgm
EPILOGUE = words('70008cbf 001070e008140a80 041070e008150a80 081070e008160a80 0c1070e008170a80 '
                 '101070e008180a80 000081bf')
REFUSALS = [
    ('008030dc01007d14', 'DWORD loads with a scalar address pair'),               # v[1:2], off
    ('008030dc01001414', 'global load base is not a root buffer address'),        # s[20:21], a table pointer
    ('028030dc01000414', 'global load at an unaligned immediate offset'),         # offset:2
    ('008034dc010004ff', 'global load destination register range'),             # x2 beginning at v255
    ('00803cdc010004fe', 'global load destination register range'),             # x3 beginning at v254
    ('008038dc010004fd', 'global load destination register range'),             # x4 beginning at v253
    ('008070dc01140400', 'DWORD loads with a scalar address pair'),             # global_store_dword
    ('800385be 008030dc01000414', 'global load base is an advanced or mismatched root address'),  # s_mov_b32 s5, 0
    ('04900480 008030dc01000414', 'unbound numeric SGPR s4'),  # s_add_u32 s4, s4, 16 on the root address
    ('00a030dc01000414', 'DWORD loads with a scalar address pair'),               # the lds bit
    ('000030dc01007d14', 'DWORD loads with a scalar address pair'),               # flat_load_dword (segment 0)
    ('004030dc01007d14', 'DWORD loads with a scalar address pair'),               # scratch_load_dword (segment 1)
    ('0080c8dc01140400', 'DWORD loads with a scalar address pair'),               # global_atomic_add
]
POLICY = '009033dc01000414'  # global_load_dword v20, v1, s[4:5] glc slc dlc: accepted
INACTIVE = 0x13579bdf
WIDE_OPS = {2: 13, 3: 15, 4: 14}


def reference(drop_offset=False, drop_wrap_guard=False):
    expected = [0x69696969] * 2064
    for group in range(2):
        for lane in range(32):
            base = 8 + (group * 32 + lane) * 8

            def word(byte):
                return 0x51aa0000 + byte // 4 if 0 <= byte < 256 else 0
            b = word(4 * lane + (0 if drop_offset else 16))
            c_byte = 4 * lane - 8
            c = word(c_byte) if c_byte >= 0 else 0
            d_byte = 0xfffffff0 | 4 * (lane & 3)
            d = word((d_byte + 16) & 0xffffffff) if drop_wrap_guard else 0
            expected[base:base + 5] = [word(4 * lane), b, c, d, word(4 * lane)]
    return expected


def wide_program(width, case, *, reverse=False):
    code = PROLOGUE.copy()
    dst, coord, offset = 20, 1, 0
    if case == 'alias':
        dst, offset = 1, 16
    elif case == 'tail':
        code += [vop2(0x25, 1, 255, 1), 244]
    elif case == 'sparse':
        offset = -4
        for j in range(width):
            code += [vmove(dst + j, 255), INACTIVE]
        code += [sop1(3, 126, 255), 0x55555555]
    elif case == 'wrap':
        coord, offset = 2, 16
    elif case == 'extent':
        offset = 2044
    else:
        raise ValueError(case)
    code += [0xdc008000 | WIDE_OPS[width] << 18 | (offset & 0xfff), dst << 24 | 4 << 16 | coord]
    code.append(0xbf8c0070)  # s_waitcnt vmcnt(0) lgkmcnt(0)
    if case == 'sparse':
        code.append(sop1(3, 126, 193))
    for j in range(width):
        source = dst + (width - 1 - j if reverse else j)
        code += [0xe0701000 | (j * 4), 0x800a0008 | source << 8]
    return code + [0xbf810000]


def wide_reference(width, case):
    expected = [0x69696969] * 2064
    for lane in range(64):
        local = lane % 32
        if case == 'sparse' and local % 2:
            values = [INACTIVE] * width
        else:
            start = {'alias': local + 4, 'tail': local + 61, 'sparse': local - 1, 'wrap': 1 << 30}[case]
            values = [0x51aa0000 + start + j if start >= 0 and start + j < 64 else 0 for j in range(width)]
        base = 8 + lane * 8
        expected[base:base + width] = values
    return expected


def main(compiler, replay, sdk):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        source, root = work / 'root.hlsl', work / 'root.bin'
        source.write_text('#define SIG "%s"' % ROOT)
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, source], check=True)

        def compile_words(code):
            marker = write_request(work, code, root.read_bytes(), SA)
            return marker, subprocess.run([compiler, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker,
                                           marker.with_suffix('.cso')], capture_output=True, text=True)

        for code, message in REFUSALS:
            _, run = compile_words(PROLOGUE + words(code) + EPILOGUE)
            assert run.returncode and message in run.stderr, (code, message, run.stdout, run.stderr)
        _, run = compile_words(PROLOGUE + words(POLICY) + EPILOGUE)
        assert run.returncode == 0, run.stderr
        print('refused: off, a table pointer, offset:2, VGPR overflows, store, a replaced high half, an advanced pointer, '
              'lds, flat and scratch segments, an atomic; glc slc dlc accepted')

        code = PROLOGUE + words(' '.join(LOADS.values())) + EPILOGUE
        marker, run = compile_words(code)
        assert run.returncode == 0, run.stderr
        assert 't0 root=0 bytes=0 min_size=256' in run.stdout, run.stdout
        # The module, before the GPU run: five global loads plus the scalar root load, the two upper-bound guards
        # (B and D) and the lower-bound guard (C), each a select on a raw buffer load.
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', marker.with_suffix('.cso')],
                              capture_output=True, text=True, check=True).stdout
        counts = [dump.count(t) for t in ('rawBufferLoad.i32(i32 139', 'icmp ule i32', 'icmp uge i32')]
        assert counts == [6, 2, 1], counts
        print('module: 6 raw buffer loads, 2 upper-bound and 1 lower-bound guards')
        def replay_words(request):
            dest = work / 'readback.bin'
            result = subprocess.run([replay, sdk, request, work / 'cache', '0', dest, '--direct'],
                                    capture_output=True, text=True)
            if result.returncode == 77:
                print(result.stdout + result.stderr)
                sys.exit(77)
            assert result.returncode == 0, (result.stdout, result.stderr)
            return list(struct.unpack('<2064I', dest.read_bytes()))

        actual = replay_words(marker)
        expected = reference()
        diff = [(i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        assert not diff, diff[:10]
        for label, mutant in (('offset dropped', reference(drop_offset=True)),
                              ('wrap guard dropped', reference(drop_wrap_guard=True))):
            assert mutant != expected and actual != mutant, label
        print('global_load_dword: 320 words of two groups matched (offsets 0, 16, -8, the 2^32 guard and an '
              'unaligned address); the offset-dropped and wrap-guard-dropped references differ')
        for width in WIDE_OPS:
            for case in ('alias', 'tail', 'sparse', 'wrap'):
                marker, run = compile_words(wide_program(width, case))
                assert run.returncode == 0, (width, case, run.stdout, run.stderr)
                got, expected = replay_words(marker), wide_reference(width, case)
                diff = [(i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(got, expected, strict=True)) if a != b]
                assert not diff, (width, case, diff[:10])
            _, extent = compile_words(wide_program(width, 'extent'))
            assert extent.returncode == 0 and f't0 root=0 bytes=0 min_size={2044 + width * 4}' in extent.stdout, (
                width, extent.stdout, extent.stderr)
            marker, run = compile_words(wide_program(width, 'alias', reverse=True))
            assert run.returncode == 0, run.stderr
            wrong = replay_words(marker)
            changed = sum(a != b for a, b in zip(wrong, wide_reference(width, 'alias'), strict=True))
            assert changed == 64 * (width - (width % 2)), (width, changed)
            print(f'global_load_dwordx{width}: alias, tail, sparse EXEC and wrap match; reversing results changes {changed} words')


if __name__ == '__main__':
    main(*sys.argv[1:])
