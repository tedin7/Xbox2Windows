"""Direct DXIL regression for signed/unsigned three-source 32-bit integer min/max/median.

The generic lowering predates this fixture (4036577). Earlier Python paths
are covered by #1992, #1574, #1577 and #308. Exercise all six VOP3 encodings
through the native compiler, not the Python translator.

Two full 32-thread groups write six words per lane. Sparse guest EXEC
suppresses the ALU writes, not the final stores: inactive lanes must keep
the old destination (including an S0/destination alias for signed median).
Full physical waves avoid the WARP 1.0.22 partial-wave ballot issue (#3160).
"""
from itertools import permutations
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove

OPS = (("v_min3_u32", 0x153, 0, False), ("v_max3_u32", 0x156, 2, False),
       ("v_med3_u32", 0x159, 1, False), ("v_min3_i32", 0x152, 0, True),
       ("v_max3_i32", 0x155, 2, True), ("v_med3_i32", 0x158, 1, True))
ALL = 0xffffffff
SPARSE = 0xa55aa55a
SEED = 0x13579bdf
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
CASES = [(t, ALL) for t in permutations((0x80000000, 0x7fffffff, 0xffffffff))]
CASES += [((0, 0, 0xffffffff), ALL), ((1, 0, 1), ALL),
          ((0x80000000, 0x80000000, 0x80000000), ALL),
          ((0xffffffff, 0x7fffffff, 0x80000000), SPARSE),
          ((0, 1, 0xffffffff), SPARSE)]


def program(triple, active=ALL, ops=OPS, clamp=False):
    # Root table s4, group.x in s5, 1024 bytes/group, 32 bytes/lane.
    p = [sop1(3, 20, 4), sop1(3, 21, 128), 0xf408080a, 0xfa000020,
         scalar(0x1e, 7, 5, 138), vop2(0x1a, 8, 133, 0), vop2(0x25, 8, 7, 8)]
    for dst, value in enumerate(triple, 1):
        p += [vmove(dst, 255), value]
    for dst in range(10, 15):
        p += [vmove(dst, 255), SEED]
    p += [sop1(3, 126, 255), active]
    destinations = []
    for j, (_, opcode, _, _) in enumerate(ops):
        # Last op aliases S0; all other destinations are disjoint.
        dst = 1 if j == len(ops) - 1 else 10 + j
        destinations.append(dst)
        p += [0xd4000000 | opcode << 16 | dst | (0x8000 if clamp else 0),
              257 | 258 << 9 | 259 << 18]
    p += [sop1(3, 126, 193), 0xbf8cc07f]
    for j, dst in enumerate(destinations):
        p += [0xe0701000 | 4*j, 0x80080008 | dst << 8]
    return p + [0xbf810000]


def reference(triple, active, wrong_sign=False, drop_third=False, ignore_exec=False, wrong_rank=False):
    expected = [SENTINEL] * OUTPUT_WORDS
    operands = triple[:2] + triple[1:2] if drop_third else triple
    results = []
    for _, _, rank, signed in OPS:
        order = sorted(operands, key=lambda v: v - (1 << 32) if (signed != wrong_sign) and v & 0x80000000 else v)
        results.append(order[(rank + 1) % 3 if wrong_rank else rank])
    for n in range(64):
        base = GUARD_WORDS + n * 8
        enabled = ignore_exec or (active >> (n % 32)) & 1
        expected[base:base+6] = results if enabled else [SEED]*5 + [triple[0]]
    return expected


def compile_case(cli, sdk, work, root, p):
    marker = request(work, root, p)
    output = marker.with_suffix(".cso")
    result = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                            capture_output=True, text=True)
    return result, output


def controls(cli, sdk, work, root):
    for op in OPS:
        r, out = compile_case(cli, sdk, work, root, program(CASES[0][0], ops=(op,)))
        assert r.returncode == 0 and out.is_file(), (op, r.stdout, r.stderr)
        r, out = compile_case(cli, sdk, work, root, program(CASES[0][0], ops=(op,), clamp=True))
        assert r.returncode and "direct ALU operand modifiers" in r.stderr and not out.exists(), (op, r.stdout, r.stderr)
    for mutation in ("wrong_sign", "drop_third", "ignore_exec", "wrong_rank"):
        assert any(reference(t, a) != reference(t, a, **{mutation: True}) for t, a in CASES), mutation
    # Ensure every opcode independently detects swapped signedness.
    for j in range(len(OPS)):
        assert any(reference(t, a)[GUARD_WORDS+j] != reference(t, a, wrong_sign=True)[GUARD_WORDS+j]
                   for t, a in CASES if a == ALL), OPS[j][0]
    print("six canonical forms compile; six clamp controls refused; four incorrect references differ")
    return 0


def hardware(replay, sdk, work, root):
    for triple, active in CASES:
        marker = request(work, root, program(triple, active))
        output = work / "out.bin"
        r = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if r.returncode == 77:
            return 77
        r.check_returncode()
        actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
        expected = reference(triple, active)
        bad = [(i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        assert not bad, (triple, hex(active), bad[:10])
        print("triple", *(f"{v:08x}" for v in triple), "EXEC", f"{active:08x}", f"{len(actual)} words match", flush=True)
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        src = work / "root.hlsl"
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        rootfile = work / "root.bin"
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", rootfile, src], check=True)
        return {"controls": controls, "hardware": hardware}[mode](tool, sdk, work, rootfile.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
