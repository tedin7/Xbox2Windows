"""Draw source-built guest VS programs whose per-vertex VCC crosses a two-arm join."""
import re
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from native_direct_container_check import signature_rows, vertex
from native_scalar_lds_check import sop1, vmove

NOP = 0xbf800000
ARM_WORD = 8
ARM_B_COMPARE = 15
JOIN = 17
LANE_JOIN = 17


def cndmask(dst, taken, other):
    # dst = VCC ? v[taken] : other
    return 0x02000000 | dst << 17 | taken << 9 | other


def program(arm_b):
    # Fullscreen triangle (-1,-1), (4,-1), (-1,4) from SV_VertexID in v0.
    seq = [vmove(6, 246),
           0x7d040081, cndmask(1, 6, 243),      # x = id == 1 ? 4 : -1
           0x7d040082, cndmask(2, 6, 243),      # y = id == 2 ? 4 : -1
           vmove(3, 128), vmove(7, 242),
           # The harness supplies zero in b0, so the arm is chosen at draw time.
           sop1(3, 4, 255), int(arm_b), 0x89040004, 0xbf008004,
           0xbf840003,                          # s4 != 0: arm B
           0x7d040081, vmove(4, 244), 0xbf820002,   # A: VCC = id == 1, v4 = 2
           0x7d0a0082, vmove(4, 246),               # B: VCC = id != 2, v4 = 4
           vmove(5, 240), cndmask(5, 4, 261),   # after the join: VCC ? v4 : 0.5
           0xf80000cf, 0x07030201, 0xf8000a0f, 0x05050405, 0xbf810000]
    assert seq[ARM_WORD] == arm_b and seq[ARM_B_COMPARE] == 0x7d0a0082 and seq[JOIN] == vmove(5, 240)
    return seq


def lane_program():
    # The compare itself splits the wave: vertex 1 runs THEN, the others ELSE.
    # VCC is written once, before the region, and read after EXEC is restored.
    # The frontend lowers these arms to per-vertex selects in one block.
    seq = program(False)[:7]
    # A masked write keeps the previous v4 on the other lanes.
    seq += [vmove(4, 242),
            0x7d040081, 0xbe88246a, 0xbf880002,     # VCC = id == 1; save EXEC; skip THEN
            vmove(4, 244), NOP,                      # THEN: v4 = 2
            0x8afe7e08, 0xbf880001, vmove(4, 246),   # ELSE: v4 = 4
            0xbefe0408,
            0x101008f3, cndmask(5, 4, 264),          # v8 = -v4; VCC ? v4 : v8
            0xf80000cf, 0x07030201, 0xf8000a0f, 0x05050405, 0xbf810000]
    assert seq[LANE_JOIN] == 0x101008f3
    return seq


def container(seq):
    outputs = signature_rows([(b"TEXCOORD", 0, 3, 0, 15), (b"SV_Position", 1, 3, 1, 15)])
    return vertex(constants=(0, 1), program=seq, varying=True, extra=((b"OSG1", outputs),))


def reference(arm_b):
    """Per-vertex CPU model: three selected values, then the arm's own value."""
    arm = 4.0 if arm_b else 2.0
    vcc = [(vid != 2) if arm_b else (vid == 1) for vid in range(3)]
    return [arm if bit else 0.5 for bit in vcc] + [arm]


def main(cli, raster, sdk, backend):
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        source, vs, ps, expected = work/"source", work/"vs.cso", work/"ps.cso", work/"expected.bin"
        # Every pixel reports the unblended value of each of the three vertices.
        source.write_text("float4 main(nointerpolation float4 v:TEXCOORD, float4 p:SV_Position):SV_Target {"
                          "return float4(GetAttributeAtVertex(v,0).x,GetAttributeAtVertex(v,1).x,"
                          "GetAttributeAtVertex(v,2).x,GetAttributeAtVertex(v,0).y);}")
        subprocess.run([Path(sdk)/"dxc.exe", "-T", "ps_6_6", "-Fo", ps, source], check=True)

        def draw(seq, words):
            source.write_bytes(container(seq))
            subprocess.run([cli, sdk, work/"cache", "gfx1013-container-vs", source, vs], check=True)
            expected.write_bytes(struct.pack("<4f", *words)*8)
            result = subprocess.run([raster, backend, "--startup-pixel-words", vs, ps, "4", "2", expected],
                                    capture_output=True, text=True)
            if result.returncode == 77:
                print(result.stdout, result.stderr)
                sys.exit(77)
            return result

        def differs(result):
            assert result.returncode == 1 and "8 of 8 pixels differ" in result.stderr, (result.stdout, result.stderr)
            return result.stderr.splitlines()[0]

        for arm_b in (False, True):
            result = draw(program(arm_b), reference(arm_b))
            assert result.returncode == 0, (result.stdout, result.stderr)
            ir = subprocess.check_output([Path(sdk)/"dxc.exe", "-dumpbin", vs], text=True)
            # Each arm supplies its own compare to the mask the select reads.
            joined = re.search(r"(%\w+) = phi i32 \[ (%\w+), %g\d+ \], \[ (%\w+), %g\d+ \]\n", ir)
            assert joined and joined[2] != joined[3] and "@dx.op.wave" not in ir, ir
            mask = re.search(rf"(%\w+) = icmp ne i32 {joined[1]}, 0\n", ir)
            assert mask and f"select i1 {mask[1]}," in ir, ir
            print(*(line for line in result.stdout.splitlines()
                    if line.startswith(("backend=", "adapter=", "driver=", "debug_"))), sep="\n")
            print(f"arm {'AB'[arm_b]}: vertices 0..2 and arm value matched {reference(arm_b)}")
        # Vertex 0 supplies the fourth word, so it is the ELSE value.
        lane = [-4.0, 2.0, -4.0, 4.0]
        result = draw(lane_program(), lane)
        assert result.returncode == 0, (result.stdout, result.stderr)
        print(f"per-vertex arms: THEN on vertex 1, ELSE on vertices 0 and 2 matched {lane}")
        uniform = lane_program()
        uniform.insert(LANE_JOIN, 0x7d040100)
        print("uniform mask after the per-vertex arms:", differs(draw(uniform, lane)))
        # Control: arm B no longer contributes its own compare. The join
        # then receives the stale id == 2 mask on that edge only.
        dropped = program(True)
        dropped[ARM_B_COMPARE] = NOP
        print("dropped arm B contribution:", differs(draw(dropped, reference(True))))
        dropped[ARM_WORD] = 0
        assert draw(dropped, reference(False)).returncode == 0
        # Control: a uniform v0 == v0 replaces the joined mask.
        for arm_b in (False, True):
            uniform = program(arm_b)
            uniform.insert(JOIN, 0x7d040100)
            print(f"uniform mask after the join, arm {'AB'[arm_b]}:", differs(draw(uniform, reference(arm_b))))


if __name__ == "__main__":
    main(*sys.argv[1:])
