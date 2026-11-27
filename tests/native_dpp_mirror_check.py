"""GPU differential for source-built, direct-DXIL guest row mirrors."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_direct_container_check import patched_pixel, vertex
from native_scalar_lds_check import scalar, sop1, vmove


def vop2(op, dst, src0, src1):
    return op << 25 | dst << 17 | src1 << 9 | src0


def compute_program(control, active):
    # v1 holds a unique value even after guest EXEC is narrowed. The DPP
    # src0 aliases the destination. A distinct local src1 distinguishes
    # suppress-write from adding zero when the mirror points outside EXEC.
    return [sop1(3,20,4), sop1(3,21,128), 0xf4080000 | 32 << 6 | 10, 0xfa000020,
            scalar(0x1e,7,5,135), vop2(0x1a,8,130,0), vop2(0x25,8,7,8),
            vop2(0x25,1,145,0), vop2(0x25,2,148,0), sop1(3,126,255), active,
            vop2(0x25,1,250,2), 0xff000001 | control, sop1(3,126,193),
            0xe0701000, 0x80080108, 0xbf810000]


def pixel_program(start, half, control):
    # Explicitly populate a guest register array even when the rasterizer
    # schedules fewer than eight pixels. Read back four individual guest lanes.
    program = [sop1(0x25,62,193), vmove(17,128)]
    for lane in range(16):
        program += [0xd7610011, (129+lane) | ((128+start+lane) << 9)]
    program += [vmove(17,250), 0xff040011 | control << 8]
    for col in range(4):
        program += [0xd7600000 | col, 273 | ((128+start+4*half+col) << 9)]
    program += [sop1(4,126,62)]
    program += [0x7e000c00 | col << 17 | col for col in range(4)]
    return patched_pixel(len(program), program, export=(0xf800180f,0x03020100))


def reduction_program():
    # Seed two guest wave64 halves differently, including unsigned values
    # above INT_MAX. Inspect both the intermediate 32-lane extrema and
    # the 64-lane extrema after the readlane/broadcast step.
    program = [sop1(0x25,62,193), vmove(17,144)]
    for lane, value in ((12,1),(19,27),(33,3),(60,39)):
        program += [0xd7610011, (128+value) | ((128+lane) << 9)]
    program += [0xd7610011, 193 | (155 << 9)]  # lane 27 holds UINT_MAX
    for output, op in enumerate((0x13,0x14)):
        program += [vmove(18,273)]
        for control in (0xb1,0x4e,0x141,0x140):
            program += [vop2(op,18,250,18), 0xff040012 | (control << 8)]
        program += [0xd7780013,274 | 128 << 9 | 128 << 18, vop2(op,18,274,19),
                    0xd7600000 | output,274 | 191 << 9,
                    0xd7600004,274 | 128 << 9,vop2(op,18,4,18),
                    0xd7600002 | output,274 | 191 << 9]
    # Keep the low byte of each result so the uint max is observable as 255.
    program += [sop1(4,126,62)]
    for col in range(4):
        program += [vmove(col,col),vop2(0x1b,col,255,col),255,0x7e000c00 | col << 17 | (256+col)]
    return patched_pixel(len(program),program,export=(0xf800180f,0x03020100))


def vertex_program(last_id, branchless=False):
    program = [vmove(1,243),vmove(2,243),vmove(3,128),vmove(4,242)]
    # Start at (-1,-1). Separate masked arms move vertices 1 and 2 to
    # (3,-1) and (-1,3). Each contains a scalar branch over the literal move.
    for reg, vertex_id in ((1,1),(2,last_id)):
        if branchless:
            program += [sop1(4,8,126),0xbf8cc07f,0x7d240080 | vertex_id,
                        vmove(reg,255),0x40400000,0x8afe7e08,vmove(reg,243),sop1(4,126,8)]
        else:
            program += [0x7d040080 | vertex_id, 0xbe88246a, 0xbf880004,
                        0xbf008080,0xbf840002,vmove(reg,255),0x40400000,0xbefe0408]
    return vertex(program=program+[0xf80008cf,0x04030201,0xbf810000])


def main(cli, replay, raster, sdk):
    reverse = (7,6,5,4,3,2,1,0)
    sentinel = 0x69696969
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shader = work/"fixture.hlsl"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"\n'
                          'float4 vs(uint id:SV_VertexID):SV_Position {'
                          'return float4(id==1?3:-1,id==2?3:-1,0,1);}')
        dxc = Path(sdk)/"dxc.exe"
        root, vs = work/"root", work/"vs.cso"
        subprocess.run([dxc,"-T","rootsig_1_1","-E","SIG","-Fo",root,shader],check=True)
        subprocess.run([dxc,"-T","vs_6_6","-E","vs","-Fo",vs,shader],check=True)
        sa = struct.pack("<7I",0,0,0,32 | 1 << 16,1,0,0x8a)
        sparse = 0x955ad3b6
        for mask, active, fi, zero in ((7,0xffffffff,1,0), (15,0xffffffff,1,0),
                                        (7,sparse,1,0), (7,sparse,0,0), (7,sparse,0,1)):
            ctrl = ((0x141 if mask == 7 else 0x140) << 8) | fi << 18 | zero << 19
            marker = write_request(work,compute_program(ctrl, active),root.read_bytes(),sa)
            dest = work/"out.bin"
            result = subprocess.run([replay,sdk,marker,work/"cache","0",dest,"--direct"])
            if result.returncode == 77:
                return 77
            result.check_returncode()
            expected = [sentinel]*2064
            for lane in range(32):
                peer = lane // (mask+1) * (mask+1) + mask - lane % (mask+1)
                own, value = lane+20, lane+17
                if active & (1 << lane):
                    if fi or active & (1 << peer): value = own + peer + 17
                    elif zero: value = own
                for group in range(2):
                    expected[8+group*32+lane] = value
            actual = struct.unpack("<2064I",dest.read_bytes())
            assert list(actual) == expected, (mask, fi, zero, actual[8:40], expected[8:40])
            print(f"CS xor={mask} EXEC={active:08x} FI={fi} BC={zero}: 64 results and guards matched")
        for start, half, ctrl in ((0,0,0x141),(0,1,0x141),(32,0,0x141),(32,1,0x141),(0,0,0x140)):
            src, ps = work/"pixel.fssm", work/"pixel.cso"
            src.write_bytes(pixel_program(start,half,ctrl))
            subprocess.run([cli,sdk,work/"cache","gfx1013-container-ps",src,ps],check=True)
            expected = [1 + x for x in reverse[4*half:4*half+4]]
            reference = work/"pixels.bin"
            reference.write_bytes(struct.pack("<4f",*expected)*8)
            result = subprocess.run([raster,"hardware","--startup-pixel-words",vs,ps,"4","2",reference],
                                    capture_output=True,text=True)
            if result.returncode == 77:
                return 77
            if ctrl == 0x140:
                assert result.returncode == 1 and "8 of 8 pixels differ" in result.stderr, result.stderr
            elif result.returncode:
                raise AssertionError(result.stderr)
            print(f"PS guest lanes {start+4*half}..{start+4*half+3}: "
                  + (f"matched {expected}" if ctrl == 0x141 else "wrong full-row control differed at all 8 pixels"))
        src.write_bytes(reduction_program())
        subprocess.run([cli,sdk,work/"cache","gfx1013-container-ps",src,ps],check=True)
        reference.write_bytes(struct.pack("<4f",3,39,1,255)*8)
        subprocess.run([raster,"hardware","--startup-pixel-words",vs,ps,"4","2",reference],check=True)
        print("PS unsigned min/max, two 32-lane halves and final wave64 reduction matched [3,39,1,255]")
        for branchless,last_id in ((False,2),(False,1),(True,2),(True,1)):
            src.write_bytes(vertex_program(last_id,branchless))
            subprocess.run([cli,sdk,work/"cache","gfx1013-container-vs",src,vs],check=True)
            result = subprocess.run([raster,"hardware","--startup-pixel-words",vs,ps,"4","2",reference],
                                    capture_output=True,text=True)
            if last_id == 2: result.check_returncode()
            else: assert result.returncode == 1 and "8 of 8 pixels differ" in result.stderr, result.stderr
        print("Translated masked VS and reduction PS matched 8 pixels; collapsed triangle lost all 8")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
