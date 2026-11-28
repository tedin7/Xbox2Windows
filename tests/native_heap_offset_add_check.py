"""A slot 67 request that forms its resource-heap descriptor address by adding a byte offset (issue #3224).

usage: native_heap_offset_add_check.py COMPILE_DXIL SDK

A waterfall loop over a lane-selected heap slot adds the slot's byte offset to the heap base and carries into the high
word instead of passing the offset as the SGPR offset of the load:

  s_add_u32 s10, s5, s8 ; s_add_u32 s11, 0, src_scc ; s_load_dwordx4 s[12:15], s[10:11], 0x0

The requests are written from source with the entry layout of the start-up shaders that do this: user SGPRs s4 (a
descriptor table), s5 (the resource heap) and s6 (the sampler heap) under both directly indexed heap flags, wave32. Such a
load is the same GPU-selected heap read as the SGPR-offset form, so both now stop at the heap view contract, which a
.c67 request has no metadata for. Before this change the add was refused as a numeric use of the heap table. The
controls keep their refusals: a high word that is not the carry of the add, SCC clobbered between the adds, and a
constant offset. The GPU readback of the same address form is in native_global_srv_heap_replay_check.py (section 11).
Encodings were checked with tools/disasm_rdna2.py; the test needs no LLVM.
"""
import pathlib
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request

ROOT = "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED), DescriptorTable(SRV(t0), UAV(u0))"
HEAP_CONTRACT = "direct resource-heap load requires the selected GPU slot and its view type"
HIGH_WORD = "heap descriptor address high word is not the carry of its offset add"
SELECT = [0x7E100500, 0x8F088508]  # v_readfirstlane_b32 s8, v0 ; s_lshl_b32 s8, s8, 5
ADD_LO, ADD_HI = 0x800A0805, 0x800BFD80  # s_add_u32 s10, s5, s8 ; s_add_u32 s11, 0, src_scc
LOAD = [0xF4080305, 0xFA000000]  # s_load_dwordx4 s[12:15], s[10:11], 0x0
END = [0xBF8CC07F, 0xBF810000]  # s_waitcnt lgkmcnt(0) ; s_endpgm
PROGRAMS = {
    "added": (SELECT + [ADD_LO, ADD_HI] + LOAD + END, HEAP_CONTRACT),
    # s_mov_b32 s10, s5 ; s_mov_b32 s11, 0 ; s_load_dwordx4 s[12:15], s[10:11], s8: the SGPR-offset form
    "sgpr_offset": (SELECT + [0xBE8A0305, 0xBE8B0380, 0xF4080305, 0x10000000] + END, HEAP_CONTRACT),
    # s_mov_b32 s11, 0 instead of the carry
    "mov_high": (SELECT + [ADD_LO, 0xBE8B0380] + LOAD + END, HIGH_WORD),
    # s_cmp_lg_u32 s8, s8 between the adds
    "scc_clobbered": (SELECT + [ADD_LO, 0xBF070808, ADD_HI] + LOAD + END, HIGH_WORD),
    # s_add_u32 s10, s5, 0x40: a constant offset is not a GPU selection; the heap table is used as a number
    "constant": (SELECT + [0x800AFF05, 0x40, ADD_HI] + LOAD + END, "unbound numeric SGPR s5"),
    # Wave64, s_load_dwordx4 s[104:107], s[10:11], 0x0: the slot index would be written into VCC, which the wave64
    # model keeps as mask data, so the overlap refusal comes before the heap contract.
    "wave64_vcc_dst": (SELECT + [ADD_LO, ADD_HI, 0xF4081A05, 0xFA000000] + END,
                       "symbolic scalar memory destination overlaps wave64 VCC", 64),
}


def main(compile_dxil, sdk):
    failures = 0
    with tempfile.TemporaryDirectory(prefix="x2w_heap_offset_add_") as scratch:
        base = pathlib.Path(scratch)
        source = base / "root.hlsl"
        source.write_text(f'#define RS "{ROOT}"')
        binary = base / "root.bin"
        subprocess.run([pathlib.Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "RS", "-Fo", binary, source],
                       check=True, capture_output=True)
        root = binary.read_bytes()
        for name, (program, expected, *wave) in PROGRAMS.items():
            # Group 32x1x1 in wave32 (bit 20 of dword 4), or 64x1x1 in wave64, 7 user SGPRs.
            threads = wave[0] if wave else 32
            sa = struct.pack("<7I", 0, 0, 0, threads | (1 << 16), 1 | (0 if wave else 1 << 20), 0, 7 << 1)
            folder = base / name
            folder.mkdir()
            marker = write_request(folder, program, root, sa)
            output = marker.with_suffix(".cso")
            result = subprocess.run([compile_dxil, sdk, base / "cache", "gfx1013-direct-cs@0", marker, output],
                                    capture_output=True, text=True, timeout=60)
            ok = result.returncode != 0 and expected in result.stderr and not output.exists()
            print(("ok: " if ok else "FAIL: ") + f"{name}: refused with '{expected}' and no .cso")
            if not ok:
                print(result.stdout + result.stderr)
                failures += 1
    if failures:
        return 1
    print("native_heap_offset_add: all checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(*sys.argv[1:]))
