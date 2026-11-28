"""Samplers read from the directly indexed sampler heap at a fixed index, translated natively (issue #3076).

usage: native_sampler_heap_check.py FIXTURE COMPILE_DXIL SDK ADAPTER

Compute requests are written from source: user SGPRs s4 (the SRV+UAV descriptor table), s5 (the resource heap) and s6 (the
sampler heap) under RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED). The programs copy s6
into an even SGPR pair with a constant-zero high word, load one sampler (s_load_dwordx4) or two (s_load_dwordx8) at a
16-byte aligned immediate offset (0x430 is entry 67) and sample a two-mip texture at a coordinate outside [0, 1]:

  SAMPLE  image_sample_lz with entry 67           TWO    entries 67 and 68 from one s_load_dwordx8
  LOD     image_sample_l at LOD 1, MaxLOD decides MIXED  one literal static sampler and entry 67

The `native_sampler_heap` fixture runs them with a sampler heap of decoys, compares every lane with a CPU model, puts each
decoy in place of the entry under test (the result must change), checks the frontend cache round trip and the binding
record, and shows that bind() without a sampler heap throws. This script also compiles the refusal shapes and checks each
refusal text (and that no .cso is written), each with a control that compiles. Exit 77 when the fixture reports a
capability skip (WARP has no 32-lane waves).

Instruction words were assembled once with LLVM 15 (clang -cc1as -triple amdgcn-amd-amdhsa -target-cpu gfx1013
-filetype asm -show-encoding); the comment on each line is the assembler source. The test needs no LLVM.
"""
import pathlib
import struct
import subprocess
import sys
import tempfile

# Same helper as native_compute_raw_root_check.py: writes req_<key>.c67 and its parts.
from native_compute_clear_check import write_request

ROOT_RW = "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED), DescriptorTable(SRV(t0), UAV(u0))"
ROOT_VIEWS_ONLY = "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), DescriptorTable(SRV(t0), UAV(u0))"

REFUSALS = {
    "buffer": "scalar buffer load from the sampler heap is not translated",
    "sgpr": "sampler heap load with an SGPR offset (a GPU-selected sampler) is not translated",
    "shape": "sampler heap load must read one or two samplers",
    "align": "sampler heap offset is not a multiple of 16",
    "bound": "sampler heap index beyond 2046 (2047 is the runtime fallback entry)",
    "hlsl": "sampler heap reads need the direct DXIL path",
    "resource": "a sampler heap descriptor is read as a resource",
    "gather": "gather through the sampler heap is not translated",
    "compare": "comparison sampler from the sampler heap is not translated",
    "implicit": "sampler from the sampler heap needs image_sample_l or image_sample_lz",
    "mixed": "nonconstant sampler",
    "table": "direct descriptor address outside bound table",
}

# --- instruction blocks -------------------------------------------------------------------------------------------
BASES = [
    0xbe880304,  # s_mov_b32 s8, s4      (s[8:9]: the table pointer, high word constant 0)
    0xbe890380,  # s_mov_b32 s9, 0
    0xbe8a0306,  # s_mov_b32 s10, s6     (s[10:11]: the sampler heap base, high word constant 0)
    0xbe8b0380,  # s_mov_b32 s11, 0
]
TABLE_LOADS = [
    0xf40c0404, 0xfa000000,  # s_load_dwordx8 s[16:23], s[8:9], 0x0     (t0: the texture)
    0xf4080604, 0xfa000020,  # s_load_dwordx4 s[24:27], s[8:9], 0x20    (u0: the output buffer)
]
SAMPLER_X4 = [
    0xf4080305, 0xfa000430,  # s_load_dwordx4 s[12:15], s[10:11], 0x430   (entry 67)
]
COORDINATES = [
    0x7e0202ff, 0x3fb00000,  # v_mov_b32_e32 v1, 0x3fb00000   (u = 1.375)
    0x7e0402ff, 0xbec00000,  # v_mov_b32_e32 v2, 0xbec00000   (v = -0.375)
]
LOD_ONE = [
    0x7e0602f2,  # v_mov_b32_e32 v3, 1.0     (LOD 1)
]
WAIT_SCALAR = [
    0xbf8cc07f,  # s_waitcnt lgkmcnt(0)
]
WAIT_VECTOR = [
    0xbf8c3f70,  # s_waitcnt vmcnt(0)
]
END = [
    0xbf810000,  # s_endpgm
]
LITERAL_STATIC = [
    0xbea803ff, 0x00000092,  # s_mov_b32 s40, 0x92   (static sampler: clamp x, y and z, point)
    0xbea90380,  # s_mov_b32 s41, 0
    0xbeaa0380,  # s_mov_b32 s42, 0
    0xbeab0380,  # s_mov_b32 s43, 0
]
STORE_ONE = [
    0x34100084,  # v_lshlrev_b32_e32 v8, 4, v0
    0xe0781000, 0x80060408,  # buffer_store_dwordx4 v[4:7], v8, s[24:27], 0 offen
]
STORE_TWO = [
    0x34100085,  # v_lshlrev_b32_e32 v8, 5, v0
    0xe0781000, 0x80060408,  # buffer_store_dwordx4 v[4:7], v8, s[24:27], 0 offen
    0xe0781010, 0x80060c08,  # buffer_store_dwordx4 v[12:15], v8, s[24:27], 0 offen offset:16
]
SAMPLE_LZ_HEAP = [
    0xf09c0f08, 0x00640401,  # image_sample_lz v[4:7], v[1:2], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D
]
# Operands for the refusal and control shapes: four coordinates after u and v, a sampler built from literals.
FILL_V3_V6 = [
    0x7e0602f2,  # v_mov_b32_e32 v3, 1.0
    0x7e0802f2,  # v_mov_b32_e32 v4, 1.0
    0x7e0a02f2,  # v_mov_b32_e32 v5, 1.0
    0x7e0c02f2,  # v_mov_b32_e32 v6, 1.0
]
SAMPLER_LITERAL = [
    0xbe8c03ff, 0x08001012,  # s_mov_b32 s12, 0x8001012   (comparison_program of native_direct_container_check.py)
    0xbe8d03ff, 0x00fff000,  # s_mov_b32 s13, 0xfff000
    0xbe8e03ff, 0x04000000,  # s_mov_b32 s14, 0x4000000
    0xbe8f0380,  # s_mov_b32 s15, 0
]


def operation(op_words):
    return [*BASES, *TABLE_LOADS, "SAMPLER", *COORDINATES, *FILL_V3_V6, *WAIT_SCALAR, *op_words, *END]


OPERATIONS = {
    "gather4_lz": [0xf11c0108, 0x00640801],  # image_gather4_lz v[8:11], v[1:2], s[16:23], s[12:15] dmask:0x1 dim:SQ_RSRC_IMG_2D
    "gather4_c_lz": [0xf13c0108, 0x00640801],  # image_gather4_c_lz v[8:11], v[1:3], s[16:23], s[12:15] dmask:0x1 dim:SQ_RSRC_IMG_2D
    "sample_c_lz": [0xf0bc0108, 0x00640801],  # image_sample_c_lz v8, v[1:3], s[16:23], s[12:15] dmask:0x1 dim:SQ_RSRC_IMG_2D
    "sample_d": [0xf0880f08, 0x00640801],  # image_sample_d v[8:11], v[1:6], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D
}


def with_sampler(program, sampler):
    out = []
    for word in program:
        out.extend(sampler if word == "SAMPLER" else [word])
    return out


PROGRAMS = {
    "sample": [*BASES, *TABLE_LOADS, *SAMPLER_X4, *COORDINATES, *WAIT_SCALAR, *SAMPLE_LZ_HEAP, *WAIT_VECTOR, *STORE_ONE, *END],
    "two": [
        *BASES, *TABLE_LOADS,
        0xf40c0805, 0xfa000430,  # s_load_dwordx8 s[32:39], s[10:11], 0x430   (entries 67 and 68)
        *COORDINATES, *WAIT_SCALAR,
        0xf09c0f08, 0x01040401,  # image_sample_lz v[4:7], v[1:2], s[16:23], s[32:35] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xf09c0f08, 0x01240c01,  # image_sample_lz v[12:15], v[1:2], s[16:23], s[36:39] dmask:0xf dim:SQ_RSRC_IMG_2D
        *WAIT_VECTOR, *STORE_TWO, *END],
    "lod": [
        *BASES, *TABLE_LOADS, *SAMPLER_X4, *COORDINATES, *LOD_ONE, *WAIT_SCALAR,
        0xf0900f08, 0x00640401,  # image_sample_l v[4:7], v[1:3], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D
        *WAIT_VECTOR, *STORE_ONE, *END],
    "mixed": [
        *BASES, *TABLE_LOADS, *SAMPLER_X4, *LITERAL_STATIC, *COORDINATES, *WAIT_SCALAR,
        0xf09c0f08, 0x01440401,  # image_sample_lz v[4:7], v[1:2], s[16:23], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xf09c0f08, 0x00640c01,  # image_sample_lz v[12:15], v[1:2], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D
        *WAIT_VECTOR, *STORE_TWO, *END],
    # Refusal shapes. Each one stops at the instruction under test.
    "dx2": [*BASES,
            0xf4040305, 0xfa000430,  # s_load_dwordx2 s[12:13], s[10:11], 0x430
            *END],
    "dx16": [*BASES,
             0xf4100405, 0xfa000400,  # s_load_dwordx16 s[16:31], s[10:11], 0x400
             *END],
    "off438": [*BASES,
               0xf4080305, 0xfa000438,  # s_load_dwordx4 s[12:15], s[10:11], 0x438
               *END],
    "idx2047": [*BASES,
                0xf4080305, 0xfa007ff0,  # s_load_dwordx4 s[12:15], s[10:11], 0x7ff0
                *END],
    "bound2": [*BASES,
               0xf40c0405, 0xfa007fe0,  # s_load_dwordx8 s[16:23], s[10:11], 0x7fe0   (entries 2046 and 2047)
               *END],
    "sgpr_offset": [*BASES,
                    0xbe9403ff, 0x00000430,  # s_mov_b32 s20, 0x430
                    0xf4080305, 0x28000000,  # s_load_dwordx4 s[12:15], s[10:11], s20
                    *END],
    "bufload": [
        0xbe880306,  # s_mov_b32 s8, s6
        0xbe890380,  # s_mov_b32 s9, 0
        0xf4200304, 0xfa000000,  # s_buffer_load_dword s12, s[8:11], 0x0
        *END],
    "asres": [
        *BASES,
        0xf40c0405, 0xfa000400,  # s_load_dwordx8 s[16:23], s[10:11], 0x400   (a sampler heap entry as the image descriptor)
        *LITERAL_STATIC, *COORDINATES, *WAIT_SCALAR,
        0xf09c0f08, 0x01440401,  # image_sample_lz v[4:7], v[1:2], s[16:23], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D
        *END],
    "mixed_words": [
        *BASES, *TABLE_LOADS, *SAMPLER_X4,
        0xbe8c03ff, 0x00000092,  # s_mov_b32 s12, 0x92   (word 0 of the sampler becomes a literal; words 1 to 3 stay heap words)
        *COORDINATES, *WAIT_SCALAR, *SAMPLE_LZ_HEAP, *END],
    "join": [
        *BASES,
        0xbe940381,  # s_mov_b32 s20, 1
        0xbf068014,  # s_cmp_eq_u32 s20, 0
        0xbf850001,  # s_cbranch_scc1 1      (skips the next instruction)
        0xbe8a0380,  # s_mov_b32 s10, 0      (one path replaces the copied heap base)
        0xf4080305, 0xfa000430,  # s_load_dwordx4 s[12:15], s[10:11], 0x430
        *END],
    # Controls: the same shapes with the one change that makes them valid.
    "join_control": [
        *BASES,
        0xbe940381,  # s_mov_b32 s20, 1
        0xbf068014,  # s_cmp_eq_u32 s20, 0
        0xbf850001,  # s_cbranch_scc1 1
        0xbe950380,  # s_mov_b32 s21, 0      (the skipped instruction leaves the heap base alone)
        0xf4080305, 0xfa000430,  # s_load_dwordx4 s[12:15], s[10:11], 0x430
        *END],
    "edge2046": [*BASES,
                 0xf4080305, 0xfa007fe0,  # s_load_dwordx4 s[12:15], s[10:11], 0x7fe0   (entry 2046, the last one)
                 *END],
    "x8_edge": [*BASES,
                0xf40c0405, 0xfa007fd0,  # s_load_dwordx8 s[16:23], s[10:11], 0x7fd0   (entries 2045 and 2046)
                *END],
}
for op_name, op_words in OPERATIONS.items():
    PROGRAMS["heap_" + op_name] = with_sampler(operation(op_words), SAMPLER_X4)
    PROGRAMS["literal_" + op_name] = with_sampler(operation(op_words), SAMPLER_LITERAL)


def main():
    fixture, compile_dxil, sdk, adapter = sys.argv[1:5]
    ok_lines = 0

    def require(condition, message):
        nonlocal ok_lines
        if not condition:
            print("FAIL: " + message)
            sys.exit(1)
        ok_lines += 1
        print("ok: " + message)

    def run(command, expect_skip=False):
        result = subprocess.run([str(part) for part in command], capture_output=True, timeout=150)
        out = result.stdout.decode("utf-8", "replace")
        err = result.stderr.decode("utf-8", "replace")
        print("$ " + " ".join(pathlib.Path(str(c)).name if i == 0 else str(c) for i, c in enumerate(command)))
        for text in (out, err):
            for line in text.strip().splitlines():
                print("    " + line)
        print(f"    -> exit {result.returncode}")
        if result.returncode == 77 and expect_skip:
            print("fixture capability skip (77)")
            sys.exit(77)
        return result.returncode, out, err

    with tempfile.TemporaryDirectory(prefix="x2w_sampler_heap_") as scratch:
        base = pathlib.Path(scratch)
        # Sampler heap content, texture and expectations are in the fixture; the requests are written here.
        sa = struct.pack("<7I", 0, 0, 0, 32 | (1 << 16), 1, 0, 7 << 1)  # group 32x1x1, 7 user SGPRs
        roots = {}

        def root(text):
            if text not in roots:
                source = base / f"root{len(roots)}.hlsl"
                source.write_text(f'#define RS "{text}"')
                binary = base / f"root{len(roots)}.bin"
                subprocess.run([pathlib.Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "RS", "-Fo", binary, source],
                               check=True, capture_output=True)
                roots[text] = binary.read_bytes()
            return roots[text]

        counter = 0

        def request(name, root_text=ROOT_RW):
            nonlocal counter
            counter += 1
            folder = base / f"req_{counter}_{name}"
            folder.mkdir()
            return write_request(folder, PROGRAMS[name], root(root_text), sa)

        def compile_request(marker, profile="gfx1013-direct-cs@0", cache=None):
            output = marker.with_suffix(".cso")
            code, out, err = run([compile_dxil, sdk, cache or marker.parent / "cache", profile, marker, output])
            return code, out, err, output

        def refused(name, key, root_text=ROOT_RW, profile="gfx1013-direct-cs@0"):
            marker = request(name, root_text)
            code, out, err, output = compile_request(marker, profile)
            require(code != 0 and REFUSALS[key] in err and not output.exists(),
                    f"{name}: refused with \"{REFUSALS[key]}\" and no .cso")

        def accepted(name, message):
            marker = request(name)
            code, out, err, output = compile_request(marker)
            require(code == 0 and output.exists() and "sampler_heap span" not in out, message)
            return out

        # 1. Fixture runs. A fresh frontend cache per run: the fixture asserts a miss and then a hit.
        expected_lines = {
            "sample": ["entry 67 (result 0) equals the CPU model in all 32 lanes",
                       *[f"decoy {n} in place of entry 67 changes result 0" for n in range(4)],
                       "{compute_sampler_heap, 0, 0, 1088 bytes}"],
            "two": ["entry 67 (result 0) equals the CPU model", "entry 68 (result 1) equals the CPU model",
                    *[f"decoy {n} in place of entry {e} changes result {e - 67}" for e in (67, 68) for n in range(3)],
                    "{compute_sampler_heap, 0, 0, 1104 bytes}"],
            "lod": ["entry 67 (result 0) equals the CPU model",
                    *[f"decoy {n} in place of entry 67 changes result 0" for n in range(3)],
                    "{compute_sampler_heap, 0, 0, 1088 bytes}"],
            "mixed": ["the literal static sampler result matches the CPU model", "entry 67 (result 1) equals the CPU model",
                      *[f"decoy {n} in place of entry 67 changes result 1" for n in range(3)],
                      "1088 bytes}"],
        }
        for mode, lines in expected_lines.items():
            marker = request(mode)
            cache = base / f"cache_{mode}"
            cache.mkdir()
            code, out, err = run([fixture, sdk, cache, marker, adapter, mode], expect_skip=True)
            require(code == 0, f"fixture mode {mode} exits 0")
            for line in lines:
                require(line in out, f"{mode}: {line}")
            require("pass: prepare_direct_compute misses on a fresh cache and hits on the second call" in out,
                    f"{mode}: frontend cache round trip (miss, then hit)")
            require("pass: pipeline has sampler_heap_indexed" in out, f"{mode}: pipeline asks for the sampler heap flag only")
        code, out, err = run([fixture, sdk, base / "cache_unbound", request("sample"), adapter, "unbound"], expect_skip=True)
        require(code == 0 and "bind without a sampler heap throws" in out, "bind without a sampler heap throws")

        # 2. compile_dxil: the module reads the sampler from the heap and the binding record is reported.
        marker = request("sample")
        code, out, err, output = compile_request(marker)
        require(code == 0 and output.exists(), "compile_dxil compiles SAMPLE")
        require("sampler_heap span=1088" in out, "compile_dxil reports sampler_heap span=1088 for entry 67")
        code2, out2, err2, _ = compile_request(marker)
        require(code2 == 0 and "frontend_cache_hit=1" in out2, "the second compile hits the frontend cache")
        listing = subprocess.run([pathlib.Path(sdk) / "dxc.exe", "-dumpbin", output], capture_output=True, text=True).stdout
        require("@dx.op.createHandleFromHeap(i32 218, i32 67, i1 true, i1 false)" in listing,
                "the module creates the sampler with createHandleFromHeap(218, index 67, sampler heap, uniform)")
        require(listing.count("@dx.op.createHandleFromBinding(") >= 2,
                "the texture and the output buffer stay table bindings (createHandleFromBinding)")
        marker = request("two")
        code, out, err, output = compile_request(marker)
        listing = subprocess.run([pathlib.Path(sdk) / "dxc.exe", "-dumpbin", output], capture_output=True, text=True).stdout
        require(code == 0 and "sampler_heap span=1104" in out and
                "createHandleFromHeap(i32 218, i32 67, i1 true, i1 false)" in listing and
                "createHandleFromHeap(i32 218, i32 68, i1 true, i1 false)" in listing,
                "TWO: one s_load_dwordx8 gives sampler heap handles 67 and 68")
        marker = request("sample")
        requests = marker.parent
        out_dir = base / "requests_out"
        code, out, err = run([compile_dxil, sdk, base / "cache_requests", "gfx1013-direct-cs@0", "--requests", requests, out_dir])
        stem = marker.stem[len("req_"):]
        require(code == 0 and (out_dir / (stem + ".cs.cso")).exists() and not (out_dir / (stem + ".txt")).exists(),
                "--requests publishes .cs.cso and no runtime manifest (.txt) for a sampler heap program")

        # 3. Refusals, each with its text, and controls that compile.
        refused("off438", "align")
        refused("idx2047", "bound")
        refused("bound2", "bound")
        refused("dx2", "shape")
        refused("dx16", "shape")
        refused("sgpr_offset", "sgpr")
        refused("bufload", "buffer")
        refused("asres", "resource")
        refused("mixed_words", "mixed")
        refused("join", "table")
        refused("heap_gather4_lz", "gather")
        refused("heap_gather4_c_lz", "gather")
        refused("heap_sample_c_lz", "compare")
        refused("heap_sample_d", "implicit")
        accepted("edge2046", "control: entry 2046 (offset 0x7fe0) compiles")
        accepted("x8_edge", "control: s_load_dwordx8 at entries 2045 and 2046 compiles")
        accepted("join_control", "control: the same branch without the overwrite compiles")
        for op_name in OPERATIONS:
            accepted("literal_" + op_name, f"control: {op_name} with a literal sampler compiles")
        # The HLSL path has no sampler heap handle; a root with only the views flag has no sampler heap base.
        refused("sample", "hlsl", profile="gfx1013-compute-wave32@0")
        refused("sample", "table", root_text=ROOT_VIEWS_ONLY)

    print(f"{ok_lines} assertions held")


if __name__ == "__main__":
    main()
