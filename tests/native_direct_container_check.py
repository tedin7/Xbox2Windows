"""Source-built containers exercise direct DXIL assembly and rejection paths."""
import pathlib
import hashlib
import re
import struct
import subprocess
import sys
import tempfile
import zlib


def words(*values):
    return struct.pack(f"<{len(values)}I", *values)


def compute(code, threads=32, table=False, root_descriptor=False, wave32=True, uses=()):
    body = bytearray(256)
    struct.pack_into("<Q", body, 0, 6)
    for offset in (8, 16, 24, 32, 40, 48, 72):
        struct.pack_into("<Q", body, offset, 0xDEADBEEFDEADBEEF)
    record = 112
    struct.pack_into("<Q", body, 96, record)
    if table or root_descriptor:
        struct.pack_into("<Q",body, record+40, len(body))
        struct.pack_into("<I",body, record+56, 1)
        body.extend(words(1,0,2,0,0) if root_descriptor else words(3,0,1,0,0))
    if uses:
        struct.pack_into("<Q", body, record+48, len(body))
        struct.pack_into("<I", body, record+60, len(uses))
        for use in uses:
            body.extend(words(*use))
    struct.pack_into("<QQ", body, record, len(body), len(code))
    struct.pack_into("<I", body, record+20, 32 if wave32 else 0)
    struct.pack_into("<II", body, record+76, 512, 4)
    user_flags = 4 if root_descriptor else (0x1002 if table else 0)
    struct.pack_into("<IIII", body, record+120, 0x8400 | user_flags, threads, 1, 1)
    body.extend(code)
    return pack(body)


def pack(body, extra=()):
    raw = b"FSSM" + struct.pack("<IQ", 0x101, len(body)) + body
    compressed = zlib.compress(raw)
    blob = b"XBCH" + struct.pack("<IQQ", 3, len(compressed), len(raw)) + compressed
    part = bytes(24) + b"XDAT" + words(1, 0, 13, 48, len(blob)) + blob
    chunks = [(b"DXIL", part), *extra]
    table = []
    data = b""
    for name, payload in chunks:
        table.append(32+4*len(chunks)+len(data))
        data += name+words(len(payload))+payload
        data += bytes(-len(data)%4)
    return b"DXBC"+bytes(16)+words(1, 32+4*len(chunks)+len(data), len(chunks), *table)+data

def signature_rows(rows):
    records, names = bytearray(), bytearray()
    for semantic, system, component, row, mask in rows:
        name_at = 8 + 32*len(rows) + len(names)
        records.extend(words(0,name_at,0,system,component,row,mask*257,0))
        names.extend(semantic+b"\0")
    return words(len(rows),8)+records+names


def signature(semantic, system=0, component=3, row=0, mask=15):
    return signature_rows([(semantic,system,component,row,mask)])


def add_root_input(body, record, parameter, count, kind=0):
    struct.pack_into("<Q", body, record+40, len(body))
    struct.pack_into("<I", body, record+56, 1)
    body.extend(words(kind, 0, count, 0, parameter << 8))


def vertex(fetch=False, constants=None, program=None, extra=(), varying=False, table=False, uses=()):
    record=112
    body=bytearray(record+84+0x18c)
    struct.pack_into("<Q",body,0,6)
    for index in (8,16,24,32,48,72,96):
        struct.pack_into("<Q",body,index,0xdeadbeefdeadbeef)
    struct.pack_into("<Q",body,40,record)
    # Convert incoming vertex ID before a full position export.
    code=words(0x7e020b00,0xf80008cf,0x01010101,0xbf810000)
    if fetch:
        io = record + 84
        struct.pack_into("<I", body, io+0xc4, 1)
        body[io+0x24:io+0x29] = bytes((6, 0, 15, 1, 3))
        code = words(0xbe842104, 0xf80008cf, 0x04030201, 0xbf810000)
    if constants:
        add_root_input(body, record, *constants)
        code = words(0x7e000200, 0xf80008cf, 0, 0xbf810000)
    if table:
        add_root_input(body, record, 0, 1, kind=3)
    if uses:
        struct.pack_into("<Q", body, record+48, len(body))
        struct.pack_into("<I", body, record+60, len(uses))
        for use in uses:
            body.extend(words(*use))
    if program is not None:
        code = words(*program)
    if varying:
        io=record+84
        struct.pack_into("<I", body,io+0x188,1)
        body[io+0xc8:io+0xce]=bytes((6,0,0,0,0,0))
    struct.pack_into("<QQ",body,record,len(body),len(code))
    body.extend(code)
    return pack(body, (((b"ISG1", signature(b"POSITION")),) if fetch else ()) + extra)

def patched_pixel(where=1, prefix=None, table=False, input_address=0, input_signature=(), constants=None,
                  export=(0xf800180f, 0), root_inputs=(), wave32=False, tail=(), attributes=()):
    record = 112
    io = record + 84
    body = bytearray(io + 0x14c + 8 * 404)
    struct.pack_into("<Q", body, 0, 6)
    for offset in (8, 16, 24, 32, 40, 72, 96):
        struct.pack_into("<Q", body, offset, 0xDEADBEEFDEADBEEF)
    struct.pack_into("<Q", body, 48, record)
    struct.pack_into("<I", body, record+20, 32 if wave32 else 0)
    struct.pack_into("<I", body, io+0x134, input_address)
    struct.pack_into("<I", body, io+0x124, len(attributes))
    for index, attribute in enumerate(attributes):
        struct.pack_into("<8B", body, io+0x24+index*8, *attribute)
    if table:
        struct.pack_into("<Q", body, record+40, len(body))
        struct.pack_into("<I", body, record+56, 1)
        body.extend(words(3,0,1,0,0))
    if prefix is None:
        prefix = [0x7e0002f2]
    if constants:
        add_root_input(body, record, *constants)
    if root_inputs:
        struct.pack_into("<Q", body, record+40, len(body))
        struct.pack_into("<I", body, record+56, len(root_inputs))
        for record_words in root_inputs:
            body.extend(words(*record_words))
    code = words(*prefix, 0xffffffff, 0xffffffff, 0xbf810000, *tail)
    struct.pack_into("<QQ", body, record, len(body), len(code))
    struct.pack_into("<I", body, io+0x148, 2)
    for mrt in range(8):
        struct.pack_into("<I", body, io+0x14c+404*mrt, where if mrt == 0 else 0xffffffff)
    struct.pack_into("<II", body, io+0x14c+364, *export)
    body.extend(code)
    return pack(body, input_signature)

def interpolated_pixel(mismatched=False):
    record=112
    io=record+84
    body=bytearray(io+0x14c+8*404)
    struct.pack_into("<Q",body,0,6)
    for offset in (8,16,24,32,40,72,96):
        struct.pack_into("<Q",body,offset,0xdeadbeefdeadbeef)
    struct.pack_into("<Q",body,48,record)
    struct.pack_into("<I",body,io+0x124,1)
    body[io+0x24:io+0x28]=bytes([6,0,15,0])
    struct.pack_into("<I",body,io+0x134,2)
    for t in range(8):
        struct.pack_into("<I",body,io+0x14c+404*t,0xffffffff)
    p1=0xc8000000 | 4<<18 | 2<<8
    p2=p1 | (1<<16) | 1
    code=words(p1,p2^(0x100 if mismatched else 0),0xf800180f,0x04040404,0xbf810000)
    struct.pack_into("<QQ",body,record,len(body),len(code))
    body.extend(code)
    return pack(body,[(b"ISG1",signature(b"TEXCOORD"))])

def publish(directory, *stages, ps_inputs=None):
    directory.mkdir(exist_ok=True)
    key = 1469598103934665603
    tail = b"" if ps_inputs is None else b"psin1" + struct.pack("<II", *ps_inputs)
    for byte in b"x2w-startup-shader-28" + b"".join(stages) + tail:
        key = ((key ^ byte) * 1099511628211) & ((1 << 64) - 1)
    hex_key = f"{key:016x}"
    stem = f"req_{hex_key}"
    for suffix, data in zip((".a", ".b", ".root", ".ia"), stages):
        (directory / (stem + suffix)).write_bytes(data)
    if ps_inputs is not None:
        (directory / (stem + ".psin")).write_bytes(
            f"psin1 {hex_key} ena=0x{ps_inputs[0]:04x} addr=0x{ps_inputs[1]:04x}\n".encode("ascii")
        )
    return hex_key


def comparison_program(op=47, dim=1, sampler=0x08001012):
    # Clamp X/Y, point with TRUNC_COORD, LESS, nonzero maximum LOD.
    setup = [0xbe810380, 0xf40c0200, 0xfa000000, 0xbe8403ff, sampler,
             0xbe8503ff, 0x00fff000, 0xbe8603ff, 0x04000000, 0xbe870380,
             0x7e0002f0, 0x7e0202f2, 0x7e0402f4, 0x7e060280]
    return [*setup, 0xf0000100 | (dim << 3) | op << 18, 0x00220400]


def main():
    if len(sys.argv) == 4 and sys.argv[1] in ("--sampler-pso", "--graphics-sampler-pso"):
        sdk, probe = sys.argv[2:]
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            # Keep both comparison and regular uses of identical sampler words.
            code = [*comparison_program(), 0xf09c0108, 0x00220500, 0x06000b04]
            if sys.argv[1] == "--graphics-sampler-pso":
                vs, ps, root = (folder/name for name in ("vertex", "pixel", "root.bin"))
                vs_code = [0xbe810380, 0xf40c0200, 0xfa000040, 0x7e020280, 0x7e040280,
                           0xf0200108, 0x80020201, 0x7e020b00, 0xf80008cf, 0x01010101, 0xbf810000]
                vs.write_bytes(vertex(program=vs_code, table=True))
                # Reload the next descriptor in the same table as a writable image.
                code += [0xf40c0200, 0xfa000020, 0xf0200108, 0x80020400]
                ps.write_bytes(patched_pixel(len(code), code, True))
                source = folder/"root.hlsl"
                source.write_text('#define RS "DescriptorTable(SRV(t0), UAV(u0, numDescriptors=2))"\n')
                subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-T", "rootsig_1_1", "-E", "RS", source, "-Fo", root], check=True)
                raise SystemExit(subprocess.run([probe, "--graphics-sampler", sdk, vs, ps, root, folder/"cache"]).returncode)
            marker = folder / "sample.cdx"
            marker.write_bytes(compute(words(*code, 0xbf810000), table=True))
            raise SystemExit(subprocess.run([probe, "--sampler", sdk, marker, folder/"cache"]).returncode)
    if len(sys.argv)==5 and sys.argv[1]=="--rect-pso":
        exe,sdk,probe=sys.argv[2:]
        with tempfile.TemporaryDirectory() as tmp:
            folder=pathlib.Path(tmp)
            source=folder/"inputs"
            out=folder/"compiled"
            vs=vertex(program=[0x7e020b00,0xf80000cf,0x01010101,0xf8000a0f,0x01010101,0xbf810000],
                      varying=True, extra=((b"OSG1",signature_rows([(b"TEXCOORD",0,3,0,15),
                                                                     (b"SV_Position",1,3,1,15)])),))
            root=b"DXBC"+bytes(16)+words(1,68,1,36)+b"RTS0"+words(24,2,0,24,0,24,0)
            stem=publish(source,vs,interpolated_pixel(),root)
            subprocess.run([exe,sdk,folder/"cache","gfx1013-container-rect","--requests",source,out],check=True)
            raise SystemExit(subprocess.run([probe,*(out/f"{stem}.{stage}.cso" for stage in ("vs","gs","ps"))]).returncode)
    exe, sdk = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        def check(name, profile, data, expected=""):
            source = folder / name
            source.write_bytes(data)
            binary = source.with_suffix(".cso")
            result = subprocess.run([exe, sdk, folder/"cache", profile, source, binary],
                                    capture_output=True, text=True)
            if expected:
                assert result.returncode and expected in result.stderr and not binary.exists(), result.stderr
            else:
                assert result.returncode == 0, result.stderr
                assert binary.read_bytes().startswith(b"DXBC")
            return result.stdout
        cs = "gfx1013-container-cs"
        end = 0xbf810000
        # Direct reads of the I/J VGPRs export the y/z weights. Include a
        # preceding (unused) pull-model triple to distinguish VGPR allocation
        # from the two DXIL system-value elements used by the shader.
        for label, mask, first, second in (
                ("center", 2, 0, 0), ("centroid", 4, 0, 0), ("sample", 1, 0, 0),
                ("linear-center", 32, 0, 0), ("linear-centroid", 64, 0, 0),
                ("linear-sample", 16, 0, 0), ("two-pairs", 2 | 8 | 32, 0, 5)):
            reads = [0x7e080300 | first, 0x7e0a0300 | (second + 1)]
            shader = patched_pixel(len(reads), reads, input_address=mask, export=(0xf800180f, 0x05040504))
            name = "bary-" + label
            for timing in ("cold_compile_and_publish_ms=", "cache_hit_ms="):
                assert timing in check(name, "gfx1013-container-ps", shader)
            dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/(name+".cso")],
                                  check=True, capture_output=True, text=True).stdout
            loads = re.findall(r"call float @dx\.op\.loadInput\.f32\(i32 4, i32 (\d+), i32 0, i8 (\d+)", dump)
            assert loads == [("0", "1"), ("1" if first != second else "0", "2")], (name, loads)
            binary = (folder/(name+".cso")).read_bytes()
            offsets = struct.unpack_from(f"<{struct.unpack_from('<I', binary, 28)[0]}I", binary, 32)
            psv = next(off for off in offsets if binary[off:off+4] == b"PSV0")
            assert binary[psv+13] == bool(mask & 0x11), name  # PSVRuntimeInfo0.PS.SampleFrequency
        for mask in (2 | 4, 32 | 64):
            reads = [0x7e080300, 0x7e0a0303]
            check("bary-conflict", "gfx1013-container-ps",
                  patched_pixel(2, reads, input_address=mask, export=(0xf800180f, 0x05040504)),
                  "two barycentric locations in the same perspective mode")
        check("bary-pull", "gfx1013-container-ps", patched_pixel(1, [0x7e000300], input_address=8),
              "unsupported pixel entry VGPR")
        # Exact EXEC in odd s13 survives an unrelated write to s12 in wave32.
        wqm = [0xbe8d037e, 0xbefe097e, 0xbe8c0380, 0x7e0002f2, 0xbefe030d]
        check("wave32-save-exact", "gfx1013-container-ps", patched_pixel(len(wqm), wqm, wave32=True))
        check("wave64-low-only-wqm", "gfx1013-container-ps",
              patched_pixel(2, [0xbefe097e, 0x7e0002f2]), "only half of the guest EXEC")
        # Both successors are valid, including the block beyond the first return.
        branch = compute(words(0xbf880001, end, end))
        assert "frontend_cache_hit=0" in check("branches", cs, branch)
        assert "frontend_cache_hit=1" in check("branches", cs, branch)
        [cached] = (folder/"cache").glob("*.csir")
        data = bytearray(cached.read_bytes())
        data[-1] ^= 1
        cached.write_bytes(data)
        assert "frontend_cache_hit=0" in check("branches", cs, branch)
        # The read target aliases its byte-address source. Include a group barrier.
        ds = words(0xd8340000, 0, 0xbf8a0000, 0xd8d80000, 0, end)
        for n in (32, 64, 96):
            check(f"lds{n}", cs, compute(ds, n))
        # A split wave ballots both halves and restores EXEC from an SGPR pair.
        mask_program = words(0x7c0400f2, 0xbe84246a, 0x7e0c0500, 0xd7650002, 128 << 9 | 193,
                             0xd7660003, 258 << 9 | 193, 0xbefe0404, end)
        check("split-wave64", cs, compute(mask_program, 64, wave32=False))
        check("two-wave64", cs, compute(mask_program, 128, wave32=False))
        check("multi-wave64-local-branch", cs, compute(words(0xbf880001, end, end), 128, wave32=False))
        check("multi-wave64-exchange-branch", cs, compute(words(0xbf880001, 0x7c0400f2, end), 128, wave32=False),
              "converged group control flow")
        check("partial-wave64", cs, compute(mask_program, 32, wave32=False), "complete guest waves")
        # Wide-carry destination and source are aligned SGPR pairs in a
        # guest wave64. Both host halves must participate in the ballot.
        carry = [0xbe9e04c1, 0xd5281e00, 128 | 256 << 9 | 30 << 18]
        check("wave64-vector-carry", cs, compute(words(*carry, end), 64, wave32=False))
        check("wave64-odd-carry", cs,
              compute(words(carry[0], carry[1] | 0x100, carry[2], end), 64, wave32=False),
              "unsupported carry mask destination")
        check("wave64-divergent-carry", cs,
              compute(words(carry[0], 0xbf880002, *carry[1:], end), 128, wave32=False),
              "multi-wave64 exchange requires converged group control flow")
        split_ir = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"split-wave64.cso"],
                                  check=True, capture_output=True, text=True).stdout
        assert "waveExchange" in split_ir and "waveReadLaneAt" in split_ir and "waveActiveBallot" in split_ir, split_ir
        # A descriptor's stride is symbolic. Its SCC may be discarded
        # only if no path can consume it before a concrete definition.
        stride_field = [0xbe810380, 0xf4080500, 0xfa000000, 0x9391ff15, 0xe0010]
        shift = 0x901a8381  # s_lshr_b32 s26, 1, 3
        read_scc = 0xbe9903fd  # s_mov_b32 s25, src_scc
        check("stride-scc-jump", cs, compute(words(*stride_field, 0xbf820001, read_scc, shift, end), table=True))
        check("stride-scc-join", cs, compute(words(*stride_field, 0xbf880001, shift, read_scc, end), table=True),
              "numeric SCC from a symbolic descriptor field")
        address_field = [0xbe810380, 0xf4080500, 0xfa000000, 0x8715ff15, 0xc000ffff]
        saveexec = 0xbe983c81  # s_and_saveexec_b32 s24, 1
        check("descriptor-saveexec-kill", cs, compute(words(*address_field, saveexec, read_scc, end), table=True))
        check("descriptor-saveexec-bypass", cs,
              compute(words(*address_field, 0xbf880001, saveexec, read_scc, end), table=True),
              "numeric SCC from a symbolic descriptor field")
        pair_zero = [0xbe960380, 0xbe970380]  # s_mov_b32 s22, 0 / s23, 0 (both preserve SCC)
        ff1_pair = 0xbe8e1416  # s_ff1_i32_b64 s14, s[22:23]
        brev = 0xbe8f0b0e  # s_brev_b32 s15, s14
        use_scan = 0x7e00020f  # v_mov_b32 v0, s15
        select_scc = 0x85198e80  # s_cselect_b32 s25, 0, s14
        check("descriptor-sop1-preserves-scc", cs,
              compute(words(*address_field, *pair_zero, ff1_pair, brev, use_scan, end), table=True))
        check("descriptor-sop1-before-scc-read", cs,
              compute(words(*address_field, *pair_zero, ff1_pair, select_scc, end), table=True),
              "numeric SCC from a symbolic descriptor field")
        # The previous symbolic SCC is dead at WQM. Reading it before WQM
        # must still refuse. The nonzero high word supplies the new SCC.
        numeric_wqm = [0xbe9a0380, 0xbe9b0381, 0xbe980a1a]  # s[24:25] = WQM(s[26:27])
        for early in (False, True):
            seq = [*stride_field, *([read_scc] if early else []), *numeric_wqm,
                   0xbf840001, 0xbf800000, 0x7e000218]
            check("pixel-stride-scc-"+("read" if early else "overwritten"), "gfx1013-container-ps",
                  patched_pixel(len(seq),seq,table=True),
                  *(["numeric SCC from a symbolic descriptor field"] if early else []))
        check("sdwa-class-float-byte", cs, compute(words(0x7d100af9, 0x03908e80, end)),
              "unsupported floating class operand selection")
        # A real FSSM root table feeds eight descriptor words. The image
        # dimension changes both the emitted handle and the operand count.
        for dim in (0,1,2,4,5,6,7):
            image = 0xf0000100 | dim << 3
            stream = words(0xbe810380, 0xf40c0200, 0xfa000000,
                           image, 0x00020400, end)
            check(f"image{dim}", cs, compute(stream, table=True))
        check("bad-target", cs, compute(words(0xbf820010, end)), "branch outside")
        # Exercise every 32-bit raw UAV atomic, including GLC's old-value
        # writeback to the VGPR also used as the address.
        atomics = []
        for op in (0x30, 0x31, 0x32, 0x33, *range(0x35, 0x3c)):
            atomics.extend((0xe0005000 | op << 18, 0x80020000))
        def atomic_program(instructions):
            return compute(words(0xbe810380, 0xf4080200, 0xfa000000,
                                 *instructions, end), table=True)
        check("raw-atomics", cs, atomic_program(atomics))
        # gfx1013 VOP3 forms for signed BFE, min3/median, high products
        # and mixed three-source logic. V0 is both source and destination.
        vec = []
        for op in (329, 339, 344, 362, 364, 837, 877, 882):
            vec.extend((0xd4000000 | op << 16, 256 | 129 << 9 | 130 << 18))
        check("integer-vector", cs, compute(words(*vec, end)))
        # SDWA byte/word extraction, signed extension, preservation of the
        # untouched destination, plus scalar or vector source selection.
        sdwa = [0x3a0000f9, 0x06091100, 0x060000f9, 0x06162600]
        for opcode in (17, 18, 19, 20, 32, 33, 34):
            sdwa.extend((0x7e0000f9 | opcode << 9, 0x00060600))
        check("subword-alu", cs, compute(words(*sdwa, end)))
        # Read a V# from the table, clear stride, set byte-address bounds and
        # preserve its extent as num_records * max(1,stride). The subsequent
        # vector access still selects this table and byte window.
        byte_view = [0xbe810380, 0xf4080700, 0xfa000000, 0x939bff1d, 0xe0010, 0x849b811b,
                     0x871dff1d, 0xc000ffff, 0x871fff1f, 0xcfffffff, 0x931e1b1e, 0x881fff1f, 0x30000000,
                     0x7e0a0280, 0xe0381000, 0x80070805]
        check("byte-count-rewrite", "gfx1013-container-ps",
              patched_pixel(len(byte_view), byte_view, table=True, export=(0xf800180f, 0x0b0a0908)))
        broken = byte_view.copy()
        broken[10] = 0xbf800000  # Leave NUM_RECORDS in elements instead of bytes.
        check("byte-count-missing-product", "gfx1013-container-ps",
              patched_pixel(len(broken), broken, table=True), "non-contiguous resource descriptor")
        # The same rewrite in compute, read by MTBUF loads through s[28:31] (v_mov_b32 v5, 0 as the offset):
        # tbuffer_load_format_x v8, v5, s[28:31], 0 format:[BUF_FMT_32_UINT] offen and
        # tbuffer_load_format_xyzw v[8:11], v5, s[28:31], 0 format:[BUF_FMT_8_8_8_8_UNORM] offen.
        tbuffer_view = [*byte_view[:13], 0x7e0a0280]
        check("compute-tbuffer-byte-window", cs, compute(words(*tbuffer_view, 0xe8a01000, 0x80070805, end), table=True))
        check("compute-tbuffer-byte-window-unorm", cs,
              compute(words(*tbuffer_view, 0xe9c31000, 0x80070805, end), table=True))
        # Over-acceptance control only: without the byte-count product the window fields do not match.
        tbuffer_broken = tbuffer_view.copy()
        tbuffer_broken[10] = 0xbf800000
        check("compute-tbuffer-byte-count-missing-product", cs,
              compute(words(*tbuffer_broken, 0xe8a01000, 0x80070805, end), table=True),
              "non-contiguous resource descriptor")
        # Convert v11 to a half in the high word of v6, retaining v6.low.
        # Widen the selected high word back from a scalar source into v7.
        halves = [0x7e0c02ff, 0xa5a53600, 0x7e1602ff, 0x3f801000, 0x7e0c14f9, 0x06150b,
                  0xbe8e03ff, 0x40003c00, 0x7e0e16f9, 0x85060e]
        check("sdwa-half-pair", "gfx1013-container-ps",
              patched_pixel(len(halves), halves, export=(0xf800180f, 0x07060706)))
        half_ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"sdwa-half-pair.cso"],
                                          text=True)
        assert "@dx.op.legacyF16ToF32" in half_ir and "65535" in half_ir, half_ir
        invalid = halves.copy()
        invalid[-1] |= 1 << 19
        check("sdwa-half-signextend", "gfx1013-container-ps", patched_pixel(len(invalid), invalid),
              "invalid SDWA source modifier")
        check("invalid-subword", cs, compute(words(0x3a0000f9, 0x06071700, end)),
              "unsupported SDWA destination modifier")
        for bad in (0x2000, 0x4000, 0x100):
            check("half-mad-mod-" + str(bad), cs,
                  compute(words(0xd7730000 | bad, 257 | 258 << 9 | 259 << 18, end)),
                  "unsupported halfword MAD operand modifiers")
        # Compare different subwords into VCC, then select and preserve
        # the other bytes of the destination. CMPX must write EXEC.
        cmp_sdwa = (0x7c000000 | (2 << 17) | 249, 0x06010000)
        choose_byte = (1 << 25 | 249, 0x06021100)
        check("compare-select-subword", cs, compute(words(*cmp_sdwa, *choose_byte, end)))
        check("cmpx-explicit-destination", cs, compute(words(cmp_sdwa[0] | (0x10 << 17),
              cmp_sdwa[1] | 0x8000, end)), "invalid SDWA compare destination")
        for name, opcode, ext, error in (
            ("class", 0x7d100af9, 0x0396eb80, "invalid SDWA class modifiers or mask destination"),
            ("ne", 0x7d0a0cf9, 0x0086eb6b, "invalid SDWA compare destination"),
        ):
            check(f"sdwa-{name}-vcc-hi-wave64", cs, compute(words(opcode, ext, end), 64, wave32=False), error)
            check(f"sdwa-{name}-sdst-108", cs, compute(words(opcode, (ext & ~0x7f00) | (108 << 8), end)), error)
        check("sdwa-cmpx-vcc-hi", cs, compute(words(0x7d2a0cf9, 0x0086eb6b, end)),
              "invalid SDWA compare destination")
        check("atomic-x2", cs, atomic_program((0xe1445000, 0x80020000)),
              "unsupported buffer atomic width")
        check("bad-container", cs, compute(ds)[:-1], "invalid DXBC length")
        check("vertex", "gfx1013-container-vs", vertex())
        fetch_body = [0xbe842104, 0xf80008cf, 0x04030201, end]
        check("prefixed-fetch", "gfx1013-container-vs",
              vertex(fetch=True, program=[0x7e000080, *fetch_body]))
        check("late-fetch", "gfx1013-container-vs",
              vertex(fetch=True, program=[0x7e000080, 0x7e0002f2, *fetch_body]), "outside the entry")
        # Copy a root address into s[2:3], read 8 dwords at byte 48,
        # then use the last word in an observable vertex position export.
        # Overwriting only the high SGPR must invalidate the pointer.
        root_load = [0xbe820400, 0xf40c0201, 0xfa000030]
        root_vs = vertex(constants=(0,2,1), program=[*root_load, 0x7e00020f, 0xf80008cf, 0, end])
        check("address-vs", "gfx1013-container-vs", root_vs)
        check("address-cs", cs, compute(words(*root_load, end), root_descriptor=True))
        for offset in (106, 107):
            check(f"wave64-smem-offset-{offset}", cs,
                  compute(words(0xf4000200, offset << 25, end), threads=64,
                          root_descriptor=True, wave32=False), "unsupported scalar memory addressing")
        check("wave64-smem-vcc-dst", cs,
              compute(words(0xf4000000 | 107 << 6, 125 << 25, end), threads=64,
                      root_descriptor=True, wave32=False), "unsupported scalar memory destination range")
        check("wave32-smem-cross-vcc", cs,
              compute(words(0xf40c0000 | 104 << 6, 125 << 25, end), root_descriptor=True),
              "unsupported scalar memory destination range")
        # A descriptor into s[104:107] must tag both VCC words as nonnumeric.
        for offset in (106, 107):
            check(f"wave32-smem-symbolic-offset-{offset}", cs,
                  compute(words(0xbe810380, 0xf4080000 | 104 << 6, 125 << 25,
                                0xf4080200, offset << 25, end), table=True),
                  f"unbound numeric SGPR s{offset}")
        wrong_high = [root_load[0], 0xbe830380, *root_load[1:], end]
        check("bad-address-cs", cs, compute(words(*wrong_high), root_descriptor=True), "mismatched root address high half")
        # Build a zero-stride byte-address descriptor in s[4:7]. The x8
        # load starts at 96 and ends exactly at the 128-byte view bound.
        def built_load(bound=128, control=0x31016fac, high=0xbe850301):
            return [0xbe840300, high, 0xbe8603ff, bound, 0xbe8703ff, control, 0xf42c0202, 0xfa000060]
        check("built-root-cs", cs, compute(words(*built_load(), end), root_descriptor=True))
        check("built-root-range", cs, compute(words(*built_load(bound=127), end), root_descriptor=True),
              "exceeds synthesized descriptor byte bound")
        check("built-root-fields", cs, compute(words(*built_load(control=0x01016fac), end), root_descriptor=True),
              "unsupported synthesized scalar buffer descriptor fields")
        check("built-root-pair", cs, compute(words(*built_load(high=0xbe850380), end), root_descriptor=True),
              "mismatched root address high half")
        check("built-root-vs", "gfx1013-container-vs", vertex(constants=(0,2,1),
              program=[*built_load(), 0x7e00020f, 0xf80008cf, 0, end]))
        dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"built-root-vs.cso"],
                              check=True, capture_output=True, text=True).stdout
        assert "@dx.op.rawBufferLoad.i32(i32 139" in dump and ", i32 112," in dump, dump
        dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"address-vs.cso"],
                              check=True, capture_output=True, text=True).stdout
        assert "@dx.op.rawBufferLoad.i32(i32 139" in dump and ", i32 64," in dump, dump
        check("smooth", "gfx1013-container-ps", interpolated_pixel())
        check("mismatch", "gfx1013-container-ps", interpolated_pixel(True), "unmatched interpolation pair")
        check("patched", "gfx1013-container-ps", patched_pixel())
        check("null-export", "gfx1013-container-ps", patched_pixel(export=(0xf8001890,0)))
        clip_sig=signature_rows([(b"SV_Position",1,3,0,15),(b"SV_ClipDistance",2,3,1,3),
                                 (b"SV_CullDistance",3,3,1,4)])
        clip=[0x7e0002f2,0xf80000cf,0,0xf80008d7,0,end]
        check("clip-cull","gfx1013-container-vs",vertex(program=clip,extra=((b"OSG1",clip_sig),)))
        check("clip-unmapped","gfx1013-container-vs",vertex(program=clip),"unmapped position export")
        diamond = [0xbf008180, 0xbf850002, 0x7e0002f2, 0xbf820001, 0x7e0002f4]
        check("pixel-branch", "gfx1013-container-ps", patched_pixel(len(diamond), diamond))
        # Carry the position input through an SCC-controlled two-iteration loop.
        loop = [0xb0000002, 0x7e020300, 0x06020300, 0xb780ffff, 0xbf078000,
                0xbf85fffc, 0x7e000301]
        position_input = ((b"ISG1", signature(b"SV_Position", 1)),)
        check("pixel-loop", "gfx1013-container-ps",
              patched_pixel(len(loop), loop, input_address=1 << 8, input_signature=position_input))
        uninitialized = loop.copy()
        uninitialized[1] = 0xbf800000
        check("pixel-loop-undefined", "gfx1013-container-ps",
              patched_pixel(len(loop), uninitialized, input_address=1 << 8, input_signature=position_input),
              "unsupported pixel entry VGPR")
        missing = diamond.copy()
        missing[-1] = 0xbf800000
        check("undefined-at-join", "gfx1013-container-ps", patched_pixel(len(missing), missing),
              "undefined on an incoming path")
        check("position-preserved-at-join", "gfx1013-container-ps",
              patched_pixel(len(missing), missing, input_address=1 << 8, input_signature=position_input))
        lanes = [0x7e0002f2, 0x7c0400f2, 0x02000080]
        check("lane-select", "gfx1013-container-ps", patched_pixel(len(lanes), lanes))
        # Guest SRC0 is an sNaN bit pattern and V0 is the integer class mask.
        # Do not convert SRC0 to host float before distinguishing sNaN/qNaN.
        for name, word, snan, quiet_bit in (("f16", 0x7d1e00ff, 0x7c01, 512),
                                             ("f32", 0x7d1000ff, 0x7f800001, 4194304)):
            ops = [0x7e000281, word, snan, 0x02000080]
            check("class-"+name, "gfx1013-container-ps", patched_pixel(len(ops), ops))
            ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/f"class-{name}.cso"], text=True)
            fraction = re.search(rf"(%\w+) = and i32 {snan}, {2*quiet_bit-1}\b", ir)
            assert fraction and f"and i32 {fraction[1]}, {quiet_bit}" in ir and "@dx.op.legacyF16ToF32" not in ir
        vcc_diamond = [*lanes[:2], 0xbf870002, *diamond[2:]]
        check("uniform-vcc-join", "gfx1013-container-ps", patched_pixel(len(vcc_diamond), vcc_diamond))
        varying_vcc = vcc_diamond[1:]
        check("varying-vcc-join", "gfx1013-container-ps", patched_pixel(len(varying_vcc), varying_vcc,
              input_address=1 << 8, input_signature=position_input), "condition depends on a varying input")
        masked = [0x7c0400f2, 0xbe84246a, 0xbf880001, 0x7e0002f4, 0xbefe0404]
        check("masked-pixel", "gfx1013-container-ps", patched_pixel(len(masked), masked,
              input_address=1 << 8, input_signature=position_input))
        two_arms = [0x7c0400f2, 0xbe84047e, 0x87fe7e6a, 0xbf880001, 0x7e0002f4,
                    0x8afe7e04, 0xbf880001, 0x7e0002f2, 0xbefe0404]
        # VertexID chooses a mask. A uniform branch inside THEN skips the
        # write to v4; ELSE covers its complement. Both paths reach POS0.
        vertex_arms = [0x7e0802f2, 0x7d040080, 0xbe88246a, 0xbf880006,
                       0xbf008080, 0xbf840002, 0x7e0802f4, 0xbf800000,
                       0x7e0a0304, 0x7e080305, 0x8afe7e08, 0xbf880002,
                       0x7e080281, 0xbf800000, 0xbefe0408, 0xf80008cf, 0x04040404, end]
        check("vertex-mask-branch", "gfx1013-container-vs", vertex(program=vertex_arms))
        ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"vertex-mask-branch.cso"], text=True)
        assert ir.count("br i1") == 3 and "phi i32" in ir, ir
        # v4 has only constant data operands, but VertexID selected the arm.
        # The resulting VCC cannot drive a wave-uniform scalar branch.
        varying_join = vertex_arms[:15] + [0x7d040880, 0xbf870001, 0xbf800000] + vertex_arms[15:]
        check("vertex-mask-phi-dependency", "gfx1013-container-vs", vertex(program=varying_join),
              "condition depends on a varying input")
        # The outer THEN contains an independently saved inner mask. Its ELSE
        # must only change v4 for lanes which entered that outer THEN.
        nested = [0x7e0802f2, 0x7d040080, 0xbe88246a, 0xbf880009,
                  0x7d040081, 0xbe8a246a, 0xbf880002, 0x7e0802f4, 0xbf800000,
                  0x8afe7e0a, 0xbf880001, 0x7e080281, 0xbefe040a, 0xbefe0408,
                  0xf80008cf, 0x04040404, end]
        check("vertex-nested-mask", "gfx1013-container-vs", vertex(program=nested))
        ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"vertex-nested-mask.cso"], text=True)
        assert "br i1" in ir and "select i1" in ir and "phi i32" in ir, ir
        # Capture VCC & EXEC at the CMPX join. The next compare intersects
        # that captured bit before narrowing EXEC for the second region.
        captured_vs = [0x7e0802f2, 0xbe98047e, 0x7d2a0080, 0xbf880002,
                       0x7e0802f4, 0x7d0a0081, 0x87887e6a, 0x8afe7e18,
                       0xbf880001, 0x7e080281, 0xbefe0418, 0x7d0a0082,
                       0x8788086a, 0x87fe7e08, 0xbf880002, 0x7e080284,
                       0xbf800000, 0x8afe7e18, 0xbf880001, 0x7e080285,
                       0xbefe0418, 0xf80008cf, 0x04040404, end]
        check("vertex-cmpx-intersection", "gfx1013-container-vs", vertex(program=captured_vs))
        ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"vertex-cmpx-intersection.cso"], text=True)
        assert "phi i32 [ 0," in ir or ", [ 0," in ir, ir
        assert "and i32" in ir and ir.count("br i1") == 4 and "@dx.op.wave" not in ir, ir
        numeric_mask = captured_vs.copy()
        numeric_mask[9] = 0x7e080208  # No numeric s8 exists on the empty edge.
        check("vertex-cmpx-numeric-wave-mask", "gfx1013-container-vs", vertex(program=numeric_mask),
              "undefined on an incoming path")
        changed_exec = captured_vs.copy()
        changed_exec[4] = 0x7d2a0083
        check("vertex-cmpx-changed-exec", "gfx1013-container-vs", vertex(program=changed_exec),
              "reaching CMPX mask on every path")
        mask_scc = [0x7e0802f2, 0x7d0a0081, 0xd485001a, 0x00020080,
                    0x879a1a6a, 0xbe9a241a, 0xbf880007,
                    0xd40d002a, 0x000100f2, 0x87aa7e2a, 0xbf840001,
                    0x7e080284, 0xbf800000, 0xbf800000, 0xbefe041a,
                    0xf80008cf, 0x04040404, end]
        check("vertex-active-uniform-scc", "gfx1013-container-vs", vertex(program=mask_scc))
        varying_scc = mask_scc.copy()
        varying_scc[8] = 0x00020081  # Other lanes can disagree on this SCC.
        check("vertex-active-varying-scc", "gfx1013-container-vs", vertex(program=varying_scc),
              "native compute +0x24: unbound numeric SGPR s42")
        # Both arms load t0's first dword after their lane-local v4 writes.
        # Its result is read after restoring EXEC, including for an empty ELSE.
        scalar_tail = [0xbe98047e, 0x7d0a0080, 0x87fe7e6a, 0xbf880002,
                       0xbe9a0381, 0x7e0802f4, 0x8afe7e18, 0x7e0802f2,
                       0xf4000200, 0xfa000000, 0xbefe0418, 0x06080808,
                       0xf80008cf, 0x04040404, end]
        check("vertex-shared-scalar-tail", "gfx1013-container-vs",
              vertex(constants=(0,2,1), program=scalar_tail))
        ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"vertex-shared-scalar-tail.cso"], text=True)
        assert "phi i32" in ir and ir.count("call %dx.types.ResRet.i32 @dx.op.rawBufferLoad") == 1, ir
        unsafe_tail = scalar_tail.copy()
        unsafe_tail[7] = 0x7e08021a
        check("vertex-scalar-tail-escaping-sgpr", "gfx1013-container-vs",
              vertex(constants=(0,2,1), program=unsafe_tail), "unbound numeric SGPR s26")
        # SGPR -> f32 -> reciprocal -> scaled integer is uniform at every
        # step. Replacing the first conversion's input with VertexID is not.
        uniform_vs = [0xf4000200, 0xfa000000, 0x7e020c08, 0x7e025501,
                      0x100202ff, 0x41800000, 0x7e020f01, 0x7e120501,
                      0x7e080c09, 0xf80008cf, 0x04040404, end]
        check("vertex-uniform-readfirst-chain", "gfx1013-container-vs",
              vertex(constants=(0,2,1), program=uniform_vs))
        varying_vs = uniform_vs.copy()
        varying_vs[2] = 0x7e020d00
        check("vertex-varying-readfirst-chain", "gfx1013-container-vs",
              vertex(constants=(0,2,1), program=varying_vs), "requires a wave-uniform reaching value")
        # The first ALU overwrites its input VGPR and outputs a separate
        # s[20:21] mask. CNDMASK still sees the original VCC. The next ALU
        # consumes s[20:21], writes VCC and exposes that carry via v6.
        vertex_carry = [0x7d0a0080, 0x7e020300, 0xd5281401, 195 | (257<<9) | (106<<18),
                        0x7e0a02f4, 0x02000000 | (2<<17) | (5<<9) | 242,
                        0xd5286a03, 128 | (193<<9) | (20<<18), 0x02000000 | (6<<17) | (5<<9) | 242,
                        0x7e020b01, 0x7e060b03, 0xf80008cf, 0x06030201, end]
        for op in range(3):
            chain = vertex_carry.copy()
            chain[2] += op<<16
            check("vertex-carry-chain-"+str(op), "gfx1013-container-vs", vertex(program=chain))
        replaced = vertex_carry[:6] + [0xbe950380] + vertex_carry[6:]
        check("vertex-carry-high-overwrite", "gfx1013-container-vs", vertex(program=replaced),
              "unproven vertex carry-in mask")
        numeric = vertex_carry.copy()
        numeric[5] = 0x7e040214
        check("vertex-carry-numeric-sgpr", "gfx1013-container-vs", vertex(program=numeric),
              "unbound numeric SGPR s20")
        # Match the decoder's DWORD input (+0.0), BYTE_3 class mask and
        # s[40:41] destination. A second compare defines s[26:27]; both can
        # still select a result while VCC retains the original vertex test.
        class_select = [0x7d0a0080, 0x3446009e, 0x7d1046f9, 0x0386a880,
                        0x7e0a02f4, 0xd5010004, 242 | (261<<9) | (40<<18),
                        0x020c0af2, 0xd485001a, 130 | (256<<9),
                        0xd5010007, 242 | (261<<9) | (26<<18),
                        0xd5010008, 261 | (242<<9) | (40<<18),
                        0xf80008cf, 0x08070604, end]
        check("vertex-class-mask-select", "gfx1013-container-vs", vertex(program=class_select))
        pixel_class = [0x7e000281, *class_select[:-3]]
        check("pixel-class-mask-select", "gfx1013-container-ps", patched_pixel(len(pixel_class),pixel_class,
              export=(0xf800180f,0x08070604)))
        changed_high = class_select[:5] + [0xbea90380] + class_select[5:]
        check("vertex-class-mask-high-overwrite", "gfx1013-container-vs", vertex(program=changed_high),
              "unproven graphics CNDMASK source")
        bad_class = class_select.copy()
        bad_class[3] |= 1<<16
        check("vertex-class-reserved-selection", "gfx1013-container-vs", vertex(program=bad_class),
              "invalid or unsupported integer SDWA modifiers")
        # s8 is a uniform root-buffer read. Bit 0 chooses identical high
        # and low scalar words and the corresponding CNDMASK bit.
        scalar_select = [0xf4000200,0xfa000000,0xbf0d8008,0x85a480c1,
                         0x7e0a02f4,0xd5010004,242 | (261<<9) | (36<<18),
                         0x7e0c0a24,0x7e0e0a25,0xf80008cf,0x04070604,end]
        check("vertex-scalar-full-mask", "gfx1013-container-vs",
              vertex(constants=(0,2,1), program=scalar_select))
        no_scalar_scc = scalar_select.copy()
        no_scalar_scc[2] = 0x7d0a0080  # VOPC writes this lane's VCC, not SCC.
        check("vertex-no-scalar-mask-proof", "gfx1013-container-vs",
              vertex(constants=(0,2,1), program=no_scalar_scc), "requires a reaching scalar SCC comparison")
        outside = [0xbf840006, *vertex_arms]  # Jump into THEN before saving its mask.
        check("vertex-mask-entry", "gfx1013-container-vs", vertex(program=outside),
              "branch enters or crosses a masked vertex arm")
        store = vertex_arms.copy()
        store[6:8] = [0xe0701000,0x80020100]
        check("vertex-mask-store", "gfx1013-container-vs", vertex(program=store),
              "masked vertex arm has an unsupported exit or memory effect")
        overwritten = vertex_arms.copy()
        overwritten[7] = 0xbe880380
        check("vertex-mask-clobber", "gfx1013-container-vs", vertex(program=overwritten),
              "masked vertex arm overwrites its saved EXEC")
        escaping_scalar = vertex_arms.copy()
        escaping_scalar[7] = 0xbe9a0381  # THEN writes s26; ELSE must not see a per-lane surrogate.
        escaping_scalar[12] = 0x7e08021a
        source = folder/"vertex-mask-scalar"
        source.write_bytes(vertex(program=escaping_scalar))
        rejected = subprocess.run([exe, sdk, folder/"cache", "gfx1013-container-vs", source, source.with_suffix(".cso")],
                                  capture_output=True, text=True)
        assert rejected.returncode and ("unbound numeric SGPR s26" in rejected.stderr or
                                        "undefined on an incoming path: 26" in rejected.stderr), rejected.stderr
        check("split-mask-two-arms", "gfx1013-container-ps", patched_pixel(len(two_arms), two_arms,
              input_address=1 << 8, input_signature=position_input))
        cmpx = [0xbefe0a7e, 0xbe8a047e, 0x7c2800f4, 0x7e0002f2,
                0x8afe7e0a, 0x7e0002f0, 0xbefe040a]
        check("cmpx-partition", "gfx1013-container-ps", patched_pixel(len(cmpx), cmpx,
              input_address=1 << 8, input_signature=position_input))
        unproven = cmpx.copy()
        unproven[1] = 0xbf800000
        check("cmpx-wrong-parent", "gfx1013-container-ps", patched_pixel(len(unproven), unproven,
              input_address=1 << 8, input_signature=position_input), "unproven enclosing whole-quad EXEC")
        first = [0xd557000e, (240 << 18) | (128 << 9) | 242, 0x7e28050e, 0x7e000214]
        check("readfirstlane-uniform-median", "gfx1013-container-ps", patched_pixel(len(first), first))
        varying = [0x7e280500, 0x7e000214]
        check("readfirstlane-varying-position", "gfx1013-container-ps", patched_pixel(len(varying), varying,
              input_address=1 << 8, input_signature=position_input), "wave-uniform reaching value")
        partition = [0xbe8c047e, 0xbefe0a7e, 0xbe8a047e, 0x7c0400f2,
                     0xd4010016, 0x000200f2, 0x8896166a, 0x87fe7e16,
                     0xbf880001, 0x7e0002f4, 0x8afe7e0a, 0xbf880003,
                     0xbe96240c, 0x7e0002f2, 0xbe9e2516, 0xbefe040a]
        check("compare-union-exec-arms", "gfx1013-container-ps", patched_pixel(len(partition), partition,
              input_address=1 << 8, input_signature=position_input))
        foreign_mask = partition.copy()
        foreign_mask[12] = 0xbe962408
        check("unproven-inner-coverage", "gfx1013-container-ps", patched_pixel(len(foreign_mask), foreign_mask,
              input_address=1 << 8, input_signature=position_input), "inner EXEC requires saved entry coverage")
        unsafe = masked.copy()
        unsafe[3] = 0xb0000002  # scalar state changes for the whole guest wave
        check("masked-scalar", "gfx1013-container-ps", patched_pixel(len(unsafe), unsafe,
              input_address=1 << 8, input_signature=position_input), "lane-local vector arithmetic")
        full_mask = lanes + [0xbe80036a]
        check("full-mask", "gfx1013-container-ps", patched_pixel(len(full_mask), full_mask),
              "consumes a full wave mask")
        # Two barycentric registers, three pull-model registers, one line
        # stipple register, then position.y in V6 (position.x disabled).
        position = [0x7e000306]
        mask = (1 << 1) | (1 << 3) | (1 << 7) | (1 << 9)
        check("pixel-position", "gfx1013-container-ps", patched_pixel(1, position, input_address=mask,
              input_signature=((b"ISG1", signature(b"SV_Position", 1)),)))
        # Export the numeric I,J,J,I registers. DXIL orders the three
        # weights by vertex, so I/J must select components 1/2.
        ij = patched_pixel(prefix=[0xbf800000], input_address=2, export=(0xf800180f,0x00010100))
        check("center-weights", "gfx1013-container-ps", ij)
        ij_ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe","-dumpbin",folder/"center-weights.cso"],text=True)
        assert "SV_Barycentrics" in ij_ir and all(
            f"@dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 {col}," in ij_ir for col in (1,2)), ij_ir
        # Covered lanes replace I with 1/4. A subsequent horizontal quad
        # shuffle must still see original I on the other side of an edge.
        edge = [0xbe8c047e,0xbefe0a7e,0xbe8a047e,0xbefe040c,0x7e0002ff,0x3e800000,
                0xbefe040a,0x060000fa,0xff00b100]
        check("masked-ij-quad-edge", "gfx1013-container-ps", patched_pixel(len(edge),edge,input_address=2))
        check("pull-weights", "gfx1013-container-ps",
              patched_pixel(prefix=[0xbf800000], input_address=8, export=(0xf800180f,0x00010100)),
              "unsupported pixel entry VGPR")
        for semantic,system,component,channel in (
                (b"SV_Depth",65,3,0), (b"SV_DepthLessEqual",68,3,0),
                (b"SV_DepthGreaterEqual",67,3,0), (b"SV_StencilRef",69,1,1),
                (b"SV_Coverage",66,1,2)):
            output=((b"OSG1",signature(semantic,system,component,0xffffffff,1)),)
            data=patched_pixel(input_signature=output, export=(0xf8001880 | (1<<channel),0))
            check(semantic.decode(), "gfx1013-container-ps", data)
        check("missing-depth", "gfx1013-container-ps", patched_pixel(export=(0xf8001881,0)), "unmapped depth")
        face=((b"ISG1",signature(b"SV_IsFrontFace",9,1,0,1)),(b"OSG1",signature(b"SV_Target",64,1)))
        check("face", "gfx1013-container-ps",
              patched_pixel(prefix=[0x7e000300],input_address=1<<12,input_signature=face))
        # Use the same VGPR for the shuffled operand, unshuffled operand and result.
        for control in (0xaa,0xb1,0x4e,0x1b,0xf5):
            quad=[0xbefe0a7e,0x060000fa,0xff000000 | control<<8]
            check(f"quad-{control}", "gfx1013-container-ps", patched_pixel(len(quad), quad,
                  input_address=1<<8,input_signature=position_input))
        # In WQM every source of a quad-local broadcast or swap is active.
        for perm in (0, 0xb1):
            lanes = [(perm >> (2 * n)) & 3 for n in range(4)]
            dpp8 = sum((lanes[n % 4] + (n & 4)) << (8 + 3 * n) for n in range(8))
            variants = [(0x060000fa, 0xff0c0000 | perm << 8), (0x060000e9, dpp8), (0x060000ea, dpp8)]
            for n, (op, control) in enumerate(variants):
                label = f"quad-fi-{perm}-{n}"
                args = [0xbefe0a7e, op, control]
                check(label, "gfx1013-container-ps", patched_pixel(len(args), args, input_address=2))
                ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/(label+".cso")], text=True)
                assert ("@dx.op.quadReadLaneAt.i32(" if not perm else "@dx.op.quadOp.i32(") in ir
            check("quad-dpp8-cross"+str(perm), "gfx1013-container-ps",
                  patched_pixel(3, [0xbefe0a7e, 0x060000ea, dpp8 ^ (4 << 8)], input_address=2),
                  "repeating quad-local selections")
        check("quad-fi-exact", "gfx1013-container-ps",
              patched_pixel(2, [0x060000fa, 0xff0c0000], input_address=2), "requires a proven whole quad")
        check("quad-half-row-mirror", "gfx1013-container-ps",
              patched_pixel(3, [0xbefe0a7e, 0x060000fa, 0xff054100], input_address=2),
              "half-row mirror requires a proven eight-lane pixel layout")
        full = [0xbebe25c1,0x7e2202fa,0xff054100,0xbefe043e]
        check("mirror-unknown-pixel-lanes","gfx1013-container-ps",
              patched_pixel(len(full),full,input_address=2),"needs an unobserved source lane")
        full[-1] += 2
        check("mirror-unmatched-exec","gfx1013-container-ps",
              patched_pixel(len(full),full,input_address=2),"matching restore in the same block")
        quad[0]=0xbf800000
        check("quad-no-wqm","gfx1013-container-ps",patched_pixel(len(quad),quad),
              "without whole-quad entry")
        quad[-1]^=1<<28
        check("quad-partial-mask","gfx1013-container-ps",patched_pixel(len(quad),quad),
              "full-row, full-bank")
        check("missing-position", "gfx1013-container-ps", patched_pixel(1, position, input_address=mask),
              "position absent from pixel input signature")
        for op in (32, 37):
            prefix = [0xbe810380, 0xf40c0200, 0xfa000000,
                      0xbe840380, 0xbe8503ff, 0x00fff000, 0xbe8603ff, 0x04000000, 0xbe870380,
                      0x7e0002f0, 0x7e0202f2, 0x7e0402f4, 0xf0000108 | op << 18, 0x00220000]
            check(f"sample{op}", "gfx1013-container-ps", patched_pixel(len(prefix), prefix, True))
            check(f"derivative{op}", cs, compute(words(*prefix, end), table=True),
                  "implicit derivatives require a pixel stage")
        image_setup = prefix[:-2]
        for op in (32, 34, 36, 37, 39):
            coords = [0x7e060280, 0x7e080280, 0x7e0a0280]
            sample = [*image_setup, *coords, 0xf0000e08 | (op << 18), 0x80220400]
            name = f"packed-sample-{op}"
            check(name, "gfx1013-container-ps",
                  patched_pixel(len(sample), sample, table=True, export=(0xf800180f, 0x05040504)))
            ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/(name+".cso")], text=True)
            components = re.findall(r"extractvalue %dx.types.ResRet.f32 %\w+, (\d)", ir)
            assert components == ["1", "2", "3"] and "shl i32" in ir, (name, components)
        invalid = [*image_setup, 0xf09c0e18, 0x80220400]  # Cube needs a separate view contract.
        check("packed-sample-cube", "gfx1013-container-ps", patched_pixel(len(invalid),invalid,table=True),
              "unsupported image dimension or D16")
        # The derivative sample executes with its quad intact. Coverage masks
        # the writeback to v0; a helper must keep the pre-sample coordinate.
        window = [0xbe95037e, 0xbefe097e, 0xbe97037e, *image_setup, 0xbf820000,
                  0xbefe0315, 0xf0800108, 0x00220000, 0xbefe0317, 0x060000fa, 0xff00b100, 0xbefe0315]
        check("wave32-exact-sample", "gfx1013-container-ps",
              patched_pixel(len(window), window, table=True, wave32=True))
        sample_ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin",
                                             folder/"wave32-exact-sample.cso"], text=True)
        assert all(op in sample_ir for op in ("@dx.op.sample.f32", "@dx.op.isHelperLane.i1", "select i1",
                                               "@dx.op.quadOp.i32"))
        clobbered = window.copy()
        clobbered.insert(3, 0xbe950380)  # Overwrite the exact coverage mask in s21.
        check("wave32-clobbered-exec", "gfx1013-container-ps",
              patched_pixel(len(clobbered), clobbered, table=True, wave32=True),
              "unproven EXEC for a conditional sample")
        for name, inst, intrinsic in (("raw", 0xe0701000, "rawBufferStore.i32"),
                                       ("typed", 0xe0102000, "bufferStore.i32")):
            store = [*image_setup, inst, 0x80020100]
            check("pixel-buffer-" + name, "gfx1013-container-ps", patched_pixel(len(store), store, True))
            dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/("pixel-buffer-" + name + ".cso")],
                                  check=True, capture_output=True, text=True).stdout
            assert "@dx.op." + intrinsic + "(" in dump, dump
        for channels in (1, 2, 4):
            # Two distinguishable binary16 values in each VGPR.
            mask = (1 << channels) - 1
            store = 0xf0200008 | mask << 8
            code = [*image_setup, 0x7e0802ff, 0x40003c00, 0x7e0a02ff, 0xc000bc00, store, 0x80020400, end]
            label = f"d16-store-{channels}"
            check(label, cs, compute(words(*code), table=True))
            check("pixel-" + label, "gfx1013-container-ps", patched_pixel(len(code)-1, code[:-1], True))
            dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/(label+".cso")],
                                  check=True, capture_output=True, text=True).stdout
            assert dump.count("call float @dx.op.legacyF16ToF32(") == channels and "@dx.op.textureStore.f32(" in dump, dump
        vs_store = [*code[:-1], 0xf80008cf, 0, end]
        check("vertex-uav", "gfx1013-container-vs", vertex(program=vs_store, table=True))
        code[-2] = 0x8002ff00
        check("d16-store-overflow", cs, compute(words(*code), table=True), "image data overflow")
        code[-3], code[-2] = 0xf0200508, 0x80020400
        check("d16-store-holes", cs, compute(words(*code), table=True), "image opcode or descriptor modifiers")
        for op, profile, dim, method in ((40, "gfx1013-container-ps", 1, "sampleCmp"),
                                         (47, cs, 5, "sampleCmpLevelZero")):
            code = comparison_program(op, dim)
            label = f"comparison-{op}"
            source = compute(words(*code, end), table=True) if profile == cs else patched_pixel(len(code), code, True)
            check(label, profile, source)
            dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/(label+".cso")],
                                  check=True, capture_output=True, text=True).stdout
            assert f"@dx.op.{method}.f32(" in dump and "i32 32782" in dump, dump
        cube_cmp = comparison_program(dim=3)
        check("cube-array-compare", "gfx1013-container-ps", patched_pixel(len(cube_cmp), cube_cmp, True))
        code = comparison_program(sampler=0x1012)  # point without coordinate truncation
        check("comparison-invalid", cs, compute(words(*code, end), table=True), "unsupported comparison sampler fields")
        for op in (32, 39):
            code = [*image_setup, 0xf0000508 | op << 18, 0x40220002]
            check(f"packed-{op}", "gfx1013-container-ps", patched_pixel(len(code), code, True))
            dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/f"packed-{op}.cso"],
                                  check=True, capture_output=True, text=True).stdout
            assert dump.count("call float @dx.op.legacyF16ToF32(") == 2, dump
        # Lower all four cube ALU operations, then use face coordinates in a sample.
        cube = list(image_setup)
        for op in range(0x144, 0x148):
            cube.extend((0xd4000000 | op << 16 | (4 + op - 0x144), 256 | 257 << 9 | 258 << 18))
        cube.extend((0xf09c011a, 0x00220005, 0x00000406))  # S=V5, T=V6, face=V4
        check("cube", "gfx1013-container-ps", patched_pixel(len(cube), cube, True))
        dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"cube.cso"],
                              check=True, capture_output=True, text=True).stdout
        assert "@dx.op.sampleLevel.f32(" in dump and "fcmp oge float" in dump and "fptosi float" in dump and "cubearray" in dump.lower(), dump
        code = [*image_setup, 0xf0000118, 0x00020000]
        check("cube-load", "gfx1013-container-ps", patched_pixel(len(code), code, True), "unsupported cube")
        for label, op, dim, mask, flags, nsa, expected_ir in (
                ("gradient", 34, 1, 1, 0, [0x00020100, 1], "sampleGrad.f32"),
                ("gradient-cube-array", 34, 3, 1, 0, [0x00020100, 0x0201], "sampleGrad.f32"),
                ("gradient-array", 34, 5, 1, 0, [0x00020100, 0x0201], "sampleGrad.f32"),
                ("lod-query", 96, 1, 3, 0x1000, [], "calculateLOD.f32"),
                ("cube-array-lod", 96, 3, 3, 0, [], "calculateLOD.f32"),
                ("cube-array-gather", 71, 3, 2, 0, [], "textureGather.f32"),
                ("gather-green", 64, 1, 2, 0, [], "textureGather.f32"),
                ("gather-blue-lz", 71, 1, 4, 0, [], "textureGather.f32"),
                ("mip-size", 14, 1, 15, 0, [], "getDimensions")):
            instruction = 0xf0000000 | op << 18 | dim << 3 | mask << 8 | flags | len(nsa) << 1
            code = [*image_setup, instruction, 0x00220000, *nsa]
            if op == 64:
                check(label+"-mips", "gfx1013-container-ps", patched_pixel(len(code), code, True),
                      "implicit gather requires a sampler fixed to mip zero")
                code[image_setup.index(0x00fff000)] = 0
            check(label, "gfx1013-container-ps", patched_pixel(len(code), code, True))
            dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/(label+".cso")],
                                  check=True, capture_output=True, text=True).stdout
            assert f"@dx.op.{expected_ir}(" in dump, dump
            if op == 96 and dim == 1:
                assert "i1 true)" in dump and "i1 false)" in dump and "fdiv float" in dump, dump
            if op == 34:
                wrong = [*image_setup, instruction - 2, 0x00220000, nsa[0]]
                check(label+"-short", "gfx1013-container-ps", patched_pixel(len(wrong), wrong, True),
                      "incomplete image NSA address list")
        pair=[exe,sdk,folder/"cache","gfx1013-container-pair",
              folder/"vertex",folder/"linked.vs.cso",folder/"patched",folder/"linked.ps.cso"]
        for expected in (0, 1):
            result = subprocess.run(pair, check=True, capture_output=True, text=True)
            assert f"graphics_pair frontend_cache_hit={expected}" in result.stdout
        assert (folder/"linked.vs.cso").read_bytes()==(folder/"vertex.cso").read_bytes()
        assert (folder/"linked.ps.cso").read_bytes()==(folder/"patched.cso").read_bytes()
        pair[-2] = folder/"bary-two-pairs"
        for warm in (0, 1):
            result = subprocess.run(pair, check=True, capture_output=True, text=True)
            assert f"graphics_pair frontend_cache_hit={warm}" in result.stdout
        pair[-2]=folder/"smooth"
        mismatch=subprocess.run(pair,capture_output=True,text=True)
        assert mismatch.returncode and "stages disagree" in mismatch.stderr
        # VS reads root 4, PS reads root 1. The linked b0 puts root 4 after
        # four PS words although both stages load the guest value from s0.
        check("root-vs", "gfx1013-container-vs", vertex(constants=(4, 1)))
        check("root-ps", "gfx1013-container-ps", patched_pixel(prefix=[0x7e000200], constants=(1, 4)))
        pair[-4], pair[-2] = folder/"root-vs", folder/"root-ps"
        for warm in range(2):
            result = subprocess.run(pair, check=True, capture_output=True, text=True)
            assert f"frontend_cache_hit={warm}" in result.stdout
        for output, col in (("linked.vs.cso", 1), ("linked.ps.cso", 0)):
            dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/output],
                                  check=True, capture_output=True, text=True).stdout
            assert any("@dx.op.cbufferLoadLegacy.i32(i32 59" in line and f", i32 {col})" in line
                       for line in dump.splitlines()), dump
        (folder/"root-ps").write_bytes(patched_pixel(prefix=[0x7e000200], constants=(1, 32)))
        overflow = subprocess.run(pair, capture_output=True, text=True)
        assert overflow.returncode and "exceeds 32 words" in overflow.stderr, overflow.stderr
        # Directory mode consumes the runtime's last-published .a marker and
        # matches the FNV key over both containers, the root, and optional IA.
        capture = folder / "requests"
        capture.mkdir()
        root = b"DXBC" + bytes(16) + words(1, 68, 1, 36) + b"RTS0" + words(24, 2, 0, 24, 0, 24, 0)
        stem = publish(capture, vertex(), patched_pixel(), root)
        batch = [exe, sdk, folder/"cache", "--jobs", "2", "gfx1013-container-pair",
                 "--requests", capture, folder/"warm"]
        raw_vs = words(0x7e020b00, 0xf80008cf, 0x01010101, end)
        raw_ps = words(0x7e0002f2, 0xf8001c0f, 0, end)
        # A recorded logical table can contain scalar buffers outside the
        # serialized desktop range. Both stage identities and each used
        # record are checked, including when a warm cache already exists.
        rs = folder/"short-root.hlsl"
        rs.write_text('#define RS "DescriptorTable(CBV(b27,space=1))"\n')
        short = folder/"short-root.bin"
        subprocess.run([pathlib.Path(sdk)/"dxc.exe","-T","rootsig_1_1","-E","RS",rs,"-Fo",short],check=True)
        interfaces = folder/"reviewed-cache"/"interfaces"
        interfaces.mkdir(parents=True)
        def install(raw, container):
            (interfaces/(hashlib.sha256(raw).hexdigest()+".cdxil")).write_bytes(container)
        ps_id = [0xd5800000,255,0x789,0xd5800000,255,0xabc]
        ps = words(*ps_id) + raw_ps
        install(ps, patched_pixel(7,[*ps_id,0x7e0002f2]))
        for slot in (2,5):
            vs_id = [0xd5800000,255,0x123,0xd5800000,255,slot]
            op = [*vs_id,0xbe890380,0xf4080204,0xfa000000+32*slot,
                  0xf42000c4,0xfa000000,0x7e000203,0xf80008cf,0,end]
            raw = words(*op)
            fixture_dir = folder/f"reflected-slot-{slot}"
            publish(fixture_dir, ps, raw, short.read_bytes(), ps_inputs=(0x8080, 0x8080))
            job = [exe,sdk,interfaces.parent,"gfx1013-container-pair","--requests",fixture_dir,folder/"warm"]
            for offset,error in ((8*slot,None),(8*slot+8,"raw load disagrees with reflected table record")):
                install(raw, vertex(program=op, uses=[(0x302,offset,0,0,4 | 8<<8)]))
                result = subprocess.run(job, capture_output=True,text=True)
                if error:
                    assert result.returncode and error in result.stderr,result.stderr
                else:
                    assert result.returncode == 0,result.stderr
        install(raw, patched_pixel(len(vs_id),vs_id))
        install(ps, vertex(program=[*ps_id, *struct.unpack("<4I",raw_vs)]))
        result = subprocess.run(job,capture_output=True,text=True)
        assert result.returncode and "raw exports disagree with reflected stage" in result.stderr,result.stderr
        for label, fragment, diagnostic in (
                ("raw-export", raw_ps, ""),
                ("raw-branch", words(0xbf820000) + raw_ps, "verified stage interface or wave state")):
            raw_dir = folder / label
            raw_key = publish(raw_dir, fragment, raw_vs, root)
            raw_job = batch.copy()
            raw_job[-2] = raw_dir
            result = subprocess.run(raw_job, capture_output=True, text=True)
            if diagnostic:
                assert result.returncode and diagnostic in result.stderr, result.stderr
            else:
                assert result.returncode == 0, result.stderr
                for stage in ("ps", "vs"):
                    assert (folder/"warm"/f"{raw_key}.{stage}.cso").read_bytes().startswith(b"DXBC")
        # One scalar entry precedes the packed screen coordinate. A different
        # mask moves that entry and must change the request and compilation.
        packed_ps = words(0x7e000cf9, 0x00040601, 0xf800180f, 0, end)
        # Two independent comparisons kill fragment lanes before their color
        # export. Empty waves branch to EXP_NULL beyond the first endpgm.
        kill = [0xbe90047e, 0xbefe0410, 0x7e1002f2, 0x7e0a0280,
                0xd4010008, 0x21100, 0x7d040a80, 0x8888086a, 0x8afe087e, 0xbf880003,
                0xf800180f, 0, end, 0xf8001890, 0x80808080, end]
        for name, clobber in (("terminal-discard", False), ("terminal-discard-clobber", True)):
            sequence = kill.copy()
            if clobber:
                sequence.insert(6, 0xbe880380)
            fixture = folder/name
            key = publish(fixture, words(*sequence), raw_vs, root, ps_inputs=(2, 2))
            job = batch.copy()
            job[-2] = fixture
            result = subprocess.run(job, capture_output=True, text=True)
            if clobber:
                assert result.returncode and "discard mask has no reaching lane comparison" in result.stderr
            else:
                assert result.returncode == 0, result.stderr
                ir = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin",
                                              folder/"warm"/f"{key}.ps.cso"], text=True)
                assert "call void @dx.op.discard(i32 82, i1 %" in ir
        for name, entry, ps, vs, ok in (
                ("packed-input", (0x8080, 0x8080), packed_ps, raw_vs, True),
                ("packed-absent", (0, 0), packed_ps, raw_vs, False),
                ("center-position-zw", (0x0c02, 0x0c02),
                 words(0xf800180f, 0x03020302, end), raw_vs, True),
                ("center-position-zw-packed", (0x8c02, 0x8c02),
                 words(0xf800180f, 0x04040302, end), raw_vs, True),
                ("center-position-missing-w", (0x0402, 0x0402),
                 words(0xf800180f, 0x03020302, end), raw_vs, False),
                ("whole-quad-second-param", (2, 2),
                 words(0xbe8c047e, 0xbefe0a7e, 0xc8100400, 0xc8110401,
                       0xbefe040c, 0xf800180f, 0x04040404, end),
                 words(0xbe8c0400, 0x938dff0d, 0x80000, 0xf4080406, 0xfa000000,
                       0x4a000002, 0xbf8cc07f, 0xe03c2000, 0x80040100,
                       0xe0342010, 0x80040400, 0xe9c3200c, 0x80040800,
                       0xf80008cf, 0x08030201, 0xf8000211, 0x04040404, end), True),
                ("whole-quad-param", (0x8002, 0x8002),
                   words(0xbe8c047e, 0xbefe0a7e, 0xbe8a047e, 0xc8100000, 0xc8140100,
                         0xc8110001, 0xc8150101, 0xbefe040c, 0xf800180f, 0x05040504, end),
                 words(0x7e020b00,0xf80008cf,0x01010101,0xf800020f,0x01010101,end), True)):
            fixture = folder / name
            key = publish(fixture, ps, vs, root, ps_inputs=entry)
            job = batch.copy()
            job[-2] = fixture
            result = subprocess.run(job, capture_output=True, text=True)
            assert (result.returncode == 0) == ok, (name, result.stderr)
            if not ok:
                assert "pixel entry VGPR" in result.stderr, result.stderr
            elif name.startswith("center-position-"):
                binary = folder/"warm"/f"{key}.ps.cso"
                dump = subprocess.check_output([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", binary], text=True)
                channels = re.findall(r"call float @dx\.op\.loadInput\.f32\(i32 4, i32 0, i32 0, i8 (\d+)", dump)
                assert set(channels) == set("0123" if name.endswith("packed") else "23"), (name, channels)
        # Mask a real PS root constant to select table byte offset 0 or 32.
        # Reading through the resulting descriptor must require two SRV slots.
        indexed = [0xbe810380, 0x8702a002, 0xf4080200, 2 << 25,
                   0xf42000c4, 0xfa000000, 0x7e000203]
        indexed_ps = patched_pixel(len(indexed), indexed,
                                   root_inputs=((3,0,1,0,0), (0,0,1,0,0x102)))
        # s4 supplies the byte offset and is also the first of eight destinations.
        # The second raw load still consumes the address computed at entry.
        load8 = [0x8704a002, 0xf40c0100, (4 << 25) + 16, 0x7e00020b]
        descriptor_inputs = ((1,0,2,0,0), (0,0,1,0,0x102))
        check("root-index", "gfx1013-container-ps",
              patched_pixel(len(load8), load8, root_inputs=descriptor_inputs))
        dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"root-index.cso"],
                              check=True, capture_output=True, text=True).stdout
        first, second = re.findall(r"call .*?@dx\.op\.rawBufferLoad\.i32\(i32 139, [^,]+, i32 (%\w+),", dump)
        assert f"{second} = add i32 {first}, 16" in dump, dump
        unknown8 = [*load8[1:]]
        unknown8[1] = (2 << 25) + 16
        check("root-index-range", "gfx1013-container-ps",
              patched_pixel(len(unknown8), unknown8, root_inputs=descriptor_inputs),
              "unbounded or unaligned root descriptor index")
        for label, declaration, error in (
                ("cbv", "CBV(b0)", ""),
                ("srv", "SRV(t0, visibility=SHADER_VISIBILITY_VERTEX)", ""),
                ("uav", "UAV(u0)", "requires a root CBV or SRV"),
                ("hidden", "SRV(t0, visibility=SHADER_VISIBILITY_PIXEL)", "visibility mismatch"),
                ("array2", "DescriptorTable(SRV(t0,numDescriptors=2)),RootConstants(num32BitConstants=1,b0)", ""),
                ("array1", "DescriptorTable(SRV(t0)),RootConstants(num32BitConstants=1,b0)", "exceeds its declared root range")):
            rs = folder/"root.hlsl"
            rs.write_text(f'#define RS "{declaration}"\n')
            compiled_root = folder/"root.bin"
            subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-T", "rootsig_1_1", "-E", "RS",
                            rs, "-Fo", compiled_root], check=True)
            source = folder/("descriptor-" + label)
            vs, ps = (vertex(), indexed_ps) if label.startswith("array") else (root_vs, patched_pixel())
            descriptor_stem = publish(source, vs, ps, compiled_root.read_bytes())
            job = batch.copy()
            job[-2] = source
            for warm in range(1 if error else 2):
                result = subprocess.run(job, capture_output=True, text=True)
                if error:
                    assert result.returncode and error in result.stderr, result.stderr
                else:
                    assert result.returncode == 0 and f"frontend_cache_hit={warm}" in result.stdout, result.stderr
            if label == "array2":
                dump = subprocess.run([pathlib.Path(sdk)/"dxc.exe", "-dumpbin", folder/"warm"/f"{descriptor_stem}.ps.cso"],
                                      check=True, capture_output=True, text=True).stdout
                assert "i32 0, i32 1, i32 0, i8 0 }" in dump and "i1 true)" in dump, dump
        for expected in (0, 1):
            result = subprocess.run(batch, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert f"graphics_pair frontend_cache_hit={expected}" in result.stdout
            assert "requests=1 compiled=1 refused=0" in result.stdout
        for stage, reference in (("vs", "vertex"), ("ps", "patched")):
            assert (folder/"warm"/f"{stem}.{stage}.cso").read_bytes() == (folder/f"{reference}.cso").read_bytes()
        rect=batch.copy()
        rect[5]="gfx1013-container-rect"
        for _ in range(2):
            result=subprocess.run(rect,capture_output=True,text=True)
            assert result.returncode==0, result.stderr
        gs=folder/"warm"/f"{stem}.gs.cso"
        dump=subprocess.run([pathlib.Path(sdk)/"dxc.exe","-dumpbin",gs],capture_output=True,text=True,check=True).stdout
        assert dump.count("call void @dx.op.emitStream(")==6 and dump.count("call void @dx.op.cutStream(")==2
        (capture / f"req_{stem}.root").write_bytes(root + b"!")
        result = subprocess.run(batch, capture_output=True, text=True)
        assert result.returncode and "graphics request key mismatch" in result.stderr
        for case, element, error in (
                ("stream", b"position 0 2 3 4294967295 0 0", ""),
                ("instanced", b"POSITION 0 2 5 16 1 2", ""),
                ("semantic", b"COLOR 0 2 0 0 0 0", "missing vertex input"),
                ("component", b"POSITION 0 3 0 0 0 0", "vertex format disagrees")):
            folder_input = folder/case
            publish(folder_input, vertex(True), patched_pixel(), root, b"ia1 1\n"+element+b"\n")
            batch[-2] = folder_input
            for warm in range(1 if error else 2):
                result = subprocess.run(batch, capture_output=True, text=True)
                if error:
                    assert result.returncode and error in result.stderr, result.stderr
                else:
                    assert result.returncode == 0, result.stderr
                    assert f"frontend_cache_hit={warm}" in result.stdout
        check("patch-range", "gfx1013-container-ps", patched_pixel(3), "record outside")
        check("pixel", "gfx1013-direct-constant-ps", words(0x7e0002f2, 0xf800180f, 0, end))


if __name__ == "__main__":
    main()
