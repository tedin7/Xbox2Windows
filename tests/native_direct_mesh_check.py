"""Source-built mesh ISA, direct DXIL validation and fail-closed controls."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def triangle():
    return [
        0xbefc03ff, 0x1003, 0xbf900009,  # m0 = (1 primitive << 12) | 3 vertices; GS_ALLOC_REQ
        0xd7650000, 0x100c1,             # v0 = mbcnt_lo(-1, 0)
        0x36020081, 0x2c040081,          # v1 = v0 & 1; v2 = v0 >> 1
        0x7e020d01, 0x7e040d02,          # convert to f32
        0x7e060280, 0x7e0802f2,          # z = 0; w = 1
        0xf80008cf, 0x04030201,          # POS0
        0x7e0a02ff, 0x200400,            # triangle indices 0,1,2
        0xf8000941, 0x80808005, 0xbf810000,
    ]


def checked_pixel_entry(folder, compiler, sdk):
    from mesh57_pixel_entry import pixel_entry
    from gfx2hlsl import fnv, mesh57_key

    # One source-built v_mov/EXP pixel program. Only a reviewed version 2
    # profile with all six exact spans may supply its entry to the offline compiler.
    ms = struct.pack("<%dI" % len(triangle()), *triangle())
    ps = struct.pack("<8I", *([0x7e0002f2] * 5), 0xf800180f, 0, 0xbf810000)
    root_source = folder / "checked-root.hlsl"
    root_source.write_text('#define RS "RootFlags(0)"')
    root_file = folder / "checked-root.bin"
    subprocess.run([sdk / "dxc.exe", "-T", "rootsig_1_1", "-E", "RS", "-Fo", root_file, root_source], check=True)
    root, state_a, state_b = root_file.read_bytes(), b"abcd", b"efgh"
    key = mesh57_key(ms, ps, root)
    request = folder / f"req_{key:016x}.ps57"
    for ext, part in ((".m57", ms), (".ps57", ps), (".root", root)):
        request.with_suffix(ext).write_bytes(part)
    desc = bytearray(0x300)
    for word, length in ((2, len(state_a)), (4, len(state_b)), (6, len(ps)), (14, len(ms))):
        struct.pack_into("<Q", desc, 8 * word, length)
    fixed = [desc, state_a, state_b]
    variant = key
    for part in fixed:
        variant = fnv(part, fnv(struct.pack("<Q", len(part)), variant))
    archive = folder / f"req_{variant:016x}.s57"
    archive.write_bytes(struct.pack("<Q3I", key, *map(len, fixed)) + b"".join(fixed))
    parts = fixed + [root, ms, ps]
    header = bytearray(532)
    struct.pack_into("<8s6I", header, 0, b"X2WM572", *map(len, parts))
    profile = folder / f"{variant:016x}.mesh57-state"
    blob = header + struct.pack("<3I", 0, 0, 64) + b"".join(parts)
    profile.write_bytes(blob)
    layout = pixel_entry(request, archive, profile)
    assert layout == "meshps1 0 0 64"
    request.with_suffix(".pslayout").write_text(layout)
    out = folder / "checked-pixel.dxil"
    subprocess.run([compiler, sdk, folder / "cache", "gfx1013-raw-mesh-ps", request, out], check=True)
    assert out.read_bytes()[:4] == b"DXBC"
    # The v6 profile has the same exact-span requirement and enables the captured guest root.
    header[:8] = b"X2WM576\0"
    extra = struct.pack("<8I", 12, 4, 1, 1, 16, 0, 0, 47)
    profile.write_bytes(header + struct.pack("<3I", 0, 0, 64) + extra + b"".join(parts))
    assert pixel_entry(request, archive, profile) == "meshps1 0 0 64 guest_root"
    # The first 24 stage bytes and profile filename still match, but the hardware variant differs.
    blob[-1] ^= 1
    profile.write_bytes(blob)
    try:
        pixel_entry(request, archive, profile)
    except ValueError as error:
        assert "bytes differ" in str(error)
    else:
        raise AssertionError("accepted another hardware variant")


def main():
    compiler, sdk = map(Path, sys.argv[1:])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "startup"))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checked_pixel_entry(root, compiler, sdk)
        raw = root / 'fixture.m57'
        layout = raw.with_suffix('.mslayout')
        layout.write_text('mesh1 128 3 1 2 128 64')
        def run(name, words, expected=None):
            raw.write_bytes(struct.pack('<%dI' % len(words), *words))
            out = root / (name + '.dxil')
            result = subprocess.run([compiler, sdk, root/'cache', 'gfx1013-raw-ms', raw, out],
                                    capture_output=True, text=True)
            if expected is not None:
                assert result.returncode and expected in result.stderr and not out.exists(), (result.stdout,result.stderr)
                return ''
            assert result.returncode == 0, result.stderr
            return subprocess.check_output([sdk/'dxc.exe','-dumpbin',out], text=True)

        program = triangle()
        text = run('triangle',program)
        for marker in ('MeshOutputTopology=triangle','setMeshOutputCounts(i32 168, i32 3, i32 1)',
                       'storeVertexOutput.f32','emitIndices','flattenedThreadIdInGroup'):
            assert marker in text, marker
        # A per-vertex raw PARAM0.xy must have the same semantic/channel
        # indices as the corresponding fixed PS reads.
        varying = program[:13] + [0xf8000203,0x00000201] + program[13:]
        text = run('varyings',varying)
        assert 'TEXCOORD' in text and 'storeVertexOutput.f32(i32 171, i32 2' in text
        # Runtime root constants select an actual scalar diamond. Each path
        # defines s20, consumed by a vertex export after the join.
        rs = root / 'fixture.hlsl'
        rs.write_text('#define RS "RootConstants(num32BitConstants=1,b0),DescriptorTable(SRV(t0))"')
        subprocess.run([sdk/'dxc.exe','-T','rootsig_1_1','-E','RS','-Fo',raw.with_suffix('.root'),rs], check=True)
        layout.write_text('mesh1 128 3 1 2 128 64\nconstants 21 0 1\ntable 16 1')
        controlled = program.copy()
        controlled[7] = 0x7e020c14  # read joined s20
        # Clear the high address. Read the four-word V# at table byte
        # offset zero, then fetch a word from that SRV only on the else arm.
        words = [0xbe910380,0xf4080608,0xfa000000,
                 0xbf0c8015,0xbf840002,0xbe940381,0xbf820002,
                 0xf420050c,0xfa000000]
        text = run('root-branch',words+controlled)
        for marker in ('cbufferLoadLegacy','rawBufferLoad','phi i32','storeVertexOutput'):
            assert marker in text, marker
        typed = [0xbe9e0317,0xbe9f0380,          # copy the heap base; zero high address
                 0x8715ff15,31,0x8f158515,        # (root0[0] & 31) * descriptor size
                 0xf408080f,0x2a000000,           # load V# into s32:s35
                 0x7e020280,0xe0002000,0x80080a01] # v10 = formatted element v1 (index zero)
        exported = program.copy()
        exported[7] = 0x7e020d0a  # v_cvt_f32_u32 v1,v10
        layout.write_text('mesh1 128 3 1 2 128 64\nconstants 21 0 1\nresource_heap 23')
        run('unknown-heap-view',typed+exported,'selected GPU slot')
        with layout.open('a') as meta:
            meta.write('\nvalidate_uint_heap_reads')
        text=run('typed-heap',typed+exported)
        for marker in ('createHandleFromHeap', 'bufferLoad.i32', 'i32 261'):
            assert marker in text, marker
        overwritten = typed[:7] + [0xbea10380] + typed[7:]
        run('overwritten-heap-word',overwritten+exported,'overwritten mesh heap descriptor')
        # Three incoming scalar edges share one join. Distinct reaching
        # values of s20 must all be linked to the post-join mesh position.
        layout.write_text('mesh1 128 3 1 2 128 64\nconstants 21 0 1')
        nested = [0xbf0c8015,0xbf840006,0xbf0c8115,0xbf840002,
                  0xbe940381,0xbf820003,0xbe940382,0xbf820001,0xbe940383]
        text=run('multi-edge',nested+controlled)
        assert text.count('phi i32')>=2
        # Count active lanes in both halves of a guest wave using the
        # logical workgroup index. The unsigned root parameter is the bound.
        # Use source-built encodings for the group index and range test.
        masks=[0x9380ff03 | (22<<16),0x40018,0x8f008616 | (28<<16),
               0xd7650000,0x000038c1,0xd7660000,0x000200c1,
               0x7da80015,0xbea8047e,0xbefe04c1,0xbe941028]
        text=run('mesh-range-mask',masks+controlled)
        assert 'unaryBits.i32' in text
        # Two guest waves share their output counts in LDS. s34 and s35
        # are bounded by source masks on the dynamic root input. Lane 0
        # writes both LDS slots; wave 1 reads the same pair after rendezvous.
        count_setup=[0x9380ff03|(22<<16),0x40018,0xbe880381,0xbe890380,
                     0x87228315,0x87238115]
        def movv(v,s): return 0x7e000200|(v<<17)|s
        p_reg,v_reg=35,34
        allocate=[0xbf8a0000,0xbf008016,0xbf840010,0xbefe0408,0xbf880007,
                  movv(2,p_reg),movv(3,128),movv(4,v_reg),0xd8340000,0x0203,
                  0xd8340004,0x0403,0xbefe04c1,0xbfa3ffe3,0xbf8a0000,
                  0x8f088c00|p_reg,0x887c0008|(v_reg<<8),0xbf900009,0xbf82000d,
                  0xbf8a0000,0xbefe0408,0xbf880009,movv(6,128),0xd8d80000,
                  (7<<24)|6,0xd8d80004,(6<<24)|6,0xbf8cc17f,
                  0x7e000500|(p_reg<<17)|7,0xbf8cc07f,0x7e000500|(v_reg<<17)|6,0xbefe04c1]
        text=run('group-counts',count_setup+allocate+program[3:])
        assert '@meshCounts' in text and 'setMeshOutputCounts(i32 168, i32 %' in text
        bad_slot=allocate.copy()
        bad_slot[25]=0xd8d80008
        run('wrong-count-slot',count_setup+bad_slot+program[3:],'unsupported mesh instruction')
        overflow=count_setup.copy()
        overflow[-1]=0x87238315
        run('too-many-primitives',overflow+allocate+program[3:],'exceeds declared range')
        layout.write_text('mesh1 128 3 1 2 128 64\ntable 16 0')
        run('mistyped-root',words+controlled,'root parameter kind')
        layout.write_text('mesh1 128 3 1 2 128 64')
        # A second allocation request/invalid maxima and unknown descriptors never emit partial bytecode.
        run('second-allocation', program[:3]+[0xbf900009]+program[3:], 'requires one uniform')
        run('over-limit', [program[0],0x1004]+program[2:], 'exceeds declared')
        run('unknown-root', [0xf4080308,0xfa000b20]+program, 'verified guest root')

        # Lane zero of wave zero stores its group identifier, then both
        # waves rendezvous before the other wave reads the word.
        exchange = [0xbefe0481,0xbe88047e,0x870aff03,0x0f000000,0xbf850007,0x2c0a0a87,
                    0x7e000280,0xd8340008,0x500,0xbfa3ffe3,0xbf8a0000,0xbf820004,
                    0xbf8a0000,0x7e000280,0xd8d80008,0x05000000,0xbefe04c1,0x7e240505]
        reads_group = program.copy()
        reads_group[7] = 0x7e020c12  # v_cvt_f32_u32 v1,s18 uses the readfirstlane result
        assert 'call i32 @dx.op.groupId.i32' in run('group',exchange+reads_group)
        broken = exchange.copy()
        broken[12] = 0xbf800000  # missing rendezvous on the reading wave
        run('mismatched-barriers',broken+reads_group,'unsupported mesh instruction')
    print('mesh allocation, indexed outputs, group exchange and negative controls passed')


if __name__ == '__main__':
    main()
