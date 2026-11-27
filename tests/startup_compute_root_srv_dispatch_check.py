"""Startup compute manifest consumer for `srv <i> buf root <param>` at slot-14 Dispatch (Refs #2618, #2274)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_compute_dispatch_texture_check import clean, lines, run as _run, translate
from startup_compute_global_load_check import (
    GLOBAL_LOAD_DWORD,
    encode_global,
    encode_mubuf,
    encode_smem,
    encode_sop1,
    encode_sop2,
    encode_sopp,
    encode_vop2,
    encode_vop3a,
)
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_gfx2hlsl_compute_check import (
    KEY,
    MASK,
    SENTINEL,
    STRIDE,
    TGID_X,
    TGID_Y,
    TGID_Z,
    WORDS,
    tidig,
)

IN = [(0x8b02a7f1 + i * 0x413b1) & MASK for i in range(WORDS)]
IN2 = [(0x3c6ef372 + i * 0x19660d) & MASK for i in range(WORDS)]
NEIGHBOR0 = [(0xdeadbeef + i) & MASK for i in range(64)]
NEIGHBOR2 = [(0xfeedface + i) & MASK for i in range(64)]
GROUP = (2, 3, 4)


def run(directory, consumer, env, mode):
    return _run(directory, consumer, env, mode, f"<{WORDS}I")


def no_debug_errors(result):
    errors = [line for line in lines(result, "x2w-d3d12-debug ") if " sev=0 " in line or " sev=1 " in line]
    assert not errors, errors[0]


def root_srv_blob(*, two_srvs=False, param2_type=3):
    """DXBC RTS0 (v1.1) container matching startup_stream_consumer.cpp root-SRV bindings:
    param 0 = 3 root constants at b0 (s4-s6; shader uses s4-s5),
    param 1 = descriptor table with 2 UAV slots at u0 (s7; output UAV at table byte 16),
    param 2 = root descriptor at register(t1/b1, space0) (s[8:9]; type 3=SRV, 2=CBV),
    optional param 3 = root SRV descriptor at register(t2, space0) (s[10:11]).
    """
    nparams = 4 if two_srvs else 3
    poff = 24
    coff = poff + 12 * nparams
    toff = coff + 12
    roff = toff + 8
    srv2_off = roff + 24
    srv3_off = srv2_off + 12
    soff = (srv3_off + 12) if two_srvs else (srv2_off + 12)
    params = [
        struct.pack("<3I", 1, 0, coff),
        struct.pack("<3I", 0, 0, toff),
        struct.pack("<3I", param2_type, 0, srv2_off),
    ]
    if two_srvs:
        params.append(struct.pack("<3I", 3, 0, srv3_off))
    payload = (
        struct.pack("<3I", 0, 0, 3)
        + struct.pack("<2I", 1, roff)
        + struct.pack("<6I", 1, 2, 0, 0, 0, 0xffffffff)
        + struct.pack("<3I", 1, 0, 0)
    )
    if two_srvs:
        payload += struct.pack("<3I", 2, 0, 0)
    rts0 = struct.pack("<6I", 2, nparams, poff, 0, soff, 0) + b"".join(params) + payload
    assert len(rts0) == soff
    part = b"RTS0" + struct.pack("<I", len(rts0)) + rts0
    return b"DXBC" + bytes(16) + struct.pack("<III", 1, 36 + len(part), 1) + struct.pack("<I", 36) + part


def make_program(*, two_srvs=False):
    """Build gfx1013 compute ISA loading from 1 or 2 64-bit root SRV pointers via global_load_dword."""
    # With 3 root constants (s4-s6), 1 table (s7), and 1 or 2 root SRVs (s[8:9], optional s[10:11]):
    # user SGPR count is 10 (1 root SRV) or 12 (2 root SRVs), followed by TGID_X, TGID_Y, TGID_Z.
    gx = 12 if two_srvs else 10
    gy, gz = gx + 1, gx + 2
    rsrc2 = ((gx << 1) | TGID_X | TGID_Y | TGID_Z | tidig(3))
    second_saddr = 10 if two_srvs else 8
    code = b"".join([
        encode_sop1(0x03, 16, 7),                                # s_mov_b32 s16, s7
        encode_sop1(0x03, 17, 0x80),                             # s_mov_b32 s17, 0
        encode_smem(0x02, 20, 16, imm=0x10),                     # s_load_dwordx4 s[20:23], s[16:17], 0x10
        encode_sopp(0x0c, 0xc07f),                               # s_waitcnt lgkmcnt(0)
        encode_vop3a(0x143, 3, 256 + 1, 0x80 + 2, 256 + 0),      # v_mad_u32_u24 v3, v1, 2, v0
        encode_vop3a(0x143, 3, 256 + 2, 0x80 + 6, 256 + 3),      # v_mad_u32_u24 v3, v2, 6, v3
        encode_sop2(0x26, 24, gy, 0x80 + 3),                     # s_mul_i32 s24, s<gy>, 3
        encode_sop2(0x00, 24, 24, gx),                           # s_add_u32 s24, s24, s<gx>
        encode_sop2(0x26, 25, gz, 0x80 + 6),                     # s_mul_i32 s25, s<gz>, 6
        encode_sop2(0x00, 24, 24, 25),                           # s_add_u32 s24, s24, s25
        encode_sop2(0x26, 24, 24, 0x80 + 24),                    # s_mul_i32 s24, s24, 24
        encode_vop2(0x25, 3, 24, 3),                             # v_add_nc_u32 v3, s24, v3
        encode_vop2(0x1a, 4, 0x80 + 2, 3),                       # v_lshlrev_b32 v4, 2, v3
        encode_global(GLOBAL_LOAD_DWORD, 5, 4, 8, offset=0),     # global_load_dword v5, v4, s[8:9]
        encode_global(GLOBAL_LOAD_DWORD, 7, 4, second_saddr, offset=128),  # global_load_dword v7, v4, s[...] offset:128
        encode_vop3a(0x169, 6, 256 + 3, 5, 0),                   # v_mul_lo_u32 v6, v3, s5
        encode_vop2(0x25, 6, 4, 6),                              # v_add_nc_u32 v6, s4, v6
        encode_sopp(0x0c, 0x3f70),                               # s_waitcnt vmcnt(0)
        encode_vop3a(0x169, 5, 256 + 5, 0x80 + 33, 0),           # v_mul_lo_u32 v5, v5, 33
        encode_vop2(0x25, 5, 256 + 5, 7),                        # v_add_nc_u32 v5, v5, v7
        encode_vop2(0x1d, 5, 256 + 5, 6),                        # v_xor_b32 v5, v5, v6
        encode_mubuf(0x1c, 5, 4, 20, offen=True),                # buffer_store_dword v5, v4, s[20:23], 0 offen
        encode_sopp(0x01, 0),                                    # s_endpgm
    ])
    return code, rsrc2


def reference(*, two_srvs=False, offset0=0, offset1=0, shift1=32, swap_srvs=False, leak_neighbor=False):
    src0 = (IN2 if swap_srvs else IN) + (NEIGHBOR2 if swap_srvs else NEIGHBOR0)
    src1 = (IN if swap_srvs else (IN2 if two_srvs else IN)) + (NEIGHBOR0 if swap_srvs else (NEIGHBOR2 if two_srvs else NEIGHBOR0))
    off0 = offset1 if swap_srvs else offset0
    off1 = offset0 if swap_srvs else (offset1 if two_srvs else offset0)
    bound0 = WORDS - off0
    bound1 = WORDS - off1
    out = [SENTINEL] * WORDS
    for i in range(12 * 24):
        v5 = src0[off0 + i] if (leak_neighbor or i < bound0) else 0
        idx1 = i + shift1
        v7 = src1[off1 + idx1] if (leak_neighbor or idx1 < bound1) else 0
        out[i] = (((v5 * 33 + v7) & MASK) ^ ((KEY + i * STRIDE) & MASK))
    return out


def setup_case(directory, *, two_srvs=False, param2_type=3, adapter="warp"):
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP="1", X2W_D3D12_DEBUG="1")
    if adapter == "hardware":
        env["X2W_TEST_DISPATCH_HARDWARE"] = "1"
    (directory / "compute_root").write_bytes(root_srv_blob(two_srvs=two_srvs, param2_type=param2_type))
    env["X2W_TEST_COMPUTE_ROOT"] = str(directory / "compute_root")
    code, rsrc2 = make_program(two_srvs=two_srvs)
    (directory / "program").write_bytes(code)
    (directory / "data").write_bytes(DATA)
    (directory / "state").write_bytes(state_a(GROUP, rsrc2))
    return cache, env


def check_single_root_srv(base, consumer, dxc, adapter):
    directory = base / "single"
    cache, env = setup_case(directory, two_srvs=False, adapter=adapter)

    # First run before translation publishes the request and leaves output sentinels untouched.
    got, result = run(directory, consumer, env, "dispatch-root-srv")
    assert got == [SENTINEL] * WORDS
    assert len(lines(result, "Dispatch(3, 2, 2) refused: no translation")) == 1, result.stderr[-3000:]
    key = int(lines(result, "x2w-draw compute request")[0].split()[3], 16)
    translate(cache, dxc, key)

    manifest_path = cache / f"{key:016x}.txt"
    hlsl_path = cache / f"{key:016x}.cs.hlsl"
    cso_path = cache / f"{key:016x}.cs.cso"
    manifest = manifest_path.read_text(encoding="utf-8")
    hlsl = hlsl_path.read_text(encoding="utf-8")
    assert "srv 0 buf root 2" in manifest.splitlines(), manifest
    assert "ByteAddressBuffer r0 : register(t0);" in hlsl, hlsl
    assert "r0.Load(a)" in hlsl and "r0.Load(a + 128u)" in hlsl, hlsl

    # Manifest parser refusal controls on bad `srv ... buf root ...` lines and descriptor-word udsrc on a root SRV.
    for bad_manifest, expected_reason in (
        (manifest.replace("srv 0 buf root 2", "srv 0 buf root -1"),
         "a resource that is not a raw table buffer, a 2D texture or a cube map"),
        (manifest.replace("srv 0 buf root 2", "srv 0 buf root 2 trailing"),
         "a resource that is not a raw table buffer, a 2D texture or a cube map"),
        (manifest.replace("srv 0 buf root 2", "srv 1 buf root 2"),
         "a resource that is not a raw table buffer, a 2D texture or a cube map"),
        (manifest.replace("ud 3\nudsrc 0 rootconst 0 0\nudsrc 1 rootconst 0 1\nudsrc 2 rootconst 0 2",
                          "ud 4\nudsrc 0 rootconst 0 0\nudsrc 1 rootconst 0 1\nudsrc 2 rootconst 0 2\nudsrc 3 stride 0"),
         "an entry value names a descriptor that is not a buffer input or output"),
    ):
        sub = base / f"bad-manifest-{ abs(hash(bad_manifest)) }"
        sub_cache, sub_env = setup_case(sub, two_srvs=False, adapter=adapter)
        (sub_cache / f"{key:016x}.txt").write_text(bad_manifest, encoding="utf-8")
        (sub_cache / f"{key:016x}.cs.cso").write_bytes(cso_path.read_bytes())
        got, result = run(sub, consumer, sub_env, "dispatch-root-srv")
        assert got == [SENTINEL] * WORDS
        assert lines(result, f"compute pipeline {key:016x} not translated: {expected_reason}"), result.stderr[-3000:]

    # Translated single-root-SRV dispatch (and re-binding the same compute root signature).
    expected0 = reference(two_srvs=False, offset0=0)
    for mode in ("dispatch-root-srv", "dispatch-root-srv-same-root"):
        got, result = run(directory, consumer, env, mode)
        assert len(lines(result, "Dispatch(3, 2, 2) translated")) == 1, result.stderr[-3000:]
        debug = clean(result)
        assert got == expected0, (mode, sum(a != b for a, b in zip(got, expected0)))
    print(f"{adapter}: single root SRV (srv 0 buf root 2) and same-root rebind verified; debug: {debug}", flush=True)

    # Non-zero byte offset into placed buffer (+16 bytes = 4 dwords) and upload allocation (+16 bytes = 4 dwords),
    # plus out-of-bounds offset past the bounded backing returning 0 rather than reading neighbor 0xdeadbeef words.
    expected_off4 = reference(two_srvs=False, offset0=4)
    leaked_off4 = reference(two_srvs=False, offset0=4, leak_neighbor=True)
    assert sum(a != b for a, b in zip(expected_off4, leaked_off4)) == 4
    got, result = run(directory, consumer, env, "dispatch-root-srv-offset")
    assert len(lines(result, "Dispatch(3, 2, 2) translated")) == 1, result.stderr[-3000:]
    clean(result)
    assert got == expected_off4 and got != leaked_off4

    got, result = run(directory, consumer, env, "dispatch-root-srv-upload")
    assert len(lines(result, "Dispatch(3, 2, 2) translated")) == 1, result.stderr[-3000:]
    clean(result)
    assert got == expected_off4 and got != leaked_off4
    print(f"{adapter}: single root SRV non-zero offset (+16B placed, +16B upload) and bounded OOB=0 verified", flush=True)

    # Refusal controls leaving output sentinels untouched and logging the exact refusal reason.
    for mode, reason in (
        ("dispatch-root-srv-unset", "a compute root SRV was not set since the compute root"),
        ("dispatch-root-srv-reset", "a compute root SRV was not set since the compute root"),
        ("dispatch-root-srv-root-changed", "a compute root SRV was not set since the compute root"),
        ("dispatch-root-srv-null", "a compute root SRV has a null address"),
        ("dispatch-root-srv-unaligned", "a compute root SRV address is not 4-byte aligned"),
        ("dispatch-root-srv-unaligned-16", "a compute root SRV range is not 16-byte aligned in its resource"),
        ("dispatch-root-srv-unmapped", "a compute root SRV is in title memory outside any placed buffer"),
        ("dispatch-root-srv-copied", "a compute root SRV is in a placed buffer that does not alias title memory"),
        ("dispatch-root-srv-same-buffer", "an output buffer is also an input"),
    ):
        got, result = run(directory, consumer, env, mode)
        assert got == [SENTINEL] * WORDS, mode
        refusal = lines(result, "Dispatch(3, 2, 2) refused:")
        assert len(refusal) == 1 and refusal[0].endswith(reason), (mode, refusal, result.stderr[-3000:])
        assert not lines(result, "Dispatch(3, 2, 2) translated"), mode
        no_debug_errors(result)
    print(f"{adapter}: single root SRV refusal controls (unset, reset, root-changed, unaligned, unmapped, copied, UAV alias) verified", flush=True)

    # Wrong root parameter kind refusal control: param 2 declared as root CBV (type 2) instead of root SRV (type 3).
    cbv_dir = base / "wrong-kind-cbv"
    cbv_cache, cbv_env = setup_case(cbv_dir, two_srvs=False, param2_type=2, adapter=adapter)
    got, result = run(cbv_dir, consumer, cbv_env, "dispatch-root-srv")
    assert got == [SENTINEL] * WORDS
    cbv_key = int(lines(result, "x2w-draw compute request")[0].split()[3], 16)
    translate(cbv_cache, dxc, cbv_key)
    assert "srv 0 buf root 2" in (cbv_cache / f"{cbv_key:016x}.txt").read_text(encoding="utf-8").splitlines()
    got, result = run(cbv_dir, consumer, cbv_env, "dispatch-root-srv")
    assert got == [SENTINEL] * WORDS
    refusal = lines(result, "Dispatch(3, 2, 2) refused:")
    assert len(refusal) == 1 and refusal[0].endswith("a compute root input is not a shader resource view parameter"), (
        refusal,
        result.stderr[-3000:],
    )
    no_debug_errors(result)
    print(f"{adapter}: wrong root parameter kind (root CBV at param 2) refused cleanly", flush=True)


def check_two_root_srvs(base, consumer, dxc, adapter):
    directory = base / "two"
    cache, env = setup_case(directory, two_srvs=True, adapter=adapter)

    got, result = run(directory, consumer, env, "dispatch-root-srv")
    assert got == [SENTINEL] * WORDS
    assert len(lines(result, "Dispatch(3, 2, 2) refused: no translation")) == 1, result.stderr[-3000:]
    key = int(lines(result, "x2w-draw compute request")[0].split()[3], 16)
    translate(cache, dxc, key)

    manifest_path = cache / f"{key:016x}.txt"
    hlsl_path = cache / f"{key:016x}.cs.hlsl"
    cso_path = cache / f"{key:016x}.cs.cso"
    manifest = manifest_path.read_text(encoding="utf-8")
    hlsl = hlsl_path.read_text(encoding="utf-8")
    assert "srv 0 buf root 2" in manifest.splitlines() and "srv 1 buf root 3" in manifest.splitlines(), manifest
    assert "ByteAddressBuffer r0 : register(t0);" in hlsl and "ByteAddressBuffer r1 : register(t1);" in hlsl, hlsl
    assert "r0.Load(a)" in hlsl and "r1.Load(a + 128u)" in hlsl, hlsl

    expected_two = reference(two_srvs=True, offset0=0, offset1=0)
    assert expected_two != reference(two_srvs=False, offset0=0)
    got, result = run(directory, consumer, env, "dispatch-root-srv")
    assert len(lines(result, "Dispatch(3, 2, 2) translated")) == 1, result.stderr[-3000:]
    clean(result)
    assert got == expected_two
    print(f"{adapter}: two root SRVs (srv 0 buf root 2, srv 1 buf root 3) verified", flush=True)

    # Non-zero offsets into both placed buffers (+16B = 4 dwords on root SRV 0, +32B = 8 dwords on root SRV 1) + OOB=0 on root SRV 1.
    expected_off = reference(two_srvs=True, offset0=4, offset1=8)
    leaked_off = reference(two_srvs=True, offset0=4, offset1=8, leak_neighbor=True)
    assert sum(a != b for a, b in zip(expected_off, leaked_off)) == 8
    got, result = run(directory, consumer, env, "dispatch-root-srv-offset")
    assert len(lines(result, "Dispatch(3, 2, 2) translated")) == 1, result.stderr[-3000:]
    clean(result)
    assert got == expected_off and got != leaked_off
    print(f"{adapter}: two root SRVs non-zero offsets (+16B/+32B) and bounded OOB=0 (8 OOB lanes) verified", flush=True)

    # Second root SRV unset while first root SRV is set must refuse cleanly.
    got, result = run(directory, consumer, env, "dispatch-root-srv-unset-second")
    assert got == [SENTINEL] * WORDS
    refusal = lines(result, "Dispatch(3, 2, 2) refused:")
    assert len(refusal) == 1 and refusal[0].endswith("a compute root SRV was not set since the compute root"), (
        refusal,
        result.stderr[-3000:],
    )
    no_debug_errors(result)

    # Negative control 1: skipping the Dispatch leaves all 320 output words at SENTINEL (288 mismatches vs expected).
    got, result = run(directory, consumer, env, "dispatch-root-srv-none")
    assert not lines(result, "Dispatch(3, 2, 2) translated")
    assert got == [SENTINEL] * WORDS and sum(a != b for a, b in zip(got, expected_two)) == 288

    # Negative control 2: mutating the root SRV binding in the manifest (swapping root 2 and root 3) changes all 288 active words.
    swapped_manifest = manifest.replace("srv 0 buf root 2\nsrv 1 buf root 3", "srv 0 buf root 3\nsrv 1 buf root 2")
    assert swapped_manifest != manifest
    manifest_path.write_text(swapped_manifest, encoding="utf-8")
    expected_swapped = reference(two_srvs=True, swap_srvs=True)
    assert sum(a != b for a, b in zip(expected_swapped, expected_two)) == 288
    got, result = run(directory, consumer, env, "dispatch-root-srv")
    assert len(lines(result, "Dispatch(3, 2, 2) translated")) == 1, result.stderr[-3000:]
    clean(result)
    assert got == expected_swapped and sum(a != b for a, b in zip(got, expected_two)) == 288
    manifest_path.write_text(manifest, encoding="utf-8")

    # Negative control 3: mutating the translated shader (dropping the +128u byte offset on r1.Load) changes all 288 active words.
    mutated_hlsl = hlsl.replace("r1.Load(a + 128u)", "r1.Load(a)")
    assert mutated_hlsl != hlsl
    hlsl_path.write_text(mutated_hlsl, encoding="utf-8")
    subprocess.run([dxc, "-Gis", "-T", "cs_6_0", "-E", "main", "-Fo", cso_path, hlsl_path], check=True)
    expected_mutated = reference(two_srvs=True, shift1=0)
    assert sum(a != b for a, b in zip(expected_mutated, expected_two)) == 288
    got, result = run(directory, consumer, env, "dispatch-root-srv")
    assert len(lines(result, "Dispatch(3, 2, 2) translated")) == 1, result.stderr[-3000:]
    clean(result)
    assert got == expected_mutated and sum(a != b for a, b in zip(got, expected_two)) == 288
    print(f"{adapter}: negative controls (skip Dispatch, swap root SRV bindings, mutate HLSL offset) verified", flush=True)


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        check_single_root_srv(base, consumer, dxc, adapter)
        check_two_root_srvs(base, consumer, dxc, adapter)
    print(f"startup_compute_root_srv_dispatch: PASS on {adapter}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in ("warp", "hardware"):
        sys.exit(__doc__)
    main(*sys.argv[1:4])
