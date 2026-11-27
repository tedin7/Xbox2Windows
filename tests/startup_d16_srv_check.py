"""End-to-end bound SRV format validation and UNORM16 mode dispatch for D16 image_load (#2023, item D).

Translates a gfx1013 pixel stage using level-zero image_load ... d16, checks whether the narrowed half equals
0x3bff (green = 1.0 when v4[15:0] == 0x3bff, 0.0 otherwise), widens v4 via v_cvt_f32_f16, verifies 'd16srv 0 exact'
and 'udsrc 3 d16unorm16 0' in the published manifest, and draws eight columns through d3d12_xs.dll:
  col 0: R16_UNORM (0xffef) + explicit R16_UNORM SRV -> d16unorm16=1 gives exact 0x3bff -> (255, 255, 0, 255)
  col 1: R32_FLOAT (0x3f7ff000) + default SRV        -> d16unorm16=0 gives 0x3c00       -> (255, 0, 0, 255)
  col 2: R16_TYPELESS (0xaaaa) + explicit R16_UNORM  -> d16unorm16=1 gives 0x3955       -> (170, 0, 0, 255)
  col 3: R16_TYPELESS + explicit R16_UINT SRV        -> refused, stays clear            -> (0, 0, 255, 255)
  col 4: R16_UINT + default SRV                      -> refused, stays clear            -> (0, 0, 255, 255)
  col 5: R16_SNORM + explicit R16_SNORM SRV          -> refused (unverified format)     -> (0, 0, 255, 255)
  col 6: null-resource SRV and unwritten slot        -> both refused, stays clear       -> (0, 0, 255, 255)
  col 7: rebind R16_UNORM (0xffef) + default SRV     -> d16unorm16=1 gives exact 0x3bff -> (255, 255, 0, 255)
Also checks that a malformed 'd16srv' manifest line is rejected by load_translation.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_image_load_d16_check import IMG_LOAD, encode_mimg
from startup_stream_check import container
from startup_stream_draw_check import gfx2hlsl, vertex


def d16_pixel_container():
    code = (
        bytes.fromhex(
            '04040cf4000000fa'  # s_load_dwordx8 s[16:23], s[8:9], 0x0
            '8002007e'          # v_mov_b32_e32 v0, 0
            '8002027e'          # v_mov_b32_e32 v1, 0
            'f202047e'          # v_mov_b32_e32 v2, 1.0
            '7fc08cbf'          # s_waitcnt lgkmcnt(0)
        )
        + encode_mimg(IMG_LOAD, dmask=0x1, vaddr=0, vdata=4, srsrc=16, d16=1)
        + bytes.fromhex(
            '703f8cbf'          # s_waitcnt vmcnt(0)
            'ff080a36ffff0000'  # v_and_b32_e32 v5, 0xffff, v4
            'ff0a847dff3b0000'  # v_cmp_eq_u32_e32 vcc_lo, 0x3bff, v5
            '80040a02'          # v_cndmask_b32_e32 v5, 0, v2, vcc_lo
            '0417087e'          # v_cvt_f32_f16_e32 v4, v4
            '0f1800f804050002'  # exp mrt0 v4, v5, v0, v2 done vm
            '000081bf'          # s_endpgm
        )
    )
    words = struct.unpack('<%dI' % (len(code) // 4), code)
    return container(0, words)


def check(directory, consumer, backend):
    cache = directory / 'cache'
    cache.mkdir()
    vs, ps, root, output = (directory / name for name in ('vs', 'ps', 'root', 'pixels'))
    trap_log = directory / 'trap.log'
    vs.write_bytes(vertex())
    ps.write_bytes(d16_pixel_container())

    env = dict(
        os.environ,
        X2W_D3D12_PERMISSIVE='1',
        X2W_D3D12_LIST_NOOP='1',
        X2W_D3D12_RENDER='1',
        X2W_D3D12_DEBUG='0',
        X2W_D3D12_DEVICE_NOOP='1',
        X2W_SHADER_CACHE=str(cache),
        X2W_TEST_D16_SRV='1',
        X2W_TRAP_LOG=str(trap_log),
    )
    args = [consumer, vs, ps, root]
    subprocess.run(args, env=env, check=True)
    subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', cache], env=env, check=True)

    manifest, = (path for path in cache.glob('*.txt') if path.name not in ('revision.txt', 'x2w_key_seed.txt'))
    manifest_text = manifest.read_text(encoding='ascii')
    manifest_lines = manifest_text.splitlines()
    assert 'udsrc 3 d16unorm16 0' in manifest_lines, manifest_text
    assert 'srv 0 tex2d table 0 0' in manifest_lines, manifest_text
    assert 'd16srv 0 exact' in manifest_lines, manifest_text

    result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end='')
    if result.returncode == 77:
        return 77
    if result.returncode:
        raise RuntimeError(f'consumer failed ({result.returncode})\n{result.stderr[-5000:]}')

    actual = output.read_bytes()
    rgba_at_x = (
        (255, 255, 0, 255),  # col 0: R16_UNORM 0xffef -> exact 0x3bff (d16unorm16=1)
        (255, 0, 0, 255),    # col 1: R32_FLOAT 0x3f7ff000 -> 0x3c00 (d16unorm16=0)
        (170, 0, 0, 255),    # col 2: R16_TYPELESS 0xaaaa + R16_UNORM -> 0x3955
        (0, 0, 255, 255),    # col 3: R16_TYPELESS + R16_UINT -> refused
        (0, 0, 255, 255),    # col 4: R16_UINT -> refused
        (0, 0, 255, 255),    # col 5: R16_SNORM -> refused
        (0, 0, 255, 255),    # col 6: null resource + unwritten slot -> refused
        (255, 255, 0, 255),  # col 7: R16_UNORM 0xffef default SRV -> exact 0x3bff
    )
    expected = bytes(c for _ in range(4) for pixel in rgba_at_x for c in pixel)
    if actual != expected:
        raise AssertionError(f'actual {actual.hex()} expected {expected.hex()}')

    log_text = trap_log.read_text(errors='replace')
    for expected_reason in ('(format 57)', '(format 58)', '(missing texture resource 0)', '(missing view token 0)'):
        assert f'srv 0: d16 exact refused {expected_reason}' in log_text, (expected_reason, log_text)

    # Negative control: a malformed d16srv manifest line must be rejected by load_translation (all draws skipped).
    bad_log = directory / 'bad_trap.log'
    bad_out = directory / 'bad_pixels'
    manifest.write_text(manifest_text.replace('d16srv 0 exact', 'd16srv 0 uint'), encoding='ascii')
    bad_env = dict(env, X2W_TRAP_LOG=str(bad_log))
    bad_run = subprocess.run([*args, backend, bad_out], env=bad_env, capture_output=True, text=True)
    assert bad_run.returncode == 0, bad_run.stderr
    assert bad_out.read_bytes() == bytes((0, 0, 255, 255)) * 32
    assert 'invalid d16srv manifest line' in bad_log.read_text(errors='replace')

    print(f'{backend}: exact UNORM16 (0xffef->0x3bff) vs R32_FLOAT (0x3c00) dispatched, 5 invalid SRVs refused, bad manifest rejected')
    return 0


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
