"""Exercise the opt-in gfx2hlsl diagnostic cache with original synthetic shaders."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import startup_gfx2hlsl_wave_reduce_check as wave
import tester_pack
from compute_load_format_check import check_references
from startup_gfx2hlsl_compute_check import G, SENTINEL, USER7, WORDS, root_blob

SWITCH = 'X2W_GFX_APPROX'
READ_REFUSAL = 'v_readfirstlane_b32 requires a proven uniform source'
REDUCE_REFUSAL = 'wave reduction: EXEC is not proven to be the entry mask on every incoming path'
VERTEX = (
    wave.vop1_mov(2, 128) + wave.vop1_mov(3, 242)  # v2 = 0, v3 = 1.0
    + struct.pack('<II', 0xf80008cf, 0x03020202)   # exp pos0 v2, v2, v2, v3 done
    + wave.sopp(1)
)
# SHA-256 of the accepted artifacts on the parent revision, with LF newlines.
BASELINE_KEYS = {'cs': 0x821d7ecb3f3903cf, 'ps': 0x666697888b30f325}
BASELINE = {
    'cs': (
        'ab95d049722dac0fd4e756edd7534958c4b106c473689b640e4b87fd89fb89b5',
        '2b2771c7af0213bcfed106b168bb049c425e5ab9e4ebe9bd0c26be168dec4427',
    ),
    'ps': (
        '219eeef2550351ed4b2429d416297b106ac8b12b8401148e3555249a5facaf23',
        '14c2cfa3b9b35a80575b862d1a4e474901110171070f453510f0b2d17f3d0973',
        'fe949a2b68feb861d1fd0e3fe537527397579a2c9e8cb42d22de6fadcf00c6d5',
    ),
}


def compute_program(proven=False):
    # Eight threads read eight separate words. The WARP fixture fills all eight
    # with the same value, which the translator cannot prove from the code.
    return bytes.fromhex(
        '06038abe 80038bbe'          # s_mov_b32 s10, s6; s_mov_b32 s11, 0
        '05030cf4000000fa 7fc08cbf'  # s_load_dwordx8 s[12:19], s[10:11], 0; s_waitcnt lgkmcnt(0)
        '82000834'                  # v_lshlrev_b32 v4, 2, v0
        '001030e004050380 703f8cbf'  # buffer_load_dword v5, v4, s[12:15], 0 offen; s_waitcnt vmcnt(0)
        + ('a5020a7e' if proven else '')  # v_mov_b32 v5, 37
        + '05052c7e'                # v_readfirstlane_b32 s22, v5
        '16020e7e'                  # v_mov_b32 v7, s22
        '001070e004070480 000081bf'  # buffer_store_dword v7, v4, s[16:19], 0 offen; s_endpgm
    )


def permissive_program(body):
    return bytes.fromhex(
        '06038abe 80038bbe'          # s_mov_b32 s10, s6; s_mov_b32 s11, 0
        '05030cf4000000fa 7fc08cbf'  # s_load_dwordx8 s[12:19], s[10:11], 0; s_waitcnt lgkmcnt(0)
        '83000834 91000e3a'          # v_lshlrev_b32 v4, 3, v0; v_xor_b32 v7, 17, v0
        + body
        + '001074e004070480 000081bf'  # buffer_store_dwordx2 v[7:8], v4, s[16:19], 0 offen; s_endpgm
    )


UNKNOWN_PROGRAM = permissive_program(
    'b502107e 007f107e'  # v_mov_b32 v8, 53; v_frexp_exp_i32_f32 v8, v0
)
STORE_PROGRAM = permissive_program(
    '8d02107e 021070e004070480'  # v_mov_b32 v8, 13; buffer_store_dword v7, v4, s[16:19], 0 offen offset:2
)
EXECZ_PROGRAM = permissive_program(
    '8b02107e 010088bf 000092bf'  # v_mov_b32 v8, 11; s_cbranch_execz over s_trap 0
)


def pixel_program(proven=False):
    # Reach the reduction by jumping back from a block that narrows EXEC.
    init = [('v_mov_b32_e32 v1, v0', wave.vop1_mov(1, 256))]
    restore = [('s_or_b32 exec_lo, exec_lo, s6', wave.sop2(0x10, wave.EXEC_LO, wave.EXEC_LO, 6))]
    reduce = wave.chain('v_max_u32') + (restore if not proven else []) + [wave.move(4, 1)] + wave.export('v4', 'v1', 'v4')
    body = [('v_cmp_lt_u32_e32 vcc_lo, 5, v0', struct.pack('<I', 0x7d820085)),
            ('s_and_saveexec_b32 s6, vcc_lo', wave.sop1(wave.S_AND_SAVEEXEC_B32, 6, 106))]
    if proven:
        body += restore
    reduce_words = sum(len(data) for _, data in reduce) // 4
    return wave.program(init, [('s_branch dispatch', wave.sopp(2, reduce_words))], reduce, body,
                        [('s_branch reduce', wave.sopp(2, -(reduce_words + len(body) + 1)))])[1]


def compute_request(directory, code):
    root = root_blob()
    state = struct.pack('<7I', 0, 0, 0, (1 << 16) | 8, 1, 0, USER7)
    desc = struct.pack('<12Q', 0, 0, len(state), 0, 0, 0, len(code), 0, 0, 0, 0, 0)
    key = G.compute_key(code, b'', root, desc, state)
    for suffix, blob in zip(('c67', 'data', 'root', 'desc', 'sa', 'sb'), (code, b'', root, desc, state, b'')):
        (directory / f'req_{key:016x}.{suffix}').write_bytes(blob)
    return key


def compute(directory, proven=False):
    key = compute_request(directory, compute_program(proven))
    return key, G.translate_request(directory, key)


def pixel_request(directory, code):
    root = root_blob()
    key = G.pipeline_key(VERTEX, code, root, ps_inputs=wave.PACKED)
    ps_inputs = f'psin1 {key:016x} ena=0x8000 addr=0x8000\n'.encode('ascii')
    for suffix, blob in (('a', VERTEX), ('b', code), ('root', root), ('psin', ps_inputs)):
        (directory / f'req_{key:016x}.{suffix}').write_bytes(blob)
    return key


def pixel(directory, proven=False):
    key = pixel_request(directory, pixel_program(proven))
    return key, G.translate_request(directory, key)


def artifact(directory, key, suffix):
    return directory / f'{key:016x}.{suffix}'


def check_approx(directory, key, stage, reads, reductions):
    derived = G.fnv(b'approx1', key)
    assert derived != key
    marker = f'x2w-gfx approx: key {derived:016x} stage {stage} lifted {reads} readfirstlane, {reductions} reductions'
    assert artifact(directory, key, 'approx.txt').read_text().splitlines() == [marker]
    assert artifact(directory, derived, 'txt').read_text().splitlines().count('# ' + marker) == 1
    return artifact(directory, derived, f'{stage}.hlsl')


def check_approx2(directory, key, stage, lifted):
    derived = G.fnv(b'approx2', key)
    marker = f'x2w-gfx approx2: key {derived:016x} stage {stage} lifted {lifted}'
    assert derived != key and G.APPROX2_MARKER.fullmatch(marker)
    assert artifact(directory, key, 'approx2.txt').read_text().splitlines() == [marker]
    assert '# ' + marker in artifact(directory, derived, 'txt').read_text().splitlines()
    assert not artifact(directory, key, 'approx.txt').exists()
    return marker, artifact(directory, derived, stage + '.hlsl')


def controls(directory):
    for translate, refusal, stage, reads, reductions in (
            (compute, READ_REFUSAL, 'cs', 1, 0), (pixel, REDUCE_REFUSAL, 'ps', 0, 1)):
        with patch.dict(os.environ, {'X2W_FXC': ''}):
            os.environ.pop(SWITCH, None)
            key, result = translate(directory)
            assert result == 'unsupported: ' + refusal, ('switch unset must preserve strict behavior', stage, result)
            os.environ[SWITCH] = '0'
            assert translate(directory) == (key, 'unsupported: ' + refusal)
            assert artifact(directory, key, 'fail').read_text().splitlines()[0] == refusal
            assert not artifact(directory, key, 'approx.txt').exists()
            strict, result = translate(directory, proven=True)
            assert (strict, result) == (BASELINE_KEYS[stage], 'ok'), (stage, strict, result)
            suffixes = ('cs.hlsl', 'txt') if stage == 'cs' else ('vs.hlsl', 'ps.hlsl', 'txt')
            baseline = [artifact(directory, strict, suffix).read_bytes() for suffix in suffixes]
            hashes = tuple(hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() for data in baseline)
            assert hashes == BASELINE[stage], (stage, hashes)
            for value in ('0', '1', '2'):
                os.environ[SWITCH] = value
                assert translate(directory, proven=True) == (strict, 'ok')
                assert baseline == [artifact(directory, strict, suffix).read_bytes() for suffix in suffixes]
                assert not artifact(directory, strict, 'approx.txt').exists()
                assert not artifact(directory, strict, 'approx2.txt').exists()
            os.environ[SWITCH] = '1'
            assert key in G.pending_keys(directory, skip_failed=True)
            result = translate(directory)
            assert result == (key, 'ok'), result
            hlsl = check_approx(directory, key, stage, reads, reductions).read_text()
            if reads:
                assert 's22 = v5;' in hlsl and 'WaveReadLaneFirst' not in hlsl
            else:
                assert 'WaveActiveMax(v1)' in hlsl
            assert not artifact(directory, key, 'txt').exists()
            assert key not in G.pending_keys(directory, skip_failed=True)
            assert next(row for row in tester_pack.cache_results(directory) if row[0] == f'{key:016x}')[2] == 'approximated'
            os.environ[SWITCH] = '0'
            assert G.translate_request(directory, key) == 'unsupported: ' + refusal
            check_approx(directory, key, stage, reads, reductions)
            os.environ[SWITCH] = '1'
            assert key not in G.pending_keys(directory, skip_failed=True)

    markers = [row[3] for row in tester_pack.cache_results(directory) if row[2] == 'approximated']
    (directory / 'stderr.txt').write_text('\n'.join(markers * 2) + '\n')
    report = tester_pack.format_results(tester_pack.cache_results(directory), loaded=tester_pack.loaded_approximations(directory))
    assert 'approximated shaders loaded 2' in report
    assert all(report.splitlines().count(marker) == 1 for marker in markers)

    with patch.dict(os.environ, {SWITCH: '1', 'X2W_FXC': ''}):
        # Leave an unsupported instruction after the lifted read.
        code = compute_program()[:-4] + bytes.fromhex('000092bf 000081bf')  # s_trap 0; s_endpgm
        key = compute_request(directory, code)
        with patch.dict(os.environ, {SWITCH: '0'}):
            assert G.translate_request(directory, key) == 'unsupported: ' + READ_REFUSAL
        assert G.translate_request(directory, key) == 'unsupported: s_trap'
        assert not artifact(directory, key, 'approx.txt').exists()


def permissive_controls(directory):
    fixtures = (
        (UNKNOWN_PROGRAM, 'v_frexp_exp_i32_f32', 'unknown_op=1'),
        (STORE_PROGRAM, 'buffer store at unaligned immediate offset 2', 'store_dropped=1'),
        (EXECZ_PROGRAM, 's_trap inside a block s_cbranch_execz skips per lane: effect outside scalar registers',
         'execz=1'),
    )
    for reason, instruction, kind in (
        (READ_REFUSAL, 'v_readfirstlane_b32', 'readfirstlane'),
        (REDUCE_REFUSAL, None, 'reduction'),
        ('s_nand_b32 on a lane mask', 's_nand_b32', 'uniform'),
        ('branch out of the block', None, 'execz'),
        ('s_load_dwordx4 with a register offset', 's_load_dwordx4', 'resource'),
        ('slot 57 mesh: a loop outside the group-memory exchanges is not covered', None, 'mesh'),
        ('unsupported memory operation', 'ds_max_i32', 'store_dropped'),
        ('v_frexp_exp_i32_f32', 'v_frexp_exp_i32_f32_e32', 'unknown_op'),
        ('unreadable shader request', None, 'other'),
    ):
        assert G.approx2_class(reason, instruction) == kind, reason

    regs = G.approx2_destinations('ds_max_rtn_i32', 'ds_max_rtn_i32 v8, v4, v7')
    assert tuple(map(tuple, regs)) == ((), (8,))
    regs = G.approx2_destinations('buffer_atomic_add', 'buffer_atomic_add v8, v4, s[16:19], 0 offen')
    assert tuple(map(tuple, regs)) == ((), ())
    regs = G.approx2_destinations('image_atomic_cmpswap',
                                  'image_atomic_cmpswap v[8:11], v4, s[16:23] dmask:0xf glc')
    assert tuple(map(tuple, regs)) == ((), (8, 9))

    markers = []
    with patch.dict(os.environ, {'X2W_FXC': '', 'X2W_PLACEHOLDER_PS': '0', 'X2W_PRE_RUN_FAMILY_RETRY': '0'}):
        for code, reason, lifted in fixtures:
            key = compute_request(directory, code)
            os.environ.pop(SWITCH, None)
            for value in (None, '0', '1', '02'):
                if value is not None:
                    os.environ[SWITCH] = value
                result = G.translate_request(directory, key)
                assert result == 'unsupported: ' + reason, (value, result)
                assert artifact(directory, key, 'fail').read_text().splitlines()[0] == reason

            os.environ[SWITCH] = '2'
            assert key in G.pending_keys(directory, skip_failed=True)
            stderr = io.StringIO()
            write_alias = G.atomic_retry_text

            def after_manifest(path, data):
                if path == str(artifact(directory, key, 'approx2.txt')):
                    derived_manifest = artifact(directory, G.fnv(b'approx2', key), 'txt')
                    assert '# ' + data.strip() in derived_manifest.read_text().splitlines()
                write_alias(path, data)

            with patch.object(G, 'atomic_retry_text', side_effect=after_manifest), redirect_stderr(stderr):
                assert G.translate_request(directory, key) == 'ok'
            marker, source = check_approx2(directory, key, 'cs', lifted)
            assert stderr.getvalue().splitlines() == [marker]
            markers.append(marker)
            assert key not in G.pending_keys(directory, skip_failed=True)
            assert (f'{key:016x}', 'compute', 'approximated', marker) in tester_pack.cache_results(directory)
            hlsl = source.read_text()
            if lifted == 'unknown_op=1':
                assert 'v8 = 0u;' in hlsl
            elif lifted == 'store_dropped=1':
                assert '.Store(' not in hlsl and hlsl.count('.Store2(') == 1 and 'v7 = 0u;' not in hlsl

        key = compute_request(directory, UNKNOWN_PROGRAM)
        original = fixtures[0][1]
        alias = artifact(directory, key, 'approx2.txt')
        # Simulate a newer request publication even on filesystems with coarse mtimes.
        stamp = alias.stat().st_mtime - 2
        os.utime(alias, (stamp, stamp))
        assert key in G.pending_keys(directory, skip_failed=True)
        derived = G.fnv(b'approx2', key)
        artifact(directory, derived, 'txt').unlink()
        artifact(directory, derived, 'cs.hlsl').unlink()
        with patch.object(G, 'permissive_translation', side_effect=RuntimeError('injected failure')):
            assert G.translate_request(directory, key) == 'unsupported: ' + original
        assert artifact(directory, key, 'fail').read_text().splitlines() == [original, G.APPROX2_ATTEMPT]
        assert not alias.exists() and not artifact(directory, derived, 'txt').exists()
        assert not artifact(directory, derived, 'cs.hlsl').exists()
        assert key not in G.pending_keys(directory, skip_failed=False)
        alias.write_text(markers[0] + '\n')
        assert (f'{key:016x}', 'compute', 'refused', original) in tester_pack.cache_results(directory)
        artifact(directory, key, 'fail').unlink()
        assert (f'{key:016x}', 'compute', 'pending', '') in tester_pack.cache_results(directory)
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            G.translate_pending(directory, skip_failed=True, jobs=1)
        assert stdout.getvalue().splitlines()[-1] == 'approximated (level 2) 1'

    (directory / 'stderr.txt').write_text('\n'.join(markers * 2) + '\n')
    assert set(tester_pack.loaded_approximations(directory).values()) == set(markers)


def permissive_lifts_and_placeholder(directory):
    with patch.dict(os.environ, {SWITCH: '2', 'X2W_FXC': '', 'X2W_PLACEHOLDER_PS': '0'}):
        for translate, stage, lifted, expression in (
            (compute, 'cs', 'readfirstlane=1', 's22 = v5;'),
            (pixel, 'ps', 'reduction=1', 'WaveActiveMax(v1)'),
        ):
            key, status = translate(directory)
            assert status == 'ok', status
            _, source = check_approx2(directory, key, stage, lifted)
            assert expression in source.read_text()
        code = compute_program()[:-4] + bytes.fromhex('007f107e 000081bf')
        key = compute_request(directory, code)
        assert G.translate_request(directory, key) == 'ok'
        check_approx2(directory, key, 'cs', 'readfirstlane=1 unknown_op=1')

        code = wave.program([('v_mov_b32_e32 v1, v0', wave.vop1_mov(1, 256))], wave.WQM,
                            wave.chain('v_max_u32'), [wave.move(4, 1)], wave.export('v4', 'v1', 'v4'))[1]
        key = pixel_request(directory, code)
        with patch.dict(os.environ, {SWITCH: '1'}):
            assert 'helper lanes' in G.translate_request(directory, key)
        assert G.translate_request(directory, key) == 'ok'
        _, source = check_approx2(directory, key, 'ps', 'reduction=1')
        assert 'WaveActiveMax(v1)' in source.read_text()

    with patch.dict(os.environ, {SWITCH: '0', 'X2W_FXC': '', 'X2W_PLACEHOLDER_PS': '1'}):
        code = bytes.fromhex('007f107e') + b''.join(data for _, data in wave.export('v8', 'v8', 'v8'))
        key = pixel_request(directory, code)
        assert G.translate_request(directory, key) == 'ok'
        exact = artifact(directory, key, 'txt')
        fallback = exact.read_bytes()
        assert b'placeholder rgb ' in fallback
        fail = artifact(directory, key, 'fail')
        reason = fail.read_text().splitlines()[0]

        os.environ[SWITCH] = '2'
        assert key in G.pending_keys(directory, skip_failed=True)
        with patch.object(G, 'permissive_translation', side_effect=RuntimeError('injected failure')):
            assert G.translate_request(directory, key) == 'ok'
        assert fallback == exact.read_bytes()
        assert fail.read_text().splitlines()[0] == reason
        assert G.APPROX2_ATTEMPT in fail.read_text().splitlines()
        assert not artifact(directory, key, 'approx2.txt').exists()
        assert key not in G.pending_keys(directory, skip_failed=False)

        os.environ[SWITCH] = '0'
        assert G.translate_request(directory, key) == 'ok'
        os.environ[SWITCH] = '2'
        assert key in G.pending_keys(directory, skip_failed=True)
        assert G.translate_request(directory, key) == 'ok'
        check_approx2(directory, key, 'ps', 'unknown_op=1')
        assert fallback == exact.read_bytes()
        assert key not in G.pending_keys(directory, skip_failed=True)


def permissive_dispatch(directory, dxc, runner, adapter):
    with patch.dict(os.environ, {SWITCH: '2', 'X2W_FXC': ''}):
        key = compute_request(directory, UNKNOWN_PROGRAM)
        assert G.translate_request(directory, key) == 'ok'
    _, src = check_approx2(directory, key, 'cs', 'unknown_op=1')
    shader = src.with_suffix('.cso')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
    expected = [word for tid in range(8) for word in (tid ^ 17, 0)] + [SENTINEL] * (WORDS - 16)
    words = [1, *expected, *([0] * 16), *([0] * WORDS)]
    case = directory / 'permissive-zero.case'
    case.write_bytes(struct.pack('<%dI' % len(words), *words))
    status = check_references('permissive zero', runner, adapter, shader, [(case, 'eight threads', None)],
                              allow_skip=True)
    if status == 77:
        return status
    hlsl = src.read_text()
    assert hlsl.count('v8 = 0u;') == 1
    broken = directory / 'permissive-nonzero.hlsl'
    broken.write_text(hlsl.replace('v8 = 0u;', 'v8 = 23u;'), encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, broken], check=True)
    return check_references('permissive nonzero control', runner, adapter, shader,
                            [(case, 'eight threads', 'word=1 expected=00000000 actual=00000017')])


def dispatch(directory, dxc, runner, adapter):
    # Scope the opt-in to translating this fixture inside the existing WARP suite.
    with patch.dict(os.environ, {SWITCH: '1', 'X2W_FXC': ''}):
        key, result = compute(directory)
    assert result == 'ok', result
    src = check_approx(directory, key, 'cs', 1, 0)
    shader = src.with_suffix('.cso')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, src], check=True)
    cases = []
    for value in (37, 0xa13f058b):
        data = [value] * 8 + [0] * (WORDS - 8)
        expected = data[:8] + [SENTINEL] * (WORDS - 8)
        words = [1, *expected, *([0] * 16), *data]
        case = directory / f'uniform-{value}.case'
        case.write_bytes(struct.pack('<%dI' % len(words), *words))
        cases.append((case, f'uniform input {value:08x}', None))
    status = check_references('approximate readfirstlane', runner, adapter, shader, cases, allow_skip=True)
    if status == 77:
        return status
    broken = directory / 'zero-firstlane.hlsl'
    hlsl = src.read_text()
    assert hlsl.count('s22 = v5;') == 1
    broken.write_text(hlsl.replace('s22 = v5;', 's22 = 0u;'), encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, broken], check=True)
    case, label, _ = cases[0]
    check_references('zeroed readfirstlane control', runner, adapter, shader,
                     [(case, label, 'word=0 expected=00000025 actual=00000000')])
    return permissive_dispatch(directory, dxc, runner, adapter)


if __name__ == '__main__':
    output = Path(__file__).resolve().parents[1] / 'out'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        directory = Path(tmp)
        controls(directory)
        permissive = directory / 'permissive'
        permissive.mkdir()
        permissive_controls(permissive)
        with redirect_stderr(io.StringIO()):
            permissive_lifts_and_placeholder(permissive)
    print('approximate translation, cache keys and strict controls passed')
