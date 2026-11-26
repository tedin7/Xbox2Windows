#!/usr/bin/env python3
"""v_readfirstlane_b32 of a per-lane value under a partial EXEC mask, read back against the guest rule.

The guest reads the VGPR of the lowest lane of its wave whose EXEC bit is set, and lane 0 when no lane is set (RDNA 2
ISA, VALU lane operations). Every case loads a per-lane word, reads the first lane into an SGPR, xors it into the lane's
own word and stores the result:
  sparse32  wave32, odd lanes enabled: the first lane is 1, lane 0 holds a stale word the guest does not read.
  none32    wave32, only lane 31 enabled for the read, then all lanes: the full wave reads lane 31, the 8-lane last
            wave has no enabled lane (bit 31 is no lane) and reads its own lane 0.
  sparse64  wave64, lanes 5.. enabled: the first lane is 5 in both the 64-lane and the 32-lane last wave.
  none64    wave64, only lane 40 enabled for the read, then all lanes: the full wave reads lane 40, the 32-lane
            last wave has no enabled lane and reads its own lane 0.
Controls rewrite the translated read to lane 0 of the wave and to the host wave's first lane; both must fail the readback.

  startup_engine_first_lane_sparse_check.py                              translation checks
  startup_engine_first_lane_sparse_check.py binary <llvm-mc>            recorded words equal the assembler's
  startup_engine_first_lane_sparse_check.py dispatch <dxc> <runner> <adapter>   readback, controls included
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import startup_engine_compute_check as E  # noqa: E402  (shares the case runner and the read-buffer fill)

PROLOGUE = 's_mov_b32 s10, s6\ns_mov_b32 s11, 0\ns_load_dwordx8 s[12:19], s[10:11], 0x0\ns_waitcnt lgkmcnt(0)\n'
BODY = ('v_lshlrev_b32 v4, 2, v0\nbuffer_load_dword v2, v4, s[12:15], 0 offen\ns_waitcnt vmcnt(0)\n'
        'v_readfirstlane_b32 s20, v2\n%sv_xor_b32 v3, s20, v2\nbuffer_store_dword v3, v4, s[16:19], 0 offen\ns_endpgm\n')

# name -> (EXEC set before the read, EXEC restored before the store, recorded words, group, guest wave size)
VARIANTS = {
    'sparse32': ('s_mov_b32 exec_lo, 0xaaaaaaaa\n', '',
                 '06038abe80038bbe05030cf4000000fa7fc08cbfff03febeaaaaaaaa82000834001030e004020380703f8cbf'
                 '0205287e1404063a001070e004030480000081bf', (40, 1, 1), 32),
    'none32': ('s_mov_b32 exec_lo, 0x80000000\n', 's_mov_b32 exec_lo, -1\n',
               '06038abe80038bbe05030cf4000000fa7fc08cbfff03febe0000008082000834001030e004020380703f8cbf'
               '0205287ec103febe1404063a001070e004030480000081bf', (40, 1, 1), 32),
    'sparse64': ('s_mov_b32 s26, 0xffffffe0\ns_mov_b32 s27, -1\ns_mov_b64 exec, s[26:27]\n', '',
                 '06038abe80038bbe05030cf4000000fa7fc08cbfff039abee0ffffffc1039bbe1a04febe82000834'
                 '001030e004020380703f8cbf0205287e1404063a001070e004030480000081bf', (96, 1, 1), 64),
    'none64': ('s_mov_b32 s26, 0\ns_mov_b32 s27, 0x100\ns_mov_b64 exec, s[26:27]\n', 's_mov_b64 exec, -1\n',
               '06038abe80038bbe05030cf4000000fa7fc08cbf80039abeff039bbe000100001a04febe82000834'
               '001030e004020380703f8cbf0205287ec104febe1404063a001070e004030480000081bf', (96, 1, 1), 64),
}
SIZES = {'sparse32': ('odd', 32), 'none32': ('top', 32), 'sparse64': ('from5', 64), 'none64': ('lane40', 64)}


def source(name):
    setup, restore = VARIANTS[name][:2]
    return PROLOGUE + setup + BODY % restore


def words(name):
    return bytes.fromhex(VARIANTS[name][2])


def translate(name):
    group = VARIANTS[name][3]
    return E.translate('N', code=words(name), group=group, wave=VARIANTS[name][4])


def expected(name):
    """The words the guest rule gives: lane l of wave w stores in[first(w)] ^ in[its own thread] when it is enabled
    at the store, SENTINEL when not."""
    mask_kind, wave = SIZES[name]
    threads = VARIANTS[name][3][0]
    out = [E.SENTINEL] * E.WORDS
    for thread in range(threads):
        w, lane = divmod(thread, wave)
        lanes = min(wave, threads - w * wave)          # lanes the wave holds
        if mask_kind == 'odd':
            enabled, store = [l for l in range(lanes) if l % 2], lane % 2 == 1
            first = enabled[0]
        elif mask_kind == 'from5':
            enabled, store = [l for l in range(lanes) if l >= 5], lane >= 5
            first = enabled[0]
        else:                                           # one lane for the read, the full wave for the store
            only = 31 if mask_kind == 'top' else 40
            enabled, store = [only] if only < lanes else [], True
            first = enabled[0] if enabled else 0
        if store:
            out[thread] = E.input_word(w * wave + first) ^ E.input_word(thread)
    return out


# control -> (variant, text the translation holds once, replacement)
def controls_for(name):
    wave = SIZES[name][1]
    guest = 'x2w_first_0 == 0xffffffffu ? x2w_val[x2w_snap_0][x2w_wave * %du] : x2w_value_0' % wave
    return {
        'lane-zero': (guest, 'x2w_val[x2w_snap_0][x2w_wave * %du]' % wave),
        'host-first': (guest, 'WaveReadLaneFirst(v2)'),
    }


def mutate(hlsl, old, new):
    assert hlsl.count(old) == 1, (old, hlsl.count(old))
    return hlsl.replace(old, new)


def check_controls():
    for name in VARIANTS:
        hlsl = translate(name).hlsl_cs()
        assert 'InterlockedMin(x2w_first[' in hlsl and 'x2w_val[x2w_snap_0]' in hlsl, hlsl
        assert hlsl.count('GroupMemoryBarrierWithGroupSync();') == 2, hlsl   # the prologue and the one gather
        for control, (old, new) in controls_for(name).items():
            assert mutate(hlsl, old, new) != hlsl, (name, control)
        # The expected words are not the lane-0 answer: every case tells the controls apart.
        assert len(set(expected(name))) > 3, name
    print('first-lane sparse translation and expectation checks passed')


def check_binary(llvm_mc):
    from startup_flat_param_check import assemble
    for name in VARIANTS:
        built = assemble(llvm_mc, source(name))
        assert built == words(name), '%s: assembled %s, recorded %s' % (name, built.hex(), words(name).hex())
    print('first-lane sparse fixture words match llvm-mc')


def run_case(name, dxc, runner, adapter, directory, control=None):
    hlsl = translate(name).hlsl_cs()
    if control:
        hlsl = mutate(hlsl, *controls_for(name)[control])
    src, cso, case = (directory / ('%s-%s.%s' % (name, control or 'plain', ext)) for ext in ('hlsl', 'cso', 'case'))
    src.write_text(hlsl, encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    constants = [E.KEY, E.STRIDE] + [0] * 14
    case.write_bytes(struct.pack('<I', 1 | 1 << 8 | 1 << 16) + struct.pack('<%dI' % E.WORDS, *expected(name))
                     + struct.pack('<16I', *constants))
    return subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)


def check_dispatch(dxc, runner, adapter):
    runner = str(Path(runner).resolve())
    out = HERE.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for name in VARIANTS:
            for control in (None, 'lane-zero', 'host-first'):
                run = run_case(name, dxc, runner, adapter, Path(tmp), control)
                print(run.stdout, end='')
                if run.returncode == 77:
                    return 77
                diagnostics = [line for line in run.stderr.splitlines()
                               if not (adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '))]
                if control:
                    if run.returncode != 1 or not any('expected=' in line for line in diagnostics):
                        raise AssertionError('control %s on %s: not rejected (exit %d) %s' % (control, name, run.returncode, run.stderr))
                    print('control %s on %s: rejected at %s' % (control, name, diagnostics[0]))
                elif run.returncode or diagnostics:
                    raise AssertionError('case %s: %s' % (name, run.stderr or 'exit %d' % run.returncode))
                else:
                    print('case %s: all %d words match' % (name, E.WORDS))
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_controls()
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        check_binary(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(check_dispatch(*sys.argv[2:]))
    else:
        sys.exit(__doc__)
