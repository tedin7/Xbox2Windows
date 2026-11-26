"""Source-built gfx1013 compute fixtures: one raw buffer descriptor that the program both loads and stores through.

  (no arguments)                                decode and translate, check the HLSL and the manifest (pure Python)
  --llvm-mc <llvm-mc>                           the same, with the fixtures assembled by llvm-mc
  --dispatch <dxc> <offscreen_check> <adapter>  compile the HLSL with DXC, run GPU readback against a CPU reference

The old translation refused such a program ("used for a load and a store") because a read-only view and a UAV of one
buffer cannot be bound in one dispatch. The loads now read the same RWByteAddressBuffer the stores write.

  load_then_store: out[i] = out[i] + 7        (the runner fills the output with the sentinel word)
  store_then_load: out[i] = i; out[i] = out[i] + 7   (a store followed by a load of the word it wrote)
"""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))

import startup_compute_readfirstlane_check as rf  # noqa: E402
from startup_gfx2hlsl_compute_check import G, KEY, MASK, SENTINEL, STRIDE, WORDS, assemble, refused, root_blob  # noqa: E402

PROLOGUE = """\
s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
s_lshl_b32 s27, s7, 2
v_add_nc_u32_e32 v4, s27, v0
v_mov_b32_e32 v6, v4
v_lshlrev_b32_e32 v4, 2, v4
"""
LOAD_THEN_STORE = PROLOGUE + """\
buffer_load_dword v5, v4, s[16:19], 0 offen
s_waitcnt vmcnt(0)
v_add_nc_u32_e32 v5, 7, v5
buffer_store_dword v5, v4, s[16:19], 0 offen
s_endpgm
"""
STORE_THEN_LOAD = PROLOGUE + """\
buffer_store_dword v6, v4, s[16:19], 0 offen
buffer_load_dword v5, v4, s[16:19], 0 offen
s_waitcnt vmcnt(0)
v_add_nc_u32_e32 v5, 7, v5
buffer_store_dword v5, v4, s[16:19], 0 offen
s_endpgm
"""
IDXEN_LOAD = LOAD_THEN_STORE.replace('buffer_load_dword v5, v4, s[16:19], 0 offen', 'buffer_load_dword v5, v6, s[16:19], 0 idxen')
THREADS = rf.LOCAL[0] * rf.GROUPS[0]
FIXTURES = {'load_then_store': LOAD_THEN_STORE, 'store_then_load': STORE_THEN_LOAD, 'idxen_load': IDXEN_LOAD}
# The assembled words, so the test needs no assembler; --llvm-mc reassembles and compares.
WORDS_HEX = {
    'load_then_store': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e82080834001030e004050480'
                       '703f8cbf870a0a4a001070e004050480000081bf',
    'store_then_load': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e82080834001070e004060480'
                       '001030e004050480703f8cbf870a0a4a001070e004050480000081bf',
    'idxen_load': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e82080834002030e006050480'
                  '703f8cbf870a0a4a001070e004050480000081bf',
}


def translate(name):
    return G.ComputePipeline(bytes.fromhex(WORDS_HEX[name]), root_blob(), rf.RSRC2, rf.LOCAL)


def expected(name):
    out = [SENTINEL] * WORDS
    for i in range(THREADS):
        out[i] = ((SENTINEL if name == 'load_then_store' else i) + 7) & MASK
    return out


def controls(llvm_mc=None):
    if llvm_mc:
        for name, source in FIXTURES.items():
            built = assemble(llvm_mc, source).hex()
            assert built == WORDS_HEX[name], '%s: assembled %s' % (name, built)
        print('fixture words match %s' % llvm_mc)
    for name in ('load_then_store', 'store_then_load'):
        pipe = translate(name)
        hlsl = pipe.hlsl_cs()
        assert hlsl.count('RWByteAddressBuffer u0 : register(u0);') == 1, hlsl
        assert 'u0.Load(' in hlsl and 'u0.Store(' in hlsl, hlsl
        assert 'ByteAddressBuffer r0' not in hlsl and ' r0.' not in hlsl, hlsl
        manifest = pipe.manifest().splitlines()
        assert 'uav 0 table 1 16' in manifest, manifest
        assert not [line for line in manifest if line.startswith('srv ')], manifest
        print('%s: one RW view, loads and stores through u0' % name)
    # The descriptor stays refused where a read-only view is needed: a buffer index needs the SRV stride.
    refused('read-write descriptor', lambda: translate('idxen_load').hlsl_cs())
    print('rw buffer checks passed')


def dispatch(dxc, runner, adapter):
    controls()
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        td = Path(tmp)
        for name in ('load_then_store', 'store_then_load'):
            hlsl = translate(name).hlsl_cs()
            case = td / (name + '.case')
            case.write_bytes(struct.pack('<I', rf.GROUPS[0] | (rf.GROUPS[1] << 8) | (rf.GROUPS[2] << 16))
                             + struct.pack('<%dI' % WORDS, *expected(name))
                             + struct.pack('<16I', KEY, STRIDE, *([0] * 14)))
            # The control reads zero instead of the buffer: the readback must then differ from the reference.
            mutant = hlsl.replace('u0.Load(', '(0u & u0.Load(', 1)
            assert mutant != hlsl and mutant.count('(0u & u0.Load(') == 1, hlsl
            for label, text in (('baseline', hlsl), ('dropped-load', mutant)):
                src, cso = td / f'{name}_{label}.hlsl', td / f'{name}_{label}.cso'
                src.write_text(text, encoding='ascii')
                subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
                proc = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)],
                                      capture_output=True, text=True)
                print(proc.stdout, end='')
                if proc.returncode == 77:
                    return 77
                failed = proc.returncode != 0
                if label == 'baseline' and (failed or proc.stderr.strip()):
                    raise AssertionError('%s baseline (%s): %s' % (name, adapter, proc.stderr or 'exit %d' % proc.returncode))
                if label != 'baseline' and not failed:
                    raise AssertionError('%s %s (%s) still matched the reference' % (name, label, adapter))
                print('%s %s (%s): %s' % (name, label, adapter, 'all %d words match' % WORDS if label == 'baseline' else 'rejected'))
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--dispatch', nargs=3, metavar=('DXC', 'RUNNER', 'ADAPTER'))
    args = parser.parse_args(argv)
    if args.dispatch:
        return dispatch(*args.dispatch)
    controls(args.llvm_mc)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
