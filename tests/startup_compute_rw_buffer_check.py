"""Source-built gfx1013 compute fixtures: one raw buffer descriptor that the program both loads and stores through.

  (no arguments)                                decode and translate, check the HLSL and the manifest (pure Python)
  --llvm-mc <llvm-mc>                           the same, with the fixtures assembled by llvm-mc
  --dispatch <dxc> <offscreen_check> <adapter>  compile the HLSL with DXC, run GPU readback against a CPU reference

Each lane updates its own word, with padding for indexed addressing. The formatted 16-bit
store updates only the low half and the formatted XYZW store round-trips the other components.
The store-then-load case reads back a word written earlier by the same lane.
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
from startup_gfx2hlsl_compute_check import G, MASK, WORDS, assemble, root_blob  # noqa: E402

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
# Spread lanes across records so a zero or hard-coded stride cannot pass.
IDXEN_LOAD = IDXEN_LOAD.replace('v4, 2, v4', 'v4, 3, v4')
IDXEN_OFFEN_LOAD = IDXEN_LOAD.replace(
    'buffer_load_dword v5, v6, s[16:19], 0 idxen',
    'v_mov_b32_e32 v7, 4\nbuffer_load_dword v5, v[6:7], s[16:19], 0 idxen offen',
).replace('buffer_store_dword v5, v4, s[16:19], 0 offen', 'buffer_store_dword v5, v4, s[16:19], 0 offen offset:4')
ATOMIC = LOAD_THEN_STORE.replace('buffer_store_dword', 'buffer_atomic_swap')
D16 = IDXEN_LOAD.replace('buffer_store_dword v5, v4, s[16:19], 0 offen',
                         'buffer_store_format_d16_x v5, v6, s[16:19], 0 idxen')
FORMAT_XYZW = IDXEN_LOAD.replace(
    'buffer_load_dword v5, v6, s[16:19], 0 idxen', 'buffer_load_format_xyzw v[8:11], v6, s[16:19], 0 idxen',
).replace('v_add_nc_u32_e32 v5, 7, v5', 'v_add_nc_u32_e32 v8, 7, v8').replace(
    'buffer_store_dword v5, v4, s[16:19], 0 offen', 'buffer_store_format_xyzw v[8:11], v6, s[16:19], 0 idxen')
THREADS = rf.LOCAL[0] * rf.GROUPS[0]
FIXTURES = {'load_then_store': LOAD_THEN_STORE, 'store_then_load': STORE_THEN_LOAD, 'idxen_load': IDXEN_LOAD,
            'idxen_offen_load': IDXEN_OFFEN_LOAD, 'atomic': ATOMIC, 'd16': D16, 'format_xyzw': FORMAT_XYZW}
# The assembled words, so the test needs no assembler; --llvm-mc reassembles and compares.
WORDS_HEX = {
    'load_then_store': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e82080834001030e004050480'
                       '703f8cbf870a0a4a001070e004050480000081bf',
    'store_then_load': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e82080834001070e004060480'
                       '001030e004050480703f8cbf870a0a4a001070e004050480000081bf',
    'idxen_load': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e83080834002030e006050480'
                  '703f8cbf870a0a4a001070e004050480000081bf',
    'idxen_offen_load': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e8308083484020e7e'
                        '003030e006050480703f8cbf870a0a4a041070e004050480000081bf',
    'atomic': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e82080834001030e004050480'
              '703f8cbf870a0a4a0010c0e004050480000081bf',
    'd16': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e83080834002030e006050480'
           '703f8cbf870a0a4a002010e206050480000081bf',
    'format_xyzw': '06038abe80038bbe05030cf4000000fa7fc08cbf07821b8f1b00084a04030c7e8308083400200ce006080480'
                   '703f8cbf8710104a00201ce006080480000081bf',
}

INDEXED = {'idxen_load': 8, 'idxen_offen_load': 8, 'd16': 8, 'format_xyzw': 16}
# RESOURCE_LEVEL=1, DST_SEL=xyzw. The structured records use OOB_SELECT=0.
WORD3 = {'d16': (1 << 24) | (11 << 12) | 0xfac, 'format_xyzw': (1 << 24) | (77 << 12) | 0xfac}
INITIAL = [0x3f80007b + i * 17 for i in range(WORDS)]


def translate(name):
    return G.ComputePipeline(bytes.fromhex(WORDS_HEX[name]), root_blob(), rf.RSRC2, rf.LOCAL)


def expected(name):
    out = INITIAL.copy()
    for i in range(THREADS):
        word = i * (INDEXED.get(name, 4) // 4) + (name == 'idxen_offen_load')
        old = i if name == 'store_then_load' else INITIAL[word]
        out[word] = ((old & 0xffff0000) | ((old + 7) & 0xffff)) if name == 'd16' else (old + 7) & MASK
    return out


def controls(llvm_mc=None):
    for reason in ('descriptor (("table", 1), 16) used for a load and a store',
                   'indexed or formatted load through a read-write descriptor (("table", 1), 16)'):
        assert G.lifted_family('unsupported: ' + reason) == 'rw-buffer', reason
    if llvm_mc:
        for name, source in FIXTURES.items():
            built = assemble(llvm_mc, source).hex()
            assert built == WORDS_HEX[name], '%s: assembled %s' % (name, built)
        print('fixture words match %s' % llvm_mc)

    for name in FIXTURES:
        census = []
        pipe = G.ComputePipeline(bytes.fromhex(WORDS_HEX[name]), root_blob(), rf.RSRC2, rf.LOCAL, refusals=census)
        hlsl = translate(name).hlsl_cs()
        assert pipe.hlsl_cs() == hlsl and not census, census
        assert hlsl.count('RWByteAddressBuffer u0 : register(u0);') == 1, hlsl
        assert 'u0.Load(' in hlsl and ': register(t' not in hlsl, hlsl
        write = {'atomic': 'u0.InterlockedExchange(', 'd16': 'store_format_d16_x(u0,',
                 'format_xyzw': 'store_format_xyzw(u0,'}.get(name, 'u0.Store(')
        assert write in hlsl, hlsl
        manifest = pipe.manifest().splitlines()
        assert [line for line in manifest if line.startswith(('srv ', 'uav '))] == ['uav 0 table 1 16'], manifest
        sources = ['rootconst 0 0', 'rootconst 0 1']
        if name in INDEXED:
            sources.append('uav_stride 0')
        if name in WORD3:
            sources.extend(('uav_descword3 0', 'uav_numrecords 0'))
        assert [line for line in manifest if line.startswith('udsrc ')] == [
            f'udsrc {i} {source}' for i, source in enumerate(sources)
        ], manifest
        print('%s: one UAV with matching entry metadata' % name)


def dispatch(dxc, runner, adapter):
    controls()
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        td = Path(tmp)
        for name in FIXTURES:
            pipe = translate(name)
            hlsl = pipe.hlsl_cs()
            case = td / (name + '.case')
            header = (1 << 31) | (rf.GROUPS[0] | (rf.GROUPS[1] << 8) | (rf.GROUPS[2] << 16))
            descriptor = {'uav_stride': INDEXED.get(name, 0), 'uav_descword3': WORD3.get(name, 0),
                          'uav_numrecords': WORDS * 4 // INDEXED.get(name, 1)}
            constants = [0] * 16
            for line in pipe.manifest().splitlines():
                if line.startswith('udsrc '):
                    _, index, source, *_ = line.split()
                    constants[int(index)] = descriptor.get(source, 0)
            case.write_bytes(struct.pack('<I', header)
                             + struct.pack('<%dI' % WORDS, *expected(name))
                             + struct.pack('<%dI' % WORDS, *INITIAL)
                             + struct.pack('<16I', *constants))
            # Drop the loaded word that feeds the modification. Case 77 is the selected XYZW format.
            old = 'case 77u: c0 = u0.Load(a);' if name == 'format_xyzw' else 'v5 = u0.Load(a + 0);'
            assert hlsl.count(old) == 1, hlsl
            mutant = hlsl.replace(old, old.split(' = ')[0] + ' = 0u;')
            for label, text in (('baseline', hlsl), ('dropped-load', mutant)):
                src, cso = td / f'{name}_{label}.hlsl', td / f'{name}_{label}.cso'
                src.write_text(text, encoding='ascii')
                subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
                proc = subprocess.run([runner, adapter, '--compute-store-sized-case', str(cso), str(case)],
                                      capture_output=True, text=True)
                print(proc.stdout, end='')
                if proc.returncode == 77:
                    return 77
                if label == 'baseline' and (proc.returncode or proc.stderr.strip()):
                    raise AssertionError('%s baseline (%s): %s' % (name, adapter, proc.stderr or 'exit %d' % proc.returncode))
                mismatch_count = THREADS - (name == 'store_then_load')
                if label != 'baseline' and (proc.returncode != 1
                                            or f'mismatches={mismatch_count}\n' not in proc.stdout
                                            or 'Compute readback differs from CPU reference' not in proc.stderr):
                    raise AssertionError('%s mutant (%s): expected readback mismatch, got %s' %
                                         (name, adapter, proc.stderr or 'exit %d' % proc.returncode))
                print('%s %s (%s): %s' % (name, label, adapter, 'all %d words match' % WORDS if label == 'baseline'
                                         else 'readback mismatch'))
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
