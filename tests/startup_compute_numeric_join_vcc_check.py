"""Synthetic numeric SGPR join and odd VCC mask, with compute WARP readback."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_gfx2hlsl_compute_check as C

G = C.G
LOW_MASK = 0x80010004
HIGH_MASK = 0x00020010
ELSE_WORD = 47


def smem(op, dst, base, offset):
    return (0xf4000000 | op << 18 | dst << 6 | base // 2, 0xfa000000 | offset)


def mov(dst, src):
    return 0xbe800300 | dst << 16 | src


def fixture(wave):
    # s[6:7] is the table. Load the source V# at byte 32 and the store
    # V# at byte 96, into disjoint register ranges.
    words = [mov(7, 128), *smem(2, 32, 6, 32), *smem(2, 48, 6, 96), 0xbf8c0000]
    if wave is None:
        # s35 holds V# word 3 on the taken edge and a literal on fallthrough.
        words += [0xbf068004, 0xbf850001, mov(35, 128 + ELSE_WORD), 0x7e0a0223]
    else:
        # s_mov_b64 vcc, s[35:36]; v_cndmask_b32 v5, 0, v0, vcc_lo.
        # The odd low SGPR is descriptor word 3. The high SGPR has a
        # distinct bitmap so the upper half of wave64 detects swapping.
        words += [mov(36, 255), HIGH_MASK, 0xbeea0423, 0x020a0080]
    # v4 = 4 * v0; UAV[v4 / 4] = v5.
    words += [0x34080082, 0xe0701000, 0x800c0504, 0xbf810000]
    return struct.pack('<%dI' % len(words), *words)


def translate(wave):
    code = fixture(wave)
    pipe = G.ComputePipeline(code, C.root_blob(), C.USER7, (wave or 4, 1, 1), wave_size=wave)
    return pipe.hlsl_cs(), pipe.uds.index(('descword3', 0))


def cases():
    for wave in (None, 32, 64):
        text, index = translate(wave)
        if wave is None:
            old = f's35 = ud{index};'
            new = 's35 = 0u;'
            choices = (0, 1)
        else:
            start = next(line for line in text.splitlines() if 'vcc = (' in line)
            old = start.strip()
            new = (old.replace(f'ud{index}', f'0x{HIGH_MASK:08x}u') if wave == 32
                   else old.replace(f'0x{HIGH_MASK:08x}u', f'ud{index}'))
            assert new != old, old
            choices = (0,)
        assert text.count(old) == 1, text
        for selector in choices:
            if wave is None:
                values = [LOW_MASK if selector == 0 else ELSE_WORD] * 4
            else:
                bits = LOW_MASK | (HIGH_MASK << 32 if wave == 64 else 0)
                values = [lane if bits & (1 << lane) else 0 for lane in range(wave)]
            constants = [selector] + [0] * 15
            constants[index] = LOW_MASK
            yield f'{wave or "phi"}_{selector}', text, text.replace(old, new), values, constants


def dispatch(dxc, runner):
    out = Path(__file__).resolve().parents[1] / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as directory:
        src, cso, case = (Path(directory) / name for name in ('test.hlsl', 'test.cso', 'test.case'))
        for name, hlsl, mutant, values, constants in cases():
            expected = values + [C.SENTINEL] * (C.WORDS - len(values))
            case.write_bytes(struct.pack('<I', 0x010101) + struct.pack('<%dI' % C.WORDS, *expected)
                             + struct.pack('<16I', *constants))
            for is_mutant, source in enumerate((hlsl, mutant)):
                if name == 'phi_1' and is_mutant:
                    continue  # The other selector exercises the removed edge store.
                src.write_text(source, encoding='ascii')
                subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
                run = subprocess.run([runner, 'warp', '--compute-store-case', str(cso), str(case)],
                                     capture_output=True, text=True)
                print(run.stdout, end='')
                if run.returncode == 77:
                    return 77
                if is_mutant:
                    assert run.returncode == 1 and 'expected=' in run.stderr, (name, run.returncode, run.stderr)
                else:
                    assert run.returncode == 0 and not run.stderr, (name, run.returncode, run.stderr)
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        for name, *_ in cases():
            print(name)
    elif len(sys.argv) == 4 and sys.argv[1] == '--dispatch':
        sys.exit(dispatch(*sys.argv[2:]))
    else:
        sys.exit(__doc__)
