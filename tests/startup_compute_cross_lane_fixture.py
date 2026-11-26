"""Source-built, two-wave compute fixtures and the shared WARP readback driver."""
from pathlib import Path
import struct
import subprocess
import tempfile

from startup_gfx2hlsl_compute_check import G, SENTINEL, WORDS, refused, root_blob
from startup_compute_tg_size_check import (
    S_ENDPGM, S_LOAD_X8_S12_S10, S_MOV_B32, S_WAITCNT_LGKM0, TGID_X, USER7,
    V_ADD_NC_U32, V_LSHLREV_B32, V_MAD_U32_U24, V_MOV_B32,
    sop1, sopp, tidig, vop1, vop2, vop3,
)
from startup_compute_buffer_atomics_check import mubuf
from startup_compute_lds_dual_atomics_check import vopc

GROUP = (32, 2, 1)
RSRC2 = USER7 | TGID_X | tidig(2)
WAIT_LOAD = sopp(0x0c, 0)


def inline(value):
    return 128 + value


def vgpr(index):
    return 256 + index


ADDRESS = b''.join([
    sop1(S_MOV_B32, 10, 6),
    sop1(S_MOV_B32, 11, inline(0)),
    S_LOAD_X8_S12_S10,
    S_WAITCNT_LGKM0,
    vop3(V_MAD_U32_U24, 29, vgpr(1), inline(32), vgpr(0)),
    vop2(V_LSHLREV_B32, 30, inline(2), 29),
])


def load():
    return mubuf(0x0c, 41, 30, 12, inline(0))


def store(vdata, offset=0):
    return mubuf(0x1c, vdata, 30, 16, inline(0), offset=offset)


def move(dst, src):
    return vop1(V_MOV_B32, dst, src)


def translate(body, group=GROUP, wave_size=32):
    return G.ComputePipeline(ADDRESS + body + S_ENDPGM, root_blob(), RSRC2, group,
                             wave_size=wave_size).hlsl_cs()


def readlane(selector):
    return vop3(0x360, 46, vgpr(41), selector, 0)


def readlane_program(selector=inline(5)):
    return b''.join([
        load(), WAIT_LOAD, readlane(selector), move(42, 46),
        vop2(V_ADD_NC_U32, 43, vgpr(42), 41), store(43),
        store(42, 256),
    ])


def writelane_program(lane=11):
    return b''.join([
        move(37, inline(17)),
        sop1(S_MOV_B32, 49, 126),          # save entry EXEC
        vopc(0xd2, inline(11), 0),         # one lane in each wave
        sopp(0x08, 2),                    # branch past the two-word write
        vop3(0x361, 37, inline(59), inline(lane), 0),
        sop1(S_MOV_B32, 126, 49),
        store(37),
    ])


def dispatch(dxc, runner, adapter, hlsl, expected, source, mutate=None, mutant=None):
    """Check every output word, then require the predicted first mutant mismatch."""
    expected = list(expected)
    assert len(expected) <= WORDS
    shaders = [(hlsl, None)]
    if mutate is not None:
        mutant = list(mutant)
        assert len(expected) == len(mutant) and expected != mutant
        first = next(i for i, (good, bad) in enumerate(zip(expected, mutant)) if good != bad)
        mismatch = 'word=%d expected=%08x actual=%08x' % (first, expected[first], mutant[first])
        bad_hlsl = mutate(hlsl)
        assert bad_hlsl != hlsl
        shaders.append((bad_hlsl, mismatch))
    expected += [SENTINEL] * (WORDS - len(expected))
    source = list(source)
    source += [0] * (WORDS - len(source))
    out = Path(__file__).resolve().parents[1] / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        src, cso, case = [Path(tmp) / name for name in ('shader.hlsl', 'shader.cso', 'data.case')]
        payload = [0x010101, *expected, *([0] * 16), *source]
        case.write_bytes(struct.pack('<%dI' % len(payload), *payload))
        for shader, want in shaders:
            src.write_text(shader, encoding='ascii')
            subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
            proc = subprocess.run([str(Path(runner).resolve()), adapter, '--compute-store-case', str(cso), str(case)],
                                  capture_output=True, text=True)
            print(proc.stdout, end='')
            if proc.returncode == 77:
                return 77
            diagnostics = [line for line in proc.stderr.splitlines()
                           if not (adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '))]
            if want:
                assert proc.returncode == 1 and diagnostics[:1] == [want], (want, proc.returncode, proc.stderr)
                print('control rejected at %s' % want)
            else:
                assert proc.returncode == 0 and not diagnostics, (proc.returncode, proc.stderr)
    return 0
