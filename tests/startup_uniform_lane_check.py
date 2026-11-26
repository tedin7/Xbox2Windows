"""Read a fixed element from a source-built shader and reject a lane-varying or stale VGPR."""
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/startup"))
import gfx2hlsl as gfx


class TestPipeline:
    root = {"params": [{"type": 2}], "flags": 0}

    def resource(self, key, kind):
        assert key == (("root", 0), None) and kind == "buf"
        return "r0"


def vop1(opcode, dst, src):
    return 0x7e000000 | dst << 17 | opcode << 9 | src


def translate(*extra, src=7, dest=4):
    # s_buffer_load_dword s7, s[8:11], 12
    # v_cvt_i32_f32 v4, src; [extra instructions]; v_readfirstlane_b32 s6, v[dest]
    words = [0xf4000000 | 8 << 18 | 7 << 6 | 4, 125 << 25 | 12,
             vop1(8, 4, src), *extra, vop1(2, 6, 256 + dest), 0xbf810000]
    return gfx.Stage(TestPipeline(), "ps", struct.pack(f"<{len(words)}I", *words))


stage = translate()
body = "\n".join(stage.blocks[0][1])
assert "s7 = r0.Load(0x0000000cu + 0);" in body
assert "v4 = uint(int(F(s7)));" in body
assert "s6 = v4;" in body
for words, args in [
    ([], {"src": 256}),             # Per-lane source.
    ([], {"dest": 0}),              # Read a different VGPR.
    ([vop1(1, 4, 256)], {}),        # Clobber the proven VGPR with a per-lane value.
    ([0xbe800300 | 126 << 16 | 128], {}),  # s_mov_b32 exec_lo, 0
    ([0xbf820000], {}),             # Branch target at the read.
]:
    try:
        translate(*words, **args)
    except gfx.Unsupported as error:
        assert "requires a proven uniform source" in str(error), str(error)
    else:
        raise AssertionError("accepted an unproven readfirstlane")
print("fixed scalar buffer value translated; five unproven reads rejected")


def compute_controls():
    import startup_compute_cross_lane_fixture as fixture

    hlsl = fixture.translate(fixture.readlane_program())
    assert 's46 = x2w_quad[' in hlsl and 'v43 = v42 + v41;' in hlsl
    uniform = fixture.translate(fixture.readlane_program(selector=7))
    assert '(s7 & 31u)' in uniform
    first = b''.join([
        fixture.move(41, fixture.inline(22)),
        fixture.sop1(0x13, 48, 126),
        fixture.readlane(48),
        fixture.move(42, 46), fixture.store(42),
    ])
    hlsl = fixture.translate(first)
    assert 's46 = v41;' in hlsl and 'GroupMemoryBarrierWithGroupSync' not in hlsl
    fixture.refused('requires a constant or wave-uniform lane',
                    lambda: fixture.translate(fixture.readlane_program(selector=fixture.vgpr(0))))
    fixture.refused('requires a group of whole waves',
                    lambda: fixture.translate(fixture.readlane_program(), group=(30, 2, 1)))

    hlsl = fixture.translate(fixture.writelane_program())
    assert 'if (exec & 1) v37 = 0x0000003bu;' in hlsl
    fixture.refused('effect outside scalar registers',
                    lambda: fixture.translate(fixture.writelane_program(lane=12)))


def compute_dispatch(dxc, runner, adapter):
    import startup_compute_cross_lane_fixture as fixture

    source = [7 + 13 * index for index in range(64)]
    broadcast = [source[5 + 32 * (index // 32)] for index in range(64)]
    wave_zero = [source[5]] * 64
    expected = [a + b for a, b in zip(source, broadcast)] + broadcast
    mutant = [a + b for a, b in zip(source, wave_zero)] + wave_zero
    rc = fixture.dispatch(dxc, runner, adapter, fixture.translate(fixture.readlane_program()),
                          expected, source, lambda h: h.replace('& ~31u', '& 0u'), mutant)
    if rc:
        return rc
    expected = [59 if lane % 32 == 11 else 17 for lane in range(64)]
    return fixture.dispatch(dxc, runner, adapter, fixture.translate(fixture.writelane_program()),
                            expected, [], lambda h: h.replace('v37 = 0x00000011u;', 'v37 = 0x0000003bu;'),
                            [59] * 64)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        compute_controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(compute_dispatch(*sys.argv[2:]))
    else:
        sys.exit(__doc__)
