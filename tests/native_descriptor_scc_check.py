"""Observe the SCC of a buffer stride extraction and both MAX operand orders."""

from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from native_compute_clear_check import write_request
from native_scalar_lds_check import scalar, sop1, vmove
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_compute_translate_check import state_a
from startup_gfx2hlsl_compute_check import KEY, SENTINEL, TGID_X, TGID_Y, TGID_Z, USER7, WORDS


def program(*, runtime=False):
    offset = 0 if runtime else 32
    table, constant = (6, 4) if runtime else (4, 5)
    words = [sop1(3, 20, table), sop1(3, 21, 128),
             0xf408040a, 0xfa000000 | offset,          # input SRV in s[16:19]
             0xf408080a, 0xfa000000 | (offset + 32),  # output UAV in s[32:35]
             0xbf8cc07f,
             scalar(0x27, 25, 17, 255), 0xe0010, # s_bfe_u32: 14-bit stride
             sop1(3, 26, 253),
             scalar(9, 27, 25, 129), sop1(3, 28, 253),
             scalar(9, 29, 129, 25), sop1(3, 30, 253)]
    if runtime:
        # Flatten the 3x2x2 Dispatch so each group owns four output words.
        words += [scalar(0x26, 8, 8, 131), scalar(0x26, 9, 9, 134),
                  scalar(0, 6, 7, 8), scalar(0, 6, 6, 9)]
    words += [scalar(0x1e, 7, 6, 132), vmove(8, 7), sop1(3, 126, 129)]
    for i, reg in enumerate((26, 28, 30, constant)):
        words += [vmove(10, reg), 0xe0701000 | (4 * i), 0x80080a08]
    return words + [0xbf810000]


def main(compile_dxil, replay, sdk, backend):
    with TemporaryDirectory() as directory:
        work = Path(directory)
        dxc = Path(sdk) / "dxc.exe"
        source, root = work / "root.hlsl", work / "root.bin"
        requests = work / "requests"
        requests.mkdir()
        # s5 is one root constant and s6 is the X group ID. Exercise b0 with
        # the descriptor metadata appended to a regular root constant.
        sa = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1, 0, 0x80 | 6 << 1)

        def root_bytes(srv_count):
            source.write_text(f'#define RS "DescriptorTable(SRV(t0,numDescriptors={srv_count}),'
                              f'UAV(u0,numDescriptors={3 - srv_count},offset={srv_count})),'
                              'RootConstants(num32BitConstants=1,b0)"\n', encoding="ascii")
            subprocess.run([dxc, "-T", "rootsig_1_1", "-E", "RS", source, "-Fo", root], check=True)
            return root.read_bytes()

        marker = write_request(requests, program(), root_bytes(2), sa)
        prefix = [compile_dxil, sdk, work / "cache", "gfx1013-direct-cs@0"]
        for hit in (0, 1):
            published = work / f"published-{hit}"
            result = subprocess.run([*prefix, "--requests", requests, published],
                                    capture_output=True, text=True)
            assert result.returncode == 0 and f"frontend_cache_hit={hit}" in result.stdout, result.stdout + result.stderr
            [manifest] = published.glob("*.txt")
            # The native output revision line (#3006) is checked separately; the binding lines stay exact.
            lines = manifest.read_text().splitlines()
            assert sum(line.startswith("native ") for line in lines) == 1, manifest.read_text()
            assert [line for line in lines if not line.startswith("native ")] == [
                "x2w-manifest 2", "stage cs", "group 32 1 1", "ud 2",
                "udsrc 0 rootconst 1 0", "udsrc 1 stride 0",
                "srv 0 buf table 0 32", "uav 0 table 0 64",
            ], manifest.read_text()
        print("cold and warm manifests both bind the stride of SRV 0 after the root constant")

        negative = work / "negative"
        negative.mkdir()
        bad = write_request(negative, program(), root_bytes(1), sa)
        result = subprocess.run([*prefix, bad, work / "bad.cso"], capture_output=True, text=True)
        assert result.returncode and "numeric SCC from a symbolic descriptor field" in result.stderr, result.stderr
        print("the same descriptor load from a UAV range is refused")

        mode = "--direct-warp" if backend == "warp" else "--direct"
        readback, metadata = work / "readback.bin", work / "stride.txt"
        for value in (0, 1, 4, 16, 0x4000, 0x7fff):
            metadata.write_text(str(value), encoding="ascii")
            result = subprocess.run([replay, sdk, marker, work / "cache", "0", readback, mode, metadata],
                                    capture_output=True, text=True)
            if result.returncode == 77:
                print(result.stderr.strip())
                return 77
            assert result.returncode == 0, result.stdout + result.stderr
            stride = value & 0x3fff
            expected = [0x69696969] * 2064
            for group in range(2):
                start = 8 + group * 4
                expected[start:start + 4] = [int(stride != 0), int(stride > 1), int(1 > stride), 3]
            actual = list(struct.unpack("<2064I", readback.read_bytes()))
            assert actual == expected, next((i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
            print(f"stride word {value}: two groups match {expected[8:12]} and all guards")
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", readback, mode],
                                capture_output=True, text=True)
        assert result.returncode == 1 and "compute descriptor stride metadata missing" in result.stderr, result.stderr
        print("missing stride metadata is rejected before dispatch")
        return 0


def runtime_main(compile_dxil, consumer, sdk, backend):
    with TemporaryDirectory() as directory:
        work = Path(directory)
        cache, env = prepare_case(work, backend)
        env.pop("X2W_NATIVE_SHADERS", None)
        env.pop("X2W_NATIVE_SHADERS_ONLY", None)
        env["X2W_TEST_DISPATCH_FIELDS"] = str(work / "descriptor.bin")
        fields = Path(env["X2W_TEST_DISPATCH_FIELDS"])
        fields.write_bytes(struct.pack("<6I", 4, 64, 0, 4, WORDS, 0))
        code = program(runtime=True)
        (work / "program").write_bytes(struct.pack(f"<{len(code)}I", *code))
        (work / "state").write_bytes(state_a((32, 1, 1), USER7 | TGID_X | TGID_Y | TGID_Z))
        got, result = dispatch(work, consumer, env, "dispatch-records32", probe=True)
        assert got == [SENTINEL] * WORDS and lines(result, "refused: no translation"), result.stderr
        [marker] = cache.glob("req_*.c67")

        for route in ("native", "published"):
            selected = dict(env)
            if route == "native":
                selected["X2W_NATIVE_SHADERS"] = sdk
            else:
                subprocess.run([compile_dxil, sdk, work / "frontend", "gfx1013-direct-cs@0",
                                "--requests", cache, cache], check=True)
                manifest = cache / (marker.stem.removeprefix("req_") + ".txt")
                assert "udsrc 2 stride 0\n" in manifest.read_text(), manifest.read_text()
            for stride in (0, 1, 4, 16):
                fields.write_bytes(struct.pack("<6I", stride, 64, 0, 4, WORDS, 0))
                got, result = dispatch(work, consumer, selected, "dispatch-records32")
                if lines(result, "the adapter has no 32-lane waves"):
                    assert got == [SENTINEL] * WORDS
                    print("skip: runtime adapter has no wave32")
                    return 77
                expected = [int(stride != 0), int(stride > 1), int(1 > stride), KEY] * 12
                expected += [SENTINEL] * (WORDS - len(expected))
                assert got == expected, (
                    next((i, a, b) for i, (a, b) in enumerate(zip(got, expected)) if a != b), result.stderr[-3000:]
                )
                assert bool(lines(result, "translated natively: 3 entry values, 1 inputs, 1 outputs")) == (
                    route == "native"
                ), result.stderr[-3000:]
                assert lines(result, "Dispatch(3, 2, 2) translated") and not lines(result, "refused: "), result.stderr
                assert not [line for line in lines(result, "x2w-d3d12-debug ") if " sev=0 " in line or " sev=1 " in line]
                print(f"{route}: stride {stride}, 12 groups match {expected[:4]} and all guards")
        return 0


if __name__ == "__main__":
    if sys.argv[1] == "runtime":
        sys.exit(runtime_main(*sys.argv[2:]))
    sys.exit(main(*sys.argv[1:]))
