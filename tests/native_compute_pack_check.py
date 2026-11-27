"""Replay source-built frontend pack and check version, input identity and integrity."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

from native_startup_request_cache_check import (compile_compute_root, compute_program_words,
                                               put_compute_request, state_a, COMPUTE_GROUP, COMPUTE_RSRC2)


def run(cli, checker, sdk):
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        inputs = base / "requests"
        inputs.mkdir()
        root = compile_compute_root(sdk, base)
        code = compute_program_words()
        state = state_a(COMPUTE_GROUP, COMPUTE_RSRC2)
        put_compute_request(inputs, code, root, state)
        put_compute_request(inputs, bytes.fromhex("0000f8d8 00000000 000081bf"), root, state)
        build = [cli, "--prepare-compute-pack", inputs, base / "cache", "0"]
        result = subprocess.run(build, capture_output=True, text=True)
        assert result.returncode == 0 and "prepared=1 unsupported=1 errors=0" in result.stdout, result
        pack = base / "cache" / "compute-0.pack"
        subprocess.run([checker, inputs, pack], check=True)
        original = pack.read_bytes()
        for position in (0, 64, len(original) - 1):
            damaged = bytearray(original)
            damaged[position] ^= 1
            pack.write_bytes(damaged)
            rejected = subprocess.run([checker, inputs, pack], capture_output=True, text=True)
            assert rejected.returncode and "damaged compute pack" in rejected.stderr, rejected
        pack.write_bytes(original)
        # Invalid source remains retryable. Other valid records still publish.
        (inputs / "req_0000000000000000.c67").write_bytes(code)
        result = subprocess.run(build, capture_output=True, text=True)
        assert result.returncode == 1 and "prepared=1 unsupported=1 errors=1" in result.stdout, result
        assert pack.read_bytes() == original


def hardware(cli, consumer, sdk):
    import startup_gfx2hlsl_compute_check as C
    from startup_compute_dispatch_check import dispatch, lines, prepare_case
    from startup_compute_dispatch_native_check import LOAD_X8, LOADS_X4
    from startup_stream_check import gfx2hlsl as G

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        cache, env = prepare_case(work, "hardware")
        env.pop("X2W_NATIVE_COMPUTE_PACK", None)
        env["X2W_NATIVE_SHADERS"] = str(sdk)
        code_file = work / "program"
        original = code_file.read_bytes()
        assert original.count(LOAD_X8) == 1
        words, refused = dispatch(work, str(consumer), env, "dispatch")
        assert words == [C.SENTINEL] * C.WORDS and lines(refused, "unaligned descriptor table entry"), refused.stderr
        code_file.write_bytes(original.replace(LOAD_X8, LOADS_X4))
        words, result = dispatch(work, str(consumer), env, "dispatch-records32")
        print(result.stdout, end="")
        if lines(result, "the adapter has no 32-lane waves"):
            print("SKIP: native compute requires a 32-lane wave")
            sys.exit(77)
        assert words == C.expected("A") and lines(result, "outputs (frontend live)"), result.stderr
        build = subprocess.run([cli, "--prepare-compute-pack", cache, work / "pack", "0"], check=True,
                               capture_output=True, text=True)
        assert "prepared=1 unsupported=1 errors=0" in build.stdout, build.stdout
        env["X2W_NATIVE_COMPUTE_PACK"] = str(work / "pack" / "compute-0.pack")
        words, result = dispatch(work, str(consumer), env, "dispatch-records32")
        assert words == C.expected("A") and lines(result, "outputs (frontend prepared)"), result.stderr
        errors = [line for line in lines(result, "x2w-d3d12-debug ") if " sev=0 " in line or " sev=1 " in line]
        assert not errors, errors[0]
        layer = lines(result, "x2w-d3d12 debug layer: ")
        print("Debug layer:", "enabled" if layer and "0x00000000" in layer[0] else "unavailable")
        words, result = dispatch(work, str(consumer), env, "dispatch")
        assert words == [C.SENTINEL] * C.WORDS and lines(result, "refused:"), result.stderr
        env["X2W_D3D12_COMPUTE_OUTPUT_TRACE"] = "1"
        for mode in ("raw", "structured", "typed", "wrong-slot", "copied"):
            words, result = dispatch(work, str(consumer), env, f"dispatch-uav-token-{mode}")
            expected = [C.SENTINEL] * C.WORDS
            if mode in ("raw", "structured", "typed"):
                expected[4:-4] = C.expected("A")[:-8]
                assert lines(result, "offset=16 bytes=1248 result=bound"), result.stderr
            elif mode == "wrong-slot":
                assert lines(result, "does not identify a slot-19 UAV"), result.stderr
            else:
                assert lines(result, "does not uniquely alias its guest byte interval"), result.stderr
                assert lines(result, "use=copy-source-attempt") and not lines(result, "use=copy-read-recorded"), result.stderr
                assert lines(result, "use=map-guest-read-request") and lines(result, "x2w-compute-output-submit"), result.stderr
            assert words == expected, (mode, words[:16], expected[:16], result.stderr)
            errors = [line for line in lines(result, "x2w-d3d12-debug ") if " sev=0 " in line or " sev=1 " in line]
            assert not errors, errors[0]
            print(f"slot-19 {mode}: all {len(words)} readback words match")
        # Publish the reference artifact offline. A native refusal must still try the published shader.
        subprocess.run([sys.executable, "-B", G.__file__, "--pending", cache], check=True, capture_output=True,
                       env=dict(os.environ, X2W_DXC=str(sdk / "dxc.exe")))
        code_file.write_bytes(original)
        words, result = dispatch(work, str(consumer), env, "dispatch")
        assert words == C.expected("A") and lines(result, "unaligned descriptor table entry"), result.stderr
        assert lines(result, "Dispatch(3, 2, 2) translated"), result.stderr
        print(f"Direct and prepared dispatch matched {C.WORDS} words. Invalid binding preserved every sentinel.")
        print(f"Prepared refusal followed by published translation matched {C.WORDS} words.")


if __name__ == "__main__":
    if sys.argv[1] == "hardware":
        hardware(*(Path(a).resolve() for a in sys.argv[2:]))
    else:
        run(*(Path(a).resolve() for a in sys.argv[1:]))
