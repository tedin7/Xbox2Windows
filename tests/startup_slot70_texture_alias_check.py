"""Compare WARP readback of GPU-written slot 133 mips through slot 70 aliases."""
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_address_srv_check import prepare_draw_cache

SHADER = """Texture2DArray<float> t : register(t0);
float4 main(float4 p : SV_Position) : SV_Target {
    uint width, height, slices, mips;
    t.GetDimensions(0, width, height, slices, mips);
    return float4(t.Load(int4(0, 0, 0, uint(p.x) % 2)), 0, mips > 2 ? 1 : 0, 1);
}
"""
REFUSAL = "unavailable texture at the supplied base and layout: reason=overlapping_resource"


def expected_pixels(slice_index):
    return bytes(channel for _ in range(4) for x in range(8)
                 for channel in (17 + 100 * slice_index + 31 * (1 + x % 2), 0, 0, 255))


def check(directory, consumer):
    env, args, output = prepare_draw_cache(directory, consumer, SHADER, {}, "srv 0 tex2da table 0 0")
    cases = (
        ("disabled", "0", "1", None),
        ("tail", "1", "1", 0),
        ("interior", "1", "2", 1),
        ("misaligned", "1", "3", None),
        ("wrong-slice mutant", "1", "1", 0),
    )
    for name, switch, mode, slice_index in cases:
        trap = directory / f"{name}.log"
        run_env = dict(env, X2W_SLOT70_ALIAS_TEXTURE=switch, X2W_TEST_SLOT70_TEXTURE_ALIAS=mode,
                       X2W_TRAP_LOG=str(trap))
        result = subprocess.run([*args, "warp", output], env=run_env, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode == 77:
            return 77
        if result.returncode:
            raise RuntimeError(f"{name}: consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
        refused = slice_index is None
        log = trap.read_text(errors="replace") if trap.exists() else result.stderr
        if (REFUSAL in log) != refused or f"slot70 alias descriptor changed={int(not refused)}" not in result.stdout:
            raise AssertionError(f"{name}: unexpected slot70 verdict\n{log[-5000:]}")
        if switch == "1":
            counts = f"alias registered 1 resolved {int(not refused)} unreadable 0"
            if counts not in result.stdout:
                raise AssertionError(f"{name}: missing {counts}\n{result.stdout}")
        elif "memory SRV switches:" in result.stdout:
            raise AssertionError(f"{name}: unexpected switch census\n{result.stdout}")
        actual = output.read_bytes()
        expected = bytes((0, 0, 0, 255)) * 32 if refused else expected_pixels(slice_index)
        if actual != expected:
            raise AssertionError(f"{name}: actual {actual.hex()} expected {expected.hex()}")
        if name == "wrong-slice mutant":
            correct = expected_pixels(1)
            mismatches = sum(actual[i:i + 4] != correct[i:i + 4] for i in range(0, len(correct), 4))
            if mismatches != 32:
                raise AssertionError(f"wrong-slice mutant: expected 32 mismatches, got {mismatches}")
        print(f"{name}: all 32 RGBA pixels match the case reference")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), sys.argv[1]))
