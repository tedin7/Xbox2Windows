"""Compile source-built gfx1013 image stores and exercise the startup Dispatch binder."""
from pathlib import Path
import subprocess
import sys
import tempfile

from native_direct_container_check import words
from startup_compute_dispatch_check import lines
from startup_compute_dispatch_texture_check import clean, run
from startup_compute_image_store_check import ARRAY_END, ARRAY_PROLOGUE, ARRAY_SLICE, RSRC2
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_gfx2hlsl_compute_check import root_blob


CASES = {
    "r": (1, 1, "tex2d", 1),
    "rg": (3, 1, "tex2d", 2),
    "rgb": (7, 1, "tex2d", 1),
    "array": (7, 5, "tex2da", 1),
    "volume": (7, 2, "tex3d", 1),
    "rgba": (7, 1, "tex2d", 1),
    "sparse": (9, 1, "tex2d", 1),
}
OPCODE_REFUSAL = "direct image opcode or descriptor modifiers"
CHANNEL_REFUSAL = "an exact-width image store requires a matching output texture channel count"


def program(mask, dimension):
    # ARRAY_PROLOGUE loads the image descriptor from table 1 + 64 into s[24:31].
    # v8 = 4 * group_id.x + local_id.x, v9 = 2 * group_id.y + local_id.y.
    # v10 = 1 for array/volume. EXEC explicitly selects the even lanes of each 4x2 group.
    values = (0x3F800000, 0x40000000, 0x40800000)
    data = []
    for i in range(mask.bit_count()):
        data.extend((0x7E1C02FF + (i << 17), values[i]))  # v_mov_b32 v14+i, literal
    store = words(0xF0201000 | (mask << 8) | (dimension << 3), 0x00060E08)
    return (ARRAY_PROLOGUE + ARRAY_SLICE[1] + words(0xBF8CC07F) + words(*data)
            + words(0xBEFE03FF, 0x55) + store + ARRAY_END)


def check(directory, cli, sdk, consumer, adapter, case):
    mask, dimension, view, texel_words = CASES[case]
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP="1", X2W_D3D12_DEBUG="1",
               X2W_TEST_COMPUTE_ROOT=str(directory / "compute_root"))
    env.pop("X2W_NATIVE_SHADERS", None)
    env.pop("X2W_TEST_DISPATCH_HARDWARE", None)
    if adapter == "hardware":
        env["X2W_TEST_DISPATCH_HARDWARE"] = "1"
    (directory / "compute_root").write_bytes(root_blob(3, table_range=True))
    (directory / "program").write_bytes(program(mask, dimension))
    (directory / "data").write_bytes(DATA)
    (directory / "state").write_bytes(state_a((4, 2, 1), RSRC2))

    # Each mip under test is 32x8. Array/volume use six slices and a view starting
    # at slice 1, so the shader's relative coordinate 1 writes physical slice 2.
    layers = 6 if dimension != 1 else 1
    initial = [0x30000000 + i for i in range(layers * 32 * 8 * texel_words)]
    mode = "texture-exact-" + ("rgba" if case == "sparse" else case)
    layout = f"<{len(initial)}I"
    got, result = run(directory, consumer, env, mode, layout)
    assert got == initial and lines(result, "Dispatch(2, 1, 1) refused: no translation"), result.stderr
    key = lines(result, "x2w-draw compute request")[0].split()[3]
    compiled = subprocess.run([cli, sdk, directory / "compiler-cache", "gfx1013-direct-cs@0",
                               "--requests", cache, cache], capture_output=True, text=True, timeout=30)
    diagnostic = compiled.stdout + compiled.stderr
    manifest = cache / f"{key}.txt"
    if case == "sparse":
        assert compiled.returncode and OPCODE_REFUSAL in diagnostic and not manifest.exists(), diagnostic
        print(diagnostic, end="")
        return 0
    assert compiled.returncode == 0 and manifest.is_file(), diagnostic
    assert f"uav 0 {view} float table 1 64 channels {mask.bit_count()}" in manifest.read_text().splitlines()

    expected = initial.copy()
    if case != "rgba":
        # Packed unsigned floats for (1, 2, 4), with 6/6/5 mantissa bits.
        packed = 0x3C0 | (0x400 << 11) | (0x220 << 22)
        values = {"r": [0x3F800000], "rg": [0x3F800000, 0x40000000]}.get(case, [packed])
        for y in range(2):
            for x in (0, 2, 4, 6):  # two X groups, local lanes 0, 2, 4 and 6 in each group
                i = (((2 if layers > 1 else 0) * 8 + y) * 32 + x) * texel_words
                expected[i:i + texel_words] = values

    # Native translation first with the published manifest removed (no fallback), so an
    # adapter without 32-lane waves reports the capability skip before any PSO is made;
    # then bind solely from the restored manifest.
    published = manifest.read_bytes()
    for native in (True, False):
        if native:
            manifest.unlink()
            env["X2W_NATIVE_SHADERS"] = sdk
        else:
            manifest.write_bytes(published)
            env.pop("X2W_NATIVE_SHADERS", None)
        got, result = run(directory, consumer, env, mode, layout)
        if native and lines(result, "not translated natively: the adapter has no 32-lane waves"):
            assert got == initial, result.stderr
            return 77
        assert got == expected, (case, native, result.stderr,
                                 next(i for i, (actual, want) in enumerate(zip(got, expected)) if actual != want))
        if native:
            assert lines(result, "translated natively:"), result.stderr
        if case == "rgba":
            assert lines(result, f"Dispatch(2, 1, 1) refused: {CHANNEL_REFUSAL}"), result.stderr
        else:
            assert lines(result, "Dispatch(2, 1, 1) translated"), result.stderr
            clean(result)
    print(f"{adapter} {case}: checked all {len(initial)} words through both binders")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
