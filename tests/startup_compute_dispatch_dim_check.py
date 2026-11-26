"""Startup FP32 1D and 3D image_store UAV binding through the production slot-67 / slot-14 path.

Model: startup_compute_dispatch_array_check.py. The WARP readback tests of startup_compute_image_store_check.py render
through offscreen_check; this one runs the stand-in's own record_translated_dispatch (d3d12_compute_dispatch.inl), which
parses the `uav N tex1d|tex3d float table P B channels 4` manifest line and validates the caller's UAV view:

  - untranslated: the texture is unchanged and the dispatch is refused for want of a translation;
  - translated: the 1D texture (x < 8) or the 3D slices (relative z 1 and 3 of a view that starts at slice 1, rows
    y < 2, x < 8) hold the stored values and nothing else changed, with no debug-layer error;
  - a view of the wrong shape is refused before the dispatch: a 2D view for either manifest, and the 3D view for the 1D
    manifest and the other way round, so a 3D manifest parsed as a 1D one fails here;
  - controls through the same binding: a store one texel (1D) or one slice (3D) off changes the readback by exactly the
    predicted number of words.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_compute_dispatch_array_check as R
import startup_compute_dispatch_texture_check as T
import startup_compute_image_store_check as A
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a

WORDS_1D = 32 * 4
WORDS_2D = R.WORDS                    # one 32 x 8 RGBA32F slice
WORDS_3D = R.WORDS * R.LAYERS         # six slices; the view covers slices 1..4


def reference_1d(shift=0):
    result = [0xbf000000 | i for i in range(WORDS_1D)]
    for x in range(4 * A.fixture.GROUPS):
        at = (x + shift) * 4
        result[at:at + 4] = A.ARRAY_VALUES
    return result


def run_case(directory, consumer, env, program, mode, layout_words):
    (directory / "program").write_bytes(program)
    return T.run(directory, consumer, env, mode, "<%dI" % layout_words)


def check(directory, consumer, dxc, adapter, shape, z):
    directory.mkdir()
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP="1", X2W_D3D12_DEBUG="1")
    if adapter == "hardware":
        env["X2W_TEST_DISPATCH_HARDWARE"] = "1"
    (directory / "compute_root").write_bytes(T.C.root_blob(3, table_range=True))
    env["X2W_TEST_COMPUTE_ROOT"] = str(directory / "compute_root")
    (directory / "data").write_bytes(DATA)
    (directory / "state").write_bytes(state_a(A.fixture.GROUP, A.RSRC2))
    one_d = shape == "1d"
    program = A.program_1d() if one_d else A.program_3d(z)
    mode, words = ("texture-1d", WORDS_1D) if one_d else ("texture-3d", WORDS_3D)
    initial = [0xbf000000 | i for i in range(WORDS_1D)] if one_d else R.reference()
    expected = reference_1d() if one_d else R.reference(z)
    label = "1D" if one_d else "3D relative slice %d (physical %d)" % (z, R.FIRST + z)

    got, result = run_case(directory, consumer, env, program, mode, words)
    print(result.stdout, end="")
    assert got == initial, "untranslated dispatch wrote the texture"
    assert len(T.lines(result, "Dispatch(2, 1, 1) refused: no translation")) == 1, result.stderr[-3000:]
    key = int(T.lines(result, "x2w-draw compute request")[0].split()[3], 16)
    T.translate(cache, dxc, key)
    manifest = (cache / f"{key:016x}.txt").read_text()
    kind = "tex1d" if one_d else "tex3d"
    assert f"uav 0 {kind} float table 1 64 channels 4" in manifest.splitlines(), manifest
    hlsl = cache / f"{key:016x}.cs.hlsl"
    cso = cache / f"{key:016x}.cs.cso"
    source = hlsl.read_text()
    assert ("RWTexture1D<float4> w0 : register(u0);" if one_d else "RWTexture3D<float4> w0 : register(u0);") in source
    store = A.LINE_1D if one_d else A.LINE_3D
    assert source.count(store) == 1, source

    got, result = run_case(directory, consumer, env, program, mode, words)
    assert len(T.lines(result, "Dispatch(2, 1, 1) translated")) == 1, result.stderr[-3000:]
    debug = T.clean(result)
    wrong = sum(a != b for a, b in zip(got, expected, strict=True))
    assert got == expected, (label, wrong)
    changed = sum(a != b for a, b in zip(initial, expected, strict=True))
    assert changed > 0
    print(f"{adapter}: {label}: {len(got)} words, {changed} stored, mismatches={wrong}; debug layer: {debug}", flush=True)

    # A view of another shape than the manifest names is refused before the dispatch, the texture untouched.
    wanted = "an output texture view is not 1D" if one_d else "an output texture view is not 3D"
    others = [("texture-array-2d", WORDS_2D, R.reference(layers=1))]
    others.append(("texture-3d", WORDS_3D, R.reference()) if one_d else ("texture-1d", WORDS_1D, [0xbf000000 | i for i in range(WORDS_1D)]))
    for other, other_words, unchanged in others:
        got, result = run_case(directory, consumer, env, program, other, other_words)
        refusal = T.lines(result, "Dispatch(2, 1, 1) refused:")
        assert len(refusal) == 1 and refusal[0].endswith(wanted), (other, result.stderr[-3000:])
        assert got == unchanged and not T.lines(result, "Dispatch(2, 1, 1) translated"), other
        print(f"{adapter}: {label}: a {other} view is refused ({wanted})", flush=True)

    # Control through the same binding: the store lands one texel (1D) or one slice (3D) away.
    if one_d:
        mutant, control = source.replace("w0[v8]", "w0[v8 + 1]"), reference_1d(1)
    else:
        index = "uint3(v8, v9, v10)"
        mutant = source.replace(index, f"uint3(v8, v9, v10 {'+ 1u' if z == 1 else '- 1u'})")
        control = R.reference(2)
    assert mutant != source
    predicted = sum(a != b for a, b in zip(control, expected, strict=True))
    assert predicted == (8 if one_d else 128), (label, predicted)
    hlsl.write_text(mutant, encoding="ascii")
    subprocess.run([dxc, "-Gis", "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
    got, result = run_case(directory, consumer, env, program, mode, words)
    assert len(T.lines(result, "Dispatch(2, 1, 1) translated")) == 1, result.stderr[-3000:]
    T.clean(result)
    wrong = sum(a != b for a, b in zip(got, expected, strict=True))
    assert wrong == predicted and got == control, (label, wrong, predicted)
    print(f"{adapter}: {label}: off-by-one control: mismatches={wrong} (expected {predicted})", flush=True)


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temp:
        check(Path(temp) / "1d", consumer, dxc, adapter, "1d", 0)
        for z in (1, 3):
            check(Path(temp) / f"3d-{z}", consumer, dxc, adapter, "3d", z)
    print(f"startup_compute_dispatch_dim: PASS on {adapter}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in ("warp", "hardware"):
        sys.exit(__doc__)
    main(*sys.argv[1:])
