"""Pure-Python unit and CLI hardening tests for tools/sampled_coverage_aot.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "sampled_coverage_aot.py"
sys.path.insert(0, str(ROOT / "tools"))

import pixel_aot  # noqa: E402
import sampled_coverage_aot  # noqa: E402
from resource_inventory import parts  # noqa: E402


S_ENDPGM = b"\x00\x00\x81\xbf"
S_NOP = b"\x00\x00\x9f\xbf"


def _build_fssm_image(code: bytes) -> bytes:
    start = 256
    image = bytearray(start) + code
    image[:4] = b"FSSM"
    struct.pack_into("<I", image, 192, start - 40)
    return bytes(image)


def _build_dxil_part(code: bytes, *, stage: int = 0) -> bytes:
    image = _build_fssm_image(code)
    native = b"XBCH" + bytes(12) + struct.pack("<II", len(image), 0) + zlib.compress(image)
    header = bytearray(24)
    struct.pack_into("<HH", header, 0, 0, stage)
    return bytes(header) + b"XDAT" + struct.pack("<II", 1, 0) + struct.pack("<III", 13, 48, len(native)) + native


def _build_dxbc(part_items: list[tuple[bytes, bytes]], *, reverse_directory: bool = False) -> bytes:
    encoded_parts = [tag + struct.pack("<I", len(payload)) + payload for tag, payload in part_items]
    table_end = 32 + 4 * len(encoded_parts)
    offsets = []
    cursor = table_end
    for chunk in encoded_parts:
        offsets.append(cursor)
        cursor += len(chunk)
    if reverse_directory:
        offsets.reverse()
    total_size = table_end + sum(len(chunk) for chunk in encoded_parts)
    return (
        b"DXBC"
        + bytes(16)
        + struct.pack("<III", 1, total_size, len(offsets))
        + struct.pack(f"<{len(offsets)}I", *offsets)
        + b"".join(encoded_parts)
    )


def _bindings(slot: int = 0) -> dict:
    return {
        "interp": 33,
        "words": [[7, 16]],
        "selector_byte": 16,
        "descriptors": [
            [10, 0, "selector", slot],
            [20, 64, "cb", 0],
            [20, 112, "cb", 4],
            [30, 16, "sampler", 0],
            [40, 0, "texture", 0],
            [40, 32, "texture", 1],
        ],
    }


def _decoded_stream(lines: list[str]) -> list[tuple[int, str]]:
    sizes = [
        8 if line.startswith(("s_load_", "s_buffer_load_", "image_sample", "exp ")) or "_e64 " in line else 4
        for line in lines
    ]
    return [(sum(sizes[:i]), line) for i, line in enumerate(lines)]


def _coverage_program(
    *,
    sample_op: str = "image_sample",
    mul_op: str = "v_mul_legacy_f32_e32",
    cmpx_op: str = "v_cmpx_gt_f32_e32",
) -> list[str]:
    lines = [
        "s_mov_b32 s11, 0",
        "s_mov_b32 s21, 0",
        "s_mov_b32 s31, 0",
        "s_mov_b32 s41, 0",
        "s_load_dwordx4 s[60:63], s[30:31], 0x10",
        "s_load_dwordx4 s[64:67], s[20:21], 0x40",
        "s_load_dwordx4 s[68:71], s[20:21], 0x70",
        "s_load_dwordx4 s[72:75], s[10:11], null",
        "s_mov_b64 s[48:49], exec",
        "s_wqm_b64 exec, exec",
        "s_mov_b64 s[52:53], exec",
        "s_mov_b32 m0, s33",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dword s6, s[72:75], s7",
        "s_buffer_load_dword s9, s[68:71], 0x8",
        "s_buffer_load_dword s3, s[64:67], 0x4",
    ]
    if sample_op in ("image_sample_l", "image_sample_b"):
        lines.append("s_buffer_load_dword s5, s[64:67], 0xc")
    if sample_op == "image_sample_b":
        lines.extend(
            [
                "v_interp_p1_f32_e32 v9, v0, attr0.x",
                "v_interp_p2_f32_e32 v9, v1, attr0.x",
                "v_interp_p1_f32_e32 v10, v0, attr0.y",
                "v_interp_p2_f32_e32 v10, v1, attr0.y",
                "s_waitcnt lgkmcnt(0)",
                "v_mov_b32_e32 v8, s5",
                "s_lshl_b32 s12, s6, 5",
                "s_load_dwordx8 s[64:71], s[40:41], s12",
                "s_waitcnt lgkmcnt(0)",
                "image_sample_b v12, v[8:10], s[64:71], s[60:63] dmask:0x8 dim:SQ_RSRC_IMG_2D",
                "s_waitcnt vmcnt(0)",
                f"{mul_op} v11, s3, v12",
            ]
        )
    elif sample_op == "image_sample_l":
        lines.extend(
            [
                "v_interp_p1_f32_e32 v8, v0, attr0.x",
                "v_interp_p2_f32_e32 v8, v1, attr0.x",
                "v_interp_p1_f32_e32 v9, v0, attr0.y",
                "v_interp_p2_f32_e32 v9, v1, attr0.y",
                "s_waitcnt lgkmcnt(0)",
                "v_mov_b32_e32 v10, s5",
                "s_lshl_b32 s12, s6, 5",
                "s_load_dwordx8 s[64:71], s[40:41], s12",
                "s_waitcnt lgkmcnt(0)",
                "image_sample_l v12, v[8:10], s[64:71], s[60:63] dmask:0x8 dim:SQ_RSRC_IMG_2D",
                "s_waitcnt vmcnt(0)",
                f"{mul_op} v11, s3, v12",
            ]
        )
    else:
        lines.extend(
            [
                "v_interp_p1_f32_e32 v8, v0, attr0.x",
                "v_interp_p2_f32_e32 v8, v1, attr0.x",
                "v_interp_p1_f32_e32 v9, v0, attr0.y",
                "v_interp_p2_f32_e32 v9, v1, attr0.y",
                "s_waitcnt lgkmcnt(0)",
                "s_lshl_b32 s12, s6, 5",
                "s_load_dwordx8 s[64:71], s[40:41], s12",
                "s_waitcnt lgkmcnt(0)",
                f"{sample_op} v10, v[8:9], s[64:71], s[60:63] dmask:0x8 dim:SQ_RSRC_IMG_2D",
                "s_waitcnt vmcnt(0)",
                f"{mul_op} v11, s3, v10",
            ]
        )
    lines.extend(
        [
            f"{cmpx_op} s9, v11",
            "s_andn2_b64 s[48:49], s[48:49], exec",
            "s_cbranch_scc0 6",
            "s_wqm_b64 s[54:55], s[48:49]",
            "s_and_b64 exec, s[52:53], s[54:55]",
            "s_and_b64 exec, exec, s[48:49]",
            "exp null off, off, off, off done vm",
            "s_endpgm",
            "s_and_b64 exec, exec, s[48:49]",
            "s_mov_b64 exec, s[48:49]",
            "exp null off, off, off, off done vm",
            "s_endpgm",
        ]
    )
    return lines


class SampledCoverageAotPositiveTests(unittest.TestCase):
    def test_synthetic_dxbc_parts_and_container_code(self):
        native_code = S_NOP + S_ENDPGM
        dxil = _build_dxil_part(native_code, stage=0)
        blob = _build_dxbc([(b"DXIL", dxil), (b"NOTE", b"meta")], reverse_directory=True)
        extracted = parts(blob)
        self.assertEqual(set(extracted), {b"DXIL", b"NOTE"})
        self.assertEqual(extracted[b"DXIL"], dxil)
        self.assertEqual(pixel_aot.container_code(extracted[b"DXIL"], stage=0, llvm="mock-llvm.dll"), native_code)

    def test_lower_sample_multiply_and_cmpx_variants(self):
        sample_expectations = {
            "image_sample": "float alpha = tex0.Sample(samp,uv).a;",
            "image_sample_lz": "float alpha = tex0.SampleLevel(samp,uv,0).a;",
            "image_sample_l": "float alpha = tex0.SampleLevel(samp,uv,params[0].w).a;",
            "image_sample_b": "float alpha = tex0.SampleBias(samp,uv,params[0].w).a;",
        }
        for sample_op, alpha_line in sample_expectations.items():
            with self.subTest(sample_op=sample_op):
                hlsl = sampled_coverage_aot.lower(_decoded_stream(_coverage_program(sample_op=sample_op)), _bindings(0))
                self.assertIn(f"    {alpha_line}", hlsl.splitlines())
                self.assertIn("    precise float product = (params[0].y == 0 || alpha == 0) ? 0.0f : params[0].y * alpha;", hlsl.splitlines())
                self.assertIn("    if (params[1].z > product) discard;", hlsl.splitlines())

        for mul_op in ("v_mul_legacy_f32_e32", "v_mul_legacy_f32_e64", "v_mul_f32_e32", "v_mul_f32_e64"):
            is_legacy = "legacy" in mul_op
            expected_prod = "(params[0].y == 0 || alpha == 0) ? 0.0f : params[0].y * alpha" if is_legacy else "params[0].y * alpha"
            for relation, sym in (("lt", "<"), ("le", "<="), ("gt", ">"), ("ge", ">=")):
                for form in ("e32", "e64"):
                    cmpx_op = f"v_cmpx_{relation}_f32_{form}"
                    with self.subTest(mul_op=mul_op, cmpx_op=cmpx_op):
                        hlsl = sampled_coverage_aot.lower(
                            _decoded_stream(_coverage_program(mul_op=mul_op, cmpx_op=cmpx_op)),
                            _bindings(1),
                        )
                        self.assertIn("    float alpha = tex1.Sample(samp,uv).a;", hlsl.splitlines())
                        self.assertIn(f"    precise float product = {expected_prod};", hlsl.splitlines())
                        self.assertIn(f"    if (params[1].z {sym} product) discard;", hlsl.splitlines())

    def test_translate_and_main_emit_valid_hlsl_with_mocked_disassembly(self):
        native_code = S_NOP + S_ENDPGM
        blob = _build_dxbc([(b"DXIL", _build_dxil_part(native_code, stage=0))])
        decoded = _decoded_stream(
            _coverage_program(sample_op="image_sample_l", mul_op="v_mul_f32_e64", cmpx_op="v_cmpx_le_f32_e64")
        )
        bindings = _bindings(1)
        expected = sampled_coverage_aot.lower(decoded, bindings)

        with mock.patch.object(sampled_coverage_aot, "disassemble", return_value=decoded) as disasm_mock:
            self.assertEqual(sampled_coverage_aot.translate(blob, bindings, "mock-llvm.dll"), expected)
            disasm_mock.assert_called_once_with(native_code, "mock-llvm.dll", offsets=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(blob)
            manifest = root / "bindings.json"
            manifest.write_text(json.dumps(bindings), encoding="utf-8")

            for out_rel in ("nested/coverage.hlsl", "UPPER.HLSL"):
                out_path = root / out_rel
                seen = []

                def fake_disassemble(code, llvm_lib, *, offsets=False):
                    seen.append((code, llvm_lib, offsets))
                    return decoded

                with (
                    mock.patch.object(sampled_coverage_aot, "disassemble", side_effect=fake_disassemble),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    rc = sampled_coverage_aot.main(
                        [
                            str(container),
                            "--bindings",
                            str(manifest),
                            "--llvm",
                            "mock-llvm.dll",
                            "--output",
                            str(out_path),
                        ]
                    )
                self.assertEqual(rc, 0)
                self.assertEqual(seen, [(native_code, "mock-llvm.dll", True)])
                self.assertEqual(out_path.read_text(encoding="ascii"), expected)
                self.assertEqual(list(out_path.parent.glob("*.tmp")), [])


class SampledCoverageAotNegativeTests(unittest.TestCase):
    def _assert_cli_fails(self, argv: list[str], out_dir: Path, *, expected_text: str = "") -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        before_entries = sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                sampled_coverage_aot.main(argv)
        self.assertIsInstance(exc.exception.code, int)
        self.assertNotEqual(exc.exception.code, 0)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(stderr.getvalue().strip())
        if expected_text:
            self.assertIn(expected_text, stderr.getvalue())
        if out_dir.exists():
            self.assertEqual(list(out_dir.glob("*.tmp")), [])
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), before_entries)
        return stderr.getvalue()

    def test_missing_or_unreadable_inputs_and_bad_json_exit_cleanly(self):
        valid_blob = _build_dxbc([(b"DXIL", _build_dxil_part(S_NOP + S_ENDPGM))])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            out_dir.mkdir()
            container = root / "shader.dxbc"
            container.write_bytes(valid_blob)
            manifest = root / "bindings.json"
            manifest.write_text(json.dumps(_bindings()), encoding="utf-8")

            missing_container = root / "missing.dxbc"
            missing_bindings = root / "missing.json"
            folder = root / "dir"
            folder.mkdir()
            bad_json = root / "bad.json"
            bad_json.write_text("{not-json", encoding="utf-8")

            cases = (
                ("missing_container", missing_container, manifest),
                ("directory_container", folder, manifest),
                ("missing_bindings", container, missing_bindings),
                ("directory_bindings", container, folder),
                ("bad_json", container, bad_json),
            )
            for label, c_path, b_path in cases:
                with self.subTest(case=label):
                    out_file = out_dir / f"{label}.hlsl"
                    self._assert_cli_fails(
                        [
                            str(c_path),
                            "--bindings",
                            str(b_path),
                            "--llvm",
                            "mock-llvm.dll",
                            "--output",
                            str(out_file),
                        ],
                        out_dir,
                        expected_text="Unsupported coverage input",
                    )
                    self.assertFalse(out_file.exists())

            # Subprocess smoke test on missing container -> non-zero without Traceback.
            proc = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(TOOL),
                    str(missing_container),
                    "--bindings",
                    str(manifest),
                    "--llvm",
                    "mock-llvm.dll",
                    "--output",
                    str(out_dir / "sub.hlsl"),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("Unsupported coverage input", proc.stderr)

    def test_bindings_validation_output_suffix_and_collision_rejected(self):
        valid_blob = _build_dxbc([(b"DXIL", _build_dxil_part(S_NOP + S_ENDPGM))])
        decoded = _decoded_stream(_coverage_program())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            out_dir.mkdir()
            container = root / "shader.hlsl"
            container.write_bytes(valid_blob)
            manifest = root / "bindings.hlsl"
            manifest.write_text(json.dumps(_bindings()), encoding="utf-8")

            with mock.patch.object(sampled_coverage_aot, "disassemble", return_value=decoded) as disasm_mock:
                for bad_out in (out_dir / "bad.txt", out_dir / "no_suffix", out_dir / "bad.hlsl.bak"):
                    with self.subTest(bad_out=bad_out.name):
                        self._assert_cli_fails(
                            [
                                str(container),
                                "--bindings",
                                str(manifest),
                                "--llvm",
                                "mock-llvm.dll",
                                "--output",
                                str(bad_out),
                            ],
                            out_dir,
                            expected_text="Expected a separate HLSL output",
                        )
                        self.assertFalse(bad_out.exists())

                for same_out in (
                    container,
                    root / "out" / ".." / "shader.hlsl",
                    manifest,
                    root / "out" / ".." / "bindings.hlsl",
                ):
                    with self.subTest(same_out=str(same_out)):
                        self._assert_cli_fails(
                            [
                                str(container),
                                "--bindings",
                                str(manifest),
                                "--llvm",
                                "mock-llvm.dll",
                                "--output",
                                str(same_out),
                            ],
                            root,
                            expected_text="Expected a separate HLSL output",
                        )
                        self.assertEqual(container.read_bytes(), valid_blob)

                bad_bindings_cases = (
                    ("empty_dict", {}),
                    ("non_dict", []),
                    ("missing_descriptors", {"interp": 33, "words": [], "selector_byte": 16}),
                    ("empty_descriptors", {**_bindings(), "descriptors": []}),
                    ("short_descriptor_entry", {**_bindings(), "descriptors": [[10, 0, "selector"]]}),
                    ("bad_words_entry", {**_bindings(), "words": [[7]]}),
                )
                for label, bad_payload in bad_bindings_cases:
                    with self.subTest(bad_bindings=label):
                        bad_file = root / f"{label}.json"
                        bad_file.write_text(json.dumps(bad_payload), encoding="utf-8")
                        self._assert_cli_fails(
                            [
                                str(container),
                                "--bindings",
                                str(bad_file),
                                "--llvm",
                                "mock-llvm.dll",
                                "--output",
                                str(out_dir / f"{label}.hlsl"),
                            ],
                            out_dir,
                            expected_text="Unsupported coverage input",
                        )
                        with self.assertRaises(ValueError):
                            sampled_coverage_aot.lower(decoded, bad_payload)

                disasm_mock.assert_not_called()

    def test_malformed_dxbc_missing_dxil_and_unsupported_lowering_fail_cleanly(self):
        valid_blob = _build_dxbc([(b"DXIL", _build_dxil_part(S_NOP + S_ENDPGM))])
        short_header = b"DXBC" + b"\x00" * 12
        truncated_payload = bytearray(valid_blob)
        offset0 = struct.unpack_from("<I", truncated_payload, 32)[0]
        struct.pack_into("<I", truncated_payload, offset0 + 4, len(valid_blob))
        no_dxil = _build_dxbc([(b"NOTE", b"only-note")])

        with self.assertRaises(ValueError):
            sampled_coverage_aot.translate(no_dxil, _bindings(), "mock-llvm.dll")

        base_lines = _coverage_program()
        unsupported_insn = _decoded_stream(
            ["v_sin_f32_e32 v10, v8" if line.startswith("image_sample ") else line for line in base_lines]
        )
        missing_exit = _decoded_stream(base_lines[:-12])
        for bad_decoded in (unsupported_insn, missing_exit):
            with self.assertRaises(ValueError):
                sampled_coverage_aot.lower(bad_decoded, _bindings())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            out_dir.mkdir()
            manifest = root / "bindings.json"
            manifest.write_text(json.dumps(_bindings()), encoding="utf-8")

            for label, payload in (
                ("truncated_dxbc", short_header),
                ("truncated_payload", bytes(truncated_payload)),
                ("missing_dxil", no_dxil),
            ):
                with self.subTest(container=label):
                    c_path = root / f"{label}.dxbc"
                    c_path.write_bytes(payload)
                    self._assert_cli_fails(
                        [
                            str(c_path),
                            "--bindings",
                            str(manifest),
                            "--llvm",
                            "mock-llvm.dll",
                            "--output",
                            str(out_dir / f"{label}.hlsl"),
                        ],
                        out_dir,
                        expected_text="Unsupported coverage input",
                    )

            good_container = root / "good.dxbc"
            good_container.write_bytes(valid_blob)
            for label, bad_decoded in (("unsupported_insn", unsupported_insn), ("missing_exit", missing_exit)):
                with self.subTest(lower=label):
                    with mock.patch.object(sampled_coverage_aot, "disassemble", return_value=bad_decoded):
                        self._assert_cli_fails(
                            [
                                str(good_container),
                                "--bindings",
                                str(manifest),
                                "--llvm",
                                "mock-llvm.dll",
                                "--output",
                                str(out_dir / f"{label}.hlsl"),
                            ],
                            out_dir,
                            expected_text="Unsupported coverage input",
                        )

            # Atomic temp cleanup on replace OSError leaves existing output intact and zero .tmp files.
            existing = out_dir / "existing.hlsl"
            existing.write_text("original", encoding="ascii")
            with (
                mock.patch.object(sampled_coverage_aot, "disassemble", return_value=_decoded_stream(base_lines)),
                mock.patch.object(Path, "replace", side_effect=OSError("disk full")),
            ):
                self._assert_cli_fails(
                    [
                        str(good_container),
                        "--bindings",
                        str(manifest),
                        "--llvm",
                        "mock-llvm.dll",
                        "--output",
                        str(existing),
                    ],
                    out_dir,
                    expected_text="disk full",
                )
            self.assertEqual(existing.read_text(encoding="ascii"), "original")


if __name__ == "__main__":
    unittest.main()
