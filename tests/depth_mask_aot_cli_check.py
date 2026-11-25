"""Pure-Python unit and CLI hardening tests for tools/depth_mask_aot.py."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "depth_mask_aot.py"
sys.path.insert(0, str(ROOT / "tools"))

import depth_mask_aot  # noqa: E402
from pixel_aot import container_code  # noqa: E402
from resource_inventory import parts  # noqa: E402


TABLES = [(6, 32), (10, 16)]
S_ENDPGM = b"\x00\x00\x81\xbf"
S_NOP = b"\x00\x00\x9f\xbf"


def _build_fssm_image(code: bytes, *, bounded: bool = False) -> bytes:
    if bounded:
        section_base = 240
        start = section_base + 40
        min_end = 40 + len(code) + 4
        section_size = ((min_end + 15) // 16) * 16
        limit = section_base + section_size
        image = bytearray(limit)
        image[:4] = b"FSSM"
        struct.pack_into("<I", image, 4, 0x101)
        struct.pack_into("<QQ", image, 192, section_base, section_size)
        struct.pack_into("<Q", image, 232, limit)
        image[start : start + len(code)] = code
        for off in range(start + len(code), limit, 4):
            image[off : off + 4] = S_NOP
        return bytes(image)

    start = 256
    image = bytearray(start) + code
    image[:4] = b"FSSM"
    struct.pack_into("<I", image, 192, start - 40)
    return bytes(image)


def _build_dxil_part(code: bytes, *, stage: int = 0, bounded: bool = False) -> bytes:
    image = _build_fssm_image(code, bounded=bounded)
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


def _decoded_stream(lines: list[str]) -> list[tuple[int, str]]:
    sizes = [
        8 if line.startswith(("s_load_", "s_buffer_load_", "exp ")) or "_e64 " in line else 4
        for line in lines
    ]
    return [(sum(sizes[:i]), line) for i, line in enumerate(lines)]


def _depth_program(*, binary_op: str | None = None, unary_op: str | None = None, cmpx_op: str = "v_cmpx_lt_f32_e32") -> list[str]:
    lines = [
        "s_mov_b32 s7, 0",
        "s_mov_b32 s11, 0",
        "s_mov_b64 s[22:23], exec",
        "s_mov_b64 s[24:25], exec",
        "s_load_dwordx4 s[12:15], s[6:7], 0x20",
        "s_load_dwordx4 s[16:19], s[10:11], 0x10",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dword s30, s[12:15], 0x18",
        "s_buffer_load_dword s31, s[16:19], null",
        "s_waitcnt lgkmcnt(0)",
        "s_add_i32 s30, s30, 12",
        "s_and_b32 s30, s30, 31",
        "v_cvt_f32_ubyte0_e32 v5, s30",
        "v_mul_legacy_f32_e32 v7, 0x3d000000, v3",
        "v_fmac_f32_e32 v7, 0x3d000000, v5",
        "v_fract_f32_e32 v7, v7",
    ]
    target = "v7"
    if binary_op is not None:
        lines.append(f"{binary_op} v6, 0.5, v7")
        target = "v6"
    elif unary_op is not None:
        lines.append(f"{unary_op} v6, v7")
        target = "v6"
    lines.extend(
        [
            f"{cmpx_op} s31, {target}",
            "s_andn2_b64 s[22:23], s[22:23], exec",
            "s_cbranch_scc0 4",
            "s_and_b64 exec, s[24:25], s[22:23]",
            "exp mrtz, v4, off, off, off done vm",
            "s_endpgm",
            "s_mov_b64 exec, s[22:23]",
            "exp null, off, off, off, off done vm",
            "s_endpgm",
        ]
    )
    return lines


class DepthMaskAotPositiveTests(unittest.TestCase):
    def test_synthetic_dxbc_parts_and_container_code(self):
        native_code = S_NOP + S_ENDPGM
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                dxil = _build_dxil_part(native_code, stage=0, bounded=bounded)
                blob = _build_dxbc([(b"DXIL", dxil), (b"NOTE", b"meta")], reverse_directory=bounded)
                extracted = parts(blob)
                self.assertEqual(set(extracted), {b"DXIL", b"NOTE"})
                self.assertEqual(extracted[b"DXIL"], dxil)
                self.assertEqual(extracted[b"NOTE"], b"meta")
                self.assertEqual(container_code(extracted[b"DXIL"], stage=0, llvm="mock-llvm.dll"), native_code)

    def test_lower_binary_unary_cmpx_and_e64_patterns(self):
        binary_expectations = {
            "v_add_f32_e32": "(0.5f + value15)",
            "v_sub_f32_e32": "(0.5f - value15)",
            "v_subrev_f32_e32": "(value15 - 0.5f)",
            "v_mul_f32_e32": "(0.5f * value15)",
            "v_min_f32_e32": "min(0.5f, value15)",
            "v_max_f32_e32": "max(0.5f, value15)",
        }
        for op, expr in binary_expectations.items():
            with self.subTest(binary_op=op):
                hlsl = depth_mask_aot.lower(_decoded_stream(_depth_program(binary_op=op)), TABLES)
                self.assertIn("cbuffer First : register(b0) { uint4 first[128]; };", hlsl)
                self.assertIn("cbuffer Second : register(b1) { uint4 second[128]; };", hlsl)
                self.assertIn("float main(float4 pos : SV_Position) : SV_Depth {", hlsl)
                self.assertIn(f"    precise float value16 = {expr};", hlsl.splitlines())
                self.assertIn("    if (asfloat(second[0].x) < value16) discard;", hlsl.splitlines())
                self.assertIn("    return pos.z;", hlsl.splitlines())

        unary_expectations = {
            "v_rcp_f32_e32": "(1.0f / value15)",
            "v_rsq_f32_e32": "rsqrt(value15)",
            "v_sqrt_f32_e32": "sqrt(value15)",
            "v_floor_f32_e32": "floor(value15)",
            "v_trunc_f32_e32": "trunc(value15)",
            "v_rndne_f32_e32": "round(value15)",
            "v_ceil_f32_e32": "ceil(value15)",
            "v_exp_f32_e32": "exp2(value15)",
            "v_log_f32_e32": "log2(value15)",
        }
        for op, expr in unary_expectations.items():
            with self.subTest(unary_op=op):
                hlsl = depth_mask_aot.lower(_decoded_stream(_depth_program(unary_op=op)), TABLES)
                self.assertIn(f"    precise float value16 = {expr};", hlsl.splitlines())

        cmpx_expectations = {
            "v_cmpx_lt_f32_e32": "<",
            "v_cmpx_le_f32_e32": "<=",
            "v_cmpx_gt_f32_e32": ">",
            "v_cmpx_ge_f32_e32": ">=",
        }
        for op, symbol in cmpx_expectations.items():
            with self.subTest(cmpx_op=op):
                hlsl = depth_mask_aot.lower(_decoded_stream(_depth_program(cmpx_op=op)), TABLES)
                self.assertIn(f"    if (asfloat(second[0].x) {symbol} value15) discard;", hlsl.splitlines())

        e32_lines = _depth_program(binary_op="v_add_f32_e32", cmpx_op="v_cmpx_ge_f32_e32")
        e64_lines = [
            line.replace("v_add_f32_e32", "v_add_f32_e64").replace("v_cmpx_ge_f32_e32", "v_cmpx_ge_f32_e64")
            for line in e32_lines
        ]
        self.assertEqual(
            depth_mask_aot.lower(_decoded_stream(e32_lines), TABLES),
            depth_mask_aot.lower(_decoded_stream(e64_lines), TABLES),
        )

    def test_main_generates_valid_hlsl_with_mocked_disassembly(self):
        native_code = S_NOP + S_ENDPGM
        blob = _build_dxbc([(b"DXIL", _build_dxil_part(native_code, stage=0))])
        decoded = _decoded_stream(_depth_program(unary_op="v_sqrt_f32_e32", cmpx_op="v_cmpx_le_f32_e32"))
        expected = depth_mask_aot.lower(decoded, [(6, 32), (10, 16)])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(blob)
            for name in ("nested/depth_mask.hlsl", "UPPER.HLSL"):
                out_path = root / name
                seen = []

                def fake_disassemble(code, llvm_lib, *, offsets=False):
                    seen.append((code, llvm_lib, offsets))
                    return decoded

                with (
                    mock.patch.object(depth_mask_aot, "disassemble", side_effect=fake_disassemble),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    rc = depth_mask_aot.main(
                        [
                            str(container),
                            "--llvm",
                            "mock-llvm.dll",
                            "--table",
                            "6",
                            "0x20",
                            "--table",
                            "10",
                            "16",
                            "--output",
                            str(out_path),
                        ]
                    )
                self.assertEqual(rc, 0)
                self.assertEqual(seen, [(native_code, "mock-llvm.dll", True)])
                self.assertEqual(out_path.read_text(encoding="ascii"), expected)
                self.assertEqual(list(out_path.parent.glob("*.tmp")), [])


class DepthMaskAotNegativeTests(unittest.TestCase):
    def _assert_cli_fails(self, argv: list[str], out_dir: Path, *, expected_code: int | None = None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        before_entries = sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                depth_mask_aot.main(argv)
        self.assertIsInstance(exc.exception.code, int)
        self.assertGreaterEqual(exc.exception.code, 1)
        if expected_code is not None:
            self.assertEqual(exc.exception.code, expected_code)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(stderr.getvalue().strip())
        if out_dir.exists():
            self.assertEqual(list(out_dir.glob("*.tmp")), [])
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), before_entries)
        return stderr.getvalue()

    def test_malformed_dxbc_containers_raise_value_error_and_exit_1(self):
        valid = _build_dxbc([(b"DXIL", _build_dxil_part(S_NOP + S_ENDPGM))])

        # 1) Non-DXBC magic
        bad_magic = b"XXXX" + valid[4:]
        # 2) Truncated header (<32 bytes and mismatched size)
        short_header = b"DXBC" + b"\x00" * 12
        bad_size = bytearray(valid)
        struct.pack_into("<I", bad_size, 24, len(valid) + 64)
        # 3) Out-of-bounds part offsets
        oob_offset = bytearray(valid)
        struct.pack_into("<I", oob_offset, 32, len(valid) - 4)
        low_offset = bytearray(valid)
        struct.pack_into("<I", low_offset, 32, 16)
        # 4) Truncated part payload
        truncated_payload = bytearray(valid)
        offset0 = struct.unpack_from("<I", truncated_payload, 32)[0]
        struct.pack_into("<I", truncated_payload, offset0 + 4, len(valid))

        malformed = [
            ("non_dxbc_magic", bad_magic),
            ("short_header", short_header),
            ("mismatched_size", bytes(bad_size)),
            ("oob_offset_high", bytes(oob_offset)),
            ("oob_offset_low", bytes(low_offset)),
            ("truncated_payload", bytes(truncated_payload)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            out_dir.mkdir()
            for label, payload in malformed:
                with self.subTest(case=label):
                    with self.assertRaises(ValueError):
                        parts(payload)
                    container = root / f"{label}.dxbc"
                    container.write_bytes(payload)
                    out_file = out_dir / f"{label}.hlsl"
                    self._assert_cli_fails(
                        [
                            str(container),
                            "--llvm",
                            "mock-llvm.dll",
                            "--table",
                            "6",
                            "32",
                            "--table",
                            "10",
                            "16",
                            "--output",
                            str(out_file),
                        ],
                        out_dir,
                        expected_code=1,
                    )
                    self.assertFalse(out_file.exists())

    def test_output_path_and_table_cli_validation(self):
        valid = _build_dxbc([(b"DXIL", _build_dxil_part(S_NOP + S_ENDPGM))])
        decoded = _decoded_stream(_depth_program())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.hlsl"
            container.write_bytes(valid)
            out_dir = root / "out"
            out_dir.mkdir()

            with mock.patch.object(depth_mask_aot, "disassemble", return_value=decoded) as disasm_mock:
                for bad_out in (out_dir / "bad.txt", out_dir / "no_suffix", out_dir / "bad.hlsl.bak"):
                    with self.subTest(bad_out=bad_out.name):
                        err = self._assert_cli_fails(
                            [
                                str(container),
                                "--llvm",
                                "mock-llvm.dll",
                                "--table",
                                "6",
                                "32",
                                "--table",
                                "10",
                                "16",
                                "--output",
                                str(bad_out),
                            ],
                            out_dir,
                            expected_code=2,
                        )
                        self.assertIn("Choose an HLSL destination distinct from the input", err)
                        self.assertFalse(bad_out.exists())

                # Output resolving to input container must fail and leave input untouched.
                for same_out in (container, root / "out" / ".." / "shader.hlsl"):
                    with self.subTest(same_out=str(same_out)):
                        err = self._assert_cli_fails(
                            [
                                str(container),
                                "--llvm",
                                "mock-llvm.dll",
                                "--table",
                                "6",
                                "32",
                                "--table",
                                "10",
                                "16",
                                "--output",
                                str(same_out),
                            ],
                            root,
                            expected_code=2,
                        )
                        self.assertIn("Choose an HLSL destination distinct from the input", err)
                        self.assertEqual(container.read_bytes(), valid)

                # Negative --table operands rejected by CLI and lower().
                for bad_tables in (["-1", "32", "10", "16"], ["6", "-4", "10", "16"]):
                    with self.subTest(bad_tables=bad_tables):
                        out_file = out_dir / "neg.hlsl"
                        err = self._assert_cli_fails(
                            [
                                str(container),
                                "--llvm",
                                "mock-llvm.dll",
                                "--table",
                                bad_tables[0],
                                bad_tables[1],
                                "--table",
                                bad_tables[2],
                                bad_tables[3],
                                "--output",
                                str(out_file),
                            ],
                            out_dir,
                            expected_code=2,
                        )
                        self.assertIn(
                            "Expected non-empty --table pairs with non-negative SGPR and byte offset",
                            err,
                        )
                        self.assertFalse(out_file.exists())

                with mock.patch.object(
                    argparse.ArgumentParser,
                    "parse_args",
                    return_value=argparse.Namespace(
                        container=container,
                        llvm="mock-llvm.dll",
                        table=[],
                        output=out_dir / "empty_table.hlsl",
                    ),
                ):
                    err = self._assert_cli_fails([], out_dir, expected_code=2)
                    self.assertIn(
                        "Expected non-empty --table pairs with non-negative SGPR and byte offset",
                        err,
                    )

                disasm_mock.assert_not_called()

        for bad_pair in ([(-1, 32), (10, 16)], [(6, -4), (10, 16)], [(6, 32)]):
            with self.assertRaises(ValueError):
                depth_mask_aot.lower(decoded, bad_pair)

    def test_missing_dxil_and_missing_or_unreadable_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            out_dir.mkdir()

            # Missing DXIL part
            no_dxil = root / "no_dxil.dxbc"
            no_dxil.write_bytes(_build_dxbc([(b"NOTE", b"only-note")]))
            err = self._assert_cli_fails(
                [
                    str(no_dxil),
                    "--llvm",
                    "mock-llvm.dll",
                    "--table",
                    "6",
                    "32",
                    "--table",
                    "10",
                    "16",
                    "--output",
                    str(out_dir / "no_dxil.hlsl"),
                ],
                out_dir,
                expected_code=1,
            )
            self.assertIn("DXIL", err)

            # Missing container (OSError)
            missing = root / "does_not_exist.dxbc"
            self._assert_cli_fails(
                [
                    str(missing),
                    "--llvm",
                    "mock-llvm.dll",
                    "--table",
                    "6",
                    "32",
                    "--table",
                    "10",
                    "16",
                    "--output",
                    str(out_dir / "missing.hlsl"),
                ],
                out_dir,
                expected_code=1,
            )

            # Unreadable container (OSError)
            unreadable = root / "unreadable.dxbc"
            unreadable.write_bytes(b"placeholder")
            with mock.patch.object(Path, "read_bytes", side_effect=PermissionError("access denied")):
                err = self._assert_cli_fails(
                    [
                        str(unreadable),
                        "--llvm",
                        "mock-llvm.dll",
                        "--table",
                        "6",
                        "32",
                        "--table",
                        "10",
                        "16",
                        "--output",
                        str(out_dir / "unreadable.hlsl"),
                    ],
                    out_dir,
                    expected_code=1,
                )
                self.assertIn("access denied", err)

            # Subprocess smoke test on missing container -> exit 1 without Traceback.
            proc = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(TOOL),
                    str(missing),
                    "--llvm",
                    "mock-llvm.dll",
                    "--table",
                    "6",
                    "32",
                    "--table",
                    "10",
                    "16",
                    "--output",
                    str(out_dir / "sub.hlsl"),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip())
            self.assertEqual(list(out_dir.iterdir()), [])

    def test_lower_and_cli_fail_closed_on_unsupported_instructions_and_write_error(self):
        base = _depth_program()
        unsupported_variants = [
            ("unknown_opcode", "v_fract_f32_e32 v7, v7", "v_sin_f32_e32 v7, v7"),
            ("unlisted_e64", "v_cvt_f32_ubyte0_e32 v5, s30", "v_cvt_f32_ubyte0_e64 v5, s30"),
            ("vop3_clamp", "v_fract_f32_e32 v7, v7", "v_fract_f32_e64 v7, v7 clamp"),
            ("vop3_neg", "v_fract_f32_e32 v7, v7", "v_fract_f32_e64 v7, -v7"),
            ("unordered_cmpx", "v_cmpx_lt_f32_e32 s31, v7", "v_cmpx_neq_f32_e32 s31, v7"),
            ("unwritten_fmac_dest", "v_fmac_f32_e32 v7, 0x3d000000, v5", "v_fmac_f32_e32 v9, 0x3d000000, v5"),
        ]
        for label, needle, replacement in unsupported_variants:
            with self.subTest(case=label):
                lines = [replacement if line == needle else line for line in base]
                with self.assertRaises(ValueError):
                    depth_mask_aot.lower(_decoded_stream(lines), TABLES)

        blob = _build_dxbc([(b"DXIL", _build_dxil_part(S_NOP + S_ENDPGM))])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            container = root / "shader.dxbc"
            container.write_bytes(blob)
            out_dir = root / "out"
            out_dir.mkdir()

            bad_decoded = _decoded_stream(["v_sin_f32_e32 v7, v7" if l == "v_fract_f32_e32 v7, v7" else l for l in base])
            with mock.patch.object(depth_mask_aot, "disassemble", return_value=bad_decoded):
                self._assert_cli_fails(
                    [
                        str(container),
                        "--llvm",
                        "mock-llvm.dll",
                        "--table",
                        "6",
                        "32",
                        "--table",
                        "10",
                        "16",
                        "--output",
                        str(out_dir / "bad_insn.hlsl"),
                    ],
                    out_dir,
                    expected_code=1,
                )

            # Temp cleanup on output replace failure leaves pre-existing destination intact and zero .tmp files.
            existing = out_dir / "existing.hlsl"
            existing.write_text("original", encoding="ascii")
            good_decoded = _decoded_stream(base)
            with (
                mock.patch.object(depth_mask_aot, "disassemble", return_value=good_decoded),
                mock.patch.object(Path, "replace", side_effect=OSError("disk full")),
            ):
                err = self._assert_cli_fails(
                    [
                        str(container),
                        "--llvm",
                        "mock-llvm.dll",
                        "--table",
                        "6",
                        "32",
                        "--table",
                        "10",
                        "16",
                        "--output",
                        str(existing),
                    ],
                    out_dir,
                    expected_code=1,
                )
            self.assertIn("disk full", err)
            self.assertEqual(existing.read_text(encoding="ascii"), "original")


if __name__ == "__main__":
    unittest.main()
