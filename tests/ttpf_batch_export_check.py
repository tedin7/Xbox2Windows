"""Unit and regression tests for ttpf_batch_export."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import bc7_blocks
import bcpack_mode6_vector as mode6
import ttpf_batch_export as batch


def make_synthetic_ttpf_archive() -> bytearray:
    """Generate a minimal valid 12544-byte RDS5/ETPF pool containing BC1 and BC7 entries."""
    archive = bytearray(12544)
    count = 4  # 2 chunks + 2 sentinels/boundaries
    virtual_size = 12288
    size_word = (22 << 56) | virtual_size
    struct.pack_into("<4sIQ", archive, 0, b"RDS5", count, size_word)
    struct.pack_into("<II", archive, 16, 1, 0)

    # Table entries: chunk 0, chunk 1, sentinel
    c0 = (1 << 30) | 17  # virt 4096 (1), stored 4352 (17), codec 0
    c1 = (2 << 30) | 33  # virt 8192 (2), stored 8448 (33), codec 0
    c2 = (3 << 30) | 49  # virt 12288 (3), stored 12544 (49)
    struct.pack_into("<QQQ", archive, 24, c0, c1, c2)

    # ETPF at offset 256: version 634, 2 textures, 32 bytes records area
    struct.pack_into("<4sIII", archive, 256, b"ETPF", 634, 2, 32)

    # Texture 0: 4x4, BC1 (format 71), base page 1 (4096), 1 mip, mode 4
    a0 = 3 | (3 << 14)
    b0 = (4 << 23) | (1 << 21) | (71 << 3) | 1
    struct.pack_into("<4I", archive, 272, a0, b0, 1, 0)

    # Texture 1: 4x4, BC7 (format 98), base page 2 (8192), 1 mip, mode 4
    a1 = 3 | (3 << 14)
    b1 = (4 << 23) | (1 << 21) | (98 << 3) | 1
    struct.pack_into("<4I", archive, 288, a1, b1, 2, 0)

    # Texture IDs table (32 bytes per texture)
    archive[304:336] = b"tex_0_bc1\x00" + b"\x00" * 22
    archive[336:368] = b"tex_1_bc7\x00" + b"\x00" * 22

    # Chunk 0 payload at 4352: BC1 block (Red: color0=0xF800, color1=0x0000, all 16 indices=0)
    archive[4352:4352 + 8] = struct.pack("<HHI", 0xF800, 0x0000, 0x00000000)

    # Chunk 1 payload at 8448: Mode-6 BC7 block pinned in bcpack_mode6_vector
    archive[8448:8448 + 16] = mode6.BLOCK
    return archive


def decode_bc1_block(block: bytes) -> bytes:
    """Decode an 8-byte BC1 block into 64 bytes of RGBA8 pixels."""
    c0, c1, bits = struct.unpack("<HHI", block)
    r0 = ((c0 >> 11) & 31) * 255 // 31
    g0 = ((c0 >> 5) & 63) * 255 // 63
    b0 = (c0 & 31) * 255 // 31
    r1 = ((c1 >> 11) & 31) * 255 // 31
    g1 = ((c1 >> 5) & 63) * 255 // 63
    b1 = (c1 & 31) * 255 // 31
    palette = [
        (r0, g0, b0, 255),
        (r1, g1, b1, 255),
        ((2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3, 255)
        if c0 > c1 else ((r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2, 255),
        ((r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3, 255)
        if c0 > c1 else (0, 0, 0, 0),
    ]
    pixels = bytearray()
    for i in range(16):
        idx = (bits >> (2 * i)) & 3
        pixels.extend(palette[idx])
    return bytes(pixels)


class MockPool:
    def __init__(self, path: str):
        self.path = path
        self.textures = 6

    def close(self):
        pass


class MockTtpfModule:
    BC_FORMATS = {71, 98}

    @staticmethod
    def Pool(path: str):
        return MockPool(path)

    @staticmethod
    def analyze(pool, index: int) -> dict:
        if index == 0:
            # Valid BC1 texture
            return {
                "layout": {"size": 131072},
                "available": [True],
                "full_chain": True,
                "descriptor": {"format": 71, "width": 512, "height": 512, "mips": 1},
                "shape": "2d",
            }
        elif index == 1:
            # Valid BC7 texture requiring provisional calibration
            return {
                "layout": {"size": 131072},
                "available": [True],
                "full_chain": True,
                "descriptor": {"format": 98, "width": 512, "height": 256, "mips": 1},
                "shape": "2d",
            }
        elif index == 2:
            # Uncompressed format
            return {
                "layout": {"size": 65536},
                "available": [True],
                "full_chain": True,
                "descriptor": {"format": 28, "width": 128, "height": 128, "mips": 1},
                "shape": "2d",
            }
        elif index == 3:
            # Incomplete codec chain
            return {
                "layout": {"size": 65536},
                "available": [False],
                "full_chain": False,
                "descriptor": {"format": 71, "width": 256, "height": 256, "mips": 1},
                "shape": "2d",
            }
        elif index == 4:
            # Missing layout / blocked
            return {"blocked": "allocation span mismatch"}
        else:
            # Broken export
            return {
                "layout": {"size": 131072},
                "available": [True],
                "full_chain": True,
                "descriptor": {"format": 98, "width": 256, "height": 256, "mips": 1},
                "shape": "2d",
            }

    @staticmethod
    def export(pool, index: int, out_path: str, *, allow_provisional: bool = False, history=None) -> int:
        target = Path(out_path)
        if index == 0:
            # Strict mode succeeds
            target.write_bytes(b"DDS_STRICT_BC1_DATA")
            return 0
        elif index == 1:
            # Fails in strict, succeeds with provisional
            if allow_provisional:
                target.write_bytes(b"DDS_PROVISIONAL_BC7_DATA")
                return 0
            return 2
        elif index == 5:
            # Fail after writing partial output to exercise temporary cleanup.
            target.write_bytes(b"PARTIAL_DDS")
            return 2
        return 0


class BatchExportTests(unittest.TestCase):
    def test_parse_indices(self):
        self.assertEqual(batch.parse_indices("0,8253"), [0, 8253])
        self.assertEqual(batch.parse_indices("0-3, 5, 10-12"), [0, 1, 2, 3, 5, 10, 11, 12])
        self.assertEqual(batch.parse_indices("0-10", total_count=4), [0, 1, 2, 3])
        self.assertEqual(batch.parse_indices("  2, , 4 "), [2, 4])
        with self.assertRaises(ValueError):
            batch.parse_indices("10-2")

    def test_malformed_indices_are_usage_errors(self):
        cases = {
            "5-3": "Invalid range '5-3'",
            "1-": "Invalid index '' in '1-'",
            "abc": "Invalid index 'abc' in 'abc'",
            "-2": "Invalid index '' in '-2'",
        }
        self.assertEqual(batch.parse_indices("0-1,3"), [0, 1, 3])
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / "empty.ttpf"
            pool.write_bytes(b"")
            for spec, expected in cases.items():
                out_dir = Path(tmp) / "out"
                with self.subTest(spec=spec):
                    proc = subprocess.run(
                        [sys.executable, "-B", str(ROOT / "tools" / "ttpf_batch_export.py"),
                         str(pool), "-o", str(out_dir), f"--indices={spec}"],
                        capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
                    self.assertEqual(proc.returncode, 2, proc.stderr)
                    self.assertIn("--indices:", proc.stderr)
                    self.assertIn(expected, proc.stderr)
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertFalse(out_dir.exists())

    def test_export_texture_entry_strict_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pool = MockPool("mock.ttpf")
            # Strict mode by default
            res = batch.export_texture_entry(pool, 0, out_dir, ttpf_module=MockTtpfModule)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["mode"], "strict")
            self.assertEqual(res["format_name"], "BC1")
            self.assertEqual(res["sha256"], hashlib.sha256(b"DDS_STRICT_BC1_DATA").hexdigest())
            final_file = out_dir / "tex_00000.dds"
            self.assertTrue(final_file.exists())
            # Ensure no leftover temporary files
            tmp_files = list(out_dir.glob(".tmp_*"))
            self.assertEqual(len(tmp_files), 0)

    def test_export_texture_entry_provisional_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pool = MockPool("mock.ttpf")
            # Default allow_provisional=False -> fails
            res_strict = batch.export_texture_entry(pool, 1, out_dir, ttpf_module=MockTtpfModule)
            self.assertEqual(res_strict["status"], "failed")
            self.assertFalse((out_dir / "tex_00001.dds").exists())

            # Explicit opt-in allow_provisional=True -> succeeds
            res_prov = batch.export_texture_entry(pool, 1, out_dir, allow_provisional=True,
                                                  ttpf_module=MockTtpfModule)
            self.assertEqual(res_prov["status"], "success")
            self.assertEqual(res_prov["mode"], "provisional")
            self.assertEqual(res_prov["format_name"], "BC7")
            self.assertEqual(res_prov["sha256"], hashlib.sha256(b"DDS_PROVISIONAL_BC7_DATA").hexdigest())
            self.assertTrue((out_dir / "tex_00001.dds").exists())

    def test_requested_provisional_history_is_read_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, history = root / "pool.ttpf", root / "history.json"
            pool.write_bytes(make_synthetic_ttpf_archive())
            dds, manifest = root / "tex_00001.dds", root / "manifest.json"
            dds.write_bytes(b"previous DDS")
            manifest.write_bytes(b"previous manifest")
            command = [sys.executable, "-B", batch.__file__, pool, "-o", root,
                       "--indices", "1", "--allow-provisional", "--history", history]
            for data in (None, b"{", b"\xff", b"null", b"[]"):
                with self.subTest(data=data):
                    if data is not None:
                        history.write_bytes(data)
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("--history " + str(history), result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(dds.read_bytes(), b"previous DDS")
                    self.assertEqual(manifest.read_bytes(), b"previous manifest")

            with patch.object(batch, "load_tools", return_value=MockTtpfModule), patch.object(
                MockTtpfModule, "export", wraps=MockTtpfModule.export
            ) as export:
                for profile in ({}, {"bc7": {"historyIndices": [2, 5]}}):
                    history.write_text(json.dumps(profile), encoding="utf-8")
                    summary = batch.batch_export(pool, root, indices=[1], allow_provisional=True, history_path=history)
                    self.assertEqual(summary["stats"]["provisional"], 1)
                    self.assertEqual(export.call_args.kwargs["history"], profile)

    def test_require_rust_checks_selected_tools_and_kernels(self):
        library = SimpleNamespace(
            unpack_lanes=object(), decode_strict_bc1=object(),
            decode_strict_mode6=object(), rebuild_calibrated_bc1=object(),
        )
        native = SimpleNamespace(library=Mock(return_value=library))
        tools = SimpleNamespace(bcpack=SimpleNamespace(_native=native))
        batch.require_rust_backend(tools)
        native.library.assert_called_once_with()
        del library.decode_strict_mode6
        with self.assertRaisesRegex(RuntimeError, "Could not load the Rust BCPack kernels"):
            batch.require_rust_backend(tools)
        with self.assertRaisesRegex(RuntimeError, "do not include the optional Rust BCPack backend"):
            batch.require_rust_backend(MockTtpfModule)

    def test_local_profile_retries_strict_failures_and_records_current_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out_dir = tmp / "dds_out"
            out_dir.mkdir()
            existing_dds = out_dir / "tex_00005.dds"
            existing_dds.write_bytes(b"PREVIOUS_DDS")
            profile = tmp / "profile.json"
            profile.write_text('{"version": 1}', encoding="utf-8")
            fitted = b"DDS_FITTED_DATA"
            decoder = Mock()
            decoder.dds.side_effect = lambda index: {1: fitted}[index]
            create_decoder = Mock(return_value=decoder)
            tools = SimpleNamespace(
                __file__=str(tmp / "ttpf_mips.py"),
                Pool=MockTtpfModule.Pool,
                analyze=MockTtpfModule.analyze,
                export=MockTtpfModule.export,
                BC_FORMATS=MockTtpfModule.BC_FORMATS,
            )
            calibrated = SimpleNamespace(
                __file__=str(tmp / "calibrated_texture.py"), ttpf=tools,
                native=Mock(), LocalDecoder=create_decoder,
            )
            with patch.object(batch, "load_tools", return_value=tools), patch.dict(
                sys.modules, {"calibrated_texture": calibrated}
            ):
                summary = batch.batch_export(tmp / "mock.ttpf", out_dir, indices=[0, 1, 5], threads=1,
                                             calibration_path=profile)
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    batch.batch_export(tmp / "mock.ttpf", out_dir, allow_provisional=True,
                                       calibration_path=profile)

            calibrated.native.assert_called_once_with()
            self.assertEqual([call.args for call in decoder.dds.call_args_list], [(1,), (5,)])
            self.assertEqual(decoder.read_chunk.cache_clear.call_count, 2)
            self.assertEqual(summary["stats"], {"strict": 1, "provisional": 0, "fitted": 1,
                                                "failed": 1, "skipped": 0})
            strict, fitted_row, failed = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(strict["mode"], "strict")
            self.assertEqual(fitted_row["mode"], "fitted")
            self.assertEqual(fitted_row["sha256"], hashlib.sha256(fitted).hexdigest())
            self.assertEqual((out_dir / fitted_row["file"]).read_bytes(), fitted)
            self.assertEqual(failed["status"], "failed")
            self.assertNotIn("file", failed)
            self.assertNotIn("sha256", failed)
            self.assertEqual(existing_dds.read_bytes(), b"PREVIOUS_DDS")
            self.assertEqual(list(out_dir.glob(".tmp_*")), [])

    def test_export_texture_entry_atomic_publish_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pool = MockPool("mock.ttpf")
            # Failed export (index 5)
            res = batch.export_texture_entry(pool, 5, out_dir, ttpf_module=MockTtpfModule)
            self.assertEqual(res["status"], "failed")
            self.assertFalse((out_dir / "tex_00005.dds").exists())
            # No temporary file leaked
            self.assertEqual(len(list(out_dir.glob(".tmp_*"))), 0)

    def test_analysis_failure_preserves_existing_dds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            existing_dds = out_dir / "tex_00000.dds"
            original_bytes = b"PREVIOUS_DDS"
            existing_dds.write_bytes(original_bytes)
            with (
                patch.object(MockTtpfModule, "analyze", side_effect=ValueError("invalid descriptor")),
                patch.object(MockTtpfModule, "export") as export,
            ):
                result = batch.export_texture_entry(MockPool("mock.ttpf"), 0, out_dir,
                                                    ttpf_module=MockTtpfModule)
            export.assert_not_called()
            self.assertEqual(result, {"index": 0, "status": "failed", "reason": "invalid descriptor"})
            self.assertEqual(existing_dds.read_bytes(), original_bytes)

    def test_export_texture_entry_uncompressed_and_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pool = MockPool("mock.ttpf")
            res_uncompressed = batch.export_texture_entry(pool, 2, out_dir, ttpf_module=MockTtpfModule)
            self.assertEqual(res_uncompressed["status"], "uncompressed")

            res_codec = batch.export_texture_entry(pool, 3, out_dir, ttpf_module=MockTtpfModule)
            self.assertEqual(res_codec["status"], "unsupported_codec")

            res_layout = batch.export_texture_entry(pool, 4, out_dir, ttpf_module=MockTtpfModule)
            self.assertEqual(res_layout["status"], "no_layout")

    def test_manifest_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "dds_out"
            manifest_file = out_dir / "manifest.json"
            dummy_pool = Path(tmp) / "test.ttpf"
            dummy_pool.write_bytes(b"DUMMY_POOL")

            orig_load = batch.load_tools
            try:
                batch.load_tools = lambda *args, **kwargs: MockTtpfModule
                summary = batch.batch_export(
                    pool_path=dummy_pool,
                    out_dir=out_dir,
                    indices=[0, 1, 2, 4],
                    allow_provisional=True,
                    manifest_path=manifest_file,
                )
                self.assertEqual(summary["total_textures"], 6)
                self.assertEqual(summary["processed"], 4)
                self.assertEqual(summary["stats"]["strict"], 1)
                self.assertEqual(summary["stats"]["provisional"], 1)
                self.assertEqual(summary["stats"]["skipped"], 2)

                self.assertTrue(manifest_file.exists())
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                self.assertEqual(len(manifest), 4)
                self.assertEqual([m["index"] for m in manifest], [0, 1, 2, 4])
                self.assertEqual(manifest[0]["mode"], "strict")
                self.assertEqual(manifest[1]["mode"], "provisional")
            finally:
                batch.load_tools = orig_load

    def test_manifest_cannot_replace_a_requested_dds(self):
        for index in (0, 1, 2):
            for existing in (False, True):
                with self.subTest(index=index, existing=existing), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    pool = root / "pool.ttpf"
                    pool.write_bytes(b"DUMMY_POOL")
                    out_dir = root / "out"
                    dds = out_dir / f"tex_{index:05d}.dds"
                    if existing:
                        out_dir.mkdir()
                        dds.write_bytes(b"previous dds")
                    with patch.object(batch, "load_tools", return_value=MockTtpfModule), \
                            patch.object(MockTtpfModule, "export", wraps=MockTtpfModule.export) as export:
                        with self.assertRaisesRegex(ValueError, "manifest path collides with DDS"):
                            batch.batch_export(pool, out_dir, indices=[index], manifest_path=dds)
                    export.assert_not_called()
                    self.assertEqual(pool.read_bytes(), b"DUMMY_POOL")
                    if existing:
                        self.assertEqual(dds.read_bytes(), b"previous dds")
                    else:
                        self.assertFalse(out_dir.exists())

    def test_manifest_dds_name_in_a_different_directory_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool.ttpf"
            pool.write_bytes(b"DUMMY_POOL")
            out_dir = root / "out"
            manifest = root / "tex_00000.dds"
            with patch.object(batch, "load_tools", return_value=MockTtpfModule):
                summary = batch.batch_export(pool, out_dir, indices=[0], manifest_path=manifest)
            self.assertEqual(summary["stats"]["strict"], 1)
            row, = json.loads(manifest.read_text())
            dds = out_dir / row["file"]
            self.assertEqual(dds.read_bytes(), b"DDS_STRICT_BC1_DATA")
            self.assertEqual(row["sha256"], hashlib.sha256(dds.read_bytes()).hexdigest())

    def test_manifest_cannot_replace_a_history_or_calibration_input(self):
        for option in ("history", "calibration"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pool = root / "pool.ttpf"
                pool.write_bytes(b"DUMMY_POOL")
                profile = root / "profile.json"
                original = b'{"version": 1}'
                profile.write_bytes(original)
                out_dir = root / "out"
                tools = SimpleNamespace(
                    __file__=str(root / "ttpf_mips.py"), Pool=MockTtpfModule.Pool,
                    analyze=MockTtpfModule.analyze, export=MockTtpfModule.export,
                    BC_FORMATS=MockTtpfModule.BC_FORMATS,
                )
                calibrated = SimpleNamespace(
                    __file__=str(root / "calibrated_texture.py"), ttpf=tools,
                    native=Mock(), LocalDecoder=Mock(),
                )
                kwargs = ({"history_path": profile, "allow_provisional": True} if option == "history"
                          else {"calibration_path": profile})
                with patch.object(batch, "load_tools", return_value=tools) as load, patch.dict(
                    sys.modules, {"calibrated_texture": calibrated}
                ):
                    with self.assertRaisesRegex(ValueError, f"manifest path collides with --{option}"):
                        batch.batch_export(pool, out_dir, indices=[0], manifest_path=profile, **kwargs)
                load.assert_not_called()
                self.assertEqual(profile.read_bytes(), original)
                self.assertFalse(out_dir.exists())

    def test_export_fitted_then_strict_reused_directory(self):
        """A failed strict export keeps the prior DDS outside the current manifest."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "reused_dds"
            manifest_file = out_dir / "manifest.json"
            dummy_pool = Path(tmp) / "test.ttpf"
            dummy_pool.write_bytes(b"DUMMY_POOL")
            tex1_file = out_dir / "tex_00001.dds"

            orig_load = batch.load_tools
            try:
                batch.load_tools = lambda *args, **kwargs: MockTtpfModule

                # 1. First run: provisional opt-in produces fitted DDS
                summary1 = batch.batch_export(
                    pool_path=dummy_pool,
                    out_dir=out_dir,
                    indices=[0, 1],
                    allow_provisional=True,
                    manifest_path=manifest_file,
                )
                self.assertEqual(summary1["stats"]["strict"], 1)
                self.assertEqual(summary1["stats"]["provisional"], 1)
                fitted = tex1_file.read_bytes()
                manifest1 = json.loads(manifest_file.read_text(encoding="utf-8"))
                self.assertEqual(manifest1[1]["mode"], "provisional")

                # 2. Second run: strict-by-default on the same directory
                summary2 = batch.batch_export(
                    pool_path=dummy_pool,
                    out_dir=out_dir,
                    indices=[0, 1],
                    allow_provisional=False,
                    manifest_path=manifest_file,
                )
                self.assertEqual(summary2["stats"]["strict"], 1)
                self.assertEqual(summary2["stats"]["failed"], 1)

                self.assertEqual(tex1_file.read_bytes(), fitted)
                manifest2 = json.loads(manifest_file.read_text(encoding="utf-8"))
                failed_entry = manifest2[1]
                self.assertEqual(failed_entry["index"], 1)
                self.assertEqual(failed_entry["status"], "failed")
                self.assertEqual(failed_entry["reason"], "export returned exit code 2")
                self.assertNotIn("file", failed_entry)
                self.assertNotIn("sha256", failed_entry)
            finally:
                batch.load_tools = orig_load

    def test_real_exporter_dds_and_pixel_verification(self):
        """Verify real pinned exporter with generated synthetic TTPF input and independent pixel decoding."""
        tools_path = None
        if not (os.environ.get("TTPF_TOOLS_DIR") or os.environ.get("BCPACK_REFERENCE_TOOLS")):
            tools_path = ROOT / "out" / "Leonida" / "tools"
            if not (tools_path / "ttpf_mips.py").is_file():
                self.skipTest("optional local ttpf tools directory")
        tools = batch.load_tools(tools_path)
        self.assertIsNotNone(tools, "tools module could not be loaded")

        archive = make_synthetic_ttpf_archive()
        with tempfile.TemporaryDirectory() as tmp:
            pool_path = Path(tmp) / "synthetic.ttpf"
            pool_path.write_bytes(archive)
            out_dir = Path(tmp) / "dds_out"

            # Batch export both entries in strict mode
            summary = batch.batch_export(
                pool_path=pool_path,
                out_dir=out_dir,
                indices=[0, 1],
                allow_provisional=False,
                tools_path=tools_path,
            )
            self.assertEqual(summary["total_textures"], 2)
            self.assertEqual(summary["processed"], 2)
            self.assertEqual(summary["stats"]["strict"], 2)
            self.assertEqual(summary["stats"]["failed"], 0)

            # 1. Independent verification of BC1 texture (entry 0)
            dds0_path = out_dir / "tex_00000.dds"
            self.assertTrue(dds0_path.exists())
            dds0_bytes = dds0_path.read_bytes()
            self.assertEqual(dds0_bytes[:4], b"DDS ")
            self.assertEqual(struct.unpack_from("<I", dds0_bytes, 128)[0], 71)  # DXGI_FORMAT_BC1_UNORM
            bc1_payload = dds0_bytes[148:]
            self.assertEqual(len(bc1_payload), 8)
            decoded_bc1 = decode_bc1_block(bc1_payload)
            expected_bc1_pixels = bytes([255, 0, 0, 255] * 16)
            self.assertEqual(decoded_bc1, expected_bc1_pixels)

            # 2. Independent verification of BC7 texture (entry 1)
            dds1_path = out_dir / "tex_00001.dds"
            self.assertTrue(dds1_path.exists())
            dds1_bytes = dds1_path.read_bytes()
            self.assertEqual(dds1_bytes[:4], b"DDS ")
            self.assertEqual(struct.unpack_from("<I", dds1_bytes, 128)[0], 98)  # DXGI_FORMAT_BC7_UNORM
            bc7_payload = dds1_bytes[148:]
            self.assertEqual(len(bc7_payload), 16)
            self.assertEqual(bc7_payload, mode6.BLOCK)
            decoded_bc7 = bc7_blocks.decode(bc7_payload)
            self.assertEqual(decoded_bc7, mode6.RGBA)

            # Verify manifest
            manifest_file = out_dir / "manifest.json"
            self.assertTrue(manifest_file.exists())
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest[0]["mode"], "strict")
            self.assertEqual(manifest[0]["format_name"], "BC1")
            self.assertEqual(manifest[1]["mode"], "strict")
            self.assertEqual(manifest[1]["format_name"], "BC7")

            # Negative controls:
            # 1. Mutated BC1 color fails pixel comparison
            mutated_bc1 = struct.pack("<HHI", 0x001F, 0x0000, 0x00000000)
            self.assertNotEqual(decode_bc1_block(mutated_bc1), expected_bc1_pixels)

            # 2. Mutated BC7 selector fails pixel comparison
            mutated_bc7 = mode6.BLOCK[:-1] + bytes([mode6.BLOCK[-1] & 0x0F])
            self.assertNotEqual(bc7_blocks.decode(mutated_bc7), mode6.RGBA)

            # 3. Out-of-bounds index records failure without writing DDS
            pool_obj = tools.Pool(str(pool_path))
            try:
                res_oob = batch.export_texture_entry(pool_obj, 99, out_dir, ttpf_module=tools)
                self.assertEqual(res_oob["status"], "failed")
                self.assertFalse((out_dir / "tex_00099.dds").exists())
            finally:
                pool_obj.close()


if __name__ == "__main__":
    unittest.main()
