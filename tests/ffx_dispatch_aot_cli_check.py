"""Offline CLI and refusal unit suite for tools/ffx_dispatch_aot.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ffx_dispatch_aot.py"
sys.path.insert(0, str(ROOT / "tools"))
import ffx_dispatch_aot  # noqa: E402
from ffx_scratch_reference import simm16  # noqa: E402


INITIALIZER_SHA256 = "a54e552ff69a3f7199861417f7ae461d1f67f636ab532c6ec9d8ea5d2963735a"


def _synthetic_programs():
    pass24 = [
        (0x0, "buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
        (0x4, "v_mov_b32_e32 v0, v0"),
        (0x8, "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
        (0xc, "s_endpgm"),
    ]
    pass25 = [
        (0x0, "s_buffer_load_dwordx2 s[8:9], s[0:3], 0xd0"),
        (0x4, "v_mov_b32_e32 v1, v0"),
        (0x8, "s_endpgm"),
    ]
    producer = [
        (0x14c, "buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
        (0x2880, "v_mov_b32_e32 v10, v0"),
        (0x4370, "v_mov_b32_e32 v11, v0"),
        (0x4ac0, "v_mov_b32_e32 v12, v0"),
        (0x51ec, "v_mov_b32_e32 v13, v0"),
        (0x51f0, "s_endpgm"),
    ]
    rcas = [(i * 8, "image_load v[2:3], v[0:1], s[24:31] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm d16") for i in range(20)]
    rcas.extend(
        (20 * 8 + i * 8, "image_store v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D unorm")
        for i in range(4)
    )
    rcas.append((24 * 8, "s_endpgm"))
    return {
        "pass24": pass24,
        "pass25": pass25,
        "producer": producer,
        "rcas": rcas,
    }


def _populate_synthetic_dir(folder: Path):
    init_bytes = b"synthetic-initializer" + b"\x00" * 235
    bin_bytes = {name: f"synthetic-{name}-code".encode("ascii") for name in ffx_dispatch_aot.DIGEST}
    (folder / "initializer.bin").write_bytes(init_bytes)
    for name, payload in bin_bytes.items():
        (folder / f"{name}.bin").write_bytes(payload)
    digest_map = {
        init_bytes: INITIALIZER_SHA256,
        **{bin_bytes[name]: digest for name, digest in ffx_dispatch_aot.DIGEST.items()},
    }
    return init_bytes, bin_bytes, digest_map


def _fake_sha256_factory(digest_map: dict[bytes, str]):
    real_sha256 = hashlib.sha256

    class _Digest:
        def __init__(self, hex_value: str):
            self._hex = hex_value

        def hexdigest(self) -> str:
            return self._hex

    def _sha256(data: bytes):
        if data in digest_map:
            return _Digest(digest_map[data])
        return real_sha256(data)

    return _sha256


class FfxDispatchAotCliPositiveTests(unittest.TestCase):
    def test_main_generates_all_shaders_and_scratch_reference(self):
        programs = _synthetic_programs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, bin_bytes, digest_map = _populate_synthetic_dir(root)
            reverse_lookup = {payload: name for name, payload in bin_bytes.items()}
            seen_cpus = {}

            def fake_disassemble(code, llvm_lib, *, cpu=b"gfx1030", offsets=False):
                name = reverse_lookup[code]
                seen_cpus[name] = (llvm_lib, cpu, offsets)
                return programs[name]

            stdout = io.StringIO()
            with (
                mock.patch.object(ffx_dispatch_aot, "disassemble", side_effect=fake_disassemble),
                mock.patch.object(ffx_dispatch_aot.hashlib, "sha256", side_effect=_fake_sha256_factory(digest_map)),
                redirect_stdout(stdout),
            ):
                rc = ffx_dispatch_aot.main([str(root), "--llvm", "mock-llvm.dll"])

            self.assertEqual(rc, 0)
            self.assertEqual(
                seen_cpus,
                {
                    "pass24": ("mock-llvm.dll", b"gfx1030", True),
                    "pass25": ("mock-llvm.dll", b"gfx1030", True),
                    "producer": ("mock-llvm.dll", b"gfx1030", True),
                    "rcas": ("mock-llvm.dll", b"gfx1013", True),
                },
            )
            pass24_hlsl = (root / "pass24.hlsl").read_text(encoding="ascii")
            pass25_hlsl = (root / "pass25.hlsl").read_text(encoding="ascii")
            producer_hlsl = (root / "producer.hlsl").read_text(encoding="ascii")
            rcas_hlsl = (root / "rcas.hlsl").read_text(encoding="ascii")
            scratch_ref = (root / "scratch_reference.h").read_text(encoding="ascii")

            self.assertIn("[numthreads(64,1,1)]", pass24_hlsl)
            self.assertIn("work.Load4(", pass24_hlsl)
            self.assertIn("work.Store4(", pass24_hlsl)

            self.assertIn("[numthreads(32,1,1)]", pass25_hlsl)
            self.assertIn("s8=4; s9=4;", pass25_hlsl)
            self.assertIn("if (exec_lo)", pass25_hlsl)

            self.assertIn("[numthreads(256,1,1)]", producer_hlsl)
            self.assertIn("[RootSignature(PRODUCER_ROOT)]", producer_hlsl)
            self.assertIn("audit_load(", producer_hlsl)

            self.assertIn("[numthreads(64,1,1)]", rcas_hlsl)
            self.assertIn("[RootSignature(CONSUMER_ROOT)]", rcas_hlsl)
            self.assertIn("load16(", rcas_hlsl)
            self.assertIn("finalOutput[", rcas_hlsl)

            self.assertIn("static constexpr uint32_t before_padding[]", scratch_ref)
            self.assertIn("static constexpr int history_logits[]", scratch_ref)


class FfxDispatchAotCliRefusalTests(unittest.TestCase):
    def _assert_main_exit_1(self, argv: list[str], expected_msg: str = ""):
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                ffx_dispatch_aot.main(argv)
        self.assertEqual(exc.exception.code, 1)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(stderr.getvalue().strip())
        if expected_msg:
            self.assertIn(expected_msg, stderr.getvalue())

    def test_missing_directory_and_files_exit_1_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_dir = root / "missing_dir"
            self._assert_main_exit_1([str(missing_dir), "--llvm", "mock-llvm.dll"])

            # Real subprocess smoke test on missing directory -> exit 1 without Traceback.
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(missing_dir), "--llvm", "mock-llvm.dll"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip())

            # Missing pass24.bin after initializer passes digest check.
            empty_prog_dir = root / "missing_program"
            empty_prog_dir.mkdir()
            init_bytes, _, digest_map = _populate_synthetic_dir(empty_prog_dir)
            self.assertTrue(init_bytes)
            (empty_prog_dir / "pass24.bin").unlink()
            with mock.patch.object(ffx_dispatch_aot.hashlib, "sha256", side_effect=_fake_sha256_factory(digest_map)):
                self._assert_main_exit_1([str(empty_prog_dir), "--llvm", "mock-llvm.dll"])

    def test_tampered_initializer_and_program_digests_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, bin_bytes, digest_map = _populate_synthetic_dir(root)

            # Tamper initializer.bin without mocking sha256 -> real SHA-256 mismatch.
            (root / "initializer.bin").write_bytes(b"tampered-initializer")
            self._assert_main_exit_1(
                [str(root), "--llvm", "mock-llvm.dll"],
                expected_msg="Unknown model initializer",
            )

            # Subprocess check on tampered initializer -> exit 1 without Traceback.
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(root), "--llvm", "mock-llvm.dll"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("Unknown model initializer", proc.stderr)

        # Tamper each program binary independently.
        for bad_name in ffx_dispatch_aot.DIGEST:
            with self.subTest(program=bad_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, bin_bytes, digest_map = _populate_synthetic_dir(root)
                reverse_lookup = {payload: name for name, payload in bin_bytes.items()}
                programs = _synthetic_programs()
                (root / f"{bad_name}.bin").write_bytes(b"tampered-code")
                with (
                    mock.patch.object(
                        ffx_dispatch_aot,
                        "disassemble",
                        side_effect=lambda code, llvm, **kw: programs[reverse_lookup[code]],
                    ),
                    mock.patch.object(
                        ffx_dispatch_aot.hashlib,
                        "sha256",
                        side_effect=_fake_sha256_factory(digest_map),
                    ),
                ):
                    self._assert_main_exit_1(
                        [str(root), "--llvm", "mock-llvm.dll"],
                        expected_msg=f"Unknown {bad_name} machine code",
                    )

    def test_wrong_rcas_image_access_counts_exit_1(self):
        cases = (
            ("too_few_loads", 19, 4),
            ("too_few_stores", 20, 3),
            ("extra_store", 20, 5),
        )
        for label, loads, stores in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, bin_bytes, digest_map = _populate_synthetic_dir(root)
                reverse_lookup = {payload: name for name, payload in bin_bytes.items()}
                programs = _synthetic_programs()
                bad_rcas = [
                    (i * 8, "image_load v[2:3], v[0:1], s[24:31] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm d16")
                    for i in range(loads)
                ]
                bad_rcas.extend(
                    ((loads + i) * 8, "image_store v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D unorm")
                    for i in range(stores)
                )
                bad_rcas.append(((loads + stores) * 8, "s_endpgm"))
                programs["rcas"] = bad_rcas

                with (
                    mock.patch.object(
                        ffx_dispatch_aot,
                        "disassemble",
                        side_effect=lambda code, llvm, **kw: programs[reverse_lookup[code]],
                    ),
                    mock.patch.object(
                        ffx_dispatch_aot.hashlib,
                        "sha256",
                        side_effect=_fake_sha256_factory(digest_map),
                    ),
                ):
                    self._assert_main_exit_1(
                        [str(root), "--llvm", "mock-llvm.dll"],
                        expected_msg="Expected twenty integer texture loads and four output stores",
                    )

    def test_assign_and_image_op_refusals(self):
        with self.assertRaises(ValueError):
            ffx_dispatch_aot.assign(["v0", "v1"], ["0x1u"])

        with self.assertRaises(ValueError):
            ffx_dispatch_aot.image_op("rcas", "image_sample", ["v[0:3]", "v[4:5]", "s[24:31]"], "dmask:0xf")

        for bad_load in (
            (["v[2:3]", "v[0:1]", "s[32:39]"], "dmask:0x7 d16"),
            (["v[2:3]", "v[0:1]", "s[24:31]"], "dmask:0xf d16"),
            (["v[2:3]", "v[0:1]", "s[24:31]"], "dmask:0x7"),
        ):
            with self.assertRaises(ValueError):
                ffx_dispatch_aot.image_op("rcas", "image_load", *bad_load)

        with self.assertRaises(ValueError):
            ffx_dispatch_aot.image_op("rcas", "image_store", ["v[4:7]", "v[0:1]", "s[40:47]"], "dmask:0xf")

    def test_alu_predecessor_translate_and_simm16_refusals(self):
        alu_bad_cases = (
            ("v_pk_add_f16", ["v0", "v1", "v2"], "clamp clamp"),
            ("v_pk_add_f16", ["v0", "v1", "v2"], "op_sel:[0,1,0]"),
            ("v_pk_add_f16", ["v0", "v1", "v2"], "op_sel:[0,1] op_sel:[1,0]"),
            ("v_pk_add_f16", ["v0", "v1", "v2"], "neg_lo:[1,0] bogus"),
            ("v_pk_add_u16", ["v0", "v1", "v2"], "clamp"),
            ("v_add_b32_sdwa", ["v0", "v1", "v2"], ""),
            ("v_fma_mix_f32", ["v0", "v1", "v2", "v3"], "op_sel_hi:[0,1,0]"),
            ("v_fma_mix_f32", ["v0", "v1", "v2", "neg(0)"], ""),
            ("s_lshrrev_b64", ["s[0:1]", "1", "s[2:3]"], ""),
            ("s_movk_i32", ["s0", "1", "2"], ""),
            ("v_dot4_i32_i8", ["v0", "v1", "v2", "v3"], "op_sel:[1,1,1]"),
            ("v_add_f32_e32", ["v0", "v1", "v2"], "clamp"),
            ("v_add_f32_e32", ["v0", "v1", "v2"], "mul:4"),
            ("v_add_f32_e32", ["v0", "v1", "v2"], "div:2"),
            ("v_sqrt_f32_e64", ["v0", "v1", "v2"], ""),
            ("v_add_nc_u32_e64", ["v0", "v1", "v2"], "clamp"),
        )
        for mn, tokens, mods in alu_bad_cases:
            with self.subTest(mn=mn, mods=mods):
                with self.assertRaises(ValueError):
                    ffx_dispatch_aot.alu(mn, tokens, mods)

        with self.assertRaises(ValueError):
            ffx_dispatch_aot.predecessor(
                "pass24",
                [(0x0, "s_buffer_load_dwordx2 s[8:9], s[0:3], 0xd0"), (0x4, "s_endpgm")],
                b"",
            )
        with self.assertRaises(ValueError):
            ffx_dispatch_aot.predecessor(
                "pass25",
                [(0x0, "s_buffer_load_dwordx2 s[8:9], s[0:3], 0xd8"), (0x4, "s_endpgm")],
                b"",
            )

        with self.assertRaises(ValueError):
            ffx_dispatch_aot.translate(
                "producer",
                [(0x100, "buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen"), (0x108, "s_endpgm")],
                b"",
            )
        with self.assertRaises(ValueError):
            ffx_dispatch_aot.translate(
                "producer",
                [(0x2410, "s_branch 4"), (0x2414, "s_endpgm")],
                b"",
            )

        for bad_imm in ("0x10000", "-32769", "65536", "not_a_number"):
            with self.subTest(imm=bad_imm):
                with self.assertRaises(ValueError):
                    simm16(bad_imm)


if __name__ == "__main__":
    unittest.main()
