"""Unit and CLI suite for tools/mesh_od_pipeline.py."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "mesh_od_pipeline.py"
sys.path.insert(0, str(ROOT / "tools"))
import mesh_od_pipeline as tool  # noqa: E402


def _make_record(
    variant="control",
    family="control",
    size=0,
    optimization="default",
    compile_outcome="dxil",
    draws=None,
    controls=None,
):
    if draws is None:
        draws = [
            {
                "outcome": "pass",
                "detail": "",
                "exit_code": 0,
                "wall_s": 0.125,
                "peak_mib": 12.5,
                "pipeline_ms": 1.25,
            }
        ]
    if controls is None:
        controls = {"wrong_seed": "mismatch", "wrong_reference": "mismatch"}
    return {
        "variant": variant,
        "family": family,
        "size": size,
        "optimization": optimization,
        "hlsl_bytes": 512,
        "expected": [f"0x{v:08x}" for v in tool.expected_values(family, size)],
        "compile_outcome": compile_outcome,
        "compile_s": 0.25,
        "compile_peak_mib": 24.0,
        "compile_diagnostic": "" if compile_outcome == "dxil" else "error: synthetic",
        "dxil_bytes": 1024 if compile_outcome == "dxil" else 0,
        "command": "dxc -T ms_6_5 -E main",
        "draws": draws,
        "controls": controls,
    }


class MeshOdPipelinePositiveTests(unittest.TestCase):
    def test_expected_values_deterministic_across_families(self):
        self.assertEqual(
            tool.expected_values("control", 0),
            [tool.SEED, tool.SEED, tool.SEED],
        )
        self.assertEqual(
            tool.expected_values("control", 0, seed=0x1_0000_0005),
            [5, 5, 5],
        )

        # Deterministic across repeated calls and matching manual step math.
        for family, size in (("chain", 4), ("branches", 6), ("loops", 3)):
            first = tool.expected_values(family, size, seed=0x12345678, base=5)
            second = tool.expected_values(family, size, seed=0x12345678, base=5)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)
            self.assertTrue(all(0 <= v <= tool.MASK for v in first))

        # Chain depends on seed and steps identically across threads 0..2.
        chain_vals = tool.expected_values("chain", 4, seed=0x00000001)
        expected_chain = 1
        for k in range(4):
            expected_chain = tool._chain_step(expected_chain, k)
        self.assertEqual(chain_vals, [expected_chain, expected_chain, expected_chain])
        self.assertNotEqual(chain_vals[0], 1)

        # Branches diverge across threads 0, 1, 2.
        branch_vals = tool.expected_values("branches", 8, seed=0x9E3779B9)
        manual_branches = []
        for tid in range(3):
            h = 0x9E3779B9
            for k in range(8):
                h = tool._branch_step(h, k, tid)
            manual_branches.append(h)
        self.assertEqual(branch_vals, manual_branches)
        self.assertGreater(len(set(branch_vals)), 1)

        # Loops depend on base and thread id.
        loop_vals = tool.expected_values("loops", 3, seed=0x5EED0001, base=5)
        manual_loops = []
        for tid in range(3):
            h = 0x5EED0001
            for k in range(3):
                h = tool._loop_step(h, k, tid, 5)
            manual_loops.append(h)
        self.assertEqual(loop_vals, manual_loops)
        self.assertNotEqual(
            loop_vals,
            tool.expected_values("loops", 3, seed=0x5EED0001, base=6),
        )

    def test_variant_info_hlsl_embedding(self):
        family, size, source, expected = tool.variant_info("control", seed=0x5EED0001)
        self.assertEqual((family, size, expected), ("control", 0, [0x5EED0001] * 3))
        self.assertTrue(source.startswith(tool.HEADER))
        self.assertTrue(source.endswith(tool.FOOTER))
        self.assertIn(
            "static const uint expected[3] = { 0x5eed0001u, 0x5eed0001u, 0x5eed0001u };",
            source,
        )
        self.assertNotIn("0x2c1b3c6du", source)
        self.assertNotIn("seed[1].x", source)

        for name, exp_family, exp_size, needle in (
            ("chain_4", "chain", 4, "* 0x2c1b3c6du + 3u;"),
            ("branches_5", "branches", 5, "if (((tid + 4u) &"),
            ("loops_3", "loops", 3, "for (uint i2 = 0u; i2 < seed[1].x + ((tid + 2u) & 3u); ++i2)"),
        ):
            fam, sz, hlsl, exp = tool.variant_info(name, seed=0x11223344)
            self.assertEqual((fam, sz), (exp_family, exp_size))
            self.assertEqual(exp, tool.expected_values(exp_family, exp_size, seed=0x11223344))
            table = ", ".join(f"0x{v:08x}u" for v in exp)
            self.assertIn(f"static const uint expected[3] = {{ {table} }};", hlsl)
            self.assertIn(needle, hlsl)
            self.assertTrue(hlsl.startswith(tool.HEADER))
            self.assertTrue(hlsl.endswith(tool.FOOTER))

    def test_variant_info_hlsl_body_matches_expected_and_size(self):
        import re

        chain_re = re.compile(
            r"^\s+h = \(h \^ \(h >> 15u\)\) \* 0x2c1b3c6du \+ (\d+)u;$"
        )
        branch_if_re = re.compile(
            r"^\s+if \(\(\(tid \+ (\d+)u\) & (\d+)u\) != 0u\) h \+= (\d+)u \* 0x01000193u;$"
        )
        branch_else_re = re.compile(
            r"^\s+else h \^= (\d+)u \* 0x85ebca6bu;$"
        )
        loop_for_re = re.compile(
            r"^\s+for \(uint i(\d+) = 0u; i\1 < seed\[1\]\.x \+ \(\(tid \+ \1u\) & 3u\); \+\+i\1\)$"
        )
        loop_body_re = re.compile(
            r"^\s+h = \(h \* 0x01000193u\) \^ \(i(\d+) \+ \1u\);$"
        )

        seed = 0x13572468
        for name in (
            "control",
            "chain_1",
            "chain_4",
            "branches_1",
            "branches_5",
            "loops_1",
            "loops_3",
        ):
            with self.subTest(variant=name):
                family, size, hlsl, exp = tool.variant_info(name, seed=seed)
                self.assertEqual(exp, tool.expected_values(family, size, seed=seed))
                expected_line_prefix = "    static const uint expected[3] = {"
                body_start = hlsl.index(expected_line_prefix)
                body_start = hlsl.index("\n", body_start) + 1
                body_end = len(hlsl) - len(tool.FOOTER)
                body = hlsl[body_start:body_end]
                body_lines = body.splitlines()

                if family == "control":
                    self.assertEqual(body_lines, [])
                    sim = [seed & tool.MASK] * 3
                elif family == "chain":
                    self.assertEqual(len(body_lines), size)
                    ks = []
                    for line in body_lines:
                        m = chain_re.fullmatch(line)
                        self.assertIsNotNone(m, line)
                        ks.append(int(m.group(1)))
                    self.assertEqual(ks, list(range(size)))
                    sim = []
                    for _tid in range(3):
                        h = seed & tool.MASK
                        for k in ks:
                            h = ((((h ^ (h >> 15)) * 0x2C1B3C6D) & tool.MASK) + k) & tool.MASK
                        sim.append(h)
                elif family == "branches":
                    self.assertEqual(len(body_lines), size * 2)
                    steps = []
                    for idx in range(0, len(body_lines), 2):
                        m_if = branch_if_re.fullmatch(body_lines[idx])
                        m_else = branch_else_re.fullmatch(body_lines[idx + 1])
                        self.assertIsNotNone(m_if, body_lines[idx])
                        self.assertIsNotNone(m_else, body_lines[idx + 1])
                        k_if, mask_val, k_then = map(int, m_if.groups())
                        k_else = int(m_else.group(1))
                        self.assertEqual((k_if, k_then, k_else), (idx // 2, idx // 2, idx // 2))
                        steps.append((k_if, mask_val))
                    sim = []
                    for tid in range(3):
                        h = seed & tool.MASK
                        for k, mask_val in steps:
                            if ((tid + k) & mask_val) != 0:
                                h = (h + k * 0x01000193) & tool.MASK
                            else:
                                h = (h ^ (k * 0x85EBCA6B)) & tool.MASK
                        sim.append(h)
                else:
                    self.assertEqual(family, "loops")
                    self.assertEqual(len(body_lines), size * 2)
                    ks = []
                    for idx in range(0, len(body_lines), 2):
                        m_for = loop_for_re.fullmatch(body_lines[idx])
                        m_body = loop_body_re.fullmatch(body_lines[idx + 1])
                        self.assertIsNotNone(m_for, body_lines[idx])
                        self.assertIsNotNone(m_body, body_lines[idx + 1])
                        self.assertEqual(m_for.group(1), m_body.group(1))
                        ks.append(int(m_for.group(1)))
                    self.assertEqual(ks, list(range(size)))
                    sim = []
                    for tid in range(3):
                        h = seed & tool.MASK
                        for k in ks:
                            trips = (tool.LOOP_BASE + ((tid + k) & 3)) & tool.MASK
                            for i in range(trips):
                                h = ((h * 0x01000193) & tool.MASK) ^ ((i + k) & tool.MASK)
                        sim.append(h)

                self.assertEqual(sim, exp)

    def test_fixture_layout_and_reference_images(self):
        seed = 0x1234ABCD
        visible_blob = tool.fixture(seed=seed, visible=True)
        black_blob = tool.fixture(seed=seed, visible=False)

        expected_len = tool.CB_BYTES + tool.CB_BYTES + tool.TRACE_BYTES + tool.SIDE * tool.SIDE * 4
        self.assertEqual(len(visible_blob), expected_len)
        self.assertEqual(len(black_blob), expected_len)

        cb0 = visible_blob[:512]
        cb1 = visible_blob[512:1024]
        trace = visible_blob[1024:1104]
        green_ref = visible_blob[1104:]
        black_ref = black_blob[1104:]

        self.assertEqual(struct.unpack_from("<I", cb0, 0)[0], seed)
        self.assertEqual(struct.unpack_from("<I", cb0, 16)[0], tool.LOOP_BASE)
        self.assertEqual(cb0[4:16], bytes(12))
        self.assertEqual(cb0[20:], bytes(512 - 20))

        self.assertEqual(struct.unpack_from("<f", cb1, 0)[0], 0.5)
        self.assertEqual(cb1[4:], bytes(512 - 4))

        self.assertEqual(trace, bytes([0xCD]) * 80)
        self.assertEqual(green_ref, bytes([0, 255, 0, 255]) * (32 * 32))
        self.assertEqual(black_ref, bytes([0, 0, 0, 255]) * (32 * 32))

    def test_classify_draw_all_outcomes(self):
        self.assertEqual(
            tool.classify_draw(0, False, "backend=d3d12\nmesh_pipeline_ms=1.23\n"),
            ("pass", ""),
        )
        self.assertEqual(
            tool.classify_draw(1, False, "Draw reference mismatch\npixels=1024 mismatches=42\n"),
            ("mismatch", "42 pixels"),
        )
        self.assertEqual(
            tool.classify_draw(1, False, "Draw reference mismatch\n"),
            ("mismatch", ""),
        )
        self.assertEqual(
            tool.classify_draw(77, False, ""),
            ("unsupported", "mesh shader tier"),
        )
        self.assertEqual(
            tool.classify_draw(1, False, "mesh_shader_tier=unsupported\n"),
            ("unsupported", "mesh shader tier"),
        )
        self.assertEqual(
            tool.classify_draw(1, False, "CreatePipelineState failed (HRESULT 0x80070057)\n"),
            ("hresult", "CreatePipelineState 0x80070057"),
        )
        self.assertEqual(
            tool.classify_draw(0, True, "mesh_pipeline_ms=1.0"),
            ("timeout", ""),
        )
        self.assertEqual(
            tool.classify_draw(-11, False, "segfault"),
            ("crash", "0xfffffff5"),
        )
        self.assertEqual(
            tool.classify_draw(0xC0000005, False, ""),
            ("crash", "0xc0000005"),
        )
        self.assertEqual(
            tool.classify_draw(None, False, ""),
            ("error", "no exit code"),
        )
        self.assertEqual(
            tool.classify_draw(2, False, "first line\nfinal diagnostic line\n"),
            ("error", "final diagnostic line"),
        )
        self.assertEqual(
            tool.classify_draw(3, False, "\n"),
            ("error", "exit 3"),
        )

    def test_summarize_and_check_failures_pass(self):
        draws = [
            {"outcome": "pass", "detail": "", "exit_code": 0, "wall_s": 0.30, "peak_mib": 10.0, "pipeline_ms": 2.0},
            {"outcome": "pass", "detail": "", "exit_code": 0, "wall_s": 0.10, "peak_mib": 14.5, "pipeline_ms": 1.0},
            {"outcome": "pass", "detail": "", "exit_code": 0, "wall_s": 0.20, "peak_mib": 12.0, "pipeline_ms": 1.5},
        ]
        rec = _make_record(draws=draws)
        summary = tool.summarize(rec)
        self.assertEqual(
            summary,
            {
                "draw_outcome": "pass",
                "draw_detail": "",
                "pipeline_ms_first": 2.0,
                "pipeline_ms_median": 1.5,
                "draw_wall_s_median": 0.2,
                "draw_peak_mib_max": 14.5,
                "controls": "wrong_seed=mismatch;wrong_reference=mismatch",
            },
        )
        empty_summary = tool.summarize(_make_record(draws=[], controls={}))
        self.assertEqual(
            empty_summary,
            {
                "draw_outcome": "not_run",
                "draw_detail": "",
                "pipeline_ms_first": None,
                "pipeline_ms_median": None,
                "draw_wall_s_median": None,
                "draw_peak_mib_max": None,
                "controls": "",
            },
        )
        self.assertEqual(tool.check_failures([rec, _make_record(optimization="Od")]), [])

    def test_write_report_csv_and_json(self):
        records = [
            _make_record(variant="control", family="control", size=0, optimization="default"),
            _make_record(variant="control", family="control", size=0, optimization="Od"),
        ]
        payload = {"adapter": "warp", "seed": "0x5eed0001"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "nested" / "report.csv"
            json_path = root / "nested" / "report.json"

            tool.write_report(csv_path, payload, records)
            tool.write_report(json_path, payload, records)

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows[0].keys()), list(tool.CSV_FIELDS))
            self.assertEqual(rows[0]["variant"], "control")
            self.assertEqual(rows[0]["optimization"], "default")
            self.assertEqual(rows[0]["draw_outcome"], "pass")
            self.assertEqual(rows[0]["pipeline_ms_median"], "1.25")
            self.assertEqual(rows[0]["controls"], "wrong_seed=mismatch;wrong_reference=mismatch")

            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["adapter"], "warp")
            self.assertEqual(loaded["seed"], "0x5eed0001")
            self.assertEqual(len(loaded["variants"]), 2)
            self.assertEqual(loaded["variants"][1]["optimization"], "Od")

    def test_list_variants_cli_direct_and_subprocess(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = tool.main(["--list-variants", "--variants", "control,chain_2,loops_1", "--seed", "0x12345678"])
        self.assertEqual(rc, 0)
        self.assertEqual(stderr.getvalue(), "")
        lines = stdout.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("control", lines[0])
        self.assertIn("0x12345678 0x12345678 0x12345678", lines[0])
        exp_loops = tool.expected_values("loops", 1, seed=0x12345678)
        self.assertIn(" ".join(f"0x{v:08x}" for v in exp_loops), lines[2])

        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                str(TOOL),
                "--list-variants",
                "--variants",
                "control,branches_3",
                "--seed",
                "0x5eed0001",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertIn("control", proc.stdout)
        self.assertIn("branches_3", proc.stdout)
        self.assertIn("0x5eed0001 0x5eed0001 0x5eed0001", proc.stdout)

    def test_main_mocked_check_pass_unsupported_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc = root / "dxc.exe"
            renderer = root / "offscreen_check.exe"
            dxc.write_bytes(b"fake")
            renderer.write_bytes(b"fake")
            temp_root = root / "temp_parent"
            temp_root.mkdir()
            out_json = root / "out.json"

            def fake_run_once(_dxc, _hlsl, cso, _flags, _timeout):
                cso.write_bytes(b"DXBC" + b"\x00" * 32)
                return {
                    "args": ["dxc", "-Fo", str(cso)],
                    "wall_s": 0.01,
                    "peak_bytes": 1024 * 1024,
                    "exit_code": 0,
                    "outcome": "dxil",
                    "cso_bytes": 36,
                    "first_diagnostic": "",
                    "output": "",
                }

            def fake_run_draw(_renderer, _adapter, _mesh, _pixel, data_path, _timeout):
                is_ref = data_path.name == "reference.bin"
                return {
                    "outcome": "pass" if is_ref else "mismatch",
                    "detail": "" if is_ref else "1024 pixels",
                    "exit_code": 0 if is_ref else 1,
                    "wall_s": 0.02,
                    "peak_mib": 8.0,
                    "pipeline_ms": 0.75 if is_ref else None,
                    "output": "adapter=mock\nmesh_pipeline_ms=0.75\n",
                }

            stdout = io.StringIO()
            with (
                mock.patch.object(tool, "run_once", side_effect=fake_run_once),
                mock.patch.object(tool, "run_draw", side_effect=fake_run_draw),
                mock.patch.object(tool, "dxc_version", return_value="dxcompiler 1.7"),
                mock.patch.object(tool.tempfile, "tempdir", str(temp_root)),
                redirect_stdout(stdout),
            ):
                rc = tool.main([
                    "--dxc", str(dxc),
                    "--renderer", str(renderer),
                    "--adapter", "warp",
                    "--variants", "control",
                    "--compile-timeout", "10",
                    "--pipeline-timeout", "10",
                    "--check",
                    "--out", str(out_json),
                ])

            self.assertEqual(rc, 0)
            self.assertEqual(list(temp_root.iterdir()), [])
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["dxc_version"], "dxcompiler 1.7")
            self.assertEqual(len(report["variants"]), 2)
            self.assertEqual(report["variants"][0]["controls"]["wrong_seed"], "mismatch")

            # Capability skip exit 77 when all draws are unsupported.
            def fake_unsupported(*_args, **_kwargs):
                return {
                    "outcome": "unsupported",
                    "detail": "mesh shader tier",
                    "exit_code": 77,
                    "wall_s": 0.01,
                    "peak_mib": 4.0,
                    "pipeline_ms": None,
                    "output": "mesh_shader_tier=unsupported\n",
                }

            stdout = io.StringIO()
            with (
                mock.patch.object(tool, "run_once", side_effect=fake_run_once),
                mock.patch.object(tool, "run_draw", side_effect=fake_unsupported),
                mock.patch.object(tool.tempfile, "tempdir", str(temp_root)),
                redirect_stdout(stdout),
            ):
                rc_skip = tool.main([
                    "--dxc", str(dxc),
                    "--renderer", str(renderer),
                    "--adapter", "hardware",
                    "--variants", "control",
                    "--compile-timeout", "10",
                    "--pipeline-timeout", "10",
                    "--check",
                ])

            self.assertEqual(rc_skip, 77)
            self.assertEqual(list(temp_root.iterdir()), [])


class MeshOdPipelineNegativeTests(unittest.TestCase):
    def _assert_main_exit_2(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                tool.main(argv)
        self.assertEqual(exc.exception.code, 2, (argv, stderr.getvalue()))
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(stderr.getvalue().strip())

    def test_expected_values_rejects_invalid_inputs(self):
        for bad_family in ("bogus", "samplebranch", "", "CONTROL", "chain_1", None):
            with self.subTest(family=bad_family):
                with self.assertRaises(ValueError):
                    tool.expected_values(bad_family, 0)

        for bad_size in (-1, -64, True, 1.5, "4"):
            with self.subTest(size=bad_size):
                with self.assertRaises(ValueError):
                    tool.expected_values("chain", bad_size)

        for nonzero in (1, 16):
            with self.subTest(control_size=nonzero):
                with self.assertRaises(ValueError):
                    tool.expected_values("control", nonzero)

    def test_variant_info_rejects_bad_syntax_and_size(self):
        bad_names = (
            "",
            "bogus",
            "control_0",
            "control_1",
            "chain",
            "chain_",
            "chain_-1",
            "chain_0",
            "branches_0",
            "loops_0",
            f"chain_{tool.MAX_SIZE + 1}",
            f"branches_{tool.MAX_SIZE + 1}",
            f"loops_{tool.MAX_SIZE + 1}",
            "chain_1_2",
            "chain_1\n",
            None,
        )
        for bad in bad_names:
            with self.subTest(variant=bad):
                with self.assertRaises(ValueError):
                    tool.variant_info(bad)

    def test_check_failures_catches_all_failure_modes(self):
        compile_err = [_make_record(compile_outcome="diagnostic")]
        self.assertEqual(len(tool.check_failures(compile_err)), 1)
        self.assertIn("compile diagnostic", tool.check_failures(compile_err)[0])

        no_draws = [_make_record(draws=[])]
        self.assertIn("draw not_run", tool.check_failures(no_draws)[0])

        non_pass_draw = [
            _make_record(
                draws=[{
                    "outcome": "mismatch",
                    "detail": "1024 pixels",
                    "exit_code": 1,
                    "wall_s": 0.1,
                    "peak_mib": 8.0,
                    "pipeline_ms": 1.0,
                }]
            )
        ]
        self.assertIn("draw mismatch", tool.check_failures(non_pass_draw)[0])

        missing_pipeline_ms = [
            _make_record(
                draws=[{
                    "outcome": "pass",
                    "detail": "",
                    "exit_code": 0,
                    "wall_s": 0.1,
                    "peak_mib": 8.0,
                    "pipeline_ms": None,
                }]
            )
        ]
        self.assertIn("no mesh_pipeline_ms", tool.check_failures(missing_pipeline_ms)[0])

        bad_seed_ctrl = [_make_record(controls={"wrong_seed": "pass", "wrong_reference": "mismatch"})]
        self.assertIn("control wrong_seed gave pass", tool.check_failures(bad_seed_ctrl)[0])

        bad_ref_ctrl = [_make_record(controls={"wrong_seed": "mismatch", "wrong_reference": "error"})]
        self.assertIn("control wrong_reference gave error", tool.check_failures(bad_ref_ctrl)[0])

        missing_ctrl = [_make_record(controls={"wrong_seed": "mismatch"})]
        self.assertIn("control wrong_reference gave None", tool.check_failures(missing_ctrl)[0])

    def test_main_rejects_bad_cli_flags_with_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc = root / "dxc.exe"
            renderer = root / "offscreen_check.exe"
            dxc.write_bytes(b"fake")
            renderer.write_bytes(b"fake")
            missing_dxc = root / "missing_dxc.exe"
            missing_renderer = root / "missing_renderer.exe"

            base = [
                "--dxc", str(dxc),
                "--renderer", str(renderer),
                "--adapter", "warp",
                "--compile-timeout", "10",
                "--pipeline-timeout", "10",
            ]

            for bad_variants in ("", ",", "   ", "bogus", "chain_0", f"loops_{tool.MAX_SIZE + 1}"):
                self._assert_main_exit_2([*base, "--variants", bad_variants])

            for bad_seed in ("-1", "0x100000000", "not_a_number"):
                self._assert_main_exit_2([*base, "--seed", bad_seed])

            for bad_repeat in ("0", "-2"):
                self._assert_main_exit_2([*base, "--repeat", bad_repeat])

            for bad_timeout in ("0", "-1", "nan", "inf", "-inf"):
                self._assert_main_exit_2([
                    "--dxc", str(dxc),
                    "--renderer", str(renderer),
                    "--adapter", "warp",
                    "--compile-timeout", bad_timeout,
                    "--pipeline-timeout", "10",
                ])
                self._assert_main_exit_2([
                    "--dxc", str(dxc),
                    "--renderer", str(renderer),
                    "--adapter", "warp",
                    "--compile-timeout", "10",
                    "--pipeline-timeout", bad_timeout,
                ])

            # Missing each required flag individually.
            required_pairs = [
                ("--dxc", str(dxc)),
                ("--renderer", str(renderer)),
                ("--adapter", "warp"),
                ("--compile-timeout", "10"),
                ("--pipeline-timeout", "10"),
            ]
            for omitted_flag, _ in required_pairs:
                partial = []
                for flag, val in required_pairs:
                    if flag != omitted_flag:
                        partial.extend([flag, val])
                self._assert_main_exit_2(partial)

            # Missing executable paths (and directory passed instead of file).
            self._assert_main_exit_2([
                "--dxc", str(missing_dxc),
                "--renderer", str(renderer),
                "--adapter", "warp",
                "--compile-timeout", "10",
                "--pipeline-timeout", "10",
            ])
            self._assert_main_exit_2([
                "--dxc", str(dxc),
                "--renderer", str(missing_renderer),
                "--adapter", "warp",
                "--compile-timeout", "10",
                "--pipeline-timeout", "10",
            ])
            self._assert_main_exit_2([
                "--dxc", str(root),
                "--renderer", str(renderer),
                "--adapter", "warp",
                "--compile-timeout", "10",
                "--pipeline-timeout", "10",
            ])

    def test_subprocess_cli_rejection_and_temp_cleanup_on_failure(self):
        proc = subprocess.run(
            [sys.executable, "-B", str(TOOL), "--variants", "unknown_1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertTrue(proc.stderr.strip())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc = root / "dxc.exe"
            renderer = root / "offscreen_check.exe"
            dxc.write_bytes(b"fake")
            renderer.write_bytes(b"fake")
            temp_root = root / "temp_parent"
            temp_root.mkdir()

            def fail_pixel_compile(*_args, **_kwargs):
                return {
                    "args": ["dxc"],
                    "wall_s": 0.01,
                    "peak_bytes": None,
                    "exit_code": 1,
                    "outcome": "diagnostic",
                    "cso_bytes": 0,
                    "first_diagnostic": "error: bad pixel shader",
                    "output": "error: bad pixel shader",
                }

            stderr = io.StringIO()
            with (
                mock.patch.object(tool, "run_once", side_effect=fail_pixel_compile),
                mock.patch.object(tool.tempfile, "tempdir", str(temp_root)),
                redirect_stderr(stderr),
            ):
                rc = tool.main([
                    "--dxc", str(dxc),
                    "--renderer", str(renderer),
                    "--adapter", "warp",
                    "--variants", "control",
                    "--compile-timeout", "10",
                    "--pipeline-timeout", "10",
                ])

            self.assertEqual(rc, 1)
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertIn("pixel shader did not compile", stderr.getvalue())
            self.assertEqual(list(temp_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
