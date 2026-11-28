"""CLI, csv.Error/directory guard, and mutation control suite for tools/startup_audio_timeline.py."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "startup_audio_timeline.py"
sys.path.insert(0, str(ROOT / "tools"))

import startup_audio_timeline as timeline  # noqa: E402


EVENT_KEYS = {
    "action",
    "qpc",
    "start_qpc",
    "end_qpc",
    "coverage",
    "packets",
    "discontinuities",
    "rms_estimate",
    "packet_peak",
}


def _csv_bytes(rows: list[list[object]], *, bom: bool = True) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["process_s", "frames", "rms", "peak", "flags"])
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8-sig" if bom else "utf-8")


def _jsonl_bytes(events: list[dict[str, object]], *, bom: bool = True) -> bytes:
    text = "\n\n".join(json.dumps(item) for item in events) + "\n"
    return text.encode("utf-8-sig" if bom else "utf-8")


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = timeline.main(argv)
        except SystemExit as exc:
            return exc.code, stdout.getvalue(), stderr.getvalue()
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class StartupAudioTimelineCliPositiveTests(unittest.TestCase):
    def test_bom_csv_multiple_actions_flags4_skip_silence_and_json_schema(self) -> None:
        # sample_rate=2000 -> 500 frames = 0.25 s.
        packets_bytes = _csv_bytes(
            [
                [1.0, 500, 0.4, 0.8, 1],     # 101.00..101.25 (discontinuity)
                [1.25, 500, 0.2, 0.5, 0],    # 101.25..101.50
                [2.0, 500, 0.9, 0.95, 4],    # flags & 4: WASAPI timestamp error -> skipped!
            ],
            bom=True,
        )
        inputs_bytes = _jsonl_bytes(
            [
                {"action": "jump", "qpc": 101.25},
                {"action": "ignored", "qpc": 101.25},
                {"action": "silence", "qpc": 102.25},
            ],
            bom=True,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets = root / "packets.csv"
            inputs = root / "inputs.jsonl"
            packets.write_bytes(packets_bytes)
            inputs.write_bytes(inputs_bytes)

            argv = [
                str(packets),
                str(inputs),
                "--process-origin-qpc", "100",
                "--sample-rate", "2000",
                "--before", "0.25",
                "--after", "0.25",
                "--action", "jump",
                "--action", "silence",
            ]
            code, out, err = _run_main(argv)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            doc = json.loads(out)
            self.assertEqual(set(doc), {"events"})
            self.assertEqual(len(doc["events"]), 2)
            for row in doc["events"]:
                self.assertEqual(set(row), EVENT_KEYS)

            by_action = {row["action"]: row for row in doc["events"]}
            jump = by_action["jump"]
            self.assertAlmostEqual(jump["start_qpc"], 101.0)
            self.assertAlmostEqual(jump["end_qpc"], 101.5)
            self.assertAlmostEqual(jump["coverage"], 1.0)
            self.assertEqual(jump["packets"], 2)
            self.assertEqual(jump["discontinuities"], 1)
            self.assertAlmostEqual(jump["rms_estimate"], math.sqrt((0.4**2 + 0.2**2) / 2))
            self.assertEqual(jump["packet_peak"], 0.8)

            silence = by_action["silence"]
            self.assertAlmostEqual(silence["coverage"], 0.0)
            self.assertEqual(silence["packets"], 0)
            self.assertEqual(silence["discontinuities"], 0)
            self.assertIsNone(silence["rms_estimate"])
            self.assertIsNone(silence["packet_peak"])

            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(json.loads(proc.stdout), doc)

    def test_main_forwards_sys_argv_and_script_entrypoint_returns_zero(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            packets = root / "packets.csv"
            inputs = root / "inputs.jsonl"
            packets.write_bytes(_csv_bytes([[0.5, 4800, 0.25, 0.5, 0]]))
            inputs.write_bytes(_jsonl_bytes([{"action": "fire", "qpc": 10.5}]))
            cli_argv = [
                str(TOOL),
                str(packets),
                str(inputs),
                "--process-origin-qpc", "10",
                "--action", "fire",
            ]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(len(json.loads(out)["events"]), 1)

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertIs(type(exited.exception.code), int)
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(len(json.loads(stdout.getvalue())["events"]), 1)


class StartupAudioTimelineCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_exit_2(self, argv: list[str], *, expected_fragment: str = "", check_subprocess: bool = True) -> None:
        code, out, err = _run_main(argv)
        self.assertEqual(code, 2, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip())
        if expected_fragment:
            self.assertIn(expected_fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip())
            if expected_fragment:
                self.assertIn(expected_fragment, proc.stderr)

    def test_missing_and_directory_paths_rejected_upfront_exit_2(self) -> None:
        # Kills removal of OSError guard and removal of is_dir() guards for packets and inputs.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_csv = root / "packets.csv"
            valid_jsonl = root / "inputs.jsonl"
            valid_csv.write_bytes(_csv_bytes([[0.0, 4800, 0.1, 0.2, 0]]))
            valid_jsonl.write_bytes(_jsonl_bytes([{"action": "a", "qpc": 100.0}]))
            missing = root / "missing"
            dir_path = root / "a_dir"
            dir_path.mkdir()

            base_flags = ["--process-origin-qpc", "100", "--action", "a"]

            self._assert_exit_2([str(missing), str(valid_jsonl), *base_flags], expected_fragment=missing.name)
            self._assert_exit_2([str(valid_csv), str(missing), *base_flags], expected_fragment=missing.name)

            with (
                mock.patch.object(timeline, "read_packets") as read_packets_mock,
                mock.patch.object(timeline, "read_events") as read_events_mock,
            ):
                self._assert_exit_2(
                    [str(dir_path), str(valid_jsonl), *base_flags],
                    expected_fragment="Packets path is a directory",
                    check_subprocess=False,
                )
                self._assert_exit_2(
                    [str(valid_csv), str(dir_path), *base_flags],
                    expected_fragment="Inputs path is a directory",
                    check_subprocess=False,
                )
            read_packets_mock.assert_not_called()
            read_events_mock.assert_not_called()

            self._assert_exit_2([str(dir_path), str(valid_jsonl), *base_flags],
                                expected_fragment="Packets path is a directory")
            self._assert_exit_2([str(valid_csv), str(dir_path), *base_flags],
                                expected_fragment="Inputs path is a directory")

    def test_csv_error_and_non_utf8_inputs_exit_2_without_traceback(self) -> None:
        # NUL byte triggers csv.Error in csv.DictReader (kills removal of csv.Error from except tuple).
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_csv = root / "valid.csv"
            valid_jsonl = root / "valid.jsonl"
            valid_csv.write_bytes(_csv_bytes([[0.0, 4800, 0.1, 0.2, 0]]))
            valid_jsonl.write_bytes(_jsonl_bytes([{"action": "a", "qpc": 100.0}]))
            flags = ["--process-origin-qpc", "100", "--action", "a"]

            nul_csv = root / "nul_byte.csv"
            nul_csv.write_bytes(b"process_s,frames,rms,peak,flags\n0.0,4800,\x000.1,0.2,0\n")
            self._assert_exit_2([str(nul_csv), str(valid_jsonl), *flags], expected_fragment="nul_byte.csv:2:")

            # Field larger than csv.field_size_limit() triggers real csv.Error in both in-process & subprocess.
            oversized_csv = root / "oversized.csv"
            oversized_csv.write_bytes(
                b"process_s,frames,rms,peak,flags\n" + b"0." + b"1" * (csv.field_size_limit() + 16) + b",4800,0.1,0.2,0\n"
            )
            self._assert_exit_2([str(oversized_csv), str(valid_jsonl), *flags], expected_fragment="field limit")

            with mock.patch.object(timeline, "read_packets", side_effect=csv.Error("malformed CSV record")):
                self._assert_exit_2(
                    [str(valid_csv), str(valid_jsonl), *flags],
                    expected_fragment="malformed CSV record",
                    check_subprocess=False,
                )

            bad_utf8_csv = root / "bad_utf8.csv"
            bad_utf8_csv.write_bytes(b"\xff\xfe\x80")
            self._assert_exit_2([str(bad_utf8_csv), str(valid_jsonl), *flags])

            bad_utf8_jsonl = root / "bad_utf8.jsonl"
            bad_utf8_jsonl.write_bytes(b"\xff\xfe\x80")
            self._assert_exit_2([str(valid_csv), str(bad_utf8_jsonl), *flags])

    def test_malformed_jsonl_nan_inf_and_window_bounds_exit_2(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_csv = root / "valid.csv"
            valid_jsonl = root / "valid.jsonl"
            valid_csv.write_bytes(_csv_bytes([[0.0, 4800, 0.1, 0.2, 0]]))
            valid_jsonl.write_bytes(_jsonl_bytes([{"action": "a", "qpc": 100.0}]))
            flags = ["--process-origin-qpc", "100", "--action", "a"]

            for label, content in (
                ("non_object_list", "[1, 2]\n"),
                ("non_object_int", "42\n"),
                ("bad_json", "{broken\n"),
                ("nan_qpc", '{"action": "a", "qpc": NaN}\n'),
                ("inf_qpc", '{"action": "a", "qpc": Infinity}\n'),
            ):
                with self.subTest(jsonl=label):
                    bad_jsonl = root / f"{label}.jsonl"
                    bad_jsonl.write_text(content, encoding="utf-8")
                    self._assert_exit_2([str(valid_csv), str(bad_jsonl), *flags], expected_fragment=bad_jsonl.name)

            for label, row in (
                ("nan_process_s", ["nan", 4800, 0.1, 0.2, 0]),
                ("inf_rms", [0.0, 4800, "inf", 0.2, 0]),
                ("nan_peak", [0.0, 4800, 0.1, "nan", 0]),
                ("bad_header", None),
            ):
                with self.subTest(csv_case=label):
                    bad_csv = root / f"{label}.csv"
                    bad_csv.write_bytes(
                        b"wrong,columns\n1,2\n" if row is None else _csv_bytes([row])
                    )
                    self._assert_exit_2([str(bad_csv), str(valid_jsonl), *flags])

            for bad_opts in (
                ["--process-origin-qpc", "nan", "--action", "a"],
                ["--process-origin-qpc", "inf", "--action", "a"],
                ["--process-origin-qpc", "100", "--action", "a", "--before", "0", "--after", "0"],
                ["--process-origin-qpc", "100", "--action", "a", "--before", "-0.1"],
                ["--process-origin-qpc", "100", "--action", "a", "--sample-rate", "0"],
            ):
                with self.subTest(opts=bad_opts):
                    self._assert_exit_2([str(valid_csv), str(valid_jsonl), *bad_opts])


if __name__ == "__main__":
    unittest.main()
