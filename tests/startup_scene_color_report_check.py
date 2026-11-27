"""Synthetic histories for tools/startup/scene_color_report.py.

Refs #870.  No dump, title file or GPU is used here.  Each case writes the exact
``pixel_history.txt`` shape from ``src/startup_stubs/pixel_history.h`` in a
temporary directory and checks the bounded Markdown result.
"""

from __future__ import annotations

from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "startup" / "scene_color_report.py"


def raw_rgba(red: int, green: int, blue: int, alpha: int, fmt: int = 28) -> int:
    """Return the little-endian word pixel_history_decode() reads."""
    if fmt in (87, 91):
        return blue | (green << 8) | (red << 16) | (alpha << 24)
    return red | (green << 8) | (blue << 16) | (alpha << 24)


def sample(draw: int, *, frame: int = 1, x: int = 1, y: int = 1,
           key: int = 1, count: int = 3, fmt: int = 28,
           rgba: tuple[int, int, int, int] | None = None) -> str:
    if rgba is None:
        rgba = (0, 0, 0, 0)
    raw = raw_rgba(*rgba, fmt=fmt)
    line = (f"draw={draw} frame={frame} x={x} y={y} key={key:016x} "
            f"count={count} format={fmt} raw={raw:08x}")
    if fmt in (28, 29, 87, 91):
        line += " rgba=%d,%d,%d,%d" % rgba
    return line


def history(*lines: str) -> str:
    return "x2w-pixel-history 1 samples=%d\n" % len(lines) + "".join(line + "\n" for line in lines)


def write_bmp(path: Path, width: int, height: int,
              pixels: dict[tuple[int, int], tuple[int, int, int]]) -> None:
    """Write the 32-bit bottom-up BI_RGB BMP shape draw_bmp.h produces."""
    body = bytearray()
    for y in range(height - 1, -1, -1):
        for x in range(width):
            red, green, blue = pixels.get((x, y), (0, 0, 0))
            body += bytes((blue, green, red, 255))
    header = struct.pack("<2sIHHI", b"BM", 54 + len(body), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 32, 0, len(body), 0, 0, 0, 0)
    path.write_bytes(header + info + body)


class SceneColorReportCheck(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def dump_dir(self, name: str = "dump") -> Path:
        folder = self.root / name
        folder.mkdir()
        return folder

    def write_history(self, folder: Path, text: str) -> None:
        (folder / "pixel_history.txt").write_text(text, encoding="ascii")

    def run_report(self, folder: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(TOOL), str(folder), *args],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    def test_first_change_is_not_the_last_writer(self) -> None:
        folder = self.dump_dir()
        self.write_history(folder, history(
            sample(1, rgba=(0, 0, 0, 0), key=0x1111111111111111),
            sample(2, rgba=(0, 255, 0, 255), key=0x2222222222222222),
            sample(3, rgba=(0, 255, 0, 255), key=0x2222222222222222),
            sample(4, rgba=(0, 0, 255, 255), key=0x4444444444444444),
        ))
        result = self.run_report(folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Classification: `target_write`", result.stdout)
        self.assertIn("First observed change: `draw=2`", result.stdout)
        self.assertIn("Last observed writer: `draw=4`", result.stdout)
        self.assertIn("Frame: `1`", result.stdout)
        self.assertIn("Pixel: `1,1`", result.stdout)

    def test_last_writer_can_be_the_first_change(self) -> None:
        folder = self.dump_dir()
        self.write_history(folder, history(
            sample(1, rgba=(0, 0, 0, 0)),
            sample(2, rgba=(0, 255, 0, 255), key=0x2222222222222222),
            sample(3, rgba=(0, 255, 0, 255), key=0x2222222222222222),
        ))
        result = self.run_report(folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("First observed change: `draw=2`", result.stdout)
        self.assertIn("Last observed writer: `draw=2`", result.stdout)
        self.assertIn("first change; last writer", result.stdout)

    def test_value_unchanged_after_a_restore(self) -> None:
        folder = self.dump_dir()
        self.write_history(folder, history(
            sample(1, rgba=(0, 0, 0, 0)),
            sample(2, rgba=(0, 255, 0, 255), key=0x2222222222222222),
            sample(3, rgba=(0, 0, 0, 0), key=0x3333333333333333),
        ))
        result = self.run_report(folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Classification: `value_unchanged`", result.stdout)
        self.assertIn("First observed change: `draw=2`", result.stdout)
        self.assertIn("Last observed writer: `draw=3`", result.stdout)

    def test_ui_only_when_no_sample_changes(self) -> None:
        folder = self.dump_dir()
        self.write_history(folder, history(
            sample(1, rgba=(0, 0, 0, 0)),
            sample(2, rgba=(0, 0, 0, 0)),
            sample(3, rgba=(0, 0, 0, 0)),
        ))
        result = self.run_report(folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Classification: `ui_only`", result.stdout)
        self.assertIn("First observed change: _none among the retained samples", result.stdout)
        self.assertIn("Last observed writer: _none_", result.stdout)

    def test_empty_history_is_no_draw(self) -> None:
        folder = self.dump_dir()
        self.write_history(folder, "x2w-pixel-history 1 samples=0\n")
        result = self.run_report(folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Classification: `no_draw`", result.stdout)
        self.assertIn("Selected sample group: _none_", result.stdout)
        self.assertIn("_No retained sample for the selected frame and pixel._", result.stdout)

    def test_malformed_line_is_ignored_without_echo(self) -> None:
        folder = self.dump_dir("private-title-folder")
        malformed = "draw=2 frame=1 x=1 y=1 key=2222222222222222 count=3 format=28 raw=00000000 " \
                    "C:\\Users\\someone\\secret\\GameName.exe"
        self.write_history(folder, history(
            sample(1, rgba=(0, 0, 0, 0)),
            malformed,
            sample(3, rgba=(0, 255, 0, 255), key=0x3333333333333333),
        ))
        result = self.run_report(folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Malformed records ignored: `1`", result.stdout)
        self.assertIn("First observed change: `draw=3`", result.stdout)
        self.assertNotIn("GameName", result.stdout)
        self.assertNotIn("secret", result.stdout)
        self.assertNotIn("Users", result.stdout)
        self.assertNotIn("private-title-folder", result.stdout)
        self.assertNotIn(str(self.root), result.stdout)

    def test_non_rgba8_target_is_not_decoded(self) -> None:
        folder = self.dump_dir()
        self.write_history(folder, history(
            "draw=1 frame=1 x=1 y=1 key=1111111111111111 count=3 format=10 raw=00000000",
        ))
        result = self.run_report(folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Classification: `target_non_rgba8`", result.stdout)
        self.assertIn("First observed change: _not evaluated", result.stdout)
        self.assertIn("target is not RGBA8", result.stdout)

    def test_bgra_format_uses_the_documented_channel_order(self) -> None:
        folder = self.dump_dir()
        self.write_history(folder, history(
            sample(1, fmt=87, rgba=(255, 0, 0, 255)),
        ))
        result = self.run_report(folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DXGI_FORMAT_B8G8R8A8_UNORM (`87`)", result.stdout)
        self.assertIn("`255,0,0,255`", result.stdout)

    def test_history_cap_marks_truncated_and_bounds_rows(self) -> None:
        folder = self.dump_dir()
        lines = [sample(1, rgba=(0, 0, 0, 0), key=1)]
        for draw in range(2, 49):
            lines.append(sample(draw, rgba=(0, 255, 0, 255), key=0x2000000000000000 + draw))
        self.write_history(folder, history(*lines))
        result = self.run_report(folder, "--max-rows", "12")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Classification: `history_truncated`", result.stdout)
        self.assertIn("First observed change: `draw=2`", result.stdout)
        self.assertIn("History limit: the runtime cap", result.stdout)
        timeline = result.stdout.split("## Retained timeline", 1)[1].split("## Input accounting", 1)[0]
        rows = [line for line in timeline.splitlines() if line.startswith("| `")]
        self.assertLessEqual(len(rows), 12, timeline)
        self.assertIn("row(s) omitted", timeline)

    def test_frame_and_pixel_selection(self) -> None:
        folder = self.dump_dir()
        self.write_history(folder, history(
            sample(1, frame=1, x=0, y=0, rgba=(0, 0, 0, 0)),
            sample(2, frame=1, x=0, y=0, rgba=(0, 255, 0, 255), key=0x2222222222222222),
            sample(3, frame=2, x=1, y=1, rgba=(0, 0, 0, 0)),
            sample(4, frame=2, x=1, y=1, rgba=(0, 0, 255, 255), key=0x4444444444444444),
            sample(5, frame=2, x=1, y=1, rgba=(0, 0, 255, 255), key=0x4444444444444444),
        ))
        selected = self.run_report(folder, "--frame", "1", "--pixel", "0,0")
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertIn("Frame: `1`", selected.stdout)
        self.assertIn("Pixel: `0,0`", selected.stdout)
        self.assertIn("First observed change: `draw=2`", selected.stdout)

        default = self.run_report(folder)
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertIn("Frame: `2`", default.stdout)
        self.assertIn("Pixel: `1,1`", default.stdout)

    def test_supplied_baseline_can_name_the_first_draw(self) -> None:
        folder = self.dump_dir()
        self.write_history(folder, history(
            sample(1, rgba=(0, 255, 0, 255)),
        ))
        without = self.run_report(folder)
        self.assertEqual(without.returncode, 0, without.stderr)
        self.assertIn("Classification: `value_unchanged`", without.stdout)
        with_baseline = self.run_report(folder, "--baseline-raw", "0x00000000")
        self.assertEqual(with_baseline.returncode, 0, with_baseline.stderr)
        self.assertIn("Classification: `target_write`", with_baseline.stdout)
        self.assertIn("First observed change: `draw=1`", with_baseline.stdout)

    def test_bmp_cross_check_and_mismatch_control(self) -> None:
        folder = self.dump_dir()
        self.write_history(folder, history(
            sample(1, rgba=(0, 0, 0, 0), key=0x1111111111111111),
            sample(2, rgba=(0, 255, 0, 255), key=0x2222222222222222),
        ))
        bmp = folder / "draw_02_2222222222222222_3.bmp"
        write_bmp(bmp, 4, 4, {(1, 1): (0, 255, 0)})
        matched = self.run_report(folder)
        self.assertEqual(matched.returncode, 0, matched.stderr)
        self.assertIn("BMP files: `1` (valid `1`, invalid `0`)", matched.stdout)
        self.assertIn("Target dimensions: `4x4`", matched.stdout)
        self.assertIn("Matching selected samples: `1` matched", matched.stdout)

        write_bmp(bmp, 4, 4, {(1, 1): (0, 0, 255)})
        mismatched = self.run_report(folder)
        self.assertEqual(mismatched.returncode, 0, mismatched.stderr)
        self.assertIn("Matching selected samples: `0` matched, `1` mismatched", mismatched.stdout)
        self.assertIn("BMP mismatch at `draw=2`", mismatched.stdout)

    def test_header_mismatch_is_bounded(self) -> None:
        folder = self.dump_dir()
        text = "x2w-pixel-history 1 samples=5\n" + sample(1, rgba=(0, 0, 0, 0)) + "\n"
        self.write_history(folder, text)
        result = self.run_report(folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Classification: `history_truncated`", result.stdout)
        self.assertIn("Header count and physical record count differ", result.stdout)

    def test_missing_history_is_a_controlled_error(self) -> None:
        folder = self.dump_dir()
        result = self.run_report(folder)
        self.assertEqual(result.returncode, 2)
        self.assertIn("pixel_history.txt not found", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
