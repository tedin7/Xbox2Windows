"""Check the pure parts of the startup debugger's periodic window pictures (tools/startup/window_capture.py).

usage: startup_window_capture_check.py

Covers the BGRA -> PNG conversion (decoded again and compared with a per-pixel reference), the file names, the
skip of a frame that equals the previous one, the file cap, and the summary line. The Windows side (a real
window whose picture changes at a known time) is tests/startup_debug_check.py, group "window".
"""
import os
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "startup"))
from window_capture import MenuPace, PNG_SIGNATURE, Series, frame_hash, frame_mean, frame_name, pick_window, png_bytes  # noqa: E402


def decode_png(data):
    """(width, height, [rows of RGB bytes]) of a truecolour 8 bit PNG with filter 0 on every row."""
    assert data[:8] == PNG_SIGNATURE
    at, idat, header = 8, b"", None
    while at < len(data):
        size, kind = struct.unpack_from(">I4s", data, at)
        body = data[at + 8:at + 8 + size]
        assert struct.unpack_from(">I", data, at + 8 + size)[0] == zlib.crc32(kind + body) & 0xFFFFFFFF, kind
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        at += 12 + size
    width, height, depth, colour, _, _, _ = header
    assert (depth, colour) == (8, 2)
    raw = zlib.decompress(idat)
    stride = 1 + width * 3
    assert len(raw) == stride * height
    rows = []
    for y in range(height):
        assert raw[y * stride] == 0, "filter byte"
        rows.append(raw[y * stride + 1:(y + 1) * stride])
    return width, height, rows


def reference_rows(width, height, bgra):
    """Per-pixel BGRA -> RGB, the slow way the first capture code did it."""
    return [bytes(b for x in range(width) for b in (bgra[(y * width + x) * 4 + 2], bgra[(y * width + x) * 4 + 1],
                                                    bgra[(y * width + x) * 4])) for y in range(height)]


def picture(width, height, seed):
    return bytes((seed * 31 + i * 7 + (i >> 3)) & 0xFF for i in range(width * height * 4))


class Conversion(unittest.TestCase):
    def test_png_matches_the_per_pixel_reference(self):
        for width, height in ((1, 1), (3, 2), (17, 5), (64, 3)):
            pixels = picture(width, height, width)
            self.assertEqual(decode_png(png_bytes(width, height, pixels)), (width, height, reference_rows(width, height, pixels)))

    def test_alpha_is_dropped_and_channels_are_swapped(self):
        self.assertEqual(decode_png(png_bytes(1, 1, bytes([10, 20, 30, 99])))[2], [bytes([30, 20, 10])])

    def test_level_changes_the_size_not_the_picture(self):
        pixels = bytes(4 * 40 * 40)
        self.assertEqual(decode_png(png_bytes(40, 40, pixels, 1)), decode_png(png_bytes(40, 40, pixels, 9)))

    def test_a_buffer_that_does_not_fit_is_refused(self):
        for args in ((2, 2, bytes(15)), (0, 2, b""), (2, 0, b"")):
            with self.assertRaises(ValueError):
                png_bytes(*args)


class Naming(unittest.TestCase):
    def test_names_are_whole_seconds_with_three_digits(self):
        self.assertEqual([frame_name(s) for s in (0, 1.9, 74, 120.4, 1000)],
                         ["window_000s.png", "window_001s.png", "window_074s.png", "window_120s.png", "window_1000s.png"])

    def test_hash_depends_on_pixels_and_size(self):
        pixels = bytes(16)
        self.assertEqual(frame_hash(2, 2, pixels), frame_hash(2, 2, bytes(16)))
        self.assertNotEqual(frame_hash(2, 2, pixels), frame_hash(4, 1, pixels))
        self.assertNotEqual(frame_hash(2, 2, pixels), frame_hash(2, 2, bytes(15) + b"\x01"))


class Pace(unittest.TestCase):
    def test_frame_mean_is_the_channel_mean(self):
        self.assertEqual(frame_mean(4, 4, bytes([10, 20, 30, 255]) * 16), 20.0)
        self.assertEqual(frame_mean(1, 1, b""), 0.0)

    def test_menu_pictures_come_fast_until_the_first_loading_like_picture(self):
        pace = MenuPace(15, 5)
        self.assertEqual(pace.interval(), 5)
        for t, mean, ends in ((5, 3.0, False), (10, 95.0, False), (15, 90.0, False), (20, 60.0, False), (25, 30.0, True), (30, 10.0, False)):
            self.assertEqual(pace.note(t, mean), ends, (t, mean))
        self.assertEqual((pace.interval(), pace.loading_at), (15, 25), "back to -CaptureEvery from the loading-like picture on")

    def test_a_dark_start_is_not_a_loading_screen_and_no_menu_pace_keeps_the_normal_one(self):
        pace = MenuPace(15, 5)
        self.assertFalse(pace.note(5, 2.0) or pace.note(10, 0.5))   # black frames before any real screen
        self.assertEqual(pace.interval(), 5)
        self.assertEqual(MenuPace(15, 0).interval(), 15)


class PickWindow(unittest.TestCase):
    def test_the_output_window_wins_over_the_titles_own_window(self):
        windows = [(11, "Game"), (22, "Xbox2Windows abc1234 - output"), (33, "")]
        self.assertEqual(pick_window(windows), 22)
        self.assertEqual(pick_window(list(reversed(windows))), 22)

    def test_without_an_output_window_the_first_visible_one_is_used(self):
        self.assertEqual(pick_window([(11, "Game"), (33, "")]), 11)


class Frames(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.a, self.b, self.c = (picture(4, 4, seed) for seed in (1, 2, 3))

    def files(self):
        return sorted(os.listdir(self.dir.name))

    def test_identical_frames_are_saved_too_and_counted_as_repeats(self):
        series = Series(self.dir.name)
        results = [series.offer(t, 4, 4, p) for t, p in ((5, self.a), (10, self.a), (15, self.a), (20, self.b), (25, self.b), (30, self.a))]
        self.assertEqual(results, ["saved"] * 6)
        self.assertEqual(self.files(), ["window_%03ds.png" % t for t in (5, 10, 15, 20, 25, 30)])
        self.assertEqual((series.saved, series.repeated, series.first_change), (6, 3, 20))
        with open(os.path.join(self.dir.name, "window_020s.png"), "rb") as handle:
            self.assertEqual(decode_png(handle.read())[2], reference_rows(4, 4, self.b))

    def test_a_size_change_is_a_change(self):
        series = Series(self.dir.name)
        self.assertEqual(series.offer(1, 4, 4, bytes(64)), "saved")
        self.assertEqual(series.offer(2, 2, 8, bytes(64)), "saved")

    def test_file_cap_stops_saving_and_counts_what_it_dropped(self):
        series = Series(self.dir.name, cap=2)
        results = [series.offer(t, 4, 4, picture(4, 4, t)) for t in range(1, 6)]
        self.assertEqual(results, ["saved", "saved", "capped", "capped", "capped"])
        self.assertEqual(self.files(), ["window_001s.png", "window_002s.png"])
        self.assertEqual(series.offer(9, 4, 4, picture(4, 4, 50)), "capped")
        self.assertIn("file cap 2 reached (4 changed frames dropped)", series.summary())

    def test_default_cap_is_120_files(self):
        series = Series(self.dir.name, write=lambda path, data: None)
        for t in range(300):
            series.offer(t, 2, 2, bytes([t & 255, (t >> 8) & 255] * 8))
        self.assertEqual((series.saved, series.dropped), (120, 180))

    def test_summary_line(self):
        series = Series(self.dir.name)
        self.assertEqual(series.summary(), "window captures: 0 saved (0 identical to the one before), no change seen")
        series.offer(5, 4, 4, self.a)
        series.offer(10, 4, 4, self.a)
        self.assertEqual(series.summary(), "window captures: 2 saved (1 identical to the one before), no change seen")
        series.offer(74, 4, 4, self.b)
        series.miss()
        self.assertEqual(series.summary(), "window captures: 3 saved (1 identical to the one before), first change at 74 s, 1 ticks without a window")

    def test_a_failing_write_leaves_the_series_unchanged(self):
        def refuse(path, data):
            raise OSError("disk full")
        series = Series(self.dir.name, write=refuse)
        with self.assertRaises(OSError):
            series.offer(1, 4, 4, self.a)
        self.assertEqual((series.saved, series.repeated), (0, 0))
        series._write = Series._write_file
        self.assertEqual(series.offer(2, 4, 4, self.a), "saved")


class ShapeNote(unittest.TestCase):
    def test_the_first_picture_and_every_change_of_the_window_are_logged_once(self):
        from window_capture import shape_note
        first = shape_note(None, 1600, 900, 31, 5)
        self.assertEqual(first, "  window capture: first picture at 5 s: 1600x900 window, title text length 31")
        self.assertIsNone(shape_note((1600, 900, 31), 1600, 900, 31, 10), "the same window is not logged again")
        self.assertEqual(shape_note((1600, 900, 31), 1920, 1080, 12, 400),
                         "  window capture: the window changed at 400 s: 1920x1080 window, title text length 12")
        self.assertIn("unavailable", shape_note(None, 4, 4, "unavailable", 1))


if __name__ == "__main__":
    unittest.main()
