"""Periodic window pictures of a startup run (docs/startup-runtime.md, "Window captures").

debug_launch.py grabs the client area of the title's window as raw 32-bit BGRA rows. This module is the part
that needs no Windows API: the fast BGRA -> PNG conversion, the file names, the skip of a frame that equals
the previous one, the file cap and the one summary line. tests/startup_window_capture_check.py covers it.

PNG: ISO/IEC 15948 (W3C PNG Specification, Second Edition), truecolour 8 bit, filter type 0 on every row.
"""
import hashlib
import os
import struct
import zlib

FILE_CAP = 120  # periodic files per run; window.png at the end is not counted
PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def _chunk(kind, payload):
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def png_bytes(width, height, bgra, level=6):
    """PNG of width x height BGRA pixels (the alpha byte is dropped). Whole-buffer slice copies, no per-byte loop."""
    if width <= 0 or height <= 0 or len(bgra) != width * height * 4:
        raise ValueError("pixel buffer does not match %dx%d" % (width, height))
    rgb = bytearray(width * height * 3)
    rgb[0::3], rgb[1::3], rgb[2::3] = bgra[2::4], bgra[1::4], bgra[0::4]
    line = width * 3
    rows = bytearray((1 + line) * height)  # each row starts with its filter byte (0 = none)
    for y in range(height):
        rows[y * (1 + line) + 1:(y + 1) * (1 + line)] = rgb[y * line:(y + 1) * line]
    return (PNG_SIGNATURE + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            _chunk(b"IDAT", zlib.compress(bytes(rows), level)) + _chunk(b"IEND", b""))


def frame_name(seconds):
    """window_<sss>s.png: whole seconds since the process started, at least three digits."""
    return "window_%03ds.png" % int(seconds)


def frame_hash(width, height, bgra):
    return hashlib.blake2b(bgra, digest_size=16, key=struct.pack("<II", width, height)).digest()


def pick_window(windows):
    """windows: (handle, title text) of the visible windows of the process, in enumeration order. The window to photograph
    is the stand-in's output window (its title ends in "- output": the presented picture is shown there), else the first
    one. The title's own window is another, often wrong-sized window (Run H: a 1920x1080 picture of it)."""
    for handle, text in windows:
        if text.endswith("- output"):
            return handle
    return windows[0][0]


class Series:
    """The periodic frames of one run. Frames are offered in time order and every one is saved, until the file cap is
    reached: a frame identical to the previous one is saved too (a dark loading screen that looks "unchanged" is still a
    picture the reader wants, 26/11 Run I) and only counted in `repeated`."""

    def __init__(self, logdir, cap=FILE_CAP, level=1, write=None):
        self.logdir, self.cap, self.level = logdir, cap, level
        self._write = write or self._write_file
        self._last = None
        self.saved = self.repeated = self.missed = self.dropped = 0
        self.first_change = None  # seconds of the first saved frame after the baseline
        self.names = []

    @staticmethod
    def _write_file(path, data):
        with open(path, "wb") as handle:
            handle.write(data)

    @property
    def full(self):
        return self.saved >= self.cap

    def miss(self):
        """A tick with no usable window (none visible yet, empty client area, capture error)."""
        self.missed += 1

    def offer(self, seconds, width, height, bgra):
        """Returns "saved" or "capped"."""
        digest = frame_hash(width, height, bgra)
        repeat = digest == self._last
        if self.full:
            self.dropped += 1
            return "capped"
        name = frame_name(seconds)
        self._write(os.path.join(self.logdir, name), png_bytes(width, height, bgra, self.level))
        if repeat:
            self.repeated += 1
        elif self.saved and self.first_change is None:
            self.first_change = int(seconds)
        self._last = digest
        self.saved += 1
        self.names.append(name)
        return "saved"

    def summary(self):
        text = "window captures: %d saved (%d identical to the one before), " % (self.saved, self.repeated)
        text += "first change at %d s" % self.first_change if self.first_change is not None else "no change seen"
        if self.dropped:
            text += ", file cap %d reached (%d changed frames dropped)" % (self.cap, self.dropped)
        if self.missed:
            text += ", %d ticks without a window" % self.missed
        return text
