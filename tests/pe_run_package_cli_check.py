#!/usr/bin/env python3
"""Pure-Python CLI, path-guard, and error-handling suite for tools/pe_run_package.py."""

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

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "pe_run_package.py"
sys.path.insert(0, str(TOOLS_DIR))
import pe_run_package  # noqa: E402


def _minimal_pe32_plus(*, dll_name: bytes = b"kernel32.dll", symbol_name: bytes = b"ExitProcess") -> bytes:
    """Build a synthetic PE32+ executable with one normal import table entry."""
    section_rva = 0x1000
    raw_offset = 0x200
    raw_size = 0x400
    image_base = 0x180000000
    rdata = bytearray(raw_size)

    def put(offset: int, blob: bytes) -> int:
        rdata[offset:offset + len(blob)] = blob
        return section_rva + offset

    dll_rva = put(0x00, dll_name + b"\0")
    hint_rva = put(0x20, struct.pack("<H", 0) + symbol_name + b"\0")
    ilt_rva = put(0x60, struct.pack("<QQ", hint_rva, 0))
    iat_rva = put(0x80, struct.pack("<QQ", hint_rva, 0))
    idt_size = 40
    put(0xA0, struct.pack("<5I", ilt_rva, 0, 0, dll_rva, iat_rva) + bytes(20))
    idt_rva = section_rva + 0xA0

    pe_offset = 0x80
    buf = bytearray(raw_offset + raw_size)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 60, pe_offset)
    buf[pe_offset:pe_offset + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_offset + 4, 0x8664, 1, 0, 0, 0, 240, 0x2022)
    opt = pe_offset + 24
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        buf,
        opt,
        0x20B, 0, 0, 0, raw_size, 0, 0, section_rva, image_base,
        0x1000, 0x200, 6, 0, 0, 0, 6, 0, 0, 0x2000, raw_offset,
        0, 3, 0, 0, 0, 0, 0, 0, 16,
    )
    struct.pack_into("<II", buf, opt + 112 + 8, idt_rva, idt_size)
    struct.pack_into(
        "<8sIIIIIIHHI",
        buf,
        opt + 240,
        b".rdata\0\0", raw_size, section_rva, raw_size, raw_offset, 0, 0, 0, 0, 0x40000040,
    )
    buf[raw_offset:raw_offset + raw_size] = rdata
    return bytes(buf)


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            rc = pe_run_package.main(argv)
        except SystemExit as exc:
            code = exc.code
            rc = 0 if code is None else (int(code) if isinstance(code, int) else 1)
            if not isinstance(code, (type(None), int)):
                stderr.write(f"{code}\n")
    return rc, stdout.getvalue(), stderr.getvalue()


SEM_FAILCRITICALERRORS = 0x0001


def _thread_error_mode() -> int:
    kernel = pe_run_package.C.WinDLL("kernel32")
    get_mode = kernel.GetThreadErrorMode
    get_mode.argtypes, get_mode.restype = [], pe_run_package.W.DWORD
    return get_mode()


def _set_thread_error_mode(mode: int) -> int:
    """Set the calling thread's error mode and return the previous one."""
    kernel = pe_run_package.C.WinDLL("kernel32", use_last_error=True)
    set_mode = kernel.SetThreadErrorMode
    set_mode.argtypes = [pe_run_package.W.DWORD, pe_run_package.C.POINTER(pe_run_package.W.DWORD)]
    set_mode.restype = pe_run_package.W.BOOL
    previous = pe_run_package.W.DWORD()
    if not set_mode(mode, pe_run_package.C.byref(previous)):
        raise OSError(pe_run_package.C.get_last_error(), "SetThreadErrorMode failed")
    return previous.value


class _RecordingKernel:
    """kernel32 for pe_run_package.run whose LoadLibraryExW records the thread error mode of each call."""

    def __init__(self, real, modes):
        self._real = real
        self._modes = modes

    def __getattr__(self, name):
        function = getattr(self._real, name)
        if name != "LoadLibraryExW":
            return function
        modes = self._modes

        class Recorder:
            argtypes = restype = None

            def __call__(self, *args):
                function.argtypes, function.restype = self.argtypes, self.restype
                modes.append((args[0], _thread_error_mode()))
                return function(*args)

        return Recorder()


class PeRunPackageCliTests(unittest.TestCase):
    def test_provider_load_has_no_critical_error_dialog(self):
        # A provider that is not a loadable image (here a 0-byte file) raises a critical error inside
        # LoadLibraryExW. Without SEM_FAILCRITICALERRORS Windows shows a modal "Bad Image" dialog and the
        # load blocks until someone closes it. The flag must be set for the provider load only: the
        # thread's previous mode is back once the run returns, so the guest keeps the host's setting.
        # The thread starts the run with mode 0: a host that already set the flag (as some CI runners
        # do) would otherwise make the restore check pass whether or not the runner restores the mode.
        real_win_dll = pe_run_package.C.WinDLL
        modes = []

        def win_dll(name, *args, **kwargs):
            library = real_win_dll(name, *args, **kwargs)
            return _RecordingKernel(library, modes) if name == "kernel32" else library

        system_kernel = Path(pe_run_package.os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "kernel32.dll"
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "package"
            pkg.mkdir()
            (pkg / "app.exe").write_bytes(_minimal_pe32_plus())
            mapper = Path(tmp) / "mapper.dll"
            mapper.write_bytes(b"")
            original = _set_thread_error_mode(0)
            try:
                before = _thread_error_mode()
                with mock.patch.object(pe_run_package.C, "WinDLL", win_dll):
                    rc, stdout, stderr = _run_cli([
                        str(mapper), str(pkg), "app.exe", "--preflight-only",
                        "--shim", "kernel32.dll", str(system_kernel),
                    ])
                after = _thread_error_mode()
            finally:
                _set_thread_error_mode(original)
        self.assertEqual(before, 0, "the thread error mode could not be cleared before the run")
        self.assertEqual(rc, 0, stderr)
        self.assertEqual(json.loads(stdout)["missing"], [])
        self.assertEqual([Path(path).name.casefold() for path, _ in modes], ["kernel32.dll"])
        for path, mode in modes:
            self.assertTrue(mode & SEM_FAILCRITICALERRORS,
                            f"LoadLibraryExW({path}) ran with thread error mode {mode:#x}")
        self.assertEqual(after, before, "the thread error mode was not restored after the run")

    def test_positive_preflight_in_process_and_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg = base / "package"
            pkg.mkdir()
            (pkg / "app.exe").write_bytes(_minimal_pe32_plus())
            mapper = base / "mapper.dll"
            mapper.write_bytes(b"")
            shim = base / "kernel32_shim.dll"
            shim.write_bytes(b"")

            rc, stdout, stderr = _run_cli([
                str(mapper),
                str(pkg),
                "app.exe",
                "--preflight-only",
                "--shim",
                "kernel32.dll",
                str(shim),
            ])
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            payload = json.loads(stdout)
            self.assertEqual(payload["image_order"], ["app.exe"])
            self.assertFalse(payload["entry_executed"])
            self.assertEqual(payload["missing"], ["app.exe: kernel32.dll!ExitProcess"])
            self.assertEqual(sorted(p.name for p in pkg.iterdir()), ["app.exe"])

            # Non-existent mapper placeholder is also permitted in --preflight-only.
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(base / "absent_mapper.dll"), str(pkg), "app.exe", "--preflight-only"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            sub_payload = json.loads(proc.stdout)
            self.assertEqual(sub_payload["image_order"], ["app.exe"])
            self.assertFalse(sub_payload["entry_executed"])

    def test_path_and_argparse_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg = base / "package"
            pkg.mkdir()
            (pkg / "app.exe").write_bytes(_minimal_pe32_plus())
            mapper = base / "mapper.dll"
            mapper.write_bytes(b"")
            mapper_dir = base / "mapper_dir"
            mapper_dir.mkdir()
            shim_dir = base / "shim_dir"
            shim_dir.mkdir()
            not_a_dir = base / "regular.txt"
            not_a_dir.write_text("file", encoding="utf-8")

            # Missing args (Py3.14-safe argparse assertion).
            rc, _, stderr = _run_cli([])
            self.assertNotEqual(rc, 0)
            self.assertIn("required", stderr.lower())
            self.assertNotIn("Traceback", stderr)

            cases = (
                ("missing_package", [str(mapper), str(base / "missing_pkg"), "app.exe", "--preflight-only"], ""),
                ("file_package", [str(mapper), str(not_a_dir), "app.exe", "--preflight-only"], "directory"),
                ("dir_mapper", [str(mapper_dir), str(pkg), "app.exe", "--preflight-only"], "directory"),
                ("missing_shim", [str(mapper), str(pkg), "app.exe", "--shim", "kernel32.dll", str(base / "missing.dll"), "--preflight-only"], "missing.dll"),
                ("dir_shim", [str(mapper), str(pkg), "app.exe", "--shim", "kernel32.dll", str(shim_dir), "--preflight-only"], "directory"),
            )
            for label, argv, expected_substr in cases:
                with self.subTest(case=label):
                    with mock.patch.object(
                        pe_run_package,
                        "inspect",
                        side_effect=AssertionError(f"inspect called for invalid path case {label}"),
                    ):
                        rc, stdout, stderr = _run_cli(argv)
                    self.assertEqual(rc, 1)
                    self.assertEqual(stdout, "")
                    self.assertTrue(stderr.strip())
                    if expected_substr:
                        self.assertIn(expected_substr, stderr)
                    self.assertNotIn("Traceback", stderr)

    def test_exception_guards_and_subprocess_negative_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg = base / "package"
            pkg.mkdir()
            (pkg / "app.exe").write_bytes(b"MZ not a valid PE")
            mapper = base / "mapper.dll"
            mapper.write_bytes(b"")

            # pefile.PEFormatError -> exit 1 without traceback.
            rc, stdout, stderr = _run_cli([str(mapper), str(pkg), "app.exe", "--preflight-only"])
            self.assertEqual(rc, 1)
            self.assertEqual(stdout, "")
            self.assertTrue(stderr.strip())
            self.assertNotIn("Traceback", stderr)

            # struct.error -> exit 1 without traceback.
            with mock.patch.object(pe_run_package, "inspect", side_effect=struct.error("short unpack")):
                rc, stdout, stderr = _run_cli([str(mapper), str(pkg), "app.exe", "--preflight-only"])
            self.assertEqual(rc, 1)
            self.assertEqual(stdout, "")
            self.assertIn("short unpack", stderr)
            self.assertNotIn("Traceback", stderr)

            # Subprocess negative smoke test (mapper is a directory).
            mapper_dir = base / "mapper_dir"
            mapper_dir.mkdir()
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(mapper_dir), str(pkg), "app.exe", "--preflight-only"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(proc.stdout, "")
            self.assertIn("directory", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
