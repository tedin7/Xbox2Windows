#!/usr/bin/env python3
"""Standalone pure-Python synthetic PE32+ unit, CLI, path-guard, atomic-write, and mutation suite for tools/cpu_leaf_aot.py."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import pefile

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "cpu_leaf_aot.py"
sys.path.insert(0, str(TOOLS_DIR))

import cpu_leaf_aot  # noqa: E402


SECTION_ALIGN = 0x1000
FILE_ALIGN = 0x200
TEXT_RVA = 0x1000
RDATA_RVA = 0x2000
PDATA_RVA = 0x3000

# Exercises mov eax, ecx (89 c8), unsigned cmp imm8 + jb, signed cmp imm32 + jle,
# add/sub imm8/imm32, jmp short, mov [rdx], eax, and ret.
RICH_LEAF_CODE = bytes.fromhex(
    "89 c8"              # mov eax, ecx
    "83 f8 05 72 05"     # cmp eax, 5; jb +5 -> sub
    "83 c0 03"           # add eax, 3
    "eb 03"              # jmp short +3 -> cmp imm32
    "83 e8 07"           # sub eax, 7
    "3d 9c ff ff ff"     # cmp eax, -100
    "7e 07"              # jle +7 -> add imm32
    "2d 04 01 00 00"     # sub eax, 260
    "eb 05"              # jmp short +5 -> store
    "05 00 02 00 00"     # add eax, 512
    "89 02"              # mov [rdx], eax
    "c3"                 # ret
)

SIMPLE_LEAF_CODE = bytes.fromhex("8b c1 83 c0 01 89 02 c3")


def build_synthetic_leaf_dll(
    *,
    code: bytes = SIMPLE_LEAF_CODE,
    export: str = "leaf",
    machine: int = 0x8664,
    magic: int = 0x20B,
    directories: int = 16,
    duplicate_export: bool = False,
    forwarder: bytes | None = None,
    executable_text: bool = True,
) -> bytes:
    """Build a minimal synthetic PE32+ (or deliberately malformed) DLL image in memory."""
    rdata = bytearray()
    rdata += bytes([1, 0, 0, 0])  # UNWIND_INFO v1, no prolog, no codes
    exports_rva = RDATA_RVA + len(rdata)

    if forwarder is not None:
        fwd_rva = RDATA_RVA + len(rdata) + 40 + 4 + 4 + 2
        names_rva = fwd_rva + len(forwarder) + 1
        dll_name_rva = names_rva + len(export) + 1
        rdata += struct.pack(
            "<IIHHIIIIIII",
            0, 0, 0, 0, dll_name_rva, 1, 1, 1,
            exports_rva + 40, exports_rva + 44, exports_rva + 48,
        )
        rdata += struct.pack("<IIH", fwd_rva, names_rva, 0)
        rdata += forwarder + b"\0" + export.encode("ascii") + b"\0x.dll\0"
    elif duplicate_export:
        names_rva = RDATA_RVA + len(rdata) + 40 + 8 + 8 + 4
        name_bytes = export.encode("ascii") + b"\0"
        dll_name_rva = names_rva + len(name_bytes)
        rdata += struct.pack(
            "<IIHHIIIIIII",
            0, 0, 0, 0, dll_name_rva, 1, 2, 2,
            exports_rva + 40, exports_rva + 48, exports_rva + 56,
        )
        rdata += struct.pack("<II", TEXT_RVA, TEXT_RVA)
        rdata += struct.pack("<II", names_rva, names_rva)
        rdata += struct.pack("<HH", 0, 1)
        rdata += name_bytes + b"x.dll\0"
    else:
        names_rva = RDATA_RVA + len(rdata) + 40 + 4 + 4 + 2
        dll_name_rva = names_rva + len(export) + 1
        rdata += struct.pack(
            "<IIHHIIIIIII",
            0, 0, 0, 0, dll_name_rva, 1, 1, 1,
            exports_rva + 40, exports_rva + 44, exports_rva + 48,
        )
        rdata += struct.pack("<IIH", TEXT_RVA, names_rva, 0)
        rdata += export.encode("ascii") + b"\0x.dll\0"

    pdata = struct.pack("<III", TEXT_RVA, TEXT_RVA + len(code), RDATA_RVA)
    text_chars = 0x60000020 if executable_text else 0x40000040
    sections = (
        (b".text", TEXT_RVA, code, text_chars),
        (b".rdata", RDATA_RVA, bytes(rdata), 0x40000040),
        (b".pdata", PDATA_RVA, pdata, 0x40000040),
    )

    entries = [(0, 0)] * 16
    entries[0] = (exports_rva, len(rdata) - 4)
    entries[3] = (PDATA_RVA, len(pdata))

    optional = struct.pack(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        magic, 14, 0, len(code), 0, 0, TEXT_RVA, TEXT_RVA,
        0x140000000, SECTION_ALIGN, FILE_ALIGN, 6, 0, 0, 0, 6, 0, 0,
        0x4000, FILE_ALIGN, 0, 3, 0x8160,
        0x100000, SECTION_ALIGN, 0x100000, SECTION_ALIGN, 0, directories,
    )
    optional += b"".join(struct.pack("<II", *entry) for entry in entries[:directories])

    headers = b""
    body = b""
    offset = FILE_ALIGN
    for name, va, data, flags in sections:
        raw_size = -(-len(data) // FILE_ALIGN) * FILE_ALIGN
        headers += name.ljust(8, b"\0") + struct.pack(
            "<IIIIIIHHI", len(data), va, raw_size, offset + len(body), 0, 0, 0, 0, flags,
        )
        body += data.ljust(raw_size, b"\0")

    coff = struct.pack("<HHIIIHH", machine, len(sections), 0, 0, 0, len(optional), 0x2022)
    image = b"MZ" + bytes(58) + struct.pack("<I", 0x40) + b"PE\0\0" + coff + optional + headers
    return image.ljust(FILE_ALIGN, b"\0") + body


def _run_main(module, argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = module.main(argv)
        except SystemExit as exc:
            code = exc.code
            if not isinstance(code, int):
                if code is not None:
                    stderr.write(f"{code}\n")
                code = 0 if code is None else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class CpuLeafAotPositiveTests(unittest.TestCase):
    def test_translate_semantics_and_bytes_like_inputs(self) -> None:
        expected = "\n".join([
            "#include <cstdint>",
            'extern "C" __declspec(dllexport) std::uint32_t lifted(std::uint32_t input, std::uint32_t* output) {',
            "    std::uint32_t eax = input;",
            "    if (eax < 5u) goto loc_c;",
            "    eax += 3u;",
            "    goto loc_f;",
            "loc_c:",
            "    eax -= 7u;",
            "loc_f:",
            "    if (static_cast<std::int32_t>(eax) <= static_cast<std::int32_t>(4294967196u)) goto loc_1d;",
            "    eax -= 260u;",
            "    goto loc_22;",
            "loc_1d:",
            "    eax += 512u;",
            "loc_22:",
            "    *output = eax;",
            "    return eax;",
            "}",
            "",
        ])
        for payload in (RICH_LEAF_CODE, bytearray(RICH_LEAF_CODE), memoryview(RICH_LEAF_CODE)):
            with self.subTest(payload_type=type(payload).__name__):
                self.assertEqual(cpu_leaf_aot.translate(0x1000, payload), expected)

        simple_cpp = cpu_leaf_aot.translate(0, SIMPLE_LEAF_CODE)
        self.assertIn("eax += 1u;", simple_cpp)
        self.assertIn("*output = eax;", simple_cpp)
        self.assertIn("return eax;", simple_cpp)

    def test_pe_helpers_atomic_cli_nested_dirs_and_entrypoint(self) -> None:
        raw = build_synthetic_leaf_dll(code=RICH_LEAF_CODE, export="branch_leaf")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dll = base / "leaf.dll"
            dll.write_bytes(raw)

            rva, code = cpu_leaf_aot.function_bytes(dll, "branch_leaf")
            self.assertEqual(rva, TEXT_RVA)
            self.assertEqual(code, RICH_LEAF_CODE)
            with pefile.PE(str(dll), fast_load=True) as pe:
                self.assertEqual(cpu_leaf_aot.export_rva(pe, "branch_leaf"), TEXT_RVA)
                self.assertEqual(cpu_leaf_aot.export_rva(pe, 1), TEXT_RVA)
                self.assertEqual(cpu_leaf_aot.bounded_code(pe, TEXT_RVA), RICH_LEAF_CODE)

            expected_cpp = cpu_leaf_aot.translate(rva, code)
            nested_out = base / "nested" / "dir" / "lifted.cpp"
            self.assertFalse(nested_out.parent.exists())

            real_replace = os.replace
            replaced: list[tuple[Path, Path]] = []

            def tracking_replace(src, dst):
                src_p, dst_p = Path(src), Path(dst)
                replaced.append((src_p, dst_p))
                self.assertEqual(src_p.parent.resolve(), dst_p.parent.resolve())
                self.assertEqual(src_p.suffix, ".tmp")
                return real_replace(src, dst)

            with mock.patch.object(cpu_leaf_aot.os, "replace", side_effect=tracking_replace):
                code_rc, out, err = _run_main(cpu_leaf_aot, [str(dll), "branch_leaf", str(nested_out)])

            self.assertIs(type(code_rc), int)
            self.assertEqual(code_rc, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(nested_out.read_text(encoding="utf-8"), expected_cpp)
            self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["lifted.cpp"])
            self.assertEqual(len(replaced), 1)
            self.assertEqual(replaced[0][1].resolve(), nested_out.resolve())

            # Atomic overwrite replaces stale output with 0 leftover .tmp files.
            nested_out.write_text("// stale\n", encoding="utf-8")
            proc = _run_subprocess(str(dll), "branch_leaf", str(nested_out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(proc.stderr, "")
            self.assertEqual(nested_out.read_text(encoding="utf-8"), expected_cpp)
            self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["lifted.cpp"])

            # argv=None forwards sys.argv cleanly.
            nested_out.write_text("// stale-again\n", encoding="utf-8")
            with mock.patch.object(sys, "argv", [str(TOOL), str(dll), "branch_leaf", str(nested_out)]):
                rc_none, out_none, err_none = _run_main(cpu_leaf_aot, None)
            self.assertEqual(rc_none, 0, err_none)
            self.assertEqual(out_none, "")
            self.assertEqual(err_none, "")
            self.assertEqual(nested_out.read_text(encoding="utf-8"), expected_cpp)

            # __main__ entrypoint raises SystemExit(0) on success and SystemExit(1) cleanly on failure.
            for argv, expected_exit, err_frag in (
                ([str(TOOL), str(dll), "branch_leaf", str(nested_out)], 0, ""),
                ([str(TOOL), str(dll), "missing_export", str(nested_out)], 1, "expected one local export"),
            ):
                stdout_m, stderr_m = io.StringIO(), io.StringIO()
                with (
                    mock.patch.object(sys, "argv", argv),
                    redirect_stdout(stdout_m),
                    redirect_stderr(stderr_m),
                    self.assertRaises(SystemExit) as exited,
                ):
                    runpy.run_path(str(TOOL), run_name="__main__")
                self.assertEqual(exited.exception.code, expected_exit)
                self.assertEqual(stdout_m.getvalue(), "")
                self.assertNotIn("Traceback", stderr_m.getvalue())
                if err_frag:
                    self.assertIn(err_frag, stderr_m.getvalue())


class CpuLeafAotNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 1,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(cpu_leaf_aot, argv)
        if expected_code is None:
            self.assertNotEqual(code, 0)
        else:
            self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip())
        for fragment in fragments:
            self.assertIn(fragment.lower(), err.lower())

        if check_subprocess:
            proc = _run_subprocess(*argv)
            if expected_code is None:
                self.assertNotEqual(proc.returncode, 0)
            else:
                self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            for fragment in fragments:
                self.assertIn(fragment.lower(), proc.stderr.lower())
        return err

    def test_upfront_path_and_export_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "leaf.dll"
            image.write_bytes(b"synthetic-image")
            dir_image = base / "dir_image"
            dir_image.mkdir()
            same_cpp = base / "same.cpp"
            same_cpp.write_bytes(b"synthetic-cpp")
            hardlink_cpp = base / "hardlink.cpp"
            hardlink_cpp.hardlink_to(image)
            sub_dir = base / "sub"
            sub_dir.mkdir()
            out_dir = base / "dir_out.cpp"
            out_dir.mkdir()
            file_parent = base / "file_parent.txt"
            file_parent.write_text("keep-parent", encoding="utf-8")
            absent_out = base / "absent_dir" / "lifted.cpp"

            with (
                mock.patch.object(
                    cpu_leaf_aot,
                    "function_bytes",
                    side_effect=AssertionError("function_bytes called past upfront validation"),
                ),
                mock.patch.object(
                    cpu_leaf_aot,
                    "translate",
                    side_effect=AssertionError("translate called past upfront validation"),
                ),
                mock.patch.object(
                    cpu_leaf_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called past upfront validation"),
                ),
            ):
                for label, argv, frags in (
                    ("missing_image", [str(base / "missing.dll"), "leaf", str(absent_out)], ("image file not found",)),
                    ("dir_image", [str(dir_image), "leaf", str(absent_out)], ("image path is a directory",)),
                    ("empty_image", ["", "leaf", str(absent_out)], ("image path is a directory",)),
                    ("empty_export", [str(image), "", str(absent_out)], ("export name must not be empty",)),
                    ("blank_export", [str(image), "   ", str(absent_out)], ("export name must not be empty",)),
                    ("non_cpp_suffix", [str(image), "leaf", str(base / "out.txt")], (".cpp suffix",)),
                    ("empty_output", [str(image), "leaf", ""], (".cpp suffix",)),
                    ("dot_cpp_only", [str(image), "leaf", str(base / ".cpp")], (".cpp suffix",)),
                    ("output_is_dir", [str(image), "leaf", str(out_dir)], ("output path is a directory",)),
                    ("parent_is_file", [str(image), "leaf", str(file_parent / "lifted.cpp")], ("output parent is not a directory",)),
                    ("ancestor_is_file", [str(image), "leaf", str(file_parent / "sub" / "lifted.cpp")], ("output parent is not a directory",)),
                    ("same_path", [str(same_cpp), "leaf", str(same_cpp)], ("must not match input image",)),
                    ("resolve_alias", [str(same_cpp), "leaf", str(sub_dir / ".." / "same.cpp")], ("must not match input image",)),
                    ("hardlink_samefile", [str(image), "leaf", str(hardlink_cpp)], ("must not match input image",)),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, expected_code=1, fragments=frags)

            self.assertFalse(absent_out.parent.exists())
            self.assertEqual(same_cpp.read_bytes(), b"synthetic-cpp")
            self.assertEqual(image.read_bytes(), b"synthetic-image")
            self.assertEqual(file_parent.read_text(encoding="utf-8"), "keep-parent")
            self.assertFalse((base / "out.txt").exists())

    def test_output_symlink_rejected_upfront(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "leaf.dll"
            image.write_bytes(b"synthetic-image")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            real_target = out_dir / "real.cpp"
            real_target.write_text("untouched", encoding="utf-8")

            with (
                mock.patch.object(Path, "is_symlink", return_value=True),
                mock.patch.object(
                    cpu_leaf_aot,
                    "function_bytes",
                    side_effect=AssertionError("function_bytes called for symlink output"),
                ),
                mock.patch.object(
                    cpu_leaf_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for symlink output"),
                ),
            ):
                self._assert_clean_failure(
                    [str(image), "leaf", str(out_dir / "mock_link.cpp")],
                    expected_code=1,
                    fragments=("output path must not be a symlink",),
                    check_subprocess=False,
                )

            live_link = out_dir / "live_link.cpp"
            dangling_link = out_dir / "dangling_link.cpp"
            try:
                live_link.symlink_to(real_target)
                dangling_link.symlink_to(out_dir / "missing.cpp")
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                self.skipTest("this Windows account cannot create symlinks (WinError 1314)")

            with (
                mock.patch.object(
                    cpu_leaf_aot,
                    "function_bytes",
                    side_effect=AssertionError("function_bytes called for symlink output"),
                ),
                mock.patch.object(
                    cpu_leaf_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for symlink output"),
                ),
            ):
                for label, link in (("live_symlink", live_link), ("dangling_symlink", dangling_link)):
                    with self.subTest(case=label):
                        self._assert_clean_failure(
                            [str(image), "leaf", str(link)],
                            expected_code=1,
                            fragments=("output path must not be a symlink",),
                        )

            self.assertEqual(real_target.read_text(encoding="utf-8"), "untouched")
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["dangling_link.cpp", "live_link.cpp", "real.cpp"],
            )

    def test_pe_and_translation_cli_errors_create_no_dirs_or_temps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            absent_out = base / "never_created" / "lifted.cpp"

            cases = (
                ("corrupt_pe", b"not a PE32+ image", "leaf", ("dos header",)),
                ("x86_pe32", build_synthetic_leaf_dll(machine=0x014C, magic=0x10B), "leaf", ("expected amd64 pe32+",)),
                ("missing_pdata", build_synthetic_leaf_dll(directories=3), "leaf", ("no runtime-function directory",)),
                ("missing_export", build_synthetic_leaf_dll(), "other", ("expected one local export for other",)),
                ("duplicate_export", build_synthetic_leaf_dll(duplicate_export=True), "leaf", ("expected one local export for leaf",)),
                ("forwarder_export", build_synthetic_leaf_dll(forwarder=b"other.fn"), "leaf", ("expected one local export for leaf",)),
                ("non_exec_section", build_synthetic_leaf_dll(executable_text=False), "leaf", ("invalid file-backed range",)),
                ("bad_prologue", build_synthetic_leaf_dll(code=b"\x90\xc3"), "leaf", ("expected mov eax, ecx",)),
            )

            for label, payload, export_name, frags in cases:
                with self.subTest(case=label):
                    dll = base / f"{label}.dll"
                    dll.write_bytes(payload)
                    self._assert_clean_failure(
                        [str(dll), export_name, str(absent_out)],
                        expected_code=1,
                        fragments=frags,
                    )

            self.assertFalse(absent_out.parent.exists())
            self.assertEqual([p for p in base.iterdir() if p.suffix == ".tmp"], [])

    def test_atomic_write_failure_paths_leave_zero_tmp_files(self) -> None:
        raw = build_synthetic_leaf_dll()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dll = base / "leaf.dll"
            dll.write_bytes(raw)
            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_cpp = out_dir / "lifted.cpp"
            out_cpp.write_text("original", encoding="utf-8")

            # 1) os.replace failure preserves original file and unlinks sibling temp.
            with mock.patch.object(cpu_leaf_aot.os, "replace", side_effect=OSError("simulated replace failure")):
                self._assert_clean_failure(
                    [str(dll), "leaf", str(out_cpp)],
                    expected_code=1,
                    fragments=("simulated replace failure",),
                    check_subprocess=False,
                )
            self.assertEqual(out_cpp.read_text(encoding="utf-8"), "original")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["lifted.cpp"])

            # 2) UTF-8 encoding failure (lone surrogate) unlinks sibling temp and preserves original.
            with mock.patch.object(cpu_leaf_aot, "translate", return_value="bad-surrogate-\ud800"):
                self._assert_clean_failure(
                    [str(dll), "leaf", str(out_cpp)],
                    expected_code=1,
                    fragments=("surrogates not allowed",),
                    check_subprocess=False,
                )
            self.assertEqual(out_cpp.read_text(encoding="utf-8"), "original")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["lifted.cpp"])

            # 3) Write-time OSError inside handle unlinks sibling temp and preserves original.
            with mock.patch.object(cpu_leaf_aot.tempfile, "NamedTemporaryFile", side_effect=OSError("disk full")):
                self._assert_clean_failure(
                    [str(dll), "leaf", str(out_cpp)],
                    expected_code=1,
                    fragments=("disk full",),
                    check_subprocess=False,
                )
            self.assertEqual(out_cpp.read_text(encoding="utf-8"), "original")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["lifted.cpp"])

    def test_translate_input_validation_and_argparse_errors(self) -> None:
        for bad_start in (True, False, -1, -0x1000, 1.5, "0x1000", None):
            with self.subTest(bad_start=bad_start):
                with self.assertRaisesRegex(ValueError, "start RVA"):
                    cpu_leaf_aot.translate(bad_start, SIMPLE_LEAF_CODE)

        for bad_code in ("8bc1c3", None, 123, [0x8B, 0xC1, 0xC3]):
            with self.subTest(bad_code=bad_code):
                with self.assertRaisesRegex(ValueError, "bytes-like code"):
                    cpu_leaf_aot.translate(0x1000, bad_code)

        for body, msg in (
            ("90 c3", "unsupported or incomplete instruction at RVA 0x1002"),
            ("eb 01 83 c0 05 c3", "branch into an instruction at RVA 0x1005"),
            ("eb 7f c3", "expected forward in-function branch at RVA 0x1002"),
            ("89 02", "missing terminal return at RVA 0x1004"),
        ):
            with self.subTest(body=body):
                with self.assertRaisesRegex(ValueError, msg):
                    cpu_leaf_aot.translate(0x1000, bytes.fromhex("8b c1 " + body))

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dll = base / "leaf.dll"
            dll.write_bytes(b"synthetic")
            for argv, frags in (
                ([], ("required",)),
                ([str(dll)], ("required",)),
                ([str(dll), "leaf"], ("required",)),
                ([str(dll), "leaf", str(base / "out.cpp"), "--bogus-flag"], ("unrecognized",)),
            ):
                with self.subTest(argv=argv):
                    self._assert_clean_failure(argv, expected_code=None, fragments=frags)

    def test_fast_temp_mutations(self) -> None:
        source = TOOL.read_text(encoding="utf-8")
        raw = build_synthetic_leaf_dll()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dll = base / "leaf.dll"
            dll.write_bytes(raw)
            dir_image = base / "dir_image"
            dir_image.mkdir()
            hardlink_cpp = base / "hardlink.cpp"
            hardlink_cpp.hardlink_to(dll)
            file_parent = base / "file_parent.txt"
            file_parent.write_text("keep", encoding="utf-8")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_cpp = out_dir / "lifted.cpp"
            out_cpp.write_text("original", encoding="utf-8")

            def _check_drop_dir_image(mod) -> None:
                _, _, err = _run_main(mod, [str(dir_image), "leaf", str(out_dir / "a.cpp")])
                self.assertNotIn("image path is a directory", err.lower())

            def _check_drop_image_is_file(mod) -> None:
                with mock.patch.object(mod, "function_bytes", side_effect=RuntimeError("reached_function_bytes")):
                    with self.assertRaisesRegex(RuntimeError, "reached_function_bytes"):
                        _run_main(mod, [str(base / "missing.dll"), "leaf", str(out_dir / "a.cpp")])

            def _check_drop_export_guard(mod) -> None:
                with mock.patch.object(mod, "function_bytes", side_effect=RuntimeError("reached_function_bytes")):
                    with self.assertRaisesRegex(RuntimeError, "reached_function_bytes"):
                        _run_main(mod, [str(dll), "   ", str(out_dir / "a.cpp")])

            def _check_drop_cpp_suffix(mod) -> None:
                code, _, err = _run_main(mod, [str(dll), "leaf", str(out_dir / "a.txt")])
                self.assertEqual(code, 0, err)

            def _check_drop_symlink_guard(mod) -> None:
                with mock.patch.object(Path, "is_symlink", return_value=True):
                    _, _, err = _run_main(mod, [str(dll), "leaf", str(out_dir / "sym.cpp")])
                self.assertNotIn("output path must not be a symlink", err.lower())

            def _check_drop_parent_file_guard(mod) -> None:
                _, _, err = _run_main(mod, [str(dll), "leaf", str(file_parent / "lifted.cpp")])
                self.assertNotIn("output parent is not a directory", err.lower())

            def _check_drop_samefile_guard(mod) -> None:
                with mock.patch.object(mod, "function_bytes", side_effect=RuntimeError("reached_function_bytes")):
                    with self.assertRaisesRegex(RuntimeError, "reached_function_bytes"):
                        _run_main(mod, [str(dll), "leaf", str(hardlink_cpp)])

            def _check_drop_tmp_unlink(mod) -> None:
                with mock.patch.object(mod.os, "replace", side_effect=OSError("boom")):
                    code, _, _ = _run_main(mod, [str(dll), "leaf", str(out_cpp)])
                self.assertEqual(code, 1)
                leaked = [p for p in out_dir.iterdir() if p.suffix == ".tmp"]
                self.assertTrue(leaked, "expected leaked .tmp file under mutation")
                for p in leaked:
                    p.unlink(missing_ok=True)

            def _check_drop_bool_start_guard(mod) -> None:
                self.assertIn("return eax;", mod.translate(True, SIMPLE_LEAF_CODE))

            def _check_drop_bytes_like_guard(mod) -> None:
                with self.assertRaises(AttributeError):
                    mod.translate(0x1000, memoryview(SIMPLE_LEAF_CODE))

            mutations = (
                (
                    "drop_image_is_dir_guard",
                    '    if image.is_dir():\n        raise IsADirectoryError(f"image path is a directory: {image}")\n',
                    "    pass\n",
                    _check_drop_dir_image,
                ),
                (
                    "drop_image_is_file_guard",
                    '    if not image.is_file():\n        raise FileNotFoundError(f"image file not found: {image}")\n',
                    "    pass\n",
                    _check_drop_image_is_file,
                ),
                (
                    "drop_export_non_empty_guard",
                    '    if not isinstance(export, str) or not export.strip():\n        raise ValueError("export name must not be empty")\n',
                    "    pass\n",
                    _check_drop_export_guard,
                ),
                (
                    "drop_cpp_suffix_guard",
                    '    if output.suffix.lower() != ".cpp":\n        raise ValueError(f"output path must have .cpp suffix: {output}")\n',
                    "    pass\n",
                    _check_drop_cpp_suffix,
                ),
                (
                    "drop_output_is_symlink_guard",
                    '    if output.is_symlink():\n        raise ValueError(f"output path must not be a symlink: {output}")\n',
                    "    pass\n",
                    _check_drop_symlink_guard,
                ),
                (
                    "drop_parent_is_file_guard",
                    '        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):\n'
                    '            raise NotADirectoryError(f"output parent is not a directory: {parent}")\n',
                    "        pass\n",
                    _check_drop_parent_file_guard,
                ),
                (
                    "drop_samefile_guard",
                    "    if output.resolve() == image.resolve() or (output.exists() and image.samefile(output)):\n",
                    "    if output.resolve() == image.resolve():\n",
                    _check_drop_samefile_guard,
                ),
                (
                    "drop_tmp_unlink",
                    "        if tmp_path is not None:\n            tmp_path.unlink(missing_ok=True)\n",
                    "        pass\n",
                    _check_drop_tmp_unlink,
                ),
                (
                    "drop_bool_start_guard",
                    "    if isinstance(start, bool) or not isinstance(start, int):\n",
                    "    if not isinstance(start, int):\n",
                    _check_drop_bool_start_guard,
                ),
                (
                    "drop_bytes_like_guard",
                    "    if not isinstance(code, bytes):\n"
                    "        try:\n"
                    "            code = memoryview(code).tobytes()\n"
                    "        except TypeError as exc:\n"
                    '            raise ValueError(f"expected bytes-like code, got {type(code).__name__}") from exc\n',
                    "    pass\n",
                    _check_drop_bytes_like_guard,
                ),
            )

            for label, target, replacement, check_fn in mutations:
                with self.subTest(mutation=label):
                    self.assertEqual(source.count(target), 1, target)
                    mod = types.ModuleType(f"cpu_leaf_aot_mut_{label}")
                    mod.__file__ = str(TOOL)
                    sys.modules[mod.__name__] = mod
                    try:
                        exec(compile(source.replace(target, replacement, 1), str(TOOL), "exec"), mod.__dict__)
                        check_fn(mod)
                    finally:
                        sys.modules.pop(mod.__name__, None)


if __name__ == "__main__":
    unittest.main()
