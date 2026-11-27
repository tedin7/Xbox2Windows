#!/usr/bin/env python3
"""Standalone pure-Python synthetic PE32+ unit, CLI, path-guard, atomic-write, and mutation suite for tools/cpu_call_aot.py."""

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
TOOL = TOOLS_DIR / "cpu_call_aot.py"
sys.path.insert(0, str(TOOLS_DIR))

import cpu_call_aot  # noqa: E402


SECTION_ALIGN = 0x1000
FILE_ALIGN = 0x200
TEXT_RVA = 0x1000
RDATA_RVA = 0x2000
PDATA_RVA = 0x3000

CALLER_PROLOGUE = bytes.fromhex("53 56 48 83 ec 28")
CALLER_EPILOGUE = bytes.fromhex("48 83 c4 28 5e 5b c3")
CALLEE_PROLOGUE = bytes.fromhex("53 56 48 83 ec 08")
CALLEE_EPILOGUE = bytes.fromhex("48 83 c4 08 5e 5b c3")


def build_synthetic_pe(
    *,
    functions: list[tuple[str | None, int, bytes]],
    exports: list[tuple[str | None, int, int]] | None = None,
    imports: list[tuple[str, list[tuple[str | None, int | None, int]]]] | None = None,
    text_executable: bool = True,
    directories: int = 16,
    overlapping_pdata: bool = False,
) -> bytes:
    """Build a synthetic AMD64 PE32+ DLL with multiple functions, exports, .pdata, and optional IAT imports."""
    text = bytearray()
    fn_spans: list[tuple[int, int]] = []
    for _name, rel_off, code in functions:
        if len(text) < rel_off:
            text.extend(b"\xcc" * (rel_off - len(text)))
        begin = TEXT_RVA + rel_off
        text[rel_off:rel_off + len(code)] = code
        fn_spans.append((begin, begin + len(code)))

    rdata = bytearray()
    unwind_rva = RDATA_RVA
    rdata += bytes([1, 0, 0, 0])  # UNWIND_INFO v1, no prolog, no codes

    def align_rdata(align: int = 8) -> int:
        while len(rdata) % align:
            rdata.append(0)
        return RDATA_RVA + len(rdata)

    # Export directory
    if exports is None:
        exports = [(name, idx + 1, TEXT_RVA + rel_off)
                   for idx, (name, rel_off, _code) in enumerate(functions) if name is not None]

    exports_rva = 0
    exports_size = 0
    if exports:
        exports_rva = align_rdata(4)
        base_ord = min(ord_num for _n, ord_num, _rva in exports)
        max_ord = max(ord_num for _n, ord_num, _rva in exports)
        num_funcs = max_ord - base_ord + 1
        named = sorted([(n, ord_num) for n, ord_num, _rva in exports if n is not None], key=lambda item: item[0])
        eat_off = len(rdata) + 40
        npt_off = eat_off + 4 * num_funcs
        eot_off = npt_off + 4 * len(named)
        str_off = eot_off + 2 * len(named)
        dll_name = b"synth.dll\0"
        rdata.extend(b"\x00" * (str_off - len(rdata)))
        dll_name_rva = RDATA_RVA + len(rdata)
        rdata.extend(dll_name)
        name_rvas: list[tuple[int, int]] = []
        for sym_name, ord_num in named:
            name_rvas.append((RDATA_RVA + len(rdata), ord_num - base_ord))
            rdata.extend(sym_name.encode("ascii") + b"\0")
        struct.pack_into(
            "<IIHHIIIIIII",
            rdata,
            exports_rva - RDATA_RVA,
            0, 0, 0, 0, dll_name_rva, base_ord, num_funcs, len(named),
            RDATA_RVA + eat_off, RDATA_RVA + npt_off, RDATA_RVA + eot_off,
        )
        for _n, ord_num, rva in exports:
            struct.pack_into("<I", rdata, eat_off + 4 * (ord_num - base_ord), rva)
        for idx, (nrva, ord_idx) in enumerate(name_rvas):
            struct.pack_into("<I", rdata, npt_off + 4 * idx, nrva)
            struct.pack_into("<H", rdata, eot_off + 2 * idx, ord_idx)
        exports_size = (RDATA_RVA + len(rdata)) - exports_rva

    # Import directory + ILT + IAT
    imports_rva = 0
    imports_size = 0
    if imports:
        # Place IAT slots first at known RVAs requested by callers.
        for _dll, items in imports:
            for _imp_name, _imp_ord, slot_rva in items:
                needed = (slot_rva - RDATA_RVA) + 16
                if len(rdata) < needed:
                    rdata.extend(b"\x00" * (needed - len(rdata)))

        desc_rva = align_rdata(4)
        imports_rva = desc_rva
        desc_count = len(imports) + 1
        rdata.extend(b"\x00" * (20 * desc_count))
        imports_size = 20 * desc_count

        for idx, (dll_name, items) in enumerate(imports):
            dll_rva = RDATA_RVA + len(rdata)
            rdata.extend(dll_name.encode("ascii") + b"\0")
            ilt_rva = align_rdata(8)
            rdata.extend(b"\x00" * (8 * (len(items) + 1)))
            first_slot_rva = items[0][2]
            for j, (imp_name, imp_ord, slot_rva) in enumerate(items):
                assert slot_rva == first_slot_rva + 8 * j
                if imp_ord is not None:
                    entry_val = (1 << 63) | imp_ord
                else:
                    ibn_rva = align_rdata(2)
                    rdata.extend(b"\x00\x00" + imp_name.encode("ascii") + b"\0")
                    entry_val = ibn_rva
                struct.pack_into("<Q", rdata, (ilt_rva - RDATA_RVA) + 8 * j, entry_val)
                struct.pack_into("<Q", rdata, (slot_rva - RDATA_RVA), entry_val)
            struct.pack_into(
                "<IIIII",
                rdata,
                (desc_rva - RDATA_RVA) + 20 * idx,
                ilt_rva, 0, 0, dll_rva, first_slot_rva,
            )

    pdata = bytearray()
    spans = sorted(fn_spans)
    for idx, (begin, end) in enumerate(spans):
        if overlapping_pdata and idx == 0 and len(spans) > 1:
            end = spans[1][0] + 4
        pdata.extend(struct.pack("<III", begin, end, unwind_rva))

    text_flags = 0x60000020 if text_executable else 0x40000040
    sections = (
        (b".text", TEXT_RVA, bytes(text), text_flags),
        (b".rdata", RDATA_RVA, bytes(rdata), 0x40000040),
        (b".pdata", PDATA_RVA, bytes(pdata), 0x40000040),
    )

    entries = [(0, 0)] * 16
    if exports_size:
        entries[0] = (exports_rva, exports_size)
    if imports_size:
        entries[1] = (imports_rva, imports_size)
    entries[3] = (PDATA_RVA, len(pdata))

    optional = struct.pack(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        0x20B, 14, 0, len(text), 0, 0, TEXT_RVA, TEXT_RVA,
        0x140000000, SECTION_ALIGN, FILE_ALIGN, 6, 0, 0, 0, 6, 0, 0,
        0x4000, FILE_ALIGN, 0, 3, 0x8160,
        0x100000, SECTION_ALIGN, 0x100000, SECTION_ALIGN, 0, directories,
    )
    optional += b"".join(struct.pack("<II", *entry) for entry in entries[:directories])

    headers = b""
    body = b""
    offset = FILE_ALIGN
    for name, va, data, flags in sections:
        raw_size = max(FILE_ALIGN, -(-len(data) // FILE_ALIGN) * FILE_ALIGN)
        headers += name.ljust(8, b"\0") + struct.pack(
            "<IIIIIIHHI", len(data), va, raw_size, offset + len(body), 0, 0, 0, 0, flags,
        )
        body += data.ljust(raw_size, b"\0")

    coff = struct.pack("<HHIIIHH", 0x8664, len(sections), 0, 0, 0, len(optional), 0x2022)
    image = b"MZ" + bytes(58) + struct.pack("<I", 0x40) + b"PE\0\0" + coff + optional + headers
    return image.ljust(FILE_ALIGN, b"\0") + body


def _make_direct_caller_and_callees() -> bytes:
    """Build a caller invoking two distinct callees at 0x1080 and 0x10c0 (caller at 0x1000)."""
    callee1_body = bytes.fromhex(
        "49 89 08"        # mov [r8], rcx
        "49 89 50 08"     # mov [r8+8], rdx
        "48 89 c8"        # mov rax, rcx
        "48 29 d0"        # sub rax, rdx
        "48 83 c0 0b"     # add rax, 11
        "49 89 40 10"     # mov [r8+16], rax
    )
    callee2_body = bytes.fromhex(
        "48 89 c8"        # mov rax, rcx
        "48 31 d0"        # xor rax, rdx
        "48 83 c0 1d"     # add rax, 29
        "49 89 40 10"     # mov [r8+16], rax
    )
    # Build caller with rel32 calls to 0x1080 and 0x10c0.
    part1 = bytes.fromhex(
        "48 89 cb"        # mov rbx, rcx
        "4c 89 c6"        # mov rsi, r8
        "48 83 c1 07"     # add rcx, 7
        "48 83 ea 03"     # sub rdx, 3
    )
    call1_rva = TEXT_RVA + len(CALLER_PROLOGUE) + len(part1)
    call1 = b"\xe8" + struct.pack("<i", (TEXT_RVA + 0x80) - (call1_rva + 5))
    part2 = bytes.fromhex(
        "48 89 c1"        # mov rcx, rax
        "48 01 d9"        # add rcx, rbx
        "48 89 da"        # mov rdx, rbx
        "49 89 f0"        # mov r8, rsi
    )
    call2_rva = call1_rva + 5 + len(part2)
    call2 = b"\xe8" + struct.pack("<i", (TEXT_RVA + 0xc0) - (call2_rva + 5))
    part3 = bytes.fromhex("48 01 d8")  # add rax, rbx
    caller = CALLER_PROLOGUE + part1 + call1 + part2 + call2 + part3 + CALLER_EPILOGUE
    return build_synthetic_pe(
        functions=[
            ("call_pair", 0x00, caller),
            ("mix_args", 0x80, CALLEE_PROLOGUE + callee1_body + CALLEE_EPILOGUE),
            ("mix_other", 0xc0, CALLEE_PROLOGUE + callee2_body + CALLEE_EPILOGUE),
        ],
    )


def _make_provider_dll(*, export_name: str | None = "mix_args", ordinal: int = 1, executable: bool = True) -> bytes:
    callee_body = bytes.fromhex("48 89 c8 48 83 c0 01")  # mov rax, rcx; add rax, 1
    code = CALLEE_PROLOGUE + callee_body + CALLEE_EPILOGUE
    return build_synthetic_pe(
        functions=[(export_name, 0x00, code)],
        exports=[(export_name, ordinal, TEXT_RVA)],
        text_executable=executable,
    )


def _make_iat_caller_dll(*, dll_name: str, imp_name: str | None, imp_ord: int | None) -> tuple[bytes, int]:
    slot_rva = RDATA_RVA + 0x200
    part1 = bytes.fromhex("48 89 cb 4c 89 c6")  # mov rbx, rcx; mov rsi, r8
    call1_rva = TEXT_RVA + len(CALLER_PROLOGUE) + len(part1)
    call1 = b"\xff\x15" + struct.pack("<i", slot_rva - (call1_rva + 6))
    part2 = bytes.fromhex("48 89 c1 48 89 da 49 89 f0")  # mov rcx, rax; mov rdx, rbx; mov r8, rsi
    call2_rva = call1_rva + 6 + len(part2)
    call2 = b"\xff\x15" + struct.pack("<i", slot_rva - (call2_rva + 6))
    caller = CALLER_PROLOGUE + part1 + call1 + part2 + call2 + CALLER_EPILOGUE
    raw = build_synthetic_pe(
        functions=[("call_mix", 0x00, caller)],
        imports=[(dll_name, [(imp_name, imp_ord, slot_rva)])],
    )
    return raw, slot_rva


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


class CpuCallAotPositiveTests(unittest.TestCase):
    def test_direct_calls_concrete_cpp_and_atomic_cli(self) -> None:
        raw = _make_direct_caller_and_callees()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dll = base / "caller.dll"
            dll.write_bytes(raw)
            expected = cpu_call_aot.translate(dll, "call_pair")

            self.assertIn("#include <cstdint>", expected)
            self.assertIn("using word = std::uint64_t;", expected)
            self.assertIn('extern "C" __declspec(dllexport) __declspec(noinline) word lifted_callee(', expected)
            self.assertIn('extern "C" __declspec(dllexport) __declspec(noinline) word lifted_callee2(', expected)
            self.assertIn('extern "C" __declspec(dllexport) __declspec(noinline) word lifted(', expected)
            self.assertIn("rax = lifted_callee(rcx, rdx, r8);", expected)
            self.assertIn("rax = lifted_callee2(rcx, rdx, r8);", expected)
            self.assertIn("r8[2] = rax;", expected)

            nested_out = base / "nested" / "sub" / "lifted.cpp"
            self.assertFalse(nested_out.parent.exists())

            real_replace = os.replace
            replaced: list[tuple[Path, Path]] = []

            def tracking_replace(src, dst):
                src_p, dst_p = Path(src), Path(dst)
                replaced.append((src_p, dst_p))
                self.assertEqual(src_p.parent.resolve(), dst_p.parent.resolve())
                self.assertEqual(src_p.suffix, ".tmp")
                return real_replace(src, dst)

            with mock.patch.object(cpu_call_aot.os, "replace", side_effect=tracking_replace):
                rc, out, err = _run_main(cpu_call_aot, [str(dll), "call_pair", str(nested_out)])

            self.assertIs(type(rc), int)
            self.assertEqual(rc, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(nested_out.read_text(encoding="utf-8"), expected)
            self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["lifted.cpp"])
            self.assertEqual(len(replaced), 1)
            self.assertEqual(replaced[0][1].resolve(), nested_out.resolve())

            # Atomic overwrite via subprocess and argv=None forwarding.
            nested_out.write_text("// stale\n", encoding="utf-8")
            proc = _run_subprocess(str(dll), "call_pair", str(nested_out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(proc.stderr, "")
            self.assertEqual(nested_out.read_text(encoding="utf-8"), expected)
            self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["lifted.cpp"])

            nested_out.write_text("// stale-2\n", encoding="utf-8")
            with mock.patch.object(sys, "argv", [str(TOOL), str(dll), "call_pair", str(nested_out)]):
                rc_none, out_none, err_none = _run_main(cpu_call_aot, None)
            self.assertEqual(rc_none, 0, err_none)
            self.assertEqual(out_none, "")
            self.assertEqual(err_none, "")
            self.assertEqual(nested_out.read_text(encoding="utf-8"), expected)

    def test_named_and_ordinal_iat_bindings_and_entrypoint(self) -> None:
        named_raw, _ = _make_iat_caller_dll(dll_name="prov.dll", imp_name="mix_args", imp_ord=None)
        ord_raw, _ = _make_iat_caller_dll(dll_name="ws2_32.dll", imp_name=None, imp_ord=77)
        prov_named = _make_provider_dll(export_name="mix_args", ordinal=1)
        prov_ord = _make_provider_dll(export_name=None, ordinal=77)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            caller_named = base / "caller_named.dll"
            caller_named.write_bytes(named_raw)
            prov_named_dll = base / "PROV.DLL"
            prov_named_dll.write_bytes(prov_named)
            out_named = base / "named.cpp"

            rc, out, err = _run_main(
                cpu_call_aot,
                [str(caller_named), "call_mix", str(out_named), "--bind", str(prov_named_dll), "mix_args"],
            )
            self.assertEqual(rc, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            cpp_named = out_named.read_text(encoding="utf-8")
            self.assertIn('extern "C" __declspec(dllimport) word mix_args(word, word, word*);', cpp_named)
            self.assertIn("rax = ::mix_args(rcx, rdx, r8);", cpp_named)

            # System ordinal (#77 on ws2_32.dll) with --link-name keeps stored ordinal instead of pefile table name.
            caller_ord = base / "caller_ord.dll"
            caller_ord.write_bytes(ord_raw)
            prov_ord_dll = base / "ws2_32.dll"
            prov_ord_dll.write_bytes(prov_ord)
            out_ord = base / "ord.cpp"

            proc = _run_subprocess(
                str(caller_ord), "call_mix", str(out_ord),
                "--bind", str(prov_ord_dll), "#77", "--link-name", "mix_args",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout, "")
            cpp_ord = out_ord.read_text(encoding="utf-8")
            self.assertIn('extern "C" __declspec(dllimport) word mix_args(word, word, word*);', cpp_ord)
            self.assertIn("rax = ::mix_args(rcx, rdx, r8);", cpp_ord)
            self.assertEqual([p.name for p in base.iterdir() if p.suffix == ".tmp"], [])

            # __main__ entrypoint raises SystemExit(0) and SystemExit(1) cleanly.
            for argv, expected_exit, err_frag in (
                ([str(TOOL), str(caller_named), "call_mix", str(out_named), "--bind", str(prov_named_dll), "mix_args"], 0, ""),
                ([str(TOOL), str(caller_named), "missing", str(out_named)], 1, "expected one local export"),
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


class CpuCallAotNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 1,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(cpu_call_aot, argv)
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

    def test_upfront_path_bind_and_link_name_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "caller.dll"
            image.write_bytes(b"synthetic-caller")
            provider = base / "prov.dll"
            provider.write_bytes(b"synthetic-provider")
            provider_cpp = base / "prov.cpp"
            provider_cpp.write_bytes(b"synthetic-provider-cpp")
            dir_path = base / "dir_path"
            dir_path.mkdir()
            same_cpp = base / "same.cpp"
            same_cpp.write_bytes(b"synthetic-cpp")
            hardlink_image = base / "hardlink_img.cpp"
            hardlink_image.hardlink_to(image)
            hardlink_prov = base / "hardlink_prov.cpp"
            hardlink_prov.hardlink_to(provider)
            sub_dir = base / "sub"
            sub_dir.mkdir()
            out_dir = base / "dir_out.cpp"
            out_dir.mkdir()
            file_parent = base / "file_parent.txt"
            file_parent.write_text("keep", encoding="utf-8")
            absent_out = base / "absent_dir" / "lifted.cpp"

            with (
                mock.patch.object(
                    cpu_call_aot,
                    "translate",
                    side_effect=AssertionError("translate called past upfront validation"),
                ),
                mock.patch.object(
                    cpu_call_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called past upfront validation"),
                ),
            ):
                for label, argv, frags in (
                    ("missing_image", [str(base / "missing.dll"), "call_mix", str(absent_out)], ("image file not found",)),
                    ("dir_image", [str(dir_path), "call_mix", str(absent_out)], ("image path is a directory",)),
                    ("empty_image", ["", "call_mix", str(absent_out)], ("image path is a directory",)),
                    ("empty_export", [str(image), "", str(absent_out)], ("export name must not be empty",)),
                    ("blank_export", [str(image), "   ", str(absent_out)], ("export name must not be empty",)),
                    ("non_cpp_suffix", [str(image), "call_mix", str(base / "out.txt")], (".cpp suffix",)),
                    ("empty_output", [str(image), "call_mix", ""], (".cpp suffix",)),
                    ("dot_cpp_only", [str(image), "call_mix", str(base / ".cpp")], (".cpp suffix",)),
                    ("output_is_dir", [str(image), "call_mix", str(out_dir)], ("output path is a directory",)),
                    ("parent_is_file", [str(image), "call_mix", str(file_parent / "lifted.cpp")], ("output parent is not a directory",)),
                    ("ancestor_is_file", [str(image), "call_mix", str(file_parent / "sub" / "lifted.cpp")], ("output parent is not a directory",)),
                    ("same_as_image", [str(same_cpp), "call_mix", str(same_cpp)], ("must not match input path",)),
                    ("resolve_as_image", [str(same_cpp), "call_mix", str(sub_dir / ".." / "same.cpp")], ("must not match input path",)),
                    ("hardlink_image", [str(image), "call_mix", str(hardlink_image)], ("must not match input path",)),
                    ("same_as_bind", [str(image), "call_mix", str(provider_cpp), "--bind", str(provider_cpp), "fn"], ("must not match input path",)),
                    ("resolve_as_bind", [str(image), "call_mix", str(sub_dir / ".." / "prov.cpp"), "--bind", str(provider_cpp), "fn"], ("must not match input path",)),
                    ("hardlink_bind", [str(image), "call_mix", str(hardlink_prov), "--bind", str(provider), "fn"], ("must not match input path",)),
                    ("missing_provider", [str(image), "call_mix", str(absent_out), "--bind", str(base / "missing_prov.dll"), "fn"], ("missing provider for missing_prov.dll!fn",)),
                    ("dir_provider", [str(image), "call_mix", str(absent_out), "--bind", str(dir_path), "fn"], ("missing provider for dir_path!fn",)),
                    ("empty_bind_provider", [str(image), "call_mix", str(absent_out), "--bind", "", "fn"], ("provider path and symbol must not be empty",)),
                    ("empty_bind_symbol", [str(image), "call_mix", str(absent_out), "--bind", str(provider), "   "], ("provider path and symbol must not be empty",)),
                    ("link_name_no_bind", [str(image), "call_mix", str(absent_out), "--link-name", "mix_args"], ("link name requires a provider binding",)),
                    ("empty_link_name", [str(image), "call_mix", str(absent_out), "--bind", str(provider), "#77", "--link-name", ""], ("link name must not be empty",)),
                    ("blank_link_name", [str(image), "call_mix", str(absent_out), "--bind", str(provider), "#77", "--link-name", "   "], ("link name must not be empty",)),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, expected_code=1, fragments=frags)

            self.assertFalse(absent_out.parent.exists())
            self.assertEqual(same_cpp.read_bytes(), b"synthetic-cpp")
            self.assertEqual(image.read_bytes(), b"synthetic-caller")
            self.assertEqual(provider.read_bytes(), b"synthetic-provider")
            self.assertEqual(provider_cpp.read_bytes(), b"synthetic-provider-cpp")

    def test_output_symlink_rejected_upfront(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "caller.dll"
            image.write_bytes(b"synthetic-caller")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            real_target = out_dir / "real.cpp"
            real_target.write_text("untouched", encoding="utf-8")

            with (
                mock.patch.object(Path, "is_symlink", return_value=True),
                mock.patch.object(
                    cpu_call_aot,
                    "translate",
                    side_effect=AssertionError("translate called for symlink output"),
                ),
                mock.patch.object(
                    cpu_call_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for symlink output"),
                ),
            ):
                self._assert_clean_failure(
                    [str(image), "call_mix", str(out_dir / "mock_link.cpp")],
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
                    cpu_call_aot,
                    "translate",
                    side_effect=AssertionError("translate called for symlink output"),
                ),
                mock.patch.object(
                    cpu_call_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for symlink output"),
                ),
            ):
                for label, link in (("live_symlink", live_link), ("dangling_symlink", dangling_link)):
                    with self.subTest(case=label):
                        self._assert_clean_failure(
                            [str(image), "call_mix", str(link)],
                            expected_code=1,
                            fragments=("output path must not be a symlink",),
                        )

            self.assertEqual(real_target.read_text(encoding="utf-8"), "untouched")

    def test_semantic_refusals_mixed_overlapping_non_pointer_r8_out_of_trace_and_non_exec(self) -> None:
        # 1) Mixed direct + import calls and two distinct IAT slots in lift_body.
        direct = b"\xe8\x00\x01\x00\x00"
        imp1 = b"\xff\x15\x00\x02\x00\x00"
        imp2 = b"\xff\x15\x08\x02\x00\x00"
        for body, needle in (
            (direct + imp1, "mixed import and direct calls"),
            (imp1 + direct, "mixed import and direct calls"),
            (imp1 + imp2, "call to a second IAT slot"),
        ):
            with self.subTest(needle=needle):
                with self.assertRaisesRegex(ValueError, needle):
                    cpu_call_aot.lift_body(0x1000, CALLER_PROLOGUE + body + CALLER_EPILOGUE,
                                           caller=True, bind_import=lambda _slot: "mix_args")

        # 2) Reloading R8 from a non-pointer GPR (RBX overwritten by mov rbx, rcx).
        bad_r8_code = CALLER_PROLOGUE + bytes.fromhex("48 89 cb 49 89 d8") + direct + CALLER_EPILOGUE
        with self.assertRaisesRegex(ValueError, "R8 reloaded without the output pointer"):
            cpu_call_aot.lift_body(0x1000, bad_r8_code, caller=True)

        # 3) Store outside trace ([r8+24]).
        out_of_trace = CALLER_PROLOGUE + direct + bytes.fromhex("49 89 40 18") + CALLER_EPILOGUE
        with self.assertRaisesRegex(ValueError, "store outside trace"):
            cpu_call_aot.lift_body(0x1000, out_of_trace, caller=True)

        # 4) PE-backed refusals via CLI & direct API.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            absent_out = base / "never_created" / "lifted.cpp"

            # Overlapping call target (call into caller itself).
            call_self = b"\xe8" + struct.pack("<i", -5 - len(CALLER_PROLOGUE))
            overlap_dll = base / "overlap.dll"
            overlap_dll.write_bytes(build_synthetic_pe(
                functions=[("call_pair", 0x00, CALLER_PROLOGUE + call_self + CALLER_EPILOGUE)],
            ))
            self._assert_clean_failure(
                [str(overlap_dll), "call_pair", str(absent_out)],
                fragments=("overlapping call target",),
            )

            # Overlapping runtime-function intervals in .pdata.
            pdata_overlap_dll = base / "pdata_overlap.dll"
            pdata_overlap_dll.write_bytes(build_synthetic_pe(
                functions=[
                    ("call_pair", 0x00, CALLER_PROLOGUE + b"\xe8\x75\x00\x00\x00" + CALLER_EPILOGUE),
                    ("mix_args", 0x80, CALLEE_PROLOGUE + b"\x48\x89\xc8" + CALLEE_EPILOGUE),
                ],
                overlapping_pdata=True,
            ))
            self._assert_clean_failure(
                [str(pdata_overlap_dll), "call_pair", str(absent_out)],
                fragments=("invalid runtime-function interval",),
            )

            # Non-executable .text section in caller.
            non_exec_caller = base / "non_exec_caller.dll"
            non_exec_caller.write_bytes(build_synthetic_pe(
                functions=[("call_pair", 0x00, CALLER_PROLOGUE + direct + CALLER_EPILOGUE)],
                text_executable=False,
            ))
            self._assert_clean_failure(
                [str(non_exec_caller), "call_pair", str(absent_out)],
                fragments=("invalid file-backed range",),
            )

            # Direct call supplied with --bind.
            direct_dll = base / "direct.dll"
            direct_dll.write_bytes(_make_direct_caller_and_callees())
            prov_dll = base / "prov.dll"
            prov_dll.write_bytes(_make_provider_dll())
            self._assert_clean_failure(
                [str(direct_dll), "call_pair", str(absent_out), "--bind", str(prov_dll), "mix_args"],
                fragments=("provider binding supplied for a direct call",),
            )

            # Non-executable provider export.
            non_exec_prov = base / "non_exec_prov.dll"
            non_exec_prov.write_bytes(_make_provider_dll(executable=False))
            iat_raw, slot_rva = _make_iat_caller_dll(dll_name="non_exec_prov.dll", imp_name="mix_args", imp_ord=None)
            iat_dll = base / "iat_non_exec.dll"
            iat_dll.write_bytes(iat_raw)
            self._assert_clean_failure(
                [str(iat_dll), "call_mix", str(absent_out), "--bind", str(non_exec_prov), "mix_args"],
                fragments=("non-executable export for non_exec_prov.dll!mix_args",),
            )

            # Direct resolve_import() & translate() guards for empty link_name and non-file provider.
            dir_prov = base / "dir_prov_sub" / "non_exec_prov.dll"
            dir_prov.mkdir(parents=True)
            with pefile.PE(str(iat_dll), fast_load=True) as pe:
                for bad_link in ("", "   "):
                    with self.assertRaisesRegex(ValueError, "link name must not be empty"):
                        cpu_call_aot.resolve_import(pe, slot_rva, non_exec_prov, "mix_args", link_name=bad_link)
                with self.assertRaisesRegex(ValueError, "missing provider for"):
                    cpu_call_aot.resolve_import(pe, slot_rva, dir_prov, "mix_args")

            with self.assertRaisesRegex(ValueError, "link name must not be empty"):
                cpu_call_aot.translate(iat_dll, "call_mix", binding=(str(prov_dll), "#77"), link_name="  ")
            with self.assertRaisesRegex(ValueError, "missing provider for"):
                cpu_call_aot.translate(iat_dll, "call_mix", binding=(str(base), "mix_args"))

            self.assertFalse(absent_out.parent.exists())
            self.assertEqual([p for p in base.iterdir() if p.suffix == ".tmp"], [])

    def test_atomic_write_failure_paths_and_argparse_errors(self) -> None:
        raw = _make_direct_caller_and_callees()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dll = base / "caller.dll"
            dll.write_bytes(raw)
            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_cpp = out_dir / "lifted.cpp"
            out_cpp.write_text("original", encoding="utf-8")

            with mock.patch.object(cpu_call_aot.os, "replace", side_effect=OSError("simulated replace failure")):
                self._assert_clean_failure(
                    [str(dll), "call_pair", str(out_cpp)],
                    fragments=("simulated replace failure",),
                    check_subprocess=False,
                )
            self.assertEqual(out_cpp.read_text(encoding="utf-8"), "original")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["lifted.cpp"])

            with mock.patch.object(cpu_call_aot, "translate", return_value="bad-surrogate-\ud800"):
                self._assert_clean_failure(
                    [str(dll), "call_pair", str(out_cpp)],
                    fragments=("surrogates not allowed",),
                    check_subprocess=False,
                )
            self.assertEqual(out_cpp.read_text(encoding="utf-8"), "original")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["lifted.cpp"])

            with mock.patch.object(cpu_call_aot.tempfile, "NamedTemporaryFile", side_effect=OSError("disk full")):
                self._assert_clean_failure(
                    [str(dll), "call_pair", str(out_cpp)],
                    fragments=("disk full",),
                    check_subprocess=False,
                )
            self.assertEqual(out_cpp.read_text(encoding="utf-8"), "original")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["lifted.cpp"])

            for argv, frags in (
                ([], ("required",)),
                ([str(dll)], ("required",)),
                ([str(dll), "call_pair"], ("required",)),
                ([str(dll), "call_pair", str(out_cpp), "--bogus-flag"], ("unrecognized",)),
            ):
                with self.subTest(argv=argv):
                    self._assert_clean_failure(argv, expected_code=None, fragments=frags)

    def test_fast_temp_mutations(self) -> None:
        source = TOOL.read_text(encoding="utf-8")
        raw = _make_direct_caller_and_callees()
        prov_raw = _make_provider_dll()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dll = base / "caller.dll"
            dll.write_bytes(raw)
            prov = base / "prov.dll"
            prov.write_bytes(prov_raw)
            dir_image = base / "dir_image"
            dir_image.mkdir()
            hardlink_prov = base / "hardlink_prov.cpp"
            hardlink_prov.hardlink_to(prov)
            file_parent = base / "file_parent.txt"
            file_parent.write_text("keep", encoding="utf-8")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_cpp = out_dir / "lifted.cpp"
            out_cpp.write_text("original", encoding="utf-8")

            iat_raw, slot_rva = _make_iat_caller_dll(dll_name="prov.dll", imp_name="mix_args", imp_ord=None)
            iat_dll = base / "iat.dll"
            iat_dll.write_bytes(iat_raw)

            def _check_drop_image_dir(mod) -> None:
                _, _, err = _run_main(mod, [str(dir_image), "call_pair", str(out_dir / "a.cpp")])
                self.assertNotIn("image path is a directory", err.lower())

            def _check_drop_image_file(mod) -> None:
                with mock.patch.object(mod, "translate", side_effect=RuntimeError("reached_translate")):
                    with self.assertRaisesRegex(RuntimeError, "reached_translate"):
                        _run_main(mod, [str(base / "missing.dll"), "call_pair", str(out_dir / "a.cpp")])

            def _check_drop_export(mod) -> None:
                with mock.patch.object(mod, "translate", side_effect=RuntimeError("reached_translate")):
                    with self.assertRaisesRegex(RuntimeError, "reached_translate"):
                        _run_main(mod, [str(dll), "   ", str(out_dir / "a.cpp")])

            def _check_drop_cpp_suffix(mod) -> None:
                code, _, err = _run_main(mod, [str(dll), "call_pair", str(base / "a.txt")])
                self.assertEqual(code, 0, err)

            def _check_drop_symlink(mod) -> None:
                with mock.patch.object(Path, "is_symlink", return_value=True):
                    _, _, err = _run_main(mod, [str(dll), "call_pair", str(out_dir / "sym.cpp")])
                self.assertNotIn("output path must not be a symlink", err.lower())

            def _check_drop_parent_file(mod) -> None:
                _, _, err = _run_main(mod, [str(dll), "call_pair", str(file_parent / "lifted.cpp")])
                self.assertNotIn("output parent is not a directory", err.lower())

            def _check_drop_samefile(mod) -> None:
                with mock.patch.object(mod, "translate", side_effect=RuntimeError("reached_translate")):
                    with self.assertRaisesRegex(RuntimeError, "reached_translate"):
                        _run_main(mod, [str(dll), "call_pair", str(hardlink_prov), "--bind", str(prov), "mix_args"])

            def _check_drop_empty_link_name(mod) -> None:
                with pefile.PE(str(iat_dll), fast_load=True) as pe:
                    with self.assertRaisesRegex(ValueError, "unexpected link name"):
                        mod.resolve_import(pe, slot_rva, prov, "mix_args", link_name="")

            def _check_drop_tmp_unlink(mod) -> None:
                with mock.patch.object(mod.os, "replace", side_effect=OSError("boom")):
                    code, _, _ = _run_main(mod, [str(dll), "call_pair", str(out_cpp)])
                self.assertEqual(code, 1)
                leaked = [p for p in out_dir.iterdir() if p.suffix == ".tmp"]
                self.assertTrue(leaked, "expected leaked .tmp file under mutation")
                for p in leaked:
                    p.unlink(missing_ok=True)

            def _check_drop_r8_ptr(mod) -> None:
                bad_r8_code = CALLER_PROLOGUE + bytes.fromhex("48 89 cb 49 89 d8 e8 00 01 00 00") + CALLER_EPILOGUE
                text_out, _ = mod.lift_body(0x1000, bad_r8_code, caller=True)
                self.assertIn("r8 = reinterpret_cast<word*>(rbx);", text_out)

            mutations = (
                (
                    "drop_image_is_dir_guard",
                    '    if image.is_dir():\n        raise IsADirectoryError(f"image path is a directory: {image}")\n',
                    "    pass\n",
                    _check_drop_image_dir,
                ),
                (
                    "drop_image_is_file_guard",
                    '    if not image.is_file():\n        raise FileNotFoundError(f"image file not found: {image}")\n',
                    "    pass\n",
                    _check_drop_image_file,
                ),
                (
                    "drop_export_non_empty_guard",
                    '    if not isinstance(export, str) or not export.strip():\n        raise ValueError("export name must not be empty")\n',
                    "    pass\n",
                    _check_drop_export,
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
                    _check_drop_symlink,
                ),
                (
                    "drop_parent_is_file_guard",
                    '        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):\n'
                    '            raise NotADirectoryError(f"output parent is not a directory: {parent}")\n',
                    "        pass\n",
                    _check_drop_parent_file,
                ),
                (
                    "drop_samefile_guard",
                    "        if out_resolved == input_path.resolve() or (output.exists() and input_path.samefile(output)):\n",
                    "        if out_resolved == input_path.resolve():\n",
                    _check_drop_samefile,
                ),
                (
                    "drop_empty_link_name_guard",
                    '    if link_name is not None and (not isinstance(link_name, str) or not link_name.strip()):\n        raise ValueError("link name must not be empty")\n',
                    "    pass\n",
                    _check_drop_empty_link_name,
                ),
                (
                    "drop_tmp_unlink",
                    "        if tmp_path is not None:\n            tmp_path.unlink(missing_ok=True)\n",
                    "        pass\n",
                    _check_drop_tmp_unlink,
                ),
                (
                    "drop_r8_pointer_guard",
                    '        if r8_move and write == "r8" and source not in pointer:\n            raise ValueError(f"R8 reloaded without the output pointer at RVA {rva:#x}")\n',
                    "        pass\n",
                    _check_drop_r8_ptr,
                ),
            )

            for label, target, replacement, check_fn in mutations:
                with self.subTest(mutation=label):
                    self.assertEqual(source.count(target), 1, target)
                    mod = types.ModuleType(f"cpu_call_aot_mut_{label}")
                    mod.__file__ = str(TOOL)
                    sys.modules[mod.__name__] = mod
                    try:
                        exec(compile(source.replace(target, replacement, 1), str(TOOL), "exec"), mod.__dict__)
                        check_fn(mod)
                    finally:
                        sys.modules.pop(mod.__name__, None)


if __name__ == "__main__":
    unittest.main()
