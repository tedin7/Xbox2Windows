"""Unit, CLI, path-guard, atomic-write, and mutation suite for tools/cpu_chained_aot.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "cpu_chained_aot.py"
sys.path.insert(0, str(ROOT / "tools"))

import cpu_chained_aot as tool  # noqa: E402


IMAGE_BASE = 0x140000000
FILE_ALIGN = 0x200
SECTION_ALIGN = 0x1000
TEXT_RVA = 0x1000
RDATA_RVA = 0x2000
PDATA_RVA = 0x3000
SECTION_SIZE = 0x1000


def _rel32(next_ip: int, target: int) -> bytes:
    return struct.pack("<i", target - next_ip)


def build_chained_pe(mode: str = "chained", *, export_name: str = "chained_fn", directories: int = 16,
                     callback_disp: int = 8) -> bytes:
    """Build a minimal synthetic AMD64 PE32+ DLL exercising chained unwind ranges."""
    text = bytearray(SECTION_SIZE)
    rdata = bytearray(SECTION_SIZE)
    rdata_cursor = 0

    def alloc_rdata(size: int, align: int = 4) -> int:
        nonlocal rdata_cursor
        rdata_cursor = (rdata_cursor + align - 1) & ~(align - 1)
        rva = RDATA_RVA + rdata_cursor
        rdata_cursor += size
        return rva

    def put_rdata(rva: int, blob: bytes) -> None:
        offset = rva - RDATA_RVA
        rdata[offset:offset + len(blob)] = blob

    import_rva = 0
    import_size = 0
    free_slot_rva = 0
    if mode == "ffx_destroy":
        dll_name = b"api-ms-win-crt-heap-l1-1-0.dll\0"
        dll_rva = alloc_rdata(len(dll_name), 2)
        put_rdata(dll_rva, dll_name)
        hint_rva = alloc_rdata(2 + len(b"free\0"), 2)
        put_rdata(hint_rva, b"\0\0free\0")
        table = struct.pack("<QQ", hint_rva, 0)
        ilt_rva = alloc_rdata(len(table), 8)
        put_rdata(ilt_rva, table)
        iat_rva = alloc_rdata(len(table), 8)
        put_rdata(iat_rva, table)
        free_slot_rva = iat_rva
        import_size = 40
        import_rva = alloc_rdata(import_size, 4)
        put_rdata(import_rva, struct.pack("<5I", ilt_rva, 0, 0, dll_rva, iat_rva))

    if mode == "chained":
        # Primary [0x1000..0x100d): test rcx, rcx; je 0x1020; mov rax, [rcx]; ret
        # Child   [0x1020..0x1023): xor eax, eax; ret
        primary = bytearray.fromhex("48 85 C9 0F 84 17 00 00 00 48 8B 01 C3")
        child = bytearray.fromhex("31 C0 C3")
        text[0:len(primary)] = primary
        text[0x20:0x20 + len(child)] = child
        prim_begin, prim_end = TEXT_RVA, TEXT_RVA + len(primary)
        child_begin, child_end = TEXT_RVA + 0x20, TEXT_RVA + 0x20 + len(child)
        unwind_prim = alloc_rdata(4, 4)
        put_rdata(unwind_prim, bytes([1, 0, 0, 0]))
        unwind_child = alloc_rdata(16, 4)
        put_rdata(unwind_child, bytes([(4 << 3) | 1, 0, 0, 0]) + struct.pack("<III", prim_begin, prim_end, unwind_prim))
        pdata = struct.pack("<IIIIII", prim_begin, prim_end, unwind_prim, child_begin, child_end, unwind_child)
    elif mode == "partial":
        # test rcx, rcx; je 0x1020; call qword ptr [rcx+8]; mov eax, 1; ret
        primary = bytearray.fromhex("48 85 C9 0F 84 19 00 00 00 FF 51 08 B8 01 00 00 00 C3")
        child = bytearray.fromhex("31 C0 C3")
        text[0:len(primary)] = primary
        text[0x22:0x22 + len(child)] = child
        prim_begin, prim_end = TEXT_RVA, TEXT_RVA + len(primary)
        child_begin, child_end = TEXT_RVA + 0x22, TEXT_RVA + 0x22 + len(child)
        unwind_prim = alloc_rdata(4, 4)
        put_rdata(unwind_prim, bytes([1, 0, 0, 0]))
        unwind_child = alloc_rdata(16, 4)
        put_rdata(unwind_child, bytes([(4 << 3) | 1, 0, 0, 0]) + struct.pack("<III", prim_begin, prim_end, unwind_prim))
        pdata = struct.pack("<IIIIII", prim_begin, prim_end, unwind_prim, child_begin, child_end, unwind_child)
    elif mode == "callback":
        # sub rsp, 28h; call qword ptr [rcx+disp]; add rsp, 28h; ret
        primary = bytearray.fromhex("48 83 EC 28 FF 51") + bytes([callback_disp]) + bytearray.fromhex("48 83 C4 28 C3")
        text[0:len(primary)] = primary
        prim_begin, prim_end = TEXT_RVA, TEXT_RVA + len(primary)
        unwind_prim = alloc_rdata(4, 4)
        put_rdata(unwind_prim, bytes([1, 0, 0, 0]))
        pdata = struct.pack("<III", prim_begin, prim_end, unwind_prim)
    elif mode == "ffx_destroy":
        # push rsi; push rdi; sub rsp, 28h; mov rsi, rdx; mov rdi, rcx; mov rax, [rcx]; call [rax+8];
        # mov [rsp+20h], rax; mov rcx, rdi; test rsi, rsi; je free_path; mov rdx, rsi; call [rsi+10h]; jmp done;
        # free_path: call [rip+free_slot]; done: mov rax, [rsp+20h]; add rsp, 28h; pop rdi; pop rsi; ret
        code = bytearray.fromhex(
            "56 57 48 83 EC 28 "
            "48 89 D6 48 89 CF 48 8B 01 FF 50 08 "
            "48 89 44 24 20 48 89 F9 48 85 F6 "
            "74 08 48 89 F2 FF 56 10 EB 06 "
        )
        call_ip = TEXT_RVA + len(code)
        code += b"\xFF\x15" + _rel32(call_ip + 6, free_slot_rva)
        code += bytearray.fromhex("48 8B 44 24 20 48 83 C4 28 5F 5E C3")
        text[0:len(code)] = code
        prim_begin, prim_end = TEXT_RVA, TEXT_RVA + len(code)
        unwind_prim = alloc_rdata(4, 4)
        put_rdata(unwind_prim, bytes([1, 0, 0, 0]))
        pdata = struct.pack("<III", prim_begin, prim_end, unwind_prim)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    exp_dll_rva = alloc_rdata(len(b"chained_test.dll\0"), 2)
    put_rdata(exp_dll_rva, b"chained_test.dll\0")
    exp_name_bytes = export_name.encode("ascii") + b"\0"
    exp_name_rva = alloc_rdata(len(exp_name_bytes), 2)
    put_rdata(exp_name_rva, exp_name_bytes)
    eat_rva = alloc_rdata(4, 4)
    put_rdata(eat_rva, struct.pack("<I", TEXT_RVA))
    ent_rva = alloc_rdata(4, 4)
    put_rdata(ent_rva, struct.pack("<I", exp_name_rva))
    eot_rva = alloc_rdata(2, 2)
    put_rdata(eot_rva, struct.pack("<H", 0))
    export_size = 40
    export_rva = alloc_rdata(export_size, 4)
    put_rdata(
        export_rva,
        struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, exp_dll_rva, 1, 1, 1, eat_rva, ent_rva, eot_rva),
    )

    pdata_sec = bytearray(SECTION_SIZE)
    pdata_sec[0:len(pdata)] = pdata

    buf = bytearray(FILE_ALIGN + SECTION_SIZE * 3)
    buf[0:2] = b"MZ"
    pe_off = 0x80
    struct.pack_into("<I", buf, 60, pe_off)
    buf[pe_off:pe_off + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_off + 4, 0x8664, 3, 0, 0, 0, 0xF0, 0x2022)
    opt = pe_off + 24
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        buf, opt,
        0x20B, 14, 0, SECTION_SIZE, SECTION_SIZE * 2, 0, TEXT_RVA, TEXT_RVA, IMAGE_BASE,
        SECTION_ALIGN, FILE_ALIGN, 6, 0, 0, 0, 6, 0, 0, 0x4000, FILE_ALIGN, 0, 3, 0x8160,
        0x100000, 0x1000, 0x100000, 0x1000, 0, directories,
    )
    dd = opt + 112
    if directories > 0:
        struct.pack_into("<II", buf, dd + 0 * 8, export_rva, export_size)
    if directories > 1 and import_size:
        struct.pack_into("<II", buf, dd + 1 * 8, import_rva, import_size)
    if directories > 3:
        struct.pack_into("<II", buf, dd + 3 * 8, PDATA_RVA, len(pdata))

    sections_off = opt + 0xF0
    for idx, (sec_name, va, roff, chars) in enumerate((
        (b".text\0\0\0", TEXT_RVA, FILE_ALIGN, 0x60000020),
        (b".rdata\0\0", RDATA_RVA, FILE_ALIGN + SECTION_SIZE, 0x40000040),
        (b".pdata\0\0", PDATA_RVA, FILE_ALIGN + SECTION_SIZE * 2, 0x40000040),
    )):
        struct.pack_into(
            "<8sIIIIIIHHI", buf, sections_off + idx * 40,
            sec_name, SECTION_SIZE, va, SECTION_SIZE, roff, 0, 0, 0, 0, chars,
        )

    buf[FILE_ALIGN:FILE_ALIGN + SECTION_SIZE] = text
    buf[FILE_ALIGN + SECTION_SIZE:FILE_ALIGN + SECTION_SIZE * 2] = rdata
    buf[FILE_ALIGN + SECTION_SIZE * 2:FILE_ALIGN + SECTION_SIZE * 3] = pdata_sec
    return bytes(buf)


def _run_main(argv):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = tool.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args):
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
    )


class ChainedAotCliPositiveTests(unittest.TestCase):
    def test_default_partial_callback_and_ffx_destroy(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            img = root / "chained.dll"
            img.write_bytes(build_chained_pe("chained"))
            out = root / "nested" / "dir" / "out.cpp"

            code, stdout, stderr = _run_main([str(img), "chained_fn", str(out)])
            self.assertEqual(code, 0, stderr)
            self.assertEqual(stderr, "")
            self.assertIn("2 owned ranges, 2 return paths, indirect stops: []", stdout)
            cpp = out.read_text(encoding="utf-8")
            self.assertIn("extern \"C\" __declspec(dllexport) word try_lifted(", cpp)
            self.assertIn("load_always<word>((first + 0ull))", cpp)

            partial_img = root / "partial.dll"
            partial_img.write_bytes(build_chained_pe("partial"))
            partial_out = root / "partial.cpp"
            proc = _run_subprocess(partial_img, "chained_fn", partial_out, "--partial")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("2 owned ranges, 1 return paths, indirect stops: [4105]", proc.stdout)
            self.assertIn("return 0;", partial_out.read_text(encoding="utf-8"))

            cb_img = root / "callback.dll"
            cb_img.write_bytes(build_chained_pe("callback", callback_disp=16))
            cb_out = root / "callback.cpp"
            code, stdout, stderr = _run_main([str(cb_img), "chained_fn", str(cb_out), "--callback-offset", "0x10"])
            self.assertEqual(code, 0, stderr)
            self.assertIn("try_lifted_callback(", cb_out.read_text(encoding="utf-8"))
            self.assertIn("read_word(binding->table + 16ull)", cb_out.read_text(encoding="utf-8"))

            ffx_img = root / "ffx.dll"
            ffx_img.write_bytes(build_chained_pe("ffx_destroy"))
            ffx_out = root / "ffx.cpp"
            code, stdout, stderr = _run_main([str(ffx_img), "chained_fn", str(ffx_out), "--ffx-destroy"])
            self.assertEqual(code, 0, stderr)
            ffx_cpp = ffx_out.read_text(encoding="utf-8")
            self.assertIn("try_destroy(", ffx_cpp)
            self.assertIn("known->dealloc(", ffx_cpp)
            self.assertIn("known->release(", ffx_cpp)


class ChainedAotCliRefusalAndAtomicTests(unittest.TestCase):
    def _assert_clean_exit(self, argv, expected_code, fragment="", *, check_subprocess=True):
        code, stdout, stderr = _run_main(argv)
        self.assertEqual(code, expected_code, f"stdout={stdout!r} stderr={stderr!r}")
        self.assertEqual(stdout, "")
        self.assertNotIn("Traceback", stderr)
        if fragment:
            self.assertIn(fragment, stderr)
        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            if fragment:
                self.assertIn(fragment, proc.stderr)

    def test_translate_callback_offset_and_contract_guards(self):
        with TemporaryDirectory() as tmp:
            img = Path(tmp) / "chained.dll"
            img.write_bytes(build_chained_pe("chained"))
            for bad_offset in (False, True, -8, 4, 0x80000000, "8"):
                with self.subTest(bad_offset=bad_offset):
                    with self.assertRaisesRegex(ValueError, "callback slot"):
                        tool.translate(img, "chained_fn", callback_offset=bad_offset)
            with self.assertRaisesRegex(ValueError, "destroy contract"):
                tool.translate(img, "chained_fn", partial=True, ffx_destroy=True)
            with self.assertRaisesRegex(ValueError, "destroy contract"):
                tool.translate(img, "chained_fn", callback_offset=8, ffx_destroy=True)
            for bad_export in ("", "   ", None):
                with self.subTest(bad_export=bad_export):
                    with self.assertRaisesRegex(ValueError, "export name"):
                        tool.translate(img, bad_export)

    def test_hostile_cli_paths_and_refusals(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            img = root / "chained.dll"
            img.write_bytes(build_chained_pe("chained"))
            img_dir = root / "img_dir"
            img_dir.mkdir()
            out_cpp = root / "out.cpp"

            self._assert_clean_exit([str(root / "missing.dll"), str(out_cpp), str(out_cpp)], 1, "not an existing file")
            self._assert_clean_exit([str(img_dir), "chained_fn", str(out_cpp)], 1, "not an existing file")
            self._assert_clean_exit([str(img), "", str(out_cpp)], 1, "export name")
            self._assert_clean_exit([str(img), "   ", str(out_cpp)], 1, "export name")
            self._assert_clean_exit([str(img), "chained_fn", str(root / "bad.txt")], 1, ".cpp suffix")
            self._assert_clean_exit([str(img), "chained_fn", str(root / "noext")], 1, ".cpp suffix")

            out_dir = root / "out_dir.cpp"
            out_dir.mkdir()
            self._assert_clean_exit([str(img), "chained_fn", str(out_dir)], 1, "is a directory")

            parent_file = root / "parent_file"
            parent_file.write_bytes(b"keep-me")
            self._assert_clean_exit([str(img), "chained_fn", str(parent_file / "child.cpp")], 1, "parent is not a directory")
            self.assertEqual(parent_file.read_bytes(), b"keep-me")

            # Samefile / resolve collision directly and via hardlink
            code, stdout, stderr = _run_main([str(img), "chained_fn", str(img.with_suffix(".cpp"))])
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn("2 owned ranges", stdout)
            cpp_input = root / "input_alias.cpp"
            cpp_input.write_bytes(build_chained_pe("chained"))
            self._assert_clean_exit([str(cpp_input), "chained_fn", str(cpp_input)], 1, "collide with input image")

            hardlink_out = root / "hardlink.cpp"
            hardlink_out.hardlink_to(img)
            self._assert_clean_exit([str(img), "chained_fn", str(hardlink_out)], 1, "collide with input image")

            # Symlink output guard (real symlink if permitted + mock fallback)
            real_target = root / "real_target.cpp"
            real_target.write_text("do-not-overwrite\n", encoding="utf-8")
            sym_out = root / "sym.cpp"
            try:
                sym_out.symlink_to(real_target)
                created_sym = True
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                created_sym = False
            if created_sym:
                self._assert_clean_exit([str(img), "chained_fn", str(sym_out)], 1, "symbolic link")
            else:
                with mock.patch.object(Path, "is_symlink", autospec=True,
                                       side_effect=lambda p: p.resolve() == sym_out.resolve()):
                    self._assert_clean_exit(
                        [str(img), "chained_fn", str(sym_out)],
                        1,
                        "symbolic link",
                        check_subprocess=False,
                    )
            self.assertEqual(real_target.read_text(encoding="utf-8"), "do-not-overwrite\n")

            # Bad CLI callback-offset values
            self._assert_clean_exit([str(img), "chained_fn", str(out_cpp), "--callback-offset", "-8"], 1, "callback slot")
            self._assert_clean_exit([str(img), "chained_fn", str(out_cpp), "--callback-offset", "4"], 1, "callback slot")
            self._assert_clean_exit([str(img), "chained_fn", str(out_cpp), "--callback-offset", "0x80000000"], 1, "callback slot")
            self._assert_clean_exit([str(img), "chained_fn", str(out_cpp), "--callback-offset", "nope"], 2, "invalid")

            # Short directory PE refusal
            short_img = root / "short.dll"
            short_img.write_bytes(build_chained_pe("chained", directories=3))
            self._assert_clean_exit([str(short_img), "chained_fn", str(out_cpp)], 1, "runtime-function")

    def test_atomic_write_failures_preserve_prior_and_unlink_temps(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            img = root / "chained.dll"
            img.write_bytes(build_chained_pe("chained"))
            out_dir = root / "atomic"
            out_dir.mkdir()
            out_cpp = out_dir / "lifted.cpp"
            out_cpp.write_text("prior-contents\n", encoding="utf-8")

            with mock.patch.object(tool.os, "replace", side_effect=OSError("simulated replace fail")):
                self._assert_clean_exit(
                    [str(img), "chained_fn", str(out_cpp)],
                    1,
                    "simulated replace fail",
                    check_subprocess=False,
                )
            self.assertEqual(out_cpp.read_text(encoding="utf-8"), "prior-contents\n")
            self.assertEqual(list(out_dir.iterdir()), [out_cpp])

            with mock.patch.object(tool.os, "fdopen", side_effect=OSError("simulated fdopen fail")):
                self._assert_clean_exit(
                    [str(img), "chained_fn", str(out_cpp)],
                    1,
                    "simulated fdopen fail",
                    check_subprocess=False,
                )
            self.assertEqual(out_cpp.read_text(encoding="utf-8"), "prior-contents\n")
            self.assertEqual(list(out_dir.iterdir()), [out_cpp])


class ChainedAotCliMutationTests(unittest.TestCase):
    def test_fast_self_mutations_are_killed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            img = root / "chained.dll"
            img.write_bytes(build_chained_pe("chained"))
            orig_translate = tool.translate
            orig_validate = tool._validate_cli_inputs
            mutants = []

            def run_case(test_cls, method_name):
                res = unittest.TestResult()
                test_cls(method_name).run(res)
                return res.wasSuccessful()

            # Mutant A: allow bool callback_offset
            def mut_a(image, name, partial=False, callback_offset=None, ffx_destroy=False):
                if isinstance(callback_offset, bool):
                    callback_offset = int(callback_offset)
                return orig_translate(image, name, partial, callback_offset, ffx_destroy)

            with mock.patch.object(tool, "translate", side_effect=mut_a):
                mutants.append(("allow_bool_callback_offset",
                                run_case(ChainedAotCliRefusalAndAtomicTests,
                                         "test_translate_callback_offset_and_contract_guards")))

            # Mutant B: skip .cpp suffix check
            def mut_b(image, export, output):
                if output.suffix.lower() != ".cpp":
                    return orig_validate(image, export, output.with_suffix(".cpp"))
                return orig_validate(image, export, output)

            with mock.patch.object(tool, "_validate_cli_inputs", side_effect=mut_b):
                mutants.append(("skip_cpp_suffix_guard",
                                run_case(ChainedAotCliRefusalAndAtomicTests, "test_hostile_cli_paths_and_refusals")))

            # Mutant C: leave temp file behind on os.replace failure
            def mut_c(path, text):
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tool.tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
                with tool.os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(text)
                tool.os.replace(Path(tmp_name), path)

            with mock.patch.object(tool, "_write_atomic", side_effect=mut_c):
                mutants.append(("leak_temp_on_replace_fail",
                                run_case(ChainedAotCliRefusalAndAtomicTests,
                                         "test_atomic_write_failures_preserve_prior_and_unlink_temps")))

            # Mutant D: skip samefile/collision guard
            def mut_d(image, export, output):
                if output.suffix.lower() == ".cpp" and output.exists() and output.samefile(image):
                    return
                return orig_validate(image, export, output)

            with mock.patch.object(tool, "_validate_cli_inputs", side_effect=mut_d):
                mutants.append(("skip_collision_guard",
                                run_case(ChainedAotCliRefusalAndAtomicTests, "test_hostile_cli_paths_and_refusals")))

            # Mutant E: skip symlink guard
            def mut_e(image, export, output):
                if output.is_symlink():
                    return
                return orig_validate(image, export, output)

            with mock.patch.object(tool, "_validate_cli_inputs", side_effect=mut_e):
                mutants.append(("skip_symlink_guard",
                                run_case(ChainedAotCliRefusalAndAtomicTests, "test_hostile_cli_paths_and_refusals")))

            for name, passed in mutants:
                self.assertFalse(passed, f"mutant {name} unexpectedly survived")


if __name__ == "__main__":
    unittest.main()
