"""Synthetic AMD64 PE32+ unit and refusal suite for tools/cpu_stdio_exe_aot.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "cpu_stdio_exe_aot.py"
sys.path.insert(0, str(ROOT / "tools"))
import cpu_stdio_exe_aot  # noqa: E402
from cpu_switch_aot import HostImport  # noqa: E402


IMAGE_BASE = 0x140000000
RAW_OFFSET = 0x200
SECTION_ALIGN = 0x1000
TEXT_RVA = 0x1000
RDATA_RVA = 0x2000
SECTION_SIZE = 0x1000


def _rel32(next_ip: int, target: int) -> bytes:
    return struct.pack("<i", target - next_ip)


def build_console_exe(
    *,
    machine: int = 0x8664,
    magic: int = 0x20B,
    subsystem: int = 3,
    characteristics: int = 0x0022,
    tls_rva: int = 0,
    delay_import_rva: int = 0,
    include_finish_exe: bool = False,
    extra_imports: list[tuple[bytes, list[bytes | int]]] | None = None,
    duplicate_import_slot: bool = False,
    entry_mode: str = "stdio",
    export_native_entry: bool = True,
    register_entry_pdata: bool = True,
    entry_rva_override: int | None = None,
):
    """Build a minimal synthetic AMD64 PE32+ console EXE for cpu_stdio_exe_aot."""
    text = bytearray(SECTION_SIZE)
    rdata = bytearray(SECTION_SIZE)
    text_cursor = 0
    rdata_cursor = 0

    def emit_text(blob: bytes, align: int = 16) -> int:
        nonlocal text_cursor
        text_cursor = (text_cursor + align - 1) & ~(align - 1)
        rva = TEXT_RVA + text_cursor
        text[text_cursor:text_cursor + len(blob)] = blob
        text_cursor += len(blob)
        return rva

    def alloc_rdata(size: int, align: int = 8) -> int:
        nonlocal rdata_cursor
        rdata_cursor = (rdata_cursor + align - 1) & ~(align - 1)
        rva = RDATA_RVA + rdata_cursor
        rdata_cursor += size
        return rva

    def put_rdata(rva: int, blob: bytes) -> None:
        offset = rva - RDATA_RVA
        rdata[offset:offset + len(blob)] = blob

    # Assemble import descriptors in .rdata.
    stdio_symbols: list[bytes | int] = [b"getchar", b"putchar", b"fflush"]
    modules: list[tuple[bytes, list[bytes | int]]] = [
        (cpu_stdio_exe_aot.STDIO.encode("ascii"), stdio_symbols),
    ]
    if include_finish_exe:
        modules.append((b"cpu_exe_io.dll", [b"finish_exe"]))
    if extra_imports:
        modules.extend(extra_imports)

    # Reserve import descriptors (one per module + optional duplicate + null terminator).
    desc_count = len(modules) + (1 if duplicate_import_slot else 0) + 1
    import_size = 20 * desc_count
    import_rva = alloc_rdata(import_size, 4)

    iat_slots: dict[tuple[bytes, bytes | int], int] = {}
    first_iat_rva = None
    first_ilt_rva = None
    first_dll_rva = None

    for idx, (dll_name, symbols) in enumerate(modules):
        dll_rva = alloc_rdata(len(dll_name) + 1, 2)
        put_rdata(dll_rva, dll_name + b"\0")
        entries: list[int] = []
        for sym in symbols:
            if isinstance(sym, int):
                entries.append(0x8000000000000000 | sym)
            else:
                hint_rva = alloc_rdata(2 + len(sym) + 1, 2)
                put_rdata(hint_rva, b"\0\0" + sym + b"\0")
                entries.append(hint_rva)
        table_bytes = struct.pack(f"<{len(entries) + 1}Q", *entries, 0)
        ilt_rva = alloc_rdata(len(table_bytes), 8)
        put_rdata(ilt_rva, table_bytes)
        iat_rva = alloc_rdata(len(table_bytes), 8)
        put_rdata(iat_rva, table_bytes)

        if idx == 0:
            first_iat_rva = iat_rva
            first_ilt_rva = ilt_rva
            first_dll_rva = dll_rva

        for sym_idx, sym in enumerate(symbols):
            iat_slots[(dll_name.lower(), sym)] = iat_rva + sym_idx * 8

        put_rdata(import_rva + idx * 20, struct.pack("<5I", ilt_rva, 0, 0, dll_rva, iat_rva))

    if duplicate_import_slot:
        dup_off = len(modules) * 20
        put_rdata(import_rva + dup_off, struct.pack("<5I", first_ilt_rva, 0, 0, first_dll_rva, first_iat_rva))

    getchar_slot = iat_slots[(cpu_stdio_exe_aot.STDIO.encode("ascii"), b"getchar")]
    putchar_slot = iat_slots[(cpu_stdio_exe_aot.STDIO.encode("ascii"), b"putchar")]
    fflush_slot = iat_slots[(cpu_stdio_exe_aot.STDIO.encode("ascii"), b"fflush")]

    # Emit entry function body in .text.
    code = bytearray()
    entry_rva = TEXT_RVA + ((text_cursor + 15) & ~15)
    if entry_mode == "stdio":
        # sub rsp, 0x28
        code += b"\x48\x83\xEC\x28"
        # call qword ptr [rip + getchar_slot]
        code += b"\xFF\x15" + _rel32(entry_rva + len(code) + 6, getchar_slot)
        # mov ecx, eax
        code += b"\x89\xC1"
        # call qword ptr [rip + putchar_slot]
        code += b"\xFF\x15" + _rel32(entry_rva + len(code) + 6, putchar_slot)
        # xor ecx, ecx
        code += b"\x31\xC9"
        # call qword ptr [rip + fflush_slot]
        code += b"\xFF\x15" + _rel32(entry_rva + len(code) + 6, fflush_slot)
        # add rsp, 0x28; ret
        code += b"\x48\x83\xC4\x28\xC3"
    elif entry_mode == "call_finish_exe":
        finish_slot = iat_slots[(b"cpu_exe_io.dll", b"finish_exe")]
        code += b"\x48\x83\xEC\x28"
        code += b"\x31\xC9"
        code += b"\xFF\x15" + _rel32(entry_rva + len(code) + 6, finish_slot)
        code += b"\x31\xC0"
        code += b"\x48\x83\xC4\x28\xC3"
    elif entry_mode == "noreturn_exit":
        exit_slot = iat_slots[(b"api-ms-win-crt-runtime-l1-1-0.dll", b"exit")]
        code += b"\x48\x83\xEC\x28"
        code += b"\xB9\x07\x00\x00\x00"  # mov ecx, 7
        code += b"\xFF\x15" + _rel32(entry_rva + len(code) + 6, exit_slot)
        code += b"\xCC"
    elif entry_mode == "void_ret":
        code += b"\xC3"
    elif entry_mode == "narrow_ret":
        code += b"\xB0\x01\xC3"  # mov al, 1; ret
    elif entry_mode == "loop_forever":
        code += b"\xEB\xFE"  # jmp $
    else:
        raise ValueError(f"unknown entry_mode {entry_mode!r}")

    actual_entry = emit_text(bytes(code))
    assert actual_entry == entry_rva
    entry_end = entry_rva + len(code)

    # Emit a minimal native_startup wrapper that sits outside native_entry and calls finish_exe.
    startup_rva = TEXT_RVA + ((text_cursor + 15) & ~15)
    startup = bytearray()
    startup += b"\x48\x83\xEC\x28"
    startup += b"\xE8" + _rel32(startup_rva + len(startup) + 5, entry_rva)
    if include_finish_exe:
        finish_slot = iat_slots[(b"cpu_exe_io.dll", b"finish_exe")]
        startup += b"\x89\xC1"
        startup += b"\xFF\x15" + _rel32(startup_rva + len(startup) + 6, finish_slot)
    startup += b"\x48\x83\xC4\x28\xC3"
    actual_startup = emit_text(bytes(startup))
    assert actual_startup == startup_rva
    startup_end = startup_rva + len(startup)

    # Optional export table publishing native_entry.
    export_rva = 0
    export_size = 0
    if export_native_entry:
        exp_dll_rva = alloc_rdata(len(b"synthetic_console.exe\0"), 2)
        put_rdata(exp_dll_rva, b"synthetic_console.exe\0")
        exp_name_rva = alloc_rdata(len(b"native_entry\0"), 2)
        put_rdata(exp_name_rva, b"native_entry\0")
        eat_rva = alloc_rdata(4, 4)
        put_rdata(eat_rva, struct.pack("<I", entry_rva))
        ent_rva = alloc_rdata(4, 4)
        put_rdata(ent_rva, struct.pack("<I", exp_name_rva))
        eot_rva = alloc_rdata(2, 2)
        put_rdata(eot_rva, struct.pack("<H", 0))
        export_size = 40
        export_rva = alloc_rdata(export_size, 4)
        put_rdata(
            export_rva,
            struct.pack(
                "<IIHHIIIIIII",
                0, 0, 0, 0, exp_dll_rva, 1, 1, 1, eat_rva, ent_rva, eot_rva,
            ),
        )

    # .pdata and UNWIND_INFO in .rdata.
    pdata_rva = 0
    pdata_size = 0
    if register_entry_pdata:
        unwind_rva = alloc_rdata(4, 4)
        put_rdata(unwind_rva, b"\x01\x00\x00\x00")
        records = [(entry_rva, entry_end), (startup_rva, startup_end)]
        records.sort()
        pdata_size = 12 * len(records)
        pdata_rva = alloc_rdata(pdata_size, 4)
        for idx, (begin, end) in enumerate(records):
            put_rdata(pdata_rva + idx * 12, struct.pack("<III", begin, end, unwind_rva))

    address_of_entry = entry_rva_override if entry_rva_override is not None else entry_rva

    # Assemble PE32+ headers + 2 sections (.text, .rdata).
    buf = bytearray(RAW_OFFSET + SECTION_SIZE * 2)
    buf[0:2] = b"MZ"
    pe_off = 0x80
    struct.pack_into("<I", buf, 60, pe_off)
    buf[pe_off:pe_off + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_off + 4, machine, 2, 0, 0, 0, 0xF0, characteristics)
    opt = pe_off + 24
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        buf,
        opt,
        magic,
        14,
        0,
        SECTION_SIZE,
        SECTION_SIZE,
        0,
        address_of_entry,
        TEXT_RVA,
        IMAGE_BASE,
        SECTION_ALIGN,
        0x200,
        6,
        0,
        0,
        0,
        6,
        0,
        0,
        0x3000,
        RAW_OFFSET,
        0,
        subsystem,
        0x160,
        0x100000,
        0x1000,
        0x100000,
        0x1000,
        0,
        16,
    )
    dd = opt + 112
    if export_size:
        struct.pack_into("<II", buf, dd + 0 * 8, export_rva, export_size)
    struct.pack_into("<II", buf, dd + 1 * 8, import_rva, import_size)
    if pdata_size:
        struct.pack_into("<II", buf, dd + 3 * 8, pdata_rva, pdata_size)
    if tls_rva:
        struct.pack_into("<II", buf, dd + 9 * 8, tls_rva, 0x28)
    if delay_import_rva:
        struct.pack_into("<II", buf, dd + 13 * 8, delay_import_rva, 0x20)

    sections_off = opt + 0xF0
    sections = [
        (b".text\0\0\0", SECTION_SIZE, TEXT_RVA, SECTION_SIZE, RAW_OFFSET, 0x60000020),
        (b".rdata\0\0", SECTION_SIZE, RDATA_RVA, SECTION_SIZE, RAW_OFFSET + SECTION_SIZE, 0x40000040),
    ]
    for idx, (name, vsize, va, rsize, roff, chars) in enumerate(sections):
        struct.pack_into(
            "<8sIIIIIIHHI",
            buf,
            sections_off + idx * 40,
            name,
            vsize,
            va,
            rsize,
            roff,
            0,
            0,
            0,
            0,
            chars,
        )

    buf[RAW_OFFSET:RAW_OFFSET + SECTION_SIZE] = text
    buf[RAW_OFFSET + SECTION_SIZE:RAW_OFFSET + SECTION_SIZE * 2] = rdata

    return bytes(buf), {"entry_rva": entry_rva, "startup_rva": startup_rva}


class CpuStdioExeAotPositiveTests(unittest.TestCase):
    def test_translate_direct_and_entry_export_provenance(self):
        raw, _ = build_console_exe(include_finish_exe=True)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "stdio_console.exe"
            image.write_bytes(raw)

            cpp_export = cpu_stdio_exe_aot.translate(image, entry_export="native_entry")
            self.assertIn("static word host_getchar()", cpp_export)
            self.assertIn("static word host_putchar(word ch)", cpp_export)
            self.assertIn("static word host_fflush(word stream)", cpp_export)
            self.assertIn("host_getchar()", cpp_export)
            self.assertIn("host_putchar(", cpp_export)
            self.assertIn("host_fflush(", cpp_export)
            self.assertIn(
                f'extern "C" __declspec(dllexport) const char aot_source_sha256[] = "{digest}";',
                cpp_export,
            )
            self.assertIn("int main() { return static_cast<int>(process_entry()); }", cpp_export)

            # Without finish_exe in the import table, AddressOfEntryPoint translates directly without provenance.
            raw_plain, _ = build_console_exe(include_finish_exe=False)
            plain_image = Path(tmp) / "plain_console.exe"
            plain_image.write_bytes(raw_plain)
            cpp_direct = cpu_stdio_exe_aot.translate(plain_image)
            self.assertNotIn("aot_source_sha256", cpp_direct)
            self.assertIn("int main() { return static_cast<int>(process_entry()); }", cpp_direct)

    def test_allow_process_exit_for_noreturn_import(self):
        runtime_dll = "api-ms-win-crt-runtime-l1-1-0.dll"
        raw, _ = build_console_exe(
            extra_imports=[(runtime_dll.encode("ascii"), [b"exit"])],
            entry_mode="noreturn_exit",
        )
        bindings = {
            **cpu_stdio_exe_aot.IMPORTS,
            (runtime_dll, b"exit"): HostImport("host_exit", (4,), 0, noreturn=True),
        }
        host = cpu_stdio_exe_aot.HOST + "\n[[noreturn]] static void host_exit(word code) { std::exit(static_cast<int>(code)); }\n"
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "exit_console.exe"
            image.write_bytes(raw)
            with self.assertRaises(ValueError) as ctx:
                cpu_stdio_exe_aot.translate(image, bindings=bindings, host=host, allow_process_exit=False)
            self.assertIn("32-bit exit code", str(ctx.exception))

            cpp = cpu_stdio_exe_aot.translate(image, bindings=bindings, host=host, allow_process_exit=True)
            self.assertIn("host_exit(", cpp)
            self.assertIn("int main() { return static_cast<int>(process_entry()); }", cpp)

    def test_main_argv_writes_cpp_and_provenance(self):
        raw, _ = build_console_exe(include_finish_exe=True)
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "cli_console.exe"
            out_cpp = Path(tmp) / "cli_lifted.cpp"
            image.write_bytes(raw)

            rc = cpu_stdio_exe_aot.main([str(image), str(out_cpp), "--entry-export", "native_entry"])
            self.assertEqual(rc, 0)
            cpp = out_cpp.read_text(encoding="ascii")
            self.assertIn("aot_source_sha256", cpp)
            self.assertIn(digest, cpp)
            self.assertIn("int main() { return static_cast<int>(process_entry()); }", cpp)


class CpuStdioExeAotRefusalTests(unittest.TestCase):
    def _assert_translate_fails(self, raw: bytes, expected_msg: str, **translate_kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "bad.exe"
            image.write_bytes(raw)
            with self.assertRaises(ValueError) as ctx:
                cpu_stdio_exe_aot.translate(image, **translate_kwargs)
            self.assertIn(expected_msg, str(ctx.exception))

    def _assert_cli_exit_1(self, image_path: Path, *extra_args: str, expected_msg: str = ""):
        with tempfile.TemporaryDirectory() as tmp:
            out_cpp = Path(tmp) / "out.cpp"
            stderr = io.StringIO()
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exc:
                    cpu_stdio_exe_aot.main([str(image_path), str(out_cpp), *extra_args])
            self.assertEqual(exc.exception.code, 1)
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertTrue(stderr.getvalue().strip())
            if expected_msg:
                self.assertIn(expected_msg, stderr.getvalue())
            self.assertFalse(out_cpp.exists())

    def test_header_and_characteristic_refusals(self):
        cases = (
            ("non-AMD64 machine", dict(machine=0x014C)),
            ("non-PE32+ magic", dict(magic=0x10B)),
            ("non-console subsystem", dict(subsystem=2)),
            ("DLL characteristic", dict(characteristics=0x2022)),
            ("non-executable image", dict(characteristics=0x0000)),
        )
        for label, kwargs in cases:
            with self.subTest(label=label):
                raw, _ = build_console_exe(**kwargs)
                self._assert_translate_fails(raw, "expected an AMD64 console EXE")

    def test_tls_and_delay_import_directories_refused(self):
        raw_tls, _ = build_console_exe(tls_rva=RDATA_RVA + 0x400)
        self._assert_translate_fails(raw_tls, "unsupported TLS startup in console EXE")

        raw_delay, _ = build_console_exe(delay_import_rva=RDATA_RVA + 0x440)
        self._assert_translate_fails(raw_delay, "unsupported delay imports in console EXE")

    def test_duplicate_import_slot_refused(self):
        raw, _ = build_console_exe(duplicate_import_slot=True)
        self._assert_translate_fails(raw, "duplicate import slot")

    def test_unbound_named_ordinal_and_startup_only_imports_refused(self):
        raw_named, _ = build_console_exe(extra_imports=[(b"kernel32.dll", [b"ExitProcess"])])
        self._assert_translate_fails(raw_named, "unbound console import kernel32.dll!ExitProcess")

        raw_ordinal, _ = build_console_exe(extra_imports=[(b"kernel32.dll", [42])])
        self._assert_translate_fails(raw_ordinal, "unbound console import kernel32.dll!#42")

        # STARTUP_ONLY (cpu_exe_io.dll!finish_exe) is refused when entry_export is None,
        # and also refused if called directly from the lifted body even with entry_export set.
        raw_startup, _ = build_console_exe(include_finish_exe=True)
        self._assert_translate_fails(raw_startup, "unbound console import cpu_exe_io.dll!finish_exe")

        raw_body_call, _ = build_console_exe(include_finish_exe=True, entry_mode="call_finish_exe")
        self._assert_translate_fails(raw_body_call, "no binding", entry_export="native_entry")

    def test_unverified_entry_rva_refused(self):
        raw, meta = build_console_exe(export_native_entry=False, register_entry_pdata=False)
        self._assert_translate_fails(raw, f"unverified process entry at RVA {meta['entry_rva']:#x}")

        # Also test an interior RVA inside .text with no .pdata start or export.
        raw_interior, meta = build_console_exe(
            export_native_entry=False,
            register_entry_pdata=True,
            entry_rva_override=meta["entry_rva"] + 4,
        )
        self._assert_translate_fails(raw_interior, f"unverified process entry at RVA {meta['entry_rva'] + 4:#x}")

    def test_void_and_narrow_returns_without_noreturn_exit_refused(self):
        for mode in ("void_ret", "narrow_ret", "loop_forever"):
            with self.subTest(mode=mode):
                raw, _ = build_console_exe(entry_mode=mode)
                self._assert_translate_fails(raw, "process entry must define its 32-bit exit code")

    def test_cli_exits_1_without_traceback_on_refusals_and_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Missing input file -> OSError -> exit 1 without traceback.
            self._assert_cli_exit_1(root / "does_not_exist.exe")

            # Corrupt non-PE input -> pefile.PEFormatError -> exit 1 without traceback.
            corrupt = root / "corrupt.exe"
            corrupt.write_bytes(b"not a PE image")
            self._assert_cli_exit_1(corrupt)

            # ValueError from invalid PE (GUI subsystem) -> exit 1 without traceback (both in-process & subprocess).
            gui_exe = root / "gui.exe"
            raw_gui, _ = build_console_exe(subsystem=2)
            gui_exe.write_bytes(raw_gui)
            self._assert_cli_exit_1(gui_exe, expected_msg="expected an AMD64 console EXE")

            out_cpp = root / "sub_out.cpp"
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(gui_exe), str(out_cpp)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("expected an AMD64 console EXE", proc.stderr)
            self.assertFalse(out_cpp.exists())


if __name__ == "__main__":
    unittest.main()
