"""Pure-Python unit and CLI hardening suite for tools/intake_exe.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pefile

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "intake_exe.py"
sys.path.insert(0, str(ROOT / "tools"))

import intake_exe  # noqa: E402
from pe_inventory import import_tables  # noqa: E402


IMAGE_BASE = 0x180000000
TEXT_RVA = 0x1000
TEXT_RAW = 0x400
TEXT_SIZE = 0x200
RDATA_RVA = 0x2000
RDATA_RAW = 0x600
RDATA_SIZE = 0x1000
ENTRY_RVA = TEXT_RVA + 0x10
TLS_CB_RVA = TEXT_RVA + 0x40
UNWIND_RVA = TEXT_RVA + 0x100
CALLER = "main.exe"


def _build_pe(
    *,
    imports: list[tuple[str, str, str]] = (
        ("normal", "xmem.dll", "XMemAlloc"),
        ("normal", "kernel32.dll", "ExitProcess"),
    ),
    include_tls: bool = True,
    pdata_records: list[tuple[int, int, int]] | None = ((TEXT_RVA, TEXT_RVA + 0x30, UNWIND_RVA), (TEXT_RVA + 0x30, TEXT_RVA + 0x80, UNWIND_RVA)),
    pdata_size_override: int | None = None,
    machine: int = 0x8664,
    magic: int = 0x20B,
    characteristics: int = 0x0022,
    entry_rva: int = ENTRY_RVA,
    truncate_bytes: int = 0,
) -> bytes:
    """Build a minimal synthetic PE32+ image with .text, .rdata imports, TLS, and .pdata."""
    rdata = bytearray(RDATA_SIZE)
    cursor = 0

    def alloc(size: int, align: int = 4) -> int:
        nonlocal cursor
        cursor = (cursor + align - 1) & ~(align - 1)
        rva = RDATA_RVA + cursor
        if cursor + size > len(rdata):
            raise AssertionError("synthetic .rdata overflow")
        cursor += size
        return rva

    def put(rva: int, blob: bytes) -> None:
        offset = rva - RDATA_RVA
        rdata[offset:offset + len(blob)] = blob

    module_rvas = {}
    for module in sorted({m for _, m, _ in imports}):
        raw = module.encode("ascii") + b"\0"
        rva = alloc(len(raw), 2)
        put(rva, raw)
        module_rvas[module] = rva

    hint_rvas = {}
    for kind, module, symbol in imports:
        raw = struct.pack("<H", 0) + symbol.encode("ascii") + b"\0"
        rva = alloc(len(raw), 2)
        put(rva, raw)
        hint_rvas[(kind, module, symbol)] = rva

    groups: dict[tuple[str, str], list[str]] = {}
    for kind, module, symbol in imports:
        groups.setdefault((kind, module), []).append(symbol)

    tables = {}
    for key in sorted(groups):
        values = [hint_rvas[(key[0], key[1], sym)] for sym in groups[key]]
        blob = b"".join(struct.pack("<Q", v) for v in [*values, 0])
        lookup = alloc(len(blob), 8)
        put(lookup, blob)
        iat = alloc(len(blob), 8)
        put(iat, blob)
        tables[key] = (lookup, iat)

    normal_groups = sorted(k for k in groups if k[0] == "normal")
    delay_groups = sorted(k for k in groups if k[0] == "delay")
    normal_rva = normal_size = delay_rva = delay_size = 0
    if normal_groups:
        normal_size = 20 * (len(normal_groups) + 1)
        normal_rva = alloc(normal_size, 4)
        for idx, key in enumerate(normal_groups):
            lookup, iat = tables[key]
            put(normal_rva + idx * 20, struct.pack("<5I", lookup, 0, 0, module_rvas[key[1]], iat))
    if delay_groups:
        delay_size = 32 * (len(delay_groups) + 1)
        delay_rva = alloc(delay_size, 4)
        for idx, key in enumerate(delay_groups):
            lookup, iat = tables[key]
            put(delay_rva + idx * 32, struct.pack("<8I", 1, module_rvas[key[1]], 0, iat, lookup, 0, 0, 0))

    tls_rva = tls_size = 0
    if include_tls:
        cb_table_rva = alloc(16, 8)
        put(cb_table_rva, struct.pack("<2Q", IMAGE_BASE + TLS_CB_RVA, 0))
        tls_rva = alloc(40, 8)
        tls_size = 40
        put(tls_rva, struct.pack("<4Q2I", IMAGE_BASE + RDATA_RVA, IMAGE_BASE + RDATA_RVA + 8,
                               IMAGE_BASE + RDATA_RVA + 16, IMAGE_BASE + cb_table_rva, 0, 0))

    pdata_rva = pdata_size = 0
    if pdata_records is not None:
        raw_pdata = b"".join(struct.pack("<3I", b, e, u) for b, e, u in pdata_records)
        pdata_rva = alloc(max(len(raw_pdata), 16), 4)
        put(pdata_rva, raw_pdata)
        pdata_size = pdata_size_override if pdata_size_override is not None else len(raw_pdata)

    pe_off = 0x80
    buf = bytearray(RDATA_RAW + RDATA_SIZE)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 60, pe_off)
    buf[pe_off:pe_off + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_off + 4, machine, 2, 0, 0, 0, 0xF0, characteristics)
    opt = pe_off + 24
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        buf,
        opt,
        magic, 14, 0, TEXT_SIZE, RDATA_SIZE, 0, entry_rva, TEXT_RVA, IMAGE_BASE,
        0x1000, 0x200, 6, 0, 0, 0, 6, 0, 0, 0x3000, TEXT_RAW,
        0, 3, 0, 0x100000, 0x1000, 0x100000, 0x1000, 0, 16,
    )
    dd = opt + 112
    if normal_groups:
        struct.pack_into("<II", buf, dd + 1 * 8, normal_rva, normal_size)
    if pdata_records is not None:
        struct.pack_into("<II", buf, dd + 3 * 8, pdata_rva, pdata_size)
    if include_tls:
        struct.pack_into("<II", buf, dd + 9 * 8, tls_rva, tls_size)
    if delay_groups:
        struct.pack_into("<II", buf, dd + 13 * 8, delay_rva, delay_size)

    sec_off = opt + 0xF0
    struct.pack_into(
        "<8sIIIIIIHHI", buf, sec_off,
        b".text\0\0\0", TEXT_SIZE, TEXT_RVA, TEXT_SIZE, TEXT_RAW, 0, 0, 0, 0, 0x60000020,
    )
    struct.pack_into(
        "<8sIIIIIIHHI", buf, sec_off + 40,
        b".rdata\0\0", RDATA_SIZE, RDATA_RVA, RDATA_SIZE, RDATA_RAW, 0, 0, 0, 0, 0x40000040,
    )
    buf[TEXT_RAW + 0x10:TEXT_RAW + 0x11] = b"\xc3"
    buf[TEXT_RAW + 0x40:TEXT_RAW + 0x41] = b"\xc3"
    buf[RDATA_RAW:RDATA_RAW + RDATA_SIZE] = rdata
    if truncate_bytes:
        del buf[-truncate_bytes:]
    return bytes(buf)


def _synthetic_package(tmp: Path, *, write_candidate: bool = True, pe_bytes: bytes | None = None):
    root = (tmp / "scarlett-pkg").resolve()
    root.mkdir(parents=True)
    (root / "MicrosoftGame.Config").write_text(
        '<Game><ExecutableList><Executable TargetDeviceFamily="Scarlett" Name="main.exe"/></ExecutableList></Game>',
        encoding="utf-8",
    )
    (root / "layout.xml").write_text(
        '<Package><Chunk Id="1001"><FileGroup DestinationPath="/" Include="main.exe"/></Chunk></Package>',
        encoding="utf-8",
    )
    torrent_bytes = b"synthetic-torrent-payload"
    (root / "package.torrent").write_bytes(torrent_bytes)
    integrity = tmp / "integrity.json"
    integrity.write_text(
        json.dumps(
            {
                "directory": root.name,
                "torrent_payload_passed": True,
                "torrent": "package.torrent",
                "torrent_sha256": hashlib.sha256(torrent_bytes).hexdigest(),
                "layout": {"source": "layout.xml", "absent_from_torrent": {CALLER: ["1001"]}},
                "pieces": {"matched": 1},
                "finished_utc": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp / "baseline.json"
    baseline.write_text(json.dumps({"schema": 1, "images": {}, "imports": [], "unresolved_edges": []}), encoding="utf-8")
    if write_candidate:
        (root / CALLER).write_bytes(pe_bytes if pe_bytes is not None else _build_pe())
    return root, baseline, integrity


def _fake_scan_for(expected_root: Path, image_bytes: bytes):
    with pefile.PE(data=image_bytes, fast_load=True) as pe:
        imports = import_tables(pe, CALLER)
    digest = hashlib.sha256(image_bytes).hexdigest()
    rows = [
        {
            "image": CALLER,
            "kind": kind,
            "dll": dll,
            "symbol": symbol,
            "bundle_exports_in": [],
            "desktop_provider": None,
            "desktop_export": "provider_not_found",
            "desktop_candidate": None,
            "desktop_resolution": {
                "status": "provider_not_found",
                "chain": [{"dll": dll, "provider": None, "symbol": symbol}],
            },
            "package_resolution": None,
        }
        for kind, dll, symbol in imports
    ]
    inventory = {
        "schema": 1,
        "images": {CALLER: {"sha256": digest, "exports": []}},
        "imports": rows,
        "unresolved_edges": [dict(r) for r in rows],
        "summary": {"images": 1, "edges": len(rows)},
    }

    def fake_scan(package, *, extra_image=None, system=None):
        assert package == expected_root
        return json.loads(json.dumps(inventory))

    return fake_scan


class IntakeExeUnitTests(unittest.TestCase):
    def test_inside_rejects_traversal_root_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            (root / "sub").mkdir(parents=True)
            sub = root / "bin" / "main.exe"
            self.assertEqual(intake_exe.inside(root, "bin/main.exe"), sub.resolve())
            # Unnormalized root containing '..' must still accept valid relative package paths.
            self.assertEqual(intake_exe.inside(root / "sub" / "..", "bin/main.exe"), sub.resolve())
            # Pin each refusal reason explicitly (POSIX/drive/UNC absolute, traversal, root itself, empty).
            cases = (
                ("absolute_posix", "/etc/passwd"),
                ("absolute_drive", "C:/Windows/win.ini"),
                ("absolute_unc", "\\\\server\\share\\x"),
                ("escape_parent", "../escape.txt"),
                ("escape_nested", "bin/../../escape"),
                ("root_dot", "."),
                ("empty", ""),
            )
            for reason, bad in cases:
                with self.subTest(reason=reason, bad=bad), self.assertRaisesRegex(ValueError, "Invalid package path"):
                    intake_exe.inside(root, bad)

    def test_import_routes_and_routing_failures(self):
        pe_bytes = _build_pe(
            imports=[
                ("normal", "xmem.dll", "XMemAlloc"),
                ("normal", "kernel32.dll", "ExitProcess"),
                ("delay", "xgameruntime.dll", "XUserGetId"),
                ("normal", "mystery.dll", "MissingExport"),
            ]
        )
        rows = intake_exe.import_routes(pe_bytes, CALLER)
        by_dll = {r["dll"]: r for r in rows}
        self.assertEqual(by_dll["xmem.dll"]["route"], "hle")
        self.assertEqual(by_dll["kernel32.dll"]["route"], "windows")
        self.assertEqual(by_dll["xgameruntime.dll"]["route"], "stub")
        self.assertEqual(by_dll["mystery.dll"]["route"], "unsupported")
        failed = intake_exe.routing_failures(rows)
        self.assertEqual([(r["dll"], r["symbol"]) for r in failed], [("mystery.dll", "MissingExport")])

    def test_inspect_executable_tls_pdata_and_malformed_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.exe"
            valid.write_bytes(_build_pe())
            info = intake_exe.inspect_executable(valid)
            self.assertEqual(info["machine"], "AMD64")
            self.assertEqual(info["format"], "PE32+")
            self.assertEqual(info["entry_rva"], ENTRY_RVA)
            self.assertEqual(info["entry_section"], ".text")
            self.assertEqual(info["tls"]["callbacks_rva"], [TLS_CB_RVA])
            startup = info["startup"]
            self.assertEqual(startup["exception_table"], "indexed")
            self.assertEqual(startup["runtime_functions"], 2)
            self.assertEqual([loc["role"] for loc in startup["locations"]], ["tls_callback", "entry"])
            self.assertEqual(startup["locations"][0]["runtime_function"]["begin_rva"], TEXT_RVA + 0x30)
            self.assertEqual(startup["locations"][1]["runtime_function"]["begin_rva"], TEXT_RVA)

            # Absent .pdata -> exception_table == "absent", runtime_function is None.
            no_pdata = root / "no_pdata.exe"
            no_pdata.write_bytes(_build_pe(pdata_records=None, include_tls=False))
            no_pdata_info = intake_exe.inspect_executable(no_pdata)
            self.assertEqual(no_pdata_info["startup"]["exception_table"], "absent")
            self.assertIsNone(no_pdata_info["startup"]["locations"][0]["runtime_function"])

            # Misaligned and overlapping .pdata fall back cleanly to exception_table == "unusable".
            misaligned = root / "misaligned_pdata.exe"
            misaligned.write_bytes(_build_pe(pdata_size_override=10))
            self.assertEqual(intake_exe.inspect_executable(misaligned)["startup"]["exception_table"], "unusable")

            overlapping = root / "overlap_pdata.exe"
            overlapping.write_bytes(
                _build_pe(pdata_records=[(TEXT_RVA, TEXT_RVA + 0x50, UNWIND_RVA), (TEXT_RVA + 0x40, TEXT_RVA + 0x80, UNWIND_RVA)])
            )
            overlap_startup = intake_exe.inspect_executable(overlapping)["startup"]
            self.assertEqual(overlap_startup["exception_table"], "unusable")
            self.assertIn("Invalid or overlapping", overlap_startup["detail"])

            # Non-x64, DLL, and truncated sections are rejected with ValueError.
            for label, pe_kwargs in (
                ("non_x64", {"machine": 0x014C, "magic": 0x10B}),
                ("dll", {"characteristics": 0x2022}),
                ("truncated_section", {"truncate_bytes": 128}),
            ):
                with self.subTest(case=label):
                    bad = root / f"{label}.exe"
                    bad.write_bytes(_build_pe(**pe_kwargs))
                    with self.assertRaises(ValueError):
                        intake_exe.inspect_executable(bad)


class IntakeExeCliTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str, str]:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = intake_exe.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_collect_and_main_awaiting_inspected_and_require_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_tmp = Path(tmp)
            # 1. awaiting_executable -> exit 3
            pkg_wait, baseline_w, integrity_w = _synthetic_package(root_tmp / "wait", write_candidate=False)
            (root_tmp / "wait" / "reports").mkdir()
            rep_wait = root_tmp / "wait" / "reports"
            rc_wait, out_wait, err_wait = self._run_main(
                ["--build", str(pkg_wait), "--baseline", str(baseline_w), "--integrity", str(integrity_w), "--reports", str(rep_wait)]
            )
            self.assertEqual(rc_wait, 3, (out_wait, err_wait))
            self.assertEqual(err_wait, "")
            self.assertIn("awaiting_executable |", out_wait)
            self.assertEqual(len(list(rep_wait.glob("*.awaiting_executable.json"))), 1)

            # 2. inspected -> exit 0, and --require-routing with unsupported import -> exit 1
            pe_bytes = _build_pe(
                imports=[
                    ("normal", "xmem.dll", "XMemAlloc"),
                    ("normal", "mystery.dll", "MissingExport"),
                ]
            )
            pkg_ok, baseline_ok, integrity_ok = _synthetic_package(root_tmp / "ok", write_candidate=True, pe_bytes=pe_bytes)
            rep_ok = root_tmp / "ok" / "reports"
            common = ["--build", str(pkg_ok), "--baseline", str(baseline_ok), "--integrity", str(integrity_ok), "--reports", str(rep_ok)]

            with mock.patch.object(intake_exe, "scan", _fake_scan_for(pkg_ok, pe_bytes)):
                rc_ok, out_ok, err_ok = self._run_main(common)
                self.assertEqual(rc_ok, 0, (out_ok, err_ok))
                self.assertIn("inspected |", out_ok)

                rc_req, out_req, err_req = self._run_main([*common, "--require-routing"])
                self.assertEqual(rc_req, 1, (out_req, err_req))
                self.assertIn("unsupported: main.exe -> mystery.dll!MissingExport", err_req)
                self.assertIn("inspected |", out_req)

            # Direct collect() smoke check.
            report_dict = {}
            with mock.patch.object(intake_exe, "scan", _fake_scan_for(pkg_ok, pe_bytes)):
                collect_rc = intake_exe.collect(pkg_ok, None, baseline_ok, integrity_ok, report_dict)
            self.assertEqual(collect_rc, 0)
            self.assertEqual(report_dict["status"], "inspected")
            self.assertEqual(report_dict["pe"]["startup"]["exception_table"], "indexed")

            # Subprocess smoke test on awaiting_executable -> exit 3 without traceback.
            proc = subprocess.run(
                [
                    sys.executable, "-B", str(TOOL),
                    "--build", str(pkg_wait),
                    "--baseline", str(baseline_w),
                    "--integrity", str(integrity_w),
                    "--reports", str(rep_wait),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("awaiting_executable |", proc.stdout)

    def test_negative_inputs_and_atomic_temp_cleanup_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_tmp = Path(tmp)
            pkg, baseline, integrity = _synthetic_package(root_tmp / "neg", write_candidate=False)
            reports = root_tmp / "neg" / "reports"
            reports.mkdir()

            # Missing candidate passed explicitly -> error report, exit 2, zero traceback.
            missing_exe = root_tmp / "neg" / "missing.exe"
            rc_missing, out_missing, err_missing = self._run_main(
                [str(missing_exe), "--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(reports)]
            )
            self.assertEqual(rc_missing, 2)
            self.assertNotIn("Traceback", err_missing)
            self.assertIn("error |", out_missing)

            # Malformed MicrosoftGame.Config -> error report, exit 2, zero traceback.
            (pkg / "MicrosoftGame.Config").write_text("<broken", encoding="utf-8")
            rc_xml, out_xml, err_xml = self._run_main(
                ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(reports)]
            )
            self.assertEqual(rc_xml, 2)
            self.assertNotIn("Traceback", err_xml)
            self.assertIn("error |", out_xml)

            # --reports colliding with build directory -> SystemExit(2), zero traceback.
            out_buf, err_buf = io.StringIO(), io.StringIO()
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                with self.assertRaises(SystemExit) as exc:
                    intake_exe.main(["--build", str(pkg), "--reports", str(pkg / "sub")])
            self.assertEqual(exc.exception.code, 2)
            self.assertNotIn("Traceback", err_buf.getvalue())

            # --reports pointing to an existing regular file -> exit 1, zero traceback.
            file_reports = root_tmp / "neg" / "not_a_dir.txt"
            file_reports.write_text("file", encoding="utf-8")
            rc_file_rep, out_file_rep, err_file_rep = self._run_main(
                ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(file_reports)]
            )
            self.assertEqual(rc_file_rep, 1)
            self.assertEqual(out_file_rep, "")
            self.assertNotIn("Traceback", err_file_rep)
            self.assertIn("failed to write report", err_file_rep)

            # Failure during json.dump must unlink temp file and exit 1 without traceback.
            clean_rep = root_tmp / "neg" / "clean_dump"
            clean_rep.mkdir()
            with mock.patch.object(intake_exe.json, "dump", side_effect=OSError("disk full during dump")):
                rc_dump, out_dump, err_dump = self._run_main(
                    ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(clean_rep)]
                )
            self.assertEqual(rc_dump, 1)
            self.assertEqual(out_dump, "")
            self.assertNotIn("Traceback", err_dump)
            self.assertIn("disk full during dump", err_dump)
            self.assertEqual(list(clean_rep.iterdir()), [])

            # Failure during os.replace must unlink temp file and exit 1 without traceback.
            clean_rep2 = root_tmp / "neg" / "clean_replace"
            clean_rep2.mkdir()
            with mock.patch.object(intake_exe.os, "replace", side_effect=PermissionError("locked dest")):
                rc_rep, out_rep, err_rep = self._run_main(
                    ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(clean_rep2)]
                )
            self.assertEqual(rc_rep, 1)
            self.assertEqual(out_rep, "")
            self.assertNotIn("Traceback", err_rep)
            self.assertIn("locked dest", err_rep)
            self.assertEqual(list(clean_rep2.iterdir()), [])

            # OSError during Path.resolve -> exit 1 without traceback.
            with mock.patch.object(Path, "resolve", side_effect=OSError("bad symlink")):
                rc_res, out_res, err_res = self._run_main(["--build", str(pkg), "--reports", str(clean_rep2)])
            self.assertEqual(rc_res, 1)
            self.assertEqual(out_res, "")
            self.assertNotIn("Traceback", err_res)
            self.assertIn("bad symlink", err_res)


if __name__ == "__main__":
    unittest.main()
