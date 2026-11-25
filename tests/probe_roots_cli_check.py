"""Pure-Python unit and CLI hardening suite for tools/probe_roots.py."""

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

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "probe_roots.py"
sys.path.insert(0, str(ROOT / "tools"))

import probe_roots as pr  # noqa: E402


IMAGE_BASE = 0x180000000
SECTION_RVA = 0x1000
RAW_OFFSET = 0x200
RAW_SIZE = 0x1000


def _make_dxbc(rts0_payload: bytes | None = b"RTS0_SYNTHETIC_PAYLOAD_01", *, include_dxil: bool = True) -> bytes:
    parts_list = []
    if include_dxil:
        program = struct.pack("<II4sIII", 0x00050060, 7, b"DXIL", 0, 16, 4) + b"BC\xc0\xde"
        parts_list.append(b"DXIL" + struct.pack("<I", len(program)) + program)
    if rts0_payload is not None:
        parts_list.append(b"RTS0" + struct.pack("<I", len(rts0_payload)) + rts0_payload)
    table_end = 32 + 4 * len(parts_list)
    offsets = []
    cursor = table_end
    for chunk in parts_list:
        offsets.append(cursor)
        cursor += len(chunk)
    body = b"".join(parts_list)
    total = table_end + len(body)
    return (
        b"DXBC"
        + bytes(16)
        + struct.pack("<III", 1, total, len(parts_list))
        + struct.pack(f"<{len(parts_list)}I", *offsets)
        + body
    )


def _make_pe_with_dxbc(blobs: list[bytes]) -> tuple[bytes, list[tuple[int, int]]]:
    payload = bytearray(RAW_SIZE)
    cursor = 0
    locations = []
    for blob in blobs:
        if cursor + len(blob) > len(payload):
            raise AssertionError("synthetic .rdata overflow")
        payload[cursor:cursor + len(blob)] = blob
        locations.append((SECTION_RVA + cursor, len(blob)))
        cursor = (cursor + len(blob) + 15) & ~15

    pe_off = 0x80
    buf = bytearray(RAW_OFFSET + RAW_SIZE)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 60, pe_off)
    buf[pe_off:pe_off + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_off + 4, 0x8664, 1, 0, 0, 0, 0xF0, 0x2022)
    opt = pe_off + 24
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        buf,
        opt,
        0x20B, 14, 0, 0, RAW_SIZE, 0, 0, SECTION_RVA, IMAGE_BASE,
        0x1000, 0x200, 6, 0, 0, 0, 6, 0, 0, 0x2000, RAW_OFFSET,
        0, 3, 0, 0x100000, 0x1000, 0x100000, 0x1000, 0, 16,
    )
    sec = opt + 0xF0
    struct.pack_into(
        "<8sIIIIIIHHI",
        buf,
        sec,
        b".rdata\0\0", RAW_SIZE, SECTION_RVA, RAW_SIZE, RAW_OFFSET, 0, 0, 0, 0, 0x40000040,
    )
    buf[RAW_OFFSET:RAW_OFFSET + RAW_SIZE] = payload
    return bytes(buf), locations


SYNTHETIC_IR = """\
target triple = "dxil-ms-dx"
!xbox.compiler.options = !{!0}
!0 = !{!"meta;keep xbox.meta_only"}
declare void @xbox.unused_decl()
@xbox.alias_sym = alias void (), void ()* @xbox.bitcast_target
@xbox.global_ptr = constant i8* bitcast (void ()* @xbox.global_target to i8*)
define void @main(i32 %idx, i64 %base, i64 %other) {
    %sel = and i32 3, %idx
    %a = call i32 @xbox.V_ALIGNBYTE_B32(i32 1, i32 2, i32 3, i32 %sel) ; trailing comment @xbox.ignored
    %b = call i32 bitcast (i32 (i32)* @xbox.bitcast_callee to i32 (i32)*)(i32 %a)
    call void @llvm.dbg.value(metadata void ()* @xbox.debug_ref, metadata !0, metadata !DIExpression())
    store void ()* @xbox.stored_ref, void ()** undef
    call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 0, float 1.0)
    call void @other_fn(void ()* @xbox.operand_ref)
    call void @xbox.global_store_dwordx3_with_cache(i32 1, i64 %base, { i32, i32, i32 } { i32 2139095039, i32 2139095039, i32 2139095039 })
    %next = add i64 %base, 12
    call void @xbox.global_store_dwordx3_with_cache(i32 1, i64 %next, { i32, i32, i32 } { i32 -8388609, i32 -8388609, i32 -8388609 })
    ret void
}
"""


def _fake_subprocess_run(cmd, *, input=None, capture_output=False, check=False, timeout=None):
    flag = cmd[1] if len(cmd) > 1 else ""
    if flag in ("--container", "--module"):
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"status": "0x00000000"}).encode("utf-8"), b"")
    if flag == "--disassemble":
        return subprocess.CompletedProcess(cmd, 0, SYNTHETIC_IR.encode("utf-8"), b"")
    row = {
        "legacy_hr": "0x00000000",
        "versioned_hr": "0x00000000",
        "original_serialize_hr": "0x00000000",
        "without_observed_bits_hr": "0x00000000",
    }
    return subprocess.CompletedProcess(cmd, 0, json.dumps(row).encode("utf-8"), b"")


class ProbeRootsUnitTests(unittest.TestCase):
    def test_strip_comment_and_call_operand_paren(self):
        self.assertEqual(pr.strip_comment('!"a;b" ; comment'), '!"a;b" ')
        self.assertEqual(pr.strip_comment("ret void ; @xbox.ignored"), "ret void ")
        self.assertEqual(pr.strip_comment('call void @"semi;name"()'), 'call void @"semi;name"()')

        direct = "  %x = call i32 @xbox.foo(i32 1, i32 2)"
        self.assertEqual(direct[pr.call_operand_paren(direct):], "(i32 1, i32 2)")

        bitcast = "  call void bitcast (void (i32)* @xbox.callee to void (i32)*)(i32 %a)"
        self.assertEqual(bitcast[pr.call_operand_paren(bitcast):], "(i32 %a)")
        self.assertEqual(pr.call_operand_paren("ret void"), -1)

    def test_classify_line_inventory_and_packing_evidence(self):
        calls, declarations, refs = {}, set(), {}
        from collections import Counter
        calls = Counter()
        pr.classify_line("declare void @xbox.decl()", calls, declarations, refs)
        pr.classify_line("  %x = call i32 @xbox.direct(i32 1)", calls, declarations, refs)
        pr.classify_line("  call void @llvm.dbg.value(metadata void ()* @xbox.dbg, metadata !0)", calls, declarations, refs)
        self.assertEqual(declarations, {"xbox.decl"})
        self.assertEqual(dict(calls), {"xbox.direct": 1})
        self.assertEqual(refs["xbox.dbg"], {"sites": 0, "debug_sites": 1, "kinds": {"debug-only"}})

        inv = pr.xbox_intrinsic_inventory(SYNTHETIC_IR)
        self.assertEqual(inv["calls"], {"xbox.V_ALIGNBYTE_B32": 1, "xbox.bitcast_callee": 1, "xbox.global_store_dwordx3_with_cache": 2})
        self.assertEqual(inv["declarations"], ["xbox.unused_decl"])
        self.assertEqual(inv["metadata_only"], ["xbox.meta_only"])
        self.assertEqual(inv["references"]["xbox.bitcast_target"]["kinds"], ["alias"])
        self.assertEqual(inv["references"]["xbox.global_target"]["kinds"], ["bitcast"])
        self.assertEqual(inv["references"]["xbox.debug_ref"]["debug_sites"], 1)
        self.assertEqual(inv["references"]["xbox.debug_ref"]["sites"], 0)
        self.assertEqual(inv["references"]["xbox.stored_ref"]["kinds"], ["store"])
        self.assertEqual(inv["references"]["xbox.operand_ref"]["kinds"], ["call-operand"])

        pack = pr.packing_evidence(SYNTHETIC_IR)
        self.assertEqual(pack["alignbyte_selectors"], {"and_3": 1})
        self.assertEqual(pack["store_addresses"], {"dwordx3:other_or_unknown": 1, "dwordx3:add_12": 1})
        self.assertEqual(pack["adjacent_extrema_store_pairs"], 1)

    def test_measure_on_synthetic_pe_and_dxbc(self):
        rts0 = bytearray(28)
        struct.pack_into("<I", rts0, 20, 0x03000001)
        dxbc = _make_dxbc(bytes(rts0))
        pe_bytes, locs = _make_pe_with_dxbc([dxbc])
        desktop_rts0 = bytes(rts0[:20]) + struct.pack("<I", 0x00000001) + bytes(rts0[24:])

        inventory = {
            name: (
                {"sha256": hashlib.sha256(pe_bytes).hexdigest(), "containers": [{"rva": hex(locs[0][0]), "bytes": locs[0][1]}]}
                if i == 0 else {"sha256": "0" * 64, "containers": []}
            )
            for i, name in enumerate(pr.NAMES)
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            build.mkdir()
            (build / pr.NAMES[0]).write_bytes(pe_bytes)
            probe = root / "root_probe.exe"
            probe.write_bytes(b"stub")

            with mock.patch.object(pr.subprocess, "run", side_effect=_fake_subprocess_run):
                measured = pr.measure(build, inventory, probe, {"compute_root": desktop_rts0})

        entry = measured[pr.NAMES[0]]
        self.assertEqual(entry["containers"], 1)
        self.assertEqual(entry["unique_root_payloads"], 1)
        self.assertEqual(entry["dxil_validation"], {"container": {"0x00000000": 1}, "module": {"0x00000000": 1}})
        self.assertEqual(entry["modules_with_xbox_calls"], 1)
        self.assertEqual(entry["fixture_matches"], {"compute_root": 1})
        self.assertEqual(entry["entries"][0]["program_kind"], 5)
        self.assertEqual(entry["entries"][0]["shader_model"], "6.0")


class ProbeRootsCliTests(unittest.TestCase):
    def _setup_workspace(self, tmp: Path):
        rts0 = bytearray(28)
        struct.pack_into("<I", rts0, 20, 0x03000001)
        desktop_rts0 = bytes(rts0[:20]) + struct.pack("<I", 0x00000001) + bytes(rts0[24:])
        ctrl_blob = _make_dxbc(desktop_rts0)
        dxbc = _make_dxbc(bytes(rts0))
        pe_bytes, locs = _make_pe_with_dxbc([dxbc])

        build = tmp / "build"
        build.mkdir()
        (build / pr.NAMES[0]).write_bytes(pe_bytes)
        controls = tmp / "controls"
        controls.mkdir()
        (controls / "compute_root.cso").write_bytes(ctrl_blob)
        (controls / "compute_tables.cso").write_bytes(ctrl_blob)
        probe = tmp / "root_probe.exe"
        probe.write_bytes(b"stub")

        inv_path = tmp / "six-apis.json"
        inventory = {
            name: (
                {"sha256": hashlib.sha256(pe_bytes).hexdigest(), "containers": [{"rva": hex(locs[0][0]), "bytes": locs[0][1]}]}
                if i == 0 else {"sha256": "0" * 64, "containers": []}
            )
            for i, name in enumerate(pr.NAMES)
        }
        inv_path.write_text(json.dumps(inventory), encoding="utf-8")
        return build, inv_path, probe, controls

    def _assert_main_fails(self, argv: list[str], *, expected_code: int, fragment: str = "") -> str:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit) as exc:
                pr.main(argv)
        self.assertEqual(exc.exception.code, expected_code)
        err_text = err.getvalue()
        self.assertNotIn("Traceback", err_text)
        self.assertTrue(err_text.strip(), f"expected stderr for {argv!r}")
        if fragment:
            self.assertIn(fragment, err_text)
        return err_text

    def test_main_argv_output_and_controls_exit_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build, inv_path, probe, controls = self._setup_workspace(root)
            out_file = root / "reports" / "desktop-roots.json"

            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch.object(pr.subprocess, "run", side_effect=_fake_subprocess_run), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                rc = pr.main([
                    str(build),
                    "--inventory", str(inv_path),
                    "--probe", str(probe),
                    "--controls", str(controls),
                    "--output", str(out_file),
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn(pr.NAMES[0], stdout.getvalue())
            doc = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(doc["schema"], 1)
            self.assertIn("compute_root", doc["control_validation"])
            self.assertEqual(doc["images"][pr.NAMES[0]]["fixture_matches"], {"compute_root": 1, "compute_tables": 1})

            # Default --output resolves to PROJECT / out/calls/desktop-roots.json.
            fake_proj = root / "proj"
            fake_proj.mkdir()
            with mock.patch.object(pr, "PROJECT", fake_proj), \
                    mock.patch.object(pr.subprocess, "run", side_effect=_fake_subprocess_run), \
                    redirect_stdout(io.StringIO()):
                rc_def = pr.main([str(build), "--inventory", str(inv_path), "--probe", str(probe)])
            self.assertEqual(rc_def, 0)
            self.assertTrue((fake_proj / "out" / "calls" / "desktop-roots.json").is_file())

    def test_negative_and_mutation_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build, inv_path, probe, controls = self._setup_workspace(root)
            out_file = root / "out.json"

            # Subprocess missing build -> exit 1 without traceback.
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(root / "missing-build"), "--output", str(out_file)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip())

            # Missing probe / missing inventory / wrong path types -> exit 1 without traceback.
            self._assert_main_fails([str(build), "--probe", str(root / "missing.exe"), "--inventory", str(inv_path), "--output", str(out_file)], expected_code=1)
            self._assert_main_fails([str(build), "--probe", str(probe), "--inventory", str(root / "missing.json"), "--output", str(out_file)], expected_code=1)
            self._assert_main_fails([str(probe), "--probe", str(probe), "--inventory", str(inv_path), "--output", str(out_file)], expected_code=1, fragment="not a directory")
            self._assert_main_fails([str(build), "--probe", str(build), "--inventory", str(inv_path), "--output", str(out_file)], expected_code=1, fragment="not a file")

            # Output overlapping --build, --probe, --inventory, --controls, or existing dir -> exit 2 without traceback.
            existing_dir = root / "existing_dir"
            existing_dir.mkdir()
            for bad_out in (build, build / "nested.json", probe, inv_path, controls / "r.json", existing_dir, Path("")):
                with self.subTest(bad_out=str(bad_out)):
                    self._assert_main_fails(
                        [str(build), "--inventory", str(inv_path), "--probe", str(probe), "--controls", str(controls), "--output", str(bad_out)],
                        expected_code=2, fragment="overlaps inputs or is a directory",
                    )

            # Malformed inventory JSON and wrong JSON shape -> exit 1 without traceback.
            bad_json = root / "bad.json"
            bad_json.write_text("{not json", encoding="utf-8")
            self._assert_main_fails([str(build), "--inventory", str(bad_json), "--probe", str(probe), "--output", str(out_file)], expected_code=1)

            list_json = root / "list.json"
            list_json.write_text("[]", encoding="utf-8")
            self._assert_main_fails([str(build), "--inventory", str(list_json), "--probe", str(probe), "--output", str(out_file)], expected_code=1)

            # Failing desktop positive control -> exit 1 without traceback.
            def fake_bad_control(cmd, *, input=None, capture_output=False, check=False, timeout=None):
                flag = cmd[1] if len(cmd) > 1 else ""
                if flag in ("--container", "--module"):
                    return subprocess.CompletedProcess(cmd, 0, json.dumps({"status": "0x80004005"}).encode("utf-8"), b"")
                return _fake_subprocess_run(cmd, input=input, capture_output=capture_output, check=check, timeout=timeout)

            with mock.patch.object(pr.subprocess, "run", side_effect=fake_bad_control):
                self._assert_main_fails(
                    [str(build), "--inventory", str(inv_path), "--probe", str(probe), "--controls", str(controls), "--output", str(out_file)],
                    expected_code=1, fragment="Desktop positive control failed",
                )

            # SubprocessError during probe -> exit 1 without traceback.
            with mock.patch.object(pr.subprocess, "run", side_effect=subprocess.CalledProcessError(2, ["probe"])):
                self._assert_main_fails(
                    [str(build), "--inventory", str(inv_path), "--probe", str(probe), "--output", str(out_file)],
                    expected_code=1,
                )

            # Corrupt PE & SHA mismatch & PEFormatError & struct.error & KeyError -> exit 1 without traceback.
            bad_pe_build = root / "bad_pe_build"
            bad_pe_build.mkdir()
            bad_pe_bytes = b"not a PE"
            (bad_pe_build / pr.NAMES[0]).write_bytes(bad_pe_bytes)
            self._assert_main_fails(
                [str(bad_pe_build), "--inventory", str(inv_path), "--probe", str(probe), "--output", str(out_file)],
                expected_code=1, fragment="Rebuild the call inventory",
            )

            # Matching SHA on non-PE -> pefile.PEFormatError caught cleanly.
            bad_pe_inv = root / "bad_pe_inv.json"
            bad_pe_inv.write_text(json.dumps({
                name: ({"sha256": hashlib.sha256(bad_pe_bytes).hexdigest(), "containers": [{"rva": "0x1000", "bytes": 32}]}
                       if i == 0 else {"sha256": "0" * 64, "containers": []})
                for i, name in enumerate(pr.NAMES)
            }), encoding="utf-8")
            self._assert_main_fails(
                [str(bad_pe_build), "--inventory", str(bad_pe_inv), "--probe", str(probe), "--output", str(out_file)],
                expected_code=1,
            )

            # Truncated RTS0 (<24 bytes) -> struct.error caught cleanly.
            short_rts0_pe, short_locs = _make_pe_with_dxbc([_make_dxbc(b"1234")])
            short_build = root / "short_rts0_build"
            short_build.mkdir()
            (short_build / pr.NAMES[0]).write_bytes(short_rts0_pe)
            short_inv = root / "short_rts0_inv.json"
            short_inv.write_text(json.dumps({
                name: ({"sha256": hashlib.sha256(short_rts0_pe).hexdigest(),
                        "containers": [{"rva": hex(short_locs[0][0]), "bytes": short_locs[0][1]}]}
                       if i == 0 else {"sha256": "0" * 64, "containers": []})
                for i, name in enumerate(pr.NAMES)
            }), encoding="utf-8")
            with mock.patch.object(pr.subprocess, "run", side_effect=_fake_subprocess_run):
                self._assert_main_fails(
                    [str(short_build), "--inventory", str(short_inv), "--probe", str(probe), "--output", str(out_file)],
                    expected_code=1,
                )

            # Missing RTS0 part & empty inventory dict -> KeyError caught cleanly.
            no_rts0_pe, no_rts0_locs = _make_pe_with_dxbc([_make_dxbc(None)])
            no_rts0_build = root / "no_rts0_build"
            no_rts0_build.mkdir()
            (no_rts0_build / pr.NAMES[0]).write_bytes(no_rts0_pe)
            no_rts0_inv = root / "no_rts0_inv.json"
            no_rts0_inv.write_text(json.dumps({
                name: ({"sha256": hashlib.sha256(no_rts0_pe).hexdigest(),
                        "containers": [{"rva": hex(no_rts0_locs[0][0]), "bytes": no_rts0_locs[0][1]}]}
                       if i == 0 else {"sha256": "0" * 64, "containers": []})
                for i, name in enumerate(pr.NAMES)
            }), encoding="utf-8")
            self._assert_main_fails(
                [str(no_rts0_build), "--inventory", str(no_rts0_inv), "--probe", str(probe), "--output", str(out_file)],
                expected_code=1,
            )
            empty_dict_inv = root / "empty_dict_inv.json"
            empty_dict_inv.write_text("{}", encoding="utf-8")
            self._assert_main_fails(
                [str(build), "--inventory", str(empty_dict_inv), "--probe", str(probe), "--output", str(out_file)],
                expected_code=1,
            )

            # Temp file unlink verified on both json.dump and os.replace failures.
            rep_dir = root / "atomic_rep"
            rep_dir.mkdir()
            dest_json = rep_dir / "desktop-roots.json"
            dest_json.write_text("existing", encoding="utf-8")

            with mock.patch.object(pr.subprocess, "run", side_effect=_fake_subprocess_run), \
                    mock.patch.object(pr.json, "dump", side_effect=OSError("disk full dump")):
                self._assert_main_fails(
                    [str(build), "--inventory", str(inv_path), "--probe", str(probe), "--output", str(dest_json)],
                    expected_code=1, fragment="disk full dump",
                )
            self.assertEqual(list(rep_dir.iterdir()), [dest_json])
            self.assertEqual(dest_json.read_text(encoding="utf-8"), "existing")

            with mock.patch.object(pr.subprocess, "run", side_effect=_fake_subprocess_run), \
                    mock.patch.object(pr.os, "replace", side_effect=PermissionError("locked target")):
                self._assert_main_fails(
                    [str(build), "--inventory", str(inv_path), "--probe", str(probe), "--output", str(dest_json)],
                    expected_code=1, fragment="locked target",
                )
            self.assertEqual(list(rep_dir.iterdir()), [dest_json])
            self.assertEqual(dest_json.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
