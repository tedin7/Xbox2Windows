"""Pure-Python unit and CLI validation suite for tools/script_state_probe.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import mmap
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "script_state_probe.py"
sys.path.insert(0, str(ROOT / "tools"))

import script_state_probe as probe  # noqa: E402


def _attr(name: str, spelling: str, raw: int, kind: int) -> bytes:
    return struct.pack("<IIIBBH", probe.joaat(name), probe.joaat(spelling), raw, kind, 1, 0)


def _camera_attrs(priority: int = 9, *, orbit_raw: int = 0x3FC00000, drop: int | None = None) -> list[bytes]:
    rows = [
        _attr("priority", str(priority), priority, 3),
        _attr("blendduration", "37", 37, 3),
        _attr("orbitdistance", "1.500000", orbit_raw, 2),
        _attr("pitchlimits", "-0.250000,0.500000,0.000000", 0, 1),
        _attr("headinglimits", "-1.000000,1.000000,0.000000", 1, 1),
    ]
    return [row for idx, row in enumerate(rows, 1) if idx != drop]


def _build_record_fields(
    *,
    priorities: tuple[int, ...] = (9,),
    state_label: str = "synthetic_orbit",
    duplicate_state_id: bool = False,
    id_kind: int = 5,
    root_id: int = 1,
    marker: int = 0xAAAAAAAAAAAAAAAA,
    truncate_schema_by: int = 0,
    lrsc_magic: bytes = b"0DRPz",
    chks_payload_len: int = 12,
    chks_total_delta: int = 0,
    corrupt_zlib: bool = False,
) -> bytes:
    """Build one 16-byte-aligned zlib-compressed PSIN..CHKS record payload (without 9FPR header)."""
    attrs = bytearray()
    nodes = bytearray()

    def add_node(name: str, parent: int, row_list: list[bytes]) -> int:
        idx = len(nodes) // 16
        nodes.extend(
            struct.pack(
                "<I5H2B",
                probe.joaat(name),
                parent,
                0,
                0,
                0,
                len(attrs) // 16,
                len(row_list),
                1,
            )
        )
        attrs.extend(b"".join(row_list))
        return idx

    root_parent = 65535
    if len(priorities) > 1:
        root_parent = add_node("root", root_parent, [])
    for priority in priorities:
        ids = [_attr("id", state_label, 0, id_kind)]
        if duplicate_state_id:
            ids.append(_attr("id", "other_label", 0, 5))
        state_idx = add_node("state", root_parent, ids)
        add_node("camera", state_idx, _camera_attrs(priority))

    vecs = struct.pack("<6f", -0.25, 0.5, 0.0, -1.0, 1.0, 0.0)
    arrays = ((len(attrs) // 16, 2), (0, 0), (2, 4), (0, 0), (len(nodes) // 16, 3))
    root_blob = b"".join(struct.pack("<IIQ", n, n, ref) for n, ref in arrays)
    blocks = (root_blob, bytes(attrs), bytes(nodes), vecs)

    instances = bytearray()
    for kind, blob in enumerate(blocks, 1):
        instances.extend(struct.pack("<IIQ", kind, len(blob), marker))
    for blob in blocks:
        instances.extend(bytes(-(len(instances) + 8) % 16))
        instances.extend(blob)

    definitions = (
        (80, (("attributes", 0, 0x5C),)),
        (16, (("name", 0, 0x4A), ("valueHash", 4, 0x4A))),
        (16, (("name", 0, 0x4A), ("parentIndex", 4, 0x13), ("attributeCount", 14, 0x11))),
        (12, tuple((axis, i * 4, 0x14) for i, axis in enumerate("xyz"))),
    )
    schema = bytearray()
    for kind, (size, fields) in enumerate(definitions, 1):
        schema.extend(struct.pack("<IIIIQIHH", kind, 0, 0, 0, 0, size, 0, len(fields)))
    for _, fields in definitions:
        for field_name, off, code in fields:
            schema.extend(struct.pack("<IIBBHI", probe.joaat(field_name), off, code, 0, 0, 0))
    if truncate_schema_by:
        del schema[-truncate_schema_by:]

    header = bytearray(80)
    header[:5] = lrsc_magic
    struct.pack_into("<I", header, 16, root_id)
    struct.pack_into("<3H", header, 56, 4, 0, 4)

    body = bytearray()
    for tag, part in (
        (b"PSIN", instances),
        (b"LSCH", schema),
        (b"LENM", b""),
        (b"LSNS", b""),
        (b"LGID", b""),
        (b"LRSC", header),
    ):
        body.extend(struct.pack(">4sI", tag, len(part) + 8))
        body.extend(part)

    chks_section_size = 8 + chks_payload_len
    declared_total = len(body) + chks_section_size + chks_total_delta
    chks_payload = struct.pack(">I", declared_total) + bytes(max(0, chks_payload_len - 4))
    body.extend(struct.pack(">4sI", b"CHKS", chks_section_size))
    body.extend(chks_payload[:chks_payload_len])

    compressed = zlib.compress(bytes(body))
    if corrupt_zlib:
        # Keep valid zlib CMF/FLG prefix at 16-byte boundary, corrupt deflate payload.
        compressed = compressed[:4] + b"\xff\x00\xff\x00" + compressed[8:]
    return compressed


def _build_rpf9(*records_payloads: bytes) -> bytes:
    archive = bytearray(b"9FPR")
    for payload in records_payloads:
        archive.extend(bytes(-len(archive) % 16))
        archive.extend(payload)
    return bytes(archive)


def _unpack_first_fields(archive_bytes: bytes) -> dict[str, memoryview]:
    mv = memoryview(archive_bytes)
    found = list(probe.records(mv))
    assert len(found) == 1, f"expected 1 record, got {len(found)}"
    return found[0]


class ScriptStateProbeUnitTests(unittest.TestCase):
    def test_joaat_and_literal_spellings(self):
        self.assertEqual(probe.joaat(""), 0)
        self.assertEqual(probe.joaat("a"), 0xCA2E9442)
        self.assertEqual(probe.joaat("state"), 0x77A848C7)
        with self.assertRaises(UnicodeEncodeError):
            probe.joaat("café")

        vecs = struct.pack("<3f", -0.25, 0.5, 0.0)
        self.assertEqual(probe.literal(0xFFFFFFFB, 3, vecs), "-5")
        self.assertEqual(probe.literal(0x3FC00000, 2, vecs), "1.500000")
        self.assertEqual(probe.literal(0, 1, vecs), "-0.250000,0.500000,0.000000")
        self.assertIsNone(probe.literal(1, 1, vecs))
        self.assertIsNone(probe.literal(0x7FC00000, 2, vecs))
        self.assertIsNone(probe.literal(0, 5, vecs))
        non_finite_vecs = struct.pack("<3f", 1.0, float("inf"), 0.0)
        self.assertIsNone(probe.literal(0, 1, non_finite_vecs))

    def test_verified_camera_accepts_exact_hashes_and_rejects_mismatches(self):
        vecs = struct.pack("<6f", -0.25, 0.5, 0.0, -1.0, 1.0, 0.0)
        values = {
            probe.joaat("priority"): (probe.joaat("9"), 9, 3),
            probe.joaat("blendduration"): (probe.joaat("37"), 37, 3),
            probe.joaat("orbitdistance"): (probe.joaat("1.500000"), 0x3FC00000, 2),
            probe.joaat("pitchlimits"): (probe.joaat("-0.250000,0.500000,0.000000"), 0, 1),
            probe.joaat("headinglimits"): (probe.joaat("-1.000000,1.000000,0.000000"), 1, 1),
        }
        verified = probe.verified_camera(values, vecs)
        self.assertEqual(
            verified,
            {
                "priority": "9",
                "blendduration": "37",
                "orbitdistance": "1.500000",
                "pitchlimits": "-0.250000,0.500000,0.000000",
                "headinglimits": "-1.000000,1.000000,0.000000",
            },
        )
        missing_key = dict(values)
        missing_key.pop(probe.joaat("priority"))
        self.assertIsNone(probe.verified_camera(missing_key, vecs))

        bad_digest = dict(values)
        bad_digest[probe.joaat("priority")] = (probe.joaat("10"), 9, 3)
        self.assertIsNone(probe.verified_camera(bad_digest, vecs))

    def test_tree_arrays_inspect_tree_and_inspect_single_vs_ambiguous(self):
        single_archive = _build_rpf9(_build_record_fields(priorities=(9,)))
        fields = _unpack_first_fields(single_archive)
        nodes, attrs, vectors = probe.tree_arrays(fields)
        counts, matches = probe.inspect_tree(nodes, attrs, vectors, probe.joaat("synthetic_orbit"), record_index=0)
        self.assertEqual(dict(counts), {"state_nodes": 1, "camera_parameter_sets": 1, "all_five_literals_verified": 1})
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["node_path"], [0])
        self.assertEqual(matches[0]["first_verified_example"]["priority"], "9")

        ambiguous_archive = _build_rpf9(_build_record_fields(priorities=(9, 17)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single_path = root / "single.rpf"
            single_path.write_bytes(single_archive)
            single_res = probe.inspect(single_path, "synthetic_orbit")
            self.assertEqual(single_res["first_verified_example"]["priority"], "9")
            self.assertNotIn("ambiguous_state_candidates", single_res)

            ambig_path = root / "ambiguous.rpf"
            ambig_path.write_bytes(ambiguous_archive)
            ambig_res = probe.inspect(ambig_path, "synthetic_orbit")
            self.assertIsNone(ambig_res["first_verified_example"])
            self.assertEqual(
                [c["node_path"] for c in ambig_res["ambiguous_state_candidates"]],
                [[0, 1], [0, 3]],
            )

            no_state_res = probe.inspect(single_path)
            self.assertIsNone(no_state_res["requested_state"])
            self.assertIsNone(no_state_res["first_verified_example"])


class ScriptStateProbeCliPositiveTests(unittest.TestCase):
    def test_main_argv_json_output_and_subprocess_exit_0(self):
        archive_bytes = _build_rpf9(_build_record_fields(priorities=(9,)))
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "valid.rpf"
            archive.write_bytes(archive_bytes)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = probe.main([str(archive), "--state", "synthetic_orbit"])
            self.assertEqual(rc, 0)
            self.assertEqual(stderr.getvalue(), "")
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["requested_state"], "synthetic_orbit")
            self.assertEqual(payload["first_verified_example"]["orbitdistance"], "1.500000")
            self.assertEqual(payload["counts"]["node_trees"], 1)

            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(archive), "--state", "synthetic_orbit"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(json.loads(proc.stdout), payload)


class ScriptStateProbeNegativeAndMutationTests(unittest.TestCase):
    def _assert_main_exit(self, argv: list[str], *, expected_code: int, fragment: str = "") -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                probe.main(argv)
        self.assertEqual(exc.exception.code, expected_code)
        self.assertEqual(stdout.getvalue(), "")
        err = stderr.getvalue()
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip(), f"expected stderr error for {argv!r}")
        if fragment:
            self.assertIn(fragment, err)
        return err

    def test_missing_dir_empty_file_and_bad_magic_controls(self):
        self._assert_main_exit([], expected_code=2)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.rpf"
            self._assert_main_exit([str(missing)], expected_code=1, fragment=missing.name)

            folder = root / "dir_archive"
            folder.mkdir()
            self._assert_main_exit([str(folder)], expected_code=1, fragment=folder.name)

            # 0-byte archive must fail cleanly before calling mmap.mmap.
            empty = root / "empty.rpf"
            empty.write_bytes(b"")
            with mock.patch.object(mmap, "mmap", side_effect=AssertionError("mmap must not run on 0-byte file")):
                with self.assertRaisesRegex(ValueError, "Empty archive"):
                    probe.inspect(empty)
            err = self._assert_main_exit([str(empty)], expected_code=1, fragment="Empty archive")
            self.assertNotIn("cannot mmap an empty file", err)

            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(empty)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("Empty archive", proc.stderr)

            bad_magic = root / "bad_magic.rpf"
            bad_magic.write_bytes(b"NOPE" + b"\x00" * 16)
            with self.assertRaisesRegex(ValueError, "Expected an RPF9 archive"):
                probe.inspect(bad_magic)
            self._assert_main_exit([str(bad_magic)], expected_code=1, fragment="Expected an RPF9 archive")

    def test_malformed_rpf9_psin_records_and_tree_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Corrupt zlib stream -> 0 section records decoded, no crash.
            corrupt_zlib = root / "corrupt_zlib.rpf"
            corrupt_zlib.write_bytes(_build_rpf9(_build_record_fields(corrupt_zlib=True)))
            self.assertEqual(probe.inspect(corrupt_zlib)["counts"], {})

            # Invalid CHKS payload length or mismatched CHKS total length -> skipped cleanly.
            for label, kw in (
                ("bad_chks_len", {"chks_payload_len": 8}),
                ("bad_chks_total", {"chks_total_delta": 4}),
            ):
                with self.subTest(case=label):
                    path = root / f"{label}.rpf"
                    path.write_bytes(_build_rpf9(_build_record_fields(**kw)))
                    self.assertEqual(probe.inspect(path)["counts"], {})

            # Bad LRSC magic -> record counted, tree_arrays returns None.
            bad_lrsc = root / "bad_lrsc.rpf"
            bad_lrsc.write_bytes(_build_rpf9(_build_record_fields(lrsc_magic=b"BAD!0")))
            self.assertEqual(probe.inspect(bad_lrsc)["counts"], {"section_records": 1})

            # Structural errors guarded inside inspect / main(argv) -> exit 1 without traceback.
            error_cases = (
                ("bad_root_id", {"root_id": 0}, "Invalid instance or schema directory"),
                ("invalid_marker", {"marker": 0xBBBBBBBBBBBBBBBB}, "Invalid instance block"),
                ("truncated_schema", {"truncate_schema_by": 8}, "Truncated schema members"),
                ("duplicate_state_id", {"duplicate_state_id": True}, "Duplicate state ID attribute"),
                ("non_label_state_id", {"id_kind": 3}, "Non-label state ID attribute"),
            )
            for label, opts, expected_msg in error_cases:
                with self.subTest(case=label):
                    path = root / f"{label}.rpf"
                    path.write_bytes(_build_rpf9(_build_record_fields(**opts)))
                    with self.assertRaisesRegex(ValueError, expected_msg):
                        probe.inspect(path, "synthetic_orbit")
                    self._assert_main_exit([str(path), "--state", "synthetic_orbit"], expected_code=1, fragment=expected_msg)

            # Bad instance/schema directory sizes in LRSC.
            bad_dir_bytes = _build_rpf9(_build_record_fields())
            fields = _unpack_first_fields(bad_dir_bytes)
            bad_lrsc_hdr = bytearray(fields["LRSC"])
            struct.pack_into("<H", bad_lrsc_hdr, 60, 999)
            fields["LRSC"] = memoryview(bad_lrsc_hdr)
            with self.assertRaisesRegex(ValueError, "Invalid instance or schema directory"):
                probe.tree_arrays(fields)

    def test_main_catches_struct_and_zlib_errors_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "dummy.rpf"
            archive.write_bytes(b"9FPR")
            for exc in (struct.error("unpack failed"), zlib.error("bad stream")):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(probe, "inspect", side_effect=exc):
                        self._assert_main_exit([str(archive)], expected_code=1, fragment=str(exc))


if __name__ == "__main__":
    unittest.main()
