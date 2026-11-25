"""Synthetic PE32+ and mocked-disassembly controls for tools/call_evidence.py."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pefile

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "call_evidence.py"
sys.path.insert(0, str(ROOT / "tools"))
import call_evidence  # noqa: E402


IMAGE_BASE = 0x180000000
SECTION_RVA = 0x1000
RAW_OFFSET = 0x200
RAW_SIZE = 0x1000


def make_dxbc_container(*, with_rts0=True):
    """Build a minimal synthetic DXBC container with optional RTS0 header and framed DXIL."""
    program = struct.pack("<II4sIII", 0, 8, b"DXIL", 0, 16, 4) + b"BC\xc0\xde" + bytes(4)
    parts = [b"DXIL" + struct.pack("<I", len(program)) + program]
    if with_rts0:
        rts0 = struct.pack("<6I", 2, 3, 24, 1, 0, 0x10)
        parts.append(b"RTS0" + struct.pack("<I", len(rts0)) + rts0)
    table_end = 32 + 4 * len(parts)
    offsets = []
    cursor = table_end
    for part in parts:
        offsets.append(cursor)
        cursor += len(part)
    body = b"".join(parts)
    total = table_end + len(body)
    return (b"DXBC" + bytes(16) + struct.pack("<III", 1, total, len(parts))
            + struct.pack(f"<{len(parts)}I", *offsets) + body)


def build_pe32_plus(imports=(), *, containers=(), virtual_size=None, raw_size=RAW_SIZE,
                    data_dir_count=16, overlay=b"", image_base=IMAGE_BASE):
    """Build a synthetic AMD64 PE32+ image with optional imports and mapped DXBC containers.

    imports: sequence of (dll_bytes, symbol_bytes_or_ordinal_int)
    containers: sequence of container bytes to map into .rdata alongside size+pointer pairs
    Returns (pe_bytes, iat_rvas, container_layout).
    """
    payload = bytearray(raw_size)
    cursor = 0

    def alloc(size, align):
        nonlocal cursor
        cursor = (cursor + align - 1) & ~(align - 1)
        rva = SECTION_RVA + cursor
        if cursor + size > len(payload):
            raise AssertionError("synthetic section overflow")
        cursor += size
        return rva

    def put(rva, blob):
        offset = rva - SECTION_RVA
        payload[offset:offset + len(blob)] = blob

    grouped = {}
    for dll_bytes, sym in imports:
        grouped.setdefault(dll_bytes, []).append(sym)

    dll_rvas = {}
    for dll_bytes in grouped:
        blob = dll_bytes + b"\0"
        rva = alloc(len(blob), 2)
        put(rva, blob)
        dll_rvas[dll_bytes] = rva

    hint_rvas = {}
    for dll_bytes, syms in grouped.items():
        for sym in syms:
            if isinstance(sym, int):
                continue
            blob = struct.pack("<H", 0) + sym + b"\0"
            rva = alloc(len(blob), 2)
            put(rva, blob)
            hint_rvas[(dll_bytes, sym)] = rva

    iat_rvas = {}
    descriptors = []
    for dll_bytes, syms in grouped.items():
        entries = []
        for sym in syms:
            if isinstance(sym, int):
                entries.append((1 << 63) | sym)
            else:
                entries.append(hint_rvas[(dll_bytes, sym)])
        table_blob = b"".join(struct.pack("<Q", value) for value in [*entries, 0])
        lookup_rva = alloc(len(table_blob), 8)
        put(lookup_rva, table_blob)
        iat_rva = alloc(len(table_blob), 8)
        put(iat_rva, table_blob)
        for index, sym in enumerate(syms):
            key = (dll_bytes, sym)
            iat_rvas[key] = iat_rva + index * 8
        descriptors.append((lookup_rva, dll_rvas[dll_bytes], iat_rva))

    import_dir_rva = import_dir_size = 0
    if descriptors:
        import_dir_size = 20 * (len(descriptors) + 1)
        import_dir_rva = alloc(import_dir_size, 4)
        for index, (lookup_rva, dll_rva, iat_rva) in enumerate(descriptors):
            put(import_dir_rva + index * 20,
                struct.pack("<5I", lookup_rva, 0, 0, dll_rva, iat_rva))

    container_layout = []
    for blob in containers:
        c_rva = alloc(len(blob), 16)
        put(c_rva, blob)
        pair_rva = alloc(16, 8)
        put(pair_rva, struct.pack("<IIQ", len(blob), 0, image_base + c_rva))
        container_layout.append((c_rva, pair_rva, pair_rva + 8, len(blob)))

    vsize = virtual_size if virtual_size is not None else max(1, cursor)
    size_of_image = SECTION_RVA + ((vsize + 0xFFF) // 0x1000) * 0x1000
    pe_offset = 0x80
    buf = bytearray(RAW_OFFSET + raw_size)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 60, pe_offset)
    buf[pe_offset:pe_offset + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_offset + 4, 0x8664, 1, 0, 0, 0, 240, 0x2022)
    opt = pe_offset + 24
    struct.pack_into("<HBBIIIIIQIIHHHHHHIIIIHHQQQQII", buf, opt,
                     0x20B, 0, 0, 0, raw_size, 0, 0, SECTION_RVA, image_base,
                     0x1000, 0x200, 6, 0, 0, 0, 6, 0, 0, size_of_image, RAW_OFFSET,
                     0, 3, 0, 0, 0, 0, 0, 0, data_dir_count)
    sec = opt + 240
    struct.pack_into("<8sIIIIIIHHI", buf, sec,
                     b".rdata\0\0", vsize, SECTION_RVA, raw_size, RAW_OFFSET,
                     0, 0, 0, 0, 0x40000040)
    if import_dir_rva and data_dir_count > 1:
        struct.pack_into("<II", buf, opt + 112 + 8, import_dir_rva, import_dir_size)
    buf[RAW_OFFSET:RAW_OFFSET + raw_size] = payload
    return bytes(buf) + overlay, iat_rvas, container_layout


def format_disasm(base, lines):
    text = "\n".join([
        "Dump of file fixture.dll",
        "",
        "File Type: DLL",
        "",
        *[f"  {base + rva:016X}: {body}" for rva, body in lines],
        "",
        "  Summary",
        "        1000 .rdata",
    ])
    return text.encode("ascii")


class CallEvidenceChecks(unittest.TestCase):
    def test_instruction_refs_maps_calls_jmps_and_memory_rvas(self):
        disasm = format_disasm(IMAGE_BASE, [
            (0x2000, "call        qword ptr [0000000180001080h]"),
            (0x2010, "jmp         qword ptr [0000000180001088h]"),
            (0x2020, "call        0000000180002010"),
            (0x2030, "jmp         0000000180002010"),
            (0x2040, "lea         rcx, [0000000180001200h]"),
            (0x2050, "ret"),
        ])
        with patch("call_evidence.subprocess.check_output", return_value=disasm) as mock_run:
            memory, branches = call_evidence.instruction_refs(Path("fixture.dll"), IMAGE_BASE)
        mock_run.assert_called_once_with(["dumpbin", "/nologo", "/disasm:nobytes", "fixture.dll"])
        self.assertEqual(memory[0x1080], [(0x2000, "call")])
        self.assertEqual(memory[0x1088], [(0x2010, "jmp")])
        self.assertEqual(memory[0x1200], [(0x2040, "lea")])
        self.assertEqual(branches[0x2010], [(0x2020, "call"), (0x2030, "jmp")])
        self.assertNotIn(0x2050, branches)

    def test_inspect_positive_controls_iat_sites_and_containers(self):
        container = make_dxbc_container(with_rts0=True)
        overlay_container = make_dxbc_container(with_rts0=False)
        pe_bytes, iat_rvas, layout = build_pe32_plus(
            imports=[
                (b"XMem.DLL", b"XMemAlloc"),
                (b"d3d12_xs.dll", b"D3D12SerializeVersionedRootSignature"),
                (b"kernel32.dll", b"ExitProcess"),
                (b"xmem.dll", 7),
            ],
            containers=[container],
            virtual_size=0x600,
            overlay=overlay_container,
        )
        xmem_slot = iat_rvas[(b"XMem.DLL", b"XMemAlloc")]
        d3d12_slot = iat_rvas[(b"d3d12_xs.dll", b"D3D12SerializeVersionedRootSignature")]
        c_rva, size_slot, ptr_slot, c_len = layout[0]
        disasm = format_disasm(IMAGE_BASE, [
            (0x2000, f"call        qword ptr [{IMAGE_BASE + xmem_slot:016X}h]"),
            (0x2010, f"jmp         qword ptr [{IMAGE_BASE + d3d12_slot:016X}h]"),
            (0x2020, f"call        {IMAGE_BASE + 0x2010:016X}"),
            (0x2030, f"jmp         {IMAGE_BASE + 0x2010:016X}"),
            (0x2040, f"lea         rcx, [{IMAGE_BASE + c_rva:016X}h]"),
            (0x2050, f"mov         eax, dword ptr [{IMAGE_BASE + size_slot:016X}h]"),
            (0x2060, f"mov         rdx, qword ptr [{IMAGE_BASE + ptr_slot:016X}h]"),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            dll_path = Path(tmp) / "fixture.dll"
            dll_path.write_bytes(pe_bytes)
            with patch("call_evidence.subprocess.check_output", return_value=disasm):
                report = call_evidence.inspect(dll_path)

        self.assertEqual(report["sha256"], hashlib.sha256(pe_bytes).hexdigest())
        self.assertEqual(set(report["calls"].keys()), {"XMemAlloc", "D3D12SerializeVersionedRootSignature"})
        self.assertEqual(report["calls"]["XMemAlloc"], {
            "iat_rva": hex(xmem_slot),
            "sites": [{"rva": "0x2000", "operation": "call", "upstream": []}],
        })
        self.assertEqual(report["calls"]["D3D12SerializeVersionedRootSignature"], {
            "iat_rva": hex(d3d12_slot),
            "sites": [{"rva": "0x2010", "operation": "jmp", "upstream": ["0x2020", "0x2030"]}],
        })
        self.assertEqual(len(report["containers"]), 1)
        row = report["containers"][0]
        self.assertEqual(row["rva"], hex(c_rva))
        self.assertEqual(row["bytes"], c_len)
        self.assertEqual(row["parts"], ["DXIL", "RTS0"])
        self.assertEqual(row["root_header"], {
            "version": 2, "parameter_count": 3, "static_sampler_count": 1, "flags": "0x10",
        })
        self.assertTrue(row["framed_dxil"])
        self.assertEqual(row["direct_code_refs"], ["0x2040"])
        self.assertEqual(row["size_pointer_pairs"], [{
            "pointer_slot": hex(ptr_slot),
            "size_slot": hex(size_slot),
            "pointer_code_refs": ["0x2060"],
            "size_code_refs": ["0x2050"],
        }])

    def test_missing_import_directory_and_non_ascii_names_do_not_crash(self):
        for data_dir_count in (16, 0):
            with self.subTest(data_dir_count=data_dir_count):
                pe_bytes, _, _ = build_pe32_plus(imports=(), data_dir_count=data_dir_count)
                with tempfile.TemporaryDirectory() as tmp:
                    dll_path = Path(tmp) / "no_imports.dll"
                    dll_path.write_bytes(pe_bytes)
                    with patch("call_evidence.subprocess.check_output", return_value=b""):
                        report = call_evidence.inspect(dll_path)
                self.assertEqual(report["calls"], {})
                self.assertEqual(report["containers"], [])

        non_ascii_pe, _, _ = build_pe32_plus(
            imports=[(b"\xffbad.dll", b"Sym\xfeName")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            dll_path = Path(tmp) / "non_ascii.dll"
            dll_path.write_bytes(non_ascii_pe)
            with patch("call_evidence.subprocess.check_output", return_value=b""):
                report = call_evidence.inspect(dll_path)
        self.assertEqual(report["calls"], {})

    def test_mapped_rva_skips_unmapped_offsets_and_slack(self):
        container = make_dxbc_container(with_rts0=False)
        pe_bytes, _, _ = build_pe32_plus(containers=[container], virtual_size=0x100, raw_size=0x200, overlay=b"tail")
        with pefile.PE(data=pe_bytes, fast_load=True) as pe:
            self.assertEqual(call_evidence._mapped_rva(pe, RAW_OFFSET + 0x10, 16), SECTION_RVA + 0x10)
            self.assertIsNone(call_evidence._mapped_rva(pe, 0x40, 16))  # PE header before section
            self.assertIsNone(call_evidence._mapped_rva(pe, RAW_OFFSET + 0x100, 16))  # Raw slack beyond VirtualSize
            self.assertIsNone(call_evidence._mapped_rva(pe, RAW_OFFSET + 0xF8, 16))  # Crosses VirtualSize
            self.assertIsNone(call_evidence._mapped_rva(pe, RAW_OFFSET + 0x200, 4))  # Overlay after section

    def test_main_writes_expected_json_for_three_images(self):
        pe1, iat1, _ = build_pe32_plus(imports=[(b"xmem.dll", b"XMemAlloc")])
        pe2, iat2, _ = build_pe32_plus(imports=[(b"d3d12_xs.dll", b"D3D12SerializeVersionedRootSignature")])
        pe3, _, _ = build_pe32_plus(imports=(), data_dir_count=0)
        images = {
            call_evidence.NAMES[0]: pe1,
            call_evidence.NAMES[1]: pe2,
            call_evidence.NAMES[2]: pe3,
        }
        disasm_by_name = {
            call_evidence.NAMES[0]: format_disasm(IMAGE_BASE, [
                (0x2100, f"call        qword ptr [{IMAGE_BASE + iat1[(b'xmem.dll', b'XMemAlloc')]:016X}h]"),
            ]),
            call_evidence.NAMES[1]: format_disasm(IMAGE_BASE, [
                (0x2200, f"jmp         qword ptr [{IMAGE_BASE + iat2[(b'd3d12_xs.dll', b'D3D12SerializeVersionedRootSignature')]:016X}h]"),
            ]),
            call_evidence.NAMES[2]: b"",
        }

        def fake_dumpbin(cmd):
            return disasm_by_name[Path(cmd[-1]).name]

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build_dir = base / "build"
            build_dir.mkdir()
            for name, blob in images.items():
                (build_dir / name).write_bytes(blob)
            fake_tool = base / "repo" / "tools" / "call_evidence.py"
            fake_tool.parent.mkdir(parents=True)
            stdout = io.StringIO()
            with patch.object(call_evidence, "__file__", str(fake_tool)), \
                 patch("call_evidence.subprocess.check_output", side_effect=fake_dumpbin), \
                 contextlib.redirect_stdout(stdout):
                rc = call_evidence.main([str(build_dir)])
            self.assertEqual(rc, 0)
            expected_output = base / "repo" / "out" / "calls" / "six-apis.json"
            self.assertEqual(Path(stdout.getvalue().strip()), expected_output.resolve())
            self.assertTrue(expected_output.is_file())
            parsed = json.loads(expected_output.read_text(encoding="utf-8"))
            self.assertEqual(list(parsed.keys()), list(call_evidence.NAMES))
            self.assertIn("XMemAlloc", parsed[call_evidence.NAMES[0]]["calls"])
            self.assertIn("D3D12SerializeVersionedRootSignature", parsed[call_evidence.NAMES[1]]["calls"])
            self.assertEqual(parsed[call_evidence.NAMES[2]]["calls"], {})

    def test_main_rejects_missing_dir_missing_images_and_overlapping_output_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            missing_dir = base / "does-not-exist"
            proc = subprocess.run([sys.executable, "-B", str(TOOL), str(missing_dir)],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip())

            partial_dir = base / "partial"
            partial_dir.mkdir()
            proc_partial = subprocess.run([sys.executable, "-B", str(TOOL), str(partial_dir)],
                                          capture_output=True, text=True)
            self.assertEqual(proc_partial.returncode, 2)
            self.assertNotIn("Traceback", proc_partial.stderr)
            self.assertIn("Missing expected image", proc_partial.stderr)

            a_file = base / "not_a_dir.bin"
            a_file.write_bytes(b"data")
            proc_file = subprocess.run([sys.executable, "-B", str(TOOL), str(a_file)],
                                       capture_output=True, text=True)
            self.assertEqual(proc_file.returncode, 2)
            self.assertNotIn("Traceback", proc_file.stderr)

        proc_overlap = subprocess.run([sys.executable, "-B", str(TOOL), str(ROOT)],
                                      capture_output=True, text=True)
        self.assertEqual(proc_overlap.returncode, 2)
        self.assertNotIn("Traceback", proc_overlap.stderr)
        self.assertIn("separate from the report directory", proc_overlap.stderr)


if __name__ == "__main__":
    unittest.main()
