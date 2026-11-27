#!/usr/bin/env python3
"""Standalone synthetic PE32+ unit, bounds, unwind-chain, and mutation suite for tools/cpu_ranges.py."""

from __future__ import annotations

from pathlib import Path
import struct
import sys
import types
import unittest
from unittest import mock

import pefile

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "cpu_ranges.py"
sys.path.insert(0, str(ROOT / "tools"))

import cpu_ranges  # noqa: E402


IMAGE_BASE = 0x180000000
RAW_OFFSET = 0x200
SECTION_ALIGN = 0x1000
TEXT_RVA = 0x1000
RDATA_RVA = 0x2000
SECTION_SIZE = 0x1000


def build_pe(
    *,
    text_bytes: bytes | None = None,
    text_vsize: int = SECTION_SIZE,
    text_rsize: int = SECTION_SIZE,
    pdata_records: list[tuple[int, int, int]] | None = None,
    rdata_blobs: dict[int, bytes] | None = None,
    pdata_rva_override: int | None = None,
    pdata_size_override: int | None = None,
    num_data_dirs: int = 16,
) -> pefile.PE:
    """Construct an in-memory AMD64 PE32+ object for cpu_ranges unit tests."""
    text = bytearray(SECTION_SIZE)
    if text_bytes:
        text[:len(text_bytes)] = text_bytes
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

    if rdata_blobs:
        for rva, blob in rdata_blobs.items():
            put_rdata(rva, blob)
            rdata_cursor = max(rdata_cursor, (rva - RDATA_RVA) + len(blob))

    pdata_rva = 0
    pdata_size = 0
    if pdata_records is not None:
        pdata_size = 12 * len(pdata_records)
        pdata_rva = alloc_rdata(max(pdata_size, 4), 4)
        for idx, (begin, end, info) in enumerate(pdata_records):
            put_rdata(pdata_rva + idx * 12, struct.pack("<III", begin, end, info))

    if pdata_rva_override is not None:
        pdata_rva = pdata_rva_override
    if pdata_size_override is not None:
        pdata_size = pdata_size_override

    buf = bytearray(RAW_OFFSET + SECTION_SIZE * 2)
    buf[0:2] = b"MZ"
    pe_off = 0x80
    struct.pack_into("<I", buf, 60, pe_off)
    buf[pe_off:pe_off + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_off + 4, 0x8664, 2, 0, 0, 0, 0xF0, 0x2022)
    opt = pe_off + 24
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        buf,
        opt,
        0x20B, 14, 0, SECTION_SIZE, SECTION_SIZE, 0, TEXT_RVA, TEXT_RVA,
        IMAGE_BASE, SECTION_ALIGN, 0x200, 6, 0, 0, 0, 6, 0, 0,
        0x3000, RAW_OFFSET, 0, 3, 0x160,
        0x100000, 0x1000, 0x100000, 0x1000, 0, num_data_dirs,
    )
    dd = opt + 112
    if num_data_dirs >= 4 and (pdata_rva or pdata_size):
        struct.pack_into("<II", buf, dd + 3 * 8, pdata_rva, pdata_size)

    sections_off = opt + 0xF0
    sections = [
        (b".text\0\0\0", text_vsize, TEXT_RVA, text_rsize, RAW_OFFSET, 0x60000020),
        (b".rdata\0\0", SECTION_SIZE, RDATA_RVA, SECTION_SIZE, RAW_OFFSET + SECTION_SIZE, 0x40000040),
    ]
    for idx, (name, vsize, va, rsize, roff, chars) in enumerate(sections):
        struct.pack_into(
            "<8sIIIIIIHHI",
            buf,
            sections_off + idx * 40,
            name, vsize, va, rsize, roff, 0, 0, 0, 0, chars,
        )

    buf[RAW_OFFSET:RAW_OFFSET + SECTION_SIZE] = text
    buf[RAW_OFFSET + SECTION_SIZE:RAW_OFFSET + SECTION_SIZE * 2] = rdata
    return pefile.PE(data=bytes(buf), fast_load=True)


def _chained_blob(version: int, code_count: int, parent: tuple[int, int, int]) -> bytes:
    header = bytes([version | (4 << 3), 0, code_count, 0])
    slots = ((code_count + 1) & ~1) * 2
    return header + (b"\x00" * slots) + struct.pack("<III", *parent)


class CpuRangesPositiveTests(unittest.TestCase):
    def test_image_bytes_exec_and_non_exec_sections(self) -> None:
        text_pattern = bytes((i * 11 + 3) & 0xFF for i in range(64))
        rdata_pattern = b"CONST-DATA-12345"
        with build_pe(text_bytes=text_pattern, rdata_blobs={RDATA_RVA + 0x40: rdata_pattern}) as pe:
            self.assertEqual(cpu_ranges.image_bytes(pe, TEXT_RVA, 32, executable=True), text_pattern[:32])
            self.assertEqual(cpu_ranges.image_bytes(pe, TEXT_RVA + 8, 16, executable=False), text_pattern[8:24])
            self.assertEqual(cpu_ranges.image_bytes(pe, RDATA_RVA + 0x40, len(rdata_pattern)), rdata_pattern)

    def test_single_range_and_chained_unwind_v1_v2_odd_even_sorting(self) -> None:
        primary_info = RDATA_RVA + 0x00
        chain_v1_odd = RDATA_RVA + 0x10
        chain_v2_even = RDATA_RVA + 0x30
        chain_v1_even0 = RDATA_RVA + 0x50
        chain_v2_odd3 = RDATA_RVA + 0x70

        primary = (0x1200, 0x1240, primary_info)
        frag_before = (0x1050, 0x1070, chain_v1_odd)
        frag_after = (0x1300, 0x1320, chain_v2_even)
        frag_grandchild = (0x1400, 0x1410, chain_v2_odd3)
        other_fn = (0x1100, 0x1120, primary_info)
        other_child = (0x1150, 0x1160, chain_v1_even0)

        blobs = {
            primary_info: b"\x01\x00\x00\x00",
            # v1, odd CountOfUnwindCodes=1 -> 4 bytes of codes -> trailer at +8
            chain_v1_odd: _chained_blob(1, 1, primary),
            # v2, even CountOfUnwindCodes=2 -> 4 bytes of codes -> trailer at +8
            chain_v2_even: _chained_blob(2, 2, primary),
            # v1, even CountOfUnwindCodes=0 -> 0 bytes of codes -> trailer at +4
            chain_v1_even0: _chained_blob(1, 0, other_fn),
            # v2, odd CountOfUnwindCodes=3 -> 8 bytes of codes -> trailer at +12 (2-hop chain)
            chain_v2_odd3: _chained_blob(2, 3, frag_before),
        }
        records = sorted([frag_before, other_fn, other_child, primary, frag_after, frag_grandchild])
        with build_pe(pdata_records=records, rdata_blobs=blobs) as pe:
            self.assertEqual(
                cpu_ranges.function_ranges(pe, 0x1200),
                [(0x1050, 0x1070), (0x1200, 0x1240), (0x1300, 0x1320), (0x1400, 0x1410)],
            )
            self.assertEqual(
                cpu_ranges.function_ranges(pe, 0x1100),
                [(0x1100, 0x1120), (0x1150, 0x1160)],
            )


class CpuRangesNegativeAndMutationTests(unittest.TestCase):
    def test_image_bytes_refusals_exec_bounds_and_types(self) -> None:
        with build_pe(text_vsize=0x200, text_rsize=0x100, rdata_blobs={RDATA_RVA: b"DATA"}) as pe:
            # Non-executable section refused when executable=True.
            with self.assertRaisesRegex(ValueError, "invalid file-backed range"):
                cpu_ranges.image_bytes(pe, RDATA_RVA, 4, executable=True)

            # Out-of-bounds beyond min(Misc_VirtualSize, SizeOfRawData) and unmapped RVA.
            with self.assertRaisesRegex(ValueError, "invalid file-backed range"):
                cpu_ranges.image_bytes(pe, TEXT_RVA, 0x101, executable=True)
            with self.assertRaisesRegex(ValueError, "invalid file-backed range"):
                cpu_ranges.image_bytes(pe, 0x5000, 4)

            # Bool, non-int, negative RVA, and non-positive size refused.
            for bad_rva in (True, False, -1, -0x1000, 1.5, "0x1000", None):
                with self.subTest(bad_rva=bad_rva):
                    with self.assertRaises(ValueError):
                        cpu_ranges.image_bytes(pe, bad_rva, 4)
            for bad_size in (True, False, 0, -1, -16, 1.5, "4", None):
                with self.subTest(bad_size=bad_size):
                    with self.assertRaises(ValueError):
                        cpu_ranges.image_bytes(pe, TEXT_RVA, bad_size)

            # rva < section.VirtualAddress refused even if get_section_by_rva returns a section.
            fake_sec = types.SimpleNamespace(
                VirtualAddress=0x2000,
                Characteristics=0x60000020,
                Misc_VirtualSize=0x1000,
                SizeOfRawData=0x1000,
                get_data=lambda _rva, length: b"\x90" * length,
            )
            with mock.patch.object(pe, "get_section_by_rva", return_value=fake_sec):
                with self.assertRaisesRegex(ValueError, "invalid file-backed range"):
                    cpu_ranges.image_bytes(pe, 0x1000, 4)

    def test_function_ranges_refusals(self) -> None:
        # 1) Invalid entry types / negative entry / missing runtime-function directory.
        with build_pe(pdata_records=None) as pe_no_pdata:
            for bad_entry in (True, False, -1, -0x100, 1.5, "0x1000", None):
                with self.subTest(bad_entry=bad_entry):
                    with self.assertRaises(ValueError):
                        cpu_ranges.function_ranges(pe_no_pdata, bad_entry)
            with self.assertRaisesRegex(ValueError, "no runtime-function directory"):
                cpu_ranges.function_ranges(pe_no_pdata, TEXT_RVA)

        with build_pe(pdata_records=None, pdata_rva_override=RDATA_RVA, pdata_size_override=0) as pe_zero_size:
            with self.assertRaisesRegex(ValueError, "no runtime-function directory"):
                cpu_ranges.function_ranges(pe_zero_size, TEXT_RVA)

        with build_pe(pdata_records=None, num_data_dirs=3) as pe_short_dd:
            with self.assertRaisesRegex(ValueError, "no runtime-function directory"):
                cpu_ranges.function_ranges(pe_short_dd, TEXT_RVA)

        # 2) Truncated (.pdata Size % 12 != 0), out-of-order, overlapping, inverted, unaligned info.
        with build_pe(pdata_records=[(0x1000, 0x1010, RDATA_RVA)], rdata_blobs={RDATA_RVA: b"\x01\x00\x00\x00"},
                      pdata_size_override=10) as pe_trunc:
            with self.assertRaisesRegex(ValueError, "incomplete runtime-function record"):
                cpu_ranges.function_ranges(pe_trunc, 0x1000)

        for label, bad_records in (
            ("overlapping", [(0x1000, 0x1020, RDATA_RVA), (0x1010, 0x1030, RDATA_RVA)]),
            ("out_of_order", [(0x1020, 0x1030, RDATA_RVA), (0x1000, 0x1010, RDATA_RVA)]),
            ("empty_interval", [(0x1000, 0x1000, RDATA_RVA)]),
            ("unaligned_info", [(0x1000, 0x1010, RDATA_RVA + 2)]),
        ):
            with self.subTest(case=label):
                with build_pe(pdata_records=bad_records, rdata_blobs={RDATA_RVA: b"\x01\x00\x00\x00"}) as pe:
                    with self.assertRaisesRegex(ValueError, "invalid runtime-function interval"):
                        cpu_ranges.function_ranges(pe, 0x1000)

        # 3) Unsupported unwind header / truncated trailer / dangling parent / cyclic unwind chain / missing entry.
        with build_pe(pdata_records=[(0x1000, 0x1010, RDATA_RVA)],
                      rdata_blobs={RDATA_RVA: bytes([3 | (4 << 3), 0, 0, 0]) + bytes(12)}) as pe_bad_ver:
            with self.assertRaisesRegex(ValueError, "unsupported chained unwind header"):
                cpu_ranges.function_ranges(pe_bad_ver, 0x1000)

        with build_pe(pdata_records=[(0x1000, 0x1010, RDATA_RVA)],
                      rdata_blobs={RDATA_RVA: bytes([1 | (5 << 3), 0, 0, 0]) + bytes(12)}) as pe_bad_flags:
            with self.assertRaisesRegex(ValueError, "unsupported chained unwind header"):
                cpu_ranges.function_ranges(pe_bad_flags, 0x1000)

        # Truncated chained trailer right at end of .rdata section.
        trunc_info = RDATA_RVA + SECTION_SIZE - 8
        with build_pe(pdata_records=[(0x1000, 0x1010, trunc_info)],
                      rdata_blobs={trunc_info: bytes([1 | (4 << 3), 0, 0, 0])}) as pe_trunc_trailer:
            with self.assertRaisesRegex(ValueError, "invalid file-backed range"):
                cpu_ranges.function_ranges(pe_trunc_trailer, 0x1000)

        # Dangling parent not in .pdata.
        with build_pe(pdata_records=[(0x1000, 0x1010, RDATA_RVA)],
                      rdata_blobs={RDATA_RVA: _chained_blob(1, 0, (0x1500, 0x1510, RDATA_RVA))}) as pe_dangling:
            with self.assertRaisesRegex(ValueError, "invalid unwind chain"):
                cpu_ranges.function_ranges(pe_dangling, 0x1000)

        # Cyclic unwind chain (A -> B -> A).
        info_a = RDATA_RVA + 0x00
        info_b = RDATA_RVA + 0x20
        rec_a = (0x1000, 0x1010, info_a)
        rec_b = (0x1020, 0x1030, info_b)
        with build_pe(pdata_records=[rec_a, rec_b],
                      rdata_blobs={info_a: _chained_blob(1, 0, rec_b),
                                   info_b: _chained_blob(1, 0, rec_a)}) as pe_cycle:
            with self.assertRaisesRegex(ValueError, "invalid unwind chain"):
                cpu_ranges.function_ranges(pe_cycle, 0x1000)

        # Missing entry in valid .pdata.
        with build_pe(pdata_records=[(0x1000, 0x1010, RDATA_RVA)],
                      rdata_blobs={RDATA_RVA: b"\x01\x00\x00\x00"}) as pe_valid:
            with self.assertRaisesRegex(ValueError, "no unique function interval"):
                cpu_ranges.function_ranges(pe_valid, 0x1100)

    def test_fast_temp_mutations(self) -> None:
        source = TOOL.read_text(encoding="utf-8")

        with build_pe(
            pdata_records=[(0x1050, 0x1070, RDATA_RVA + 0x10), (0x1100, 0x1120, RDATA_RVA), (0x1200, 0x1240, RDATA_RVA)],
            rdata_blobs={
                RDATA_RVA: b"\x01\x00\x00\x00",
                RDATA_RVA + 0x10: _chained_blob(1, 1, (0x1200, 0x1240, RDATA_RVA)),
            },
        ) as pe_valid, build_pe(
            pdata_records=[(0x1000, 0x1020, RDATA_RVA), (0x1010, 0x1030, RDATA_RVA)],
            rdata_blobs={RDATA_RVA: b"\x01\x00\x00\x00"},
        ) as pe_overlap:
            def _check_bool_mut(mod) -> None:
                self.assertEqual(len(mod.image_bytes(pe_valid, TEXT_RVA, True)), 1)

            def _check_rva_lt_va_mut(mod) -> None:
                fake_sec = types.SimpleNamespace(
                    VirtualAddress=0x2000,
                    Characteristics=0x60000020,
                    Misc_VirtualSize=0x2000,
                    SizeOfRawData=0x2000,
                    get_data=lambda _rva, length: b"\x90" * length,
                )
                with mock.patch.object(pe_valid, "get_section_by_rva", return_value=fake_sec):
                    self.assertEqual(len(mod.image_bytes(pe_valid, 0x1000, 4)), 4)

            def _check_exec_mut(mod) -> None:
                self.assertEqual(len(mod.image_bytes(pe_valid, RDATA_RVA, 4, executable=True)), 4)

            def _check_odd_align_mut(mod) -> None:
                with self.assertRaisesRegex(ValueError, "invalid unwind chain"):
                    mod.function_ranges(pe_valid, 0x1200)

            def _check_overlap_mut(mod) -> None:
                self.assertEqual(mod.function_ranges(pe_overlap, 0x1000), [(0x1000, 0x1020)])

            mutations = (
                (
                    "drop_bool_guard",
                    "    if isinstance(value, bool) or not isinstance(value, int):\n",
                    "    if not isinstance(value, int):\n",
                    _check_bool_mut,
                ),
                (
                    "drop_rva_lt_section_va_guard",
                    "    if (not section or rva < section.VirtualAddress\n",
                    "    if (not section\n",
                    _check_rva_lt_va_mut,
                ),
                (
                    "drop_executable_guard",
                    "            or (executable and not section.Characteristics & 0x20000000)\n",
                    "            or False\n",
                    _check_exec_mut,
                ),
                (
                    "drop_odd_unwind_align",
                    "            trailer = info + 4 + ((header[2] + 1) & ~1) * 2\n",
                    "            trailer = info + 4 + header[2] * 2\n",
                    _check_odd_align_mut,
                ),
                (
                    "drop_overlap_guard",
                    "        if begin < previous or end <= begin or info % 4:\n",
                    "        if end <= begin or info % 4:\n",
                    _check_overlap_mut,
                ),
            )

            for label, target, replacement, check_fn in mutations:
                with self.subTest(mutation=label):
                    self.assertEqual(source.count(target), 1, target)
                    mod = types.ModuleType(f"cpu_ranges_mut_{label}")
                    mod.__file__ = str(TOOL)
                    sys.modules[mod.__name__] = mod
                    try:
                        exec(compile(source.replace(target, replacement, 1), str(TOOL), "exec"), mod.__dict__)
                        check_fn(mod)
                    finally:
                        sys.modules.pop(mod.__name__, None)


if __name__ == "__main__":
    unittest.main()
