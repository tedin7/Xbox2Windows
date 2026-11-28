#!/usr/bin/env python3
"""Standalone unit, bounds-validation, and mutation suite for tools/ffx_scratch_reference.py."""

from __future__ import annotations

from pathlib import Path
import re
import struct
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ffx_scratch_reference.py"
sys.path.insert(0, str(ROOT / "tools"))

import ffx_scratch_reference  # noqa: E402


def _build_initializer() -> bytes:
    # 128 bytes = 32 dwords; offset 0x00 and offset 0x40 are valid 64-byte windows.
    dwords = [
        0x01020304, 0xFFFDFCFB, 0x3F800000, 0x40000000,
        0x11223344, 0x80000001, 0x00000005, 0x00000007,
        0x7F800000, 0x42FE0000, 0x43800000, 0xC1200000,
        0x00000001, 0x00000002, 0x00000003, 0x00000004,
        *[(0x01010101 * (i + 1)) & 0xFFFFFFFF for i in range(16)],
    ]
    return struct.pack("<32I", *dwords)


def _synthetic_pass24() -> list[tuple[int, str]]:
    return [
        (0x00, "s_buffer_load_dwordx16 s[0:15], s[16:19], 0x0"),
        (0x08, "buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
        (0x10, "s_movk_i32 s20, 0xffff"),
        (0x14, "s_addk_i32 s20, 3"),
        (0x18, "s_mulk_i32 s20, -2"),
        (0x1C, "s_brev_b32 s21, s0"),
        (0x20, "v_xor_b32 v0, v0, s20"),
        (0x24, "v_add_f32 v1, v1, 1.5"),
        (0x28, "v_min_f32 v2, v2, 127.0"),
        (0x2C, "v_cvt_pk_u8_f32 v3, 127.0, 0, v3"),
        (0x30, "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
    ]


def _synthetic_producer() -> list[tuple[int, str]]:
    return [
        (0x0000, "s_buffer_load_dwordx16 s[0:15], s[16:19], 0x40"),
        (0x0010, "buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
        # Skipped under stage=26: [0x2414, 0x26dc) and [0x2a40, 0x3c48).
        (0x2414, "v_mov_b32 v0, 0"),
        (0x2800, "s_mov_b32 s20, 0"),
        (0x2810, "v_dot4_i32_i8 v10, v0, s0, 0 op_sel:[0,1,0]"),
        (0x2880, "v_dot4c_i32_i8 v10, v1, s1"),
        (0x2A40, "v_mov_b32 v10, 0"),
        (0x4370, "v_dot4_i32_i8 v11, v2, s2, -5 op_sel:[0,1,1]"),
        (0x4AC0, "v_xor_b32 v12, v3, s3"),
        (0x51EC, "v_mov_b32_e32 v13, v10"),
    ]


class FfxScratchReferencePositiveTests(unittest.TestCase):
    def test_simm16_and_bit_helpers_preserve_exact_semantics(self) -> None:
        self.assertEqual(ffx_scratch_reference.simm16("0"), 0)
        self.assertEqual(ffx_scratch_reference.simm16("0x7fff"), 0x7FFF)
        self.assertEqual(ffx_scratch_reference.simm16("0x8000"), 0xFFFF8000)
        self.assertEqual(ffx_scratch_reference.simm16("0xffff"), 0xFFFFFFFF)
        self.assertEqual(ffx_scratch_reference.simm16("-1"), 0xFFFFFFFF)
        self.assertEqual(ffx_scratch_reference.simm16("-32768"), 0xFFFF8000)
        self.assertEqual(ffx_scratch_reference.simm16("-0x8000"), 0xFFFF8000)
        self.assertEqual(ffx_scratch_reference.simm16(" 42 "), 42)

        self.assertEqual(ffx_scratch_reference.bits(1.5), 0x3FC00000)
        self.assertEqual(ffx_scratch_reference.real(0x3FC00000), 1.5)
        self.assertEqual(ffx_scratch_reference.signed(0x7FFFFFFF), 0x7FFFFFFF)
        self.assertEqual(ffx_scratch_reference.signed(0x80000000), -0x80000000)
        self.assertEqual(ffx_scratch_reference.signed(0xFFFFFFFF), -1)

    def test_network_stage_24_and_26_and_deterministic_make_reference(self) -> None:
        init = _build_initializer()
        pass24 = _synthetic_pass24()
        producer = _synthetic_producer()
        taps = [[ffx_scratch_reference.seed(x, y, c) for c in range(4)] for y in range(3) for x in range(3)]

        out24 = ffx_scratch_reference.network(pass24, taps, init, 24)
        self.assertEqual(len(out24), 4)
        self.assertEqual(out24, ffx_scratch_reference.network(pass24, taps, init, 24))

        out26 = ffx_scratch_reference.network(producer, taps, init, 26)
        self.assertEqual(len(out26), 4)
        self.assertEqual(out26, ffx_scratch_reference.network(producer, taps, init, 26))
        self.assertTrue(all(isinstance(x, int) and not isinstance(x, bool) for x in out26))

        programs = {"pass24": pass24, "producer": producer}
        cpp_a = ffx_scratch_reference.make_reference(programs, init)
        cpp_b = ffx_scratch_reference.make_reference(programs, init)
        self.assertEqual(cpp_a, cpp_b)

        lines = cpp_a.splitlines()
        self.assertEqual(len(lines), 2)
        m_rows = re.fullmatch(r"static constexpr uint32_t before_padding\[\] = \{([^}]+)\};", lines[0])
        m_gates = re.fullmatch(r"static constexpr int history_logits\[\] = \{([^}]+)\};", lines[1])
        self.assertIsNotNone(m_rows)
        self.assertIsNotNone(m_gates)
        row_tokens = m_rows.group(1).split(",")
        gate_tokens = m_gates.group(1).split(",")
        self.assertEqual(len(row_tokens), 3 * 24 * 4)  # 3 modes * 6x4 * 4 words = 288
        self.assertTrue(all(tok.startswith("0x") and tok.endswith("u") for tok in row_tokens))
        self.assertEqual(len(gate_tokens), 3 * 32)  # 3 modes * 32 logits = 96
        # Mode 0 has all-zero before_padding rows.
        self.assertTrue(all(tok == "0x0u" for tok in row_tokens[:96]))
        # Modes 1 and 2 must differ.
        self.assertNotEqual(row_tokens[96:192], row_tokens[192:])
        self.assertNotEqual(gate_tokens[32:64], gate_tokens[64:])


class FfxScratchReferenceNegativeAndMutationTests(unittest.TestCase):
    def test_simm16_rejects_non_str_bool_and_out_of_range(self) -> None:
        for bad in (True, False, 0, -1, 1.5, None, b"0", "", "  ", "bad", "0x10000", "65536", "-0x8001", "-32769"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ffx_scratch_reference.simm16(bad)

    def test_network_rejects_invalid_stage_and_bad_cbuf_offsets(self) -> None:
        init = _build_initializer()
        pass24 = _synthetic_pass24()
        taps = [[0, 0, 0, 0]] * 9

        for bad_stage in (0, 1, 25, 27, -24, True, False, 24.0, "24", None):
            with self.subTest(bad_stage=bad_stage):
                with self.assertRaises(ValueError):
                    ffx_scratch_reference.network(pass24, taps, init, bad_stage)

        for bad_offset in ("-4", "2", "6", str(len(init) - 60), str(len(init)), "0x1000"):
            with self.subTest(bad_offset=bad_offset):
                prog = [
                    (0x0, f"s_buffer_load_dwordx16 s[0:15], s[16:19], {bad_offset}"),
                    (0x4, "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
                ]
                with self.assertRaises(ValueError):
                    ffx_scratch_reference.network(prog, taps, init, 24)

        # Empty initializer triggers ValueError (not struct.error) even at offset 0.
        with self.assertRaises(ValueError):
            ffx_scratch_reference.network(pass24, taps, b"", 24)

    def test_make_reference_rejects_missing_keys_and_undifferentiated_fixture(self) -> None:
        init = _build_initializer()
        pass24 = _synthetic_pass24()
        producer = _synthetic_producer()

        for bad_programs in ({}, {"pass24": pass24}, {"producer": producer}, [], None):
            with self.subTest(bad_programs=bad_programs):
                with self.assertRaises(ValueError):
                    ffx_scratch_reference.make_reference(bad_programs, init)

        # Constant pass24 + constant producer fails undifferentiated fixture guard.
        const_pass24 = [
            (0x0, "v_mov_b32 v0, 1"),
            (0x4, "v_mov_b32 v1, 2"),
            (0x8, "v_mov_b32 v2, 3"),
            (0xC, "v_mov_b32 v3, 4"),
            (0x10, "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
        ]
        const_producer = [
            (0x2880, "v_mov_b32 v10, 1"),
            (0x4370, "v_mov_b32 v11, 2"),
            (0x4AC0, "v_mov_b32 v12, 3"),
            (0x51EC, "v_mov_b32 v13, 4"),
        ]
        with self.assertRaisesRegex(ValueError, "does not distinguish"):
            ffx_scratch_reference.make_reference({"pass24": const_pass24, "producer": const_producer}, init)

    def test_fast_temp_mutations(self) -> None:
        source = TOOL.read_text(encoding="utf-8")
        init = _build_initializer()
        pass24 = _synthetic_pass24()
        producer = _synthetic_producer()
        taps = [[1, 2, 3, 4]] * 9

        def _check_simm16_str_mut(mod) -> None:
            with self.assertRaises(TypeError):
                mod.simm16(True)

        def _check_simm16_range_mut(mod) -> None:
            self.assertEqual(mod.simm16("0x10000"), 0x10000)

        def _check_stage_mut(mod) -> None:
            with self.assertRaisesRegex(ValueError, "Incomplete tensor reference"):
                mod.network(pass24, taps, init, 25)

        def _check_cbuf_bounds_mut(mod) -> None:
            prog_unaligned = [
                (0x0, "s_buffer_load_dwordx16 s[0:15], s[16:19], 2"),
                (0x4, "buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
                (0x8, "v_mov_b32 v0, s0"),
                (0xC, "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
            ]
            res = mod.network(prog_unaligned, taps, init, 24)
            self.assertEqual(len(res), 4)
            prog_oob = [
                (0x0, f"s_buffer_load_dwordx16 s[0:15], s[16:19], {len(init)}"),
                (0x4, "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
            ]
            with self.assertRaises(struct.error):
                mod.network(prog_oob, taps, init, 24)

        def _check_programs_keys_mut(mod) -> None:
            with self.assertRaises(KeyError):
                mod.make_reference({}, init)

        def _check_undiff_mut(mod) -> None:
            const_pass24 = [
                (0x0, "v_mov_b32 v0, 1"),
                (0x4, "v_mov_b32 v1, 2"),
                (0x8, "v_mov_b32 v2, 3"),
                (0xC, "v_mov_b32 v3, 4"),
                (0x10, "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
            ]
            const_producer = [
                (0x2880, "v_mov_b32 v10, 1"),
                (0x4370, "v_mov_b32 v11, 2"),
                (0x4AC0, "v_mov_b32 v12, 3"),
                (0x51EC, "v_mov_b32 v13, 4"),
            ]
            out = mod.make_reference({"pass24": const_pass24, "producer": const_producer}, init)
            self.assertIn("before_padding", out)

        mutations = (
            (
                "drop_simm16_str_guard",
                '    if not isinstance(token, str):\n        raise ValueError(f"Invalid 16-bit signed immediate: {token!r}")\n',
                "    pass\n",
                _check_simm16_str_mut,
            ),
            (
                "drop_simm16_upper_bound",
                "    if 0 <= value <= 0xffff:\n",
                "    if 0 <= value <= 0x1ffff:\n",
                _check_simm16_range_mut,
            ),
            (
                "drop_stage_guard",
                '    if isinstance(stage, bool) or not isinstance(stage, int) or stage not in (24, 26):\n'
                '        raise ValueError(f"Expected network stage 24 or 26, got {stage!r}")\n',
                "    pass\n",
                _check_stage_mut,
            ),
            (
                "drop_cbuf_bounds_guard",
                '            if offset < 0 or offset % 4 != 0 or offset + 64 > len(initializer):\n'
                '                raise ValueError(\n'
                '                    f"Constant-buffer offset {offset} out of bounds for {len(initializer)}-byte initializer"\n'
                '                )\n',
                "            pass\n",
                _check_cbuf_bounds_mut,
            ),
            (
                "drop_programs_keys_guard",
                '    if not isinstance(programs, Mapping) or "pass24" not in programs or "producer" not in programs:\n'
                '        raise ValueError("Expected programs mapping with \'pass24\' and \'producer\'")\n',
                "    pass\n",
                _check_programs_keys_mut,
            ),
            (
                "drop_undifferentiated_guard",
                '    if rows[96:192] == rows[192:] or gates[32:64] == gates[64:]:\n'
                '        raise ValueError("Fixture does not distinguish the two upstream inputs")\n',
                "    pass\n",
                _check_undiff_mut,
            ),
        )

        for label, target, replacement, check_fn in mutations:
            with self.subTest(mutation=label):
                self.assertEqual(source.count(target), 1, target)
                mod = types.ModuleType(f"ffx_scratch_reference_mut_{label}")
                mod.__file__ = str(TOOL)
                sys.modules[mod.__name__] = mod
                try:
                    exec(compile(source.replace(target, replacement, 1), str(TOOL), "exec"), mod.__dict__)
                    check_fn(mod)
                finally:
                    sys.modules.pop(mod.__name__, None)


if __name__ == "__main__":
    unittest.main()
