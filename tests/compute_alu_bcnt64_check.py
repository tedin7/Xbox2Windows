"""Check scalar pair popcount lowering and the unsupported register forms."""
from unittest.mock import patch

import compute_store_check as base

aot = base.aot
COUNT = "s_bcnt1_i32_b64 s4, s[4:5]"
SETUP = "s_mov_b32 s4, 0x80000001\ns_mov_b32 s5, 0xa0000007"
LINES = base.variant(base.BCNT_EXEC, ("s_bcnt1_i32_b32 s4, exec_lo", SETUP + "\n" + COUNT))


def lower(lines=LINES):
    return aot.lower(lines, 64, **base.BCNT_EXEC_BINDING)


def controls():
    # D aliases the low source word. Two low bits plus five high bits gives seven.
    assert "+ 0x00000007u)" in lower()
    def setup(low, high):
        return base.variant(LINES, ("s_mov_b32 s4, 0x80000001", "s_mov_b32 s4, " + low),
                            ("s_mov_b32 s5, 0xa0000007", "s_mov_b32 s5, " + high))

    assert "+ 0x00000000u)" in lower(setup("0", "0"))
    assert "+ 0x00000040u)" in lower(setup("-1", "-1"))

    compare = "v_cmp_gt_u32_e32 vcc_lo, s3, v0"
    dynamic = base.variant(
        base.GUARD_END,
        (compare, "s_mov_b32 s20, s3\ns_mov_b32 s21, 0x80000001\n"
                  "s_bcnt1_i32_b64 s21, s[20:21]\nv_mov_b32_e32 v1, s21\n" + compare),
        ("buffer_store_dword v0,", "buffer_store_dword v1,"),
    )
    assert "Store(tid.x * 4u + 0u, (countbits(params[0].x) + countbits(0x80000001u)))" in (
        aot.lower(dynamic, 32, **base.GUARD_BINDING)
    )

    for operands, diagnostic in (
        ("s4, exec", "Unsupported compute instruction"),
        ("s4, vcc", "Unsupported compute instruction"),
        ("s4, 5", "Unsupported compute instruction"),
        ("s4, s[5:6]", "Invalid scalar bit count register"),
        ("s4, s[4:6]", "Invalid scalar bit count register"),
        ("s4, s[106:107]", "Invalid scalar bit count register"),
        ("s106, s[4:5]", "Invalid scalar bit count register"),
        ("s12, s[4:5]", "Invalid scalar bit count register"),
        ("s4, s[12:13]", "Unverified scalar bit count source"),
        ("s4, s[20:21]", "Unverified scalar bit count source"),
    ):
        base.rejects(diagnostic, lower, base.variant(LINES, (COUNT, "s_bcnt1_i32_b64 " + operands)))
    branch = base.variant(LINES, (COUNT, COUNT + "\ns_cbranch_scc1 .LBB0_2"))
    base.rejects("Unsupported compute instruction: s_cbranch_scc1", lower, branch)

    # Mutate the translator's known low/high word lookup to drop the upper word.
    original = aot.alu_word
    with patch.object(aot, "alu_word", side_effect=lambda token, *state: 0 if token == "s5" else original(token, *state)):
        mutated = lower()
    assert "+ 0x00000002u)" in mutated and "+ 0x00000007u)" not in mutated, mutated


if __name__ == "__main__":
    controls()
