"""Check integer inequality as the existing wave32 lane guard."""
from unittest.mock import patch

import compute_store_check as base

aot = base.aot
COMPARE = "v_cmp_gt_u32_e32 vcc_lo, s3, v0"


def lower(compare):
    return aot.lower(base.variant(base.GUARD_END, (COMPARE, compare)), 32, **base.GUARD_BINDING).splitlines()


def controls():
    for compare, expected in (
        ("v_cmp_ne_i32_e32 vcc_lo, s3, v0", "params[0].x != tid.x"),
        ("v_cmp_ne_i32_e32 vcc_lo, -1, v0", "0xffffffffu != tid.x"),
        ("v_cmp_ne_i32_e32 vcc_lo, 0x80000000, v0", "0x80000000u != tid.x"),
        ("v_cmp_ne_i32_e32 vcc_lo, v0, v0", "tid.x != tid.x"),
        ("v_cmp_ne_i32_e64 vcc_lo, v0, s3", "tid.x != params[0].x"),
        ("v_cmp_ne_i32_e64 vcc_lo, v0, 0", "tid.x != 0x00000000u"),
    ):
        assert lower(compare)[-4:] == [f"    if ({expected}) {{",
                                      "        outputData.Store(tid.x * 4u + 0u, tid.x);", "    }", "}"]

    for compare, diagnostic in (
        ("v_cmp_ne_i32_e32 vcc_lo, v0, s3", "Unsupported compute instruction"),
        ("v_cmp_ne_i32_e32 vcc_hi, s3, v0", "Unsupported compute instruction"),
        # v_cmp_ne_i32_e64 s20, ... sets s20 to a per-lane cond without writing vcc_lo, so s_and_saveexec_b32 refuses it.
        ("v_cmp_ne_i32_e64 s20, s3, v0", "Unverified or repeated lane guard"),
        ("v_cmp_ne_i32_e64 vcc_lo, -v0, s3", "Unsupported compute instruction"),
        ("v_cmpx_ne_i32_e32 vcc_lo, s3, v0", "Unsupported compute instruction"),
        ("v_cmp_ne_i32_e32 vcc_lo, s20, v0", "Unverified lane guard compare"),
        ("v_cmp_ne_i32_e32 vcc_lo, s4, v0", "Unverified lane guard compare"),
        ("v_cmp_ne_i32_e32 vcc_lo, s3, v256", "Unverified lane guard compare"),
        ("v_cmp_ne_i32_e32 vcc_lo, 0x100000000, v0", "Unverified lane guard compare"),
        # A word computed in the program is not a verified guard operand, including in _e64 source 1.
        ("s_add_u32 s20, s3, 1\nv_cmp_ne_i32_e64 vcc_lo, v0, s20", "Unverified lane guard compare"),
    ):
        base.rejects(diagnostic, lower, compare)

    compare = "v_cmp_ne_i32_e32 vcc_lo, s3, v0"
    good = lower(compare)[-4]
    with patch.dict(aot.GUARD_COMPARES, ne="=="):
        wrong = lower(compare)[-4]
    assert wrong == good.replace("!=", "==") and wrong != good


if __name__ == "__main__":
    controls()
